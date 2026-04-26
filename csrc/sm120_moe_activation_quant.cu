#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <torch/python.h>

#include "jit_kernels/impls/sm120_moe_activation_quant.hpp"
#include "utils/exception.hpp"

namespace deep_gemm::sm120_moe {
namespace {

constexpr float kFp8E4M3Max = 448.0f;
constexpr float kFp8E4M3Min = -448.0f;
constexpr float kScaleEps = 1.0e-10f;

__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + expf(-x));
}

__device__ __forceinline__ float warp_reduce_max(float value) {
    value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, 16));
    value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, 8));
    value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, 4));
    value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, 2));
    value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, 1));
    return value;
}

__global__ void silu_mul_quant_fp8_packed_kernel(
    const __nv_bfloat16* __restrict__ input,
    __nv_fp8_e4m3* __restrict__ output,
    uint32_t* __restrict__ output_scales,
    int group_size,
    int num_groups_padded,
    int groups_per_block,
    int padded_groups_per_row,
    int groups_per_row,
    int mn,
    int hidden,
    int tma_aligned_mn,
    int num_scale_elems) {
    extern __shared__ float smem[];

    const int local_group = threadIdx.y;
    const int local_tid = threadIdx.x;
    const int global_group =
        static_cast<int>(blockIdx.x) * groups_per_block + local_group;
    if (global_group >= num_groups_padded)
        return;

    const int sf_k_idx = global_group % padded_groups_per_row;
    const int mn_idx = global_group / padded_groups_per_row;
    const bool valid = mn_idx < mn && sf_k_idx < groups_per_row;

    float* values = smem + static_cast<size_t>(local_group) * group_size;
    float local_absmax = kScaleEps;

    if (valid) {
        const int col_base = sf_k_idx * group_size;
        const __nv_bfloat16* gate =
            input + static_cast<int64_t>(mn_idx) * 2 * hidden + col_base;
        const __nv_bfloat16* up = gate + hidden;
        for (int i = local_tid; i < group_size; i += blockDim.x) {
            const float gate_v = __bfloat162float(gate[i]);
            const __nv_bfloat16 silu_bf16 = __float2bfloat16_rn(silu(gate_v));
            const float value = __bfloat162float(__hmul(silu_bf16, up[i]));
            values[i] = value;
            local_absmax = fmaxf(local_absmax, fabsf(value));
        }
    }

    float absmax = warp_reduce_max(local_absmax);
    __shared__ float scale_slots[16];
    if (local_tid == 0) {
        float scale = absmax / kFp8E4M3Max;
        scale = exp2f(ceilf(log2f(fmaxf(fabsf(scale), kScaleEps))));
        scale_slots[local_group] = scale;

        const int pack = sf_k_idx >> 2;
        const int pos = sf_k_idx & 3;
        const int out_idx = pack * tma_aligned_mn + mn_idx;
        if (valid) {
            const uint32_t bits = __float_as_uint(scale);
            const uint8_t exponent = static_cast<uint8_t>((bits >> 23u) & 0xffu);
            reinterpret_cast<uint8_t*>(output_scales)[out_idx * 4 + pos] = exponent;
        } else if (out_idx < num_scale_elems) {
            reinterpret_cast<uint8_t*>(output_scales)[out_idx * 4 + pos] = 0;
        }
    }
    __syncthreads();

    if (!valid)
        return;

    const float inv_scale = 1.0f / scale_slots[local_group];
    __nv_fp8_e4m3* out =
        output + static_cast<int64_t>(mn_idx) * hidden + sf_k_idx * group_size;
    for (int i = local_tid; i < group_size; i += blockDim.x) {
        const float q = fminf(fmaxf(values[i] * inv_scale, kFp8E4M3Min), kFp8E4M3Max);
        out[i] = __nv_fp8_e4m3(q);
    }
}

int groups_per_block_for(int64_t num_groups) {
    if (num_groups % 16 == 0)
        return 16;
    if (num_groups % 8 == 0)
        return 8;
    if (num_groups % 4 == 0)
        return 4;
    if (num_groups % 2 == 0)
        return 2;
    return 1;
}

}  // namespace

torch::Tensor sm120_silu_mul_quant_fp8_packed(const torch::Tensor& input,
                                              const torch::Tensor& output,
                                              int64_t group_size) {
    DG_HOST_ASSERT(input.is_cuda());
    DG_HOST_ASSERT(output.is_cuda());
    DG_HOST_ASSERT(input.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(output.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(input.dim() == 2);
    DG_HOST_ASSERT(output.dim() == 2);
    DG_HOST_ASSERT(input.is_contiguous());
    DG_HOST_ASSERT(output.is_contiguous());
    DG_HOST_ASSERT(input.size(1) % 2 == 0);
    const int64_t mn = input.size(0);
    const int64_t hidden = input.size(1) / 2;
    DG_HOST_ASSERT(output.size(0) == mn and output.size(1) == hidden);
    DG_HOST_ASSERT(group_size > 0 and hidden % group_size == 0);

    const int64_t groups_per_row = hidden / group_size;
    const int64_t packed_groups_per_row = (groups_per_row + 3) / 4;
    const int64_t padded_groups_per_row = packed_groups_per_row * 4;
    const int64_t tma_aligned_mn = ((mn + 3) / 4) * 4;

    auto scales = torch::empty_strided(
        {mn, packed_groups_per_row},
        {1, tma_aligned_mn},
        torch::TensorOptions().device(input.device()).dtype(torch::kInt32));

    const int64_t num_groups_padded = tma_aligned_mn * padded_groups_per_row;
    const int64_t num_scale_elems =
        mn + (packed_groups_per_row - 1) * tma_aligned_mn;
    if (num_groups_padded == 0)
        return scales;

    const int groups_per_block = groups_per_block_for(num_groups_padded);
    const dim3 block(32, groups_per_block);
    const dim3 grid(num_groups_padded / groups_per_block);
    const size_t smem_bytes =
        static_cast<size_t>(groups_per_block) * group_size * sizeof(float);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    silu_mul_quant_fp8_packed_kernel<<<grid, block, smem_bytes, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
        reinterpret_cast<__nv_fp8_e4m3*>(output.data_ptr()),
        reinterpret_cast<uint32_t*>(scales.data_ptr()),
        static_cast<int>(group_size),
        static_cast<int>(num_groups_padded),
        groups_per_block,
        static_cast<int>(padded_groups_per_row),
        static_cast<int>(groups_per_row),
        static_cast<int>(mn),
        static_cast<int>(hidden),
        static_cast<int>(tma_aligned_mn),
        static_cast<int>(num_scale_elems));
    return scales;
}

void register_apis(pybind11::module& m) {
    m.def("sm120_silu_mul_quant_fp8_packed",
          &sm120_silu_mul_quant_fp8_packed,
          pybind11::arg("input"), pybind11::arg("output"),
          pybind11::arg("group_size"));
}

}  // namespace deep_gemm::sm120_moe
