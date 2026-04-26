#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <type_traits>
#include <unordered_map>

#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/python.h>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cute/arch/mma_sm120.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/util/packed_stride.hpp"

#include "jit_kernels/impls/sm120_fp8_fp4_cutlass.hpp"
#include "sm120_profile.hpp"
#include "utils/exception.hpp"

namespace deep_gemm {
namespace {

using namespace cute;

using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;
using ElementInputA = cutlass::float_e4m3_t;
using ElementInputB = cutlass::float_e2m1_t;
using ElementA = cutlass::mx_float8_t<ElementInputA>;
using ElementB = cutlass::mx_float4_t<ElementInputB>;
using ElementSF = cutlass::float_ue8m0_t;
using ElementC = void;
using ElementD = cutlass::bfloat16_t;
using ElementAccumulator = float;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutD = cutlass::layout::RowMajor;
using TileShape = Shape<_128, _128, _128>;
using ClusterShape = Shape<_1, _1, _1>;

constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementInputA>::value;
constexpr int AlignmentB = 128;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp,
    TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutD*, 1,
    ElementD, LayoutD*, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp,
    ElementA, LayoutA*, AlignmentA,
    ElementB, LayoutB*, AlignmentB,
    ElementAccumulator,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpong>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    ProblemShape, CollectiveMainloop, CollectiveEpilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
using StrideA = typename Gemm::GemmKernel::InternalStrideA;
using StrideB = typename Gemm::GemmKernel::InternalStrideB;
using StrideD = typename Gemm::GemmKernel::InternalStrideD;
using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::InternalLayoutSFA;
using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::InternalLayoutSFB;
using MxScaleConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

struct Scratch {
    int device = -1;
    int ptr_a_capacity = 0;
    int ptr_b_capacity = 0;
    int ptr_sfa_capacity = 0;
    int ptr_sfb_capacity = 0;
    int ptr_d_capacity = 0;
    int stride_a_capacity = 0;
    int stride_b_capacity = 0;
    int stride_d_capacity = 0;
    int layout_sfa_capacity = 0;
    int layout_sfb_capacity = 0;
    int problems_capacity = 0;
    int active_count_capacity = 0;
    size_t sfa_capacity = 0;
    size_t workspace_capacity = 0;
    ElementInputA const** ptr_a = nullptr;
    ElementInputB const** ptr_b = nullptr;
    ElementSF const** ptr_sfa = nullptr;
    ElementSF const** ptr_sfb = nullptr;
    ElementD** ptr_d = nullptr;
    StrideA* stride_a = nullptr;
    StrideB* stride_b = nullptr;
    StrideD* stride_d = nullptr;
    LayoutSFA* layout_sfa = nullptr;
    LayoutSFB* layout_sfb = nullptr;
    typename ProblemShape::UnderlyingProblemShape* problems = nullptr;
    int* active_count = nullptr;
    ElementSF* sfa_ue8 = nullptr;
    void* workspace = nullptr;
};

struct StaticScale {
    int device = -1;
    int num_groups = 0;
    int m = 0;
    int n = 0;
    int k32 = 0;
    size_t group_elems = 0;
    torch::Tensor owner;
    torch::Tensor packed;
    ElementSF* data = nullptr;
};

std::mutex g_mutex;
Scratch g_scratch[16];
std::unordered_map<uintptr_t, StaticScale> g_sfb_cache;

void cuda_check(cudaError_t status) {
    DG_HOST_ASSERT(status == cudaSuccess);
}

template <typename T>
void reserve_array(T*& ptr, int& capacity, int needed) {
    if (ptr != nullptr && capacity >= needed)
        return;
    if (ptr != nullptr)
        cuda_check(cudaFree(ptr));
    cuda_check(cudaMalloc(&ptr, sizeof(T) * needed));
    capacity = needed;
}

void* reserve_bytes(void*& ptr, size_t& capacity, size_t needed) {
    if (capacity >= needed)
        return ptr;
    if (ptr != nullptr)
        cuda_check(cudaFree(ptr));
    cuda_check(cudaMalloc(&ptr, needed));
    capacity = needed;
    return ptr;
}

bool env_flag_enabled(const char* name) {
    const char* value = std::getenv(name);
    return value != nullptr && std::strcmp(value, "0") != 0 &&
           std::strcmp(value, "false") != 0 &&
           std::strcmp(value, "False") != 0;
}

int env_int_or_default(const char* name, int fallback) {
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0')
        return fallback;
    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value)
        return fallback;
    if (parsed < std::numeric_limits<int>::min())
        return std::numeric_limits<int>::min();
    if (parsed > std::numeric_limits<int>::max())
        return std::numeric_limits<int>::max();
    return static_cast<int>(parsed);
}

Scratch& get_scratch(int device, int num_groups, size_t sfa_bytes, size_t workspace_bytes) {
    std::lock_guard<std::mutex> lock(g_mutex);
    Scratch& s = g_scratch[device];
    s.device = device;
    reserve_array(s.ptr_a, s.ptr_a_capacity, num_groups);
    reserve_array(s.ptr_b, s.ptr_b_capacity, num_groups);
    reserve_array(s.ptr_sfa, s.ptr_sfa_capacity, num_groups);
    reserve_array(s.ptr_sfb, s.ptr_sfb_capacity, num_groups);
    reserve_array(s.ptr_d, s.ptr_d_capacity, num_groups);
    reserve_array(s.stride_a, s.stride_a_capacity, num_groups);
    reserve_array(s.stride_b, s.stride_b_capacity, num_groups);
    reserve_array(s.stride_d, s.stride_d_capacity, num_groups);
    reserve_array(s.layout_sfa, s.layout_sfa_capacity, num_groups);
    reserve_array(s.layout_sfb, s.layout_sfb_capacity, num_groups);
    reserve_array(s.problems, s.problems_capacity, num_groups);
    reserve_array(s.active_count, s.active_count_capacity, 1);
    reserve_bytes(reinterpret_cast<void*&>(s.sfa_ue8), s.sfa_capacity, sfa_bytes);
    reserve_bytes(s.workspace, s.workspace_capacity, workspace_bytes);
    return s;
}

__device__ __forceinline__ ElementSF to_ue8m0(float x) {
    return ElementSF(x);
}

__global__ void fill_scale_kernel(ElementSF* __restrict__ out, size_t total) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < total)
        out[idx] = ElementSF(1.0f);
}

__global__ void convert_sfb_kernel(const float* __restrict__ sfb,
                                   ElementSF* __restrict__ out,
                                   int total, int64_t sfb_s0, int64_t sfb_s1,
                                   int64_t sfb_s2, int m, int n, int k,
                                   int k32, size_t group_elems) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total)
        return;
    const int kk = idx % k32;
    const int tmp = idx / k32;
    const int col = tmp % n;
    const int group = tmp / n;
    const auto layout = MxScaleConfig::tile_atom_to_shape_SFB(make_shape(m, n, k, 1));
    out[group * group_elems + layout(col, kk * 32, 0)] =
        to_ue8m0(sfb[group * sfb_s0 + col * sfb_s1 + kk * sfb_s2]);
}

__global__ void copy_sfb_ue8_kernel(const uint8_t* __restrict__ sfb,
                                    ElementSF* __restrict__ out,
                                    int total, int64_t sfb_s0, int64_t sfb_s1,
                                    int64_t sfb_s2, int m, int n, int k,
                                    int k32, size_t group_elems) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total)
        return;
    const int kk = idx % k32;
    const int tmp = idx / k32;
    const int col = tmp % n;
    const int group = tmp / n;
    const auto layout = MxScaleConfig::tile_atom_to_shape_SFB(make_shape(m, n, k, 1));
    reinterpret_cast<uint8_t*>(out)[group * group_elems + layout(col, kk * 32, 0)] =
        sfb[group * sfb_s0 + col * sfb_s1 + kk * sfb_s2];
}

__device__ __forceinline__ uint8_t load_sfa_ue8m0_i32(
    const int32_t* __restrict__ sfa, int row, int k_block32,
    int64_t sfa_s0, int64_t sfa_s1) {
    const int k_block128 = k_block32 >> 2;
    const int32_t packed =
        sfa[static_cast<int64_t>(row) * sfa_s0 +
            static_cast<int64_t>(k_block128 >> 2) * sfa_s1];
    return static_cast<uint8_t>((packed >> ((k_block128 & 3) * 8)) & 0xff);
}

__device__ __forceinline__ uint8_t load_fp4_for_mma(
    const int8_t* __restrict__ b, int group, int n_col, int k_col,
    int64_t b_s0, int64_t b_s1, int64_t b_s2) {
    const uint8_t packed = static_cast<uint8_t>(
        b[static_cast<int64_t>(group) * b_s0 +
          static_cast<int64_t>(n_col) * b_s1 +
          static_cast<int64_t>(k_col >> 1) * b_s2]);
    const uint8_t nibble = (k_col & 1) ? (packed >> 4) : (packed & 0x0f);
    // SM120 mxf8f6f4 MMA expects FP4 in bits [5:2], not the low nibble.
    return static_cast<uint8_t>(nibble << 2);
}

__device__ __forceinline__ void store_bf16(ElementD* __restrict__ d,
                                           int64_t offset, float value) {
    auto* out = reinterpret_cast<__nv_bfloat16*>(d);
    out[offset] = __float2bfloat16(value);
}

__global__ void small_m_fp8_fp4_mma_kernel(
    const uint8_t* __restrict__ a, const int32_t* __restrict__ sfa,
    const int8_t* __restrict__ b, const ElementSF* __restrict__ sfb,
    ElementD* __restrict__ d, const int32_t* __restrict__ grouped_layout,
    int m, int n, int k, int k32, int sfb_layout_m,
    int64_t a_s0, int64_t a_s1, int64_t sfa_s0, int64_t sfa_s1,
    int64_t b_s0, int64_t b_s1, int64_t b_s2,
    int64_t d_s0, int64_t d_s1, size_t sfb_group_elems) {
#if defined(CUTE_ARCH_MXF8F6F4_MMA_ENABLED)
    using MmaOp = cute::SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<
        cutlass::float_e4m3_t, cutlass::float_e2m1_t, float,
        cutlass::float_ue8m0_t, 32>;

    constexpr int kWarpsPerBlock = 4;
    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int t0 = lane & 3;
    const int t1 = lane >> 2;
    const int row = blockIdx.y;
    if (warp_id >= kWarpsPerBlock)
        return;
    const int col_base = (blockIdx.x * kWarpsPerBlock + warp_id) * 8;
    if (row >= m || col_base >= n)
        return;

    const int group = grouped_layout[row];
    if (group < 0)
        return;

    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const auto sfb_layout = MxScaleConfig::tile_atom_to_shape_SFB(
        make_shape(sfb_layout_m, n, k, 1));

#pragma unroll 1
    for (int kb = 0; kb < k32; ++kb) {
        uint32_t a_reg[4] = {0u, 0u, 0u, 0u};
        uint32_t b_reg[2] = {0u, 0u};

#pragma unroll
        for (int v = 0; v < 16; ++v) {
            const int v0 = v & 3;
            const int v1 = (v >> 2) & 1;
            const int v2 = v >> 3;
            const int offset = t0 * 64 + t1 + v0 * 16 + v1 * 8 + v2 * 256;
            const int local_row = offset & 15;
            const int kk = offset >> 4;
            uint8_t value = 0;
            if (local_row == 0) {
                value = a[static_cast<int64_t>(row) * a_s0 +
                          static_cast<int64_t>(kb * 32 + kk) * a_s1];
            }
            a_reg[v >> 2] |= static_cast<uint32_t>(value) << ((v & 3) * 8);
        }

#pragma unroll
        for (int v = 0; v < 8; ++v) {
            const int v0 = v & 3;
            const int v1 = v >> 2;
            const int offset = t0 * 32 + t1 + v0 * 8 + v1 * 128;
            const int kk = offset >> 3;
            const int n_local = offset & 7;
            const int col = col_base + n_local;
            uint8_t value = 0;
            if (col < n) {
                value = load_fp4_for_mma(b, group, col, kb * 32 + kk,
                                         b_s0, b_s1, b_s2);
            }
            b_reg[v >> 2] |= static_cast<uint32_t>(value) << ((v & 3) * 8);
        }

        const uint8_t sfa_value =
            load_sfa_ue8m0_i32(sfa, row, kb, sfa_s0, sfa_s1);
        const int sfb_col = min(col_base + t1, n - 1);
        const uint8_t sfb_value = reinterpret_cast<const uint8_t*>(
            sfb + static_cast<size_t>(group) * sfb_group_elems)[
            sfb_layout(sfb_col, kb * 32, 0)];

        MmaOp::fma(acc[0], acc[1], acc[2], acc[3],
                   a_reg[0], a_reg[1], a_reg[2], a_reg[3],
                   b_reg[0], b_reg[1],
                   acc[0], acc[1], acc[2], acc[3],
                   sfa_value, sfb_value);
    }

    if (t1 == 0) {
        const int col0 = col_base + t0 * 2;
        if (col0 < n) {
            store_bf16(d,
                       static_cast<int64_t>(row) * d_s0 +
                           static_cast<int64_t>(col0) * d_s1,
                       acc[0]);
        }
        const int col1 = col0 + 1;
        if (col1 < n) {
            store_bf16(d,
                       static_cast<int64_t>(row) * d_s0 +
                           static_cast<int64_t>(col1) * d_s1,
                       acc[1]);
        }
    }
#endif
}

__global__ void prepare_grouped_kernel(
    const uint8_t* __restrict__ a, const void* __restrict__ sfa,
    const int8_t* __restrict__ b, ElementD* __restrict__ d,
    const int32_t* __restrict__ grouped_layout,
    const int32_t* __restrict__ expert_starts,
    const int32_t* __restrict__ expert_counts,
    ElementInputA const** __restrict__ ptr_a,
    ElementInputB const** __restrict__ ptr_b,
    ElementSF const** __restrict__ ptr_sfa,
    ElementSF const** __restrict__ ptr_sfb,
    ElementD** __restrict__ ptr_d,
    StrideA* __restrict__ stride_a,
    StrideB* __restrict__ stride_b,
    StrideD* __restrict__ stride_d,
    LayoutSFA* __restrict__ layout_sfa,
    LayoutSFB* __restrict__ layout_sfb,
    typename ProblemShape::UnderlyingProblemShape* __restrict__ problems,
    ElementSF* __restrict__ sfa_ue8,
    const ElementSF* __restrict__ sfb_ue8,
    int num_groups, int m, int n, int k, int k32,
    int sfa_kind, int sfb_layout_m,
    size_t sfa_group_elems,
    size_t sfb_group_elems,
    int64_t a_s0, int64_t a_s1, int64_t sfa_s0, int64_t sfa_s1,
    int64_t b_s0, int64_t d_s0, int64_t d_s1) {
    const int group = blockIdx.x;
    int first = m;
    int rows = 0;
    int scale_rows = 0;
    if (expert_starts != nullptr && expert_counts != nullptr) {
        const int count = expert_counts[group];
        first = expert_starts[group];
        if (count <= 0 || first < 0 || first >= m) {
            first = m;
        } else {
            rows = min(count, m - first);
            scale_rows = rows;
        }
    } else {
        int next_first = m;
        for (int row = threadIdx.x; row < m; row += blockDim.x) {
            const int row_group = grouped_layout[row];
            if (row_group == group)
                first = min(first, row);
            if (row_group == group + 1)
                next_first = min(next_first, row);
        }
        __shared__ int s_first[256];
        __shared__ int s_next_first[256];
        s_first[threadIdx.x] = first;
        s_next_first[threadIdx.x] = next_first;
        __syncthreads();
        for (int step = blockDim.x / 2; step > 0; step >>= 1) {
            if (threadIdx.x < step) {
                s_first[threadIdx.x] = min(s_first[threadIdx.x], s_first[threadIdx.x + step]);
                s_next_first[threadIdx.x] = min(s_next_first[threadIdx.x], s_next_first[threadIdx.x + step]);
            }
            __syncthreads();
        }
        first = s_first[0];
        next_first = s_next_first[0];
        rows = first < m ? max(0, next_first - first) : 0;
        scale_rows = rows;
    }

    ElementSF* group_sfa = sfa_ue8 + static_cast<size_t>(group) * sfa_group_elems;
    const auto group_layout_sfa = MxScaleConfig::tile_atom_to_shape_SFA(make_shape(rows, n, k, 1));

    for (int idx = threadIdx.x; idx < scale_rows * k32; idx += blockDim.x) {
        const int kk32 = idx % k32;
        const int row = idx / k32;
        const auto out_idx = group_layout_sfa(row, kk32 * 32, 0);
        if (sfa_kind == 0) {
            const auto* sfa_f32 = static_cast<const float*>(sfa);
            group_sfa[out_idx] =
                to_ue8m0(sfa_f32[(first + row) * sfa_s0 + (kk32 / 4) * sfa_s1]);
        } else {
            const auto* sfa_i32 = static_cast<const int32_t*>(sfa);
            const int kk128 = kk32 / 4;
            const int32_t packed =
                sfa_i32[(first + row) * sfa_s0 + (kk128 / 4) * sfa_s1];
            reinterpret_cast<uint8_t*>(group_sfa)[out_idx] =
                static_cast<uint8_t>((packed >> ((kk128 & 3) * 8)) & 0xff);
        }
    }

    if (threadIdx.x == 0) {
        const int ptr_row = first < m ? first : 0;
        problems[group] = make_shape(rows, n, k);
        ptr_a[group] = reinterpret_cast<ElementInputA const*>(a + ptr_row * a_s0);
        ptr_b[group] = reinterpret_cast<ElementInputB const*>(b + group * b_s0);
        ptr_sfa[group] = group_sfa;
        ptr_sfb[group] = sfb_ue8 + group * sfb_group_elems;
        ptr_d[group] = d + ptr_row * d_s0;
        stride_a[group] = cutlass::make_cute_packed_stride(StrideA{}, make_shape(rows, k, 1));
        stride_b[group] = cutlass::make_cute_packed_stride(StrideB{}, make_shape(n, k, 1));
        stride_d[group] = cutlass::make_cute_packed_stride(StrideD{}, make_shape(rows, n, 1));
        layout_sfa[group] = MxScaleConfig::tile_atom_to_shape_SFA(make_shape(rows, n, k, 1));
        layout_sfb[group] = MxScaleConfig::tile_atom_to_shape_SFB(
            make_shape(sfb_layout_m > 0 ? sfb_layout_m : rows, n, k, 1));
    }
}

__global__ void init_compact_grouped_kernel(
    const uint8_t* __restrict__ a, const int8_t* __restrict__ b,
    ElementD* __restrict__ d,
    ElementInputA const** __restrict__ ptr_a,
    ElementInputB const** __restrict__ ptr_b,
    ElementSF const** __restrict__ ptr_sfa,
    ElementSF const** __restrict__ ptr_sfb,
    ElementD** __restrict__ ptr_d,
    StrideA* __restrict__ stride_a,
    StrideB* __restrict__ stride_b,
    StrideD* __restrict__ stride_d,
    LayoutSFA* __restrict__ layout_sfa,
    LayoutSFB* __restrict__ layout_sfb,
    typename ProblemShape::UnderlyingProblemShape* __restrict__ problems,
    ElementSF* __restrict__ sfa_ue8,
    const ElementSF* __restrict__ sfb_ue8,
    int* __restrict__ active_count,
    int max_compact_groups, int n, int k,
    int sfb_layout_m, size_t sfa_group_elems, size_t sfb_group_elems,
    int64_t d_s0) {
    if (blockIdx.x == 0 && threadIdx.x == 0)
        *active_count = 0;

    for (int slot = blockIdx.x * blockDim.x + threadIdx.x;
         slot < max_compact_groups;
         slot += gridDim.x * blockDim.x) {
        problems[slot] = make_shape(0, n, k);
        ptr_a[slot] = reinterpret_cast<ElementInputA const*>(a);
        ptr_b[slot] = reinterpret_cast<ElementInputB const*>(b);
        ptr_sfa[slot] = sfa_ue8 + static_cast<size_t>(slot) * sfa_group_elems;
        ptr_sfb[slot] = sfb_ue8;
        ptr_d[slot] = d;
        stride_a[slot] =
            cutlass::make_cute_packed_stride(StrideA{}, make_shape(0, k, 1));
        stride_b[slot] =
            cutlass::make_cute_packed_stride(StrideB{}, make_shape(n, k, 1));
        stride_d[slot] =
            cutlass::make_cute_packed_stride(StrideD{}, make_shape(0, n, 1));
        layout_sfa[slot] =
            MxScaleConfig::tile_atom_to_shape_SFA(make_shape(0, n, k, 1));
        layout_sfb[slot] = MxScaleConfig::tile_atom_to_shape_SFB(
            make_shape(sfb_layout_m > 0 ? sfb_layout_m : 1, n, k, 1));

    }
}

__global__ void prepare_compact_grouped_kernel(
    const uint8_t* __restrict__ a, const void* __restrict__ sfa,
    const int8_t* __restrict__ b, ElementD* __restrict__ d,
    const int32_t* __restrict__ expert_starts,
    const int32_t* __restrict__ expert_counts,
    ElementInputA const** __restrict__ ptr_a,
    ElementInputB const** __restrict__ ptr_b,
    ElementSF const** __restrict__ ptr_sfa,
    ElementSF const** __restrict__ ptr_sfb,
    ElementD** __restrict__ ptr_d,
    StrideA* __restrict__ stride_a,
    StrideB* __restrict__ stride_b,
    StrideD* __restrict__ stride_d,
    LayoutSFA* __restrict__ layout_sfa,
    LayoutSFB* __restrict__ layout_sfb,
    typename ProblemShape::UnderlyingProblemShape* __restrict__ problems,
    ElementSF* __restrict__ sfa_ue8,
    const ElementSF* __restrict__ sfb_ue8,
    int* __restrict__ active_count,
    int num_groups, int max_compact_groups, int m, int n, int k, int k32,
    int sfa_kind, int sfb_layout_m,
    size_t sfa_group_elems,
    size_t sfb_group_elems,
    int64_t a_s0, int64_t a_s1, int64_t sfa_s0, int64_t sfa_s1,
    int64_t b_s0, int64_t d_s0, int64_t d_s1) {
    const int group = blockIdx.x;
    if (group >= num_groups)
        return;

    const int count = expert_counts[group];
    const int first = expert_starts[group];
    if (count <= 0 || first < 0 || first >= m)
        return;

    const int rows = min(count, m - first);
    if (rows <= 0)
        return;

    __shared__ int slot_s;
    if (threadIdx.x == 0)
        slot_s = atomicAdd(active_count, 1);
    __syncthreads();

    const int slot = slot_s;
    if (slot >= max_compact_groups)
        return;

    ElementSF* group_sfa =
        sfa_ue8 + static_cast<size_t>(slot) * sfa_group_elems;
    const auto group_layout_sfa =
        MxScaleConfig::tile_atom_to_shape_SFA(make_shape(rows, n, k, 1));

    for (int idx = threadIdx.x; idx < rows * k32; idx += blockDim.x) {
        const int kk32 = idx % k32;
        const int row = idx / k32;
        const auto out_idx = group_layout_sfa(row, kk32 * 32, 0);
        if (sfa_kind == 0) {
            const auto* sfa_f32 = static_cast<const float*>(sfa);
            group_sfa[out_idx] =
                to_ue8m0(sfa_f32[(first + row) * sfa_s0 +
                                  (kk32 / 4) * sfa_s1]);
        } else {
            const auto* sfa_i32 = static_cast<const int32_t*>(sfa);
            const int kk128 = kk32 / 4;
            const int32_t packed =
                sfa_i32[(first + row) * sfa_s0 + (kk128 / 4) * sfa_s1];
            reinterpret_cast<uint8_t*>(group_sfa)[out_idx] =
                static_cast<uint8_t>((packed >> ((kk128 & 3) * 8)) & 0xff);
        }
    }

    if (threadIdx.x == 0) {
        problems[slot] = make_shape(rows, n, k);
        ptr_a[slot] = reinterpret_cast<ElementInputA const*>(a + first * a_s0);
        ptr_b[slot] = reinterpret_cast<ElementInputB const*>(b + group * b_s0);
        ptr_sfa[slot] = group_sfa;
        ptr_sfb[slot] = sfb_ue8 + static_cast<size_t>(group) * sfb_group_elems;
        ptr_d[slot] = d + first * d_s0;
        stride_a[slot] =
            cutlass::make_cute_packed_stride(StrideA{}, make_shape(rows, k, 1));
        stride_b[slot] =
            cutlass::make_cute_packed_stride(StrideB{}, make_shape(n, k, 1));
        stride_d[slot] =
            cutlass::make_cute_packed_stride(StrideD{}, make_shape(rows, n, 1));
        layout_sfa[slot] =
            MxScaleConfig::tile_atom_to_shape_SFA(make_shape(rows, n, k, 1));
        layout_sfb[slot] = MxScaleConfig::tile_atom_to_shape_SFB(
            make_shape(sfb_layout_m > 0 ? sfb_layout_m : rows, n, k, 1));
    }
}

__global__ void prepare_row_grouped_kernel(
    const uint8_t* __restrict__ a, const void* __restrict__ sfa,
    const int8_t* __restrict__ b, ElementD* __restrict__ d,
    const int32_t* __restrict__ grouped_layout,
    ElementInputA const** __restrict__ ptr_a,
    ElementInputB const** __restrict__ ptr_b,
    ElementSF const** __restrict__ ptr_sfa,
    ElementSF const** __restrict__ ptr_sfb,
    ElementD** __restrict__ ptr_d,
    StrideA* __restrict__ stride_a,
    StrideB* __restrict__ stride_b,
    StrideD* __restrict__ stride_d,
    LayoutSFA* __restrict__ layout_sfa,
    LayoutSFB* __restrict__ layout_sfb,
    typename ProblemShape::UnderlyingProblemShape* __restrict__ problems,
    ElementSF* __restrict__ sfa_ue8,
    const ElementSF* __restrict__ sfb_ue8,
    int m, int n, int k, int k32, int sfa_kind, int sfb_layout_m,
    size_t sfa_group_elems, size_t sfb_group_elems,
    int64_t a_s0, int64_t a_s1, int64_t sfa_s0, int64_t sfa_s1,
    int64_t b_s0, int64_t d_s0, int64_t d_s1) {
    const int row = blockIdx.x;
    if (row >= m)
        return;

    const int group = grouped_layout[row];
    ElementSF* row_sfa = sfa_ue8 + static_cast<size_t>(row) * sfa_group_elems;
    const auto row_layout_sfa =
        MxScaleConfig::tile_atom_to_shape_SFA(make_shape(1, n, k, 1));

    if (group >= 0) {
        for (int kk32 = threadIdx.x; kk32 < k32; kk32 += blockDim.x) {
            const auto out_idx = row_layout_sfa(0, kk32 * 32, 0);
            if (sfa_kind == 0) {
                const auto* sfa_f32 = static_cast<const float*>(sfa);
                row_sfa[out_idx] =
                    to_ue8m0(sfa_f32[row * sfa_s0 + (kk32 / 4) * sfa_s1]);
            } else {
                const auto* sfa_i32 = static_cast<const int32_t*>(sfa);
                const int kk128 = kk32 / 4;
                const int32_t packed =
                    sfa_i32[row * sfa_s0 + (kk128 / 4) * sfa_s1];
                reinterpret_cast<uint8_t*>(row_sfa)[out_idx] =
                    static_cast<uint8_t>((packed >> ((kk128 & 3) * 8)) & 0xff);
            }
        }
    }

    if (threadIdx.x == 0) {
        const int rows = group >= 0 ? 1 : 0;
        problems[row] = make_shape(rows, n, k);
        ptr_a[row] = reinterpret_cast<ElementInputA const*>(a + row * a_s0);
        ptr_b[row] = reinterpret_cast<ElementInputB const*>(
            b + static_cast<int64_t>(max(group, 0)) * b_s0);
        ptr_sfa[row] = row_sfa;
        ptr_sfb[row] =
            sfb_ue8 + static_cast<size_t>(max(group, 0)) * sfb_group_elems;
        ptr_d[row] = d + row * d_s0;
        stride_a[row] =
            cutlass::make_cute_packed_stride(StrideA{}, make_shape(rows, k, 1));
        stride_b[row] =
            cutlass::make_cute_packed_stride(StrideB{}, make_shape(n, k, 1));
        stride_d[row] =
            cutlass::make_cute_packed_stride(StrideD{}, make_shape(rows, n, 1));
        layout_sfa[row] = row_layout_sfa;
        layout_sfb[row] = MxScaleConfig::tile_atom_to_shape_SFB(
            make_shape(sfb_layout_m > 0 ? sfb_layout_m : rows, n, k, 1));
    }
}

StaticScale get_or_create_sfb(const torch::Tensor& sfb, int num_groups, int n, int k32,
                              int m, int k, size_t sfb_group_elems, cudaStream_t stream) {
    const int device = sfb.get_device();
    if (sfb.scalar_type() == torch::kUInt8 && sfb.dim() == 2 &&
        sfb.size(0) == num_groups &&
        sfb.size(1) == static_cast<int64_t>(sfb_group_elems)) {
        StaticScale scale;
        scale.device = device;
        scale.num_groups = num_groups;
        scale.m = m;
        scale.n = n;
        scale.k32 = k32;
        scale.group_elems = sfb_group_elems;
        scale.owner = sfb;
        scale.data = reinterpret_cast<ElementSF*>(sfb.data_ptr<uint8_t>());
        return scale;
    }

    const uintptr_t key = reinterpret_cast<uintptr_t>(sfb.data_ptr());
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        auto it = g_sfb_cache.find(key);
        if (it != g_sfb_cache.end() && it->second.device == device &&
            it->second.num_groups == num_groups && it->second.m == m &&
            it->second.n == n &&
            it->second.k32 == k32 && it->second.group_elems == sfb_group_elems) {
            return it->second;
        }
    }

    StaticScale scale;
    scale.device = device;
    scale.num_groups = num_groups;
    scale.m = m;
    scale.n = n;
    scale.k32 = k32;
    scale.group_elems = sfb_group_elems;
    scale.owner = sfb;
    scale.packed = torch::empty(
        {num_groups, static_cast<int64_t>(sfb_group_elems)},
        sfb.options().dtype(torch::kUInt8));
    scale.data = reinterpret_cast<ElementSF*>(scale.packed.data_ptr<uint8_t>());
    const size_t scale_total = static_cast<size_t>(num_groups) * sfb_group_elems;
    fill_scale_kernel<<<static_cast<unsigned>((scale_total + 255) / 256), 256, 0, stream>>>(
        scale.data, scale_total);
    DG_CUDA_RUNTIME_CHECK(cudaGetLastError());
    const int total = num_groups * n * k32;
    if (sfb.scalar_type() == torch::kFloat32) {
        convert_sfb_kernel<<<(total + 255) / 256, 256, 0, stream>>>(
            sfb.data_ptr<float>(), scale.data, total, sfb.stride(0), sfb.stride(1),
            sfb.stride(2), m, n, k, k32, sfb_group_elems);
    } else if (sfb.scalar_type() == torch::kUInt8 && sfb.dim() == 3) {
        copy_sfb_ue8_kernel<<<(total + 255) / 256, 256, 0, stream>>>(
            sfb.data_ptr<uint8_t>(), scale.data, total, sfb.stride(0), sfb.stride(1),
            sfb.stride(2), m, n, k, k32, sfb_group_elems);
    } else {
        DG_HOST_UNREACHABLE("Unsupported SM120 FP4 scale tensor");
    }
    DG_CUDA_RUNTIME_CHECK(cudaGetLastError());

    std::lock_guard<std::mutex> lock(g_mutex);
    g_sfb_cache[key] = scale;
    return scale;
}

bool sm120_m_grouped_fp8_fp4_gemm_nt_contiguous_cutlass_impl(
    const torch::Tensor& a, const torch::Tensor& sfa,
    const torch::Tensor& b, const torch::Tensor& sfb,
    const torch::Tensor& d, const torch::Tensor& grouped_layout,
    const torch::Tensor* expert_starts, const torch::Tensor* expert_counts,
    int num_groups, int m, int n, int k, int gran_k_a, int gran_k_b,
    const cute::UMMA::Major& major_b, bool use_psum_layout) {
    if (std::getenv("DG_SM120_DISABLE_CUTLASS_MX") != nullptr)
        return false;
    if (std::getenv("VLLM_USE_DEEP_GEMM_E8M0") == nullptr &&
        std::getenv("DG_SM120_ASSUME_UE8M0_SCALES") == nullptr)
        return false;
    if (use_psum_layout || gran_k_a != 128 || gran_k_b != 32 ||
        major_b != cute::UMMA::Major::K || k % 128 != 0 || n % 128 != 0)
        return false;
    if (!(sfa.scalar_type() == torch::kFloat32 || sfa.scalar_type() == torch::kInt32))
        return false;
    if (!(sfb.scalar_type() == torch::kFloat32 || sfb.scalar_type() == torch::kUInt8))
        return false;
    if ((expert_starts == nullptr) != (expert_counts == nullptr))
        return false;
    if (expert_starts != nullptr) {
        if (expert_starts->scalar_type() != torch::kInt ||
            expert_counts->scalar_type() != torch::kInt ||
            expert_starts->numel() != num_groups ||
            expert_counts->numel() != num_groups ||
            !expert_starts->is_contiguous() ||
            !expert_counts->is_contiguous()) {
            return false;
        }
    }

    const int device = a.get_device();
    const bool compact_active_groups = expert_starts != nullptr && expert_counts != nullptr &&
                                       !use_psum_layout;
    const int gemm_num_groups = compact_active_groups ? std::min(num_groups, m) : num_groups;
    const int k32 = k / 32;
    const int sfa_kind = sfa.scalar_type() == torch::kFloat32 ? 0 : 1;
    const bool sfb_is_prepacked = sfb.scalar_type() == torch::kUInt8 && sfb.dim() == 2;
    const int sfb_layout_m = sfb_is_prepacked ? 128 : 0;
    const auto max_layout_sfa = MxScaleConfig::tile_atom_to_shape_SFA(make_shape(m, n, k, 1));
    const auto max_layout_sfb = MxScaleConfig::tile_atom_to_shape_SFB(
        make_shape(sfb_is_prepacked ? sfb_layout_m : m, n, k, 1));
    const size_t sfa_group_elems = static_cast<size_t>(size(filter_zeros(max_layout_sfa)));
    const size_t sfa_total_elems = static_cast<size_t>(num_groups) * sfa_group_elems;
    const size_t sfb_group_elems = static_cast<size_t>(size(filter_zeros(max_layout_sfb)));
    const auto stream = at::cuda::getCurrentCUDAStream();
    const int row_grouped_max_m =
        env_int_or_default("DG_SM120_MOE_ROW_GROUPED_MAX_M", 16);
    const bool row_grouped =
        compact_active_groups && env_flag_enabled("DG_SM120_MOE_ROW_GROUPED") &&
        (row_grouped_max_m < 0 || m <= row_grouped_max_m);
    static sm120_profile::KernelProfileCounter profile_counter(
        "sm120_moe_fp8_fp4_cutlass");
    sm120_profile::ScopedTimer profile_timer(
        profile_counter, stream, m, n, k, num_groups);
    StaticScale sfb_ue8 = get_or_create_sfb(sfb, num_groups, n, k32, m, k,
                                            sfb_group_elems, stream);

    if (compact_active_groups && m <= 64 && n % 8 == 0 && k % 32 == 0 &&
        sfa_kind == 1 && sfb_is_prepacked &&
        env_flag_enabled("DG_SM120_ENABLE_SMALL_M_MMA")) {
        constexpr int small_m_warps_per_block = 4;
        small_m_fp8_fp4_mma_kernel<<<
            dim3((n + small_m_warps_per_block * 8 - 1) /
                     (small_m_warps_per_block * 8),
                 m),
            small_m_warps_per_block * 32, 0, stream>>>(
            reinterpret_cast<const uint8_t*>(a.data_ptr()),
            sfa.data_ptr<int32_t>(), b.data_ptr<int8_t>(), sfb_ue8.data,
            reinterpret_cast<ElementD*>(d.data_ptr()),
            grouped_layout.data_ptr<int32_t>(), m, n, k, k32, sfb_layout_m,
            a.stride(0), a.stride(1), sfa.stride(0), sfa.stride(1),
            b.stride(0), b.stride(1), b.stride(2), d.stride(0),
            d.stride(1), sfb_group_elems);
        DG_CUDA_RUNTIME_CHECK(cudaGetLastError());
        return true;
    }

    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = device;
    hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(device);

    typename Gemm::GemmKernel::TileSchedulerArguments scheduler;
    scheduler.raster_order = cutlass::gemm::kernel::detail::RasterOrderOptions::AlongM;

    typename Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGrouped,
        {row_grouped ? m : gemm_num_groups, nullptr, nullptr},
        {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr},
        {{}, nullptr, nullptr, nullptr, nullptr},
        hw_info,
        scheduler};

    const size_t workspace_size = Gemm::get_workspace_size(arguments);
    Scratch& scratch = get_scratch(device, row_grouped ? m : num_groups,
                                   sizeof(ElementSF) *
                                       (row_grouped
                                            ? static_cast<size_t>(m) *
                                                  sfa_group_elems
                                        : compact_active_groups
                                            ? static_cast<size_t>(gemm_num_groups) *
                                                  sfa_group_elems
                                            : sfa_total_elems),
                                   workspace_size);

    if (row_grouped) {
        const size_t row_sfa_total = static_cast<size_t>(m) * sfa_group_elems;
        const bool skip_row_sfa_fill =
            sfa_kind == 1 && env_flag_enabled("DG_SM120_MOE_ROW_GROUPED_SKIP_SFA_FILL");
        if (!skip_row_sfa_fill) {
            fill_scale_kernel<<<static_cast<unsigned>((row_sfa_total + 255) / 256), 256, 0, stream>>>(
                scratch.sfa_ue8, row_sfa_total);
            DG_CUDA_RUNTIME_CHECK(cudaGetLastError());
        }

        prepare_row_grouped_kernel<<<m, 256, 0, stream>>>(
            reinterpret_cast<const uint8_t*>(a.data_ptr()), sfa.data_ptr(),
            b.data_ptr<int8_t>(), reinterpret_cast<ElementD*>(d.data_ptr()),
            grouped_layout.data_ptr<int32_t>(), scratch.ptr_a, scratch.ptr_b,
            scratch.ptr_sfa, scratch.ptr_sfb, scratch.ptr_d, scratch.stride_a,
            scratch.stride_b, scratch.stride_d, scratch.layout_sfa,
            scratch.layout_sfb, scratch.problems, scratch.sfa_ue8,
            sfb_ue8.data, m, n, k, k32, sfa_kind, sfb_layout_m,
            sfa_group_elems, sfb_group_elems, a.stride(0), a.stride(1),
            sfa.stride(0), sfa.stride(1), b.stride(0), d.stride(0),
            d.stride(1));
        DG_CUDA_RUNTIME_CHECK(cudaGetLastError());
    } else if (compact_active_groups) {
        const size_t compact_sfa_total =
            static_cast<size_t>(gemm_num_groups) * sfa_group_elems;
        const int skip_fill_max_m =
            env_int_or_default("DG_SM120_MOE_SKIP_SFA_FILL_MAX_M", 16);
        const bool skip_sfa_fill =
            env_flag_enabled("DG_SM120_MOE_SKIP_SFA_FILL") &&
            (skip_fill_max_m < 0 || m <= skip_fill_max_m);
        if (!skip_sfa_fill) {
            fill_scale_kernel<<<static_cast<unsigned>((compact_sfa_total + 255) / 256), 256, 0, stream>>>(
                scratch.sfa_ue8, compact_sfa_total);
            DG_CUDA_RUNTIME_CHECK(cudaGetLastError());
        }

        init_compact_grouped_kernel<<<
            std::max(1, (gemm_num_groups + 255) / 256), 256, 0, stream>>>(
            reinterpret_cast<const uint8_t*>(a.data_ptr()), b.data_ptr<int8_t>(),
            reinterpret_cast<ElementD*>(d.data_ptr()), scratch.ptr_a,
            scratch.ptr_b, scratch.ptr_sfa, scratch.ptr_sfb, scratch.ptr_d,
            scratch.stride_a, scratch.stride_b, scratch.stride_d,
            scratch.layout_sfa, scratch.layout_sfb, scratch.problems,
            scratch.sfa_ue8, sfb_ue8.data, scratch.active_count,
            gemm_num_groups, n, k, sfb_layout_m, sfa_group_elems,
            sfb_group_elems, d.stride(0));
        DG_CUDA_RUNTIME_CHECK(cudaGetLastError());

        prepare_compact_grouped_kernel<<<num_groups, 256, 0, stream>>>(
            reinterpret_cast<const uint8_t*>(a.data_ptr()), sfa.data_ptr(),
            b.data_ptr<int8_t>(), reinterpret_cast<ElementD*>(d.data_ptr()),
            expert_starts->data_ptr<int32_t>(), expert_counts->data_ptr<int32_t>(),
            scratch.ptr_a, scratch.ptr_b, scratch.ptr_sfa, scratch.ptr_sfb,
            scratch.ptr_d, scratch.stride_a, scratch.stride_b, scratch.stride_d,
            scratch.layout_sfa, scratch.layout_sfb, scratch.problems,
            scratch.sfa_ue8, sfb_ue8.data, scratch.active_count, num_groups,
            gemm_num_groups, m, n, k, k32, sfa_kind, sfb_layout_m,
            sfa_group_elems, sfb_group_elems, a.stride(0), a.stride(1),
            sfa.stride(0), sfa.stride(1), b.stride(0), d.stride(0),
            d.stride(1));
        DG_CUDA_RUNTIME_CHECK(cudaGetLastError());
    } else {
        fill_scale_kernel<<<static_cast<unsigned>((sfa_total_elems + 255) / 256), 256, 0, stream>>>(
            scratch.sfa_ue8, sfa_total_elems);
        DG_CUDA_RUNTIME_CHECK(cudaGetLastError());

        prepare_grouped_kernel<<<num_groups, 256, 0, stream>>>(
            reinterpret_cast<const uint8_t*>(a.data_ptr()), sfa.data_ptr(),
            b.data_ptr<int8_t>(), reinterpret_cast<ElementD*>(d.data_ptr()),
            grouped_layout.data_ptr<int32_t>(),
            expert_starts == nullptr ? nullptr : expert_starts->data_ptr<int32_t>(),
            expert_counts == nullptr ? nullptr : expert_counts->data_ptr<int32_t>(),
            scratch.ptr_a, scratch.ptr_b,
            scratch.ptr_sfa, scratch.ptr_sfb, scratch.ptr_d, scratch.stride_a,
            scratch.stride_b, scratch.stride_d, scratch.layout_sfa, scratch.layout_sfb,
            scratch.problems, scratch.sfa_ue8, sfb_ue8.data, num_groups, m, n, k,
            k32, sfa_kind, sfb_layout_m, sfa_group_elems, sfb_group_elems,
            a.stride(0), a.stride(1), sfa.stride(0), sfa.stride(1),
            b.stride(0), d.stride(0), d.stride(1));
        DG_CUDA_RUNTIME_CHECK(cudaGetLastError());
    }

    arguments.problem_shape.problem_shapes = scratch.problems;
    arguments.mainloop.ptr_A = scratch.ptr_a;
    arguments.mainloop.dA = scratch.stride_a;
    arguments.mainloop.ptr_B = scratch.ptr_b;
    arguments.mainloop.dB = scratch.stride_b;
    arguments.mainloop.ptr_SFA = scratch.ptr_sfa;
    arguments.mainloop.layout_SFA = scratch.layout_sfa;
    arguments.mainloop.ptr_SFB = scratch.ptr_sfb;
    arguments.mainloop.layout_SFB = scratch.layout_sfb;
    arguments.epilogue.ptr_D = scratch.ptr_d;
    arguments.epilogue.dD = scratch.stride_d;
    arguments.epilogue.thread.alpha = 1.0f;
    arguments.epilogue.thread.beta = 0.0f;

    Gemm gemm_op;
    cutlass::Status status = gemm_op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess)
        return false;
    status = gemm_op.initialize(arguments, scratch.workspace, stream);
    if (status != cutlass::Status::kSuccess)
        return false;
    status = gemm_op.run(stream);
    if (status != cutlass::Status::kSuccess)
        return false;
    DG_CUDA_RUNTIME_CHECK(cudaGetLastError());
    return true;
}

} // namespace

bool sm120_m_grouped_fp8_fp4_gemm_nt_contiguous_cutlass(
    const torch::Tensor& a, const torch::Tensor& sfa,
    const torch::Tensor& b, const torch::Tensor& sfb,
    const torch::Tensor& d, const torch::Tensor& grouped_layout,
    int num_groups, int m, int n, int k, int gran_k_a, int gran_k_b,
    const cute::UMMA::Major& major_b, bool use_psum_layout) {
    return sm120_m_grouped_fp8_fp4_gemm_nt_contiguous_cutlass_impl(
        a, sfa, b, sfb, d, grouped_layout, nullptr, nullptr, num_groups, m, n, k,
        gran_k_a, gran_k_b, major_b, use_psum_layout);
}

bool sm120_m_grouped_fp8_fp4_gemm_nt_contiguous_cutlass_with_starts(
    const torch::Tensor& a, const torch::Tensor& sfa,
    const torch::Tensor& b, const torch::Tensor& sfb,
    const torch::Tensor& d, const torch::Tensor& grouped_layout,
    const torch::Tensor& expert_starts, const torch::Tensor& expert_counts,
    int num_groups, int m, int n, int k, int gran_k_a, int gran_k_b,
    const cute::UMMA::Major& major_b, bool use_psum_layout) {
    return sm120_m_grouped_fp8_fp4_gemm_nt_contiguous_cutlass_impl(
        a, sfa, b, sfb, d, grouped_layout, &expert_starts, &expert_counts,
        num_groups, m, n, k, gran_k_a, gran_k_b, major_b, use_psum_layout);
}

int64_t sm120_fp8_fp4_sfb_layout_numel(int layout_m, int n, int k) {
    DG_HOST_ASSERT(layout_m > 0 && n > 0 && k > 0 && k % 128 == 0 && n % 128 == 0);
    const auto layout = MxScaleConfig::tile_atom_to_shape_SFB(
        make_shape(layout_m, n, k, 1));
    return static_cast<int64_t>(size(filter_zeros(layout)));
}

torch::Tensor sm120_prepack_fp8_fp4_sfb(const torch::Tensor& sfb,
                                        int layout_m, int n, int k) {
    DG_HOST_ASSERT(sfb.is_cuda());
    DG_HOST_ASSERT(layout_m > 0 && n > 0 && k > 0);
    DG_HOST_ASSERT(k % 128 == 0 && n % 128 == 0);
    DG_HOST_ASSERT(sfb.scalar_type() == torch::kFloat32 ||
                   sfb.scalar_type() == torch::kUInt8);
    DG_HOST_ASSERT(sfb.dim() == 3);
    const int num_groups = static_cast<int>(sfb.size(0));
    const int k32 = k / 32;
    DG_HOST_ASSERT(sfb.size(1) == n && sfb.size(2) == k32);

    const size_t group_elems =
        static_cast<size_t>(sm120_fp8_fp4_sfb_layout_numel(layout_m, n, k));
    auto packed = torch::empty(
        {num_groups, static_cast<int64_t>(group_elems)},
        sfb.options().dtype(torch::kUInt8));
    auto* out = reinterpret_cast<ElementSF*>(packed.data_ptr<uint8_t>());
    const auto stream = at::cuda::getCurrentCUDAStream();
    const size_t total_elems = static_cast<size_t>(num_groups) * group_elems;
    fill_scale_kernel<<<static_cast<unsigned>((total_elems + 255) / 256), 256, 0, stream>>>(
        out, total_elems);
    DG_CUDA_RUNTIME_CHECK(cudaGetLastError());

    const int total = num_groups * n * k32;
    if (sfb.scalar_type() == torch::kFloat32) {
        convert_sfb_kernel<<<(total + 255) / 256, 256, 0, stream>>>(
            sfb.data_ptr<float>(), out, total, sfb.stride(0), sfb.stride(1),
            sfb.stride(2), layout_m, n, k, k32, group_elems);
    } else {
        copy_sfb_ue8_kernel<<<(total + 255) / 256, 256, 0, stream>>>(
            sfb.data_ptr<uint8_t>(), out, total, sfb.stride(0), sfb.stride(1),
            sfb.stride(2), layout_m, n, k, k32, group_elems);
    }
    DG_CUDA_RUNTIME_CHECK(cudaGetLastError());
    return packed;
}

} // namespace deep_gemm
