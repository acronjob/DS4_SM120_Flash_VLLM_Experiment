#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <cstdint>
#include <limits>
#include <tuple>
#include <type_traits>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <pybind11/pybind11.h>
#include <torch/python.h>

#include "jit_kernels/impls/sm120_sparse_mla_decode.hpp"
#include "sm120_profile.hpp"
#include "utils/exception.hpp"

namespace deep_gemm {
namespace sm120_mla {
namespace {

constexpr int kHeadDim = 512;
constexpr int kFp8Dim = 448;
constexpr int kBf16Dim = 64;
constexpr int kQuantBlock = 64;
constexpr int kNumQuantBlocks = 7;
constexpr int kTokenDataBytes = kFp8Dim + kBf16Dim * 2;
constexpr int kScaleBytes = 8;
constexpr int kThreads = 256;
constexpr int kWarpSize = 32;
constexpr int kScoreGroupSize = 8;
constexpr int kScoreCandidatesPerBlock = kThreads / kScoreGroupSize;
constexpr int kGroupedScoreGroupSize = 16;
constexpr int kGroupedScoreGroups = kThreads / kGroupedScoreGroupSize;
constexpr int kMaxGroupedCandidateSlots = 768;

__device__ __forceinline__ float decode_ue8m0_scale(uint8_t exponent) {
    if (exponent == 0)
        return 0.0f;
    return exp2f(static_cast<float>(exponent) - 127.0f);
}

__device__ __forceinline__ float fp8_e4m3fn_to_float(uint8_t raw) {
    const uint8_t mag = raw & 0x7fu;
    if (mag == 0)
        return 0.0f;
    const int fp8_exp = static_cast<int>((mag >> 3) & 0x0fu);
    const int mant = static_cast<int>(mag & 0x07u);
    const float value =
        fp8_exp == 0
            ? ldexpf(static_cast<float>(mant), -9)
            : ldexpf(1.0f + static_cast<float>(mant) * 0.125f,
                     fp8_exp - 7);
    return raw & 0x80u ? -value : value;
}

__device__ __forceinline__ const uint8_t* token_ptr_from_linear(
    const uint8_t* cache_flat, int64_t block_stride_bytes, int block_size,
    int64_t linear_index) {
    const int64_t block_id = linear_index / block_size;
    const int block_offset =
        static_cast<int>(linear_index - block_id * block_size);
    const uint8_t* block = cache_flat + block_id * block_stride_bytes;
    return block + static_cast<int64_t>(block_offset) * kTokenDataBytes;
}

__device__ __forceinline__ const uint8_t* scale_ptr_from_linear(
    const uint8_t* cache_flat, int64_t block_stride_bytes, int block_size,
    int64_t linear_index) {
    const int64_t block_id = linear_index / block_size;
    const int block_offset =
        static_cast<int>(linear_index - block_id * block_size);
    const uint8_t* block = cache_flat + block_id * block_stride_bytes;
    return block + static_cast<int64_t>(block_size) * kTokenDataBytes +
           static_cast<int64_t>(block_offset) * kScaleBytes;
}

__device__ __forceinline__ float load_cache_value_from_token(
    const uint8_t* token, const float* scales, int dim) {
    if (dim < kFp8Dim) {
        return fp8_e4m3fn_to_float(token[dim]) * scales[dim / kQuantBlock];
    }

    const auto* rope = reinterpret_cast<const __nv_bfloat16*>(token + kFp8Dim);
    return __bfloat162float(rope[dim - kFp8Dim]);
}

template <typename T>
__device__ __forceinline__ T* align_shared_ptr(unsigned char*& ptr) {
    constexpr uintptr_t alignment = alignof(T);
    uintptr_t value = reinterpret_cast<uintptr_t>(ptr);
    value = (value + alignment - 1) & ~(alignment - 1);
    ptr = reinterpret_cast<unsigned char*>(value);
    return reinterpret_cast<T*>(ptr);
}

__host__ __forceinline__ size_t align_up_size(size_t value, size_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

__host__ __forceinline__ size_t grouped_decode_shared_bytes(
    int candidate_slots) {
    size_t bytes = static_cast<size_t>(kHeadDim) * sizeof(float);
    bytes += static_cast<size_t>(candidate_slots) * sizeof(float);
    bytes = align_up_size(bytes, alignof(int64_t));
    bytes += static_cast<size_t>(candidate_slots) * sizeof(int64_t);
    bytes = align_up_size(bytes, alignof(int));
    bytes += static_cast<size_t>(candidate_slots) * sizeof(int);
    bytes = align_up_size(bytes, alignof(float));
    bytes += static_cast<size_t>(candidate_slots) * kNumQuantBlocks *
             sizeof(float);
    return bytes;
}

__host__ __forceinline__ size_t full_context_decode_shared_bytes(
    int candidate_slots) {
    size_t bytes = static_cast<size_t>(kHeadDim) * sizeof(float);
    bytes += static_cast<size_t>(candidate_slots) * sizeof(float);
    bytes = align_up_size(bytes, alignof(int64_t));
    bytes += static_cast<size_t>(candidate_slots) * sizeof(int64_t);
    bytes = align_up_size(bytes, alignof(float));
    bytes += static_cast<size_t>(candidate_slots) * kNumQuantBlocks *
             sizeof(float);
    return bytes;
}

__host__ __forceinline__ int sm120_active_heads(int num_heads) {
    const char* env = std::getenv("DG_SM120_ACTIVE_HEADS");
    if (env == nullptr || env[0] == '\0')
        return num_heads;
    char* end = nullptr;
    const long value = std::strtol(env, &end, 10);
    if (end == env || value <= 0)
        return num_heads;
    return std::min(num_heads, static_cast<int>(value));
}

__host__ __forceinline__ bool sm120_fast_sparse_mla_enabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("DG_SM120_FAST_SPARSE_MLA");
        if (env == nullptr || env[0] == '\0') return false;
        return env[0] != '0';
    }();
    return enabled;
}

__host__ __forceinline__ bool sm120_score_mma_enabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("DG_SM120_SCORE_MMA");
        if (env == nullptr || env[0] == '\0') return false;
        return env[0] != '0';
    }();
    return enabled;
}

// Debug knob: when set, replaces the m16n8k16 bf16 MMA with a hand-rolled
// scalar matmul that uses the SAME q_smem / kv_smem reads. Lets us bisect
// "is the bug in the MMA fragment layout, or in the smem load?".
__host__ __forceinline__ bool sm120_score_mma_scalar_check() {
    static const bool enabled = []() {
        const char* env = std::getenv("DG_SM120_SCORE_MMA_SCALAR_CHECK");
        if (env == nullptr || env[0] == '\0') return false;
        return env[0] != '0';
    }();
    return enabled;
}

// Split-K SCORE MMA — sibling of SCORE_MMA tuned for SM_120 parallelism on
// the production active_heads=32, topk≤128 shape. The non-split kernel
// produces a (≤4, 2, batch) grid that under-fills 188 SMs; this variant uses
// (slots/8, batch, splitk=4) → ~512 blocks at B=8, slots=128.
__host__ __forceinline__ bool sm120_score_mma_splitk_enabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("DG_SM120_SCORE_MMA_SPLITK");
        if (env == nullptr || env[0] == '\0') return false;
        return env[0] != '0';
    }();
    return enabled;
}

__host__ __forceinline__ bool sm120_score_mma_splitk_scalar_check() {
    static const bool enabled = []() {
        const char* env = std::getenv("DG_SM120_SCORE_MMA_SPLITK_SCALAR_CHECK");
        if (env == nullptr || env[0] == '\0') return false;
        return env[0] != '0';
    }();
    return enabled;
}

// OUTPUT MMA — bf16 mma.sync over the slot reduction for
// out[b,h,d] = gate(b,h) * sum_j scores[b,h,j] * kv[b,j,d]. Mirrors
// SCORE_MMA_SPLITK in spirit but uses M=heads=32, N=8 (one dim slice), K=slots
// (full reduction, no split-K) — at B=8 / kHeadDim=512 the grid is
// (8, 64, 1) = 512 blocks vs 188 SMs (~3 waves) with a single-pass kernel
// (no reduce stage). Validity already encoded in scores via the upstream
// softmax (-inf → 0), so we iterate the full padded slot range.
__host__ __forceinline__ bool sm120_output_mma_enabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("DG_SM120_OUTPUT_MMA");
        if (env == nullptr || env[0] == '\0') return false;
        return env[0] != '0';
    }();
    return enabled;
}

__host__ __forceinline__ bool sm120_output_mma_scalar_check() {
    static const bool enabled = []() {
        const char* env = std::getenv("DG_SM120_OUTPUT_MMA_SCALAR_CHECK");
        if (env == nullptr || env[0] == '\0') return false;
        return env[0] != '0';
    }();
    return enabled;
}

// OUTPUT_FUSED_SOFTMAX — fold the small softmax kernel into the output
// kernel, eliminating one launch + one DRAM round-trip on `scores`.
// Reads RAW scores, computes max/exp/sum/lse in smem, applies inv_sum to
// smem-resident probabilities, then runs the existing sstat-style PV
// loop. lse is written by the same block. Numerics match the
// softmax+sstat sequence at the source level (same op order: divide
// each prob by row_sum BEFORE the kv multiply).
__host__ __forceinline__ bool sm120_output_fused_softmax_enabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("DG_SM120_OUTPUT_FUSED_SOFTMAX");
        if (env == nullptr || env[0] == '\0') return false;
        return env[0] != '0';
    }();
    return enabled;
}

// SCORE_QSTAT_VEC — bf16x4 (uint64) vectorized load variant of the
// score_tiled_qstat kernel. Same block grid, same accumulation breadth, but
// each thread issues 16 LDG.E.U64 instead of 64 LDG.E.U16 over head_dim=512.
// Requires kv_stride_d == 1 and head_dim % 32 == 0; falls back to scalar
// kernel otherwise. Math: float accumulation order changes from
// stride-8-spaced singles to grouped quads (q[d]*kv[d] + q[d+1]*kv[d+1] +
// q[d+2]*kv[d+2] + q[d+3]*kv[d+3] then stride-32 groups), so output is
// bit-similar but not bit-identical to the scalar kernel.
__host__ __forceinline__ bool sm120_score_qstat_vec_enabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("DG_SM120_SCORE_QSTAT_VEC");
        if (env == nullptr || env[0] == '\0') return false;
        return env[0] != '0';
    }();
    return enabled;
}

__host__ __forceinline__ bool sm120_score_qstat_vec_scalar_check() {
    static const bool enabled = []() {
        const char* env = std::getenv("DG_SM120_SCORE_QSTAT_VEC_SCALAR_CHECK");
        if (env == nullptr || env[0] == '\0') return false;
        return env[0] != '0';
    }();
    return enabled;
}

// SCORE_QSTAT_VEC8 — bf16x8 (uint4 / 16-byte) load variant. When set,
// switches the qstat_vec kernel template instantiation from kVecWidth=4
// to kVecWidth=8: half as many LDGs, single 16-byte transaction per
// per-thread-per-iteration K read. Default OFF; ship if microbench beats
// vec4 and e2e A/B confirms.
__host__ __forceinline__ bool sm120_score_qstat_vec8_enabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("DG_SM120_SCORE_QSTAT_VEC8");
        if (env == nullptr || env[0] == '\0') return false;
        return env[0] != '0';
    }();
    return enabled;
}

// OUTPUT_SSTAT_VEC — bf16x4 vec-load variant of
// sparse_mla_workspace_output_sstat_kernel. Each thread accumulates 4
// contiguous head_dim outputs simultaneously, issuing one LDG.E.U64 per j
// in place of 4 single-element bf16 LDGs. Per-d FMA order is preserved
// (each d is an independent score-FMA chain identical to scalar), so math
// is bit-identical to the scalar kernel — no scalar_check flag needed.
// Requires kv_stride_d == 1 and kHeadDim % 4 == 0.
__host__ __forceinline__ bool sm120_output_sstat_vec_enabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("DG_SM120_OUTPUT_SSTAT_VEC");
        if (env == nullptr || env[0] == '\0') return false;
        return env[0] != '0';
    }();
    return enabled;
}

// FLASH_FUSION — full score+softmax+output fusion in a single kernel with
// smem-resident K for the (b, h) tile. Loads K once into ~128 KB of dynamic
// smem (requires cudaFuncSetAttribute(MaxDynamicSmem)), reuses it for both
// the QK score MMA-equivalent FMA loop and the PV output loop. Eliminates
// two kernel launches AND the second DRAM read of K. Numerics are within
// ~1 bf16 ULP of the score+softmax+output 3-kernel chain (different
// reduction order; identical math at op level).
__host__ __forceinline__ bool sm120_flash_fusion_enabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("DG_SM120_FLASH_FUSION");
        if (env == nullptr || env[0] == '\0') return false;
        return env[0] != '0';
    }();
    return enabled;
}

template <typename T>
__device__ __forceinline__ float load_q_value(const T* q, int64_t offset) {
    return static_cast<float>(q[offset]);
}

template <>
__device__ __forceinline__ float load_q_value<__nv_bfloat16>(
    const __nv_bfloat16* q, int64_t offset) {
    return __bfloat162float(q[offset]);
}

template <>
__device__ __forceinline__ float load_q_value<half>(const half* q,
                                                    int64_t offset) {
    return __half2float(q[offset]);
}

template <typename T>
__device__ __forceinline__ void store_out_value(T* out, int64_t offset,
                                                float value) {
    out[offset] = static_cast<T>(value);
}

template <>
__device__ __forceinline__ void store_out_value<__nv_bfloat16>(
    __nv_bfloat16* out, int64_t offset, float value) {
    out[offset] = __float2bfloat16(value);
}

template <>
__device__ __forceinline__ void store_out_value<half>(half* out, int64_t offset,
                                                      float value) {
    out[offset] = __float2half(value);
}

__device__ __forceinline__ float load_cache_value(const uint8_t* cache_flat,
                                                  int64_t block_stride_bytes,
                                                  int block_size,
                                                  int64_t linear_index,
                                                  int dim) {
    const uint8_t* token = token_ptr_from_linear(
        cache_flat, block_stride_bytes, block_size, linear_index);

    if (dim < kFp8Dim) {
        const uint8_t* scales = scale_ptr_from_linear(
            cache_flat, block_stride_bytes, block_size, linear_index);
        return fp8_e4m3fn_to_float(token[dim]) *
               decode_ue8m0_scale(scales[dim / kQuantBlock]);
    }

    const auto* rope = reinterpret_cast<const __nv_bfloat16*>(token + kFp8Dim);
    return __bfloat162float(rope[dim - kFp8Dim]);
}

template <typename T>
__device__ __forceinline__ float load_workspace_value(const T* kv,
                                                      int64_t offset) {
    return static_cast<float>(kv[offset]);
}

template <>
__device__ __forceinline__ float load_workspace_value<__nv_bfloat16>(
    const __nv_bfloat16* kv, int64_t offset) {
    return __bfloat162float(kv[offset]);
}

template <>
__device__ __forceinline__ float load_workspace_value<half>(
    const half* kv, int64_t offset) {
    return __half2float(kv[offset]);
}

template <typename index_t>
__device__ __forceinline__ int64_t load_index(const void* ptr, int64_t offset) {
    return static_cast<int64_t>(static_cast<const index_t*>(ptr)[offset]);
}

__device__ __forceinline__ int load_length_value(const void* ptr, int kind,
                                                 int64_t offset,
                                                 int default_value) {
    if (ptr == nullptr)
        return default_value;
    if (kind == 0)
        return static_cast<int>(static_cast<const int32_t*>(ptr)[offset]);
    return static_cast<int>(static_cast<const int64_t*>(ptr)[offset]);
}

template <typename out_t, typename seq_t, typename block_t, typename gather_t>
__global__ void dequantize_gather_k_cache_kernel(
    out_t* __restrict__ out, const uint8_t* __restrict__ k_cache,
    const seq_t* __restrict__ seq_lens,
    const gather_t* __restrict__ gather_lens,
    const block_t* __restrict__ block_table, int num_reqs, int max_out_rows,
    int head_dim, int block_size, int offset, int64_t cache_blocks,
    int64_t cache_stride0_bytes, int64_t out_stride_b, int64_t out_stride_s,
    int64_t out_stride_d, int64_t block_stride_b, int64_t block_stride_s) {
    const int dim = blockIdx.x * blockDim.x + threadIdx.x;
    const int out_idx = blockIdx.y;
    const int batch_idx = blockIdx.z;
    if (batch_idx >= num_reqs || out_idx >= max_out_rows || dim >= head_dim)
        return;

    const int seq_len = max(0, static_cast<int>(seq_lens[batch_idx]));
    int gather_len = gather_lens == nullptr
                         ? seq_len
                         : max(0, static_cast<int>(gather_lens[batch_idx]));
    gather_len = min(gather_len, max_out_rows);

    float value = 0.0f;
    if (out_idx < gather_len) {
        const int start_pos = max(0, seq_len - gather_len);
        const int pos = start_pos + out_idx;
        const int block_in_seq = pos / block_size;
        const int pos_in_block = pos - block_in_seq * block_size;
        const int64_t physical_block =
            static_cast<int64_t>(block_table[static_cast<int64_t>(batch_idx) *
                                             block_stride_b +
                                             static_cast<int64_t>(block_in_seq) *
                                             block_stride_s]);
        if (physical_block >= 0 && physical_block < cache_blocks &&
            pos_in_block >= 0 && pos_in_block < block_size) {
            const int64_t linear_index =
                physical_block * static_cast<int64_t>(block_size) + pos_in_block;
            value = load_cache_value(k_cache, cache_stride0_bytes, block_size,
                                     linear_index, dim);
        }
    }

    store_out_value<out_t>(
        out, static_cast<int64_t>(batch_idx) * out_stride_b +
                 static_cast<int64_t>(offset + out_idx) * out_stride_s +
                 static_cast<int64_t>(dim) * out_stride_d,
        value);
}

template <typename out_t, typename index_t, typename len_t>
__global__ void dequantize_gather_indexed_k_cache_kernel(
    out_t* __restrict__ out, const uint8_t* __restrict__ k_cache,
    const void* __restrict__ indices, const len_t* __restrict__ topk_length,
    int batch_size, int topk, int head_dim, int block_size, int offset,
    int64_t cache_blocks, int64_t cache_stride0_bytes, int64_t out_stride_b,
    int64_t out_stride_s, int64_t out_stride_d, int64_t index_stride_b,
    int64_t index_stride_s, int64_t index_stride_k, bool indices_3d) {
    const int dim = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y;
    const int b = blockIdx.z;
    if (b >= batch_size || j >= topk || dim >= head_dim)
        return;

    const int limit =
        topk_length == nullptr ? topk : max(0, min(topk, static_cast<int>(topk_length[b])));
    float value = 0.0f;
    if (j < limit) {
        const int64_t index_offset =
            static_cast<int64_t>(b) * index_stride_b +
            (indices_3d ? static_cast<int64_t>(0) * index_stride_s : 0) +
            static_cast<int64_t>(j) * index_stride_k;
        const int64_t linear = load_index<index_t>(indices, index_offset);
        if (linear >= 0 && linear < cache_blocks * static_cast<int64_t>(block_size)) {
            value = load_cache_value(k_cache, cache_stride0_bytes, block_size,
                                     linear, dim);
        }
    }

    store_out_value<out_t>(
        out, static_cast<int64_t>(b) * out_stride_b +
                 static_cast<int64_t>(offset + j) * out_stride_s +
                 static_cast<int64_t>(dim) * out_stride_d,
        value);
}

__global__ void fill_decode_all_indices_kernel(
    int32_t* __restrict__ out, const int32_t* __restrict__ seq_lens,
    int num_rows, int next_n, int topk, int64_t out_stride0,
    int64_t out_stride1, bool seq_lens_is_2d) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y;
    if (row >= num_rows || col >= topk)
        return;

    const int batch_idx = row / next_n;
    const int next_idx = row - batch_idx * next_n;
    const int seq_len =
        seq_lens_is_2d
            ? seq_lens[row]
            : seq_lens[batch_idx] - next_n + next_idx + 1;
    const int row_end = max(0, seq_len);
    out[static_cast<int64_t>(row) * out_stride0 +
        static_cast<int64_t>(col) * out_stride1] =
        col < row_end ? col : -1;
}

template <typename q_t, typename out_t, typename index_t>
__global__ void sparse_mla_decode_kernel(
    const q_t* __restrict__ q, const uint8_t* __restrict__ k_cache,
    const void* __restrict__ indices, const int64_t* __restrict__ topk_length,
    const float* __restrict__ attn_sink,
    const uint8_t* __restrict__ extra_k_cache,
    const void* __restrict__ extra_indices,
    const int64_t* __restrict__ extra_topk_length, out_t* __restrict__ out,
    float* __restrict__ lse, int batch_size, int active_heads, int num_heads,
    int block_size, int64_t cache_blocks, int64_t extra_cache_blocks,
    int main_topk, int extra_topk, int64_t q_stride_b, int64_t q_stride_s,
    int64_t q_stride_h, int64_t q_stride_d,
    int64_t out_stride_b, int64_t out_stride_s, int64_t out_stride_h,
    int64_t out_stride_d, int64_t index_stride_b, int64_t index_stride_s,
    int64_t index_stride_k, int64_t extra_index_stride_b,
    int64_t extra_index_stride_s, int64_t extra_index_stride_k,
    int64_t cache_stride0_bytes, int64_t extra_cache_stride0_bytes,
    float softmax_scale) {
    extern __shared__ float scores[];
    const int bh = blockIdx.x;
    const int b = bh / active_heads;
    const int h = bh - b * active_heads;
    if (b >= batch_size)
        return;

    const int main_limit = topk_length == nullptr
                               ? main_topk
                               : max(0, min(main_topk, static_cast<int>(topk_length[b])));
    const int extra_limit =
        (extra_k_cache == nullptr || extra_indices == nullptr)
            ? 0
            : (extra_topk_length == nullptr
                   ? extra_topk
                   : max(0, min(extra_topk,
                                static_cast<int>(extra_topk_length[b]))));
    const int candidate_count = main_limit + extra_limit;
    const int64_t max_main_linear = cache_blocks * block_size;

    float local_max = -INFINITY;
    for (int j = threadIdx.x; j < candidate_count; j += blockDim.x) {
        const bool is_extra = j >= main_limit;
        const int local_j = is_extra ? j - main_limit : j;
        const void* index_ptr = is_extra ? extra_indices : indices;
        const int64_t index_offset =
            is_extra ? (static_cast<int64_t>(b) * extra_index_stride_b +
                        static_cast<int64_t>(local_j) * extra_index_stride_k)
                     : (static_cast<int64_t>(b) * index_stride_b +
                        static_cast<int64_t>(local_j) * index_stride_k);
        const int64_t linear = load_index<index_t>(index_ptr, index_offset);
        const int64_t max_linear =
            is_extra ? extra_cache_blocks * block_size : max_main_linear;
        float score = -INFINITY;
        if (linear >= 0 && linear < max_linear) {
            const uint8_t* cache_ptr = is_extra ? extra_k_cache : k_cache;
            const int64_t cache_stride =
                is_extra ? extra_cache_stride0_bytes : cache_stride0_bytes;
            float partial = 0.0f;
            for (int d = 0; d < kHeadDim; ++d) {
                const float qv = load_q_value<q_t>(
                    q, static_cast<int64_t>(b) * q_stride_b +
                           static_cast<int64_t>(h) * q_stride_h +
                           static_cast<int64_t>(d) * q_stride_d);
                partial += qv *
                           load_cache_value(cache_ptr, cache_stride, block_size,
                                            linear, d);
            }
            score = partial * softmax_scale;
        }
        scores[j] = score;
        local_max = fmaxf(local_max, score);
    }

    __shared__ float max_buf[kThreads];
    max_buf[threadIdx.x] = local_max;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset)
            max_buf[threadIdx.x] = fmaxf(max_buf[threadIdx.x],
                                         max_buf[threadIdx.x + offset]);
        __syncthreads();
    }
    const float row_max = max_buf[0];

    float local_sum = 0.0f;
    for (int j = threadIdx.x; j < candidate_count; j += blockDim.x) {
        const float p = isfinite(row_max) ? expf(scores[j] - row_max) : 0.0f;
        scores[j] = p;
        local_sum += p;
    }
    __shared__ float sum_buf[kThreads];
    sum_buf[threadIdx.x] = local_sum;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset)
            sum_buf[threadIdx.x] += sum_buf[threadIdx.x + offset];
        __syncthreads();
    }
    const float row_sum = sum_buf[0];
    const float row_lse = row_sum > 0.0f ? logf(row_sum) + row_max : -INFINITY;

    const float sink =
        attn_sink == nullptr ? 0.0f : attn_sink[h];
    const float sink_gate =
        attn_sink == nullptr ? 1.0f : 1.0f / (1.0f + expf(-(row_lse - sink)));

    for (int d = threadIdx.x; d < kHeadDim; d += blockDim.x) {
        float accum = 0.0f;
        if (row_sum > 0.0f) {
            for (int j = 0; j < candidate_count; ++j) {
                const float p = scores[j] / row_sum;
                const bool is_extra = j >= main_limit;
                const int local_j = is_extra ? j - main_limit : j;
                const void* index_ptr = is_extra ? extra_indices : indices;
                const int64_t index_offset =
                    is_extra ? (static_cast<int64_t>(b) * extra_index_stride_b +
                                static_cast<int64_t>(local_j) *
                                    extra_index_stride_k)
                             : (static_cast<int64_t>(b) * index_stride_b +
                                static_cast<int64_t>(local_j) * index_stride_k);
                const int64_t linear = load_index<index_t>(index_ptr, index_offset);
                const int64_t max_linear =
                    is_extra ? extra_cache_blocks * block_size : max_main_linear;
                if (linear < 0 || linear >= max_linear)
                    continue;
                const uint8_t* cache_ptr = is_extra ? extra_k_cache : k_cache;
                const int64_t cache_stride =
                    is_extra ? extra_cache_stride0_bytes : cache_stride0_bytes;
                accum += p * load_cache_value(cache_ptr, cache_stride, block_size,
                                              linear, d);
            }
        }
        store_out_value<out_t>(
            out, static_cast<int64_t>(b) * out_stride_b +
                     static_cast<int64_t>(h) * out_stride_h +
                     static_cast<int64_t>(d) * out_stride_d,
            accum * sink_gate);
    }

    if (threadIdx.x == 0)
        lse[static_cast<int64_t>(b) * num_heads + h] = row_lse;
}

template <typename q_t, typename out_t, typename index_t>
__global__ void sparse_mla_decode_grouped_kernel(
    const q_t* __restrict__ q, const uint8_t* __restrict__ k_cache,
    const void* __restrict__ indices, const int64_t* __restrict__ topk_length,
    const float* __restrict__ attn_sink,
    const uint8_t* __restrict__ extra_k_cache,
    const void* __restrict__ extra_indices,
    const int64_t* __restrict__ extra_topk_length, out_t* __restrict__ out,
    float* __restrict__ lse, int batch_size, int active_heads, int num_heads,
    int block_size, int64_t cache_blocks, int64_t extra_cache_blocks,
    int main_topk, int extra_topk, int candidate_slots, int64_t q_stride_b,
    int64_t q_stride_s, int64_t q_stride_h, int64_t q_stride_d,
    int64_t out_stride_b, int64_t out_stride_s, int64_t out_stride_h,
    int64_t out_stride_d, int64_t index_stride_b, int64_t index_stride_s,
    int64_t index_stride_k, int64_t extra_index_stride_b,
    int64_t extra_index_stride_s, int64_t extra_index_stride_k,
    int64_t cache_stride0_bytes, int64_t extra_cache_stride0_bytes,
    float softmax_scale) {
    extern __shared__ unsigned char smem[];
    unsigned char* cursor = smem;
    float* q_s = reinterpret_cast<float*>(cursor);
    cursor += static_cast<size_t>(kHeadDim) * sizeof(float);
    float* scores = reinterpret_cast<float*>(cursor);
    cursor += static_cast<size_t>(candidate_slots) * sizeof(float);
    int64_t* linear_s = align_shared_ptr<int64_t>(cursor);
    cursor += static_cast<size_t>(candidate_slots) * sizeof(int64_t);
    int* source_s = align_shared_ptr<int>(cursor);
    cursor += static_cast<size_t>(candidate_slots) * sizeof(int);
    float* scale_s = align_shared_ptr<float>(cursor);

    const int bh = blockIdx.x;
    const int b = bh / active_heads;
    const int h = bh - b * active_heads;
    if (b >= batch_size)
        return;

    for (int d = threadIdx.x; d < kHeadDim; d += blockDim.x) {
        q_s[d] = load_q_value<q_t>(
            q, static_cast<int64_t>(b) * q_stride_b +
                   static_cast<int64_t>(h) * q_stride_h +
                   static_cast<int64_t>(d) * q_stride_d);
    }

    const int main_limit = topk_length == nullptr
                               ? main_topk
                               : max(0, min(main_topk, static_cast<int>(topk_length[b])));
    const int extra_limit =
        (extra_k_cache == nullptr || extra_indices == nullptr)
            ? 0
            : (extra_topk_length == nullptr
                   ? extra_topk
                   : max(0, min(extra_topk,
                                static_cast<int>(extra_topk_length[b]))));
    const int candidate_count = main_limit + extra_limit;
    const int64_t max_main_linear = cache_blocks * block_size;

    for (int j = threadIdx.x; j < candidate_count; j += blockDim.x) {
        const bool is_extra = j >= main_limit;
        const int local_j = is_extra ? j - main_limit : j;
        const void* index_ptr = is_extra ? extra_indices : indices;
        const int64_t index_offset =
            is_extra ? (static_cast<int64_t>(b) * extra_index_stride_b +
                        static_cast<int64_t>(local_j) * extra_index_stride_k)
                     : (static_cast<int64_t>(b) * index_stride_b +
                        static_cast<int64_t>(local_j) * index_stride_k);
        const int64_t linear = load_index<index_t>(index_ptr, index_offset);
        const int64_t max_linear =
            is_extra ? extra_cache_blocks * block_size : max_main_linear;
        const bool valid = linear >= 0 && linear < max_linear;
        linear_s[j] = valid ? linear : -1;
        source_s[j] = is_extra ? 1 : 0;

        float* cand_scales = scale_s + static_cast<int64_t>(j) * kNumQuantBlocks;
        #pragma unroll
        for (int s = 0; s < kNumQuantBlocks; ++s)
            cand_scales[s] = 0.0f;
        if (valid) {
            const uint8_t* cache_ptr = is_extra ? extra_k_cache : k_cache;
            const int64_t cache_stride =
                is_extra ? extra_cache_stride0_bytes : cache_stride0_bytes;
            const uint8_t* scales = scale_ptr_from_linear(
                cache_ptr, cache_stride, block_size, linear);
            #pragma unroll
            for (int s = 0; s < kNumQuantBlocks; ++s)
                cand_scales[s] = decode_ue8m0_scale(scales[s]);
        }
    }
    __syncthreads();

    const int group_id = threadIdx.x / kGroupedScoreGroupSize;
    const int lane = threadIdx.x - group_id * kGroupedScoreGroupSize;
    for (int base = 0; base < candidate_count; base += kGroupedScoreGroups) {
        const int j = base + group_id;
        float partial = 0.0f;
        bool valid = false;
        if (j < candidate_count && linear_s[j] >= 0) {
            valid = true;
            const bool is_extra = source_s[j] != 0;
            const uint8_t* cache_ptr = is_extra ? extra_k_cache : k_cache;
            const int64_t cache_stride =
                is_extra ? extra_cache_stride0_bytes : cache_stride0_bytes;
            const uint8_t* token = token_ptr_from_linear(
                cache_ptr, cache_stride, block_size, linear_s[j]);
            const float* cand_scales =
                scale_s + static_cast<int64_t>(j) * kNumQuantBlocks;
            for (int d = lane; d < kHeadDim; d += kGroupedScoreGroupSize) {
                partial += q_s[d] *
                           load_cache_value_from_token(token, cand_scales, d);
            }
        }

        for (int offset = kGroupedScoreGroupSize / 2; offset > 0; offset >>= 1)
            partial += __shfl_down_sync(0xffffffffu, partial, offset,
                                        kGroupedScoreGroupSize);
        if (lane == 0 && j < candidate_count)
            scores[j] = valid ? partial * softmax_scale : -INFINITY;
    }
    __syncthreads();

    __shared__ float reduce_buf[kThreads];
    float local_max = -INFINITY;
    for (int j = threadIdx.x; j < candidate_count; j += blockDim.x)
        local_max = fmaxf(local_max, scores[j]);
    reduce_buf[threadIdx.x] = local_max;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset)
            reduce_buf[threadIdx.x] =
                fmaxf(reduce_buf[threadIdx.x], reduce_buf[threadIdx.x + offset]);
        __syncthreads();
    }
    const float row_max = reduce_buf[0];

    float local_sum = 0.0f;
    for (int j = threadIdx.x; j < candidate_count; j += blockDim.x) {
        const float p = isfinite(row_max) ? expf(scores[j] - row_max) : 0.0f;
        scores[j] = p;
        local_sum += p;
    }
    reduce_buf[threadIdx.x] = local_sum;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset)
            reduce_buf[threadIdx.x] += reduce_buf[threadIdx.x + offset];
        __syncthreads();
    }
    const float row_sum = reduce_buf[0];
    const float row_lse = row_sum > 0.0f ? logf(row_sum) + row_max : -INFINITY;
    for (int j = threadIdx.x; j < candidate_count; j += blockDim.x)
        scores[j] = row_sum > 0.0f ? scores[j] / row_sum : 0.0f;
    __syncthreads();

    const float sink = attn_sink == nullptr ? 0.0f : attn_sink[h];
    const float sink_gate =
        attn_sink == nullptr ? 1.0f : 1.0f / (1.0f + expf(-(row_lse - sink)));

    for (int d = threadIdx.x; d < kHeadDim; d += blockDim.x) {
        float accum = 0.0f;
        if (row_sum > 0.0f) {
            for (int j = 0; j < candidate_count; ++j) {
                const float p = scores[j];
                if (p == 0.0f || linear_s[j] < 0)
                    continue;
                const bool is_extra = source_s[j] != 0;
                const uint8_t* cache_ptr = is_extra ? extra_k_cache : k_cache;
                const int64_t cache_stride =
                    is_extra ? extra_cache_stride0_bytes : cache_stride0_bytes;
                const uint8_t* token = token_ptr_from_linear(
                    cache_ptr, cache_stride, block_size, linear_s[j]);
                const float* cand_scales =
                    scale_s + static_cast<int64_t>(j) * kNumQuantBlocks;
                accum += p * load_cache_value_from_token(token, cand_scales, d);
            }
        }
        store_out_value<out_t>(
            out, static_cast<int64_t>(b) * out_stride_b +
                     static_cast<int64_t>(h) * out_stride_h +
                     static_cast<int64_t>(d) * out_stride_d,
            accum * sink_gate);
    }

    if (threadIdx.x == 0)
        lse[static_cast<int64_t>(b) * num_heads + h] = row_lse;
}

template <typename q_t, typename kv_t, typename out_t>
__global__ void sparse_mla_decode_workspace_kernel(
    const q_t* __restrict__ q, const kv_t* __restrict__ kv_workspace,
    const void* __restrict__ topk_length,
    const void* __restrict__ extra_topk_length,
    const float* __restrict__ attn_sink, out_t* __restrict__ out,
    float* __restrict__ lse, int batch_size, int active_heads, int num_heads,
    int main_topk, int extra_topk, int candidate_slots, int64_t q_stride_b,
    int64_t q_stride_s, int64_t q_stride_h, int64_t q_stride_d,
    int64_t kv_stride_b, int64_t kv_stride_s, int64_t kv_stride_d,
    int64_t out_stride_b, int64_t out_stride_s, int64_t out_stride_h,
    int64_t out_stride_d, int topk_length_kind, int extra_topk_length_kind,
    float softmax_scale) {
    extern __shared__ unsigned char smem[];
    auto* q_s = reinterpret_cast<float*>(smem);
    auto* scores = reinterpret_cast<float*>(
        smem + static_cast<size_t>(kHeadDim) * sizeof(float));

    const int bh = blockIdx.x;
    const int b = bh / num_heads;
    const int h = bh - b * num_heads;
    if (b >= batch_size)
        return;

    if (h >= active_heads) {
        for (int d = threadIdx.x; d < kHeadDim; d += blockDim.x) {
            store_out_value<out_t>(
                out, static_cast<int64_t>(b) * out_stride_b +
                         static_cast<int64_t>(h) * out_stride_h +
                         static_cast<int64_t>(d) * out_stride_d,
                0.0f);
        }
        if (threadIdx.x == 0)
            lse[static_cast<int64_t>(b) * num_heads + h] = -INFINITY;
        return;
    }

    for (int d = threadIdx.x; d < kHeadDim; d += blockDim.x) {
        q_s[d] = load_q_value<q_t>(
            q, static_cast<int64_t>(b) * q_stride_b +
                   static_cast<int64_t>(0) * q_stride_s +
                   static_cast<int64_t>(h) * q_stride_h +
                   static_cast<int64_t>(d) * q_stride_d);
    }
    __syncthreads();

    const int main_limit = max(
        0, min(main_topk,
               load_length_value(topk_length, topk_length_kind, b, main_topk)));
    const int extra_limit =
        max(0, min(extra_topk,
                   load_length_value(extra_topk_length, extra_topk_length_kind,
                                     b, extra_topk)));
    const int valid_count = main_limit + extra_limit;

    const int group_id = threadIdx.x / kGroupedScoreGroupSize;
    const int lane = threadIdx.x - group_id * kGroupedScoreGroupSize;
    for (int base = 0; base < candidate_slots; base += kGroupedScoreGroups) {
        const int j = base + group_id;
        float partial = 0.0f;
        const bool valid =
            j < main_limit ||
            (j >= main_topk && j < main_topk + extra_limit);
        if (j < candidate_slots && valid) {
            for (int d = lane; d < kHeadDim; d += kGroupedScoreGroupSize) {
                partial += q_s[d] *
                           load_workspace_value<kv_t>(
                               kv_workspace,
                               static_cast<int64_t>(b) * kv_stride_b +
                                   static_cast<int64_t>(j) * kv_stride_s +
                                   static_cast<int64_t>(d) * kv_stride_d);
            }
        }

        for (int offset = kGroupedScoreGroupSize / 2; offset > 0; offset >>= 1)
            partial += __shfl_down_sync(0xffffffffu, partial, offset,
                                        kGroupedScoreGroupSize);
        if (lane == 0 && j < candidate_slots)
            scores[j] = valid ? partial * softmax_scale : -INFINITY;
    }
    __syncthreads();

    __shared__ float reduce_buf[kThreads];
    float local_max = -INFINITY;
    for (int j = threadIdx.x; j < candidate_slots; j += blockDim.x)
        local_max = fmaxf(local_max, scores[j]);
    reduce_buf[threadIdx.x] = local_max;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset)
            reduce_buf[threadIdx.x] =
                fmaxf(reduce_buf[threadIdx.x], reduce_buf[threadIdx.x + offset]);
        __syncthreads();
    }
    const float row_max = reduce_buf[0];

    float local_sum = 0.0f;
    for (int j = threadIdx.x; j < candidate_slots; j += blockDim.x) {
        const float p = isfinite(row_max) ? expf(scores[j] - row_max) : 0.0f;
        scores[j] = p;
        local_sum += p;
    }
    reduce_buf[threadIdx.x] = local_sum;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset)
            reduce_buf[threadIdx.x] += reduce_buf[threadIdx.x + offset];
        __syncthreads();
    }
    const float row_sum = reduce_buf[0];
    const float row_lse = row_sum > 0.0f ? logf(row_sum) + row_max : -INFINITY;
    for (int j = threadIdx.x; j < candidate_slots; j += blockDim.x)
        scores[j] = row_sum > 0.0f ? scores[j] / row_sum : 0.0f;
    __syncthreads();

    const float sink = attn_sink == nullptr ? 0.0f : attn_sink[h];
    const float sink_gate =
        attn_sink == nullptr ? 1.0f : 1.0f / (1.0f + expf(-(row_lse - sink)));

    for (int d = threadIdx.x; d < kHeadDim; d += blockDim.x) {
        float accum = 0.0f;
        if (valid_count > 0 && row_sum > 0.0f) {
            for (int j = 0; j < main_limit; ++j) {
                accum += scores[j] *
                         load_workspace_value<kv_t>(
                             kv_workspace,
                             static_cast<int64_t>(b) * kv_stride_b +
                                 static_cast<int64_t>(j) * kv_stride_s +
                                 static_cast<int64_t>(d) * kv_stride_d);
            }
            for (int j = 0; j < extra_limit; ++j) {
                const int workspace_j = main_topk + j;
                accum += scores[workspace_j] *
                         load_workspace_value<kv_t>(
                             kv_workspace,
                             static_cast<int64_t>(b) * kv_stride_b +
                                 static_cast<int64_t>(workspace_j) *
                                     kv_stride_s +
                                 static_cast<int64_t>(d) * kv_stride_d);
            }
        }
        store_out_value<out_t>(
            out, static_cast<int64_t>(b) * out_stride_b +
                     static_cast<int64_t>(0) * out_stride_s +
                     static_cast<int64_t>(h) * out_stride_h +
                     static_cast<int64_t>(d) * out_stride_d,
            accum * sink_gate);
    }

    if (threadIdx.x == 0)
        lse[static_cast<int64_t>(b) * num_heads + h] = row_lse;
}

template <typename q_t, typename out_t, typename block_t, typename seq_t,
          typename req_t>
__global__ void sparse_mla_decode_full_context_kernel(
    const q_t* __restrict__ q, const uint8_t* __restrict__ k_cache,
    const block_t* __restrict__ block_table,
    const seq_t* __restrict__ seq_lens,
    const req_t* __restrict__ req_id_per_token,
    const float* __restrict__ attn_sink, out_t* __restrict__ out,
    float* __restrict__ lse, int batch_size, int num_heads, int block_size,
    int64_t cache_blocks, int block_table_rows, int block_table_width,
    int candidate_slots, int64_t q_stride_b, int64_t q_stride_s,
    int64_t q_stride_h, int64_t q_stride_d, int64_t out_stride_b,
    int64_t out_stride_s, int64_t out_stride_h, int64_t out_stride_d,
    int64_t block_stride_b, int64_t block_stride_s, int64_t seq_lens_stride,
    int64_t req_stride, int64_t cache_stride0_bytes, float softmax_scale) {
    extern __shared__ unsigned char smem[];
    unsigned char* cursor = smem;
    float* q_s = reinterpret_cast<float*>(cursor);
    cursor += static_cast<size_t>(kHeadDim) * sizeof(float);
    float* scores = reinterpret_cast<float*>(cursor);
    cursor += static_cast<size_t>(candidate_slots) * sizeof(float);
    int64_t* linear_s = align_shared_ptr<int64_t>(cursor);
    cursor += static_cast<size_t>(candidate_slots) * sizeof(int64_t);
    float* scale_s = align_shared_ptr<float>(cursor);

    const int bh = blockIdx.x;
    const int b = bh / num_heads;
    const int h = bh - b * num_heads;
    if (b >= batch_size)
        return;

    for (int d = threadIdx.x; d < kHeadDim; d += blockDim.x) {
        q_s[d] = load_q_value<q_t>(
            q, static_cast<int64_t>(b) * q_stride_b +
                   static_cast<int64_t>(h) * q_stride_h +
                   static_cast<int64_t>(d) * q_stride_d);
    }

    const int req = static_cast<int>(
        req_id_per_token[static_cast<int64_t>(b) * req_stride]);
    const bool req_valid = req >= 0 && req < block_table_rows;
    const int unclamped_seq_len =
        req_valid ? static_cast<int>(
                        seq_lens[static_cast<int64_t>(req) * seq_lens_stride])
                  : 0;
    const int seq_len = max(0, min(candidate_slots, unclamped_seq_len));
    const int64_t max_linear = cache_blocks * static_cast<int64_t>(block_size);

    for (int j = threadIdx.x; j < candidate_slots; j += blockDim.x) {
        int64_t linear = -1;
        if (j < seq_len) {
            const int block_in_seq = j / block_size;
            const int block_offset = j - block_in_seq * block_size;
            if (block_in_seq >= 0 && block_in_seq < block_table_width) {
                const int64_t physical_block = static_cast<int64_t>(
                    block_table[static_cast<int64_t>(req) * block_stride_b +
                                static_cast<int64_t>(block_in_seq) *
                                    block_stride_s]);
                const int64_t candidate_linear =
                    physical_block * static_cast<int64_t>(block_size) +
                    block_offset;
                if (physical_block >= 0 && candidate_linear >= 0 &&
                    candidate_linear < max_linear) {
                    linear = candidate_linear;
                }
            }
        }
        linear_s[j] = linear;

        float* cand_scales = scale_s + static_cast<int64_t>(j) * kNumQuantBlocks;
        #pragma unroll
        for (int s = 0; s < kNumQuantBlocks; ++s)
            cand_scales[s] = 0.0f;
        if (linear >= 0) {
            const uint8_t* scales = scale_ptr_from_linear(
                k_cache, cache_stride0_bytes, block_size, linear);
            #pragma unroll
            for (int s = 0; s < kNumQuantBlocks; ++s)
                cand_scales[s] = decode_ue8m0_scale(scales[s]);
        }
    }
    __syncthreads();

    const int group_id = threadIdx.x / kGroupedScoreGroupSize;
    const int lane = threadIdx.x - group_id * kGroupedScoreGroupSize;
    for (int base = 0; base < candidate_slots; base += kGroupedScoreGroups) {
        const int j = base + group_id;
        float partial = 0.0f;
        bool valid = false;
        if (j < candidate_slots && linear_s[j] >= 0) {
            valid = true;
            const uint8_t* token = token_ptr_from_linear(
                k_cache, cache_stride0_bytes, block_size, linear_s[j]);
            const float* cand_scales =
                scale_s + static_cast<int64_t>(j) * kNumQuantBlocks;
            for (int d = lane; d < kHeadDim; d += kGroupedScoreGroupSize) {
                partial += q_s[d] *
                           load_cache_value_from_token(token, cand_scales, d);
            }
        }

        for (int offset = kGroupedScoreGroupSize / 2; offset > 0; offset >>= 1)
            partial += __shfl_down_sync(0xffffffffu, partial, offset,
                                        kGroupedScoreGroupSize);
        if (lane == 0 && j < candidate_slots)
            scores[j] = valid ? partial * softmax_scale : -INFINITY;
    }
    __syncthreads();

    __shared__ float reduce_buf[kThreads];
    float local_max = -INFINITY;
    for (int j = threadIdx.x; j < candidate_slots; j += blockDim.x)
        local_max = fmaxf(local_max, scores[j]);
    reduce_buf[threadIdx.x] = local_max;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset)
            reduce_buf[threadIdx.x] =
                fmaxf(reduce_buf[threadIdx.x], reduce_buf[threadIdx.x + offset]);
        __syncthreads();
    }
    const float row_max = reduce_buf[0];

    float local_sum = 0.0f;
    for (int j = threadIdx.x; j < candidate_slots; j += blockDim.x) {
        const float p = isfinite(row_max) ? expf(scores[j] - row_max) : 0.0f;
        scores[j] = p;
        local_sum += p;
    }
    reduce_buf[threadIdx.x] = local_sum;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset)
            reduce_buf[threadIdx.x] += reduce_buf[threadIdx.x + offset];
        __syncthreads();
    }
    const float row_sum = reduce_buf[0];
    const float row_lse = row_sum > 0.0f ? logf(row_sum) + row_max : -INFINITY;
    for (int j = threadIdx.x; j < candidate_slots; j += blockDim.x)
        scores[j] = row_sum > 0.0f ? scores[j] / row_sum : 0.0f;
    __syncthreads();

    const float sink = attn_sink == nullptr ? 0.0f : attn_sink[h];
    const float sink_gate =
        attn_sink == nullptr ? 1.0f : 1.0f / (1.0f + expf(-(row_lse - sink)));

    for (int d = threadIdx.x; d < kHeadDim; d += blockDim.x) {
        float accum = 0.0f;
        if (row_sum > 0.0f) {
            for (int j = 0; j < candidate_slots; ++j) {
                const float p = scores[j];
                if (p == 0.0f || linear_s[j] < 0)
                    continue;
                const uint8_t* token = token_ptr_from_linear(
                    k_cache, cache_stride0_bytes, block_size, linear_s[j]);
                const float* cand_scales =
                    scale_s + static_cast<int64_t>(j) * kNumQuantBlocks;
                accum += p * load_cache_value_from_token(token, cand_scales, d);
            }
        }
        store_out_value<out_t>(
            out, static_cast<int64_t>(b) * out_stride_b +
                     static_cast<int64_t>(h) * out_stride_h +
                     static_cast<int64_t>(d) * out_stride_d,
            accum * sink_gate);
    }

    if (threadIdx.x == 0)
        lse[static_cast<int64_t>(b) * num_heads + h] = row_lse;
}

template <typename q_t, typename index_t>
__global__ void sparse_mla_score_kernel(
    const q_t* __restrict__ q, const uint8_t* __restrict__ k_cache,
    const void* __restrict__ indices, const int64_t* __restrict__ topk_length,
    float* __restrict__ scores, int batch_size, int num_heads, int block_size,
    int64_t cache_blocks, int topk, int64_t q_stride_b, int64_t q_stride_h,
    int64_t q_stride_d, int64_t index_stride_b, int64_t index_stride_k,
    int64_t cache_stride0_bytes, float softmax_scale) {
    __shared__ float reductions[kThreads];
    const int j = blockIdx.x;
    const int h = blockIdx.y;
    const int b = blockIdx.z;
    if (b >= batch_size || h >= num_heads || j >= topk)
        return;

    const int limit =
        topk_length == nullptr ? topk
                               : max(0, min(topk, static_cast<int>(topk_length[b])));
    const int64_t score_offset =
        (static_cast<int64_t>(b) * num_heads + h) * topk + j;
    if (j >= limit) {
        if (threadIdx.x == 0)
            scores[score_offset] = -INFINITY;
        return;
    }

    const int64_t linear = load_index<index_t>(
        indices, static_cast<int64_t>(b) * index_stride_b +
                     static_cast<int64_t>(j) * index_stride_k);
    if (linear < 0 || linear >= cache_blocks * block_size) {
        if (threadIdx.x == 0)
            scores[score_offset] = -INFINITY;
        return;
    }

    float partial = 0.0f;
    for (int d = threadIdx.x; d < kHeadDim; d += blockDim.x) {
        const float qv = load_q_value<q_t>(
            q, static_cast<int64_t>(b) * q_stride_b +
                   static_cast<int64_t>(h) * q_stride_h +
                   static_cast<int64_t>(d) * q_stride_d);
        partial += qv * load_cache_value(k_cache, cache_stride0_bytes,
                                         block_size, linear, d);
    }
    reductions[threadIdx.x] = partial;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset)
            reductions[threadIdx.x] += reductions[threadIdx.x + offset];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        scores[score_offset] = reductions[0] * softmax_scale;
}

template <typename q_t, typename index_t>
__global__ void sparse_mla_score_tiled_kernel(
    const q_t* __restrict__ q, const uint8_t* __restrict__ k_cache,
    const void* __restrict__ indices, const int64_t* __restrict__ topk_length,
    float* __restrict__ scores, int batch_size, int num_heads, int block_size,
    int64_t cache_blocks, int topk, int64_t q_stride_b, int64_t q_stride_h,
    int64_t q_stride_d, int64_t index_stride_b, int64_t index_stride_k,
    int64_t cache_stride0_bytes, float softmax_scale) {
    const int group_id = threadIdx.x / kScoreGroupSize;
    const int lane = threadIdx.x - group_id * kScoreGroupSize;
    const int j = blockIdx.x * kScoreCandidatesPerBlock + group_id;
    const int h = blockIdx.y;
    const int b = blockIdx.z;
    if (b >= batch_size || h >= num_heads || j >= topk)
        return;

    const int limit =
        topk_length == nullptr ? topk
                               : max(0, min(topk, static_cast<int>(topk_length[b])));
    const int64_t score_offset =
        (static_cast<int64_t>(b) * num_heads + h) * topk + j;
    if (j >= limit) {
        if (lane == 0)
            scores[score_offset] = -INFINITY;
        return;
    }

    const int64_t linear = load_index<index_t>(
        indices, static_cast<int64_t>(b) * index_stride_b +
                     static_cast<int64_t>(j) * index_stride_k);
    if (linear < 0 || linear >= cache_blocks * block_size) {
        if (lane == 0)
            scores[score_offset] = -INFINITY;
        return;
    }

    float partial = 0.0f;
    for (int d = lane; d < kHeadDim; d += kScoreGroupSize) {
        const float qv = load_q_value<q_t>(
            q, static_cast<int64_t>(b) * q_stride_b +
                   static_cast<int64_t>(h) * q_stride_h +
                   static_cast<int64_t>(d) * q_stride_d);
        partial += qv * load_cache_value(k_cache, cache_stride0_bytes,
                                         block_size, linear, d);
    }

    unsigned mask = 0xffffffffu;
    for (int offset = kScoreGroupSize / 2; offset > 0; offset >>= 1)
        partial += __shfl_down_sync(mask, partial, offset, kScoreGroupSize);

    if (lane == 0)
        scores[score_offset] = partial * softmax_scale;
}

template <typename q_t, typename index_t>
__global__ void sparse_mla_score_tiled_scaled_kernel(
    const q_t* __restrict__ q, const uint8_t* __restrict__ k_cache,
    const void* __restrict__ indices, const int64_t* __restrict__ topk_length,
    float* __restrict__ scores, int batch_size, int num_heads, int block_size,
    int64_t cache_blocks, int topk, int64_t q_stride_b, int64_t q_stride_h,
    int64_t q_stride_d, int64_t index_stride_b, int64_t index_stride_k,
    int64_t cache_stride0_bytes, float softmax_scale) {
    const int group_id = threadIdx.x / kScoreGroupSize;
    const int lane = threadIdx.x - group_id * kScoreGroupSize;
    const int j = blockIdx.x * kScoreCandidatesPerBlock + group_id;
    const int h = blockIdx.y;
    const int b = blockIdx.z;
    if (b >= batch_size || h >= num_heads || j >= topk)
        return;

    const int limit =
        topk_length == nullptr ? topk
                               : max(0, min(topk, static_cast<int>(topk_length[b])));
    const int64_t score_offset =
        (static_cast<int64_t>(b) * num_heads + h) * topk + j;
    if (j >= limit) {
        if (lane == 0)
            scores[score_offset] = -INFINITY;
        return;
    }

    const int64_t linear = load_index<index_t>(
        indices, static_cast<int64_t>(b) * index_stride_b +
                     static_cast<int64_t>(j) * index_stride_k);
    if (linear < 0 || linear >= cache_blocks * block_size) {
        if (lane == 0)
            scores[score_offset] = -INFINITY;
        return;
    }

    const uint8_t* token = token_ptr_from_linear(
        k_cache, cache_stride0_bytes, block_size, linear);
    const uint8_t* scale_bytes = scale_ptr_from_linear(
        k_cache, cache_stride0_bytes, block_size, linear);
    float scales[kNumQuantBlocks];
    #pragma unroll
    for (int s = 0; s < kNumQuantBlocks; ++s)
        scales[s] = decode_ue8m0_scale(scale_bytes[s]);

    float partial = 0.0f;
    for (int d = lane; d < kHeadDim; d += kScoreGroupSize) {
        const float qv = load_q_value<q_t>(
            q, static_cast<int64_t>(b) * q_stride_b +
                   static_cast<int64_t>(h) * q_stride_h +
                   static_cast<int64_t>(d) * q_stride_d);
        partial += qv * load_cache_value_from_token(token, scales, d);
    }

    unsigned mask = 0xffffffffu;
    for (int offset = kScoreGroupSize / 2; offset > 0; offset >>= 1)
        partial += __shfl_down_sync(mask, partial, offset, kScoreGroupSize);

    if (lane == 0)
        scores[score_offset] = partial * softmax_scale;
}

__global__ void sparse_mla_softmax_kernel(
    float* __restrict__ scores, float* __restrict__ lse,
    const float* __restrict__ attn_sink, int batch_size, int num_heads,
    int active_heads, int topk) {
    __shared__ float reduce_buf[kThreads];
    const int bh = blockIdx.x;
    const int b = bh / active_heads;
    const int h = bh - b * active_heads;
    if (b >= batch_size)
        return;

    float local_max = -INFINITY;
    const int64_t base = static_cast<int64_t>(bh) * topk;
    for (int j = threadIdx.x; j < topk; j += blockDim.x)
        local_max = fmaxf(local_max, scores[base + j]);
    reduce_buf[threadIdx.x] = local_max;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset)
            reduce_buf[threadIdx.x] =
                fmaxf(reduce_buf[threadIdx.x], reduce_buf[threadIdx.x + offset]);
        __syncthreads();
    }
    const float row_max = reduce_buf[0];

    float local_sum = 0.0f;
    for (int j = threadIdx.x; j < topk; j += blockDim.x) {
        const float p = isfinite(row_max) ? expf(scores[base + j] - row_max) : 0.0f;
        scores[base + j] = p;
        local_sum += p;
    }
    reduce_buf[threadIdx.x] = local_sum;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset)
            reduce_buf[threadIdx.x] += reduce_buf[threadIdx.x + offset];
        __syncthreads();
    }
    const float row_sum = reduce_buf[0];
    const float row_lse = row_sum > 0.0f ? logf(row_sum) + row_max : -INFINITY;
    for (int j = threadIdx.x; j < topk; j += blockDim.x)
        scores[base + j] = row_sum > 0.0f ? scores[base + j] / row_sum : 0.0f;
    if (threadIdx.x == 0)
        lse[static_cast<int64_t>(b) * num_heads + h] = row_lse;
}

template <typename out_t, typename index_t>
__global__ void sparse_mla_output_kernel(
    const uint8_t* __restrict__ k_cache, const void* __restrict__ indices,
    const int64_t* __restrict__ topk_length, const float* __restrict__ scores,
    const float* __restrict__ lse, const float* __restrict__ attn_sink,
    out_t* __restrict__ out, int batch_size, int active_heads, int num_heads,
    int block_size, int64_t cache_blocks, int topk, int64_t out_stride_b,
    int64_t out_stride_h, int64_t out_stride_d, int64_t index_stride_b,
    int64_t index_stride_k, int64_t cache_stride0_bytes) {
    const int bh = blockIdx.x;
    const int tile = blockIdx.y;
    const int b = bh / active_heads;
    const int h = bh - b * active_heads;
    const int d = tile * blockDim.x + threadIdx.x;
    if (b >= batch_size || d >= kHeadDim)
        return;

    const int limit =
        topk_length == nullptr ? topk
                               : max(0, min(topk, static_cast<int>(topk_length[b])));
    float accum = 0.0f;
    const int64_t score_base = static_cast<int64_t>(bh) * topk;
    for (int j = 0; j < limit; ++j) {
        const float p = scores[score_base + j];
        if (p == 0.0f)
            continue;
        const int64_t linear = load_index<index_t>(
            indices, static_cast<int64_t>(b) * index_stride_b +
                         static_cast<int64_t>(j) * index_stride_k);
        if (linear < 0 || linear >= cache_blocks * block_size)
            continue;
        accum += p * load_cache_value(k_cache, cache_stride0_bytes, block_size,
                                      linear, d);
    }

    const float row_lse = lse[static_cast<int64_t>(b) * num_heads + h];
    const float sink = attn_sink == nullptr ? 0.0f : attn_sink[h];
    const float gate =
        attn_sink == nullptr ? 1.0f : 1.0f / (1.0f + expf(-(row_lse - sink)));
    store_out_value<out_t>(
        out, static_cast<int64_t>(b) * out_stride_b +
                 static_cast<int64_t>(h) * out_stride_h +
                 static_cast<int64_t>(d) * out_stride_d,
        accum * gate);
}

template <typename out_t, typename index_t>
__global__ void sparse_mla_output_scaled_kernel(
    const uint8_t* __restrict__ k_cache, const void* __restrict__ indices,
    const int64_t* __restrict__ topk_length, const float* __restrict__ scores,
    const float* __restrict__ lse, const float* __restrict__ attn_sink,
    out_t* __restrict__ out, int batch_size, int active_heads, int num_heads,
    int block_size, int64_t cache_blocks, int topk, int64_t out_stride_b,
    int64_t out_stride_h, int64_t out_stride_d, int64_t index_stride_b,
    int64_t index_stride_k, int64_t cache_stride0_bytes) {
    extern __shared__ float scale_s[];
    const int bh = blockIdx.x;
    const int tile = blockIdx.y;
    const int b = bh / active_heads;
    const int h = bh - b * active_heads;
    const int lane = threadIdx.x;
    const int d = tile * kQuantBlock + lane;
    if (b >= batch_size)
        return;

    const int limit =
        topk_length == nullptr ? topk
                               : max(0, min(topk, static_cast<int>(topk_length[b])));
    const bool fp8_tile = tile < kNumQuantBlocks;
    for (int j = threadIdx.x; j < limit; j += blockDim.x) {
        float scale = 1.0f;
        if (fp8_tile) {
            const int64_t linear = load_index<index_t>(
                indices, static_cast<int64_t>(b) * index_stride_b +
                             static_cast<int64_t>(j) * index_stride_k);
            if (linear >= 0 && linear < cache_blocks * block_size) {
                const uint8_t* scales = scale_ptr_from_linear(
                    k_cache, cache_stride0_bytes, block_size, linear);
                scale = decode_ue8m0_scale(scales[tile]);
            } else {
                scale = 0.0f;
            }
        }
        scale_s[j] = scale;
    }
    __syncthreads();

    if (lane >= kQuantBlock || d >= kHeadDim)
        return;

    float accum = 0.0f;
    const int64_t score_base = static_cast<int64_t>(bh) * topk;
    for (int j = 0; j < limit; ++j) {
        const float p = scores[score_base + j];
        if (p == 0.0f)
            continue;
        const int64_t linear = load_index<index_t>(
            indices, static_cast<int64_t>(b) * index_stride_b +
                         static_cast<int64_t>(j) * index_stride_k);
        if (linear < 0 || linear >= cache_blocks * block_size)
            continue;
        const uint8_t* token = token_ptr_from_linear(
            k_cache, cache_stride0_bytes, block_size, linear);
        float value;
        if (d < kFp8Dim) {
            value = fp8_e4m3fn_to_float(token[d]) * scale_s[j];
        } else {
            const auto* rope =
                reinterpret_cast<const __nv_bfloat16*>(token + kFp8Dim);
            value = __bfloat162float(rope[d - kFp8Dim]);
        }
        accum += p * value;
    }

    const float row_lse = lse[static_cast<int64_t>(b) * num_heads + h];
    const float sink = attn_sink == nullptr ? 0.0f : attn_sink[h];
    const float gate =
        attn_sink == nullptr ? 1.0f : 1.0f / (1.0f + expf(-(row_lse - sink)));
    store_out_value<out_t>(
        out, static_cast<int64_t>(b) * out_stride_b +
                 static_cast<int64_t>(h) * out_stride_h +
                 static_cast<int64_t>(d) * out_stride_d,
        accum * gate);
}

template <typename q_t, typename kv_t>
__global__ void sparse_mla_workspace_score_tiled_kernel(
    const q_t* __restrict__ q, const kv_t* __restrict__ kv_workspace,
    const void* __restrict__ topk_length,
    const void* __restrict__ extra_topk_length, float* __restrict__ scores,
    int batch_size, int active_heads, int num_heads, int main_topk,
    int extra_topk, int candidate_slots, int64_t q_stride_b,
    int64_t q_stride_h, int64_t q_stride_d, int64_t kv_stride_b,
    int64_t kv_stride_s, int64_t kv_stride_d, int topk_length_kind,
    int extra_topk_length_kind, float softmax_scale) {
    const int group_id = threadIdx.x / kScoreGroupSize;
    const int lane = threadIdx.x - group_id * kScoreGroupSize;
    const int j = blockIdx.x * kScoreCandidatesPerBlock + group_id;
    const int h = blockIdx.y;
    const int b = blockIdx.z;
    if (b >= batch_size || h >= active_heads || j >= candidate_slots)
        return;

    const int main_limit = max(
        0, min(main_topk,
               load_length_value(topk_length, topk_length_kind, b, main_topk)));
    const int extra_limit =
        max(0, min(extra_topk,
                   load_length_value(extra_topk_length, extra_topk_length_kind,
                                     b, extra_topk)));
    const bool valid =
        j < main_limit || (j >= main_topk && j < main_topk + extra_limit);
    const int64_t score_offset =
        (static_cast<int64_t>(b) * active_heads + h) * candidate_slots + j;
    if (!valid) {
        if (lane == 0)
            scores[score_offset] = -INFINITY;
        return;
    }

    float partial = 0.0f;
    for (int d = lane; d < kHeadDim; d += kScoreGroupSize) {
        const float qv = load_q_value<q_t>(
            q, static_cast<int64_t>(b) * q_stride_b +
                   static_cast<int64_t>(h) * q_stride_h +
                   static_cast<int64_t>(d) * q_stride_d);
        partial += qv *
                   load_workspace_value<kv_t>(
                       kv_workspace,
                       static_cast<int64_t>(b) * kv_stride_b +
                           static_cast<int64_t>(j) * kv_stride_s +
                           static_cast<int64_t>(d) * kv_stride_d);
    }

    for (int offset = kScoreGroupSize / 2; offset > 0; offset >>= 1)
        partial += __shfl_down_sync(0xffffffffu, partial, offset,
                                    kScoreGroupSize);
    if (lane == 0)
        scores[score_offset] = partial * softmax_scale;
}

// Q-stationary variant of sparse_mla_workspace_score_tiled_kernel.
// All 32 candidate-groups in a block share the same Q[b, h, :] (1 KB bf16).
// The original kernel reread Q from gmem 32× per block — staging Q into smem
// once gives identical float arithmetic (same accumulation order per lane,
// same bf16→float conversion) so numerics are bit-identical.
template <typename q_t, typename kv_t>
__global__ void sparse_mla_workspace_score_tiled_qstat_kernel(
    const q_t* __restrict__ q, const kv_t* __restrict__ kv_workspace,
    const void* __restrict__ topk_length,
    const void* __restrict__ extra_topk_length, float* __restrict__ scores,
    int batch_size, int active_heads, int num_heads, int main_topk,
    int extra_topk, int candidate_slots, int64_t q_stride_b,
    int64_t q_stride_h, int64_t q_stride_d, int64_t kv_stride_b,
    int64_t kv_stride_s, int64_t kv_stride_d, int topk_length_kind,
    int extra_topk_length_kind, float softmax_scale) {
    __shared__ float q_smem[kHeadDim];

    const int h = blockIdx.y;
    const int b = blockIdx.z;
    if (b >= batch_size || h >= active_heads) return;

    const int64_t q_base = static_cast<int64_t>(b) * q_stride_b +
                           static_cast<int64_t>(h) * q_stride_h;
    for (int d = threadIdx.x; d < kHeadDim; d += kThreads) {
        q_smem[d] = load_q_value<q_t>(
            q, q_base + static_cast<int64_t>(d) * q_stride_d);
    }
    __syncthreads();

    const int group_id = threadIdx.x / kScoreGroupSize;
    const int lane = threadIdx.x - group_id * kScoreGroupSize;
    const int j = blockIdx.x * kScoreCandidatesPerBlock + group_id;
    if (j >= candidate_slots) return;

    const int main_limit = max(
        0, min(main_topk,
               load_length_value(topk_length, topk_length_kind, b, main_topk)));
    const int extra_limit =
        max(0, min(extra_topk,
                   load_length_value(extra_topk_length, extra_topk_length_kind,
                                     b, extra_topk)));
    const bool valid =
        j < main_limit || (j >= main_topk && j < main_topk + extra_limit);
    const int64_t score_offset =
        (static_cast<int64_t>(b) * active_heads + h) * candidate_slots + j;
    if (!valid) {
        if (lane == 0)
            scores[score_offset] = -INFINITY;
        return;
    }

    float partial = 0.0f;
    for (int d = lane; d < kHeadDim; d += kScoreGroupSize) {
        const float qv = q_smem[d];
        partial += qv *
                   load_workspace_value<kv_t>(
                       kv_workspace,
                       static_cast<int64_t>(b) * kv_stride_b +
                           static_cast<int64_t>(j) * kv_stride_s +
                           static_cast<int64_t>(d) * kv_stride_d);
    }

    for (int offset = kScoreGroupSize / 2; offset > 0; offset >>= 1)
        partial += __shfl_down_sync(0xffffffffu, partial, offset,
                                    kScoreGroupSize);
    if (lane == 0)
        scores[score_offset] = partial * softmax_scale;
}

// Vectorized-load (bf16x4 = uint64) variant of score_tiled_qstat.
// Block grid is unchanged: (ceil(slots/32), active_heads, batch_size).
// Per-thread inner loop issues 16 LDG.E.U64 over head_dim=512 instead of
// 64 LDG.E.U16. Requires kv_stride_d == 1 (innermost contiguous head_dim),
// which the dispatcher checks at runtime. q_stride_d may be anything since
// Q is staged into smem first.
//
// Accumulation order differs from scalar: each iter reduces a 4-elt quad
// (q[d]*kv[d] + q[d+1]*kv[d+1] + q[d+2]*kv[d+2] + q[d+3]*kv[d+3]) before
// adding to running partial, vs the scalar kernel's stride-8 single-element
// running sum. Float arithmetic is non-associative so output is bit-similar
// but not bit-identical. Quality probe required.
//
// kScalarCheck template flag: when true, runs the scalar reduction order
// inside this kernel for FMA-oracle parity debug. Production shape with
// kScalarCheck=false issues the wide loads.
template <typename q_t, typename kv_t, bool kScalarCheck = false,
          int kVecWidth = 4>
__global__ void sparse_mla_workspace_score_tiled_qstat_vec_kernel(
    const q_t* __restrict__ q, const kv_t* __restrict__ kv_workspace,
    const void* __restrict__ topk_length,
    const void* __restrict__ extra_topk_length, float* __restrict__ scores,
    int batch_size, int active_heads, int num_heads, int main_topk,
    int extra_topk, int candidate_slots, int64_t q_stride_b,
    int64_t q_stride_h, int64_t q_stride_d, int64_t kv_stride_b,
    int64_t kv_stride_s, int64_t kv_stride_d, int topk_length_kind,
    int extra_topk_length_kind, float softmax_scale) {
    static_assert(kScoreGroupSize == 8,
                  "qstat_vec kernel assumes kScoreGroupSize == 8");
    static_assert(kHeadDim % 32 == 0,
                  "qstat_vec kernel assumes head_dim multiple of 32");
    static_assert(kVecWidth == 4 || kVecWidth == 8,
                  "qstat_vec kernel supports kVecWidth in {4, 8}");
    static_assert(kHeadDim % (kScoreGroupSize * kVecWidth) == 0,
                  "head_dim must be multiple of group_size * vec_width");
    __shared__ float q_smem[kHeadDim];

    const int h = blockIdx.y;
    const int b = blockIdx.z;
    if (b >= batch_size || h >= active_heads) return;

    const int64_t q_base = static_cast<int64_t>(b) * q_stride_b +
                           static_cast<int64_t>(h) * q_stride_h;
    for (int d = threadIdx.x; d < kHeadDim; d += kThreads) {
        q_smem[d] = load_q_value<q_t>(
            q, q_base + static_cast<int64_t>(d) * q_stride_d);
    }
    __syncthreads();

    const int group_id = threadIdx.x / kScoreGroupSize;
    const int lane = threadIdx.x - group_id * kScoreGroupSize;
    const int j = blockIdx.x * kScoreCandidatesPerBlock + group_id;
    if (j >= candidate_slots) return;

    const int main_limit = max(
        0, min(main_topk,
               load_length_value(topk_length, topk_length_kind, b, main_topk)));
    const int extra_limit =
        max(0, min(extra_topk,
                   load_length_value(extra_topk_length, extra_topk_length_kind,
                                     b, extra_topk)));
    const bool valid =
        j < main_limit || (j >= main_topk && j < main_topk + extra_limit);
    const int64_t score_offset =
        (static_cast<int64_t>(b) * active_heads + h) * candidate_slots + j;
    if (!valid) {
        if (lane == 0)
            scores[score_offset] = -INFINITY;
        return;
    }

    float partial = 0.0f;
    if constexpr (kScalarCheck) {
        // Scalar oracle: same reduction order as qstat kernel.
        for (int d = lane; d < kHeadDim; d += kScoreGroupSize) {
            const float qv = q_smem[d];
            partial += qv *
                       load_workspace_value<kv_t>(
                           kv_workspace,
                           static_cast<int64_t>(b) * kv_stride_b +
                               static_cast<int64_t>(j) * kv_stride_s +
                               static_cast<int64_t>(d) * kv_stride_d);
        }
    } else {
        constexpr int kEltsPerVec = kVecWidth;
        constexpr int kStridePerIter = kScoreGroupSize * kEltsPerVec;
        constexpr int kIters = kHeadDim / kStridePerIter;
        const int64_t kv_row_base = static_cast<int64_t>(b) * kv_stride_b +
                                    static_cast<int64_t>(j) * kv_stride_s;
        const kv_t* kv_row_ptr = kv_workspace + kv_row_base;
        #pragma unroll
        for (int g = 0; g < kIters; ++g) {
            const int d_base = g * kStridePerIter + lane * kEltsPerVec;
            if constexpr (kEltsPerVec == 4) {
                const __nv_bfloat162* kv_pair_ptr =
                    reinterpret_cast<const __nv_bfloat162*>(kv_row_ptr +
                                                            d_base);
                const __nv_bfloat162 kv_pair_lo = kv_pair_ptr[0];
                const __nv_bfloat162 kv_pair_hi = kv_pair_ptr[1];
                const float kv0 = __bfloat162float(kv_pair_lo.x);
                const float kv1 = __bfloat162float(kv_pair_lo.y);
                const float kv2 = __bfloat162float(kv_pair_hi.x);
                const float kv3 = __bfloat162float(kv_pair_hi.y);
                partial += q_smem[d_base] * kv0;
                partial += q_smem[d_base + 1] * kv1;
                partial += q_smem[d_base + 2] * kv2;
                partial += q_smem[d_base + 3] * kv3;
            } else {  // kEltsPerVec == 8 — single 16-byte (uint4) load
                const uint4* kv_quad_ptr =
                    reinterpret_cast<const uint4*>(kv_row_ptr + d_base);
                const uint4 kv_quad = *kv_quad_ptr;
                const __nv_bfloat162 kv_pair0 =
                    *reinterpret_cast<const __nv_bfloat162*>(&kv_quad.x);
                const __nv_bfloat162 kv_pair1 =
                    *reinterpret_cast<const __nv_bfloat162*>(&kv_quad.y);
                const __nv_bfloat162 kv_pair2 =
                    *reinterpret_cast<const __nv_bfloat162*>(&kv_quad.z);
                const __nv_bfloat162 kv_pair3 =
                    *reinterpret_cast<const __nv_bfloat162*>(&kv_quad.w);
                const float kv0 = __bfloat162float(kv_pair0.x);
                const float kv1 = __bfloat162float(kv_pair0.y);
                const float kv2 = __bfloat162float(kv_pair1.x);
                const float kv3 = __bfloat162float(kv_pair1.y);
                const float kv4 = __bfloat162float(kv_pair2.x);
                const float kv5 = __bfloat162float(kv_pair2.y);
                const float kv6 = __bfloat162float(kv_pair3.x);
                const float kv7 = __bfloat162float(kv_pair3.y);
                partial += q_smem[d_base] * kv0;
                partial += q_smem[d_base + 1] * kv1;
                partial += q_smem[d_base + 2] * kv2;
                partial += q_smem[d_base + 3] * kv3;
                partial += q_smem[d_base + 4] * kv4;
                partial += q_smem[d_base + 5] * kv5;
                partial += q_smem[d_base + 6] * kv6;
                partial += q_smem[d_base + 7] * kv7;
            }
        }
    }

    for (int offset = kScoreGroupSize / 2; offset > 0; offset >>= 1)
        partial += __shfl_down_sync(0xffffffffu, partial, offset,
                                    kScoreGroupSize);
    if (lane == 0)
        scores[score_offset] = partial * softmax_scale;
}

// Cross-head bf16 MMA variant of the score kernel.
// Each block computes a tile D[h_base:h_base+16, n_base:n_base+32] of scores
// using `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`. K=512 is
// streamed in 64-element chunks; Q[16,64] and KV[32,64] are staged into smem
// once per chunk. 4 warps × (M=16, N=8) sub-tiles = full 16×32 tile per block.
//
// Per the lesson from HC_PRENORM_MMA: the kScalarCheck template flag swaps
// the asm for a hand-rolled scalar matmul reading the same smem slots. Use it
// to validate fragment layout before trusting the MMA path.
constexpr int kScoreMmaM = 16;
constexpr int kScoreMmaN = 32;
constexpr int kScoreMmaK = 16;
constexpr int kScoreMmaKChunk = 64;
constexpr int kScoreMmaNumChunks = kHeadDim / kScoreMmaKChunk;          // 8
constexpr int kScoreMmaNumWarps = kScoreMmaN / 8;                        // 4
constexpr int kScoreMmaThreads = 32 * kScoreMmaNumWarps;                 // 128
constexpr size_t kScoreMmaSmemBytes =
    static_cast<size_t>(kScoreMmaM + kScoreMmaN) * kScoreMmaKChunk *
    sizeof(__nv_bfloat16);

template <typename q_t, typename kv_t, bool kScalarCheck>
__global__ void sparse_mla_workspace_score_mma_kernel(
    const q_t* __restrict__ q, const kv_t* __restrict__ kv_workspace,
    const void* __restrict__ topk_length,
    const void* __restrict__ extra_topk_length, float* __restrict__ scores,
    int batch_size, int active_heads, int num_heads, int main_topk,
    int extra_topk, int candidate_slots, int64_t q_stride_b,
    int64_t q_stride_h, int64_t q_stride_d, int64_t kv_stride_b,
    int64_t kv_stride_s, int64_t kv_stride_d, int topk_length_kind,
    int extra_topk_length_kind, float softmax_scale) {
    const int b = blockIdx.z;
    const int h_tile = blockIdx.y;
    const int n_tile = blockIdx.x;
    const int h_base = h_tile * kScoreMmaM;
    const int n_base = n_tile * kScoreMmaN;
    if (b >= batch_size || h_base >= active_heads || n_base >= candidate_slots)
        return;

    extern __shared__ __nv_bfloat16 score_mma_smem[];
    __nv_bfloat16* q_smem = score_mma_smem;
    __nv_bfloat16* kv_smem = q_smem + kScoreMmaM * kScoreMmaKChunk;

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int n_warp_in_block = warp_id * 8;
    const int groupID = lane >> 2;
    const int tid_in_group = lane & 3;

    const int64_t q_b_base = static_cast<int64_t>(b) * q_stride_b;
    const int64_t kv_b_base = static_cast<int64_t>(b) * kv_stride_b;

    float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;

    #pragma unroll 1
    for (int chunk = 0; chunk < kScoreMmaNumChunks; ++chunk) {
        const int k_off = chunk * kScoreMmaKChunk;

        // Cooperative load Q[h_base + m, k_off + kk] -> q_smem[m, kk].
        for (int idx = tid; idx < kScoreMmaM * kScoreMmaKChunk;
             idx += kScoreMmaThreads) {
            const int m = idx / kScoreMmaKChunk;
            const int kk = idx - m * kScoreMmaKChunk;
            const int h = h_base + m;
            float v = 0.f;
            if (h < active_heads && h < num_heads) {
                const int64_t off = q_b_base +
                                    static_cast<int64_t>(h) * q_stride_h +
                                    static_cast<int64_t>(k_off + kk) *
                                        q_stride_d;
                v = load_q_value<q_t>(q, off);
            }
            q_smem[m * kScoreMmaKChunk + kk] = __float2bfloat16_rn(v);
        }

        // Cooperative load KV[n_base + n, k_off + kk] -> kv_smem[n, kk].
        for (int idx = tid; idx < kScoreMmaN * kScoreMmaKChunk;
             idx += kScoreMmaThreads) {
            const int n = idx / kScoreMmaKChunk;
            const int kk = idx - n * kScoreMmaKChunk;
            const int j = n_base + n;
            float v = 0.f;
            if (j < candidate_slots) {
                const int64_t off = kv_b_base +
                                    static_cast<int64_t>(j) * kv_stride_s +
                                    static_cast<int64_t>(k_off + kk) *
                                        kv_stride_d;
                v = load_workspace_value<kv_t>(kv_workspace, off);
            }
            kv_smem[n * kScoreMmaKChunk + kk] = __float2bfloat16_rn(v);
        }
        __syncthreads();

        // 4 inner MMA steps per chunk (K_CHUNK=64 / K_MMA=16 = 4).
        #pragma unroll
        for (int kc = 0; kc < kScoreMmaKChunk; kc += kScoreMmaK) {
            // A fragment (M=16, K=16) row-major, .row in ptx:
            //   a[0] = A[gID  ][2*tid+0..1]
            //   a[1] = A[gID+8][2*tid+0..1]
            //   a[2] = A[gID  ][2*tid+8..9]
            //   a[3] = A[gID+8][2*tid+8..9]
            const int a_col_base = 2 * tid_in_group;
            const __nv_bfloat16* a_row0_ptr =
                q_smem + groupID * kScoreMmaKChunk + kc;
            const __nv_bfloat16* a_row1_ptr =
                q_smem + (groupID + 8) * kScoreMmaKChunk + kc;
            // B fragment (N=8, K=16), .col in ptx -- our kv_smem stores
            // [n][k] row-major, which IS B^T in [n][k] order, perfect for .col:
            //   b[0] = B[2*tid+0..1, gID] = kv_smem[gID][2*tid+0..1]
            //   b[1] = B[2*tid+8..9, gID] = kv_smem[gID][2*tid+8..9]
            const int b_n = n_warp_in_block + groupID;
            const __nv_bfloat16* b_row_ptr =
                kv_smem + b_n * kScoreMmaKChunk + kc;

            const uint32_t a0 = *reinterpret_cast<const uint32_t*>(
                a_row0_ptr + a_col_base);
            const uint32_t a1 = *reinterpret_cast<const uint32_t*>(
                a_row1_ptr + a_col_base);
            const uint32_t a2 = *reinterpret_cast<const uint32_t*>(
                a_row0_ptr + a_col_base + 8);
            const uint32_t a3 = *reinterpret_cast<const uint32_t*>(
                a_row1_ptr + a_col_base + 8);
            const uint32_t b0 = *reinterpret_cast<const uint32_t*>(
                b_row_ptr + a_col_base);
            const uint32_t b1 = *reinterpret_cast<const uint32_t*>(
                b_row_ptr + a_col_base + 8);

            if constexpr (kScalarCheck) {
                // Hand-rolled matmul replicating MMA fragment semantics.
                // D fragment (M=16, N=8) row-major, stride-2 cols:
                //   d[0] -> D[gID  ][2*tid+0]
                //   d[1] -> D[gID  ][2*tid+1]
                //   d[2] -> D[gID+8][2*tid+0]
                //   d[3] -> D[gID+8][2*tid+1]
                const int row0 = groupID;
                const int row1 = groupID + 8;
                const int col0_local =
                    n_warp_in_block + 2 * tid_in_group;
                const int col1_local = col0_local + 1;
                float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
                #pragma unroll
                for (int kk = 0; kk < kScoreMmaK; ++kk) {
                    const float ar0 = __bfloat162float(
                        q_smem[row0 * kScoreMmaKChunk + kc + kk]);
                    const float ar1 = __bfloat162float(
                        q_smem[row1 * kScoreMmaKChunk + kc + kk]);
                    const float bn0 = __bfloat162float(
                        kv_smem[col0_local * kScoreMmaKChunk + kc + kk]);
                    const float bn1 = __bfloat162float(
                        kv_smem[col1_local * kScoreMmaKChunk + kc + kk]);
                    acc0 += ar0 * bn0;
                    acc1 += ar0 * bn1;
                    acc2 += ar1 * bn0;
                    acc3 += ar1 * bn1;
                }
                d0 += acc0;
                d1 += acc1;
                d2 += acc2;
                d3 += acc3;
                (void)a0; (void)a1; (void)a2; (void)a3;
                (void)b0; (void)b1;
            } else {
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                    "{%0, %1, %2, %3}, "
                    "{%4, %5, %6, %7}, "
                    "{%8, %9}, "
                    "{%0, %1, %2, %3};\n"
                    : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                      "r"(b0), "r"(b1));
            }
        }
        __syncthreads();
    }

    const int main_limit = max(
        0, min(main_topk,
               load_length_value(topk_length, topk_length_kind, b, main_topk)));
    const int extra_limit = max(
        0, min(extra_topk,
               load_length_value(extra_topk_length, extra_topk_length_kind, b,
                                 extra_topk)));

    auto write_score = [&](int h, int j, float val) {
        if (h >= active_heads || j >= candidate_slots) return;
        const bool valid =
            j < main_limit || (j >= main_topk && j < main_topk + extra_limit);
        const int64_t off =
            (static_cast<int64_t>(b) * active_heads + h) * candidate_slots + j;
        scores[off] = valid ? (val * softmax_scale) : -INFINITY;
    };

    const int h0 = h_base + groupID;
    const int h1 = h_base + groupID + 8;
    const int j0 = n_base + n_warp_in_block + 2 * tid_in_group;
    const int j1 = j0 + 1;
    write_score(h0, j0, d0);
    write_score(h0, j1, d1);
    write_score(h1, j0, d2);
    write_score(h1, j1, d3);
}

// ===== Split-K SCORE MMA =================================================
// Layout (production active_heads=32, candidate_slots≤128):
//   block tile  : M=32 (all active heads), N=8 (one slot N-tile), K-window
//                 = kHeadDim / kSplit = 128 (out of 512).
//   block grid  : (ceil(slots/8), batch, kSplit) — at slots=128, B=8,
//                 kSplit=4 → 16 × 8 × 4 = 512 blocks vs 188 SMs.
//   warps/block : 2  (warp 0 → M[0..15], warp 1 → M[16..31]).
//   smem        : (M+N) * Kchunk * 2B = 5 KB; trivially fits with high
//                 occupancy.
// Each block streams its assigned K-window (128 elements) in two 64-elem
// chunks, accumulates float, and writes a partial to
//   partials[kSplit, batch, head, slot]
// A second tiny reduce kernel sums partials → final scores with the
// validity mask + softmax_scale.
constexpr int kScoreMmaSpkM = 32;
constexpr int kScoreMmaSpkN = 8;
constexpr int kScoreMmaSpkK = 16;
constexpr int kScoreMmaSpkKChunk = 64;
constexpr int kScoreMmaSpkSplit = 4;
constexpr int kScoreMmaSpkKPerBlock = kHeadDim / kScoreMmaSpkSplit;
constexpr int kScoreMmaSpkChunksPerBlock =
    kScoreMmaSpkKPerBlock / kScoreMmaSpkKChunk;
constexpr int kScoreMmaSpkNumWarps = kScoreMmaSpkM / 16;
constexpr int kScoreMmaSpkThreads = 32 * kScoreMmaSpkNumWarps;
constexpr size_t kScoreMmaSpkSmemBytes =
    static_cast<size_t>(kScoreMmaSpkM + kScoreMmaSpkN) * kScoreMmaSpkKChunk *
    sizeof(__nv_bfloat16);

template <typename q_t, typename kv_t, bool kScalarCheck>
__global__ void sparse_mla_workspace_score_mma_splitk_kernel(
    const q_t* __restrict__ q, const kv_t* __restrict__ kv_workspace,
    float* __restrict__ partials, int batch_size, int active_heads,
    int num_heads, int candidate_slots, int64_t q_stride_b,
    int64_t q_stride_h, int64_t q_stride_d, int64_t kv_stride_b,
    int64_t kv_stride_s, int64_t kv_stride_d) {
    const int n_tile = blockIdx.x;
    const int b = blockIdx.y;
    const int sk = blockIdx.z;
    const int n_base = n_tile * kScoreMmaSpkN;
    const int sk_k_base = sk * kScoreMmaSpkKPerBlock;
    if (b >= batch_size || n_base >= candidate_slots) return;

    extern __shared__ __nv_bfloat16 score_mma_spk_smem[];
    __nv_bfloat16* q_smem = score_mma_spk_smem;
    __nv_bfloat16* kv_smem = q_smem + kScoreMmaSpkM * kScoreMmaSpkKChunk;

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int groupID = lane >> 2;
    const int tid_in_group = lane & 3;

    const int m_warp_base = warp_id * 16;  // 0 or 16
    const int64_t q_b_base = static_cast<int64_t>(b) * q_stride_b;
    const int64_t kv_b_base = static_cast<int64_t>(b) * kv_stride_b;

    float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;

    #pragma unroll
    for (int chunk = 0; chunk < kScoreMmaSpkChunksPerBlock; ++chunk) {
        const int k_off = sk_k_base + chunk * kScoreMmaSpkKChunk;

        // Cooperative load Q[h_base + m, k_off + kk] -> q_smem[m, kk].
        // 64 threads × 32 elements = 2048 bf16 = M*Kchunk.
        for (int idx = tid; idx < kScoreMmaSpkM * kScoreMmaSpkKChunk;
             idx += kScoreMmaSpkThreads) {
            const int m = idx / kScoreMmaSpkKChunk;
            const int kk = idx - m * kScoreMmaSpkKChunk;
            float v = 0.f;
            if (m < active_heads && m < num_heads) {
                const int64_t off = q_b_base +
                                    static_cast<int64_t>(m) * q_stride_h +
                                    static_cast<int64_t>(k_off + kk) *
                                        q_stride_d;
                v = load_q_value<q_t>(q, off);
            }
            q_smem[m * kScoreMmaSpkKChunk + kk] = __float2bfloat16_rn(v);
        }

        // Cooperative load KV[n_base + n, k_off + kk] -> kv_smem[n, kk].
        // 64 threads × 8 elements = 512 bf16 = N*Kchunk.
        for (int idx = tid; idx < kScoreMmaSpkN * kScoreMmaSpkKChunk;
             idx += kScoreMmaSpkThreads) {
            const int n = idx / kScoreMmaSpkKChunk;
            const int kk = idx - n * kScoreMmaSpkKChunk;
            const int j = n_base + n;
            float v = 0.f;
            if (j < candidate_slots) {
                const int64_t off = kv_b_base +
                                    static_cast<int64_t>(j) * kv_stride_s +
                                    static_cast<int64_t>(k_off + kk) *
                                        kv_stride_d;
                v = load_workspace_value<kv_t>(kv_workspace, off);
            }
            kv_smem[n * kScoreMmaSpkKChunk + kk] = __float2bfloat16_rn(v);
        }
        __syncthreads();

        // 4 inner MMA steps per chunk (Kchunk=64 / Kmma=16 = 4).
        // Each warp owns its M=16 sub-tile (warp 0 → M[0..15], warp 1 → [16..31]).
        #pragma unroll
        for (int kc = 0; kc < kScoreMmaSpkKChunk; kc += kScoreMmaSpkK) {
            const int a_col_base = 2 * tid_in_group;
            const __nv_bfloat16* a_row0_ptr =
                q_smem + (m_warp_base + groupID) * kScoreMmaSpkKChunk + kc;
            const __nv_bfloat16* a_row1_ptr =
                q_smem + (m_warp_base + groupID + 8) * kScoreMmaSpkKChunk + kc;
            // B fragment (N=8, K=16), .col PTX layout — kv_smem stores [n][k]
            // row-major which is exactly B^T in (n,k) order.
            const __nv_bfloat16* b_row_ptr =
                kv_smem + groupID * kScoreMmaSpkKChunk + kc;

            const uint32_t a0 = *reinterpret_cast<const uint32_t*>(
                a_row0_ptr + a_col_base);
            const uint32_t a1 = *reinterpret_cast<const uint32_t*>(
                a_row1_ptr + a_col_base);
            const uint32_t a2 = *reinterpret_cast<const uint32_t*>(
                a_row0_ptr + a_col_base + 8);
            const uint32_t a3 = *reinterpret_cast<const uint32_t*>(
                a_row1_ptr + a_col_base + 8);
            const uint32_t b0 = *reinterpret_cast<const uint32_t*>(
                b_row_ptr + a_col_base);
            const uint32_t b1 = *reinterpret_cast<const uint32_t*>(
                b_row_ptr + a_col_base + 8);

            if constexpr (kScalarCheck) {
                const int row0 = m_warp_base + groupID;
                const int row1 = m_warp_base + groupID + 8;
                const int col0 = 2 * tid_in_group;
                const int col1 = col0 + 1;
                float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
                #pragma unroll
                for (int kk = 0; kk < kScoreMmaSpkK; ++kk) {
                    const float ar0 = __bfloat162float(
                        q_smem[row0 * kScoreMmaSpkKChunk + kc + kk]);
                    const float ar1 = __bfloat162float(
                        q_smem[row1 * kScoreMmaSpkKChunk + kc + kk]);
                    const float bn0 = __bfloat162float(
                        kv_smem[col0 * kScoreMmaSpkKChunk + kc + kk]);
                    const float bn1 = __bfloat162float(
                        kv_smem[col1 * kScoreMmaSpkKChunk + kc + kk]);
                    acc0 += ar0 * bn0;
                    acc1 += ar0 * bn1;
                    acc2 += ar1 * bn0;
                    acc3 += ar1 * bn1;
                }
                d0 += acc0;
                d1 += acc1;
                d2 += acc2;
                d3 += acc3;
                (void)a0; (void)a1; (void)a2; (void)a3;
                (void)b0; (void)b1;
            } else {
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                    "{%0, %1, %2, %3}, "
                    "{%4, %5, %6, %7}, "
                    "{%8, %9}, "
                    "{%0, %1, %2, %3};\n"
                    : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                      "r"(b0), "r"(b1));
            }
        }
        __syncthreads();
    }

    // Write partial (no scaling, no validity masking — reduce kernel handles).
    // partials layout: [kSplit, batch, active_heads, candidate_slots].
    auto write_partial = [&](int h, int j, float val) {
        if (h >= active_heads || j >= candidate_slots) return;
        const int64_t off =
            ((static_cast<int64_t>(sk) * batch_size + b) * active_heads + h) *
                candidate_slots +
            j;
        partials[off] = val;
    };

    const int h0 = m_warp_base + groupID;
    const int h1 = m_warp_base + groupID + 8;
    const int j0 = n_base + 2 * tid_in_group;
    const int j1 = j0 + 1;
    write_partial(h0, j0, d0);
    write_partial(h0, j1, d1);
    write_partial(h1, j0, d2);
    write_partial(h1, j1, d3);
}

// Reduce partials[kSplit, B, H, S] → scores[B, H, S] with validity mask
// and softmax_scale. Bandwidth-bound; one float load per split per cell.
__global__ void sparse_mla_workspace_score_mma_splitk_reduce_kernel(
    const float* __restrict__ partials,
    const void* __restrict__ topk_length,
    const void* __restrict__ extra_topk_length, float* __restrict__ scores,
    int batch_size, int active_heads, int main_topk, int extra_topk,
    int candidate_slots, int topk_length_kind, int extra_topk_length_kind,
    float softmax_scale) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int h = blockIdx.y;
    const int b = blockIdx.z;
    if (j >= candidate_slots) return;
    if (h >= active_heads || b >= batch_size) return;

    const int main_limit = max(
        0, min(main_topk,
               load_length_value(topk_length, topk_length_kind, b, main_topk)));
    const int extra_limit = max(
        0, min(extra_topk,
               load_length_value(extra_topk_length, extra_topk_length_kind, b,
                                 extra_topk)));
    const bool valid =
        j < main_limit || (j >= main_topk && j < main_topk + extra_limit);

    const int64_t out_off =
        (static_cast<int64_t>(b) * active_heads + h) * candidate_slots + j;
    if (!valid) {
        scores[out_off] = -INFINITY;
        return;
    }

    float sum = 0.f;
    #pragma unroll
    for (int sk = 0; sk < kScoreMmaSpkSplit; ++sk) {
        const int64_t in_off =
            ((static_cast<int64_t>(sk) * batch_size + b) * active_heads + h) *
                candidate_slots +
            j;
        sum += partials[in_off];
    }
    scores[out_off] = sum * softmax_scale;
}

template <typename kv_t, typename out_t>
__global__ void sparse_mla_workspace_output_kernel(
    const kv_t* __restrict__ kv_workspace, const void* __restrict__ topk_length,
    const void* __restrict__ extra_topk_length,
    const float* __restrict__ scores, const float* __restrict__ lse,
    const float* __restrict__ attn_sink, out_t* __restrict__ out,
    int batch_size, int active_heads, int num_heads, int main_topk,
    int extra_topk, int candidate_slots, int64_t kv_stride_b,
    int64_t kv_stride_s, int64_t kv_stride_d, int64_t out_stride_b,
    int64_t out_stride_h, int64_t out_stride_d, int topk_length_kind,
    int extra_topk_length_kind) {
    const int bh = blockIdx.x;
    const int tile = blockIdx.y;
    const int b = bh / active_heads;
    const int h = bh - b * active_heads;
    const int d = tile * blockDim.x + threadIdx.x;
    if (b >= batch_size || d >= kHeadDim)
        return;

    const int main_limit = max(
        0, min(main_topk,
               load_length_value(topk_length, topk_length_kind, b, main_topk)));
    const int extra_limit =
        max(0, min(extra_topk,
                   load_length_value(extra_topk_length, extra_topk_length_kind,
                                     b, extra_topk)));
    const int64_t score_base =
        (static_cast<int64_t>(b) * active_heads + h) * candidate_slots;

    float accum = 0.0f;
    for (int j = 0; j < main_limit; ++j) {
        accum += scores[score_base + j] *
                 load_workspace_value<kv_t>(
                     kv_workspace,
                     static_cast<int64_t>(b) * kv_stride_b +
                         static_cast<int64_t>(j) * kv_stride_s +
                         static_cast<int64_t>(d) * kv_stride_d);
    }
    for (int j = 0; j < extra_limit; ++j) {
        const int workspace_j = main_topk + j;
        accum += scores[score_base + workspace_j] *
                 load_workspace_value<kv_t>(
                     kv_workspace,
                     static_cast<int64_t>(b) * kv_stride_b +
                         static_cast<int64_t>(workspace_j) * kv_stride_s +
                         static_cast<int64_t>(d) * kv_stride_d);
    }

    const float row_lse = lse[static_cast<int64_t>(b) * num_heads + h];
    const float sink = attn_sink == nullptr ? 0.0f : attn_sink[h];
    const float gate =
        attn_sink == nullptr ? 1.0f : 1.0f / (1.0f + expf(-(row_lse - sink)));
    store_out_value<out_t>(
        out, static_cast<int64_t>(b) * out_stride_b +
                 static_cast<int64_t>(h) * out_stride_h +
                 static_cast<int64_t>(d) * out_stride_d,
        accum * gate);
}

// Score-stationary variant of sparse_mla_workspace_output_kernel.
// Every thread in a block reads the same scores[score_base, 0..cand-1] —
// stage them into smem once. KV reads are still per-thread (d differs).
// Multiplication / accumulation order is preserved → bit-identical numerics.
// Dynamic smem sized to candidate_slots * sizeof(float) at launch.
template <typename kv_t, typename out_t>
__global__ void sparse_mla_workspace_output_sstat_kernel(
    const kv_t* __restrict__ kv_workspace, const void* __restrict__ topk_length,
    const void* __restrict__ extra_topk_length,
    const float* __restrict__ scores, const float* __restrict__ lse,
    const float* __restrict__ attn_sink, out_t* __restrict__ out,
    int batch_size, int active_heads, int num_heads, int main_topk,
    int extra_topk, int candidate_slots, int64_t kv_stride_b,
    int64_t kv_stride_s, int64_t kv_stride_d, int64_t out_stride_b,
    int64_t out_stride_h, int64_t out_stride_d, int topk_length_kind,
    int extra_topk_length_kind) {
    extern __shared__ float s_smem[];

    const int bh = blockIdx.x;
    const int tile = blockIdx.y;
    const int b = bh / active_heads;
    const int h = bh - b * active_heads;
    if (b >= batch_size) return;

    const int64_t score_base =
        (static_cast<int64_t>(b) * active_heads + h) * candidate_slots;
    for (int s = threadIdx.x; s < candidate_slots; s += blockDim.x) {
        s_smem[s] = scores[score_base + s];
    }
    __syncthreads();

    const int d = tile * blockDim.x + threadIdx.x;
    if (d >= kHeadDim) return;

    const int main_limit = max(
        0, min(main_topk,
               load_length_value(topk_length, topk_length_kind, b, main_topk)));
    const int extra_limit =
        max(0, min(extra_topk,
                   load_length_value(extra_topk_length, extra_topk_length_kind,
                                     b, extra_topk)));

    float accum = 0.0f;
    for (int j = 0; j < main_limit; ++j) {
        accum += s_smem[j] *
                 load_workspace_value<kv_t>(
                     kv_workspace,
                     static_cast<int64_t>(b) * kv_stride_b +
                         static_cast<int64_t>(j) * kv_stride_s +
                         static_cast<int64_t>(d) * kv_stride_d);
    }
    for (int j = 0; j < extra_limit; ++j) {
        const int workspace_j = main_topk + j;
        accum += s_smem[workspace_j] *
                 load_workspace_value<kv_t>(
                     kv_workspace,
                     static_cast<int64_t>(b) * kv_stride_b +
                         static_cast<int64_t>(workspace_j) * kv_stride_s +
                         static_cast<int64_t>(d) * kv_stride_d);
    }

    const float row_lse = lse[static_cast<int64_t>(b) * num_heads + h];
    const float sink = attn_sink == nullptr ? 0.0f : attn_sink[h];
    const float gate =
        attn_sink == nullptr ? 1.0f : 1.0f / (1.0f + expf(-(row_lse - sink)));
    store_out_value<out_t>(
        out, static_cast<int64_t>(b) * out_stride_b +
                 static_cast<int64_t>(h) * out_stride_h +
                 static_cast<int64_t>(d) * out_stride_d,
        accum * gate);
}

// bf16x4 vec-load variant of sparse_mla_workspace_output_sstat_kernel.
// Each thread owns 4 contiguous head_dim outputs (d_base..d_base+3) and
// reads K via reinterpret_cast<__nv_bfloat162> (one LDG.E.U64 per j) instead
// of 4 single-element bf16 LDGs. Per-d FMA chain is identical to scalar:
// for each d in {d_base..+3}, accum_d = sum_j s_smem[j] * kv[b,j,d] —
// bit-identical to running the scalar kernel four times. Requires
// kv_stride_d == 1 (innermost head_dim contiguous) and kHeadDim % 4 == 0.
template <typename kv_t, typename out_t>
__global__ void sparse_mla_workspace_output_sstat_vec_kernel(
    const kv_t* __restrict__ kv_workspace, const void* __restrict__ topk_length,
    const void* __restrict__ extra_topk_length,
    const float* __restrict__ scores, const float* __restrict__ lse,
    const float* __restrict__ attn_sink, out_t* __restrict__ out,
    int batch_size, int active_heads, int num_heads, int main_topk,
    int extra_topk, int candidate_slots, int64_t kv_stride_b,
    int64_t kv_stride_s, int64_t kv_stride_d, int64_t out_stride_b,
    int64_t out_stride_h, int64_t out_stride_d, int topk_length_kind,
    int extra_topk_length_kind) {
    extern __shared__ float s_smem[];

    const int bh = blockIdx.x;
    const int tile = blockIdx.y;
    const int b = bh / active_heads;
    const int h = bh - b * active_heads;
    if (b >= batch_size) return;

    const int64_t score_base =
        (static_cast<int64_t>(b) * active_heads + h) * candidate_slots;
    for (int s = threadIdx.x; s < candidate_slots; s += blockDim.x) {
        s_smem[s] = scores[score_base + s];
    }
    __syncthreads();

    const int d_base = (tile * blockDim.x + threadIdx.x) * 4;
    if (d_base >= kHeadDim) return;

    const int main_limit = max(
        0, min(main_topk,
               load_length_value(topk_length, topk_length_kind, b, main_topk)));
    const int extra_limit =
        max(0, min(extra_topk,
                   load_length_value(extra_topk_length, extra_topk_length_kind,
                                     b, extra_topk)));

    float accum0 = 0.0f, accum1 = 0.0f, accum2 = 0.0f, accum3 = 0.0f;
    for (int j = 0; j < main_limit; ++j) {
        const float s = s_smem[j];
        const int64_t kv_off = static_cast<int64_t>(b) * kv_stride_b +
                               static_cast<int64_t>(j) * kv_stride_s +
                               static_cast<int64_t>(d_base) * kv_stride_d;
        const __nv_bfloat162* p2 =
            reinterpret_cast<const __nv_bfloat162*>(kv_workspace + kv_off);
        __nv_bfloat162 v01 = p2[0];
        __nv_bfloat162 v23 = p2[1];
        accum0 += s * __bfloat162float(v01.x);
        accum1 += s * __bfloat162float(v01.y);
        accum2 += s * __bfloat162float(v23.x);
        accum3 += s * __bfloat162float(v23.y);
    }
    for (int j = 0; j < extra_limit; ++j) {
        const int workspace_j = main_topk + j;
        const float s = s_smem[workspace_j];
        const int64_t kv_off =
            static_cast<int64_t>(b) * kv_stride_b +
            static_cast<int64_t>(workspace_j) * kv_stride_s +
            static_cast<int64_t>(d_base) * kv_stride_d;
        const __nv_bfloat162* p2 =
            reinterpret_cast<const __nv_bfloat162*>(kv_workspace + kv_off);
        __nv_bfloat162 v01 = p2[0];
        __nv_bfloat162 v23 = p2[1];
        accum0 += s * __bfloat162float(v01.x);
        accum1 += s * __bfloat162float(v01.y);
        accum2 += s * __bfloat162float(v23.x);
        accum3 += s * __bfloat162float(v23.y);
    }

    const float row_lse = lse[static_cast<int64_t>(b) * num_heads + h];
    const float sink = attn_sink == nullptr ? 0.0f : attn_sink[h];
    const float gate =
        attn_sink == nullptr ? 1.0f : 1.0f / (1.0f + expf(-(row_lse - sink)));
    const int64_t out_bh =
        static_cast<int64_t>(b) * out_stride_b +
        static_cast<int64_t>(h) * out_stride_h;
    store_out_value<out_t>(
        out, out_bh + static_cast<int64_t>(d_base + 0) * out_stride_d,
        accum0 * gate);
    if (d_base + 1 < kHeadDim) {
        store_out_value<out_t>(
            out, out_bh + static_cast<int64_t>(d_base + 1) * out_stride_d,
            accum1 * gate);
    }
    if (d_base + 2 < kHeadDim) {
        store_out_value<out_t>(
            out, out_bh + static_cast<int64_t>(d_base + 2) * out_stride_d,
            accum2 * gate);
    }
    if (d_base + 3 < kHeadDim) {
        store_out_value<out_t>(
            out, out_bh + static_cast<int64_t>(d_base + 3) * out_stride_d,
            accum3 * gate);
    }
}

// Fused softmax + output kernel (stepping stone toward full flash-fusion).
//
// Reads RAW (un-softmaxed) scores from DRAM, computes max/exp/sum/lse in
// dynamic smem (size = candidate_slots * sizeof(float)), divides each
// smem slot by row_sum to produce probabilities, then runs the same
// per-thread PV loop as `sparse_mla_workspace_output_sstat_kernel`. lse
// is written by thread 0 of each (b, h, tile=0) block.
//
// Numerics: the softmax is computed inline (same -inf gating, same
// expf - row_max formula), then probabilities are normalized in smem
// before the PV multiply. Multiplication / accumulation order in the PV
// loop is preserved → bit-identical to softmax_kernel + sstat_kernel
// when the hardware exp/log matches (it does — same intrinsics).
//
// Grid: (batch * active_heads, ceil(kHeadDim/blockDim.x)) — same as the
// stand-alone sstat output kernel. Saves one kernel launch per
// (chunk × layer) and one fp32 DRAM round-trip on `scores` per (b, h).
template <typename kv_t, typename out_t>
__global__ void sparse_mla_workspace_output_fused_softmax_kernel(
    const kv_t* __restrict__ kv_workspace, const void* __restrict__ topk_length,
    const void* __restrict__ extra_topk_length,
    const float* __restrict__ scores, const float* __restrict__ attn_sink,
    out_t* __restrict__ out, float* __restrict__ lse, int batch_size,
    int active_heads, int num_heads, int main_topk, int extra_topk,
    int candidate_slots, int64_t kv_stride_b, int64_t kv_stride_s,
    int64_t kv_stride_d, int64_t out_stride_b, int64_t out_stride_h,
    int64_t out_stride_d, int topk_length_kind, int extra_topk_length_kind) {
    extern __shared__ float s_smem[];
    __shared__ float reduce_buf[kThreads];

    const int bh = blockIdx.x;
    const int tile = blockIdx.y;
    const int b = bh / active_heads;
    const int h = bh - b * active_heads;
    if (b >= batch_size) return;

    const int main_limit = max(
        0, min(main_topk,
               load_length_value(topk_length, topk_length_kind, b, main_topk)));
    const int extra_limit =
        max(0, min(extra_topk,
                   load_length_value(extra_topk_length, extra_topk_length_kind,
                                     b, extra_topk)));

    const int64_t score_base =
        (static_cast<int64_t>(b) * active_heads + h) * candidate_slots;

    // Pass 1: load raw scores into smem and find row_max.
    // (sstat path also stages scores once; we add the max/exp/sum work
    //  inline rather than reading post-softmax scores from DRAM.)
    float local_max = -INFINITY;
    for (int s = threadIdx.x; s < candidate_slots; s += blockDim.x) {
        const float v = scores[score_base + s];
        s_smem[s] = v;
        local_max = fmaxf(local_max, v);
    }
    reduce_buf[threadIdx.x] = local_max;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset)
            reduce_buf[threadIdx.x] =
                fmaxf(reduce_buf[threadIdx.x], reduce_buf[threadIdx.x + offset]);
        __syncthreads();
    }
    const float row_max = reduce_buf[0];

    // Pass 2: in-place exp(- row_max) on smem + sum.
    float local_sum = 0.0f;
    for (int s = threadIdx.x; s < candidate_slots; s += blockDim.x) {
        const float v = s_smem[s];
        const float p = isfinite(row_max) ? expf(v - row_max) : 0.0f;
        s_smem[s] = p;
        local_sum += p;
    }
    reduce_buf[threadIdx.x] = local_sum;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset)
            reduce_buf[threadIdx.x] += reduce_buf[threadIdx.x + offset];
        __syncthreads();
    }
    const float row_sum = reduce_buf[0];
    const float row_lse = row_sum > 0.0f ? logf(row_sum) + row_max : -INFINITY;

    // Pass 3: divide each smem slot by row_sum to get probabilities.
    // Match the stand-alone softmax_kernel's order of ops exactly so the
    // PV loop sees the same per-slot values as the existing path.
    for (int s = threadIdx.x; s < candidate_slots; s += blockDim.x) {
        s_smem[s] = row_sum > 0.0f ? (s_smem[s] / row_sum) : 0.0f;
    }
    __syncthreads();

    // Each (b, h, tile=0) block writes lse — match the softmax_kernel
    // contract so downstream code (e.g. the prefill workspace lse copy)
    // sees the right value.
    if (tile == 0 && threadIdx.x == 0)
        lse[static_cast<int64_t>(b) * num_heads + h] = row_lse;

    const int d = tile * blockDim.x + threadIdx.x;
    if (d >= kHeadDim) return;

    // Pass 4: PV loop, identical structure to sstat output kernel so
    // multiplication / accumulation order is preserved.
    float accum = 0.0f;
    for (int j = 0; j < main_limit; ++j) {
        accum += s_smem[j] *
                 load_workspace_value<kv_t>(
                     kv_workspace,
                     static_cast<int64_t>(b) * kv_stride_b +
                         static_cast<int64_t>(j) * kv_stride_s +
                         static_cast<int64_t>(d) * kv_stride_d);
    }
    for (int j = 0; j < extra_limit; ++j) {
        const int workspace_j = main_topk + j;
        accum += s_smem[workspace_j] *
                 load_workspace_value<kv_t>(
                     kv_workspace,
                     static_cast<int64_t>(b) * kv_stride_b +
                         static_cast<int64_t>(workspace_j) * kv_stride_s +
                         static_cast<int64_t>(d) * kv_stride_d);
    }

    const float sink = attn_sink == nullptr ? 0.0f : attn_sink[h];
    const float gate =
        attn_sink == nullptr ? 1.0f : 1.0f / (1.0f + expf(-(row_lse - sink)));
    store_out_value<out_t>(
        out, static_cast<int64_t>(b) * out_stride_b +
                 static_cast<int64_t>(h) * out_stride_h +
                 static_cast<int64_t>(d) * out_stride_d,
        accum * gate);
}

// ===== FLASH_FUSION =======================================================
// Full score+softmax+output fusion. One block per (b, h); K is loaded once
// from DRAM into smem and reused for both the QK score (FMA) phase and the
// PV output (FMA) phase. Total dynamic smem (candidate_slots ≤ 128,
// kHeadDim=512): Q[512]bf16 + K[128*512]bf16 + scores[128]fp32 ≈ 130 KB.
// Requires cudaFuncSetAttribute(MaxDynamicSmem) at launch.
//
// Math design:
//   - Score:    score[k] = softmax_scale * sum_d Q[d] * K[k,d]; -INF if invalid.
//   - Softmax:  identical formulas to sparse_mla_softmax_kernel.
//   - PV:       out[d] = sum_{k=0..candidate_slots} prob[k] * K[k,d]
//                       (invalid k contribute 0 because softmax wrote 0).
// Result is within ~1 bf16 ULP of the 3-kernel chain (different reduction
// order on QK partial-sums and PV accum order — math identical at op level).
//
// FMA prototype is capped at candidate_slots ≤ 64 because SM_120 per-CTA
// dynamic-smem max is ~99 KB and we need (1+N)*1024 + N*4 bytes (N=64 →
// ~66 KB; N=128 would need ~130 KB). The production shape mix uses topk=128,
// which needs a K-tiling variant (online-softmax over K chunks). Algorithm
// validation + parity vs the 3-kernel chain runs at N=64.
constexpr int kFlashTopkMax = 64;
// 2-bf16 (4-byte) row pad on K_smem. Phase-2 has 16 lane-pairs each reading
// row offset 0 of a different slot; with stride 1024 B (gcd(256, 32)=32) all
// 16 slots map to bank 0 → 16-way conflict. With stride 1028 B, slot k maps
// to bank (k * 257) mod 32 = k mod 32 → 16 unique banks → no conflict.
// kFlashKPad must (a) break smem bank conflicts on K_tile rows and
// (b) keep each row 16-byte aligned so ldmatrix.sync.aligned can be used by
// the prefill PV-MMA path. Pad of 8 bf16 = 16 bytes preserves both: 520 % 32
// banks ≠ 0 (still shifts banks) and 520 * 2 = 1040 bytes is a multiple of 16.
constexpr int kFlashKPad = 8;
constexpr int kFlashKStride = kHeadDim + kFlashKPad;

template <typename q_t, typename kv_t, typename out_t>
__global__ void sparse_mla_workspace_flash_kernel(
    const q_t* __restrict__ q, const kv_t* __restrict__ kv_workspace,
    const void* __restrict__ topk_length,
    const void* __restrict__ extra_topk_length,
    const float* __restrict__ attn_sink, out_t* __restrict__ out,
    float* __restrict__ lse, int batch_size, int active_heads, int num_heads,
    int main_topk, int extra_topk, int candidate_slots, int64_t q_stride_b,
    int64_t q_stride_h, int64_t q_stride_d, int64_t kv_stride_b,
    int64_t kv_stride_s, int64_t kv_stride_d, int64_t out_stride_b,
    int64_t out_stride_h, int64_t out_stride_d, int topk_length_kind,
    int extra_topk_length_kind, float softmax_scale) {
    extern __shared__ uint8_t flash_smem_raw[];
    __nv_bfloat16* Q_smem = reinterpret_cast<__nv_bfloat16*>(flash_smem_raw);
    __nv_bfloat16* K_smem = Q_smem + kHeadDim;
    float* scores_smem = reinterpret_cast<float*>(
        K_smem + static_cast<size_t>(candidate_slots) * kFlashKStride);
    __shared__ float reduce_buf[kThreads];

    const int b = blockIdx.x;
    const int h = blockIdx.y;
    if (b >= batch_size || h >= active_heads) return;

    const int main_limit = max(
        0, min(main_topk,
               load_length_value(topk_length, topk_length_kind, b, main_topk)));
    const int extra_limit =
        max(0, min(extra_topk,
                   load_length_value(extra_topk_length, extra_topk_length_kind,
                                     b, extra_topk)));

    const int tid = threadIdx.x;

    // Phase 0: cooperative load Q[h, :] into smem (head_dim=512, kThreads=256
    // → 2 elements per thread).
    const int64_t q_base = static_cast<int64_t>(b) * q_stride_b +
                           static_cast<int64_t>(h) * q_stride_h;
    for (int d = tid; d < kHeadDim; d += kThreads) {
        const float v = load_q_value<q_t>(
            q, q_base + static_cast<int64_t>(d) * q_stride_d);
        Q_smem[d] = __float2bfloat16_rn(v);
    }

    // Phase 1: cooperative load K[k, d] into smem with validity mask.
    // Invalid slots get K=0 so PV-loop contribution is 0 even if softmax
    // doesn't fully zero them (defense-in-depth against NaN in workspace).
    const int64_t kv_b_base = static_cast<int64_t>(b) * kv_stride_b;
    const int total_k = candidate_slots * kHeadDim;
    for (int idx = tid; idx < total_k; idx += kThreads) {
        const int k = idx / kHeadDim;
        const int d = idx - k * kHeadDim;
        const bool valid =
            k < main_limit || (k >= main_topk && k < main_topk + extra_limit);
        float v = 0.f;
        if (valid) {
            v = load_workspace_value<kv_t>(
                kv_workspace,
                kv_b_base + static_cast<int64_t>(k) * kv_stride_s +
                    static_cast<int64_t>(d) * kv_stride_d);
        }
        K_smem[static_cast<size_t>(k) * kFlashKStride + d] =
            __float2bfloat16_rn(v);
    }
    __syncthreads();

    // Phase 2: scores[k] = softmax_scale * dot(Q, K[k]). 2 threads per slot,
    // each handles head_dim/2 = 256 elements with stride 2; 2-way warp shfl.
    // 256 threads / 2 = 128 slots per round → fits candidate_slots ≤ 128.
    //
    // CRITICAL: __shfl_xor_sync(0xffffffff, ...) requires ALL 32 lanes in
    // the warp to call it with the same mask. We must NOT guard the shfl
    // by `slot < candidate_slots` or by validity, since either can split
    // a warp (e.g. lens=20, candidate=64 → warp 1 covers slots 16..31 with
    // mixed validity → deadlock). Solution: compute `partial` unconditionally
    // (loop is divergent-safe; invalid slots have K=0 from Phase 1 and OOR
    // slots get partial=0 because we skip the load), call shfl unconditionally
    // across all 256 threads, and branch only at the final store.
    {
        const int slot = tid >> 1;
        const int half = tid & 1;
        const bool in_range = slot < candidate_slots;
        const bool valid = in_range && (slot < main_limit ||
                                        (slot >= main_topk &&
                                         slot < main_topk + extra_limit));
        float partial = 0.f;
        if (in_range) {
            const __nv_bfloat16* k_row =
                K_smem + static_cast<size_t>(slot) * kFlashKStride;
            #pragma unroll 4
            for (int d = half; d < kHeadDim; d += 2) {
                partial += __bfloat162float(Q_smem[d]) *
                           __bfloat162float(k_row[d]);
            }
        }
        partial += __shfl_xor_sync(0xffffffffu, partial, 1, 32);
        if (in_range && half == 0) {
            scores_smem[slot] =
                valid ? partial * softmax_scale : -INFINITY;
        }
    }
    __syncthreads();

    // Phase 3: row_max reduction (mirrors sparse_mla_softmax_kernel).
    float local_max = -INFINITY;
    for (int j = tid; j < candidate_slots; j += kThreads)
        local_max = fmaxf(local_max, scores_smem[j]);
    reduce_buf[tid] = local_max;
    __syncthreads();
    for (int offset = kThreads / 2; offset > 0; offset >>= 1) {
        if (tid < offset)
            reduce_buf[tid] = fmaxf(reduce_buf[tid], reduce_buf[tid + offset]);
        __syncthreads();
    }
    const float row_max = reduce_buf[0];

    // Phase 4: in-place exp(- row_max) + row_sum.
    float local_sum = 0.f;
    for (int j = tid; j < candidate_slots; j += kThreads) {
        const float p =
            isfinite(row_max) ? expf(scores_smem[j] - row_max) : 0.f;
        scores_smem[j] = p;
        local_sum += p;
    }
    reduce_buf[tid] = local_sum;
    __syncthreads();
    for (int offset = kThreads / 2; offset > 0; offset >>= 1) {
        if (tid < offset) reduce_buf[tid] += reduce_buf[tid + offset];
        __syncthreads();
    }
    const float row_sum = reduce_buf[0];
    const float row_lse =
        row_sum > 0.f ? logf(row_sum) + row_max : -INFINITY;

    // Phase 5: divide by row_sum to get probabilities.
    for (int j = tid; j < candidate_slots; j += kThreads)
        scores_smem[j] = row_sum > 0.f ? scores_smem[j] / row_sum : 0.f;
    __syncthreads();

    // Write lse (one thread per (b, h)).
    if (tid == 0)
        lse[static_cast<int64_t>(b) * num_heads + h] = row_lse;

    // Phase 6: PV loop. Each thread owns kHeadDim/kThreads = 2 output dims.
    // Iterate the full padded slot range — invalid slots have prob=0 and
    // K=0, so they contribute 0. Order matches sstat output kernel within
    // the valid range; trailing zeros add no rounding noise.
    const float sink_val = attn_sink == nullptr ? 0.0f : attn_sink[h];
    const float gate = attn_sink == nullptr
                           ? 1.0f
                           : 1.0f / (1.0f + expf(-(row_lse - sink_val)));
    for (int d = tid; d < kHeadDim; d += kThreads) {
        float accum = 0.f;
        #pragma unroll 4
        for (int k = 0; k < candidate_slots; ++k) {
            accum += scores_smem[k] *
                     __bfloat162float(
                         K_smem[static_cast<size_t>(k) * kFlashKStride + d]);
        }
        store_out_value<out_t>(
            out, static_cast<int64_t>(b) * out_stride_b +
                     static_cast<int64_t>(h) * out_stride_h +
                     static_cast<int64_t>(d) * out_stride_d,
            accum * gate);
    }
}

// ===== MMA FLASH FUSION ===================================================
// Full score+softmax+output fusion using mma.sync.aligned.m16n8k16 for QK
// and PV. Successor to the scalar-FMA flash kernel above (which validated
// the algorithm but couldn't beat the chain's occupancy on this rig because
// 67 KB smem/CTA pinned us at 1 block/SM).
//
// Design (production active_heads=32, candidate_slots≤128, kHeadDim=512):
//   block tile  : M=16 (half of active_heads), TILE_K=32 (slot tile),
//                 K=512 (head_dim, MMA-streamed in 16-wide chunks).
//   block grid  : (batch, ceil(active_heads / kFlashMmaM)) = (B, 2).
//   warps/block : 4 — each warp owns a full M=16 fragment for one N=8
//                 sub-tile of TILE_K (4 warps × N=8 = TILE_K=32).
//   tiles/block : ceil(candidate_slots / TILE_K) — at topk=128 → 4 tiles
//                 traversed sequentially with online softmax merge.
//   threads     : 128.
//
// Per-block smem (fits in SM_120's ~99 KB per-CTA dynamic smem cap):
//   Q[M=16, kHeadDim=512]                bf16  = 16 KiB (loaded once)
//   K_tile[TILE_K=32, kHeadDim=512]      bf16  = 32 KiB (reloaded per tile)
//   K_tile_T[kHeadDim=512, TILE_K=32]    bf16  = 32 KiB (transpose for PV
//                                                — ldmatrix.trans avoids
//                                                this in v2; v1 builds it
//                                                in software during load)
//   O_accum[M=16, kHeadDim=512]          fp32  = 32 KiB (running output)
//   m[M=16], l[M=16]                     fp32  = 128 B
//   scores[M=16, TILE_K=32]              fp32  = 2 KiB (working buffer)
//   reduce_buf[kFlashMmaThreads=128]     fp32  = 512 B
//   Total                                       ≈ 115 KiB
//
// 115 KiB exceeds the SM_120 per-CTA cap. v1 mitigations:
//   (a) Drop K_tile_T entirely: store K only as [slot, d] for QK; for PV,
//       use ldmatrix.sync.aligned.m8n8.x4.trans to transpose-on-load into
//       MMA fragments. Smem drops to ~83 KiB. SUPPORTED on SM_120.
//   (b) Halve M to 8 (block grid becomes batch × 4 head_blocks). Loses
//       some MMA M-utilization since m16n8k16 needs M=16 — pad with zeros
//       or duplicate for two N-groups. Hurts occupancy gain less than
//       expected.
//   v1 picks (a). K_tile_T is NOT allocated; PV reads K_tile via ldmatrix.
//
// Online flash-attention softmax (per tile):
//   m_new[h]    = max(m_old[h], max_k score[h, k])
//   rescale[h]  = exp(m_old[h] - m_new[h])
//   P[h, k]     = exp(score[h, k] - m_new[h])  (fp32, then bf16 for PV MMA)
//   l_new[h]    = rescale[h] * l_old[h] + sum_k P[h, k]
//   O_accum[h, d] = rescale[h] * O_accum[h, d] + sum_k P[h, k] * K_tile[k, d]
//   Final: O[h, d] = O_accum[h, d] / l[h] * gate(lse[h] - sink[h])
//
// MMA fragment layout (m16n8k16 .row.col, copied verbatim from SCORE_MMA_
// SPLITK which is bit-validated via kScalarCheck):
//   - groupID  = lane >> 2     ∈ {0..7}, maps lane to N-row pair
//   - tid_in_g = lane & 3      ∈ {0..3}, maps lane to K-col pair
//   - A[2 rows × 2 K-pairs] = a0, a1, a2, a3 (one b32 per pair)
//   - B[1 N-row  × 2 K-pairs] = b0, b1
//   - D[2 rows × 2 N-cols]  = d0, d1, d2, d3 (fp32 accumulators)
//
// Numerical notes:
//   - QK and PV both run in fp32 accumulation (mma.f32.bf16.bf16.f32);
//     the scalar-FMA reference is bit-exact at this fragment layout in
//     SCORE_MMA_SPLITK.
//   - Online-softmax adds two ULP-class divergences: (1) per-tile reduction
//     order vs the chain's all-at-once softmax; (2) running fp32 O_accum
//     accumulation order differs from sstat output kernel. Expected sub-
//     bf16-ULP drift, like OUTPUT_FUSED_SOFTMAX (32/36 bit-exact, 4/36
//     sub-ULP). Validate via DG_SM120_FLASH_FUSION_MMA_SCALAR_CHECK=1.
//
// Activation: DG_SM120_FAST_SPARSE_MLA=1 + DG_SM120_FLASH_FUSION_MMA=1
// (default OFF). Falls through to the chain when batch_size × head_blocks
// is too small to fill the GPU (decode at b<48 stays on the chain).
//
// IMPLEMENTATION STATUS (v0 scaffolding):
//   Phase 0 (Q load)              : IMPLEMENTED
//   Phase 1 (init m, l, O_accum)  : IMPLEMENTED
//   Phase 2a (K_tile load)        : IMPLEMENTED
//   Phase 2b (QK MMA)             : IMPLEMENTED (m16n8k16, fp32 accum)
//   Phase 2c (scale + mask)       : IMPLEMENTED
//   Phase 2d-g (online softmax)   : TODO — needs per-row blockwide reduce,
//                                   rescale, exp, sum reduce
//   Phase 2h-i (PV via ldmatrix)  : TODO — needs ldmatrix.trans for K^T
//   Phase 3 (final O / l + gate)  : TODO
//   Dispatcher hook + env flag    : IMPLEMENTED (gated, falls through)
//   kScalarCheck oracle           : TODO (template flag, FMA reference)
//
// Until 2d-i + 3 are implemented, the kernel is dead-coded behind its env
// flag — DG_SM120_FLASH_FUSION_MMA defaults OFF so prod is unaffected.
constexpr int kFlashMmaM = 16;
constexpr int kFlashMmaN = 8;
constexpr int kFlashMmaK = 16;
constexpr int kFlashMmaTileK = 32;
constexpr int kFlashMmaNumWarps = kFlashMmaTileK / kFlashMmaN;  // = 4
constexpr int kFlashMmaThreads = 32 * kFlashMmaNumWarps;        // = 128
constexpr int kFlashMmaTopkMax = 128;
// K_tile uses the same row-pad as the FMA prototype to avoid 16-way bank
// conflicts on the cooperative load and on the QK-A fragment scratch reads.
constexpr int kFlashMmaKStride = kHeadDim + kFlashKPad;

// Split-K config: split candidate_slots across kFlashMmaSplitK partial blocks
// per (b, h_block). At kFlashMmaSplitK=4 and topk=128, each split owns
// candidate_slots/4 = 32 slots = exactly one K-tile (no inner-tile loop).
// Grid becomes (batch, h_block_count, split_k) → 4× more blocks → ~70% SM
// util at b=16 vs the 17% util of the single-block kernel.
constexpr int kFlashMmaSplitK = 4;
constexpr int kFlashMmaSplitStep = kFlashMmaTopkMax / kFlashMmaSplitK;  // 32

__host__ __forceinline__ bool sm120_flash_fusion_mma_splitk_enabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("DG_SM120_FLASH_FUSION_MMA_SPLITK");
        if (env == nullptr || env[0] == '\0') return false;
        return env[0] != '0';
    }();
    return enabled;
}

__host__ __forceinline__ bool sm120_flash_fusion_mma_scalar_check() {
    static const bool enabled = []() {
        const char* env = std::getenv("DG_SM120_FLASH_FUSION_MMA_SCALAR_CHECK");
        if (env == nullptr || env[0] == '\0') return false;
        return env[0] != '0';
    }();
    return enabled;
}

__host__ __forceinline__ bool sm120_flash_fusion_mma_enabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("DG_SM120_FLASH_FUSION_MMA");
        if (env == nullptr || env[0] == '\0') return false;
        return env[0] != '0';
    }();
    return enabled;
}

template <typename q_t, typename kv_t, typename out_t, bool kScalarCheck = false>
__global__ void sparse_mla_workspace_flash_mma_kernel(
    const q_t* __restrict__ q, const kv_t* __restrict__ kv_workspace,
    const void* __restrict__ topk_length,
    const void* __restrict__ extra_topk_length,
    const float* __restrict__ attn_sink, out_t* __restrict__ out,
    float* __restrict__ lse, int batch_size, int active_heads, int num_heads,
    int main_topk, int extra_topk, int candidate_slots, int64_t q_stride_b,
    int64_t q_stride_h, int64_t q_stride_d, int64_t kv_stride_b,
    int64_t kv_stride_s, int64_t kv_stride_d, int64_t out_stride_b,
    int64_t out_stride_h, int64_t out_stride_d, int topk_length_kind,
    int extra_topk_length_kind, float softmax_scale) {
    extern __shared__ uint8_t flash_mma_smem_raw[];
    // Layout: Q | K_tile | O_accum | scores | m | l
    __nv_bfloat16* Q_smem =
        reinterpret_cast<__nv_bfloat16*>(flash_mma_smem_raw);
    __nv_bfloat16* K_tile_smem = Q_smem + kFlashMmaM * kHeadDim;
    float* O_accum_smem = reinterpret_cast<float*>(
        K_tile_smem + static_cast<size_t>(kFlashMmaTileK) * kFlashMmaKStride);
    float* scores_smem =
        O_accum_smem + static_cast<size_t>(kFlashMmaM) * kHeadDim;
    float* m_smem = scores_smem + kFlashMmaM * kFlashMmaTileK;
    float* l_smem = m_smem + kFlashMmaM;
    float* tile_max_smem = l_smem + kFlashMmaM;
    float* rescale_smem = tile_max_smem + kFlashMmaM;
    float* tile_sum_smem = rescale_smem + kFlashMmaM;
    __shared__ float reduce_buf[kFlashMmaThreads];

    const int b = blockIdx.x;
    const int h_block = blockIdx.y;
    const int h_base = h_block * kFlashMmaM;
    if (b >= batch_size || h_base >= active_heads) return;

    const int main_limit = max(
        0, min(main_topk,
               load_length_value(topk_length, topk_length_kind, b, main_topk)));
    const int extra_limit =
        max(0, min(extra_topk,
                   load_length_value(extra_topk_length, extra_topk_length_kind,
                                     b, extra_topk)));

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int groupID = lane >> 2;
    const int tid_in_group = lane & 3;

    // ---------------------------------------------------------------------
    // Phase 0: cooperative load Q[h_base + m, :] into Q_smem[m, :].
    // 128 threads × 64 dims = M=16 × kHeadDim=512.
    // ---------------------------------------------------------------------
    const int64_t q_b_base = static_cast<int64_t>(b) * q_stride_b;
    for (int idx = tid; idx < kFlashMmaM * kHeadDim; idx += kFlashMmaThreads) {
        const int m = idx / kHeadDim;
        const int d = idx - m * kHeadDim;
        const int h = h_base + m;
        float v = 0.f;
        if (h < active_heads && h < num_heads) {
            v = load_q_value<q_t>(
                q, q_b_base + static_cast<int64_t>(h) * q_stride_h +
                       static_cast<int64_t>(d) * q_stride_d);
        }
        Q_smem[m * kHeadDim + d] = __float2bfloat16_rn(v);
    }

    // ---------------------------------------------------------------------
    // Phase 1: initialize m, l, O_accum.
    // m[h] = -inf, l[h] = 0, O_accum[h, d] = 0.
    // ---------------------------------------------------------------------
    for (int idx = tid; idx < kFlashMmaM; idx += kFlashMmaThreads) {
        m_smem[idx] = -INFINITY;
        l_smem[idx] = 0.f;
    }
    for (int idx = tid; idx < kFlashMmaM * kHeadDim; idx += kFlashMmaThreads) {
        O_accum_smem[idx] = 0.f;
    }
    __syncthreads();

    // ---------------------------------------------------------------------
    // Phase 2: outer loop over slot tiles.
    // ---------------------------------------------------------------------
    const int64_t kv_b_base = static_cast<int64_t>(b) * kv_stride_b;
    const int num_tiles =
        (candidate_slots + kFlashMmaTileK - 1) / kFlashMmaTileK;

    for (int tile = 0; tile < num_tiles; ++tile) {
        const int slot_base = tile * kFlashMmaTileK;

        // Phase 2a: cooperative load K_tile[k, d] = kv[slot_base+k, d].
        // Invalid slots zero-padded so QK-MMA produces 0 (final softmax mask
        // overwrites with -INFINITY on scores_smem before exp).
        for (int idx = tid; idx < kFlashMmaTileK * kHeadDim;
             idx += kFlashMmaThreads) {
            const int k_local = idx / kHeadDim;
            const int d = idx - k_local * kHeadDim;
            const int slot = slot_base + k_local;
            const bool in_main = slot < main_limit;
            const bool in_extra = slot >= main_topk &&
                                  slot < main_topk + extra_limit;
            float v = 0.f;
            if (slot < candidate_slots && (in_main || in_extra)) {
                v = load_workspace_value<kv_t>(
                    kv_workspace,
                    kv_b_base + static_cast<int64_t>(slot) * kv_stride_s +
                        static_cast<int64_t>(d) * kv_stride_d);
            }
            K_tile_smem[k_local * kFlashMmaKStride + d] =
                __float2bfloat16_rn(v);
        }
        __syncthreads();

        // Phase 2b: QK MMA. scores[M=16, TILE_K=32] = Q @ K_tile.T.
        //   - Each warp owns one N=8 sub-tile of TILE_K (warp 0 → N[0..7],
        //     warp 1 → N[8..15], warp 2 → N[16..23], warp 3 → N[24..31]).
        //   - K-axis (head_dim=512) streamed in 32 chunks of 16 elems each.
        //   - Per chunk: 1 mma.sync.m16n8k16, accumulating fp32 partials.
        // Fragment owners (per lane within warp):
        //   d_qk0/2 = scores[m_base + groupID,     n_base + 2*tid_in_group + {0,1}]
        //   d_qk1/3 = scores[m_base + groupID + 8, n_base + 2*tid_in_group + {0,1}]
        const int n_base = warp_id * kFlashMmaN;
        float d_qk0 = 0.f, d_qk1 = 0.f, d_qk2 = 0.f, d_qk3 = 0.f;

        #pragma unroll
        for (int kc = 0; kc < kHeadDim; kc += kFlashMmaK) {
            const int a_col_base = 2 * tid_in_group;
            // A fragment from Q_smem[h_row, kc..kc+15].
            const __nv_bfloat16* a_row0_ptr =
                Q_smem + groupID * kHeadDim + kc;
            const __nv_bfloat16* a_row1_ptr =
                Q_smem + (groupID + 8) * kHeadDim + kc;
            // B fragment from K_tile_smem[n_row, kc..kc+15] — kFlashMmaKStride
            // padded row to break Phase-2a writeback bank conflicts.
            const __nv_bfloat16* b_row_ptr =
                K_tile_smem + (n_base + groupID) * kFlashMmaKStride + kc;

            const uint32_t a0 = *reinterpret_cast<const uint32_t*>(
                a_row0_ptr + a_col_base);
            const uint32_t a1 = *reinterpret_cast<const uint32_t*>(
                a_row1_ptr + a_col_base);
            const uint32_t a2 = *reinterpret_cast<const uint32_t*>(
                a_row0_ptr + a_col_base + 8);
            const uint32_t a3 = *reinterpret_cast<const uint32_t*>(
                a_row1_ptr + a_col_base + 8);
            const uint32_t b0 = *reinterpret_cast<const uint32_t*>(
                b_row_ptr + a_col_base);
            const uint32_t b1 = *reinterpret_cast<const uint32_t*>(
                b_row_ptr + a_col_base + 8);

            if constexpr (kScalarCheck) {
                // Scalar-FMA oracle reading the SAME smem (validates the
                // mma.sync fragment layout). Bit-identical to the mma path.
                const int row0 = groupID;
                const int row1 = groupID + 8;
                const int col0 = 2 * tid_in_group;
                const int col1 = col0 + 1;
                float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
                #pragma unroll
                for (int kk = 0; kk < kFlashMmaK; ++kk) {
                    const float ar0 = __bfloat162float(
                        Q_smem[row0 * kHeadDim + kc + kk]);
                    const float ar1 = __bfloat162float(
                        Q_smem[row1 * kHeadDim + kc + kk]);
                    const float bn0 = __bfloat162float(
                        K_tile_smem[(n_base + col0) * kFlashMmaKStride + kc + kk]);
                    const float bn1 = __bfloat162float(
                        K_tile_smem[(n_base + col1) * kFlashMmaKStride + kc + kk]);
                    acc0 += ar0 * bn0;
                    acc1 += ar0 * bn1;
                    acc2 += ar1 * bn0;
                    acc3 += ar1 * bn1;
                }
                d_qk0 += acc0; d_qk1 += acc1;
                d_qk2 += acc2; d_qk3 += acc3;
                (void)a0; (void)a1; (void)a2; (void)a3;
                (void)b0; (void)b1;
            } else {
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                    "{%0, %1, %2, %3}, "
                    "{%4, %5, %6, %7}, "
                    "{%8, %9}, "
                    "{%0, %1, %2, %3};\n"
                    : "+f"(d_qk0), "+f"(d_qk1), "+f"(d_qk2), "+f"(d_qk3)
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                      "r"(b0), "r"(b1));
            }
        }

        // Phase 2c: apply softmax_scale + validity mask, write scores_smem.
        // Each lane owns 4 score cells; in-range/valid checks per cell.
        auto write_score = [&](int m, int n, float val) {
            if (m >= kFlashMmaM || n >= kFlashMmaTileK) return;
            const int slot = slot_base + n;
            const bool in_main = slot < main_limit;
            const bool in_extra = slot >= main_topk &&
                                  slot < main_topk + extra_limit;
            const bool valid = (slot < candidate_slots) && (in_main || in_extra);
            scores_smem[m * kFlashMmaTileK + n] =
                valid ? val * softmax_scale : -INFINITY;
        };
        const int m0 = groupID;
        const int m1 = groupID + 8;
        const int n0 = n_base + 2 * tid_in_group;
        const int n1 = n0 + 1;
        write_score(m0, n0, d_qk0);
        write_score(m0, n1, d_qk1);
        write_score(m1, n0, d_qk2);
        write_score(m1, n1, d_qk3);
        __syncthreads();

        // Phase 2d: per-row max reduction over scores_smem[h, 0..32).
        // 4 warps × 4 row-iterations = 16 rows. Within a warp, lane reads
        // scores[h, lane] and reduces via warp-shfl butterfly.
        #pragma unroll
        for (int rsub = 0; rsub < 4; ++rsub) {
            const int h = warp_id * 4 + rsub;
            float v = scores_smem[h * kFlashMmaTileK + lane];
            #pragma unroll
            for (int s = 16; s > 0; s >>= 1) {
                v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, s));
            }
            if (lane == 0) tile_max_smem[h] = v;
        }
        __syncthreads();

        // Phase 2e: m_new[h] = max(m[h], tile_max[h]); rescale[h] = exp(m_old - m_new).
        if (tid < kFlashMmaM) {
            const int h = tid;
            const float m_old = m_smem[h];
            const float tm = tile_max_smem[h];
            const float m_new = fmaxf(m_old, tm);
            // rescale = exp(m_old - m_new); if m_old=-inf and m_new finite, exp=0.
            // If both -inf (entire history empty AND tile all -inf), m_old-m_new=NaN
            // → guard.
            const float diff = (isfinite(m_old) && isfinite(m_new))
                                   ? (m_old - m_new)
                                   : (isfinite(m_new) ? -INFINITY : 0.f);
            rescale_smem[h] = expf(diff);
            m_smem[h] = m_new;
        }
        __syncthreads();

        // Phase 2f: P[h, k] = exp(scores[h, k] - m_new[h]) (overwrite scores_smem
        // in-place, kept fp32 for scalar-PV reuse). Per-row sum reduction.
        #pragma unroll
        for (int rsub = 0; rsub < 4; ++rsub) {
            const int h = warp_id * 4 + rsub;
            const float m_new = m_smem[h];
            const int k = lane;
            const float s_old = scores_smem[h * kFlashMmaTileK + k];
            // -INFINITY masked entries → exp = 0.
            const float p =
                (isfinite(s_old) && isfinite(m_new)) ? expf(s_old - m_new) : 0.f;
            scores_smem[h * kFlashMmaTileK + k] = p;
            float sum = p;
            #pragma unroll
            for (int sh = 16; sh > 0; sh >>= 1) {
                sum += __shfl_xor_sync(0xffffffffu, sum, sh);
            }
            if (lane == 0) tile_sum_smem[h] = sum;
        }
        __syncthreads();

        // Phase 2g: l[h] = rescale[h] * l[h] + tile_sum[h].
        if (tid < kFlashMmaM) {
            const int h = tid;
            l_smem[h] = rescale_smem[h] * l_smem[h] + tile_sum_smem[h];
        }
        __syncthreads();

        // Phase 2h: O_accum[h, d] *= rescale[h].
        for (int idx = tid; idx < kFlashMmaM * kHeadDim;
             idx += kFlashMmaThreads) {
            const int m = idx / kHeadDim;
            O_accum_smem[idx] *= rescale_smem[m];
        }
        __syncthreads();

        // Phase 2i: scalar PV. O_accum[h, d] += sum_k P[h, k] * K_tile[k, d].
        // 16 × 512 = 8192 cells, K=32 inner reduction. 64 cells/thread @ 128 thr.
        // (PV-MMA upgrade is a follow-up: needs ldmatrix.trans for B fragment
        //  to read K_tile in [d, slot] orientation without a second smem copy.)
        for (int idx = tid; idx < kFlashMmaM * kHeadDim;
             idx += kFlashMmaThreads) {
            const int m = idx / kHeadDim;
            const int d = idx - m * kHeadDim;
            float acc = 0.f;
            #pragma unroll
            for (int kk = 0; kk < kFlashMmaTileK; ++kk) {
                acc += scores_smem[m * kFlashMmaTileK + kk] *
                       __bfloat162float(
                           K_tile_smem[kk * kFlashMmaKStride + d]);
            }
            O_accum_smem[idx] += acc;
        }
        __syncthreads();
    }

    // ---------------------------------------------------------------------
    // Phase 3: finalize.
    //   lse[h]  = log(l[h]) + m[h]                  (or -INFINITY if l <= 0)
    //   gate[h] = sigmoid(lse[h] - sink[h])         (or 1.0 if attn_sink == nullptr)
    //   out[b, h, d] = O_accum[h, d] / l[h] * gate[h]
    // We stash gate/l into rescale_smem (reused as final-scale) so the
    // output writeback is a single multiply per cell.
    // ---------------------------------------------------------------------
    if (tid < kFlashMmaM) {
        const int h = h_base + tid;
        const float l_val = l_smem[tid];
        const float m_val = m_smem[tid];
        const float lse_v =
            (l_val > 0.f && isfinite(m_val)) ? (logf(l_val) + m_val) : -INFINITY;
        if (h < num_heads && lse != nullptr) {
            lse[static_cast<int64_t>(b) * num_heads + h] = lse_v;
        }
        float gate = 1.f;
        if (attn_sink != nullptr && h < num_heads) {
            const float sink_v = attn_sink[h];
            const float x = lse_v - sink_v;
            // sigmoid(-INFINITY - finite) = 0; guard the exp.
            gate = isfinite(x) ? (1.f / (1.f + expf(-x))) : (lse_v > 0.f ? 1.f : 0.f);
        }
        rescale_smem[tid] = (l_val > 0.f) ? (gate / l_val) : 0.f;
    }
    __syncthreads();

    for (int idx = tid; idx < kFlashMmaM * kHeadDim;
         idx += kFlashMmaThreads) {
        const int m = idx / kHeadDim;
        const int d = idx - m * kHeadDim;
        const int h = h_base + m;
        if (h < active_heads) {
            const float v = O_accum_smem[idx] * rescale_smem[m];
            store_out_value<out_t>(
                out, static_cast<int64_t>(b) * out_stride_b +
                         static_cast<int64_t>(h) * out_stride_h +
                         static_cast<int64_t>(d) * out_stride_d,
                v);
        }
    }
}

// ===== FLASH_FUSION_MMA SPLIT-K ============================================
// Split candidate_slots across kFlashMmaSplitK=4 blocks per (b, h_block).
// Each partial block accumulates (m, l, O) over its 32-slot slice, writes
// to gmem partial buffers; reduce kernel merges via online-softmax combine.
// Goal: lift grid from 32 (b=16, h_block=2) to 128 blocks (~70% SM util).
//
// Partial buffer layout (allocated by dispatcher):
//   O_partial[batch, h_block_count, split_k, M=16, head_dim]  fp32
//   m_partial[batch, h_block_count, split_k, M=16]            fp32
//   l_partial[batch, h_block_count, split_k, M=16]            fp32
//
// Activation: DG_SM120_FAST_SPARSE_MLA=1 + DG_SM120_FLASH_FUSION_MMA_SPLITK=1.
template <typename q_t, typename kv_t, bool kScalarCheck = false>
__global__ void sparse_mla_workspace_flash_mma_splitk_partial_kernel(
    const q_t* __restrict__ q, const kv_t* __restrict__ kv_workspace,
    const void* __restrict__ topk_length,
    const void* __restrict__ extra_topk_length,
    float* __restrict__ O_partial, float* __restrict__ m_partial,
    float* __restrict__ l_partial, int batch_size, int active_heads,
    int num_heads, int main_topk, int extra_topk, int candidate_slots,
    int h_block_count, int split_k, int split_step, int64_t q_stride_b,
    int64_t q_stride_h, int64_t q_stride_d, int64_t kv_stride_b,
    int64_t kv_stride_s, int64_t kv_stride_d, int topk_length_kind,
    int extra_topk_length_kind, float softmax_scale) {
    extern __shared__ uint8_t flash_mma_spk_smem_raw[];
    __nv_bfloat16* Q_smem =
        reinterpret_cast<__nv_bfloat16*>(flash_mma_spk_smem_raw);
    __nv_bfloat16* K_tile_smem = Q_smem + kFlashMmaM * kHeadDim;
    float* O_accum_smem = reinterpret_cast<float*>(
        K_tile_smem + static_cast<size_t>(kFlashMmaTileK) * kFlashMmaKStride);
    float* scores_smem =
        O_accum_smem + static_cast<size_t>(kFlashMmaM) * kHeadDim;
    float* m_smem = scores_smem + kFlashMmaM * kFlashMmaTileK;
    float* l_smem = m_smem + kFlashMmaM;
    float* tile_max_smem = l_smem + kFlashMmaM;
    float* rescale_smem = tile_max_smem + kFlashMmaM;
    float* tile_sum_smem = rescale_smem + kFlashMmaM;

    const int b = blockIdx.x;
    const int h_block = blockIdx.y;
    const int split_id = blockIdx.z;
    const int h_base = h_block * kFlashMmaM;
    if (b >= batch_size || h_base >= active_heads || split_id >= split_k) return;

    const int main_limit = max(
        0, min(main_topk,
               load_length_value(topk_length, topk_length_kind, b, main_topk)));
    const int extra_limit =
        max(0, min(extra_topk,
                   load_length_value(extra_topk_length, extra_topk_length_kind,
                                     b, extra_topk)));

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int groupID = lane >> 2;
    const int tid_in_group = lane & 3;

    const int slot_lo = split_id * split_step;
    const int slot_hi = min(slot_lo + split_step, candidate_slots);

    // Phase 0: load Q[h_base + m, :] into Q_smem[m, :].
    const int64_t q_b_base = static_cast<int64_t>(b) * q_stride_b;
    for (int idx = tid; idx < kFlashMmaM * kHeadDim; idx += kFlashMmaThreads) {
        const int m = idx / kHeadDim;
        const int d = idx - m * kHeadDim;
        const int h = h_base + m;
        float v = 0.f;
        if (h < active_heads && h < num_heads) {
            v = load_q_value<q_t>(
                q, q_b_base + static_cast<int64_t>(h) * q_stride_h +
                       static_cast<int64_t>(d) * q_stride_d);
        }
        Q_smem[m * kHeadDim + d] = __float2bfloat16_rn(v);
    }

    // Phase 1: init m=-inf, l=0, O=0.
    for (int idx = tid; idx < kFlashMmaM; idx += kFlashMmaThreads) {
        m_smem[idx] = -INFINITY;
        l_smem[idx] = 0.f;
    }
    for (int idx = tid; idx < kFlashMmaM * kHeadDim; idx += kFlashMmaThreads) {
        O_accum_smem[idx] = 0.f;
    }
    __syncthreads();

    // Phase 2: outer loop over slot tiles within this split's range.
    // At kFlashMmaSplitK=4 and topk<=128, split_step<=32 so num_tiles<=1 in
    // the common case — but we keep the loop for correctness when callers
    // pass step > 32.
    const int64_t kv_b_base = static_cast<int64_t>(b) * kv_stride_b;
    const int slot_span = max(0, slot_hi - slot_lo);
    const int num_tiles = (slot_span + kFlashMmaTileK - 1) / kFlashMmaTileK;

    for (int tile = 0; tile < num_tiles; ++tile) {
        const int tile_slot_base = slot_lo + tile * kFlashMmaTileK;

        // Phase 2a: load K_tile.
        for (int idx = tid; idx < kFlashMmaTileK * kHeadDim;
             idx += kFlashMmaThreads) {
            const int k_local = idx / kHeadDim;
            const int d = idx - k_local * kHeadDim;
            const int slot = tile_slot_base + k_local;
            const bool in_main = slot < main_limit;
            const bool in_extra = slot >= main_topk &&
                                  slot < main_topk + extra_limit;
            float v = 0.f;
            if (slot < slot_hi && slot < candidate_slots && (in_main || in_extra)) {
                v = load_workspace_value<kv_t>(
                    kv_workspace,
                    kv_b_base + static_cast<int64_t>(slot) * kv_stride_s +
                        static_cast<int64_t>(d) * kv_stride_d);
            }
            K_tile_smem[k_local * kFlashMmaKStride + d] =
                __float2bfloat16_rn(v);
        }
        __syncthreads();

        // Phase 2b: QK MMA, identical to the non-split kernel.
        const int n_base = warp_id * kFlashMmaN;
        float d_qk0 = 0.f, d_qk1 = 0.f, d_qk2 = 0.f, d_qk3 = 0.f;
        #pragma unroll
        for (int kc = 0; kc < kHeadDim; kc += kFlashMmaK) {
            const int a_col_base = 2 * tid_in_group;
            const __nv_bfloat16* a_row0_ptr = Q_smem + groupID * kHeadDim + kc;
            const __nv_bfloat16* a_row1_ptr =
                Q_smem + (groupID + 8) * kHeadDim + kc;
            const __nv_bfloat16* b_row_ptr =
                K_tile_smem + (n_base + groupID) * kFlashMmaKStride + kc;
            const uint32_t a0 = *reinterpret_cast<const uint32_t*>(
                a_row0_ptr + a_col_base);
            const uint32_t a1 = *reinterpret_cast<const uint32_t*>(
                a_row1_ptr + a_col_base);
            const uint32_t a2 = *reinterpret_cast<const uint32_t*>(
                a_row0_ptr + a_col_base + 8);
            const uint32_t a3 = *reinterpret_cast<const uint32_t*>(
                a_row1_ptr + a_col_base + 8);
            const uint32_t b0 = *reinterpret_cast<const uint32_t*>(
                b_row_ptr + a_col_base);
            const uint32_t b1 = *reinterpret_cast<const uint32_t*>(
                b_row_ptr + a_col_base + 8);
            if constexpr (kScalarCheck) {
                const int row0 = groupID;
                const int row1 = groupID + 8;
                const int col0 = 2 * tid_in_group;
                const int col1 = col0 + 1;
                float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
                #pragma unroll
                for (int kk = 0; kk < kFlashMmaK; ++kk) {
                    const float ar0 = __bfloat162float(
                        Q_smem[row0 * kHeadDim + kc + kk]);
                    const float ar1 = __bfloat162float(
                        Q_smem[row1 * kHeadDim + kc + kk]);
                    const float bn0 = __bfloat162float(
                        K_tile_smem[(n_base + col0) * kFlashMmaKStride + kc + kk]);
                    const float bn1 = __bfloat162float(
                        K_tile_smem[(n_base + col1) * kFlashMmaKStride + kc + kk]);
                    acc0 += ar0 * bn0;
                    acc1 += ar0 * bn1;
                    acc2 += ar1 * bn0;
                    acc3 += ar1 * bn1;
                }
                d_qk0 += acc0; d_qk1 += acc1; d_qk2 += acc2; d_qk3 += acc3;
                (void)a0; (void)a1; (void)a2; (void)a3; (void)b0; (void)b1;
            } else {
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                    "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
                    "{%0, %1, %2, %3};\n"
                    : "+f"(d_qk0), "+f"(d_qk1), "+f"(d_qk2), "+f"(d_qk3)
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
            }
        }

        // Phase 2c: scale + validity mask, write scores_smem.
        auto write_score = [&](int m, int n, float val) {
            if (m >= kFlashMmaM || n >= kFlashMmaTileK) return;
            const int slot = tile_slot_base + n;
            const bool in_main = slot < main_limit;
            const bool in_extra = slot >= main_topk &&
                                  slot < main_topk + extra_limit;
            const bool valid = (slot < slot_hi) && (slot < candidate_slots) &&
                               (in_main || in_extra);
            scores_smem[m * kFlashMmaTileK + n] =
                valid ? val * softmax_scale : -INFINITY;
        };
        const int m0 = groupID;
        const int m1 = groupID + 8;
        const int n0 = n_base + 2 * tid_in_group;
        const int n1 = n0 + 1;
        write_score(m0, n0, d_qk0);
        write_score(m0, n1, d_qk1);
        write_score(m1, n0, d_qk2);
        write_score(m1, n1, d_qk3);
        __syncthreads();

        // Phase 2d: per-row max.
        #pragma unroll
        for (int rsub = 0; rsub < 4; ++rsub) {
            const int h = warp_id * 4 + rsub;
            float v = scores_smem[h * kFlashMmaTileK + lane];
            #pragma unroll
            for (int s = 16; s > 0; s >>= 1) {
                v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, s));
            }
            if (lane == 0) tile_max_smem[h] = v;
        }
        __syncthreads();

        // Phase 2e: m_new + rescale.
        if (tid < kFlashMmaM) {
            const int h = tid;
            const float m_old = m_smem[h];
            const float tm = tile_max_smem[h];
            const float m_new = fmaxf(m_old, tm);
            const float diff = (isfinite(m_old) && isfinite(m_new))
                                   ? (m_old - m_new)
                                   : (isfinite(m_new) ? -INFINITY : 0.f);
            rescale_smem[h] = expf(diff);
            m_smem[h] = m_new;
        }
        __syncthreads();

        // Phase 2f: P + sum.
        #pragma unroll
        for (int rsub = 0; rsub < 4; ++rsub) {
            const int h = warp_id * 4 + rsub;
            const float m_new = m_smem[h];
            const int k = lane;
            const float s_old = scores_smem[h * kFlashMmaTileK + k];
            const float p =
                (isfinite(s_old) && isfinite(m_new)) ? expf(s_old - m_new) : 0.f;
            scores_smem[h * kFlashMmaTileK + k] = p;
            float sum = p;
            #pragma unroll
            for (int sh = 16; sh > 0; sh >>= 1) {
                sum += __shfl_xor_sync(0xffffffffu, sum, sh);
            }
            if (lane == 0) tile_sum_smem[h] = sum;
        }
        __syncthreads();

        // Phase 2g: l update.
        if (tid < kFlashMmaM) {
            const int h = tid;
            l_smem[h] = rescale_smem[h] * l_smem[h] + tile_sum_smem[h];
        }
        __syncthreads();

        // Phase 2h: rescale O_accum.
        for (int idx = tid; idx < kFlashMmaM * kHeadDim;
             idx += kFlashMmaThreads) {
            const int m = idx / kHeadDim;
            O_accum_smem[idx] *= rescale_smem[m];
        }
        __syncthreads();

        // Phase 2i: scalar PV.
        for (int idx = tid; idx < kFlashMmaM * kHeadDim;
             idx += kFlashMmaThreads) {
            const int m = idx / kHeadDim;
            const int d = idx - m * kHeadDim;
            float acc = 0.f;
            #pragma unroll
            for (int kk = 0; kk < kFlashMmaTileK; ++kk) {
                acc += scores_smem[m * kFlashMmaTileK + kk] *
                       __bfloat162float(
                           K_tile_smem[kk * kFlashMmaKStride + d]);
            }
            O_accum_smem[idx] += acc;
        }
        __syncthreads();
    }

    // Phase 3 (partial-write): emit (m, l, O) to gmem so the reduce kernel
    // can merge across splits via online-softmax combine.
    // Layout: [batch, h_block, split, M, ...] linearized.
    const int64_t partial_idx_base =
        ((static_cast<int64_t>(b) * h_block_count + h_block) * split_k +
         split_id);
    if (tid < kFlashMmaM) {
        const int m = tid;
        m_partial[partial_idx_base * kFlashMmaM + m] = m_smem[m];
        l_partial[partial_idx_base * kFlashMmaM + m] = l_smem[m];
    }
    for (int idx = tid; idx < kFlashMmaM * kHeadDim; idx += kFlashMmaThreads) {
        O_partial[partial_idx_base * kFlashMmaM * kHeadDim + idx] =
            O_accum_smem[idx];
    }
}

// Reduce kernel: for each (b, h_block), merge SPLIT_K partials.
//   m_global   = max_s m_partial[s, m]
//   rescale[s] = exp(m_partial[s, m] - m_global)
//   l_global   = sum_s rescale[s] * l_partial[s, m]
//   lse        = log(l_global) + m_global  (or -INF if l_global<=0)
//   gate       = sigmoid(lse - sink) if attn_sink else 1
//   O_global[d]= sum_s rescale[s] * O_partial[s, m, d] * (gate / l_global)
template <typename out_t>
__global__ void sparse_mla_workspace_flash_mma_splitk_reduce_kernel(
    const float* __restrict__ O_partial, const float* __restrict__ m_partial,
    const float* __restrict__ l_partial, const float* __restrict__ attn_sink,
    out_t* __restrict__ out, float* __restrict__ lse, int batch_size,
    int active_heads, int num_heads, int h_block_count, int split_k,
    int64_t out_stride_b, int64_t out_stride_h, int64_t out_stride_d) {
    extern __shared__ uint8_t flash_mma_spk_red_smem_raw[];
    float* m_global_smem =
        reinterpret_cast<float*>(flash_mma_spk_red_smem_raw);
    float* l_global_smem = m_global_smem + kFlashMmaM;
    float* gate_over_l_smem = l_global_smem + kFlashMmaM;
    float* rescale_smem = gate_over_l_smem + kFlashMmaM;  // [SPLIT_K, M]

    const int b = blockIdx.x;
    const int h_block = blockIdx.y;
    const int h_base = h_block * kFlashMmaM;
    if (b >= batch_size || h_base >= active_heads) return;

    const int tid = threadIdx.x;
    const int64_t base_idx =
        (static_cast<int64_t>(b) * h_block_count + h_block) * split_k;

    // Step 1: compute m_global per row. Each thread handles one (s, m) cell.
    // SPLIT_K * M = 4 * 16 = 64 cells. With 128 threads, each does 0 or 1.
    if (tid < kFlashMmaM) {
        const int m = tid;
        float m_g = -INFINITY;
        #pragma unroll
        for (int s = 0; s < kFlashMmaSplitK; ++s) {
            if (s < split_k) {
                const float v =
                    m_partial[(base_idx + s) * kFlashMmaM + m];
                m_g = fmaxf(m_g, v);
            }
        }
        m_global_smem[m] = m_g;
    }
    __syncthreads();

    // Step 2: rescale[s][m] = exp(m_partial[s,m] - m_global[m]); l_global[m].
    if (tid < kFlashMmaSplitK * kFlashMmaM) {
        const int s = tid / kFlashMmaM;
        const int m = tid - s * kFlashMmaM;
        if (s < split_k) {
            const float mg = m_global_smem[m];
            const float mp = m_partial[(base_idx + s) * kFlashMmaM + m];
            const float diff = (isfinite(mp) && isfinite(mg))
                                   ? (mp - mg)
                                   : (isfinite(mg) ? -INFINITY : 0.f);
            rescale_smem[s * kFlashMmaM + m] = expf(diff);
        } else {
            rescale_smem[s * kFlashMmaM + m] = 0.f;
        }
    }
    __syncthreads();

    if (tid < kFlashMmaM) {
        const int m = tid;
        float lg = 0.f;
        #pragma unroll
        for (int s = 0; s < kFlashMmaSplitK; ++s) {
            if (s < split_k) {
                const float r = rescale_smem[s * kFlashMmaM + m];
                const float lp = l_partial[(base_idx + s) * kFlashMmaM + m];
                lg += r * lp;
            }
        }
        l_global_smem[m] = lg;

        const int h = h_base + m;
        const float mg = m_global_smem[m];
        const float lse_v =
            (lg > 0.f && isfinite(mg)) ? (logf(lg) + mg) : -INFINITY;
        if (h < num_heads && lse != nullptr) {
            lse[static_cast<int64_t>(b) * num_heads + h] = lse_v;
        }
        float gate = 1.f;
        if (attn_sink != nullptr && h < num_heads) {
            const float sink_v = attn_sink[h];
            const float x = lse_v - sink_v;
            gate = isfinite(x) ? (1.f / (1.f + expf(-x)))
                               : (lse_v > 0.f ? 1.f : 0.f);
        }
        gate_over_l_smem[m] = (lg > 0.f) ? (gate / lg) : 0.f;
    }
    __syncthreads();

    // Step 3: O_global[m, d] = (sum_s rescale[s][m] * O_partial[s, m, d]) * (gate/l)
    for (int idx = tid; idx < kFlashMmaM * kHeadDim; idx += kFlashMmaThreads) {
        const int m = idx / kHeadDim;
        const int d = idx - m * kHeadDim;
        const int h = h_base + m;
        float acc = 0.f;
        #pragma unroll
        for (int s = 0; s < kFlashMmaSplitK; ++s) {
            if (s < split_k) {
                const float r = rescale_smem[s * kFlashMmaM + m];
                const float v =
                    O_partial[((base_idx + s) * kFlashMmaM + m) * kHeadDim + d];
                acc += r * v;
            }
        }
        const float final_v = acc * gate_over_l_smem[m];
        if (h < active_heads) {
            store_out_value<out_t>(
                out, static_cast<int64_t>(b) * out_stride_b +
                         static_cast<int64_t>(h) * out_stride_h +
                         static_cast<int64_t>(d) * out_stride_d,
                final_v);
        }
    }
}

// ===== OUTPUT MMA =========================================================
// Layout (production active_heads=32, candidate_slots≤128, kHeadDim=512):
//   block tile  : M=32 (all active heads), N=8 (one dim slice), K=128
//                 (full slot reduction, padded with zeros if topk<128).
//   block grid  : (batch, ceil(kHeadDim/8), 1) — at B=8, kHeadDim=512
//                 → 8 × 64 = 512 blocks vs 188 SMs (~3 waves).
//   warps/block : 2  (warp 0 → M[0..15], warp 1 → M[16..31]).
//   smem        : score[M=32, K=128] bf16  = 8 KB
//                 + kv [N=8,  K=128] bf16  = 2 KB → 10 KB.
//
// Iterates the full padded slot range — softmax already wrote -inf → 0 for
// invalid candidates so no validity mask is needed in this kernel. Result
// gating gate(b, h) = sigmoid(lse - sink) is applied at store time.
constexpr int kOutMmaM = 32;
constexpr int kOutMmaN = 8;
constexpr int kOutMmaK = 16;
constexpr int kOutMmaKMax = 128;
constexpr int kOutMmaNumWarps = kOutMmaM / 16;
constexpr int kOutMmaThreads = 32 * kOutMmaNumWarps;

template <typename kv_t, typename out_t, bool kScalarCheck>
__global__ void sparse_mla_workspace_output_mma_kernel(
    const kv_t* __restrict__ kv_workspace,
    const float* __restrict__ scores, const float* __restrict__ lse,
    const float* __restrict__ attn_sink, out_t* __restrict__ out,
    int batch_size, int active_heads, int num_heads, int candidate_slots,
    int64_t kv_stride_b, int64_t kv_stride_s, int64_t kv_stride_d,
    int64_t out_stride_b, int64_t out_stride_h, int64_t out_stride_d) {
    const int b = blockIdx.x;
    const int n_tile = blockIdx.y;
    const int d_base = n_tile * kOutMmaN;
    if (b >= batch_size || d_base >= kHeadDim) return;

    __shared__ __nv_bfloat16 score_smem[kOutMmaM * kOutMmaKMax];
    __shared__ __nv_bfloat16 kv_smem[kOutMmaN * kOutMmaKMax];

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int groupID = lane >> 2;
    const int tid_in_group = lane & 3;
    const int m_warp_base = warp_id * 16;

    const int64_t score_b_base =
        static_cast<int64_t>(b) * active_heads * candidate_slots;
    for (int idx = tid; idx < kOutMmaM * kOutMmaKMax; idx += kOutMmaThreads) {
        const int h = idx / kOutMmaKMax;
        const int j = idx - h * kOutMmaKMax;
        float v = 0.f;
        if (h < active_heads && j < candidate_slots) {
            v = scores[score_b_base + h * candidate_slots + j];
        }
        score_smem[h * kOutMmaKMax + j] = __float2bfloat16_rn(v);
    }

    const int64_t kv_b_base = static_cast<int64_t>(b) * kv_stride_b;
    for (int idx = tid; idx < kOutMmaN * kOutMmaKMax; idx += kOutMmaThreads) {
        const int n = idx / kOutMmaKMax;
        const int j = idx - n * kOutMmaKMax;
        const int d = d_base + n;
        float v = 0.f;
        if (j < candidate_slots && d < kHeadDim) {
            const int64_t off = kv_b_base +
                                static_cast<int64_t>(j) * kv_stride_s +
                                static_cast<int64_t>(d) * kv_stride_d;
            v = load_workspace_value<kv_t>(kv_workspace, off);
        }
        kv_smem[n * kOutMmaKMax + j] = __float2bfloat16_rn(v);
    }
    __syncthreads();

    float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
    #pragma unroll
    for (int kc = 0; kc < kOutMmaKMax; kc += kOutMmaK) {
        const int a_col_base = 2 * tid_in_group;
        const __nv_bfloat16* a_row0_ptr =
            score_smem + (m_warp_base + groupID) * kOutMmaKMax + kc;
        const __nv_bfloat16* a_row1_ptr =
            score_smem + (m_warp_base + groupID + 8) * kOutMmaKMax + kc;
        const __nv_bfloat16* b_row_ptr =
            kv_smem + groupID * kOutMmaKMax + kc;

        const uint32_t a0 = *reinterpret_cast<const uint32_t*>(
            a_row0_ptr + a_col_base);
        const uint32_t a1 = *reinterpret_cast<const uint32_t*>(
            a_row1_ptr + a_col_base);
        const uint32_t a2 = *reinterpret_cast<const uint32_t*>(
            a_row0_ptr + a_col_base + 8);
        const uint32_t a3 = *reinterpret_cast<const uint32_t*>(
            a_row1_ptr + a_col_base + 8);
        const uint32_t b0 = *reinterpret_cast<const uint32_t*>(
            b_row_ptr + a_col_base);
        const uint32_t b1 = *reinterpret_cast<const uint32_t*>(
            b_row_ptr + a_col_base + 8);

        if constexpr (kScalarCheck) {
            const int row0 = m_warp_base + groupID;
            const int row1 = m_warp_base + groupID + 8;
            const int col0 = 2 * tid_in_group;
            const int col1 = col0 + 1;
            float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
            #pragma unroll
            for (int kk = 0; kk < kOutMmaK; ++kk) {
                const float ar0 = __bfloat162float(
                    score_smem[row0 * kOutMmaKMax + kc + kk]);
                const float ar1 = __bfloat162float(
                    score_smem[row1 * kOutMmaKMax + kc + kk]);
                const float bn0 = __bfloat162float(
                    kv_smem[col0 * kOutMmaKMax + kc + kk]);
                const float bn1 = __bfloat162float(
                    kv_smem[col1 * kOutMmaKMax + kc + kk]);
                acc0 += ar0 * bn0;
                acc1 += ar0 * bn1;
                acc2 += ar1 * bn0;
                acc3 += ar1 * bn1;
            }
            d0 += acc0; d1 += acc1; d2 += acc2; d3 += acc3;
            (void)a0; (void)a1; (void)a2; (void)a3;
            (void)b0; (void)b1;
        } else {
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                "{%0, %1, %2, %3}, "
                "{%4, %5, %6, %7}, "
                "{%8, %9}, "
                "{%0, %1, %2, %3};\n"
                : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                  "r"(b0), "r"(b1));
        }
    }

    auto write_out = [&](int h, int d, float val) {
        if (h >= active_heads || d >= kHeadDim) return;
        const float row_lse = lse[static_cast<int64_t>(b) * num_heads + h];
        const float sink = attn_sink == nullptr ? 0.0f : attn_sink[h];
        const float gate = attn_sink == nullptr
                               ? 1.0f
                               : 1.0f / (1.0f + expf(-(row_lse - sink)));
        store_out_value<out_t>(
            out, static_cast<int64_t>(b) * out_stride_b +
                     static_cast<int64_t>(h) * out_stride_h +
                     static_cast<int64_t>(d) * out_stride_d,
            val * gate);
    };

    const int h0 = m_warp_base + groupID;
    const int h1 = h0 + 8;
    const int d0_glob = d_base + 2 * tid_in_group;
    const int d1_glob = d0_glob + 1;
    write_out(h0, d0_glob, d0);
    write_out(h0, d1_glob, d1);
    write_out(h1, d0_glob, d2);
    write_out(h1, d1_glob, d3);
}

template <typename q_t, typename kv_t, typename index_t>
__global__ void sparse_mla_prefill_indexed_score_tiled_kernel(
    const q_t* __restrict__ q, const kv_t* __restrict__ kv,
    const index_t* __restrict__ indices, const void* __restrict__ topk_length,
    float* __restrict__ scores, int num_tokens, int active_heads, int num_heads,
    int kv_tokens, int topk, int64_t q_stride_t, int64_t q_stride_h,
    int64_t q_stride_d, int64_t kv_stride_t, int64_t kv_stride_d,
    int64_t indices_stride_t, int64_t indices_stride_k, int topk_length_kind,
    float softmax_scale) {
    const int group_id = threadIdx.x / kScoreGroupSize;
    const int lane = threadIdx.x - group_id * kScoreGroupSize;
    const int k_idx = blockIdx.x * kScoreCandidatesPerBlock + group_id;
    const int head = blockIdx.y;
    const int token = blockIdx.z;
    if (token >= num_tokens || head >= active_heads || k_idx >= topk)
        return;

    const int limit = max(
        0, min(topk,
               load_length_value(topk_length, topk_length_kind, token, topk)));
    const int64_t score_offset =
        (static_cast<int64_t>(token) * active_heads + head) * topk + k_idx;
    if (k_idx >= limit) {
        if (lane == 0)
            scores[score_offset] = -INFINITY;
        return;
    }

    const int64_t selected = load_index<index_t>(
        indices, static_cast<int64_t>(token) * indices_stride_t +
                     static_cast<int64_t>(k_idx) * indices_stride_k);
    if (selected < 0 || selected >= kv_tokens) {
        if (lane == 0)
            scores[score_offset] = -INFINITY;
        return;
    }

    float partial = 0.0f;
    for (int d = lane; d < kHeadDim; d += kScoreGroupSize) {
        const float qv = load_q_value<q_t>(
            q, static_cast<int64_t>(token) * q_stride_t +
                   static_cast<int64_t>(head) * q_stride_h +
                   static_cast<int64_t>(d) * q_stride_d);
        partial += qv * load_workspace_value<kv_t>(
                            kv, selected * kv_stride_t +
                                    static_cast<int64_t>(d) * kv_stride_d);
    }

    for (int offset = kScoreGroupSize / 2; offset > 0; offset >>= 1)
        partial += __shfl_down_sync(0xffffffffu, partial, offset,
                                    kScoreGroupSize);
    if (lane == 0)
        scores[score_offset] = partial * softmax_scale;
}

template <typename kv_t, typename index_t>
__global__ void gather_bf16_workspace_kernel(
    const kv_t* __restrict__ kv, const index_t* __restrict__ indices,
    kv_t* __restrict__ out, int num_tokens, int topk, int head_dim,
    int64_t kv_tokens, int64_t kv_stride_t, int64_t kv_stride_d,
    int64_t indices_stride_t,
    int64_t indices_stride_h, int64_t indices_stride_k,
    int64_t out_stride_t, int64_t out_stride_k, int64_t out_stride_d,
    bool indices_3d) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    const int k_idx = blockIdx.y;
    const int token = blockIdx.z;
    if (token >= num_tokens || k_idx >= topk || d >= head_dim)
        return;

    const int64_t index_offset =
        static_cast<int64_t>(token) * indices_stride_t +
        (indices_3d ? indices_stride_h : 0) +
        static_cast<int64_t>(k_idx) * indices_stride_k;
    int64_t selected = load_index<index_t>(indices, index_offset);
    if (selected < 0)
        selected = 0;
    if (selected >= kv_tokens)
        selected = kv_tokens - 1;

    const int64_t kv_offset =
        selected * kv_stride_t + static_cast<int64_t>(d) * kv_stride_d;
    out[static_cast<int64_t>(token) * out_stride_t +
        static_cast<int64_t>(k_idx) * out_stride_k +
        static_cast<int64_t>(d) * out_stride_d] = kv[kv_offset];
}

template <typename kv_t, typename out_t, typename index_t>
__global__ void sparse_mla_prefill_indexed_output_kernel(
    const kv_t* __restrict__ kv, const index_t* __restrict__ indices,
    const void* __restrict__ topk_length, const float* __restrict__ scores,
    const float* __restrict__ lse, const float* __restrict__ attn_sink,
    out_t* __restrict__ out, int num_tokens, int active_heads, int num_heads,
    int kv_tokens, int topk, int64_t kv_stride_t, int64_t kv_stride_d,
    int64_t indices_stride_t, int64_t indices_stride_k, int64_t out_stride_t,
    int64_t out_stride_h, int64_t out_stride_d, int topk_length_kind) {
    const int th = blockIdx.x;
    const int tile = blockIdx.y;
    const int token = th / active_heads;
    const int head = th - token * active_heads;
    const int d = tile * blockDim.x + threadIdx.x;
    if (token >= num_tokens || d >= kHeadDim)
        return;

    const int limit = max(
        0, min(topk,
               load_length_value(topk_length, topk_length_kind, token, topk)));
    const int64_t score_base =
        (static_cast<int64_t>(token) * active_heads + head) * topk;

    float accum = 0.0f;
    for (int k_idx = 0; k_idx < limit; ++k_idx) {
        const float p = scores[score_base + k_idx];
        if (p == 0.0f)
            continue;
        const int64_t selected = load_index<index_t>(
            indices, static_cast<int64_t>(token) * indices_stride_t +
                         static_cast<int64_t>(k_idx) * indices_stride_k);
        if (selected < 0 || selected >= kv_tokens)
            continue;
        accum += p * load_workspace_value<kv_t>(
                         kv, selected * kv_stride_t +
                                 static_cast<int64_t>(d) * kv_stride_d);
    }

    const float row_lse =
        lse[static_cast<int64_t>(token) * num_heads + head];
    const float sink = attn_sink == nullptr ? 0.0f : attn_sink[head];
    const float gate =
        attn_sink == nullptr ? 1.0f : 1.0f / (1.0f + expf(-(row_lse - sink)));
    store_out_value<out_t>(
        out, static_cast<int64_t>(token) * out_stride_t +
                 static_cast<int64_t>(head) * out_stride_h +
                 static_cast<int64_t>(d) * out_stride_d,
        accum * gate);
}

// ===== PREFILL FLASH_FUSION (MMA) ==========================================
// Port of the decode flash-fusion (online flash-attention with bf16
// mma.sync.m16n8k16 QK + scalar-fp32 PV + tile-wise online softmax) onto
// the prefill kernel. Reads kv via gather: kv[indices[token, k_idx], d].
// Block tile: M=16 heads × N=32 slots × kHeadDim=512.
// Grid: (num_tokens, ceil(active_heads/16)) ≈ 8192 blocks at 4096 tokens
// — abundant grid parallelism, split-K not needed.
//
// Activation: DG_SM120_FAST_SPARSE_MLA=1 + DG_SM120_FLASH_FUSION_PREFILL=1.
__host__ __forceinline__ bool sm120_flash_fusion_prefill_enabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("DG_SM120_FLASH_FUSION_PREFILL");
        if (env == nullptr || env[0] == '\0') return false;
        return env[0] != '0';
    }();
    return enabled;
}

__host__ __forceinline__ bool sm120_flash_fusion_prefill_scalar_check() {
    static const bool enabled = []() {
        const char* env = std::getenv("DG_SM120_FLASH_FUSION_PREFILL_SCALAR_CHECK");
        if (env == nullptr || env[0] == '\0') return false;
        return env[0] != '0';
    }();
    return enabled;
}

// PV-MMA path: replaces scalar fp32 PV with mma.sync.aligned.m16n8k16 +
// ldmatrix.x2.trans for K_tile B-fragment. Default ON when prefill flash-
// fusion is active; disable via DG_SM120_FLASH_FUSION_PREFILL_PV_SCALAR=1
// for fallback / debugging.
__host__ __forceinline__ bool sm120_flash_fusion_prefill_pv_scalar() {
    static const bool enabled = []() {
        const char* env = std::getenv("DG_SM120_FLASH_FUSION_PREFILL_PV_SCALAR");
        if (env == nullptr || env[0] == '\0') return false;
        return env[0] != '0';
    }();
    return enabled;
}

// cp.async double-buffer K_tile: pipeline the per-tile K-gather behind the
// QK+PV MMA on the prior tile. Flag-gated via
// DG_SM120_FLASH_FUSION_PREFILL_CPASYNC=1 (default OFF). Doubles K_tile smem
// (~33 KB extra) to ~117 KB total; SM_120 dynamic smem cap is 228 KB so fits.
__host__ __forceinline__ bool sm120_flash_fusion_prefill_cpasync() {
    static const bool enabled = []() {
        const char* env = std::getenv("DG_SM120_FLASH_FUSION_PREFILL_CPASYNC");
        if (env == nullptr || env[0] == '\0') return false;
        return env[0] != '0';
    }();
    return enabled;
}

// Register-resident D-frag path: hold the 16×512 fp32 O_accum in per-lane
// MMA fragment registers (64 fp32/lane = 256 B/thread) instead of 32 KB of
// shared memory. Drops the per-tile rescale + load/store smem round-trip and
// frees enough smem on SM_120 (99 KB cap) to fit cp.async double-buffer.
// Requires kPvMma=true (the scalar-PV fallback still uses smem O). Default
// OFF; enable via DG_SM120_FLASH_FUSION_PREFILL_REG_O=1.
__host__ __forceinline__ bool sm120_flash_fusion_prefill_reg_o() {
    static const bool enabled = []() {
        const char* env = std::getenv("DG_SM120_FLASH_FUSION_PREFILL_REG_O");
        if (env == nullptr || env[0] == '\0') return false;
        return env[0] != '0';
    }();
    return enabled;
}

template <typename q_t, typename kv_t, typename out_t, typename index_t,
          bool kScalarCheck = false, bool kPvMma = true,
          bool kPipeline = false, bool kRegO = false>
__global__ void sparse_mla_prefill_workspace_flash_mma_kernel(
    const q_t* __restrict__ q, const kv_t* __restrict__ kv,
    const index_t* __restrict__ indices,
    const void* __restrict__ topk_length, const float* __restrict__ attn_sink,
    out_t* __restrict__ out, float* __restrict__ lse, int num_tokens,
    int active_heads, int num_heads, int kv_tokens, int topk,
    int64_t q_stride_t, int64_t q_stride_h, int64_t q_stride_d,
    int64_t kv_stride_t, int64_t kv_stride_d, int64_t indices_stride_t,
    int64_t indices_stride_k, int64_t out_stride_t, int64_t out_stride_h,
    int64_t out_stride_d, int topk_length_kind, float softmax_scale) {
    extern __shared__ uint8_t flash_pre_smem_raw[];
    __nv_bfloat16* Q_smem =
        reinterpret_cast<__nv_bfloat16*>(flash_pre_smem_raw);
    __nv_bfloat16* K_tile_smem_base = Q_smem + kFlashMmaM * kHeadDim;
    constexpr size_t kKTileBufElems =
        static_cast<size_t>(kFlashMmaTileK) * kFlashMmaKStride;
    constexpr size_t kKTileTotalElems =
        kKTileBufElems * (kPipeline ? 2 : 1);
    // O_accum lives in smem only when kRegO=false (scalar-PV fallback or
    // legacy PV-MMA path). When kRegO=true the D-frag stays in registers and
    // we skip the 32 KB allocation entirely.
    float* O_accum_smem = kRegO ? nullptr : reinterpret_cast<float*>(
        K_tile_smem_base + kKTileTotalElems);
    float* scores_smem = kRegO
        ? reinterpret_cast<float*>(K_tile_smem_base + kKTileTotalElems)
        : O_accum_smem + static_cast<size_t>(kFlashMmaM) * kHeadDim;
    float* m_smem = scores_smem + kFlashMmaM * kFlashMmaTileK;
    float* l_smem = m_smem + kFlashMmaM;
    float* tile_max_smem = l_smem + kFlashMmaM;
    float* rescale_smem = tile_max_smem + kFlashMmaM;
    float* tile_sum_smem = rescale_smem + kFlashMmaM;

    const int token = blockIdx.x;
    const int h_block = blockIdx.y;
    const int h_base = h_block * kFlashMmaM;
    if (token >= num_tokens || h_base >= active_heads) return;

    const int limit = max(
        0, min(topk,
               load_length_value(topk_length, topk_length_kind, token, topk)));

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int groupID = lane >> 2;
    const int tid_in_group = lane & 3;

    // Phase 0: Q[token, h_base + m, :] → Q_smem[m, :].
    const int64_t q_t_base = static_cast<int64_t>(token) * q_stride_t;
    for (int idx = tid; idx < kFlashMmaM * kHeadDim; idx += kFlashMmaThreads) {
        const int m = idx / kHeadDim;
        const int d = idx - m * kHeadDim;
        const int h = h_base + m;
        float v = 0.f;
        if (h < active_heads && h < num_heads) {
            v = load_q_value<q_t>(
                q, q_t_base + static_cast<int64_t>(h) * q_stride_h +
                       static_cast<int64_t>(d) * q_stride_d);
        }
        Q_smem[m * kHeadDim + d] = __float2bfloat16_rn(v);
    }

    // Per-warp d-col strip + N-issue count. Used by both kRegO=true (Phase 1
    // register zero / Phase 2h register rescale / Phase 3 register writeout)
    // and kPvMma=true (Phase 2i mma issue loop). Hoisted here so all phases
    // see the same constants.
    static_assert(kHeadDim % (kFlashMmaNumWarps * kFlashMmaN) == 0,
                  "headdim must split across warps × N");
    constexpr int kPvNIssuesPerWarp =
        kHeadDim / (kFlashMmaNumWarps * kFlashMmaN);
    const int d_warp_base = warp_id * (kHeadDim / kFlashMmaNumWarps);

    // Per-lane D-frag register array (kRegO=true only). 4 fp32 per N-issue
    // matches the m16n8k16 D-fragment lane layout:
    //   reg0 = O[groupID,    n_base + 2*tid_in_group + 0]
    //   reg1 = O[groupID,    n_base + 2*tid_in_group + 1]
    //   reg2 = O[groupID+8,  n_base + 2*tid_in_group + 0]
    //   reg3 = O[groupID+8,  n_base + 2*tid_in_group + 1]
    constexpr int kRegOArrayLen = kRegO ? kPvNIssuesPerWarp : 1;
    float d_o[kRegOArrayLen][4];

    // Phase 1: init.
    for (int idx = tid; idx < kFlashMmaM; idx += kFlashMmaThreads) {
        m_smem[idx] = -INFINITY;
        l_smem[idx] = 0.f;
    }
    if constexpr (kRegO) {
        #pragma unroll
        for (int n_iss = 0; n_iss < kPvNIssuesPerWarp; ++n_iss) {
            d_o[n_iss][0] = 0.f;
            d_o[n_iss][1] = 0.f;
            d_o[n_iss][2] = 0.f;
            d_o[n_iss][3] = 0.f;
        }
    } else {
        for (int idx = tid; idx < kFlashMmaM * kHeadDim;
             idx += kFlashMmaThreads) {
            O_accum_smem[idx] = 0.f;
        }
    }
    __syncthreads();

    // Phase 2: outer loop over K-tiles.
    const int64_t indices_t_base =
        static_cast<int64_t>(token) * indices_stride_t;
    const int num_tiles = (topk + kFlashMmaTileK - 1) / kFlashMmaTileK;

    // cp.async issue helper (kPipeline=true): per-thread loop emits cp.async
    // .cg.shared.global ops covering kFlashMmaTileK rows × kHeadDim cols of K
    // into the given dst buffer. Each cp.async copies 16 B (8 bf16 cells).
    // Invalid slots (slot >= limit, slot >= topk, or selected out of range)
    // pass src-size=0 — cp.async then zero-fills the dst, matching the
    // synchronous path's `v = 0` masking. Caller emits cp.async.commit_group.
    auto issue_cp_async_tile = [&](int sb_arg, __nv_bfloat16* dst_buf) {
        constexpr int kChunksPerRow = kHeadDim / 8;
        constexpr int kTotalChunks = kFlashMmaTileK * kChunksPerRow;
        constexpr int kPerThread = kTotalChunks / kFlashMmaThreads;
        static_assert(kTotalChunks % kFlashMmaThreads == 0,
                      "cp.async chunk grid must divide evenly across threads");
        #pragma unroll
        for (int i = 0; i < kPerThread; ++i) {
            const int chunk_idx = tid + i * kFlashMmaThreads;
            const int k_local = chunk_idx / kChunksPerRow;
            const int d_chunk = chunk_idx - k_local * kChunksPerRow;
            const int d = d_chunk * 8;
            const int slot = sb_arg + k_local;
            int src_size = 0;
            int64_t src_off = 0;
            if (slot < limit && slot < topk) {
                const int64_t selected = load_index<index_t>(
                    indices,
                    indices_t_base +
                        static_cast<int64_t>(slot) * indices_stride_k);
                if (selected >= 0 && selected < kv_tokens) {
                    src_size = 16;
                    src_off = selected * kv_stride_t +
                              static_cast<int64_t>(d) * kv_stride_d;
                }
            }
            const __nv_bfloat16* src_ptr =
                reinterpret_cast<const __nv_bfloat16*>(kv) + src_off;
            __nv_bfloat16* dst_ptr =
                dst_buf + k_local * kFlashMmaKStride + d;
            const uint32_t dst_addr = __cvta_generic_to_shared(dst_ptr);
            asm volatile(
                "cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                :: "r"(dst_addr), "l"(src_ptr), "r"(src_size));
        }
    };

    // Prologue: kick off tile 0 into buf 0 before the loop so iter 0's QK
    // can start as soon as wait_group resolves.
    if constexpr (kPipeline) {
        issue_cp_async_tile(0, K_tile_smem_base);
        asm volatile("cp.async.commit_group;\n");
    }

    for (int tile = 0; tile < num_tiles; ++tile) {
        const int slot_base = tile * kFlashMmaTileK;
        __nv_bfloat16* K_tile_smem = K_tile_smem_base +
            (kPipeline ? (tile & 1) * kKTileBufElems : 0);

        // Phase 2a: synchronous gather (kPipeline=false) or pipelined wait +
        // prefetch (kPipeline=true). Both leave K_tile_smem populated for
        // tile `tile` after the trailing __syncthreads.
        if constexpr (kPipeline) {
            const bool has_next = (tile + 1 < num_tiles);
            if (has_next) {
                __nv_bfloat16* next_buf =
                    K_tile_smem_base + ((tile + 1) & 1) * kKTileBufElems;
                issue_cp_async_tile(slot_base + kFlashMmaTileK, next_buf);
                asm volatile("cp.async.commit_group;\n");
                asm volatile("cp.async.wait_group 1;\n");
            } else {
                asm volatile("cp.async.wait_group 0;\n");
            }
            __syncthreads();
        } else {
            for (int idx = tid; idx < kFlashMmaTileK * kHeadDim;
                 idx += kFlashMmaThreads) {
                const int k_local = idx / kHeadDim;
                const int d = idx - k_local * kHeadDim;
                const int slot = slot_base + k_local;
                float v = 0.f;
                if (slot < limit && slot < topk) {
                    const int64_t selected = load_index<index_t>(
                        indices, indices_t_base +
                                     static_cast<int64_t>(slot) * indices_stride_k);
                    if (selected >= 0 && selected < kv_tokens) {
                        v = load_workspace_value<kv_t>(
                            kv, selected * kv_stride_t +
                                    static_cast<int64_t>(d) * kv_stride_d);
                    }
                }
                K_tile_smem[k_local * kFlashMmaKStride + d] =
                    __float2bfloat16_rn(v);
            }
            __syncthreads();
        }

        // Phase 2b: QK MMA.
        const int n_base = warp_id * kFlashMmaN;
        float d_qk0 = 0.f, d_qk1 = 0.f, d_qk2 = 0.f, d_qk3 = 0.f;
        #pragma unroll
        for (int kc = 0; kc < kHeadDim; kc += kFlashMmaK) {
            const int a_col_base = 2 * tid_in_group;
            const __nv_bfloat16* a_row0_ptr = Q_smem + groupID * kHeadDim + kc;
            const __nv_bfloat16* a_row1_ptr =
                Q_smem + (groupID + 8) * kHeadDim + kc;
            const __nv_bfloat16* b_row_ptr =
                K_tile_smem + (n_base + groupID) * kFlashMmaKStride + kc;
            const uint32_t a0 = *reinterpret_cast<const uint32_t*>(
                a_row0_ptr + a_col_base);
            const uint32_t a1 = *reinterpret_cast<const uint32_t*>(
                a_row1_ptr + a_col_base);
            const uint32_t a2 = *reinterpret_cast<const uint32_t*>(
                a_row0_ptr + a_col_base + 8);
            const uint32_t a3 = *reinterpret_cast<const uint32_t*>(
                a_row1_ptr + a_col_base + 8);
            const uint32_t b0 = *reinterpret_cast<const uint32_t*>(
                b_row_ptr + a_col_base);
            const uint32_t b1 = *reinterpret_cast<const uint32_t*>(
                b_row_ptr + a_col_base + 8);
            if constexpr (kScalarCheck) {
                const int row0 = groupID;
                const int row1 = groupID + 8;
                const int col0 = 2 * tid_in_group;
                const int col1 = col0 + 1;
                float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
                #pragma unroll
                for (int kk = 0; kk < kFlashMmaK; ++kk) {
                    const float ar0 = __bfloat162float(
                        Q_smem[row0 * kHeadDim + kc + kk]);
                    const float ar1 = __bfloat162float(
                        Q_smem[row1 * kHeadDim + kc + kk]);
                    const float bn0 = __bfloat162float(
                        K_tile_smem[(n_base + col0) * kFlashMmaKStride + kc + kk]);
                    const float bn1 = __bfloat162float(
                        K_tile_smem[(n_base + col1) * kFlashMmaKStride + kc + kk]);
                    acc0 += ar0 * bn0;
                    acc1 += ar0 * bn1;
                    acc2 += ar1 * bn0;
                    acc3 += ar1 * bn1;
                }
                d_qk0 += acc0; d_qk1 += acc1; d_qk2 += acc2; d_qk3 += acc3;
                (void)a0; (void)a1; (void)a2; (void)a3; (void)b0; (void)b1;
            } else {
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                    "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
                    "{%0, %1, %2, %3};\n"
                    : "+f"(d_qk0), "+f"(d_qk1), "+f"(d_qk2), "+f"(d_qk3)
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
            }
        }

        // Phase 2c: scale + validity.
        auto write_score = [&](int m, int n, float val) {
            if (m >= kFlashMmaM || n >= kFlashMmaTileK) return;
            const int slot = slot_base + n;
            const bool valid = (slot < limit) && (slot < topk);
            scores_smem[m * kFlashMmaTileK + n] =
                valid ? val * softmax_scale : -INFINITY;
        };
        const int m0 = groupID;
        const int m1 = groupID + 8;
        const int n0 = n_base + 2 * tid_in_group;
        const int n1 = n0 + 1;
        write_score(m0, n0, d_qk0);
        write_score(m0, n1, d_qk1);
        write_score(m1, n0, d_qk2);
        write_score(m1, n1, d_qk3);
        __syncthreads();

        // Phase 2d-g: online softmax (identical to decode kernel).
        #pragma unroll
        for (int rsub = 0; rsub < 4; ++rsub) {
            const int h = warp_id * 4 + rsub;
            float v = scores_smem[h * kFlashMmaTileK + lane];
            #pragma unroll
            for (int s = 16; s > 0; s >>= 1) {
                v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, s));
            }
            if (lane == 0) tile_max_smem[h] = v;
        }
        __syncthreads();
        if (tid < kFlashMmaM) {
            const int h = tid;
            const float m_old = m_smem[h];
            const float tm = tile_max_smem[h];
            const float m_new = fmaxf(m_old, tm);
            const float diff = (isfinite(m_old) && isfinite(m_new))
                                   ? (m_old - m_new)
                                   : (isfinite(m_new) ? -INFINITY : 0.f);
            rescale_smem[h] = expf(diff);
            m_smem[h] = m_new;
        }
        __syncthreads();
        #pragma unroll
        for (int rsub = 0; rsub < 4; ++rsub) {
            const int h = warp_id * 4 + rsub;
            const float m_new = m_smem[h];
            const int k = lane;
            const float s_old = scores_smem[h * kFlashMmaTileK + k];
            const float p =
                (isfinite(s_old) && isfinite(m_new)) ? expf(s_old - m_new) : 0.f;
            scores_smem[h * kFlashMmaTileK + k] = p;
            float sum = p;
            #pragma unroll
            for (int sh = 16; sh > 0; sh >>= 1) {
                sum += __shfl_xor_sync(0xffffffffu, sum, sh);
            }
            if (lane == 0) tile_sum_smem[h] = sum;
        }
        __syncthreads();
        if (tid < kFlashMmaM) {
            const int h = tid;
            l_smem[h] = rescale_smem[h] * l_smem[h] + tile_sum_smem[h];
        }
        __syncthreads();

        // Phase 2h: O *= rescale.
        if constexpr (kRegO) {
            // Per-lane register multiply. rescale_smem[h] was just written in
            // Phase 2g and synced; this read needs no extra sync. No O smem
            // touch → no post-rescale sync either (Phase 2i's reads of
            // scores_smem are already covered by the post-2f sync, and the
            // post-2i sync still guards K_tile reads vs next-iter writes).
            const float r_lo = rescale_smem[groupID];
            const float r_hi = rescale_smem[groupID + 8];
            #pragma unroll
            for (int n_iss = 0; n_iss < kPvNIssuesPerWarp; ++n_iss) {
                d_o[n_iss][0] *= r_lo;
                d_o[n_iss][1] *= r_lo;
                d_o[n_iss][2] *= r_hi;
                d_o[n_iss][3] *= r_hi;
            }
        } else {
            for (int idx = tid; idx < kFlashMmaM * kHeadDim;
                 idx += kFlashMmaThreads) {
                const int m = idx / kHeadDim;
                O_accum_smem[idx] *= rescale_smem[m];
            }
            __syncthreads();
        }

        // Phase 2i: PV — bf16 mma.sync.m16n8k16 (kPvMma=true) or scalar fp32
        // fallback (kPvMma=false). Updates O_accum_smem in-place when
        // kRegO=false; accumulates into d_o registers when kRegO=true.
        if constexpr (kPvMma) {
            // Each warp owns kHeadDim/kFlashMmaNumWarps = 128 d_cols. With
            // kFlashMmaN=8 cols per mma issue, that's kPvNIssuesPerWarp=16
            // N-issues per warp per K-tile, with 2 K-iters of K=16 each
            // spanning the TILE_K=32 slot dim. (kPvNIssuesPerWarp /
            // d_warp_base hoisted to the kernel-top alongside Phase 1 init.)
            constexpr int kPvKItersPerTile = kFlashMmaTileK / kFlashMmaK;
            #pragma unroll
            for (int n_iss = 0; n_iss < kPvNIssuesPerWarp; ++n_iss) {
                const int n_base = d_warp_base + n_iss * kFlashMmaN;
                // D-frag: 4 fp32 per lane = O[(groupID, groupID+8) ×
                //                              (n_base+2tig, n_base+2tig+1)].
                // kRegO=true keeps the running fragment in d_o registers
                // across K-tiles; kRegO=false reloads from O_accum_smem each
                // tile (where Phase 2h rescaled the smem in place).
                float d_pv0, d_pv1, d_pv2, d_pv3;
                if constexpr (kRegO) {
                    d_pv0 = d_o[n_iss][0];
                    d_pv1 = d_o[n_iss][1];
                    d_pv2 = d_o[n_iss][2];
                    d_pv3 = d_o[n_iss][3];
                } else {
                    d_pv0 = O_accum_smem[
                        groupID * kHeadDim + n_base + 2 * tid_in_group + 0];
                    d_pv1 = O_accum_smem[
                        groupID * kHeadDim + n_base + 2 * tid_in_group + 1];
                    d_pv2 = O_accum_smem[
                        (groupID + 8) * kHeadDim + n_base + 2 * tid_in_group + 0];
                    d_pv3 = O_accum_smem[
                        (groupID + 8) * kHeadDim + n_base + 2 * tid_in_group + 1];
                }

                #pragma unroll
                for (int kk = 0; kk < kPvKItersPerTile; ++kk) {
                    const int k_base = kk * kFlashMmaK;

                    // A-frag: scores[M, K] cast to bf16. Layout per lane
                    // matches mma m16n8k16.row.col A:
                    //   reg0 = scores[groupID,    k_base + 2tig + 0..1]
                    //   reg1 = scores[groupID+8,  k_base + 2tig + 0..1]
                    //   reg2 = scores[groupID,    k_base + 2tig + 8..9]
                    //   reg3 = scores[groupID+8,  k_base + 2tig + 8..9]
                    const int a_col_lo = k_base + 2 * tid_in_group;
                    const int a_col_hi = a_col_lo + 8;
                    const float s0a = scores_smem[
                        groupID * kFlashMmaTileK + a_col_lo + 0];
                    const float s0b = scores_smem[
                        groupID * kFlashMmaTileK + a_col_lo + 1];
                    const float s1a = scores_smem[
                        (groupID + 8) * kFlashMmaTileK + a_col_lo + 0];
                    const float s1b = scores_smem[
                        (groupID + 8) * kFlashMmaTileK + a_col_lo + 1];
                    const float s0c = scores_smem[
                        groupID * kFlashMmaTileK + a_col_hi + 0];
                    const float s0d = scores_smem[
                        groupID * kFlashMmaTileK + a_col_hi + 1];
                    const float s1c = scores_smem[
                        (groupID + 8) * kFlashMmaTileK + a_col_hi + 0];
                    const float s1d = scores_smem[
                        (groupID + 8) * kFlashMmaTileK + a_col_hi + 1];
                    const __nv_bfloat162 ar0 = __floats2bfloat162_rn(s0a, s0b);
                    const __nv_bfloat162 ar1 = __floats2bfloat162_rn(s1a, s1b);
                    const __nv_bfloat162 ar2 = __floats2bfloat162_rn(s0c, s0d);
                    const __nv_bfloat162 ar3 = __floats2bfloat162_rn(s1c, s1d);
                    const uint32_t a_pv0 =
                        *reinterpret_cast<const uint32_t*>(&ar0);
                    const uint32_t a_pv1 =
                        *reinterpret_cast<const uint32_t*>(&ar1);
                    const uint32_t a_pv2 =
                        *reinterpret_cast<const uint32_t*>(&ar2);
                    const uint32_t a_pv3 =
                        *reinterpret_cast<const uint32_t*>(&ar3);

                    // B-frag: K_tile[K=k_base..+15, N=n_base..+7]. Smem layout
                    // is [slot=row, d=col] row-major, so smem rows are K and
                    // smem cols are N — opposite of mma B's "K contiguous
                    // within N". Use ldmatrix.x2.trans which loads two 8x8
                    // bf16 blocks from row-major smem and writes them into
                    // the B-fragment register layout (transposed).
                    //
                    // ldmatrix.x2 uses 16 lanes. Lane t (t<16) gives address
                    // of smem row t of the combined 16x8 block:
                    //   row(t) = K_tile[k_base + t, n_base..n_base+7].
                    uint32_t b_pv0, b_pv1;
                    {
                        const int row = lane & 15;
                        const __nv_bfloat16* row_ptr =
                            K_tile_smem + (k_base + row) * kFlashMmaKStride
                            + n_base;
                        const uint32_t row_addr =
                            __cvta_generic_to_shared(row_ptr);
                        asm volatile(
                            "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 "
                            "{%0, %1}, [%2];\n"
                            : "=r"(b_pv0), "=r"(b_pv1)
                            : "r"(row_addr));
                    }

                    if constexpr (kScalarCheck) {
                        // Oracle: replicate the same lane→D-frag math via
                        // scalar fp32 FMAs reading the same smem. If the
                        // kernel's PV-MMA result matches the kScalarCheck
                        // result, the fragment layout is correct.
                        float a0 = 0, a1 = 0, a2 = 0, a3 = 0;
                        #pragma unroll
                        for (int kx = 0; kx < kFlashMmaK; ++kx) {
                            const float aa0 = scores_smem[
                                groupID * kFlashMmaTileK + k_base + kx];
                            const float aa1 = scores_smem[
                                (groupID + 8) * kFlashMmaTileK + k_base + kx];
                            const float bb0 = __bfloat162float(K_tile_smem[
                                (k_base + kx) * kFlashMmaKStride
                                + n_base + 2 * tid_in_group + 0]);
                            const float bb1 = __bfloat162float(K_tile_smem[
                                (k_base + kx) * kFlashMmaKStride
                                + n_base + 2 * tid_in_group + 1]);
                            a0 += aa0 * bb0;
                            a1 += aa0 * bb1;
                            a2 += aa1 * bb0;
                            a3 += aa1 * bb1;
                        }
                        d_pv0 += a0; d_pv1 += a1;
                        d_pv2 += a2; d_pv3 += a3;
                        (void)a_pv0; (void)a_pv1; (void)a_pv2; (void)a_pv3;
                        (void)b_pv0; (void)b_pv1;
                    } else {
                        asm volatile(
                            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
                            "{%0, %1, %2, %3};\n"
                            : "+f"(d_pv0), "+f"(d_pv1), "+f"(d_pv2), "+f"(d_pv3)
                            : "r"(a_pv0), "r"(a_pv1), "r"(a_pv2), "r"(a_pv3),
                              "r"(b_pv0), "r"(b_pv1));
                    }
                }

                if constexpr (kRegO) {
                    d_o[n_iss][0] = d_pv0;
                    d_o[n_iss][1] = d_pv1;
                    d_o[n_iss][2] = d_pv2;
                    d_o[n_iss][3] = d_pv3;
                } else {
                    O_accum_smem[
                        groupID * kHeadDim + n_base + 2 * tid_in_group + 0] =
                        d_pv0;
                    O_accum_smem[
                        groupID * kHeadDim + n_base + 2 * tid_in_group + 1] =
                        d_pv1;
                    O_accum_smem[
                        (groupID + 8) * kHeadDim + n_base + 2 * tid_in_group + 0] =
                        d_pv2;
                    O_accum_smem[
                        (groupID + 8) * kHeadDim + n_base + 2 * tid_in_group + 1] =
                        d_pv3;
                }
            }
        } else {
            // Scalar fp32 PV fallback. Requires kRegO=false (the launcher
            // refuses kRegO=true with kPvMma=false).
            static_assert(!kRegO,
                          "kRegO=true requires kPvMma=true (scalar PV path "
                          "still goes through O_accum_smem)");
            for (int idx = tid; idx < kFlashMmaM * kHeadDim;
                 idx += kFlashMmaThreads) {
                const int m = idx / kHeadDim;
                const int d = idx - m * kHeadDim;
                float acc = 0.f;
                #pragma unroll
                for (int kk = 0; kk < kFlashMmaTileK; ++kk) {
                    acc += scores_smem[m * kFlashMmaTileK + kk] *
                           __bfloat162float(
                               K_tile_smem[kk * kFlashMmaKStride + d]);
                }
                O_accum_smem[idx] += acc;
            }
        }
        __syncthreads();
    }

    // Phase 3: finalize. Write per-(token, head) lse + final out scaled by
    // gate / l. Match the prefill chain's lse + sigmoid-gate semantics.
    if (tid < kFlashMmaM) {
        const int h = h_base + tid;
        const float l_val = l_smem[tid];
        const float m_val = m_smem[tid];
        const float lse_v =
            (l_val > 0.f && isfinite(m_val)) ? (logf(l_val) + m_val) : -INFINITY;
        if (h < num_heads && lse != nullptr) {
            lse[static_cast<int64_t>(token) * num_heads + h] = lse_v;
        }
        float gate = 1.f;
        if (attn_sink != nullptr && h < num_heads) {
            const float sink_v = attn_sink[h];
            const float x = lse_v - sink_v;
            gate = isfinite(x) ? (1.f / (1.f + expf(-x)))
                               : (lse_v > 0.f ? 1.f : 0.f);
        }
        rescale_smem[tid] = (l_val > 0.f) ? (gate / l_val) : 0.f;
    }
    __syncthreads();
    if constexpr (kRegO) {
        // Per-lane writeout from register-resident D-frag. Lane t covers
        // m ∈ {groupID, groupID+8} × n_iss ∈ [0, kPvNIssuesPerWarp) × col ∈
        // {2*tid_in_group, 2*tid_in_group+1}, totalling 64 fp32 cells per
        // lane (2 m × 16 n_iss × 2 col).
        const float r_lo = rescale_smem[groupID];
        const float r_hi = rescale_smem[groupID + 8];
        const int h_lo = h_base + groupID;
        const int h_hi = h_base + groupID + 8;
        const int64_t out_t_base =
            static_cast<int64_t>(token) * out_stride_t;
        #pragma unroll
        for (int n_iss = 0; n_iss < kPvNIssuesPerWarp; ++n_iss) {
            const int n_base = d_warp_base + n_iss * kFlashMmaN;
            const int d0 = n_base + 2 * tid_in_group + 0;
            const int d1 = d0 + 1;
            if (h_lo < active_heads) {
                store_out_value<out_t>(
                    out,
                    out_t_base +
                        static_cast<int64_t>(h_lo) * out_stride_h +
                        static_cast<int64_t>(d0) * out_stride_d,
                    d_o[n_iss][0] * r_lo);
                store_out_value<out_t>(
                    out,
                    out_t_base +
                        static_cast<int64_t>(h_lo) * out_stride_h +
                        static_cast<int64_t>(d1) * out_stride_d,
                    d_o[n_iss][1] * r_lo);
            }
            if (h_hi < active_heads) {
                store_out_value<out_t>(
                    out,
                    out_t_base +
                        static_cast<int64_t>(h_hi) * out_stride_h +
                        static_cast<int64_t>(d0) * out_stride_d,
                    d_o[n_iss][2] * r_hi);
                store_out_value<out_t>(
                    out,
                    out_t_base +
                        static_cast<int64_t>(h_hi) * out_stride_h +
                        static_cast<int64_t>(d1) * out_stride_d,
                    d_o[n_iss][3] * r_hi);
            }
        }
    } else {
        for (int idx = tid; idx < kFlashMmaM * kHeadDim;
             idx += kFlashMmaThreads) {
            const int m = idx / kHeadDim;
            const int d = idx - m * kHeadDim;
            const int h = h_base + m;
            if (h < active_heads) {
                const float v = O_accum_smem[idx] * rescale_smem[m];
                store_out_value<out_t>(
                    out, static_cast<int64_t>(token) * out_stride_t +
                             static_cast<int64_t>(h) * out_stride_h +
                             static_cast<int64_t>(d) * out_stride_d,
                    v);
            }
        }
    }
}

template <typename q_t, typename kv_t, typename out_t, typename index_t>
__global__ void sparse_mla_prefill_workspace_kernel(
    const q_t* __restrict__ q, const kv_t* __restrict__ kv,
    const index_t* __restrict__ indices, const void* __restrict__ topk_length,
    int topk_length_kind, const float* __restrict__ attn_sink,
    out_t* __restrict__ out, float* __restrict__ max_logits,
    float* __restrict__ lse, int num_heads, int kv_tokens, int topk,
    int64_t q_stride_t, int64_t q_stride_h, int64_t q_stride_d,
    int64_t kv_stride_t, int64_t kv_stride_d, int64_t indices_stride_t,
    int64_t indices_stride_k, int64_t out_stride_t, int64_t out_stride_h,
    int64_t out_stride_d, double softmax_scale) {
    extern __shared__ unsigned char smem[];
    float* scores = reinterpret_cast<float*>(smem);
    float* reduce = scores + topk;

    const int token = blockIdx.x;
    const int head = blockIdx.y;
    const int tid = threadIdx.x;

    int limit = load_length_value(topk_length, topk_length_kind, token, topk);
    limit = max(0, min(limit, topk));

    float local_max = -INFINITY;
    for (int k_idx = tid; k_idx < limit; k_idx += blockDim.x) {
        const int64_t selected = load_index<index_t>(
            indices, static_cast<int64_t>(token) * indices_stride_t +
                         static_cast<int64_t>(k_idx) * indices_stride_k);
        float score = -INFINITY;
        if (selected >= 0 && selected < kv_tokens) {
            float dot = 0.0f;
#pragma unroll 4
            for (int d = 0; d < kHeadDim; ++d) {
                const float qv = load_q_value<q_t>(
                    q, static_cast<int64_t>(token) * q_stride_t +
                           static_cast<int64_t>(head) * q_stride_h +
                           static_cast<int64_t>(d) * q_stride_d);
                const float kvv = load_workspace_value<kv_t>(
                    kv, selected * kv_stride_t +
                            static_cast<int64_t>(d) * kv_stride_d);
                dot += qv * kvv;
            }
            score = dot * static_cast<float>(softmax_scale);
        }
        scores[k_idx] = score;
        local_max = fmaxf(local_max, score);
    }

    reduce[tid] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            reduce[tid] = fmaxf(reduce[tid], reduce[tid + stride]);
        __syncthreads();
    }
    const float row_max = reduce[0];

    float local_sum = 0.0f;
    if (isfinite(row_max)) {
        for (int k_idx = tid; k_idx < limit; k_idx += blockDim.x) {
            const float p = isfinite(scores[k_idx])
                                ? expf(scores[k_idx] - row_max)
                                : 0.0f;
            scores[k_idx] = p;
            local_sum += p;
        }
    }
    reduce[tid] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            reduce[tid] += reduce[tid + stride];
        __syncthreads();
    }

    const float denom = reduce[0];
    const float row_lse = denom > 0.0f ? logf(denom) + row_max : -INFINITY;
    const float inv_denom = denom > 0.0f ? 1.0f / denom : 0.0f;
    const float sink = attn_sink == nullptr ? 0.0f : attn_sink[head];
    const float gate =
        attn_sink == nullptr ? 1.0f : 1.0f / (1.0f + expf(-(row_lse - sink)));

    for (int d = tid; d < kHeadDim; d += blockDim.x) {
        float accum = 0.0f;
        if (denom > 0.0f) {
            for (int k_idx = 0; k_idx < limit; ++k_idx) {
                const float p = scores[k_idx] * inv_denom;
                if (p == 0.0f)
                    continue;
                const int64_t selected = load_index<index_t>(
                    indices, static_cast<int64_t>(token) * indices_stride_t +
                                 static_cast<int64_t>(k_idx) *
                                     indices_stride_k);
                if (selected >= 0 && selected < kv_tokens) {
                    accum += p * load_workspace_value<kv_t>(
                                     kv, selected * kv_stride_t +
                                             static_cast<int64_t>(d) *
                                                 kv_stride_d);
                }
            }
        }
        store_out_value<out_t>(
            out, static_cast<int64_t>(token) * out_stride_t +
                     static_cast<int64_t>(head) * out_stride_h +
                     static_cast<int64_t>(d) * out_stride_d,
            accum * gate);
    }

    if (tid == 0) {
        max_logits[static_cast<int64_t>(token) * num_heads + head] = row_max;
        lse[static_cast<int64_t>(token) * num_heads + head] = row_lse;
    }
}

template <typename BlockT, typename SeqT, typename StartT>
__global__ void prefill_workspace_map_kernel(
    int32_t* __restrict__ out, const BlockT* __restrict__ block_table,
    const SeqT* __restrict__ seq_lens, const StartT* __restrict__ workspace_starts,
    int num_reqs, int block_size, int64_t out_rows, int64_t block_stride_b,
    int64_t block_stride_s, int64_t seq_stride, int64_t starts_stride) {
    constexpr int kMapThreads = 256;
    if (blockIdx.y == 0) {
        const int64_t row = static_cast<int64_t>(blockIdx.x) * kMapThreads + threadIdx.x;
        if (row < out_rows)
            out[row] = -1;
        return;
    }

    const int req = static_cast<int>(blockIdx.x);
    if (req >= num_reqs)
        return;

    const int64_t start =
        static_cast<int64_t>(workspace_starts[static_cast<int64_t>(req) * starts_stride]);
    const int seq_len =
        max(0, static_cast<int>(seq_lens[static_cast<int64_t>(req) * seq_stride]));
    for (int pos = threadIdx.x; pos < seq_len; pos += blockDim.x) {
        const int64_t row = start + pos;
        if (row < 0 || row >= out_rows)
            continue;
        const int block_in_seq = pos / block_size;
        const int block_offset = pos - block_in_seq * block_size;
        const int64_t physical_block = static_cast<int64_t>(
            block_table[static_cast<int64_t>(req) * block_stride_b +
                        static_cast<int64_t>(block_in_seq) * block_stride_s]);
        if (physical_block >= 0 && block_offset >= 0 && block_offset < block_size) {
            const int64_t linear =
                physical_block * static_cast<int64_t>(block_size) + block_offset;
            out[row] = linear <= 2147483647LL
                           ? static_cast<int32_t>(linear)
                           : -1;
        }
    }
}

template <typename BlockT, typename SeqT, typename GatherT>
__global__ void prefill_strided_workspace_map_kernel(
    int32_t* __restrict__ out, const BlockT* __restrict__ block_table,
    const SeqT* __restrict__ seq_lens, const GatherT* __restrict__ gather_lens,
    int num_reqs, int block_size, int64_t out_rows, int64_t row_stride,
    int64_t offset, bool encode_negative, int64_t block_stride_b,
    int64_t block_stride_s, int64_t seq_stride, int64_t gather_stride) {
    const int req = static_cast<int>(blockIdx.x);
    if (req >= num_reqs)
        return;

    const int seq_len =
        max(0, static_cast<int>(seq_lens[static_cast<int64_t>(req) * seq_stride]));
    int gather_len =
        gather_lens == nullptr
            ? seq_len
            : max(0, static_cast<int>(
                         gather_lens[static_cast<int64_t>(req) * gather_stride]));
    gather_len = min(gather_len, max(0, static_cast<int>(row_stride - offset)));
    const int start_pos = max(0, seq_len - gather_len);
    for (int out_idx = threadIdx.x; out_idx < gather_len; out_idx += blockDim.x) {
        const int pos = start_pos + out_idx;
        const int block_in_seq = pos / block_size;
        const int block_offset = pos - block_in_seq * block_size;
        const int64_t row =
            static_cast<int64_t>(req) * row_stride + offset + out_idx;
        if (row < 0 || row >= out_rows)
            continue;
        const int64_t physical_block = static_cast<int64_t>(
            block_table[static_cast<int64_t>(req) * block_stride_b +
                        static_cast<int64_t>(block_in_seq) * block_stride_s]);
        if (physical_block >= 0 && block_offset >= 0 && block_offset < block_size) {
            const int64_t linear =
                physical_block * static_cast<int64_t>(block_size) + block_offset;
            if (linear <= 2147483645LL) {
                const int32_t encoded = encode_negative
                                            ? -static_cast<int32_t>(linear) - 2
                                            : static_cast<int32_t>(linear);
                out[row] = encoded;
            }
        }
    }
}

template <typename q_t, typename index_t>
__global__ void sparse_mla_prefill_fp8_map_score_tiled_kernel(
    const q_t* __restrict__ q, const uint8_t* __restrict__ k_cache,
    const uint8_t* __restrict__ extra_k_cache,
    const int32_t* __restrict__ workspace_map,
    const index_t* __restrict__ indices, const void* __restrict__ topk_length,
    float* __restrict__ scores, int num_tokens, int active_heads, int num_heads,
    int workspace_rows, int topk, int block_size, int extra_block_size, int64_t q_stride_t,
    int64_t q_stride_h, int64_t q_stride_d, int64_t cache_blocks,
    int64_t extra_cache_blocks, int64_t cache_stride0_bytes,
    int64_t extra_cache_stride0_bytes, int64_t indices_stride_t,
    int64_t indices_stride_k, int topk_length_kind, float softmax_scale) {
    const int group_id = threadIdx.x / kScoreGroupSize;
    const int lane = threadIdx.x - group_id * kScoreGroupSize;
    const int k_idx = blockIdx.x * kScoreCandidatesPerBlock + group_id;
    const int head = blockIdx.y;
    const int token = blockIdx.z;
    if (token >= num_tokens || head >= active_heads || k_idx >= topk)
        return;

    const int limit = max(
        0, min(topk,
               load_length_value(topk_length, topk_length_kind, token, topk)));
    const int64_t score_offset =
        (static_cast<int64_t>(token) * active_heads + head) * topk + k_idx;
    if (k_idx >= limit) {
        if (lane == 0)
            scores[score_offset] = -INFINITY;
        return;
    }

    const int64_t workspace_idx = load_index<index_t>(
        indices, static_cast<int64_t>(token) * indices_stride_t +
                     static_cast<int64_t>(k_idx) * indices_stride_k);
    int64_t linear = -1;
    bool use_extra = false;
    if (workspace_idx >= 0 && workspace_idx < workspace_rows)
        linear = static_cast<int64_t>(workspace_map[workspace_idx]);
    if (linear < -1) {
        use_extra = true;
        linear = -linear - 2;
    }
    const int64_t selected_cache_blocks =
        use_extra ? extra_cache_blocks : cache_blocks;
    const int selected_block_size = use_extra ? extra_block_size : block_size;
    const int64_t max_linear =
        selected_cache_blocks * static_cast<int64_t>(selected_block_size);
    if (linear < 0 || linear >= max_linear) {
        if (lane == 0)
            scores[score_offset] = -INFINITY;
        return;
    }

    float partial = 0.0f;
    const uint8_t* selected_cache =
        use_extra ? extra_k_cache : k_cache;
    const int64_t selected_stride =
        use_extra ? extra_cache_stride0_bytes : cache_stride0_bytes;
    if (selected_cache == nullptr) {
        if (lane == 0)
            scores[score_offset] = -INFINITY;
        return;
    }
    const uint8_t* token_ptr =
        token_ptr_from_linear(selected_cache, selected_stride, selected_block_size, linear);
    float scales[kNumQuantBlocks];
    const uint8_t* scale_ptr =
        scale_ptr_from_linear(selected_cache, selected_stride, selected_block_size, linear);
#pragma unroll
    for (int s = 0; s < kNumQuantBlocks; ++s)
        scales[s] = decode_ue8m0_scale(scale_ptr[s]);
    for (int d = lane; d < kHeadDim; d += kScoreGroupSize) {
        const float qv = load_q_value<q_t>(
            q, static_cast<int64_t>(token) * q_stride_t +
                   static_cast<int64_t>(head) * q_stride_h +
                   static_cast<int64_t>(d) * q_stride_d);
        partial += qv * load_cache_value_from_token(token_ptr, scales, d);
    }

    for (int offset = kScoreGroupSize / 2; offset > 0; offset >>= 1)
        partial += __shfl_down_sync(0xffffffffu, partial, offset,
                                    kScoreGroupSize);
    if (lane == 0)
        scores[score_offset] = partial * softmax_scale;
}

template <typename out_t, typename index_t>
__global__ void sparse_mla_prefill_fp8_map_output_kernel(
    const uint8_t* __restrict__ k_cache,
    const uint8_t* __restrict__ extra_k_cache,
    const int32_t* __restrict__ workspace_map,
    const index_t* __restrict__ indices, const void* __restrict__ topk_length,
    const float* __restrict__ scores, const float* __restrict__ lse,
    const float* __restrict__ attn_sink, out_t* __restrict__ out,
    int num_tokens, int active_heads, int num_heads, int workspace_rows,
    int topk, int block_size, int extra_block_size, int64_t cache_blocks, int64_t extra_cache_blocks,
    int64_t cache_stride0_bytes, int64_t extra_cache_stride0_bytes,
    int64_t indices_stride_t, int64_t indices_stride_k, int64_t out_stride_t,
    int64_t out_stride_h, int64_t out_stride_d, int topk_length_kind) {
    const int th = blockIdx.x;
    const int tile = blockIdx.y;
    const int token = th / active_heads;
    const int head = th - token * active_heads;
    const int d = tile * blockDim.x + threadIdx.x;
    if (token >= num_tokens || d >= kHeadDim)
        return;

    const int limit = max(
        0, min(topk,
               load_length_value(topk_length, topk_length_kind, token, topk)));
    const int64_t score_base =
        (static_cast<int64_t>(token) * active_heads + head) * topk;
    float accum = 0.0f;
    for (int k_idx = 0; k_idx < limit; ++k_idx) {
        const float p = scores[score_base + k_idx];
        if (p == 0.0f)
            continue;
        const int64_t workspace_idx = load_index<index_t>(
            indices, static_cast<int64_t>(token) * indices_stride_t +
                         static_cast<int64_t>(k_idx) * indices_stride_k);
        int64_t linear = -1;
        bool use_extra = false;
        if (workspace_idx >= 0 && workspace_idx < workspace_rows)
            linear = static_cast<int64_t>(workspace_map[workspace_idx]);
        if (linear < -1) {
            use_extra = true;
            linear = -linear - 2;
        }
        const uint8_t* selected_cache =
            use_extra ? extra_k_cache : k_cache;
        if (selected_cache == nullptr)
            continue;
        const int64_t selected_cache_blocks =
            use_extra ? extra_cache_blocks : cache_blocks;
        const int selected_block_size = use_extra ? extra_block_size : block_size;
        const int64_t max_selected_linear =
            selected_cache_blocks * static_cast<int64_t>(selected_block_size);
        if (linear < 0 || linear >= max_selected_linear)
            continue;
        accum += p * load_cache_value(
                         selected_cache,
                         use_extra ? extra_cache_stride0_bytes
                                   : cache_stride0_bytes,
                         selected_block_size, linear, d);
    }

    const float row_lse =
        lse[static_cast<int64_t>(token) * num_heads + head];
    const float sink = attn_sink == nullptr ? 0.0f : attn_sink[head];
    const float gate =
        attn_sink == nullptr ? 1.0f : 1.0f / (1.0f + expf(-(row_lse - sink)));
    store_out_value<out_t>(
        out, static_cast<int64_t>(token) * out_stride_t +
                 static_cast<int64_t>(head) * out_stride_h +
                 static_cast<int64_t>(d) * out_stride_d,
        accum * gate);
}

template <typename q_t, typename out_t, typename index_t>
__global__ void sparse_mla_prefill_fp8_map_grouped_kernel(
    const q_t* __restrict__ q, const uint8_t* __restrict__ k_cache,
    const uint8_t* __restrict__ extra_k_cache,
    const int32_t* __restrict__ workspace_map,
    const index_t* __restrict__ indices, const void* __restrict__ topk_length,
    const float* __restrict__ attn_sink, out_t* __restrict__ out,
    float* __restrict__ lse, int num_tokens, int active_heads, int num_heads,
    int workspace_rows, int topk, int block_size, int extra_block_size,
    int64_t cache_blocks, int64_t extra_cache_blocks,
    int64_t cache_stride0_bytes, int64_t extra_cache_stride0_bytes,
    int64_t q_stride_t, int64_t q_stride_h, int64_t q_stride_d,
    int64_t indices_stride_t, int64_t indices_stride_k, int64_t out_stride_t,
    int64_t out_stride_h, int64_t out_stride_d, int topk_length_kind,
    float softmax_scale) {
    extern __shared__ unsigned char smem[];
    unsigned char* cursor = smem;
    float* q_s = reinterpret_cast<float*>(cursor);
    cursor += static_cast<size_t>(kHeadDim) * sizeof(float);
    float* scores = reinterpret_cast<float*>(cursor);
    cursor += static_cast<size_t>(topk) * sizeof(float);
    int64_t* linear_s = align_shared_ptr<int64_t>(cursor);
    cursor += static_cast<size_t>(topk) * sizeof(int64_t);
    int* source_s = align_shared_ptr<int>(cursor);
    cursor += static_cast<size_t>(topk) * sizeof(int);
    float* scale_s = align_shared_ptr<float>(cursor);

    const int th = blockIdx.x;
    const int token = th / active_heads;
    const int head = th - token * active_heads;
    if (token >= num_tokens)
        return;

    for (int d = threadIdx.x; d < kHeadDim; d += blockDim.x) {
        q_s[d] = load_q_value<q_t>(
            q, static_cast<int64_t>(token) * q_stride_t +
                   static_cast<int64_t>(head) * q_stride_h +
                   static_cast<int64_t>(d) * q_stride_d);
    }

    const int limit = max(
        0, min(topk,
               load_length_value(topk_length, topk_length_kind, token, topk)));
    const int64_t max_primary_linear =
        cache_blocks * static_cast<int64_t>(block_size);
    const int64_t max_extra_linear =
        extra_cache_blocks * static_cast<int64_t>(extra_block_size);

    for (int j = threadIdx.x; j < topk; j += blockDim.x) {
        int64_t linear = -1;
        bool use_extra = false;
        if (j < limit) {
            const int64_t workspace_idx = load_index<index_t>(
                indices, static_cast<int64_t>(token) * indices_stride_t +
                             static_cast<int64_t>(j) * indices_stride_k);
            if (workspace_idx >= 0 && workspace_idx < workspace_rows)
                linear = static_cast<int64_t>(workspace_map[workspace_idx]);
            if (linear < -1) {
                use_extra = true;
                linear = -linear - 2;
            }
            const int64_t max_linear =
                use_extra ? max_extra_linear : max_primary_linear;
            if (linear < 0 || linear >= max_linear)
                linear = -1;
        }

        linear_s[j] = linear;
        source_s[j] = use_extra ? 1 : 0;
        float* cand_scales = scale_s + static_cast<int64_t>(j) * kNumQuantBlocks;
#pragma unroll
        for (int s = 0; s < kNumQuantBlocks; ++s)
            cand_scales[s] = 0.0f;
        if (linear >= 0) {
            const uint8_t* selected_cache = use_extra ? extra_k_cache : k_cache;
            const int64_t selected_stride =
                use_extra ? extra_cache_stride0_bytes : cache_stride0_bytes;
            const int selected_block_size =
                use_extra ? extra_block_size : block_size;
            if (selected_cache != nullptr) {
                const uint8_t* scales = scale_ptr_from_linear(
                    selected_cache, selected_stride, selected_block_size,
                    linear);
#pragma unroll
                for (int s = 0; s < kNumQuantBlocks; ++s)
                    cand_scales[s] = decode_ue8m0_scale(scales[s]);
            } else {
                linear_s[j] = -1;
            }
        }
    }
    __syncthreads();

    const int group_id = threadIdx.x / kGroupedScoreGroupSize;
    const int lane = threadIdx.x - group_id * kGroupedScoreGroupSize;
    for (int base = 0; base < topk; base += kGroupedScoreGroups) {
        const int j = base + group_id;
        float partial = 0.0f;
        bool valid = false;
        if (j < topk && j < limit && linear_s[j] >= 0) {
            valid = true;
            const bool use_extra = source_s[j] != 0;
            const uint8_t* selected_cache = use_extra ? extra_k_cache : k_cache;
            const int64_t selected_stride =
                use_extra ? extra_cache_stride0_bytes : cache_stride0_bytes;
            const int selected_block_size =
                use_extra ? extra_block_size : block_size;
            const uint8_t* token_ptr = token_ptr_from_linear(
                selected_cache, selected_stride, selected_block_size,
                linear_s[j]);
            const float* cand_scales =
                scale_s + static_cast<int64_t>(j) * kNumQuantBlocks;
            for (int d = lane; d < kHeadDim; d += kGroupedScoreGroupSize) {
                partial += q_s[d] *
                           load_cache_value_from_token(token_ptr, cand_scales, d);
            }
        }
        for (int offset = kGroupedScoreGroupSize / 2; offset > 0; offset >>= 1)
            partial += __shfl_down_sync(0xffffffffu, partial, offset,
                                        kGroupedScoreGroupSize);
        if (lane == 0 && j < topk)
            scores[j] = valid ? partial * softmax_scale : -INFINITY;
    }
    __syncthreads();

    __shared__ float reduce_buf[kThreads];
    float local_max = -INFINITY;
    for (int j = threadIdx.x; j < topk; j += blockDim.x)
        local_max = fmaxf(local_max, scores[j]);
    reduce_buf[threadIdx.x] = local_max;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset)
            reduce_buf[threadIdx.x] =
                fmaxf(reduce_buf[threadIdx.x], reduce_buf[threadIdx.x + offset]);
        __syncthreads();
    }
    const float row_max = reduce_buf[0];

    float local_sum = 0.0f;
    for (int j = threadIdx.x; j < topk; j += blockDim.x) {
        const float p = isfinite(row_max) ? expf(scores[j] - row_max) : 0.0f;
        scores[j] = p;
        local_sum += p;
    }
    reduce_buf[threadIdx.x] = local_sum;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset)
            reduce_buf[threadIdx.x] += reduce_buf[threadIdx.x + offset];
        __syncthreads();
    }
    const float row_sum = reduce_buf[0];
    const float row_lse = row_sum > 0.0f ? logf(row_sum) + row_max : -INFINITY;
    for (int j = threadIdx.x; j < topk; j += blockDim.x)
        scores[j] = row_sum > 0.0f ? scores[j] / row_sum : 0.0f;
    __syncthreads();

    const float sink = attn_sink == nullptr ? 0.0f : attn_sink[head];
    const float gate =
        attn_sink == nullptr ? 1.0f : 1.0f / (1.0f + expf(-(row_lse - sink)));
    for (int d = threadIdx.x; d < kHeadDim; d += blockDim.x) {
        float accum = 0.0f;
        if (row_sum > 0.0f) {
            for (int j = 0; j < limit; ++j) {
                const float p = scores[j];
                if (p == 0.0f || linear_s[j] < 0)
                    continue;
                const bool use_extra = source_s[j] != 0;
                const uint8_t* selected_cache = use_extra ? extra_k_cache : k_cache;
                const int64_t selected_stride =
                    use_extra ? extra_cache_stride0_bytes : cache_stride0_bytes;
                const int selected_block_size =
                    use_extra ? extra_block_size : block_size;
                const uint8_t* token_ptr = token_ptr_from_linear(
                    selected_cache, selected_stride, selected_block_size,
                    linear_s[j]);
                const float* cand_scales =
                    scale_s + static_cast<int64_t>(j) * kNumQuantBlocks;
                accum += p * load_cache_value_from_token(token_ptr, cand_scales, d);
            }
        }
        store_out_value<out_t>(
            out, static_cast<int64_t>(token) * out_stride_t +
                     static_cast<int64_t>(head) * out_stride_h +
                     static_cast<int64_t>(d) * out_stride_d,
            accum * gate);
    }

    if (threadIdx.x == 0)
        lse[static_cast<int64_t>(token) * num_heads + head] = row_lse;
}


bool is_none(const pybind11::object& obj) {
    return obj.is_none();
}

torch::Tensor tensor_or_empty(const pybind11::object& obj) {
    return is_none(obj) ? torch::Tensor() : obj.cast<torch::Tensor>();
}

int64_t byte_stride(const torch::Tensor& tensor, int dim) {
    return tensor.stride(dim) * tensor.element_size();
}

int length_tensor_kind(const torch::Tensor& tensor) {
    if (!tensor.defined())
        return 1;
    if (tensor.scalar_type() == torch::kInt)
        return 0;
    DG_HOST_ASSERT(tensor.scalar_type() == torch::kInt64);
    return 1;
}

template <typename out_t, typename seq_t, typename block_t, typename gather_t>
void launch_dequantize_gather_k_cache(
    const torch::Tensor& out, const torch::Tensor& k_cache,
    const torch::Tensor& seq_lens, const torch::Tensor& gather_lens,
    const torch::Tensor& block_table, int block_size, int offset) {
    const int num_reqs = static_cast<int>(seq_lens.size(0));
    const int max_out_rows = static_cast<int>(out.size(1)) - offset;
    const int head_dim = min(static_cast<int>(out.size(2)), kHeadDim);
    if (num_reqs <= 0 || max_out_rows <= 0 || head_dim <= 0)
        return;

    const auto stream = at::cuda::getCurrentCUDAStream();
    const dim3 block(kThreads);
    const dim3 grid((head_dim + kThreads - 1) / kThreads, max_out_rows,
                    num_reqs);
    dequantize_gather_k_cache_kernel<out_t, seq_t, block_t, gather_t>
        <<<grid, block, 0, stream>>>(
            reinterpret_cast<out_t*>(out.data_ptr()),
            reinterpret_cast<const uint8_t*>(k_cache.data_ptr()),
            seq_lens.data_ptr<seq_t>(),
            gather_lens.defined() ? gather_lens.data_ptr<gather_t>() : nullptr,
            block_table.data_ptr<block_t>(), num_reqs, max_out_rows, head_dim,
            block_size, offset, k_cache.size(0), byte_stride(k_cache, 0),
            out.stride(0), out.stride(1), out.stride(2), block_table.stride(0),
            block_table.stride(1));
    DG_CUDA_RUNTIME_CHECK(cudaGetLastError());
}

template <typename out_t, typename index_t, typename len_t>
void launch_dequantize_gather_indexed_k_cache(
    const torch::Tensor& out, const torch::Tensor& k_cache,
    const torch::Tensor& indices, const torch::Tensor& topk_length,
    int block_size, int offset) {
    const bool indices_3d = indices.dim() == 3;
    const int batch_size = static_cast<int>(indices.size(0));
    const int topk = static_cast<int>(indices.size(indices_3d ? 2 : 1));
    const int head_dim = min(static_cast<int>(out.size(2)), kHeadDim);
    if (batch_size <= 0 || topk <= 0 || head_dim <= 0)
        return;

    const auto stream = at::cuda::getCurrentCUDAStream();
    const dim3 block(kThreads);
    const dim3 grid((head_dim + kThreads - 1) / kThreads, topk, batch_size);
    dequantize_gather_indexed_k_cache_kernel<out_t, index_t, len_t>
        <<<grid, block, 0, stream>>>(
            reinterpret_cast<out_t*>(out.data_ptr()),
            reinterpret_cast<const uint8_t*>(k_cache.data_ptr()),
            indices.data_ptr(),
            topk_length.defined() ? topk_length.data_ptr<len_t>() : nullptr,
            batch_size, topk, head_dim, block_size, offset, k_cache.size(0),
            byte_stride(k_cache, 0), out.stride(0), out.stride(1),
            out.stride(2), indices.stride(0),
            indices_3d ? indices.stride(1) : 0,
            indices.stride(indices_3d ? 2 : 1), indices_3d);
    DG_CUDA_RUNTIME_CHECK(cudaGetLastError());
}

template <typename q_t, typename out_t, typename index_t>
void launch_sparse_mla_decode(
    const torch::Tensor& q, const torch::Tensor& k_cache,
    const torch::Tensor& indices, const torch::Tensor& topk_length,
    const torch::Tensor& attn_sink, const torch::Tensor& extra_k_cache,
    const torch::Tensor& extra_indices, const torch::Tensor& extra_topk_length,
    const torch::Tensor& out, const torch::Tensor& lse, double softmax_scale) {
    const int batch_size = static_cast<int>(q.size(0));
    const int num_heads = static_cast<int>(q.size(2));
    const int active_heads = sm120_active_heads(num_heads);
    const int block_size = static_cast<int>(k_cache.size(1));
    const int main_topk = static_cast<int>(indices.size(2));
    const int extra_topk = extra_indices.defined() ? static_cast<int>(extra_indices.size(2)) : 0;
    const int candidate_count = main_topk + extra_topk;
    DG_HOST_ASSERT(candidate_count > 0);
    DG_HOST_ASSERT(candidate_count <= 8192);

    const auto stream = at::cuda::getCurrentCUDAStream();
    const int grid = batch_size * active_heads;

    if (candidate_count <= kMaxGroupedCandidateSlots) {
        const size_t grouped_shared_bytes =
            grouped_decode_shared_bytes(candidate_count);
        sparse_mla_decode_grouped_kernel<q_t, out_t, index_t>
            <<<grid, kThreads, grouped_shared_bytes, stream>>>(
                reinterpret_cast<const q_t*>(q.data_ptr()),
                reinterpret_cast<const uint8_t*>(k_cache.data_ptr()),
                indices.data_ptr(),
                topk_length.defined() ? topk_length.data_ptr<int64_t>() : nullptr,
                attn_sink.defined() ? attn_sink.data_ptr<float>() : nullptr,
                extra_k_cache.defined()
                    ? reinterpret_cast<const uint8_t*>(extra_k_cache.data_ptr())
                    : nullptr,
                extra_indices.defined() ? extra_indices.data_ptr() : nullptr,
                extra_topk_length.defined()
                    ? extra_topk_length.data_ptr<int64_t>()
                    : nullptr,
                reinterpret_cast<out_t*>(out.data_ptr()), lse.data_ptr<float>(),
                batch_size, active_heads, num_heads, block_size, k_cache.size(0),
                extra_k_cache.defined() ? extra_k_cache.size(0) : 0, main_topk,
                extra_topk, candidate_count, q.stride(0), q.stride(1),
                q.stride(2), q.stride(3), out.stride(0), out.stride(1),
                out.stride(2), out.stride(3), indices.stride(0),
                indices.stride(1), indices.stride(2),
                extra_indices.defined() ? extra_indices.stride(0) : 0,
                extra_indices.defined() ? extra_indices.stride(1) : 0,
                extra_indices.defined() ? extra_indices.stride(2) : 0,
                byte_stride(k_cache, 0),
                extra_k_cache.defined() ? byte_stride(extra_k_cache, 0) : 0,
                static_cast<float>(softmax_scale));
        DG_CUDA_RUNTIME_CHECK(cudaGetLastError());
        return;
    }

    const size_t shared_bytes = static_cast<size_t>(candidate_count) * sizeof(float);

    sparse_mla_decode_kernel<q_t, out_t, index_t><<<grid, kThreads, shared_bytes, stream>>>(
        reinterpret_cast<const q_t*>(q.data_ptr()), reinterpret_cast<const uint8_t*>(k_cache.data_ptr()),
        indices.data_ptr(), topk_length.defined() ? topk_length.data_ptr<int64_t>() : nullptr,
        attn_sink.defined() ? attn_sink.data_ptr<float>() : nullptr,
        extra_k_cache.defined() ? reinterpret_cast<const uint8_t*>(extra_k_cache.data_ptr()) : nullptr,
        extra_indices.defined() ? extra_indices.data_ptr() : nullptr,
        extra_topk_length.defined() ? extra_topk_length.data_ptr<int64_t>() : nullptr,
        reinterpret_cast<out_t*>(out.data_ptr()), lse.data_ptr<float>(), batch_size,
        active_heads, num_heads, block_size, k_cache.size(0),
        extra_k_cache.defined() ? extra_k_cache.size(0) : 0, main_topk, extra_topk,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3), out.stride(0),
        out.stride(1), out.stride(2), out.stride(3), indices.stride(0),
        indices.stride(1), indices.stride(2),
        extra_indices.defined() ? extra_indices.stride(0) : 0,
        extra_indices.defined() ? extra_indices.stride(1) : 0,
        extra_indices.defined() ? extra_indices.stride(2) : 0,
        byte_stride(k_cache, 0),
        extra_k_cache.defined() ? byte_stride(extra_k_cache, 0) : 0,
        static_cast<float>(softmax_scale));
    DG_CUDA_RUNTIME_CHECK(cudaGetLastError());
}

template <typename q_t, typename kv_t, typename out_t>
void launch_sparse_mla_decode_from_workspace(
    const torch::Tensor& q, const torch::Tensor& kv_workspace,
    const torch::Tensor& topk_length, const torch::Tensor& extra_topk_length,
    const torch::Tensor& attn_sink, const torch::Tensor& out,
    const torch::Tensor& lse, int main_topk, int extra_topk,
    double softmax_scale) {
    const int batch_size = static_cast<int>(q.size(0));
    const int num_heads = static_cast<int>(q.size(2));
    const int active_heads = sm120_active_heads(num_heads);
    const int candidate_slots = main_topk + extra_topk;
    DG_HOST_ASSERT(batch_size > 0);
    DG_HOST_ASSERT(num_heads > 0);
    DG_HOST_ASSERT(candidate_slots > 0);
    DG_HOST_ASSERT(candidate_slots <= kMaxGroupedCandidateSlots);
    DG_HOST_ASSERT(kv_workspace.size(0) >= batch_size);
    DG_HOST_ASSERT(kv_workspace.size(1) >= candidate_slots);
    DG_HOST_ASSERT(kv_workspace.size(2) >= kHeadDim);

    const auto stream = at::cuda::getCurrentCUDAStream();
    const int grid = batch_size * num_heads;
    const size_t shared_bytes =
        static_cast<size_t>(kHeadDim + candidate_slots) * sizeof(float);
    sparse_mla_decode_workspace_kernel<q_t, kv_t, out_t>
        <<<grid, kThreads, shared_bytes, stream>>>(
            reinterpret_cast<const q_t*>(q.data_ptr()),
            reinterpret_cast<const kv_t*>(kv_workspace.data_ptr()),
            topk_length.defined() ? topk_length.data_ptr() : nullptr,
            extra_topk_length.defined() ? extra_topk_length.data_ptr()
                                        : nullptr,
            attn_sink.defined() ? attn_sink.data_ptr<float>() : nullptr,
            reinterpret_cast<out_t*>(out.data_ptr()), lse.data_ptr<float>(),
            batch_size, active_heads, num_heads, main_topk, extra_topk,
            candidate_slots, q.stride(0), q.stride(1), q.stride(2),
            q.stride(3), kv_workspace.stride(0), kv_workspace.stride(1),
            kv_workspace.stride(2), out.stride(0), out.stride(1),
            out.stride(2), out.stride(3), length_tensor_kind(topk_length),
            length_tensor_kind(extra_topk_length),
            static_cast<float>(softmax_scale));
    DG_CUDA_RUNTIME_CHECK(cudaGetLastError());
}

template <typename q_t, typename kv_t, typename out_t>
void launch_sparse_mla_decode_from_workspace_split(
    const torch::Tensor& q, const torch::Tensor& kv_workspace,
    const torch::Tensor& topk_length, const torch::Tensor& extra_topk_length,
    const torch::Tensor& attn_sink, const torch::Tensor& out,
    const torch::Tensor& lse, int main_topk, int extra_topk,
    double softmax_scale) {
    const int batch_size = static_cast<int>(q.size(0));
    const int num_heads = static_cast<int>(q.size(2));
    const int active_heads = sm120_active_heads(num_heads);
    const int candidate_slots = main_topk + extra_topk;
    DG_HOST_ASSERT(batch_size > 0);
    DG_HOST_ASSERT(num_heads > 0);
    DG_HOST_ASSERT(candidate_slots > 0);
    DG_HOST_ASSERT(candidate_slots <= 4096);

    const auto stream = at::cuda::getCurrentCUDAStream();
    const bool fast_path = sm120_fast_sparse_mla_enabled();

    // FLASH_FUSION_MMA_SPLITK: split candidate_slots across kFlashMmaSplitK=4
    // partial blocks per (b, h_block); reduce kernel merges via online-softmax.
    // Goal: lift grid from 32 (b=16) to 128 blocks (~70% SM util).
    // Default OFF — gated on DG_SM120_FLASH_FUSION_MMA_SPLITK=1.
    if constexpr (std::is_same_v<q_t, __nv_bfloat16> &&
                  std::is_same_v<kv_t, __nv_bfloat16>) {
        if (fast_path && sm120_flash_fusion_mma_splitk_enabled() &&
            candidate_slots <= kFlashMmaTopkMax) {
            const int h_block_count =
                (active_heads + kFlashMmaM - 1) / kFlashMmaM;
            const int split_step =
                (candidate_slots + kFlashMmaSplitK - 1) / kFlashMmaSplitK;
            // Effective split_k = number of splits actually doing work.
            const int eff_split_k = (candidate_slots + split_step - 1) / split_step;

            auto O_partial_t = torch::empty(
                {batch_size, h_block_count, eff_split_k, kFlashMmaM, kHeadDim},
                q.options().dtype(torch::kFloat32));
            auto m_partial_t = torch::empty(
                {batch_size, h_block_count, eff_split_k, kFlashMmaM},
                q.options().dtype(torch::kFloat32));
            auto l_partial_t = torch::empty(
                {batch_size, h_block_count, eff_split_k, kFlashMmaM},
                q.options().dtype(torch::kFloat32));

            const size_t partial_smem_bytes =
                static_cast<size_t>(kFlashMmaM) * kHeadDim *
                    sizeof(__nv_bfloat16) +
                static_cast<size_t>(kFlashMmaTileK) * kFlashMmaKStride *
                    sizeof(__nv_bfloat16) +
                static_cast<size_t>(kFlashMmaM) * kHeadDim * sizeof(float) +
                static_cast<size_t>(kFlashMmaM) * kFlashMmaTileK *
                    sizeof(float) +
                static_cast<size_t>(kFlashMmaM) * 5 * sizeof(float);
            const bool spk_scalar_check = sm120_flash_fusion_mma_scalar_check();
            const dim3 partial_grid(batch_size, h_block_count, eff_split_k);
            auto launch_partial = [&](auto check_tag) {
                constexpr bool kCheck = decltype(check_tag)::value;
                auto* kernel_ptr =
                    sparse_mla_workspace_flash_mma_splitk_partial_kernel<
                        q_t, kv_t, kCheck>;
                DG_CUDA_RUNTIME_CHECK(cudaFuncSetAttribute(
                    kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize,
                    static_cast<int>(partial_smem_bytes)));
                kernel_ptr<<<partial_grid, kFlashMmaThreads, partial_smem_bytes,
                             stream>>>(
                    reinterpret_cast<const q_t*>(q.data_ptr()),
                    reinterpret_cast<const kv_t*>(kv_workspace.data_ptr()),
                    topk_length.defined() ? topk_length.data_ptr() : nullptr,
                    extra_topk_length.defined() ? extra_topk_length.data_ptr()
                                                : nullptr,
                    O_partial_t.data_ptr<float>(),
                    m_partial_t.data_ptr<float>(),
                    l_partial_t.data_ptr<float>(), batch_size, active_heads,
                    num_heads, main_topk, extra_topk, candidate_slots,
                    h_block_count, eff_split_k, split_step, q.stride(0),
                    q.stride(2), q.stride(3), kv_workspace.stride(0),
                    kv_workspace.stride(1), kv_workspace.stride(2),
                    length_tensor_kind(topk_length),
                    length_tensor_kind(extra_topk_length),
                    static_cast<float>(softmax_scale));
            };
            if (spk_scalar_check) launch_partial(std::true_type{});
            else launch_partial(std::false_type{});
            DG_CUDA_RUNTIME_CHECK(cudaGetLastError());

            // Reduce.
            const size_t reduce_smem_bytes =
                static_cast<size_t>(kFlashMmaM) * 3 * sizeof(float) +
                static_cast<size_t>(kFlashMmaSplitK) * kFlashMmaM *
                    sizeof(float);
            const dim3 reduce_grid(batch_size, h_block_count);
            sparse_mla_workspace_flash_mma_splitk_reduce_kernel<out_t>
                <<<reduce_grid, kFlashMmaThreads, reduce_smem_bytes, stream>>>(
                    O_partial_t.data_ptr<float>(),
                    m_partial_t.data_ptr<float>(),
                    l_partial_t.data_ptr<float>(),
                    attn_sink.defined() ? attn_sink.data_ptr<float>() : nullptr,
                    reinterpret_cast<out_t*>(out.data_ptr()),
                    lse.data_ptr<float>(), batch_size, active_heads, num_heads,
                    h_block_count, eff_split_k, out.stride(0), out.stride(2),
                    out.stride(3));
            DG_CUDA_RUNTIME_CHECK(cudaGetLastError());
            return;
        }
    }

    // FLASH_FUSION_MMA: bf16 mma.sync m16n8k16 QK + scalar-PV online flash-attn.
    // Same K-stationary intent as FLASH_FUSION but K is K-tiled (TILE_K=32) so
    // candidate_slots up to kFlashMmaTopkMax=128 fit in the 99 KB smem budget.
    // Default OFF — gated on DG_SM120_FLASH_FUSION_MMA=1.
    if (fast_path && sm120_flash_fusion_mma_enabled() &&
        candidate_slots <= kFlashMmaTopkMax) {
        const size_t flash_mma_smem_bytes =
            static_cast<size_t>(kFlashMmaM) * kHeadDim *
                sizeof(__nv_bfloat16) +                                    // Q
            static_cast<size_t>(kFlashMmaTileK) * kFlashMmaKStride *
                sizeof(__nv_bfloat16) +                                    // K_tile
            static_cast<size_t>(kFlashMmaM) * kHeadDim * sizeof(float) +   // O_accum
            static_cast<size_t>(kFlashMmaM) * kFlashMmaTileK *
                sizeof(float) +                                            // scores
            static_cast<size_t>(kFlashMmaM) * 5 * sizeof(float);           // m,l,tm,resc,tsum
        const bool mma_scalar_check = sm120_flash_fusion_mma_scalar_check();
        const dim3 mma_grid(
            batch_size, (active_heads + kFlashMmaM - 1) / kFlashMmaM);
        auto launch_mma = [&](auto check_tag) {
            constexpr bool kCheck = decltype(check_tag)::value;
            auto* kernel_ptr =
                sparse_mla_workspace_flash_mma_kernel<q_t, kv_t, out_t, kCheck>;
            DG_CUDA_RUNTIME_CHECK(cudaFuncSetAttribute(
                kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(flash_mma_smem_bytes)));
            kernel_ptr<<<mma_grid, kFlashMmaThreads, flash_mma_smem_bytes,
                         stream>>>(
                reinterpret_cast<const q_t*>(q.data_ptr()),
                reinterpret_cast<const kv_t*>(kv_workspace.data_ptr()),
                topk_length.defined() ? topk_length.data_ptr() : nullptr,
                extra_topk_length.defined() ? extra_topk_length.data_ptr()
                                            : nullptr,
                attn_sink.defined() ? attn_sink.data_ptr<float>() : nullptr,
                reinterpret_cast<out_t*>(out.data_ptr()),
                lse.data_ptr<float>(), batch_size, active_heads, num_heads,
                main_topk, extra_topk, candidate_slots, q.stride(0),
                q.stride(2), q.stride(3), kv_workspace.stride(0),
                kv_workspace.stride(1), kv_workspace.stride(2), out.stride(0),
                out.stride(2), out.stride(3), length_tensor_kind(topk_length),
                length_tensor_kind(extra_topk_length),
                static_cast<float>(softmax_scale));
        };
        if (mma_scalar_check) {
            launch_mma(std::true_type{});
        } else {
            launch_mma(std::false_type{});
        }
        DG_CUDA_RUNTIME_CHECK(cudaGetLastError());
        return;
    }

    // FLASH_FUSION early-out: replaces score+softmax+output with one kernel
    // that keeps K smem-resident across QK and PV. Bounded by smem budget
    // (candidate_slots ≤ kFlashTopkMax = 128) and bf16 inputs (kernel
    // converts via float, fastest path is q_t == kv_t == bf16).
    if (fast_path && sm120_flash_fusion_enabled() &&
        candidate_slots <= kFlashTopkMax) {
        const size_t flash_smem_bytes =
            static_cast<size_t>(kHeadDim) * sizeof(__nv_bfloat16) +
            static_cast<size_t>(candidate_slots) * kFlashKStride *
                sizeof(__nv_bfloat16) +
            static_cast<size_t>(candidate_slots) * sizeof(float);
        auto* kernel_ptr = sparse_mla_workspace_flash_kernel<q_t, kv_t, out_t>;
        DG_CUDA_RUNTIME_CHECK(cudaFuncSetAttribute(
            kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(flash_smem_bytes)));
        const dim3 flash_grid(batch_size, active_heads);
        kernel_ptr<<<flash_grid, kThreads, flash_smem_bytes, stream>>>(
            reinterpret_cast<const q_t*>(q.data_ptr()),
            reinterpret_cast<const kv_t*>(kv_workspace.data_ptr()),
            topk_length.defined() ? topk_length.data_ptr() : nullptr,
            extra_topk_length.defined() ? extra_topk_length.data_ptr()
                                        : nullptr,
            attn_sink.defined() ? attn_sink.data_ptr<float>() : nullptr,
            reinterpret_cast<out_t*>(out.data_ptr()), lse.data_ptr<float>(),
            batch_size, active_heads, num_heads, main_topk, extra_topk,
            candidate_slots, q.stride(0), q.stride(2), q.stride(3),
            kv_workspace.stride(0), kv_workspace.stride(1),
            kv_workspace.stride(2), out.stride(0), out.stride(2),
            out.stride(3), length_tensor_kind(topk_length),
            length_tensor_kind(extra_topk_length),
            static_cast<float>(softmax_scale));
        DG_CUDA_RUNTIME_CHECK(cudaGetLastError());
        return;
    }

    auto scores = torch::empty({batch_size, active_heads, candidate_slots},
                               q.options().dtype(torch::kFloat32));
    const dim3 score_grid(
        (candidate_slots + kScoreCandidatesPerBlock - 1) /
            kScoreCandidatesPerBlock,
        active_heads, batch_size);
    bool used_mma_score = false;
    if constexpr (std::is_same_v<q_t, __nv_bfloat16> &&
                  std::is_same_v<kv_t, __nv_bfloat16>) {
        // Split-K SCORE MMA — preferred when active_heads fits the M=32 tile.
        if (fast_path && sm120_score_mma_splitk_enabled() &&
            active_heads <= kScoreMmaSpkM) {
            auto partials = torch::empty(
                {kScoreMmaSpkSplit, batch_size, active_heads, candidate_slots},
                q.options().dtype(torch::kFloat32));
            const dim3 spk_grid(
                (candidate_slots + kScoreMmaSpkN - 1) / kScoreMmaSpkN,
                batch_size, kScoreMmaSpkSplit);
            const bool scalar_check = sm120_score_mma_splitk_scalar_check();
            if (scalar_check) {
                sparse_mla_workspace_score_mma_splitk_kernel<q_t, kv_t, true>
                    <<<spk_grid, kScoreMmaSpkThreads, kScoreMmaSpkSmemBytes,
                       stream>>>(
                        reinterpret_cast<const q_t*>(q.data_ptr()),
                        reinterpret_cast<const kv_t*>(kv_workspace.data_ptr()),
                        partials.data_ptr<float>(), batch_size, active_heads,
                        num_heads, candidate_slots, q.stride(0), q.stride(2),
                        q.stride(3), kv_workspace.stride(0),
                        kv_workspace.stride(1), kv_workspace.stride(2));
            } else {
                sparse_mla_workspace_score_mma_splitk_kernel<q_t, kv_t, false>
                    <<<spk_grid, kScoreMmaSpkThreads, kScoreMmaSpkSmemBytes,
                       stream>>>(
                        reinterpret_cast<const q_t*>(q.data_ptr()),
                        reinterpret_cast<const kv_t*>(kv_workspace.data_ptr()),
                        partials.data_ptr<float>(), batch_size, active_heads,
                        num_heads, candidate_slots, q.stride(0), q.stride(2),
                        q.stride(3), kv_workspace.stride(0),
                        kv_workspace.stride(1), kv_workspace.stride(2));
            }
            const int reduce_block = 128;
            const dim3 reduce_grid(
                (candidate_slots + reduce_block - 1) / reduce_block,
                active_heads, batch_size);
            sparse_mla_workspace_score_mma_splitk_reduce_kernel
                <<<reduce_grid, reduce_block, 0, stream>>>(
                    partials.data_ptr<float>(),
                    topk_length.defined() ? topk_length.data_ptr() : nullptr,
                    extra_topk_length.defined()
                        ? extra_topk_length.data_ptr()
                        : nullptr,
                    scores.data_ptr<float>(), batch_size, active_heads,
                    main_topk, extra_topk, candidate_slots,
                    length_tensor_kind(topk_length),
                    length_tensor_kind(extra_topk_length),
                    static_cast<float>(softmax_scale));
            used_mma_score = true;
        } else if (fast_path && sm120_score_mma_enabled()) {
            const dim3 mma_grid(
                (candidate_slots + kScoreMmaN - 1) / kScoreMmaN,
                (active_heads + kScoreMmaM - 1) / kScoreMmaM, batch_size);
            const bool scalar_check = sm120_score_mma_scalar_check();
            if (scalar_check) {
                sparse_mla_workspace_score_mma_kernel<q_t, kv_t, true>
                    <<<mma_grid, kScoreMmaThreads, kScoreMmaSmemBytes, stream>>>(
                        reinterpret_cast<const q_t*>(q.data_ptr()),
                        reinterpret_cast<const kv_t*>(kv_workspace.data_ptr()),
                        topk_length.defined() ? topk_length.data_ptr() : nullptr,
                        extra_topk_length.defined() ? extra_topk_length.data_ptr()
                                                    : nullptr,
                        scores.data_ptr<float>(), batch_size, active_heads,
                        num_heads, main_topk, extra_topk, candidate_slots,
                        q.stride(0), q.stride(2), q.stride(3),
                        kv_workspace.stride(0), kv_workspace.stride(1),
                        kv_workspace.stride(2), length_tensor_kind(topk_length),
                        length_tensor_kind(extra_topk_length),
                        static_cast<float>(softmax_scale));
            } else {
                sparse_mla_workspace_score_mma_kernel<q_t, kv_t, false>
                    <<<mma_grid, kScoreMmaThreads, kScoreMmaSmemBytes, stream>>>(
                        reinterpret_cast<const q_t*>(q.data_ptr()),
                        reinterpret_cast<const kv_t*>(kv_workspace.data_ptr()),
                        topk_length.defined() ? topk_length.data_ptr() : nullptr,
                        extra_topk_length.defined() ? extra_topk_length.data_ptr()
                                                    : nullptr,
                        scores.data_ptr<float>(), batch_size, active_heads,
                        num_heads, main_topk, extra_topk, candidate_slots,
                        q.stride(0), q.stride(2), q.stride(3),
                        kv_workspace.stride(0), kv_workspace.stride(1),
                        kv_workspace.stride(2), length_tensor_kind(topk_length),
                        length_tensor_kind(extra_topk_length),
                        static_cast<float>(softmax_scale));
            }
            used_mma_score = true;
        }
    }
    const bool qstat_vec_path =
        used_mma_score
            ? false
            : (fast_path && sm120_score_qstat_vec_enabled() &&
               std::is_same_v<q_t, __nv_bfloat16> &&
               std::is_same_v<kv_t, __nv_bfloat16> &&
               kv_workspace.stride(2) == 1 && (kHeadDim % 32 == 0));
    if (used_mma_score) {
        // already launched
    } else if (qstat_vec_path) {
        if constexpr (std::is_same_v<q_t, __nv_bfloat16> &&
                      std::is_same_v<kv_t, __nv_bfloat16>) {
            const bool scalar_check = sm120_score_qstat_vec_scalar_check();
            const bool vec8 = sm120_score_qstat_vec8_enabled() &&
                              (kHeadDim % (kScoreGroupSize * 8) == 0) &&
                              (reinterpret_cast<uintptr_t>(
                                   kv_workspace.data_ptr()) %
                                   16 ==
                               0);
            if (scalar_check) {
                sparse_mla_workspace_score_tiled_qstat_vec_kernel<q_t, kv_t,
                                                                  true, 4>
                    <<<score_grid, kThreads, 0, stream>>>(
                        reinterpret_cast<const q_t*>(q.data_ptr()),
                        reinterpret_cast<const kv_t*>(kv_workspace.data_ptr()),
                        topk_length.defined() ? topk_length.data_ptr() : nullptr,
                        extra_topk_length.defined()
                            ? extra_topk_length.data_ptr()
                            : nullptr,
                        scores.data_ptr<float>(), batch_size, active_heads,
                        num_heads, main_topk, extra_topk, candidate_slots,
                        q.stride(0), q.stride(2), q.stride(3),
                        kv_workspace.stride(0), kv_workspace.stride(1),
                        kv_workspace.stride(2), length_tensor_kind(topk_length),
                        length_tensor_kind(extra_topk_length),
                        static_cast<float>(softmax_scale));
            } else if (vec8) {
                sparse_mla_workspace_score_tiled_qstat_vec_kernel<q_t, kv_t,
                                                                  false, 8>
                    <<<score_grid, kThreads, 0, stream>>>(
                        reinterpret_cast<const q_t*>(q.data_ptr()),
                        reinterpret_cast<const kv_t*>(kv_workspace.data_ptr()),
                        topk_length.defined() ? topk_length.data_ptr() : nullptr,
                        extra_topk_length.defined()
                            ? extra_topk_length.data_ptr()
                            : nullptr,
                        scores.data_ptr<float>(), batch_size, active_heads,
                        num_heads, main_topk, extra_topk, candidate_slots,
                        q.stride(0), q.stride(2), q.stride(3),
                        kv_workspace.stride(0), kv_workspace.stride(1),
                        kv_workspace.stride(2), length_tensor_kind(topk_length),
                        length_tensor_kind(extra_topk_length),
                        static_cast<float>(softmax_scale));
            } else {
                sparse_mla_workspace_score_tiled_qstat_vec_kernel<q_t, kv_t,
                                                                  false, 4>
                    <<<score_grid, kThreads, 0, stream>>>(
                        reinterpret_cast<const q_t*>(q.data_ptr()),
                        reinterpret_cast<const kv_t*>(kv_workspace.data_ptr()),
                        topk_length.defined() ? topk_length.data_ptr() : nullptr,
                        extra_topk_length.defined()
                            ? extra_topk_length.data_ptr()
                            : nullptr,
                        scores.data_ptr<float>(), batch_size, active_heads,
                        num_heads, main_topk, extra_topk, candidate_slots,
                        q.stride(0), q.stride(2), q.stride(3),
                        kv_workspace.stride(0), kv_workspace.stride(1),
                        kv_workspace.stride(2), length_tensor_kind(topk_length),
                        length_tensor_kind(extra_topk_length),
                        static_cast<float>(softmax_scale));
            }
        }
    } else if (fast_path) {
        sparse_mla_workspace_score_tiled_qstat_kernel<q_t, kv_t>
            <<<score_grid, kThreads, 0, stream>>>(
                reinterpret_cast<const q_t*>(q.data_ptr()),
                reinterpret_cast<const kv_t*>(kv_workspace.data_ptr()),
                topk_length.defined() ? topk_length.data_ptr() : nullptr,
                extra_topk_length.defined() ? extra_topk_length.data_ptr()
                                            : nullptr,
                scores.data_ptr<float>(), batch_size, active_heads, num_heads,
                main_topk, extra_topk, candidate_slots, q.stride(0),
                q.stride(2), q.stride(3), kv_workspace.stride(0),
                kv_workspace.stride(1), kv_workspace.stride(2),
                length_tensor_kind(topk_length),
                length_tensor_kind(extra_topk_length),
                static_cast<float>(softmax_scale));
    } else {
        sparse_mla_workspace_score_tiled_kernel<q_t, kv_t>
            <<<score_grid, kThreads, 0, stream>>>(
                reinterpret_cast<const q_t*>(q.data_ptr()),
                reinterpret_cast<const kv_t*>(kv_workspace.data_ptr()),
                topk_length.defined() ? topk_length.data_ptr() : nullptr,
                extra_topk_length.defined() ? extra_topk_length.data_ptr()
                                            : nullptr,
                scores.data_ptr<float>(), batch_size, active_heads, num_heads,
                main_topk, extra_topk, candidate_slots, q.stride(0),
                q.stride(2), q.stride(3), kv_workspace.stride(0),
                kv_workspace.stride(1), kv_workspace.stride(2),
                length_tensor_kind(topk_length),
                length_tensor_kind(extra_topk_length),
                static_cast<float>(softmax_scale));
    }
    DG_CUDA_RUNTIME_CHECK(cudaGetLastError());

    const bool fused_softmax_path =
        fast_path && sm120_output_fused_softmax_enabled();
    if (!fused_softmax_path) {
        sparse_mla_softmax_kernel<<<batch_size * active_heads, kThreads, 0,
                                    stream>>>(
            scores.data_ptr<float>(), lse.data_ptr<float>(),
            attn_sink.defined() ? attn_sink.data_ptr<float>() : nullptr,
            batch_size, num_heads, active_heads, candidate_slots);
        DG_CUDA_RUNTIME_CHECK(cudaGetLastError());
    }

    const dim3 out_grid(batch_size * active_heads,
                        (kHeadDim + kThreads - 1) / kThreads);
    if (fused_softmax_path) {
        const size_t out_smem_bytes =
            static_cast<size_t>(candidate_slots) * sizeof(float);
        sparse_mla_workspace_output_fused_softmax_kernel<kv_t, out_t>
            <<<out_grid, kThreads, out_smem_bytes, stream>>>(
                reinterpret_cast<const kv_t*>(kv_workspace.data_ptr()),
                topk_length.defined() ? topk_length.data_ptr() : nullptr,
                extra_topk_length.defined() ? extra_topk_length.data_ptr()
                                            : nullptr,
                scores.data_ptr<float>(),
                attn_sink.defined() ? attn_sink.data_ptr<float>() : nullptr,
                reinterpret_cast<out_t*>(out.data_ptr()),
                lse.data_ptr<float>(), batch_size, active_heads, num_heads,
                main_topk, extra_topk, candidate_slots,
                kv_workspace.stride(0), kv_workspace.stride(1),
                kv_workspace.stride(2), out.stride(0), out.stride(2),
                out.stride(3), length_tensor_kind(topk_length),
                length_tensor_kind(extra_topk_length));
    } else if (fast_path && sm120_output_mma_enabled() &&
        active_heads <= kOutMmaM && candidate_slots <= kOutMmaKMax) {
        const dim3 out_mma_grid(batch_size,
                                (kHeadDim + kOutMmaN - 1) / kOutMmaN);
        const bool scalar_check = sm120_output_mma_scalar_check();
        if (scalar_check) {
            sparse_mla_workspace_output_mma_kernel<kv_t, out_t, true>
                <<<out_mma_grid, kOutMmaThreads, 0, stream>>>(
                    reinterpret_cast<const kv_t*>(kv_workspace.data_ptr()),
                    scores.data_ptr<float>(), lse.data_ptr<float>(),
                    attn_sink.defined() ? attn_sink.data_ptr<float>()
                                        : nullptr,
                    reinterpret_cast<out_t*>(out.data_ptr()), batch_size,
                    active_heads, num_heads, candidate_slots,
                    kv_workspace.stride(0), kv_workspace.stride(1),
                    kv_workspace.stride(2), out.stride(0), out.stride(2),
                    out.stride(3));
        } else {
            sparse_mla_workspace_output_mma_kernel<kv_t, out_t, false>
                <<<out_mma_grid, kOutMmaThreads, 0, stream>>>(
                    reinterpret_cast<const kv_t*>(kv_workspace.data_ptr()),
                    scores.data_ptr<float>(), lse.data_ptr<float>(),
                    attn_sink.defined() ? attn_sink.data_ptr<float>()
                                        : nullptr,
                    reinterpret_cast<out_t*>(out.data_ptr()), batch_size,
                    active_heads, num_heads, candidate_slots,
                    kv_workspace.stride(0), kv_workspace.stride(1),
                    kv_workspace.stride(2), out.stride(0), out.stride(2),
                    out.stride(3));
        }
    } else if (fast_path && sm120_output_sstat_vec_enabled() &&
               std::is_same_v<kv_t, __nv_bfloat16> &&
               kv_workspace.stride(2) == 1 && (kHeadDim % 4) == 0) {
        if constexpr (std::is_same_v<kv_t, __nv_bfloat16>) {
            const size_t out_smem_bytes =
                static_cast<size_t>(candidate_slots) * sizeof(float);
            const dim3 out_vec_grid(
                batch_size * active_heads,
                (kHeadDim + kThreads * 4 - 1) / (kThreads * 4));
            sparse_mla_workspace_output_sstat_vec_kernel<kv_t, out_t>
                <<<out_vec_grid, kThreads, out_smem_bytes, stream>>>(
                    reinterpret_cast<const kv_t*>(kv_workspace.data_ptr()),
                    topk_length.defined() ? topk_length.data_ptr() : nullptr,
                    extra_topk_length.defined()
                        ? extra_topk_length.data_ptr()
                        : nullptr,
                    scores.data_ptr<float>(), lse.data_ptr<float>(),
                    attn_sink.defined() ? attn_sink.data_ptr<float>()
                                        : nullptr,
                    reinterpret_cast<out_t*>(out.data_ptr()), batch_size,
                    active_heads, num_heads, main_topk, extra_topk,
                    candidate_slots, kv_workspace.stride(0),
                    kv_workspace.stride(1), kv_workspace.stride(2),
                    out.stride(0), out.stride(2), out.stride(3),
                    length_tensor_kind(topk_length),
                    length_tensor_kind(extra_topk_length));
        }
    } else if (fast_path) {
        const size_t out_smem_bytes =
            static_cast<size_t>(candidate_slots) * sizeof(float);
        sparse_mla_workspace_output_sstat_kernel<kv_t, out_t>
            <<<out_grid, kThreads, out_smem_bytes, stream>>>(
                reinterpret_cast<const kv_t*>(kv_workspace.data_ptr()),
                topk_length.defined() ? topk_length.data_ptr() : nullptr,
                extra_topk_length.defined() ? extra_topk_length.data_ptr()
                                            : nullptr,
                scores.data_ptr<float>(), lse.data_ptr<float>(),
                attn_sink.defined() ? attn_sink.data_ptr<float>() : nullptr,
                reinterpret_cast<out_t*>(out.data_ptr()), batch_size,
                active_heads, num_heads, main_topk, extra_topk, candidate_slots,
                kv_workspace.stride(0), kv_workspace.stride(1),
                kv_workspace.stride(2), out.stride(0), out.stride(2),
                out.stride(3), length_tensor_kind(topk_length),
                length_tensor_kind(extra_topk_length));
    } else {
        sparse_mla_workspace_output_kernel<kv_t, out_t>
            <<<out_grid, kThreads, 0, stream>>>(
                reinterpret_cast<const kv_t*>(kv_workspace.data_ptr()),
                topk_length.defined() ? topk_length.data_ptr() : nullptr,
                extra_topk_length.defined() ? extra_topk_length.data_ptr()
                                            : nullptr,
                scores.data_ptr<float>(), lse.data_ptr<float>(),
                attn_sink.defined() ? attn_sink.data_ptr<float>() : nullptr,
                reinterpret_cast<out_t*>(out.data_ptr()), batch_size,
                active_heads, num_heads, main_topk, extra_topk, candidate_slots,
                kv_workspace.stride(0), kv_workspace.stride(1),
                kv_workspace.stride(2), out.stride(0), out.stride(2),
                out.stride(3), length_tensor_kind(topk_length),
                length_tensor_kind(extra_topk_length));
    }
    DG_CUDA_RUNTIME_CHECK(cudaGetLastError());
}

template <typename q_t, typename kv_t, typename out_t, typename index_t>
void launch_sparse_mla_prefill_from_workspace_split(
    const torch::Tensor& q, const torch::Tensor& kv,
    const torch::Tensor& indices, const torch::Tensor& topk_length,
    const torch::Tensor& attn_sink, const torch::Tensor& out,
    const torch::Tensor& lse, double softmax_scale) {
    const int num_tokens = static_cast<int>(q.size(0));
    const int num_heads = static_cast<int>(q.size(1));
    const int active_heads = sm120_active_heads(num_heads);
    const int kv_tokens = static_cast<int>(kv.size(0));
    const int topk = static_cast<int>(indices.size(2));
    DG_HOST_ASSERT(num_tokens > 0);
    DG_HOST_ASSERT(num_heads > 0);
    DG_HOST_ASSERT(active_heads > 0);
    DG_HOST_ASSERT(kv_tokens > 0);
    DG_HOST_ASSERT(topk > 0);
    DG_HOST_ASSERT(topk <= 4096);

    const auto stream = at::cuda::getCurrentCUDAStream();
    const int topk_kind = length_tensor_kind(topk_length);

    // FLASH_FUSION_PREFILL: bf16 mma.sync m16n8k16 QK + scalar-PV online flash-
    // attention fused into a single kernel. Replaces the 3-kernel chain
    // (score+softmax+output) for the workspace prefill path. K is gathered via
    // indices[token, k_idx] in a TILE_K=32 outer loop. Grid (num_tokens,
    // ceil(active_heads/16)) ≈ 8k blocks at num_tokens=4096 — abundant
    // parallelism, no split-K needed. Default OFF — gated on
    // DG_SM120_FLASH_FUSION_PREFILL=1.
    if (sm120_flash_fusion_prefill_enabled()) {
        const bool pre_scalar_check = sm120_flash_fusion_prefill_scalar_check();
        const bool pv_scalar = sm120_flash_fusion_prefill_pv_scalar();
        bool pipeline = sm120_flash_fusion_prefill_cpasync();
        bool reg_o = sm120_flash_fusion_prefill_reg_o();
        // kRegO requires kPvMma=true (scalar-PV fallback still uses smem O).
        // If both are requested, refuse kRegO with a one-time warning so the
        // flag is benign instead of compile-failing the kernel.
        if (reg_o && pv_scalar) {
            static std::atomic<bool> warned{false};
            if (!warned.exchange(true)) {
                std::fprintf(
                    stderr,
                    "[sm120-flash-fusion-prefill] REG_O=1 ignored: requires "
                    "PV-MMA path (PV_SCALAR=1 keeps smem O_accum).\n");
            }
            reg_o = false;
        }
        // Smem layout sizing. O_accum (32 KB at M=16, headdim=512, fp32) drops
        // when kRegO=true. Compute both single-buffer + pipelined sizes so the
        // cp.async cap fallback can decide whether the 99 KB SM_120 cap fits.
        const size_t q_bytes =
            static_cast<size_t>(kFlashMmaM) * kHeadDim *
            sizeof(__nv_bfloat16);
        const size_t k_tile_bytes =
            static_cast<size_t>(kFlashMmaTileK) * kFlashMmaKStride *
            sizeof(__nv_bfloat16);
        const size_t o_accum_bytes = reg_o ? 0 : (
            static_cast<size_t>(kFlashMmaM) * kHeadDim * sizeof(float));
        const size_t scores_bytes =
            static_cast<size_t>(kFlashMmaM) * kFlashMmaTileK * sizeof(float);
        const size_t scratch_bytes =
            static_cast<size_t>(kFlashMmaM) * 5 * sizeof(float);
        // cp.async double-buffer doubles K_tile smem (~33 KB extra). On SM_120
        // (Pro 6000) the per-CTA dynamic smem cap is ~101 KB. With kRegO=true
        // the pipelined config drops from ~118 KB to ~85 KB and fits; with
        // kRegO=false it overflows and we fall back to single-buffer.
        if (pipeline) {
            int max_smem_optin = 0;
            cudaDeviceGetAttribute(&max_smem_optin,
                                   cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
            const size_t pipelined_bytes =
                q_bytes + k_tile_bytes * 2 + o_accum_bytes +
                scores_bytes + scratch_bytes;
            if (max_smem_optin > 0 &&
                pipelined_bytes > static_cast<size_t>(max_smem_optin)) {
                static std::atomic<bool> warned{false};
                if (!warned.exchange(true)) {
                    std::fprintf(
                        stderr,
                        "[sm120-flash-fusion-prefill] CPASYNC requested but "
                        "pipelined smem (%zu B) exceeds device cap (%d B); "
                        "falling back to single-buffer. Try REG_O=1 to free "
                        "32 KB.\n",
                        pipelined_bytes, max_smem_optin);
                }
                pipeline = false;
            }
        }
        const size_t flash_pre_smem_bytes =
            q_bytes +
            k_tile_bytes * (pipeline ? 2 : 1) +
            o_accum_bytes +
            scores_bytes +
            scratch_bytes;
        const dim3 pre_grid(
            num_tokens, (active_heads + kFlashMmaM - 1) / kFlashMmaM);
        auto launch_pre = [&](auto check_tag, auto pv_tag, auto pipe_tag,
                              auto reg_tag) {
            constexpr bool kCheck = decltype(check_tag)::value;
            constexpr bool kPvMma = decltype(pv_tag)::value;
            constexpr bool kPipeline = decltype(pipe_tag)::value;
            constexpr bool kRegO = decltype(reg_tag)::value;
            auto* kernel_ptr =
                sparse_mla_prefill_workspace_flash_mma_kernel<q_t, kv_t, out_t,
                                                              index_t, kCheck,
                                                              kPvMma, kPipeline,
                                                              kRegO>;
            DG_CUDA_RUNTIME_CHECK(cudaFuncSetAttribute(
                kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(flash_pre_smem_bytes)));
            kernel_ptr<<<pre_grid, kFlashMmaThreads, flash_pre_smem_bytes,
                         stream>>>(
                reinterpret_cast<const q_t*>(q.data_ptr()),
                reinterpret_cast<const kv_t*>(kv.data_ptr()),
                reinterpret_cast<const index_t*>(indices.data_ptr()),
                topk_length.defined() ? topk_length.data_ptr() : nullptr,
                attn_sink.defined() ? attn_sink.data_ptr<float>() : nullptr,
                reinterpret_cast<out_t*>(out.data_ptr()),
                lse.data_ptr<float>(), num_tokens, active_heads, num_heads,
                kv_tokens, topk, q.stride(0), q.stride(1), q.stride(2),
                kv.stride(0), kv.stride(2), indices.stride(0),
                indices.stride(2), out.stride(0), out.stride(1), out.stride(2),
                topk_kind, static_cast<float>(softmax_scale));
        };
        auto launch_pre_with_reg = [&](auto check_tag, auto pv_tag,
                                       auto pipe_tag) {
            // kRegO only meaningful with kPvMma=true (scalar-PV fallback
            // still uses smem O). Skip the kRegO=true instantiation when
            // pv_tag is false so the kernel template's static_assert in the
            // scalar-PV branch never triggers.
            if constexpr (decltype(pv_tag)::value) {
                if (reg_o) {
                    launch_pre(check_tag, pv_tag, pipe_tag, std::true_type{});
                } else {
                    launch_pre(check_tag, pv_tag, pipe_tag, std::false_type{});
                }
            } else {
                launch_pre(check_tag, pv_tag, pipe_tag, std::false_type{});
            }
        };
        auto launch_pre_with_pipe = [&](auto check_tag, auto pv_tag) {
            if (pipeline) {
                launch_pre_with_reg(check_tag, pv_tag, std::true_type{});
            } else {
                launch_pre_with_reg(check_tag, pv_tag, std::false_type{});
            }
        };
        auto launch_pre_with_pv = [&](auto check_tag) {
            if (pv_scalar) {
                launch_pre_with_pipe(check_tag, std::false_type{});
            } else {
                launch_pre_with_pipe(check_tag, std::true_type{});
            }
        };
        if (pre_scalar_check) {
            launch_pre_with_pv(std::true_type{});
        } else {
            launch_pre_with_pv(std::false_type{});
        }
        DG_CUDA_RUNTIME_CHECK(cudaGetLastError());
        return;
    }

    auto scores = torch::empty({num_tokens, active_heads, topk},
                               q.options().dtype(torch::kFloat32));
    const dim3 score_grid(
        (topk + kScoreCandidatesPerBlock - 1) / kScoreCandidatesPerBlock,
        active_heads, num_tokens);
    sparse_mla_prefill_indexed_score_tiled_kernel<q_t, kv_t, index_t>
        <<<score_grid, kThreads, 0, stream>>>(
            reinterpret_cast<const q_t*>(q.data_ptr()),
            reinterpret_cast<const kv_t*>(kv.data_ptr()),
            reinterpret_cast<const index_t*>(indices.data_ptr()),
            topk_length.defined() ? topk_length.data_ptr() : nullptr,
            scores.data_ptr<float>(), num_tokens, active_heads, num_heads,
            kv_tokens, topk, q.stride(0), q.stride(1), q.stride(2),
            kv.stride(0), kv.stride(2), indices.stride(0), indices.stride(2),
            topk_kind, static_cast<float>(softmax_scale));
    DG_CUDA_RUNTIME_CHECK(cudaGetLastError());

    sparse_mla_softmax_kernel<<<num_tokens * active_heads, kThreads, 0, stream>>>(
        scores.data_ptr<float>(), lse.data_ptr<float>(),
        attn_sink.defined() ? attn_sink.data_ptr<float>() : nullptr, num_tokens,
        num_heads, active_heads, topk);
    DG_CUDA_RUNTIME_CHECK(cudaGetLastError());

    const dim3 out_grid(num_tokens * active_heads,
                        (kHeadDim + kThreads - 1) / kThreads);
    sparse_mla_prefill_indexed_output_kernel<kv_t, out_t, index_t>
        <<<out_grid, kThreads, 0, stream>>>(
            reinterpret_cast<const kv_t*>(kv.data_ptr()),
            reinterpret_cast<const index_t*>(indices.data_ptr()),
            topk_length.defined() ? topk_length.data_ptr() : nullptr,
            scores.data_ptr<float>(), lse.data_ptr<float>(),
            attn_sink.defined() ? attn_sink.data_ptr<float>() : nullptr,
            reinterpret_cast<out_t*>(out.data_ptr()), num_tokens, active_heads,
            num_heads, kv_tokens, topk, kv.stride(0), kv.stride(2),
            indices.stride(0), indices.stride(2), out.stride(0),
            out.stride(1), out.stride(2), topk_kind);
    DG_CUDA_RUNTIME_CHECK(cudaGetLastError());
}

template <typename BlockT, typename SeqT, typename StartT>
void launch_build_prefill_workspace_map(const torch::Tensor& out,
                                        const torch::Tensor& block_table,
                                        const torch::Tensor& seq_lens,
                                        const torch::Tensor& workspace_starts,
                                        int block_size) {
    constexpr int kMapThreads = 256;
    const int64_t out_rows = out.numel();
    const int64_t init_blocks = (out_rows + kMapThreads - 1) / kMapThreads;
    const int64_t num_reqs = block_table.size(0);
    const int64_t grid_x = std::max<int64_t>(init_blocks, num_reqs);
    if (grid_x <= 0)
        return;
    DG_HOST_ASSERT(grid_x <= std::numeric_limits<unsigned>::max());
    const auto stream = at::cuda::getCurrentCUDAStream();
    const dim3 grid(static_cast<unsigned>(grid_x), 2);
    prefill_workspace_map_kernel<BlockT, SeqT, StartT>
        <<<grid, kMapThreads, 0, stream>>>(
            out.data_ptr<int32_t>(), block_table.data_ptr<BlockT>(),
            seq_lens.data_ptr<SeqT>(), workspace_starts.data_ptr<StartT>(),
            static_cast<int>(num_reqs), block_size, out_rows,
            block_table.stride(0), block_table.stride(1), seq_lens.stride(0),
            workspace_starts.stride(0));
    DG_CUDA_RUNTIME_CHECK(cudaGetLastError());
}

template <typename BlockT, typename SeqT, typename GatherT>
void launch_build_prefill_strided_workspace_map(
    const torch::Tensor& out, const torch::Tensor& block_table,
    const torch::Tensor& seq_lens, const torch::Tensor& gather_lens,
    int block_size, int64_t row_stride, int64_t offset, bool encode_negative) {
    constexpr int kMapThreads = 256;
    const int64_t num_reqs = block_table.size(0);
    if (num_reqs <= 0 || out.numel() <= 0)
        return;
    DG_HOST_ASSERT(num_reqs <= std::numeric_limits<unsigned>::max());
    const auto stream = at::cuda::getCurrentCUDAStream();
    prefill_strided_workspace_map_kernel<BlockT, SeqT, GatherT>
        <<<static_cast<unsigned>(num_reqs), kMapThreads, 0, stream>>>(
            out.data_ptr<int32_t>(), block_table.data_ptr<BlockT>(),
            seq_lens.data_ptr<SeqT>(),
            gather_lens.defined() ? gather_lens.data_ptr<GatherT>() : nullptr,
            static_cast<int>(num_reqs), block_size, out.numel(), row_stride,
            offset, encode_negative, block_table.stride(0), block_table.stride(1),
            seq_lens.stride(0), gather_lens.defined() ? gather_lens.stride(0) : 0);
    DG_CUDA_RUNTIME_CHECK(cudaGetLastError());
}

template <typename q_t, typename out_t, typename index_t>
void launch_sparse_mla_prefill_from_fp8_workspace_map(
    const torch::Tensor& q, const torch::Tensor& k_cache,
    const torch::Tensor& extra_k_cache,
    const torch::Tensor& workspace_map, const torch::Tensor& indices,
    const torch::Tensor& topk_length, const torch::Tensor& attn_sink,
    const torch::Tensor& out, const torch::Tensor& lse, int block_size,
    int extra_block_size, double softmax_scale) {
    const int num_tokens = static_cast<int>(q.size(0));
    const int num_heads = static_cast<int>(q.size(1));
    const int active_heads = sm120_active_heads(num_heads);
    const int topk = static_cast<int>(indices.size(2));
    DG_HOST_ASSERT(num_tokens > 0);
    DG_HOST_ASSERT(num_heads > 0);
    DG_HOST_ASSERT(active_heads > 0);
    DG_HOST_ASSERT(topk > 0);
    DG_HOST_ASSERT(topk <= 4096);

    const auto stream = at::cuda::getCurrentCUDAStream();
    const int topk_kind = length_tensor_kind(topk_length);
    if (topk <= kMaxGroupedCandidateSlots) {
        const size_t shared_bytes = grouped_decode_shared_bytes(topk);
        sparse_mla_prefill_fp8_map_grouped_kernel<q_t, out_t, index_t>
            <<<num_tokens * active_heads, kThreads, shared_bytes, stream>>>(
                reinterpret_cast<const q_t*>(q.data_ptr()),
                reinterpret_cast<const uint8_t*>(k_cache.data_ptr()),
                extra_k_cache.defined()
                    ? reinterpret_cast<const uint8_t*>(extra_k_cache.data_ptr())
                    : nullptr,
                workspace_map.data_ptr<int32_t>(),
                reinterpret_cast<const index_t*>(indices.data_ptr()),
                topk_length.defined() ? topk_length.data_ptr() : nullptr,
                attn_sink.defined() ? attn_sink.data_ptr<float>() : nullptr,
                reinterpret_cast<out_t*>(out.data_ptr()), lse.data_ptr<float>(),
                num_tokens, active_heads, num_heads,
                static_cast<int>(workspace_map.numel()), topk, block_size,
                extra_block_size, k_cache.size(0),
                extra_k_cache.defined() ? extra_k_cache.size(0) : 0,
                byte_stride(k_cache, 0),
                extra_k_cache.defined() ? byte_stride(extra_k_cache, 0) : 0,
                q.stride(0), q.stride(1), q.stride(2), indices.stride(0),
                indices.stride(2), out.stride(0), out.stride(1),
                out.stride(2), topk_kind, static_cast<float>(softmax_scale));
        DG_CUDA_RUNTIME_CHECK(cudaGetLastError());
        return;
    }

    auto scores = torch::empty({num_tokens, active_heads, topk},
                               q.options().dtype(torch::kFloat32));
    const dim3 score_grid(
        (topk + kScoreCandidatesPerBlock - 1) / kScoreCandidatesPerBlock,
        active_heads, num_tokens);
    sparse_mla_prefill_fp8_map_score_tiled_kernel<q_t, index_t>
        <<<score_grid, kThreads, 0, stream>>>(
            reinterpret_cast<const q_t*>(q.data_ptr()),
            reinterpret_cast<const uint8_t*>(k_cache.data_ptr()),
            extra_k_cache.defined()
                ? reinterpret_cast<const uint8_t*>(extra_k_cache.data_ptr())
                : nullptr,
            workspace_map.data_ptr<int32_t>(),
            reinterpret_cast<const index_t*>(indices.data_ptr()),
            topk_length.defined() ? topk_length.data_ptr() : nullptr,
            scores.data_ptr<float>(), num_tokens, active_heads, num_heads,
            static_cast<int>(workspace_map.numel()), topk, block_size,
            extra_block_size, q.stride(0), q.stride(1), q.stride(2), k_cache.size(0),
            extra_k_cache.defined() ? extra_k_cache.size(0) : 0,
            byte_stride(k_cache, 0),
            extra_k_cache.defined() ? byte_stride(extra_k_cache, 0) : 0,
            indices.stride(0), indices.stride(2),
            topk_kind, static_cast<float>(softmax_scale));
    DG_CUDA_RUNTIME_CHECK(cudaGetLastError());

    sparse_mla_softmax_kernel<<<num_tokens * active_heads, kThreads, 0, stream>>>(
        scores.data_ptr<float>(), lse.data_ptr<float>(),
        attn_sink.defined() ? attn_sink.data_ptr<float>() : nullptr, num_tokens,
        num_heads, active_heads, topk);
    DG_CUDA_RUNTIME_CHECK(cudaGetLastError());

    const dim3 out_grid(num_tokens * active_heads,
                        (kHeadDim + kThreads - 1) / kThreads);
    sparse_mla_prefill_fp8_map_output_kernel<out_t, index_t>
        <<<out_grid, kThreads, 0, stream>>>(
            reinterpret_cast<const uint8_t*>(k_cache.data_ptr()),
            extra_k_cache.defined()
                ? reinterpret_cast<const uint8_t*>(extra_k_cache.data_ptr())
                : nullptr,
            workspace_map.data_ptr<int32_t>(),
            reinterpret_cast<const index_t*>(indices.data_ptr()),
            topk_length.defined() ? topk_length.data_ptr() : nullptr,
            scores.data_ptr<float>(), lse.data_ptr<float>(),
            attn_sink.defined() ? attn_sink.data_ptr<float>() : nullptr,
            reinterpret_cast<out_t*>(out.data_ptr()), num_tokens, active_heads,
            num_heads, static_cast<int>(workspace_map.numel()), topk,
            block_size, extra_block_size, k_cache.size(0),
            extra_k_cache.defined() ? extra_k_cache.size(0) : 0,
            byte_stride(k_cache, 0),
            extra_k_cache.defined() ? byte_stride(extra_k_cache, 0) : 0,
            indices.stride(0), indices.stride(2), out.stride(0), out.stride(1),
            out.stride(2), topk_kind);
    DG_CUDA_RUNTIME_CHECK(cudaGetLastError());
}

template <typename q_t, typename out_t, typename index_t>
void launch_sparse_mla_decode_fast(
    const torch::Tensor& q, const torch::Tensor& k_cache,
    const torch::Tensor& indices, const torch::Tensor& topk_length,
    const torch::Tensor& attn_sink, const torch::Tensor& out,
    const torch::Tensor& lse, double softmax_scale) {
    const int batch_size = static_cast<int>(q.size(0));
    const int num_heads = static_cast<int>(q.size(2));
    const int active_heads = sm120_active_heads(num_heads);
    const int block_size = static_cast<int>(k_cache.size(1));
    const int topk = static_cast<int>(indices.size(2));
    DG_HOST_ASSERT(topk > 0);
    DG_HOST_ASSERT(topk <= 4096);

    auto scores = torch::empty({batch_size, num_heads, topk},
                               q.options().dtype(torch::kFloat32));
    const auto stream = at::cuda::getCurrentCUDAStream();
    const dim3 score_grid(
        (topk + kScoreCandidatesPerBlock - 1) / kScoreCandidatesPerBlock,
        active_heads, batch_size);
    sparse_mla_score_tiled_kernel<q_t, index_t><<<score_grid, kThreads, 0, stream>>>(
        reinterpret_cast<const q_t*>(q.data_ptr()),
        reinterpret_cast<const uint8_t*>(k_cache.data_ptr()), indices.data_ptr(),
        topk_length.defined() ? topk_length.data_ptr<int64_t>() : nullptr,
        scores.data_ptr<float>(), batch_size, num_heads, block_size,
        k_cache.size(0), topk, q.stride(0), q.stride(2), q.stride(3),
        indices.stride(0), indices.stride(2), byte_stride(k_cache, 0),
        static_cast<float>(softmax_scale));
    DG_CUDA_RUNTIME_CHECK(cudaGetLastError());

    sparse_mla_softmax_kernel<<<batch_size * active_heads, kThreads, 0, stream>>>(
        scores.data_ptr<float>(), lse.data_ptr<float>(),
        attn_sink.defined() ? attn_sink.data_ptr<float>() : nullptr, batch_size,
        num_heads, active_heads, topk);
    DG_CUDA_RUNTIME_CHECK(cudaGetLastError());

    const dim3 out_grid(batch_size * active_heads,
                        (kHeadDim + kThreads - 1) / kThreads);
    sparse_mla_output_kernel<out_t, index_t><<<out_grid, kThreads, 0, stream>>>(
        reinterpret_cast<const uint8_t*>(k_cache.data_ptr()), indices.data_ptr(),
        topk_length.defined() ? topk_length.data_ptr<int64_t>() : nullptr,
        scores.data_ptr<float>(), lse.data_ptr<float>(),
        attn_sink.defined() ? attn_sink.data_ptr<float>() : nullptr,
        reinterpret_cast<out_t*>(out.data_ptr()), batch_size, active_heads,
        num_heads, block_size, k_cache.size(0), topk, out.stride(0),
        out.stride(2), out.stride(3), indices.stride(0), indices.stride(2),
        byte_stride(k_cache, 0));
    DG_CUDA_RUNTIME_CHECK(cudaGetLastError());
}

template <typename q_t, typename out_t, typename index_t>
void launch_sparse_mla_decode_scaled(
    const torch::Tensor& q, const torch::Tensor& k_cache,
    const torch::Tensor& indices, const torch::Tensor& topk_length,
    const torch::Tensor& attn_sink, const torch::Tensor& out,
    const torch::Tensor& lse, double softmax_scale) {
    const int batch_size = static_cast<int>(q.size(0));
    const int num_heads = static_cast<int>(q.size(2));
    const int active_heads = sm120_active_heads(num_heads);
    const int block_size = static_cast<int>(k_cache.size(1));
    const int topk = static_cast<int>(indices.size(2));
    DG_HOST_ASSERT(topk > 0);
    DG_HOST_ASSERT(topk <= 4096);

    auto scores = torch::empty({batch_size, num_heads, topk},
                               q.options().dtype(torch::kFloat32));
    const auto stream = at::cuda::getCurrentCUDAStream();
    const dim3 score_grid(
        (topk + kScoreCandidatesPerBlock - 1) / kScoreCandidatesPerBlock,
        active_heads, batch_size);
    sparse_mla_score_tiled_scaled_kernel<q_t, index_t>
        <<<score_grid, kThreads, 0, stream>>>(
            reinterpret_cast<const q_t*>(q.data_ptr()),
            reinterpret_cast<const uint8_t*>(k_cache.data_ptr()),
            indices.data_ptr(),
            topk_length.defined() ? topk_length.data_ptr<int64_t>() : nullptr,
            scores.data_ptr<float>(), batch_size, num_heads, block_size,
            k_cache.size(0), topk, q.stride(0), q.stride(2), q.stride(3),
            indices.stride(0), indices.stride(2), byte_stride(k_cache, 0),
            static_cast<float>(softmax_scale));
    DG_CUDA_RUNTIME_CHECK(cudaGetLastError());

    sparse_mla_softmax_kernel<<<batch_size * active_heads, kThreads, 0, stream>>>(
        scores.data_ptr<float>(), lse.data_ptr<float>(),
        attn_sink.defined() ? attn_sink.data_ptr<float>() : nullptr, batch_size,
        num_heads, active_heads, topk);
    DG_CUDA_RUNTIME_CHECK(cudaGetLastError());

    const dim3 out_grid(batch_size * active_heads,
                        (kHeadDim + kQuantBlock - 1) / kQuantBlock);
    const size_t shared_bytes = static_cast<size_t>(topk) * sizeof(float);
    sparse_mla_output_scaled_kernel<out_t, index_t>
        <<<out_grid, kThreads, shared_bytes, stream>>>(
            reinterpret_cast<const uint8_t*>(k_cache.data_ptr()),
            indices.data_ptr(),
            topk_length.defined() ? topk_length.data_ptr<int64_t>() : nullptr,
            scores.data_ptr<float>(), lse.data_ptr<float>(),
            attn_sink.defined() ? attn_sink.data_ptr<float>() : nullptr,
            reinterpret_cast<out_t*>(out.data_ptr()), batch_size, active_heads,
            num_heads, block_size, k_cache.size(0), topk, out.stride(0),
            out.stride(2), out.stride(3), indices.stride(0),
            indices.stride(2), byte_stride(k_cache, 0));
    DG_CUDA_RUNTIME_CHECK(cudaGetLastError());
}

template <typename q_t, typename out_t, typename block_t, typename seq_t,
          typename req_t>
void launch_sparse_mla_decode_full_context(
    const torch::Tensor& q, const torch::Tensor& k_cache,
    const torch::Tensor& block_table, const torch::Tensor& seq_lens,
    const torch::Tensor& req_id_per_token, const torch::Tensor& attn_sink,
    const torch::Tensor& out, const torch::Tensor& lse,
    double softmax_scale) {
    const int batch_size = static_cast<int>(q.size(0));
    const int num_heads = static_cast<int>(q.size(2));
    const int block_size = static_cast<int>(k_cache.size(1));
    const int block_table_rows = static_cast<int>(block_table.size(0));
    const int block_table_width = static_cast<int>(block_table.size(1));
    const int candidate_slots = block_table_width * block_size;
    DG_HOST_ASSERT(batch_size > 0);
    DG_HOST_ASSERT(num_heads > 0);
    DG_HOST_ASSERT(block_size > 0);
    DG_HOST_ASSERT(block_table_rows > 0 && block_table_width > 0);
    DG_HOST_ASSERT(candidate_slots > 0);
    DG_HOST_ASSERT(candidate_slots <= kMaxGroupedCandidateSlots);

    const auto stream = at::cuda::getCurrentCUDAStream();
    const int grid = batch_size * num_heads;
    const size_t shared_bytes = full_context_decode_shared_bytes(candidate_slots);
    sparse_mla_decode_full_context_kernel<q_t, out_t, block_t, seq_t, req_t>
        <<<grid, kThreads, shared_bytes, stream>>>(
            reinterpret_cast<const q_t*>(q.data_ptr()),
            reinterpret_cast<const uint8_t*>(k_cache.data_ptr()),
            block_table.data_ptr<block_t>(), seq_lens.data_ptr<seq_t>(),
            req_id_per_token.data_ptr<req_t>(),
            attn_sink.defined() ? attn_sink.data_ptr<float>() : nullptr,
            reinterpret_cast<out_t*>(out.data_ptr()), lse.data_ptr<float>(),
            batch_size, num_heads, block_size, k_cache.size(0),
            block_table_rows, block_table_width, candidate_slots, q.stride(0),
            q.stride(1), q.stride(2), q.stride(3), out.stride(0),
            out.stride(1), out.stride(2), out.stride(3),
            block_table.stride(0), block_table.stride(1), seq_lens.stride(0),
            req_id_per_token.stride(0), byte_stride(k_cache, 0),
            static_cast<float>(softmax_scale));
    DG_CUDA_RUNTIME_CHECK(cudaGetLastError());
}

} // namespace

void dequantize_and_gather_k_cache(
    const torch::Tensor& out, const torch::Tensor& k_cache,
    const torch::Tensor& seq_lens, const pybind11::object& gather_lens_obj,
    const torch::Tensor& block_table, int block_size, int offset) {
    DG_HOST_ASSERT(out.is_cuda() && k_cache.is_cuda() && seq_lens.is_cuda() &&
                   block_table.is_cuda());
    DG_HOST_ASSERT(out.dim() == 3 && out.size(2) >= kHeadDim);
    DG_HOST_ASSERT(k_cache.dim() >= 2);
    DG_HOST_ASSERT(block_table.dim() >= 2);
    DG_HOST_ASSERT(block_size > 0);
    DG_HOST_ASSERT(offset >= 0 && offset <= out.size(1));
    DG_HOST_ASSERT(k_cache.size(k_cache.dim() - 1) >= kTokenDataBytes + kScaleBytes);

    auto gather_lens = tensor_or_empty(gather_lens_obj);
    if (gather_lens.defined())
        DG_HOST_ASSERT(gather_lens.is_cuda() && gather_lens.numel() >= seq_lens.numel());

    auto launch_for_block = [&](auto out_tag, auto seq_tag) {
        using out_t = decltype(out_tag);
        using seq_t = decltype(seq_tag);
        if (block_table.scalar_type() == torch::kInt64) {
            if (gather_lens.defined() && gather_lens.scalar_type() == torch::kInt) {
                launch_dequantize_gather_k_cache<out_t, seq_t, int64_t, int32_t>(
                    out, k_cache, seq_lens, gather_lens, block_table, block_size,
                    offset);
            } else {
                if (gather_lens.defined())
                    DG_HOST_ASSERT(gather_lens.scalar_type() == torch::kInt64);
                launch_dequantize_gather_k_cache<out_t, seq_t, int64_t, int64_t>(
                    out, k_cache, seq_lens, gather_lens, block_table, block_size,
                    offset);
            }
        } else {
            DG_HOST_ASSERT(block_table.scalar_type() == torch::kInt);
            if (gather_lens.defined() && gather_lens.scalar_type() == torch::kInt) {
                launch_dequantize_gather_k_cache<out_t, seq_t, int32_t, int32_t>(
                    out, k_cache, seq_lens, gather_lens, block_table, block_size,
                    offset);
            } else {
                if (gather_lens.defined())
                    DG_HOST_ASSERT(gather_lens.scalar_type() == torch::kInt64);
                launch_dequantize_gather_k_cache<out_t, seq_t, int32_t, int64_t>(
                    out, k_cache, seq_lens, gather_lens, block_table, block_size,
                    offset);
            }
        }
    };

    if (out.scalar_type() == torch::kBFloat16) {
        if (seq_lens.scalar_type() == torch::kInt64)
            launch_for_block(__nv_bfloat16{}, int64_t{});
        else {
            DG_HOST_ASSERT(seq_lens.scalar_type() == torch::kInt);
            launch_for_block(__nv_bfloat16{}, int32_t{});
        }
    } else if (out.scalar_type() == torch::kFloat16) {
        if (seq_lens.scalar_type() == torch::kInt64)
            launch_for_block(half{}, int64_t{});
        else {
            DG_HOST_ASSERT(seq_lens.scalar_type() == torch::kInt);
            launch_for_block(half{}, int32_t{});
        }
    } else if (out.scalar_type() == torch::kFloat32) {
        if (seq_lens.scalar_type() == torch::kInt64)
            launch_for_block(float{}, int64_t{});
        else {
            DG_HOST_ASSERT(seq_lens.scalar_type() == torch::kInt);
            launch_for_block(float{}, int32_t{});
        }
    } else {
        DG_HOST_UNREACHABLE("SM120 gather currently supports bf16, fp16, or fp32 output");
    }
}

void dequantize_and_gather_indexed_k_cache(
    const torch::Tensor& out, const torch::Tensor& k_cache,
    const torch::Tensor& indices, const pybind11::object& topk_length_obj,
    int block_size, int offset) {
    DG_HOST_ASSERT(out.is_cuda() && k_cache.is_cuda() && indices.is_cuda());
    DG_HOST_ASSERT(out.dim() == 3 && out.size(2) >= kHeadDim);
    DG_HOST_ASSERT(indices.dim() == 2 || indices.dim() == 3);
    DG_HOST_ASSERT(indices.size(0) <= out.size(0));
    DG_HOST_ASSERT(block_size > 0);
    DG_HOST_ASSERT(offset >= 0);
    DG_HOST_ASSERT(offset + indices.size(indices.dim() == 3 ? 2 : 1) <= out.size(1));
    DG_HOST_ASSERT(k_cache.dim() >= 4 && k_cache.size(2) == 1);
    DG_HOST_ASSERT(k_cache.size(k_cache.dim() - 1) >= kTokenDataBytes + kScaleBytes);

    auto topk_length = tensor_or_empty(topk_length_obj);
    if (topk_length.defined())
        DG_HOST_ASSERT(topk_length.is_cuda() && topk_length.numel() >= indices.size(0));

    auto launch_for_len = [&](auto out_tag, auto index_tag) {
        using out_t = decltype(out_tag);
        using index_t = decltype(index_tag);
        if (topk_length.defined() && topk_length.scalar_type() == torch::kInt) {
            launch_dequantize_gather_indexed_k_cache<out_t, index_t, int32_t>(
                out, k_cache, indices, topk_length, block_size, offset);
        } else {
            if (topk_length.defined())
                DG_HOST_ASSERT(topk_length.scalar_type() == torch::kInt64);
            launch_dequantize_gather_indexed_k_cache<out_t, index_t, int64_t>(
                out, k_cache, indices, topk_length, block_size, offset);
        }
    };

    auto launch_for_index = [&](auto out_tag) {
        if (indices.scalar_type() == torch::kInt64) {
            launch_for_len(out_tag, int64_t{});
        } else {
            DG_HOST_ASSERT(indices.scalar_type() == torch::kInt);
            launch_for_len(out_tag, int32_t{});
        }
    };

    if (out.scalar_type() == torch::kBFloat16) {
        launch_for_index(__nv_bfloat16{});
    } else if (out.scalar_type() == torch::kFloat16) {
        launch_for_index(half{});
    } else if (out.scalar_type() == torch::kFloat32) {
        launch_for_index(float{});
    } else {
        DG_HOST_UNREACHABLE("SM120 indexed gather currently supports bf16, fp16, or fp32 output");
    }
}

void fill_decode_all_indices(const torch::Tensor& topk_indices,
                             const torch::Tensor& seq_lens, int num_rows,
                             int next_n, int topk) {
    DG_HOST_ASSERT(topk_indices.is_cuda() && seq_lens.is_cuda());
    DG_HOST_ASSERT(topk_indices.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(seq_lens.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(topk_indices.dim() == 2);
    DG_HOST_ASSERT(seq_lens.dim() == 1 || seq_lens.dim() == 2);
    DG_HOST_ASSERT(seq_lens.is_contiguous());
    DG_HOST_ASSERT(num_rows >= 0 && next_n > 0 && topk >= 0);
    DG_HOST_ASSERT(topk_indices.size(0) >= num_rows);
    DG_HOST_ASSERT(topk_indices.size(1) >= topk);
    DG_HOST_ASSERT(seq_lens.dim() == 2
                       ? seq_lens.numel() >= num_rows
                       : seq_lens.size(0) * next_n >= num_rows);
    if (num_rows == 0 || topk == 0)
        return;

    const auto stream = at::cuda::getCurrentCUDAStream();
    constexpr int threads = 256;
    const dim3 grid((topk + threads - 1) / threads, num_rows);
    fill_decode_all_indices_kernel<<<grid, threads, 0, stream>>>(
        topk_indices.data_ptr<int32_t>(), seq_lens.data_ptr<int32_t>(),
        num_rows, next_n, topk, topk_indices.stride(0),
        topk_indices.stride(1), seq_lens.dim() == 2);
    DG_CUDA_RUNTIME_CHECK(cudaGetLastError());
}

std::tuple<torch::Tensor, torch::Tensor> sparse_mla_decode_from_bf16_workspace(
    const torch::Tensor& q, const torch::Tensor& kv_workspace,
    const pybind11::object& topk_length_obj,
    const pybind11::object& extra_topk_length_obj,
    const pybind11::object& attn_sink_obj, int main_topk, int extra_topk,
    int head_dim_v, double softmax_scale, const pybind11::object& out_obj) {
    DG_HOST_ASSERT(q.is_cuda() && kv_workspace.is_cuda());
    DG_HOST_ASSERT(q.dim() == 4 && q.size(1) == 1 && q.size(3) == kHeadDim);
    DG_HOST_ASSERT(kv_workspace.dim() == 3 && kv_workspace.size(2) >= kHeadDim);
    DG_HOST_ASSERT(head_dim_v == kHeadDim);
    DG_HOST_ASSERT(main_topk >= 0 && extra_topk >= 0);
    DG_HOST_ASSERT(main_topk + extra_topk > 0);
    DG_HOST_ASSERT(main_topk + extra_topk <= kv_workspace.size(1));

    auto topk_length = tensor_or_empty(topk_length_obj);
    auto extra_topk_length = tensor_or_empty(extra_topk_length_obj);
    auto attn_sink = tensor_or_empty(attn_sink_obj);
    if (topk_length.defined()) {
        DG_HOST_ASSERT(topk_length.is_cuda() && topk_length.numel() >= q.size(0));
        DG_HOST_ASSERT(topk_length.scalar_type() == torch::kInt ||
                       topk_length.scalar_type() == torch::kInt64);
    }
    if (extra_topk_length.defined()) {
        DG_HOST_ASSERT(extra_topk_length.is_cuda() &&
                       extra_topk_length.numel() >= q.size(0));
        DG_HOST_ASSERT(extra_topk_length.scalar_type() == torch::kInt ||
                       extra_topk_length.scalar_type() == torch::kInt64);
    }
    if (attn_sink.defined() && attn_sink.scalar_type() != torch::kFloat32)
        attn_sink = attn_sink.to(torch::kFloat32);

    torch::Tensor out;
    if (is_none(out_obj)) {
        out = torch::empty({q.size(0), q.size(1), q.size(2), head_dim_v},
                           q.options());
    } else {
        out = out_obj.cast<torch::Tensor>();
    }
    auto lse = torch::empty({q.size(0), q.size(2), q.size(1)},
                            q.options().dtype(torch::kFloat32));
    const auto stream = at::cuda::getCurrentCUDAStream();
    static sm120_profile::KernelProfileCounter profile_counter(
        "sm120_sparse_mla_workspace");
    sm120_profile::ScopedTimer profile_timer(
        profile_counter, stream, static_cast<int>(q.size(0)),
        static_cast<int>(q.size(2)), main_topk + extra_topk, 1);

    if (q.scalar_type() == torch::kBFloat16 &&
        kv_workspace.scalar_type() == torch::kBFloat16 &&
        out.scalar_type() == torch::kBFloat16) {
        launch_sparse_mla_decode_from_workspace<__nv_bfloat16, __nv_bfloat16,
                                                __nv_bfloat16>(
            q, kv_workspace, topk_length, extra_topk_length, attn_sink, out,
            lse, main_topk, extra_topk, softmax_scale);
    } else if (q.scalar_type() == torch::kFloat16 &&
               kv_workspace.scalar_type() == torch::kFloat16 &&
               out.scalar_type() == torch::kFloat16) {
        launch_sparse_mla_decode_from_workspace<half, half, half>(
            q, kv_workspace, topk_length, extra_topk_length, attn_sink, out,
            lse, main_topk, extra_topk, softmax_scale);
    } else {
        DG_HOST_UNREACHABLE(
            "SM120 workspace sparse MLA decode requires matching bf16 or fp16 tensors");
    }

    return std::make_tuple(out, lse);
}

std::tuple<torch::Tensor, torch::Tensor>
sparse_mla_decode_from_bf16_workspace_split(
    const torch::Tensor& q, const torch::Tensor& kv_workspace,
    const pybind11::object& topk_length_obj,
    const pybind11::object& extra_topk_length_obj,
    const pybind11::object& attn_sink_obj, int main_topk, int extra_topk,
    int head_dim_v, double softmax_scale, const pybind11::object& out_obj) {
    DG_HOST_ASSERT(q.is_cuda() && kv_workspace.is_cuda());
    DG_HOST_ASSERT(q.dim() == 4 && q.size(1) == 1 && q.size(3) == kHeadDim);
    DG_HOST_ASSERT(kv_workspace.dim() == 3 && kv_workspace.size(2) >= kHeadDim);
    DG_HOST_ASSERT(head_dim_v == kHeadDim);
    DG_HOST_ASSERT(main_topk >= 0 && extra_topk >= 0);
    DG_HOST_ASSERT(main_topk + extra_topk > 0);
    DG_HOST_ASSERT(main_topk + extra_topk <= kv_workspace.size(1));

    auto topk_length = tensor_or_empty(topk_length_obj);
    auto extra_topk_length = tensor_or_empty(extra_topk_length_obj);
    auto attn_sink = tensor_or_empty(attn_sink_obj);
    if (topk_length.defined()) {
        DG_HOST_ASSERT(topk_length.is_cuda() && topk_length.numel() >= q.size(0));
        DG_HOST_ASSERT(topk_length.scalar_type() == torch::kInt ||
                       topk_length.scalar_type() == torch::kInt64);
    }
    if (extra_topk_length.defined()) {
        DG_HOST_ASSERT(extra_topk_length.is_cuda() &&
                       extra_topk_length.numel() >= q.size(0));
        DG_HOST_ASSERT(extra_topk_length.scalar_type() == torch::kInt ||
                       extra_topk_length.scalar_type() == torch::kInt64);
    }
    if (attn_sink.defined() && attn_sink.scalar_type() != torch::kFloat32)
        attn_sink = attn_sink.to(torch::kFloat32);

    torch::Tensor out;
    if (is_none(out_obj)) {
        out = torch::empty({q.size(0), q.size(1), q.size(2), head_dim_v},
                           q.options());
    } else {
        out = out_obj.cast<torch::Tensor>();
    }
    auto lse = torch::empty({q.size(0), q.size(2), q.size(1)},
                            q.options().dtype(torch::kFloat32));
    const auto stream = at::cuda::getCurrentCUDAStream();
    static sm120_profile::KernelProfileCounter profile_counter(
        "sm120_sparse_mla_workspace_split");
    sm120_profile::ScopedTimer profile_timer(
        profile_counter, stream, static_cast<int>(q.size(0)),
        static_cast<int>(q.size(2)), main_topk + extra_topk, 1);

    if (q.scalar_type() == torch::kBFloat16 &&
        kv_workspace.scalar_type() == torch::kBFloat16 &&
        out.scalar_type() == torch::kBFloat16) {
        launch_sparse_mla_decode_from_workspace_split<__nv_bfloat16,
                                                      __nv_bfloat16,
                                                      __nv_bfloat16>(
            q, kv_workspace, topk_length, extra_topk_length, attn_sink, out,
            lse, main_topk, extra_topk, softmax_scale);
    } else if (q.scalar_type() == torch::kFloat16 &&
               kv_workspace.scalar_type() == torch::kFloat16 &&
               out.scalar_type() == torch::kFloat16) {
        launch_sparse_mla_decode_from_workspace_split<half, half, half>(
            q, kv_workspace, topk_length, extra_topk_length, attn_sink, out,
            lse, main_topk, extra_topk, softmax_scale);
    } else {
        DG_HOST_UNREACHABLE(
            "SM120 split workspace sparse MLA decode requires matching bf16 or fp16 tensors");
    }

    return std::make_tuple(out, lse);
}

std::tuple<torch::Tensor, torch::Tensor> sparse_mla_decode(
    const torch::Tensor& q, const torch::Tensor& k_cache,
    const torch::Tensor& indices, const pybind11::object& topk_length_obj,
    const pybind11::object& attn_sink_obj,
    const pybind11::object& extra_k_cache_obj,
    const pybind11::object& extra_indices_obj,
    const pybind11::object& extra_topk_length_obj, int head_dim_v,
    double softmax_scale, const pybind11::object& out_obj) {
    DG_HOST_ASSERT(q.is_cuda() && k_cache.is_cuda() && indices.is_cuda());
    DG_HOST_ASSERT(q.dim() == 4 && q.size(1) == 1 && q.size(3) == kHeadDim);
    DG_HOST_ASSERT(k_cache.dim() >= 4 && k_cache.size(2) == 1);
    DG_HOST_ASSERT(head_dim_v == kHeadDim);
    DG_HOST_ASSERT(k_cache.size(k_cache.dim() - 1) >= kTokenDataBytes + kScaleBytes);

    auto topk_length = tensor_or_empty(topk_length_obj);
    auto attn_sink = tensor_or_empty(attn_sink_obj);
    auto extra_k_cache = tensor_or_empty(extra_k_cache_obj);
    auto extra_indices = tensor_or_empty(extra_indices_obj);
    auto extra_topk_length = tensor_or_empty(extra_topk_length_obj);

    if (topk_length.defined() && topk_length.scalar_type() != torch::kInt64)
        topk_length = topk_length.to(torch::kInt64);
    if (extra_topk_length.defined() &&
        extra_topk_length.scalar_type() != torch::kInt64)
        extra_topk_length = extra_topk_length.to(torch::kInt64);
    if (attn_sink.defined() && attn_sink.scalar_type() != torch::kFloat32)
        attn_sink = attn_sink.to(torch::kFloat32);

    torch::Tensor out;
    if (is_none(out_obj)) {
        out = torch::empty({q.size(0), q.size(1), q.size(2), head_dim_v},
                           q.options());
    } else {
        out = out_obj.cast<torch::Tensor>();
    }
    auto lse = torch::empty({q.size(0), q.size(2), q.size(1)},
                            q.options().dtype(torch::kFloat32));
    const auto stream = at::cuda::getCurrentCUDAStream();
    static sm120_profile::KernelProfileCounter profile_counter(
        "sm120_sparse_mla_decode");
    sm120_profile::ScopedTimer profile_timer(
        profile_counter, stream, static_cast<int>(q.size(0)),
        static_cast<int>(q.size(2)), static_cast<int>(indices.size(-1)), 1);

    if (q.scalar_type() == torch::kBFloat16 && out.scalar_type() == torch::kBFloat16) {
        if (indices.scalar_type() == torch::kInt64) {
            if (!extra_k_cache.defined() && !extra_indices.defined()) {
                launch_sparse_mla_decode_fast<__nv_bfloat16, __nv_bfloat16, int64_t>(
                    q, k_cache, indices, topk_length, attn_sink, out, lse,
                    softmax_scale);
            } else {
                launch_sparse_mla_decode<__nv_bfloat16, __nv_bfloat16, int64_t>(
                    q, k_cache, indices, topk_length, attn_sink, extra_k_cache,
                    extra_indices, extra_topk_length, out, lse, softmax_scale);
            }
        } else if (indices.scalar_type() == torch::kInt32) {
            if (!extra_k_cache.defined() && !extra_indices.defined()) {
                launch_sparse_mla_decode_fast<__nv_bfloat16, __nv_bfloat16, int32_t>(
                    q, k_cache, indices, topk_length, attn_sink, out, lse,
                    softmax_scale);
            } else {
                launch_sparse_mla_decode<__nv_bfloat16, __nv_bfloat16, int32_t>(
                    q, k_cache, indices, topk_length, attn_sink, extra_k_cache,
                    extra_indices, extra_topk_length, out, lse, softmax_scale);
            }
        } else {
            DG_HOST_UNREACHABLE("SM120 sparse MLA decode indices must be int32 or int64");
        }
    } else if (q.scalar_type() == torch::kFloat16 && out.scalar_type() == torch::kFloat16) {
        if (indices.scalar_type() == torch::kInt64) {
            if (!extra_k_cache.defined() && !extra_indices.defined()) {
                launch_sparse_mla_decode_fast<half, half, int64_t>(
                    q, k_cache, indices, topk_length, attn_sink, out, lse,
                    softmax_scale);
            } else {
                launch_sparse_mla_decode<half, half, int64_t>(
                    q, k_cache, indices, topk_length, attn_sink, extra_k_cache,
                    extra_indices, extra_topk_length, out, lse, softmax_scale);
            }
        } else if (indices.scalar_type() == torch::kInt32) {
            if (!extra_k_cache.defined() && !extra_indices.defined()) {
                launch_sparse_mla_decode_fast<half, half, int32_t>(
                    q, k_cache, indices, topk_length, attn_sink, out, lse,
                    softmax_scale);
            } else {
                launch_sparse_mla_decode<half, half, int32_t>(
                    q, k_cache, indices, topk_length, attn_sink, extra_k_cache,
                    extra_indices, extra_topk_length, out, lse, softmax_scale);
            }
        } else {
            DG_HOST_UNREACHABLE("SM120 sparse MLA decode indices must be int32 or int64");
        }
    } else {
        DG_HOST_UNREACHABLE("SM120 sparse MLA decode currently supports bf16 or fp16 q/out");
    }

    return std::make_tuple(out, lse);
}

std::tuple<torch::Tensor, torch::Tensor> sparse_mla_decode_fused(
    const torch::Tensor& q, const torch::Tensor& k_cache,
    const torch::Tensor& indices, const pybind11::object& topk_length_obj,
    const pybind11::object& attn_sink_obj, int head_dim_v,
    double softmax_scale, const pybind11::object& out_obj) {
    DG_HOST_ASSERT(q.is_cuda() && k_cache.is_cuda() && indices.is_cuda());
    DG_HOST_ASSERT(q.dim() == 4 && q.size(1) == 1 && q.size(3) == kHeadDim);
    DG_HOST_ASSERT(k_cache.dim() >= 4 && k_cache.size(2) == 1);
    DG_HOST_ASSERT(indices.dim() == 3);
    DG_HOST_ASSERT(head_dim_v == kHeadDim);
    DG_HOST_ASSERT(k_cache.size(k_cache.dim() - 1) >=
                   kTokenDataBytes + kScaleBytes);
    DG_HOST_ASSERT(indices.size(2) <= kMaxGroupedCandidateSlots);

    auto topk_length = tensor_or_empty(topk_length_obj);
    auto attn_sink = tensor_or_empty(attn_sink_obj);
    if (topk_length.defined() && topk_length.scalar_type() != torch::kInt64)
        topk_length = topk_length.to(torch::kInt64);
    if (attn_sink.defined() && attn_sink.scalar_type() != torch::kFloat32)
        attn_sink = attn_sink.to(torch::kFloat32);

    torch::Tensor out;
    if (is_none(out_obj)) {
        out = torch::empty({q.size(0), q.size(1), q.size(2), head_dim_v},
                           q.options());
    } else {
        out = out_obj.cast<torch::Tensor>();
    }
    auto lse = torch::empty({q.size(0), q.size(2), q.size(1)},
                            q.options().dtype(torch::kFloat32));
    const auto stream = at::cuda::getCurrentCUDAStream();
    static sm120_profile::KernelProfileCounter profile_counter(
        "sm120_sparse_mla_decode_fused");
    sm120_profile::ScopedTimer profile_timer(
        profile_counter, stream, static_cast<int>(q.size(0)),
        static_cast<int>(q.size(2)), static_cast<int>(indices.size(2)), 1);

    if (q.scalar_type() == torch::kBFloat16 &&
        out.scalar_type() == torch::kBFloat16) {
        if (indices.scalar_type() == torch::kInt64) {
            launch_sparse_mla_decode_scaled<__nv_bfloat16, __nv_bfloat16,
                                            int64_t>(
                q, k_cache, indices, topk_length, attn_sink, out, lse,
                softmax_scale);
        } else if (indices.scalar_type() == torch::kInt32) {
            launch_sparse_mla_decode_scaled<__nv_bfloat16, __nv_bfloat16,
                                            int32_t>(
                q, k_cache, indices, topk_length, attn_sink, out, lse,
                softmax_scale);
        } else {
            DG_HOST_UNREACHABLE(
                "SM120 fused sparse MLA decode indices must be int32 or int64");
        }
    } else if (q.scalar_type() == torch::kFloat16 &&
               out.scalar_type() == torch::kFloat16) {
        if (indices.scalar_type() == torch::kInt64) {
            launch_sparse_mla_decode_scaled<half, half, int64_t>(
                q, k_cache, indices, topk_length, attn_sink, out, lse,
                softmax_scale);
        } else if (indices.scalar_type() == torch::kInt32) {
            launch_sparse_mla_decode_scaled<half, half, int32_t>(
                q, k_cache, indices, topk_length, attn_sink, out, lse,
                softmax_scale);
        } else {
            DG_HOST_UNREACHABLE(
                "SM120 fused sparse MLA decode indices must be int32 or int64");
        }
    } else {
        DG_HOST_UNREACHABLE(
            "SM120 fused sparse MLA decode currently supports bf16 or fp16 q/out");
    }

    return std::make_tuple(out, lse);
}

std::tuple<torch::Tensor, torch::Tensor> sparse_mla_decode_full_context(
    const torch::Tensor& q, const torch::Tensor& k_cache,
    const torch::Tensor& block_table, const torch::Tensor& seq_lens,
    const torch::Tensor& req_id_per_token,
    const pybind11::object& attn_sink_obj, int head_dim_v,
    double softmax_scale, const pybind11::object& out_obj) {
    DG_HOST_ASSERT(q.is_cuda() && k_cache.is_cuda() && block_table.is_cuda() &&
                   seq_lens.is_cuda() && req_id_per_token.is_cuda());
    DG_HOST_ASSERT(q.dim() == 4 && q.size(1) == 1 && q.size(3) == kHeadDim);
    DG_HOST_ASSERT(k_cache.dim() >= 4 && k_cache.size(2) == 1);
    DG_HOST_ASSERT(block_table.dim() == 2);
    DG_HOST_ASSERT(seq_lens.dim() == 1);
    DG_HOST_ASSERT(req_id_per_token.dim() == 1);
    DG_HOST_ASSERT(req_id_per_token.numel() >= q.size(0));
    DG_HOST_ASSERT(head_dim_v == kHeadDim);
    DG_HOST_ASSERT(k_cache.size(k_cache.dim() - 1) >= kTokenDataBytes + kScaleBytes);

    auto attn_sink = tensor_or_empty(attn_sink_obj);
    if (attn_sink.defined() && attn_sink.scalar_type() != torch::kFloat32)
        attn_sink = attn_sink.to(torch::kFloat32);

    torch::Tensor out;
    if (is_none(out_obj)) {
        out = torch::empty({q.size(0), q.size(1), q.size(2), head_dim_v},
                           q.options());
    } else {
        out = out_obj.cast<torch::Tensor>();
    }
    auto lse = torch::empty({q.size(0), q.size(2), q.size(1)},
                            q.options().dtype(torch::kFloat32));
    const auto stream = at::cuda::getCurrentCUDAStream();
    static sm120_profile::KernelProfileCounter profile_counter(
        "sm120_sparse_mla_full_context");
    sm120_profile::ScopedTimer profile_timer(
        profile_counter, stream, static_cast<int>(q.size(0)),
        static_cast<int>(q.size(2)), static_cast<int>(seq_lens.numel()), 1);

    auto launch_for_req = [&](auto q_tag, auto out_tag, auto block_tag,
                              auto seq_tag, auto req_tag) {
        using q_t = decltype(q_tag);
        using out_t = decltype(out_tag);
        using block_t = decltype(block_tag);
        using seq_t = decltype(seq_tag);
        using req_t = decltype(req_tag);
        launch_sparse_mla_decode_full_context<q_t, out_t, block_t, seq_t,
                                              req_t>(
            q, k_cache, block_table, seq_lens, req_id_per_token, attn_sink, out,
            lse, softmax_scale);
    };

    auto dispatch_req = [&](auto q_tag, auto out_tag, auto block_tag,
                            auto seq_tag) {
        if (req_id_per_token.scalar_type() == torch::kInt64) {
            launch_for_req(q_tag, out_tag, block_tag, seq_tag, int64_t{});
        } else {
            DG_HOST_ASSERT(req_id_per_token.scalar_type() == torch::kInt);
            launch_for_req(q_tag, out_tag, block_tag, seq_tag, int32_t{});
        }
    };

    auto dispatch_seq = [&](auto q_tag, auto out_tag, auto block_tag) {
        if (seq_lens.scalar_type() == torch::kInt64) {
            dispatch_req(q_tag, out_tag, block_tag, int64_t{});
        } else {
            DG_HOST_ASSERT(seq_lens.scalar_type() == torch::kInt);
            dispatch_req(q_tag, out_tag, block_tag, int32_t{});
        }
    };

    auto dispatch_block = [&](auto q_tag, auto out_tag) {
        if (block_table.scalar_type() == torch::kInt64) {
            dispatch_seq(q_tag, out_tag, int64_t{});
        } else {
            DG_HOST_ASSERT(block_table.scalar_type() == torch::kInt);
            dispatch_seq(q_tag, out_tag, int32_t{});
        }
    };

    if (q.scalar_type() == torch::kBFloat16 &&
        out.scalar_type() == torch::kBFloat16) {
        dispatch_block(__nv_bfloat16{}, __nv_bfloat16{});
    } else if (q.scalar_type() == torch::kFloat16 &&
               out.scalar_type() == torch::kFloat16) {
        dispatch_block(half{}, half{});
    } else {
        DG_HOST_UNREACHABLE(
            "SM120 full-context sparse MLA decode currently supports bf16 or fp16 q/out");
    }

    return std::make_tuple(out, lse);
}

template <typename q_t, typename kv_t, typename out_t, typename index_t>
void launch_sparse_mla_prefill_from_workspace(
    const torch::Tensor& q, const torch::Tensor& kv,
    const torch::Tensor& indices, const torch::Tensor& topk_length,
    const torch::Tensor& attn_sink, const torch::Tensor& out,
    const torch::Tensor& max_logits, const torch::Tensor& lse,
    double softmax_scale) {
    const int num_tokens = static_cast<int>(q.size(0));
    const int num_heads = static_cast<int>(q.size(1));
    const int active_heads = sm120_active_heads(num_heads);
    const int kv_tokens = static_cast<int>(kv.size(0));
    const int topk = static_cast<int>(indices.size(2));
    const int topk_kind = length_tensor_kind(topk_length);
    const void* topk_ptr =
        topk_length.defined() ? topk_length.data_ptr() : nullptr;
    const float* sink_ptr =
        attn_sink.defined() ? attn_sink.data_ptr<float>() : nullptr;
    const auto stream = at::cuda::getCurrentCUDAStream();
    const dim3 grid(num_tokens, active_heads);
    const size_t shared_bytes =
        (static_cast<size_t>(topk) + kThreads) * sizeof(float);

    DG_CUDA_RUNTIME_CHECK(cudaFuncSetAttribute(
        sparse_mla_prefill_workspace_kernel<q_t, kv_t, out_t, index_t>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shared_bytes)));
    sparse_mla_prefill_workspace_kernel<q_t, kv_t, out_t, index_t>
        <<<grid, kThreads, shared_bytes, stream>>>(
            reinterpret_cast<const q_t*>(q.data_ptr()),
            reinterpret_cast<const kv_t*>(kv.data_ptr()),
            reinterpret_cast<const index_t*>(indices.data_ptr()), topk_ptr,
            topk_kind, sink_ptr, reinterpret_cast<out_t*>(out.data_ptr()),
            max_logits.data_ptr<float>(), lse.data_ptr<float>(), num_heads,
            kv_tokens, topk, q.stride(0), q.stride(1), q.stride(2),
            kv.stride(0), kv.stride(2), indices.stride(0), indices.stride(2),
            out.stride(0), out.stride(1), out.stride(2), softmax_scale);
    DG_CUDA_RUNTIME_CHECK(cudaGetLastError());
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
sparse_mla_prefill_from_bf16_workspace(
    const torch::Tensor& q, const torch::Tensor& kv,
    const torch::Tensor& indices, const pybind11::object& topk_length_obj,
    const pybind11::object& attn_sink_obj, int head_dim_v, double softmax_scale,
    const pybind11::object& out_obj) {
    DG_HOST_ASSERT(q.is_cuda() && kv.is_cuda() && indices.is_cuda());
    DG_HOST_ASSERT(q.dim() == 3 && q.size(2) == kHeadDim);
    DG_HOST_ASSERT(kv.dim() == 3 && kv.size(1) == 1 && kv.size(2) >= kHeadDim);
    DG_HOST_ASSERT(indices.dim() == 3 && indices.size(0) == q.size(0) &&
                   indices.size(1) == 1);
    DG_HOST_ASSERT(head_dim_v == kHeadDim);

    auto topk_length = tensor_or_empty(topk_length_obj);
    auto attn_sink = tensor_or_empty(attn_sink_obj);
    if (topk_length.defined()) {
        DG_HOST_ASSERT(topk_length.is_cuda() && topk_length.numel() >= q.size(0));
        DG_HOST_ASSERT(topk_length.scalar_type() == torch::kInt ||
                       topk_length.scalar_type() == torch::kInt64);
    }
    if (attn_sink.defined() && attn_sink.scalar_type() != torch::kFloat32)
        attn_sink = attn_sink.to(torch::kFloat32);

    torch::Tensor out;
    if (is_none(out_obj)) {
        out = torch::empty({q.size(0), q.size(1), head_dim_v}, q.options());
    } else {
        out = out_obj.cast<torch::Tensor>();
    }
    auto max_logits =
        torch::empty({q.size(0), q.size(1)}, q.options().dtype(torch::kFloat32));
    auto lse =
        torch::empty({q.size(0), q.size(1)}, q.options().dtype(torch::kFloat32));

    const auto stream = at::cuda::getCurrentCUDAStream();
    static sm120_profile::KernelProfileCounter profile_counter(
        "sm120_sparse_mla_prefill_workspace");
    sm120_profile::ScopedTimer profile_timer(
        profile_counter, stream, static_cast<int>(q.size(0)),
        static_cast<int>(q.size(1)), static_cast<int>(indices.size(2)), 1);

    if (q.scalar_type() == torch::kBFloat16 &&
        kv.scalar_type() == torch::kBFloat16 &&
        out.scalar_type() == torch::kBFloat16) {
        if (indices.scalar_type() == torch::kInt64) {
            launch_sparse_mla_prefill_from_workspace<__nv_bfloat16,
                                                     __nv_bfloat16,
                                                     __nv_bfloat16, int64_t>(
                q, kv, indices, topk_length, attn_sink, out, max_logits, lse,
                softmax_scale);
        } else if (indices.scalar_type() == torch::kInt32) {
            launch_sparse_mla_prefill_from_workspace<__nv_bfloat16,
                                                     __nv_bfloat16,
                                                     __nv_bfloat16, int32_t>(
                q, kv, indices, topk_length, attn_sink, out, max_logits, lse,
                softmax_scale);
        } else {
            DG_HOST_UNREACHABLE(
                "SM120 sparse MLA prefill indices must be int32 or int64");
        }
    } else if (q.scalar_type() == torch::kFloat16 &&
               kv.scalar_type() == torch::kFloat16 &&
               out.scalar_type() == torch::kFloat16) {
        if (indices.scalar_type() == torch::kInt64) {
            launch_sparse_mla_prefill_from_workspace<half, half, half, int64_t>(
                q, kv, indices, topk_length, attn_sink, out, max_logits, lse,
                softmax_scale);
        } else if (indices.scalar_type() == torch::kInt32) {
            launch_sparse_mla_prefill_from_workspace<half, half, half, int32_t>(
                q, kv, indices, topk_length, attn_sink, out, max_logits, lse,
                softmax_scale);
        } else {
            DG_HOST_UNREACHABLE(
                "SM120 sparse MLA prefill indices must be int32 or int64");
        }
    } else {
        DG_HOST_UNREACHABLE(
            "SM120 sparse MLA prefill currently supports bf16 or fp16 q/kv/out");
    }

    return std::make_tuple(out, max_logits, lse);
}

template <typename kv_t, typename index_t>
void launch_gather_bf16_workspace(const torch::Tensor& kv,
                                  const torch::Tensor& indices,
                                  const torch::Tensor& out) {
    const bool kv_3d = kv.dim() == 3;
    const bool indices_3d = indices.dim() == 3;
    const int num_tokens = static_cast<int>(indices.size(0));
    const int topk = static_cast<int>(indices.size(indices_3d ? 2 : 1));
    const int head_dim = static_cast<int>(out.size(2));
    DG_HOST_ASSERT(num_tokens > 0);
    DG_HOST_ASSERT(topk > 0);
    DG_HOST_ASSERT(head_dim > 0);
    DG_HOST_ASSERT(kv.size(0) > 0);
    DG_HOST_ASSERT(out.size(0) >= num_tokens);
    DG_HOST_ASSERT(out.size(1) >= topk);
    DG_HOST_ASSERT(kv.size(kv_3d ? 2 : 1) >= head_dim);

    const auto stream = at::cuda::getCurrentCUDAStream();
    const dim3 block(kThreads);
    const dim3 grid((head_dim + kThreads - 1) / kThreads, topk, num_tokens);
    gather_bf16_workspace_kernel<kv_t, index_t>
        <<<grid, block, 0, stream>>>(
            reinterpret_cast<const kv_t*>(kv.data_ptr()),
            reinterpret_cast<const index_t*>(indices.data_ptr()),
            reinterpret_cast<kv_t*>(out.data_ptr()), num_tokens, topk,
            head_dim, kv.size(0), kv.stride(0),
            kv.stride(kv_3d ? 2 : 1), indices.stride(0),
            indices_3d ? indices.stride(1) : 0,
            indices.stride(indices_3d ? 2 : 1), out.stride(0),
            out.stride(1), out.stride(2), indices_3d);
    DG_CUDA_RUNTIME_CHECK(cudaGetLastError());
}

torch::Tensor gather_bf16_workspace(const torch::Tensor& kv,
                                    const torch::Tensor& indices,
                                    const torch::Tensor& out) {
    DG_HOST_ASSERT(kv.is_cuda());
    DG_HOST_ASSERT(indices.is_cuda());
    DG_HOST_ASSERT(out.is_cuda());
    DG_HOST_ASSERT(kv.is_contiguous());
    DG_HOST_ASSERT(indices.is_contiguous());
    DG_HOST_ASSERT(out.is_contiguous());
    DG_HOST_ASSERT(kv.dim() == 2 || (kv.dim() == 3 && kv.size(1) == 1));
    DG_HOST_ASSERT(indices.dim() == 2 ||
                   (indices.dim() == 3 && indices.size(1) == 1));
    DG_HOST_ASSERT(out.dim() == 3);
    DG_HOST_ASSERT(kv.scalar_type() == out.scalar_type());

    if (kv.scalar_type() == torch::kBFloat16) {
        if (indices.scalar_type() == torch::kInt64) {
            launch_gather_bf16_workspace<__nv_bfloat16, int64_t>(
                kv, indices, out);
        } else if (indices.scalar_type() == torch::kInt32) {
            launch_gather_bf16_workspace<__nv_bfloat16, int32_t>(
                kv, indices, out);
        } else {
            DG_HOST_UNREACHABLE(
                "SM120 BF16 workspace gather indices must be int32 or int64");
        }
    } else if (kv.scalar_type() == torch::kFloat16) {
        if (indices.scalar_type() == torch::kInt64) {
            launch_gather_bf16_workspace<half, int64_t>(kv, indices, out);
        } else if (indices.scalar_type() == torch::kInt32) {
            launch_gather_bf16_workspace<half, int32_t>(kv, indices, out);
        } else {
            DG_HOST_UNREACHABLE(
                "SM120 FP16 workspace gather indices must be int32 or int64");
        }
    } else {
        DG_HOST_UNREACHABLE(
            "SM120 workspace gather currently supports bf16 or fp16 KV");
    }
    return out;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
sparse_mla_prefill_from_bf16_workspace_split(
    const torch::Tensor& q, const torch::Tensor& kv,
    const torch::Tensor& indices, const pybind11::object& topk_length_obj,
    const pybind11::object& attn_sink_obj, int head_dim_v, double softmax_scale,
    const pybind11::object& out_obj) {
    DG_HOST_ASSERT(q.is_cuda() && kv.is_cuda() && indices.is_cuda());
    DG_HOST_ASSERT(q.dim() == 3 && q.size(2) == kHeadDim);
    DG_HOST_ASSERT(kv.dim() == 3 && kv.size(1) == 1 && kv.size(2) >= kHeadDim);
    DG_HOST_ASSERT(indices.dim() == 3 && indices.size(0) == q.size(0) &&
                   indices.size(1) == 1);
    DG_HOST_ASSERT(head_dim_v == kHeadDim);

    auto topk_length = tensor_or_empty(topk_length_obj);
    auto attn_sink = tensor_or_empty(attn_sink_obj);
    if (topk_length.defined()) {
        DG_HOST_ASSERT(topk_length.is_cuda() && topk_length.numel() >= q.size(0));
        DG_HOST_ASSERT(topk_length.scalar_type() == torch::kInt ||
                       topk_length.scalar_type() == torch::kInt64);
    }
    if (attn_sink.defined() && attn_sink.scalar_type() != torch::kFloat32)
        attn_sink = attn_sink.to(torch::kFloat32);

    torch::Tensor out;
    if (is_none(out_obj)) {
        out = torch::empty({q.size(0), q.size(1), head_dim_v}, q.options());
    } else {
        out = out_obj.cast<torch::Tensor>();
    }
    auto max_logits =
        torch::empty({q.size(0), q.size(1)}, q.options().dtype(torch::kFloat32));
    auto lse =
        torch::empty({q.size(0), q.size(1)}, q.options().dtype(torch::kFloat32));

    const auto stream = at::cuda::getCurrentCUDAStream();
    static sm120_profile::KernelProfileCounter profile_counter(
        "sm120_sparse_mla_prefill_workspace_split");
    sm120_profile::ScopedTimer profile_timer(
        profile_counter, stream, static_cast<int>(q.size(0)),
        static_cast<int>(q.size(1)), static_cast<int>(indices.size(2)), 1);

    if (q.scalar_type() == torch::kBFloat16 &&
        kv.scalar_type() == torch::kBFloat16 &&
        out.scalar_type() == torch::kBFloat16) {
        if (indices.scalar_type() == torch::kInt64) {
            launch_sparse_mla_prefill_from_workspace_split<
                __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, int64_t>(
                q, kv, indices, topk_length, attn_sink, out, lse,
                softmax_scale);
        } else if (indices.scalar_type() == torch::kInt32) {
            launch_sparse_mla_prefill_from_workspace_split<
                __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, int32_t>(
                q, kv, indices, topk_length, attn_sink, out, lse,
                softmax_scale);
        } else {
            DG_HOST_UNREACHABLE(
                "SM120 split sparse MLA prefill indices must be int32 or int64");
        }
    } else if (q.scalar_type() == torch::kFloat16 &&
               kv.scalar_type() == torch::kFloat16 &&
               out.scalar_type() == torch::kFloat16) {
        if (indices.scalar_type() == torch::kInt64) {
            launch_sparse_mla_prefill_from_workspace_split<half, half, half,
                                                           int64_t>(
                q, kv, indices, topk_length, attn_sink, out, lse,
                softmax_scale);
        } else if (indices.scalar_type() == torch::kInt32) {
            launch_sparse_mla_prefill_from_workspace_split<half, half, half,
                                                           int32_t>(
                q, kv, indices, topk_length, attn_sink, out, lse,
                softmax_scale);
        } else {
            DG_HOST_UNREACHABLE(
                "SM120 split sparse MLA prefill indices must be int32 or int64");
        }
    } else {
        DG_HOST_UNREACHABLE(
            "SM120 split sparse MLA prefill currently supports bf16 or fp16 q/kv/out");
    }

    max_logits.fill_(std::numeric_limits<float>::quiet_NaN());
    return std::make_tuple(out, max_logits, lse);
}

void build_prefill_workspace_map(const torch::Tensor& out,
                                 const torch::Tensor& block_table,
                                 const torch::Tensor& seq_lens,
                                 const torch::Tensor& workspace_starts,
                                 int block_size) {
    DG_HOST_ASSERT(out.is_cuda() && block_table.is_cuda() && seq_lens.is_cuda() &&
                   workspace_starts.is_cuda());
    DG_HOST_ASSERT(out.scalar_type() == torch::kInt32);
    DG_HOST_ASSERT(out.dim() == 1);
    DG_HOST_ASSERT(block_table.dim() == 2);
    DG_HOST_ASSERT(seq_lens.dim() == 1);
    DG_HOST_ASSERT(workspace_starts.dim() == 1);
    DG_HOST_ASSERT(block_table.size(0) <= seq_lens.numel());
    DG_HOST_ASSERT(block_table.size(0) <= workspace_starts.numel());
    DG_HOST_ASSERT(block_size > 0);
    DG_HOST_ASSERT(block_table.scalar_type() == torch::kInt32 ||
                   block_table.scalar_type() == torch::kInt64);
    DG_HOST_ASSERT(seq_lens.scalar_type() == torch::kInt32 ||
                   seq_lens.scalar_type() == torch::kInt64);
    DG_HOST_ASSERT(workspace_starts.scalar_type() == torch::kInt32 ||
                   workspace_starts.scalar_type() == torch::kInt64);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(out));

#define DISPATCH_START(BLOCK_T, SEQ_T)                                             \
    do {                                                                           \
        if (workspace_starts.scalar_type() == torch::kInt32) {                     \
            launch_build_prefill_workspace_map<BLOCK_T, SEQ_T, int32_t>(           \
                out, block_table, seq_lens, workspace_starts, block_size);         \
        } else {                                                                   \
            launch_build_prefill_workspace_map<BLOCK_T, SEQ_T, int64_t>(           \
                out, block_table, seq_lens, workspace_starts, block_size);         \
        }                                                                          \
    } while (0)

#define DISPATCH_SEQ(BLOCK_T)                                                      \
    do {                                                                           \
        if (seq_lens.scalar_type() == torch::kInt32) {                             \
            DISPATCH_START(BLOCK_T, int32_t);                                      \
        } else {                                                                   \
            DISPATCH_START(BLOCK_T, int64_t);                                      \
        }                                                                          \
    } while (0)

    if (block_table.scalar_type() == torch::kInt32) {
        DISPATCH_SEQ(int32_t);
    } else {
        DISPATCH_SEQ(int64_t);
    }

#undef DISPATCH_SEQ
#undef DISPATCH_START
}

void build_prefill_strided_workspace_map(
    const torch::Tensor& out, const torch::Tensor& block_table,
    const torch::Tensor& seq_lens, const pybind11::object& gather_lens_obj,
    int block_size, int64_t row_stride, int64_t offset, bool encode_negative) {
    DG_HOST_ASSERT(out.is_cuda() && block_table.is_cuda() && seq_lens.is_cuda());
    DG_HOST_ASSERT(out.scalar_type() == torch::kInt32);
    DG_HOST_ASSERT(out.dim() == 1);
    DG_HOST_ASSERT(block_table.dim() == 2);
    DG_HOST_ASSERT(seq_lens.dim() == 1);
    DG_HOST_ASSERT(block_table.size(0) <= seq_lens.numel());
    DG_HOST_ASSERT(block_size > 0);
    DG_HOST_ASSERT(row_stride > 0);
    DG_HOST_ASSERT(offset >= 0 && offset < row_stride);
    DG_HOST_ASSERT(block_table.scalar_type() == torch::kInt32 ||
                   block_table.scalar_type() == torch::kInt64);
    DG_HOST_ASSERT(seq_lens.scalar_type() == torch::kInt32 ||
                   seq_lens.scalar_type() == torch::kInt64);
    auto gather_lens = tensor_or_empty(gather_lens_obj);
    if (gather_lens.defined()) {
        DG_HOST_ASSERT(gather_lens.is_cuda());
        DG_HOST_ASSERT(gather_lens.dim() == 1);
        DG_HOST_ASSERT(block_table.size(0) <= gather_lens.numel());
        DG_HOST_ASSERT(gather_lens.scalar_type() == torch::kInt32 ||
                       gather_lens.scalar_type() == torch::kInt64);
    }

    const at::cuda::OptionalCUDAGuard device_guard(device_of(out));

#define DISPATCH_GATHER(BLOCK_T, SEQ_T)                                            \
    do {                                                                           \
        if (!gather_lens.defined() || gather_lens.scalar_type() == torch::kInt64) { \
            launch_build_prefill_strided_workspace_map<BLOCK_T, SEQ_T, int64_t>(   \
                out, block_table, seq_lens, gather_lens, block_size, row_stride,   \
                offset, encode_negative);                                          \
        } else {                                                                   \
            launch_build_prefill_strided_workspace_map<BLOCK_T, SEQ_T, int32_t>(   \
                out, block_table, seq_lens, gather_lens, block_size, row_stride,   \
                offset, encode_negative);                                          \
        }                                                                          \
    } while (0)

#define DISPATCH_SEQ(BLOCK_T)                                                      \
    do {                                                                           \
        if (seq_lens.scalar_type() == torch::kInt32) {                             \
            DISPATCH_GATHER(BLOCK_T, int32_t);                                     \
        } else {                                                                   \
            DISPATCH_GATHER(BLOCK_T, int64_t);                                     \
        }                                                                          \
    } while (0)

    if (block_table.scalar_type() == torch::kInt32) {
        DISPATCH_SEQ(int32_t);
    } else {
        DISPATCH_SEQ(int64_t);
    }

#undef DISPATCH_SEQ
#undef DISPATCH_GATHER
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
sparse_mla_prefill_from_fp8_workspace_map(
    const torch::Tensor& q, const torch::Tensor& k_cache,
    const torch::Tensor& workspace_map, const torch::Tensor& indices,
    const pybind11::object& topk_length_obj,
    const pybind11::object& attn_sink_obj, int block_size, int head_dim_v,
    double softmax_scale, const pybind11::object& out_obj) {
    DG_HOST_ASSERT(q.is_cuda() && k_cache.is_cuda() && workspace_map.is_cuda() &&
                   indices.is_cuda());
    DG_HOST_ASSERT(q.dim() == 3 && q.size(2) == kHeadDim);
    DG_HOST_ASSERT(k_cache.dim() >= 4 && k_cache.size(2) == 1);
    DG_HOST_ASSERT(k_cache.size(k_cache.dim() - 1) >= kTokenDataBytes + kScaleBytes);
    DG_HOST_ASSERT(workspace_map.dim() == 1 &&
                   workspace_map.scalar_type() == torch::kInt32);
    DG_HOST_ASSERT(indices.dim() == 3 && indices.size(0) == q.size(0) &&
                   indices.size(1) == 1);
    DG_HOST_ASSERT(head_dim_v == kHeadDim);
    DG_HOST_ASSERT(block_size > 0);

    auto topk_length = tensor_or_empty(topk_length_obj);
    auto attn_sink = tensor_or_empty(attn_sink_obj);
    if (topk_length.defined()) {
        DG_HOST_ASSERT(topk_length.is_cuda() && topk_length.numel() >= q.size(0));
        DG_HOST_ASSERT(topk_length.scalar_type() == torch::kInt ||
                       topk_length.scalar_type() == torch::kInt64);
    }
    if (attn_sink.defined() && attn_sink.scalar_type() != torch::kFloat32)
        attn_sink = attn_sink.to(torch::kFloat32);

    torch::Tensor out;
    if (is_none(out_obj)) {
        out = torch::empty({q.size(0), q.size(1), head_dim_v}, q.options());
    } else {
        out = out_obj.cast<torch::Tensor>();
    }
    auto max_logits =
        torch::empty({q.size(0), q.size(1)}, q.options().dtype(torch::kFloat32));
    auto lse =
        torch::empty({q.size(0), q.size(1)}, q.options().dtype(torch::kFloat32));

    const auto stream = at::cuda::getCurrentCUDAStream();
    static sm120_profile::KernelProfileCounter profile_counter(
        "sm120_sparse_mla_prefill_fp8_map");
    sm120_profile::ScopedTimer profile_timer(
        profile_counter, stream, static_cast<int>(q.size(0)),
        static_cast<int>(q.size(1)), static_cast<int>(indices.size(2)), 1);

    if (q.scalar_type() == torch::kBFloat16 &&
        out.scalar_type() == torch::kBFloat16) {
        if (indices.scalar_type() == torch::kInt64) {
            launch_sparse_mla_prefill_from_fp8_workspace_map<
                __nv_bfloat16, __nv_bfloat16, int64_t>(
                q, k_cache, torch::Tensor(), workspace_map, indices, topk_length, attn_sink, out,
                lse, block_size, block_size, softmax_scale);
        } else if (indices.scalar_type() == torch::kInt32) {
            launch_sparse_mla_prefill_from_fp8_workspace_map<
                __nv_bfloat16, __nv_bfloat16, int32_t>(
                q, k_cache, torch::Tensor(), workspace_map, indices, topk_length, attn_sink, out,
                lse, block_size, block_size, softmax_scale);
        } else {
            DG_HOST_UNREACHABLE(
                "SM120 direct FP8 sparse MLA prefill indices must be int32 or int64");
        }
    } else if (q.scalar_type() == torch::kFloat16 &&
               out.scalar_type() == torch::kFloat16) {
        if (indices.scalar_type() == torch::kInt64) {
            launch_sparse_mla_prefill_from_fp8_workspace_map<half, half, int64_t>(
                q, k_cache, torch::Tensor(), workspace_map, indices, topk_length, attn_sink, out,
                lse, block_size, block_size, softmax_scale);
        } else if (indices.scalar_type() == torch::kInt32) {
            launch_sparse_mla_prefill_from_fp8_workspace_map<half, half, int32_t>(
                q, k_cache, torch::Tensor(), workspace_map, indices, topk_length, attn_sink, out,
                lse, block_size, block_size, softmax_scale);
        } else {
            DG_HOST_UNREACHABLE(
                "SM120 direct FP8 sparse MLA prefill indices must be int32 or int64");
        }
    } else {
        DG_HOST_UNREACHABLE(
            "SM120 direct FP8 sparse MLA prefill currently supports bf16 or fp16 q/out");
    }

    max_logits.fill_(std::numeric_limits<float>::quiet_NaN());
    return std::make_tuple(out, max_logits, lse);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
sparse_mla_prefill_from_two_fp8_workspace_map(
    const torch::Tensor& q, const torch::Tensor& primary_k_cache,
    const torch::Tensor& extra_k_cache, const torch::Tensor& workspace_map,
    const torch::Tensor& indices, const pybind11::object& topk_length_obj,
    const pybind11::object& attn_sink_obj, int primary_block_size, int extra_block_size, int head_dim_v,
    double softmax_scale, const pybind11::object& out_obj) {
    DG_HOST_ASSERT(q.is_cuda() && primary_k_cache.is_cuda() &&
                   extra_k_cache.is_cuda() && workspace_map.is_cuda() &&
                   indices.is_cuda());
    DG_HOST_ASSERT(q.dim() == 3 && q.size(2) == kHeadDim);
    DG_HOST_ASSERT(primary_k_cache.dim() >= 4 && primary_k_cache.size(2) == 1);
    DG_HOST_ASSERT(extra_k_cache.dim() >= 4 && extra_k_cache.size(2) == 1);
    DG_HOST_ASSERT(primary_k_cache.size(primary_k_cache.dim() - 1) >=
                   kTokenDataBytes + kScaleBytes);
    DG_HOST_ASSERT(extra_k_cache.size(extra_k_cache.dim() - 1) >=
                   kTokenDataBytes + kScaleBytes);
    DG_HOST_ASSERT(workspace_map.dim() == 1 &&
                   workspace_map.scalar_type() == torch::kInt32);
    DG_HOST_ASSERT(indices.dim() == 3 && indices.size(0) == q.size(0) &&
                   indices.size(1) == 1);
    DG_HOST_ASSERT(head_dim_v == kHeadDim);
    DG_HOST_ASSERT(primary_block_size > 0);
    DG_HOST_ASSERT(extra_block_size > 0);

    auto topk_length = tensor_or_empty(topk_length_obj);
    auto attn_sink = tensor_or_empty(attn_sink_obj);
    if (topk_length.defined()) {
        DG_HOST_ASSERT(topk_length.is_cuda() && topk_length.numel() >= q.size(0));
        DG_HOST_ASSERT(topk_length.scalar_type() == torch::kInt ||
                       topk_length.scalar_type() == torch::kInt64);
    }
    if (attn_sink.defined() && attn_sink.scalar_type() != torch::kFloat32)
        attn_sink = attn_sink.to(torch::kFloat32);

    torch::Tensor out;
    if (is_none(out_obj)) {
        out = torch::empty({q.size(0), q.size(1), head_dim_v}, q.options());
    } else {
        out = out_obj.cast<torch::Tensor>();
    }
    auto max_logits =
        torch::empty({q.size(0), q.size(1)}, q.options().dtype(torch::kFloat32));
    auto lse =
        torch::empty({q.size(0), q.size(1)}, q.options().dtype(torch::kFloat32));

    const auto stream = at::cuda::getCurrentCUDAStream();
    static sm120_profile::KernelProfileCounter profile_counter(
        "sm120_sparse_mla_prefill_two_fp8_map");
    sm120_profile::ScopedTimer profile_timer(
        profile_counter, stream, static_cast<int>(q.size(0)),
        static_cast<int>(q.size(1)), static_cast<int>(indices.size(2)), 1);

    if (q.scalar_type() == torch::kBFloat16 &&
        out.scalar_type() == torch::kBFloat16) {
        if (indices.scalar_type() == torch::kInt64) {
            launch_sparse_mla_prefill_from_fp8_workspace_map<
                __nv_bfloat16, __nv_bfloat16, int64_t>(
                q, primary_k_cache, extra_k_cache, workspace_map, indices,
                topk_length, attn_sink, out, lse, primary_block_size, extra_block_size, softmax_scale);
        } else if (indices.scalar_type() == torch::kInt32) {
            launch_sparse_mla_prefill_from_fp8_workspace_map<
                __nv_bfloat16, __nv_bfloat16, int32_t>(
                q, primary_k_cache, extra_k_cache, workspace_map, indices,
                topk_length, attn_sink, out, lse, primary_block_size, extra_block_size, softmax_scale);
        } else {
            DG_HOST_UNREACHABLE(
                "SM120 direct two-cache FP8 prefill indices must be int32 or int64");
        }
    } else if (q.scalar_type() == torch::kFloat16 &&
               out.scalar_type() == torch::kFloat16) {
        if (indices.scalar_type() == torch::kInt64) {
            launch_sparse_mla_prefill_from_fp8_workspace_map<half, half, int64_t>(
                q, primary_k_cache, extra_k_cache, workspace_map, indices,
                topk_length, attn_sink, out, lse, primary_block_size, extra_block_size, softmax_scale);
        } else if (indices.scalar_type() == torch::kInt32) {
            launch_sparse_mla_prefill_from_fp8_workspace_map<half, half, int32_t>(
                q, primary_k_cache, extra_k_cache, workspace_map, indices,
                topk_length, attn_sink, out, lse, primary_block_size, extra_block_size, softmax_scale);
        } else {
            DG_HOST_UNREACHABLE(
                "SM120 direct two-cache FP8 prefill indices must be int32 or int64");
        }
    } else {
        DG_HOST_UNREACHABLE(
            "SM120 direct two-cache FP8 sparse MLA prefill supports bf16/fp16 q/out");
    }

    max_logits.fill_(std::numeric_limits<float>::quiet_NaN());
    return std::make_tuple(out, max_logits, lse);
}

void register_apis(pybind11::module& m) {
    m.def("sm120_sparse_mla_decode", &sparse_mla_decode,
          "SM120 sparse MLA decode for DeepSeek-V4 Flash");
    m.def("sm120_sparse_mla_decode_fused", &sparse_mla_decode_fused,
          "SM120 fused indexed sparse MLA decode for DeepSeek-V4 Flash");
    m.def("sm120_sparse_mla_decode_full_context",
          &sparse_mla_decode_full_context,
          "SM120 full-context sparse MLA decode for DeepSeek-V4 Flash");
    m.def("sm120_dequantize_and_gather_k_cache", &dequantize_and_gather_k_cache,
          "SM120 DeepSeek-V4 fp8_ds_mla KV cache gather");
    m.def("sm120_dequantize_and_gather_indexed_k_cache",
          &dequantize_and_gather_indexed_k_cache,
          "SM120 DeepSeek-V4 fp8_ds_mla indexed KV cache gather");
    m.def("sm120_sparse_mla_decode_from_bf16_workspace",
          &sparse_mla_decode_from_bf16_workspace,
          "SM120 sparse MLA decode from gathered BF16/FP16 KV workspace");
    m.def("sm120_sparse_mla_decode_from_bf16_workspace_split",
          &sparse_mla_decode_from_bf16_workspace_split,
          "SM120 split sparse MLA decode from gathered BF16/FP16 KV workspace");
    m.def("sm120_gather_bf16_workspace",
          &gather_bf16_workspace,
          "SM120 gather BF16/FP16 sparse MLA KV rows into a reusable workspace");
    m.def("sm120_sparse_mla_prefill_from_bf16_workspace",
          &sparse_mla_prefill_from_bf16_workspace,
          "SM120 sparse MLA prefill from gathered BF16/FP16 KV workspace");
    m.def("sm120_sparse_mla_prefill_from_bf16_workspace_split",
          &sparse_mla_prefill_from_bf16_workspace_split,
          "SM120 split sparse MLA prefill directly from indexed BF16/FP16 KV");
    m.def("sm120_build_prefill_workspace_map",
          &build_prefill_workspace_map,
          "SM120 map FP8 prefill chunk workspace rows to physical KV slots");
    m.def("sm120_build_prefill_strided_workspace_map",
          &build_prefill_strided_workspace_map,
          "SM120 map strided DeepSeek-V4 prefill workspace rows to physical KV slots");
    m.def("sm120_sparse_mla_prefill_from_fp8_workspace_map",
          &sparse_mla_prefill_from_fp8_workspace_map,
          "SM120 sparse MLA prefill directly from FP8 KV cache plus workspace map");
    m.def("sm120_sparse_mla_prefill_from_two_fp8_workspace_map",
          &sparse_mla_prefill_from_two_fp8_workspace_map,
          "SM120 sparse MLA prefill directly from compressed+SWA FP8 KV caches");
    m.def("sm120_fill_decode_all_indices", &fill_decode_all_indices,
          "SM120 fill full-context sparse decode indices");
}

} // namespace sm120_mla
} // namespace deep_gemm
