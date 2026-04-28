#include <algorithm>
#include <cstdlib>

#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <torch/python.h>

#include "jit_kernels/impls/sm120_hc_prenorm_fallback.hpp"
#include "utils/exception.hpp"

namespace deep_gemm {

__host__ __forceinline__ bool sm120_hc_prenorm_mma_enabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("DG_SM120_HC_PRENORM_MMA");
        if (env == nullptr || env[0] == '\0') return false;
        return env[0] != '0';
    }();
    return enabled;
}

// Debug knob: when set, replaces the m16n8k8 tf32 MMA with a hand-rolled
// scalar matmul that uses the SAME a_smem / b_smem reads. Lets us bisect
// "is the bug in the MMA fragment layout, or in the smem load?".
__host__ __forceinline__ bool sm120_hc_prenorm_mma_scalar_check() {
    static const bool enabled = []() {
        const char* env = std::getenv("DG_SM120_HC_PRENORM_MMA_SCALAR_CHECK");
        if (env == nullptr || env[0] == '\0') return false;
        return env[0] != '0';
    }();
    return enabled;
}

// 2xTF32 emulation: split each f32 B value into a hi-tf32 + lo-tf32 residual
// pair, run two mma.sync calls per K-step, sum into the same f32 accumulator.
// Recovers ~20 bits of effective mantissa precision (vs 10 from single tf32),
// which is enough to keep argmax stable on tight-margin softmax outputs.
__host__ __forceinline__ bool sm120_hc_prenorm_mma_2xtf32() {
    static const bool enabled = []() {
        const char* env = std::getenv("DG_SM120_HC_PRENORM_MMA_2XTF32");
        if (env == nullptr || env[0] == '\0') return false;
        return env[0] != '0';
    }();
    return enabled;
}

namespace sm120_fallback {

__device__ __forceinline__ int split_k_begin(int k, int num_splits,
                                             int split_idx) {
    constexpr int block_k = 64;
    const int k_blocks = (k + block_k - 1) / block_k;
    const int blocks_per_split = k_blocks / num_splits;
    const int remain_blocks = k_blocks - blocks_per_split * num_splits;
    return (split_idx * blocks_per_split + min(split_idx, remain_blocks)) *
           block_k;
}

__device__ __forceinline__ int split_k_end(int k, int num_splits,
                                           int split_idx) {
    return min(k, split_k_begin(k, num_splits, split_idx + 1));
}

__device__ __forceinline__ float round_to_tf32(float value) {
    uint32_t bits = __float_as_uint(value);
    const uint32_t abs_bits = bits & 0x7fffffffu;
    if (abs_bits >= 0x7f800000u)
        return value;

    // Round FP32 mantissa to TF32's 10 explicit mantissa bits.
    const uint32_t lsb = (bits >> 13) & 1u;
    bits += 0x0fffu + lsb;
    bits &= 0xffffe000u;
    return __uint_as_float(bits);
}

__global__ void hc_prenorm_gemm_kernel(const __nv_bfloat16* a,
                                       const float* b, float* d,
                                       int64_t a_stride_m,
                                       int64_t a_stride_k,
                                       int64_t b_stride_n,
                                       int64_t b_stride_k,
                                       int64_t d_stride_split,
                                       int64_t d_stride_m,
                                       int64_t d_stride_n,
                                       int m, int n, int k,
                                       int num_splits) {
    const int64_t total =
        static_cast<int64_t>(num_splits) * static_cast<int64_t>(m) * n;
    for (int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
         linear < total;
         linear += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int col = static_cast<int>(linear % n);
        const int64_t row_linear = linear / n;
        const int row = static_cast<int>(row_linear % m);
        const int split_idx = static_cast<int>(row_linear / m);
        const int k_begin = split_k_begin(k, num_splits, split_idx);
        const int k_end = split_k_end(k, num_splits, split_idx);

        float sum = 0.0f;
        for (int kk = k_begin; kk < k_end; ++kk) {
            const float av =
                __bfloat162float(a[static_cast<int64_t>(row) * a_stride_m +
                                   static_cast<int64_t>(kk) * a_stride_k]);
            const float bv =
                round_to_tf32(b[static_cast<int64_t>(col) * b_stride_n +
                                static_cast<int64_t>(kk) * b_stride_k]);
            sum += av * bv;
        }

        d[static_cast<int64_t>(split_idx) * d_stride_split +
          static_cast<int64_t>(row) * d_stride_m +
          static_cast<int64_t>(col) * d_stride_n] = sum;
    }
}

__global__ void hc_prenorm_sqr_sum_kernel(const __nv_bfloat16* a,
                                          float* sqr_sum,
                                          int64_t a_stride_m,
                                          int64_t a_stride_k,
                                          int64_t s_stride_split,
                                          int64_t s_stride_m,
                                          int m, int k,
                                          int num_splits) {
    const int64_t total =
        static_cast<int64_t>(num_splits) * static_cast<int64_t>(m);
    for (int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
         linear < total;
         linear += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int row = static_cast<int>(linear % m);
        const int split_idx = static_cast<int>(linear / m);
        const int k_begin = split_k_begin(k, num_splits, split_idx);
        const int k_end = split_k_end(k, num_splits, split_idx);

        float sum = 0.0f;
        for (int kk = k_begin; kk < k_end; ++kk) {
            const float av =
                __bfloat162float(a[static_cast<int64_t>(row) * a_stride_m +
                                   static_cast<int64_t>(kk) * a_stride_k]);
            sum += av * av;
        }

        sqr_sum[static_cast<int64_t>(split_idx) * s_stride_split +
                static_cast<int64_t>(row) * s_stride_m] = sum;
    }
}

__global__ void hc_prenorm_block_reduce_kernel(
    const __nv_bfloat16* __restrict__ a, const float* __restrict__ b,
    float* __restrict__ d, float* __restrict__ sqr_sum,
    int64_t a_stride_m, int64_t a_stride_k, int64_t b_stride_n,
    int64_t b_stride_k, int64_t d_stride_split, int64_t d_stride_m,
    int64_t d_stride_n, int64_t s_stride_split, int64_t s_stride_m, int m,
    int n, int k, int num_splits) {
    extern __shared__ float smem[];

    const int linear = blockIdx.x;
    const int row = linear % m;
    const int split_idx = linear / m;
    if (split_idx >= num_splits)
        return;

    const int k_begin = split_k_begin(k, num_splits, split_idx);
    const int k_end = split_k_end(k, num_splits, split_idx);
    float partial[32];
#pragma unroll
    for (int col = 0; col < 32; ++col)
        partial[col] = 0.0f;
    float sq_partial = 0.0f;

    for (int kk = k_begin + threadIdx.x; kk < k_end; kk += blockDim.x) {
        const float av =
            __bfloat162float(a[static_cast<int64_t>(row) * a_stride_m +
                               static_cast<int64_t>(kk) * a_stride_k]);
        sq_partial += av * av;
        for (int col = 0; col < n; ++col) {
            const float bv =
                round_to_tf32(b[static_cast<int64_t>(col) * b_stride_n +
                                static_cast<int64_t>(kk) * b_stride_k]);
            partial[col] += av * bv;
        }
    }

    for (int col = 0; col < n; ++col)
        smem[col * blockDim.x + threadIdx.x] = partial[col];
    smem[n * blockDim.x + threadIdx.x] = sq_partial;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            for (int col = 0; col < n; ++col) {
                smem[col * blockDim.x + threadIdx.x] +=
                    smem[col * blockDim.x + threadIdx.x + offset];
            }
            smem[n * blockDim.x + threadIdx.x] +=
                smem[n * blockDim.x + threadIdx.x + offset];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        for (int col = 0; col < n; ++col) {
            d[static_cast<int64_t>(split_idx) * d_stride_split +
              static_cast<int64_t>(row) * d_stride_m +
              static_cast<int64_t>(col) * d_stride_n] =
                smem[col * blockDim.x];
        }
        sqr_sum[static_cast<int64_t>(split_idx) * s_stride_split +
                static_cast<int64_t>(row) * s_stride_m] =
            smem[n * blockDim.x];
    }
}

int hc_fallback_grid(int64_t total) {
    constexpr int threads = 256;
    const int64_t blocks = (total + threads - 1) / threads;
    return static_cast<int>(std::min<int64_t>(blocks, 4096));
}

// TF32 MMA variant of hc_prenorm_block_reduce_kernel.
// Same problem (D = A @ B^T with B pre-rounded to tf32, plus sqr_sum = sum(A^2))
// re-tiled to feed mma.sync.aligned.m16n8k8 row.col.f32.tf32.tf32.f32.
//
// A is bf16 [m,k] K-major; B is f32 [n,k] K-major; D is f32 [splits,m,n]; sqr_sum is f32 [splits,m].
// Block grid: (n_blocks_m * num_splits). Each block computes a 16-row tile of D
// for one split, plus that tile's contribution to sqr_sum.
//
// Numerics differ from the scalar kernel: tf32 MMA truncates the per-mul
// product mantissa to 10 bits (vs 24-bit f32 mul in the scalar path). This is a
// soft-divergence change. bf16 inputs convert to tf32 losslessly (bf16 mantissa
// is 7 bits, fits within tf32's 10).
constexpr int kMmaM = 16;
constexpr int kMmaN = 8;
constexpr int kMmaK = 8;
constexpr int kMmaThreads = 128;        // 4 warps
constexpr int kKChunk = 128;            // K tile per smem load
constexpr int kMaxBlockN = 32;          // block_reduce path is gated to n <= 32

__device__ __forceinline__ uint32_t round_to_tf32_bits_dev(float v) {
    uint32_t b = __float_as_uint(v);
    const uint32_t abs_b = b & 0x7fffffffu;
    if (abs_b >= 0x7f800000u) return b;
    const uint32_t lsb = (b >> 13) & 1u;
    b += 0x0fffu + lsb;
    b &= 0xffffe000u;
    return b;
}

template <bool kScalarCheck, bool kEnable2xTF32>
__global__ void hc_prenorm_block_reduce_mma_kernel(
    const __nv_bfloat16* __restrict__ a, const float* __restrict__ b,
    float* __restrict__ d, float* __restrict__ sqr_sum,
    int64_t a_stride_m, int64_t a_stride_k, int64_t b_stride_n,
    int64_t b_stride_k, int64_t d_stride_split, int64_t d_stride_m,
    int64_t d_stride_n, int64_t s_stride_split, int64_t s_stride_m,
    int m, int n, int k, int num_splits) {

    const int n_blocks_m = (m + kMmaM - 1) / kMmaM;
    const int linear = blockIdx.x;
    const int m_tile = linear % n_blocks_m;
    const int split_idx = linear / n_blocks_m;
    if (split_idx >= num_splits) return;

    const int m_base = m_tile * kMmaM;
    const int rows_in_block = min(kMmaM, m - m_base);

    const int k_begin = split_k_begin(k, num_splits, split_idx);
    const int k_end = split_k_end(k, num_splits, split_idx);

    extern __shared__ float smem_buf[];
    float* a_smem    = smem_buf;                              // [kMmaM, kKChunk]
    float* b_smem_hi = a_smem + kMmaM * kKChunk;              // [n,     kKChunk]
    // 2xTF32 mode adds a residual buffer right after b_smem_hi (host sizes smem).
    float* b_smem_lo = kEnable2xTF32 ? b_smem_hi + n * kKChunk : b_smem_hi;

    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x & 31;

    const int n_base = warp * kMmaN;
    const bool warp_active_mma = n_base < n;

    float c_acc[4] = {0.f, 0.f, 0.f, 0.f};
    float sq_partial[4] = {0.f, 0.f, 0.f, 0.f};
    const int sq_row_base = warp * 4;  // 4 warps × 4 rows = 16 rows total

    for (int k_off = k_begin; k_off < k_end; k_off += kKChunk) {
        const int kc_size = min(kKChunk, k_end - k_off);

        // Cooperative load A[rows_in_block, kc_size] → a_smem (bf16 → f32).
        for (int idx = threadIdx.x; idx < kMmaM * kKChunk; idx += kMmaThreads) {
            const int row = idx / kKChunk;
            const int kk = idx - row * kKChunk;
            float v = 0.f;
            if (row < rows_in_block && kk < kc_size) {
                const __nv_bfloat16 av =
                    a[static_cast<int64_t>(m_base + row) * a_stride_m +
                      static_cast<int64_t>(k_off + kk) * a_stride_k];
                v = __bfloat162float(av);
            }
            a_smem[row * kKChunk + kk] = v;
        }
        // Cooperative load B[n, kc_size] → b_smem with explicit tf32 rounding.
        // 2xTF32: also stage the residual (b_raw - b_hi rounded back to tf32).
        for (int idx = threadIdx.x; idx < n * kKChunk; idx += kMmaThreads) {
            const int col = idx / kKChunk;
            const int kk = idx - col * kKChunk;
            float v_hi = 0.f, v_lo = 0.f;
            if (col < n && kk < kc_size) {
                const float raw =
                    b[static_cast<int64_t>(col) * b_stride_n +
                      static_cast<int64_t>(k_off + kk) * b_stride_k];
                v_hi = __uint_as_float(round_to_tf32_bits_dev(raw));
                if (kEnable2xTF32) {
                    v_lo = __uint_as_float(round_to_tf32_bits_dev(raw - v_hi));
                }
            }
            b_smem_hi[col * kKChunk + kk] = v_hi;
            if (kEnable2xTF32) {
                b_smem_lo[col * kKChunk + kk] = v_lo;
            }
        }
        __syncthreads();

        if (warp_active_mma) {
            // Iterate this k-chunk in 8-element MMA steps.
            for (int kc = 0; kc < kc_size; kc += kMmaK) {
                const int t = lane;
                const int groupID = t >> 2;        // 0..7
                const int tid_in_group = t & 3;    // 0..3
                const int a_row0 = groupID;
                const int a_row1 = groupID + 8;
                // m16n8k8 .f32.tf32.tf32.f32 fragment layout for A (16x8 tf32):
                //   a[0] = A[groupID    ][tid_in_group    ]
                //   a[1] = A[groupID + 8][tid_in_group    ]
                //   a[2] = A[groupID    ][tid_in_group + 4]
                //   a[3] = A[groupID + 8][tid_in_group + 4]
                const uint32_t a0 = __float_as_uint(
                    a_smem[a_row0 * kKChunk + kc + tid_in_group]);
                const uint32_t a1 = __float_as_uint(
                    a_smem[a_row1 * kKChunk + kc + tid_in_group]);
                const uint32_t a2 = __float_as_uint(
                    a_smem[a_row0 * kKChunk + kc + tid_in_group + 4]);
                const uint32_t a3 = __float_as_uint(
                    a_smem[a_row1 * kKChunk + kc + tid_in_group + 4]);

                // B (8x8 tf32, .col layout):
                //   b[0] = B[tid_in_group    ][groupID]
                //   b[1] = B[tid_in_group + 4][groupID]
                // Our smem stores B as b_smem[n][k], so col=k is contiguous:
                const int b_n_idx = n_base + groupID;
                const uint32_t b0_hi = __float_as_uint(
                    b_smem_hi[b_n_idx * kKChunk + kc + tid_in_group]);
                const uint32_t b1_hi = __float_as_uint(
                    b_smem_hi[b_n_idx * kKChunk + kc + tid_in_group + 4]);
                uint32_t b0_lo = 0u, b1_lo = 0u;
                if (kEnable2xTF32) {
                    b0_lo = __float_as_uint(
                        b_smem_lo[b_n_idx * kKChunk + kc + tid_in_group]);
                    b1_lo = __float_as_uint(
                        b_smem_lo[b_n_idx * kKChunk + kc + tid_in_group + 4]);
                }

                if (kScalarCheck) {
                    // Manual matmul: each lane writes the 4 output elements
                    // that the MMA fragment would deposit (D layout: stride-2
                    // contiguous cols, since output is f32 not tf32):
                    //   c_acc[0] -> D[groupID    ][2*tid_in_group    ]
                    //   c_acc[1] -> D[groupID    ][2*tid_in_group + 1]
                    //   c_acc[2] -> D[groupID + 8][2*tid_in_group    ]
                    //   c_acc[3] -> D[groupID + 8][2*tid_in_group + 1]
                    // 2xTF32 oracle: read (b_hi + b_lo) so the scalar matmul
                    // models what the two mma.sync calls compute.
                    const int dn0 = n_base + 2 * tid_in_group;
                    const int dn1 = dn0 + 1;
                    float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
                    for (int kk = 0; kk < kMmaK; ++kk) {
                        const float a_r0 = a_smem[a_row0 * kKChunk + kc + kk];
                        const float a_r1 = a_smem[a_row1 * kKChunk + kc + kk];
                        float b_n0 = 0.f, b_n1 = 0.f;
                        if (dn0 < n) {
                            b_n0 = b_smem_hi[dn0 * kKChunk + kc + kk];
                            if (kEnable2xTF32) {
                                b_n0 += b_smem_lo[dn0 * kKChunk + kc + kk];
                            }
                        }
                        if (dn1 < n) {
                            b_n1 = b_smem_hi[dn1 * kKChunk + kc + kk];
                            if (kEnable2xTF32) {
                                b_n1 += b_smem_lo[dn1 * kKChunk + kc + kk];
                            }
                        }
                        acc0 += a_r0 * b_n0;
                        acc1 += a_r0 * b_n1;
                        acc2 += a_r1 * b_n0;
                        acc3 += a_r1 * b_n1;
                    }
                    (void)a0; (void)a1; (void)a2; (void)a3;
                    (void)b0_hi; (void)b1_hi; (void)b0_lo; (void)b1_lo;
                    c_acc[0] += acc0;
                    c_acc[1] += acc1;
                    c_acc[2] += acc2;
                    c_acc[3] += acc3;
                } else {
                    // First mma: c += a * b_hi
                    asm volatile(
                        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
                        "{%0, %1, %2, %3};\n"
                        : "+f"(c_acc[0]), "+f"(c_acc[1]),
                          "+f"(c_acc[2]), "+f"(c_acc[3])
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                          "r"(b0_hi), "r"(b1_hi));
                    if (kEnable2xTF32) {
                        // Second mma: c += a * b_lo (residual)
                        asm volatile(
                            "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
                            "{%0, %1, %2, %3};\n"
                            : "+f"(c_acc[0]), "+f"(c_acc[1]),
                              "+f"(c_acc[2]), "+f"(c_acc[3])
                            : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                              "r"(b0_lo), "r"(b1_lo));
                    }
                }
            }
        }

        // sqr_sum: every warp owns 4 rows.
        #pragma unroll
        for (int wr = 0; wr < 4; ++wr) {
            const int row = sq_row_base + wr;
            if (row < rows_in_block) {
                float lsq = 0.f;
                for (int kk = lane; kk < kc_size; kk += 32) {
                    const float a = a_smem[row * kKChunk + kk];
                    lsq += a * a;
                }
                #pragma unroll
                for (int o = 16; o > 0; o >>= 1)
                    lsq += __shfl_down_sync(0xffffffffu, lsq, o);
                if (lane == 0) sq_partial[wr] += lsq;
            }
        }
        __syncthreads();
    }

    // Store D fragment.
    if (warp_active_mma) {
        const int t = lane;
        const int d_row0 = m_base + (t >> 2);
        const int d_row1 = d_row0 + 8;
        const int d_col0 = n_base + (t & 3) * 2;
        const int d_col1 = d_col0 + 1;
        const int64_t base = static_cast<int64_t>(split_idx) * d_stride_split;

        if (d_row0 < m && d_col0 < n) {
            d[base + static_cast<int64_t>(d_row0) * d_stride_m +
              static_cast<int64_t>(d_col0) * d_stride_n] = c_acc[0];
        }
        if (d_row0 < m && d_col1 < n) {
            d[base + static_cast<int64_t>(d_row0) * d_stride_m +
              static_cast<int64_t>(d_col1) * d_stride_n] = c_acc[1];
        }
        if (d_row1 < m && d_col0 < n) {
            d[base + static_cast<int64_t>(d_row1) * d_stride_m +
              static_cast<int64_t>(d_col0) * d_stride_n] = c_acc[2];
        }
        if (d_row1 < m && d_col1 < n) {
            d[base + static_cast<int64_t>(d_row1) * d_stride_m +
              static_cast<int64_t>(d_col1) * d_stride_n] = c_acc[3];
        }
    }

    // Store sqr_sum: lane 0 of each warp holds finalized sums for its 4 rows.
    if (lane == 0) {
        #pragma unroll
        for (int wr = 0; wr < 4; ++wr) {
            const int row = sq_row_base + wr;
            if (row < rows_in_block) {
                sqr_sum[static_cast<int64_t>(split_idx) * s_stride_split +
                        static_cast<int64_t>(m_base + row) * s_stride_m] =
                    sq_partial[wr];
            }
        }
    }
}

} // namespace sm120_fallback

void sm120_tf32_hc_prenorm_gemm_fallback(const torch::Tensor& a,
                                         const torch::Tensor& b,
                                         const torch::Tensor& d,
                                         const torch::Tensor& sqr_sum,
                                         int m, int n, int k,
                                         int num_splits) {
    constexpr int threads = 256;
    const auto stream = at::cuda::getCurrentCUDAStream();
    const int64_t d_total =
        static_cast<int64_t>(num_splits) * static_cast<int64_t>(m) * n;
    const int64_t s_total =
        static_cast<int64_t>(num_splits) * static_cast<int64_t>(m);

    const int64_t d_stride_split = num_splits == 1 ? 0 : d.stride(0);
    const int64_t d_stride_m = num_splits == 1 ? d.stride(0) : d.stride(1);
    const int64_t d_stride_n = num_splits == 1 ? d.stride(1) : d.stride(2);
    const int64_t s_stride_split = num_splits == 1 ? 0 : sqr_sum.stride(0);
    const int64_t s_stride_m = num_splits == 1 ? sqr_sum.stride(0)
                                               : sqr_sum.stride(1);

    // Heuristic: MMA grid is (n_blocks_m × num_splits). When that's too small
    // to fill enough waves on the GPU, the scalar kernel (which has finer-grained
    // m-row-per-block parallelism) wins despite no tensor cores. Threshold of
    // 384 was picked from microbench: m=4096 sp=0 (256 blocks) regresses 0.7×
    // vs scalar; m=8192 sp=0 (512 blocks) wins. 384 lands between, slightly
    // biased toward safety so 2xTF32's extra ~10% cost stays in the win column.
    constexpr int64_t kMmaMinBlocks = 384;
    if (n <= 32) {
        const int n_blocks_m = (m + sm120_fallback::kMmaM - 1) /
                               sm120_fallback::kMmaM;
        const int64_t mma_total_blocks =
            static_cast<int64_t>(num_splits) *
            static_cast<int64_t>(n_blocks_m);
        if (sm120_hc_prenorm_mma_enabled() && n > 0
            && mma_total_blocks >= kMmaMinBlocks) {
            using sm120_fallback::kMmaM;
            using sm120_fallback::kMmaThreads;
            using sm120_fallback::kKChunk;
            const int64_t total_blocks = mma_total_blocks;
            const bool scalar_check = sm120_hc_prenorm_mma_scalar_check();
            const bool two_xtf32 = sm120_hc_prenorm_mma_2xtf32();
            // 2xTF32 stages a residual B buffer alongside the hi buffer.
            const size_t shared_bytes =
                static_cast<size_t>(kMmaM + (two_xtf32 ? 2 : 1) * n) *
                kKChunk * sizeof(float);
            #define DG_HC_LAUNCH_MMA(SC, TX) \
                sm120_fallback::hc_prenorm_block_reduce_mma_kernel<SC, TX> \
                    <<<static_cast<unsigned>(total_blocks), kMmaThreads, \
                       shared_bytes, stream>>>( \
                    reinterpret_cast<const __nv_bfloat16*>(a.data_ptr()), \
                    b.data_ptr<float>(), d.data_ptr<float>(), \
                    sqr_sum.data_ptr<float>(), a.stride(0), a.stride(1), \
                    b.stride(0), b.stride(1), d_stride_split, d_stride_m, \
                    d_stride_n, s_stride_split, s_stride_m, m, n, k, num_splits)
            if (scalar_check && two_xtf32)        { DG_HC_LAUNCH_MMA(true,  true);  }
            else if (scalar_check && !two_xtf32)  { DG_HC_LAUNCH_MMA(true,  false); }
            else if (!scalar_check && two_xtf32)  { DG_HC_LAUNCH_MMA(false, true);  }
            else                                  { DG_HC_LAUNCH_MMA(false, false); }
            #undef DG_HC_LAUNCH_MMA
            DG_CUDA_RUNTIME_CHECK(cudaGetLastError());
            return;
        }

        const int64_t total_blocks =
            static_cast<int64_t>(num_splits) * static_cast<int64_t>(m);
        const size_t shared_bytes =
            static_cast<size_t>(n + 1) * threads * sizeof(float);
        sm120_fallback::hc_prenorm_block_reduce_kernel<<<
            static_cast<unsigned>(total_blocks), threads, shared_bytes,
            stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(a.data_ptr()),
            b.data_ptr<float>(), d.data_ptr<float>(),
            sqr_sum.data_ptr<float>(), a.stride(0), a.stride(1),
            b.stride(0), b.stride(1), d_stride_split, d_stride_m, d_stride_n,
            s_stride_split, s_stride_m, m, n, k, num_splits);
        DG_CUDA_RUNTIME_CHECK(cudaGetLastError());
        return;
    }

    sm120_fallback::hc_prenorm_gemm_kernel<<<
        sm120_fallback::hc_fallback_grid(d_total), threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(a.data_ptr()), b.data_ptr<float>(),
        d.data_ptr<float>(), a.stride(0), a.stride(1), b.stride(0),
        b.stride(1), d_stride_split, d_stride_m, d_stride_n, m, n, k,
        num_splits);
    DG_CUDA_RUNTIME_CHECK(cudaGetLastError());

    sm120_fallback::hc_prenorm_sqr_sum_kernel<<<
        sm120_fallback::hc_fallback_grid(s_total), threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(a.data_ptr()),
        sqr_sum.data_ptr<float>(), a.stride(0), a.stride(1), s_stride_split,
        s_stride_m, m, k, num_splits);
    DG_CUDA_RUNTIME_CHECK(cudaGetLastError());
}

} // namespace deep_gemm
