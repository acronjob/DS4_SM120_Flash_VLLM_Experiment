# DeepGEMM SM120 / DeepSeek V4 Flash Handoff

This repository is a research fork of DeepGEMM aimed at running
`deepseek-ai/DeepSeek-V4-Flash` under vLLM on 2x NVIDIA RTX PRO 6000
Blackwell workstation GPUs, which report compute capability SM120.

## Current Functional State

- Target container image: `vllm/vllm-openai:deepseekv4-cu130`.
- Service entry point: `docker-compose.yml`.
- API port: host `8080` mapped to container `8000`.
- Model: `deepseek-ai/DeepSeek-V4-Flash`.
- vLLM configuration:
  - `--attention-backend FLASHMLA_SPARSE`
  - `--moe-backend deep_gemm`
  - `--kv-cache-dtype fp8_ds_mla`
  - tensor parallel size `2`
  - expert parallel enabled
  - max model length `131072`
- The model loads, serves, and returns valid chat completions.
- The target of `100+ tok/s` per single request has not been reached.

## Runtime Defaults

`docker-compose.yml` currently defaults to:

- `VLLM_MAX_MODEL_LEN=131072`
- `VLLM_KV_CACHE_MEMORY_BYTES=8589934592`
- `VLLM_MAX_NUM_BATCHED_TOKENS=4096`
- `VLLM_MAX_NUM_SEQS=4`
- `VLLM_MAX_CUDAGRAPH_CAPTURE_SIZE=4`
- `VLLM_PERFORMANCE_MODE=balanced`
- `VLLM_OPTIMIZATION_LEVEL=2`
- `DG_SM120_ACTIVE_HEADS=32`
- `DG_SM120_SEQUENTIAL_COMPRESSOR=1`
- `DG_SM120_ENABLE_B12X_MOE=0`
- `DG_SM120_PREFILL_WORKSPACE_CHUNK=64`

Note: `docker/patch_vllm_deepseekv4.py` currently has an internal default of
`DG_SM120_PREFILL_WORKSPACE_CHUNK=16`, but the compose environment overrides it
to `64`. The running service was last validated with chunk `64`.

## What Has Been Implemented

### vLLM Container and Patcher

- `Dockerfile.vllm-nightly-sm120` builds from the DeepSeek V4 CUDA 13 vLLM
  image and installs the development dependencies needed for local extension
  builds.
- `docker-compose.yml` installs this repo editable, patches installed vLLM at
  startup, and serves DeepSeek V4 Flash.
- `docker/patch_vllm_deepseekv4.py` patches the installed vLLM package in-place
  instead of requiring a full vLLM rebuild.
- SM120 is treated as DeepGEMM-capable in vLLM platform checks.
- DeepGEMM MXFP4/FP8 paths are enabled on SM120.
- MXFP4 scales are prepacked for the SM120 DeepGEMM path.
- DeepSeek V4 FP8 einsum dispatches to SM120 DeepGEMM kernels where available.
- Sparse MLA decode paths are patched to use SM120 extension kernels.
- Sparse-indexer behavior is patched for SM120 and small/full-context shapes.
- DeepSeek V4 compressor scheduling is patched to avoid first-request OOM.
- Optional layer and kernel profiling hooks exist but are off by default.

### DeepGEMM Extension Work

SM120-specific CUDA/C++ sources are present:

- `csrc/sm120_fp8_fp8_cutlass.cu`
- `csrc/sm120_fp8_fp4_cutlass.cu`
- `csrc/sm120_fp8_gemm_fallback.cu`
- `csrc/sm120_mqa_logits_fallback.cu`
- `csrc/sm120_hc_prenorm_fallback.cu`
- `csrc/sm120_sparse_mla_decode.cu`
- `csrc/sm120_profile.hpp`

Important implemented paths:

- SM120 FP8/FP4 MoE support sufficient for model load.
- SM120 C128 FP8 decode/projection support sufficient for functional serving.
- SM120 FP8 `bhr,hdr->bhd` einsum path for the DeepSeek V4 output projection.
- SM120 sparse MLA decode kernels, including BF16 workspace variants.
- SM120 sparse MLA prefill from BF16 workspace:
  - exported as `deep_gemm._C.sm120_sparse_mla_prefill_from_bf16_workspace`
  - validated synthetically for BF16 tolerance
  - not enough by itself to fix long-prompt TTFT
- SM120 subchunked sparse prefill path in the vLLM patcher:
  - builds a bounded BF16 workspace per chunk
  - calls `sm120_sparse_mla_decode_from_bf16_workspace_split`
  - avoids allocating one huge gathered workspace for long prompts

### Bench and Utility Scripts

Useful scripts include:

- `scripts/test_deepseek_v4_flash_api.sh`
- `scripts/bench_deepseek_v4_flash_api.sh`
- `scripts/bench_deepseek_v4_flash_parallel_sweep.sh`
- `scripts/estimate_deepseek_v4_flash_memory.py`
- `scripts/bench_sm120_fp8_bhr_hdr_bhd.py`
- `scripts/bench_sm120_fp8_m1.py`
- `scripts/bench_sm120_fp8_projection_shapes.py`
- `scripts/bench_sm120_mhc.py`
- `scripts/bench_sm120_moe_small_m.py`
- `scripts/bench_sm120_workspace_attention.py`

## What Worked

- DeepSeek V4 Flash can load and serve on 2x RTX PRO 6000 with this fork.
- Explicit KV reservation fixed the first-request OOM:
  - `--kv-cache-memory-bytes 8589934592`
- CUDA graph capture works with the current non-eager configuration.
- Parallel serving works better than single-request serving:
  - earlier 512-token decode tests reached roughly `140 tok/s` aggregate at
    concurrency `4`
  - single request remained around `60-65 tok/s` decode-only in the best tests
- C128 FP8 M=1 projection microbenchmarks improved substantially after the
  contiguous UE8M0 k-block specialization:
  - example projection shapes improved from roughly `18-25 us` to `14-18 us`
  - this did not translate into a large end-to-end speedup
- Native SM120 sparse MLA prefill op compiled and passed synthetic checks:
  - BF16 output max error around `0.0039-0.0078`
  - LSE max error around `4.8e-7`
- Synthetic sparse prefill showed the existing workspace split attention kernel
  is much faster than the naive native fused prefill kernel:
  - `S=64,H=64,K=2048`: native about `11.3 ms`, workspace split plus gather
    about `2.85 ms`
  - `S=128,H=64,K=2048`: native about `17.4 ms`, workspace split plus gather
    about `6.3 ms`
- Long decode length alone is not the main problem:
  - small-prompt completions from `128` to `1024` output tokens stayed around
    `58-65 tok/s`

## What Failed or Underperformed

- Direct FlashMLA sparse prefill cannot be used as-is on SM120:
  - calling the bundled binary directly fails with
    `Sparse Attention Forward Kernel is only supported on SM90a and SM100f architectures.`
  - do not just bypass Python-side capability checks; the failure is enforced
    by the compiled FlashMLA extension itself
  - this is why `flash_mla_sparse_fwd` had to be patched for SM120 instead of
    simply allowing SM120 through `is_flashmla_sparse_supported`
- Long uncached prompts are still too slow because TTFT/prefill dominates.
  - decode-only tests with short prompts can look acceptable and are misleading
    for long-context use
  - repeated prompts can also be misleading because prefix caching hides prefill
    cost; use unique random text from token 0 when measuring TTFT
  - non-streaming completion tok/s is a bad metric for long prompts because it
    divides completion tokens by prefill + decode wall time
- The first Python/PyTorch sparse prefill correctness fallback was functional
  but extremely slow.
  - it did per-token Python looping, `index_select`, FP32 conversion, matmul,
    softmax, and output copy
  - it kept the model usable but caused severe TTFT collapse on substantial
    prompts
  - do not revive this path except as a correctness reference
- Native `sm120_sparse_mla_prefill_from_bf16_workspace` improved the fallback
  but was still too slow end-to-end.
  - it passed synthetic correctness, but it is still scalar CUDA over BF16
    workspace values rather than a production tensor-core sparse attention
  - synthetic result example: `S=128,H=64,K=2048` took about `17.4 ms`
  - it should be treated as a fallback/reference kernel, not the final prefill
    solution
- The existing workspace split attention kernel was faster synthetically, but
  using it from vLLM is still not a complete fix.
  - it requires materializing gathered BF16 workspace chunks from sparse
    indices
  - the gather/materialization overhead and repeated per-layer calls still
    dominate enough that live TTFT only improved modestly
  - a full-size gathered workspace for long prompts would be too large, so the
    path must stay subchunked unless memory use is redesigned
- The subchunked workspace path improved TTFT only modestly in the live server:
  - around `4.1k` prompt tokens: `13.2s` TTFT -> `11.3s`
  - around `8.2k` prompt tokens: `28.7s` TTFT -> `24.3s`
  - around `16.4k` prompt tokens: `62-64s` TTFT -> `55.8s`
- Synthetic chunk sweep suggested chunk `16` may be faster than chunk `64`, but
  this has not been validated end-to-end because the running compose default
  still overrides to `64`.
  - chunk `16` was best in one synthetic sweep for `S=512,H=64,K=2048`
  - larger chunks reduce Python/operator loop count but increase gathered
    workspace pressure and were not consistently faster
  - do not assume bigger chunk is better
- C128 FP8 M=1 projection microbench improvements did not materially change API
  throughput.
  - the contiguous UE8M0 k-block path is real and faster on projection-shaped
    microbenches
  - end-to-end speed barely moved, so C128 projection is not the only bottleneck
  - future work should not spend another cycle only tuning this path without a
    profile proving it is hot
- The first SM120 strategy of "just use SM100" is invalid.
  - DeepGEMM's SM100 path relies on TMEM assumptions that do not map directly
    to SM120
  - C4 can fall back elsewhere, but C128 routes into DeepGEMM and needed SM120
    handling
- vLLM rebuilds are expensive and were mostly avoided.
  - patching the installed vLLM package at container startup was enough for the
    experiments here
  - rebuilding vLLM is unlikely by itself to fix performance unless it includes
    new compiled FlashMLA/CUDA kernels
- A vLLM profiler restart attempt failed due to invalid profiler CLI key names.
  Retry with a valid JSON `--profiler-config` if profiling is needed.
  - dotted kebab-case CLI keys like
    `--profiler-config.torch-profiler-dir` were rejected
  - use a single JSON config string if trying again
  - avoid enabling `DG_SM120_KERNEL_PROFILE` while CUDA graph capture is active
    unless you intend to benchmark with synchronization overhead
- Optional b12x MoE integration exists behind `DG_SM120_ENABLE_B12X_MOE=1` but
  remains disabled because it has not been proven stable end-to-end.
  - it may still be useful as a reference for SM120 FP8 activation x packed FP4
    expert-weight flow and scale swizzles
  - do not enable it by default without correctness and model-load validation
- Single-request `100+ tok/s` has not been achieved.
  - aggregate throughput can exceed `100 tok/s` with concurrent requests, but
    per-request decode and long-prompt TTFT remain below the goal
  - do not report aggregate concurrency numbers as satisfying the single-request
    target

## Current Bottleneck Understanding

There are two separate regimes:

- Decode-only throughput is decent but still below target for one request.
- Long prompt throughput collapses because uncached prefill/TTFT is slow.

The most important remaining bottleneck is DeepSeek V4 sparse MLA prefill on
SM120. The bundled FlashMLA sparse prefill kernel rejects SM120, and the current
replacement paths still involve BF16 workspace materialization and many
per-layer operations.

The second likely bottleneck is sparse MLA decode and MoE decode efficiency.
The current implementation still contains fallback/workspace-heavy paths that
are functional but not production-grade.

## Next Work

1. Build a production SM120 sparse MLA prefill kernel.
   Avoid Python loops, avoid full BF16 workspace materialization, and compute
   directly from FP8 MLA cache plus sparse indices where possible.

2. Validate `DG_SM120_PREFILL_WORKSPACE_CHUNK=16` end-to-end.
   The code default is now `16`, but compose currently exports `64`. Change one
   source of truth, restart once, and measure token-counted TTFT for about
   `4k`, `8k`, and `16k` uncached prompt tokens.

3. Replace workspace-heavy sparse MLA decode with direct SM120 tensor-core
   computation.
   Current workspace split kernels are useful stopgaps but should not be the
   final production path.

4. Profile the live prefill path with valid vLLM profiler config.
   Use unique prompts from token 0 so prefix caching does not hide TTFT.

5. Revisit the C128 decode/MoE path after prefill is fixed.
   The C128 projection microbench improved, but end-to-end did not. The
   remaining runtime is likely elsewhere.

6. Audit all remaining fallback paths.
   Files with `fallback` in the name are functional scaffolding, not proof that
   performance is production-ready.

## Useful Commands

Start or restart:

```bash
docker compose up -d --force-recreate vllm
```

Health check:

```bash
curl -sS http://127.0.0.1:8080/health
```

Tiny correctness request:

```bash
curl -sS http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-ai/DeepSeek-V4-Flash","messages":[{"role":"user","content":"Reply exactly OK."}],"max_tokens":8,"temperature":0}'
```

512-token API benchmark:

```bash
BASE_URL=http://127.0.0.1:8080 \
MAX_TOKENS=512 \
MIN_TOKENS=512 \
CONCURRENCY=4 \
scripts/bench_deepseek_v4_flash_api.sh
```

DeepGEMM extension rebuild inside the container:

```bash
docker compose exec -T vllm bash -lc \
  'cd /workspace/DeepGEMM && MAX_JOBS=8 DG_FORCE_BUILD=1 python3 setup.py build_ext --inplace'
```

## Caveats

- This is a working research prototype, not an upstream-quality patch.
- vLLM is patched in-place at container startup rather than rebuilt from source.
- `vllm_src/` is only reference source; the running container patches the
  installed package in the image.
- `VLLM_*` local config variables produce "unknown vLLM environment variable"
  warnings. They are consumed by the compose startup shell before `vllm serve`,
  so the warnings are noisy but expected.
- Keep `DG_SM120_KERNEL_PROFILE=0` for real benchmarks. The kernel profiler uses
  CUDA event synchronization and will distort throughput when enabled.
