#!/usr/bin/env python3
"""Parity test for DG_SM120_OUTPUT_FUSED_SOFTMAX.

Compares output / lse from the production `sparse_mla_decode_from_bf16_workspace_split`
path with the standalone softmax + sstat output kernel (FUSED=0) vs. the new
`sparse_mla_workspace_output_fused_softmax_kernel` (FUSED=1).

Math design preserves multiplication / accumulation order vs the standalone
softmax_kernel + sstat output_kernel chain, so we expect bit-exact results.
"""
import argparse
import os
import sys

import torch

import deep_gemm


def make_case(batch, heads, topk, head_dim=512, dtype=torch.bfloat16, device="cuda", seed=20260427):
    torch.manual_seed(seed)
    q = torch.randn((batch, 1, heads, head_dim), device=device, dtype=dtype) * 0.05
    kv = torch.randn((batch, topk, head_dim), device=device, dtype=dtype) * 0.05
    # Vary lengths per-batch to stress validity masking in softmax.
    lens = torch.randint(low=max(1, topk // 4), high=topk + 1, size=(batch,),
                         device=device, dtype=torch.int64)
    sink = torch.zeros((heads,), device=device, dtype=torch.float32)
    return q.contiguous(), kv.contiguous(), lens, sink


def call_split(q, kv, lens, sink, head_dim=512):
    out = torch.empty_like(q)
    main_topk = kv.shape[1]
    softmax_scale = head_dim ** -0.5
    deep_gemm._C.sm120_sparse_mla_decode_from_bf16_workspace_split(
        q, kv, lens, None, sink, main_topk, 0, head_dim, softmax_scale, out
    )
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", type=int, nargs="+",
                        default=[1, 4, 8, 16])
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--topks", type=int, nargs="+",
                        default=[64, 96, 128])
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[20260427, 20260428, 20260429])
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available", file=sys.stderr)
        return 2

    # ENV is captured by static lambda — caller toggles
    # DG_SM120_OUTPUT_FUSED_SOFTMAX externally; we just record what's set.
    fused = os.environ.get("DG_SM120_OUTPUT_FUSED_SOFTMAX", "0")
    print(f"DG_SM120_OUTPUT_FUSED_SOFTMAX={fused}", flush=True)
    out_records = {}
    for batch in args.batches:
        for topk in args.topks:
            for seed in args.seeds:
                case = make_case(batch, args.heads, topk, seed=seed)
                q, kv, lens, sink = case
                out = call_split(q, kv, lens, sink)
                key = (batch, topk, seed)
                out_records[key] = out.detach().cpu()
                print(f"  case b={batch} topk={topk} seed={seed}: "
                      f"out_max={out.float().abs().max().item():.5e} "
                      f"out_mean={out.float().abs().mean().item():.5e}", flush=True)

    # Persist result to disk under a path keyed by env, so the caller can diff.
    save_path = os.environ.get(
        "PARITY_OUT", f"/tmp/parity_fused_softmax_FUSED{fused}.pt"
    )
    torch.save(out_records, save_path)
    print(f"saved -> {save_path}", flush=True)

    # Micro-bench: prod-shape decode call timing.
    if os.environ.get("MICROBENCH", "0") == "1":
        import time
        warmup = int(os.environ.get("MICROBENCH_WARMUP", "20"))
        iters = int(os.environ.get("MICROBENCH_ITERS", "200"))
        # Production prefill chunk shape: 16 rows, heads=32, topk=128
        case = make_case(16, 32, 128, seed=20260427)
        q, kv, lens, sink = case
        # Warm up
        for _ in range(warmup):
            call_split(q, kv, lens, sink)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            call_split(q, kv, lens, sink)
        torch.cuda.synchronize()
        us = (time.perf_counter() - t0) * 1e6 / iters
        print(f"MICROBENCH FUSED={fused} b=16 h=32 topk=128: "
              f"{us:.2f} us/call (avg of {iters})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
