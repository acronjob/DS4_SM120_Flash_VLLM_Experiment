#!/usr/bin/env python3
"""Parity + microbench harness for DG_SM120_SCORE_QSTAT_VEC8.

Compares the score+softmax+output chain output between bf16x4 and bf16x8
variants of sparse_mla_workspace_score_tiled_qstat_vec_kernel. Per-d FMA
order changes only in stride width (4 vs 8 elements per inner unroll),
which is sub-bf16-ULP drift over the running fp32 partial.
"""
import argparse
import os
import sys
import time

import torch
import deep_gemm


def make_case(batch, heads, topk, head_dim=512, dtype=torch.bfloat16,
              device="cuda", seed=20260428):
    torch.manual_seed(seed)
    q = torch.randn((batch, 1, heads, head_dim), device=device, dtype=dtype) * 0.05
    kv = torch.randn((batch, topk, head_dim), device=device, dtype=dtype) * 0.05
    lens = torch.randint(low=max(1, topk // 4), high=topk + 1,
                         size=(batch,), device=device, dtype=torch.int64)
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
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 4, 8, 16])
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--topks", type=int, nargs="+", default=[64, 96, 128])
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[20260428, 20260429, 20260430])
    args = parser.parse_args()

    fast = os.environ.get("DG_SM120_FAST_SPARSE_MLA", "0")
    vec = os.environ.get("DG_SM120_SCORE_QSTAT_VEC", "0")
    vec8 = os.environ.get("DG_SM120_SCORE_QSTAT_VEC8", "0")
    print(f"FAST={fast} VEC={vec} VEC8={vec8}", flush=True)

    out_records = {}
    for batch in args.batches:
        for topk in args.topks:
            for seed in args.seeds:
                q, kv, lens, sink = make_case(batch, args.heads, topk,
                                              seed=seed)
                out = call_split(q, kv, lens, sink)
                key = (batch, topk, seed)
                out_records[key] = out.detach().cpu()
                m = out.float().abs().max().item()
                a = out.float().abs().mean().item()
                print(f"  b={batch} topk={topk} seed={seed}: "
                      f"out_max={m:.5e} out_mean={a:.5e}", flush=True)

    save_path = os.environ.get(
        "PARITY_OUT", f"/tmp/parity_score_qstat_vec8_VEC{vec}_V8{vec8}.pt"
    )
    torch.save(out_records, save_path)
    print(f"saved -> {save_path}", flush=True)

    if os.environ.get("MICROBENCH", "0") == "1":
        warmup = int(os.environ.get("MICROBENCH_WARMUP", "20"))
        iters = int(os.environ.get("MICROBENCH_ITERS", "300"))
        # Prod chunk-prefill shape b=16, h=32, topk=128
        for shape in [(1, 32, 128), (4, 32, 128), (8, 32, 128), (16, 32, 128)]:
            b, h, k = shape
            q, kv, lens, sink = make_case(b, h, k, seed=20260428)
            for _ in range(warmup):
                call_split(q, kv, lens, sink)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                call_split(q, kv, lens, sink)
            torch.cuda.synchronize()
            us = (time.perf_counter() - t0) * 1e6 / iters
            print(f"MICROBENCH VEC={vec} VEC8={vec8} b={b} h={h} topk={k}: "
                  f"{us:.2f} us/call (avg of {iters})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
