#!/usr/bin/env python3
"""Parity test for DG_SM120_FLASH_FUSION.

Compares output / lse from `sparse_mla_decode_from_bf16_workspace_split` with
the 3-kernel chain (FLASH=0, score+softmax+output) vs. the new fused
`sparse_mla_workspace_flash_kernel` (FLASH=1).

The flash kernel uses a different reduction order (2 threads/slot for QK,
full slot range for PV) so we don't expect bit-exact — target ≤1e-3 max_abs
on bf16 outputs (well within bf16 epsilon at this output magnitude scale).
"""
import argparse
import os
import sys
import time

import torch

import deep_gemm


def make_case(batch, heads, topk, head_dim=512, dtype=torch.bfloat16,
              device="cuda", seed=20260427):
    torch.manual_seed(seed)
    q = torch.randn((batch, 1, heads, head_dim), device=device, dtype=dtype) * 0.05
    kv = torch.randn((batch, topk, head_dim), device=device, dtype=dtype) * 0.05
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
    p = argparse.ArgumentParser()
    p.add_argument("--batches", type=int, nargs="+", default=[1, 4, 8, 16])
    p.add_argument("--heads", type=int, default=32)
    p.add_argument("--topks", type=int, nargs="+", default=[64, 96, 128])
    p.add_argument("--seeds", type=int, nargs="+",
                   default=[20260427, 20260428, 20260429])
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available", file=sys.stderr)
        return 2

    flash = os.environ.get("DG_SM120_FLASH_FUSION", "0")
    print(f"DG_SM120_FLASH_FUSION={flash} "
          f"DG_SM120_FAST_SPARSE_MLA={os.environ.get('DG_SM120_FAST_SPARSE_MLA','?')}",
          flush=True)
    out_records = {}
    for batch in args.batches:
        for topk in args.topks:
            for seed in args.seeds:
                q, kv, lens, sink = make_case(batch, args.heads, topk, seed=seed)
                out = call_split(q, kv, lens, sink)
                key = (batch, topk, seed)
                out_records[key] = out.detach().cpu()
                print(f"  case b={batch} topk={topk} seed={seed}: "
                      f"out_max={out.float().abs().max().item():.5e} "
                      f"out_mean={out.float().abs().mean().item():.5e}",
                      flush=True)

    save_path = os.environ.get(
        "PARITY_OUT", f"/tmp/parity_flash_fusion_FLASH{flash}.pt"
    )
    torch.save(out_records, save_path)
    print(f"saved -> {save_path}", flush=True)

    if os.environ.get("MICROBENCH", "0") == "1":
        warmup = int(os.environ.get("MICROBENCH_WARMUP", "20"))
        iters = int(os.environ.get("MICROBENCH_ITERS", "200"))
        bench_topk = int(os.environ.get("MICROBENCH_TOPK", "128"))
        q, kv, lens, sink = make_case(16, 32, bench_topk, seed=20260427)
        for _ in range(warmup):
            call_split(q, kv, lens, sink)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            call_split(q, kv, lens, sink)
        torch.cuda.synchronize()
        us = (time.perf_counter() - t0) * 1e6 / iters
        print(f"MICROBENCH FLASH={flash} b=16 h=32 topk={bench_topk}: "
              f"{us:.2f} us/call (avg of {iters})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
