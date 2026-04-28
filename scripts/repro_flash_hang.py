#!/usr/bin/env python3
"""Probe hang: run the SAME case twice, then a different one."""
import os
import sys
import time

import torch

import deep_gemm


def make_case(batch, heads, topk, head_dim=512, seed=20260427):
    torch.manual_seed(seed)
    q = torch.randn((batch, 1, heads, head_dim), device="cuda",
                    dtype=torch.bfloat16) * 0.05
    kv = torch.randn((batch, topk, head_dim), device="cuda",
                     dtype=torch.bfloat16) * 0.05
    lens = torch.randint(low=max(1, topk // 4), high=topk + 1, size=(batch,),
                         device="cuda", dtype=torch.int64)
    sink = torch.zeros((heads,), device="cuda", dtype=torch.float32)
    return q.contiguous(), kv.contiguous(), lens, sink


def call(q, kv, lens, sink, head_dim=512):
    out = torch.empty_like(q)
    main_topk = kv.shape[1]
    deep_gemm._C.sm120_sparse_mla_decode_from_bf16_workspace_split(
        q, kv, lens, None, sink, main_topk, 0, head_dim, head_dim ** -0.5, out
    )
    torch.cuda.synchronize()
    return out


def main():
    flash = os.environ.get("DG_SM120_FLASH_FUSION", "0")
    print(f"FLASH={flash}", flush=True)
    # Two identical calls (same shape, same seed) to test call-count vs data.
    cases = [
        ("ident-1", 1, 32, 64, 20260427),
        ("ident-2", 1, 32, 64, 20260427),
        ("ident-3", 1, 32, 64, 20260427),
        ("seed-B",  1, 32, 64, 20260428),
    ]
    for tag, b, h, k, seed in cases:
        print(f"  [{tag}] b={b} h={h} k={k} seed={seed}: ", end="", flush=True)
        q, kv, lens, sink = make_case(b, h, k, seed=seed)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = call(q, kv, lens, sink)
        elapsed = time.perf_counter() - t0
        m = out.float().abs().max().item()
        print(f"out_max={m:.5e}  ({elapsed*1000:.1f} ms)", flush=True)


if __name__ == "__main__":
    sys.exit(main())
