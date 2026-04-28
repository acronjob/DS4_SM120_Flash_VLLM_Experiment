#!/usr/bin/env python3
"""Parity test for DG_SM120_FLASH_FUSION_PREFILL.

Compares output / lse from `sm120_sparse_mla_prefill_from_bf16_workspace_split`
with the 3-kernel chain (FLASH_FUSION_PREFILL=0, score+softmax+output) vs.
the new fused `sparse_mla_prefill_workspace_flash_mma_kernel` (FLASH_FUSION_PREFILL=1).

Tile order (M=16 heads × N=32 slots) and the online-softmax merge add small
numeric drift relative to the chain — target ≤1e-3 max_abs on bf16 outputs.
"""
import argparse
import os
import sys
import time

import torch

import deep_gemm


def make_case(tokens, heads, kv_tokens, topk, head_dim=512,
              dtype=torch.bfloat16, device="cuda", seed=20260427):
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    q = (torch.randn((tokens, heads, head_dim), device=device,
                     dtype=torch.float32, generator=g) * 0.05).to(dtype)
    kv = (torch.randn((kv_tokens, 1, head_dim), device=device,
                      dtype=torch.float32, generator=g) * 0.05).to(dtype)
    indices = torch.randint(0, kv_tokens, (tokens, 1, topk),
                            device=device, dtype=torch.int32, generator=g)
    lens = torch.randint(low=max(1, topk // 4), high=topk + 1,
                         size=(tokens,), device=device, dtype=torch.int64,
                         generator=g)
    sink = torch.zeros((heads,), device=device, dtype=torch.float32)
    return (q.contiguous(), kv.contiguous(), indices.contiguous(), lens, sink)


def call_split(case, head_dim=512):
    q, kv, indices, lens, sink = case
    out = torch.empty((q.shape[0], q.shape[1], head_dim),
                      device=q.device, dtype=q.dtype)
    softmax_scale = head_dim ** -0.5
    out_t, _max_logits, lse_t = (
        deep_gemm._C.sm120_sparse_mla_prefill_from_bf16_workspace_split(
            q, kv, indices, lens, sink, head_dim, softmax_scale, out
        )
    )
    return out_t, lse_t


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", type=int, nargs="+", default=[64, 256, 1024])
    p.add_argument("--heads", type=int, default=32)
    p.add_argument("--kv-tokens", type=int, default=4096)
    p.add_argument("--topks", type=int, nargs="+", default=[256, 1024, 2048])
    p.add_argument("--seeds", type=int, nargs="+",
                   default=[20260427, 20260428, 20260429])
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available", file=sys.stderr)
        return 2

    flash = os.environ.get("DG_SM120_FLASH_FUSION_PREFILL", "0")
    print(
        f"DG_SM120_FLASH_FUSION_PREFILL={flash} "
        f"DG_SM120_FAST_SPARSE_MLA={os.environ.get('DG_SM120_FAST_SPARSE_MLA','?')} "
        f"SCALAR_CHECK={os.environ.get('DG_SM120_FLASH_FUSION_PREFILL_SCALAR_CHECK','?')}",
        flush=True,
    )

    out_records = {}
    for tokens in args.tokens:
        for topk in args.topks:
            if topk > args.kv_tokens:
                continue
            for seed in args.seeds:
                case = make_case(tokens, args.heads, args.kv_tokens, topk,
                                 seed=seed)
                out, lse = call_split(case)
                key = (tokens, topk, seed)
                out_records[key] = (out.detach().cpu(), lse.detach().cpu())
                print(
                    f"  case t={tokens} topk={topk} seed={seed}: "
                    f"out_max={out.float().abs().max().item():.5e} "
                    f"out_mean={out.float().abs().mean().item():.5e} "
                    f"lse_max={lse.float().abs().max().item():.5e}",
                    flush=True,
                )

    save_path = os.environ.get(
        "PARITY_OUT", f"/tmp/parity_flash_fusion_prefill_FLASH{flash}.pt"
    )
    torch.save(out_records, save_path)
    print(f"saved -> {save_path}", flush=True)

    if os.environ.get("MICROBENCH", "0") == "1":
        warmup = int(os.environ.get("MICROBENCH_WARMUP", "5"))
        iters = int(os.environ.get("MICROBENCH_ITERS", "20"))
        bench_tokens = int(os.environ.get("MICROBENCH_TOKENS", "1024"))
        bench_topk = int(os.environ.get("MICROBENCH_TOPK", "2048"))
        bench_kv = int(os.environ.get("MICROBENCH_KV", "4096"))
        case = make_case(bench_tokens, args.heads, bench_kv, bench_topk,
                         seed=20260427)
        for _ in range(warmup):
            call_split(case)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            call_split(case)
        torch.cuda.synchronize()
        us = (time.perf_counter() - t0) * 1e6 / iters
        print(
            f"MICROBENCH FLASH_PREFILL={flash} t={bench_tokens} h={args.heads} "
            f"topk={bench_topk} kv={bench_kv}: {us:.2f} us/call (avg of {iters})",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
