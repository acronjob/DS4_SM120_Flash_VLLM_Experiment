#!/usr/bin/env python3
import argparse
import os
import time

import torch

import deep_gemm


def make_case(m: int, n: int, k: int, groups: int, device: str):
    assert k % 128 == 0
    assert n % 128 == 0
    assert m <= groups

    torch.manual_seed(1234)
    a_f32 = torch.randn((m, k), device=device, dtype=torch.float32) * 0.25
    a = a_f32.to(torch.float8_e4m3fn).contiguous()

    # Byte 0x7f is UE8M0 scale 1.0. int32 packs four 128-wide scale blocks.
    sfa_words = (k + 511) // 512
    sfa = torch.full((m, sfa_words), 0x7F7F7F7F, device=device, dtype=torch.int32)

    b = torch.randint(
        -128, 128, (groups, n, k // 2), device=device, dtype=torch.int8
    ).contiguous()
    sfb_raw = torch.full((groups, n, k // 32), 0x7F, device=device, dtype=torch.uint8)
    sfb = deep_gemm._C.sm120_prepack_fp8_fp4_sfb(sfb_raw, 128, n, k)

    grouped_layout = torch.arange(m, device=device, dtype=torch.int32)
    starts = torch.zeros((groups,), device=device, dtype=torch.int32)
    counts = torch.zeros((groups,), device=device, dtype=torch.int32)
    starts[:m] = torch.arange(m, device=device, dtype=torch.int32)
    counts[:m] = 1

    return a, sfa, b, sfb, grouped_layout, starts, counts


def call_kernel(case, out):
    a, sfa, b, sfb, grouped_layout, starts, counts = case
    deep_gemm._C.m_grouped_fp8_fp4_gemm_nt_contiguous_with_starts(
        (a, sfa),
        (b, sfb),
        out,
        grouped_layout,
        starts,
        counts,
        recipe_a=(1, 128),
        recipe_b=(1, 32),
    )


def set_mode(mode: str) -> None:
    os.environ.pop("DG_SM120_ENABLE_SMALL_M_MMA", None)
    os.environ.pop("DG_SM120_MOE_SKIP_SFA_FILL", None)
    os.environ.pop("DG_SM120_MOE_ROW_GROUPED", None)
    os.environ.pop("DG_SM120_MOE_ROW_GROUPED_SKIP_SFA_FILL", None)
    if mode == "small_m":
        os.environ["DG_SM120_ENABLE_SMALL_M_MMA"] = "1"
    elif mode == "skip_fill":
        os.environ["DG_SM120_MOE_SKIP_SFA_FILL"] = "1"
    elif mode == "row_grouped":
        os.environ["DG_SM120_MOE_ROW_GROUPED"] = "1"
    elif mode == "row_grouped_skip_fill":
        os.environ["DG_SM120_MOE_ROW_GROUPED"] = "1"
        os.environ["DG_SM120_MOE_ROW_GROUPED_SKIP_SFA_FILL"] = "1"


def bench(case, out, mode: str, warmup: int, iters: int) -> float:
    set_mode(mode)

    for _ in range(warmup):
        call_kernel(case, out)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        call_kernel(case, out)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return elapsed * 1e6 / iters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=16)
    parser.add_argument("--n", type=int, default=7168)
    parser.add_argument("--k", type=int, default=2048)
    parser.add_argument("--groups", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--modes",
        default="default,skip_fill,small_m,row_grouped,row_grouped_skip_fill",
        help="Comma-separated modes: default, skip_fill, small_m, row_grouped, row_grouped_skip_fill",
    )
    args = parser.parse_args()

    case = make_case(args.m, args.n, args.k, args.groups, args.device)
    modes = [x for x in args.modes.split(",") if x]
    outputs = {
        mode: torch.empty((args.m, args.n), device=args.device, dtype=torch.bfloat16)
        for mode in modes
    }

    set_mode("default")
    d_ref = torch.empty((args.m, args.n), device=args.device, dtype=torch.bfloat16)
    call_kernel(case, d_ref)
    for mode, out in outputs.items():
        set_mode(mode)
        call_kernel(case, out)
    torch.cuda.synchronize()

    ref_abs = d_ref.float().abs().clamp_min(1e-6)
    print(f"shape: m={args.m} n={args.n} k={args.k} groups={args.groups}")
    timings = {}
    for mode in modes:
        diff = (d_ref.float() - outputs[mode].float()).abs()
        timings[mode] = bench(case, outputs[mode], mode, args.warmup, args.iters)
        print(
            f"{mode}: us={timings[mode]:.3f} "
            f"speedup={timings['default'] / timings[mode]:.3f}x "
            f"max_abs_diff={diff.max().item():.6g} "
            f"mean_abs_diff={diff.mean().item():.6g} "
            f"max_rel_diff={(diff / ref_abs).max().item():.6g}"
        )


if __name__ == "__main__":
    main()
