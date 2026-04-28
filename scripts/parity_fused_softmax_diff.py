#!/usr/bin/env python3
"""Diff two parity_fused_softmax.py output dumps."""
import argparse
import sys

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("a", help="path to FUSED=0 dump")
    p.add_argument("b", help="path to FUSED=1 dump")
    args = p.parse_args()
    a = torch.load(args.a, map_location="cpu", weights_only=True)
    b = torch.load(args.b, map_location="cpu", weights_only=True)
    bad = 0
    for k in sorted(a.keys()):
        if k not in b:
            print(f"MISSING in b: {k}")
            bad += 1
            continue
        ta, tb = a[k].float(), b[k].float()
        if ta.shape != tb.shape:
            print(f"SHAPE  {k}: {ta.shape} vs {tb.shape}")
            bad += 1
            continue
        diff = (ta - tb).abs()
        max_abs = diff.max().item()
        mean_abs = diff.mean().item()
        rel = (diff / (ta.abs() + 1e-30)).max().item()
        bit = torch.equal(a[k], b[k])
        tag = "BIT-EXACT" if bit else (
            "OK" if max_abs < 1e-3 else "FAIL"
        )
        print(f"  {k}: max_abs={max_abs:.3e} mean_abs={mean_abs:.3e} "
              f"max_rel={rel:.3e} -> {tag}")
        if tag == "FAIL":
            bad += 1
    print(f"\nsummary: {len(a)} cases, {bad} fail/mismatch")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
