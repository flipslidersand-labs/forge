"""RoPE (Rotary Position Embedding) を @forge.optimize で自動最適化するサンプル。

RoPE は Llama / Mistral 等で Q/K に適用される位置エンコード:
    rotate_half(x) = cat([-x[..., d//2:], x[..., :d//2]])
    apply_rope(x, cos, sin) = x*cos + rotate_half(x)*sin

forge は torch.fx でこのパターンを自動検出し Triton カーネルに置き換える。

GPU + venv で実行:
    .venv/bin/python examples/optimize_rope.py

オプション:
    --rows   入力行数  (default: 2048)
    --cols   列数/次元 (default: 4096, 偶数必須)
    --dtype  fp16 / bf16 / fp32 (default: fp16)
    --budget 探索予算  (default: 16)

DS1 RTX4060 実測例 (2048×4096, fp16):
    baseline x*cos + rotate_half(x)*sin : ~? us
    forge best                           : ~? us  (speedup ~?x)
"""

from __future__ import annotations

import argparse
import time

import torch

import forge

_DTYPE_MAP = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)


@forge.optimize(budget=16, progress=print)
def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """RoPE: forge が {getitem:3,mul:2,getattr:1,floordiv:1,neg:1,cat:1,add:1} を検出。"""
    return (x * cos) + (_rotate_half(x) * sin)


def _baseline(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    h = x.shape[-1] // 2
    rot = torch.cat([-x[..., h:], x[..., :h]], dim=-1)
    return x * cos + rot * sin


def main() -> None:
    parser = argparse.ArgumentParser(description="RoPE forge optimization demo")
    parser.add_argument("--rows", type=int, default=2048)
    parser.add_argument("--cols", type=int, default=4096)
    parser.add_argument("--dtype", choices=_DTYPE_MAP, default="fp16")
    parser.add_argument("--budget", type=int, default=16)
    args = parser.parse_args()

    assert args.cols % 2 == 0, "--cols must be even for RoPE"
    dtype = _DTYPE_MAP[args.dtype]
    device = "cuda"

    x   = torch.randn(args.rows, args.cols, dtype=dtype, device=device)
    cos = torch.randn(args.rows, args.cols, dtype=dtype, device=device)
    sin = torch.randn(args.rows, args.cols, dtype=dtype, device=device)

    print(f"\n=== RoPE {args.rows}×{args.cols} {args.dtype} ===")

    # baseline timing (eager PyTorch)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(50):
        _ = _baseline(x, cos, sin)
    torch.cuda.synchronize()
    baseline_us = (time.perf_counter() - t0) / 50 * 1e6
    print(f"baseline (x*cos + rotate_half(x)*sin): {baseline_us:.1f} us")

    print("\n=== first call: search + compile ===")
    t0 = time.perf_counter()
    out = apply_rope(x, cos, sin)
    torch.cuda.synchronize()
    print(f"first call total: {time.perf_counter() - t0:.1f}s")

    # correctness check
    ref = _baseline(x.float(), cos.float(), sin.float()).to(dtype)
    ok = torch.allclose(out.float(), ref.float(), atol=2e-3, rtol=1e-2)
    print(f"correct: {ok}")
    if not ok:
        max_diff = (out.float() - ref.float()).abs().max().item()
        print(f"  max_diff={max_diff:.4e}")

    print("\n=== second call: cached kernel ===")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(50):
        _ = apply_rope(x, cos, sin)
    torch.cuda.synchronize()
    forged_us = (time.perf_counter() - t0) / 50 * 1e6
    speedup = baseline_us / forged_us if forged_us > 0 else float("nan")
    print(f"forge best: {forged_us:.1f} us  (speedup {speedup:.3f}x vs baseline)")


if __name__ == "__main__":
    main()
