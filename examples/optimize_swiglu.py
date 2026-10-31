"""SwiGLU (SiLU-gated linear unit) を @forge.optimize で自動最適化するサンプル。

SwiGLU は Llama / PaLM 等の FFN で使われる活性化関数:
    output = F.silu(gate) * x

forge は torch.fx でこのパターンを自動検出し Triton カーネルに置き換える。

GPU + venv で実行:
    .venv/bin/python examples/optimize_swiglu.py

オプション:
    --rows   入力行数  (default: 2048)
    --cols   列数      (default: 4096)
    --dtype  fp16 / bf16 / fp32 (default: fp16)
    --budget 探索予算  (default: 16)

DS1 RTX4060 実測例 (2048×4096, fp16):
    baseline F.silu(gate) * x : ~22.0 us
    forge best                : ~21.3 us  (speedup ~1.03x)
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F

import forge

_DTYPE_MAP = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


@forge.optimize(budget=16, progress=print)
def swiglu(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """SwiGLU: forge が {silu:1, mul:1} パターンを検出して Triton に置き換える。"""
    return F.silu(gate) * x


def main() -> None:
    parser = argparse.ArgumentParser(description="SwiGLU forge optimization demo")
    parser.add_argument("--rows", type=int, default=2048)
    parser.add_argument("--cols", type=int, default=4096)
    parser.add_argument("--dtype", choices=_DTYPE_MAP, default="fp16")
    parser.add_argument("--budget", type=int, default=16)
    args = parser.parse_args()

    dtype = _DTYPE_MAP[args.dtype]
    device = "cuda"

    x = torch.randn(args.rows, args.cols, dtype=dtype, device=device)
    gate = torch.randn(args.rows, args.cols, dtype=dtype, device=device)

    print(f"\n=== SwiGLU {args.rows}×{args.cols} {args.dtype} ===")

    # baseline timing (eager PyTorch)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(50):
        _ = F.silu(gate) * x
    torch.cuda.synchronize()
    baseline_us = (time.perf_counter() - t0) / 50 * 1e6
    print(f"baseline (F.silu(gate)*x): {baseline_us:.1f} us")

    print("\n=== first call: search + compile ===")
    t0 = time.perf_counter()
    out = swiglu(x, gate)
    torch.cuda.synchronize()
    print(f"first call total: {time.perf_counter() - t0:.1f}s")

    # correctness check
    ref = F.silu(gate.float()) * x.float()
    ok = torch.allclose(out.float(), ref, atol=2e-3, rtol=1e-2)
    print(f"correct: {ok}")
    if not ok:
        max_diff = (out.float() - ref).abs().max().item()
        print(f"  max_diff={max_diff:.4e}")

    print("\n=== second call: cached kernel ===")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(50):
        _ = swiglu(x, gate)
    torch.cuda.synchronize()
    forged_us = (time.perf_counter() - t0) / 50 * 1e6
    speedup = baseline_us / forged_us if forged_us > 0 else float("nan")
    print(f"forge best: {forged_us:.1f} us  (speedup {speedup:.3f}x vs baseline)")


if __name__ == "__main__":
    main()
