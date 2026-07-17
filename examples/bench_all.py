"""全対応 op の forge vs PyTorch eager ベンチマーク。

GPU (cc7.0+) + venv で実行:
    .venv/bin/python examples/bench_all.py

出力例:
    op       | shape           | dtype  | eager_us | forge_us | speedup
    RMSNorm  | (2048, 4096)    | fp16   |   112.3  |    44.8  |  2.51x
    ...
"""

from __future__ import annotations

import statistics

import torch
import torch.nn.functional as F

import forge

SHAPES = [(2048, 4096), (1024, 8192)]
WARMUP = 25
REPEAT = 200


def measure_us(fn, warmup=WARMUP, repeat=REPEAT):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeat):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        samples.append(s.elapsed_time(e) * 1000.0)
    return statistics.median(samples)


def run_benchmarks():
    device = "cuda"
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"cc:  {torch.cuda.get_device_capability(0)}")
    print()

    header = f"{'op':<12} {'shape':<18} {'dtype':<8} {'eager_µs':>9} {'forge_µs':>9} {'speedup':>8}"
    print(header)
    print("-" * len(header))

    ops = [
        ("RMSNorm", "rmsnorm"),
        ("Softmax",  "softmax"),
        ("LayerNorm", "layernorm"),
        ("GELU",     "gelu"),
    ]

    for op_name, op_key in ops:
        for shape in SHAPES:
            for dtype in [torch.float16]:
                x = torch.randn(*shape, dtype=dtype, device=device)

                if op_key == "rmsnorm":
                    w = torch.ones(shape[-1], dtype=dtype, device=device)
                    eager_fn = lambda: F.rms_norm(x, (shape[-1],), w)  # noqa: E731

                    @forge.optimize(budget=50, progress=None)
                    def forge_fn(x, w):
                        return torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + 1e-6) * x * w

                    forged = lambda: forge_fn(x, w)  # noqa: E731

                elif op_key == "softmax":
                    eager_fn = lambda: torch.softmax(x, dim=-1)  # noqa: E731

                    @forge.optimize(budget=50, progress=None)
                    def forge_fn2(x):
                        return torch.softmax(x, dim=-1)

                    forged = lambda: forge_fn2(x)  # noqa: E731

                elif op_key == "layernorm":
                    w = torch.ones(shape[-1], dtype=dtype, device=device)
                    b = torch.zeros(shape[-1], dtype=dtype, device=device)
                    eager_fn = lambda: F.layer_norm(x, (shape[-1],), w, b)  # noqa: E731

                    @forge.optimize(budget=50, progress=None)
                    def forge_fn3(x, w, b):
                        return F.layer_norm(x, (x.shape[-1],), w, b)

                    forged = lambda: forge_fn3(x, w, b)  # noqa: E731

                else:  # gelu
                    eager_fn = lambda: F.gelu(x)  # noqa: E731

                    @forge.optimize(budget=50, progress=None)
                    def forge_fn4(x):
                        return F.gelu(x)

                    forged = lambda: forge_fn4(x)  # noqa: E731

                try:
                    # 初回: 探索実行
                    forged()
                    torch.cuda.synchronize()

                    eager_us = measure_us(eager_fn)
                    forge_us = measure_us(forged)
                    speedup = eager_us / forge_us

                    dtype_str = "fp16" if dtype == torch.float16 else "fp32"
                    print(
                        f"{op_name:<12} {str(shape):<18} {dtype_str:<8}"
                        f" {eager_us:>9.1f} {forge_us:>9.1f} {speedup:>7.2f}x"
                    )
                except Exception as e:
                    print(f"{op_name:<12} {str(shape):<18} fp16     ERROR: {e}")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available")
    else:
        run_benchmarks()
