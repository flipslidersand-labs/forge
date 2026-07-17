"""全対応 op の forge vs PyTorch eager ベンチマーク。

GPU + venv で実行:
    .venv/bin/python examples/bench_all.py

出力: README に貼れる Markdown テーブル + 環境情報（stderr）
"""

from __future__ import annotations

import statistics
import sys

import torch
import torch.nn.functional as F

import forge

WARMUP = 25
REPEAT = 200
BUDGET = 50


def measure_us(fn, warmup: int = WARMUP, repeat: int = REPEAT) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(repeat):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        samples.append(s.elapsed_time(e) * 1_000)
    return statistics.median(samples)


# ── forge 最適化済み関数 ────────────────────────────────────────────────────────


@forge.optimize(budget=BUDGET, progress=None)
def _rmsnorm(x, weight, eps=1e-6):
    return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps) * weight


@forge.optimize(budget=BUDGET, progress=None)
def _softmax(x):
    return torch.softmax(x, dim=-1)


@forge.optimize(budget=BUDGET, progress=None)
def _layernorm(x, weight, bias, eps=1e-5):
    return F.layer_norm(x, (x.shape[-1],), weight, bias, eps)


@forge.optimize(budget=BUDGET, progress=None)
def _gelu(x):
    return F.gelu(x)


@forge.optimize(budget=BUDGET, progress=None)
def _sdpa(q, k, v):
    return F.scaled_dot_product_attention(q, k, v)


@forge.optimize(budget=BUDGET, progress=None)
def _sdpa_causal(q, k, v):
    return F.scaled_dot_product_attention(q, k, v, is_causal=True)


# ── ベンチケース定義 ────────────────────────────────────────────────────────────
# (表示名, shape, eager_fn, forge_fn, args_fn)

def _make_cases() -> list[tuple]:
    cases = []
    for m, n in [(2048, 4096), (1024, 8192)]:
        x = torch.randn(m, n, dtype=torch.float16, device="cuda")
        w = torch.ones(n, dtype=torch.float16, device="cuda")
        b = torch.zeros(n, dtype=torch.float16, device="cuda")

        cases.append((
            "RMSNorm", f"({m}, {n})",
            lambda x=x, w=w: F.rms_norm(x, (x.shape[-1],), w),
            lambda x=x, w=w: _rmsnorm(x, w),
        ))
        cases.append((
            "Softmax", f"({m}, {n})",
            lambda x=x: torch.softmax(x, dim=-1),
            lambda x=x: _softmax(x),
        ))
        cases.append((
            "LayerNorm", f"({m}, {n})",
            lambda x=x, w=w, b=b: F.layer_norm(x, (x.shape[-1],), w, b),
            lambda x=x, w=w, b=b: _layernorm(x, w, b),
        ))
        cases.append((
            "GELU", f"({m}, {n})",
            lambda x=x: F.gelu(x),
            lambda x=x: _gelu(x),
        ))

    # SDPA: [B*H, S, D] 形式
    for bh, s, d in [(8, 256, 64), (8, 512, 64)]:
        q = torch.randn(bh, s, d, dtype=torch.float16, device="cuda")
        k = torch.randn(bh, s, d, dtype=torch.float16, device="cuda")
        v = torch.randn(bh, s, d, dtype=torch.float16, device="cuda")
        cases.append((
            "SDPA", f"({bh}, {s}, {d})",
            lambda q=q, k=k, v=v: F.scaled_dot_product_attention(q, k, v),
            lambda q=q, k=k, v=v: _sdpa(q, k, v),
        ))
        cases.append((
            "SDPA causal", f"({bh}, {s}, {d})",
            lambda q=q, k=k, v=v: F.scaled_dot_product_attention(q, k, v, is_causal=True),
            lambda q=q, k=k, v=v: _sdpa_causal(q, k, v),
        ))

    return cases


def main() -> None:
    if not torch.cuda.is_available():
        print("CUDA not available", file=sys.stderr)
        return

    device_name = torch.cuda.get_device_name(0)
    cc = "{}.{}".format(*torch.cuda.get_device_capability(0))
    tv = torch.__version__.split("+")[0]
    try:
        import triton
        trv = triton.__version__
    except ImportError:
        trv = "n/a"

    print(f"GPU: {device_name} (cc{cc})", file=sys.stderr)
    print(f"PyTorch {tv} / Triton {trv}", file=sys.stderr)
    print(f"budget={BUDGET}, warmup={WARMUP}, repeat={REPEAT}\n", file=sys.stderr)

    cases = _make_cases()
    rows: list[tuple[str, str, float, float, float]] = []

    for op, shape, eager_fn, forge_fn in cases:
        print(f"  [{op} {shape}] warming up forge...", end=" ", file=sys.stderr, flush=True)
        forge_fn()  # 探索 + キャッシュ
        print("done", file=sys.stderr)

        eager_us = measure_us(eager_fn)
        forge_us = measure_us(forge_fn)
        speedup = eager_us / forge_us
        rows.append((op, shape, eager_us, forge_us, speedup))
        print(f"    eager={eager_us:.1f}µs  forge={forge_us:.1f}µs  {speedup:.2f}x",
              file=sys.stderr)

    # ── Markdown 出力（stdout） ───────────────────────────────────────────────────
    print(f"\n**測定環境**: {device_name} (cc{cc}), PyTorch {tv}, Triton {trv}。")
    print(f"budget={BUDGET}, warmup={WARMUP}, repeat={REPEAT} の中央値。")
    baselines = "`F.rms_norm` / `torch.softmax` / `F.layer_norm` / `F.gelu` / `F.scaled_dot_product_attention`"
    print(f"Baseline: {baselines}。")
    print()
    print("| op | shape | dtype | eager µs | forge µs | speedup |")
    print("| --- | --- | --- | ---: | ---: | ---: |")
    for op, shape, eager_us, forge_us, speedup in rows:
        marker = " ✅" if speedup >= 1.1 else (" ⚠️" if speedup < 0.95 else "")
        print(f"| {op} | {shape} | fp16 | {eager_us:.1f} | {forge_us:.1f} | {speedup:.2f}x{marker} |")


if __name__ == "__main__":
    main()
