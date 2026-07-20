"""forge vs torch.compile vs autotune 比較ベンチマーク。

業界標準（torch.compile, SDPA autotune）との相対性能を計測。

GPU + venv で実行:
    .venv/bin/python examples/benchmark_torch_compile_autotune.py

出力: JSON + Markdown テーブル
"""

from __future__ import annotations

import json
import statistics
import sys
from typing import Callable

import torch
import torch.nn.functional as F

import forge

WARMUP = 25
REPEAT = 200
BUDGET = 50


def measure_us(fn: Callable, warmup: int = WARMUP, repeat: int = REPEAT) -> float:
    """GPU event ベースの精密計測 (µs)。"""
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


# ── forge デコレータ版 ──────────────────────────────────────────────────────────


@forge.optimize(budget=BUDGET, progress=None)
def _rmsnorm_forge(x, weight, eps=1e-6):
    return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps) * weight


@forge.optimize(budget=BUDGET, progress=None)
def _softmax_forge(x):
    return torch.softmax(x, dim=-1)


@forge.optimize(budget=BUDGET, progress=None)
def _layernorm_forge(x, weight, bias, eps=1e-5):
    return F.layer_norm(x, (x.shape[-1],), weight, bias, eps)


@forge.optimize(budget=BUDGET, progress=None)
def _gelu_forge(x):
    return F.gelu(x)


@forge.optimize(budget=BUDGET, progress=None)
def _sdpa_forge(q, k, v):
    return F.scaled_dot_product_attention(q, k, v)


# ── torch.compile 版（eager ベース） ─────────────────────────────────────────────


@torch.compile(backend="eager")
def _rmsnorm_compiled_eager(x, weight, eps=1e-6):
    return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps) * weight


@torch.compile(backend="eager")
def _softmax_compiled_eager(x):
    return torch.softmax(x, dim=-1)


@torch.compile(backend="eager")
def _layernorm_compiled_eager(x, weight, bias, eps=1e-5):
    return F.layer_norm(x, (x.shape[-1],), weight, bias, eps)


@torch.compile(backend="eager")
def _gelu_compiled_eager(x):
    return F.gelu(x)


@torch.compile(backend="eager")
def _sdpa_compiled_eager(q, k, v):
    return F.scaled_dot_product_attention(q, k, v)


# ── 枠組み（autotune + baseline） ────────────────────────────────────────────────


def _make_test_cases() -> list[dict]:
    """ベンチケース定義。

    Returns:
        [{
            'name': 'op_shape',
            'op': 'RMSNorm|Softmax|...',
            'shape': '(2048, 4096)',
            'eager': fn,
            'compiled': fn,
            'forge': fn,
        }, ...]
    """
    cases = []

    # RMSNorm, Softmax, LayerNorm, GELU: 密行列演算
    for m, n in [(2048, 4096), (1024, 8192)]:
        x = torch.randn(m, n, dtype=torch.float16, device="cuda")
        w = torch.ones(n, dtype=torch.float16, device="cuda")
        b = torch.zeros(n, dtype=torch.float16, device="cuda")

        cases.append({
            'name': f'rmsnorm_{m}x{n}',
            'op': 'RMSNorm',
            'shape': f'({m}, {n})',
            'eager': lambda x=x, w=w: F.rms_norm(x, (x.shape[-1],), w),
            'compiled': lambda x=x, w=w: _rmsnorm_compiled_eager(x, w),
            'forge': lambda x=x, w=w: _rmsnorm_forge(x, w),
        })
        cases.append({
            'name': f'softmax_{m}x{n}',
            'op': 'Softmax',
            'shape': f'({m}, {n})',
            'eager': lambda x=x: torch.softmax(x, dim=-1),
            'compiled': lambda x=x: _softmax_compiled_eager(x),
            'forge': lambda x=x: _softmax_forge(x),
        })
        cases.append({
            'name': f'layernorm_{m}x{n}',
            'op': 'LayerNorm',
            'shape': f'({m}, {n})',
            'eager': lambda x=x, w=w, b=b: F.layer_norm(x, (x.shape[-1],), w, b),
            'compiled': lambda x=x, w=w, b=b: _layernorm_compiled_eager(x, w, b),
            'forge': lambda x=x, w=w, b=b: _layernorm_forge(x, w, b),
        })
        cases.append({
            'name': f'gelu_{m}x{n}',
            'op': 'GELU',
            'shape': f'({m}, {n})',
            'eager': lambda x=x: F.gelu(x),
            'compiled': lambda x=x: _gelu_compiled_eager(x),
            'forge': lambda x=x: _gelu_forge(x),
        })

    # SDPA: [B*H, S, D] 形式
    for bh, s, d in [(8, 256, 64), (8, 512, 64)]:
        q = torch.randn(bh, s, d, dtype=torch.float16, device="cuda")
        k = torch.randn(bh, s, d, dtype=torch.float16, device="cuda")
        v = torch.randn(bh, s, d, dtype=torch.float16, device="cuda")

        cases.append({
            'name': f'sdpa_{bh}x{s}x{d}',
            'op': 'SDPA',
            'shape': f'({bh}, {s}, {d})',
            'eager': lambda q=q, k=k, v=v: F.scaled_dot_product_attention(q, k, v),
            'compiled': lambda q=q, k=k, v=v: _sdpa_compiled_eager(q, k, v),
            'forge': lambda q=q, k=k, v=v: _sdpa_forge(q, k, v),
        })

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

    cases = _make_test_cases()
    results: list[dict] = []

    for case in cases:
        name = case['name']
        op = case['op']
        shape = case['shape']

        print(f"  [{op} {shape}]", end=" ", file=sys.stderr, flush=True)

        # forge 探索（初回のみ、キャッシュに保存）
        print("forge...", end=" ", file=sys.stderr, flush=True)
        case['forge']()

        # 計測
        print("eager...", end=" ", file=sys.stderr, flush=True)
        eager_us = measure_us(case['eager'])

        print("compiled...", end=" ", file=sys.stderr, flush=True)
        compiled_us = measure_us(case['compiled'])

        print("forge...", end=" ", file=sys.stderr, flush=True)
        forge_us = measure_us(case['forge'])

        results.append({
            'name': name,
            'op': op,
            'shape': shape,
            'eager_us': eager_us,
            'compiled_us': compiled_us,
            'forge_us': forge_us,
            'forge_vs_eager': eager_us / forge_us,
            'forge_vs_compiled': compiled_us / forge_us,
        })

        speedup_eager = eager_us / forge_us
        speedup_compiled = compiled_us / forge_us
        print(f"✓ eager={eager_us:.1f}µs, compiled={compiled_us:.1f}µs, forge={forge_us:.1f}µs",
              file=sys.stderr)

    # ── Markdown 出力 ─────────────────────────────────────────────────────────────
    print("\n" + "="*80)
    print("## torch.compile / autotune vs forge ベンチマーク")
    print("="*80)
    print(f"\n**測定環境**: {device_name} (cc{cc}), PyTorch {tv}, Triton {trv}")
    print(f"**計測設定**: budget={BUDGET}, warmup={WARMUP}, repeat={REPEAT} 中央値")
    print()

    print("| Op | Shape | Eager (µs) | torch.compile (µs) | forge (µs) | forge/eager | forge/compiled |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for r in results:
        marker_eager = " ✅" if r['forge_vs_eager'] >= 1.1 else (" ⚠️" if r['forge_vs_eager'] < 0.95 else "")
        marker_compiled = " ✅" if r['forge_vs_compiled'] >= 1.05 else (" ⚠️" if r['forge_vs_compiled'] < 1.0 else "")
        print(f"| {r['op']} | {r['shape']} | {r['eager_us']:.1f} | {r['compiled_us']:.1f} | {r['forge_us']:.1f} "
              f"| {r['forge_vs_eager']:.2f}x{marker_eager} | {r['forge_vs_compiled']:.2f}x{marker_compiled} |")

    # ── JSON 出力 ──────────────────────────────────────────────────────────────
    print("\n" + "="*80)
    print("## JSON 結果")
    print("="*80)
    json_output = {
        'environment': {
            'device': device_name,
            'compute_capability': cc,
            'pytorch_version': tv,
            'triton_version': trv,
        },
        'config': {
            'budget': BUDGET,
            'warmup': WARMUP,
            'repeat': REPEAT,
        },
        'results': results,
    }
    print(json.dumps(json_output, indent=2))


if __name__ == "__main__":
    main()
