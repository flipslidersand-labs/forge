"""全対応 op のベンチマーク計測スクリプト。

GPU + venv で実行:
    .venv/bin/python examples/benchmark_all.py
"""

from __future__ import annotations

import torch

from forge.codegen.triton_codegen import graph_hash_for
from forge.ir.kernel_spec import KernelSpec
from forge.ir.tensor_spec import TensorSpec
from forge.orchestrator import Orchestrator


def _orch() -> Orchestrator:
    return Orchestrator(warmup=25, repeat=200)


def bench_rmsnorm(rows: int = 2048, hidden: int = 4096) -> None:
    spec = KernelSpec(
        op_type="rmsnorm",
        input_specs=(
            TensorSpec((rows, hidden), torch.float16, True),
            TensorSpec((hidden,), torch.float16, True),
        ),
        output_specs=(TensorSpec((rows, hidden), torch.float16, True),),
        constants={"eps": 1e-6},
        graph_hash=graph_hash_for("rmsnorm"),
        constraints=(),
    )
    result = _orch().optimize(spec, budget=50, use_cache=False)
    _print("RMSNorm", rows, hidden, result)


def bench_softmax(rows: int = 2048, hidden: int = 4096) -> None:
    spec = KernelSpec(
        op_type="softmax",
        input_specs=(TensorSpec((rows, hidden), torch.float16, True),),
        output_specs=(TensorSpec((rows, hidden), torch.float16, True),),
        constants={},
        graph_hash=graph_hash_for("softmax"),
        constraints=(),
    )
    result = _orch().optimize(spec, budget=50, use_cache=False)
    _print("Softmax", rows, hidden, result)


def bench_layernorm(rows: int = 2048, hidden: int = 4096) -> None:
    spec = KernelSpec(
        op_type="layernorm",
        input_specs=(
            TensorSpec((rows, hidden), torch.float16, True),
            TensorSpec((hidden,), torch.float16, True),
            TensorSpec((hidden,), torch.float16, True),
        ),
        output_specs=(TensorSpec((rows, hidden), torch.float16, True),),
        constants={"eps": 1e-5},
        graph_hash=graph_hash_for("layernorm"),
        constraints=(),
    )
    result = _orch().optimize(spec, budget=50, use_cache=False)
    _print("LayerNorm", rows, hidden, result)


def bench_gelu(rows: int = 2048, hidden: int = 4096) -> None:
    spec = KernelSpec(
        op_type="gelu",
        input_specs=(TensorSpec((rows, hidden), torch.float16, True),),
        output_specs=(TensorSpec((rows, hidden), torch.float16, True),),
        constants={},
        graph_hash=graph_hash_for("gelu"),
        constraints=(),
    )
    result = _orch().optimize(spec, budget=50, use_cache=False)
    _print("GELU", rows, hidden, result)


def _print(name: str, rows: int, hidden: int, result) -> None:  # type: ignore[type-arg]
    shape = f"{rows}x{hidden}"
    if result.best_params is None:
        print(f"| {name:<12} | {shape:<10} | — | — | — |")
        return
    bl = result.baseline_benchmark
    best = result.best_benchmark
    if bl and best:
        print(
            f"| {name:<12} | {shape:<10} | {bl.median_us:>10.1f} | {best.median_us:>8.1f} | {result.speedup:>7.2f}x |"
        )
    else:
        print(f"| {name:<12} | {shape:<10} | — | — | — |")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available")
        raise SystemExit(1)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()
    print(f"| {'Op':<12} | {'Shape':<10} | {'Baseline (µs)':>13} | {'Forge (µs)':>10} | {'Speedup':>8} |")
    print(f"|{'-'*14}|{'-'*12}|{'-'*15}|{'-'*12}|{'-'*10}|")
    bench_rmsnorm()
    bench_softmax()
    bench_layernorm()
    bench_gelu()
