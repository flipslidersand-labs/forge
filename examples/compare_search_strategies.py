"""GridSearch / BayesianGenerator / SHA の性能比較スクリプト。

rmsnorm (512×4096, fp16) で 3 戦略を同一 budget で比較し、
best_median_us / n_evaluated / wall_time_s / speedup_vs_baseline を報告する。

GPU 環境で実行:
    .venv/bin/python examples/compare_search_strategies.py

optuna 未インストール時は Bayesian をスキップ:
    pip install "forge-kernel[search]"
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import torch

from forge.ir.kernel_spec import KernelSpec
from forge.ir.tensor_spec import TensorSpec
from forge.orchestrator import Orchestrator, SearchResult
from forge.search.grid import GridSearch


def _build_spec(rows: int, hidden: int) -> KernelSpec:
    return KernelSpec(
        op_type="rmsnorm",
        input_specs=(
            TensorSpec((rows, hidden), torch.float16, True),
            TensorSpec((hidden,), torch.float16, True),
        ),
        output_specs=(TensorSpec((rows, hidden), torch.float16, True),),
        constants={"eps": 1e-6},
        graph_hash=f"rmsnorm_compare_{rows}x{hidden}",
        constraints=(),
    )


@dataclass
class StrategyReport:
    name: str
    best_median_us: float | None
    n_evaluated: int
    wall_time_s: float
    speedup_vs_baseline: float | None
    baseline_median_us: float | None


def _report(name: str, result: SearchResult, wall_time_s: float) -> StrategyReport:
    return StrategyReport(
        name=name,
        best_median_us=result.best_benchmark.median_us if result.best_benchmark else None,
        n_evaluated=len(result.experiments),
        wall_time_s=wall_time_s,
        speedup_vs_baseline=result.speedup,
        baseline_median_us=(
            result.baseline_benchmark.median_us if result.baseline_benchmark else None
        ),
    )


def _print_table(reports: list[StrategyReport]) -> None:
    print()
    print(f"{'Strategy':<20} {'best_us':>10} {'n_eval':>8} {'time_s':>8} {'speedup':>10}")
    print("-" * 62)
    for r in reports:
        best = f"{r.best_median_us:.1f}" if r.best_median_us is not None else "N/A"
        speedup = f"{r.speedup_vs_baseline:.3f}x" if r.speedup_vs_baseline is not None else "N/A"
        print(f"{r.name:<20} {best:>10} {r.n_evaluated:>8} {r.wall_time_s:>8.1f} {speedup:>10}")
    print()

    # baseline 行があれば表示
    for r in reports:
        if r.baseline_median_us is not None:
            print(f"  baseline (PyTorch): {r.baseline_median_us:.1f}us  (from {r.name})")
            break


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare search strategies for rmsnorm.")
    ap.add_argument("--rows", type=int, default=512)
    ap.add_argument("--hidden", type=int, default=4096)
    ap.add_argument(
        "--budget",
        type=int,
        default=32,
        help="Number of candidates for GridSearch and BayesianGenerator",
    )
    ap.add_argument(
        "--sha-budget",
        type=int,
        default=32,
        help="initial_budget for SHA (same as --budget by default)",
    )
    ap.add_argument("--sha-rounds", type=int, default=3)
    ap.add_argument("--skip-bayesian", action="store_true")
    ap.add_argument("--skip-sha", action="store_true")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA is required. Run on a GPU machine.")
        return

    spec = _build_spec(args.rows, args.hidden)
    sha_budget = args.sha_budget if args.sha_budget != 32 else args.budget

    print(f"Comparing strategies — rmsnorm {args.rows}x{args.hidden} fp16")
    print(f"  budget={args.budget}  sha_budget={sha_budget}  sha_rounds={args.sha_rounds}")
    print(f"  device: {torch.cuda.get_device_name(0)}")
    print()

    reports: list[StrategyReport] = []

    # --- GridSearch ---
    print("=== GridSearch ===")
    orch = Orchestrator(progress=print, warmup=25, repeat=200)
    t0 = time.time()
    grid_result = orch.optimize(spec, budget=args.budget, search=GridSearch(), use_cache=False)
    reports.append(_report("GridSearch", grid_result, time.time() - t0))

    # --- BayesianGenerator ---
    if not args.skip_bayesian:
        try:
            from forge.search.bayesian_generator import BayesianGenerator

            print("\n=== BayesianGenerator ===")
            orch2 = Orchestrator(progress=print, warmup=25, repeat=200)
            t0 = time.time()
            bay_result = orch2.optimize(
                spec, budget=args.budget, search=BayesianGenerator(seed=42), use_cache=False
            )
            reports.append(_report("BayesianGenerator", bay_result, time.time() - t0))
        except ImportError:
            print("BayesianGenerator: skipped (optuna not installed)")
            print("  Install: pip install 'forge-kernel[search]'")

    # --- Successive Halving ---
    if not args.skip_sha:
        print("\n=== Successive Halving (SHA) ===")
        orch3 = Orchestrator(progress=print)
        t0 = time.time()
        sha_result = orch3.optimize_sha(
            spec,
            initial_budget=sha_budget,
            halving_rounds=args.sha_rounds,
            search=GridSearch(),
            use_cache=False,
        )
        reports.append(_report(f"SHA({args.sha_rounds}r)", sha_result, time.time() - t0))

    # --- Summary ---
    print("\n" + "=" * 62)
    print("SUMMARY")
    _print_table(reports)

    if len(reports) >= 2:
        baseline_us = reports[0].best_median_us
        if baseline_us is not None:
            print("  Relative to GridSearch best:")
            for r in reports[1:]:
                if r.best_median_us is not None:
                    ratio = baseline_us / r.best_median_us
                    saved = (1 - r.n_evaluated / max(reports[0].n_evaluated, 1)) * 100
                    time_ratio = r.wall_time_s / reports[0].wall_time_s
                    print(
                        f"    {r.name}: latency_ratio={ratio:.3f}x  "
                        f"budget_saving={saved:.0f}%  time_ratio={time_ratio:.2f}x"
                    )


if __name__ == "__main__":
    main()
