"""コスト考慮型探索の使用例。

CostModel と BudgetTracker を組み合わせて、予算内で最良の候補を探す。
"""

from __future__ import annotations

import random
from pathlib import Path

from forge.search.cost_model import BudgetTracker, CostModel, scalarize
from forge.search.params import SearchParams

CANDIDATES = [
    SearchParams(block_size=bs, num_warps=nw, num_stages=ns)
    for bs in [64, 128, 256, 512]
    for nw in [2, 4, 8]
    for ns in [2, 3]
]


def mock_benchmark(params: SearchParams) -> tuple[float, float]:
    """実際のベンチマーク代わりの疑似計測。(speedup, bench_ms) を返す。"""
    speedup = 1.0 + (params.block_size / 256) * 0.5 + (params.num_warps / 4) * 0.2
    bench_ms = params.block_size * params.num_warps * 0.5
    return speedup, bench_ms


def run(budget_s: float = 10.0, lam: float = 0.1) -> None:
    with CostModel(db_path=Path("/tmp/forge_demo_cost.db")) as model:
        tracker = BudgetTracker(max_total_s=budget_s)
        best_score = float("-inf")
        best_params = None
        evaluated = 0

        for params in CANDIDATES:
            est = model.predict_cost(params)
            if tracker.should_skip(est.estimated_bench_ms):
                print(f"  skip {params.block_size}x{params.num_warps} (est {est.estimated_bench_ms:.0f}ms, budget exhausted)")
                continue

            speedup, bench_ms = mock_benchmark(params)
            model.record(params, bench_ms)
            tracker.record(bench_ms)
            evaluated += 1

            score = scalarize(speedup=speedup, cost=bench_ms / 1000.0, lam=lam)
            if score > best_score:
                best_score = score
                best_params = params
                print(f"  new best: {params.block_size}x{params.num_warps}x{params.num_stages} score={score:.3f}")

        print(f"\nevaluated={evaluated}/{len(CANDIDATES)}, best={best_params}, score={best_score:.3f}")


if __name__ == "__main__":
    run(budget_s=5.0, lam=0.05)
