"""Orchestrator のオフライン単体テスト。GPU 不要。"""

import torch

from forge.benchmark.statistics import BenchmarkResult
from forge.ir.kernel_spec import KernelSpec
from forge.ir.tensor_spec import TensorSpec
from forge.orchestrator import ExperimentResult, MultiRoundResult, RoundResult
from forge.runtime.worker import ExtendedBaselineResult
from forge.search.llm_generator import TokenUsage
from forge.search.params import SearchParams


def _spec() -> KernelSpec:
    return KernelSpec(
        op_type="rmsnorm",
        input_specs=(
            TensorSpec((512, 4096), torch.float16, True),
            TensorSpec((4096,), torch.float16, True),
        ),
        output_specs=(TensorSpec((512, 4096), torch.float16, True),),
        constants={"eps": 1e-6},
        graph_hash="rmsnorm_v1",
        constraints=(),
    )


def _make_params(**kw) -> SearchParams:
    base = dict(block_size=4096, num_warps=8, num_stages=1, acc_dtype="fp32")
    base.update(kw)
    return SearchParams(**base)


class TestMultiRoundResult:
    def test_speedup_none_without_benchmarks(self) -> None:
        r = MultiRoundResult(
            spec=_spec(),
            rounds=[],
            best_params=None,
            best_benchmark=None,
            baseline_benchmark=None,
            baseline_name=None,
            token_usage=TokenUsage(),
        )
        assert r.speedup is None

    def test_all_experiments_flattens_rounds(self) -> None:
        p1 = _make_params(num_warps=4)
        p2 = _make_params(num_warps=8)
        exp1 = ExperimentResult(p1, True, True, 50.0, None)
        exp2 = ExperimentResult(p2, True, True, 45.0, None)
        r = MultiRoundResult(
            spec=_spec(),
            rounds=[
                RoundResult(1, [exp1], p1, 50.0),
                RoundResult(2, [exp2], p2, 45.0),
            ],
            best_params=p2,
            best_benchmark=None,
            baseline_benchmark=None,
            baseline_name=None,
            token_usage=TokenUsage(),
            total_candidates_evaluated=2,
        )
        assert len(r.all_experiments) == 2

    def test_best_round_returns_correct_round(self) -> None:
        p1 = _make_params(num_warps=4)
        p2 = _make_params(num_warps=8)
        r = MultiRoundResult(
            spec=_spec(),
            rounds=[
                RoundResult(1, [], p1, 50.0),
                RoundResult(2, [], p2, 45.0),
            ],
            best_params=p2,
            best_benchmark=None,
            baseline_benchmark=None,
            baseline_name=None,
            token_usage=TokenUsage(),
        )
        assert r.best_round == 2

    def test_extended_baselines_defaults_empty(self) -> None:
        r = MultiRoundResult(
            spec=_spec(),
            rounds=[],
            best_params=None,
            best_benchmark=None,
            baseline_benchmark=None,
            baseline_name=None,
            token_usage=TokenUsage(),
        )
        assert r.extended_baselines == []


class TestExtendedBaselineResult:
    def _make(self) -> ExtendedBaselineResult:
        bench = BenchmarkResult.from_samples([10.0, 12.0, 11.0, 9.0, 13.0], warmup=2, repeat=5)
        return ExtendedBaselineResult(
            name="torch.compile(reference)",
            benchmark=bench,
            compile_time_s=3.2,
        )

    def test_dict_roundtrip(self) -> None:
        r = self._make()
        r2 = ExtendedBaselineResult.from_dict(r.to_dict())
        assert r2.name == r.name
        assert abs(r2.benchmark.median_us - r.benchmark.median_us) < 1e-6
        assert abs(r2.compile_time_s - r.compile_time_s) < 1e-6

    def test_p95_preserved(self) -> None:
        r = self._make()
        d = r.to_dict()
        assert "p95_us" in d["benchmark"]
        r2 = ExtendedBaselineResult.from_dict(d)
        assert r2.benchmark.p95_us > 0

    def test_compile_time_defaults_to_zero(self) -> None:
        bench = BenchmarkResult.from_samples([10.0], warmup=0, repeat=1)
        r = ExtendedBaselineResult.from_dict(
            {"name": "x", "benchmark": bench.to_dict()}
        )
        assert r.compile_time_s == 0.0
