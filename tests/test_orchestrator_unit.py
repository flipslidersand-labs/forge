"""Orchestrator のオフライン単体テスト。GPU 不要。"""

import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch

import torch

from forge.benchmark.statistics import BenchmarkResult
from forge.cache.repository import KernelRepository
from forge.ir.kernel_spec import KernelSpec
from forge.ir.tensor_spec import TensorSpec
from forge.orchestrator import (
    ExperimentResult,
    MultiRoundResult,
    Orchestrator,
    RoundResult,
    SearchResult,
)
from forge.runtime.worker import ExtendedBaselineResult, WorkerResult
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
        r = ExtendedBaselineResult.from_dict({"name": "x", "benchmark": bench.to_dict()})
        assert r.compile_time_s == 0.0

    def test_error_field_roundtrip(self) -> None:
        bench = BenchmarkResult.from_samples([0.0], warmup=0, repeat=1)
        r = ExtendedBaselineResult(
            name="torch.compile(reference)",
            benchmark=bench,
            compile_time_s=0.0,
            error="RuntimeError: CUDA not available",
        )
        d = r.to_dict()
        assert d["error"] == "RuntimeError: CUDA not available"
        r2 = ExtendedBaselineResult.from_dict(d)
        assert r2.error == "RuntimeError: CUDA not available"
        assert r2.failed is True

    def test_no_error_field_not_in_dict(self) -> None:
        r = self._make()
        d = r.to_dict()
        assert "error" not in d

    def test_failed_property_true_when_error(self) -> None:
        bench = BenchmarkResult.from_samples([0.0], warmup=0, repeat=1)
        r = ExtendedBaselineResult(name="x", benchmark=bench, error="something went wrong")
        assert r.failed is True

    def test_failed_property_false_when_success(self) -> None:
        r = self._make()
        assert r.failed is False

    def test_from_dict_without_error_field_is_success(self) -> None:
        bench = BenchmarkResult.from_samples([10.0], warmup=0, repeat=1)
        r = ExtendedBaselineResult.from_dict({"name": "x", "benchmark": bench.to_dict()})
        assert r.error is None
        assert r.failed is False


class TestOrchestratorLifecycle:
    def test_owns_repo_when_none_passed(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            orch = Orchestrator(repo=KernelRepository(Path(d) / "a.db"))
            assert not orch._owns_repo

    def test_does_not_own_external_repo(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            repo = KernelRepository(Path(d) / "ext.db")
            orch = Orchestrator(repo=repo)
            assert not orch._owns_repo
            orch.close()
            # 外部 repo は close されていない（まだ使える）
            repo.close()

    def test_context_manager_closes_owned_repo(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            owned_repo = KernelRepository(Path(d) / "owned.db")
            # _owns_repo を強制的に True にして close() が呼ばれることを確認
            with Orchestrator(repo=owned_repo) as orch:
                orch._owns_repo = True
                repo_ref = orch.repo
            try:
                repo_ref.conn.execute("SELECT 1")
                raise AssertionError("should have raised")
            except sqlite3.ProgrammingError:
                pass

    def test_context_manager_does_not_close_external_repo(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            repo = KernelRepository(Path(d) / "ext.db")
            with Orchestrator(repo=repo) as _:
                pass
            # 外部 repo は close されていない（まだ使える）
            repo.conn.execute("SELECT 1")
            repo.close()


def _bench() -> BenchmarkResult:
    return BenchmarkResult(median_us=10.0, p20_us=9.0, p80_us=11.0)


def _eval_args(orch: Orchestrator) -> tuple:
    spec = _spec()
    params = _make_params()
    bench_input = [{"shape": [512, 4096], "dtype": "float16", "init": "randn", "seed": 0}]
    return (spec, params, bench_input, None, {"atol": 2e-3, "rtol": 1e-2}, "test")


class TestEvalOneFailureBranches:
    """Orchestrator._eval_one の3つの失敗ブランチを GPU なしで検証。"""

    def test_codegen_value_error_returns_failure(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            orch = Orchestrator(repo=KernelRepository(Path(d) / "c.db"))
            with patch("forge.orchestrator.generate", side_effect=ValueError("block too small")):
                exp, cand, bl, bl_name = orch._eval_one(*_eval_args(orch))

        assert exp.success is False
        assert exp.correct is False
        assert exp.error == "block too small"
        assert cand is None
        assert bl is None
        assert bl_name is None

    def test_worker_failure_returns_failure(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            orch = Orchestrator(repo=KernelRepository(Path(d) / "c.db"))
            with patch("forge.orchestrator.generate", return_value="def k(): pass"):
                with patch(
                    "forge.orchestrator.run_in_worker",
                    return_value=WorkerResult(success=False, error="CUDA crash"),
                ):
                    exp, cand, bl, bl_name = orch._eval_one(*_eval_args(orch))

        assert exp.success is False
        assert exp.correct is False
        assert exp.error == "CUDA crash"
        assert cand is None

    def test_worker_incorrect_returns_incorrect(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            orch = Orchestrator(repo=KernelRepository(Path(d) / "c.db"))
            with patch("forge.orchestrator.generate", return_value="def k(): pass"):
                with patch(
                    "forge.orchestrator.run_in_worker",
                    return_value=WorkerResult(success=True, correct=False),
                ):
                    exp, cand, bl, bl_name = orch._eval_one(*_eval_args(orch))

        assert exp.success is True
        assert exp.correct is False
        assert cand is None

    def test_worker_success_returns_benchmarks(self) -> None:
        bench = _bench()
        with tempfile.TemporaryDirectory() as d:
            orch = Orchestrator(repo=KernelRepository(Path(d) / "c.db"))
            with patch("forge.orchestrator.generate", return_value="def k(): pass"):
                with patch(
                    "forge.orchestrator.run_in_worker",
                    return_value=WorkerResult(
                        success=True,
                        correct=True,
                        candidate=bench,
                        baseline=bench,
                        baseline_name="ref",
                    ),
                ):
                    exp, cand, bl, bl_name = orch._eval_one(*_eval_args(orch))

        assert exp.success is True
        assert exp.correct is True
        assert cand is bench
        assert bl is bench
        assert bl_name == "ref"


class TestSearchResultMetrics:
    """SearchResult の新メトリクスフィールドのユニットテスト。"""

    def _make_search_result(self, experiments):
        import torch

        from forge.ir.kernel_spec import KernelSpec
        from forge.ir.tensor_spec import TensorSpec

        spec = KernelSpec(
            op_type="rmsnorm",
            input_specs=(
                TensorSpec((16, 64), torch.float16, True),
                TensorSpec((64,), torch.float16, True),
            ),
            output_specs=(TensorSpec((16, 64), torch.float16, True),),
            constants={"eps": 1e-6},
            graph_hash="h",
            constraints=(),
        )
        return SearchResult(
            spec=spec,
            cache_hit=False,
            best_params=None,
            best_benchmark=None,
            baseline_benchmark=None,
            baseline_name=None,
            experiments=experiments,
        )

    def _exp(self, success, correct, error=None):
        params = SearchParams(block_size=64, num_warps=4, num_stages=1)
        return ExperimentResult(
            params=params, success=success, correct=correct, median_us=None, error=error
        )

    def test_default_values(self):
        r = self._make_search_result([])
        assert r.total_time_s == 0.0
        assert r.failed_count == 0
        assert r.incorrect_count == 0

    def test_failed_count_counts_unsuccessful(self):
        exps = [
            self._exp(True, True),
            self._exp(False, False, "crash"),
            self._exp(False, False, "timeout"),
        ]
        r = self._make_search_result(exps)
        r = SearchResult(
            spec=r.spec,
            cache_hit=False,
            best_params=None,
            best_benchmark=None,
            baseline_benchmark=None,
            baseline_name=None,
            experiments=exps,
            failed_count=sum(1 for e in exps if not e.success),
            incorrect_count=sum(1 for e in exps if e.success and not e.correct),
        )
        assert r.failed_count == 2
        assert r.incorrect_count == 0

    def test_incorrect_count_counts_correct_false_success_true(self):
        exps = [
            self._exp(True, True),
            self._exp(True, False),
            self._exp(True, False),
        ]
        r = SearchResult(
            spec=self._make_search_result([]).spec,
            cache_hit=False,
            best_params=None,
            best_benchmark=None,
            baseline_benchmark=None,
            baseline_name=None,
            experiments=exps,
            failed_count=sum(1 for e in exps if not e.success),
            incorrect_count=sum(1 for e in exps if e.success and not e.correct),
        )
        assert r.failed_count == 0
        assert r.incorrect_count == 2


class TestMultiRoundResultMetrics:
    """MultiRoundResult の新メトリクスフィールドのユニットテスト。"""

    def test_default_values(self):
        import torch

        from forge.ir.kernel_spec import KernelSpec
        from forge.ir.tensor_spec import TensorSpec

        spec = KernelSpec(
            op_type="rmsnorm",
            input_specs=(
                TensorSpec((16, 64), torch.float16, True),
                TensorSpec((64,), torch.float16, True),
            ),
            output_specs=(TensorSpec((16, 64), torch.float16, True),),
            constants={"eps": 1e-6},
            graph_hash="h",
            constraints=(),
        )
        r = MultiRoundResult(
            spec=spec,
            rounds=[],
            best_params=None,
            best_benchmark=None,
            baseline_benchmark=None,
            baseline_name=None,
            token_usage=None,
        )
        assert r.failed_count == 0
        assert r.incorrect_count == 0


class TestProgressEvent:
    """ProgressEvent dataclass とコールバック dispatch のユニットテスト。"""

    def test_event_fields(self):
        from forge.orchestrator import ProgressEvent

        e = ProgressEvent(kind="candidate_ok", label="ok", params=None, median_us=50.0, speedup=2.0)
        assert e.kind == "candidate_ok"
        assert e.label == "ok"
        assert e.median_us == 50.0
        assert e.speedup == 2.0

    def test_typed_callback_receives_event(self):
        """progress=Callable[[ProgressEvent], None] はイベントをそのまま受け取る。"""
        from forge.orchestrator import Orchestrator, ProgressEvent

        events: list[ProgressEvent] = []
        orch = Orchestrator(progress=lambda e: events.append(e))
        orch._emit(ProgressEvent(kind="info", label="hello"))
        assert len(events) == 1
        assert events[0].kind == "info"

    def test_string_callback_receives_label(self):
        """progress=Callable[[str], None] は label を受け取る（後方互換）。"""
        from forge.orchestrator import Orchestrator, ProgressEvent

        msgs: list[str] = []

        def on_str(msg: str) -> None:
            msgs.append(msg)

        import warnings

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            orch = Orchestrator(progress=on_str)
        orch._emit(ProgressEvent(kind="info", label="hello"))
        assert msgs == ["hello"]

    def test_none_progress_is_noop(self):
        """progress=None でもエラーにならない。"""
        from forge.orchestrator import Orchestrator, ProgressEvent

        orch = Orchestrator(progress=None)
        orch._emit(ProgressEvent(kind="info", label="x"))  # no exception

    def test_worker_fail_emits_candidate_fail_event(self):
        """_eval_one でワーカー失敗 → candidate_fail イベント。"""
        import tempfile

        from forge.orchestrator import Orchestrator, ProgressEvent

        events: list[ProgressEvent] = []
        with tempfile.TemporaryDirectory() as d:
            orch = Orchestrator(
                repo=KernelRepository(Path(d) / "c.db"),
                progress=lambda e: events.append(e),
            )
            with patch("forge.orchestrator.generate", return_value="def k(): pass"):
                with patch(
                    "forge.orchestrator.run_in_worker",
                    return_value=WorkerResult(success=False, error="crash"),
                ):
                    orch._eval_one(*_eval_args(orch))

        fail_events = [e for e in events if e.kind == "candidate_fail"]
        assert len(fail_events) == 1
        assert "crash" in fail_events[0].label

    def test_worker_success_emits_candidate_ok_event(self):
        """_eval_one でワーカー成功 → candidate_ok イベント。"""
        import tempfile

        from forge.orchestrator import Orchestrator, ProgressEvent

        events: list[ProgressEvent] = []
        bench = _bench()
        with tempfile.TemporaryDirectory() as d:
            orch = Orchestrator(
                repo=KernelRepository(Path(d) / "c.db"),
                progress=lambda e: events.append(e),
            )
            with patch("forge.orchestrator.generate", return_value="def k(): pass"):
                with patch(
                    "forge.orchestrator.run_in_worker",
                    return_value=WorkerResult(
                        success=True,
                        correct=True,
                        candidate=bench,
                        baseline=bench,
                        baseline_name="ref",
                    ),
                ):
                    orch._eval_one(*_eval_args(orch))

        ok_events = [e for e in events if e.kind == "candidate_ok"]
        assert len(ok_events) == 1
        assert ok_events[0].median_us == bench.median_us
