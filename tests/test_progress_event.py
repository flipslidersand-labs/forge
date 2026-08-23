"""ProgressEvent 構造化イベントのテスト（#190）。GPU 不要。"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from unittest.mock import MagicMock, patch

import pytest
import torch

from forge.benchmark.statistics import BenchmarkResult
from forge.cache.repository import KernelRepository
from forge.ir.kernel_spec import KernelSpec
from forge.ir.tensor_spec import TensorSpec
from forge.orchestrator import Orchestrator
from forge.progress import ProgressEvent, make_progress_callback
from forge.runtime.worker import WorkerResult
from forge.search.params import SearchParams


# --------------------------------------------------------------------------- #
# ProgressEvent dataclass                                                      #
# --------------------------------------------------------------------------- #


class TestProgressEvent:
    def test_required_fields_only(self) -> None:
        e = ProgressEvent(kind="candidate_ok", label="test")
        assert e.kind == "candidate_ok"
        assert e.label == "test"
        assert e.params is None
        assert e.median_us is None
        assert e.round_num is None
        assert e.speedup is None

    def test_all_fields(self) -> None:
        p = SearchParams(block_size=256, num_warps=4, num_stages=1)
        e = ProgressEvent(
            kind="search_done",
            label="done",
            params=p,
            median_us=88.5,
            round_num=2,
            speedup=3.14,
        )
        assert e.params == p
        assert e.median_us == pytest.approx(88.5)
        assert e.round_num == 2
        assert e.speedup == pytest.approx(3.14)

    def test_all_kinds_valid(self) -> None:
        for kind in ("candidate_ok", "candidate_fail", "candidate_incorrect",
                     "cache_hit", "cache_miss", "round_done", "search_done", "search_start"):
            e = ProgressEvent(kind=kind, label="x")  # type: ignore[arg-type]
            assert e.kind == kind


# --------------------------------------------------------------------------- #
# make_progress_callback                                                       #
# --------------------------------------------------------------------------- #


class TestMakeProgressCallback:
    def test_none_returns_noop(self) -> None:
        cb = make_progress_callback(None)
        cb(ProgressEvent(kind="cache_hit", label="x"))  # must not raise

    def test_progress_event_callback_passthrough(self) -> None:
        received: list[ProgressEvent] = []

        def cb(e: ProgressEvent) -> None:
            received.append(e)

        wrapped = make_progress_callback(cb)
        evt = ProgressEvent(kind="candidate_ok", label="hi")
        wrapped(evt)
        assert received == [evt]

    def test_str_callback_wrapped_with_deprecation(self) -> None:
        received: list[str] = []

        def str_cb(msg: str) -> None:
            received.append(msg)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            wrapped = make_progress_callback(str_cb)
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "#190" in str(w[0].message)

        evt = ProgressEvent(kind="candidate_ok", label="hello world")
        wrapped(evt)
        assert received == ["hello world"]

    def test_unannotated_callback_no_warning(self) -> None:
        received: list[ProgressEvent] = []

        def cb(e):  # no annotation
            received.append(e)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            wrapped = make_progress_callback(cb)
        dep = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert not dep

        evt = ProgressEvent(kind="cache_hit", label="x")
        wrapped(evt)
        assert received == [evt]


# --------------------------------------------------------------------------- #
# Orchestrator emits ProgressEvent                                             #
# --------------------------------------------------------------------------- #


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


def _bench(median: float = 100.0) -> BenchmarkResult:
    return BenchmarkResult(
        median_us=median, p20_us=median * 0.9, p80_us=median * 1.1, p95_us=median * 1.2
    )


def _tmp_repo(tmp_path) -> KernelRepository:
    return KernelRepository(tmp_path / "test.db")


def _ok_worker(median: float = 80.0) -> WorkerResult:
    return WorkerResult(
        success=True,
        correct=True,
        candidate=_bench(median),
        baseline=_bench(200.0),
        baseline_name="eager",
    )


class TestOrchestratorEmitsProgressEvent:
    def test_search_start_event_emitted(self, tmp_path) -> None:
        events: list[ProgressEvent] = []
        repo = _tmp_repo(tmp_path)
        orch = Orchestrator(repo=repo, progress=events.append)

        with (
            patch("forge.orchestrator.run_in_worker", return_value=_ok_worker()),
            patch("forge.orchestrator.generate", return_value="# stub"),
        ):
            orch.optimize(_spec(), budget=1)

        kinds = [e.kind for e in events]
        assert "search_start" in kinds
        repo.close()

    def test_candidate_ok_event_emitted(self, tmp_path) -> None:
        events: list[ProgressEvent] = []
        repo = _tmp_repo(tmp_path)
        orch = Orchestrator(repo=repo, progress=events.append)

        with (
            patch("forge.orchestrator.run_in_worker", return_value=_ok_worker()),
            patch("forge.orchestrator.generate", return_value="# stub"),
        ):
            orch.optimize(_spec(), budget=1)

        ok_events = [e for e in events if e.kind == "candidate_ok"]
        assert ok_events
        assert ok_events[0].median_us is not None
        assert ok_events[0].median_us == pytest.approx(80.0)
        repo.close()

    def test_candidate_fail_event_emitted(self, tmp_path) -> None:
        events: list[ProgressEvent] = []
        repo = _tmp_repo(tmp_path)
        orch = Orchestrator(repo=repo, progress=events.append)
        fail = WorkerResult(success=False, error="CUDA boom")

        with (
            patch("forge.orchestrator.run_in_worker", return_value=fail),
            patch("forge.orchestrator.generate", return_value="# stub"),
        ):
            orch.optimize(_spec(), budget=1)

        fail_events = [e for e in events if e.kind == "candidate_fail"]
        assert fail_events
        repo.close()

    def test_search_done_event_has_speedup(self, tmp_path) -> None:
        events: list[ProgressEvent] = []
        repo = _tmp_repo(tmp_path)
        orch = Orchestrator(repo=repo, progress=events.append)

        with (
            patch("forge.orchestrator.run_in_worker", return_value=_ok_worker(80.0)),
            patch("forge.orchestrator.generate", return_value="# stub"),
        ):
            orch.optimize(_spec(), budget=1)

        done_events = [e for e in events if e.kind == "search_done"]
        assert done_events
        assert done_events[0].speedup is not None
        assert done_events[0].speedup == pytest.approx(200.0 / 80.0, rel=1e-3)
        repo.close()

    def test_cache_hit_event_emitted(self, tmp_path) -> None:
        from datetime import UTC, datetime

        from forge.cache.key import CacheKey
        from forge.cache.repository import CachedKernel

        events: list[ProgressEvent] = []
        spec = _spec()
        repo = _tmp_repo(tmp_path)
        key = CacheKey.from_spec_and_env(spec)
        p = SearchParams(block_size=4096, num_warps=8, num_stages=1)
        repo.put(key, CachedKernel(
            cache_key=key,
            params=p.to_dict(),
            kernel_code="# stub",
            benchmark_json=_bench().to_dict(),
            baseline_us=200.0,
            created_at=datetime.now(UTC),
        ))

        orch = Orchestrator(repo=repo, progress=events.append)
        orch.optimize(spec, budget=1)

        kinds = [e.kind for e in events]
        assert "cache_hit" in kinds
        repo.close()

    def test_str_callback_still_works_with_deprecation(self, tmp_path) -> None:
        """後方互換: str コールバックでも動作する（DeprecationWarning あり）。"""
        messages: list[str] = []

        def str_cb(msg: str) -> None:
            messages.append(msg)

        repo = _tmp_repo(tmp_path)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            orch = Orchestrator(repo=repo, progress=str_cb)
        assert any(issubclass(x.category, DeprecationWarning) for x in w)

        with (
            patch("forge.orchestrator.run_in_worker", return_value=_ok_worker()),
            patch("forge.orchestrator.generate", return_value="# stub"),
        ):
            orch.optimize(_spec(), budget=1)

        # 文字列メッセージが届いていること
        assert any(messages)
        assert all(isinstance(m, str) for m in messages)
        repo.close()

    def test_event_params_field_set_for_candidate(self, tmp_path) -> None:
        events: list[ProgressEvent] = []
        repo = _tmp_repo(tmp_path)
        orch = Orchestrator(repo=repo, progress=events.append)

        with (
            patch("forge.orchestrator.run_in_worker", return_value=_ok_worker()),
            patch("forge.orchestrator.generate", return_value="# stub"),
        ):
            orch.optimize(_spec(), budget=1)

        ok_events = [e for e in events if e.kind == "candidate_ok"]
        assert ok_events
        assert isinstance(ok_events[0].params, SearchParams)
        repo.close()
