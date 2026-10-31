"""Python logging 統合テスト（#213）。GPU 不要。caplog でレコードを検証。"""

from __future__ import annotations

import json
import logging
import sqlite3
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest
import torch

from forge.benchmark.statistics import BenchmarkResult
from forge.cache.key import CacheKey
from forge.cache.repository import KernelRepository
from forge.ir.kernel_spec import KernelSpec
from forge.ir.tensor_spec import TensorSpec
from forge.orchestrator import Orchestrator
from forge.runtime.worker import WorkerResult, run_in_worker
from forge.search.params import SearchParams

# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #


def _spec(op_type: str = "rmsnorm") -> KernelSpec:
    return KernelSpec(
        op_type=op_type,
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
        median_us=median,
        p20_us=median * 0.9,
        p80_us=median * 1.1,
        p95_us=median * 1.2,
        samples_us=[median] * 5,
    )


def _params() -> SearchParams:
    return SearchParams(block_size=4096, num_warps=8, num_stages=1, acc_dtype="fp32")


@contextmanager
def _tmp_repo() -> Generator[KernelRepository, None, None]:
    with tempfile.TemporaryDirectory() as d:
        repo = KernelRepository(Path(d) / "test.db")
        try:
            yield repo
        finally:
            repo.close()


# --------------------------------------------------------------------------- #
# forge.cache logger                                                           #
# --------------------------------------------------------------------------- #


class TestCacheLogging:
    def test_miss_logged_at_debug(self, caplog: pytest.LogCaptureFixture) -> None:
        with _tmp_repo() as repo:
            key = CacheKey.from_spec_and_env(_spec())
            with caplog.at_level(logging.DEBUG, logger="forge.cache"):
                result = repo.get(key)
        assert result is None
        assert any("miss" in r.message for r in caplog.records if r.name == "forge.cache")

    def test_hit_logged_at_debug(self, caplog: pytest.LogCaptureFixture) -> None:
        from datetime import UTC, datetime

        from forge.cache.repository import CachedKernel

        spec = _spec()
        with _tmp_repo() as repo:
            key = CacheKey.from_spec_and_env(spec)
            kernel = CachedKernel(
                cache_key=key,
                params=_params().to_dict(),
                kernel_code="# stub",
                benchmark_json=_bench().to_dict(),
                baseline_us=200.0,
                created_at=datetime.now(UTC),
            )
            with caplog.at_level(logging.DEBUG, logger="forge.cache"):
                repo.put(key, kernel)
            assert any("write" in r.message for r in caplog.records if r.name == "forge.cache")

            caplog.clear()
            with caplog.at_level(logging.DEBUG, logger="forge.cache"):
                repo.get(key)
        assert any("hit" in r.message for r in caplog.records if r.name == "forge.cache")


# --------------------------------------------------------------------------- #
# forge.runtime logger                                                         #
# --------------------------------------------------------------------------- #


class TestRuntimeLogging:
    def test_timeout_logged_as_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="forge.runtime"):
            with patch(
                "subprocess.run", side_effect=__import__("subprocess").TimeoutExpired("x", 0.1)
            ):
                result = run_in_worker(
                    kernel_code="# stub",
                    op_type="rmsnorm",
                    benchmark_input=[],
                    constants={},
                    timeout_s=0.1,
                )
        assert not result.success
        warnings = [
            r for r in caplog.records if r.name == "forge.runtime" and r.levelno == logging.WARNING
        ]
        assert warnings, "timeout should emit a WARNING"
        assert "timeout" in warnings[0].message.lower()

    def test_worker_crash_logged_as_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        mock_proc = MagicMock()
        mock_proc.returncode = 1
        mock_proc.stderr = "CUDA error"
        mock_proc.stdout = ""
        with caplog.at_level(logging.WARNING, logger="forge.runtime"):
            with patch("subprocess.run", return_value=mock_proc):
                result = run_in_worker(
                    kernel_code="# stub",
                    op_type="rmsnorm",
                    benchmark_input=[],
                    constants={},
                )
        assert not result.success
        warnings = [
            r for r in caplog.records if r.name == "forge.runtime" and r.levelno == logging.WARNING
        ]
        assert warnings
        assert "crash" in warnings[0].message.lower()


# --------------------------------------------------------------------------- #
# forge.orchestrator logger                                                    #
# --------------------------------------------------------------------------- #


class TestOrchestratorLogging:
    def _mock_worker_success(self, median: float = 80.0) -> WorkerResult:
        return WorkerResult(
            success=True,
            correct=True,
            candidate=_bench(median),
            baseline=_bench(200.0),
            baseline_name="eager",
        )

    def test_search_start_logged_at_info(self, caplog: pytest.LogCaptureFixture) -> None:
        spec = _spec()
        with _tmp_repo() as repo:
            orch = Orchestrator(repo=repo)

            with (
                caplog.at_level(logging.INFO, logger="forge.orchestrator"),
                patch("forge.orchestrator.run_in_worker", return_value=self._mock_worker_success()),
                patch("forge.orchestrator.generate", return_value="# stub"),
            ):
                orch.optimize(spec, budget=1)

        info_msgs = [
            r.message
            for r in caplog.records
            if r.name == "forge.orchestrator" and r.levelno == logging.INFO
        ]
        assert any("search start" in m for m in info_msgs)

    def test_cache_hit_logged_at_info(self, caplog: pytest.LogCaptureFixture) -> None:
        from datetime import UTC, datetime

        from forge.cache.repository import CachedKernel

        spec = _spec()
        with _tmp_repo() as repo:
            key = CacheKey.from_spec_and_env(spec)
            repo.put(
                key,
                CachedKernel(
                    cache_key=key,
                    params=_params().to_dict(),
                    kernel_code="# stub",
                    benchmark_json=_bench().to_dict(),
                    baseline_us=200.0,
                    created_at=datetime.now(UTC),
                ),
            )

            orch = Orchestrator(repo=repo)
            with caplog.at_level(logging.INFO, logger="forge.orchestrator"):
                orch.optimize(spec, budget=1)

        info_msgs = [r.message for r in caplog.records if r.name == "forge.orchestrator"]
        assert any("cache HIT" in m for m in info_msgs)

    def test_worker_fail_logged_as_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        spec = _spec()
        with _tmp_repo() as repo:
            orch = Orchestrator(repo=repo)
            fail = WorkerResult(success=False, error="CUDA boom")

            with (
                caplog.at_level(logging.WARNING, logger="forge.orchestrator"),
                patch("forge.orchestrator.run_in_worker", return_value=fail),
                patch("forge.orchestrator.generate", return_value="# stub"),
            ):
                orch.optimize(spec, budget=1)

        warns = [
            r
            for r in caplog.records
            if r.name == "forge.orchestrator" and r.levelno == logging.WARNING
        ]
        assert any("FAIL" in r.message for r in warns)

    def test_cached_best_logged_at_info(self, caplog: pytest.LogCaptureFixture) -> None:
        spec = _spec()
        with _tmp_repo() as repo:
            orch = Orchestrator(repo=repo)

            with (
                caplog.at_level(logging.INFO, logger="forge.orchestrator"),
                patch("forge.orchestrator.run_in_worker", return_value=self._mock_worker_success()),
                patch("forge.orchestrator.generate", return_value="# stub"),
            ):
                orch.optimize(spec, budget=1)

        info_msgs = [
            r.message
            for r in caplog.records
            if r.name == "forge.orchestrator" and r.levelno == logging.INFO
        ]
        assert any("cached best" in m for m in info_msgs)

    def test_no_best_found_logged_as_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        spec = _spec()
        with _tmp_repo() as repo:
            orch = Orchestrator(repo=repo)
            fail = WorkerResult(success=False, error="bad")

            with (
                caplog.at_level(logging.WARNING, logger="forge.orchestrator"),
                patch("forge.orchestrator.run_in_worker", return_value=fail),
                patch("forge.orchestrator.generate", return_value="# stub"),
            ):
                orch.optimize(spec, budget=1)

        warns = [
            r.message
            for r in caplog.records
            if r.name == "forge.orchestrator" and r.levelno == logging.WARNING
        ]
        assert any("no best found" in m for m in warns)

    def test_logger_hierarchy(self) -> None:
        logger = logging.getLogger("forge.orchestrator")
        assert logger.name == "forge.orchestrator"
        assert logger.parent is not None
        assert logger.parent.name in ("forge", "root")

    def test_forge_root_level_suppresses_info(self, caplog: pytest.LogCaptureFixture) -> None:
        """logging.getLogger("forge").setLevel(WARNING) で INFO が抑制される。"""
        spec = _spec()
        with _tmp_repo() as repo:
            orch = Orchestrator(repo=repo)

            with (
                caplog.at_level(logging.WARNING, logger="forge"),
                patch("forge.orchestrator.run_in_worker", return_value=self._mock_worker_success()),
                patch("forge.orchestrator.generate", return_value="# stub"),
            ):
                orch.optimize(spec, budget=1)

        info_records = [r for r in caplog.records if r.levelno == logging.INFO]
        assert not info_records, "INFO logs should be suppressed at WARNING level"


# --------------------------------------------------------------------------- #
# forge.search logger                                                          #
# --------------------------------------------------------------------------- #


class TestSearchLogging:
    def test_grid_search_logs_debug(self, caplog: pytest.LogCaptureFixture) -> None:
        from forge.search.grid import GridSearch

        spec = _spec()
        gs = GridSearch()
        with caplog.at_level(logging.DEBUG, logger="forge.search.grid"):
            candidates = gs.generate(spec, compute_capability="8.0", budget=5)

        assert len(candidates) <= 5
        debug_msgs = [r.message for r in caplog.records if r.name == "forge.search.grid"]
        assert debug_msgs, "GridSearch.generate should emit a DEBUG log"

    def test_random_search_logs_debug(self, caplog: pytest.LogCaptureFixture) -> None:
        from forge.search.random_search import RandomSearch

        spec = _spec()
        rs = RandomSearch(seed=42)
        with caplog.at_level(logging.DEBUG, logger="forge.search.random"):
            candidates = rs.generate(spec, compute_capability="8.0", budget=3)

        assert len(candidates) <= 3
        debug_msgs = [r.message for r in caplog.records if r.name == "forge.search.random"]
        assert debug_msgs, "RandomSearch.generate should emit a DEBUG log"

    def test_search_logger_hierarchy(self) -> None:
        import logging

        grid_log = logging.getLogger("forge.search.grid")
        assert grid_log.parent is not None
        assert "forge" in grid_log.parent.name or grid_log.parent.name == "root"
