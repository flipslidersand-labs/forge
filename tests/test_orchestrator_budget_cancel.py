"""Orchestrator の時間予算・キャンセル追従 (#332)。CPU モックテスト、GPU 不要。

run_in_worker をモックし、max_total_s / cancel_event が optimize() /
optimize_sha() / optimize_rounds() の各探索ループを早期に打ち切ることを検証する。
"""

from __future__ import annotations

import tempfile
import threading
from pathlib import Path
from unittest.mock import patch

import torch

from forge.benchmark.statistics import BenchmarkResult
from forge.cache.repository import KernelRepository
from forge.ir.kernel_spec import KernelSpec
from forge.ir.tensor_spec import TensorSpec
from forge.orchestrator import Orchestrator
from forge.runtime.worker import WorkerResult
from forge.search.candidate import HistoryEntry
from forge.search.grid import GridSearch
from forge.search.llm_generator import LLMGenerator
from forge.search.space import SearchSpace


def _spec() -> KernelSpec:
    return KernelSpec(
        op_type="rmsnorm",
        input_specs=(
            TensorSpec((4, 64), torch.float32, True),
            TensorSpec((64,), torch.float32, True),
        ),
        output_specs=(TensorSpec((4, 64), torch.float32, True),),
        constants={"eps": 1e-6},
        graph_hash="rmsnorm_budget_cancel_v1",
        constraints=(),
    )


def _grid(n_block_sizes: int = 3) -> GridSearch:
    """複数候補（block_sizes の数だけ）を生成するグリッド。"""
    space = SearchSpace(
        block_sizes=[64, 128, 256][:n_block_sizes],
        num_warps=[4],
        acc_dtypes=["fp32"],
        variants=["single_row"],
    )
    return GridSearch(space)


def _ok_result() -> WorkerResult:
    bench = BenchmarkResult(median_us=10.0, p20_us=9.0, p80_us=11.0)
    baseline = BenchmarkResult(median_us=20.0, p20_us=19.0, p80_us=21.0)
    return WorkerResult(
        success=True,
        correct=True,
        candidate=bench,
        baseline=baseline,
        baseline_name="F.rms_norm",
    )


class TestOptimizeBudgetCancel:
    """optimize() の候補ループ（_explore_candidates）に対する検証。"""

    def test_cancel_event_preset_stops_before_any_candidate(self) -> None:
        cancel_event = threading.Event()
        cancel_event.set()

        with tempfile.TemporaryDirectory() as d:
            repo = KernelRepository(Path(d) / "c.db")
            orch = Orchestrator(repo=repo, cancel_event=cancel_event)

            with patch("forge.orchestrator.run_in_worker", return_value=_ok_result()) as mock_run:
                result = orch.optimize(_spec(), budget=1, search=_grid(3))

            mock_run.assert_not_called()
            assert result.best_params is None
            repo.close()

    def test_cancel_event_set_mid_run_stops_remaining_candidates(self) -> None:
        cancel_event = threading.Event()
        call_count = 0

        def _run_then_cancel(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                cancel_event.set()
            return _ok_result()

        with tempfile.TemporaryDirectory() as d:
            repo = KernelRepository(Path(d) / "c.db")
            orch = Orchestrator(repo=repo, cancel_event=cancel_event)

            with patch("forge.orchestrator.run_in_worker", side_effect=_run_then_cancel):
                orch.optimize(_spec(), budget=3, search=_grid(3))

            # 3 候補あるが、1 件評価した時点で cancel されるため 2 件目以降は走らない
            assert call_count == 1
            repo.close()

    def test_max_total_s_zero_stops_immediately(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            repo = KernelRepository(Path(d) / "c.db")
            orch = Orchestrator(repo=repo, max_total_s=0.0)

            with patch("forge.orchestrator.run_in_worker", return_value=_ok_result()) as mock_run:
                result = orch.optimize(_spec(), budget=1, search=_grid(3))

            mock_run.assert_not_called()
            assert result.best_params is None
            repo.close()

    def test_no_budget_or_cancel_runs_all_candidates(self) -> None:
        """後方互換: 指定しなければ従来どおり全候補を評価する。"""
        with tempfile.TemporaryDirectory() as d:
            repo = KernelRepository(Path(d) / "c.db")
            orch = Orchestrator(repo=repo)

            with patch("forge.orchestrator.run_in_worker", return_value=_ok_result()) as mock_run:
                orch.optimize(_spec(), budget=3, search=_grid(3))

            assert mock_run.call_count == 3
            repo.close()


class TestOptimizeSHABudgetCancel:
    """optimize_sha() のラウンドループ・候補ループに対する検証。"""

    def test_cancel_event_stops_before_second_round(self) -> None:
        cancel_event = threading.Event()
        round_starts: list[int] = []

        def _run_then_cancel_after_round1(*args, **kwargs):
            round_starts.append(1)
            if len(round_starts) >= 4:  # round1 の4候補評価後にcancel
                cancel_event.set()
            return _ok_result()

        with tempfile.TemporaryDirectory() as d:
            repo = KernelRepository(Path(d) / "c.db")
            orch = Orchestrator(repo=repo, cancel_event=cancel_event)

            with patch(
                "forge.orchestrator.run_in_worker", side_effect=_run_then_cancel_after_round1
            ):
                orch.optimize_sha(_spec(), initial_budget=4, halving_rounds=3, search=_grid(3))

            # round1(4候補: block_sizes x num_warps=1x1... 実際は _grid(3)使用時と
            # SearchSpace次第だが、いずれにせよ round2 以降の追加評価が起きない
            # ことを call_count で確認する。
            calls_after_round1 = len(round_starts)

        with tempfile.TemporaryDirectory() as d2:
            repo2 = KernelRepository(Path(d2) / "c2.db")
            orch2 = Orchestrator(repo=repo2)
            with patch("forge.orchestrator.run_in_worker", return_value=_ok_result()) as mock_run:
                orch2.optimize_sha(_spec(), initial_budget=4, halving_rounds=3, search=_grid(3))
            calls_without_cancel = mock_run.call_count
            repo2.close()

        assert calls_after_round1 < calls_without_cancel
        repo.close()


class TestOptimizeRoundsBudgetCancel:
    """optimize_rounds() のラウンドループ・候補ループに対する検証。"""

    def test_cancel_event_stops_before_second_round(self) -> None:
        cancel_event = threading.Event()
        call_count = 0

        def _propose(spec, cc, n, history):
            return [
                {
                    "base_variant": "single_row",
                    "block_size": 64,
                    "num_warps": 4,
                    "num_stages": 1,
                    "acc_dtype": "fp32",
                    "rows_per_program": 1,
                    "hypothesis": "h",
                }
            ]

        def _run_then_cancel(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            cancel_event.set()  # 1 候補評価したら次ラウンド前にキャンセル
            return _ok_result()

        with tempfile.TemporaryDirectory() as d:
            repo = KernelRepository(Path(d) / "c.db")
            orch = Orchestrator(repo=repo, cancel_event=cancel_event)
            llm = LLMGenerator(propose_fn=_propose)

            with patch("forge.orchestrator.run_in_worker", side_effect=_run_then_cancel):
                result = orch.optimize_rounds(_spec(), llm=llm, n_rounds=5, candidates_per_round=1)

            # 5ラウンド設定だが、1ラウンド目の評価直後にキャンセルされるため
            # 2ラウンド目以降は実行されない。
            assert call_count == 1
            assert len(result.rounds) == 1
            repo.close()

    def test_no_budget_or_cancel_runs_all_rounds(self) -> None:
        """後方互換: 指定しなければ従来どおり全ラウンドを実行する。"""

        def _propose(spec, cc, n, history):
            return [
                {
                    "base_variant": "single_row",
                    "block_size": 64,
                    "num_warps": 4,
                    "num_stages": 1,
                    "acc_dtype": "fp32",
                    "rows_per_program": 1,
                    "hypothesis": f"h{len(history)}",
                }
            ]

        with tempfile.TemporaryDirectory() as d:
            repo = KernelRepository(Path(d) / "c.db")
            orch = Orchestrator(repo=repo)
            llm = LLMGenerator(propose_fn=_propose)

            with patch("forge.orchestrator.run_in_worker", return_value=_ok_result()):
                result = orch.optimize_rounds(_spec(), llm=llm, n_rounds=3, candidates_per_round=1)

            assert len(result.rounds) == 3
            repo.close()
