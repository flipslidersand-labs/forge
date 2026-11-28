"""HistoryEntry の outcome (#486)。CPU モックテスト、GPU 不要。

worker 失敗・SKIP で評価できなかった候補が、数値的に誤った候補と同じ
INCORRECT として LLM にフィードバックされないことを検証する。
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import torch

from forge.cache.repository import KernelRepository
from forge.ir.kernel_spec import KernelSpec
from forge.ir.tensor_spec import TensorSpec
from forge.orchestrator import Orchestrator
from forge.runtime.worker import WorkerResult
from forge.search.candidate import HistoryEntry
from forge.search.iterative_search import _curate_history, build_feedback_prompt
from forge.search.llm_generator import LLMGenerator
from forge.search.params import SearchParams


def _spec() -> KernelSpec:
    return KernelSpec(
        op_type="rmsnorm",
        input_specs=(
            TensorSpec((4, 64), torch.float32, True),
            TensorSpec((64,), torch.float32, True),
        ),
        output_specs=(TensorSpec((4, 64), torch.float32, True),),
        constants={"eps": 1e-6},
        graph_hash="rmsnorm_history_outcome_v1",
        constraints=(),
    )


def _params(ws: int = 4) -> SearchParams:
    return SearchParams(
        block_size=64,
        num_warps=ws,
        num_stages=1,
        variant="single_row",
        acc_dtype="fp32",
        rows_per_program=1,
    )


def _proposal(ws: int) -> dict:
    return {
        "base_variant": "single_row",
        "block_size": 64,
        "num_warps": ws,
        "num_stages": 1,
        "acc_dtype": "fp32",
        "rows_per_program": 1,
        "hypothesis": "h",
    }


class TestSummary:
    def test_default_incorrect_is_unchanged(self) -> None:
        e = HistoryEntry(params=_params(), correct=False, median_us=None)
        assert e.summary().endswith("-> INCORRECT")

    def test_error_is_not_incorrect(self) -> None:
        e = HistoryEntry(
            params=_params(),
            correct=False,
            median_us=None,
            outcome="error",
            error="no template for variant welford",
        )
        s = e.summary()
        assert "INCORRECT" not in s
        assert "ERROR: no template for variant welford" in s

    def test_error_reason_is_shortened_to_one_line(self) -> None:
        e = HistoryEntry(
            params=_params(),
            correct=False,
            median_us=None,
            outcome="error",
            error="line1\n" + "x" * 500,
        )
        s = e.summary()
        assert "\n" not in s
        assert len(s) < 300

    def test_error_without_reason(self) -> None:
        e = HistoryEntry(params=_params(), correct=False, median_us=None, outcome="error")
        assert e.summary().endswith("-> ERROR")


class TestCurateHistory:
    def test_error_entries_are_not_failure_patterns(self) -> None:
        err = HistoryEntry(_params(4), False, None, outcome="error", error="crash")
        bad = HistoryEntry(_params(8), False, None)
        curated = _curate_history([err, bad], top_k=3)
        assert bad in curated
        assert err not in curated
        assert "ERROR" not in build_feedback_prompt([err, bad])


class TestOptimizeRoundsHistory:
    def _run(self, worker_result: WorkerResult) -> list[HistoryEntry]:
        seen: list[list[HistoryEntry]] = []

        def _propose(spec, cc, n, history):
            seen.append(list(history))
            return [_proposal(4 * (len(seen)))]

        with tempfile.TemporaryDirectory() as d:
            repo = KernelRepository(Path(d) / "c.db")
            orch = Orchestrator(repo=repo)
            llm = LLMGenerator(propose_fn=_propose)
            with patch("forge.orchestrator.run_in_worker", return_value=worker_result):
                orch.optimize_rounds(_spec(), llm=llm, n_rounds=2, candidates_per_round=1)
            repo.close()
        return seen[1]  # 2 ラウンド目に渡された history

    def test_worker_failure_is_not_reported_as_incorrect(self) -> None:
        history = self._run(WorkerResult(success=False, error="worker crashed: boom"))
        assert len(history) == 1
        assert history[0].outcome == "error"
        assert "INCORRECT" not in history[0].summary()
        assert "ERROR: worker crashed: boom" in history[0].summary()

    def test_numerically_wrong_candidate_stays_incorrect(self) -> None:
        history = self._run(WorkerResult(success=True, correct=False))
        assert len(history) == 1
        assert history[0].outcome == "incorrect"
        assert history[0].summary().endswith("-> INCORRECT")


class TestSkippedOutcome:
    """SKIP（generate() が候補を拒否。未実行）とワーカー失敗（実行して落ちた）の区別 (#504)。"""

    def test_summary_distinguishes_skipped_from_error(self) -> None:
        sk = HistoryEntry(_params(), False, None, outcome="skipped", error="No codegen template")
        er = HistoryEntry(_params(), False, None, outcome="error", error="No codegen template")
        assert "SKIPPED: No codegen template" in sk.summary()
        assert "ERROR" not in sk.summary()
        assert "INCORRECT" not in sk.summary()
        assert "ERROR: No codegen template" in er.summary()
        assert "SKIPPED" not in er.summary()

    def test_skipped_entries_are_not_failure_patterns(self) -> None:
        sk = HistoryEntry(_params(4), False, None, outcome="skipped", error="no template")
        bad = HistoryEntry(_params(8), False, None)
        curated = _curate_history([sk, bad], top_k=3)
        assert bad in curated
        assert sk not in curated
        assert "SKIPPED" not in build_feedback_prompt([sk, bad])

    def test_eval_one_marks_skip_and_not_worker_failure(self) -> None:
        orch = Orchestrator(repo=None)  # type: ignore[arg-type]
        with patch("forge.orchestrator.generate", side_effect=ValueError("no template")):
            exp, *_ = orch._eval_one(_spec(), _params(), [], None, {}, "lbl")
        assert exp.skipped is True
        assert exp.success is False  # failed_count の集計は従来どおり
        wr = WorkerResult(success=False, error="worker crashed")
        with patch("forge.orchestrator.run_in_worker", return_value=wr):
            exp, *_ = orch._eval_one(_spec(), _params(), [], None, {}, "lbl")
        assert exp.skipped is False

    def test_skip_is_recorded_as_skipped_in_history(self) -> None:
        seen: list[list[HistoryEntry]] = []

        def _propose(spec, cc, n, history):
            seen.append(list(history))
            return [_proposal(4 * len(seen))]

        with tempfile.TemporaryDirectory() as d:
            repo = KernelRepository(Path(d) / "c.db")
            orch = Orchestrator(repo=repo)
            llm = LLMGenerator(propose_fn=_propose)
            with patch("forge.orchestrator.generate", side_effect=ValueError("no template")):
                orch.optimize_rounds(_spec(), llm=llm, n_rounds=2, candidates_per_round=1)
            repo.close()
        entry = seen[1][0]
        assert entry.outcome == "skipped"
        assert "SKIPPED: no template" in entry.summary()
