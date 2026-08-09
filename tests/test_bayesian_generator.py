"""BayesianGenerator のオフラインテスト。optuna CPU mock のみ使用。"""

from __future__ import annotations

import pytest
import torch

from forge.ir.kernel_spec import KernelSpec
from forge.ir.tensor_spec import TensorSpec
from forge.search.bayesian_generator import BayesianGenerator
from forge.search.candidate import CandidateGenerator, HistoryEntry
from forge.search.params import SearchParams

optuna = pytest.importorskip("optuna", reason="requires optuna (pip install forge-kernel[search])")


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


def _entry(block_size: int, num_warps: int, median_us: float) -> HistoryEntry:
    return HistoryEntry(
        params=SearchParams(
            block_size=block_size,
            num_warps=num_warps,
            num_stages=2,
            acc_dtype="fp32",
            variant="single_row",
            rows_per_program=1,
        ),
        correct=True,
        median_us=median_us,
    )


class TestBayesianGenerator:
    def _gen(self, n_startup: int = 2, seed: int = 42) -> BayesianGenerator:
        return BayesianGenerator(n_startup_trials=n_startup, seed=seed)

    # ── Protocol 適合 ─────────────────────────────────────────────────────────

    def test_satisfies_candidate_generator_protocol(self) -> None:
        assert isinstance(self._gen(), CandidateGenerator)

    # ── cold start（history なし）─────────────────────────────────────────────

    def test_cold_start_returns_requested_budget(self) -> None:
        gen = self._gen()
        results = gen.generate(_spec(), "8.0", budget=5)
        assert len(results) == 5

    def test_cold_start_returns_valid_search_params(self) -> None:
        gen = self._gen()
        results = gen.generate(_spec(), "8.0", budget=4)
        for p in results:
            assert isinstance(p, SearchParams)

    def test_cold_start_no_duplicates(self) -> None:
        gen = self._gen()
        results = gen.generate(_spec(), "8.0", budget=8)
        assert len(results) == len(set(results))

    # ── warm start（history あり）─────────────────────────────────────────────

    def test_warm_start_uses_history_returns_valid_params(self) -> None:
        gen = self._gen(n_startup=1)
        history = [
            _entry(4096, 8, 120.0),
            _entry(2048, 8, 180.0),
            _entry(1024, 4, 250.0),
        ]
        results = gen.generate(_spec(), "8.0", budget=4, history=history)
        assert len(results) >= 1
        for p in results:
            assert isinstance(p, SearchParams)

    def test_incorrect_history_entries_are_ignored(self) -> None:
        gen = self._gen(n_startup=1)
        history = [
            HistoryEntry(
                params=SearchParams(
                    block_size=4096,
                    num_warps=8,
                    num_stages=2,
                    acc_dtype="fp32",
                    variant="single_row",
                    rows_per_program=1,
                ),
                correct=False,
                median_us=None,
            ),
            _entry(2048, 4, 200.0),
        ]
        results = gen.generate(_spec(), "8.0", budget=3, history=history)
        assert len(results) >= 1

    # ── 再現性（seed）─────────────────────────────────────────────────────────

    def test_same_seed_produces_same_results(self) -> None:
        spec = _spec()
        gen_a = BayesianGenerator(n_startup_trials=2, seed=99)
        gen_b = BayesianGenerator(n_startup_trials=2, seed=99)
        assert gen_a.generate(spec, "8.0", budget=5) == gen_b.generate(spec, "8.0", budget=5)

    # ── reset_usage ───────────────────────────────────────────────────────────

    def test_reset_usage_is_no_op(self) -> None:
        gen = self._gen()
        gen.reset_usage()
        results = gen.generate(_spec(), "8.0", budget=3)
        assert len(results) == 3

    # ── budget=0 のエッジケース ───────────────────────────────────────────────

    def test_zero_budget_returns_empty(self) -> None:
        gen = self._gen()
        results = gen.generate(_spec(), "8.0", budget=0)
        assert results == []

    # ── optuna 未インストール時のエラー ──────────────────────────────────────

    def test_import_error_when_optuna_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import forge.search.bayesian_generator as mod

        monkeypatch.setattr(mod, "_OPTUNA_AVAILABLE", False)
        with pytest.raises(ImportError, match="optuna"):
            BayesianGenerator()
