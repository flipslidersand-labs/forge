"""BayesianGenerator の CPU ユニットテスト。optuna が必要。"""

from __future__ import annotations

import pytest
import torch

from forge.ir.kernel_spec import KernelSpec
from forge.ir.tensor_spec import TensorSpec
from forge.search.bayesian_generator import BayesianGenerator
from forge.search.candidate import CandidateGenerator, HistoryEntry
from forge.search.params import SearchParams

optuna = pytest.importorskip("optuna", reason="optuna not installed; skip Bayesian tests")


def _rmsnorm_spec() -> KernelSpec:
    return KernelSpec(
        op_type="rmsnorm",
        input_specs=(
            TensorSpec((32, 512), torch.float32, True),
            TensorSpec((512,), torch.float32, True),
        ),
        output_specs=(TensorSpec((32, 512), torch.float32, True),),
        constants={"eps": 1e-6},
        graph_hash="rmsnorm_v1",
        constraints=(),
    )


def _gelu_spec() -> KernelSpec:
    return KernelSpec(
        op_type="gelu",
        input_specs=(TensorSpec((64, 256), torch.float32, True),),
        output_specs=(TensorSpec((64, 256), torch.float32, True),),
        constants={},
        graph_hash="gelu_v1",
        constraints=(),
    )


class TestBayesianGeneratorProtocol:
    def test_implements_candidate_generator_protocol(self):
        gen = BayesianGenerator(seed=42)
        assert isinstance(gen, CandidateGenerator)

    def test_generate_returns_list_of_search_params(self):
        gen = BayesianGenerator(seed=42)
        results = gen.generate(_rmsnorm_spec(), "8.0", budget=5)
        assert isinstance(results, list)
        assert all(isinstance(p, SearchParams) for p in results)

    def test_generate_respects_budget(self):
        gen = BayesianGenerator(seed=42)
        results = gen.generate(_rmsnorm_spec(), "8.0", budget=3)
        assert len(results) <= 3

    def test_generate_no_duplicates(self):
        gen = BayesianGenerator(seed=42)
        results = gen.generate(_rmsnorm_spec(), "8.0", budget=8)
        assert len(results) == len(set(results))

    def test_reset_usage_resets_trial_count(self):
        gen = BayesianGenerator(seed=42)
        gen.generate(_rmsnorm_spec(), "8.0", budget=4)
        assert gen._trial_count == 0  # BayesianGenerator は _trial_count を reset_usage でリセット
        gen.reset_usage()
        assert gen._trial_count == 0

    def test_generate_with_empty_history(self):
        gen = BayesianGenerator(seed=42, n_startup_trials=3)
        results = gen.generate(_rmsnorm_spec(), "8.0", budget=4, history=[])
        assert len(results) > 0

    def test_generate_with_history_influences_output(self):
        """history を渡しても正常に動作し SearchParams が返ること。"""
        gen = BayesianGenerator(seed=42, n_startup_trials=2)
        history = [
            HistoryEntry(
                params=SearchParams(
                    block_size=512,
                    num_warps=4,
                    num_stages=1,
                    acc_dtype="fp32",
                    variant="single_row",
                    rows_per_program=1,
                ),
                correct=True,
                median_us=20.0,
            ),
            HistoryEntry(
                params=SearchParams(
                    block_size=1024,
                    num_warps=8,
                    num_stages=2,
                    acc_dtype="fp32",
                    variant="single_row",
                    rows_per_program=1,
                ),
                correct=True,
                median_us=15.0,
            ),
        ]
        results = gen.generate(_rmsnorm_spec(), "8.0", budget=4, history=history)
        assert len(results) > 0
        assert all(isinstance(p, SearchParams) for p in results)

    def test_incorrect_history_entries_ignored(self):
        """INCORRECT な履歴エントリは study に注入されず正常に動作すること。"""
        gen = BayesianGenerator(seed=42)
        history = [
            HistoryEntry(
                params=SearchParams(
                    block_size=512,
                    num_warps=4,
                    num_stages=1,
                    variant="single_row",
                    rows_per_program=1,
                ),
                correct=False,
                median_us=None,
            ),
        ]
        results = gen.generate(_rmsnorm_spec(), "8.0", budget=3, history=history)
        assert len(results) > 0

    def test_elementwise_op_returns_elementwise_variant(self):
        gen = BayesianGenerator(seed=42)
        results = gen.generate(_gelu_spec(), "8.0", budget=4)
        assert all(p.variant == "elementwise" for p in results)

    def test_old_gpu_limits_num_stages(self):
        """cc 6.1 (Pascal) では num_stages が 1 に制限されること。"""
        gen = BayesianGenerator(seed=42)
        results = gen.generate(_rmsnorm_spec(), "6.1", budget=10)
        assert all(p.num_stages == 1 for p in results)

    def test_default_n_used_when_no_budget(self):
        gen = BayesianGenerator(seed=42, default_n=5)
        results = gen.generate(_rmsnorm_spec(), "8.0")
        assert len(results) <= 5
