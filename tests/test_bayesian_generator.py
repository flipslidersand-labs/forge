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

    @pytest.mark.parametrize(
        ("cc", "overrides"),
        [
            ("8.0", {"num_warps": 2}),
            ("8.0", {"num_warps": 64}),
            ("8.0", {"num_warps": 0}),
            ("8.0", {"num_stages": 0}),
            ("8.0", {"variant": "multi_row", "rows_per_program": 32}),
            ("6.1", {"num_stages": 3}),  # max_stages=1 へクランプされる既存挙動
        ],
        ids=["warps2", "warps64", "warps0", "stages0", "rows32", "cc6_stages3"],
    )
    def test_out_of_range_history_does_not_raise(self, cc, overrides):
        """num_warps/num_stages/rows_per_program が Optuna 分布の範囲外の履歴があっても、
        generate が例外にならず budget 件の SearchParams を返すこと。"""
        base = {
            "block_size": 512,
            "num_warps": 4,
            "num_stages": 1,
            "acc_dtype": "fp32",
            "variant": "single_row",
            "rows_per_program": 1,
        }
        history = [
            HistoryEntry(
                params=SearchParams(**{**base, **overrides}),
                correct=True,
                median_us=10.0,
            )
        ]
        gen = BayesianGenerator(seed=42, n_startup_trials=2)
        results = gen.generate(_rmsnorm_spec(), cc, budget=4, history=history)
        assert len(results) == 4
        assert all(isinstance(p, SearchParams) for p in results)

    def test_out_of_range_history_entry_is_skipped_not_clamped(self, caplog):
        """範囲外のエントリは study に注入されず（クランプされず）、debug ログが残ること。"""
        import logging

        history = [
            HistoryEntry(
                params=SearchParams(
                    block_size=512,
                    num_warps=64,
                    num_stages=1,
                    acc_dtype="fp32",
                    variant="single_row",
                    rows_per_program=1,
                ),
                correct=True,
                median_us=10.0,
            ),
            HistoryEntry(
                params=SearchParams(
                    block_size=512,
                    num_warps=8,
                    num_stages=1,
                    acc_dtype="fp32",
                    variant="single_row",
                    rows_per_program=1,
                ),
                correct=True,
                median_us=12.0,
            ),
        ]
        gen = BayesianGenerator(seed=42, n_startup_trials=2)
        with caplog.at_level(logging.DEBUG, logger="forge.search.bayesian_generator"):
            gen.generate(_rmsnorm_spec(), "8.0", budget=2, history=history)
        assert any("num_warps=64" in r.getMessage() for r in caplog.records)

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


class TestBayesianGeneratorNonReduction:
    def test_gelu_wide_hidden_returns_candidates(self):
        # N=4096 の gelu: 候補 block [64..2048] は全て N 未満。以前は全件捨てられ空だった
        spec = KernelSpec(
            op_type="gelu",
            input_specs=(TensorSpec((64, 4096), torch.float32, True),),
            output_specs=(TensorSpec((64, 4096), torch.float32, True),),
            constants={},
            graph_hash="gelu_v1",
            constraints=(),
        )
        out = BayesianGenerator(seed=0).generate(spec, "8.6", budget=4)
        assert len(out) >= 1
        assert all(p.variant == "elementwise" and p.block_size < 4096 for p in out)


def _sdpa_spec(s: int, d: int = 64) -> KernelSpec:
    qkv = TensorSpec((8, s, d), torch.float16, True)
    return KernelSpec(
        op_type="scaled_dot_product_attention",
        input_specs=(qkv, qkv, qkv),
        output_specs=(qkv,),
        constants={"is_causal": False},
        graph_hash="sdpa_v1",
        constraints=(),
    )


def _linear_spec() -> KernelSpec:
    f16 = torch.float16
    return KernelSpec(
        op_type="linear",
        input_specs=(
            TensorSpec((128, 1024), f16, True),
            TensorSpec((512, 1024), f16, True),
            TensorSpec((512,), f16, True),
        ),
        output_specs=(TensorSpec((128, 512), f16, True),),
        constants={},
        graph_hash="linear_v1",
        constraints=(),
    )


class TestBayesianAttentionBlocks:
    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    @pytest.mark.parametrize("s", [8, 16, 32, 64, 256])
    def test_blocks_do_not_exceed_seq_len(self, s, seed):
        out = BayesianGenerator(seed=seed, n_startup_trials=3).generate(
            _sdpa_spec(s), "8.9", budget=6
        )
        assert out  # S が小さくても空にならない
        assert all(16 <= p.block_size <= max(s, 16) for p in out)
        assert {p.variant for p in out} <= {"flash", "flash_causal_opt"}

    def test_small_seq_len_still_fills_budget(self):
        # S=8 は有効 block が 16 の 1 つだけ。試行の大半が捨てられず budget を満たす。
        out = BayesianGenerator(seed=0).generate(_sdpa_spec(8), "8.9", budget=6)
        assert len(out) == 6
        assert {p.block_size for p in out} == {16}

    def test_head_dim_not_multiple_of_16_returns_nothing(self):
        out = BayesianGenerator(seed=0).generate(_sdpa_spec(64, d=24), "8.9", budget=6)
        assert out == []


class TestBayesianGemmBlockK:
    def test_block_k_is_explored(self):
        out = BayesianGenerator(seed=0, n_startup_trials=20).generate(
            _linear_spec(), "8.9", budget=20
        )
        assert out
        assert {p.block_k for p in out} <= {16, 32, 64}
        assert len({p.block_k for p in out}) > 1  # 既定の 32 に固定されない

    def test_history_block_k_is_injected(self):
        hist = [
            HistoryEntry(
                SearchParams(block_size=64, num_warps=4, num_stages=2, variant="gemm", block_k=bk),
                True,
                10.0 + bk,
            )
            for bk in (16, 64)
        ]
        out = BayesianGenerator(seed=0, n_startup_trials=1).generate(
            _linear_spec(), "8.9", budget=4, history=hist
        )
        assert out
        assert all(p.block_k in (16, 32, 64) for p in out)

    def test_non_gemm_keeps_default_block_k(self):
        out = BayesianGenerator(seed=0).generate(_gelu_spec(), "8.9", budget=4)
        assert out and {p.block_k for p in out} == {32}
