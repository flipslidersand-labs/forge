"""LLMGenerator のオフラインテスト。propose_fn を注入し、ネットワーク/API キー不要。"""

import pytest
import torch

from forge.ir.kernel_spec import KernelSpec
from forge.ir.tensor_spec import TensorSpec
from forge.search._base_generator import _valid_block
from forge.search.bayesian_generator import (
    _block_ok,
    _block_ok_for_spec,
    _candidate_blocks,
    _candidate_variants,
)
from forge.search.candidate import CandidateGenerator, HistoryEntry
from forge.search.grid import GridSearch
from forge.search.llm_generator import (
    _SYSTEM,
    DEFAULT_MODEL,
    LLMGenerator,
    TokenUsage,
    build_prompt,
)
from forge.search.params import SUPPORTED_VARIANTS, SearchParams
from forge.search.random_search import RandomSearch
from forge.search.space import SearchSpace


def spec(n: int = 4096) -> KernelSpec:
    return KernelSpec(
        op_type="rmsnorm",
        input_specs=(
            TensorSpec((2048, n), torch.float16, True),
            TensorSpec((n,), torch.float16, True),
        ),
        output_specs=(TensorSpec((2048, n), torch.float16, True),),
        constants={"eps": 1e-6},
        graph_hash="rmsnorm_v1",
        constraints=(),
    )


def _f16(shape: tuple[int, ...]) -> TensorSpec:
    return TensorSpec(shape, torch.float16, True)


def op_spec(op: str) -> KernelSpec:
    """op ごとの代表 spec。gelu は N=4096、linear は K=1024、sdpa は head_dim=64。"""
    if op == "gelu":
        return KernelSpec("gelu", (_f16((64, 4096)),), (_f16((64, 4096)),), {}, "gelu_v1", ())
    if op == "linear":
        return KernelSpec(
            "linear",
            (_f16((128, 1024)), _f16((512, 1024)), _f16((512,))),
            (_f16((128, 512)),),
            {},
            "linear_v1",
            (),
        )
    if op == "scaled_dot_product_attention":
        qkv = _f16((8, 64, 64))
        return KernelSpec(op, (qkv, qkv, qkv), (qkv,), {"is_causal": False}, "sdpa_v1", ())
    return spec(4096)


def cand(**kw) -> dict:
    base = dict(
        base_variant="single_row",
        block_size=4096,
        num_warps=8,
        num_stages=1,
        acc_dtype="fp32",
        rows_per_program=1,
        hypothesis="test",
    )
    base.update(kw)
    return base


class TestProtocolConformance:
    def test_all_generators_satisfy_protocol(self) -> None:
        assert isinstance(GridSearch(), CandidateGenerator)
        assert isinstance(RandomSearch(), CandidateGenerator)
        assert isinstance(LLMGenerator(propose_fn=lambda *a: []), CandidateGenerator)


class TestLLMGeneratorCoercion:
    def _gen(self, raw: list[dict]) -> LLMGenerator:
        return LLMGenerator(propose_fn=lambda s, cc, n, h: raw)

    def test_valid_candidates_pass_through(self) -> None:
        out = self._gen([cand(), cand(num_warps=16)]).generate(spec(), "8.9")
        assert len(out) == 2
        assert all(isinstance(p, SearchParams) for p in out)

    def test_invalid_block_size_dropped(self) -> None:
        # 3000 は 2 のべき乗でない → SearchParams が弾く
        out = self._gen([cand(block_size=3000)]).generate(spec(), "8.9")
        assert out == []

    def test_single_row_block_below_hidden_dropped(self) -> None:
        # single_row で block < N(4096) は無効
        out = self._gen([cand(block_size=2048)]).generate(spec(4096), "8.9")
        assert out == []

    def test_two_pass_small_tile_allowed(self) -> None:
        out = self._gen([cand(base_variant="two_pass", block_size=1024)]).generate(
            spec(4096), "8.9"
        )
        assert len(out) == 1
        assert out[0].variant == "two_pass"

    def test_unknown_variant_dropped(self) -> None:
        out = self._gen([cand(base_variant="bogus")]).generate(spec(), "8.9")
        assert out == []

    def test_bad_rows_for_non_multi_row_dropped(self) -> None:
        # rows>1 は multi_row のみ許可
        out = self._gen([cand(base_variant="single_row", rows_per_program=4)]).generate(
            spec(), "8.9"
        )
        assert out == []

    def test_multi_row_with_rows_ok(self) -> None:
        out = self._gen([cand(base_variant="multi_row", rows_per_program=4)]).generate(
            spec(), "8.9"
        )
        assert len(out) == 1
        assert out[0].rows_per_program == 4

    def test_missing_key_dropped_not_crash(self) -> None:
        bad = {"base_variant": "single_row", "block_size": 4096}  # warps/stages 欠落
        out = self._gen([bad, cand()]).generate(spec(), "8.9")
        assert len(out) == 1  # 壊れた候補は捨て、正常な 1 件のみ

    def test_dedup(self) -> None:
        out = self._gen([cand(), cand(), cand()]).generate(spec(), "8.9")
        assert len(out) == 1

    def test_budget_caps(self) -> None:
        raw = [cand(num_warps=w) for w in (4, 8, 16)] + [cand(block_size=8192)]
        out = self._gen(raw).generate(spec(), "8.9", budget=2)
        assert len(out) == 2


class TestProposeFnContract:
    def test_propose_fn_receives_history_and_n(self) -> None:
        captured = {}

        def fake(s, cc, n, h):
            captured["cc"] = cc
            captured["n"] = n
            captured["history"] = h
            return [cand()]

        hist = [HistoryEntry(SearchParams(4096, 8, 1, "fp16"), correct=False, median_us=None)]
        LLMGenerator(propose_fn=fake, default_n=7).generate(spec(), "6.1", history=hist)
        assert captured["cc"] == "6.1"
        assert captured["n"] == 7  # budget 未指定なら default_n
        assert captured["history"] == hist

    def test_budget_overrides_n(self) -> None:
        captured = {}

        def fake(s, cc, n, h):
            captured["n"] = n
            return []

        LLMGenerator(propose_fn=fake).generate(spec(), "8.9", budget=20)
        assert captured["n"] == 20


class TestHistoryEntry:
    def test_summary_incorrect(self) -> None:
        e = HistoryEntry(SearchParams(4096, 8, 1, "fp16"), correct=False, median_us=None)
        assert "INCORRECT" in e.summary()

    def test_summary_with_time(self) -> None:
        e = HistoryEntry(SearchParams(4096, 8, 1, "fp32"), correct=True, median_us=42.3)
        assert "42.3us" in e.summary()


class TestBuildPrompt:
    def test_includes_shape_and_constraints(self) -> None:
        p = build_prompt(spec(4096), "6.1", 12, [])
        assert "6.1" in p
        assert "(2048, 4096)" in p
        assert "two_pass" in p

    def test_includes_history(self) -> None:
        hist = [HistoryEntry(SearchParams(4096, 8, 1, "fp32"), correct=True, median_us=60.0)]
        p = build_prompt(spec(), "8.9", 12, hist)
        assert "60.0us" in p


class TestTokenUsage:
    def test_initial_zero(self) -> None:
        gen = LLMGenerator(propose_fn=lambda *a: [])
        assert gen.token_usage.input_tokens == 0
        assert gen.token_usage.output_tokens == 0
        assert gen.token_usage.total == 0

    def test_total_is_sum(self) -> None:
        usage = TokenUsage(input_tokens=100, output_tokens=50)
        assert usage.total == 150

    def test_add_from_response_with_usage(self) -> None:
        class FakeUsage:
            input_tokens = 300
            output_tokens = 150

        class FakeResp:
            usage = FakeUsage()

        gen = LLMGenerator(propose_fn=lambda *a: [])
        gen.token_usage._add_from_response(FakeResp())
        assert gen.token_usage.input_tokens == 300
        assert gen.token_usage.output_tokens == 150

    def test_add_from_response_accumulates(self) -> None:
        class FakeUsage:
            input_tokens = 100
            output_tokens = 50

        class FakeResp:
            usage = FakeUsage()

        gen = LLMGenerator(propose_fn=lambda *a: [])
        gen.token_usage._add_from_response(FakeResp())
        gen.token_usage._add_from_response(FakeResp())
        assert gen.token_usage.total == 300

    def test_add_from_response_no_usage_attr(self) -> None:
        class FakeResp:
            pass

        gen = LLMGenerator(propose_fn=lambda *a: [])
        gen.token_usage._add_from_response(FakeResp())  # should not raise
        assert gen.token_usage.total == 0

    def test_independent_per_generator_instance(self) -> None:
        g1 = LLMGenerator(propose_fn=lambda *a: [])
        g2 = LLMGenerator(propose_fn=lambda *a: [])
        g1.token_usage.input_tokens = 100
        assert g2.token_usage.input_tokens == 0


class TestSystemPromptGeneric:
    """_SYSTEM プロンプトが op_type 固定記述を含まないことを確認する (#278)。"""

    def test_system_prompt_does_not_mention_rmsnorm(self) -> None:
        assert "RMSNorm" not in _SYSTEM
        assert "rmsnorm" not in _SYSTEM.lower()

    def test_system_prompt_is_op_agnostic(self) -> None:
        # op_type 名を列挙する記述がないこと
        for op in ("softmax", "sdpa", "rms"):
            assert op not in _SYSTEM.lower()

    def test_build_prompt_uses_op_type_from_spec(self) -> None:
        softmax_spec = KernelSpec(
            op_type="softmax",
            input_specs=(TensorSpec((8, 512), torch.float16, True),),
            output_specs=(TensorSpec((8, 512), torch.float16, True),),
            constants={},
            graph_hash="softmax_v1",
            constraints=(),
        )
        p = build_prompt(softmax_spec, "8.9", 5, [])
        assert "Softmax" in p

    def test_build_prompt_uses_sdpa_op_type(self) -> None:
        sdpa_spec = KernelSpec(
            op_type="sdpa",
            input_specs=(TensorSpec((2, 8, 64, 64), torch.float16, True),),
            output_specs=(TensorSpec((2, 8, 64, 64), torch.float16, True),),
            constants={},
            graph_hash="sdpa_v1",
            constraints=(),
        )
        p = build_prompt(sdpa_spec, "8.9", 3, [])
        assert "Sdpa" in p


class TestModelGuard:
    """__init__ でのモデル名バリデーションを確認する (#278)。"""

    def test_default_model_is_accepted(self) -> None:
        # DEFAULT_MODEL は Anthropic SDK の既知モデルに含まれるはず
        gen = LLMGenerator(propose_fn=lambda *a: [])
        assert gen.model == DEFAULT_MODEL

    def test_invalid_model_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown Anthropic model"):
            LLMGenerator(model="claude-nonexistent-99-99", propose_fn=lambda *a: [])


class TestTimeout:
    """デフォルトクライアントに timeout が明示されることを確認する (#323)。"""

    def test_default_timeout(self) -> None:
        gen = LLMGenerator(propose_fn=lambda *a: [])
        assert gen.timeout == 60.0

    def test_timeout_passed_to_default_anthropic_client(self) -> None:
        from unittest.mock import MagicMock, patch

        gen = LLMGenerator(model=DEFAULT_MODEL, timeout=15.0)
        spec = KernelSpec(
            op_type="rmsnorm",
            input_specs=(TensorSpec(shape=(2048, 4096), dtype=torch.float16, is_contiguous=True),),
            output_specs=(TensorSpec(shape=(2048, 4096), dtype=torch.float16, is_contiguous=True),),
            constants={"eps": 1e-6},
            graph_hash="rmsnorm_v1",
            constraints=(),
        )
        with patch("anthropic.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_client.messages.parse.return_value = MagicMock(parsed_output=None)
            mock_anthropic.return_value = mock_client
            gen.generate(spec, "8.6", budget=1)
        mock_anthropic.assert_called_once_with(timeout=15.0)

    def test_injected_client_timeout_not_overridden(self) -> None:
        """client を自前で渡した場合は anthropic.Anthropic() が呼ばれない
        （= 注入クライアントの timeout 設定がそのまま使われる）。"""
        from unittest.mock import MagicMock, patch

        injected_client = MagicMock()
        injected_client.messages.parse.return_value = MagicMock(parsed_output=None)
        gen = LLMGenerator(model=DEFAULT_MODEL, client=injected_client, timeout=15.0)
        spec = KernelSpec(
            op_type="rmsnorm",
            input_specs=(TensorSpec(shape=(2048, 4096), dtype=torch.float16, is_contiguous=True),),
            output_specs=(TensorSpec(shape=(2048, 4096), dtype=torch.float16, is_contiguous=True),),
            constants={"eps": 1e-6},
            graph_hash="rmsnorm_v1",
            constraints=(),
        )
        with patch("anthropic.Anthropic") as mock_anthropic:
            gen.generate(spec, "8.6", budget=1)
        mock_anthropic.assert_not_called()
        injected_client.messages.parse.assert_called_once()


# (op_type, base_variant, block_size, 通るべきか)。N は op_spec 参照:
# gelu N=4096 / linear K=1024 / sdpa head_dim=64 / rmsnorm N=4096
_BLOCK_CASES = [
    ("gelu", "elementwise", 64, True),
    ("gelu", "elementwise", 1024, True),
    ("linear", "gemm", 64, True),
    ("linear", "gemm", 128, True),
    ("scaled_dot_product_attention", "flash", 16, True),
    ("scaled_dot_product_attention", "flash", 32, True),
    ("scaled_dot_product_attention", "flash_causal_opt", 16, True),
    ("scaled_dot_product_attention", "flash_causal_opt", 128, True),
    # tl.dot の最小タイル (16) 未満は無効
    ("scaled_dot_product_attention", "flash", 8, False),
    ("linear", "gemm", 8, False),
    # reduction variant の既存挙動（回帰）
    ("rmsnorm", "single_row", 4096, True),
    ("rmsnorm", "single_row", 2048, False),
    ("rmsnorm", "multi_row", 2048, False),
    ("rmsnorm", "two_pass", 1024, True),
    ("rmsnorm", "two_pass", 8192, False),
]


class TestCoerceByOpType:
    """非 reduction op（elementwise / gemm / flash）の候補が block_size < N でも通る。"""

    @pytest.mark.parametrize(("op", "variant", "block", "ok"), _BLOCK_CASES)
    def test_block_rule_per_variant(self, op: str, variant: str, block: int, ok: bool) -> None:
        raw = [cand(base_variant=variant, block_size=block)]
        out = LLMGenerator(propose_fn=lambda s, cc, n, h: raw).generate(op_spec(op), "8.9")
        assert len(out) == (1 if ok else 0)
        if ok:
            assert (out[0].variant, out[0].block_size) == (variant, block)

    @pytest.mark.parametrize("op", ["gelu", "linear", "scaled_dot_product_attention", "rmsnorm"])
    def test_search_space_candidates_all_accepted(self, op: str) -> None:
        # SearchSpace が列挙する候補は LLM 経路（_coerce）でも全件通る。
        s = op_spec(op)
        space = list(SearchSpace().enumerate(s, "8.9"))
        raw = [{**p.to_dict(), "base_variant": p.variant, "hypothesis": "space"} for p in space]
        out = LLMGenerator(propose_fn=lambda *a: raw).generate(s, "8.9", budget=len(raw))

        def key(p: SearchParams) -> tuple:
            return (p.variant, p.block_size, p.num_warps, p.num_stages, p.acc_dtype, p.block_k)

        assert {key(p) for p in out} == {key(p) for p in space}


class TestBlockK:
    """gemm の block_k を _coerce・プロンプト・構造化出力スキーマが扱う（#502）。"""

    def _coerce_one(self, op: str, **kw) -> list[SearchParams]:
        raw = [cand(**kw)]
        return LLMGenerator(propose_fn=lambda *a: raw).generate(op_spec(op), "8.9")

    @pytest.mark.parametrize("bk", [16, 32, 64])
    def test_gemm_block_k_is_taken(self, bk: int) -> None:
        out = self._coerce_one("linear", base_variant="gemm", block_size=64, block_k=bk)
        assert [p.block_k for p in out] == [bk]

    def test_gemm_block_k_default_is_32(self) -> None:
        out = self._coerce_one("linear", base_variant="gemm", block_size=64)
        assert [p.block_k for p in out] == [32]

    @pytest.mark.parametrize("bk", [8, 4, 1, 24, 0, -16])
    def test_gemm_invalid_block_k_is_dropped(self, bk: int) -> None:
        # tl.dot の K 次元は 16 以上の 2 のべき乗
        assert not self._coerce_one("linear", base_variant="gemm", block_size=64, block_k=bk)

    def test_non_gemm_block_k_is_ignored(self) -> None:
        # block_k はテンプレートで使われない。取り込むと dedup をすり抜けて budget を食う。
        raw = [
            cand(base_variant="flash", block_size=32, block_k=16),
            cand(base_variant="flash", block_size=32, block_k=64),
        ]
        out = LLMGenerator(propose_fn=lambda *a: raw).generate(
            op_spec("scaled_dot_product_attention"), "8.9"
        )
        assert [p.block_k for p in out] == [32]

    def test_prompt_mentions_block_k_for_linear_only(self) -> None:
        assert "block_k" in build_prompt(op_spec("linear"), "8.9", 5, [])
        assert "block_k" not in build_prompt(op_spec("gelu"), "8.9", 5, [])
        assert "block_k" not in build_prompt(spec(), "8.9", 5, [])

    def test_prompt_history_shows_block_k_for_gemm(self) -> None:
        # block_k は HistoryEntry.summary() が出す。build_prompt は summary() をそのまま使う。
        hist = [
            HistoryEntry(
                SearchParams(block_size=64, num_warps=4, num_stages=2, variant="gemm", block_k=64),
                True,
                12.0,
            )
        ]
        prompt = build_prompt(op_spec("linear"), "8.9", 5, hist)
        assert f"  {hist[0].summary()}" in prompt.splitlines()
        assert prompt.count("block_k=64") == 1

    def test_summary_shows_block_k_for_gemm_only(self) -> None:
        gemm = SearchParams(block_size=64, num_warps=4, num_stages=2, variant="gemm", block_k=16)
        flash = SearchParams(block_size=64, num_warps=4, num_stages=2, variant="flash")
        assert "block_k=16" in HistoryEntry(gemm, True, 1.0).summary()
        assert "block_k" not in HistoryEntry(flash, True, 1.0).summary()

    def test_gemm_block_k_null_falls_back_to_default(self) -> None:
        # LLM が block_k: null を返しても候補ごと捨てず既定 32 にする（#531 レビュー）
        out = self._coerce_one("linear", base_variant="gemm", block_size=64, block_k=None)
        assert [p.block_k for p in out] == [32]

    def test_gemm_non_numeric_block_k_is_still_dropped(self) -> None:
        assert not self._coerce_one("linear", base_variant="gemm", block_size=64, block_k="x")

    def test_proposal_schema_has_block_k(self) -> None:
        pytest.importorskip("pydantic")
        from forge.search._proposal_models import Candidate

        assert "block_k" in Candidate.model_fields
        c = Candidate(
            base_variant="gemm",
            block_size=64,
            num_warps=4,
            num_stages=2,
            acc_dtype="fp32",
            rows_per_program=1,
            hypothesis="h",
        )
        assert c.block_k == 32  # 未指定でも既存の出力は読める


class TestBayesianCandidatesSurviveCoerce:
    """Bayesian が提案しうる (block, variant) 全組合せのうち _block_ok を通るものは _coerce も通る。

    optuna 非依存（候補空間の関数だけを使う）。
    """

    @pytest.mark.parametrize("op", ["gelu", "linear", "scaled_dot_product_attention", "rmsnorm"])
    def test_prefiltered_candidates_not_dropped(self, op: str) -> None:
        s = op_spec(op)
        n = s.input_specs[0].shape[-1]
        blocks = _candidate_blocks(
            n, op == "gelu", op == "scaled_dot_product_attention", op == "linear"
        )
        pre = [
            cand(base_variant=v, block_size=b)
            for v in _candidate_variants(op)
            for b in blocks
            if _block_ok(b, v, n)
        ]
        assert pre  # Bayesian の事前フィルタを通る候補が 1 件以上ある
        out = LLMGenerator(propose_fn=lambda *a: pre).generate(s, "8.9", budget=len(pre))
        assert len(out) == len(pre)  # _coerce が追加で捨てない（食い違いなし）


class TestBlockRuleSingleSource:
    @pytest.mark.parametrize("variant", SUPPORTED_VARIANTS)
    @pytest.mark.parametrize("n", [16, 64, 4096])
    @pytest.mark.parametrize("block", [8, 16, 64, 1024, 8192])
    def test_valid_block_matches_block_ok(self, variant: str, n: int, block: int) -> None:
        params = SearchParams(
            block_size=block, num_warps=4, num_stages=1, variant=variant, rows_per_program=1
        )
        assert _valid_block(params, n) == _block_ok(block, variant, n)


def _sdpa_spec(s: int, d: int = 64) -> KernelSpec:
    qkv = _f16((8, s, d))
    return KernelSpec(
        "scaled_dot_product_attention",
        (qkv, qkv, qkv),
        (qkv,),
        {"is_causal": False},
        "sdpa_v1",
        (),
    )


class TestBayesianAttentionBlocksDependOnSeqLen:
    """Bayesian の attention 候補は S・head_dim を見て弾く（#532）。optuna 非依存。

    block_ok の flash 系は block >= 16 しか見ないので、S 超過や head_dim % 16 != 0 は
    _block_ok_for_spec が追加で弾く。S < 16 でも SearchSpace と同じく 16 は 1 つ残す。
    """

    @staticmethod
    def _valid_blocks(s: int, d: int = 64) -> list[int]:
        sp = _sdpa_spec(s, d)
        blocks = _candidate_blocks(d, False, True, False)
        return sorted(
            {
                b
                for v in _candidate_variants(sp.op_type)
                for b in blocks
                if _block_ok_for_spec(sp, b, v)
            }
        )

    @pytest.mark.parametrize("s", [1, 8, 15, 16, 17, 32, 64, 100, 4096])
    def test_blocks_never_exceed_seq_len_and_stay_non_empty(self, s: int) -> None:
        valid = self._valid_blocks(s)
        assert valid  # S が小さくても候補は空にならない
        assert all(16 <= b <= max(s, 16) for b in valid)

    @pytest.mark.parametrize("s", [16, 32, 64, 128])
    def test_matches_search_space(self, s: int) -> None:
        # Bayesian が出す block は SearchSpace が列挙する block の集合に含まれる
        space = {p.block_size for p in SearchSpace().enumerate(_sdpa_spec(s), "8.9")}
        assert set(self._valid_blocks(s)) >= space

    @pytest.mark.parametrize("d", [8, 24, 60, 100])
    def test_head_dim_not_multiple_of_16_is_rejected(self, d: int) -> None:
        assert self._valid_blocks(64, d) == []

    def test_non_attention_ops_ignore_seq_len(self) -> None:
        # attention 以外は従来どおり block_ok だけ
        s = op_spec("linear")
        assert _block_ok_for_spec(s, 64, "gemm") == _block_ok(64, "gemm", 1024)
        assert _block_ok_for_spec(s, 8, "gemm") is False
