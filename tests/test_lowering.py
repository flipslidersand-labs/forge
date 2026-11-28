"""lowering の CPU テスト（torch.fx は CPU で動くため GPU 不要）。"""

import pytest
import torch

from forge.lowering import identify
from forge.lowering.registry import OpPattern, graph_op_counts, match_counts


def rmsnorm(x, weight, eps=1e-6):
    return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps) * weight


class TestIdentify:
    def test_recognizes_rmsnorm(self) -> None:
        assert identify(rmsnorm) == "rmsnorm"

    def test_recognizes_softmax(self) -> None:
        # dim は定数（torch.softmax は trace 時に concrete int を要求する）
        def softmax(x):
            return torch.softmax(x, dim=-1)

        assert identify(softmax) == "softmax"

    def test_recognizes_layernorm(self) -> None:
        import torch.nn.functional as F

        def layernorm(x, weight, bias):
            return F.layer_norm(x, (4096,), weight, bias, 1e-5)

        assert identify(layernorm) == "layernorm"

    def test_recognizes_fused_add_rmsnorm(self) -> None:
        def fused_add_rmsnorm(x, residual, weight, eps=1e-6):
            hidden = x + residual
            rms = torch.rsqrt(torch.mean(hidden * hidden, dim=-1, keepdim=True) + eps)
            return hidden * rms * weight

        assert identify(fused_add_rmsnorm) == "fused_add_rmsnorm"

    def test_recognizes_gelu(self) -> None:
        import torch.nn.functional as F

        def gelu(x):
            return F.gelu(x)

        assert identify(gelu) == "gelu"

    def test_recognizes_swiglu(self) -> None:
        import torch.nn.functional as F

        def swiglu(x, gate):
            return F.silu(gate) * x

        assert identify(swiglu) == "swiglu"

    def test_recognizes_rope(self) -> None:
        def rotate_half(x):
            h = x.shape[-1] // 2
            return torch.cat([-x[..., h:], x[..., :h]], dim=-1)

        def apply_rope(x, cos, sin):
            return (x * cos) + (rotate_half(x) * sin)

        assert identify(apply_rope) == "rope"

    def test_unrelated_op_returns_none(self) -> None:
        def relu(x):
            return torch.relu(x)

        assert identify(relu) is None

    def test_unrelated_fn_returns_none(self) -> None:
        def add(x, y):
            return x + y

        assert identify(add) is None

    def test_untraceable_returns_none(self) -> None:
        # データ依存の制御フローは symbolic_trace 不能 → None
        def dynamic(x):
            if x.sum() > 0:
                return x * 2
            return x

        assert identify(dynamic) is None


class TestSdpaIdentify:
    def test_recognizes_sdpa(self) -> None:
        import torch.nn.functional as F

        def attn(q, k, v):
            return F.scaled_dot_product_attention(q, k, v)

        assert identify(attn) == "scaled_dot_product_attention"

    def test_recognizes_sdpa_causal_passed_as_argument(self) -> None:
        # is_causal を引数で受ける形は constants 経由でカーネルに渡るので対応のまま
        import torch.nn.functional as F

        def attn_causal(q, k, v, is_causal=False):
            return F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)

        assert identify(attn_causal) == "scaled_dot_product_attention"

    def test_sdpa_pattern_count(self) -> None:
        from collections import Counter

        import torch.nn.functional as F

        def attn(q, k, v):
            return F.scaled_dot_product_attention(q, k, v)

        graph = torch.fx.symbolic_trace(attn).graph
        counts = graph_op_counts(graph)
        assert counts == Counter({"scaled_dot_product_attention": 1})


class TestRegistry:
    def test_graph_op_counts_rmsnorm(self) -> None:
        graph = torch.fx.symbolic_trace(rmsnorm).graph
        counts = graph_op_counts(graph)
        assert counts == {"mul": 3, "mean": 1, "add": 1, "rsqrt": 1}

    def test_match_counts_hit(self) -> None:
        from collections import Counter

        assert match_counts(Counter({"mul": 3, "mean": 1, "add": 1, "rsqrt": 1})) == "rmsnorm"

    def test_match_counts_miss(self) -> None:
        from collections import Counter

        assert match_counts(Counter({"relu": 1})) is None

    def test_pattern_matches_exact_multiset(self) -> None:
        from collections import Counter

        p = OpPattern(op_type="x", op_counts={"mul": 2})
        assert p.matches(Counter({"mul": 2}))
        assert not p.matches(Counter({"mul": 3}))  # 個数が違えば不一致


class TestLiteralArgsFallBack:
    """#472: op 名が同じでも reference と意味が変わるリテラル引数は None（eager）にする。"""

    def test_softmax_dim0_literal_is_none(self) -> None:
        def f(x):
            return torch.softmax(x, dim=0)

        assert identify(f) is None

    def test_softmax_positional_dim_is_none(self) -> None:
        def f(x):
            return torch.softmax(x, 0)

        assert identify(f) is None

    def test_softmax_functional_dim0_is_none(self) -> None:
        import torch.nn.functional as F

        def f(x):
            return F.softmax(x, dim=0)

        assert identify(f) is None

    def test_softmax_functional_dim_last_still_recognized(self) -> None:
        import torch.nn.functional as F

        def f(x):
            return F.softmax(x, dim=-1)

        assert identify(f) == "softmax"

    def test_gelu_tanh_literal_is_none(self) -> None:
        import torch.nn.functional as F

        def f(x):
            return F.gelu(x, approximate="tanh")

        assert identify(f) is None

    def test_gelu_approximate_none_still_recognized(self) -> None:
        import torch.nn.functional as F

        def f(x):
            return F.gelu(x, approximate="none")

        assert identify(f) == "gelu"

    def test_sdpa_literal_is_causal_true_is_none(self) -> None:
        import torch.nn.functional as F

        def f(q, k, v):
            return F.scaled_dot_product_attention(q, k, v, is_causal=True)

        assert identify(f) is None

    def test_sdpa_literal_is_causal_false_still_recognized(self) -> None:
        import torch.nn.functional as F

        def f(q, k, v):
            return F.scaled_dot_product_attention(q, k, v, is_causal=False)

        assert identify(f) == "scaled_dot_product_attention"

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"scale": 0.5},
            {"dropout_p": 0.5},
            {"enable_gqa": True},
        ],
        ids=["scale", "dropout_p", "enable_gqa"],
    )
    def test_sdpa_non_default_literal_is_none(self, kwargs) -> None:
        import torch.nn.functional as F

        def f(q, k, v):
            return F.scaled_dot_product_attention(q, k, v, **kwargs)

        assert identify(f) is None

    def test_sdpa_non_default_positional_literals_are_none(self) -> None:
        import torch.nn.functional as F

        # (q, k, v, attn_mask, dropout_p, is_causal, scale, enable_gqa)
        def dropout(q, k, v):
            return F.scaled_dot_product_attention(q, k, v, None, 0.5)

        def scale(q, k, v):
            return F.scaled_dot_product_attention(q, k, v, None, 0.0, False, 0.5)

        def gqa(q, k, v):
            return F.scaled_dot_product_attention(q, k, v, None, 0.0, False, None, True)

        assert identify(dropout) is None
        assert identify(scale) is None
        assert identify(gqa) is None

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"attn_mask": None},
            {"dropout_p": 0.0},
            {"dropout_p": 0},
            {"scale": None},
            {"enable_gqa": False},
        ],
        ids=["attn_mask_none", "dropout_float0", "dropout_int0", "scale_none", "gqa_false"],
    )
    def test_sdpa_default_literal_still_recognized(self, kwargs) -> None:
        import torch.nn.functional as F

        def f(q, k, v):
            return F.scaled_dot_product_attention(q, k, v, **kwargs)

        assert identify(f) == "scaled_dot_product_attention"

    def test_sdpa_attn_mask_closure_tensor_is_none(self) -> None:
        import torch.nn.functional as F

        # 非 None のリテラル float は trace 自体が失敗して None になる。
        # クロージャのテンソルは fx で get_attr ノードになり trace は通る。マスクは
        # カーネルに渡らないので、一致させると結果が静かに変わる。
        mask = torch.ones(4, 4, dtype=torch.bool)

        def f(q, k, v):
            return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

        assert identify(f) is None

    def test_sdpa_attn_mask_argument_is_none(self) -> None:
        import torch.nn.functional as F

        # 既定値が None でも、呼び出し時に渡される可能性があるので静的には None と言えない
        def f(q, k, v, mask=None):
            return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

        assert identify(f) is None

    def test_layernorm_multi_dim_normalized_shape_is_none(self) -> None:
        import torch.nn.functional as F

        def f(x, weight, bias):
            return F.layer_norm(x, (8, 4096), weight, bias, 1e-5)

        assert identify(f) is None

    def test_graph_op_counts_marks_unsupported_literal(self) -> None:
        def f(x):
            return torch.softmax(x, dim=0)

        counts = graph_op_counts(torch.fx.symbolic_trace(f).graph)
        assert counts == {"softmax!literal": 1}


class TestKnownLimits:
    """リテラル検査では防げない既知の限界を strict xfail で固定する（直ったら XPASS で気づける）。"""

    @pytest.mark.xfail(
        strict=True,
        reason="#472: identify() は静的照合のため弾けない(実行時は decorator が出力照合で eager に倒す)",
    )
    def test_rmsnorm_lookalike_with_different_expression_is_none(self) -> None:
        # mean(x) （x*x ではない）を使う。{mul:3, mean:1, add:1, rsqrt:1} は同じ
        def f(x, weight, eps=1e-6):
            return x * x * weight * torch.rsqrt(torch.mean(x, dim=-1) + eps)

        assert identify(f) is None

    @pytest.mark.xfail(
        strict=True,
        reason="#472: normalized_shape が実入力の最終次元かは trace 時に不明(実行時は出力照合で eager)",
    )
    def test_layernorm_single_dim_shape_is_not_checked_against_input(self) -> None:
        import torch.nn.functional as F

        # 4096 は入力の最終次元とは限らない。静的には区別できないので現状は一致する
        def f(x, weight, bias):
            return F.layer_norm(x, (7,), weight, bias, 1e-5)

        assert identify(f) is None
