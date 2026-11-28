"""OP_REGISTRY の primary_input_fn / correctness_cases_fn の出力を固定する特性化テスト。

#360 (ops/registry.py の同型ヘルパー共通化) のリファクタで挙動が変わらないことを
確認するための安全網。両関数の戻り値はテンソルではなく
{shape, dtype, init, scale, seed} の辞書(と case 名)なので、JSON にそのまま保存して
完全一致で比較する。GPU・乱数・テンソル生成は一切使わない。

fixture: tests/data/registry_golden.json

再生成手順(registry.py の出力を意図して変えたときだけ行い、差分をレビューすること)::

    FORGE_UPDATE_GOLDEN=1 python -m pytest tests/test_registry_golden.py

環境変数を付けると fixture を現行コードの出力で上書きしてからテストを実行する。
上書き後の diff が意図どおりかを `git diff tests/data/registry_golden.json` で確認する。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
import torch

from forge.ir.kernel_spec import KernelSpec
from forge.ir.tensor_spec import TensorSpec
from forge.ops.registry import OP_REGISTRY

FIXTURE_PATH = Path(__file__).parent / "data" / "registry_golden.json"
UPDATE_ENV = "FORGE_UPDATE_GOLDEN"

_DTYPES = {"float16": torch.float16, "float32": torch.float32}

# op -> 入力 shape 群。値は意図的に互いに異なる数にし(m=5, n=48 など)、
# どの引数がどの shape 位置に流れたかが fixture から読み取れるようにする。
# cases が override する行数(2048/1/7/64/8)・bh(1/4) とも衝突させない。
_OP_SHAPES: dict[str, list[tuple[int, ...]]] = {
    "rmsnorm": [(5, 48), (48,)],
    "softmax": [(5, 48)],
    "layernorm": [(5, 48), (48,), (48,)],
    "fused_add_rmsnorm": [(5, 48), (5, 48), (48,)],
    "gelu": [(5, 48)],
    "rope": [(5, 48), (5, 48), (5, 48)],
    "swiglu": [(5, 48), (5, 48)],
    # (bh, s, d)。bh=3 は cases の 1 / 4 と区別できる値。
    "scaled_dot_product_attention": [(3, 32, 16), (3, 32, 16), (3, 32, 16)],
    # x(m,k) / weight(n,k) / bias(n)。n=24 は input_specs[1] 由来であることの確認用。
    "linear": [(5, 48), (24, 48), (24,)],
}

_NINE_OPS = tuple(_OP_SHAPES)


def _make_spec(op: str, dtype_name: str) -> KernelSpec:
    dtype = _DTYPES[dtype_name]
    ins = tuple(TensorSpec(shape, dtype, True) for shape in _OP_SHAPES[op])
    return KernelSpec(
        op_type=op,
        input_specs=ins,
        output_specs=(ins[0],),
        constants={},
        graph_hash=f"registry_golden_{op}",
        constraints=(),
    )


def _entry_id(op: str, dtype_name: str) -> str:
    return f"{op}/{dtype_name}"


def _all_ids() -> list[tuple[str, str]]:
    return [(op, dt) for op in _NINE_OPS for dt in _DTYPES]


def _jsonable(obj: Any) -> Any:
    """tuple を list にするなど、JSON 往復後と同じ形に正規化する。"""
    return json.loads(json.dumps(obj))


def _current_output(op: str, dtype_name: str) -> dict[str, Any]:
    spec = _make_spec(op, dtype_name)
    op_def = OP_REGISTRY[op]
    return {
        "spec": {
            "input_shapes": [list(s) for s in _OP_SHAPES[op]],
            "dtype": dtype_name,
        },
        "primary_input": op_def.primary_input_fn(spec),
        "correctness_cases": op_def.correctness_cases_fn(spec),
    }


def _build_golden() -> dict[str, Any]:
    return {_entry_id(op, dt): _jsonable(_current_output(op, dt)) for op, dt in _all_ids()}


def _load_golden() -> dict[str, Any]:
    if os.environ.get(UPDATE_ENV) == "1":
        FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
        FIXTURE_PATH.write_text(
            json.dumps(_build_golden(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


GOLDEN = _load_golden()


def test_fixture_covers_exactly_the_nine_ops() -> None:
    assert set(GOLDEN) == {_entry_id(op, dt) for op, dt in _all_ids()}
    assert {op for op, _ in _all_ids()} <= set(OP_REGISTRY)
    assert len(_NINE_OPS) == 9


@pytest.mark.parametrize(("op", "dtype_name"), _all_ids(), ids=lambda v: str(v))
def test_primary_and_cases_match_golden(op: str, dtype_name: str) -> None:
    expected = GOLDEN[_entry_id(op, dtype_name)]
    actual = _jsonable(_current_output(op, dtype_name))
    assert actual["spec"] == expected["spec"]
    assert actual["primary_input"] == expected["primary_input"]
    assert [c["name"] for c in actual["correctness_cases"]] == [
        c["name"] for c in expected["correctness_cases"]
    ]
    assert actual["correctness_cases"] == expected["correctness_cases"]


# --- fixture が「ほぼ同型に見えて挙動が違う箇所」を含んでいることの明示確認 ---
# (golden 比較だけだと、再生成で気づかず消える恐れがあるため fixture 側の値を直接見る)


def _primary(op: str, dt: str = "float16") -> list[dict[str, Any]]:
    return GOLDEN[_entry_id(op, dt)]["primary_input"]


def _cases(op: str, dt: str = "float16") -> dict[str, list[dict[str, Any]]]:
    return {c["name"]: c["input_specs"] for c in GOLDEN[_entry_id(op, dt)]["correctness_cases"]}


class TestFixtureContainsBehaviourDifferences:
    def test_rope_ignores_x_scale_and_uses_randn_cos_sin(self) -> None:
        p = _primary("rope")
        assert [t["scale"] for t in p] == [1.0, 1.0, 1.0]
        assert [t["init"] for t in p] == ["randn", "randn", "randn"]
        assert [t["seed"] for t in p] == [0, 1, 2]
        # "large_values" という名前だが x の scale は 1.0 のまま(現行挙動の固定。#360 の共通化で変えない)
        assert _cases("rope")["large_values"][0]["scale"] == 1.0
        assert _cases("rope")["zeros"][0]["init"] == "zeros"

    def test_linear_three_dim_args_default_scales_and_n_from_input_specs(self) -> None:
        p = _primary("linear")
        assert [t["shape"] for t in p] == [[5, 48], [24, 48], [24]]
        assert [t["scale"] for t in p] == [0.1, 0.1, 0.0]
        assert [t["init"] for t in p] == ["randn", "randn", "zeros"]
        assert [t["seed"] for t in p] == [0, 1, 2]
        cases = _cases("linear")
        assert cases["single_row"][0]["shape"] == [1, 48]
        assert cases["odd_rows"][0]["shape"] == [7, 48]
        assert cases["large_values"][0]["shape"] == [64, 48]
        assert cases["large_values"][0]["scale"] == 10.0
        assert cases["zeros_bias"][2]["init"] == "zeros"
        # m 以外の n=24 は全 case で保たれる
        assert all(inp[1]["shape"] == [24, 48] for inp in cases.values())

    def test_sdpa_shape_is_bh_s_d_and_scale_applies_to_all_three(self) -> None:
        p = _primary("scaled_dot_product_attention")
        assert [t["shape"] for t in p] == [[3, 32, 16]] * 3
        assert [t["scale"] for t in p] == [1.0, 1.0, 1.0]
        cases = _cases("scaled_dot_product_attention")
        assert cases["single_batch"][0]["shape"] == [1, 32, 16]
        assert cases["multi_batch"][0]["shape"] == [4, 32, 16]
        assert [t["scale"] for t in cases["large_values"]] == [10.0, 10.0, 10.0]
        assert [t["seed"] for t in cases["large_values"]] == [7, 8, 9]

    def test_layernorm_bias_defaults_to_zeros(self) -> None:
        p = _primary("layernorm")
        assert [t["init"] for t in p] == ["randn", "ones", "zeros"]
        assert _cases("layernorm")["wb_randn"][2]["init"] == "randn"

    def test_swiglu_and_fused_add_rmsnorm_use_distinct_kw_names(self) -> None:
        # swiglu: x_init / g_init (gate)。zeros case は両方 zeros
        sw = _cases("swiglu")["zeros"]
        assert [t["init"] for t in sw] == ["zeros", "zeros"]
        # fused_add_rmsnorm: x_init / r_init (residual) / w_init。zeros_x は x のみ zeros
        fa = _cases("fused_add_rmsnorm")
        assert [t["init"] for t in fa["zeros_x"]] == ["zeros", "randn", "ones"]
        assert [t["init"] for t in fa["weight_randn"]] == ["randn", "randn", "randn"]

    def test_cases_override_row_counts(self) -> None:
        for op in (
            "rmsnorm",
            "softmax",
            "layernorm",
            "fused_add_rmsnorm",
            "gelu",
            "rope",
            "swiglu",
        ):
            cases = _cases(op)
            rows = {name: inp[0]["shape"][0] for name, inp in cases.items()}
            assert rows["basic"] == 2048
            assert rows["single_row"] == 1
            assert rows["odd_rows"] == 7
            zeros_name = "zeros_x" if op == "fused_add_rmsnorm" else "zeros"
            assert rows[zeros_name] == 8
