"""OP_REGISTRY / OP_INFO 整合性テスト (#205)。"""

from __future__ import annotations


def test_op_info_and_registry_keys_match() -> None:
    """OP_INFO と OP_REGISTRY のキーセットが一致することを確認。

    片方だけに op を追加した場合はここで検知される。
    """
    from forge.ops import OP_INFO
    from forge.ops.registry import OP_REGISTRY

    assert set(OP_INFO.keys()) == set(OP_REGISTRY.keys()), (
        f"OP_INFO keys:     {sorted(OP_INFO.keys())}\n"
        f"OP_REGISTRY keys: {sorted(OP_REGISTRY.keys())}"
    )


def test_op_registry_kind_field() -> None:
    """OP_REGISTRY の各エントリが有効な kind を持つ。"""
    from forge.ops.registry import OP_REGISTRY

    valid_kinds = {"reduction", "elementwise", "matmul", "gemm"}
    for op, defn in OP_REGISTRY.items():
        assert defn.kind in valid_kinds, f"{op}: kind={defn.kind!r} not in {valid_kinds}"


def test_op_registry_n_tensor_inputs_positive() -> None:
    """OP_REGISTRY の各エントリが正の n_tensor_inputs を持つ。"""
    from forge.ops.registry import OP_REGISTRY

    for op, defn in OP_REGISTRY.items():
        assert defn.n_tensor_inputs > 0, f"{op}: n_tensor_inputs={defn.n_tensor_inputs}"


def test_is_elementwise_uses_registry() -> None:
    """is_elementwise が OP_REGISTRY の kind を参照する。"""
    from forge.ops import is_elementwise
    from forge.ops.registry import OP_REGISTRY

    for op, defn in OP_REGISTRY.items():
        assert is_elementwise(op) == (defn.kind == "elementwise"), f"{op}: is_elementwise mismatch"


def test_is_matmul_uses_registry() -> None:
    from forge.ops import is_matmul
    from forge.ops.registry import OP_REGISTRY

    for op, defn in OP_REGISTRY.items():
        assert is_matmul(op) == (defn.kind == "matmul"), f"{op}: is_matmul mismatch"


def test_is_gemm_uses_registry() -> None:
    from forge.ops import is_gemm
    from forge.ops.registry import OP_REGISTRY

    for op, defn in OP_REGISTRY.items():
        assert is_gemm(op) == (defn.kind == "gemm"), f"{op}: is_gemm mismatch"


def test_op_info_derived_from_registry() -> None:
    """OP_INFO は OP_REGISTRY から動的生成され値が一致する。"""
    from forge.ops import OP_INFO
    from forge.ops.registry import OP_REGISTRY

    for op, defn in OP_REGISTRY.items():
        info = OP_INFO[op]
        assert info.kind == defn.kind, f"{op}: kind mismatch"
        assert info.n_tensor_inputs == defn.n_tensor_inputs, f"{op}: n_tensor_inputs mismatch"
