from __future__ import annotations

from typing import Any

from forge.ir.kernel_spec import KernelSpec
from forge.ops.registry import OP_REGISTRY


def primary_input(spec: KernelSpec) -> list[dict[str, Any]]:
    """ベンチマークに使う代表入力（spec の shape そのまま）。"""
    if spec.op_type not in OP_REGISTRY:
        raise ValueError(f"primary_input: unsupported op_type {spec.op_type!r}")
    return OP_REGISTRY[spec.op_type].primary_input_fn(spec)


def correctness_cases(spec: KernelSpec) -> list[dict[str, Any]]:
    """正確性検証のエッジケース群。"""
    if spec.op_type not in OP_REGISTRY:
        raise ValueError(f"correctness_cases: unsupported op_type {spec.op_type!r}")
    return OP_REGISTRY[spec.op_type].correctness_cases_fn(spec)
