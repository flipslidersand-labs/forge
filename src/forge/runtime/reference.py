"""runtime/reference.py — reference / baseline 関数の後方互換エントリポイント。

実装は ops/registry.py の OP_REGISTRY に移管済み。
このモジュールは既存テスト・コードの import 互換を維持するために残している。
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from forge.ops.registry import OP_REGISTRY

# --- 後方互換 dict ---

REFERENCE_IMPLS: dict[str, Callable[..., torch.Tensor]] = {
    op: defn.reference_fn for op, defn in OP_REGISTRY.items()
}

BASELINE_IMPLS: dict[str, Callable[..., torch.Tensor]] = {
    op: defn.baseline_fn for op, defn in OP_REGISTRY.items()
}

# --- 後方互換関数 ---


def get_reference(op_type: str) -> Callable[..., torch.Tensor]:
    """正確性検証用の ground truth 実装を返す。"""
    if op_type not in OP_REGISTRY:
        raise ValueError(f"No reference implementation for op_type={op_type!r}")
    return OP_REGISTRY[op_type].reference_fn


def get_baseline(op_type: str) -> Callable[..., torch.Tensor]:
    """速度比較の対抗馬（PyTorch 最適化実装）を返す。"""
    if op_type not in OP_REGISTRY:
        raise ValueError(f"No baseline implementation for op_type={op_type!r}")
    return OP_REGISTRY[op_type].baseline_fn


def baseline_name(op_type: str) -> str:
    if op_type not in OP_REGISTRY:
        return "fp32_reference"
    return OP_REGISTRY[op_type].baseline_display_name


# --- 個別関数エイリアス（既存テストの直接 import 互換） ---

rmsnorm_reference = REFERENCE_IMPLS.get("rmsnorm")
softmax_reference = REFERENCE_IMPLS.get("softmax")
layernorm_reference = REFERENCE_IMPLS.get("layernorm")
gelu_reference = REFERENCE_IMPLS.get("gelu")
sdpa_reference = REFERENCE_IMPLS.get("scaled_dot_product_attention")
linear_reference = REFERENCE_IMPLS.get("linear")
