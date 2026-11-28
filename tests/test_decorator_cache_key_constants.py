"""decorator の in-process キャッシュキーが constants を含むことの検証 (#470)。GPU 不要。

SDPA の is_causal のようにレンダリング時に焼き込まれる定数が違う呼び出しで、
同 shape でも別カーネルが使われること(最初のカーネルの使い回しで結果が静かに
間違わないこと)を、_build を差し替えて確認する。
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import torch

from forge.decorator import optimize


def _fake_cuda_tensor(shape=(8, 64, 64), dtype=torch.float16):
    t = MagicMock(spec=torch.Tensor)
    t.is_cuda = True
    t.shape = shape
    t.dtype = dtype
    t.device = "cuda:0"
    t.is_contiguous.return_value = True
    return t


def _run(calls):
    """calls: [(is_causal 指定 kwargs)] を順に呼び、(_build に渡った constants 一覧, 結果一覧) を返す。"""
    built: list[dict] = []

    def fake_build(fn, op_type, tensors, constants, *rest):
        built.append(dict(constants))
        causal = constants["is_causal"]
        return lambda *ts, **kw: f"kernel(is_causal={causal})"

    with (
        patch("forge.decorator.identify", return_value="sdpa"),
        patch("forge.decorator._build", side_effect=fake_build),
    ):

        @optimize(budget=1)
        def attn(q, k, v, is_causal=False):
            return "eager"

        q, k, v = _fake_cuda_tensor(), _fake_cuda_tensor(), _fake_cuda_tensor()
        results = [attn(q, k, v, **kw) for kw in calls]
    return built, results


def test_different_constants_same_shape_build_separately():
    built, results = _run([{"is_causal": False}, {"is_causal": True}])
    assert len(built) == 2
    assert results == ["kernel(is_causal=False)", "kernel(is_causal=True)"]


def test_same_constants_same_shape_build_once():
    built, results = _run([{}, {"is_causal": False}, {"is_causal": False}])
    # 省略 (既定値 False) と明示 False は同じキー
    assert len(built) == 1
    assert results == ["kernel(is_causal=False)"] * 3
