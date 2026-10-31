"""forge.prewarm() のユニットテスト (#215)。GPU 不要。"""

from __future__ import annotations

import pytest
import torch


def test_prewarm_import_does_not_require_gpu() -> None:
    """forge.prewarm は GPU なしで import できる。"""
    from forge import prewarm  # noqa: F401 — import だけ確認

    assert callable(prewarm)


def test_prewarm_raises_type_error_for_non_forge_fn() -> None:
    """@forge.optimize 未適用の関数を渡すと TypeError。"""
    from forge import prewarm

    def plain_fn(x: torch.Tensor) -> torch.Tensor:
        return x

    with pytest.raises(TypeError, match="_forge_compiled"):
        prewarm(plain_fn, [(4, 4)], dtype=torch.float32, device="cpu")


def test_prewarm_calls_fn_with_zero_tensors(monkeypatch) -> None:
    """prewarm() が指定 shape のゼロテンソルで fn を呼び出す。"""
    from forge import optimize, prewarm

    called_with: list[tuple] = []

    @optimize(budget=1)
    def dummy(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        called_with.append((x.shape, w.shape))
        return x + w

    # eager fallback（CPU テンソル → GPU 不要）。fx トレーシング経由で複数回 call されることがある
    prewarm(dummy, [(4, 8), (8,)], dtype=torch.float32, device="cpu")

    # 実テンソルでの呼び出しが含まれることを確認（fx Proxy によるトレース分は除外）
    real_calls = [c for c in called_with if isinstance(c[0], torch.Size)]
    assert real_calls == [(torch.Size([4, 8]), torch.Size([8]))]


def test_prewarm_progress_callback_called() -> None:
    """progress コールバックが prewarm 中に呼ばれる。"""
    from forge import optimize, prewarm

    msgs: list[str] = []

    @optimize(budget=1)
    def dummy(x: torch.Tensor) -> torch.Tensor:
        return x

    prewarm(dummy, [(4,)], dtype=torch.float32, device="cpu", progress=msgs.append)

    assert any("prewarm" in m for m in msgs)


def test_prewarm_handles_exception_without_raising() -> None:
    """fn が例外を投げても prewarm() は伝播させない（ログのみ）。"""
    from forge import optimize, prewarm

    @optimize(budget=1)
    def boom(x: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("boom")

    # should not raise
    prewarm(boom, [(4,)], dtype=torch.float32, device="cpu")


def test_prewarm_exported_from_forge_namespace() -> None:
    """forge.prewarm が forge 名前空間からアクセスできる。"""
    import forge

    assert hasattr(forge, "prewarm")
    assert forge.prewarm is not None
