"""forge.prewarm — デプロイ前キャッシュ事前生成 API (#215)。"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import Any

import torch

_log = logging.getLogger(__name__)


def prewarm(
    fn: Callable[..., Any],
    input_shapes: Sequence[tuple[int, ...]],
    dtype: torch.dtype = torch.float16,
    device: str = "cuda",
    progress: Callable[[str], None] | None = None,
) -> None:
    """@forge.optimize 付き関数のキャッシュを事前生成する。

    Args:
        fn: ``@forge.optimize`` デコレータを適用した関数。
        input_shapes: 各入力テンソルの shape のリスト。
        dtype: テンソルのデータ型（デフォルト torch.float16）。
        device: "cuda" または "cpu"（テスト用）。
        progress: 進捗文字列を受け取るコールバック（省略可）。

    Raises:
        TypeError: ``@forge.optimize`` を適用していない関数を渡した場合。
    """
    if not hasattr(fn, "_forge_compiled"):
        raise TypeError(
            f"prewarm() は @forge.optimize を適用した関数にのみ使えます。"
            f" {fn!r} には _forge_compiled 属性がありません。"
        )

    if progress:
        progress(f"prewarm: {fn.__name__} shapes={input_shapes}")
    _log.info(
        "prewarm fn=%s shapes=%s dtype=%s device=%s", fn.__name__, input_shapes, dtype, device
    )

    inputs = [torch.zeros(shape, dtype=dtype, device=device) for shape in input_shapes]
    try:
        fn(*inputs)
        if progress:
            progress(f"prewarm: {fn.__name__} done")
        _log.info("prewarm done fn=%s", fn.__name__)
    except Exception as exc:  # noqa: BLE001
        _log.warning("prewarm fn=%s failed: %s", fn.__name__, exc)
        if progress:
            progress(f"prewarm: {fn.__name__} failed: {exc}")
