"""構造化 ProgressEvent — 外部メトリクス収集・structlog 連携用（#190）。

後方互換: ``progress: Callable[[str], None]`` を渡すと非推奨警告を出しつつ
``event.label`` を渡すラッパーに変換する。
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from forge.search.params import SearchParams

ProgressKind = Literal[
    "candidate_ok",
    "candidate_fail",
    "candidate_incorrect",
    "cache_hit",
    "cache_miss",
    "round_done",
    "search_done",
    "search_start",
]


@dataclass
class ProgressEvent:
    """Orchestrator が発行する構造化進捗イベント。

    Attributes:
        kind:      イベント種別。Prometheus ラベルや structlog のキーとして使用。
        label:     人間可読文字列（旧 progress コールバックの文字列に対応）。
        params:    候補 SearchParams（candidate_* イベントのみ）。
        median_us: ベンチマーク中央値 µs（candidate_ok / round_done / search_done）。
        round_num: LLM マルチラウンド探索のラウンド番号（round_done のみ）。
        speedup:   baseline に対する高速化倍率（search_done のみ）。
    """

    kind: ProgressKind
    label: str
    params: SearchParams | None = field(default=None)
    median_us: float | None = field(default=None)
    round_num: int | None = field(default=None)
    speedup: float | None = field(default=None)


ProgressCallback = Callable[[ProgressEvent], None]


def make_progress_callback(
    fn: Callable[[ProgressEvent], None] | Callable[[str], None] | None,
) -> Callable[[ProgressEvent], None]:
    """任意の progress 指定を ``Callable[[ProgressEvent], None]`` に正規化する。

    - ``None`` → no-op
    - ``Callable[[ProgressEvent], None]`` → そのまま
    - ``Callable[[str], None]`` → ``event.label`` を渡すラッパー（非推奨警告付き）

    str 版かどうかの判定はアノテーションではなく呼び出し時のエラーで判断する。
    1 回呼んでみて TypeError なら str コールバックと見なす。
    """
    if fn is None:
        return lambda _e: None

    # アノテーション情報でヒューリスティック判定（inspect.signature は重い）
    import inspect

    sig = inspect.signature(fn)
    params = list(sig.parameters.values())
    if params:
        ann = params[0].annotation
        if ann is str or ann == "str":
            warnings.warn(
                "progress=Callable[[str], None] は非推奨です。"
                " Callable[[ProgressEvent], None] を使ってください（#190）。",
                DeprecationWarning,
                stacklevel=3,
            )
            str_fn: Callable[[str], None] = fn  # type: ignore[assignment]
            return lambda e: str_fn(e.label)

    # 型アノテーションなしや ProgressEvent アノテーション → そのまま使う
    return fn  # type: ignore[return-value]
