from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Callable

from .registry import graph_op_counts, match_counts


# パターン登録は各モジュールの import 副作用。import 漏れで identify() が永久に None を
# 返す事故（Issue #373）を防ぐため、パッケージ内のモジュールを自動で import する。
# 登録順（= match_counts の走査順）が決定的になるよう名前順に読み込む。
def _import_pattern_modules() -> None:
    for name in sorted(m.name for m in pkgutil.iter_modules(__path__)):
        if name != "registry":
            importlib.import_module(f"{__name__}.{name}")


_import_pattern_modules()


def identify(fn: Callable[..., object]) -> str | None:
    """純粋関数を torch.fx で trace し、既知の op_type にパターンマッチする。

    trace 不能（動的制御フロー等）や未知のパターンの場合は None を返す。呼び出し側は
    None を eager フォールバックとして扱う。
    """
    try:
        import torch.fx

        graph = torch.fx.symbolic_trace(fn).graph
    except Exception:  # noqa: BLE001 — trace 失敗は単に「最適化不可」
        return None
    return match_counts(graph_op_counts(graph))


__all__ = ["identify"]
