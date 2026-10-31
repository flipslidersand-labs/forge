from __future__ import annotations

import logging

from forge.ir.kernel_spec import KernelSpec

from .candidate import HistoryEntry
from .params import SearchParams
from .space import SearchSpace

_log = logging.getLogger("forge.search.random")


class RandomSearch:
    """探索空間からランダムに候補をサンプリングする。

    variant を加えるとグリッドが大きくなるため、全列挙の代わりに budget 件だけ
    一様サンプリングする。LLM 等の高度な探索器の効果は、この乱択との比較で測る。

    Math.random は使えない環境のため、seed と線形合同法で決定的に列挙してから
    シャッフルする（再現性のため）。
    """

    def __init__(self, space: SearchSpace | None = None, seed: int = 0) -> None:
        self.space = space or SearchSpace()
        self.seed = seed

    def generate(
        self,
        spec: KernelSpec,
        compute_capability: str,
        budget: int | None = None,
        history: list[HistoryEntry] | None = None,
    ) -> list[SearchParams]:
        candidates = list(self.space.enumerate(spec, compute_capability))
        shuffled = _lcg_shuffle(candidates, self.seed)
        total = len(shuffled)
        if budget is not None:
            shuffled = shuffled[:budget]
        _log.debug(
            "random search op=%s cc=%s total=%d budget=%s returning=%d seed=%d",
            spec.op_type,
            compute_capability,
            total,
            budget,
            len(shuffled),
            self.seed,
        )
        return shuffled

    def reset_usage(self) -> None:
        pass


def _lcg_shuffle(items: list, seed: int) -> list:
    """線形合同法による決定的 Fisher-Yates シャッフル（外部乱数に依存しない）。"""
    out = list(items)
    state = (seed * 2654435761 + 12345) & 0xFFFFFFFF
    for i in range(len(out) - 1, 0, -1):
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        j = state % (i + 1)
        out[i], out[j] = out[j], out[i]
    return out
