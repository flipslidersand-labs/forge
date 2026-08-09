from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from forge.ir.kernel_spec import KernelSpec

from .candidate import HistoryEntry
from .params import SearchParams


class _BaseGenerator(ABC):
    """LLM 系 CandidateGenerator の共通基底。

    dedup・budget cap・構造化 dict → SearchParams 変換 (_coerce) を集約する。
    サブクラスは ``_propose()`` で候補 dict のリストを返すだけでよい。
    """

    default_n: int = 12

    def generate(
        self,
        spec: KernelSpec,
        compute_capability: str,
        budget: int | None = None,
        history: list[HistoryEntry] | None = None,
    ) -> list[SearchParams]:
        n = budget or self.default_n
        raw = self._propose(spec, compute_capability, n, history or [])

        out: list[SearchParams] = []
        seen: set[SearchParams] = set()
        for d in raw:
            params = self._coerce(d, spec)
            if params is not None and params not in seen:
                seen.add(params)
                out.append(params)
        if budget is not None:
            out = out[:budget]
        return out

    @abstractmethod
    def _propose(
        self,
        spec: KernelSpec,
        compute_capability: str,
        n: int,
        history: list[HistoryEntry],
    ) -> list[dict[str, Any]]:
        """構造化された候補 dict のリストを返す。"""

    def reset_usage(self) -> None:
        """トークン使用量などの累積状態を初期化する。デフォルトは no-op。"""

    # --- 構造化 dict -> SearchParams（無効な候補は捨てる） ---

    @staticmethod
    def _coerce(d: dict[str, Any], spec: KernelSpec) -> SearchParams | None:
        try:
            params = SearchParams(
                block_size=int(d["block_size"]),
                num_warps=int(d["num_warps"]),
                num_stages=int(d["num_stages"]),
                acc_dtype=str(d.get("acc_dtype", "fp32")),
                variant=str(d.get("base_variant", d.get("variant", "single_row"))),
                rows_per_program=int(d.get("rows_per_program", 1)),
            )
        except (KeyError, ValueError, TypeError):
            return None
        if not _valid_block(params, spec.input_specs[0].shape[-1]):
            return None
        return params


def _valid_block(params: SearchParams, n: int) -> bool:
    """variant ごとの block_size 制約（SearchSpace と同じルール）。"""
    if params.variant == "two_pass":
        return params.block_size <= n
    # single_row / multi_row は行全体を 1 タイルに収める
    return params.block_size >= n
