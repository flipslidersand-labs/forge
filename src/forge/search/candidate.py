from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from forge.ir.kernel_spec import KernelSpec

from .params import SearchParams

_MAX_ERROR_CHARS = 80
# 評価自体ができなかった outcome。「避けるパターン」には含めない。
NOT_EVALUATED_OUTCOMES = ("error", "skipped")


@dataclass(frozen=True)
class HistoryEntry:
    """探索済み候補の結果。LLM へのフィードバックに使う。"""

    params: SearchParams
    correct: bool
    median_us: float | None
    # "ok" / "incorrect" / "error" / "skipped"。error は worker 失敗など実行して評価できなかった
    # 候補、skipped は generate() が拒否して実行しなかった候補（SKIP）。
    # 既定値 "ok" のとき correct=False は従来どおり INCORRECT として扱う。
    outcome: str = "ok"
    error: str | None = None

    def summary(self) -> str:
        if self.outcome in NOT_EVALUATED_OUTCOMES:
            reason = " ".join((self.error or "").split())
            if len(reason) > _MAX_ERROR_CHARS:
                reason = reason[: _MAX_ERROR_CHARS - 3] + "..."
            label = self.outcome.upper()
            outcome = f"{label}: {reason}" if reason else label
        elif not self.correct:
            outcome = "INCORRECT"
        elif self.median_us is not None:
            outcome = f"{self.median_us:.1f}us"
        else:
            outcome = "ok"
        # block_k は gemm のテンプレートだけが使うので、gemm のときだけ出す
        block_k = f" block_k={self.params.block_k}" if self.params.variant == "gemm" else ""
        return (
            f"variant={self.params.variant} block={self.params.block_size} "
            f"warps={self.params.num_warps} stages={self.params.num_stages} "
            f"acc={self.params.acc_dtype} rows={self.params.rows_per_program}{block_k} -> {outcome}"
        )


@runtime_checkable
class CandidateGenerator(Protocol):
    """探索候補を生成する戦略の共通インターフェース。

    GridSearch / RandomSearch / LLMGenerator が実装する。history は過去の
    実験結果で、LLM 等のフィードバック型生成器が利用する（grid/random は無視）。
    reset_usage() は複数 spec にまたがって再利用する場合に呼ぶ（no-op でよい）。
    """

    def generate(
        self,
        spec: KernelSpec,
        compute_capability: str,
        budget: int | None = None,
        history: list[HistoryEntry] | None = None,
    ) -> list[SearchParams]: ...

    def reset_usage(self) -> None: ...
