"""ライブ LLM 反復探索オーケストレーター。

LLMGenerator を複数ラウンド呼び出し、前ラウンドの評価結果をフィードバックして
次ラウンドの候補を改善する。各ラウンドのトークン消費を追跡し、予算超過で停止する。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from forge.ir.kernel_spec import KernelSpec

from .candidate import HistoryEntry
from .llm_generator import LLMGenerator, TokenUsage
from .params import SearchParams

EvaluateFn = Callable[[SearchParams], HistoryEntry]


@dataclass
class RoundResult:
    """1 ラウンドの探索結果。"""

    round_num: int
    evaluated: list[HistoryEntry]
    token_usage: TokenUsage


@dataclass
class IterativeSearchResult:
    """全ラウンドの探索結果サマリー。"""

    rounds: list[RoundResult]
    best: HistoryEntry | None
    total_token_usage: TokenUsage = field(default_factory=TokenUsage)

    def __post_init__(self) -> None:
        self.total_token_usage = TokenUsage(
            input_tokens=sum(r.token_usage.input_tokens for r in self.rounds),
            output_tokens=sum(r.token_usage.output_tokens for r in self.rounds),
        )


class IterativeLLMSearch:
    """LLM 反復探索オーケストレーター。

    Args:
        generator: 候補提案に使う LLMGenerator。propose_fn を注入してテスト可能。
        evaluate_fn: 1 候補を評価して HistoryEntry を返す関数。
        max_rounds: 最大ラウンド数。デフォルト 3。
        candidates_per_round: 1 ラウンドで生成する候補数。デフォルト 5。
        max_tokens: 全ラウンド合計の最大入力 + 出力トークン数。None で無制限。
        top_k_feedback: フィードバックに含める上位候補数。デフォルト 3。
    """

    def __init__(
        self,
        generator: LLMGenerator,
        evaluate_fn: EvaluateFn,
        max_rounds: int = 3,
        candidates_per_round: int = 5,
        max_tokens: int | None = None,
        top_k_feedback: int = 3,
    ) -> None:
        self._gen = generator
        self._evaluate = evaluate_fn
        self._max_rounds = max_rounds
        self._n = candidates_per_round
        self._max_tokens = max_tokens
        self._top_k = top_k_feedback

    def run(self, spec: KernelSpec, compute_capability: str) -> IterativeSearchResult:
        """探索を実行して IterativeSearchResult を返す。"""
        self._gen.reset_usage()
        history: list[HistoryEntry] = []
        seen: set[SearchParams] = set()
        rounds: list[RoundResult] = []

        for rnd in range(1, self._max_rounds + 1):
            if self._budget_exhausted(rounds):
                break

            before = TokenUsage(
                input_tokens=self._gen.token_usage.input_tokens,
                output_tokens=self._gen.token_usage.output_tokens,
            )

            proposals = self._gen.generate(
                spec=spec,
                compute_capability=compute_capability,
                budget=self._n,
                history=history,
            )

            after = self._gen.token_usage
            round_usage = TokenUsage(
                input_tokens=after.input_tokens - before.input_tokens,
                output_tokens=after.output_tokens - before.output_tokens,
            )

            evaluated: list[HistoryEntry] = []
            for p in proposals:
                if p in seen:
                    continue
                seen.add(p)
                entry = self._evaluate(p)
                evaluated.append(entry)
                history.append(entry)

            rounds.append(RoundResult(round_num=rnd, evaluated=evaluated, token_usage=round_usage))

        best = _best_entry(history)
        return IterativeSearchResult(rounds=rounds, best=best)

    def _budget_exhausted(self, rounds: list[RoundResult]) -> bool:
        if self._max_tokens is None:
            return False
        total = self._gen.token_usage.total
        return total >= self._max_tokens


def _best_entry(history: list[HistoryEntry]) -> HistoryEntry | None:
    """正解候補の中で最速のものを返す。正解なしは None。"""
    correct = [h for h in history if h.correct and h.median_us is not None]
    if not correct:
        return None
    return min(correct, key=lambda h: h.median_us)  # type: ignore[arg-type]


def build_feedback_prompt(history: list[HistoryEntry], top_k: int = 3) -> str:
    """前ラウンドの結果をまとめたフィードバック文字列を生成する。

    LLMGenerator.build_prompt() の history 引数として渡せるように、
    HistoryEntry リストをそのまま受け取る。この関数はプロンプト拡張のユーティリティ。
    """
    correct = sorted(
        [h for h in history if h.correct and h.median_us is not None],
        key=lambda h: h.median_us,  # type: ignore[arg-type]
    )
    failed = [h for h in history if not h.correct]

    lines: list[str] = []
    if correct:
        lines.append(f"Top-{min(top_k, len(correct))} successful configs so far:")
        for h in correct[:top_k]:
            lines.append(f"  {h.summary()}")
    if failed:
        lines.append(f"Failure patterns ({len(failed)} total — avoid these):")
        for h in failed[-top_k:]:
            lines.append(f"  {h.summary()}")
    return "\n".join(lines)
