"""パレート最適化による候補比較。

速度とコスト（API トークン、GPU ベンチマーク時間）のトレードオフを可視化。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from forge.search.params import SearchParams

# コスト換算レート定数（単一箇所で管理）
_TOKEN_COST_PER_TOKEN: float = 0.000003  # $3/M tokens（Anthropic API 平均）
_GPU_COST_PER_SECOND: float = 0.00001  # $0.00001/s（自宅 GTX 1080 電気代相当）


@dataclass
class CostMetrics:
    """探索に要したコストの計測。"""

    tokens: int  # Anthropic API トークン数（input + output）
    benchmark_time_s: float  # ベンチマーク実行時間（秒）
    evaluated_candidates: int  # 評価した候補数

    @property
    def total_cost(self) -> float:
        """コスト統合指標（実験値）。tokens と time を重み付けして統合。

        tokens の重みが 6x 大きいと仮定（API が割高）。
        """
        return (tokens_to_cost(self.tokens) * 6) + time_to_cost_s(self.benchmark_time_s)


@dataclass
class CandidateWithCost:
    """候補と関連コスト＋成果。"""

    params: SearchParams
    median_us: float  # ベンチマーク結果（µs）
    tokens_for_proposal: int  # この候補を提案した際の LLM トークン数
    benchmark_time_ms: float  # この候補をベンチした時間（ms）
    correct: bool  # 数値正確性
    baseline_us: float = field(default=0.0)  # eager baseline（µs）。0 = 未指定

    @property
    def speedup_ratio(self) -> float:
        """baseline（eager）に対する speedup。baseline_us が 0 または未指定なら 1.0。"""
        if self.baseline_us <= 0 or self.median_us <= 0:
            return 1.0
        return self.baseline_us / self.median_us

    def cost_per_1x_speedup(self, baseline_us: float) -> float | None:
        """「baseline から 1 倍速度アップを達成するのに要したコスト」。

        speedup が 1.0x なら None（改善なし）。
        """
        if self.median_us >= baseline_us:
            return None  # 逆方向
        speedup = baseline_us / self.median_us
        if speedup <= 1.0:
            return None
        # 「1x speedup あたりのコスト」
        total_cost = tokens_to_cost(self.tokens_for_proposal) + time_to_cost_s(
            self.benchmark_time_ms / 1000.0
        )
        return total_cost / (speedup - 1.0)

    def is_pareto_optimal(self, candidates: list[CandidateWithCost]) -> bool:
        """this が others の中でパレート最適か。

        「速度の維持・向上 OR コストの低減」を両立する候補がなければ Pareto 最適。
        """
        for other in candidates:
            if other is self or not other.correct:
                continue
            if not self.correct:
                continue
            # other が self より「速度が同等以上＆コストが低い」または「速度が高速＆コスト同等」？
            speed_better_or_equal = other.median_us <= self.median_us
            cost_lower_or_equal = (other.tokens_for_proposal + other.benchmark_time_ms / 100) <= (
                self.tokens_for_proposal + self.benchmark_time_ms / 100
            )
            if speed_better_or_equal and cost_lower_or_equal:
                # other のほうが優れている
                return False
        return True


def tokens_to_cost(tokens: int) -> float:
    """Anthropic API トークン数をコスト（ドル）に換算。

    reference: claude-opus-4-8 は input=$0.3/M, output=$1.5/M。
    ざっくり平均 $3/M（output が多めと仮定）。レート = _TOKEN_COST_PER_TOKEN。
    """
    return tokens * _TOKEN_COST_PER_TOKEN


def time_to_cost_s(seconds: float) -> float:
    """GPU ベンチマーク時間をコスト（ドル）に換算。

    GTX 1080 の電力: ~250W。電気代 $0.10/kWh と仮定。
    250W * 0.0001 kWh / W / 3600 s ≈ $7e-6 / s。
    自宅 GTX 用途として $0.00001/s（_GPU_COST_PER_SECOND）を使用。
    """
    return seconds * _GPU_COST_PER_SECOND


class ParetoFrontier:
    """候補群のパレート前線を管理。"""

    def __init__(self, candidates: list[CandidateWithCost]) -> None:
        self.all_candidates = candidates
        self.frontier = [c for c in candidates if c.is_pareto_optimal(candidates)]

    def recommend(self) -> CandidateWithCost | None:
        """最もコスト効率の良い候補を推奨（速度とコストのバランス）。

        strategy:
        - 最速候補がコスト効率良好なら推奨
        - 中程度の速度で低コストなら推奨（トレードオフ）
        """
        if not self.frontier:
            return None
        # Weighted score: speedup_score * 0.7 - cost_score * 0.3
        # （速度重視だが、コストも加味）
        best = None
        best_score = -float("inf")
        for cand in self.frontier:
            if not cand.correct:
                continue
            # 例：median_us が小さいほど高スコア
            speed_score = 1.0 / max(cand.median_us, 0.01)
            cost = tokens_to_cost(cand.tokens_for_proposal) + time_to_cost_s(
                cand.benchmark_time_ms / 1000.0
            )
            cost_score = cost
            # Combined score (speed up, cost down)
            score = speed_score * 0.7 - cost_score * 100  # コストを顕著にペナルティ
            if score > best_score:
                best_score = score
                best = cand
        return best
