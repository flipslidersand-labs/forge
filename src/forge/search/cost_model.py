"""コスト考慮型探索のコストモデル。

各候補の評価コスト（ベンチマーク時間）を予測・キャッシュし、
早期打ち切り (early stopping) の判断材料を提供する。
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from forge.search.params import SearchParams

_DEFAULT_DB = Path.home() / ".cache" / "forge" / "cost_profile.db"

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS cost_profile (
    variant      TEXT    NOT NULL,
    block_size   INTEGER NOT NULL,
    num_warps    INTEGER NOT NULL,
    num_stages   INTEGER NOT NULL,
    acc_dtype    TEXT    NOT NULL,
    bench_time_ms REAL   NOT NULL,
    PRIMARY KEY (variant, block_size, num_warps, num_stages, acc_dtype)
)
"""


@dataclass
class CostEstimate:
    """候補1件の評価コスト推定結果。"""

    params: SearchParams
    estimated_bench_ms: float
    is_cached: bool  # True = DB キャッシュから、False = デフォルト推定


class CostModel:
    """Shape / variant ごとのベンチマーク時間を SQLite にキャッシュし予測する。

    Args:
        db_path: キャッシュ DB のパス。省略で ~/.cache/forge/cost_profile.db。
        default_ms: キャッシュミス時のデフォルト推定値（ms）。
    """

    def __init__(
        self,
        db_path: Path | None = None,
        default_ms: float = 50.0,
    ) -> None:
        self._db_path = db_path or _DEFAULT_DB
        self._default_ms = default_ms
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.execute(_CREATE_TABLE)
        self._conn.commit()

    def predict_cost(self, params: SearchParams) -> CostEstimate:
        """過去の実績からベンチマーク時間を予測する。"""
        row = self._conn.execute(
            """
            SELECT bench_time_ms FROM cost_profile
            WHERE variant=? AND block_size=? AND num_warps=?
              AND num_stages=? AND acc_dtype=?
            """,
            _params_key(params),
        ).fetchone()

        if row:
            return CostEstimate(params=params, estimated_bench_ms=row[0], is_cached=True)

        # キャッシュミス: block_size * num_warps に比例する簡易推定
        num_warps = max(params.num_warps, 1)
        estimated = self._default_ms * (params.block_size / 256) * (num_warps / 4)
        return CostEstimate(params=params, estimated_bench_ms=estimated, is_cached=False)

    def record(self, params: SearchParams, bench_time_ms: float) -> None:
        """実測値をキャッシュに記録する（UPSERT）。"""
        self._conn.execute(
            """
            INSERT INTO cost_profile
                (variant, block_size, num_warps, num_stages, acc_dtype, bench_time_ms)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(variant, block_size, num_warps, num_stages, acc_dtype)
            DO UPDATE SET bench_time_ms = excluded.bench_time_ms
            """,
            (*_params_key(params), bench_time_ms),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> CostModel:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class BudgetTracker:
    """探索全体の経過時間を追跡し、予算超過を検知する。

    Args:
        max_total_s: 探索全体の最大許容時間（秒）。None で無制限。
    """

    def __init__(self, max_total_s: float | None = None) -> None:
        self._max_s = max_total_s
        self._start = time.monotonic()

    def record(self, bench_ms: float) -> None:
        """1 候補の実測時間を記録する（API 互換性のため保持）。"""

    @property
    def elapsed_s(self) -> float:
        return time.monotonic() - self._start

    @property
    def budget_exhausted(self) -> bool:
        """予算（max_total_s）を超過しているか。"""
        if self._max_s is None:
            return False
        return self.elapsed_s >= self._max_s

    def should_skip(self, estimated_ms: float) -> bool:
        """この候補を評価すると予算超過する見込みなら True を返す。"""
        if self._max_s is None:
            return False
        remaining_s = self._max_s - self.elapsed_s
        return (estimated_ms / 1000.0) > remaining_s


def _params_key(
    params: SearchParams,
) -> tuple[str, int, int, int, str]:
    """SearchParams を DB キー用タプルに変換する。predict_cost と record で共用。"""
    return (
        params.variant,
        params.block_size,
        params.num_warps,
        params.num_stages,
        params.acc_dtype,
    )


def scalarize(speedup: float, cost: float, lam: float = 0.1) -> float:
    """速度とコストを線形スカラー化する。

    score = speedup - lam * cost

    Args:
        speedup: baseline に対する速度倍率（>= 1.0 で改善）。
        cost: 評価コスト（任意単位; 秒を想定）。
        lam: コストのペナルティ係数（デフォルト 0.1）。
    """
    return speedup - lam * cost
