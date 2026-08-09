from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from forge.benchmark.statistics import BenchmarkResultDict

from .key import CacheKey

# CLI (forge.cli) と KernelRepository が共有する既定 DB パスの単一定義。
DEFAULT_DB_PATH = "~/.forge/cache.db"


@dataclass
class CachedKernel:
    cache_key: CacheKey
    params: dict[str, object]
    kernel_code: str
    benchmark_json: BenchmarkResultDict
    created_at: datetime
    # ベスト採用時の baseline（eager 実測）median_us。speedup 算出に使う。
    # 旧キャッシュや baseline 未計測時は None。
    baseline_us: float | None = None


@dataclass
class KernelSummary:
    """`forge cache list` 用の 1 行分の要約（full CacheKey は復元しない）。"""

    cache_key_hash: str
    graph_hash: str
    shapes: list[list[int]]
    dtypes: list[str]
    median_us: float | None
    # baseline_us / median_us。baseline 未保存の旧キャッシュでは None。
    speedup: float | None
    created_at: str


class KernelRepository:
    def __init__(self, path: str | Path = DEFAULT_DB_PATH) -> None:
        db_path = Path(path).expanduser()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path))
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS kernels (
                cache_key_hash  TEXT PRIMARY KEY,
                cache_key_json  TEXT NOT NULL,
                params_json     TEXT NOT NULL,
                kernel_code     TEXT NOT NULL,
                benchmark_json  TEXT NOT NULL,
                baseline_us     REAL,
                created_at      TEXT NOT NULL
            );
        """)
        # 既存 DB（baseline_us 列なし）を破壊せずマイグレーションする。
        cols = {row[1] for row in self.conn.execute("PRAGMA table_info(kernels)")}
        if "baseline_us" not in cols:
            self.conn.execute("ALTER TABLE kernels ADD COLUMN baseline_us REAL")
        self.conn.commit()

    def get(self, key: CacheKey) -> CachedKernel | None:
        row = self.conn.execute(
            "SELECT cache_key_json, params_json, kernel_code, benchmark_json, baseline_us, "
            "created_at FROM kernels WHERE cache_key_hash = ?",
            (key.digest(),),
        ).fetchone()
        if row is None:
            return None
        return CachedKernel(
            cache_key=key,
            params=json.loads(row[1]),
            kernel_code=row[2],
            benchmark_json=json.loads(row[3]),
            baseline_us=row[4],
            created_at=datetime.fromisoformat(row[5]),
        )

    def put(self, key: CacheKey, kernel: CachedKernel) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO kernels "
            "(cache_key_hash, cache_key_json, params_json, kernel_code, benchmark_json, "
            "baseline_us, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                key.digest(),
                json.dumps(key.__dict__, default=list),
                json.dumps(kernel.params),
                kernel.kernel_code,
                json.dumps(kernel.benchmark_json),
                kernel.baseline_us,
                datetime.now(UTC).isoformat(),
            ),
        )
        self.conn.commit()

    def list_summaries(self) -> list[KernelSummary]:
        """キャッシュ済みカーネルを新しい順に要約して返す（`forge cache list` 用）。

        speedup = baseline_us / median_us。baseline 未保存の旧キャッシュでは None。
        """
        rows = self.conn.execute(
            "SELECT cache_key_hash, cache_key_json, benchmark_json, baseline_us, created_at "
            "FROM kernels ORDER BY created_at DESC"
        ).fetchall()
        summaries: list[KernelSummary] = []
        for key_hash, key_json, bench_json, baseline_us, created_at in rows:
            key = json.loads(key_json)
            bench = json.loads(bench_json)
            median = bench.get("median_us")
            median_us = float(median) if median is not None else None
            speedup: float | None = None
            if baseline_us is not None and median_us is not None and median_us > 0:
                speedup = float(baseline_us) / median_us
            summaries.append(
                KernelSummary(
                    cache_key_hash=key_hash,
                    graph_hash=key.get("graph_hash", "?"),
                    shapes=key.get("shapes", []),
                    dtypes=key.get("dtypes", []),
                    median_us=median_us,
                    speedup=speedup,
                    created_at=created_at,
                )
            )
        return summaries

    def count(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) FROM kernels").fetchone()[0])

    def clear(self) -> int:
        """全キャッシュを削除し、削除件数を返す（`forge cache clear` 用）。"""
        deleted = self.conn.execute("DELETE FROM kernels").rowcount
        self.conn.commit()
        return deleted

    def __enter__(self) -> KernelRepository:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self.conn.close()
