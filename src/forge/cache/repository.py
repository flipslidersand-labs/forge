from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .key import CacheKey


@dataclass
class CachedKernel:
    cache_key: CacheKey
    params: dict[str, object]
    kernel_code: str
    benchmark_json: dict[str, object]
    created_at: datetime


@dataclass
class KernelSummary:
    """`forge cache list` 用の 1 行分の要約（full CacheKey は復元しない）。"""

    cache_key_hash: str
    graph_hash: str
    shapes: list[list[int]]
    dtypes: list[str]
    median_us: float | None
    created_at: str


class KernelRepository:
    def __init__(self, path: str | Path = "~/.forge/cache.db") -> None:
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
                created_at      TEXT NOT NULL
            );
        """)
        self.conn.commit()

    def get(self, key: CacheKey) -> CachedKernel | None:
        row = self.conn.execute(
            "SELECT cache_key_json, params_json, kernel_code, benchmark_json, created_at "
            "FROM kernels WHERE cache_key_hash = ?",
            (key.digest(),),
        ).fetchone()
        if row is None:
            return None
        return CachedKernel(
            cache_key=key,
            params=json.loads(row[1]),
            kernel_code=row[2],
            benchmark_json=json.loads(row[3]),
            created_at=datetime.fromisoformat(row[4]),
        )

    def put(self, key: CacheKey, kernel: CachedKernel) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO kernels VALUES (?, ?, ?, ?, ?, ?)",
            (
                key.digest(),
                json.dumps(key.__dict__, default=list),
                json.dumps(kernel.params),
                kernel.kernel_code,
                json.dumps(kernel.benchmark_json),
                datetime.now(UTC).isoformat(),
            ),
        )
        self.conn.commit()

    def list_summaries(self) -> list[KernelSummary]:
        """キャッシュ済みカーネルを新しい順に要約して返す（`forge cache list` 用）。

        speedup は baseline を保存していないため median_us で代替する。
        """
        rows = self.conn.execute(
            "SELECT cache_key_hash, cache_key_json, benchmark_json, created_at "
            "FROM kernels ORDER BY created_at DESC"
        ).fetchall()
        summaries: list[KernelSummary] = []
        for key_hash, key_json, bench_json, created_at in rows:
            key = json.loads(key_json)
            bench = json.loads(bench_json)
            median = bench.get("median_us")
            summaries.append(
                KernelSummary(
                    cache_key_hash=key_hash,
                    graph_hash=key.get("graph_hash", "?"),
                    shapes=key.get("shapes", []),
                    dtypes=key.get("dtypes", []),
                    median_us=float(median) if median is not None else None,
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

    def close(self) -> None:
        self.conn.close()
