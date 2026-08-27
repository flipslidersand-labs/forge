from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from forge.benchmark.statistics import BenchmarkResultDict

from .key import CacheKey

_log = logging.getLogger("forge.cache")

# CLI (forge.cli) と KernelRepository が共有する既定 DB パスの単一定義。
# FORGE_DB_PATH 環境変数でオーバーライド可能。
DEFAULT_DB_PATH = os.environ.get("FORGE_DB_PATH", "~/.forge/cache.db")


@dataclass
class CachedKernel:
    """SQLite キャッシュから取得した最適カーネル情報。

    Attributes:
        cache_key: 入出力仕様・定数・環境を含めた一意キー。
        params: カーネルパラメータ。例: {'block_size': 256, 'num_warps': 8}
        kernel_code: 生成された Triton カーネルのソースコード。
        benchmark_json: ベンチマーク結果。中央値・P25・P75 等を含む。
        created_at: キャッシュ作成日時（タイムスタンプ）。
        baseline_us: PyTorch eager 実装の計測時間（μs）。speedup 計算に使用。
            旧キャッシュでは None になる可能性がある。
    """

    cache_key: CacheKey
    params: dict[str, object]
    kernel_code: str
    benchmark_json: BenchmarkResultDict
    created_at: datetime
    baseline_us: float | None = None


@dataclass
class KernelSummary:
    """`forge cache list` 用の 1 行分の要約。

    CacheKey を保持することで graph_hash / shapes / dtypes の3重定義を解消する。
    """

    cache_key_hash: str
    cache_key: CacheKey
    median_us: float | None
    # baseline_us / median_us。baseline 未保存の旧キャッシュでは None。
    speedup: float | None
    created_at: str


# 番号付きマイグレーションリスト。追加するときは末尾に append するだけでよい。
# version 1: baseline_us 列追加（旧 DB との後方互換）
_MIGRATIONS: list[tuple[int, str]] = [
    (1, "ALTER TABLE kernels ADD COLUMN baseline_us REAL"),
]


class KernelRepository:
    """SQLite を使った Triton カーネルキャッシュの永続化・検索・管理。

    Repository Pattern を採用。カーネルの CRUD 操作を単一責任に集約し、
    上位層（orchestrator）は SQL 詳細を意識しない設計。

    Attributes:
        conn: SQLite 接続。WAL mode で並行書き込みを許容。
    """

    def __init__(self, path: str | Path = DEFAULT_DB_PATH) -> None:
        """SQLite DB を初期化。テーブルが無ければ作成。

        Args:
            path: SQLite DB ファイルパス。デフォルト: ~/.forge/cache.db
                既存テーブルへの自動スキーママイグレーションに対応。
        """
        db_path = Path(path).expanduser()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._lock = threading.Lock()
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
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
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version     INTEGER PRIMARY KEY,
                    applied_at  TEXT NOT NULL
                );
            """)
            self._run_migrations()
            self.conn.commit()

    def _run_migrations(self) -> None:
        """未適用のマイグレーションを昇順に実行する。ロック内から呼ぶこと。"""
        applied = {
            row[0] for row in self.conn.execute("SELECT version FROM schema_migrations").fetchall()
        }
        # 旧 DB（baseline_us 列あり・schema_migrations なし）の後方互換:
        # 既に列が存在するなら version 1 を適用済みとみなしてスキップする。
        cols = {row[1] for row in self.conn.execute("PRAGMA table_info(kernels)")}
        if "baseline_us" in cols and 1 not in applied:
            self.conn.execute(
                "INSERT OR IGNORE INTO schema_migrations (version, applied_at) VALUES (1, ?)",
                (datetime.now(UTC).isoformat(),),
            )
            applied.add(1)

        for version, sql in sorted(_MIGRATIONS):
            if version in applied:
                continue
            _log.info("applying schema migration v%d", version)
            self.conn.execute(sql)
            self.conn.execute(
                "INSERT INTO schema_migrations (version, applied_at) VALUES (?, ?)",
                (version, datetime.now(UTC).isoformat()),
            )
            _log.info("schema migration v%d applied", version)

    def get(self, key: CacheKey) -> CachedKernel | None:
        with self._lock:
            row = self.conn.execute(
                "SELECT cache_key_json, params_json, kernel_code, benchmark_json, baseline_us, "
                "created_at FROM kernels WHERE cache_key_hash = ?",
                (key.digest(),),
            ).fetchone()
        if row is None:
            _log.debug("cache miss key=%s", key.digest()[:8])
            return None
        _log.debug("cache hit key=%s", key.digest()[:8])
        return CachedKernel(
            cache_key=key,
            params=json.loads(row[1]),
            kernel_code=row[2],
            benchmark_json=json.loads(row[3]),
            baseline_us=row[4],
            created_at=datetime.fromisoformat(row[5]),
        )

    def put(self, key: CacheKey, kernel: CachedKernel) -> None:
        _log.debug("cache write key=%s", key.digest()[:8])
        with self._lock:
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

        各行を fetchall() で一括展開せず、カーソルを逐次イテレートしながら
        JSON デシリアライズを行う。raw 行と KernelSummary を同時にメモリに
        保持しないため、エントリ数が多い場合のピーク使用量を削減できる。
        """
        summaries: list[KernelSummary] = []
        with self._lock:
            cur = self.conn.execute(
                "SELECT cache_key_hash, cache_key_json, benchmark_json, baseline_us, created_at "
                "FROM kernels ORDER BY created_at DESC"
            )
            for key_hash, key_json, bench_json, baseline_us, created_at in cur:
                cache_key = CacheKey.from_json(key_json)
                bench = json.loads(bench_json)
                median = bench.get("median_us")
                median_us = float(median) if median is not None else None
                speedup: float | None = None
                if baseline_us is not None and median_us is not None and median_us > 0:
                    speedup = float(baseline_us) / median_us
                summaries.append(
                    KernelSummary(
                        cache_key_hash=key_hash,
                        cache_key=cache_key,
                        median_us=median_us,
                        speedup=speedup,
                        created_at=created_at,
                    )
                )
        return summaries

    def count(self) -> int:
        with self._lock:
            return int(self.conn.execute("SELECT COUNT(*) FROM kernels").fetchone()[0])

    def prune(
        self,
        before: datetime | None = None,
        keep_latest: int | None = None,
        *,
        dry_run: bool = False,
    ) -> int:
        """期間または件数を指定して古いキャッシュを削除する（`forge cache prune` 用）。

        Args:
            before: この日時より前に作成されたエントリを削除。
            keep_latest: 最新 N 件を残し、それ以外を削除。
            dry_run: True の場合は削除せず対象件数のみ返す。

        Returns:
            削除件数（dry_run=True の場合は削除対象件数）。
            before / keep_latest が両方 None の場合は 0 を返す。
        """
        if before is None and keep_latest is None:
            return 0

        params: list[object] = []
        conditions: list[str] = []

        if before is not None:
            conditions.append("created_at < ?")
            params.append(before.isoformat())

        if keep_latest is not None:
            if not isinstance(keep_latest, int):
                raise TypeError(f"keep_latest must be int, got {type(keep_latest).__name__!r}")
            conditions.append(
                "cache_key_hash NOT IN "
                "(SELECT cache_key_hash FROM kernels ORDER BY created_at DESC LIMIT ?)"
            )
            params.append(keep_latest)

        where = " AND ".join(conditions)

        with self._lock:
            if dry_run:
                row = self.conn.execute(
                    f"SELECT COUNT(*) FROM kernels WHERE {where}", params
                ).fetchone()
                return int(row[0])

            result = self.conn.execute(f"DELETE FROM kernels WHERE {where}", params)
            self.conn.commit()
        return result.rowcount

    def clear(self) -> int:
        """全キャッシュを削除し、削除件数を返す（`forge cache clear` 用）。"""
        with self._lock:
            deleted = self.conn.execute("DELETE FROM kernels").rowcount
            self.conn.commit()
        return deleted

    def __enter__(self) -> KernelRepository:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self.conn.close()
