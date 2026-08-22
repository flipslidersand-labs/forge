import json
import tempfile
from datetime import UTC, datetime, timezone
from pathlib import Path

import torch

from forge.cache.key import CacheKey
from forge.cache.repository import CachedKernel, KernelRepository
from forge.ir.kernel_spec import KernelSpec
from forge.ir.tensor_spec import TensorSpec


def make_spec() -> KernelSpec:
    return KernelSpec(
        op_type="rmsnorm",
        input_specs=(TensorSpec(shape=(2048, 4096), dtype=torch.float16, is_contiguous=True),),
        output_specs=(TensorSpec(shape=(2048, 4096), dtype=torch.float16, is_contiguous=True),),
        constants={"eps": 1e-6},
        graph_hash="rmsnorm_v1",
        constraints=(),
    )


class TestCacheKey:
    def test_digest_deterministic(self) -> None:
        key = CacheKey(
            graph_hash="abc",
            shapes=((2048, 4096),),
            dtypes=("float16",),
            constants_hash="deadbeef",
            compute_capability="8.9",
            torch_version="2.3.0",
            triton_version="3.0.0",
            cuda_version="12.1",
            library_version="0.1.0",
            template_hash="feedfacecafe",
        )
        assert key.digest() == key.digest()

    def test_digest_differs_on_shape(self) -> None:
        def make(shape: tuple[int, int]) -> CacheKey:
            return CacheKey(
                graph_hash="abc",
                shapes=(shape,),
                dtypes=("float16",),
                constants_hash="deadbeef",
                compute_capability="8.9",
                torch_version="2.3.0",
                triton_version="3.0.0",
                cuda_version="12.1",
                library_version="0.1.0",
                template_hash="feedfacecafe",
            )

        assert make((2048, 4096)).digest() != make((1024, 4096)).digest()

    def test_digest_differs_on_cuda_version(self) -> None:
        base = dict(
            graph_hash="abc",
            shapes=((2048, 4096),),
            dtypes=("float16",),
            constants_hash="deadbeef",
            compute_capability="8.9",
            torch_version="2.3.0",
            triton_version="3.0.0",
            library_version="0.1.0",
            template_hash="feedfacecafe",
        )
        k1 = CacheKey(**base, cuda_version="12.1")
        k2 = CacheKey(**base, cuda_version="12.4")
        assert k1.digest() != k2.digest()


class TestKernelRepository:
    def _make_key(self) -> CacheKey:
        return CacheKey(
            graph_hash="abc",
            shapes=((2048, 4096),),
            dtypes=("float16",),
            constants_hash="deadbeef",
            compute_capability="8.9",
            torch_version="2.3.0",
            triton_version="3.0.0",
            cuda_version="12.1",
            library_version="0.1.0",
            template_hash="feedfacecafe",
        )

    def _make_kernel(self, key: CacheKey) -> CachedKernel:
        from datetime import datetime

        return CachedKernel(
            cache_key=key,
            params={"block_size": 1024, "num_warps": 8},
            kernel_code="def rmsnorm(): pass",
            benchmark_json={"median_us": 42.0, "p20_us": 40.0, "p80_us": 44.0},
            created_at=datetime.now(UTC),
        )

    def test_get_miss(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            repo = KernelRepository(Path(d) / "cache.db")
            assert repo.get(self._make_key()) is None
            repo.close()

    def test_put_and_get(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            repo = KernelRepository(Path(d) / "cache.db")
            key = self._make_key()
            kernel = self._make_kernel(key)
            repo.put(key, kernel)

            result = repo.get(key)
            assert result is not None
            assert result.kernel_code == "def rmsnorm(): pass"
            assert result.params["block_size"] == 1024
            repo.close()

    def test_put_overwrites(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            repo = KernelRepository(Path(d) / "cache.db")
            key = self._make_key()

            k1 = self._make_kernel(key)
            k1.kernel_code = "def v1(): pass"
            repo.put(key, k1)

            k2 = self._make_kernel(key)
            k2.kernel_code = "def v2(): pass"
            repo.put(key, k2)

            result = repo.get(key)
            assert result is not None
            assert result.kernel_code == "def v2(): pass"
            repo.close()

    def test_baseline_us_roundtrip_and_speedup(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            repo = KernelRepository(Path(d) / "cache.db")
            key = self._make_key()
            kernel = self._make_kernel(key)
            kernel.baseline_us = 84.0  # median 42.0 -> 2.0x
            repo.put(key, kernel)

            got = repo.get(key)
            assert got is not None
            assert got.baseline_us == 84.0

            (summary,) = repo.list_summaries()
            assert summary.speedup == 2.0
            assert summary.median_us == 42.0
            repo.close()

    def test_summary_speedup_none_without_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            repo = KernelRepository(Path(d) / "cache.db")
            key = self._make_key()
            repo.put(key, self._make_kernel(key))  # baseline 未指定
            (summary,) = repo.list_summaries()
            assert summary.speedup is None
            repo.close()

    def test_migrates_legacy_db_without_baseline_column(self) -> None:
        import sqlite3

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "cache.db"
            # baseline_us 列が無い旧スキーマ + 旧レイアウトの行を用意
            conn = sqlite3.connect(str(path))
            conn.executescript(
                "CREATE TABLE kernels ("
                "cache_key_hash TEXT PRIMARY KEY, cache_key_json TEXT NOT NULL, "
                "params_json TEXT NOT NULL, kernel_code TEXT NOT NULL, "
                "benchmark_json TEXT NOT NULL, created_at TEXT NOT NULL);"
            )
            legacy_key_json = json.dumps(
                {
                    "graph_hash": "g",
                    "shapes": [[2048, 4096]],
                    "dtypes": ["float16"],
                    "constants_hash": "deadbeef",
                    "compute_capability": "8.9",
                    "torch_version": "2.3.0",
                    "triton_version": "3.0.0",
                    "cuda_version": "12.1",
                    "library_version": "0.1.0",
                    "template_hash": "feedfacecafe",
                }
            )
            conn.execute(
                "INSERT INTO kernels VALUES (?, ?, ?, ?, ?, ?)",
                (
                    "legacyhash",
                    legacy_key_json,
                    "{}",
                    "def k(): pass",
                    '{"median_us": 10.0}',
                    "2026-01-01T00:00:00+00:00",
                ),
            )
            conn.commit()
            conn.close()

            repo = KernelRepository(path)  # ALTER TABLE マイグレーションが走る
            (summary,) = repo.list_summaries()
            assert summary.median_us == 10.0
            assert summary.speedup is None  # 旧行に baseline は無い
            # 新規 put も壊れない
            key = self._make_key()
            repo.put(key, self._make_kernel(key))
            assert repo.count() == 2
            repo.close()

    def test_persists_across_instances(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "cache.db"
            key = self._make_key()

            repo1 = KernelRepository(path)
            repo1.put(key, self._make_kernel(key))
            repo1.close()

            repo2 = KernelRepository(path)
            result = repo2.get(key)
            assert result is not None
            repo2.close()

    def test_context_manager(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            key = self._make_key()
            with KernelRepository(Path(d) / "cache.db") as repo:
                repo.put(key, self._make_kernel(key))
                result = repo.get(key)
            assert result is not None
            # close() 後は接続が閉じている
            import sqlite3

            try:
                repo.conn.execute("SELECT 1")
                raise AssertionError("should have raised")
            except sqlite3.ProgrammingError:
                pass

    def test_context_manager_closes_on_exception(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            try:
                with KernelRepository(Path(d) / "cache.db") as repo:
                    raise ValueError("test error")
            except ValueError:
                pass
            import sqlite3

            try:
                repo.conn.execute("SELECT 1")
                raise AssertionError("should have raised")
            except sqlite3.ProgrammingError:
                pass

    def test_put_list_roundtrip_all_fields(self) -> None:
        """put → list_summaries roundtrip: all CacheKey fields survive serialization."""
        with tempfile.TemporaryDirectory() as d:
            key = self._make_key()
            kernel = self._make_kernel(key)
            with KernelRepository(Path(d) / "cache.db") as repo:
                repo.put(key, kernel)
                summaries = repo.list_summaries()

            assert len(summaries) == 1
            restored = summaries[0].cache_key
            assert restored == key
            assert restored.graph_hash == key.graph_hash
            assert restored.shapes == key.shapes
            assert restored.dtypes == key.dtypes
            assert restored.constants_hash == key.constants_hash
            assert restored.compute_capability == key.compute_capability
            assert restored.library_version == key.library_version

    def test_list_summaries_has_cache_key(self) -> None:
        """KernelSummary must expose cache_key, not flat graph_hash/shapes/dtypes fields."""
        with tempfile.TemporaryDirectory() as d:
            key = self._make_key()
            with KernelRepository(Path(d) / "cache.db") as repo:
                repo.put(key, self._make_kernel(key))
                s = repo.list_summaries()[0]
            assert hasattr(s, "cache_key")
            assert not hasattr(s, "graph_hash"), "use s.cache_key.graph_hash"
            assert not hasattr(s, "shapes"), "flat shapes field removed — use s.cache_key.shapes"
            assert not hasattr(s, "dtypes"), "flat dtypes field removed — use s.cache_key.dtypes"


class TestKernelRepositoryPrune:
    """KernelRepository.prune() のユニットテスト。"""

    def _make_key(self, graph_hash: str = "rmsnorm_v1") -> CacheKey:
        return CacheKey(
            graph_hash=graph_hash,
            shapes=((2048, 4096),),
            dtypes=("float16",),
            constants_hash="deadbeef",
            compute_capability="8.9",
            torch_version="2.3.0",
            triton_version="3.0.0",
            cuda_version="12.1",
            library_version="0.1.0",
            template_hash="feedfacecafe",
        )

    def _put(self, repo: KernelRepository, graph_hash: str) -> None:
        from datetime import datetime

        key = self._make_key(graph_hash)
        repo.put(
            key,
            CachedKernel(
                cache_key=key,
                params={"block_size": 1024},
                kernel_code="def k(): pass",
                benchmark_json={"median_us": 42.0},
                created_at=datetime.now(UTC),
            ),
        )

    def test_prune_no_options_returns_zero(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            with KernelRepository(Path(d) / "cache.db") as repo:
                self._put(repo, "a")
                self._put(repo, "b")
                assert repo.prune() == 0
                assert repo.count() == 2

    def test_prune_before_removes_old_entries(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            with KernelRepository(Path(d) / "cache.db") as repo:
                self._put(repo, "old")
                cutoff = datetime.now(timezone.utc)
                self._put(repo, "new")
                deleted = repo.prune(before=cutoff)
                assert deleted == 1
                assert repo.count() == 1

    def test_prune_keep_latest_removes_oldest(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            with KernelRepository(Path(d) / "cache.db") as repo:
                for i in range(5):
                    self._put(repo, f"v{i}")
                deleted = repo.prune(keep_latest=2)
                assert deleted == 3
                assert repo.count() == 2

    def test_prune_keep_latest_zero_removes_all(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            with KernelRepository(Path(d) / "cache.db") as repo:
                for i in range(3):
                    self._put(repo, f"v{i}")
                deleted = repo.prune(keep_latest=0)
                assert deleted == 3
                assert repo.count() == 0

    def test_prune_keep_latest_larger_than_count_removes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            with KernelRepository(Path(d) / "cache.db") as repo:
                self._put(repo, "only")
                deleted = repo.prune(keep_latest=100)
                assert deleted == 0
                assert repo.count() == 1

    def test_prune_dry_run_does_not_delete(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            with KernelRepository(Path(d) / "cache.db") as repo:
                for i in range(4):
                    self._put(repo, f"v{i}")
                target = repo.prune(keep_latest=1, dry_run=True)
                assert target == 3
                assert repo.count() == 4

    def test_prune_both_options_union(self) -> None:
        """--before と --keep-latest の同時指定は OR 条件で削除する。"""
        with tempfile.TemporaryDirectory() as d:
            with KernelRepository(Path(d) / "cache.db") as repo:
                self._put(repo, "old")
                cutoff = datetime.now(timezone.utc)
                for i in range(4):
                    self._put(repo, f"new{i}")
                # keep_latest=2 → 最新2件を保持: new2,new3 が残る
                # before=cutoff  → old が削除対象
                # OR条件: old + new0 + new1 の計3件が削除される
                deleted = repo.prune(before=cutoff, keep_latest=2)
                assert deleted == 3
                assert repo.count() == 2

    def test_prune_empty_db_returns_zero(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            with KernelRepository(Path(d) / "cache.db") as repo:
                deleted = repo.prune(keep_latest=0)
                assert deleted == 0


class TestCacheKeyFromDict:
    def _base(self) -> dict[str, object]:
        return {
            "graph_hash": "abc",
            "shapes": [[2048, 4096]],
            "dtypes": ["float16"],
            "constants_hash": "deadbeef",
            "compute_capability": "8.9",
            "torch_version": "2.3.0",
            "triton_version": "3.0.0",
            "cuda_version": "12.1",
            "library_version": "0.1.0",
            "template_hash": "feedfacecafe",
        }

    def test_from_dict_roundtrip(self) -> None:
        d = self._base()
        key = CacheKey.from_dict(d)
        assert key.graph_hash == "abc"
        assert key.shapes == ((2048, 4096),)  # list → tuple restored
        assert key.dtypes == ("float16",)  # list → tuple restored
        assert key.constants_hash == "deadbeef"

    def test_from_json_roundtrip(self) -> None:
        original = CacheKey(
            graph_hash="xyz",
            shapes=((1024, 512), (256,)),
            dtypes=("float32", "int32"),
            constants_hash="aabbcc",
            compute_capability="9.0",
            torch_version="2.4.0",
            triton_version="3.1.0",
            cuda_version="12.4",
            library_version="0.2.0",
            template_hash="feedfacecafe",
        )
        raw = json.dumps(
            {k: list(v) if isinstance(v, tuple) else v for k, v in original.__dict__.items()}
        )
        restored = CacheKey.from_json(raw)
        assert restored == original

    def test_from_dict_missing_field_raises(self) -> None:
        d = self._base()
        del d["library_version"]
        try:
            CacheKey.from_dict(d)
            raise AssertionError("should raise KeyError")
        except KeyError:
            pass

    def test_from_dict_digest_stable(self) -> None:
        """from_dict roundtrip must produce the same digest as the original."""
        original = CacheKey(
            graph_hash="abc",
            shapes=((2048, 4096),),
            dtypes=("float16",),
            constants_hash="deadbeef",
            compute_capability="8.9",
            torch_version="2.3.0",
            triton_version="3.0.0",
            cuda_version="12.1",
            library_version="0.1.0",
            template_hash="feedfacecafe",
        )
        raw = json.dumps(
            {k: list(v) if isinstance(v, tuple) else v for k, v in original.__dict__.items()}
        )
        restored = CacheKey.from_json(raw)
        assert restored.digest() == original.digest()
