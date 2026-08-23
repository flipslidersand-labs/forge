"""IR 型の round-trip property tests（#214）。GPU 不要。

hypothesis で SearchParams / BenchmarkResult / CacheKey の
シリアライズ→デシリアライズが恒等変換であることを検証する。
"""

from __future__ import annotations

import json
from dataclasses import asdict

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from forge.benchmark.statistics import BenchmarkResult
from forge.cache.key import CacheKey
from forge.ir.hashing import hash_constants
from forge.ir.tensor_spec import TensorSpec
from forge.search.params import SUPPORTED_ACC_DTYPES, SUPPORTED_VARIANTS, SearchParams

# --------------------------------------------------------------------------- #
# SearchParams strategies                                                      #
# --------------------------------------------------------------------------- #

_POW2 = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
_BLOCK_K = [16, 32, 64, 128]
_WARPS = [1, 2, 4, 8, 16, 32]
_STAGES = [1, 2, 3, 4]


@st.composite
def search_params(draw: st.DrawFn) -> SearchParams:
    variant = draw(st.sampled_from(SUPPORTED_VARIANTS))
    # rows_per_program > 1 は multi_row 専用
    if variant == "multi_row":
        rows = draw(st.integers(min_value=1, max_value=16))
    else:
        rows = 1
    return SearchParams(
        block_size=draw(st.sampled_from(_POW2)),
        num_warps=draw(st.sampled_from(_WARPS)),
        num_stages=draw(st.sampled_from(_STAGES)),
        acc_dtype=draw(st.sampled_from(list(SUPPORTED_ACC_DTYPES))),
        variant=variant,
        rows_per_program=rows,
        block_k=draw(st.sampled_from(_BLOCK_K)),
    )


# --------------------------------------------------------------------------- #
# BenchmarkResult strategies                                                   #
# --------------------------------------------------------------------------- #

_pos_float = st.floats(min_value=0.1, max_value=1e6, allow_nan=False, allow_infinity=False)


@st.composite
def benchmark_result(draw: st.DrawFn) -> BenchmarkResult:
    median = draw(_pos_float)
    p20 = draw(_pos_float)
    p80 = draw(_pos_float)
    p95 = draw(_pos_float)
    warmup = draw(st.integers(min_value=0, max_value=100))
    measure = draw(st.integers(min_value=0, max_value=200))
    return BenchmarkResult(
        median_us=median,
        p20_us=p20,
        p80_us=p80,
        p95_us=p95,
        warmup_count=warmup,
        measure_count=measure,
    )


# --------------------------------------------------------------------------- #
# CacheKey strategies                                                          #
# --------------------------------------------------------------------------- #

_short_str = st.text(
    alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd"), whitelist_characters="._-"),
    min_size=1,
    max_size=32,
)
_hex_str = st.text(alphabet="0123456789abcdef", min_size=8, max_size=64)
_version_str = st.text(
    alphabet=st.characters(whitelist_categories=("Nd",), whitelist_characters="."),
    min_size=1,
    max_size=16,
)
_cc_str = st.builds(
    lambda major, minor: f"{major}.{minor}",
    major=st.integers(min_value=0, max_value=9),
    minor=st.integers(min_value=0, max_value=9),
)


@st.composite
def cache_key(draw: st.DrawFn) -> CacheKey:
    n_inputs = draw(st.integers(min_value=1, max_value=4))
    shapes = tuple(
        tuple(draw(st.integers(min_value=1, max_value=8192)) for _ in range(draw(st.integers(1, 4))))
        for _ in range(n_inputs)
    )
    dtypes = tuple(draw(st.sampled_from(["float16", "float32", "bfloat16"])) for _ in range(n_inputs))
    return CacheKey(
        graph_hash=draw(_hex_str),
        shapes=shapes,
        dtypes=dtypes,
        constants_hash=draw(_hex_str),
        compute_capability=draw(_cc_str),
        torch_version=draw(_version_str),
        triton_version=draw(_version_str),
        cuda_version=draw(_version_str),
        library_version=draw(_version_str),
        template_hash=draw(_hex_str),
    )


# --------------------------------------------------------------------------- #
# SearchParams round-trip                                                      #
# --------------------------------------------------------------------------- #


class TestSearchParamsRoundTrip:
    @given(search_params())
    @settings(max_examples=200)
    def test_to_dict_from_dict_identity(self, params: SearchParams) -> None:
        assert SearchParams.from_dict(params.to_dict()) == params

    @given(search_params())
    @settings(max_examples=200)
    def test_dict_is_json_serializable(self, params: SearchParams) -> None:
        d = params.to_dict()
        restored = json.loads(json.dumps(d))
        assert SearchParams.from_dict(restored) == params

    @given(search_params())
    @settings(max_examples=100)
    def test_to_dict_contains_all_fields(self, params: SearchParams) -> None:
        d = params.to_dict()
        for field in ("block_size", "num_warps", "num_stages", "acc_dtype", "variant", "rows_per_program", "block_k"):
            assert field in d

    @given(search_params())
    @settings(max_examples=100)
    def test_roundtrip_preserves_variant(self, params: SearchParams) -> None:
        restored = SearchParams.from_dict(params.to_dict())
        assert restored.variant == params.variant

    @given(search_params())
    @settings(max_examples=100)
    def test_roundtrip_preserves_block_k(self, params: SearchParams) -> None:
        restored = SearchParams.from_dict(params.to_dict())
        assert restored.block_k == params.block_k


# --------------------------------------------------------------------------- #
# BenchmarkResult round-trip                                                   #
# --------------------------------------------------------------------------- #


class TestBenchmarkResultRoundTrip:
    @given(benchmark_result())
    @settings(max_examples=200)
    def test_to_dict_from_dict_identity(self, br: BenchmarkResult) -> None:
        restored = BenchmarkResult.from_dict(br.to_dict())
        assert restored.median_us == pytest.approx(br.median_us, rel=1e-9)
        assert restored.p20_us == pytest.approx(br.p20_us, rel=1e-9)
        assert restored.p80_us == pytest.approx(br.p80_us, rel=1e-9)
        assert restored.p95_us == pytest.approx(br.p95_us, rel=1e-9)
        assert restored.warmup_count == br.warmup_count
        assert restored.measure_count == br.measure_count

    @given(benchmark_result())
    @settings(max_examples=200)
    def test_dict_is_json_serializable(self, br: BenchmarkResult) -> None:
        d = br.to_dict()
        restored = BenchmarkResult.from_dict(json.loads(json.dumps(d)))
        assert restored.median_us == pytest.approx(br.median_us, rel=1e-9)

    @given(benchmark_result())
    @settings(max_examples=100)
    def test_to_dict_contains_required_keys(self, br: BenchmarkResult) -> None:
        d = br.to_dict()
        for key in ("median_us", "p20_us", "p80_us", "p95_us"):
            assert key in d


# --------------------------------------------------------------------------- #
# CacheKey round-trip                                                          #
# --------------------------------------------------------------------------- #


class TestCacheKeyRoundTrip:
    @given(cache_key())
    @settings(max_examples=200)
    def test_from_dict_asdict_identity(self, key: CacheKey) -> None:
        restored = CacheKey.from_dict(asdict(key))
        assert restored == key

    @given(cache_key())
    @settings(max_examples=200)
    def test_from_json_identity(self, key: CacheKey) -> None:
        raw = json.dumps(asdict(key), default=list)
        restored = CacheKey.from_json(raw)
        assert restored == key

    @given(cache_key())
    @settings(max_examples=100)
    def test_digest_is_deterministic(self, key: CacheKey) -> None:
        assert key.digest() == key.digest()

    @given(cache_key())
    @settings(max_examples=100)
    def test_digest_is_hex_string(self, key: CacheKey) -> None:
        d = key.digest()
        assert len(d) == 64
        int(d, 16)  # must be valid hex


# --------------------------------------------------------------------------- #
# hash_constants determinism                                                   #
# --------------------------------------------------------------------------- #


_json_scalar = st.one_of(
    st.integers(min_value=-1000, max_value=1000),
    st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    st.text(max_size=20),
    st.booleans(),
)


class TestHashConstantsDeterminism:
    @given(st.dictionaries(st.text(max_size=10), _json_scalar, max_size=8))
    @settings(max_examples=200)
    def test_deterministic(self, constants: dict) -> None:
        assert hash_constants(constants) == hash_constants(constants)

    @given(st.dictionaries(st.text(max_size=10), _json_scalar, max_size=8))
    @settings(max_examples=100)
    def test_returns_16_char_hex(self, constants: dict) -> None:
        h = hash_constants(constants)
        assert len(h) == 16
        int(h, 16)


# --------------------------------------------------------------------------- #
# TensorSpec dtype_str determinism                                              #
# --------------------------------------------------------------------------- #


_DTYPES = [torch.float16, torch.float32, torch.bfloat16, torch.float64]


class TestTensorSpecDtypeStr:
    @given(
        dtype=st.sampled_from(_DTYPES),
        shape=st.lists(st.integers(1, 8192), min_size=1, max_size=4),
        contiguous=st.booleans(),
    )
    @settings(max_examples=100)
    def test_dtype_str_deterministic(self, dtype: torch.dtype, shape: list[int], contiguous: bool) -> None:
        spec = TensorSpec(shape=tuple(shape), dtype=dtype, is_contiguous=contiguous)
        assert spec.dtype_str() == spec.dtype_str()

    @given(dtype=st.sampled_from(_DTYPES))
    @settings(max_examples=20)
    def test_dtype_str_nonempty(self, dtype: torch.dtype) -> None:
        spec = TensorSpec(shape=(1,), dtype=dtype, is_contiguous=True)
        assert spec.dtype_str()
