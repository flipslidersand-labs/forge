"""IR 型の property-based tests（#214）。Hypothesis を使用。GPU 不要。"""

from __future__ import annotations

import json

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

# --- Hypothesis strategies ---

_DTYPES = [torch.float16, torch.float32, torch.bfloat16, torch.float64]

_dtype_st = st.sampled_from(_DTYPES)

_dim_st = st.integers(min_value=1, max_value=8192)

_shape_st = st.one_of(
    st.tuples(_dim_st),
    st.tuples(_dim_st, _dim_st),
    st.tuples(_dim_st, _dim_st, _dim_st),
)

_bool_st = st.booleans()


@st.composite
def tensor_spec_st(draw):
    from forge.ir.tensor_spec import TensorSpec

    shape = draw(_shape_st)
    dtype = draw(_dtype_st)
    is_contiguous = draw(_bool_st)
    return TensorSpec(shape=shape, dtype=dtype, is_contiguous=is_contiguous)


# CacheKey strategy — construct directly (skip GPU env detection)
_str_st = st.text(
    alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd"), whitelist_characters="_-."),
    min_size=1,
    max_size=64,
)
_version_st = st.from_regex(r"\d+\.\d+\.\d+", fullmatch=True)
_cc_st = st.from_regex(r"\d+\.\d+", fullmatch=True)


@st.composite
def cache_key_st(draw):
    from forge.cache.key import CacheKey

    n = draw(st.integers(min_value=1, max_value=4))
    shapes = tuple(tuple(draw(st.lists(_dim_st, min_size=1, max_size=3))) for _ in range(n))
    dtypes = tuple(draw(st.sampled_from(["float16", "float32", "bfloat16"])) for _ in range(n))

    return CacheKey(
        graph_hash=draw(_str_st),
        shapes=shapes,
        dtypes=dtypes,
        constants_hash=draw(_str_st),
        compute_capability=draw(_cc_st),
        torch_version=draw(_version_st),
        triton_version=draw(_version_st),
        cuda_version=draw(_version_st),
        library_version=draw(_version_st),
        template_hash=draw(_str_st),
    )


# --- Tests ---


@given(spec=tensor_spec_st())
@settings(max_examples=100)
def test_tensor_spec_equality_is_reflexive(spec) -> None:
    """TensorSpec は自身と等しい（frozen dataclass の __eq__ 決定性確認）。"""
    assert spec == spec


@given(a=tensor_spec_st(), b=tensor_spec_st())
@settings(max_examples=100)
def test_tensor_spec_equality_matches_field_values(a, b) -> None:
    """TensorSpec の __eq__ はフィールド値の一致と一致する。"""
    fields_equal = (a.shape == b.shape and a.dtype == b.dtype and a.is_contiguous == b.is_contiguous)
    assert (a == b) == fields_equal


@given(key=cache_key_st())
@settings(max_examples=100)
def test_cache_key_digest_is_deterministic(key) -> None:
    """同一 CacheKey の digest() は何度呼んでも同じ値を返す。"""
    assert key.digest() == key.digest()


@given(key=cache_key_st())
@settings(max_examples=100)
def test_cache_key_json_round_trip(key) -> None:
    """CacheKey → from_json → 等価。"""
    from forge.cache.key import CacheKey

    raw = json.dumps(
        {
            "graph_hash": key.graph_hash,
            "shapes": [list(s) for s in key.shapes],
            "dtypes": list(key.dtypes),
            "constants_hash": key.constants_hash,
            "compute_capability": key.compute_capability,
            "torch_version": key.torch_version,
            "triton_version": key.triton_version,
            "cuda_version": key.cuda_version,
            "library_version": key.library_version,
            "template_hash": key.template_hash,
        }
    )
    restored = CacheKey.from_json(raw)
    assert restored == key


@given(key=cache_key_st())
@settings(max_examples=100)
def test_cache_key_from_dict_round_trip(key) -> None:
    """CacheKey → asdict → from_dict → 等価。"""
    from dataclasses import asdict

    from forge.cache.key import CacheKey

    d = asdict(key)
    restored = CacheKey.from_dict(d)
    assert restored == key


@given(key=cache_key_st())
@settings(max_examples=50)
def test_cache_key_different_graph_hash_different_digest(key) -> None:
    """graph_hash が異なれば digest() も（ほぼ確実に）異なる。"""
    from dataclasses import replace

    other = replace(key, graph_hash=key.graph_hash + "_x")
    assert key.digest() != other.digest()
