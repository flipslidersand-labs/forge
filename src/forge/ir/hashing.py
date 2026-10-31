from __future__ import annotations

import hashlib
import json
from typing import Any

from .kernel_spec import KernelSpec


def hash_constants(constants: dict[str, Any]) -> str:
    """定数辞書を SHA256 ハッシュの前 16 文字に変換。

    Args:
        constants: キャッシング対象の定数。例: {'eps': 1e-6}

    Returns:
        16 文字の 16 進ハッシュ文字列。CacheKey の一部として使用。
    """
    serialized = json.dumps(constants, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()[:16]


def hash_kernel_spec(spec: KernelSpec) -> str:
    """KernelSpec 全体を SHA256 ハッシュの前 24 文字に変換。

    同じ仕様なら同じハッシュが得られるため、CacheKey 生成に使用。
    op_type, input/output shapes, dtypes, contiguity, constants, constraints を含める。

    Args:
        spec: KernelSpec インスタンス

    Returns:
        24 文字の 16 進ハッシュ文字列。キャッシュキーの主要部分。
    """
    data = {
        "op_type": spec.op_type,
        "inputs": [
            {"shape": s.shape, "dtype": s.dtype_str(), "contiguous": s.is_contiguous}
            for s in spec.input_specs
        ],
        "outputs": [
            {"shape": s.shape, "dtype": s.dtype_str(), "contiguous": s.is_contiguous}
            for s in spec.output_specs
        ],
        "constants": hash_constants(spec.constants),
        "constraints": list(spec.constraints),
    }
    serialized = json.dumps(data, sort_keys=True)
    return hashlib.sha256(serialized.encode()).hexdigest()[:24]
