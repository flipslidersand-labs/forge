from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TensorSpec:
    """入力テンソルの仕様を記述する不変クラス。

    Attributes:
        shape: テンソルの形状。例: (2048, 4096)
        dtype: データ型。torch.float16, torch.float32 など
        is_contiguous: メモリ上の連続性フラグ。True なら行優先（C-order）レイアウト。
            Triton カーネルは連続テンソルを最適化できるため重要。
    """

    shape: tuple[int, ...]
    dtype: torch.dtype
    is_contiguous: bool

    @classmethod
    def from_tensor(cls, t: torch.Tensor) -> TensorSpec:
        """PyTorch テンソルから TensorSpec を生成。"""
        return cls(
            shape=tuple(t.shape),
            dtype=t.dtype,
            is_contiguous=t.is_contiguous(),
        )

    def dtype_str(self) -> str:
        """データ型を文字列表現に変換。例: torch.float16 → 'float16'。

        Raises:
            ValueError: サポート外の dtype が渡された場合。repr() に依存すると
                PyTorch バージョン間で出力が変わり CacheKey が不安定になるため、
                未知の dtype は明示的にエラーとする。
        """
        _map = {
            torch.float16: "float16",
            torch.float32: "float32",
            torch.float64: "float64",
            torch.bfloat16: "bfloat16",
            torch.int8: "int8",
            torch.int16: "int16",
            torch.int32: "int32",
            torch.int64: "int64",
            torch.uint8: "uint8",
            torch.bool: "bool",
        }
        result = _map.get(self.dtype)
        if result is None:
            raise ValueError(
                f"Unsupported dtype for CacheKey: {self.dtype!r}. "
                "Add it to TensorSpec._dtype_map to enable stable cache keys."
            )
        return result
