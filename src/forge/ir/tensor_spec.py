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
        """データ型を文字列表現に変換。例: torch.float16 → 'float16'。"""
        _map = {
            torch.float32: "float32",
            torch.float16: "float16",
            torch.bfloat16: "bfloat16",
            torch.float64: "float64",
        }
        return _map.get(self.dtype, repr(self.dtype))
