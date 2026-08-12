from __future__ import annotations

from dataclasses import dataclass

from .tensor_spec import TensorSpec


@dataclass(frozen=True)
class KernelSpec:
    """PyTorch 操作を Triton カーネルに変換するための仕様情報。

    KernelSpec は演算の「シグネチャ」を記述し、探索空間を限定する基準となる。

    Attributes:
        op_type: 操作タイプ。'rmsnorm', 'softmax', 'sdpa' など。OP_REGISTRY から検索される。
        input_specs: 入力テンソル仕様のタプル。例: (TensorSpec(...), TensorSpec(...))
        output_specs: 出力テンソル仕様のタプル。
        constants: 定数値。例: {'eps': 1e-6, 'dim': -1}。Triton カーネルのテンプレート引数。
        graph_hash: torch.fx グラフのハッシュ値。キャッシング時の同一性判定に使用。
        constraints: 実装上の制約。例: ('head_dim_power_of_2',)
            — Attention では head_dim は 2 のべき乗必須。

    Note:
        frozen=True により不変クラスとなり、ハッシング可能。CacheKey 生成時に使用。
    """
    op_type: str
    input_specs: tuple[TensorSpec, ...]
    output_specs: tuple[TensorSpec, ...]
    constants: dict[str, object]
    graph_hash: str
    constraints: tuple[str, ...]

    def validate(self) -> None:
        """KernelSpec の妥当性を検証。op_type が登録済みか、入力が空でないか確認。"""
        from forge.ops import OP_INFO

        if self.op_type not in OP_INFO:
            raise ValueError(
                f"Unsupported op_type: {self.op_type!r}. Must be one of {set(OP_INFO)}"
            )
        if not self.input_specs:
            raise ValueError("input_specs must not be empty")
