from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

from forge.ir.kernel_spec import KernelSpec
from forge.ops import get_op_info

from .params import SearchParams

# Volta (cc 7.0) 未満では num_stages による非同期パイプラインが効かない。
_MIN_CC_FOR_PIPELINING = 70


def _cc_to_int(compute_capability: str) -> int:
    """'8.9' -> 89, '6.1' -> 61。"""
    major, _, minor = compute_capability.partition(".")
    return int(major) * 10 + int(minor or 0)


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


@dataclass
class SearchSpace:
    """探索する各軸の候補値。spec・GPU によって不可能な組み合わせは enumerate で除外する。

    variant ごとに block_size の意味と制約が異なる:
      single_row / multi_row : block_size は N 以上（行全体を 1 タイルで処理）
      two_pass               : block_size はタイルサイズで N 未満も可
      elementwise            : block_size は flat タイル（N 非依存。elementwise op 専用）

    elementwise op（gelu 等）では reduction 用 variant は使わず、elementwise_blocks を
    タイルサイズとして variant="elementwise" のみを列挙する。
    """

    block_sizes: list[int] = field(default_factory=lambda: [512, 1024, 2048, 4096, 8192])
    elementwise_blocks: list[int] = field(default_factory=lambda: [256, 512, 1024, 2048])
    # Flash Attention の BLOCK_M 候補。head_dim の倍数かつ 2 のべき乗。
    attention_blocks: list[int] = field(default_factory=lambda: [16, 32, 64, 128])
    num_warps: list[int] = field(default_factory=lambda: [4, 8, 16])
    num_stages: list[int] = field(default_factory=lambda: [1, 2, 3])
    acc_dtypes: list[str] = field(default_factory=lambda: ["fp32", "fp16"])
    variants: list[str] = field(default_factory=lambda: ["single_row", "multi_row", "two_pass"])
    rows_per_program: list[int] = field(default_factory=lambda: [2, 4])

    def _blocks_for_variant(self, variant: str, n: int) -> list[int]:
        if variant == "two_pass":
            tiles = sorted({b for b in self.block_sizes if b <= n})
            return tiles or [min(self.block_sizes)]
        # single_row / multi_row: 行全体を 1 タイルに収める
        blocks = sorted({b for b in self.block_sizes if b >= n})
        return blocks or [_next_pow2(n)]

    def _rows_for_variant(self, variant: str) -> list[int]:
        return self.rows_per_program if variant == "multi_row" else [1]

    def _enumerate_attention(self, stages: list[int]) -> Iterator[SearchParams]:
        """Flash Attention 用の探索空間。BLOCK_M = BLOCK_N = block_size（正方タイル）。"""
        seen: set[tuple] = set()
        for block in sorted(set(self.attention_blocks)):
            for warps in self.num_warps:
                for stage in stages:
                    for acc in self.acc_dtypes:
                        key = (block, warps, stage, acc)
                        if key in seen:
                            continue
                        seen.add(key)
                        yield SearchParams(
                            block_size=block,
                            num_warps=warps,
                            num_stages=stage,
                            acc_dtype=acc,
                            variant="attention",
                            rows_per_program=1,
                        )

    def _enumerate_elementwise(self, stages: list[int]) -> Iterator[SearchParams]:
        seen: set[tuple] = set()
        for block in sorted(set(self.elementwise_blocks)):
            for warps in self.num_warps:
                for stage in stages:
                    for acc in self.acc_dtypes:
                        key = (block, warps, stage, acc)
                        if key in seen:
                            continue
                        seen.add(key)
                        yield SearchParams(
                            block_size=block,
                            num_warps=warps,
                            num_stages=stage,
                            acc_dtype=acc,
                            variant="elementwise",
                            rows_per_program=1,
                        )

    def enumerate(self, spec: KernelSpec, compute_capability: str) -> Iterator[SearchParams]:
        """spec と GPU に対して有効な SearchParams を列挙する。

        Pascal 等 cc<7.0 では num_stages を [1] に制限する。
        """
        n = spec.input_specs[0].shape[-1]
        cc = _cc_to_int(compute_capability)
        stages = self.num_stages if cc >= _MIN_CC_FOR_PIPELINING else [1]

        op_kind = get_op_info(spec.op_type).kind
        if op_kind == "elementwise":
            yield from self._enumerate_elementwise(stages)
            return
        if op_kind == "attention":
            yield from self._enumerate_attention(stages)
            return

        seen: set[tuple] = set()
        for variant in self.variants:
            for block in self._blocks_for_variant(variant, n):
                for rows in self._rows_for_variant(variant):
                    for warps in self.num_warps:
                        for stage in stages:
                            for acc in self.acc_dtypes:
                                key = (variant, block, rows, warps, stage, acc)
                                if key in seen:
                                    continue
                                seen.add(key)
                                yield SearchParams(
                                    block_size=block,
                                    num_warps=warps,
                                    num_stages=stage,
                                    acc_dtype=acc,
                                    variant=variant,
                                    rows_per_program=rows,
                                )
