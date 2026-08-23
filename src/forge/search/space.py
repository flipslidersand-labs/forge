from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

from forge.ir.kernel_spec import KernelSpec
from forge.ops import is_elementwise, is_gemm, is_matmul

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
    # GEMM (linear) 用タイルサイズ。BLOCK_M=BLOCK_N で正方出力タイルを使う。
    gemm_output_blocks: list[int] = field(default_factory=lambda: [64, 128])
    # GEMM の K 次元タイルサイズ（tl.dot 制約上 16 以上の 2 のべき乗）。
    gemm_k_blocks: list[int] = field(default_factory=lambda: [16, 32, 64])
    # attention 用シーケンスブロックサイズ。tl.dot 制約上 16 の倍数かつ ≥ 16 が必須。
    # 小 shape (S=64) では BLOCK_S=16/32 で CTA 数が増え GPU 利用率が上がる。
    attention_seq_blocks: list[int] = field(default_factory=lambda: [16, 32, 64, 128])
    num_warps: list[int] = field(default_factory=lambda: [4, 8, 16, 32])
    num_stages: list[int] = field(default_factory=lambda: [1, 2, 3])
    acc_dtypes: list[str] = field(default_factory=lambda: ["fp32", "fp16"])
    variants: list[str] = field(
        default_factory=lambda: ["single_row", "multi_row", "two_pass", "welford"]
    )
    rows_per_program: list[int] = field(default_factory=lambda: [2, 4, 8, 16])

    def _blocks_for_variant(self, variant: str, n: int) -> list[int]:
        if variant == "two_pass":
            tiles = sorted({b for b in self.block_sizes if b <= n})
            return tiles or [min(self.block_sizes)]
        # single_row / multi_row: 行全体を 1 タイルに収める
        blocks = sorted({b for b in self.block_sizes if b >= n})
        return blocks or [_next_pow2(n)]

    def _rows_for_variant(self, variant: str) -> list[int]:
        return self.rows_per_program if variant == "multi_row" else [1]

    def _enumerate_matmul(self, spec: KernelSpec, stages: list[int]) -> Iterator[SearchParams]:
        """Attention 系 matmul op 向けの探索空間。

        block_size はシーケンス長（S）のタイルサイズ。tl.dot 制約上 ≥ 16 かつ
        head_dim も 16 の倍数である必要がある。acc_dtype は fp32 固定（精度優先）。
        """
        d = spec.input_specs[0].shape[-1]
        if d <= 0 or d % 16 != 0:
            return
        s = spec.input_specs[0].shape[-2]
        valid = sorted({b for b in self.attention_seq_blocks if b <= s})
        if not valid:
            valid = [min(64, _next_pow2(s))]

        seen: set[tuple] = set()
        for variant in ("flash", "flash_causal_opt"):
            for block in valid:
                for warps in self.num_warps:
                    for stage in stages:
                        key = (variant, block, warps, stage)
                        if key in seen:
                            continue
                        seen.add(key)
                        yield SearchParams(
                            block_size=block,
                            num_warps=warps,
                            num_stages=stage,
                            acc_dtype="fp32",
                            variant=variant,
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

    def _enumerate_gemm(self, stages: list[int]) -> Iterator[SearchParams]:
        """GEMM (linear) 向け探索空間。BLOCK_M=BLOCK_N=block_size、BLOCK_K=block_k で探索。"""
        seen: set[tuple] = set()
        for block_mn in self.gemm_output_blocks:
            for block_k in self.gemm_k_blocks:
                for warps in self.num_warps:
                    for stage in stages:
                        for acc in self.acc_dtypes:
                            key = (block_mn, block_k, warps, stage, acc)
                            if key in seen:
                                continue
                            seen.add(key)
                            yield SearchParams(
                                block_size=block_mn,
                                num_warps=warps,
                                num_stages=stage,
                                acc_dtype=acc,
                                variant="gemm",
                                rows_per_program=1,
                                block_k=block_k,
                            )

    def enumerate(self, spec: KernelSpec, compute_capability: str) -> Iterator[SearchParams]:
        """spec と GPU に対して有効な SearchParams を列挙する。

        Pascal 等 cc<7.0 では num_stages を [1] に制限する。
        """
        n = spec.input_specs[0].shape[-1]
        cc = _cc_to_int(compute_capability)
        stages = self.num_stages if cc >= _MIN_CC_FOR_PIPELINING else [1]

        # elementwise op は flat タイルのみ（行 variant を使わない）
        if is_elementwise(spec.op_type):
            yield from self._enumerate_elementwise(stages)
            return

        # matmul (attention) op は flash variant のみ
        if is_matmul(spec.op_type):
            yield from self._enumerate_matmul(spec, stages)
            return

        # GEMM (linear) op は gemm variant
        if is_gemm(spec.op_type):
            yield from self._enumerate_gemm(stages)
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
