from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from forge.ir.kernel_spec import KernelSpec

from .candidate import HistoryEntry
from .params import SearchParams


class _BaseGenerator(ABC):
    """LLM 系 CandidateGenerator の共通基底。

    dedup・budget cap・構造化 dict → SearchParams 変換 (_coerce) を集約する。
    サブクラスは ``_propose()`` で候補 dict のリストを返すだけでよい。
    """

    default_n: int = 12

    def generate(
        self,
        spec: KernelSpec,
        compute_capability: str,
        budget: int | None = None,
        history: list[HistoryEntry] | None = None,
    ) -> list[SearchParams]:
        n = budget or self.default_n
        raw = self._propose(spec, compute_capability, n, history or [])

        out: list[SearchParams] = []
        seen: set[SearchParams] = set()
        for d in raw:
            params = self._coerce(d, spec)
            if params is not None and params not in seen:
                seen.add(params)
                out.append(params)
        if budget is not None:
            out = out[:budget]
        return out

    @abstractmethod
    def _propose(
        self,
        spec: KernelSpec,
        compute_capability: str,
        n: int,
        history: list[HistoryEntry],
    ) -> list[dict[str, Any]]:
        """構造化された候補 dict のリストを返す。"""

    def reset_usage(self) -> None:  # noqa: B027 — 意図的な no-op デフォルト
        """トークン使用量などの累積状態を初期化する。デフォルトは no-op。"""

    # --- 構造化 dict -> SearchParams（無効な候補は捨てる） ---

    @staticmethod
    def _coerce(d: dict[str, Any], spec: KernelSpec) -> SearchParams | None:
        variant = str(d.get("base_variant", d.get("variant", "single_row")))
        try:
            params = SearchParams(
                block_size=int(d["block_size"]),
                num_warps=int(d["num_warps"]),
                num_stages=int(d["num_stages"]),
                acc_dtype=str(d.get("acc_dtype", "fp32")),
                variant=variant,
                rows_per_program=int(d.get("rows_per_program", 1)),
                # block_k は gemm のテンプレートだけが使う。他 variant で取り込むと
                # 同一カーネルが dedup をすり抜けて budget を食うので既定値のままにする。
                block_k=_coerce_block_k(d) if variant == "gemm" else 32,
            )
        except (KeyError, ValueError, TypeError):
            return None
        if not _valid_block(params, spec.input_specs[0].shape[-1]):
            return None
        # gemm の K タイルも tl.dot の次元なので 16 以上が必要（SearchParams は 2 のべき乗のみ検査）
        if params.variant == "gemm" and params.block_k < _MIN_DOT_TILE:
            return None
        return params


_DEFAULT_BLOCK_K = 32


def _coerce_block_k(d: dict[str, Any]) -> int:
    """block_k が無い・null の応答は既定 32 に倒す（候補ごと捨てない）。数値でなければ例外。"""
    raw = d.get("block_k")
    return _DEFAULT_BLOCK_K if raw is None else int(raw)


# tl.dot（例: templates/linear.py.jinja の tl.dot(x, tl.trans(w))）の次元は 16 以上が必要な variant
_TILE_VARIANTS = ("gemm", "flash", "flash_causal_opt")
_MIN_DOT_TILE = 16


def block_ok(block_size: int, variant: str, n: int) -> bool:
    """variant ごとの block_size 制約。Bayesian の事前フィルタと LLM 系 _coerce の共有定義。

    n は ``spec.input_specs[0].shape[-1]``（reduction では行幅）。
      two_pass                      : タイルなので n 以下
      elementwise                   : flat タイルで n に縛られない
      gemm / flash / flash_causal_opt: n に縛られない（tl.dot 制約で 16 以上）
      それ以外（single_row 等）      : 行全体を 1 タイルに収めるため n 以上

    SearchSpace は独自に列挙する（block_ok は使わない）。S・head_dim の整合（b <= S、
    head_dim % 16）など spec 全体の検証もここでは行わない。
    """
    if variant == "two_pass":
        return block_size <= n
    if variant == "elementwise":
        return True
    if variant in _TILE_VARIANTS:
        return block_size >= _MIN_DOT_TILE
    return block_size >= n


def attention_block_ok(block_size: int, seq_len: int, head_dim: int) -> bool:
    """attention（flash 系）の block_size を S・head_dim まで含めて検査する。

    block_ok は n=head_dim しか見ないので、Bayesian の提案側がこれで追加に弾く。
    SearchSpace._enumerate_matmul と同じ条件: head_dim は 16 の倍数、tl.dot の最小タイル
    16 以上、block <= S。S < 16 のときだけ 16 を許す（範囲外は kernel 側の mask で除外）。
    """
    if head_dim <= 0 or head_dim % _MIN_DOT_TILE != 0:
        return False
    return _MIN_DOT_TILE <= block_size <= max(seq_len, _MIN_DOT_TILE)


def _valid_block(params: SearchParams, n: int) -> bool:
    return block_ok(params.block_size, params.variant, n)
