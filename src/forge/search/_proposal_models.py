"""LLM 系ジェネレータ共有の構造化出力スキーマ。

pydantic は optional dependency (`forge[llm]`) のため、このモジュールは
各ジェネレータの呼び出し時にのみ import されること（core の import 経路に
含めないこと）。
"""

from __future__ import annotations

from pydantic import BaseModel


class Candidate(BaseModel):
    base_variant: str
    block_size: int
    num_warps: int
    num_stages: int
    acc_dtype: str
    rows_per_program: int
    # gemm (linear) の K タイル。他 variant では無視される。省略可（既存の出力を読めるように）。
    block_k: int = 32
    hypothesis: str


class Proposal(BaseModel):
    candidates: list[Candidate]
