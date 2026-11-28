"""BayesianGenerator — Optuna TPE サンプラーによる候補生成器。

Optuna の Trial Suggest + Study.tell() で history を注入し、
次の n 候補を TPE サロゲートモデルで選択する。

依存: optuna>=3.0 (optional extra [search])
  pip install "forge-kernel[search]"
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any

from forge.ir.kernel_spec import KernelSpec

from ._base_generator import _BaseGenerator, attention_block_ok, block_ok
from .candidate import HistoryEntry
from .params import SUPPORTED_ACC_DTYPES

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# study に注入する IntDistribution の範囲（探索時の suggest_int と同じ値）
_NUM_WARPS_CHOICES = (4, 8, 16, 32)  # num_warps_log2 = 2..5
_MAX_ROWS_PER_PROGRAM = 16
# gemm の K タイル候補。LLM 経路（build_prompt）・SearchSpace.gemm_k_blocks と同じ値。
# 空間は gemm のときだけ 3 倍になる（block 6 × warps 4 × stages 3 × acc 2 に対して）。
# TPE の探索幅としては小さく、既定の 32 に固定するより K タイルの効果を拾える。
_GEMM_BLOCK_K_CHOICES = (16, 32, 64)


# variant ごとの block_size 制約は _base_generator.block_ok が共有定義
_block_ok = block_ok


def _block_ok_for_spec(spec: KernelSpec, block_size: int, variant: str) -> bool:
    """提案側の事前フィルタ。block_ok に加え、attention は S・head_dim も見る。"""
    n_cols = spec.input_specs[0].shape[-1]
    if not block_ok(block_size, variant, n_cols):
        return False
    if spec.op_type == "scaled_dot_product_attention":
        return attention_block_ok(block_size, spec.input_specs[0].shape[-2], n_cols)
    return True


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


class BayesianGenerator(_BaseGenerator):
    """Optuna TPE サンプラーで探索候補を提案する CandidateGenerator。

    ``history`` に過去の評価結果（median_us）を注入することで、
    サロゲートモデルが高速な領域を重点的にサンプリングする。

    n_startup_trials 回未満は Optuna 組み込みのランダムサンプリング（cold start）。
    optuna が未インストールの場合は ImportError を遅延して送出する。

    Parameters
    ----------
    n_startup_trials:
        ランダムサンプリングで探索する初期試行回数（デフォルト 10）。
    seed:
        再現性のための乱数シード。None の場合は非決定的。
    default_n:
        budget 未指定時に生成する候補数。
    """

    default_n: int = 12

    def __init__(
        self,
        n_startup_trials: int = 10,
        seed: int | None = None,
        default_n: int = 12,
    ) -> None:
        self.n_startup_trials = n_startup_trials
        self.seed = seed
        self.default_n = default_n
        self._trial_count: int = 0

    def reset_usage(self) -> None:
        self._trial_count = 0

    def _propose(
        self,
        spec: KernelSpec,
        compute_capability: str,
        n: int,
        history: list[HistoryEntry],
    ) -> list[dict[str, Any]]:
        try:
            import optuna  # type: ignore[import]
        except ImportError as e:
            raise ImportError(
                "BayesianGenerator requires optuna. "
                'Install it with: pip install "forge-kernel[search]"'
            ) from e

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        # op の input 幅から block_size の有効範囲を決定
        n_cols = spec.input_specs[0].shape[-1]
        op_type = spec.op_type
        is_elementwise = op_type in ("gelu",)
        is_attention = op_type in ("scaled_dot_product_attention",)
        is_gemm = op_type in ("linear",)

        cc_int = _cc_to_int(compute_capability)
        max_stages = 3 if cc_int >= 70 else 1

        # 有効な block_size 候補を事前計算
        candidate_blocks = _candidate_blocks(n_cols, is_elementwise, is_attention, is_gemm)
        candidate_variants = _candidate_variants(op_type)
        if is_attention:
            # S・head_dim で無効な block は先に除く。ループ内で弾くだけだと S が小さい shape で
            # 試行の大半が捨てられ、候補が少なくなる（head_dim 不正なら全滅）。
            candidate_blocks = [
                b
                for b in candidate_blocks
                if any(_block_ok_for_spec(spec, b, v) for v in candidate_variants)
            ]
            if not candidate_blocks:
                return []

        sampler = optuna.samplers.TPESampler(
            n_startup_trials=self.n_startup_trials,
            seed=self.seed,
        )
        study = optuna.create_study(direction="minimize", sampler=sampler)

        # history を Optuna study に注入
        for entry in history:
            if not entry.correct or entry.median_us is None:
                continue
            params = entry.params
            block_idx = _find_idx(candidate_blocks, params.block_size)
            variant_idx = _find_idx(candidate_variants, params.variant)
            acc_idx = _find_idx(list(SUPPORTED_ACC_DTYPES), params.acc_dtype)
            if block_idx < 0 or variant_idx < 0 or acc_idx < 0:
                continue
            # Optuna 分布の範囲外の値は注入しない（create_trial が ValueError を投げる）。
            # 実測していない値に丸めて学習させないため、クランプせずスキップする。
            if is_gemm and params.block_k not in _GEMM_BLOCK_K_CHOICES:
                continue
            if (
                params.num_warps not in _NUM_WARPS_CHOICES
                or params.num_stages < 1
                or not 1 <= params.rows_per_program <= _MAX_ROWS_PER_PROGRAM
            ):
                logger.debug(
                    "skip history entry out of Optuna range: num_warps=%d num_stages=%d "
                    "rows_per_program=%d",
                    params.num_warps,
                    params.num_stages,
                    params.rows_per_program,
                )
                continue
            trial_params: dict[str, int] = {
                "block_idx": block_idx,
                "num_warps_log2": int(math.log2(max(params.num_warps, 1))),
                "num_stages": min(params.num_stages, max_stages),
                "acc_idx": acc_idx,
                "variant_idx": variant_idx,
                "rows_per_program": params.rows_per_program,
            }
            distributions: dict[str, Any] = {
                "block_idx": optuna.distributions.IntDistribution(0, len(candidate_blocks) - 1),
                "num_warps_log2": optuna.distributions.IntDistribution(2, 5),
                "num_stages": optuna.distributions.IntDistribution(1, max_stages),
                "acc_idx": optuna.distributions.IntDistribution(0, len(SUPPORTED_ACC_DTYPES) - 1),
                "variant_idx": optuna.distributions.IntDistribution(0, len(candidate_variants) - 1),
                "rows_per_program": optuna.distributions.IntDistribution(1, _MAX_ROWS_PER_PROGRAM),
            }
            if is_gemm:
                trial_params["block_k_idx"] = _GEMM_BLOCK_K_CHOICES.index(params.block_k)
                distributions["block_k_idx"] = optuna.distributions.IntDistribution(
                    0, len(_GEMM_BLOCK_K_CHOICES) - 1
                )
            trial = optuna.trial.create_trial(
                params=trial_params, distributions=distributions, value=entry.median_us
            )
            study.add_trial(trial)

        # n 候補をサンプリング
        results: list[dict[str, Any]] = []
        for _ in range(n * 3):  # 無効候補を弾くため多めに試行
            if len(results) >= n:
                break
            trial = study.ask()
            block_idx = trial.suggest_int("block_idx", 0, len(candidate_blocks) - 1)
            warps_log2 = trial.suggest_int("num_warps_log2", 2, 5)  # 4~32
            stages = trial.suggest_int("num_stages", 1, max_stages)
            acc_idx = trial.suggest_int("acc_idx", 0, len(SUPPORTED_ACC_DTYPES) - 1)
            variant_idx = trial.suggest_int("variant_idx", 0, len(candidate_variants) - 1)
            rows = trial.suggest_int("rows_per_program", 1, 16)
            block_k_idx = (
                trial.suggest_int("block_k_idx", 0, len(_GEMM_BLOCK_K_CHOICES) - 1)
                if is_gemm
                else None
            )
            study.tell(trial, float("inf"))  # 未評価は worst として記録

            block_size = candidate_blocks[block_idx]
            variant = candidate_variants[variant_idx]
            num_warps = 2**warps_log2
            acc_dtype = SUPPORTED_ACC_DTYPES[acc_idx]

            # multi_row 以外は rows_per_program=1 に固定
            if variant != "multi_row":
                rows = 1

            if not _block_ok_for_spec(spec, block_size, variant):
                continue

            proposal: dict[str, Any] = {
                "block_size": block_size,
                "num_warps": num_warps,
                "num_stages": stages,
                "acc_dtype": acc_dtype,
                "base_variant": variant,
                "rows_per_program": rows,
            }
            if block_k_idx is not None:
                proposal["block_k"] = _GEMM_BLOCK_K_CHOICES[block_k_idx]
            results.append(proposal)

        return results


# --- ヘルパー ---


def _cc_to_int(cc: str) -> int:
    major, _, minor = cc.partition(".")
    return int(major) * 10 + int(minor or 0)


def _candidate_blocks(n_cols: int, elementwise: bool, attention: bool, gemm: bool) -> list[int]:
    if attention:
        # 16・32 を足す: S が小さい shape（S=16/32）でも block <= S の候補が残るように。
        # S 超過は _block_ok_for_spec が弾く。
        return [16, 32, 64, 128, 256, 512, 1024, 2048]
    if elementwise or gemm:
        return [64, 128, 256, 512, 1024, 2048]
    # reduction: n_cols 以上の 2 のべき乗
    base = _next_pow2(n_cols)
    return sorted({base, base * 2, base * 4, base // 2 if base // 2 >= 64 else base} - {0})


def _candidate_variants(op_type: str) -> list[str]:
    from forge.codegen.triton_codegen import _TEMPLATES  # type: ignore[attr-defined]

    return [v for (op, v) in _TEMPLATES if op == op_type]


def _find_idx(lst: list[Any], value: Any) -> int:
    try:
        return lst.index(value)
    except ValueError:
        return -1
