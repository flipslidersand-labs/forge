from __future__ import annotations

import importlib.util
from typing import Any

from forge.ir.kernel_spec import KernelSpec

from ._base_generator import _BaseGenerator
from .candidate import HistoryEntry
from .params import SearchParams

_OPTUNA_AVAILABLE: bool = importlib.util.find_spec("optuna") is not None

# Discrete choices that cover the full SearchSpace.
# _BaseGenerator._coerce() / _valid_block() filter out spec-incompatible values.
_BLOCK_SIZES = (256, 512, 1024, 2048, 4096, 8192, 16384)
_NUM_WARPS = (4, 8, 16, 32)
_NUM_STAGES = (1, 2, 3, 4)
_ACC_DTYPES = ("fp32", "fp16")
_VARIANTS = ("single_row", "multi_row", "two_pass", "elementwise", "welford")
_ROWS_PER_PROGRAM = (1, 2, 4, 8, 16)


def _distributions() -> dict[str, Any]:
    from optuna.distributions import CategoricalDistribution  # type: ignore[import-untyped]

    return {
        "block_size": CategoricalDistribution(_BLOCK_SIZES),
        "num_warps": CategoricalDistribution(_NUM_WARPS),
        "num_stages": CategoricalDistribution(_NUM_STAGES),
        "acc_dtype": CategoricalDistribution(_ACC_DTYPES),
        "variant": CategoricalDistribution(_VARIANTS),
        "rows_per_program": CategoricalDistribution(_ROWS_PER_PROGRAM),
    }


def _in_distributions(params: dict[str, Any], dists: dict[str, Any]) -> bool:
    for k, v in params.items():
        if k not in dists or v not in dists[k].choices:
            return False
    return True


class BayesianGenerator(_BaseGenerator):
    """Optuna TPE サンプラーで次候補を選択する探索器。

    history (SearchParams, median_us) を Optuna study に注入し、過去のベンチマーク
    結果を surrogate model のフィードバックとして利用する。
    history 数が n_startup_trials 未満の場合は random sampling（cold start）。
    """

    def __init__(self, n_startup_trials: int = 10, seed: int | None = None) -> None:
        if not _OPTUNA_AVAILABLE:
            raise ImportError(
                "optuna is required for BayesianGenerator. "
                "Install it with: pip install forge-kernel[search]"
            )
        self._n_startup = n_startup_trials
        self._seed = seed

    # _BaseGenerator.generate() が dedup・budget cap・_coerce を担当する。
    # ここでは raw dict リストを返すだけでよい。

    def _propose(
        self,
        spec: KernelSpec,
        compute_capability: str,
        n: int,
        history: list[HistoryEntry],
    ) -> list[dict[str, Any]]:
        import optuna  # type: ignore[import-untyped]
        from optuna.samplers import TPESampler  # type: ignore[import-untyped]

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        sampler = TPESampler(seed=self._seed, n_startup_trials=self._n_startup)
        study = optuna.create_study(direction="minimize", sampler=sampler)

        dists = _distributions()

        # history を completed trial として注入 — TPE の surrogate model を warm start する。
        for entry in history:
            if not entry.correct or entry.median_us is None:
                continue
            raw = {
                "block_size": entry.params.block_size,
                "num_warps": entry.params.num_warps,
                "num_stages": entry.params.num_stages,
                "acc_dtype": entry.params.acc_dtype,
                "variant": entry.params.variant,
                "rows_per_program": entry.params.rows_per_program,
            }
            if not _in_distributions(raw, dists):
                continue
            trial = optuna.trial.create_trial(
                params=raw,
                distributions=dists,
                value=entry.median_us,
            )
            study.add_trial(trial)

        # n 件の有効候補を得るまで過剰サンプリングする。
        # _coerce がフィルタするため、raw dict の 19% 程度しか有効にならない。
        # 早期終了を有効候補数で判定し、raw key で重複排除する。
        results: list[dict[str, Any]] = []
        seen_keys: set[tuple[Any, ...]] = set()
        seen_valid: set[SearchParams] = set()
        max_attempts = n * 40 + 80  # 有効率 ~20% を想定した余裕係数

        for _ in range(max_attempts):
            if len(results) >= n:
                break
            trial = study.ask(dists)
            key = tuple(trial.params[k] for k in sorted(trial.params))
            # PRUNED にして使用済みにしつつ TPE model への影響を限定する。
            study.tell(trial, state=optuna.trial.TrialState.PRUNED)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            # spec に対して有効かどうかを事前確認して効率化する。
            valid = self._coerce(dict(trial.params), spec)
            if valid is not None and valid not in seen_valid:
                seen_valid.add(valid)
                results.append(dict(trial.params))

        return results

    def reset_usage(self) -> None:
        pass  # study は呼び出しごとに新規作成するため、状態なし
