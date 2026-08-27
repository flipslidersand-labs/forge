from __future__ import annotations

import inspect
import logging
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

from forge.benchmark.statistics import BenchmarkResult, is_improvement
from forge.cache.key import CacheKey
from forge.cache.repository import CachedKernel, KernelRepository
from forge.codegen.triton_codegen import generate
from forge.ir.kernel_spec import KernelSpec
from forge.notifiers.discord import DiscordNotifier
from forge.runtime.worker import (
    ExtendedBaselineResult,
    WorkerResult,
    run_extended_baseline_in_worker,
    run_in_worker,
)
from forge.search.candidate import CandidateGenerator, HistoryEntry
from forge.search.grid import GridSearch
from forge.search.params import SearchParams
from forge.validation.test_cases import correctness_cases, primary_input
from forge.validation.tolerance import get_tolerance

_log = logging.getLogger("forge.orchestrator")

if TYPE_CHECKING:
    from forge.search.llm_generator import LLMGenerator, TokenUsage


@dataclass
class ProgressEvent:
    """構造化進捗イベント。Orchestrator の progress コールバックに渡される。

    kind で種別を判別し、params / median_us / round_num / speedup で詳細を取得できる。
    label は人間可読文字列（従来の progress 文字列と等価）。
    """

    kind: Literal[
        "candidate_ok",
        "candidate_fail",
        "candidate_incorrect",
        "cache_hit",
        "round_done",
        "search_done",
        "info",
    ]
    label: str
    params: SearchParams | None = None
    median_us: float | None = None
    round_num: int | None = None
    speedup: float | None = None


@dataclass
class ExperimentResult:
    params: SearchParams
    success: bool
    correct: bool
    median_us: float | None
    error: str | None
    is_best: bool = False


@dataclass
class SearchResult:
    spec: KernelSpec
    cache_hit: bool
    best_params: SearchParams | None
    best_benchmark: BenchmarkResult | None
    baseline_benchmark: BenchmarkResult | None
    baseline_name: str | None
    experiments: list[ExperimentResult]
    extended_baselines: list[ExtendedBaselineResult] = field(default_factory=list)
    total_time_s: float = field(default=0.0)
    failed_count: int = field(default=0)
    incorrect_count: int = field(default=0)

    @property
    def speedup(self) -> float | None:
        if self.best_benchmark and self.baseline_benchmark and self.best_benchmark.median_us > 0:
            return self.baseline_benchmark.median_us / self.best_benchmark.median_us
        return None


@dataclass
class RoundResult:
    """1 ラウンド分の探索結果。"""

    round_num: int
    experiments: list[ExperimentResult]
    best_params: SearchParams | None
    best_median_us: float | None


@dataclass
class MultiRoundResult:
    """LLM 反復探索全体の結果。"""

    spec: KernelSpec
    rounds: list[RoundResult]
    best_params: SearchParams | None
    best_benchmark: BenchmarkResult | None
    baseline_benchmark: BenchmarkResult | None
    baseline_name: str | None
    token_usage: TokenUsage | None
    total_candidates_evaluated: int = field(default=0)
    extended_baselines: list[ExtendedBaselineResult] = field(default_factory=list)
    total_benchmark_time_s: float = field(default=0.0)  # コスト計測用
    failed_count: int = field(default=0)
    incorrect_count: int = field(default=0)

    @property
    def speedup(self) -> float | None:
        if self.best_benchmark and self.baseline_benchmark and self.best_benchmark.median_us > 0:
            return self.baseline_benchmark.median_us / self.best_benchmark.median_us
        return None

    @property
    def best_round(self) -> int | None:
        """best_params が見つかったラウンド番号（1 始まり）。"""
        for r in self.rounds:
            if r.best_params == self.best_params:
                return r.round_num
        return None

    @property
    def all_experiments(self) -> list[ExperimentResult]:
        return [exp for r in self.rounds for exp in r.experiments]


@dataclass
class _SearchContext:
    """optimize() / optimize_rounds() 共通の前処理結果。"""

    spec: KernelSpec
    key: CacheKey
    bench_input: list[dict[str, Any]]
    cases: list[dict[str, Any]] | None
    tol: dict[str, Any]
    extended: list[ExtendedBaselineResult]
    start_time: float


class Orchestrator:
    """KernelSpec を受け取り、探索 → 検証 → ベンチマーク → キャッシュの一連を回す。

    再現可能なパイプラインが中心で、探索器 (GridSearch) は差し替え可能な一要素。
    """

    def __init__(
        self,
        repo: KernelRepository | None = None,
        python_executable: str | None = None,
        min_speedup: float = 1.03,
        warmup: int = 25,
        repeat: int = 200,
        timeout_s: float = 60.0,
        progress: Callable[[ProgressEvent], None] | Callable[[str], None] | None = None,
        measure_extended: bool = False,
        notifier: DiscordNotifier | None = None,
    ) -> None:
        if repo is None:
            self.repo = KernelRepository()
            self._owns_repo = True
        else:
            self.repo = repo
            self._owns_repo = False
        self.python_executable = python_executable
        self.min_speedup = min_speedup
        self.warmup = warmup
        self.repeat = repeat
        self.timeout_s = timeout_s
        self._event_handler = self._wrap_progress(progress)
        self.measure_extended = measure_extended
        self.notifier = notifier or DiscordNotifier()

    @staticmethod
    def _wrap_progress(
        fn: Callable[[ProgressEvent], None] | Callable[[str], None] | None,
    ) -> Callable[[ProgressEvent], None]:
        """progress 引数を ProgressEvent → void に統一する。

        - None: no-op
        - Callable[[ProgressEvent], None]: そのまま使う
        - Callable[[str], None]: event.label を渡すラッパー（後方互換、非推奨）
        """
        if fn is None:
            return lambda _e: None
        try:
            sig = inspect.signature(fn)
            params = list(sig.parameters.values())
        except (ValueError, TypeError):
            params = []
        if params:
            ann = params[0].annotation
            if ann is str or ann == "str":
                warnings.warn(
                    "progress=Callable[[str], None] は非推奨です。"
                    " Callable[[ProgressEvent], None] に変更してください。",
                    DeprecationWarning,
                    stacklevel=3,
                )
                str_fn: Callable[[str], None] = fn  # type: ignore[assignment]
                return lambda e: str_fn(e.label)
        return fn  # type: ignore[return-value]

    def _progress(self, msg: str, *, kind: str = "info") -> None:
        """内部文字列ベースの進捗通知（後方互換用）。ProgressEvent を wrap して送出する。"""
        self._event_handler(
            ProgressEvent(kind=kind, label=msg)  # type: ignore[arg-type]
        )

    def _emit(self, event: ProgressEvent) -> None:
        """構造化 ProgressEvent を送出する。"""
        self._event_handler(event)

    def close(self) -> None:
        if self._owns_repo:
            self.repo.close()

    def __enter__(self) -> Orchestrator:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # --- optimize()/optimize_rounds() 共通の前処理・後処理 ---

    def _prepare(
        self, spec: KernelSpec, use_cache: bool
    ) -> tuple[_SearchContext, CachedKernel | None]:
        """検証・キャッシュ照会・入力生成・extended baseline 計測をまとめて行う。

        キャッシュヒット時は入力生成・extended 計測をスキップし、
        (最小限の ctx, CachedKernel) を返す。ミス時は (完全な ctx, None)。
        """
        spec.validate()
        key = CacheKey.from_spec_and_env(spec)
        start_time = time.time()

        if use_cache and (cached := self.repo.get(key)) is not None:
            self._progress(f"cache HIT: {cached.params}")
            _log.info("cache HIT op=%s", spec.op_type)
            ctx = _SearchContext(
                spec=spec,
                key=key,
                bench_input=[],
                cases=None,
                tol={},
                extended=[],
                start_time=start_time,
            )
            return ctx, cached

        _log.debug("cache MISS op=%s", spec.op_type)
        bench_input = primary_input(spec)
        cases = correctness_cases(spec)
        tol = get_tolerance(spec.op_type).to_dict()

        extended: list[ExtendedBaselineResult] = []
        if self.measure_extended:
            self._progress("measuring extended baselines (torch.compile) …")
            extended = run_extended_baseline_in_worker(
                spec.op_type,
                bench_input,
                spec.constants,
                warmup=self.warmup,
                repeat=self.repeat,
                python_executable=self.python_executable,
            )
            for eb in extended:
                if eb.failed:
                    self._progress(f"  extended: {eb.name} FAILED: {eb.error}")
                    _log.warning("extended baseline FAILED name=%s error=%s", eb.name, eb.error)
                else:
                    self._progress(
                        f"  extended: {eb.name} median={eb.benchmark.median_us:.1f}µs "
                        f"p95={eb.benchmark.p95_us:.1f}µs compile={eb.compile_time_s:.1f}s"
                    )
                    _log.debug(
                        "extended baseline name=%s median=%.1fus compile=%.1fs",
                        eb.name,
                        eb.benchmark.median_us,
                        eb.compile_time_s,
                    )

        ctx = _SearchContext(
            spec=spec,
            key=key,
            bench_input=bench_input,
            cases=cases,
            tol=tol,
            extended=extended,
            start_time=start_time,
        )
        return ctx, None

    def _finalize(
        self,
        ctx: _SearchContext,
        best_params: SearchParams | None,
        best_bench: BenchmarkResult | None,
        *,
        num_candidates: int,
        baseline_us: float | None = None,
        notify: bool = False,
        speedup: float | None = None,
        best_round: int | None = None,
        failed_count: int = 0,
    ) -> None:
        """ベスト候補のキャッシュ書き込みと Discord 通知を行う。"""
        if best_params is not None and best_bench is not None:
            code = generate(ctx.spec, best_params)
            self.repo.put(
                ctx.key,
                CachedKernel(
                    cache_key=ctx.key,
                    params=best_params.to_dict(),
                    kernel_code=code,
                    benchmark_json=best_bench.to_dict(),
                    baseline_us=baseline_us,
                    created_at=datetime.now(UTC),
                ),
            )
            self._progress(f"cached best: {best_params} ({best_bench.median_us:.1f}us)")
            _log.info(
                "cached best op=%s median=%.1fus %s",
                ctx.spec.op_type,
                best_bench.median_us,
                best_params,
            )
            if notify:
                duration_seconds = time.time() - ctx.start_time
                failed_rate = failed_count / num_candidates if num_candidates > 0 else None
                self.notifier.send_optimization_complete(
                    op_name=ctx.spec.op_type,
                    best_time=best_bench.median_us / 1000.0,  # Convert us to ms
                    num_candidates=num_candidates,
                    duration_seconds=duration_seconds,
                    speedup=speedup,
                    best_round=best_round,
                    failed_rate=failed_rate,
                )
        elif notify:
            if num_candidates > 0:
                error_msg = (
                    f"No successful candidates found after exploring {num_candidates} options"
                )
            else:
                error_msg = "No candidates to explore"
            _log.warning("no best found op=%s: %s", ctx.spec.op_type, error_msg)
            self.notifier.send_optimization_error(
                op_name=ctx.spec.op_type,
                error_message=error_msg,
                error_type="OPTIMIZATION_FAILED",
            )

    def optimize(
        self,
        spec: KernelSpec,
        budget: int = 50,
        search: CandidateGenerator | None = None,
        use_cache: bool = True,
    ) -> SearchResult:
        ctx, cached = self._prepare(spec, use_cache)
        if cached_result := self._check_cache(spec, ctx, cached):
            return cached_result

        candidates = self._generate_candidates(spec, ctx, search, budget)
        experiments, best_params, best_bench, baseline_bench, baseline_name = (
            self._explore_candidates(spec, ctx, candidates)
        )

        failed_count = sum(1 for e in experiments if not e.success)
        speedup: float | None = None
        if best_bench and baseline_bench and best_bench.median_us > 0:
            speedup = baseline_bench.median_us / best_bench.median_us
        self._persist_result(
            ctx,
            best_params,
            best_bench,
            len(candidates),
            baseline_bench,
            failed_count=failed_count,
            speedup=speedup,
        )

        return SearchResult(
            spec=spec,
            cache_hit=False,
            best_params=best_params,
            best_benchmark=best_bench,
            baseline_benchmark=baseline_bench,
            baseline_name=baseline_name,
            experiments=experiments,
            extended_baselines=ctx.extended,
            total_time_s=time.time() - ctx.start_time,
            failed_count=sum(1 for e in experiments if not e.success),
            incorrect_count=sum(1 for e in experiments if e.success and not e.correct),
        )

    def _check_cache(
        self,
        spec: KernelSpec,
        ctx: _SearchContext,
        cached: CachedKernel | None,
    ) -> SearchResult | None:
        """キャッシュヒット時、結果を返す。ミス時は None。"""
        if cached is None:
            return None
        bench = BenchmarkResult.from_dict(cached.benchmark_json)
        self.notifier.send_cache_hit(spec.op_type)
        return SearchResult(
            spec=spec,
            cache_hit=True,
            best_params=SearchParams.from_dict(cached.params),
            best_benchmark=bench,
            baseline_benchmark=None,
            baseline_name=None,
            experiments=[],
        )

    def _generate_candidates(
        self,
        spec: KernelSpec,
        ctx: _SearchContext,
        search: CandidateGenerator | None,
        budget: int,
    ) -> list[SearchParams]:
        """探索器から候補を生成。"""
        search = search or GridSearch()
        candidates = search.generate(spec, ctx.key.compute_capability, budget=budget)
        self._progress(f"searching {len(candidates)} candidates (cc {ctx.key.compute_capability})")
        _log.info(
            "search start op=%s candidates=%d cc=%s",
            spec.op_type,
            len(candidates),
            ctx.key.compute_capability,
        )
        return candidates

    def _explore_candidates(
        self,
        spec: KernelSpec,
        ctx: _SearchContext,
        candidates: list[SearchParams],
    ) -> tuple[
        list[ExperimentResult],
        SearchParams | None,
        BenchmarkResult | None,
        BenchmarkResult | None,
        str | None,
    ]:
        """各候補を評価。最良候補を返す。"""
        experiments: list[ExperimentResult] = []
        best_params: SearchParams | None = None
        best_bench: BenchmarkResult | None = None
        baseline_bench: BenchmarkResult | None = None
        baseline_name: str | None = None

        for i, params in enumerate(candidates, 1):
            label = f"[{i}/{len(candidates)}] {params.block_size}/{params.num_warps}"
            exp, cand_bench, bl_bench, bl_name = self._eval_one(
                spec, params, ctx.bench_input, ctx.cases, ctx.tol, label
            )
            if bl_bench is not None and baseline_bench is None:
                baseline_bench, baseline_name = bl_bench, bl_name

            if exp.correct and cand_bench is not None:
                improved = best_bench is None or is_improvement(
                    cand_bench, best_bench, self.min_speedup
                )
                if improved:
                    best_params, best_bench = params, cand_bench
                    exp.is_best = True
                    _log.debug("new best %.1fus %s", cand_bench.median_us, params)
                else:
                    _log.debug("candidate %.1fus %s", cand_bench.median_us, params)
            experiments.append(exp)

        return experiments, best_params, best_bench, baseline_bench, baseline_name

    def _persist_result(
        self,
        ctx: _SearchContext,
        best_params: SearchParams | None,
        best_bench: BenchmarkResult | None,
        num_candidates: int,
        baseline_bench: BenchmarkResult | None,
        failed_count: int = 0,
        speedup: float | None = None,
    ) -> None:
        """最良候補をキャッシュに永続化。"""
        self._finalize(
            ctx,
            best_params,
            best_bench,
            num_candidates=num_candidates,
            baseline_us=baseline_bench.median_us if baseline_bench else None,
            notify=True,
            speedup=speedup,
            failed_count=failed_count,
        )

    def optimize_rounds(
        self,
        spec: KernelSpec,
        llm: LLMGenerator,
        n_rounds: int = 3,
        candidates_per_round: int = 12,
        use_cache: bool = True,
    ) -> MultiRoundResult:
        """LLM を使った反復探索。各ラウンドの結果を history として次ラウンドへ渡す。

        ANTHROPIC_API_KEY が必要（llm に propose_fn を注入することでテスト可能）。
        GPU 必須（ベンチマーク・検証を実行するため）。
        """
        ctx, cached = self._prepare(spec, use_cache)
        if cached is not None:
            bench = BenchmarkResult.from_dict(cached.benchmark_json)
            return MultiRoundResult(
                spec=spec,
                rounds=[],
                best_params=SearchParams.from_dict(cached.params),
                best_benchmark=bench,
                baseline_benchmark=None,
                baseline_name=None,
                token_usage=llm.token_usage,
                total_benchmark_time_s=0.0,
            )

        llm.reset_usage()
        history: list[HistoryEntry] = []
        rounds: list[RoundResult] = []
        overall_best_params: SearchParams | None = None
        overall_best_bench: BenchmarkResult | None = None
        baseline_bench: BenchmarkResult | None = None
        baseline_name: str | None = None
        seen_params: set[SearchParams] = set()
        total_benchmark_time_s = 0.0
        round_start_time = time.time()

        for round_num in range(1, n_rounds + 1):
            self._progress(
                f"[round {round_num}/{n_rounds}] generating {candidates_per_round} candidates "
                f"(history={len(history)})"
            )
            candidates = llm.generate(
                spec,
                ctx.key.compute_capability,
                budget=candidates_per_round,
                history=history,
            )

            round_experiments: list[ExperimentResult] = []
            round_best_params: SearchParams | None = None
            round_best_us: float | None = None

            for i, params in enumerate(candidates, 1):
                if params in seen_params:
                    self._progress(f"  [r{round_num}.{i}] skip duplicate")
                    continue
                seen_params.add(params)
                label = f"  [r{round_num}.{i}] {params.block_size}/{params.num_warps}"
                exp, cand_bench, bl_bench, bl_name = self._eval_one(
                    spec, params, ctx.bench_input, ctx.cases, ctx.tol, label
                )
                if bl_bench is not None:
                    baseline_bench, baseline_name = bl_bench, bl_name

                if exp.correct and cand_bench is not None:
                    history.append(
                        HistoryEntry(params=params, correct=True, median_us=cand_bench.median_us)
                    )
                    improved = overall_best_bench is None or is_improvement(
                        cand_bench, overall_best_bench, self.min_speedup
                    )
                    if improved:
                        overall_best_params, overall_best_bench = params, cand_bench
                        exp.is_best = True
                        self._progress(f"  [r{round_num}.{i}] {cand_bench.median_us:.1f}us BEST")
                    if round_best_us is None or cand_bench.median_us < round_best_us:
                        round_best_params = params
                        round_best_us = cand_bench.median_us
                else:
                    history.append(HistoryEntry(params=params, correct=exp.correct, median_us=None))

                round_experiments.append(exp)

            round_elapsed = time.time() - round_start_time
            total_benchmark_time_s += round_elapsed
            rounds.append(
                RoundResult(
                    round_num=round_num,
                    experiments=round_experiments,
                    best_params=round_best_params,
                    best_median_us=round_best_us,
                )
            )
            self._progress(
                f"  round {round_num} done: best={round_best_us:.1f}us (time={round_elapsed:.1f}s)"
                if round_best_us is not None
                else f"  round {round_num} done: no valid candidates (time={round_elapsed:.1f}s)"
            )
            round_start_time = time.time()

        total = sum(len(r.experiments) for r in rounds)
        all_exps_for_rounds = [e for r in rounds for e in r.experiments]
        round_failed_count = sum(1 for e in all_exps_for_rounds if not e.success)
        round_speedup: float | None = None
        if overall_best_bench and baseline_bench and overall_best_bench.median_us > 0:
            round_speedup = baseline_bench.median_us / overall_best_bench.median_us
        best_round_num: int | None = None
        for r in rounds:
            if r.best_params == overall_best_params and overall_best_params is not None:
                best_round_num = r.round_num
                break
        self._finalize(
            ctx,
            overall_best_params,
            overall_best_bench,
            num_candidates=total,
            baseline_us=baseline_bench.median_us if baseline_bench else None,
            speedup=round_speedup,
            best_round=best_round_num,
            failed_count=round_failed_count,
        )

        all_exps = [e for r in rounds for e in r.experiments]
        return MultiRoundResult(
            spec=spec,
            rounds=rounds,
            best_params=overall_best_params,
            best_benchmark=overall_best_bench,
            baseline_benchmark=baseline_bench,
            baseline_name=baseline_name,
            token_usage=llm.token_usage,
            total_candidates_evaluated=total,
            extended_baselines=ctx.extended,
            total_benchmark_time_s=total_benchmark_time_s,
            failed_count=sum(1 for e in all_exps if not e.success),
            incorrect_count=sum(1 for e in all_exps if e.success and not e.correct),
        )

    def optimize_sha(
        self,
        spec: KernelSpec,
        initial_budget: int = 32,
        halving_rounds: int = 3,
        search: CandidateGenerator | None = None,
        use_cache: bool = True,
    ) -> SearchResult:
        """Successive Halving で探索する。

        各ラウンドで warmup/repeat を増やしながら上位 50% に候補を絞り込む。

        | ラウンド | 候補数              | warmup | repeat |
        |---------|---------------------|--------|--------|
        | 1       | initial_budget      | 5      | 20     |
        | 2       | initial_budget // 2 | 10     | 50     |
        | 3       | initial_budget // 4 | 25     | 200    |
        """
        if initial_budget < 64:
            warnings.warn(
                f"optimize_sha: initial_budget={initial_budget} が小さいため "
                "ラウンド合計 eval 数が initial_budget を超える場合があります。"
                "initial_budget >= 64 を推奨します。",
                UserWarning,
                stacklevel=2,
            )
        ctx, cached = self._prepare(spec, use_cache)
        if cached is not None:
            bench = BenchmarkResult.from_dict(cached.benchmark_json)
            self.notifier.send_cache_hit(spec.op_type)
            return SearchResult(
                spec=spec,
                cache_hit=True,
                best_params=SearchParams.from_dict(cached.params),
                best_benchmark=bench,
                baseline_benchmark=None,
                baseline_name=None,
                experiments=[],
            )

        _ROUND_CONFIGS: list[tuple[int, int]] = [
            (5, 20),
            (10, 50),
            (25, 200),
        ]

        search = search or GridSearch()
        candidates = search.generate(spec, ctx.key.compute_capability, budget=initial_budget)
        self._progress(
            f"SHA: {len(candidates)} candidates, {halving_rounds} rounds "
            f"(cc {ctx.key.compute_capability})"
        )

        all_experiments: list[ExperimentResult] = []
        baseline_bench: BenchmarkResult | None = None
        baseline_name: str | None = None
        surviving: list[SearchParams] = list(candidates)
        round_results: list[tuple[SearchParams, BenchmarkResult]] = []

        for round_num in range(1, halving_rounds + 1):
            warmup, repeat = _ROUND_CONFIGS[min(round_num - 1, len(_ROUND_CONFIGS) - 1)]
            self._progress(
                f"  [SHA round {round_num}/{halving_rounds}] {len(surviving)} candidates "
                f"warmup={warmup} repeat={repeat}"
            )

            round_results = []
            for i, params in enumerate(surviving, 1):
                label = f"  [sha r{round_num}.{i}] {params.block_size}/{params.num_warps}"
                exp, cand_bench, bl_bench, bl_name = self._eval_one(
                    spec,
                    params,
                    ctx.bench_input,
                    ctx.cases,
                    ctx.tol,
                    label,
                    warmup=warmup,
                    repeat=repeat,
                )
                if bl_bench is not None:
                    baseline_bench, baseline_name = bl_bench, bl_name
                all_experiments.append(exp)

                if exp.correct and cand_bench is not None:
                    round_results.append((params, cand_bench))

            if not round_results:
                self._progress(f"  [SHA round {round_num}] no valid candidates — stopping")
                break

            # 上位 50% に絞る（最終ラウンドは絞らない）
            round_results.sort(key=lambda t: t[1].median_us)
            if round_num < halving_rounds:
                keep = max(1, len(round_results) // 2)
                surviving = [p for p, _ in round_results[:keep]]
                self._progress(
                    f"  [SHA round {round_num}] keeping top {keep}/{len(round_results)}: "
                    f"best={round_results[0][1].median_us:.1f}us"
                )
            else:
                surviving = [p for p, _ in round_results]

        best_params: SearchParams | None = None
        best_bench: BenchmarkResult | None = None
        # 最終ラウンドの round_results（ソート済み）の先頭が最良。
        # round_results が空（全ラウンド break）の場合は None のまま _finalize に委ねる。
        if round_results:
            best_params, best_bench = round_results[0]
            for exp in all_experiments:
                if exp.params == best_params:
                    exp.is_best = True
                    break

        sha_failed_count = sum(1 for e in all_experiments if not e.success)
        sha_speedup: float | None = None
        if best_bench and baseline_bench and best_bench.median_us > 0:
            sha_speedup = baseline_bench.median_us / best_bench.median_us
        self._finalize(
            ctx,
            best_params,
            best_bench,
            num_candidates=len(all_experiments),
            baseline_us=baseline_bench.median_us if baseline_bench else None,
            notify=True,
            speedup=sha_speedup,
            failed_count=sha_failed_count,
        )

        return SearchResult(
            spec=spec,
            cache_hit=False,
            best_params=best_params,
            best_benchmark=best_bench,
            baseline_benchmark=baseline_bench,
            baseline_name=baseline_name,
            experiments=all_experiments,
            extended_baselines=ctx.extended,
            total_time_s=time.time() - ctx.start_time,
            failed_count=sum(1 for e in all_experiments if not e.success),
            incorrect_count=sum(1 for e in all_experiments if e.success and not e.correct),
        )

    def _eval_one(
        self,
        spec: KernelSpec,
        params: SearchParams,
        bench_input: list[dict[str, Any]],
        cases: list[dict[str, Any]] | None,
        tol: dict,
        label: str,
        warmup: int | None = None,
        repeat: int | None = None,
    ) -> tuple[ExperimentResult, BenchmarkResult | None, BenchmarkResult | None, str | None]:
        """1 候補を評価。戻り値: (experiment, candidate_bench, baseline_bench, baseline_name)。"""
        try:
            code = generate(spec, params)
        except ValueError as e:
            self._progress(f"{label} SKIP: {e}")
            _log.debug("SKIP %s: %s", label, e)
            return ExperimentResult(params, False, False, None, str(e)), None, None, None

        wr: WorkerResult = run_in_worker(
            code,
            spec.op_type,
            bench_input,
            spec.constants,
            correctness_cases=cases,
            task="full",
            warmup=warmup if warmup is not None else self.warmup,
            repeat=repeat if repeat is not None else self.repeat,
            tolerance=tol,
            timeout_s=self.timeout_s,
            python_executable=self.python_executable,
        )

        if not wr.success:
            self._emit(
                ProgressEvent(
                    kind="candidate_fail",
                    label=f"{label} FAIL: {wr.error}",
                    params=params,
                )
            )
            _log.warning("FAIL %s: %s", label, wr.error)
            return ExperimentResult(params, False, False, None, wr.error), None, None, None
        if not wr.correct:
            self._emit(
                ProgressEvent(
                    kind="candidate_incorrect",
                    label=f"{label} INCORRECT",
                    params=params,
                )
            )
            _log.debug("INCORRECT %s", label)
            return ExperimentResult(params, True, False, None, "incorrect"), None, None, None

        assert wr.candidate is not None and wr.baseline is not None
        speedup: float | None = None
        if wr.baseline.median_us > 0 and wr.candidate.median_us > 0:
            speedup = wr.baseline.median_us / wr.candidate.median_us
        self._emit(
            ProgressEvent(
                kind="candidate_ok",
                label=f"{label}/{params.acc_dtype} -> {wr.candidate.median_us:.1f}us",
                params=params,
                median_us=wr.candidate.median_us,
                speedup=speedup,
            )
        )
        return (
            ExperimentResult(params, True, True, wr.candidate.median_us, None),
            wr.candidate,
            wr.baseline,
            wr.baseline_name,
        )
