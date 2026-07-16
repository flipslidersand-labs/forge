from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from forge.benchmark.statistics import BenchmarkResult, is_improvement
from forge.cache.key import CacheKey
from forge.cache.repository import CachedKernel, KernelRepository
from forge.codegen.triton_codegen import generate
from forge.ir.kernel_spec import KernelSpec
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

if TYPE_CHECKING:
    from forge.search.llm_generator import LLMGenerator, TokenUsage


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
        progress: Callable[[str], None] | None = None,
        measure_extended: bool = False,
    ) -> None:
        self.repo = repo or KernelRepository()
        self.python_executable = python_executable
        self.min_speedup = min_speedup
        self.warmup = warmup
        self.repeat = repeat
        self.timeout_s = timeout_s
        self._progress = progress or (lambda _msg: None)
        self.measure_extended = measure_extended

    def optimize(
        self,
        spec: KernelSpec,
        budget: int = 50,
        search: CandidateGenerator | None = None,
        use_cache: bool = True,
    ) -> SearchResult:
        spec.validate()
        key = CacheKey.from_spec_and_env(spec)

        if use_cache and (cached := self.repo.get(key)) is not None:
            self._progress(f"cache HIT: {cached.params}")
            bench = BenchmarkResult.from_dict(cached.benchmark_json)
            return SearchResult(
                spec=spec,
                cache_hit=True,
                best_params=SearchParams.from_dict(cached.params),
                best_benchmark=bench,
                baseline_benchmark=None,
                baseline_name=None,
                experiments=[],
            )

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
                else:
                    self._progress(
                        f"  extended: {eb.name} median={eb.benchmark.median_us:.1f}µs "
                        f"p95={eb.benchmark.p95_us:.1f}µs compile={eb.compile_time_s:.1f}s"
                    )

        search = search or GridSearch()
        candidates = search.generate(spec, key.compute_capability, budget=budget)
        self._progress(f"searching {len(candidates)} candidates (cc {key.compute_capability})")

        experiments: list[ExperimentResult] = []
        best_params: SearchParams | None = None
        best_bench: BenchmarkResult | None = None
        baseline_bench: BenchmarkResult | None = None
        baseline_name: str | None = None

        for i, params in enumerate(candidates, 1):
            label = f"[{i}/{len(candidates)}] {params.block_size}/{params.num_warps}"
            exp, cand_bench, bl_bench, bl_name = self._eval_one(
                spec, params, bench_input, cases, tol, label
            )
            if bl_bench is not None:
                baseline_bench, baseline_name = bl_bench, bl_name

            if exp.correct and cand_bench is not None:
                improved = best_bench is None or is_improvement(
                    cand_bench, best_bench, self.min_speedup
                )
                if improved:
                    best_params, best_bench = params, cand_bench
                    exp.is_best = True
                    self._progress(
                        f"{label}/{params.acc_dtype} -> {cand_bench.median_us:.1f}us BEST"
                    )
                else:
                    self._progress(f"{label}/{params.acc_dtype} -> {cand_bench.median_us:.1f}us")
            experiments.append(exp)

        if best_params is not None and best_bench is not None:
            code = generate(spec, best_params)
            self.repo.put(
                key,
                CachedKernel(
                    cache_key=key,
                    params=best_params.to_dict(),
                    kernel_code=code,
                    benchmark_json=best_bench.to_dict(),
                    created_at=datetime.now(UTC),
                ),
            )
            self._progress(f"cached best: {best_params} ({best_bench.median_us:.1f}us)")

        return SearchResult(
            spec=spec,
            cache_hit=False,
            best_params=best_params,
            best_benchmark=best_bench,
            baseline_benchmark=baseline_bench,
            baseline_name=baseline_name,
            experiments=experiments,
            extended_baselines=extended,
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
        spec.validate()
        key = CacheKey.from_spec_and_env(spec)

        if use_cache and (cached := self.repo.get(key)) is not None:
            self._progress(f"cache HIT: {cached.params}")
            bench = BenchmarkResult.from_dict(cached.benchmark_json)
            return MultiRoundResult(
                spec=spec,
                rounds=[],
                best_params=SearchParams.from_dict(cached.params),
                best_benchmark=bench,
                baseline_benchmark=None,
                baseline_name=None,
                token_usage=llm.token_usage,
            )

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
                else:
                    self._progress(
                        f"  extended: {eb.name} median={eb.benchmark.median_us:.1f}µs "
                        f"p95={eb.benchmark.p95_us:.1f}µs compile={eb.compile_time_s:.1f}s"
                    )

        llm.reset_usage()
        history: list[HistoryEntry] = []
        rounds: list[RoundResult] = []
        overall_best_params: SearchParams | None = None
        overall_best_bench: BenchmarkResult | None = None
        baseline_bench: BenchmarkResult | None = None
        baseline_name: str | None = None
        seen_params: set[SearchParams] = set()

        for round_num in range(1, n_rounds + 1):
            self._progress(
                f"[round {round_num}/{n_rounds}] generating {candidates_per_round} candidates "
                f"(history={len(history)})"
            )
            candidates = llm.generate(
                spec,
                key.compute_capability,
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
                    spec, params, bench_input, cases, tol, label
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

            rounds.append(
                RoundResult(
                    round_num=round_num,
                    experiments=round_experiments,
                    best_params=round_best_params,
                    best_median_us=round_best_us,
                )
            )
            self._progress(
                f"  round {round_num} done: best={round_best_us:.1f}us"
                if round_best_us is not None
                else f"  round {round_num} done: no valid candidates"
            )

        if overall_best_params is not None and overall_best_bench is not None:
            code = generate(spec, overall_best_params)
            self.repo.put(
                key,
                CachedKernel(
                    cache_key=key,
                    params=overall_best_params.to_dict(),
                    kernel_code=code,
                    benchmark_json=overall_best_bench.to_dict(),
                    created_at=datetime.now(UTC),
                ),
            )
            self._progress(
                f"cached best: {overall_best_params} ({overall_best_bench.median_us:.1f}us)"
            )

        total = sum(len(r.experiments) for r in rounds)
        return MultiRoundResult(
            spec=spec,
            rounds=rounds,
            best_params=overall_best_params,
            best_benchmark=overall_best_bench,
            baseline_benchmark=baseline_bench,
            baseline_name=baseline_name,
            token_usage=llm.token_usage,
            total_candidates_evaluated=total,
            extended_baselines=extended,
        )

    def _eval_one(
        self,
        spec: KernelSpec,
        params: SearchParams,
        bench_input: object,
        cases: object,
        tol: dict,
        label: str,
    ) -> tuple[ExperimentResult, BenchmarkResult | None, BenchmarkResult | None, str | None]:
        """1 候補を評価。戻り値: (experiment, candidate_bench, baseline_bench, baseline_name)。"""
        try:
            code = generate(spec, params)
        except ValueError as e:
            self._progress(f"{label} SKIP: {e}")
            return ExperimentResult(params, False, False, None, str(e)), None, None, None

        wr: WorkerResult = run_in_worker(
            code,
            spec.op_type,
            bench_input,
            spec.constants,
            correctness_cases=cases,
            task="full",
            warmup=self.warmup,
            repeat=self.repeat,
            tolerance=tol,
            timeout_s=self.timeout_s,
            python_executable=self.python_executable,
        )

        if not wr.success:
            self._progress(f"{label} FAIL: {wr.error}")
            return ExperimentResult(params, False, False, None, wr.error), None, None, None
        if not wr.correct:
            self._progress(f"{label} INCORRECT")
            return ExperimentResult(params, True, False, None, "incorrect"), None, None, None

        assert wr.candidate is not None and wr.baseline is not None
        return (
            ExperimentResult(params, True, True, wr.candidate.median_us, None),
            wr.candidate,
            wr.baseline,
            wr.baseline_name,
        )
