"""Subprocess ワーカープロセスの結果型・JSON (de)serialization。

Triton カーネルの検証・ベンチマークは子プロセス（ワーカー）で隔離して実行。
親プロセス（orchestrator）はワーカーの結果を JSON で受け取り、
BenchmarkResult や ExtendedBaselineResult に変換する。

このモジュールは結果型と serialization ロジックを集約。
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import warnings
from dataclasses import dataclass, field
from typing import Any

from forge.benchmark.statistics import BenchmarkResult

_log = logging.getLogger("forge.runtime")

_STDERR_LIMIT = 4096


def _tail(text: str, limit: int = _STDERR_LIMIT) -> str:
    """text が limit 文字を超える場合は末尾 limit 文字だけ残す。

    Python トレースバックや Triton のコンパイルエラーは原因の行が末尾に出るため、
    先頭ではなく末尾を残す。切り詰めた場合は先頭に印を付ける。
    """
    if len(text) <= limit:
        return text
    return "...(truncated)\n" + text[-limit:]


def _head_tail(text: str, limit: int = _STDERR_LIMIT) -> str:
    """text が limit 文字を超える場合は先頭と末尾を limit//2 文字ずつ残し、間を省略する。

    JSON 解析失敗時の stdout 用。パース対象は最終行なので、先行ノイズが長いと
    失敗した最終行は末尾側にあり、最終行自体が長く途中で切れているときは
    success / error などの先頭側のフィールドが診断に要るため、両端を残す。
    """
    if len(text) <= limit:
        return text
    half = limit // 2
    omitted = len(text) - 2 * half
    return f"{text[:half]}\n...(truncated {omitted} chars)...\n{text[-half:]}"


def warn_python_executable_deprecated(stacklevel: int = 3) -> None:
    """python_executable 引数の非推奨警告（次の major リリースで削除予定）。"""
    warnings.warn(
        "python_executable は非推奨で、次の major リリースで削除されます。"
        "ワーカーは現在のインタプリタ（sys.executable）で起動する形に統一する予定です。",
        DeprecationWarning,
        stacklevel=stacklevel,
    )


@dataclass
class ExtendedBaselineResult:
    """torch.compile 等の重い baseline を 1 回だけ計測した結果。

    計測に失敗した場合は ``error`` フィールドに理由が入り、``benchmark`` の値は
    すべて 0 になる。利用側は ``eb.failed`` を確認してから数値を使うこと。
    """

    name: str
    benchmark: BenchmarkResult
    compile_time_s: float = 0.0
    error: str | None = None

    @property
    def failed(self) -> bool:
        return self.error is not None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "name": self.name,
            "benchmark": self.benchmark.to_dict(),
            "compile_time_s": self.compile_time_s,
        }
        if self.error is not None:
            d["error"] = self.error
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ExtendedBaselineResult:
        return cls(
            name=str(d["name"]),
            benchmark=BenchmarkResult.from_dict(d["benchmark"]),
            compile_time_s=float(d.get("compile_time_s", 0.0)),
            error=d.get("error"),
        )


@dataclass
class WorkerResult:
    success: bool
    correct: bool = False
    max_abs_diff: float | None = None
    failures: list[dict[str, Any]] = field(default_factory=list)
    candidate: BenchmarkResult | None = None
    baseline: BenchmarkResult | None = None
    baseline_name: str | None = None
    error: str | None = None
    error_type: str | None = None  # ワーカーが返す例外種別。OOM は "cuda_oom"
    traceback: str | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> WorkerResult:
        if not d.get("success"):
            return cls(
                success=False,
                error=d.get("error"),
                error_type=d.get("error_type"),
                traceback=d.get("traceback"),
            )
        return cls(
            success=True,
            correct=bool(d.get("correct", False)),
            max_abs_diff=d.get("max_abs_diff"),
            failures=list(d.get("failures", [])),
            candidate=BenchmarkResult.from_dict(d["candidate"]) if "candidate" in d else None,
            baseline=BenchmarkResult.from_dict(d["baseline"]) if "baseline" in d else None,
            baseline_name=d.get("baseline_name"),
        )


def run_in_worker(
    kernel_code: str,
    op_type: str,
    benchmark_input: list[dict[str, Any]],
    constants: dict[str, Any],
    correctness_cases: list[dict[str, Any]] | None = None,
    task: str = "full",
    warmup: int = 25,
    repeat: int = 200,
    tolerance: dict[str, Any] | None = None,
    timeout_s: float = 60.0,
    python_executable: str | None = None,
) -> WorkerResult:
    """生成カーネルを使い捨て subprocess で実行する。

    benchmark_input はタイミング計測に使う代表入力。correctness_cases を渡すと
    各ケースで正確性を検証する（省略時は benchmark_input で 1 回検証）。
    CUDA エラー・タイムアウト・コンパイル失敗のいずれも WorkerResult(success=False)
    として返し、親プロセスは生き残る。python_executable 省略時は現在のインタプリタ。
    python_executable は非推奨（次の major で削除）。
    """
    payload = json.dumps(
        {
            "kernel_code": kernel_code,
            "op_type": op_type,
            "benchmark_input": benchmark_input,
            "correctness_cases": correctness_cases,
            "constants": constants,
            "task": task,
            "warmup": warmup,
            "repeat": repeat,
            "tolerance": tolerance or {"atol": 2e-3, "rtol": 1e-2, "equal_nan": False},
        }
    )
    exe = python_executable or sys.executable
    try:
        proc = subprocess.run(
            [exe, "-m", "forge.runtime._worker_entry"],
            input=payload,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        _log.warning("worker timeout op=%s after %.1fs", op_type, timeout_s)
        return WorkerResult(success=False, error=f"timeout after {timeout_s}s")

    if proc.returncode != 0:
        stderr = proc.stderr.strip()
        detail = _tail(stderr) if stderr else f"exit {proc.returncode}"
        _log.warning("worker crashed op=%s rc=%d", op_type, proc.returncode)
        return WorkerResult(success=False, error=f"worker crashed: {detail}")

    stdout = proc.stdout.strip()
    if not stdout:
        stderr = proc.stderr.strip()
        detail = _tail(stderr) if stderr else "empty stdout"
        _log.warning("bad worker output op=%s: empty stdout", op_type)
        return WorkerResult(success=False, error=f"bad worker output: {detail}")
    try:
        parsed = json.loads(stdout.splitlines()[-1])
        if not isinstance(parsed, dict):
            raise TypeError(f"expected JSON object, got {type(parsed).__name__}")
        return WorkerResult.from_dict(parsed)
    except (json.JSONDecodeError, IndexError, KeyError, TypeError) as exc:
        snippet = _head_tail(stdout)
        _log.warning("bad worker output op=%s: %r", op_type, exc)
        return WorkerResult(success=False, error=f"bad worker output: {exc!r} stdout={snippet!r}")


def run_extended_baseline_in_worker(
    op_type: str,
    benchmark_input: list[dict[str, Any]],
    constants: dict[str, Any],
    warmup: int = 25,
    repeat: int = 200,
    timeout_s: float = 180.0,
    python_executable: str | None = None,
) -> list[ExtendedBaselineResult]:
    """torch.compile(reference) を subprocess で 1 回だけ計測する。

    GPU 必須。torch.compile の初回コンパイル時間も記録する。
    失敗した場合は空リストを返す（探索は継続できる）。
    """
    payload = json.dumps(
        {
            "op_type": op_type,
            "benchmark_input": benchmark_input,
            "constants": constants,
            "task": "extended_baseline",
            "warmup": warmup,
            "repeat": repeat,
        }
    )
    exe = python_executable or sys.executable
    try:
        proc = subprocess.run(
            [exe, "-m", "forge.runtime._worker_entry"],
            input=payload,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        _log.warning("extended_baseline worker timeout op=%s after %.1fs", op_type, timeout_s)
        return []

    if proc.returncode != 0:
        stderr = proc.stderr.strip()
        detail = _tail(stderr) if stderr else f"exit {proc.returncode}"
        _log.warning(
            "extended_baseline worker crashed op=%s rc=%d: %s", op_type, proc.returncode, detail
        )
        return []

    try:
        result = json.loads(proc.stdout.strip().splitlines()[-1])
        if not isinstance(result, dict):
            raise TypeError(f"expected JSON object, got {type(result).__name__}")
        if not result.get("success"):
            err = result.get("error", "unknown error")
            _log.warning("extended_baseline worker reported failure op=%s: %s", op_type, err)
            return []
        return [ExtendedBaselineResult.from_dict(b) for b in result.get("baselines", [])]
    except (json.JSONDecodeError, IndexError, KeyError, TypeError) as exc:
        snippet = _head_tail(proc.stdout.strip())
        _log.warning(
            "extended_baseline bad worker output op=%s: %r stdout=%r", op_type, exc, snippet
        )
        return []
