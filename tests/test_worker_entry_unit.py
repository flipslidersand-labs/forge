"""_worker_entry.main() の例外処理を直接テスト (#220)。GPU 不要。

KeyboardInterrupt / SystemExit が except ブロックで握り潰されないこと、
および RuntimeError が error_type フィールド付きで JSON 出力されることを検証する。
"""

from __future__ import annotations

import io
import json
import sys

import pytest

from forge.runtime._worker_entry import _build_tensor as _REAL_BUILD_TENSOR


def _make_payload(**overrides) -> str:
    base = {
        "op_type": "rmsnorm",
        "kernel_code": "pass",
        "benchmark_input": [{"shape": [4], "dtype": "float16", "init": "randn", "seed": 0}],
    }
    base.update(overrides)
    return json.dumps(base)


def test_keyboard_interrupt_propagates(monkeypatch) -> None:
    """KeyboardInterrupt は main() の except ブロックを素通りする。"""
    import forge.runtime.loader as loader_mod
    from forge.runtime._worker_entry import main

    def raise_ki(code: str):
        raise KeyboardInterrupt

    monkeypatch.setattr(sys, "stdin", io.StringIO(_make_payload()))
    monkeypatch.setattr(loader_mod, "load_kernel_fn", raise_ki)

    with pytest.raises(KeyboardInterrupt):
        main()


def test_system_exit_propagates(monkeypatch) -> None:
    """SystemExit は main() の except ブロックを素通りする。"""
    import forge.runtime.loader as loader_mod
    from forge.runtime._worker_entry import main

    def raise_exit(code: str):
        raise SystemExit(1)

    monkeypatch.setattr(sys, "stdin", io.StringIO(_make_payload()))
    monkeypatch.setattr(loader_mod, "load_kernel_fn", raise_exit)

    with pytest.raises(SystemExit):
        main()


def test_runtime_error_captured_with_error_type(monkeypatch, capsys) -> None:
    """RuntimeError は except ブロックで捕捉され error_type フィールドを含む JSON を出力する。"""
    import forge.runtime.loader as loader_mod
    from forge.runtime._worker_entry import main

    def raise_runtime(code: str):
        raise RuntimeError("synthetic error")

    monkeypatch.setattr(sys, "stdin", io.StringIO(_make_payload()))
    monkeypatch.setattr(loader_mod, "load_kernel_fn", raise_runtime)

    main()  # should not raise

    out = capsys.readouterr().out.strip()
    result = json.loads(out)
    assert result["success"] is False
    assert "synthetic error" in result["error"]
    assert result["error_type"] == "RuntimeError"


def test_unknown_op_type_returns_error(monkeypatch, capsys) -> None:
    """未知の op_type は OP_REGISTRY 検証で弾かれ success=False を返す (#266)。"""
    from forge.runtime._worker_entry import main

    payload = json.dumps(
        {
            "op_type": "../../evil",
            "kernel_code": "pass",
            "benchmark_input": [{"shape": [4], "dtype": "float16", "init": "randn", "seed": 0}],
        }
    )
    monkeypatch.setattr(sys, "stdin", io.StringIO(payload))

    main()

    out = capsys.readouterr().out.strip()
    result = json.loads(out)
    assert result["success"] is False
    assert "unknown op_type" in result["error"]
    assert "../../evil" in result["error"]


@pytest.mark.parametrize("exc", [TypeError("bad arg"), KeyError("missing")])
def test_non_listed_exception_returns_structured_error(monkeypatch, capsys, exc) -> None:
    """TypeError / KeyError 等も未捕捉でプロセスを落とさず、構造化 JSON で返す。"""
    import forge.runtime.loader as loader_mod
    from forge.runtime._worker_entry import main

    def raise_it(code: str):
        raise exc

    monkeypatch.setattr(sys, "stdin", io.StringIO(_make_payload()))
    monkeypatch.setattr(loader_mod, "load_kernel_fn", raise_it)

    main()  # should not raise

    result = json.loads(capsys.readouterr().out.strip())
    assert result["success"] is False
    assert result["error_type"] == type(exc).__name__
    assert type(exc).__name__ in result["error"]
    assert "Traceback" in result["traceback"]


def test_exception_class_named_out_of_memory_maps_to_cuda_oom(monkeypatch, capsys) -> None:
    """RuntimeError 派生でなくても OutOfMemoryError という名前なら cuda_oom になる。"""
    import forge.runtime.loader as loader_mod
    from forge.runtime._worker_entry import main

    class OutOfMemoryError(Exception):
        pass

    def raise_oom(code: str):
        raise OutOfMemoryError("CUDA out of memory")

    monkeypatch.setattr(sys, "stdin", io.StringIO(_make_payload()))
    monkeypatch.setattr(loader_mod, "load_kernel_fn", raise_oom)

    main()

    result = json.loads(capsys.readouterr().out.strip())
    assert result["error_type"] == "cuda_oom"


def test_no_base_exception_catch_in_source() -> None:
    """BaseException / bare except は使わない（KeyboardInterrupt・SystemExit を握り潰さない）。"""
    import inspect
    import re

    import forge.runtime._worker_entry as mod

    source = inspect.getsource(mod)
    assert "except BaseException" not in source
    assert not re.search(r"except\s*:", source), "bare except が残っています"


# ---------------------------------------------------------------------------
# #478: main() の判定ロジック（正確性集約・task 分岐・extended_baseline・OOM 分類）と
# _build_tensor。_build_tensor / load_kernel_fn / measure / torch.cuda.synchronize を
# CPU 上の fake に差し替えるので GPU・CUDA は不要。
# ---------------------------------------------------------------------------

_BENCH_KEYS = ("candidate", "baseline", "baseline_name")


class _FakeBench:
    """measure() の戻り値の代役。to_dict() だけ持つ。"""

    def __init__(self, median_us: float) -> None:
        self.median_us = median_us

    def to_dict(self) -> dict:
        return {"median_us": self.median_us, "p20_us": 1.0, "p80_us": 3.0, "p95_us": 4.0}


class _Env:
    """main() を CPU 上で回すための差し替え一式。呼び出し履歴を保持する。"""

    def __init__(self) -> None:
        self.kernel_inputs: list[float] = []  # kernel_fn が受け取った入力の先頭要素
        self.baseline_inputs: list[float] = []
        self.measure_calls: list[tuple[int, int]] = []  # (warmup, repeat)
        self.load_calls: list[str] = []
        self.kernel_fn = None  # テストごとに差し替え可
        self.reference_fn = None
        self.measure_raises: BaseException | None = None


def _identity_kernel(x, **kwargs):
    return x.clone()


def _identity_reference(x, **kwargs):
    return x.clone()


@pytest.fixture
def env(monkeypatch, capsys) -> _Env:
    import torch

    import forge.benchmark.timer as timer_mod
    import forge.runtime._worker_entry as entry_mod
    import forge.runtime.loader as loader_mod
    import forge.runtime.reference as reference_mod

    e = _Env()
    e.kernel_fn = _identity_kernel
    e.reference_fn = _identity_reference

    def fake_build_tensor(spec, torch_mod):
        # spec["fill"] で埋めた CPU テンソル。"fill" 無しなら 1.0。
        return torch.full(tuple(spec["shape"]), float(spec.get("fill", 1.0)))

    def fake_load_kernel_fn(code: str):
        e.load_calls.append(code)

        def kernel(x, **kwargs):
            e.kernel_inputs.append(float(x.flatten()[0]))
            return e.kernel_fn(x, **kwargs)

        return kernel

    def fake_get_reference(op_type: str):
        return lambda x, **kwargs: e.reference_fn(x, **kwargs)

    def fake_get_baseline(op_type: str):
        def baseline(x, **kwargs):
            e.baseline_inputs.append(float(x.flatten()[0]))
            return x.clone()

        return baseline

    def fake_measure(fn, warmup=25, repeat=200):
        if e.measure_raises is not None:
            raise e.measure_raises
        e.measure_calls.append((warmup, repeat))
        fn()
        # 1 回目 = candidate、2 回目 = baseline を median_us で区別できるようにする
        return _FakeBench(float(len(e.measure_calls)))

    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    monkeypatch.setattr(entry_mod, "_build_tensor", fake_build_tensor)
    monkeypatch.setattr(loader_mod, "load_kernel_fn", fake_load_kernel_fn)
    monkeypatch.setattr(reference_mod, "get_reference", fake_get_reference)
    monkeypatch.setattr(reference_mod, "get_baseline", fake_get_baseline)
    monkeypatch.setattr(timer_mod, "measure", fake_measure)
    return e


def _run_main(monkeypatch, capsys, payload: dict | str) -> dict:
    from forge.runtime._worker_entry import main

    raw = payload if isinstance(payload, str) else json.dumps(payload)
    monkeypatch.setattr(sys, "stdin", io.StringIO(raw))
    main()
    return json.loads(capsys.readouterr().out.strip())


def _spec(fill: float = 1.0) -> dict:
    return {"shape": [4], "dtype": "float32", "fill": fill}


def _payload(**overrides) -> dict:
    base: dict = {
        "op_type": "rmsnorm",
        "kernel_code": "# fake",
        "benchmark_input": [_spec(7.0)],
        "warmup": 3,
        "repeat": 5,
    }
    base.update(overrides)
    return base


def _cases(*names: str) -> list[dict]:
    return [{"name": n, "input_specs": [_spec(1.0)]} for n in names]


# --- 正確性集約 ------------------------------------------------------------


def test_all_cases_match_runs_benchmark(env, monkeypatch, capsys) -> None:
    """全ケース一致なら correct=True で、candidate/baseline/baseline_name が出る。"""
    from forge.runtime.reference import baseline_name

    result = _run_main(
        monkeypatch,
        capsys,
        _payload(correctness_cases=_cases("a", "b", "c"), constants={"eps": 1e-6}),
    )

    assert result["success"] is True
    assert result["correct"] is True
    assert result["failures"] == []
    assert result["max_abs_diff"] == 0.0
    # candidate が先、baseline が後。warmup/repeat は payload の値がそのまま渡る
    assert env.measure_calls == [(3, 5), (3, 5)]
    assert result["candidate"]["median_us"] == 1.0
    assert result["baseline"]["median_us"] == 2.0
    assert result["baseline_name"] == baseline_name("rmsnorm")
    assert result["baseline_name"]  # 空でない
    # ベンチの入力は benchmark_input（fill=7.0）。正確性ケースの入力（fill=1.0）ではない
    assert env.kernel_inputs == [1.0, 1.0, 1.0, 7.0]
    assert env.baseline_inputs == [7.0]


def test_max_abs_diff_is_max_over_cases(env, monkeypatch, capsys) -> None:
    """許容内の差は failures に入らず、max_abs_diff は全ケースの最大値になる。"""

    def kernel(x, **kwargs):
        # case ごとに fill を変えて差を変える（0.001 と 0.0005 はどちらも atol=2e-3 以内）
        return x + (0.001 if float(x[0]) == 2.0 else 0.0005)

    env.kernel_fn = kernel
    cases = [
        {"name": "a", "input_specs": [_spec(1.0)]},
        {"name": "b", "input_specs": [_spec(2.0)]},
        {"name": "c", "input_specs": [_spec(3.0)]},
    ]
    result = _run_main(monkeypatch, capsys, _payload(correctness_cases=cases))

    assert result["correct"] is True
    assert result["failures"] == []
    assert result["max_abs_diff"] == pytest.approx(0.001, abs=1e-6)


def test_single_mismatching_case_reported_and_not_benchmarked(env, monkeypatch, capsys) -> None:
    """1 ケースだけ不一致なら failures にその case 名と max_diff が入り、ベンチされない。"""

    def kernel(x, **kwargs):
        return x + 0.5 if float(x[0]) == 5.0 else x.clone()

    env.kernel_fn = kernel
    cases = [
        {"name": "ok_first", "input_specs": [_spec(1.0)]},
        {"name": "bad", "input_specs": [_spec(5.0)]},
        {"name": "ok_last", "input_specs": [_spec(2.0)]},
    ]
    result = _run_main(monkeypatch, capsys, _payload(correctness_cases=cases))

    assert result["success"] is True
    assert result["correct"] is False
    assert [f["case"] for f in result["failures"]] == ["bad"]
    assert result["failures"][0]["max_diff"] == pytest.approx(0.5)
    assert result["max_abs_diff"] == pytest.approx(0.5)
    for key in _BENCH_KEYS:
        assert key not in result
    assert env.measure_calls == []
    assert env.baseline_inputs == []
    # 不一致があっても後続ケースは評価される（ok_last まで回る）
    assert env.kernel_inputs == [1.0, 5.0, 2.0]


def test_multiple_failures_listed_in_case_order(env, monkeypatch, capsys) -> None:
    def kernel(x, **kwargs):
        return x + float(x[0]) * 0.1  # fill=1 -> 差 0.1、fill=2 -> 差 0.2、fill=0 -> 差 0

    env.kernel_fn = kernel
    cases = [
        {"name": "f1", "input_specs": [_spec(1.0)]},
        {"name": "pass", "input_specs": [_spec(0.0)]},
        {"name": "f2", "input_specs": [_spec(2.0)]},
    ]
    result = _run_main(monkeypatch, capsys, _payload(correctness_cases=cases))

    assert result["correct"] is False
    assert [f["case"] for f in result["failures"]] == ["f1", "f2"]
    assert result["failures"][0]["max_diff"] == pytest.approx(0.1, abs=1e-5)
    assert result["failures"][1]["max_diff"] == pytest.approx(0.2, abs=1e-5)
    assert result["max_abs_diff"] == pytest.approx(0.2, abs=1e-5)


@pytest.mark.parametrize("cases", [None, []])
def test_without_correctness_cases_uses_primary_from_benchmark_input(
    env, monkeypatch, capsys, cases
) -> None:
    """correctness_cases が無い（または空）なら benchmark_input を 'primary' ケースとして検証する。"""
    env.kernel_fn = lambda x, **kwargs: x + 0.5
    extra = {} if cases is None else {"correctness_cases": cases}
    result = _run_main(monkeypatch, capsys, _payload(**extra))

    assert result["correct"] is False
    assert [f["case"] for f in result["failures"]] == ["primary"]
    assert env.kernel_inputs == [7.0]  # benchmark_input（fill=7.0）で評価された


def test_tolerance_from_payload_is_applied(env, monkeypatch, capsys) -> None:
    """差 0.5 は既定の許容では不一致、payload の atol を広げれば一致になる。"""
    env.kernel_fn = lambda x, **kwargs: x + 0.5
    strict = _run_main(monkeypatch, capsys, _payload(correctness_cases=_cases("a")))
    loose = _run_main(
        monkeypatch,
        capsys,
        _payload(
            correctness_cases=_cases("a"),
            tolerance={"atol": 1.0, "rtol": 0.0, "equal_nan": False},
        ),
    )

    assert strict["correct"] is False
    assert loose["correct"] is True
    assert loose["max_abs_diff"] == pytest.approx(0.5)


def test_equal_nan_flag_is_forwarded(env, monkeypatch, capsys) -> None:
    """NaN 同士は equal_nan=False なら不一致、True なら一致。"""
    nan_fn = lambda x, **kwargs: x * float("nan")  # noqa: E731
    env.kernel_fn = nan_fn
    env.reference_fn = nan_fn
    tol = {"atol": 2e-3, "rtol": 1e-2}
    off = _run_main(
        monkeypatch,
        capsys,
        _payload(correctness_cases=_cases("a"), tolerance={**tol, "equal_nan": False}),
    )
    on = _run_main(
        monkeypatch,
        capsys,
        _payload(correctness_cases=_cases("a"), tolerance={**tol, "equal_nan": True}),
    )

    assert off["correct"] is False
    assert on["correct"] is True


# --- task 分岐 -------------------------------------------------------------


@pytest.mark.parametrize(
    ("task", "benchmarks"),
    [("correctness", False), ("benchmark", True), ("full", True), (None, True)],
)
def test_task_controls_benchmark(env, monkeypatch, capsys, task, benchmarks) -> None:
    """ベンチするのは task が benchmark/full（未指定=full）で correct のときだけ。"""
    extra = {} if task is None else {"task": task}
    result = _run_main(monkeypatch, capsys, _payload(correctness_cases=_cases("a"), **extra))

    assert result["success"] is True
    assert result["correct"] is True
    assert all((key in result) for key in _BENCH_KEYS) is benchmarks
    assert any((key in result) for key in _BENCH_KEYS) is benchmarks
    assert len(env.measure_calls) == (2 if benchmarks else 0)


@pytest.mark.parametrize("task", ["correctness", "benchmark", "full"])
def test_incorrect_candidate_never_benchmarked(env, monkeypatch, capsys, task) -> None:
    env.kernel_fn = lambda x, **kwargs: x + 1.0
    result = _run_main(monkeypatch, capsys, _payload(correctness_cases=_cases("a"), task=task))

    assert result["correct"] is False
    for key in _BENCH_KEYS:
        assert key not in result
    assert env.measure_calls == []


# --- extended_baseline -----------------------------------------------------


@pytest.fixture
def compile_env(env, monkeypatch):
    """torch.compile を差し替え可能にした env。extended_baseline 用。"""
    import torch

    state = {"compile_calls": [], "compiled_args": []}

    def fake_compile(fn, *a, **k):
        state["compile_calls"].append(fn)

        def compiled(*args, **kwargs):
            state["compiled_args"].append((args, kwargs))
            return fn(*args, **kwargs)

        return compiled

    monkeypatch.setattr(torch, "compile", fake_compile)
    env.reference_fn = lambda x, scale=1.0: x * scale
    env.state = state
    return env


def _ext_payload(**overrides) -> dict:
    return _payload(task="extended_baseline", constants={"scale": 2.0}, **overrides)


def test_extended_baseline_success(compile_env, monkeypatch, capsys) -> None:
    env = compile_env
    result = _run_main(monkeypatch, capsys, _ext_payload())

    assert result["success"] is True
    assert result["task"] == "extended_baseline"
    assert len(result["baselines"]) == 1
    b = result["baselines"][0]
    assert b["name"] == "torch.compile(reference)"
    assert b["benchmark"]["median_us"] == 1.0  # fake measure の 1 回目
    assert isinstance(b["compile_time_s"], float)
    assert b["compile_time_s"] >= 0.0
    assert "error" not in b
    # 初回呼び出し(compile) + measure 内の 1 回、どちらも constants を kwargs で受ける
    assert [kw for _, kw in env.state["compiled_args"]] == [{"scale": 2.0}, {"scale": 2.0}]
    assert env.measure_calls == [(3, 5)]
    # 入力は benchmark_input から作られる
    assert float(env.state["compiled_args"][0][0][0][0]) == 7.0
    # kernel_code は使わない（ロードも正確性検証もしない）
    assert env.load_calls == []
    assert env.kernel_inputs == []
    for key in ("correct", "failures", *_BENCH_KEYS):
        assert key not in result


def test_extended_baseline_does_not_require_kernel_code(compile_env, monkeypatch, capsys) -> None:
    payload = _ext_payload()
    del payload["kernel_code"]
    result = _run_main(monkeypatch, capsys, payload)

    assert result["success"] is True
    assert result["baselines"][0]["name"] == "torch.compile(reference)"


@pytest.mark.parametrize(
    ("where", "exc"),
    [
        ("compile", RuntimeError),
        ("first_call", RuntimeError),
        ("first_call", OSError),
        ("first_call", ValueError),
        ("measure", RuntimeError),
    ],
)
def test_extended_baseline_compile_failure_returns_zero_benchmark_with_error(
    compile_env, monkeypatch, capsys, where, exc
) -> None:
    """torch.compile 系の失敗は success=True のまま benchmark=0 と error 付き baseline で返る。"""
    import torch

    env = compile_env
    boom = exc("synthetic compile failure")
    if where == "compile":

        def bad_compile(fn, *a, **k):
            raise boom

        monkeypatch.setattr(torch, "compile", bad_compile)
    elif where == "first_call":
        monkeypatch.setattr(torch, "compile", lambda fn, *a, **k: _raiser(boom))
    else:
        env.measure_raises = boom

    result = _run_main(monkeypatch, capsys, _ext_payload())

    assert result["success"] is True
    assert result["task"] == "extended_baseline"
    (b,) = result["baselines"]
    assert b["name"] == "torch.compile(reference)"
    assert b["benchmark"] == {"median_us": 0.0, "p20_us": 0.0, "p80_us": 0.0, "p95_us": 0.0}
    assert b["compile_time_s"] == 0.0
    assert "synthetic compile failure" in b["error"]


def _raiser(exc: BaseException):
    def f(*a, **k):
        raise exc

    return f


@pytest.mark.parametrize("exc", [AttributeError, TypeError, KeyError], ids=lambda e: e.__name__)
def test_extended_baseline_unexpected_exception_is_warning_not_failure(
    compile_env, monkeypatch, capsys, exc
) -> None:
    """想定外の例外でも全体は success=True。error 付きの 0 計測 baseline として返る。"""
    import torch

    monkeypatch.setattr(torch, "compile", lambda fn, *a, **k: _raiser(exc("nope")))
    result = _run_main(monkeypatch, capsys, _ext_payload())

    assert result["success"] is True
    assert result["task"] == "extended_baseline"
    (b,) = result["baselines"]
    assert b["name"] == "torch.compile(reference)"
    assert b["benchmark"] == {"median_us": 0.0, "p20_us": 0.0, "p80_us": 0.0, "p95_us": 0.0}
    assert b["compile_time_s"] == 0.0
    assert b["error"].startswith(f"{exc.__name__}:")
    assert "nope" in b["error"]
    # 親(ExtendedBaselineResult.from_dict)が読む形であること
    from forge.runtime.worker import ExtendedBaselineResult

    assert ExtendedBaselineResult.from_dict(b).failed is True


# --- エラー分類 ------------------------------------------------------------


def test_oom_in_kernel_classified_as_cuda_oom(env, monkeypatch, capsys) -> None:
    """OutOfMemoryError（型名で判定）は error_type='cuda_oom' に写される。"""
    import torch

    def kernel(x, **kwargs):
        raise torch.cuda.OutOfMemoryError("CUDA out of memory. synthetic")

    env.kernel_fn = kernel
    result = _run_main(monkeypatch, capsys, _payload(correctness_cases=_cases("a")))

    assert result["success"] is False
    assert result["error_type"] == "cuda_oom"
    assert result["error"].startswith("OutOfMemoryError:")
    assert "synthetic" in result["error"]
    assert "OutOfMemoryError" in result["traceback"]


def test_oom_during_benchmark_classified_as_cuda_oom(env, monkeypatch, capsys) -> None:
    import torch

    env.measure_raises = torch.cuda.OutOfMemoryError("CUDA out of memory. in bench")
    result = _run_main(monkeypatch, capsys, _payload(correctness_cases=_cases("a")))

    assert result["success"] is False
    assert result["error_type"] == "cuda_oom"


def test_oom_classification_is_by_type_name(env, monkeypatch, capsys) -> None:
    """型名が OutOfMemoryError なら torch 由来でなくても cuda_oom になる（現行仕様の固定）。"""

    class OutOfMemoryError(RuntimeError):
        pass

    env.kernel_fn = _raiser(OutOfMemoryError("fake oom"))
    result = _run_main(monkeypatch, capsys, _payload(correctness_cases=_cases("a")))

    assert result["error_type"] == "cuda_oom"


@pytest.mark.parametrize(
    "exc",
    [RuntimeError, OSError, ValueError, ImportError, AttributeError],
    ids=lambda e: e.__name__,
)
def test_other_errors_keep_their_type_name(env, monkeypatch, capsys, exc) -> None:
    env.kernel_fn = _raiser(exc("synthetic"))
    result = _run_main(monkeypatch, capsys, _payload(correctness_cases=_cases("a")))

    assert result["success"] is False
    assert result["error_type"] == exc.__name__
    assert result["error"] == f"{exc.__name__}: synthetic"
    assert "Traceback" in result["traceback"]


def test_error_before_benchmark_has_no_partial_result(env, monkeypatch, capsys) -> None:
    """例外時は正確性の途中結果を混ぜず、success=False の 4 キーだけを返す。"""
    env.kernel_fn = _raiser(RuntimeError("x"))
    result = _run_main(monkeypatch, capsys, _payload(correctness_cases=_cases("a")))

    assert set(result) == {"success", "error", "error_type", "traceback"}


# --- 必須キー欠落・不正 payload の契約（現状の挙動を固定） -----------------------
# 外側の try は Exception 全般（JSON 不正を含む）を構造化 JSON にして返す（#496, #505）。


@pytest.mark.parametrize("raw", ["{not json", ""], ids=["malformed", "empty"])
def test_invalid_json_payload_returns_structured_error(monkeypatch, capsys, raw) -> None:
    """JSON 不正も構造化エラー JSON で返る（例外は伝播しない）。"""
    from forge.runtime._worker_entry import main

    monkeypatch.setattr(sys, "stdin", io.StringIO(raw))
    main()
    result = json.loads(capsys.readouterr().out.strip())
    assert result["success"] is False
    assert result["error_type"] == "JSONDecodeError"
    assert result["error"].startswith("JSONDecodeError:")
    assert "Traceback" in result["traceback"]


@pytest.mark.parametrize("missing", ["op_type", "kernel_code", "benchmark_input"])
def test_missing_required_key_returns_keyerror_json(env, monkeypatch, capsys, missing) -> None:
    """必須キー欠落は KeyError として構造化エラー JSON で親に返る（#496 以降）。"""
    payload = _payload()
    del payload[missing]
    result = _run_main(monkeypatch, capsys, payload)

    assert result["success"] is False
    assert result["error_type"] == "KeyError"
    assert missing in result["error"]


def test_missing_benchmark_input_is_tolerated_when_cases_given_and_no_benchmark(
    env, monkeypatch, capsys
) -> None:
    """correctness_cases があり task=correctness なら benchmark_input は参照されない。"""
    payload = _payload(correctness_cases=_cases("a"), task="correctness")
    del payload["benchmark_input"]
    result = _run_main(monkeypatch, capsys, payload)

    assert result["success"] is True
    assert result["correct"] is True


def test_missing_case_name_returns_keyerror_json_on_mismatch(env, monkeypatch, capsys) -> None:
    """correctness case に name が無く不一致になった場合も KeyError の構造化 JSON で返る。"""
    env.kernel_fn = lambda x, **kwargs: x + 1.0
    payload = _payload(correctness_cases=[{"input_specs": [_spec()]}])
    result = _run_main(monkeypatch, capsys, payload)

    assert result["success"] is False
    assert result["error_type"] == "KeyError"
    assert "name" in result["error"]


# --- _build_tensor ---------------------------------------------------------


class _CpuTorch:
    """torch の代役。device="cuda" 指定を記録して CPU に読み替える。dtype 等は本物へ委譲。"""

    def __init__(self) -> None:
        import torch

        self._torch = torch
        self.devices: list[str] = []

    def __getattr__(self, name: str):
        return getattr(self._torch, name)

    def Generator(self, device):  # noqa: N802
        self.devices.append(device)
        return self._torch.Generator(device="cpu")

    def randn(self, shape, dtype, device, generator):
        self.devices.append(device)
        return self._torch.randn(shape, dtype=dtype, device="cpu", generator=generator)

    def ones(self, shape, dtype, device):
        self.devices.append(device)
        return self._torch.ones(shape, dtype=dtype, device="cpu")

    def zeros(self, shape, dtype, device):
        self.devices.append(device)
        return self._torch.zeros(shape, dtype=dtype, device="cpu")


def _expected_randn(shape, dtype, seed):
    import torch

    gen = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(shape, dtype=dtype, device="cpu", generator=gen)


def test_build_tensor_randn_is_seeded_and_scaled() -> None:
    import torch

    from forge.runtime._worker_entry import _build_tensor

    fake = _CpuTorch()
    spec = {"shape": [3, 5], "dtype": "float32", "init": "randn", "seed": 7, "scale": 2.5}
    t = _build_tensor(spec, fake)

    assert t.shape == (3, 5)
    assert t.dtype == torch.float32
    assert torch.equal(t, _expected_randn((3, 5), torch.float32, 7) * 2.5)
    assert set(fake.devices) == {"cuda"}  # 本番は cuda 上に作る
    # 同 seed なら同一、別 seed なら別
    assert torch.equal(t, _build_tensor(spec, _CpuTorch()))
    assert not torch.equal(t, _build_tensor({**spec, "seed": 8}, _CpuTorch()))


def test_build_tensor_defaults_to_randn_seed0_scale1() -> None:
    import torch

    from forge.runtime._worker_entry import _build_tensor

    t = _build_tensor({"shape": [4], "dtype": "float16"}, _CpuTorch())

    assert t.dtype == torch.float16
    assert torch.equal(t, _expected_randn((4,), torch.float16, 0))


def test_build_tensor_ones_scaled() -> None:
    import torch

    from forge.runtime._worker_entry import _build_tensor

    fake = _CpuTorch()
    t = _build_tensor({"shape": [2, 3], "dtype": "float32", "init": "ones", "scale": 3.0}, fake)
    default = _build_tensor({"shape": [2], "dtype": "float32", "init": "ones"}, fake)

    assert torch.equal(t, torch.full((2, 3), 3.0))
    assert torch.equal(default, torch.ones(2))
    assert set(fake.devices) == {"cuda"}


def test_build_tensor_zeros_ignores_scale() -> None:
    import torch

    from forge.runtime._worker_entry import _build_tensor

    fake = _CpuTorch()
    t = _build_tensor({"shape": [2, 2], "dtype": "bfloat16", "init": "zeros", "scale": 9.0}, fake)

    assert t.dtype == torch.bfloat16
    assert torch.equal(t, torch.zeros(2, 2, dtype=torch.bfloat16))
    assert set(fake.devices) == {"cuda"}


def test_build_tensor_unknown_init_raises_value_error() -> None:
    from forge.runtime._worker_entry import _build_tensor

    with pytest.raises(ValueError, match="unknown init: bogus"):
        _build_tensor({"shape": [2], "dtype": "float32", "init": "bogus"}, _CpuTorch())


def test_unknown_init_surfaces_as_value_error_json(env, monkeypatch, capsys) -> None:
    """不明 init は main() 経由で error_type=ValueError の JSON になる（実 _build_tensor を使用）。"""
    import forge.runtime._worker_entry as entry_mod

    # env fixture が差し替えた fake を本物へ戻す。不明 init は cuda に触れる前に ValueError。
    monkeypatch.setattr(entry_mod, "_build_tensor", _REAL_BUILD_TENSOR)
    bad = {"shape": [2], "dtype": "float32", "init": "bogus"}
    result = _run_main(
        monkeypatch,
        capsys,
        _payload(correctness_cases=[{"name": "a", "input_specs": [bad]}]),
    )

    assert result["success"] is False
    assert result["error_type"] == "ValueError"
    assert "unknown init: bogus" in result["error"]
