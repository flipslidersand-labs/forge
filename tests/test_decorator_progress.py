"""@forge.optimize(progress=...) に渡したコールバックが受け取る値の契約。GPU 不要。

宣言型は Callable[[str], None]。注釈なし lambda / print 相当 / str 注釈付きは label (str) を、
ProgressEvent 注釈付きだけ ProgressEvent をそのまま受け取る。
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Optional
from unittest.mock import patch

import pytest
import torch

import forge
import forge.orchestrator as fo
from forge import decorator as dec
from forge.orchestrator import Orchestrator, ProgressEvent
from forge.orchestrator import ProgressEvent as PE

_EVENT = ProgressEvent(kind="info", label="hello")


class _FakeOrchestrator:
    """実 Orchestrator と同じ _wrap_progress を通して progress を受け取り、イベントを 1 つ流す。"""

    def __init__(self, *, repo: Any = None, min_speedup: float = 1.0, progress: Any = None) -> None:
        self._handler = Orchestrator._wrap_progress(progress)

    def __enter__(self) -> _FakeOrchestrator:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def optimize(self, spec: Any, budget: int, search: Any) -> Any:
        self._handler(_EVENT)
        return SimpleNamespace(best_params=None)


def _run_build(progress: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dec, "Orchestrator", _FakeOrchestrator)
    dec._build(
        lambda x: x,
        "rmsnorm",
        [torch.zeros(2)],
        {},
        dec._BuildConfig(
            budget=1,
            repo=None,
            search=None,
            min_speedup=1.0,
            python_executable=None,
            progress=dec._progress_to_event_handler(progress),
        ),
    )


def test_unannotated_lambda_receives_label_str(monkeypatch: pytest.MonkeyPatch) -> None:
    got: list[Any] = []
    _run_build(lambda m: got.append(m), monkeypatch)
    assert got == ["hello"]


def test_append_function_receives_label_str(monkeypatch: pytest.MonkeyPatch) -> None:
    """print 相当（注釈なしの組み込み/ビルトイン風 callable）。list.append は signature が取れない。"""
    got: list[Any] = []
    _run_build(got.append, monkeypatch)
    assert got == ["hello"]


def test_str_annotated_receives_label_str(monkeypatch: pytest.MonkeyPatch) -> None:
    got: list[Any] = []

    def on_str(msg: str) -> None:
        got.append(msg)

    _run_build(on_str, monkeypatch)
    assert got == ["hello"]


def test_event_annotated_still_receives_event(monkeypatch: pytest.MonkeyPatch) -> None:
    got: list[Any] = []

    def on_event(e: ProgressEvent) -> None:
        got.append(e)

    _run_build(on_event, monkeypatch)
    assert got == [_EVENT]


def test_none_progress_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    _run_build(None, monkeypatch)


def test_handler_adoption_message_is_str_for_unannotated() -> None:
    """adoption 通知（str 起点）も同じ口を通り、注釈なし callback には str が届く。"""
    got: list[Any] = []
    handler = dec._progress_to_event_handler(lambda m: got.append(m))
    handler(ProgressEvent(kind="info", label="adoption: x"))
    assert got == ["adoption: x"]


def test_handler_adoption_message_is_event_for_event_annotated() -> None:
    got: list[Any] = []

    def on_event(e: ProgressEvent) -> None:
        got.append(e)

    handler = dec._progress_to_event_handler(on_event)
    handler(ProgressEvent(kind="info", label="adoption: x"))
    assert [e.label for e in got] == ["adoption: x"]


# --- ProgressEvent 注釈の判定（修飾名・Optional・別名）。従来どおりイベントを受ける ---


_SINK: list[Any] = []


def _qualified(e: fo.ProgressEvent) -> None:
    _SINK.append(e)


def _union_none(e: ProgressEvent | None) -> None:
    _SINK.append(e)


def _optional(e: Optional[ProgressEvent]) -> None:  # noqa: UP045
    _SINK.append(e)


def _alias(e: PE) -> None:
    _SINK.append(e)


def _with_raw_annotation(annotation: Any) -> Any:
    """モジュールの名前空間では解決できない注釈（get_type_hints が失敗する）を持つ関数。"""
    got: list[Any] = []

    def fn(e):  # type: ignore[no-untyped-def]
        got.append(e)

    fn.__annotations__ = {"e": annotation}
    fn.got = got  # type: ignore[attr-defined]
    return fn


@pytest.mark.parametrize("fn", [_qualified, _union_none, _optional, _alias])
def test_event_annotation_variants_resolved_receive_event(
    fn: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _SINK.clear()
    _run_build(fn, monkeypatch)
    assert _SINK == [_EVENT]


@pytest.mark.parametrize(
    "annotation",
    [
        "forge.orchestrator.ProgressEvent",
        "ProgressEvent | None",
        "Optional[ProgressEvent]",
        "typing.Optional[forge.orchestrator.ProgressEvent]",
    ],
)
def test_event_annotation_unresolvable_strings_receive_event(
    annotation: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """get_type_hints で解決できない文字列注釈は末尾名 ProgressEvent で判定する。"""
    fn = _with_raw_annotation(annotation)
    _run_build(fn, monkeypatch)
    assert fn.got == [_EVENT]


def test_unresolvable_alias_falls_back_to_label(monkeypatch: pytest.MonkeyPatch) -> None:
    """解決できない別名など曖昧なときは str（label）に倒す（docstring に明記）。"""
    fn = _with_raw_annotation("SomeUnknownAlias")
    _run_build(fn, monkeypatch)
    assert fn.got == ["hello"]


# --- 公開経路（forge.optimize の progress=）で届く値 ---


class _FakeCudaTensor(torch.Tensor):
    """CPU 上で is_cuda=True を装い、wrapper を Orchestrator 呼び出しまで進める。"""

    is_cuda = property(lambda self: True)  # type: ignore[assignment]


def _call_public(progress: Any) -> None:
    x = torch.zeros(4).as_subclass(_FakeCudaTensor)
    with (
        patch("forge.decorator.identify", return_value="softmax"),
        patch("forge.decorator.Orchestrator", _FakeOrchestrator),
    ):

        @forge.optimize(budget=1, progress=progress)
        def fn(t):
            return t

        fn(x)


def test_public_optimize_unannotated_lambda_receives_label_str() -> None:
    got: list[Any] = []
    _call_public(lambda m: got.append(m))
    assert got == ["hello"]


def test_public_optimize_event_annotated_receives_event() -> None:
    got: list[Any] = []

    def on_event(e: fo.ProgressEvent) -> None:
        got.append(e)

    _call_public(on_event)
    assert got == [_EVENT]
