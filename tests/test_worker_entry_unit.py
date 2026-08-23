"""_worker_entry.main() の例外処理を直接テスト (#220)。GPU 不要。

KeyboardInterrupt / SystemExit が except ブロックで握り潰されないこと、
および RuntimeError が error_type フィールド付きで JSON 出力されることを検証する。
"""

from __future__ import annotations

import io
import json
import sys

import pytest


def _make_payload(**overrides) -> str:
    base = {
        "op_type": "rmsnorm",
        "kernel_code": "pass",
        "benchmark_input": [
            {"shape": [4], "dtype": "float16", "init": "randn", "seed": 0}
        ],
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


def test_no_bare_exception_in_source() -> None:
    """ソース内に bare 'except Exception' が無いことを静的確認。"""
    import inspect

    import forge.runtime._worker_entry as mod

    source = inspect.getsource(mod)
    assert "except Exception" not in source, "bare except Exception が残っています"
    assert "noqa: BLE001" not in source, "noqa: BLE001 が残っています"
