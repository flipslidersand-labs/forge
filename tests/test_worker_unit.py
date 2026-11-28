"""run_in_worker / run_extended_baseline_in_worker のエラーパスを CPU mock でテスト。

GPU 不要。subprocess.run を patch して全ケースを CI で常時実行する。
"""

from __future__ import annotations

import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from forge.runtime.worker import (
    WorkerResult,
    _head_tail,
    _tail,
    run_extended_baseline_in_worker,
    run_in_worker,
)

_DUMMY_KERNEL = "def kernel(): pass"
_DUMMY_OP = "rmsnorm"
_DUMMY_INPUT = [{"shape": [512, 4096], "dtype": "float16", "init": "randn", "seed": 0}]
_DUMMY_CONSTANTS: dict = {"eps": 1e-6}


def _proc(returncode: int = 0, stdout: str = "", stderr: str = "") -> MagicMock:
    p = MagicMock()
    p.returncode = returncode
    p.stdout = stdout
    p.stderr = stderr
    return p


def _ok_output() -> str:
    return json.dumps(
        {
            "success": True,
            "correct": True,
            "max_abs_diff": 1e-4,
            "failures": [],
            "candidate": {"median_us": 42.0, "p20_us": 40.0, "p80_us": 44.0, "p95_us": 46.0},
            "baseline": {"median_us": 60.0, "p20_us": 58.0, "p80_us": 62.0, "p95_us": 65.0},
            "baseline_name": "torch.rmsnorm",
        }
    )


class TestRunInWorkerErrorPaths:
    """subprocess エラーパス 3種を CPU mock で検証。"""

    def test_timeout_returns_failure(self) -> None:
        with patch(
            "subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="python", timeout=5.0)
        ):
            r = run_in_worker(
                _DUMMY_KERNEL, _DUMMY_OP, _DUMMY_INPUT, _DUMMY_CONSTANTS, timeout_s=5.0
            )

        assert r.success is False
        assert r.error is not None
        assert "timeout" in r.error.lower()
        assert "5.0" in r.error

    def test_nonzero_returncode_returns_failure(self) -> None:
        proc = _proc(returncode=1, stderr="CUDA error: device-side assert triggered\nAborted\n")
        with patch("subprocess.run", return_value=proc):
            r = run_in_worker(_DUMMY_KERNEL, _DUMMY_OP, _DUMMY_INPUT, _DUMMY_CONSTANTS)

        assert r.success is False
        assert r.error is not None
        assert "worker crashed" in r.error
        assert "Aborted" in r.error  # stderr の最終行

    def test_nonzero_returncode_empty_stderr(self) -> None:
        proc = _proc(returncode=139, stderr="")
        with patch("subprocess.run", return_value=proc):
            r = run_in_worker(_DUMMY_KERNEL, _DUMMY_OP, _DUMMY_INPUT, _DUMMY_CONSTANTS)

        assert r.success is False
        assert r.error is not None
        assert "worker crashed" in r.error
        assert "139" in r.error  # exit コードを含む

    def test_bad_json_stdout_returns_failure(self) -> None:
        proc = _proc(returncode=0, stdout="this is not json at all")
        with patch("subprocess.run", return_value=proc):
            r = run_in_worker(_DUMMY_KERNEL, _DUMMY_OP, _DUMMY_INPUT, _DUMMY_CONSTANTS)

        assert r.success is False
        assert r.error is not None
        assert "bad worker output" in r.error

    @pytest.mark.parametrize("last_line", ["123", "null", "[1]"])
    def test_non_dict_json_last_line_returns_failure(self, last_line: str) -> None:
        proc = _proc(returncode=0, stdout=f"noise\n{last_line}\n")
        with patch("subprocess.run", return_value=proc):
            r = run_in_worker(_DUMMY_KERNEL, _DUMMY_OP, _DUMMY_INPUT, _DUMMY_CONSTANTS)

        assert r.success is False
        assert r.error is not None
        assert "bad worker output" in r.error
        assert last_line in r.error

    def test_empty_stdout_returns_failure(self) -> None:
        proc = _proc(returncode=0, stdout="")
        with patch("subprocess.run", return_value=proc):
            r = run_in_worker(_DUMMY_KERNEL, _DUMMY_OP, _DUMMY_INPUT, _DUMMY_CONSTANTS)

        assert r.success is False
        assert r.error is not None
        assert "bad worker output" in r.error

    def test_empty_stdout_with_stderr_includes_stderr_detail(self) -> None:
        proc = _proc(returncode=0, stdout="", stderr="OOM: CUDA out of memory")
        with patch("subprocess.run", return_value=proc):
            r = run_in_worker(_DUMMY_KERNEL, _DUMMY_OP, _DUMMY_INPUT, _DUMMY_CONSTANTS)

        assert r.success is False
        assert r.error is not None
        assert "OOM" in r.error

    def test_nonzero_returncode_full_stderr_included(self) -> None:
        long_stderr = "line1\n" * 200 + "FATAL: last line"
        proc = _proc(returncode=1, stderr=long_stderr)
        with patch("subprocess.run", return_value=proc):
            r = run_in_worker(_DUMMY_KERNEL, _DUMMY_OP, _DUMMY_INPUT, _DUMMY_CONSTANTS)

        assert r.success is False
        assert "FATAL" in r.error
        assert "line1" in r.error

    def test_nonzero_returncode_long_stderr_keeps_tail(self) -> None:
        long_stderr = "WARN: noisy startup\n" * 500 + "RuntimeError: the real cause"
        assert len(long_stderr) > 4096
        proc = _proc(returncode=1, stderr=long_stderr)
        with patch("subprocess.run", return_value=proc):
            r = run_in_worker(_DUMMY_KERNEL, _DUMMY_OP, _DUMMY_INPUT, _DUMMY_CONSTANTS)

        assert r.success is False
        assert r.error is not None
        assert r.error.endswith("RuntimeError: the real cause")
        assert "(truncated)" in r.error
        assert len(r.error) < 4096 + 100

    def test_empty_stdout_long_stderr_keeps_tail(self) -> None:
        long_stderr = "WARN: noisy startup\n" * 500 + "CompilationError: the real cause"
        proc = _proc(returncode=0, stdout="", stderr=long_stderr)
        with patch("subprocess.run", return_value=proc):
            r = run_in_worker(_DUMMY_KERNEL, _DUMMY_OP, _DUMMY_INPUT, _DUMMY_CONSTANTS)

        assert r.success is False
        assert r.error is not None
        assert r.error.endswith("CompilationError: the real cause")
        assert "(truncated)" in r.error

    def test_short_stderr_has_no_truncation_marker(self) -> None:
        proc = _proc(returncode=1, stderr="short failure")
        with patch("subprocess.run", return_value=proc):
            r = run_in_worker(_DUMMY_KERNEL, _DUMMY_OP, _DUMMY_INPUT, _DUMMY_CONSTANTS)

        assert r.error == "worker crashed: short failure"

    def test_success_path_parsed_correctly(self) -> None:
        proc = _proc(returncode=0, stdout=_ok_output())
        with patch("subprocess.run", return_value=proc):
            r = run_in_worker(_DUMMY_KERNEL, _DUMMY_OP, _DUMMY_INPUT, _DUMMY_CONSTANTS)

        assert r.success is True
        assert r.correct is True
        assert r.candidate is not None
        assert r.candidate.median_us == pytest.approx(42.0)
        assert r.baseline is not None
        assert r.baseline_name == "torch.rmsnorm"

    def test_worker_reported_failure_propagated(self) -> None:
        stdout = json.dumps(
            {"success": False, "error": "ValueError: unsupported op", "traceback": "..."}
        )
        proc = _proc(returncode=0, stdout=stdout)
        with patch("subprocess.run", return_value=proc):
            r = run_in_worker(_DUMMY_KERNEL, _DUMMY_OP, _DUMMY_INPUT, _DUMMY_CONSTANTS)

        assert r.success is False
        assert r.error == "ValueError: unsupported op"

    def test_only_last_stdout_line_is_parsed(self) -> None:
        """ワーカーが debug print を出した後に JSON を吐くケース。"""
        stdout = "debug: starting\nwarning: slow path\n" + _ok_output()
        proc = _proc(returncode=0, stdout=stdout)
        with patch("subprocess.run", return_value=proc):
            r = run_in_worker(_DUMMY_KERNEL, _DUMMY_OP, _DUMMY_INPUT, _DUMMY_CONSTANTS)

        assert r.success is True

    def test_bad_json_long_noise_keeps_failing_last_line(self) -> None:
        """パース対象は最終行。先行ノイズが 4096 文字を超えても失敗した最終行を残す。

        先頭優先（修正前の stdout[:4096]）だと最終行が切り落とされていた。
        """
        stdout = "NOISE-BEFORE " * 600 + "\n" + _ok_output() + "\nROOT-CAUSE: late message"
        assert len(stdout) > 4096
        proc = _proc(returncode=0, stdout=stdout)
        with patch("subprocess.run", return_value=proc):
            r = run_in_worker(_DUMMY_KERNEL, _DUMMY_OP, _DUMMY_INPUT, _DUMMY_CONSTANTS)

        assert r.success is False
        assert r.error is not None
        assert "ROOT-CAUSE: late message" in r.error
        assert "chars)..." in r.error

    def test_bad_json_long_truncated_line_keeps_leading_fields(self) -> None:
        """最終行自体が長く途中で切れている場合は、先頭側の success/error も残す。

        末尾優先だけだと '"success": false, "error": ...' が落ちる。
        """
        line = json.dumps({"success": False, "error": "E" * 6000, "traceback": "T" * 6000})
        stdout = line[:5000]  # JSON が途中で切れた状態（rc=0 のまま）
        assert len(stdout) > 4096
        proc = _proc(returncode=0, stdout=stdout)
        with patch("subprocess.run", return_value=proc):
            r = run_in_worker(_DUMMY_KERNEL, _DUMMY_OP, _DUMMY_INPUT, _DUMMY_CONSTANTS)

        assert r.success is False
        assert r.error is not None
        assert '"success": false' in r.error
        assert '"error": "EEE' in r.error
        assert r.error.endswith("EEE'")  # 切れた末尾も残る

    def test_bad_json_short_stdout_has_no_truncation_marker(self) -> None:
        proc = _proc(returncode=0, stdout="this is not json at all")
        with patch("subprocess.run", return_value=proc):
            r = run_in_worker(_DUMMY_KERNEL, _DUMMY_OP, _DUMMY_INPUT, _DUMMY_CONSTANTS)

        assert r.error is not None
        assert "this is not json at all" in r.error
        assert "(truncated)" not in r.error


class TestHeadTail:
    """_head_tail の境界値（上限は文字数。両端 limit//2 ずつ残し、間に印を入れる）。"""

    def test_empty(self) -> None:
        assert _head_tail("") == ""

    def test_exactly_limit_is_unchanged(self) -> None:
        text = "a" * 4096
        assert _head_tail(text) == text

    def test_limit_plus_one_omits_one_char_in_the_middle(self) -> None:
        text = "H" * 2048 + "m" + "T" * 2048
        assert len(text) == 4097
        assert _head_tail(text) == "H" * 2048 + "\n...(truncated 1 chars)...\n" + "T" * 2048

    def test_keeps_both_ends(self) -> None:
        text = "HEAD" + "x" * 10000 + "TAIL"
        out = _head_tail(text)
        assert out.startswith("HEAD")
        assert out.endswith("TAIL")
        assert "(truncated 5912 chars)" in out

    def test_multibyte_counts_characters_not_bytes(self) -> None:
        text = "あ" * 4096  # 12288 バイトだが 4096 文字なので切り詰めない
        assert _head_tail(text) == text

    def test_multibyte_truncation_keeps_whole_characters(self) -> None:
        text = "あ" * 2048 + "い" + "う" * 2048
        out = _head_tail(text)
        assert out == "あ" * 2048 + "\n...(truncated 1 chars)...\n" + "う" * 2048
        out.encode("utf-8")

    def test_custom_limit(self) -> None:
        assert _head_tail("abcdef", limit=6) == "abcdef"
        assert _head_tail("abcdefg", limit=4) == "ab\n...(truncated 3 chars)...\nfg"


class TestTail:
    """_tail の境界値。上限は文字数（バイト数ではない）で、印の分だけ出力が上限を超える。"""

    _MARK = "...(truncated)\n"

    def test_empty(self) -> None:
        assert _tail("") == ""

    def test_exactly_limit_is_unchanged(self) -> None:
        text = "a" * 4096
        assert _tail(text) == text

    def test_limit_plus_one_is_truncated_with_marker(self) -> None:
        text = "b" + "a" * 4096
        out = _tail(text)
        assert out == self._MARK + "a" * 4096

    def test_truncated_length_exceeds_limit_by_marker_length(self) -> None:
        out = _tail("x" * 10000)
        assert len(out) == 4096 + len(self._MARK)

    def test_keeps_last_chars(self) -> None:
        text = "".join(chr(0x30 + i % 10) for i in range(5000))
        assert _tail(text).endswith(text[-4096:])

    def test_multibyte_counts_characters_not_bytes(self) -> None:
        text = "あ" * 4096  # 12288 バイトだが 4096 文字なので切り詰めない
        assert _tail(text) == text

    def test_multibyte_truncation_keeps_whole_characters(self) -> None:
        text = "い" + "あ" * 4096
        out = _tail(text)
        assert out == self._MARK + "あ" * 4096
        out.encode("utf-8")  # サロゲート等で壊れていない

    def test_custom_limit(self) -> None:
        assert _tail("abcdef", limit=6) == "abcdef"
        assert _tail("abcdef", limit=3) == self._MARK + "def"


class TestRunExtendedBaselineInWorkerErrorPaths:
    """run_extended_baseline_in_worker のエラーパスも CPU mock でカバー。"""

    def test_timeout_returns_empty_list(self) -> None:
        with patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="python", timeout=180.0),
        ):
            result = run_extended_baseline_in_worker(_DUMMY_OP, _DUMMY_INPUT, _DUMMY_CONSTANTS)

        assert result == []

    def test_timeout_logs_warning(self) -> None:
        with patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="python", timeout=180.0),
        ):
            with patch("forge.runtime.worker._log") as mock_log:
                run_extended_baseline_in_worker(
                    _DUMMY_OP, _DUMMY_INPUT, _DUMMY_CONSTANTS, timeout_s=180.0
                )
        mock_log.warning.assert_called_once()
        call_args = mock_log.warning.call_args[0]
        assert "timeout" in call_args[0]
        assert _DUMMY_OP in call_args[1]

    def test_nonzero_returncode_returns_empty_list(self) -> None:
        proc = _proc(returncode=1, stderr="crash")
        with patch("subprocess.run", return_value=proc):
            result = run_extended_baseline_in_worker(_DUMMY_OP, _DUMMY_INPUT, _DUMMY_CONSTANTS)

        assert result == []

    def test_nonzero_returncode_logs_warning(self) -> None:
        proc = _proc(returncode=1, stderr="CUDA error: illegal memory access")
        with patch("subprocess.run", return_value=proc):
            with patch("forge.runtime.worker._log") as mock_log:
                result = run_extended_baseline_in_worker(_DUMMY_OP, _DUMMY_INPUT, _DUMMY_CONSTANTS)
        assert result == []
        mock_log.warning.assert_called_once()
        call_args = mock_log.warning.call_args[0]
        assert "crashed" in call_args[0]
        assert _DUMMY_OP in call_args[1]

    def test_bad_json_returns_empty_list(self) -> None:
        proc = _proc(returncode=0, stdout="not json")
        with patch("subprocess.run", return_value=proc):
            result = run_extended_baseline_in_worker(_DUMMY_OP, _DUMMY_INPUT, _DUMMY_CONSTANTS)

        assert result == []

    def test_bad_json_logs_warning(self) -> None:
        proc = _proc(returncode=0, stdout="not json")
        with patch("subprocess.run", return_value=proc):
            with patch("forge.runtime.worker._log") as mock_log:
                result = run_extended_baseline_in_worker(_DUMMY_OP, _DUMMY_INPUT, _DUMMY_CONSTANTS)
        assert result == []
        mock_log.warning.assert_called_once()
        call_args = mock_log.warning.call_args[0]
        assert "bad worker output" in call_args[0]

    @pytest.mark.parametrize("last_line", ["123", "null", "[1]"])
    def test_non_dict_json_last_line_returns_empty_list(self, last_line: str) -> None:
        proc = _proc(returncode=0, stdout=f"noise\n{last_line}\n")
        with patch("subprocess.run", return_value=proc):
            with patch("forge.runtime.worker._log") as mock_log:
                result = run_extended_baseline_in_worker(_DUMMY_OP, _DUMMY_INPUT, _DUMMY_CONSTANTS)
        assert result == []
        mock_log.warning.assert_called_once()
        call_args = mock_log.warning.call_args[0]
        assert "bad worker output" in call_args[0]
        assert last_line in call_args[-1]

    def test_bad_json_long_stdout_logs_both_ends(self) -> None:
        stdout = "FIRST-LINE\n" + "NOISE-BEFORE " * 600 + "\nROOT-CAUSE: late message"
        proc = _proc(returncode=0, stdout=stdout)
        with patch("subprocess.run", return_value=proc):
            with patch("forge.runtime.worker._log") as mock_log:
                result = run_extended_baseline_in_worker(_DUMMY_OP, _DUMMY_INPUT, _DUMMY_CONSTANTS)
        assert result == []
        mock_log.warning.assert_called_once()
        logged_stdout = mock_log.warning.call_args[0][-1]
        assert "ROOT-CAUSE: late message" in logged_stdout
        assert "FIRST-LINE" in logged_stdout

    def test_success_false_in_payload_returns_empty_list(self) -> None:
        proc = _proc(returncode=0, stdout=json.dumps({"success": False}))
        with patch("subprocess.run", return_value=proc):
            result = run_extended_baseline_in_worker(_DUMMY_OP, _DUMMY_INPUT, _DUMMY_CONSTANTS)

        assert result == []

    def test_success_false_logs_warning(self) -> None:
        proc = _proc(returncode=0, stdout=json.dumps({"success": False, "error": "OOM"}))
        with patch("subprocess.run", return_value=proc):
            with patch("forge.runtime.worker._log") as mock_log:
                result = run_extended_baseline_in_worker(_DUMMY_OP, _DUMMY_INPUT, _DUMMY_CONSTANTS)
        assert result == []
        mock_log.warning.assert_called_once()
        call_args = mock_log.warning.call_args[0]
        assert "failure" in call_args[0]
        assert _DUMMY_OP in call_args[1]

    def test_success_path_returns_baselines(self) -> None:
        bench = {"median_us": 80.0, "p20_us": 78.0, "p80_us": 82.0, "p95_us": 85.0}
        payload = json.dumps(
            {
                "success": True,
                "task": "extended_baseline",
                "baselines": [
                    {"name": "torch.compile(reference)", "benchmark": bench, "compile_time_s": 3.2}
                ],
            }
        )
        proc = _proc(returncode=0, stdout=payload)
        with patch("subprocess.run", return_value=proc):
            result = run_extended_baseline_in_worker(_DUMMY_OP, _DUMMY_INPUT, _DUMMY_CONSTANTS)

        assert len(result) == 1
        assert result[0].name == "torch.compile(reference)"
        assert result[0].benchmark.median_us == pytest.approx(80.0)
        assert result[0].compile_time_s == pytest.approx(3.2)
        assert result[0].failed is False


class TestWorkerResultFromDict:
    """WorkerResult.from_dict の境界ケース。"""

    def test_success_false_ignores_other_fields(self) -> None:
        r = WorkerResult.from_dict({"success": False, "error": "boom"})
        assert r.success is False
        assert r.error == "boom"
        assert r.correct is False

    def test_success_without_candidate_baseline(self) -> None:
        r = WorkerResult.from_dict(
            {"success": True, "correct": True, "max_abs_diff": 0.0, "failures": []}
        )
        assert r.success is True
        assert r.candidate is None
        assert r.baseline is None

    def test_failures_list_preserved(self) -> None:
        r = WorkerResult.from_dict(
            {
                "success": True,
                "correct": False,
                "max_abs_diff": 0.05,
                "failures": [{"case": "primary", "max_diff": 0.05}],
            }
        )
        assert r.correct is False
        assert len(r.failures) == 1
        assert r.failures[0]["case"] == "primary"


class TestWorkerResultErrorFields:
    """error_type / traceback が親まで届く (cuda_oom の判別を含む)。"""

    def test_defaults_keep_existing_constructor_working(self) -> None:
        r = WorkerResult(success=False, error="x")
        assert r.error_type is None
        assert r.traceback is None

    def test_from_dict_keeps_error_type_and_traceback(self) -> None:
        r = WorkerResult.from_dict(
            {"success": False, "error": "E", "error_type": "TypeError", "traceback": "tb"}
        )
        assert r.error_type == "TypeError"
        assert r.traceback == "tb"

    def test_from_dict_without_error_fields(self) -> None:
        r = WorkerResult.from_dict({"success": False, "error": "E"})
        assert r.error_type is None
        assert r.traceback is None

    def test_run_in_worker_propagates_cuda_oom(self) -> None:
        out = json.dumps(
            {
                "success": False,
                "error": "OutOfMemoryError: CUDA out of memory",
                "error_type": "cuda_oom",
                "traceback": "Traceback ...",
            }
        )
        with patch("subprocess.run", return_value=_proc(stdout=out)):
            r = run_in_worker(_DUMMY_KERNEL, _DUMMY_OP, _DUMMY_INPUT, _DUMMY_CONSTANTS)
        assert r.success is False
        assert r.error_type == "cuda_oom"
        assert r.traceback == "Traceback ..."


class TestRealWorkerStructuredError:
    """実 subprocess（GPU 不要: カーネル import 時点で失敗させる）。"""

    @pytest.mark.parametrize("exc_name", ["TypeError", "KeyError"])
    def test_kernel_raising_non_listed_exception_is_structured(self, exc_name: str) -> None:
        code = f"raise {exc_name}('fake kernel failure')\n"
        r = run_in_worker(code, _DUMMY_OP, _DUMMY_INPUT, _DUMMY_CONSTANTS, timeout_s=120.0)
        assert r.success is False
        assert r.error is not None
        assert "worker crashed" not in r.error
        assert "fake kernel failure" in r.error
        assert r.error_type == exc_name
        assert r.traceback is not None and "Traceback" in r.traceback
