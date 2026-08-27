"""DiscordNotifier のユニットテスト（CPU / ネットワーク不要）。

webhook 未設定時はエラーにならず False を返すこと、設定時は _send_webhook が
バックグラウンドスレッドを起動して True を返すこと、通信失敗はスレッド内で
logger.warning に格下げされ呼び出し元をブロックしないことを検証する。
"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

from forge.notifiers.discord import DiscordNotifier

_VALID_COMPLETION = "https://discord.com/api/webhooks/000/completion"
_VALID_ERRORS = "https://discord.com/api/webhooks/000/errors"


def _configured() -> DiscordNotifier:
    with patch.dict(
        "os.environ",
        {
            "DISCORD_WEBHOOK_COMPLETION": _VALID_COMPLETION,
            "DISCORD_WEBHOOK_ERRORS": _VALID_ERRORS,
        },
    ):
        return DiscordNotifier()


def _wait_for_notifier_thread(timeout: float = 2.0) -> None:
    """バックグラウンドの discord-notifier スレッドが終了するまで待機する。"""
    for t in threading.enumerate():
        if t.name == "discord-notifier":
            t.join(timeout=timeout)


class TestUnconfigured:
    """webhook 未設定なら全メソッドが False（例外を出さない）。"""

    def test_no_env_reads_none(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            n = DiscordNotifier()
        assert n.completion_webhook is None
        assert n.error_webhook is None

    def test_send_cache_hit_returns_false(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            n = DiscordNotifier()
        assert n.send_cache_hit("rmsnorm") is False

    def test_send_optimization_complete_returns_false(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            n = DiscordNotifier()
        assert n.send_optimization_complete("rmsnorm", 1.0, 10, 2.0) is False

    def test_send_optimization_error_returns_false(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            n = DiscordNotifier()
        assert n.send_optimization_error("rmsnorm", "boom", "ERR") is False


class TestConfiguredSuccess:
    """webhook 設定済みなら True（スレッド起動成功）。urlopen をモックしてネットワークを遮断。"""

    def _mock_204(self):
        resp = MagicMock()
        resp.status = 204
        ctx = MagicMock()
        ctx.__enter__.return_value = resp
        return ctx

    def test_cache_hit_success(self) -> None:
        n = _configured()
        with patch("forge.notifiers.discord.urlopen", return_value=self._mock_204()):
            result = n.send_cache_hit("rmsnorm")
        _wait_for_notifier_thread()
        assert result is True

    def test_optimization_complete_success(self) -> None:
        n = _configured()
        with patch("forge.notifiers.discord.urlopen", return_value=self._mock_204()):
            result = n.send_optimization_complete("softmax", 0.5, 20, 3.0)
        _wait_for_notifier_thread()
        assert result is True

    def test_optimization_error_success(self) -> None:
        n = _configured()
        with patch("forge.notifiers.discord.urlopen", return_value=self._mock_204()):
            result = n.send_optimization_error("gelu", "boom", "OPT_FAIL")
        _wait_for_notifier_thread()
        assert result is True


class TestConfiguredFailure:
    """通信失敗・非 204 応答でも呼び出し元をブロックせず True を返す（スレッド起動成功）。

    HTTP エラーはバックグラウンドスレッドで logger.warning に格下げされる。
    """

    def test_non_204_does_not_block(self) -> None:
        """非 204 レスポンスでもスレッドは起動し True が返る。"""
        n = _configured()
        resp = MagicMock()
        resp.status = 500
        ctx = MagicMock()
        ctx.__enter__.return_value = resp
        with patch("forge.notifiers.discord.urlopen", return_value=ctx):
            result = n.send_cache_hit("rmsnorm")
        _wait_for_notifier_thread()
        assert result is True

    def test_urlopen_raises_does_not_block(self) -> None:
        """urlopen が例外を投げてもスレッドは起動し True が返る。"""
        n = _configured()
        with patch("forge.notifiers.discord.urlopen", side_effect=OSError("network down")):
            result = n.send_optimization_complete("rmsnorm", 1.0, 5, 1.0)
        _wait_for_notifier_thread()
        assert result is True

    def test_non_204_logs_warning(self, caplog) -> None:
        """非 204 レスポンスは logger.warning として記録される。"""
        import logging

        n = _configured()
        resp = MagicMock()
        resp.status = 500
        ctx = MagicMock()
        ctx.__enter__.return_value = resp
        with caplog.at_level(logging.WARNING, logger="forge.notifiers.discord"):
            with patch("forge.notifiers.discord.urlopen", return_value=ctx):
                n.send_cache_hit("rmsnorm")
            _wait_for_notifier_thread()
        assert any("500" in r.message for r in caplog.records)

    def test_urlopen_raises_logs_warning(self, caplog) -> None:
        """urlopen 例外は logger.warning として記録される。"""
        import logging

        n = _configured()
        with caplog.at_level(logging.WARNING, logger="forge.notifiers.discord"):
            with patch("forge.notifiers.discord.urlopen", side_effect=OSError("network down")):
                n.send_optimization_complete("rmsnorm", 1.0, 5, 1.0)
            _wait_for_notifier_thread()
        assert any("network down" in r.message for r in caplog.records)


class TestUnexpectedExceptions:
    """URLError 以外の例外でも呼び出し元をブロックしないことを検証。

    discord.py の構造:
    - 外層 except Exception: embed 構築中の予期しない例外 → False を返す
    - 内層（スレッド内）: urlopen が例外を投げても warning ログのみ・スレッドで完結
    """

    def test_send_webhook_runtime_error_does_not_block(self) -> None:
        """_send_webhook スレッド内の RuntimeError → 呼び出し元は True を受け取る。"""
        n = _configured()
        with patch(
            "forge.notifiers.discord.urlopen", side_effect=RuntimeError("ssl context broken")
        ):
            result = n.send_optimization_complete("rmsnorm", 1.0, 5, 2.0)
        _wait_for_notifier_thread()
        assert result is True

    def test_send_webhook_attribute_error_does_not_block(self) -> None:
        """urlopen が AttributeError を投げても呼び出し元はブロックされない。"""
        n = _configured()
        with patch("forge.notifiers.discord.urlopen", side_effect=AttributeError("mock attr")):
            result = n.send_cache_hit("rmsnorm")
        _wait_for_notifier_thread()
        assert result is True

    def test_send_webhook_value_error_does_not_block(self) -> None:
        """urlopen が ValueError を投げても呼び出し元はブロックされない。"""
        n = _configured()
        with patch("forge.notifiers.discord.urlopen", side_effect=ValueError("unexpected value")):
            result = n.send_optimization_error("rmsnorm", "msg", "ERR")
        _wait_for_notifier_thread()
        assert result is True

    def test_embed_construction_error_returns_false(self) -> None:
        """embed 構築中（datetime.now）の例外を外層 except Exception が握り False を返す。"""
        n = _configured()
        with patch("forge.notifiers.discord.datetime") as mock_dt:
            mock_dt.now.side_effect = RuntimeError("clock unavailable")
            assert n.send_optimization_complete("rmsnorm", 1.0, 5, 2.0) is False

    def test_send_optimization_error_embed_construction_error_returns_false(self) -> None:
        n = _configured()
        with patch("forge.notifiers.discord.datetime") as mock_dt:
            mock_dt.now.side_effect = RuntimeError("clock unavailable")
            assert n.send_optimization_error("rmsnorm", "fail", "TYPE") is False

    def test_send_cache_hit_embed_construction_error_returns_false(self) -> None:
        n = _configured()
        with patch("forge.notifiers.discord.datetime") as mock_dt:
            mock_dt.now.side_effect = RuntimeError("clock unavailable")
            assert n.send_cache_hit("rmsnorm") is False


class TestOptimizationCompleteNewFields:
    """send_optimization_complete の新パラメータ（speedup/best_round/failed_rate）を検証。"""

    def _capture_payload(self, n, **kwargs):
        """urlopen を mock して送信ペイロードを捕捉する。スレッド完了後に返す。"""
        captured = {}

        def _fake_urlopen(req, timeout=None):
            import json

            captured["payload"] = json.loads(req.data.decode())
            mock_resp = type(
                "R", (), {"status": 204, "__enter__": lambda s: s, "__exit__": lambda s, *a: None}
            )()
            return mock_resp

        with patch("forge.notifiers.discord.urlopen", side_effect=_fake_urlopen):
            n.send_optimization_complete("rmsnorm", 1.0, 10, 2.0, **kwargs)
        _wait_for_notifier_thread()
        return captured.get("payload", {})

    def test_speedup_field_in_embed(self):
        n = _configured()
        payload = self._capture_payload(n, speedup=4.25)
        fields = payload["embeds"][0]["fields"]
        names = {f["name"]: f["value"] for f in fields}
        assert "Speedup" in names
        assert names["Speedup"] == "4.25×"

    def test_best_round_field_in_embed(self):
        n = _configured()
        payload = self._capture_payload(n, best_round=2)
        fields = payload["embeds"][0]["fields"]
        names = {f["name"]: f["value"] for f in fields}
        assert "Best Round" in names
        assert names["Best Round"] == "2"

    def test_failed_rate_field_in_embed(self):
        n = _configured()
        payload = self._capture_payload(n, failed_rate=0.2)
        fields = payload["embeds"][0]["fields"]
        names = {f["name"]: f["value"] for f in fields}
        assert "Fail Rate" in names
        assert names["Fail Rate"] == "20%"

    def test_none_fields_omitted(self):
        n = _configured()
        payload = self._capture_payload(n)
        fields = payload["embeds"][0]["fields"]
        field_names = {f["name"] for f in fields}
        assert "Speedup" not in field_names
        assert "Best Round" not in field_names
        assert "Fail Rate" not in field_names


<<<<<<< HEAD
class TestWebhookUrlValidation:
    """_validate_webhook_url が SSRF を防ぐことを検証（#262）。"""

    def test_valid_discord_com_url_passes(self) -> None:
        """https://discord.com/... は ValueError を出さない。"""
        DiscordNotifier._validate_webhook_url("https://discord.com/api/webhooks/123/abc")

    def test_valid_subdomain_discord_com_passes(self) -> None:
        """https://canary.discord.com/... も許可される。"""
        DiscordNotifier._validate_webhook_url("https://canary.discord.com/api/webhooks/123/abc")

    def test_http_scheme_raises(self) -> None:
        """http:// は拒否（暗号化なし）。"""
        import pytest

        with pytest.raises(ValueError, match="Invalid Discord webhook URL"):
            DiscordNotifier._validate_webhook_url("http://discord.com/api/webhooks/123/abc")

    def test_metadata_endpoint_raises(self) -> None:
        """SSRF: AWS EC2 メタデータエンドポイントへの送信を拒否。"""
        import pytest

        with pytest.raises(ValueError, match="Invalid Discord webhook URL"):
            DiscordNotifier._validate_webhook_url("https://169.254.169.254/latest/meta-data/")

    def test_internal_host_raises(self) -> None:
        """SSRF: 内部ホスト名への送信を拒否。"""
        import pytest

        with pytest.raises(ValueError, match="Invalid Discord webhook URL"):
            DiscordNotifier._validate_webhook_url("https://internal-service/webhook")

    def test_non_discord_com_domain_raises(self) -> None:
        """discord.com を含まない外部ドメインを拒否。"""
        import pytest

        with pytest.raises(ValueError, match="Invalid Discord webhook URL"):
            DiscordNotifier._validate_webhook_url("https://evil.com/discord.com/webhook")

    def test_send_webhook_invalid_url_returns_false(self) -> None:
        """_send_webhook に不正 URL を渡すと False を返し例外を伝播しない。"""
        n = _configured()
        result = n._send_webhook("https://169.254.169.254/latest/meta-data/", {"embeds": []})
        assert result is False
=======
class TestAsyncBehavior:
    """_send_webhook のバックグラウンド実行を検証するテスト群。"""

    def test_send_webhook_does_not_block_on_slow_urlopen(self) -> None:
        """urlopen が遅くても send_* は即座に制御を返す。"""
        import time

        call_times: list[float] = []

        def _slow_urlopen(req, timeout=None):
            time.sleep(0.05)  # 50ms delay
            raise OSError("simulated slow failure")

        n = _configured()
        t0 = time.monotonic()
        with patch("forge.notifiers.discord.urlopen", side_effect=_slow_urlopen):
            n.send_cache_hit("rmsnorm")
        elapsed = time.monotonic() - t0
        _wait_for_notifier_thread()
        # send_cache_hit は HTTP 待機前に即座に返るはず（< 10ms が理想だが余裕を持って < 40ms）
        assert elapsed < 0.040, f"send_cache_hit blocked for {elapsed:.3f}s"
        call_times.append(elapsed)

    def test_send_webhook_runs_in_daemon_thread(self) -> None:
        """_send_webhook が daemon スレッドを起動することを確認する。"""
        launched: list[threading.Thread] = []
        original_start = threading.Thread.start

        def _capture_start(self_thread):
            launched.append(self_thread)
            original_start(self_thread)

        n = _configured()
        resp = MagicMock()
        resp.status = 204
        ctx = MagicMock()
        ctx.__enter__.return_value = resp

        with patch.object(threading.Thread, "start", _capture_start):
            with patch("forge.notifiers.discord.urlopen", return_value=ctx):
                n.send_cache_hit("rmsnorm")

        _wait_for_notifier_thread()
        assert len(launched) == 1
        assert launched[0].daemon is True
        assert launched[0].name == "discord-notifier"

    def test_retry_on_429(self) -> None:
        """429 レスポンスで Retry-After を読んで 1 回リトライすることを確認する。"""
        from urllib.error import HTTPError

        call_count = 0

        def _rate_limit_then_ok(req, timeout=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                headers = MagicMock()
                headers.get.return_value = "0.01"  # 10ms retry-after
                raise HTTPError(req.full_url, 429, "Too Many Requests", headers, None)
            resp = MagicMock()
            resp.status = 204
            ctx = MagicMock()
            ctx.__enter__.return_value = resp
            ctx.__exit__.return_value = None
            return ctx

        n = _configured()
        # patch はスレッドが完了するまで生きている必要があるため、with ブロック内で join する
        with patch("forge.notifiers.discord.urlopen", side_effect=_rate_limit_then_ok):
            n.send_cache_hit("rmsnorm")
            _wait_for_notifier_thread()
        assert call_count == 2, f"Expected 2 calls (initial + retry), got {call_count}"
>>>>>>> 6e06dc7 (fix(#273): make DiscordNotifier._send_webhook non-blocking via daemon thread)
