"""Discord webhook notifications for forge optimization events."""

import json
import logging
import os
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import UTC, datetime
from http.client import HTTPResponse
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

_RETRY_AFTER_DEFAULT = 1.0  # seconds to wait after a 429 with no Retry-After header

# #330: 通知のたびに無制限に daemon スレッドを生成していると、短命な
# Orchestrator インスタンスを高頻度・並行に使う運用でスレッドが積み上がり
# うる。プロセス全体で共有する固定サイズの ThreadPoolExecutor に一本化し、
# 同時実行数を明示的に上限管理する。
_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="discord-notifier")
_pending_futures: set[Future] = set()
_pending_lock = threading.Lock()


def _track_future(future: Future) -> None:
    with _pending_lock:
        _pending_futures.add(future)
    future.add_done_callback(_untrack_future)


def _untrack_future(future: Future) -> None:
    with _pending_lock:
        _pending_futures.discard(future)


class DiscordNotifier:
    """Send optimization events to Discord via webhook."""

    def __init__(self) -> None:
        """Initialize Discord notifier from environment variables."""
        self.completion_webhook = os.getenv("DISCORD_WEBHOOK_COMPLETION")
        self.error_webhook = os.getenv("DISCORD_WEBHOOK_ERRORS")

    def send_optimization_complete(
        self,
        op_name: str,
        best_time: float,
        num_candidates: int,
        duration_seconds: float,
        speedup: float | None = None,
        best_round: int | None = None,
        failed_rate: float | None = None,
    ) -> bool:
        """Send notification when optimization completes.

        Args:
            op_name: Operation name (e.g., 'rmsnorm', 'softmax')
            best_time: Best execution time in milliseconds
            num_candidates: Number of candidates explored
            duration_seconds: Total optimization duration
            speedup: baseline_us / best_us ratio (optional)
            best_round: Round number where best was found (optional)
            failed_rate: failed_count / num_candidates (optional)

        Returns:
            True if notification dispatched to background thread
        """
        if not self.completion_webhook:
            logger.debug("Discord completion webhook not configured")
            return False

        try:
            fields: list[dict[str, Any]] = [
                {"name": "Operation", "value": op_name, "inline": True},
                {"name": "Best Time", "value": f"{best_time:.3f}ms", "inline": True},
                {"name": "Candidates", "value": str(num_candidates), "inline": True},
                {"name": "Optimization Time", "value": f"{duration_seconds:.2f}s", "inline": True},
            ]
            if speedup is not None:
                fields.append({"name": "Speedup", "value": f"{speedup:.2f}×", "inline": True})
            if best_round is not None:
                fields.append({"name": "Best Round", "value": str(best_round), "inline": True})
            if failed_rate is not None:
                fields.append({"name": "Fail Rate", "value": f"{failed_rate:.0%}", "inline": True})

            embed = {
                "title": f"GPU Kernel Optimized: {op_name}",
                "description": (
                    f"Found optimal kernel after exploring {num_candidates} candidates\n"
                    f"Best execution time: **{best_time:.3f}ms**"
                ),
                "color": 65280,  # Green
                "fields": fields,
                "timestamp": datetime.now(UTC).isoformat(),
            }

            return self._send_webhook(self.completion_webhook, {"embeds": [embed]})
        except Exception as e:
            logger.warning(f"Failed to send optimization notification: {e}", exc_info=True)
            return False

    def send_optimization_error(
        self,
        op_name: str,
        error_message: str,
        error_type: str | None = None,
    ) -> bool:
        """Send notification when optimization fails.

        Args:
            op_name: Operation name
            error_message: Error message
            error_type: Error type (optional)

        Returns:
            True if notification dispatched to background thread
        """
        if not self.error_webhook:
            logger.debug("Discord error webhook not configured")
            return False

        try:
            fields = [
                {"name": "Operation", "value": op_name, "inline": True},
                {"name": "Status", "value": "FAILED", "inline": True},
            ]

            if error_type:
                fields.append({"name": "Error Type", "value": error_type, "inline": True})

            embed = {
                "title": f"Optimization Failed: {op_name}",
                "description": error_message,
                "color": 15158332,  # Red
                "fields": fields,
                "timestamp": datetime.now(UTC).isoformat(),
            }

            return self._send_webhook(self.error_webhook, {"embeds": [embed]})
        except Exception as e:
            logger.warning(f"Failed to send error notification: {e}", exc_info=True)
            return False

    def send_cache_hit(self, op_name: str) -> bool:
        """Send notification when cache hit occurs.

        Args:
            op_name: Operation name

        Returns:
            True if notification dispatched to background thread
        """
        if not self.completion_webhook:
            return False

        try:
            embed = {
                "title": f"Cache Hit: {op_name}",
                "description": "Using previously optimized kernel",
                "color": 3447003,  # Blue
                "timestamp": datetime.now(UTC).isoformat(),
            }

            return self._send_webhook(self.completion_webhook, {"embeds": [embed]})
        except Exception as e:
            logger.warning(f"Failed to send cache hit notification: {e}", exc_info=True)
            return False

    @staticmethod
    def _validate_webhook_url(url: str) -> None:
        """Validate that the webhook URL is a legitimate Discord HTTPS endpoint.

        Raises:
            ValueError: If the URL scheme is not https or the host is not
                        a discord.com subdomain (SSRF prevention).
        """
        parsed = urlparse(url)
        if parsed.scheme != "https" or not (parsed.hostname or "").endswith("discord.com"):
            raise ValueError(f"Invalid Discord webhook URL: {url!r}")

    def _send_webhook(self, webhook_url: str, payload: dict[str, Any]) -> bool:
        """Dispatch payload to Discord webhook via the shared background executor.

        The HTTP request is performed off the critical path so that DNS delays
        or Discord congestion cannot block ``optimize()`` completion or inflate
        ``total_time_s`` measurements.  A single 429 retry (honouring
        ``Retry-After``) is attempted before giving up.

        Submissions go through a process-wide, fixed-size ``ThreadPoolExecutor``
        (``_EXECUTOR``) rather than spawning a new ``threading.Thread`` per call
        (#330), bounding concurrent Discord requests instead of letting them
        pile up unbounded under high-frequency notification workloads.

        Args:
            webhook_url: Discord webhook URL
            payload: JSON payload to send

        Returns:
            True immediately (task successfully submitted); False if the task
            could not be submitted.
        """
        try:
            self._validate_webhook_url(webhook_url)
            data = json.dumps(payload).encode("utf-8")
        except Exception as e:
            logger.warning(f"Failed to serialise Discord payload: {e}", exc_info=True)
            return False

        def _do_send() -> None:
            req = Request(
                webhook_url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urlopen(req, timeout=5) as response:
                    _handle_response(response)
            except HTTPError as exc:
                if exc.code == 429:
                    retry_after = _retry_after_seconds(exc)
                    logger.warning(
                        "Discord webhook rate-limited (429); retrying after %.1fs", retry_after
                    )
                    time.sleep(retry_after)
                    _retry_once(webhook_url, data)
                else:
                    logger.warning("Discord webhook HTTP error %s: %s", exc.code, exc)
            except URLError as e:
                logger.warning(f"Failed to connect to Discord: {e}", exc_info=True)
            except Exception as e:
                logger.warning(f"Unexpected error sending Discord notification: {e}", exc_info=True)

        try:
            future = _EXECUTOR.submit(_do_send)
            _track_future(future)
            return True
        except Exception as e:
            logger.warning(f"Failed to submit Discord notification: {e}", exc_info=True)
            return False


# ---------------------------------------------------------------------------
# Module-level helpers (no access to self needed)
# ---------------------------------------------------------------------------


def _handle_response(response: HTTPResponse) -> None:
    if response.status == 204:
        logger.debug("Discord notification sent successfully")
    else:
        logger.warning("Discord webhook returned %s", response.status)


def _retry_after_seconds(exc: HTTPError) -> float:
    """Parse Retry-After header from a 429 HTTPError; fall back to default."""
    try:
        value = exc.headers.get("Retry-After", "")
        return float(value) if value else _RETRY_AFTER_DEFAULT
    except (ValueError, AttributeError):
        return _RETRY_AFTER_DEFAULT


def _retry_once(webhook_url: str, data: bytes) -> None:
    """Attempt a single retry of the webhook POST after a 429."""
    req = Request(
        webhook_url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(req, timeout=5) as response:
            _handle_response(response)
    except Exception as e:
        logger.warning("Discord webhook retry failed: %s", e, exc_info=True)


def wait_for_pending_notifications(timeout: float = 5.0) -> None:
    """Block until all currently in-flight webhook submissions complete.

    Intended for tests: with a shared long-lived ``ThreadPoolExecutor`` (#330),
    worker threads persist between tasks, so joining by thread name is no
    longer a valid way to detect completion. This waits on the tracked
    ``Future`` objects instead.
    """
    import concurrent.futures

    with _pending_lock:
        futures = list(_pending_futures)
    concurrent.futures.wait(futures, timeout=timeout)
