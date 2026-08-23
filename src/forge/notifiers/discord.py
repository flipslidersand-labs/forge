"""Discord webhook notifications for forge optimization events."""

import json
import logging
import os
from datetime import UTC, datetime
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)


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
            True if notification sent successfully
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
            logger.error(f"Failed to send optimization notification: {e}")
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
            True if notification sent successfully
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
            logger.error(f"Failed to send error notification: {e}")
            return False

    def send_cache_hit(self, op_name: str) -> bool:
        """Send notification when cache hit occurs.

        Args:
            op_name: Operation name

        Returns:
            True if notification sent successfully
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
            logger.error(f"Failed to send cache hit notification: {e}")
            return False

    def _send_webhook(self, webhook_url: str, payload: dict[str, Any]) -> bool:
        """Send payload to Discord webhook.

        Args:
            webhook_url: Discord webhook URL
            payload: JSON payload to send

        Returns:
            True if successful (HTTP 204)
        """
        try:
            data = json.dumps(payload).encode("utf-8")
            req = Request(
                webhook_url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            with urlopen(req, timeout=5) as response:
                if response.status == 204:
                    logger.debug("Discord notification sent successfully")
                    return True
                else:
                    logger.error(f"Discord webhook returned {response.status}")
                    return False
        except URLError as e:
            logger.error(f"Failed to connect to Discord: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error sending Discord notification: {e}")
            return False
