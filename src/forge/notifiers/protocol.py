"""通知先の共通インターフェース。Orchestrator は具象クラスでなくこれに依存する。"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Notifier(Protocol):
    """最適化イベントの通知先。DiscordNotifier などが実装する。"""

    def send_optimization_complete(
        self,
        op_name: str,
        best_time: float,
        num_candidates: int,
        duration_seconds: float,
        speedup: float | None = None,
        best_round: int | None = None,
        failed_rate: float | None = None,
    ) -> bool: ...

    def send_optimization_error(
        self,
        op_name: str,
        error_message: str,
        error_type: str | None = None,
    ) -> bool: ...

    def send_cache_hit(self, op_name: str) -> bool: ...
