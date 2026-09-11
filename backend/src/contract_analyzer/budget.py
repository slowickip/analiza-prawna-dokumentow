from __future__ import annotations

import time
from typing import Any

from contract_analyzer.storage import InterruptionReason


class BudgetTracker:
    """The wall clock a run works against.

    A run is bounded by time, by the per-role turn and tool caps, and by the
    per-response output cap. It is not bounded by a total token count: that
    ceiling never decided a run, and the reservation logic it needed to be safe
    under concurrency was the most intricate code here.

    ``exhausted`` reads the clock rather than a flag someone had to refresh.
    Callers ask it at points where no model call has just been made, and a stale
    answer there would start work the budget no longer allows.
    """

    def __init__(
        self,
        *,
        wall_budget_seconds: float,
        clock: Any | None = None,
    ) -> None:
        self._clock = clock or time.monotonic
        self._deadline = self._clock() + wall_budget_seconds
        self._exhausted = False
        self.exhaustion_reason: InterruptionReason | None = None

    @property
    def exhausted(self) -> bool:
        if self._exhausted:
            return True
        if self._clock() >= self._deadline:
            self._mark_exhausted("wall_time")
            return True
        return False

    def can_schedule(self) -> bool:
        return not self.exhausted

    def remaining_seconds(self) -> float:
        if self.exhausted:
            return 0.0
        return self._deadline - self._clock()

    def expire(self) -> None:
        self._mark_exhausted("wall_time")

    def _mark_exhausted(self, reason: InterruptionReason) -> None:
        self._exhausted = True
        if self.exhaustion_reason is None:
            self.exhaustion_reason = reason
