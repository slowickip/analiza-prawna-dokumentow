"""Run event models and in-process delivery."""

from __future__ import annotations

import asyncio
from typing import Self, cast
from uuid import UUID

from pydantic import BaseModel, ConfigDict, model_validator

from contract_analyzer.storage.records import EventKind, _require_code


class RunEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: EventKind
    status: str | None = None
    counter_name: str | None = None
    count: int | None = None
    finding_id: UUID | None = None
    # A worksheet event says that a role wrote an entry of some kind about some
    # call unit. It deliberately cannot carry what the entry said: the worksheet
    # holds the roles' prose, and the event stream is not a place for it.
    role: str | None = None
    entry_kind: str | None = None
    unit_id: str | None = None

    @model_validator(mode="after")
    def validate_shape(self) -> Self:
        worksheet_fields = (self.role, self.entry_kind, self.unit_id)
        if self.kind == "worksheet":
            if any(value is None for value in worksheet_fields) or any(
                value is not None
                for value in (
                    self.status,
                    self.counter_name,
                    self.count,
                    self.finding_id,
                )
            ):
                raise ValueError("invalid worksheet event")
            _require_code(cast(str, self.role), "role")
            _require_code(cast(str, self.entry_kind), "entry_kind")
            if not cast(str, self.unit_id).strip():
                raise ValueError("invalid worksheet event")
            return self
        if any(value is not None for value in worksheet_fields):
            raise ValueError("worksheet fields belong to worksheet events")
        if self.kind == "status":
            if self.status is None or any(
                value is not None
                for value in (self.counter_name, self.count, self.finding_id)
            ):
                raise ValueError("invalid status event")
            _require_code(self.status, "status")
        elif self.kind == "counter":
            if (
                self.counter_name is None
                or self.count is None
                or self.count < 0
                or self.status is not None
                or self.finding_id is not None
            ):
                raise ValueError("invalid counter event")
            _require_code(self.counter_name, "counter_name")
        elif (
            self.finding_id is None
            or self.status is not None
            or self.counter_name is not None
            or self.count is not None
        ):
            raise ValueError("invalid finding event")
        return self

    @classmethod
    def status_changed(cls, status: str) -> RunEvent:
        return cls(kind="status", status=status)

    @classmethod
    def counter(cls, name: str, count: int) -> RunEvent:
        return cls(kind="counter", counter_name=name, count=count)

    @classmethod
    def finding(cls, finding_id: UUID) -> RunEvent:
        return cls(kind="finding", finding_id=finding_id)

    @classmethod
    def worksheet(cls, *, role: str, entry_kind: str, unit_id: str) -> RunEvent:
        return cls(kind="worksheet", role=role, entry_kind=entry_kind, unit_id=unit_id)


class EventBus:
    """Per-run bounded queues; on overflow the oldest event is discarded."""

    def __init__(self, *, queue_size: int = 100) -> None:
        if queue_size < 1:
            raise ValueError("queue_size must be positive")
        self._queue_size = queue_size
        self._subscribers: dict[UUID, set[asyncio.Queue[RunEvent]]] = {}

    def subscribe(self, run_id: UUID) -> asyncio.Queue[RunEvent]:
        queue: asyncio.Queue[RunEvent] = asyncio.Queue(maxsize=self._queue_size)
        self._subscribers.setdefault(run_id, set()).add(queue)
        return queue

    def unsubscribe(self, run_id: UUID, queue: asyncio.Queue[RunEvent]) -> None:
        subscribers = self._subscribers.get(run_id)
        if subscribers is None:
            return
        subscribers.discard(queue)
        if not subscribers:
            del self._subscribers[run_id]

    def publish(self, run_id: UUID, event: RunEvent) -> None:
        for queue in tuple(self._subscribers.get(run_id, ())):
            if queue.full():
                queue.get_nowait()
            queue.put_nowait(event)
