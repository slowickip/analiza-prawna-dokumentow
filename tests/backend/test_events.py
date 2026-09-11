from __future__ import annotations

from uuid import uuid4

import pytest
from pydantic import ValidationError

from contract_analyzer.storage import EventBus, RunEvent


def test_publish_delivers_only_to_matching_run_subscribers() -> None:
    bus = EventBus(queue_size=2)
    run_id = uuid4()
    other_run_id = uuid4()
    subscribed = bus.subscribe(run_id)
    other = bus.subscribe(other_run_id)
    event = RunEvent.status_changed("running")

    bus.publish(run_id, event)

    assert subscribed.get_nowait() == event
    assert other.empty()


def test_full_queue_drops_oldest_event_and_keeps_latest_state() -> None:
    bus = EventBus(queue_size=2)
    run_id = uuid4()
    queue = bus.subscribe(run_id)
    first = RunEvent.counter("processed_units", 1)
    second = RunEvent.counter("processed_units", 2)
    third = RunEvent.finding(uuid4())

    bus.publish(run_id, first)
    bus.publish(run_id, second)
    bus.publish(run_id, third)

    assert queue.get_nowait() == second
    assert queue.get_nowait() == third
    assert queue.empty()


def test_unsubscribe_stops_delivery() -> None:
    bus = EventBus(queue_size=1)
    run_id = uuid4()
    queue = bus.subscribe(run_id)

    bus.unsubscribe(run_id, queue)
    bus.publish(run_id, RunEvent.status_changed("completed"))

    assert queue.empty()


def test_events_reject_prose_and_invalid_shapes() -> None:
    with pytest.raises(ValidationError):
        RunEvent(kind="status", status="running", counter_name="document text")
    with pytest.raises(ValidationError):
        RunEvent(kind="counter", counter_name="processed units", count=1)
    with pytest.raises(ValidationError):
        RunEvent(kind="finding", status="fixture prose", finding_id=uuid4())


def test_queue_size_must_be_positive() -> None:
    with pytest.raises(ValueError, match="queue_size"):
        EventBus(queue_size=0)
