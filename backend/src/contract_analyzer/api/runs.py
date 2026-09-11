"""Run lifecycle and event-stream endpoints."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from typing import Literal, cast
from uuid import UUID

from fastapi import APIRouter, Depends, Request
from fastapi.responses import EventSourceResponse
from fastapi.sse import ServerSentEvent

from contract_analyzer.agents.session import RunRequest, RunResult
from contract_analyzer.api.errors import _raise_api
from contract_analyzer.api.run_builders import _build_run
from contract_analyzer.api.schemas import CreateRunBody, Error, Run
from contract_analyzer.api.state import (
    _AppState,
    _content_available,
    _ensure_corpus_available,
    _state,
)
from contract_analyzer.domain import ArmCode
from contract_analyzer.storage import EventBus, RunEvent, RunRecord, RunTotals

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1")


def _settle_run(
    state: _AppState,
    run_id: UUID,
    status: Literal["failed", "cancelled"],
    *,
    error_code: str | None = None,
    elapsed_ms: float | None = None,
) -> RunRecord | None:
    record = state.services.metadata.get_run(run_id)
    if record is None:
        logger.warning("run %s vanished while settling to status %s", run_id, status)
        return None
    if record.status != "running":
        return record

    if elapsed_ms is None:
        elapsed_ms = state.services.metadata.elapsed_ms_for_run(run_id)
        if elapsed_ms is None and status == "cancelled":
            raise RuntimeError(f"run {run_id} has no recorded monotonic start")
    try:
        state.services.metadata.finish_run(
            run_id,
            status,
            totals=RunTotals(error_code=error_code, elapsed_ms=elapsed_ms),
        )
    except KeyError:
        logger.warning("run %s vanished while settling to status %s", run_id, status)
        return None

    if state.services.events is not None:
        state.services.events.publish(run_id, RunEvent.status_changed(status))
    state.runner.retire_run(run_id)
    return state.services.metadata.get_run(run_id)


def _on_run_task_done(
    state: _AppState, run_id: UUID, task: asyncio.Task[RunResult]
) -> None:
    try:
        task.result()
    except asyncio.CancelledError:
        if run_id in state.tasks.settling:
            # A cancel request owns this one: it settles and releases.
            return
        # Nothing asked for this cancellation, so nobody else will write a
        # terminal record. Left alone the run stays registered for ever and
        # an evaluation batch of one never admits another.
        try:
            _settle_run(state, run_id, "cancelled")
        finally:
            state.tasks.release(run_id)
    except Exception:
        logger.exception("run %s failed with an unhandled exception", run_id)
        try:
            _settle_run(state, run_id, "failed", error_code="internal_error")
        finally:
            state.tasks.release(run_id)
    else:
        state.tasks.release(run_id)


@router.post(
    "/runs",
    status_code=202,
    response_model=Run,
    responses={
        404: {"model": Error},
        409: {"model": Error},
        410: {"model": Error},
        422: {"model": Error},
        503: {"model": Error},
    },
)
async def create_run(body: CreateRunBody, request: Request) -> Run:
    state = _state(request)

    if body.document_id not in state.services.sessions:
        _raise_api("not_found")
    if not _content_available(state, body.document_id):
        _raise_api("content_expired")

    if state.settings.evaluation_batch_open and state.tasks.has_running():
        _raise_api("run_already_active")

    _ensure_corpus_available(state)
    if not state.settings.model_api_key:
        _raise_api("dependency_unavailable")

    try:
        arm = ArmCode(body.arm)
    except ValueError:
        _raise_api("not_found")

    run_request = RunRequest(document_id=body.document_id, arm=arm)
    prepared = state.runner.start_run(run_request)
    record = state.services.metadata.get_run(prepared.run_id)
    if record is None:
        raise RuntimeError("run record not found after start")

    task = asyncio.create_task(state.runner.execute_run(prepared))
    state.tasks.register(prepared.run_id, task, run_request.document_id)
    task.add_done_callback(
        lambda completed: _on_run_task_done(state, prepared.run_id, completed)
    )
    return _build_run(record, state)


@router.get(
    "/runs/{runId}",
    response_model=Run,
    responses={404: {"model": Error}, 422: {"model": Error}},
)
async def get_run(runId: UUID, request: Request) -> Run:
    state = _state(request)
    record = state.services.metadata.get_run(runId)
    if record is None:
        _raise_api("not_found")
    return _build_run(record, state)


@router.post(
    "/runs/{runId}/cancel",
    status_code=202,
    response_model=Run,
    responses={
        404: {"model": Error},
        409: {"model": Error},
        422: {"model": Error},
    },
)
async def cancel_run(runId: UUID, request: Request) -> Run:
    state = _state(request)
    record = state.services.metadata.get_run(runId)
    if record is None:
        _raise_api("not_found")
    if record.status in ("completed", "failed", "cancelled"):
        _raise_api("run_already_terminal")

    # A task that has already ended settles itself; its callback is only
    # waiting for the loop. Cancelling it here would record the run as
    # cancelled when what actually happened is that it raised.
    task = state.tasks.active_tasks.get(runId)
    if task is not None and task.done():
        _raise_api("run_already_terminal")

    state.tasks.settling.add(runId)
    await state.tasks.cancel_for_run(runId)
    try:
        settled = _settle_run(state, runId, "cancelled")
        if settled is None:
            _raise_api("not_found")
        return _build_run(settled, state)
    finally:
        # The slot was held through the cancellation so no other run could be
        # admitted while this one had no terminal record.
        state.tasks.release(runId)


def _require_streamable_run(runId: UUID, request: Request) -> None:
    """Reject an unknown run before the stream exists."""
    state = _state(request)
    if state.services.metadata.get_run(runId) is None:
        _raise_api("not_found")
    if state.services.events is None:
        _raise_api("dependency_unavailable")


@router.get(
    "/runs/{runId}/events",
    dependencies=[Depends(_require_streamable_run)],
    response_class=EventSourceResponse,
    responses={
        200: {"content": {"text/event-stream": {}}},
        404: {"model": Error},
        422: {"model": Error},
    },
)
async def stream_events(
    runId: UUID, request: Request
) -> AsyncIterator[ServerSentEvent]:
    state = _state(request)
    events: EventBus = cast(EventBus, state.services.events)
    queue = events.subscribe(runId)
    logger.debug("event stream subscribed run_id=%s", runId)

    try:
        current = state.services.metadata.get_run(runId)
        if current is not None and current.status in (
            "completed",
            "failed",
            "cancelled",
        ):
            data = {
                "kind": "status",
                "status": current.status,
                "run_id": str(runId),
            }
            yield ServerSentEvent(raw_data=json.dumps(data))
            return

        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=30.0)
            except TimeoutError:
                yield ServerSentEvent(comment="keepalive")
                continue
            data = event.model_dump(mode="json")
            data["run_id"] = str(runId)
            yield ServerSentEvent(raw_data=json.dumps(data))
            if event.kind == "status" and event.status in (
                "completed",
                "failed",
                "cancelled",
            ):
                break
    finally:
        events.unsubscribe(runId, queue)
        logger.debug("event stream closed run_id=%s", runId)
