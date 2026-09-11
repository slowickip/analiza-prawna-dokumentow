"""HTTP endpoints for immutable post-run chat, re-analysis, and worksheets."""

from __future__ import annotations

import asyncio
import logging
from typing import Literal
from uuid import UUID, uuid4

from fastapi import APIRouter, Request

from contract_analyzer.agents.errors import (
    ChatOutOfScope,
    InvalidChatCitation,
    MessageNotRoutable,
)
from contract_analyzer.agents.message_router import routed_uuid, selected_for_ask
from contract_analyzer.agents.session import (
    ChatMessage,
    ChatRequest,
    RouteRequest,
    RunRequest,
    build_call_units,
)
from contract_analyzer.api.errors import _ApiError, _raise_api
from contract_analyzer.api.run_builders import _build_run
from contract_analyzer.api.runs import _on_run_task_done
from contract_analyzer.api.schemas import (
    Error,
    MessageBody,
    MessageResponse,
    Run,
    WorksheetResponse,
    WorksheetUnit,
)
from contract_analyzer.api.state import (
    _AppState,
    _content_available,
    _ensure_corpus_available,
    _state,
)
from contract_analyzer.storage import RunRecord

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1")


def _reject_sealed(state: _AppState) -> None:
    if state.settings.evaluation_batch_open:
        _raise_api("evaluation_batch_open")


@router.post(
    "/runs/{runId}/message",
    response_model=MessageResponse,
    responses={
        404: {"model": Error},
        409: {"model": Error},
        410: {"model": Error},
        422: {"model": Error},
        503: {"model": Error},
    },
    tags=["chat"],
)
async def message(runId: UUID, body: MessageBody, request: Request) -> MessageResponse:
    """Read one reader message and dispatch to ask, contest, or analyse."""
    state = _state(request)
    _reject_sealed(state)
    parent = state.services.metadata.get_run(runId)
    if parent is None:
        _raise_api("not_found")
    if parent.status != "completed":
        _raise_api("run_not_finished")
    if not _content_available(state, parent.document_id):
        _raise_api("content_expired")
    _ensure_corpus_available(state)
    if not state.settings.model_api_key:
        _raise_api("dependency_unavailable")

    chat_history = tuple(
        ChatMessage(role=item["role"].value, content=item["content"])
        for item in body.history
    )

    slot = uuid4()
    route_task = asyncio.create_task(
        state.runner.route(
            RouteRequest(
                run_id=runId,
                message=body.message,
                history=chat_history,
            )
        )
    )
    state.tasks.register(slot, route_task, parent.document_id)
    try:
        routed = await route_task
    except KeyError:
        _raise_api("not_found")
    except MessageNotRoutable as error:
        raise _ApiError(error.code) from error
    finally:
        state.tasks.release(slot)

    try:
        if routed.intent == "contest":
            return MessageResponse(
                intent="contest",
                run=_start_interaction(
                    state,
                    parent,
                    kind="contest",
                    note=body.message,
                    finding_id=(
                        routed_uuid(routed.finding_id) if routed.finding_id else None
                    ),
                    unit_id=None,
                ),
            )
        if routed.intent == "analyse":
            return MessageResponse(
                intent="analyse",
                run=_start_interaction(
                    state,
                    parent,
                    kind="analyse",
                    note=body.message,
                    finding_id=None,
                    unit_id=routed.unit_id,
                ),
            )

        findings = state.services.metadata.list_findings(parent.id)
        selected_finding_ids = selected_for_ask(routed, findings)
    except MessageNotRoutable as error:
        raise _ApiError(error.code) from error

    chat_request = ChatRequest(
        run_id=runId,
        question=body.message,
        selected_finding_ids=selected_finding_ids,
        history=chat_history,
    )
    ask_slot = uuid4()
    task = asyncio.create_task(state.runner.explain(chat_request))
    state.tasks.register(ask_slot, task, parent.document_id)
    try:
        answer = await task
    except KeyError:
        _raise_api("not_found")
    except (ChatOutOfScope, InvalidChatCitation) as error:
        raise _ApiError(error.code) from error
    finally:
        state.tasks.release(ask_slot)
    return MessageResponse(intent="ask", answer=answer)


def _start_interaction(
    state: _AppState,
    parent: RunRecord,
    *,
    kind: Literal["contest", "analyse"],
    note: str,
    finding_id: UUID | None,
    unit_id: str | None,
) -> Run:
    """Start one unmeasured child analysis for an objection or a re-analysis."""
    try:
        session = state.services.sessions[parent.document_id]
    except KeyError:
        _raise_api("content_expired")
    call_units = build_call_units(session, parent.arm)
    resolved_finding_id: UUID | None = None
    if kind == "contest":
        finding = next(
            (
                item
                for item in state.services.metadata.list_findings(parent.id)
                if item.id == finding_id
            ),
            None,
        )
        if finding is None:
            _raise_api("not_found")
        target_unit_id = finding.unit_id
        resolved_finding_id = finding.id
    else:
        target_unit_id = unit_id or ""
        if target_unit_id not in {item.unit_id for item in call_units}:
            _raise_api("not_found")

    prepared = state.runner.start_run(
        RunRequest(
            document_id=parent.document_id,
            arm=parent.arm,
            measurement_valid=False,
            wall_budget_seconds=parent.wall_budget_seconds,
            concurrency=1,
            parameters=parent.parameters,
            parent_run_id=parent.id,
            interaction=kind,
            unit_id=target_unit_id,
            user_note=note,
            user_note_finding_id=resolved_finding_id,
        )
    )
    child = state.services.metadata.get_run(prepared.run_id)
    if child is None:
        raise RuntimeError("child run record not found after start")
    task = asyncio.create_task(state.runner.execute_run(prepared))
    state.tasks.register(child.id, task, parent.document_id)
    task.add_done_callback(
        lambda completed: _on_run_task_done(state, child.id, completed)
    )
    logger.info(
        "interaction created parent=%s child=%s kind=%s unit=%s",
        parent.id,
        child.id,
        kind,
        target_unit_id,
    )
    return _build_run(child, state)


@router.get(
    "/runs/{runId}/worksheet",
    response_model=WorksheetResponse,
    responses={
        404: {"model": Error},
        409: {"model": Error},
        410: {"model": Error},
        422: {"model": Error},
    },
    tags=["interactions"],
)
async def get_worksheet(runId: UUID, request: Request) -> WorksheetResponse:
    """Return retained worksheet prose for one known run."""
    state = _state(request)
    _reject_sealed(state)
    run = state.services.metadata.get_run(runId)
    if run is None:
        _raise_api("not_found")
    if not _content_available(state, run.document_id):
        _raise_api("content_expired")
    if run.status == "running":
        _raise_api("run_not_finished")
    unit_ids = state.runner.worksheet_unit_ids(runId)
    units: list[WorksheetUnit] = []
    for unit_id in unit_ids:
        record = state.runner.worksheet_for(runId, unit_id)
        if record is None:
            _raise_api("content_expired")
        units.append(
            WorksheetUnit(
                unit_id=unit_id,
                entries=[entry.model_dump(mode="json") for entry in record.entries],
            )
        )
    return WorksheetResponse(run_id=runId, units=units)
