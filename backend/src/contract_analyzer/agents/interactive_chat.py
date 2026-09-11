"""Tool-based, immutable post-run explanation over selected findings.

The explainer is its own role, not the measured synthesizer under another name: it
reads finding, unit and provision text the synthesizer is never given, runs only
after a measured run has closed, and may not create or change a finding.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Mapping
from typing import Literal
from uuid import UUID

from contract_analyzer.agents.chat_grounding import (
    grounding,
    provisions,
    read_finding,
    read_provision,
    read_unit,
)
from contract_analyzer.agents.completion import finish_run
from contract_analyzer.agents.errors import ChatOutOfScope, InvalidChatCitation
from contract_analyzer.agents.interactive_tools import (
    CHAT_MAX_QUESTIONS,
    CHAT_MAX_TURNS,
    CHAT_TOOLS,
    CHAT_WALL_BUDGET_SECONDS,
    POST_ANSWER,
    FindingIdArgs,
    OpenQuestionArgs,
    PostAnswerArgs,
    RouteMessageArgs,
    UnitIdArgs,
)
from contract_analyzer.agents.message_router import route_message
from contract_analyzer.agents.researcher_question import answer_question
from contract_analyzer.agents.runner import start_run
from contract_analyzer.agents.runtime import GraphRuntime
from contract_analyzer.agents.services import AnalysisServices
from contract_analyzer.agents.session import (
    ChatRequest,
    ChatResponse,
    PreparedRun,
    RouteRequest,
    RunRequest,
    build_call_units,
)
from contract_analyzer.agents.state import ActiveRun
from contract_analyzer.agents.tool_args import ReadProvisionArgs
from contract_analyzer.agents.tools import parse_tool
from contract_analyzer.agents.turns import (
    ToolResult,
    refusal,
    refused,
    run_conversation,
)
from contract_analyzer.model import AttemptTelemetry, ToolInvocation
from contract_analyzer.storage import FindingRecord, InteractionKind, RunRecord

logger = logging.getLogger(__name__)


async def explain(
    services: AnalysisServices, runtime: GraphRuntime, request: ChatRequest
) -> ChatResponse:
    """Create an ask child run, answer with tools, and close only that child."""
    if not request.selected_finding_ids:
        raise ChatOutOfScope("out_of_scope")
    parent = services.metadata.get_run(request.run_id)
    if parent is None:
        raise KeyError(request.run_id)
    session = services.sessions.get(parent.document_id)
    if session is None:
        raise KeyError(parent.document_id)
    by_id = {item.id: item for item in services.metadata.list_findings(parent.id)}
    selected: list[FindingRecord] = []
    for finding_id in request.selected_finding_ids:
        if finding_id not in by_id:
            raise KeyError(finding_id)
        selected.append(by_id[finding_id])
    call_units_by_id = {
        unit.unit_id: unit for unit in build_call_units(session, parent.arm)
    }

    active, prepared = _begin(services, runtime, parent)
    answer: ChatResponse | None = None
    questions = 0
    corpus_consulted = False
    messages: list[Mapping[str, object]] = [
        {"role": "system", "content": services.prompt_bundle.prompts["explain.md"]}
    ]
    messages.extend(
        {"role": item.role, "content": item.content} for item in request.history[-6:]
    )

    async def handle(
        invocation: ToolInvocation, attempts: tuple[AttemptTelemetry, ...]
    ) -> ToolResult:
        nonlocal answer, questions, corpus_consulted
        del attempts
        parsed = parse_tool(CHAT_TOOLS, invocation.name, invocation.arguments)
        if isinstance(parsed, dict):
            return refused(parsed, invocation.name, active=active)
        if isinstance(parsed, FindingIdArgs):
            return ToolResult(read_finding(parsed.finding_id, selected, runtime))
        if isinstance(parsed, UnitIdArgs):
            return ToolResult(
                read_unit(parsed.unit_id, session, call_units_by_id, selected)
            )
        if isinstance(parsed, ReadProvisionArgs):
            result = await read_provision(active, parsed.locator)
            if "error" not in result:
                corpus_consulted = True
            return ToolResult(result)
        if isinstance(parsed, OpenQuestionArgs):
            if questions >= CHAT_MAX_QUESTIONS:
                return refused(
                    refusal("question_limit_reached", limit=CHAT_MAX_QUESTIONS)
                )
            questions += 1
            # Delegating is not consulting. corpus_consulted tells the reader
            # the answer rests on the frozen corpus, and the researcher may
            # refuse every phrase before the corpus is queried and then answer
            # from the question alone. Both counters rise only where it was
            # reached, so the delta is what the flag can honestly rest on.
            before = (active.finder_search_calls, active.provision_reads)
            result = await answer_question(active, parsed.text)
            if (active.finder_search_calls, active.provision_reads) != before:
                corpus_consulted = True
            return ToolResult(result)
        if isinstance(parsed, PostAnswerArgs):
            _validate_citations(parsed.cited_finding_ids, request.selected_finding_ids)
            answer = ChatResponse(
                answer=parsed.answer,
                cited_finding_ids=parsed.cited_finding_ids,
                interaction_run_id=active.run_id,
                corpus_consulted=corpus_consulted,
            )
            return ToolResult({"recorded": True}, committed=True)
        return refused(refusal("unknown_tool"))

    # Everything after the child record exists is inside this: the eager grounding
    # reads the corpus and can fail, and a cancelled ask is still an ask that ended.
    # Whatever leaves here leaves a terminal record behind, never a running one.
    try:
        messages.append(
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "prompt_version": "explain",
                        "payload": {
                            "question": request.question,
                            "grounding": {
                                "findings": [
                                    grounding(item, session, call_units_by_id)
                                    for item in selected
                                ],
                                "provisions": await provisions(active, selected),
                            },
                        },
                    },
                    sort_keys=True,
                ),
            }
        )
        turns = await run_conversation(
            active,
            messages=messages,
            tools=CHAT_TOOLS,
            prompt_version="explain",
            commit_tool=POST_ANSWER.name,
            max_turns=CHAT_MAX_TURNS,
            handle_tool=handle,
        )
    except asyncio.CancelledError:
        _finish(runtime, prepared, "failed", "process_restarted")
        raise
    except BaseException as error:
        _finish(runtime, prepared, "failed", getattr(error, "code", "internal_error"))
        raise
    if answer is None:
        raise RuntimeError("chat committed without an answer")
    elapsed = _finish(runtime, prepared, "completed")
    logger.info(
        "chat answered run=%s parent=%s turns=%d corpus_consulted=%s elapsed_ms=%.0f",
        active.run_id,
        parent.id,
        turns,
        corpus_consulted,
        elapsed,
    )
    return answer


async def route(
    services: AnalysisServices, runtime: GraphRuntime, request: RouteRequest
) -> RouteMessageArgs:
    """Read one reader message in its own unmeasured run and return the reading.

    Routing owns a run because it spends a model call, and every model call this
    system makes is answerable to a run record. The run it owns is the reading,
    not the work the reading selects: whatever the reader turns out to have asked
    for starts afterwards, in a run of its own.
    """
    parent = services.metadata.get_run(request.run_id)
    if parent is None:
        raise KeyError(request.run_id)
    session = services.sessions.get(parent.document_id)
    if session is None:
        raise KeyError(parent.document_id)
    findings = services.metadata.list_findings(parent.id)
    unit_ids = tuple(unit.unit_id for unit in build_call_units(session, parent.arm))

    active, prepared = _begin(services, runtime, parent, interaction="route")
    try:
        routed = await route_message(
            active, request.message, findings, unit_ids, request.history
        )
    except asyncio.CancelledError:
        _finish(runtime, prepared, "failed", "process_restarted")
        raise
    except BaseException as error:
        _finish(runtime, prepared, "failed", getattr(error, "code", "internal_error"))
        raise
    elapsed = _finish(runtime, prepared, "completed")
    logger.info(
        "message routed run=%s parent=%s intent=%s elapsed_ms=%.0f",
        active.run_id,
        parent.id,
        routed.intent,
        elapsed,
    )
    return routed


def _begin(
    services: AnalysisServices,
    runtime: GraphRuntime,
    parent: RunRecord,
    interaction: InteractionKind = "ask",
) -> tuple[ActiveRun, PreparedRun]:
    request = RunRequest(
        document_id=parent.document_id,
        arm=parent.arm,
        measurement_valid=False,
        wall_budget_seconds=CHAT_WALL_BUDGET_SECONDS,
        concurrency=1,
        parent_run_id=parent.id,
        interaction=interaction,
    )
    prepared = start_run(services, runtime, request, call_units=[], parent=parent)
    return runtime.active(prepared.run_id), prepared


def _finish(
    runtime: GraphRuntime,
    prepared: PreparedRun,
    status: Literal["completed", "failed"],
    error_code: str | None = None,
) -> float:
    elapsed = finish_run(
        runtime.services,
        runtime,
        prepared,
        status,
        error_code=error_code,
        analysis_counters=False,
    )
    runtime.retire_run(prepared.run_id)
    return elapsed


def _validate_citations(cited: tuple[str, ...], selected: tuple[UUID, ...]) -> None:
    selected_ids = set(selected)
    for value in cited:
        try:
            finding_id = UUID(value)
        except ValueError as error:
            raise InvalidChatCitation("chat_citation_unknown") from error
        if finding_id not in selected_ids:
            raise InvalidChatCitation("chat_citation_unknown")
