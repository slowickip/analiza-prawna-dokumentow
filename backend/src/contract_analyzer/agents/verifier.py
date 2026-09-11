"""The verifier: decides one candidate at a time, on what the worksheet shows it.

It cannot search. Its reads are confined to locators already on this candidate's
worksheet, with one cap shared by every verifier visit for that candidate.
That is the point of the role: a verdict reached by browsing the corpus would be
the verifier doing the researcher's work and then agreeing with itself.
"""

from __future__ import annotations

import logging
from functools import partial

from contract_analyzer.agents.entries import CandidateEntry
from contract_analyzer.agents.retrieval import read_provision, run_read
from contract_analyzer.agents.scheduler import (
    TASK_MAX_TURNS,
    VERIFIER_MAX_READS_PER_CANDIDATE,
    Step,
)
from contract_analyzer.agents.session import role_payload
from contract_analyzer.agents.state import UnitContext
from contract_analyzer.agents.tools import (
    ReadProvisionArgs,
    RelationVerdictArgs,
    RelevanceVerdictArgs,
    parse_tool,
    tools_for,
)
from contract_analyzer.agents.turns import ToolResult, refusal, refused, run_task
from contract_analyzer.agents.verdicts import record_relation, record_relevance
from contract_analyzer.model import AttemptTelemetry, ToolInvocation

logger = logging.getLogger(__name__)


async def act(context: UnitContext, step: Step) -> list[str]:
    """Run the verifier task the scheduler named; returns the findings it stored."""
    if step.task not in ("verifier.relevance", "verifier.relation"):
        raise RuntimeError(f"verifier node reached with {step.task}")
    candidate = context.candidate_for(step)
    findings: list[str] = []
    if step.task == "verifier.relevance":
        outcome = await run_task(
            context.active,
            role="verifier",
            task="verifier.relevance",
            unit_id=context.call_unit.unit_id,
            payload=await _payload(context, candidate, "candidate_text"),
            tools=tools_for("verifier.relevance"),
            handle_tool=partial(_handle_relevance, context, candidate, findings),
            max_turns=TASK_MAX_TURNS["verifier.relevance"],
        )
    else:
        outcome = await run_task(
            context.active,
            role="verifier",
            task="verifier.relation",
            unit_id=context.call_unit.unit_id,
            payload=await _relation_payload(context, candidate),
            tools=tools_for("verifier.relation"),
            handle_tool=partial(_handle_relation, context, candidate, findings),
            max_turns=TASK_MAX_TURNS["verifier.relation"],
        )
    context.verifier_turns += outcome.turns
    context.active.verifier_tool_turns += outcome.turns
    return findings


async def _payload(
    context: UnitContext, candidate: CandidateEntry, text_field: str
) -> dict[str, object]:
    provision = await read_provision(context.active, candidate.locator)
    if provision is None:
        raise RuntimeError("worksheet candidate missing from corpus")
    return {
        **role_payload(context.active.session, context.call_unit),
        "worksheet": context.worksheet.render("verifier", candidate.id),
        "candidate_locator": candidate.locator,
        "candidate_act_identifier": candidate.act_identifier,
        text_field: provision.text,
    }


async def _relation_payload(
    context: UnitContext, candidate: CandidateEntry
) -> dict[str, object]:
    """The relation payload, with the characterisation the direction rule needs.

    The character record decides whether a direction is required of this role at
    all, so it travels with the basis rather than being implied.
    """
    character = context.worksheet.character_for(candidate.id)
    if character is None:
        raise RuntimeError("verifier.relation scheduled without a characterisation")
    direction = character.character.semi_imperative_direction
    return {
        **await _payload(context, candidate, "basis_text"),
        "basis_character": character.character.kind.value,
        "basis_permitted_direction": (
            direction.model_dump() if direction is not None else None
        ),
    }


async def _handle_relevance(
    context: UnitContext,
    candidate: CandidateEntry,
    findings: list[str],
    invocation: ToolInvocation,
    attempts: tuple[AttemptTelemetry, ...],
) -> ToolResult:
    parsed = parse_tool(
        tools_for("verifier.relevance"), invocation.name, invocation.arguments
    )
    if isinstance(parsed, dict):
        return refused(parsed, invocation.name, context=context)
    if isinstance(parsed, ReadProvisionArgs):
        return ToolResult(await _guarded_read(context, candidate, parsed.locator))
    if isinstance(parsed, RelevanceVerdictArgs):
        dependency_error = _dependency_error(context, candidate, parsed.based_on)
        if dependency_error is not None:
            return refused(dependency_error, invocation.name, context=context)
        result = record_relevance(context, candidate, parsed, attempts)
        if isinstance(result, ToolResult):
            return result
        findings.extend(result)
        return ToolResult({"recorded": True}, committed=True)
    return refused(refusal("unknown_tool"), invocation.name, context=context)


async def _handle_relation(
    context: UnitContext,
    candidate: CandidateEntry,
    findings: list[str],
    invocation: ToolInvocation,
    attempts: tuple[AttemptTelemetry, ...],
) -> ToolResult:
    parsed = parse_tool(
        tools_for("verifier.relation"), invocation.name, invocation.arguments
    )
    if isinstance(parsed, dict):
        return refused(parsed, invocation.name, context=context)
    if isinstance(parsed, ReadProvisionArgs):
        return ToolResult(await _guarded_read(context, candidate, parsed.locator))
    if isinstance(parsed, RelationVerdictArgs):
        dependency_error = _dependency_error(context, candidate, parsed.based_on)
        if dependency_error is not None:
            return refused(dependency_error, invocation.name, context=context)
        result = record_relation(context, candidate, parsed, attempts)
        if isinstance(result, ToolResult):
            return result
        findings.extend(result)
        return ToolResult({"recorded": True}, committed=True)
    return refused(refusal("unknown_tool"), invocation.name, context=context)


async def _guarded_read(
    context: UnitContext, candidate: CandidateEntry, locator: str
) -> dict[str, object]:
    """Read a provision only if this unit already has a reason to look at it.

    The cap counts calls rather than successful reads: a refused read has already
    cost the exchange a slot, and counting only what succeeded would let a role
    probe the corpus for free by naming locators outside its act.
    """
    reads = context.verifier_reads_by_candidate.get(candidate.id, 0)
    if reads >= VERIFIER_MAX_READS_PER_CANDIDATE:
        logger.warning(
            "tool refusal run=%s unit=%s tool=read_provision code=read_limit_reached",
            context.active.run_id,
            context.call_unit.unit_id,
        )
        return refusal("read_limit_reached", limit=VERIFIER_MAX_READS_PER_CANDIDATE)
    context.verifier_reads_by_candidate[candidate.id] = reads + 1
    if locator not in context.worksheet.known_locators(candidate.id):
        logger.warning(
            "tool refusal run=%s unit=%s tool=read_provision "
            "code=locator_not_on_worksheet",
            context.active.run_id,
            context.call_unit.unit_id,
        )
        return refusal("locator_not_on_worksheet")
    return await run_read(context, "verifier", locator)


def _dependency_error(
    context: UnitContext, candidate: CandidateEntry, based_on: tuple[str, ...]
) -> dict[str, object] | None:
    visible = context.worksheet.verifier_visible_entry_ids(candidate.id)
    if not set(based_on) <= visible:
        return refusal("verdict_dependency_not_visible")
    return None
