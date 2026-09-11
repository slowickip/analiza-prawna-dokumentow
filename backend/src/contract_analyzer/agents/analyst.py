"""The analyst characterises a provision the verifier has already found relevant.

One task, one decision: whether the provision binds absolutely, yields to the
parties, or protects one of them, and on what evidence. It holds no search tool,
so it can only reason over what the researcher retrieved for this unit, and it
does not rule on the relation its characterisation then governs.
"""

from __future__ import annotations

import logging
from functools import partial

from contract_analyzer.agents.entries import CandidateEntry
from contract_analyzer.agents.retrieval import read_provision, run_read
from contract_analyzer.agents.scheduler import TASK_MAX_TURNS, Step
from contract_analyzer.agents.session import role_payload
from contract_analyzer.agents.state import UnitContext
from contract_analyzer.agents.tools import (
    PostCharacterArgs,
    ReadProvisionArgs,
    parse_tool,
    tools_for,
)
from contract_analyzer.agents.turns import ToolResult, refusal, refused, run_task
from contract_analyzer.domain import ProvisionCharacter, ProvisionKind
from contract_analyzer.model import AttemptTelemetry, ToolInvocation

logger = logging.getLogger(__name__)


async def act(context: UnitContext, step: Step) -> None:
    """Run the analyst's characterisation task for the candidate under review."""
    if step.task != "analyst.characterise":
        raise RuntimeError(f"analyst node reached with {step.task}")
    await _characterise(context, step)


async def _characterise(context: UnitContext, step: Step) -> None:
    active = context.active
    candidate = context.candidate_for(step)
    payload = {
        **role_payload(active.session, context.call_unit),
        "worksheet": context.worksheet.render("analyst"),
        "candidate_locator": candidate.locator,
        "candidate_text": await _provision_text(context, candidate.locator),
    }
    outcome = await run_task(
        active,
        role="analyst",
        task="analyst.characterise",
        unit_id=context.call_unit.unit_id,
        payload=payload,
        tools=tools_for("analyst.characterise"),
        handle_tool=partial(_handle_characterise, context, candidate),
        max_turns=TASK_MAX_TURNS["analyst.characterise"],
        allow_uncommitted=True,
    )
    if outcome.committed:
        return
    context.worksheet.add_character(
        candidate_id=candidate.id,
        character=ProvisionCharacter(
            kind=ProvisionKind.UNDETERMINED,
            undetermined_reason="unusable_model_answer",
        ),
        author="system",
    )
    active.defaulted_characterisations += 1
    logger.warning(
        "defaulted characterisation run=%s unit=%s candidate=%s",
        active.run_id,
        context.call_unit.unit_id,
        candidate.id,
    )


async def _handle_characterise(
    context: UnitContext,
    candidate: CandidateEntry,
    invocation: ToolInvocation,
    attempts: tuple[AttemptTelemetry, ...],
) -> ToolResult:
    parsed = parse_tool(
        tools_for("analyst.characterise"), invocation.name, invocation.arguments
    )
    if isinstance(parsed, dict):
        return refused(parsed, invocation.name, context=context)
    if isinstance(parsed, ReadProvisionArgs):
        # The analyst holds no search tool, and a read outside the ledger would be
        # one: it would let the role reach a provision nobody retrieved for this
        # unit and then cite it. Reading is therefore confined to what the
        # researcher already brought back.
        if parsed.locator not in context.locator_ledger:
            return refused(
                refusal("locator_not_in_provenance_ledger"),
                invocation.name,
                context=context,
            )
        return ToolResult(await run_read(context, "analyst", parsed.locator))
    if isinstance(parsed, PostCharacterArgs):
        return await _post_character(context, candidate, parsed, invocation.name)
    return refused(refusal("unknown_tool"), invocation.name, context=context)


async def _post_character(
    context: UnitContext,
    candidate: CandidateEntry,
    args: PostCharacterArgs,
    tool_name: str,
) -> ToolResult:
    # Existing in the corpus is not the same as having been found for this unit:
    # a locator nobody retrieved or opened for it is remembered, not evidence, and
    # it would commit silently and correctly with the wrong provision behind it.
    # The ledger the analyst is held to is the one the researcher filled, so the
    # pair shares a provenance record rather than each keeping its own.
    for item in args.evidence:
        if await read_provision(context.active, item.locator) is None:
            return refused(
                refusal("evidence_locator_unknown"), tool_name, context=context
            )
        if item.locator not in context.locator_ledger:
            return refused(
                refusal("locator_not_in_provenance_ledger"),
                tool_name,
                context=context,
            )
    context.worksheet.add_character(candidate_id=candidate.id, character=args)
    return ToolResult({"recorded": True}, committed=True)


async def _provision_text(context: UnitContext, locator: str) -> str:
    record = await read_provision(context.active, locator)
    if record is None:
        raise RuntimeError("worksheet candidate missing from corpus")
    return record.text
