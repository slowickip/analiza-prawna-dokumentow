"""The synthesizer: groups the findings of a whole run, and nothing else.

A real agent with two tools: it lists the findings this run already stored and
commits one grouping over them. It receives identifiers, codes and locators --
never document or provision text, never a corpus search -- so a grouping it
produces can only rearrange what the units already decided.
"""

from __future__ import annotations

import logging
from uuid import UUID

from contract_analyzer.agents.errors import BudgetExhausted
from contract_analyzer.agents.retention import TextRetention
from contract_analyzer.agents.scheduler import TASK_MAX_TURNS
from contract_analyzer.agents.schema import FrozenModel
from contract_analyzer.agents.state import ActiveRun
from contract_analyzer.agents.tool_args import GroupFindingsArgs, ListFindingsArgs
from contract_analyzer.agents.tools import TASK_TOOLS, parse_tool
from contract_analyzer.agents.turns import ToolResult, refusal, refused, run_task
from contract_analyzer.model import AttemptTelemetry, ToolInvocation

logger = logging.getLogger(__name__)


class SynthesisGroup(FrozenModel):
    title: str
    summary: str
    finding_ids: tuple[str, ...]


class SynthesisResponse(FrozenModel):
    groups: tuple[SynthesisGroup, ...] = ()


def synthesis_payload(
    active: ActiveRun, finding_ids: list[UUID]
) -> tuple[dict[str, object], list[UUID]]:
    """The findings this run may group, under the role's own numbering.

    One entry per finding, not two parallel lists: a flat locator list cannot be
    attributed back to a finding, so the role would have nothing to group on but
    opaque identifiers. The result code travels with it because the grouping is
    read by the user, and a group is only meaningful if the role can see what it
    groups.

    The reference is this list's own numbering rather than the finding's UUID.
    The role has to hand the reference back to say what it grouped, and a run
    was lost to a model that mistyped one character of one 36-character
    identifier out of twenty-one. A number it can carry accurately is the whole
    requirement; the UUID never meant anything to it.
    """
    emitted = set(finding_ids)
    ordered = [
        record
        for record in active.services.metadata.list_findings(active.run_id)
        if record.id in emitted
    ]
    return {
        "findings": [
            {
                "ref": str(position),
                "code": record.code.value,
                "legal_locators": list(record.legal_locators),
            }
            for position, record in enumerate(ordered, start=1)
        ]
    }, [record.id for record in ordered]


async def synthesise(
    active: ActiveRun, retention: TextRetention, finding_ids: list[UUID]
) -> bool:
    """Group this run's findings, or report that the budget ended first.

    Returns whether the run was interrupted rather than raising: a run that
    produced findings and ran out of budget before grouping them is a partial
    result, not a failure. A grouping the role never commits is the same
    ending: the findings stay, and no grouping is kept over them.
    """
    payload, ordered = synthesis_payload(active, finding_ids)

    async def handle(
        invocation: ToolInvocation, attempts: tuple[AttemptTelemetry, ...]
    ) -> ToolResult:
        del attempts
        parsed = parse_tool(
            TASK_TOOLS["synthesizer.group"], invocation.name, invocation.arguments
        )
        if isinstance(parsed, dict):
            return refused(parsed, invocation.name, active=active)
        if isinstance(parsed, ListFindingsArgs):
            return ToolResult(dict(payload))
        if isinstance(parsed, GroupFindingsArgs):
            return _commit_grouping(active, retention, parsed, ordered, invocation.name)
        return refused(refusal("unknown_tool"), invocation.name, active=active)

    try:
        await run_task(
            active,
            role="synthesizer",
            task="synthesizer.group",
            prompt_version="synthesise",
            payload=payload,
            tools=TASK_TOOLS["synthesizer.group"],
            handle_tool=handle,
            max_turns=TASK_MAX_TURNS["synthesizer.group"],
        )
    except BudgetExhausted:
        return True
    return active.budget.exhausted


def _commit_grouping(
    active: ActiveRun,
    retention: TextRetention,
    args: GroupFindingsArgs,
    ordered: list[UUID],
    tool_name: str,
) -> ToolResult:
    """Keep the grouping, resolved back to identifiers, if every reference is real.

    The role answers in the numbering it was given, so this is where a grouping
    becomes a thing the reader can open. A reference outside the list is the one
    fault left worth refusing: it would reach the reader as a finding that does
    not exist. Such a grouping is kept nowhere, and the role may correct it on
    its next turn.
    """
    resolved: list[SynthesisGroup] = []
    for group in args.groups:
        finding_ids: list[str] = []
        for reference in group.finding_ids:
            try:
                position = int(reference)
            except ValueError:
                return refused(
                    refusal("unknown_ref", reference=reference),
                    tool_name,
                    active=active,
                )
            if not 1 <= position <= len(ordered):
                return refused(
                    refusal("unknown_ref", reference=reference),
                    tool_name,
                    active=active,
                )
            finding_ids.append(str(ordered[position - 1]))
        resolved.append(
            SynthesisGroup(
                title=group.title,
                summary=group.summary,
                finding_ids=tuple(finding_ids),
            )
        )
    retention.keep_synthesis(
        run_id=active.run_id,
        document_id=active.session.document_id,
        payload=SynthesisResponse(groups=tuple(resolved))
        .model_dump_json()
        .encode("utf-8"),
    )
    return ToolResult({"recorded": True}, committed=True)


def synthesis_for(retention: TextRetention, run_id: UUID) -> SynthesisResponse | None:
    raw = retention.synthesis_bytes(run_id)
    return None if raw is None else SynthesisResponse.model_validate_json(raw)
