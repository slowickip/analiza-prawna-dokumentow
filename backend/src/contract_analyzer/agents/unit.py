"""The body of the unit cycle: act, schedule, and close the unit.

After any of the three roles acts, the same worksheet scheduler chooses the next
one. When nothing is outstanding the unit closes, and if no candidate produced a
finding the unit's own result is recorded there.
"""

from __future__ import annotations

from contract_analyzer.agents import analyst, researcher, verifier
from contract_analyzer.agents.errors import RunPipelineError
from contract_analyzer.agents.findings import store_unit_finding
from contract_analyzer.agents.scheduler import MAX_STEPS_PER_CANDIDATE, next_step
from contract_analyzer.agents.state import UnitContext
from contract_analyzer.domain import DecisionFacts, resolve_finding


async def run(context: UnitContext) -> list[str]:
    """Drive one worksheet through its bounded role cycle."""
    finding_ids: list[str] = []
    steps = 0
    while (step := next_step(context.worksheet)) is not None:
        if steps >= 1 + len(context.worksheet.candidates()) * MAX_STEPS_PER_CANDIDATE:
            # The published error name predates the plain loop.
            raise RunPipelineError("unit_graph_recursion_exceeded")
        context.pending_step = step
        if step.role == "researcher":
            await researcher.act(context, step)
        elif step.role == "analyst":
            await analyst.act(context, step)
        else:
            finding_ids.extend(await verifier.act(context, step))
        steps += 1
    context.pending_step = None
    finding_ids.extend(finalise(context))
    return finding_ids


def finalise(context: UnitContext) -> list[str]:
    """The unit's own finding, when no candidate produced one.

    Zero candidates resolves to no basis found; candidates that were all found
    irrelevant resolve to no relation. Both are results about the clause, so the
    unit is never left without one.
    """
    if context.findings:
        return []
    finding = resolve_finding(
        DecisionFacts(
            candidates_cleared=len(context.worksheet.candidates()),
            relevant=False,
        )
    )
    record = store_unit_finding(context.active, context.call_unit.unit_id, finding)
    context.findings.append(record)
    return [str(record.id)]
