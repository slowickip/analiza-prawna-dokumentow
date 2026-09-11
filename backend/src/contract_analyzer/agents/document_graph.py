"""Run every document unit concurrently, then synthesize their findings."""

from __future__ import annotations

import asyncio

from contract_analyzer.agents.errors import first_leaf as _first_leaf
from contract_analyzer.agents.runtime import GraphRuntime, UnitOutcome
from contract_analyzer.agents.session import PreparedRun


async def run_document(
    runtime: GraphRuntime, prepared: PreparedRun
) -> tuple[bool, int]:
    """Fan out units with sibling-safe failure and return interruption totals."""
    tasks: list[asyncio.Task[UnitOutcome]] = []

    try:
        async with asyncio.TaskGroup() as group:
            tasks.extend(
                group.create_task(
                    runtime.process_unit(prepared.run_id, call_unit.unit_id)
                )
                for call_unit in prepared.call_units
            )
    except BaseExceptionGroup as error:
        raise _first_leaf(error) from None

    processed_ids: list[str] = []
    unprocessed_ids: list[str] = []
    finding_ids: list[str] = []
    interruption = False
    for task in tasks:
        result = task.result()
        processed_ids.extend(result.processed_unit_ids)
        unprocessed_ids.extend(result.unprocessed_unit_ids)
        finding_ids.extend(result.finding_ids)
        interruption = interruption or result.interruption

    active = runtime.active(prepared.run_id)
    completed_ids = set(processed_ids) | set(unprocessed_ids)
    if active.budget.exhausted:
        for call_unit in prepared.call_units:
            if call_unit.unit_id in completed_ids:
                continue
            finding = runtime.store_not_processed(prepared.run_id, call_unit.unit_id)
            unprocessed_ids.append(call_unit.unit_id)
            finding_ids.append(str(finding.id))
            interruption = True

    interruption = (
        await runtime.synthesize(prepared.run_id, finding_ids) or interruption
    )
    return interruption, len(unprocessed_ids)
