"""Read-only grounding helpers offered to the explainer."""

from __future__ import annotations

import asyncio
from uuid import UUID

from contract_analyzer.agents.errors import RunPipelineError, first_leaf
from contract_analyzer.agents.retrieval import read_provision as bounded_read
from contract_analyzer.agents.runtime import GraphRuntime
from contract_analyzer.agents.session import (
    CallUnit,
    DocumentSession,
    query_text,
)
from contract_analyzer.agents.state import ActiveRun
from contract_analyzer.agents.turns import refusal
from contract_analyzer.corpus import LegalUnit
from contract_analyzer.storage import FindingRecord


def read_finding(
    value: str, selected: list[FindingRecord], runtime: GraphRuntime
) -> dict[str, object]:
    """Return one selected finding and its retained worksheet entries."""
    try:
        finding_id = UUID(value)
    except ValueError:
        return refusal("unknown_finding", finding_id=value)
    finding = next((item for item in selected if item.id == finding_id), None)
    if finding is None:
        return refusal("unknown_finding", finding_id=value)
    worksheet = runtime.worksheet_for(finding.run_id, finding.unit_id)
    return {
        "finding_id": str(finding.id),
        "code": finding.code.value,
        "legal_locators": list(finding.legal_locators),
        "unit_id": finding.unit_id,
        "uncertain_cause": (
            finding.uncertain_cause.value if finding.uncertain_cause else None
        ),
        "worksheet_entries": (
            [entry.model_dump(mode="json") for entry in worksheet.entries]
            if worksheet is not None
            else None
        ),
    }


def read_unit(
    unit_id: str,
    session: DocumentSession,
    call_units_by_id: dict[str, CallUnit],
    selected: list[FindingRecord],
) -> dict[str, object]:
    """Return one call unit, of those the selected findings sit on.

    The chat explains findings the reader picked. A clause nobody picked is the
    subject of an analyse interaction, which produces its own run and its own
    findings, so answering about it here would route analysis through the one
    endpoint that is documented not to perform any. The refusal also says no more
    about the document than the reader has already selected.
    """
    if unit_id not in {finding.unit_id for finding in selected}:
        return refusal("unknown_unit", unit_id=unit_id)
    call_unit = call_units_by_id.get(unit_id)
    if call_unit is not None:
        return {"unit_id": unit_id, "text": query_text(session, call_unit)}
    return refusal("unknown_unit", unit_id=unit_id)


async def read_provision(active: ActiveRun, locator: str) -> dict[str, object]:
    """Return one provision from the frozen corpus, on the run's own budget.

    The corpus client is synchronous and speaks over the network, so a reader's
    question would otherwise stall the event loop and run past the ask's wall
    deadline. This is the same bounded call the measured roles make.
    """
    provision = await bounded_read(active, locator)
    if provision is None:
        return refusal("unknown_locator", locator=locator)
    active.provision_reads += 1
    return {"locator": provision.locator, "text": provision.text}


def grounding(
    finding: FindingRecord,
    session: DocumentSession,
    call_units_by_id: dict[str, CallUnit],
) -> dict[str, object]:
    """Build the unchanged eager grounding shape for one selected finding."""
    entry: dict[str, object] = {
        "id": str(finding.id),
        "code": finding.code.value,
        "unit_id": finding.unit_id,
        "anchor": {
            "start_offset": finding.start_offset,
            "end_offset": finding.end_offset,
            "page": finding.page,
            **({"bbox": list(finding.bbox)} if finding.bbox is not None else {}),
        },
        "legal_locators": list(finding.legal_locators),
    }
    if finding.uncertain_cause is not None:
        entry["uncertain_cause"] = finding.uncertain_cause.value
    call_unit = call_units_by_id.get(finding.unit_id)
    if call_unit is not None:
        entry["unit_text"] = query_text(session, call_unit)
    return entry


async def provisions(
    active: ActiveRun, selected: list[FindingRecord]
) -> list[dict[str, str]]:
    """Build the eager provision grounding for all selected findings.

    A cited locator the corpus no longer holds ends the ask rather than quietly
    grounding the answer in less than the finding claims; it means the deployment
    reads a different snapshot from the one that produced the finding.
    """
    locators = sorted({item for finding in selected for item in finding.legal_locators})
    tasks: list[asyncio.Task[LegalUnit | None]] = []
    try:
        async with asyncio.TaskGroup() as group:
            tasks.extend(
                group.create_task(bounded_read(active, locator)) for locator in locators
            )
    except BaseExceptionGroup as error:
        raise first_leaf(error) from None
    provision_units = [task.result() for task in tasks]
    grounded: list[dict[str, str]] = []
    for locator, provision in zip(locators, provision_units, strict=True):
        if provision is None:
            raise RunPipelineError("evidence_locator_unknown")
        grounded.append({"locator": locator, "text": provision.text})
    return grounded
