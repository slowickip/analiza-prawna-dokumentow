"""How a run ends: the terminal metadata write and the result it reports.

Kept apart from the runner because ending a run is its own decision. Two of the
three endings are not failures of the artefact -- a budget that ran out leaves
real findings behind, and only an unusable answer makes the run itself void -- and
the record has to say which happened.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Literal

from contract_analyzer.agents.runtime import GraphRuntime
from contract_analyzer.agents.services import AnalysisServices
from contract_analyzer.agents.session import PreparedRun, RunResult, context_ids
from contract_analyzer.domain import FindingCode
from contract_analyzer.storage import InterruptionReason, RunEvent, RunTotals

logger = logging.getLogger(__name__)


def completed(
    services: AnalysisServices,
    runtime: GraphRuntime,
    prepared: PreparedRun,
    *,
    interruption: bool,
    unprocessed_count: int,
) -> RunResult:
    """Record and report a run that reached the end of its analysis."""
    run_id = prepared.run_id
    findings = services.metadata.list_findings(run_id)
    finish_run(
        services,
        runtime,
        prepared,
        "completed",
        interruption_reason=(
            runtime.interruption_reason_for(run_id) if interruption else None
        ),
    )
    result = RunResult(
        run_id=run_id,
        status="completed",
        parity_bundle=replace(
            prepared.parity, returned_model=runtime.returned_model_for(run_id)
        ),
        call_units=[unit.unit_id for unit in prepared.call_units],
        context_unit_ids=context_ids(prepared.call_units),
        findings=findings,
        interruption=interruption,
        unprocessed_count=unprocessed_count,
        tokens_used=runtime.tokens_for(run_id),
    )
    runtime.retire_run(run_id)
    return result


def failed(
    services: AnalysisServices,
    runtime: GraphRuntime,
    prepared: PreparedRun,
    error_code: str,
) -> RunResult:
    """Record and report a run that died on an unusable answer or a dependency.

    The units that already produced a finding keep it; the rest are counted as
    unprocessed, because a run that failed halfway is a partial measurement rather
    than an empty one.
    """
    run_id = prepared.run_id
    findings = services.metadata.list_findings(run_id)
    terminal_unit_ids = {
        finding.unit_id
        for finding in findings
        if finding.code is not FindingCode.NOT_PROCESSED
    }
    unprocessed_count = sum(
        unit.unit_id not in terminal_unit_ids for unit in prepared.call_units
    )
    interruption_reason = runtime.interruption_reason_for(run_id)
    finish_run(
        services,
        runtime,
        prepared,
        "failed",
        error_code=error_code,
        interruption_reason=interruption_reason,
    )
    result = RunResult(
        run_id=run_id,
        status="failed",
        parity_bundle=prepared.parity,
        call_units=[unit.unit_id for unit in prepared.call_units],
        context_unit_ids=context_ids(prepared.call_units),
        findings=findings,
        interruption=interruption_reason is not None or unprocessed_count > 0,
        unprocessed_count=unprocessed_count,
        tokens_used=runtime.tokens_for(run_id),
        error_code=error_code,
    )
    runtime.retire_run(run_id)
    return result


def finish_run(
    services: AnalysisServices,
    runtime: GraphRuntime,
    prepared: PreparedRun,
    status: Literal["completed", "failed"],
    *,
    error_code: str | None = None,
    interruption_reason: InterruptionReason | None = None,
    analysis_counters: bool = True,
) -> float:
    run_id = prepared.run_id
    spend = runtime.spend_for(run_id)
    elapsed = prepared.elapsed_ms()
    services.metadata.finish_run(
        run_id,
        status,
        totals=RunTotals(
            error_code=error_code,
            elapsed_ms=elapsed,
            interruption_reason=interruption_reason,
            finder_tool_turns=spend.finder_tool_turns,
            finder_search_calls=spend.finder_search_calls,
            retrieval_cache_hits=spend.retrieval_cache_hits,
            finder_budget_exhausted_units=spend.finder_budget_exhausted_units,
            verifier_tool_turns=spend.verifier_tool_turns,
            provision_reads=spend.provision_reads,
            defaulted_characterisations=(
                spend.defaulted_characterisations if analysis_counters else None
            ),
        ),
    )
    if services.events is not None:
        services.events.publish(run_id, RunEvent.status_changed(status))
    logger.info(
        "run finished run=%s status=%s elapsed_ms=%.0f tokens=%d spend=%s",
        run_id,
        status,
        elapsed,
        runtime.tokens_for(run_id),
        spend,
    )
    return elapsed
