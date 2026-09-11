"""Run response builders."""

from __future__ import annotations

from collections.abc import Sequence
from uuid import UUID

from contract_analyzer.api.errors import _message_for
from contract_analyzer.api.schemas import (
    AttemptTelemetry,
    CostRecord,
    Error,
    Finding,
    LegalBasisReference,
    Prominence,
    Run,
    RunMetrics,
    SynthesisGroup,
)
from contract_analyzer.api.state import _AppState, _content_available
from contract_analyzer.domain import FindingCode, SourceAnchor
from contract_analyzer.storage import AttemptRecord, FindingRecord, RunRecord
from contract_analyzer.structure.blocks import block_for_offset

_PROMINENCE: dict[FindingCode, Prominence] = {
    FindingCode.CONTRADICTORY: Prominence.CRITICAL,
    FindingCode.BASIS_NOT_IN_FORCE: Prominence.CRITICAL,
    FindingCode.PERMISSIBLE_DEPARTURE: Prominence.WARNING,
    FindingCode.UNCERTAIN: Prominence.WARNING,
    FindingCode.UNIT_NOT_ADJUDICABLE: Prominence.WARNING,
    FindingCode.NOT_PROCESSED: Prominence.WARNING,
    FindingCode.CONSISTENT: Prominence.NEUTRAL,
    FindingCode.NO_RELATION: Prominence.NEUTRAL,
    FindingCode.NO_BASIS_FOUND: Prominence.NEUTRAL,
}


def _build_finding(
    record: FindingRecord, anchors: Sequence[SourceAnchor] = ()
) -> Finding:
    has_anchor = record.start_offset is not None and record.end_offset is not None
    anchor = None
    if has_anchor:
        anchor = SourceAnchor(
            start_offset=record.start_offset,  # type: ignore[arg-type]
            end_offset=record.end_offset,  # type: ignore[arg-type]
            page=record.page,
            bbox=record.bbox,
        )
    return Finding(
        id=str(record.id),
        unit_id=record.unit_id,
        code=record.code,
        prominence=_PROMINENCE[record.code],
        uncertain_cause=record.uncertain_cause,
        raw_confidence=record.raw_confidence,
        anchor=anchor,
        anchor_resolved=has_anchor,
        # Where the reader sees it. Resolved anchors only: a finding whose quote
        # never resolved has no position to give, and inventing one is what the
        # browser used to do by falling back to its call unit.
        block_id=(
            block_for_offset(anchors, record.start_offset) if has_anchor else None
        ),
        quote_resolution=record.quote_resolution,
        basis=(
            LegalBasisReference.model_validate(record.basis, from_attributes=True)
            if record.basis is not None
            else None
        ),
        legal_locators=list(record.legal_locators) if record.legal_locators else None,
    )


def _build_metrics(
    findings: list[FindingRecord],
    attempts: list[AttemptRecord],
    record: RunRecord,
) -> RunMetrics:
    input_tokens = sum(a.input_tokens for a in attempts)
    output_tokens = sum(a.output_tokens for a in attempts)
    units_with_finding = len(
        {
            finding.unit_id
            for finding in findings
            if finding.code != FindingCode.NOT_PROCESSED
        }
    )
    # Derived from the recorded call-unit count, not from the findings that happen to
    # exist: a unit killed mid-fan-out writes no finding at all, so counting findings
    # drops it from the numerator AND the denominator and reports the run as complete.
    if record.call_unit_count is not None:
        units_total = record.call_unit_count
        units_not_processed = units_total - units_with_finding
    else:
        units_total = len({finding.unit_id for finding in findings})
        units_not_processed = len(
            {
                finding.unit_id
                for finding in findings
                if finding.code == FindingCode.NOT_PROCESSED
            }
        )
    elapsed_ms = record.elapsed_ms
    if elapsed_ms is None:
        raise ValueError("cannot build metrics without recorded elapsed_ms")
    api_cost = CostRecord.model_validate(record.cost, from_attributes=True)
    api_attempts = [
        AttemptTelemetry.model_validate(a, from_attributes=True) for a in attempts
    ]
    return RunMetrics(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        elapsed_ms=elapsed_ms,
        units_total=units_total,
        units_with_finding=units_with_finding,
        units_not_processed=units_not_processed,
        context_edge_count=record.context_edge_count,
        finder_tool_turns=record.finder_tool_turns,
        finder_search_calls=record.finder_search_calls,
        retrieval_cache_hits=record.retrieval_cache_hits,
        finder_budget_exhausted_units=record.finder_budget_exhausted_units,
        verifier_tool_turns=record.verifier_tool_turns,
        provision_reads=record.provision_reads,
        defaulted_characterisations=record.defaulted_characterisations,
        cost=api_cost,
        attempts=api_attempts,
    )


def _build_run(record: RunRecord, state: _AppState) -> Run:
    findings = state.services.metadata.list_findings(record.id)
    attempts = state.services.metadata.list_attempts(record.id)
    content = _content_available(state, record.document_id)
    # Placement is derived from the in-memory anchors, so it is present exactly
    # while the document is, and null once the retention window closes.
    session = state.services.sessions.get(record.document_id)
    anchors = session.payload.anchors if session is not None and content else ()
    interrupted = record.interruption_reason is not None or any(
        f.code == FindingCode.NOT_PROCESSED for f in findings
    )
    metrics = (
        _build_metrics(findings, attempts, record)
        if record.status != "running" and record.elapsed_ms is not None
        else None
    )
    error = (
        Error(
            code=record.error_code,
            message_pl=_message_for(record.error_code),
        )
        if record.error_code
        else None
    )
    return Run.model_validate(
        {
            **record.__dict__,
            "interrupted": interrupted,
            "content_available": content,
            "error": error,
            "findings": [_build_finding(f, anchors) for f in findings],
            "synthesis": _build_synthesis(state, record.id),
            "metrics": metrics,
        },
        from_attributes=True,
    )


def _build_synthesis(state: _AppState, run_id: UUID) -> list[SynthesisGroup] | None:
    grouping = state.runner.synthesis_for(run_id)
    if grouping is None:
        return None
    return [
        SynthesisGroup.model_validate(group, from_attributes=True)
        for group in grouping.groups
    ]
