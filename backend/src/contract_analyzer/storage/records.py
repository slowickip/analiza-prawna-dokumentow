"""Metadata records and validation helpers."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Literal
from uuid import UUID

from pydantic import TypeAdapter

from contract_analyzer.domain import (
    ArmCode,
    AttemptStatus,
    EmittedBasis,
    FindingCode,
    ParameterValue,
    QuoteResolution,
    UncertainCause,
)

InterruptionReason = Literal["wall_time"]

RunStatus = Literal["running", "completed", "failed", "cancelled"]
# "route" is the classification step a single reader message runs before the
# system knows which of the other three it is. It reads no corpus and changes
# no finding, but it spends a model call, so it owns a run record like any other.
InteractionKind = Literal["route", "ask", "contest", "analyse"]
EventKind = Literal["status", "counter", "finding", "worksheet"]
_CODE = re.compile(r"^[a-z][a-z0-9_]*$")
_PARAMETERS_ADAPTER = TypeAdapter(dict[str, ParameterValue])


@dataclass(frozen=True)
class RunTotals:
    error_code: str | None = None
    error_detail: str | None = None
    cost: CostRecord | None = None
    elapsed_ms: float | None = None
    interruption_reason: InterruptionReason | None = None
    source_text: str | None = None
    returned_model: str | None = None
    context_edge_count: int | None = None
    call_unit_count: int | None = None
    finder_tool_turns: int | None = None
    finder_search_calls: int | None = None
    retrieval_cache_hits: int | None = None
    finder_budget_exhausted_units: int | None = None
    verifier_tool_turns: int | None = None
    provision_reads: int | None = None
    defaulted_characterisations: int | None = None


class SchemaVersionError(Exception):
    pass


@dataclass(frozen=True)
class CostRecord:
    monetary_cost_microunits: int | None = None
    price_table_date: date | None = None
    price_table_hash: str | None = None
    unknown_reason: str | None = None

    def __post_init__(self) -> None:
        if self.monetary_cost_microunits is None:
            if not self.unknown_reason:
                raise ValueError("unknown cost requires unknown_reason")
            _require_code(self.unknown_reason, "unknown_reason")
            if self.price_table_date is not None or self.price_table_hash is not None:
                raise ValueError("unknown cost cannot carry a price table")
            return
        if self.monetary_cost_microunits < 0:
            raise ValueError("monetary cost cannot be negative")
        if self.price_table_date is None or not self.price_table_hash:
            raise ValueError("known cost requires a date-pinned price table")
        if self.unknown_reason is not None:
            raise ValueError("known cost cannot carry unknown_reason")

    @classmethod
    def unknown(cls, reason: str) -> CostRecord:
        return cls(unknown_reason=reason)

    @classmethod
    def known(
        cls,
        *,
        monetary_cost_microunits: int,
        price_table_date: date,
        price_table_hash: str,
    ) -> CostRecord:
        return cls(
            monetary_cost_microunits=monetary_cost_microunits,
            price_table_date=price_table_date,
            price_table_hash=price_table_hash,
        )


@dataclass(frozen=True)
class RunRecord:
    id: UUID
    document_id: UUID
    arm: ArmCode
    input_hash: str
    config_version: str
    prompt_bundle_version: str
    corpus_snapshot_id: str
    tool_bundle_version: str
    requested_model: str
    parameters: Mapping[str, ParameterValue]
    retry_policy: str
    concurrency: int
    wall_budget_seconds: float
    measurement_valid: bool
    cost: CostRecord
    parent_run_id: UUID | None = None
    interaction: InteractionKind | None = None
    status: RunStatus = "running"
    graph_topology_version: str | None = None
    returned_model: str | None = None
    error_code: str | None = None
    source_text: str | None = None
    error_detail: str | None = None
    elapsed_ms: float | None = None
    interruption_reason: InterruptionReason | None = None
    # Store-assigned timestamps, exposed by the API.
    created_at: str | None = None
    finished_at: str | None = None
    # Resolved context edges supplied to this run; zero is valid for ON.
    context_edge_count: int | None = None
    # Persist the denominator, including units that fail before emitting findings.
    call_unit_count: int | None = None
    # Search effort summed over call units.
    finder_tool_turns: int | None = None
    finder_search_calls: int | None = None
    retrieval_cache_hits: int | None = None
    # Units stopped by the finder cap, distinct from searches finding no basis.
    finder_budget_exhausted_units: int | None = None
    # Role activity and defaulted characterisations, summed over call units.
    verifier_tool_turns: int | None = None
    provision_reads: int | None = None
    defaulted_characterisations: int | None = None


@dataclass(frozen=True)
class AttemptRecord:
    id: UUID
    run_id: UUID
    requested_model: str
    returned_model: str | None
    prompt_version: str
    temperature: float
    parameters: Mapping[str, ParameterValue]
    input_tokens: int
    output_tokens: int
    latency_ms: float
    status: AttemptStatus
    retry_number: int
    error_code: str | None
    # Which call unit spent this attempt, where one did. Run-level tasks -- the
    # synthesis, the explainer and its delegated question -- carry None. It is an
    # identifier, so it stays inside the retention boundary the metadata store
    # keeps, and it is what lets a run report its critical path instead of an
    # average over units.
    unit_id: str | None = None
    prompt_hash: str = ""
    response_hash: str | None = None
    prompt: str | None = None
    response: str | None = None
    error_detail: str | None = None


@dataclass(frozen=True)
class FindingRecord:
    id: UUID
    run_id: UUID
    unit_id: str
    code: FindingCode
    uncertain_cause: UncertainCause | None = None
    raw_confidence: float | None = None
    start_offset: int | None = None
    end_offset: int | None = None
    page: int | None = None
    bbox: tuple[float, float, float, float] | None = None
    legal_locators: tuple[str, ...] = ()
    basis: EmittedBasis | None = None
    quote_resolution: QuoteResolution | None = None
    excerpt: str | None = None
    synthesis_prose: str | None = None
    error_detail: str | None = None


def _require_code(value: str, field: str) -> None:
    if _CODE.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lower-snake-case code")


def _hash_optional(value: str | None) -> str | None:
    if value is None:
        return None
    return hashlib.sha256(value.encode()).hexdigest()


def _parameters_json(parameters: Mapping[str, ParameterValue]) -> str:
    for key in parameters:
        _require_code(key, "parameter name")
    return json.dumps(dict(parameters), sort_keys=True, separators=(",", ":"))


def _parameters(value: str) -> dict[str, ParameterValue]:
    return _PARAMETERS_ADAPTER.validate_json(value)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _validate_run(run: RunRecord) -> None:
    if run.concurrency < 1 or run.wall_budget_seconds <= 0:
        raise ValueError("invalid run bounds")
    for key in run.parameters:
        _require_code(key, "parameter name")
    if run.error_code is not None:
        _require_code(run.error_code, "error_code")
    if run.interruption_reason not in (None, "wall_time"):
        raise ValueError("invalid interruption reason")
    if (run.parent_run_id is None) != (run.interaction is None):
        raise ValueError("interactive run requires both parent_run_id and interaction")
    if run.interaction is not None and run.measurement_valid:
        raise ValueError("interactive run cannot be measurement-valid")


def _validate_attempt(attempt: AttemptRecord) -> None:
    if (
        attempt.input_tokens < 0
        or attempt.output_tokens < 0
        or attempt.latency_ms < 0
        or attempt.retry_number < 0
    ):
        raise ValueError("attempt counters cannot be negative")
    for key in attempt.parameters:
        _require_code(key, "parameter name")
    if attempt.error_code is not None:
        _require_code(attempt.error_code, "error_code")
