"""HTTP request and response schemas."""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum
from typing import Any, Literal, TypedDict, cast
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_serializer, model_validator

from contract_analyzer.agents.session import ChatResponse
from contract_analyzer.domain import (
    ArmCode,
    ConversionProvenance,
    FindingCode,
    ForceState,
    ProvisionKind,
    QuoteResolution,
    ReadMode,
    ReferenceRecord,
    SourceAnchor,
    UncertainCause,
    Unit,
    validate_force_record_pair,
)
from contract_analyzer.storage import InteractionKind, InterruptionReason, RunStatus


class Prominence(StrEnum):
    CRITICAL = "critical"
    WARNING = "warning"
    NEUTRAL = "neutral"


class Error(BaseModel):
    code: str
    message_pl: str
    detail: dict[str, Any] | None = None


class Block(BaseModel):
    id: str
    text: str
    anchor: SourceAnchor


class DocumentDescriptor(BaseModel):
    id: UUID
    content_hash: str
    read_mode: ReadMode | None = None
    conversion: ConversionProvenance | None = None
    unit_count: int
    expires_at: datetime | None = None


class DocumentContent(BaseModel):
    document_id: UUID
    read_mode: ReadMode | None = None
    blocks: list[Block]
    units: list[Unit]
    references: list[ReferenceRecord]


class LegalBasisReference(BaseModel):
    """The two force records as emitted, never collapsed into one."""

    provision_locator: str
    act_identifier: str
    act_force: ForceState
    provision_force: ForceState
    character_kind: ProvisionKind

    @model_validator(mode="after")
    def validate_force_records(self) -> LegalBasisReference:
        validate_force_record_pair(self.act_force, self.provision_force)
        return self


class Finding(BaseModel):
    id: str
    unit_id: str
    code: FindingCode
    prominence: Prominence
    uncertain_cause: UncertainCause | None = None
    raw_confidence: float | None = None
    anchor: SourceAnchor | None = None
    anchor_resolved: bool | None = None
    block_id: str | None = None
    quote_resolution: QuoteResolution | None = None
    basis: LegalBasisReference | None = None
    legal_locators: list[str] | None = None

    @model_serializer(mode="wrap")
    def _serialize_finding(self, handler: Any) -> dict[str, Any]:
        data = cast(dict[str, Any], handler(self))
        if not self.anchor_resolved:
            data.pop("anchor", None)
        return data


class CostRecord(BaseModel):
    monetary_cost_microunits: int | None
    unknown_reason: str | None
    price_table_date: date | None = None
    price_table_hash: str | None = None


class AttemptTelemetry(BaseModel):
    id: UUID
    requested_model: str
    returned_model: str | None = None
    prompt_version: str
    temperature: float
    parameters: dict[str, Any]
    input_tokens: int
    output_tokens: int
    latency_ms: float
    status: str
    retry_number: int
    error_code: str | None = None
    unit_id: str | None = None
    prompt_hash: str
    response_hash: str | None = None


class RunMetrics(BaseModel):
    input_tokens: int
    output_tokens: int
    elapsed_ms: float
    units_total: int
    units_with_finding: int
    units_not_processed: int
    context_edge_count: int | None
    finder_tool_turns: int | None = None
    finder_search_calls: int | None = None
    retrieval_cache_hits: int | None = None
    finder_budget_exhausted_units: int | None = None
    verifier_tool_turns: int | None = None
    provision_reads: int | None = None
    defaulted_characterisations: int | None = None
    cost: CostRecord
    attempts: list[AttemptTelemetry]


class SynthesisGroup(BaseModel):
    """One themed grouping of findings the synthesis role returned.

    finding_ids name findings on this same run: the graph rejects a grouping that
    cites anything else, so every identifier here resolves against findings.
    """

    title: str
    summary: str
    finding_ids: list[str]


class Run(BaseModel):
    id: UUID
    document_id: UUID
    arm: ArmCode
    status: RunStatus
    measurement_valid: bool
    parent_run_id: UUID | None = None
    interaction: InteractionKind | None = None
    interrupted: bool | None = None
    interruption_reason: InterruptionReason | None = None
    content_available: bool
    requested_model: str | None = None
    returned_model: str | None = None
    input_hash: str | None = None
    prompt_bundle_version: str | None = None
    corpus_snapshot_id: str | None = None
    config_version: str | None = None
    tool_bundle_version: str | None = None
    parameters: dict[str, Any] | None = None
    retry_policy: str | None = None
    concurrency: int | None = None
    wall_budget_seconds: float | None = None
    created_at: datetime | None = None
    finished_at: datetime | None = None
    error: Error | None = None
    findings: list[Finding]
    synthesis: list[SynthesisGroup] | None = None
    metrics: RunMetrics | None


class PublicConfigLimits(BaseModel):
    max_input_bytes: int
    max_pdf_pages: int
    content_ttl_seconds: int


class PublicConfig(BaseModel):
    model_request_id: str
    # Host of the endpoint that answers a model call. The provider is a parity
    # dimension and the identifier does not name it: two endpoints serve one
    # model differently. Host only, never the credential.
    model_endpoint: str
    prompt_bundle_version: str
    corpus_snapshot: str
    embedding_model: str
    # Fingerprint of declared embedding model, width, and source space.
    embedding_space_fingerprint: str
    # Whether every act was read from its declared source format.
    corpus_source_format: Literal["as_declared", "fallback", "unrecorded"]
    tool_bundle_version: str
    arms: list[ArmCode]
    measured_mode: bool
    evaluation_batch_open: bool = False
    limits: PublicConfigLimits


class DependencyStatus(BaseModel):
    name: str
    ok: bool
    detail: str | None = None


class Readiness(BaseModel):
    ready: bool
    dependencies: list[DependencyStatus]


class HealthLive(BaseModel):
    status: str


# ---------------------------------------------------------------------------
# Request bodies
# ---------------------------------------------------------------------------


class CreateRunBody(BaseModel):
    document_id: UUID
    arm: ArmCode


class ChatRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


class ChatHistoryItem(TypedDict):
    role: ChatRole
    content: str


class MessageBody(BaseModel):
    """One reader message, before the system knows what it asks for."""

    model_config = ConfigDict(strict=False)
    message: str = Field(min_length=1, max_length=2000)
    history: list[ChatHistoryItem] = Field(default_factory=list, max_length=20)


class MessageResponse(BaseModel):
    """Result of a reader message: in-place explanation or opened child run."""

    intent: Literal["ask", "contest", "analyse"]
    answer: ChatResponse | None = None
    run: Run | None = None

    @model_validator(mode="after")
    def exactly_one_result(self) -> MessageResponse:
        """Ensure exactly one of answer or run is present matching intent."""
        if (self.answer is None) == (self.run is None):
            raise ValueError(
                "a routed message answers in place or opens a child run, "
                "never both and never neither"
            )
        if (self.intent == "ask") != (self.answer is not None):
            raise ValueError(
                f"intent {self.intent!r} does not match the result it carries"
            )
        return self


class WorksheetUnit(BaseModel):
    """One retained worksheet in stored entry order."""

    unit_id: str
    entries: list[dict[str, Any]]


class WorksheetResponse(BaseModel):
    """All retained worksheets for one run."""

    run_id: UUID
    units: list[WorksheetUnit]
