"""What a run is asked to do, over what, and what it hands back.

The document session holds the ingested contract for as long as the retention
window allows; the call units are the slices of it one arm analyses; the request
and result records are the run's input and output.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Literal
from uuid import UUID

from contract_analyzer.agents.parity import ParityBundle
from contract_analyzer.agents.schema import FrozenModel
from contract_analyzer.domain import ArmCode, DocumentPayload, ReferenceRecord, Unit
from contract_analyzer.storage import FindingRecord, InteractionKind
from contract_analyzer.structure import context_for


@dataclass(frozen=True)
class CallUnit:
    unit_id: str
    context_unit_ids: tuple[str, ...]
    is_whole_document: bool


@dataclass
class DocumentSession:
    document_id: UUID
    payload: DocumentPayload
    units: tuple[Unit, ...]
    references: tuple[ReferenceRecord, ...]
    units_by_id: dict[str, Unit] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.units_by_id = {unit.id: unit for unit in self.units}

    @property
    def whole_document_id(self) -> str:
        return f"whole:{self.payload.content_hash}"

    @property
    def input_hash(self) -> str:
        """The parity dimension identifying this document and its segmentation.

        Derived on every read rather than stored: the units are replaced when
        retention purges their text, and a stored copy would then describe a
        segmentation the session no longer has. Twenty reads of a four-unit
        document cost about twenty microseconds, which buys nothing worth an
        invariant on a value the run record carries.
        """
        unit_fingerprint = "|".join(
            f"{unit.id}:{unit.anchor.start_offset}:{unit.anchor.end_offset}"
            for unit in self.units
        )
        material = (
            f"{self.payload.content_hash}|{self.payload.read_mode}|{unit_fingerprint}"
        )
        return hashlib.sha256(material.encode()).hexdigest()


@dataclass(frozen=True)
class RunRequest:
    document_id: UUID
    arm: ArmCode
    measurement_valid: bool = True
    wall_budget_seconds: float = 900.0
    # Chosen by measurement; protocol.json run_parameters owns the evidence.
    concurrency: int = 8
    parameters: Mapping[str, str | int | float | bool | None] = field(
        default_factory=lambda: {
            "temperature": 0.0,
            "reasoning_effort": "low",
        }
    )
    clock: Callable[[], float] | None = field(default=None, compare=False, repr=False)
    parent_run_id: UUID | None = None
    interaction: InteractionKind | None = None
    unit_id: str | None = None
    user_note: str | None = field(default=None, compare=False, repr=False)
    user_note_finding_id: UUID | None = None


def with_model_safeguards(request: RunRequest) -> RunRequest:
    """Reject request parameters that would silently break the output cap."""
    parameters = dict(request.parameters)
    if "max_tokens" in parameters:
        raise ValueError(
            "max_tokens does not bound reasoning tokens; use max_completion_tokens"
        )
    output_cap = parameters.get("max_completion_tokens")
    if output_cap is not None and (
        not isinstance(output_cap, int)
        or isinstance(output_cap, bool)
        or output_cap <= 0
    ):
        raise ValueError("max_completion_tokens must be a positive integer")
    parameters.setdefault("reasoning_effort", "low")
    return replace(request, parameters=parameters)


@dataclass(frozen=True)
class RunResult:
    run_id: UUID
    status: Literal["completed", "failed", "cancelled"]
    parity_bundle: ParityBundle
    call_units: list[str]
    context_unit_ids: list[str]
    findings: list[FindingRecord]
    interruption: bool
    unprocessed_count: int
    tokens_used: int
    error_code: str | None = None


@dataclass(frozen=True)
class PreparedRun:
    run_id: UUID
    call_units: list[CallUnit]
    parity: ParityBundle
    elapsed_ms: Callable[[], float]


@dataclass(frozen=True)
class ChatMessage:
    role: Literal["user", "assistant"]
    content: str


@dataclass(frozen=True)
class RouteRequest:
    """One reader message awaiting a reading of what it asks for.

    The recent turns travel with it because a composer invites follow-ups, and
    "przeanalizuj ją jeszcze raz" has no referent on its own. Without them the
    router had to guess a target or fall back to a question about the whole run.
    """

    run_id: UUID
    message: str
    history: tuple[ChatMessage, ...] = ()


@dataclass(frozen=True)
class ChatRequest:
    run_id: UUID
    question: str
    selected_finding_ids: tuple[UUID, ...]
    history: tuple[ChatMessage, ...] = ()


class ChatResponse(FrozenModel):
    answer: str
    cited_finding_ids: tuple[str, ...] = ()
    interaction_run_id: UUID
    corpus_consulted: bool


def build_call_units(session: DocumentSession, arm: ArmCode) -> list[CallUnit]:
    """Call units for one arm: whole document (OFF), units (MID), units+context (ON).

    ON may legitimately produce no context at all -- a document segmented into
    windows has no resolvable internal references, and a single-unit document has
    nothing to refer to. Such a run is byte-identical to MID, which the evaluation
    protocol relies on as a noise floor (cohort zero_edge_holdout). The run record
    therefore counts the context edges this arm actually received, so an analysis
    can tell the two situations apart without the arm refusing to run.
    """
    if arm is ArmCode.OFF:
        return [
            CallUnit(
                unit_id=session.whole_document_id,
                context_unit_ids=(),
                is_whole_document=True,
            )
        ]
    units: list[CallUnit] = []
    unit_list = list(session.units)
    reference_list = list(session.references)
    for unit in session.units:
        if arm is ArmCode.ON:
            context_ids = tuple(
                target.id for target in context_for(unit.id, unit_list, reference_list)
            )
        else:
            context_ids = ()
        units.append(
            CallUnit(
                unit_id=unit.id,
                context_unit_ids=context_ids,
                is_whole_document=False,
            )
        )
    return units


def purge_session_plaintext(session: DocumentSession) -> None:
    """Blank every text field the session holds, in place."""
    session.payload = session.payload.model_copy(update={"text": ""})
    session.units = tuple(
        unit.model_copy(update={"text": ""}) for unit in session.units
    )
    session.units_by_id = {unit.id: unit for unit in session.units}
    session.references = tuple(
        reference.model_copy(update={"raw_text": ""})
        for reference in session.references
    )


def context_ids(call_units: Sequence[CallUnit]) -> list[str]:
    """Every context unit the run received, once, in first-seen order."""
    return list(
        dict.fromkeys(
            unit_id
            for call_unit in call_units
            for unit_id in call_unit.context_unit_ids
        )
    )


def unit_by_id(session: DocumentSession, unit_id: str) -> Unit:
    return session.units_by_id[unit_id]


def query_text(session: DocumentSession, call_unit: CallUnit) -> str:
    if call_unit.is_whole_document:
        return session.payload.text
    return unit_by_id(session, call_unit.unit_id).text


def role_payload(session: DocumentSession, call_unit: CallUnit) -> dict[str, object]:
    """What every role is shown of the contract, and only that.

    The arm is expressed here and nowhere else: OFF sees the whole document as one
    unit, MID sees the unit, ON sees the unit plus the units it refers to.
    """
    payload: dict[str, object] = {
        "unit_id": call_unit.unit_id,
        "unit_text": query_text(session, call_unit),
    }
    if call_unit.context_unit_ids:
        payload["context_units"] = [
            {
                "unit_id": context_id,
                "text": unit_by_id(session, context_id).text,
            }
            for context_id in call_unit.context_unit_ids
        ]
    return payload
