"""Immutable domain records and their validation."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date
from typing import Literal
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from .enums import (
    FindingCode,
    ForceScope,
    ForceValue,
    InterpretiveMethod,
    ProvisionKind,
    ReadMode,
    ReferenceStatus,
    ReferenceType,
    SourceKind,
    UncertainCause,
)


def _non_blank(message: str) -> Callable[[str], str]:
    def check(value: str) -> str:
        if not value.strip():
            raise ValueError(message)
        return value

    return check


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True)


class SourceAnchor(FrozenModel):
    start_offset: int
    end_offset: int
    read_mode: ReadMode | None = None
    block_id: str | None = None
    line_start: int | None = None
    line_end: int | None = None
    page: int | None = None
    bbox: tuple[float, float, float, float] | None = None


class ConversionProvenance(FrozenModel):
    converter: str
    converter_version: str
    output_hash: str

    _validate_non_blank = field_validator("converter_version", "output_hash")(
        _non_blank("must not be blank")
    )


class DocumentPayload(FrozenModel):
    document_id: UUID
    text: str = Field(repr=False)
    anchors: tuple[SourceAnchor, ...]
    read_mode: ReadMode | None = None
    content_hash: str
    conversion: ConversionProvenance | None = None


class Unit(FrozenModel):
    id: str
    text: str = Field(repr=False)
    anchor: SourceAnchor


class ReferenceRecord(FrozenModel):
    citing_unit_id: str
    reference_type: ReferenceType
    status: ReferenceStatus
    raw_text: str = Field(repr=False)
    target_unit_id: str | None = None


class ForceState(FrozenModel):
    value: ForceValue
    scope: ForceScope
    snapshot_date: date
    source_locator: str

    _validate_non_blank = field_validator("source_locator")(
        _non_blank("source_locator must not be blank")
    )


class CharacterEvidence(FrozenModel):
    source_kind: SourceKind
    locator: str
    pinpoint: str
    interpretive_methods: tuple[InterpretiveMethod, ...] = Field(
        examples=[["linguistic"]]
    )
    rationale: str

    _validate_non_blank = field_validator("locator", "pinpoint", "rationale")(
        _non_blank("must not be empty")
    )

    @field_validator("interpretive_methods")
    @classmethod
    def reject_empty_methods(
        cls, value: tuple[InterpretiveMethod, ...]
    ) -> tuple[InterpretiveMethod, ...]:
        if not value:
            raise ValueError("interpretive_methods must not be empty")
        return value


class SemiImperativeDirection(FrozenModel):
    relation: Literal["more_favourable_to"] = "more_favourable_to"
    protected_party_role: str

    _validate_non_blank = field_validator("protected_party_role")(
        _non_blank("protected_party_role must not be empty")
    )


class ProvisionCharacter(FrozenModel):
    kind: ProvisionKind
    evidence: tuple[CharacterEvidence, ...] = ()
    semi_imperative_direction: SemiImperativeDirection | None = None
    undetermined_reason: str | None = None

    @model_validator(mode="after")
    def validate_closed_record(self) -> ProvisionCharacter:
        if self.kind is ProvisionKind.UNDETERMINED:
            if self.evidence:
                raise ValueError(
                    "undetermined provision character cannot carry evidence"
                )
            if not self.undetermined_reason or not self.undetermined_reason.strip():
                raise ValueError("undetermined provision character requires a reason")
            if self.semi_imperative_direction is not None:
                raise ValueError(
                    "semi_imperative_direction is only valid for semi-imperative "
                    "provision character"
                )
            return self

        if not self.evidence:
            raise ValueError("determined provision character requires evidence")

        if self.kind is ProvisionKind.SEMI_IMPERATIVE:
            has_direction = self.semi_imperative_direction is not None
            has_reason = bool(
                self.undetermined_reason and self.undetermined_reason.strip()
            )
            if has_direction == has_reason:
                raise ValueError(
                    "semi-imperative provision character requires exactly one of "
                    "semi_imperative_direction or undetermined_reason"
                )
            return self

        if self.semi_imperative_direction is not None or self.undetermined_reason:
            raise ValueError(
                "semi_imperative_direction and undetermined_reason are only valid "
                "for semi-imperative or undetermined provision character"
            )
        return self


def validate_force_record_pair(
    act_force: ForceState, provision_force: ForceState
) -> None:
    """Enforce the RF-05 v1.4 roles: the act record admits, the provision vetoes.

    Enforced at construction rather than at each use, so no caller can hold a
    candidate or unit whose two records disagree about which is which. Without
    this, ``candidate_admits`` and ``InForceLegalBasis`` reach opposite
    verdicts on the same pairing, and a provision record carrying ``in_force``
    -- an RF-05 v1.4 violation clause -- reaches the decision as an admission.
    """
    if act_force.scope is not ForceScope.ACT:
        raise ValueError("the admitting force record must be act-scope")
    if provision_force.scope is not ForceScope.PROVISION:
        raise ValueError("the vetoing force record must be provision-scope")
    if provision_force.value is ForceValue.IN_FORCE:
        raise ValueError("a provision record never carries in_force")


class RetrievalCandidate(FrozenModel):
    locator: str
    snapshot_id: str
    act_identifier: str
    act_force: ForceState
    provision_force: ForceState
    rank: int
    sparse_score: float
    dense_score: float

    @model_validator(mode="after")
    def validate_force_records(self) -> RetrievalCandidate:
        validate_force_record_pair(self.act_force, self.provision_force)
        return self


def candidate_decision_force(candidate: RetrievalCandidate) -> ForceValue:
    if candidate.provision_force.value is ForceValue.NOT_IN_FORCE:
        return ForceValue.NOT_IN_FORCE
    return candidate.act_force.value


def candidate_admits(candidate: RetrievalCandidate) -> bool:
    return candidate_decision_force(candidate) is ForceValue.IN_FORCE


class InForceLegalBasis(FrozenModel):
    """An admitted, in-force legal basis with its characterisation for adjudication.

    Carries the collapsed admitting force state (which must be IN_FORCE and
    ACT-scope), the corpus snapshot date, and the characterised normative role
    (ProvisionCharacter). Contrasts with the wire DTO LegalBasisReference (in
    contract_analyzer.api.schemas) and EmittedBasis (in
    contract_analyzer.domain.records), both of which preserve the
    uncollapsed act/provision force record pair required by RF-05.
    """

    provision_locator: str
    act_identifier: str
    snapshot_date: date
    force_state: ForceState
    character: ProvisionCharacter

    @model_validator(mode="after")
    def validate_in_force_only(self) -> InForceLegalBasis:
        if self.force_state.value is not ForceValue.IN_FORCE:
            raise ValueError("in-force legal basis requires in_force force state")
        if self.force_state.scope is not ForceScope.ACT:
            raise ValueError("in-force legal basis requires act-scope admitting force")
        return self


class EmittedBasis(FrozenModel):
    """An admitted legal basis in the shape that reaches the result.

    Both force records travel together: RF-05 v1.4 carries the provision
    record's ``undetermined`` state through to the result, and forbids an
    output that states or implies provision-level force where only the act
    record supports it. Only the character's *kind* is carried -- the evidence
    apparatus is prose, it does not belong in a text-free metadata store, and
    the evaluation protocol puts it outside adjudication scope. A vetoed candidate never
    becomes one of these: RF-05 forbids presenting a rejected retrieval
    candidate as a legal basis, so the veto is reported by the finding code
    and its locator instead.
    """

    provision_locator: str
    act_identifier: str
    act_force: ForceState
    provision_force: ForceState
    character_kind: ProvisionKind

    @model_validator(mode="after")
    def validate_admitted(self) -> EmittedBasis:
        validate_force_record_pair(self.act_force, self.provision_force)
        if self.act_force.value is not ForceValue.IN_FORCE:
            raise ValueError("an emitted basis requires an admitting act record")
        if self.provision_force.value is ForceValue.NOT_IN_FORCE:
            raise ValueError("a vetoed candidate is never an emitted basis")
        return self


class Finding(FrozenModel):
    code: FindingCode
    uncertain_cause: UncertainCause | None = None
    raw_confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_uncertainty_fields(self) -> Finding:
        if self.code is FindingCode.UNCERTAIN:
            if self.uncertain_cause is None:
                raise ValueError(
                    "uncertain findings require exactly one uncertain_cause"
                )
            if self.raw_confidence is None:
                raise ValueError("uncertain findings require raw_confidence")
        else:
            if self.uncertain_cause is not None:
                raise ValueError("uncertain_cause is only valid for uncertain findings")
            if self.raw_confidence is not None:
                raise ValueError("raw_confidence is only valid for uncertain findings")
        return self
