"""Finding decision facts and resolution table."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from .enums import (
    DepartureDirection,
    DepartureState,
    FindingCode,
    ForceValue,
    ProvisionKind,
    UncertainCause,
)
from .records import (
    Finding,
    FrozenModel,
    InForceLegalBasis,
    RetrievalCandidate,
    candidate_decision_force,
)

_DIRECTION_CODES = {
    DepartureDirection.WITH_PERMITTED_DIRECTION: FindingCode.PERMISSIBLE_DEPARTURE,
    DepartureDirection.AGAINST_PERMITTED_DIRECTION: FindingCode.CONTRADICTORY,
}


class DecisionFacts(FrozenModel):
    processed: bool = True
    adjudicable: bool = True
    candidates_cleared: int = 0
    relevant: bool | None = True
    force: ForceValue | None = None
    character: ProvisionKind | None = None
    departure: DepartureState | None = None
    permitted: Literal["known"] | None = None
    direction: DepartureDirection | None = None
    raw_confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    @classmethod
    def from_candidate(
        cls,
        candidate: RetrievalCandidate,
        *,
        relevant: bool | None,
        candidates_cleared: int,
        raw_confidence: float | None = None,
    ) -> DecisionFacts:
        """Facts for a candidate that has not become a legal basis.

        Force is derived from the candidate's act and provision records and cannot be
        passed in.
        """
        return cls(
            candidates_cleared=candidates_cleared,
            relevant=relevant,
            force=candidate_decision_force(candidate),
            raw_confidence=raw_confidence,
        )

    @classmethod
    def from_basis(
        cls,
        basis: InForceLegalBasis,
        *,
        departure: DepartureState,
        direction: DepartureDirection | None = None,
        candidates_cleared: int,
        raw_confidence: float | None = None,
    ) -> DecisionFacts:
        """Facts for an in-force basis under adjudication.

        Force, character and permitted are derived from the authoritative records and
        cannot be passed in.
        """
        character = basis.character
        permitted: Literal["known"] | None = (
            "known" if character.semi_imperative_direction is not None else None
        )
        return cls(
            candidates_cleared=candidates_cleared,
            relevant=True,
            force=basis.force_state.value,
            character=character.kind,
            departure=departure,
            direction=direction,
            permitted=permitted,
            raw_confidence=raw_confidence,
        )


def _uncertain_finding(facts: DecisionFacts, cause: UncertainCause) -> Finding:
    if facts.raw_confidence is None:
        raise ValueError("uncertain decision facts require raw_confidence")
    return Finding(
        code=FindingCode.UNCERTAIN,
        uncertain_cause=cause,
        raw_confidence=facts.raw_confidence,
    )


def resolve_finding(facts: DecisionFacts) -> Finding:
    if not facts.processed:
        return Finding(code=FindingCode.NOT_PROCESSED)
    if not facts.adjudicable:
        return Finding(code=FindingCode.UNIT_NOT_ADJUDICABLE)
    if facts.candidates_cleared == 0:
        return Finding(code=FindingCode.NO_BASIS_FOUND)
    if facts.relevant is False:
        return Finding(code=FindingCode.NO_RELATION)
    if facts.relevant is None:
        return _uncertain_finding(facts, UncertainCause.RELATION_BELOW_THRESHOLD)
    if facts.force is None:
        raise ValueError("relevant decision facts require explicit force")
    force = facts.force
    if force is ForceValue.UNDETERMINED:
        return _uncertain_finding(facts, UncertainCause.FORCE_STATE_UNDETERMINED)
    if force is ForceValue.NOT_IN_FORCE:
        return Finding(code=FindingCode.BASIS_NOT_IN_FORCE)
    if facts.departure is DepartureState.UNDETERMINED:
        return _uncertain_finding(facts, UncertainCause.RELATION_BELOW_THRESHOLD)

    character = facts.character
    if character is ProvisionKind.UNDETERMINED:
        return _uncertain_finding(
            facts, UncertainCause.PROVISION_CHARACTER_UNDETERMINED
        )
    if character is None:
        raise ValueError("relevant decision facts require explicit character")
    if character is ProvisionKind.SEMI_IMPERATIVE and facts.permitted is None:
        return _uncertain_finding(
            facts, UncertainCause.PERMITTED_DIRECTION_UNDETERMINED
        )

    departure = facts.departure
    if departure is None:
        raise ValueError("relevant decision facts require explicit departure")
    if departure is DepartureState.NONE:
        return Finding(code=FindingCode.CONSISTENT)
    if departure is DepartureState.PRESENT:
        if character is ProvisionKind.IMPERATIVE:
            return Finding(code=FindingCode.CONTRADICTORY)
        if character is ProvisionKind.DISPOSITIVE:
            return Finding(code=FindingCode.PERMISSIBLE_DEPARTURE)
        if character is ProvisionKind.SEMI_IMPERATIVE:
            if facts.direction is None:
                raise ValueError(
                    "semi-imperative departure requires an explicit direction"
                )
            if facts.direction is DepartureDirection.UNDETERMINED:
                return _uncertain_finding(
                    facts, UncertainCause.DEPARTURE_DIRECTION_UNDETERMINED
                )
            if facts.direction in _DIRECTION_CODES:
                return Finding(code=_DIRECTION_CODES[facts.direction])

    raise ValueError("unmapped decision facts")
