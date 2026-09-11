from __future__ import annotations

from datetime import date
from itertools import product

import pytest
from pydantic import ValidationError

from contract_analyzer.agents.guards import uncertain_outcome_requires_confidence
from contract_analyzer.domain import (
    CharacterEvidence,
    DecisionFacts,
    DepartureDirection,
    DepartureState,
    EmittedBasis,
    Finding,
    FindingCode,
    ForceScope,
    ForceState,
    ForceValue,
    InForceLegalBasis,
    InterpretiveMethod,
    ProvisionCharacter,
    ProvisionKind,
    ReferenceStatus,
    RetrievalCandidate,
    SemiImperativeDirection,
    SourceKind,
    UncertainCause,
    Unit,
    candidate_admits,
    resolve_finding,
)


def relevant_facts(**overrides: object) -> DecisionFacts:
    facts: dict[str, object] = {
        "processed": True,
        "adjudicable": True,
        "candidates_cleared": 1,
        "relevant": True,
        "force": "in_force",
        "character": "imperative",
        "departure": "none",
    }
    facts.update(overrides)
    return DecisionFacts.model_validate(facts)


def semi_facts(**overrides: object) -> DecisionFacts:
    facts: dict[str, object] = {
        "processed": True,
        "adjudicable": True,
        "candidates_cleared": 1,
        "relevant": True,
        "force": "in_force",
        "character": "semi_imperative",
        "departure": "present",
        "permitted": "known",
        "direction": "with_permitted_direction",
    }
    facts.update(overrides)
    return DecisionFacts.model_validate(facts)


@pytest.mark.parametrize(
    ("facts", "code", "cause", "confidence"),
    [
        (DecisionFacts(processed=False), FindingCode.NOT_PROCESSED, None, None),
        (
            DecisionFacts(adjudicable=False),
            FindingCode.UNIT_NOT_ADJUDICABLE,
            None,
            None,
        ),
        (DecisionFacts(candidates_cleared=0), FindingCode.NO_BASIS_FOUND, None, None),
        (
            DecisionFacts(candidates_cleared=1, relevant=False),
            FindingCode.NO_RELATION,
            None,
            None,
        ),
        (
            DecisionFacts(candidates_cleared=1, relevant=None, raw_confidence=0.41),
            FindingCode.UNCERTAIN,
            UncertainCause.RELATION_BELOW_THRESHOLD,
            0.41,
        ),
        (
            relevant_facts(force="undetermined", raw_confidence=0.52),
            FindingCode.UNCERTAIN,
            UncertainCause.FORCE_STATE_UNDETERMINED,
            0.52,
        ),
        (
            relevant_facts(force="not_in_force"),
            FindingCode.BASIS_NOT_IN_FORCE,
            None,
            None,
        ),
        (
            relevant_facts(character="undetermined", raw_confidence=0.63),
            FindingCode.UNCERTAIN,
            UncertainCause.PROVISION_CHARACTER_UNDETERMINED,
            0.63,
        ),
        (
            relevant_facts(character="imperative", departure="present"),
            FindingCode.CONTRADICTORY,
            None,
            None,
        ),
        (
            relevant_facts(character="dispositive", departure="present"),
            FindingCode.PERMISSIBLE_DEPARTURE,
            None,
            None,
        ),
        (
            relevant_facts(character="imperative", departure="none"),
            FindingCode.CONSISTENT,
            None,
            None,
        ),
        (
            relevant_facts(departure="undetermined", raw_confidence=0.44),
            FindingCode.UNCERTAIN,
            UncertainCause.RELATION_BELOW_THRESHOLD,
            0.44,
        ),
        (
            semi_facts(permitted=None, raw_confidence=0.55),
            FindingCode.UNCERTAIN,
            UncertainCause.PERMITTED_DIRECTION_UNDETERMINED,
            0.55,
        ),
        (
            semi_facts(
                direction=DepartureDirection.UNDETERMINED,
                raw_confidence=0.66,
            ),
            FindingCode.UNCERTAIN,
            UncertainCause.DEPARTURE_DIRECTION_UNDETERMINED,
            0.66,
        ),
        (
            semi_facts(direction="with_permitted_direction"),
            FindingCode.PERMISSIBLE_DEPARTURE,
            None,
            None,
        ),
        (
            semi_facts(direction="against_permitted_direction"),
            FindingCode.CONTRADICTORY,
            None,
            None,
        ),
    ],
)
def test_closed_mapping(
    facts: DecisionFacts,
    code: FindingCode,
    cause: UncertainCause | None,
    confidence: float | None,
) -> None:
    result = resolve_finding(facts)
    assert (result.code, result.uncertain_cause, result.raw_confidence) == (
        code,
        cause,
        confidence,
    )


def test_resolve_finding_rejects_semi_imperative_missing_direction() -> None:
    facts = semi_facts(direction=None)
    with pytest.raises(
        ValueError, match="semi-imperative departure requires an explicit direction"
    ):
        resolve_finding(facts)


def test_resolve_finding_not_in_force_before_undetermined_departure() -> None:
    facts = relevant_facts(
        force="not_in_force",
        departure="undetermined",
        raw_confidence=0.44,
    )
    result = resolve_finding(facts)
    assert result.code is FindingCode.BASIS_NOT_IN_FORCE


def test_resolve_finding_undetermined_force_before_undetermined_departure() -> None:
    facts = relevant_facts(
        force="undetermined",
        departure="undetermined",
        raw_confidence=0.52,
    )
    result = resolve_finding(facts)
    assert (result.code, result.uncertain_cause, result.raw_confidence) == (
        FindingCode.UNCERTAIN,
        UncertainCause.FORCE_STATE_UNDETERMINED,
        0.52,
    )


def test_resolve_finding_rejects_relevant_missing_force() -> None:
    facts = DecisionFacts(
        processed=True,
        adjudicable=True,
        candidates_cleared=1,
        relevant=True,
        character=ProvisionKind.IMPERATIVE,
        departure=DepartureState.NONE,
    )
    with pytest.raises(ValueError, match="force"):
        resolve_finding(facts)


def test_resolve_finding_rejects_relevant_missing_character_before_departure() -> None:
    facts = DecisionFacts(
        processed=True,
        adjudicable=True,
        candidates_cleared=1,
        relevant=True,
        force=ForceValue.IN_FORCE,
        departure=DepartureState.NONE,
    )
    with pytest.raises(ValueError, match="character"):
        resolve_finding(facts)


def test_resolve_finding_rejects_uncertain_missing_raw_confidence() -> None:
    facts = DecisionFacts(candidates_cleared=1, relevant=None)
    with pytest.raises(ValueError, match="raw_confidence"):
        resolve_finding(facts)


def test_resolve_finding_rejects_missing_departure() -> None:
    facts = DecisionFacts(
        processed=True,
        adjudicable=True,
        candidates_cleared=1,
        relevant=True,
        force=ForceValue.IN_FORCE,
        character=ProvisionKind.IMPERATIVE,
        departure=None,
    )
    with pytest.raises(
        ValueError, match="relevant decision facts require explicit departure"
    ):
        resolve_finding(facts)


def test_finding_requires_cause_for_uncertain() -> None:
    with pytest.raises(ValidationError):
        Finding(code=FindingCode.UNCERTAIN, raw_confidence=0.5)


def test_finding_requires_raw_confidence_for_uncertain() -> None:
    with pytest.raises(ValidationError):
        Finding(
            code=FindingCode.UNCERTAIN,
            uncertain_cause=UncertainCause.RELATION_BELOW_THRESHOLD,
        )


def test_finding_rejects_raw_confidence_for_non_uncertain() -> None:
    with pytest.raises(ValidationError):
        Finding(code=FindingCode.CONSISTENT, raw_confidence=0.5)


def test_finding_rejects_raw_confidence_outside_unit_interval() -> None:
    with pytest.raises(ValidationError):
        Finding(
            code=FindingCode.UNCERTAIN,
            uncertain_cause=UncertainCause.RELATION_BELOW_THRESHOLD,
            raw_confidence=1.1,
        )


def test_finding_rejects_cause_for_non_uncertain() -> None:
    with pytest.raises(ValidationError):
        Finding(
            code=FindingCode.CONSISTENT,
            uncertain_cause=UncertainCause.RELATION_BELOW_THRESHOLD,
        )


def test_force_state_requires_provenance_fields() -> None:
    with pytest.raises(ValidationError):
        ForceState(value=ForceValue.IN_FORCE)


def test_force_state_rejects_blank_source_locator() -> None:
    with pytest.raises(ValidationError):
        ForceState(
            value=ForceValue.IN_FORCE,
            scope=ForceScope.PROVISION,
            snapshot_date=date(2026, 1, 1),
            source_locator="   ",
        )


def test_force_state_accepts_complete_record() -> None:
    record = ForceState(
        value=ForceValue.IN_FORCE,
        scope=ForceScope.PROVISION,
        snapshot_date=date(2026, 1, 1),
        source_locator="https://api.sejm.gov.pl/eli/acts/DZU/2024/1200",
    )
    assert record.value is ForceValue.IN_FORCE


@pytest.mark.parametrize(
    "status",
    [
        ReferenceStatus.RESOLVED,
        ReferenceStatus.TARGET_DOES_NOT_EXIST,
        ReferenceStatus.WITHIN_UNIT,
        ReferenceStatus.AMBIGUOUS_DIRECTIONAL,
        ReferenceStatus.OUTSIDE_INPUT,
        ReferenceStatus.WHOLE_DOCUMENT,
        ReferenceStatus.EXTERNAL_ACT,
    ],
)
def test_reference_status_wire_values(status: ReferenceStatus) -> None:
    assert status.value == status.name.lower()


def test_unit_uses_id_field() -> None:
    unit = Unit.model_validate(
        {
            "id": "unit-1",
            "text": "§ 1",
            "anchor": {"start_offset": 0, "end_offset": 3},
        }
    )
    assert unit.id == "unit-1"


def test_retrieval_candidate_uses_locator_and_snapshot_id() -> None:
    candidate = RetrievalCandidate(
        locator="art.659@2024",
        snapshot_id="snapshot-2024-01-01",
        act_identifier="KC",
        act_force=ForceState(
            value=ForceValue.IN_FORCE,
            scope=ForceScope.ACT,
            snapshot_date=date(2026, 1, 1),
            source_locator="https://api.sejm.gov.pl/eli/acts/DZU/2024/1200",
        ),
        provision_force=ForceState(
            value=ForceValue.UNDETERMINED,
            scope=ForceScope.PROVISION,
            snapshot_date=date(2026, 1, 1),
            source_locator="https://api.sejm.gov.pl/eli/acts/DZU/2024/1200/art/659",
        ),
        rank=1,
        sparse_score=0.2,
        dense_score=0.8,
    )
    assert candidate.locator == "art.659@2024"
    assert candidate.snapshot_id == "snapshot-2024-01-01"


def _sample_evidence() -> CharacterEvidence:
    return CharacterEvidence(
        source_kind=SourceKind.OFFICIAL_NORMATIVE_TEXT,
        locator="https://example.test/act",
        pinpoint="art. 659 § 1",
        interpretive_methods=(InterpretiveMethod.LINGUISTIC,),
        rationale="Uses mandatory wording.",
    )


def test_provision_character_requires_evidence_for_determined_kind() -> None:
    with pytest.raises(ValidationError):
        ProvisionCharacter(kind=ProvisionKind.IMPERATIVE, evidence=())


def test_provision_character_undetermined_requires_reason_without_evidence() -> None:
    with pytest.raises(ValidationError):
        ProvisionCharacter(
            kind=ProvisionKind.UNDETERMINED,
            evidence=(_sample_evidence(),),
            undetermined_reason="No qualifying source.",
        )
    with pytest.raises(ValidationError):
        ProvisionCharacter(kind=ProvisionKind.UNDETERMINED)
    character = ProvisionCharacter(
        kind=ProvisionKind.UNDETERMINED,
        undetermined_reason="No qualifying source.",
    )
    assert character.evidence == ()


def test_provision_character_undetermined_rejects_semi_imperative_direction() -> None:
    with pytest.raises(ValidationError, match="semi_imperative_direction"):
        ProvisionCharacter(
            kind=ProvisionKind.UNDETERMINED,
            undetermined_reason="No qualifying source.",
            semi_imperative_direction=SemiImperativeDirection(
                protected_party_role="tenant"
            ),
        )


def test_provision_character_semi_imperative_requires_direction_or_reason() -> None:
    with pytest.raises(ValidationError):
        ProvisionCharacter(
            kind=ProvisionKind.SEMI_IMPERATIVE,
            evidence=(_sample_evidence(),),
        )
    with pytest.raises(ValidationError):
        ProvisionCharacter(
            kind=ProvisionKind.SEMI_IMPERATIVE,
            evidence=(_sample_evidence(),),
            semi_imperative_direction=SemiImperativeDirection(protected_party_role=""),
        )
    direction = SemiImperativeDirection(protected_party_role="tenant")
    character = ProvisionCharacter(
        kind=ProvisionKind.SEMI_IMPERATIVE,
        evidence=(_sample_evidence(),),
        semi_imperative_direction=direction,
    )
    assert character.semi_imperative_direction == direction
    undetermined = ProvisionCharacter(
        kind=ProvisionKind.SEMI_IMPERATIVE,
        evidence=(_sample_evidence(),),
        undetermined_reason="Direction not established.",
    )
    assert undetermined.semi_imperative_direction is None


def test_character_evidence_rejects_empty_fields() -> None:
    with pytest.raises(ValidationError):
        CharacterEvidence(
            source_kind=SourceKind.OFFICIAL_NORMATIVE_TEXT,
            locator="",
            pinpoint="art. 659 § 1",
            interpretive_methods=(InterpretiveMethod.LINGUISTIC,),
            rationale="Uses mandatory wording.",
        )


def test_in_force_legal_basis_requires_in_force_force_state() -> None:
    with pytest.raises(ValidationError):
        InForceLegalBasis(
            provision_locator="art.659@2024",
            act_identifier="KC",
            snapshot_date=date(2026, 1, 1),
            force_state=ForceState(
                value=ForceValue.NOT_IN_FORCE,
                scope=ForceScope.ACT,
                snapshot_date=date(2026, 1, 1),
                source_locator="https://api.sejm.gov.pl/eli/acts/DZU/2024/1200",
            ),
            character=ProvisionCharacter(
                kind=ProvisionKind.IMPERATIVE,
                evidence=(_sample_evidence(),),
            ),
        )
    with pytest.raises(ValidationError, match="act-scope"):
        InForceLegalBasis(
            provision_locator="art.659@2024",
            act_identifier="KC",
            snapshot_date=date(2026, 1, 1),
            force_state=ForceState(
                value=ForceValue.IN_FORCE,
                scope=ForceScope.PROVISION,
                snapshot_date=date(2026, 1, 1),
                source_locator="https://api.sejm.gov.pl/eli/acts/DZU/2024/1200/art/659",
            ),
            character=ProvisionCharacter(
                kind=ProvisionKind.IMPERATIVE,
                evidence=(_sample_evidence(),),
            ),
        )


def _act_force_state(value: ForceValue) -> ForceState:
    return ForceState(
        value=value,
        scope=ForceScope.ACT,
        snapshot_date=date(2026, 1, 1),
        source_locator="https://api.sejm.gov.pl/eli/acts/DZU/2024/1200",
    )


def _provision_force_state(value: ForceValue) -> ForceState:
    return ForceState(
        value=value,
        scope=ForceScope.PROVISION,
        snapshot_date=date(2026, 1, 1),
        source_locator="https://api.sejm.gov.pl/eli/acts/DZU/2024/1200/art/659",
    )


def _retrieval_candidate(
    act_force: ForceValue,
    *,
    provision_force: ForceValue = ForceValue.UNDETERMINED,
) -> RetrievalCandidate:
    return RetrievalCandidate(
        locator="art.659@2024",
        snapshot_id="snapshot-2024-01-01",
        act_identifier="KC",
        act_force=_act_force_state(act_force),
        provision_force=_provision_force_state(provision_force),
        rank=1,
        sparse_score=0.2,
        dense_score=0.8,
    )


def _legal_basis(character: ProvisionCharacter) -> InForceLegalBasis:
    return InForceLegalBasis(
        provision_locator="art.659@2024",
        act_identifier="KC",
        snapshot_date=date(2026, 1, 1),
        force_state=_act_force_state(ForceValue.IN_FORCE),
        character=character,
    )


def test_from_basis_rejects_contradictory_when_permitted_direction_undetermined() -> (
    None
):
    basis = _legal_basis(
        ProvisionCharacter(
            kind=ProvisionKind.SEMI_IMPERATIVE,
            evidence=(_sample_evidence(),),
            undetermined_reason="Direction not established.",
        )
    )
    facts = DecisionFacts.from_basis(
        basis,
        departure=DepartureState.PRESENT,
        direction=DepartureDirection.AGAINST_PERMITTED_DIRECTION,
        candidates_cleared=1,
        raw_confidence=0.71,
    )
    result = resolve_finding(facts)
    assert (result.code, result.uncertain_cause) == (
        FindingCode.UNCERTAIN,
        UncertainCause.PERMITTED_DIRECTION_UNDETERMINED,
    )
    assert result.code is not FindingCode.CONTRADICTORY


def test_from_basis_derives_known_permitted_and_reaches_direction_branch() -> None:
    basis = _legal_basis(
        ProvisionCharacter(
            kind=ProvisionKind.SEMI_IMPERATIVE,
            evidence=(_sample_evidence(),),
            semi_imperative_direction=SemiImperativeDirection(
                protected_party_role="tenant"
            ),
        )
    )
    facts = DecisionFacts.from_basis(
        basis,
        departure=DepartureState.PRESENT,
        direction=DepartureDirection.AGAINST_PERMITTED_DIRECTION,
        candidates_cleared=1,
    )
    assert facts.permitted == "known"
    result = resolve_finding(facts)
    assert result.code is FindingCode.CONTRADICTORY


def test_from_candidate_provision_veto_overrides_in_force_act() -> None:
    candidate = _retrieval_candidate(
        ForceValue.IN_FORCE,
        provision_force=ForceValue.NOT_IN_FORCE,
    )
    facts = DecisionFacts.from_candidate(
        candidate,
        relevant=True,
        candidates_cleared=1,
    )
    result = resolve_finding(facts)
    assert result.code is FindingCode.BASIS_NOT_IN_FORCE


@pytest.mark.parametrize(
    ("force", "relevant", "code", "cause", "confidence"),
    [
        (
            ForceValue.UNDETERMINED,
            True,
            FindingCode.UNCERTAIN,
            UncertainCause.FORCE_STATE_UNDETERMINED,
            0.52,
        ),
        (
            ForceValue.NOT_IN_FORCE,
            True,
            FindingCode.BASIS_NOT_IN_FORCE,
            None,
            None,
        ),
        (ForceValue.IN_FORCE, False, FindingCode.NO_RELATION, None, None),
        (
            ForceValue.IN_FORCE,
            None,
            FindingCode.UNCERTAIN,
            UncertainCause.RELATION_BELOW_THRESHOLD,
            0.41,
        ),
    ],
)
def test_from_candidate_carries_force_to_matching_result(
    force: ForceValue,
    relevant: bool | None,
    code: FindingCode,
    cause: UncertainCause | None,
    confidence: float | None,
) -> None:
    candidate = _retrieval_candidate(force)
    facts = DecisionFacts.from_candidate(
        candidate,
        relevant=relevant,
        candidates_cleared=1,
        raw_confidence=confidence,
    )
    assert facts.force is force
    assert facts.character is None
    assert facts.departure is None
    assert facts.direction is None
    assert facts.permitted is None
    result = resolve_finding(facts)
    assert (result.code, result.uncertain_cause, result.raw_confidence) == (
        code,
        cause,
        confidence,
    )


def test_from_basis_derives_authoritative_fields() -> None:
    character = ProvisionCharacter(
        kind=ProvisionKind.SEMI_IMPERATIVE,
        evidence=(_sample_evidence(),),
        undetermined_reason="Direction not established.",
    )
    basis = _legal_basis(character)
    facts = DecisionFacts.from_basis(
        basis,
        departure=DepartureState.PRESENT,
        direction=DepartureDirection.WITH_PERMITTED_DIRECTION,
        candidates_cleared=1,
    )
    assert facts.relevant is True
    assert facts.force is ForceValue.IN_FORCE
    assert facts.character is ProvisionKind.SEMI_IMPERATIVE
    assert facts.permitted is None


_RAW_CONFIDENCE_VALUE_ERROR = "uncertain decision facts require raw_confidence"


def test_uncertain_confidence_predicate_tracks_resolve_finding() -> None:
    disagreements: list[str] = []
    checked = 0

    for (
        processed,
        adjudicable,
        candidates_cleared,
        relevant,
        force,
        character,
        departure,
        permitted,
        direction,
    ) in product(
        (True, False),
        (True, False),
        (0, 1),
        (True, False, None),
        (None, *ForceValue),
        (None, *ProvisionKind),
        (None, *DepartureState),
        (None, "known"),
        (None, *DepartureDirection),
    ):
        try:
            facts = DecisionFacts(
                processed=processed,
                adjudicable=adjudicable,
                candidates_cleared=candidates_cleared,
                relevant=relevant,
                force=force,
                character=character,
                departure=departure,
                permitted=permitted,
                direction=direction,
                raw_confidence=None,
            )
        except ValidationError:
            continue

        checked += 1
        try:
            resolve_finding(facts)
            resolve_needs_confidence = False
        except ValueError as error:
            resolve_needs_confidence = str(error) == _RAW_CONFIDENCE_VALUE_ERROR

        predicate_needs_confidence = uncertain_outcome_requires_confidence(facts)
        if predicate_needs_confidence != resolve_needs_confidence:
            disagreements.append(
                f"{facts!r}: predicate={predicate_needs_confidence}, "
                f"resolve_finding_needs_confidence={resolve_needs_confidence}"
            )

    assert checked > 0
    assert not disagreements, "predicate disagrees with resolve_finding:\n" + "\n".join(
        disagreements
    )


@pytest.mark.parametrize(
    ("act_value", "act_scope", "provision_value", "provision_scope", "message"),
    [
        (
            ForceValue.IN_FORCE,
            ForceScope.PROVISION,
            ForceValue.UNDETERMINED,
            ForceScope.ACT,
            "the admitting force record must be act-scope",
        ),
        (
            ForceValue.IN_FORCE,
            ForceScope.ACT,
            ForceValue.UNDETERMINED,
            ForceScope.ACT,
            "the vetoing force record must be provision-scope",
        ),
        (
            ForceValue.IN_FORCE,
            ForceScope.PROVISION,
            ForceValue.UNDETERMINED,
            ForceScope.PROVISION,
            "the admitting force record must be act-scope",
        ),
        (
            ForceValue.IN_FORCE,
            ForceScope.ACT,
            ForceValue.IN_FORCE,
            ForceScope.PROVISION,
            "a provision record never carries in_force",
        ),
    ],
)
def test_retrieval_candidate_rejects_swapped_or_admitting_force_records(
    act_value: ForceValue,
    act_scope: ForceScope,
    provision_value: ForceValue,
    provision_scope: ForceScope,
    message: str,
) -> None:
    """RF-05 v1.4 roles hold at construction, not at each use.

    Before this guard every pairing here constructed. Two of them decided a
    finding: swapped scopes made ``candidate_admits`` and
    ``InForceLegalBasis`` disagree about the same candidate, and a provision
    record carrying ``in_force`` reached ``DecisionFacts.force`` as an
    admission -- an RF-05 v1.4 violation clause.
    """
    with pytest.raises(ValidationError, match=message):
        RetrievalCandidate(
            locator="art. 659",
            snapshot_id="snapshot-2026-05-19",
            act_identifier="DU/2026/795",
            act_force=ForceState(
                value=act_value,
                scope=act_scope,
                snapshot_date=date(2026, 5, 19),
                source_locator="https://api.sejm.gov.pl/eli/acts/DU/2026/795",
            ),
            provision_force=ForceState(
                value=provision_value,
                scope=provision_scope,
                snapshot_date=date(2026, 5, 19),
                source_locator="https://api.sejm.gov.pl/eli/acts/DU/2026/795",
            ),
            rank=1,
            sparse_score=0.2,
            dense_score=0.8,
        )


def test_retrieval_candidate_still_accepts_the_two_lawful_pairings() -> None:
    """The guard must reject bad states without rejecting good ones."""
    admitted = _retrieval_candidate(ForceValue.IN_FORCE)
    vetoed = _retrieval_candidate(
        ForceValue.IN_FORCE, provision_force=ForceValue.NOT_IN_FORCE
    )
    assert candidate_admits(admitted) is True
    assert candidate_admits(vetoed) is False


def test_emitted_basis_refuses_a_vetoed_or_unadmitted_candidate() -> None:
    """RF-05 v1.4 forbids presenting a rejected candidate as a legal basis."""
    act = ForceState(
        value=ForceValue.IN_FORCE,
        scope=ForceScope.ACT,
        snapshot_date=date(2026, 5, 19),
        source_locator="https://api.sejm.gov.pl/eli/acts/DU/2026/795",
    )
    provision = ForceState(
        value=ForceValue.NOT_IN_FORCE,
        scope=ForceScope.PROVISION,
        snapshot_date=date(2026, 5, 19),
        source_locator="https://api.sejm.gov.pl/eli/acts/DU/2026/795",
    )
    with pytest.raises(ValidationError, match="a vetoed candidate is never"):
        EmittedBasis(
            provision_locator="art. 659",
            act_identifier="DU/2026/795",
            act_force=act,
            provision_force=provision,
            character_kind=ProvisionKind.IMPERATIVE,
        )
    with pytest.raises(ValidationError, match="requires an admitting act record"):
        EmittedBasis(
            provision_locator="art. 659",
            act_identifier="DU/2026/795",
            act_force=act.model_copy(update={"value": ForceValue.UNDETERMINED}),
            provision_force=provision.model_copy(
                update={"value": ForceValue.UNDETERMINED}
            ),
            character_kind=ProvisionKind.IMPERATIVE,
        )


def test_character_evidence_blank_message_is_unchanged() -> None:
    """The refusal detail the model reads names the original rule wording."""
    with pytest.raises(ValidationError, match="must not be empty"):
        CharacterEvidence(
            source_kind=SourceKind.OFFICIAL_NORMATIVE_TEXT,
            locator="https://api.sejm.gov.pl/eli/acts/DU/2023/725/text.html/arti=6",
            pinpoint="ust. 1",
            interpretive_methods=(InterpretiveMethod.LINGUISTIC,),
            rationale="   ",
        )
