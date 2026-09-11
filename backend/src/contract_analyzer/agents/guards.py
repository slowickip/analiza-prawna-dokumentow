"""Predicates and guards for verdicts the decision rules cannot map.

This module alone owns the conditions. Worksheet tool handlers translate a raised
guard into a refusal so the model can correct its verdict within the task cap.
"""

from __future__ import annotations

from contract_analyzer.domain import (
    DecisionFacts,
    DepartureDirection,
    DepartureState,
    ForceValue,
    ProvisionKind,
)
from contract_analyzer.model import AttemptTelemetry, InvalidModelResponse


def uncertain_outcome_requires_confidence(facts: DecisionFacts) -> bool:
    """Whether these facts can only resolve to an uncertain finding.

    An uncertain finding carries a confidence, so a verdict that leads to one and
    omits it is incomplete rather than merely imprecise.
    """
    if not facts.processed or not facts.adjudicable:
        return False
    if facts.candidates_cleared == 0:
        return False
    if facts.relevant is False:
        return False
    if facts.relevant is None:
        return True
    if facts.force is ForceValue.UNDETERMINED:
        return True
    if facts.force is not ForceValue.IN_FORCE:
        return False
    if facts.departure is DepartureState.UNDETERMINED:
        return True
    character = facts.character
    if character is ProvisionKind.UNDETERMINED:
        return True
    if character is ProvisionKind.SEMI_IMPERATIVE and facts.permitted is None:
        return True
    if (
        facts.departure is DepartureState.PRESENT
        and character is ProvisionKind.SEMI_IMPERATIVE
        and facts.direction is DepartureDirection.UNDETERMINED
    ):
        return True
    return False


def require_uncertain_confidence(
    facts: DecisionFacts,
    attempts: tuple[AttemptTelemetry, ...],
) -> None:
    if uncertain_outcome_requires_confidence(facts) and facts.raw_confidence is None:
        raise InvalidModelResponse("model_response_schema_invalid", attempts)


def require_relation_direction(
    facts: DecisionFacts,
    attempts: tuple[AttemptTelemetry, ...],
) -> None:
    """Reject a departure from a semi-imperative provision that names no direction.

    ``resolve_finding`` maps that combination to nothing: with the permitted
    direction known, the direction decides between a permissible departure and a
    contradictory one, so it raises rather than guess. The model can produce it --
    the relation prompt allows a null direction under every departure state -- and
    an unhandled raise would end the whole run and every unit still queued behind
    it.
    """
    if (
        facts.departure is DepartureState.PRESENT
        and facts.character is ProvisionKind.SEMI_IMPERATIVE
        and facts.permitted is not None
        and facts.direction is None
    ):
        raise InvalidModelResponse("model_response_schema_invalid", attempts)
