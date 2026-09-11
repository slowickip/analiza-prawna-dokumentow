"""Apply the decision rules to worksheet verdicts and store their findings."""

from __future__ import annotations

from contract_analyzer.agents.entries import (
    CandidateEntry,
    admits,
    retrieval_candidate,
)
from contract_analyzer.agents.findings import resolve_quote_span, store_unit_finding
from contract_analyzer.agents.guards import (
    require_relation_direction,
    require_uncertain_confidence,
)
from contract_analyzer.agents.state import UnitContext
from contract_analyzer.agents.tools import RelationVerdictArgs, RelevanceVerdictArgs
from contract_analyzer.agents.turns import ToolResult, refusal, refused
from contract_analyzer.domain import (
    DecisionFacts,
    EmittedBasis,
    Finding,
    InForceLegalBasis,
    QuoteResolution,
    resolve_finding,
)
from contract_analyzer.model import AttemptTelemetry, InvalidModelResponse


def record_relevance(
    context: UnitContext,
    candidate: CandidateEntry,
    args: RelevanceVerdictArgs,
    attempts: tuple[AttemptTelemetry, ...],
) -> list[str] | ToolResult:
    """Record the relevance verdict, and the finding it settles if it settles one.

    An uncertain verdict and a force-vetoed candidate both end the candidate here,
    with a finding. An irrelevant candidate ends without one: the unit may still
    have a basis among the others, and only when none does is the unit's own
    no-relation finding recorded.
    """
    facts = DecisionFacts.from_candidate(
        retrieval_candidate(candidate, rank=1),
        relevant=args.relevant,
        candidates_cleared=len(context.worksheet.candidates()),
        raw_confidence=args.raw_confidence,
    )
    settles = args.relevant is None or (args.relevant is True and not admits(candidate))
    if settles:
        try:
            require_uncertain_confidence(facts, attempts)
        except InvalidModelResponse:
            return refused(
                refusal(
                    "raw_confidence_required",
                    msg="raw_confidence is required for this verdict",
                ),
                "post_verdict",
                context=context,
            )
    context.worksheet.add_verdict(
        candidate_id=candidate.id,
        stage="relevance",
        relevant=args.relevant,
        quote=args.quote,
        raw_confidence=args.raw_confidence,
        based_on=args.based_on,
    )
    if not settles:
        return []
    quote = resolve_quote_span(context.active.session, context.call_unit, args.quote)
    return _store(
        context,
        candidate,
        resolve_finding(facts),
        quote_span=quote.span,
        quote_resolution=quote.resolution,
    )


def record_relation(
    context: UnitContext,
    candidate: CandidateEntry,
    args: RelationVerdictArgs,
    attempts: tuple[AttemptTelemetry, ...],
) -> list[str] | ToolResult:
    """Record the relation verdict and the finding it produces."""
    character = context.worksheet.character_for(candidate.id)
    if character is None:
        raise RuntimeError("verifier.relation scheduled without a characterisation")
    basis = InForceLegalBasis(
        provision_locator=candidate.locator,
        act_identifier=candidate.act_identifier,
        snapshot_date=candidate.act_force.snapshot_date,
        force_state=candidate.act_force,
        character=character.character,
    )
    facts = DecisionFacts.from_basis(
        basis,
        departure=args.departure,
        direction=args.direction,
        candidates_cleared=len(context.worksheet.candidates()),
        raw_confidence=args.raw_confidence,
    )
    try:
        require_uncertain_confidence(facts, attempts)
    except InvalidModelResponse:
        return refused(
            refusal(
                "raw_confidence_required",
                msg="raw_confidence is required for this verdict",
            ),
            "post_verdict",
            context=context,
        )
    try:
        require_relation_direction(facts, attempts)
    except InvalidModelResponse:
        return refused(
            refusal("direction_required", msg="direction is required for this verdict"),
            "post_verdict",
            context=context,
        )
    context.worksheet.add_verdict(
        candidate_id=candidate.id,
        stage="relation",
        departure=args.departure,
        direction=args.direction,
        raw_confidence=args.raw_confidence,
        based_on=args.based_on,
    )
    relevance = context.worksheet.verdict_for(candidate.id, "relevance")
    quote = resolve_quote_span(
        context.active.session,
        context.call_unit,
        relevance.quote if relevance is not None and relevance.quote else "",
    )
    return _store(
        context,
        candidate,
        resolve_finding(facts),
        quote_span=quote.span,
        quote_resolution=quote.resolution,
        basis=EmittedBasis(
            provision_locator=candidate.locator,
            act_identifier=candidate.act_identifier,
            act_force=candidate.act_force,
            provision_force=candidate.provision_force,
            character_kind=character.character.kind,
        ),
    )


def _store(
    context: UnitContext,
    candidate: CandidateEntry,
    finding: Finding,
    *,
    quote_span: tuple[int, int] | None,
    quote_resolution: QuoteResolution,
    basis: EmittedBasis | None = None,
) -> list[str]:
    record = store_unit_finding(
        context.active,
        context.call_unit.unit_id,
        finding,
        legal_locators=(candidate.locator,),
        basis=basis,
        quote_span=quote_span,
        quote_resolution=quote_resolution,
    )
    context.findings.append(record)
    return [str(record.id)]
