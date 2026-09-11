"""From a verifier's verdict to a stored finding a reader can open.

Two subjects: what a unit resolves to when its candidates were decided in various
ways, and how the quoted fragment is anchored back into the contract -- resolved
to a span, fallen back to the unit, or recorded as unusable.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import pytest
from graph_harness import (
    DOCUMENT_TEXT,
    LOCATOR,
    LOCATOR_ART19A,
    LOCATOR_ART41,
    GraphDocument,
    TickingClock,
    build_runner,
    collect_retained_plaintext,
    content_hash,
    make_document,
    make_session,
    request_for,
)
from worksheet_transport import (
    Turn,
    character_arguments,
    characterising,
    deciding_relation,
    deciding_relevance,
    listed_refs,
    posting,
    relevance_quoting,
    silence,
    synthesis_empty,
    tool_call,
)

from contract_analyzer.agents.runner import AnalysisRunner
from contract_analyzer.agents.session import DocumentSession, RunRequest
from contract_analyzer.config import RunConfig
from contract_analyzer.corpus import QdrantCorpusIndex
from contract_analyzer.corpus import qdrant_index as qdrant_index_module
from contract_analyzer.domain import (
    ArmCode,
    DocumentPayload,
    FindingCode,
    ForceScope,
    ForceState,
    ForceValue,
    QuoteResolution,
    ReadMode,
    SourceAnchor,
    UncertainCause,
)
from contract_analyzer.storage import RunTextStore
from contract_analyzer.structure import parse_references, segment

MULTI_BASIS_TEXT = (
    "§ 1. Najemca składa kaucję zabezpieczającą "
    "i może wypowiedzieć najem zgodnie z ustawą."
)
THREE_LOCATORS = (LOCATOR, LOCATOR_ART19A, LOCATOR_ART41)


@pytest.fixture
def graph_document() -> GraphDocument:
    return make_document()


def _multi_basis_runner(
    built_corpus: QdrantCorpusIndex,
    tmp_path: Path,
    session: DocumentSession,
    *,
    db_name: str,
    relevance=None,
    relation=None,
    on_turn=None,
) -> AnalysisRunner:
    """A runner whose analyst posts all three fixture provisions for one unit."""
    return build_runner(
        session,
        built_corpus,
        tmp_path,
        db_name=db_name,
        on_turn=on_turn,
        script={
            "researcher.search": posting(*THREE_LOCATORS),
            "analyst.characterise": characterising(
                lambda locator: (
                    "dispositive" if locator == LOCATOR_ART19A else "imperative"
                )
            ),
            "verifier.relevance": deciding_relevance(
                relevance or (lambda _: {"relevant": True, "raw_confidence": 0.9})
            ),
            "verifier.relation": deciding_relation(
                relation or (lambda _: {"departure": "present"})
            ),
        },
    )


def _unit_findings(result: Any, unit_id: str) -> list[Any]:
    return [finding for finding in result.findings if finding.unit_id == unit_id]


def _relevant_findings(result: Any) -> list[Any]:
    return [
        finding
        for finding in result.findings
        if finding.code is not FindingCode.NOT_PROCESSED
    ]


def _run(runner: AnalysisRunner, session: DocumentSession, arm: ArmCode = ArmCode.MID):
    return runner.run(
        RunRequest(
            document_id=session.document_id,
            arm=arm,
            measurement_valid=True,
            wall_budget_seconds=300.0,
            concurrency=1,
        )
    )


def test_unit_with_multiple_bases_retains_every_finding(
    built_corpus: QdrantCorpusIndex, tmp_path: Path
) -> None:
    session = make_session(MULTI_BASIS_TEXT)
    unit_id = session.units[0].id
    runner = _multi_basis_runner(
        built_corpus, tmp_path, session, db_name="multi-basis.sqlite3"
    )

    result = asyncio.run(_run(runner, session))

    findings = _unit_findings(result, unit_id)
    codes = {finding.code for finding in findings}
    assert FindingCode.CONTRADICTORY in codes
    assert FindingCode.PERMISSIBLE_DEPARTURE in codes
    assert len(findings) == len(THREE_LOCATORS)


def test_all_undecided_candidates_yield_per_candidate_uncertain_findings(
    built_corpus: QdrantCorpusIndex, tmp_path: Path
) -> None:
    session = make_session(MULTI_BASIS_TEXT)
    unit_id = session.units[0].id
    runner = _multi_basis_runner(
        built_corpus,
        tmp_path,
        session,
        db_name="all-undecided.sqlite3",
        relevance=lambda _: {"relevant": None, "raw_confidence": 0.4},
    )

    result = asyncio.run(_run(runner, session))

    findings = _unit_findings(result, unit_id)
    assert findings
    assert all(finding.code is FindingCode.UNCERTAIN for finding in findings)
    assert all(
        finding.uncertain_cause is UncertainCause.RELATION_BELOW_THRESHOLD
        for finding in findings
    )
    assert all(finding.legal_locators for finding in findings)
    assert FindingCode.NO_RELATION not in {finding.code for finding in findings}


def test_mixed_relevance_retains_undecided_candidate_finding(
    built_corpus: QdrantCorpusIndex, tmp_path: Path
) -> None:
    session = make_session(MULTI_BASIS_TEXT)
    unit_id = session.units[0].id

    def relevance(locator: str) -> dict[str, object]:
        if locator == LOCATOR:
            return {"relevant": True, "raw_confidence": 0.9}
        if locator == LOCATOR_ART19A:
            return {"relevant": False, "raw_confidence": 0.9}
        return {"relevant": None, "raw_confidence": 0.4}

    runner = _multi_basis_runner(
        built_corpus,
        tmp_path,
        session,
        db_name="mixed-relevance.sqlite3",
        relevance=relevance,
    )

    result = asyncio.run(_run(runner, session))

    findings = _unit_findings(result, unit_id)
    codes = {finding.code for finding in findings}
    assert FindingCode.CONTRADICTORY in codes
    assert FindingCode.UNCERTAIN in codes
    assert FindingCode.NO_RELATION not in codes
    uncertain = [
        finding for finding in findings if finding.code is FindingCode.UNCERTAIN
    ]
    assert len(uncertain) == 1
    assert uncertain[0].uncertain_cause is UncertainCause.RELATION_BELOW_THRESHOLD
    assert uncertain[0].legal_locators == (LOCATOR_ART41,)


def test_all_irrelevant_candidates_yield_unit_level_no_relation(
    built_corpus: QdrantCorpusIndex, tmp_path: Path
) -> None:
    session = make_session(MULTI_BASIS_TEXT)
    unit_id = session.units[0].id
    runner = _multi_basis_runner(
        built_corpus,
        tmp_path,
        session,
        db_name="all-irrelevant.sqlite3",
        relevance=lambda _: {"relevant": False, "raw_confidence": 0.9},
    )

    result = asyncio.run(_run(runner, session))

    findings = _unit_findings(result, unit_id)
    assert [finding.code for finding in findings] == [FindingCode.NO_RELATION]


def test_a_candidate_its_force_records_veto_is_never_characterised(
    built_corpus: QdrantCorpusIndex, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A repealed provision cannot ground a finding, however relevant it looks."""
    session = make_session(MULTI_BASIS_TEXT)
    unit_id = session.units[0].id
    tasks: list[str] = []
    corpus = _corpus_with_repealed_provision(built_corpus, LOCATOR, monkeypatch)
    runner = build_runner(
        session,
        built_corpus,
        tmp_path,
        db_name="force-veto.sqlite3",
        corpus=corpus,
        on_turn=lambda turn: tasks.append(turn.task),
        script={"researcher.search": posting(LOCATOR)},
    )

    result = asyncio.run(_run(runner, session))

    findings = _unit_findings(result, unit_id)
    assert [finding.code for finding in findings] == [FindingCode.BASIS_NOT_IN_FORCE]
    assert "analyst.characterise" not in tasks
    assert "verifier.relation" not in tasks


def _corpus_with_repealed_provision(
    built_corpus: QdrantCorpusIndex, locator: str, monkeypatch: pytest.MonkeyPatch
) -> QdrantCorpusIndex:
    """A corpus whose one provision is repealed, on every path that decodes it.

    Reading and searching both decode a stored payload through one function, so
    replacing that function is what makes the provision repealed for both. Wrapping
    read alone left search returning the provision still in force.
    """
    corpus = built_corpus
    real_decode = qdrant_index_module._legal_unit_from_payload

    def decode(payload: object) -> Any:
        record = real_decode(payload)
        if record.locator != locator:
            return record
        return record.model_copy(
            update={
                "provision_force": ForceState(
                    value=ForceValue.NOT_IN_FORCE,
                    scope=ForceScope.PROVISION,
                    snapshot_date=record.provision_force.snapshot_date,
                    source_locator=locator,
                )
            }
        )

    monkeypatch.setattr(qdrant_index_module, "_legal_unit_from_payload", decode)
    return corpus


def test_relation_payload_carries_the_character_record(
    graph_document: GraphDocument, built_corpus: QdrantCorpusIndex, tmp_path: Path
) -> None:
    """The verifier is asked for a direction only a semi-imperative provision needs.

    The relation prompt makes the direction conditional on the provision's
    character, so withholding that character asks the role to apply a rule it
    cannot evaluate. The permitted direction travels with it.
    """
    payloads: list[dict[str, Any]] = []

    def capture(turn: Turn) -> None:
        if turn.task == "verifier.relation":
            payloads.append(turn.payload)

    runner = build_runner(
        graph_document.session,
        built_corpus,
        tmp_path,
        db_name="relation-character-payload.sqlite3",
        on_turn=capture,
        script={
            "analyst.characterise": lambda turn: tool_call(
                "post_character", character_arguments(turn, "semi_imperative")
            ),
            "verifier.relation": deciding_relation(
                lambda _: {
                    "departure": "present",
                    "direction": "with_permitted_direction",
                }
            ),
        },
    )

    asyncio.run(runner.run(request_for(graph_document, ArmCode.MID)))

    assert payloads
    for payload in payloads:
        assert payload["basis_character"] == "semi_imperative"
        assert payload["basis_permitted_direction"] == {
            "relation": "more_favourable_to",
            "protected_party_role": "najemca",
        }


def test_synthesis_payload_carries_each_finding_with_its_code_and_locators(
    graph_document: GraphDocument, built_corpus: QdrantCorpusIndex, tmp_path: Path
) -> None:
    """The role can only group what it can see, one entry per finding."""
    captured: list[dict[str, Any]] = []

    def synthesis(turn: Turn) -> Any:
        captured.append(turn.payload)
        return synthesis_empty(turn)

    runner = build_runner(
        graph_document.session,
        built_corpus,
        tmp_path,
        db_name="synthesis-payload.sqlite3",
        script={"synthesise": synthesis},
    )

    result = asyncio.run(runner.run(request_for(graph_document, ArmCode.MID)))

    assert result.status == "completed"
    persisted = runner.services.metadata.list_findings(result.run_id)
    assert persisted
    assert captured
    for payload in captured:
        assert "legal_locators" not in payload
        entries = cast(list[dict[str, Any]], payload["findings"])
        # The role is given this list's own numbering, never the identifiers: it
        # has to hand the reference back, and a number is something it can copy
        # exactly.
        assert [entry["ref"] for entry in entries] == [
            str(position) for position in range(1, len(persisted) + 1)
        ]
        assert not any("finding_id" in entry for entry in entries)
        assert [entry["code"] for entry in entries] == [
            record.code.value for record in persisted
        ]
        assert [entry["legal_locators"] for entry in entries] == [
            list(record.legal_locators) for record in persisted
        ]


def test_synthesis_group_citing_an_unknown_finding_is_dropped_not_fatal(
    graph_document: GraphDocument, built_corpus: QdrantCorpusIndex, tmp_path: Path
) -> None:
    """The invented group is refused; the findings it tried to group survive.

    Grouping is the last step and rearranges findings that are already decided
    and already stored. A role that names an identifier no unit produced has
    written an unusable grouping, and none of it is kept -- but the run itself
    analysed every unit, so discarding that work would throw away the analysis
    over a fault in its table of contents. This is the same ending the budget
    path already has: findings, and no grouping over them.
    """
    refusals: list[dict[str, Any]] = []

    def synthesis(turn: Turn) -> Any:
        if not turn.calls_made("list_findings"):
            return tool_call("list_findings", {})
        if not turn.calls_made("group_findings"):
            return tool_call(
                "group_findings",
                {
                    "groups": [
                        {
                            "title": "Grupa",
                            "summary": "Podsumowanie grupy.",
                            "finding_ids": ["9999"],
                        }
                    ]
                },
            )
        refusals.extend(result for result in turn.tool_results if "error" in result)
        return silence()

    runner = build_runner(
        graph_document.session,
        built_corpus,
        tmp_path,
        db_name="synthesis-unknown-finding.sqlite3",
        script={"synthesise": synthesis},
    )

    result = asyncio.run(runner.run(request_for(graph_document, ArmCode.MID)))

    assert result.status == "completed"
    assert result.error_code is None
    assert runner.services.metadata.list_findings(result.run_id)
    assert [refusal["error"] for refusal in refusals] == ["unknown_ref"]
    # Refused means kept nowhere: the reader is never offered a group naming a
    # finding they cannot open.
    assert runner.synthesis_for(result.run_id) is None


def test_synthesis_references_are_resolved_back_to_finding_identifiers(
    graph_document: GraphDocument, built_corpus: QdrantCorpusIndex, tmp_path: Path
) -> None:
    """The role answers in references; the reader is served identifiers.

    The numbering exists only to spare the role a transcription it kept getting
    wrong. Nothing outside this module may see it: a group has to name findings
    the client can open, which is what the published schema promises.
    """

    def synthesis(turn: Turn) -> Any:
        if not turn.calls_made("list_findings"):
            return tool_call("list_findings", {})
        if not turn.calls_made("group_findings"):
            return tool_call(
                "group_findings",
                {
                    "groups": [
                        {
                            "title": "Grupa",
                            "summary": "Podsumowanie grupy.",
                            "finding_ids": listed_refs(turn),
                        }
                    ]
                },
            )
        return silence()

    runner = build_runner(
        graph_document.session,
        built_corpus,
        tmp_path,
        db_name="synthesis-reference-resolution.sqlite3",
        script={"synthesise": synthesis},
    )

    result = asyncio.run(runner.run(request_for(graph_document, ArmCode.MID)))

    assert result.status == "completed"
    grouping = runner.synthesis_for(result.run_id)
    assert grouping is not None
    grouped = list(grouping.groups[0].finding_ids)
    persisted = [
        str(record.id)
        for record in runner.services.metadata.list_findings(result.run_id)
    ]
    assert grouped == persisted
    # The reference numbering never reaches the stored grouping.
    assert all(len(value) == 36 for value in grouped)


def test_synthesis_is_unreadable_once_its_retention_window_closes(
    graph_document: GraphDocument, built_corpus: QdrantCorpusIndex, tmp_path: Path
) -> None:
    """The store enforces the window, so the reader has to survive it closing."""
    now = {"value": 0.0}
    store = RunTextStore(RunConfig(content_ttl_seconds=60), clock=lambda: now["value"])

    def synthesis(turn: Turn) -> Any:
        if not turn.calls_made("list_findings"):
            return tool_call("list_findings", {})
        if not turn.calls_made("group_findings"):
            return tool_call(
                "group_findings",
                {
                    "groups": [
                        {
                            "title": "Grupa",
                            "summary": "Podsumowanie grupy.",
                            "finding_ids": listed_refs(turn),
                        }
                    ]
                },
            )
        return silence()

    runner = build_runner(
        graph_document.session,
        built_corpus,
        tmp_path,
        db_name="synthesis-retention.sqlite3",
        script={"synthesise": synthesis},
        text_store=store,
    )

    result = asyncio.run(runner.run(request_for(graph_document, ArmCode.MID)))

    grouping = runner.synthesis_for(result.run_id)
    assert grouping is not None and len(grouping.groups) == 1

    now["value"] = 61.0
    assert runner.synthesis_for(result.run_id) is None
    assert runner._runtime.worksheet_for(result.run_id, result.call_units[0]) is None


# ── quote resolution ──

QUOTE_PAYMENT = "płatności czynszu"
QUOTE_PAYMENT_START = DOCUMENT_TEXT.index(QUOTE_PAYMENT)
QUOTE_PAYMENT_END = QUOTE_PAYMENT_START + len(QUOTE_PAYMENT)
QUOTE_UNIT_OPENING = "§ 1. Najemca"
QUOTE_UNIT_OPENING_START = DOCUMENT_TEXT.index(QUOTE_UNIT_OPENING)
QUOTE_UNIT_OPENING_END = QUOTE_UNIT_OPENING_START + len(QUOTE_UNIT_OPENING)
AMBIGUOUS_DOCUMENT_TEXT = "§ 1. alpha beta gamma. alpha beta delta.\n"
PAGE_TWO_START = DOCUMENT_TEXT.index("§ 3.")


def _runner_quoting(
    document: GraphDocument,
    built_corpus: QdrantCorpusIndex,
    tmp_path: Path,
    *,
    quote: str,
    db_name: str = "quote-anchor.sqlite3",
) -> AnalysisRunner:
    return build_runner(
        document.session,
        built_corpus,
        tmp_path,
        db_name=db_name,
        script={"verifier.relevance": relevance_quoting(quote)},
    )


def _unit_for(session: DocumentSession, unit_id: str):
    return next(unit for unit in session.units if unit.id == unit_id)


def _assert_matches_unit_anchor(finding: Any, session: DocumentSession) -> None:
    unit = _unit_for(session, finding.unit_id)
    assert finding.start_offset == unit.anchor.start_offset
    assert finding.end_offset == unit.anchor.end_offset


def _ambiguous_document() -> GraphDocument:
    session = make_session(AMBIGUOUS_DOCUMENT_TEXT)
    return GraphDocument(
        session=session,
        whole_document_id=session.whole_document_id,
        resolved_target_id=session.units[0].id,
    )


def _expected_page(offset: int) -> int:
    return 1 if offset < PAGE_TWO_START else 2


def _paged_document() -> GraphDocument:
    """Payload shaped like PDF ingestion: one anchor per word, with page and bbox."""
    anchors = tuple(
        SourceAnchor(
            start_offset=match.start(),
            end_offset=match.end(),
            read_mode=ReadMode.NATIVE_PDF,
            page=_expected_page(match.start()),
            bbox=(
                50.0 + float(match.start()),
                100.0 * _expected_page(match.start()),
                50.0 + float(match.end()),
                100.0 * _expected_page(match.start()) + 12.0,
            ),
        )
        for match in re.finditer(r"\S+", DOCUMENT_TEXT)
    )
    payload = DocumentPayload(
        document_id=uuid4(),
        text=DOCUMENT_TEXT,
        anchors=anchors,
        read_mode=ReadMode.NATIVE_PDF,
        content_hash=content_hash(DOCUMENT_TEXT),
    )
    units = segment(payload, RunConfig())
    session = DocumentSession(
        document_id=payload.document_id,
        payload=payload,
        units=tuple(units),
        references=tuple(parse_references(units)),
    )
    return GraphDocument(
        session=session,
        whole_document_id=session.whole_document_id,
        resolved_target_id=units[0].id,
    )


def test_off_finding_resolves_quote_to_document_subspan(
    graph_document: GraphDocument, built_corpus: QdrantCorpusIndex, tmp_path: Path
) -> None:
    runner = _runner_quoting(
        graph_document,
        built_corpus,
        tmp_path,
        quote=QUOTE_PAYMENT,
        db_name="off-quote-subspan.sqlite3",
    )

    result = asyncio.run(runner.run(request_for(graph_document, ArmCode.OFF)))

    findings = _relevant_findings(result)
    assert findings
    for finding in findings:
        assert finding.start_offset == QUOTE_PAYMENT_START
        assert finding.end_offset == QUOTE_PAYMENT_END


def test_quote_surrounding_whitespace_resolves_to_same_offsets(
    graph_document: GraphDocument, built_corpus: QdrantCorpusIndex, tmp_path: Path
) -> None:
    runner = _runner_quoting(
        graph_document,
        built_corpus,
        tmp_path,
        quote=f"\n  {QUOTE_PAYMENT}\n",
        db_name="quote-surrounding-whitespace.sqlite3",
    )

    result = asyncio.run(runner.run(request_for(graph_document, ArmCode.MID)))

    unit_one = graph_document.session.units[0]
    matched = [
        finding
        for finding in _relevant_findings(result)
        if finding.unit_id == unit_one.id
    ]
    assert matched
    for finding in matched:
        assert finding.start_offset == QUOTE_PAYMENT_START
        assert finding.end_offset == QUOTE_PAYMENT_END


def test_quote_at_window_start_resolves_with_leading_whitespace(
    graph_document: GraphDocument, built_corpus: QdrantCorpusIndex, tmp_path: Path
) -> None:
    runner = _runner_quoting(
        graph_document,
        built_corpus,
        tmp_path,
        quote=f"\n{QUOTE_UNIT_OPENING}",
        db_name="quote-window-start.sqlite3",
    )

    result = asyncio.run(runner.run(request_for(graph_document, ArmCode.MID)))

    unit_one = graph_document.session.units[0]
    matched = [
        finding
        for finding in _relevant_findings(result)
        if finding.unit_id == unit_one.id
    ]
    assert matched
    for finding in matched:
        assert finding.start_offset == QUOTE_UNIT_OPENING_START
        assert finding.end_offset == QUOTE_UNIT_OPENING_END


@pytest.mark.parametrize("arm", [ArmCode.MID, ArmCode.ON])
def test_unit_arm_finding_resolves_quote_to_narrower_subspan(
    graph_document: GraphDocument,
    built_corpus: QdrantCorpusIndex,
    tmp_path: Path,
    arm: ArmCode,
) -> None:
    runner = _runner_quoting(
        graph_document,
        built_corpus,
        tmp_path,
        quote=QUOTE_PAYMENT,
        db_name=f"{arm.value}-quote-subspan.sqlite3",
    )

    result = asyncio.run(runner.run(request_for(graph_document, arm)))

    unit_one = graph_document.session.units[0]
    matched = [
        finding
        for finding in _relevant_findings(result)
        if finding.unit_id == unit_one.id
    ]
    assert matched
    for finding in matched:
        assert finding.start_offset >= unit_one.anchor.start_offset
        assert finding.end_offset <= unit_one.anchor.end_offset
        assert (
            finding.start_offset > unit_one.anchor.start_offset
            or finding.end_offset < unit_one.anchor.end_offset
        )
        assert finding.start_offset == QUOTE_PAYMENT_START
        assert finding.end_offset == QUOTE_PAYMENT_END


@pytest.mark.parametrize("arm", [ArmCode.MID, ArmCode.ON])
def test_unresolvable_quote_falls_back_to_unit_anchor(
    graph_document: GraphDocument,
    built_corpus: QdrantCorpusIndex,
    tmp_path: Path,
    arm: ArmCode,
) -> None:
    runner = _runner_quoting(
        graph_document,
        built_corpus,
        tmp_path,
        quote="___no_such_passage___",
        db_name=f"{arm.value}-quote-fallback.sqlite3",
    )

    result = asyncio.run(runner.run(request_for(graph_document, arm)))

    findings = _relevant_findings(result)
    assert findings
    for finding in findings:
        _assert_matches_unit_anchor(finding, graph_document.session)


def test_unresolvable_quote_off_stays_unresolved(
    graph_document: GraphDocument, built_corpus: QdrantCorpusIndex, tmp_path: Path
) -> None:
    """The document as a whole is not a place a reader can be sent to."""
    runner = _runner_quoting(
        graph_document,
        built_corpus,
        tmp_path,
        quote="___no_such_passage___",
        db_name="off-quote-unresolved.sqlite3",
    )

    result = asyncio.run(runner.run(request_for(graph_document, ArmCode.OFF)))

    findings = _relevant_findings(result)
    assert findings
    for finding in findings:
        assert finding.start_offset is None
        assert finding.end_offset is None
        assert finding.page is None
        assert finding.bbox is None


def test_ambiguous_quote_mid_falls_back_to_unit_anchor_not_first_match(
    built_corpus: QdrantCorpusIndex, tmp_path: Path
) -> None:
    document = _ambiguous_document()
    first_match_start = AMBIGUOUS_DOCUMENT_TEXT.index("alpha beta")

    runner = _runner_quoting(
        document,
        built_corpus,
        tmp_path,
        quote="alpha beta",
        db_name="ambiguous-quote-mid.sqlite3",
    )

    result = asyncio.run(runner.run(request_for(document, ArmCode.MID)))

    findings = _relevant_findings(result)
    assert findings
    for finding in findings:
        unit = _unit_for(document.session, finding.unit_id)
        assert finding.start_offset == unit.anchor.start_offset
        assert finding.end_offset == unit.anchor.end_offset
        assert finding.start_offset != first_match_start


def test_whitespace_tolerant_quote_resolution(
    graph_document: GraphDocument, built_corpus: QdrantCorpusIndex, tmp_path: Path
) -> None:
    canonical = "się do płatności czynszu"
    canonical_start = DOCUMENT_TEXT.index(canonical)

    runner = _runner_quoting(
        graph_document,
        built_corpus,
        tmp_path,
        quote="się\ndo płatności czynszu",
        db_name="whitespace-quote.sqlite3",
    )

    result = asyncio.run(runner.run(request_for(graph_document, ArmCode.MID)))

    unit_one = graph_document.session.units[0]
    matched = [
        finding
        for finding in _relevant_findings(result)
        if finding.unit_id == unit_one.id
    ]
    assert matched
    for finding in matched:
        assert finding.start_offset == canonical_start
        assert finding.end_offset == canonical_start + len(canonical)


def test_resolved_subspan_takes_page_from_offset_and_carries_no_bbox(
    built_corpus: QdrantCorpusIndex, tmp_path: Path
) -> None:
    document = _paged_document()
    runner = _runner_quoting(
        document,
        built_corpus,
        tmp_path,
        quote=QUOTE_PAYMENT,
        db_name="resolved-subspan-bbox.sqlite3",
    )

    result = asyncio.run(runner.run(request_for(document, ArmCode.MID)))

    quoted_unit = document.session.units[0]
    matched = [
        finding
        for finding in _relevant_findings(result)
        if finding.unit_id == quoted_unit.id
    ]
    assert matched
    assert any(anchor.bbox is not None for anchor in document.session.payload.anchors)
    for finding in matched:
        assert finding.start_offset == QUOTE_PAYMENT_START
        assert finding.page == _expected_page(QUOTE_PAYMENT_START)
        assert finding.bbox is None


def test_unresolvable_quote_mid_takes_page_from_unit_offset(
    built_corpus: QdrantCorpusIndex, tmp_path: Path
) -> None:
    document = _paged_document()
    runner = _runner_quoting(
        document,
        built_corpus,
        tmp_path,
        quote="___no_such_passage___",
        db_name="mid-quote-fallback-page.sqlite3",
    )

    result = asyncio.run(runner.run(request_for(document, ArmCode.MID)))

    findings = _relevant_findings(result)
    assert findings
    for finding in findings:
        _assert_matches_unit_anchor(finding, document.session)
        assert finding.page == _expected_page(finding.start_offset)
    assert {finding.page for finding in findings} == {1, 2}


@pytest.mark.parametrize(
    ("quote", "resolution"),
    [
        ("   ", QuoteResolution.QUOTE_EMPTY),
        ("___this_text_does_not_exist___", QuoteResolution.QUOTE_FABRICATED),
    ],
)
def test_unusable_quotes_are_recorded_as_such(
    graph_document: GraphDocument,
    built_corpus: QdrantCorpusIndex,
    tmp_path: Path,
    quote: str,
    resolution: QuoteResolution,
) -> None:
    runner = _runner_quoting(
        graph_document,
        built_corpus,
        tmp_path,
        quote=quote,
        db_name=f"{resolution.value}-recorded.sqlite3",
    )

    result = asyncio.run(runner.run(request_for(graph_document, ArmCode.OFF)))

    findings = _relevant_findings(result)
    assert findings
    assert findings[0].quote_resolution == resolution


def test_ambiguous_quote_is_recorded_as_ambiguous(
    built_corpus: QdrantCorpusIndex, tmp_path: Path
) -> None:
    document = _ambiguous_document()
    runner = _runner_quoting(
        document,
        built_corpus,
        tmp_path,
        quote="alpha beta",
        db_name="ambiguous-quote-recorded.sqlite3",
    )

    result = asyncio.run(runner.run(request_for(document, ArmCode.OFF)))

    findings = _relevant_findings(result)
    assert findings
    assert findings[0].quote_resolution == QuoteResolution.QUOTE_AMBIGUOUS


def test_quote_text_is_not_persisted(
    graph_document: GraphDocument, built_corpus: QdrantCorpusIndex, tmp_path: Path
) -> None:
    sentinel = "z9_quote_sentinel_not_in_source_z9"
    db_name = "quote-not-persisted.sqlite3"
    runner = _runner_quoting(
        graph_document, built_corpus, tmp_path, quote=sentinel, db_name=db_name
    )

    result = asyncio.run(runner.run(request_for(graph_document, ArmCode.MID)))

    findings = _relevant_findings(result)
    assert findings
    for finding in findings:
        assert finding.excerpt is None
    assert sentinel not in collect_retained_plaintext(runner)
    assert sentinel.encode() not in (tmp_path / db_name).read_bytes()


def test_not_processed_finding_carries_unit_anchor(
    graph_document: GraphDocument, built_corpus: QdrantCorpusIndex, tmp_path: Path
) -> None:
    clock = TickingClock()
    runner = build_runner(
        graph_document.session,
        built_corpus,
        tmp_path,
        db_name="not-processed-anchor.sqlite3",
        on_turn=clock.tick,
    )

    result = asyncio.run(
        runner.run(
            request_for(
                graph_document,
                ArmCode.MID,
                concurrency=1,
                # One second per model call, so the run runs out before the
                # last unit is decided.
                wall_budget_seconds=6.0,
                clock=clock,
                parameters={"temperature": 0.0, "max_completion_tokens": 2},
            )
        )
    )

    not_processed = [
        finding
        for finding in result.findings
        if finding.code is FindingCode.NOT_PROCESSED
    ]
    assert not_processed
    for finding in not_processed:
        _assert_matches_unit_anchor(finding, graph_document.session)
