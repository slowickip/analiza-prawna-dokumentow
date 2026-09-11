"""The worksheet loop over a real run: the exchange, its bounds, its refusals.

These are the tests the design rests on. They drive whole runs against a scripted
provider and assert on what the roles were allowed to do: that the verifier
adjudicates relevance itself without returning to the researcher, that it cannot
read its way around the researcher, that a refused tool call is answered rather
than fatal, and that a role which never commits fails the run instead of quietly
deciding nothing.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest
from graph_harness import (
    LOCATOR,
    LOCATOR_ART19A,
    TickingClock,
    build_runner,
    make_session,
)
from worksheet_transport import (
    Turn,
    characterising_from_an_unseen_locator,
    deciding_relation,
    deciding_relevance,
    posting,
    posting_no_basis,
    posting_no_basis_without_looking,
    posting_nothing,
    quote_of,
    repeating_one_phrase,
    searching_with_a_refused_phrase,
    silence,
    tool_call,
    tool_calls,
)

from contract_analyzer.agents.retrieval import run_read
from contract_analyzer.agents.runner import AnalysisRunner
from contract_analyzer.agents.session import RunRequest
from contract_analyzer.corpus import QdrantCorpusIndex
from contract_analyzer.domain import ArmCode, FindingCode
from contract_analyzer.storage import RunTextStore

UNSEEN_LOCATOR = "https://api.sejm.gov.pl/eli/acts/DU/2023/725/text.html/arti=9999"

UNIT_TEXT = (
    "§ 1. Najemca składa kaucję zabezpieczającą "
    "i może wypowiedzieć najem zgodnie z ustawą."
)
OTHER_ACT_LOCATOR = "https://api.sejm.gov.pl/eli/acts/DU/2026/795/text.pdf#article=659"
UNSEEN_LOCATOR = "https://api.sejm.gov.pl/eli/acts/DU/2023/725/text.html/arti=99"


def _session():
    session = make_session(UNIT_TEXT)
    return session, session.units[0].id


def _run(
    runner: AnalysisRunner,
    document_id: UUID,
    *,
    arm: ArmCode = ArmCode.MID,
    **overrides: Any,
):
    return runner.run(
        RunRequest(
            document_id=document_id,
            arm=arm,
            measurement_valid=True,
            wall_budget_seconds=300.0,
            concurrency=1,
            **overrides,
        )
    )


def test_repeated_role_reads_share_the_per_run_provision_cache(
    built_corpus: QdrantCorpusIndex,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, unit_id = _session()
    runner = build_runner(session, built_corpus, tmp_path)
    prepared = runner.start_run(
        RunRequest(
            document_id=session.document_id,
            arm=ArmCode.MID,
            measurement_valid=True,
            concurrency=1,
        )
    )
    active = runner._runtime.active(prepared.run_id)
    context = runner._runtime._open_unit(active, unit_id)
    reads = 0
    original_read = built_corpus.read

    def counted_read(locator: str) -> Any:
        nonlocal reads
        reads += 1
        return original_read(locator)

    monkeypatch.setattr(built_corpus, "read", counted_read)

    async def exercise() -> None:
        await run_read(context, "researcher", LOCATOR)
        await run_read(context, "researcher", LOCATOR)

    asyncio.run(exercise())

    assert reads == 1
    assert active.provision_reads == 2


def _codes(result: Any, unit_id: str) -> list[FindingCode]:
    return [finding.code for finding in result.findings if finding.unit_id == unit_id]


def test_two_candidates_one_relevant_produce_one_consistent_finding(
    built_corpus: QdrantCorpusIndex, tmp_path: Path
) -> None:
    """(a) The unit's result is what the verifier accepted, not what was posted."""
    session, unit_id = _session()
    runner = build_runner(
        session,
        built_corpus,
        tmp_path,
        db_name="two-candidates.sqlite3",
        script={
            "researcher.search": posting(LOCATOR, LOCATOR_ART19A),
            "verifier.relevance": deciding_relevance(
                lambda locator: {"relevant": locator == LOCATOR, "raw_confidence": 0.9}
            ),
            "verifier.relation": deciding_relation(lambda _: {"departure": "none"}),
        },
    )

    result = asyncio.run(_run(runner, session.document_id))

    unit_findings = [
        finding for finding in result.findings if finding.unit_id == unit_id
    ]
    assert result.status == "completed"
    assert [finding.code for finding in unit_findings] == [FindingCode.CONSISTENT]
    assert unit_findings[0].legal_locators == (LOCATOR,)
    record = runner._runtime.worksheet_for(result.run_id, unit_id)
    assert record is not None
    assert [entry.kind for entry in record.entries].count("candidate") == 2
    assert [entry.kind for entry in record.entries].count("verdict") == 3


def test_verifier_adjudicates_directly_and_refuses_raise_challenge(
    built_corpus: QdrantCorpusIndex, tmp_path: Path
) -> None:
    """(b) The verifier cannot raise challenges: relevance offers only read and verdict,
    and attempting raise_challenge is refused as an unknown tool. The verifier hands off
    directly to the analyst rather than returning to the researcher."""
    session, unit_id = _session()
    refusals: list[dict[str, Any]] = []
    observed_tasks: list[str] = []

    def verifier_relevance(turn: Turn) -> Mapping[str, Any]:
        if not turn.calls_made("raise_challenge"):
            assert "raise_challenge" not in turn.tools
            assert set(turn.tools) == {"read_provision", "post_verdict"}

            return tool_call(
                "raise_challenge",
                {
                    "missing": "art. 19a ustawy",
                    "reason": "bez niego trafność pozostaje niepewna",
                },
            )
        refusals.extend(result for result in turn.tool_results if "error" in result)
        return tool_call(
            "post_verdict",
            {
                "relevant": True,
                "quote": quote_of(turn),
                "raw_confidence": 0.9,
                "based_on": [],
            },
        )

    runner = build_runner(
        session,
        built_corpus,
        tmp_path,
        db_name="no-challenge-loop.sqlite3",
        script={
            "researcher.search": posting(LOCATOR),
            "verifier.relevance": verifier_relevance,
        },
        on_turn=lambda turn: observed_tasks.append(turn.task),
    )

    result = asyncio.run(_run(runner, session.document_id))
    assert result.status == "completed"
    assert _codes(result, unit_id) == [FindingCode.CONSISTENT]
    record = runner._runtime.worksheet_for(result.run_id, unit_id)
    assert record is not None

    assert [refusal["error"] for refusal in refusals] == ["unknown_tool"]

    kinds = [
        entry.kind for entry in record.entries if entry.kind not in {"search", "read"}
    ]
    assert kinds == [
        "candidate",
        "verdict",
        "character",
        "verdict",
    ]
    assert "challenge" not in kinds
    assert "challenge_answer" not in kinds

    assert observed_tasks == [
        "researcher.search",
        "researcher.search",
        "verifier.relevance",
        "verifier.relevance",
        "analyst.characterise",
        "verifier.relation",
        "synthesise",
        "synthesise",
    ]
    assert "researcher.answer" not in observed_tasks


def test_verifier_reads_only_worksheet_locators_with_one_candidate_cap(
    built_corpus: QdrantCorpusIndex, tmp_path: Path
) -> None:
    """(c) A refused probe counts, and the cap survives into relation."""
    session, unit_id = _session()
    seen: list[dict[str, Any]] = []

    def verifier_relevance(turn: Turn) -> Mapping[str, Any]:
        reads = turn.calls_made("read_provision")
        if reads == 0:
            return tool_calls(
                [
                    ("read_provision", {"locator": LOCATOR}),
                    ("read_provision", {"locator": LOCATOR_ART19A}),
                    ("read_provision", {"locator": LOCATOR}),
                ]
            )
        seen.extend(turn.tool_results)
        return tool_call(
            "post_verdict",
            {
                "relevant": True,
                "quote": quote_of(turn),
                "raw_confidence": 0.9,
                "based_on": [],
            },
        )

    def verifier_relation(turn: Turn) -> Mapping[str, Any]:
        if not turn.calls_made("read_provision"):
            return tool_call("read_provision", {"locator": LOCATOR})
        seen.extend(turn.tool_results)
        return tool_call("post_verdict", {"departure": "none", "based_on": []})

    runner = build_runner(
        session,
        built_corpus,
        tmp_path,
        db_name="verifier-read-guard.sqlite3",
        script={
            "researcher.search": posting(LOCATOR),
            "verifier.relevance": verifier_relevance,
            "verifier.relation": verifier_relation,
        },
    )

    result = asyncio.run(_run(runner, session.document_id))

    assert result.status == "completed"
    assert _codes(result, unit_id) == [FindingCode.CONSISTENT]
    allowed = [item for item in seen if "text" in item]
    refused = [item.get("error") for item in seen if "error" in item]
    assert [item["locator"] for item in allowed] == [LOCATOR, LOCATOR]
    assert refused == ["locator_not_on_worksheet", "read_limit_reached"]
    record = runner._runtime.worksheet_for(result.run_id, unit_id)
    assert record is not None
    reads = [entry for entry in record.entries if entry.kind == "read"]
    assert [entry.locator for entry in reads] == [LOCATOR, LOCATOR]  # type: ignore[union-attr]
    saved = runner.services.metadata.get_run(result.run_id)
    assert saved is not None
    assert saved.provision_reads == 2


def test_a_researcher_that_posts_nothing_ends_the_unit(
    built_corpus: QdrantCorpusIndex, tmp_path: Path
) -> None:
    """(d) An empty shortlist is a result, and the verifier is never asked."""
    session, unit_id = _session()
    tasks: list[str] = []
    runner = build_runner(
        session,
        built_corpus,
        tmp_path,
        db_name="posts-nothing.sqlite3",
        script={"researcher.search": posting_nothing()},
        on_turn=lambda turn: tasks.append(turn.task),
    )

    result = asyncio.run(_run(runner, session.document_id))

    assert _codes(result, unit_id) == [FindingCode.NO_BASIS_FOUND]
    assert not [task for task in tasks if task.startswith("verifier.")]
    record = runner._runtime.worksheet_for(result.run_id, unit_id)
    assert record is not None
    assert record.search_complete is True
    assert [entry.kind for entry in record.entries] == ["search"]


def test_stating_no_basis_ends_the_search_instead_of_spending_its_turns(
    built_corpus: QdrantCorpusIndex, tmp_path: Path
) -> None:
    """An empty commit is the researcher's way to conclude, not to give up.

    Without it the only exit from the search loop was silence, which the model
    reliably declined: in a measured run of 2026-09-04 the three units that found
    no basis spent 72 of the run's 104 searches reaching that result.
    """
    session, unit_id = _session()
    runner = build_runner(
        session,
        built_corpus,
        tmp_path,
        db_name="states-no-basis.sqlite3",
        script={"researcher.search": posting_no_basis()},
    )

    result = asyncio.run(_run(runner, session.document_id))

    assert _codes(result, unit_id) == [FindingCode.NO_BASIS_FOUND]
    record = runner._runtime.worksheet_for(result.run_id, unit_id)
    assert record is not None
    assert record.search_complete is True
    assert [entry.kind for entry in record.entries] == ["search"]
    saved = runner.services.metadata.get_run(result.run_id)
    assert saved is not None
    # The point of the exit: the unit is not recorded as one the budget cut off,
    # and it costs one search turn plus the commit rather than the whole cap. A
    # schema that could not carry an empty list would refuse the commit here and
    # spend a further turn, so the count is what makes this test about the exit.
    assert saved.finder_budget_exhausted_units == 0
    assert saved.finder_tool_turns == 2


def test_no_basis_cannot_be_claimed_before_the_corpus_was_searched(
    built_corpus: QdrantCorpusIndex, tmp_path: Path
) -> None:
    """A floor of zero must not also authorise giving up on the first turn."""
    session, unit_id = _session()
    refusals: list[str] = []

    runner = build_runner(
        session,
        built_corpus,
        tmp_path,
        db_name="no-basis-unlooked.sqlite3",
        script={"researcher.search": posting_no_basis_without_looking()},
        on_turn=lambda turn: refusals.extend(
            str(message.get("content"))
            for message in turn.messages
            if "no_search_before_empty_commit" in str(message.get("content"))
        ),
    )

    result = asyncio.run(_run(runner, session.document_id))

    assert refusals, "the empty commit was accepted without a search"
    assert _codes(result, unit_id) == [FindingCode.NO_BASIS_FOUND]
    record = runner._runtime.worksheet_for(result.run_id, unit_id)
    assert record is not None
    # The refusal sent it back to the corpus, so the conclusion rests on a look.
    assert [entry.kind for entry in record.entries] == ["search"]


def test_each_attempt_names_the_call_unit_that_spent_it(
    built_corpus: QdrantCorpusIndex, tmp_path: Path
) -> None:
    """Without this a run can only report an average, never its critical path."""
    session, unit_id = _session()
    runner = build_runner(
        session,
        built_corpus,
        tmp_path,
        db_name="attempt-unit-id.sqlite3",
        script={"researcher.search": posting(LOCATOR)},
    )

    result = asyncio.run(_run(runner, session.document_id))

    attempts = runner.services.metadata.list_attempts(result.run_id)
    by_unit = {
        attempt.prompt_version: attempt.unit_id
        for attempt in attempts
        if attempt.status == "success"
    }
    assert by_unit["researcher.search"] == unit_id
    assert by_unit["verifier.relevance"] == unit_id
    # The synthesis is not about one unit, and saying it was would be a false
    # attribution rather than a missing one.
    assert by_unit["synthesise"] is None


def test_a_repeated_search_phrase_is_counted_as_a_cache_hit(
    built_corpus: QdrantCorpusIndex, tmp_path: Path
) -> None:
    """The cache makes a repeated phrase free; an uncounted saving is uncheckable."""
    session, _ = _session()
    runner = build_runner(
        session,
        built_corpus,
        tmp_path,
        db_name="cache-hits.sqlite3",
        script={"researcher.search": repeating_one_phrase()},
    )

    result = asyncio.run(_run(runner, session.document_id))

    saved = runner.services.metadata.get_run(result.run_id)
    assert saved is not None
    # Two asks, one corpus call: the second was answered from the run's cache.
    assert saved.finder_search_calls == 2
    assert saved.retrieval_cache_hits == 1


def test_a_refused_phrase_is_not_a_search_and_cannot_ground_no_basis(
    built_corpus: QdrantCorpusIndex, tmp_path: Path
) -> None:
    """The guard has to mean "looked at the corpus", not "called the tool".

    A phrase that sanitises to nothing is refused before any corpus call, yet the
    refusal payload still carries an empty results list. Recording that as a search
    let the researcher conclude that the corpus holds no basis without ever
    querying it.
    """
    session, unit_id = _session()
    refusals: list[str] = []
    runner = build_runner(
        session,
        built_corpus,
        tmp_path,
        db_name="refused-phrase.sqlite3",
        script={"researcher.search": searching_with_a_refused_phrase()},
        on_turn=lambda turn: refusals.extend(
            str(message.get("content"))
            for message in turn.messages
            if "no_search_before_empty_commit" in str(message.get("content"))
        ),
    )

    result = asyncio.run(_run(runner, session.document_id))

    assert refusals, "an unexecuted query was accepted as a search"
    record = runner._runtime.worksheet_for(result.run_id, unit_id)
    assert record is not None
    assert [entry.kind for entry in record.entries] == []
    saved = runner.services.metadata.get_run(result.run_id)
    assert saved is not None
    assert saved.finder_search_calls == 0


def test_the_analyst_cannot_reach_a_provision_the_researcher_never_retrieved(
    built_corpus: QdrantCorpusIndex, tmp_path: Path
) -> None:
    """ "No search tool" is not the same as "only researcher-retrieved evidence".

    The analyst keeps read_provision so it can open the candidate it was handed.
    Left unbounded, that read reached any valid locator and entered it into the
    shared ledger, so the role could manufacture its own evidence and cite it.
    """
    session, unit_id = _session()
    refusals: list[str] = []
    runner = build_runner(
        session,
        built_corpus,
        tmp_path,
        db_name="analyst-unseen-locator.sqlite3",
        script={
            "researcher.search": posting(LOCATOR),
            "analyst.characterise": characterising_from_an_unseen_locator(
                UNSEEN_LOCATOR
            ),
        },
        on_turn=lambda turn: refusals.extend(
            str(message.get("content"))
            for message in turn.messages
            if "locator_not_in_provenance_ledger" in str(message.get("content"))
        ),
    )

    result = asyncio.run(_run(runner, session.document_id))

    assert refusals, "the analyst opened a provision outside the ledger"
    record = runner._runtime.worksheet_for(result.run_id, unit_id)
    assert record is not None
    reads = [e for e in record.entries if e.kind == "read"]
    assert all(
        e.locator != UNSEEN_LOCATOR  # type: ignore[union-attr]
        for e in reads
    ), "the refused locator still reached the record"


def test_a_verifier_that_never_commits_fails_the_run(
    built_corpus: QdrantCorpusIndex, tmp_path: Path
) -> None:
    """(e) One nudge, then the run fails as it would on an unusable answer."""
    session, _ = _session()
    nudges: list[str] = []

    def never_commits(turn: Turn) -> Mapping[str, Any]:
        nudges.extend(
            str(message.get("content"))
            for message in turn.messages[-1:]
            if message.get("role") == "user"
            and "Zakończ" in str(message.get("content"))
        )
        return silence()

    runner = build_runner(
        session,
        built_corpus,
        tmp_path,
        db_name="no-commit.sqlite3",
        script={
            "researcher.search": posting(LOCATOR),
            "verifier.relevance": never_commits,
        },
    )

    result = asyncio.run(_run(runner, session.document_id))

    assert result.status == "failed"
    assert result.error_code == "model_response_schema_invalid"
    assert len(nudges) == 1
    assert "post_verdict" in nudges[0]


@pytest.mark.parametrize("ending", ["turn_cap", "silence_after_nudge"])
def test_an_unusable_characterisation_defaults_and_reaches_an_uncertain_finding(
    ending: str,
    built_corpus: QdrantCorpusIndex,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    session, unit_id = _session()

    def never_commits(turn: Turn) -> Mapping[str, Any]:
        if ending == "turn_cap":
            return tool_call("read_provision", {"locator": turn.locator})
        return silence()

    def relation(turn: Turn) -> Mapping[str, Any]:
        assert turn.payload["basis_character"] == "undetermined"
        return tool_call(
            "post_verdict",
            {"departure": "none", "raw_confidence": 0.4, "based_on": []},
        )

    runner = build_runner(
        session,
        built_corpus,
        tmp_path,
        db_name=f"defaulted-character-{ending}.sqlite3",
        script={
            "researcher.search": posting(LOCATOR),
            "analyst.characterise": never_commits,
            "verifier.relation": relation,
        },
    )
    prepared = runner.start_run(
        RunRequest(
            document_id=session.document_id,
            arm=ArmCode.MID,
            measurement_valid=True,
            wall_budget_seconds=300.0,
            concurrency=1,
        )
    )
    active = runner._runtime.active(prepared.run_id)
    caplog.set_level(logging.INFO)

    result = asyncio.run(runner.execute_run(prepared))

    assert result.status == "completed"
    assert _codes(result, unit_id) == [FindingCode.UNCERTAIN]
    record = runner._runtime.worksheet_for(result.run_id, unit_id)
    assert record is not None
    character = next(entry for entry in record.entries if entry.kind == "character")
    assert character.author == "system"
    assert character.character.kind == "undetermined"  # type: ignore[union-attr]
    assert (  # type: ignore[union-attr]
        character.character.undetermined_reason == "unusable_model_answer"
    )
    assert active.defaulted_characterisations == 1
    assert active.spend().defaulted_characterisations == 1
    # On the record, not only in the log: this counter is what separates a run
    # that contained a failure from one whose analyst genuinely could not settle
    # the character, and the finding keeps no trace of the difference.
    stored = runner.services.metadata.get_run(result.run_id)
    assert stored is not None
    assert stored.defaulted_characterisations == 1
    assert f"run={result.run_id}" in caplog.text
    assert f"unit={unit_id}" in caplog.text
    assert f"candidate={character.candidate_id}" in caplog.text  # type: ignore[union-attr]
    assert "defaulted_characterisations=1" in caplog.text


def test_invalid_commit_arguments_are_returned_to_the_model(
    built_corpus: QdrantCorpusIndex, tmp_path: Path
) -> None:
    """(f) A malformed commit is a correctable mistake, not a dead run."""
    session, unit_id = _session()
    refusals: list[dict[str, Any]] = []

    def relevance(turn: Turn) -> Mapping[str, Any]:
        if not turn.calls_made("post_verdict"):
            return tool_call(
                "post_verdict",
                {"relevant": True},
            )
        refusals.extend(result for result in turn.tool_results if "error" in result)
        return tool_call(
            "post_verdict",
            {
                "relevant": True,
                "quote": quote_of(turn),
                "raw_confidence": 0.9,
                "based_on": [],
            },
        )

    runner = build_runner(
        session,
        built_corpus,
        tmp_path,
        db_name="invalid-arguments.sqlite3",
        script={
            "researcher.search": posting(LOCATOR),
            "verifier.relevance": relevance,
        },
    )

    result = asyncio.run(_run(runner, session.document_id))

    assert result.status == "completed"
    assert _codes(result, unit_id) == [FindingCode.CONSISTENT]
    assert [refusal["error"] for refusal in refusals] == ["invalid_arguments"]
    assert "quote" in str(refusals[0]["detail"])


def test_a_memorised_locator_absent_from_the_ledger_is_refused(
    built_corpus: QdrantCorpusIndex, tmp_path: Path
) -> None:
    """A real corpus locator is not authority unless a tool returned it."""
    session, unit_id = _session()
    refusals: list[dict[str, Any]] = []

    def search(turn: Turn) -> Mapping[str, Any]:
        if not turn.calls_made("post_candidates"):
            return tool_call(
                "post_candidates",
                {"candidates": [{"locator": LOCATOR, "why": "zapamiętany"}]},
            )
        refusals.extend(result for result in turn.tool_results if "error" in result)
        return silence()

    runner = build_runner(
        session,
        built_corpus,
        tmp_path,
        db_name="refused-candidate.sqlite3",
        script={"researcher.search": search},
    )

    result = asyncio.run(_run(runner, session.document_id))

    assert result.status == "completed"
    assert [refusal["error"] for refusal in refusals] == [
        "locator_not_in_provenance_ledger"
    ]
    record = runner._runtime.worksheet_for(result.run_id, unit_id)
    assert record is not None
    assert [
        entry.locator  # type: ignore[union-attr]
        for entry in record.entries
        if entry.kind == "candidate"
    ] == []


def test_a_read_is_provenance_as_much_as_a_search(
    built_corpus: QdrantCorpusIndex, tmp_path: Path
) -> None:
    """The ledger records what a corpus tool returned, and a read is a corpus tool.

    The analyst may therefore name a provision the search did not surface, as long
    as it opened it and the corpus answered. What it may not do is cite a locator
    no tool ever answered for, which the two tests around this one hold to.
    """
    session, unit_id = _session()

    def search(turn: Turn) -> Mapping[str, Any]:
        if not turn.calls_made("read_provision"):
            return tool_call("read_provision", {"locator": LOCATOR})
        return tool_call(
            "post_candidates",
            {"candidates": [{"locator": LOCATOR, "why": "odczytany, nie wyszukany"}]},
        )

    runner = build_runner(
        session,
        built_corpus,
        tmp_path,
        db_name="read-provenance.sqlite3",
        script={"researcher.search": search},
    )

    result = asyncio.run(_run(runner, session.document_id))

    record = runner._runtime.worksheet_for(result.run_id, unit_id)
    assert record is not None
    assert [
        entry.locator  # type: ignore[union-attr]
        for entry in record.entries
        if entry.kind == "candidate"
    ] == [LOCATOR]
    assert not [entry for entry in record.entries if entry.kind == "search"]


def test_one_unseen_locator_refuses_the_whole_candidate_batch(
    built_corpus: QdrantCorpusIndex, tmp_path: Path
) -> None:
    """A batch is one commit: one locator outside the ledger saves none of them."""
    session, unit_id = _session()
    refusals: list[dict[str, Any]] = []

    def search(turn: Turn) -> Mapping[str, Any]:
        if not turn.calls_made("search_corpus"):
            return tool_call("search_corpus", {"phrase": "kaucja najem"})
        if not turn.calls_made("post_candidates"):
            return tool_call(
                "post_candidates",
                {
                    "candidates": [
                        {"locator": LOCATOR, "why": "znaleziony"},
                        {"locator": UNSEEN_LOCATOR, "why": "zapamiętany"},
                    ]
                },
            )
        refusals.extend(result for result in turn.tool_results if "error" in result)
        return silence()

    runner = build_runner(
        session,
        built_corpus,
        tmp_path,
        db_name="refused-batch.sqlite3",
        script={"researcher.search": search},
    )

    result = asyncio.run(_run(runner, session.document_id))

    assert result.status == "completed"
    assert [refusal["error"] for refusal in refusals] == [
        "locator_not_in_provenance_ledger"
    ]
    record = runner._runtime.worksheet_for(result.run_id, unit_id)
    assert record is not None
    assert [entry.kind for entry in record.entries].count("candidate") == 0


def test_a_mixed_commit_turn_is_rejected_without_side_effects(
    built_corpus: QdrantCorpusIndex, tmp_path: Path
) -> None:
    session, unit_id = _session()
    refusals: list[dict[str, Any]] = []

    def search(turn: Turn) -> Mapping[str, Any]:
        if not turn.calls_made("post_candidates"):
            return tool_calls(
                [
                    ("search_corpus", {"phrase": "kaucja"}),
                    (
                        "post_candidates",
                        {"candidates": [{"locator": LOCATOR, "why": "w"}]},
                    ),
                ]
            )
        refusals.extend(result for result in turn.tool_results if "error" in result)
        return silence()

    runner = build_runner(
        session,
        built_corpus,
        tmp_path,
        db_name="sixth-candidate.sqlite3",
        script={"researcher.search": search},
    )

    result = asyncio.run(_run(runner, session.document_id))

    assert [refusal["error"] for refusal in refusals] == [
        "commit_must_be_alone",
        "commit_must_be_alone",
    ]
    record = runner._runtime.worksheet_for(result.run_id, unit_id)
    assert record is not None
    assert record.entries == ()


def test_the_worksheet_is_retained_and_purged_with_the_document(
    built_corpus: QdrantCorpusIndex, tmp_path: Path
) -> None:
    """(g) The record outlives the unit only inside the retention window."""
    session, unit_id = _session()
    text_store = RunTextStore()
    runner = build_runner(
        session,
        built_corpus,
        tmp_path,
        db_name="worksheet-retention.sqlite3",
        text_store=text_store,
        script={"researcher.search": posting(LOCATOR)},
    )

    result = asyncio.run(_run(runner, session.document_id))

    record = runner._runtime.worksheet_for(result.run_id, unit_id)
    assert record is not None
    assert record.unit_id == unit_id
    assert any(entry.kind == "candidate" for entry in record.entries)

    runner.purge_document_text(session.document_id)

    assert runner._runtime.worksheet_for(result.run_id, unit_id) is None


def test_neither_driver_results_nor_events_carry_model_prose(
    built_corpus: QdrantCorpusIndex,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(g) Prose stays on the worksheet; the wire carries identifiers and kinds."""
    from contract_analyzer.storage import EventBus

    session, unit_id = _session()
    events = EventBus()
    outcomes: list[object] = []
    runner = build_runner(
        session,
        built_corpus,
        tmp_path,
        db_name="no-prose.sqlite3",
        events=events,
        script={"researcher.search": posting(LOCATOR)},
    )
    original_process_unit = runner._runtime.process_unit

    async def capture_outcome(run_id: UUID, selected_unit_id: str):
        outcome = await original_process_unit(run_id, selected_unit_id)
        outcomes.append(outcome)
        return outcome

    monkeypatch.setattr(runner._runtime, "process_unit", capture_outcome)

    async def exercise() -> None:
        prepared = runner.start_run(
            RunRequest(
                document_id=session.document_id,
                arm=ArmCode.MID,
                measurement_valid=True,
                wall_budget_seconds=300.0,
                concurrency=1,
            )
        )
        assert runner.services.events is not None
        queue = runner.services.events.subscribe(prepared.run_id)
        await runner.execute_run(prepared)
        published = []
        while True:
            try:
                published.append(queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        worksheet_events = [event for event in published if event.kind == "worksheet"]
        assert worksheet_events
        assert {event.unit_id for event in worksheet_events} == {unit_id}
        assert {event.role for event in worksheet_events} <= {
            "researcher",
            "analyst",
            "verifier",
        }
        assert {event.entry_kind for event in worksheet_events} <= {
            "search",
            "candidate",
            "verdict",
            "character",
            "read",
        }
        serialised = json.dumps(
            [event.model_dump(mode="json") for event in published]
            + [str(outcome) for outcome in outcomes],
            ensure_ascii=False,
        )
        for fragment in (UNIT_TEXT[:30], "kaucja najem", "Przepis dotyczy najmu."):
            assert fragment not in serialised

    asyncio.run(exercise())


def test_budget_exhaustion_mid_unit_keeps_what_was_decided(
    built_corpus: QdrantCorpusIndex, tmp_path: Path
) -> None:
    """(i) A run cut short is a partial measurement, not an empty one."""
    text = UNIT_TEXT + "\n\n§ 2. Wynajmujący wydaje lokal najemcy.\n"
    session = make_session(text)
    clock = TickingClock()
    runner = build_runner(
        session,
        built_corpus,
        tmp_path,
        db_name="mid-unit-budget.sqlite3",
        on_turn=clock.tick,
        script={
            "researcher.search": posting(LOCATOR, LOCATOR_ART19A),
            "verifier.relevance": deciding_relevance(
                lambda _: {"relevant": True, "raw_confidence": 0.9}
            ),
        },
    )

    async def exercise() -> None:
        prepared = runner.start_run(
            RunRequest(
                document_id=session.document_id,
                arm=ArmCode.MID,
                measurement_valid=True,
                # The clock moves one second per model call. Six calls carry the
                # first candidate to a finding; the wall budget then stops the
                # unit before the second candidate's verdict.
                wall_budget_seconds=6.0,
                clock=clock,
                concurrency=1,
                parameters={"temperature": 0.0, "max_completion_tokens": 2},
            )
        )
        active = runner._runtime.active(prepared.run_id)
        result = await runner.execute_run(prepared)

        assert result.interruption is True
        codes = {finding.code for finding in result.findings}
        assert FindingCode.CONSISTENT in codes
        assert FindingCode.NOT_PROCESSED in codes
        assert active.units == {}

    asyncio.run(exercise())


def test_the_verifier_payload_shows_one_candidate_and_no_searches(
    built_corpus: QdrantCorpusIndex, tmp_path: Path
) -> None:
    """(k) What the verifier is shown is the whole basis of its verdict."""
    session, _ = _session()
    payloads: list[dict[str, Any]] = []

    def capture(turn: Turn) -> None:
        if turn.task.startswith("verifier."):
            payloads.append(turn.payload)

    runner = build_runner(
        session,
        built_corpus,
        tmp_path,
        db_name="verifier-payload.sqlite3",
        script={
            "researcher.search": posting(LOCATOR, LOCATOR_ART19A),
            "verifier.relevance": deciding_relevance(
                lambda locator: {"relevant": locator == LOCATOR, "raw_confidence": 0.9}
            ),
        },
        on_turn=capture,
    )

    asyncio.run(_run(runner, session.document_id))

    assert payloads
    for payload in payloads:
        assert "candidate_id" not in payload
        worksheet = payload["worksheet"]
        candidates = [entry for entry in worksheet if entry["kind"] == "candidate"]
        assert len(candidates) == 1
        assert candidates[0]["locator"] == payload["candidate_locator"]
        assert all(entry["kind"] != "search" for entry in worksheet)
        assert all(entry["kind"] != "read" for entry in worksheet)
        assert (
            LOCATOR_ART19A not in json.dumps(worksheet)
            or payload["candidate_locator"] == LOCATOR_ART19A
        )
