"""The researcher's search task: what it may do, what it is refused, what it costs.

The role that searches is the only one that can reach the corpus, so its bounds
are what keep a unit's cost bounded. These tests drive the search task over a real
run and read back both the worksheet it wrote and the counters the run recorded.
"""

from __future__ import annotations

import asyncio
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from uuid import uuid4

import pytest
from graph_harness import (
    LOCATOR,
    LOCATOR_ART19A,
    build_runner,
    make_session,
)
from worksheet_transport import (
    Turn,
    deciding_relevance,
    posting,
    silence,
    tool_call,
    tool_calls,
)

from contract_analyzer.agents.prompts import TASK_PROMPTS, load_prompt_bundle
from contract_analyzer.agents.scheduler import (
    FINDER_MAX_PHRASE_CHARS,
    FINDER_MAX_PHRASES_PER_TURN,
    FINDER_SNIPPET_CHARS,
    RETRIEVAL_TOP_K,
)
from contract_analyzer.agents.session import RunRequest
from contract_analyzer.corpus import QdrantCorpusIndex
from contract_analyzer.domain import ArmCode, FindingCode
from contract_analyzer.domain.records import ForceScope, ForceState, ForceValue

DOCUMENT_TEXT = "§ 1. Najemca zobowiązuje się do płatności czynszu."


def _session():
    session = make_session(DOCUMENT_TEXT)
    return session, session.units[0].id


def _run(runner: Any, document_id: Any, arm: ArmCode = ArmCode.MID):
    return runner.run(
        RunRequest(
            document_id=document_id,
            arm=arm,
            measurement_valid=True,
            wall_budget_seconds=300.0,
            concurrency=1,
        )
    )


def test_the_characterise_prompt_reads_the_candidate_it_is_given() -> None:
    """The characterisation is about one provision, whose text travels with it."""
    prompt = load_prompt_bundle().prompts[TASK_PROMPTS["analyst.characterise"]]

    assert "candidate_text" in prompt
    assert "search_corpus" not in prompt
    assert "read_provision" in prompt


def test_the_characterise_payload_carries_the_candidate_text(
    built_corpus: QdrantCorpusIndex, tmp_path: Path
) -> None:

    session, _ = _session()
    payloads: list[dict[str, Any]] = []

    def capture(turn: Turn) -> None:
        if turn.task == "analyst.characterise":
            payloads.append(turn.payload)

    runner = build_runner(
        session,
        built_corpus,
        tmp_path,
        db_name="basis-finder.sqlite3",
        script={"researcher.search": posting(LOCATOR)},
        on_turn=capture,
    )

    asyncio.run(_run(runner, session.document_id))

    corpus = built_corpus
    assert payloads
    for payload in payloads:
        assert "candidate_id" not in payload
        locator = str(payload["candidate_locator"])
        assert payload["candidate_text"] == corpus.read(locator).text


def test_search_results_are_bounded_snippets_of_the_frozen_corpus(
    built_corpus: QdrantCorpusIndex, tmp_path: Path
) -> None:
    """The analyst is shown enough of a provision to judge it and no more."""
    session, _ = _session()
    results: list[dict[str, Any]] = []

    def search(turn: Turn) -> Any:
        if not turn.calls_made("search_corpus"):
            return tool_call("search_corpus", {"phrase": "czynsz najem"})
        results.extend(turn.tool_results)
        return silence()

    runner = build_runner(
        session,
        built_corpus,
        tmp_path,
        db_name="search-snippets.sqlite3",
        script={"researcher.search": search},
    )

    asyncio.run(_run(runner, session.document_id))

    assert results
    hits = results[0]["results"]
    assert 0 < len(hits) <= RETRIEVAL_TOP_K
    for hit in hits:
        assert set(hit) == {"locator", "act_identifier", "snippet"}
        assert len(hit["snippet"]) <= FINDER_SNIPPET_CHARS


def test_an_over_long_phrase_is_refused_to_the_model(
    built_corpus: QdrantCorpusIndex, tmp_path: Path
) -> None:
    """The embedding service has a defined input contract, so a phrase is a phrase."""
    session, _ = _session()
    refusals: list[dict[str, Any]] = []

    def search(turn: Turn) -> Any:
        if not turn.calls_made("search_corpus"):
            return tool_call(
                "search_corpus", {"phrase": "czynsz " * FINDER_MAX_PHRASE_CHARS}
            )
        refusals.extend(result for result in turn.tool_results if "error" in result)
        return silence()

    runner = build_runner(
        session,
        built_corpus,
        tmp_path,
        db_name="over-long-phrase.sqlite3",
        script={"researcher.search": search},
    )

    result = asyncio.run(_run(runner, session.document_id))

    assert result.status == "completed"
    assert [refusal["error"] for refusal in refusals] == ["phrase_too_long"]


def test_only_the_first_phrases_of_a_turn_are_searched(
    built_corpus: QdrantCorpusIndex, tmp_path: Path
) -> None:
    """The per-turn ceiling is what keeps one turn from becoming a whole run."""
    session, _ = _session()
    refusals: list[dict[str, Any]] = []
    phrases = FINDER_MAX_PHRASES_PER_TURN + 2

    def search(turn: Turn) -> Any:
        if not turn.calls_made("search_corpus"):
            return tool_calls(
                [
                    ("search_corpus", {"phrase": f"czynsz {index}"})
                    for index in range(phrases)
                ]
            )
        refusals.extend(result for result in turn.tool_results if "error" in result)
        return silence()

    runner = build_runner(
        session,
        built_corpus,
        tmp_path,
        db_name="phrases-per-turn.sqlite3",
        script={"researcher.search": search},
    )

    result = asyncio.run(_run(runner, session.document_id))

    saved = runner.services.metadata.get_run(result.run_id)
    assert saved is not None
    assert saved.finder_search_calls == FINDER_MAX_PHRASES_PER_TURN
    assert [refusal["error"] for refusal in refusals] == [
        "phrase_budget_exceeded_for_this_turn"
    ] * (phrases - FINDER_MAX_PHRASES_PER_TURN)


def test_the_worksheet_records_what_the_researcher_searched_and_found(
    built_corpus: QdrantCorpusIndex, tmp_path: Path
) -> None:
    session, unit_id = _session()
    runner = build_runner(
        session,
        built_corpus,
        tmp_path,
        db_name="search-worksheet.sqlite3",
        script={"researcher.search": posting(LOCATOR, phrase="kaucja najem")},
    )

    result = asyncio.run(_run(runner, session.document_id))

    record = runner._runtime.worksheet_for(result.run_id, unit_id)
    assert record is not None
    searches = [entry for entry in record.entries if entry.kind == "search"]
    assert len(searches) == 1
    assert searches[0].phrase == "kaucja najem"  # type: ignore[union-attr]
    assert searches[0].result_locators  # type: ignore[union-attr]


def test_reading_a_provision_is_recorded_on_the_worksheet(
    built_corpus: QdrantCorpusIndex, tmp_path: Path
) -> None:
    """A candidate the analyst read is a different claim from one it only ranked."""
    session, unit_id = _session()

    def search(turn: Turn) -> Any:
        if not turn.calls_made("search_corpus"):
            return tool_call("search_corpus", {"phrase": "czynsz"})
        if not turn.calls_made("read_provision"):
            return tool_call("read_provision", {"locator": LOCATOR})
        if not turn.calls_made("post_candidates"):
            return tool_call(
                "post_candidates",
                {
                    "candidates": [
                        {
                            "locator": LOCATOR,
                            "why": "Przepis dotyczy wypowiedzenia najmu.",
                            "supporting_locators": [LOCATOR_ART19A],
                        }
                    ]
                },
            )
        return silence()

    runner = build_runner(
        session,
        built_corpus,
        tmp_path,
        db_name="analyst-read.sqlite3",
        script={"researcher.search": search},
    )

    result = asyncio.run(_run(runner, session.document_id))

    record = runner._runtime.worksheet_for(result.run_id, unit_id)
    assert record is not None
    reads = [entry for entry in record.entries if entry.kind == "read"]
    assert [entry.locator for entry in reads] == [LOCATOR]  # type: ignore[union-attr]
    assert [entry.author for entry in reads] == ["researcher"]
    candidate = next(entry for entry in record.entries if entry.kind == "candidate")
    assert candidate.supporting_locators == (LOCATOR_ART19A,)  # type: ignore[union-attr]
    saved = runner.services.metadata.get_run(result.run_id)
    assert saved is not None
    assert saved.provision_reads == 1


def test_an_unknown_locator_is_refused_to_the_reading_analyst(
    built_corpus: QdrantCorpusIndex, tmp_path: Path
) -> None:
    session, _ = _session()
    refusals: list[dict[str, Any]] = []

    def search(turn: Turn) -> Any:
        if not turn.calls_made("read_provision"):
            return tool_call("read_provision", {"locator": "nie-ma-takiego"})
        refusals.extend(result for result in turn.tool_results if "error" in result)
        return silence()

    runner = build_runner(
        session,
        built_corpus,
        tmp_path,
        db_name="analyst-unknown-read.sqlite3",
        script={"researcher.search": search},
    )

    result = asyncio.run(_run(runner, session.document_id))

    assert result.status == "completed"
    assert [refusal["error"] for refusal in refusals] == ["unknown_locator"]


def test_run_records_what_the_analyst_spent(
    built_corpus: QdrantCorpusIndex, tmp_path: Path
) -> None:
    session, _ = _session()
    runner = build_runner(
        session,
        built_corpus,
        tmp_path,
        db_name="finder-spend.sqlite3",
        script={
            "researcher.search": posting(LOCATOR),
            "verifier.relevance": deciding_relevance(
                lambda _: {"relevant": False, "raw_confidence": 0.9}
            ),
        },
    )

    result = asyncio.run(_run(runner, session.document_id))

    saved = runner.services.metadata.get_run(result.run_id)
    assert saved is not None
    assert saved.finder_tool_turns == 2
    assert saved.finder_search_calls == 1
    assert saved.finder_budget_exhausted_units == 0
    assert saved.verifier_tool_turns == 1
    assert [finding.code for finding in result.findings] == [FindingCode.NO_RELATION]


@pytest.mark.parametrize("arm", [ArmCode.OFF, ArmCode.MID])
def test_the_searching_and_characterising_payloads_show_the_whole_worksheet(
    built_corpus: QdrantCorpusIndex, tmp_path: Path, arm: ArmCode
) -> None:
    """The researcher and the analyst each decide what the unit still needs on the
    evidence in front of them, so both see all of it."""
    payloads: list[dict[str, Any]] = []
    session, _ = _session()

    def capture(turn: Turn) -> None:
        if turn.task.startswith(("researcher.", "analyst.")):
            payloads.append({"task": turn.task, **turn.payload})

    runner = build_runner(
        session,
        built_corpus,
        tmp_path,
        db_name=f"search-payload-{arm.value}.sqlite3",
        script={"researcher.search": posting(LOCATOR)},
        on_turn=capture,
    )

    asyncio.run(_run(runner, session.document_id, arm))

    search_payload = next(
        item for item in payloads if item["task"] == "researcher.search"
    )
    assert search_payload["worksheet"] == []

    later = next(item for item in payloads if item["task"] == "analyst.characterise")
    assert [entry["kind"] for entry in later["worksheet"]] == [
        "search",
        "candidate",
        "verdict",
    ]


_IN_FORCE = ForceState(
    value=ForceValue.IN_FORCE,
    scope=ForceScope.ACT,
    snapshot_date=date(2026, 9, 1),
    source_locator="https://example.invalid/act",
)


def _candidate(locator: str) -> Any:
    from contract_analyzer.agents.tool_args import CandidateArgs

    return CandidateArgs(locator=locator, why="instytucja prawna")


def _context_with_ledger(locators: list[str]) -> Any:
    """A unit context whose ledger already admits these locators.

    The provenance ledger is what _post_candidates checks before it reads, so a
    test of the reads has to start from a ledger that lets them through.
    """
    from contract_analyzer.agents.state import UnitContext
    from contract_analyzer.agents.worksheet import Worksheet

    run_id = uuid4()
    active = SimpleNamespace(
        services=SimpleNamespace(
            corpus=SimpleNamespace(snapshot=SimpleNamespace(id="s"))
        ),
        run_id=run_id,
    )
    context = UnitContext(
        active=cast(Any, active),
        call_unit=cast(Any, SimpleNamespace(unit_id="unit-1")),
        worksheet=Worksheet(run_id, "unit-1"),
    )
    context.locator_ledger.update(locators)
    return context


def test_a_candidate_the_corpus_does_not_hold_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty read refuses the whole commit and writes no candidate."""
    from contract_analyzer.agents import researcher as researcher_module
    from contract_analyzer.agents.tool_args import PostCandidatesArgs

    async def missing_read(active: object, locator: str) -> object | None:
        return (
            None
            if locator == "loc-b"
            else SimpleNamespace(
                locator=locator,
                act_identifier="DU/2023/725",
                act_force=_IN_FORCE,
                provision_force=_IN_FORCE,
                text="treść",
            )
        )

    monkeypatch.setattr(researcher_module, "read_provision", missing_read)
    context = _context_with_ledger(["loc-a", "loc-b"])
    args = PostCandidatesArgs(candidates=[_candidate("loc-a"), _candidate("loc-b")])

    result = asyncio.run(
        researcher_module._post_candidates(context, args, "post_candidates")
    )

    assert "unknown_locator" in str(result), f"wrong refusal: {result}"
    assert list(context.worksheet.entries) == [], "a candidate was written anyway"
