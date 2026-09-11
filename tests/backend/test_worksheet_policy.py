"""Integration tripwires for the worksheet's trust and execution policies."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from graph_harness import LOCATOR, build_runner, make_session
from worksheet_transport import (
    Turn,
    posting,
    quote_of,
    silence,
    tool_call,
    tool_calls,
)

from conftest import publish_corpus
from contract_analyzer.agents import unit as unit_cycle
from contract_analyzer.agents.errors import RunPipelineError
from contract_analyzer.agents.session import RunRequest
from contract_analyzer.corpus import QdrantCorpusIndex
from contract_analyzer.corpus.qdrant_index import SearchResult
from contract_analyzer.domain import ArmCode, RetrievalCandidate, candidate_admits

UNIT_TEXT = "§ 1. Najemca składa kaucję zabezpieczającą."


def _request(document_id, *, wall_seconds: float = 300.0) -> RunRequest:
    return RunRequest(
        document_id=document_id,
        arm=ArmCode.MID,
        wall_budget_seconds=wall_seconds,
        concurrency=1,
    )


def test_verifier_payload_never_contains_analyst_prose(
    built_corpus: QdrantCorpusIndex, tmp_path: Path
) -> None:
    sentinel = "sentinel_analyst_prose_must_not_reach_verifier"
    session = make_session(UNIT_TEXT)
    verifier_payloads: list[str] = []

    def search(turn: Turn) -> Mapping[str, Any]:
        if not turn.calls_made("search_corpus"):
            return tool_call("search_corpus", {"phrase": "kaucja najem"})
        return tool_call(
            "post_candidates",
            {"candidates": [{"locator": LOCATOR, "why": sentinel}]},
        )

    def relevance(turn: Turn) -> Mapping[str, Any]:
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
        script={
            "researcher.search": search,
            "verifier.relevance": relevance,
        },
        on_turn=lambda turn: (
            verifier_payloads.append(json.dumps(turn.payload))
            if turn.task.startswith("verifier.")
            else None
        ),
    )

    result = asyncio.run(runner.run(_request(session.document_id)))

    assert result.status == "completed"
    assert verifier_payloads
    assert all(sentinel not in payload for payload in verifier_payloads)


def test_two_commits_in_one_turn_are_rejected_atomically(
    built_corpus: QdrantCorpusIndex, tmp_path: Path
) -> None:
    session = make_session(UNIT_TEXT)
    refusals: list[dict[str, Any]] = []

    def search(turn: Turn) -> Mapping[str, Any]:
        if not turn.calls_made("post_candidates"):
            commit = {"candidates": [{"locator": LOCATOR, "why": "w"}]}
            return tool_calls(
                [("post_candidates", commit), ("post_candidates", commit)]
            )
        refusals.extend(item for item in turn.tool_results if "error" in item)
        return silence()

    runner = build_runner(
        session, built_corpus, tmp_path, script={"researcher.search": search}
    )
    result = asyncio.run(runner.run(_request(session.document_id)))
    record = runner._runtime.worksheet_for(result.run_id, session.units[0].id)

    assert [item["error"] for item in refusals] == [
        "commit_must_be_alone",
        "commit_must_be_alone",
    ]
    assert record is not None
    assert record.entries == ()


def test_a_verdict_shares_its_turn_with_nothing(
    built_corpus: QdrantCorpusIndex, tmp_path: Path
) -> None:
    """The relevance task's commit tool must be alone in its turn."""
    session = make_session(UNIT_TEXT)
    refusals: list[dict[str, Any]] = []

    def relevance(turn: Turn) -> Mapping[str, Any]:
        if not turn.calls_made("read_provision"):
            return tool_calls(
                [
                    (
                        "post_verdict",
                        {
                            "relevant": False,
                            "quote": quote_of(turn),
                            "raw_confidence": 0.4,
                            "based_on": [],
                        },
                    ),
                    ("read_provision", {"locator": LOCATOR}),
                ]
            )
        refusals.extend(item for item in turn.tool_results if "error" in item)
        return tool_call(
            "post_verdict",
            {
                "relevant": False,
                "quote": quote_of(turn),
                "raw_confidence": 0.4,
                "based_on": [],
            },
        )

    runner = build_runner(
        session,
        built_corpus,
        tmp_path,
        script={"researcher.search": posting(LOCATOR), "verifier.relevance": relevance},
    )
    result = asyncio.run(runner.run(_request(session.document_id)))
    record = runner._runtime.worksheet_for(result.run_id, session.units[0].id)

    assert [item["error"] for item in refusals] == [
        "commit_must_be_alone",
        "commit_must_be_alone",
    ]
    assert record is not None
    kinds = [entry.kind for entry in record.entries]
    assert "verdict" in kinds
    assert "read" not in kinds


def test_verdict_refuses_a_dependency_hidden_from_this_candidate(
    built_corpus: QdrantCorpusIndex, tmp_path: Path
) -> None:
    session = make_session(UNIT_TEXT)
    refusals: list[dict[str, Any]] = []

    def relevance(turn: Turn) -> Mapping[str, Any]:
        if not turn.calls_made("post_verdict"):
            return tool_call(
                "post_verdict",
                {
                    "relevant": False,
                    "quote": quote_of(turn),
                    "raw_confidence": 0.4,
                    "based_on": ["e999"],
                },
            )
        refusals.extend(item for item in turn.tool_results if "error" in item)
        return tool_call(
            "post_verdict",
            {
                "relevant": False,
                "quote": quote_of(turn),
                "raw_confidence": 0.4,
                "based_on": [],
            },
        )

    runner = build_runner(
        session,
        built_corpus,
        tmp_path,
        script={
            "researcher.search": posting(LOCATOR),
            "verifier.relevance": relevance,
        },
    )
    result = asyncio.run(runner.run(_request(session.document_id)))

    assert result.status == "completed"
    assert [item["error"] for item in refusals] == ["verdict_dependency_not_visible"]


def test_role_payloads_include_the_provision_under_decision(
    built_corpus: QdrantCorpusIndex, tmp_path: Path
) -> None:
    session = make_session(UNIT_TEXT)
    payloads: dict[str, dict[str, Any]] = {}
    runner = build_runner(
        session,
        built_corpus,
        tmp_path,
        script={"researcher.search": posting(LOCATOR)},
        on_turn=lambda turn: payloads.setdefault(turn.task, turn.payload),
    )

    result = asyncio.run(runner.run(_request(session.document_id)))
    provision_text = built_corpus.read(LOCATOR).text

    assert result.status == "completed"
    assert payloads["verifier.relevance"]["candidate_act_identifier"]
    assert payloads["verifier.relevance"]["candidate_text"] == provision_text
    assert payloads["analyst.characterise"]["candidate_text"] == provision_text
    assert payloads["verifier.relation"]["basis_text"] == provision_text


def test_blocking_corpus_search_expires_without_late_writes(
    built_corpus: QdrantCorpusIndex, tmp_path: Path
) -> None:
    session = make_session(UNIT_TEXT)
    corpus = built_corpus
    original = corpus.search

    def blocking_search(query: str, limit: int):
        time.sleep(0.05)
        return original(query, limit)

    corpus.search = blocking_search  # type: ignore[method-assign]
    runner = build_runner(session, built_corpus, tmp_path, corpus=corpus)
    runner._runtime.retire_run = lambda run_id: None  # type: ignore[method-assign]

    result = asyncio.run(runner.run(_request(session.document_id, wall_seconds=0.01)))
    active = runner._runtime.active(result.run_id)
    record = runner._runtime.worksheet_for(result.run_id, session.units[0].id)

    assert result.interruption is True
    assert active.retrieval_cache == {}
    assert record is not None
    assert not [
        entry for entry in record.entries if entry.kind in {"search", "candidate"}
    ]


def test_unit_step_error_becomes_a_typed_pipeline_failure(
    built_corpus: QdrantCorpusIndex,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = make_session(UNIT_TEXT)
    runner = build_runner(session, built_corpus, tmp_path)

    async def exceed(context: object) -> list[str]:
        del context
        raise RunPipelineError("unit_graph_recursion_exceeded")

    monkeypatch.setattr(unit_cycle, "run", exceed)
    result = asyncio.run(runner.run(_request(session.document_id)))

    assert result.status == "failed"
    assert result.error_code == "unit_graph_recursion_exceeded"


@pytest.mark.parametrize("arm", list(ArmCode))
def test_six_candidates_from_two_searches_complete_the_longest_path(
    arm: ArmCode,
    built_corpus: QdrantCorpusIndex,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = make_session(UNIT_TEXT)
    units = [item.unit for item in _admitted_candidates(built_corpus, count=5)]
    extra = units[0].model_copy(
        update={
            "id": "synthetic-sixth-provision",
            "locator": "https://example.test/provisions/6",
            "article_identifier": "6",
        }
    )
    corpus = publish_corpus(
        [*units, extra], act_currency=built_corpus.snapshot.act_currency
    )
    candidates = _admitted_candidates(corpus, count=6)

    def search_page(query: str, limit: int) -> list[SearchResult]:
        assert limit == 5
        return candidates[:5] if query == "najem" else candidates[5:]

    corpus.search = search_page  # type: ignore[method-assign]
    steps = 0

    def search(turn: Turn) -> Mapping[str, Any]:
        if not turn.calls_made("search_corpus"):
            return tool_call("search_corpus", {"phrase": "najem"})
        if turn.calls_made("search_corpus") == 1:
            return tool_call("search_corpus", {"phrase": "kaucja"})
        locators = [
            item["locator"]
            for result in turn.tool_results
            for item in result.get("results", [])
        ]
        return tool_call(
            "post_candidates",
            {"candidates": [{"locator": locator, "why": "w"} for locator in locators]},
        )

    def relevance(turn: Turn) -> Mapping[str, Any]:
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
        corpus=corpus,
        script={"researcher.search": search, "verifier.relevance": relevance},
    )
    original_next_step = unit_cycle.next_step

    def count_step(worksheet):
        nonlocal steps
        step = original_next_step(worksheet)
        if step is not None:
            steps += 1
        return step

    monkeypatch.setattr(unit_cycle, "next_step", count_step)
    request = replace(_request(session.document_id), arm=arm)
    result = asyncio.run(runner.run(request))

    assert result.status == "completed"
    assert steps == 19  # Search, then three visits for each of six candidates.


def _admitted_candidates(
    corpus: QdrantCorpusIndex, *, count: int
) -> list[SearchResult]:
    points, _ = corpus._client.scroll(
        collection_name=corpus.collection, limit=64, with_payload=True
    )
    candidates = []
    for point in points:
        unit = corpus.read(str(point.payload["locator"]))
        candidate = RetrievalCandidate(
            locator=unit.locator,
            snapshot_id=corpus.snapshot.id,
            act_identifier=unit.act_identifier,
            act_force=unit.act_force,
            provision_force=unit.provision_force,
            rank=len(candidates) + 1,
            sparse_score=1.0,
            dense_score=1.0,
        )
        if candidate_admits(candidate):
            candidates.append(SearchResult(candidate=candidate, unit=unit))
        if len(candidates) == count:
            return candidates
    raise AssertionError(f"fixture corpus has fewer than {count} admitted provisions")
