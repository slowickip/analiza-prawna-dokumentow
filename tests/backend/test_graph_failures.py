"""How a run dies, and what it leaves behind when it does.

Three kinds of ending are distinguished here: an answer the decision rules cannot
use, which voids the run; a dependency that went away, which is not a defect in
this server; and a budget that ran out, which leaves the finished units alone.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from uuid import UUID

import httpx
import pytest
from graph_harness import (
    LOCATOR,
    LOCATOR_ART19A,
    GraphDocument,
    build_runner,
    make_document,
    make_session,
    make_transport,
    request_for,
)
from model_client import mock_model_client
from pydantic import ValidationError
from worksheet_transport import (
    Turn,
    character_arguments,
    deciding_relation,
    deciding_relevance,
    envelope,
    play,
    posting,
    silence,
    tool_call,
)

from contract_analyzer.agents import unit as unit_cycle
from contract_analyzer.agents.prompts import load_prompt_bundle
from contract_analyzer.agents.runner import AnalysisRunner
from contract_analyzer.agents.services import AnalysisServices
from contract_analyzer.agents.session import RunRequest, build_call_units
from contract_analyzer.agents.synthesizer import SynthesisResponse
from contract_analyzer.agents.tool_args import (
    PLACEHOLDER_LOCATOR,
    PostCharacterArgs,
    RelationVerdictArgs,
    RelevanceVerdictArgs,
)
from contract_analyzer.corpus import EmbeddingError, QdrantCorpusIndex
from contract_analyzer.domain import ArmCode, FindingCode, RetrievalCandidate
from contract_analyzer.model import (
    AttemptTelemetry,
    ModelCallFailed,
)
from contract_analyzer.storage import MetadataStore


@pytest.fixture
def graph_document() -> GraphDocument:
    return make_document()


def test_undecided_relevance_without_confidence_is_refused_then_corrected(
    graph_document: GraphDocument, built_corpus: QdrantCorpusIndex, tmp_path: Path
) -> None:
    """A relevance verdict always carries a confidence, and the schema says so.

    Whether the verdict resolves to an uncertain finding depends on the candidate's
    force state, which the verifier is never shown, so the field is required of
    every relevance verdict rather than of the ones that turn out to need it. The
    refusal therefore comes from argument validation, in the same turn, naming the
    field.
    """
    refusals: list[dict[str, Any]] = []

    def relevance(turn: Turn) -> Any:
        if not turn.calls_made("post_verdict"):
            return tool_call(
                "post_verdict",
                {
                    "relevant": None,
                    "quote": turn.payload["unit_text"],
                    "raw_confidence": None,
                    "based_on": [],
                },
            )
        refusals.extend(result for result in turn.tool_results if "error" in result)
        return tool_call(
            "post_verdict",
            {
                "relevant": None,
                "quote": turn.payload["unit_text"],
                "raw_confidence": 0.4,
                "based_on": [],
            },
        )

    runner = build_runner(
        graph_document.session,
        built_corpus,
        tmp_path,
        db_name="null-relevance-confidence.sqlite3",
        script={"verifier.relevance": relevance},
    )

    result = asyncio.run(
        runner.run(request_for(graph_document, ArmCode.MID, concurrency=1))
    )

    assert result.status == "completed"
    assert {item["error"] for item in refusals} == {"invalid_arguments"}
    assert all(
        any(issue["field"] == "raw_confidence" for issue in item["detail"])
        for item in refusals
    )
    for unit_id in result.call_units:
        record = runner._runtime.worksheet_for(result.run_id, unit_id)
        assert record is not None
        assert (
            sum(
                entry.kind == "verdict" and entry.stage == "relevance"  # type: ignore[union-attr]
                for entry in record.entries
            )
            == 1
        )


def test_undetermined_relation_without_confidence_is_refused_then_corrected(
    graph_document: GraphDocument, built_corpus: QdrantCorpusIndex, tmp_path: Path
) -> None:
    refusals: list[dict[str, Any]] = []

    def relation(turn: Turn) -> Any:
        if not turn.calls_made("post_verdict"):
            return tool_call(
                "post_verdict",
                {"departure": "undetermined", "raw_confidence": None, "based_on": []},
            )
        refusals.extend(result for result in turn.tool_results if "error" in result)
        return tool_call(
            "post_verdict",
            {"departure": "undetermined", "raw_confidence": 0.4, "based_on": []},
        )

    runner = build_runner(
        graph_document.session,
        built_corpus,
        tmp_path,
        db_name="null-relation-confidence.sqlite3",
        script={"verifier.relation": relation},
    )

    result = asyncio.run(runner.run(request_for(graph_document, ArmCode.MID)))

    assert result.status == "completed"
    assert {item["error"] for item in refusals} == {"raw_confidence_required"}


def test_semi_imperative_departure_without_direction_is_refused_then_corrected(
    graph_document: GraphDocument, built_corpus: QdrantCorpusIndex, tmp_path: Path
) -> None:
    """An undirected departure from a semi-imperative provision maps to nothing.

    The relation tool permits a null direction under every departure state, so the
    model may produce it. The decision rules have no mapping for that combination
    and raise; it is rejected at the model boundary instead of ending the run with
    an unhandled exception that would lose every unit still queued.
    """
    refusals: list[dict[str, Any]] = []

    def relation(turn: Turn) -> Any:
        if not turn.calls_made("post_verdict"):
            return tool_call(
                "post_verdict",
                {"departure": "present", "direction": None, "based_on": []},
            )
        refusals.extend(result for result in turn.tool_results if "error" in result)
        return tool_call(
            "post_verdict",
            {
                "departure": "present",
                "direction": "with_permitted_direction",
                "based_on": [],
            },
        )

    runner = build_runner(
        graph_document.session,
        built_corpus,
        tmp_path,
        db_name="semi-imperative-without-direction.sqlite3",
        script={
            "analyst.characterise": lambda turn: tool_call(
                "post_character", character_arguments(turn, "semi_imperative")
            ),
            "verifier.relation": relation,
        },
    )

    result = asyncio.run(runner.run(request_for(graph_document, ArmCode.MID)))

    assert result.status == "completed"
    assert {item["error"] for item in refusals} == {"direction_required"}
    assert all("direction" in str(item["msg"]) for item in refusals)
    for unit_id in result.call_units:
        record = runner._runtime.worksheet_for(result.run_id, unit_id)
        assert record is not None
        assert (
            sum(
                entry.kind == "verdict" and entry.stage == "relation"  # type: ignore[union-attr]
                for entry in record.entries
            )
            == 1
        )


def test_repeated_guard_refusals_fail_only_at_the_verifier_turn_cap(
    graph_document: GraphDocument, built_corpus: QdrantCorpusIndex, tmp_path: Path
) -> None:
    turns: list[Turn] = []
    runner = build_runner(
        graph_document.session,
        built_corpus,
        tmp_path,
        db_name="guard-refusal-cap.sqlite3",
        script={
            "verifier.relevance": deciding_relevance(
                lambda _: {"relevant": None, "raw_confidence": None}
            )
        },
        on_turn=lambda turn: (
            turns.append(turn) if turn.task == "verifier.relevance" else None
        ),
    )

    result = asyncio.run(
        runner.run(request_for(graph_document, ArmCode.MID, concurrency=1))
    )

    assert result.status == "failed"
    assert result.error_code == "model_response_schema_invalid"
    assert len(turns) == 3


def test_a_repeatedly_invalid_characterisation_is_defaulted_after_refusals(
    graph_document: GraphDocument, built_corpus: QdrantCorpusIndex, tmp_path: Path
) -> None:
    """Undetermined with evidence breaks the closed field rules of the record.

    The refusal reaches the model, which may correct it; repeated invalid calls
    exhaust the task cap and default only that candidate's characterisation.
    """
    refusals: list[dict[str, Any]] = []

    def characterise(turn: Turn) -> Any:
        refusals.extend(result for result in turn.tool_results if "error" in result)
        arguments = character_arguments(turn, "imperative")
        arguments["kind"] = "undetermined"
        arguments["undetermined_reason"] = "Brak rozstrzygającego dowodu."
        return tool_call("post_character", arguments)

    runner = build_runner(
        graph_document.session,
        built_corpus,
        tmp_path,
        db_name="undetermined-character-with-evidence.sqlite3",
        script={
            "analyst.characterise": characterise,
            "verifier.relation": deciding_relation(
                lambda _: {"departure": "none", "raw_confidence": 0.4}
            ),
        },
    )

    result = asyncio.run(runner.run(request_for(graph_document, ArmCode.MID)))

    assert result.status == "completed"
    assert refusals
    assert {refusal["error"] for refusal in refusals} == {"invalid_arguments"}
    assert all(finding.code is FindingCode.UNCERTAIN for finding in result.findings)


def test_a_fabricated_evidence_locator_is_refused_and_the_run_continues(
    graph_document: GraphDocument, built_corpus: QdrantCorpusIndex, tmp_path: Path
) -> None:
    """A locator the corpus does not hold is a correctable mistake, not a dead run."""
    refusals: list[dict[str, Any]] = []

    def characterise(turn: Turn) -> Any:
        if not turn.calls_made("post_character"):
            arguments = character_arguments(turn, "imperative")
            arguments["evidence"][0]["locator"] = "art. 9999 ... Dz.U. 2099 poz. 0"
            return tool_call("post_character", arguments)
        refusals.extend(result for result in turn.tool_results if "error" in result)
        return tool_call("post_character", character_arguments(turn, "imperative"))

    runner = build_runner(
        graph_document.session,
        built_corpus,
        tmp_path,
        db_name="fabricated-evidence.sqlite3",
        script={"analyst.characterise": characterise},
    )

    result = asyncio.run(runner.run(request_for(graph_document, ArmCode.MID)))

    assert result.status == "completed"
    assert {refusal["error"] for refusal in refusals} == {"evidence_locator_unknown"}
    assert all(finding.code is FindingCode.CONSISTENT for finding in result.findings)


def test_an_invented_candidate_locator_is_refused_and_the_unit_finds_no_basis(
    graph_document: GraphDocument, built_corpus: QdrantCorpusIndex, tmp_path: Path
) -> None:
    """The analyst cannot invent a provision: the corpus decides what exists."""
    invented = "https://api.sejm.gov.pl/eli/acts/DU/2099/0/text.html/arti=9999"
    refusals: list[dict[str, Any]] = []

    def search(turn: Turn) -> Any:
        if not turn.calls_made("search_corpus"):
            return tool_call("search_corpus", {"phrase": "czynsz"})
        if not turn.calls_made("post_candidates"):
            return tool_call(
                "post_candidates",
                {"candidates": [{"locator": invented, "why": "wymysł"}]},
            )
        refusals.extend(result for result in turn.tool_results if "error" in result)
        return silence()

    runner = build_runner(
        graph_document.session,
        built_corpus,
        tmp_path,
        db_name="invented-candidate.sqlite3",
        script={"researcher.search": search},
    )

    result = asyncio.run(runner.run(request_for(graph_document, ArmCode.MID)))

    assert result.status == "completed"
    assert {refusal["error"] for refusal in refusals} == {
        "locator_not_in_provenance_ledger"
    }
    assert all(
        finding.code is FindingCode.NO_BASIS_FOUND for finding in result.findings
    )


def test_embedding_outage_mid_run_is_a_dependency_failure_not_a_defect(
    graph_document: GraphDocument, built_corpus: QdrantCorpusIndex, tmp_path: Path
) -> None:
    """A self-hosted service going away must not be filed against the artefact.

    Retrieval encodes its query over HTTP, so the embedding service can disappear
    between one unit and the next. EmbeddingError escaping the run would give it
    internal_error, the code the contract reserves for a defect in this server.
    """
    runner = build_runner(
        graph_document.session,
        built_corpus,
        tmp_path,
        db_name="embedding-outage.sqlite3",
    )

    def unavailable(*_args: object, **_kwargs: object) -> list[RetrievalCandidate]:
        raise EmbeddingError(
            "embedding_service_unavailable",
            "embedding_service_unavailable after 3 attempts",
        )

    runner.services.corpus.search = unavailable  # type: ignore[method-assign]

    result = asyncio.run(runner.run(request_for(graph_document, ArmCode.MID)))

    assert result.status == "failed"
    assert result.error_code == "dependency_unavailable"


def _failed_attempts(
    count: int, *, input_tokens: int = 111, output_tokens: int = 22
) -> tuple[AttemptTelemetry, ...]:
    return tuple(
        AttemptTelemetry(
            requested_model="deepseek-v4-flash",
            returned_model=None,
            prompt_version="researcher.search",
            temperature=0.0,
            parameters={"temperature": 0.0},
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=10.0,
            status="transport_error",
            retry_number=index,
            error_code="model_transport_error",
            prompt_hash="prompt-hash",
            response_hash=None,
        )
        for index in range(count)
    )


def _runner_with_failing_client(
    graph_document: GraphDocument,
    built_corpus: QdrantCorpusIndex,
    tmp_path: Path,
    *,
    attempts: tuple[AttemptTelemetry, ...],
    db_name: str,
) -> AnalysisRunner:
    runner = build_runner(
        graph_document.session, built_corpus, tmp_path, db_name=db_name
    )

    async def fail_converse(*args: Any, **kwargs: Any) -> Any:
        raise ModelCallFailed("model_transport_error", attempts)

    runner.services.client.converse = fail_converse  # type: ignore[method-assign]
    return runner


def test_terminal_failure_persists_attempt_telemetry(
    graph_document: GraphDocument, built_corpus: QdrantCorpusIndex, tmp_path: Path
) -> None:
    attempts = _failed_attempts(3)
    runner = _runner_with_failing_client(
        graph_document,
        built_corpus,
        tmp_path,
        attempts=attempts,
        db_name="failed-attempts.sqlite3",
    )

    result = asyncio.run(runner.run(request_for(graph_document, ArmCode.OFF)))

    assert result.status == "failed"
    persisted = runner.services.metadata.list_attempts(result.run_id)
    assert len(persisted) == len(attempts)
    assert sum(item.input_tokens for item in persisted) == 3 * 111
    assert sum(item.output_tokens for item in persisted) == 3 * 22


def test_terminal_failure_charges_budget_for_attempt_tokens(
    graph_document: GraphDocument, built_corpus: QdrantCorpusIndex, tmp_path: Path
) -> None:
    runner = _runner_with_failing_client(
        graph_document,
        built_corpus,
        tmp_path,
        attempts=_failed_attempts(3),
        db_name="failed-budget.sqlite3",
    )

    result = asyncio.run(
        runner.run(request_for(graph_document, ArmCode.OFF, concurrency=1))
    )

    assert result.status == "failed"
    assert result.tokens_used == 3 * (111 + 22)


def test_retried_then_successful_attempts_are_recorded_exactly_once(
    graph_document: GraphDocument, built_corpus: QdrantCorpusIndex, tmp_path: Path
) -> None:
    requests = 0
    inner = make_transport()

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        if requests == 1:
            raise httpx.ConnectError("temporary", request=request)
        return await inner.handle_async_request(request)

    runner = build_runner(
        graph_document.session,
        built_corpus,
        tmp_path,
        db_name="retry-success.sqlite3",
        transport=httpx.MockTransport(handler),
    )

    result = asyncio.run(
        runner.run(request_for(graph_document, ArmCode.MID, concurrency=1))
    )

    assert result.status == "completed"
    assert len(runner.services.metadata.list_attempts(result.run_id)) == requests


def test_failed_run_unprocessed_count_reflects_units_lacking_terminal_findings(
    graph_document: GraphDocument,
    built_corpus: QdrantCorpusIndex,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def exercise() -> None:
        runner = build_runner(
            graph_document.session,
            built_corpus,
            tmp_path,
            db_name="mid-flight-failure-unprocessed.sqlite3",
        )
        call_units = build_call_units(graph_document.session, ArmCode.MID)
        assert len(call_units) >= 2

        original_process_unit = runner._runtime.process_unit
        invocation_count = 0
        unit_one_completed = asyncio.Event()

        async def failing_process_unit(run_id, unit_id):
            nonlocal invocation_count
            invocation_count += 1
            if invocation_count == 1:
                outcome = await original_process_unit(run_id, unit_id)
                unit_one_completed.set()
                return outcome
            await unit_one_completed.wait()
            raise ModelCallFailed("model_transport_error", ())

        monkeypatch.setattr(runner._runtime, "process_unit", failing_process_unit)
        result = await runner.run(
            request_for(graph_document, ArmCode.MID, concurrency=1)
        )

        assert result.status == "failed"
        persisted = runner.services.metadata.list_findings(result.run_id)
        persisted_not_processed = sum(
            finding.code is FindingCode.NOT_PROCESSED for finding in persisted
        )
        assert persisted_not_processed == 0
        assert result.unprocessed_count == len(call_units) - 1

    asyncio.run(exercise())


def test_failed_unit_cancels_siblings_before_the_terminal_write(
    graph_document: GraphDocument,
    built_corpus: QdrantCorpusIndex,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def exercise() -> None:
        runner = build_runner(
            graph_document.session,
            built_corpus,
            tmp_path,
            db_name="cancel-siblings-before-failure.sqlite3",
        )
        prepared = runner.start_run(
            request_for(graph_document, ArmCode.MID, concurrency=2)
        )
        unit_ids = [unit.unit_id for unit in prepared.call_units]
        assert len(unit_ids) >= 2
        blocked, cancelled = asyncio.Event(), asyncio.Event()
        terminal_counts: tuple[int, int] | None = None

        async def fail_one_unit(run_id: UUID, unit_id: str) -> Any:
            del run_id
            if unit_id == unit_ids[0]:
                blocked.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    cancelled.set()
                    raise
            if unit_id == unit_ids[1]:
                await blocked.wait()
                raise ModelCallFailed("model_transport_error", ())
            await asyncio.Event().wait()
            raise AssertionError("cancelled worker resumed")

        original_finish_run = runner.services.metadata.finish_run

        def record_terminal(*args: Any, **kwargs: Any) -> None:
            nonlocal terminal_counts
            original_finish_run(*args, **kwargs)
            terminal_counts = (
                len(runner.services.metadata.list_findings(prepared.run_id)),
                len(runner.services.metadata.list_attempts(prepared.run_id)),
            )

        monkeypatch.setattr(runner._runtime, "process_unit", fail_one_unit)
        monkeypatch.setattr(runner.services.metadata, "finish_run", record_terminal)

        result = await asyncio.wait_for(runner.execute_run(prepared), timeout=2.0)
        await asyncio.sleep(0)

        assert result.status == "failed"
        assert result.error_code == "model_transport_error"
        assert cancelled.is_set()
        assert terminal_counts == (
            len(runner.services.metadata.list_findings(result.run_id)),
            len(runner.services.metadata.list_attempts(result.run_id)),
        )

    asyncio.run(exercise())


def test_wall_budget_cancels_in_flight_unit_model_call(
    graph_document: GraphDocument, built_corpus: QdrantCorpusIndex, tmp_path: Path
) -> None:
    async def exercise() -> None:
        runner = build_runner(
            graph_document.session,
            built_corpus,
            tmp_path,
            db_name="wall-in-flight-unit.sqlite3",
        )
        cancelled = asyncio.Event()

        async def blocking_handler(request: httpx.Request) -> httpx.Response:
            del request
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise

        runner.services.client = mock_model_client(
            httpx.MockTransport(blocking_handler), settings=runner.services.settings
        )
        result = await asyncio.wait_for(
            runner.run(
                request_for(
                    graph_document,
                    ArmCode.MID,
                    concurrency=1,
                    wall_budget_seconds=0.5,
                )
            ),
            timeout=2.0,
        )

        assert cancelled.is_set()
        assert result.status == "completed"
        assert result.interruption is True
        assert result.unprocessed_count == len(
            build_call_units(graph_document.session, ArmCode.MID)
        )
        assert result.tokens_used == 0
        attempts = runner.services.metadata.list_attempts(result.run_id)
        assert len(attempts) == 1
        assert attempts[0].status == "transport_error"
        assert attempts[0].error_code == "budget_exhausted"

    asyncio.run(exercise())


def test_wall_budget_cancels_in_flight_synthesis_model_call(
    graph_document: GraphDocument, built_corpus: QdrantCorpusIndex, tmp_path: Path
) -> None:
    async def exercise() -> None:
        synthesis_started = asyncio.Event()
        cancelled = asyncio.Event()

        async def blocking_synthesis(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            turn = Turn(body)
            if turn.task == "synthesise":
                synthesis_started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    cancelled.set()
                    raise
            return httpx.Response(
                200,
                request=request,
                json=envelope(play(turn), input_tokens=10, output_tokens=2),
            )

        runner = build_runner(
            graph_document.session,
            built_corpus,
            tmp_path,
            db_name="wall-in-flight-synthesis.sqlite3",
            transport=httpx.MockTransport(blocking_synthesis),
        )
        result = await asyncio.wait_for(
            runner.run(
                request_for(
                    graph_document,
                    ArmCode.OFF,
                    concurrency=1,
                    wall_budget_seconds=0.5,
                )
            ),
            timeout=5.0,
        )

        assert synthesis_started.is_set()
        assert cancelled.is_set()
        assert result.status == "completed"
        assert result.interruption is True
        assert result.unprocessed_count == 0
        saved = runner.services.metadata.get_run(result.run_id)
        assert saved is not None
        assert saved.interruption_reason == "wall_time"
        attempts = runner.services.metadata.list_attempts(result.run_id)
        assert attempts[-1].status == "transport_error"
        assert attempts[-1].error_code == "budget_exhausted"

    asyncio.run(exercise())


def test_expired_wall_budget_blocks_the_unit_cycle(
    graph_document: GraphDocument,
    built_corpus: QdrantCorpusIndex,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock_at = 0.0

    def clock() -> float:
        return clock_at

    async def exercise() -> None:
        nonlocal clock_at
        searches = 0
        corpus = built_corpus
        real_search = corpus.search

        def counting_search(query: str, limit: int):
            nonlocal searches
            searches += 1
            return real_search(query, limit)

        corpus.search = counting_search  # type: ignore[method-assign]
        runner = build_runner(
            graph_document.session,
            built_corpus,
            tmp_path,
            db_name="retrieve-budget-guard.sqlite3",
            corpus=corpus,
        )
        original_run = unit_cycle.run

        async def expire_wall_before_cycle(context):
            nonlocal clock_at
            clock_at = 301.0
            return await original_run(context)

        monkeypatch.setattr(unit_cycle, "run", expire_wall_before_cycle)
        result = await runner.run(
            RunRequest(
                document_id=graph_document.session.document_id,
                arm=ArmCode.OFF,
                measurement_valid=True,
                wall_budget_seconds=300.0,
                concurrency=1,
                clock=clock,
            )
        )
        assert result.interruption is True
        assert searches == 0
        assert any(
            finding.code is FindingCode.NOT_PROCESSED for finding in result.findings
        )

    asyncio.run(exercise())


def test_a_run_whose_analyst_never_stops_searching_is_recorded_as_cut_off(
    graph_document: GraphDocument, built_corpus: QdrantCorpusIndex, tmp_path: Path
) -> None:
    """A unit stopped by the turn cap must be distinguishable from one that gave up."""
    from contract_analyzer.agents.scheduler import FINDER_MAX_TOOL_TURNS

    session = make_session("§ 1. Najemca płaci czynsz.")
    runner = build_runner(
        session,
        built_corpus,
        tmp_path,
        db_name="finder-cut-off.sqlite3",
        script={
            "researcher.search": lambda turn: tool_call(
                "search_corpus", {"phrase": "czynsz najem"}
            )
        },
    )

    result = asyncio.run(
        runner.run(
            RunRequest(
                document_id=session.document_id,
                arm=ArmCode.OFF,
                measurement_valid=True,
                wall_budget_seconds=300.0,
                concurrency=1,
            )
        )
    )

    saved = runner.services.metadata.get_run(result.run_id)
    assert saved is not None
    assert saved.finder_budget_exhausted_units == 1
    assert saved.finder_tool_turns == FINDER_MAX_TOOL_TURNS
    assert all(
        finding.code is FindingCode.NO_BASIS_FOUND for finding in result.findings
    )


def test_agent_schemas_forbid_unexpected_extra_keys() -> None:
    """A field the schema does not name is a rejected answer, not extra."""
    with pytest.raises(ValidationError):
        RelevanceVerdictArgs.model_validate(
            {
                "relevant": True,
                "quote": "test",
                "unexpected_field": "disallowed",
            }
        )

    with pytest.raises(ValidationError):
        RelationVerdictArgs.model_validate(
            {
                "departure": "none",
                "unexpected_field": "disallowed",
            }
        )

    with pytest.raises(ValidationError):
        PostCharacterArgs.model_validate(
            {
                "kind": "imperative",
                "evidence": [
                    {
                        "source_kind": "official_normative_text",
                        "locator": LOCATOR,
                        "pinpoint": "art. 1",
                        "interpretive_methods": ["linguistic"],
                        "rationale": "reason",
                    }
                ],
                "unexpected_field": "disallowed",
            }
        )

    with pytest.raises(ValidationError):
        SynthesisResponse.model_validate({"groups": [], "unexpected_field": "no"})


def test_services_without_a_text_store_still_complete_a_run(
    graph_document: GraphDocument, built_corpus: QdrantCorpusIndex, tmp_path: Path
) -> None:
    """Retention is optional; nothing on the analysis path may depend on it."""
    settings = build_runner(
        graph_document.session, built_corpus, tmp_path, db_name="unused.sqlite3"
    ).services.settings
    services = AnalysisServices(
        settings=settings,
        corpus=built_corpus,
        client=mock_model_client(
            make_transport(script={"researcher.search": posting(LOCATOR)}),
            settings=settings,
        ),
        metadata=MetadataStore(tmp_path / "no-text-store.sqlite3"),
        prompt_bundle=load_prompt_bundle(),
    )
    services.sessions[graph_document.session.document_id] = graph_document.session
    runner = AnalysisRunner(services)

    result = asyncio.run(runner.run(request_for(graph_document, ArmCode.MID)))

    assert result.status == "completed"
    assert runner.synthesis_for(result.run_id) is None
    assert runner._runtime.worksheet_for(result.run_id, result.call_units[0]) is None


def test_evidence_the_analyst_never_retrieved_is_refused(
    graph_document: GraphDocument, built_corpus: QdrantCorpusIndex, tmp_path: Path
) -> None:
    """A real provision is not evidence unless a tool handed it to this unit.

    Existence in the corpus is the weaker test: a locator the model remembered,
    or copied from the schema's example, would otherwise commit silently and
    correctly with the wrong provision behind the characterisation. The analyst
    here reads one provision and posts it, so the unit's ledger holds that one
    locator and nothing else.
    """
    refusals: list[dict[str, Any]] = []

    def search(turn: Turn) -> Any:
        if not turn.calls_made("read_provision"):
            return tool_call("read_provision", {"locator": LOCATOR})
        return tool_call(
            "post_candidates",
            {"candidates": [{"locator": LOCATOR, "why": "Przepis dotyczy najmu."}]},
        )

    def characterise(turn: Turn) -> Any:
        if not turn.calls_made("post_character"):
            arguments = character_arguments(turn, "imperative")
            arguments["evidence"][0]["locator"] = LOCATOR_ART19A
            return tool_call("post_character", arguments)
        refusals.extend(result for result in turn.tool_results if "error" in result)
        return tool_call("post_character", character_arguments(turn, "imperative"))

    runner = build_runner(
        graph_document.session,
        built_corpus,
        tmp_path,
        db_name="unretrieved-evidence.sqlite3",
        script={"researcher.search": search, "analyst.characterise": characterise},
    )

    result = asyncio.run(runner.run(request_for(graph_document, ArmCode.MID)))

    assert result.status == "completed"
    assert {refusal["error"] for refusal in refusals} == {
        "locator_not_in_provenance_ledger"
    }


def test_the_placeholder_locator_resolves_to_nothing(
    built_corpus: QdrantCorpusIndex,
) -> None:
    """The examples are safe to copy only while the corpus refuses them."""
    with pytest.raises(KeyError):
        built_corpus.read(PLACEHOLDER_LOCATOR)
