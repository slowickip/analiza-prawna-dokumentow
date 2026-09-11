from __future__ import annotations

import asyncio
import hashlib
import json
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from uuid import UUID, uuid4

import httpx
import pytest
from model_client import mock_model_client
from worksheet_transport import Turn, play
from worksheet_transport import envelope as response_envelope

from contract_analyzer.agents import chat_grounding as chat_grounding_module
from contract_analyzer.agents import retrieval as retrieval_module
from contract_analyzer.agents.errors import BudgetExhausted
from contract_analyzer.agents.prompts import load_prompt_bundle
from contract_analyzer.agents.retrieval import read_provision, run_search
from contract_analyzer.agents.runner import AnalysisRunner
from contract_analyzer.agents.services import AnalysisServices
from contract_analyzer.agents.session import DocumentSession, RunRequest
from contract_analyzer.agents.state import ActiveRun
from contract_analyzer.budget import BudgetTracker
from contract_analyzer.config import RunConfig, Settings
from contract_analyzer.corpus import QdrantCorpusIndex
from contract_analyzer.domain import (
    ArmCode,
    DocumentPayload,
    FindingCode,
    ReadMode,
    SourceAnchor,
)
from contract_analyzer.storage import MetadataStore
from contract_analyzer.structure import parse_references, segment

MULTI_UNIT_TEXT = """§ 1. Pierwsza jednostka umowy.

§ 2. Druga jednostka umowy.

§ 3. Trzecia jednostka umowy.

§ 4. Czwarta jednostka umowy.
"""

LOCATOR = "https://api.sejm.gov.pl/eli/acts/DU/2023/725/text.html/arti=11"


class FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self._now = start

    def __call__(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


@dataclass(frozen=True)
class BudgetDocument:
    document_id: UUID
    session: DocumentSession


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@pytest.fixture
def budget_document() -> BudgetDocument:
    payload = DocumentPayload(
        document_id=uuid4(),
        text=MULTI_UNIT_TEXT,
        anchors=(SourceAnchor(start_offset=0, end_offset=len(MULTI_UNIT_TEXT)),),
        read_mode=ReadMode.NATIVE_PDF,
        content_hash=_content_hash(MULTI_UNIT_TEXT),
    )
    units = segment(payload, RunConfig())
    session = DocumentSession(
        document_id=payload.document_id,
        payload=payload,
        units=tuple(units),
        references=tuple(parse_references(units)),
    )
    return BudgetDocument(document_id=session.document_id, session=session)


def _runner(
    document: BudgetDocument,
    built_corpus: QdrantCorpusIndex,
    tmp_path: Path,
    *,
    tokens_per_call: int,
    request_hook: Callable[[dict[str, object]], None] | None = None,
    payload_hook: Callable[[dict[str, object]], None] | None = None,
    db_name: str = "budget.sqlite3",
) -> AnalysisRunner:
    settings = Settings(
        model_api_key="test-secret-key",
        model_base_url="https://openrouter.ai/api/v1",
        model_name="deepseek-v4-flash",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        if payload_hook is not None:
            payload_hook(payload)
        turn = Turn(payload)
        if request_hook is not None:
            request_hook({"prompt_version": turn.task, "payload": turn.payload})
        if "tools" in payload:
            message = dict(play(turn))
        else:
            raise AssertionError("the pipeline sent a request carrying no tools")
        return httpx.Response(
            200,
            request=request,
            json=response_envelope(
                message,
                input_tokens=tokens_per_call,
                output_tokens=2,
            ),
        )

    client = mock_model_client(httpx.MockTransport(handler), settings=settings)
    services = AnalysisServices(
        settings=settings,
        corpus=built_corpus,
        client=client,
        metadata=MetadataStore(tmp_path / db_name),
        prompt_bundle=load_prompt_bundle(),
    )
    services.sessions[document.session.document_id] = document.session
    return AnalysisRunner(services)


def test_default_run_uses_low_reasoning_without_a_completion_cutoff(
    budget_document: BudgetDocument,
    built_corpus: QdrantCorpusIndex,
    tmp_path: Path,
) -> None:
    payloads: list[dict[str, object]] = []
    request = RunRequest(
        document_id=budget_document.document_id,
        arm=ArmCode.MID,
        measurement_valid=True,
    )

    async def exercise() -> None:
        runner = _runner(
            budget_document,
            built_corpus,
            tmp_path,
            tokens_per_call=1,
            payload_hook=payloads.append,
            db_name="bounded-output.sqlite3",
        )
        result = await runner.run(request)

        assert payloads
        # The wire spells the effort its own way; the record keeps this
        # project's name for it, and neither cap is sent when none was asked for.
        assert all("max_output_tokens" not in payload for payload in payloads)
        assert all(
            payload.get("reasoning") == {"effort": "low"} for payload in payloads
        )
        assert all("max_tokens" not in payload for payload in payloads)
        assert "max_completion_tokens" not in result.parity_bundle.parameters
        assert result.parity_bundle.parameters["reasoning_effort"] == "low"
        attempts = runner.services.metadata.list_attempts(result.run_id)
        assert attempts
        assert all(
            "max_completion_tokens" not in attempt.parameters for attempt in attempts
        )
        assert {attempt.parameters.get("reasoning_effort") for attempt in attempts} == {
            "low"
        }

    asyncio.run(exercise())


def test_default_run_has_no_implicit_aggregate_token_cutoff(
    budget_document: BudgetDocument,
    built_corpus: QdrantCorpusIndex,
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        runner = _runner(
            budget_document,
            built_corpus,
            tmp_path,
            tokens_per_call=10_000,
            db_name="no-aggregate-token-budget.sqlite3",
        )
        result = await runner.run(
            RunRequest(
                document_id=budget_document.document_id,
                arm=ArmCode.MID,
                measurement_valid=True,
                wall_budget_seconds=300.0,
                concurrency=1,
            )
        )

        assert result.tokens_used > 50_000
        assert result.interruption is False
        assert result.unprocessed_count == 0

    asyncio.run(exercise())


def test_explicit_completion_cap_reaches_request_telemetry_and_parity(
    budget_document: BudgetDocument,
    built_corpus: QdrantCorpusIndex,
    tmp_path: Path,
) -> None:
    payloads: list[dict[str, object]] = []

    async def exercise() -> None:
        runner = _runner(
            budget_document,
            built_corpus,
            tmp_path,
            tokens_per_call=1,
            payload_hook=payloads.append,
            db_name="explicit-completion-cap.sqlite3",
        )
        result = await runner.run(
            RunRequest(
                document_id=budget_document.document_id,
                arm=ArmCode.MID,
                parameters={
                    "temperature": 0.0,
                    "reasoning_effort": "low",
                    "max_completion_tokens": 37,
                },
            )
        )

        assert payloads
        # One cap, named max_completion_tokens everywhere this project records
        # it and max_output_tokens only on the wire.
        assert {payload["max_output_tokens"] for payload in payloads} == {37}
        assert all("max_completion_tokens" not in payload for payload in payloads)
        assert result.parity_bundle.parameters["max_completion_tokens"] == 37
        attempts = runner.services.metadata.list_attempts(result.run_id)
        assert attempts
        assert {
            attempt.parameters["max_completion_tokens"] for attempt in attempts
        } == {37}

    asyncio.run(exercise())


@pytest.mark.parametrize("invalid_cap", [0, -1, True, "37"])
def test_completion_cap_must_be_a_positive_integer(
    budget_document: BudgetDocument,
    built_corpus: QdrantCorpusIndex,
    tmp_path: Path,
    invalid_cap: object,
) -> None:
    runner = _runner(
        budget_document,
        built_corpus,
        tmp_path,
        tokens_per_call=1,
        db_name=f"invalid-completion-cap-{invalid_cap!s}.sqlite3",
    )

    with pytest.raises(ValueError, match="must be a positive integer"):
        runner.start_run(
            RunRequest(
                document_id=budget_document.document_id,
                arm=ArmCode.MID,
                parameters={
                    "temperature": 0.0,
                    "reasoning_effort": "low",
                    "max_completion_tokens": invalid_cap,
                },
            )
        )


def test_legacy_max_tokens_is_rejected_for_reasoning_models(
    budget_document: BudgetDocument,
    built_corpus: QdrantCorpusIndex,
    tmp_path: Path,
) -> None:
    runner = _runner(
        budget_document,
        built_corpus,
        tmp_path,
        tokens_per_call=1,
        db_name="legacy-max-tokens.sqlite3",
    )

    with pytest.raises(ValueError, match="does not bound reasoning tokens"):
        runner.start_run(
            RunRequest(
                document_id=budget_document.document_id,
                arm=ArmCode.MID,
                parameters={"temperature": 0.0, "max_tokens": 4_096},
            )
        )


def test_wall_budget_marks_remaining_units_not_processed(
    budget_document: BudgetDocument,
    built_corpus: QdrantCorpusIndex,
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    call_count = 0

    def hook(_envelope: dict[str, object]) -> None:
        nonlocal call_count
        call_count += 1
        if call_count >= 13:
            clock.advance(301.0)

    async def exercise() -> None:
        runner = _runner(
            budget_document,
            built_corpus,
            tmp_path,
            tokens_per_call=1,
            request_hook=hook,
            db_name="wall-budget.sqlite3",
        )
        result = await runner.run(
            RunRequest(
                document_id=budget_document.document_id,
                arm=ArmCode.MID,
                measurement_valid=True,
                wall_budget_seconds=300.0,
                concurrency=1,
                clock=clock,
            )
        )
        assert result.interruption is True
        assert result.unprocessed_count == 2
        not_processed = [
            finding
            for finding in result.findings
            if finding.code is FindingCode.NOT_PROCESSED
        ]
        assert len(not_processed) == 2
        processed = [
            finding
            for finding in result.findings
            if finding.code is not FindingCode.NOT_PROCESSED
        ]
        assert len(processed) == 2

    asyncio.run(exercise())


def test_wall_budget_with_concurrency_cancels_queued_units(
    budget_document: BudgetDocument,
    built_corpus: QdrantCorpusIndex,
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    call_count = 0

    def hook(_envelope: dict[str, object]) -> None:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            clock.advance(301.0)

    async def exercise() -> None:
        runner = _runner(
            budget_document,
            built_corpus,
            tmp_path,
            tokens_per_call=1,
            request_hook=hook,
            db_name="wall-concurrency.sqlite3",
        )
        result = await runner.run(
            RunRequest(
                document_id=budget_document.document_id,
                arm=ArmCode.MID,
                measurement_valid=True,
                wall_budget_seconds=300.0,
                concurrency=2,
                clock=clock,
            )
        )
        assert result.interruption is True
        assert result.unprocessed_count >= 1
        not_processed = [
            finding
            for finding in result.findings
            if finding.code is FindingCode.NOT_PROCESSED
        ]
        assert not_processed

    asyncio.run(exercise())


def test_wall_budget_during_synthesis_sets_interruption(
    budget_document: BudgetDocument,
    built_corpus: QdrantCorpusIndex,
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    role_calls = 0
    prompt_versions: list[str] = []

    def hook(envelope: dict[str, object]) -> None:
        nonlocal role_calls
        prompt_versions.append(str(envelope.get("prompt_version")))
        if envelope.get("prompt_version") != "synthesise":
            role_calls += 1
            if role_calls == 5:
                clock.advance(301.0)

    async def exercise() -> None:
        runner = _runner(
            budget_document,
            built_corpus,
            tmp_path,
            tokens_per_call=1,
            request_hook=hook,
            db_name="wall-synthesis.sqlite3",
        )
        result = await runner.run(
            RunRequest(
                document_id=budget_document.document_id,
                arm=ArmCode.OFF,
                measurement_valid=True,
                wall_budget_seconds=300.0,
                concurrency=1,
                clock=clock,
            )
        )
        assert result.interruption is True
        assert result.unprocessed_count == 0
        processed = [
            finding
            for finding in result.findings
            if finding.code is not FindingCode.NOT_PROCESSED
        ]
        assert len(processed) == 1
        assert "synthesise" not in prompt_versions

    asyncio.run(exercise())


def test_budget_tracker_wall_deadline() -> None:
    """Time is the only thing that exhausts a budget, however much was spent."""
    clock = FakeClock()
    budget = BudgetTracker(
        wall_budget_seconds=10.0,
        clock=clock,
    )
    assert budget.can_schedule() is True
    clock.advance(9.0)
    assert budget.can_schedule() is True
    clock.advance(1.0)
    assert budget.can_schedule() is False
    assert budget.exhausted is True
    assert budget.exhaustion_reason == "wall_time"


def test_budget_tracker_reads_the_clock_rather_than_a_refreshed_flag() -> None:
    """A caller that asks between calls must not get a stale answer.

    document_graph and synthesizer both ask whether the budget is exhausted at
    points where nothing has just refreshed it, so the question has to answer
    from the clock.
    """
    clock = FakeClock()
    budget = BudgetTracker(wall_budget_seconds=10.0, clock=clock)
    clock.advance(11.0)

    assert budget.exhausted is True
    assert budget.remaining_seconds() == 0.0


def _measured_active(
    clock: FakeClock, corpus: object, wall_budget: float = 10.0
) -> ActiveRun:
    """An ActiveRun with a fake clock and a stub corpus, for budget pins."""
    return ActiveRun(
        services=SimpleNamespace(corpus=corpus),
        run_id=uuid4(),
        request=RunRequest(document_id=uuid4(), arm=ArmCode.MID),
        session=SimpleNamespace(),
        call_units=[],
        call_units_by_id={},
        budget=BudgetTracker(
            wall_budget_seconds=wall_budget,
            clock=clock,
        ),
        semaphore=asyncio.Semaphore(8),
        retrieval_cache={},
        provisions={},
    )


def test_a_cached_provision_read_still_meets_the_wall_deadline() -> None:
    """A locator cached before the deadline expires with it, not past it."""

    async def exercise() -> None:
        clock = FakeClock()
        reads: list[str] = []

        class StubCorpus:
            def read(self, locator: str) -> object:
                reads.append(locator)
                return SimpleNamespace(locator=locator, text="tresc")

        active = _measured_active(clock, StubCorpus())
        locator = "https://example.invalid/arti=11"
        await read_provision(active, locator)
        assert reads == [locator]

        clock.advance(11.0)
        with pytest.raises(BudgetExhausted):
            await read_provision(active, locator)
        assert reads == [locator]

    asyncio.run(exercise())


def test_run_search_snippet_fanout_leaves_no_late_cache_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed snippet read leaves no late cache entry behind it."""

    async def exercise() -> None:
        clock = FakeClock()
        active = _measured_active(clock, SimpleNamespace())
        boom = RuntimeError("boom")
        slow_done: list[bool] = []
        candidates = [
            SimpleNamespace(locator="a-fast", act_identifier="DU/23"),
            SimpleNamespace(locator="b-slow", act_identifier="DU/23"),
        ]

        async def fake_search(active_run: object, phrase: str) -> object:
            del active_run, phrase
            return candidates

        async def fake_snippet(active_run: ActiveRun, locator: str) -> str:
            if locator == "a-fast":
                raise boom
            await asyncio.sleep(0.2)
            slow_done.append(True)
            active_run.provisions[locator] = SimpleNamespace(text="s")
            return "slow-snippet"

        monkeypatch.setattr(retrieval_module, "search_corpus", fake_search)
        monkeypatch.setattr(retrieval_module, "provision_snippet", fake_snippet)
        with pytest.raises(RuntimeError) as excinfo:
            await run_search(active, "wypowiedzenie najmu")
        assert excinfo.value is boom
        await asyncio.sleep(1.0)
        assert slow_done == []
        assert "b-slow" not in active.provisions

    asyncio.run(exercise())


def test_a_corpus_read_finishing_after_its_sibling_failed_writes_nothing() -> None:
    """The real corpus call runs in a thread, which no cancellation can stop.

    The thread therefore runs to completion after a sibling has failed. What must
    hold is that its result reaches nothing: the cache write sits after the await,
    which cancellation does reach, so the finished read is discarded.
    """

    async def exercise() -> None:
        clock = FakeClock()
        boom = RuntimeError("boom")
        thread_done: list[str] = []

        started = threading.Event()

        class BlockingCorpus:
            def read(self, locator: str) -> object:
                if locator == "a-fast":
                    assert started.wait(5.0), "the slow read never entered its thread"
                    raise boom
                started.set()
                time.sleep(0.2)
                thread_done.append(locator)
                return SimpleNamespace(locator=locator, text="slow")

        active = _measured_active(clock, BlockingCorpus())
        selected = [
            SimpleNamespace(legal_locators=["a-fast"]),
            SimpleNamespace(legal_locators=["b-slow"]),
        ]

        with pytest.raises(RuntimeError) as excinfo:
            await chat_grounding_module.provisions(active, cast(Any, selected))
        assert excinfo.value is boom

        await asyncio.sleep(1.0)
        assert thread_done == ["b-slow"]
        assert "b-slow" not in active.provisions

    asyncio.run(exercise())


def test_provisions_fanout_leaves_no_late_cache_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed grounding read leaves no late cache entry behind it."""

    async def exercise() -> None:
        clock = FakeClock()
        active = _measured_active(clock, SimpleNamespace())
        boom = RuntimeError("boom")
        slow_done: list[bool] = []
        selected = [
            SimpleNamespace(legal_locators=["a-fast"]),
            SimpleNamespace(legal_locators=["b-slow"]),
        ]

        async def fake_bounded_read(active_run: ActiveRun, locator: str) -> object:
            if locator == "a-fast":
                raise boom
            await asyncio.sleep(0.2)
            slow_done.append(True)
            unit = SimpleNamespace(locator=locator, text="slow")
            active_run.provisions[locator] = unit
            return unit

        monkeypatch.setattr(chat_grounding_module, "bounded_read", fake_bounded_read)
        with pytest.raises(RuntimeError) as excinfo:
            await chat_grounding_module.provisions(active, selected)
        assert excinfo.value is boom
        await asyncio.sleep(1.0)
        assert slow_done == []
        assert "b-slow" not in active.provisions

    asyncio.run(exercise())
