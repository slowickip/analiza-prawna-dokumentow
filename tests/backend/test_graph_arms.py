"""What the three configurations show a role, and what the run graph does with them.

The arms share model, corpus, prompts and tools and differ only in the call unit,
so these tests are about the payload each role receives, the fan-out over call
units, and the run-level facts recorded about both.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from importlib import resources
from pathlib import Path
from typing import Any, cast
from uuid import UUID

import pytest
from graph_harness import (
    DOCUMENT_TEXT,
    GraphDocument,
    TickingClock,
    build_runner,
    make_document,
    make_session,
    request_for,
    role_payloads,
)
from worksheet_transport import Turn

from contract_analyzer.agents import unit as unit_cycle
from contract_analyzer.agents.errors import RunPipelineError
from contract_analyzer.agents.parity import GRAPH_TOPOLOGY_VERSION
from contract_analyzer.agents.prompts import (
    PROMPT_FILES,
    hash_prompt_bundle,
    load_prompt_bundle,
)
from contract_analyzer.agents.retrieval import sanitize_retrieval_query
from contract_analyzer.agents.runner import AnalysisRunner
from contract_analyzer.agents.session import RunRequest, build_call_units
from contract_analyzer.agents.synthesizer import SynthesisResponse
from contract_analyzer.config import RunConfig
from contract_analyzer.corpus import QdrantCorpusIndex
from contract_analyzer.domain import ArmCode, FindingCode, ForceScope, ForceValue
from contract_analyzer.storage import EventBus, RunEvent


@pytest.fixture
def graph_document() -> GraphDocument:
    return make_document()


@pytest.fixture
def runner(
    graph_document: GraphDocument, built_corpus: QdrantCorpusIndex, tmp_path: Path
) -> AnalysisRunner:
    return build_runner(graph_document.session, built_corpus, tmp_path)


def _runner_capturing_turns(
    graph_document: GraphDocument,
    built_corpus: QdrantCorpusIndex,
    tmp_path: Path,
    *,
    db_name: str,
) -> tuple[AnalysisRunner, list[Turn]]:
    turns: list[Turn] = []
    runner = build_runner(
        graph_document.session,
        built_corpus,
        tmp_path,
        db_name=db_name,
        on_turn=turns.append,
    )
    return runner, turns


def test_mid_and_on_payloads_differ_for_referencing_unit(
    graph_document: GraphDocument, built_corpus: QdrantCorpusIndex, tmp_path: Path
) -> None:
    citing_unit = next(
        unit for unit in graph_document.session.units if "Zgodnie" in unit.text
    )
    target_unit = next(
        unit
        for unit in graph_document.session.units
        if unit.id == graph_document.resolved_target_id
    )

    async def exercise() -> None:
        mid_runner, mid_turns = _runner_capturing_turns(
            graph_document, built_corpus, tmp_path, db_name="mid-payloads.sqlite3"
        )
        on_runner, on_turns = _runner_capturing_turns(
            graph_document, built_corpus, tmp_path, db_name="on-payloads.sqlite3"
        )
        await mid_runner.run(request_for(graph_document, ArmCode.MID))
        await on_runner.run(request_for(graph_document, ArmCode.ON))

        mid_payloads = role_payloads(mid_turns, citing_unit.id)
        on_payloads = role_payloads(on_turns, citing_unit.id)
        assert mid_payloads
        assert on_payloads

        mid_relevance = next(
            payload for payload in mid_payloads if "candidate_locator" in payload
        )
        on_relevance = next(
            payload for payload in on_payloads if "candidate_locator" in payload
        )

        assert mid_relevance != on_relevance
        context_units = cast(list[dict[str, Any]], on_relevance["context_units"])
        assert context_units
        assert target_unit.text in [str(item["text"]) for item in context_units]
        assert "context_units" not in mid_relevance
        assert target_unit.text not in str(mid_relevance.get("unit_text", ""))

    asyncio.run(exercise())


def test_role_payloads_include_unit_text(
    graph_document: GraphDocument, built_corpus: QdrantCorpusIndex, tmp_path: Path
) -> None:
    citing_unit = next(
        unit for unit in graph_document.session.units if "Zgodnie" in unit.text
    )

    async def exercise() -> None:
        runner, turns = _runner_capturing_turns(
            graph_document, built_corpus, tmp_path, db_name="unit-text.sqlite3"
        )
        await runner.run(request_for(graph_document, ArmCode.MID))

        payloads = role_payloads(turns, citing_unit.id)
        assert payloads
        for payload in payloads:
            assert payload.get("unit_text") == citing_unit.text

    asyncio.run(exercise())


def test_off_payload_includes_whole_document_text(
    graph_document: GraphDocument, built_corpus: QdrantCorpusIndex, tmp_path: Path
) -> None:
    async def exercise() -> None:
        runner, turns = _runner_capturing_turns(
            graph_document, built_corpus, tmp_path, db_name="off-payload.sqlite3"
        )
        await runner.run(request_for(graph_document, ArmCode.OFF))

        payloads = role_payloads(turns, graph_document.whole_document_id)
        assert payloads
        for payload in payloads:
            assert payload.get("unit_text") == DOCUMENT_TEXT
            assert "context_units" not in payload

    asyncio.run(exercise())


def test_context_units_are_separate_from_unit_text(
    graph_document: GraphDocument, built_corpus: QdrantCorpusIndex, tmp_path: Path
) -> None:
    citing_unit = next(
        unit for unit in graph_document.session.units if "Zgodnie" in unit.text
    )
    target_unit = next(
        unit
        for unit in graph_document.session.units
        if unit.id == graph_document.resolved_target_id
    )

    async def exercise() -> None:
        runner, turns = _runner_capturing_turns(
            graph_document, built_corpus, tmp_path, db_name="context-separate.sqlite3"
        )
        await runner.run(request_for(graph_document, ArmCode.ON))

        payloads = role_payloads(turns, citing_unit.id)
        assert payloads
        for payload in payloads:
            assert payload.get("unit_text") == citing_unit.text
            context_units = cast(list[dict[str, Any]], payload["context_units"])
            assert target_unit.text in [str(item["text"]) for item in context_units]
            assert target_unit.text not in str(payload.get("unit_text", ""))

    asyncio.run(exercise())


def test_only_call_unit_context_differs_between_arms(
    runner: AnalysisRunner, graph_document: GraphDocument
) -> None:
    async def exercise() -> None:
        results = [
            await runner.run(request_for(graph_document, arm)) for arm in ArmCode
        ]
        assert (
            results[0].parity_bundle
            == results[1].parity_bundle
            == results[2].parity_bundle
        )
        assert results[0].call_units == [graph_document.whole_document_id]
        assert results[1].context_unit_ids == []
        assert results[2].context_unit_ids == [graph_document.resolved_target_id]

    asyncio.run(exercise())


def test_synthesizer_schema_cannot_create_findings() -> None:
    assert "findings" not in SynthesisResponse.model_fields
    assert set(SynthesisResponse.model_fields) == {"groups"}


def test_processed_units_event_stream_observes_fan_out_per_call_unit(
    graph_document: GraphDocument, built_corpus: QdrantCorpusIndex, tmp_path: Path
) -> None:
    """Observe worker fan-out across evaluation arms through the run's event stream."""

    async def exercise() -> None:
        events = EventBus()
        runner = build_runner(
            graph_document.session,
            built_corpus,
            tmp_path,
            db_name="event-stream-fan-out.sqlite3",
            events=events,
        )
        counts: dict[ArmCode, int] = {}
        for arm in ArmCode:
            prepared = runner.start_run(request_for(graph_document, arm))
            assert runner.services.events is not None
            queue = runner.services.events.subscribe(prepared.run_id)
            await runner.execute_run(prepared)
            processed_units = 0
            while True:
                try:
                    event = queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if event.kind == "counter" and event.counter_name == "processed_units":
                    processed_units += 1
            counts[arm] = processed_units
        assert counts[ArmCode.OFF] == 1
        assert counts[ArmCode.MID] == 3
        assert counts[ArmCode.ON] == 3

    asyncio.run(exercise())


def test_prompt_bundle_hash_is_a_stable_sha256() -> None:
    first = hash_prompt_bundle()
    assert first == hash_prompt_bundle()
    assert len(first) == 64
    assert all(character in "0123456789abcdef" for character in first)


@pytest.mark.parametrize("prompt_name", PROMPT_FILES)
def test_every_prompt_file_contributes_to_the_bundle_hash(prompt_name: str) -> None:
    """prompt_bundle_version is a parity field: a prompt it ignores is a silent lie."""
    prompts = dict(load_prompt_bundle().prompts)
    baseline = hash_prompt_bundle(prompts)

    prompts[prompt_name] = prompts[prompt_name] + "\nzmiana"

    assert hash_prompt_bundle(prompts) != baseline


def test_prompt_files_lists_every_prompt_shipped_in_the_package() -> None:
    """A prompt on disk but absent from PROMPT_FILES would never reach the hash."""
    shipped = {
        entry.name
        for entry in resources.files("contract_analyzer.prompts").iterdir()
        if entry.name.endswith(".md")
    }
    assert shipped == set(PROMPT_FILES)


def test_every_task_prompt_names_the_tools_its_role_is_given() -> None:
    """A prompt that describes a response format would be describing the old design."""
    from contract_analyzer.agents.prompts import TASK_PROMPTS
    from contract_analyzer.agents.tools import tools_for

    prompts = load_prompt_bundle().prompts
    for task, filename in TASK_PROMPTS.items():
        text = prompts[filename]
        for spec in tools_for(task):  # type: ignore[arg-type]
            assert spec.name in text, (task, spec.name)


def _fallback_session():
    """A session whose segmentation fell back to overlapping sentence windows."""
    text = (
        "Najemca zobowiazuje sie do platnosci czynszu. "
        "Wynajmujacy przekazuje lokal w dniu podpisania. "
        "Umowa podlega prawu polskiemu. "
        "Strony ustalaja termin platnosci na dziesiaty dzien miesiaca."
    )
    return make_session(text, config=RunConfig(structural_min=99))


def test_on_arm_runs_on_a_window_fallback_document_with_zero_context() -> None:
    """ON must run everywhere and record what it received.

    A window-fallback document has no resolvable internal references, so ON is
    byte-identical to MID. That is a real experimental condition -- the protocol's
    zero_edge_holdout cohort reads it as the noise floor -- not a defect to refuse.
    """
    session = _fallback_session()

    on = build_call_units(session, ArmCode.ON)
    mid = build_call_units(session, ArmCode.MID)

    assert on == mid
    assert sum(len(unit.context_unit_ids) for unit in on) == 0


@pytest.mark.parametrize("arm", [ArmCode.OFF, ArmCode.MID, ArmCode.ON])
def test_every_arm_accepts_a_window_fallback_document(arm: ArmCode) -> None:
    assert build_call_units(_fallback_session(), arm)


def test_on_and_mid_differ_when_structure_is_available(
    graph_document: GraphDocument,
) -> None:
    """The arms must be distinguishable, or the comparison measures nothing."""
    mid = build_call_units(graph_document.session, ArmCode.MID)
    on = build_call_units(graph_document.session, ArmCode.ON)
    assert [unit.context_unit_ids for unit in mid] != [
        unit.context_unit_ids for unit in on
    ]
    assert any(unit.context_unit_ids for unit in on)


def test_completed_run_produces_consistent_findings(
    runner: AnalysisRunner, graph_document: GraphDocument
) -> None:
    async def exercise() -> None:
        result = await runner.run(request_for(graph_document, ArmCode.MID))
        assert result.status == "completed"
        assert result.findings
        assert all(
            finding.code is FindingCode.CONSISTENT for finding in result.findings
        )

    asyncio.run(exercise())


def test_admitted_findings_carry_both_force_records_through_storage(
    runner: AnalysisRunner, graph_document: GraphDocument
) -> None:
    """RF-05 v1.4's carry-through, proven over the whole path.

    The candidate entry copies the corpus force records when it is posted, so this
    drives a real run and reads the findings back out of storage: the emitted basis
    is shown to survive worksheet -> store -> reload.
    """

    async def exercise() -> None:
        result = await runner.run(request_for(graph_document, ArmCode.MID))
        assert result.status == "completed"

        reloaded = runner.services.metadata.list_findings(result.run_id)
        assert reloaded

        for finding in reloaded:
            assert finding.code is FindingCode.CONSISTENT
            basis = finding.basis
            assert basis is not None, "an admitted finding must carry its basis"
            assert basis.act_force.value is ForceValue.IN_FORCE
            assert basis.act_force.scope is ForceScope.ACT
            # The ordinary case, and the one the requirement exists for: a
            # consolidated text cannot establish that a provision is in force.
            assert basis.provision_force.value is ForceValue.UNDETERMINED
            assert basis.provision_force.scope is ForceScope.PROVISION
            assert basis.provision_locator == finding.legal_locators[0]

    asyncio.run(exercise())


def test_runner_stores_graph_topology_version(
    runner: AnalysisRunner, graph_document: GraphDocument
) -> None:
    async def exercise() -> None:
        result = await runner.run(request_for(graph_document, ArmCode.MID))
        saved_run = runner.services.metadata.get_run(result.run_id)
        assert saved_run is not None
        assert saved_run.graph_topology_version == GRAPH_TOPOLOGY_VERSION
        assert GRAPH_TOPOLOGY_VERSION == "worksheet-unit-cycle-v5"

    asyncio.run(exercise())


def test_run_records_what_the_two_roles_spent(
    runner: AnalysisRunner, graph_document: GraphDocument
) -> None:
    """Every counter the record carries has to be filled by the loop that spends it."""

    async def exercise() -> None:
        result = await runner.run(request_for(graph_document, ArmCode.MID))
        saved = runner.services.metadata.get_run(result.run_id)
        assert saved is not None
        assert saved.finder_tool_turns > 0
        assert saved.finder_search_calls > 0
        assert saved.finder_budget_exhausted_units == 0
        assert saved.verifier_tool_turns > 0
        assert saved.provision_reads == 0

    asyncio.run(exercise())


def test_terminal_run_releases_document_session(
    built_corpus: QdrantCorpusIndex, tmp_path: Path
) -> None:
    """Completed runs drop the run's session while preserving response metadata."""
    document = make_document(f"TERMINAL_RELEASE_CANARY_1d4f7b2a\n{DOCUMENT_TEXT}")
    runner = build_runner(
        document.session, built_corpus, tmp_path, db_name="terminal-release.sqlite3"
    )

    async def exercise() -> None:
        result = await runner.run(request_for(document, ArmCode.MID))
        assert result.status == "completed"
        assert result.tokens_used > 0
        assert result.parity_bundle.returned_model is not None
        record = runner.services.metadata.get_run(result.run_id)
        assert record is not None
        assert record.returned_model is not None
        assert result.run_id not in runner._runtime._runs

    asyncio.run(exercise())


def test_sanitize_retrieval_query_keeps_all_tokens() -> None:
    tokens = [f"token{i}" for i in range(40)]
    assert sanitize_retrieval_query(" ".join(tokens)) == " ".join(tokens)


def test_sanitize_retrieval_query_raises_on_empty() -> None:
    with pytest.raises(RunPipelineError, match="retrieval_query_empty"):
        sanitize_retrieval_query("... --- !!!")


def test_sanitize_retrieval_query_strips_punctuation() -> None:
    assert sanitize_retrieval_query("§ 1. Najemca, zobowiązuje się.") == (
        "1 Najemca zobowiązuje się"
    )


def _runner_with_event_capture(
    graph_document: GraphDocument,
    built_corpus: QdrantCorpusIndex,
    tmp_path: Path,
    *,
    db_name: str,
    on_turn: Callable[[Turn], None] | None = None,
) -> tuple[AnalysisRunner, list[tuple[UUID, RunEvent]]]:
    captured: list[tuple[UUID, RunEvent]] = []
    bus = EventBus(queue_size=1_000)

    def capture_publish(run_id: UUID, event: RunEvent) -> None:
        captured.append((run_id, event))
        EventBus.publish(bus, run_id, event)

    bus.publish = capture_publish  # type: ignore[method-assign]
    runner = build_runner(
        graph_document.session,
        built_corpus,
        tmp_path,
        db_name=db_name,
        events=bus,
        on_turn=on_turn,
    )
    return runner, captured


def _processed_unit_counts(
    captured: list[tuple[UUID, RunEvent]], run_id: UUID
) -> list[int]:
    return [
        event.count
        for published_run_id, event in captured
        if published_run_id == run_id
        and event.kind == "counter"
        and event.counter_name == "processed_units"
        and event.count is not None
    ]


def test_completed_run_publishes_monotonic_processed_units_counter(
    graph_document: GraphDocument, built_corpus: QdrantCorpusIndex, tmp_path: Path
) -> None:
    async def exercise() -> None:
        runner, captured = _runner_with_event_capture(
            graph_document,
            built_corpus,
            tmp_path,
            db_name="processed-units-counter.sqlite3",
        )
        result = await runner.run(request_for(graph_document, ArmCode.MID))
        call_unit_count = len(build_call_units(graph_document.session, ArmCode.MID))
        assert call_unit_count > 1
        counts = _processed_unit_counts(captured, result.run_id)
        assert counts
        assert all(
            later > earlier for earlier, later in zip(counts, counts[1:], strict=False)
        )
        assert counts[-1] == call_unit_count

    asyncio.run(exercise())


def test_budget_exhausted_run_omits_unprocessed_units_from_counter(
    graph_document: GraphDocument, built_corpus: QdrantCorpusIndex, tmp_path: Path
) -> None:
    async def exercise() -> None:
        clock = TickingClock()
        runner, captured = _runner_with_event_capture(
            graph_document,
            built_corpus,
            tmp_path,
            db_name="processed-units-budget.sqlite3",
            on_turn=clock.tick,
        )
        result = await runner.run(
            request_for(
                graph_document,
                ArmCode.MID,
                concurrency=1,
                # One second per model call, so the run is cut off part way
                # through the document rather than finishing it.
                wall_budget_seconds=6.0,
                clock=clock,
                parameters={"temperature": 0.0, "max_completion_tokens": 2},
            )
        )
        call_unit_count = len(build_call_units(graph_document.session, ArmCode.MID))
        not_processed = [
            finding
            for finding in result.findings
            if finding.code is FindingCode.NOT_PROCESSED
        ]
        assert not_processed
        processed_count = call_unit_count - len(not_processed)
        counts = _processed_unit_counts(captured, result.run_id)
        assert counts
        assert max(counts) == processed_count
        assert max(counts) < call_unit_count

    asyncio.run(exercise())


def test_unit_driver_concurrency_never_exceeds_request_limit(
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
            db_name="unit-concurrency.sqlite3",
        )
        active_count = 0
        max_active = 0
        second_entered = asyncio.Event()
        original_run = unit_cycle.run

        async def probing_run(context):
            nonlocal active_count, max_active
            active_count += 1
            slot = active_count
            max_active = max(max_active, active_count)
            if slot == 1:
                await asyncio.wait_for(second_entered.wait(), timeout=2.0)
            elif slot == 2:
                second_entered.set()
            try:
                return await original_run(context)
            finally:
                active_count -= 1

        monkeypatch.setattr(unit_cycle, "run", probing_run)
        await runner.run(
            RunRequest(
                document_id=graph_document.session.document_id,
                arm=ArmCode.MID,
                measurement_valid=True,
                concurrency=2,
            )
        )
        assert max_active == 2

    asyncio.run(exercise())


def test_concurrency_one_keeps_unit_pass_sequence_contiguous(
    graph_document: GraphDocument, built_corpus: QdrantCorpusIndex, tmp_path: Path
) -> None:
    sequence: list[str] = []

    async def exercise() -> None:
        def record_unit_id(turn: Turn) -> None:
            if turn.task.startswith(("analyst.", "verifier.")):
                sequence.append(str(turn.payload.get("unit_id", "")))

        runner = build_runner(
            graph_document.session,
            built_corpus,
            tmp_path,
            db_name="contiguous-passes.sqlite3",
            on_turn=record_unit_id,
        )
        await runner.run(
            RunRequest(
                document_id=graph_document.session.document_id,
                arm=ArmCode.MID,
                measurement_valid=True,
                concurrency=1,
            )
        )
        assert sequence
        blocks: list[list[str]] = []
        block: list[str] = []
        for unit_id in sequence:
            if not block or unit_id == block[-1]:
                block.append(unit_id)
            else:
                blocks.append(block)
                block = [unit_id]
        if block:
            blocks.append(block)
        assert len(blocks) == len(build_call_units(graph_document.session, ArmCode.MID))

    asyncio.run(exercise())
