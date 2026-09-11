"""The measured roles run on real LangChain graphs, not labels on one completion.

Ports of the compatibility probe's load-bearing assertions against the production
runtime: a provider failure after a successful tool turn keeps the earlier
attempts without duplicating state, all four tool-using roles commit through
actual graphs, and a task that never commits is stopped by its cap instead of
running on.
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
    GraphDocument,
    build_runner,
    make_document,
    request_for,
)
from worksheet_transport import (
    Turn,
    envelope,
    silence,
    synthesis_group_all,
    tool_call,
    tool_calls,
)

from contract_analyzer.agents import retrieval as retrieval_module
from contract_analyzer.corpus import QdrantCorpusIndex
from contract_analyzer.domain import ArmCode


def test_provider_error_after_a_successful_tool_turn_keeps_earlier_attempts(
    built_corpus: QdrantCorpusIndex, tmp_path: Path
) -> None:
    """A failure on the second turn keeps the first turn's usage and work.

    The search already ran and was recorded, so the run fails with the
    provider's code, the successful attempt stays in storage, and the worksheet
    holds exactly the one search -- nothing ran twice.
    """
    graph_document: GraphDocument = make_document()

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        turn = Turn(body)
        if turn.task == "researcher.search" and not turn.calls_made("search_corpus"):
            return httpx.Response(
                200,
                request=request,
                json=envelope(
                    tool_call("search_corpus", {"phrase": "czynsz najem"}),
                    input_tokens=7,
                    output_tokens=3,
                ),
            )
        return httpx.Response(500, json={"error": {"message": "synthetic busy"}})

    async def exercise() -> None:
        runner = build_runner(
            graph_document.session,
            built_corpus,
            tmp_path,
            db_name="provider-error-keeps-attempts.sqlite3",
            transport=httpx.MockTransport(handler),
        )
        result = await runner.run(request_for(graph_document, ArmCode.OFF))

        assert result.status == "failed"
        assert result.error_code == "model_http_500"
        attempts = runner.services.metadata.list_attempts(result.run_id)
        assert len(attempts) == 4
        first, rest = attempts[0], attempts[1:]
        assert first.status == "success"
        assert (first.input_tokens, first.output_tokens) == (7, 3)
        assert first.prompt_version == "researcher.search"
        assert all(item.status == "http_error" for item in rest)
        assert sum(item.input_tokens for item in attempts) >= 7
        saved = runner.services.metadata.get_run(result.run_id)
        assert saved is not None
        assert saved.finder_search_calls == 1
        record = runner._runtime.worksheet_for(result.run_id, result.call_units[0])
        assert record is not None
        assert sum(1 for entry in record.entries if entry.kind == "search") == 1

    asyncio.run(exercise())


def test_four_tool_using_roles_commit_through_real_graphs(
    built_corpus: QdrantCorpusIndex, tmp_path: Path
) -> None:
    """Researcher, analyst, verifier and synthesizer each spend tool turns.

    The synthesizer lists the existing findings and commits one grouping over
    them; the stored grouping names identifiers, never the numbered references
    the role was given.
    """
    graph_document: GraphDocument = make_document()
    synthesis_turns: list[Turn] = []

    async def exercise() -> None:
        runner = build_runner(
            graph_document.session,
            built_corpus,
            tmp_path,
            db_name="four-graph-roles.sqlite3",
            script={"synthesise": synthesis_group_all},
            on_turn=lambda turn: (
                synthesis_turns.append(turn) if turn.task == "synthesise" else None
            ),
        )
        result = await runner.run(request_for(graph_document, ArmCode.MID))

        assert result.status == "completed"
        assert result.findings
        versions = {
            attempt.prompt_version
            for attempt in runner.services.metadata.list_attempts(result.run_id)
        }
        assert {
            "researcher.search",
            "analyst.characterise",
            "verifier.relevance",
            "verifier.relation",
            "synthesise",
        } <= versions
        calls = [
            call["function"]["name"]
            for turn in synthesis_turns
            for message in turn.messages
            if message.get("role") == "assistant"
            for call in message.get("tool_calls") or []
        ]
        assert "list_findings" in calls
        # The grouping itself is the commit tool's effect: it is stored only
        # where the commit handler ran, resolved back to identifiers.
        grouping = runner.synthesis_for(result.run_id)
        assert grouping is not None
        assert len(grouping.groups) == 1
        grouped = list(grouping.groups[0].finding_ids)
        assert grouped == [str(record.id) for record in result.findings]
        assert all(len(value) == 36 for value in grouped)
        assert all(UUID(value) for value in grouped)

    asyncio.run(exercise())


def test_a_synthesis_that_never_commits_is_stopped_by_its_cap(
    built_corpus: QdrantCorpusIndex, tmp_path: Path
) -> None:
    """The listing script never ends, so the enforced cap has to end it.

    An unusable grouping is a fault in the table of contents, not in the
    analysis: the run completes with its findings and keeps no grouping.
    """
    from contract_analyzer.agents.scheduler import TASK_MAX_TURNS

    graph_document: GraphDocument = make_document()
    synthesis_tasks: list[str] = []

    def endless_listing(turn: Turn) -> Any:
        return tool_call("list_findings", {})

    async def exercise() -> None:
        runner = build_runner(
            graph_document.session,
            built_corpus,
            tmp_path,
            db_name="synthesis-cap.sqlite3",
            script={"synthesise": endless_listing},
            on_turn=lambda turn: (
                synthesis_tasks.append(turn.task) if turn.task == "synthesise" else None
            ),
        )
        result = await runner.run(request_for(graph_document, ArmCode.MID))

        assert result.status == "completed"
        assert result.findings
        assert runner.synthesis_for(result.run_id) is None

    asyncio.run(exercise())
    assert len(synthesis_tasks) == TASK_MAX_TURNS["synthesizer.group"]


def test_a_mixed_commit_and_unknown_tool_turn_runs_no_handlers(
    built_corpus: QdrantCorpusIndex, tmp_path: Path
) -> None:
    """Whole-turn guards come before domain dispatch, whatever the names are.

    A commit beside an unoffered tool is refused whole: neither the commit nor
    the unknown call may reach a handler. The model then recovers through the
    ordinary path, so the run still completes.
    """
    graph_document: GraphDocument = make_document()
    refusals: list[str] = []
    seen_results = 0

    def collect(turn: Turn) -> None:
        nonlocal seen_results
        fresh = turn.tool_results[seen_results:]
        seen_results = len(turn.tool_results)
        refusals.extend(
            str(result.get("error")) for result in fresh if "error" in result
        )

    def mixed_then_recover(turn: Turn) -> Any:
        if not turn.calls_made("search_corpus") and not turn.calls_made(
            "post_candidates"
        ):
            return tool_calls(
                [
                    (
                        "post_candidates",
                        {
                            "candidates": [
                                {"locator": LOCATOR, "why": "Przepis dotyczy najmu."}
                            ]
                        },
                    ),
                    ("mystery_tool", {}),
                ]
            )
        collect(turn)
        if not turn.calls_made("search_corpus"):
            return tool_call("search_corpus", {"phrase": "czynsz najem"})
        if not any("candidate_ids" in result for result in turn.tool_results):
            found = turn.first_result_locator()
            if found is not None:
                return tool_call(
                    "post_candidates",
                    {
                        "candidates": [
                            {"locator": found, "why": "Przepis dotyczy najmu."}
                        ]
                    },
                )
        return silence()

    async def exercise() -> None:
        runner = build_runner(
            graph_document.session,
            built_corpus,
            tmp_path,
            db_name="mixed-commit-unknown.sqlite3",
            script={"researcher.search": mixed_then_recover},
        )
        result = await runner.run(request_for(graph_document, ArmCode.OFF))

        assert result.status == "completed"
        assert refusals == ["commit_must_be_alone", "commit_must_be_alone"]
        record = runner._runtime.worksheet_for(result.run_id, result.call_units[0])
        assert record is not None
        # Only the recovery ran handlers: one search, one posted candidate.
        assert sum(1 for entry in record.entries if entry.kind == "search") == 1
        assert sum(1 for entry in record.entries if entry.kind == "candidate") == 1
        saved = runner.services.metadata.get_run(result.run_id)
        assert saved is not None
        assert saved.finder_search_calls == 1

    asyncio.run(exercise())


def test_cap_and_domain_refusals_cover_unknown_tools_by_position(
    built_corpus: QdrantCorpusIndex, tmp_path: Path
) -> None:
    """The per-turn cap counts every call; the handler names the unknown one.

    Five searches fill the researcher's per-turn bound, so the sixth call --
    unoffered -- gets the cap refusal without reaching a handler. Alone on the
    next turn it reaches domain dispatch and gets the unknown-tool refusal.
    """
    graph_document: GraphDocument = make_document()
    refusals: list[str] = []
    seen_results = 0

    def collect(turn: Turn) -> None:
        nonlocal seen_results
        fresh = turn.tool_results[seen_results:]
        seen_results = len(turn.tool_results)
        refusals.extend(
            str(result.get("error")) for result in fresh if "error" in result
        )

    def over_then_lone(turn: Turn) -> Any:
        if not turn.calls_made("search_corpus"):
            searches = [
                ("search_corpus", {"phrase": f"fraza numer {index}"})
                for index in range(5)
            ]
            return tool_calls(searches + [("mystery_tool", {})])
        collect(turn)
        if turn.calls_made("mystery_tool") < 2:
            return tool_call("mystery_tool", {})
        return tool_call("post_candidates", {"candidates": []})

    async def exercise() -> None:
        runner = build_runner(
            graph_document.session,
            built_corpus,
            tmp_path,
            db_name="unknown-tool-positions.sqlite3",
            script={"researcher.search": over_then_lone},
        )
        result = await runner.run(request_for(graph_document, ArmCode.OFF))

        assert result.status == "completed"
        assert refusals[0] == "phrase_budget_exceeded_for_this_turn"
        assert refusals[1] == "unknown_tool"
        saved = runner.services.metadata.get_run(result.run_id)
        assert saved is not None
        # All five searches ran exactly once; the refused calls ran nothing.
        assert saved.finder_search_calls == 5

    asyncio.run(exercise())


def test_parallel_tool_calls_execute_handlers_sequentially(
    built_corpus: QdrantCorpusIndex,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The graph may dispatch a turn's calls together; handlers still run alone.

    Two searches and one unknown tool leave in one turn. The instrumented
    corpus entry shows the searches never overlap, and the unknown call is
    refused through the same serialised dispatch.
    """
    graph_document: GraphDocument = make_document()
    events: list[str] = []
    real_search = retrieval_module.search_corpus

    async def tracked(active: Any, phrase: str) -> Any:
        events.append(f"enter {phrase}")
        await asyncio.sleep(0.2)
        try:
            return await real_search(active, phrase)
        finally:
            events.append(f"exit {phrase}")

    monkeypatch.setattr(retrieval_module, "search_corpus", tracked)
    refusals: list[str] = []
    seen_results = 0

    def parallel(turn: Turn) -> Any:
        nonlocal seen_results
        if not turn.calls_made("search_corpus"):
            return tool_calls(
                [
                    ("search_corpus", {"phrase": "czynsz najem"}),
                    ("search_corpus", {"phrase": "kaucja umowa"}),
                    ("mystery_tool", {}),
                ]
            )
        fresh = turn.tool_results[seen_results:]
        seen_results = len(turn.tool_results)
        refusals.extend(
            str(result.get("error")) for result in fresh if "error" in result
        )
        if not turn.calls_made("post_candidates"):
            return tool_call("post_candidates", {"candidates": []})
        return silence()

    async def exercise() -> None:
        runner = build_runner(
            graph_document.session,
            built_corpus,
            tmp_path,
            db_name="sequential-handlers.sqlite3",
            script={"researcher.search": parallel},
        )
        result = await runner.run(request_for(graph_document, ArmCode.OFF))

        assert result.status == "completed"
        assert refusals == ["unknown_tool"]
        depth = 0
        for event in events:
            if event.startswith("enter"):
                depth += 1
                assert depth == 1, f"handlers overlapped: {events}"
            else:
                depth -= 1
        assert depth == 0
        assert len(events) == 4
        saved = runner.services.metadata.get_run(result.run_id)
        assert saved is not None
        assert saved.finder_search_calls == 2

    asyncio.run(exercise())


@pytest.mark.parametrize("first_id,second_id", [("dup", "dup"), ("", "call_1")])
def test_ambiguous_tool_call_ids_fail_before_any_handler_runs(
    built_corpus: QdrantCorpusIndex,
    tmp_path: Path,
    first_id: str,
    second_id: str,
) -> None:
    """A duplicate or empty id would let a call masquerade past the cap.

    The envelope fails at the model-response boundary instead: the recorded
    attempt is kept, and no handler runs.
    """
    graph_document: GraphDocument = make_document()

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        turn = Turn(body)
        if not turn.calls_made("search_corpus") and not turn.calls_made(
            "read_provision"
        ):
            return httpx.Response(
                200,
                request=request,
                json=envelope(
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": first_id,
                                "type": "function",
                                "function": {
                                    "name": "search_corpus",
                                    "arguments": json.dumps({"phrase": "czynsz najem"}),
                                },
                            },
                            {
                                "id": second_id,
                                "type": "function",
                                "function": {
                                    "name": "read_provision",
                                    "arguments": json.dumps({"locator": LOCATOR}),
                                },
                            },
                        ],
                    },
                    input_tokens=7,
                    output_tokens=3,
                ),
            )
        raise AssertionError("the ambiguous envelope must fail the run first")

    async def exercise() -> None:
        runner = build_runner(
            graph_document.session,
            built_corpus,
            tmp_path,
            db_name=f"ambiguous-ids-{first_id or 'empty'}.sqlite3",
            transport=httpx.MockTransport(handler),
        )
        result = await runner.run(request_for(graph_document, ArmCode.OFF))

        assert result.status == "failed"
        assert result.error_code == "model_response_invalid_envelope"
        attempts = runner.services.metadata.list_attempts(result.run_id)
        assert len(attempts) == 1
        assert attempts[0].status == "success"
        assert (attempts[0].input_tokens, attempts[0].output_tokens) == (7, 3)
        record = runner._runtime.worksheet_for(result.run_id, result.call_units[0])
        assert record is not None
        assert record.entries == ()
        saved = runner.services.metadata.get_run(result.run_id)
        assert saved is not None
        assert saved.finder_search_calls == 0

    asyncio.run(exercise())


def test_malformed_tool_arguments_fail_with_attempts_retained(
    built_corpus: QdrantCorpusIndex, tmp_path: Path
) -> None:
    """Non-JSON arguments are an envelope violation, not a correctable mistake.

    Valid JSON objects that fail the tool schema stay correctable
    ``invalid_arguments`` refusals; text that is not JSON at all fails the run
    loudly, keeping the recorded attempt and running no handler.
    """
    graph_document: GraphDocument = make_document()

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        turn = Turn(body)
        if not turn.calls_made("search_corpus"):
            return httpx.Response(
                200,
                request=request,
                json=envelope(
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_0",
                                "type": "function",
                                "function": {
                                    "name": "search_corpus",
                                    "arguments": "{niepoprawny json",
                                },
                            }
                        ],
                    },
                    input_tokens=7,
                    output_tokens=3,
                ),
            )
        raise AssertionError("the malformed envelope must fail the run first")

    async def exercise() -> None:
        runner = build_runner(
            graph_document.session,
            built_corpus,
            tmp_path,
            db_name="malformed-arguments.sqlite3",
            transport=httpx.MockTransport(handler),
        )
        result = await runner.run(request_for(graph_document, ArmCode.OFF))

        assert result.status == "failed"
        assert result.error_code == "model_response_invalid_envelope"
        attempts = runner.services.metadata.list_attempts(result.run_id)
        assert len(attempts) == 1
        assert attempts[0].status == "success"
        assert (attempts[0].input_tokens, attempts[0].output_tokens) == (7, 3)
        record = runner._runtime.worksheet_for(result.run_id, result.call_units[0])
        assert record is not None
        assert record.entries == ()

    asyncio.run(exercise())
