"""Post-run interactions preserve measured parents and stay out of parity."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, cast
from uuid import UUID, uuid4

import httpx
import pytest
from model_client import mock_model_client
from pydantic import ValidationError
from starlette.testclient import TestClient
from worksheet_transport import Turn, envelope, play, tool_call

from contract_analyzer.agents.parity import compare_arm_parity
from contract_analyzer.agents.prompts import load_prompt_bundle
from contract_analyzer.agents.runner import AnalysisRunner
from contract_analyzer.agents.services import AnalysisServices
from contract_analyzer.api import create_app
from contract_analyzer.config import Settings
from contract_analyzer.domain import ArmCode
from contract_analyzer.storage import (
    CostRecord,
    EventBus,
    MetadataStore,
    RunRecord,
    RunTextStore,
)

# A test that needs the reader's synthesizer to do something particular puts its
# own script here for the length of that test.
_CHAT_SCRIPT: list[Callable[[Turn], dict[str, object]]] = []
_ROUTE_SCRIPT: list[Callable[[Turn], dict[str, object]]] = []


def _chat_turn(turn: Turn) -> dict[str, object]:
    """Read one grounded provision, then answer: the ordinary reader exchange."""
    if _CHAT_SCRIPT:
        return _CHAT_SCRIPT[0](turn)
    grounded = turn.payload.get("grounding", {})
    locators = [item["locator"] for item in grounded.get("provisions", [])]
    if locators and not turn.calls_made("read_provision"):
        return tool_call("read_provision", {"locator": locators[0]})
    return tool_call(
        "post_answer", {"answer": "System odpowiada.", "cited_finding_ids": []}
    )


def _route_turn(turn: Turn) -> dict[str, object]:
    if _ROUTE_SCRIPT:
        return _ROUTE_SCRIPT.pop(0)(turn)
    return tool_call("route_message", {"intent": "ask"})


def _transport(captured: list[Turn]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        if "tools" in payload:
            turn = Turn(payload)
            captured.append(turn)
            if turn.task == "interaction.route":
                message = _route_turn(turn)
            elif turn.task == "explain":
                message = _chat_turn(turn)
            else:
                message = play(turn)
        else:
            raise AssertionError("the pipeline sent a request carrying no tools")
        return httpx.Response(200, request=request, json=envelope(message))

    return httpx.MockTransport(handler)


@pytest.fixture()
def interaction_client(built_corpus):
    _CHAT_SCRIPT.clear()
    _ROUTE_SCRIPT.clear()
    captured: list[Turn] = []
    settings = Settings(model_api_key="test-secret-key")
    services = AnalysisServices(
        settings=settings,
        corpus=built_corpus,
        client=mock_model_client(_transport(captured), settings=settings),
        metadata=MetadataStore(),
        prompt_bundle=load_prompt_bundle(),
        text_store=RunTextStore(),
        events=EventBus(),
    )
    app = create_app(settings, services)
    with TestClient(app) as client:
        upload = client.post(
            "/api/v1/documents",
            files={
                "file": (
                    "contract.txt",
                    "§ 1. Czynsz najmu.\n§ 2. Kaucja.\n§ 3. Wypowiedzenie.",
                )
            },
        )
        assert upload.status_code == 201
        yield client, app.state.app_state, UUID(upload.json()["id"]), captured
    _CHAT_SCRIPT.clear()
    _ROUTE_SCRIPT.clear()


def _wait(client: TestClient, run_id: str) -> dict[str, object]:
    for _ in range(200):
        response = client.get(f"/api/v1/runs/{run_id}")
        assert response.status_code == 200
        body = response.json()
        if body["status"] != "running":
            return body
        time.sleep(0.01)
    raise AssertionError("run did not finish")


def _parent(
    client: TestClient, document_id: UUID, arm: str = "mid"
) -> dict[str, object]:
    response = client.post(
        "/api/v1/runs", json={"document_id": str(document_id), "arm": arm}
    )
    assert response.status_code == 202
    return _wait(client, response.json()["id"])


def _bytes(value: object) -> bytes:
    return json.dumps(value, default=str, sort_keys=True).encode()


def test_contest_child_is_one_unit_and_parent_is_immutable(interaction_client) -> None:
    client, state, document_id, captured = interaction_client
    parent = _parent(client, document_id)
    parent_id = UUID(parent["id"])
    finding_id = parent["findings"][0]["id"]
    record_before = _bytes(asdict(state.services.metadata.get_run(parent_id)))
    findings_before = _bytes(
        [asdict(item) for item in state.services.metadata.list_findings(parent_id)]
    )

    _ROUTE_SCRIPT.append(
        lambda _turn: tool_call(
            "route_message", {"intent": "contest", "finding_id": finding_id}
        )
    )
    response = client.post(
        f"/api/v1/runs/{parent_id}/message",
        json={"message": "System ma rozważyć sprzeciw."},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "contest"
    child = _wait(client, body["run"]["id"])
    assert child["parent_run_id"] == str(parent_id)
    assert child["interaction"] == "contest"
    assert child["measurement_valid"] is False
    assert child["metrics"]["units_total"] == 1
    child_turns = [
        turn
        for turn in captured
        if turn.request.get("metadata") is None
        and turn.task in {"researcher.search", "verifier.relevance"}
        and any(entry.get("kind") == "user_note" for entry in turn.entries("user_note"))
    ]
    roles = {turn.task for turn in child_turns}
    assert roles == {"researcher.search", "verifier.relevance"}
    assert all(
        "user_note jest sprzeciwem" in str(turn.messages[0]["content"])
        and "nigdy nie jest werdyktem" in str(turn.messages[0]["content"])
        for turn in child_turns
    )
    assert _bytes(asdict(state.services.metadata.get_run(parent_id))) == record_before
    assert (
        _bytes(
            [asdict(item) for item in state.services.metadata.list_findings(parent_id)]
        )
        == findings_before
    )


def test_analyse_off_uses_whole_document_unit(interaction_client) -> None:
    client, state, document_id, _ = interaction_client
    parent = _parent(client, document_id, "off")
    parent_id = UUID(parent["id"])
    whole_id = state.services.sessions[document_id].whole_document_id
    _ROUTE_SCRIPT.append(
        lambda _turn: tool_call(
            "route_message", {"intent": "analyse", "unit_id": whole_id}
        )
    )
    response = client.post(
        f"/api/v1/runs/{parent_id}/message",
        json={"message": "System ma ponowić analizę."},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "analyse"
    child = _wait(client, body["run"]["id"])
    assert child["findings"][0]["unit_id"] == whole_id


def test_worksheet_returns_entries_then_expires(interaction_client) -> None:
    client, state, document_id, _ = interaction_client
    parent = _parent(client, document_id)
    response = client.get(f"/api/v1/runs/{parent['id']}/worksheet")
    assert response.status_code == 200
    assert response.json()["units"][0]["entries"]
    state.services.text_store.expire(state.text_keys[document_id])
    expired = client.get(f"/api/v1/runs/{parent['id']}/worksheet")
    assert expired.status_code == 410


def test_evaluation_batch_seals_all_interactive_endpoints(interaction_client) -> None:
    client, state, document_id, _ = interaction_client
    parent = _parent(client, document_id)
    state.settings = replace(state.settings, evaluation_batch_open=True)
    paths = [
        ("get", f"/api/v1/runs/{parent['id']}/worksheet", None),
        (
            "post",
            f"/api/v1/runs/{parent['id']}/message",
            {"message": "Dlaczego?"},
        ),
    ]
    for method, path, body in paths:
        response = client.request(method, path, json=body)
        assert response.status_code == 409
        assert response.json()["code"] == "evaluation_batch_open"


def _run(arm: ArmCode, *, parent: UUID | None = None) -> RunRecord:
    return RunRecord(
        id=uuid4(),
        document_id=uuid4(),
        arm=arm,
        input_hash="same",
        config_version="config",
        prompt_bundle_version="prompts",
        corpus_snapshot_id="corpus",
        tool_bundle_version="tools",
        requested_model="model",
        returned_model="model-version",
        parameters={},
        retry_policy="bounded-3",
        concurrency=1,
        wall_budget_seconds=10,
        measurement_valid=parent is None,
        parent_run_id=parent,
        interaction="analyse" if parent else None,
        cost=CostRecord.unknown("unknown"),
        status="completed",
    )


def test_parity_ignores_interactive_children_even_if_input_is_mixed() -> None:
    roots = [_run(arm) for arm in ArmCode]
    children = [_run(root.arm, parent=root.id) for root in roots]
    children[0] = replace(children[0], measurement_valid=True)
    assert compare_arm_parity([*roots, *children]).measurement_valid is True


def test_an_ask_joins_the_registry_without_queueing_behind_it(
    interaction_client,
) -> None:
    """Chat drives the model, so shutdown must find it -- but it no longer waits."""
    client, state, document_id, _ = interaction_client
    parent = _parent(client, document_id)
    parent_id = UUID(parent["id"])
    body = {"message": "Na czym oparto ustalenie?"}

    # A run in flight is any entry in the registry; an ask now proceeds beside it.
    occupied = uuid4()
    state.tasks.active_tasks[occupied] = cast(Any, object())
    beside = client.post(f"/api/v1/runs/{parent_id}/message", json=body)
    state.tasks.active_tasks.pop(occupied)

    assert beside.status_code == 200, beside.text

    original = state.runner.explain
    held: list[int] = []

    async def watched(request: object) -> object:
        held.append(len(state.tasks.active_tasks))
        return await original(request)

    state.runner.explain = watched  # type: ignore[method-assign]
    answered = client.post(f"/api/v1/runs/{parent_id}/message", json=body)
    state.runner.explain = original  # type: ignore[method-assign]

    assert answered.status_code == 200
    assert held == [1]
    assert not state.tasks.active_tasks


def test_a_chat_corpus_read_leaves_the_event_loop_and_is_counted(
    interaction_client,
) -> None:
    """The corpus client blocks, so a reader's question must not run it inline."""
    client, state, document_id, _ = interaction_client
    parent = _parent(client, document_id)
    parent_id = UUID(parent["id"])
    finding = next(item for item in parent["findings"] if item.get("legal_locators"))
    locator = finding["legal_locators"][0]
    corpus = state.services.corpus
    threads: list[int] = []
    loop_thread: list[int] = []
    original_read = corpus.read
    original_explain = state.runner.explain

    def watched(value: str) -> object:
        threads.append(threading.get_ident())
        return original_read(value)

    async def watched_explain(chat_request: object) -> object:
        # Recorded from inside a coroutine, so this is the thread the loop runs on.
        loop_thread.append(threading.get_ident())
        return await original_explain(chat_request)

    corpus.read = watched  # type: ignore[method-assign]
    state.runner.explain = watched_explain  # type: ignore[method-assign]
    answered = client.post(
        f"/api/v1/runs/{parent_id}/message",
        json={"message": "Co mówi przepis?"},
    )
    corpus.read = original_read  # type: ignore[method-assign]
    state.runner.explain = original_explain  # type: ignore[method-assign]

    assert answered.status_code == 200
    assert threads, "the ask never read the corpus"
    assert loop_thread and all(ident != loop_thread[0] for ident in threads)
    body = answered.json()["answer"]
    assert body["corpus_consulted"] is True
    child = client.get(f"/api/v1/runs/{body['interaction_run_id']}")
    # The eager grounding is handed to the model unasked; only what the reader's
    # synthesizer opened for itself counts as a read.
    assert child.json()["metrics"]["provision_reads"] == 1
    assert locator


def test_an_ask_that_cannot_ground_itself_leaves_no_running_record(
    interaction_client,
) -> None:
    """A failure before the first model turn still terminalises the child."""
    client, state, document_id, _ = interaction_client
    parent = _parent(client, document_id)
    parent_id = UUID(parent["id"])
    before = {item.id for item in state.services.metadata.list_runs()}
    corpus = state.services.corpus
    original = corpus.read

    def gone(value: str) -> object:
        raise KeyError(value)

    corpus.read = gone  # type: ignore[method-assign]
    # The app renders an unhandled failure as an Error body; this client reports
    # that response instead of re-raising it inside the test.
    reporting = TestClient(client.app, raise_server_exceptions=False)
    response = reporting.post(
        f"/api/v1/runs/{parent_id}/message",
        json={"message": "Co mówi przepis?"},
    )
    corpus.read = original  # type: ignore[method-assign]

    assert response.status_code == 500
    assert response.json()["code"] == "internal_error"
    children = [
        item for item in state.services.metadata.list_runs() if item.id not in before
    ]
    ask_children = [item for item in children if item.interaction == "ask"]
    assert len(ask_children) == 1
    assert ask_children[0].status == "failed"
    assert ask_children[0].error_code == "evidence_locator_unknown"
    assert all(item.status != "running" for item in children)
    assert not state.tasks.active_tasks


def test_a_worksheet_is_absent_without_being_expired(interaction_client) -> None:
    """410 means the window closed; an ask child simply analyses no unit."""
    client, state, document_id, _ = interaction_client
    parent = _parent(client, document_id)
    parent_id = UUID(parent["id"])
    answered = client.post(
        f"/api/v1/runs/{parent_id}/message",
        json={"message": "Co dalej?"},
    )
    assert answered.status_code == 200
    child_id = answered.json()["answer"]["interaction_run_id"]

    empty = client.get(f"/api/v1/runs/{child_id}/worksheet")
    assert empty.status_code == 200
    assert empty.json()["units"] == []

    running = replace(
        state.services.metadata.get_run(parent_id), id=uuid4(), status="running"
    )
    state.services.metadata.create_run(running)
    unfinished = client.get(f"/api/v1/runs/{running.id}/worksheet")
    assert unfinished.status_code == 409
    assert unfinished.json()["code"] == "run_not_finished"


def test_chat_reads_only_the_units_its_findings_sit_on(interaction_client) -> None:
    """Explaining a selected finding is not a way to analyse another clause."""
    client, state, document_id, _ = interaction_client
    parent = _parent(client, document_id)
    parent_id = UUID(parent["id"])
    findings = parent["findings"]
    selected = findings[0]
    other = next(item for item in findings if item["unit_id"] != selected["unit_id"])
    results: list[dict[str, object]] = []

    def script(turn: Turn) -> dict[str, object]:
        if not turn.calls_made("read_unit"):
            return tool_call("read_unit", {"unit_id": other["unit_id"]})
        results.extend(turn.tool_results)
        return tool_call(
            "post_answer", {"answer": "System odpowiada.", "cited_finding_ids": []}
        )

    _CHAT_SCRIPT.append(script)
    _ROUTE_SCRIPT.append(
        lambda _turn: tool_call(
            "route_message",
            {"intent": "ask", "finding_ids": (selected["id"],)},
        )
    )
    try:
        answered = client.post(
            f"/api/v1/runs/{parent_id}/message",
            json={"message": "A co z inną klauzulą?"},
        )
    finally:
        _CHAT_SCRIPT.clear()
        _ROUTE_SCRIPT.clear()

    assert answered.status_code == 200
    assert [item.get("error") for item in results] == ["unknown_unit"]
    assert all("text" not in item for item in results)


def _note_turns(captured: list[Turn]) -> list[Turn]:
    """The role turns of an interaction child: those carrying the reader's note."""
    return [
        turn
        for turn in captured
        if turn.task in {"researcher.search", "verifier.relevance"}
        and turn.entries("user_note")
    ]


def test_a_reanalysis_note_reaches_the_roles_as_guidance_not_an_objection(
    interaction_client,
) -> None:
    """Only a contested finding comes with an objection; analyse is a fresh pass."""
    client, state, document_id, captured = interaction_client
    parent = _parent(client, document_id)
    parent_id = UUID(parent["id"])
    unit_id = parent["findings"][0]["unit_id"]
    captured.clear()

    _ROUTE_SCRIPT.append(
        lambda _turn: tool_call(
            "route_message", {"intent": "analyse", "unit_id": unit_id}
        )
    )
    response = client.post(
        f"/api/v1/runs/{parent_id}/message",
        json={"message": "System ma ponowić analizę tej klauzuli."},
    )
    assert response.status_code == 200
    child = _wait(client, response.json()["run"]["id"])
    assert child["interaction"] == "analyse"

    prompts = [str(turn.messages[0]["content"]) for turn in _note_turns(captured)]
    assert prompts
    assert all("sprzeciwem użytkownika" not in prompt for prompt in prompts)
    assert all(
        "wskazówką użytkownika do ponownej analizy" in prompt for prompt in prompts
    )
    assert all("nie jest ani sprzeciwem" in prompt for prompt in prompts)
    entries = [
        entry for turn in _note_turns(captured) for entry in turn.entries("user_note")
    ]
    assert {entry["purpose"] for entry in entries} == {"analyse"}


def test_the_protocol_reports_the_interactive_bounds_the_code_enforces() -> None:
    """The protocol asserts these numbers; nothing was checking it still could.

    interactive_phase.bounds is what a reader of the protocol takes the
    interactive phase to be bounded by, and the constants beside the tools are
    what actually bounds it. Every one of these has moved at least once -- the
    researcher's cap arrived with the role split, the router's with the composer
    -- and a number that drifts here is a protocol claiming a limit the system
    does not apply.
    """
    # Imported here for the same reason tools.py does: the bundle digest reads
    # this module, so a module-level import closes the cycle.
    from contract_analyzer.agents.interactive_tools import interactive_tool_policy

    protocol = json.loads(
        (
            Path(__file__).resolve().parents[2] / "evaluation-data" / "protocol.json"
        ).read_text(encoding="utf-8")
    )
    declared = protocol["run_parameters"]["interactive_phase"]["bounds"]
    enforced = interactive_tool_policy()

    assert declared == {name: enforced[name] for name in declared}, (
        "protocol.json declares an interactive bound the code does not enforce"
    )
    # And the other way: a bound the code grew must reach the protocol.
    assert set(declared) == {
        name for name in enforced if name.endswith(("_turns", "_questions", "_seconds"))
    }


def test_a_routed_message_answers_or_opens_a_run_but_never_both() -> None:
    """The client narrows on this and renders nothing when it does not hold.

    ChatSection reads answer when the intent is ask and run otherwise, so a
    response carrying both, neither, or the wrong one for its intent disappears
    in the browser with no error anywhere. The rule was written in the docstring
    and in OpenAPI and enforced in neither.
    """
    from contract_analyzer.api.schemas import ChatResponse, MessageResponse

    answer = ChatResponse(
        answer="a",
        cited_finding_ids=[],
        interaction_run_id=uuid4(),
        corpus_consulted=False,
    )
    assert MessageResponse(intent="ask", answer=answer).answer is answer

    for label, kwargs in (
        ("neither", {"intent": "ask"}),
        ("an answer for a contest", {"intent": "contest", "answer": answer}),
    ):
        with pytest.raises(ValidationError):
            MessageResponse(**kwargs)  # type: ignore[arg-type]
        assert label


def test_the_router_refuses_a_finding_the_parent_does_not_have(
    interaction_client,
) -> None:
    """A contest aimed at a finding this run never made is refused, not started.

    The router takes identifiers as free text, so nothing before it guarantees the
    target exists. It checks the catalogue itself and refuses the commit; a router
    that spends its turns refusing has read nothing, which is what 422
    message_not_routable means. The point is that no child run is started: a
    mis-routed objection would otherwise bill a full model run against a finding
    that is not there.
    """
    client, state, document_id, _captured = interaction_client
    parent = _parent(client, document_id)
    parent_id = UUID(str(parent["id"]))
    runs_before = len(state.services.metadata.list_runs())

    stranger = str(uuid4())
    for _ in range(2):
        _ROUTE_SCRIPT.append(
            lambda _turn: tool_call(
                "route_message", {"intent": "contest", "finding_id": stranger}
            )
        )
    response = client.post(
        f"/api/v1/runs/{parent_id}/message",
        json={"message": "Nie zgadzam się z tym ustaleniem."},
    )

    assert response.status_code == 422
    assert response.json()["code"] == "message_not_routable"
    started = [
        run
        for run in state.services.metadata.list_runs()
        if run.parent_run_id == parent_id and run.interaction == "contest"
    ]
    assert started == []
    assert len(state.services.metadata.list_runs()) >= runs_before


def test_the_router_refuses_a_unit_the_document_does_not_have(
    interaction_client,
) -> None:
    """The same catalogue gate on the analyse side, where the target is a unit."""
    client, state, document_id, _captured = interaction_client
    parent = _parent(client, document_id)
    parent_id = UUID(str(parent["id"]))

    for _ in range(2):
        _ROUTE_SCRIPT.append(
            lambda _turn: tool_call(
                "route_message",
                {"intent": "analyse", "unit_id": "unit-that-is-not-here"},
            )
        )
    response = client.post(
        f"/api/v1/runs/{parent_id}/message",
        json={"message": "Przeanalizuj tę jednostkę jeszcze raz."},
    )

    assert response.status_code == 422
    assert response.json()["code"] == "message_not_routable"
    assert [
        run
        for run in state.services.metadata.list_runs()
        if run.parent_run_id == parent_id and run.interaction == "analyse"
    ] == []


def test_a_message_waits_for_its_parent_to_finish(interaction_client) -> None:
    """A parent still running has no findings to talk about yet: 409."""
    client, _state, document_id, _captured = interaction_client
    started = client.post(
        "/api/v1/runs", json={"document_id": str(document_id), "arm": "mid"}
    )
    assert started.status_code == 202
    run_id = started.json()["id"]

    response = client.post(
        f"/api/v1/runs/{run_id}/message",
        json={"message": "Co to znaczy?"},
    )

    assert response.status_code == 409
    assert response.json()["code"] == "run_not_finished"
    _wait(client, run_id)


def test_purging_mid_ask_stops_the_ask_and_leaves_no_run_behind(
    interaction_client,
) -> None:
    """An erasure reaches a chat call already in flight, not only analyses.

    A routing or explaining task is registered under a slot of its own, because
    its run does not exist yet when the task is created. An erasure that looked
    only at run records would miss it and answer 204 while a model call carried
    on over the document -- the explainer holds the session object, and its
    grounding already carries the text the reader just asked to have removed.
    """
    client, state, document_id, _captured = interaction_client
    parent = _parent(client, document_id)
    parent_id = UUID(str(parent["id"]))

    release = threading.Event()
    started = threading.Event()
    original_explain = AnalysisRunner.explain

    async def blocking_explain(self, request):  # type: ignore[no-untyped-def]
        started.set()
        while not release.is_set():
            await asyncio.sleep(0.01)
        return await original_explain(self, request)

    AnalysisRunner.explain = blocking_explain  # type: ignore[method-assign]
    asked: list[int] = []

    def ask() -> None:
        response = client.post(
            f"/api/v1/runs/{parent_id}/message",
            json={"message": "Wyjaśnij to ustalenie."},
        )
        asked.append(response.status_code)

    resident = lambda: [  # noqa: E731
        run_id
        for run_id, active in state.runner._runtime._runs.items()
        if active.session.document_id == document_id
    ]

    caller = threading.Thread(target=ask, daemon=True)
    try:
        caller.start()
        assert started.wait(timeout=10.0)
        # The ask is registered under a slot, not under any run id, which is
        # exactly what an erasure reading run records alone would fail to find.
        tracked = state.tasks.task_documents.items()
        slots = {key for key, doc in tracked if doc == document_id}
        assert slots
        assert not slots & {run.id for run in state.services.metadata.list_runs()}

        assert client.delete(f"/api/v1/documents/{document_id}").status_code == 204

        # Asserted while the ask is still blocked: were it released first, it
        # would tidy itself away and these would hold however the purge behaved.
        assert document_id not in set(state.tasks.task_documents.values())
        assert not (slots & set(state.tasks.active_tasks))
        assert resident() == []
    finally:
        release.set()
        AnalysisRunner.explain = original_explain  # type: ignore[method-assign]
        caller.join(timeout=10.0)

    # The ask never answered, because the work it was doing was erased under it.
    assert asked != [200]
    assert [
        run
        for run in state.services.metadata.list_runs()
        if run.document_id == document_id
    ] == []


def test_purging_erases_an_interactive_child_that_is_still_running(
    interaction_client,
) -> None:
    """A child run in flight leaves nothing resident after the erasure.

    Cancelling the task is not enough on its own: the ActiveRun behind it holds
    the RunRequest that started it, and for a contest that request carries the
    note the reader typed. An erasure that deleted the database record and left
    the run in the process would keep the reader's own words in memory after
    telling them the document was gone.
    """
    client, state, document_id, _captured = interaction_client
    parent = _parent(client, document_id)
    parent_id = UUID(str(parent["id"]))
    finding_id = parent["findings"][0]["id"]

    release = threading.Event()
    started = threading.Event()
    original_execute = AnalysisRunner.execute_run

    async def blocking_execute(self, prepared):  # type: ignore[no-untyped-def]
        started.set()
        while not release.is_set():
            await asyncio.sleep(0.01)
        return await original_execute(self, prepared)

    resident = lambda: [  # noqa: E731
        run_id
        for run_id, active in state.runner._runtime._runs.items()
        if active.session.document_id == document_id
    ]

    AnalysisRunner.execute_run = blocking_execute  # type: ignore[method-assign]
    try:
        _ROUTE_SCRIPT.append(
            lambda _turn: tool_call(
                "route_message", {"intent": "contest", "finding_id": finding_id}
            )
        )
        opened = client.post(
            f"/api/v1/runs/{parent_id}/message",
            json={"message": "Nie zgadzam się z tym ustaleniem."},
        )
        assert opened.status_code == 200
        child_id = UUID(str(opened.json()["run"]["id"]))
        assert started.wait(timeout=10.0)
        assert child_id in resident()

        assert client.delete(f"/api/v1/documents/{document_id}").status_code == 204

        # Checked while the child would otherwise still be running.
        assert resident() == []
        assert document_id not in set(state.tasks.task_documents.values())
    finally:
        release.set()
        AnalysisRunner.execute_run = original_execute  # type: ignore[method-assign]

    assert client.get(f"/api/v1/runs/{child_id}").status_code == 404


def test_work_started_during_an_erasure_is_refused(interaction_client) -> None:
    """Nothing new is admitted for a document once its erasure has begun.

    Cancelling the work already running takes awaits, and the event loop serves
    other requests during them. A run or a message admitted in that window would
    register after the erasure had listed what to stop, so it would outlive the
    erasure still holding the session -- working over text the reader has been
    told is gone. The document is claimed before the list is taken.
    """
    client, state, document_id, _captured = interaction_client
    parent = _parent(client, document_id)
    parent_id = UUID(str(parent["id"]))
    finding_id = parent["findings"][0]["id"]

    started = threading.Event()
    cancelling = threading.Event()
    resume = threading.Event()
    original_execute = AnalysisRunner.execute_run

    async def slow_to_cancel(self, prepared):  # type: ignore[no-untyped-def]
        started.set()
        try:
            while True:
                await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            # Hold the erasure inside its own await, which is the window the
            # racing request has to slip through.
            cancelling.set()
            for _ in range(500):
                if resume.is_set():
                    break
                await asyncio.sleep(0.01)
            raise

    AnalysisRunner.execute_run = slow_to_cancel  # type: ignore[method-assign]
    purged: list[int] = []

    def purge() -> None:
        purged.append(client.delete(f"/api/v1/documents/{document_id}").status_code)

    eraser = threading.Thread(target=purge, daemon=True)
    try:
        _ROUTE_SCRIPT.append(
            lambda _turn: tool_call(
                "route_message", {"intent": "contest", "finding_id": finding_id}
            )
        )
        opened = client.post(
            f"/api/v1/runs/{parent_id}/message",
            json={"message": "Nie zgadzam się z tym ustaleniem."},
        )
        assert opened.status_code == 200
        assert started.wait(timeout=10.0)

        eraser.start()
        assert cancelling.wait(timeout=10.0)

        # The erasure is mid-drain. Both doors are shut.
        racing_run = client.post(
            "/api/v1/runs", json={"document_id": str(document_id), "arm": "mid"}
        )
        assert racing_run.status_code == 410
        assert racing_run.json()["code"] == "content_expired"

        racing_message = client.post(
            f"/api/v1/runs/{parent_id}/message",
            json={"message": "A co z tym?"},
        )
        assert racing_message.status_code == 410
        assert racing_message.json()["code"] == "content_expired"

        assert document_id not in set(state.tasks.task_documents.values())
    finally:
        resume.set()
        eraser.join(timeout=15.0)
        AnalysisRunner.execute_run = original_execute  # type: ignore[method-assign]

    assert purged == [204]
    assert state.runner.drop_document_runs(document_id) == 0
