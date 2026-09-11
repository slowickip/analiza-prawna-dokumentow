"""One reader message: what it is read as, and what that reading may name."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable
from dataclasses import replace
from uuid import UUID, uuid4

import httpx
import pytest
from model_client import mock_model_client
from starlette.testclient import TestClient
from worksheet_transport import Turn, envelope, play, tool_call

from contract_analyzer.agents.interactive_tools import RouteMessageArgs
from contract_analyzer.agents.message_router import selected_for_ask
from contract_analyzer.agents.prompts import load_prompt_bundle
from contract_analyzer.agents.services import AnalysisServices
from contract_analyzer.api import create_app
from contract_analyzer.config import Settings
from contract_analyzer.storage import (
    EventBus,
    FindingRecord,
    MetadataStore,
    RunTextStore,
)

# The routing reply for the test currently running. A test that wants a specific
# reading puts its own function here; the default reads every message as an ask.
_ROUTE_SCRIPT: list[Callable[[Turn], dict[str, object]]] = []


def _route_turn(turn: Turn) -> dict[str, object]:
    if _ROUTE_SCRIPT:
        return _ROUTE_SCRIPT[0](turn)
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
                message = tool_call(
                    "post_answer",
                    {"answer": "System odpowiada.", "cited_finding_ids": []},
                )
            else:
                message = play(turn)
        else:
            raise AssertionError("the pipeline sent a request carrying no tools")
        return httpx.Response(200, request=request, json=envelope(message))

    return httpx.MockTransport(handler)


@pytest.fixture()
def message_client(built_corpus):
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


def _parent(client: TestClient, document_id: UUID) -> dict[str, object]:
    response = client.post(
        "/api/v1/runs", json={"document_id": str(document_id), "arm": "mid"}
    )
    assert response.status_code == 202
    return _wait(client, response.json()["id"])


def test_unnamed_ask_is_scoped_to_every_finding_on_the_run() -> None:
    """The reader no longer picks a finding, so an unnamed question is about all."""
    findings = [
        FindingRecord(id=uuid4(), run_id=uuid4(), unit_id="u-1", code="consistent"),
        FindingRecord(id=uuid4(), run_id=uuid4(), unit_id="u-2", code="contradictory"),
    ]
    routed = RouteMessageArgs(intent="ask")
    assert selected_for_ask(routed, findings) == tuple(item.id for item in findings)


def test_named_ask_is_scoped_to_exactly_the_named_findings() -> None:
    findings = [
        FindingRecord(id=uuid4(), run_id=uuid4(), unit_id="u-1", code="consistent"),
        FindingRecord(id=uuid4(), run_id=uuid4(), unit_id="u-2", code="contradictory"),
    ]
    routed = RouteMessageArgs(intent="ask", finding_ids=(str(findings[1].id),))
    assert selected_for_ask(routed, findings) == (findings[1].id,)


def test_message_read_as_a_question_answers_in_place(message_client) -> None:
    client, _state, document_id, _captured = message_client
    parent = _parent(client, document_id)

    response = client.post(
        f"/api/v1/runs/{parent['id']}/message",
        json={"message": "Dlaczego system tak ocenił tę umowę?"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "ask"
    assert body["answer"]["answer"] == "System odpowiada."
    assert body.get("run") is None


def test_an_unnamed_question_reaches_every_finding_of_the_parent(
    message_client,
) -> None:
    """What the reader gets for free by not choosing: the whole analysis in scope."""
    client, state, document_id, captured = message_client
    parent = _parent(client, document_id)
    parent_id = UUID(parent["id"])
    every_finding = {
        str(item.id) for item in state.services.metadata.list_findings(parent_id)
    }
    assert len(every_finding) > 1, "needs several findings to be worth asserting"
    captured.clear()

    response = client.post(
        f"/api/v1/runs/{parent_id}/message",
        json={"message": "Co wynika z tej analizy?"},
    )
    assert response.status_code == 200

    # What the synthesizer was actually handed, not what the run happens to hold.
    chat_turn = next(turn for turn in captured if turn.task == "explain")
    grounded = {item["id"] for item in chat_turn.payload["grounding"]["findings"]}
    assert grounded == every_finding


def test_message_read_as_an_objection_starts_a_contest_child(message_client) -> None:
    client, _state, document_id, _captured = message_client
    parent = _parent(client, document_id)
    finding_id = parent["findings"][0]["id"]
    _ROUTE_SCRIPT.append(
        lambda _turn: tool_call(
            "route_message", {"intent": "contest", "finding_id": finding_id}
        )
    )

    response = client.post(
        f"/api/v1/runs/{parent['id']}/message",
        json={"message": "Nie zgadzam się z tym ustaleniem."},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "contest"
    assert body.get("answer") is None
    child = _wait(client, body["run"]["id"])
    assert child["interaction"] == "contest"
    assert child["parent_run_id"] == parent["id"]
    assert child["measurement_valid"] is False


def test_message_read_as_a_re_analysis_starts_an_analyse_child(message_client) -> None:
    client, _state, document_id, _captured = message_client
    parent = _parent(client, document_id)
    unit_id = parent["findings"][0]["unit_id"]
    _ROUTE_SCRIPT.append(
        lambda _turn: tool_call(
            "route_message", {"intent": "analyse", "unit_id": unit_id}
        )
    )

    response = client.post(
        f"/api/v1/runs/{parent['id']}/message",
        json={"message": "Przeanalizuj tę klauzulę jeszcze raz."},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "analyse"
    child = _wait(client, body["run"]["id"])
    assert child["interaction"] == "analyse"
    assert child["measurement_valid"] is False


def test_routing_is_recorded_as_its_own_unmeasured_run(message_client) -> None:
    """Routing spends a model call, so it answers to a run record like any other."""
    client, state, document_id, _captured = message_client
    parent = _parent(client, document_id)
    parent_id = UUID(parent["id"])

    response = client.post(
        f"/api/v1/runs/{parent_id}/message",
        json={"message": "Dlaczego tak?"},
    )
    assert response.status_code == 200

    route_runs = [
        run
        for run in state.services.metadata.list_runs()
        if run.parent_run_id == parent_id and run.interaction == "route"
    ]
    assert len(route_runs) == 1
    assert route_runs[0].measurement_valid is False
    assert route_runs[0].status == "completed"
    # The classifier reaches no corpus. Counting its turns as finder turns made a
    # route child report retrieval work beside zero searches and zero reads, and
    # cost is a reported dimension, so the classifier would have read as retrieval.
    assert route_runs[0].finder_tool_turns == 0
    assert route_runs[0].finder_search_calls == 0
    assert route_runs[0].provision_reads == 0
    assert route_runs[0].defaulted_characterisations is None


def test_a_target_the_run_does_not_hold_is_refused_not_accepted(
    message_client,
) -> None:
    """A contest against an unknown finding must not become a contest of
    something else."""
    client, _state, document_id, _captured = message_client
    parent = _parent(client, document_id)
    stranger = str(uuid4())
    _ROUTE_SCRIPT.append(
        lambda _turn: tool_call(
            "route_message", {"intent": "contest", "finding_id": stranger}
        )
    )

    response = client.post(
        f"/api/v1/runs/{parent['id']}/message",
        json={"message": "Nie zgadzam się."},
    )

    assert response.status_code == 422
    assert response.json()["code"] == "message_not_routable"


def test_a_message_the_router_never_commits_fails_loudly(message_client) -> None:
    """Silence is a model failure, not licence to answer a question nobody asked."""
    client, _state, document_id, _captured = message_client
    parent = _parent(client, document_id)
    _ROUTE_SCRIPT.append(
        lambda _turn: {"role": "assistant", "content": "Nie wiem, co z tym zrobić."}
    )

    response = client.post(
        f"/api/v1/runs/{parent['id']}/message",
        json={"message": "???"},
    )

    assert response.status_code == 422
    assert response.json()["code"] == "message_not_routable"


def test_the_router_is_given_the_recent_turns_of_the_conversation(
    message_client,
) -> None:
    """A composer invites follow-ups, and a follow-up needs a referent.

    The client already sends bounded history for the explanation path. Dropping it
    before routing left "przeanalizuj ja jeszcze raz" with nothing to resolve, so
    the router had to guess a target or fall back to a question about the whole run.
    """
    client, _state, document_id, captured = message_client
    parent = _parent(client, document_id)

    response = client.post(
        f"/api/v1/runs/{parent['id']}/message",
        json={
            "message": "Przeanalizuj ja jeszcze raz.",
            "history": [
                {"role": "user", "content": "Co z paragrafem o wynagrodzeniu?"},
                {"role": "assistant", "content": "System wyjasnil to ustalenie."},
            ],
        },
    )
    assert response.status_code == 200

    route_turns = [turn for turn in captured if turn.task == "interaction.route"]
    assert route_turns, "the router was never called"
    history = route_turns[-1].payload.get("history")
    assert history, "the router was given no history"
    assert [item["role"] for item in history] == ["user", "assistant"]
    assert "wynagrodzeniu" in history[0]["content"]


def test_a_unit_the_run_does_not_hold_is_a_message_that_did_not_route(
    message_client,
) -> None:
    """The tool takes targets as free text, so the run has to vet them.

    finding_id and unit_id are strings, because a model made to guess a UUID
    shape guesses worse; the handler is what makes them mean something, by
    refusing any target the run does not hold. Only the contest side of that was
    covered. Refusing rather than dropping is the point: a dropped target turns
    a re-analysis of one clause into a question about nothing in particular.
    """
    client, _state, document_id, _captured = message_client
    parent = _parent(client, document_id)
    _ROUTE_SCRIPT.append(
        lambda _turn: tool_call(
            "route_message", {"intent": "analyse", "unit_id": "u-nonexistent"}
        )
    )

    response = client.post(
        f"/api/v1/runs/{parent['id']}/message",
        json={"message": "Przeanalizuj jeszcze raz tamtą klauzulę."},
    )

    assert response.status_code == 422, response.text
    assert response.json()["code"] == "message_not_routable"


def test_an_identifier_that_is_not_one_does_not_leave_the_router(
    message_client,
) -> None:
    """The parse itself, since the handler's vetting is membership in a set.

    An unparseable target cannot be in the run's identifiers, so the handler
    already refuses it and this cannot be reached through the endpoint today. It
    is here because the conversion is a boundary between the model's free text
    and a UUID, and a boundary that raises ValueError out of a request handler is
    one nobody chose.
    """
    del message_client
    # Imported in the body: the tool digest reads this module's package, so a
    # module-level import closes the cycle.
    from contract_analyzer.agents.errors import MessageNotRoutable
    from contract_analyzer.agents.message_router import routed_uuid

    with pytest.raises(MessageNotRoutable):
        routed_uuid("to drugie ustalenie")
    assert routed_uuid("11111111-1111-4111-8111-111111111111") == UUID(
        "11111111-1111-4111-8111-111111111111"
    )


def test_a_document_swept_while_the_router_ran_is_expired_and_not_a_crash(
    message_client,
) -> None:
    """The content check happens before a model call that takes seconds.

    Routing is a real model call, and the retention sweep runs against the same
    session store, so the document the endpoint checked can be gone by the time
    the interaction it selected starts. The session was read without a guard,
    which turned an ordinary expiry into a 500 on the reader's screen.
    """
    client, state, document_id, _captured = message_client
    parent = _parent(client, document_id)
    finding_id = parent["findings"][0]["id"]

    def route_then_sweep(_turn):
        state.services.sessions.pop(document_id, None)
        return tool_call(
            "route_message", {"intent": "contest", "finding_id": finding_id}
        )

    _ROUTE_SCRIPT.append(route_then_sweep)

    response = client.post(
        f"/api/v1/runs/{parent['id']}/message",
        json={"message": "Nie zgadzam się z tym ustaleniem."},
    )

    assert response.status_code == 410, response.text
    assert response.json()["code"] == "content_expired"


def test_a_reader_message_is_answered_while_other_work_is_running(
    message_client,
) -> None:
    """Parallel work is the ordinary local case, and this endpoint alone refused it.

    /message gated on measured_mode, which the server sets once and never
    unsets, so a question was refused whenever any task was in flight -- while
    the endpoints it replaced, doing the same thing, never were. The isolation a
    measurement needs comes from the seal instead: sealed refuses every
    interactive action outright, and POST /runs admits one run at a time.
    """
    client, state, document_id, _captured = message_client
    parent = _parent(client, document_id)
    # A stand-in for whatever else the operator has running; taken back out so
    # shutdown does not try to cancel it.
    busy = uuid4()
    state.tasks.active_tasks[busy] = asyncio.Event()

    _ROUTE_SCRIPT.append(lambda _turn: tool_call("route_message", {"intent": "ask"}))
    try:
        response = client.post(
            f"/api/v1/runs/{parent['id']}/message",
            json={"message": "Czego dotyczy to ustalenie?"},
        )
    finally:
        state.tasks.active_tasks.pop(busy, None)

    assert response.status_code == 200, response.text
    assert response.json()["intent"] == "ask"


def test_a_sealed_batch_refuses_the_message_endpoint_outright(message_client) -> None:
    """What a measurement needs is the seal, not a slot the endpoint arbitrates."""
    client, state, document_id, _captured = message_client
    parent = _parent(client, document_id)
    state.settings = replace(state.settings, evaluation_batch_open=True)

    response = client.post(
        f"/api/v1/runs/{parent['id']}/message",
        json={"message": "Czego dotyczy to ustalenie?"},
    )

    assert response.status_code == 409, response.text
    assert response.json()["code"] == "evaluation_batch_open"
