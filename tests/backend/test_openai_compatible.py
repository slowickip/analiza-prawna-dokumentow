"""The model connector: request contract, attempt policy, telemetry.

The mock transport is ``httpx2``'s rather than ``httpx``'s because that is the
HTTP stack the ``openai`` SDK uses; the project's own ``httpx`` is a separate
distribution and its transports do not apply here.
Tests of the pipeline script model answers in plain ``httpx`` through
``model_client.mock_model_client``; these ones stay on the real wire.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import replace
from typing import Any
from uuid import uuid4

import httpx2
import pytest

from contract_analyzer.config import RunConfig, Settings
from contract_analyzer.model import (
    InvalidModelResponse,
    ModelCallFailed,
    ModelDeadlineExceeded,
    ModelIdentityChanged,
    ToolConversation,
    ToolDefinition,
    hash_json,
    token_counts,
    tool_payload,
    usage_from_body,
)
from contract_analyzer.openai_compatible import OpenAICompatibleClient

BASE_URL = "https://openrouter.ai/api/v1"
MODEL = "deepseek-v4-flash"
RETURNED_MODEL = "deepseek-v4-flash-20260828"


def settings() -> Settings:
    return Settings(
        model_api_key="test-secret-key",
        model_base_url=BASE_URL,
        model_name=MODEL,
    )


def call(
    *,
    run_id: Any = None,
    prompt: str = "fixture prompt",
    parameters: dict[str, object] | None = None,
) -> ToolConversation:
    return ToolConversation(
        run_id=run_id or uuid4(),
        messages=({"role": "user", "content": prompt},),
        tools=(
            ToolDefinition(
                name="search_corpus",
                description="Search the legal corpus",
                parameters={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                    "additionalProperties": False,
                },
            ),
        ),
        prompt_version="basis-v1",
        temperature=0.0,
        parameters={"seed": 7} if parameters is None else parameters,
    )


def envelope(
    *,
    model: str = RETURNED_MODEL,
    content: str | None = "ok",
    tool_calls: object | None = None,
    usage: object | None = {"input_tokens": 11, "output_tokens": 4},
    status: str = "completed",
    output: object | None = None,
) -> dict[str, object]:
    """One provider answer on the wire.

    Every answer carries a reasoning item, because every real one does: the
    client has to walk past it to find the text and the calls, and a fixture
    without it would not notice the client starting to record it.
    """
    items: list[object] = [{"type": "reasoning", "id": "rs_1", "summary": []}]
    if content is not None:
        items.append(
            {
                "type": "message",
                "id": "msg_1",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {"type": "output_text", "text": content, "annotations": []}
                ],
            }
        )
    items.extend(tool_calls or ())
    body: dict[str, object] = {
        "id": "resp-1",
        "object": "response",
        "created_at": 0,
        "model": model,
        "status": status,
        "output": items if output is None else output,
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
    }
    if usage is not None:
        body["usage"] = usage
    return body


def function_call(
    *,
    call_id: str = "call_1",
    name: str = "search_corpus",
    arguments: str = '{"query":"czynsz"}',
    identifier: str = "fc_1",
) -> dict[str, object]:
    return {
        "type": "function_call",
        "id": identifier,
        "call_id": call_id,
        "name": name,
        "arguments": arguments,
    }


def make_client(
    handler: Any,
    *,
    attempts: int = 3,
    model_settings: Settings | None = None,
) -> OpenAICompatibleClient:
    resolved = model_settings or settings()
    return OpenAICompatibleClient(
        resolved,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
        max_attempts=attempts,
    )


def responder(*bodies: tuple[int, dict[str, object]]) -> Any:
    """Answer each request with the next scripted (status, body) pair."""
    scripted = list(bodies)

    def handler(request: httpx2.Request) -> httpx2.Response:
        status, body = scripted.pop(0) if len(scripted) > 1 else scripted[0]
        return httpx2.Response(status, json=body, request=request)

    return handler


# ── The request ──


def test_a_turn_sends_the_shared_payload_and_reports_what_came_back() -> None:
    seen: list[dict[str, object]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(json.loads(request.content))
        return httpx2.Response(200, json=envelope(), request=request)

    client = make_client(handler)
    request = call(prompt="genuine document text")
    result = asyncio.run(client.converse(request))
    asyncio.run(client.aclose())

    assert seen == [tool_payload(request, MODEL)]
    assert result.content == "ok"
    assert result.tool_calls == ()
    assert result.requested_model == MODEL
    assert result.returned_model == RETURNED_MODEL
    assert (result.input_tokens, result.output_tokens) == (11, 4)

    attempt = result.attempts[0]
    assert attempt.status == "success"
    assert attempt.retry_number == 0
    assert attempt.error_code is None
    assert attempt.returned_model == RETURNED_MODEL
    assert attempt.prompt_version == "basis-v1"
    assert attempt.parameters == {"seed": 7}
    assert attempt.latency_ms >= 0


def test_prompt_hash_identifies_the_request_not_its_encoding() -> None:
    """The recorded hash is of the logical request, so the library cannot move it."""
    request = call(prompt="identical prompt")
    client = make_client(responder((200, envelope())))
    result = asyncio.run(client.converse(request))
    asyncio.run(client.aclose())

    assert result.attempts[0].prompt_hash == hash_json(tool_payload(request, MODEL))


def test_provider_extension_parameters_reach_the_wire_unchanged() -> None:
    seen: list[dict[str, object]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(json.loads(request.content))
        return httpx2.Response(200, json=envelope(), request=request)

    client = make_client(handler)
    asyncio.run(
        client.converse(
            call(parameters={"provider_extension": {"mode": "strict", "level": 2}})
        )
    )
    asyncio.run(client.aclose())

    assert seen[0]["provider_extension"] == {"mode": "strict", "level": 2}


def test_reserved_parameters_are_refused() -> None:
    client = make_client(responder((200, envelope())))
    request = call(parameters={"model": "other"})
    with pytest.raises(ValueError, match="reserved model parameters"):
        asyncio.run(client.converse(request))
    asyncio.run(client.aclose())


# ── Tool turns ──


def test_tool_turn_reports_calls_and_sends_the_shared_payload() -> None:
    seen: list[dict[str, object]] = []
    tool_calls = [function_call(arguments='{"query": "najem lokalu"}')]

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(json.loads(request.content))
        return httpx2.Response(
            200,
            json=envelope(content="szukam", tool_calls=tool_calls),
            request=request,
        )

    client = make_client(handler)
    request = call()
    turn = asyncio.run(client.converse(request))
    asyncio.run(client.aclose())

    assert seen == [tool_payload(request, MODEL)]
    assert turn.content == "szukam"
    assert len(turn.tool_calls) == 1
    invocation = turn.tool_calls[0]
    assert (invocation.id, invocation.name) == ("call_1", "search_corpus")
    assert json.loads(invocation.arguments) == {"query": "najem lokalu"}
    assert turn.returned_model == RETURNED_MODEL


def test_tool_arguments_reach_the_caller_verbatim_and_in_wire_order() -> None:
    """The tool executor decides what malformed arguments mean, not the client."""
    tool_calls = [
        function_call(call_id="call_1", arguments="{not json"),
        function_call(
            call_id="call_2",
            identifier="fc_2",
            arguments='{"query": "najem \\u0142odzi"}',
        ),
    ]
    client = make_client(
        responder((200, envelope(content="", tool_calls=tool_calls))),
    )
    turn = asyncio.run(client.converse(call()))
    asyncio.run(client.aclose())

    assert [(c.id, c.name, c.arguments) for c in turn.tool_calls] == [
        ("call_1", "search_corpus", "{not json"),
        ("call_2", "search_corpus", '{"query": "najem \\u0142odzi"}'),
    ]


def test_null_content_becomes_an_empty_string() -> None:
    client = make_client(responder((200, envelope(content=None))))
    turn = asyncio.run(client.converse(call()))
    asyncio.run(client.aclose())

    assert turn.content == ""


def test_tool_call_without_an_identifier_is_an_invalid_response() -> None:
    """Without a call_id the turn cannot be answered: a fault, not a call.

    A tool result names the call it answers by call_id, so a call that arrives
    without one can never be replied to.
    """
    tool_calls = [
        {
            "type": "function_call",
            "id": "fc_1",
            "name": "search_corpus",
            "arguments": '{"query": "x"}',
        }
    ]
    client = make_client(responder((200, envelope(content="", tool_calls=tool_calls))))
    with pytest.raises(InvalidModelResponse) as error:
        asyncio.run(client.converse(call()))
    asyncio.run(client.aclose())

    assert error.value.code == "model_response_invalid_envelope"


def test_no_tool_calls_yields_an_empty_turn() -> None:
    client = make_client(responder((200, envelope(content="nothing to search"))))
    turn = asyncio.run(client.converse(call()))
    asyncio.run(client.aclose())

    assert turn.tool_calls == ()
    assert turn.content == "nothing to search"


def test_converse_without_usage_is_a_typed_failure() -> None:
    client = make_client(responder((200, envelope(content="x", usage=None))))
    with pytest.raises(InvalidModelResponse) as error:
        asyncio.run(client.converse(call()))
    asyncio.run(client.aclose())

    assert error.value.code == "model_response_missing_usage"


# ── Retry policy ──


def test_retries_only_retryable_failures_and_counts_every_attempt() -> None:
    requests = 0

    def handler(request: httpx2.Request) -> httpx2.Response:
        nonlocal requests
        requests += 1
        if requests == 1:
            raise httpx2.ConnectError("private response prose", request=request)
        if requests == 2:
            return httpx2.Response(
                429,
                json=envelope(
                    usage={"input_tokens": 3, "output_tokens": 2},
                ),
                request=request,
            )
        if requests == 3:
            return httpx2.Response(
                503,
                json=envelope(
                    usage={"input_tokens": 4, "output_tokens": 1},
                ),
                request=request,
            )
        return httpx2.Response(
            200,
            json=envelope(usage={"input_tokens": 5, "output_tokens": 7}),
            request=request,
        )

    client = make_client(handler, attempts=4)
    result = asyncio.run(client.converse(call()))
    asyncio.run(client.aclose())

    assert requests == 4
    assert [attempt.retry_number for attempt in result.attempts] == [0, 1, 2, 3]
    assert [attempt.status for attempt in result.attempts] == [
        "transport_error",
        "http_error",
        "http_error",
        "success",
    ]
    assert result.input_tokens == 12
    assert result.output_tokens == 10


def test_server_error_retries_to_the_attempt_ceiling() -> None:
    calls = 0

    def handler(request: httpx2.Request) -> httpx2.Response:
        nonlocal calls
        calls += 1
        return httpx2.Response(503, json={"error": {"message": "unavailable"}})

    client = make_client(handler, attempts=3)
    with pytest.raises(ModelCallFailed) as error:
        asyncio.run(client.converse(call()))
    asyncio.run(client.aclose())

    assert calls == 3
    assert error.value.code == "model_http_503"
    assert [a.status for a in error.value.attempts] == ["http_error"] * 3
    assert [a.retry_number for a in error.value.attempts] == [0, 1, 2]


@pytest.mark.parametrize("status", [400, 401, 403, 404, 501])
def test_non_retryable_http_status_stops_after_one_attempt(status: int) -> None:
    requests = 0

    def handler(request: httpx2.Request) -> httpx2.Response:
        nonlocal requests
        requests += 1
        return httpx2.Response(status, text="private model prose", request=request)

    client = make_client(handler)
    with pytest.raises(ModelCallFailed) as error:
        asyncio.run(client.converse(call()))
    asyncio.run(client.aclose())

    assert requests == 1
    assert error.value.code == f"model_http_{status}"
    # The provider's prose and the credential stay out of the raised error, and
    # the library's own exception is not chained onto it.
    assert error.value.__cause__ is None
    assert error.value.__context__ is None
    assert "private model prose" not in str(error.value)
    assert "test-secret-key" not in str(error.value)


def test_transport_error_retries_and_keeps_telemetry_safe() -> None:
    calls = 0

    def handler(request: httpx2.Request) -> httpx2.Response:
        nonlocal calls
        calls += 1
        raise httpx2.ReadError("fixture response body", request=request)

    client = make_client(handler, attempts=2)
    with pytest.raises(ModelCallFailed) as error:
        asyncio.run(client.converse(call()))
    asyncio.run(client.aclose())

    assert calls == 2
    assert error.value.code == "model_transport_error"
    assert [a.status for a in error.value.attempts] == ["transport_error"] * 2
    assert error.value.__cause__ is None
    assert error.value.__context__ is None
    assert "fixture response body" not in str(error.value)


def test_rate_limit_retry_then_success() -> None:
    client = make_client(
        responder(
            (429, {"error": {"message": "slow down"}}),
            (200, envelope()),
        )
    )
    result = asyncio.run(client.converse(call()))
    asyncio.run(client.aclose())

    assert [a.status for a in result.attempts] == ["http_error", "success"]
    assert result.attempts[0].error_code == "model_http_429"


# ── Unusable answers ──


def test_an_unusable_answer_is_not_retried() -> None:
    """Retrying an unusable answer would spend budget hiding the defect."""
    calls = 0
    unanswerable = {
        "type": "function_call",
        "id": "fc_1",
        "name": "search_corpus",
        "arguments": '{"query": "x"}',
    }

    def handler(request: httpx2.Request) -> httpx2.Response:
        nonlocal calls
        calls += 1
        return httpx2.Response(
            200, json=envelope(content="", tool_calls=[unanswerable]), request=request
        )

    client = make_client(handler, attempts=3)
    with pytest.raises(InvalidModelResponse) as error:
        asyncio.run(client.converse(call()))
    asyncio.run(client.aclose())

    assert calls == 1
    assert len(error.value.attempts) == 1
    assert error.value.code == "model_response_invalid_envelope"


def test_an_answer_cut_off_at_the_cap_is_recorded_rather_than_discarded() -> None:
    """A response cut off at the output cap must not escape the attempt loop.

    It arrives with status "incomplete" and whatever the model managed to emit.
    The tokens it spent were spent, so the attempt is recorded and the partial
    answer goes to the role's turn loop, which already handles a turn that
    committed nothing. Dropping it would leave a run poorer by an unrecorded call.
    """
    client = make_client(responder((200, envelope(content="tru", status="incomplete"))))
    turn = asyncio.run(client.converse(call()))
    asyncio.run(client.aclose())

    assert turn.content == "tru"
    assert turn.tool_calls == ()
    attempt = turn.attempts[0]
    assert attempt.status == "success"
    assert attempt.returned_model == RETURNED_MODEL
    assert (attempt.input_tokens, attempt.output_tokens) == (11, 4)


def test_an_answer_with_no_output_at_all_is_an_invalid_envelope() -> None:
    """The SDK accepts an empty output list; this client must type the fault."""
    client = make_client(responder((200, envelope(output=[]))))
    with pytest.raises(InvalidModelResponse) as error:
        asyncio.run(client.converse(call()))
    asyncio.run(client.aclose())

    assert error.value.code == "model_response_invalid_envelope"


# ── Token accounting ──


def test_missing_usage_fails_rather_than_counting_zero() -> None:
    """A 200 that reports no tokens must raise, never spend as if it cost nothing.

    Probed live 2026-09-01: deepseek-v4-flash returns ``usage`` with integer
    counts on a 200. One call cannot prove it always does, so the absence is
    treated as a failure rather than assumed impossible -- a budget that
    silently stops advancing is the worse error.
    """
    client = make_client(responder((200, envelope(usage=None))))
    with pytest.raises(InvalidModelResponse) as error:
        asyncio.run(client.converse(call()))
    asyncio.run(client.aclose())

    assert error.value.code == "model_response_missing_usage"
    assert error.value.attempts[0].status == "invalid_response"


@pytest.mark.parametrize(
    ("input_tokens", "output_tokens"),
    [
        ("11", 4),
        (11, None),
        (11.5, 4),
        (True, 4),
        (11, False),
    ],
    ids=["string", "null", "float", "bool-true", "bool-false"],
)
def test_a_count_that_is_not_a_count_is_missing_usage(
    input_tokens: object, output_tokens: object
) -> None:
    """Every unusable shape is one fact: nothing usable was reported.

    A bool would otherwise be spent as 1 or 0 through Python's int, and a null
    as 0, putting a number the provider never sent into a stored record.
    """
    client = make_client(
        responder(
            (
                200,
                envelope(
                    usage={
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                    }
                ),
            )
        )
    )
    with pytest.raises(InvalidModelResponse) as error:
        asyncio.run(client.converse(call()))
    asyncio.run(client.aclose())

    assert error.value.code == "model_response_missing_usage"
    assert len(error.value.attempts) == 1


def test_one_rule_classifies_usage_on_both_the_success_and_the_error_path() -> None:
    """The success path and an error body must not answer the same bytes twice."""
    usable = {"input_tokens": 90, "output_tokens": 21}
    for unusable in (
        {"input_tokens": None, "output_tokens": 4},
        {"input_tokens": True, "output_tokens": 4},
        {"input_tokens": -1, "output_tokens": 4},
    ):
        assert usage_from_body({"usage": unusable}) is None
        assert token_counts(unusable["input_tokens"], unusable["output_tokens"]) is None
    assert usage_from_body({"usage": usable}) == (90, 21)
    assert token_counts(90, 21) == (90, 21)


def test_negative_input_tokens_produces_missing_usage() -> None:
    client = make_client(
        responder((200, envelope(usage={"input_tokens": -1, "output_tokens": 4})))
    )
    with pytest.raises(InvalidModelResponse) as error:
        asyncio.run(client.converse(call()))
    asyncio.run(client.aclose())

    assert error.value.code == "model_response_missing_usage"
    assert len(error.value.attempts) == 1


def test_negative_output_tokens_produces_missing_usage() -> None:
    client = make_client(
        responder((200, envelope(usage={"input_tokens": 11, "output_tokens": -1})))
    )
    with pytest.raises(InvalidModelResponse) as error:
        asyncio.run(client.converse(call()))
    asyncio.run(client.aclose())

    assert error.value.code == "model_response_missing_usage"
    assert len(error.value.attempts) == 1


def test_tool_turn_with_null_content_is_an_ordinary_turn() -> None:
    """A provider sends content: null beside tool calls; that is the wire shape."""
    client = make_client(
        responder(
            (
                200,
                envelope(
                    content=None,
                    tool_calls=[
                        function_call(call_id="call-1", arguments='{"query": "kaucja"}')
                    ],
                ),
            )
        )
    )
    turn = asyncio.run(client.converse(call()))
    asyncio.run(client.aclose())

    assert turn.content == ""
    assert [call.name for call in turn.tool_calls] == ["search_corpus"]
    assert turn.tool_calls[0].arguments == '{"query": "kaucja"}'


def test_http_error_without_usage_stays_tolerant() -> None:
    """Only the success path tightened; an error response reports what it has."""
    client = make_client(responder((500, envelope(usage=None))), attempts=1)
    with pytest.raises(ModelCallFailed) as error:
        asyncio.run(client.converse(call()))
    asyncio.run(client.aclose())

    assert error.value.code == "model_http_500"
    assert (
        error.value.attempts[0].input_tokens,
        error.value.attempts[0].output_tokens,
    ) == (0, 0)


def test_http_error_reports_the_usage_the_provider_sent() -> None:
    client = make_client(responder((500, envelope())), attempts=1)
    with pytest.raises(ModelCallFailed) as error:
        asyncio.run(client.converse(call()))
    asyncio.run(client.aclose())

    assert (
        error.value.attempts[0].input_tokens,
        error.value.attempts[0].output_tokens,
    ) == (11, 4)


# ── Model identity ──


def test_changed_model_identity_aborts_the_run() -> None:
    run_id = uuid4()
    client = make_client(
        responder(
            (200, envelope()),
            (200, envelope(model="deepseek-v4-flash-20260901")),
        )
    )
    asyncio.run(client.converse(call(run_id=run_id)))
    with pytest.raises(ModelIdentityChanged) as error:
        asyncio.run(client.converse(call(run_id=run_id)))
    asyncio.run(client.aclose())

    assert error.value.code == "model_identity_changed"
    assert error.value.attempts[0].returned_model == "deepseek-v4-flash-20260901"


def test_model_identity_is_pinned_per_run() -> None:
    client = make_client(
        responder(
            (200, envelope()),
            (200, envelope(model="deepseek-v4-flash-20260901")),
        )
    )
    asyncio.run(client.converse(call(run_id=uuid4())))
    other = asyncio.run(client.converse(call(run_id=uuid4())))
    asyncio.run(client.aclose())

    assert other.returned_model == "deepseek-v4-flash-20260901"


def test_converse_and_complete_share_per_run_model_pinning() -> None:
    returned = iter(("model-build-a", "model-build-b"))

    def handler(request: httpx2.Request) -> httpx2.Response:
        payload = json.loads(request.content)
        model = next(returned)
        if "tools" in payload:
            return httpx2.Response(
                200,
                json=envelope(model=model, content="tool phase done"),
                request=request,
            )
        return httpx2.Response(200, json=envelope(model=model), request=request)

    run_id = uuid4()
    client = make_client(handler)
    first = asyncio.run(client.converse(call(run_id=run_id)))
    with pytest.raises(ModelIdentityChanged) as error:
        asyncio.run(client.converse(call(run_id=run_id)))
    asyncio.run(client.aclose())

    assert first.returned_model == "model-build-a"
    assert error.value.code == "model_identity_changed"
    assert error.value.attempts[-1].status == "model_identity_changed"


# ── Budgets and configuration ──


def test_deadline_cuts_the_call_and_is_not_retried() -> None:
    """The run's remaining wall budget bounds one model call.

    A call that outlives it ends the run: retrying inside a budget that has
    already run out would spend time the run does not have, so the attempt is
    recorded once and raised as budget exhaustion rather than as a transport
    fault the caller might retry.
    """
    calls = 0

    async def handler(request: httpx2.Request) -> httpx2.Response:
        nonlocal calls
        calls += 1
        await asyncio.sleep(1)
        return httpx2.Response(200, json=envelope(), request=request)

    client = make_client(handler, attempts=3)
    request = replace(call(parameters={}), timeout_seconds=0.05)
    with pytest.raises(ModelDeadlineExceeded) as error:
        asyncio.run(client.converse(request))
    asyncio.run(client.aclose())

    assert calls == 1
    assert error.value.code == "budget_exhausted"
    assert len(error.value.attempts) == 1
    assert error.value.attempts[0].status == "transport_error"


def test_a_positive_timeout_is_required() -> None:
    client = make_client(responder((200, envelope())))
    request = replace(call(parameters={}), timeout_seconds=0.0)
    with pytest.raises(ValueError, match="timeout must be positive"):
        asyncio.run(client.converse(request))
    asyncio.run(client.aclose())


def test_constructed_client_uses_configured_timeout_not_library_default() -> None:
    client = OpenAICompatibleClient(settings(), model_timeout_seconds=120)
    timeout = client._client.timeout
    asyncio.run(client.aclose())

    assert timeout.read == 120
    assert timeout.connect == 10
    assert timeout.write == 30
    assert timeout.pool == 10


def test_default_model_timeout_matches_run_config() -> None:
    client = OpenAICompatibleClient(settings())
    timeout = client._client.timeout
    asyncio.run(client.aclose())

    assert timeout.read == RunConfig().model_timeout_seconds


def test_library_retries_are_disabled() -> None:
    """A retry the library performs is a call this project cannot record."""
    client = OpenAICompatibleClient(settings())
    retries = client._client.max_retries
    asyncio.run(client.aclose())

    assert retries == 0


def test_slow_response_past_old_threshold_does_not_raise_transport_error() -> None:
    async def handler(request: httpx2.Request) -> httpx2.Response:
        await asyncio.sleep(6)
        return httpx2.Response(200, json=envelope(), request=request)

    async def scenario() -> None:
        client = OpenAICompatibleClient(
            settings(),
            model_timeout_seconds=60,
            http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
        )
        try:
            result = await client.converse(call())
        finally:
            await client.aclose()
        assert result.content == "ok"
        assert result.attempts[0].status == "success"

    asyncio.run(scenario())


def test_live_settings_are_required() -> None:
    with pytest.raises(ValueError, match="MODEL_API_KEY"):
        OpenAICompatibleClient(Settings(model_api_key=None, model_name="m"))
    with pytest.raises(ValueError, match="MODEL_NAME"):
        OpenAICompatibleClient(Settings(model_api_key="k", model_name=""))
    with pytest.raises(ValueError, match="MODEL_BASE_URL"):
        OpenAICompatibleClient(Settings(model_api_key="k", model_base_url=""))


def test_configured_provider_headers_reach_the_request() -> None:
    """A gateway can refuse an otherwise valid request for a missing header.

    The default endpoint answers 400 without its routing header, so a run fails
    on its first model call. Configuration carries the header and the transport
    has to put it on the wire, not merely hold it.
    """
    seen: list[httpx2.Headers] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request.headers)
        return httpx2.Response(200, json=envelope(), request=request)

    client = make_client(
        handler,
        model_settings=Settings(
            model_api_key="test-secret-key",
            model_base_url=BASE_URL,
            model_name=MODEL,
            model_extra_headers={"x-routing-hint": "batch-7"},
        ),
    )
    asyncio.run(client.converse(call()))
    asyncio.run(client.aclose())

    assert seen[0]["x-routing-hint"] == "batch-7"


SDK_REQUEST_HEADERS = [
    "accept",
    "accept-encoding",
    "authorization",
    "connection",
    "content-length",
    "content-type",
    "host",
    "user-agent",
    "x-stainless-arch",
    "x-stainless-async",
    "x-stainless-lang",
    "x-stainless-os",
    "x-stainless-package-version",
    "x-stainless-read-timeout",
    "x-stainless-retry-count",
    "x-stainless-runtime",
    "x-stainless-runtime-version",
]
"""Every header the pinned client sends with no configuration of our own.

A library upgrade that changes this list should fail here and be looked at,
because it changes what every measured call carries.
"""


def test_no_configured_headers_leaves_the_request_alone() -> None:
    """Absent configuration must add nothing and take nothing away.

    Naming one header the tests themselves invented asserted nothing: the
    request carries it under no configuration either way.
    """
    seen: list[httpx2.Headers] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request.headers)
        return httpx2.Response(200, json=envelope(), request=request)

    client = make_client(handler)
    asyncio.run(client.converse(call()))
    asyncio.run(client.aclose())

    assert seen[0]["authorization"] == "Bearer test-secret-key"
    assert seen[0]["content-type"] == "application/json"
    # The whole set, not a prefix: filtering for "x-" passed a request carrying
    # an injected OpenAI-Organization, User-Agent or Accept.
    assert sorted(seen[0].keys()) == SDK_REQUEST_HEADERS


def test_a_provider_refusal_records_its_reason_in_the_log(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An attempt keeps only a status code, so the reason has nowhere else to go.

    Two runs died on a single HTTP 400 whose cause could not be established
    afterwards, because nothing kept what the provider said. The text can quote
    the request, so it goes to the log and never to the run record.
    """
    body = {"error": {"type": "MissingSessionID", "message": "cannot be routed"}}

    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(400, json=body, request=request)

    client = make_client(handler, attempts=1)
    with caplog.at_level(logging.WARNING):
        with pytest.raises(ModelCallFailed):
            asyncio.run(client.converse(call()))
    asyncio.run(client.aclose())

    logged = [
        r.getMessage() for r in caplog.records if "provider refused" in r.getMessage()
    ]
    assert logged, "the refusal was not logged"
    assert "MissingSessionID" in logged[0]
    assert "status=400" in logged[0]
