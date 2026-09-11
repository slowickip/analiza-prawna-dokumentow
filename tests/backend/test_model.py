"""The transport-neutral half of a model call: the request this route sends.

The pipeline speaks one transcript shape and the provider's Responses route
another, so these pin the translation between them. Every expectation here was
measured against the live provider on 2026-09-08 rather than read off a
specification: a request shaped any other way was refused or answered wrongly.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from contract_analyzer.model import (
    CONVERSE_RESERVED,
    ToolConversation,
    ToolDefinition,
    tool_payload,
    validate_request,
)

TOOL = ToolDefinition(
    name="search_corpus",
    description="Search the legal corpus",
    parameters={"type": "object", "properties": {}},
)


def _conversation(
    messages: tuple[dict[str, object], ...],
    parameters: dict[str, object] | None = None,
) -> ToolConversation:
    return ToolConversation(
        run_id=uuid4(),
        messages=messages,
        tools=(TOOL,),
        prompt_version="find_basis",
        temperature=0.0,
        parameters=parameters or {},
    )


def test_a_tool_is_named_flat_rather_than_nested_under_a_function_key() -> None:
    """The chat route nests a tool under "function"; this route does not.

    Sending the nested shape here is not a stylistic difference: the provider
    rejects it, because it looks for the name one level up.
    """
    payload = tool_payload(_conversation(({"role": "user", "content": "find"},)), "m")

    assert payload["tools"] == [
        {
            "type": "function",
            "name": "search_corpus",
            "description": "Search the legal corpus",
            "parameters": {"type": "object", "properties": {}},
        }
    ]


def test_a_conversation_becomes_input_items_and_carries_no_server_state() -> None:
    payload = tool_payload(_conversation(({"role": "user", "content": "find"},)), "m")

    assert payload["model"] == "m"
    assert payload["temperature"] == 0.0
    assert payload["input"] == [{"role": "user", "content": "find"}]
    # Each call carries its whole transcript, so nothing is left with the
    # provider to be recalled by a later one.
    assert payload["store"] is False


def test_a_tool_result_is_matched_to_its_call_by_call_id() -> None:
    """The link between a call and its answer is call_id, not message order.

    An assistant turn's calls become items of their own, and the result names
    the call it answers. Getting this wrong does not fail loudly: the provider
    answers as though the tool had never run.
    """
    payload = tool_payload(
        _conversation(
            (
                {"role": "system", "content": "rules"},
                {"role": "user", "content": "find"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_7",
                            "type": "function",
                            "function": {
                                "name": "search_corpus",
                                "arguments": '{"phrase":"czynsz"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_7",
                    "content": '{"results":[]}',
                },
            )
        ),
        "m",
    )

    assert payload["input"] == [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "find"},
        {
            "type": "function_call",
            "call_id": "call_7",
            "name": "search_corpus",
            "arguments": '{"phrase":"czynsz"}',
        },
        {
            "type": "function_call_output",
            "call_id": "call_7",
            "output": '{"results":[]}',
        },
    ]


def test_an_assistant_turn_with_text_keeps_it_beside_its_calls() -> None:
    payload = tool_payload(
        _conversation(
            (
                {
                    "role": "assistant",
                    "content": "Sprawdzę korpus.",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "search_corpus", "arguments": "{}"},
                        }
                    ],
                },
            )
        ),
        "m",
    )

    assert payload["input"] == [
        {"role": "assistant", "content": "Sprawdzę korpus."},
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "search_corpus",
            "arguments": "{}",
        },
    ]


def test_this_projects_parameter_names_are_translated_at_the_wire() -> None:
    """A recorded parameter keeps this project's name; the wire gets the route's.

    The run record, the parity bundle and the protocol all say
    max_completion_tokens and reasoning_effort. Renaming those to suit a route
    would make a stored run incomparable with a later one for no reason, so the
    translation happens here and nowhere else.
    """
    payload = tool_payload(
        _conversation(
            ({"role": "user", "content": "find"},),
            {"max_completion_tokens": 4000, "reasoning_effort": "low", "seed": 7},
        ),
        "m",
    )

    assert payload["max_output_tokens"] == 4000
    assert payload["reasoning"] == {"effort": "low"}
    assert payload["seed"] == 7
    assert "max_completion_tokens" not in payload
    assert "reasoning_effort" not in payload


def test_a_caller_cannot_set_a_wire_field_this_module_derives() -> None:
    """Reserved under either name: the project's or the transport's."""
    for reserved in ("input", "tools", "max_output_tokens", "reasoning", "store"):
        with pytest.raises(ValueError, match="reserved model parameters"):
            validate_request(
                _conversation(({"role": "user", "content": "x"},), {reserved: 1}),
                reserved=CONVERSE_RESERVED,
            )
