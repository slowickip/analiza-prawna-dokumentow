"""Model request payloads for the provider's Responses route."""

from collections.abc import Mapping, Sequence
from typing import cast

from .types import RequestContext, ToolConversation


def validate_request(request: RequestContext, *, reserved: set[str]) -> None:
    if request.timeout_seconds is not None and request.timeout_seconds <= 0:
        raise ValueError("model call timeout must be positive")
    forbidden = reserved.intersection(request.parameters)
    if forbidden:
        raise ValueError(f"reserved model parameters: {sorted(forbidden)}")


# A caller owns the request's own vocabulary below; it may not reach past it and
# set a wire field this module derives, whether under this project's name for it
# or the transport's.
CONVERSE_RESERVED = {
    "input",
    "instructions",
    "max_output_tokens",
    "model",
    "reasoning",
    "store",
    "temperature",
    "tools",
}

# This project names an output ceiling and a reasoning level once, in the run
# request that records them; the transport spells both differently. The
# translation lives here so a recorded parameter keeps its meaning across a
# change of route, and a stored run stays comparable with a later one.
OUTPUT_CAP_PARAMETER = "max_completion_tokens"
REASONING_EFFORT_PARAMETER = "reasoning_effort"


def tool_payload(conversation: ToolConversation, model_name: str) -> dict[str, object]:
    """The Responses request for one tool conversation.

    The prompt hash is taken over this dictionary rather than the serialised
    body, so it identifies the request the pipeline made, not the way the client
    library happened to encode it.
    """
    payload: dict[str, object] = {
        "model": model_name,
        "input": input_items(conversation.messages),
        "temperature": conversation.temperature,
        "tools": [
            {
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": dict(tool.parameters),
            }
            for tool in conversation.tools
        ],
        # Each request includes the full conversation; provider storage is unnecessary.
        "store": False,
    }
    payload.update(_wire_parameters(conversation.parameters))
    return payload


def input_items(
    messages: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """The pipeline's transcript as Responses input items.

    The pipeline speaks one message shape, the one its own graph produces: a
    role and its text, an assistant turn carrying tool calls, and a tool result
    naming the call it answers. This route spells the last two as items of their
    own rather than as messages, and matches a result to its call by ``call_id``.
    Converting here is what keeps that spelling out of every caller above.
    """
    items: list[dict[str, object]] = []
    for message in messages:
        role = message.get("role")
        content = cast(str, message.get("content") or "")
        if role == "tool":
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": message["tool_call_id"],
                    "output": content,
                }
            )
            continue
        if role == "assistant":
            if content:
                items.append({"role": "assistant", "content": content})
            for call in cast(
                Sequence[Mapping[str, object]], message.get("tool_calls") or ()
            ):
                function = cast(Mapping[str, object], call["function"])
                items.append(
                    {
                        "type": "function_call",
                        "call_id": call["id"],
                        "name": function["name"],
                        "arguments": function["arguments"],
                    }
                )
            continue
        items.append({"role": role, "content": content})
    return items


def _wire_parameters(
    parameters: Mapping[str, object],
) -> dict[str, object]:
    wire: dict[str, object] = {}
    for key, value in parameters.items():
        if key == OUTPUT_CAP_PARAMETER:
            wire["max_output_tokens"] = value
        elif key == REASONING_EFFORT_PARAMETER:
            wire["reasoning"] = {"effort": value}
        else:
            wire[key] = value
    return wire
