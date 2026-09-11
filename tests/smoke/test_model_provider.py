"""Opt-in live check that the connector works against a real endpoint.

Skipped unless ``LIVE_PROVIDER_SMOKE=1``, because it spends provider tokens:

    LIVE_PROVIDER_SMOKE=1 uv run pytest tests/smoke/test_model_provider.py

It uses the configured endpoint, credential and model, which default to the
OpenRouter deployment; ``MODEL_BASE_URL`` and ``MODEL_NAME`` point the same
check at any other endpoint serving the same route.

This is a mechanism check, not evaluation data: it proves the transport, the
tool round trip and the token record work end to end; it measures nothing.
"""

from __future__ import annotations

import asyncio
import json
import os
from uuid import uuid4

import pytest

from contract_analyzer.config import Settings
from contract_analyzer.model import (
    ToolConversation,
    ToolDefinition,
    ToolTurn,
)
from contract_analyzer.openai_compatible import OpenAICompatibleClient

pytestmark = pytest.mark.skipif(
    os.environ.get("LIVE_PROVIDER_SMOKE") != "1",
    reason="live provider smoke is opt-in and spends tokens: set LIVE_PROVIDER_SMOKE=1",
)

TOOL_NAME = "search_corpus"
COMMIT_TOOL = "post_candidates"
# Measured 2026-09-03 against deepseek-v4-flash on this prompt: 273-837 output
# tokens over four calls, the variation being reasoning the model does not
# return. The cap sits well above that, so a truncated answer here means the
# provider changed, not that the check was drawn too tight.
MAX_OUTPUT_TOKENS = 4000

TOOLS = (
    ToolDefinition(
        name=TOOL_NAME,
        description="Przeszukaj korpus aktów prawnych",
        parameters={
            "type": "object",
            "properties": {"phrase": {"type": "string"}},
            "required": ["phrase"],
            "additionalProperties": False,
        },
    ),
    ToolDefinition(
        name=COMMIT_TOOL,
        description="Zapisz znalezione podstawy i zakończ zadanie",
        parameters={
            "type": "object",
            "properties": {"locators": {"type": "array", "items": {"type": "string"}}},
            "required": ["locators"],
            "additionalProperties": False,
        },
    ),
)


def live_client() -> OpenAICompatibleClient:
    settings = Settings.from_env()
    if not settings.model_api_key:
        pytest.skip("no provider credential: set MODEL_API_KEY")
    return OpenAICompatibleClient(settings)


def conversation(messages: tuple[dict[str, object], ...]) -> ToolConversation:
    return ToolConversation(
        run_id=uuid4(),
        messages=messages,
        tools=TOOLS,
        prompt_version="live-smoke",
        temperature=0.0,
        parameters={"max_completion_tokens": MAX_OUTPUT_TOKENS},
        timeout_seconds=120.0,
    )


async def _round_trip() -> tuple[ToolTurn, ToolTurn]:
    """Two turns on one client, with the tool result fed back between them."""
    client = live_client()
    try:
        first = await client.converse(
            conversation(
                (
                    {
                        "role": "system",
                        "content": (
                            f"Użyj narzędzia {TOOL_NAME} dokładnie raz, "
                            f"a po jego wyniku zakończ przez {COMMIT_TOOL}. "
                            "W jednej turze wywołaj tylko jedno narzędzie."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            "Znajdź podstawę prawną terminu zapłaty czynszu najmu."
                        ),
                    },
                )
            )
        )
        searches = [c for c in first.tool_calls if c.name == TOOL_NAME]
        if not searches:
            pytest.fail(f"the model did not call {TOOL_NAME}: {first.content!r}")
        search = searches[0]
        second = await client.converse(
            conversation(
                (
                    {
                        "role": "system",
                        "content": (
                            f"Użyj narzędzia {TOOL_NAME} dokładnie raz, "
                            f"a po jego wyniku zakończ przez {COMMIT_TOOL}. "
                            "W jednej turze wywołaj tylko jedno narzędzie."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            "Znajdź podstawę prawną terminu zapłaty czynszu najmu."
                        ),
                    },
                    {
                        "role": "assistant",
                        "content": first.content,
                        "tool_calls": [
                            {
                                "id": search.id,
                                "type": "function",
                                "function": {
                                    "name": search.name,
                                    "arguments": search.arguments,
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": search.id,
                        "content": json.dumps(
                            {
                                "results": [
                                    {
                                        "locator": (
                                            "https://api.sejm.gov.pl/eli/acts/DU/"
                                            "1964/93/text.html/arti=669"
                                        ),
                                        "snippet": (
                                            "Najemca obowiązany jest uiszczać "
                                            "czynsz w terminie umówionym."
                                        ),
                                    }
                                ]
                            },
                            ensure_ascii=False,
                        ),
                    },
                )
            )
        )
        return first, second
    finally:
        await client.aclose()


def test_a_tool_round_trip_reaches_the_live_endpoint() -> None:
    """A call goes out, its result goes back, and the model acts on it.

    A turn that merely survives the wire proves less than it looks: the route
    this connector uses matches a result to its call by identifier, and getting
    that wrong is silent -- the model simply answers as though the tool had never
    run. So the check is that the second turn, given the result, does something
    with it rather than repeating the search.
    """
    first, second = asyncio.run(_round_trip())

    for turn in (first, second):
        assert turn.returned_model
        assert turn.input_tokens > 0
        assert turn.output_tokens > 0
        assert turn.attempts[-1].status == "success"
        assert turn.attempts[-1].response_hash is not None

    search = next(c for c in first.tool_calls if c.name == TOOL_NAME)
    assert search.id
    assert isinstance(json.loads(search.arguments), dict)

    # Having been handed the result, the model must move on rather than search
    # again: repeating the search is what a lost tool result looks like.
    assert [c.name for c in second.tool_calls] != [TOOL_NAME], (
        f"the tool result did not reach the model: {second.tool_calls}"
    )
    for invocation in second.tool_calls:
        assert invocation.id
        assert isinstance(json.loads(invocation.arguments), dict)
