"""Model response parsing and deterministic hashing."""

from __future__ import annotations

import hashlib
import json
from typing import cast

from .types import ToolInvocation


def hash_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def hash_json(value: object) -> str:
    return hash_text(json.dumps(value, sort_keys=True, separators=(",", ":")))


def token_counts(prompt: object, completion: object) -> tuple[int, int] | None:
    """One pair of token counts, or ``None`` when the provider reported none.

    ``None`` and ``(0, 0)`` are different facts: the first means nothing usable
    was reported, the second means the provider reported no tokens. A caller must
    not spend the first as if it were the second -- that is how a budget silently
    stops advancing, and how a fabricated count reaches a stored record. Every
    shape that is not a non-negative count is the first fact: a null, a string, a
    float, a negative, and a bool, which Python would otherwise count as 1 or 0.
    A 200 that omits the counts is not assumed impossible.
    """
    if type(prompt) is not int or type(completion) is not int:
        return None
    if prompt < 0 or completion < 0:
        return None
    return prompt, completion


def usage_from_body(value: object) -> tuple[int, int] | None:
    """Token counts from a response body, or ``None`` when none were reported."""
    if not isinstance(value, dict):
        return None
    usage = cast(dict[str, object], value).get("usage")
    if not isinstance(usage, dict):
        return None
    return token_counts(
        cast(dict[str, object], usage).get("input_tokens"),
        cast(dict[str, object], usage).get("output_tokens"),
    )


def tool_turn_hash(content: str, tool_calls: tuple[ToolInvocation, ...]) -> str:
    return hash_json(
        {
            "content": content,
            "tool_calls": [
                {
                    "id": tool_call.id,
                    "name": tool_call.name,
                    "arguments": tool_call.arguments,
                }
                for tool_call in tool_calls
            ],
        }
    )
