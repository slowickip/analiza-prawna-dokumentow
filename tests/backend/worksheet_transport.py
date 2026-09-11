"""A scripted provider that plays the worksheet loop, task by task.

Every agent turn is a tool conversation, so a pipeline test cannot script one
answer per node any more: it has to answer as a role that calls tools and then
commits. This module holds that script once. It answers on the task name the
request carries (``prompt_version``), reads what it needs out of the payload and
out of the tool results already in the transcript, and leaves the explanation to
the caller.

The factories at the bottom build the variants tests need: which locators the
analyst puts forward, how the verifier decides each one, what kind of norm each
provision is.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

DEFAULT_MODEL = "deepseek-v4-flash-20260828"

TaskHandler = Callable[["Turn"], Mapping[str, Any]]
Decide = Callable[[str], Mapping[str, Any]]


class Turn:
    """One provider request, parsed into what a scripted role needs from it.

    The wire carries Responses input items; a scripted role is written against
    the transcript the pipeline itself speaks. ``messages`` is that transcript,
    reconstructed, so a script says what a role saw rather than how the route
    spelled it.
    """

    def __init__(self, payload: Mapping[str, Any]) -> None:
        self.request = payload
        self.messages = transcript(payload.get("input", []))
        self.tools = [
            cast(dict[str, Any], tool)["name"] for tool in payload.get("tools", [])
        ]
        envelope: dict[str, Any] = {}
        for message in self.messages:
            if message.get("role") == "user" and isinstance(
                message.get("content"), str
            ):
                try:
                    candidate = json.loads(cast(str, message["content"]))
                except json.JSONDecodeError:
                    continue
                if isinstance(candidate, dict) and "prompt_version" in candidate:
                    envelope = candidate
                    break
        self.task = str(envelope.get("prompt_version", ""))
        self.payload = cast(dict[str, Any], envelope.get("payload", {}))

    @property
    def locator(self) -> str:
        """The candidate under review, as the role's payload names it."""
        return str(self.payload.get("candidate_locator", ""))

    @property
    def tool_results(self) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for message in self.messages:
            if message.get("role") != "tool":
                continue
            try:
                results.append(json.loads(cast(str, message["content"])))
            except (json.JSONDecodeError, TypeError):
                continue
        return results

    def calls_made(self, name: str) -> int:
        """How many times this transcript already called one tool."""
        total = 0
        for message in self.messages:
            if message.get("role") != "assistant":
                continue
            for call in message.get("tool_calls") or []:
                if cast(dict[str, Any], call)["function"]["name"] == name:
                    total += 1
        return total

    def entries(self, kind: str) -> list[dict[str, Any]]:
        worksheet = self.payload.get("worksheet")
        if not isinstance(worksheet, list):
            return []
        return [
            cast(dict[str, Any], entry)
            for entry in worksheet
            if isinstance(entry, dict) and entry.get("kind") == kind
        ]

    def first_result_locator(self) -> str | None:
        for result in self.tool_results:
            results = result.get("results")
            if isinstance(results, list) and results:
                return str(cast(Mapping[str, Any], results[0])["locator"])
        return None


def tool_call(name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
    return tool_calls([(name, arguments)])


def tool_calls(calls: Sequence[tuple[str, Mapping[str, Any]]]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": f"call_{index}_{name}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }
            for index, (name, arguments) in enumerate(calls)
        ],
    }


def silence() -> dict[str, Any]:
    return {"role": "assistant", "content": ""}


def quote_of(turn: Turn, length: int = 40) -> str:
    unit_text = str(turn.payload.get("unit_text", ""))
    return unit_text[: min(len(unit_text), length)].strip() or unit_text


# ── the default script ──


def search_then_post(turn: Turn) -> Mapping[str, Any]:
    """Search once, put the first result forward, then fall silent."""
    if not turn.calls_made("search_corpus"):
        return tool_call("search_corpus", {"phrase": quote_of(turn, 60) or "umowa"})
    if not turn.calls_made("post_candidates"):
        found = turn.first_result_locator()
        if found is None:
            return silence()
        return tool_call(
            "post_candidates",
            {
                "candidates": [
                    {
                        "locator": found,
                        "why": "Przepis dotyczy tej instytucji prawnej.",
                    }
                ]
            },
        )
    return silence()


def relevant_verdict(turn: Turn) -> Mapping[str, Any]:
    return tool_call(
        "post_verdict",
        {
            "relevant": True,
            "quote": quote_of(turn),
            "raw_confidence": 0.9,
            "based_on": [],
        },
    )


def imperative_character(turn: Turn) -> Mapping[str, Any]:
    return tool_call("post_character", character_arguments(turn, "imperative"))


def no_departure(turn: Turn) -> Mapping[str, Any]:
    return tool_call(
        "post_verdict",
        {"departure": "none", "based_on": []},
    )


def synthesis_empty(turn: Turn) -> Mapping[str, Any]:
    """List the findings once, then commit an empty grouping over them."""
    return list_then_group(turn, ())


def list_then_group(
    turn: Turn, groups: Sequence[Mapping[str, Any]]
) -> Mapping[str, Any]:
    """List the existing findings, then commit exactly these groups."""
    if not turn.calls_made("list_findings"):
        return tool_call("list_findings", {})
    if not turn.calls_made("group_findings"):
        return tool_call("group_findings", {"groups": list(groups)})
    return silence()


def listed_refs(turn: Turn) -> list[str]:
    """The references the listing returned, in the order the role saw them."""
    return [
        str(item["ref"])
        for result in turn.tool_results
        for item in result.get("findings", [])
        if isinstance(item, dict) and "ref" in item
    ]


def synthesis_group_all(turn: Turn) -> Mapping[str, Any]:
    """List the findings, then commit one group over every listed reference."""
    if not turn.calls_made("list_findings"):
        return tool_call("list_findings", {})
    if not turn.calls_made("group_findings"):
        return tool_call(
            "group_findings",
            {
                "groups": [
                    {
                        "title": "Grupa",
                        "summary": "System zgrupował ustalenia tej analizy.",
                        "finding_ids": listed_refs(turn),
                    }
                ]
            },
        )
    return silence()


DEFAULT_SCRIPT: Mapping[str, TaskHandler] = {
    "researcher.search": search_then_post,
    "analyst.characterise": imperative_character,
    "verifier.relevance": relevant_verdict,
    "verifier.relation": no_departure,
    "synthesise": synthesis_empty,
}


def play(
    turn: Turn, script: Mapping[str, TaskHandler] | None = None
) -> Mapping[str, Any]:
    """The assistant message this task answers with, under the default script."""
    handlers = dict(DEFAULT_SCRIPT)
    if script is not None:
        handlers.update(script)
    handler = handlers.get(turn.task)
    if handler is None:
        raise AssertionError(f"no scripted answer for task {turn.task!r}")
    return handler(turn)


def envelope(
    message: Mapping[str, Any],
    *,
    model: str = DEFAULT_MODEL,
    input_tokens: int = 1,
    output_tokens: int = 1,
) -> dict[str, Any]:
    """One scripted assistant turn as the Responses route returns it.

    A real answer carries a reasoning item the pipeline does not act on, so every
    scripted answer carries one too: a script that only ever saw text and calls
    would not notice the client silently starting to record it.
    """
    output: list[dict[str, Any]] = [{"type": "reasoning", "summary": []}]
    content = message.get("content")
    if content:
        output.append(
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": content}],
            }
        )
    for call in message.get("tool_calls") or []:
        output.append(
            {
                "type": "function_call",
                "call_id": call["id"],
                "name": call["function"]["name"],
                "arguments": call["function"]["arguments"],
            }
        )
    return {
        "model": model,
        "status": "completed",
        "output": output,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        },
    }


def transcript(items: Any) -> list[dict[str, Any]]:
    """Responses input items back into the pipeline's own message shape."""
    messages: list[dict[str, Any]] = []
    for item in cast(list[dict[str, Any]], items):
        kind = item.get("type")
        if kind == "function_call_output":
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": item["call_id"],
                    "content": item["output"],
                }
            )
            continue
        if kind == "function_call":
            call = {
                "id": item["call_id"],
                "type": "function",
                "function": {"name": item["name"], "arguments": item["arguments"]},
            }
            if messages and messages[-1].get("role") == "assistant":
                messages[-1]["tool_calls"].append(call)
            else:
                messages.append(
                    {"role": "assistant", "content": "", "tool_calls": [call]}
                )
            continue
        messages.append({"role": item["role"], "content": item["content"]})
    return messages


# ── variants ──


def character_arguments(turn: Turn, kind: str) -> dict[str, Any]:
    arguments: dict[str, Any] = {
        "kind": kind,
        "evidence": [
            {
                "source_kind": "official_normative_text",
                "locator": turn.payload.get("candidate_locator"),
                "pinpoint": "ust. 1",
                "interpretive_methods": ["linguistic"],
                "rationale": "brzmienie przepisu",
            }
        ],
    }
    if kind == "undetermined":
        arguments["evidence"] = []
        arguments["undetermined_reason"] = "brak podstawy do rozstrzygnięcia"
    if kind == "semi_imperative":
        arguments["semi_imperative_direction"] = {
            "relation": "more_favourable_to",
            "protected_party_role": "najemca",
        }
    return arguments


def posting(*locators: str, phrase: str = "kaucja najem") -> TaskHandler:
    """A researcher that searches once and puts exactly these locators forward."""

    def handler(turn: Turn) -> Mapping[str, Any]:
        if not turn.calls_made("search_corpus"):
            return tool_call("search_corpus", {"phrase": phrase})
        if not turn.calls_made("post_candidates"):
            return tool_call(
                "post_candidates",
                {
                    "candidates": [
                        {"locator": locator, "why": "Przepis dotyczy najmu."}
                        for locator in locators
                    ]
                },
            )
        return silence()

    return handler


def posting_nothing(*, phrase: str = "kaucja najem") -> TaskHandler:
    """A researcher that searches and then falls silent without concluding."""

    def handler(turn: Turn) -> Mapping[str, Any]:
        if not turn.calls_made("search_corpus"):
            return tool_call("search_corpus", {"phrase": phrase})
        return silence()

    return handler


def posting_no_basis(*, phrase: str = "kaucja najem") -> TaskHandler:
    """A researcher that searches and then states that the corpus holds nothing."""

    def handler(turn: Turn) -> Mapping[str, Any]:
        if not turn.calls_made("search_corpus"):
            return tool_call("search_corpus", {"phrase": phrase})
        if not turn.calls_made("post_candidates"):
            return tool_call("post_candidates", {"candidates": []})
        return silence()

    return handler


def posting_no_basis_without_looking(*, phrase: str = "kaucja najem") -> TaskHandler:
    """A researcher that concludes before looking, is refused, then looks.

    The refusal is the point: the empty commit is only a conclusion once the
    corpus has been consulted, so the corrected path searches and commits again.
    """

    def handler(turn: Turn) -> Mapping[str, Any]:
        if not turn.calls_made("post_candidates"):
            return tool_call("post_candidates", {"candidates": []})
        if not turn.calls_made("search_corpus"):
            return tool_call("search_corpus", {"phrase": phrase})
        return tool_call("post_candidates", {"candidates": []})

    return handler


def repeating_one_phrase(*, phrase: str = "kaucja najem") -> TaskHandler:
    """A researcher that asks the same phrase twice, then posts nothing.

    The second ask is what the run's retrieval cache is for, so it is what the
    cache-hit counter has to see.
    """

    def handler(turn: Turn) -> Mapping[str, Any]:
        asked = turn.calls_made("search_corpus")
        if asked < 2:
            return tool_call("search_corpus", {"phrase": phrase})
        if not turn.calls_made("post_candidates"):
            return tool_call("post_candidates", {"candidates": []})
        return silence()

    return handler


def searching_with_a_refused_phrase() -> TaskHandler:
    """A researcher whose phrase never reaches the corpus, then concludes anyway.

    A phrase of punctuation sanitises to nothing, so the tool refuses it before any
    corpus call. If that still counted as a search, the empty commit below would be
    a conclusion about a corpus the run never consulted.
    """

    def handler(turn: Turn) -> Mapping[str, Any]:
        if not turn.calls_made("search_corpus"):
            return tool_call("search_corpus", {"phrase": "!!!"})
        return tool_call("post_candidates", {"candidates": []})

    return handler


def characterising_from_an_unseen_locator(locator: str) -> TaskHandler:
    """An analyst that opens a provision the researcher never retrieved."""

    def handler(turn: Turn) -> Mapping[str, Any]:
        if not turn.calls_made("read_provision"):
            return tool_call("read_provision", {"locator": locator})
        return tool_call("post_character", character_arguments(turn, "imperative"))

    return handler


def deciding_relevance(decide: Decide) -> TaskHandler:
    """A verifier whose relevance verdict depends on the candidate's locator."""

    def handler(turn: Turn) -> Mapping[str, Any]:
        arguments = {
            "quote": quote_of(turn),
            "based_on": [],
            **decide(turn.locator),
        }
        return tool_call("post_verdict", arguments)

    return handler


def relevance_quoting(quote: str, *, relevant: bool | None = True) -> TaskHandler:
    """A verifier that returns one fixed quote, whatever the unit says."""

    def handler(turn: Turn) -> Mapping[str, Any]:
        return tool_call(
            "post_verdict",
            {
                "relevant": relevant,
                "quote": quote,
                "raw_confidence": 0.9,
                "based_on": [],
            },
        )

    return handler


def deciding_relation(decide: Decide) -> TaskHandler:
    def handler(turn: Turn) -> Mapping[str, Any]:
        arguments = {
            "based_on": [],
            **decide(turn.locator),
        }
        return tool_call("post_verdict", arguments)

    return handler


def characterising(kind_for: Callable[[str], str]) -> TaskHandler:
    def handler(turn: Turn) -> Mapping[str, Any]:
        return tool_call(
            "post_character", character_arguments(turn, kind_for(turn.locator))
        )

    return handler
