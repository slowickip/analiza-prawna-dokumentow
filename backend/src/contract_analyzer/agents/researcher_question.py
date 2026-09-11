"""One bounded researcher task answering the explainer's question from the corpus."""

from __future__ import annotations

import json
from collections.abc import Mapping

from contract_analyzer.agents.interactive_tools import (
    ANSWER_QUESTION,
    QUESTION_TOOLS,
    RESEARCHER_QUESTION_MAX_TURNS,
    AnswerQuestionArgs,
)
from contract_analyzer.agents.retrieval import read_provision, run_search
from contract_analyzer.agents.state import ActiveRun
from contract_analyzer.agents.tool_args import ReadProvisionArgs, SearchCorpusArgs
from contract_analyzer.agents.tools import parse_tool
from contract_analyzer.agents.turns import (
    ToolResult,
    refusal,
    refused,
    run_conversation,
)
from contract_analyzer.model import AttemptTelemetry, ToolInvocation


async def answer_question(active: ActiveRun, text: str) -> dict[str, object]:
    """Run exactly one researcher.question task and return its committed answer."""
    answer: dict[str, object] = {}
    messages: list[Mapping[str, object]] = [
        {
            "role": "system",
            "content": active.services.prompt_bundle.prompts["researcher_question.md"],
        },
        {
            "role": "user",
            "content": json.dumps(
                {"prompt_version": "researcher.question", "payload": {"text": text}},
                sort_keys=True,
            ),
        },
    ]

    # What the corpus actually handed this question, as the measured path keeps it
    # per unit: existing in the corpus is not provenance, being answered for is.
    # The ledger is this question's own; the measured run's ledger is not shared
    # with it, so an interactive answer cannot lean on a locator a unit retrieved.
    ledger: set[str] = set()

    async def handle(
        invocation: ToolInvocation, attempts: tuple[AttemptTelemetry, ...]
    ) -> ToolResult:
        del attempts
        parsed = parse_tool(QUESTION_TOOLS, invocation.name, invocation.arguments)
        if isinstance(parsed, dict):
            return refused(parsed, invocation.name, active=active)
        if isinstance(parsed, SearchCorpusArgs):
            result, candidates = await run_search(active, parsed.phrase)
            # Same rule as the measured path: an error payload means the phrase was
            # refused before the corpus was queried, and a refusal is not a search.
            # A rejected query still carries an empty "results" list, so keying off
            # that alone counted a corpus call nobody made.
            if "error" not in result and "results" in result:
                active.finder_search_calls += 1
                ledger.update(candidate.locator for candidate in candidates)
            return ToolResult(result)
        if isinstance(parsed, ReadProvisionArgs):
            provision = await read_provision(active, parsed.locator)
            if provision is None:
                return refused(refusal("unknown_locator", locator=parsed.locator))
            active.provision_reads += 1
            ledger.add(provision.locator)
            return ToolResult(
                {
                    "locator": provision.locator,
                    "act_identifier": provision.act_identifier,
                    "text": provision.text,
                }
            )
        if isinstance(parsed, AnswerQuestionArgs):
            if not set(parsed.locators) <= ledger:
                return refused(refusal("locator_not_in_provenance_ledger"))
            answer.update(parsed.model_dump(mode="json"))
            return ToolResult({"recorded": True}, committed=True)
        return refused(refusal("unknown_tool"))

    turns = await run_conversation(
        active,
        messages=messages,
        tools=QUESTION_TOOLS,
        prompt_version="researcher.question",
        commit_tool=ANSWER_QUESTION.name,
        max_turns=RESEARCHER_QUESTION_MAX_TURNS,
        handle_tool=handle,
    )
    active.finder_tool_turns += turns
    return answer
