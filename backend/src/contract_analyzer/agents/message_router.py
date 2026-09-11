"""One bounded model call deciding what a single reader message asks for.

The reader writes one message and presses send. Which of the three post-run
interactions it means, and which finding or unit it means it about, is a reading
of the message, so the system reads it with the model rather than making the
reader choose from a menu. Routing reads no corpus and changes no finding: it
returns an intent and a target drawn from identifiers the run already holds.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from uuid import UUID

from contract_analyzer.agents.errors import MessageNotRoutable
from contract_analyzer.agents.interactive_tools import (
    ROUTE_MESSAGE,
    ROUTE_MESSAGE_MAX_TURNS,
    ROUTE_TOOLS,
    RouteMessageArgs,
)
from contract_analyzer.agents.session import ChatMessage
from contract_analyzer.agents.state import ActiveRun
from contract_analyzer.agents.tools import parse_tool
from contract_analyzer.agents.turns import (
    ToolResult,
    refusal,
    refused,
    run_conversation,
)
from contract_analyzer.model import AttemptTelemetry, ToolInvocation
from contract_analyzer.model.attempts import InvalidModelResponse
from contract_analyzer.storage import FindingRecord

logger = logging.getLogger(__name__)


def _catalogue(
    findings: Sequence[FindingRecord], unit_ids: Sequence[str]
) -> dict[str, object]:
    """What the router may choose from: nothing here is new information."""
    return {
        "findings": [
            {
                "finding_id": str(item.id),
                "code": item.code,
                "unit_id": item.unit_id,
                "legal_locators": list(item.legal_locators),
                "synthesis_prose": item.synthesis_prose,
            }
            for item in findings
        ],
        "unit_ids": list(unit_ids),
    }


async def route_message(
    active: ActiveRun,
    message: str,
    findings: Sequence[FindingRecord],
    unit_ids: Sequence[str],
    history: Sequence[ChatMessage] = (),
) -> RouteMessageArgs:
    """Classify one reader message, or fail loudly when the model commits nothing."""
    known_findings = {str(item.id) for item in findings}
    known_units = set(unit_ids)
    routed: list[RouteMessageArgs] = []

    messages: list[Mapping[str, object]] = [
        {
            "role": "system",
            "content": active.services.prompt_bundle.prompts["route_message.md"],
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "prompt_version": "interaction.route",
                    "payload": {
                        "message": message,
                        "history": [
                            {"role": item.role, "content": item.content}
                            for item in history
                        ],
                        **_catalogue(findings, unit_ids),
                    },
                },
                sort_keys=True,
                ensure_ascii=False,
            ),
        },
    ]

    async def handle(
        invocation: ToolInvocation, attempts: tuple[AttemptTelemetry, ...]
    ) -> ToolResult:
        del attempts
        parsed = parse_tool(ROUTE_TOOLS, invocation.name, invocation.arguments)
        if isinstance(parsed, dict):
            return refused(parsed, invocation.name, active=active)
        if not isinstance(parsed, RouteMessageArgs):
            return refused(refusal("unknown_tool"))

        # A target the run does not hold is not a target. Refusing here rather
        # than dropping it keeps the router from silently downgrading a contest
        # into a question about nothing in particular.
        if parsed.intent == "contest":
            if parsed.finding_id not in known_findings:
                return refused(refusal("unknown_finding_id"))
        elif parsed.intent == "analyse":
            if parsed.unit_id not in known_units:
                return refused(refusal("unknown_unit_id"))
        else:
            unknown = [
                value for value in parsed.finding_ids if value not in known_findings
            ]
            if unknown:
                return refused(refusal("unknown_finding_id"))

        routed.append(parsed)
        return ToolResult({"recorded": True}, committed=True)

    try:
        turns = await run_conversation(
            active,
            messages=messages,
            tools=ROUTE_TOOLS,
            prompt_version="interaction.route",
            commit_tool=ROUTE_MESSAGE.name,
            max_turns=ROUTE_MESSAGE_MAX_TURNS,
            handle_tool=handle,
        )
    except InvalidModelResponse as error:
        # Exhausting the bounded turns without a usable commit is, for the reader,
        # a message the system could not read. The underlying model failure is
        # logged rather than dropped, so a schema that breaks every message stays
        # visible as such instead of hiding behind a rephrase prompt.
        logger.warning(
            "message routing committed nothing run=%s cause=%s",
            active.run_id,
            error.code,
        )
        raise MessageNotRoutable("message_not_routable") from error
    # Deliberately not counted as finder turns. The router reaches no corpus, so a
    # route child reporting finder turns beside zero searches and zero reads makes
    # the classifier's cost read as retrieval, and cost is a reported dimension.
    # Its spend is on the attempt records under prompt_version interaction.route.
    del turns
    if not routed:
        raise MessageNotRoutable("message_not_routable")
    return routed[0]


def routed_uuid(value: str) -> UUID:
    """One identifier the router named, or a refusal to route the message.

    The tool takes these as free text, because a model that has to guess a UUID
    shape guesses worse. "the second one" therefore reaches here as a string,
    and parsing it where it is used raised ValueError out of a request handler.
    A message the router could not tie to a real identifier is a message it did
    not read, which is what MessageNotRoutable already means.
    """
    try:
        return UUID(value)
    except ValueError as error:
        raise MessageNotRoutable("message_not_routable") from error


def selected_for_ask(
    routed: RouteMessageArgs, findings: Sequence[FindingRecord]
) -> tuple[UUID, ...]:
    """Findings an ask is scoped to: the named ones, else the whole run.

    The reader no longer picks a finding before writing, so an unnamed question
    is a question about the analysis, not a question about nothing.
    """
    if routed.finding_ids:
        return tuple(routed_uuid(value) for value in routed.finding_ids)
    return tuple(item.id for item in findings)
