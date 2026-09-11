"""One bounded tool conversation: the graph every role task runs inside.

The provider ignores tools when a strict response format is also sent, so a role
does not answer a schema -- it calls a tool. This module owns that exchange:
build the two opening messages, execute them on a real LangChain graph, hand
each tool call to the task's handler, and stop at the commit, at a silent turn,
or at the cap. There is exactly one execution path: the custom while-loop this
replaces is gone, and no fallback loop sits beside the graph.

The graph is thin on purpose. The model node is a minimal adapter over the
tested provider seam: every turn goes through ``converse``, so budgets,
per-attempt telemetry, retries and the missing-usage contract are exactly what
they were. Native tool validation and message routing do the plumbing; the
application keeps only the guards the library cannot know -- a commit ends the
task alone, extra calls in one turn are refused, and every attempt is recorded
under the task that spent it.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import StructuredTool
from langgraph.prebuilt.tool_node import ToolCallRequest
from pydantic import PrivateAttr, ValidationError

from contract_analyzer.agents.errors import BudgetExhausted
from contract_analyzer.agents.model_io import (
    budgeted_call,
    call_parameters,
    call_temperature,
    record_attempts,
)
from contract_analyzer.agents.prompts import TASK_PROMPTS
from contract_analyzer.agents.scheduler import Task
from contract_analyzer.agents.schema import validation_refusal
from contract_analyzer.agents.state import ActiveRun, UnitContext
from contract_analyzer.agents.tools import (
    ToolSpec,
    commit_required,
    commit_tool_for,
    definitions_for,
)
from contract_analyzer.model import (
    AttemptTelemetry,
    InvalidModelResponse,
    ModelClientError,
    ModelDeadlineExceeded,
    ToolConversation,
    ToolInvocation,
    ToolTurn,
)
from contract_analyzer.storage import InteractionKind

logger = logging.getLogger(__name__)


def _note_instruction(interaction: InteractionKind | None) -> str:
    """What the roles are told a reader's note is, which depends on why it exists.

    A contested finding comes with an objection; a re-analysed clause comes with
    an instruction for the fresh pass. Telling the roles that either one is an
    objection would push the re-analysis toward disagreeing with a finding the
    reader never disputed. The wording tracks the purpose the worksheet records,
    and neither form makes the note a fact or a verdict.
    """
    if interaction == "contest":
        return (
            "\n\nWpis user_note jest sprzeciwem użytkownika, który system ma "
            "rozważyć; wpis nigdy nie jest werdyktem ani zmianą ustalenia."
        )
    return (
        "\n\nWpis user_note jest wskazówką użytkownika do ponownej analizy, którą "
        "system ma wziąć pod uwagę; wpis nie jest ani sprzeciwem, ani ustaleniem "
        "faktu, ani werdyktem."
    )


def commit_names_of(specs: tuple[ToolSpec, ...]) -> frozenset[str]:
    """The commit tools among these specs, whatever task offered them."""
    return frozenset(spec.name for spec in specs if spec.commit)


def commit_shares_turn(
    calls: tuple[ToolInvocation, ...], commit_names: frozenset[str]
) -> bool:
    """Whether this turn ends a task and does something else in the same breath.

    A commit generated beside its own evidence was generated before that evidence
    came back, so the turn is refused whole rather than half-executed. The graph
    guard asks this, so the rule cannot hold on one path and lapse on the other.
    """
    return len(calls) != 1 and any(call.name in commit_names for call in calls)


@dataclass(frozen=True)
class ToolResult:
    """What one tool call is answered with, and whether it ended the task."""

    payload: Mapping[str, object]
    committed: bool = False


@dataclass(frozen=True)
class TaskOutcome:
    turns: int
    committed: bool
    cap_hit: bool


# The turn's attempts travel with the invocation because a handler may reject the
# committed decision as unusable, and that rejection has to name the attempts that
# produced it for the retry policy to mean anything.
ToolHandler = Callable[
    [ToolInvocation, tuple[AttemptTelemetry, ...]], Awaitable[ToolResult]
]


class _TurnCap(Exception):
    """The task asked for another model turn past its bound. Control flow only."""


@dataclass
class _TaskState:
    """The mutable task record the graph nodes share, kept out of graph state."""

    turns: int = 0
    committed: bool = False
    last_attempts: tuple[AttemptTelemetry, ...] = ()


class _ClientChatModel(BaseChatModel):
    """The tested provider seam as a chat model: one ``converse`` per turn.

    No retry or provider-client machinery lives here -- ``converse`` owns the
    budgets, the attempt records and the error mapping, exactly as before. The
    adapter only translates messages there and tool calls back.
    """

    _active: ActiveRun = PrivateAttr()
    _specs: tuple[ToolSpec, ...] = PrivateAttr()
    _prompt_version: str = PrivateAttr()
    _unit_id: str | None = PrivateAttr()
    _state: _TaskState = PrivateAttr()
    _max_turns: int = PrivateAttr()

    def __init__(
        self,
        *,
        active: ActiveRun,
        specs: tuple[ToolSpec, ...],
        prompt_version: str,
        unit_id: str | None,
        state: _TaskState,
        max_turns: int,
    ) -> None:
        super().__init__()
        self._active = active
        self._specs = specs
        self._prompt_version = prompt_version
        self._unit_id = unit_id
        self._state = state
        self._max_turns = max_turns

    @property
    def _llm_type(self) -> str:
        return "contract-analyzer-model-client"

    def bind_tools(
        self,
        tools: Any,
        *,
        tool_choice: Any = None,
        **kwargs: Any,
    ) -> Any:
        """The tools are already fixed by the task; binding changes nothing."""
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        raise NotImplementedError("this model runs asynchronously: use ainvoke")

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self._state.turns >= self._max_turns:
            raise _TurnCap
        turn = await converse(
            self._active,
            [_to_provider(message) for message in messages],
            tools=self._specs,
            prompt_version=self._prompt_version,
            unit_id=self._unit_id,
        )
        self._state.turns += 1
        self._state.last_attempts = turn.attempts
        _reject_ambiguous_ids(turn)
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(
                        content=turn.content,
                        tool_calls=[
                            {
                                "name": invocation.name,
                                "args": _tool_args(invocation, turn),
                                "id": invocation.id,
                                "type": "tool_call",
                            }
                            for invocation in turn.tool_calls
                        ],
                    )
                )
            ]
        )


def _reject_ambiguous_ids(turn: ToolTurn) -> None:
    """Reject duplicate or empty tool-call ids before any handler runs.

    The per-turn cap locates a call by its id, so an id shared by two calls
    would let the second masquerade as the first and slip past the bound. An
    ambiguous envelope fails loudly, with the recorded attempt kept.
    """
    seen: set[str] = set()
    for invocation in turn.tool_calls:
        if not invocation.id or invocation.id in seen:
            raise InvalidModelResponse("model_response_invalid_envelope", turn.attempts)
        seen.add(invocation.id)


def _tool_args(invocation: ToolInvocation, turn: ToolTurn) -> dict[str, Any]:
    """The call's arguments as the graph needs them: a mapping, not text.

    A provider envelope whose arguments are not JSON is an envelope violation,
    like a non-function tool call, and fails loudly rather than reaching a
    handler as a correctable mistake.
    """
    try:
        parsed = json.loads(invocation.arguments)
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        raise InvalidModelResponse(
            "model_response_invalid_envelope", turn.attempts
        ) from error
    if not isinstance(parsed, dict):
        raise InvalidModelResponse("model_response_invalid_envelope", turn.attempts)
    return parsed


def _to_provider(message: BaseMessage) -> dict[str, object]:
    """One graph message back into the provider transcript shape every test reads."""
    content = (
        message.content
        if isinstance(message.content, str)
        else json.dumps(message.content, ensure_ascii=False)
    )
    if isinstance(message, SystemMessage):
        return {"role": "system", "content": content}
    if isinstance(message, HumanMessage):
        return {"role": "user", "content": content}
    if isinstance(message, ToolMessage):
        return {
            "role": "tool",
            "tool_call_id": message.tool_call_id,
            "content": content,
        }
    if isinstance(message, AIMessage):
        return {
            "role": "assistant",
            "content": content,
            "tool_calls": [
                {
                    "id": call["id"],
                    "type": "function",
                    "function": {
                        "name": call["name"],
                        "arguments": json.dumps(call["args"], ensure_ascii=False),
                    },
                }
                for call in message.tool_calls
            ],
        }
    raise InvalidModelResponse("model_response_invalid_envelope", ())


def _refusal_text(error: ValidationError | Any) -> str:
    """Native validation failures in the application's refusal shape."""
    if isinstance(error, ValidationError):
        return json.dumps(validation_refusal(error), ensure_ascii=False)
    return json.dumps({"error": "invalid_arguments"}, ensure_ascii=False)


def _wrap_tool(
    spec: ToolSpec,
    handle_tool: ToolHandler,
    state: _TaskState,
    *,
    active: ActiveRun,
    role: str,
    task: str,
    unit_id: str | None,
) -> StructuredTool:
    """One offered tool: native validation, then the unchanged domain handler."""

    async def _call(**kwargs: Any) -> str:
        logger.debug(
            "tool call run=%s unit=%s role=%s task=%s tool=%s arguments=%d",
            active.run_id,
            unit_id,
            role,
            task,
            spec.name,
            len(kwargs),
        )
        invocation = ToolInvocation(
            id="",
            name=spec.name,
            arguments=spec.arguments.model_validate(kwargs).model_dump_json(),
        )
        result = await handle_tool(invocation, state.last_attempts)
        if result.committed:
            state.committed = True
        return json.dumps(result.payload, ensure_ascii=False)

    return StructuredTool(
        name=spec.name,
        description=spec.description,
        args_schema=spec.arguments,
        coroutine=_call,
        handle_validation_error=_refusal_text,
    )


class _TaskGuards(AgentMiddleware):
    """The rules the library cannot know, refused before any handler runs.

    A commit beside another call is refused whole: the sibling ran in raw
    LangChain would execute both, and the commit would mutate state beside
    evidence generated in the same breath. A call past the per-turn bound is
    refused the same way. Neither refusal runs a handler, so neither leaves a
    trace the refused call could later be mistaken for.

    Handler execution is serialised on one lock per conversation: the graph may
    dispatch a turn's calls concurrently, while the baseline ran them in
    provider wire order. The lock covers offered and unoffered calls alike, so
    a refusal for an unknown tool cannot slip between two state mutations.
    """

    def __init__(
        self,
        *,
        active: ActiveRun,
        unit_id: str | None,
        commit_names: frozenset[str],
        max_calls_per_turn: int | None,
        handle_tool: ToolHandler,
        state: _TaskState,
    ) -> None:
        self._active = active
        self._unit_id = unit_id
        self._commit_names = commit_names
        self._max_calls_per_turn = max_calls_per_turn
        self._handle_tool = handle_tool
        self._state = state
        self._lock = asyncio.Lock()

    async def awrap_model_call(self, request: Any, handler: Any) -> AIMessage:
        """A committed task needs no further model turn: end it without a call.

        The graph would otherwise spend one more provider turn on a farewell
        after every commit, charging every task for a call that decides nothing.
        """
        if self._state.committed:
            return AIMessage(content="")
        result: AIMessage = await handler(request)
        return result

    async def awrap_tool_call(
        self, request: ToolCallRequest, handler: Any
    ) -> ToolMessage:
        siblings = request.state["messages"][-1].tool_calls
        if commit_shares_turn(
            tuple(
                ToolInvocation(id=call["id"], name=call["name"], arguments="")
                for call in siblings
            ),
            self._commit_names,
        ):
            for call in siblings:
                logger.warning(
                    "tool refusal run=%s unit=%s tool=%s code=commit_must_be_alone",
                    self._active.run_id,
                    self._unit_id,
                    call["name"],
                )
            return ToolMessage(
                content=json.dumps(refusal("commit_must_be_alone")),
                tool_call_id=request.tool_call["id"],
                status="error",
            )
        if self._max_calls_per_turn is not None:
            position = next(
                index
                for index, call in enumerate(siblings)
                if call["id"] == request.tool_call["id"]
            )
            if position >= self._max_calls_per_turn:
                logger.warning(
                    "tool refusal run=%s unit=%s tool=%s "
                    "code=phrase_budget_exceeded_for_this_turn",
                    self._active.run_id,
                    self._unit_id,
                    request.tool_call["name"],
                )
                return ToolMessage(
                    content=json.dumps(
                        refusal(
                            "phrase_budget_exceeded_for_this_turn",
                            limit_per_turn=self._max_calls_per_turn,
                        )
                    ),
                    tool_call_id=request.tool_call["id"],
                    status="error",
                )
        async with self._lock:
            if request.tool is None:
                # Not offered on this task, but still the handler's to refuse:
                # the verifier's challenge tool is unoffered once a challenge
                # exists, and the refusal the model corrects on names the
                # limit, not the routing. Same handler, same codes, and still
                # behind both turn guards, so a mixed or over-budget turn never
                # reaches it.
                invocation = ToolInvocation(
                    id=request.tool_call["id"] or "",
                    name=request.tool_call["name"],
                    arguments=json.dumps(request.tool_call["args"], ensure_ascii=False),
                )
                result = await self._handle_tool(invocation, self._state.last_attempts)
                if result.committed:
                    self._state.committed = True
                return ToolMessage(
                    content=json.dumps(result.payload, ensure_ascii=False),
                    tool_call_id=request.tool_call["id"],
                )
            result = await handler(request)
            message: ToolMessage = result
            return message


async def converse(
    active: ActiveRun,
    messages: list[Mapping[str, object]],
    *,
    tools: tuple[ToolSpec, ...],
    prompt_version: str,
    unit_id: str | None = None,
) -> ToolTurn:
    """One tool turn, under the same budgets and telemetry as a schema call."""
    async with budgeted_call(active) as remaining_seconds:
        try:
            turn = await active.services.client.converse(
                ToolConversation(
                    run_id=active.run_id,
                    messages=tuple(messages),
                    tools=definitions_for(tools),
                    prompt_version=prompt_version,
                    temperature=call_temperature(active),
                    parameters=call_parameters(active),
                    timeout_seconds=remaining_seconds,
                )
            )
        except ModelDeadlineExceeded as error:
            record_attempts(active, error.attempts, unit_id)
            active.budget.expire()
            raise BudgetExhausted("budget_exhausted") from None
        except ModelClientError as error:
            record_attempts(active, error.attempts, unit_id)
            raise
        record_attempts(active, turn.attempts, unit_id)
        return turn


async def run_task(
    active: ActiveRun,
    *,
    role: str,
    task: Task,
    payload: Mapping[str, object],
    tools: tuple[ToolSpec, ...],
    handle_tool: ToolHandler,
    max_turns: int,
    max_calls_per_turn: int | None = None,
    allow_uncommitted: bool = False,
    unit_id: str | None = None,
    prompt_version: str | None = None,
) -> TaskOutcome:
    """Run one role task to its commit, to silence, or to its turn cap.

    The recorded ``prompt_version`` defaults to the task name, so an attempt
    record says which role spent the tokens and under which prompt. A task may
    override it when its attempts must stay attributable to a stable name.
    """
    version = task if prompt_version is None else prompt_version
    system_prompt = active.services.prompt_bundle.prompts[TASK_PROMPTS[task]]
    if active.request.user_note is not None:
        system_prompt += _note_instruction(active.request.interaction)
    return await _drive(
        active,
        role=role,
        task=task,
        prompt_version=version,
        opening=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                # The task field attributes the transcript without matching prose.
                "content": json.dumps(
                    {"prompt_version": version, "payload": payload}, sort_keys=True
                ),
            },
        ],
        tools=tools,
        commit_tool=commit_tool_for(task),
        must_commit=commit_required(task),
        max_turns=max_turns,
        handle_tool=handle_tool,
        max_calls_per_turn=max_calls_per_turn,
        allow_uncommitted=allow_uncommitted,
        unit_id=unit_id,
    )


async def run_conversation(
    active: ActiveRun,
    *,
    messages: list[Mapping[str, object]],
    tools: tuple[ToolSpec, ...],
    prompt_version: str,
    commit_tool: str,
    max_turns: int,
    handle_tool: ToolHandler,
) -> int:
    """Run one transcript-owned interactive task to its required commit."""
    outcome = await _drive(
        active,
        role="interactive",
        task=prompt_version,
        prompt_version=prompt_version,
        opening=messages,
        tools=tools,
        commit_tool=commit_tool,
        must_commit=True,
        max_turns=max_turns,
        handle_tool=handle_tool,
    )
    return outcome.turns


async def _drive(
    active: ActiveRun,
    *,
    role: str,
    task: str,
    prompt_version: str,
    opening: list[Mapping[str, object]],
    tools: tuple[ToolSpec, ...],
    commit_tool: str | None,
    must_commit: bool,
    max_turns: int,
    handle_tool: ToolHandler,
    max_calls_per_turn: int | None = None,
    allow_uncommitted: bool = False,
    unit_id: str | None = None,
) -> TaskOutcome:
    """The single bounded tool execution shared by measured and interactive tasks.

    The graph ends on a commit or on a silent turn. A task that must end in a
    commit gets one nudge if the model falls silent instead; the nudge resumes
    the same history, so no tool runs twice. A task that asks past its bound
    stops there, with the attempts it already recorded kept.
    """
    state = _TaskState()
    agent = create_agent(
        model=_ClientChatModel(
            active=active,
            specs=tools,
            prompt_version=prompt_version,
            unit_id=unit_id,
            state=state,
            max_turns=max_turns,
        ),
        tools=[
            _wrap_tool(
                spec,
                handle_tool,
                state,
                active=active,
                role=role,
                task=task,
                unit_id=unit_id,
            )
            for spec in tools
        ],
        middleware=[
            _TaskGuards(
                active=active,
                unit_id=unit_id,
                commit_names=commit_names_of(tools),
                max_calls_per_turn=max_calls_per_turn,
                handle_tool=handle_tool,
                state=state,
            )
        ],
    )
    nudged = False
    history: list[Any] = [dict(message) for message in opening]
    while True:
        try:
            result = await agent.ainvoke({"messages": history})
        except _TurnCap:
            break
        history = result["messages"]
        if state.committed:
            break
        last = history[-1]
        if isinstance(last, AIMessage) and not last.tool_calls:
            if commit_tool is None or not must_commit:
                break
            if nudged:
                logger.warning(
                    "task ended without commit run=%s unit=%s task=%s tool=%s",
                    active.run_id,
                    unit_id,
                    task,
                    commit_tool,
                )
                if allow_uncommitted:
                    break
                raise InvalidModelResponse(
                    "model_response_schema_invalid", state.last_attempts
                )
            nudged = True
            logger.warning(
                "nudging task to commit run=%s unit=%s task=%s tool=%s",
                active.run_id,
                unit_id,
                task,
                commit_tool,
            )
            history = [
                *history,
                HumanMessage(
                    content=f"Zakończ, wywołując narzędzie {commit_tool}.",
                ),
            ]
            continue
        break
    cap_hit = state.turns >= max_turns and not state.committed
    if cap_hit:
        logger.warning(
            "task hit turn cap run=%s unit=%s task=%s limit=%d",
            active.run_id,
            unit_id,
            task,
            max_turns,
        )
    if (
        must_commit
        and commit_tool is not None
        and not state.committed
        and not allow_uncommitted
    ):
        # A missing commit leaves the worksheet unable to advance.
        raise InvalidModelResponse("model_response_schema_invalid", ())
    return TaskOutcome(turns=state.turns, committed=state.committed, cap_hit=cap_hit)


def refusal(code: str, **fields: object) -> dict[str, Any]:
    """The code-only refusal a tool call is answered with."""
    return {"error": code, **fields}


def refused(
    payload: Mapping[str, object],
    tool_name: str | None = None,
    *,
    active: ActiveRun | None = None,
    unit_id: str | None = None,
    context: UnitContext | None = None,
) -> ToolResult:
    """Pass a validation refusal back to the model, recording that it happened."""
    if tool_name is not None:
        if context is not None:
            active = context.active
            unit_id = context.call_unit.unit_id
        logger.warning(
            "tool refusal run=%s unit=%s tool=%s code=%s",
            None if active is None else active.run_id,
            unit_id,
            tool_name,
            payload.get("error"),
        )
    return ToolResult(payload)
