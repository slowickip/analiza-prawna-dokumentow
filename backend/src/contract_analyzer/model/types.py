"""Model request and tool wire types."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

from contract_analyzer.domain import ParameterValue


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    parameters: Mapping[str, object]


@dataclass(frozen=True)
class ToolInvocation:
    id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class ToolConversation:
    run_id: UUID
    messages: tuple[Mapping[str, object], ...]
    tools: tuple[ToolDefinition, ...]
    prompt_version: str
    temperature: float
    parameters: Mapping[str, ParameterValue]
    timeout_seconds: float | None = None


class RequestContext(Protocol):
    """What the attempt driver needs from a call, whatever its payload."""

    @property
    def run_id(self) -> UUID: ...

    @property
    def prompt_version(self) -> str: ...

    @property
    def temperature(self) -> float: ...

    @property
    def parameters(self) -> Mapping[str, ParameterValue]: ...

    @property
    def timeout_seconds(self) -> float | None: ...
