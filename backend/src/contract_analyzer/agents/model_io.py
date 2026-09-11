"""Budgeted access to the model, and the attempt telemetry every call records.

One policy governs a tool turn: clamp the request to the wall budget that is
left, record every attempt, and turn a deadline into budget exhaustion rather
than a failed run.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import uuid4

from contract_analyzer.agents.errors import BudgetExhausted
from contract_analyzer.agents.state import ActiveRun
from contract_analyzer.model import AttemptTelemetry
from contract_analyzer.storage import AttemptRecord


@asynccontextmanager
async def budgeted_call(active: ActiveRun) -> AsyncIterator[float]:
    """Yield the seconds this call has left, or refuse to start it."""
    if not active.budget.can_schedule():
        raise BudgetExhausted("budget_exhausted")
    remaining_seconds = active.budget.remaining_seconds()
    if remaining_seconds <= 0:
        raise BudgetExhausted("budget_exhausted")
    yield remaining_seconds


def record_attempts(
    active: ActiveRun,
    attempts: tuple[AttemptTelemetry, ...],
    unit_id: str | None = None,
) -> None:
    """Charge the attempts to the run's budget and store one record for each."""
    if not attempts:
        return
    tokens_added = sum(
        attempt.input_tokens + attempt.output_tokens for attempt in attempts
    )
    active.tokens_used += tokens_added
    for attempt in attempts:
        if attempt.status == "success" and attempt.returned_model is not None:
            active.returned_model = attempt.returned_model
        active.services.metadata.record_attempt(
            AttemptRecord(
                id=uuid4(),
                run_id=active.run_id,
                requested_model=attempt.requested_model,
                returned_model=attempt.returned_model,
                prompt_version=attempt.prompt_version,
                temperature=attempt.temperature,
                parameters=attempt.parameters,
                input_tokens=attempt.input_tokens,
                output_tokens=attempt.output_tokens,
                latency_ms=attempt.latency_ms,
                status=attempt.status,
                retry_number=attempt.retry_number,
                error_code=attempt.error_code,
                unit_id=unit_id,
                prompt_hash=attempt.prompt_hash,
                response_hash=attempt.response_hash,
            )
        )


def call_temperature(active: ActiveRun) -> float:
    value = active.request.parameters.get("temperature", 0.0)
    return float(value) if isinstance(value, (int, float)) else 0.0


def call_parameters(active: ActiveRun) -> dict[str, str | int | float | bool | None]:
    return {
        key: value
        for key, value in active.request.parameters.items()
        if key != "temperature"
    }
