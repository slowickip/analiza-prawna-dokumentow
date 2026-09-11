"""Model attempt records, retry policy, and telemetry."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from uuid import UUID

from contract_analyzer.domain import AttemptStatus, ParameterValue

from .types import RequestContext, ToolInvocation

RETRYABLE_SERVER_STATUSES = frozenset({500, 502, 503, 504})


@dataclass(frozen=True)
class AttemptTelemetry:
    requested_model: str
    returned_model: str | None
    prompt_version: str
    temperature: float
    parameters: Mapping[str, ParameterValue]
    input_tokens: int
    output_tokens: int
    latency_ms: float
    status: AttemptStatus
    retry_number: int
    error_code: str | None
    prompt_hash: str
    response_hash: str | None


@dataclass(frozen=True)
class ToolTurn:
    content: str
    tool_calls: tuple[ToolInvocation, ...]
    requested_model: str
    returned_model: str
    input_tokens: int
    output_tokens: int
    attempts: tuple[AttemptTelemetry, ...]


@dataclass(frozen=True)
class AttemptSuccess[ContentT]:
    content: ContentT
    returned_model: str
    response_hash: str
    # ``None`` means the provider reported no token counts at all, which is not
    # the same fact as reporting zero; see ``run_attempts``.
    usage: tuple[int, int] | None


@dataclass(frozen=True)
class AttemptFault:
    status: AttemptStatus
    code: str
    retryable: bool = False
    returned_model: str | None = None
    response_hash: str | None = None
    usage: tuple[int, int] | None = None


def invalid_response(
    code: str,
    returned_model: str | None = None,
    response_hash: str | None = None,
) -> AttemptFault:
    """A well-formed HTTP answer whose body the pipeline cannot use.

    Never retried: the same prompt at the same temperature would produce the
    same unusable shape, and a retry would spend budget hiding the defect.
    """
    return AttemptFault(
        status="invalid_response",
        code=code,
        retryable=False,
        returned_model=returned_model,
        response_hash=response_hash,
    )


@dataclass(frozen=True)
class RequestResult[ContentT]:
    content: ContentT
    returned_model: str
    input_tokens: int
    output_tokens: int
    attempts: tuple[AttemptTelemetry, ...]


class ModelClientError(Exception):
    def __init__(self, code: str, attempts: tuple[AttemptTelemetry, ...]) -> None:
        super().__init__(code)
        self.code = code
        self.attempts = attempts


class ModelCallFailed(ModelClientError):
    pass


class InvalidModelResponse(ModelClientError):
    pass


class ModelIdentityChanged(ModelClientError):
    pass


class ModelDeadlineExceeded(ModelClientError):
    pass


async def run_attempts[ContentT](
    request: RequestContext,
    *,
    requested_model: str,
    prompt_hash: str,
    max_attempts: int,
    pinned_models: dict[UUID, str],
    attempt: Callable[[], Awaitable[AttemptSuccess[ContentT] | AttemptFault]],
) -> RequestResult[ContentT]:
    """Drive one model request to a result or a raised failure.

    ``attempt`` performs a single provider call and reports what came back; it
    must not retry on its own. Retries, the deadline, model-identity pinning and
    the per-attempt record are decided here, where the run's budget and its
    telemetry are one policy rather than a property of whichever library makes
    the call.
    """
    attempts: list[AttemptTelemetry] = []
    deadline = (
        time.monotonic() + request.timeout_seconds
        if request.timeout_seconds is not None
        else None
    )

    def record(
        *,
        retry_number: int,
        started: float,
        status: AttemptStatus,
        error_code: str | None,
        response_hash: str | None = None,
        returned_model: str | None = None,
        usage: tuple[int, int] = (0, 0),
    ) -> None:
        attempts.append(
            AttemptTelemetry(
                requested_model=requested_model,
                returned_model=returned_model,
                prompt_version=request.prompt_version,
                temperature=request.temperature,
                parameters=dict(request.parameters),
                input_tokens=usage[0],
                output_tokens=usage[1],
                latency_ms=(time.perf_counter() - started) * 1000,
                status=status,
                retry_number=retry_number,
                error_code=error_code,
                prompt_hash=prompt_hash,
                response_hash=response_hash,
            )
        )

    for retry_number in range(max_attempts):
        started = time.perf_counter()
        outcome: AttemptSuccess[ContentT] | AttemptFault
        try:
            if deadline is None:
                outcome = await attempt()
            else:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError
                outcome = await asyncio.wait_for(attempt(), timeout=remaining)
        except TimeoutError:
            record(
                retry_number=retry_number,
                started=started,
                status="transport_error",
                error_code="budget_exhausted",
            )
            raise ModelDeadlineExceeded("budget_exhausted", tuple(attempts)) from None

        # Error and invalid-response paths stay tolerant: telemetry records what
        # it has. Only the success path below treats a missing count as a failure.
        usage = outcome.usage if outcome.usage is not None else (0, 0)

        if isinstance(outcome, AttemptFault):
            record(
                retry_number=retry_number,
                started=started,
                status=outcome.status,
                error_code=outcome.code,
                response_hash=outcome.response_hash,
                returned_model=outcome.returned_model,
                usage=usage,
            )
            if outcome.retryable and retry_number + 1 < max_attempts:
                continue
            if outcome.status == "invalid_response":
                raise InvalidModelResponse(outcome.code, tuple(attempts))
            raise ModelCallFailed(outcome.code, tuple(attempts))

        pinned = pinned_models.get(request.run_id)
        if pinned is not None and pinned != outcome.returned_model:
            code = "model_identity_changed"
            record(
                retry_number=retry_number,
                started=started,
                status="model_identity_changed",
                error_code=code,
                response_hash=outcome.response_hash,
                returned_model=outcome.returned_model,
                usage=usage,
            )
            raise ModelIdentityChanged(code, tuple(attempts))
        if outcome.usage is None:
            code = "model_response_missing_usage"
            record(
                retry_number=retry_number,
                started=started,
                status="invalid_response",
                error_code=code,
                response_hash=outcome.response_hash,
                returned_model=outcome.returned_model,
            )
            raise InvalidModelResponse(code, tuple(attempts))

        pinned_models.setdefault(request.run_id, outcome.returned_model)
        record(
            retry_number=retry_number,
            started=started,
            status="success",
            error_code=None,
            response_hash=outcome.response_hash,
            returned_model=outcome.returned_model,
            usage=usage,
        )
        return RequestResult(
            content=outcome.content,
            returned_model=outcome.returned_model,
            input_tokens=sum(record_.input_tokens for record_ in attempts),
            output_tokens=sum(record_.output_tokens for record_ in attempts),
            attempts=tuple(attempts),
        )

    raise AssertionError("bounded attempt loop did not terminate")
