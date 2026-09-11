"""The model provider, over its Responses route."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, cast
from uuid import UUID

import openai
import pydantic
from openai.types.responses import Response, ResponseInputParam, ToolParam

if TYPE_CHECKING:
    import httpx2

from contract_analyzer.config import RunConfig, Settings
from contract_analyzer.model import (
    CONVERSE_RESERVED,
    RETRYABLE_SERVER_STATUSES,
    AttemptFault,
    AttemptSuccess,
    ToolConversation,
    ToolInvocation,
    ToolTurn,
    hash_json,
    invalid_response,
    run_attempts,
    token_counts,
    tool_payload,
    tool_turn_hash,
    usage_from_body,
    validate_request,
)

logger = logging.getLogger(__name__)


def _log_provider_error(status: int, body: object) -> None:
    """Log why the provider refused, where the reason cannot reach the record.

    An attempt records only its error code, so a refusal is indistinguishable
    from any other with the same status, and classifying one as transient or
    permanent then rests on nothing. The provider's own text says which, but it
    is free-form and can quote the request, so it stays in this machine's log
    and out of the run record, which is committed and must hold no document
    text.
    """
    detail: object = body
    if isinstance(body, dict):
        error = body.get("error")
        detail = error if isinstance(error, dict) else body
    logger.warning("model provider refused status=%s detail=%r", status, detail)


class OpenAICompatibleClient:
    """Any OpenAI-compatible endpoint, with per-run observable model pinning."""

    def __init__(
        self,
        settings: Settings,
        *,
        http_client: httpx2.AsyncClient | None = None,
        max_attempts: int = 3,
        model_timeout_seconds: int | None = None,
    ) -> None:
        settings.require_live()
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        self._settings = settings
        read_timeout = (
            RunConfig().model_timeout_seconds
            if model_timeout_seconds is None
            else model_timeout_seconds
        )
        self._client = openai.AsyncOpenAI(
            base_url=settings.model_base_url,
            # require_live above is what guarantees the credential is present.
            api_key=cast(str, settings.model_api_key),
            # Retries and their telemetry belong to run_attempts: a retry the
            # provider SDK performs on its own is a call this project cannot
            # record, cost or bound.
            max_retries=0,
            timeout=openai.Timeout(
                connect=10.0,
                read=read_timeout,
                write=30.0,
                pool=10.0,
            ),
            http_client=http_client,
            # A provider may require a header the OpenAI interface does not
            # define; the gateway called by default refuses a request without
            # its routing header. Configuration carries it so no provider is
            # named here.
            default_headers=dict(settings.model_extra_headers) or None,
        )
        self._max_attempts = max_attempts
        self._pinned_models: dict[UUID, str] = {}

    @property
    def max_attempts(self) -> int:
        return self._max_attempts

    async def aclose(self) -> None:
        await self._client.close()

    async def converse(self, conversation: ToolConversation) -> ToolTurn:
        validate_request(conversation, reserved=CONVERSE_RESERVED)
        payload = tool_payload(conversation, self._settings.model_name)

        async def attempt() -> (
            AttemptSuccess[tuple[str, tuple[ToolInvocation, ...]]] | AttemptFault
        ):
            response = await self._invoke(payload)
            if isinstance(response, AttemptFault):
                return response
            parts = _response_parts(response)
            if isinstance(parts, AttemptFault):
                return parts
            returned_model, content, usage, tool_calls = parts
            return AttemptSuccess(
                content=(content, tool_calls),
                returned_model=returned_model,
                response_hash=tool_turn_hash(content, tool_calls),
                usage=usage,
            )

        result = await run_attempts(
            conversation,
            requested_model=self._settings.model_name,
            prompt_hash=hash_json(payload),
            max_attempts=self._max_attempts,
            pinned_models=self._pinned_models,
            attempt=attempt,
        )
        return ToolTurn(
            content=result.content[0],
            tool_calls=result.content[1],
            requested_model=self._settings.model_name,
            returned_model=result.returned_model,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            attempts=result.attempts,
        )

    async def _invoke(self, payload: dict[str, object]) -> Response | AttemptFault:
        """Call the provider once and classify transport and status faults.

        Only the fields this route names go as arguments; anything else the
        request carries rides in ``extra_body``, so a parameter this project
        records reaches the provider whether or not the installed client library
        happens to have a keyword for it.
        """
        extra_body = {
            key: value for key, value in payload.items() if key not in _NAMED_FIELDS
        }
        try:
            return await self._client.responses.create(
                model=cast(str, payload["model"]),
                input=cast(ResponseInputParam, payload["input"]),
                temperature=cast(float, payload["temperature"]),
                tools=cast(list[ToolParam], payload["tools"]),
                store=False,
                extra_body=extra_body,
            )
        except openai.APIStatusError as error:
            status = error.status_code
            _log_provider_error(status, error.body)
            return AttemptFault(
                status="http_error",
                code=f"model_http_{status}",
                retryable=status == 429 or status in RETRYABLE_SERVER_STATUSES,
                usage=usage_from_body(error.body),
            )
        except openai.APIConnectionError:
            # APITimeoutError is also an APIConnectionError.
            return AttemptFault(
                status="transport_error",
                code="model_transport_error",
                retryable=True,
            )
        except (
            TypeError,
            ValueError,
            KeyError,
            IndexError,
            AttributeError,
            pydantic.ValidationError,
        ):
            return invalid_response("model_response_invalid_envelope")


_NAMED_FIELDS = frozenset({"model", "input", "temperature", "tools", "store"})


def _response_parts(
    response: Response,
) -> tuple[str, str, tuple[int, int] | None, tuple[ToolInvocation, ...]] | AttemptFault:
    """Validate the lenient SDK envelope before the attempt is recorded."""
    try:
        model = response.model
        if not isinstance(model, str) or not model:
            raise TypeError
        if not response.output:
            raise ValueError
        content: list[str] = []
        tool_calls: list[ToolInvocation] = []
        for item in response.output:
            if item.type == "message":
                for part in item.content:
                    text = getattr(part, "text", None)
                    if text is None:
                        continue
                    if not isinstance(text, str):
                        raise TypeError
                    content.append(text)
            elif item.type == "function_call":
                identifier = item.call_id
                name = item.name
                arguments = item.arguments
                if not all(
                    isinstance(value, str) for value in (identifier, name, arguments)
                ):
                    raise TypeError
                tool_calls.append(ToolInvocation(identifier, name, arguments))
            # Ignore reasoning and auxiliary items so hashes cover only acted-on output.
        return model, "".join(content), _usage(response), tuple(tool_calls)
    except (
        TypeError,
        ValueError,
        KeyError,
        IndexError,
        AttributeError,
        pydantic.ValidationError,
    ):
        return invalid_response("model_response_invalid_envelope")


def _usage(response: Response) -> tuple[int, int] | None:
    """Read token counts with the same validation used for error responses.

    run_attempts reports unusable counts as model_response_missing_usage.
    """
    usage = response.usage
    if usage is None:
        return None
    return token_counts(usage.input_tokens, usage.output_tokens)
