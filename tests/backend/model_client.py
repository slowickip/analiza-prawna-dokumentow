"""A model client whose answers are scripted by an ``httpx`` mock transport.

The connector reaches the provider through the ``openai`` SDK, which speaks
``httpx2`` -- a different distribution from the ``httpx`` this
project uses elsewhere. Pipeline tests script model answers with plain ``httpx``
handlers, so this adapter carries an ``httpx2`` request into the given ``httpx``
transport and the answer back out. It translates transport and nothing else:
method, URL, headers and body cross unchanged in both directions, and an
``httpx`` transport error becomes its ``httpx2`` equivalent so the SDK reports
it as a connection failure rather than swallowing it.

Tests of the client itself script ``httpx2`` directly instead, so that the wire
they assert on is the real one.
"""

from __future__ import annotations

import httpx
import httpx2

from contract_analyzer.config import Settings
from contract_analyzer.openai_compatible import OpenAICompatibleClient

TEST_API_KEY = "test-secret-key"


class _RelayTransport(httpx2.AsyncBaseTransport):
    def __init__(self, transport: httpx.AsyncBaseTransport) -> None:
        self._transport = transport

    async def handle_async_request(self, request: httpx2.Request) -> httpx2.Response:
        relayed = httpx.Request(
            request.method,
            str(request.url),
            headers=list(request.headers.raw),
            content=request.content,
        )
        try:
            answer = await self._transport.handle_async_request(relayed)
            await answer.aread()
        except httpx.TransportError as error:
            raise httpx2.ConnectError(str(error), request=request) from None
        return httpx2.Response(
            answer.status_code,
            headers=list(answer.headers.raw),
            content=answer.content,
            request=request,
        )


def mock_model_client(
    transport: httpx.AsyncBaseTransport,
    *,
    settings: Settings | None = None,
    max_attempts: int = 3,
) -> OpenAICompatibleClient:
    resolved = settings or Settings(model_api_key=TEST_API_KEY)
    return OpenAICompatibleClient(
        resolved,
        http_client=httpx2.AsyncClient(transport=_RelayTransport(transport)),
        max_attempts=max_attempts,
    )
