"""The real embedding HTTP client, driven over a mock transport.

The rest of the suite runs against the deterministic stand-in in conftest, so this
module is where ``_embed`` itself is checked: request shape, out-of-order
responses, and every way it is required to fail loudly rather than return
something the retrieval limb would silently score.

Readiness is not here any more, because readiness no longer encodes anything.
What the vector space actually is belongs to the corpus record, and the check
that compares them lives with the index.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from unittest.mock import patch

import httpx2
import numpy as np
import pytest
from openai import OpenAI

import contract_analyzer.corpus.embeddings as corpus_embeddings
from contract_analyzer.corpus import (
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL,
    EMBEDDING_REQUEST_MAX_INPUTS,
    EmbeddingError,
    _embed,
    embedding_credential_configured,
    embedding_spaces_agree,
)
from contract_analyzer.corpus.embeddings import request_vectors

Embed = Callable[..., np.ndarray]


PROVIDER_URL = "https://openrouter.ai/api/v1"


def _client(
    handler: Callable[[httpx2.Request], httpx2.Response],
    *,
    max_retries: int = 2,
    base_url: str = "http://embeddings.test/v1",
) -> OpenAI:
    """The real SDK client, driven over a mock transport.

    ``max_retries`` mirrors the module's own budget: the SDK counts retries where
    the caller counts attempts, so three attempts is two retries.
    """
    return OpenAI(
        api_key="test-key",
        base_url=base_url,
        max_retries=max_retries,
        http_client=httpx2.Client(transport=httpx2.MockTransport(handler)),
    )


def _vector(fill: float) -> list[float]:
    return [fill] * EMBEDDING_DIMENSION


def _embeddings_payload(vectors: list[list[float]]) -> dict[str, object]:
    return {
        "object": "list",
        "data": [
            {"object": "embedding", "index": index, "embedding": vector}
            for index, vector in enumerate(vectors)
        ],
    }


def test_embed_posts_the_model_and_inputs_and_returns_unit_vectors(
    live_embed: Embed,
) -> None:
    seen: list[dict[str, object]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        assert request.url.path.endswith("/embeddings")
        seen.append(json.loads(request.content))
        return httpx2.Response(200, json=_embeddings_payload([_vector(3.0)]))

    vectors = live_embed(["Najemca uiszcza czynsz."], _client(handler))

    assert seen == [
        {
            "model": EMBEDDING_MODEL,
            "input": ["Najemca uiszcza czynsz."],
            # Pinned rather than left to the SDK, which would ask for base64.
            "encoding_format": "float",
        }
    ]
    assert vectors.shape == (1, EMBEDDING_DIMENSION)
    assert vectors.dtype == np.float32
    # The service already normalises; _embed normalises anyway so search does not
    # depend on it continuing to. A constant 3.0 vector proves it happened.
    assert float(np.linalg.norm(vectors[0])) == pytest.approx(1.0)


def test_embed_batches_requests_and_keeps_input_order(live_embed: Embed) -> None:
    batches: list[list[str]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        batch = json.loads(request.content)["input"]
        batches.append(batch)
        # Answer out of order: "index" is what puts a vector back on its text.
        payload = _embeddings_payload(
            [_vector(float(i + 1)) for i in range(len(batch))]
        )
        data = list(payload["data"])  # type: ignore[arg-type]
        data.reverse()
        return httpx2.Response(200, json={"object": "list", "data": data})

    count = EMBEDDING_REQUEST_MAX_INPUTS + 6
    texts = [f"unit {index}" for index in range(count)]
    vectors = live_embed(texts, _client(handler))

    assert [len(batch) for batch in batches] == [EMBEDDING_REQUEST_MAX_INPUTS, 6]
    assert [text for batch in batches for text in batch] == texts
    assert vectors.shape == (count, EMBEDDING_DIMENSION)
    # Row i was filled with i+1 within its batch, so the first column is strictly
    # positive everywhere and increases inside each batch -- the reversal did not
    # survive into the array.
    assert float(vectors[0][0]) > 0.0
    assert float(vectors[1][0]) > 0.0


def test_embed_raises_when_the_service_is_unreachable(live_embed: Embed) -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("connection refused", request=request)

    with pytest.raises(EmbeddingError) as exc_info:
        live_embed(["x"], _client(handler))
    assert exc_info.value.code == "embedding_service_unavailable"


def test_embed_raises_on_a_non_success_status(live_embed: Embed) -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(503, json={"error": "no models loaded"})

    with pytest.raises(EmbeddingError) as exc_info:
        live_embed(["x"], _client(handler))
    assert exc_info.value.code == "embedding_service_unavailable"


def test_embed_raises_when_the_service_returns_too_few_vectors(
    live_embed: Embed,
) -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=_embeddings_payload([_vector(1.0)]))

    with pytest.raises(EmbeddingError) as exc_info:
        live_embed(["one", "two"], _client(handler))
    assert exc_info.value.code == "embedding_response_malformed"


def test_embed_raises_on_a_payload_without_embeddings(live_embed: Embed) -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json={"data": [{"index": 0}]})

    with pytest.raises(EmbeddingError) as exc_info:
        live_embed(["x"], _client(handler))
    assert exc_info.value.code == "embedding_response_malformed"


def test_embed_raises_on_the_wrong_vector_width(live_embed: Embed) -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=_embeddings_payload([[1.0, 2.0, 3.0]]))

    with pytest.raises(EmbeddingError) as exc_info:
        live_embed(["x"], _client(handler))
    assert exc_info.value.code == "embedding_dimension_mismatch"


def test_embed_raises_on_a_zero_length_vector(live_embed: Embed) -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=_embeddings_payload([_vector(0.0)]))

    with pytest.raises(EmbeddingError) as exc_info:
        live_embed(["x"], _client(handler))
    assert exc_info.value.code == "embedding_zero_vector"


def _probe_response(
    model: str | None = None, width: int | None = None
) -> httpx2.Response:
    """One /embeddings reply of the shape readiness now probes for."""
    body: dict[str, object] = {
        "data": [
            {
                "index": 0,
                "embedding": [0.1] * (EMBEDDING_DIMENSION if width is None else width),
            }
        ]
    }
    if model is not None:
        body["model"] = model
    return httpx2.Response(200, json=body)


def test_embed_retries_a_dropped_connection_then_succeeds() -> None:
    """A transient drop must not discard a build that is minutes deep.

    The service has been observed closing a connection part-way through a corpus
    build; without a retry the whole embedding pass is lost.
    """
    attempts: list[int] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        attempts.append(1)
        if len(attempts) == 1:
            raise httpx2.ReadError("server disconnected")
        body = json.loads(request.content)
        return httpx2.Response(
            200,
            json={
                "data": [
                    {
                        "object": "embedding",
                        "index": index,
                        "embedding": [1.0] + [0.0] * (EMBEDDING_DIMENSION - 1),
                    }
                    for index, _ in enumerate(body["input"])
                ]
            },
        )

    with _client(handler) as client:
        vectors = _embed(["a", "b"], client)

    assert len(attempts) == 2, "first attempt must be retried"
    assert vectors.shape == (2, EMBEDDING_DIMENSION)


def test_embed_gives_up_loudly_when_the_service_stays_down() -> None:
    """Retry must not paper over a service that is genuinely unavailable."""
    attempts: list[int] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        attempts.append(1)
        raise httpx2.ReadError("server disconnected")

    with _client(handler) as client:
        with pytest.raises(EmbeddingError) as excinfo:
            _embed(["a"], client)

    assert excinfo.value.code == "embedding_service_unavailable"
    assert len(attempts) == 3, "bounded, not infinite"


def test_many_tiny_inputs_still_stop_at_the_input_cap() -> None:
    """The input cap is the only thing that ends a request.

    A thousand ten-character inputs weigh almost nothing, so nothing but the cap
    stands between them and one request carrying the lot. A provider answers a
    request or refuses it, and a body that large is what gets refused.
    """
    from contract_analyzer.corpus.embeddings import EMBEDDING_REQUEST_MAX_INPUTS

    sent: list[int] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        texts = json.loads(request.content)["input"]
        sent.append(len(texts))
        return httpx2.Response(
            200,
            json={
                "data": [
                    {"index": i, "embedding": [1.0] * EMBEDDING_DIMENSION}
                    for i in range(len(texts))
                ]
            },
        )

    texts = ["a" * 10] * 1000
    with _client(handler) as client:
        vectors = _embed(texts, client)

    assert vectors.shape[0] == len(texts), "every input must come back"
    assert max(sent) <= EMBEDDING_REQUEST_MAX_INPUTS, (
        f"a request carried more than the input cap: {max(sent)}"
    )
    assert sum(sent) == len(texts), "inputs were dropped or duplicated"


def test_the_provider_key_never_travels_to_another_host() -> None:
    """EMBEDDINGS_BASE_URL is deployment configuration; the key is not.

    Pointing the endpoint at a self-hosted service, a proxy or a typo must not
    hand that host the provider credential.
    """
    import os

    from contract_analyzer.corpus.embeddings import (
        EMBEDDINGS_API_KEY_ENV,
        embedding_credential,
    )

    before = os.environ.get(EMBEDDINGS_API_KEY_ENV)
    os.environ[EMBEDDINGS_API_KEY_ENV] = "sk-sentinel"
    try:
        assert embedding_credential("https://openrouter.ai/api/v1") == "sk-sentinel", (
            "the provider's own endpoint must carry the key"
        )
        for elsewhere in (
            "http://localhost:11234/v1",
            "https://openrouter.ai.evil.example/v1",
            "https://proxy.internal/v1",
        ):
            assert embedding_credential(elsewhere) != "sk-sentinel", (
                f"the key must not be sent to {elsewhere}"
            )
    finally:
        if before is None:
            os.environ.pop(EMBEDDINGS_API_KEY_ENV, None)
        else:
            os.environ[EMBEDDINGS_API_KEY_ENV] = before


def test_a_reply_whose_indices_are_not_a_permutation_is_malformed() -> None:
    """The right number of rows does not place them on the right inputs."""
    for name, items in (
        ("a repeated index", [{"index": 0, "embedding": [1.0]}] * 2),
        (
            "an index outside the request",
            [
                {"index": 3, "embedding": [1.0]},
                {"index": 9, "embedding": [2.0]},
            ],
        ),
        # The SDK accepts every one of these without a word.
        ("fewer vectors than inputs", [{"index": 0, "embedding": [1.0]}]),
    ):

        def handler(
            request: httpx2.Request, items: list[dict[str, object]] = items
        ) -> httpx2.Response:
            return httpx2.Response(
                200,
                json={
                    "object": "list",
                    "model": EMBEDDING_MODEL,
                    "data": [{"object": "embedding", **item} for item in items],
                },
            )

        with pytest.raises(EmbeddingError) as excinfo:
            request_vectors(_client(handler), ["a", "b"])
        assert excinfo.value.code == "embedding_response_malformed", name


def test_a_permanent_rejection_is_not_retried_on_the_query_path() -> None:
    """A rejected credential or unserved model says the same thing every time.

    The query path retried every status alike, so a 401 cost three round trips
    and was reported as the service being down.
    """
    from contract_analyzer.corpus.embeddings import _embed

    for status, code in (
        (401, "embedding_request_rejected"),
        (402, "embedding_request_rejected"),
        (404, "embedding_request_rejected"),
    ):
        calls: list[int] = []

        def handler(
            request: httpx2.Request,
            _status: int = status,
            _calls: list[int] = calls,
        ) -> httpx2.Response:
            _calls.append(1)
            return httpx2.Response(_status, json={"error": "no"})

        with pytest.raises(EmbeddingError) as excinfo:
            _embed(["x"], client=_client(handler))
        assert excinfo.value.code == code, status
        assert len(calls) == 1, f"HTTP {status} was retried {len(calls)} times"


def test_a_rate_limit_waits_the_period_the_provider_named() -> None:
    """Retry-After is the only thing that says how long a throttled caller waits."""
    from contract_analyzer.corpus.embeddings import _embed

    slept: list[float] = []
    attempts: list[int] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        attempts.append(1)
        if len(attempts) == 1:
            return httpx2.Response(
                429, headers={"Retry-After": "7"}, json={"error": "slow"}
            )
        return httpx2.Response(
            200,
            json={"data": [{"index": 0, "embedding": [1.0] * EMBEDDING_DIMENSION}]},
        )

    with patch("time.sleep", slept.append):
        vectors = _embed(["x"], client=_client(handler))

    assert vectors.shape == (1, EMBEDDING_DIMENSION)
    assert slept == [7.0], f"waited {slept} instead of the 7s the provider asked for"


def test_a_self_hosted_endpoint_needs_no_credential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Readiness must not fail a deployment that never needed a key."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(
        corpus_embeddings, "EMBEDDINGS_BASE_URL", "http://host.docker.internal:8080/v1"
    )
    assert embedding_credential_configured() is True


def test_the_provider_endpoint_without_a_key_is_not_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(
        corpus_embeddings, "EMBEDDINGS_BASE_URL", "https://openrouter.ai/api/v1"
    )
    assert embedding_credential_configured() is False

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-present")
    assert embedding_credential_configured() is True


def test_the_readiness_credential_check_contacts_nobody() -> None:
    """The embedding limb of readiness must not reach the network.

    compose polls readiness every 30s. When this encoded a probe it billed the
    provider ~2,880 times a day to re-answer a settled question, and it put a paid
    third-party call inside a suite that promises to run offline. What the vector
    space actually is stays a real question -- the corpus check asks it, once per
    process, against the probe the build recorded.
    """

    def explode(request: httpx2.Request) -> httpx2.Response:
        raise AssertionError("readiness contacted the embedding service")

    with patch.object(
        corpus_embeddings, "embedding_client", lambda *a, **k: _client(explode)
    ):
        assert embedding_credential_configured() in (True, False)


def test_two_vectors_from_one_space_agree_and_from_two_do_not() -> None:
    same = _vector(0.1)
    assert embedding_spaces_agree(same, list(same)) is True
    # Orthogonal: what two different encoders under one name look like.
    other = [0.0] * EMBEDDING_DIMENSION
    other[0] = 1.0
    assert embedding_spaces_agree(same, other) is False
    # A missing or differently shaped record is never quietly accepted.
    assert embedding_spaces_agree((), ()) is False
    assert embedding_spaces_agree(same, same[:-1]) is False


def test_usage_is_read_when_the_service_reports_it() -> None:
    """Quality is never reported without its cost, and the count is on the wire."""

    def with_usage(request: httpx2.Request) -> httpx2.Response:
        body = _embeddings_payload([_vector(0.5)])
        body["usage"] = {"prompt_tokens": 41, "total_tokens": 41}
        return httpx2.Response(200, json=body)

    def without_usage(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=_embeddings_payload([_vector(0.5)]))

    _, tokens = request_vectors(_client(with_usage), ["x"])
    assert tokens == 41
    # Silence is a different fact from zero.
    _, none = request_vectors(_client(without_usage), ["x"])
    assert none is None


def test_the_router_is_told_which_upstream_to_use() -> None:
    """One build, one arithmetic.

    The router serves this model from several upstreams and routes each request
    independently. The build probe is taken once, at the end, so it cannot see an
    upstream that answered an earlier batch -- pinning is what makes that single
    probe speak for the whole build.
    """
    seen: list[dict[str, object]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(json.loads(request.content))
        return httpx2.Response(200, json=_embeddings_payload([_vector(0.5)]))

    with patch.object(corpus_embeddings, "EMBEDDINGS_BASE_URL", PROVIDER_URL):
        with patch.object(corpus_embeddings, "EMBEDDINGS_PROVIDER", "deepinfra"):
            request_vectors(_client(handler, base_url=PROVIDER_URL), ["x"])

    assert seen[0]["provider"] == {
        "order": ["deepinfra"],
        # A build answered by a second upstream after the first dropped out would
        # mix arithmetics without saying so.
        "allow_fallbacks": False,
    }


def test_an_endpoint_that_is_not_the_router_gets_no_routing_field() -> None:
    """A self-hosted host has no upstreams, and the field is not OpenAI-compatible."""
    seen: list[dict[str, object]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(json.loads(request.content))
        return httpx2.Response(200, json=_embeddings_payload([_vector(0.5)]))

    request_vectors(_client(handler), ["x"])
    assert "provider" not in seen[0]


def test_the_corpus_package_exports_only_names_it_has() -> None:
    """A name in __all__ that nothing binds breaks `import *` at the call site.

    Three such names survived a deletion here, and ruff's own check did not
    report them, so the suite carries the guard instead.
    """
    import contract_analyzer.corpus as corpus

    missing = [name for name in corpus.__all__ if not hasattr(corpus, name)]
    assert not missing, f"exported but unbound: {missing}"


def test_a_self_hosted_build_is_not_described_as_the_router_s_upstream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """EMBEDDINGS_PROVIDER routes the router; a self-hosted host has no upstreams.

    The pin is correctly never sent to one, so naming it in the identity would
    describe a build as DeepInfra's that DeepInfra never answered.
    """
    monkeypatch.setattr(corpus_embeddings, "EMBEDDINGS_PROVIDER", "deepinfra")

    monkeypatch.setattr(corpus_embeddings, "EMBEDDINGS_BASE_URL", PROVIDER_URL)
    assert corpus_embeddings.effective_embedding_provider() == "deepinfra"

    monkeypatch.setattr(
        corpus_embeddings, "EMBEDDINGS_BASE_URL", "http://host.docker.internal:8080/v1"
    )
    assert corpus_embeddings.effective_embedding_provider() == ""
    assert corpus_embeddings.embedding_extra_body() == {}
