"""Embedding transport and vector-space identity using the OpenAI SDK."""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from itertools import batched
from urllib.parse import urlsplit

import numpy as np
from openai import (
    APIConnectionError,
    APIStatusError,
    AsyncOpenAI,
    OpenAI,
    RateLimitError,
)
from openai.types import CreateEmbeddingResponse

from contract_analyzer.corpus.errors import EmbeddingError

# A hosted provider over the OpenAI-compatible /v1/embeddings route, so no model
# weights enter the image. The host is deployment configuration; the model and its
# dimension are not, because the dimension guard only means something while it
# names one encoder.
EMBEDDINGS_BASE_URL = os.environ.get(
    "EMBEDDINGS_BASE_URL", "https://openrouter.ai/api/v1"
)
# Qwen3-Embedding-8B, full precision. Chosen because its family leads MTEB's
# multilingual board, and the corpus is Polish.
EMBEDDING_MODEL = "qwen/qwen3-embedding-8b"
EMBEDDING_DIMENSION = 4096
# Which upstream the router must use. It routes each request independently unless
# told otherwise, so one build could be answered by several (measured 2026-09-06:
# 0.9999 cosine apart). The build probe is taken once and cannot see an upstream
# that answered an earlier batch, so pinning is what makes it speak for the whole
# build. Empty restores the router's own load balancing.
EMBEDDINGS_PROVIDER = os.environ.get("EMBEDDINGS_PROVIDER", "deepinfra")
EMBEDDINGS_API_KEY_ENV = "OPENROUTER_API_KEY"
# The credential travels only to the provider it belongs to. EMBEDDINGS_BASE_URL
# may name a self-hosted endpoint, a proxy or a typo, and attaching the key to
# whatever it names would hand the secret to that host.
EMBEDDINGS_API_KEY_HOST = "openrouter.ai"
# What every other host gets. Not a secret, and not None: None makes the SDK read
# OPENAI_API_KEY from the environment and send whatever it finds anywhere.
_NO_CREDENTIAL = "not-used"

# Qwen3-Embedding is asymmetric by instruction: a passage is sent bare, a query
# carries a task line. The wording is chosen by argument, not measurement.
EMBEDDING_QUERY_INSTRUCTION = (
    "Given a question about a contract, retrieve provisions of Polish law "
    "that bear on it"
)
EMBEDDING_QUERY_PREFIX = f"Instruct: {EMBEDDING_QUERY_INSTRUCTION}\nQuery: "

# How many inputs one request carries. A hosted provider answers or refuses a
# whole request, so nothing depends on what the inputs weigh relative to each
# other; the cap exists only so a build does not send the corpus in one body.
EMBEDDING_REQUEST_MAX_INPUTS = 128

# Short, and Polish, so the probe exercises the same tokenizer path the corpus does.
EMBEDDING_PROBE_TEXT = "Umowa."
# How close two probe vectors must be to count as the same space. Measured
# 2026-09-06: three upstreams of this model came back 0.9999 apart, two hosts of
# the earlier local model 0.53 apart. The threshold separates those two.
EMBEDDING_SPACE_COSINE_MIN = 0.99

# Slack for a cold model load, not a quality boundary. Measured 2026-09-02: 8
# passages of ~170 tokens took 1.66s, 64 took 11.51s.
EMBEDDING_TIMEOUT_SECONDS = 300.0
# One retry policy, two budgets: a search has someone watching it, a build is
# unattended and resumes from Qdrant, so it can ride out a blip.
EMBEDDING_QUERY_ATTEMPTS = 3
EMBEDDING_BUILD_ATTEMPTS = 6

logger = logging.getLogger("contract_analyzer.corpus")


def _is_provider_host(base_url: str | None = None) -> bool:
    """Whether this endpoint is the provider the credential belongs to."""
    host = (urlsplit(base_url or EMBEDDINGS_BASE_URL).hostname or "").casefold()
    return host == EMBEDDINGS_API_KEY_HOST or host.endswith(
        f".{EMBEDDINGS_API_KEY_HOST}"
    )


def effective_embedding_provider(base_url: str | None = None) -> str:
    """The upstream that will actually answer, which is not always the configured one.

    EMBEDDINGS_PROVIDER routes the router. A self-hosted endpoint has no upstreams
    and is never sent the pin, so naming one in its identity would describe a build
    as DeepInfra's that DeepInfra never touched.
    """
    return EMBEDDINGS_PROVIDER if _is_provider_host(base_url) else ""


def embedding_credential(base_url: str | None = None) -> str:
    """The credential this endpoint gets: the real one, or a non-secret stand-in.

    Only the provider the key belongs to is given it. Any other host -- a
    self-hosted endpoint, a proxy, a typo -- gets the stand-in, so a misconfigured
    base URL cannot walk off with the secret.
    """
    if not _is_provider_host(base_url):
        return _NO_CREDENTIAL
    return os.environ.get(EMBEDDINGS_API_KEY_ENV, "").strip() or _NO_CREDENTIAL


def embedding_credential_configured() -> bool:
    """Whether this endpoint has the credential it needs, if it needs one.

    All readiness can establish without paying for a forward pass on every poll.
    Whether the service answers in the corpus's vector space is a separate and
    costlier question, asked once per process by ``QdrantCorpusIndex.verify``.
    """
    return not _is_provider_host() or embedding_credential() != _NO_CREDENTIAL


def _client_options(max_attempts: int) -> dict[str, object]:
    """SDK options for one client. The header is set explicitly, not left to the SDK.

    Passing it is what stops OPENAI_CUSTOM_HEADERS in the process environment from
    replacing our credential with an ambient one.
    """
    key = embedding_credential()
    logger.debug(
        "Embedding client: %s, model %s, upstream %r, %d attempts, credential %s",
        EMBEDDINGS_BASE_URL,
        EMBEDDING_MODEL,
        effective_embedding_provider(),
        max_attempts,
        "configured" if key != _NO_CREDENTIAL else "none",
    )
    return {
        "api_key": key,
        "base_url": EMBEDDINGS_BASE_URL,
        "timeout": EMBEDDING_TIMEOUT_SECONDS,
        "max_retries": max_attempts - 1,
        "default_headers": {"Authorization": f"Bearer {key}"},
    }


def embedding_client(max_attempts: int = EMBEDDING_QUERY_ATTEMPTS) -> OpenAI:
    """A client for the query path, which has someone waiting on it."""
    return OpenAI(**_client_options(max_attempts))  # type: ignore[arg-type]


def async_embedding_client(
    max_attempts: int = EMBEDDING_BUILD_ATTEMPTS,
) -> AsyncOpenAI:
    """A client for the build path, which is unattended and can wait longer."""
    return AsyncOpenAI(**_client_options(max_attempts))  # type: ignore[arg-type]


def embedding_error_for(exc: Exception) -> EmbeddingError:
    """Name one SDK failure the same way on the query path and the build path.

    The split that decides anything is two-way -- worth re-sending or not -- and
    the SDK has already acted on it by the time an exception escapes. A rate limit
    keeps its own name because it is the one failure an operator reads
    differently, not because the code branches on it.
    """
    if isinstance(exc, RateLimitError):
        return EmbeddingError(
            "embedding_rate_limited", f"embedding_rate_limited: {exc}"
        )
    if isinstance(exc, APIStatusError):
        if 400 <= exc.status_code < 500 and exc.status_code not in (408, 409):
            return EmbeddingError(
                "embedding_request_rejected",
                f"embedding_request_rejected: HTTP {exc.status_code}: {exc}",
            )
        return EmbeddingError(
            "embedding_service_unavailable", f"embedding_service_unavailable: {exc}"
        )
    if isinstance(exc, APIConnectionError):
        return EmbeddingError(
            "embedding_service_unavailable", f"embedding_service_unavailable: {exc}"
        )
    return EmbeddingError(
        "embedding_response_invalid", f"embedding_response_invalid: {exc}"
    )


def embedding_extra_body(base_url: str | None = None) -> dict[str, object]:
    """Provider routing, for the router that has any. Any other endpoint gets none.

    ``allow_fallbacks`` is refused deliberately: a build answered by a second
    upstream after the first dropped out would mix arithmetics without saying so,
    and the cost of refusing is a failed build that can simply be re-run, which
    delta seeding makes cheap.
    """
    provider = effective_embedding_provider(base_url)
    if not provider:
        return {}
    return {"provider": {"order": [provider], "allow_fallbacks": False}}


def embed_query_text(query: str) -> str:
    """The exact string a search query is embedded as."""
    return f"{EMBEDDING_QUERY_PREFIX}{query}"


def vectors_from_response(
    response: CreateEmbeddingResponse, expected: int
) -> tuple[list[list[float]], int | None]:
    """Place ``expected`` vectors on their inputs, and read what they cost.

    The SDK does none of this: ``data`` comes back in wire order, a repeated index
    is accepted, and so is a reply carrying fewer vectors than there were inputs.
    Indexing by ``index`` does the placing, the counting and the duplicate check
    at once.
    """
    try:
        by_index = {int(item.index): item.embedding for item in response.data}
    except (AttributeError, TypeError, ValueError) as exc:
        raise EmbeddingError(
            "embedding_response_malformed", f"embedding_response_malformed: {exc}"
        ) from exc
    if len(by_index) != expected or any(i not in by_index for i in range(expected)):
        raise EmbeddingError(
            "embedding_response_malformed",
            f"embedding_response_malformed: expected {expected} vectors on indices "
            f"0..{expected - 1}, got {sorted(by_index)}",
        )
    try:
        rows = [[float(value) for value in by_index[i]] for i in range(expected)]
    except (TypeError, ValueError) as exc:
        raise EmbeddingError(
            "embedding_response_malformed", f"embedding_response_malformed: {exc}"
        ) from exc
    usage = response.usage.prompt_tokens if response.usage else None
    return rows, usage


def request_vectors(
    client: OpenAI, texts: list[str]
) -> tuple[list[list[float]], int | None]:
    """One request for these exact texts, in the order they were given.

    ``encoding_format`` is explicit because the SDK otherwise asks for base64, a
    wire format some compatible hosts do not implement.
    """
    try:
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=texts,
            encoding_format="float",
            extra_body=embedding_extra_body(str(client.base_url)),
        )
    except EmbeddingError:
        raise
    except Exception as exc:
        raise embedding_error_for(exc) from exc
    return vectors_from_response(response, len(texts))


async def arequest_vectors(
    client: AsyncOpenAI, texts: list[str]
) -> tuple[list[list[float]], int | None]:
    """``request_vectors`` for the build path. Same contract, same guards."""
    try:
        response = await client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=texts,
            encoding_format="float",
            extra_body=embedding_extra_body(str(client.base_url)),
        )
    except EmbeddingError:
        raise
    except Exception as exc:
        raise embedding_error_for(exc) from exc
    return vectors_from_response(response, len(texts))


def checked_unit_vectors(rows: Sequence[Sequence[float]]) -> np.ndarray:
    """Validate a batch of vectors and return it unit length.

    Both the corpus being written and the query matched against it pass through
    here. A zero vector scores nothing against every query -- a unit present but
    unfindable -- and fusion scores by dot product, so length is not decoration.
    """
    array = np.asarray(rows, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != EMBEDDING_DIMENSION:
        raise EmbeddingError(
            "embedding_dimension_mismatch",
            (
                f"embedding_dimension_mismatch: expected {EMBEDDING_DIMENSION} "
                f"columns, got {array.shape}"
            ),
        )
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if not bool(np.all(norms > 0.0)):
        raise EmbeddingError(
            "embedding_zero_vector",
            "embedding_zero_vector: the service returned a vector of length zero",
        )
    return np.asarray(array / norms, dtype=np.float32)


def _embed(texts: list[str], client: OpenAI | None = None) -> np.ndarray:
    """Encode ``texts`` on the configured embedding service.

    Every failure is loud: a short or malformed payload, a wrong width, or a zero
    vector all raise rather than yield something the retrieval limb would silently
    score.
    """
    rows: list[list[float]] = []
    opened = client if client is not None else embedding_client()
    try:
        for group in batched(texts, EMBEDDING_REQUEST_MAX_INPUTS):
            vectors, _ = request_vectors(opened, list(group))
            rows.extend(vectors)
    finally:
        if client is None:
            opened.close()
    return checked_unit_vectors(rows)


def embedding_space_probe(client: OpenAI | None = None) -> tuple[float, ...]:
    """The unit vector this service encodes the probe text as.

    The corpus's vector-space identity, and the only honest one: a model name is
    what was asked for, this is what came back. One snapshot identifier has
    already stood for two spaces 0.53 cosine apart under a single model name.
    """
    probe = tuple(_encode([EMBEDDING_PROBE_TEXT], client=client)[0].tolist())
    logger.info(
        "Embedding space: model %s at %s, upstream %r",
        EMBEDDING_MODEL,
        EMBEDDINGS_BASE_URL,
        effective_embedding_provider(),
    )
    return probe


def embedding_spaces_agree(left: Sequence[float], right: Sequence[float]) -> bool:
    """Whether two probe vectors came out of the same embedding space."""
    if not len(left) or len(left) != len(right):
        return False
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    scale = float(np.linalg.norm(a) * np.linalg.norm(b))
    if scale == 0.0:
        return False
    return bool(float(np.dot(a, b)) / scale >= EMBEDDING_SPACE_COSINE_MIN)


def _encode(texts: list[str], client: OpenAI | None = None) -> np.ndarray:
    """Encode through the package name: ``corpus._embed`` is the test seam."""
    from contract_analyzer import corpus as corpus_module

    return corpus_module._embed(texts, client=client)
