from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field

DEFAULT_MODEL_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL_NAME = "meta/muse-spark-1.3-contributor"


def _env_flag(name: str) -> bool:
    """A boolean environment variable; an unknown spelling is refused."""
    raw = os.environ.get(name, "false").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"", "0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true or false, got {raw!r}")


# The SDK merges configured headers over its own by exact key, so any of these
# would displace what the client set: the credential, the body's encoding, and
# the two that decide which account a call is billed to.
_RESERVED_HEADERS = frozenset(
    {"authorization", "content-type", "openai-organization", "openai-project"}
)


def _refuse_reserved_headers(headers: Mapping[str, str]) -> None:
    """Refuse a header that would displace one the client sets for itself."""
    found = sorted({k.strip().lower() for k in headers} & _RESERVED_HEADERS)
    if found:
        raise ValueError(f"MODEL_EXTRA_HEADERS must not set {', '.join(found)}")


def _env_headers(name: str) -> Mapping[str, str]:
    """Headers a provider requires beyond the OpenAI interface, as a JSON object.

    Some endpoints refuse a request that is otherwise well formed: the gateway
    this project calls by default rejects one without a routing header. Keeping
    them in configuration means a provider's own requirement never becomes a
    provider name in the transport.
    """
    raw = os.environ.get(name, "").strip()
    if not raw:
        return MappingProxyType({})
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"{name} must be a JSON object: {error}") from error
    if not isinstance(parsed, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in parsed.items()
    ):
        raise ValueError(f"{name} must be a JSON object of string to string")
    return MappingProxyType(dict(parsed))


@dataclass(frozen=True)
class Settings:
    """Runtime configuration, including the model endpoint a run calls."""

    model_api_key: str | None = field(default=None, repr=False)
    model_base_url: str = DEFAULT_MODEL_BASE_URL
    model_name: str = DEFAULT_MODEL_NAME
    model_extra_headers: Mapping[str, str] = field(
        default=MappingProxyType({}), repr=False
    )
    database_url: str | None = None
    qdrant_url: str | None = None
    qdrant_collection: str = "legal_units"
    corpus_snapshot_id: str | None = None
    evaluation_batch_open: bool = False

    @classmethod
    def from_env(cls) -> Settings:
        snapshot_id = os.environ.get("CORPUS_SNAPSHOT_ID") or None
        batch_open = _env_flag("EVALUATION_BATCH_OPEN")
        return cls(
            model_api_key=os.environ.get("MODEL_API_KEY"),
            model_base_url=os.environ.get("MODEL_BASE_URL", DEFAULT_MODEL_BASE_URL),
            model_name=os.environ.get("MODEL_NAME", DEFAULT_MODEL_NAME),
            model_extra_headers=_env_headers("MODEL_EXTRA_HEADERS"),
            database_url=os.environ.get("DATABASE_URL"),
            qdrant_url=os.environ.get("QDRANT_URL"),
            qdrant_collection=os.environ.get("QDRANT_COLLECTION", "legal_units"),
            corpus_snapshot_id=snapshot_id,
            evaluation_batch_open=batch_open,
        )

    def __post_init__(self) -> None:
        """Own the headers, so a caller's later edit cannot reach this run."""
        _refuse_reserved_headers(self.model_extra_headers)
        object.__setattr__(
            self,
            "model_extra_headers",
            MappingProxyType(dict(self.model_extra_headers)),
        )

    @property
    def model_endpoint(self) -> str:
        """The host that answers a model call, which the model name does not say.

        The protocol treats the provider as a parity dimension: two endpoints
        serving one identifier do not serve it identically, so a run has to
        record which one answered. The host alone, never the path and never the
        credential -- a path can carry a routing token and a credential in a run
        record is a leak.
        """
        return (urlsplit(self.model_base_url).hostname or "").casefold()

    def require_live(self) -> None:
        if not self.model_api_key:
            raise ValueError("MODEL_API_KEY is required for a live run")
        if not self.model_base_url:
            raise ValueError("MODEL_BASE_URL is required for a live run")
        if not self.model_name:
            raise ValueError("MODEL_NAME is required for a live run")


class RunConfig(BaseModel):
    """Runtime thresholds chosen by argument, not measurement, except where a
    field's own description records a measurement and its date.
    """

    model_config = ConfigDict(frozen=True)

    structural_min: int = 3
    structural_monotonicity_share: float = 0.8
    structural_coverage_share: float = 0.6
    structural_max_tail_share: float = Field(
        default=0.5,
        description=(
            "Largest share of the document allowed after the last marker. Chosen by "
            "argument: a marker run that stops before the bulk of the text describes a "
            "preamble, not the document."
        ),
    )
    window_sentences: int = 3
    window_overlap: int = 1
    pdf_min_native_chars_per_page: int = Field(
        default=1,
        description="Minimum text characters for a page to count as text-bearing.",
    )
    pdf_native_coverage_share: float = Field(
        default=1.0,
        description=(
            "Share of text-bearing pages needed to read natively; 1.0 because a "
            "native read of a page with no text layer yields nothing silently, so "
            "one such page sends the whole document to OCR."
        ),
    )
    pdf_ocr_render_scale: float = Field(
        default=200 / 72,
        description="Rasterisation scale when OCR is selected.",
    )
    max_input_bytes: int = Field(
        default=25 * 1024 * 1024,
        description=(
            "Fail-fast ceiling by argument, not measurement: one operative "
            "document, not an archive. At the scan fixture's ~376 KB per page "
            "the byte limit binds first, near 70 pages."
        ),
    )
    max_pdf_pages: int = Field(
        default=200,
        description=(
            "Fail-fast ceiling by argument, not measurement: one operative "
            "document, not an archive. At the scan fixture's ~376 KB per page "
            "the byte limit binds first, near 70 pages."
        ),
    )
    blank_page_ink_share: float = Field(
        default=0.0001,
        description=(
            "Dark-pixel share on a ~400 px downsample, so scan noise averages "
            "away; a blank render measures 0.0 and a lone glyph reads as blank "
            "by design."
        ),
    )
    ocr_language: str = Field(
        default="pol",
        description="Language pack passed to Tesseract.",
    )
    ocr_timeout_seconds: int = Field(
        default=120,
        description="Per-page ceiling on the Tesseract subprocess.",
    )
    conversion_timeout_seconds: int = Field(
        default=120,
        description="Ceiling on the LibreOffice conversion subprocess.",
    )
    model_timeout_seconds: int = Field(
        default=180,
        description=(
            "Read timeout of one model request. Measured 2026-08-28 on the live "
            "endpoint: median 22.4 s, longest success 79.1 s; 180 s is 2.3x that, "
            "and three attempts (540 s) stay inside the 900 s wall budget."
        ),
    )
    content_ttl_seconds: int = Field(
        default=3600,
        description="Ephemeral document text retention TTL and sweep interval.",
    )
    content_sweep_interval_seconds: int = Field(
        default=60,
        description="Ephemeral document text retention TTL and sweep interval.",
    )
