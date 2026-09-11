from __future__ import annotations

import hashlib
import json
import re
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path

import httpx
import numpy as np
import pytest
from qdrant_client import QdrantClient, models

from contract_analyzer import corpus as corpus_module
from contract_analyzer.corpus import (
    CORPUS_ALIAS,
    CORPUS_SNAPSHOTS_COLLECTION,
    EMBEDDING_DIMENSION,
    EMBEDDINGS_API_KEY_ENV,
    ActCurrency,
    CorpusSnapshot,
    LegalUnit,
    QdrantCorpusIndex,
    corpus_collection_name,
    corpus_snapshot_id,
    embedding_space_probe,
    snapshot_record_id,
    unit_digest_of,
)
from seeder.ingest import fetch_manifest_units

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "eli"
ELI_BASE = "https://api.sejm.gov.pl/eli/acts"

TEST_MANIFEST = {
    "version": 2,
    "corpus_target_date": "2026-08-30",
    "acts": [
        {
            "publisher": "DU",
            "year": 2026,
            "position": 795,
            "source_format": "pdf",
            "base_act": "DU/1964/93",
            "legal_status_date": "2026-05-19",
            "amendments_not_carried": [],
            "repeated_articles": [],
        },
        {
            "publisher": "DU",
            "year": 2023,
            "position": 725,
            "source_format": "html",
            "base_act": "DU/2001/733",
            "legal_status_date": "2023-03-09",
            "amendments_not_carried": [],
            "repeated_articles": [],
        },
    ],
}


@dataclass
class FakeEli:
    overrides: dict[str, tuple[int, bytes]] = field(default_factory=dict)
    _client: httpx.Client | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._client = httpx.Client(
            transport=httpx.MockTransport(self._handle),
            base_url=ELI_BASE,
        )

    @property
    def client(self) -> httpx.Client:
        assert self._client is not None
        return self._client

    def respond(self, *, url: str, status: int, body: bytes) -> None:
        self.overrides[url] = (status, body)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url in self.overrides:
            status, body = self.overrides[url]
            return httpx.Response(status, content=body, request=request)
        status, body = _route(url)
        return httpx.Response(status, content=body, request=request)


def _fixture(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def _route(url: str) -> tuple[int, bytes]:
    if url.endswith("/DU/1964/93"):
        return 200, _fixture("du_1964_93_metadata.json")
    if url.endswith("/DU/2001/733"):
        return 200, _fixture("du_2001_733_metadata.json")
    if url.endswith("/DU/2023/725") and not url.endswith(("/struct", "/text.pdf")):
        return 200, _fixture("du_2023_725_metadata.json")
    if url.endswith("/DU/2023/725/struct"):
        return 200, _fixture("du_2023_725_struct.json")
    if url.endswith("/DU/2023/725/text.html/arti=11"):
        return 200, _fixture("du_2023_725_art11.html")
    if url.endswith("/DU/2023/725/text.html/arti=19a"):
        return 200, _fixture("du_2023_725_art19a.html")
    if url.endswith("/DU/2023/725/text.html/arti=41(1)"):
        return 200, _fixture("du_2023_725_art41_1.html")
    if url.endswith("/DU/2026/795") and not url.endswith(("/struct", "/text.pdf")):
        return 200, _fixture("du_2026_795_metadata.json")
    if url.endswith("/DU/2026/795/text.pdf"):
        return 200, _fixture("du_2026_795_text.pdf")
    return 404, b""


@pytest.fixture
def fake_eli() -> FakeEli:
    return FakeEli()


@pytest.fixture
def manifest_path(tmp_path: Path) -> Path:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(TEST_MANIFEST), encoding="utf-8")
    return path


def publish_corpus(
    units: list[LegalUnit],
    manifest_hash: str = "test-manifest",
    act_currency: tuple[ActCurrency, ...] = (),
) -> QdrantCorpusIndex:
    """Publish these units as an in-memory Qdrant corpus, the way the seeder does.

    The payload and vector shapes come from the seeder itself, so the suite
    exercises the corpus the deployment serves rather than one of its own.

    ``act_currency`` matters more than it looks: an empty one reads as
    ``unrecorded``, the class a measured run refuses, so a fixture that left it
    empty would never touch the ``as_declared`` path production gates on.
    """
    from seeder.pipeline import _unit_payload, _unit_vectors, ensure_qdrant_collection

    client = QdrantClient(":memory:")
    collection = corpus_collection_name(manifest_hash)
    ensure_qdrant_collection(client, collection)
    dense = corpus_module._embed([unit.text for unit in units])
    client.upsert(
        collection_name=collection,
        points=[
            models.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_DNS, unit.id)),
                vector=_unit_vectors(unit, [float(v) for v in row]),
                payload=_unit_payload(unit),
            )
            for unit, row in zip(units, dense, strict=True)
        ],
        wait=True,
    )
    digest = unit_digest_of((unit.locator, unit.content_hash) for unit in units)
    snapshot = CorpusSnapshot(
        id=corpus_snapshot_id(manifest_hash, digest),
        built_at=datetime(2026, 1, 1, tzinfo=UTC),
        manifest_hash=manifest_hash,
        model_name=corpus_module.EMBEDDING_MODEL,
        embedding_space_probe=embedding_space_probe(),
        file_hashes={},
        unit_count=len(units),
        act_identifiers=tuple(sorted({unit.act_identifier for unit in units})),
        corpus_target_date=date(2026, 8, 30),
        act_currency=act_currency,
        unit_digest=digest,
    )
    client.create_collection(
        collection_name=CORPUS_SNAPSHOTS_COLLECTION,
        vectors_config=models.VectorParams(size=1, distance=models.Distance.COSINE),
    )
    client.upsert(
        collection_name=CORPUS_SNAPSHOTS_COLLECTION,
        points=[
            models.PointStruct(
                id=snapshot_record_id(collection),
                vector=[0.0],
                payload={
                    "collection": collection,
                    "snapshot": snapshot.model_dump_json(),
                },
            )
        ],
        wait=True,
    )
    client.update_collection_aliases(
        change_aliases_operations=[
            models.CreateAliasOperation(
                create_alias=models.CreateAlias(
                    collection_name=collection, alias_name=CORPUS_ALIAS
                )
            )
        ]
    )
    return QdrantCorpusIndex(client=client, collection=CORPUS_ALIAS)


@pytest.fixture
def corpus_ingestion(
    fake_eli: FakeEli, manifest_path: Path
) -> tuple[list[LegalUnit], tuple[ActCurrency, ...]]:
    """The units the manifest names, fetched and validated as the seeder does.

    Segmentation is the seeder's: both paths call deduped_struct_article_paths,
    which emits articles and nothing under them. What this path does not do is
    probe an act's HTML and fall back to its PDF -- there is nothing to fall back
    from behind a fixture client -- so it reads every act as declared and says so
    in the currency rather than leaving it blank.
    """
    _, _, units, act_currency = fetch_manifest_units(manifest_path, fake_eli.client)
    return units, act_currency


@pytest.fixture
def corpus_units(
    corpus_ingestion: tuple[list[LegalUnit], tuple[ActCurrency, ...]],
) -> list[LegalUnit]:
    return corpus_ingestion[0]


@pytest.fixture
def built_corpus(
    corpus_ingestion: tuple[list[LegalUnit], tuple[ActCurrency, ...]],
) -> QdrantCorpusIndex:
    units, act_currency = corpus_ingestion
    return publish_corpus(units, act_currency=act_currency)


# The suite runs with no network and no inference server. The seam is
# corpus._embed, a late attribute lookup through the package rather than a flag,
# so a monkeypatch lands where production reads it. The real function's HTTP
# behaviour is covered over a mock transport in test_embedding_client.py.
_LIVE_EMBED = corpus_module._embed
_TOKEN = re.compile(r"\w+", re.UNICODE)


def _hashing_embed(texts: list[str], client: object | None = None) -> np.ndarray:
    """Signed feature hashing, not a language model.

    Texts sharing tokens get similar vectors and unrelated texts do not, which is
    the property the fusion, ranking and guard tests rest on. It says nothing
    about how the real encoder ranks Polish legal text, and no test may read it
    as evidence of that.
    """
    rows = np.zeros((len(texts), EMBEDDING_DIMENSION), dtype=np.float32)
    for row, text in zip(rows, texts, strict=True):
        for token in _TOKEN.findall(text.lower()):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "big") % EMBEDDING_DIMENSION
            row[bucket] += 1.0 if digest[4] & 1 else -1.0
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    # A text with no word characters hashes to nothing. The real service raises on
    # a zero vector; that is a property of the service, not of this stand-in.
    norms[norms == 0.0] = 1.0
    return np.asarray(rows / norms, dtype=np.float32)


@pytest.fixture(autouse=True)
def deterministic_embedding_credential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Readiness reads the provider key from the environment, so the suite must set it.

    Without this the result depends on whether the developer happens to have a
    real key exported: CI has none and reported a missing credential where a
    machine with one reported the corpus fault the test was written for. A test
    that needs the key absent clears it itself.
    """
    monkeypatch.setenv(EMBEDDINGS_API_KEY_ENV, "test-credential-not-a-real-key")


@pytest.fixture(scope="session", autouse=True)
def offline_embeddings() -> Iterator[None]:
    corpus_module._embed = _hashing_embed
    try:
        yield
    finally:
        corpus_module._embed = _LIVE_EMBED


@pytest.fixture
def live_embed() -> Callable[..., np.ndarray]:
    """The real HTTP implementation, for the tests that drive it over a transport."""
    return _LIVE_EMBED
