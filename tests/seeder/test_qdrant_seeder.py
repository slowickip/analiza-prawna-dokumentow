from __future__ import annotations

import asyncio
import json
import uuid
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx2
import numpy as np
import pytest
from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient, QdrantClient, models

from contract_analyzer import corpus as corpus_module
from contract_analyzer.corpus import (
    CORPUS_ALIAS,
    CORPUS_SNAPSHOTS_COLLECTION,
    DENSE_VECTOR_NAME,
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL,
    EMBEDDING_PROBE_TEXT,
    SPARSE_VECTOR_NAME,
    CorpusBuildError,
    CorpusIntegrityError,
    CorpusSnapshot,
    ForceScope,
    ForceState,
    ForceValue,
    LegalUnit,
    QdrantCorpusIndex,
    corpus_collection_name,
    embedding_space_probe,
    embedding_spaces_agree,
    sparse_document_vector,
)
from seeder import (
    ensure_qdrant_collection,
    ensure_qdrant_collection_async,
    pipeline,
    seed_qdrant_async,
)
from seeder.pipeline import _push_worker, _verify_and_publish


def _make_unit(idx: int, hash_suffix: str = "v1", text: str | None = None) -> LegalUnit:
    return LegalUnit(
        id=f"act-1/art-{idx}",
        locator=f"Dz.U. 2020 poz. 1 art. {idx}",
        act_identifier="DU-2020-1",
        article_identifier=f"art. {idx}",
        text=text if text is not None else f"Treść artykułu {idx}",
        content_hash=f"hash-{idx}-{hash_suffix}",
        act_force=ForceState(
            value=ForceValue.IN_FORCE,
            scope=ForceScope.ACT,
            snapshot_date=date(2026, 1, 1),
            source_locator="DU-2020-1",
        ),
        provision_force=ForceState(
            value=ForceValue.UNDETERMINED,
            scope=ForceScope.PROVISION,
            snapshot_date=date(2026, 1, 1),
            source_locator=f"DU-2020-1/art-{idx}",
        ),
        legal_status_date=date(2026, 1, 1),
    )


def _embed_client(handler: object, *, max_retries: int = 0) -> AsyncOpenAI:
    """The build path's real client, driven over a mock transport.

    ``max_retries=0`` by default so a test that counts requests counts what the
    handler saw; the ladder itself is the SDK's and is exercised where a test
    asks for it.
    """
    return AsyncOpenAI(
        api_key="test-key",
        base_url="http://embeddings.test/v1",
        max_retries=max_retries,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )


async def _snapshot_record(client: AsyncQdrantClient, collection: str) -> Any:
    from seeder.pipeline import _published_snapshot_record

    return await _published_snapshot_record(client, collection)


def _manifest_hash(path: Path) -> str:
    from contract_analyzer.corpus import sha256_bytes

    return sha256_bytes(path.read_bytes())


def _snapshot(unit_count: int, snapshot_id: str = "snap-1") -> CorpusSnapshot:
    return CorpusSnapshot(
        id=snapshot_id,
        built_at=datetime(2026, 1, 1, tzinfo=UTC),
        manifest_hash="manifest-1",
        model_name=EMBEDDING_MODEL,
        # The space the suite's offline encoder writes in. verify() compares its
        # own probe against this, so a fixture that omitted it would be refused --
        # which is the point of the guard.
        embedding_space_probe=embedding_space_probe(),
        file_hashes={},
        unit_count=unit_count,
        act_identifiers=("DU-2020-1",),
        corpus_target_date=date(2026, 1, 1),
        act_currency=(),
        unit_digest="digest-1",
    )


def _seed_units(client: QdrantClient, collection: str, units: list[LegalUnit]) -> None:
    ensure_qdrant_collection(client, collection)
    points = []
    for unit in units:
        weights = sparse_document_vector(unit.text)
        points.append(
            models.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_DNS, unit.id)),
                vector={
                    DENSE_VECTOR_NAME: [1.0] * EMBEDDING_DIMENSION,
                    SPARSE_VECTOR_NAME: models.SparseVector(
                        indices=list(weights), values=list(weights.values())
                    ),
                },
                payload={
                    "id": unit.id,
                    "locator": unit.locator,
                    "act_identifier": unit.act_identifier,
                    "article_identifier": unit.article_identifier,
                    "text": unit.text,
                    "content_hash": unit.content_hash,
                    "act_force_value": unit.act_force.value.value,
                    "act_force_scope": unit.act_force.scope.value,
                    "act_force_source_locator": unit.act_force.source_locator,
                    "act_force_snapshot_date": (
                        unit.act_force.snapshot_date.isoformat()
                    ),
                    "provision_force_value": unit.provision_force.value.value,
                    "provision_force_scope": unit.provision_force.scope.value,
                    "provision_force_source_locator": (
                        unit.provision_force.source_locator
                    ),
                    "provision_force_snapshot_date": (
                        unit.provision_force.snapshot_date.isoformat()
                    ),
                    "legal_status_date": unit.legal_status_date.isoformat(),
                },
            )
        )
    client.upsert(collection_name=collection, points=points)


def _publish(client: QdrantClient, collection: str, snapshot: CorpusSnapshot) -> None:
    """Publish through the seeder's own path, so the tests exercise what ships."""
    asyncio.run(_verify_and_publish(_as_async(client), collection, snapshot))


def _as_async(client: QdrantClient) -> Any:
    """Drive the sync local client through the async seeder helpers.

    ``AsyncQdrantClient(":memory:")`` opens a second, empty local store, so an async
    helper cannot be pointed at a collection a sync client built. The local client's
    methods are synchronous either way; awaiting their results is what differs.
    """

    class _Awaitable:
        def __init__(self, inner: QdrantClient) -> None:
            self._inner = inner

        def __getattr__(self, name: str) -> Any:
            attribute = getattr(self._inner, name)

            async def call(*args: Any, **kwargs: Any) -> Any:
                return attribute(*args, **kwargs)

            return call

    return _Awaitable(client)


def test_ensure_qdrant_collection_creates_both_limbs() -> None:
    client = QdrantClient(":memory:")
    ensure_qdrant_collection(client, "test_units")
    config = client.get_collection("test_units").config.params
    assert DENSE_VECTOR_NAME in (config.vectors or {})
    assert SPARSE_VECTOR_NAME in (config.sparse_vectors or {})


@pytest.mark.anyio
async def test_push_worker_batches_and_flushes() -> None:
    client = AsyncQdrantClient(":memory:")
    collection = "test_push_worker"
    await ensure_qdrant_collection_async(client, collection)

    embed_queue: asyncio.Queue[LegalUnit | None] = asyncio.Queue()
    embedded_batches: list[list[str]] = []
    written: list[tuple[str, str]] = []

    async def mock_embed_fn(texts: list[str]) -> list[list[float]]:
        embedded_batches.append(texts)
        return [[1.0] * EMBEDDING_DIMENSION for _ in texts]

    push_task = asyncio.create_task(
        _push_worker(
            embed_queue=embed_queue,
            qdrant=client,
            collection=collection,
            embed_fn=mock_embed_fn,
            batch_size=2,
            written=written,
        )
    )

    await embed_queue.put(_make_unit(1))
    await embed_queue.put(_make_unit(2))  # triggers a batch of 2
    await embed_queue.put(_make_unit(3))  # stays in the buffer
    await embed_queue.put(None)

    count = await push_task
    assert count == 3
    assert len(embedded_batches) == 2

    points, _ = await client.scroll(collection, limit=10)
    assert len(points) == 3
    # The digest is taken over what was actually stored, so the log has to match.
    assert len(written) == 3


_METADATA_RESPONSE = {
    "eli": "DU/2020/1",
    "title": "Ustawa testowa",
    "status": "obowiązujący",
    "legalStatusDate": "2026-01-01",
    "dates": {"inForce": "2020-01-01"},
    "type": "Ustawa",
    "textHTML": "text.html",
    "references": {
        "Tekst jednolity dla aktu": [{"id": "DU/2020/1"}],
        "Akty zmieniające": [],
    },
}


def _write_manifest(path: Path) -> Path:
    path.write_text(
        '{"version": 2, "corpus_target_date": "2026-08-30", "acts": ['
        '{"publisher": "DU", "year": 2020, "position": 1, "source_format": "html", '
        '"base_act": "DU/2020/1", "legal_status_date": "2026-01-01", '
        '"amendments_not_carried": [], "repeated_articles": []}'
        "]}",
        encoding="utf-8",
    )
    return path


def _eli_patches(articles: list[str], bodies: dict[str, bytes]) -> Any:
    async def mock_fetch(
        client: Any, url: str, accept: str = "", **kwargs: Any
    ) -> bytes:
        if "struct" in url:
            return json.dumps(
                [
                    {"id": name, "name": name, "type": "arti", "title": f"Art. {name}."}
                    for name in articles
                ]
            ).encode("utf-8")
        if "text.html" in url:
            for suffix, body in bodies.items():
                if url.endswith(suffix):
                    return body
            raise AssertionError(f"unexpected unit url: {url}")
        return json.dumps(_METADATA_RESPONSE).encode("utf-8")

    return (
        patch("seeder.pipeline.fetch_async", side_effect=mock_fetch),
        patch(
            "seeder.pipeline.deduped_struct_article_paths",
            return_value=[((name,), {"title": f"Art. {name}."}) for name in articles],
        ),
    )


@pytest.mark.anyio
async def test_manifest_names_the_collection_it_builds(tmp_path: Path) -> None:
    """Each manifest builds its own collection, and publishes it under the alias."""
    manifest = _write_manifest(tmp_path / "manifest.json")
    client = AsyncQdrantClient(":memory:")

    async def embed(texts: list[str]) -> list[list[float]]:
        return [[1.0] * EMBEDDING_DIMENSION for _ in texts]

    fetch_patch, struct_patch = _eli_patches(
        ["1", "2"],
        {
            "text.html/1": b"<p>Art. 1. Tresc</p>",
            "text.html/2": b"<p>Art. 2. Inna</p>",
        },
    )
    with fetch_patch, struct_patch:
        result = await seed_qdrant_async(
            manifest, qdrant_client=client, embed_fn=embed, batch_size=10
        )

    expected = corpus_collection_name(
        __import__("hashlib").sha256(manifest.read_bytes()).hexdigest()
    )
    assert result.collection == expected
    assert result.rebuilt is True
    assert result.total_units == 2
    aliases = (await client.get_aliases()).aliases
    assert [(a.alias_name, a.collection_name) for a in aliases] == [
        (CORPUS_ALIAS, expected)
    ]


@pytest.mark.anyio
async def test_rebuild_does_not_leave_units_the_manifest_dropped(
    tmp_path: Path,
) -> None:
    """The central stale-unit guarantee: a rebuilt corpus holds only what it built."""
    manifest = _write_manifest(tmp_path / "manifest.json")
    client = AsyncQdrantClient(":memory:")

    async def embed(texts: list[str]) -> list[list[float]]:
        return [[1.0] * EMBEDDING_DIMENSION for _ in texts]

    two_articles = _eli_patches(
        ["1", "2"],
        {
            "text.html/1": b"<p>Art. 1. Tresc</p>",
            "text.html/2": b"<p>Art. 2. Inna</p>",
        },
    )
    with two_articles[0], two_articles[1]:
        first = await seed_qdrant_async(
            manifest, qdrant_client=client, embed_fn=embed, batch_size=10
        )
    assert first.total_units == 2

    one_article = _eli_patches(["1"], {"text.html/1": b"<p>Art. 1. Tresc</p>"})
    with one_article[0], one_article[1]:
        second = await seed_qdrant_async(
            manifest,
            qdrant_client=client,
            embed_fn=embed,
            batch_size=10,
            rebuild=True,
        )

    assert second.total_units == 1
    stored = (await client.get_collection(second.collection)).points_count
    assert stored == 1
    points, _ = await client.scroll(second.collection, limit=10, with_payload=True)
    assert [point.payload["article_identifier"] for point in points] == ["Art. 1"]
    # A changed corpus is a different corpus, and says so.
    assert second.snapshot_id != first.snapshot_id


@pytest.mark.anyio
async def test_reseeding_a_published_corpus_rebuilds_nothing(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path / "manifest.json")
    client = AsyncQdrantClient(":memory:")
    embed_calls = 0

    async def embed(texts: list[str]) -> list[list[float]]:
        nonlocal embed_calls
        # The build also encodes the space probe through this same function, which
        # is what fingerprints an injected encoder as itself. It is not a passage,
        # so it does not count as rebuilding one.
        if texts != [EMBEDDING_PROBE_TEXT]:
            embed_calls += 1
        return [[1.0] * EMBEDDING_DIMENSION for _ in texts]

    patches = _eli_patches(
        ["1", "2"],
        {
            "text.html/1": b"<p>Art. 1. Tresc</p>",
            "text.html/2": b"<p>Art. 2. Inna</p>",
        },
    )
    with patches[0], patches[1]:
        first = await seed_qdrant_async(
            manifest, qdrant_client=client, embed_fn=embed, batch_size=10
        )
        embed_calls = 0
        second = await seed_qdrant_async(
            manifest, qdrant_client=client, embed_fn=embed, batch_size=10
        )

    assert embed_calls == 0
    assert second.rebuilt is False
    assert second.snapshot_id == first.snapshot_id
    assert second.total_units == first.total_units


@pytest.mark.anyio
async def test_an_invalid_worker_count_leaves_the_published_corpus_alone(
    tmp_path: Path,
) -> None:
    """Zero workers is refused before any point is read, written or dropped."""
    manifest = _write_manifest(tmp_path / "manifest.json")
    client = AsyncQdrantClient(":memory:")

    async def embed(texts: list[str]) -> list[list[float]]:
        return [[1.0] * EMBEDDING_DIMENSION for _ in texts]

    patches = _eli_patches(
        ["1", "2"],
        {
            "text.html/1": b"<p>Art. 1. Tresc</p>",
            "text.html/2": b"<p>Art. 2. Inna</p>",
        },
    )
    with patches[0], patches[1]:
        first = await seed_qdrant_async(
            manifest, qdrant_client=client, embed_fn=embed, batch_size=10
        )
        with pytest.raises(ValueError, match="num_workers"):
            await seed_qdrant_async(
                manifest,
                qdrant_client=client,
                embed_fn=embed,
                batch_size=10,
                num_workers=0,
            )

    stored = await client.get_collection(first.collection)
    assert stored.points_count == first.total_units


@pytest.mark.anyio
async def test_a_build_that_stored_nothing_is_not_published(tmp_path: Path) -> None:
    client = AsyncQdrantClient(":memory:")
    await ensure_qdrant_collection_async(client, "legal_units_empty")
    with pytest.raises(Exception) as excinfo:
        await _verify_and_publish(client, "legal_units_empty", _snapshot(0))
    assert getattr(excinfo.value, "code", "") == "empty_corpus"
    assert (await client.get_aliases()).aliases == []


def test_qdrant_corpus_index_reads_and_searches() -> None:
    client = QdrantClient(":memory:")
    collection = "legal_units_index_test"
    unit = _make_unit(1)
    _seed_units(client, collection, [unit])
    index = QdrantCorpusIndex(
        client=client, collection=collection, snapshot=_snapshot(1)
    )

    assert index.read(unit.locator).id == unit.id
    assert index._read_by_id(unit.id).id == unit.id
    with pytest.raises(KeyError):
        index.read("non-existent-locator")

    with patch(
        "contract_analyzer.corpus._embed",
        return_value=np.ones((1, EMBEDDING_DIMENSION), dtype=np.float32),
    ):
        candidates = index.search("zapytanie", limit=5)
    assert [result.candidate.locator for result in candidates] == [unit.locator]
    assert candidates[0].candidate.rank == 1


def test_search_retrieves_a_lexical_match_the_dense_limb_misses() -> None:
    """The sparse limb has to be doing work, not reporting a zero beside the dense one.

    Every unit carries the same dense vector, so the dense limb ranks by nothing and
    only a lexical match can put the right unit first.
    """
    client = QdrantClient(":memory:")
    collection = "legal_units_hybrid"
    units = [
        _make_unit(1, text="Przepisy ogólne o zobowiązaniach umownych"),
        _make_unit(2, text="Kara umowna zastrzeżona na wypadek zwłoki"),
        _make_unit(3, text="Przedawnienie roszczeń majątkowych"),
    ]
    _seed_units(client, collection, units)
    index = QdrantCorpusIndex(
        client=client, collection=collection, snapshot=_snapshot(3)
    )

    with patch(
        "contract_analyzer.corpus._embed",
        return_value=np.ones((1, EMBEDDING_DIMENSION), dtype=np.float32),
    ):
        candidates = index.search("kara umowna zwłoka", limit=3)

    assert candidates[0].candidate.locator == units[1].locator
    assert candidates[0].candidate.sparse_score > 0.0


def test_index_refuses_a_collection_no_build_published() -> None:
    """An unpublished collection has no identity, so it cannot be read as a corpus."""
    client = QdrantClient(":memory:")
    ensure_qdrant_collection(client, "legal_units_unpublished")
    index = QdrantCorpusIndex(client=client, collection="legal_units_unpublished")

    with pytest.raises(CorpusIntegrityError) as excinfo:
        index.verify()
    assert excinfo.value.code == "corpus_snapshot_missing"

    # And it cannot quietly supply a snapshot id for a RunRecord either.
    with pytest.raises(CorpusIntegrityError):
        _ = index.snapshot


def test_index_refuses_a_corpus_it_is_not_pinned_to() -> None:
    client = QdrantClient(":memory:")
    collection = "legal_units_pinned"
    _seed_units(client, collection, [_make_unit(1)])
    _publish(client, collection, _snapshot(1, "snapshot-actually-published"))

    QdrantCorpusIndex(
        client=client,
        collection=collection,
        expected_snapshot_id="snapshot-actually-published",
    ).verify()

    pinned = QdrantCorpusIndex(
        client=client, collection=collection, expected_snapshot_id="snapshot-expected"
    )
    with pytest.raises(CorpusIntegrityError) as excinfo:
        pinned.verify()
    assert excinfo.value.code == "corpus_snapshot_mismatch"


def test_index_refuses_a_corpus_that_lost_points() -> None:
    """The point count is checked against the snapshot, not merely against zero."""
    client = QdrantClient(":memory:")
    collection = "legal_units_short"
    units = [_make_unit(1), _make_unit(2)]
    _seed_units(client, collection, units)
    _publish(client, collection, _snapshot(2))

    client.delete(
        collection_name=collection,
        points_selector=models.PointIdsList(
            points=[str(uuid.uuid5(uuid.NAMESPACE_DNS, units[1].id))]
        ),
    )
    assert client.get_collection(collection).points_count == 1

    with pytest.raises(CorpusIntegrityError) as excinfo:
        QdrantCorpusIndex(client=client, collection=collection).verify()
    assert excinfo.value.code == "corpus_incomplete"


def test_index_resolves_the_published_alias() -> None:
    client = QdrantClient(":memory:")
    collection = "legal_units_aliased"
    _seed_units(client, collection, [_make_unit(1)])
    _publish(client, collection, _snapshot(1, "aliased-snapshot"))

    index = QdrantCorpusIndex(client=client, collection=CORPUS_ALIAS)
    index.verify()
    assert index.snapshot.id == "aliased-snapshot"


@pytest.mark.anyio
async def test_delta_seeding_skips_unchanged_units_and_embeds_deltas(
    tmp_path: Path,
) -> None:
    manifest = _write_manifest(tmp_path / "manifest.json")
    client = AsyncQdrantClient(":memory:")
    embedded_passages: list[str] = []

    async def embed(texts: list[str]) -> list[list[float]]:
        embedded_passages.extend(t for t in texts if t != EMBEDDING_PROBE_TEXT)
        return [[1.0] * EMBEDDING_DIMENSION for _ in texts]

    patches1 = _eli_patches(
        ["1", "2"],
        {
            "text.html/1": b"<p>Art. 1. Tresc</p>",
            "text.html/2": b"<p>Art. 2. Stara tresc</p>",
        },
    )
    with patches1[0], patches1[1]:
        first = await seed_qdrant_async(
            manifest, qdrant_client=client, embed_fn=embed, batch_size=10
        )
    assert first.total_units == 2
    assert first.newly_embedded == 2
    assert first.skipped_existing == 0
    embedded_passages.clear()

    # Clear snapshot record to simulate delta run without no-op shortcut
    await client.delete_collection(CORPUS_SNAPSHOTS_COLLECTION)

    patches2 = _eli_patches(
        ["1", "2"],
        {
            "text.html/1": b"<p>Art. 1. Tresc</p>",
            "text.html/2": b"<p>Art. 2. Nowa tresc</p>",
        },
    )
    with patches2[0], patches2[1]:
        second = await seed_qdrant_async(
            manifest, qdrant_client=client, embed_fn=embed, batch_size=10
        )

    assert second.total_units == 2
    assert second.skipped_existing == 1
    assert second.newly_embedded == 1
    assert len(embedded_passages) == 1
    assert "Nowa tresc" in embedded_passages[0]
    stored = (await client.get_collection(second.collection)).points_count
    assert stored == 2


@pytest.mark.anyio
async def test_delta_seeding_purges_dropped_units(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path / "manifest.json")
    client = AsyncQdrantClient(":memory:")

    async def embed(texts: list[str]) -> list[list[float]]:
        return [[1.0] * EMBEDDING_DIMENSION for _ in texts]

    patches1 = _eli_patches(
        ["1", "2"],
        {
            "text.html/1": b"<p>Art. 1. Tresc</p>",
            "text.html/2": b"<p>Art. 2. Inna</p>",
        },
    )
    with patches1[0], patches1[1]:
        first = await seed_qdrant_async(
            manifest, qdrant_client=client, embed_fn=embed, batch_size=10
        )
    assert first.total_units == 2

    # Clear snapshot record to trigger delta pass
    await client.delete_collection(CORPUS_SNAPSHOTS_COLLECTION)

    patches2 = _eli_patches(["1"], {"text.html/1": b"<p>Art. 1. Tresc</p>"})
    with patches2[0], patches2[1]:
        second = await seed_qdrant_async(
            manifest, qdrant_client=client, embed_fn=embed, batch_size=10
        )

    assert second.total_units == 1
    assert second.skipped_existing == 1
    assert second.newly_embedded == 0
    stored = (await client.get_collection(second.collection)).points_count
    assert stored == 1
    points, _ = await client.scroll(second.collection, limit=10, with_payload=True)
    assert [p.payload["article_identifier"] for p in points] == ["Art. 1"]


@pytest.mark.anyio
async def test_eli_disk_cache_avoids_repeated_network_calls(tmp_path: Path) -> None:
    import httpx

    from seeder.eli import fetch_async

    cache_dir = tmp_path / "eli_cache"
    url = "https://api.sejm.gov.pl/test-unit"

    call_count = 0

    class MockTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            return httpx.Response(200, content=b"<p>cached content</p>")

    async with httpx.AsyncClient(transport=MockTransport()) as client:
        res1 = await fetch_async(client, url, accept="text/html", cache_dir=cache_dir)
        assert res1 == b"<p>cached content</p>"
        assert call_count == 1
        assert len(list(cache_dir.glob("*.bin"))) == 1

        res2 = await fetch_async(client, url, accept="text/html", cache_dir=cache_dir)
        assert res2 == b"<p>cached content</p>"
        assert call_count == 1


@pytest.mark.anyio
async def test_push_worker_error_fails_fast_without_deadlock(tmp_path: Path) -> None:
    from contract_analyzer.corpus import EmbeddingError

    manifest = _write_manifest(tmp_path / "manifest.json")
    client = AsyncQdrantClient(":memory:")

    async def broken_embed(texts: list[str]) -> list[list[float]]:
        raise EmbeddingError("service_down", "Simulated embedding service failure")

    patches = _eli_patches(
        ["1", "2"],
        {
            "text.html/1": b"<p>Art. 1. Tresc</p>",
            "text.html/2": b"<p>Art. 2. Inna</p>",
        },
    )
    with patches[0], patches[1]:
        with pytest.raises(EmbeddingError) as excinfo:
            await seed_qdrant_async(
                manifest, qdrant_client=client, embed_fn=broken_embed, batch_size=1
            )
        assert excinfo.value.code == "service_down"


@pytest.mark.anyio
async def test_put_with_check_raises_when_worker_fails_even_if_queue_full() -> None:
    from contract_analyzer.corpus import EmbeddingError
    from seeder.pipeline import _put_with_check

    queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=1)
    queue.put_nowait("existing_item")
    assert queue.full()

    async def broken_task() -> None:
        raise EmbeddingError("service_down", "Dead consumer")

    task = asyncio.create_task(broken_task())
    await asyncio.sleep(0.01)

    with pytest.raises(EmbeddingError) as excinfo:
        await _put_with_check(queue, task, None)
    assert excinfo.value.code == "service_down"


@pytest.mark.anyio
async def test_delta_seeding_updates_metadata_when_content_hash_matches(
    tmp_path: Path,
) -> None:
    """When text is unchanged, vectors are NOT re-embedded, but metadata is updated."""
    manifest = _write_manifest(tmp_path / "manifest.json")
    client = AsyncQdrantClient(":memory:")
    embed_calls = 0

    async def embed(texts: list[str]) -> list[list[float]]:
        nonlocal embed_calls
        # The space probe goes through this function too; it is not a passage.
        embed_calls += sum(1 for t in texts if t != EMBEDDING_PROBE_TEXT)
        return [[1.0] * EMBEDDING_DIMENSION for _ in texts]

    async def mock_initial_fetch(
        http_client: Any, url: str, accept: str = "", **kwargs: Any
    ) -> bytes:
        if "struct" in url:
            return json.dumps(
                [{"type": "arti", "title": "Art. 1.", "children": []}]
            ).encode()
        if "text.html" in url:
            return b"<p>Art. 1. Tresc</p>"
        return json.dumps({**_METADATA_RESPONSE, "inForce": "IN_FORCE"}).encode()

    from unittest.mock import patch

    struct_patch = patch(
        "seeder.pipeline.deduped_struct_article_paths",
        return_value=[(("1",), {"title": "Art. 1."})],
    )

    with (
        patch("seeder.pipeline.fetch_async", side_effect=mock_initial_fetch),
        struct_patch,
    ):
        first = await seed_qdrant_async(
            manifest, qdrant_client=client, embed_fn=embed, batch_size=10
        )
    assert first.total_units == 1
    assert first.newly_embedded == 1
    assert embed_calls == 1

    points, _ = await client.scroll(first.collection, limit=1, with_payload=True)
    assert points[0].payload["act_force_value"] == "in_force"

    # Clear snapshot record to trigger delta pass on the same collection
    await client.delete_collection(CORPUS_SNAPSHOTS_COLLECTION)

    async def mock_repealed_fetch(
        http_client: Any, url: str, accept: str = "", **kwargs: Any
    ) -> bytes:
        if "struct" in url:
            return json.dumps(
                [{"type": "arti", "title": "Art. 1.", "children": []}]
            ).encode()
        if "text.html" in url:
            return b"<p>Art. 1. Tresc</p>"
        return json.dumps({**_METADATA_RESPONSE, "inForce": "NOT_IN_FORCE"}).encode()

    with (
        patch("seeder.pipeline.fetch_async", side_effect=mock_repealed_fetch),
        struct_patch,
    ):
        second = await seed_qdrant_async(
            manifest, qdrant_client=client, embed_fn=embed, batch_size=10
        )

    assert second.total_units == 1
    assert second.skipped_existing == 1
    assert second.newly_embedded == 0
    assert embed_calls == 1  # Not called again!

    points2, _ = await client.scroll(second.collection, limit=1, with_payload=True)
    assert points2[0].payload["act_force_value"] == "not_in_force"


@pytest.mark.anyio
async def test_publish_alias_switches_atomically(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path / "manifest.json")
    client = AsyncQdrantClient(":memory:")

    async def embed(texts: list[str]) -> list[list[float]]:
        return [[1.0] * EMBEDDING_DIMENSION for _ in texts]

    patches = _eli_patches(["1"], {"text.html/1": b"<p>Art. 1. Tresc</p>"})
    with patches[0], patches[1]:
        await seed_qdrant_async(
            manifest,
            qdrant_client=client,
            embed_fn=embed,
            collection="legal_units_first",
        )
        await seed_qdrant_async(
            manifest,
            qdrant_client=client,
            embed_fn=embed,
            collection="legal_units_second",
        )

    aliases = await client.get_aliases()
    alias_map = {a.alias_name: a.collection_name for a in aliases.aliases}
    assert alias_map[CORPUS_ALIAS] == "legal_units_second"


@pytest.mark.anyio
async def test_publish_alias_refuses_colliding_physical_collection_without_rebuild(
    tmp_path: Path,
) -> None:
    from contract_analyzer.domain import CorpusBuildError

    manifest = _write_manifest(tmp_path / "manifest.json")
    client = AsyncQdrantClient(":memory:")

    await client.create_collection(
        collection_name=CORPUS_ALIAS,
        vectors_config=models.VectorParams(size=1, distance=models.Distance.COSINE),
    )

    async def embed(texts: list[str]) -> list[list[float]]:
        return [[1.0] * EMBEDDING_DIMENSION for _ in texts]

    patches = _eli_patches(["1"], {"text.html/1": b"<p>Art. 1. Tresc</p>"})
    with patches[0], patches[1]:
        with pytest.raises(CorpusBuildError) as excinfo:
            await seed_qdrant_async(
                manifest,
                qdrant_client=client,
                embed_fn=embed,
                collection="target_coll",
                rebuild=False,
            )
        assert excinfo.value.code == "physical_collection_conflict"

        res = await seed_qdrant_async(
            manifest,
            qdrant_client=client,
            embed_fn=embed,
            collection="target_coll",
            rebuild=True,
        )
        assert res.rebuilt is True
        aliases = await client.get_aliases()
        alias_map = {a.alias_name: a.collection_name for a in aliases.aliases}
        assert alias_map[CORPUS_ALIAS] == "target_coll"


@pytest.mark.anyio
async def test_eli_disk_cache_evicts_corrupt_content_and_supports_refresh(
    tmp_path: Path,
) -> None:
    import httpx

    from seeder.eli import fetch_async

    cache_dir = tmp_path / "eli_cache"
    url = "https://api.sejm.gov.pl/eli/acts/test"

    call_count = 0

    class MockTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            return httpx.Response(200, content=f'{{"count": {call_count}}}'.encode())

    async with httpx.AsyncClient(transport=MockTransport()) as client:
        res1 = await fetch_async(
            client, url, accept="application/json", cache_dir=cache_dir
        )
        assert json.loads(res1) == {"count": 1}
        assert call_count == 1

        res2 = await fetch_async(
            client, url, accept="application/json", cache_dir=cache_dir
        )
        assert json.loads(res2) == {"count": 1}
        assert call_count == 1

        cache_files = list(cache_dir.glob("*.bin"))
        assert len(cache_files) == 1
        cache_files[0].write_bytes(b"<html>500 Error page</html>")

        # Third call: detects corrupt cached file, evicts it, and fetches from network
        res3 = await fetch_async(
            client, url, accept="application/json", cache_dir=cache_dir
        )
        assert json.loads(res3) == {"count": 2}
        assert call_count == 2

        res4 = await fetch_async(
            client,
            url,
            accept="application/json",
            cache_dir=cache_dir,
            refresh_cache=True,
        )
        assert json.loads(res4) == {"count": 3}
        assert call_count == 3


@pytest.mark.anyio
async def test_eli_disk_cache_no_cache_ignores_eli_cache_dir_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import httpx

    from seeder.eli import fetch_async

    cache_dir = tmp_path / "env_eli_cache"
    monkeypatch.setenv("ELI_CACHE_DIR", str(cache_dir))
    url = "https://api.sejm.gov.pl/eli/acts/test-no-cache"

    call_count = 0

    class MockTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            return httpx.Response(200, content=f'{{"count": {call_count}}}'.encode())

    async with httpx.AsyncClient(transport=MockTransport()) as client:
        res1 = await fetch_async(client, url, accept="application/json", no_cache=True)
        assert json.loads(res1) == {"count": 1}
        assert call_count == 1
        assert not cache_dir.exists()

        res2 = await fetch_async(client, url, accept="application/json", no_cache=True)
        assert json.loads(res2) == {"count": 2}
        assert call_count == 2
        assert not cache_dir.exists()


def test_seeder_cli_no_cache_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    from unittest.mock import patch

    from seeder.__main__ import main
    from seeder.pipeline import SeedResult

    monkeypatch.setenv("ELI_CACHE_DIR", "/tmp/env_cache")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "seeder",
            "--no-cache",
            "--manifest",
            "corpus/manifest.smoke.json",
        ],
    )

    with patch("seeder.__main__.seed_qdrant") as mock_seed:
        mock_seed.return_value = SeedResult(
            total_units=0,
            skipped_existing=0,
            newly_embedded=0,
            embedding_tokens=None,
            collection="c",
            snapshot_id="s",
            rebuilt=False,
        )
        main()
        assert mock_seed.call_count == 1
        _, kwargs = mock_seed.call_args
        assert kwargs["no_cache"] is True


@pytest.mark.anyio
async def test_a_single_input_that_cannot_be_embedded_still_raises() -> None:
    """Splitting must not become a way to drop a unit."""

    from contract_analyzer.corpus import EmbeddingError
    from seeder import pipeline

    def handler(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.RemoteProtocolError("dead", request=request)

    async with _embed_client(handler) as client:
        with pytest.raises(EmbeddingError):
            await pipeline._embed_batch_async(["a", "b"], client)


@pytest.mark.anyio
async def test_seeder_passage_prefix_matches_backend_query_side() -> None:
    """A corpus written under one convention and queried under another retrieves
    worse, silently -- nothing else in the suite would notice.

    The model is asymmetric by instruction: a query carries a task line and a
    passage does not. This asserts both halves, because asserting that a passage
    goes through a pass-through function only ever asserted that it equals
    itself.
    """
    from contract_analyzer.corpus import embed_query_text

    client = AsyncQdrantClient(":memory:")
    collection = "test_passage_prefix"
    await ensure_qdrant_collection_async(client, collection)

    embed_queue: asyncio.Queue[LegalUnit | None] = asyncio.Queue()
    sent: list[str] = []

    async def capture(texts: list[str]) -> list[list[float]]:
        sent.extend(texts)
        return [[1.0] * EMBEDDING_DIMENSION for _ in texts]

    task = asyncio.create_task(
        _push_worker(
            embed_queue=embed_queue,
            qdrant=client,
            collection=collection,
            embed_fn=capture,
            batch_size=2,
        )
    )
    unit = _make_unit(1)
    await embed_queue.put(unit)
    await embed_queue.put(None)
    await task

    assert sent, "nothing was embedded"
    assert sent[0] == unit.text, "the seeder must send a passage bare"
    queried = embed_query_text(unit.text)
    assert queried != sent[0], "the query side must carry the task instruction"
    assert queried.endswith(unit.text), "the instruction goes before the text"


@pytest.mark.anyio
async def test_a_rejected_request_is_not_retried() -> None:
    """A 4xx says the same thing however often it is sent."""
    from contract_analyzer.corpus import EmbeddingError
    from seeder import pipeline

    calls: list[int] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        calls.append(len(json.loads(request.content)["input"]))
        return httpx2.Response(401, json={"error": "invalid api key"})

    async with _embed_client(handler) as client:
        with pytest.raises(EmbeddingError) as caught:
            await pipeline._embed_batch_async([f"unit {i}" for i in range(16)], client)

    assert caught.value.code == "embedding_request_rejected"
    assert calls == [16], f"a permanent rejection was retried: {calls}"


@pytest.mark.anyio
async def test_a_malformed_body_is_not_retried() -> None:
    """A healthy host returning the wrong shape returns it again for every subset."""
    from contract_analyzer.corpus import EmbeddingError
    from seeder import pipeline

    calls: list[int] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        calls.append(len(json.loads(request.content)["input"]))
        return httpx2.Response(200, json={"data": [{"no_index": 1}]})

    async with _embed_client(handler) as client:
        with pytest.raises(EmbeddingError) as caught:
            await pipeline._embed_batch_async([f"unit {i}" for i in range(16)], client)

    assert caught.value.code == "embedding_response_malformed"
    assert calls == [16], f"a malformed body was retried: {calls}"


@pytest.mark.anyio
async def test_a_short_response_is_malformed_rather_than_a_short_result() -> None:
    """Fewer vectors than inputs is a malformed body, not a result to carry on with.

    Every item here is well formed, so nothing in the parse itself raises and only
    counting the vectors catches it. Returning the short list instead left the
    mismatch to surface as a bare ValueError from a later strict zip, which carries
    no code and so could not be told apart from a transport failure that retrying
    might actually have fixed.
    """
    from contract_analyzer.corpus import EmbeddingError
    from seeder import pipeline

    calls: list[int] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        count = len(json.loads(request.content)["input"])
        calls.append(count)
        return httpx2.Response(
            200,
            json={"data": [{"index": i, "embedding": [0.1]} for i in range(count - 1)]},
        )

    async with _embed_client(handler) as client:
        with pytest.raises(EmbeddingError) as caught:
            await pipeline._embed_batch_async([f"unit {i}" for i in range(16)], client)

    assert caught.value.code == "embedding_response_malformed"
    assert calls == [16], f"a short body was retried: {calls}"


@pytest.mark.anyio
async def test_a_rate_limit_is_waited_out() -> None:
    """Splitting a throttled batch doubles the rate against the thing throttling.

    A rate limit is about how often we ask, not about the shape of the batch, so
    halving it cannot help and makes the next second worse. It must come back as
    its own code and leave the request count where it was.
    """
    from contract_analyzer.corpus import EmbeddingError
    from seeder import pipeline

    calls: list[int] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        calls.append(len(json.loads(request.content)["input"]))
        return httpx2.Response(429, headers={"Retry-After": "0"}, json={"e": "slow"})

    async with _embed_client(handler) as client:
        with pytest.raises(EmbeddingError) as caught:
            await pipeline._embed_batch_async([f"unit {i}" for i in range(16)], client)

    assert caught.value.code == "embedding_rate_limited"
    assert set(calls) == {16}, f"a rate limit changed the batch: {calls}"


@pytest.mark.anyio
async def test_requests_run_in_parallel_but_no_more_than_the_limit() -> None:
    """Groups overlap, and the number in flight never exceeds the configured bound.

    Unbounded would be faster still and is exactly how a build walks into a rate
    limit; one at a time is what the throughput measurement rejected, at 5,626
    characters a second against 21,724 for eight.
    """
    from seeder import pipeline

    in_flight = 0
    peak = 0

    async def handler(request: httpx2.Request) -> httpx2.Response:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        try:
            # Yields the loop, so overlapping requests are genuinely concurrent
            # rather than merely interleaved on paper.
            await asyncio.sleep(0.01)
            count = len(json.loads(request.content)["input"])
            return httpx2.Response(
                200,
                json={
                    "data": [
                        {"index": i, "embedding": [0.1] * EMBEDDING_DIMENSION}
                        for i in range(count)
                    ]
                },
            )
        finally:
            in_flight -= 1

    # Enough inputs that batching yields many more requests than the cap.
    texts = [f"unit {i}" for i in range(pipeline.EMBEDDING_CONCURRENCY * 300)]
    async with _embed_client(handler) as client:
        got, _ = await pipeline._embed_batch_async(texts, client)

    assert len(got) == len(texts)
    assert peak > 1, "requests were issued one at a time"
    assert peak <= pipeline.EMBEDDING_CONCURRENCY, (
        f"{peak} requests were in flight, over the "
        f"{pipeline.EMBEDDING_CONCURRENCY} allowed"
    )


def test_verify_refuses_a_corpus_built_in_another_vector_space() -> None:
    """The failure that a model name cannot catch.

    One snapshot identifier has already stood for two spaces 0.53 cosine apart.
    A corpus like that answers every query, ranks worse, and reports nothing.
    """
    client = QdrantClient(":memory:")
    units = [_make_unit(1)]
    _seed_units(client, CORPUS_ALIAS, units)
    elsewhere = tuple([1.0] + [0.0] * (EMBEDDING_DIMENSION - 1))
    _publish(
        client,
        CORPUS_ALIAS,
        _snapshot(len(units)).model_copy(update={"embedding_space_probe": elsewhere}),
    )

    index = QdrantCorpusIndex(client=client, collection=CORPUS_ALIAS)
    with pytest.raises(CorpusIntegrityError) as caught:
        index.verify()
    assert caught.value.code == "corpus_embedding_space_mismatch"


def test_verify_refuses_a_corpus_that_recorded_no_vector_space() -> None:
    """A snapshot from before the space was recorded is rebuilt, not served."""
    client = QdrantClient(":memory:")
    units = [_make_unit(1)]
    _seed_units(client, CORPUS_ALIAS, units)
    _publish(
        client,
        CORPUS_ALIAS,
        _snapshot(len(units)).model_copy(update={"embedding_space_probe": ()}),
    )

    index = QdrantCorpusIndex(client=client, collection=CORPUS_ALIAS)
    with pytest.raises(CorpusIntegrityError) as caught:
        index.verify()
    assert caught.value.code == "corpus_embedding_space_unrecorded"


def test_verify_encodes_its_probe_once_however_often_it_is_polled() -> None:
    """compose polls readiness every 30s; the space is not a property of the minute."""
    client = QdrantClient(":memory:")
    units = [_make_unit(1)]
    _seed_units(client, CORPUS_ALIAS, units)
    _publish(client, CORPUS_ALIAS, _snapshot(len(units)))

    index = QdrantCorpusIndex(client=client, collection=CORPUS_ALIAS)
    calls = 0
    real = corpus_module._embed

    def counting(texts: list[str], client: object | None = None) -> Any:
        nonlocal calls
        calls += 1
        return real(texts, client=client)

    corpus_module._embed = counting
    try:
        for _ in range(5):
            index.verify()
    finally:
        corpus_module._embed = real
    assert calls == 1, f"readiness encoded {calls} times for 5 polls"


@pytest.mark.anyio
async def test_the_build_records_the_vector_space_it_actually_wrote_in(
    tmp_path: Path,
) -> None:
    """The published snapshot must carry the probe, not merely have room for one.

    The field existed for a while with nothing writing it, which is the same as
    not having it: verify() would have compared against an empty record and the
    guard would have been decorative.
    """
    manifest = _write_manifest(tmp_path / "manifest.json")
    client = AsyncQdrantClient(":memory:")
    # A space of this encoder's own, distinguishable from any other.
    space = [0.0] * EMBEDDING_DIMENSION
    space[3] = 1.0

    async def embed(texts: list[str]) -> list[list[float]]:
        return [list(space) for _ in texts]

    patches = _eli_patches(["1"], {"text.html/1": b"<p>Art. 1. Tresc</p>"})
    with patches[0], patches[1]:
        await seed_qdrant_async(
            manifest, qdrant_client=client, embed_fn=embed, batch_size=10
        )

    records, _ = await client.scroll(
        collection_name=CORPUS_SNAPSHOTS_COLLECTION, limit=2, with_payload=True
    )
    assert len(records) == 1
    published = CorpusSnapshot.model_validate_json(records[0].payload["snapshot"])
    assert published.embedding_space_probe, "the build published no vector space"
    assert embedding_spaces_agree(published.embedding_space_probe, space)


def test_two_declared_vector_spaces_do_not_share_a_collection() -> None:
    """A collection name has to separate what the operator declared.

    The name folded in the model but not the width or the upstream, so two
    corpora built in different spaces took the same name -- and with it the same
    snapshot record id, so the second build overwrote the first's record and a
    pinned snapshot id then meant whichever had been written last.
    """
    import contract_analyzer.corpus.identity as identity

    baseline = identity.corpus_collection_name("manifest-1")
    baseline_id = identity.corpus_snapshot_id("manifest-1", "digest-1")

    with patch.object(identity, "EMBEDDING_DIMENSION", EMBEDDING_DIMENSION * 2):
        assert identity.corpus_collection_name("manifest-1") != baseline
        assert identity.corpus_snapshot_id("manifest-1", "digest-1") != baseline_id

    with patch.object(identity, "effective_embedding_provider", lambda: "nebius"):
        assert identity.corpus_collection_name("manifest-1") != baseline
        assert identity.corpus_snapshot_id("manifest-1", "digest-1") != baseline_id


@pytest.mark.anyio
async def test_the_published_snapshot_names_the_upstream_it_built_through(
    tmp_path: Path,
) -> None:
    """The field existed and was never written, which is the same as not having it."""
    manifest = _write_manifest(tmp_path / "manifest.json")
    client = AsyncQdrantClient(":memory:")

    async def embed(texts: list[str]) -> list[list[float]]:
        return [[1.0] * EMBEDDING_DIMENSION for _ in texts]

    patches = _eli_patches(["1"], {"text.html/1": b"<p>Art. 1. Tresc</p>"})
    with patches[0], patches[1]:
        await seed_qdrant_async(
            manifest, qdrant_client=client, embed_fn=embed, batch_size=10
        )

    records, _ = await client.scroll(
        collection_name=CORPUS_SNAPSHOTS_COLLECTION, limit=2, with_payload=True
    )
    published = CorpusSnapshot.model_validate_json(records[0].payload["snapshot"])
    assert published.embedding_provider == corpus_module.EMBEDDINGS_PROVIDER
    assert published.embedding_provider, "the upstream was not recorded"


@pytest.mark.anyio
async def test_a_build_records_what_the_provider_billed_it(tmp_path: Path) -> None:
    """Quality is never reported without its cost, and this is where the cost is.

    A build encodes the whole corpus; a query encodes one phrase. The count was
    already on the wire and was being parsed and thrown away.
    """
    manifest = _write_manifest(tmp_path / "manifest.json")
    client = AsyncQdrantClient(":memory:")

    def handler(request: httpx2.Request) -> httpx2.Response:
        count = len(json.loads(request.content)["input"])
        return httpx2.Response(
            200,
            json={
                "object": "list",
                "model": "m",
                "data": [
                    {
                        "object": "embedding",
                        "index": i,
                        "embedding": [1.0] * EMBEDDING_DIMENSION,
                    }
                    for i in range(count)
                ],
                "usage": {"prompt_tokens": 7 * count, "total_tokens": 7 * count},
            },
        )

    patches = _eli_patches(["1"], {"text.html/1": b"<p>Art. 1. Tresc</p>"})
    with patches[0], patches[1]:
        async with _embed_client(handler) as embed_client:
            with patch.object(
                pipeline, "async_embedding_client", lambda **_: embed_client
            ):
                result = await seed_qdrant_async(
                    manifest, qdrant_client=client, batch_size=10
                )

    # One unit plus the space probe, both billed at seven.
    assert result.embedding_tokens == 14


@pytest.mark.anyio
async def test_a_rebuild_in_another_vector_space_refuses_to_overwrite(
    tmp_path: Path,
) -> None:
    """The hole the snapshot id cannot close, closed where it actually opens.

    The id is content-addressed and deliberately excludes the probe: measured,
    two probes from one provider seconds apart differ by 3.7e-03 per component,
    so no digest of one is stable and hashing it would change the id on every
    rebuild. What must not happen is a rebuild in a different space quietly
    replacing the record, leaving earlier runs citing an id whose corpus they
    never searched.
    """
    manifest = _write_manifest(tmp_path / "manifest.json")
    client = AsyncQdrantClient(":memory:")
    here = [1.0] + [0.0] * (EMBEDDING_DIMENSION - 1)
    elsewhere = [0.0] * (EMBEDDING_DIMENSION - 1) + [1.0]

    def encoder(space: list[float]):
        async def embed(texts: list[str]) -> list[list[float]]:
            return [list(space) for _ in texts]

        return embed

    patches = _eli_patches(["1"], {"text.html/1": b"<p>Art. 1. Tresc</p>"})
    with patches[0], patches[1]:
        await seed_qdrant_async(
            manifest, qdrant_client=client, embed_fn=encoder(here), batch_size=10
        )
        with pytest.raises(CorpusBuildError) as caught:
            await seed_qdrant_async(
                manifest,
                qdrant_client=client,
                embed_fn=encoder(elsewhere),
                batch_size=10,
                rebuild=True,
            )
    assert caught.value.code == "corpus_embedding_space_collision"

    # Refusing is only worth anything if the corpus is still the one that was
    # published. A rebuild discards the collection before it builds, so a guard
    # that fires at publish time leaves the record naming the old space while the
    # collection holds vectors from the new one -- worse than failing outright.
    collection = corpus_collection_name(_manifest_hash(manifest))
    points, _ = await client.scroll(
        collection, limit=5, with_payload=False, with_vectors=True
    )
    assert points, "the published corpus was destroyed"
    for point in points:
        assert embedding_spaces_agree(point.vector[DENSE_VECTOR_NAME], here), (
            "the collection holds vectors from the space the rebuild was refused for"
        )


@pytest.mark.anyio
async def test_a_rebuild_in_the_same_vector_space_is_allowed(tmp_path: Path) -> None:
    """Re-running a build is ordinary; only a substitution is refused."""
    manifest = _write_manifest(tmp_path / "manifest.json")
    client = AsyncQdrantClient(":memory:")
    space = [1.0] + [0.0] * (EMBEDDING_DIMENSION - 1)

    async def embed(texts: list[str]) -> list[list[float]]:
        return [list(space) for _ in texts]

    patches = _eli_patches(["1"], {"text.html/1": b"<p>Art. 1. Tresc</p>"})
    with patches[0], patches[1]:
        await seed_qdrant_async(
            manifest, qdrant_client=client, embed_fn=embed, batch_size=10
        )
        again = await seed_qdrant_async(
            manifest, qdrant_client=client, embed_fn=embed, batch_size=10, rebuild=True
        )
    assert again.total_units == 1


@pytest.mark.anyio
async def test_an_act_falls_back_to_its_pdf_when_eli_s_html_is_down(
    tmp_path: Path,
) -> None:
    """ELI serves the two source formats from separate routes, and the HTML has failed
    on its own: struct, text.html and every article path under it returning 500
    while the PDF answered normally. A build is still possible from the PDF."""
    from fpdf import FPDF

    from font_paths import resolve_test_font

    pdf = FPDF()
    pdf.add_font("DejaVu", "", resolve_test_font())
    pdf.set_font("DejaVu", size=12)
    pdf.add_page()
    for line in ("Art. 1. Pierwszy przepis.", "Art. 2. Drugi przepis."):
        pdf.cell(0, 10, text=line, new_x="LMARGIN", new_y="NEXT")
    pdf_bytes = bytes(pdf.output())

    async def html_is_down(client: Any, url: str, accept: str = "", **kw: Any) -> bytes:
        if "struct" in url or "text.html" in url:
            raise CorpusBuildError("eli_unavailable", f"eli_unavailable: {url}: 500")
        if url.endswith("text.pdf"):
            return pdf_bytes
        return json.dumps(_METADATA_RESPONSE).encode("utf-8")

    manifest = _write_manifest(tmp_path / "manifest.json")
    client = AsyncQdrantClient(":memory:")

    async def embed(texts: list[str]) -> list[list[float]]:
        return [[1.0] * EMBEDDING_DIMENSION for _ in texts]

    with patch("seeder.pipeline.fetch_async", side_effect=html_is_down):
        result = await seed_qdrant_async(
            manifest, qdrant_client=client, embed_fn=embed, batch_size=10
        )

    assert result.total_units == 2, "the act was not built from its PDF"
    points, _ = await client.scroll(result.collection, limit=10, with_payload=True)
    # PDF units, not article paths under text.html: the source formats differ in shape.
    assert all("text.pdf#article=" in p.payload["locator"] for p in points)

    records, _ = await client.scroll(
        collection_name=CORPUS_SNAPSHOTS_COLLECTION, limit=2, with_payload=True
    )
    published = CorpusSnapshot.model_validate_json(records[0].payload["snapshot"])
    assert [a.source_format_used for a in published.act_currency] == ["pdf"], (
        "a fallback must say so; the units it produced are not the manifest's"
    )
    # Both halves, because the comparison has to survive the manifest: what the
    # snapshot was asked for is as much a fact about the build as what it read.
    assert [a.source_format_declared for a in published.act_currency] == ["html"]
    assert published.source_format == "fallback"
    assert published.fallback_acts == ("DU/2020/1",), (
        "the snapshot must name which acts were substituted, not only that one was"
    )


@pytest.mark.anyio
async def test_a_fallback_publishes_the_repeats_its_pdf_actually_prints(
    tmp_path: Path,
) -> None:
    """The snapshot has to carry what the fallback read, not the manifest's blank.

    An act declaring HTML may not carry a non-empty repeated_articles at all, so
    copying the manifest's value onto a PDF-built act would publish "prints
    nothing twice" about a document that does. DU/2024/1513 is the real case: its
    PDF prints 3b, 10, 24 and 25 twice.
    """
    from fpdf import FPDF

    from font_paths import resolve_test_font

    pdf = FPDF()
    pdf.add_font("DejaVu", "", resolve_test_font())
    pdf.set_font("DejaVu", size=12)
    pdf.add_page()
    for line in ("Art. 1. Pierwszy.", "Art. 2. Drugi.", "Art. 2. Drugi ponownie."):
        pdf.cell(0, 10, text=line, new_x="LMARGIN", new_y="NEXT")
    pdf_bytes = bytes(pdf.output())

    async def html_is_down(client: Any, url: str, accept: str = "", **kw: Any) -> bytes:
        if "struct" in url or "text.html" in url:
            raise CorpusBuildError("eli_unavailable", f"eli_unavailable: {url}: 500")
        if url.endswith("text.pdf"):
            return pdf_bytes
        return json.dumps(_METADATA_RESPONSE).encode("utf-8")

    manifest = _write_manifest(tmp_path / "manifest.json")
    client = AsyncQdrantClient(":memory:")

    async def embed(texts: list[str]) -> list[list[float]]:
        return [[1.0] * EMBEDDING_DIMENSION for _ in texts]

    with patch("seeder.pipeline.fetch_async", side_effect=html_is_down):
        result = await seed_qdrant_async(
            manifest, qdrant_client=client, embed_fn=embed, batch_size=10
        )

    assert result.total_units == 3, "both printings of art. 2 belong in the corpus"
    records, _ = await client.scroll(
        collection_name=CORPUS_SNAPSHOTS_COLLECTION, limit=2, with_payload=True
    )
    published = CorpusSnapshot.model_validate_json(records[0].payload["snapshot"])
    (currency,) = published.act_currency
    assert currency.source_format_used == "pdf"
    assert [(r.article, r.printings) for r in currency.repeated_articles] == [
        ("2", 2)
    ], "the snapshot published the manifest's blank instead of what it read"


@pytest.mark.anyio
async def test_a_failure_that_is_not_an_outage_does_not_reach_for_the_pdf(
    tmp_path: Path,
) -> None:
    """The fallback answers an unreachable service, not every failure.

    ELI returning an empty body for the structure is a defect to report, and
    quietly building a different corpus from the PDF instead would hide it. The
    fetch itself has to raise here: a malformed body raises after the fetch
    returns, so it never reaches the branch this is about.
    """

    async def struct_is_empty(
        client: Any, url: str, accept: str = "", **kw: Any
    ) -> bytes:
        if "struct" in url:
            raise CorpusBuildError(
                "empty_source_response", f"empty_source_response: {url}"
            )
        return json.dumps(_METADATA_RESPONSE).encode("utf-8")

    manifest = _write_manifest(tmp_path / "manifest.json")
    client = AsyncQdrantClient(":memory:")

    async def embed(texts: list[str]) -> list[list[float]]:
        return [[1.0] * EMBEDDING_DIMENSION for _ in texts]

    with patch("seeder.pipeline.fetch_async", side_effect=struct_is_empty):
        with pytest.raises(CorpusBuildError) as caught:
            await seed_qdrant_async(
                manifest, qdrant_client=client, embed_fn=embed, batch_size=10
            )
    assert caught.value.code == "empty_source_response", (
        "an outage is not the only way a fetch fails, and only an outage\n"
        "may change the source format"
    )


@pytest.mark.anyio
async def test_a_cached_structure_over_a_dead_text_route_still_falls_back(
    tmp_path: Path,
) -> None:
    """The structure and the article texts are separate routes.

    ELI_CACHE_DIR is set in compose, so a structure can be served from disk while
    live article fetches return 500. Guarding only the structure fetch left that
    case dying half way through an act, after units had already been published.
    """
    from fpdf import FPDF

    from font_paths import resolve_test_font

    pdf = FPDF()
    pdf.add_font("DejaVu", "", resolve_test_font())
    pdf.set_font("DejaVu", size=12)
    pdf.add_page()
    pdf.cell(0, 10, text="Art. 1. Przepis z PDF.", new_x="LMARGIN", new_y="NEXT")
    pdf_bytes = bytes(pdf.output())

    async def struct_cached_text_down(
        client: Any, url: str, accept: str = "", **kw: Any
    ) -> bytes:
        if "struct" in url:
            return json.dumps(
                [{"id": "1", "name": "1", "type": "arti", "title": "Art. 1."}]
            ).encode("utf-8")
        if "text.html" in url:
            raise CorpusBuildError("eli_unavailable", f"eli_unavailable: {url}: 500")
        if url.endswith("text.pdf"):
            return pdf_bytes
        return json.dumps(_METADATA_RESPONSE).encode("utf-8")

    manifest = _write_manifest(tmp_path / "manifest.json")
    client = AsyncQdrantClient(":memory:")

    async def embed(texts: list[str]) -> list[list[float]]:
        return [[1.0] * EMBEDDING_DIMENSION for _ in texts]

    with patch("seeder.pipeline.fetch_async", side_effect=struct_cached_text_down):
        result = await seed_qdrant_async(
            manifest, qdrant_client=client, embed_fn=embed, batch_size=10
        )

    assert result.total_units == 1
    points, _ = await client.scroll(result.collection, limit=10, with_payload=True)
    assert all("text.pdf#article=" in p.payload["locator"] for p in points), (
        "the act was published from the HTML route whose text was unreachable"
    )


def _currency(declared: str | None, used: str | None, act: str = "DU/2020/1") -> Any:
    from contract_analyzer.corpus.models import ActCurrency

    return ActCurrency(
        act_identifier=act,
        base_act=act,
        legal_status_date=date(2026, 1, 1),
        amendments_not_carried=(),
        repeated_articles=(),
        source_format_declared=declared,  # type: ignore[arg-type]
        source_format_used=used,  # type: ignore[arg-type]
    )


@pytest.mark.parametrize(
    ("currency", "expected"),
    [
        ((("html", "html"), ("pdf", "pdf")), "as_declared"),
        ((("html", "pdf"), ("pdf", "pdf")), "fallback"),
        ((("html", "html"), (None, "pdf")), "unrecorded"),
        ((("html", "html"), ("pdf", None)), "unrecorded"),
        ((), "unrecorded"),
    ],
)
def test_a_snapshot_says_whether_it_is_the_corpus_its_manifest_declared(
    currency: tuple[tuple[str | None, str | None], ...], expected: str
) -> None:
    """The source format question has three answers, and one of them is 'cannot say'.

    A snapshot written before the source formats were recorded, or one listing no acts
    at all, has not shown that its acts were read as declared. Answering
    as_declared for either would turn silence into a clean bill of health, which
    is the operator-discipline failure the evaluation runner's gate replaces.
    """
    snapshot = _snapshot(1).model_copy(
        update={"act_currency": tuple(_currency(d, u) for d, u in currency)}
    )
    assert snapshot.source_format == expected


def test_a_snapshot_names_the_acts_it_substituted() -> None:
    """Which acts fell back is the fact a deviation is interpreted against."""
    snapshot = _snapshot(1).model_copy(
        update={
            "act_currency": (
                _currency("html", "pdf", "DU/2023/725"),
                _currency("pdf", "pdf", "DU/2025/9"),
                _currency("html", "pdf", "DU/2024/1796"),
            )
        }
    )
    assert snapshot.source_format == "fallback"
    assert snapshot.fallback_acts == ("DU/2023/725", "DU/2024/1796"), (
        "an act that declared pdf and read pdf is not a substitution"
    )
