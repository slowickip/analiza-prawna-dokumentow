from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import batched
from pathlib import Path
from typing import Any, Literal

import httpx
from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient, QdrantClient, models

from contract_analyzer.corpus import (
    CORPUS_ALIAS,
    CORPUS_SNAPSHOTS_COLLECTION,
    DENSE_VECTOR_NAME,
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL,
    EMBEDDING_PROBE_TEXT,
    EMBEDDING_REQUEST_MAX_INPUTS,
    SPARSE_VECTOR_NAME,
    ActCurrency,
    CorpusBuildError,
    CorpusSnapshot,
    LegalUnit,
    ManifestAct,
    act_force_state,
    checked_unit_vectors,
    corpus_collection_name,
    corpus_snapshot_id,
    effective_embedding_provider,
    eli_act_url,
    embedding_space_probe,
    embedding_spaces_agree,
    load_manifest,
    path_to_url,
    sha256_bytes,
    snapshot_record_id,
    sparse_document_vector,
    unit_digest_of,
    validate_act_metadata,
    validate_amendments_not_carried,
)
from contract_analyzer.corpus import (
    act_identifier as build_act_identifier,
)
from contract_analyzer.corpus import (
    act_url as build_act_url,
)
from contract_analyzer.corpus.embeddings import arequest_vectors, async_embedding_client
from contract_analyzer.domain import ForceState
from seeder.eli import (
    ELI_BASE,
    decode_metadata,
    decode_struct,
    deduped_struct_article_paths,
    fetch_async,
    html_unit_from_article,
    parse_pdf_units,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SeedResult:
    """What one build did, including what it was billed.

    ``embedding_tokens`` is None where the provider reported none. A build encodes
    the whole corpus and a query encodes one phrase, so this is where the
    embedding spend of the artefact actually is.
    """

    total_units: int
    skipped_existing: int
    newly_embedded: int
    embedding_tokens: int | None
    collection: str
    snapshot_id: str
    rebuilt: bool


def _vectors_config() -> dict[str, models.VectorParams]:
    return {
        DENSE_VECTOR_NAME: models.VectorParams(
            size=EMBEDDING_DIMENSION,
            distance=models.Distance.COSINE,
        )
    }


def _sparse_vectors_config() -> dict[str, models.SparseVectorParams]:
    """Term weights, with Qdrant supplying the inverse document frequency.

    That is the one corpus-wide statistic the seeder cannot hold while it streams.
    """
    return {SPARSE_VECTOR_NAME: models.SparseVectorParams(modifier=models.Modifier.IDF)}


async def ensure_qdrant_collection_async(
    client: AsyncQdrantClient,
    collection: str = CORPUS_ALIAS,
) -> None:
    if not await client.collection_exists(collection):
        await client.create_collection(
            collection_name=collection,
            vectors_config=_vectors_config(),
            sparse_vectors_config=_sparse_vectors_config(),
        )
        for field in ("locator", "id", "content_hash"):
            await client.create_payload_index(
                collection_name=collection,
                field_name=field,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )


def ensure_qdrant_collection(
    client: QdrantClient | AsyncQdrantClient,
    collection: str = CORPUS_ALIAS,
) -> None:
    if isinstance(client, AsyncQdrantClient):
        asyncio.run(ensure_qdrant_collection_async(client, collection))
        return
    if not client.collection_exists(collection):
        client.create_collection(
            collection_name=collection,
            vectors_config=_vectors_config(),
            sparse_vectors_config=_sparse_vectors_config(),
        )
        for field in ("locator", "id", "content_hash"):
            client.create_payload_index(
                collection_name=collection,
                field_name=field,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )


async def get_existing_units_metadata_async(
    client: AsyncQdrantClient,
    collection: str = CORPUS_ALIAS,
) -> dict[str, dict[str, Any]]:
    existing: dict[str, dict[str, Any]] = {}
    offset: Any = None
    while True:
        points, next_offset = await client.scroll(
            collection_name=collection,
            limit=500,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for p in points:
            if p.payload and "id" in p.payload:
                existing[str(p.payload["id"])] = dict(p.payload)
        if next_offset is None:
            break
        offset = next_offset
    return existing


def _unit_point_id(unit_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, unit_id))


def _unit_payload(unit: LegalUnit) -> dict[str, Any]:
    return {
        "id": unit.id,
        "locator": unit.locator,
        "act_identifier": unit.act_identifier,
        "article_identifier": unit.article_identifier,
        "text": unit.text,
        "content_hash": unit.content_hash,
        "act_force_value": unit.act_force.value.value,
        "act_force_scope": unit.act_force.scope.value,
        "act_force_source_locator": unit.act_force.source_locator,
        "act_force_snapshot_date": (unit.act_force.snapshot_date.isoformat()),
        "provision_force_value": unit.provision_force.value.value,
        "provision_force_scope": unit.provision_force.scope.value,
        "provision_force_source_locator": (unit.provision_force.source_locator),
        "provision_force_snapshot_date": (
            unit.provision_force.snapshot_date.isoformat()
        ),
        "legal_status_date": unit.legal_status_date.isoformat(),
    }


# Measured on 2026-09-06 against hosted provider: eight concurrent requests
# maximize throughput.
EMBEDDING_CONCURRENCY = max(1, int(os.environ.get("EMBEDDING_CONCURRENCY", "8")))


async def _embed_batch_async(
    texts: list[str], client: AsyncOpenAI
) -> tuple[list[list[float]], int | None]:
    """Embed these texts, a bounded number of requests at a time.

    The request is the query path's in its async form; only the budget differs,
    and that is carried by the client rather than by this call. ``asyncio.gather``
    returns in submission order, so vectors come back on their own texts without
    any index bookkeeping, and the same length and width guard the query path
    applies runs before they are written.
    """
    limit = asyncio.Semaphore(EMBEDDING_CONCURRENCY)

    async def embed_group(
        group: tuple[str, ...],
    ) -> tuple[list[list[float]], int | None]:
        async with limit:
            return await arequest_vectors(client, list(group))

    tasks = [
        asyncio.create_task(embed_group(group))
        for group in batched(texts, EMBEDDING_REQUEST_MAX_INPUTS)
    ]
    try:
        done = await asyncio.gather(*tasks)
    except BaseException:
        pending = [task for task in tasks if not task.done()]
        if pending:
            logger.warning(
                "An embedding request failed; cancelling %d still in flight",
                len(pending),
            )
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    billed = [tokens for _, tokens in done if tokens is not None]
    rows = checked_unit_vectors([r for group, _ in done for r in group]).tolist()
    return rows, (sum(billed) if billed else None)


def _unit_vectors(unit: LegalUnit, dense: list[float]) -> dict[str, Any]:
    weights = sparse_document_vector(unit.text)
    return {
        DENSE_VECTOR_NAME: dense,
        SPARSE_VECTOR_NAME: models.SparseVector(
            indices=list(weights), values=list(weights.values())
        ),
    }


def _leaf_exception(exc: BaseException) -> BaseException:
    while isinstance(exc, BaseExceptionGroup):
        exc = exc.exceptions[0]
    return exc


async def _dispatch_unit(
    unit: LegalUnit,
    existing_metadata: dict[str, dict[str, Any]],
    counters: dict[str, int],
    payload_updates: list[models.OverwritePayloadOperation],
    embed_queue: asyncio.Queue[LegalUnit | None],
    push_task: asyncio.Task[Any] | None,
    written: list[tuple[str, str]] | None = None,
) -> None:
    counters["total"] += 1
    existing_payload = existing_metadata.get(unit.id)
    if (
        existing_payload is not None
        and existing_payload.get("content_hash") == unit.content_hash
    ):
        counters["skipped"] += 1
        current_payload = _unit_payload(unit)
        if current_payload != existing_payload:
            payload_updates.append(
                models.OverwritePayloadOperation(
                    overwrite_payload=models.SetPayload(
                        payload=current_payload,
                        points=[_unit_point_id(unit.id)],
                    )
                )
            )
            logger.debug(
                "Updating metadata payload for unit %s (content unchanged)",
                unit.id,
            )
        else:
            logger.debug("Skipping unchanged unit %s (cached)", unit.id)
        if written is not None:
            written.append((unit.locator, unit.content_hash))
    else:
        await _put_with_check(embed_queue, push_task, unit)


async def _flush_payload_updates(
    client: AsyncQdrantClient,
    collection: str,
    payload_updates: list[models.OverwritePayloadOperation],
) -> None:
    if not payload_updates:
        return
    logger.info(
        "Updating metadata payload for %d existing units in %s...",
        len(payload_updates),
        collection,
    )
    await client.batch_update_points(
        collection_name=collection,
        update_operations=list(payload_updates),
        wait=True,
    )
    payload_updates.clear()


async def _put_with_check(
    queue: asyncio.Queue[LegalUnit | None],
    push_task: asyncio.Task[Any] | None,
    item: LegalUnit | None,
) -> None:
    while True:
        if push_task is not None and push_task.done():
            exc = push_task.exception()
            if exc is not None:
                raise exc
        try:
            await asyncio.wait_for(queue.put(item), timeout=1.0)
            break
        except TimeoutError:
            continue


async def _push_worker(
    embed_queue: asyncio.Queue[LegalUnit | None],
    qdrant: AsyncQdrantClient,
    collection: str,
    embed_fn: Callable[[list[str]], Awaitable[list[list[float]]]],
    batch_size: int = 256,
    written: list[tuple[str, str]] | None = None,
) -> int:
    newly_embedded = 0
    buffer: list[LegalUnit] = []

    async def flush() -> None:
        nonlocal newly_embedded
        if not buffer:
            return
        batch = list(buffer)
        buffer.clear()

        total_chars = sum(len(u.text) for u in batch)
        logger.info(
            "Embedding batch of %d units (~%d chars, items %d..%d) "
            "for collection '%s'...",
            len(batch),
            total_chars,
            newly_embedded + 1,
            newly_embedded + len(batch),
            collection,
        )
        t0 = time.monotonic()
        passages = [u.text for u in batch]
        try:
            vectors = await embed_fn(passages)
        except Exception as exc:
            logger.error(
                "Failed to generate embeddings for batch of %d units: %s",
                len(batch),
                exc,
            )
            raise
        elapsed = time.monotonic() - t0
        logger.info(
            "Batch of %d units embedded in %.2fs. Upserting to %s...",
            len(batch),
            elapsed,
            collection,
        )

        points = [
            models.PointStruct(
                id=_unit_point_id(unit.id),
                vector=_unit_vectors(unit, vec),
                payload=_unit_payload(unit),
            )
            for unit, vec in zip(batch, vectors, strict=True)
        ]
        try:
            await qdrant.upsert(collection_name=collection, points=points, wait=True)
        except Exception as exc:
            logger.error(
                "Failed to upsert %d points to collection %s: %s",
                len(points),
                collection,
                exc,
            )
            raise

        if written is not None:
            written.extend((unit.locator, unit.content_hash) for unit in batch)
        newly_embedded += len(batch)
        logger.info(
            "Upserted batch of %d units to %s (embedded so far: %d)",
            len(batch),
            collection,
            newly_embedded,
        )

    while True:
        try:
            unit = await asyncio.wait_for(embed_queue.get(), timeout=0.5)
        except TimeoutError:
            await flush()
            continue

        if unit is None:
            await flush()
            embed_queue.task_done()
            break

        buffer.append(unit)
        embed_queue.task_done()
        if len(buffer) >= batch_size:
            await flush()

    return newly_embedded


async def _act_source_format(
    client: httpx.AsyncClient,
    act: ManifestAct,
    act_id: str,
    act_url: str,
    cache_kwargs: dict[str, Any],
) -> tuple[Literal["html", "pdf"], bytes | None]:
    """Which source format to read this act from, and its struct if that is HTML.

    ELI serves an act's HTML and its PDF from separate routes, and the HTML one
    has failed on its own: on 2026-09-06 /struct, /text.html and every article
    path under it returned 500 while /text.pdf answered normally. A build can
    still be made from the PDF, so it is.

    The whole act is decided here, before a single unit is written, because the
    alternative is a half-built one. The structure and the article texts are
    separate routes and either can be the one that is down -- notably with
    ELI_CACHE_DIR set, where a cached structure is served from disk while live
    article fetches fail -- so both are tried before committing to HTML.

    The fallback is never silent. The two source formats cut an act into
    different units, so the corpus takes its own unit digest and snapshot id
    rather than passing for the one HTML would have built, and the caller records
    which format was read per act.
    """
    if act.source_format != "html":
        return act.source_format, None
    try:
        struct_raw = await fetch_async(
            client, f"{act_url}/struct", accept="application/json", **cache_kwargs
        )
        # One article, to prove the text route answers too. The workers refetch
        # it, which ELI's disk cache absorbs.
        probe_paths = deduped_struct_article_paths(decode_struct(struct_raw), act_id)
        if probe_paths:
            await fetch_async(
                client,
                path_to_url(act_url, probe_paths[0][0]),
                accept="text/html",
                **cache_kwargs,
            )
    except CorpusBuildError as exc:
        if exc.code != "eli_unavailable":
            raise
        logger.warning(
            "ELI's HTML source format of %s is unavailable (%s); "
            "building this act from its PDF instead.",
            act_id,
            exc,
        )
        return "pdf", None
    return "html", struct_raw


async def _fetch_html_worker(
    task_queue: asyncio.Queue[
        tuple[str, tuple[str, ...], dict[str, Any], ManifestAct, ForceState, str] | None
    ],
    embed_queue: asyncio.Queue[LegalUnit | None],
    http_client: httpx.AsyncClient,
    existing_metadata: dict[str, dict[str, Any]],
    seen_ids: set[str],
    counters: dict[str, int],
    payload_updates: list[models.OverwritePayloadOperation],
    written: list[tuple[str, str]] | None = None,
    push_task: asyncio.Task[Any] | None = None,
    cache_dir: Path | None = None,
    refresh_cache: bool = False,
    no_cache: bool = False,
) -> None:
    while True:
        task = await task_queue.get()
        if task is None:
            break

        act_url, path, node, act, act_force, act_id = task
        try:
            url = path_to_url(act_url, path)
            cache_kwargs: dict[str, Any] = {}
            if cache_dir is not None:
                cache_kwargs["cache_dir"] = cache_dir
            if refresh_cache:
                cache_kwargs["refresh_cache"] = refresh_cache
            if no_cache:
                cache_kwargs["no_cache"] = no_cache
            html_bytes = await fetch_async(
                http_client, url, accept="text/html", **cache_kwargs
            )
            unit = html_unit_from_article(
                act_url=act_url,
                path=path,
                node=node,
                act=act,
                act_force=act_force,
                act_id=act_id,
                html_bytes=html_bytes,
            )
            if unit.id in seen_ids:
                raise CorpusBuildError(
                    "duplicate_unit_id", f"duplicate_unit_id: {unit.id}"
                )
            seen_ids.add(unit.id)
            await _dispatch_unit(
                unit=unit,
                existing_metadata=existing_metadata,
                counters=counters,
                payload_updates=payload_updates,
                embed_queue=embed_queue,
                push_task=push_task,
                written=written,
            )
        except Exception as exc:
            logger.error(
                "Error processing task for %s (%s): %s",
                act_id,
                "/".join(path),
                exc,
            )
            raise


async def _space_probe(
    embed_fn: Callable[[list[str]], Awaitable[list[list[float]]]] | None,
) -> tuple[float, ...]:
    """The vector space this build encodes in, through whichever encoder it uses."""
    if embed_fn is not None:
        return tuple(float(v) for v in (await embed_fn([EMBEDDING_PROBE_TEXT]))[0])
    return embedding_space_probe()


async def _published_snapshot_record(
    client: AsyncQdrantClient, collection: str
) -> CorpusSnapshot | None:
    """The record beside this collection, whatever state the collection is in.

    ``_published_snapshot`` answers "is there a finished corpus here" and says no
    when the point count disagrees. A rebuild empties the collection first, which
    is exactly the window where this record still has to be readable.
    """
    if not await client.collection_exists(CORPUS_SNAPSHOTS_COLLECTION):
        return None
    points = await client.retrieve(
        collection_name=CORPUS_SNAPSHOTS_COLLECTION,
        ids=[snapshot_record_id(collection)],
        with_payload=True,
    )
    if not points or not points[0].payload:
        return None
    return CorpusSnapshot.model_validate_json(points[0].payload["snapshot"])


async def _published_snapshot(
    client: AsyncQdrantClient, collection: str
) -> CorpusSnapshot | None:
    """Return the snapshot a finished build published for this collection, if any."""
    if not await client.collection_exists(collection):
        return None
    snapshot = await _published_snapshot_record(client, collection)
    if snapshot is None:
        return None
    stored = (await client.get_collection(collection)).points_count
    if stored != snapshot.unit_count:
        logger.warning(
            "Collection %s holds %s points against a recorded %s; rebuilding",
            collection,
            stored,
            snapshot.unit_count,
        )
        return None
    return snapshot


async def _verify_and_publish(
    client: AsyncQdrantClient,
    collection: str,
    snapshot: CorpusSnapshot,
    rebuild: bool = False,
) -> None:
    """Publish a built collection only once it holds exactly what was written.

    The snapshot id is content-addressed over the manifest, the units and the
    declared space, and deliberately does not fold in the probe vector: measured
    2026-09-06, two probes taken seconds apart from one provider differ by up to
    3.7e-03 per component -- one space by cosine, but no stable digest. Hashing it
    would change the id on every rebuild, which is the opposite of what it is for.

    That leaves one hole, closed here rather than in the id: a rebuild in a
    genuinely different space would otherwise overwrite the record under the same
    id, and runs citing that id would no longer agree on what they searched.
    Comparing the spaces is what tells a re-run apart from a substitution.
    """
    if snapshot.unit_count == 0:
        raise CorpusBuildError("empty_corpus", f"empty_corpus: {collection}")
    stored = (await client.get_collection(collection)).points_count
    if stored != snapshot.unit_count:
        raise CorpusBuildError(
            "incomplete_corpus",
            f"incomplete_corpus: {collection} holds {stored} of "
            f"{snapshot.unit_count} units",
        )
    existing = await _published_snapshot_record(client, collection)
    if (
        existing is not None
        and existing.embedding_space_probe
        and not embedding_spaces_agree(
            existing.embedding_space_probe, snapshot.embedding_space_probe
        )
    ):
        raise CorpusBuildError(
            "corpus_embedding_space_collision",
            (
                f"corpus_embedding_space_collision: {collection} already publishes "
                f"snapshot {existing.id} built in a different vector space. "
                f"Rebuilding over it would leave earlier runs citing an id whose "
                f"corpus they never searched."
            ),
        )
    if not await client.collection_exists(CORPUS_SNAPSHOTS_COLLECTION):
        await client.create_collection(
            collection_name=CORPUS_SNAPSHOTS_COLLECTION,
            # Never searched, but a Qdrant collection must declare some space.
            vectors_config=models.VectorParams(size=1, distance=models.Distance.COSINE),
        )
    await client.upsert(
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
    await _publish_alias(client, collection, rebuild=rebuild)
    logger.info(
        "Published %s as snapshot %s (%d units)",
        collection,
        snapshot.id,
        snapshot.unit_count,
    )


async def _publish_alias(
    client: AsyncQdrantClient, collection: str, rebuild: bool = False
) -> None:
    """Point the name the backend opens at this collection, atomically."""
    if collection == CORPUS_ALIAS:
        return

    aliases_resp = await client.get_aliases()
    existing_alias_map = {a.alias_name: a.collection_name for a in aliases_resp.aliases}

    if CORPUS_ALIAS in existing_alias_map:
        if existing_alias_map[CORPUS_ALIAS] == collection:
            logger.info("Alias %s already points to %s", CORPUS_ALIAS, collection)
            return
        logger.info(
            "Atomically updating alias %s: %s -> %s",
            CORPUS_ALIAS,
            existing_alias_map[CORPUS_ALIAS],
            collection,
        )
        await client.update_collection_aliases(
            change_aliases_operations=[
                models.DeleteAliasOperation(
                    delete_alias=models.DeleteAlias(alias_name=CORPUS_ALIAS)
                ),
                models.CreateAliasOperation(
                    create_alias=models.CreateAlias(
                        collection_name=collection, alias_name=CORPUS_ALIAS
                    )
                ),
            ]
        )
        return

    if await client.collection_exists(CORPUS_ALIAS):
        if rebuild:
            logger.warning(
                "Physical collection '%s' exists without alias mapping; "
                "removing on rebuild to establish alias to %s",
                CORPUS_ALIAS,
                collection,
            )
            await client.delete_collection(CORPUS_ALIAS)
        else:
            raise CorpusBuildError(
                "physical_collection_conflict",
                f"Cannot publish alias '{CORPUS_ALIAS}': a physical "
                f"collection named '{CORPUS_ALIAS}' already exists. "
                f"Run with --rebuild to replace it, or remove it manually.",
            )

    await client.update_collection_aliases(
        change_aliases_operations=[
            models.CreateAliasOperation(
                create_alias=models.CreateAlias(
                    collection_name=collection, alias_name=CORPUS_ALIAS
                )
            )
        ]
    )


async def seed_qdrant_async(
    manifest_path: Path,
    qdrant_url: str | None = None,
    *,
    collection: str | None = None,
    num_workers: int = 8,
    batch_size: int = 256,
    rebuild: bool = False,
    qdrant_client: AsyncQdrantClient | None = None,
    embed_fn: (Callable[[list[str]], Awaitable[list[list[float]]]] | None) = None,
    cache_dir: Path | None = None,
    refresh_cache: bool = False,
    no_cache: bool = False,
) -> SeedResult:
    """Build one manifest's corpus into its own collection and publish it.

    The collection a manifest builds is named after that manifest, so a changed
    manifest never writes into the corpus already in use and a unit dropped from the
    manifest cannot survive as a retrievable leftover. The build starts from an empty
    collection, and the alias the backend reads moves onto it only after the stored
    point count matches what was written, so a failed build leaves the previous corpus
    published rather than a half-filled one.

    Re-running against an already published corpus is a no-op: the snapshot record is
    the evidence that a build finished, and ``rebuild`` is what discards it
    deliberately.
    """
    if num_workers < 1:
        raise ValueError("num_workers must be at least 1")
    if qdrant_client is not None:
        qdrant = qdrant_client
    elif qdrant_url is not None:
        qdrant = AsyncQdrantClient(url=qdrant_url)
    else:
        raise ValueError("Either qdrant_url or qdrant_client must be provided")

    manifest = load_manifest(manifest_path)
    manifest_hash = sha256_bytes(manifest_path.read_bytes())
    target = collection or corpus_collection_name(manifest_hash)

    published = await _published_snapshot(qdrant, target)
    if published is not None and not rebuild:
        await _publish_alias(qdrant, target, rebuild=rebuild)
        logger.info(
            "Collection %s already publishes snapshot %s; nothing to build",
            target,
            published.id,
        )
        return SeedResult(
            total_units=published.unit_count,
            skipped_existing=published.unit_count,
            newly_embedded=0,
            embedding_tokens=None,
            collection=target,
            snapshot_id=published.id,
            rebuilt=False,
        )

    target_exists = await qdrant.collection_exists(target)
    if rebuild and target_exists:
        # Before the delete, not at publish time: a rebuild builds into the empty
        # collection, so a guard that fires when the new corpus is ready is
        # guarding vectors destroyed twenty minutes earlier.
        existing = await _published_snapshot_record(qdrant, target)
        if existing is not None and existing.embedding_space_probe:
            probe = await _space_probe(embed_fn)
            if not embedding_spaces_agree(existing.embedding_space_probe, probe):
                raise CorpusBuildError(
                    "corpus_embedding_space_collision",
                    (
                        f"corpus_embedding_space_collision: {target} publishes "
                        f"snapshot {existing.id}, built in a different vector "
                        f"space than this one encodes in. Refusing to discard it: "
                        f"build the new space beside it instead."
                    ),
                )
        logger.info("Discarding collection %s for explicit rebuild", target)
        await qdrant.delete_collection(target)
        target_exists = False

    await ensure_qdrant_collection_async(qdrant, target)
    existing_metadata: dict[str, dict[str, Any]] = {}
    if target_exists and not rebuild:
        existing_metadata = await get_existing_units_metadata_async(qdrant, target)
        if existing_metadata:
            logger.info(
                "Collection %s exists with %d units. Operating in delta mode.",
                target,
                len(existing_metadata),
            )
        else:
            logger.info(
                "Collection %s exists but is blank. Operating in full build mode.",
                target,
            )
    else:
        logger.info("Building collection %s from scratch.", target)

    written: list[tuple[str, str]] = []
    act_currency: list[ActCurrency] = []

    observed_on = datetime.now(tz=UTC).date()
    base_act_cache: dict[str, dict[str, Any]] = {}
    seen_ids: set[str] = set()
    counters: dict[str, int] = {"total": 0, "skipped": 0}
    payload_updates: list[models.OverwritePayloadOperation] = []

    embed_queue: asyncio.Queue[LegalUnit | None] = asyncio.Queue(maxsize=64)
    # Per build, not global: an injected encoder bills nothing, and None says so.
    embedding_tokens: int | None = None

    async with (
        httpx.AsyncClient(base_url=ELI_BASE, timeout=60.0) as eli_client,
        async_embedding_client() as embed_client,
    ):

        async def _default_embed_fn(texts: list[str]) -> list[list[float]]:
            vectors, tokens = await _embed_batch_async(texts, embed_client)
            if tokens is not None:
                nonlocal embedding_tokens
                embedding_tokens = (embedding_tokens or 0) + tokens
            return vectors

        active_embed_fn = embed_fn or _default_embed_fn

        push_task = asyncio.create_task(
            _push_worker(
                embed_queue=embed_queue,
                qdrant=qdrant,
                collection=target,
                embed_fn=active_embed_fn,
                batch_size=batch_size,
                written=written,
            )
        )

        cache_kwargs: dict[str, Any] = {}
        if cache_dir is not None:
            cache_kwargs["cache_dir"] = cache_dir
        if refresh_cache:
            cache_kwargs["refresh_cache"] = refresh_cache
        if no_cache:
            cache_kwargs["no_cache"] = no_cache

        for act in manifest.acts:
            act_url = build_act_url(act.publisher, act.year, act.position)
            act_id = build_act_identifier(act.publisher, act.year, act.position)
            metadata_raw = await fetch_async(
                eli_client, act_url, accept="application/json", **cache_kwargs
            )
            metadata = decode_metadata(metadata_raw, act_url)

            validate_act_metadata(metadata, act, act_url)

            if act.base_act not in base_act_cache:
                base_raw = await fetch_async(
                    eli_client,
                    eli_act_url(act.base_act),
                    accept="application/json",
                    **cache_kwargs,
                )
                base_act_cache[act.base_act] = decode_metadata(
                    base_raw, eli_act_url(act.base_act)
                )

            base_metadata = base_act_cache[act.base_act]
            validate_amendments_not_carried(
                act=act,
                act_id=act_id,
                base_metadata=base_metadata,
                corpus_target_date=manifest.corpus_target_date,
            )

            base_act_url = eli_act_url(act.base_act)
            act_force = act_force_state(base_metadata, observed_on, base_act_url)

            format_used, struct_raw = await _act_source_format(
                eli_client, act, act_id, act_url, cache_kwargs
            )

            act_currency.append(
                ActCurrency(
                    act_identifier=act_id,
                    base_act=act.base_act,
                    legal_status_date=act.legal_status_date,
                    amendments_not_carried=act.amendments_not_carried,
                    repeated_articles=act.repeated_articles,
                    source_format_declared=act.source_format,
                    source_format_used=format_used,
                )
            )

            if format_used == "html":
                assert struct_raw is not None
                path_entries = deduped_struct_article_paths(
                    decode_struct(struct_raw), act_id
                )
                if not path_entries:
                    raise CorpusBuildError(
                        "malformed_structure", "malformed_structure: no units"
                    )
                logger.info("%s: %d articles read from HTML", act_id, len(path_entries))

                task_queue: asyncio.Queue[
                    tuple[
                        str,
                        tuple[str, ...],
                        dict[str, Any],
                        ManifestAct,
                        ForceState,
                        str,
                    ]
                    | None
                ] = asyncio.Queue()

                for path, node in path_entries:
                    task_queue.put_nowait((act_url, path, node, act, act_force, act_id))
                for _ in range(num_workers):
                    task_queue.put_nowait(None)

                worker_tasks: list[asyncio.Task[None]] = []

                def _abort_workers(
                    _task: asyncio.Task[Any],
                    tasks: list[asyncio.Task[None]] = worker_tasks,
                ) -> None:
                    if _task.cancelled() or _task.exception() is not None:
                        for w in tasks:
                            w.cancel()

                push_task.add_done_callback(_abort_workers)
                try:
                    async with asyncio.TaskGroup() as tg:
                        worker_tasks.extend(
                            [
                                tg.create_task(
                                    _fetch_html_worker(
                                        task_queue=task_queue,
                                        embed_queue=embed_queue,
                                        http_client=eli_client,
                                        existing_metadata=existing_metadata,
                                        seen_ids=seen_ids,
                                        counters=counters,
                                        payload_updates=payload_updates,
                                        written=written,
                                        push_task=push_task,
                                        cache_dir=cache_dir,
                                        refresh_cache=refresh_cache,
                                        no_cache=no_cache,
                                    )
                                )
                                for _ in range(num_workers)
                            ]
                        )
                except* BaseException as eg:
                    if push_task.done():
                        if push_task.cancelled():
                            raise asyncio.CancelledError() from None
                        push_exc = push_task.exception()
                        if push_exc is not None:
                            raise push_exc from None
                    raise _leaf_exception(eg) from None
                finally:
                    push_task.remove_done_callback(_abort_workers)

                await _flush_payload_updates(qdrant, target, payload_updates)

            else:
                pdf_bytes = await fetch_async(
                    eli_client,
                    f"{act_url}/text.pdf",
                    accept="application/pdf",
                    **cache_kwargs,
                )
                pdf_units, pdf_repeats = parse_pdf_units(
                    pdf_bytes=pdf_bytes,
                    act=act,
                    act_id=act_id,
                    act_url=act_url,
                    act_force=act_force,
                    seen_ids=seen_ids,
                    pins_apply=act.source_format == "pdf",
                )
                logger.info("%s: %d articles read from PDF", act_id, len(pdf_units))
                if act.source_format != "pdf":
                    act_currency[-1] = act_currency[-1].model_copy(
                        update={"repeated_articles": pdf_repeats}
                    )
                for unit in pdf_units:
                    await _dispatch_unit(
                        unit=unit,
                        existing_metadata=existing_metadata,
                        counters=counters,
                        payload_updates=payload_updates,
                        embed_queue=embed_queue,
                        push_task=push_task,
                        written=written,
                    )

                await _flush_payload_updates(qdrant, target, payload_updates)

        await _put_with_check(embed_queue, push_task, None)
        newly_embedded = await push_task
        # Through the same encoder the corpus went through, so an injected one is
        # fingerprinted as itself rather than as the configured service.
        space_probe = await _space_probe(active_embed_fn)

    await _flush_payload_updates(qdrant, target, payload_updates)

    dropped_ids = set(existing_metadata.keys()) - seen_ids
    if dropped_ids:
        logger.info(
            "Purging %d stale units dropped from manifest from %s",
            len(dropped_ids),
            target,
        )
        point_ids: list[models.ExtendedPointId] = [
            _unit_point_id(uid) for uid in dropped_ids
        ]
        await qdrant.delete(
            collection_name=target,
            points_selector=models.PointIdsList(points=point_ids),
            wait=True,
        )

    digest = unit_digest_of(written)
    snapshot = CorpusSnapshot(
        id=corpus_snapshot_id(manifest_hash, digest),
        built_at=datetime.now(tz=UTC),
        manifest_hash=manifest_hash,
        model_name=EMBEDDING_MODEL,
        embedding_space_probe=space_probe,
        embedding_provider=effective_embedding_provider(),
        # Qdrant holds the corpus, so there are no files to hash; the unit digest
        # and the stored point count secure what a file hash would have.
        file_hashes={},
        unit_count=len(written),
        act_identifiers=tuple(
            build_act_identifier(act.publisher, act.year, act.position)
            for act in manifest.acts
        ),
        corpus_target_date=manifest.corpus_target_date,
        act_currency=tuple(act_currency),
        unit_digest=digest,
    )
    await _verify_and_publish(qdrant, target, snapshot, rebuild=rebuild)
    fallbacks = snapshot.fallback_acts
    if fallbacks:
        logger.warning(
            "Corpus %s read %d of %d acts from a source format its manifest did "
            "not declare: %s. It is servable, but it is not the corpus the "
            "manifest describes.",
            snapshot.id,
            len(fallbacks),
            len(act_currency),
            ", ".join(fallbacks),
        )
    logger.info(
        "Built %s: %d units, %d newly embedded, %d unchanged, billed %s "
        "embedding tokens",
        snapshot.id,
        counters["total"],
        newly_embedded,
        counters["skipped"],
        embedding_tokens if embedding_tokens is not None else "no",
    )

    return SeedResult(
        total_units=counters["total"],
        skipped_existing=counters["skipped"],
        newly_embedded=newly_embedded,
        embedding_tokens=embedding_tokens,
        collection=target,
        snapshot_id=snapshot.id,
        rebuilt=True,
    )


def seed_qdrant(
    manifest_path: Path,
    qdrant_url: str | None = None,
    *,
    collection: str | None = None,
    num_workers: int = 8,
    batch_size: int = 256,
    rebuild: bool = False,
    qdrant_client: AsyncQdrantClient | None = None,
    embed_fn: (Callable[[list[str]], Awaitable[list[list[float]]]] | None) = None,
    cache_dir: Path | None = None,
    refresh_cache: bool = False,
    no_cache: bool = False,
) -> SeedResult:
    """``seed_qdrant_async`` for the CLI, which has no event loop of its own."""
    return asyncio.run(
        seed_qdrant_async(
            manifest_path=manifest_path,
            qdrant_url=qdrant_url,
            collection=collection,
            num_workers=num_workers,
            batch_size=batch_size,
            rebuild=rebuild,
            qdrant_client=qdrant_client,
            embed_fn=embed_fn,
            cache_dir=cache_dir,
            refresh_cache=refresh_cache,
            no_cache=no_cache,
        )
    )
