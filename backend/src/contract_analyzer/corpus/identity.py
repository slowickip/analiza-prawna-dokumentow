"""Corpus hashing and snapshot identity."""

from __future__ import annotations

import hashlib
import uuid
from collections.abc import Iterable
from typing import Any

from contract_analyzer.corpus.embeddings import (
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL,
    effective_embedding_provider,
)
from contract_analyzer.corpus.errors import CorpusBuildError, CorpusIntegrityError
from contract_analyzer.corpus.models import CorpusSnapshot
from contract_analyzer.corpus.sparse import SPARSE_POLICY_VERSION


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def validate_expected_hash(body: bytes, expected_hash: str | None) -> None:
    if expected_hash is None:
        return
    actual = sha256_bytes(body)
    if actual != expected_hash.lower():
        raise CorpusBuildError(
            "source_hash_mismatch",
            f"source_hash_mismatch: expected {expected_hash}, got {actual}",
        )


def unit_digest_of(pairs: Iterable[tuple[str, str]]) -> str:
    """Digest a corpus by its (locator, content hash) pairs, in locator order."""
    hasher = hashlib.sha256()
    for locator, content_hash in sorted(pairs):
        hasher.update(f"{locator}\n{content_hash}\n".encode())
    return hasher.hexdigest()


# Corpus collections are named by specification;
# CORPUS_ALIAS points to the verified collection.
CORPUS_ALIAS = "legal_units"
CORPUS_SNAPSHOTS_COLLECTION = "corpus_snapshots"
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"


def embedding_space_fingerprint() -> str:
    """A short hash of the vector space this deployment declares it encodes in.

    Served on /config and carried as a run parity dimension, so a batch cannot
    silently span two spaces. Model, width and upstream, because those are stable
    and the probe vector is not -- naming has to be built on something a rebuild
    reproduces. Necessary and not sufficient: the probe on the snapshot is what
    compares the space that actually answered.
    """
    return sha256_bytes(_embedding_space_spec().encode("utf-8"))[:16]


def _embedding_space_spec() -> str:
    return f"{EMBEDDING_MODEL}\n{EMBEDDING_DIMENSION}\n{effective_embedding_provider()}"


def corpus_collection_name(manifest_hash: str) -> str:
    """Name the collection one corpus specification builds."""
    spec = f"{manifest_hash}\n{_embedding_space_spec()}\n{SPARSE_POLICY_VERSION}"
    return f"{CORPUS_ALIAS}_{sha256_bytes(spec.encode('utf-8'))[:16]}"


def corpus_snapshot_id(manifest_hash: str, unit_digest: str) -> str:
    """Identify a corpus by what is in it, not by when it was built.

    Two builds of the same manifest against unchanged sources produce the same id, and
    any change to a unit's text produces a different one. The build timestamp is
    recorded in the snapshot but kept out of the id, so a rebuild does not invalidate a
    pin that is still describing the same corpus.
    """
    spec = (
        f"{manifest_hash}\n{unit_digest}\n{_embedding_space_spec()}\n"
        f"{SPARSE_POLICY_VERSION}"
    )
    return sha256_bytes(spec.encode("utf-8"))[:24]


def snapshot_record_id(collection: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"corpus-snapshot:{collection}"))


def resolve_corpus_collection(client: Any, name: str) -> str:
    """Resolve an alias to the collection it publishes, or pass a real name through.

    Aliases are consulted first because a Qdrant client answers ``collection_exists``
    for an alias as readily as for a collection, and it is the collection behind the
    alias that carries the snapshot record.
    """
    for alias in client.get_aliases().aliases:
        if alias.alias_name == name:
            return str(alias.collection_name)
    if client.collection_exists(name):
        return name
    raise CorpusIntegrityError(
        "corpus_unpublished", f"no Qdrant collection or alias named {name}"
    )


def read_snapshot_record(client: Any, collection: str) -> CorpusSnapshot | None:
    if not client.collection_exists(CORPUS_SNAPSHOTS_COLLECTION):
        return None
    points = client.retrieve(
        collection_name=CORPUS_SNAPSHOTS_COLLECTION,
        ids=[snapshot_record_id(collection)],
        with_payload=True,
    )
    if not points or not points[0].payload:
        return None
    return CorpusSnapshot.model_validate_json(points[0].payload["snapshot"])
