"""Read-only Qdrant corpus index."""

from __future__ import annotations

import threading
from typing import Any, NamedTuple

from openai import OpenAI
from qdrant_client import QdrantClient, models

from contract_analyzer.corpus.embeddings import (
    EMBEDDING_MODEL,
    _encode,
    embed_query_text,
    embedding_client,
    embedding_space_probe,
    embedding_spaces_agree,
)
from contract_analyzer.corpus.errors import CorpusIntegrityError
from contract_analyzer.corpus.identity import (
    CORPUS_ALIAS,
    DENSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
    read_snapshot_record,
    resolve_corpus_collection,
)
from contract_analyzer.corpus.models import CorpusSnapshot, LegalUnit
from contract_analyzer.corpus.retrieval import _fusion_order, _legal_unit_from_payload
from contract_analyzer.corpus.sparse import sparse_query_vector
from contract_analyzer.domain import RetrievalCandidate


class SearchResult(NamedTuple):
    """A fused candidate with the legal unit search already decoded for it."""

    candidate: RetrievalCandidate
    unit: LegalUnit


class QdrantCorpusIndex:
    """Read access to one frozen corpus published in Qdrant.

    The index never invents its own identity. ``CorpusSnapshot`` is read back from the
    record the seeder wrote beside the collection, so ``snapshot.id`` names the corpus
    that was actually built and ``corpus_snapshot_id`` on a ``RunRecord`` is a fact
    about the data rather than a constant every corpus shares. Until that record is
    readable there is no snapshot, and asking for one fails.

    ``expected_snapshot_id`` pins a deployment to one corpus: with it set, verification
    refuses any collection whose record says something else.
    """

    def __init__(
        self,
        *,
        client: Any,
        collection: str = CORPUS_ALIAS,
        snapshot: CorpusSnapshot | None = None,
        expected_snapshot_id: str | None = None,
    ) -> None:
        self._client = client
        self._collection = collection
        self._snapshot = snapshot
        self._expected_snapshot_id = expected_snapshot_id
        self._resolved: str | None = collection if snapshot is not None else None
        self._probe: tuple[float, ...] | None = None
        self._embed_client: OpenAI | None = None
        self._embed_client_lock = threading.Lock()

    @property
    def _embedding_client(self) -> OpenAI:
        """One client for the life of the index, created on first use.

        A search used to open a connection, send one request and close it, at a
        handshake each. The lock matters because searches run in worker threads,
        where two first uses would otherwise build two clients and leak one.
        """
        with self._embed_client_lock:
            if self._embed_client is None:
                self._embed_client = embedding_client()
            return self._embed_client

    def close(self) -> None:
        """Close the embedding client if one was opened. Shutdown calls this."""
        with self._embed_client_lock:
            if self._embed_client is not None:
                self._embed_client.close()
                self._embed_client = None

    @property
    def _own_space(self) -> tuple[float, ...]:
        """This process's probe vector, encoded once and kept.

        Readiness is polled; the vector space behind a configured endpoint is not
        a property of the minute, so this is bought once per process rather than
        every thirty seconds.
        """
        if self._probe is None:
            self._probe = embedding_space_probe(client=self._embedding_client)
        return self._probe

    @property
    def snapshot(self) -> CorpusSnapshot:
        if self._snapshot is None:
            self._load_snapshot()
        assert self._snapshot is not None
        return self._snapshot

    @property
    def collection(self) -> str:
        if self._resolved is None:
            self._resolved = resolve_corpus_collection(self._client, self._collection)
        return self._resolved

    @classmethod
    def open(
        cls,
        qdrant_url: str,
        collection: str = CORPUS_ALIAS,
        snapshot: CorpusSnapshot | None = None,
        expected_snapshot_id: str | None = None,
    ) -> QdrantCorpusIndex:
        client = QdrantClient(url=qdrant_url)
        return cls(
            client=client,
            collection=collection,
            snapshot=snapshot,
            expected_snapshot_id=expected_snapshot_id,
        )

    def _load_snapshot(self) -> CorpusSnapshot:
        try:
            resolved = resolve_corpus_collection(self._client, self._collection)
            record = read_snapshot_record(self._client, resolved)
        except CorpusIntegrityError:
            raise
        except Exception as exc:
            raise CorpusIntegrityError("qdrant_unavailable", str(exc)) from exc
        if record is None:
            raise CorpusIntegrityError(
                "corpus_snapshot_missing",
                f"no snapshot record for Qdrant collection {resolved}",
            )
        self._resolved = resolved
        self._snapshot = record
        return record

    def verify(self) -> None:
        """Check that the published collection is the frozen corpus it claims to be."""
        snapshot = self._load_snapshot()
        try:
            points_count = self._client.get_collection(self.collection).points_count
        except Exception as exc:
            raise CorpusIntegrityError("qdrant_unavailable", str(exc)) from exc
        if snapshot.unit_count == 0 or points_count == 0:
            raise CorpusIntegrityError(
                "corpus_empty", f"Qdrant collection {self.collection} is empty"
            )
        if points_count != snapshot.unit_count:
            raise CorpusIntegrityError(
                "corpus_incomplete",
                f"{self.collection} holds {points_count} points, "
                f"snapshot {snapshot.id} declares {snapshot.unit_count}",
            )
        # Name first: free, and it names the likelier mistake precisely.
        if snapshot.model_name != EMBEDDING_MODEL:
            raise CorpusIntegrityError(
                "corpus_model_mismatch",
                f"snapshot {snapshot.id} was embedded with {snapshot.model_name}, "
                f"this build queries with {EMBEDDING_MODEL}",
            )
        # Then the space itself: one identifier has already stood for two spaces
        # 0.53 cosine apart, which answers every query with worse candidates and
        # no error at all.
        if not snapshot.embedding_space_probe:
            raise CorpusIntegrityError(
                "corpus_embedding_space_unrecorded",
                f"snapshot {snapshot.id} was built before the embedding space was "
                f"recorded; rebuild the corpus to publish one",
            )
        if not embedding_spaces_agree(self._own_space, snapshot.embedding_space_probe):
            raise CorpusIntegrityError(
                "corpus_embedding_space_mismatch",
                f"snapshot {snapshot.id} was built in a different vector space "
                f"than this deployment encodes queries in",
            )
        if (
            self._expected_snapshot_id is not None
            and snapshot.id != self._expected_snapshot_id
        ):
            raise CorpusIntegrityError(
                "corpus_snapshot_mismatch",
                f"{self.collection} publishes snapshot {snapshot.id}, "
                f"this deployment is pinned to {self._expected_snapshot_id}",
            )

    def read(self, locator: str) -> LegalUnit:
        return self._read_one("locator", locator)

    def _read_by_id(self, unit_id: str) -> LegalUnit:
        return self._read_one("id", unit_id)

    def _read_one(self, field: str, value: str) -> LegalUnit:
        points, _ = self._client.scroll(
            collection_name=self.collection,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key=field, match=models.MatchValue(value=value)
                    )
                ]
            ),
            limit=1,
            with_payload=True,
        )
        if not points:
            raise KeyError(value)
        return _legal_unit_from_payload(points[0].payload)

    def search(
        self,
        query: str,
        limit: int,
    ) -> list[SearchResult]:
        """Retrieve over both limbs and fuse them, as the file-backed index did.

        The limbs are queried separately rather than through Qdrant's own fusion so
        that each candidate keeps the score its limb gave it, and so that one
        ``_fusion_order`` governs both stores.
        """
        query_terms = sparse_query_vector(query)
        sparse_points: list[Any] = []
        if query_terms:
            sparse_points = self._client.query_points(
                collection_name=self.collection,
                query=models.SparseVector(
                    indices=list(query_terms), values=list(query_terms.values())
                ),
                using=SPARSE_VECTOR_NAME,
                limit=limit,
                with_payload=True,
            ).points
        sparse_scores = {
            str(point.payload["id"]): float(point.score)
            for point in sparse_points
            if point.payload
        }
        sparse_ranks = {
            unit_id: rank for rank, unit_id in enumerate(sparse_scores, start=1)
        }

        query_vector = _encode(
            [embed_query_text(query)], client=self._embedding_client
        )[0]
        dense_points = self._client.query_points(
            collection_name=self.collection,
            query=query_vector.tolist(),
            using=DENSE_VECTOR_NAME,
            limit=limit,
            with_payload=True,
        ).points
        dense_scores = {
            str(point.payload["id"]): float(point.score)
            for point in dense_points
            if point.payload
        }
        dense_ranks = {
            unit_id: rank for rank, unit_id in enumerate(dense_scores, start=1)
        }

        point_payloads = {
            str(point.payload["id"]): point.payload
            for point in [*sparse_points, *dense_points]
            if point.payload and "id" in point.payload
        }

        results: list[SearchResult] = []
        for rank, unit_id in enumerate(
            _fusion_order(sparse_ranks, dense_ranks)[:limit], start=1
        ):
            unit = _legal_unit_from_payload(point_payloads[unit_id])
            candidate = RetrievalCandidate(
                locator=unit.locator,
                snapshot_id=self.snapshot.id,
                act_identifier=unit.act_identifier,
                act_force=unit.act_force,
                provision_force=unit.provision_force,
                rank=rank,
                sparse_score=sparse_scores.get(unit_id, 0.0),
                dense_score=dense_scores.get(unit_id, 0.0),
            )
            results.append(SearchResult(candidate=candidate, unit=unit))
        return results
