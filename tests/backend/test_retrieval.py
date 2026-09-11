from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch
from uuid import uuid4

import numpy as np
import pytest

from contract_analyzer import corpus as corpus_module
from contract_analyzer.agents.retrieval import run_search
from contract_analyzer.agents.scheduler import FINDER_SNIPPET_CHARS
from contract_analyzer.agents.session import RunRequest
from contract_analyzer.agents.state import ActiveRun
from contract_analyzer.budget import BudgetTracker
from contract_analyzer.corpus import QdrantCorpusIndex, _fusion_order
from contract_analyzer.corpus import embeddings as corpus_embeddings
from contract_analyzer.domain import ArmCode, ForceScope


@pytest.fixture
def index(built_corpus: QdrantCorpusIndex) -> QdrantCorpusIndex:
    return built_corpus


def test_hybrid_search_returns_auditable_candidate(index: QdrantCorpusIndex) -> None:
    result = index.search("wypowiedzenie najmu", 5)[0].candidate
    assert result.locator.startswith("https://api.sejm.gov.pl/eli/")
    assert result.snapshot_id == index.snapshot.id
    # Both scores are typed float and stay 0.0 for a limb that never ranked the
    # candidate, so 0.0 is what "this limb contributed nothing" looks like. Asserting
    # they are not None asserted the type annotation, and stayed green throughout the
    # period the sparse limb matched nothing.
    assert result.sparse_score != 0.0
    assert result.dense_score != 0.0


def _capture_embed(monkeypatch: pytest.MonkeyPatch, seen: list[str]) -> None:
    original = corpus_module._embed

    def capture(texts: list[str], client: object | None = None) -> np.ndarray:
        seen.extend(texts)
        return original(texts, client=client)

    monkeypatch.setattr(corpus_module, "_embed", capture)


def test_search_sends_the_query_under_its_task_instruction(
    index: QdrantCorpusIndex, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[str] = []
    _capture_embed(monkeypatch, seen)
    index.search("wypowiedzenie najmu", 3)
    assert seen[0].startswith(corpus_module.EMBEDDING_QUERY_PREFIX)


def test_read_returns_legal_unit_with_locator(index: QdrantCorpusIndex) -> None:
    locator = "https://api.sejm.gov.pl/eli/acts/DU/2023/725/text.html/arti=11"
    unit = index.read(locator)
    assert unit.locator == locator
    assert unit.text.strip()
    assert unit.act_identifier == "DU/2023/725"


def test_act_level_candidate_scope_pinned(index: QdrantCorpusIndex) -> None:
    candidate = index.search("wypowiedzenie najmu", 1)[0].candidate
    assert candidate.act_force.scope is ForceScope.ACT
    assert candidate.provision_force.scope is ForceScope.PROVISION


def test_long_query_returns_sparse_candidates(index: QdrantCorpusIndex) -> None:
    filler = " ".join(f"zztoken{i}" for i in range(40))
    query = f"wypowiedzenie najmu lokalu mieszkalnego {filler}"
    results = index.search(query, 5)
    assert any(result.candidate.sparse_score != 0.0 for result in results)


def test_fts5_keyword_tokens_do_not_break_search(index: QdrantCorpusIndex) -> None:
    query = "wypowiedzenie OR NEAR AND NOT najmu"
    results = index.search(query, 5)
    assert results


def test_dense_encode_receives_plain_query_without_or_expression(
    index: QdrantCorpusIndex, monkeypatch: pytest.MonkeyPatch
) -> None:
    query = "wypowiedzenie najmu lokalu mieszkalnego"
    seen: list[str] = []
    _capture_embed(monkeypatch, seen)
    index.search(query, 3)
    assert seen == [f"{corpus_module.EMBEDDING_QUERY_PREFIX}{query}"]
    assert " OR " not in seen[0]


def test_sparse_and_dense_receive_different_query_forms(
    index: QdrantCorpusIndex, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The limbs are fed different strings, and must stay that way.

    The dense limb carries the model's task instruction; the sparse limb is
    term frequencies over the bare words. Feeding either one the other's form
    is the kind of drift that costs recall without raising anything.
    """
    query = "wypowiedzenie najmu"
    encode_inputs: list[str] = []
    _capture_embed(monkeypatch, encode_inputs)
    index.search(query, 3)

    dense_input = encode_inputs[0]
    assert dense_input == f"{corpus_module.EMBEDDING_QUERY_PREFIX}{query}"

    sparse_terms = corpus_module.sparse_query_vector(query)
    assert sparse_terms, "the sparse limb saw no terms"
    # The instruction words must not enter the sparse limb: they are in every
    # query, so they carry no signal and would only dilute the term weights.
    instruction_terms = corpus_module.sparse_query_vector(
        corpus_module.EMBEDDING_QUERY_INSTRUCTION
    )
    assert not (set(sparse_terms) & set(instruction_terms))


def test_fusion_tie_break_does_not_favour_the_older_act() -> None:
    older = "DU/2023/725:chpt=1/arti=1"
    newer = "DU/2026/795:article=659"
    # Both limbs ranked both units, in opposite order, so the fused scores are equal
    # (1/61 + 1/62 either way) and only the tie-break separates them. Ordering ties by
    # unit id puts every DU/2023 unit ahead of every DU/2026 one, whatever was asked.
    assert _fusion_order({newer: 1, older: 2}, {older: 1, newer: 2}) == [newer, older]


def test_the_fixture_corpus_is_one_a_measured_run_would_accept(
    built_corpus: QdrantCorpusIndex,
) -> None:
    """The suite must not stand entirely on the class production refuses.

    An empty act_currency reads as 'unrecorded', which run_evaluation.py declines
    for a registered measurement beside an outright fallback. The fixture left it
    empty, so every test using this corpus exercised a state no measured run would
    accept, and the as_declared path the gate turns on was never built here.
    """
    snapshot = built_corpus.snapshot
    assert snapshot.act_currency, "the fixture corpus records no acts at all"
    assert snapshot.source_format == "as_declared", snapshot.source_format
    assert snapshot.fallback_acts == ()


def test_two_searches_on_one_index_construct_one_embedding_client(
    built_corpus: QdrantCorpusIndex,
) -> None:
    index = QdrantCorpusIndex(
        client=built_corpus._client,
        collection=built_corpus.collection,
        snapshot=built_corpus.snapshot,
    )
    calls = 0
    both_arrived = threading.Barrier(2, timeout=5.0)

    def spy(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        return corpus_embeddings.embedding_client(*args, **kwargs)

    def first_use(query: str) -> None:
        # The barrier has to sit outside the guarded section. Putting it inside
        # deadlocks against the very lock under test: when the lock works only
        # one thread ever enters, so the second could never arrive.
        both_arrived.wait()
        index.search(query, limit=2)

    with patch(
        "contract_analyzer.corpus.qdrant_index.embedding_client",
        side_effect=spy,
    ):
        threads = [
            threading.Thread(target=first_use, args=(query,))
            for query in ("pierwsze zapytanie", "drugie zapytanie")
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10.0)
            assert not thread.is_alive(), "a search never finished"

        assert calls == 1, f"{calls} clients were built for one index"
        index.close()
        index.search("trzecie zapytanie", limit=2)
        assert calls == 2
        index.close()


@pytest.mark.anyio
async def test_search_followed_by_snippet_reads_issues_zero_scroll_calls(
    built_corpus: QdrantCorpusIndex,
) -> None:
    scroll_calls: list[Any] = []
    original_scroll = built_corpus._client.scroll

    def spy_scroll(*args: Any, **kwargs: Any) -> Any:
        scroll_calls.append((args, kwargs))
        return original_scroll(*args, **kwargs)

    built_corpus._client.scroll = spy_scroll

    active = ActiveRun(
        services=SimpleNamespace(corpus=built_corpus),  # type: ignore[arg-type]
        run_id=uuid4(),
        request=RunRequest(document_id=uuid4(), arm=ArmCode.MID),
        session=SimpleNamespace(input_hash="test-input-hash"),  # type: ignore[arg-type]
        call_units=[],
        call_units_by_id={},
        budget=BudgetTracker(wall_budget_seconds=300.0),
        semaphore=asyncio.Semaphore(8),
        retrieval_cache={},
        provisions={},
    )

    data, candidates = await run_search(active, "wypowiedzenie najmu")
    assert len(candidates) == 5
    assert len(scroll_calls) == 0

    results = data["results"]
    assert len(results) == 5
    for hit in results:
        cached_unit = active.provisions[hit["locator"]]
        assert cached_unit is not None
        assert hit["snippet"] == cached_unit.text[:FINDER_SNIPPET_CHARS]


def test_the_application_closes_the_embedding_client_it_opened() -> None:
    """The index holds one HTTP client for its life, so shutdown must close it."""
    import asyncio
    from unittest.mock import MagicMock

    from contract_analyzer.api.app import _shutdown

    corpus = MagicMock()
    state = MagicMock()
    state.services.corpus = corpus
    state.text_keys = {}
    state.services.text_store = None
    state.tasks.cancel_all = MagicMock(return_value=asyncio.sleep(0))

    asyncio.run(_shutdown(state, None))

    corpus.close.assert_called_once_with()
