"""Corpus access as the roles are given it: bounded search and provision reads.

The refusals live here rather than at the call site because they are the input
contract the embedding service does not provide: an empty or over-long phrase is
the model's mistake to correct on its next turn, and ending a run over it would
lose every unit queued behind it.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Callable
from typing import Any

from contract_analyzer.agents.entries import Author
from contract_analyzer.agents.errors import (
    BudgetExhausted,
    RetrievalDependencyUnavailable,
    RunPipelineError,
    first_leaf,
)
from contract_analyzer.agents.scheduler import (
    FINDER_MAX_PHRASE_CHARS,
    FINDER_SNIPPET_CHARS,
    RETRIEVAL_TOP_K,
)
from contract_analyzer.agents.state import ActiveRun, UnitContext
from contract_analyzer.corpus import EmbeddingError, LegalUnit
from contract_analyzer.domain import RetrievalCandidate

logger = logging.getLogger(__name__)


def sanitize_retrieval_query(text: str) -> str:
    """Reduce a phrase to the words an encoder can be handed."""
    tokens = re.findall(r"[\wąćęłńóśźżĄĆĘŁŃÓŚŹŻ]+", text, flags=re.UNICODE)
    if not tokens:
        raise RunPipelineError("retrieval_query_empty")
    return " ".join(tokens)


async def search_corpus(active: ActiveRun, phrase: str) -> list[RetrievalCandidate]:
    """Run one model-chosen phrase against the frozen corpus.

    The cache is keyed on the phrase, the document and the corpus snapshot, so the
    near-duplicate phrases a model produces on later turns cost nothing. The
    protocol permits a retrieval cache inside one run and forbids one across runs.
    """
    query = sanitize_retrieval_query(phrase)
    cache_key = (
        f"{query}|{active.session.input_hash}|{active.services.corpus.snapshot.id}"
    )
    if cache_key in active.retrieval_cache:
        # Counted because the cache is the reason a repeated phrase is free, and a
        # saving nobody records is one nobody can check against the search counts.
        active.retrieval_cache_hits += 1
        return active.retrieval_cache[cache_key]
    # search() encodes the query over HTTP against the embedding service. Off the
    # loop, or a slow encoder stalls event streaming and the wall-budget timer
    # along with it.
    try:
        results = await _bounded_corpus_call(
            active,
            active.services.corpus.search,
            query,
            RETRIEVAL_TOP_K,
        )
    except EmbeddingError as error:
        raise RetrievalDependencyUnavailable("dependency_unavailable") from error

    # Search decoded these units already; keeping them spares the snippet reads
    # a scroll each. The cache sits below the layer the protocol counts.
    candidates: list[RetrievalCandidate] = []
    for candidate, unit in results:
        active.provisions[candidate.locator] = unit
        candidates.append(candidate)

    active.retrieval_cache[cache_key] = candidates
    return candidates


async def read_provision(active: ActiveRun, locator: str) -> LegalUnit | None:
    """The provision at this locator, or None when the corpus does not hold it."""
    _require_budget(active)
    if locator in active.provisions:
        return active.provisions[locator]
    try:
        provision = await _bounded_corpus_call(
            active, active.services.corpus.read, locator
        )
    except KeyError:
        provision = None
    active.provisions[locator] = provision
    return provision


async def provision_snippet(active: ActiveRun, locator: str) -> str:
    unit = await read_provision(active, locator)
    return "" if unit is None else unit.text[:FINDER_SNIPPET_CHARS]


async def run_read(
    context: UnitContext, author: Author, locator: str
) -> dict[str, Any]:
    """Read one provision for a role, recording the read on the worksheet.

    The read is on the record because it is part of how the role reached its
    conclusion: a verdict supported by a provision nobody opened is a different
    claim from one supported by a provision that was read.
    """
    unit = await read_provision(context.active, locator)
    if unit is None:
        logger.warning(
            "tool refusal run=%s unit=%s tool=read_provision code=unknown_locator",
            context.active.run_id,
            context.call_unit.unit_id,
        )
        return {"error": "unknown_locator", "locator": locator}
    context.active.provision_reads += 1
    context.worksheet.add_read(author=author, locator=locator)
    # Only the researcher widens the ledger. Letting the analyst add to it made
    # "the analyst may cite only what the researcher retrieved" false: a read of
    # any valid locator would have entered its own evidence into the record.
    if author == "researcher":
        context.locator_ledger.add(locator)
    logger.debug("read locator_length=%d text=%d", len(locator), len(unit.text))
    return {
        "locator": unit.locator,
        "act_identifier": unit.act_identifier,
        "text": unit.text,
    }


async def run_search(
    active: ActiveRun, phrase: object, *, unit_id: str | None = None
) -> tuple[dict[str, Any], list[RetrievalCandidate]]:
    """Execute one search phrase and return what the model is told about it."""
    if not isinstance(phrase, str) or not phrase.strip():
        logger.warning(
            "tool refusal run=%s unit=%s tool=search_corpus code=phrase_empty",
            active.run_id,
            unit_id,
        )
        return {"error": "phrase_empty"}, []
    if len(phrase) > FINDER_MAX_PHRASE_CHARS:
        logger.warning(
            "tool refusal run=%s unit=%s tool=search_corpus code=phrase_too_long",
            active.run_id,
            unit_id,
        )
        return (
            {
                "error": "phrase_too_long",
                "limit_characters": FINDER_MAX_PHRASE_CHARS,
                "received_characters": len(phrase),
            },
            [],
        )
    try:
        candidates = await search_corpus(active, phrase)
    except BudgetExhausted:
        raise
    except RetrievalDependencyUnavailable:
        raise
    except RunPipelineError as error:
        logger.warning(
            "tool refusal run=%s unit=%s tool=search_corpus code=%s",
            active.run_id,
            unit_id,
            error.code,
        )
        return {"error": error.code, "results": []}, []
    logger.debug("search phrase_length=%d results=%d", len(phrase), len(candidates))
    tasks: list[asyncio.Task[str]] = []
    try:
        async with asyncio.TaskGroup() as group:
            tasks.extend(
                group.create_task(provision_snippet(active, candidate.locator))
                for candidate in candidates
            )
    except BaseExceptionGroup as error:
        raise first_leaf(error) from None
    ordered = [task.result() for task in tasks]
    results = [
        {
            "locator": candidate.locator,
            "act_identifier": candidate.act_identifier,
            "snippet": snippet,
        }
        for candidate, snippet in zip(candidates, ordered, strict=True)
    ]
    return {"results": results}, candidates


def _require_budget(active: ActiveRun) -> float:
    """The seconds left on the wall clock, or raise as an expired run does."""
    remaining = active.budget.remaining_seconds()
    if remaining <= 0:
        raise BudgetExhausted("budget_exhausted")
    return remaining


async def _bounded_corpus_call[ResultT](
    active: ActiveRun, function: Callable[..., ResultT], *args: object
) -> ResultT:
    remaining = _require_budget(active)
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(function, *args), timeout=remaining
        )
    except TimeoutError:
        active.budget.expire()
        raise BudgetExhausted("budget_exhausted") from None
