"""Shared corpus retrieval ranking and record decoding."""

from __future__ import annotations

import hashlib
from datetime import date
from typing import Any

from contract_analyzer.corpus.models import LegalUnit
from contract_analyzer.domain import ForceScope, ForceState, ForceValue

RRF_K = 60


def _fusion_order(
    sparse_ranks: dict[str, int],
    dense_ranks: dict[str, int],
) -> list[str]:
    """Order both limbs' candidates by reciprocal-rank fusion, best first.

    Equal fused scores are separated by a digest of the unit id. Chosen by argument,
    not measurement. Breaking ties on the unit id itself, as this did until now, broke
    them on act identity: ids read ``ACT:article``, so every tie went to the act with
    the lower publisher, year and position -- ``DU/2023/725`` ahead of ``DU/2026/795``
    whatever was asked. The digest keeps the order reproducible, which a fused list has
    to be, while saying nothing about the act, its year or its article number.

    Preferring the better of the two limb ranks was the alternative considered, and it
    cannot separate anything at this scale: with one ``RRF_K`` shared by both limbs and
    both limbs cut at the same limit, equal fused scores imply an identical multiset of
    limb ranks for every limit below 24, and ``graph.RETRIEVAL_TOP_K`` is far below it.
    """
    scores = {
        unit_id: sum(
            1.0 / (RRF_K + ranks[unit_id])
            for ranks in (sparse_ranks, dense_ranks)
            if unit_id in ranks
        )
        for unit_id in set(sparse_ranks) | set(dense_ranks)
    }
    return sorted(
        scores,
        key=lambda unit_id: (
            -scores[unit_id],
            hashlib.sha256(unit_id.encode("utf-8")).hexdigest(),
        ),
    )


def _force_state_from_row(
    value: str,
    scope: str,
    source_locator: str,
    snapshot_date: str,
) -> ForceState:
    return ForceState(
        value=ForceValue(value),
        scope=ForceScope(scope),
        snapshot_date=date.fromisoformat(snapshot_date),
        source_locator=source_locator,
    )


def _legal_unit_from_payload(payload: dict[str, Any] | None) -> LegalUnit:
    if payload is None:
        raise ValueError("point has no payload")
    return LegalUnit(
        id=payload["id"],
        locator=payload["locator"],
        act_identifier=payload["act_identifier"],
        article_identifier=payload["article_identifier"],
        text=payload["text"],
        content_hash=payload["content_hash"],
        act_force=_force_state_from_row(
            payload["act_force_value"],
            payload["act_force_scope"],
            payload["act_force_source_locator"],
            payload["act_force_snapshot_date"],
        ),
        provision_force=_force_state_from_row(
            payload["provision_force_value"],
            payload["provision_force_scope"],
            payload["provision_force_source_locator"],
            payload["provision_force_snapshot_date"],
        ),
        legal_status_date=date.fromisoformat(payload["legal_status_date"]),
    )
