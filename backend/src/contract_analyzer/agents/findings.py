"""Turning a decided finding into a stored record with its grounding.

A finding is only useful if the reader can open the place in the contract it is
about, so this module owns the anchoring: resolve the quoted fragment to a span,
fall back to the unit's own anchor, and map the offset to a page.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from uuid import uuid4

from contract_analyzer.agents.session import (
    CallUnit,
    DocumentSession,
    unit_by_id,
)
from contract_analyzer.agents.state import ActiveRun
from contract_analyzer.domain import (
    DecisionFacts,
    EmittedBasis,
    Finding,
    QuoteResolution,
    resolve_finding,
)
from contract_analyzer.storage import FindingRecord, RunEvent

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QuoteSpanResult:
    resolution: QuoteResolution
    span: tuple[int, int] | None = None


def _quote_pattern(quote: str) -> str:
    return r"\s+".join(re.escape(word) for word in quote.split())


def quote_occurs(text: str, quote: str) -> bool:
    """Whether an exact quote occurs when whitespace runs are treated alike."""
    return (
        bool(quote.strip())
        and re.search(_quote_pattern(quote.strip()), text) is not None
    )


def resolve_quote_span(
    session: DocumentSession, call_unit: CallUnit, quote: str
) -> QuoteSpanResult:
    """Locate the quoted fragment inside the call unit's own window.

    Whitespace is matched loosely because a model reproduces line breaks freely,
    but the words are matched exactly: a quote that is not there is recorded as
    fabricated rather than anchored to the nearest thing that looks like it.
    """
    if not quote.strip():
        return QuoteSpanResult(QuoteResolution.QUOTE_EMPTY)
    if call_unit.is_whole_document:
        window_start = 0
        window_text = session.payload.text
    else:
        unit = unit_by_id(session, call_unit.unit_id)
        window_start = unit.anchor.start_offset
        window_text = session.payload.text[window_start : unit.anchor.end_offset]
    matches = list(re.finditer(_quote_pattern(quote.strip()), window_text))
    if len(matches) == 0:
        return QuoteSpanResult(QuoteResolution.QUOTE_FABRICATED)
    if len(matches) > 1:
        return QuoteSpanResult(QuoteResolution.QUOTE_AMBIGUOUS)
    match = matches[0]
    return QuoteSpanResult(
        QuoteResolution.RESOLVED,
        (window_start + match.start(), window_start + match.end()),
    )


def page_for_offset(session: DocumentSession, offset: int) -> int | None:
    for anchor in session.payload.anchors:
        if anchor.start_offset <= offset < anchor.end_offset:
            return anchor.page
    return None


def store_unit_finding(
    active: ActiveRun,
    unit_id: str,
    finding: Finding,
    *,
    legal_locators: tuple[str, ...] = (),
    basis: EmittedBasis | None = None,
    quote_span: tuple[int, int] | None = None,
    quote_resolution: QuoteResolution | None = None,
) -> FindingRecord:
    """Persist one finding with its grounding anchors and announce it.

    Grounding precedence: a resolved quote span pinpoints the fragment; otherwise a
    segmented unit falls back to its own anchor and bounding box; a whole-document
    call unit without a quote stays unpinned, because the document as a whole is
    not a place a reader can be sent to.
    """
    call_unit = active.call_units_by_id.get(unit_id)
    start_offset: int | None = None
    end_offset: int | None = None
    page: int | None = None
    bbox: tuple[float, float, float, float] | None = None
    if quote_span is not None:
        start_offset, end_offset = quote_span
    elif call_unit is None or not call_unit.is_whole_document:
        unit = unit_by_id(active.session, unit_id)
        start_offset = unit.anchor.start_offset
        end_offset = unit.anchor.end_offset
        bbox = unit.anchor.bbox
    if start_offset is not None:
        page = page_for_offset(active.session, start_offset)
    record = FindingRecord(
        id=uuid4(),
        run_id=active.run_id,
        unit_id=unit_id,
        code=finding.code,
        uncertain_cause=finding.uncertain_cause,
        raw_confidence=finding.raw_confidence,
        start_offset=start_offset,
        end_offset=end_offset,
        page=page,
        bbox=bbox,
        legal_locators=legal_locators,
        basis=basis,
        quote_resolution=quote_resolution,
    )
    active.services.metadata.store_finding(record)
    if active.services.events is not None:
        active.services.events.publish(active.run_id, RunEvent.finding(record.id))
    return record


def store_not_processed(active: ActiveRun, unit_id: str) -> FindingRecord:
    """Record that this unit was never analysed, which is a result of its own."""
    return store_unit_finding(
        active, unit_id, resolve_finding(DecisionFacts(processed=False))
    )
