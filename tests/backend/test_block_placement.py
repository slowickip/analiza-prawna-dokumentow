"""Where a finding sits in the reader's view, decided on the server.

The browser used to answer this from raw offsets with a three-tier fallback, and
two of the tiers were wrong: a resolved finding matched both at its own anchor and
under every block its call unit spanned, and a finding whose quote never resolved
was placed under its unit anyway. These tests pin the single rule that replaced it.
"""

from __future__ import annotations

from contract_analyzer.domain import SourceAnchor
from contract_analyzer.structure.blocks import (
    block_for_offset,
    block_id_at,
)

ANCHORS = (
    SourceAnchor(start_offset=0, end_offset=10),
    SourceAnchor(start_offset=10, end_offset=25),
    SourceAnchor(start_offset=25, end_offset=40),
)


def test_a_finding_belongs_to_the_block_its_anchor_starts_in() -> None:
    assert block_for_offset(ANCHORS, 0) == "block-0"
    assert block_for_offset(ANCHORS, 9) == "block-0"
    assert block_for_offset(ANCHORS, 10) == "block-1"
    assert block_for_offset(ANCHORS, 39) == "block-2"


def test_a_finding_spanning_blocks_still_belongs_to_exactly_one() -> None:
    """Containment of the start, not overlap.

    Overlap is what produced duplicates: a finding running from block 0 into block
    2 overlapped all three and was rendered under each.
    """
    placements = [b for b in (block_for_offset(ANCHORS, 5),) if b]
    assert placements == ["block-0"]
    assert len({block_for_offset(ANCHORS, 5)}) == 1


def test_an_unresolved_anchor_belongs_to_no_block() -> None:
    """The contract forbids approximating the position of an unresolved finding."""
    assert block_for_offset(ANCHORS, None) is None


def test_an_offset_outside_every_block_places_nowhere() -> None:
    assert block_for_offset(ANCHORS, 40) is None
    assert block_for_offset((), 0) is None


def test_a_reader_that_names_its_blocks_keeps_the_name() -> None:
    """Only the docx reader emits block identifiers; the rest are positional."""
    named = SourceAnchor(start_offset=0, end_offset=5, block_id="para-7")
    assert block_id_at(3, named) == "para-7"
    assert block_id_at(3, SourceAnchor(start_offset=0, end_offset=5)) == "block-3"
    assert [block_id_at(index, anchor) for index, anchor in enumerate(ANCHORS)] == [
        "block-0",
        "block-1",
        "block-2",
    ]
