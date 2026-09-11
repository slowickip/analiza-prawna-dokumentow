"""Finding placement within rendered document blocks.

A finding belongs to the block containing the start of its resolved anchor,
and a finding without a resolved anchor belongs to no block at all.
"""

from __future__ import annotations

from collections.abc import Sequence

from contract_analyzer.domain import SourceAnchor


def block_id_at(index: int, anchor: SourceAnchor) -> str:
    """The identifier a block is addressed by.

    Readers that name their blocks keep the name; the rest are addressed by
    position, which is stable for one document because the anchors are.
    """
    return anchor.block_id or f"block-{index}"


def block_for_offset(
    anchors: Sequence[SourceAnchor], start_offset: int | None
) -> str | None:
    """The block holding this offset, or None when nothing holds it.

    Containment of the start, not overlap: a finding that runs past the end of its
    block still belongs to the block it starts in, and belongs to that one only.
    """
    if start_offset is None:
        return None
    for index, anchor in enumerate(anchors):
        if anchor.start_offset <= start_offset < anchor.end_offset:
            return block_id_at(index, anchor)
    return None
