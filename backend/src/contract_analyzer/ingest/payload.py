"""Document payload and source-anchor assembly."""

import hashlib
from uuid import uuid4

from contract_analyzer.domain import (
    ConversionProvenance,
    DocumentPayload,
    ReadMode,
    SourceAnchor,
)


def _payload(
    text: str,
    anchors: tuple[SourceAnchor, ...],
    read_mode: ReadMode | None,
    *,
    conversion: ConversionProvenance | None = None,
) -> DocumentPayload:
    return DocumentPayload(
        document_id=uuid4(),
        text=text,
        anchors=anchors,
        read_mode=read_mode,
        content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        conversion=conversion,
    )


def _record_blank_page(
    *,
    anchors: list[SourceAnchor],
    page_number: int,
    read_mode: ReadMode,
    offset: int,
) -> None:
    anchors.append(
        SourceAnchor(
            start_offset=offset,
            end_offset=offset,
            read_mode=read_mode,
            block_id=f"blank_page_{page_number}",
            page=page_number,
        )
    )


def _append_word(
    *,
    parts: list[str],
    anchors: list[SourceAnchor],
    offset: int,
    word: str,
    page_number: int,
    bbox: tuple[float, float, float, float],
    read_mode: ReadMode,
    separator: str = " ",
) -> int:
    if parts:
        parts.append(separator)
        offset += len(separator)
    start = offset
    parts.append(word)
    offset += len(word)
    anchors.append(
        SourceAnchor(
            start_offset=start,
            end_offset=offset,
            read_mode=read_mode,
            page=page_number,
            bbox=bbox,
        )
    )
    return offset
