"""DOCX block reading and payload assembly."""

import io
import logging
from zipfile import BadZipFile

from docx import Document as open_docx_document
from docx.document import Document as DocxDocument
from docx.opc.exceptions import PackageNotFoundError
from docx.table import Table, _Cell
from docx.text.paragraph import Paragraph
from lxml.etree import XMLSyntaxError  # type: ignore[import-untyped]

from contract_analyzer.domain import (
    ConversionProvenance,
    DocumentPayload,
    SourceAnchor,
)

from .errors import IngestError
from .payload import _payload

logger = logging.getLogger(__name__)


def _iter_docx_blocks(document: DocxDocument) -> list[tuple[str, str]]:
    blocks: list[tuple[str, str]] = []
    block_index = 0

    def append_paragraph(paragraph: Paragraph, block_id: str) -> None:
        nonlocal block_index
        if paragraph.text.strip():
            blocks.append((block_id, paragraph.text))
        block_index += 1

    def walk_cell(cell: _Cell, cell_id: str) -> None:
        nested_index = 0
        for child in cell._tc:
            tag = child.tag.rsplit("}", 1)[-1]
            if tag == "p":
                append_paragraph(Paragraph(child, document), cell_id)
            elif tag == "tbl":
                walk_table(Table(child, document), f"{cell_id}_t{nested_index}")
                nested_index += 1

    def walk_table(table: Table, table_id: str) -> None:
        seen_cells: set[object] = set()
        for row_index, row in enumerate(table.rows):
            for col_index, cell in enumerate(row.cells):
                if cell._tc in seen_cells:
                    continue
                seen_cells.add(cell._tc)
                cell_id = f"{table_id}_r{row_index}_c{col_index}"
                walk_cell(cell, cell_id)

    for child in document.element.body:
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            append_paragraph(Paragraph(child, document), f"p{block_index}")
        elif tag == "tbl":
            walk_table(Table(child, document), f"t{block_index}")
            block_index += 1

    return blocks


def _read_docx(
    data: bytes, *, conversion: ConversionProvenance | None = None
) -> DocumentPayload:
    try:
        document = open_docx_document(io.BytesIO(data))
    except (BadZipFile, PackageNotFoundError, ValueError, XMLSyntaxError):
        logger.warning("DOCX read failed: corrupt input")
        raise IngestError("corrupt_input", "corrupt_input") from None
    blocks = _iter_docx_blocks(document)
    if not blocks:
        logger.warning("DOCX read failed: no blocks extracted")
        raise IngestError("empty_input", "empty_input")
    parts: list[str] = []
    anchors: list[SourceAnchor] = []
    offset = 0
    for block_id, block_text in blocks:
        if parts:
            parts.append("\n")
            offset += 1
        start = offset
        parts.append(block_text)
        offset += len(block_text)
        anchors.append(
            SourceAnchor(
                start_offset=start,
                end_offset=offset,
                block_id=block_id,
            )
        )
    text = "".join(parts)
    if not text.strip():
        logger.warning("DOCX read failed: empty text")
        raise IngestError("empty_input", "empty_input")
    return _payload(text, tuple(anchors), None, conversion=conversion)
