"""Plain-text ingestion."""

import logging

from contract_analyzer.domain import DocumentPayload, SourceAnchor

from .errors import IngestError
from .payload import _payload

logger = logging.getLogger(__name__)


def _decode_txt_bytes(data: bytes) -> str:
    if not data:
        logger.warning("TXT decode failed: empty input")
        raise IngestError("empty_input", "empty_input")
    last_error: UnicodeDecodeError | None = None
    # utf-8 omitted: every utf-8-decodable sequence is also decoded by utf-8-sig.
    for encoding in ("utf-8-sig", "cp1250"):
        try:
            text = data.decode(encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
        if text.strip():
            return text
    if last_error is not None:
        logger.warning("TXT decode failed: corrupt input")
        raise IngestError("corrupt_input", "corrupt_input") from last_error
    logger.warning("TXT decode failed: whitespace-only text")
    raise IngestError("empty_input", "empty_input")


def _read_txt(data: bytes) -> DocumentPayload:
    return _read_txt_from_text(_decode_txt_bytes(data))


def _read_txt_from_text(text: str) -> DocumentPayload:
    anchors: list[SourceAnchor] = []
    offset = 0
    for line_number, line in enumerate(text.splitlines(keepends=True), start=1):
        end = offset + len(line)
        anchors.append(
            SourceAnchor(
                start_offset=offset,
                end_offset=end,
                line_start=line_number,
                line_end=line_number,
            )
        )
        offset = end
    if not anchors or not text.strip():
        logger.warning("TXT read failed: empty anchors or text")
        raise IngestError("empty_input", "empty_input")
    return _payload(text, tuple(anchors), None)
