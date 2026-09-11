"""Ingestion errors and limit failures."""

import logging

logger = logging.getLogger(__name__)


class IngestError(Exception):
    """Raised when raw contract bytes fail validation or conversion into payload.

    Covers empty/corrupt input bytes, unsupported file extensions, size limits,
    and subprocess boundary failures (LibreOffice or Tesseract). Handled by API
    upload endpoints to map directly to client HTTP error responses with the
    stable machine code.
    """

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


def _raise_limit_breach(limit_name: str, limit: int, observed: int) -> None:
    logger.warning(
        "ingest refused: %s %d exceeds limit %d", limit_name, observed, limit
    )
    raise IngestError(
        "limit_breach",
        f"limit_breach: {limit_name} exceeded (limit {limit})",
    )
