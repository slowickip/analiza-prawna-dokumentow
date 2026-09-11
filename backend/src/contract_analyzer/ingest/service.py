"""Filename-based ingestion service."""

import logging
from pathlib import Path

from contract_analyzer.config import RunConfig
from contract_analyzer.domain import DocumentPayload

from .doc_conversion import _read_doc
from .docx import _read_docx
from .errors import IngestError, _raise_limit_breach
from .pdf import _read_pdf
from .txt import _read_txt

logger = logging.getLogger(__name__)

_ZIP_SIGNATURE = b"PK\x03\x04"


class IngestService:
    def __init__(self, config: RunConfig | None = None) -> None:
        self._config = config or RunConfig()

    def ingest(self, filename: str, data: bytes) -> DocumentPayload:
        if len(data) > self._config.max_input_bytes:
            _raise_limit_breach(
                "max_input_bytes", self._config.max_input_bytes, len(data)
            )
        extension = Path(filename).suffix.lower()
        try:
            if extension == ".txt":
                return _read_txt(data)
            if extension == ".doc":
                # OOXML under .doc only: filename stays authoritative per openapi.yaml.
                if data.startswith(_ZIP_SIGNATURE):
                    return _read_docx(data)
                return _read_doc(data, self._config)
            if extension == ".docx":
                return _read_docx(data)
            if extension == ".pdf":
                return _read_pdf(data, self._config)
            logger.warning("ingest refused: unsupported extension")
            raise IngestError("unsupported_input", "unsupported_input")
        except IngestError:
            raise
        except Exception:
            logger.warning("ingest failed: input unreadable")
            raise IngestError("corrupt_input", "corrupt_input") from None
