"""Contract document ingestion."""

from .errors import IngestError as IngestError
from .service import IngestService as IngestService

__all__ = ["IngestError", "IngestService"]
