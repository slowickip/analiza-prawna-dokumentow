"""Domain errors."""


class CorpusBuildError(Exception):
    """Raised when offline corpus ingestion, ELI metadata retrieval, or text fails."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)
