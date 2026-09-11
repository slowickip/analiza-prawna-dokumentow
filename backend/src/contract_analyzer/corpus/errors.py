"""Corpus-specific exceptions."""

from contract_analyzer.domain import CorpusBuildError as CorpusBuildError


class CorpusIntegrityError(Exception):
    """Raised when a corpus snapshot violates cryptographic or structural invariants.

    Signals corrupted manifest hashes, missing SQLite indices, or unreadable
    provision stores. Unlike transient build errors, integrity failures are
    unrecoverable at runtime and require re-verifying or rebuilding the frozen
    corpus artifact.
    """

    code: str

    def __init__(self, code: str, message: str) -> None:
        object.__setattr__(self, "code", code)
        super().__init__(message)


class EmbeddingError(Exception):
    """Raised when the embedding client fails to vectorize provisions or queries."""

    code: str

    def __init__(self, code: str, message: str) -> None:
        object.__setattr__(self, "code", code)
        super().__init__(message)
