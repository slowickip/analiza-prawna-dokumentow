"""Metadata store interface and factory."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal
from uuid import UUID

from contract_analyzer.storage.records import (
    AttemptRecord,
    FindingRecord,
    RunRecord,
    RunTotals,
)


class MetadataStore:
    """Base class for metadata storage engines."""

    SCHEMA_VERSION = 11

    def __new__(
        cls,
        target: Path | str = ":memory:",
        *,
        clock: Callable[[], float] | None = None,
        **kwargs: Any,
    ) -> MetadataStore:
        # A subclass may take arguments of its own; only the base dispatches.
        if cls is MetadataStore:
            return open_metadata_store(target, clock=clock)
        return super().__new__(cls)

    def create_run(self, run: RunRecord) -> None:
        raise NotImplementedError

    def record_attempt(self, attempt: AttemptRecord) -> None:
        raise NotImplementedError

    def store_finding(self, finding: FindingRecord) -> None:
        raise NotImplementedError

    def finish_run(
        self,
        run_id: UUID,
        status: Literal["completed", "failed", "cancelled"],
        *,
        totals: RunTotals | None = None,
    ) -> None:
        raise NotImplementedError

    def set_run_monotonic_start(self, run_id: UUID, monotonic_start: float) -> None:
        raise NotImplementedError

    def elapsed_ms_for_run(self, run_id: UUID) -> float | None:
        raise NotImplementedError

    def mark_interrupted_runs_failed(self) -> int:
        raise NotImplementedError

    def get_run(self, run_id: UUID) -> RunRecord | None:
        raise NotImplementedError

    def list_runs(self) -> list[RunRecord]:
        raise NotImplementedError

    def list_attempts(self, run_id: UUID) -> list[AttemptRecord]:
        raise NotImplementedError

    def delete_document_runs(self, document_id: UUID) -> int:
        """Erase every run of one document, and everything hanging off it."""
        raise NotImplementedError

    def list_findings(self, run_id: UUID) -> list[FindingRecord]:
        raise NotImplementedError


def open_metadata_store(
    path_or_url: Path | str = ":memory:",
    *,
    clock: Callable[[], float] | None = None,
    read_only: bool = False,
) -> MetadataStore:
    """Open the target store.

    For PostgreSQL, read_only skips schema creation and interrupted-run
    reconciliation on open; it does not disable subsequent write methods.
    The in-memory store does not reconcile runs and ignores this flag. A target
    other than :memory: creates its parent directory and touches the target file.
    """
    target = str(path_or_url)
    if target.startswith(("postgres://", "postgresql://")):
        from contract_analyzer.storage.postgres import PostgresMetadataStore

        return PostgresMetadataStore(target, clock=clock, read_only=read_only)
    from contract_analyzer.storage.memory import InMemoryMetadataStore

    return InMemoryMetadataStore(target, clock=clock)
