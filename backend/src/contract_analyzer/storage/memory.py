"""In-memory metadata storage."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import fields, replace
from pathlib import Path
from typing import Any, Literal
from uuid import UUID

from contract_analyzer.storage.base import MetadataStore
from contract_analyzer.storage.records import (
    AttemptRecord,
    FindingRecord,
    RunRecord,
    RunTotals,
    _now,
    _require_code,
    _validate_attempt,
    _validate_run,
)


class InMemoryMetadataStore(MetadataStore):
    """In-memory thread-safe metadata store for unit tests and local development."""

    def __init__(
        self,
        path: Path | str = ":memory:",
        *,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.path = Path(path)
        if str(path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.touch(exist_ok=True)
        self._clock = clock or time.monotonic
        self._lock = threading.Lock()
        self._runs: dict[UUID, RunRecord] = {}
        self._attempts: dict[UUID, list[AttemptRecord]] = {}
        self._findings: dict[UUID, list[FindingRecord]] = {}
        self._monotonic_starts: dict[UUID, float] = {}
        self._elapsed_ms: dict[UUID, float] = {}

    def create_run(self, run: RunRecord) -> None:
        _validate_run(run)
        with self._lock:
            if run.id in self._runs:
                raise ValueError(f"Run {run.id} already exists")
            self._runs[run.id] = replace(run, created_at=run.created_at or _now())

    def record_attempt(self, attempt: AttemptRecord) -> None:
        _validate_attempt(attempt)
        with self._lock:
            if attempt.run_id not in self._runs:
                raise KeyError(f"Run {attempt.run_id} not found")
            self._attempts.setdefault(attempt.run_id, []).append(attempt)
            if attempt.status == "success" and attempt.returned_model is not None:
                run = self._runs.get(attempt.run_id)
                if run is not None and run.returned_model is None:
                    self._runs[attempt.run_id] = replace(
                        run, returned_model=attempt.returned_model
                    )

    def store_finding(self, finding: FindingRecord) -> None:
        with self._lock:
            if finding.run_id not in self._runs:
                raise KeyError(f"Run {finding.run_id} not found")
            self._findings.setdefault(finding.run_id, []).append(finding)

    def finish_run(
        self,
        run_id: UUID,
        status: Literal["completed", "failed", "cancelled"],
        *,
        totals: RunTotals | None = None,
    ) -> None:
        if totals is None:
            totals = RunTotals()
        if totals.error_code is not None:
            _require_code(totals.error_code, "error_code")
        if totals.elapsed_ms is not None and totals.elapsed_ms < 0:
            raise ValueError("elapsed_ms cannot be negative")
        if totals.interruption_reason not in (None, "wall_time"):
            raise ValueError("invalid interruption reason")

        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                raise KeyError(run_id)
            computed_elapsed = totals.elapsed_ms
            if computed_elapsed is None and run_id in self._monotonic_starts:
                computed_elapsed = (
                    self._clock() - self._monotonic_starts[run_id]
                ) * 1000.0
            if computed_elapsed is not None:
                self._elapsed_ms[run_id] = computed_elapsed

            updates: dict[str, Any] = {
                "status": status,
                "finished_at": _now(),
                "elapsed_ms": computed_elapsed,
                "error_code": totals.error_code,
                "error_detail": totals.error_detail,
                "source_text": totals.source_text,
            }
            for f in fields(totals):
                if f.name in (
                    "elapsed_ms",
                    "error_code",
                    "error_detail",
                    "source_text",
                ):
                    continue
                val = getattr(totals, f.name)
                if val is not None:
                    updates[f.name] = val
            self._runs[run_id] = replace(run, **updates)

    def set_run_monotonic_start(self, run_id: UUID, monotonic_start: float) -> None:
        with self._lock:
            if run_id not in self._runs:
                raise KeyError(run_id)
            if self._runs[run_id].status != "running":
                raise KeyError(run_id)
            self._monotonic_starts[run_id] = monotonic_start

    def elapsed_ms_for_run(self, run_id: UUID) -> float | None:
        with self._lock:
            if run_id not in self._runs:
                return None
            if run_id in self._elapsed_ms:
                return self._elapsed_ms[run_id]
            if run_id in self._monotonic_starts:
                return (self._clock() - self._monotonic_starts[run_id]) * 1000.0
            return None

    def mark_interrupted_runs_failed(self) -> int:
        with self._lock:
            count = 0
            for r_id, run in list(self._runs.items()):
                if run.status == "running":
                    self._runs[r_id] = replace(
                        run,
                        status="failed",
                        error_code="process_restarted",
                        error_detail="Process interrupted or crashed",
                        finished_at=_now(),
                    )
                    count += 1
            return count

    def get_run(self, run_id: UUID) -> RunRecord | None:
        with self._lock:
            return self._runs.get(run_id)

    def list_runs(self) -> list[RunRecord]:
        with self._lock:
            return list(self._runs.values())

    def list_attempts(self, run_id: UUID) -> list[AttemptRecord]:
        with self._lock:
            return list(self._attempts.get(run_id, []))

    def delete_document_runs(self, document_id: UUID) -> int:
        """Drop this document's runs and everything filed under them."""
        with self._lock:
            doomed = [
                run_id
                for run_id, run in self._runs.items()
                if run.document_id == document_id
            ]
            for run_id in doomed:
                self._runs.pop(run_id, None)
                self._attempts.pop(run_id, None)
                self._findings.pop(run_id, None)
            return len(doomed)

    def list_findings(self, run_id: UUID) -> list[FindingRecord]:
        with self._lock:
            return list(self._findings.get(run_id, []))
