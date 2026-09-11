"""PostgreSQL metadata store implementation."""

from __future__ import annotations

import dataclasses
import json
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, Literal
from uuid import UUID

from contract_analyzer.storage.base import MetadataStore
from contract_analyzer.storage.record_rows import (
    _attempt_from_row,
    _finding_from_row,
    _run_from_row,
)
from contract_analyzer.storage.records import (
    AttemptRecord,
    FindingRecord,
    RunRecord,
    RunTotals,
    _hash_optional,
    _now,
    _parameters_json,
    _require_code,
    _validate_attempt,
    _validate_run,
)

POOL_ACQUIRE_TIMEOUT_SECONDS = 5.0
POOL_RECONNECT_TIMEOUT_SECONDS = 5.0
POOL_OPEN_TIMEOUT_SECONDS = 10.0


class PostgresMetadataStore(MetadataStore):
    """PostgreSQL store for text-free run metadata."""

    def __init__(
        self,
        dsn: str,
        *,
        clock: Callable[[], float] | None = None,
        connection_factory: Callable[..., Any] | None = None,
        read_only: bool = False,
    ) -> None:
        """Open the store, reconciling interrupted runs unless read_only.

        Opening writes twice by default: it creates the schema if it is absent,
        and it settles every run left in the running state, which is what a
        server that died mid-run needs on its way back up. Neither is scoped to
        the process opening the store, so both reach the runs of whichever
        server is live. That is correct for the server, which owns those runs,
        and wrong for anything that only wants to read: opening a diagnostic
        connection must not mark an active run as failed.

        read_only opens the same store and does neither. It creates no schema
        and settles nothing, so a reader cannot end a measurement it is only
        looking at. Nothing about how a run is executed, recorded or scored
        changes with it; the server opens the store exactly as before.
        """
        self.dsn = dsn
        self._clock = clock or time.monotonic
        self._connection_factory = connection_factory
        self.read_only = read_only
        self._open_pool()
        if read_only:
            return
        self._initialize()
        self.mark_interrupted_runs_failed()

    def _open_pool(self) -> None:
        """One pool for the store's life, opened eagerly so a bad DSN is loud.

        A statement used to buy its own connection, which cost a handshake per
        attempt and per finding. The pool's own default is to retry an
        unreachable database in the background while callers block; opening it
        with a wait turns that into one failure here instead.
        """
        import psycopg
        from psycopg.rows import dict_row
        from psycopg_pool import ConnectionPool

        factory = self._connection_factory

        class _PooledConnection:
            @classmethod
            def connect(cls, conninfo: str = "", **kwargs: Any) -> Any:
                if factory is not None:
                    return factory(conninfo, **kwargs)
                return psycopg.connect(conninfo, **kwargs)

        self._pool: ConnectionPool[Any] = ConnectionPool(
            conninfo=self.dsn,
            connection_class=_PooledConnection,
            kwargs={"row_factory": dict_row, "autocommit": True},
            min_size=1,
            max_size=10,
            timeout=POOL_ACQUIRE_TIMEOUT_SECONDS,
            reconnect_timeout=POOL_RECONNECT_TIMEOUT_SECONDS,
            open=False,
        )
        self._pool.open(wait=True, timeout=POOL_OPEN_TIMEOUT_SECONDS)

    def close(self) -> None:
        """Close the pool. The application calls this on shutdown."""
        pool = getattr(self, "_pool", None)
        if pool is not None:
            pool.close()

    @contextmanager
    def _connection(self) -> Iterator[Any]:
        with self._pool.connection() as conn:
            yield conn

    def _initialize(self) -> None:
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS runs (
                        id TEXT PRIMARY KEY,
                        document_id TEXT NOT NULL,
                        arm TEXT NOT NULL,
                        input_hash TEXT NOT NULL,
                        config_version TEXT NOT NULL,
                        prompt_bundle_version TEXT NOT NULL,
                        corpus_snapshot_id TEXT NOT NULL,
                        tool_bundle_version TEXT NOT NULL,
                        requested_model TEXT NOT NULL,
                        returned_model TEXT,
                        parameters_json TEXT NOT NULL,
                        retry_policy TEXT NOT NULL,
                        concurrency INTEGER NOT NULL,
                        wall_budget_seconds DOUBLE PRECISION NOT NULL,
                        measurement_valid INTEGER NOT NULL,
                        parent_run_id TEXT,
                        interaction TEXT,
                        status TEXT NOT NULL,
                        error_code TEXT,
                        error_hash TEXT,
                        source_hash TEXT,
                        monetary_cost_microunits BIGINT,
                        price_table_date TEXT,
                        price_table_hash TEXT,
                        cost_unknown_reason TEXT,
                        created_at TEXT NOT NULL,
                        finished_at TEXT,
                        context_edge_count INTEGER,
                        call_unit_count INTEGER,
                        finder_tool_turns INTEGER,
                        finder_search_calls INTEGER,
                        retrieval_cache_hits INTEGER,
                        finder_budget_exhausted_units INTEGER,
                        verifier_tool_turns INTEGER,
                        provision_reads INTEGER,
                        defaulted_characterisations INTEGER,
                        elapsed_ms DOUBLE PRECISION,
                        started_at_monotonic DOUBLE PRECISION,
                        graph_topology_version TEXT,
                        interruption_reason TEXT
                    );
                    CREATE TABLE IF NOT EXISTS attempts (
                        rowid BIGSERIAL PRIMARY KEY,
                        id TEXT NOT NULL,
                        run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                        requested_model TEXT NOT NULL,
                        returned_model TEXT,
                        prompt_version TEXT NOT NULL,
                        temperature DOUBLE PRECISION NOT NULL,
                        parameters_json TEXT NOT NULL,
                        input_tokens INTEGER NOT NULL,
                        output_tokens INTEGER NOT NULL,
                        latency_ms DOUBLE PRECISION NOT NULL,
                        status TEXT NOT NULL,
                        retry_number INTEGER NOT NULL,
                        error_code TEXT,
                        error_hash TEXT,
                        prompt_hash TEXT,
                        response_hash TEXT,
                        unit_id TEXT
                    );
                    CREATE TABLE IF NOT EXISTS findings (
                        rowid BIGSERIAL PRIMARY KEY,
                        id TEXT NOT NULL,
                        run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                        unit_id TEXT NOT NULL,
                        code TEXT NOT NULL,
                        uncertain_cause TEXT,
                        raw_confidence DOUBLE PRECISION,
                        start_offset INTEGER,
                        end_offset INTEGER,
                        page INTEGER,
                        bbox_json TEXT,
                        legal_locators_json TEXT NOT NULL,
                        basis_json TEXT,
                        quote_resolution TEXT,
                        excerpt_hash TEXT,
                        synthesis_hash TEXT,
                        error_hash TEXT
                    );
                    CREATE INDEX IF NOT EXISTS idx_attempts_run_id_rowid
                        ON attempts (run_id, rowid);
                    CREATE INDEX IF NOT EXISTS idx_findings_run_id_rowid
                        ON findings (run_id, rowid);
                    """
                )

    def create_run(self, run: RunRecord) -> None:
        _validate_run(run)
        cost = run.cost
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO runs (
                        id, document_id, arm, input_hash, config_version,
                        prompt_bundle_version, corpus_snapshot_id, tool_bundle_version,
                        requested_model, returned_model, parameters_json, retry_policy,
                        concurrency, wall_budget_seconds,
                        measurement_valid, parent_run_id, interaction, status,
                        error_code, error_hash, source_hash,
                        monetary_cost_microunits, price_table_date, price_table_hash,
                        cost_unknown_reason, created_at, graph_topology_version,
                        interruption_reason, context_edge_count, call_unit_count,
                        finder_tool_turns, finder_search_calls, retrieval_cache_hits,
                        finder_budget_exhausted_units, verifier_tool_turns,
                        provision_reads,
                        defaulted_characterisations
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s, %s
                    )
                    """,
                    (
                        str(run.id),
                        str(run.document_id),
                        run.arm.value,
                        run.input_hash,
                        run.config_version,
                        run.prompt_bundle_version,
                        run.corpus_snapshot_id,
                        run.tool_bundle_version,
                        run.requested_model,
                        run.returned_model,
                        _parameters_json(run.parameters),
                        run.retry_policy,
                        run.concurrency,
                        run.wall_budget_seconds,
                        int(run.measurement_valid),
                        (
                            str(run.parent_run_id)
                            if run.parent_run_id is not None
                            else None
                        ),
                        run.interaction,
                        run.status,
                        run.error_code,
                        _hash_optional(run.error_detail),
                        _hash_optional(run.source_text),
                        cost.monetary_cost_microunits,
                        cost.price_table_date.isoformat()
                        if cost.price_table_date
                        else None,
                        cost.price_table_hash,
                        cost.unknown_reason,
                        _now(),
                        run.graph_topology_version,
                        run.interruption_reason,
                        run.context_edge_count,
                        run.call_unit_count,
                        run.finder_tool_turns,
                        run.finder_search_calls,
                        run.retrieval_cache_hits,
                        run.finder_budget_exhausted_units,
                        run.verifier_tool_turns,
                        run.provision_reads,
                        run.defaulted_characterisations,
                    ),
                )

    def record_attempt(self, attempt: AttemptRecord) -> None:
        _validate_attempt(attempt)
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO attempts (
                        id, run_id, requested_model, returned_model, prompt_version,
                        temperature, parameters_json, input_tokens, output_tokens,
                        latency_ms, status, retry_number, error_code, error_hash,
                        prompt_hash, response_hash, unit_id
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s, %s
                    )
                    """,
                    (
                        str(attempt.id),
                        str(attempt.run_id),
                        attempt.requested_model,
                        attempt.returned_model,
                        attempt.prompt_version,
                        attempt.temperature,
                        _parameters_json(attempt.parameters),
                        attempt.input_tokens,
                        attempt.output_tokens,
                        attempt.latency_ms,
                        attempt.status,
                        attempt.retry_number,
                        attempt.error_code,
                        _hash_optional(attempt.error_detail),
                        attempt.prompt_hash,
                        attempt.response_hash,
                        attempt.unit_id,
                    ),
                )
                if attempt.status == "success" and attempt.returned_model is not None:
                    cur.execute(
                        "UPDATE runs SET returned_model = COALESCE(returned_model, %s) "
                        "WHERE id = %s",
                        (attempt.returned_model, str(attempt.run_id)),
                    )

    def store_finding(self, finding: FindingRecord) -> None:
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO findings (
                        id, run_id, unit_id, code, uncertain_cause, raw_confidence,
                        start_offset, end_offset, page, bbox_json, legal_locators_json,
                        basis_json, quote_resolution, excerpt_hash, synthesis_hash,
                        error_hash
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                    """,
                    (
                        str(finding.id),
                        str(finding.run_id),
                        finding.unit_id,
                        finding.code.value,
                        (
                            finding.uncertain_cause.value
                            if finding.uncertain_cause
                            else None
                        ),
                        finding.raw_confidence,
                        finding.start_offset,
                        finding.end_offset,
                        finding.page,
                        json.dumps(finding.bbox) if finding.bbox is not None else None,
                        json.dumps(finding.legal_locators, separators=(",", ":")),
                        (
                            finding.basis.model_dump_json()
                            if finding.basis is not None
                            else None
                        ),
                        (
                            finding.quote_resolution.value
                            if finding.quote_resolution
                            else None
                        ),
                        _hash_optional(finding.excerpt),
                        _hash_optional(finding.synthesis_prose),
                        _hash_optional(finding.error_detail),
                    ),
                )

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

        assignments: list[str] = [
            "status = %s",
            "error_code = %s",
            "error_hash = %s",
            "finished_at = %s",
        ]
        values: list[object] = [
            status,
            totals.error_code,
            _hash_optional(totals.error_detail),
            _now(),
        ]
        if totals.source_text is not None:
            assignments.append("source_hash = %s")
            values.append(_hash_optional(totals.source_text))
        if totals.cost is not None:
            assignments.extend(
                [
                    "monetary_cost_microunits = %s",
                    "price_table_date = %s",
                    "price_table_hash = %s",
                    "cost_unknown_reason = %s",
                ]
            )
            values.extend(
                [
                    totals.cost.monetary_cost_microunits,
                    totals.cost.price_table_date.isoformat()
                    if totals.cost.price_table_date
                    else None,
                    totals.cost.price_table_hash,
                    totals.cost.unknown_reason,
                ]
            )
        for f in dataclasses.fields(totals):
            if f.name in (
                "error_code",
                "error_detail",
                "source_text",
                "cost",
            ):
                continue
            val = getattr(totals, f.name)
            if val is not None:
                assignments.append(f"{f.name} = %s")
                values.append(val)

        values.append(str(run_id))
        stmt = f"UPDATE runs SET {', '.join(assignments)} WHERE id = %s"

        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(stmt, tuple(values))
                if cur.rowcount == 0:
                    cur.execute("SELECT 1 FROM runs WHERE id = %s", (str(run_id),))
                    if cur.fetchone() is None:
                        raise KeyError(run_id)

    def set_run_monotonic_start(self, run_id: UUID, monotonic_start: float) -> None:
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE runs SET started_at_monotonic = %s "
                    "WHERE id = %s AND status = 'running'",
                    (monotonic_start, str(run_id)),
                )
                if cur.rowcount != 1:
                    raise KeyError(run_id)

    def elapsed_ms_for_run(self, run_id: UUID) -> float | None:
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT started_at_monotonic FROM runs WHERE id = %s",
                    (str(run_id),),
                )
                row = cur.fetchone()
        if row is None:
            return None
        started_at = row.get("started_at_monotonic")
        if started_at is None:
            return None
        return (self._clock() - float(started_at)) * 1000

    def mark_interrupted_runs_failed(self) -> int:
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE runs SET status = 'failed', "
                    "error_code = 'process_restarted', "
                    "finished_at = %s WHERE status = 'running'",
                    (_now(),),
                )
                return int(cur.rowcount or 0)

    def get_run(self, run_id: UUID) -> RunRecord | None:
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM runs WHERE id = %s", (str(run_id),))
                row = cur.fetchone()
        return _run_from_row(row) if row is not None else None

    def list_runs(self) -> list[RunRecord]:
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM runs ORDER BY created_at ASC")
                rows = cur.fetchall()
        return [_run_from_row(row) for row in rows]

    def list_attempts(self, run_id: UUID) -> list[AttemptRecord]:
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM attempts WHERE run_id = %s ORDER BY rowid ASC",
                    (str(run_id),),
                )
                rows = cur.fetchall()
        return [_attempt_from_row(row) for row in rows]

    def delete_document_runs(self, document_id: UUID) -> int:
        """Delete this document's runs; findings and attempts cascade with them."""
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM runs WHERE document_id = %s", (str(document_id),)
                )
                return cur.rowcount or 0

    def list_findings(self, run_id: UUID) -> list[FindingRecord]:
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM findings WHERE run_id = %s ORDER BY rowid ASC",
                    (str(run_id),),
                )
                rows = cur.fetchall()
        return [_finding_from_row(row) for row in rows]
