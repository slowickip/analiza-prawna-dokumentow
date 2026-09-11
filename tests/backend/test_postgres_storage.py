from __future__ import annotations

import inspect
import json
import re
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from contract_analyzer.domain import (
    ArmCode,
    AttemptStatus,
    FindingCode,
    ParameterValue,
)
from contract_analyzer.storage import (
    AttemptRecord,
    CostRecord,
    FindingRecord,
    MetadataStore,
    PostgresMetadataStore,
    RunRecord,
    RunTotals,
    open_metadata_store,
)
from contract_analyzer.storage.postgres import POOL_OPEN_TIMEOUT_SECONDS
from contract_analyzer.storage.records import _parameters_json


def _make_run(
    *,
    parameters: Mapping[str, ParameterValue] | None = None,
) -> RunRecord:
    return RunRecord(
        id=uuid4(),
        document_id=uuid4(),
        arm=ArmCode.MID,
        input_hash="hash-123",
        config_version="v1",
        prompt_bundle_version="v1",
        corpus_snapshot_id="v1",
        tool_bundle_version="v1",
        requested_model="deepseek-v4-flash",
        parameters=parameters or {},
        retry_policy="bounded-3",
        concurrency=2,
        wall_budget_seconds=60.0,
        measurement_valid=True,
        cost=CostRecord.unknown("no_price_table"),
    )


def _make_attempt(
    *,
    status: AttemptStatus = "success",
    returned_model: str | None = "deepseek-v4-flash",
    parameters: Mapping[str, ParameterValue] | None = None,
) -> AttemptRecord:
    return AttemptRecord(
        id=uuid4(),
        run_id=uuid4(),
        requested_model="deepseek-v4-flash",
        returned_model=returned_model,
        prompt_version="v1",
        temperature=0.0,
        parameters=parameters or {},
        input_tokens=100,
        output_tokens=50,
        latency_ms=250.0,
        status=status,
        retry_number=0,
        error_code=None,
        prompt_hash="p-hash",
        response_hash="r-hash",
    )


@contextmanager
def _mock_pg_store() -> Iterator[tuple[PostgresMetadataStore, MagicMock]]:
    store = PostgresMetadataStore.__new__(PostgresMetadataStore)
    cursor = MagicMock()
    cursor.rowcount = 1
    connection = MagicMock()
    connection.__enter__.return_value = connection
    connection.cursor.return_value.__enter__.return_value = cursor
    with patch.object(PostgresMetadataStore, "_connection", return_value=connection):
        yield store, cursor


def test_open_metadata_store_dispatch(tmp_path) -> None:
    in_memory_store = open_metadata_store(tmp_path / "metadata.sqlite3")
    assert isinstance(in_memory_store, MetadataStore)

    with (
        patch.object(PostgresMetadataStore, "_open_pool"),
        patch.object(PostgresMetadataStore, "_initialize"),
        patch.object(PostgresMetadataStore, "mark_interrupted_runs_failed"),
    ):
        pg_store = open_metadata_store("postgresql://app:secret@localhost:5432/db")
        assert isinstance(pg_store, PostgresMetadataStore)


def test_postgres_metadata_store_operations() -> None:
    mock_cursor = MagicMock()
    mock_conn = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

    with patch("psycopg.connect", return_value=mock_conn):
        store = PostgresMetadataStore("postgresql://app:secret@localhost:5432/db")

        # 1. create_run
        run = _make_run()
        store.create_run(run)
        assert mock_cursor.execute.call_count >= 2  # _initialize + create_run
        insert_call = mock_cursor.execute.call_args_list[-1]
        assert "INSERT INTO runs" in insert_call[0][0]
        assert "%s" in insert_call[0][0]
        assert str(run.id) in insert_call[0][1]

        # 2. record_attempt
        attempt = AttemptRecord(
            id=uuid4(),
            run_id=run.id,
            requested_model="deepseek-v4-flash",
            returned_model="deepseek-v4-flash",
            prompt_version="v1",
            temperature=0.0,
            parameters={},
            input_tokens=100,
            output_tokens=50,
            latency_ms=250.0,
            status="success",
            retry_number=0,
            error_code=None,
            prompt_hash="p-hash",
            response_hash="r-hash",
        )
        store.record_attempt(attempt)
        attempt_calls = [
            c
            for c in mock_cursor.execute.call_args_list
            if "INSERT INTO attempts" in c[0][0]
        ]
        assert len(attempt_calls) == 1

        # 3. store_finding
        finding = FindingRecord(
            id=uuid4(),
            run_id=run.id,
            unit_id="u1",
            code=FindingCode.CONSISTENT,
            legal_locators=("loc1",),
        )
        store.store_finding(finding)
        finding_calls = [
            c
            for c in mock_cursor.execute.call_args_list
            if "INSERT INTO findings" in c[0][0]
        ]
        assert len(finding_calls) == 1

        # 4. finish_run
        mock_cursor.rowcount = 1
        store.finish_run(
            run.id,
            "completed",
            totals=RunTotals(
                elapsed_ms=500.0,
                provision_reads=2,
                defaulted_characterisations=1,
            ),
        )
        update_calls = [
            c
            for c in mock_cursor.execute.call_args_list
            if "UPDATE runs SET" in c[0][0] and "status = %s" in c[0][0]
        ]
        assert len(update_calls) == 1
        assert "provision_reads = %s" in update_calls[0][0][0]
        assert 2 in update_calls[0][0][1]
        assert "defaulted_characterisations = %s" in update_calls[0][0][0]
        assert 1 in update_calls[0][0][1]


_ALL_RUN_TOTALS: dict[str, Any] = {
    "elapsed_ms": 500.0,
    "returned_model": "deepseek-v4-flash",
    "context_edge_count": 2,
    "call_unit_count": 8,
    "finder_tool_turns": 12,
    "finder_search_calls": 9,
    "retrieval_cache_hits": 3,
    "finder_budget_exhausted_units": 0,
    "verifier_tool_turns": 7,
    "provision_reads": 4,
    "defaulted_characterisations": 1,
}


def test_every_write_binds_one_value_per_column_it_names() -> None:
    """The column list, the placeholders and the value tuple are three lists.

    They are hand-maintained inside one statement, and the rest of the suite
    exercises the in-memory store, so a column added to two of the three reaches
    a live database rather than a failing test. Both halves have happened:
    retrieval_cache_hits went into the runs INSERT with a value and no
    placeholder, and record_attempt's columns and placeholders agreed while its
    value tuple was one short. The deployed backend answered createRun with a 500
    each time until the three were matched up, so every write the store performs
    is executed here rather than one of them read out of the source.
    """
    run = _make_run()
    attempt = AttemptRecord(
        id=uuid4(),
        run_id=run.id,
        requested_model="deepseek-v4-flash",
        returned_model="deepseek-v4-flash",
        prompt_version="researcher.search",
        temperature=0.0,
        parameters={},
        input_tokens=100,
        output_tokens=50,
        latency_ms=250.0,
        status="success",
        retry_number=0,
        error_code=None,
        unit_id="u1",
        prompt_hash="p-hash",
        response_hash="r-hash",
    )
    finding = FindingRecord(
        id=uuid4(),
        run_id=run.id,
        unit_id="u1",
        code=FindingCode.CONSISTENT,
        legal_locators=("loc1",),
    )
    writes = (
        ("create_run", lambda store: store.create_run(run)),
        ("record_attempt", lambda store: store.record_attempt(attempt)),
        ("store_finding", lambda store: store.store_finding(finding)),
        # finish_run builds its assignment list at runtime, so it is checked with
        # every optional supplied: that is the only shape in which a column added
        # without its value shows up.
        (
            "finish_run",
            lambda store: store.finish_run(
                run.id, "completed", totals=RunTotals(**_ALL_RUN_TOTALS)
            ),
        ),
    )
    for name, write in writes:
        store = PostgresMetadataStore.__new__(PostgresMetadataStore)
        cursor = MagicMock()
        # finish_run reads it back to tell a missing run from an updated one.
        cursor.rowcount = 1
        connection = MagicMock()
        connection.__enter__.return_value = connection
        connection.cursor.return_value.__enter__.return_value = cursor
        with patch.object(
            PostgresMetadataStore, "_connection", return_value=connection
        ):
            write(store)

        assert cursor.execute.call_args_list, f"{name} issued no statement"
        for call in cursor.execute.call_args_list:
            sql, values = call[0]
            assert sql.count("%s") == len(values), (
                f"{name}: {sql.count('%s')} placeholders, {len(values)} values"
            )
            named = re.search(r"INSERT INTO \w+ \((.*?)\)", sql, re.S)
            if named is None:
                continue
            columns = [c.strip() for c in named.group(1).split(",") if c.strip()]
            assert len(columns) == len(values), (
                f"{name}: {len(columns)} columns, {len(values)} values"
            )


def test_no_table_is_altered_before_it_is_created() -> None:
    """A fresh database runs this DDL top to bottom, and stops at the first error.

    ALTER TABLE ... ADD COLUMN IF NOT EXISTS is idempotent for an existing table
    and fatal for a missing one, so an ALTER placed above its CREATE works on
    every developer machine that already has the schema and breaks every new
    deployment. That is how unit_id shipped: the ALTER sat two lines above the
    CREATE TABLE attempts it depended on.
    """
    from contract_analyzer.storage import postgres

    source = inspect.getsource(postgres)
    created: set[str] = set()
    statements = re.findall(r"(CREATE TABLE IF NOT EXISTS|ALTER TABLE)\s+(\w+)", source)
    assert statements, "no DDL found to check"
    for kind, table in statements:
        if kind == "ALTER TABLE":
            assert table in created, (
                f"ALTER TABLE {table} runs before CREATE TABLE {table}"
            )
        else:
            created.add(table)


def test_record_attempt_updates_run_returned_model_only_on_success() -> None:
    """record_attempt writes returned_model on success and skips it otherwise."""
    expected_sql = (
        "UPDATE runs SET returned_model = COALESCE(returned_model, %s) WHERE id = %s"
    )

    with _mock_pg_store() as (store, cursor):
        attempt = _make_attempt(status="success", returned_model="deepseek-v4-flash")
        store.record_attempt(attempt)
        assert len(cursor.execute.call_args_list) == 2
        sql, values = cursor.execute.call_args_list[1][0]
        assert sql == expected_sql
        assert values == (attempt.returned_model, str(attempt.run_id))

    with _mock_pg_store() as (store, cursor):
        failed_attempt = _make_attempt(
            status="transport_error", returned_model="deepseek-v4-flash"
        )
        store.record_attempt(failed_attempt)
        assert len(cursor.execute.call_args_list) == 1
        assert not any(
            "UPDATE runs" in call[0][0] for call in cursor.execute.call_args_list
        )

    with _mock_pg_store() as (store, cursor):
        no_model_attempt = _make_attempt(status="success", returned_model=None)
        store.record_attempt(no_model_attempt)
        assert len(cursor.execute.call_args_list) == 1
        assert not any(
            "UPDATE runs" in call[0][0] for call in cursor.execute.call_args_list
        )


def test_store_finding_binds_bbox_and_compact_legal_locators() -> None:
    """store_finding binds bbox and compact legal locators JSON."""
    finding_with_bbox = FindingRecord(
        id=uuid4(),
        run_id=uuid4(),
        unit_id="u1",
        code=FindingCode.CONSISTENT,
        bbox=(0.1, 0.2, 0.3, 0.4),
        legal_locators=("art_1", "art_2"),
    )
    with _mock_pg_store() as (store, cursor):
        store.store_finding(finding_with_bbox)
        assert len(cursor.execute.call_args_list) == 1
        sql, values = cursor.execute.call_args_list[0][0]
        assert "INSERT INTO findings" in sql
        assert values[9] == json.dumps(finding_with_bbox.bbox)
        assert values[10] == '["art_1","art_2"]'

    finding_without_bbox = FindingRecord(
        id=uuid4(),
        run_id=uuid4(),
        unit_id="u1",
        code=FindingCode.CONSISTENT,
        bbox=None,
        legal_locators=("art_1", "art_2"),
    )
    with _mock_pg_store() as (store, cursor):
        store.store_finding(finding_without_bbox)
        assert len(cursor.execute.call_args_list) == 1
        sql, values = cursor.execute.call_args_list[0][0]
        assert "INSERT INTO findings" in sql
        assert values[9] is None
        assert values[10] == '["art_1","art_2"]'


def test_parameters_column_binds_canonically_sorted_compact_json() -> None:
    """parameters column binds sorted keys and compact JSON separators."""
    params: dict[str, ParameterValue] = {"b": 1, "a": 0}
    expected = '{"a":0,"b":1}'

    with _mock_pg_store() as (store, cursor):
        run = _make_run(parameters=params)
        store.create_run(run)
        sql, values = cursor.execute.call_args_list[0][0]
        assert "INSERT INTO runs" in sql
        assert values[10] == expected

    with _mock_pg_store() as (store, cursor):
        attempt = _make_attempt(parameters=params)
        store.record_attempt(attempt)
        sql, values = cursor.execute.call_args_list[0][0]
        assert "INSERT INTO attempts" in sql
        assert values[6] == expected


def test_parameters_json_refuses_invalid_key_code() -> None:
    """_parameters_json rejects parameter keys that are not valid codes."""
    with pytest.raises(ValueError, match="parameter name"):
        _parameters_json({"invalid key!": 1})
    with pytest.raises(ValueError, match="parameter name"):
        _parameters_json({"NotSnakeCase": 1})
    with pytest.raises(ValueError, match="parameter name"):
        _parameters_json({"123numeric": 1})


def test_the_pool_reuses_one_connection_across_many_writes() -> None:
    """A statement used to buy its own connection, at a handshake each."""
    opened: list[int] = []
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = None
    mock_conn = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

    def counting_factory(conninfo: str = "", **kwargs: Any) -> Any:
        opened.append(1)
        return mock_conn

    store = PostgresMetadataStore(
        "postgresql://app:secret@localhost:5432/db",
        connection_factory=counting_factory,
    )
    before = len(opened)
    for _ in range(5):
        store.list_runs()
        store.create_run(_make_run())

    assert len(opened) == before, (
        f"{len(opened) - before} connections were opened for five reads"
    )


def test_a_database_it_cannot_reach_fails_when_the_store_is_built() -> None:
    """An unreachable database stops the store rather than blocking its callers.

    The pool retries in the background by default, so a caller would wait for
    the acquire timeout on every statement while the process looked healthy.
    """

    from psycopg_pool import PoolTimeout

    attempts: list[int] = []

    def faulty_factory(conninfo: str = "", **kwargs: Any) -> Any:
        attempts.append(1)
        raise RuntimeError("no route to the database")

    # Patch out everything that runs a statement, so the only thing left that
    # can raise is opening the pool. Without this the test passes against a
    # lazily-opened pool too, because the first statement times out instead.
    started = time.monotonic()
    with (
        patch.object(PostgresMetadataStore, "_initialize"),
        patch.object(PostgresMetadataStore, "mark_interrupted_runs_failed"),
        pytest.raises(PoolTimeout),
    ):
        PostgresMetadataStore(
            "postgresql://app:secret@localhost:5432/db",
            connection_factory=faulty_factory,
        )
    elapsed = time.monotonic() - started

    assert attempts, "the store never tried to connect"
    assert elapsed < POOL_OPEN_TIMEOUT_SECONDS + 5.0, (
        f"construction took {elapsed:.1f}s; it must fail, not retry for ever"
    )


def test_both_foreign_key_lookups_are_indexed() -> None:
    """Postgres does not index a foreign key, and both tables are filtered by it."""
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = None
    mock_conn = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

    PostgresMetadataStore(
        "postgresql://app:secret@localhost:5432/db",
        connection_factory=lambda conninfo="", **kwargs: mock_conn,
    )
    ddl = " ".join(str(call.args[0]) for call in mock_cursor.execute.call_args_list)
    for table in ("attempts", "findings"):
        assert f"ON {table} (run_id, rowid)" in ddl, (
            f"{table} has no index over (run_id, rowid): {ddl[:200]}"
        )


def test_closing_the_store_closes_its_pool() -> None:
    """The store holds connections and pool workers until something closes it.

    The shutdown test drives a stand-in, so only this one sees the real pool
    change state. It builds the store through its own constructor, because a
    hand-assembled instance would pass without the pool ever having opened.
    """
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = None
    mock_conn = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
    store = PostgresMetadataStore(
        "postgresql://app:secret@localhost:5432/db",
        connection_factory=lambda conninfo="", **kwargs: mock_conn,
    )
    assert not store._pool.closed

    store.close()

    assert store._pool.closed


def test_read_only_open_executes_no_statement() -> None:
    """Opening a diagnostic store executes no SQL that could alter live runs."""
    executed: list[str] = []
    cursor = MagicMock()
    cursor.execute.side_effect = lambda sql, *args: executed.append(str(sql))
    connection = MagicMock()
    connection.__enter__.return_value = connection
    connection.cursor.return_value.__enter__.return_value = cursor

    with patch.object(PostgresMetadataStore, "_open_pool"):
        with patch.object(
            PostgresMetadataStore, "_connection", return_value=connection
        ):
            reader = PostgresMetadataStore(
                "postgresql://app:secret@localhost:5432/db", read_only=True
            )
            assert reader.read_only is True
            assert executed == []

            writer = PostgresMetadataStore("postgresql://app:secret@localhost:5432/db")
            assert writer.read_only is False

    joined = " ".join(executed).upper()
    assert "CREATE TABLE" in joined, "the writing open still creates the schema"
    assert "UPDATE RUNS SET STATUS = 'FAILED'" in joined.replace("\n", " "), (
        "the writing open still reconciles interrupted runs"
    )


def test_open_metadata_store_passes_read_only_through() -> None:
    """The flag has to survive the factory, which is what callers use."""
    with (
        patch.object(PostgresMetadataStore, "_open_pool"),
        patch.object(PostgresMetadataStore, "_initialize") as initialize,
        patch.object(PostgresMetadataStore, "mark_interrupted_runs_failed") as sweep,
    ):
        store = open_metadata_store(
            "postgresql://app:secret@localhost:5432/db", read_only=True
        )
        assert isinstance(store, PostgresMetadataStore)
        initialize.assert_not_called()
        sweep.assert_not_called()

        open_metadata_store("postgresql://app:secret@localhost:5432/db")
        initialize.assert_called_once()
        sweep.assert_called_once()
