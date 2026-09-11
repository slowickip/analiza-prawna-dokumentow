"""The row mapper must carry every stored run column the API reads back.

A run's elapsed time and timestamps were written by finish_run and stored, yet
the mapper never copied them into the record, so every PostgreSQL-backed run
reported null metrics while the in-memory store reported them fine. This pins
the round trip on the mapper alone, without a database.
"""

from __future__ import annotations

from uuid import uuid4

from contract_analyzer.storage.record_rows import _run_from_row


def _full_row() -> dict[str, object]:
    return {
        "id": str(uuid4()),
        "document_id": str(uuid4()),
        "arm": "mid",
        "input_hash": "in",
        "config_version": "run-config-v1",
        "prompt_bundle_version": "p",
        "corpus_snapshot_id": "c",
        "tool_bundle_version": "t",
        "requested_model": "m",
        "returned_model": "m",
        "parameters_json": "{}",
        "retry_policy": "bounded-3",
        "concurrency": 2,
        "wall_budget_seconds": 900.0,
        "measurement_valid": 1,
        "parent_run_id": None,
        "interaction": None,
        "status": "completed",
        "error_code": None,
        "error_hash": None,
        "source_hash": None,
        "monetary_cost_microunits": None,
        "price_table_date": None,
        "price_table_hash": None,
        "cost_unknown_reason": "price_table_unavailable",
        "created_at": "2026-09-03T14:45:39+00:00",
        "finished_at": "2026-09-03T14:49:23+00:00",
        "elapsed_ms": 223474.5,
        "graph_topology_version": "worksheet-unit-graph-v2",
        "interruption_reason": None,
        "context_edge_count": 0,
        "call_unit_count": 1,
        "finder_tool_turns": 5,
        "finder_search_calls": 10,
        "finder_budget_exhausted_units": 0,
        "verifier_tool_turns": 11,
        "provision_reads": 11,
        "defaulted_characterisations": 2,
    }


def test_run_row_carries_elapsed_time_and_timestamps() -> None:
    record = _run_from_row(_full_row())
    assert record.elapsed_ms == 223474.5
    assert record.created_at == "2026-09-03T14:45:39+00:00"
    assert record.finished_at == "2026-09-03T14:49:23+00:00"


def test_run_row_carries_every_counter_the_run_recorded() -> None:
    record = _run_from_row(_full_row())
    assert (record.finder_tool_turns, record.finder_search_calls) == (5, 10)
    assert (record.verifier_tool_turns, record.provision_reads) == (11, 11)
    assert record.defaulted_characterisations == 2
    assert record.call_unit_count == 1
