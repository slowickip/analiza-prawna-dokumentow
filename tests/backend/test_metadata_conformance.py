"""Conformance tests verifying behavioural equivalence between storage engines."""

from __future__ import annotations

import os
from collections.abc import Iterator
from datetime import date
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from contract_analyzer.domain import (
    ArmCode,
    FindingCode,
    QuoteResolution,
    UncertainCause,
)
from contract_analyzer.storage import (
    AttemptRecord,
    CostRecord,
    FindingRecord,
    MetadataStore,
    RunRecord,
    RunTotals,
    open_metadata_store,
)


def _make_run(document_id: UUID | None = None) -> RunRecord:
    return RunRecord(
        id=uuid4(),
        document_id=document_id or uuid4(),
        arm=ArmCode.MID,
        input_hash="hash-123",
        config_version="v1",
        prompt_bundle_version="v1",
        corpus_snapshot_id="v1",
        tool_bundle_version="v1",
        requested_model="deepseek-v4-flash",
        parameters={"temp": 0.0},
        retry_policy="bounded-3",
        concurrency=2,
        wall_budget_seconds=60.0,
        measurement_valid=True,
        cost=CostRecord.unknown("no_price_table"),
    )


@pytest.fixture(params=["memory", "postgres"])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[MetadataStore]:
    if request.param == "memory":
        yield open_metadata_store(tmp_path / f"test-{uuid4()}.sqlite3")
    elif request.param == "postgres":
        url = os.environ.get("TEST_DATABASE_URL")
        if not url:
            pytest.skip("No PostgreSQL URL configured (set TEST_DATABASE_URL)")
        store_instance = open_metadata_store(url)
        yield store_instance


def test_conformance_run_lifecycle_and_finish_totals(store: MetadataStore) -> None:
    run = _make_run()
    store.create_run(run)

    fetched = store.get_run(run.id)
    assert fetched is not None
    assert fetched.id == run.id
    assert fetched.status == "running"
    assert fetched.parameters == {"temp": 0.0}

    totals = RunTotals(
        elapsed_ms=123.4,
        call_unit_count=3,
        finder_tool_turns=5,
        finder_search_calls=4,
        retrieval_cache_hits=2,
        cost=CostRecord.known(
            monetary_cost_microunits=5000,
            price_table_date=date(2026, 9, 1),
            price_table_hash="hash-abc",
        ),
    )
    store.finish_run(run.id, "completed", totals=totals)

    finished = store.get_run(run.id)
    assert finished is not None
    assert finished.status == "completed"
    assert finished.elapsed_ms == 123.4
    assert finished.call_unit_count == 3
    assert finished.finder_tool_turns == 5
    assert finished.finder_search_calls == 4
    assert finished.retrieval_cache_hits == 2
    assert finished.cost.monetary_cost_microunits == 5000


def test_conformance_attempts_and_findings(store: MetadataStore) -> None:
    run = _make_run()
    store.create_run(run)

    attempt = AttemptRecord(
        id=uuid4(),
        run_id=run.id,
        requested_model="deepseek-v4-flash",
        returned_model="deepseek-v4-flash",
        prompt_version="researcher.search",
        temperature=0.0,
        parameters={"step": 1},
        input_tokens=150,
        output_tokens=60,
        latency_ms=300.0,
        status="success",
        retry_number=0,
        error_code=None,
        unit_id="u-1",
        prompt_hash="p-hash",
        response_hash="r-hash",
    )
    store.record_attempt(attempt)
    run_after_attempt = store.get_run(run.id)
    assert run_after_attempt is not None
    assert run_after_attempt.returned_model == "deepseek-v4-flash"

    attempts = store.list_attempts(run.id)
    assert len(attempts) == 1
    assert attempts[0].id == attempt.id
    assert attempts[0].input_tokens == 150
    assert attempts[0].unit_id == "u-1"

    finding = FindingRecord(
        id=uuid4(),
        run_id=run.id,
        unit_id="u-1",
        code=FindingCode.UNCERTAIN,
        uncertain_cause=UncertainCause.RELATION_BELOW_THRESHOLD,
        legal_locators=("art_1", "art_2"),
        start_offset=10,
        end_offset=50,
        quote_resolution=QuoteResolution.RESOLVED,
    )
    store.store_finding(finding)

    findings = store.list_findings(run.id)
    assert len(findings) == 1
    assert findings[0].id == finding.id
    assert findings[0].code == FindingCode.UNCERTAIN
    assert findings[0].uncertain_cause == UncertainCause.RELATION_BELOW_THRESHOLD
    assert findings[0].legal_locators == ("art_1", "art_2")
    assert findings[0].start_offset == 10
    assert findings[0].end_offset == 50


def test_conformance_mark_interrupted_runs_failed(store: MetadataStore) -> None:
    run = _make_run()
    store.create_run(run)

    count = store.mark_interrupted_runs_failed()
    assert count >= 1

    interrupted = store.get_run(run.id)
    assert interrupted is not None
    assert interrupted.status == "failed"
    assert interrupted.error_code == "process_restarted"


def test_conformance_delete_document_runs_cascades(store: MetadataStore) -> None:
    doc_id = uuid4()
    run = _make_run(document_id=doc_id)
    store.create_run(run)

    attempt = AttemptRecord(
        id=uuid4(),
        run_id=run.id,
        requested_model="m",
        returned_model="m",
        prompt_version="p",
        temperature=0.0,
        parameters={},
        input_tokens=10,
        output_tokens=10,
        latency_ms=10.0,
        status="success",
        retry_number=0,
        error_code=None,
    )
    store.record_attempt(attempt)

    finding = FindingRecord(
        id=uuid4(),
        run_id=run.id,
        unit_id="u1",
        code=FindingCode.CONSISTENT,
        legal_locators=("loc1",),
    )
    store.store_finding(finding)

    deleted_count = store.delete_document_runs(doc_id)
    assert deleted_count >= 1
    assert store.get_run(run.id) is None
    assert len(store.list_attempts(run.id)) == 0
    assert len(store.list_findings(run.id)) == 0
