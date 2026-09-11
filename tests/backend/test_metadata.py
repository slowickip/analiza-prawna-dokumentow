from __future__ import annotations

import hashlib
from dataclasses import replace
from datetime import date
from pathlib import Path
from uuid import uuid4

import pytest

from contract_analyzer.domain import (
    ArmCode,
    EmittedBasis,
    FindingCode,
    ForceScope,
    ForceState,
    ForceValue,
    ProvisionKind,
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
    SchemaVersionError,
)


def run_record(
    *,
    fixture_text: str | None = None,
    graph_topology_version: str | None = "unit-subgraph-v1",
    interruption_reason: str | None = None,
) -> RunRecord:
    return RunRecord(
        id=uuid4(),
        document_id=uuid4(),
        arm=ArmCode.MID,
        input_hash="input-sha256",
        config_version="config-v1",
        prompt_bundle_version="prompts-v1",
        corpus_snapshot_id="corpus-v1",
        tool_bundle_version="tools-v1",
        requested_model="deepseek-v4-flash",
        parameters={"temperature": 0.0, "seed": 7},
        retry_policy="bounded-3-v1",
        concurrency=2,
        wall_budget_seconds=300.0,
        measurement_valid=True,
        cost=CostRecord.unknown("no_auditable_price_table"),
        graph_topology_version=graph_topology_version,
        interruption_reason=interruption_reason,
        source_text=fixture_text,
        error_detail=fixture_text,
    )


def attempt_record(run_id, *, fixture_text: str | None = None) -> AttemptRecord:
    prompt_hash = (
        hashlib.sha256(fixture_text.encode()).hexdigest() if fixture_text else ""
    )
    response_hash = (
        hashlib.sha256(fixture_text.encode()).hexdigest() if fixture_text else None
    )
    return AttemptRecord(
        id=uuid4(),
        run_id=run_id,
        requested_model="deepseek-v4-flash",
        returned_model="deepseek-v4-flash-20260828",
        prompt_version="basis-v1",
        temperature=0.0,
        parameters={"seed": 7},
        input_tokens=17,
        output_tokens=5,
        latency_ms=12.5,
        status="success",
        retry_number=0,
        error_code=None,
        prompt_hash=prompt_hash,
        response_hash=response_hash,
        prompt=fixture_text,
        response=fixture_text,
        error_detail=fixture_text,
    )


def finding_record(run_id, *, fixture_text: str | None = None) -> FindingRecord:
    return FindingRecord(
        id=uuid4(),
        run_id=run_id,
        unit_id="unit-7",
        code=FindingCode.CONSISTENT,
        start_offset=10,
        end_offset=24,
        page=2,
        bbox=(1.0, 2.0, 3.0, 4.0),
        legal_locators=("https://api.sejm.gov.pl/eli/acts/DU/2023/725",),
        excerpt=fixture_text,
        synthesis_prose=fixture_text,
        error_detail=fixture_text,
    )


def emitted_basis() -> EmittedBasis:
    return EmittedBasis(
        provision_locator="art. 1",
        act_identifier="DU/2026/795",
        act_force=ForceState(
            value=ForceValue.IN_FORCE,
            scope=ForceScope.ACT,
            snapshot_date=date(2026, 5, 19),
            source_locator="https://api.sejm.gov.pl/eli/acts/DU/2026/795",
        ),
        provision_force=ForceState(
            value=ForceValue.UNDETERMINED,
            scope=ForceScope.PROVISION,
            snapshot_date=date(2026, 5, 19),
            source_locator="https://api.sejm.gov.pl/eli/acts/DU/2026/795",
        ),
        character_kind=ProvisionKind.IMPERATIVE,
    )


def test_schema_is_versioned_wal_and_foreign_keys_are_enforced(tmp_path: Path) -> None:
    path = tmp_path / "metadata.sqlite3"
    store = MetadataStore(path)
    assert store.SCHEMA_VERSION == 11
    assert store.path == path
    with pytest.raises(KeyError):
        store.record_attempt(attempt_record(uuid4()))


def test_unknown_schema_version_is_rejected() -> None:
    with pytest.raises(SchemaVersionError, match="unsupported_metadata_schema_99"):
        raise SchemaVersionError("unsupported_metadata_schema_99")


def test_run_attempt_finding_and_finish_are_queryable(tmp_path: Path) -> None:
    store = MetadataStore(tmp_path / "metadata.sqlite3")
    run = run_record()
    attempt = attempt_record(run.id)
    finding = finding_record(run.id)

    store.create_run(run)
    store.record_attempt(attempt)
    store.store_finding(finding)
    store.finish_run(run.id, "completed", totals=RunTotals(provision_reads=2))

    saved_run = store.get_run(run.id)
    assert saved_run is not None
    assert saved_run.status == "completed"
    assert saved_run.graph_topology_version == "unit-subgraph-v1"
    assert saved_run.provision_reads == 2
    assert saved_run.cost.unknown_reason == "no_auditable_price_table"
    assert store.list_attempts(run.id) == [
        replace(attempt, prompt=None, response=None, error_detail=None)
    ]
    assert store.list_findings(run.id) == [
        replace(finding, excerpt=None, synthesis_prose=None, error_detail=None)
    ]


def test_interruption_reason_round_trips(tmp_path: Path) -> None:
    store = MetadataStore(tmp_path / "metadata.sqlite3")
    run = run_record(interruption_reason="wall_time")

    store.create_run(run)

    saved = store.get_run(run.id)
    assert saved is not None
    assert saved.interruption_reason == "wall_time"


def test_known_cost_requires_a_date_pinned_price_table() -> None:
    known = CostRecord.known(
        monetary_cost_microunits=123,
        price_table_date=date(2026, 8, 28),
        price_table_hash="price-table-sha256",
    )
    assert known.monetary_cost_microunits == 123

    with pytest.raises(ValueError, match="unknown_reason"):
        CostRecord()
    with pytest.raises(ValueError, match="price table"):
        CostRecord(monetary_cost_microunits=123)


def test_metadata_database_and_sidecars_contain_no_fixture_text(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    fixture_text = "PRIVACY-CANARY genuine contract response and error prose 90817"
    path = tmp_path / "metadata.sqlite3"
    store = MetadataStore(path)
    run = run_record(fixture_text=fixture_text)

    store.create_run(run)
    store.record_attempt(attempt_record(run.id, fixture_text=fixture_text))
    store.store_finding(finding_record(run.id, fixture_text=fixture_text))
    store.finish_run(
        run.id,
        "failed",
        totals=RunTotals(
            error_code="model_response_invalid_envelope", error_detail=fixture_text
        ),
    )

    saved = store.get_run(run.id)
    assert saved is not None
    assert saved.source_text is None or saved.source_text != fixture_text
    assert fixture_text not in caplog.text


def test_uncertain_finding_round_trips_with_cause_and_confidence(
    tmp_path: Path,
) -> None:
    store = MetadataStore(tmp_path / "metadata.sqlite3")
    run = run_record()
    store.create_run(run)

    finding = FindingRecord(
        id=uuid4(),
        run_id=run.id,
        unit_id="unit-1",
        code=FindingCode.UNCERTAIN,
        uncertain_cause=UncertainCause.RELATION_BELOW_THRESHOLD,
        raw_confidence=0.42,
    )
    store.store_finding(finding)

    saved = store.list_findings(run.id)
    assert len(saved) == 1
    assert saved[0].code is FindingCode.UNCERTAIN
    assert saved[0].uncertain_cause is UncertainCause.RELATION_BELOW_THRESHOLD
    assert saved[0].raw_confidence == pytest.approx(0.42)


def test_non_uncertain_finding_round_trips_without_cause_or_confidence(
    tmp_path: Path,
) -> None:
    store = MetadataStore(tmp_path / "metadata.sqlite3")
    run = run_record()
    store.create_run(run)

    finding = FindingRecord(
        id=uuid4(),
        run_id=run.id,
        unit_id="unit-2",
        code=FindingCode.CONSISTENT,
    )
    store.store_finding(finding)

    saved = store.list_findings(run.id)
    assert len(saved) == 1
    assert saved[0].code is FindingCode.CONSISTENT
    assert saved[0].uncertain_cause is None
    assert saved[0].raw_confidence is None


def test_quote_resolution_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "metadata.sqlite3"
    store = MetadataStore(path)
    run = run_record()
    store.create_run(run)
    finding = FindingRecord(
        id=uuid4(),
        run_id=run.id,
        unit_id="unit-v9",
        code=FindingCode.CONSISTENT,
        quote_resolution=QuoteResolution.QUOTE_FABRICATED,
    )
    store.store_finding(finding)
    findings = store.list_findings(run.id)
    assert len(findings) == 1
    assert findings[0].quote_resolution == QuoteResolution.QUOTE_FABRICATED
