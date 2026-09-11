"""Parity between the runs forming one comparison.

The evaluation protocol registers parity as a procedure: a divergence on any dimension
invalidates the comparison. This parametrized test breaks each of the ten dimensions
separately and requires the report to name exactly that one.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Literal
from uuid import uuid4

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import check_runs  # noqa: E402
from contract_analyzer.agents.parity import (  # noqa: E402
    PARITY_FIELDS,
    compare_arm_parity,
)
from contract_analyzer.domain import ArmCode  # noqa: E402
from contract_analyzer.storage import CostRecord, MetadataStore, RunRecord  # noqa: E402

# Each dimension paired with a mutation that must move it, so the parametrization
# below covers PARITY_FIELDS exhaustively rather than sampling it.
_MUTATIONS: dict[str, dict[str, object]] = {
    "input_hash": {"input_hash": "different-input"},
    "requested_model": {"requested_model": "other-model"},
    "returned_model": {"returned_model": "other-returned"},
    "prompt_bundle_version": {"prompt_bundle_version": "prompt-bundle-v2"},
    "corpus_snapshot_id": {"corpus_snapshot_id": "snapshot-2"},
    "tool_bundle_version": {"tool_bundle_version": "tools-v2"},
    "parameters": {"parameters": {"temperature": 0.1}},
    "retry_policy": {"retry_policy": "bounded-5"},
    "concurrency": {"concurrency": 4},
    "budgets": {"wall_budget_seconds": 1.0},
}


def _base_run(**overrides: object) -> RunRecord:
    run = RunRecord(
        id=uuid4(),
        document_id=uuid4(),
        arm=ArmCode.MID,
        input_hash="input-hash",
        config_version="run-config-v1",
        prompt_bundle_version="prompt-bundle-v1",
        corpus_snapshot_id="snapshot-1",
        tool_bundle_version="tools-v1",
        requested_model="deepseek-v4-flash",
        parameters={"temperature": 0.0},
        retry_policy="bounded-3",
        concurrency=2,
        wall_budget_seconds=300.0,
        measurement_valid=True,
        cost=CostRecord.unknown("price_table_unavailable"),
        returned_model="deepseek-v4-flash-20260828",
    )
    if overrides:
        return replace(run, **overrides)  # type: ignore[arg-type]
    return run


def test_every_parity_dimension_has_a_mutation() -> None:
    """A new dimension must fail here until someone says how to break it."""
    assert set(_MUTATIONS) == set(PARITY_FIELDS)


@pytest.mark.parametrize("field", PARITY_FIELDS)
def test_each_parity_mismatch_is_reported_alone(field: str) -> None:
    """Breaking one dimension must name that one and no other."""
    mutated = _MUTATIONS[field]

    result = compare_arm_parity((_base_run(), _base_run(**mutated)))

    assert result.measurement_valid is False
    assert result.mismatched_fields == (field,)


def test_matching_runs_are_measurement_valid() -> None:
    """The arms differ by design; that difference is not a parity dimension."""
    runs = (_base_run(), _base_run(arm=ArmCode.OFF), _base_run(arm=ArmCode.ON))

    result = compare_arm_parity(runs)

    assert result.measurement_valid is True
    assert result.mismatched_fields == ()


def test_a_demonstration_run_taints_the_comparison() -> None:
    """measurement_valid false is not evaluation material, however parity looks."""
    result = compare_arm_parity((_base_run(), _base_run(measurement_valid=False)))

    assert result.measurement_valid is False
    assert "measurement_valid" in result.mismatched_fields


def test_no_runs_is_not_a_valid_comparison() -> None:
    result = compare_arm_parity(())

    assert result.measurement_valid is False
    assert result.mismatched_fields == ("runs",)


def _insert_finished_run(
    store: MetadataStore,
    status: Literal["completed", "failed", "cancelled"] = "completed",
    **overrides: object,
) -> RunRecord:
    run = _base_run(**overrides)
    store.create_run(run)
    store.finish_run(run.id, status)
    record = store.get_run(run.id)
    assert record is not None
    return record


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> MetadataStore:
    """The store the test fills, handed back to the script by its own open call.

    The file is still created so the script's own existence check is exercised.
    """
    opened = MetadataStore(tmp_path / "metadata.sqlite3")
    monkeypatch.setattr(check_runs, "open_metadata_store", lambda *_, **__: opened)
    return opened


def _write_protocol(tmp_path: Path, comparisons: list[dict[str, object]]) -> Path:
    protocol_file = tmp_path / "protocol.json"
    protocol_file.write_text(
        json.dumps({"registered_comparisons": comparisons}),
        encoding="utf-8",
    )
    return protocol_file


def test_metadata_store_list_runs_ordered_by_created_at() -> None:
    store = MetadataStore(":memory:")
    run_1 = _base_run()
    run_2 = _base_run()
    store.create_run(run_1)
    store.create_run(run_2)

    runs = store.list_runs()
    assert len(runs) == 2
    assert [r.id for r in runs] == [run_1.id, run_2.id]
    assert runs[0].created_at is not None
    assert runs[1].created_at is not None
    assert runs[0].created_at <= runs[1].created_at


def test_check_runs_clean_matching_pair_passes(
    store: MetadataStore, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    doc_hash = "hash-one"
    _insert_finished_run(store, input_hash=doc_hash, arm=ArmCode.MID)
    _insert_finished_run(store, input_hash=doc_hash, arm=ArmCode.OFF)

    protocol_file = _write_protocol(
        tmp_path,
        [{"id": "configuration_contrast", "pair": ["MID", "OFF"]}],
    )

    exit_code = check_runs.main(
        [
            "--metadata",
            str(tmp_path / "metadata.sqlite3"),
            "--protocol",
            str(protocol_file),
        ]
    )

    assert exit_code == 0
    captured = capsys.readouterr().out
    assert f"configuration_contrast {doc_hash} OK" in captured
    assert "configuration_contrast: 1 pairs checked, 0 failed, 0 skipped" in captured


def test_check_runs_differing_dimension_fails_and_names_dimension(
    store: MetadataStore, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    doc_hash = "hash-one"
    _insert_finished_run(store, input_hash=doc_hash, arm=ArmCode.MID)
    _insert_finished_run(
        store,
        input_hash=doc_hash,
        arm=ArmCode.OFF,
        prompt_bundle_version="different-prompt-bundle",
    )

    protocol_file = _write_protocol(
        tmp_path,
        [{"id": "configuration_contrast", "pair": ["MID", "OFF"]}],
    )

    exit_code = check_runs.main(
        [
            "--metadata",
            str(tmp_path / "metadata.sqlite3"),
            "--protocol",
            str(protocol_file),
        ]
    )

    assert exit_code != 0
    captured = capsys.readouterr().out
    assert (
        f"configuration_contrast {doc_hash} MISMATCH: prompt_bundle_version" in captured
    )
    assert "configuration_contrast: 1 pairs checked, 1 failed, 0 skipped" in captured


def test_check_runs_ambiguous_completed_runs_fail(
    store: MetadataStore, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    doc_hash = "hash-one"
    run_mid_1 = _insert_finished_run(store, input_hash=doc_hash, arm=ArmCode.MID)
    run_mid_2 = _insert_finished_run(store, input_hash=doc_hash, arm=ArmCode.MID)
    _insert_finished_run(store, input_hash=doc_hash, arm=ArmCode.OFF)

    protocol_file = _write_protocol(
        tmp_path,
        [{"id": "configuration_contrast", "pair": ["MID", "OFF"]}],
    )

    exit_code = check_runs.main(
        [
            "--metadata",
            str(tmp_path / "metadata.sqlite3"),
            "--protocol",
            str(protocol_file),
        ]
    )

    assert exit_code != 0
    captured = capsys.readouterr().out
    assert "AMBIGUOUS" in captured
    assert doc_hash in captured
    assert "MID" in captured
    assert str(run_mid_1.id) in captured
    assert str(run_mid_2.id) in captured


def test_check_runs_no_eligible_document_fails_as_nothing_checked(
    store: MetadataStore, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert store.list_runs() == []

    protocol_file = _write_protocol(
        tmp_path,
        [{"id": "configuration_contrast", "pair": ["MID", "OFF"]}],
    )

    exit_code = check_runs.main(
        [
            "--metadata",
            str(tmp_path / "metadata.sqlite3"),
            "--protocol",
            str(protocol_file),
        ]
    )

    assert exit_code != 0
    captured = capsys.readouterr().out
    assert "0 pairs checked" in captured
    assert "configuration_contrast: 0 pairs checked, 0 failed, 0 skipped" in captured


def test_check_runs_one_armed_document_is_skipped_and_counted(
    store: MetadataStore, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    doc_1 = "hash-one"
    doc_2 = "hash-two"

    _insert_finished_run(store, input_hash=doc_1, arm=ArmCode.MID)
    _insert_finished_run(store, input_hash=doc_1, arm=ArmCode.OFF)
    _insert_finished_run(store, input_hash=doc_2, arm=ArmCode.MID)

    protocol_file = _write_protocol(
        tmp_path,
        [{"id": "configuration_contrast", "pair": ["MID", "OFF"]}],
    )

    exit_code = check_runs.main(
        [
            "--metadata",
            str(tmp_path / "metadata.sqlite3"),
            "--protocol",
            str(protocol_file),
        ]
    )

    assert exit_code == 0
    captured = capsys.readouterr().out
    assert f"configuration_contrast {doc_1} OK" in captured
    assert "configuration_contrast: 1 pairs checked, 0 failed, 1 skipped" in captured


def test_check_runs_ignores_non_completed_runs(
    store: MetadataStore, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    doc_1 = "hash-one"
    _insert_finished_run(store, input_hash=doc_1, arm=ArmCode.MID)
    _insert_finished_run(store, input_hash=doc_1, arm=ArmCode.OFF)

    # The second input has a completed MID, but its OFF run failed.
    doc_2 = "hash-two"
    _insert_finished_run(store, input_hash=doc_2, arm=ArmCode.MID)
    _insert_finished_run(store, status="failed", input_hash=doc_2, arm=ArmCode.OFF)

    protocol_file = _write_protocol(
        tmp_path,
        [{"id": "configuration_contrast", "pair": ["MID", "OFF"]}],
    )

    exit_code = check_runs.main(
        [
            "--metadata",
            str(tmp_path / "metadata.sqlite3"),
            "--protocol",
            str(protocol_file),
        ]
    )

    assert exit_code == 0
    captured = capsys.readouterr().out
    assert "configuration_contrast: 1 pairs checked, 0 failed, 1 skipped" in captured


def test_check_runs_ignores_completed_interactive_children(
    store: MetadataStore, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    doc_hash = "hash-with-children"
    mid = _insert_finished_run(store, input_hash=doc_hash, arm=ArmCode.MID)
    off = _insert_finished_run(store, input_hash=doc_hash, arm=ArmCode.OFF)
    for parent in (mid, off):
        _insert_finished_run(
            store,
            input_hash=doc_hash,
            arm=parent.arm,
            measurement_valid=False,
            parent_run_id=parent.id,
            interaction="analyse",
        )
    protocol_file = _write_protocol(
        tmp_path,
        [{"id": "configuration_contrast", "pair": ["MID", "OFF"]}],
    )

    exit_code = check_runs.main(
        [
            "--metadata",
            str(tmp_path / "metadata.sqlite3"),
            "--protocol",
            str(protocol_file),
        ]
    )

    assert exit_code == 0
    assert "configuration_contrast: 1 pairs checked, 0 failed, 0 skipped" in (
        capsys.readouterr().out
    )


def test_check_runs_pairs_arms_run_from_separate_uploads(
    store: MetadataStore, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """One file uploaded twice is still one document for a registered comparison.

    document_id identifies an upload session. Running each arm from its own upload,
    which the interface makes the natural thing to do, gives the same file a different
    document_id per arm; grouping on it would report that nothing was compared while
    the runs sit in the database.
    """
    doc_hash = "hash-one"
    _insert_finished_run(
        store, input_hash=doc_hash, document_id=uuid4(), arm=ArmCode.MID
    )
    _insert_finished_run(
        store, input_hash=doc_hash, document_id=uuid4(), arm=ArmCode.OFF
    )

    protocol_file = _write_protocol(
        tmp_path,
        [{"id": "configuration_contrast", "pair": ["MID", "OFF"]}],
    )

    exit_code = check_runs.main(
        [
            "--metadata",
            str(tmp_path / "metadata.sqlite3"),
            "--protocol",
            str(protocol_file),
        ]
    )

    assert exit_code == 0
    captured = capsys.readouterr().out
    assert f"configuration_contrast {doc_hash} OK" in captured
    assert "configuration_contrast: 1 pairs checked, 0 failed, 0 skipped" in captured


def test_check_runs_reads_a_postgres_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Runs are written to PostgreSQL, so the parity control has to read one.

    The check gated its input on ``Path.is_file`` and constructed the store from that
    path, which for a database URL meant refusing to start, and for a file meant an
    empty in-memory store rather than the runs. Neither can check a real comparison.
    """
    store = MetadataStore(tmp_path / "stand-in.sqlite3")
    doc_hash = "hash-postgres"
    _insert_finished_run(store, input_hash=doc_hash, arm=ArmCode.MID)
    _insert_finished_run(store, input_hash=doc_hash, arm=ArmCode.OFF)
    protocol_file = _write_protocol(
        tmp_path, [{"id": "configuration_contrast", "pair": ["MID", "OFF"]}]
    )

    opened: list[str] = []

    def fake_open(target: str, **kwargs: object) -> MetadataStore:
        opened.append(target)
        return store

    monkeypatch.setattr(check_runs, "open_metadata_store", fake_open)

    url = "postgresql://app:secret@postgre:5432/contract_analyzer"
    exit_code = check_runs.main(["--metadata", url, "--protocol", str(protocol_file)])

    assert exit_code == 0
    # Passed through untouched: no path resolution, no is_file gate.
    assert opened == [url]
    assert "configuration_contrast: 1 pairs checked" in capsys.readouterr().out


def test_check_runs_defaults_to_the_deployment_database_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = MetadataStore(tmp_path / "stand-in-default.sqlite3")
    doc_hash = "hash-default"
    _insert_finished_run(store, input_hash=doc_hash, arm=ArmCode.MID)
    _insert_finished_run(store, input_hash=doc_hash, arm=ArmCode.OFF)
    protocol_file = _write_protocol(
        tmp_path, [{"id": "configuration_contrast", "pair": ["MID", "OFF"]}]
    )

    opened: list[str] = []
    monkeypatch.setenv(
        "DATABASE_URL", "postgresql://app@postgre:5432/contract_analyzer"
    )
    monkeypatch.setattr(
        check_runs,
        "open_metadata_store",
        lambda target, **kwargs: (opened.append(target), store)[1],
    )

    assert check_runs.main(["--protocol", str(protocol_file)]) == 0
    assert opened == ["postgresql://app@postgre:5432/contract_analyzer"]


def _write_batch(
    directory: Path, runs: dict[str, RunRecord], artifact_id: str = "HD-TEST"
) -> Path:
    """The result files a batch leaves behind, holding the ids of its runs."""
    for arm, run in runs.items():
        path = directory / artifact_id / f"{arm}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"id": str(run.id), "arm": arm}), encoding="utf-8")
    return directory


def test_repeated_series_are_ambiguous_together_and_checkable_apart(
    store: MetadataStore, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A second series is not a second run of the first.

    Read as one batch, three series give three completed runs per document and
    arm, and every comparison becomes ambiguous. The result files say which runs
    belong together, so naming them separates the series instead of forcing a
    choice between them.
    """
    doc_hash = "hash-one"
    first = {
        arm.value: _insert_finished_run(store, input_hash=doc_hash, arm=arm)
        for arm in (ArmCode.MID, ArmCode.OFF)
    }
    second = {
        arm.value: _insert_finished_run(store, input_hash=doc_hash, arm=arm)
        for arm in (ArmCode.MID, ArmCode.OFF)
    }
    protocol_file = _write_protocol(
        tmp_path, [{"id": "configuration_contrast", "pair": ["MID", "OFF"]}]
    )
    argv = [
        "--metadata",
        str(tmp_path / "metadata.sqlite3"),
        "--protocol",
        str(protocol_file),
    ]

    assert check_runs.main(argv) != 0
    assert "AMBIGUOUS" in capsys.readouterr().out

    _write_batch(tmp_path / "series-1", first)
    _write_batch(tmp_path / "series-2", second)
    exit_code = check_runs.main(
        argv
        + ["--batch", str(tmp_path / "series-1"), "--batch", str(tmp_path / "series-2")]
    )

    captured = capsys.readouterr().out
    assert exit_code == 0, captured
    assert "AMBIGUOUS" not in captured
    assert "series-1 configuration_contrast: 1 pairs checked" in captured
    assert "series-2 configuration_contrast: 1 pairs checked" in captured
    assert (
        "series-1 vs series-2: 2 cells checked, 0 failed, 0 uncovered, 0 unmatched"
        in captured
    )


def test_a_repeat_series_that_did_not_repeat_the_configuration_is_named(
    store: MetadataStore, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Within-batch parity cannot see a series that moved as a whole.

    A second series run under another prompt bundle is internally consistent:
    its arms agree with each other on all ten dimensions and its own check is
    green. What it is not is a repeat of the first series, and the comparison
    across batches is the only thing that looks.
    """
    doc_hash = "hash-one"
    first = {
        arm.value: _insert_finished_run(store, input_hash=doc_hash, arm=arm)
        for arm in (ArmCode.MID, ArmCode.OFF)
    }
    second = {
        arm.value: _insert_finished_run(
            store,
            input_hash=doc_hash,
            arm=arm,
            prompt_bundle_version="prompt-bundle-v2",
        )
        for arm in (ArmCode.MID, ArmCode.OFF)
    }
    _write_batch(tmp_path / "series-1", first)
    _write_batch(tmp_path / "series-2", second)
    protocol_file = _write_protocol(
        tmp_path, [{"id": "configuration_contrast", "pair": ["MID", "OFF"]}]
    )

    exit_code = check_runs.main(
        [
            "--metadata",
            str(tmp_path / "metadata.sqlite3"),
            "--protocol",
            str(protocol_file),
            "--batch",
            str(tmp_path / "series-1"),
            "--batch",
            str(tmp_path / "series-2"),
        ]
    )

    captured = capsys.readouterr().out
    assert exit_code != 0
    assert "series-2 configuration_contrast: 1 pairs checked, 0 failed" in captured
    assert "series-1 vs series-2 hash-one MID MISMATCH: prompt_bundle_version" in (
        captured
    )
    assert "series-1 vs series-2: 2 cells checked, 2 failed" in captured


def test_a_batch_whose_runs_the_store_does_not_hold_fails(
    store: MetadataStore, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A result file naming a run the store never finished is a broken record."""
    doc_hash = "hash-one"
    first = {
        arm.value: _insert_finished_run(store, input_hash=doc_hash, arm=arm)
        for arm in (ArmCode.MID, ArmCode.OFF)
    }
    _write_batch(tmp_path / "series-1", first)
    (tmp_path / "series-1" / "HD-TEST" / "on.json").write_text(
        json.dumps({"id": "11111111-2222-3333-4444-555555555555"}), encoding="utf-8"
    )
    protocol_file = _write_protocol(
        tmp_path, [{"id": "configuration_contrast", "pair": ["MID", "OFF"]}]
    )

    exit_code = check_runs.main(
        [
            "--metadata",
            str(tmp_path / "metadata.sqlite3"),
            "--protocol",
            str(protocol_file),
            "--batch",
            str(tmp_path / "series-1"),
        ]
    )

    captured = capsys.readouterr().out
    assert exit_code != 0
    assert "MISSING: series-1 records 1 run(s)" in captured


def test_an_unreadable_result_file_fails_instead_of_thinning_the_batch(
    store: MetadataStore, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A half-parseable batch must not come back quieter than a whole one."""
    doc_hash = "hash-one"
    first = {
        arm.value: _insert_finished_run(store, input_hash=doc_hash, arm=arm)
        for arm in (ArmCode.MID, ArmCode.OFF)
    }
    _write_batch(tmp_path / "series-1", first)
    (tmp_path / "series-1" / "HD-TEST" / "on.json").write_text("{", encoding="utf-8")
    (tmp_path / "series-1" / "HD-TEST" / "off.json").write_text(
        json.dumps({"arm": "off"}), encoding="utf-8"
    )
    protocol_file = _write_protocol(
        tmp_path, [{"id": "configuration_contrast", "pair": ["MID", "OFF"]}]
    )

    exit_code = check_runs.main(
        [
            "--metadata",
            str(tmp_path / "metadata.sqlite3"),
            "--protocol",
            str(protocol_file),
            "--batch",
            str(tmp_path / "series-1"),
        ]
    )

    captured = capsys.readouterr().out
    assert exit_code != 0
    assert captured.count("UNREADABLE") == 2
    assert "no run id" in captured


def test_a_repeat_missing_a_cell_of_the_series_it_repeats_fails(
    store: MetadataStore, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Comparing the one cell two series happen to share proves nothing.

    A repeat is a repeat cell for cell. Passing on a single pair while eight
    cells have no counterpart is the reading this mode exists to refuse.
    """
    first = {
        arm.value: _insert_finished_run(store, input_hash="hash-one", arm=arm)
        for arm in (ArmCode.MID, ArmCode.OFF)
    }
    second = {
        arm.value: _insert_finished_run(store, input_hash="hash-two", arm=arm)
        for arm in (ArmCode.MID, ArmCode.OFF)
    }
    _write_batch(tmp_path / "series-1", first)
    _write_batch(tmp_path / "series-2", second)
    protocol_file = _write_protocol(
        tmp_path, [{"id": "configuration_contrast", "pair": ["MID", "OFF"]}]
    )

    exit_code = check_runs.main(
        [
            "--metadata",
            str(tmp_path / "metadata.sqlite3"),
            "--protocol",
            str(protocol_file),
            "--batch",
            str(tmp_path / "series-1"),
            "--batch",
            str(tmp_path / "series-2"),
        ]
    )

    captured = capsys.readouterr().out
    assert exit_code != 0
    assert (
        "series-1 vs series-2: 0 cells checked, 0 failed, 2 uncovered, 2 unmatched"
        in captured
    )
    assert "2 cell(s) of series-1 have no counterpart in series-2" in captured
    assert "2 cell(s) of series-2 have no counterpart in series-1" in captured


def test_a_repeat_covering_only_part_of_the_baseline_fails(
    store: MetadataStore, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Coverage has to be symmetric, or a third of a series passes as a repeat.

    Walking the repeat's cells and asking whether the baseline has each one
    answers only half the question. A repeat that ran three of nine cells has a
    counterpart for every cell it holds, skips nothing, and is reported as a
    clean comparison of three cells.
    """
    baseline = {}
    for document in ("hash-one", "hash-two", "hash-three"):
        for arm in (ArmCode.OFF, ArmCode.MID, ArmCode.ON):
            baseline[(document, arm)] = _insert_finished_run(
                store, input_hash=document, arm=arm
            )
    partial = {
        arm: _insert_finished_run(store, input_hash="hash-one", arm=arm)
        for arm in (ArmCode.OFF, ArmCode.MID, ArmCode.ON)
    }
    for (document, arm), run in baseline.items():
        path = tmp_path / "series-1" / document / f"{arm.value}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"id": str(run.id)}), encoding="utf-8")
    _write_batch(
        tmp_path / "series-2", {arm.value: run for arm, run in partial.items()}
    )
    protocol_file = _write_protocol(
        tmp_path, [{"id": "configuration_contrast", "pair": ["MID", "OFF"]}]
    )

    exit_code = check_runs.main(
        [
            "--metadata",
            str(tmp_path / "metadata.sqlite3"),
            "--protocol",
            str(protocol_file),
            "--batch",
            str(tmp_path / "series-1"),
            "--batch",
            str(tmp_path / "series-2"),
        ]
    )

    captured = capsys.readouterr().out
    assert exit_code != 0, captured
    assert "6 cell(s) of series-1 have no counterpart in series-2" in captured


def test_a_batch_compared_with_itself_is_not_a_repetition(
    store: MetadataStore, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Comparing a run with itself agrees on all ten dimensions and proves nothing."""
    runs = {
        arm.value: _insert_finished_run(store, input_hash="hash-one", arm=arm)
        for arm in (ArmCode.MID, ArmCode.OFF)
    }
    _write_batch(tmp_path / "series-1", runs)
    _write_batch(tmp_path / "copy-of-series-1", runs)
    protocol_file = _write_protocol(
        tmp_path, [{"id": "configuration_contrast", "pair": ["MID", "OFF"]}]
    )

    exit_code = check_runs.main(
        [
            "--metadata",
            str(tmp_path / "metadata.sqlite3"),
            "--protocol",
            str(protocol_file),
            "--batch",
            str(tmp_path / "series-1"),
            "--batch",
            str(tmp_path / "copy-of-series-1"),
        ]
    )

    captured = capsys.readouterr().out
    assert exit_code != 0, captured
    assert "the same run" in captured
