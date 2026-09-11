"""Tests for scripts/summarize_results.py using a tmp_path result tree."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import summarize_results  # noqa: E402

# ── helpers ────────────────────────────────────────────────────────────────────

_ARM_RESULTS: list[dict[str, Any]] = [
    {
        "artifact_id": "ED-001",
        "arm": "off",
        "status": "completed",
        "elapsed_ms": 3000.0,
        "input_tokens": 100,
        "output_tokens": 50,
        "units_total": 5,
        "units_with_finding": 4,
        "units_not_processed": 1,
        "context_edge_count": 0,
        "finding_histogram": {"consistent": 2, "no_basis_found": 2},
        "error_code": None,
        "attempts_count": 1,
        "attempts_statuses": ["success"],
        "monetary_cost_microunits": None,
        "cost_unknown_reason": "no_price_table",
        "model_calls": 1,
        "finder_tool_turns": 3,
        "finder_search_calls": 5,
        "finder_budget_exhausted_units": 0,
    },
    {
        "artifact_id": "ED-001",
        "arm": "mid",
        "status": "completed",
        "elapsed_ms": 5000.0,
        "input_tokens": 200,
        "output_tokens": 80,
        "units_total": 5,
        "units_with_finding": 5,
        "units_not_processed": 0,
        "context_edge_count": 0,
        "finding_histogram": {"consistent": 3, "contradictory": 1, "no_basis_found": 1},
        "error_code": None,
        "attempts_count": 1,
        "attempts_statuses": ["success"],
        "monetary_cost_microunits": None,
        "cost_unknown_reason": "no_price_table",
        "model_calls": 1,
        "finder_tool_turns": None,
        "finder_search_calls": None,
        "finder_budget_exhausted_units": None,
    },
    {
        "artifact_id": "RT-001",
        "arm": "off",
        "status": "failed",
        "elapsed_ms": 1000.0,
        "input_tokens": 50,
        "output_tokens": 10,
        "units_total": 3,
        "units_with_finding": 0,
        "units_not_processed": 3,
        "context_edge_count": None,
        "finding_histogram": {},
        "error_code": "budget_exhausted",
        "attempts_count": 2,
        "attempts_statuses": ["transport_error", "success"],
        "monetary_cost_microunits": 1234,
        "cost_unknown_reason": None,
        "model_calls": 2,
        "finder_tool_turns": 1,
        "finder_search_calls": 2,
        "finder_budget_exhausted_units": 1,
    },
]

_STRATUM_MAP = {"ED-001": "synthetic_fixture", "RT-001": "real_source"}


@pytest.fixture
def result_dir(tmp_path: Path) -> Path:
    """Build a result directory with a summary.json."""
    rd = tmp_path / "results" / "development" / "2026-09-03-test"
    rd.mkdir(parents=True)
    summary = {
        "config": {"model_request_id": "test-model"},
        "split": "development",
        "started_at": "2026-09-03T10:00:00+00:00",
        "finished_at": "2026-09-03T10:30:00+00:00",
        "git_head": "abc123",
        "arm_results": _ARM_RESULTS,
    }
    (rd / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return rd


@pytest.fixture
def result_dir_no_summary(tmp_path: Path) -> Path:
    """Build a result directory using per-arm JSON files (no summary.json)."""
    rd = tmp_path / "results" / "development" / "2026-09-03-raw"
    rd.mkdir(parents=True)
    for rec in _ARM_RESULTS:
        artifact_id: str = rec["artifact_id"]
        arm: str = rec["arm"]
        artifact_dir = rd / artifact_id
        artifact_dir.mkdir(exist_ok=True)
        # Write in Run format (as GET /runs/{id} returns)
        run_data = {
            "id": "00000000-0000-0000-0000-000000000001",
            "document_id": "00000000-0000-0000-0000-000000000002",
            "arm": arm,
            "status": rec["status"],
            "measurement_valid": True,
            "content_available": True,
            "findings": [
                {
                    "id": "f1",
                    "unit_id": "u1",
                    "code": code,
                    "prominence": "neutral",
                    "anchor_resolved": False,
                }
                for code, cnt in rec.get("finding_histogram", {}).items()
                for _ in range(cnt)
            ],
            "metrics": {
                "input_tokens": rec["input_tokens"],
                "output_tokens": rec["output_tokens"],
                "elapsed_ms": rec["elapsed_ms"],
                "units_total": rec["units_total"],
                "units_with_finding": rec["units_with_finding"],
                "units_not_processed": rec["units_not_processed"],
                "context_edge_count": rec["context_edge_count"],
                "cost": {
                    "monetary_cost_microunits": rec["monetary_cost_microunits"],
                    "unknown_reason": rec["cost_unknown_reason"],
                },
                "attempts": [],
            },
        }
        (artifact_dir / f"{arm}.json").write_text(
            json.dumps(run_data), encoding="utf-8"
        )
    return rd


# ── tests ──────────────────────────────────────────────────────────────────────


def _run_summarize(dirs: list[Path]) -> str:
    buf = io.StringIO()
    with patch.object(
        summarize_results, "_load_stratum_map", return_value=_STRATUM_MAP
    ):
        rc = summarize_results.main(
            [str(d) for d in dirs],
            output_stream=buf,
        )
    assert rc == 0
    return buf.getvalue()


def test_markdown_output_has_headings(result_dir: Path) -> None:
    output = _run_summarize([result_dir])
    assert "# Evaluation summary" in output
    assert "## Per-arm / per-stratum aggregates" in output
    assert "## Per-artifact detail" in output


def test_arms_appear_in_output(result_dir: Path) -> None:
    output = _run_summarize([result_dir])
    assert "`off`" in output
    assert "`mid`" in output


def test_strata_appear_in_output(result_dir: Path) -> None:
    output = _run_summarize([result_dir])
    assert "synthetic_fixture" in output
    assert "real_source" in output


def test_cost_unknown_text_when_no_price_table(result_dir: Path) -> None:
    output = _run_summarize([result_dir])
    # The ED-001 arms have no price table; the RT-001/off arm has one
    assert "unknown, because price table unavailable" in output


def test_finding_histogram_in_output(result_dir: Path) -> None:
    output = _run_summarize([result_dir])
    assert "consistent" in output
    assert "contradictory" in output


def test_detail_table_has_artifact_ids(result_dir: Path) -> None:
    output = _run_summarize([result_dir])
    assert "ED-001" in output
    assert "RT-001" in output


def test_no_document_text_in_output(result_dir: Path) -> None:
    """The output must not contain document text."""
    output = _run_summarize([result_dir])
    # The fake document text is never in the arm records, but assert defensively
    assert "Umowa" not in output
    assert "lokalu" not in output


def test_scanner_fallback_no_summary_json(result_dir_no_summary: Path) -> None:
    """When no summary.json exists, arm files are scanned and normalized."""
    output = _run_summarize([result_dir_no_summary])
    assert "# Evaluation summary" in output
    assert "ED-001" in output or "RT-001" in output
    # Verify that metrics and histograms were properly extracted from Run metrics
    assert "Elapsed s: mean 3.00" in output
    assert "consistent=2" in output


def test_invalid_measurement_separated(tmp_path: Path) -> None:
    """Runs with measurement_valid=False are flagged and separated from aggregates."""
    rd = tmp_path / "results" / "development" / "invalid-test"
    rd.mkdir(parents=True)
    inv_rec = dict(_ARM_RESULTS[0])
    inv_rec["measurement_valid"] = False
    summary = {
        "split": "development",
        "arm_results": [inv_rec],
    }
    (rd / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    output = _run_summarize([rd])
    assert "invalid: 1" in output
    assert "(invalid)" in output
    # Quantitative aggregates are not populated from invalid run
    assert "Elapsed s: mean —" in output


def test_multiple_dirs_combined(result_dir: Path, tmp_path: Path) -> None:
    """Two result directories are combined into one report."""
    rd2 = tmp_path / "results" / "development" / "2026-09-03-run2"
    rd2.mkdir(parents=True)
    summary2 = {
        "split": "development",
        "arm_results": [_ARM_RESULTS[0]],  # just one record
    }
    (rd2 / "summary.json").write_text(json.dumps(summary2), encoding="utf-8")

    output = _run_summarize([result_dir, rd2])
    assert "Total arm records: 4" in output


def test_missing_dir_returns_error(tmp_path: Path) -> None:
    buf = io.StringIO()
    rc = summarize_results.main(
        [str(tmp_path / "nonexistent")],
        output_stream=buf,
    )
    assert rc != 0


def test_empty_dir_prints_no_results(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    buf = io.StringIO()
    with patch.object(summarize_results, "_load_stratum_map", return_value={}):
        rc = summarize_results.main([str(empty)], output_stream=buf)
    assert rc == 0
    assert "No results found" in buf.getvalue()


def test_finder_counters_when_present(result_dir: Path) -> None:
    output = _run_summarize([result_dir])
    # ED-001/off has finder_tool_turns=3, finder_search_calls=5
    assert "Finder tool turns" in output
    assert "Finder search calls" in output
