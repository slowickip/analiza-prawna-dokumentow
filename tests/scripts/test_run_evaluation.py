"""Tests for scripts/run_evaluation.py using httpx.MockTransport (no network)."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import run_evaluation  # noqa: E402  (script-as-module import)

_DOC_TEXT = b"Umowa najmu lokalu mieszkalnego."
_DOC_SHA256 = hashlib.sha256(_DOC_TEXT).hexdigest()

_DOC_ID = "aaaaaaaa-0000-0000-0000-000000000001"
_RUN_ID_OFF = "bbbbbbbb-0000-0000-0000-000000000001"
_RUN_ID_MID = "bbbbbbbb-0000-0000-0000-000000000002"
_RUN_ID_ON = "bbbbbbbb-0000-0000-0000-000000000003"

_ARM_TO_RUN_ID = {"off": _RUN_ID_OFF, "mid": _RUN_ID_MID, "on": _RUN_ID_ON}

_CONFIG_RESP: dict[str, Any] = {
    "model_request_id": "test-model",
    "model_endpoint": "test.endpoint",
    "prompt_bundle_version": "1.0.0",
    "corpus_snapshot": "snapshot-1",
    "embedding_model": "test-embedding-model",
    "embedding_space_fingerprint": "aaaa1111bbbb2222",
    "corpus_source_format": "as_declared",
    "tool_bundle_version": "worksheet-roles-test",
    "arms": ["off", "mid", "on"],
    "measured_mode": True,
    "evaluation_batch_open": True,
    "limits": {
        "max_input_bytes": 10_000_000,
        "max_pdf_pages": 100,
        "content_ttl_seconds": 3600,
    },
}


def _make_completed_run(
    run_id: str,
    arm: str,
    document_id: str = _DOC_ID,
    measurement_valid: bool = True,
) -> dict[str, Any]:
    return {
        "id": run_id,
        "document_id": document_id,
        "arm": arm,
        "status": "completed",
        "measurement_valid": measurement_valid,
        "content_available": True,
        "findings": [
            {
                "id": "f1",
                "unit_id": "u1",
                "code": "consistent",
                "prominence": "neutral",
                "anchor_resolved": False,
            }
        ],
        "metrics": {
            "input_tokens": 100,
            "output_tokens": 50,
            "elapsed_ms": 3000.0,
            "units_total": 5,
            "units_with_finding": 4,
            "units_not_processed": 1,
            "context_edge_count": 2,
            "finder_tool_turns": 2,
            "finder_search_calls": 3,
            "finder_budget_exhausted_units": 0,
            "cost": {
                "monetary_cost_microunits": None,
                "unknown_reason": "no_price_table",
            },
            "attempts": [
                {
                    "id": "cccccccc-0000-0000-0000-000000000001",
                    "requested_model": "test-model",
                    "prompt_version": "1",
                    "temperature": 0.0,
                    "parameters": {},
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "latency_ms": 2000.0,
                    "status": "success",
                    "retry_number": 0,
                    "prompt_hash": "abc123",
                }
            ],
        },
    }


class _FakeTransport(httpx.MockTransport):
    """Fake API exercising 410-retry on POST /runs and 409-proceed paths."""

    def __init__(self) -> None:
        # Lets a test serve a second server whose parity facts differ.
        self.config_overrides: dict[str, Any] = {}
        self._upload_count = 0
        self._run_polls: dict[str, int] = {}
        self._content_expired_arms: set[str] = {
            "mid"
        }  # trigger 410 once on POST /runs for mid
        self._active_run_strikes: int = 1  # one 409 before on arm
        self._uploaded_doc_ids: list[str] = []
        self._runs_created: list[dict[str, Any]] = []

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path

        if path.endswith("/config"):
            return httpx.Response(200, json={**_CONFIG_RESP, **self.config_overrides})

        if path.endswith("/documents") and request.method == "POST":
            self._upload_count += 1
            doc_id = f"aaaaaaaa-0000-0000-0000-{self._upload_count:012d}"
            self._uploaded_doc_ids.append(doc_id)
            return httpx.Response(
                201,
                json={"id": doc_id, "content_hash": "abc", "unit_count": 5},
            )

        if (
            "/runs" in path
            and request.method == "POST"
            and not path.endswith("/cancel")
        ):
            body = json.loads(request.content)
            arm: str = body["arm"]
            doc_id: str = body["document_id"]
            if arm in self._content_expired_arms:
                self._content_expired_arms.discard(arm)
                return httpx.Response(
                    410,
                    json={"code": "content_expired", "message_pl": "wygasła"},
                )
            if arm == "on" and self._active_run_strikes > 0:
                self._active_run_strikes -= 1
                return httpx.Response(
                    409,
                    json={"code": "run_already_active", "message_pl": "inna analiza"},
                )
            run_id = _ARM_TO_RUN_ID.get(arm, _RUN_ID_OFF)
            self._runs_created.append(
                {"arm": arm, "run_id": run_id, "document_id": doc_id}
            )
            running = _make_completed_run(run_id, arm, document_id=doc_id)
            running["status"] = "running"
            running["metrics"] = None
            return httpx.Response(202, json=running)

        for arm, run_id in _ARM_TO_RUN_ID.items():
            if path.endswith(f"/runs/{run_id}"):
                poll_count = self._run_polls.get(run_id, 0)
                self._run_polls[run_id] = poll_count + 1
                return httpx.Response(200, json=_make_completed_run(run_id, arm))

        if "/runs/" in path and request.method == "GET":
            run_id = path.split("/")[-1]
            return httpx.Response(200, json=_make_completed_run(run_id, "off"))

        return httpx.Response(404, json={"code": "not_found", "message_pl": "brak"})


@pytest.fixture
def fake_doc_file(tmp_path: Path) -> Path:
    """Write a fake evaluation document where manifest expects it."""
    doc_path = tmp_path / "evaluation-data" / "documents" / "ED-TEST-lease.txt"
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_bytes(_DOC_TEXT)
    return doc_path


@pytest.fixture
def fake_manifest(tmp_path: Path, fake_doc_file: Path) -> Path:
    """Write a minimal manifest that points at the fake document."""
    manifest = {
        "schema_version": 2,
        "dataset_id": "test",
        "dataset_version": "0.4.0",
        "documents": [
            {
                "id": "ED-TEST",
                "logical_case": "ED-TEST",
                "stratum": "synthetic_fixture",
                "canonical": True,
                "split": "development",
                "path": "documents/ED-TEST-lease.txt",
                "sha256": _DOC_SHA256,
                "format": "txt",
            },
            {
                "id": "HD-TEST",
                "logical_case": "HD-TEST",
                "stratum": "synthetic_fixture",
                "canonical": True,
                "split": "holdout",
                "path": "documents/ED-TEST-lease.txt",
                "sha256": _DOC_SHA256,
                "format": "txt",
            },
        ],
    }
    m = tmp_path / "evaluation-data" / "manifest.json"
    m.parent.mkdir(parents=True, exist_ok=True)
    m.write_text(json.dumps(manifest), encoding="utf-8")
    return m


@pytest.fixture
def fake_protocol(tmp_path: Path) -> Path:
    proto = {
        "state": "not_frozen",
        "protocol_version": "2.3.0",
        "final_or_holdout_data_seen": False,
        "holdout_discipline": {
            "forbidden_until_final_run": ["Do not open holdout before final run."]
        },
        "decision_rule": {
            "thresholds": {
                "tau_RelComp": {"value": None, "owed": True},
                "delta_DetPrec": {"value": None, "owed": True},
                "tau_CiteCorr": {"value": None, "owed": True},
            }
        },
    }
    p = tmp_path / "evaluation-data" / "protocol.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(proto), encoding="utf-8")
    return p


def test_holdout_refused_without_flag(tmp_path: Path, fake_protocol: Path) -> None:
    """--split holdout must fail without --i-confirm-holdout."""
    with patch.object(run_evaluation, "PROTOCOL", fake_protocol):
        rc = run_evaluation.main(
            ["--split", "holdout", "--out", str(tmp_path / "out")],
            transport=httpx.MockTransport(lambda r: httpx.Response(200, json={})),
        )
    assert rc == 2


def test_holdout_refused_when_protocol_not_frozen(
    tmp_path: Path, fake_manifest: Path, fake_protocol: Path
) -> None:
    """Holdout refused even with confirmation flag if protocol is not frozen."""
    with (
        patch.object(run_evaluation, "MANIFEST", fake_manifest),
        patch.object(run_evaluation, "PROTOCOL", fake_protocol),
        patch.object(run_evaluation, "ROOT", fake_manifest.parent.parent),
    ):
        rc = run_evaluation.main(
            [
                "--split",
                "holdout",
                "--i-confirm-holdout",
                "--out",
                str(tmp_path / "out"),
            ],
            transport=httpx.MockTransport(lambda r: httpx.Response(200, json={})),
        )
    assert rc == 2


def test_holdout_refused_when_thresholds_null(
    tmp_path: Path, fake_manifest: Path
) -> None:
    """Holdout must be refused if owed thresholds are not fixed."""
    proto = {
        "state": "frozen",
        "protocol_version": "2.3.0",
        "final_or_holdout_data_seen": False,
        "holdout_discipline": {"forbidden_until_final_run": []},
        "decision_rule": {
            "thresholds": {
                "tau_RelComp": {"value": None, "owed": True},
            }
        },
    }
    p = tmp_path / "protocol.json"
    p.write_text(json.dumps(proto), encoding="utf-8")
    with (
        patch.object(run_evaluation, "MANIFEST", fake_manifest),
        patch.object(run_evaluation, "PROTOCOL", p),
        patch.object(run_evaluation, "ROOT", fake_manifest.parent.parent),
    ):
        rc = run_evaluation.main(
            [
                "--split",
                "holdout",
                "--i-confirm-holdout",
                "--out",
                str(tmp_path / "out"),
            ],
            transport=httpx.MockTransport(lambda r: httpx.Response(200, json={})),
        )
    assert rc == 2


def test_holdout_still_runnable_after_an_exposure_that_was_only_recorded(
    tmp_path: Path, fake_manifest: Path
) -> None:
    """An accidental exposure is recorded, not a reason to abandon the experiment."""
    proto = {
        "state": "frozen",
        "protocol_version": "2.5.7",
        "final_or_holdout_data_seen": True,
        "final_registered_measurement_state": "not_started",
        "holdout_discipline": {"forbidden_until_final_run": []},
        "decision_rule": {"thresholds": {"tau_RelComp": {"value": 0.05, "owed": True}}},
        "terminal_failure_policy": {"owed": False, "value": "fixed"},
    }
    p = tmp_path / "protocol.json"
    p.write_text(json.dumps(proto), encoding="utf-8")
    with (
        patch.object(run_evaluation, "MANIFEST", fake_manifest),
        patch.object(run_evaluation, "PROTOCOL", p),
        patch.object(run_evaluation, "ROOT", fake_manifest.parent.parent),
    ):
        rc = run_evaluation.main(
            [
                "--split",
                "holdout",
                "--i-confirm-holdout",
                "--out",
                str(tmp_path / "out"),
            ],
            transport=httpx.MockTransport(lambda r: httpx.Response(200, json={})),
        )
    # It may still fail for other reasons in this stub environment; what it must
    # not do is refuse on the exposure flag alone.
    assert rc != 2, "a recorded exposure must not forbid the final measurement"


def test_holdout_refused_once_the_registered_batch_has_run(
    tmp_path: Path, fake_manifest: Path
) -> None:
    """Holdout is refused after the registered batch, not after any exposure.

    These are different facts. Whether holdout material has ever been seen is a
    permanent audit record; gating on it would forbid the final measurement for
    good, which is what happened when ED-001's exposure set that flag. What must
    not be repeated is the registered batch itself.
    """
    proto = {
        "state": "frozen",
        "protocol_version": "2.3.0",
        "final_or_holdout_data_seen": True,
        "final_registered_measurement_state": "completed",
        "holdout_discipline": {"forbidden_until_final_run": []},
        "decision_rule": {
            "thresholds": {
                "tau_RelComp": {"value": 0.05, "owed": True},
            }
        },
    }
    p = tmp_path / "protocol.json"
    p.write_text(json.dumps(proto), encoding="utf-8")
    with (
        patch.object(run_evaluation, "MANIFEST", fake_manifest),
        patch.object(run_evaluation, "PROTOCOL", p),
        patch.object(run_evaluation, "ROOT", fake_manifest.parent.parent),
    ):
        rc = run_evaluation.main(
            [
                "--split",
                "holdout",
                "--i-confirm-holdout",
                "--out",
                str(tmp_path / "out"),
            ],
            transport=httpx.MockTransport(lambda r: httpx.Response(200, json={})),
        )
    assert rc == 2


def test_development_run_end_to_end(
    tmp_path: Path, fake_manifest: Path, fake_protocol: Path
) -> None:
    """Happy path: 3 arms, 1 artifact, 410 on POST /runs, 409 on on, complete."""
    out_dir = tmp_path / "out"
    transport = _FakeTransport()

    with (
        patch.object(run_evaluation, "MANIFEST", fake_manifest),
        patch.object(run_evaluation, "PROTOCOL", fake_protocol),
        patch.object(run_evaluation, "ROOT", fake_manifest.parent.parent),
        patch("time.sleep"),
    ):
        rc = run_evaluation.main(
            [
                "--split",
                "development",
                "--arms",
                "off,mid,on",
                "--out",
                str(out_dir),
                "--poll-seconds",
                "0.01",
                "--label",
                "test-batch-1",
            ],
            transport=transport,
        )

    assert rc == 0, "all runs completed so exit code should be 0"

    assert transport._upload_count == 2
    doc_file = out_dir / "ED-TEST" / "document.json"
    assert doc_file.exists()
    doc_saved = json.loads(doc_file.read_text(encoding="utf-8"))
    assert doc_saved["id"] == transport._uploaded_doc_ids[1]

    on_run_req = [r for r in transport._runs_created if r["arm"] == "on"][0]
    assert on_run_req["document_id"] == transport._uploaded_doc_ids[1]

    for arm in ("off", "mid", "on"):
        arm_file = out_dir / "ED-TEST" / f"{arm}.json"
        assert arm_file.exists(), f"missing {arm}.json"
        data = json.loads(arm_file.read_text(encoding="utf-8"))
        assert data["status"] == "completed"
        assert data["measurement_valid"] is True
        assert not (out_dir / "ED-TEST" / f"{arm}.pending.json").exists()

    summary_file = out_dir / "summary.json"
    assert summary_file.exists()
    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    assert summary["split"] == "development"
    assert summary["label"] == "test-batch-1"
    assert summary["dataset_version"] == "0.4.0"
    assert summary["protocol_version"] == "2.3.0"
    assert len(summary["arm_results"]) == 3
    assert summary["arm_results"][0]["finder_tool_turns"] == 2
    assert summary["arm_results"][0]["finder_search_calls"] == 3


def test_resume_skips_existing_results(
    tmp_path: Path, fake_manifest: Path, fake_protocol: Path
) -> None:
    """Arms whose result file already exists are skipped."""
    out_dir = tmp_path / "out"
    artifact_dir = out_dir / "ED-TEST"
    artifact_dir.mkdir(parents=True)
    doc_resp = {"id": _DOC_ID, "content_hash": "abc", "unit_count": 5}
    (artifact_dir / "document.json").write_text(json.dumps(doc_resp), encoding="utf-8")
    for arm in ("off", "mid"):
        completed = _make_completed_run(_ARM_TO_RUN_ID[arm], arm)
        (artifact_dir / f"{arm}.json").write_text(
            json.dumps(completed), encoding="utf-8"
        )

    transport = _FakeTransport()
    with (
        patch.object(run_evaluation, "MANIFEST", fake_manifest),
        patch.object(run_evaluation, "PROTOCOL", fake_protocol),
        patch.object(run_evaluation, "ROOT", fake_manifest.parent.parent),
        patch("time.sleep"),
    ):
        rc = run_evaluation.main(
            ["--split", "development", "--arms", "off,mid,on", "--out", str(out_dir)],
            transport=transport,
        )

    assert rc == 0
    assert (out_dir / "ED-TEST" / "on.json").exists()


def test_resume_pending_run_without_recreation(
    tmp_path: Path, fake_manifest: Path, fake_protocol: Path
) -> None:
    """Resume polling an existing pending file directly without recreating."""
    out_dir = tmp_path / "out"
    artifact_dir = out_dir / "ED-TEST"
    artifact_dir.mkdir(parents=True)
    doc_resp = {"id": _DOC_ID, "content_hash": "abc", "unit_count": 5}
    (artifact_dir / "document.json").write_text(json.dumps(doc_resp), encoding="utf-8")

    pending_run_id = "pending-run-999"
    (artifact_dir / "off.pending.json").write_text(
        json.dumps(
            {
                "run_id": pending_run_id,
                "arm": "off",
                "artifact_id": "ED-TEST",
            }
        ),
        encoding="utf-8",
    )

    transport = _FakeTransport()
    with (
        patch.object(run_evaluation, "MANIFEST", fake_manifest),
        patch.object(run_evaluation, "PROTOCOL", fake_protocol),
        patch.object(run_evaluation, "ROOT", fake_manifest.parent.parent),
        patch("time.sleep"),
    ):
        rc = run_evaluation.main(
            ["--split", "development", "--arms", "off", "--out", str(out_dir)],
            transport=transport,
        )

    assert rc == 0
    assert len(transport._runs_created) == 0
    assert (artifact_dir / "off.json").exists()
    assert not (artifact_dir / "off.pending.json").exists()
    data = json.loads((artifact_dir / "off.json").read_text(encoding="utf-8"))
    assert data["id"] == pending_run_id


def test_an_unsealed_server_fails(
    tmp_path: Path, fake_manifest: Path, fake_protocol: Path, capsys
) -> None:
    """A batch against a server that still admits asks is refused, not run.

    Sealing is what keeps an interactive action from starting beside a measured
    run and spending the wall clock the batch reports.
    """
    config_unmeasured = dict(_CONFIG_RESP)
    config_unmeasured["evaluation_batch_open"] = False
    requested: list[str] = []

    def _handler(r: httpx.Request) -> httpx.Response:
        requested.append(r.url.path)
        if r.url.path.endswith("/config"):
            return httpx.Response(200, json=config_unmeasured)
        return httpx.Response(200, json={})

    with (
        patch.object(run_evaluation, "MANIFEST", fake_manifest),
        patch.object(run_evaluation, "PROTOCOL", fake_protocol),
        patch.object(run_evaluation, "ROOT", fake_manifest.parent.parent),
    ):
        rc = run_evaluation.main(
            ["--split", "development", "--out", str(tmp_path / "out")],
            transport=httpx.MockTransport(_handler),
        )

    assert rc == 1
    # Refused on the flag, not somewhere downstream: /config is the only request
    # that was made, and the reason names the setting the operator has to change.
    assert requested == ["/api/v1/config"]
    assert "evaluation_batch_open" in capsys.readouterr().err


def test_measurement_valid_false_yields_nonzero_exit(
    tmp_path: Path, fake_manifest: Path, fake_protocol: Path
) -> None:
    """If a run has measurement_valid=False, exit code must be non-zero."""

    def _handler(r: httpx.Request) -> httpx.Response:
        if r.url.path.endswith("/config"):
            return httpx.Response(200, json=_CONFIG_RESP)
        if r.url.path.endswith("/documents"):
            return httpx.Response(
                201, json={"id": _DOC_ID, "content_hash": "x", "unit_count": 1}
            )
        if "/runs" in r.url.path and r.method == "POST":
            return httpx.Response(
                202,
                json=_make_completed_run(_RUN_ID_OFF, "off", measurement_valid=False),
            )
        if r.url.path.endswith(f"/runs/{_RUN_ID_OFF}"):
            return httpx.Response(
                200,
                json=_make_completed_run(_RUN_ID_OFF, "off", measurement_valid=False),
            )
        return httpx.Response(404, json={"code": "not_found", "message_pl": "brak"})

    with (
        patch.object(run_evaluation, "MANIFEST", fake_manifest),
        patch.object(run_evaluation, "PROTOCOL", fake_protocol),
        patch.object(run_evaluation, "ROOT", fake_manifest.parent.parent),
        patch("time.sleep"),
    ):
        rc = run_evaluation.main(
            ["--split", "development", "--arms", "off", "--out", str(tmp_path / "out")],
            transport=httpx.MockTransport(_handler),
        )

    assert rc != 0
    arm_data = json.loads(
        (tmp_path / "out" / "ED-TEST" / "off.json").read_text(encoding="utf-8")
    )
    assert arm_data["measurement_valid"] is False


def test_sha256_mismatch_aborts_artifact(tmp_path: Path, fake_protocol: Path) -> None:
    """A SHA256 mismatch should cause an artifact upload failure (non-zero exit)."""
    doc_path = tmp_path / "evaluation-data" / "documents" / "ED-BAD.txt"
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_bytes(b"different content")

    manifest = {
        "schema_version": 2,
        "dataset_id": "test",
        "dataset_version": "0.0.1",
        "documents": [
            {
                "id": "ED-BAD",
                "logical_case": "ED-BAD",
                "stratum": "synthetic_fixture",
                "canonical": True,
                "split": "development",
                "path": "documents/ED-BAD.txt",
                "sha256": "0" * 64,
                "format": "txt",
            }
        ],
    }
    m = tmp_path / "evaluation-data" / "manifest.json"
    m.write_text(json.dumps(manifest), encoding="utf-8")

    with (
        patch.object(run_evaluation, "MANIFEST", m),
        patch.object(run_evaluation, "PROTOCOL", fake_protocol),
        patch.object(run_evaluation, "ROOT", tmp_path),
    ):
        rc = run_evaluation.main(
            ["--split", "development", "--out", str(tmp_path / "out")],
            transport=httpx.MockTransport(lambda r: httpx.Response(201, json={})),
        )

    assert rc != 0, "sha256 mismatch must yield non-zero exit"


def test_cases_filter(tmp_path: Path, fake_manifest: Path, fake_protocol: Path) -> None:
    """A --cases filter matching nothing selects nothing and exits non-zero."""
    with (
        patch.object(run_evaluation, "MANIFEST", fake_manifest),
        patch.object(run_evaluation, "PROTOCOL", fake_protocol),
        patch.object(run_evaluation, "ROOT", fake_manifest.parent.parent),
    ):
        rc = run_evaluation.main(
            [
                "--split",
                "development",
                "--cases",
                "NONEXISTENT-001",
                "--out",
                str(tmp_path / "out"),
            ],
            transport=httpx.MockTransport(lambda r: httpx.Response(404, json={})),
        )
    assert rc != 0


def test_failed_run_yields_nonzero_exit(
    tmp_path: Path, fake_manifest: Path, fake_protocol: Path
) -> None:
    """A failed run should still write the file but return non-zero."""

    def _handler(r: httpx.Request) -> httpx.Response:
        if r.url.path.endswith("/config"):
            return httpx.Response(200, json=_CONFIG_RESP)
        if r.url.path.endswith("/documents"):
            return httpx.Response(
                201, json={"id": _DOC_ID, "content_hash": "x", "unit_count": 1}
            )
        if "/runs" in r.url.path and r.method == "POST":
            return httpx.Response(
                202,
                json={
                    "id": _RUN_ID_OFF,
                    "document_id": _DOC_ID,
                    "arm": "off",
                    "status": "running",
                    "measurement_valid": True,
                    "content_available": True,
                    "findings": [],
                    "metrics": None,
                },
            )
        if r.url.path.endswith(f"/runs/{_RUN_ID_OFF}"):
            return httpx.Response(
                200,
                json={
                    "id": _RUN_ID_OFF,
                    "document_id": _DOC_ID,
                    "arm": "off",
                    "status": "failed",
                    "measurement_valid": True,
                    "content_available": True,
                    "findings": [],
                    "error": {"code": "budget_exhausted", "message_pl": "budżet"},
                    "metrics": {
                        "input_tokens": 10,
                        "output_tokens": 5,
                        "elapsed_ms": 100.0,
                        "units_total": 1,
                        "units_with_finding": 0,
                        "units_not_processed": 1,
                        "context_edge_count": None,
                        "cost": {
                            "monetary_cost_microunits": None,
                            "unknown_reason": "x",
                        },
                        "attempts": [],
                    },
                },
            )
        return httpx.Response(404, json={"code": "not_found", "message_pl": "brak"})

    with (
        patch.object(run_evaluation, "MANIFEST", fake_manifest),
        patch.object(run_evaluation, "PROTOCOL", fake_protocol),
        patch.object(run_evaluation, "ROOT", fake_manifest.parent.parent),
        patch("time.sleep"),
    ):
        rc = run_evaluation.main(
            ["--split", "development", "--arms", "off", "--out", str(tmp_path / "out")],
            transport=httpx.MockTransport(_handler),
        )

    assert rc != 0
    assert (tmp_path / "out" / "ED-TEST" / "off.json").exists()


def test_resume_refuses_a_batch_started_under_other_code(
    tmp_path: Path, fake_manifest: Path, fake_protocol: Path, capsys
) -> None:
    """A results directory is one batch, so it is one implementation state.

    Skipping the runs that already exist and executing the rest under changed
    code would leave one summary describing results from two states, signed by
    whichever invocation finished last. Parity would not catch it: prompts,
    tools, model and budgets can all be identical while the implementation
    behind them differs.
    """
    out_dir = tmp_path / "out"
    artifact_dir = out_dir / "ED-TEST"
    artifact_dir.mkdir(parents=True)
    doc_resp = {"id": _DOC_ID, "content_hash": "abc", "unit_count": 5}
    (artifact_dir / "document.json").write_text(json.dumps(doc_resp), encoding="utf-8")
    (artifact_dir / "off.json").write_text(
        json.dumps(_make_completed_run(_ARM_TO_RUN_ID["off"], "off")), encoding="utf-8"
    )

    proto_a = tmp_path / "proto_a.json"
    proto_b = tmp_path / "proto_b.json"
    proto_data = json.loads(fake_protocol.read_text(encoding="utf-8"))
    proto_a.write_text(
        json.dumps({**proto_data, "protocol_version": "2.6.0"}), encoding="utf-8"
    )
    proto_b.write_text(
        json.dumps({**proto_data, "protocol_version": "2.6.1"}), encoding="utf-8"
    )

    def run_with(proto: Path) -> int:
        with (
            patch.object(run_evaluation, "MANIFEST", fake_manifest),
            patch.object(run_evaluation, "PROTOCOL", proto),
            patch.object(run_evaluation, "ROOT", fake_manifest.parent.parent),
            patch("time.sleep"),
        ):
            return run_evaluation.main(
                ["--split", "development", "--arms", "off,mid", "--out", str(out_dir)],
                transport=_FakeTransport(),
            )

    assert run_with(proto_a) == 0
    identity = json.loads(
        (out_dir / run_evaluation.IDENTITY_FILE).read_text(encoding="utf-8")
    )
    assert identity["protocol_version"] == "2.6.0"

    (artifact_dir / "mid.json").unlink()
    assert run_with(proto_b) == 2
    assert "cannot be resumed" in capsys.readouterr().err
    # The batch keeps the identity it started with, and nothing new was run.
    assert not (artifact_dir / "mid.json").exists()
    assert (
        json.loads(
            (out_dir / run_evaluation.IDENTITY_FILE).read_text(encoding="utf-8")
        )["protocol_version"]
        == "2.6.0"
    )


def test_summary_window_comes_from_the_runs_not_the_invocation(
    tmp_path: Path, fake_manifest: Path, fake_protocol: Path
) -> None:
    """The window belongs to the batch.

    A resumed batch reassembles its summary in a fraction of a second, and a
    summary claiming that window would describe nothing that happened.
    """
    out_dir = tmp_path / "out"
    artifact_dir = out_dir / "ED-TEST"
    artifact_dir.mkdir(parents=True)
    doc_resp = {"id": _DOC_ID, "content_hash": "abc", "unit_count": 5}
    (artifact_dir / "document.json").write_text(json.dumps(doc_resp), encoding="utf-8")
    completed = _make_completed_run(_ARM_TO_RUN_ID["off"], "off")
    completed["finished_at"] = "2026-09-03T19:09:53.629147Z"
    (artifact_dir / "off.json").write_text(json.dumps(completed), encoding="utf-8")

    with (
        patch.object(run_evaluation, "MANIFEST", fake_manifest),
        patch.object(run_evaluation, "PROTOCOL", fake_protocol),
        patch.object(run_evaluation, "ROOT", fake_manifest.parent.parent),
        patch("time.sleep"),
    ):
        rc = run_evaluation.main(
            ["--split", "development", "--arms", "off", "--out", str(out_dir)],
            transport=_FakeTransport(),
        )

    assert rc == 0
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["finished_at"] == "2026-09-03T19:09:53.629147Z"
    assert summary["summary_generated_at"] != summary["finished_at"]


def test_holdout_refused_while_the_failure_rule_is_owed(
    tmp_path: Path, fake_manifest: Path, capsys
) -> None:
    """A rule chosen after seeing a final result is chosen from the answer."""
    proto = {
        "state": "frozen",
        "protocol_version": "2.4.7",
        "final_or_holdout_data_seen": False,
        "holdout_discipline": {"forbidden_until_final_run": []},
        "decision_rule": {"thresholds": {"tau_RelComp": {"value": 0.1, "owed": True}}},
        "terminal_failure_policy": {"value": None, "owed": True},
    }
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps(proto), encoding="utf-8")
    with (
        patch.object(run_evaluation, "MANIFEST", fake_manifest),
        patch.object(run_evaluation, "PROTOCOL", protocol),
        patch.object(run_evaluation, "ROOT", fake_manifest.parent.parent),
    ):
        rc = run_evaluation.main(
            [
                "--split",
                "holdout",
                "--i-confirm-holdout",
                "--out",
                str(tmp_path / "out"),
            ],
            transport=httpx.MockTransport(lambda r: httpx.Response(200, json={})),
        )

    assert rc == 2
    assert "terminal_failure_policy" in capsys.readouterr().err


def _identity_of(out_dir: Path) -> dict[str, object]:
    return json.loads(
        (out_dir / run_evaluation.IDENTITY_FILE).read_text(encoding="utf-8")
    )


def test_resume_refuses_a_different_split_in_the_same_directory(
    tmp_path: Path, fake_protocol: Path
) -> None:
    """A holdout run must never land in a development directory.

    Checked on the guard rather than through the CLI, because a holdout
    invocation is stopped by the frozen-protocol gate long before it reaches
    the identity, and the pin has to hold for the day that gate is satisfied.
    """
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    proto = json.loads(fake_protocol.read_text(encoding="utf-8"))
    config = {"corpus_snapshot": "s1", "measured_mode": True}

    def identity_for(split: str) -> tuple[dict[str, Any], str | None]:
        args = argparse.Namespace(split=split, cases=None, include_variants=False)
        with (
            patch.object(run_evaluation, "PROTOCOL", fake_protocol),
            patch.object(run_evaluation, "ROOT", tmp_path),
        ):
            return run_evaluation._batch_identity(out_dir, args, proto, config, ["off"])

    recorded, error = identity_for("development")
    assert error is None
    assert recorded["split"] == "development"

    _, error = identity_for("holdout")
    assert error is not None
    assert "split" in error
    assert _identity_of(out_dir)["split"] == "development"


@pytest.mark.parametrize(
    "parity_field",
    [
        # Which vector space the corpus was built in.
        "embedding_space_fingerprint",
        # Which provider answered. One identifier served by two endpoints is
        # two servings, so the batch has to pin the host as well as the name.
        "model_endpoint",
    ],
)
def test_resume_refuses_a_server_whose_parity_facts_changed(
    parity_field: str,
    tmp_path: Path,
    fake_manifest: Path,
    fake_protocol: Path,
    capsys,
) -> None:
    """Two servers in one batch are two measurements wearing one summary."""
    out_dir = tmp_path / "out"

    def run_against(served: str) -> int:
        transport = _FakeTransport()
        # Everything else the server reports is identical, including the model
        # identifier, so the refusal can only come from the field under test.
        transport.config_overrides = {parity_field: served}
        with (
            patch.object(run_evaluation, "MANIFEST", fake_manifest),
            patch.object(run_evaluation, "PROTOCOL", fake_protocol),
            patch.object(run_evaluation, "ROOT", fake_manifest.parent.parent),
            patch("time.sleep"),
        ):
            return run_evaluation.main(
                ["--split", "development", "--arms", "off", "--out", str(out_dir)],
                transport=transport,
            )

    assert run_against("served-by-host-a") == 0
    assert run_against("served-by-host-b") == 2
    assert "config_fingerprint" in capsys.readouterr().err


@pytest.mark.parametrize(
    "relative_path",
    [
        "backend/src/mod.py",
        "pyproject.toml",
        "uv.lock",
        "compose.yaml",
        "backend/Dockerfile",
        "seeder/Dockerfile",
        "web/Dockerfile",
    ],
)
def test_resume_refuses_when_runtime_implementation_changes(
    tmp_path: Path,
    fake_manifest: Path,
    fake_protocol: Path,
    capsys,
    relative_path: str,
) -> None:
    """Modifying submitted runtime inputs mid-batch invalidates resume."""
    out_dir = tmp_path / "out"
    runtime_file = tmp_path / relative_path
    runtime_file.parent.mkdir(parents=True, exist_ok=True)
    runtime_file.write_text("version 1\n", encoding="utf-8")

    transport = _FakeTransport()
    with (
        patch.object(run_evaluation, "MANIFEST", fake_manifest),
        patch.object(run_evaluation, "PROTOCOL", fake_protocol),
        patch.object(run_evaluation, "ROOT", tmp_path),
        patch("time.sleep"),
    ):
        res1 = run_evaluation.main(
            ["--split", "development", "--arms", "off", "--out", str(out_dir)],
            transport=transport,
        )
        assert res1 == 0

        runtime_file.write_text("version 2\n", encoding="utf-8")

        res2 = run_evaluation.main(
            ["--split", "development", "--arms", "off", "--out", str(out_dir)],
            transport=transport,
        )
        assert res2 == 2
        assert "implementation_fingerprint" in capsys.readouterr().err


def test_a_resumed_batch_reports_every_arm_it_holds(
    tmp_path: Path, fake_manifest: Path, fake_protocol: Path
) -> None:
    """Finishing the arms a pause left behind is a resume, not a new batch.

    Pinning the arms would forbid it; reporting only the arms of the last
    invocation would describe a third of the measurement. The summary covers
    what the directory holds.
    """
    out_dir = tmp_path / "out"

    def run_arms(arms: str) -> int:
        with (
            patch.object(run_evaluation, "MANIFEST", fake_manifest),
            patch.object(run_evaluation, "PROTOCOL", fake_protocol),
            patch.object(run_evaluation, "ROOT", fake_manifest.parent.parent),
            patch("time.sleep"),
        ):
            return run_evaluation.main(
                ["--split", "development", "--arms", arms, "--out", str(out_dir)],
                transport=_FakeTransport(),
            )

    assert run_arms("off") == 0
    claimed = _identity_of(out_dir)["batch_id"]
    assert run_arms("mid,on") == 0

    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert sorted(row["arm"] for row in summary["arm_results"]) == ["mid", "off", "on"]
    # The header describes the same batch as the rows beneath it.
    assert summary["arms"] == ["mid", "off", "on"]
    assert summary["invocations"] == 2
    identity = _identity_of(out_dir)
    assert identity["requested"]["arms"] == ["mid", "off", "on"]
    assert identity["requested"]["invocations"] == 2
    # The id is what the registered final receipt binds to, so a directory that
    # minted a fresh one per invocation would lock the operator out of a batch
    # halfway through: the early-claim failure arriving from the other side.
    assert identity["batch_id"] == claimed


def test_a_failure_from_an_earlier_invocation_still_fails_the_batch(
    tmp_path: Path, fake_manifest: Path, fake_protocol: Path
) -> None:
    """The exit code covers the batch, because the summary does.

    A batch finished across two invocations must not report success because the
    invocation that happened to finish it went cleanly.
    """
    out_dir = tmp_path / "out"
    artifact_dir = out_dir / "ED-TEST"
    artifact_dir.mkdir(parents=True)
    doc_resp = {"id": _DOC_ID, "content_hash": "abc", "unit_count": 5}
    (artifact_dir / "document.json").write_text(json.dumps(doc_resp), encoding="utf-8")
    failed = _make_completed_run(_ARM_TO_RUN_ID["off"], "off")
    failed["status"] = "failed"
    failed["error"] = {"code": "model_transport_error", "message_pl": "x"}
    (artifact_dir / "off.json").write_text(json.dumps(failed), encoding="utf-8")

    with (
        patch.object(run_evaluation, "MANIFEST", fake_manifest),
        patch.object(run_evaluation, "PROTOCOL", fake_protocol),
        patch.object(run_evaluation, "ROOT", fake_manifest.parent.parent),
        patch("time.sleep"),
    ):
        rc = run_evaluation.main(
            ["--split", "development", "--arms", "mid", "--out", str(out_dir)],
            transport=_FakeTransport(),
        )

    assert rc != 0
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert {row["arm"]: row["status"] for row in summary["arm_results"]} == {
        "mid": "completed",
        "off": "failed",
    }


def test_an_unreadable_result_is_reported_not_dropped(
    tmp_path: Path, fake_manifest: Path, fake_protocol: Path, capsys
) -> None:
    """A summary that quietly describes less than the directory holds is worse."""
    out_dir = tmp_path / "out"
    artifact_dir = out_dir / "ED-TEST"
    artifact_dir.mkdir(parents=True)
    doc_resp = {"id": _DOC_ID, "content_hash": "abc", "unit_count": 5}
    (artifact_dir / "document.json").write_text(json.dumps(doc_resp), encoding="utf-8")
    (artifact_dir / "off.json").write_text("{ truncated", encoding="utf-8")

    with (
        patch.object(run_evaluation, "MANIFEST", fake_manifest),
        patch.object(run_evaluation, "PROTOCOL", fake_protocol),
        patch.object(run_evaluation, "ROOT", fake_manifest.parent.parent),
        patch("time.sleep"),
    ):
        rc = run_evaluation.main(
            ["--split", "development", "--arms", "mid", "--out", str(out_dir)],
            transport=_FakeTransport(),
        )

    assert rc != 0
    assert "cannot be read" in capsys.readouterr().err


def test_an_exposed_document_is_dropped_from_the_holdout_cohort() -> None:
    """A recorded exposure must exclude the document, not just be written down.

    ED-001 kept split=holdout while its exposure was recorded, because
    re-splitting after exposure is the author's decision. Without this the final
    run would still have selected it as clean material.
    """
    manifest = [
        {"id": "A", "split": "holdout", "canonical": True, "logical_case": "A"},
        {
            "id": "B",
            "split": "holdout",
            "canonical": True,
            "logical_case": "B",
            "holdout_exposure": "used as a benchmark before the final run",
        },
    ]

    selected = run_evaluation._select_artifacts(manifest, "holdout", None, False)

    assert [d["id"] for d in selected] == ["A"]
    # The exposure is recorded on the artifact it happened to, and a format
    # variant is the same document read another way rather than an independent
    # observation. Reading one variant's output is reading the case, so the
    # canonical artifact of that case must not stay in the clean cohort.
    variants = [
        {"id": "C-DOC", "split": "holdout", "canonical": True, "logical_case": "C"},
        {
            "id": "C-PDF",
            "split": "holdout",
            "canonical": False,
            "logical_case": "C",
            "holdout_exposure": "its output was read during development",
        },
    ]
    assert run_evaluation._select_artifacts(variants, "holdout", None, True) == []
    assert run_evaluation._registered_cohort(variants) == []
    # The same document is untouched for a development selection, which never
    # claims cleanliness.
    dev = [dict(d, split="development") for d in manifest]
    assert len(run_evaluation._select_artifacts(dev, "development", None, False)) == 2


def test_holdout_refused_while_any_obligation_is_open(
    tmp_path: Path, fake_manifest: Path
) -> None:
    """Any open obligation blocks the batch, not just the one we wrote first."""
    proto = {
        "state": "frozen",
        "protocol_version": "2.5.8",
        "final_or_holdout_data_seen": True,
        "final_registered_measurement_state": "not_started",
        # Deliberately unrelated wording: the gate must enforce the rule, not
        # recognise the sentence an earlier version was written against.
        "open_obligations": ["A stability procedure has not been fixed."],
        "holdout_discipline": {"forbidden_until_final_run": []},
        "decision_rule": {"thresholds": {"tau_RelComp": {"value": 0.05, "owed": True}}},
        "terminal_failure_policy": {"owed": False, "value": "fixed"},
    }
    p = tmp_path / "protocol.json"
    p.write_text(json.dumps(proto), encoding="utf-8")
    with (
        patch.object(run_evaluation, "MANIFEST", fake_manifest),
        patch.object(run_evaluation, "PROTOCOL", p),
        patch.object(run_evaluation, "ROOT", fake_manifest.parent.parent),
    ):
        rc = run_evaluation.main(
            [
                "--split",
                "holdout",
                "--i-confirm-holdout",
                "--out",
                str(tmp_path / "out"),
            ],
            transport=httpx.MockTransport(lambda r: httpx.Response(200, json={})),
        )
    assert rc == 2


def test_a_failed_precondition_does_not_consume_the_registered_batch(
    tmp_path: Path, fake_manifest: Path
) -> None:
    """The one registered batch is spent by measuring, not by trying to.

    The receipt was first written before the client opened, so an unreachable
    server, an unsealed one, a parity mismatch or a mistyped flag burned the
    single batch the protocol allows and the operator had to delete an evidence
    file to retry. It is claimed after every read-only check has passed instead.
    """
    root, protocol = _holdout_fixture(tmp_path, fake_manifest)
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        rc = _run_holdout(tmp_path, fake_manifest, protocol, root, "out-a")

    assert rc == 1, err.getvalue()[:500]
    assert "evaluation_batch_open" in err.getvalue()
    receipt = root / "evaluation-data" / "results" / run_evaluation.FINAL_RECEIPT
    assert not receipt.exists(), (
        "a batch that never measured anything was recorded as claimed"
    )


def test_the_registered_batch_resumes_its_directory_and_refuses_a_second(
    tmp_path: Path,
) -> None:
    """A protocol field nothing writes is a declaration, not a guard.

    final_registered_measurement_state reads not_started after a completed batch
    because no code moves it, so a second invocation with a fresh --out passed the
    same check. Nor is the cohort enough to tell the two apart: an empty --out
    over the same documents is every run executed again, and the claim used to
    read it as a resume. The batch is the directory whose results it produced.
    """
    root = tmp_path / "root"
    (root / "evaluation-data" / "results").mkdir(parents=True)
    identity = {field: field.upper() for field in run_evaluation._IDENTITY_FIELDS}
    identity["batch_id"] = "batch-one"
    cohort = ["ED-004", "ED-005"]
    receipt = root / "evaluation-data" / "results" / run_evaluation.FINAL_RECEIPT

    with patch.object(run_evaluation, "ROOT", root):
        first = tmp_path / "out-a"
        assert run_evaluation._claim_final_batch(identity, cohort, first) is None
        claimed_at = json.loads(receipt.read_text(encoding="utf-8"))["started_at"]

        # The same directory presenting the id it minted: the resume the runner
        # already supports, for the arms or cases the batch still owes.
        assert run_evaluation._claim_final_batch(identity, cohort, first) is None

        refusal = run_evaluation._claim_final_batch(
            {**identity, "batch_id": "two"}, cohort, tmp_path / "out-b"
        )
        assert refusal is not None, (
            "a fresh --out over the same cohort was accepted as the same batch"
        )
        assert "new protocol version" in refusal

        # A batch with no id of its own must not match another that has none.
        nameless = {k: v for k, v in identity.items() if k != "batch_id"}
        assert run_evaluation._claim_final_batch(nameless, cohort, first) is not None

    held = json.loads(receipt.read_text(encoding="utf-8"))
    assert held["started_at"] == claimed_at
    assert held["batch_id"] == "batch-one"
    assert held["out_dir"] == str(first)


def test_a_refused_batch_does_not_modify_existing_receipt(
    tmp_path: Path,
) -> None:
    """Failing safe must not also cost the operator the experiment."""
    root = tmp_path / "clone"
    results = root / "evaluation-data" / "results"
    results.mkdir(parents=True)
    identity = {field: field.upper() for field in run_evaluation._IDENTITY_FIELDS}
    real = tmp_path / "final-batch"
    real.mkdir()
    receipt_file = results / run_evaluation.FINAL_RECEIPT

    receipt_file.write_text(
        json.dumps(
            {
                "started_at": "2026-09-04T05:00:00+00:00",
                "batch_id": "the-real-batch",
                "registered_cohort": ["ED-004"],
                "out_dir": str(real.resolve()),
            }
        ),
        encoding="utf-8",
    )

    with patch.object(run_evaluation, "ROOT", root):
        assert (
            run_evaluation._claim_final_batch(
                {**identity, "batch_id": "a-mistake"}, ["ED-004"], tmp_path / "oops"
            )
            is not None
        )
        assert (
            json.loads(receipt_file.read_text(encoding="utf-8"))["batch_id"]
            == "the-real-batch"
        )
        assert (
            run_evaluation._claim_final_batch(
                {**identity, "batch_id": "the-real-batch"}, ["ED-004"], real
            )
            is None
        )


def test_a_copied_results_directory_is_not_the_batch_it_was_copied_from(
    tmp_path: Path,
) -> None:
    """Copying the directory copies the id it was named by.

    cp -r on a results directory produces two directories carrying one batch id,
    and both would have resumed the registered batch while the receipt named only
    the first. The claim records where the batch runs, and a resume has to come
    from there.
    """
    root = tmp_path / "root"
    (root / "evaluation-data" / "results").mkdir(parents=True)
    identity = {field: field.upper() for field in run_evaluation._IDENTITY_FIELDS}
    identity["batch_id"] = "one-batch"
    original = tmp_path / "out-a"
    copy = tmp_path / "out-a-copy"

    with patch.object(run_evaluation, "ROOT", root):
        assert run_evaluation._claim_final_batch(identity, ["ED-004"], original) is None
        refusal = run_evaluation._claim_final_batch(identity, ["ED-004"], copy)

    assert refusal is not None, "the copy resumed the batch it was copied from"
    assert "two places" in refusal


def test_the_registered_cohort_is_the_clean_holdout_not_the_selection() -> None:
    """--cases names the work slice; it does not get to name the experiment.

    The claim used to record whichever cases the invocation selected, so a first
    run over one document registered a one-document final comparison and refused
    the rest of the stratum afterwards as a different batch. The cohort is keyed
    by logical case, which is also why format variants do not enter it twice.
    """
    manifest = [
        {"id": "ED-004", "split": "holdout", "canonical": True},
        {"id": "RT-001-DOC", "split": "holdout", "logical_case": "RT-001"},
        {"id": "RT-001-PDF", "split": "holdout", "logical_case": "RT-001"},
        {"id": "ED-001", "split": "holdout", "holdout_exposure": "benchmark"},
        {"id": "CA-009", "split": "development", "canonical": True},
    ]

    assert run_evaluation._registered_cohort(manifest) == ["ED-004", "RT-001"]


def test_an_unreadable_cell_matrix_is_refused_before_the_batch_is_claimed(
    tmp_path: Path, fake_manifest: Path
) -> None:
    """A typo in the protocol must not cost the one registered batch.

    The completeness report is built at the end of the run, so a comparison
    naming a cohort the protocol no longer defines raised after every case had
    executed and the claim had been taken -- the batch spent, and no summary to
    show for it. The matrix is read while the run is still free to refuse.
    """
    root, protocol = _holdout_fixture(tmp_path, fake_manifest)
    broken = json.loads(protocol.read_text(encoding="utf-8"))
    broken["registered_comparisons"][0]["cohort"] = "renamed_last_week"
    protocol.write_text(json.dumps(broken), encoding="utf-8")

    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        rc = _run_holdout(
            tmp_path, fake_manifest, protocol, root, "out", transport=_FakeTransport()
        )

    assert rc == 2, err.getvalue()[-500:]
    assert "renamed_last_week" in err.getvalue()
    receipt = root / "evaluation-data" / "results" / run_evaluation.FINAL_RECEIPT
    assert not receipt.exists(), "an unreadable protocol consumed the registered batch"


def test_a_registered_batch_that_ran_a_slice_does_not_report_a_measurement(
    tmp_path: Path, fake_manifest: Path
) -> None:
    """Registering the whole cohort is not the same as having run it.

    --cases and --arms may work through the batch in pieces, and an invocation
    that finishes its piece cleanly exits zero, so nothing said whether the
    experiment the receipt registered actually exists. A batch could be claimed
    over ten documents, run one arm of one of them, and be recorded as a
    completed measurement.
    """
    root, protocol = _holdout_fixture(tmp_path, fake_manifest)
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        rc = _run_holdout(
            tmp_path,
            fake_manifest,
            protocol,
            root,
            "out",
            arms="off",
            transport=_FakeTransport(),
        )
    assert rc == 0, err.getvalue()[-800:]

    summary = json.loads(
        (tmp_path / "out" / "summary.json").read_text(encoding="utf-8")
    )
    registered = summary["registered_batch"]
    assert registered["cohort"] == ["HD-TEST"]
    # MID/OFF over the whole stratum; the graph ablation's cohort holds no case
    # of this manifest, so ON is owed by nothing.
    assert registered["required_cells"] == 2
    assert registered["missing_cells"] == [{"case": "HD-TEST", "arm": "mid"}]
    assert registered["complete"] is False
    assert "the registered batch is not complete" in err.getvalue()

    with contextlib.redirect_stderr(err):
        assert (
            _run_holdout(
                tmp_path,
                fake_manifest,
                protocol,
                root,
                "out",
                arms="mid",
                transport=_FakeTransport(),
            )
            == 0
        )
    finished = json.loads(
        (tmp_path / "out" / "summary.json").read_text(encoding="utf-8")
    )["registered_batch"]
    assert finished["missing_cells"] == []
    assert finished["complete"] is True


def test_a_format_variant_does_not_fill_the_cell_its_case_owes() -> None:
    """The comparison is over canonical artifacts, so only those close a cell.

    RT-001 enters the holdout through RT-001-DOC and carries RT-001-PDF beside
    it. A run of the variant alone would otherwise report the case as measured,
    while no quality aggregate would ever read it.
    """
    manifest = [
        {
            "id": "RT-001-DOC",
            "split": "holdout",
            "logical_case": "RT-001",
            "canonical": True,
        },
        {
            "id": "RT-001-PDF",
            "split": "holdout",
            "logical_case": "RT-001",
            "canonical": False,
        },
    ]
    proto = {
        "registered_comparisons": [
            {"id": "c", "pair": ["OFF"], "cohort": "full stratum"}
        ]
    }

    variant_only = run_evaluation._batch_coverage(
        proto, manifest, [{"artifact_id": "RT-001-PDF", "arm": "off"}]
    )
    assert variant_only["missing_cells"] == [{"case": "RT-001", "arm": "off"}]

    canonical = run_evaluation._batch_coverage(
        proto, manifest, [{"artifact_id": "RT-001-DOC", "arm": "off"}]
    )
    assert canonical["complete"] is True


def test_a_comparison_over_an_unknown_cohort_is_not_silently_the_whole_stratum(
    tmp_path: Path,
) -> None:
    """The protocol owns what is compared over what, so it has to be readable.

    Defaulting an unrecognised cohort name to the full stratum would make the
    completeness report agree with any protocol, including one whose comparison
    points at a cohort that no longer exists.
    """
    manifest = [{"id": "HD", "split": "holdout", "canonical": True}]
    proto = {
        "registered_comparisons": [
            {"id": "graph_ablation", "pair": ["ON", "MID"], "cohort": "renamed_cohort"}
        ],
        "cohorts": {"graph_eligible_holdout": {"cases": ["HD"]}},
    }

    with pytest.raises(RuntimeError, match="renamed_cohort"):
        run_evaluation._registered_cells(proto, manifest)


def _holdout_fixture(tmp_path: Path, manifest: Path) -> tuple[Path, Path]:
    protocol = tmp_path / "protocol.json"
    protocol.write_text(
        json.dumps(
            {
                "state": "frozen",
                "protocol_version": "2.5.11",
                "final_or_holdout_data_seen": True,
                "final_registered_measurement_state": "not_started",
                "open_obligations": [],
                "holdout_discipline": {"forbidden_until_final_run": []},
                "decision_rule": {
                    "thresholds": {"tau_RelComp": {"value": 0.05, "owed": True}}
                },
                "terminal_failure_policy": {"owed": False, "value": "fixed"},
                "registered_comparisons": [
                    {
                        "id": "configuration_contrast",
                        "pair": ["MID", "OFF"],
                        "cohort": "full stratum",
                    },
                    {
                        "id": "graph_ablation",
                        "pair": ["ON", "MID"],
                        "cohort": "graph_eligible_holdout",
                    },
                ],
                "cohorts": {"graph_eligible_holdout": {"cases": ["NOT-HD-TEST"]}},
            }
        ),
        encoding="utf-8",
    )
    # The manifest's own tree, so the documents it points at are where the
    # runner looks for them.
    root = manifest.parent.parent
    (root / "evaluation-data" / "results").mkdir(parents=True, exist_ok=True)
    return root, protocol


def _run_holdout(
    tmp_path: Path,
    manifest: Path,
    protocol: Path,
    root: Path,
    out_name: str,
    *,
    arms: str | None = None,
    transport: httpx.BaseTransport | None = None,
    extra_argv: list[str] | None = None,
) -> int:
    argv = [
        "--split",
        "holdout",
        "--i-confirm-holdout",
        "--out",
        str(tmp_path / out_name),
    ]
    if arms is not None:
        argv += ["--arms", arms]
    argv += extra_argv or []
    with (
        patch.object(run_evaluation, "MANIFEST", manifest),
        patch.object(run_evaluation, "PROTOCOL", protocol),
        patch.object(run_evaluation, "ROOT", root),
        patch("time.sleep"),
    ):
        return run_evaluation.main(
            argv,
            transport=transport
            or httpx.MockTransport(lambda r: httpx.Response(200, json={})),
        )


@pytest.mark.parametrize("source_format", ["fallback", "unrecorded"])
def test_holdout_refused_when_the_corpus_is_not_the_one_the_manifest_declares(
    tmp_path: Path, fake_manifest: Path, source_format: str
) -> None:
    """A substituted corpus must not be measured because nobody read a field.

    ELI serves an act's HTML and its PDF from separate routes and the HTML has
    failed on its own, so the seeder falls back and records what it read. The
    protocol used to close the loop by telling an operator to check that record
    before the final run, which is the kind of discipline this project does not
    rely on anywhere else.

    unrecorded is refused beside fallback: a snapshot from before the source formats
    were recorded cannot answer the question, and reading silence as a clean
    answer is exactly the failure the check replaces.
    """
    root, protocol = _holdout_fixture(tmp_path, fake_manifest)
    transport = _FakeTransport()
    transport.config_overrides = {"corpus_source_format": source_format}

    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        rc = _run_holdout(
            tmp_path, fake_manifest, protocol, root, "out", transport=transport
        )

    assert rc == 2, err.getvalue()[-500:]
    assert "corpus_source_format" in err.getvalue()
    receipt = root / "evaluation-data" / "results" / run_evaluation.FINAL_RECEIPT
    assert not receipt.exists(), (
        "a batch refused on its corpus consumed the one registered measurement"
    )


def test_a_fallback_corpus_is_measurable_as_an_explicit_deviation(
    tmp_path: Path, fake_manifest: Path
) -> None:
    """The gate stops an unnoticed substitution, not a declared one.

    A corpus built through the fallback is a real experimental condition when the
    person running it says so and the batch identity records it. Refusing that
    outright would make an ELI outage the end of the experiment rather than a
    deviation to interpret.
    """
    root, protocol = _holdout_fixture(tmp_path, fake_manifest)
    transport = _FakeTransport()
    transport.config_overrides = {"corpus_source_format": "fallback"}

    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        rc = _run_holdout(
            tmp_path,
            fake_manifest,
            protocol,
            root,
            "out",
            transport=transport,
            extra_argv=["--allow-source-format-fallback"],
        )

    assert rc != 2, (
        "an explicitly allowed fallback was still refused: " + err.getvalue()[-400:]
    )


def test_the_measured_source_format_is_written_into_the_batch_identity(
    tmp_path: Path, fake_manifest: Path
) -> None:
    """A deviation nobody can read afterwards is not recorded.

    --allow-source-format-fallback lets the batch run; what makes it a deviation
    rather than a silent substitution is that the source format it ran against is
    written into the identity the results carry. Asserting the field is in the
    parity tuple only proved it was spelled right somewhere.
    """
    root, protocol = _holdout_fixture(tmp_path, fake_manifest)
    transport = _FakeTransport()
    transport.config_overrides = {"corpus_source_format": "fallback"}

    with contextlib.redirect_stderr(io.StringIO()):
        _run_holdout(
            tmp_path,
            fake_manifest,
            protocol,
            root,
            "out",
            transport=transport,
            extra_argv=["--allow-source-format-fallback"],
        )

    identity = json.loads(
        (tmp_path / "out" / run_evaluation.IDENTITY_FILE).read_text(encoding="utf-8")
    )
    assert identity["config"]["corpus_source_format"] == "fallback", identity["config"]


def test_a_corpus_refusal_leaves_the_directory_reusable(
    tmp_path: Path, fake_manifest: Path
) -> None:
    """_batch_identity promises a refusal costs nothing, and this one did.

    The identity file is written on the first call for a directory and folds
    corpus_source_format in as a parity field, so refusing after it recorded the
    refused value: rebuilding the corpus and rerunning the same --out then failed
    on config_fingerprint drift, which reads as a second, unrelated problem.
    """
    root, protocol = _holdout_fixture(tmp_path, fake_manifest)
    refused = _FakeTransport()
    refused.config_overrides = {"corpus_source_format": "fallback"}

    with contextlib.redirect_stderr(io.StringIO()):
        rc = _run_holdout(
            tmp_path, fake_manifest, protocol, root, "out", transport=refused
        )
    assert rc == 2
    assert not (tmp_path / "out" / run_evaluation.IDENTITY_FILE).exists(), (
        "the refused corpus was recorded as this directory's batch identity"
    )

    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        rc = _run_holdout(
            tmp_path, fake_manifest, protocol, root, "out", transport=_FakeTransport()
        )
    assert "config_fingerprint" not in err.getvalue(), (
        "a fixed corpus was refused for drifting from the corpus that was refused"
    )


_SERIES_2_DIR = "evaluation-data/results/final/series-2"


def _repeat_fixture(tmp_path: Path, manifest: Path) -> tuple[Path, Path]:
    """A protocol whose registered batch has run and which authorizes a repeat."""
    root, protocol = _holdout_fixture(tmp_path, manifest)
    proto = json.loads(protocol.read_text(encoding="utf-8"))
    proto["final_registered_measurement_state"] = "completed"
    proto["registered_repeat_series"] = {
        "series": [
            {
                "series": 1,
                "results": "evaluation-data/results/final/series-1/",
                "receipt": run_evaluation.FINAL_RECEIPT,
                "state": "completed",
            },
            {
                "series": 2,
                "results": _SERIES_2_DIR,
                "receipt": "final-registered-measurement-series-2.json",
                "state": "authorized",
            },
        ],
        "series_1_config": {
            field: _CONFIG_RESP[field]
            for field in run_evaluation._CONFIG_PARITY_FIELDS
            if field != "model_endpoint"
        },
        "required_values": {
            "model_endpoint": _CONFIG_RESP["model_endpoint"],
            "evaluation_batch_open": True,
        },
    }
    protocol.write_text(json.dumps(proto), encoding="utf-8")
    return root, protocol


def test_an_authorized_repeat_series_runs_and_claims_a_receipt_of_its_own(
    tmp_path: Path, fake_manifest: Path
) -> None:
    """The second series of a registered measurement is registered, not exempt.

    The single-batch rule refuses a second set of model runs after the first has
    been seen, and it should: that is how a measurement becomes a choice. A
    repeat the protocol authorized in advance, into the directory it names, is
    the case that rule leaves to a new protocol version, and it takes a claim of
    its own so that the first series' receipt keeps holding the first series.
    """
    root, protocol = _repeat_fixture(tmp_path, fake_manifest)
    results = root / "evaluation-data" / "results"
    first = results / run_evaluation.FINAL_RECEIPT
    first.write_text(json.dumps({"batch_id": "series-1", "out_dir": "elsewhere"}))
    before = first.read_bytes()

    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        rc = _run_holdout(
            tmp_path,
            fake_manifest,
            protocol,
            root,
            _SERIES_2_DIR,
            transport=_FakeTransport(),
        )

    assert rc == 0, err.getvalue()[-800:]
    receipt = json.loads(
        (results / "final-registered-measurement-series-2.json").read_text(
            encoding="utf-8"
        )
    )
    assert receipt["registered_series"] == 2
    # Relative to the artefact root, so the submitted receipt does not carry the
    # machine it ran on and reads the same in every copy of the tree.
    assert receipt["out_dir"] == _SERIES_2_DIR
    assert first.read_bytes() == before, (
        "the repeat rewrote the first series' claim instead of taking its own"
    )


def test_an_unregistered_directory_is_still_refused_after_the_first_batch(
    tmp_path: Path, fake_manifest: Path
) -> None:
    """Authorizing two repeats does not open the door to a third."""
    root, protocol = _repeat_fixture(tmp_path, fake_manifest)

    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        rc = _run_holdout(
            tmp_path, fake_manifest, protocol, root, "somewhere-else", transport=None
        )

    assert rc == 2
    assert "registered_repeat_series" in err.getvalue()
    assert not (tmp_path / "somewhere-else" / run_evaluation.IDENTITY_FILE).exists()


def test_a_repeat_is_refused_when_the_server_is_not_the_one_series_1_used(
    tmp_path: Path, fake_manifest: Path
) -> None:
    """A repeat against another model, corpus or provider repeats nothing.

    The batch identity would record the difference, but only in the file written
    beside results that had already been paid for. The served configuration is
    compared with the one the first series ran under before the first request.
    """
    root, protocol = _repeat_fixture(tmp_path, fake_manifest)
    transport = _FakeTransport()
    transport.config_overrides = {"model_request_id": "some-other-model"}

    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        rc = _run_holdout(
            tmp_path,
            fake_manifest,
            protocol,
            root,
            _SERIES_2_DIR,
            transport=transport,
        )

    assert rc == 2
    assert "model_request_id" in err.getvalue()
    receipt = (
        root
        / "evaluation-data"
        / "results"
        / "final-registered-measurement-series-2.json"
    )
    assert not receipt.exists(), "a refused repeat consumed its registered series"


def test_a_repeat_is_refused_when_the_endpoint_is_not_the_required_one(
    tmp_path: Path, fake_manifest: Path
) -> None:
    """The model identifier does not name the endpoint that served it.

    Two endpoints serving one identifier do not serve it identically, which is
    why the provider is a parity dimension and why a repeat names the host it
    requires rather than inheriting whatever the operator's shell last exported.
    """
    root, protocol = _repeat_fixture(tmp_path, fake_manifest)
    transport = _FakeTransport()
    transport.config_overrides = {"model_endpoint": "gateway.example"}

    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        rc = _run_holdout(
            tmp_path,
            fake_manifest,
            protocol,
            root,
            _SERIES_2_DIR,
            transport=transport,
        )

    assert rc == 2
    assert "model_endpoint" in err.getvalue()


def test_a_parity_field_no_recorded_value_covers_stops_the_repeat(
    tmp_path: Path, fake_manifest: Path
) -> None:
    """A parity field nothing compares is a field that can drift silently.

    The recorded series-1 configuration and the required values are two hand
    written maps; a dimension added to the parity tuple later would be checked by
    neither and would pass without being looked at.
    """
    root, protocol = _repeat_fixture(tmp_path, fake_manifest)
    proto = json.loads(protocol.read_text(encoding="utf-8"))
    del proto["registered_repeat_series"]["series_1_config"]["corpus_snapshot"]
    protocol.write_text(json.dumps(proto), encoding="utf-8")

    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        rc = _run_holdout(
            tmp_path,
            fake_manifest,
            protocol,
            root,
            _SERIES_2_DIR,
            transport=_FakeTransport(),
        )

    assert rc == 2
    assert "corpus_snapshot" in err.getvalue()


def test_a_repeat_series_resumes_its_own_directory_and_refuses_a_copy(
    tmp_path: Path, fake_manifest: Path
) -> None:
    """The per-series claim keeps the guarantees the single claim had."""
    root, _ = _repeat_fixture(tmp_path, fake_manifest)
    slot = {"series": 2, "receipt": "final-registered-measurement-series-2.json"}
    identity = {"batch_id": "repeat-2", "protocol_version": "2.8.0"}
    directory = tmp_path / _SERIES_2_DIR
    directory.mkdir(parents=True)
    copy = tmp_path / "copied-series-2"
    copy.mkdir()

    with patch.object(run_evaluation, "ROOT", root):
        assert (
            run_evaluation._claim_final_batch(identity, ["HD-TEST"], directory, slot)
            is None
        )
        assert (
            run_evaluation._claim_final_batch(identity, ["HD-TEST"], directory, slot)
            is None
        )
        refusal = run_evaluation._claim_final_batch(identity, ["HD-TEST"], copy, slot)
        traversal = run_evaluation._claim_final_batch(
            identity,
            ["HD-TEST"],
            directory,
            {"series": 2, "receipt": "../final-registered-measurement.json"},
        )

    assert refusal is not None and "copied" in refusal
    assert traversal is not None and "file name" in traversal
    assert not (
        root / "evaluation-data" / "results" / run_evaluation.FINAL_RECEIPT
    ).exists()


def test_poll_run_adaptive_polling() -> None:
    """_poll_run starts at min(0.5, poll_seconds) and adapts intervals up to
    poll_seconds.
    """
    polls = 0
    sleeps: list[float] = []

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal polls
        polls += 1
        if polls < 4:
            return httpx.Response(200, json={"status": "running"})
        return httpx.Response(200, json={"status": "completed"})

    client = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url="http://test",
    )
    with patch("time.sleep", side_effect=sleeps.append):
        run = run_evaluation._poll_run(
            client=client,
            run_id="r1",
            poll_seconds=2.0,
            timeout_seconds=10.0,
        )
    assert run["status"] == "completed"
    assert len(sleeps) == 3
    assert sleeps[0] == 0.5
    assert sleeps[1] == 0.75
    assert sleeps[2] == 1.125


def test_a_finished_series_directory_is_not_an_authorized_slot(
    tmp_path: Path, fake_manifest: Path
) -> None:
    """A series the protocol records as finished is not a door left open.

    Matching a slot by directory alone would let the first series' own output
    directory walk past the refusal that the single registered batch exists to
    make, and the guard the protocol names would no longer be the one working.
    """
    root, protocol = _repeat_fixture(tmp_path, fake_manifest)
    proto = json.loads(protocol.read_text(encoding="utf-8"))
    proto["registered_repeat_series"]["series"][1]["state"] = "completed"
    protocol.write_text(json.dumps(proto), encoding="utf-8")

    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        rc = _run_holdout(
            tmp_path,
            fake_manifest,
            protocol,
            root,
            _SERIES_2_DIR,
            transport=_FakeTransport(),
        )

    assert rc == 2
    assert "registered_repeat_series" in err.getvalue()


def test_a_directory_copied_inside_the_artefact_is_still_not_the_batch(
    tmp_path: Path, fake_manifest: Path
) -> None:
    """Relative locations still tell one directory from another.

    Recording where a batch ran relative to the artefact keeps the submitted
    receipt free of the machine it ran on. What it must not cost is the check
    that refuses continuing one registered batch in two places, so a copy under
    another name is still refused.
    """
    root, _ = _repeat_fixture(tmp_path, fake_manifest)
    slot = {"series": 2, "receipt": "final-registered-measurement-series-2.json"}
    identity = {"batch_id": "repeat-2", "protocol_version": "2.8.0"}
    directory = root / _SERIES_2_DIR
    directory.mkdir(parents=True)
    copy = root / "evaluation-data" / "results" / "final" / "copy-of-series-2"
    copy.mkdir(parents=True)

    with patch.object(run_evaluation, "ROOT", root):
        assert (
            run_evaluation._claim_final_batch(identity, ["HD-TEST"], directory, slot)
            is None
        )
        refusal = run_evaluation._claim_final_batch(identity, ["HD-TEST"], copy, slot)
        held = json.loads(
            (
                root
                / "evaluation-data"
                / "results"
                / "final-registered-measurement-series-2.json"
            ).read_text(encoding="utf-8")
        )

    assert refusal is not None and "copied" in refusal
    assert held["out_dir"] == _SERIES_2_DIR
    assert str(root) not in json.dumps(held)


def test_a_batch_outside_the_artefact_is_recorded_as_it_is(tmp_path: Path) -> None:
    """A directory with no relative form keeps its own; a registered one has one."""
    root = tmp_path / "artifact"
    root.mkdir()
    with patch.object(run_evaluation, "ROOT", root):
        inside = run_evaluation._recorded_location(
            root / "evaluation-data" / "results" / "final" / "batch"
        )
        outside = run_evaluation._recorded_location(tmp_path / "elsewhere")

    assert inside == "evaluation-data/results/final/batch"
    assert outside == str((tmp_path / "elsewhere").resolve())
