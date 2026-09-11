"""Tests for the HTTP layer (api.py).

Covers:
- OpenAPI conformance against the frozen openapi.yaml
- Upload-then-run journey reaching findings
- DELETE then GET content returning 404 while run metrics remain readable
- Privacy canary: text must not leak to logs, SSE events, or error bodies
- Error code mapping
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from collections.abc import Callable, Generator
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock, patch
from uuid import UUID, uuid4

import httpx
import pytest
import yaml  # type: ignore[import-untyped]
from model_client import mock_model_client
from qdrant_client import models
from starlette.testclient import TestClient
from worksheet_transport import envelope, transcript

from contract_analyzer.agents.parity import CONFIG_VERSION, RETRY_POLICY, ParityBundle
from contract_analyzer.agents.prompts import load_prompt_bundle
from contract_analyzer.agents.runner import AnalysisRunner
from contract_analyzer.agents.services import AnalysisServices
from contract_analyzer.agents.session import (
    DocumentSession,
    PreparedRun,
    RunRequest,
    RunResult,
)
from contract_analyzer.agents.tools import TOOL_BUNDLE_VERSION
from contract_analyzer.api import Prominence, create_app
from contract_analyzer.api import runs as run_endpoints
from contract_analyzer.api.run_builders import _PROMINENCE, _build_metrics
from contract_analyzer.api.state import _AppState
from contract_analyzer.config import RunConfig, Settings
from contract_analyzer.corpus import EMBEDDINGS_API_KEY_ENV, QdrantCorpusIndex
from contract_analyzer.domain import (
    ArmCode,
    EmittedBasis,
    FindingCode,
    ForceScope,
    ForceState,
    ForceValue,
    ProvisionKind,
)
from contract_analyzer.ingest import IngestService
from contract_analyzer.model import AttemptTelemetry
from contract_analyzer.storage import (
    ContentExpired,
    CostRecord,
    EventBus,
    FindingRecord,
    MetadataStore,
    RunRecord,
    RunTextStore,
    RunTotals,
)

_ROOT = Path(__file__).resolve().parent.parent.parent
_SPEC_PATH = _ROOT / "openapi.yaml"

DOCUMENT_TEXT = (
    "\u00a7 1. Najemca zobowi\u0105zuje si\u0119 do p\u0142atno\u015bci czynszu.\n"
    "\n"
    "\u00a7 2. Zgodnie z \u00a7 1 najemca ponosi koszty medi\u00f3w.\n"
    "\n"
    "\u00a7 3. Umowa podlega prawu polskiemu.\n"
)


def _make_transport() -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        if "tools" in payload:
            messages = cast(
                list[dict[str, object]], transcript(payload.get("input", []))
            )
            user_envelope = json.loads(cast(str, messages[1]["content"]))
            if user_envelope.get("prompt_version") == "synthesise":
                message = _synthesis_message(messages, user_envelope)
            elif any(m.get("role") == "tool" for m in messages):
                message = {"role": "assistant", "content": ""}
            else:
                unit_text = (
                    user_envelope["payload"]["unit_text"]
                    if isinstance(user_envelope.get("payload"), dict)
                    and "unit_text" in user_envelope["payload"]
                    else user_envelope.get("unit_text", "")
                )
                phrase = str(unit_text)[:60]
                message = {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "search_corpus",
                                "arguments": json.dumps({"phrase": phrase}),
                            },
                        }
                    ],
                }
            return httpx.Response(
                200,
                request=request,
                json=envelope(
                    message,
                    model="deepseek-v4-flash-20260828",
                    input_tokens=10,
                    output_tokens=2,
                ),
            )
        raise AssertionError("the pipeline sent a request carrying no tools")

    return httpx.MockTransport(handler)


def _synthesis_message(
    messages: list[dict[str, object]], user_envelope: dict[str, object]
) -> dict[str, object]:
    """List the findings once, then group every listed reference."""
    calls = [
        call["function"]["name"]
        for message in messages
        if message.get("role") == "assistant"
        for call in cast(list[dict[str, Any]], message.get("tool_calls") or [])
    ]
    if "list_findings" not in calls:
        return {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_list",
                    "type": "function",
                    "function": {"name": "list_findings", "arguments": "{}"},
                }
            ],
        }
    if "group_findings" not in calls:
        payload = cast(dict[str, Any], user_envelope.get("payload", {}))
        findings = cast(list[dict[str, Any]], payload.get("findings", []))
        groups: list[dict[str, Any]] = []
        if findings:
            groups = [
                {
                    "title": "Zobowiązania stron",
                    "summary": "System zgrupował ustalenia tej analizy.",
                    "finding_ids": [entry["ref"] for entry in findings],
                }
            ]
        return {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_group",
                    "type": "function",
                    "function": {
                        "name": "group_findings",
                        "arguments": json.dumps({"groups": groups}),
                    },
                }
            ],
        }
    return {"role": "assistant", "content": ""}


def _mock_corpus() -> Any:
    """Build a mock QdrantCorpusIndex that satisfies the API layer without network."""
    mock = MagicMock()
    mock.snapshot.id = "test-snapshot-id"
    mock.snapshot.file_hashes = {}
    mock.snapshot.source_format = "as_declared"

    # search returns empty candidates (no corpus matches -> no_basis_found)
    mock.search.return_value = []
    return mock


@pytest.fixture()
def _services(tmp_path: Path) -> AnalysisServices:
    settings = Settings(
        model_api_key="test-secret-key",
        model_base_url="https://openrouter.ai/api/v1",
        model_name="deepseek-v4-flash",
    )
    client = mock_model_client(_make_transport(), settings=settings)
    return AnalysisServices(
        settings=settings,
        corpus=_mock_corpus(),
        client=client,
        metadata=MetadataStore(tmp_path / "api_test.sqlite3"),
        prompt_bundle=load_prompt_bundle(),
        text_store=RunTextStore(),
        events=EventBus(),
    )


@pytest.fixture()
def client(
    _services: AnalysisServices, monkeypatch: pytest.MonkeyPatch
) -> Generator[TestClient, None, None]:
    settings = _services.settings
    app = create_app(settings, _services)
    with TestClient(app, raise_server_exceptions=False) as tc:
        yield tc


@pytest.fixture()
def run_invoked() -> list[bool]:
    return []


@pytest.fixture()
def _real_corpus_services(
    built_corpus: QdrantCorpusIndex, tmp_path: Path
) -> AnalysisServices:
    settings = Settings(
        model_api_key="test-secret-key",
        model_base_url="https://openrouter.ai/api/v1",
        model_name="deepseek-v4-flash",
    )
    client = mock_model_client(_make_transport(), settings=settings)
    return AnalysisServices(
        settings=settings,
        corpus=built_corpus,
        client=client,
        metadata=MetadataStore(tmp_path / "api_real_corpus.sqlite3"),
        prompt_bundle=load_prompt_bundle(),
        text_store=RunTextStore(),
        events=EventBus(),
    )


@pytest.fixture()
def real_corpus_client(
    _real_corpus_services: AnalysisServices,
    run_invoked: list[bool],
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[TestClient, None, None]:
    def stub_start_run(self: AnalysisRunner, run_request: RunRequest) -> PreparedRun:
        run_invoked.append(True)
        run_id = uuid4()
        session = _real_corpus_services.sessions[run_request.document_id]
        run_record = RunRecord(
            id=run_id,
            document_id=session.document_id,
            arm=run_request.arm,
            input_hash=session.input_hash,
            config_version=CONFIG_VERSION,
            prompt_bundle_version=_real_corpus_services.prompt_bundle.version,
            corpus_snapshot_id=_real_corpus_services.corpus.snapshot.id,
            tool_bundle_version=TOOL_BUNDLE_VERSION,
            requested_model=_real_corpus_services.settings.model_name,
            parameters=dict(run_request.parameters),
            retry_policy=RETRY_POLICY,
            concurrency=run_request.concurrency,
            wall_budget_seconds=run_request.wall_budget_seconds,
            measurement_valid=run_request.measurement_valid,
            cost=CostRecord.unknown("price_table_unavailable"),
            status="running",
        )
        _real_corpus_services.metadata.create_run(run_record)
        parity = ParityBundle(
            input_hash=session.input_hash,
            requested_model=_real_corpus_services.settings.model_name,
            returned_model=_real_corpus_services.settings.model_name,
            prompt_bundle_version=_real_corpus_services.prompt_bundle.version,
            corpus_snapshot_id=_real_corpus_services.corpus.snapshot.id,
            tool_bundle_version=TOOL_BUNDLE_VERSION,
            parameters=dict(run_request.parameters),
            retry_policy=RETRY_POLICY,
            concurrency=run_request.concurrency,
            wall_budget_seconds=run_request.wall_budget_seconds,
        )
        return PreparedRun(
            run_id=run_id,
            call_units=[],
            parity=parity,
            elapsed_ms=lambda: 0.0,
        )

    async def stub_execute_run(
        self: AnalysisRunner, prepared: PreparedRun
    ) -> RunResult:
        _real_corpus_services.metadata.finish_run(prepared.run_id, "completed")
        return RunResult(
            run_id=prepared.run_id,
            status="completed",
            parity_bundle=prepared.parity,
            call_units=[],
            context_unit_ids=[],
            findings=[],
            interruption=False,
            unprocessed_count=0,
            tokens_used=0,
        )

    monkeypatch.setattr(AnalysisRunner, "start_run", stub_start_run)
    monkeypatch.setattr(AnalysisRunner, "execute_run", stub_execute_run)
    settings = _real_corpus_services.settings
    app = create_app(settings, _real_corpus_services)
    with TestClient(app, raise_server_exceptions=False) as tc:
        yield tc


def _collect_retained_plaintext(
    state: _AppState, *, document_id: UUID | None = None
) -> str:
    """Concatenate document plaintext still held in sessions, runs, or text store."""
    parts: list[str] = []

    def consider(session: DocumentSession) -> None:
        if document_id is not None and session.document_id != document_id:
            return
        parts.append(session.payload.text)
        parts.extend(unit.text for unit in session.units)

    for session in state.services.sessions.values():
        consider(session)
    for active in state.runner._runtime._runs.values():
        consider(active.session)
    if document_id is not None:
        text_key = state.text_keys.get(document_id)
        if text_key is not None and state.services.text_store is not None:
            try:
                parts.append(state.services.text_store.get(text_key).decode("utf-8"))
            except ContentExpired:
                pass
    return "".join(parts)


def _upload_txt(tc: TestClient, text: str = DOCUMENT_TEXT) -> dict[str, Any]:
    resp = tc.post(
        "/api/v1/documents",
        files={"file": ("umowa.txt", text.encode("utf-8"), "text/plain")},
    )
    assert resp.status_code == 201, resp.text
    result: dict[str, Any] = resp.json()
    return result


def _wait_for_terminal_run(
    client: TestClient, run_id: str, *, timeout_seconds: float = 30.0
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        resp = client.get(f"/api/v1/runs/{run_id}")
        assert resp.status_code == 200, resp.text
        run_data = resp.json()
        if run_data["status"] in ("completed", "failed", "cancelled"):
            return run_data
        time.sleep(0.01)
    pytest.fail(
        f"run {run_id} did not reach a terminal state within {timeout_seconds}s"
    )


def _wait_until(condition: Callable[[], bool], timeout: float = 5.0) -> None:
    """Wait for something a run's own completion does not order.

    A run reports terminal when its record is written, inside execution; its
    slot is freed later, by the task's done callback.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(0.01)
    raise AssertionError("condition was not met within the timeout")


def _blocking_execute_patch(
    release: threading.Event,
) -> tuple[object, threading.Event, threading.Event]:
    started = threading.Event()
    original_entered = threading.Event()
    original_execute = AnalysisRunner.execute_run

    async def blocking_execute(
        self: AnalysisRunner, prepared: PreparedRun
    ) -> RunResult:
        started.set()
        while not release.is_set():
            await asyncio.sleep(0.01)
        original_entered.set()
        return await original_execute(self, prepared)

    AnalysisRunner.execute_run = blocking_execute  # type: ignore[method-assign]
    return original_execute, started, original_entered


def _load_spec() -> dict[str, Any]:
    result: dict[str, Any] = yaml.safe_load(_SPEC_PATH.read_text(encoding="utf-8"))
    return result


def _resolve_ref(spec: dict[str, Any], ref: str) -> dict[str, Any]:
    parts = ref.lstrip("#/").split("/")
    node: Any = spec
    for part in parts:
        node = node[part]
    result: dict[str, Any] = node
    return result


def _schema_field_names(spec: dict[str, Any], schema: dict[str, Any]) -> set[str]:
    if "$ref" in schema:
        schema = _resolve_ref(spec, schema["$ref"])
    return set(schema.get("properties", {}).keys())


def test_openapi_conformance() -> None:
    """FastAPI-generated schema conforms to the frozen openapi.yaml.

    Checks: every path, every method, every response status code, and response
    model field names for success responses.
    """
    spec = _load_spec()
    settings = Settings(
        model_api_key="conformance-key",
        model_name="deepseek-v4-flash",
    )
    services = AnalysisServices(
        settings=settings,
        corpus=_mock_corpus(),
        client=MagicMock(),
        metadata=MagicMock(),
        prompt_bundle=MagicMock(version="test-v1"),
        text_store=MagicMock(),
        events=MagicMock(),
    )
    app = create_app(settings, services)
    generated = app.openapi()

    problems: list[str] = []

    spec_info = spec.get("info", {})
    gen_info = generated.get("info", {})
    if spec_info.get("title") != gen_info.get("title"):
        problems.append(
            f"info.title mismatch: spec={spec_info.get('title')!r} "
            f"generated={gen_info.get('title')!r}"
        )
    if spec_info.get("version") != gen_info.get("version"):
        problems.append(
            f"info.version mismatch: spec={spec_info.get('version')!r} "
            f"generated={gen_info.get('version')!r}"
        )

    for spec_path, spec_methods in spec.get("paths", {}).items():
        full_path = f"/api/v1{spec_path}"
        if full_path not in generated.get("paths", {}):
            problems.append(f"missing path: {full_path}")
            continue
        gen_path = generated["paths"][full_path]

        for method in spec_methods:
            if method in ("parameters", "summary", "description"):
                continue
            if method not in gen_path:
                problems.append(f"missing method: {method.upper()} {full_path}")
                continue
            spec_responses = spec_methods[method].get("responses", {})
            gen_responses = gen_path[method].get("responses", {})
            for status_code in spec_responses:
                if status_code not in gen_responses:
                    problems.append(
                        f"missing status {status_code}: {method.upper()} {full_path}"
                    )
                    continue
                spec_resp = spec_responses[status_code]
                gen_resp = gen_responses[status_code]
                spec_content = spec_resp.get("content", {})
                gen_content = gen_resp.get("content", {})
                if (
                    "application/json" in spec_content
                    and "application/json" in gen_content
                ):
                    spec_schema = spec_content["application/json"].get("schema", {})
                    gen_schema = gen_content["application/json"].get("schema", {})
                    spec_fields = _schema_field_names(spec, spec_schema)
                    gen_fields = _schema_field_names(generated, gen_schema)
                    if spec_fields and gen_fields and spec_fields != gen_fields:
                        missing = spec_fields - gen_fields
                        extra = gen_fields - spec_fields
                        if missing:
                            problems.append(
                                f"missing fields in {method.upper()} {full_path} "
                                f"{status_code}: {sorted(missing)}"
                            )
                        if extra:
                            problems.append(
                                f"extra fields in {method.upper()} {full_path} "
                                f"{status_code}: {sorted(extra)}"
                            )

    assert not problems, "OpenAPI conformance failures:\n" + "\n".join(problems)


def test_upload_then_run_journey(client: TestClient) -> None:
    """Upload a document, start a run, and verify findings are returned."""
    doc = _upload_txt(client)
    doc_id = doc["id"]
    assert doc["unit_count"] >= 3

    content_resp = client.get(f"/api/v1/documents/{doc_id}/content")
    assert content_resp.status_code == 200
    content = content_resp.json()
    assert len(content["units"]) >= 3
    assert len(content["blocks"]) >= 1

    # Start a run (mock corpus returns no candidates -> no_basis_found findings)
    run_resp = client.post(
        "/api/v1/runs",
        json={"document_id": doc_id, "arm": "mid"},
    )
    assert run_resp.status_code == 202, run_resp.text
    run = run_resp.json()
    run_id = run["id"]
    assert run["status"] == "running"
    assert run["document_id"] == doc_id
    assert run["arm"] == "mid"
    assert run["measurement_valid"] is True
    assert run["content_available"] is True
    assert run["metrics"] is None

    run_data = _wait_for_terminal_run(client, run_id)
    assert run_data["status"] == "completed"
    assert len(run_data["findings"]) > 0
    assert run_data["metrics"] is not None

    finding = run_data["findings"][0]
    assert "id" in finding
    assert "unit_id" in finding
    assert "code" in finding
    assert "prominence" in finding
    assert finding["prominence"] in ("critical", "warning", "neutral")


def test_delete_then_get_content_returns_404(client: TestClient) -> None:
    """A deleted document has no session left, so its content is not found.

    410 is the other half of the route and means something else: the session is
    still there and its text has expired, which
    ``test_expired_text_before_the_sweep_returns_410`` covers.
    """
    doc = _upload_txt(client)
    doc_id = doc["id"]

    run_resp = client.post(
        "/api/v1/runs",
        json={"document_id": doc_id, "arm": "mid"},
    )
    assert run_resp.status_code == 202
    run_id = run_resp.json()["id"]
    _wait_for_terminal_run(client, run_id)
    del_resp = client.delete(f"/api/v1/documents/{doc_id}")
    assert del_resp.status_code == 204

    content_resp = client.get(f"/api/v1/documents/{doc_id}/content")
    assert content_resp.status_code == 404

    # The purge takes the analysis with the text: the run that read this
    # document, its findings and its attempts are gone, not merely unreadable.
    assert client.get(f"/api/v1/runs/{run_id}").status_code == 404


UNSTRUCTURED_TEXT = (
    "Najemca zobowiazuje sie do platnosci czynszu. "
    "Wynajmujacy przekazuje lokal w dniu podpisania. "
    "Umowa podlega prawu polskiemu. "
    "Strony ustalaja termin platnosci na dziesiaty dzien miesiaca."
)


@pytest.mark.parametrize("arm", ["off", "mid", "on"])
def test_every_arm_accepts_an_unstructured_document(
    client: TestClient, arm: str
) -> None:
    """No arm refuses a document; a zero-context ON run is a result, not an error."""
    doc = _upload_txt(client, UNSTRUCTURED_TEXT)

    resp = client.post("/api/v1/runs", json={"document_id": doc["id"], "arm": arm})

    assert resp.status_code == 202, resp.text
    run = _wait_for_terminal_run(client, resp.json()["id"])
    # The document has no markers to cite, so even ON runs on zero edges. Reported
    # rather than refused: this is the ablation's noise floor.
    assert run["metrics"]["context_edge_count"] == 0


@pytest.mark.parametrize(("arm", "expected_edges"), [("mid", 0), ("on", 1)])
def test_context_edge_count_separates_on_from_mid(
    client: TestClient, arm: str, expected_edges: int
) -> None:
    """The two per-unit arms differ only by context, so the wire must report it."""
    # DOCUMENT_TEXT's second paragraph cites its first: one resolvable edge.
    doc = _upload_txt(client, DOCUMENT_TEXT)

    resp = client.post("/api/v1/runs", json={"document_id": doc["id"], "arm": arm})

    assert resp.status_code == 202, resp.text
    run = _wait_for_terminal_run(client, resp.json()["id"])
    assert run["metrics"]["context_edge_count"] == expected_edges


def test_delete_purges_plaintext_from_runtime(client: TestClient) -> None:
    """DELETE must leave no reachable copy of the contract plaintext."""
    marker = "DELETE_PURGE_CANARY_7f3a9c2e"
    doc = _upload_txt(client, f"{marker}\n{DOCUMENT_TEXT}")
    doc_id = UUID(doc["id"])
    state: _AppState = client.app.state.app_state

    run_resp = client.post(
        "/api/v1/runs",
        json={"document_id": doc["id"], "arm": "mid"},
    )
    assert run_resp.status_code == 202
    run_id = run_resp.json()["id"]
    _wait_for_terminal_run(client, run_id)
    assert marker in _collect_retained_plaintext(state, document_id=doc_id)

    del_resp = client.delete(f"/api/v1/documents/{doc_id}")
    assert del_resp.status_code == 204

    assert marker not in _collect_retained_plaintext(state, document_id=doc_id)

    # And no record of the analysis survives it either.
    assert client.get(f"/api/v1/runs/{run_id}").status_code == 404
    assert state.services.metadata.get_run(UUID(run_id)) is None


def test_expiry_purges_without_lazy_touch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """TTL expiry must purge sessions and runs without a content GET touch."""
    marker = "EXPIRY_PURGE_CANARY_4b8d1e6f"
    now = {"value": 0.0}
    text_store = RunTextStore(
        RunConfig(content_ttl_seconds=10),
        clock=lambda: now["value"],
    )
    settings = Settings(
        model_api_key="test-secret-key",
        model_base_url="https://openrouter.ai/api/v1",
        model_name="deepseek-v4-flash",
    )
    services = AnalysisServices(
        settings=settings,
        corpus=_mock_corpus(),
        client=mock_model_client(_make_transport(), settings=settings),
        metadata=MetadataStore(tmp_path / "expiry.sqlite3"),
        prompt_bundle=load_prompt_bundle(),
        text_store=text_store,
        events=EventBus(),
        run_config=RunConfig(content_ttl_seconds=10),
    )
    app = create_app(settings, services)
    with TestClient(app, raise_server_exceptions=False) as client:
        doc = _upload_txt(client, f"{marker}\n{DOCUMENT_TEXT}")
        doc_id = UUID(doc["id"])
        state: _AppState = client.app.state.app_state
        assert marker in _collect_retained_plaintext(state, document_id=doc_id)

        now["value"] = 11.0
        from contract_analyzer.api.state import _purge_expired_content

        _purge_expired_content(state)

        assert marker not in _collect_retained_plaintext(state, document_id=doc_id)


def test_expired_text_before_the_sweep_returns_410(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A live session whose text has expired answers 410, not 404.

    The two codes divide the route: 404 is a session that is gone, 410 a session
    that survives its text. Only the gap between the TTL elapsing and the purge
    sweep running reaches the second, because every sweep drops the session with
    the text and would answer 404 instead.
    """
    now = {"value": 0.0}
    text_store = RunTextStore(
        RunConfig(content_ttl_seconds=10),
        clock=lambda: now["value"],
    )
    settings = Settings(
        model_api_key="test-secret-key",
        model_base_url="https://openrouter.ai/api/v1",
        model_name="deepseek-v4-flash",
    )
    services = AnalysisServices(
        settings=settings,
        corpus=_mock_corpus(),
        client=mock_model_client(_make_transport(), settings=settings),
        metadata=MetadataStore(tmp_path / "expired-410.sqlite3"),
        prompt_bundle=load_prompt_bundle(),
        text_store=text_store,
        events=EventBus(),
        run_config=RunConfig(content_ttl_seconds=10),
    )
    app = create_app(settings, services)
    with TestClient(app, raise_server_exceptions=False) as client:
        doc = _upload_txt(client)
        doc_id = doc["id"]
        assert client.get(f"/api/v1/documents/{doc_id}/content").status_code == 200

        # The TTL elapses and nothing sweeps: the session is still registered,
        # so the route gets past its 404 guard and asks whether the text is there.
        now["value"] = 11.0
        state: _AppState = client.app.state.app_state
        assert UUID(doc_id) in state.services.sessions

        resp = client.get(f"/api/v1/documents/{doc_id}/content")
        assert resp.status_code == 410, resp.text


def test_shutdown_purges_plaintext(
    _services: AnalysisServices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Process shutdown must purge every retained copy of document plaintext."""
    marker = "SHUTDOWN_PURGE_CANARY_9e2c5a8b"
    settings = _services.settings
    app = create_app(settings, _services)
    release = threading.Event()
    original_execute, started, _ = _blocking_execute_patch(release)
    run_id: str | None = None
    doc_id: UUID | None = None
    state: _AppState | None = None
    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            doc = _upload_txt(client, f"{marker}\n{DOCUMENT_TEXT}")
            doc_id = UUID(doc["id"])
            state = client.app.state.app_state
            run_resp = client.post(
                "/api/v1/runs",
                json={"document_id": doc["id"], "arm": "mid"},
            )
            assert run_resp.status_code == 202
            run_id = run_resp.json()["id"]
            assert started.wait(timeout=5.0)
            assert marker in _collect_retained_plaintext(state, document_id=doc_id)
    finally:
        release.set()
        AnalysisRunner.execute_run = original_execute  # type: ignore[method-assign]

    assert state is not None
    assert doc_id is not None
    assert run_id is not None
    assert marker not in _collect_retained_plaintext(state, document_id=doc_id)

    record = _services.metadata.get_run(UUID(run_id))
    assert record is not None
    assert record.status in ("failed", "cancelled", "completed")


def test_unsupported_extension_returns_415(client: TestClient) -> None:
    resp = client.post(
        "/api/v1/documents",
        files={"file": ("document.xyz", b"hello", "application/octet-stream")},
    )
    assert resp.status_code == 415
    assert resp.json()["code"] == "unsupported_input"


def test_empty_file_returns_422(client: TestClient) -> None:
    resp = client.post(
        "/api/v1/documents",
        files={"file": ("empty.txt", b"", "text/plain")},
    )
    assert resp.status_code == 422
    assert resp.json()["code"] == "empty_input"


def test_nonexistent_document_returns_404(client: TestClient) -> None:
    fake_id = str(uuid4())
    resp = client.get(f"/api/v1/documents/{fake_id}/content")
    assert resp.status_code == 404


def test_nonexistent_run_returns_404(client: TestClient) -> None:
    fake_id = str(uuid4())
    resp = client.get(f"/api/v1/runs/{fake_id}")
    assert resp.status_code == 404


def test_cancel_terminal_run_returns_409(client: TestClient) -> None:
    doc = _upload_txt(client)
    run_resp = client.post(
        "/api/v1/runs",
        json={"document_id": doc["id"], "arm": "mid"},
    )
    assert run_resp.status_code == 202
    run_id = run_resp.json()["id"]
    _wait_for_terminal_run(client, run_id)
    cancel_resp = client.post(f"/api/v1/runs/{run_id}/cancel")
    assert cancel_resp.status_code == 409
    assert cancel_resp.json()["code"] == "run_already_terminal"


def test_create_run_returns_running_before_analysis_finishes(
    client: TestClient,
) -> None:
    doc = _upload_txt(client)
    state: _AppState = client.app.state.app_state
    release = threading.Event()
    original_execute, started, _ = _blocking_execute_patch(release)
    run_id: str | None = None
    try:
        run_resp = client.post(
            "/api/v1/runs",
            json={"document_id": doc["id"], "arm": "mid"},
        )
        assert run_resp.status_code == 202, run_resp.text
        run = run_resp.json()
        run_id = run["id"]
        assert run["status"] == "running"
        assert run["metrics"] is None

        assert started.wait(timeout=5.0), "background run did not start"
        assert UUID(run_id) in state.tasks.active_tasks

        get_resp = client.get(f"/api/v1/runs/{run_id}")
        assert get_resp.status_code == 200
        assert get_resp.json()["status"] == "running"
    finally:
        release.set()
        AnalysisRunner.execute_run = original_execute  # type: ignore[method-assign]
        if run_id is not None:
            _wait_for_terminal_run(client, run_id)


def test_active_task_cleared_after_run_completes(client: TestClient) -> None:
    doc = _upload_txt(client)
    state: _AppState = client.app.state.app_state
    release = threading.Event()
    original_execute, started, _ = _blocking_execute_patch(release)
    run_id: str | None = None
    try:
        run_resp = client.post(
            "/api/v1/runs",
            json={"document_id": doc["id"], "arm": "mid"},
        )
        assert run_resp.status_code == 202
        run_id = run_resp.json()["id"]
        assert started.wait(timeout=5.0)
        assert UUID(run_id) in state.tasks.active_tasks
    finally:
        release.set()
        AnalysisRunner.execute_run = original_execute  # type: ignore[method-assign]
        if run_id is not None:
            _wait_for_terminal_run(client, run_id)
            _wait_until(lambda: UUID(run_id) not in state.tasks.active_tasks)


def test_a_second_run_starts_beside_the_first(client: TestClient) -> None:
    """Two windows analysing two documents at once is the case this permits.

    Two documents, not one: a second run on the same document shares that
    document's session object, and every purge path reaches all of them at once,
    so the same-document case is a limitation rather than the behaviour under
    test.
    """
    first_doc = _upload_txt(client)
    second_doc = _upload_txt(client, f"INNY DOKUMENT\n{DOCUMENT_TEXT}")
    assert first_doc["id"] != second_doc["id"]
    state: _AppState = client.app.state.app_state
    release = threading.Event()
    original_execute, started, _ = _blocking_execute_patch(release)
    run_ids: list[str] = []
    try:
        first = client.post(
            "/api/v1/runs",
            json={"document_id": first_doc["id"], "arm": "mid"},
        )
        assert first.status_code == 202, first.text
        run_ids.append(first.json()["id"])
        assert started.wait(timeout=5.0)

        second = client.post(
            "/api/v1/runs",
            json={"document_id": second_doc["id"], "arm": "off"},
        )
        assert second.status_code == 202, second.text
        assert second.json()["status"] == "running"
        run_ids.append(second.json()["id"])

        assert {UUID(value) for value in run_ids} <= set(state.tasks.active_tasks)
    finally:
        release.set()
        AnalysisRunner.execute_run = original_execute  # type: ignore[method-assign]

    finished = [_wait_for_terminal_run(client, value) for value in run_ids]
    assert [run["status"] for run in finished] == ["completed", "completed"]
    assert [run["document_id"] for run in finished] == [
        first_doc["id"],
        second_doc["id"],
    ]
    assert finished[0]["findings"] and finished[1]["findings"]
    assert not state.tasks.active_tasks


def test_a_run_refused_for_the_slot_never_reaches_the_corpus(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The corpus preflight is a network call, so the refusal must precede it."""
    doc = _upload_txt(client)
    state: _AppState = client.app.state.app_state
    state.settings = replace(state.settings, evaluation_batch_open=True)
    entered: list[str] = []
    original_preflight = run_endpoints._ensure_corpus_available

    def watched(app_state: object) -> None:
        entered.append("preflight")
        original_preflight(app_state)

    monkeypatch.setattr(run_endpoints, "_ensure_corpus_available", watched)

    release = threading.Event()
    original_execute, started, _ = _blocking_execute_patch(release)
    first_run_id: str | None = None
    try:
        first = client.post(
            "/api/v1/runs",
            json={"document_id": doc["id"], "arm": "mid"},
        )
        assert first.status_code == 202, first.text
        first_run_id = first.json()["id"]
        assert started.wait(timeout=5.0)
        assert entered == ["preflight"]

        refused = client.post(
            "/api/v1/runs",
            json={"document_id": doc["id"], "arm": "off"},
        )
        assert refused.status_code == 409
        assert entered == ["preflight"]
    finally:
        release.set()
        AnalysisRunner.execute_run = original_execute  # type: ignore[method-assign]
        if first_run_id is not None:
            _wait_for_terminal_run(client, first_run_id)


def test_a_sealed_evaluation_batch_admits_one_run(client: TestClient) -> None:
    """The slot exists for the measured batch: a second run would spend its clock."""
    doc = _upload_txt(client)
    state: _AppState = client.app.state.app_state
    state.settings = replace(state.settings, evaluation_batch_open=True)
    release = threading.Event()
    original_execute, started, _ = _blocking_execute_patch(release)
    first_run_id: str | None = None
    try:
        first = client.post(
            "/api/v1/runs",
            json={"document_id": doc["id"], "arm": "mid"},
        )
        assert first.status_code == 202, first.text
        assert first.json()["status"] == "running"
        first_run_id = first.json()["id"]
        assert started.wait(timeout=5.0)

        second = client.post(
            "/api/v1/runs",
            json={"document_id": doc["id"], "arm": "off"},
        )
        assert second.status_code == 409
        assert second.json()["code"] == "run_already_active"
    finally:
        release.set()
        AnalysisRunner.execute_run = original_execute  # type: ignore[method-assign]
        if first_run_id is not None:
            _wait_for_terminal_run(client, first_run_id)


def test_cancel_in_flight_run_stops_analysis(client: TestClient) -> None:
    doc = _upload_txt(client)
    state: _AppState = client.app.state.app_state
    release = threading.Event()
    original_execute, started, original_entered = _blocking_execute_patch(release)
    run_id: str | None = None
    try:
        run_resp = client.post(
            "/api/v1/runs",
            json={"document_id": doc["id"], "arm": "mid"},
        )
        assert run_resp.status_code == 202, run_resp.text
        run_id = run_resp.json()["id"]
        assert started.wait(timeout=5.0)

        cancel_resp = client.post(f"/api/v1/runs/{run_id}/cancel")
        assert cancel_resp.status_code == 202, cancel_resp.text
        cancelled = cancel_resp.json()
        assert cancelled["status"] == "cancelled"
        assert cancelled["metrics"] is not None
        assert UUID(run_id) not in state.tasks.active_tasks

        release.set()
        assert not original_entered.wait(timeout=1.0)

        get_resp = client.get(f"/api/v1/runs/{run_id}")
        assert get_resp.status_code == 200
        assert get_resp.json()["status"] == "cancelled"
    finally:
        release.set()
        AnalysisRunner.execute_run = original_execute  # type: ignore[method-assign]


def test_run_events_deliver_counter_while_work_continues(
    client: TestClient,
) -> None:
    doc = _upload_txt(client)
    release = threading.Event()
    original_execute, started, _ = _blocking_execute_patch(release)
    run_id: str | None = None
    counter_seen = threading.Event()
    collected: list[dict[str, Any]] = []
    reader_error: list[BaseException] = []

    def read_events() -> None:
        try:
            buffer = ""
            with client.stream("GET", f"/api/v1/runs/{run_id}/events") as response:
                assert response.status_code == 200
                for chunk in response.iter_text():
                    buffer += chunk
                    while "\n\n" in buffer:
                        block, buffer = buffer.split("\n\n", 1)
                        for line in block.splitlines():
                            if not line.startswith("data:"):
                                continue
                            payload = json.loads(line.removeprefix("data:").strip())
                            collected.append(payload)
                            if (
                                payload.get("kind") == "counter"
                                and payload.get("counter_name") == "processed_units"
                            ):
                                counter_seen.set()
                                return
        except BaseException as exc:
            reader_error.append(exc)
            raise

    try:
        run_resp = client.post(
            "/api/v1/runs",
            json={"document_id": doc["id"], "arm": "mid"},
        )
        assert run_resp.status_code == 202, run_resp.text
        run = run_resp.json()
        assert run["status"] == "running"
        run_id = run["id"]
        assert started.wait(timeout=5.0)

        reader = threading.Thread(target=read_events)
        reader.start()
        release.set()

        assert counter_seen.wait(timeout=30.0), (
            f"expected processed_units counter on SSE stream, got: {collected}"
        )
        reader.join(timeout=5.0)
        assert not reader.is_alive()
        if reader_error:
            raise reader_error[0]
        _wait_for_terminal_run(client, run_id)
    finally:
        release.set()
        AnalysisRunner.execute_run = original_execute  # type: ignore[method-assign]


def test_health_live(client: TestClient) -> None:
    resp = client.get("/api/v1/health/live")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_health_ready(client: TestClient) -> None:
    resp = client.get("/api/v1/health/ready")
    # May be 200 or 503 depending on environment
    assert resp.status_code in (200, 503)
    data = resp.json()
    assert "ready" in data
    assert "dependencies" in data
    assert isinstance(data["dependencies"], list)
    for dep in data["dependencies"]:
        assert "name" in dep
        assert "ok" in dep


def _drop_a_unit(corpus: QdrantCorpusIndex) -> None:
    """Make the published collection disagree with the snapshot beside it.

    The snapshot declares a unit count; deleting a point leaves the corpus
    incomplete, which is the shape of corruption verification exists to catch.
    """
    points, _ = corpus._client.scroll(collection_name=corpus.collection, limit=1)
    corpus._client.delete(
        collection_name=corpus.collection,
        points_selector=models.PointIdsList(points=[points[0].id]),
        wait=True,
    )


def test_readiness_reports_bad_corpus_detail(
    real_corpus_client: TestClient,
    built_corpus: QdrantCorpusIndex,
) -> None:
    _drop_a_unit(built_corpus)
    resp = real_corpus_client.get("/api/v1/health/ready")
    assert resp.status_code == 503
    corpus = next(dep for dep in resp.json()["dependencies"] if dep["name"] == "corpus")
    assert corpus["ok"] is False
    assert corpus["detail"] is not None
    assert "corpus_incomplete" in corpus["detail"]


def test_readiness_reports_missing_model_credential(
    _services: AnalysisServices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The credential check must tell a configured key from none at all."""
    app = create_app(Settings(model_api_key=None), _services)
    with TestClient(app, raise_server_exceptions=False) as tc:
        resp = tc.get("/api/v1/health/ready")
    assert resp.status_code == 503
    credential = next(
        dep for dep in resp.json()["dependencies"] if dep["name"] == "model_credential"
    )
    assert credential["ok"] is False
    assert credential["detail"] == "API key not configured"


def test_run_refuses_bad_corpus(
    real_corpus_client: TestClient,
    built_corpus: QdrantCorpusIndex,
    run_invoked: list[bool],
) -> None:
    _drop_a_unit(built_corpus)
    doc = _upload_txt(real_corpus_client)
    run_resp = real_corpus_client.post(
        "/api/v1/runs",
        json={"document_id": doc["id"], "arm": "mid"},
    )
    assert run_resp.status_code == 503
    assert run_resp.json()["code"] == "dependency_unavailable"
    assert run_invoked == []


def test_run_starts_with_valid_corpus(
    real_corpus_client: TestClient,
    run_invoked: list[bool],
) -> None:
    doc = _upload_txt(real_corpus_client)
    run_resp = real_corpus_client.post(
        "/api/v1/runs",
        json={"document_id": doc["id"], "arm": "mid"},
    )
    assert run_resp.status_code == 202, run_resp.text
    assert run_invoked == [True]


def test_config_endpoint(client: TestClient) -> None:
    resp = client.get("/api/v1/config")
    assert resp.status_code == 200
    data = resp.json()
    assert data["model_request_id"] == "deepseek-v4-flash"
    assert "prompt_bundle_version" in data
    assert "corpus_snapshot" in data
    assert data["embedding_model"] == "qwen/qwen3-embedding-8b"
    assert isinstance(data["arms"], list)
    assert set(data["arms"]) == {"off", "mid", "on"}
    assert "limits" in data
    limits = data["limits"]
    assert "max_input_bytes" in limits
    assert "max_pdf_pages" in limits
    assert "content_ttl_seconds" in limits


class _LogCapture(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(self.format(record))


def test_privacy_canary_http(client: TestClient) -> None:
    """Document text must not appear in logs, error bodies, or SSE events."""
    canary = "CANARY_PRYWATNOSC_abc123_tajne"
    doc_text = (
        f"\u00a7 1. {canary} zobowi\u0105zuje si\u0119.\n\n"
        "\u00a7 2. Drugie postanowienie umowy.\n\n"
        "\u00a7 3. Trzecie postanowienie umowy.\n"
    )

    handler = _LogCapture()
    logging.root.addHandler(handler)
    try:
        doc = _upload_txt(client, doc_text)
        doc_id = doc["id"]

        run_resp = client.post(
            "/api/v1/runs",
            json={"document_id": doc_id, "arm": "mid"},
        )
        assert run_resp.status_code == 202
        run_id = run_resp.json()["id"]

        assert canary not in run_resp.text

        run_data = _wait_for_terminal_run(client, run_id)
        assert canary not in json.dumps(run_data)

        err_resp = client.get(f"/api/v1/documents/{uuid4()}/content")
        assert canary not in err_resp.text

        for log_line in handler.records:
            assert canary not in log_line, f"Text leaked in log: {log_line[:200]}"

    finally:
        logging.root.removeHandler(handler)


def test_app_starts_without_web_dist(client: TestClient) -> None:
    """App must start even when web/dist does not exist."""
    # client fixture already starts the app; if we got here, it worked
    resp = client.get("/api/v1/health/live")
    assert resp.status_code == 200


def test_units_total_counts_distinct_unit_ids_not_findings() -> None:
    run = RunRecord(
        id=uuid4(),
        document_id=uuid4(),
        arm=ArmCode.MID,
        input_hash="input-sha256",
        config_version="config-v1",
        prompt_bundle_version="prompts-v1",
        corpus_snapshot_id="corpus-v1",
        tool_bundle_version="tools-v1",
        requested_model="deepseek-v4-flash",
        parameters={"temperature": 0.0},
        retry_policy="bounded-3-v1",
        concurrency=1,
        wall_budget_seconds=300.0,
        measurement_valid=True,
        cost=CostRecord.unknown("no_auditable_price_table"),
        status="completed",
        elapsed_ms=10.0,
    )
    unit_id = "unit-1"
    findings = [
        FindingRecord(
            id=uuid4(),
            run_id=run.id,
            unit_id=unit_id,
            code=FindingCode.CONTRADICTORY,
        ),
        FindingRecord(
            id=uuid4(),
            run_id=run.id,
            unit_id=unit_id,
            code=FindingCode.PERMISSIBLE_DEPARTURE,
        ),
    ]
    metrics = _build_metrics(findings, [], run)
    assert metrics.units_total == 1
    assert metrics.units_with_finding == 1
    assert metrics.units_not_processed == 0


class _InflatedLatencyClient:
    _LATENCY_MS = 100.0
    max_attempts = 3

    async def converse(self, conversation: Any) -> Any:
        from contract_analyzer.model import ToolInvocation, ToolTurn

        messages = conversation.messages
        if any(m.get("role") == "tool" for m in messages):
            tool_calls: tuple[ToolInvocation, ...] = ()
        else:
            user_envelope = json.loads(cast(str, messages[1]["content"]))
            unit_text = (
                user_envelope["payload"]["unit_text"]
                if isinstance(user_envelope.get("payload"), dict)
                and "unit_text" in user_envelope["payload"]
                else user_envelope.get("unit_text", "")
            )
            phrase = str(unit_text)[:60]
            tool_calls = (
                ToolInvocation(
                    id="call_1",
                    name="search_corpus",
                    arguments=json.dumps({"phrase": phrase}),
                ),
            )
        attempts = tuple(
            AttemptTelemetry(
                requested_model="deepseek-v4-flash",
                returned_model="deepseek-v4-flash-20260828",
                prompt_version=conversation.prompt_version,
                temperature=conversation.temperature,
                parameters=dict(conversation.parameters),
                input_tokens=10,
                output_tokens=2,
                latency_ms=self._LATENCY_MS,
                status="success",
                retry_number=0,
                error_code=None,
                prompt_hash="prompt-hash",
                response_hash="response-hash",
            )
            for _ in range(4)
        )
        return ToolTurn(
            content="",
            tool_calls=tool_calls,
            requested_model="deepseek-v4-flash",
            returned_model="deepseek-v4-flash-20260828",
            input_tokens=40,
            output_tokens=8,
            attempts=attempts,
        )

    async def aclose(self) -> None:
        return None


def test_elapsed_ms_reports_wall_clock_not_attempt_latency_sum(
    _real_corpus_services: AnalysisServices,
    run_invoked: list[bool],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hashlib

    from contract_analyzer.agents.session import DocumentSession
    from contract_analyzer.config import RunConfig
    from contract_analyzer.domain import (
        ArmCode,
        DocumentPayload,
        ReadMode,
        SourceAnchor,
    )
    from contract_analyzer.structure import parse_references, segment

    class FakeClock:
        def __call__(self) -> float:
            return 0.0

    text = "§ 1. Najemca składa kaucję zabezpieczającą.\n"
    payload = DocumentPayload(
        document_id=uuid4(),
        text=text,
        anchors=(SourceAnchor(start_offset=0, end_offset=len(text)),),
        read_mode=ReadMode.NATIVE_PDF,
        content_hash=hashlib.sha256(text.encode()).hexdigest(),
    )
    units = segment(payload, RunConfig())
    session = DocumentSession(
        document_id=payload.document_id,
        payload=payload,
        units=tuple(units),
        references=tuple(parse_references(units)),
    )
    _real_corpus_services.sessions[session.document_id] = session
    _real_corpus_services.client = cast(Any, _InflatedLatencyClient())  # type: ignore[assignment]

    async def run_once() -> None:
        runner = AnalysisRunner(_real_corpus_services)
        result = await runner.run(
            RunRequest(
                document_id=session.document_id,
                arm=ArmCode.MID,
                measurement_valid=True,
                wall_budget_seconds=300.0,
                concurrency=4,
                clock=FakeClock(),
            )
        )
        record = _real_corpus_services.metadata.get_run(result.run_id)
        assert record is not None
        attempts = _real_corpus_services.metadata.list_attempts(result.run_id)
        findings = _real_corpus_services.metadata.list_findings(result.run_id)
        metrics = _build_metrics(findings, attempts, record)
        latency_sum = sum(attempt.latency_ms for attempt in attempts)
        assert latency_sum >= 400.0
        assert metrics.elapsed_ms < latency_sum / 2

    asyncio.run(run_once())


def test_cancelled_run_reports_nonzero_elapsed(
    built_corpus: QdrantCorpusIndex,
    tmp_path: Path,
) -> None:
    class FakeClock:
        def __init__(self, start: float = 1000.0) -> None:
            self._now = start

        def __call__(self) -> float:
            return self._now

        def advance(self, seconds: float) -> None:
            self._now += seconds

    async def exercise() -> None:
        import hashlib
        from unittest.mock import MagicMock

        from contract_analyzer.agents.prompts import load_prompt_bundle
        from contract_analyzer.agents.runner import AnalysisRunner
        from contract_analyzer.agents.services import AnalysisServices
        from contract_analyzer.agents.session import DocumentSession, RunRequest
        from contract_analyzer.api import cancel_run
        from contract_analyzer.api.state import _AppState
        from contract_analyzer.config import RunConfig, Settings
        from contract_analyzer.domain import (
            DocumentPayload,
            ReadMode,
            SourceAnchor,
        )
        from contract_analyzer.storage import MetadataStore
        from contract_analyzer.structure import parse_references, segment

        clock = FakeClock()
        release = asyncio.Event()

        class BlockingClient:
            async def converse(self, conversation: Any) -> Any:
                await release.wait()
                raise AssertionError("run should be cancelled before completing")

            async def aclose(self) -> None:
                return None

        text = "§ 1. Najemca składa kaucję zabezpieczającą.\n"
        payload = DocumentPayload(
            document_id=uuid4(),
            text=text,
            anchors=(SourceAnchor(start_offset=0, end_offset=len(text)),),
            read_mode=ReadMode.NATIVE_PDF,
            content_hash=hashlib.sha256(text.encode()).hexdigest(),
        )
        units = segment(payload, RunConfig())
        session = DocumentSession(
            document_id=payload.document_id,
            payload=payload,
            units=tuple(units),
            references=tuple(parse_references(units)),
        )

        settings = Settings(
            model_api_key="test-secret-key",
            model_base_url="https://openrouter.ai/api/v1",
            model_name="deepseek-v4-flash",
        )
        metadata = MetadataStore(tmp_path / "cancel-elapsed.sqlite3", clock=clock)
        services = AnalysisServices(
            settings=settings,
            corpus=built_corpus,
            client=cast(Any, BlockingClient()),
            metadata=metadata,
            prompt_bundle=load_prompt_bundle(),
        )
        services.sessions[session.document_id] = session
        runner = AnalysisRunner(services)
        state = _AppState(
            services=services,
            runner=runner,
            ingest_service=MagicMock(),
            settings=settings,
        )

        run_task = asyncio.create_task(
            runner.run(
                RunRequest(
                    document_id=session.document_id,
                    arm=ArmCode.MID,
                    measurement_valid=True,
                    wall_budget_seconds=300.0,
                    concurrency=1,
                    clock=clock,
                )
            )
        )

        run_id: UUID | None = None
        for _ in range(200):
            running = [r for r in metadata.list_runs() if r.status == "running"]
            if running:
                run_id = running[0].id
                break
            await asyncio.sleep(0)
        assert run_id is not None

        state.tasks.active_tasks[run_id] = run_task
        clock.advance(2.5)

        request = MagicMock()
        request.app.state.app_state = state
        cancelled = await cancel_run(run_id, request)

        assert cancelled.status == "cancelled"
        assert cancelled.metrics is not None
        assert cancelled.metrics.elapsed_ms > 0.0
        assert cancelled.metrics.elapsed_ms >= 2500.0

    asyncio.run(exercise())


def test_settle_run_refuses_cancelled_run_without_monotonic_start(
    tmp_path: Path,
) -> None:
    """A cancelled run without a monotonic start raises RuntimeError."""
    from unittest.mock import MagicMock

    from contract_analyzer.api.runs import _settle_run
    from contract_analyzer.api.state import _AppState
    from contract_analyzer.domain import ArmCode
    from contract_analyzer.storage import CostRecord, MetadataStore, RunRecord

    store = MetadataStore(tmp_path / "cancel-no-start.sqlite3")
    run = RunRecord(
        id=uuid4(),
        document_id=uuid4(),
        arm=ArmCode.MID,
        input_hash="input-sha256",
        config_version="config-v1",
        prompt_bundle_version="prompts-v1",
        corpus_snapshot_id="corpus-v1",
        tool_bundle_version="tools-v1",
        requested_model="deepseek-v4-flash",
        parameters={},
        retry_policy="bounded-3-v1",
        concurrency=1,
        wall_budget_seconds=60.0,
        measurement_valid=True,
        cost=CostRecord.unknown("no_price_table"),
    )
    store.create_run(run)

    state = _AppState(
        services=MagicMock(metadata=store, events=None),
        runner=MagicMock(),
        ingest_service=MagicMock(),
        settings=MagicMock(),
    )

    with pytest.raises(RuntimeError) as exc_info:
        _settle_run(state, run.id, "cancelled")

    assert str(run.id) in str(exc_info.value)
    stored = store.get_run(run.id)
    assert stored is not None
    assert stored.status == "running"
    assert stored.elapsed_ms is None
    assert stored.elapsed_ms != 0.0


def test_process_restarted_run_omits_metrics_without_elapsed(
    tmp_path: Path,
) -> None:
    from unittest.mock import MagicMock

    from contract_analyzer.api.run_builders import _build_run
    from contract_analyzer.api.state import _AppState
    from contract_analyzer.domain import ArmCode
    from contract_analyzer.storage import CostRecord, MetadataStore, RunRecord

    store = MetadataStore(tmp_path / "restarted.sqlite3")
    run = RunRecord(
        id=uuid4(),
        document_id=uuid4(),
        arm=ArmCode.MID,
        input_hash="input-sha256",
        config_version="config-v1",
        prompt_bundle_version="prompts-v1",
        corpus_snapshot_id="corpus-v1",
        tool_bundle_version="tools-v1",
        requested_model="deepseek-v4-flash",
        parameters={"temperature": 0.0},
        retry_policy="bounded-3-v1",
        concurrency=2,
        wall_budget_seconds=300.0,
        measurement_valid=True,
        cost=CostRecord.unknown("no_auditable_price_table"),
    )
    store.create_run(run)

    store.mark_interrupted_runs_failed()

    record = store.get_run(run.id)
    assert record is not None
    assert record.status == "failed"
    assert record.error_code == "process_restarted"
    assert record.elapsed_ms is None

    state = _AppState(
        services=MagicMock(metadata=store),
        runner=MagicMock(),
        ingest_service=MagicMock(),
        settings=MagicMock(),
    )
    built = _build_run(record, state)
    assert built.metrics is None


def test_build_run_uses_persisted_interruption_without_not_processed_findings(
    tmp_path: Path,
) -> None:
    from unittest.mock import MagicMock

    from contract_analyzer.api.run_builders import _build_run
    from contract_analyzer.api.state import _AppState

    store = MetadataStore(tmp_path / "interrupted.sqlite3")
    run = RunRecord(
        id=uuid4(),
        document_id=uuid4(),
        arm=ArmCode.MID,
        input_hash="input-sha256",
        config_version="config-v1",
        prompt_bundle_version="prompts-v1",
        corpus_snapshot_id="corpus-v1",
        tool_bundle_version="tools-v1",
        requested_model="deepseek-v4-flash",
        parameters={"temperature": 0.0},
        retry_policy="bounded-3-v1",
        concurrency=1,
        wall_budget_seconds=300.0,
        measurement_valid=True,
        cost=CostRecord.unknown("no_auditable_price_table"),
    )
    store.create_run(run)
    store.finish_run(
        run.id,
        "completed",
        totals=RunTotals(elapsed_ms=300_000.0, interruption_reason="wall_time"),
    )
    record = store.get_run(run.id)
    assert record is not None
    state = _AppState(
        services=MagicMock(metadata=store),
        runner=MagicMock(),
        ingest_service=MagicMock(),
        settings=MagicMock(),
    )

    built = _build_run(record, state)

    assert built.interrupted is True
    assert built.interruption_reason == "wall_time"
    assert built.metrics is not None
    assert built.metrics.units_not_processed == 0


def test_unresolved_finding_omits_anchor_from_serialised_json() -> None:
    from contract_analyzer.api.run_builders import _build_finding

    record = FindingRecord(
        id=uuid4(),
        run_id=uuid4(),
        unit_id="whole:deadbeef",
        code=FindingCode.NO_BASIS_FOUND,
    )
    payload = _build_finding(record).model_dump(mode="json")
    assert payload["anchor_resolved"] is False
    assert "anchor" not in payload
    assert "raw_confidence" in payload
    assert payload["raw_confidence"] is None


def test_resolved_finding_includes_anchor_in_serialised_json() -> None:
    from contract_analyzer.api.run_builders import _build_finding

    record = FindingRecord(
        id=uuid4(),
        run_id=uuid4(),
        unit_id="unit-1",
        code=FindingCode.CONSISTENT,
        start_offset=12,
        end_offset=34,
    )
    payload = _build_finding(record).model_dump(mode="json")
    assert payload["anchor_resolved"] is True
    assert payload["anchor"]["start_offset"] == 12
    assert payload["anchor"]["end_offset"] == 34


def test_off_run_wire_json_omits_anchor(client: TestClient) -> None:
    doc = _upload_txt(client)
    run_resp = client.post(
        "/api/v1/runs",
        json={"document_id": doc["id"], "arm": "off"},
    )
    assert run_resp.status_code == 202, run_resp.text
    run_id = run_resp.json()["id"]
    run_data = _wait_for_terminal_run(client, run_id)
    finding = run_data["findings"][0]
    assert finding["anchor_resolved"] is False
    assert "anchor" not in finding
    assert "raw_confidence" in finding


def _admitted_basis() -> EmittedBasis:
    return EmittedBasis(
        provision_locator="https://api.sejm.gov.pl/eli/acts/DU/2026/795/art/659",
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
            source_locator="https://api.sejm.gov.pl/eli/acts/DU/2026/795/art/659",
        ),
        character_kind=ProvisionKind.IMPERATIVE,
    )


def test_admitted_finding_carries_both_force_records_to_the_api_result() -> None:
    """RF-05 v1.4: the provision record reaches the result, not just the act one.

    Before this, ``_build_finding`` never populated ``basis`` on any path, so the
    field was dead output exercised only by web test fixtures.
    """
    from contract_analyzer.api.run_builders import _build_finding

    record = FindingRecord(
        id=uuid4(),
        run_id=uuid4(),
        unit_id="unit-1",
        code=FindingCode.CONSISTENT,
        start_offset=12,
        end_offset=48,
        legal_locators=("https://api.sejm.gov.pl/eli/acts/DU/2026/795/art/659",),
        basis=_admitted_basis(),
    )
    payload = _build_finding(record).model_dump(mode="json")

    assert payload["basis"]["act_force"] == {
        "value": "in_force",
        "scope": "act",
        "snapshot_date": "2026-05-19",
        "source_locator": "https://api.sejm.gov.pl/eli/acts/DU/2026/795",
    }
    assert payload["basis"]["provision_force"] == {
        "value": "undetermined",
        "scope": "provision",
        "snapshot_date": "2026-05-19",
        "source_locator": "https://api.sejm.gov.pl/eli/acts/DU/2026/795/art/659",
    }
    assert payload["basis"]["character_kind"] == "imperative"


def test_vetoed_candidate_is_never_presented_as_a_legal_basis() -> None:
    """RF-05 v1.4 forbids presenting a rejected retrieval candidate as a basis.

    The veto reaches the consumer as the finding code and its locator, which is
    why ``basis`` stays absent here rather than carrying the blocking record.
    """
    from contract_analyzer.api.run_builders import _build_finding

    record = FindingRecord(
        id=uuid4(),
        run_id=uuid4(),
        unit_id="unit-2",
        code=FindingCode.BASIS_NOT_IN_FORCE,
        start_offset=12,
        end_offset=48,
        legal_locators=("https://api.sejm.gov.pl/eli/acts/DU/2026/795/art/420",),
    )
    payload = _build_finding(record).model_dump(mode="json")

    assert payload["basis"] is None
    assert payload["code"] == "basis_not_in_force"
    assert payload["legal_locators"] == [
        "https://api.sejm.gov.pl/eli/acts/DU/2026/795/art/420"
    ]


@pytest.mark.parametrize("code", list(FindingCode))
def test_every_finding_code_has_its_exact_prominence(code: FindingCode) -> None:
    """The severity a user sees is derived from the code, so pin the whole mapping.

    An earlier test only asserted membership in the Prominence enum, so flattening
    the map -- rendering a contradictory clause as neutral -- passed the suite.
    The else branch makes a new FindingCode fail here until its severity is chosen.
    """
    if code in (FindingCode.CONTRADICTORY, FindingCode.BASIS_NOT_IN_FORCE):
        expected = Prominence.CRITICAL
    elif code in (
        FindingCode.PERMISSIBLE_DEPARTURE,
        FindingCode.UNCERTAIN,
        FindingCode.UNIT_NOT_ADJUDICABLE,
        FindingCode.NOT_PROCESSED,
    ):
        expected = Prominence.WARNING
    elif code in (
        FindingCode.CONSISTENT,
        FindingCode.NO_RELATION,
        FindingCode.NO_BASIS_FOUND,
    ):
        expected = Prominence.NEUTRAL
    else:
        raise AssertionError(f"unhandled finding code: {code}")

    assert _PROMINENCE[code] is expected


def test_unhandled_exception_is_internal_error_not_a_dependency_outage(
    client: TestClient,
) -> None:
    """A defect in this server must not be reported as an external outage."""
    state: _AppState = client.app.state.app_state

    def boom(_run_id: object) -> None:
        raise RuntimeError("induced defect")

    original = state.services.metadata.get_run
    state.services.metadata.get_run = boom  # type: ignore[method-assign]
    try:
        response = client.get(
            "/api/v1/runs/00000000-0000-0000-0000-000000000009",
            headers={"accept": "application/json"},
        )
    finally:
        state.services.metadata.get_run = original  # type: ignore[method-assign]

    assert response.status_code == 500, response.text
    body = response.json()
    assert body["code"] == "internal_error"
    assert body["code"] != "dependency_unavailable"
    # The traceback belongs in the log, never the response body.
    assert "induced defect" not in response.text
    assert "Traceback" not in response.text


def test_a_file_at_the_size_limit_is_still_accepted(client: TestClient) -> None:
    """The pre-check must not reject a legitimate file sitting on the bound.

    Content-Length covers the whole multipart envelope, so comparing it directly
    against max_input_bytes rejected a file that was exactly at the limit.
    """
    payload = "§ 1. " + "a" * 200 + "\n\n§ 2. " + "b" * 200 + "\n\n§ 3. koniec.\n"
    data = payload.encode("utf-8")
    small_limit = len(data)
    client.app.state.app_state.services.run_config = RunConfig(
        max_input_bytes=small_limit
    )

    resp = client.post(
        "/api/v1/documents",
        files={"file": ("exact.txt", data, "text/plain")},
    )

    assert resp.status_code == 201, (
        f"a {small_limit}-byte file at a {small_limit}-byte limit was refused: "
        f"{resp.status_code} {resp.text}"
    )


def test_an_oversized_upload_is_refused_without_naming_the_file(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    """The bound is enforced while reading, and the log names no filename or text."""
    canary = "PRIVACY-CANARY-oversized-90817"
    data = ("\u00a7 1. " + canary + " " + "a" * 8192 + "\n").encode("utf-8")
    client.app.state.app_state.services.run_config = RunConfig(max_input_bytes=64)

    with caplog.at_level(logging.INFO):
        response = client.post(
            "/api/v1/documents",
            files={"file": (f"{canary}.txt", data, "text/plain")},
        )

    assert response.status_code == 413, response.text
    assert response.json()["code"] == "limit_breach"
    logged = "\n".join(record.getMessage() for record in caplog.records)
    assert canary not in logged


def test_read_bounded_stops_reading_at_limit() -> None:
    """_read_bounded raises limit_breach and stops reading beyond the limit."""
    from contract_analyzer.api.documents import _read_bounded
    from contract_analyzer.api.errors import _ApiError

    chunk_size = 64 * 1024
    total_chunks = 10
    limit = 100 * 1024

    class StandInUpload:
        def __init__(self) -> None:
            self.requested_sizes: list[int] = []
            self.chunks_yielded = 0

        async def read(self, size: int = -1) -> bytes:
            self.requested_sizes.append(size)
            if self.chunks_yielded < total_chunks:
                self.chunks_yielded += 1
                return b"x" * chunk_size
            return b""

    upload = StandInUpload()

    async def exercise() -> None:
        with pytest.raises(_ApiError) as exc_info:
            await _read_bounded(cast(Any, upload), limit)
        assert exc_info.value.code == "limit_breach"

    asyncio.run(exercise())

    assert upload.chunks_yielded == 2
    assert upload.chunks_yielded < total_chunks
    assert upload.requested_sizes == [chunk_size, chunk_size]


def test_upload_does_not_block_the_event_loop(
    _services: AnalysisServices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ingestion runs off the loop, so other requests are served while it works.

    Driven over ASGITransport rather than TestClient: TestClient serialises requests
    through one portal, so a blocking handler is indistinguishable from a threaded one
    there and the test would pass either way.
    """
    order: list[str] = []
    original = IngestService.ingest

    def slow(self: IngestService, filename: str, data: bytes) -> Any:
        time.sleep(0.4)
        order.append("ingest")
        return original(self, filename, data)

    monkeypatch.setattr(IngestService, "ingest", slow)
    app = create_app(_services.settings, _services)

    async def exercise() -> None:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:

            async def upload() -> httpx.Response:
                return await ac.post(
                    "/api/v1/documents",
                    files={"file": ("umowa.txt", DOCUMENT_TEXT.encode(), "text/plain")},
                )

            async def health() -> httpx.Response:
                await asyncio.sleep(0.05)
                resp = await ac.get("/api/v1/health/live")
                order.append("health")
                return resp

            up, hl = await asyncio.gather(upload(), health())
            assert up.status_code == 201, up.text
            assert hl.status_code == 200, hl.text

    asyncio.run(exercise())

    # If ingest ran on the loop, health could not have been answered until it finished.
    assert order == ["health", "ingest"], f"loop was blocked during ingest: {order}"


def test_content_sweeper_survives_a_failing_pass(
    _services: AnalysisServices,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """One bad sweep must not silently end TTL purging.

    Retention is a privacy property, so the sweeper logs and keeps going.
    """
    calls: list[int] = []

    def exploding(_state: object) -> None:
        calls.append(1)
        raise RuntimeError("sweep boom")

    monkeypatch.setattr("contract_analyzer.api.app._purge_expired_content", exploding)
    _services.run_config = RunConfig(content_sweep_interval_seconds=0)

    with caplog.at_level(logging.ERROR):
        app = create_app(_services.settings, _services)
        with TestClient(app, raise_server_exceptions=False):
            deadline = time.monotonic() + 2.0
            while len(calls) < 2 and time.monotonic() < deadline:
                time.sleep(0.01)

    assert len(calls) >= 2, "sweeper stopped after its first failure"
    assert "content sweep failed" in caplog.text


def test_sse_on_unknown_run_returns_404_not_an_empty_stream(client: TestClient) -> None:
    """The declared 404 must be reachable.

    Raising from inside the SSE generator happens after the streaming response is
    built, so the client saw 200 with an empty body and waited forever.
    """
    resp = client.get("/api/v1/runs/00000000-0000-0000-0000-000000000099/events")

    assert resp.status_code == 404, f"{resp.status_code} {resp.headers} {resp.text!r}"
    assert resp.json()["code"] == "not_found"
    assert "text/event-stream" not in resp.headers.get("content-type", "")


def test_run_response_populates_every_non_nullable_spec_field(
    real_corpus_client: TestClient,
) -> None:
    """A field the spec declares non-nullable must never come back null.

    Schema-to-schema comparison cannot catch this: Pydantic marks every
    ``X | None = None`` field nullable, so the generated schema disagrees with the
    frozen spec on roughly twenty properties for modelling reasons alone. The real
    defect is narrower and only visible at runtime -- created_at was declared
    non-nullable and was returned as null on every response, because _build_run
    never read the column the store had been writing all along.
    """
    doc = _upload_txt(real_corpus_client)
    run_resp = real_corpus_client.post(
        "/api/v1/runs", json={"document_id": doc["id"], "arm": "mid"}
    )
    assert run_resp.status_code == 202, run_resp.text
    run_id = run_resp.json()["id"]
    _wait_for_terminal_run(real_corpus_client, run_id)
    body = real_corpus_client.get(f"/api/v1/runs/{run_id}").json()

    spec = yaml.safe_load(_SPEC_PATH.read_text(encoding="utf-8"))
    run_schema = spec["components"]["schemas"]["Run"]
    non_nullable = [
        name
        for name, prop in run_schema["properties"].items()
        if "null" not in _spec_types(prop)
    ]
    assert "created_at" in non_nullable, (
        "spec no longer pins created_at as non-nullable"
    )

    nulls = [name for name in non_nullable if name in body and body[name] is None]
    assert not nulls, (
        f"spec declares these non-nullable but the server sent null: {nulls}"
    )


def _spec_types(prop: dict[str, Any]) -> set[str]:
    """Types a spec property permits, flattening the oneOf-with-null idiom."""
    declared = prop.get("type")
    if isinstance(declared, list):
        return set(declared)
    if declared is not None:
        return {declared}
    types: set[str] = set()
    for variant in prop.get("oneOf", ()) or prop.get("anyOf", ()):
        types |= _spec_types(variant)
    return types


def _minimal_run_record(
    run_id: UUID, document_id: UUID, *, call_unit_count: int | None = None
) -> RunRecord:
    """A stored run with only the fields the metadata store requires."""
    return RunRecord(
        id=run_id,
        document_id=document_id,
        arm=ArmCode.MID,
        input_hash="h",
        config_version=CONFIG_VERSION,
        prompt_bundle_version="pb",
        corpus_snapshot_id="snap",
        tool_bundle_version=TOOL_BUNDLE_VERSION,
        requested_model="deepseek-v4-flash",
        parameters={"temperature": 0.0},
        retry_policy=RETRY_POLICY,
        concurrency=2,
        wall_budget_seconds=900.0,
        measurement_valid=False,
        cost=CostRecord.unknown("price_table_unavailable"),
        call_unit_count=call_unit_count,
    )


def test_dynamic_provider_error_code_keeps_the_run_readable(client: TestClient) -> None:
    """A model_http_<status> code must not make GET /runs/{id} a permanent 500.

    model.py mints one code per HTTP status, so no fixed table can list them all.
    Indexing the message table directly turned every rate-limited run into an
    unreadable 500 for the rest of its life.
    """
    state: _AppState = client.app.state.app_state
    doc = _upload_txt(client)
    run_id = uuid4()
    state.services.metadata.create_run(_minimal_run_record(run_id, UUID(doc["id"])))
    state.services.metadata.finish_run(
        run_id,
        "failed",
        totals=RunTotals(error_code="model_http_429", elapsed_ms=1.0),
    )

    resp = client.get(f"/api/v1/runs/{run_id}")

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["error"]["code"] == "model_http_429"
    assert body["error"]["message_pl"]


def test_units_not_processed_reaches_the_wire_when_a_unit_never_finishes(
    client: TestClient,
) -> None:
    """The corrected count must be visible to a client, not just inside RunResult.

    _build_metrics derived units_total from the findings that exist, so a unit that
    died before writing one vanished from the numerator and the denominator both,
    and the run reported as fully processed.
    """
    state: _AppState = client.app.state.app_state
    doc = _upload_txt(client)
    run_id = uuid4()
    record = _minimal_run_record(run_id, UUID(doc["id"]), call_unit_count=3)
    state.services.metadata.create_run(record)
    # one unit produced a finding; two died mid-fan-out and wrote nothing at all
    state.services.metadata.store_finding(
        FindingRecord(
            id=uuid4(),
            run_id=run_id,
            unit_id="u1",
            code=FindingCode.NO_BASIS_FOUND,
            legal_locators=(),
        )
    )
    state.services.metadata.finish_run(
        run_id, "failed", totals=RunTotals(elapsed_ms=1.0)
    )

    metrics = client.get(f"/api/v1/runs/{run_id}").json()["metrics"]

    assert metrics["units_total"] == 3, "denominator must be the dispatched call units"
    assert metrics["units_not_processed"] == 2, (
        "two units never reached a terminal finding and must be reported as such"
    )


def test_run_response_carries_the_synthesis_grouping(client: TestClient) -> None:
    """The synthesis role's output reaches the client instead of being discarded."""
    doc = _upload_txt(client)
    resp = client.post("/api/v1/runs", json={"document_id": doc["id"], "arm": "mid"})
    assert resp.status_code == 202, resp.text
    run = _wait_for_terminal_run(client, resp.json()["id"])

    assert run["status"] == "completed", run
    groups = run["synthesis"]
    assert groups is not None and len(groups) == 1
    group = groups[0]
    assert group["title"] and group["summary"]
    assert group["finding_ids"] == [finding["id"] for finding in run["findings"]]


def test_deleting_the_document_takes_the_synthesis_with_it(client: TestClient) -> None:
    """The grouping is derived from the document and shares its retention window."""
    doc = _upload_txt(client)
    resp = client.post("/api/v1/runs", json={"document_id": doc["id"], "arm": "mid"})
    assert resp.status_code == 202, resp.text
    run_id = resp.json()["id"]
    assert _wait_for_terminal_run(client, run_id)["synthesis"]

    assert client.delete(f"/api/v1/documents/{doc['id']}").status_code == 204

    # The grouping goes with the run that made it, which goes with the document.
    assert client.get(f"/api/v1/runs/{run_id}").status_code == 404


def test_expired_content_takes_the_synthesis_with_it(client: TestClient) -> None:
    """The retention window closing removes the grouping, not only the document text."""
    doc = _upload_txt(client)
    resp = client.post("/api/v1/runs", json={"document_id": doc["id"], "arm": "mid"})
    assert resp.status_code == 202, resp.text
    run_id = resp.json()["id"]
    assert _wait_for_terminal_run(client, run_id)["synthesis"]

    state: _AppState = client.app.state.app_state
    assert state.services.text_store is not None
    for text_key in list(state.text_keys.values()):
        state.services.text_store.expire(text_key)

    after = client.get(f"/api/v1/runs/{run_id}").json()
    assert after["content_available"] is False
    assert after["synthesis"] is None


def test_shutdown_purge_takes_the_synthesis_with_it(client: TestClient) -> None:
    """Process exit is one of the three moments when retained content must be erased."""
    doc = _upload_txt(client)
    resp = client.post("/api/v1/runs", json={"document_id": doc["id"], "arm": "mid"})
    assert resp.status_code == 202, resp.text
    run_id = resp.json()["id"]
    assert _wait_for_terminal_run(client, run_id)["synthesis"]

    state: _AppState = client.app.state.app_state
    state.runner.purge_all_text()

    assert state.runner.synthesis_for(UUID(run_id)) is None


def test_deleting_a_document_mid_run_stops_the_run_and_takes_it(
    client: TestClient,
) -> None:
    """Purging during a run ends it, rather than letting it answer without text.

    This used to be a recorded defect: the purge blanked the session text in
    place and terminalized nothing, so the run carried on against empty strings
    and completed with findings it never had the text to support -- a silent
    wrong answer where the rule is to fail loudly. The purge now cancels every
    run still reading the document before erasing it, so no such finding is
    produced, and the run goes with everything else the document left behind.
    """
    doc = _upload_txt(client)
    release = threading.Event()
    original_execute, started, _ = _blocking_execute_patch(release)
    run_id: str | None = None
    try:
        run = client.post("/api/v1/runs", json={"document_id": doc["id"], "arm": "mid"})
        assert run.status_code == 202, run.text
        run_id = run.json()["id"]
        assert started.wait(timeout=5.0)
        assert client.delete(f"/api/v1/documents/{doc['id']}").status_code == 204
    finally:
        release.set()
        AnalysisRunner.execute_run = original_execute  # type: ignore[method-assign]

    assert run_id is not None
    # Nothing is left to report on: the run the purge interrupted is gone, so it
    # cannot stand as an analysis of a document the reader erased.
    assert client.get(f"/api/v1/runs/{run_id}").status_code == 404
    state: _AppState = client.app.state.app_state
    assert state.services.metadata.get_run(UUID(run_id)) is None


def test_readiness_names_the_missing_credential_rather_than_the_corpus(
    real_corpus_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without a key, say so; do not spend a request finding out.

    The corpus check compares vector spaces and so needs one embedding call.
    Asked before the credential check it paid for that call, failed, and filed
    the result against the corpus.
    """
    monkeypatch.delenv(EMBEDDINGS_API_KEY_ENV, raising=False)
    resp = real_corpus_client.get("/api/v1/health/ready")
    assert resp.status_code == 503
    deps = {d["name"]: d for d in resp.json()["dependencies"]}
    assert deps["embedding_credential"]["ok"] is False
    assert deps["corpus"]["detail"] == "embedding_credential_missing"


def test_purging_a_document_twice_reports_success_twice(client) -> None:
    """Erasure is a postcondition: a document already gone is already purged.

    A retention sweep or a restart drops the text while the reader's tab still
    names the document. Answering their purge with 404 told them the erasure had
    failed at the one moment they most needed to know it had not.
    """
    upload = client.post(
        "/api/v1/documents",
        files={"file": ("contract.txt", "§ 1. Czynsz najmu.\n§ 2. Kaucja.")},
    )
    assert upload.status_code == 201
    document_id = upload.json()["id"]

    assert client.delete(f"/api/v1/documents/{document_id}").status_code == 204
    assert client.delete(f"/api/v1/documents/{document_id}").status_code == 204
    # A document this process never held is in the same state as one it purged.
    assert client.delete(f"/api/v1/documents/{uuid4()}").status_code == 204


def test_purge_erases_the_findings_not_only_the_text(client: TestClient) -> None:
    """The reader's purge takes the analysis, not just the contract it read.

    A finding carries the offsets, boxes and legal locators saying where in their
    contract the system looked. Leaving those behind made the button mean "forget
    the text", which is not what it says.
    """
    doc = _upload_txt(client)
    run = client.post("/api/v1/runs", json={"document_id": doc["id"], "arm": "mid"})
    assert run.status_code == 202
    run_id = run.json()["id"]
    assert _wait_for_terminal_run(client, run_id)["findings"]

    state: _AppState = client.app.state.app_state
    assert state.services.metadata.list_findings(UUID(run_id))
    assert state.services.metadata.list_attempts(UUID(run_id))

    assert client.delete(f"/api/v1/documents/{doc['id']}").status_code == 204

    assert state.services.metadata.get_run(UUID(run_id)) is None
    assert state.services.metadata.list_findings(UUID(run_id)) == []
    assert state.services.metadata.list_attempts(UUID(run_id)) == []
    assert client.get(f"/api/v1/runs/{run_id}").status_code == 404


def test_a_sealed_batch_refuses_the_purge(client: TestClient) -> None:
    """While a measurement is sealed the records are the measurement in progress.

    Every other interactive action is refused outright under the seal; erasing
    the batch's own runs must be too, or a reader's button could empty a
    registered measurement while it runs.
    """
    doc = _upload_txt(client)
    state: _AppState = client.app.state.app_state
    state.settings = replace(state.settings, evaluation_batch_open=True)
    try:
        refused = client.delete(f"/api/v1/documents/{doc['id']}")
        assert refused.status_code == 409
        assert refused.json()["code"] == "evaluation_batch_open"
    finally:
        state.settings = replace(state.settings, evaluation_batch_open=False)

    assert client.delete(f"/api/v1/documents/{doc['id']}").status_code == 204


def test_erasing_a_run_closes_the_stream_watching_it(client: TestClient) -> None:
    """A stream ends on a terminal status and on nothing else.

    Erasing a run removes the record a subscriber is watching. Without a last
    event the stream has nothing to end on, so a reader who left the run open in
    another tab would sit on keepalives for a run that no longer exists.
    """
    doc = _upload_txt(client)
    run = client.post("/api/v1/runs", json={"document_id": doc["id"], "arm": "mid"})
    assert run.status_code == 202
    run_id = run.json()["id"]
    _wait_for_terminal_run(client, run_id)

    state: _AppState = client.app.state.app_state
    assert state.services.events is not None
    watcher = state.services.events.subscribe(UUID(run_id))

    assert client.delete(f"/api/v1/documents/{doc['id']}").status_code == 204

    seen = []
    while not watcher.empty():
        seen.append(watcher.get_nowait())
    assert any(getattr(event, "status", None) == "cancelled" for event in seen), (
        f"no terminal event published before erasure: {seen}"
    )


def test_a_finished_run_does_not_hold_the_evaluation_slot(
    _services: AnalysisServices,
) -> None:
    """The batch runner posts the next case the moment a run reports terminal.

    The slot is freed by the task's done callback, which runs later, so asking
    whether anything is registered refused the next case for no reason.
    """
    _services.settings = replace(_services.settings, evaluation_batch_open=True)
    app = create_app(_services.settings, _services)
    client = TestClient(app)
    doc = _upload_txt(client)
    state = app.state.app_state

    finished: Any = MagicMock()
    finished.done = MagicMock(return_value=True)
    finished.cancelled = MagicMock(return_value=False)
    finished.exception = MagicMock(return_value=None)
    state.tasks.active_tasks[uuid4()] = finished
    admitted = client.post(
        "/api/v1/runs", json={"document_id": doc["id"], "arm": "mid"}
    )
    assert admitted.status_code == 202, (
        f"a finished run refused the next case of the batch: {admitted.text}"
    )

    running: Any = MagicMock()
    running.done = MagicMock(return_value=False)
    state.tasks.active_tasks.clear()
    state.tasks.active_tasks[uuid4()] = running
    refused = client.post("/api/v1/runs", json={"document_id": doc["id"], "arm": "mid"})
    assert refused.status_code == 409, (
        f"a run still executing must hold the batch slot: {refused.text}"
    )

    # A task that raised has written no terminal record; the callback that
    # writes one has not run yet, so that run is still in flight.
    failed: Any = MagicMock()
    failed.done = MagicMock(return_value=True)
    failed.cancelled = MagicMock(return_value=False)
    failed.exception = MagicMock(return_value=RuntimeError("induced defect"))
    state.tasks.active_tasks.clear()
    state.tasks.active_tasks[uuid4()] = failed
    unsettled = client.post(
        "/api/v1/runs", json={"document_id": doc["id"], "arm": "mid"}
    )
    assert unsettled.status_code == 409, (
        "a run that raised holds its slot until it is settled as failed: "
        f"{unsettled.status_code} {unsettled.text}"
    )


def test_the_application_closes_the_metadata_store_on_shutdown() -> None:
    """The pool holds connections and background workers for the store's life."""
    from contract_analyzer.api.app import _shutdown
    from contract_analyzer.storage.postgres import PostgresMetadataStore

    # Unconstrained, this stand-in grows a close() the real store need not have.
    metadata = MagicMock(spec=PostgresMetadataStore)
    state = MagicMock()
    state.services.metadata = metadata
    state.text_keys = {}
    state.services.text_store = None
    state.tasks.cancel_all = MagicMock(return_value=asyncio.sleep(0))

    asyncio.run(_shutdown(state, None))

    metadata.close.assert_called_once_with()


def test_a_cancelling_run_holds_its_slot_until_it_is_settled(
    _services: AnalysisServices,
) -> None:
    """Cancellation leaves the run without a terminal record until it settles.

    Removing the registry entry when the cancellation started freed the slot
    during that window, so an evaluation batch of one admitted a second run.
    """
    _services.settings = replace(_services.settings, evaluation_batch_open=True)
    app = create_app(_services.settings, _services)
    tc = TestClient(app)
    doc = _upload_txt(tc)

    running = asyncio.Event()
    admitted_during: list[int] = []

    async def held_execute(self: Any, prepared: Any) -> Any:
        running.set()
        await asyncio.sleep(30)
        raise AssertionError("the held run was never cancelled")

    async def exercise() -> None:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            first = await ac.post(
                "/api/v1/runs", json={"document_id": doc["id"], "arm": "mid"}
            )
            assert first.status_code == 202, first.text
            run_id = first.json()["id"]
            await asyncio.wait_for(running.wait(), timeout=5.0)

            async def cancel() -> None:
                await ac.post(f"/api/v1/runs/{run_id}/cancel")

            async def admit() -> None:
                response = await ac.post(
                    "/api/v1/runs", json={"document_id": doc["id"], "arm": "mid"}
                )
                admitted_during.append(response.status_code)

            await asyncio.gather(cancel(), admit())

    with patch.object(AnalysisRunner, "execute_run", held_execute):
        asyncio.run(exercise())

    assert admitted_during == [409], (
        "a run was admitted while a cancelled one had no terminal record: "
        f"{admitted_during}"
    )


def test_a_run_that_raised_is_recorded_failed_not_cancelled(
    _services: AnalysisServices,
) -> None:
    """Cancelling a task that already ended would rewrite why the run ended.

    The endpoint saw a record still marked running, cancelled the finished
    task, and stored `cancelled`. What actually happened is that the run
    raised, and the callback settling it as failed then found a terminal
    record and left it alone.
    """
    app = create_app(_services.settings, _services)
    tc = TestClient(app)
    doc = _upload_txt(tc)

    raised = asyncio.Event()

    async def raising_execute(self: Any, prepared: Any) -> Any:
        raised.set()
        raise RuntimeError("induced execution defect")

    async def exercise() -> tuple[int, str]:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            created = await ac.post(
                "/api/v1/runs", json={"document_id": doc["id"], "arm": "mid"}
            )
            assert created.status_code == 202, created.text
            run_id = created.json()["id"]

            # Wait for the task to have actually raised, so this exercises a
            # finished task rather than one cancelled before it started.
            await asyncio.wait_for(raised.wait(), timeout=5.0)
            cancelled = await ac.post(f"/api/v1/runs/{run_id}/cancel")
            for _ in range(50):
                await asyncio.sleep(0.01)
                read = await ac.get(f"/api/v1/runs/{run_id}")
                if read.json()["status"] != "running":
                    break
            return cancelled.status_code, read.json()["status"]

    with patch.object(AnalysisRunner, "execute_run", raising_execute):
        cancel_status, final_status = asyncio.run(exercise())

    assert final_status == "failed", (
        f"a run that raised was recorded as {final_status!r}"
    )
    assert cancel_status == 409, (
        f"cancelling an already finished task was accepted: {cancel_status}"
    )


def test_a_run_cancelled_by_nobody_still_settles_and_frees_its_slot(
    _services: AnalysisServices,
) -> None:
    """A task can end cancelled without any request having asked for it.

    The callback stepped aside on CancelledError because a cancel request
    normally owns the settlement. When nothing asked, nobody wrote a terminal
    record and the run stayed registered for ever, holding the only slot an
    evaluation batch has.
    """
    _services.settings = replace(_services.settings, evaluation_batch_open=True)
    app = create_app(_services.settings, _services)
    tc = TestClient(app)
    doc = _upload_txt(tc)
    state = app.state.app_state

    async def self_cancelling(self: Any, prepared: Any) -> Any:
        raise asyncio.CancelledError

    with patch.object(AnalysisRunner, "execute_run", self_cancelling):
        created = tc.post("/api/v1/runs", json={"document_id": doc["id"], "arm": "mid"})
        assert created.status_code == 202, created.text
        run_id = created.json()["id"]
        _wait_until(lambda: not state.tasks.has_running())

    stored = tc.get(f"/api/v1/runs/{run_id}").json()
    assert stored["status"] == "cancelled", (
        f"an unowned cancellation left the run {stored['status']!r}"
    )
    assert state.tasks.active_tasks == {}, "the run kept its slot for ever"

    admitted = tc.post("/api/v1/runs", json={"document_id": doc["id"], "arm": "mid"})
    assert admitted.status_code == 202, (
        f"the orphaned run still blocks the batch: {admitted.text}"
    )


def test_shutdown_records_interrupted_runs_as_failed_not_cancelled(
    _services: AnalysisServices,
) -> None:
    """Shutdown marks what it interrupted; a callback must not relabel it.

    cancel_all cancels every task and awaits it, so each callback runs before
    the mark does. Once callbacks settled unowned cancellations, they wrote
    `cancelled` and the mark then found nothing running.

    Claiming a run tells its callback to stand aside, so the claim outlives the
    task unless cancel_all drops it, and the registry grows for every restart.
    """
    from contract_analyzer.api.app import _shutdown

    app = create_app(_services.settings, _services)
    running = asyncio.Event()
    run_ids: list[UUID] = []

    async def held(self: Any, prepared: Any) -> Any:
        running.set()
        await asyncio.sleep(30)
        raise AssertionError("the held run was never cancelled")

    async def exercise() -> None:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            upload = await ac.post(
                "/api/v1/documents",
                files={
                    "file": ("umowa.txt", DOCUMENT_TEXT.encode("utf-8"), "text/plain")
                },
            )
            assert upload.status_code == 201, upload.text
            created = await ac.post(
                "/api/v1/runs",
                json={"document_id": upload.json()["id"], "arm": "mid"},
            )
            assert created.status_code == 202, created.text
            run_ids.append(UUID(created.json()["id"]))
            await asyncio.wait_for(running.wait(), timeout=5.0)
        await _shutdown(app.state.app_state, None)

    with patch.object(AnalysisRunner, "execute_run", held):
        asyncio.run(exercise())

    record = _services.metadata.get_run(run_ids[0])
    assert record is not None
    assert record.status == "failed", f"shutdown recorded the run as {record.status!r}"
    assert record.error_code == "process_restarted", (
        f"shutdown recorded error code {record.error_code!r}"
    )

    state = app.state.app_state
    assert not state.tasks.settling, (
        f"shutdown left {len(state.tasks.settling)} claimed runs in the registry"
    )


def test_a_settle_that_raises_still_frees_the_slot(
    _services: AnalysisServices,
) -> None:
    """The callback settles and releases; a raising settle must not keep the slot.

    The cancel endpoint already paired those in a finally. The callback did
    not, so a database error while settling left the run registered and an
    evaluation batch of one refusing every later request.
    """
    _services.settings = replace(_services.settings, evaluation_batch_open=True)
    app = create_app(_services.settings, _services)
    tc = TestClient(app)
    doc = _upload_txt(tc)
    state = app.state.app_state

    async def raising_execute(self: Any, prepared: Any) -> Any:
        raise ValueError("induced execution defect")

    def failing_finish(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("database unavailable")

    with (
        patch.object(AnalysisRunner, "execute_run", raising_execute),
        patch.object(type(_services.metadata), "finish_run", failing_finish),
    ):
        created = tc.post("/api/v1/runs", json={"document_id": doc["id"], "arm": "mid"})
        assert created.status_code == 202, created.text
        _wait_until(lambda: not state.tasks.has_running())

    assert state.tasks.active_tasks == {}, (
        f"a failed settlement kept the slot: {state.tasks.active_tasks}"
    )
