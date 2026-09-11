"""Tests for the production composition root (serve.py)."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from contract_analyzer.api import create_app
from contract_analyzer.config import Settings
from contract_analyzer.corpus import QdrantCorpusIndex
from contract_analyzer.openai_compatible import OpenAICompatibleClient
from contract_analyzer.serve import build_app, build_services
from contract_analyzer.storage import MetadataStore, open_metadata_store


def _app(built_corpus: QdrantCorpusIndex, store: MetadataStore) -> object:
    """The production graph over one already-published corpus.

    ``build_app`` reads its corpus from QDRANT_URL, which a test has no server
    for; everything else on the composition root is exercised as it ships.
    """
    settings = Settings(model_api_key="test-secret-key")
    services = build_services(settings, corpus=built_corpus, metadata_store=store)
    return create_app(settings, services)


def test_build_app_wires_the_production_object_graph(
    built_corpus: QdrantCorpusIndex, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MODEL_API_KEY", "test-secret-key")
    store = open_metadata_store(tmp_path / "metadata.sqlite3")
    app = _app(built_corpus, store)
    with TestClient(app) as client:
        assert client.get("/api/v1/health/live").json() == {"status": "ok"}
        config = client.get("/api/v1/config").json()
    assert config["corpus_snapshot"] == built_corpus.snapshot.id
    assert config["model_request_id"] == "meta/muse-spark-1.3-contributor"
    assert config["model_endpoint"] == "openrouter.ai"


def test_services_are_wired_to_the_openai_compatible_connector(
    built_corpus: QdrantCorpusIndex, tmp_path: Path
) -> None:
    settings = Settings(model_api_key="test-secret-key")
    store = open_metadata_store(tmp_path / "metadata.sqlite3")
    services = build_services(
        settings,
        corpus=built_corpus,
        metadata_store=store,
    )
    assert isinstance(services.client, OpenAICompatibleClient)


def test_build_app_follows_the_configured_endpoint_and_model(
    built_corpus: QdrantCorpusIndex, tmp_path: Path
) -> None:
    settings = Settings(
        model_api_key="test-secret-key",
        model_base_url="http://vllm.internal:8000/v1",
        model_name="bielik-11b",
    )
    store = open_metadata_store(tmp_path / "metadata.sqlite3")
    services = build_services(settings, corpus=built_corpus, metadata_store=store)
    app = create_app(settings, services)
    with TestClient(app) as client:
        config = client.get("/api/v1/config").json()
        readiness = client.get("/api/v1/health/ready").json()
    assert config["model_request_id"] == "bielik-11b"
    # The host the calls go to, without the port, the path or the credential:
    # two endpoints serving one identifier do not serve it identically, and the
    # served configuration is what a batch pins.
    assert config["model_endpoint"] == "vllm.internal"
    credential = [
        d for d in readiness["dependencies"] if d["name"] == "model_credential"
    ]
    assert credential[0]["ok"] is True


def test_missing_credential_stops_startup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing key stops the process rather than serving an app that cannot run."""
    monkeypatch.setenv("DATABASE_URL", "postgresql://app:secret@localhost:5432/db")
    monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    with pytest.raises(ValueError, match="MODEL_API_KEY"):
        build_app()


def test_missing_database_url_stops_startup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MODEL_API_KEY", "test-secret-key")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
    with pytest.raises(RuntimeError, match="DATABASE_URL"):
        build_app()


def test_missing_qdrant_url_stops_startup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MODEL_API_KEY", "test-secret-key")
    monkeypatch.setenv("DATABASE_URL", "postgresql://app:secret@localhost:5432/db")
    monkeypatch.delenv("QDRANT_URL", raising=False)
    with pytest.raises(RuntimeError, match="QDRANT_URL"):
        build_app()
