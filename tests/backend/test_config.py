from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import ValidationError

from contract_analyzer.agents.session import RunRequest
from contract_analyzer.config import RunConfig, Settings
from contract_analyzer.domain import ArmCode


def test_settings_imports_without_live_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    settings = Settings.from_env()
    assert settings.model_api_key is None


def test_settings_require_live_key_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    settings = Settings.from_env()
    with pytest.raises(ValueError, match="MODEL_API_KEY"):
        settings.require_live()


def test_settings_require_live_key_when_present() -> None:
    settings = Settings(model_api_key="test-key")
    settings.require_live()


def test_settings_default_to_the_openrouter_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MODEL_BASE_URL", raising=False)
    monkeypatch.delenv("MODEL_NAME", raising=False)
    settings = Settings.from_env()
    assert settings.model_base_url == "https://openrouter.ai/api/v1"
    assert settings.model_name == "meta/muse-spark-1.3-contributor"


def test_evaluation_batch_is_closed_by_default_and_reads_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("EVALUATION_BATCH_OPEN", raising=False)
    assert Settings.from_env().evaluation_batch_open is False
    monkeypatch.setenv("EVALUATION_BATCH_OPEN", "true")
    assert Settings.from_env().evaluation_batch_open is True


def test_settings_point_the_connector_elsewhere(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One OpenAI-compatible connector, so the endpoint is configuration."""
    monkeypatch.setenv("MODEL_BASE_URL", "http://vllm.internal:8000/v1")
    monkeypatch.setenv("MODEL_NAME", "bielik-11b")
    settings = replace(Settings.from_env(), model_api_key="test-key")
    settings.require_live()
    assert settings.model_base_url == "http://vllm.internal:8000/v1"
    assert settings.model_name == "bielik-11b"


def test_settings_refuse_an_empty_endpoint_or_model() -> None:
    with pytest.raises(ValueError, match="MODEL_BASE_URL"):
        Settings(model_api_key="test-key", model_base_url="").require_live()
    with pytest.raises(ValueError, match="MODEL_NAME"):
        Settings(model_api_key="test-key", model_name="").require_live()


def test_run_config_pdf_defaults() -> None:
    config = RunConfig()
    assert config.pdf_min_native_chars_per_page == 1
    assert config.pdf_native_coverage_share == 1.0
    assert config.pdf_ocr_render_scale == 200 / 72
    assert config.max_input_bytes == 25 * 1024 * 1024
    assert config.max_pdf_pages == 200
    assert config.blank_page_ink_share == 0.0001
    assert config.ocr_language == "pol"
    assert config.ocr_timeout_seconds == 120
    assert config.conversion_timeout_seconds == 120
    assert config.model_timeout_seconds == 180
    assert config.content_ttl_seconds == 3600

    config = RunConfig()
    with pytest.raises(ValidationError):
        config.structural_min = 99


def test_budget_coherence() -> None:
    """model_timeout_seconds * max_attempts must stay below wall_budget_seconds."""
    config = RunConfig()
    max_attempts = 3  # RETRY_POLICY = "bounded-3"
    request = RunRequest(document_id=uuid4(), arm=ArmCode.MID)
    assert config.model_timeout_seconds * max_attempts < request.wall_budget_seconds


def test_settings_reads_process_environment_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("MODEL_API_KEY=file-key\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MODEL_API_KEY", raising=False)

    settings = Settings.from_env()
    assert settings.model_api_key is None


def test_unrecognised_batch_flag_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL_API_KEY", "k")
    monkeypatch.setenv("EVALUATION_BATCH_OPEN", "maybe")
    with pytest.raises(ValueError, match="EVALUATION_BATCH_OPEN"):
        Settings.from_env()


def test_provider_headers_are_read_as_a_json_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MODEL_EXTRA_HEADERS", '{"x-session": "abc"}')
    assert dict(Settings.from_env().model_extra_headers) == {"x-session": "abc"}


def test_absent_provider_headers_are_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MODEL_EXTRA_HEADERS", raising=False)
    assert dict(Settings.from_env().model_extra_headers) == {}


def test_blank_provider_headers_are_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL_EXTRA_HEADERS", "   ")
    assert dict(Settings.from_env().model_extra_headers) == {}


def test_settings_own_their_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Constructed directly, the caller's dict must not stay reachable."""
    del monkeypatch
    supplied = {"x-session": "abc"}
    settings = Settings(model_extra_headers=supplied)
    supplied["x-session"] = "changed"
    assert dict(settings.model_extra_headers) == {"x-session": "abc"}
    with pytest.raises(TypeError):
        settings.model_extra_headers["x-session"] = "changed"  # type: ignore[index]


def test_a_reserved_header_is_refused_when_set_in_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reading the environment is not the only way to reach the transport."""
    del monkeypatch
    with pytest.raises(ValueError, match="MODEL_EXTRA_HEADERS"):
        Settings(model_extra_headers={"Authorization": "Bearer other"})


@pytest.mark.parametrize(
    "name",
    [
        "Authorization",
        "authorization",
        "Content-Type",
        " Authorization",
        "OpenAI-Organization",
        "openai-project",
    ],
)
def test_a_header_that_would_replace_a_reserved_one_is_refused(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    """The SDK merges these over its own, so the credential would be replaced."""
    monkeypatch.setenv("MODEL_EXTRA_HEADERS", json.dumps({name: "Bearer other"}))
    with pytest.raises(ValueError, match="MODEL_EXTRA_HEADERS"):
        Settings.from_env()


def test_the_headers_never_reach_a_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    """An operator can put a token here, and a traceback must not print it."""
    monkeypatch.setenv("MODEL_EXTRA_HEADERS", '{"x-session": "s3cr3t-value"}')
    assert "s3cr3t-value" not in repr(Settings.from_env())


@pytest.mark.parametrize(
    "raw", ['{"x": 1}', "not json", '["x"]', '"x"'], ids=["int", "text", "list", "str"]
)
def test_a_malformed_provider_header_setting_is_refused(
    monkeypatch: pytest.MonkeyPatch, raw: str
) -> None:
    """Silently dropping it would fail every model call with a provider 400."""
    monkeypatch.setenv("MODEL_EXTRA_HEADERS", raw)
    with pytest.raises(ValueError, match="MODEL_EXTRA_HEADERS"):
        Settings.from_env()


def test_model_endpoint_is_the_host_alone() -> None:
    """A configured endpoint reports its host, and nothing else about itself."""
    settings = Settings(
        model_api_key="secret-key",
        model_base_url="https://Router.Example.COM:8443/api/v1/route-token",
    )
    assert settings.model_endpoint == "router.example.com"


def test_model_endpoint_of_an_endpoint_with_no_host() -> None:
    """A base URL naming no host reports none, rather than half of a path."""
    assert Settings(model_base_url="/v1").model_endpoint == ""
