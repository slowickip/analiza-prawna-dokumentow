"""Configuration and health endpoints."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response

from contract_analyzer.agents.tools import TOOL_BUNDLE_VERSION
from contract_analyzer.api.schemas import (
    HealthLive,
    PublicConfig,
    PublicConfigLimits,
    Readiness,
)
from contract_analyzer.api.state import _check_readiness, _state
from contract_analyzer.corpus import EMBEDDING_MODEL, embedding_space_fingerprint
from contract_analyzer.domain import ArmCode

router = APIRouter(prefix="/api/v1")


@router.get("/config", response_model=PublicConfig)
async def get_config(request: Request) -> PublicConfig:
    state = _state(request)
    config = state.services.run_config
    return PublicConfig(
        model_request_id=state.settings.model_name,
        model_endpoint=state.settings.model_endpoint,
        prompt_bundle_version=state.services.prompt_bundle.version,
        corpus_snapshot=state.services.corpus.snapshot.id,
        corpus_source_format=state.services.corpus.snapshot.source_format,
        embedding_model=EMBEDDING_MODEL,
        embedding_space_fingerprint=embedding_space_fingerprint(),
        tool_bundle_version=TOOL_BUNDLE_VERSION,
        arms=[ArmCode.OFF, ArmCode.MID, ArmCode.ON],
        measured_mode=state.measured_mode,
        evaluation_batch_open=state.settings.evaluation_batch_open,
        limits=PublicConfigLimits(
            max_input_bytes=config.max_input_bytes,
            max_pdf_pages=config.max_pdf_pages,
            content_ttl_seconds=config.content_ttl_seconds,
        ),
    )


@router.get("/health/live", response_model=HealthLive)
async def health_live() -> HealthLive:
    return HealthLive(status="ok")


@router.get(
    "/health/ready",
    response_model=Readiness,
    responses={503: {"model": Readiness}},
)
async def health_ready(request: Request) -> Response:
    state = _state(request)
    # Off the loop: the check opens a synchronous connection to the embedding
    # service and shells out to tesseract, and compose.yaml probes it every 30s,
    # so a stalled dependency would otherwise freeze the whole application.
    readiness = await asyncio.to_thread(_check_readiness, state)
    status_code = 200 if readiness.ready else 503
    return JSONResponse(
        status_code=status_code,
        content=readiness.model_dump(exclude_none=True),
    )
