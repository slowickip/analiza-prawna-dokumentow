"""FastAPI application composition and lifespan."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from contract_analyzer.agents.runner import AnalysisRunner
from contract_analyzer.agents.services import AnalysisServices
from contract_analyzer.api.documents import router as documents_router
from contract_analyzer.api.errors import register_error_handlers
from contract_analyzer.api.interactions import router as interactions_router
from contract_analyzer.api.meta import router as meta_router
from contract_analyzer.api.runs import router as runs_router
from contract_analyzer.api.state import _AppState, _purge_expired_content
from contract_analyzer.config import Settings
from contract_analyzer.ingest import IngestService

logger = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    state: _AppState = app.state.app_state
    sweep_task: asyncio.Task[None] | None = None
    if state.services.text_store is not None:
        interval = state.services.run_config.content_sweep_interval_seconds

        async def sweep_loop() -> None:
            while True:
                await asyncio.sleep(interval)
                try:
                    _purge_expired_content(state)
                except Exception:
                    # An unguarded raise here ends the task silently: the exception is
                    # never retrieved, and document text then outlives its TTL until
                    # shutdown. Retention is a privacy property, so the sweeper logs and
                    # keeps sweeping rather than dying on one bad pass.
                    logger.exception("content sweep failed; continuing")

        sweep_task = asyncio.create_task(sweep_loop())

    try:
        yield
    finally:
        await _shutdown(state, sweep_task)


async def _shutdown(state: _AppState, sweep_task: asyncio.Task[None] | None) -> None:
    """Cancel background work and purge retained text. Must run on every exit path."""
    if sweep_task is not None:
        sweep_task.cancel()
        try:
            await sweep_task
        except asyncio.CancelledError:
            pass
    # Shutdown owns settling these: it marks them failed with process_restarted
    # below. Without this the task callbacks settle each one as cancelled while
    # cancel_all awaits them, and the mark then finds nothing left to do.
    state.tasks.settling.update(state.tasks.active_tasks)
    await state.tasks.cancel_all()
    # Mark any remaining running runs as failed
    state.services.metadata.mark_interrupted_runs_failed()
    close_store = getattr(state.services.metadata, "close", None)
    if close_store is not None:
        close_store()
    state.runner.purge_all_text()
    for doc_id in list(state.text_keys):
        text_key = state.text_keys.pop(doc_id, None)
        if text_key is not None and state.services.text_store is not None:
            state.services.text_store.delete(text_key)
    corpus = state.services.corpus
    if corpus is not None:
        corpus.close()


def create_app(settings: Settings, services: AnalysisServices) -> FastAPI:
    app = FastAPI(
        title="Contract analyzer API",
        version="1.0.0",
        description=(
            "Wire contract for the three-arm Polish contract analysis artefact."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=_lifespan,
    )
    register_error_handlers(app)
    app.include_router(documents_router)
    app.include_router(runs_router)
    app.include_router(meta_router)
    app.include_router(interactions_router)

    # Serve compiled web UI when available
    web_dist = Path(__file__).resolve().parent.parent.parent.parent / "web" / "dist"
    if web_dist.is_dir():
        app.mount("/", StaticFiles(directory=str(web_dist), html=True), name="web")

    app_state = _AppState(
        services=services,
        runner=AnalysisRunner(services),
        ingest_service=IngestService(services.run_config),
        settings=settings,
    )
    app.state.app_state = app_state

    return app
