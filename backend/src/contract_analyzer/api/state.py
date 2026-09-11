"""Application state and content availability checks."""

from __future__ import annotations

import asyncio
import logging
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from fastapi import Request

from contract_analyzer.agents.runner import AnalysisRunner
from contract_analyzer.agents.services import AnalysisServices
from contract_analyzer.api.errors import _raise_api
from contract_analyzer.api.schemas import DependencyStatus, Readiness
from contract_analyzer.config import Settings
from contract_analyzer.corpus import (
    CorpusIntegrityError,
    embedding_credential_configured,
)
from contract_analyzer.ingest import IngestService

logger = logging.getLogger(__name__)


class TaskRegistry:
    """Registry tracking in-flight background tasks for runs and documents."""

    def __init__(self) -> None:
        self.active_tasks: dict[UUID, asyncio.Task[Any]] = {}
        self.task_documents: dict[UUID, UUID] = {}
        self.erasing_documents: set[UUID] = set()
        # Runs whose settlement a caller owns, so the task's own callback
        # must not settle or release them a second time.
        self.settling: set[UUID] = set()

    def register(self, key: UUID, task: asyncio.Task[Any], document_id: UUID) -> None:
        """Track one in-flight task and the document it is working on."""
        self.active_tasks[key] = task
        self.task_documents[key] = document_id

    def release(self, key: UUID) -> None:
        """Forget a task that has finished, however it finished."""
        self.active_tasks.pop(key, None)
        self.task_documents.pop(key, None)
        self.settling.discard(key)

    def has_running(self) -> bool:
        """Whether any registered run still holds its slot.

        A run that finished normally wrote its own terminal record before the
        task ended, so its slot is free even though the callback that clears
        the registry has not run yet. Counting it would refuse the next run of
        an evaluation batch, which the batch runner hits on purpose: it polls
        until a run reports terminal and posts the next case at once.

        A run whose task raised or was cancelled has no terminal record yet;
        the same callback writes it. Until that callback runs, the run is
        still registered and its record still says running, so the slot is
        held.
        """
        for task in self.active_tasks.values():
            if not task.done():
                return True
            if task.cancelled() or task.exception() is not None:
                return True
        return False

    def is_erasing(self, document_id: UUID) -> bool:
        return document_id in self.erasing_documents

    async def cancel_for_run(self, run_id: UUID) -> None:
        """Stop a run's task but keep its slot, which its canceller releases.

        Removing the entry here freed the slot while the run had no terminal
        record yet, so an evaluation batch of one admitted a second run during
        the cancellation.
        """
        task = self.active_tasks.get(run_id)
        if task is None:
            return
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass

    async def cancel_for_document(self, document_id: UUID) -> None:
        keys = [k for k, doc in self.task_documents.items() if doc == document_id]
        for key in keys:
            task = self.active_tasks.pop(key, None)
            self.task_documents.pop(key, None)
            if task is None:
                continue
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.warning(
                    "task %s failed while cancelling for document %s erasure",
                    key,
                    document_id,
                    exc_info=True,
                )

    async def cancel_all(self) -> None:
        all_tasks = list(self.active_tasks.values())
        self.active_tasks.clear()
        self.task_documents.clear()
        for task in all_tasks:
            task.cancel()
        if all_tasks:
            await asyncio.gather(*all_tasks, return_exceptions=True)
        # The callbacks stepped aside for these, so nothing else clears them.
        self.settling.clear()


@dataclass
class _AppState:
    services: AnalysisServices
    runner: AnalysisRunner
    ingest_service: IngestService
    settings: Settings
    measured_mode: bool = True
    text_keys: dict[UUID, UUID] = field(default_factory=dict)
    tasks: TaskRegistry = field(default_factory=TaskRegistry)


def _state(request: Request) -> _AppState:
    state: _AppState = request.app.state.app_state
    return state


def _content_available(state: _AppState, document_id: UUID) -> bool:
    if state.tasks.is_erasing(document_id):
        return False
    if document_id not in state.services.sessions:
        return False
    text_key = state.text_keys.get(document_id)
    if text_key is None or state.services.text_store is None:
        return document_id in state.services.sessions
    if state.services.text_store.contains(text_key):
        return True
    _drop_document_content(state, document_id)
    return False


def _drop_document_content(state: _AppState, document_id: UUID) -> None:
    state.runner.purge_document_text(document_id)
    state.services.sessions.pop(document_id, None)
    text_key = state.text_keys.pop(document_id, None)
    if text_key is not None and state.services.text_store is not None:
        state.services.text_store.delete(text_key)


def _purge_expired_content(state: _AppState) -> int:
    text_store = state.services.text_store
    if text_store is None:
        return 0
    purged = text_store.purge_expired()
    for document_id in list(state.text_keys):
        text_key = state.text_keys[document_id]
        if not text_store.contains(text_key):
            _drop_document_content(state, document_id)
            purged += 1
    return purged


def _ensure_corpus_available(state: _AppState) -> None:
    corpus = state.services.corpus
    if corpus is None:
        _raise_api("dependency_unavailable")
    try:
        corpus.verify()
    except CorpusIntegrityError:
        _raise_api("dependency_unavailable")


def _check_tesseract(ocr_language: str) -> DependencyStatus:
    tesseract = shutil.which("tesseract")
    if tesseract is None:
        return DependencyStatus(
            name="ocr_language_data", ok=False, detail="tesseract not found"
        )
    try:
        completed = subprocess.run(
            [tesseract, "--list-langs"],
            check=True,
            timeout=5.0,
            capture_output=True,
            text=True,
        )
        langs = {line.strip() for line in completed.stdout.splitlines()}
        if ocr_language in langs:
            return DependencyStatus(name="ocr_language_data", ok=True)
        return DependencyStatus(
            name="ocr_language_data",
            ok=False,
            detail=f"language '{ocr_language}' not installed",
        )
    except Exception:
        logger.exception("tesseract check failed")
        return DependencyStatus(
            name="ocr_language_data", ok=False, detail="check failed"
        )


def _check_readiness(state: _AppState) -> Readiness:
    deps: list[DependencyStatus] = []

    embedding_credential = embedding_credential_configured()
    deps.append(
        DependencyStatus(
            name="embedding_credential",
            ok=embedding_credential,
            detail=None if embedding_credential else "API key not configured",
        )
    )

    if not embedding_credential:
        deps.append(
            DependencyStatus(
                name="corpus", ok=False, detail="embedding_credential_missing"
            )
        )
    else:
        try:
            state.services.corpus.verify()
            deps.append(DependencyStatus(name="corpus", ok=True))
        except CorpusIntegrityError as exc:
            deps.append(DependencyStatus(name="corpus", ok=False, detail=exc.code))
        except Exception:
            logger.exception("corpus verification failed")
            deps.append(
                DependencyStatus(name="corpus", ok=False, detail="check failed")
            )

    deps.append(_check_tesseract(state.services.run_config.ocr_language))

    soffice = shutil.which("soffice")
    deps.append(
        DependencyStatus(
            name="doc_converter",
            ok=soffice is not None,
            detail=None if soffice else "soffice not found",
        )
    )

    has_key = bool(state.settings.model_api_key)
    deps.append(
        DependencyStatus(
            name="model_credential",
            ok=has_key,
            detail=None if has_key else "API key not configured",
        )
    )

    ready = all(d.ok for d in deps)
    return Readiness(ready=ready, dependencies=deps)
