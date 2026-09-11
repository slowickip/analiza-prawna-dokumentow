"""Document upload, deletion, and content endpoints."""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, timedelta
from uuid import UUID

from fastapi import APIRouter, Request, UploadFile
from fastapi.responses import Response

from contract_analyzer.agents.session import DocumentSession
from contract_analyzer.api.errors import _ApiError, _raise_api
from contract_analyzer.api.schemas import (
    Block,
    DocumentContent,
    DocumentDescriptor,
    Error,
)
from contract_analyzer.api.state import (
    _AppState,
    _content_available,
    _drop_document_content,
    _state,
)
from contract_analyzer.domain import DocumentPayload, ReferenceRecord, Unit
from contract_analyzer.ingest import IngestError
from contract_analyzer.storage import RunEvent
from contract_analyzer.structure import StructureError, parse_references, segment
from contract_analyzer.structure.blocks import block_id_at

router = APIRouter(prefix="/api/v1")
logger = logging.getLogger(__name__)


async def _read_bounded(file: UploadFile, max_bytes: int) -> bytes:
    """The upload's bytes, refused before more than the limit is held in memory."""
    chunks: list[bytes] = []
    total = 0
    while chunk := await file.read(64 * 1024):
        total += len(chunk)
        if total > max_bytes:
            _raise_api("limit_breach")
        chunks.append(chunk)
    return b"".join(chunks)


@router.post(
    "/documents",
    status_code=201,
    response_model=DocumentDescriptor,
    responses={
        413: {"model": Error},
        415: {"model": Error},
        422: {"model": Error},
        503: {"model": Error},
    },
)
async def upload_document(file: UploadFile, request: Request) -> DocumentDescriptor:
    state = _state(request)
    filename = file.filename or ""
    data = await _read_bounded(file, state.services.run_config.max_input_bytes)

    def ingest_and_segment() -> tuple[
        DocumentPayload, list[Unit], list[ReferenceRecord]
    ]:
        payload = state.ingest_service.ingest(filename, data)
        units = segment(payload, state.services.run_config)
        return payload, units, parse_references(units)

    try:
        payload, units, references = await asyncio.to_thread(ingest_and_segment)
    except IngestError as exc:
        logger.info("upload rejected code=%s", exc.code)
        raise _ApiError(exc.code) from exc
    except StructureError as exc:
        logger.info("upload rejected code=%s", exc.code)
        raise _ApiError(exc.code) from exc

    session = DocumentSession(
        document_id=payload.document_id,
        payload=payload,
        units=tuple(units),
        references=tuple(references),
    )
    state.services.sessions[payload.document_id] = session

    # Store text in RunTextStore for TTL-based privacy management
    if state.services.text_store is not None:
        text_key = state.services.text_store.put(payload.text.encode("utf-8"))
        state.text_keys[payload.document_id] = text_key

    ttl = state.services.run_config.content_ttl_seconds
    expires_at = datetime.now(UTC) + timedelta(seconds=ttl)

    return DocumentDescriptor(
        id=payload.document_id,
        content_hash=payload.content_hash,
        read_mode=payload.read_mode,
        conversion=payload.conversion,
        unit_count=len(units),
        expires_at=expires_at,
    )


@router.delete(
    "/documents/{documentId}",
    status_code=204,
    responses={409: {"model": Error}, 422: {"model": Error}},
)
async def delete_document(documentId: UUID, request: Request) -> Response:
    """Erase one document: its text and associated analysis records."""
    state = _state(request)
    if state.settings.evaluation_batch_open:
        _raise_api("evaluation_batch_open")

    state.tasks.erasing_documents.add(documentId)
    try:
        await _erase(state, documentId)
    finally:
        state.tasks.erasing_documents.discard(documentId)
    return Response(status_code=204)


async def _erase(state: _AppState, documentId: UUID) -> None:
    """Stop every task reading this document, then remove what it left."""
    await state.tasks.cancel_for_document(documentId)
    if documentId in state.services.sessions:
        _drop_document_content(state, documentId)
    state.runner.drop_document_runs(documentId)

    if state.services.events is not None:
        for run in state.services.metadata.list_runs():
            if run.document_id == documentId:
                state.services.events.publish(
                    run.id, RunEvent.status_changed("cancelled")
                )
    state.services.metadata.delete_document_runs(documentId)


@router.get(
    "/documents/{documentId}/content",
    response_model=DocumentContent,
    responses={
        404: {"model": Error},
        410: {"model": Error},
        422: {"model": Error},
    },
)
async def get_document_content(documentId: UUID, request: Request) -> DocumentContent:
    state = _state(request)
    session = state.services.sessions.get(documentId)
    if session is None:
        # Check if it ever existed (document was deleted/expired)
        # Since we have no history, just return 404 for unknown, 410 for known-deleted
        _raise_api("not_found")
    if not _content_available(state, documentId):
        _raise_api("content_expired")
    blocks = [
        Block(
            id=block_id_at(index, anchor),
            text=session.payload.text[anchor.start_offset : anchor.end_offset],
            anchor=anchor,
        )
        for index, anchor in enumerate(session.payload.anchors)
    ]
    return DocumentContent(
        document_id=documentId,
        read_mode=session.payload.read_mode,
        blocks=blocks,
        units=list(session.units),
        references=list(session.references),
    )
