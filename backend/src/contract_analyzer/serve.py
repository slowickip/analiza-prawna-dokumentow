"""Production composition root.

The container runs ``uvicorn contract_analyzer.serve:build_app --factory``.
"""

from __future__ import annotations

import logging
import os

from fastapi import FastAPI

from contract_analyzer.agents.prompts import load_prompt_bundle
from contract_analyzer.agents.services import AnalysisServices
from contract_analyzer.api import create_app
from contract_analyzer.config import Settings
from contract_analyzer.corpus import QdrantCorpusIndex
from contract_analyzer.openai_compatible import OpenAICompatibleClient
from contract_analyzer.storage import (
    EventBus,
    MetadataStore,
    RunTextStore,
    open_metadata_store,
)

logger = logging.getLogger(__name__)


def build_services(
    settings: Settings,
    *,
    metadata_store: MetadataStore | None = None,
    corpus: QdrantCorpusIndex | None = None,
) -> AnalysisServices:
    client = OpenAICompatibleClient(settings)

    resolved_store: MetadataStore
    if metadata_store is not None:
        resolved_store = metadata_store
    elif settings.database_url:
        resolved_store = open_metadata_store(settings.database_url)
    else:
        raise RuntimeError("DATABASE_URL is required")

    resolved_corpus: QdrantCorpusIndex
    if corpus is not None:
        resolved_corpus = corpus
    elif settings.qdrant_url:
        # CORPUS_SNAPSHOT_ID pins a deployment to one published corpus. A measured run
        # sets it, and verification then refuses any other corpus rather than analysing
        # against whatever the alias currently points at.
        resolved_corpus = QdrantCorpusIndex.open(
            settings.qdrant_url,
            collection=settings.qdrant_collection,
            expected_snapshot_id=settings.corpus_snapshot_id,
        )
    else:
        raise RuntimeError("QDRANT_URL is required")

    return AnalysisServices(
        settings=settings,
        corpus=resolved_corpus,
        client=client,
        metadata=resolved_store,
        prompt_bundle=load_prompt_bundle(),
        text_store=RunTextStore(),
        events=EventBus(),
    )


def configure_logging() -> None:
    """Emit the application's own log lines beside uvicorn's.

    Uvicorn configures only its own loggers, so without this the run, unit and
    tool-refusal lines the analysis writes at INFO and WARNING never reach the
    container log. LOG_LEVEL selects the threshold; the lines themselves never
    carry document or model text.
    """
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def build_app() -> FastAPI:
    """uvicorn factory configuring PostgreSQL metadata and Qdrant corpus."""
    configure_logging()
    settings = Settings.from_env()
    if not settings.database_url or not settings.database_url.startswith(
        ("postgres://", "postgresql://")
    ):
        raise RuntimeError("DATABASE_URL must name a PostgreSQL database")
    if not settings.qdrant_url:
        raise RuntimeError("QDRANT_URL is required")
    settings.require_live()
    services = build_services(settings)
    logger.info(
        "starting application metadata=postgres corpus_collection=%s "
        "corpus_snapshot=%s",
        settings.qdrant_collection,
        settings.corpus_snapshot_id or "unpinned",
    )
    return create_app(settings, services)
