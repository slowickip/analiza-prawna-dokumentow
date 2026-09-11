"""The collaborators a run is given, injected once and never reached for globally."""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID

from contract_analyzer.agents.prompts import PromptBundle
from contract_analyzer.agents.session import DocumentSession
from contract_analyzer.config import RunConfig, Settings
from contract_analyzer.corpus import QdrantCorpusIndex
from contract_analyzer.model import ModelClient
from contract_analyzer.storage import (
    EventBus,
    MetadataStore,
    PostgresMetadataStore,
    RunTextStore,
)


@dataclass
class AnalysisServices:
    settings: Settings
    corpus: QdrantCorpusIndex
    client: ModelClient
    metadata: MetadataStore | PostgresMetadataStore
    prompt_bundle: PromptBundle
    text_store: RunTextStore | None = None
    events: EventBus | None = None
    sessions: dict[UUID, DocumentSession] = field(default_factory=dict)
    run_config: RunConfig = field(default_factory=RunConfig)
