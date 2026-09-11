"""Shared fixtures for driving a whole run against a scripted provider.

The pipeline tests all need the same three things: a segmented document, a runner
wired to a mock transport, and a way to see what the provider was asked. They are
here so the test files can be about one subject each.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
from uuid import UUID, uuid4

import httpx
from model_client import mock_model_client
from worksheet_transport import DEFAULT_MODEL, TaskHandler, Turn, envelope, play

from contract_analyzer.agents.prompts import load_prompt_bundle
from contract_analyzer.agents.runner import AnalysisRunner
from contract_analyzer.agents.services import AnalysisServices
from contract_analyzer.agents.session import DocumentSession, RunRequest
from contract_analyzer.config import RunConfig, Settings
from contract_analyzer.corpus import QdrantCorpusIndex
from contract_analyzer.domain import (
    ArmCode,
    DocumentPayload,
    ReadMode,
    SourceAnchor,
)
from contract_analyzer.storage import EventBus, MetadataStore, RunTextStore
from contract_analyzer.structure import parse_references, segment

DOCUMENT_TEXT = """§ 1. Najemca zobowiązuje się do płatności czynszu.

§ 2. Zgodnie z § 1 najemca ponosi koszty mediów.

§ 3. Umowa podlega prawu polskiemu.
"""

LOCATOR = "https://api.sejm.gov.pl/eli/acts/DU/2023/725/text.html/arti=11"
LOCATOR_ART19A = "https://api.sejm.gov.pl/eli/acts/DU/2023/725/text.html/arti=19a"
LOCATOR_ART41 = "https://api.sejm.gov.pl/eli/acts/DU/2023/725/text.html/arti=41(1)"
FABRICATED_EVIDENCE_LOCATOR = "art. 9999 ... Dz.U. 2099 poz. 0"
INVENTED_CANDIDATE_LOCATOR = (
    "https://api.sejm.gov.pl/eli/acts/DU/2099/0/text.html/arti=9999"
)

# The tasks whose payload carries the call unit, which is where the arms differ.
ROLE_TASKS = (
    "researcher.search",
    "analyst.characterise",
    "verifier.relevance",
    "verifier.relation",
)


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class GraphDocument:
    session: DocumentSession
    whole_document_id: str
    resolved_target_id: str


def make_session(text: str, *, config: RunConfig | None = None) -> DocumentSession:
    payload = DocumentPayload(
        document_id=uuid4(),
        text=text,
        anchors=(SourceAnchor(start_offset=0, end_offset=len(text)),),
        read_mode=ReadMode.NATIVE_PDF,
        content_hash=content_hash(text),
    )
    units = segment(payload, config or RunConfig())
    return DocumentSession(
        document_id=payload.document_id,
        payload=payload,
        units=tuple(units),
        references=tuple(parse_references(units)),
    )


def make_document(text: str = DOCUMENT_TEXT) -> GraphDocument:
    session = make_session(text)
    citing_unit = next(unit for unit in session.units if "Zgodnie" in unit.text)
    target_id = next(
        reference.target_unit_id
        for reference in session.references
        if reference.citing_unit_id == citing_unit.id
        and reference.target_unit_id is not None
    )
    assert target_id is not None
    return GraphDocument(
        session=session,
        whole_document_id=session.whole_document_id,
        resolved_target_id=target_id,
    )


def make_transport(
    *,
    script: Mapping[str, TaskHandler] | None = None,
    on_turn: Callable[[Turn], None] | None = None,
    tokens_per_call: int = 10,
    model: str = DEFAULT_MODEL,
) -> httpx.MockTransport:
    """A provider that plays the worksheet script, with per-task overrides."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        turn = Turn(body)
        if on_turn is not None:
            on_turn(turn)
        if "tools" not in body:
            raise AssertionError("the pipeline sent a request carrying no tools")
        message: dict[str, Any] = dict(play(turn, script))
        return httpx.Response(
            200,
            request=request,
            json=envelope(
                message,
                model=model,
                input_tokens=tokens_per_call,
                output_tokens=2,
            ),
        )

    return httpx.MockTransport(handler)


class TickingClock:
    """A monotonic clock that only advances when the provider is called.

    Wire it to ``on_turn`` and give the run a wall budget of N seconds, and the
    budget runs out on the Nth model call. That keeps a budget scenario stated
    in calls, the way the run itself experiences it.
    """

    def __init__(self, *, seconds_per_turn: float = 1.0) -> None:
        self._now = 0.0
        self._seconds_per_turn = seconds_per_turn

    def __call__(self) -> float:
        return self._now

    def tick(self, _turn: object = None) -> None:
        self._now += self._seconds_per_turn


def build_runner(
    session: DocumentSession,
    built_corpus: QdrantCorpusIndex,
    tmp_path: Path,
    *,
    db_name: str = "metadata.sqlite3",
    transport: httpx.MockTransport | None = None,
    script: Mapping[str, TaskHandler] | None = None,
    on_turn: Callable[[Turn], None] | None = None,
    events: EventBus | None = None,
    text_store: RunTextStore | None = None,
    tokens_per_call: int = 10,
    corpus: object | None = None,
) -> AnalysisRunner:
    settings = Settings(
        model_api_key="test-secret-key",
        model_base_url="https://openrouter.ai/api/v1",
        model_name="deepseek-v4-flash",
    )
    resolved = transport or make_transport(
        script=script,
        on_turn=on_turn,
        tokens_per_call=tokens_per_call,
    )
    services = AnalysisServices(
        settings=settings,
        corpus=cast(Any, corpus if corpus is not None else built_corpus),
        client=mock_model_client(resolved, settings=settings),
        metadata=MetadataStore(tmp_path / db_name),
        prompt_bundle=load_prompt_bundle(),
        events=events,
        # A store by default, as the service has one: the worksheets and the
        # synthesis are only retained when there is somewhere to put them.
        text_store=text_store or RunTextStore(),
    )
    services.sessions[session.document_id] = session
    return AnalysisRunner(services)


def request_for(
    document: GraphDocument,
    arm: ArmCode,
    *,
    concurrency: int = 2,
    wall_budget_seconds: float = 300.0,
    **overrides: Any,
) -> RunRequest:
    return RunRequest(
        document_id=document.session.document_id,
        arm=arm,
        measurement_valid=True,
        wall_budget_seconds=wall_budget_seconds,
        concurrency=concurrency,
        **overrides,
    )


def role_payloads(turns: list[Turn], unit_id: str) -> list[dict[str, Any]]:
    return [
        turn.payload
        for turn in turns
        if turn.task in ROLE_TASKS and turn.payload.get("unit_id") == unit_id
    ]


def collect_retained_plaintext(
    runner: AnalysisRunner, *, document_id: UUID | None = None
) -> str:
    parts: list[str] = []

    def consider(session: DocumentSession) -> None:
        if document_id is not None and session.document_id != document_id:
            return
        parts.append(session.payload.text)
        parts.extend(unit.text for unit in session.units)

    for session in runner.services.sessions.values():
        consider(session)
    for active in runner._runtime._runs.values():
        consider(active.session)
    return "".join(parts)
