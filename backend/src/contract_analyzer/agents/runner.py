"""The run as the rest of the system sees it: prepare, execute, report.

Preparation is separate from execution so a caller can observe the registered run
and its parity bundle before any model call happens. Execution distinguishes the
three ways a run can end: it completes, its budget runs out and it reports a
partial result, or an answer is unusable and it fails with a machine code.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from contract_analyzer.agents.completion import completed, failed
from contract_analyzer.agents.document_graph import run_document
from contract_analyzer.agents.errors import RunPipelineError
from contract_analyzer.agents.parity import (
    CONFIG_VERSION,
    GRAPH_TOPOLOGY_VERSION,
    RETRY_POLICY,
    ParityBundle,
)
from contract_analyzer.agents.runtime import GraphRuntime
from contract_analyzer.agents.services import AnalysisServices
from contract_analyzer.agents.session import (
    CallUnit,
    ChatRequest,
    ChatResponse,
    DocumentSession,
    PreparedRun,
    RouteRequest,
    RunRequest,
    RunResult,
    build_call_units,
    purge_session_plaintext,
    with_model_safeguards,
)
from contract_analyzer.agents.synthesizer import SynthesisResponse
from contract_analyzer.agents.tools import TOOL_BUNDLE_VERSION
from contract_analyzer.agents.worksheet import WorksheetRecord
from contract_analyzer.budget import BudgetTracker
from contract_analyzer.model import (
    InvalidModelResponse,
    ModelCallFailed,
    ModelIdentityChanged,
)
from contract_analyzer.storage import (
    CostRecord,
    RunEvent,
    RunRecord,
)

if TYPE_CHECKING:
    from contract_analyzer.agents.interactive_tools import RouteMessageArgs

logger = logging.getLogger(__name__)


def start_run(
    services: AnalysisServices,
    runtime: GraphRuntime,
    request: RunRequest,
    *,
    call_units: list[CallUnit] | None = None,
    parent: RunRecord | None = None,
) -> PreparedRun:
    """Register one run and return the state needed to execute or finish it."""
    request = with_model_safeguards(request)
    session = services.sessions[request.document_id]
    selected_units = (
        build_call_units(session, request.arm) if call_units is None else call_units
    )
    if request.unit_id is not None:
        selected_units = [
            unit for unit in selected_units if unit.unit_id == request.unit_id
        ]
        if not selected_units:
            raise KeyError(request.unit_id)
    context_edge_count = sum(len(unit.context_unit_ids) for unit in selected_units)
    run_id = uuid4()
    budget = BudgetTracker(
        wall_budget_seconds=request.wall_budget_seconds,
        clock=request.clock,
    )
    runtime.begin_run(
        run_id=run_id,
        request=request,
        session=session,
        call_units=selected_units,
        budget=budget,
    )
    parity = ParityBundle(
        input_hash=session.input_hash,
        requested_model=services.settings.model_name,
        returned_model=None,
        prompt_bundle_version=services.prompt_bundle.version,
        corpus_snapshot_id=services.corpus.snapshot.id,
        tool_bundle_version=TOOL_BUNDLE_VERSION,
        parameters=dict(request.parameters),
        retry_policy=RETRY_POLICY,
        concurrency=request.concurrency,
        wall_budget_seconds=request.wall_budget_seconds,
    )
    services.metadata.create_run(
        RunRecord(
            id=run_id,
            document_id=session.document_id,
            arm=request.arm,
            input_hash=parent.input_hash if parent is not None else session.input_hash,
            config_version=(
                parent.config_version if parent is not None else CONFIG_VERSION
            ),
            prompt_bundle_version=services.prompt_bundle.version,
            corpus_snapshot_id=services.corpus.snapshot.id,
            tool_bundle_version=TOOL_BUNDLE_VERSION,
            requested_model=services.settings.model_name,
            parameters=dict(request.parameters),
            retry_policy=parent.retry_policy if parent is not None else RETRY_POLICY,
            concurrency=request.concurrency,
            wall_budget_seconds=request.wall_budget_seconds,
            measurement_valid=request.measurement_valid,
            parent_run_id=request.parent_run_id,
            interaction=request.interaction,
            cost=CostRecord.unknown("price_table_unavailable"),
            graph_topology_version=(
                parent.graph_topology_version
                if parent is not None
                else GRAPH_TOPOLOGY_VERSION
            ),
            context_edge_count=context_edge_count,
            call_unit_count=len(selected_units),
        )
    )
    if services.events is not None:
        services.events.publish(run_id, RunEvent.status_changed("running"))
    logger.info(
        "run started run=%s arm=%s call_units=%d context_edges=%d",
        run_id,
        request.arm.value,
        len(selected_units),
        context_edge_count,
    )

    clock = request.clock or time.monotonic
    started_at = clock()
    services.metadata.set_run_monotonic_start(run_id, started_at)

    def elapsed_ms() -> float:
        return (clock() - started_at) * 1000

    return PreparedRun(
        run_id=run_id,
        call_units=selected_units,
        parity=parity,
        elapsed_ms=elapsed_ms,
    )


class AnalysisRunner:
    def __init__(self, services: AnalysisServices) -> None:
        self.services = services
        self._runtime = GraphRuntime(services)

    def start_run(self, request: RunRequest) -> PreparedRun:
        """Register the run and its immutable configuration before it executes."""
        return start_run(self.services, self._runtime, request)

    async def execute_run(self, prepared: PreparedRun) -> RunResult:
        """Execute a prepared run and record how it ended.

        ``asyncio.CancelledError`` is deliberately not caught: cancellation must
        propagate so the caller can mark the run cancelled rather than leaving a
        failed record behind.
        """
        try:
            interruption, unprocessed_count = await run_document(
                self._runtime, prepared
            )
        except (
            ModelCallFailed,
            InvalidModelResponse,
            ModelIdentityChanged,
            RunPipelineError,
        ) as error:
            return failed(self.services, self._runtime, prepared, error.code)
        return completed(
            self.services,
            self._runtime,
            prepared,
            interruption=interruption,
            unprocessed_count=unprocessed_count,
        )

    async def run(self, request: RunRequest) -> RunResult:
        prepared = self.start_run(request)
        return await self.execute_run(prepared)

    async def explain(self, request: ChatRequest) -> ChatResponse:
        from contract_analyzer.agents.interactive_chat import explain

        return await explain(self.services, self._runtime, request)

    async def route(self, request: RouteRequest) -> RouteMessageArgs:
        from contract_analyzer.agents.interactive_chat import route

        return await route(self.services, self._runtime, request)

    def _session(self, document_id: UUID) -> DocumentSession:
        return self.services.sessions[document_id]

    def synthesis_for(self, run_id: UUID) -> SynthesisResponse | None:
        return self._runtime.synthesis_for(run_id)

    def worksheet_for(self, run_id: UUID, unit_id: str) -> WorksheetRecord | None:
        return self._runtime.worksheet_for(run_id, unit_id)

    def worksheet_unit_ids(self, run_id: UUID) -> tuple[str, ...]:
        """Identifiers of worksheets retained for one run."""
        return self._runtime.worksheet_unit_ids(run_id)

    def purge_document_text(self, document_id: UUID) -> None:
        self._runtime.purge_document_text(document_id)

    def retire_run(self, run_id: UUID) -> None:
        self._runtime.retire_run(run_id)

    def drop_document_runs(self, document_id: UUID) -> int:
        return self._runtime.drop_document_runs(document_id)

    def purge_all_text(self) -> None:
        for session in list(self.services.sessions.values()):
            purge_session_plaintext(session)
        self.services.sessions.clear()
        self._runtime.purge_all_text()
