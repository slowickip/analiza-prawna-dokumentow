"""The run registry: what is in flight, what it spent, and what it keeps.

Owns active runs and worksheets, and drives each call unit through its bounded
role cycle. Everything a role writes stays here until the unit finishes and its
worksheet is handed to the time-limited store.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from uuid import UUID

from contract_analyzer.agents import unit as unit_cycle
from contract_analyzer.agents.entries import Entry
from contract_analyzer.agents.errors import BudgetExhausted
from contract_analyzer.agents.findings import store_not_processed
from contract_analyzer.agents.retention import TextRetention
from contract_analyzer.agents.services import AnalysisServices
from contract_analyzer.agents.session import (
    CallUnit,
    DocumentSession,
    RunRequest,
    purge_session_plaintext,
)
from contract_analyzer.agents.state import ActiveRun, RunSpend, UnitContext
from contract_analyzer.agents.synthesizer import (
    SynthesisResponse,
    synthesis_for,
    synthesise,
)
from contract_analyzer.agents.worksheet import Worksheet, WorksheetRecord
from contract_analyzer.budget import BudgetTracker
from contract_analyzer.storage import FindingRecord, InterruptionReason, RunEvent

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class UnitOutcome:
    processed_unit_ids: tuple[str, ...] = ()
    unprocessed_unit_ids: tuple[str, ...] = ()
    finding_ids: tuple[str, ...] = ()
    interruption: bool = False


class GraphRuntime:
    def __init__(self, services: AnalysisServices) -> None:
        self.services = services
        self._runs: dict[UUID, ActiveRun] = {}
        self._retention = TextRetention(services)

    # ── run lifetime ──

    def begin_run(
        self,
        *,
        run_id: UUID,
        request: RunRequest,
        session: DocumentSession,
        call_units: list[CallUnit],
        budget: BudgetTracker,
    ) -> None:
        self._runs[run_id] = ActiveRun(
            services=self.services,
            run_id=run_id,
            request=request,
            session=session,
            call_units=call_units,
            call_units_by_id={unit.unit_id: unit for unit in call_units},
            budget=budget,
            semaphore=asyncio.Semaphore(request.concurrency),
            retrieval_cache={},
            provisions={},
        )

    def active(self, run_id: UUID) -> ActiveRun:
        return self._runs[run_id]

    def retire_run(self, run_id: UUID) -> None:
        self._runs.pop(run_id, None)

    def drop_document_runs(self, document_id: UUID) -> int:
        """Forget every in-process run of one document, not only its text.

        purge_document_text blanks what a live run is reading and leaves the run
        itself in place, which is right while the run is still going. An erasure
        is the other case: the reader asked for the work to stop existing, and an
        ActiveRun outliving its own database record still holds the request that
        started it -- including, for a contest or a re-analysis, the note the
        reader typed. Cancel the tasks before calling this.
        """
        doomed = [
            run_id
            for run_id, active in self._runs.items()
            if active.session.document_id == document_id
        ]
        for run_id in doomed:
            active = self._runs.pop(run_id)
            purge_session_plaintext(active.session)
            active.retrieval_cache.clear()
            active.provisions.clear()
        return len(doomed)

    def purge_document_text(self, document_id: UUID) -> None:
        for active in self._runs.values():
            if active.session.document_id == document_id:
                purge_session_plaintext(active.session)
                active.retrieval_cache.clear()
                active.provisions.clear()
        self._retention.drop_document(document_id)

    def purge_all_text(self) -> None:
        for active in self._runs.values():
            purge_session_plaintext(active.session)
            active.retrieval_cache.clear()
            active.provisions.clear()
        self._runs.clear()
        self._retention.drop_all()

    def tokens_for(self, run_id: UUID) -> int:
        return self._runs[run_id].tokens_used

    def returned_model_for(self, run_id: UUID) -> str | None:
        return self._runs[run_id].returned_model

    def interruption_reason_for(self, run_id: UUID) -> InterruptionReason | None:
        return self._runs[run_id].budget.exhaustion_reason

    def spend_for(self, run_id: UUID) -> RunSpend:
        return self._runs[run_id].spend()

    def synthesis_for(self, run_id: UUID) -> SynthesisResponse | None:
        return synthesis_for(self._retention, run_id)

    def worksheet_for(self, run_id: UUID, unit_id: str) -> WorksheetRecord | None:
        """One finished unit's worksheet, while the retention window is open."""
        raw = self._retention.worksheet_bytes(run_id, unit_id)
        return None if raw is None else WorksheetRecord.model_validate_json(raw)

    def worksheet_unit_ids(self, run_id: UUID) -> tuple[str, ...]:
        """Identifiers of retained worksheets for one run."""
        return self._retention.worksheet_unit_ids(run_id)

    # ── the unit worker ──

    async def process_unit(self, run_id: UUID, unit_id: str) -> UnitOutcome:
        """Drive one call unit through the cycle, under the run's concurrency limit.

        A budget that runs out is recorded as an unprocessed unit rather than a
        failed run: the units that did finish keep their findings. Model and
        pipeline failures re-raise, because an unusable answer is a defect in the
        run rather than a result of it.
        """
        active = self._runs[run_id]

        async with active.semaphore:
            if not active.budget.can_schedule():
                return self._not_processed(active, unit_id)
            context = self._open_unit(active, unit_id)
            try:
                finding_ids = await unit_cycle.run(context)
            except BudgetExhausted:
                return self._not_processed(active, unit_id)
            finally:
                self._close_unit(active, context)

            exhausted = active.budget.exhausted
            active.processed_unit_count += 1
            if self.services.events is not None:
                self.services.events.publish(
                    run_id,
                    RunEvent.counter("processed_units", active.processed_unit_count),
                )
            return UnitOutcome(
                processed_unit_ids=(unit_id,),
                finding_ids=tuple(finding_ids),
                interruption=exhausted,
            )

    def _not_processed(self, active: ActiveRun, unit_id: str) -> UnitOutcome:
        finding = store_not_processed(active, unit_id)
        return UnitOutcome(
            unprocessed_unit_ids=(unit_id,),
            finding_ids=(str(finding.id),),
            interruption=True,
        )

    def _open_unit(self, active: ActiveRun, unit_id: str) -> UnitContext:
        worksheet = Worksheet(
            active.run_id,
            unit_id,
            on_append=lambda entry: self._announce(active, unit_id, entry),
        )
        context = UnitContext(
            active=active,
            call_unit=active.call_units_by_id[unit_id],
            worksheet=worksheet,
        )
        if active.request.user_note is not None:
            worksheet.add_user_note(
                note=active.request.user_note,
                purpose=active.request.interaction or "analyse",
                finding_id=(
                    str(active.request.user_note_finding_id)
                    if active.request.user_note_finding_id is not None
                    else None
                ),
            )
        active.units[unit_id] = context
        return context

    def _close_unit(self, active: ActiveRun, context: UnitContext) -> None:
        """Retain the finished worksheet and drop it from memory.

        The worksheet does not outlive its unit in memory even when the unit died:
        a half-finished worksheet is still the record of what was tried, and the
        retained copy is where a later interactive phase reads it from.
        """
        unit_id = context.call_unit.unit_id
        active.units.pop(unit_id, None)
        self._retention.keep_worksheet(
            run_id=active.run_id,
            document_id=active.session.document_id,
            unit_id=unit_id,
            payload=context.worksheet.record().model_dump_json().encode("utf-8"),
        )
        logger.info(
            "unit finished run=%s unit=%s candidates=%d "
            "codes=%s search_turns=%d verifier_turns=%d",
            active.run_id,
            unit_id,
            context.candidate_count,
            [record.code.value for record in context.findings],
            context.search_turns,
            context.verifier_turns,
        )

    def _announce(self, active: ActiveRun, unit_id: str, entry: Entry) -> None:
        """Say that the worksheet grew, naming only who wrote what kind of entry."""
        logger.debug(
            "worksheet append run=%s unit=%s author=%s kind=%s",
            active.run_id,
            unit_id,
            entry.author,
            entry.kind,
        )
        if self.services.events is not None:
            self.services.events.publish(
                active.run_id,
                RunEvent.worksheet(
                    role=entry.author, entry_kind=entry.kind, unit_id=unit_id
                ),
            )

    async def synthesize(self, run_id: UUID, finding_ids: list[str]) -> bool:
        """Synthesize the completed unit findings, or report budget interruption."""
        active = self._runs[run_id]
        if not active.budget.can_schedule():
            return True
        return await synthesise(
            active, self._retention, [UUID(value) for value in finding_ids if value]
        )

    def store_not_processed(self, run_id: UUID, unit_id: str) -> FindingRecord:
        return store_not_processed(self._runs[run_id], unit_id)
