"""The mutable state of a run and of one call unit inside it.

Kept in memory on the run rather than in graph state: the worksheet holds the
roles' prose, and prose must not cross into graph state, events, logs or Postgres.
Everything durable leaves through a finding record, a counter or the time-limited
text store.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from uuid import UUID

from contract_analyzer.agents.entries import CandidateEntry
from contract_analyzer.agents.scheduler import Step
from contract_analyzer.agents.services import AnalysisServices
from contract_analyzer.agents.session import CallUnit, DocumentSession, RunRequest
from contract_analyzer.agents.worksheet import Worksheet
from contract_analyzer.budget import BudgetTracker
from contract_analyzer.corpus import LegalUnit
from contract_analyzer.domain import RetrievalCandidate
from contract_analyzer.storage import FindingRecord


@dataclass(frozen=True)
class RunSpend:
    """What the roles spent over one run, for the terminal metadata write.

    Recorded because the roles choose their own work under caps: the number of
    searches, turns, reads and defaulted characterisations is an
    outcome of the run rather than a parameter of it, and a unit whose search was
    cut off by the cap is not the same result as one that searched freely and found
    nothing.
    """

    finder_tool_turns: int
    finder_search_calls: int
    retrieval_cache_hits: int
    finder_budget_exhausted_units: int
    verifier_tool_turns: int
    provision_reads: int
    defaulted_characterisations: int


@dataclass(frozen=True)
class StoredText:
    """A blob in the time-limited store, remembered with the document it came from."""

    document_id: UUID
    text_key: UUID


@dataclass
class ActiveRun:
    services: AnalysisServices
    run_id: UUID
    request: RunRequest
    session: DocumentSession
    call_units: list[CallUnit]
    call_units_by_id: dict[str, CallUnit]
    budget: BudgetTracker
    semaphore: asyncio.Semaphore
    retrieval_cache: dict[str, list[RetrievalCandidate]]
    provisions: dict[str, LegalUnit | None]
    units: dict[str, UnitContext] = field(default_factory=dict)
    returned_model: str | None = None
    tokens_used: int = 0
    processed_unit_count: int = 0
    finder_tool_turns: int = 0
    finder_search_calls: int = 0
    retrieval_cache_hits: int = 0
    finder_budget_exhausted_units: int = 0
    verifier_tool_turns: int = 0
    provision_reads: int = 0
    defaulted_characterisations: int = 0

    def spend(self) -> RunSpend:
        return RunSpend(
            finder_tool_turns=self.finder_tool_turns,
            finder_search_calls=self.finder_search_calls,
            retrieval_cache_hits=self.retrieval_cache_hits,
            finder_budget_exhausted_units=self.finder_budget_exhausted_units,
            verifier_tool_turns=self.verifier_tool_turns,
            provision_reads=self.provision_reads,
            defaulted_characterisations=self.defaulted_characterisations,
        )


@dataclass
class UnitContext:
    """One call unit's worksheet and the counters its bounds are enforced against.

    ``pending_step`` hands the scheduler's decision to the next role without
    putting the candidate identifier into graph state. The provenance ledger
    contains only locators a corpus tool actually returned to a searching role, and
    it is shared: the analyst may cite evidence the researcher retrieved, because
    the pair works one worksheet, but neither may cite a locator nobody fetched.
    """

    active: ActiveRun
    call_unit: CallUnit
    worksheet: Worksheet
    pending_step: Step | None = None
    verifier_reads_by_candidate: dict[str, int] = field(default_factory=dict)
    locator_ledger: set[str] = field(default_factory=set)
    candidate_count: int = 0
    search_turns: int = 0
    verifier_turns: int = 0
    findings: list[FindingRecord] = field(default_factory=list)

    def candidate_for(self, step: Step) -> CandidateEntry:
        """The candidate this step is about, which the scheduler guarantees exists."""
        candidate = (
            None
            if step.candidate_id is None
            else self.worksheet.candidate(step.candidate_id)
        )
        if candidate is None:
            raise RuntimeError(f"{step.task} scheduled without a candidate")
        return candidate
