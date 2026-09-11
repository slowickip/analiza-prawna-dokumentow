"""The shared per-unit record both acting roles read and write.

One worksheet exists per (run, call unit). It is the only thing the roles share:
the researcher posts candidates on it, the verifier reads the
slice about the candidate it is judging and writes its verdict there, and the
scheduler reads it to decide who acts next. Nothing else carries state between
turns, so the routing is a function of the record rather than of a call order.

The worksheet lives in memory while the unit runs and is serialised into the
time-limited text store when the unit finishes. Its prose -- the researcher's
reasons, the verifier's quotes, the characterisation rationales -- never reaches
Postgres, graph state, events or logs.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from contract_analyzer.agents import views
from contract_analyzer.agents.entries import (
    Author,
    CandidateEntry,
    CharacterEntry,
    Entry,
    ReadEntry,
    SearchEntry,
    UserNoteEntry,
    VerdictEntry,
    VerdictStage,
)
from contract_analyzer.domain import (
    DepartureDirection,
    DepartureState,
    ForceState,
    ProvisionCharacter,
)


class WorksheetRecord(BaseModel):
    """The serialised worksheet, as it is retained and read back."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: UUID
    unit_id: str
    search_complete: bool
    entries: tuple[Entry, ...]


class Worksheet:
    """An ordered, append-only list of entries about one call unit."""

    def __init__(
        self,
        run_id: UUID,
        unit_id: str,
        *,
        on_append: Callable[[Entry], None] | None = None,
    ) -> None:
        self.run_id = run_id
        self.unit_id = unit_id
        # Whether the researcher's one search task has run. Posting nothing is a
        # legitimate outcome, so the flag rather than the presence of candidates
        # is what tells the scheduler that searching is behind it.
        self.search_complete = False
        self._entries: list[Entry] = []
        self._on_append = on_append

    @property
    def entries(self) -> tuple[Entry, ...]:
        return tuple(self._entries)

    def record(self) -> WorksheetRecord:
        return WorksheetRecord(
            run_id=self.run_id,
            unit_id=self.unit_id,
            search_complete=self.search_complete,
            entries=self.entries,
        )

    # ── appends ──

    def add_candidate(
        self,
        *,
        locator: str,
        act_identifier: str,
        snapshot_id: str,
        act_force: ForceState,
        provision_force: ForceState,
        why: str,
        supporting_locators: Sequence[str] = (),
    ) -> CandidateEntry:
        return self._append(
            CandidateEntry,
            author="researcher",
            locator=locator,
            act_identifier=act_identifier,
            snapshot_id=snapshot_id,
            act_force=act_force,
            provision_force=provision_force,
            why=why,
            supporting_locators=tuple(supporting_locators),
        )

    def add_character(
        self,
        *,
        candidate_id: str,
        character: ProvisionCharacter,
        author: Author = "analyst",
    ) -> CharacterEntry:
        return self._append(
            CharacterEntry,
            author=author,
            candidate_id=candidate_id,
            character=character,
        )

    def add_verdict(
        self,
        *,
        candidate_id: str,
        stage: VerdictStage,
        relevant: bool | None = None,
        quote: str | None = None,
        departure: DepartureState | None = None,
        direction: DepartureDirection | None = None,
        raw_confidence: float | None = None,
        based_on: Sequence[str] = (),
    ) -> VerdictEntry:
        return self._append(
            VerdictEntry,
            author="verifier",
            candidate_id=candidate_id,
            stage=stage,
            relevant=relevant,
            quote=quote,
            departure=departure,
            direction=direction,
            raw_confidence=raw_confidence,
            based_on=tuple(based_on),
        )

    def add_read(self, *, author: Author, locator: str) -> ReadEntry:
        return self._append(ReadEntry, author=author, locator=locator)

    def add_search(self, *, phrase: str, result_locators: Sequence[str]) -> SearchEntry:
        return self._append(
            SearchEntry,
            author="researcher",
            phrase=phrase,
            result_locators=tuple(result_locators),
        )

    def add_user_note(
        self, *, note: str, purpose: str, finding_id: str | None = None
    ) -> UserNoteEntry:
        return self._append(
            UserNoteEntry,
            author="user",
            note=note,
            purpose=purpose,
            finding_id=finding_id,
        )

    def _append[EntryT: BaseModel](
        self, entry_type: type[EntryT], **fields: object
    ) -> EntryT:
        seq = len(self._entries) + 1
        entry = entry_type(id=f"e{seq}", seq=seq, **fields)
        self._entries.append(entry)  # type: ignore[arg-type]
        if self._on_append is not None:
            self._on_append(entry)  # type: ignore[arg-type]
        return entry

    # ── queries ──

    def candidates(self) -> tuple[CandidateEntry, ...]:
        """Every posted candidate, in posting order."""
        return tuple(
            entry for entry in self._entries if isinstance(entry, CandidateEntry)
        )

    def candidate(self, candidate_id: str) -> CandidateEntry | None:
        for entry in self.candidates():
            if entry.id == candidate_id:
                return entry
        return None

    def character_for(self, candidate_id: str) -> CharacterEntry | None:
        for entry in self._entries:
            if isinstance(entry, CharacterEntry) and entry.candidate_id == candidate_id:
                return entry
        return None

    def verdict_for(
        self, candidate_id: str, stage: VerdictStage
    ) -> VerdictEntry | None:
        for entry in self._entries:
            if (
                isinstance(entry, VerdictEntry)
                and entry.candidate_id == candidate_id
                and entry.stage == stage
            ):
                return entry
        return None

    def known_locators(self, candidate_id: str) -> frozenset[str]:
        return views.known_locators(self._entries, candidate_id)

    def verifier_visible_entry_ids(self, candidate_id: str) -> frozenset[str]:
        return views.verifier_visible_entry_ids(self._entries, candidate_id)

    def render(
        self, role: Author, candidate_id: str | None = None
    ) -> list[dict[str, Any]]:
        return views.render(self._entries, role, candidate_id)
