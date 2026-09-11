"""The entry types a worksheet is made of.

Every entry is frozen and closed, and each kind names exactly the fields it
carries, so an entry cannot hold a field belonging to another kind. The union is
discriminated on ``kind``, which is what makes the serialised worksheet
readable back into typed entries.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from contract_analyzer.agents.schema import FrozenModel
from contract_analyzer.domain import (
    DepartureDirection,
    DepartureState,
    ForceState,
    ProvisionCharacter,
    RetrievalCandidate,
    candidate_admits,
)

Author = Literal["researcher", "analyst", "verifier", "user", "system"]
EntryKind = Literal[
    "candidate",
    "character",
    "verdict",
    "read",
    "search",
    "user_note",
]
VerdictStage = Literal["relevance", "relation"]


class _EntryBase(FrozenModel):
    id: str
    seq: int
    author: Author


class CandidateEntry(_EntryBase):
    """A provision the researcher put forward, with the corpus facts about it.

    The force records travel with the entry rather than being re-read at
    decision time: the admitting act record and the vetoing provision record are
    what the finding rests on, and reading them once, when the candidate is
    posted, keeps the whole unit deciding against one corpus state.
    """

    kind: Literal["candidate"] = "candidate"
    locator: str
    act_identifier: str
    snapshot_id: str
    act_force: ForceState
    provision_force: ForceState
    why: str
    supporting_locators: tuple[str, ...] = ()


class CharacterEntry(_EntryBase):
    kind: Literal["character"] = "character"
    candidate_id: str
    character: ProvisionCharacter


class VerdictEntry(_EntryBase):
    """One verifier decision about one candidate at one stage."""

    kind: Literal["verdict"] = "verdict"
    candidate_id: str
    stage: VerdictStage
    relevant: bool | None = None
    quote: str | None = None
    departure: DepartureState | None = None
    direction: DepartureDirection | None = None
    raw_confidence: float | None = None
    based_on: tuple[str, ...] = ()


class ReadEntry(_EntryBase):
    kind: Literal["read"] = "read"
    locator: str


class SearchEntry(_EntryBase):
    kind: Literal["search"] = "search"
    phrase: str
    result_locators: tuple[str, ...]


class UserNoteEntry(_EntryBase):
    """A note the reader adds to the unit, for the later interactive phase."""

    kind: Literal["user_note"] = "user_note"
    note: str
    purpose: str
    finding_id: str | None = None


Entry = Annotated[
    CandidateEntry
    | CharacterEntry
    | VerdictEntry
    | ReadEntry
    | SearchEntry
    | UserNoteEntry,
    Field(discriminator="kind"),
]


def retrieval_candidate(entry: CandidateEntry, rank: int) -> RetrievalCandidate:
    """The candidate in the shape the decision rules take.

    ``DecisionFacts.from_candidate`` derives force from the two records rather
    than accepting it, which is the guard worth keeping, so the entry is lifted
    into the record that carries them. The two retrieval scores are zero because
    a posted candidate has no rank in any single search -- the researcher may have
    reached it by reading rather than by a hit -- and nothing on the decision
    path or in storage reads them.
    """
    return RetrievalCandidate(
        locator=entry.locator,
        snapshot_id=entry.snapshot_id,
        act_identifier=entry.act_identifier,
        act_force=entry.act_force,
        provision_force=entry.provision_force,
        rank=rank,
        sparse_score=0.0,
        dense_score=0.0,
    )


def admits(entry: CandidateEntry) -> bool:
    """Whether the candidate's force records admit it to adjudication."""
    return candidate_admits(retrieval_candidate(entry, rank=1))
