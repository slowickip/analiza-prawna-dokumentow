"""What each role is shown of a worksheet, and which locators it already names.

Kept apart from the worksheet itself because these are read-only projections: a
role's view is a policy about what that role may see, not part of the record.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from contract_analyzer.agents.entries import (
    Author,
    CandidateEntry,
    CharacterEntry,
    Entry,
    UserNoteEntry,
    VerdictEntry,
)

VERIFIER_PROJECTION_FIELDS: tuple[str, ...] = (
    "payload.unit_id",
    "payload.unit_text",
    "payload.context_units",
    "payload.candidate_locator",
    "payload.candidate_act_identifier",
    "payload.candidate_text",
    "payload.basis_text",
    "payload.basis_character",
    "payload.basis_permitted_direction",
    "worksheet.entry.id",
    "worksheet.entry.seq",
    "worksheet.entry.author",
    "worksheet.entry.kind",
    "worksheet.candidate.id",
    "worksheet.candidate.locator",
    "worksheet.candidate.act_identifier",
    "worksheet.candidate.supporting_locators",
    "worksheet.character.id",
    "worksheet.character.candidate_id",
    "worksheet.character.evidence_locators",
    "worksheet.verdict.candidate_id",
    "worksheet.verdict.stage",
    "worksheet.verdict.relevant",
    "worksheet.verdict.quote",
    "worksheet.verdict.departure",
    "worksheet.verdict.direction",
    "worksheet.verdict.raw_confidence",
    "worksheet.verdict.based_on",
    "worksheet.user_note.note",
    "worksheet.user_note.purpose",
    "worksheet.user_note.finding_id",
)


def render(
    entries: Sequence[Entry], role: Author, candidate_id: str | None = None
) -> list[dict[str, Any]]:
    """The JSON-able slice of the worksheet a role's payload carries.

    The researcher and the analyst see the whole unit, because between them they
    decide what it still needs. The verifier sees only the candidate it was asked
    about: shown the other candidates or the searches behind them, its verdict
    would rest on the researcher's shortlist rather than on the provision.
    """
    if role != "verifier":
        return [entry.model_dump(mode="json") for entry in entries]
    return [
        _verifier_entry(entry)
        for entry in entries
        if isinstance(entry, UserNoteEntry) or _concerns(entries, entry, candidate_id)
    ]


def verifier_visible_entry_ids(
    entries: Sequence[Entry], candidate_id: str
) -> frozenset[str]:
    return frozenset(
        entry.id
        for entry in entries
        if isinstance(entry, UserNoteEntry) or _concerns(entries, entry, candidate_id)
    )


def known_locators(entries: Sequence[Entry], candidate_id: str) -> frozenset[str]:
    """The locators this worksheet already names for one candidate.

    The verifier's reads are confined to exactly these locators, so it can follow
    recorded evidence without browsing the candidate's act on its own.
    """
    known: set[str] = set()
    for entry in entries:
        if isinstance(entry, CandidateEntry) and entry.id == candidate_id:
            known.add(entry.locator)
            known.update(entry.supporting_locators)
        elif isinstance(entry, CharacterEntry) and entry.candidate_id == candidate_id:
            known.update(item.locator for item in entry.character.evidence)
    return frozenset(known)


def _concerns(entries: Sequence[Entry], entry: Entry, candidate_id: str | None) -> bool:
    if candidate_id is None:
        return False
    if isinstance(entry, CandidateEntry):
        return entry.id == candidate_id
    if isinstance(entry, CharacterEntry | VerdictEntry):
        return entry.candidate_id == candidate_id
    return False


def _verifier_entry(entry: Entry) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": entry.id,
        "seq": entry.seq,
        "author": entry.author,
        "kind": entry.kind,
    }
    if isinstance(entry, CandidateEntry):
        return {
            **base,
            "locator": entry.locator,
            "act_identifier": entry.act_identifier,
            "supporting_locators": list(entry.supporting_locators),
        }
    if isinstance(entry, CharacterEntry):
        return {
            **base,
            "candidate_id": entry.candidate_id,
            "evidence_locators": [item.locator for item in entry.character.evidence],
        }
    return entry.model_dump(mode="json")
