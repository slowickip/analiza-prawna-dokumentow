"""Internal reference parsing and target catalogue."""

from __future__ import annotations

import re
from re import Pattern

from contract_analyzer.domain import (
    ReferenceRecord,
    ReferenceStatus,
    ReferenceType,
    Unit,
)

from .errors import StructureError
from .segmentation import _is_fallback_units

_REFERENCE_PATTERNS: tuple[tuple[ReferenceType, Pattern[str]], ...] = (
    (
        ReferenceType.EXTERNAL_ACT,
        re.compile(
            r"\bart\.\s*\d+[^.;\n]{0,120}?"
            r"(?:kodeks(?:u)?|ustaw(?:y|ie)?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        ReferenceType.FULL_INTERNAL,
        re.compile(
            r"(?:\b(?:zgodnie\s+z|mowa\s+w|o\s+którym\s+mowa\s+w)\s+)?"
            r"(?:(?:§|par\.)\s*(\d+))(?:\s+ust\.\s*\d+)?",
            re.IGNORECASE,
        ),
    ),
    (
        ReferenceType.FULL_INTERNAL,
        re.compile(
            r"\bpkt\.?\s*(\d+(?:\.\d+)+)\b",
            re.IGNORECASE,
        ),
    ),
    (
        ReferenceType.RELATIVE_INTERNAL,
        re.compile(r"\bust\.\s*(\d+)\b", re.IGNORECASE),
    ),
    (
        ReferenceType.DIRECTIONAL,
        re.compile(
            r"\b(?:jak\s+wskazano\s+)?(?:powyżej|poniżej|wyżej|niżej)\b",
            re.IGNORECASE,
        ),
    ),
    (
        ReferenceType.ANNEX,
        re.compile(r"\bzałącznik\s+nr\s*(\d+)\b", re.IGNORECASE),
    ),
    (
        ReferenceType.WHOLE_DOCUMENT,
        re.compile(r"\bniniejsz\w*\s+umow\w*\b", re.IGNORECASE),
    ),
)

_UNIT_OPENING = re.compile(
    r"^\s*(?:(?:§|par\.|[S5$8])\s*\d+|Art\.\s*\d+|\d+(?:\.\d+)*\.?(?=\s)|ust\.\s*\d+|pkt\.\s*\d+|lit\.\s*\w+)",
    re.IGNORECASE,
)

_PARAGRAPH_LEAD = re.compile(
    r"^\s*(?:§|par\.)\s*(\d+)",
    re.IGNORECASE | re.MULTILINE,
)

_OUTLINE_LEAD = re.compile(
    r"^\s*(\d+(?:\.\d+)*)\.?(?=\s)",
    re.MULTILINE,
)


def _unit_target_identity(unit: Unit) -> tuple[str, tuple[int, ...]] | None:
    match_par = _PARAGRAPH_LEAD.search(unit.text)
    if match_par is not None:
        return ("paragraph", (int(match_par.group(1)),))
    match_outline = _OUTLINE_LEAD.search(unit.text)
    if match_outline is not None:
        return (
            "outline",
            tuple(int(part) for part in match_outline.group(1).split(".")),
        )
    return None


def _resolve_internal_target(
    units: list[Unit],
    target: tuple[str, tuple[int, ...]] | None,
) -> str | None:
    if target is None:
        return None
    scheme, path = target
    for unit in units:
        unit_target = _unit_target_identity(unit)
        if unit_target is not None and unit_target == (scheme, path):
            return unit.id
    return None


_STATIC_STATUSES: dict[ReferenceType, ReferenceStatus] = {
    ReferenceType.EXTERNAL_ACT: ReferenceStatus.EXTERNAL_ACT,
    ReferenceType.DIRECTIONAL: ReferenceStatus.AMBIGUOUS_DIRECTIONAL,
    ReferenceType.ANNEX: ReferenceStatus.OUTSIDE_INPUT,
    ReferenceType.WHOLE_DOCUMENT: ReferenceStatus.WHOLE_DOCUMENT,
    ReferenceType.RELATIVE_INTERNAL: ReferenceStatus.WITHIN_UNIT,
}


def _status_for_reference(
    reference_type: ReferenceType,
    *,
    target_unit_id: str | None,
    fallback_mode: bool,
) -> ReferenceStatus:
    static_status = _STATIC_STATUSES.get(reference_type)
    if static_status is not None:
        return static_status
    if fallback_mode or target_unit_id is None:
        return ReferenceStatus.TARGET_DOES_NOT_EXIST
    return ReferenceStatus.RESOLVED


def _is_unit_opening_marker(unit_text: str, match: re.Match[str]) -> bool:
    opening = _UNIT_OPENING.match(unit_text)
    if opening is None:
        return False
    return match.start() == opening.start() and match.end() <= opening.end()


def _find_references_in_unit(
    unit: Unit,
    units: list[Unit],
    *,
    fallback_mode: bool,
) -> list[ReferenceRecord]:
    records: list[ReferenceRecord] = []
    occupied: list[tuple[int, int]] = []

    def claim(start: int, end: int) -> bool:
        for left, right in occupied:
            if start < right and end > left:
                return False
        occupied.append((start, end))
        return True

    for reference_type, pattern in _REFERENCE_PATTERNS:
        for match in pattern.finditer(unit.text):
            if _is_unit_opening_marker(unit.text, match):
                continue
            if not claim(match.start(), match.end()):
                continue
            target: tuple[str, tuple[int, ...]] | None = None
            if reference_type is ReferenceType.FULL_INTERNAL:
                if match.group(1):
                    if "." in match.group(1):
                        target = (
                            "outline",
                            tuple(int(p) for p in match.group(1).split(".")),
                        )
                    else:
                        target = ("paragraph", (int(match.group(1)),))
            classified_type = reference_type
            if classified_type is ReferenceType.RELATIVE_INTERNAL:
                if re.search(
                    r"(?:§|par\.)\s*\d+\s*$",
                    unit.text[: match.start()],
                    re.IGNORECASE,
                ):
                    continue
            target_unit_id = _resolve_internal_target(units, target)
            status = _status_for_reference(
                classified_type,
                target_unit_id=target_unit_id,
                fallback_mode=fallback_mode,
            )
            records.append(
                ReferenceRecord(
                    citing_unit_id=unit.id,
                    reference_type=classified_type,
                    status=status,
                    raw_text=match.group(0),
                    target_unit_id=target_unit_id
                    if status is ReferenceStatus.RESOLVED
                    else None,
                )
            )
    return records


def parse_references(units: list[Unit]) -> list[ReferenceRecord]:
    if not units:
        raise StructureError("no_units", "cannot parse references without units")

    fallback_mode = _is_fallback_units(units)
    records: list[ReferenceRecord] = []
    for unit in units:
        records.extend(
            _find_references_in_unit(unit, units, fallback_mode=fallback_mode)
        )
    return records
