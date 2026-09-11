"""Act and provision force-state classification."""

from __future__ import annotations

import re
from datetime import date
from typing import Any

from contract_analyzer.domain import ForceScope, ForceState, ForceValue

_SUPERSCRIPT_DIGITS = "⁰¹²³⁴⁵⁶⁷⁸⁹"

_EDITORIAL_REPEAL_ARTICLE_HEADING = re.compile(
    rf"^\s*(?:Art\.|§)\s*\d+(?:[-–]\d+)?(?:\(\d+\)|\[\d+\]|\^\d+|[{_SUPERSCRIPT_DIGITS}]+)?[a-z]?\.?\s*",
    re.IGNORECASE,
)
_EDITORIAL_REPEAL_PHRASE = (
    r"\(\s*(?:uchylon[aye]|skreślon[aye]|utraci[łl][aye]?\s+moc)\s*\)"
    rf"(?:\s*(?:\d+\)|\[\d+\]|\^\d+|[{_SUPERSCRIPT_DIGITS}]+))?"
)
_EDITORIAL_REPEAL_SUBUNIT_PREFIX = (
    rf"(?:"
    rf"§\s*\d+(?:[-–]\d+)?(?:\(\d+\)|\[\d+\]|\^\d+|[{_SUPERSCRIPT_DIGITS}]+)?[a-z]?\.?"
    rf"|\d+[a-z]?\."
    rf"|\d+[a-z]?\)"
    rf"|[a-z]\)"
    rf")"
)
_EDITORIAL_REPEAL_SUBUNIT = (
    rf"(?:{_EDITORIAL_REPEAL_SUBUNIT_PREFIX}\s*)?{_EDITORIAL_REPEAL_PHRASE}\s*\.?"
)
_ALL_EDITORIAL_REPEAL_SUBUNITS = re.compile(
    rf"^(?:\s*{_EDITORIAL_REPEAL_SUBUNIT})+\s*$",
    re.IGNORECASE,
)
_LEADING_EDITORIAL_REPEAL_SUBUNITS = re.compile(
    rf"^(?:\s*{_EDITORIAL_REPEAL_SUBUNIT})+\s*",
    re.IGNORECASE,
)
_PAGE_FURNITURE_LINE = re.compile(
    r"^[ \t]*Dziennik\s+Ustaw\s*[-–—]\s*\d+\s*[-–—]\s*Poz\.\s*\d+[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)
_FOOTNOTE_DEF_LINE = re.compile(
    r"^[ \t]*\d+\)\s+.*$",
    re.MULTILINE,
)
_EDITORIAL_NOTE_LINE = re.compile(
    r"^[ \t]*(?:"
    r"Utraci[łl]\s+moc|Uznan[ya]|Z\s+dniem|W\s+tym\s+brzmieniu|W\s+brzmieniu"
    r"|Przez\s+art\."
    r").*$",
    re.IGNORECASE | re.MULTILINE,
)
_STRUCTURAL_HEADING_LINE = re.compile(
    r"^[ \t]*(?:"
    r"Rozdzia[łl]|Tytu[łl]|Dzia[łl]|Oddzia[łl]|Ksi[ęe]ga|Cz[ęe][śs][ćc]"
    r")\b.*$",
    re.IGNORECASE | re.MULTILINE,
)
_GLUED_INLINE_HEADING_OR_FURNITURE = re.compile(
    r"(?<=\))\s+(?:Dziennik\s+Ustaw\s*[-–—]\s*\d+\s*[-–—]\s*Poz\.\s*\d+|(?:Rozdzia[łl]|Tytu[łl]|Dzia[łl]|Oddzia[łl]|Ksi[ęe]ga|Cz[ęe][śs][ćc])\b.*)$",
    re.IGNORECASE,
)


def _parse_in_force(value: str | None) -> ForceValue:
    if value == "IN_FORCE":
        return ForceValue.IN_FORCE
    if value == "NOT_IN_FORCE":
        return ForceValue.NOT_IN_FORCE
    return ForceValue.UNDETERMINED


def act_force_state(
    metadata: dict[str, Any], snapshot_date: date, locator: str
) -> ForceState:
    return ForceState(
        value=_parse_in_force(metadata.get("inForce")),
        scope=ForceScope.ACT,
        snapshot_date=snapshot_date,
        source_locator=locator,
    )


def _is_editorial_repeal_placeholder(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False

    body = stripped
    match_heading = _EDITORIAL_REPEAL_ARTICLE_HEADING.match(body)
    if match_heading is not None:
        body = body[match_heading.end() :].strip()

    if body and _ALL_EDITORIAL_REPEAL_SUBUNITS.match(body):
        return True

    match_leading = _LEADING_EDITORIAL_REPEAL_SUBUNITS.match(body)
    if match_leading is None:
        return False

    trailing = body[match_leading.end() :].strip()
    if not trailing:
        return True

    if re.search(r"\bArt\.\s*\d+", trailing):
        return False

    cleaned_trailing = trailing
    cleaned_trailing = _PAGE_FURNITURE_LINE.sub("", cleaned_trailing)
    cleaned_trailing = _FOOTNOTE_DEF_LINE.sub("", cleaned_trailing)
    cleaned_trailing = _EDITORIAL_NOTE_LINE.sub("", cleaned_trailing)
    cleaned_trailing = _STRUCTURAL_HEADING_LINE.sub("", cleaned_trailing)
    cleaned_trailing = _GLUED_INLINE_HEADING_OR_FURNITURE.sub("", cleaned_trailing)

    remaining_lines = [
        line.strip() for line in cleaned_trailing.split("\n") if line.strip()
    ]
    if not remaining_lines:
        return True

    for line in remaining_lines:
        if re.match(r"^(?:§|\d+[\.\)]|[a-z]\))", line):
            return False
        if line.endswith(";") or (line.endswith(".") and len(line) > 100):
            return False

    return True


def provision_force_state(
    *,
    text: str,
    snapshot_date: date,
    locator: str,
) -> ForceState:
    if _is_editorial_repeal_placeholder(text):
        return ForceState(
            value=ForceValue.NOT_IN_FORCE,
            scope=ForceScope.PROVISION,
            snapshot_date=snapshot_date,
            source_locator=locator,
        )
    return ForceState(
        value=ForceValue.UNDETERMINED,
        scope=ForceScope.PROVISION,
        snapshot_date=snapshot_date,
        source_locator=locator,
    )
