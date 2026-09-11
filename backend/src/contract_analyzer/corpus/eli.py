"""ELI fetching and act metadata validation."""

from __future__ import annotations

import json
from datetime import date
from typing import Any, Literal

import httpx

from contract_analyzer.corpus.errors import CorpusBuildError
from contract_analyzer.corpus.models import ManifestAct

ELI_BASE = "https://api.sejm.gov.pl/eli/acts"


def act_identifier(publisher: str, year: int, position: int) -> str:
    return f"{publisher}/{year}/{position}"


def act_url(publisher: str, year: int, position: int) -> str:
    return f"{ELI_BASE}/{publisher}/{year}/{position}"


def eli_act_url(act_id: str) -> str:
    return f"{ELI_BASE}/{act_id}"


def _parse_metadata_date(value: Any) -> date | None:
    if not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _base_act_from_consolidation(metadata: dict[str, Any]) -> str | None:
    references = metadata.get("references")
    if not isinstance(references, dict):
        return None
    tekst = references.get("Tekst jednolity dla aktu")
    if not isinstance(tekst, list) or not tekst:
        return None
    first = tekst[0]
    if not isinstance(first, dict):
        return None
    act_id = first.get("id")
    if not isinstance(act_id, str) or not act_id:
        return None
    return act_id


def validate_legal_status_date(
    metadata: dict[str, Any], act: ManifestAct, act_url: str
) -> None:
    eli_date = _parse_metadata_date(metadata.get("legalStatusDate"))
    if eli_date != act.legal_status_date:
        raise CorpusBuildError(
            "legal_status_date_drift",
            (
                f"legal_status_date_drift: pinned {act.legal_status_date}, "
                f"ELI reports {eli_date} for {act_url}"
            ),
        )


def validate_base_act(metadata: dict[str, Any], act: ManifestAct, act_url: str) -> None:
    eli_base = _base_act_from_consolidation(metadata)
    if eli_base != act.base_act:
        raise CorpusBuildError(
            "base_act_mismatch",
            (
                f"base_act_mismatch: pinned {act.base_act!r}, "
                f"ELI reports {eli_base!r} for {act_url}"
            ),
        )


def _parse_amending_acts(
    base_metadata: dict[str, Any], base_act: str
) -> list[tuple[str, date]]:
    references = base_metadata.get("references")
    if not isinstance(references, dict):
        raise CorpusBuildError(
            "malformed_amendment_relation",
            (
                f"malformed_amendment_relation: {base_act} references "
                "missing or not a dict"
            ),
        )
    amending = references.get("Akty zmieniające")
    if not isinstance(amending, list):
        raise CorpusBuildError(
            "malformed_amendment_relation",
            (
                f"malformed_amendment_relation: {base_act} Akty zmieniające "
                "missing or not a list"
            ),
        )
    result: list[tuple[str, date]] = []
    for index, entry in enumerate(amending):
        if not isinstance(entry, dict):
            raise CorpusBuildError(
                "malformed_amendment_relation",
                (
                    f"malformed_amendment_relation: {base_act} entry {index} "
                    "is not a dict"
                ),
            )
        entry_id = entry.get("id")
        if not isinstance(entry_id, str) or not entry_id:
            raise CorpusBuildError(
                "malformed_amendment_relation",
                (
                    f"malformed_amendment_relation: {base_act} entry {index} "
                    "missing string id"
                ),
            )
        entry_date = _parse_metadata_date(entry.get("date"))
        if entry_date is None:
            # Window membership cannot be decided without parsing the date.
            raise CorpusBuildError(
                "malformed_amendment_relation",
                (
                    f"malformed_amendment_relation: {base_act} entry {entry_id} "
                    "has unparseable date"
                ),
            )
        result.append((entry_id, entry_date))
    return result


def _amendments_in_window(
    base_metadata: dict[str, Any],
    base_act: str,
    legal_status_date: date,
    corpus_target_date: date,
) -> set[tuple[str, date]]:
    """Amendments with ELI relation dates after compilation and on or before assessment.

    The lower bound is half-open: a relation dated on legal_status_date is already
    reflected in the consolidated text and is excluded.
    """
    return {
        (entry_id, entry_date)
        for entry_id, entry_date in _parse_amending_acts(base_metadata, base_act)
        if legal_status_date < entry_date <= corpus_target_date
    }


def validate_amendments_not_carried(
    *,
    act: ManifestAct,
    act_id: str,
    base_metadata: dict[str, Any],
    corpus_target_date: date,
) -> None:
    recorded = {
        (amendment.id, amendment.eli_relation_date)
        for amendment in act.amendments_not_carried
    }
    eli_reports = _amendments_in_window(
        base_metadata,
        act.base_act,
        act.legal_status_date,
        corpus_target_date,
    )
    if recorded != eli_reports:
        raise CorpusBuildError(
            "amendment_drift",
            (
                f"amendment_drift: recorded {sorted(recorded)}, "
                f"ELI reports {sorted(eli_reports)} for {act_id}"
            ),
        )


def _fetch_base_act_metadata(
    client: httpx.Client,
    base_act: str,
    cache: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if base_act in cache:
        return cache[base_act]
    url = eli_act_url(base_act)
    metadata_body = _fetch(client, url, accept="application/json")
    try:
        metadata = json.loads(metadata_body)
    except json.JSONDecodeError as exc:
        raise CorpusBuildError(
            "malformed_metadata",
            f"malformed_metadata: {url}",
        ) from exc
    if not isinstance(metadata, dict):
        raise CorpusBuildError("malformed_metadata", f"malformed_metadata: {url}")
    cache[base_act] = metadata
    return metadata


def _fetch(
    client: httpx.Client,
    url: str,
    *,
    accept: str,
) -> bytes:
    response = client.get(url, headers={"Accept": accept})
    if response.status_code != 200:
        raise CorpusBuildError(
            "source_fetch_failed",
            f"source_fetch_failed: {url} returned {response.status_code}",
        )
    body = response.content
    if not body:
        raise CorpusBuildError("empty_source_response", f"empty_source_response: {url}")
    return body


def validate_source_format(
    metadata: dict[str, Any], expected: Literal["html", "pdf"]
) -> None:
    has_html = bool(metadata.get("textHTML"))
    has_pdf = bool(metadata.get("textPDF"))
    if expected == "html" and not has_html:
        raise CorpusBuildError(
            "unexpected_source_format",
            "unexpected_source_format: act metadata lacks HTML text",
        )
    if expected == "pdf" and not has_pdf:
        raise CorpusBuildError(
            "unexpected_source_format",
            "unexpected_source_format: act metadata lacks PDF text",
        )


def validate_act_metadata(
    metadata: dict[str, Any],
    act: ManifestAct,
    act_url: str,
) -> None:
    """Validate that act metadata matches the manifest expectations."""
    validate_source_format(metadata, act.source_format)
    validate_legal_status_date(metadata, act, act_url)
    validate_base_act(metadata, act, act_url)
