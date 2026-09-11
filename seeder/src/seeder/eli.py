"""Fetching from ELI, and cutting one act into legal units.

Both source formats live here: the article paths under ``text.html`` and the
canonical article numbers in ``text.pdf``. Which one a build reads is
``pipeline``'s decision, not this module's.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import logging
import os
import re
import uuid
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx
import pdfplumber

from contract_analyzer.act_pdf import (
    canonical_article_number,
    extract_pdf_pages_text,
    iter_pdf_headings,
    pdf_article_locator,
    pdf_article_unit_id,
)
from contract_analyzer.corpus import (
    CorpusBuildError,
    LegalUnit,
    ManifestAct,
    RepeatedArticleCount,
    html_to_text,
    is_childless_stub_of,
    node_tree_fingerprint,
    path_to_url,
    provision_force_state,
    repeated_articles_in,
    sha256_bytes,
    validate_expected_hash,
    validate_repeated_articles,
)
from contract_analyzer.corpus.html import (
    STRUCT_SEGMENTS,
    article_identifier_from_title,
)
from contract_analyzer.domain import ForceState

logger = logging.getLogger(__name__)

ELI_BASE = "https://api.sejm.gov.pl"


def _is_valid_eli_content(content: bytes, accept: str) -> bool:
    """Whether a body is the kind of document its Accept header asked for.

    ELI answers an outage with 200 and an empty or truncated body, so a status
    code is not enough to decide a response is usable.
    """
    if not content or not content.strip():
        return False
    if "application/json" in accept:
        try:
            parsed = json.loads(content.decode("utf-8"))
            if not isinstance(parsed, (dict, list)):
                return False
        except Exception:
            return False
    elif "application/pdf" in accept:
        if not content.startswith(b"%PDF"):
            return False
    elif "text/html" in accept:
        if len(content.strip()) < 10:
            return False
    return True


def _atomic_write_cache(cache_path: Path, content: bytes) -> None:
    """Write through a unique temporary file, so parallel workers cannot tear one."""
    temp_path = cache_path.with_name(
        f"{cache_path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"
    )
    try:
        temp_path.write_bytes(content)
        os.replace(temp_path, cache_path)
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


async def fetch_async(
    client: httpx.AsyncClient,
    url: str,
    accept: str,
    max_attempts: int = 5,
    *,
    cache_dir: Path | None = None,
    refresh_cache: bool = False,
    no_cache: bool = False,
) -> bytes:
    """Fetch one ELI document, through the disk cache when one is configured.

    Raises ``CorpusBuildError('eli_unavailable')`` once the attempts are spent,
    which is the signal the build reads as "try this act's other source format".
    """
    effective_cache_dir: Path | None = None
    if not no_cache:
        if cache_dir is not None:
            effective_cache_dir = cache_dir
        elif "ELI_CACHE_DIR" in os.environ:
            effective_cache_dir = Path(os.environ["ELI_CACHE_DIR"])

    cache_path: Path | None = None
    if effective_cache_dir is not None:
        effective_cache_dir.mkdir(parents=True, exist_ok=True)
        key = hashlib.sha256(f"{url}|{accept}".encode()).hexdigest()
        cache_path = effective_cache_dir / f"{key}.bin"
        if cache_path.is_file() and not refresh_cache:
            try:
                cached_bytes = cache_path.read_bytes()
                if _is_valid_eli_content(cached_bytes, accept):
                    logger.debug("ELI cache hit for %s", url)
                    return cached_bytes
                logger.warning(
                    "Cached ELI response for %s failed validation; evicting", url
                )
                cache_path.unlink(missing_ok=True)
            except OSError:
                pass

    for attempt in range(max_attempts):
        try:
            resp = await client.get(url, headers={"Accept": accept})
            resp.raise_for_status()
            content = resp.content
            if not _is_valid_eli_content(content, accept):
                raise httpx.RequestError(
                    f"Malformed or empty ELI response from {url} for {accept}"
                )
            if cache_path is not None:
                _atomic_write_cache(cache_path, content)
            return content
        except (httpx.HTTPError, httpx.TimeoutException) as exc:
            if attempt + 1 == max_attempts:
                logger.error(
                    "ELI fetch failed after %d attempts for %s: %s",
                    max_attempts,
                    url,
                    exc,
                )
                raise CorpusBuildError(
                    "eli_unavailable",
                    f"eli_unavailable: {url}: {exc}",
                ) from exc
            backoff = 0.5 * (2**attempt)
            logger.warning(
                "ELI fetch attempt %d/%d failed for %s: %s. Retrying in %.1fs...",
                attempt + 1,
                max_attempts,
                url,
                exc,
                backoff,
            )
            await asyncio.sleep(backoff)
    raise RuntimeError("unreachable")


def struct_article_entries(
    nodes: list[dict[str, Any]],
    *,
    prefix: tuple[str, ...] = (),
) -> list[tuple[tuple[str, ...], dict[str, Any]]]:
    """Every article in a struct tree, whole, and nothing else.

    Only an ``arti`` is emitted, so an article arrives with its paragraphs, points
    and letters included, and the announcement, amendment lists and annex
    fragments around them stay out. None of those is operative law and the PDF
    reader never had them, so excluding them is what makes one manifest describe
    the same corpus through either source format.
    """
    entries: list[tuple[tuple[str, ...], dict[str, Any]]] = []
    for node in nodes:
        node_type = node.get("type")
        if not isinstance(node_type, str):
            continue
        if node_type not in STRUCT_SEGMENTS:
            children = node.get("children")
            if isinstance(children, list):
                entries.extend(struct_article_entries(children, prefix=prefix))
            continue
        name = node.get("name")
        if not isinstance(name, str):
            continue
        current = (*prefix, f"{node_type}={quote(name, safe='()_')}")

        if node_type == "arti":
            entries.append((current, node))
            continue

        children = node.get("children")
        if isinstance(children, list) and children:
            entries.extend(struct_article_entries(children, prefix=current))
    return entries


def deduped_struct_article_paths(
    nodes: list[dict[str, Any]],
    act_id: str,
    *,
    prefix: tuple[str, ...] = (),
) -> list[tuple[tuple[str, ...], dict[str, Any]]]:
    """Article paths in document order, with identical repeats collapsed.

    ELI reaches one path more than once: a parent can list the same child twice,
    and whole branches are duplicated. A verbatim repeat is safe to collapse, but
    two *different* nodes under one path are two provisions, so this compares the
    subtree rather than the path and refuses the ambiguity.

    One repeat is neither: the same article listed once whole and once as a bare
    heading, agreeing in every field including ELI's own id and differing only in
    whether it carries its subtree. The copy with the text wins, whichever order
    the two arrive in. Refusing that pair would reject an act for saying the same
    thing twice; collapsing it by path alone would drop a provision when the two
    really do differ.
    """
    deduped: list[tuple[tuple[str, ...], dict[str, Any]]] = []
    seen: dict[tuple[str, ...], tuple[str, int]] = {}
    for path, node in struct_article_entries(nodes, prefix=prefix):
        fingerprint = node_tree_fingerprint(node)
        if path in seen:
            kept_fingerprint, index = seen[path]
            if kept_fingerprint == fingerprint:
                continue
            kept = deduped[index][1]
            if is_childless_stub_of(node, kept):
                continue
            if is_childless_stub_of(kept, node):
                deduped[index] = (path, node)
                seen[path] = (fingerprint, index)
                continue
            raise CorpusBuildError(
                "ambiguous_struct_path",
                f"ambiguous_struct_path: {act_id} {'/'.join(path)}",
            )
        seen[path] = (fingerprint, len(deduped))
        deduped.append((path, node))
    return deduped


# An annex opens its own block and belongs to no article. Every other article is
# bounded by the next heading, so only the closing one needs this: without it, it
# runs to the end of the file and swallows the appendices.
_ANNEX_HEADING = re.compile(r"^\s*Załącznik(?:i)?\b", re.MULTILINE)


def _articles_end(text: str, last_start: int) -> int:
    """Where the act's articles stop, which is not always where the file stops."""
    annex = _ANNEX_HEADING.search(text, last_start)
    return annex.start() if annex is not None else len(text)


def parse_pdf_units(
    pdf_bytes: bytes,
    act: ManifestAct,
    act_id: str,
    act_url: str,
    act_force: ForceState,
    seen_ids: set[str],
    pins_apply: bool = True,
) -> tuple[list[LegalUnit], tuple[RepeatedArticleCount, ...]]:
    """One act's articles from its PDF, and which numbers it prints more than once.

    ``pins_apply`` is false when this reader is the fallback for an act declaring
    html. Such an act may carry neither ``expected_hash`` nor ``repeated_articles``
    -- the manifest schema rejects both -- so there is no claim to check against a
    document the manifest was never describing. The repeats are returned either
    way, so the snapshot can record what this document actually did.
    """
    if pins_apply:
        validate_expected_hash(pdf_bytes, act.expected_hash)
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            text = extract_pdf_pages_text(pdf)
    except Exception as exc:
        raise CorpusBuildError(
            "malformed_structure", f"malformed_structure: PDF unreadable: {exc}"
        ) from exc
    if not text.strip():
        raise CorpusBuildError("empty_source_response", "empty PDF response")

    headings = list(iter_pdf_headings(text))
    if not headings:
        raise CorpusBuildError("malformed_structure", "no articles in PDF")

    article_numbers = [
        canonical_article_number(raw_identifier.strip())
        for _, raw_identifier in headings
    ]
    repeated = repeated_articles_in(article_numbers)
    if pins_apply:
        validate_repeated_articles(
            act=act, act_id=act_id, article_numbers=article_numbers
        )
    elif repeated:
        logger.info(
            "%s: PDF prints %s more than once; recording what the document does, "
            "because its manifest declares html and pins nothing about this PDF",
            act_id,
            ", ".join(f"art. {entry.article}" for entry in repeated),
        )

    # Canonical numbers, not positions: a positional id changes whenever the
    # splitter improves, and the citation check asks whether a locator resolves to
    # the provision it names -- the article, and which printing of it.
    numbered = [
        (canonical_article_number(raw.strip()), start) for start, raw in headings
    ]
    printings = Counter(number for number, _ in numbered)
    seen_printings: Counter[str] = Counter()

    units: list[LegalUnit] = []
    for index, (article_number, start) in enumerate(numbered):
        end = (
            headings[index + 1][0]
            if index + 1 < len(headings)
            else _articles_end(text, start)
        )
        article_text = text[start:end].strip()
        if not article_text:
            raise CorpusBuildError(
                "malformed_structure",
                (
                    f"malformed_structure: empty article text for "
                    f"{article_number} in {act_id}"
                ),
            )
        printing: int | None = None
        if printings[article_number] > 1:
            seen_printings[article_number] += 1
            printing = seen_printings[article_number]
        locator = pdf_article_locator(act_url, article_number, printing=printing)
        unit_id = pdf_article_unit_id(act_id, article_number, printing=printing)
        if unit_id in seen_ids:
            raise CorpusBuildError("duplicate_unit_id", f"duplicate: {unit_id}")
        seen_ids.add(unit_id)

        units.append(
            LegalUnit(
                id=unit_id,
                locator=locator,
                act_identifier=act_id,
                article_identifier=f"Art. {article_number}",
                text=article_text,
                content_hash=sha256_bytes(article_text.encode("utf-8")),
                act_force=act_force,
                provision_force=provision_force_state(
                    text=article_text,
                    snapshot_date=act.legal_status_date,
                    locator=locator,
                ),
                legal_status_date=act.legal_status_date,
            )
        )
    logger.info("%s: %d articles read from PDF", act_id, len(units))
    return units, repeated


def html_unit_from_article(
    *,
    act_url: str,
    path: tuple[str, ...],
    node: dict[str, Any],
    act: ManifestAct,
    act_force: ForceState,
    act_id: str,
    html_bytes: bytes,
) -> LegalUnit:
    """One legal unit from one article's HTML.

    Fetching is the caller's: the build fetches concurrently on the event loop
    and the manifest reader fetches in order. What a unit *is* must not differ
    between them, so it is decided here, once.
    """
    url = path_to_url(act_url, path)
    text = html_to_text(html_bytes)
    if not text.strip():
        raise CorpusBuildError("empty_source_response", f"empty_source_response: {url}")
    title = node.get("title")
    article_identifier = (
        article_identifier_from_title(title)
        if isinstance(title, str)
        else "/".join(path)
    )
    return LegalUnit(
        id=f"{act_id}:{'/'.join(path)}",
        locator=url,
        act_identifier=act_id,
        article_identifier=article_identifier,
        text=text,
        content_hash=sha256_bytes(text.encode("utf-8")),
        act_force=act_force,
        provision_force=provision_force_state(
            text=text,
            snapshot_date=act.legal_status_date,
            locator=url,
        ),
        legal_status_date=act.legal_status_date,
    )


def decode_struct(raw: bytes) -> list[dict[str, Any]]:
    """Decode and validate act structure JSON bytes."""
    try:
        struct = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise CorpusBuildError(
            "malformed_structure", "malformed_structure: struct JSON"
        ) from exc
    if not isinstance(struct, list):
        raise CorpusBuildError("malformed_structure", "malformed_structure: not list")
    return struct


def decode_metadata(raw: bytes, url: str) -> dict[str, Any]:
    """Decode and validate act metadata JSON bytes."""
    try:
        metadata = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise CorpusBuildError(
            "malformed_metadata", f"malformed_metadata: {url}"
        ) from exc
    if not isinstance(metadata, dict):
        raise CorpusBuildError("malformed_metadata", f"malformed_metadata: {url}")
    return metadata
