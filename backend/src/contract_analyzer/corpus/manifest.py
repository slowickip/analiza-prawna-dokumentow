"""Corpus manifest validation and loading."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from pydantic import ValidationError

from contract_analyzer.corpus.eli import act_identifier
from contract_analyzer.corpus.errors import CorpusBuildError
from contract_analyzer.corpus.models import (
    CorpusManifest,
    ManifestAct,
    RepeatedArticleCount,
)


def _validate_corpus_target_date(manifest: CorpusManifest) -> None:
    for act in manifest.acts:
        if manifest.corpus_target_date < act.legal_status_date:
            act_id = act_identifier(act.publisher, act.year, act.position)
            raise CorpusBuildError(
                "target_date_before_text",
                (
                    f"target_date_before_text: corpus_target_date "
                    f"{manifest.corpus_target_date} is before legal_status_date "
                    f"{act.legal_status_date} for {act_id}"
                ),
            )


def _validate_repeated_articles_applicable(manifest: CorpusManifest) -> None:
    for act in manifest.acts:
        if act.source_format == "html" and act.repeated_articles:
            act_id = act_identifier(act.publisher, act.year, act.position)
            raise CorpusBuildError(
                "repeated_articles_not_applicable",
                f"repeated_articles_not_applicable: {act_id}",
            )


def _validate_expected_hash_applicable(manifest: CorpusManifest) -> None:
    """An act declaring HTML has no single document for a hash to be about.

    The PDF reader fetches one document and can check it; the HTML reader fetches
    a structure and an article per path. Left allowed, the pin was ignored while
    HTML answered and then compared against PDF bytes when the fallback ran --
    a guaranteed mismatch, reported as source_hash_mismatch.
    """
    for act in manifest.acts:
        if act.source_format == "html" and act.expected_hash is not None:
            act_id = act_identifier(act.publisher, act.year, act.position)
            raise CorpusBuildError(
                "expected_hash_not_applicable",
                f"expected_hash_not_applicable: {act_id}",
            )


def _validate_duplicate_act_entries(manifest: CorpusManifest) -> None:
    seen: set[tuple[str, int, int]] = set()
    for act in manifest.acts:
        key = (act.publisher, act.year, act.position)
        if key in seen:
            act_id = act_identifier(act.publisher, act.year, act.position)
            raise CorpusBuildError(
                "duplicate_act_entry",
                f"duplicate_act_entry: {act_id}",
            )
        seen.add(key)


def _validate_duplicate_amendment_entries(act: ManifestAct, act_id: str) -> None:
    seen: set[str] = set()
    for amendment in act.amendments_not_carried:
        if amendment.id in seen:
            raise CorpusBuildError(
                "duplicate_amendment_entry",
                f"duplicate_amendment_entry: {amendment.id} in {act_id}",
            )
        seen.add(amendment.id)


def _validate_repeated_articles_sorted(act: ManifestAct, act_id: str) -> None:
    articles = [entry.article for entry in act.repeated_articles]
    if articles != sorted(articles):
        raise CorpusBuildError(
            "invalid_manifest",
            f"invalid_manifest: repeated_articles not sorted for {act_id}",
        )


def load_manifest(path: Path) -> CorpusManifest:
    """Read a manifest and refuse it unless every claim in it is well formed.

    Version 2 only, and the cross-field rules the schema cannot express are
    checked here: a target date behind an act's text, pins an act's declared
    source format cannot carry, and duplicate acts or amendments.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CorpusBuildError(
            "invalid_manifest",
            f"invalid_manifest: {path}",
        ) from exc
    if not isinstance(raw, dict):
        raise CorpusBuildError("invalid_manifest", f"invalid_manifest: {path}")
    if raw.get("version") != 2:
        raise CorpusBuildError(
            "unsupported_manifest_version",
            f"unsupported_manifest_version: {path}",
        )
    try:
        manifest = CorpusManifest.model_validate(raw)
    except ValidationError as exc:
        raise CorpusBuildError(
            "invalid_manifest",
            f"invalid_manifest: {path}",
        ) from exc
    _validate_corpus_target_date(manifest)
    _validate_repeated_articles_applicable(manifest)
    _validate_expected_hash_applicable(manifest)
    _validate_duplicate_act_entries(manifest)
    for act in manifest.acts:
        act_id = act_identifier(act.publisher, act.year, act.position)
        _validate_duplicate_amendment_entries(act, act_id)
        _validate_repeated_articles_sorted(act, act_id)
    return manifest


def repeated_articles_in(
    article_numbers: Iterable[str],
) -> tuple[RepeatedArticleCount, ...]:
    """Which article numbers this document prints more than once, and how often."""
    return tuple(
        sorted(
            (
                RepeatedArticleCount(article=article, printings=count)
                for article, count in Counter(article_numbers).items()
                if count > 1
            ),
            key=lambda entry: entry.article,
        )
    )


def validate_repeated_articles(
    *,
    act: ManifestAct,
    act_id: str,
    article_numbers: list[str],
) -> None:
    """Check the manifest's pinned repeat counts against the document itself.

    ``repeated_articles`` is a provenance claim: this act really does print this
    article more than once, and the corpus is expected to carry both. Drift runs
    both ways -- a repeat the document stopped showing is as wrong as one it
    started showing.
    """
    actual = repeated_articles_in(article_numbers)
    if actual != act.repeated_articles:
        raise CorpusBuildError(
            "repeated_article_drift",
            (
                f"repeated_article_drift: pinned {list(act.repeated_articles)}, "
                f"computed {list(actual)} for {act_id}"
            ),
        )
