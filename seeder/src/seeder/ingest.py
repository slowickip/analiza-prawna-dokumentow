"""Read a manifest into legal units, without a store.

The seeder's own ingestion with the concurrency taken out: same validators, same
segmentation, same unit construction, fetched in order over a plain client, and
every one of those imported from the build path rather than restated here. It
returns the units instead of streaming them into Qdrant, so what a manifest means
can be checked without standing a collection up.

One decision it does not make: the build path probes an act's HTML and falls back
to its PDF when the routes are unreachable. There is nothing to fall back from
behind a fixture client, so an act is always read as its manifest declares.
Exercising the fallback needs ``seed_qdrant_async``, which is where this helper
should eventually go altogether.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from contract_analyzer.corpus import (
    ActCurrency,
    CorpusBuildError,
    CorpusManifest,
    LegalUnit,
    act_force_state,
    act_identifier,
    eli_act_url,
    load_manifest,
    path_to_url,
    sha256_bytes,
    validate_act_metadata,
    validate_amendments_not_carried,
)
from contract_analyzer.corpus import (
    act_url as act_text_url,
)
from contract_analyzer.corpus.eli import _fetch, _fetch_base_act_metadata
from seeder.eli import (
    decode_metadata,
    decode_struct,
    deduped_struct_article_paths,
    html_unit_from_article,
    parse_pdf_units,
)


def fetch_manifest_units(
    manifest_path: Path,
    client: httpx.Client,
) -> tuple[CorpusManifest, str, list[LegalUnit], list[ActCurrency]]:
    """The manifest, its hash, its units and how each act stood when they were read."""
    manifest = load_manifest(manifest_path)
    manifest_hash = sha256_bytes(manifest_path.read_bytes())
    observed_on = datetime.now(tz=UTC).date()
    units: list[LegalUnit] = []
    base_act_cache: dict[str, dict[str, Any]] = {}
    act_currency: list[ActCurrency] = []
    seen_ids: set[str] = set()

    for act in manifest.acts:
        url = act_text_url(act.publisher, act.year, act.position)
        act_id = act_identifier(act.publisher, act.year, act.position)
        metadata_bytes = _fetch(client, url, accept="application/json")
        validate_act_metadata(decode_metadata(metadata_bytes, url), act, url)
        base_metadata = _fetch_base_act_metadata(client, act.base_act, base_act_cache)
        validate_amendments_not_carried(
            act=act,
            act_id=act_id,
            base_metadata=base_metadata,
            corpus_target_date=manifest.corpus_target_date,
        )
        act_currency.append(
            ActCurrency(
                act_identifier=act_id,
                base_act=act.base_act,
                legal_status_date=act.legal_status_date,
                amendments_not_carried=act.amendments_not_carried,
                repeated_articles=act.repeated_articles,
                source_format_declared=act.source_format,
                source_format_used=act.source_format,
            )
        )
        act_force = act_force_state(
            base_metadata, observed_on, eli_act_url(act.base_act)
        )

        if act.source_format == "html":
            struct_raw = _fetch(client, f"{url}/struct", accept="application/json")
            entries = deduped_struct_article_paths(decode_struct(struct_raw), act_id)
            if not entries:
                raise CorpusBuildError(
                    "malformed_structure", "malformed_structure: no units"
                )
            for path, node in entries:
                unit = html_unit_from_article(
                    act_url=url,
                    path=path,
                    node=node,
                    act=act,
                    act_force=act_force,
                    act_id=act_id,
                    html_bytes=_fetch(
                        client, path_to_url(url, path), accept="text/html"
                    ),
                )
                if unit.id in seen_ids:
                    raise CorpusBuildError(
                        "duplicate_unit_id", f"duplicate_unit_id: {unit.id}"
                    )
                seen_ids.add(unit.id)
                units.append(unit)
        else:
            units.extend(
                parse_pdf_units(
                    pdf_bytes=_fetch(
                        client, f"{url}/text.pdf", accept="application/pdf"
                    ),
                    act=act,
                    act_id=act_id,
                    act_url=url,
                    act_force=act_force,
                    seen_ids=seen_ids,
                )[0]
            )

    return manifest, manifest_hash, units, act_currency
