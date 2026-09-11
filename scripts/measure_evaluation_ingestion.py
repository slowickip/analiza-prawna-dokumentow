#!/usr/bin/env python3
"""Measure every evaluation candidate through the artefact's own ingestion.

Runs ingestion, segmentation and reference parsing over each artifact listed in
`evaluation-data/manifest.json` and reports the covariates the protocol needs:
input length, read mode, segmentation mode, and how far the ON arm can diverge
from MID on that document.

No model call is made and no analysis is run. `--check` compares the result with
the measurements recorded in the manifest and fails if anything drifted or could
not be measured on this machine, so a green run means every value was checked.

Requires the project's runtime dependencies. `.doc` conversion additionally
requires LibreOffice (`soffice`) on PATH; the project's container image carries
it. Token counts require a local tokenizer directory passed with --tokenizer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend" / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from contract_analyzer.config import RunConfig  # noqa: E402
from contract_analyzer.ingest import IngestError, IngestService  # noqa: E402
from contract_analyzer.structure import (  # noqa: E402
    StructureError,
    _is_fallback_units,
    _select_structural_markers,
    context_for,
    parse_references,
    segment,
)
from validate_evaluation_data import MEASURED_FIELDS  # noqa: E402


def load_tokenizer(tokenizer_file: str | None) -> Any:
    """The tokenizer the recorded counts were taken with, or None.

    The file is checked against the digest the manifest records for it before the
    loader sees it. The recorded counts are only reproducible in that one
    tokenizer, and the loader panics rather than raising on a malformed model,
    so an unrecognised file is refused here instead of crashing the interpreter.
    """
    if tokenizer_file is None:
        return None

    manifest = json.loads((ROOT / "evaluation-data" / "manifest.json").read_text())
    expected = manifest["measurement"]["tokenizer"]["file_sha256"]
    data = Path(tokenizer_file).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != expected:
        raise SystemExit(
            f"tokenizer file digest {digest} is not the recorded "
            f"{expected}; the counts in the manifest were taken with "
            f"{manifest['measurement']['tokenizer']['name']}"
        )

    from tokenizers import Tokenizer

    return Tokenizer.from_file(tokenizer_file)


def count_tokens(tokenizer: Any, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False).ids)


def measure(path: Path, config: RunConfig, tokenizer: Any) -> dict[str, Any]:
    service = IngestService(config)
    try:
        payload = service.ingest(path.name, path.read_bytes())
    except IngestError as error:
        return {"unavailable": f"{error.code}: {error}"}

    try:
        units = segment(payload, config)
    except StructureError as error:
        return {"unavailable": f"{error.code}: {error}"}

    references = parse_references(units)
    edges = sum(len(context_for(unit.id, units, references)) for unit in units)
    # Two different questions. Which path `segment` took is settled by whether it
    # found structural markers. What the rest of the pipeline believes is settled
    # by `_is_fallback_units`, which infers it from overlapping units and so calls
    # a single-window document structural. The two disagree exactly there, and
    # reference resolution and ON-arm context both follow the second one.
    structural = _select_structural_markers(payload.text, config) is not None
    return {
        "read_mode": payload.read_mode.value if payload.read_mode else "plain_text",
        "chars": len(payload.text),
        "words": len(payload.text.split()),
        "tokens": count_tokens(tokenizer, payload.text) if tokenizer else None,
        "segmentation_mode": "structural" if structural else "window_fallback",
        "pipeline_treats_as_fallback": _is_fallback_units(units),
        "unit_count": len(units),
        "reference_count": len(references),
        "resolved_reference_count": sum(
            1 for record in references if record.status.value == "resolved"
        ),
        "on_context_edges": edges,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tokenizer",
        help="path to a tokenizer.json; without it token counts are not measured",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="compare with the manifest and fail on drift or on anything unmeasured",
    )
    arguments = parser.parse_args()

    data_root = ROOT / "evaluation-data"
    manifest = json.loads((data_root / "manifest.json").read_text(encoding="utf-8"))
    config = RunConfig()
    tokenizer = load_tokenizer(arguments.tokenizer)

    results: dict[str, Any] = {}
    problems: list[str] = []
    for entry in manifest["documents"]:
        identifier = entry["id"]
        measured = measure(data_root / entry["path"], config, tokenizer)
        results[identifier] = measured
        if not arguments.check:
            continue
        if "unavailable" in measured:
            problems.append(
                f"{identifier}: not measured here ({measured['unavailable']})"
            )
            continue
        recorded = entry.get("measured")
        if not isinstance(recorded, dict):
            problems.append(f"{identifier}: manifest records no measurements")
            continue
        for field in MEASURED_FIELDS:
            if measured[field] is None:
                problems.append(f"{identifier}: {field} not measured here")
            elif recorded.get(field) != measured[field]:
                problems.append(
                    f"{identifier}: {field} recorded {recorded.get(field)!r}, "
                    f"measured {measured[field]!r}"
                )

    if not arguments.check:
        json.dump(results, sys.stdout, ensure_ascii=False, indent=2, sort_keys=True)
        print()
        return 0

    if problems:
        print("ingestion measurement check failed:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1
    print(f"ingestion measurements match the manifest for {len(results)} artifacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
