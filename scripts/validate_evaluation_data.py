#!/usr/bin/env python3
"""Validate the evaluation candidate pool using only the standard library.

The pool has two strata with different rules. `synthetic_fixture` entries are
project-original text whose bytes, word counts, reference counts and clean-room
constraints are all recomputed here. `real_source` entries are files published
by third parties and held here as metadata-redacted derivatives: their bytes are
checked against the `stored_*` hash in the acquisition record, and the `acquired_*`
hash records what the publisher served. No clean-room rule applies to them,
because a real contract cites real statutes.

The recorded ingestion measurements are not re-run here -- that needs the runtime
dependencies and LibreOffice. `scripts/measure_evaluation_ingestion.py --check`
is the guard for those.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

WORD_RE = re.compile(r"\b[\wąćęłńóśźżĄĆĘŁŃÓŚŹŻ-]+\b", re.UNICODE)
REFERENCE_RE = re.compile(
    r"(?:§\s*\d+|punkt(?:u|em|cie)?\s+\d+|ust\.\s*\d+|"
    r"załącznik(?:a|u|iem)?\s+(?:nr\s+)?[A-Z0-9–-]+)",
    re.IGNORECASE,
)
PROHIBITED_PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
    "PESEL-like 11-digit value": re.compile(r"(?<!\d)\d{11}(?!\d)"),
    "NIP-like 10-digit value": re.compile(r"(?<!\d)\d{10}(?!\d)"),
    "phone-like 9-digit value": re.compile(
        r"(?<!\d)(?:\+48[\s-]?)?(?:\d[\s-]?){9}(?!\d)"
    ),
    "URL": re.compile(r"https?://|www\.", re.IGNORECASE),
    "explicit article citation": re.compile(r"\bart\.?\s*\d+", re.IGNORECASE),
    "explicit act name": re.compile(
        r"\b(?:kodeks(?:u|ie)?|ustaw(?:a|y|ie|ę)|rozporządzen(?:ie|ia))\b",
        re.IGNORECASE,
    ),
}
REQUIRED_MARKER = "MATERIAŁ SYNTETYCZNY DO EWALUACJI — NIE JEST WZOREM"

# The split, frozen at the level of the logical case. It is written here as well as
# in the manifest deliberately: a guard that reads the split only from the file it is
# checking would go green on any edit to that file. Drift between the two is what this
# table exists to catch. The protocol state lives in evaluation-data/protocol.json;
# changing the split means a new protocol version, not an edit here.
FROZEN_SPLIT = {
    "ED-003": "development",
    "ED-005": "holdout",
    "ED-009": "holdout",
    "RT-001": "holdout",
    "RT-002": "development",
}
# Every retained case is a single artifact from 2.6.23, so this table is empty and
# the canonical-artifact check below has nothing to compare. It stays because the
# check is the thing that would catch a variant being added back without a decision
# about which file the case is measured on.
CANONICAL_OF_MULTI_VARIANT_CASE: dict[str, str] = {}

STRATA = {"synthetic_fixture", "real_source"}
PROVENANCE_KINDS = {"publisher_retrieved", "author_scan"}
RIGHTS_STATUS = {
    "verified_reusable",
    "reuse_conditions_recorded",
    "unsettled",
    "url_only",
    "excluded",
}
PII_STATUS = {
    "not_screened",
    "pattern_screened_and_reviewed",
    "requires_derivative_redaction",
    "excluded_personal_data",
}
MEASURED_FIELDS = (
    "read_mode",
    "chars",
    "words",
    "tokens",
    "segmentation_mode",
    "pipeline_treats_as_fallback",
    "unit_count",
    "reference_count",
    "resolved_reference_count",
    "on_context_edges",
)


class ValidationError(Exception):
    """Raised for a candidate-pool validation failure."""


def fail(message: str) -> None:
    raise ValidationError(message)


def validate_common(
    entry: dict[str, Any], sources: dict[str, Any], seen: set[str]
) -> str:
    identifier = entry.get("id")
    if not isinstance(identifier, str) or not identifier:
        fail("every entry requires a non-empty string id")
    if identifier in seen:
        fail(f"duplicate entry id: {identifier}")
    seen.add(identifier)

    if entry.get("stratum") not in STRATA:
        fail(f"{identifier}: stratum must be one of {sorted(STRATA)}")
    if entry.get("source") not in sources:
        fail(f"{identifier}: source {entry.get('source')!r} is not declared in sources")
    if entry.get("pii_status") not in PII_STATUS:
        fail(f"{identifier}: pii_status must be one of {sorted(PII_STATUS)}")
    case = entry.get("logical_case")
    expected_split = FROZEN_SPLIT.get(case)
    if expected_split is None:
        fail(f"{identifier}: logical_case {case!r} is not in the frozen split")
    if entry.get("split") != expected_split:
        fail(
            f"{identifier}: split is {entry.get('split')!r}, but case {case} is "
            f"frozen as {expected_split!r}"
        )
    if entry.get("adjudication_status") != "not_started":
        fail(f"{identifier}: assessment must not be started here")
    if entry.get("language") != "pl":
        fail(f"{identifier}: language must be pl")

    measured = entry.get("measured")
    if not isinstance(measured, dict):
        fail(f"{identifier}: measured ingestion values are missing")
    for field in MEASURED_FIELDS:
        if field not in measured:
            fail(f"{identifier}: measured.{field} is missing")
        if measured[field] is None:
            fail(f"{identifier}: measured.{field} is null; run the measurement script")
    return identifier


def validate_synthetic(data_root: Path, entry: dict[str, Any], identifier: str) -> Path:
    relative_path = entry.get("path")
    if not isinstance(relative_path, str) or not relative_path.endswith(".txt"):
        fail(f"{identifier}: path must identify a .txt file")
    path = data_root / relative_path
    if not path.is_file():
        fail(f"{identifier}: missing file {relative_path}")

    text = path.read_text(encoding="utf-8")
    raw = text.encode("utf-8")
    if hashlib.sha256(raw).hexdigest() != entry.get("sha256"):
        fail(f"{identifier}: SHA-256 mismatch")
    if len(WORD_RE.findall(text)) != entry.get("word_count"):
        fail(f"{identifier}: word_count mismatch")
    if len(raw) != entry.get("byte_count"):
        fail(f"{identifier}: byte_count mismatch")
    if len(REFERENCE_RE.findall(text)) != entry.get("internal_reference_count"):
        fail(f"{identifier}: internal_reference_count mismatch")

    lines = text.splitlines()
    if not lines or not lines[0].startswith(REQUIRED_MARKER):
        fail(f"{identifier}: synthetic-material marker is missing")
    if entry.get("synthetic") is not True:
        fail(f"{identifier}: synthetic must be true")
    if entry.get("text_contains_personal_data") is not False:
        fail(f"{identifier}: text_contains_personal_data must be false")
    if entry.get("file_metadata_contains_personal_data") is not False:
        fail(f"{identifier}: file_metadata_contains_personal_data must be false")
    if entry.get("rights_basis") != "clean_room_project_original":
        fail(f"{identifier}: unexpected rights basis")
    if entry.get("license") != "CC0-1.0":
        fail(f"{identifier}: unexpected license")

    for label, pattern in PROHIBITED_PATTERNS.items():
        if pattern.search(text):
            fail(f"{identifier}: prohibited {label} found")
    return path.resolve()


def validate_real(
    data_root: Path,
    entry: dict[str, Any],
    identifier: str,
    acquisition: dict[str, dict[str, Any]],
) -> Path:
    relative_path = entry.get("path")
    if not isinstance(relative_path, str) or not relative_path.startswith(
        "source-files/"
    ):
        fail(f"{identifier}: path must sit under source-files/")
    path = data_root / relative_path
    if not path.is_file():
        fail(f"{identifier}: missing file {relative_path}")
    if entry.get("synthetic") is not False:
        fail(f"{identifier}: synthetic must be false")
    if entry.get("provenance_record") != "source-files/acquisition.json":
        fail(f"{identifier}: provenance_record must point at the acquisition record")
    if entry.get("file_metadata_contains_personal_data") is not False:
        fail(f"{identifier}: file metadata carrying personal data violates RNF-05")
    if entry.get("text_contains_personal_data") is not False:
        fail(
            f"{identifier}: extracted text carrying personal data excludes a candidate"
        )

    record = acquisition.get(path.name)
    if record is None:
        fail(f"{identifier}: {path.name} is absent from the acquisition record")
    raw = path.read_bytes()
    # Two hashes, deliberately. `acquired_*` identifies what the publisher served;
    # `stored_*` identifies the derivative actually held here. Only the second can
    # be recomputed from the retained file. An author-made scan has no acquired hash
    # at all: nobody served those bytes, so there is nothing to re-fetch against,
    # and it must instead name the artifact it was made from.
    kind = record.get("provenance_kind")
    if kind not in PROVENANCE_KINDS:
        fail(f"{identifier}: unknown provenance_kind {kind!r}")
    if kind == "author_scan":
        scanned = record.get("scan_of")
        if not scanned:
            fail(
                f"{identifier}: an author scan must name the artifact it was made from"
            )
        if scanned not in acquisition:
            fail(f"{identifier}: scan_of names {scanned}, which is not in the record")
    else:
        for field in ("acquired_sha256", "acquired_byte_count"):
            if not record.get(field):
                fail(f"{identifier}: acquisition record is missing {field}")
    # An empty list is a real answer here -- it means nothing needed blanking.
    if not isinstance(record.get("derivative_fields_blanked"), list):
        fail(f"{identifier}: acquisition record is missing derivative_fields_blanked")
    if len(raw) != record["stored_byte_count"]:
        fail(f"{identifier}: byte count differs from the stored derivative record")
    if hashlib.sha256(raw).hexdigest() != record["stored_sha256"]:
        fail(f"{identifier}: SHA-256 differs from the stored derivative record")
    return path.resolve()


def validate_dataset_level(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    if manifest.get("schema_version") != 2:
        fail("this validator expects manifest schema_version 2")
    if manifest.get("status") != "split_frozen":
        fail("dataset status must be split_frozen once the split is assigned")

    split_policy = manifest.get("split_policy")
    if not isinstance(split_policy, dict):
        fail("split_policy must be an object")
    if split_policy.get("current_state") != "split_frozen":
        fail("split_policy.current_state must be split_frozen")
    if split_policy.get("unit") != "logical_case":
        fail("the split is assigned per logical case, never per artifact")
    protocol = split_policy.get("protocol_document")
    if not protocol:
        fail("split_policy must name the protocol document that froze the split")
    if not (root / protocol).is_file():
        fail(f"split_policy.protocol_document does not exist: {protocol}")
    if split_policy.get("holdout_must_remain_untouched") is not True:
        fail("holdout protection must be explicit")

    sources = manifest.get("sources")
    if not isinstance(sources, dict) or not sources:
        fail("sources must be a non-empty object")
    for key, source in sources.items():
        if source.get("rights_status") not in RIGHTS_STATUS:
            fail(f"source {key}: rights_status must be one of {sorted(RIGHTS_STATUS)}")
        if not source.get("rights_evidence"):
            fail(f"source {key}: rights_evidence must say what the source states")
        if source["rights_status"] != "verified_reusable" and not source.get(
            "next_closing_action"
        ):
            fail(
                f"source {key}: an unsettled rights status needs a next_closing_action"
            )

    length = manifest.get("length_description")
    if not isinstance(length, dict):
        fail("length_description must be an object")
    if not isinstance(length.get("external_reference_tokens"), int):
        fail("length_description.external_reference_tokens must be numeric")
    if not isinstance(
        length.get("external_reference_transferable_to_this_evaluation"), bool
    ):
        fail("length_description must state whether the external reference transfers")
    if not length[
        "external_reference_transferable_to_this_evaluation"
    ] and not length.get("external_reference_limitation"):
        fail("a non-transferable external reference needs its limitation recorded")
    return sources


def validate(root: Path) -> dict[str, Any]:
    data_root = root / "evaluation-data"
    try:
        manifest = json.loads((data_root / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        fail(f"cannot read valid JSON manifest: {error}")
    if not isinstance(manifest, dict):
        fail("manifest root must be an object")

    sources = validate_dataset_level(root, manifest)

    acquisition_raw = json.loads(
        (data_root / "source-files" / "acquisition.json").read_text(encoding="utf-8")
    )
    acquisition = {item["filename"]: item for item in acquisition_raw["files"]}

    entries = manifest.get("documents")
    if not isinstance(entries, list) or not entries:
        fail("manifest documents must be a non-empty array")

    seen_ids: set[str] = set()
    manifest_paths: set[Path] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            fail("every entry must be an object")
        identifier = validate_common(entry, sources, seen_ids)
        if entry["stratum"] == "synthetic_fixture":
            manifest_paths.add(validate_synthetic(data_root, entry, identifier))
        else:
            manifest_paths.add(validate_real(data_root, entry, identifier, acquisition))

    disk_paths = {
        path.resolve()
        for path in list((data_root / "documents").glob("*"))
        + list((data_root / "source-files").glob("*"))
        # `.url` files are locators, not artifacts: RNF-06 asks for exactly those
        # where the bytes may not be redistributed, so a pointer with no manifest
        # entry is a legitimate state rather than a stray file.
        if path.is_file() and path.name != "acquisition.json" and path.suffix != ".url"
    }
    if disk_paths != manifest_paths:
        unlisted = sorted(str(path) for path in disk_paths - manifest_paths)
        missing = sorted(str(path) for path in manifest_paths - disk_paths)
        fail(f"manifest/disk path mismatch; unlisted={unlisted}, missing={missing}")

    coverage = manifest.get("coverage")
    if not isinstance(coverage, dict):
        fail("coverage must be an object")
    cases = {entry["logical_case"] for entry in entries}
    if coverage.get("artifact_count") != len(entries):
        fail("coverage artifact_count mismatch")
    if coverage.get("logical_case_count") != len(cases):
        fail("coverage logical_case_count mismatch")

    if cases != set(FROZEN_SPLIT):
        missing = sorted(set(FROZEN_SPLIT) - cases)
        extra = sorted(cases - set(FROZEN_SPLIT))
        fail(
            f"pool no longer matches the frozen split: missing {missing}, extra {extra}"
        )

    # Exactly one canonical artifact per logical case, and for the three cases that
    # exist in two formats, the one the protocol chose. Without this the split is well
    # defined but the thing measured under it is not.
    for case in sorted(cases):
        canonical = [
            entry["id"]
            for entry in entries
            if entry["logical_case"] == case and entry.get("canonical") is True
        ]
        if len(canonical) != 1:
            fail(f"{case}: expected exactly one canonical artifact, found {canonical}")
        expected = CANONICAL_OF_MULTI_VARIANT_CASE.get(case)
        if expected is not None and canonical[0] != expected:
            fail(
                f"{case}: canonical artifact is {canonical[0]}, "
                f"the frozen table says {expected}"
            )

    length = manifest["length_description"]
    recorded = (
        length["logical_case_tokens_recorded_e5"]["real"]
        + length["logical_case_tokens_recorded_e5"]["synthetic"]
    )
    if length.get("case_count") != len(recorded):
        fail("length_description case_count does not match the recorded distribution")
    if length.get("case_count") != len(cases):
        fail("length_description must cover every logical case exactly once")

    return {
        "artifacts": len(entries),
        "logical_cases": len(cases),
        "synthetic": sum(1 for e in entries if e["stratum"] == "synthetic_fixture"),
        "real": sum(1 for e in entries if e["stratum"] == "real_source"),
        "recorded_token_range": (min(recorded), max(recorded)),
        "on_context_edges": sum(e["measured"]["on_context_edges"] for e in entries),
    }


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    try:
        summary = validate(root)
    except ValidationError as error:
        print(f"evaluation-data validation failed: {error}", file=sys.stderr)
        return 1

    print(
        "evaluation-data validation passed: "
        f"{summary['artifacts']} artifacts over "
        f"{summary['logical_cases']} logical cases "
        f"({summary['synthetic']} synthetic, {summary['real']} real), "
        f"recorded-tokenizer {summary['recorded_token_range'][0]}-"
        f"{summary['recorded_token_range'][1]} tokens, "
        f"{summary['on_context_edges']} "
        f"ON-arm context edges in the whole pool"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
