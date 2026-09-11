"""What the judged evaluation counts, refuses and reproduces.

Every test here is offline: no judge is called, and the saved-judgement store is
written directly, which is exactly the material the offline report is required to
work from.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass, fields
from pathlib import Path
from threading import Barrier, Lock
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import evaluate  # noqa: E402
import evaluation_inputs  # noqa: E402
from evaluation_inputs import EvaluationInputError  # noqa: E402

ACT = "DU/2026/795"
LOCATOR = f"https://api.sejm.gov.pl/eli/acts/{ACT}/text.pdf#article=640"
OTHER_LOCATOR = f"https://api.sejm.gov.pl/eli/acts/{ACT}/text.pdf#article=750"
ABSENT_LOCATOR = f"https://api.sejm.gov.pl/eli/acts/{ACT}/text.pdf#article=9999"
UNPREPARED_ACT = "DU/2024/1796"
UNPREPARED_LOCATOR = (
    f"https://api.sejm.gov.pl/eli/acts/{UNPREPARED_ACT}/text.pdf#article=5"
)
OUTSIDE_ACT = "DU/1999/111"
OUTSIDE_LOCATOR = f"https://api.sejm.gov.pl/eli/acts/{OUTSIDE_ACT}/text.pdf#article=5"
SNAPSHOT_ID = "76378dd52cd69992398e1913"
DOCUMENTS = {"DOC-A": "synthetic_fixture", "DOC-B": "real_source"}
TEXT = {
    "DOC-A": "§ 1. Wykonawca wykona dzieło w terminie. " * 4,
    "DOC-B": "§ 1. Strony zawiązują spółkę cywilną na czas nieoznaczony. " * 4,
}


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


# The instrument a saved judgement names, in the shape a real record carries.
ADAPTER_SHA256 = _sha("adapter-identity")


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def finding(
    finding_id: str,
    *,
    code: str = "consistent",
    start: int = 0,
    end: int = 30,
    locators: tuple[str, ...] = (LOCATOR,),
    quote_resolution: str = "resolved",
    anchor_resolved: bool = True,
) -> dict[str, Any]:
    return {
        "id": finding_id,
        "code": code,
        "anchor": {"start_offset": start, "end_offset": end},
        "anchor_resolved": anchor_resolved,
        "quote_resolution": quote_resolution,
        "legal_locators": list(locators),
        "basis": {"provision_locator": locators[0], "character_kind": "dispositive"}
        if locators
        else None,
    }


def key_item(
    item_id: str,
    document_key: str,
    *,
    status: str = "scored",
    start: int = 0,
    end: int = 30,
    bases: tuple[str, ...] = (LOCATOR,),
    outside_snapshot: bool = False,
) -> dict[str, Any]:
    return {
        "item_id": item_id,
        "document_key": document_key,
        "artifact_id": document_key,
        "issue": f"oczekiwanie {item_id}",
        "span": {"start": start, "end": end},
        "quote": TEXT[document_key][start:end],
        "unit_ids": [f"{document_key}:0:30"],
        "justification": "Uzasadnienie pozycji klucza.",
        "source_evidence": [{"source_id": "KC", "reference": "art. 640"}],
        "status": status,
        "outside_snapshot": outside_snapshot,
        "candidate_finding_ids": ["should-never-reach-the-judge"],
        "accepted_answers": [
            {
                "code": "consistent",
                "accepted_bases": list(bases),
                "justification": "Przepis wprost dopuszcza takie postanowienie.",
            }
        ]
        if status == "scored"
        else [],
        **({"unscored_reason": "ambiguous_basis"} if status == "unscored" else {}),
    }


def dataset(
    tmp_path: Path,
    cells: dict[tuple[int, str, str], list[dict[str, Any]]],
    items: list[dict[str, Any]],
    *,
    provisions: tuple[str, ...] = (LOCATOR, OTHER_LOCATOR),
    ready: bool = False,
    corpus_acts: tuple[str, ...] = (ACT, UNPREPARED_ACT),
) -> Path:
    """A miniature evaluation-data tree with the hashes the protocol requires."""
    data = tmp_path / "evaluation-data"
    documents = sorted({document for _, document, _ in cells})
    series_numbers = sorted({series for series, _, _ in cells})

    manifest = {
        "schema_version": 1,
        "documents": [
            {
                "id": document,
                "logical_case": document,
                "path": f"documents/{document}.txt",
                "format": "txt",
            }
            for document in documents
        ],
    }
    _write(data / "manifest.json", manifest)
    for document in documents:
        path = data / "documents" / f"{document}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(TEXT[document], encoding="utf-8")

    series_records = []
    for series in series_numbers:
        batch = data / "results" / "final" / f"series-{series}"
        hashes: dict[str, str] = {}
        recorded = 0
        for document in documents:
            metadata = {
                "id": f"{document}-{series}",
                "content_hash": _sha(TEXT[document]),
                "unit_count": 1,
            }
            _write(batch / document / "document.json", metadata)
            hashes[f"{document}/document.json"] = hashlib.sha256(
                (batch / document / "document.json").read_bytes()
            ).hexdigest()
            for arm in ("off", "mid", "on"):
                found = cells.get((series, document, arm), [])
                run = {
                    "id": f"{series}-{document}-{arm}",
                    "arm": arm,
                    "status": "completed",
                    "measurement_valid": True,
                    "corpus_snapshot_id": SNAPSHOT_ID,
                    "findings": found,
                    "metrics": {
                        "input_tokens": 100,
                        "output_tokens": 10,
                        "elapsed_ms": 1000.0,
                    },
                }
                _write(batch / document / f"{arm}.json", run)
                hashes[f"{document}/{arm}.json"] = hashlib.sha256(
                    (batch / document / f"{arm}.json").read_bytes()
                ).hexdigest()
                recorded += len(found)
        _write(batch / "summary.json", {"series": series})
        hashes["summary.json"] = hashlib.sha256(
            (batch / "summary.json").read_bytes()
        ).hexdigest()
        series_records.append(
            {
                "series": series,
                "state": "recorded",
                "batch": f"evaluation-data/results/final/series-{series}",
                "findings_recorded": recorded,
                "sha256": hashes,
            }
        )

    _write(
        data / "answer-key.json",
        {
            "schema_version": 1,
            "key_version": "1.0.0",
            "review_state": "review_pending_author_approval",
            "documents": {
                document: {
                    "artifact_id": document,
                    "content_hash": _sha(TEXT[document]),
                    "chars": len(TEXT[document]),
                }
                for document in documents
            },
            "sources": {"KC": {"act_id": ACT}},
            "items": items,
        },
    )

    protocol = {
        "schema_version": 2,
        "scoring_protocol_version": "2.0.0",
        "state": "draft_rules_fixed_procedure_not_ready",
        "scored_runs": {
            "cohort": documents,
            "series_states": {"recorded": "ran", "authorized": "did not run"},
            "answer_key_sha256": hashlib.sha256(
                (data / "answer-key.json").read_bytes()
            ).hexdigest(),
            "documents": {
                document: {
                    "artifact": document,
                    "dir": document,
                    "stratum": DOCUMENTS[document],
                }
                for document in documents
            },
            "document_content_sha256": {
                document: _sha(TEXT[document]) for document in documents
            },
            "manifest_sha256": hashlib.sha256(
                (data / "manifest.json").read_bytes()
            ).hexdigest(),
            "series": series_records,
        },
        "judge": {
            "model": "gpt-5.6-sol",
            "reasoning_effort": "high",
            "instructions_file": "judge/instructions-pl.md",
            "response_schema_file": "judge/response-schema.json",
            "source_bundle_dir": "judge/sources",
            "verdicts_file": "results/judge/verdicts.jsonl",
            "calibration_verdicts_file": "results/judge/calibration-verdicts.jsonl",
            "report_file": "results/judge/report.json",
            "max_packet_chars": 400000,
        },
        "readiness": {
            "ready_for_full_run": ready,
            "what_is_outstanding": "the key is not approved",
        },
    }
    _write(data / "scoring-protocol.json", protocol)

    instructions = data / "judge" / "instructions-pl.md"
    instructions.parent.mkdir(parents=True, exist_ok=True)
    instructions.write_text("Instrukcja oceny.\n", encoding="utf-8")
    write_sources(data, documents, provisions, corpus_acts=corpus_acts)
    return data


def write_sources(
    data: Path,
    documents: list[str],
    provisions: tuple[str, ...],
    *,
    corpus_acts: tuple[str, ...] = (ACT,),
    prepared_acts: tuple[str, ...] = (ACT,),
    verified: bool = True,
    built_at: str = "2026-09-10T00:00:00+00:00",
) -> None:
    bundle = data / "judge" / "sources"
    _write(
        bundle / "documents.json",
        {
            "schema_version": 1,
            "built_at": built_at,
            "documents": {
                document: {
                    "artifact_id": document,
                    "content_sha256": _sha(TEXT[document]),
                    "chars": len(TEXT[document]),
                    "text": TEXT[document],
                }
                for document in documents
            },
        },
    )
    _write(
        bundle / "provisions.json",
        {
            "schema_version": 2,
            "built_at": built_at,
            "corpus": {
                "manifest_sha256": "0" * 64,
                "declared_acts": list(corpus_acts),
                "unit_digest": "1" * 64,
                "snapshot_id": SNAPSHOT_ID,
                "recorded_by_the_runs": [SNAPSHOT_ID],
                "snapshot_verified": verified,
            },
            "acts": {act: {"articles": len(provisions)} for act in prepared_acts},
            "provisions": {
                evaluation_inputs.read_locator(value).key: {
                    "locator": value,
                    "act_identifier": evaluation_inputs.read_locator(value).act,
                    "article_identifier": f"Art. {value.rsplit('=', 1)[1]}",
                    "content_sha256": _sha(provision_text(value)),
                    "text": provision_text(value),
                }
                for value in provisions
            },
        },
    )


def provision_text(locator: str) -> str:
    return f"Treść przepisu {locator.rsplit('=', 1)[1]} o terminie."


def verdict(
    *,
    classification: str = "correct",
    legal_basis: str = "supported",
    matches: tuple[tuple[str, bool], ...] = (),
    evidence_source: str = "document",
    evidence_quote: str = "Wykonawca wykona dzieło",
    evidence: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    return {
        "key_matches": [
            {"item_id": item_id, "covers_expectation": covers, "reason": "bo tak"}
            for item_id, covers in matches
        ],
        "classification": classification,
        "legal_basis": legal_basis,
        "explanation": "not_applicable",
        # A decided basis has to cite a provision, so the default reply cites one
        # beside the document. Naming another source replaces the whole list.
        "evidence": (
            [
                {"source": evidence_source, "quote": evidence_quote},
                *(
                    [{"source": LOCATOR, "quote": provision_text(LOCATOR)}]
                    if evidence_source == "document"
                    else []
                ),
            ]
            if evidence is None
            else evidence
        ),
        "reason": "Uzasadnienie oceny.",
        "error_tags": [],
    }


def reply(**answers: dict[str, Any]) -> dict[str, Any]:
    """The one reply shape the judge returns: an answer per finding of the run."""
    return {
        "assessments": [
            {**answer, "finding_id": finding_id}
            for finding_id, answer in answers.items()
        ]
    }


def check(
    answer: dict[str, Any], packet: Any, finding_id: str = "f1"
) -> evaluate.JudgeVerdict:
    """Validate one answer as part of a complete reply about its run packet.

    The other findings of the packet are answered as undecided, because a reply
    that leaves any of them out is refused before a single answer is read.
    """
    answers = {finding_id: answer}
    for view in packet.payload["assessed_findings"]:
        answers.setdefault(
            str(view["finding_id"]),
            verdict(classification="unresolved", legal_basis="unresolved", evidence=[]),
        )
    return evaluate.validate_reply(reply(**answers), packet)[finding_id]


def save(
    inputs: evaluate.Inputs,
    finding_id: str,
    answer: dict[str, Any],
    **overrides: Any,
) -> None:
    """Write the judgement of a whole run exactly as the judging command would.

    One call answers one run, so saving another finding of a run already written
    edits that record rather than adding a second judgement of the same packet.
    Saving a finding the record already answers is the other case on purpose: it
    is a repeat judgement of the packet, and it is written as the second record
    it would be.
    """
    packet = next(item for item in inputs.packets if finding_id in item.finding_ids)
    answer = {**answer, "finding_id": finding_id}
    path = inputs.config.verdicts_path
    lines = path.read_text(encoding="utf-8").splitlines() if path.is_file() else []
    kept: list[dict[str, Any]] = []
    held: dict[str, Any] | None = None
    for line in lines:
        value = json.loads(line)
        answers = value.get("assessments") or []
        repeatable = held is None and value.get("status") == "validated"
        if value.get("packet_id") == packet.packet_id and repeatable:
            if any(item["finding_id"] == finding_id for item in answers):
                held = {**value, "assessments": []}
                kept.append(value)
                continue
            held = value
            continue
        kept.append(value)
    record = held if held is not None else _run_record(inputs, packet)
    record["assessments"] = [
        *(item for item in record["assessments"] if item["finding_id"] != finding_id),
        answer,
    ]
    record["raw_result"] = {"assessments": record["assessments"]}
    record.update(overrides)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            evaluation_inputs.canonical_json(value) + "\n" for value in [*kept, record]
        ),
        encoding="utf-8",
    )


def _run_record(inputs: evaluate.Inputs, packet: Any) -> dict[str, Any]:
    """The record one judging call for one run writes, before its answers."""
    return {
        "packet_id": packet.packet_id,
        "packet_sha256": evaluation_inputs.sha256_value(packet.payload),
        "configuration_sha256": inputs.config.configuration_sha256,
        "source_bundle_sha256": inputs.bundle.digest,
        "answer_key_sha256": inputs.key.sha256,
        "adapter_sha256": ADAPTER_SHA256,
        "series": packet.series,
        "document_key": packet.document_key,
        "arm": packet.arm,
        "finding_ids": list(packet.finding_ids),
        "status": "validated",
        "elapsed_ms": 1500,
        "usage": {
            "last": {"input_tokens": 20, "output_tokens": 5},
            "total": {"input_tokens": 20, "output_tokens": 5, "total_tokens": 25},
        },
        "assessments": [],
    }


def report_of(data: Path) -> dict[str, Any]:
    inputs = evaluate.load_inputs(data)
    records = evaluate.read_records(inputs.config.verdicts_path)
    return evaluate.build_report(inputs, records)


def cell_of(
    report: dict[str, Any], series: int, document: str, arm: str
) -> dict[str, Any]:
    return next(
        cell
        for cell in report["cells"]
        if (cell["series"], cell["document_key"], cell["arm"])
        == (series, document, arm)
    )


# --------------------------------------------------------------------------


def test_counts_correct_incorrect_and_unresolved_with_their_denominators(
    tmp_path: Path,
) -> None:
    data = dataset(
        tmp_path,
        {
            (1, "DOC-A", "mid"): [
                finding("f1"),
                finding("f2", start=31, end=60),
                finding("f3", start=61, end=90),
            ]
        },
        [key_item("A-K01", "DOC-A"), key_item("A-K02", "DOC-A", start=31, end=60)],
    )
    inputs = evaluate.load_inputs(data)
    save(inputs, "f1", verdict(matches=(("A-K01", True),)))
    save(inputs, "f2", verdict(classification="incorrect", matches=(("A-K02", True),)))
    save(inputs, "f3", verdict(classification="unresolved"))

    cell = cell_of(report_of(data), 1, "DOC-A", "mid")
    assert cell["counts"]["correct"] == 1
    assert cell["counts"]["incorrect"] == 1
    assert cell["counts"]["unresolved"] == 1
    assert cell["dimensions"]["precision_of_resolved_findings"] == {
        "numerator": 1,
        "denominator": 2,
        "rate": 0.5,
    }
    assert cell["dimensions"]["unresolved_share"]["rate"] == pytest.approx(1 / 3)
    assert cell["dimensions"]["completeness_against_key"] == {
        "numerator": 1,
        "denominator": 2,
        "rate": 0.5,
    }


def test_a_zero_denominator_is_undefined_rather_than_zero(tmp_path: Path) -> None:
    data = dataset(
        tmp_path,
        {(1, "DOC-A", "off"): [finding("f1", code="uncertain")]},
        [key_item("A-K01", "DOC-A")],
    )
    cell = cell_of(report_of(data), 1, "DOC-A", "off")
    assert cell["counts"]["non_substantive"]["uncertain"] == 1
    precision = cell["dimensions"]["precision_of_resolved_findings"]
    assert precision["denominator"] == 0
    assert precision["rate"] is None
    assert cell["dimensions"]["completeness_against_key"]["rate"] == 0.0


def test_the_explanation_dimension_is_undefined_when_no_finding_carries_one(
    tmp_path: Path,
) -> None:
    data = dataset(
        tmp_path,
        {(1, "DOC-A", "mid"): [finding("f1")]},
        [key_item("A-K01", "DOC-A")],
    )
    inputs = evaluate.load_inputs(data)
    save(inputs, "f1", verdict(matches=(("A-K01", True),)))
    cell = cell_of(report_of(data), 1, "DOC-A", "mid")
    assert cell["dimensions"]["explanation_accuracy"] == {
        "numerator": 0,
        "denominator": 0,
        "rate": None,
    }


def test_an_exact_duplicate_in_one_run_is_judged_once_and_counted(
    tmp_path: Path,
) -> None:
    data = dataset(
        tmp_path,
        {(1, "DOC-A", "mid"): [finding("f1"), finding("f2")]},
        [key_item("A-K01", "DOC-A")],
    )
    inputs = evaluate.load_inputs(data)
    assert [name for packet in inputs.packets for name in packet.finding_ids] == ["f1"]
    save(inputs, "f1", verdict(matches=(("A-K01", True),)))
    report = report_of(data)
    cell = cell_of(report, 1, "DOC-A", "mid")
    assert cell["counts"]["exact_duplicates_not_judged"] == 1
    assert cell["dimensions"]["completeness_against_key"]["numerator"] == 1
    assert cell["dimensions"]["precision_of_resolved_findings"]["denominator"] == 1
    outcomes = {item["finding_id"]: item for item in report["finding_outcomes"]}
    assert outcomes["f1"]["status"] == "correct"
    assert outcomes["f2"]["disposition"] == "exact_duplicate"
    assert outcomes["f2"]["duplicate_of"] == "f1"
    assert report["accounting"] == {
        "recorded_findings": 2,
        "to_be_judged": 1,
        "exact_duplicates_within_a_run": 1,
        "non_substantive": 0,
    }


def test_identical_findings_in_two_series_stay_two_observations(tmp_path: Path) -> None:
    data = dataset(
        tmp_path,
        {
            (1, "DOC-A", "mid"): [finding("f1")],
            (3, "DOC-A", "mid"): [finding("f2")],
        },
        [key_item("A-K01", "DOC-A")],
    )
    inputs = evaluate.load_inputs(data)
    assert sorted(name for packet in inputs.packets for name in packet.finding_ids) == [
        "f1",
        "f2",
    ]
    assert len({packet.packet_id for packet in inputs.packets}) == 2
    for finding_id in ("f1", "f2"):
        save(inputs, finding_id, verdict(matches=(("A-K01", True),)))
    report = report_of(data)
    for series in (1, 3):
        cell = cell_of(report, series, "DOC-A", "mid")
        assert cell["counts"]["judged"] == 1
        assert cell["counts"]["exact_duplicates_not_judged"] == 0


def test_a_correct_finding_outside_the_key_does_not_move_completeness(
    tmp_path: Path,
) -> None:
    data = dataset(
        tmp_path,
        {(1, "DOC-A", "on"): [finding("f1"), finding("f2", start=31, end=60)]},
        [key_item("A-K01", "DOC-A")],
    )
    inputs = evaluate.load_inputs(data)
    save(inputs, "f1", verdict(matches=(("A-K01", True),)))
    save(inputs, "f2", verdict())
    cell = cell_of(report_of(data), 1, "DOC-A", "on")
    assert cell["counts"]["correct_outside_the_key"] == 1
    assert cell["dimensions"]["precision_of_resolved_findings"]["numerator"] == 2
    assert cell["dimensions"]["completeness_against_key"] == {
        "numerator": 1,
        "denominator": 1,
        "rate": 1.0,
    }


def test_a_wrong_answer_beside_a_right_one_keeps_coverage_and_costs_precision(
    tmp_path: Path,
) -> None:
    data = dataset(
        tmp_path,
        {(1, "DOC-A", "mid"): [finding("f1"), finding("f2", start=5, end=25)]},
        [key_item("A-K01", "DOC-A")],
    )
    inputs = evaluate.load_inputs(data)
    save(inputs, "f1", verdict(matches=(("A-K01", True),)))
    save(inputs, "f2", verdict(classification="incorrect", matches=(("A-K01", False),)))
    cell = cell_of(report_of(data), 1, "DOC-A", "mid")
    assert cell["dimensions"]["completeness_against_key"]["numerator"] == 1
    assert cell["dimensions"]["precision_of_resolved_findings"] == {
        "numerator": 1,
        "denominator": 2,
        "rate": 0.5,
    }
    assert cell["counts"]["items_with_a_correct_and_an_incorrect_answer"] == 1


def test_key_requires_a_settled_expectation(tmp_path: Path) -> None:
    data = dataset(
        tmp_path,
        {(1, "DOC-A", "mid"): [finding("f1")]},
        [key_item("A-K01", "DOC-A", status="unscored")],
    )
    with pytest.raises(EvaluationInputError, match="settled expectation"):
        evaluate.load_inputs(data)


def test_answer_outside_the_key_is_still_judged(tmp_path: Path) -> None:
    data = dataset(
        tmp_path,
        {(1, "DOC-A", "mid"): [finding("f1", start=31, end=60)]},
        [key_item("A-K01", "DOC-A")],
    )
    inputs = evaluate.load_inputs(data)
    save(inputs, "f1", verdict())
    cell = cell_of(report_of(data), 1, "DOC-A", "mid")
    assert cell["dimensions"]["completeness_against_key"]["denominator"] == 1
    assert cell["dimensions"]["completeness_against_key"]["numerator"] == 0
    assert cell["dimensions"]["precision_of_resolved_findings"]["numerator"] == 1
    assert cell["counts"]["correct_outside_the_key"] == 1
    assert "matched_to_an_excluded_item" not in cell["counts"]


def test_a_fabricated_quote_is_incorrect_whatever_the_judge_said(
    tmp_path: Path,
) -> None:
    data = dataset(
        tmp_path,
        {
            (1, "DOC-A", "off"): [
                finding("f1", quote_resolution="quote_fabricated"),
            ]
        },
        [key_item("A-K01", "DOC-A")],
    )
    inputs = evaluate.load_inputs(data)
    save(inputs, "f1", verdict(matches=(("A-K01", True),)))
    cell = cell_of(report_of(data), 1, "DOC-A", "off")
    assert cell["counts"]["incorrect"] == 1
    assert cell["counts"]["error_tags"]["quote_not_in_document"] == 1
    assert cell["dimensions"]["completeness_against_key"]["numerator"] == 0


def test_an_empty_basis_locator_does_not_invalidate_a_supported_finding(
    tmp_path: Path,
) -> None:
    raw = finding("f1")
    raw["basis"]["provision_locator"] = ""
    data = dataset(
        tmp_path,
        {(1, "DOC-A", "mid"): [raw]},
        [key_item("A-K01", "DOC-A")],
    )
    inputs = evaluate.load_inputs(data)
    save(inputs, "f1", verdict(matches=(("A-K01", True),)))
    cell = cell_of(report_of(data), 1, "DOC-A", "mid")
    assert cell["counts"]["correct"] == 1
    assert cell["counts"]["deterministic_errors"] == 0


def test_a_basis_the_corpus_does_not_hold_is_a_citation_error(tmp_path: Path) -> None:
    data = dataset(
        tmp_path,
        {(1, "DOC-A", "mid"): [finding("f1", locators=(ABSENT_LOCATOR,))]},
        [key_item("A-K01", "DOC-A")],
    )
    inputs = evaluate.load_inputs(data)
    packet = inputs.packets[0]
    assert packet.payload["assessed_findings"][0][
        "claimed_bases_absent_from_the_corpus"
    ] == [ABSENT_LOCATOR]
    save(inputs, "f1", verdict(matches=(("A-K01", True),)))
    cell = cell_of(report_of(data), 1, "DOC-A", "mid")
    assert cell["counts"]["incorrect"] == 1
    assert cell["counts"]["error_tags"]["basis_not_in_corpus"] == 1


def test_a_missing_prepared_source_refuses_to_build_packets(tmp_path: Path) -> None:
    data = dataset(
        tmp_path,
        {(1, "DOC-A", "mid"): [finding("f1")]},
        [key_item("A-K01", "DOC-A", bases=(OTHER_LOCATOR,))],
        provisions=(LOCATOR,),
    )
    with pytest.raises(EvaluationInputError) as error:
        evaluate.load_inputs(data)
    assert OTHER_LOCATOR in str(error.value)
    assert "prepare" in str(error.value)


def test_a_missing_source_bundle_names_the_files_it_wants(tmp_path: Path) -> None:
    data = dataset(
        tmp_path,
        {(1, "DOC-A", "mid"): [finding("f1")]},
        [key_item("A-K01", "DOC-A")],
    )
    (data / "judge" / "sources" / "provisions.json").unlink()
    with pytest.raises(EvaluationInputError) as error:
        evaluate.load_inputs(data)
    assert "provisions.json" in str(error.value)


def test_a_verdict_about_a_key_item_outside_the_packet_is_invalid(
    tmp_path: Path,
) -> None:
    data = dataset(
        tmp_path,
        {
            (1, "DOC-A", "mid"): [finding("f1")],
            (1, "DOC-B", "mid"): [finding("b1")],
        },
        [key_item("A-K01", "DOC-A"), key_item("B-K01", "DOC-B")],
    )
    inputs = evaluate.load_inputs(data)
    packet = next(item for item in inputs.packets if "f1" in item.finding_ids)
    with pytest.raises(evaluate.EvaluationError) as error:
        check(verdict(matches=(("B-K01", True),)), packet)
    assert "B-K01" in str(error.value)


def test_a_verdict_citing_a_source_outside_the_packet_is_invalid(
    tmp_path: Path,
) -> None:
    data = dataset(
        tmp_path,
        {(1, "DOC-A", "mid"): [finding("f1")]},
        [key_item("A-K01", "DOC-A")],
    )
    inputs = evaluate.load_inputs(data)
    with pytest.raises(evaluate.EvaluationError) as error:
        check(verdict(evidence_source=ABSENT_LOCATOR), inputs.packets[0])
    assert ABSENT_LOCATOR in str(error.value)


def test_a_verdict_judging_an_absent_explanation_is_invalid(tmp_path: Path) -> None:
    data = dataset(
        tmp_path,
        {(1, "DOC-A", "mid"): [finding("f1")]},
        [key_item("A-K01", "DOC-A")],
    )
    inputs = evaluate.load_inputs(data)
    answer = verdict()
    answer["explanation"] = "supported"
    with pytest.raises(evaluate.EvaluationError) as error:
        check(answer, inputs.packets[0])
    assert "not applicable" in str(error.value)


def test_an_invalid_saved_reply_is_a_judging_problem_not_an_analyzer_error(
    tmp_path: Path,
) -> None:
    data = dataset(
        tmp_path,
        {(1, "DOC-A", "mid"): [finding("f1"), finding("f2", start=31, end=60)]},
        [key_item("A-K01", "DOC-A")],
    )
    inputs = evaluate.load_inputs(data)
    save(inputs, "f1", verdict(matches=(("A-K01", True),)))
    broken = verdict()
    broken["classification"] = "brilliant"
    save(inputs, "f2", broken)

    report = report_of(data)
    state = report["completeness_of_this_report"]
    assert state["complete"] is False
    assert [problem["problem"] for problem in state["problems"]] == ["invalid_verdict"]
    # One reply answers the whole run, so an answer this procedure cannot read
    # leaves the run unjudged. Neither finding becomes an analyzer mistake.
    cell = cell_of(report, 1, "DOC-A", "mid")
    assert cell["counts"]["judged"] == 0
    assert cell["counts"]["incorrect"] == 0
    assert state["not_yet_judged_finding_ids"] == ["f1", "f2"]


def test_changed_source_material_invalidates_a_saved_judgement(
    tmp_path: Path,
) -> None:
    data = dataset(
        tmp_path,
        {(1, "DOC-A", "mid"): [finding("f1")]},
        [key_item("A-K01", "DOC-A")],
    )
    inputs = evaluate.load_inputs(data)
    save(inputs, "f1", verdict(matches=(("A-K01", True),)))
    assert report_of(data)["completeness_of_this_report"]["complete"] is True

    write_sources(data, ["DOC-A", "DOC-B"], (LOCATOR, OTHER_LOCATOR, ABSENT_LOCATOR))
    report = report_of(data)
    state = report["completeness_of_this_report"]
    assert [problem["problem"] for problem in state["problems"]] == [
        "stale_configuration"
    ]
    assert state["not_yet_judged"] == 1
    assert cell_of(report, 1, "DOC-A", "mid")["counts"]["judged"] == 0


def test_a_changed_prompt_invalidates_a_saved_judgement(tmp_path: Path) -> None:
    data = dataset(
        tmp_path,
        {(1, "DOC-A", "mid"): [finding("f1")]},
        [key_item("A-K01", "DOC-A")],
    )
    inputs = evaluate.load_inputs(data)
    save(inputs, "f1", verdict(matches=(("A-K01", True),)))
    (data / "judge" / "instructions-pl.md").write_text("Inna instrukcja.\n", "utf-8")
    state = report_of(data)["completeness_of_this_report"]
    assert [problem["problem"] for problem in state["problems"]] == [
        "stale_configuration"
    ]


def test_the_offline_report_is_reproducible_byte_for_byte(tmp_path: Path) -> None:
    data = dataset(
        tmp_path,
        {
            (1, "DOC-A", "off"): [finding("f1")],
            (1, "DOC-A", "mid"): [finding("f2"), finding("f3", start=31, end=60)],
            (1, "DOC-A", "on"): [finding("f4", code="contradictory")],
            (1, "DOC-B", "off"): [finding("b1")],
            (1, "DOC-B", "mid"): [finding("b2")],
            (1, "DOC-B", "on"): [finding("b3", code="uncertain")],
        },
        [
            key_item("A-K01", "DOC-A"),
            key_item("A-K02", "DOC-A", start=31, end=60),
            key_item("B-K01", "DOC-B"),
        ],
    )
    inputs = evaluate.load_inputs(data)
    for finding_id, matches in (
        ("f1", (("A-K01", True),)),
        ("f2", (("A-K01", True),)),
        ("f3", (("A-K02", True),)),
        ("f4", ()),
        ("b1", (("B-K01", True),)),
        ("b2", (("B-K01", False),)),
    ):
        save(inputs, finding_id, verdict(matches=matches))

    first = json.dumps(report_of(data), sort_keys=True, ensure_ascii=False)
    second = json.dumps(report_of(data), sort_keys=True, ensure_ascii=False)
    assert first == second

    report = json.loads(first)
    mid = next(
        entry
        for entry in report["per_arm"]
        if entry["arm"] == "mid" and entry["series"] == 1
    )
    assert mid["strata"]["real_source"]["documents"] == ["DOC-B"]
    assert mid["strata"]["synthetic_fixture"]["documents"] == ["DOC-A"]
    assert mid["dimensions"]["completeness_against_key"]["micro"] == {
        "numerator": 2,
        "denominator": 3,
        "rate": pytest.approx(2 / 3),
    }
    difference = next(
        entry
        for entry in report["paired_differences"]
        if entry["pair"] == "MID-OFF"
        and entry["dimension"] == "completeness_against_key"
    )
    assert difference["per_document"]["DOC-A"] == pytest.approx(0.5)
    assert "decision" not in difference  # Hypotheses combine several dimensions.
    assert report["comparisons"]["decision"] == "pending_thresholds"


def test_the_report_separates_judge_resources_from_analyzer_resources(
    tmp_path: Path,
) -> None:
    data = dataset(
        tmp_path,
        {(1, "DOC-A", "mid"): [finding("f1")]},
        [key_item("A-K01", "DOC-A")],
    )
    inputs = evaluate.load_inputs(data)
    save(inputs, "f1", verdict(matches=(("A-K01", True),)))
    cell = cell_of(report_of(data), 1, "DOC-A", "mid")
    assert cell["analyzer_resources"] == {
        "input_tokens": 100,
        "output_tokens": 10,
        "elapsed_ms": 1000.0,
    }
    assert cell["judge_resources"] == {
        "calls": 1,
        "elapsed_ms": 1500,
        "input_tokens": 20,
        "output_tokens": 5,
    }


def test_missing_judge_usage_stays_unknown_in_document_and_configuration_totals(
    tmp_path: Path,
) -> None:
    data = dataset(
        tmp_path,
        {
            (1, "DOC-A", "mid"): [finding("f1"), finding("f2", start=31, end=60)],
            (1, "DOC-B", "mid"): [finding("f3")],
        },
        [key_item("A-K01", "DOC-A"), key_item("B-K01", "DOC-B")],
    )
    inputs = evaluate.load_inputs(data)
    save(inputs, "f1", verdict())
    save(inputs, "f2", verdict(), usage=None)
    save(inputs, "f3", verdict(evidence_quote="Strony zawiązują spółkę"))
    report = report_of(data)
    for resources in (
        cell_of(report, 1, "DOC-A", "mid")["judge_resources"],
        report["per_arm"][0]["judge_resources"],
    ):
        assert resources["input_tokens"] is None
        assert resources["output_tokens"] is None
    assert cell_of(report, 1, "DOC-B", "mid")["judge_resources"]["input_tokens"] == 20
    # Two runs, two calls, and the one without usage is counted as unknown.
    assert report["judge_attempt_resources"]["input_tokens_known"] == 20
    assert report["judge_attempt_resources"]["attempts_missing_usage"] == 1


def test_a_full_run_is_refused_until_the_procedure_is_ready(tmp_path: Path) -> None:
    data = dataset(
        tmp_path,
        {(1, "DOC-A", "mid"): [finding("f1")]},
        [key_item("A-K01", "DOC-A")],
    )
    with pytest.raises(evaluate.EvaluationError) as error:
        evaluate.command_run(data, None, Path.home() / ".codex")
    assert "the key is not approved" in str(error.value)
    assert "calibrate" in str(error.value)


def test_the_published_response_schema_is_the_model_the_program_validates_with() -> (
    None
):
    published = json.loads(
        (ROOT / "evaluation-data" / "judge" / "response-schema.json").read_text(
            encoding="utf-8"
        )
    )
    assert published == evaluate.response_schema()


def test_an_unprepared_act_refuses_the_packet_and_is_not_a_citation_error(
    tmp_path: Path,
) -> None:
    """The corpus declares this act; nobody prepared its text. That is ours."""
    data = dataset(
        tmp_path,
        {(1, "DOC-A", "mid"): [finding("f1", locators=(UNPREPARED_LOCATOR,))]},
        [key_item("A-K01", "DOC-A")],
    )
    with pytest.raises(EvaluationInputError) as error:
        evaluate.load_inputs(data)
    assert UNPREPARED_LOCATOR in str(error.value)
    assert "prepare" in str(error.value)


def test_a_basis_naming_an_act_the_corpus_never_held_is_a_citation_error(
    tmp_path: Path,
) -> None:
    data = dataset(
        tmp_path,
        {(1, "DOC-A", "mid"): [finding("f1", locators=(OUTSIDE_LOCATOR,))]},
        [key_item("A-K01", "DOC-A")],
    )
    inputs = evaluate.load_inputs(data)
    save(inputs, "f1", verdict(matches=(("A-K01", True),)))
    cell = cell_of(report_of(data), 1, "DOC-A", "mid")
    assert cell["counts"]["error_tags"]["basis_not_in_corpus"] == 1
    assert cell["counts"]["incorrect"] == 1


# --------------------------------------------------------------------------
# the adapter interface, over a fake transport
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DoublePacket:
    """The request shape of scripts/judge_client.py, mirrored for the double."""

    packet_id: str
    instructions: str
    case: str
    response_model: type[BaseModel]


@dataclass(frozen=True)
class DoubleReply:
    """The reply shape of scripts/judge_client.py, mirrored for the double."""

    packet_id: str
    result: BaseModel
    raw_response: str
    items: list[dict[str, Any]]
    thread_id: str
    turn_id: str
    status: str
    requested_model: str
    requested_effort: str
    sdk_version: str
    runtime: str
    usage: dict[str, Any] | None
    started_at: int | None
    completed_at: int | None
    reported_duration_ms: int | None
    elapsed_ms: int


class DoubleError(RuntimeError):
    """The failure shape of scripts/judge_client.py, mirrored for the double."""

    def __init__(self, message: str, evidence: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.evidence = evidence or {}


class DoubleReplyError(DoubleError):
    """Answered, but not with a valid verdict."""


class DoubleIsolationError(DoubleError):
    """A capability the judge is not allowed to have reached the model."""


class DoubleUnavailableError(DoubleError):
    """Could not be run, or the turn did not complete."""


ADAPTER_CONSTANTS = {
    "JUDGE_MODEL": "gpt-5.6-sol",
    "JUDGE_EFFORT": SimpleNamespace(value="high"),
    "SDK_VERSION": "0.147.0",
    "BASE_INSTRUCTIONS": "You are an evaluation judge.",
    "RUNTIME_OVERRIDES": ('forced_login_method="chatgpt"',),
    "PROFILE_NAME": "judge",
    "ALLOWED_ITEM_TYPES": frozenset({"userMessage", "agentMessage", "reasoning"}),
}


def real_adapter() -> Any:
    """The adapter module when this tree carries one, else None."""
    try:
        import judge_client
    except ImportError:
        return None
    return judge_client


def adapter_classes() -> tuple[type[Any], type[Any]]:
    """The real adapter dataclasses when they are importable, else the doubles.

    The conformance test below is what keeps the second case honest: a drift
    between the two fails as soon as the adapter is present.
    """
    adapter = real_adapter()
    if adapter is None:
        return DoublePacket, DoubleReply
    return adapter.JudgePacket, adapter.JudgeReply


def adapter_errors() -> tuple[type[Exception], type[Exception], type[Exception]]:
    adapter = real_adapter()
    if adapter is None:
        return DoubleReplyError, DoubleIsolationError, DoubleUnavailableError
    return (
        adapter.JudgeReplyError,
        adapter.JudgeIsolationError,
        adapter.JudgeUnavailableError,
    )


def test_the_doubles_carry_the_adapter_fields_exactly() -> None:
    adapter = real_adapter()
    if adapter is None:
        pytest.skip("the judge adapter is not present in this tree")
    for double, real in (
        (DoublePacket, adapter.JudgePacket),
        (DoubleReply, adapter.JudgeReply),
    ):
        assert [(f.name, f.type) for f in fields(double)] == [
            (f.name, f.type) for f in fields(real)
        ]
    for name in ADAPTER_CONSTANTS:
        assert hasattr(adapter, name), f"the adapter no longer defines {name}"
    for error in (
        adapter.JudgeReplyError,
        adapter.JudgeIsolationError,
        adapter.JudgeUnavailableError,
    ):
        assert issubclass(error, adapter.JudgeError)
        assert hasattr(error("boom"), "evidence")


class FakeSession:
    """A judging session that answers from a script instead of a model."""

    calls: list[Any] = []
    homes: list[tuple[Path, Path]] = []
    answers: list[Any] = []

    def __init__(self, home: Path, workspace: Path) -> None:
        FakeSession.homes.append((home, workspace))
        self.home = home
        self.workspace = workspace

    def __enter__(self) -> FakeSession:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def judge(self, packet: Any) -> Any:
        FakeSession.calls.append(packet)
        answer = FakeSession.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        # A scripted answer may be written for whichever run the conversation
        # was handed, which is what several workers at once need.
        if callable(answer):
            answer = answer(packet)
        packet_class, reply_class = adapter_classes()
        assert isinstance(packet, packet_class)
        return reply_class(
            packet_id=packet.packet_id,
            result=packet.response_model.model_validate(answer),
            raw_response=json.dumps(answer, ensure_ascii=False),
            items=[{"type": "userMessage"}, {"type": "agentMessage"}],
            thread_id="thread-1",
            turn_id="turn-1",
            status="completed",
            requested_model="gpt-5.6-sol",
            requested_effort="high",
            sdk_version="0.147.0",
            runtime="codex/1.2.3",
            usage={
                "last": {"input_tokens": 20, "output_tokens": 5, "total_tokens": 25},
                "total": {"input_tokens": 20, "output_tokens": 5, "total_tokens": 25},
                "model_context_window": 400000,
            },
            started_at=1,
            completed_at=2,
            reported_duration_ms=1200,
            elapsed_ms=1500,
        )


@pytest.fixture
def fake_adapter(monkeypatch: pytest.MonkeyPatch) -> type[FakeSession]:
    """Wire the run loop to the real request shape and a scripted transport.

    Only the session is replaced. The packet class and every constant that goes
    into the recorded instrument identity come from the real adapter wherever
    this tree has one, so the run loop is exercised against the interface it will
    actually meet.
    """
    FakeSession.calls = []
    FakeSession.homes = []
    FakeSession.answers = []
    packet_class, _ = adapter_classes()
    adapter = real_adapter()
    reply_error, isolation_error, unavailable_error = adapter_errors()
    stub = SimpleNamespace(
        JudgePacket=packet_class,
        JudgeSession=FakeSession,
        JudgeReplyError=reply_error,
        JudgeIsolationError=isolation_error,
        JudgeUnavailableError=unavailable_error,
        **{
            name: getattr(adapter, name, default)
            for name, default in ADAPTER_CONSTANTS.items()
        },
    )
    monkeypatch.setattr(evaluate, "load_adapter_module", lambda: stub)
    return FakeSession


def credentials(tmp_path: Path) -> Path:
    home = tmp_path / "signed-in"
    home.mkdir(exist_ok=True)
    (home / "auth.json").write_text('{"tokens": "not-a-real-secret"}', encoding="utf-8")
    return home


def ready_dataset(tmp_path: Path) -> Path:
    return dataset(
        tmp_path,
        {(1, "DOC-A", "mid"): [finding("f1"), finding("f2", start=31, end=60)]},
        [key_item("A-K01", "DOC-A"), key_item("A-K02", "DOC-A", start=31, end=60)],
        ready=True,
    )


def test_a_run_sends_the_adapter_request_and_saves_every_reply_field(
    tmp_path: Path, fake_adapter: type[FakeSession]
) -> None:
    data = ready_dataset(tmp_path)
    fake_adapter.answers = [
        reply(
            f1=verdict(matches=(("A-K01", True),)),
            f2=verdict(matches=(("A-K02", True),)),
        )
    ]
    assert evaluate.command_run(data, None, credentials(tmp_path)) == 0

    # One run, one conversation, and both of its findings answered in it.
    assert len(fake_adapter.calls) == 1
    request = fake_adapter.calls[0]
    assert request.response_model is evaluate.JudgeReply
    assert not hasattr(request, "output_schema")
    assert request.instructions == "Instrukcja oceny.\n"
    assert "OFF" not in request.case and "mid" not in request.case

    records = evaluate.read_records(data / "results" / "judge" / "verdicts.jsonl")
    assert [record.status for record in records] == ["validated"]
    saved = records[0].raw
    assert saved["turn_status"] == "completed"
    assert saved["requested_model"] == "gpt-5.6-sol"
    assert saved["requested_effort"] == "high"
    assert saved["reported_duration_ms"] == 1200
    assert saved["started_at"] == 1 and saved["completed_at"] == 2
    assert saved["transcript_items"] == [
        {"type": "userMessage"},
        {"type": "agentMessage"},
    ]
    answered = json.loads(saved["raw_response"])["assessments"]
    assert answered[0]["classification"] == "correct"
    assert saved["raw_result"]["assessments"][0]["classification"] == "correct"
    assert saved["finding_ids"] == ["f1", "f2"]
    assert [answer["finding_id"] for answer in saved["assessments"]] == ["f1", "f2"]
    assert saved["stage"] == "cohort"
    # The record has to survive a round trip through the file it is written to.
    assert json.loads(json.dumps(saved)) == saved

    report = report_of(data)
    assert report["completeness_of_this_report"]["complete"] is True
    assert cell_of(report, 1, "DOC-A", "mid")["counts"]["correct"] == 2


def test_a_second_run_judges_only_what_is_missing(
    tmp_path: Path, fake_adapter: type[FakeSession]
) -> None:
    data = dataset(
        tmp_path,
        {
            (1, "DOC-A", "mid"): [finding("f1")],
            (1, "DOC-A", "off"): [finding("f2")],
        },
        [key_item("A-K01", "DOC-A")],
        ready=True,
    )
    fake_adapter.answers = [reply(f1=verdict(matches=(("A-K01", True),)))]
    evaluate.command_run(data, 1, credentials(tmp_path))
    assert len(fake_adapter.calls) == 1

    # The run that already has a reply is never asked again; the other one is.
    fake_adapter.answers = [reply(f2=verdict(matches=(("A-K01", True),)))]
    evaluate.command_run(data, None, credentials(tmp_path))
    assert len(fake_adapter.calls) == 2
    assert fake_adapter.calls[0].packet_id != fake_adapter.calls[1].packet_id


@pytest.mark.parametrize(
    ("answers", "expected"),
    [
        ({"f1": True}, "1 unanswered"),
        ({"f1": True, "f2": True, "f9": True}, "1 not in the packet"),
    ],
)
def test_a_reply_that_does_not_answer_the_run_exactly_once_is_refused(
    tmp_path: Path, answers: dict[str, bool], expected: str
) -> None:
    """A missing, repeated or foreign answer is invalid output, not a mistake.

    The packet asks a fixed set of questions, so an answer sheet that does not
    match it cannot be read as a judgement of the findings it did answer.
    """
    data = ready_dataset(tmp_path)
    packet = evaluate.load_inputs(data).packets[0]
    with pytest.raises(evaluate.EvaluationError) as error:
        evaluate.validate_reply(reply(**{name: verdict() for name in answers}), packet)
    assert expected in str(error.value)


def test_a_reply_answering_one_finding_twice_is_refused(tmp_path: Path) -> None:
    data = ready_dataset(tmp_path)
    packet = evaluate.load_inputs(data).packets[0]
    answered = reply(f1=verdict(), f2=verdict())
    answered["assessments"].append({**verdict(), "finding_id": "f1"})
    with pytest.raises(evaluate.EvaluationError) as error:
        evaluate.validate_reply(answered, packet)
    assert "1 answered twice (f1)" in str(error.value)


def test_one_reply_carries_the_whole_run_and_each_finding_keeps_its_own_verdict(
    tmp_path: Path,
) -> None:
    """The findings of one run are judged together and counted apart."""
    data = ready_dataset(tmp_path)
    inputs = evaluate.load_inputs(data)
    assert len(inputs.packets) == 1
    assert inputs.packets[0].finding_ids == ("f1", "f2")
    save(inputs, "f1", verdict(matches=(("A-K01", True),)))
    save(inputs, "f2", verdict(classification="incorrect", matches=(("A-K02", True),)))
    assert len(evaluate.read_records(inputs.config.verdicts_path)) == 1

    report = report_of(data)
    cell = cell_of(report, 1, "DOC-A", "mid")
    assert (cell["counts"]["correct"], cell["counts"]["incorrect"]) == (1, 1)
    # One call answered both findings, so it is counted once.
    assert cell["judge_resources"] == {
        "calls": 1,
        "elapsed_ms": 1500,
        "input_tokens": 20,
        "output_tokens": 5,
    }
    state = report["completeness_of_this_report"]
    assert state["expected_assessments"] == 2
    assert state["packets_answered_in_full"] == 1
    assert state["complete"] is True


def test_a_partly_answered_run_is_never_a_complete_report(tmp_path: Path) -> None:
    data = ready_dataset(tmp_path)
    inputs = evaluate.load_inputs(data)
    save(inputs, "f1", verdict(matches=(("A-K01", True),)))
    state = report_of(data)["completeness_of_this_report"]
    assert state["packets_answered_in_full"] == 0
    assert state["judged"] == 0
    assert state["complete"] is False


def answer_the_run(request: Any) -> dict[str, Any]:
    """A correct answer for every finding of whichever run was sent."""
    case = json.loads(request.case)
    return reply(
        **{
            str(view["finding_id"]): verdict(matches=(("A-K01", True),))
            for view in case["assessed_findings"]
        }
    )


def two_run_dataset(tmp_path: Path) -> Path:
    return dataset(
        tmp_path,
        {
            (1, "DOC-A", "mid"): [finding("f1")],
            (1, "DOC-A", "off"): [finding("f2")],
            (1, "DOC-A", "on"): [finding("f3")],
        },
        [key_item("A-K01", "DOC-A")],
        ready=True,
    )


def test_four_workers_overlap_and_reuse_isolated_runtimes(
    tmp_path: Path, fake_adapter: type[FakeSession]
) -> None:
    data = dataset(
        tmp_path,
        {
            (series, "DOC-A", arm): [finding(f"f-{series}-{arm}")]
            for series in (1, 2)
            for arm in ("off", "mid", "on")
        },
        [key_item("A-K01", "DOC-A")],
        ready=True,
    )
    barrier, lock = Barrier(4), Lock()
    active = peak = started = 0

    def overlapping_answer(request: Any) -> dict[str, Any]:
        nonlocal active, peak, started
        with lock:
            active += 1
            started += 1
            index = started
            peak = max(peak, active)
        if index <= 4:
            barrier.wait(timeout=5)
        with lock:
            active -= 1
        return answer_the_run(request)

    fake_adapter.answers = [overlapping_answer] * 6
    assert evaluate.command_run(data, None, credentials(tmp_path), 4) == 0
    assert peak == 4
    records = evaluate.read_records(data / "results/judge/verdicts.jsonl")
    assert len(records) == 6
    assert {record.status for record in records} == {"validated"}
    homes = [home for home, _ in fake_adapter.homes]
    assert len(set(homes)) == 4
    assert all(not home.exists() for home in homes)
    assert report_of(data)["completeness_of_this_report"]["complete"] is True


def test_resume_revalidates_the_entire_saved_reply(
    tmp_path: Path, fake_adapter: type[FakeSession]
) -> None:
    data = ready_dataset(tmp_path)
    inputs = evaluate.load_inputs(data)
    save(inputs, "f1", verdict())
    path = inputs.config.verdicts_path
    row = json.loads(path.read_text())
    row["adapter_sha256"] = evaluate.sha256_value(evaluate.adapter_identity())
    path.write_text(json.dumps(row) + "\n")
    fake_adapter.answers = [answer_the_run]
    evaluate.command_run(data, None, credentials(tmp_path))
    assert len(fake_adapter.calls) == 1
    assert report_of(data)["completeness_of_this_report"]["complete"] is True


def test_session_startup_failure_stops_new_packets(
    tmp_path: Path, fake_adapter: type[FakeSession], monkeypatch: pytest.MonkeyPatch
) -> None:
    data = two_run_dataset(tmp_path)
    lock = Lock()
    started = 0

    def start(session: FakeSession) -> FakeSession:
        nonlocal started
        with lock:
            started += 1
            first = started == 1
        if first:
            raise RuntimeError("session startup failed")
        return session

    monkeypatch.setattr(fake_adapter, "__enter__", start)
    fake_adapter.answers = [answer_the_run] * 3
    with pytest.raises(RuntimeError, match="session startup failed"):
        evaluate.command_run(data, None, credentials(tmp_path), 2)
    assert len(fake_adapter.calls) <= 1
    assert all(not home.exists() for home, _ in fake_adapter.homes)


def test_run_provisions_are_sent_once_even_when_claimed_and_in_the_key(
    tmp_path: Path,
) -> None:
    packet = evaluate.load_inputs(ready_dataset(tmp_path)).packets[0]
    locators = [
        view["locator"]
        for group in packet.payload["provisions"].values()
        for view in group
    ]
    assert LOCATOR in locators
    assert len(locators) == len(set(locators))


def test_a_worker_failure_stops_the_run_and_keeps_what_was_judged(
    tmp_path: Path, fake_adapter: type[FakeSession]
) -> None:
    """A failure is recorded, no new packet is taken, and a rerun continues."""
    data = two_run_dataset(tmp_path)
    _, _, unavailable_error = adapter_errors()
    fake_adapter.answers = [
        answer_the_run,
        unavailable_error("the runtime is not signed in"),
        answer_the_run,
    ]
    with pytest.raises(evaluate.EvaluationError):
        evaluate.command_run(data, None, credentials(tmp_path), 1)

    records = evaluate.read_records(data / "results" / "judge" / "verdicts.jsonl")
    assert [record.status for record in records] == ["validated", "judge_unavailable"]

    fake_adapter.answers = [answer_the_run] * 2
    assert evaluate.command_run(data, None, credentials(tmp_path), 2) == 0
    assert report_of(data)["completeness_of_this_report"]["complete"] is True


@pytest.mark.parametrize("workers", [0, 5])
def test_the_number_of_workers_is_bounded(
    tmp_path: Path, fake_adapter: type[FakeSession], workers: int
) -> None:
    data = ready_dataset(tmp_path)
    with pytest.raises(EvaluationInputError, match="judging workers"):
        evaluate.command_run(data, None, credentials(tmp_path), workers)
    assert fake_adapter.calls == []


def test_an_adapter_failure_keeps_the_evidence_it_carries(
    tmp_path: Path, fake_adapter: type[FakeSession]
) -> None:
    data = ready_dataset(tmp_path)
    reply_error, _, _ = adapter_errors()
    failure = reply_error(
        "reply does not match JudgeVerdict: classification",
        evidence={
            "raw_response": "{not json",
            "usage": {"total": {"input_tokens": 11}},
            "turn_id": "turn-9",
        },
    )
    fake_adapter.answers = [failure]

    with pytest.raises(evaluate.EvaluationError) as error:
        evaluate.command_run(data, None, credentials(tmp_path))
    assert "judging stopped" in str(error.value)

    records = evaluate.read_records(data / "results" / "judge" / "verdicts.jsonl")
    assert len(records) == 1
    saved = records[0].raw
    assert saved["status"] == "invalid_output"
    assert saved["error_type"] == reply_error.__name__
    assert saved["attempt"] == {
        "raw_response": "{not json",
        "usage": {"total": {"input_tokens": 11}},
        "turn_id": "turn-9",
    }

    report = report_of(data)
    state = report["completeness_of_this_report"]
    assert state["complete"] is False
    assert [problem["problem"] for problem in state["problems"]] == ["judge_error"]
    assert state["attempts"]["failed_attempts"] == 1
    assert cell_of(report, 1, "DOC-A", "mid")["counts"]["incorrect"] == 0


def test_an_isolation_failure_is_not_filed_as_an_unavailable_runtime(
    tmp_path: Path, fake_adapter: type[FakeSession]
) -> None:
    """Three failures, three names: only one of them invites a rerun."""
    data = ready_dataset(tmp_path)
    reply_error, isolation_error, unavailable_error = adapter_errors()
    for error_class, expected in (
        (isolation_error, "isolation_failure"),
        (unavailable_error, "judge_unavailable"),
        (reply_error, "invalid_output"),
    ):
        store = data / "results" / "judge" / "verdicts.jsonl"
        store.unlink(missing_ok=True)
        fake_adapter.answers = [error_class("boom")]
        with pytest.raises(evaluate.EvaluationError):
            evaluate.command_run(data, None, credentials(tmp_path))
        assert evaluate.read_records(store)[0].status == expected


def test_a_failure_without_attached_evidence_says_so(
    tmp_path: Path, fake_adapter: type[FakeSession]
) -> None:
    data = ready_dataset(tmp_path)
    _, _, unavailable_error = adapter_errors()
    fake_adapter.answers = [unavailable_error("the runtime is not signed in")]
    with pytest.raises(evaluate.EvaluationError):
        evaluate.command_run(data, None, credentials(tmp_path))
    saved = evaluate.read_records(data / "results" / "judge" / "verdicts.jsonl")[0].raw
    assert saved["status"] == "judge_unavailable"
    assert saved["attempt"] is None
    assert "message only" in saved["attempt_note"]


def test_the_runtime_home_is_temporary_outside_the_artefact_and_removed(
    tmp_path: Path, fake_adapter: type[FakeSession]
) -> None:
    data = ready_dataset(tmp_path)
    fake_adapter.answers = [
        reply(
            f1=verdict(matches=(("A-K01", True),)),
            f2=verdict(matches=(("A-K02", True),)),
        )
    ]
    evaluate.command_run(data, 1, credentials(tmp_path))

    home, workspace = fake_adapter.homes[0]
    assert home.is_absolute() and workspace.is_absolute()
    assert home.parent == workspace.parent
    assert not home.exists() and not workspace.exists()
    assert not home.parent.exists()
    assert ROOT not in home.parents
    assert data not in home.parents
    assert not list(data.rglob("auth.json"))


def test_the_temporary_home_carried_only_a_credential_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = credentials(tmp_path)
    with evaluate.judge_runtime(source) as (home, workspace):
        assert sorted(entry.name for entry in home.iterdir()) == ["auth.json"]
        assert list(workspace.iterdir()) == []
        assert (home / "auth.json").stat().st_mode & 0o777 == 0o600
        assert (home / "auth.json").read_text(encoding="utf-8") == (
            source / "auth.json"
        ).read_text(encoding="utf-8")
        root = home.parent
    assert not root.exists()


def test_failure_to_remove_copied_credentials_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = credentials(tmp_path)
    unlink = os.unlink

    def refuse_auth(path, *args, **kwargs):
        if Path(path).name == "auth.json":
            raise PermissionError(13, "credential cleanup denied", str(path))
        return unlink(path, *args, **kwargs)

    root = None
    try:
        with monkeypatch.context() as patch:
            patch.setattr(os, "unlink", refuse_auth)
            with pytest.raises(PermissionError, match="credential cleanup denied"):
                with evaluate.judge_runtime(source) as (home, _workspace):
                    root = home.parent
    finally:
        if root is not None:
            evaluate.shutil.rmtree(root)


def test_judging_refuses_to_start_without_a_signed_in_credentials_home(
    tmp_path: Path,
) -> None:
    with pytest.raises(evaluate.EvaluationError) as error:
        with evaluate.judge_runtime(tmp_path / "nowhere"):
            pass
    assert "auth.json" in str(error.value)


# --------------------------------------------------------------------------
# calibration, which never touches the judged cohort
# --------------------------------------------------------------------------


def calibration_case(case_id: str = "CAL-01", *, text: str = "Umowa robocza.") -> dict:
    return {
        "case_id": case_id,
        "note": "poprawna odpowiedź",
        "payload": {
            "document": {"document_key": "DEV-001", "text": text},
            "assessed_finding": {
                "code": "consistent",
                "span": {"start": 0, "end": 5},
                "quoted_text": text[:5],
                "claimed_bases": [LOCATOR],
                "explanation": None,
            },
            "criteria_applicability": {
                "legal_basis": "required",
                "explanation": "not_applicable",
            },
            "key_items": [
                {
                    "item_id": "DEV-K01",
                    "issue": "oczekiwanie rozwojowe",
                    "expected_answers": [
                        {"code": "consistent", "accepted_bases": [LOCATOR]}
                    ],
                }
            ],
            "provisions": {
                "claimed_by_the_finding": [
                    {"locator": LOCATOR, "text": provision_text(LOCATOR)}
                ],
                "cited_by_the_key": [],
            },
        },
    }


def write_cases(tmp_path: Path, cases: list[dict[str, Any]]) -> Path:
    path = tmp_path / "calibration-cases.json"
    _write(path, {"schema_version": 1, "cases": cases})
    return path


def test_calibration_judges_supplied_cases_into_their_own_store(
    tmp_path: Path, fake_adapter: type[FakeSession]
) -> None:
    data = dataset(
        tmp_path,
        {(1, "DOC-A", "mid"): [finding("f1")]},
        [key_item("A-K01", "DOC-A")],
    )
    cases = write_cases(tmp_path, [calibration_case()])
    fake_adapter.answers = [
        reply(
            **{
                "CAL-01": verdict(
                    matches=(("DEV-K01", True),), evidence_quote="Umowa robocza"
                )
            }
        )
    ]
    assert evaluate.command_calibrate(data, cases, None, credentials(tmp_path)) == 0

    assert not (data / "results" / "judge" / "verdicts.jsonl").exists()
    records = evaluate.read_records(
        data / "results" / "judge" / "calibration-verdicts.jsonl"
    )
    assert len(records) == 1
    assert records[0].raw["stage"] == "calibration"
    assert records[0].packet_id.startswith("cal-")
    assert report_of(data)["completeness_of_this_report"]["judged"] == 0


def test_a_control_case_may_state_several_findings_and_all_must_be_answered(
    tmp_path: Path, fake_adapter: type[FakeSession]
) -> None:
    """A control case with two findings checks the batch contract itself.

    The reply has to name both, so a control run is evidence that the judge can
    answer a run and not only a single finding.
    """
    case = calibration_case()
    payload = case["payload"]
    first = {**payload.pop("assessed_finding"), "finding_id": "CAL-01-a"}
    applicability = payload.pop("criteria_applicability")
    payload["assessed_findings"] = [
        {**first, "criteria_applicability": applicability},
        {**first, "finding_id": "CAL-01-b", "criteria_applicability": applicability},
    ]
    data = dataset(
        tmp_path,
        {(1, "DOC-A", "mid"): [finding("f1")]},
        [key_item("A-K01", "DOC-A")],
    )
    cases = write_cases(tmp_path, [case])

    fake_adapter.answers = [
        reply(**{"CAL-01-a": verdict(evidence_quote="Umowa robocza")})
    ]
    with pytest.raises(evaluate.EvaluationError, match="1 unanswered"):
        evaluate.command_calibrate(data, cases, None, credentials(tmp_path))

    fake_adapter.answers = [
        reply(
            **{
                name: verdict(evidence_quote="Umowa robocza")
                for name in ("CAL-01-a", "CAL-01-b")
            }
        )
    ]
    assert evaluate.command_calibrate(data, cases, None, credentials(tmp_path)) == 0
    records = evaluate.read_records(
        data / "results" / "judge" / "calibration-verdicts.jsonl"
    )
    assert [record.status for record in records] == ["invalid_output", "validated"]
    assert records[1].raw["finding_ids"] == ["CAL-01-a", "CAL-01-b"]


def test_a_calibration_case_may_not_be_a_finding_of_the_judged_runs(
    tmp_path: Path,
) -> None:
    data = dataset(
        tmp_path,
        {(1, "DOC-A", "mid"): [finding("f1")]},
        [key_item("A-K01", "DOC-A")],
    )
    cases = write_cases(tmp_path, [calibration_case("f1")])
    with pytest.raises(EvaluationInputError) as error:
        evaluate.command_calibrate(data, cases, None, credentials(tmp_path))
    assert "f1" in str(error.value)


def test_a_calibration_case_may_not_use_a_document_of_the_cohort(
    tmp_path: Path,
) -> None:
    data = dataset(
        tmp_path,
        {(1, "DOC-A", "mid"): [finding("f1")]},
        [key_item("A-K01", "DOC-A")],
    )
    cases = write_cases(tmp_path, [calibration_case(text=TEXT["DOC-A"])])
    with pytest.raises(EvaluationInputError) as error:
        evaluate.command_calibrate(data, cases, None, credentials(tmp_path))
    assert "document of the judged cohort" in str(error.value)


def test_calibration_packet_ids_are_disjoint_from_cohort_packet_ids(
    tmp_path: Path,
) -> None:
    data = dataset(
        tmp_path,
        {(1, "DOC-A", "mid"): [finding("f1")]},
        [key_item("A-K01", "DOC-A")],
    )
    inputs = evaluate.load_inputs(data)
    protocol = evaluation_inputs.load_json(data / "scoring-protocol.json")
    runs = evaluation_inputs.load_runs(data, protocol)
    calibration = evaluation_inputs.load_calibration_packets(
        write_cases(tmp_path, [calibration_case(), calibration_case("CAL-02")]),
        runs,
        400000,
    )
    cohort_ids = {packet.packet_id for packet in inputs.packets}
    calibration_ids = {packet.packet_id for packet in calibration}
    assert len(calibration_ids) == 2
    assert cohort_ids.isdisjoint(calibration_ids)
    assert all(packet.arm == "calibration" for packet in calibration)


def edit_cohort_key(data: Path) -> None:
    """Change the cohort answer key and re-pin it, as an author edit would."""
    path = data / "answer-key.json"
    key = json.loads(path.read_text(encoding="utf-8"))
    key["key_version"] = "1.1.0"
    _write(path, key)
    protocol_path = data / "scoring-protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol["scored_runs"]["answer_key_sha256"] = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()
    _write(protocol_path, protocol)


def calibrated(
    tmp_path: Path, fake_adapter: type[FakeSession], case: dict[str, Any]
) -> tuple[Path, Path]:
    """One calibration case, judged once, in a dataset with a cohort beside it."""
    data = dataset(
        tmp_path,
        {(1, "DOC-A", "mid"): [finding("f1")]},
        [key_item("A-K01", "DOC-A")],
    )
    cases = write_cases(tmp_path, [case])
    fake_adapter.answers = [
        reply(
            **{
                "CAL-01": verdict(
                    matches=(("DEV-K01", True),), evidence_quote="Umowa robocza"
                )
            }
        )
    ]
    assert evaluate.command_calibrate(data, cases, None, credentials(tmp_path)) == 0
    assert len(fake_adapter.calls) == 1
    return data, cases


def test_a_cohort_key_edit_does_not_judge_a_settled_calibration_case_again(
    tmp_path: Path, fake_adapter: type[FakeSession]
) -> None:
    """A calibration case carries the expectations it is judged against.

    The cohort answer key is not among them, so editing it says nothing about a
    calibration judgement and may not pay for the case a second time.
    """
    data, cases = calibrated(tmp_path, fake_adapter, calibration_case())
    edit_cohort_key(data)

    assert evaluate.command_calibrate(data, cases, None, credentials(tmp_path)) == 0
    assert len(fake_adapter.calls) == 1
    path = data / "results" / "judge" / "calibration-verdicts.jsonl"
    assert len(evaluate.read_records(path)) == 1


@pytest.mark.parametrize("field", ["key_items", "document"])
def test_an_edited_calibration_case_is_judged_again(
    tmp_path: Path, fake_adapter: type[FakeSession], field: str
) -> None:
    """Its own expectations and its own source text are what bind it."""
    data, _ = calibrated(tmp_path, fake_adapter, calibration_case())
    case = calibration_case()
    if field == "key_items":
        case["payload"]["key_items"][0]["issue"] = "inne oczekiwanie"
    else:
        case["payload"]["document"]["text"] = "Umowa robocza z aneksem."
    edited = write_cases(tmp_path / "edited", [case])
    fake_adapter.answers = [
        reply(
            **{
                "CAL-01": verdict(
                    matches=(("DEV-K01", True),), evidence_quote="Umowa robocza"
                )
            }
        )
    ]

    assert evaluate.command_calibrate(data, edited, None, credentials(tmp_path)) == 0
    assert len(fake_adapter.calls) == 2
    path = data / "results" / "judge" / "calibration-verdicts.jsonl"
    assert len(evaluate.read_records(path)) == 2


@pytest.mark.parametrize(
    "field",
    ["packet_sha256", "configuration_sha256", "source_bundle_sha256", "adapter_sha256"],
)
def test_calibration_reuse_still_checks_its_packet_and_instrument(
    tmp_path: Path, fake_adapter: type[FakeSession], field: str
) -> None:
    data, cases = calibrated(tmp_path, fake_adapter, calibration_case())
    inputs = evaluate.load_inputs(data)
    packet = evaluation_inputs.load_calibration_packets(
        cases, inputs.runs, inputs.config.max_packet_chars
    )[0]
    record = evaluate.read_records(inputs.config.calibration_verdicts_path)[0]
    args = (
        packet,
        inputs.config,
        evaluate._calibration_bundle().digest,
        "unrelated-cohort-key",
        record.raw["adapter_sha256"],
    )
    assert record.reusable(*args)
    changed = evaluate.Record({**record.raw, field: _sha("changed")})
    assert not changed.reusable(*args)


def test_a_cohort_key_edit_still_leaves_the_cohort_to_be_judged_again(
    tmp_path: Path, fake_adapter: type[FakeSession]
) -> None:
    """The cohort is judged against the key, so an edited key unsettles it."""
    data = ready_dataset(tmp_path)
    fake_adapter.answers = [
        reply(
            f1=verdict(matches=(("A-K01", True),)),
            f2=verdict(matches=(("A-K02", True),)),
        )
    ]
    evaluate.command_run(data, None, credentials(tmp_path))
    edit_cohort_key(data)

    inputs = evaluate.load_inputs(data)
    pending = evaluate._pending(inputs.packets, inputs.config.verdicts_path, inputs)
    assert len(pending) == len(inputs.packets)
    assert report_of(data)["completeness_of_this_report"]["attempts"][
        "stale_attempts"
    ] == len(inputs.packets)


# --------------------------------------------------------------------------
# source identity, corpus binding and the key pin
# --------------------------------------------------------------------------


def test_preparing_the_same_sources_again_keeps_the_judgements(
    tmp_path: Path, fake_adapter: type[FakeSession]
) -> None:
    """The bundle is identified by its material, not by when it was written."""
    data = ready_dataset(tmp_path)
    fake_adapter.answers = [
        reply(
            f1=verdict(matches=(("A-K01", True),)),
            f2=verdict(matches=(("A-K02", True),)),
        )
    ]
    evaluate.command_run(data, None, credentials(tmp_path))
    assert report_of(data)["completeness_of_this_report"]["complete"] is True

    write_sources(
        data,
        ["DOC-A"],
        (LOCATOR, OTHER_LOCATOR),
        corpus_acts=(ACT, UNPREPARED_ACT),
        built_at="2027-01-01T12:00:00+00:00",
    )
    state = report_of(data)["completeness_of_this_report"]
    assert state["complete"] is True
    assert state["attempts"]["stale_attempts"] == 0


def test_a_bundle_whose_provision_text_was_edited_is_refused(tmp_path: Path) -> None:
    data = dataset(
        tmp_path,
        {(1, "DOC-A", "mid"): [finding("f1")]},
        [key_item("A-K01", "DOC-A")],
    )
    path = data / "judge" / "sources" / "provisions.json"
    bundle = json.loads(path.read_text(encoding="utf-8"))
    key = next(iter(bundle["provisions"]))
    bundle["provisions"][key]["text"] = "Podmieniona treść przepisu."
    _write(path, bundle)
    with pytest.raises(EvaluationInputError) as error:
        evaluate.load_inputs(data)
    assert "content_sha256" in str(error.value)


def test_a_bundle_claiming_another_corpus_than_the_runs_is_refused(
    tmp_path: Path,
) -> None:
    data = dataset(
        tmp_path,
        {(1, "DOC-A", "mid"): [finding("f1")]},
        [key_item("A-K01", "DOC-A")],
    )
    path = data / "judge" / "sources" / "provisions.json"
    bundle = json.loads(path.read_text(encoding="utf-8"))
    bundle["corpus"]["snapshot_id"] = "another-corpus-entirely"
    _write(path, bundle)
    with pytest.raises(EvaluationInputError) as error:
        evaluate.load_inputs(data)
    assert "the runs searched" in str(error.value)


def test_an_unverified_corpus_never_charges_a_missing_provision_to_an_arm(
    tmp_path: Path,
) -> None:
    """Unproved corpus: the same absence becomes a preparation failure."""
    data = dataset(
        tmp_path,
        {(1, "DOC-A", "mid"): [finding("f1", locators=(ABSENT_LOCATOR,))]},
        [key_item("A-K01", "DOC-A")],
    )
    inputs = evaluate.load_inputs(data)
    assert inputs.packets[0].payload["assessed_findings"][0][
        "claimed_bases_absent_from_the_corpus"
    ] == [ABSENT_LOCATOR]

    write_sources(
        data,
        ["DOC-A"],
        (LOCATOR, OTHER_LOCATOR),
        corpus_acts=(ACT, UNPREPARED_ACT),
        verified=False,
    )
    with pytest.raises(EvaluationInputError) as error:
        evaluate.load_inputs(data)
    assert ABSENT_LOCATOR in str(error.value)


def test_a_missing_source_the_key_accepts_is_never_a_citation_error(
    tmp_path: Path,
) -> None:
    """The key asserting a provision exists cannot prove that it does not.

    The locator is accepted by another document's key item, so this packet does
    not require its text on the key side. Only the rule that a locator the key
    accepts anywhere is never provably absent stands between the finding and a
    citation error it has not earned.
    """
    data = dataset(
        tmp_path,
        {
            (1, "DOC-A", "mid"): [finding("f1", locators=(ABSENT_LOCATOR,))],
            (1, "DOC-B", "mid"): [],
        },
        [
            key_item("A-K01", "DOC-A"),
            key_item("B-K01", "DOC-B", bases=(ABSENT_LOCATOR,)),
        ],
    )
    with pytest.raises(EvaluationInputError) as error:
        evaluate.load_inputs(data)
    assert ABSENT_LOCATOR in str(error.value)
    assert "prepare" in str(error.value)


def test_a_changed_answer_key_invalidates_the_judgements_made_under_it(
    tmp_path: Path,
) -> None:
    data = dataset(
        tmp_path,
        {(1, "DOC-A", "mid"): [finding("f1")]},
        [key_item("A-K01", "DOC-A"), key_item("A-K02", "DOC-A", start=31, end=60)],
    )
    inputs = evaluate.load_inputs(data)
    save(inputs, "f1", verdict(matches=(("A-K01", True),)))
    assert report_of(data)["completeness_of_this_report"]["complete"] is True

    key_path = data / "answer-key.json"
    key = json.loads(key_path.read_text(encoding="utf-8"))
    key["items"].pop()
    _write(key_path, key)
    with pytest.raises(EvaluationInputError) as error:
        evaluate.load_inputs(data)
    assert "pins" in str(error.value)


def test_a_judgement_made_under_another_key_is_not_reused(tmp_path: Path) -> None:
    """The pin refuses an edited key; this refuses a judgement of an older one.

    A record whose key hash is not the one now in force was made when a different
    set of expectations was in play, whatever the packet looked like.
    """
    data = dataset(
        tmp_path,
        {(1, "DOC-A", "mid"): [finding("f1")]},
        [key_item("A-K01", "DOC-A")],
    )
    inputs = evaluate.load_inputs(data)
    save(
        inputs,
        "f1",
        verdict(matches=(("A-K01", True),)),
        answer_key_sha256="the-key-as-it-was-before",
    )
    state = report_of(data)["completeness_of_this_report"]
    assert state["attempts"]["stale_attempts"] == 1
    assert state["judged"] == 0


def test_a_store_holding_another_instrument_stops_the_run(
    tmp_path: Path, fake_adapter: type[FakeSession]
) -> None:
    """Judging again would pay for calls no report may read beside the saved ones."""
    data = ready_dataset(tmp_path)
    inputs = evaluate.load_inputs(data)
    save(
        inputs,
        "f1",
        verdict(matches=(("A-K01", True),)),
        adapter_sha256=_sha("another-instrument"),
    )
    with pytest.raises(evaluate.EvaluationError) as error:
        evaluate.command_run(data, None, credentials(tmp_path))
    assert "another instrument" in str(error.value)
    assert fake_adapter.calls == []


def test_judging_refuses_an_adapter_that_is_not_the_configured_instrument(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_adapter: type[FakeSession]
) -> None:
    data = ready_dataset(tmp_path)
    stub = evaluate.load_adapter_module()
    monkeypatch.setattr(
        evaluate,
        "load_adapter_module",
        lambda: SimpleNamespace(
            **{**vars(stub), "JUDGE_EFFORT": SimpleNamespace(value="low")}
        ),
    )
    with pytest.raises(EvaluationInputError) as error:
        evaluate.command_run(data, None, credentials(tmp_path))
    assert "the adapter is fixed to" in str(error.value)


# --------------------------------------------------------------------------
# attempts, evidence and applicability
# --------------------------------------------------------------------------


def test_a_failure_followed_by_a_judgement_leaves_the_report_complete(
    tmp_path: Path,
) -> None:
    data = dataset(
        tmp_path,
        {(1, "DOC-A", "mid"): [finding("f1")]},
        [key_item("A-K01", "DOC-A")],
    )
    inputs = evaluate.load_inputs(data)
    packet = inputs.packets[0]
    evaluate.append_record(
        inputs.config.verdicts_path,
        {
            "packet_id": packet.packet_id,
            "status": "judge_unavailable",
            "error": "the subscription quota is exhausted",
        },
    )
    save(inputs, "f1", verdict(matches=(("A-K01", True),)))

    state = report_of(data)["completeness_of_this_report"]
    assert state["complete"] is True
    assert state["attempts"]["failed_attempts"] == 1
    assert state["problems"] == []


def test_a_repeat_judgement_never_replaces_the_one_that_stands(
    tmp_path: Path,
) -> None:
    data = dataset(
        tmp_path,
        {(1, "DOC-A", "mid"): [finding("f1")]},
        [key_item("A-K01", "DOC-A")],
    )
    inputs = evaluate.load_inputs(data)
    save(inputs, "f1", verdict(matches=(("A-K01", True),)))
    save(inputs, "f1", verdict(classification="incorrect", matches=(("A-K01", False),)))

    report = report_of(data)
    cell = cell_of(report, 1, "DOC-A", "mid")
    assert cell["counts"]["correct"] == 1
    assert cell["counts"]["incorrect"] == 0
    state = report["completeness_of_this_report"]
    assert state["attempts"]["conflicting_repeat_verdicts"] == 1
    assert [problem["problem"] for problem in state["problems"]] == [
        "conflicting_repeat_verdict"
    ]
    assert state["complete"] is False


def test_an_identical_repeat_judgement_is_counted_and_not_a_problem(
    tmp_path: Path,
) -> None:
    data = dataset(
        tmp_path,
        {(1, "DOC-A", "mid"): [finding("f1")]},
        [key_item("A-K01", "DOC-A")],
    )
    inputs = evaluate.load_inputs(data)
    for _ in range(2):
        save(inputs, "f1", verdict(matches=(("A-K01", True),)))
    state = report_of(data)["completeness_of_this_report"]
    assert state["attempts"]["superseded_repeat_verdicts"] == 1
    assert state["complete"] is True


def test_a_verdict_quoting_text_absent_from_the_source_is_invalid(
    tmp_path: Path,
) -> None:
    data = dataset(
        tmp_path,
        {(1, "DOC-A", "mid"): [finding("f1")]},
        [key_item("A-K01", "DOC-A")],
    )
    inputs = evaluate.load_inputs(data)
    with pytest.raises(evaluate.EvaluationError) as error:
        check(verdict(evidence_quote="czego w dokumencie nie ma"), inputs.packets[0])
    assert "not in the source it names" in str(error.value)


def test_a_verdict_quoting_the_provision_it_names_is_accepted(tmp_path: Path) -> None:
    data = dataset(
        tmp_path,
        {(1, "DOC-A", "mid"): [finding("f1")]},
        [key_item("A-K01", "DOC-A")],
    )
    inputs = evaluate.load_inputs(data)
    accepted = check(
        verdict(evidence_source=LOCATOR, evidence_quote="Treść przepisu 640"),
        inputs.packets[0],
    )
    assert accepted.legal_basis == "supported"


def test_a_decided_verdict_without_evidence_is_invalid(tmp_path: Path) -> None:
    data = dataset(
        tmp_path,
        {(1, "DOC-A", "mid"): [finding("f1")]},
        [key_item("A-K01", "DOC-A")],
    )
    inputs = evaluate.load_inputs(data)
    with pytest.raises(evaluate.EvaluationError) as error:
        check(verdict(evidence=[]), inputs.packets[0])
    assert "without citing any source" in str(error.value)


def test_an_undecided_verdict_may_carry_no_evidence(tmp_path: Path) -> None:
    data = dataset(
        tmp_path,
        {(1, "DOC-A", "mid"): [finding("f1")]},
        [key_item("A-K01", "DOC-A")],
    )
    inputs = evaluate.load_inputs(data)
    accepted = check(
        verdict(classification="unresolved", legal_basis="unresolved", evidence=[]),
        inputs.packets[0],
    )
    assert accepted.classification == "unresolved"


def test_a_substantive_finding_may_not_be_excused_from_the_basis_criterion(
    tmp_path: Path,
) -> None:
    data = dataset(
        tmp_path,
        {(1, "DOC-A", "mid"): [finding("f1")]},
        [key_item("A-K01", "DOC-A")],
    )
    inputs = evaluate.load_inputs(data)
    assert inputs.packets[0].payload["assessed_findings"][0][
        "criteria_applicability"
    ] == {
        "legal_basis": "required",
        "explanation": "not_applicable",
    }
    with pytest.raises(evaluate.EvaluationError) as error:
        check(verdict(legal_basis="not_applicable"), inputs.packets[0])
    assert "not applicable" in str(error.value)


# --------------------------------------------------------------------------
# anchors, duplicates and empty cells
# --------------------------------------------------------------------------


def test_an_unresolved_anchor_cannot_cover_a_key_item(tmp_path: Path) -> None:
    data = dataset(
        tmp_path,
        {(1, "DOC-A", "mid"): [finding("f1", anchor_resolved=False)]},
        [key_item("A-K01", "DOC-A")],
    )
    inputs = evaluate.load_inputs(data)
    save(inputs, "f1", verdict(matches=(("A-K01", True),)))
    cell = cell_of(report_of(data), 1, "DOC-A", "mid")
    assert cell["counts"]["correct"] == 1
    assert cell["counts"]["unresolved_anchor"] == 1
    assert cell["dimensions"]["precision_of_resolved_findings"]["numerator"] == 1
    assert cell["dimensions"]["completeness_against_key"]["numerator"] == 0


def test_findings_differing_in_quote_resolution_are_not_one_emission(
    tmp_path: Path,
) -> None:
    data = dataset(
        tmp_path,
        {
            (1, "DOC-A", "mid"): [
                finding("f1"),
                finding("f2", quote_resolution="quote_ambiguous"),
            ]
        },
        [key_item("A-K01", "DOC-A")],
    )
    inputs = evaluate.load_inputs(data)
    assert sorted(name for packet in inputs.packets for name in packet.finding_ids) == [
        "f1",
        "f2",
    ]


def test_findings_differing_in_provision_character_are_not_one_emission(
    tmp_path: Path,
) -> None:
    other = finding("f2")
    other["basis"] = {"provision_locator": LOCATOR, "character_kind": "mandatory"}
    data = dataset(
        tmp_path,
        {(1, "DOC-A", "mid"): [finding("f1"), other]},
        [key_item("A-K01", "DOC-A")],
    )
    inputs = evaluate.load_inputs(data)
    assert sorted(name for packet in inputs.packets for name in packet.finding_ids) == [
        "f1",
        "f2",
    ]


def test_a_run_that_recorded_nothing_is_still_a_result(tmp_path: Path) -> None:
    data = dataset(
        tmp_path,
        {(1, "DOC-A", "mid"): [finding("f1")], (1, "DOC-A", "off"): []},
        [key_item("A-K01", "DOC-A")],
    )
    inputs = evaluate.load_inputs(data)
    save(inputs, "f1", verdict(matches=(("A-K01", True),)))
    report = report_of(data)
    off = cell_of(report, 1, "DOC-A", "off")
    assert off["counts"]["judged"] == 0
    assert off["dimensions"]["completeness_against_key"] == {
        "numerator": 0,
        "denominator": 1,
        "rate": 0.0,
    }
    assert off["dimensions"]["precision_of_resolved_findings"]["rate"] is None
    difference = next(
        entry
        for entry in report["paired_differences"]
        if entry["pair"] == "MID-OFF"
        and entry["dimension"] == "completeness_against_key"
    )
    assert difference["per_document"]["DOC-A"] == pytest.approx(1.0)


def test_the_report_reads_the_nested_token_usage_the_runtime_returns(
    tmp_path: Path, fake_adapter: type[FakeSession]
) -> None:
    data = ready_dataset(tmp_path)
    fake_adapter.answers = [
        reply(
            f1=verdict(matches=(("A-K01", True),)),
            f2=verdict(matches=(("A-K02", True),)),
        )
    ]
    evaluate.command_run(data, 1, credentials(tmp_path))
    cell = cell_of(report_of(data), 1, "DOC-A", "mid")
    assert cell["judge_resources"]["input_tokens"] == 20
    assert cell["judge_resources"]["output_tokens"] == 5


def test_a_series_state_the_protocol_does_not_define_is_refused(
    tmp_path: Path,
) -> None:
    data = dataset(
        tmp_path,
        {(1, "DOC-A", "mid"): [finding("f1")]},
        [key_item("A-K01", "DOC-A")],
    )
    path = data / "scoring-protocol.json"
    protocol = json.loads(path.read_text(encoding="utf-8"))
    protocol["scored_runs"]["series"][0]["state"] = "provisional"
    _write(path, protocol)
    with pytest.raises(EvaluationInputError) as error:
        evaluate.load_inputs(data)
    assert "provisional" in str(error.value)


# --------------------------------------------------------------------------
# locator identity
# --------------------------------------------------------------------------


def test_a_repeated_article_printing_keeps_its_own_identity() -> None:
    first = evaluation_inputs.read_locator(LOCATOR)
    second = evaluation_inputs.read_locator(f"{LOCATOR}&printing=2")
    assert first.key != second.key
    assert first.article == second.article == "640"
    assert second.key.endswith("&printing=2")


def test_an_html_act_locator_keeps_the_identity_the_corpus_minted() -> None:
    html = f"https://api.sejm.gov.pl/eli/acts/{ACT}/text.html/chpt=1/arti=7a"
    identity = evaluation_inputs.read_locator(html)
    assert identity.key == html
    assert identity.act == ACT
    assert not identity.malformed


def test_a_string_that_names_no_act_is_malformed() -> None:
    identity = evaluation_inputs.read_locator("art. 640 kc")
    assert identity.malformed
    assert identity.key == ""


def corpus_unit(locator: str, text: str = "Treść.") -> SimpleNamespace:
    """A corpus unit with the fields preparation reads off a LegalUnit."""
    return SimpleNamespace(
        locator=locator,
        act_identifier=evaluation_inputs.read_locator(locator).act,
        article_identifier="Art. 640",
        content_hash=_sha(text),
        text=text,
        legal_status_date="2026-05-19",
    )


def test_two_printings_of_one_article_are_kept_apart(tmp_path: Path) -> None:
    provisions, acts = evaluate.provision_map(
        [
            corpus_unit(f"{LOCATOR}&printing=1", "Pierwsze wydrukowanie."),
            corpus_unit(f"{LOCATOR}&printing=2", "Drugie wydrukowanie."),
            corpus_unit(
                f"https://api.sejm.gov.pl/eli/acts/{ACT}/text.html/chpt=1/arti=7a",
                "Jednostka z HTML.",
            ),
        ]
    )
    assert sorted(provisions) == [
        f"{ACT}#article=640&printing=1",
        f"{ACT}#article=640&printing=2",
        f"https://api.sejm.gov.pl/eli/acts/{ACT}/text.html/chpt=1/arti=7a",
    ]
    assert acts[ACT]["articles"] == 3
    assert provisions[f"{ACT}#article=640&printing=2"]["text"] == "Drugie wydrukowanie."


def test_two_units_claiming_one_identity_stop_preparation() -> None:
    with pytest.raises(EvaluationInputError) as error:
        evaluate.provision_map(
            [corpus_unit(LOCATOR, "Jedna treść."), corpus_unit(LOCATOR, "Inna treść.")]
        )
    assert "share the identity" in str(error.value)


def test_a_unit_whose_locator_cannot_be_identified_stops_preparation() -> None:
    with pytest.raises(EvaluationInputError) as error:
        evaluate.provision_map([corpus_unit("nie-lokalizator")])
    assert "cannot identify" in str(error.value)


def test_the_emitted_schema_meets_the_endpoint_contract() -> None:
    """Every object must require every property and refuse additional ones.

    A real call under the earlier schema failed with invalid_json_schema because
    one property was optional, so this contract is checked on every object the
    schema carries, definitions included.
    """
    schema = evaluate.response_schema()
    objects = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("type") == "object":
                objects.append(node)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(schema)
    assert len(objects) == 4
    for node in objects:
        assert sorted(node.get("properties") or {}) == sorted(
            node.get("required") or []
        )
        assert node.get("additionalProperties") is False


# --------------------------------------------------------------------------
# the real recorded cohort, offline and without a judge
# --------------------------------------------------------------------------

REAL_DATA = ROOT / "evaluation-data"


def test_every_recorded_finding_of_the_declared_series_is_accounted_for() -> None:
    protocol = evaluation_inputs.load_json(REAL_DATA / "scoring-protocol.json")
    runs = evaluation_inputs.load_runs(REAL_DATA, protocol)
    accounted = evaluation_inputs.account_findings(runs.findings)

    assert len(runs.findings) == 410
    assert len(accounted) == len(runs.findings)
    assert {entry.finding.finding_id for entry in accounted} == {
        found.finding_id for found in runs.findings
    }
    dispositions = {"judged": 0, "exact_duplicate": 0, "non_substantive": 0}
    for entry in accounted:
        dispositions[entry.disposition] += 1
    assert sum(dispositions.values()) == 410
    # 244 consistent, 29 contradictory and 28 permissible_departure are the
    # substantive emissions; 100 uncertain, 8 no_relation and 1 no_basis_found
    # are reported as behaviour and never judged as answers.
    assert dispositions["non_substantive"] == 109
    assert dispositions["judged"] + dispositions["exact_duplicate"] == 301
    assert len(runs.cells) == 18


def test_the_declared_cohort_refuses_to_judge_without_prepared_sources(
    tmp_path: Path,
) -> None:
    import shutil

    data = tmp_path / "evaluation-data"
    shutil.copytree(REAL_DATA, data, ignore=shutil.ignore_patterns("sources"))
    with pytest.raises(EvaluationInputError) as error:
        evaluate.load_inputs(data)
    assert "prepare" in str(error.value)


@pytest.mark.parametrize(
    ("coverage", "precision", "basis", "count", "complete", "expected"),
    [
        (0.6 - 0.5, -0.05, 0.25 - 0.2, 10, True, "criterion_met"),
        (0.09, 0.0, 0.1, 10, True, "below_threshold"),
        (0.2, -0.06, 0.1, 10, True, "loss"),
        (-0.01, None, None, 0, True, "undecided"),
        (-0.01, 0.0, 0.0, 10, True, "loss"),
        (0.2, 0.0, -0.01, 10, True, "loss"),
        (0.2, 0.0, None, 10, True, "undecided"),
        (0.2, 0.0, 0.1, 9, True, "undecided"),
        (0.2, 0.0, 0.1, 10, False, "incomplete"),
    ],
)
def test_comparison_uses_fixed_criteria_and_complete_evidence(
    coverage, precision, basis, count, complete, expected
):
    thresholds = {
        "minimum_completeness_gain": 0.10,
        "maximum_precision_loss": 0.05,
        "minimum_legal_basis_gain": 0.05,
        "maximum_completeness_loss": 0.0,
        "maximum_legal_basis_loss": 0.0,
        "minimum_decided_findings_per_arm": 10,
    }
    assert (
        evaluate.decide_comparison(
            coverage, precision, basis, count, thresholds, complete
        )
        == expected
    )


@pytest.mark.parametrize("top_level", [False, True])
def test_attempt_resources_read_serialized_failure_evidence_once(
    top_level: bool,
) -> None:
    error = DoubleUnavailableError(
        "interrupted response",
        evidence={
            "usage": {"total": {"input_tokens": 100, "output_tokens": 20}},
            "elapsed_ms": 700,
        },
    )
    raw = {"status": "judge_unavailable", **evaluate._failure_evidence(error)}
    if top_level:
        raw.update(
            {
                "elapsed_ms": 300,
                "usage": {"total": {"input_tokens": 50, "output_tokens": 10}},
            }
        )
    result = evaluate.attempt_resources([evaluate.Record(raw)])
    assert result["elapsed_ms_known"] == (300 if top_level else 700)
    assert result["input_tokens_known"] == (50 if top_level else 100)
    assert result["output_tokens_known"] == (10 if top_level else 20)
    assert result["attempts_missing_usage"] == 0
    assert result["attempts_missing_elapsed_ms"] == 0


def test_attempt_resources_include_failures_and_mark_missing_usage():
    attempts = [
        evaluate.Record(
            {
                "status": "invalid_output",
                "elapsed_ms": 700,
                "usage": {"total": {"input_tokens": 100, "output_tokens": 20}},
            }
        ),
        evaluate.Record(
            {
                "status": "validated",
                "elapsed_ms": 300,
                "usage": {"total": {"input_tokens": 50, "output_tokens": 10}},
            }
        ),
        evaluate.Record({"status": "unavailable"}),
    ]
    result = evaluate.attempt_resources(attempts)
    assert result["attempts"] == 3
    assert result["elapsed_ms_known"] == 1000
    assert result["input_tokens_known"] == 150
    assert result["output_tokens_known"] == 30
    assert result["attempts_missing_usage"] == 1
    assert result["attempts_missing_elapsed_ms"] == 1


def test_comparison_aggregates_match_documents_and_keep_micro_precision():
    cells = []
    for arm in ("off", "mid", "on"):
        for doc in ("A", "B"):
            count = 10 if doc == "A" else 100
            correct = (5 if doc == "A" else 90) if arm == "off" else count
            dimensions = {
                name: evaluate._ratio(correct, count) for name in evaluate.DIMENSIONS
            }
            dimensions["completeness_against_key"] = evaluate._ratio(
                5 if arm == "off" else 7, 10
            )
            cells.append(
                {
                    "series": 1,
                    "document_key": doc,
                    "arm": arm,
                    "stratum": "synthetic",
                    "dimensions": dimensions,
                    "analyzer_resources": {
                        "elapsed_ms": 0,
                        "input_tokens": 0,
                        "output_tokens": 0,
                    },
                    "judge_resources": {
                        "elapsed_ms": 0,
                        "input_tokens": 0,
                        "output_tokens": 0,
                    },
                }
            )
    thresholds = {
        "minimum_completeness_gain": 0.10,
        "maximum_precision_loss": 0.05,
        "minimum_legal_basis_gain": 0.05,
        "maximum_completeness_loss": 0.0,
        "maximum_legal_basis_loss": 0.0,
        "minimum_decided_findings_per_arm": 10,
    }
    result = next(
        item
        for item in evaluate.comparison_results(cells, thresholds, True)
        if item["scope"] == "cohort" and item["pair"] == "MID-OFF"
    )
    assert result["completeness_macro_delta"] == pytest.approx(0.2)
    assert result["precision_micro_delta"] == pytest.approx(1 - 95 / 110)
    assert result["legal_basis_macro_delta"] == pytest.approx((0.5 + 0.1) / 2)
    assert result["decision"] == "criterion_met"
    cells[0]["dimensions"]["legal_basis_accuracy"] = evaluate._ratio(0, 0)
    result = next(
        item
        for item in evaluate.comparison_results(cells, thresholds, True)
        if item["scope"] == "cohort" and item["pair"] == "MID-OFF"
    )
    assert result["legal_basis_macro_delta"] is None
    assert result["decision"] == "undecided"


# --------------------------------------------------------------------------
# what a decided basis must cite, what a report refuses, what it only lists
# --------------------------------------------------------------------------


def test_a_basis_decided_on_the_document_alone_is_invalid(tmp_path: Path) -> None:
    data = dataset(
        tmp_path,
        {(1, "DOC-A", "mid"): [finding("f1")]},
        [key_item("A-K01", "DOC-A")],
    )
    inputs = evaluate.load_inputs(data)
    with pytest.raises(evaluate.EvaluationError) as error:
        check(
            verdict(
                matches=(("A-K01", True),),
                evidence=[{"source": "document", "quote": "Wykonawca wykona dzieło"}],
            ),
            inputs.packets[0],
        )
    assert "provision" in str(error.value)


def test_a_packet_carrying_no_provision_may_decide_the_basis_on_the_document(
    tmp_path: Path,
) -> None:
    """Nothing was prepared to cite, so the absence itself is what is read."""
    data = dataset(
        tmp_path,
        {
            (1, "DOC-A", "mid"): [finding("f1", locators=())],
            (1, "DOC-B", "mid"): [],
        },
        [key_item("B-K01", "DOC-B")],
    )
    packet = evaluate.load_inputs(data).packets[0]
    assert packet.payload["provisions"] == {
        "claimed_by_the_findings": [],
        "cited_by_the_key": [],
    }
    checked = check(
        verdict(
            legal_basis="unsupported",
            evidence=[{"source": "document", "quote": "Wykonawca wykona dzieło"}],
        ),
        packet,
    )
    assert checked.legal_basis == "unsupported"


def test_a_basis_only_an_item_outside_the_snapshot_accepts_is_a_citation_error(
    tmp_path: Path,
) -> None:
    """The key places that provision outside the corpus, so its absence is proved.

    An item marked outside_snapshot asserts nothing about what preparation should
    have carried, so a finding claiming the same locator is judged, not deferred.
    """
    data = dataset(
        tmp_path,
        {
            (1, "DOC-A", "mid"): [finding("f1", locators=(ABSENT_LOCATOR,))],
            (1, "DOC-B", "mid"): [],
        },
        [
            key_item("A-K01", "DOC-A"),
            key_item("B-K01", "DOC-B", bases=(ABSENT_LOCATOR,), outside_snapshot=True),
        ],
    )
    inputs = evaluate.load_inputs(data)
    assert inputs.packets[0].payload["assessed_findings"][0][
        "claimed_bases_absent_from_the_corpus"
    ] == [ABSENT_LOCATOR]
    save(inputs, "f1", verdict(matches=(("A-K01", True),)))
    cell = cell_of(report_of(data), 1, "DOC-A", "mid")
    assert cell["counts"]["error_tags"]["basis_not_in_corpus"] == 1
    assert cell["counts"]["incorrect"] == 1


def test_a_correct_finding_outside_the_key_is_listed_as_a_candidate_gap(
    tmp_path: Path,
) -> None:
    """It is recorded as a candidate, counts in precision, and moves no denominator."""
    data = dataset(
        tmp_path,
        {(1, "DOC-A", "mid"): [finding("f1"), finding("f2", start=31, end=60)]},
        [key_item("A-K01", "DOC-A")],
    )
    inputs = evaluate.load_inputs(data)
    save(inputs, "f1", verdict(matches=(("A-K01", True),)))
    save(inputs, "f2", verdict())
    report = report_of(data)
    cell = cell_of(report, 1, "DOC-A", "mid")
    assert cell["counts"]["correct"] == 2
    assert cell["counts"]["correct_outside_the_key"] == 1
    assert cell["dimensions"]["completeness_against_key"]["denominator"] == 1
    assert cell["dimensions"]["completeness_against_key"]["numerator"] == 1
    assert report["key_gap_candidates"]["finding_ids"] == ["f2"]
    outcome = next(
        item for item in report["finding_outcomes"] if item["finding_id"] == "f2"
    )
    assert outcome["status"] == "correct"


def test_judgements_from_two_instruments_are_never_one_report(tmp_path: Path) -> None:
    data = dataset(
        tmp_path,
        {
            (1, "DOC-A", "mid"): [finding("f1")],
            (1, "DOC-A", "off"): [finding("f2", start=31, end=60)],
        },
        [key_item("A-K01", "DOC-A"), key_item("A-K02", "DOC-A", start=31, end=60)],
    )
    inputs = evaluate.load_inputs(data)
    save(inputs, "f1", verdict(matches=(("A-K01", True),)))
    save(
        inputs,
        "f2",
        verdict(matches=(("A-K02", True),)),
        adapter_sha256=_sha("another-instrument"),
    )
    state = report_of(data)["completeness_of_this_report"]
    assert [problem["problem"] for problem in state["problems"]] == ["instrument_mixed"]
    assert state["complete"] is False
    assert state["judged"] == 2


def test_a_validated_record_without_a_verdict_is_listed_not_raised(
    tmp_path: Path,
) -> None:
    data = dataset(
        tmp_path,
        {(1, "DOC-A", "mid"): [finding("f1")]},
        [key_item("A-K01", "DOC-A")],
    )
    inputs = evaluate.load_inputs(data)
    save(inputs, "f1", verdict(matches=(("A-K01", True),)), assessments=None)
    state = report_of(data)["completeness_of_this_report"]
    assert state["attempts"]["invalid_verdicts"] == 1
    assert [problem["problem"] for problem in state["problems"]] == ["invalid_verdict"]
    assert state["judged"] == 0


def test_a_calibration_provision_without_text_is_refused_before_any_call(
    tmp_path: Path, fake_adapter: type[FakeSession]
) -> None:
    data = dataset(
        tmp_path,
        {(1, "DOC-A", "mid"): [finding("f1")]},
        [key_item("A-K01", "DOC-A")],
    )
    case = calibration_case()
    case["payload"]["provisions"]["claimed_by_the_finding"] = [{"locator": LOCATOR}]
    cases = write_cases(tmp_path, [case])
    with pytest.raises(EvaluationInputError) as error:
        evaluate.command_calibrate(data, cases, None, credentials(tmp_path))
    assert "locator" in str(error.value)
    assert fake_adapter.calls == []


def test_report_command_fails_until_all_findings_are_judged(tmp_path: Path) -> None:
    data = dataset(
        tmp_path,
        {(1, "DOC-A", "mid"): [finding("f1")]},
        [key_item("A-K01", "DOC-A")],
    )
    output = tmp_path / "report.json"
    assert evaluate.command_report(data, output) == 1
    assert (
        json.loads(output.read_text())["completeness_of_this_report"]["complete"]
        is False
    )
    inputs = evaluate.load_inputs(data)
    save(inputs, "f1", verdict(matches=(("A-K01", True),)))
    assert evaluate.command_report(data, output) == 0
    assert (
        json.loads(output.read_text())["completeness_of_this_report"]["complete"]
        is True
    )


def test_prepare_rejects_changed_key_before_fetching_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from seeder import ingest

    data = dataset(
        tmp_path,
        {(1, "DOC-A", "mid"): [finding("f1")]},
        [key_item("A-K01", "DOC-A")],
    )
    (data / "answer-key.json").write_text("{}", encoding="utf-8")

    def forbidden_fetch(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("source fetching began before the key was checked")

    monkeypatch.setattr(ingest, "fetch_manifest_units", forbidden_fetch)
    with pytest.raises(EvaluationInputError, match="answer-key.json hashes to"):
        evaluate.command_prepare(data, tmp_path / "corpus.json")


def test_changed_batch_summary_is_rejected(tmp_path: Path) -> None:
    data = dataset(
        tmp_path,
        {(1, "DOC-A", "mid"): [finding("f1")]},
        [key_item("A-K01", "DOC-A")],
    )
    (data / "results/final/series-1/summary.json").write_text("{}", encoding="utf-8")
    with pytest.raises(EvaluationInputError, match="sha256 mismatch for summary.json"):
        evaluate.load_inputs(data)
