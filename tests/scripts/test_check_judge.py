"""Offline tests for the control-sample comparison.

Every case here is synthetic and tiny. The comparison never calls a model, so
these exercise the whole script: what it accepts as the same case judged under
the same instrument, what it counts, and what it refuses.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import check_judge  # noqa: E402
from evaluate import load_judge_config  # noqa: E402
from evaluation_inputs import (  # noqa: E402
    EvaluationInputError,
    calibration_packet,
    load_json,
    sha256_value,
)

DOCUMENT = "Alpha beta gamma delta."
QUOTE = "Alpha beta"


def _payload(
    case_id: str, *, code: str = "consistent", items: int = 1
) -> dict[str, Any]:
    return {
        "document": {"document_key": f"{case_id}-DOC", "text": DOCUMENT},
        "assessed_finding": {
            "code": code,
            "quoted_text": QUOTE,
            "claimed_bases": [],
            "explanation": "an explanation",
        },
        "criteria_applicability": {
            "legal_basis": "required",
            "explanation": "required",
        },
        "key_items": [
            {
                "item_id": f"{case_id}-K{number:02d}",
                "issue": "issue",
                "quote": QUOTE,
                "expected_answers": [],
                "justification": "why",
                "source_references": [],
            }
            for number in range(1, items + 1)
        ],
        "provisions": {"claimed_by_the_finding": [], "cited_by_the_key": []},
    }


def _verdict(
    case_id: str,
    *,
    classification: str = "correct",
    legal_basis: str = "supported",
    explanation: str = "supported",
    matches: list[str] | None = None,
    covers: bool = True,
) -> dict[str, Any]:
    if matches is None:
        matches = [f"{case_id}-K01"]
    return {
        "key_matches": [
            {"item_id": item, "covers_expectation": covers, "reason": "because"}
            for item in matches
        ],
        "classification": classification,
        "legal_basis": legal_basis,
        "explanation": explanation,
        "evidence": [{"source": "document", "quote": QUOTE}],
        "reason": "a reason",
        "error_tags": [],
        "finding_id": case_id,
    }


def _rating(
    case_id: str,
    *,
    classification: str = "correct",
    legal_basis: str = "correct",
    explanation: str = "correct",
    overall: str = "correct",
    matches: list[str] | None = None,
    breach: bool | None = False,
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "classification": classification,
        "legal_basis": legal_basis,
        "explanation": explanation,
        "overall": overall,
        "matching_issue_ids": [f"{case_id}-K01"] if matches is None else matches,
        "clear_breach_of_a_provision": breach,
        "error_tag": None,
        "rationale": "reference rationale",
    }


def _record(case_id: str, payload: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    """The record one judging call for one control case writes.

    The case is judged as the run packet it becomes, so the record carries the
    same wrapper the cohort records carry: the packet identity is minted from
    the converted payload, and the answer is one assessment inside a list.
    """
    packet = calibration_packet(case_id, payload)
    record = {
        "packet_id": packet.packet_id,
        "packet_sha256": sha256_value(packet.payload),
        "adapter_sha256": "adapter-hash",
        "stage": "calibration",
        "finding_ids": list(packet.finding_ids),
        "status": "validated",
        "assessments": [_verdict(case_id)],
    }
    record.update(overrides)
    return record


def write_protocol(directory: Path, *, model: str = "a-test-model") -> Path:
    """A small but real scoring protocol, read by the same loader the run uses."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "judge-instructions.md").write_text("Judge it.", encoding="utf-8")
    protocol = directory / "scoring-protocol.json"
    protocol.write_text(
        json.dumps(
            {
                "judge": {
                    "model": model,
                    "reasoning_effort": "high",
                    "instructions_file": "judge-instructions.md",
                    "max_packet_chars": 100_000,
                    "source_bundle_dir": "bundle",
                    "verdicts_file": "verdicts.jsonl",
                    "calibration_verdicts_file": "calibration-verdicts.jsonl",
                    "report_file": "report.json",
                }
            }
        ),
        encoding="utf-8",
    )
    return protocol


def configuration_of(protocol: Path) -> str:
    return load_judge_config(load_json(protocol), protocol.parent).configuration_sha256


class Bundle:
    """The files the script reads, written under one temporary directory."""

    def __init__(self, directory: Path) -> None:
        self.dir = directory
        self.cases = directory / "cases.json"
        self.reference = directory / "reference-ratings.json"
        self.criteria = directory / "acceptance.json"
        self.verdicts = directory / "verdicts.jsonl"
        self.output = directory / "report.json"
        self.protocol = write_protocol(directory / "protocol")
        self.configuration = configuration_of(self.protocol)

    def write(
        self,
        cases: list[dict[str, Any]],
        ratings: list[dict[str, Any]],
        records: list[dict[str, Any]],
        *,
        criteria: dict[str, Any] | None = None,
        cases_sha256: str | None = None,
        reference_cases_sha256: str | None = None,
    ) -> None:
        self.cases.write_text(
            json.dumps({"schema_version": 1, "cases": cases}), encoding="utf-8"
        )
        true_digest = hashlib.sha256(self.cases.read_bytes()).hexdigest()
        digest = cases_sha256 or true_digest
        rated_against = reference_cases_sha256 or digest
        self.reference.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "reviewer_type": "llm",
                    "reviewer": "an assistant",
                    "exact_model_identifier": None,
                    "source_cases_sha256": rated_against,
                    "ratings": ratings,
                }
            ),
            encoding="utf-8",
        )
        settings = {
            "schema_version": 1,
            "reference_type": "llm",
            "case_count": len(cases),
            "minimum_overall_agreements": len(cases),
            "maximum_key_match_disagreements": 0,
            "maximum_false_reassurances": 0,
            "all_cases_must_have_validated_verdicts": True,
            "interpretation": "not human ratings",
            "cases_sha256": digest,
            "reference_sha256": hashlib.sha256(self.reference.read_bytes()).hexdigest(),
        }
        settings.update(criteria or {})
        self.criteria.write_text(json.dumps(settings), encoding="utf-8")
        for record in records:
            record.setdefault("configuration_sha256", self.configuration)
        self.verdicts.write_text(
            "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
        )

    def run(self) -> int:
        return check_judge.main(
            [
                "--cases",
                str(self.cases),
                "--reference",
                str(self.reference),
                "--criteria",
                str(self.criteria),
                "--verdicts",
                str(self.verdicts),
                "--output",
                str(self.output),
                "--protocol",
                str(self.protocol),
            ]
        )

    def report(self) -> dict[str, Any]:
        return json.loads(self.output.read_text(encoding="utf-8"))


@pytest.fixture
def bundle(tmp_path: Path) -> Bundle:
    return Bundle(tmp_path)


def _agreeing(count: int = 2) -> tuple[list[Any], list[Any], list[Any]]:
    cases, ratings, records = [], [], []
    for number in range(1, count + 1):
        case_id = f"CTL-{number:02d}"
        payload = _payload(case_id)
        cases.append({"case_id": case_id, "payload": payload})
        ratings.append(_rating(case_id))
        records.append(_record(case_id, payload))
    return cases, ratings, records


# ── the happy path ─────────────────────────────────────────────────────────────


def test_full_agreement_passes_every_criterion(bundle: Bundle) -> None:
    bundle.write(*_agreeing())

    assert bundle.run() == 0

    report = bundle.report()
    assert report["state"] == "passed"
    assert report["cases"] == 2
    assert report["overall_agreements"] == 2
    assert report["key_match_disagreements"] == 0
    assert report["false_reassurances"] == 0
    assert all(check["passed"] for check in report["checks"])
    assert report["reference_type"] == "llm"


def test_report_states_the_thresholds_it_was_given(bundle: Bundle) -> None:
    bundle.write(*_agreeing())
    bundle.run()

    report = bundle.report()
    assert report["criteria"]["minimum_overall_agreements"] == 2
    assert report["criteria"]["maximum_false_reassurances"] == 0
    thresholds = {check["name"]: check["threshold"] for check in report["checks"]}
    assert thresholds["overall_agreements"] == 2
    assert thresholds["false_reassurances"] == 0


def test_dimensions_are_reported_beside_the_overall(bundle: Bundle) -> None:
    cases, ratings, records = _agreeing(1)
    records[0]["assessments"] = [_verdict("CTL-01", explanation="unsupported")]
    bundle.write(cases, ratings, records)
    bundle.run()

    case = bundle.report()["comparisons"][0]
    assert case["classification"] == {
        "reference": "correct",
        "judge": "correct",
        "agrees": True,
    }
    assert case["explanation"] == {
        "reference": "supported",
        "judge": "unsupported",
        "agrees": False,
    }
    assert case["judge_status"] == "incorrect"
    assert case["agrees"] is False


def test_the_two_criterion_vocabularies_are_translated(bundle: Bundle) -> None:
    """The reference says incorrect where the judge says unsupported."""
    cases, ratings, records = _agreeing(1)
    ratings[0] = _rating(
        "CTL-01", legal_basis="incorrect", explanation="unresolved", overall="incorrect"
    )
    records[0]["assessments"] = [
        _verdict("CTL-01", legal_basis="unsupported", explanation="unresolved")
    ]
    bundle.write(cases, ratings, records)

    assert bundle.run() == 0

    case = bundle.report()["comparisons"][0]
    assert case["legal_basis"] == {
        "reference": "unsupported",
        "judge": "unsupported",
        "agrees": True,
    }
    assert case["explanation"]["agrees"] is True
    assert case["judge_status"] == "incorrect"
    assert case["agrees"] is True


# ── selecting the record that describes this case ──────────────────────────────


def test_missing_verdict_is_reported_not_dropped(bundle: Bundle) -> None:
    cases, ratings, records = _agreeing()
    bundle.write(cases, ratings, records[:1])

    assert bundle.run() == 1

    report = bundle.report()
    assert [case["case_id"] for case in report["comparisons"]] == ["CTL-01", "CTL-02"]
    missing = report["comparisons"][1]
    assert missing["judge_status"] is None
    assert missing["validated"] is False
    checks = {check["name"]: check for check in report["checks"]}
    assert checks["cases_validated"]["passed"] is False
    assert checks["cases_validated"]["observed"] == 1


def test_failed_attempt_is_counted_and_the_later_valid_one_is_used(
    bundle: Bundle,
) -> None:
    cases, ratings, records = _agreeing(1)
    failed = _record("CTL-01", cases[0]["payload"], status="judge_unavailable")
    failed.pop("assessments")
    bundle.write(cases, ratings, [failed, records[0]])

    assert bundle.run() == 0

    case = bundle.report()["comparisons"][0]
    assert case["technical_failures"] == 1
    assert case["validated"] is True
    assert bundle.report()["technical_failures"] == 1


def test_identical_repeated_verdicts_are_accepted(bundle: Bundle) -> None:
    cases, ratings, records = _agreeing(1)
    bundle.write(cases, ratings, [records[0], dict(records[0])])

    assert bundle.run() == 0


def test_conflicting_repeated_verdicts_are_an_error(bundle: Bundle) -> None:
    cases, ratings, records = _agreeing(1)
    second = _record("CTL-01", cases[0]["payload"])
    second["assessments"] = [_verdict("CTL-01", classification="incorrect")]
    bundle.write(cases, ratings, [records[0], second])

    with pytest.raises(EvaluationInputError, match="conflicting"):
        bundle.run()


def test_records_of_other_cases_are_ignored(bundle: Bundle) -> None:
    cases, ratings, records = _agreeing(1)
    development = _record("DEV-01", _payload("DEV-01"))
    bundle.write(cases, ratings, [development, records[0]])

    assert bundle.run() == 0
    assert bundle.report()["ignored_records"] == 1


def test_a_record_from_another_stage_is_not_a_control_verdict(bundle: Bundle) -> None:
    cases, ratings, records = _agreeing(1)
    elsewhere = _record("CTL-01", cases[0]["payload"], stage="run")
    elsewhere["assessments"] = [_verdict("CTL-01", classification="incorrect")]
    bundle.write(cases, ratings, [elsewhere, records[0]])

    assert bundle.run() == 0

    report = bundle.report()
    assert report["ignored_records"] == 1
    assert report["comparisons"][0]["judge_status"] == "correct"


def test_a_changed_case_is_refused(bundle: Bundle) -> None:
    cases, ratings, records = _agreeing(1)
    records[0]["packet_sha256"] = sha256_value(_payload("CTL-01", code="contradictory"))
    bundle.write(cases, ratings, records)

    with pytest.raises(EvaluationInputError, match="packet"):
        bundle.run()


def test_a_verdict_naming_no_judge_configuration_is_refused(bundle: Bundle) -> None:
    cases, ratings, records = _agreeing(1)
    records[0]["configuration_sha256"] = ""
    bundle.write(cases, ratings, records)

    with pytest.raises(EvaluationInputError, match="names no judge configuration"):
        bundle.run()


def test_a_second_adapter_is_refused(bundle: Bundle) -> None:
    cases, ratings, records = _agreeing()
    records[0]["adapter_sha256"] = "adapter-one"
    records[1]["adapter_sha256"] = "adapter-two"
    bundle.write(cases, ratings, records)

    with pytest.raises(EvaluationInputError, match="adapter"):
        bundle.run()


def test_one_adapter_throughout_is_reported(bundle: Bundle) -> None:
    cases, ratings, records = _agreeing()
    for record in records:
        record["adapter_sha256"] = "adapter-one"
    bundle.write(cases, ratings, records)

    assert bundle.run() == 0
    assert bundle.report()["adapter_sha256"] == "adapter-one"


def test_a_case_rated_twice_is_refused(bundle: Bundle) -> None:
    cases, ratings, records = _agreeing(1)
    bundle.write(cases, ratings + [_rating("CTL-01", overall="incorrect")], records)

    with pytest.raises(EvaluationInputError, match="twice"):
        bundle.run()


def test_verdicts_made_under_another_protocol_are_refused(bundle: Bundle) -> None:
    """A coherent run of the wrong model is still not the configured judge."""
    other = write_protocol(bundle.dir / "other", model="another-model")
    cases, ratings, records = _agreeing()
    for record in records:
        record["configuration_sha256"] = configuration_of(other)
    bundle.write(cases, ratings, records)

    with pytest.raises(EvaluationInputError, match="configuration"):
        bundle.run()


def test_the_report_names_the_configured_judge(bundle: Bundle) -> None:
    bundle.write(*_agreeing())

    assert bundle.run() == 0

    report = bundle.report()
    assert report["model"] == "a-test-model"
    assert report["reasoning_effort"] == "high"
    assert report["configuration_sha256"] == bundle.configuration


def test_an_earlier_version_of_a_case_is_ignored_not_fatal(bundle: Bundle) -> None:
    """A superseded case leaves its verdict behind; the current one still counts."""
    cases, ratings, records = _agreeing(1)
    superseded = _payload("CTL-01", code="contradictory")
    older = _record("CTL-01", superseded)
    older["assessments"] = [_verdict("CTL-01", classification="incorrect")]
    bundle.write(cases, ratings, [older, records[0]])

    assert bundle.run() == 0

    report = bundle.report()
    assert report["ignored_records"] == 1
    assert report["comparisons"][0]["judge_status"] == "correct"
    assert report["overall_agreements"] == 1


def test_a_second_judge_configuration_is_refused(bundle: Bundle) -> None:
    cases, ratings, records = _agreeing()
    records[1]["configuration_sha256"] = "another-hash"
    bundle.write(cases, ratings, records)

    with pytest.raises(EvaluationInputError, match="configuration"):
        bundle.run()


def test_cases_the_reference_was_not_rated_against_are_refused(
    bundle: Bundle,
) -> None:
    cases, ratings, records = _agreeing(1)
    bundle.write(cases, ratings, records, reference_cases_sha256="0" * 64)

    with pytest.raises(EvaluationInputError, match="ratings were made against"):
        bundle.run()


def test_cases_the_criteria_were_not_fixed_for_are_refused(bundle: Bundle) -> None:
    cases, ratings, records = _agreeing(1)
    bundle.write(cases, ratings, records, criteria={"cases_sha256": "0" * 64})

    with pytest.raises(EvaluationInputError, match="acceptance criteria"):
        bundle.run()


def test_a_reference_the_criteria_were_not_fixed_for_is_refused(
    bundle: Bundle,
) -> None:
    cases, ratings, records = _agreeing(1)
    bundle.write(cases, ratings, records)
    # Same ratings, different bytes: the criteria name one exact file.
    reference = json.loads(bundle.reference.read_text(encoding="utf-8"))
    bundle.reference.write_text(json.dumps(reference, indent=2), encoding="utf-8")

    with pytest.raises(EvaluationInputError, match="acceptance"):
        bundle.run()


# ── what the counts mean ───────────────────────────────────────────────────────


def test_key_match_compares_item_ids_not_coverage(bundle: Bundle) -> None:
    cases, ratings, records = _agreeing(1)
    records[0]["assessments"] = [_verdict("CTL-01", covers=False)]
    bundle.write(cases, ratings, records)

    assert bundle.run() == 0
    assert bundle.report()["key_match_disagreements"] == 0


def test_a_different_key_match_set_is_a_disagreement(bundle: Bundle) -> None:
    cases, ratings, records = _agreeing(1)
    cases[0]["payload"] = _payload("CTL-01", items=2)
    records[0] = _record(
        "CTL-01",
        cases[0]["payload"],
        assessments=[_verdict("CTL-01", matches=["CTL-01-K01", "CTL-01-K02"])],
    )
    bundle.write(cases, ratings, records)

    assert bundle.run() == 1
    report = bundle.report()
    assert report["key_match_disagreements"] == 1
    assert report["comparisons"][0]["key_matches"] == {
        "reference": ["CTL-01-K01"],
        "judge": ["CTL-01-K01", "CTL-01-K02"],
        "agrees": False,
    }


def test_approving_a_reassuring_finding_on_a_clear_breach_fails(
    bundle: Bundle,
) -> None:
    cases, ratings, records = _agreeing(1)
    ratings[0] = _rating("CTL-01", breach=True)
    bundle.write(cases, ratings, records)

    assert bundle.run() == 1
    report = bundle.report()
    assert report["false_reassurances"] == 1
    assert report["comparisons"][0]["false_reassurance"] is True


def test_approving_a_warning_on_a_clear_breach_is_allowed(bundle: Bundle) -> None:
    cases, ratings, records = _agreeing(1)
    cases[0]["payload"] = _payload("CTL-01", code="contradictory")
    records[0] = _record("CTL-01", cases[0]["payload"])
    ratings[0] = _rating("CTL-01", breach=True)
    bundle.write(cases, ratings, records)

    assert bundle.run() == 0
    assert bundle.report()["false_reassurances"] == 0
    assert bundle.report()["comparisons"][0]["false_reassurance"] is False


def test_a_criterion_the_case_does_not_carry_is_matched(bundle: Bundle) -> None:
    """not_applicable on both sides is agreement, not a mismatched vocabulary."""
    cases, ratings, records = _agreeing(1)
    payload = _payload("CTL-01")
    payload["criteria_applicability"]["explanation"] = "not_applicable"
    payload["assessed_finding"]["explanation"] = None
    cases[0]["payload"] = payload
    ratings[0] = _rating("CTL-01", explanation="not_applicable")
    records[0] = _record(
        "CTL-01",
        payload,
        assessments=[_verdict("CTL-01", explanation="not_applicable")],
    )
    bundle.write(cases, ratings, records)

    assert bundle.run() == 0

    case = bundle.report()["comparisons"][0]
    assert case["explanation"] == {
        "reference": "not_applicable",
        "judge": "not_applicable",
        "agrees": True,
    }
    assert case["judge_status"] == "correct"


def test_agreement_below_the_threshold_fails(bundle: Bundle) -> None:
    cases, ratings, records = _agreeing()
    records[1]["assessments"] = [_verdict("CTL-02", classification="incorrect")]
    bundle.write(cases, ratings, records)

    assert bundle.run() == 1
    report = bundle.report()
    assert report["overall_agreements"] == 1
    checks = {check["name"]: check for check in report["checks"]}
    assert checks["overall_agreements"]["passed"] is False
    assert report["state"] == "failed"


def test_the_cli_turns_a_refused_input_into_an_exit_code(bundle: Bundle) -> None:
    cases, ratings, records = _agreeing(1)
    bundle.write(cases, ratings, records, cases_sha256="0" * 64)

    assert (
        check_judge.cli(
            [
                "--cases",
                str(bundle.cases),
                "--reference",
                str(bundle.reference),
                "--criteria",
                str(bundle.criteria),
                "--verdicts",
                str(bundle.verdicts),
                "--output",
                str(bundle.output),
                "--protocol",
                str(bundle.protocol),
            ]
        )
        == 2
    )


def test_case_count_must_match_the_criteria(bundle: Bundle) -> None:
    cases, ratings, records = _agreeing()
    bundle.write(cases, ratings, records, criteria={"case_count": 12})

    assert bundle.run() == 1
    checks = {check["name"]: check for check in bundle.report()["checks"]}
    assert checks["case_count"]["passed"] is False
    assert checks["case_count"]["threshold"] == 12
    assert checks["case_count"]["rule"].startswith("exactly")


def test_a_verdict_without_adapter_identity_is_refused(bundle: Bundle) -> None:
    cases, ratings, records = _agreeing()
    for record in records:
        record.pop("adapter_sha256", None)
    bundle.write(cases, ratings, records)
    with pytest.raises(EvaluationInputError, match="names no adapter"):
        bundle.run()
