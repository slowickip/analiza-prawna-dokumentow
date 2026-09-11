#!/usr/bin/env python3
"""Compare the judge's control verdicts with the reference ratings, offline.

The control sample is the gate before the full run. This script calls nothing:
it reads the cases, the ratings, the saved verdicts and the acceptance criteria,
and reports whether the thresholds fixed in the criteria file are met. The
report carries the reference type and interpretation the criteria file states.

Usage
-----
    uv run python scripts/check_judge.py \\
        --cases evaluation-data/judge/control/cases.json \\
        --reference evaluation-data/judge/control/reference-ratings.json \\
        --criteria evaluation-data/judge/control/acceptance.json \\
        --verdicts evaluation-data/results/judge/calibration-verdicts.jsonl \\
        --output evaluation-data/results/judge/control-report.json \\
        --protocol evaluation-data/scoring-protocol.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate import (  # noqa: E402
    EvaluationError,
    finding_status,
    load_judge_config,
    read_records,
    validate_reply,
)
from evaluation_inputs import (  # noqa: E402
    EvaluationInputError,
    Packet,
    calibration_packet,
    load_json,
    require,
    sha256_file,
    sha256_value,
)

STAGE = "calibration"
# The reference speaks of correct/incorrect/unresolved for every dimension; the
# judge's two criterion fields are supported/unsupported. Same judgements, two
# vocabularies, and this is the whole of the translation. A criterion the case
# does not carry is named the same on both sides.
_CRITERION = {
    "correct": "supported",
    "incorrect": "unsupported",
    "unresolved": "unresolved",
    "not_applicable": "not_applicable",
}
# A candidate that reassures where the reference sees a clear breach. A
# candidate that warns of the breach is not one of these, and may be correct.
REASSURING_CODES = ("consistent", "permissible_departure")


def _control_packets(path: Path, cases: dict[str, Any]) -> dict[str, Packet]:
    """The supplied control cases, each as the packet it was judged as.

    The packet id follows the same rule the judging path uses, so a saved
    verdict can be tied back to the case that produced it without consulting
    anything else. Each control payload carries its own key items, so the
    current answer key is not involved here.
    """
    entries = cases.get("cases")
    require(
        isinstance(entries, list) and bool(entries),
        f"{path} carries no control cases",
    )
    packets: dict[str, Packet] = {}
    for case in entries:
        case_id = str(case["case_id"])
        require(case_id not in packets, f"{path} names case {case_id} twice")
        packets[case_id] = calibration_packet(case_id, case["payload"])
    return packets


def _check_inputs(
    cases_path: Path,
    reference_path: Path,
    reference: dict[str, Any],
    criteria: dict[str, Any],
) -> None:
    """The comparison is only about these files, so it checks it has them."""
    cases_digest = sha256_file(cases_path)
    require(
        cases_digest == criteria["cases_sha256"],
        f"{cases_path} has sha256 {cases_digest}, but the acceptance criteria "
        f"were fixed for {criteria['cases_sha256']}",
    )
    require(
        cases_digest == reference["source_cases_sha256"],
        f"{cases_path} has sha256 {cases_digest}, but the ratings were made "
        f"against {reference['source_cases_sha256']}",
    )
    reference_digest = sha256_file(reference_path)
    require(
        reference_digest == criteria["reference_sha256"],
        f"{reference_path} has sha256 {reference_digest}, but the acceptance "
        f"criteria were fixed for {criteria['reference_sha256']}",
    )


def _selected(
    records: list[Any], packets: dict[str, Packet], configuration: str
) -> tuple[dict[str, dict[str, Any]], dict[str, int], int, str | None]:
    """One judgement per control case, with the failed attempts kept as counts.

    The first validated verdict is the one that counts: rejudging a case that
    already answered would be choosing an answer. A second validated verdict is
    fine only if it says the same thing; a different one is a conflict this
    script refuses to resolve on its own.

    A record belongs to this case only if it carries the case's canonical packet
    id. An earlier version of the same case has a different one and is passed
    over like any other record of something else: the superseded verdict stays
    on file without deciding anything. Claiming the current packet id while
    carrying a different payload hash is not a superseded verdict but a
    contradiction, and is refused.

    The configuration is the protocol's, never one adopted from the records, and
    the adapter has to be one throughout: an unidentified instrument cannot be
    shown to be the one the full run will use.
    """
    chosen: dict[str, dict[str, Any]] = {}
    failures = {case_id: 0 for case_id in packets}
    ignored = 0
    adapter: str | None = None
    for record in records:
        raw = record.raw
        made_for = str(raw.get("packet_id", ""))
        case_id = next(
            (name for name, item in packets.items() if item.packet_id == made_for), ""
        )
        if raw.get("stage") != STAGE or not case_id:
            ignored += 1
            continue
        packet = packets[case_id]
        digest = sha256_value(packet.payload)
        require(
            raw.get("packet_sha256") == digest,
            f"the saved verdict for {case_id} was made on a different packet "
            f"({raw.get('packet_sha256')}, expected {digest}); it does not "
            f"describe the control case supplied here",
        )
        seen = str(raw.get("configuration_sha256") or "")
        require(
            bool(seen),
            f"the saved verdict for {case_id} names no judge configuration; a "
            f"verdict that cannot say which instrument made it is not evidence "
            f"that the control sample used the configured one",
        )
        require(
            seen == configuration,
            f"the saved verdict for {case_id} was made under judge configuration "
            f"{seen}, not the configured {configuration}; the control sample has "
            f"to be the judge the protocol names",
        )
        if raw.get("status") != "validated":
            failures[case_id] += 1
            continue
        made_by = str(raw.get("adapter_sha256") or "")
        require(bool(made_by), f"the saved verdict for {case_id} names no adapter")
        if adapter is None:
            adapter = made_by
        require(
            made_by == adapter,
            f"the saved verdict for {case_id} was made by adapter {made_by}, not "
            f"{adapter}; the control sample has to be one instrument throughout",
        )
        if case_id not in chosen:
            chosen[case_id] = raw
            continue
        require(
            raw.get("assessments") == chosen[case_id].get("assessments"),
            f"case {case_id} has conflicting repeated valid verdicts; the "
            f"control sample cannot be settled by choosing between them",
        )
    return chosen, failures, ignored, adapter


def _compare(
    packet: Packet,
    view: dict[str, Any],
    rating: dict[str, Any],
    answers: dict[str, Any] | None,
    failures: int,
) -> dict[str, Any]:
    """One finding of one case: each dimension, the status, the key items matched.

    A case that states one finding names it with the case id, so a rating file
    written for the single-finding cases keeps working unchanged. A case stating
    several is rated per finding, and the reply had to answer all of them for any
    of them to be read.
    """
    reference_matches = sorted(rating["matching_issue_ids"])
    finding_id = str(view["finding_id"])
    row: dict[str, Any] = {
        "case_id": finding_id,
        "candidate_code": str(view.get("code", "")),
        "clear_breach_of_a_provision": rating["clear_breach_of_a_provision"],
        "technical_failures": failures,
        "validated": answers is not None,
        "reference_status": rating["overall"],
        "reference_rationale": rating.get("rationale"),
    }
    if answers is None:
        row.update(
            {
                "judge_status": None,
                "judge_rationale": None,
                "agrees": False,
                "key_matches": {
                    "reference": reference_matches,
                    "judge": None,
                    "agrees": False,
                },
                "false_reassurance": False,
            }
        )
        return row

    # Revalidated against the packet, not merely read: the reply has to still be
    # a reply about these key items and these sources.
    verdict = answers[finding_id]
    status = finding_status(verdict, [])
    judge_matches = sorted(match.item_id for match in verdict.key_matches)
    for dimension, expected in (
        ("classification", rating["classification"]),
        ("legal_basis", _CRITERION[rating["legal_basis"]]),
        ("explanation", _CRITERION[rating["explanation"]]),
    ):
        answered = getattr(verdict, dimension)
        row[dimension] = {
            "reference": expected,
            "judge": answered,
            "agrees": answered == expected,
        }
    row.update(
        {
            "judge_status": status,
            "judge_rationale": verdict.reason,
            "agrees": status == rating["overall"],
            "key_matches": {
                "reference": reference_matches,
                "judge": judge_matches,
                "agrees": judge_matches == reference_matches,
            },
            "false_reassurance": (
                rating["clear_breach_of_a_provision"] is True
                and row["candidate_code"] in REASSURING_CODES
                and status == "correct"
            ),
        }
    )
    return row


def _totals(rows: list[dict[str, Any]]) -> dict[str, int]:
    """The counts the acceptance criteria and the report summary both read."""
    return {
        "validated": sum(1 for row in rows if row["validated"]),
        "overall_agreements": sum(1 for row in rows if row["agrees"]),
        "key_match_disagreements": sum(
            1 for row in rows if not row["key_matches"]["agrees"]
        ),
        "false_reassurances": sum(1 for row in rows if row["false_reassurance"]),
        "technical_failures": sum(row["technical_failures"] for row in rows),
    }


def _checks(
    criteria: dict[str, Any], rows: list[dict[str, Any]], totals: dict[str, int]
) -> list[dict[str, Any]]:
    """Every acceptance criterion, with the threshold it was fixed at."""
    validated = totals["validated"]
    agreements = totals["overall_agreements"]
    key_match_disagreements = totals["key_match_disagreements"]
    reassurances = totals["false_reassurances"]
    return [
        {
            "name": "case_count",
            "rule": "exactly as many control cases as the criteria fix",
            "threshold": criteria["case_count"],
            "observed": len(rows),
            "passed": len(rows) == criteria["case_count"],
        },
        {
            "name": "cases_validated",
            "rule": "every control case has a validated verdict",
            "threshold": len(rows)
            if criteria["all_cases_must_have_validated_verdicts"]
            else 0,
            "observed": validated,
            "passed": validated == len(rows)
            or not criteria["all_cases_must_have_validated_verdicts"],
        },
        {
            "name": "overall_agreements",
            "rule": "overall status agrees with the reference at least this often",
            "threshold": criteria["minimum_overall_agreements"],
            "observed": agreements,
            "passed": agreements >= criteria["minimum_overall_agreements"],
        },
        {
            "name": "key_match_disagreements",
            "rule": "matched key issue ids differ at most this often",
            "threshold": criteria["maximum_key_match_disagreements"],
            "observed": key_match_disagreements,
            "passed": key_match_disagreements
            <= criteria["maximum_key_match_disagreements"],
        },
        {
            "name": "false_reassurances",
            "rule": "a reassuring candidate is never approved over a clear breach",
            "threshold": criteria["maximum_false_reassurances"],
            "observed": reassurances,
            "passed": reassurances <= criteria["maximum_false_reassurances"],
        },
    ]


def build_report(
    cases_path: Path,
    reference_path: Path,
    criteria_path: Path,
    verdicts_path: Path,
    protocol_path: Path,
) -> dict[str, Any]:
    criteria = load_json(criteria_path)
    reference = load_json(reference_path)
    _check_inputs(cases_path, reference_path, reference, criteria)
    packets = _control_packets(cases_path, load_json(cases_path))
    ratings: dict[str, Any] = {}
    for rating in reference["ratings"]:
        case_id = str(rating["case_id"])
        require(
            case_id not in ratings,
            f"{reference_path} rates case {case_id} twice; which rating the "
            f"comparison used would then depend on file order",
        )
        ratings[case_id] = rating
    rated = {
        str(view["finding_id"]): (case_id, packet, view)
        for case_id, packet in packets.items()
        for view in packet.payload["assessed_findings"]
    }
    missing = sorted(set(rated) - set(ratings))
    require(not missing, f"{reference_path} has no rating for {', '.join(missing)}")

    config = load_judge_config(load_json(protocol_path), protocol_path.parent)
    chosen, failures, ignored, adapter = _selected(
        read_records(verdicts_path), packets, config.configuration_sha256
    )
    answered = {
        case_id: validate_reply(
            {"assessments": raw.get("assessments")}, packets[case_id]
        )
        for case_id, raw in chosen.items()
    }
    rows = [
        _compare(
            packet,
            view,
            ratings[finding_id],
            answered.get(case_id),
            failures[case_id],
        )
        for finding_id, (case_id, packet, view) in rated.items()
    ]
    totals = _totals(rows)
    checks = _checks(criteria, rows, totals)
    return {
        "schema_version": 1,
        "state": "passed" if all(check["passed"] for check in checks) else "failed",
        "reference_type": criteria["reference_type"],
        "reference_reviewer": reference.get("reviewer"),
        "reference_model_identifier": reference.get("exact_model_identifier"),
        "interpretation": criteria["interpretation"],
        "model": config.model,
        "reasoning_effort": config.effort,
        "configuration_sha256": config.configuration_sha256,
        "adapter_sha256": adapter,
        "cases_sha256": criteria["cases_sha256"],
        "reference_sha256": criteria["reference_sha256"],
        "criteria": criteria,
        "cases": len(rows),
        "overall_agreements": totals["overall_agreements"],
        "agreement_rate": (totals["overall_agreements"] / len(rows) if rows else None),
        "key_match_disagreements": totals["key_match_disagreements"],
        "false_reassurances": totals["false_reassurances"],
        "technical_failures": totals["technical_failures"],
        "ignored_records": ignored,
        "checks": checks,
        "comparisons": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--criteria", type=Path, required=True)
    parser.add_argument("--verdicts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        required=True,
        help="scoring protocol naming the judge the verdicts must have been made by",
    )
    args = parser.parse_args(argv)

    report = build_report(
        args.cases, args.reference, args.criteria, args.verdicts, args.protocol
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    failed = [check["name"] for check in report["checks"] if not check["passed"]]
    if failed:
        print(
            f"control sample not accepted: {', '.join(failed)}. See {args.output}.",
            file=sys.stderr,
        )
        return 1
    print(
        f"control sample accepted: {report['overall_agreements']}/{report['cases']} "
        f"overall agreements, {report['key_match_disagreements']} key-match "
        f"disagreements, {report['false_reassurances']} false reassurances. "
        f"References are model-prepared, not expert ratings."
    )
    return 0


def cli(argv: list[str] | None = None) -> int:
    try:
        return main(argv)
    except (EvaluationError, EvaluationInputError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(cli())
