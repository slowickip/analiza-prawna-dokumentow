#!/usr/bin/env python3
"""One evaluation path: prepare sources, judge saved findings, report offline.

The recorded analyzer runs are never repeated here. Each substantive finding they
saved is judged once against the document, the reviewed answer key and the real
text of the frozen corpus, and every judgement is written down as it is made. The
report is then computed from those saved judgements alone, with no model call, so
the tables can be reproduced from the artefact by anyone holding the verdicts.

Commands:

    evaluate.py prepare   # fetch and check the document and provision text
    evaluate.py packets   # build and validate the judging packets, offline
    evaluate.py run       # judge the pending packets, saving each immediately
    evaluate.py report    # recompute every figure from the saved judgements
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from collections import deque
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from threading import Event, Lock
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "backend" / "src"))
sys.path.insert(0, str(ROOT / "seeder" / "src"))

from evaluation_inputs import (  # noqa: E402
    CALIBRATION_ID_PREFIX,
    NON_SUBSTANTIVE_CODES,
    Accounted,
    AnswerKey,
    EvaluationInputError,
    Packet,
    RecordedRuns,
    RunFinding,
    SourceBundle,
    account_findings,
    basis_problems,
    build_packets,
    canonical_json,
    load_calibration_packets,
    load_json,
    load_key,
    load_runs,
    read_locator,
    require,
    sha256_value,
)

DATA_DIR = ROOT / "evaluation-data"
# How many judge conversations may be open at once. Each one costs its own
# runtime home and session, and the ceiling is a convention rather than a
# measured limit of the service.
MAX_WORKERS = 4
ERROR_TAGS = (
    "wrong_classification",
    "basis_does_not_support",
    "basis_not_in_corpus",
    "quote_not_in_document",
    "claim_not_in_document",
    "expectation_missed",
    "outside_the_key",
    "overreach",
    "other",
)


class EvaluationError(RuntimeError):
    """The evaluation could not be carried out as its protocol requires."""


# --------------------------------------------------------------------------
# what the judge must answer
# --------------------------------------------------------------------------


class KeyMatch(BaseModel):
    """One key expectation the judge says this finding is actually about."""

    model_config = ConfigDict(extra="forbid")

    item_id: str
    covers_expectation: bool
    reason: str = Field(min_length=1)


class Evidence(BaseModel):
    """Where in the supplied material the judgement is grounded."""

    model_config = ConfigDict(extra="forbid")

    source: str = Field(min_length=1)
    quote: str = Field(min_length=1)


class JudgeVerdict(BaseModel):
    """The judge's answer about one finding, in the only shape accepted."""

    model_config = ConfigDict(extra="forbid")

    # Which finding of the packet this answer is about. The packet names its
    # findings by their recorded ids, which are minted uuids: the reply can be
    # matched to a finding without the judge being told which run it came from.
    finding_id: str = Field(min_length=1)
    key_matches: list[KeyMatch]
    classification: Literal["correct", "incorrect", "unresolved"]
    legal_basis: Literal["supported", "unsupported", "unresolved", "not_applicable"]
    explanation: Literal["supported", "unsupported", "unresolved", "not_applicable"]
    evidence: list[Evidence]
    reason: str = Field(min_length=1)
    # Not optional, and not for style: the structured-output endpoint rejects a
    # schema whose `required` omits any property, so a default here would have
    # made every call fail. An empty list is the way to report no error tag.
    error_tags: list[Literal[ERROR_TAGS]]  # type: ignore[valid-type]


class JudgeReply(BaseModel):
    """One reply about one whole run: an answer for each finding it carried."""

    model_config = ConfigDict(extra="forbid")

    assessments: list[JudgeVerdict]


def response_schema() -> dict[str, Any]:
    """The JSON schema the judge's reply is constrained to and checked against."""
    return JudgeReply.model_json_schema()


# --------------------------------------------------------------------------
# configuration, read from the protocol rather than from flags
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class JudgeConfig:
    model: str
    effort: str
    instructions: str
    instructions_sha256: str
    max_packet_chars: int
    bundle_dir: Path
    verdicts_path: Path
    calibration_verdicts_path: Path
    report_path: Path
    ready_for_full_run: bool
    readiness_note: str

    @property
    def configuration_sha256(self) -> str:
        """What a saved judgement has to have been produced under to be reused."""
        return sha256_value(
            {
                "model": self.model,
                "effort": self.effort,
                "instructions_sha256": self.instructions_sha256,
                "response_schema": response_schema(),
            }
        )


def load_judge_config(protocol: dict[str, Any], data_dir: Path) -> JudgeConfig:
    section = protocol.get("judge")
    require(isinstance(section, dict), "the scoring protocol has no judge section")
    instructions_path = data_dir / section["instructions_file"]
    require(
        instructions_path.is_file(),
        f"the judge instructions file is missing: {instructions_path}",
    )
    instructions = instructions_path.read_text(encoding="utf-8")
    readiness = protocol.get("readiness") or {}
    return JudgeConfig(
        model=str(section["model"]),
        effort=str(section["reasoning_effort"]),
        instructions=instructions,
        instructions_sha256=sha256_value(instructions),
        max_packet_chars=int(section["max_packet_chars"]),
        bundle_dir=data_dir / section["source_bundle_dir"],
        verdicts_path=data_dir / section["verdicts_file"],
        calibration_verdicts_path=data_dir / section["calibration_verdicts_file"],
        report_path=data_dir / section["report_file"],
        ready_for_full_run=readiness.get("ready_for_full_run") is True,
        readiness_note=str(readiness.get("what_is_outstanding", "")),
    )


# --------------------------------------------------------------------------
# saved judgements
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Record:
    """One saved judgement, exactly as it was written when it was made."""

    raw: dict[str, Any]

    @property
    def packet_id(self) -> str:
        return str(self.raw.get("packet_id", ""))

    @property
    def status(self) -> str:
        return str(self.raw.get("status", ""))

    def reusable(
        self,
        packet: Packet,
        config: JudgeConfig,
        bundle: str,
        key_sha256: str,
        adapter_sha256: str | None = None,
    ) -> bool:
        """Whether this judgement still describes the case it would be reused for.

        A changed packet, prompt, model, source text or answer key makes it a
        judgement of something else. The key hash binds the full set of
        expectations and the completeness denominator to the saved judgement.

        A calibration case is the one exception, because it carries its own
        expectations and its own source text inside the packet: the packet hash
        already covers everything it was judged against, and the cohort key is
        not among them. Editing that key says nothing about such a judgement, so
        it may not pay for the case again.

        ``adapter_sha256`` is checked only where the adapter is loaded, which is
        the judging path. A report read offline cannot import it, and says which
        instruments its judgements came from instead of pretending to check.
        """
        return (
            self.status == "validated"
            and self.raw.get("packet_sha256") == sha256_value(packet.payload)
            and self.raw.get("configuration_sha256") == config.configuration_sha256
            and self.raw.get("source_bundle_sha256") == bundle
            and (
                packet.packet_id.startswith(CALIBRATION_ID_PREFIX)
                or self.raw.get("answer_key_sha256") == key_sha256
            )
            and (
                adapter_sha256 is None
                or self.raw.get("adapter_sha256") == adapter_sha256
            )
        )


def read_records(path: Path) -> list[Record]:
    if not path.is_file():
        return []
    records: list[Record] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EvaluationError(f"{path} line {number} is not JSON: {exc}") from exc
        if not isinstance(value, dict):
            raise EvaluationError(f"{path} line {number} is not a JSON object")
        records.append(Record(value))
    return records


def append_record(path: Path, record: dict[str, Any]) -> None:
    """Write one judgement immediately; evidence of a call is never overwritten."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(canonical_json(record) + "\n")
        handle.flush()


# --------------------------------------------------------------------------
# checking a reply and turning it into a status
# --------------------------------------------------------------------------


def validate_reply(result: dict[str, Any], packet: Packet) -> dict[str, JudgeVerdict]:
    """Schema, then membership: one answer for each finding the packet carried.

    A reply that omits a finding, answers one twice or names a finding the packet
    does not carry is invalid technical output. It is never read as a judgement
    of the findings it did answer, and never becomes a mistake of the analyzer:
    the packet asked a fixed set of questions and the answer sheet has to match
    it exactly.
    """
    try:
        reply = JudgeReply.model_validate(result)
    except ValidationError as exc:
        raise EvaluationError(
            f"judge reply for packet {packet.packet_id} does not satisfy the "
            f"response schema: {exc.errors(include_url=False)}"
        ) from exc
    answered = [assessment.finding_id for assessment in reply.assessments]
    expected = set(packet.finding_ids)
    duplicated = sorted({name for name in answered if answered.count(name) > 1})
    foreign = sorted(set(answered) - expected)
    absent = sorted(expected - set(answered))
    if duplicated or foreign or absent:
        raise EvaluationError(
            f"judge reply for packet {packet.packet_id} does not answer its "
            f"{len(expected)} finding(s) exactly once: "
            f"{len(absent)} unanswered ({', '.join(absent[:5])}), "
            f"{len(duplicated)} answered twice ({', '.join(duplicated[:5])}), "
            f"{len(foreign)} not in the packet ({', '.join(foreign[:5])})"
        )
    views = {
        str(view["finding_id"]): view for view in packet.payload["assessed_findings"]
    }
    for assessment in reply.assessments:
        validate_verdict(assessment, packet, views[assessment.finding_id])
    return {assessment.finding_id: assessment for assessment in reply.assessments}


def validate_verdict(
    verdict: JudgeVerdict, packet: Packet, view: dict[str, Any]
) -> JudgeVerdict:
    """One answer, checked against the packet: a reply about items or sources
    that are not in the packet is an invalid reply, not a judgement of a finding.

    The packet is the whole contract, so what may be named is read from the
    packet rather than from the key. A control case carries its own expectations
    and is checked by the same rule.
    """
    allowed_items = {
        str(item["item_id"])
        for item in packet.payload["key_items"]
        if "item_id" in item
    }
    unknown = sorted(
        match.item_id
        for match in verdict.key_matches
        if match.item_id not in allowed_items
    )
    if unknown:
        raise EvaluationError(
            f"judge reply for packet {packet.packet_id} names key items that were "
            f"not in the packet: {', '.join(unknown)}"
        )
    seen: set[str] = set()
    for match in verdict.key_matches:
        if match.item_id in seen:
            raise EvaluationError(
                f"judge reply for packet {packet.packet_id} names key item "
                f"{match.item_id} twice"
            )
        seen.add(match.item_id)
    _check_evidence(verdict, packet)
    _check_applicability(verdict, packet, view)
    return verdict


def _normalized(text: str) -> str:
    """Whitespace-folded text, so a rewrapped quote still has to be a quote."""
    return " ".join(text.split())


def _packet_sources(packet: Packet) -> dict[str, str]:
    """Every source the judge may cite, with the text it actually carries."""
    sources = {"document": str(packet.payload["document"]["text"])}
    for group in packet.payload["provisions"].values():
        for view in group:
            sources[str(view["locator"])] = str(view.get("text", ""))
    return sources


def _check_evidence(verdict: JudgeVerdict, packet: Packet) -> None:
    """A cited quote has to be in the source it cites, and a decision needs one.

    Both halves guard the same thing. A verdict that settles a criterion without
    pointing at any source has not been grounded in the material, and a quote
    that is not in the source it names is not evidence from the packet, whatever
    else it may be. Neither is an error of the analyzer; both are replies this
    procedure does not accept.
    """
    sources = _packet_sources(packet)
    unknown = sorted(
        evidence.source
        for evidence in verdict.evidence
        if evidence.source not in sources
    )
    if unknown:
        raise EvaluationError(
            f"judge reply for packet {packet.packet_id} cites sources that were not "
            f"in the packet: {', '.join(unknown)}"
        )
    ungrounded = sorted(
        f"{evidence.source}: {evidence.quote[:60]}"
        for evidence in verdict.evidence
        if _normalized(evidence.quote) not in _normalized(sources[evidence.source])
    )
    if ungrounded:
        raise EvaluationError(
            f"judge reply for packet {packet.packet_id} quotes text that is not in "
            f"the source it names: {'; '.join(ungrounded)}"
        )
    decided = {verdict.classification, verdict.legal_basis, verdict.explanation} & {
        "correct",
        "incorrect",
        "supported",
        "unsupported",
    }
    if decided and not verdict.evidence:
        raise EvaluationError(
            f"judge reply for packet {packet.packet_id} settles "
            f"{', '.join(sorted(decided))} without citing any source. A decided "
            f"criterion has to rest on the material in the packet."
        )
    _check_basis_evidence(verdict, packet, sources)


def _check_basis_evidence(
    verdict: JudgeVerdict, packet: Packet, sources: dict[str, str]
) -> None:
    """A decided basis must cite a provision, not the contract alone.

    The criterion asks whether the cited law supports the claim, so a reply that
    settles it from the document only has answered another question. Where the
    packet carries no provision, the document is the only source there is and the
    absence itself is what is read.
    """
    if verdict.legal_basis not in {"supported", "unsupported"}:
        return
    provisions = set(sources) - {"document"}
    if not provisions:
        return
    if not any(evidence.source in provisions for evidence in verdict.evidence):
        raise EvaluationError(
            f"judge reply for packet {packet.packet_id} settles the legal basis "
            f"citing the document only, while the packet carries "
            f"{len(provisions)} provision text(s). A decided basis has to cite at "
            f"least one of them."
        )


def _check_applicability(
    verdict: JudgeVerdict, packet: Packet, view: dict[str, Any]
) -> None:
    """The packet decides which criteria apply; the judge may not excuse itself.

    A criterion the packet declares required may not come back not_applicable --
    that would take the finding out of a denominator it belongs in -- and one the
    packet declares not applicable may not come back judged, because there is
    nothing in the record to judge. Each finding of a run declares its own, so
    the declaration is read from the finding the answer is about.
    """
    applicability = view["criteria_applicability"]
    for criterion in ("legal_basis", "explanation"):
        declared = applicability[criterion]
        answered = getattr(verdict, criterion)
        if declared == "required" and answered == "not_applicable":
            raise EvaluationError(
                f"judge reply for packet {packet.packet_id} calls {criterion} not "
                f"applicable, but this finding carries it and the criterion is "
                f"required"
            )
        if declared == "not_applicable" and answered != "not_applicable":
            raise EvaluationError(
                f"judge reply for packet {packet.packet_id} judges {criterion}, "
                f"which the finding does not carry; the criterion is not applicable "
                f"there"
            )


def deterministic_errors(
    finding: RunFinding, bundle: SourceBundle, key: AnswerKey
) -> list[str]:
    """Failures settled by reading the record, before any judgement is weighed."""
    errors: list[str] = []
    if finding.raw.get("quote_resolution") == "quote_fabricated":
        errors.append("quote_not_in_document")
    uncitable, _ = basis_problems(finding, bundle, key)
    if uncitable:
        errors.append("basis_not_in_corpus")
    return errors


def finding_status(verdict: JudgeVerdict, errors: list[str]) -> str:
    """Correct, incorrect or unresolved, computed here and never by the judge.

    A failed required criterion is an error. No failure but an unsettled criterion
    is unresolved. A criterion the record cannot carry -- the per-finding
    explanation the analyzer never wrote -- is not applicable and neither helps
    nor harms.
    """
    criteria = (verdict.classification, verdict.legal_basis, verdict.explanation)
    if errors or "incorrect" in criteria or "unsupported" in criteria:
        return "incorrect"
    if "unresolved" in criteria:
        return "unresolved"
    return "correct"


# --------------------------------------------------------------------------
# loading everything the offline side needs
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Inputs:
    protocol: dict[str, Any]
    config: JudgeConfig
    runs: RecordedRuns
    key: AnswerKey
    accounted: tuple[Accounted, ...]
    bundle: SourceBundle
    packets: tuple[Packet, ...]


def load_inputs(data_dir: Path) -> Inputs:
    protocol = load_json(data_dir / "scoring-protocol.json")
    config = load_judge_config(protocol, data_dir)
    runs = load_runs(data_dir, protocol)
    key = load_key(
        data_dir / "answer-key.json",
        runs,
        protocol["scored_runs"].get("answer_key_sha256"),
    )
    accounted = account_findings(runs.findings)
    bundle = SourceBundle.load(config.bundle_dir)
    bundle.bind_to_runs(runs)
    packets = build_packets(accounted, key, bundle, runs, config.max_packet_chars)
    return Inputs(protocol, config, runs, key, accounted, bundle, packets)


# --------------------------------------------------------------------------
# the report
# --------------------------------------------------------------------------


def _ratio(numerator: int, denominator: int) -> dict[str, Any]:
    """A rate with its counts. A zero denominator is undefined, never zero."""
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": float(Fraction(numerator, denominator)) if denominator else None,
    }


def _macro(values: list[dict[str, Any]]) -> float | None:
    """The document macro average, or undefined if any document is undefined.

    An average over a reduced document set is a different quantity, and comparing
    one with another is exactly the arithmetic this evaluation refuses.
    """
    defined = [value["rate"] for value in values if value["rate"] is not None]
    if not values or len(defined) != len(values):
        return None
    return sum(defined) / len(defined)


def _micro(values: list[dict[str, Any]]) -> dict[str, Any]:
    return _ratio(
        sum(int(value["numerator"]) for value in values),
        sum(int(value["denominator"]) for value in values),
    )


@dataclass
class Judged:
    """One judged finding, joined back to the run it came from.

    What the judgement cost is not here: one call answered the whole run, so the
    resources belong to the run and are accounted once, in its cell.
    """

    accounted: Accounted
    verdict: JudgeVerdict
    status: str
    errors: list[str]


def _usage_tokens(usage: Any, field: str) -> int | None:
    """One token count out of the runtime's nested usage record.

    The adapter hands over the SDK's own dump, which reports a breakdown for the
    last turn and a total for the thread under ``total``. One thread per packet
    makes those equal; the total is read because that is what the field means.
    """
    if not isinstance(usage, dict):
        return None
    total = usage.get("total")
    if not isinstance(total, dict):
        return None
    value = total.get(field)
    return int(value) if isinstance(value, int) else None


def _judged_from(entry: Accounted, verdict: JudgeVerdict, inputs: Inputs) -> Judged:
    errors = deterministic_errors(entry.finding, inputs.bundle, inputs.key)
    return Judged(
        accounted=entry,
        verdict=verdict,
        status=finding_status(verdict, errors),
        errors=errors,
    )


def _record_resources(record: Record) -> dict[str, Any]:
    """What the one call for a run cost, read from the record that made it."""
    usage = record.raw.get("usage")
    return {
        "calls": 1,
        "elapsed_ms": int(record.raw.get("elapsed_ms") or 0),
        "input_tokens": _usage_tokens(usage, "input_tokens"),
        "output_tokens": _usage_tokens(usage, "output_tokens"),
    }


def collect_judged(
    inputs: Inputs, records: list[Record]
) -> tuple[
    dict[str, Judged],
    list[dict[str, Any]],
    dict[str, int],
    dict[tuple[int, str, str], dict[str, Any]],
]:
    """Reduce the saved attempts per run packet into at most one reply each, and
    expand each accepted reply into the per-finding judgements the report reads.

    A packet's file may hold several lines: a technical failure and then the
    reply that followed it, or -- if the store were ever appended to twice -- a
    second answer to a question already answered. The first reusable, valid reply
    owns the result and later ones never replace it, which is what makes "no
    retry after a verdict" a property of the reader and not only of the writer.
    Failures stay counted as evidence; they stop being a problem once the packet
    has an answer, because a run that stopped and was resumed is exactly what
    this procedure expects.

    An incomplete or otherwise unacceptable reply is one invalid reply for the
    whole run, not a partial result: none of its answers is read.
    """
    bundle_digest = inputs.bundle.digest
    by_packet = {packet.packet_id: packet for packet in inputs.packets}
    entry_of = {
        entry.finding.finding_id: entry
        for entry in inputs.accounted
        if entry.disposition == "judged"
    }
    grouped: dict[str, list[Record]] = {}
    problems: list[dict[str, Any]] = []
    for record in records:
        if record.packet_id in by_packet:
            grouped.setdefault(record.packet_id, []).append(record)
        else:
            problems.append(
                {"packet_id": record.packet_id, "problem": "no_matching_packet"}
            )

    judged: dict[str, Judged] = {}
    resources: dict[tuple[int, str, str], dict[str, Any]] = {}
    counts = {
        "failed_attempts": 0,
        "stale_attempts": 0,
        "invalid_verdicts": 0,
        "superseded_repeat_verdicts": 0,
        "conflicting_repeat_verdicts": 0,
    }
    for packet_id, attempts in sorted(grouped.items()):
        packet = by_packet[packet_id]
        cell = (packet.series, packet.document_key, packet.arm)
        settled: dict[str, JudgeVerdict] | None = None
        blocking: list[dict[str, Any]] = []
        for record in attempts:
            if record.status != "validated":
                counts["failed_attempts"] += 1
                blocking.append(
                    {
                        "packet_id": packet_id,
                        "problem": "judge_error",
                        "detail": str(record.raw.get("error", ""))[:200],
                    }
                )
                continue
            if not record.reusable(
                packet, inputs.config, bundle_digest, inputs.key.sha256
            ):
                counts["stale_attempts"] += 1
                blocking.append(
                    {"packet_id": packet_id, "problem": "stale_configuration"}
                )
                continue
            saved = record.raw.get("assessments")
            if not isinstance(saved, list):
                counts["invalid_verdicts"] += 1
                blocking.append(
                    {
                        "packet_id": packet_id,
                        "problem": "invalid_verdict",
                        "detail": (
                            "the record is marked validated and carries no "
                            "assessments list, so there is nothing to read as a "
                            "judgement of this run"
                        ),
                    }
                )
                continue
            try:
                answers = validate_reply({"assessments": saved}, packet)
            except EvaluationError as exc:
                counts["invalid_verdicts"] += 1
                blocking.append(
                    {
                        "packet_id": packet_id,
                        "problem": "invalid_verdict",
                        "detail": str(exc)[:200],
                    }
                )
                continue
            if settled is not None:
                if answers == settled:
                    counts["superseded_repeat_verdicts"] += 1
                else:
                    counts["conflicting_repeat_verdicts"] += 1
                    problems.append(
                        {
                            "packet_id": packet_id,
                            "problem": "conflicting_repeat_verdict",
                            "detail": (
                                "a later judgement of this packet disagrees with the "
                                "one that stands; the first is kept and the "
                                "disagreement is reported"
                            ),
                        }
                    )
                continue
            settled = answers
            resources[cell] = _record_resources(record)
            for finding_id, verdict in answers.items():
                judged[finding_id] = _judged_from(entry_of[finding_id], verdict, inputs)
        if settled is None:
            problems.extend(blocking)
    problems.extend(_mixed_instrument(records, set(by_packet)))
    return judged, problems, counts, resources


def _mixed_instrument(records: list[Record], cohort: set[str]) -> list[dict[str, Any]]:
    """Judgements from two adapter identities are not one measurement.

    The offline side cannot import the adapter, so it checks that the validated
    judgements of this cohort name one identity. A record naming none counts as
    its own identity rather than matching any.
    """
    instruments = sorted(
        {
            str(record.raw.get("adapter_sha256") or "unknown")
            for record in records
            if record.packet_id in cohort and record.status == "validated"
        }
    )
    if len(instruments) < 2:
        return []
    return [
        {
            "packet_id": "",
            "problem": "instrument_mixed",
            "detail": (
                "validated judgements of this cohort were made under "
                f"{len(instruments)} adapter identities "
                f"({', '.join(identity[:12] for identity in instruments)}); one "
                "instrument judges one report"
            ),
        }
    ]


def _sum_or_none(values: list[int | None]) -> int | None:
    present = [value for value in values if value is not None]
    return sum(present) if present and len(present) == len(values) else None


def _cell_figures(
    inputs: Inputs,
    cell: tuple[int, str, str],
    judged: list[Judged],
    duplicates: int,
    non_substantive: dict[str, int],
    resources: dict[str, Any] | None = None,
) -> dict[str, Any]:
    series, document_key, arm = cell
    correct = [item for item in judged if item.status == "correct"]
    incorrect = [item for item in judged if item.status == "incorrect"]
    unresolved = [item for item in judged if item.status == "unresolved"]
    scored_items = inputs.key.items_for(document_key)
    scored_ids = {item["item_id"] for item in scored_items}

    # An unresolved anchor leaves the passage the finding concerns unestablished,
    # so a correct statement made from it cannot be credited with covering an
    # expectation about a particular passage. It still counts as a correct answer
    # in precision: what is missing is the location, not the correctness.
    covered: set[str] = set()
    for item in correct:
        if not item.accounted.finding.anchor_resolved:
            continue
        for match in item.verdict.key_matches:
            if match.covers_expectation and match.item_id in scored_ids:
                covered.add(match.item_id)
    charged: set[str] = set()
    for item in incorrect:
        for match in item.verdict.key_matches:
            if match.item_id in scored_ids:
                charged.add(match.item_id)

    # Each criterion is its positive answers over the answers that settled it;
    # an unresolved or inapplicable one is in neither count.
    accuracy = {
        criterion: _ratio(
            sum(1 for item in judged if getattr(item.verdict, criterion) == positive),
            sum(
                1
                for item in judged
                if getattr(item.verdict, criterion) in {positive, negative}
            ),
        )
        for criterion, positive, negative in (
            ("classification", "correct", "incorrect"),
            ("legal_basis", "supported", "unsupported"),
            ("explanation", "supported", "unsupported"),
        )
    }

    metrics = inputs.runs.cell_metrics(series, document_key, arm)
    return {
        "series": series,
        "document_key": document_key,
        "arm": arm,
        "stratum": inputs.protocol["scored_runs"]["documents"][document_key]["stratum"],
        "counts": {
            "correct": len(correct),
            "incorrect": len(incorrect),
            "unresolved": len(unresolved),
            "judged": len(judged),
            "exact_duplicates_not_judged": duplicates,
            "non_substantive": dict(sorted(non_substantive.items())),
            "non_substantive_total": sum(non_substantive.values()),
            "correct_outside_the_key": sum(
                1 for item in correct if not item.verdict.key_matches
            ),
            "unresolved_anchor": sum(
                1 for item in judged if not item.accounted.finding.anchor_resolved
            ),
            "items_with_a_correct_and_an_incorrect_answer": len(covered & charged),
            "deterministic_errors": sum(len(item.errors) for item in judged),
            "error_tags": _tag_counts(judged),
        },
        "dimensions": {
            "precision_of_resolved_findings": _ratio(
                len(correct), len(correct) + len(incorrect)
            ),
            "completeness_against_key": _ratio(len(covered), len(scored_items)),
            "unresolved_share": _ratio(len(unresolved), len(judged)),
            "confirmed_share": _ratio(len(correct), len(judged)),
            "classification_accuracy": accuracy["classification"],
            "legal_basis_accuracy": accuracy["legal_basis"],
            "explanation_accuracy": accuracy["explanation"],
        },
        "analyzer_resources": metrics,
        # One call judged this run, so this is that call, not a sum over findings.
        "judge_resources": resources
        or {"calls": 0, "elapsed_ms": 0, "input_tokens": None, "output_tokens": None},
    }


def _tag_counts(judged: list[Judged]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in judged:
        for tag in list(item.verdict.error_tags) + item.errors:
            counts[str(tag)] = counts.get(str(tag), 0) + 1
    return dict(sorted(counts.items()))


DIMENSIONS = (
    "precision_of_resolved_findings",
    "completeness_against_key",
    "unresolved_share",
    "confirmed_share",
    "classification_accuracy",
    "legal_basis_accuracy",
    "explanation_accuracy",
)


def _aggregate(cells: list[dict[str, Any]], label: str) -> dict[str, Any]:
    return {
        "scope": label,
        "documents": sorted({cell["document_key"] for cell in cells}),
        "dimensions": {
            dimension: {
                "macro_over_documents": _macro(
                    [cell["dimensions"][dimension] for cell in cells]
                ),
                "micro": _micro([cell["dimensions"][dimension] for cell in cells]),
            }
            for dimension in DIMENSIONS
        },
        "analyzer_resources": {
            field: _sum_or_none([cell["analyzer_resources"][field] for cell in cells])
            for field in ("elapsed_ms", "input_tokens", "output_tokens")
        },
        "judge_resources": {
            "elapsed_ms": sum(cell["judge_resources"]["elapsed_ms"] for cell in cells),
            "input_tokens": _sum_or_none(
                [cell["judge_resources"]["input_tokens"] for cell in cells]
            ),
            "output_tokens": _sum_or_none(
                [cell["judge_resources"]["output_tokens"] for cell in cells]
            ),
        },
    }


def _paired_differences(
    cells: list[dict[str, Any]], series: int
) -> list[dict[str, Any]]:
    """MID over OFF and ON over MID, on the documents where both are defined."""
    by_arm: dict[str, dict[str, dict[str, Any]]] = {}
    for cell in cells:
        by_arm.setdefault(cell["arm"], {})[cell["document_key"]] = cell
    differences: list[dict[str, Any]] = []
    for later, earlier in (("mid", "off"), ("on", "mid")):
        for dimension in DIMENSIONS:
            per_document: dict[str, float | None] = {}
            for document_key in sorted(by_arm.get(later, {})):
                left = by_arm[later][document_key]["dimensions"][dimension]["rate"]
                right = (
                    by_arm.get(earlier, {})
                    .get(document_key, {})
                    .get("dimensions", {})
                    .get(dimension, {})
                    .get("rate")
                )
                per_document[document_key] = (
                    None if left is None or right is None else left - right
                )
            defined = [value for value in per_document.values() if value is not None]
            differences.append(
                {
                    "series": series,
                    "pair": f"{later.upper()}-{earlier.upper()}",
                    "dimension": dimension,
                    "per_document": per_document,
                    "macro_difference": (
                        sum(defined) / len(defined)
                        if defined and len(defined) == len(per_document)
                        else None
                    ),
                }
            )
    return differences


def decide_comparison(
    coverage: float | None,
    precision: float | None,
    basis: float | None,
    minimum_count: int,
    thresholds: dict[str, Any],
    complete: bool,
) -> str:
    """Which fixed outcome this pair reached; evidence is weighed before direction.

    An undefined dimension, or an arm below the fixed minimum of decided
    findings, is undecided in both directions: charging a loss takes the same
    evidence as crediting a gain. A pair that passes that test is read for a
    loss, then for a gain.
    """
    if not complete:
        return "incomplete"
    # Arithmetic at an inclusive decimal boundary can differ by one float ulp.
    epsilon = 1e-12
    if (
        coverage is None
        or precision is None
        or basis is None
        or minimum_count < thresholds["minimum_decided_findings_per_arm"]
    ):
        return "undecided"
    if (
        coverage < -thresholds["maximum_completeness_loss"] - epsilon
        or basis < -thresholds["maximum_legal_basis_loss"] - epsilon
        or precision < -thresholds["maximum_precision_loss"] - epsilon
    ):
        return "loss"
    if (
        coverage + epsilon >= thresholds["minimum_completeness_gain"]
        and basis + epsilon >= thresholds["minimum_legal_basis_gain"]
    ):
        return "criterion_met"
    return "below_threshold"


def comparison_results(
    cells: list[dict[str, Any]], thresholds: dict[str, Any] | None, complete: bool
) -> list[dict[str, Any]]:
    if thresholds is None:
        return []
    results = []
    scopes = ["cohort", *sorted({cell["stratum"] for cell in cells})]
    for series in sorted({cell["series"] for cell in cells}):
        for scope in scopes:
            selected = [
                cell
                for cell in cells
                if cell["series"] == series
                and (scope == "cohort" or cell["stratum"] == scope)
            ]
            differences = _paired_differences(selected, series)
            for later, earlier in (("mid", "off"), ("on", "mid")):
                pair = f"{later.upper()}-{earlier.upper()}"
                macro = {
                    item["dimension"]: item["macro_difference"]
                    for item in differences
                    if item["pair"] == pair
                }
                aggregates = [
                    _aggregate([cell for cell in selected if cell["arm"] == arm], scope)
                    for arm in (later, earlier)
                ]
                precision = [
                    item["dimensions"]["precision_of_resolved_findings"]["micro"][
                        "rate"
                    ]
                    for item in aggregates
                ]
                precision_delta = (
                    precision[0] - precision[1]
                    if all(value is not None for value in precision)
                    else None
                )
                minimum_count = min(
                    item["dimensions"][dimension]["micro"]["denominator"]
                    for item in aggregates
                    for dimension in (
                        "precision_of_resolved_findings",
                        "legal_basis_accuracy",
                    )
                )
                coverage = macro.get("completeness_against_key")
                basis = macro.get("legal_basis_accuracy")
                results.append(
                    {
                        "series": series,
                        "scope": scope,
                        "pair": pair,
                        "completeness_macro_delta": coverage,
                        "precision_micro_delta": precision_delta,
                        "legal_basis_macro_delta": basis,
                        "minimum_decided_findings_per_arm": minimum_count,
                        "decision": decide_comparison(
                            coverage,
                            precision_delta,
                            basis,
                            minimum_count,
                            thresholds,
                            complete,
                        ),
                    }
                )
    return results


def attempt_resources(records: list[Record]) -> dict[str, Any]:
    def value(record: Record, field: str) -> Any:
        direct = record.raw.get(field)
        return (
            direct
            if direct is not None
            else (record.raw.get("attempt") or {}).get(field)
        )

    elapsed = [value(record, "elapsed_ms") for record in records]
    incoming = [
        _usage_tokens(value(record, "usage"), "input_tokens") for record in records
    ]
    outgoing = [
        _usage_tokens(value(record, "usage"), "output_tokens") for record in records
    ]
    return {
        "attempts": len(records),
        "elapsed_ms_known": sum(value for value in elapsed if value is not None),
        "input_tokens_known": sum(value for value in incoming if value is not None),
        "output_tokens_known": sum(value for value in outgoing if value is not None),
        "attempts_missing_usage": sum(
            left is None or right is None
            for left, right in zip(incoming, outgoing, strict=True)
        ),
        "attempts_missing_elapsed_ms": sum(value is None for value in elapsed),
        "scope": "All judge attempts, including failures and repeats.",
        "cost": {
            "amount": None,
            "reason": "ChatGPT subscription; no per-call monetary charge is reported.",
        },
    }


def build_report(inputs: Inputs, records: list[Record]) -> dict[str, Any]:
    """Every reported figure, computed from saved judgements and run records."""
    judged, problems, attempts, resources = collect_judged(inputs, records)
    duplicates: dict[tuple[int, str, str], int] = {}
    non_substantive: dict[tuple[int, str, str], dict[str, int]] = {}
    judged_by_cell: dict[tuple[int, str, str], list[Judged]] = {}
    # Seeded from the runs, not from the findings: a run that recorded nothing is
    # a result -- every expectation of that document missed -- and would vanish
    # from the macro average if the cells came from the findings alone.
    for cell in inputs.runs.cells:
        duplicates.setdefault(cell, 0)
        non_substantive.setdefault(cell, {code: 0 for code in NON_SUBSTANTIVE_CODES})
        judged_by_cell.setdefault(cell, [])
    for entry in inputs.accounted:
        cell = entry.finding.cell
        duplicates.setdefault(cell, 0)
        non_substantive.setdefault(cell, {code: 0 for code in NON_SUBSTANTIVE_CODES})
        judged_by_cell.setdefault(cell, [])
        if entry.disposition == "exact_duplicate":
            duplicates[cell] += 1
        elif entry.disposition == "non_substantive":
            non_substantive[cell][entry.finding.code] += 1
        elif entry.finding.finding_id in judged:
            judged_by_cell[cell].append(judged[entry.finding.finding_id])

    cells = [
        _cell_figures(
            inputs,
            cell,
            sorted(
                judged_by_cell[cell], key=lambda item: item.accounted.finding.finding_id
            ),
            duplicates[cell],
            non_substantive[cell],
            resources.get(cell),
        )
        for cell in sorted(judged_by_cell)
    ]

    strata = sorted({cell["stratum"] for cell in cells})
    per_arm: list[dict[str, Any]] = []
    differences: list[dict[str, Any]] = []
    for series in sorted({cell["series"] for cell in cells}):
        series_cells = [cell for cell in cells if cell["series"] == series]
        for arm in sorted({cell["arm"] for cell in series_cells}):
            arm_cells = [cell for cell in series_cells if cell["arm"] == arm]
            entry = _aggregate(arm_cells, "cohort")
            entry["series"] = series
            entry["arm"] = arm
            entry["strata"] = {
                stratum: _aggregate(
                    [cell for cell in arm_cells if cell["stratum"] == stratum], stratum
                )
                for stratum in strata
            }
            per_arm.append(entry)
        differences.extend(_paired_differences(series_cells, series))

    # Both halves are required: every run packet answered, and every finding of
    # every packet answered inside it. Counting packets alone would call a report
    # complete on a reply that skipped findings, and counting findings alone
    # would hide a run nobody judged.
    expected = {
        finding_id for packet in inputs.packets for finding_id in packet.finding_ids
    }
    missing = sorted(expected - set(judged))
    answered_packets = sum(
        1
        for packet in inputs.packets
        if packet.finding_ids and all(name in judged for name in packet.finding_ids)
    )
    # Every packet is covered by this: the expected set is the findings of all of
    # them, so a run nobody answered leaves its findings missing. The count of
    # fully answered packets is reported beside it rather than checked twice.
    complete = not missing and not problems
    return {
        "schema_version": 1,
        "generated_by": "scripts/evaluate.py report",
        "scoring_protocol_version": inputs.protocol["scoring_protocol_version"],
        "scoring_protocol_state": inputs.protocol["state"],
        "answer_key_version": inputs.key.version,
        "answer_key_review_state": inputs.key.review_state,
        "answer_key_sha256": inputs.key.sha256,
        "judge": {
            "model": inputs.config.model,
            "reasoning_effort": inputs.config.effort,
            "instructions_sha256": inputs.config.instructions_sha256,
            "configuration_sha256": inputs.config.configuration_sha256,
            "source_bundle_sha256": inputs.bundle.digest,
            "adapter_identities": sorted(
                {
                    str(record.raw["adapter_sha256"])
                    for record in records
                    if record.raw.get("adapter_sha256")
                }
            ),
        },
        "corpus": {
            "snapshot_id": inputs.bundle.snapshot_id,
            "snapshot_verified_against_the_runs": inputs.bundle.corpus_verified,
            "recorded_by_the_runs": list(inputs.runs.corpus_snapshot_ids),
            "prepared_acts": list(inputs.bundle.acts),
            "declared_acts": list(inputs.bundle.corpus_acts),
            "note": (
                "A claimed provision missing from an unverified corpus is never "
                "counted against an arm; its packet is refused instead."
            ),
        },
        "completeness_of_this_report": {
            "packets": len(inputs.packets),
            "packets_answered_in_full": answered_packets,
            "expected_assessments": len(expected),
            "judged": len(judged),
            "not_yet_judged": len(missing),
            "not_yet_judged_finding_ids": missing,
            "attempts": attempts,
            "problems": sorted(
                problems, key=lambda problem: (problem["problem"], problem["packet_id"])
            ),
            "complete": complete,
            "note": (
                "Technical failures are accounted for here and never reduce a "
                "denominator quietly. A pending judgement is not an analyzer error, "
                "and a failure that a later judgement answered is counted without "
                "blocking the report."
            ),
        },
        "accounting": {
            "recorded_findings": len(inputs.runs.findings),
            "to_be_judged": sum(
                1 for entry in inputs.accounted if entry.disposition == "judged"
            ),
            "exact_duplicates_within_a_run": sum(
                1
                for entry in inputs.accounted
                if entry.disposition == "exact_duplicate"
            ),
            "non_substantive": sum(
                1
                for entry in inputs.accounted
                if entry.disposition == "non_substantive"
            ),
        },
        "cells": cells,
        "finding_outcomes": [
            {
                "finding_id": entry.finding.finding_id,
                "series": entry.finding.series,
                "document_key": entry.finding.document_key,
                "arm": entry.finding.arm,
                "analyzer_code": entry.finding.code,
                "disposition": entry.disposition,
                "duplicate_of": entry.duplicate_of,
                "status": (
                    judged[entry.finding.finding_id].status
                    if entry.finding.finding_id in judged
                    else "pending"
                    if entry.disposition == "judged"
                    else entry.disposition
                ),
            }
            for entry in inputs.accounted
        ],
        "per_arm": per_arm,
        "key_gap_candidates": {
            "finding_ids": sorted(
                finding_id
                for finding_id, item in judged.items()
                if item.status == "correct" and not item.verdict.key_matches
            ),
            "what_this_is": (
                "Findings judged correct and matched to no key item: candidates "
                "for expectations the key may not carry, not reviewed against it."
            ),
            "what_it_moves": (
                "Nothing. The completeness denominator is the frozen key, and "
                "these findings count in precision like any other correct answer."
            ),
        },
        "paired_differences": differences,
        "judge_attempt_resources": attempt_resources(records),
        "comparisons": {
            "decision": (
                ("evaluated" if complete else "incomplete")
                if inputs.protocol.get("comparisons", {}).get("thresholds")
                else "pending_thresholds"
            ),
            "thresholds": inputs.protocol.get("comparisons", {}).get("thresholds"),
            "results": comparison_results(
                cells,
                inputs.protocol.get("comparisons", {}).get("thresholds"),
                complete,
            ),
        },
    }


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------


def command_prepare(data_dir: Path, corpus_manifest: Path) -> int:
    """Prepare the source material once, check it, and write it down.

    Judging never fetches anything: it reads what this wrote or refuses to build
    a packet. The acts are read through the artefact's own seeder, which is the
    one reader whose units the corpus identity below is recomputed from.
    """
    import httpx

    from contract_analyzer.config import RunConfig
    from contract_analyzer.ingest import IngestService
    from seeder.ingest import fetch_manifest_units

    protocol = load_json(data_dir / "scoring-protocol.json")
    config = load_judge_config(protocol, data_dir)
    runs = load_runs(data_dir, protocol)
    manifest = load_json(data_dir / "manifest.json")
    by_case = {
        entry.get("logical_case"): entry
        for entry in manifest["documents"]
        if isinstance(entry, dict)
    }

    documents: dict[str, Any] = {}
    service = IngestService(RunConfig())
    for document_key, record in sorted(runs.documents.items()):
        entry = by_case.get(document_key)
        require(
            isinstance(entry, dict),
            f"manifest.json has no document for {document_key}",
        )
        assert entry is not None
        path = data_dir / entry["path"]
        require(path.is_file(), f"the document file is missing: {path}")
        if entry.get("format") == "txt":
            text = path.read_text(encoding="utf-8")
        else:
            text = service.ingest(path.name, path.read_bytes()).text
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        require(
            digest == record["content_sha256"],
            f"the text prepared for {document_key} hashes to {digest}, not the "
            f"{record['content_sha256']} the runs were measured on",
        )
        documents[document_key] = {
            "artifact_id": record["artifact_id"],
            "content_sha256": digest,
            "chars": len(text),
            "source_path": entry["path"],
            "text": text,
        }

    load_key(
        data_dir / "answer-key.json",
        runs,
        protocol["scored_runs"].get("answer_key_sha256"),
    )
    with httpx.Client(timeout=60.0) as client:
        corpus, manifest_hash, units, _ = fetch_manifest_units(corpus_manifest, client)
    target_date = str(corpus.corpus_target_date)
    corpus_acts = sorted(
        {
            f"{act.publisher}/{act.year}/{act.position}"
            for act in _manifest_acts(corpus_manifest)
        }
    )
    provisions, acts = provision_map(units)
    corpus_record = _corpus_identity(
        units, manifest_hash, target_date, corpus_acts, runs
    )

    config.bundle_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(tz=UTC).isoformat()
    (config.bundle_dir / "documents.json").write_text(
        json.dumps(
            {"schema_version": 1, "built_at": now, "documents": documents},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (config.bundle_dir / "provisions.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "built_at": now,
                "corpus": corpus_record,
                "acts": acts,
                "provisions": provisions,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"prepared {len(documents)} documents and {len(provisions)} provisions "
        f"from {len(acts)} of {len(corpus_acts)} declared acts into "
        f"{config.bundle_dir}"
    )
    if corpus_record["snapshot_verified"]:
        print(
            f"corpus snapshot {corpus_record['snapshot_id']} matches all recorded runs"
        )
    else:
        print(
            f"corpus snapshot {corpus_record['snapshot_id']} does NOT match the "
            f"{', '.join(corpus_record['recorded_by_the_runs'])} the runs recorded. "
            f"Judging will refuse any packet whose claimed provision is missing "
            f"rather than count it against an arm; nothing is scored from an "
            f"unproven corpus."
        )
    return 0


def provision_map(units: list[Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """The prepared provisions, keyed by the identity the corpus minted.

    A repeated article number and an act read as HTML keep the identity they have
    in the corpus, because that is what a claimed locator will be compared with.
    Two units may not share a key: one provision's text would answer for
    another's, under a locator that still parses.
    """
    provisions: dict[str, Any] = {}
    acts: dict[str, Any] = {}
    for unit in units:
        identity = read_locator(unit.locator)
        require(
            not identity.malformed,
            f"the corpus minted a locator this pipeline cannot identify: "
            f"{unit.locator}",
        )
        require(
            identity.key not in provisions,
            f"two corpus units share the identity {identity.key}: {unit.locator}. "
            f"Preparation refuses rather than let one provision's text stand in "
            f"for another's.",
        )
        acts.setdefault(
            unit.act_identifier,
            {"legal_status_date": str(unit.legal_status_date), "articles": 0},
        )
        acts[unit.act_identifier]["articles"] += 1
        provisions[identity.key] = {
            "locator": unit.locator,
            "act_identifier": unit.act_identifier,
            "article_identifier": unit.article_identifier,
            "content_sha256": unit.content_hash,
            "text": unit.text,
        }
    return provisions, acts


def _corpus_identity(
    units: list[Any],
    manifest_hash: str,
    target_date: str,
    declared_acts: list[str],
    runs: RecordedRuns,
) -> dict[str, Any]:
    """Recompute the corpus identity and compare it with what the runs recorded.

    The seeder names a corpus by its manifest and the digest of its units, and
    the runs recorded that name. Recomputing it here from the prepared units is
    what turns "this is the frozen corpus" from a claim into a check: only a
    match licenses reading a provision's absence as a fact about the law rather
    than about the preparation.
    """
    from contract_analyzer.corpus import (  # noqa: PLC0415
        corpus_snapshot_id,
        unit_digest_of,
    )

    digest = unit_digest_of((unit.locator, unit.content_hash) for unit in units)
    snapshot_id = corpus_snapshot_id(manifest_hash, digest)
    recorded = list(runs.corpus_snapshot_ids)
    return {
        "manifest_sha256": manifest_hash,
        "target_date": target_date,
        "declared_acts": declared_acts,
        "unit_digest": digest,
        "unit_count": len(units),
        "snapshot_id": snapshot_id,
        "recorded_by_the_runs": recorded,
        "snapshot_verified": recorded == [snapshot_id],
        "identity_inputs": (
            "The snapshot id is the manifest hash, the unit digest and the "
            "embedding space specification, as the seeder computes it. A "
            "mismatch is reported, never worked around."
        ),
    }


def _manifest_acts(corpus_manifest: Path) -> list[Any]:
    from contract_analyzer.corpus import load_manifest  # noqa: PLC0415

    return list(load_manifest(corpus_manifest).acts)


def command_packets(data_dir: Path, output: Path | None) -> int:
    """Build every packet offline and say what would be judged."""
    inputs = load_inputs(data_dir)
    print(
        f"{len(inputs.runs.findings)} recorded findings: "
        f"{sum(1 for e in inputs.accounted if e.disposition == 'judged')} to judge "
        f"in {len(inputs.packets)} run packet(s), "
        f"{sum(1 for e in inputs.accounted if e.disposition == 'exact_duplicate')} "
        f"exact duplicates within a run, "
        f"{sum(1 for e in inputs.accounted if e.disposition == 'non_substantive')} "
        f"non-substantive"
    )
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                [
                    {
                        "packet_id": packet.packet_id,
                        "series": packet.series,
                        "document_key": packet.document_key,
                        "arm": packet.arm,
                        "finding_ids": list(packet.finding_ids),
                        "findings": len(packet.finding_ids),
                        "chars": len(packet.case),
                    }
                    for packet in inputs.packets
                ],
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"wrote the packet map to {output}; it is never part of a prompt")
    return 0


def load_adapter_module() -> Any:
    """The judging adapter, imported only when a model is actually to be called.

    One seam for the whole module: imported here rather than at the top so every
    offline command, and every test of the counting, runs without the SDK
    installed.
    """
    import judge_client  # noqa: PLC0415

    return judge_client


def adapter_identity() -> dict[str, Any]:
    """The fixed instrument the adapter is, as a value a record can carry.

    Every one of these is a constant of the adapter, and a judgement made under a
    different one was made by a different instrument. Recording them is what lets
    a resumed run tell that it is continuing the same measurement, and what stops
    a report mixing two.
    """
    adapter = load_adapter_module()
    return {
        "model": adapter.JUDGE_MODEL,
        "reasoning_effort": adapter.JUDGE_EFFORT.value,
        "sdk_version": adapter.SDK_VERSION,
        "base_instructions_sha256": sha256_value(adapter.BASE_INSTRUCTIONS),
        "runtime_overrides": list(adapter.RUNTIME_OVERRIDES),
        "permission_profile": adapter.PROFILE_NAME,
        "allowed_transcript_items": sorted(adapter.ALLOWED_ITEM_TYPES),
    }


def check_configured_judge(config: JudgeConfig) -> dict[str, Any]:
    """Refuse to judge if the protocol and the adapter name different instruments.

    The protocol says which model and which reasoning effort this evaluation is
    defined over; the adapter fixes them in code. If the two ever disagree, the
    figures would be attributed to a configuration that never produced them.
    """
    identity = adapter_identity()
    require(
        identity["model"] == config.model
        and identity["reasoning_effort"] == config.effort,
        f"the scoring protocol judges with {config.model} at {config.effort} and "
        f"the adapter is fixed to {identity['model']} at "
        f"{identity['reasoning_effort']}",
    )
    return identity


@contextmanager
def judge_runtime(credentials_home: Path) -> Iterator[tuple[Path, Path]]:
    """A disposable home and an empty workspace, outside this artefact.

    The adapter requires a home that carries nothing but ``auth.json``, and a
    disjoint empty workspace. Both are made in the system temporary directory
    and removed afterwards, so a rerun always starts from the same clean state
    and no credential is ever written inside the artefact. Only ``auth.json`` is
    copied, mode 0600, from the sign-in the operator names: authentication stays
    the runtime's, and nothing here reads or reimplements it.
    """
    source = (credentials_home.expanduser() / "auth.json").resolve()
    if not source.is_file():
        raise EvaluationError(
            f"no auth.json under {credentials_home}. Sign the Codex runtime in to "
            f"the ChatGPT account first and pass its home with --credentials-home."
        )
    root = Path(tempfile.mkdtemp(prefix="contract-analyzer-judge-"))
    try:
        home = root / "home"
        workspace = root / "workspace"
        home.mkdir(mode=0o700)
        workspace.mkdir(mode=0o700)
        target = home / "auth.json"
        target.write_bytes(source.read_bytes())
        target.chmod(0o600)
        yield home, workspace
    finally:
        shutil.rmtree(root)


def _reply_evidence(reply: Any) -> dict[str, Any]:
    """Everything the adapter returned, in a form a JSON line can hold."""
    return {
        "turn_status": reply.status,
        "requested_model": reply.requested_model,
        "requested_effort": reply.requested_effort,
        "sdk_version": reply.sdk_version,
        "runtime": reply.runtime,
        "thread_id": reply.thread_id,
        "turn_id": reply.turn_id,
        "usage": reply.usage,
        "started_at": reply.started_at,
        "completed_at": reply.completed_at,
        "reported_duration_ms": reply.reported_duration_ms,
        "elapsed_ms": reply.elapsed_ms,
        "raw_response": reply.raw_response,
        "transcript_items": reply.items,
        "raw_result": reply.result.model_dump(mode="json"),
    }


def _failure_evidence(exc: BaseException) -> dict[str, Any]:
    """Serialize the available evidence of a failed adapter attempt."""
    evidence: dict[str, Any] = {
        "error_type": type(exc).__name__,
        "error": str(exc),
    }
    attached = dict(getattr(exc, "evidence", None) or {})
    attached.update(
        {
            name: getattr(exc, name)
            for name in (
                "raw_response",
                "usage",
                "thread_id",
                "turn_id",
                "items",
                "elapsed_ms",
            )
            if getattr(exc, name, None) is not None
        }
    )
    if attached:
        evidence["attempt"] = json.loads(json.dumps(attached, default=str))
    else:
        evidence["attempt"] = None
        evidence["attempt_note"] = (
            "The adapter reported this failure as a message only, so no reply "
            "body, usage or turn identity is available for it."
        )
    return evidence


def judge_packets(
    packets: list[Packet],
    inputs: Inputs,
    path: Path,
    credentials_home: Path,
    stage: str,
    workers: int = 1,
) -> int:
    """Judge each run packet, writing its record the moment the reply is made.

    One packet is one fresh judge conversation about one run, and nothing is
    carried from one to the next. ``workers`` conversations may be open at once,
    each with its own runtime home and session, because packets are independent
    by construction: no packet's answer is an input to another's.

    A failure stops the run: the worker that hit it records it, the others finish
    the packet in their hands and none takes a new one. Judging is resumed by
    running again, which judges what has no reusable record.
    """
    require(
        1 <= workers <= MAX_WORKERS,
        f"between 1 and {MAX_WORKERS} judging workers, not {workers}",
    )
    adapter = load_adapter_module()
    config = inputs.config
    identity = check_configured_judge(config)
    common = {
        "configuration_sha256": config.configuration_sha256,
        "source_bundle_sha256": inputs.bundle.digest,
        "answer_key_version": inputs.key.version,
        "answer_key_sha256": inputs.key.sha256,
        "adapter_sha256": sha256_value(identity),
        "adapter": identity,
        "corpus_snapshot_id": inputs.bundle.snapshot_id,
        "stage": stage,
    }
    queue = deque(packets)
    taking = Lock()
    writing = Lock()
    stop = Event()
    judged = 0

    def next_packet() -> Packet | None:
        with taking:
            if stop.is_set() or not queue:
                return None
            return queue.popleft()

    def write(record: dict[str, Any]) -> None:
        with writing:
            append_record(path, record)

    def work() -> int:
        done = 0
        packet = next_packet()
        if packet is None:
            return 0
        try:
            with judge_runtime(credentials_home) as (home, workspace):
                try:
                    with adapter.JudgeSession(
                        home=home, workspace=workspace
                    ) as session:
                        while packet is not None:
                            _judge_one(packet, session, adapter, config, common, write)
                            done += 1
                            packet = next_packet()
                except BaseException:
                    stop.set()
                    raise
        except BaseException:
            stop.set()
            raise
        return done

    with ThreadPoolExecutor(max_workers=workers) as pool:
        running = [pool.submit(work) for _ in range(workers)]
        failures = []
        for future in running:
            try:
                judged += future.result()
            except EvaluationError as exc:
                failures.append(exc)
    if failures:
        raise failures[0]
    return judged


def _judge_one(
    packet: Packet,
    session: Any,
    adapter: Any,
    config: JudgeConfig,
    common: dict[str, Any],
    write: Any,
) -> None:
    """One conversation about one run: ask, check the reply, write it down."""
    request = adapter.JudgePacket(
        packet_id=packet.packet_id,
        instructions=config.instructions,
        case=packet.case,
        response_model=JudgeReply,
    )
    record: dict[str, Any] = {
        **common,
        "packet_id": packet.packet_id,
        "packet_sha256": sha256_value(packet.payload),
        "series": packet.series,
        "document_key": packet.document_key,
        "arm": packet.arm,
        "finding_ids": list(packet.finding_ids),
        "judged_at": datetime.now(tz=UTC).isoformat(),
    }
    try:
        reply = session.judge(request)
    except Exception as exc:
        record.update(_failure_evidence(exc))
        record["status"] = _failure_status(exc, adapter)
        write(record)
        raise EvaluationError(
            f"judging stopped at packet {packet.packet_id}: {exc}. The attempt is "
            f"saved as a failure of the instrument, not as a finding about the "
            f"analyzer; rerun to continue with the same model and settings."
        ) from exc
    record.update(_reply_evidence(reply))
    try:
        answers = validate_reply(record["raw_result"], packet)
    except EvaluationError as exc:
        record.update({"status": "invalid_output", "error": str(exc)})
        write(record)
        raise EvaluationError(
            f"the judge answered packet {packet.packet_id} with a reply the packet "
            f"does not admit: {exc}. The reply is saved as evidence and is not "
            f"retried automatically."
        ) from exc
    record.update(
        {
            "status": "validated",
            "assessments": [
                answers[finding_id].model_dump(mode="json")
                for finding_id in packet.finding_ids
            ],
        }
    )
    write(record)


def _failure_status(exc: BaseException, adapter: Any) -> str:
    """Name the kind of instrument failure, so no two are reported as one.

    The adapter's own classes decide, not the message: a reply that did not
    validate, a capability that reached the model and a runtime that could not
    answer are three different events, and only the last invites a rerun.
    """
    if isinstance(exc, getattr(adapter, "JudgeReplyError", ())):
        return "invalid_output"
    if isinstance(exc, getattr(adapter, "JudgeIsolationError", ())):
        return "isolation_failure"
    return "judge_unavailable"


def _pending(packets: tuple[Packet, ...], path: Path, inputs: Inputs) -> list[Packet]:
    """What is left to judge: everything without a reusable judgement already.

    The adapter is loaded here, so its identity is part of the question, and a
    store already holding judgements of this cohort by another instrument stops
    the run instead of being judged again. Judging it again would pay for a
    second set of calls the report may not read beside the first.
    """
    adapter_sha256 = sha256_value(check_configured_judge(inputs.config))
    by_id = {packet.packet_id: packet for packet in packets}
    records = read_records(path)
    other = sorted(
        {
            str(record.raw.get("adapter_sha256") or "unknown")
            for record in records
            if record.packet_id in by_id and record.status == "validated"
        }
        - {adapter_sha256}
    )
    # The reader refuses judgements made by two instruments, so judging again
    # would pay for calls no report could read beside the saved ones.
    if other:
        raise EvaluationError(
            f"{path} already holds judgements made by another instrument "
            f"({', '.join(identity[:12] for identity in other)}), and this run "
            f"would add {adapter_sha256[:12]}. A report never mixes two, so the "
            f"saved judgements have to be set aside, or this run made with the "
            f"instrument that produced them."
        )
    done = set()
    for record in records:
        packet = by_id.get(record.packet_id)
        if packet is None or not record.reusable(
            packet,
            inputs.config,
            inputs.bundle.digest,
            inputs.key.sha256,
            adapter_sha256,
        ):
            continue
        try:
            validate_reply({"assessments": record.raw.get("assessments")}, packet)
        except EvaluationError:
            continue
        done.add(record.packet_id)
    return [packet for packet in packets if packet.packet_id not in done]


def command_run(
    data_dir: Path, limit: int | None, credentials_home: Path, workers: int = 1
) -> int:
    """Judge the pending packets of the declared cohort, saving as it goes."""
    inputs = load_inputs(data_dir)
    if not inputs.config.ready_for_full_run:
        raise EvaluationError(
            "the scoring protocol does not declare the procedure ready for the full "
            f"run: {inputs.config.readiness_note} Calibration runs on its own cases "
            "through `evaluate.py calibrate`; the cohort is never judged to try the "
            "instrument out."
        )
    path = inputs.config.verdicts_path
    pending = _pending(inputs.packets, path, inputs)
    if limit is not None:
        pending = pending[:limit]
    if not pending:
        print(f"nothing pending: every packet already has a judgement in {path}")
        return 0
    judged = judge_packets(pending, inputs, path, credentials_home, "cohort", workers)
    print(f"judged {judged} packet(s) into {path}")
    return 0


def command_calibrate(
    data_dir: Path,
    cases: Path,
    limit: int | None,
    credentials_home: Path,
    workers: int = 1,
) -> int:
    """Judge the author's calibration cases, which no reported figure reads.

    The cases are supplied whole and must stand outside the judged cohort. This
    is the only way to try the instrument: no packet of the reported comparison
    is ever sent to settle whether the judge works.
    """
    protocol = load_json(data_dir / "scoring-protocol.json")
    config = load_judge_config(protocol, data_dir)
    runs = load_runs(data_dir, protocol)
    key = load_key(
        data_dir / "answer-key.json",
        runs,
        protocol["scored_runs"].get("answer_key_sha256"),
    )
    packets = load_calibration_packets(cases, runs, config.max_packet_chars)
    inputs = Inputs(
        protocol=protocol,
        config=config,
        runs=runs,
        key=key,
        accounted=(),
        bundle=_calibration_bundle(),
        packets=packets,
    )
    path = config.calibration_verdicts_path
    pending = _pending(packets, path, inputs)
    if limit is not None:
        pending = pending[:limit]
    if not pending:
        print(f"nothing pending: every calibration case has a judgement in {path}")
        return 0
    judged = judge_packets(
        pending, inputs, path, credentials_home, "calibration", workers
    )
    print(
        f"judged {judged} calibration case(s) into {path}. No reported figure reads "
        f"this file."
    )
    return 0


def _calibration_bundle() -> SourceBundle:
    """A calibration case carries its own sources, so there is no bundle.

    The recorded digest still has to say which material the judgement was made
    against; for these cases it is the case file itself, and naming it as such
    keeps a calibration record from ever matching a cohort configuration.
    """
    return SourceBundle(
        documents={},
        provisions={},
        acts=(),
        corpus_acts=(),
        corpus={"snapshot_verified": False, "snapshot_id": ""},
        digest="calibration-cases-carry-their-own-sources",
        built_at="",
    )


def command_report(data_dir: Path, output: Path | None) -> int:
    """Recompute every figure from the saved judgements. No model call."""
    inputs = load_inputs(data_dir)
    records = read_records(inputs.config.verdicts_path)
    report = build_report(inputs, records)
    destination = output or inputs.config.report_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    state = report["completeness_of_this_report"]
    print(
        f"wrote {destination}: {state['packets_answered_in_full']} of "
        f"{state['packets']} run packet(s) answered in full, {state['judged']} of "
        f"{state['expected_assessments']} finding(s) judged"
        + ("" if state["complete"] else ", report incomplete")
    )
    return 0 if state["complete"] else 1


def command_schema(output: Path | None) -> int:
    """Write out the response schema the judge is held to."""
    text = json.dumps(response_schema(), ensure_ascii=False, indent=2, sort_keys=True)
    if output is None:
        print(text)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
        print(f"wrote {output}")
    return 0


def _add_workers_argument(parser: argparse.ArgumentParser) -> None:
    """How many run packets may be judged at once, each in its own runtime."""
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        choices=range(1, MAX_WORKERS + 1),
        help=(
            "how many judging conversations to keep open at once; each gets its "
            "own disposable runtime home and session, and one packet is never "
            "split across them"
        ),
    )


def _add_credentials_argument(parser: argparse.ArgumentParser) -> None:
    """Where the operator's Codex sign-in lives. Only auth.json is ever copied."""
    parser.add_argument(
        "--credentials-home",
        type=Path,
        default=Path.home() / ".codex",
        help=(
            "a Codex home already signed in to the ChatGPT account; its auth.json "
            "is copied into a temporary home for the run and deleted afterwards"
        ),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="the evaluation-data directory to read and write",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="fetch and check the source material")
    prepare.add_argument(
        "--corpus-manifest",
        type=Path,
        default=ROOT / "corpus" / "manifest.example.json",
        help="the corpus manifest whose acts the judging packets quote",
    )

    packets = commands.add_parser("packets", help="build the packets offline")
    packets.add_argument("--write-map", type=Path, default=None)

    run = commands.add_parser("run", help="judge the pending packets")
    run.add_argument("--limit", type=int, default=None)
    _add_workers_argument(run)
    _add_credentials_argument(run)

    calibrate = commands.add_parser(
        "calibrate", help="judge the supplied calibration cases, never the cohort"
    )
    calibrate.add_argument(
        "--cases",
        type=Path,
        required=True,
        help="the calibration case file; its cases must stand outside the cohort",
    )
    calibrate.add_argument("--limit", type=int, default=None)
    _add_workers_argument(calibrate)
    _add_credentials_argument(calibrate)

    report = commands.add_parser("report", help="recompute the figures offline")
    report.add_argument("--offline", action="store_true", help="accepted and implied")
    report.add_argument("--output", type=Path, default=None)

    schema = commands.add_parser("schema", help="print the judge response schema")
    schema.add_argument("--output", type=Path, default=None)

    arguments = parser.parse_args(argv)
    data_dir = arguments.data_dir
    if arguments.command == "prepare":
        return command_prepare(data_dir, arguments.corpus_manifest)
    if arguments.command == "packets":
        return command_packets(data_dir, arguments.write_map)
    if arguments.command == "run":
        return command_run(
            data_dir, arguments.limit, arguments.credentials_home, arguments.workers
        )
    if arguments.command == "calibrate":
        return command_calibrate(
            data_dir,
            arguments.cases,
            arguments.limit,
            arguments.credentials_home,
            arguments.workers,
        )
    if arguments.command == "report":
        return command_report(data_dir, arguments.output)
    return command_schema(arguments.output)


def cli(argv: list[str] | None = None) -> int:
    try:
        return main(argv)
    except (EvaluationError, EvaluationInputError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(cli())
