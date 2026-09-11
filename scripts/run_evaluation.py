#!/usr/bin/env python3
"""Drive and record evaluation runs through the contract-analyzer REST API.

Each invocation uploads selected artifacts from the evaluation manifest, runs
each requested arm, polls until terminal, and saves raw GET /runs/{id} JSON
plus a summary.  Never stores or prints document text.

Usage
-----
    uv run python scripts/run_evaluation.py \\
        --split development --out /tmp/out
    uv run python scripts/run_evaluation.py \\
        --split holdout --i-confirm-holdout --out /tmp/out

See --help for all options.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "evaluation-data" / "manifest.json"
PROTOCOL = ROOT / "evaluation-data" / "protocol.json"

_TERMINAL = {"completed", "failed", "cancelled"}
_ARMS_DEFAULT = ["off", "mid", "on"]


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _sha256_file(path: Path) -> str:
    with path.open("rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


IDENTITY_FILE = ".batch_identity.json"
# What every run in one results directory must share. The protocol, manifest,
# split and served configuration are pinned so that mid-batch drift is stopped.
#
# Arms, cases and variants are deliberately not pinned. Running the remaining
# arms of a paused batch is the resume this exists to support, not the drift it
# exists to stop; what that used to break was the summary, which now reports
# every result in the directory rather than the ones this invocation asked for.
_IDENTITY_FIELDS = (
    "protocol_version",
    "protocol_sha256",
    "manifest_sha256",
    "split",
    "implementation_fingerprint",
    "config_fingerprint",
)
# The served facts a comparison rests on. A change to any of them mid-batch
# makes the arms incomparable, which is the thing the identity is protecting.
_CONFIG_PARITY_FIELDS = (
    "model_request_id",
    # The identifier does not name the endpoint that served it, and two
    # endpoints serving one identifier do not serve it identically.
    "model_endpoint",
    "prompt_bundle_version",
    "corpus_snapshot",
    "embedding_model",
    # The identifier names the model; the declared space -- width and upstream with
    # it -- is what the vectors are, and the two have already disagreed.
    "embedding_space_fingerprint",
    # A fallback corpus cuts its acts into different units, so a batch must not
    # cross one. Listing it here also writes the measured corpus's source format
    # into the batch identity.
    "corpus_source_format",
    "tool_bundle_version",
    "measured_mode",
)


def _implementation_fingerprint(root: Path | None = None) -> str:
    """Deterministic digest of the measured implementation and runtime inputs."""
    base = root or ROOT
    hasher = hashlib.sha256()
    target_dirs = [
        base / "backend" / "src",
        base / "seeder" / "src",
        base / "scripts",
    ]
    target_files = [
        base / "pyproject.toml",
        base / "uv.lock",
        base / "compose.yaml",
        base / "openapi.yaml",
        base / "backend" / "Dockerfile",
        base / "seeder" / "Dockerfile",
        base / "web" / "Dockerfile",
    ]
    file_paths: list[Path] = []
    for d in target_dirs:
        if d.exists():
            file_paths.extend(p for p in d.rglob("*.py") if p.is_file())
    for f in target_files:
        if f.exists():
            file_paths.append(f)
    for p in sorted(file_paths):
        rel = p.relative_to(base).as_posix()
        hasher.update(rel.encode("utf-8"))
        with p.open("rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                hasher.update(chunk)
    return hasher.hexdigest()


def _config_fingerprint(config_data: dict[str, Any]) -> str:
    material = json.dumps(
        {field: config_data.get(field) for field in _CONFIG_PARITY_FIELDS},
        sort_keys=True,
    )
    return hashlib.sha256(material.encode()).hexdigest()


def _current_identity(
    args: argparse.Namespace,
    proto_data: dict[str, Any],
    config_data: dict[str, Any],
) -> dict[str, Any]:
    return {
        "started_at": datetime.now(UTC).isoformat(),
        "protocol_version": proto_data.get("protocol_version"),
        "protocol_sha256": _sha256_file(PROTOCOL) if PROTOCOL.exists() else None,
        "manifest_sha256": _sha256_file(MANIFEST) if MANIFEST.exists() else None,
        "split": args.split,
        "implementation_fingerprint": _implementation_fingerprint(),
        "config_fingerprint": _config_fingerprint(config_data),
        "config": {field: config_data.get(field) for field in _CONFIG_PARITY_FIELDS},
    }


def _requested(
    recorded: dict[str, Any], arms: list[str], args: argparse.Namespace
) -> dict[str, Any]:
    """What this directory has been asked for, across every invocation."""
    previous: dict[str, Any] = recorded.get("requested", {})
    cases = previous.get("cases") or []
    if args.cases:
        cases = sorted(
            {*cases, *(c.strip() for c in args.cases.split(",") if c.strip())}
        )
    return {
        "arms": sorted({*previous.get("arms", []), *arms}),
        "cases": cases,
        "include_variants": bool(previous.get("include_variants"))
        or bool(args.include_variants),
        "invocations": int(previous.get("invocations", 0)) + 1,
    }


def _batch_identity(
    out_root: Path,
    args: argparse.Namespace,
    proto_data: dict[str, Any],
    config_data: dict[str, Any],
    arms: list[str],
) -> tuple[dict[str, Any], str | None]:
    """The identity this batch started under, refusing a resume that drifted.

    A results directory is a batch, not a folder. Skipping the runs that already
    exist and executing the rest under changed code, a different split or a
    different server would produce one summary describing results that never
    shared a state. The check runs before the first new run of an invocation, so
    a refusal costs nothing.

    A directory's first call names its batch. That id is what the registered final
    receipt binds to, so a fresh ``--out`` is a new batch rather than a resume of
    one whose results live somewhere else.
    """
    path = out_root / IDENTITY_FILE
    current = _current_identity(args, proto_data, config_data)
    if not path.exists():
        current["batch_id"] = uuid4().hex
        current["requested"] = _requested({}, arms, args)
        path.write_text(json.dumps(current, indent=2), encoding="utf-8")
        return current, None

    try:
        recorded: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        # An unreadable identity is not a resumable batch: refuse through the
        # same path as drift rather than dying on a traceback.
        return {}, (
            f"ERROR: {path} cannot be read, so this batch cannot prove what it "
            f"started under and cannot be resumed ({exc})."
        )

    drifted = [
        field for field in _IDENTITY_FIELDS if recorded.get(field) != current.get(field)
    ]
    if drifted:
        lines = "\n".join(
            f"  {field}: batch started under {recorded.get(field)!r}, "
            f"now {current.get(field)!r}"
            for field in drifted
        )
        return recorded, (
            f"ERROR: this batch was started under a different state and cannot be "
            f"resumed.\n{lines}\nStart a new batch in another --out directory, or "
            f"return to the state named in {path}."
        )

    recorded["requested"] = _requested(recorded, arms, args)
    recorded.setdefault("batch_id", uuid4().hex)
    path.write_text(json.dumps(recorded, indent=2), encoding="utf-8")
    _log(f"NOTE: resuming the batch started at {recorded.get('started_at')}")
    return recorded, None


def _load_manifest_data() -> dict[str, Any]:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))  # type: ignore[no-any-return]


def _load_protocol() -> dict[str, Any]:
    return json.loads(PROTOCOL.read_text(encoding="utf-8"))  # type: ignore[no-any-return]


def _load_holdout_reminder() -> str:
    data = _load_protocol()
    items: list[str] = data.get("holdout_discipline", {}).get(
        "forbidden_until_final_run", []
    )
    return "\n  - ".join(["Forbidden until final run:"] + items)


# The registered final batch may happen once. A protocol field alone cannot say
# so: nothing moves it, so it reads not_started after a completed batch and a
# second invocation with a fresh --out passes the same check.
#
# The receipt at evaluation-data/results/final-registered-measurement.json is
# written immediately before the first measured request, reserving the batch
# and serving as verifiable evidence.
FINAL_RECEIPT = "final-registered-measurement.json"


def _exposed_cases(manifest: list[dict[str, Any]]) -> dict[str, str]:
    """The logical cases an exposure record has reached, and which artifact did it.

    Exposure is recorded on the artifact it happened to, and the comparison is
    over logical cases: a format variant is the same document read another way,
    never an independent observation. Reading a variant's output is therefore
    reading the case, and filtering per artifact would leave the canonical
    artifact of a contaminated case standing in the clean cohort.
    """
    exposed: dict[str, str] = {}
    for doc in manifest:
        if doc.get("holdout_exposure"):
            exposed.setdefault(doc.get("logical_case", doc["id"]), doc["id"])
    return exposed


def _registered_cohort(manifest: list[dict[str, Any]]) -> list[str]:
    """The whole clean holdout: the cohort the registered comparison is over.

    --cases says what one invocation works through, not what the experiment was.
    Claiming the batch over the selection would let a first invocation naming a
    single case register a single-case final comparison, after which the rest of
    the stratum carries a different cohort and is refused as a second batch.
    """
    exposed = _exposed_cases(manifest)
    return sorted(
        {
            case
            for doc in manifest
            if doc.get("split") == "holdout"
            and (case := doc.get("logical_case", doc["id"])) not in exposed
        }
    )


def _series_slot(proto_data: dict[str, Any], out: Path) -> dict[str, Any] | None:
    """The registered series this output directory is, where one is registered.

    A repeat of the registered measurement is a series the protocol authorized
    before it ran, into a directory it names. Any other directory is the second
    batch the single-batch rule exists to refuse, and is left to it.
    """
    target = out.resolve()
    for slot in proto_data.get("registered_repeat_series", {}).get("series", []):
        recorded = str(slot.get("results") or "")
        # A series the protocol records as finished is not an authorized slot:
        # its directory would otherwise walk past the refusal that the single
        # registered batch exists to make.
        if (
            recorded
            and slot.get("state") == "authorized"
            and (ROOT / recorded).resolve() == target
        ):
            return slot
    return None


def _series_parity_error(
    config_data: dict[str, Any], repeat: dict[str, Any]
) -> str | None:
    """Whether the server serves the configuration series 1 was measured under.

    A repeat that runs against another model, corpus, prompt bundle or provider
    measures something else, and the batch identity would record the difference
    only after the runs had been paid for. Fields series 1 recorded are compared
    to what it recorded; fields added since are compared to the value the
    protocol requires of a repeat.
    """
    expected: dict[str, Any] = repeat.get("series_1_config") or {}
    required: dict[str, Any] = repeat.get("required_values") or {}
    unchecked = [
        field
        for field in _CONFIG_PARITY_FIELDS
        if field not in expected and field not in required
    ]
    if unchecked:
        return (
            f"the protocol's repeat-series record does not cover "
            f"{', '.join(unchecked)}, which parity depends on; a parity field "
            f"nothing compares is a field that can drift silently."
        )
    problems = [
        f"  {field}: series 1 ran under {expected[field]!r}, this server "
        f"reports {config_data.get(field)!r}"
        for field in _CONFIG_PARITY_FIELDS
        if field in expected and config_data.get(field) != expected[field]
    ]
    problems += [
        f"  {field}: a repeat series requires {value!r}, this server reports "
        f"{config_data.get(field)!r}"
        for field, value in sorted(required.items())
        if config_data.get(field) != value
    ]
    if not problems:
        return None
    return (
        "the served configuration is not the one series 1 was measured under, "
        "so these runs would not repeat it.\n" + "\n".join(problems)
    )


def _recorded_location(out_dir: Path) -> str:
    """Where a batch ran, said in a way the submitted artefact can carry.

    A batch that belongs to this artefact is recorded relative to its root. An
    absolute path names the machine it ran on, which the submitted copy has no
    use for and should not carry, and it makes a receipt read differently in
    every checkout of the same tree. A directory outside the artefact has no
    relative form and is recorded as it is; a registered batch does not have
    one, because the protocol names its directory inside evaluation-data.

    What the value has to do is tell one directory from another, and a
    root-relative path still does within an artefact. A copy of the whole
    artefact resuming its own batch at the same relative place is the same
    batch; a directory copied to another name inside the artefact is not, and
    still differs here.
    """
    resolved = out_dir.resolve()
    root = ROOT.resolve()
    if resolved == root or root in resolved.parents:
        return resolved.relative_to(root).as_posix()
    return str(resolved)


def _read_claim(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"{path} is held by a batch whose record cannot be read ({exc})."


def _hold(
    path: Path, payload: dict[str, Any]
) -> tuple[dict[str, Any] | None, str | None]:
    """Take the claim at path, or read back whoever won it.

    Written whole and then linked into place: an exclusive create followed by a
    write leaves a truncated file if the runner dies between the two, and a claim
    that cannot be parsed is a batch that can never be resumed. The link is the
    exclusive step, so a second runner still loses rather than overwrites.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f"{path.name}.{os.getpid()}.partial")
    try:
        staging.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        try:
            os.link(staging, path)
        except FileExistsError:
            return _read_claim(path)
    finally:
        staging.unlink(missing_ok=True)
    return None, None


def _claim_final_batch(
    identity: dict[str, Any],
    cohort: list[str],
    out_dir: Path,
    slot: dict[str, Any] | None = None,
) -> str | None:
    """Reserve the one registered batch, or resume it, or refuse a second one.

    Called after every read-only precondition has passed and immediately before
    the first measured request, so a dead server or a failed preflight cannot
    consume the batch. The batch is the output directory: .batch_identity.json
    mints an id when the directory is created, and every invocation continuing
    that directory presents the same id. A fresh --out is a fresh id, and a fresh
    id over the same cohort is a second set of model runs, not a resume.

    A registered repeat series takes a claim file of its own, named by the
    protocol slot it runs as. The reservation is therefore per registered
    series rather than per project: the first series' receipt is left holding
    its own directory, and an unregistered directory still meets the one claim
    that refuses it.
    """
    batch_id = identity.get("batch_id")
    if not batch_id:
        return (
            f"this batch has no identity of its own, so the registered claim "
            f"cannot name it; {IDENTITY_FILE} in the output directory carries "
            f"batch_id."
        )
    here = _recorded_location(out_dir)
    payload = {
        "started_at": datetime.now(UTC).isoformat(),
        "batch_id": batch_id,
        "registered_cohort": cohort,
        "out_dir": here,
        "identity": {field: identity.get(field) for field in _IDENTITY_FIELDS},
        "note": (
            "Written immediately before the first measured request of the "
            "registered final batch. registered_cohort is the whole clean "
            "holdout the comparison is over; one invocation may work through "
            "part of it, and out_dir/.batch_identity.json records what the "
            "batch has been asked for so far. An invocation continuing that "
            "directory resumes this batch; any other is refused. Delete this "
            "only alongside a protocol version that records why."
        ),
    }
    if slot is not None:
        payload["registered_series"] = slot.get("series")
        payload["authorized_by"] = (
            "protocol registered_repeat_series; this series has a claim of its "
            "own and does not release any other series' claim"
        )
    receipt = str((slot or {}).get("receipt") or FINAL_RECEIPT)
    if Path(receipt).name != receipt:
        return (
            f"the registered series names {receipt!r} as its claim, which is not "
            f"a file name in evaluation-data/results/."
        )
    claim_path = ROOT / "evaluation-data" / "results" / receipt

    def is_this_batch(holder: dict[str, Any]) -> bool:
        return holder.get("batch_id") == batch_id and holder.get("out_dir") == here

    def refusal(holder: dict[str, Any], path: Path) -> str:
        if holder.get("batch_id") == batch_id:
            return (
                f"this batch was claimed at {holder.get('out_dir')}, and this "
                f"invocation is running it at {here}; {path} names the first. "
                f"A copied results directory would continue the one registered "
                f"batch in two places. If the directory was moved rather than "
                f"copied, say so in the receipt alongside a protocol version."
            )
        return (
            f"a different registered final batch was claimed at "
            f"{holder.get('started_at', 'an unrecorded time')} over "
            f"{holder.get('registered_cohort', 'an unrecorded cohort')} into "
            f"{holder.get('out_dir', 'an unrecorded directory')}; it holds {path}. "
            f"Running a second batch after the first has been seen is how a "
            f"measurement becomes a choice, so it needs a new protocol version "
            f"saying what the earlier result was and why."
        )

    if claim_path.exists():
        holder, problem = _read_claim(claim_path)
        if problem is not None or holder is None:
            return problem
        if not is_this_batch(holder):
            return refusal(holder, claim_path)
        _log(f"NOTE: resuming the registered batch claimed at {holder['started_at']}")
        return None

    holder, problem = _hold(claim_path, payload)
    if problem is not None:
        return problem
    if holder is not None and not is_this_batch(holder):
        return refusal(holder, claim_path)
    return None


def _select_artifacts(
    manifest: list[dict[str, Any]],
    split: str,
    cases: list[str] | None,
    include_variants: bool,
) -> list[dict[str, Any]]:
    """The artifacts this split covers, minus any holdout case already exposed.

    An exposed case is no longer clean confirmatory material whatever its split
    still says. The split itself is left alone -- re-splitting after exposure is
    the author's decision -- but the final run must not pick the case up
    meanwhile, and an obligation written in prose is not a barrier.
    """
    selected = []
    exposed = _exposed_cases(manifest)
    for doc in manifest:
        if doc.get("split") != split:
            continue
        artifact_id: str = doc["id"]
        logical_case: str = doc.get("logical_case", artifact_id)
        if split == "holdout" and logical_case in exposed:
            print(
                f"SKIP {artifact_id}: holdout_exposure recorded on "
                f"{exposed[logical_case]}; the case is excluded from the final "
                f"cohort until the exposure is remediated.",
                file=sys.stderr,
            )
            continue
        if not include_variants and not doc.get("canonical", False):
            continue
        if cases is not None:
            if artifact_id not in cases and logical_case not in cases:
                continue
        selected.append(doc)
    return selected


def _upload(
    client: httpx.Client,
    doc: dict[str, Any],
    base_dir: Path,
) -> dict[str, Any]:
    rel_path: str = doc["path"]
    file_path = base_dir / rel_path
    expected_sha: str | None = doc.get("sha256")
    if expected_sha is not None:
        actual_sha = _sha256_file(file_path)
        if actual_sha != expected_sha:
            raise RuntimeError(
                f"SHA256 mismatch for {rel_path}: "
                f"expected {expected_sha}, got {actual_sha}"
            )
    filename = file_path.name
    with file_path.open("rb") as fh:
        resp = client.post(
            "/documents",
            files={"file": (filename, fh)},
            timeout=120.0,
        )
    resp.raise_for_status()
    return resp.json()  # type: ignore[no-any-return]


def _ensure_document(
    client: httpx.Client,
    doc: dict[str, Any],
    artifact_dir: Path,
    base_dir: Path,
    force_reupload: bool = False,
) -> str:
    doc_file = artifact_dir / "document.json"
    if not force_reupload and doc_file.exists():
        try:
            cached = json.loads(doc_file.read_text(encoding="utf-8"))
            return str(cached["id"])
        except Exception:
            pass

    _log(f"[{doc['id']}] uploading document")
    doc_resp = _upload(client, doc, base_dir)
    document_id = str(doc_resp["id"])
    doc_file.write_text(json.dumps(doc_resp, default=str), encoding="utf-8")
    _log(f"[{doc['id']}] uploaded id={document_id}")
    return document_id


def _poll_run(
    client: httpx.Client,
    run_id: str,
    poll_seconds: float,
    timeout_seconds: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    interval = min(0.5, poll_seconds)
    while True:
        resp = client.get(f"/runs/{run_id}", timeout=30.0)
        resp.raise_for_status()
        run: dict[str, Any] = resp.json()
        if run.get("status") in _TERMINAL:
            return run
        if time.monotonic() > deadline:
            raise TimeoutError(f"run {run_id} did not finish within {timeout_seconds}s")
        time.sleep(interval)
        interval = min(poll_seconds, interval * 1.5)


def _create_run(
    client: httpx.Client,
    document_id: str,
    arm: str,
) -> dict[str, Any]:
    resp = client.post(
        "/runs",
        json={"document_id": document_id, "arm": arm},
        timeout=30.0,
    )
    resp.raise_for_status()
    return resp.json()  # type: ignore[no-any-return]


def _run_arm(
    client: httpx.Client,
    doc: dict[str, Any],
    artifact_dir: Path,
    base_dir: Path,
    arm: str,
    poll_seconds: float,
    timeout_seconds: float,
) -> dict[str, Any]:
    artifact_id: str = doc["id"]
    arm_file = artifact_dir / f"{arm}.json"
    pending_file = artifact_dir / f"{arm}.pending.json"
    attempts_log: list[str] = []

    if arm_file.exists():
        _log(f"[{artifact_id}/{arm}] skipping (result exists)")
        return json.loads(arm_file.read_text(encoding="utf-8"))  # type: ignore[no-any-return]

    if pending_file.exists():
        try:
            pending_data = json.loads(pending_file.read_text(encoding="utf-8"))
            run_id = str(pending_data["run_id"])
            _log(f"[{artifact_id}/{arm}] resuming pending run id={run_id}")
            run_final = _poll_run(client, run_id, poll_seconds, timeout_seconds)
            status = run_final.get("status", "unknown")
            elapsed = (run_final.get("metrics") or {}).get("elapsed_ms", 0) / 1000
            _log(
                f"[{artifact_id}/{arm}] resumed run finished "
                f"status={status} elapsed={elapsed:.1f}s"
            )
            run_final["_attempts_log"] = pending_data.get("_attempts_log", [])
            arm_file.write_text(json.dumps(run_final, default=str), encoding="utf-8")
            pending_file.unlink(missing_ok=True)
            return run_final
        except Exception as exc:
            _log(f"[{artifact_id}/{arm}] failed to resume pending run: {exc}")
            pending_file.unlink(missing_ok=True)

    while True:
        document_id = _ensure_document(client, doc, artifact_dir, base_dir)
        try:
            run_rec = _create_run(client, document_id, arm)
        except httpx.HTTPStatusError as exc:
            status_code = exc.response.status_code
            err_code = ""
            try:
                err_code = exc.response.json().get("code", "")
            except Exception:
                pass

            if status_code == 409 and err_code == "run_already_active":
                msg = (
                    f"[{artifact_id}/{arm}] 409 run_already_active"
                    f" – waiting {poll_seconds}s"
                )
                _log(msg)
                attempts_log.append("run_already_active")
                time.sleep(poll_seconds)
                continue

            if status_code in (404, 410) or err_code in (
                "content_expired",
                "not_found",
            ):
                _log(
                    f"[{artifact_id}/{arm}] HTTP {status_code} {err_code or 'expired'} "
                    f"– re-uploading document"
                )
                attempts_log.append(f"reupload_{status_code}_{err_code or 'expired'}")
                _ensure_document(
                    client, doc, artifact_dir, base_dir, force_reupload=True
                )
                continue

            raise

        run_id = str(run_rec["id"])
        _log(f"[{artifact_id}/{arm}] run started id={run_id}")

        # Persisted the moment the run is accepted, so an interrupted poll resumes
        # against the run that is already executing rather than starting a second.
        pending_payload = {
            "run_id": run_id,
            "arm": arm,
            "artifact_id": artifact_id,
            "document_id": document_id,
            "started_at": datetime.now(UTC).isoformat(),
            "_attempts_log": attempts_log,
        }
        pending_file.write_text(
            json.dumps(pending_payload, default=str), encoding="utf-8"
        )

        run_final = _poll_run(client, run_id, poll_seconds, timeout_seconds)
        status = run_final.get("status", "unknown")
        elapsed = (run_final.get("metrics") or {}).get("elapsed_ms", 0) / 1000
        _log(
            f"[{artifact_id}/{arm}] run finished status={status} elapsed={elapsed:.1f}s"
        )
        run_final["_attempts_log"] = attempts_log
        arm_file.write_text(json.dumps(run_final, default=str), encoding="utf-8")
        pending_file.unlink(missing_ok=True)
        return run_final


def _finding_histogram(run_data: dict[str, Any]) -> dict[str, int]:
    return dict(Counter(f.get("code", "unknown") for f in run_data.get("findings", [])))


def _latest_finish(rows: list[dict[str, Any]]) -> str:
    """The last moment any run of this batch ended.

    Compared as instants rather than strings: the system emits both a trailing
    Z and a +00:00 offset, and those sort against each other by punctuation.
    """
    stamps: list[tuple[datetime, str]] = []
    for row in rows:
        raw = row.get("finished_at")
        if not raw:
            continue
        try:
            stamps.append(
                (datetime.fromisoformat(str(raw).replace("Z", "+00:00")), str(raw))
            )
        except ValueError:
            continue
    if not stamps:
        return datetime.now(UTC).isoformat()
    return max(stamps)[1]


def _summaries_on_disk(
    out_root: Path, manifest: list[dict[str, Any]], already: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[Path]]:
    """Rows for results this invocation did not run but the batch holds.

    A file that cannot be read is returned rather than skipped: dropping it
    would make the summary quietly describe less than the directory contains,
    which is the failure mode this whole function exists to remove.
    """
    seen = {(row["artifact_id"], row["arm"]) for row in already}
    by_id = {doc["id"]: doc for doc in manifest}
    rows: list[dict[str, Any]] = []
    unreadable: list[Path] = []
    for artifact_dir in sorted(p for p in out_root.iterdir() if p.is_dir()):
        doc = by_id.get(artifact_dir.name)
        if doc is None:
            continue
        for arm in ("off", "mid", "on"):
            if (artifact_dir.name, arm) in seen:
                continue
            path = artifact_dir / f"{arm}.json"
            if not path.exists():
                continue
            try:
                run_data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                unreadable.append(path)
                continue
            rows.append(_summarise_arm(doc, arm, run_data))
    return rows, unreadable


def _registered_cells(
    proto_data: dict[str, Any], manifest: list[dict[str, Any]]
) -> list[tuple[str, str]]:
    """Every (logical case, arm) the registered comparisons need a result for.

    Read from the protocol, which owns what is compared with what over which
    cohort: the configuration contrast runs over the whole stratum, the graph
    ablation over its own named subset, and a case excluded from the clean
    holdout belongs to neither.
    """
    clean = set(_registered_cohort(manifest))
    cohorts: dict[str, Any] = proto_data.get("cohorts", {})
    cells: set[tuple[str, str]] = set()
    for comparison in proto_data.get("registered_comparisons", []):
        named = comparison.get("cohort")
        if named in cohorts:
            cases = clean & set(cohorts[named].get("cases", []))
        elif named == "full stratum":
            cases = clean
        else:
            raise RuntimeError(
                f"registered comparison {comparison.get('id')!r} is over cohort "
                f"{named!r}, which is neither the full stratum nor an entry in "
                f"cohorts; the batch cannot say what it owes."
            )
        for arm in comparison.get("pair", []):
            cells.update((case, str(arm).lower()) for case in cases)
    return sorted(cells)


def _batch_coverage(
    proto_data: dict[str, Any],
    manifest: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """What the registered comparisons need, against what the directory holds.

    Coverage is not validity: a cell whose run failed is present here and
    reported as a failure elsewhere. Nothing else asks this question. --cases may
    legitimately work through a slice, and an invocation that finishes its slice
    cleanly exits zero, so without this a batch could be registered over the
    whole cohort, run one case, and be recorded as a completed measurement.
    """
    logical = {doc["id"]: doc.get("logical_case", doc["id"]) for doc in manifest}
    canonical = {doc["id"] for doc in manifest if doc.get("canonical")}
    # Format variants never enter a quality aggregate, so they do not fill a cell.
    held = {
        (logical[row["artifact_id"]], row["arm"])
        for row in rows
        if row["artifact_id"] in canonical
    }
    required = _registered_cells(proto_data, manifest)
    missing = [
        {"case": case, "arm": arm} for case, arm in required if (case, arm) not in held
    ]
    return {
        "cohort": _registered_cohort(manifest),
        "required_cells": len(required),
        "missing_cells": missing,
        "complete": not missing,
    }


def _batch_has_failure(rows: list[dict[str, Any]]) -> bool:
    """Whether any result in this batch failed, whoever ran it.

    The summary covers the directory, so the exit code has to as well: a batch
    finished across two invocations must not report success because the
    invocation that happened to finish it went cleanly.
    """
    return any(
        row.get("status") in ("failed", "cancelled")
        or not row.get("measurement_valid", True)
        for row in rows
    )


def _summarise_arm(
    doc: dict[str, Any], arm: str, run_data: dict[str, Any]
) -> dict[str, Any]:
    metrics: dict[str, Any] = run_data.get("metrics") or {}
    attempts: list[dict[str, Any]] = metrics.get("attempts", [])
    cost_rec: dict[str, Any] = metrics.get("cost") or {}
    log: list[str] = run_data.get("_attempts_log", [])
    return {
        "artifact_id": doc["id"],
        "arm": arm,
        "status": run_data.get("status"),
        "measurement_valid": run_data.get("measurement_valid", True),
        # When this run ended, so the batch window can be read off its runs
        # rather than off whichever invocation last wrote the summary.
        "finished_at": run_data.get("finished_at"),
        "elapsed_ms": metrics.get("elapsed_ms"),
        "input_tokens": metrics.get("input_tokens"),
        "output_tokens": metrics.get("output_tokens"),
        "units_total": metrics.get("units_total"),
        "units_with_finding": metrics.get("units_with_finding"),
        "units_not_processed": metrics.get("units_not_processed"),
        "context_edge_count": metrics.get("context_edge_count"),
        "finding_histogram": _finding_histogram(run_data),
        "error_code": (run_data.get("error") or {}).get("code"),
        "attempts_count": len(attempts),
        "model_calls": len(attempts),
        "attempts_statuses": [a.get("status") for a in attempts],
        "orchestration_events": log,
        "monetary_cost_microunits": cost_rec.get("monetary_cost_microunits"),
        "cost_unknown_reason": cost_rec.get("unknown_reason"),
        # Newer runs carry these under metrics; older ones at the top level.
        **{
            key: (metrics[key] if metrics.get(key) is not None else run_data.get(key))
            for key in (
                "finder_tool_turns",
                "finder_search_calls",
                "finder_budget_exhausted_units",
            )
        },
    }


def main(
    argv: list[str] | None = None,
    transport: httpx.BaseTransport | None = None,
) -> int:
    parser = argparse.ArgumentParser(
        description="Drive evaluation runs through the REST API."
    )
    parser.add_argument("--split", required=True, choices=["development", "holdout"])
    parser.add_argument(
        "--cases", help="Comma-separated artifact IDs or logical case IDs"
    )
    parser.add_argument("--include-variants", action="store_true")
    parser.add_argument("--arms", default=",".join(_ARMS_DEFAULT))
    parser.add_argument("--base-url", default="http://127.0.0.1:8001/api/v1")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--label", default="")
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--timeout-seconds", type=float, default=1000.0)
    parser.add_argument("--i-confirm-holdout", action="store_true")
    parser.add_argument(
        "--allow-source-format-fallback",
        action="store_true",
        help=(
            "Measure a corpus whose acts were not all read from the source format "
            "the manifest declares. The source formats actually used are recorded "
            "in the batch identity and must be interpreted as a deviation."
        ),
    )
    args = parser.parse_args(argv)

    proto_data = _load_protocol()

    if args.split == "holdout":
        if not args.i_confirm_holdout:
            reminder = _load_holdout_reminder()
            print(
                f"ERROR: --split holdout requires --i-confirm-holdout.\n\n"
                f"Protocol holdout_discipline reminder:\n  {reminder}\n\n"
                f"Holdout is measured exactly once. Pass --i-confirm-holdout "
                f"only for the final registered run.",
                file=sys.stderr,
            )
            return 2

        if proto_data.get("state") != "frozen":
            print(
                f"ERROR: protocol state is {proto_data.get('state')!r}, "
                f"expected 'frozen'. Holdout measurement is strictly forbidden "
                f"until the protocol is frozen.",
                file=sys.stderr,
            )
            return 2

        thresholds = proto_data.get("decision_rule", {}).get("thresholds", {})
        unfixed = [
            k for k, v in thresholds.items() if v.get("owed") and v.get("value") is None
        ]
        if unfixed:
            print(
                f"ERROR: protocol has unfixed owed thresholds: {', '.join(unfixed)}. "
                f"Holdout measurement cannot run until all owed thresholds "
                f"carry fixed values.",
                file=sys.stderr,
            )
            return 2

        failure_policy = proto_data.get("terminal_failure_policy", {})
        if failure_policy.get("owed") and failure_policy.get("value") is None:
            print(
                "ERROR: protocol has no fixed terminal_failure_policy. A run that "
                "dies for a reason outside the system under test has no declared "
                "disposition, and deciding one after seeing a final result is "
                "choosing the rule from the answer. Fix it before measuring.",
                file=sys.stderr,
            )
            return 2

        # Not whether holdout material has ever been seen -- a permanent audit fact
        # that would forbid the final measurement outright rather than forbid
        # repeating it -- but whether the protocol is still open. state_reason
        # defines that as the union of owed thresholds and open_obligations, so any
        # entry blocks the final run, however it happens to be worded.
        owed_obligations = list(proto_data.get("open_obligations", []))
        if owed_obligations:
            listed = "\n  - ".join(owed_obligations)
            print(
                f"ERROR: {len(owed_obligations)} open obligation(s) remain, and the "
                f"protocol's own state rule counts every one of them as unfinished "
                f"business. The registered batch cannot run until they are closed:"
                f"\n  - {listed}",
                file=sys.stderr,
            )
            return 2

        # The protocol version that says what the earlier result was and why it
        # is being repeated is what registered_repeat_series is: a series it
        # names, at the directory it names, is the authorized repeat. Everything
        # else still meets the refusal below.
        slot = _series_slot(proto_data, Path(args.out))
        measurement_state = proto_data.get(
            "final_registered_measurement_state", "not_started"
        )
        if slot is None and measurement_state != "not_started":
            print(
                f"ERROR: protocol reports final_registered_measurement_state="
                f"{measurement_state!r}, and this output directory is not a "
                f"series registered_repeat_series names. The registered final "
                f"batch is not a thing to run twice; a rerun needs a new "
                f"protocol version that says what the earlier result was and why "
                f"it is being repeated.",
                file=sys.stderr,
            )
            return 2
        if slot is not None:
            _log(
                f"NOTE: this directory is registered series {slot.get('series')} "
                f"of the final measurement, recorded as {slot.get('state')!r}."
            )

    else:
        slot = None

    cases_filter: list[str] | None = (
        [c.strip() for c in args.cases.split(",") if c.strip()] if args.cases else None
    )
    arms: list[str] = [a.strip().lower() for a in args.arms.split(",") if a.strip()]

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    manifest_data = _load_manifest_data()
    manifest = manifest_data["documents"]
    selected = _select_artifacts(
        manifest, args.split, cases_filter, args.include_variants
    )
    if not selected:
        _log("No artifacts selected – check --split and --cases.")
        return 1

    client_kwargs: dict[str, Any] = {"base_url": args.base_url}
    if transport is not None:
        client_kwargs["transport"] = transport

    started_at = datetime.now(UTC).isoformat()
    arm_summaries: list[dict[str, Any]] = []
    any_failed = False

    with httpx.Client(**client_kwargs) as client:
        try:
            config_resp = client.get("/config", timeout=15.0)
            config_resp.raise_for_status()
            config_data: dict[str, Any] = config_resp.json()
        except Exception as exc:
            _log(f"ERROR: could not fetch /config: {exc}")
            return 1

        # Not measured_mode, which the server never unsets and which therefore
        # asserted nothing. What a batch needs is that the server refuses
        # interactive work for its duration: no ask or contest can start beside a
        # measured run and spend its wall clock, which is a reported metric.
        if not config_data.get("evaluation_batch_open"):
            _log(
                "ERROR: server evaluation_batch_open is "
                f"{config_data.get('evaluation_batch_open')!r}. "
                "A batch must run against a sealed server. Start the backend "
                "with EVALUATION_BATCH_OPEN=true."
            )
            return 1

        # Before _batch_identity, which folds corpus_source_format in as a parity
        # field: refusing after it would record the refused value, so the operator
        # rebuilds the corpus, reruns the same --out and is refused again for
        # config_fingerprint drift about the problem they just fixed.
        #
        # A fallback corpus is servable but is not the corpus the manifest
        # describes, and unrecorded is refused with it, because a snapshot that
        # cannot answer the question has not answered it cleanly.
        if args.split == "holdout":
            source_format = config_data.get("corpus_source_format")
            if source_format != "as_declared" and not args.allow_source_format_fallback:
                print(
                    f"ERROR: served corpus_source_format is {source_format!r}, "
                    f"expected 'as_declared'. The corpus was not read entirely "
                    f"from the source formats its manifest declares, so it is not "
                    f"the corpus this protocol's results are measured on. Rebuild "
                    f"once ELI's HTML routes recover, or pass "
                    f"--allow-source-format-fallback to measure it as a recorded "
                    f"deviation.",
                    file=sys.stderr,
                )
                return 2
            if source_format != "as_declared":
                _log(
                    f"DEVIATION: measuring a corpus whose source format is "
                    f"{source_format!r} rather than 'as_declared', by explicit "
                    f"--allow-source-format-fallback. The batch identity records it."
                )

        # Before the identity too, and for the same reason: a repeat measured
        # against another model, corpus or provider is not a repeat, and the
        # identity would record the difference only once the runs were paid for.
        if slot is not None:
            parity_error = _series_parity_error(
                config_data, proto_data["registered_repeat_series"]
            )
            if parity_error is not None:
                print(
                    f"ERROR: registered series {slot.get('series')} cannot run: "
                    f"{parity_error}",
                    file=sys.stderr,
                )
                return 2

        identity, identity_error = _batch_identity(
            out_root, args, proto_data, config_data, arms
        )
        if identity_error is not None:
            print(identity_error, file=sys.stderr)
            return 2

        if args.split == "holdout":
            # Read before claiming: a comparison naming a cohort the protocol
            # does not define raises, and raising after the runs had executed
            # would have spent the one registered batch on a typo and left no
            # summary to show for it.
            try:
                _registered_cells(proto_data, manifest)
            except RuntimeError as exc:
                print(f"ERROR: {exc}", file=sys.stderr)
                return 2
            claim_error = _claim_final_batch(
                identity, _registered_cohort(manifest), out_root, slot
            )
            if claim_error is not None:
                print(f"ERROR: {claim_error}", file=sys.stderr)
                return 2

        base_dir = ROOT / "evaluation-data"

        for doc in selected:
            artifact_id: str = doc["id"]
            artifact_dir = out_root / artifact_id
            artifact_dir.mkdir(parents=True, exist_ok=True)

            for arm in arms:
                try:
                    run_data = _run_arm(
                        client,
                        doc,
                        artifact_dir,
                        base_dir,
                        arm,
                        args.poll_seconds,
                        args.timeout_seconds,
                    )
                except Exception as exc:
                    _log(f"[{artifact_id}/{arm}] ERROR: {exc}")
                    any_failed = True
                    continue

                if not run_data.get("measurement_valid", True):
                    _log(f"[{artifact_id}/{arm}] ERROR: measurement_valid is false")
                    any_failed = True

                if run_data.get("status") in ("failed", "cancelled"):
                    any_failed = True

                arm_summaries.append(_summarise_arm(doc, arm, run_data))

    # A batch is what its directory holds, not what this invocation asked for.
    # Resuming with the arms that are still missing is the ordinary way to
    # finish a paused batch, and a summary covering only those would under-report
    # the measurement it is the record of.
    on_disk, unreadable = _summaries_on_disk(out_root, manifest, arm_summaries)
    arm_summaries.extend(on_disk)
    for path in unreadable:
        _log(f"ERROR: {path} exists but cannot be read; the summary omits it")
        any_failed = True
    any_failed = any_failed or _batch_has_failure(arm_summaries)
    manifest_sha = _sha256_file(MANIFEST) if MANIFEST.exists() else None
    protocol_sha = _sha256_file(PROTOCOL) if PROTOCOL.exists() else None
    # The window belongs to the batch too: a resumed batch reassembles this file
    # in a fraction of a second, and a summary claiming that window would
    # describe nothing that happened.
    finished_at = _latest_finish(arm_summaries)
    # The header describes the batch the rows describe. Reporting this
    # invocation's selection above rows covering the whole directory would tell
    # a reader that six results came from one case.
    requested: dict[str, Any] = identity.get("requested", {})
    summary: dict[str, Any] = {
        "label": args.label,
        "split": args.split,
        "cases": requested.get("cases") or cases_filter,
        "arms": requested.get("arms") or arms,
        "include_variants": bool(
            requested.get("include_variants", args.include_variants)
        ),
        "invocations": requested.get("invocations", 1),
        "dataset_version": manifest_data.get("dataset_version"),
        "manifest_sha256": manifest_sha,
        "protocol_version": proto_data.get("protocol_version"),
        "protocol_sha256": protocol_sha,
        "config": config_data,
        "started_at": identity.get("started_at", started_at),
        "finished_at": finished_at,
        "summary_generated_at": datetime.now(UTC).isoformat(),
    }
    # A registered batch that ran a slice is not a registered batch that ran.
    # The invocation still succeeded -- working through the cohort in pieces is
    # allowed -- so this is said in the record and on the way out, not in the
    # exit code, which answers whether what ran went cleanly.
    if args.split == "holdout":
        coverage = _batch_coverage(proto_data, manifest, arm_summaries)
        summary["registered_batch"] = coverage
        if not coverage["complete"]:
            cells = coverage["missing_cells"]
            owed = ", ".join(f"{cell['case']}/{cell['arm']}" for cell in cells[:6])
            if len(cells) > 6:
                owed += f", and {len(cells) - 6} more"
            _log(
                f"NOTE: the registered batch is not complete: {len(cells)} of "
                f"{coverage['required_cells']} cells have no result ({owed})."
            )
    summary["arm_results"] = arm_summaries
    summary_path = out_root / "summary.json"
    summary_path.write_text(
        json.dumps(summary, default=str, indent=2), encoding="utf-8"
    )
    _log(f"Summary written to {summary_path}")

    return 1 if any_failed else 0


if __name__ == "__main__":
    sys.exit(main())
