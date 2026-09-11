#!/usr/bin/env python3
"""Verify matching configurations across stored evaluation runs.

Reads registered comparisons from the evaluation protocol and compares stored
RunRecords in the run metadata store across the ten parity dimensions in
PARITY_FIELDS. Parity is a property of the set of runs forming one comparison:
arms differ by design, but configurations, models, prompts, tools, budgets,
and inputs must be identical for any comparison between them to mean anything.

A green run means:
- Every registered comparison had at least one eligible document pair checked.
- All checked pairs satisfied arm parity (measurement_valid is true and no
  parity dimensions diverged).
- No document had ambiguous completed runs for any arm of a comparison.

Failure conditions, each of which exits non-zero and is named in the output:
- Any checked pair whose parity result is not measurement_valid.
- An input that has more than one completed run for the same arm, making the
  pair ambiguous (runs are reported and not arbitrarily chosen).
- A registered comparison for which no input has both arms (0 pairs checked).

Runs are grouped by input_hash rather than document_id: document_id identifies an
upload session, so one file analysed under three arms across two uploads carries
three of them, while input_hash is the content identity the comparison is about.

Inputs with completed runs for only one arm of a pair are skipped and counted
as skipped without failing the check. Runs in any status other than 'completed'
and every run with a parent are ignored as not comparison material.

--batch names a recorded results directory and may be repeated. One store that
has served repeated series of the same comparison holds several completed runs
per document and arm, which is ambiguity only while the series are read as one
batch; the result files say which runs belong together. Given, each batch is
checked on its own and then every document and arm is compared across batches,
which is what asks whether a repeat series repeated the configuration rather
than merely being internally consistent.

This diagnostic opens PostgreSQL without schema creation or interrupted-run
reconciliation, then reads the recorded configuration. It must not change the
status of runs owned by a live server.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend" / "src"))

from contract_analyzer.agents.parity import compare_arm_parity  # noqa: E402
from contract_analyzer.domain import ArmCode  # noqa: E402
from contract_analyzer.storage import (  # noqa: E402
    RunRecord,
    open_metadata_store,
)


def _is_database_url(target: str) -> bool:
    return target.startswith(("postgres://", "postgresql://"))


def _batch_run_ids(directory: Path) -> tuple[set[str], list[str]]:
    """The run ids one recorded results directory holds.

    A store that has served several series holds several completed runs per
    document and arm, which is ambiguity only if the series are read as one
    batch. The result files say which runs belong together, so they are what
    separates them.
    """
    ids: set[str] = set()
    unreadable: list[str] = []
    for path in sorted(directory.glob("*/*.json")):
        if path.stem not in {arm.value for arm in ArmCode}:
            continue
        try:
            run = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            unreadable.append(f"{path}: {exc}")
            continue
        run_id = run.get("id")
        if run_id:
            ids.add(str(run_id))
        else:
            unreadable.append(f"{path}: no run id")
    return ids, unreadable


def _group_runs(
    completed_runs: list[RunRecord], batches: Sequence[Path]
) -> tuple[list[tuple[str, list[RunRecord]]], bool]:
    """The runs to check, split into the batches that produced them."""
    if not batches:
        return [("all runs", completed_runs)], False
    held = {str(run.id): run for run in completed_runs}
    groups: list[tuple[str, list[RunRecord]]] = []
    has_failure = False
    for directory in batches:
        recorded, unreadable = _batch_run_ids(directory)
        # A result file that cannot be read drops its run out of both the batch
        # and the cross-batch comparison, so a half-parseable batch would come
        # back quieter and greener than a whole one.
        for problem in unreadable:
            has_failure = True
            print(f"UNREADABLE: {problem}")
        missing = sorted(recorded - set(held))
        if missing:
            has_failure = True
            print(
                f"MISSING: {directory.name} records {len(missing)} run(s) the "
                f"metadata store does not hold as completed roots: "
                f"{', '.join(missing)}"
            )
        if not recorded:
            has_failure = True
            print(f"MISSING: {directory} holds no recorded runs")
        groups.append(
            (directory.name, [held[run_id] for run_id in sorted(recorded & set(held))])
        )
    return groups, has_failure


def _check_across_batches(groups: list[tuple[str, list[RunRecord]]]) -> bool:
    """Whether repeated series of one comparison ran under one configuration.

    The same document and arm in two series agree on the arm by construction, so
    what the ten dimensions still answer here is the question a repeat depends
    on: whether the second series measured the configuration the first did. A
    served-configuration check cannot answer it alone, because concurrency, the
    wall budget, the retry policy and the call parameters are recorded on the
    run and served nowhere.
    """
    base_name, base_runs = groups[0]
    base = {(run.input_hash, run.arm): run for run in base_runs}
    has_failure = False
    for name, runs in groups[1:]:
        later = {(run.input_hash, run.arm): run for run in runs}
        # Both directions, because a repeat that ran three of nine cells has a
        # counterpart for every cell it holds. Walking the repeat alone reports
        # that as three cells checked and nothing skipped.
        uncovered = sorted(
            f"{document} {arm.value.upper()}"
            for document, arm in base.keys() - later.keys()
        )
        unmatched = sorted(
            f"{document} {arm.value.upper()}"
            for document, arm in later.keys() - base.keys()
        )
        checked = failed = repeated = 0
        for document, arm in sorted(
            base.keys() & later.keys(), key=lambda cell: (cell[0], cell[1].value)
        ):
            counterpart, run = base[(document, arm)], later[(document, arm)]
            if counterpart.id == run.id:
                # One run compared with itself agrees on all ten dimensions and
                # says nothing about repetition.
                repeated += 1
                has_failure = True
                print(
                    f"{base_name} vs {name} {document} {arm.value.upper()}: "
                    f"the same run {run.id} on both sides, not a repetition"
                )
                continue
            checked += 1
            result = compare_arm_parity((counterpart, run))
            if not result.measurement_valid:
                failed += 1
                has_failure = True
                print(
                    f"{base_name} vs {name} {document} "
                    f"{arm.value.upper()} MISMATCH: "
                    f"{', '.join(result.mismatched_fields)}"
                )
        print(
            f"{base_name} vs {name}: {checked} cells checked, {failed} failed, "
            f"{len(uncovered)} uncovered, {len(unmatched)} unmatched"
        )
        if uncovered:
            has_failure = True
            print(
                f"FAILURE: {len(uncovered)} cell(s) of {base_name} have no "
                f"counterpart in {name}: {', '.join(uncovered)}"
            )
        if unmatched:
            has_failure = True
            print(
                f"FAILURE: {len(unmatched)} cell(s) of {name} have no "
                f"counterpart in {base_name}: {', '.join(unmatched)}"
            )
        if checked == 0 and not repeated:
            has_failure = True
            print(
                f"FAILURE: no cell of {name} was compared with {base_name}; a "
                f"repeat series is compared cell for cell or not at all"
            )
    return has_failure


def check_runs(
    metadata: str,
    protocol_path: Path,
    batches: Sequence[Path] = (),
) -> int:
    if not protocol_path.is_file():
        print(f"protocol file not found: {protocol_path}", file=sys.stderr)
        return 1
    if not _is_database_url(metadata) and not Path(metadata).is_file():
        print(f"metadata store not found: {metadata}", file=sys.stderr)
        return 1

    try:
        protocol_data = json.loads(protocol_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"failed to parse protocol {protocol_path}: {exc}", file=sys.stderr)
        return 1

    registered_comparisons = protocol_data.get("registered_comparisons")
    if not isinstance(registered_comparisons, list) or not registered_comparisons:
        print(
            f"protocol {protocol_path} contains no valid 'registered_comparisons'",
            file=sys.stderr,
        )
        return 1

    # Read-only: opening the store for writing reconciles interrupted runs,
    # which settles the runs of whichever server is live. A diagnostic must not
    # be able to end a measurement it is only inspecting.
    store = open_metadata_store(metadata, read_only=True)
    all_runs = store.list_runs()

    # Only completed roots are comparison material. Interactive children are never
    # arms, even if corrupt historical data marks one measurement-valid.
    completed_runs = [
        run
        for run in all_runs
        if run.status == "completed" and run.parent_run_id is None
    ]

    groups, has_failure = _group_runs(completed_runs, batches)
    for label, runs in groups:
        has_failure |= _check_group(
            registered_comparisons, runs, label if len(groups) > 1 else ""
        )
    if len(groups) > 1:
        has_failure |= _check_across_batches(groups)
    return 1 if has_failure else 0


def _check_group(
    registered_comparisons: list[dict[str, Any]],
    completed_runs: list[RunRecord],
    label: str,
) -> bool:
    """Whether one batch of runs satisfies every registered comparison."""
    prefix = f"{label} " if label else ""

    # Group by input_hash, then by arm. The pair a comparison needs is the same
    # document under two arms, and document_id identifies an upload session, not a
    # document: analysing one file under three arms across two uploads yields three
    # document_ids. input_hash is the content identity, and it is a parity dimension
    # in its own right, so runs grouped by it agree on the input by construction.
    runs_by_doc: dict[str, dict[ArmCode, list[RunRecord]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for run in completed_runs:
        runs_by_doc[run.input_hash][run.arm].append(run)

    has_failure = False

    # Check for ambiguous runs: a document that has more than one completed
    # run for the same arm makes any pair involving that arm ambiguous.
    # Do not pick one; report document, arm and run ids.
    for input_hash in sorted(runs_by_doc.keys()):
        for arm in sorted(runs_by_doc[input_hash].keys(), key=lambda a: a.value):
            runs = runs_by_doc[input_hash][arm]
            if len(runs) > 1:
                has_failure = True
                run_ids_str = ", ".join(str(r.id) for r in runs)
                print(
                    f"{prefix}AMBIGUOUS: input {input_hash} has {len(runs)} "
                    f"completed runs for arm {arm.value.upper()}: {run_ids_str}"
                )

    for comp in registered_comparisons:
        comp_id = comp.get("id", "unknown_comparison")
        pair = comp.get("pair")
        if not isinstance(pair, list) or len(pair) != 2:
            print(f"invalid pair for comparison {comp_id}: {pair}", file=sys.stderr)
            has_failure = True
            continue

        try:
            arm_a = ArmCode(str(pair[0]).lower())
            arm_b = ArmCode(str(pair[1]).lower())
        except ValueError as exc:
            print(f"invalid arm in pair {pair} for {comp_id}: {exc}", file=sys.stderr)
            has_failure = True
            continue

        checked_count = 0
        failed_count = 0
        skipped_count = 0

        for input_hash in sorted(runs_by_doc.keys()):
            runs_a = runs_by_doc[input_hash].get(arm_a, [])
            runs_b = runs_by_doc[input_hash].get(arm_b, [])

            # Ambiguous: multiple completed runs for an arm in this pair.
            # Do not pick one; skip pair comparison.
            if len(runs_a) > 1 or len(runs_b) > 1:
                continue

            if len(runs_a) == 1 and len(runs_b) == 1:
                checked_count += 1
                result = compare_arm_parity((runs_a[0], runs_b[0]))
                pair_ids = f"{runs_a[0].id} {runs_b[0].id}"
                if result.measurement_valid:
                    print(f"{prefix}{comp_id} {input_hash} OK ({pair_ids})")
                else:
                    failed_count += 1
                    has_failure = True
                    mismatched_str = ", ".join(result.mismatched_fields)
                    print(
                        f"{prefix}{comp_id} {input_hash} MISMATCH: "
                        f"{mismatched_str} ({pair_ids})"
                    )
            elif (len(runs_a) == 1 and len(runs_b) == 0) or (
                len(runs_a) == 0 and len(runs_b) == 1
            ):
                skipped_count += 1

        print(
            f"{prefix}{comp_id}: {checked_count} pairs checked, "
            f"{failed_count} failed, {skipped_count} skipped"
        )

        if checked_count == 0:
            has_failure = True
            print(
                f"{prefix}FAILURE: registered comparison '{comp_id}' had 0 pairs "
                f"checked; no input has completed runs in both arms "
                f"({pair[0]}, {pair[1]})"
            )

    return has_failure


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default=os.environ.get("DATABASE_URL"),
        help=(
            "PostgreSQL URL of the metadata store the runs were written to "
            "(default: $DATABASE_URL)"
        ),
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "evaluation-data" / "protocol.json",
        help="path to protocol.json (default: evaluation-data/protocol.json)",
    )
    parser.add_argument(
        "--batch",
        type=Path,
        action="append",
        default=[],
        help=(
            "a recorded results directory; repeatable. Given, the check reads "
            "the run ids each directory holds and checks each batch on its own, "
            "then checks the same document and arm across batches. A store that "
            "has served repeated series holds several completed runs per cell, "
            "which is ambiguity only when the series are read as one batch."
        ),
    )
    args = parser.parse_args(argv)

    if not args.metadata:
        parser.error("--metadata is required when DATABASE_URL is not set")

    metadata = args.metadata
    if not _is_database_url(metadata):
        candidate = Path(metadata)
        if not candidate.is_file() and not candidate.is_absolute():
            relocated = ROOT / candidate
            if relocated.is_file():
                metadata = str(relocated)

    protocol_path = args.protocol
    if not protocol_path.is_file() and not protocol_path.is_absolute():
        candidate = ROOT / protocol_path
        if candidate.is_file():
            protocol_path = candidate

    return check_runs(
        metadata=metadata,
        protocol_path=protocol_path,
        batches=args.batch,
    )


if __name__ == "__main__":
    sys.exit(main())
