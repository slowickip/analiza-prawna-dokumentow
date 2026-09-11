"""The dimensions on which two configurations must agree to be comparable.

Recorded on every run record as the run happens and compared afterwards, because
parity is a property of the set of runs forming one comparison rather than of any
single run the artefact executes.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from contract_analyzer.storage import RunRecord

RETRY_POLICY = "bounded-3"
CONFIG_VERSION = "run-config-v1"
# Chosen by the fixed worksheet cycle, not measurement.
GRAPH_TOPOLOGY_VERSION = "worksheet-unit-cycle-v5"

# The graph topology version is stamped on every run but deliberately excluded
# here: it cannot differ between arms, so including it would be a permanently
# green check.
PARITY_FIELDS: tuple[str, ...] = (
    "input_hash",
    "requested_model",
    "returned_model",
    "prompt_bundle_version",
    "corpus_snapshot_id",
    "tool_bundle_version",
    "parameters",
    "retry_policy",
    "concurrency",
    "budgets",
)


@dataclass(frozen=True)
class ParityBundle:
    input_hash: str
    requested_model: str
    returned_model: str | None
    prompt_bundle_version: str
    corpus_snapshot_id: str
    tool_bundle_version: str
    parameters: Mapping[str, str | int | float | bool | None]
    retry_policy: str
    concurrency: int
    wall_budget_seconds: float


@dataclass(frozen=True)
class ArmParityResult:
    measurement_valid: bool
    mismatched_fields: tuple[str, ...]


def _parity_value(run: RunRecord, field_name: str) -> object:
    """One run's value for a parity dimension, in a comparable form.

    One dimension is not a plain attribute: parameters is a mapping, so it is
    serialised with sorted keys rather than compared by identity. budgets is the
    wall limit, which is the only budget a run is held to.
    """
    if field_name == "parameters":
        return json.dumps(dict(run.parameters), sort_keys=True)
    if field_name == "budgets":
        return run.wall_budget_seconds
    return getattr(run, field_name)


def compare_arm_parity(runs: Sequence[RunRecord]) -> ArmParityResult:
    """Whether these runs are comparable, naming every dimension that differs.

    Takes stored run records rather than executing anything. Interactive children are
    removed first, even if malformed historical data marks one measurement-valid. A
    root recorded with measurement_valid false taints the comparison regardless of
    the ten dimensions.
    """
    runs = tuple(run for run in runs if run.parent_run_id is None)
    if not runs:
        return ArmParityResult(measurement_valid=False, mismatched_fields=("runs",))
    baseline = runs[0]
    mismatched = [
        field_name
        for field_name in PARITY_FIELDS
        if any(
            _parity_value(run, field_name) != _parity_value(baseline, field_name)
            for run in runs[1:]
        )
    ]
    if any(not run.measurement_valid for run in runs):
        mismatched.append("measurement_valid")
    return ArmParityResult(
        measurement_valid=not mismatched,
        mismatched_fields=tuple(mismatched),
    )
