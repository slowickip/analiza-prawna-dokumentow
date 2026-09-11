#!/usr/bin/env python3
"""Summarise evaluation results produced by run_evaluation.py.

Reads one or more result directories and prints a Markdown report to stdout
with per-arm / per-stratum aggregates and a per-artifact detail table.
Numbers only; no interpretation prose.  Costs are reported as
"unknown, because price table unavailable" when unknown.

Usage
-----
    uv run python scripts/summarize_results.py \\
        evaluation-data/results/final/2026-09-08-holdout
    uv run python scripts/summarize_results.py dir1 dir2
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "evaluation-data" / "manifest.json"

sys.path.insert(0, str(ROOT / "scripts"))
from run_evaluation import _summarise_arm  # noqa: E402

# ── helpers ────────────────────────────────────────────────────────────────────


def _median(values: list[float]) -> float | None:
    return float(statistics.median(values)) if values else None


def _mean(values: list[float]) -> float | None:
    return float(statistics.mean(values)) if values else None


def _load_stratum_map() -> dict[str, str]:
    """Return {artifact_id: stratum} from manifest.json."""
    try:
        docs: list[dict[str, Any]] = json.loads(MANIFEST.read_text(encoding="utf-8"))[
            "documents"
        ]
        return {d["id"]: d.get("stratum", "unknown") for d in docs}
    except Exception:
        return {}


def _normalize_arm_record(
    rec: dict[str, Any],
    fallback_artifact_id: str = "",
    fallback_arm: str = "",
) -> dict[str, Any]:
    metrics = rec.get("metrics")
    if metrics is not None and isinstance(metrics, dict):
        doc_id = fallback_artifact_id or str(rec.get("document_id", ""))
        arm = rec.get("arm") or fallback_arm
        res = _summarise_arm({"id": doc_id}, arm, rec)
        res["_artifact_id"] = fallback_artifact_id
        res["_arm"] = fallback_arm
        return res
    return rec


def _load_result_dirs(dirs: list[Path]) -> list[dict[str, Any]]:
    """Collect all arm result dicts from result directories."""
    records: list[dict[str, Any]] = []
    for result_dir in dirs:
        summary_path = result_dir / "summary.json"
        if not summary_path.exists():
            # fall back to scanning arm JSON files directly
            for artifact_dir in result_dir.iterdir():
                if not artifact_dir.is_dir():
                    continue
                for arm_file in artifact_dir.glob("*.json"):
                    if arm_file.stem == "document":
                        continue
                    try:
                        raw_data = json.loads(arm_file.read_text(encoding="utf-8"))
                        norm = _normalize_arm_record(
                            raw_data,
                            fallback_artifact_id=artifact_dir.name,
                            fallback_arm=arm_file.stem,
                        )
                        norm["_result_dir"] = str(result_dir)
                        records.append(norm)
                    except Exception:
                        pass
        else:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            for arm_rec in summary.get("arm_results", []):
                arm_rec["_result_dir"] = str(result_dir)
                records.append(arm_rec)
    return records


# ── aggregation ────────────────────────────────────────────────────────────────


class _ArmStratumBucket:
    def __init__(self) -> None:
        self.runs = 0
        self.completed = 0
        self.failed = 0
        self.invalid = 0
        self.elapsed_s: list[float] = []
        self.input_tokens: list[int] = []
        self.output_tokens: list[int] = []
        self.total_attempts = 0
        self.total_not_processed = 0
        self.histogram: dict[str, int] = {}
        self.finder_tool_turns: list[int] = []
        self.finder_search_calls: list[int] = []

    def add(self, rec: dict[str, Any]) -> None:
        self.runs += 1
        status = rec.get("status", "")
        measurement_valid = rec.get("measurement_valid", True)
        if not measurement_valid:
            self.invalid += 1
        if status == "completed":
            self.completed += 1
        elif status in ("failed", "cancelled"):
            self.failed += 1

        # Only aggregate quantitative metrics for valid measurements
        if not measurement_valid:
            return

        elapsed_ms = rec.get("elapsed_ms")
        if elapsed_ms is not None:
            self.elapsed_s.append(float(elapsed_ms) / 1000.0)
        inp = rec.get("input_tokens")
        if inp is not None:
            self.input_tokens.append(int(inp))
        out = rec.get("output_tokens")
        if out is not None:
            self.output_tokens.append(int(out))
        self.total_attempts += int(rec.get("attempts_count", 0))
        self.total_not_processed += int(rec.get("units_not_processed") or 0)
        for code, cnt in (rec.get("finding_histogram") or {}).items():
            self.histogram[code] = self.histogram.get(code, 0) + int(cnt)
        ftt = rec.get("finder_tool_turns")
        if ftt is not None:
            self.finder_tool_turns.append(int(ftt))
        fsc = rec.get("finder_search_calls")
        if fsc is not None:
            self.finder_search_calls.append(int(fsc))


# ── rendering ──────────────────────────────────────────────────────────────────


def _fmt_float(v: float | None, decimals: int = 2) -> str:
    if v is None:
        return "—"
    return f"{v:.{decimals}f}"


def _render_bucket(arm: str, stratum: str, b: _ArmStratumBucket) -> str:
    mean_elapsed = _fmt_float(_mean(b.elapsed_s))
    med_elapsed = _fmt_float(_median(b.elapsed_s))
    mean_inp = _fmt_float(_mean([float(x) for x in b.input_tokens]))
    mean_out = _fmt_float(_mean([float(x) for x in b.output_tokens]))
    status_parts = [
        f"Runs: {b.runs}",
        f"completed: {b.completed}",
        f"failed: {b.failed}",
    ]
    if b.invalid > 0:
        status_parts.append(f"invalid: {b.invalid}")
    lines = [
        f"### Arm `{arm}` · stratum `{stratum}`\n",
        f"- {', '.join(status_parts)}",
        f"- Elapsed s: mean {mean_elapsed} / median {med_elapsed}",
        f"- Input tokens: mean {mean_inp} / total {sum(b.input_tokens)}",
        f"- Output tokens: mean {mean_out} / total {sum(b.output_tokens)}",
        f"- Total attempts: {b.total_attempts}",
        f"- Total not_processed units: {b.total_not_processed}",
    ]
    if b.histogram:
        lines.append(
            "- Finding codes: "
            + ", ".join(f"{k}={v}" for k, v in sorted(b.histogram.items()))
        )
    if b.finder_tool_turns:
        mean_ftt = _fmt_float(_mean([float(x) for x in b.finder_tool_turns]))
        lines.append(
            f"- Finder tool turns: mean {mean_ftt} / total {sum(b.finder_tool_turns)}"
        )
    if b.finder_search_calls:
        mean_fsc = _fmt_float(_mean([float(x) for x in b.finder_search_calls]))
        lines.append(
            f"- Finder search calls: mean {mean_fsc}"
            f" / total {sum(b.finder_search_calls)}"
        )
    lines.append("")
    return "\n".join(lines)


def _cost_cell(rec: dict[str, Any]) -> str:
    cost_unknown = rec.get("cost_unknown_reason")
    if cost_unknown:
        return "unknown, because price table unavailable"
    microunits = rec.get("monetary_cost_microunits")
    if microunits is None:
        return "unknown, because price table unavailable"
    return str(microunits)


def _render_detail_table(records: list[dict[str, Any]]) -> str:
    if not records:
        return "_No records._\n"
    lines = [
        "| Artifact | Arm | Elapsed s | In tokens | Out tokens "
        "| Findings | Not processed | Status | Cost |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for rec in records:
        artifact_id = rec.get("artifact_id") or rec.get("_artifact_id", "?")
        arm = rec.get("arm") or rec.get("_arm", "?")
        elapsed_ms = rec.get("elapsed_ms")
        elapsed_s = f"{float(elapsed_ms) / 1000:.1f}" if elapsed_ms is not None else "—"
        inp = str(rec.get("input_tokens") or "—")
        out = str(rec.get("output_tokens") or "—")
        findings_total = sum((rec.get("finding_histogram") or {}).values())
        not_proc = str(rec.get("units_not_processed") or 0)
        status = rec.get("status", "—")
        if not rec.get("measurement_valid", True):
            status = f"{status} (invalid)"
        cost = _cost_cell(rec)
        lines.append(
            f"| {artifact_id} | {arm} | {elapsed_s} | {inp} | {out} "
            f"| {findings_total} | {not_proc} | {status} | {cost} |"
        )
    lines.append("")
    return "\n".join(lines)


# ── entry point ────────────────────────────────────────────────────────────────


def main(
    argv: list[str] | None = None,
    output_stream: Any = None,
) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Summarise evaluation results as Markdown."
    )
    parser.add_argument("dirs", nargs="+", metavar="DIR", help="Result directories")
    args = parser.parse_args(argv)

    out = output_stream if output_stream is not None else sys.stdout

    result_dirs = [Path(d) for d in args.dirs]
    missing = [str(d) for d in result_dirs if not d.exists()]
    if missing:
        print(f"ERROR: directories not found: {', '.join(missing)}", file=sys.stderr)
        return 1

    stratum_map = _load_stratum_map()
    records = _load_result_dirs(result_dirs)
    if not records:
        print("# Evaluation summary\n\n_No results found._\n", file=out)
        return 0

    # Aggregate: (arm, stratum) -> bucket
    buckets: dict[tuple[str, str], _ArmStratumBucket] = defaultdict(_ArmStratumBucket)
    for rec in records:
        artifact_id = rec.get("artifact_id") or rec.get("_artifact_id", "")
        arm = str(rec.get("arm") or rec.get("_arm", "unknown"))
        stratum = stratum_map.get(artifact_id, "unknown")
        buckets[(arm, stratum)].add(rec)

    print("# Evaluation summary\n", file=out)
    print(f"Result directories: {', '.join(str(d) for d in result_dirs)}\n", file=out)
    print(f"Total arm records: {len(records)}\n", file=out)

    print("## Per-arm / per-stratum aggregates\n", file=out)
    for (arm, stratum), bucket in sorted(buckets.items()):
        print(_render_bucket(arm, stratum, bucket), file=out)

    print("## Per-artifact detail\n", file=out)
    print(_render_detail_table(records), file=out)

    return 0


if __name__ == "__main__":
    sys.exit(main())
