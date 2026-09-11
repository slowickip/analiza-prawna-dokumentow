# Evaluation data

How to operate this directory. `manifest.json` owns the pool and its measurements;
`protocol.json` owns the recorded measurement parameters;
`scoring-protocol.json` owns how the recorded runs are scored against the answer key. This file
does not restate any of them.

There is one scoring path: `scripts/evaluate.py`, judging whole recorded runs, with a separate verdict for every substantive finding, against the answer
key and the source texts under `scoring-protocol.json`. No second scoring mechanism, assignment
file or parallel quality table is kept beside it.

## What is where

| Path | Owns |
| --- | --- |
| `manifest.json` | Every per-artifact fact: id, stratum, logical case, canonical flag, path, hashes, split, rights, privacy status and the measured ingestion figures |
| `source-files/acquisition.json` | Source URL, acquisition time, and two hashes per retrieved file: `acquired_*` for what the publisher served, `stored_*` for the redacted derivative held here |
| `protocol.json` | Measurement parameters, dataset split, run conditions and the retained measurement records |
| `scoring-protocol.json` | Judge configuration, assessment criteria, quality dimensions, denominators, comparison thresholds and readiness |
| `answer-key.json` | Expected issues, exact document spans, complete code-and-basis alternatives and source evidence |
| `documents/` | The synthetic stratum, written for this project |
| `source-files/` | The real stratum, as metadata-redacted derivatives |
| `results/` | Raw development and final run outputs; the directory name records provenance, not whether a run is an evaluation result |
| `judge/sources/` | Canonical document and provision texts, with content hashes and the verified corpus identity |
| `judge/control/` | Development and control cases, reference ratings and fixed acceptance criteria |
| `results/judge/calibration-verdicts.jsonl` | Saved development and control responses |
| `results/parity.json` | Summary of configuration comparisons for both analyzer series |
| `results/judge/control-report.json` | Offline comparison of control responses with the reference ratings |
| `results/judge/verdicts.jsonl` | Saved attempts and validated assessments of the recorded analyzer findings |
| `results/judge/report.json` | Quality fractions, per-finding outcomes, comparisons, error counts, resources and input hashes |
| `LICENSE.md` | The CC0 dedication for the synthetic files and this project's own metadata, and the attribution the two publisher-retrieved files are carried under |

`manifest.json` is the machine owner of everything it records. Do not copy a fact out of it into
prose; cite it.

The two retained series are numbered 1 and 2 in `scoring-protocol.json`. Their
`batch` fields identify the unchanged input directories.

## Current state

The two registered series contain 18 runs and 410 findings. The approved key is frozen.
The judge receives 18 whole-run packets covering the 301 substantive findings; the other
109 findings are counted separately. Each packet contains the full document and source texts
once, with a separate identified verdict required for every substantive finding.

Source-based judge control and full assessment are complete. The control passed with 10/12
overall agreements with model-prepared references, no key-match disagreements and no false
reassurances. The saved quality report accounts for all 410 findings, including 301 validated
assessments in 18 whole-run packets. One rejected quoting attempt is retained alongside the
valid responses and included in resource accounting. Run the offline commands below to
reproduce the control and quality reports.

## Split discipline

The split is frozen per logical case, so every format variant of a case carries its case's split.
`scripts/validate_evaluation_data.py` holds the frozen table and fails on any drift between it and
the manifest — the table is duplicated on purpose, because a guard reading the split only from the
file it checks would go green on any edit to that file.

Holdout material stays untouched until the final run, with one recorded exception: ED-001 was used as a benchmark on 2026-09-04 before the final measurement, and that exposure is recorded in `protocol.json` (`holdout_discipline.ed_001_exposure`) with the raw run records under `results/development/2026-09-04-ed-001-holdout-exposure/`. What that forbids, and what development
material permits, is listed in `protocol.json` under `holdout_discipline`.

## Evaluation isolation

Only the content of the selected document may be supplied to an evaluated arm. This README,
`manifest.json`, `acquisition.json`, `protocol.json`, `scoring-protocol.json`, filenames and type
labels are research metadata and leak construction choices. The same bytes must reach every arm.
The answer key is scoring material for the same reason and is never an input: putting an expected
finding in front of an arm would measure copying, not analysis.

## Reproducibility checks

Run all commands below from the artefact root. Install the locked environment
with `uv sync --locked --all-groups`.

### Reproduce saved figures without a model

```bash
uv run python scripts/check_judge.py \
  --cases evaluation-data/judge/control/cases.json \
  --reference evaluation-data/judge/control/reference-ratings.json \
  --criteria evaluation-data/judge/control/acceptance.json \
  --verdicts evaluation-data/results/judge/calibration-verdicts.jsonl \
  --protocol evaluation-data/scoring-protocol.json \
  --output evaluation-data/results/judge/control-report.json

uv run python scripts/evaluate.py report --offline
```

These commands read the saved responses and sources. They require no running
backend, model connection or credentials. The report verifies the input hashes,
revalidates the responses and records a status for every analyzer finding. An
incomplete set of judgments produces an incomplete report and a nonzero exit.
Identical inputs produce identical JSON. Each quality fraction retains its
numerator and denominator; an undefined rate is `null`.

### Reproduce the analyzer parity check

The raw response files carry each run's configuration. The check below reads them
through the same hash-verifying loader as scoring, compares every registered pair
within each series and every corresponding cell between series, and checks the
counts against `results/parity.json`. It needs no database or model connection.

```bash
PYTHONPATH=scripts uv run python - <<'PYTHON'
import json
from pathlib import Path
from types import SimpleNamespace
from contract_analyzer.agents.parity import compare_arm_parity
from evaluation_inputs import ARMS, load_json, load_runs

root = Path('evaluation-data')
protocol = load_json(root / 'scoring-protocol.json')
runs = load_runs(root, protocol).cells
series = sorted({cell[0] for cell in runs})
documents = protocol['scored_runs']['cohort']
assert len(runs) == len(series) * protocol['scored_runs']['cells_per_series']
assert len({run['id'] for run in runs.values()}) == len(runs)
assert all(run['parent_run_id'] is None for run in runs.values())
summary = {kind: {'checked': 0, 'failed': 0}
           for kind in ('within_series', 'between_series')}

def check(kind, left, right):
    result = compare_arm_parity([SimpleNamespace(**runs[left]), SimpleNamespace(**runs[right])])
    summary[kind]['checked'] += 1
    summary[kind]['failed'] += int(not result.measurement_valid)

comparisons = load_json(root / 'protocol.json')['registered_comparisons']
for number in series:
    for document in documents:
        for comparison in comparisons:
            left, right = [arm.lower() for arm in comparison['pair']]
            check('within_series', (number, document, left), (number, document, right))
for number in series[1:]:
    for document in documents:
        for arm in ARMS:
            check('between_series', (series[0], document, arm), (number, document, arm))
assert summary == load_json(root / 'results/parity.json')['summary']
assert all(value['failed'] == 0 for value in summary.values())
print(json.dumps(summary, indent=2))
PYTHON
```

### Perform the model assessments

The judge is GPT-5.6 Sol at `high`, accessed through the pinned `openai-codex` SDK
and a ChatGPT subscription. Sign in through the external Codex CLI with
`codex login`; `codex login status` checks the login. The default credentials
directory is `~/.codex`. Pass `--credentials-home` when using another directory.
The judge requires ChatGPT authentication and does not fall back to an API key.

```bash
# Check packet preparation without making model calls
uv run python scripts/evaluate.py packets

# Assess the fixed control cases (then reproduce and inspect the control report above)
uv run python scripts/evaluate.py calibrate \
  --cases evaluation-data/judge/control/cases.json --workers 4 --credentials-home ~/.codex

# Assess missing whole runs after the control passes and the readiness gate opens
uv run python scripts/evaluate.py run --workers 4 --credentials-home ~/.codex

# Recompute figures from the saved assessments
uv run python scripts/evaluate.py report --offline
```

`run` requires the reviewed key and a ready scoring protocol. It runs at most four independent assessments concurrently, with an isolated runtime for each worker
and a fresh conversation for each run. It saves each whole-run attempt immediately. After a technical interruption, use the same command to continue
with the same model, inputs and configuration. A complete valid run assessment is reused, regardless of its conclusions. Missing, duplicate or
foreign finding identifiers invalidate the response. Judge time and token use are reported separately
from the analyzer's resource use and counted once per run attempt.

The prepared sources are included. To rebuild them from the declared corpus:

```bash
uv run python scripts/evaluate.py prepare --corpus-manifest corpus/manifest.example.json
```

Preparation requires access to the declared legal sources and checks their corpus
identity against the recorded runs. The judge receives these texts in its input;
it does not fetch additional sources. Preparation or scoring never reruns the
analyzer.

### Check dataset integrity

Run from the artefact root:

```bash
uv run python scripts/validate_evaluation_data.py
```

```bash
uv run python scripts/measure_evaluation_ingestion.py --tokenizer <path-to>/tokenizer.json --check
```

Use `uv run python` (or `.venv/bin/python`), not a bare `python`. On a machine whose `python` is a
different interpreter the second command dies with `ModuleNotFoundError: No module named
'tokenizers'` — an environment failure that reads like a data failure.

The first is standard-library only. It checks manifest and disk agreement, ids, hashes and counts
for the synthetic stratum, acquisition agreement for the real stratum, closed rights and privacy
enums, the frozen split and canonical artifact per case, and that the descriptive length block is
present, numeric and internally consistent. It **cannot** check whether a `rights_status` is true;
that rests on the recorded evidence text beside it, which is why the evidence is quoted rather than
summarised.

The second re-runs the artefact's own ingestion, segmentation and reference parser over every
artifact and fails if any recorded measurement drifted **or could not be taken on that machine**, so
a green run means all five were checked. It needs the runtime dependencies; the two `.doc` files
need LibreOffice to convert, which the project's container image carries. No retained artifact is a
scan, so no OCR runs. No model call is made.

## Rights and redaction

The two ministry templates are redistributed under the publisher's re-use conditions. `LICENSE.md`
carries the source, author/editor credits, file timestamps and processing notice required with
copies of the files. `manifest.json` records the verified rights status, and `source-files/acquisition.json`
records the original metadata and hash checks.

The documents passed to the system have their person-named metadata blanked. Source attribution is
retained separately in the licence notice and acquisition record.

## Limits

A purposive coverage basket, not a prevalence sample, and not evidence about all Polish contracts.
The candidates are research fixtures, **not contract templates and not legal advice**.
