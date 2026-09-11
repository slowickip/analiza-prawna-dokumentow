# Contract Analyzer

Contract Analyzer reads a Polish-language contract and points out the clauses a reader without legal
training would want to look at, together with the provisions of Polish law each one relates to. It is
a research prototype, built for an engineering thesis. It informs; it does not advise, and it does not
say whether to sign. Every legal conclusion it prints has to be checked against the source it cites.

A reader uploads a contract — an employment agreement, a services contract, consumer credit, a company
agreement — and gets back a list of findings. Each finding names the clause it read, states what it
takes the issue to be, and cites the provisions it retrieved from a local corpus of Polish statutes
built from the Sejm's official ELI service. Where the system cannot settle a question it says so
instead of guessing, and where it never processed part of the document it says that too.

![The interface before analysis: the uploaded synthetic contract on the left, variant selection and processing notice on the right](docs/interface.png)

## The three analysis variants

The same document can be handed to the model in three ways. The interface offers all three, and the
evaluation compares them:

| Variant | What one model call sees | Interface label |
| --- | --- | --- |
| `OFF` | the whole document as a single unit | *OFF — cały dokument* |
| `MID` | one structural unit (a section or paragraph) at a time | *MID — jednostki* |
| `ON` | one unit at a time, plus the text of the units its internal cross-references point to | *ON — jednostki + odwołania* |

The variants share the model, the prompts, the corpus, the agent roles, the tools and the per-call
budgets. Only the slice of the document a call receives changes, which also changes how many calls a
document takes and what the retrieval queries look like. MID is the default in the interface, and it
is the variant with the highest coverage in both evaluation series; the table under
[Results](#results) gives the figures.

## Requirements

- Docker and Docker Compose.
- `MODEL_API_KEY` — a key for an OpenAI-compatible provider serving the analysis model over the
  `/responses` route. It defaults to OpenRouter and `meta/muse-spark-1.3-contributor`.
- `OPENROUTER_API_KEY` — a key for the embedding service, which is OpenRouter serving
  `qwen/qwen3-embedding-8b` at 4096 dimensions. One OpenRouter key can serve both variables.

The two credentials stay separate because they are two endpoints. The analysis endpoint is meant to be
repointed: `MODEL_BASE_URL` and `MODEL_NAME` accept any OpenAI-compatible service. The embedding
endpoint is not a dial. The corpus is stored as vectors from one encoder, and the backend refuses a
corpus whose vector space disagrees with the one it encodes queries in, so changing
`EMBEDDINGS_BASE_URL` means rebuilding the corpus against the new endpoint — and comparing such a run
with the recorded results is no longer meaningful.

## Quick start

```bash
export MODEL_API_KEY="your-model-provider-key"
export OPENROUTER_API_KEY="your-openrouter-key"
docker compose up --build
```

Compose resolves these while it reads its configuration, so export them before any `docker compose`
command, or put them in an `.env` file in this directory.

The first start builds the legal corpus before the backend accepts anything: the seeder downloads the
eleven acts named in `corpus/manifest.example.json`, cuts them into articles and embeds them. That
took between three and six minutes on the machine this was last checked on, and it is a one-off —
later starts find the corpus already published and skip straight past it.

The interface is then at **http://localhost:8000**. The HTTP API is at `http://127.0.0.1:8001`, and
`openapi.yaml` describes it.

## Analysing a document

1. Open http://localhost:8000.
2. Pick a variant. MID is preselected.
3. Tick the box confirming you understand what leaves the machine: OCR runs locally, retrieval sends
   model-written search phrases, which may contain document excerpts, to the embedding provider,
   and the selected document text goes to the analysis model provider.
4. Upload a contract: `.txt`, `.docx`, `.doc` or `.pdf`, up to 25 MB.
5. Start the analysis and read the findings. Selecting one highlights the clause it came from and
   shows the provisions it cites.
6. Use the chat to ask about a finding or to contest it, and the agent worksheet to see what each
   role did with each unit.

Uploaded content is held for an hour and then expires; the findings of a finished run stay readable
afterwards, but the document text behind them does not.

`evaluation-data/documents/` holds three short synthetic contracts that make good first uploads.

## The legal corpus

The seeder builds the corpus from a manifest of acts, each named by its ELI identifier and the legal
status date it is to be read at. It fetches the text from `api.sejm.gov.pl`, splits it into articles,
embeds them and publishes the collection under an alias the backend reads.

- `corpus/manifest.example.json` — the eleven acts the evaluation used, including the Civil Code, the
  Labour Code and the Consumer Rights Act. 3,317 units.
- `corpus/manifest.smoke.json` — two acts, for a faster build when the point is to see the system run.

A build is identified by what is in it, not by when it ran: the same manifest over unchanged sources
produces the same snapshot identifier. `CORPUS_MANIFEST` selects the manifest, and
`/api/v1/config` reports the snapshot the backend is serving.

```bash
CORPUS_MANIFEST=/app/corpus/manifest.smoke.json docker compose up --build
```

## Configuration

| Variable | Default | Description |
| --- | --- | --- |
| `MODEL_API_KEY` | — (required) | Model provider API key |
| `MODEL_BASE_URL` | `https://openrouter.ai/api/v1` | Endpoint serving the OpenAI `/responses` route |
| `MODEL_NAME` | `meta/muse-spark-1.3-contributor` | Model identifier |
| `MODEL_EXTRA_HEADERS` | _(unset)_ | JSON object of headers a provider requires beyond the OpenAI interface; the default endpoint needs none |
| `EMBEDDINGS_BASE_URL` | `https://openrouter.ai/api/v1` | Embedding `/v1/embeddings` endpoint |
| `OPENROUTER_API_KEY` | _(unset)_ | Bearer credential, sent only to the router's own host |
| `EMBEDDINGS_PROVIDER` | `deepinfra` | Upstream the router must use, with fallback refused; empty restores its load balancing |
| `CORPUS_MANIFEST` | `/app/corpus/manifest.example.json` | Legal corpus manifest to seed |
| `CORPUS_SNAPSHOT_ID` | _(unset)_ | Pins the corpus a measured run may serve |
| `EVALUATION_BATCH_OPEN` | `false` | Opens the installation for one measured batch and closes the interactive endpoints |
| `MODEL_DEV_API_KEY` | — (required with `compose.dev.yaml`) | Credential for the development endpoint |
| `MODEL_DEV_BASE_URL` | `https://opencode.ai/zen/go/v1` | Development endpoint, used only with `compose.dev.yaml` |
| `MODEL_DEV_NAME` | `muse-spark-1.3-contributor` | Model identifier at that endpoint |
| `MODEL_DEV_EXTRA_HEADERS` | — (required with `compose.dev.yaml`) | Headers that endpoint requires; the gateway above refuses a request without `x-opencode-session`, and `{}` declares an endpoint that needs none |

## Development and measured configurations

`compose.yaml` alone is the measured configuration: the analysis model on the endpoint the recorded
results were produced against. Exploratory work does not need that endpoint and should not pay for it,
so `compose.dev.yaml` overrides the model connector with a second one:

```bash
docker compose -f compose.yaml -f compose.dev.yaml up --build
```

Keep both credentials set — `MODEL_API_KEY` for the measured endpoint, `MODEL_DEV_API_KEY` for the
development one — and the endpoint is then chosen by which files are passed to Compose and by nothing
else. `MODEL_API_KEY` stays required either way, because `compose.yaml` resolves its own variables
before the two files merge.

Bare `docker compose up` starts the application with the analysis provider and corpus manifest used
for the recorded experiment. To recompute the published statistics from saved assessments, run
`uv run python scripts/evaluate.py report --offline`. New model calls produce new responses.
A deployment that is only ever developed on can flip its own default
without changing what a copy does, by setting `COMPOSE_FILE` in the `.env` Compose reads from this
directory:

```
COMPOSE_FILE=compose.yaml:compose.dev.yaml
```

A measured run then names its files explicitly, `docker compose -f compose.yaml`.

The override touches the analysis model only. The embedding service keeps its endpoint, for the reason
given under [Requirements](#requirements). It also pins `EVALUATION_BATCH_OPEN` false, so a stack
started this way cannot serve a measured batch. The two endpoints do not serve the same model
identically, so runs made on the development one are not comparable with the recorded results and are
not measurement material.

## Data volumes

Two named volumes hold state: `postgres-data` for run metadata and `qdrant-data` for the corpus. The
corpus is not source data — the seeder rebuilds it from the manifest — so wiping `qdrant-data` costs
the rebuild time and nothing else. The relational database is a different matter, so the reseed below
names one volume: `docker compose down --volumes` would take the run metadata with it.

The vector store is pinned to `qdrant/qdrant:v1.19.1` by image digest. Qdrant migrates its storage one
minor version at a time, so a volume written by a server older than 1.19 and then started directly
under 1.19 loads with most of its points missing — the seeder still sees a published snapshot, and the
backend then rejects the short point count. The supported upgrade path is to remove that volume and
reseed:

```bash
docker compose down
docker volume ls --filter name=_qdrant-data
docker volume rm artifact_qdrant-data
```

The listing comes first because the volume name carries the Compose project name as a prefix, and the
removal takes one exact name. Started from a directory named `artifact` the project is `artifact`, so
the volume is `artifact_qdrant-data`; under another project name, remove the `<prefix>_qdrant-data`
that the listing shows for this deployment and leave any other one alone. If the listing prints
nothing, there is no old volume. After reseeding, check `/api/v1/health/ready` and one search before
relying on the deployment.

## Tests and checks

```bash
uv sync --locked --all-groups
npm ci --prefix web

uv run ruff check backend seeder tests scripts
uv run mypy
uv run pytest --ignore=tests/smoke -q
npm test --prefix web
```

`tests/smoke` is excluded above: one of its two files drives Docker itself, and the other spends
provider tokens and runs only with `LIVE_PROVIDER_SMOKE=1`.

## Reproducing the evaluation

`evaluation-data/` holds the material and the records: the five documents, the answer key of 78
expectations, the measurement protocol, the raw runs of both series and every saved judge response.
`evaluation-data/README.md` says what each file owns; the commands below are the ones worth running
first.

```bash
# Recompute every quality figure from the saved judgements. No network, no model call.
uv run python scripts/evaluate.py report --offline

# Check the dataset against its manifest: hashes, splits, ingestion measurements
uv run python scripts/validate_evaluation_data.py
```

The offline report re-reads the inputs and the saved assessments and rewrites
`evaluation-data/results/judge/report.json`. An incomplete set produces an incomplete report and a
nonzero exit.

Quality assessment itself uses GPT-5.6 Sol as a judge, in an environment separate from the
application: it reads the saved analyzer findings, the answer key and the source texts, and it
authenticates through a ChatGPT subscription rather than through the model provider the application
uses. `uv run python scripts/evaluate.py run --workers 4` assesses whole runs that are still missing;
each attempt is saved as it happens, so an interrupted command resumes. This is the only scoring path
in the artefact. `evaluation-data/README.md` covers login, source preparation and the judge control.

Producing new analyzer runs needs a running backend and an open batch gate:

```bash
EVALUATION_BATCH_OPEN=true docker compose up -d backend
uv run python scripts/run_evaluation.py --split development --arms off,mid,on \
  --out evaluation-data/results/local/
```

`EVALUATION_BATCH_OPEN` defaults to `false` for interactive use, and the runner refuses to start
unless the backend reports it `true`. `evaluation-data/results/local/` is ignored scratch output;
development evidence worth keeping goes into a dated directory under
`evaluation-data/results/development/`.

`scripts/check_runs.py` checks whether the compared runs used matching configurations.
It reads the run metadata store, so it needs `DATABASE_URL`, and it
opens that store read-only — opening it for writing would reconcile interrupted runs and settle
everything found running, which a diagnostic must never do to a batch in flight.

```bash
# Check matching configurations within and between the two series
uv run python scripts/check_runs.py \
  --batch evaluation-data/results/final/2026-09-08-holdout \
  --batch evaluation-data/results/final/2026-09-09-holdout-series-2
```

Without `--batch` the check reads every completed run in the store, which is ambiguous once the store
has served more than one series of the same comparison.

The two complete series cover the same three documents and are numbered 1 and 2 in the protocols
and reports.

## Results

Two series, 18 runs, 410 findings. GPT-5.6 Sol assessed the 301 substantive findings against the key
and the source texts; the other 109 are answers that state no substantive finding — an uncertainty,
no relation to the corpus, no legal basis, or a fragment the run did not process — and are counted
separately. The judge control passed with 10 of 12 overall agreements with model-prepared references.

| Series | Variant | Coverage (%) | Precision (%) | Confirmed (%) | Analysis time (s) | Analysis tokens |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | OFF | 14.2 | 66.7 | 60.0 | 1,272 | 2,199,674 |
| 1 | MID | 52.4 | 72.7 | 71.6 | 1,002 | 4,265,721 |
| 1 | ON | 47.5 | 67.8 | 66.7 | 974 | 4,682,493 |
| 2 | OFF | 18.6 | 75.0 | 71.4 | 1,480 | 2,209,532 |
| 2 | MID | 46.6 | 67.1 | 66.2 | 1,107 | 4,114,756 |
| 2 | ON | 35.1 | 57.4 | 56.5 | 1,005 | 4,967,276 |

Coverage is the equal-document mean against the 78 key expectations. Precision pools resolved
findings; the confirmed share also counts unresolved findings in its denominator. Times and tokens are
sums over the three documents. Per-run monetary charges were not recorded.

MID has the highest coverage in both series. ON, which adds cross-referenced context, has lower
coverage and precision than MID and spends more tokens. Even MID covers only about half of the
expected issues. Exact fractions, per-document outcomes, comparisons and judging resources are in
[`report.json`](evaluation-data/results/judge/report.json).

## Limits

- The output is informational. It is not legal advice, not an audit, and not a statement that a
  contract is safe to sign. A finding is a reason to read the cited provision, not a conclusion.
- The assessments above are source-based model judgements, not a lawyer's opinion.
- Coverage is measured against a key of 78 expectations over the three documents that were measured.
  It says nothing about Polish contracts in general.
- The corpus is a frozen copy of eleven acts read at a stated date. Amendments after that date are not
  in it, and the system reports how many known amendments a given act does not carry.
- Findings and citations are generated by a language model and can be wrong in both directions:
  missed clauses and invented or misapplied provisions.

## Licence and re-use

- Code: MIT, see [`LICENSE`](LICENSE).
- The synthetic evaluation documents and the project's own evaluation records: CC0 1.0, see
  [`evaluation-data/LICENSE.md`](evaluation-data/LICENSE.md).
- The two contract templates under `evaluation-data/source-files/` come from the Polish Ministry of
  Development and Technology and are redistributed under its public-sector-information re-use
  conditions. Keep the source, author/editor credits, timestamps and processing notice in
  [`evaluation-data/LICENSE.md`](evaluation-data/LICENSE.md) with copies of these files.
  `evaluation-data/source-files/acquisition.json` records their retrieval and metadata redaction.
- The runtime search corpus is rebuilt from the publisher's API. The statutory texts needed for
  offline evaluation are included in `evaluation-data/judge/sources/provisions.json`, with their
  official ELI references. See [`evaluation-data/LICENSE.md`](evaluation-data/LICENSE.md).
