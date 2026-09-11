#!/usr/bin/env python3
"""Read the recorded runs, the key and the source bundle into judging packets.

Everything on this path is offline and deterministic. It verifies the hashes the
scoring protocol records, accounts for every recorded finding exactly once, and
builds one blinded packet per substantive finding. It makes no model call, and it
never fetches a source: a packet is built from prepared source material or it is
not built at all.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ARMS = ("off", "mid", "on")
SUBSTANTIVE_CODES = ("consistent", "contradictory", "permissible_departure")
NON_SUBSTANTIVE_CODES = (
    "no_relation",
    "no_basis_found",
    "basis_not_in_force",
    "unit_not_adjudicable",
    "not_processed",
    "uncertain",
)
LOCATOR_RE = re.compile(
    r"https://api\.sejm\.gov\.pl/eli/acts/([^/]+/[^/]+/[^/]+)/text\.pdf"
    r"#article=([0-9]+(?:[a-z]|\([0-9]+\))?)(?:&printing=([0-9]+))?"
)
# Every ELI locator the corpus can mint, article-level or not: the HTML reader
# names a unit by its path under the act, so a corpus act read as HTML carries a
# locator this pattern matches and the article pattern above does not.
ACT_LOCATOR_RE = re.compile(
    r"https://api\.sejm\.gov\.pl/eli/acts/([^/]+/[^/]+/[^/]+)(/.*)?"
)

DOCUMENTS_BUNDLE = "documents.json"
PROVISIONS_BUNDLE = "provisions.json"


class EvaluationInputError(ValueError):
    """An input the judging path requires is missing, stale or malformed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise EvaluationInputError(message)


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationInputError(
            f"cannot read valid JSON from {path}: {exc}"
        ) from exc
    require(isinstance(value, dict), f"{path} must contain a JSON object")
    return value


def sha256_file(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise EvaluationInputError(f"cannot hash {path}: {exc}") from exc


def canonical_json(value: Any) -> str:
    """The one serialization every hash in this pipeline is taken over."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def provision_id(act: str, article: str, printing: str | None = None) -> str:
    """One provision's identity, keeping apart what the corpus keeps apart.

    An act that prints the same article number twice mints two units, told apart
    by ``printing``. Dropping it would let one text answer for the other, so the
    number is part of the identity here exactly as it is in the locator.
    """
    suffix = f"&printing={printing}" if printing else ""
    return f"{act}#article={article}{suffix}"


@dataclass(frozen=True)
class Locator:
    """What a claimed or recorded locator names, without deciding whether it exists.

    ``key`` is the identity a provision is stored and looked up under: the
    article form for the article locators the runs recorded and the PDF reader
    mints, and the locator itself for anything else the corpus can mint, so an
    act read as HTML is not silently reshaped into an article reference it never
    carried. ``act`` is empty only when the string is not an ELI act locator at
    all, which is a claim no source can support.
    """

    raw: str
    key: str
    act: str
    article: str | None

    @property
    def malformed(self) -> bool:
        return not self.act


def read_locator(value: str) -> Locator:
    if not isinstance(value, str) or not value:
        return Locator(raw=str(value), key="", act="", article=None)
    article_match = LOCATOR_RE.fullmatch(value)
    if article_match is not None:
        act, article, printing = article_match.groups()
        return Locator(
            raw=value,
            key=provision_id(act, article, printing),
            act=act,
            article=article,
        )
    act_match = ACT_LOCATOR_RE.fullmatch(value)
    if act_match is not None:
        return Locator(raw=value, key=value, act=act_match.group(1), article=None)
    return Locator(raw=value, key="", act="", article=None)


# --------------------------------------------------------------------------
# recorded runs
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RunFinding:
    """One recorded finding, with the run it belongs to kept beside the prompt."""

    series: int
    document_key: str
    arm: str
    finding_id: str
    code: str
    start: int | None
    end: int | None
    anchor_resolved: bool
    quote_resolution: str | None
    locators: tuple[str, ...]
    raw: dict[str, Any]

    @property
    def substantive(self) -> bool:
        return self.code in SUBSTANTIVE_CODES

    @property
    def cell(self) -> tuple[int, str, str]:
        return (self.series, self.document_key, self.arm)

    @property
    def character_kind(self) -> str | None:
        basis = self.raw.get("basis") or {}
        return basis.get("character_kind") if isinstance(basis, dict) else None

    @property
    def duplicate_key(self) -> tuple[Any, ...]:
        """What makes two emissions of one run the same emission.

        Exact identity, over every field the judged claim is made of: the code,
        the anchored characters, whether that anchor and its quote resolved, the
        provision character asserted and the set of claimed locators. Two
        emissions differing anywhere here are making different claims, and one
        of them would disappear if they were merged.
        """
        return (
            self.cell,
            self.code,
            self.start,
            self.end,
            self.anchor_resolved,
            self.quote_resolution,
            self.character_kind,
            tuple(sorted(self.locators)),
        )


@dataclass(frozen=True)
class RecordedRuns:
    """The declared series, verified against the hashes the protocol records."""

    findings: tuple[RunFinding, ...]
    cells: dict[tuple[int, str, str], dict[str, Any]]
    documents: dict[str, dict[str, Any]]
    corpus_snapshot_ids: tuple[str, ...]

    def cell_metrics(self, series: int, document_key: str, arm: str) -> dict[str, Any]:
        run = self.cells[(series, document_key, arm)]
        metrics = run.get("metrics") or {}
        return {
            "input_tokens": metrics.get("input_tokens"),
            "output_tokens": metrics.get("output_tokens"),
            "elapsed_ms": metrics.get("elapsed_ms"),
        }


def _batch_dir(data_dir: Path, batch_value: str) -> Path:
    require(bool(batch_value), "scored_runs.batch must be a nonempty path")
    parts = Path(batch_value).parts
    if parts and parts[0] == "evaluation-data":
        parts = parts[1:]
    require(".." not in parts, "scored_runs.batch may not escape evaluation-data")
    batch = data_dir.joinpath(*parts).resolve()
    require(
        batch.is_relative_to(data_dir.resolve()),
        "scored_runs.batch escapes evaluation-data",
    )
    return batch


def _finding_of(
    series: int, document_key: str, arm: str, raw: dict[str, Any]
) -> RunFinding:
    finding_id = raw.get("id")
    require(
        isinstance(finding_id, str) and bool(finding_id),
        f"a finding of {document_key}/{arm} in series {series} has no id",
    )
    code = raw.get("code")
    require(
        code in SUBSTANTIVE_CODES or code in NON_SUBSTANTIVE_CODES,
        f"finding {finding_id} carries unknown code {code!r}",
    )
    anchor = raw.get("anchor") or {}
    start = anchor.get("start_offset")
    end = anchor.get("end_offset")
    locators = [
        value
        for value in (raw.get("legal_locators") or [])
        if isinstance(value, str) and value
    ]
    basis = raw.get("basis") or {}
    if (
        isinstance(basis, dict)
        and isinstance(basis.get("provision_locator"), str)
        and basis["provision_locator"]
    ):
        locators.append(basis["provision_locator"])
    return RunFinding(
        series=series,
        document_key=document_key,
        arm=arm,
        finding_id=finding_id,
        code=str(code),
        start=start if isinstance(start, int) else None,
        end=end if isinstance(end, int) else None,
        anchor_resolved=raw.get("anchor_resolved") is True,
        quote_resolution=raw.get("quote_resolution"),
        locators=tuple(sorted(set(locators))),
        raw=raw,
    )


def load_runs(data_dir: Path, protocol: dict[str, Any]) -> RecordedRuns:
    """Every recorded series of the protocol, hash-verified, as flat findings."""
    scored = protocol.get("scored_runs")
    require(isinstance(scored, dict), "scoring protocol has no scored_runs section")
    cohort = tuple(scored.get("cohort") or ())
    require(bool(cohort), "scored_runs.cohort is empty")
    layouts = scored.get("documents")
    require(isinstance(layouts, dict), "scored_runs.documents must be an object")
    content_hashes = scored.get("document_content_sha256")
    require(
        isinstance(content_hashes, dict),
        "scored_runs.document_content_sha256 must be an object",
    )
    manifest_path = data_dir / "manifest.json"
    require(
        sha256_file(manifest_path) == scored.get("manifest_sha256"),
        "sha256 mismatch for manifest.json",
    )

    findings: list[RunFinding] = []
    cells: dict[tuple[int, str, str], dict[str, Any]] = {}
    seen_ids: set[str] = set()
    snapshot_ids: set[str] = set()
    known_states = set(scored.get("series_states") or ())
    require(
        bool(known_states),
        "scored_runs.series_states must declare the states a series may carry",
    )

    for series_record in scored.get("series", []):
        state = series_record.get("state")
        require(
            state in known_states,
            f"series {series_record.get('series')} carries state {state!r}, which "
            f"the protocol does not define. An undefined state would drop a series "
            f"that ran without saying so.",
        )
        if state != "recorded":
            continue
        number = series_record["series"]
        batch = _batch_dir(data_dir, str(series_record.get("batch", "")))
        hashes = series_record.get("sha256")
        require(
            isinstance(hashes, dict) and bool(hashes),
            f"series {number} sha256 must list its recorded files",
        )
        for relative, expected in hashes.items():
            path = batch / relative
            require(
                not Path(relative).is_absolute()
                and path.resolve().is_relative_to(batch.resolve()),
                f"series {number} file must stay inside its batch: {relative}",
            )
            require(
                sha256_file(path) == expected,
                f"series {number} sha256 mismatch for {relative}",
            )
        total = 0
        for document_key in cohort:
            layout = layouts.get(document_key)
            require(
                isinstance(layout, dict) and "dir" in layout,
                f"missing scored document metadata for {document_key}",
            )
            directory = layout["dir"]
            metadata_relative = f"{directory}/document.json"
            metadata_path = batch / metadata_relative
            require(
                metadata_relative in hashes,
                f"series {number} has no sha256 for {metadata_relative}",
            )
            metadata = load_json(metadata_path)
            require(
                metadata.get("content_hash") == content_hashes.get(document_key),
                f"document {document_key} differs from the protocol content hash "
                f"in series {number}",
            )
            for arm in ARMS:
                relative = f"{directory}/{arm}.json"
                path = batch / relative
                require(
                    path.is_file(),
                    f"missing run for {document_key}/{arm} in series {number}",
                )
                require(
                    relative in hashes,
                    f"series {number} has no sha256 for {relative}",
                )
                run = load_json(path)
                require(
                    run.get("arm") == arm,
                    f"run arm mismatch for {document_key}/{arm} in series {number}",
                )
                require(
                    run.get("status") == "completed"
                    and run.get("measurement_valid") is True,
                    f"run {document_key}/{arm} of series {number} must be completed "
                    f"and measurement_valid",
                )
                recorded = run.get("findings")
                require(
                    isinstance(recorded, list),
                    f"run {document_key}/{arm} of series {number} findings "
                    f"must be a list",
                )
                cells[(number, document_key, arm)] = run
                snapshot = run.get("corpus_snapshot_id")
                require(
                    isinstance(snapshot, str) and bool(snapshot),
                    f"run {document_key}/{arm} of series {number} records no corpus "
                    f"snapshot id, so nothing binds it to a body of law",
                )
                snapshot_ids.add(str(snapshot))
                for raw in recorded:
                    require(
                        isinstance(raw, dict),
                        f"a finding of {document_key}/{arm} in series {number} "
                        f"is not an object",
                    )
                    finding = _finding_of(number, document_key, arm, raw)
                    require(
                        finding.finding_id not in seen_ids,
                        f"finding id {finding.finding_id} appears twice; ids must be "
                        f"unique across the scored series",
                    )
                    seen_ids.add(finding.finding_id)
                    findings.append(finding)
                total += len(recorded)
        expected = series_record.get("findings_recorded")
        require(
            expected is None or total == expected,
            f"series {number} recorded {total} findings, not the declared {expected}",
        )

    require(bool(findings), "no recorded series to judge")
    documents = {
        key: {
            "artifact_id": layouts[key]["artifact"],
            "content_sha256": content_hashes[key],
        }
        for key in cohort
    }
    findings.sort(key=lambda f: (f.series, f.document_key, f.arm, f.finding_id))
    return RecordedRuns(
        findings=tuple(findings),
        cells=cells,
        documents=documents,
        corpus_snapshot_ids=tuple(sorted(snapshot_ids)),
    )


# --------------------------------------------------------------------------
# prepared source material
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceBundle:
    """Canonical document text and corpus provision text, prepared beforehand.

    Both files are produced by ``evaluate.py prepare`` and nothing else. Judging
    reads them; it never reaches for a source itself, so a packet either carries
    the real declared text or is refused.

    ``digest`` identifies the material, not the preparation: it is taken over the
    documents and the provisions and leaves the build timestamp out, exactly as
    the corpus identity does, so preparing the same sources twice does not throw
    away judgements that were made against them.
    """

    documents: dict[str, dict[str, Any]]
    provisions: dict[str, dict[str, Any]]
    acts: tuple[str, ...]
    corpus_acts: tuple[str, ...]
    corpus: dict[str, Any]
    digest: str
    built_at: str

    @property
    def corpus_verified(self) -> bool:
        """Whether the prepared units are provably the corpus the runs searched."""
        return self.corpus.get("snapshot_verified") is True

    @property
    def snapshot_id(self) -> str:
        return str(self.corpus.get("snapshot_id", ""))

    @classmethod
    def load(cls, directory: Path) -> SourceBundle:
        documents_path = directory / DOCUMENTS_BUNDLE
        provisions_path = directory / PROVISIONS_BUNDLE
        missing = [
            str(path.name)
            for path in (documents_path, provisions_path)
            if not path.is_file()
        ]
        require(
            not missing,
            f"prepared source material is missing from {directory}: "
            f"{', '.join(missing)}. Run `evaluate.py prepare` first; judging never "
            f"fetches a source itself.",
        )
        documents_raw = load_json(documents_path)
        provisions_raw = load_json(provisions_path)
        documents = documents_raw.get("documents")
        provisions = provisions_raw.get("provisions")
        require(
            isinstance(documents, dict) and bool(documents),
            f"{documents_path} carries no document text",
        )
        require(
            isinstance(provisions, dict) and bool(provisions),
            f"{provisions_path} carries no provision text",
        )
        for key, entry in documents.items():
            text = entry.get("text")
            require(
                isinstance(text, str) and bool(text),
                f"document {key} in the bundle has no text",
            )
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            require(
                digest == entry.get("content_sha256"),
                f"document {key} in the bundle does not hash to its recorded "
                f"content_sha256",
            )
        for key, entry in provisions.items():
            text = entry.get("text")
            require(
                isinstance(text, str) and bool(text),
                f"provision {key} in the bundle has no text",
            )
            require(
                hashlib.sha256(text.encode("utf-8")).hexdigest()
                == entry.get("content_sha256"),
                f"provision {key} in the bundle does not hash to its recorded "
                f"content_sha256; the text is not the text it claims to be",
            )
            locator = entry.get("locator")
            require(
                isinstance(locator, str) and read_locator(locator).key == key,
                f"provision {key} in the bundle is stored under a key its locator "
                f"{locator!r} does not produce",
            )
        acts = provisions_raw.get("acts")
        require(
            isinstance(acts, dict) and bool(acts),
            f"{provisions_path} must name the acts whose text it carries",
        )
        corpus = provisions_raw.get("corpus")
        require(
            isinstance(corpus, dict),
            f"{provisions_path} must carry a corpus record saying which body of law "
            f"it prepared and whether that is the one the runs searched",
        )
        corpus_acts = corpus.get("declared_acts")
        require(
            isinstance(corpus_acts, list) and bool(corpus_acts),
            f"{provisions_path} must name every act the frozen corpus declares, so "
            f"that an unprepared act is distinguishable from one the corpus never "
            f"held",
        )
        require(
            set(acts) <= set(corpus_acts),
            f"{provisions_path} carries text for acts the corpus does not declare: "
            f"{', '.join(sorted(set(acts) - set(corpus_acts)))}",
        )
        return cls(
            documents=documents,
            provisions=provisions,
            acts=tuple(sorted(acts)),
            corpus_acts=tuple(sorted(str(act) for act in corpus_acts)),
            corpus=corpus,
            digest=sha256_value({"documents": documents, "provisions": provisions}),
            built_at=str(provisions_raw.get("built_at", "")),
        )

    def bind_to_runs(self, runs: RecordedRuns) -> None:
        """Refuse a bundle that claims a corpus the recorded runs did not search.

        A verified bundle is one whose recomputed snapshot identity equals the one
        every recorded run carries. Preparation does that arithmetic; this repeats
        the comparison against the runs themselves, so a hand-edited claim of
        verification does not survive.
        """
        if not self.corpus_verified:
            return
        recorded = set(runs.corpus_snapshot_ids)
        require(
            recorded == {self.snapshot_id},
            f"the bundle claims corpus snapshot {self.snapshot_id!r} but the runs "
            f"searched {', '.join(sorted(recorded))}. A verified bundle must be the "
            f"body of law the runs were measured against.",
        )

    def document_text(self, document_key: str, content_sha256: str) -> str:
        entry = self.documents.get(document_key)
        require(
            entry is not None,
            f"prepared text for document {document_key} is missing from the bundle",
        )
        assert entry is not None
        require(
            entry.get("content_sha256") == content_sha256,
            f"prepared text for {document_key} is not the canonical text the runs "
            f"were measured on",
        )
        return str(entry["text"])

    def provision(self, locator: str) -> dict[str, Any] | None:
        identity = read_locator(locator)
        if identity.malformed:
            return None
        return self.provisions.get(identity.key)

    def prepared_act(self, locator: str) -> bool:
        return read_locator(locator).act in self.acts

    def corpus_act(self, locator: str) -> bool:
        return read_locator(locator).act in self.corpus_acts


# --------------------------------------------------------------------------
# the answer key
# --------------------------------------------------------------------------


def accepted_bases_of(item: dict[str, Any]) -> set[str]:
    """The locators one key item accepts as a basis, in the corpus that was read.

    One rule, read the same way wherever the accepted bases matter: an item the
    key marks as lying outside the snapshot accepts nothing here, because the
    provision it names was never in the material the runs searched. Requiring its
    text would refuse every packet of that document, and treating it as a
    provision the key asserts exists would excuse a finding claiming the same
    locator from a citation error it has earned.
    """
    if item.get("outside_snapshot") is True:
        return set()
    return {
        basis
        for answer in item.get("accepted_answers") or []
        for basis in answer.get("accepted_bases") or []
    }


@dataclass(frozen=True)
class AnswerKey:
    version: str
    review_state: str
    items: tuple[dict[str, Any], ...]
    documents: dict[str, dict[str, Any]]
    sources: dict[str, dict[str, Any]]
    sha256: str

    @property
    def accepted_bases(self) -> frozenset[str]:
        """Every locator the key accepts as a basis, over the whole key.

        A locator the key accepts is a provision the key asserts exists. If the
        bundle cannot show its text, that is a gap in what was prepared and never
        a citation error of a finding that happened to claim the same provision.

        An item the key marks as outside the snapshot is the exception, and the
        same exception the packet builder makes: the key records there that the
        provision lies outside the corpus the runs could read, so its absence is
        the expected state and asserts nothing about what was prepared.
        """
        return frozenset(
            basis for item in self.items for basis in accepted_bases_of(item)
        )

    def items_for(self, document_key: str) -> tuple[dict[str, Any], ...]:
        return tuple(
            item for item in self.items if item["document_key"] == document_key
        )


def load_key(
    path: Path, runs: RecordedRuns, expected_sha256: str | None = None
) -> AnswerKey:
    """The key, checked against the runs and against the hash the protocol pins.

    The hash binds the expectations and completeness denominator to the saved
    judgements. Adding, removing or changing an expectation requires new
    judgements under the updated key.
    """
    digest = sha256_file(path)
    require(
        expected_sha256 is None or digest == expected_sha256,
        f"{path} hashes to {digest}, not the {expected_sha256} the scoring protocol "
        f"pins. Approve the key by recording its hash, not by editing it after "
        f"judgements exist.",
    )
    raw = load_json(path)
    items = raw.get("items")
    require(isinstance(items, list) and bool(items), f"{path} carries no items")
    documents = raw.get("documents")
    require(isinstance(documents, dict), f"{path} carries no documents map")
    seen: set[str] = set()
    for item in items:
        require(isinstance(item, dict), f"{path} holds a non-object item")
        item_id = item.get("item_id")
        require(
            isinstance(item_id, str) and item_id not in seen,
            f"{path} item ids must be present and unique: {item_id!r}",
        )
        seen.add(item_id)
        require(
            item.get("status") == "scored" and bool(item.get("accepted_answers")),
            f"key item {item_id} must contain a settled expectation",
        )
        document_key = item.get("document_key")
        require(
            document_key in runs.documents,
            f"key item {item_id} names document {document_key!r}, "
            f"which is not in the scored cohort",
        )
    for document_key, recorded in runs.documents.items():
        entry = documents.get(document_key)
        require(
            isinstance(entry, dict)
            and entry.get("content_hash") == recorded["content_sha256"],
            f"the key's document record for {document_key} does not carry the "
            f"content hash of the measured text",
        )
    return AnswerKey(
        version=str(raw.get("key_version", "")),
        review_state=str(raw.get("review_state", "")),
        items=tuple(items),
        documents=documents,
        sources=raw.get("sources") or {},
        sha256=digest,
    )


# --------------------------------------------------------------------------
# accounting and packets
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Accounted:
    """What became of one recorded finding before any judgement was asked for."""

    finding: RunFinding
    disposition: str  # judged, exact_duplicate, non_substantive
    duplicate_of: str | None = None


def account_findings(findings: tuple[RunFinding, ...]) -> tuple[Accounted, ...]:
    """Give every recorded finding exactly one disposition.

    Exact duplicates are collapsed within one run only. The same emission
    recorded by another arm, or by the repeat of the same run, is a separate
    observation: there is no identity rule joining them, and inventing one after
    the outputs exist would silently change a denominator.
    """
    first_seen: dict[tuple[Any, ...], str] = {}
    accounted: list[Accounted] = []
    for finding in findings:
        if not finding.substantive:
            accounted.append(Accounted(finding, "non_substantive"))
            continue
        previous = first_seen.get(finding.duplicate_key)
        if previous is None:
            first_seen[finding.duplicate_key] = finding.finding_id
            accounted.append(Accounted(finding, "judged"))
        else:
            accounted.append(Accounted(finding, "exact_duplicate", previous))
    return tuple(accounted)


@dataclass(frozen=True)
class Packet:
    """One judging case -- a whole recorded run: what the judge sees, and what it
    must never see.

    A packet carries one document, every expectation the key holds for it, the
    real text of every provision any of its findings needs, and all substantive
    findings of that run. The judge answers each of them in one reply, so the
    document and the law are read once rather than once per finding.
    """

    packet_id: str
    series: int
    document_key: str
    arm: str
    finding_ids: tuple[str, ...]
    payload: dict[str, Any]

    @property
    def case(self) -> str:
        return canonical_json(self.payload)


def _quote(text: str, start: int | None, end: int | None) -> str | None:
    if start is None or end is None:
        return None
    if not 0 <= start < end <= len(text):
        return None
    return text[start:end]


def _key_item_view(item: dict[str, Any]) -> dict[str, Any]:
    """A key item as the judge sees it: the expectation, and nothing else.

    Everything that would tell the judge which configuration produced what, or
    how an earlier scoring pass decided, is dropped here: candidate finding ids,
    unit identifiers and preparation metadata.
    """
    return {
        "item_id": item["item_id"],
        "issue": item.get("issue"),
        "quote": item.get("quote"),
        "span": item.get("span"),
        "expected_answers": [
            {
                "code": answer.get("code"),
                "accepted_bases": sorted(answer.get("accepted_bases") or []),
                "justification": answer.get("justification"),
            }
            for answer in item.get("accepted_answers") or []
        ],
        "justification": item.get("justification"),
        "source_references": item.get("source_evidence") or [],
    }


def _provision_views(
    locators: tuple[str, ...], bundle: SourceBundle, where: str
) -> tuple[list[dict[str, Any]], list[str]]:
    views: list[dict[str, Any]] = []
    missing: list[str] = []
    for locator in locators:
        provision = bundle.provision(locator)
        if provision is None:
            missing.append(f"{locator} ({where})")
            continue
        views.append(
            {
                "locator": locator,
                "act_identifier": provision.get("act_identifier"),
                "article_identifier": provision.get("article_identifier"),
                "text": provision["text"],
            }
        )
    return views, missing


def basis_problems(
    finding: RunFinding, bundle: SourceBundle, key: AnswerKey
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split the claimed bases the bundle cannot show into two different things.

    A locator is *uncitable* only when its absence is proved: the prepared units
    are the corpus the runs searched, that corpus declares the act, the act was
    prepared in full, the key does not accept the locator, and the text is still
    not there. Then the provision does not exist in the body of law the analyzer
    could read, which is a fact about the claim, and the packet is built with the
    locator marked.

    A string that is no ELI locator at all is uncitable too, for the same reason
    read the other way: no preparation could ever supply it, and naming a source
    that cannot be resolved is a defect of the citation.

    Everything else is *unprepared*, and the packet is refused: a corpus this
    bundle cannot prove it is, an act nobody prepared, or a locator the key
    itself accepts as a basis -- the key asserting that a provision exists is not
    evidence that it does not.
    """
    uncitable: list[str] = []
    unprepared: list[str] = []
    accepted = key.accepted_bases
    for locator in finding.locators:
        if bundle.provision(locator) is not None:
            continue
        if read_locator(locator).malformed:
            uncitable.append(locator)
            continue
        provable = (
            bundle.corpus_verified
            and locator not in accepted
            # Either the corpus never declared the act, so no preparation could
            # have carried it, or it declared it and this bundle prepared it in
            # full and still does not hold the article.
            and (not bundle.corpus_act(locator) or bundle.prepared_act(locator))
        )
        (uncitable if provable else unprepared).append(locator)
    return tuple(sorted(uncitable)), tuple(sorted(unprepared))


def finding_view(
    finding: RunFinding, document_text: str, uncitable: tuple[str, ...]
) -> dict[str, Any]:
    """One finding as the judge sees it, named by its opaque recorded id.

    The id is the minted uuid of the record. It says nothing about the arm or the
    series, which is what lets the reply identify each answer without telling the
    judge which configuration produced what.
    """
    return {
        "finding_id": finding.finding_id,
        "code": finding.code,
        "span": {"start": finding.start, "end": finding.end},
        "anchor_resolved": finding.anchor_resolved,
        "quoted_text": _quote(document_text, finding.start, finding.end),
        "claimed_bases": sorted(finding.locators),
        "claimed_bases_absent_from_the_corpus": sorted(uncitable),
        "claimed_provision_character": (finding.raw.get("basis") or {}).get(
            "character_kind"
        ),
        "explanation": None,
        "explanation_note": (
            "Zapis ustalenia nie zawiera pola z wyjaśnieniem; kryterium "
            "wyjaśnienia jest wtedy nieoceniane."
        ),
        # A substantive finding asserts a relation between a passage and the law,
        # so its basis is always a criterion: a judge answering not_applicable
        # would be excusing the finding from the denominator it belongs in. A
        # finding that cites nothing is judged unsupported, not exempt.
        "criteria_applicability": {
            "legal_basis": "required",
            "explanation": "not_applicable",
        },
    }


def run_payload(
    document: dict[str, Any],
    findings: list[dict[str, Any]],
    key_items: list[dict[str, Any]],
    provisions: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """The one shape the judge is ever shown, for a run and for a control case."""
    return {
        "document": document,
        "assessed_findings": findings,
        "key_items": key_items,
        "provisions": provisions,
    }


def build_run_packet(
    findings: tuple[RunFinding, ...],
    key: AnswerKey,
    bundle: SourceBundle,
    document_text: str,
    max_chars: int,
) -> tuple[Packet, list[str]]:
    """One blinded packet for a whole run, or the source material it needed.

    The returned list is empty on success. When it is not, the packet is refused
    rather than built from a smaller context: a missing source is a failure of
    the evaluation, never a finding against the analyzer. One missing source
    refuses the run, because the alternative is judging the rest of it against
    material the record cannot show.
    """
    first = findings[0]
    document_key = first.document_key
    item_views = [_key_item_view(item) for item in key.items_for(document_key)]
    accepted: set[str] = set()
    for item in key.items_for(document_key):
        accepted.update(accepted_bases_of(item))
    claimed: set[str] = set()
    views: list[dict[str, Any]] = []
    missing: list[str] = []
    for finding in findings:
        uncitable, unprepared = basis_problems(finding, bundle, key)
        claimed.update(set(finding.locators) - set(uncitable) - set(unprepared))
        missing.extend(f"{locator} (claimed basis)" for locator in unprepared)
        views.append(finding_view(finding, document_text, uncitable))
    claimed_views, _ = _provision_views(tuple(sorted(claimed)), bundle, "claimed basis")
    # Every accepted basis of this document is required, including one a finding
    # also claims. Subtracting the claimed set here is what would let a source
    # nobody prepared be reported as a citation error of a finding.
    expected_views, absent = _provision_views(
        tuple(sorted(accepted)), bundle, "key basis"
    )
    missing.extend(absent)
    finding_ids = tuple(finding.finding_id for finding in findings)
    if missing:
        return (
            Packet(
                packet_id="",
                series=first.series,
                document_key=document_key,
                arm=first.arm,
                finding_ids=finding_ids,
                payload={},
            ),
            sorted(set(missing)),
        )

    payload = run_payload(
        {"document_key": document_key, "text": document_text},
        views,
        item_views,
        {
            "claimed_by_the_findings": claimed_views,
            "cited_by_the_key": [
                view for view in expected_views if view["locator"] not in claimed
            ],
        },
    )
    # Two runs can emit byte-identical findings, so the identity of the case is
    # the findings it is about, not the text of the packet. Those ids are minted
    # uuids and say nothing about the arm.
    packet_id = sha256_value(
        {"finding_ids": list(finding_ids), "payload": sha256_value(payload)}
    )[:16]
    packet = Packet(
        packet_id=packet_id,
        series=first.series,
        document_key=document_key,
        arm=first.arm,
        finding_ids=finding_ids,
        payload=payload,
    )
    require(
        len(packet.case) <= max_chars,
        f"packet for the run of {document_key} carrying {len(finding_ids)} "
        f"finding(s) is {len(packet.case)} characters, over the fixed limit of "
        f"{max_chars}; the context rule stops the case rather than quietly "
        f"cutting its sources",
    )
    return packet, []


def build_packets(
    accounted: tuple[Accounted, ...],
    key: AnswerKey,
    bundle: SourceBundle,
    runs: RecordedRuns,
    max_chars: int,
) -> tuple[Packet, ...]:
    """One packet per recorded run with findings to judge, or a missing source.

    A run that emitted nothing substantive has no packet: there is no question to
    ask about it. It is still a cell of the report, and the report seeds its cells
    from the runs rather than from the packets for exactly that reason.
    """
    texts = {
        document_key: bundle.document_text(document_key, record["content_sha256"])
        for document_key, record in runs.documents.items()
    }
    by_run: dict[tuple[int, str, str], list[RunFinding]] = {}
    for entry in accounted:
        if entry.disposition != "judged":
            continue
        by_run.setdefault(entry.finding.cell, []).append(entry.finding)
    packets: list[Packet] = []
    missing: list[str] = []
    for cell in sorted(by_run):
        findings = by_run[cell]
        packet, absent = build_run_packet(
            tuple(findings),
            key,
            bundle,
            texts[findings[0].document_key],
            max_chars,
        )
        if absent:
            missing.extend(absent)
        else:
            packets.append(packet)
    require(
        not missing,
        "prepared source text is missing for "
        f"{len(set(missing))} provision(s): {', '.join(sorted(set(missing))[:10])}"
        f"{' ...' if len(set(missing)) > 10 else ''}. "
        "Run `evaluate.py prepare` against the declared corpus manifest; the judge "
        "is never asked to work without the source it is meant to check.",
    )
    return tuple(packets)


# --------------------------------------------------------------------------
# calibration cases
# --------------------------------------------------------------------------

CALIBRATION_ID_PREFIX = "cal-"
_PAYLOAD_KEYS = ("document", "key_items", "provisions")
_APPLICABILITY = ("required", "not_applicable")


def calibration_findings(case_id: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
    """The findings a supplied case asks about, in the judged shape.

    A case may state one finding or several. One finding is named by the case,
    which is what keeps a rating file written against a single-finding case valid
    without change; several are named by their own ids, which is what lets a
    control case check that a reply answers every finding it was given and only
    those.
    """
    several = payload.get("assessed_findings")
    if isinstance(several, list):
        return [
            {**dict(finding), "finding_id": str(finding.get("finding_id", ""))}
            for finding in several
        ]
    finding = dict(payload["assessed_finding"])
    finding["finding_id"] = case_id
    finding["criteria_applicability"] = payload["criteria_applicability"]
    return [finding]


def calibration_packet(case_id: str, payload: dict[str, Any]) -> Packet:
    """One supplied case as the packet it is judged as.

    Judging and the offline control comparison mint the id the same way, so a
    saved judgement ties back to the case that produced it without either side
    reimplementing the rule. The supplied case keeps the shape the author writes;
    what is hashed and sent is the run packet it becomes, so a control judgement
    is made under exactly the contract the cohort is judged under.
    """
    findings = calibration_findings(case_id, payload)
    converted = run_payload(
        payload["document"], findings, payload["key_items"], payload["provisions"]
    )
    return Packet(
        packet_id=CALIBRATION_ID_PREFIX
        + sha256_value({"case_id": case_id, "payload": converted})[:16],
        series=0,
        document_key=str(payload["document"]["document_key"]),
        arm="calibration",
        finding_ids=tuple(str(finding["finding_id"]) for finding in findings),
        payload=converted,
    )


def load_calibration_packets(
    path: Path,
    runs: RecordedRuns,
    max_chars: int,
) -> tuple[Packet, ...]:
    """The author's development and control cases, kept out of the comparison.

    A calibration case is supplied whole, as its own small packet: its document,
    its finding, the expectations it is to be matched against and the source text
    it is to be checked against. It is not drawn from the recorded runs, and the
    disjointness checks below are what make that a fact rather than an intention:
    no recorded finding id, and no document of the judged cohort.
    """
    raw = load_json(path)
    cases = raw.get("cases")
    require(
        isinstance(cases, list) and bool(cases),
        f"{path} carries no calibration cases",
    )
    recorded_ids = {finding.finding_id for finding in runs.findings}
    cohort_hashes = {record["content_sha256"] for record in runs.documents.values()}
    packets: list[Packet] = []
    seen: set[str] = set()
    for case in cases:
        require(isinstance(case, dict), f"{path} holds a non-object case")
        case_id = case.get("case_id")
        require(
            isinstance(case_id, str) and bool(case_id) and case_id not in seen,
            f"{path} case ids must be present and unique: {case_id!r}",
        )
        seen.add(case_id)
        require(
            case_id not in recorded_ids,
            f"calibration case {case_id} names a finding of the judged runs. "
            f"Calibration cases stand outside the reported comparison, so they may "
            f"not be one of its findings under another name.",
        )
        payload = case.get("payload")
        require(
            isinstance(payload, dict) and all(key in payload for key in _PAYLOAD_KEYS),
            f"calibration case {case_id} must carry a payload with "
            f"{', '.join(_PAYLOAD_KEYS)}",
        )
        require(
            isinstance(payload.get("assessed_findings"), list)
            or ("assessed_finding" in payload and "criteria_applicability" in payload),
            f"calibration case {case_id} must carry either assessed_findings or "
            f"an assessed_finding with its criteria_applicability",
        )
        document = payload["document"]
        require(
            isinstance(document, dict)
            and isinstance(document.get("text"), str)
            and bool(document["text"])
            and isinstance(document.get("document_key"), str),
            f"calibration case {case_id} must carry its own document text",
        )
        digest = hashlib.sha256(document["text"].encode("utf-8")).hexdigest()
        require(
            digest not in cohort_hashes,
            f"calibration case {case_id} uses a document of the judged cohort. "
            f"Calibration may not be run on the documents the comparison reports.",
        )
        named: set[str] = set()
        for finding in calibration_findings(case_id, payload):
            require(
                "explanation" in finding,
                f"calibration case {case_id} must state the finding's explanation, "
                f"even when it is null",
            )
            finding_id = str(finding["finding_id"])
            require(
                bool(finding_id) and finding_id not in named,
                f"calibration case {case_id} must name each of its findings once, "
                f"with a non-empty id; a reply can only be checked against ids it "
                f"can be matched to",
            )
            named.add(finding_id)
            applicability = finding.get("criteria_applicability")
            require(
                isinstance(applicability, dict)
                and all(
                    applicability.get(criterion) in _APPLICABILITY
                    for criterion in ("legal_basis", "explanation")
                ),
                f"calibration case {case_id} must say, for the basis and the "
                f"explanation of {finding_id}, whether the criterion is required "
                f"or not applicable",
            )
        provisions = payload["provisions"]
        require(
            isinstance(provisions, dict)
            and all(
                isinstance(provisions.get(group), list)
                for group in ("claimed_by_the_finding", "cited_by_the_key")
            ),
            f"calibration case {case_id} must carry both provision groups",
        )
        # Checked here, before any call: a view without a locator and its text is
        # a source the reply could never be checked against, and finding that out
        # while reading the answer would mean the case had already been paid for.
        for group in ("claimed_by_the_finding", "cited_by_the_key"):
            for view in provisions[group]:
                require(
                    isinstance(view, dict)
                    and isinstance(view.get("locator"), str)
                    and bool(view["locator"])
                    and isinstance(view.get("text"), str)
                    and bool(view["text"]),
                    f"calibration case {case_id} carries a provision in {group} "
                    f"without a locator and the text a quote would be checked "
                    f"against",
                )
        require(
            isinstance(payload["key_items"], list),
            f"calibration case {case_id} must carry a key_items list",
        )
        packet = calibration_packet(case_id, payload)
        require(
            len(packet.case) <= max_chars,
            f"calibration case {case_id} is {len(packet.case)} characters, over the "
            f"fixed limit of {max_chars}",
        )
        packets.append(packet)
    return tuple(packets)
