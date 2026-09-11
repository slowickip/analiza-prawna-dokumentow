"""Manifest and corpus snapshot models."""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import ConfigDict, Field, model_validator

from contract_analyzer.domain import (
    ForceState,
    FrozenModel,
    validate_force_record_pair,
)

_FORBID = ConfigDict(extra="forbid")


class AmendmentRef(FrozenModel):
    model_config = _FORBID

    id: str
    # ELI relation date; not checked against each amending act's commencement.
    eli_relation_date: date


class RepeatedArticleCount(FrozenModel):
    model_config = _FORBID

    article: str
    printings: int = Field(ge=2)


class ManifestAct(FrozenModel):
    model_config = _FORBID

    publisher: str
    year: int
    position: int
    source_format: Literal["html", "pdf"]
    base_act: str
    legal_status_date: date
    amendments_not_carried: tuple[AmendmentRef, ...]
    repeated_articles: tuple[RepeatedArticleCount, ...] = ()
    expected_hash: str | None = None


class CorpusManifest(FrozenModel):
    model_config = _FORBID

    version: Literal[2]
    corpus_target_date: date
    acts: tuple[ManifestAct, ...] = Field(min_length=1)


class ActCurrency(FrozenModel):
    """How one act stood at the moment the corpus was built.

    The two source formats are the format the manifest asked for and the one the
    build read. They differ when ELI's HTML routes were down and the act was built
    from its PDF instead, which cuts it into different units. Both are recorded
    because a snapshot has to answer the question without the manifest, which may
    have moved on. ``None`` on snapshots built before either was recorded.
    """

    act_identifier: str
    base_act: str
    legal_status_date: date
    amendments_not_carried: tuple[AmendmentRef, ...]
    repeated_articles: tuple[RepeatedArticleCount, ...]
    source_format_declared: Literal["html", "pdf"] | None = None
    source_format_used: Literal["html", "pdf"] | None = None


class LegalUnit(FrozenModel):
    id: str
    locator: str
    act_identifier: str
    article_identifier: str
    text: str = Field(repr=False)
    content_hash: str
    act_force: ForceState
    provision_force: ForceState
    legal_status_date: date

    @model_validator(mode="after")
    def validate_force_records(self) -> LegalUnit:
        validate_force_record_pair(self.act_force, self.provision_force)
        return self


class CorpusSnapshot(FrozenModel):
    id: str
    built_at: datetime
    manifest_hash: str
    model_name: str
    # Which upstream answered, and what it encoded the probe text as. A model name
    # has already stood for two spaces 0.52 cosine apart here, so the probe is the
    # only part of the record a reader can check its own service against. Empty on
    # a snapshot built before it was recorded, which retrieval refuses to serve.
    embedding_provider: str = ""
    embedding_space_probe: tuple[float, ...] = ()
    file_hashes: dict[str, str]
    unit_count: int
    act_identifiers: tuple[str, ...]
    corpus_target_date: date
    act_currency: tuple[ActCurrency, ...]
    unit_digest: str

    @property
    def source_format(self) -> Literal["as_declared", "fallback", "unrecorded"]:
        """Whether every act was read from the source format its manifest asked for.

        Not *which* format: an act's own ``source_format_used`` says html or pdf.

        Three-valued because a snapshot that predates the record, or lists no acts,
        cannot answer the question at all, and answering ``as_declared`` for it
        would turn an unknown into a clean bill of health. Callers that must not
        measure a substituted corpus refuse ``unrecorded`` alongside ``fallback``.
        """
        if not self.act_currency or any(
            act.source_format_declared is None or act.source_format_used is None
            for act in self.act_currency
        ):
            return "unrecorded"
        if any(
            act.source_format_declared != act.source_format_used
            for act in self.act_currency
        ):
            return "fallback"
        return "as_declared"

    @property
    def fallback_acts(self) -> tuple[str, ...]:
        """The acts read from a source format the manifest did not ask for."""
        return tuple(
            act.act_identifier
            for act in self.act_currency
            if act.source_format_declared is not None
            and act.source_format_used is not None
            and act.source_format_declared != act.source_format_used
        )
