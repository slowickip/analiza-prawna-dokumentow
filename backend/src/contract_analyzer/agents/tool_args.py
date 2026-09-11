"""Closed argument models for the worksheet roles' provider-facing tools."""

from __future__ import annotations

import json

from pydantic import ConfigDict, Field, field_validator

from contract_analyzer.agents.schema import FrozenModel
from contract_analyzer.domain import (
    CharacterEvidence,
    DepartureDirection,
    DepartureState,
    InterpretiveMethod,
    ProvisionCharacter,
    SourceKind,
)

# Examples reach the model as part of the provider-facing tool definition, so a
# real locator here is a value the model can copy into a commit about an entirely
# different provision. These two resolve to nothing: copied, they are refused,
# and the shape is still shown.
PLACEHOLDER_LOCATOR = "https://api.sejm.gov.pl/eli/acts/DU/0000/0/text.html/arti=0"


def _one_string(value: object) -> object:
    return (value,) if isinstance(value, str) else value


def _character_evidence(value: object) -> object:
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return value
        if not isinstance(decoded, dict | list):
            return value
        value = decoded
    items: object = [value] if isinstance(value, dict) else value
    if not isinstance(items, list | tuple):
        return items
    return [
        {
            **item,
            "interpretive_methods": _one_string(item.get("interpretive_methods")),
        }
        if isinstance(item, dict) and isinstance(item.get("interpretive_methods"), str)
        else item
        for item in items
    ]


class SearchCorpusArgs(FrozenModel):
    phrase: str = Field(
        description="Krótka fraza wyszukiwania w języku ustawy, nie fragment umowy."
    )


class ReadProvisionArgs(FrozenModel):
    locator: str = Field(description="Lokalizator przepisu w zamrożonym korpusie.")


class CandidateArgs(FrozenModel):
    locator: str = Field(description="Lokalizator zgłaszanego przepisu.")
    why: str = Field(
        description=(
            "Jedno lub dwa zdania nazywające instytucję prawną, nigdy ocena zgodności."
        )
    )
    supporting_locators: tuple[str, ...] = Field(
        default=(),
        description=(
            "Lokalizatory przepisów, które ten kandydat wymaga do odczytania. "
            "Przykład pokazuje wyłącznie kształt wartości; podaj lokalizator "
            "zwrócony przez narzędzie w tej jednostce."
        ),
        examples=[[PLACEHOLDER_LOCATOR]],
    )

    _coerce_supporting_locators = field_validator("supporting_locators", mode="before")(
        _one_string
    )


class PostCandidatesArgs(FrozenModel):
    # The lower bound is zero because "this corpus holds no basis for this clause"
    # is one of the system's results, not an absence of one. With a floor of one
    # the researcher had no way to state it and could only fall silent or spend
    # its whole turn budget searching, which is what a measured run then recorded.
    candidates: tuple[CandidateArgs, ...]


class PostCharacterArgs(ProvisionCharacter):
    """Characterisation arguments reuse the domain record's validators."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    evidence: tuple[CharacterEvidence, ...] = Field(
        default=(),
        description=(
            "Dowody charakteru przepisu. Lokalizator musi pochodzić z wyniku "
            "narzędzia w tej jednostce; przykład pokazuje wyłącznie kształt."
        ),
        examples=[
            [
                CharacterEvidence(
                    source_kind=SourceKind.OFFICIAL_NORMATIVE_TEXT,
                    locator=PLACEHOLDER_LOCATOR,
                    pinpoint="ust. 1",
                    interpretive_methods=(InterpretiveMethod.LINGUISTIC,),
                    rationale=(
                        "Przepis określa zamknięty katalog przyczyn wypowiedzenia."
                    ),
                )
            ]
        ],
    )

    _coerce_evidence = field_validator("evidence", mode="before")(_character_evidence)


class RelevanceVerdictArgs(FrozenModel):
    relevant: bool | None = Field(
        description="true, false albo null, gdy ocena pozostaje niepewna."
    )
    quote: str = Field(description="Dosłowny fragment jednostki umowy.")
    # Required, not conditional: whether this verdict resolves to an uncertain
    # finding depends on the candidate's force state, and the verifier is not
    # shown it. A condition the role cannot evaluate costs it a turn to discover.
    raw_confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Miara pewności tej oceny w przedziale [0, 1]; zawsze wymagana.",
    )
    based_on: tuple[str, ...] = Field(
        description=(
            "Identyfikatory widocznych wpisów arkusza, na których opiera się ocena, "
            "nie lokalizatory przepisów; lista może być pusta."
        ),
        examples=[["e1", "e4"]],
    )

    _coerce_based_on = field_validator("based_on", mode="before")(_one_string)


class RelationVerdictArgs(FrozenModel):
    departure: DepartureState
    direction: DepartureDirection | None = None
    raw_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    based_on: tuple[str, ...] = Field(
        description=(
            "Identyfikatory widocznych wpisów arkusza, na których opiera się ocena, "
            "nie lokalizatory przepisów; lista może być pusta."
        ),
        examples=[["e1", "e4"]],
    )

    _coerce_based_on = field_validator("based_on", mode="before")(_one_string)


class ListFindingsArgs(FrozenModel):
    """No arguments: the listing is the findings the run already stored."""


class SynthesisGroupArgs(FrozenModel):
    title: str
    summary: str
    # Numbered references from the listing, never finding identifiers: resolved
    # back to identifiers where the grouping is kept.
    finding_ids: tuple[str, ...]


class GroupFindingsArgs(FrozenModel):
    groups: tuple[SynthesisGroupArgs, ...] = ()
