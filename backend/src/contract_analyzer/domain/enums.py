"""Domain enumerations and scalar aliases."""

from enum import StrEnum
from typing import Literal


class ArmCode(StrEnum):
    OFF = "off"
    MID = "mid"
    ON = "on"


class ReadMode(StrEnum):
    NATIVE_PDF = "native_pdf"
    OCR_PDF = "ocr_pdf"


AttemptStatus = Literal[
    "success",
    "transport_error",
    "http_error",
    "invalid_response",
    "model_identity_changed",
]
ParameterValue = str | int | float | bool | None


class FindingCode(StrEnum):
    CONSISTENT = "consistent"
    CONTRADICTORY = "contradictory"
    PERMISSIBLE_DEPARTURE = "permissible_departure"
    NO_RELATION = "no_relation"
    NO_BASIS_FOUND = "no_basis_found"
    BASIS_NOT_IN_FORCE = "basis_not_in_force"
    UNIT_NOT_ADJUDICABLE = "unit_not_adjudicable"
    NOT_PROCESSED = "not_processed"
    UNCERTAIN = "uncertain"


class UncertainCause(StrEnum):
    RELATION_BELOW_THRESHOLD = "relation_below_threshold"
    FORCE_STATE_UNDETERMINED = "force_state_undetermined"
    PROVISION_CHARACTER_UNDETERMINED = "provision_character_undetermined"
    PERMITTED_DIRECTION_UNDETERMINED = "permitted_direction_undetermined"
    DEPARTURE_DIRECTION_UNDETERMINED = "departure_direction_undetermined"


class QuoteResolution(StrEnum):
    """Outcome of resolving a citation quote returned by the model against text.

    This is intentionally a dedicated enum rather than an overload of
    UncertainCause: UncertainCause strictly denotes substantive legal adjudication
    uncertainty and requires raw_confidence, whereas quote resolution tracks
    evidence anchoring. A citation resolution failure preserves the substantive
    legal finding code while explicitly recording why the source text anchor
    could not be resolved.
    """

    RESOLVED = "resolved"
    QUOTE_EMPTY = "quote_empty"
    QUOTE_FABRICATED = "quote_fabricated"
    QUOTE_AMBIGUOUS = "quote_ambiguous"


class ForceValue(StrEnum):
    IN_FORCE = "in_force"
    NOT_IN_FORCE = "not_in_force"
    UNDETERMINED = "undetermined"


class ForceScope(StrEnum):
    ACT = "act"
    PROVISION = "provision"


class ProvisionKind(StrEnum):
    IMPERATIVE = "imperative"
    DISPOSITIVE = "dispositive"
    SEMI_IMPERATIVE = "semi_imperative"
    UNDETERMINED = "undetermined"


class DepartureState(StrEnum):
    NONE = "none"
    PRESENT = "present"
    UNDETERMINED = "undetermined"


class DepartureDirection(StrEnum):
    WITH_PERMITTED_DIRECTION = "with_permitted_direction"
    AGAINST_PERMITTED_DIRECTION = "against_permitted_direction"
    UNDETERMINED = "undetermined"


class ReferenceType(StrEnum):
    FULL_INTERNAL = "full_internal"
    RELATIVE_INTERNAL = "relative_internal"
    DIRECTIONAL = "directional"
    ANNEX = "annex"
    WHOLE_DOCUMENT = "whole_document"
    EXTERNAL_ACT = "external_act"


class ReferenceStatus(StrEnum):
    RESOLVED = "resolved"
    TARGET_DOES_NOT_EXIST = "target_does_not_exist"
    WITHIN_UNIT = "within_unit"
    AMBIGUOUS_DIRECTIONAL = "ambiguous_directional"
    OUTSIDE_INPUT = "outside_input"
    WHOLE_DOCUMENT = "whole_document"
    EXTERNAL_ACT = "external_act"


class SourceKind(StrEnum):
    OFFICIAL_NORMATIVE_TEXT = "official_normative_text"
    JUDICIAL_DECISION = "judicial_decision"
    DOCTRINAL_PUBLICATION = "doctrinal_publication"
    LEGISLATIVE_MATERIAL = "legislative_material"


class InterpretiveMethod(StrEnum):
    LINGUISTIC = "linguistic"
    SYSTEMIC = "systemic"
    PURPOSIVE = "purposive"
