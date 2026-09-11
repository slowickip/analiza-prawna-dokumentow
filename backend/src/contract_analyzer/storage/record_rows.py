"""Database row conversion helpers."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import date
from typing import Any

from pydantic import TypeAdapter

from contract_analyzer.domain import EmittedBasis
from contract_analyzer.storage.records import (
    AttemptRecord,
    CostRecord,
    FindingRecord,
    RunRecord,
    _parameters,
)

_RUN_ADAPTER = TypeAdapter(RunRecord)
_ATTEMPT_ADAPTER = TypeAdapter(AttemptRecord)
_FINDING_ADAPTER = TypeAdapter(FindingRecord)


def _cost_from_row(row: Mapping[str, Any]) -> CostRecord:
    monetary = row.get("monetary_cost_microunits")
    price_date = row.get("price_table_date")
    return CostRecord(
        monetary_cost_microunits=int(monetary) if monetary is not None else None,
        price_table_date=date.fromisoformat(price_date) if price_date else None,
        price_table_hash=row.get("price_table_hash"),
        unknown_reason=row.get("cost_unknown_reason"),
    )


def _run_from_row(row: Mapping[str, Any]) -> RunRecord:
    data = dict(row)
    data["cost"] = _cost_from_row(row)
    data["parameters"] = _parameters(data["parameters_json"])
    data["measurement_valid"] = bool(data["measurement_valid"])
    data["error_detail"] = None
    data["source_text"] = None
    return _RUN_ADAPTER.validate_python(data)


def _attempt_from_row(row: Mapping[str, Any]) -> AttemptRecord:
    data = dict(row)
    data["parameters"] = _parameters(data["parameters_json"])
    data["prompt_hash"] = data.get("prompt_hash") or ""
    return _ATTEMPT_ADAPTER.validate_python(data)


def _finding_from_row(row: Mapping[str, Any]) -> FindingRecord:
    data = dict(row)
    bbox_json = data.get("bbox_json")
    if bbox_json:
        bbox_values = json.loads(bbox_json)
        if isinstance(bbox_values, list) and len(bbox_values) == 4:
            data["bbox"] = (
                bbox_values[0],
                bbox_values[1],
                bbox_values[2],
                bbox_values[3],
            )
    locators_json = data.get("legal_locators_json")
    if locators_json:
        data["legal_locators"] = tuple(json.loads(locators_json))
    basis_json = data.get("basis_json")
    if basis_json:
        data["basis"] = EmittedBasis.model_validate_json(basis_json)
    return _FINDING_ADAPTER.validate_python(data)
