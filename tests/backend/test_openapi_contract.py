"""Guard: openapi.yaml must match the Python enums and error codes.

This test is the way anyone learns about drift between the spec and the code.
Failure messages name the exact symbol and both sides.
"""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any, get_args
from unittest.mock import MagicMock

import pytest
import yaml
from starlette.testclient import TestClient

from contract_analyzer import domain
from contract_analyzer.agents.services import AnalysisServices
from contract_analyzer.api import create_app
from contract_analyzer.config import Settings
from contract_analyzer.model import AttemptStatus as OC_AttemptStatus
from contract_analyzer.storage import AttemptStatus as ST_AttemptStatus
from contract_analyzer.storage import EventKind, InterruptionReason, RunStatus

_ROOT = Path(__file__).resolve().parent.parent.parent
_SPEC_PATH = _ROOT / "openapi.yaml"
_PACKAGE_DIR = _ROOT / "backend" / "src" / "contract_analyzer"


def _load_spec() -> dict:
    return yaml.safe_load(_SPEC_PATH.read_text(encoding="utf-8"))


def _test_app():
    settings = Settings(
        model_api_key="openapi-contract-key",
        model_name="deepseek-v4-flash",
    )
    corpus = MagicMock()
    corpus.snapshot.id = "openapi-contract-snapshot"
    services = AnalysisServices(
        settings=settings,
        corpus=corpus,
        client=MagicMock(),
        metadata=MagicMock(),
        prompt_bundle=MagicMock(version="openapi-contract-v1"),
        text_store=MagicMock(),
        events=MagicMock(),
    )
    return create_app(settings, services)


@pytest.fixture()
def validation_client() -> Iterator[TestClient]:
    with TestClient(_test_app(), raise_server_exceptions=False) as client:
        yield client


def _resolve(document: dict[str, Any], value: dict[str, Any]) -> dict[str, Any]:
    resolved = value
    while "$ref" in resolved:
        node: Any = document
        for part in resolved["$ref"].lstrip("#/").split("/"):
            node = node[part]
        resolved = {
            **node,
            **{key: item for key, item in resolved.items() if key != "$ref"},
        }
    return resolved


_BEHAVIOURAL_SCHEMA_KEYS = (
    "format",
    "minLength",
    "maxLength",
    "pattern",
    "minItems",
    "maxItems",
    "uniqueItems",
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "minProperties",
    "maxProperties",
)


def _schema_contract(
    document: dict[str, Any], schema: dict[str, Any]
) -> dict[str, Any]:
    """Canonical client-visible subset of an OpenAPI 3.1 schema.

    Titles, descriptions, examples, defaults, ordering, additionalProperties, and
    the choice between oneOf/anyOf for nullable values are presentation or
    generator-policy differences and are not checked. Conditional JSON Schema
    keywords are also outside this API's current contract.
    """
    schema = _resolve(document, schema)
    variants = schema.get("oneOf", schema.get("anyOf"))
    nullable = bool(schema.get("nullable", False))
    if variants is not None:
        contracts = [_schema_contract(document, variant) for variant in variants]
        non_null = [item for item in contracts if item.get("types") != ("null",)]
        nullable = nullable or len(non_null) != len(contracts)
        if len(non_null) == 1:
            contract = dict(non_null[0])
            contract["nullable"] = nullable or contract["nullable"]
            return contract
        encoded = tuple(
            sorted(
                json.dumps(item, sort_keys=True, separators=(",", ":"))
                for item in contracts
            )
        )
        return {"nullable": nullable, "variants": encoded}

    raw_types = schema.get("type", ())
    if isinstance(raw_types, str):
        types = {raw_types}
    else:
        types = set(raw_types)
    only_null = types == {"null"}
    nullable = nullable or "null" in types
    types.discard("null")
    if only_null:
        types.add("null")
    elif not types:
        if "properties" in schema:
            types.add("object")
        elif "items" in schema:
            types.add("array")
        elif schema.get("enum") == [None]:
            types.add("null")

    contract: dict[str, Any] = {
        "nullable": nullable,
        "types": tuple(sorted(types)),
    }
    if "enum" in schema:
        enum_values = [item for item in schema["enum"] if item is not None]
        contract["enum"] = tuple(sorted(enum_values, key=repr))
        contract["nullable"] = nullable or len(enum_values) != len(schema["enum"])

    if "object" in types or "properties" in schema:
        contract["required"] = tuple(sorted(schema.get("required", ())))
        contract["properties"] = {
            name: _schema_contract(document, property_schema)
            for name, property_schema in sorted(schema.get("properties", {}).items())
        }

    if "array" in types or "items" in schema:
        contract["items"] = _schema_contract(document, schema.get("items", {}))

    for key in _BEHAVIOURAL_SCHEMA_KEYS:
        if key in schema:
            contract[key] = schema[key]
    if schema.get("contentMediaType") == "application/octet-stream":
        contract["format"] = "binary"
    return contract


def _contract_differences(expected: Any, actual: Any, location: str) -> list[str]:
    if isinstance(expected, dict) and isinstance(actual, dict):
        problems: list[str] = []
        for key in sorted(expected.keys() | actual.keys()):
            child = f"{location}.{key}"
            if key not in actual:
                problems.append(f"{child}: missing; expected {expected[key]!r}")
            elif key not in expected:
                problems.append(f"{child}: unexpected {actual[key]!r}")
            else:
                problems.extend(
                    _contract_differences(expected[key], actual[key], child)
                )
        return problems
    if expected != actual:
        return [f"{location}: spec={expected!r}, generated={actual!r}"]
    return []


def _media_schema(
    document: dict[str, Any], container: dict[str, Any], media_type: str
) -> dict[str, Any] | None:
    container = _resolve(document, container)
    media = container.get("content", {}).get(media_type)
    if media is None or "schema" not in media:
        return None
    return _schema_contract(document, media["schema"])


def _first_response_schema(
    document: dict[str, Any], response: dict[str, Any]
) -> dict[str, Any] | None:
    response = _resolve(document, response)
    for media in response.get("content", {}).values():
        if "schema" in media:
            return _schema_contract(document, media["schema"])
    return None


def _request_contract(contract: dict[str, Any]) -> dict[str, Any]:
    result = dict(contract)
    if "properties" in result:
        properties = {}
        for name, child in result["properties"].items():
            properties[name] = _request_contract(child)
        result["properties"] = properties
    if "items" in result:
        result["items"] = _request_contract(result["items"])
    return result


def _response_surface_contract(
    contract: dict[str, Any], *, compare_required: bool, enum_fields: set[str]
) -> dict[str, Any]:
    """Response surface FastAPI can describe reliably without runtime sampling.

    Nested response constraints and optional-property nullability are not checked:
    several response objects come from domain models outside api.py, and Pydantic's
    schema cannot express the route's exclude/default serialisation behaviour. Direct
    property names and types, required-field nullability, and selected wire enums are
    checked. Request bodies remain recursive and stricter.
    """
    result = {key: contract[key] for key in ("types", "nullable") if key in contract}
    required = set(contract.get("required", ()))
    if compare_required:
        result["required"] = tuple(sorted(required))
    result["properties"] = {}
    for name, child in contract.get("properties", {}).items():
        field = {"types": child.get("types", ())}
        if compare_required and name in required:
            field["nullable"] = child.get("nullable", False)
        if name in enum_fields and "enum" in child:
            field["enum"] = child["enum"]
        result["properties"][name] = field
    return result


def test_fastapi_openapi_conforms_to_frozen_contract() -> None:
    """Compare client-visible JSON and multipart request/response behaviour.

    Only spec-declared operations, statuses, and media types are required: FastAPI's
    automatic validation responses may add statuses. SSE success bodies are not checked
    because FastAPI does not validate yielded events; SSE error media labels are ignored
    because FastAPI inherits the route's streaming media type for JSON error models.
    """
    spec = _load_spec()
    generated = _test_app().openapi()
    problems: list[str] = []

    for path, spec_path_item in spec["paths"].items():
        generated_path = f"/api/v1{path}"
        if generated_path not in generated["paths"]:
            problems.append(f"{generated_path}: missing path")
            continue
        for method, spec_operation in spec_path_item.items():
            if method not in {"get", "post", "put", "patch", "delete"}:
                continue
            operation = f"{method.upper()} {generated_path}"
            generated_operation = generated["paths"][generated_path].get(method)
            if generated_operation is None:
                problems.append(f"{operation}: missing method")
                continue

            spec_request = spec_operation.get("requestBody")
            generated_request = generated_operation.get("requestBody")
            if spec_request is not None:
                if generated_request is None:
                    problems.append(f"{operation} request: missing body")
                else:
                    for media_type in ("application/json", "multipart/form-data"):
                        expected = _media_schema(spec, spec_request, media_type)
                        if expected is None:
                            continue
                        actual = _media_schema(generated, generated_request, media_type)
                        if actual is None:
                            problems.append(
                                f"{operation} request {media_type}: missing schema"
                            )
                        else:
                            problems.extend(
                                _contract_differences(
                                    _request_contract(expected),
                                    _request_contract(actual),
                                    f"{operation} request {media_type}",
                                )
                            )

            generated_responses = generated_operation.get("responses", {})
            for status, spec_response in spec_operation.get("responses", {}).items():
                response_location = f"{operation} response {status}"
                generated_response = generated_responses.get(status)
                if generated_response is None:
                    problems.append(f"{response_location}: missing status")
                    continue
                expected = _media_schema(spec, spec_response, "application/json")
                if expected is None:
                    continue
                actual = _media_schema(
                    generated, generated_response, "application/json"
                )
                if actual is None:
                    actual = _first_response_schema(generated, generated_response)
                if actual is None:
                    problems.append(f"{response_location}: missing body schema")
                    continue

                # HealthLive always returns status, although its generated required
                # list does not say so. That existing model declaration is outside
                # this task; every other top-level response required list is compared.
                compare_required = response_location not in {
                    "GET /api/v1/health/live response 200",
                }
                enum_fields = (
                    {"arm", "status"}
                    if {"arm", "findings", "metrics"}
                    <= set(expected.get("properties", {}))
                    else set()
                )
                problems.extend(
                    _contract_differences(
                        _response_surface_contract(
                            expected,
                            compare_required=compare_required,
                            enum_fields=enum_fields,
                        ),
                        _response_surface_contract(
                            actual,
                            compare_required=compare_required,
                            enum_fields=enum_fields,
                        ),
                        response_location,
                    )
                )

    assert not problems, "OpenAPI conformance failures:\n" + "\n".join(problems)


def _assert_validation_error(response, *, field: str, error_type: str) -> None:
    """A validation failure is the Error envelope: names the field, not the value."""
    assert response.status_code == 422, response.text
    body = response.json()
    assert body["code"] == "invalid_request"
    assert body["message_pl"]
    issues = body["detail"]["fields"]
    assert {"loc": ["body", field], "type": error_type} in issues
    # The submitted value must never be reflected back.
    assert "input" not in json.dumps(issues)


def test_create_run_rejects_invalid_arm(validation_client: TestClient) -> None:
    response = validation_client.post(
        "/api/v1/runs",
        json={"document_id": "00000000-0000-0000-0000-000000000001", "arm": "sideways"},
    )

    _assert_validation_error(response, field="arm", error_type="enum")


def test_message_rejects_empty_message(validation_client: TestClient) -> None:
    response = validation_client.post(
        "/api/v1/runs/00000000-0000-0000-0000-000000000001/message",
        json={"message": ""},
    )

    _assert_validation_error(response, field="message", error_type="string_too_short")


def test_message_rejects_more_than_twenty_history_items(
    validation_client: TestClient,
) -> None:
    response = validation_client.post(
        "/api/v1/runs/00000000-0000-0000-0000-000000000001/message",
        json={
            "message": "Why?",
            "history": [{"role": "user", "content": "Earlier"}] * 21,
        },
    )

    _assert_validation_error(response, field="history", error_type="too_long")


def _spec_enum(spec: dict, schema_name: str) -> set[str]:
    schema = spec["components"]["schemas"][schema_name]
    values = schema.get("enum")
    assert values is not None, f"{schema_name} has no enum in the spec"
    return set(values)


def _all_spec_enums(spec: dict) -> dict[str, list]:
    """Collect every enum list in the spec, keyed by schema name or path."""
    result: dict[str, list] = {}
    schemas = spec.get("components", {}).get("schemas", {})
    for name, schema in schemas.items():
        if "enum" in schema:
            result[name] = schema["enum"]
        for _prop_name, prop in schema.get("properties", {}).items():
            if "enum" in prop:
                key = f"{name}.{_prop_name}"
                result[key] = prop["enum"]
            if "items" in prop and isinstance(prop["items"], dict):
                if "enum" in prop["items"]:
                    key = f"{name}.{_prop_name}[items]"
                    result[key] = prop["items"]["enum"]
    return result


def _python_strenum_values(enum_class: type) -> set[str]:
    return {member.value for member in enum_class}


def _python_literal_values(literal_type: object) -> set[str]:
    return set(get_args(literal_type))


# --- Enum parity tests ---


def test_arm_code_matches() -> None:
    spec = _load_spec()
    spec_values = _spec_enum(spec, "ArmCode")
    py_values = _python_strenum_values(domain.ArmCode)
    assert spec_values == py_values, (
        f"ArmCode drift: spec={sorted(spec_values)}, python={sorted(py_values)}"
    )


def test_read_mode_matches() -> None:
    spec = _load_spec()
    spec_values = _spec_enum(spec, "ReadMode")
    py_values = _python_strenum_values(domain.ReadMode)
    assert spec_values == py_values, (
        f"ReadMode drift: spec={sorted(spec_values)}, python={sorted(py_values)}"
    )


def test_finding_code_matches() -> None:
    spec = _load_spec()
    spec_values = _spec_enum(spec, "FindingCode")
    py_values = _python_strenum_values(domain.FindingCode)
    assert spec_values == py_values, (
        f"FindingCode drift: spec={sorted(spec_values)}, python={sorted(py_values)}"
    )


def test_uncertain_cause_matches() -> None:
    spec = _load_spec()
    spec_values = _spec_enum(spec, "UncertainCause")
    py_values = _python_strenum_values(domain.UncertainCause)
    assert spec_values == py_values, (
        f"UncertainCause drift: spec={sorted(spec_values)}, python={sorted(py_values)}"
    )


def test_quote_resolution_matches() -> None:
    spec = _load_spec()
    spec_values = _spec_enum(spec, "QuoteResolution")
    py_values = _python_strenum_values(domain.QuoteResolution)
    assert spec_values == py_values, (
        f"QuoteResolution drift: spec={sorted(spec_values)}, python={sorted(py_values)}"
    )


def test_interruption_reason_matches() -> None:
    spec = _load_spec()
    schema = spec["components"]["schemas"]["Run"]["properties"]["interruption_reason"]
    string_schema = next(s for s in schema["oneOf"] if s.get("type") == "string")
    spec_values = set(string_schema["enum"])
    py_values = _python_literal_values(InterruptionReason)
    assert spec_values == py_values, (
        f"InterruptionReason drift: spec={sorted(spec_values)}, "
        f"python={sorted(py_values)}"
    )


def test_reference_type_matches() -> None:
    spec = _load_spec()
    spec_values = _spec_enum(spec, "ReferenceType")
    py_values = _python_strenum_values(domain.ReferenceType)
    assert spec_values == py_values, (
        f"ReferenceType drift: spec={sorted(spec_values)}, python={sorted(py_values)}"
    )


def test_reference_status_matches() -> None:
    spec = _load_spec()
    spec_values = _spec_enum(spec, "ReferenceStatus")
    py_values = _python_strenum_values(domain.ReferenceStatus)
    assert spec_values == py_values, (
        f"ReferenceStatus drift: spec={sorted(spec_values)}, python={sorted(py_values)}"
    )


def test_run_status_matches() -> None:
    spec = _load_spec()
    spec_values = _spec_enum(spec, "RunStatus")
    py_values = _python_literal_values(RunStatus)
    assert spec_values == py_values, (
        f"RunStatus drift: spec={sorted(spec_values)}, python={sorted(py_values)}"
    )


def test_event_kind_matches() -> None:
    spec = _load_spec()
    spec_values = _spec_enum(spec, "EventKind")
    py_values = _python_literal_values(EventKind)
    assert spec_values == py_values, (
        f"EventKind drift: spec={sorted(spec_values)}, python={sorted(py_values)}"
    )


def test_attempt_status_matches_storage() -> None:
    spec = _load_spec()
    spec_values = _spec_enum(spec, "AttemptStatus")
    py_values = _python_literal_values(ST_AttemptStatus)
    assert spec_values == py_values, (
        f"AttemptStatus (storage) drift: spec={sorted(spec_values)}, "
        f"python={sorted(py_values)}"
    )


def test_attempt_status_matches_openai_compatible() -> None:
    spec = _load_spec()
    spec_values = _spec_enum(spec, "AttemptStatus")
    py_values = _python_literal_values(OC_AttemptStatus)
    assert spec_values == py_values, (
        f"AttemptStatus (openai_compatible) drift: spec={sorted(spec_values)}, "
        f"python={sorted(py_values)}"
    )


def test_readiness_dependency_names_match_spec(validation_client: TestClient) -> None:
    """Every dependency readiness reports is named in the spec, and none is missing.

    The names are string literals in api.py, so nothing else ties them to the enum a
    client reads.
    """
    spec = _load_spec()
    schema = spec["components"]["schemas"]["Readiness"]
    spec_values = set(
        schema["properties"]["dependencies"]["items"]["properties"]["name"]["enum"]
    )
    reported = {
        dep["name"]
        for dep in validation_client.get("/api/v1/health/ready").json()["dependencies"]
    }
    assert spec_values == reported, (
        f"readiness dependency drift: spec={sorted(spec_values)}, "
        f"reported={sorted(reported)}"
    )


# --- Error code coverage (auto-discovered) ---


# Corpus-build codes are offline CLI only (invoked from main.py, never from the API).
_OFFLINE_EXCEPTION_CLASSES = frozenset({"CorpusBuildError"})


def _find_code_exception_classes() -> set[str]:
    """Find exception classes that assign self.code in __init__."""
    classes: set[str] = set()
    for py_file in _PACKAGE_DIR.rglob("*.py"):
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            for child in ast.walk(node):
                if isinstance(child, ast.Assign) and any(
                    isinstance(t, ast.Attribute)
                    and isinstance(t.value, ast.Name)
                    and t.value.id == "self"
                    and t.attr == "code"
                    for t in child.targets
                ):
                    classes.add(node.name)
    return classes


def _expand_subclasses(root_classes: set[str]) -> set[str]:
    """Expand to include subclasses defined in the package."""
    inheritance: dict[str, list[str]] = {}
    for py_file in _PACKAGE_DIR.rglob("*.py"):
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            bases: list[str] = []
            for base in node.bases:
                if isinstance(base, ast.Name):
                    bases.append(base.id)
                elif isinstance(base, ast.Attribute):
                    bases.append(base.attr)
            inheritance[node.name] = bases
    result = set(root_classes)
    changed = True
    while changed:
        changed = False
        for cls, bases in inheritance.items():
            if cls not in result and any(b in result for b in bases):
                result.add(cls)
                changed = True
    return result


def _collect_literal_codes(exception_classes: set[str]) -> dict[str, set[str]]:
    """Collect every literal string code passed as the first arg to these classes."""
    codes: dict[str, set[str]] = {cls: set() for cls in exception_classes}
    for py_file in _PACKAGE_DIR.rglob("*.py"):
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            name: str | None = None
            if isinstance(node.func, ast.Name):
                name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                name = node.func.attr
            if name not in exception_classes:
                continue
            first = node.args[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                codes[name].add(first.value)
    return codes


def test_error_codes_cover_all_package_exceptions() -> None:
    """Every error code raised in the package is in the spec, unless excluded."""
    spec = _load_spec()
    spec_codes = _spec_enum(spec, "ErrorCode")
    root_classes = _find_code_exception_classes()
    all_classes = _expand_subclasses(root_classes)
    codes_by_class = _collect_literal_codes(all_classes)
    api_codes: set[str] = set()
    for cls, codes in codes_by_class.items():
        if cls in _OFFLINE_EXCEPTION_CLASSES:
            continue
        api_codes.update(codes)
    assert api_codes, "found no error codes via AST walk"
    missing = api_codes - spec_codes
    assert not missing, f"ErrorCode enum missing codes: {sorted(missing)}"


def test_every_public_error_code_has_a_polish_message() -> None:
    """_MESSAGES_PL must cover the ErrorCode enum exactly.

    Both consumers index this table directly. Before, both used .get() with a
    generic fallback, so eleven documented codes silently produced "Wystąpił błąd
    wewnętrzny." with no test failure. Equality here is what makes the direct
    indexing safe.
    """
    from contract_analyzer.api.errors import _HTTP_STATUS, _MESSAGES_PL

    spec_codes = set(_load_spec()["components"]["schemas"]["ErrorCode"]["enum"])

    # model_response_missing_usage is a run-only code the attempt driver raises;
    # the wire contract does not list it, so the table may carry it alone.
    registered_codes = set(_MESSAGES_PL) - {"model_response_missing_usage"}

    assert registered_codes == spec_codes, (
        f"message table drift: missing={sorted(spec_codes - registered_codes)}, "
        f"unknown={sorted(registered_codes - spec_codes)}"
    )
    # Every code that maps to a status must also have a message to send with it.
    assert set(_HTTP_STATUS) <= set(_MESSAGES_PL)


# --- Response model field names ---


def test_response_model_fields_match_spec() -> None:
    """Pydantic response models returned by the API must have matching spec fields."""
    from contract_analyzer.agents.session import ChatResponse
    from contract_analyzer.agents.synthesizer import SynthesisResponse

    spec = _load_spec()
    schemas = spec["components"]["schemas"]
    models: dict[str, type] = {
        "ChatResponse": ChatResponse,
        "SynthesisResponse": SynthesisResponse,
    }
    problems: list[str] = []
    for schema_name, model_class in models.items():
        if schema_name not in schemas:
            continue
        spec_fields = set(schemas[schema_name].get("properties", {}).keys())
        py_fields = set(model_class.model_fields.keys())
        if spec_fields != py_fields:
            problems.append(
                f"{schema_name}: spec={sorted(spec_fields)}, python={sorted(py_fields)}"
            )
    assert not problems, "Response model field drift:\n" + "\n".join(problems)


# --- YAML 1.1 boolean trap ---


_YAML_11_BOOLEANS = re.compile(
    r"^(y|Y|yes|Yes|YES|n|N|no|No|NO|true|True|TRUE|false|False|FALSE"
    r"|on|On|ON|off|Off|OFF)$"
)


def test_no_enum_value_is_a_yaml_11_boolean() -> None:
    """Values that YAML 1.1 reads as booleans must be quoted in the source file.

    PyYAML's safe_load uses YAML 1.2 and preserves them as strings, so the only
    reliable check is that the raw file quotes every such value.
    """
    raw = _SPEC_PATH.read_text(encoding="utf-8")
    spec = _load_spec()
    all_enums = _all_spec_enums(spec)
    problems: list[str] = []
    for enum_path, values in all_enums.items():
        for value in values:
            if not isinstance(value, str):
                problems.append(
                    f"{enum_path}: {value!r} parsed as {type(value).__name__}, "
                    f"not string — unquoted YAML 1.1 boolean?"
                )
            elif _YAML_11_BOOLEANS.fullmatch(value):
                quoted = re.search(rf'["\x27]{re.escape(value)}["\x27]', raw)
                if quoted is None:
                    problems.append(
                        f"{enum_path}: {value!r} is a YAML 1.1 boolean and "
                        f"appears unquoted in the source file"
                    )
    assert not problems, "Enum values with YAML 1.1 boolean problems:\n" + "\n".join(
        problems
    )


# --- OpenAPI 3.1 validity ---


def test_spec_is_valid_openapi_31() -> None:
    """Structural validity: required OpenAPI 3.1 fields are present."""
    spec = _load_spec()
    assert spec.get("openapi", "").startswith("3.1"), "not OpenAPI 3.1"
    assert "info" in spec, "missing info"
    assert "title" in spec["info"], "missing info.title"
    assert "version" in spec["info"], "missing info.version"
    assert "paths" in spec, "missing paths"

    schemas = spec.get("components", {}).get("schemas", {})
    for name, schema in schemas.items():
        if "properties" in schema:
            for prop_name, prop in schema["properties"].items():
                if "$ref" in prop:
                    ref = prop["$ref"]
                    assert ref.startswith("#/"), (
                        f"{name}.{prop_name}: external $ref not supported: {ref}"
                    )
                    parts = ref.lstrip("#/").split("/")
                    target = spec
                    for part in parts:
                        assert isinstance(target, dict) and part in target, (
                            f"{name}.{prop_name}: broken $ref {ref}"
                        )
                        target = target[part]
