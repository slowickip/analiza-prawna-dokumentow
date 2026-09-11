"""The worksheet, the views each role gets of it, and the tool surface.

Pure tests: no model, no corpus, no run. They fix what the record guarantees --
order, per-role visibility, the locators a verifier may follow -- and that the
parity token moves when the tool surface or a bound does.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import date
from typing import Any
from uuid import uuid4

import pytest
from pydantic import ValidationError

from contract_analyzer.agents import tools as tools_module
from contract_analyzer.agents.entries import (
    CandidateEntry,
    ReadEntry,
    admits,
    retrieval_candidate,
)
from contract_analyzer.agents.prompts import TASK_PROMPTS, load_prompt_bundle
from contract_analyzer.agents.tool_args import PLACEHOLDER_LOCATOR, CandidateArgs
from contract_analyzer.agents.tools import (
    TASK_TOOLS,
    TOOL_BUNDLE_VERSION,
    PostCandidatesArgs,
    PostCharacterArgs,
    RelationVerdictArgs,
    RelevanceVerdictArgs,
    ToolSpec,
    commit_tool_for,
    definitions_for,
    parse_tool,
    tools_for,
)
from contract_analyzer.agents.worksheet import Worksheet, WorksheetRecord
from contract_analyzer.domain import (
    CharacterEvidence,
    ForceScope,
    ForceState,
    ForceValue,
    ProvisionCharacter,
)

ACT = "DU/2023/725"
LOCATOR = "https://api.sejm.gov.pl/eli/acts/DU/2023/725/text.html/arti=11"
OTHER_LOCATOR = "https://api.sejm.gov.pl/eli/acts/DU/2023/725/text.html/arti=19a"


def _force(value: ForceValue, scope: ForceScope, locator: str) -> ForceState:
    return ForceState(
        value=value,
        scope=scope,
        snapshot_date=date(2026, 1, 1),
        source_locator=locator,
    )


def _worksheet(**kwargs: object) -> Worksheet:
    return Worksheet(uuid4(), "unit-1", **kwargs)  # type: ignore[arg-type]


def _candidate(
    sheet: Worksheet,
    *,
    locator: str = LOCATOR,
    provision: ForceValue = ForceValue.UNDETERMINED,
    supporting: tuple[str, ...] = (),
) -> CandidateEntry:
    return sheet.add_candidate(
        locator=locator,
        act_identifier=ACT,
        snapshot_id="snapshot-test",
        act_force=_force(ForceValue.IN_FORCE, ForceScope.ACT, "act"),
        provision_force=_force(provision, ForceScope.PROVISION, locator),
        why="Przepis dotyczy kaucji.",
        supporting_locators=supporting,
    )


def _character(locator: str = LOCATOR) -> ProvisionCharacter:
    return ProvisionCharacter(
        kind="imperative",
        evidence=(
            CharacterEvidence(
                source_kind="official_normative_text",
                locator=locator,
                pinpoint="ust. 1",
                interpretive_methods=("linguistic",),
                rationale="brzmienie imperatywne",
            ),
        ),
    )


def test_entries_are_numbered_in_append_order() -> None:
    sheet = _worksheet()
    candidate = _candidate(sheet)
    character = sheet.add_character(candidate_id=candidate.id, character=_character())
    verdict = sheet.add_verdict(
        candidate_id=candidate.id, stage="relevance", relevant=True, quote="cytat"
    )

    assert [entry.id for entry in sheet.entries] == ["e1", "e2", "e3"]
    assert [entry.seq for entry in sheet.entries] == [1, 2, 3]
    assert [entry.kind for entry in sheet.entries] == [
        "candidate",
        "character",
        "verdict",
    ]
    assert [entry.author for entry in sheet.entries] == [
        "researcher",
        "analyst",
        "verifier",
    ]
    assert character.candidate_id == candidate.id
    assert verdict.stage == "relevance"


def test_every_append_is_announced_once() -> None:
    seen: list[str] = []
    sheet = _worksheet(on_append=lambda entry: seen.append(entry.kind))
    candidate = _candidate(sheet)
    sheet.add_read(author="verifier", locator=LOCATOR)
    sheet.add_search(phrase="kaucja", result_locators=(LOCATOR,))
    sheet.add_character(candidate_id=candidate.id, character=_character())

    assert seen == ["candidate", "read", "search", "character"]


def test_queries_find_what_belongs_to_one_candidate() -> None:
    sheet = _worksheet()
    first = _candidate(sheet)
    second = _candidate(sheet, locator=OTHER_LOCATOR)
    sheet.add_character(candidate_id=first.id, character=_character())
    sheet.add_verdict(candidate_id=first.id, stage="relevance", relevant=True)

    assert sheet.candidates() == (first, second)
    assert sheet.candidate(second.id) == second
    assert sheet.candidate("e99") is None
    assert sheet.character_for(first.id) is not None
    assert sheet.character_for(second.id) is None
    assert sheet.verdict_for(first.id, "relevance") is not None
    assert sheet.verdict_for(first.id, "relation") is None


def test_analyst_sees_the_whole_unit() -> None:
    sheet = _worksheet()
    _candidate(sheet)
    sheet.add_search(phrase="kaucja", result_locators=(LOCATOR,))

    rendered = sheet.render("analyst")

    assert [entry["kind"] for entry in rendered] == ["candidate", "search"]


def test_verifier_sees_only_the_candidate_under_review() -> None:
    """A verdict that saw the shortlist would be about the shortlist."""
    sheet = _worksheet()
    first = _candidate(sheet)
    second = _candidate(sheet, locator=OTHER_LOCATOR)
    sheet.add_search(phrase="kaucja", result_locators=(LOCATOR, OTHER_LOCATOR))
    sheet.add_read(author="analyst", locator=LOCATOR)
    sheet.add_character(candidate_id=first.id, character=_character())
    sheet.add_verdict(candidate_id=second.id, stage="relevance", relevant=False)
    sheet.add_user_note(note="uwaga czytelnika", purpose="clarify")

    rendered = sheet.render("verifier", first.id)
    kinds = [entry["kind"] for entry in rendered]

    assert kinds == [
        "candidate",
        "character",
        "user_note",
    ]
    assert all(entry.get("id") != second.id for entry in rendered)
    assert "search" not in kinds
    assert "read" not in kinds


def test_verifier_projection_hides_analyst_prose() -> None:
    sentinel = "sentinel_analyst_prose_must_be_blind"
    sheet = _worksheet()
    candidate = sheet.add_candidate(
        locator=LOCATOR,
        act_identifier=ACT,
        snapshot_id="snapshot-test",
        act_force=_force(ForceValue.IN_FORCE, ForceScope.ACT, "act"),
        provision_force=_force(ForceValue.IN_FORCE, ForceScope.PROVISION, LOCATOR),
        why=sentinel,
    )

    rendered = sheet.render("verifier", candidate.id)

    assert sentinel not in str(rendered)
    assert rendered[0]["locator"] == LOCATOR


def test_verifier_without_a_candidate_sees_nothing_but_user_notes() -> None:
    sheet = _worksheet()
    _candidate(sheet)
    sheet.add_user_note(note="uwaga", purpose="clarify")

    assert [entry["kind"] for entry in sheet.render("verifier")] == ["user_note"]


def test_known_locators_are_the_ones_this_unit_already_names() -> None:
    sheet = _worksheet()
    candidate = _candidate(sheet, supporting=(OTHER_LOCATOR,))
    other = _candidate(sheet, locator="locator-of-other-candidate")
    sheet.add_character(
        candidate_id=candidate.id, character=_character("evidence-locator")
    )
    sheet.add_character(
        candidate_id=other.id, character=_character("someone-elses-locator")
    )

    known = sheet.known_locators(candidate.id)

    assert known == frozenset({LOCATOR, OTHER_LOCATOR, "evidence-locator"})
    assert "someone-elses-locator" not in known


def test_record_round_trips_through_json() -> None:
    sheet = _worksheet()
    candidate = _candidate(sheet)
    sheet.add_character(candidate_id=candidate.id, character=_character())
    sheet.add_verdict(
        candidate_id=candidate.id,
        stage="relation",
        departure="none",
        raw_confidence=0.5,
    )
    sheet.search_complete = True

    restored = WorksheetRecord.model_validate_json(sheet.record().model_dump_json())

    assert restored == sheet.record()
    assert restored.search_complete is True
    assert [entry.kind for entry in restored.entries] == [
        "candidate",
        "character",
        "verdict",
    ]


def test_an_entry_cannot_carry_another_kinds_field() -> None:
    """A closed entry keeps the serialised worksheet readable back as it was."""
    with pytest.raises(ValidationError):
        ReadEntry(
            id="e1",
            seq=1,
            author="analyst",
            locator=LOCATOR,
            phrase="kaucja",  # type: ignore[call-arg]
        )


def test_candidate_force_records_decide_admission() -> None:
    sheet = _worksheet()
    admitted = _candidate(sheet)
    vetoed = _candidate(sheet, locator=OTHER_LOCATOR, provision=ForceValue.NOT_IN_FORCE)

    assert admits(admitted) is True
    assert admits(vetoed) is False
    lifted = retrieval_candidate(admitted, rank=1)
    assert lifted.locator == admitted.locator
    assert lifted.act_force == admitted.act_force
    assert lifted.provision_force == admitted.provision_force


def test_each_task_offers_the_tools_its_role_may_call() -> None:
    assert [spec.name for spec in tools_for("researcher.search")] == [
        "search_corpus",
        "read_provision",
        "post_candidates",
    ]
    assert [spec.name for spec in tools_for("analyst.characterise")] == [
        "read_provision",
        "post_character",
    ]
    assert "search_corpus" not in [
        spec.name for spec in tools_for("verifier.relevance")
    ]
    assert "search_corpus" not in [spec.name for spec in tools_for("verifier.relation")]


def test_search_has_an_optional_terminal_commit() -> None:
    assert commit_tool_for("researcher.search") == "post_candidates"
    assert commit_tool_for("analyst.characterise") == "post_character"
    assert commit_tool_for("verifier.relevance") == "post_verdict"
    assert commit_tool_for("verifier.relation") == "post_verdict"
    assert commit_tool_for("synthesizer.group") == "group_findings"


def test_commit_tool_descriptions_exclude_task_scope_identifiers() -> None:
    for specs in TASK_TOOLS.values():
        for spec in (item for item in specs if item.commit):
            expected = "Identyfikator zakresu zadania nie jest argumentem"
            assert expected in spec.description


def test_the_same_tool_name_is_validated_per_task() -> None:
    """post_verdict means different arguments to the two verifier tasks."""
    relevance = next(
        spec for spec in TASK_TOOLS["verifier.relevance"] if spec.name == "post_verdict"
    )
    assert relevance.arguments is RelevanceVerdictArgs
    relation = next(
        spec for spec in TASK_TOOLS["verifier.relation"] if spec.name == "post_verdict"
    )
    assert relation.arguments is not RelevanceVerdictArgs
    assert all(spec.name != "post_verdict" for spec in TASK_TOOLS["researcher.search"])


def test_tool_definitions_are_the_argument_schemas() -> None:
    definitions = definitions_for(tools_for("researcher.search"))
    by_name = {definition.name: definition for definition in definitions}

    assert by_name["post_candidates"].parameters == (
        PostCandidatesArgs.model_json_schema()
    )
    assert by_name["post_candidates"].description


def test_invalid_arguments_become_a_refusal_the_model_can_correct() -> None:
    parsed = TASK_TOOLS["researcher.search"][-1].parse(
        '{"candidates": [{"locator": "x"}]}'
    )

    assert isinstance(parsed, dict)
    assert parsed["error"] == "invalid_arguments"
    assert "why" in str(parsed["detail"])
    detail = parsed["detail"]
    assert isinstance(detail, list)
    assert all(
        isinstance(item, dict) and set(item) == {"field", "type", "msg"}
        for item in detail
    )


def test_string_valued_evidence_refusal_does_not_echo_the_input(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING)
    secret_prose = "tajne_uzasadnienie_w_postaci_tekstu"
    parsed = TASK_TOOLS["analyst.characterise"][-1].parse(
        json.dumps({"kind": "imperative", "evidence": secret_prose}),
    )

    assert isinstance(parsed, dict)
    assert parsed["error"] == "invalid_arguments"
    detail = parsed["detail"]
    assert isinstance(detail, list)
    evidence_error = next(item for item in detail if item["field"] == "evidence")
    assert set(evidence_error) == {"field", "type", "msg"}
    assert secret_prose not in json.dumps(parsed, ensure_ascii=False)
    assert secret_prose not in caplog.text


def test_rejected_arguments_never_reach_warning_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    sentinel = "sentinel_rejected_argument_must_not_be_logged"
    caplog.set_level(logging.WARNING)

    parsed = TASK_TOOLS["verifier.relevance"][-1].parse(
        '{"relevant": true, "quote": "x", "unexpected": "' + sentinel + '"}',
    )

    assert isinstance(parsed, dict)
    assert parsed["error"] == "invalid_arguments"
    assert "Extra inputs are not permitted" in str(parsed)
    assert sentinel not in str(parsed)
    assert sentinel not in caplog.text
    assert "Extra inputs are not permitted" not in caplog.text


def test_an_unknown_tool_name_is_refused() -> None:
    assert parse_tool(TASK_TOOLS["researcher.search"], "post_verdict", "{}") == {
        "error": "unknown_tool"
    }


def test_valid_arguments_parse_into_their_model() -> None:
    parsed = TASK_TOOLS["verifier.relevance"][-1].parse(
        '{"relevant": true, "quote": "cytat", "raw_confidence": 0.9, "based_on": []}',
    )

    assert isinstance(parsed, RelevanceVerdictArgs)
    assert parsed.relevant is True
    assert parsed.based_on == ()


def _evidence(**overrides: object) -> dict[str, object]:
    return {
        "source_kind": "official_normative_text",
        "locator": LOCATOR,
        "pinpoint": "art. 1",
        "interpretive_methods": ["linguistic"],
        "rationale": "brzmienie przepisu",
        **overrides,
    }


@pytest.mark.parametrize(
    "evidence",
    [
        _evidence(),
        json.dumps(_evidence()),
        json.dumps([_evidence()]),
    ],
)
def test_character_evidence_accepts_one_unambiguous_object(
    evidence: object,
) -> None:
    parsed = PostCharacterArgs.model_validate(
        {"kind": "imperative", "evidence": evidence}
    )

    assert len(parsed.evidence) == 1
    assert parsed.evidence[0].locator == LOCATOR


def test_character_evidence_wraps_one_interpretive_method() -> None:
    parsed = PostCharacterArgs.model_validate(
        {
            "kind": "imperative",
            "evidence": _evidence(interpretive_methods="linguistic"),
        }
    )

    assert parsed.evidence[0].interpretive_methods == ("linguistic",)


@pytest.mark.parametrize(
    ("model", "payload", "field"),
    [
        (
            CandidateArgs,
            {"locator": LOCATOR, "why": "powód", "supporting_locators": OTHER_LOCATOR},
            "supporting_locators",
        ),
        (
            RelevanceVerdictArgs,
            {
                "relevant": True,
                "quote": "cytat",
                "raw_confidence": 0.9,
                "based_on": "e1",
            },
            "based_on",
        ),
        (
            RelationVerdictArgs,
            {"departure": "none", "based_on": "e1"},
            "based_on",
        ),
    ],
)
def test_commit_string_collections_accept_one_string(
    model: Any, payload: dict[str, object], field: str
) -> None:
    parsed = model.model_validate(payload)

    assert getattr(parsed, field) == (
        OTHER_LOCATOR if field == "supporting_locators" else payload[field],
    )


@pytest.mark.parametrize(
    ("model", "payload"),
    [
        (PostCharacterArgs, {"kind": "imperative", "evidence": 7}),
        (
            PostCharacterArgs,
            {
                "kind": "imperative",
                "evidence": _evidence(interpretive_methods={"method": "linguistic"}),
            },
        ),
        (
            CandidateArgs,
            {"locator": LOCATOR, "why": "powód", "supporting_locators": {"x": 1}},
        ),
        (
            RelevanceVerdictArgs,
            {"relevant": True, "quote": "cytat", "based_on": {"x": 1}},
        ),
        (
            RelationVerdictArgs,
            {"departure": "none", "based_on": {"x": 1}},
        ),
    ],
)
def test_commit_collection_coercions_keep_other_shapes_strict(
    model: Any, payload: dict[str, object]
) -> None:
    with pytest.raises(ValidationError):
        model.model_validate(payload)


def _schema_fields(schema: dict[str, Any]) -> tuple[set[str], set[str]]:
    properties: set[str] = set()
    required: set[str] = set()
    seen_refs: set[str] = set()

    def visit(node: object) -> None:
        if not isinstance(node, dict):
            return
        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/$defs/"):
            if ref in seen_refs:
                return
            seen_refs.add(ref)
            visit(schema["$defs"][ref.removeprefix("#/$defs/")])
        node_properties = node.get("properties", {})
        if isinstance(node_properties, dict):
            properties.update(node_properties)
            for child in node_properties.values():
                visit(child)
        node_required = node.get("required", [])
        if isinstance(node_required, list):
            required.update(str(field) for field in node_required)
        visit(node.get("items"))
        for key in ("anyOf", "oneOf", "allOf"):
            variants = node.get(key, [])
            if isinstance(variants, list):
                for variant in variants:
                    visit(variant)

    visit(schema)
    return properties, required


def test_commit_arguments_listed_in_prompts_match_their_schemas() -> None:
    prompts = load_prompt_bundle().prompts
    for task, specs in TASK_TOOLS.items():
        prompt = prompts[TASK_PROMPTS[task]]
        for commit in (spec for spec in specs if spec.commit):
            marker = f"Argumenty narzędzia `{commit.name}`:"
            assert marker in prompt, task
            section = prompt.split(marker, 1)[1].split("\n\n", 1)[0]
            mentioned = set(re.findall(r"(?m)^\s*- `([a-z][a-z0-9_]*)`", section))
            after_marker = prompt.split(marker, 1)[1]
            mentioned.update(re.findall(r"`([a-z][a-z0-9_]*_id)`", after_marker))
            properties, required = _schema_fields(commit.arguments.model_json_schema())

            assert mentioned <= properties, (task, mentioned - properties)
            assert required <= mentioned, (task, required - mentioned)


def test_post_candidates_preserves_more_than_five_entries() -> None:
    payload = {
        "candidates": [
            {"locator": f"locator-{index}", "why": "powód"} for index in range(6)
        ]
    }

    parsed = TASK_TOOLS["researcher.search"][-1].parse(json.dumps(payload))

    assert isinstance(parsed, PostCandidatesArgs)
    assert [candidate.locator for candidate in parsed.candidates] == [
        item["locator"] for item in payload["candidates"]
    ]


def test_post_candidates_accepts_an_empty_list_as_a_stated_result() -> None:
    """No basis in the corpus is one of the results, so it has to be sayable."""
    parsed = TASK_TOOLS["researcher.search"][-1].parse(json.dumps({"candidates": []}))

    assert isinstance(parsed, PostCandidatesArgs)
    assert parsed.candidates == ()


def test_the_character_argument_model_keeps_the_closed_field_rules() -> None:
    with pytest.raises(ValidationError):
        PostCharacterArgs(kind="undetermined", evidence=())
    with pytest.raises(ValidationError):
        PostCharacterArgs(
            kind="semi_imperative",
            evidence=_character().evidence,
        )


@pytest.mark.parametrize(
    "bound",
    [
        "MAX_STEPS_PER_CANDIDATE",
        "VERIFIER_MAX_READS_PER_CANDIDATE",
        "FINDER_MAX_PHRASES_PER_TURN",
        "FINDER_MAX_PHRASE_CHARS",
        "FINDER_SNIPPET_CHARS",
        "RETRIEVAL_TOP_K",
    ],
)
def test_tool_bundle_version_moves_with_every_bound(
    bound: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Parity can only catch a policy change if the token it compares tracks one."""
    baseline = tools_module._tool_bundle_version()
    assert baseline == TOOL_BUNDLE_VERSION

    monkeypatch.setattr(tools_module, bound, getattr(tools_module, bound) + 1)

    assert tools_module._tool_bundle_version() != baseline


def test_tool_bundle_version_moves_with_a_turn_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = tools_module._tool_bundle_version()
    turns = dict(tools_module.TASK_MAX_TURNS)
    turns["researcher.search"] = turns["researcher.search"] + 1

    monkeypatch.setattr(tools_module, "TASK_MAX_TURNS", turns)

    assert tools_module._tool_bundle_version() != baseline


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("VERIFIER_PROJECTION_FIELDS", ("payload.unit_id",)),
        ("VERIFIER_READ_GUARD_RULE_ID", "changed-read-rule"),
        ("COMMIT_RULE_ID", "changed-commit-rule"),
        ("SCHEDULER_RULES_VERSION", "changed-scheduler-rules"),
        ("PROVENANCE_RULE_ID", "changed-provenance-rule"),
        ("SEARCH_CONCLUSION_GUARD_RULE_ID", "changed-search-conclusion-rule"),
        ("ARGUMENT_REFUSAL_RULE_ID", "changed-argument-rule"),
    ],
)
def test_tool_bundle_version_moves_with_analysis_policy(
    field: str, replacement: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline = tools_module._tool_bundle_version()
    monkeypatch.setattr(tools_module, field, replacement)
    assert tools_module._tool_bundle_version() != baseline


def test_tool_bundle_version_moves_with_task_authorisation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = tools_module._tool_bundle_version()
    tasks = dict(tools_module.TASK_TOOLS)
    tasks["researcher.search"] = tasks["researcher.search"][:-1]
    monkeypatch.setattr(tools_module, "TASK_TOOLS", tasks)
    assert tools_module._tool_bundle_version() != baseline


@pytest.mark.parametrize("field", ["name", "description", "arguments"])
def test_tool_bundle_version_moves_with_a_tool_definition(
    field: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline = tools_module._tool_bundle_version()
    original = tools_module.TASK_TOOLS["analyst.characterise"]
    changed: ToolSpec
    if field == "name":
        changed = ToolSpec(
            name="read_provisions",
            description=original[0].description,
            arguments=original[0].arguments,
        )
    elif field == "description":
        changed = ToolSpec(
            name=original[0].name,
            description=original[0].description + " Dopisek.",
            arguments=original[0].arguments,
        )
    else:
        changed = ToolSpec(
            name=original[0].name,
            description=original[0].description,
            arguments=PostCandidatesArgs,
        )
    tasks = dict(tools_module.TASK_TOOLS)
    tasks["analyst.characterise"] = (changed, *original[1:])

    monkeypatch.setattr(tools_module, "TASK_TOOLS", tasks)

    assert tools_module._tool_bundle_version() != baseline


def test_tool_bundle_version_is_one_token_for_every_arm() -> None:
    """The arms differ only in the call unit, so the tool surface cannot differ."""
    assert TOOL_BUNDLE_VERSION == tools_module._tool_bundle_version()
    assert TOOL_BUNDLE_VERSION.startswith("worksheet-roles-")


def test_tool_bundle_version_policy_digest_is_deliberately_pinned() -> None:
    # This literal forces every analysis-policy change to be acknowledged here.
    assert TOOL_BUNDLE_VERSION == "worksheet-roles-282366be7762"


def _schema_examples(schema: dict[str, Any]) -> list[tuple[str, object]]:
    """Every example value in a tool schema, paired with the field that carries it."""
    found: list[tuple[str, object]] = []

    def walk(node: object, field: str) -> None:
        if isinstance(node, dict):
            for example in node.get("examples", ()):
                found.append((field, example))
            for key, value in node.items():
                walk(value, key if key not in {"items", "properties"} else field)
        elif isinstance(node, list):
            for value in node:
                walk(value, field)

    walk(schema, "")
    return found


def test_every_locator_example_is_the_placeholder() -> None:
    """An example reaches the model as a value it may copy into a commit.

    A real locator is the dangerous kind: copied into a characterisation of a
    different provision it passes the existence check and commits silently. Every
    locator example is therefore the one placeholder, whose non-resolvability
    against the corpus is pinned in test_graph_failures.
    """
    for task, specs in TASK_TOOLS.items():
        for spec in specs:
            for field, example in _schema_examples(spec.arguments.model_json_schema()):
                for value in example if isinstance(example, list) else [example]:
                    locator = value.get("locator") if isinstance(value, dict) else value
                    if not isinstance(locator, str) or "eli/acts" not in locator:
                        continue
                    assert locator == PLACEHOLDER_LOCATOR, (task, field)


def test_based_on_examples_are_entry_ids_not_locators() -> None:
    """The handler resolves based_on against worksheet entry ids, so examples must."""
    for args in (RelevanceVerdictArgs, RelationVerdictArgs):
        examples = args.model_json_schema()["properties"]["based_on"]["examples"]
        values = [value for example in examples for value in example]
        assert values
        assert all(re.fullmatch(r"e\d+", value) for value in values), args


def test_the_document_hash_follows_the_units_it_describes() -> None:
    """It is a parity dimension in the run record, so it must not go stale.

    Retention purging replaces the session's units, so a stored copy would
    describe a segmentation the session no longer has. Deriving it on read
    costs about twenty microseconds across a run.
    """
    import hashlib

    from graph_harness import DOCUMENT_TEXT, make_session

    session = make_session(DOCUMENT_TEXT)
    unit_fingerprint = "|".join(
        f"{unit.id}:{unit.anchor.start_offset}:{unit.anchor.end_offset}"
        for unit in session.units
    )
    material = (
        f"{session.payload.content_hash}|{session.payload.read_mode}|{unit_fingerprint}"
    )
    assert session.input_hash == hashlib.sha256(material.encode()).hexdigest()

    first = session.units[0]
    session.units = (
        first.model_copy(
            update={
                "anchor": first.anchor.model_copy(
                    update={"end_offset": first.anchor.end_offset + 1}
                )
            }
        ),
        *session.units[1:],
    )
    assert session.input_hash != hashlib.sha256(material.encode()).hexdigest(), (
        "the hash kept describing the segmentation the session no longer has"
    )


def test_a_tool_definition_cannot_be_edited_for_every_later_run() -> None:
    """The schema is cached, so each turn must be handed its own copy.

    ToolDefinition is frozen only at the top level. Sharing one object would
    let a caller edit the nested schema every later run is offered, while the
    recorded bundle digest, rebuilt from the models, went on saying otherwise.
    """
    from contract_analyzer.agents.tools import TASK_TOOLS, TOOL_BUNDLE_VERSION

    spec = next(iter(next(iter(TASK_TOOLS.values()))))
    first = spec.definition()
    first.parameters["properties"]["__intruder__"] = {"type": "string"}

    second = spec.definition()
    assert "__intruder__" not in second.parameters["properties"], (
        "one caller's edit reached every later turn"
    )
    assert TOOL_BUNDLE_VERSION == "worksheet-roles-282366be7762"
