"""Routing: what the worksheet alone decides about who acts next.

``next_step`` is the whole order of work, so these tests are the specification of
the cycle. They fix the agreement between ``ROLE_HANDOFFS`` and ``next_step``:
every declared hand-off is one the scheduler can take, and back.
"""

from __future__ import annotations

import asyncio
import importlib
import subprocess
import sys
from datetime import date
from types import SimpleNamespace
from uuid import uuid4

import pytest

from contract_analyzer.agents import unit as unit_cycle
from contract_analyzer.agents.entries import CandidateEntry
from contract_analyzer.agents.errors import RunPipelineError
from contract_analyzer.agents.scheduler import (
    MAX_STEPS_PER_CANDIDATE,
    ROLE_HANDOFFS,
    TASK_MAX_TURNS,
    Step,
    next_step,
)
from contract_analyzer.agents.worksheet import Worksheet
from contract_analyzer.domain import (
    CharacterEvidence,
    ForceScope,
    ForceState,
    ForceValue,
    ProvisionCharacter,
)

LOCATOR = "https://api.sejm.gov.pl/eli/acts/DU/2023/725/text.html/arti=11"


def test_public_surface_imports_in_a_fresh_interpreter() -> None:
    """The names outside the package import from it, in a process of their own.

    Importing the packages alone would stay green if every one of these names
    were removed, which is what the entry points and scripts actually need.
    """
    program = "\n".join(
        [
            "from contract_analyzer.api import Prominence, cancel_run, create_app",
            "from contract_analyzer.serve import build_app, build_services",
            "from contract_analyzer.agents.runner import AnalysisRunner",
            "from contract_analyzer.agents.services import AnalysisServices",
            "from contract_analyzer.storage import MetadataStore, open_metadata_store",
            "from contract_analyzer.corpus import QdrantCorpusIndex",
            "print(create_app, build_app, AnalysisRunner, MetadataStore)",
        ]
    )
    result = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        check=False,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def _sheet(*, searched: bool = True) -> Worksheet:
    sheet = Worksheet(uuid4(), "unit-1")
    sheet.search_complete = searched
    return sheet


def _candidate(
    sheet: Worksheet,
    *,
    locator: str = LOCATOR,
    provision: ForceValue = ForceValue.UNDETERMINED,
) -> CandidateEntry:
    return sheet.add_candidate(
        locator=locator,
        act_identifier="DU/2023/725",
        snapshot_id="snapshot-test",
        act_force=ForceState(
            value=ForceValue.IN_FORCE,
            scope=ForceScope.ACT,
            snapshot_date=date(2026, 1, 1),
            source_locator="act",
        ),
        provision_force=ForceState(
            value=provision,
            scope=ForceScope.PROVISION,
            snapshot_date=date(2026, 1, 1),
            source_locator=locator,
        ),
        why="Przepis dotyczy kaucji.",
    )


def _character() -> ProvisionCharacter:
    return ProvisionCharacter(
        kind="imperative",
        evidence=(
            CharacterEvidence(
                source_kind="official_normative_text",
                locator=LOCATOR,
                pinpoint="ust. 1",
                interpretive_methods=("linguistic",),
                rationale="brzmienie imperatywne",
            ),
        ),
    )


def test_an_unsearched_unit_searches_first() -> None:
    assert next_step(_sheet(searched=False)) == Step(
        role="researcher", task="researcher.search"
    )


def test_a_unit_without_candidates_is_finished() -> None:
    assert next_step(_sheet()) is None


def test_a_posted_candidate_goes_to_the_verifier() -> None:
    sheet = _sheet()
    candidate = _candidate(sheet)

    assert next_step(sheet) == Step(
        role="verifier", task="verifier.relevance", candidate_id=candidate.id
    )


def test_the_verifier_never_returns_the_unit_to_the_researcher() -> None:
    """The challenge cycle was the only verifier-to-researcher edge.

    Asserting on ``ROLE_HANDOFFS`` alone would restate the declaration. These are
    the transitions the scheduler actually produces, over every shape of relevance
    verdict a unit can reach.
    """
    observed: set[tuple[str, str]] = set()
    for relevant in (
        [],
        [True],
        [False],
        [None],
        [True, True],
        [True, False],
        [False, None],
    ):
        observed.update(_drive(candidates=len(relevant), relevant=relevant))

    assert ("verifier", "researcher") not in observed
    assert "researcher" not in ROLE_HANDOFFS["verifier"]


def test_a_relevant_admitted_candidate_is_characterised_then_adjudicated() -> None:
    sheet = _sheet()
    candidate = _candidate(sheet)
    sheet.add_verdict(
        candidate_id=candidate.id, stage="relevance", relevant=True, quote="cytat"
    )

    assert next_step(sheet) == Step(
        role="analyst", task="analyst.characterise", candidate_id=candidate.id
    )

    sheet.add_character(candidate_id=candidate.id, character=_character())

    assert next_step(sheet) == Step(
        role="verifier", task="verifier.relation", candidate_id=candidate.id
    )

    sheet.add_verdict(candidate_id=candidate.id, stage="relation", departure="none")

    assert next_step(sheet) is None


@pytest.mark.parametrize("relevant", [False, None])
def test_a_candidate_the_verifier_did_not_accept_is_left_behind(
    relevant: bool | None,
) -> None:
    sheet = _sheet()
    candidate = _candidate(sheet)
    sheet.add_verdict(
        candidate_id=candidate.id,
        stage="relevance",
        relevant=relevant,
        quote="cytat",
    )

    assert next_step(sheet) is None


def test_a_candidate_its_force_records_veto_is_never_characterised() -> None:
    """The corpus decides admission, so a relevant but repealed provision stops."""
    sheet = _sheet()
    candidate = _candidate(sheet, provision=ForceValue.NOT_IN_FORCE)
    sheet.add_verdict(
        candidate_id=candidate.id, stage="relevance", relevant=True, quote="cytat"
    )

    assert next_step(sheet) is None


def test_candidates_are_worked_through_in_posting_order() -> None:
    sheet = _sheet()
    first = _candidate(sheet)
    second = _candidate(sheet, locator=LOCATOR + "#2")

    assert next_step(sheet) == Step(
        role="verifier", task="verifier.relevance", candidate_id=first.id
    )

    sheet.add_verdict(
        candidate_id=first.id, stage="relevance", relevant=False, quote="cytat"
    )

    assert next_step(sheet) == Step(
        role="verifier", task="verifier.relevance", candidate_id=second.id
    )


def test_routing_is_a_function_of_the_worksheet_alone() -> None:
    sheet = _sheet()
    candidate = _candidate(sheet)
    sheet.add_verdict(
        candidate_id=candidate.id, stage="relevance", relevant=True, quote="cytat"
    )
    before = sheet.entries

    assert next_step(sheet) == next_step(sheet)
    assert sheet.entries == before


def test_every_task_has_a_turn_limit() -> None:
    assert set(TASK_MAX_TURNS) == {
        "researcher.search",
        "analyst.characterise",
        "verifier.relevance",
        "verifier.relation",
        "synthesizer.group",
    }
    assert TASK_MAX_TURNS["analyst.characterise"] == 4
    assert TASK_MAX_TURNS["verifier.relation"] == 4
    assert TASK_MAX_TURNS["synthesizer.group"] == 4


def test_the_unit_step_bound_is_checked_before_another_role_acts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = SimpleNamespace(worksheet=_sheet(searched=False), pending_step=None)
    calls = 0

    def always_research(_: object) -> Step:
        return Step(role="researcher", task="researcher.search")

    async def act(_: object, __: Step) -> None:
        nonlocal calls
        calls += 1

    monkeypatch.setattr(unit_cycle, "next_step", always_research)
    monkeypatch.setattr(unit_cycle.researcher, "act", act)

    with pytest.raises(RunPipelineError, match="unit_graph_recursion_exceeded"):
        asyncio.run(unit_cycle.run(context))

    assert calls == 1


def _drive(*, candidates: int, relevant: list[bool | None]) -> list[tuple[str, str]]:
    """Run one unit to its end, recording which role each role handed off to.

    The scheduler is the only thing deciding here: each step is applied to the
    worksheet exactly as the corresponding role would leave it, then ``next_step``
    is asked again. What comes back is the transition set a real run can produce.
    """
    sheet = _sheet(searched=False)
    seen: list[tuple[str, str]] = []
    posted: list[CandidateEntry] = []
    step = next_step(sheet)
    while step is not None:
        if step.task == "researcher.search":
            sheet.search_complete = True
            posted = [
                _candidate(sheet, locator=f"{LOCATOR}/{index}")
                for index in range(candidates)
            ]
        elif step.task == "verifier.relevance":
            index = next(i for i, c in enumerate(posted) if c.id == step.candidate_id)
            sheet.add_verdict(
                candidate_id=posted[index].id,
                stage="relevance",
                relevant=relevant[index],
                quote="cytat",
            )
        elif step.task == "analyst.characterise":
            sheet.add_character(
                candidate_id=step.candidate_id or "", character=_character()
            )
        else:
            sheet.add_verdict(
                candidate_id=step.candidate_id or "",
                stage="relation",
                departure="none",
            )
        following = next_step(sheet)
        seen.append((step.role, "end" if following is None else following.role))
        step = following
    return seen


def test_the_declared_handoffs_are_exactly_the_ones_the_scheduler_can_take() -> None:
    """A declared edge no worksheet reaches is a diagram, not an agreement.

    ``ROLE_HANDOFFS`` declares a destination list per role. Every one of those
    has to be a hand-off ``next_step`` actually produces, and every hand-off it
    produces has to be declared, or the declaration and the scheduler disagree
    about what the system does.
    """
    observed: set[tuple[str, str]] = set()
    for candidates, relevant in (
        (0, []),
        (1, [True]),
        (1, [False]),
        (1, [None]),
        (2, [False, True]),
        (2, [False, False]),
        (2, [True, False]),
    ):
        observed.update(_drive(candidates=candidates, relevant=relevant))

    reachable: dict[str, set[str]] = {}
    for role, destination in observed:
        reachable.setdefault(role, set()).add(destination)

    assert reachable == {
        role: set(destinations) for role, destinations in ROLE_HANDOFFS.items()
    }


def test_a_unit_costs_one_search_and_at_most_three_steps_per_candidate() -> None:
    """The bound has to be what the scheduler does, not what a constant says.

    A candidate cleared as relevant is the longest path it can take: a relevance
    visit, a characterisation and a relation verdict. Every other verdict ends it
    sooner. Driving both is what makes ``MAX_STEPS_PER_CANDIDATE`` an agreement
    with the routing rather than a number kept beside it.
    """
    assert MAX_STEPS_PER_CANDIDATE == 3

    for candidates in range(4):
        longest = _drive(candidates=candidates, relevant=[True] * candidates)
        assert len(longest) == 1 + MAX_STEPS_PER_CANDIDATE * candidates

    for relevant in ([False], [None], [True, False], [False, None], [None, True]):
        taken = _drive(candidates=len(relevant), relevant=relevant)
        assert len(taken) <= 1 + MAX_STEPS_PER_CANDIDATE * len(relevant)


@pytest.mark.parametrize(
    ("module_name", "role", "task"),
    [
        ("researcher", "researcher", "analyst.characterise"),
        ("analyst", "analyst", "researcher.search"),
        ("verifier", "verifier", "researcher.search"),
    ],
)
def test_a_role_refuses_a_task_that_is_not_its_own(
    module_name: str, role: str, task: str
) -> None:
    """A renamed task must fail loudly rather than run the wrong handler.

    Each role used to fall through to one of its own tasks for anything it did not
    recognise, so a task added or renamed later would have been handled by the
    wrong code path and produced a plausible worksheet entry instead of an error.
    """
    module = importlib.import_module(f"contract_analyzer.agents.{module_name}")
    step = Step(role=role, task=task, candidate_id="e1")  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match=f"{role} node reached with {task}"):
        asyncio.run(module.act(object(), step))


def test_every_package_exports_what_it_names() -> None:
    """A name left in __all__ after its import went is a broken star import.

    Shrinking the packages removed imports; one export list kept naming three
    of them, so `from contract_analyzer.ingest import *` would have failed.
    """
    import importlib

    packages = [
        "agents",
        "api",
        "corpus",
        "domain",
        "ingest",
        "model",
        "storage",
        "structure",
    ]
    absent: dict[str, list[str]] = {}
    for name in packages:
        module = importlib.import_module(f"contract_analyzer.{name}")
        missing = [n for n in getattr(module, "__all__", []) if not hasattr(module, n)]
        if missing:
            absent[name] = missing

    assert absent == {}, (
        f"__all__ names something the package does not export: {absent}"
    )
