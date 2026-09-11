"""Who acts next, decided from the worksheet alone.

``next_step`` is a pure function of one worksheet: given what the roles have
written so far it names the role, the task and the candidate under review. The
unit driver therefore cycles between three roles, routing from the worksheet
after each action.

The researcher finds candidates, the analyst characterises the ones
that survive relevance, and the verifier rules on both questions. No role decides
that its own output was good: the researcher cannot clear its candidate, and the
analyst cannot rule on the relation it characterised for.

Every bound here is a policy choice; all of them feed ``TOOL_BUNDLE_VERSION``, so
a run whose roles were allowed more turns than another's is not silently
comparable with it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from contract_analyzer.agents.entries import admits
from contract_analyzer.agents.worksheet import Worksheet

Role = Literal["researcher", "analyst", "verifier"]
Task = Literal[
    "researcher.search",
    "analyst.characterise",
    "verifier.relevance",
    "verifier.relation",
    "synthesizer.group",
]

# Chosen by argument, not measurement: three reads permit bounded verification.
VERIFIER_MAX_READS_PER_CANDIDATE = 3
# Chosen by argument, not measurement: enough room for reformulation without
# letting a whole-document call turn the ceiling into a quota.
FINDER_MAX_TOOL_TURNS = 10
FINDER_MAX_PHRASES_PER_TURN = 5
# Chosen by argument, not measurement: a search phrase is not a document.
FINDER_MAX_PHRASE_CHARS = 500
# Chosen by argument, not measurement: snippets only support candidate selection.
FINDER_SNIPPET_CHARS = 300
# Chosen by argument, not measurement: five candidates balance prompt size against
# the chance of omitting a relevant provision.
RETRIEVAL_TOP_K = 5
# Chosen by argument, not measurement: bounded retries permit one corrected commit.
TASK_MAX_TURNS: Mapping[Task, int] = {
    "researcher.search": FINDER_MAX_TOOL_TURNS,
    "analyst.characterise": 4,
    "verifier.relevance": 3,
    "verifier.relation": 4,
    # Chosen by argument, not measurement: a listing, a grouping, and room for
    # one refusal and its correction.
    "synthesizer.group": 4,
}
SCHEDULER_RULES_VERSION = "worksheet-scheduler-rules-v4"
# Per candidate: at most one relevance visit, one characterisation and one relation
# task. The unit adds one initial search to the actual count.
MAX_STEPS_PER_CANDIDATE = 1 + 1 + 1


@dataclass(frozen=True)
class Step:
    role: Role
    task: Task
    candidate_id: str | None = None


def next_step(worksheet: Worksheet) -> Step | None:
    """The next role and task this unit needs, or None when it is finished.

    Deterministic and side-effect free, so the same worksheet always routes the
    same way and a replayed run takes the same path. Candidates are considered in
    posting order, and a candidate is only left behind once nothing about it is
    outstanding.
    """
    if not worksheet.search_complete:
        return Step(role="researcher", task="researcher.search")
    for candidate in worksheet.candidates():
        relevance = worksheet.verdict_for(candidate.id, "relevance")
        if relevance is None:
            return Step(
                role="verifier",
                task="verifier.relevance",
                candidate_id=candidate.id,
            )
        if relevance.relevant is not True or not admits(candidate):
            continue
        character = worksheet.character_for(candidate.id)
        if character is None:
            return Step(
                role="analyst",
                task="analyst.characterise",
                candidate_id=candidate.id,
            )
        if worksheet.verdict_for(candidate.id, "relation") is None:
            return Step(
                role="verifier",
                task="verifier.relation",
                candidate_id=candidate.id,
            )
    return None


# The thesis figure and scheduler test derive the possible role handoffs from this.
ROLE_HANDOFFS: Mapping[Role, tuple[str, ...]] = {
    "researcher": ("verifier", "end"),
    "analyst": ("verifier",),
    "verifier": ("analyst", "verifier", "end"),
}
