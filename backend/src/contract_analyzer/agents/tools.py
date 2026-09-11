"""Agent tool schemas, task authorisation and their parity digest."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

from pydantic import BaseModel

from contract_analyzer.agents.scheduler import (
    FINDER_MAX_PHRASE_CHARS,
    FINDER_MAX_PHRASES_PER_TURN,
    FINDER_MAX_TOOL_TURNS,
    FINDER_SNIPPET_CHARS,
    MAX_STEPS_PER_CANDIDATE,
    RETRIEVAL_TOP_K,
    SCHEDULER_RULES_VERSION,
    TASK_MAX_TURNS,
    VERIFIER_MAX_READS_PER_CANDIDATE,
    Task,
)
from contract_analyzer.agents.schema import ToolSpec
from contract_analyzer.agents.tool_args import (
    GroupFindingsArgs,
    ListFindingsArgs,
    PostCandidatesArgs,
    PostCharacterArgs,
    ReadProvisionArgs,
    RelationVerdictArgs,
    RelevanceVerdictArgs,
    SearchCorpusArgs,
)
from contract_analyzer.agents.views import VERIFIER_PROJECTION_FIELDS
from contract_analyzer.model import ToolDefinition

__all__ = (
    "GroupFindingsArgs",
    "ListFindingsArgs",
    "PostCandidatesArgs",
    "PostCharacterArgs",
    "ReadProvisionArgs",
    "RelationVerdictArgs",
    "RelevanceVerdictArgs",
    "SearchCorpusArgs",
    "ToolSpec",
)

VERIFIER_READ_GUARD_RULE_ID = "worksheet-locators-only-candidate-cap-v1"
COMMIT_RULE_ID = "commit-alone-atomic-turn-v3"
ARGUMENT_REFUSAL_RULE_ID = "argument-refusal-field-type-msg-v2"
# Which role may widen the provenance ledger and what the other may open from it.
# The digest has to carry this: it authorises what a role may reach on the measured
# path, so a change to it changes the experiment, and until it was named here the
# bundle token could stay still while that authorisation moved.
PROVENANCE_RULE_ID = "researcher-widens-ledger-analyst-reads-within-v1"
# What counts as having looked at the corpus before a unit may conclude that it
# holds no basis.
SEARCH_CONCLUSION_GUARD_RULE_ID = "executed-query-before-empty-commit-v1"


SEARCH_CORPUS = ToolSpec(
    "search_corpus",
    "Przeszukuje zamrożony korpus aktów prawnych i zwraca kandydatów: "
    "lokalizator, identyfikator aktu i początek treści przepisu.",
    SearchCorpusArgs,
)
READ_PROVISION = ToolSpec(
    "read_provision",
    "Zwraca pełną treść przepisu z zamrożonego korpusu.",
    ReadProvisionArgs,
)
POST_CANDIDATES = ToolSpec(
    "post_candidates",
    "Atomowo zapisuje listę kandydatów i kończy wyszukiwanie. Pusta "
    "lista jest stwierdzeniem, że korpus nie zawiera podstawy dla tej jednostki, "
    "i wolno ją zapisać dopiero po wyszukiwaniu. Identyfikator zakresu zadania "
    "nie jest argumentem wywołania.",
    PostCandidatesArgs,
    commit=True,
)
POST_CHARACTER = ToolSpec(
    "post_character",
    "Zapisuje charakter przepisu wraz z dowodami i kończy zadanie. "
    "Identyfikator zakresu zadania nie jest argumentem wywołania.",
    PostCharacterArgs,
    commit=True,
)
POST_RELEVANCE_VERDICT = ToolSpec(
    "post_verdict",
    "Zapisuje ocenę trafności kandydata i kończy zadanie. "
    "Identyfikator zakresu zadania nie jest argumentem wywołania.",
    RelevanceVerdictArgs,
    commit=True,
)
POST_RELATION_VERDICT = ToolSpec(
    "post_verdict",
    "Zapisuje ocenę relacji między jednostką umowy a podstawą prawną. "
    "Identyfikator zakresu zadania nie jest argumentem wywołania.",
    RelationVerdictArgs,
    commit=True,
)
LIST_FINDINGS = ToolSpec(
    "list_findings",
    "Zwraca istniejące ustalenia tej analizy: odnośnik, kod wyniku oraz "
    "lokalizatory prawne. Treści dokumentu ani treści przepisów nie ma na tej "
    "liście i nie wolno o nią pytać.",
    ListFindingsArgs,
)
GROUP_FINDINGS = ToolSpec(
    "group_findings",
    "Atomowo zapisuje grupowanie istniejących ustaleń i kończy zadanie. "
    "W finding_ids wolno wymieniać wyłącznie odnośniki z listy; odpowiedź "
    "nazywająca inne ustalenie jest odrzucana w całości. Identyfikator zakresu "
    "zadania nie jest argumentem wywołania.",
    GroupFindingsArgs,
    commit=True,
)

TASK_TOOLS: Mapping[Task, tuple[ToolSpec, ...]] = {
    "researcher.search": (SEARCH_CORPUS, READ_PROVISION, POST_CANDIDATES),
    "analyst.characterise": (READ_PROVISION, POST_CHARACTER),
    "verifier.relevance": (READ_PROVISION, POST_RELEVANCE_VERDICT),
    "verifier.relation": (READ_PROVISION, POST_RELATION_VERDICT),
    "synthesizer.group": (LIST_FINDINGS, GROUP_FINDINGS),
}


def tools_for(task: Task) -> tuple[ToolSpec, ...]:
    return TASK_TOOLS[task]


def definitions_for(specs: tuple[ToolSpec, ...]) -> tuple[ToolDefinition, ...]:
    return tuple(spec.definition() for spec in specs)


def commit_tool_for(task: Task) -> str | None:
    """The commit a silent model is nudged toward: the task's terminal one."""
    commits = [spec.name for spec in TASK_TOOLS[task] if spec.commit]
    return commits[-1] if commits else None


def commit_required(task: Task) -> bool:
    # The search concludes with a commit or with silence; the synthesis ends
    # with a grouping or without one, and a missing grouping is not a failure.
    return task not in ("researcher.search", "synthesizer.group")


def parse_tool(
    specs: tuple[ToolSpec, ...], name: str, arguments: str
) -> BaseModel | dict[str, object]:
    """Parse a named offered tool, or return the unknown-tool refusal."""
    spec = next((item for item in specs if item.name == name), None)
    return {"error": "unknown_tool"} if spec is None else spec.parse(arguments)


def _tool_bundle_version() -> str:
    from contract_analyzer.agents.interactive_tools import interactive_tool_policy

    policy = json.dumps(
        {
            "tasks": {
                task: [
                    {
                        "name": s.name,
                        "description": s.description,
                        "parameters": s.arguments.model_json_schema(),
                        "commit": s.commit,
                    }
                    for s in specs
                ]
                for task, specs in TASK_TOOLS.items()
            },
            "bounds": {
                "verifier_max_reads_per_candidate": VERIFIER_MAX_READS_PER_CANDIDATE,
                "task_max_turns": dict(TASK_MAX_TURNS),
                "finder_max_tool_turns": FINDER_MAX_TOOL_TURNS,
                "max_phrases_per_turn": FINDER_MAX_PHRASES_PER_TURN,
                "max_phrase_characters": FINDER_MAX_PHRASE_CHARS,
                "snippet_characters": FINDER_SNIPPET_CHARS,
                "results_per_search": RETRIEVAL_TOP_K,
                "max_steps_per_candidate": MAX_STEPS_PER_CANDIDATE,
            },
            "authorisation": {
                task: sorted(s.name for s in specs)
                for task, specs in TASK_TOOLS.items()
            },
            "verifier_projection_fields": VERIFIER_PROJECTION_FIELDS,
            "verifier_read_guard_rule": VERIFIER_READ_GUARD_RULE_ID,
            "commit_rule": COMMIT_RULE_ID,
            "argument_refusal_rule": ARGUMENT_REFUSAL_RULE_ID,
            "provenance_rule": PROVENANCE_RULE_ID,
            "search_conclusion_guard_rule": SEARCH_CONCLUSION_GUARD_RULE_ID,
            "scheduler_rules_version": SCHEDULER_RULES_VERSION,
            "interactive": interactive_tool_policy(),
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    digest = hashlib.sha256(policy.encode("utf-8")).hexdigest()[:12]
    return f"worksheet-roles-{digest}"


TOOL_BUNDLE_VERSION = _tool_bundle_version()
