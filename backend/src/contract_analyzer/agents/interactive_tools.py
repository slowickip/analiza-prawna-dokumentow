"""Tool contracts and bounds for post-run interaction conversations.

The explainer answers the reader; the researcher answers the explainer. Both run
outside the measured run, and neither may create or change a finding.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from contract_analyzer.agents.schema import FrozenModel, ToolSpec
from contract_analyzer.agents.tool_args import ReadProvisionArgs, SearchCorpusArgs

# Chosen by argument, not measurement. Six turns allow two grounded reads, one
# delegated corpus question, a correction, and the required final answer.
CHAT_MAX_TURNS = 6
# One delegated question prevents chat from becoming a second unmeasured analysis.
CHAT_MAX_QUESTIONS = 1
# Interactive work is user-facing and unmeasured; three minutes matches one model
# request ceiling without inheriting the measured run's fifteen-minute budget.
CHAT_WALL_BUDGET_SECONDS = 180.0
RESEARCHER_QUESTION_MAX_TURNS = 4
# Routing reads no corpus and commits on its first turn; the second turn exists
# only so one malformed call can be refused and retried before the router fails.
ROUTE_MESSAGE_MAX_TURNS = 2


class FindingIdArgs(FrozenModel):
    finding_id: str = Field(description="Identyfikator wybranego ustalenia.")


class UnitIdArgs(FrozenModel):
    unit_id: str = Field(description="Identyfikator jednostki dokumentu.")


class OpenQuestionArgs(FrozenModel):
    text: str = Field(
        min_length=1, description="Pytanie wymagające sprawdzenia korpusu."
    )


class PostAnswerArgs(FrozenModel):
    answer: str = Field(min_length=1, description="Odpowiedź systemu po polsku.")
    cited_finding_ids: tuple[str, ...] = Field(
        default=(), description="Identyfikatory cytowanych wybranych ustaleń."
    )


class RouteMessageArgs(FrozenModel):
    intent: Literal["ask", "contest", "analyse"] = Field(
        description="Czego oczekuje wiadomość czytelnika."
    )
    finding_ids: tuple[str, ...] = Field(
        default=(),
        description=(
            "Dla ask: ustalenia, których dotyczy pytanie; puste oznacza całą analizę."
        ),
    )
    finding_id: str | None = Field(
        default=None, description="Dla contest: podważane ustalenie."
    )
    unit_id: str | None = Field(
        default=None, description="Dla analyse: jednostka do ponownej analizy."
    )


class AnswerQuestionArgs(FrozenModel):
    answer: str = Field(min_length=1, description="Odpowiedź systemu po polsku.")
    locators: tuple[str, ...] = Field(
        default=(), description="Lokalizatory wspierające odpowiedź."
    )


READ_FINDING = ToolSpec("read_finding", "Odczytuje wybrane ustalenie.", FindingIdArgs)
READ_UNIT = ToolSpec("read_unit", "Odczytuje jednostkę dokumentu.", UnitIdArgs)
CHAT_READ_PROVISION = ToolSpec(
    "read_provision", "Odczytuje przepis z zamrożonego korpusu.", ReadProvisionArgs
)
OPEN_QUESTION = ToolSpec(
    "open_question", "Zleca wyszukującemu jedno pytanie do korpusu.", OpenQuestionArgs
)
POST_ANSWER = ToolSpec(
    "post_answer", "Zapisuje odpowiedź i kończy rozmowę.", PostAnswerArgs, commit=True
)
QUESTION_SEARCH = ToolSpec(
    "search_corpus",
    "Przeszukuje zamrożony korpus aktów prawnych.",
    SearchCorpusArgs,
)
QUESTION_READ = ToolSpec(
    "read_provision", "Odczytuje przepis z zamrożonego korpusu.", ReadProvisionArgs
)
ANSWER_QUESTION = ToolSpec(
    "answer_question",
    "Zwraca odpowiedź do objaśniającego.",
    AnswerQuestionArgs,
    commit=True,
)

CHAT_TOOLS = (
    READ_FINDING,
    READ_UNIT,
    CHAT_READ_PROVISION,
    OPEN_QUESTION,
    POST_ANSWER,
)
ROUTE_MESSAGE = ToolSpec(
    "route_message",
    "Zwraca rozpoznany zamiar wiadomości czytelnika i jej cel.",
    RouteMessageArgs,
    commit=True,
)

QUESTION_TOOLS = (QUESTION_SEARCH, QUESTION_READ, ANSWER_QUESTION)
ROUTE_TOOLS = (ROUTE_MESSAGE,)


def interactive_tool_policy() -> dict[str, object]:
    """Stable material included in the shared tool-bundle version."""

    def encoded(spec: ToolSpec) -> dict[str, object]:
        return {
            "name": spec.name,
            "description": spec.description,
            "parameters": spec.arguments.model_json_schema(),
            "commit": spec.commit,
        }

    return {
        "chat_tools": [encoded(spec) for spec in CHAT_TOOLS],
        "question_tools": [encoded(spec) for spec in QUESTION_TOOLS],
        "route_tools": [encoded(spec) for spec in ROUTE_TOOLS],
        "chat_max_turns": CHAT_MAX_TURNS,
        "chat_max_questions": CHAT_MAX_QUESTIONS,
        "chat_wall_budget_seconds": CHAT_WALL_BUDGET_SECONDS,
        "researcher_question_max_turns": RESEARCHER_QUESTION_MAX_TURNS,
        "route_message_max_turns": ROUTE_MESSAGE_MAX_TURNS,
    }
