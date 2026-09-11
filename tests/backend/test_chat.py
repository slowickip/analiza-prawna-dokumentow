from __future__ import annotations

import asyncio
import hashlib
import json
from uuid import UUID, uuid4

import httpx
import pytest
from model_client import mock_model_client
from pydantic import ValidationError
from worksheet_transport import Turn, play, tool_call, tool_calls
from worksheet_transport import envelope as response_envelope

from contract_analyzer.agents.errors import InvalidChatCitation, RunPipelineError
from contract_analyzer.agents.prompts import load_prompt_bundle
from contract_analyzer.agents.runner import AnalysisRunner
from contract_analyzer.agents.services import AnalysisServices
from contract_analyzer.agents.session import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    DocumentSession,
    RunRequest,
)
from contract_analyzer.config import RunConfig, Settings
from contract_analyzer.domain import (
    ArmCode,
    DocumentPayload,
    FindingCode,
    ReadMode,
    SourceAnchor,
)
from contract_analyzer.model import InvalidModelResponse
from contract_analyzer.storage import FindingRecord, MetadataStore
from contract_analyzer.structure import parse_references, segment

DOCUMENT_TEXT = "§ 1. Najemca zobowiązuje się do płatności czynszu."
LOCATOR = "https://api.sejm.gov.pl/eli/acts/DU/2023/725/text.html/arti=11"


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _session() -> tuple[DocumentSession, UUID]:
    payload = DocumentPayload(
        document_id=uuid4(),
        text=DOCUMENT_TEXT,
        anchors=(SourceAnchor(start_offset=0, end_offset=len(DOCUMENT_TEXT)),),
        read_mode=ReadMode.NATIVE_PDF,
        content_hash=_content_hash(DOCUMENT_TEXT),
    )
    units = segment(payload, RunConfig())
    session = DocumentSession(
        document_id=payload.document_id,
        payload=payload,
        units=tuple(units),
        references=tuple(parse_references(units)),
    )
    return session, payload.document_id


def _tool_turn_response(
    request: httpx.Request,
    body: dict[str, object],
    *,
    model: str = "deepseek-v4-flash-20260828",
    input_tokens: int = 10,
    output_tokens: int = 2,
) -> httpx.Response:
    return httpx.Response(
        200,
        request=request,
        json=response_envelope(
            play(Turn(body)),
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        ),
    )


def _make_transport(
    *,
    chat_cited_ids: list[str] | None = None,
    captured_payloads: list[dict[str, object]] | None = None,
    chat_call_count: list[int] | None = None,
) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        if "tools" in payload:
            turn = Turn(payload)
            if turn.task == "explain":
                if chat_call_count is not None:
                    chat_call_count[0] += 1
                if captured_payloads is not None:
                    captured_payloads.append(turn.payload)
                cited = chat_cited_ids if chat_cited_ids is not None else []
                return httpx.Response(
                    200,
                    request=request,
                    json=response_envelope(
                        tool_call(
                            "post_answer",
                            {
                                "answer": "System wyjaśnia wynik.",
                                "cited_finding_ids": cited,
                            },
                        )
                    ),
                )
            return _tool_turn_response(request, payload)
        raise AssertionError("the pipeline sent a request carrying no tools")

    return httpx.MockTransport(handler)


async def _run_analysis(
    built_corpus: object,
    tmp_path: object,
    *,
    transport: httpx.MockTransport,
) -> tuple[AnalysisRunner, object]:
    session, document_id = _session()
    settings = Settings(
        model_api_key="test-secret-key",
        model_base_url="https://openrouter.ai/api/v1",
        model_name="deepseek-v4-flash",
    )
    client = mock_model_client(transport, settings=settings)
    services = AnalysisServices(
        settings=settings,
        corpus=built_corpus,  # type: ignore[arg-type]
        client=client,
        metadata=MetadataStore(tmp_path / "chat.sqlite3"),  # type: ignore[operator]
        prompt_bundle=load_prompt_bundle(),
    )
    services.sessions[document_id] = session  # type: ignore[arg-type]
    runner = AnalysisRunner(services)
    result = await runner.run(
        RunRequest(
            document_id=document_id,
            arm=ArmCode.MID,
            wall_budget_seconds=300.0,
            concurrency=1,
        )
    )
    return runner, result


def _adjudicated_finding(result: object) -> object:
    return next(
        finding
        for finding in result.findings  # type: ignore[attr-defined]
        if finding.code is not FindingCode.NOT_PROCESSED
    )


def test_chat_response_rejects_extra_mutation_fields() -> None:
    with pytest.raises(ValidationError):
        ChatResponse.model_validate(
            {
                "answer": "System wyjaśnia wynik.",
                "cited_finding_ids": [],
                "interaction_run_id": str(uuid4()),
                "corpus_consulted": False,
                "findings": [{"code": "contradictory"}],
            }
        )


def test_explain_rejects_model_response_with_extra_fields(
    built_corpus,
    tmp_path: object,
) -> None:
    session, document_id = _session()

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        if "tools" in payload:
            turn = Turn(payload)
            if turn.task == "explain":
                return httpx.Response(
                    200,
                    request=request,
                    json=response_envelope(
                        tool_call(
                            "post_answer",
                            {
                                "answer": "ok",
                                "cited_finding_ids": [],
                                "findings": [],
                            },
                        )
                    ),
                )
            return _tool_turn_response(request, payload)
        raise AssertionError("the pipeline sent a request carrying no tools")

    async def exercise() -> None:
        settings = Settings(
            model_api_key="test-secret-key",
            model_base_url="https://openrouter.ai/api/v1",
            model_name="deepseek-v4-flash",
        )
        client = mock_model_client(httpx.MockTransport(handler), settings=settings)
        services = AnalysisServices(
            settings=settings,
            corpus=built_corpus,
            client=client,
            metadata=MetadataStore(tmp_path / "chat-extra.sqlite3"),  # type: ignore[operator]
            prompt_bundle=load_prompt_bundle(),
        )
        services.sessions[document_id] = session  # type: ignore[arg-type]
        runner = AnalysisRunner(services)
        result = await runner.run(
            RunRequest(
                document_id=document_id,
                arm=ArmCode.MID,
                wall_budget_seconds=300.0,
                concurrency=1,
            )
        )
        finding_id = next(
            finding.id
            for finding in result.findings
            if finding.code is not FindingCode.NOT_PROCESSED
        )
        with pytest.raises(InvalidModelResponse) as error:
            await runner.explain(
                ChatRequest(
                    run_id=result.run_id,
                    question="Dlaczego?",
                    selected_finding_ids=(finding_id,),
                )
            )
        assert error.value.code == "model_response_schema_invalid"

    asyncio.run(exercise())


def test_explain_rejects_unknown_cited_finding_id(
    built_corpus,
    tmp_path: object,
) -> None:
    session, document_id = _session()
    unknown_id = uuid4()

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        if "tools" in payload:
            turn = Turn(payload)
            if turn.task == "explain":
                return httpx.Response(
                    200,
                    request=request,
                    json=response_envelope(
                        tool_call(
                            "post_answer",
                            {
                                "answer": "System odwołuje się do wyniku.",
                                "cited_finding_ids": [str(unknown_id)],
                            },
                        )
                    ),
                )
            return _tool_turn_response(request, payload)
        raise AssertionError("the pipeline sent a request carrying no tools")

    async def exercise() -> None:
        settings = Settings(
            model_api_key="test-secret-key",
            model_base_url="https://openrouter.ai/api/v1",
            model_name="deepseek-v4-flash",
        )
        client = mock_model_client(httpx.MockTransport(handler), settings=settings)
        services = AnalysisServices(
            settings=settings,
            corpus=built_corpus,
            client=client,
            metadata=MetadataStore(tmp_path / "chat-citation.sqlite3"),  # type: ignore[operator]
            prompt_bundle=load_prompt_bundle(),
        )
        services.sessions[document_id] = session  # type: ignore[arg-type]
        runner = AnalysisRunner(services)
        result = await runner.run(
            RunRequest(
                document_id=document_id,
                arm=ArmCode.MID,
                wall_budget_seconds=300.0,
                concurrency=1,
            )
        )
        finding_id = result.findings[0].id
        with pytest.raises(InvalidChatCitation) as error:
            await runner.explain(
                ChatRequest(
                    run_id=result.run_id,
                    question="Dlaczego?",
                    selected_finding_ids=(finding_id,),
                )
            )
        assert error.value.code == "chat_citation_unknown"

    asyncio.run(exercise())


def test_explain_payload_carries_finding_substance(
    built_corpus,
    tmp_path: object,
) -> None:
    captured: list[dict[str, object]] = []

    async def exercise() -> None:
        runner, result = await _run_analysis(
            built_corpus,
            tmp_path,
            transport=_make_transport(captured_payloads=captured),
        )
        finding = _adjudicated_finding(result)
        await runner.explain(
            ChatRequest(
                run_id=result.run_id,
                question="Dlaczego?",
                selected_finding_ids=(finding.id,),
            )
        )
        payload = captured[0]
        grounding = payload.get("grounding")
        assert grounding is not None
        findings = grounding["findings"]  # type: ignore[index]
        assert findings[0]["code"] == finding.code.value
        assert findings[0]["legal_locators"] == list(finding.legal_locators)

    asyncio.run(exercise())


def test_explain_payload_carries_cited_provision_text(
    built_corpus,
    tmp_path: object,
) -> None:
    captured: list[dict[str, object]] = []
    corpus = built_corpus

    async def exercise() -> None:
        runner, result = await _run_analysis(
            built_corpus,
            tmp_path,
            transport=_make_transport(captured_payloads=captured),
        )
        finding = _adjudicated_finding(result)
        assert finding.legal_locators
        provision_text = corpus.read(finding.legal_locators[0]).text
        await runner.explain(
            ChatRequest(
                run_id=result.run_id,
                question="Co mówi przepis?",
                selected_finding_ids=(finding.id,),
            )
        )
        grounding = captured[0]["grounding"]  # type: ignore[index]
        provisions = grounding["provisions"]  # type: ignore[index]
        assert any(item["text"] == provision_text for item in provisions)

    asyncio.run(exercise())


def test_explain_rejects_citation_outside_selected_set(
    built_corpus,
    tmp_path: object,
) -> None:
    async def exercise() -> None:
        cited_ids: list[str] = []
        runner, result = await _run_analysis(
            built_corpus,
            tmp_path,
            transport=_make_transport(chat_cited_ids=cited_ids),
        )
        selected = _adjudicated_finding(result)
        other = FindingRecord(
            id=uuid4(),
            run_id=result.run_id,
            unit_id=selected.unit_id,
            code=FindingCode.NOT_PROCESSED,
        )
        runner.services.metadata.store_finding(other)
        cited_ids.append(str(other.id))
        with pytest.raises(InvalidChatCitation) as error:
            await runner.explain(
                ChatRequest(
                    run_id=result.run_id,
                    question="Dlaczego?",
                    selected_finding_ids=(selected.id,),
                )
            )
        assert error.value.code == "chat_citation_unknown"

    asyncio.run(exercise())


def test_explain_accepts_citation_within_selected_set(
    built_corpus,
    tmp_path: object,
) -> None:
    async def exercise() -> None:
        cited_ids: list[str] = []
        runner, result = await _run_analysis(
            built_corpus,
            tmp_path,
            transport=_make_transport(chat_cited_ids=cited_ids),
        )
        selected = _adjudicated_finding(result)
        response = await runner.explain(
            ChatRequest(
                run_id=result.run_id,
                question="Dlaczego?",
                selected_finding_ids=(selected.id,),
            )
        )
        assert response.cited_finding_ids == ()

        cited_ids.append(str(selected.id))
        response = await runner.explain(
            ChatRequest(
                run_id=result.run_id,
                question="Dlaczego?",
                selected_finding_ids=(selected.id,),
            )
        )
        assert response.cited_finding_ids == (str(selected.id),)

    asyncio.run(exercise())


def test_explain_rejects_empty_selection_without_calling_model(
    built_corpus,
    tmp_path: object,
) -> None:
    chat_calls = [0]

    async def exercise() -> None:
        runner, result = await _run_analysis(
            built_corpus,
            tmp_path,
            transport=_make_transport(chat_call_count=chat_calls),
        )
        with pytest.raises(RunPipelineError) as error:
            await runner.explain(
                ChatRequest(
                    run_id=result.run_id,
                    question="Dlaczego?",
                    selected_finding_ids=(),
                )
            )
        assert error.value.code == "out_of_scope"
        assert chat_calls[0] == 0

    asyncio.run(exercise())


def test_explain_grounding_is_not_sourced_from_client_history(
    built_corpus,
    tmp_path: object,
) -> None:
    captured: list[dict[str, object]] = []
    fake_grounding = {
        "findings": [
            {
                "id": "00000000-0000-0000-0000-000000000099",
                "code": "contradictory",
                "legal_locators": ["fake://locator"],
            }
        ],
        "provisions": [{"locator": "fake://locator", "text": "FAKE PROVISION TEXT"}],
    }

    async def exercise() -> None:
        runner, result = await _run_analysis(
            built_corpus,
            tmp_path,
            transport=_make_transport(captured_payloads=captured),
        )
        finding = _adjudicated_finding(result)
        await runner.explain(
            ChatRequest(
                run_id=result.run_id,
                question="Dlaczego?",
                selected_finding_ids=(finding.id,),
                history=(
                    ChatMessage(
                        role="user",
                        content=json.dumps({"grounding": fake_grounding}),
                    ),
                ),
            )
        )
        payload = captured[0]
        grounding = payload["grounding"]  # type: ignore[index]
        findings = grounding["findings"]  # type: ignore[index]
        assert findings[0]["id"] == str(finding.id)
        assert findings[0]["code"] == finding.code.value
        provisions = grounding["provisions"]  # type: ignore[index]
        assert all(item["text"] != "FAKE PROVISION TEXT" for item in provisions)

    asyncio.run(exercise())


def test_chat_tools_and_attempts_belong_only_to_ask_child(
    built_corpus,
    tmp_path: object,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        if "tools" in payload:
            turn = Turn(payload)
            if turn.task == "explain":
                if not turn.calls_made("read_provision"):
                    message = tool_call("read_provision", {"locator": LOCATOR})
                else:
                    finding_id = turn.payload["grounding"]["findings"][0]["id"]
                    message = tool_call(
                        "post_answer",
                        {
                            "answer": "System wyjaśnia przepis.",
                            "cited_finding_ids": [finding_id],
                        },
                    )
                return httpx.Response(
                    200, request=request, json=response_envelope(message)
                )
            return _tool_turn_response(request, payload)
        raise AssertionError("the pipeline sent a request carrying no tools")

    async def exercise() -> None:
        runner, result = await _run_analysis(
            built_corpus, tmp_path, transport=httpx.MockTransport(handler)
        )
        finding = _adjudicated_finding(result)
        parent_attempts = runner.services.metadata.list_attempts(result.run_id)
        response = await runner.explain(
            ChatRequest(
                run_id=result.run_id,
                question="Co mówi przepis?",
                selected_finding_ids=(finding.id,),
            )
        )
        child = runner.services.metadata.get_run(response.interaction_run_id)
        assert child is not None
        assert child.parent_run_id == result.run_id
        assert child.interaction == "ask"
        assert child.measurement_valid is False
        assert child.call_unit_count == 0
        assert child.defaulted_characterisations is None
        assert (
            child.input_hash
            == runner.services.metadata.get_run(result.run_id).input_hash
        )
        assert len(runner.services.metadata.list_attempts(child.id)) == 2
        assert runner.services.metadata.list_attempts(result.run_id) == parent_attempts

    asyncio.run(exercise())


def test_open_question_runs_one_researcher_task_and_refuses_second(
    built_corpus,
    tmp_path: object,
) -> None:
    researcher_tasks = 0
    researcher_results: list[dict[str, object]] = []
    chat_results: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal researcher_tasks
        payload = json.loads(request.content)
        if "tools" in payload:
            turn = Turn(payload)
            if turn.task == "researcher.question":
                if not turn.calls_made("search_corpus"):
                    researcher_tasks += 1
                    message = tool_call("search_corpus", {"phrase": "najem czynsz"})
                else:
                    researcher_results.extend(turn.tool_results)
                    locator = turn.first_result_locator()
                    message = tool_call(
                        "answer_question",
                        {
                            "answer": "System znalazł przepis.",
                            "locators": [locator] if locator else [],
                        },
                    )
            elif turn.task == "explain" and not turn.calls_made("open_question"):
                message = tool_calls(
                    [
                        ("open_question", {"text": "Jaka jest podstawa?"}),
                        ("open_question", {"text": "Drugie pytanie"}),
                    ]
                )
            elif turn.task == "explain":
                chat_results.extend(turn.tool_results)
                message = tool_call(
                    "post_answer",
                    {"answer": "System informuje o wyniku.", "cited_finding_ids": []},
                )
            else:
                return _tool_turn_response(request, payload)
            return httpx.Response(200, request=request, json=response_envelope(message))
        raise AssertionError("the pipeline sent a request carrying no tools")

    async def exercise() -> None:
        runner, result = await _run_analysis(
            built_corpus, tmp_path, transport=httpx.MockTransport(handler)
        )
        finding = _adjudicated_finding(result)
        response = await runner.explain(
            ChatRequest(
                run_id=result.run_id,
                question="Jaka jest podstawa?",
                selected_finding_ids=(finding.id,),
            )
        )
        assert response.corpus_consulted is True
        assert researcher_tasks == 1
        assert any(item.get("results") for item in researcher_results)
        assert any(
            item.get("answer") == "System znalazł przepis." for item in chat_results
        )
        assert any(
            item.get("error") == "question_limit_reached" for item in chat_results
        )

    asyncio.run(exercise())


def test_a_delegated_question_that_never_reached_the_corpus_is_not_marked_as_such(
    built_corpus,
    tmp_path: object,
) -> None:
    """Delegating a question is not consulting the corpus.

    corpus_consulted was set when the explainer opened a question, before the
    researcher had done anything, and it is what the reader is shown to say the
    answer rests on the frozen corpus. A researcher whose every phrase is refused
    before the corpus is queried can still commit an answer, and that answer was
    labelled as corpus-backed.
    """
    refusals: list[dict[str, object]] = []

    def chat_turn(turn: Turn) -> dict[str, object]:
        if not turn.calls_made("open_question"):
            return tool_call("open_question", {"text": "Jaka jest podstawa?"})
        return tool_call(
            "post_answer",
            {"answer": "System odpowiada bez korpusu.", "cited_finding_ids": []},
        )

    def question_turn(turn: Turn) -> dict[str, object]:
        if not turn.calls_made("search_corpus"):
            return tool_call("search_corpus", {"phrase": "   "})
        refusals.extend(turn.tool_results)
        return tool_call(
            "answer_question", {"answer": "Nie mam podstawy.", "locators": []}
        )

    async def exercise() -> None:
        runner, result = await _run_analysis(
            built_corpus,
            tmp_path,
            transport=_interactive_transport(chat_turn, question_turn),
        )
        finding = _adjudicated_finding(result)
        response = await runner.explain(
            ChatRequest(
                run_id=result.run_id,
                question="Jaka jest podstawa?",
                selected_finding_ids=(finding.id,),
            )
        )
        # The question ran and was refused before any corpus call, so the label
        # is the only thing under test here.
        assert [item.get("error") for item in refusals] == ["phrase_empty"]
        assert response.answer == "System odpowiada bez korpusu."
        assert response.corpus_consulted is False

    asyncio.run(exercise())


def _interactive_transport(
    chat_turn: object, question_turn: object
) -> httpx.MockTransport:
    """Drive the two interactive tasks from one scripted handler."""

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        if "tools" not in payload:
            raise AssertionError("the pipeline sent a request carrying no tools")
        turn = Turn(payload)
        if turn.task == "explain":
            message = chat_turn(turn)  # type: ignore[operator]
        elif turn.task == "researcher.question":
            message = question_turn(turn)  # type: ignore[operator]
        else:
            return _tool_turn_response(request, payload)
        return httpx.Response(200, request=request, json=response_envelope(message))

    return httpx.MockTransport(handler)


def test_chat_cannot_post_an_answer_in_the_turn_that_asks_the_researcher(
    built_corpus,
    tmp_path: object,
) -> None:
    """The answer would have been written before the researcher answered."""
    questions_run = 0
    refusals: list[dict[str, object]] = []

    def chat_turn(turn: Turn) -> dict[str, object]:
        if not turn.calls_made("open_question"):
            return tool_calls(
                [
                    ("open_question", {"text": "Jaka jest podstawa?"}),
                    (
                        "post_answer",
                        {"answer": "System odpowiada.", "cited_finding_ids": []},
                    ),
                ]
            )
        refusals.extend(item for item in turn.tool_results if "error" in item)
        return tool_call(
            "post_answer",
            {"answer": "System odpowiada po odmowie.", "cited_finding_ids": []},
        )

    def question_turn(turn: Turn) -> dict[str, object]:
        nonlocal questions_run
        questions_run += 1
        return tool_call("answer_question", {"answer": "nieoczekiwane", "locators": []})

    async def exercise() -> None:
        runner, result = await _run_analysis(
            built_corpus,
            tmp_path,
            transport=_interactive_transport(chat_turn, question_turn),
        )
        finding = _adjudicated_finding(result)
        response = await runner.explain(
            ChatRequest(
                run_id=result.run_id,
                question="Jaka jest podstawa?",
                selected_finding_ids=(finding.id,),
            )
        )
        assert [item["error"] for item in refusals] == [
            "commit_must_be_alone",
            "commit_must_be_alone",
        ]
        assert questions_run == 0
        assert response.answer == "System odpowiada po odmowie."
        assert response.corpus_consulted is False

    asyncio.run(exercise())


def test_the_researcher_question_cannot_answer_in_the_turn_that_searches(
    built_corpus,
    tmp_path: object,
) -> None:
    """Same rule on the researcher side: the answer must follow its evidence."""
    refusals: list[dict[str, object]] = []

    def chat_turn(turn: Turn) -> dict[str, object]:
        if not turn.calls_made("open_question"):
            return tool_call("open_question", {"text": "Jaka jest podstawa?"})
        return tool_call(
            "post_answer",
            {"answer": "System odpowiada.", "cited_finding_ids": []},
        )

    def question_turn(turn: Turn) -> dict[str, object]:
        if not turn.calls_made("search_corpus"):
            return tool_calls(
                [
                    ("search_corpus", {"phrase": "najem czynsz"}),
                    (
                        "answer_question",
                        {"answer": "System znalazł przepis.", "locators": []},
                    ),
                ]
            )
        refusals.extend(item for item in turn.tool_results if "error" in item)
        return tool_call(
            "answer_question", {"answer": "System odpowiada.", "locators": []}
        )

    async def exercise() -> None:
        runner, result = await _run_analysis(
            built_corpus,
            tmp_path,
            transport=_interactive_transport(chat_turn, question_turn),
        )
        finding = _adjudicated_finding(result)
        await runner.explain(
            ChatRequest(
                run_id=result.run_id,
                question="Jaka jest podstawa?",
                selected_finding_ids=(finding.id,),
            )
        )
        assert [item["error"] for item in refusals] == [
            "commit_must_be_alone",
            "commit_must_be_alone",
        ]

    asyncio.run(exercise())


def test_a_refused_phrase_is_not_counted_as_a_search_on_the_interactive_path(
    built_corpus,
    tmp_path: object,
) -> None:
    """The measured path was fixed for this; the interactive one had the same bug.

    A phrase that sanitises to nothing is refused before the corpus is queried, but
    the refusal payload still carries an empty results list. Counting that as a
    search inflated finder_search_calls with calls nobody made.
    """

    def chat_turn(turn: Turn) -> dict[str, object]:
        if not turn.calls_made("open_question"):
            return tool_call("open_question", {"text": "Jaka jest podstawa?"})
        return tool_call(
            "post_answer",
            {"answer": "System odpowiada.", "cited_finding_ids": []},
        )

    def question_turn(turn: Turn) -> dict[str, object]:
        if not turn.calls_made("search_corpus"):
            return tool_call("search_corpus", {"phrase": "!!!"})
        return tool_call(
            "answer_question", {"answer": "System odpowiada.", "locators": []}
        )

    async def exercise() -> None:
        runner, result = await _run_analysis(
            built_corpus,
            tmp_path,
            transport=_interactive_transport(chat_turn, question_turn),
        )
        finding = _adjudicated_finding(result)
        response = await runner.explain(
            ChatRequest(
                run_id=result.run_id,
                question="Jaka jest podstawa?",
                selected_finding_ids=(finding.id,),
            )
        )
        child = runner.services.metadata.get_run(response.interaction_run_id)
        assert child is not None
        assert child.finder_search_calls == 0

    asyncio.run(exercise())


def test_an_answer_cites_only_what_this_question_was_given(
    built_corpus,
    tmp_path: object,
) -> None:
    """A locator that exists in the corpus is not evidence this question obtained."""
    refusals: list[dict[str, object]] = []

    def chat_turn(turn: Turn) -> dict[str, object]:
        if not turn.calls_made("open_question"):
            return tool_call("open_question", {"text": "Jaka jest podstawa?"})
        return tool_call(
            "post_answer",
            {"answer": "System odpowiada.", "cited_finding_ids": []},
        )

    def question_turn(turn: Turn) -> dict[str, object]:
        if not turn.calls_made("answer_question"):
            return tool_call(
                "answer_question",
                {"answer": "System pamięta przepis.", "locators": [LOCATOR]},
            )
        refusals.extend(item for item in turn.tool_results if "error" in item)
        return tool_call(
            "answer_question", {"answer": "System nie cytuje.", "locators": []}
        )

    async def exercise() -> None:
        runner, result = await _run_analysis(
            built_corpus,
            tmp_path,
            transport=_interactive_transport(chat_turn, question_turn),
        )
        finding = _adjudicated_finding(result)
        await runner.explain(
            ChatRequest(
                run_id=result.run_id,
                question="Jaka jest podstawa?",
                selected_finding_ids=(finding.id,),
            )
        )
        assert [item["error"] for item in refusals] == [
            "locator_not_in_provenance_ledger"
        ]

    asyncio.run(exercise())
