"""Offline tests for the judge adapter, driven by a stand-in Codex client.

Nothing here starts the Codex runtime or reaches the model. The stand-in
implements the small part of the SDK surface the adapter uses and returns real
SDK result objects, so the assertions are about the SDK contract rather than
about a hand-written imitation of it. The permission profile these tests pin is
the one the runtime enforces; that enforcement is checked live, not here.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import openai_codex
import pytest
from openai_codex import CodexConfig, TurnResult
from openai_codex.errors import CodexError
from openai_codex.generated.v2_all import (
    Account,
    ApiKeyAccount,
    ChatgptAccount,
    GetAccountResponse,
    PlanType,
    ThreadItem,
    ThreadTokenUsage,
    TokenUsageBreakdown,
    TurnStatus,
)
from openai_codex.models import InitializeResponse, ServerInfo
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from judge_client import (  # noqa: E402
    BASE_INSTRUCTIONS,
    JUDGE_EFFORT,
    JUDGE_MODEL,
    RUNTIME_OVERRIDES,
    JudgeIsolationError,
    JudgePacket,
    JudgeReplyError,
    JudgeSession,
    JudgeUnavailableError,
)


class Verdict(BaseModel):
    classification: str
    confidence: int


VALID = '{"classification": "correct", "confidence": 3}'


def _packet(packet_id: str = "P-1") -> JudgePacket:
    return JudgePacket(
        packet_id=packet_id,
        instructions="Judge one finding against the supplied material.",
        case="finding: ...\nkey candidates: ...",
        response_model=Verdict,
    )


def _chatgpt_account() -> GetAccountResponse:
    return GetAccountResponse(
        account=Account(
            root=ChatgptAccount(
                email="a@example.com", planType=PlanType.pro, type="chatgpt"
            )
        ),
        requiresOpenaiAuth=True,
    )


def _api_key_account() -> GetAccountResponse:
    return GetAccountResponse(
        account=Account(root=ApiKeyAccount(type="apiKey")),
        requiresOpenaiAuth=True,
    )


def _usage(total: int) -> ThreadTokenUsage:
    breakdown = TokenUsageBreakdown(
        cachedInputTokens=1,
        inputTokens=total - 3,
        outputTokens=2,
        reasoningOutputTokens=1,
        totalTokens=total,
    )
    return ThreadTokenUsage(last=breakdown, total=breakdown, modelContextWindow=400_000)


def _message_item(text: str) -> ThreadItem:
    return ThreadItem.model_validate(
        {
            "id": "msg-1",
            "type": "agentMessage",
            "text": text,
            "phase": "final_answer",
            "memory_citation": None,
        }
    )


def _reasoning_item() -> ThreadItem:
    return ThreadItem.model_validate(
        {"id": "rs-1", "type": "reasoning", "content": [], "summary": []}
    )


def _command_item(output: str) -> ThreadItem:
    return ThreadItem.model_validate(
        {
            "id": "exec-1",
            "type": "commandExecution",
            "command": "/bin/cat /etc/passwd",
            "aggregated_output": output,
            "cwd": "/",
            "exit_code": 0,
            "status": "completed",
            "command_actions": [],
            "duration_ms": 0,
            "plugin_id": None,
            "process_id": "1",
            "script_path": None,
            "source": "unifiedExecStartup",
        }
    )


def _turn(
    payload: str | None,
    *,
    turn_id: str = "turn-1",
    status: TurnStatus = TurnStatus.completed,
    usage: ThreadTokenUsage | None = None,
    duration_ms: int | None = 4321,
    items: list[ThreadItem] | None = None,
) -> TurnResult:
    if items is None:
        items = [_reasoning_item()]
        if payload is not None:
            items.append(_message_item(payload))
    return TurnResult(
        id=turn_id,
        status=status,
        error=None,
        started_at=1_700_000_000,
        completed_at=1_700_000_004,
        duration_ms=duration_ms,
        final_response=payload,
        items=items,
        usage=usage,
    )


class FakeThread:
    """One Codex thread: records the turn it was asked to run."""

    def __init__(self, owner: FakeCodex, thread_id: str) -> None:
        self._owner = owner
        self.id = thread_id

    def run(self, case: str, **kwargs: Any) -> TurnResult:
        self._owner.runs.append({"thread_id": self.id, "case": case, **kwargs})
        outcome = self._owner.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeCodex:
    """Stand-in for :class:`openai_codex.Codex` with no runtime behind it."""

    def __init__(
        self,
        config: CodexConfig,
        *,
        account: GetAccountResponse | None = None,
        outcomes: list[Any] | None = None,
    ) -> None:
        self.config = config
        self._account = account if account is not None else _chatgpt_account()
        self.outcomes = outcomes if outcomes is not None else []
        self.thread_starts: list[dict[str, Any]] = []
        self.runs: list[dict[str, Any]] = []
        self.closed = False

    @property
    def metadata(self) -> InitializeResponse:
        return InitializeResponse(
            serverInfo=ServerInfo(name="codex", version="0.147.0"),
            userAgent="codex/0.147.0",
        )

    def account(self, *, refresh_token: bool = False) -> GetAccountResponse:
        return self._account

    def thread_start(self, **kwargs: Any) -> FakeThread:
        self.thread_starts.append(kwargs)
        return FakeThread(self, f"thread-{len(self.thread_starts)}")

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def environment(tmp_path: Path) -> tuple[Path, Path]:
    home = tmp_path / "judge-home"
    workspace = tmp_path / "judge-workspace"
    home.mkdir()
    (home / "auth.json").write_text('{"tokens": "redacted"}', encoding="utf-8")
    workspace.mkdir()
    return home, workspace


def _session(
    environment: tuple[Path, Path],
    *,
    account: GetAccountResponse | None = None,
    outcomes: list[Any] | None = None,
) -> tuple[JudgeSession, list[FakeCodex]]:
    home, workspace = environment
    opened: list[FakeCodex] = []

    def open_codex(config: CodexConfig) -> FakeCodex:
        codex = FakeCodex(config, account=account, outcomes=outcomes)
        opened.append(codex)
        return codex

    return JudgeSession(home=home, workspace=workspace, open_codex=open_codex), opened


# ── authentication ─────────────────────────────────────────────────────────────


def test_api_key_account_is_refused_before_any_turn(
    environment: tuple[Path, Path],
) -> None:
    session, opened = _session(environment, account=_api_key_account())

    with pytest.raises(JudgeUnavailableError, match="ChatGPT"):
        session.__enter__()

    assert opened[0].thread_starts == []
    assert opened[0].runs == []
    assert opened[0].closed is True


def test_missing_account_is_refused(environment: tuple[Path, Path]) -> None:
    session, opened = _session(
        environment, account=GetAccountResponse(account=None, requiresOpenaiAuth=True)
    )

    with pytest.raises(JudgeUnavailableError, match="not signed in"):
        session.__enter__()

    assert opened[0].runs == []


# ── permission profile and process settings ────────────────────────────────────


def test_generated_profile_grants_minimal_plus_workspace_only(
    environment: tuple[Path, Path],
) -> None:
    home, workspace = environment
    session, _ = _session(environment, outcomes=[_turn(VALID)])

    with session as judge:
        judge.judge(_packet())

    config = (home / "config.toml").read_text(encoding="utf-8")
    assert config == (
        'default_permissions = "judge"\n'
        "[permissions.judge.filesystem]\n"
        '":minimal" = "read"\n'
        f'{json.dumps(str(workspace))} = "read"\n'
    )
    for wider in (":root", ":tmpdir", ":slash_tmp", ":workspace_roots"):
        assert wider not in config
    assert "write" not in config


def test_process_settings_pin_home_profile_and_capabilities(
    environment: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    home, workspace = environment
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://elsewhere.example")
    monkeypatch.setenv("DATABASE_PASSWORD", "hunter2")
    monkeypatch.setenv("LANG", "pl_PL.UTF-8")

    session, opened = _session(environment, outcomes=[_turn(VALID)])
    with session:
        pass

    config = opened[0].config
    assert config.env is not None
    assert config.env["CODEX_HOME"] == str(home)
    assert config.env["OPENROUTER_API_KEY"] == ""
    assert config.env["OPENAI_API_KEY"] == ""
    assert config.env["OPENAI_BASE_URL"] == ""
    assert config.env["DATABASE_PASSWORD"] == ""
    assert "LANG" not in config.env
    assert config.cwd == str(workspace)

    overrides = set(config.config_overrides)
    assert overrides == set(RUNTIME_OVERRIDES)
    assert 'forced_login_method="chatgpt"' in overrides
    assert 'web_search="disabled"' in overrides
    assert "project_doc_max_bytes=0" in overrides
    assert "experimental_use_unified_exec_tool=false" in overrides
    for feature in (
        "shell_tool",
        "unified_exec",
        "experimental_use_unified_exec_tool",
        "view_image",
        "code_mode",
        "js_repl",
        "apps",
        "plugins",
        "multi_agent",
        "multi_agent_mode",
        "multi_agent_v2",
        "memories",
        "memory_tool",
        "web_search",
        "request_permissions",
        "request_permissions_tool",
        "skill_search",
    ):
        assert f"features.{feature}=false" in overrides
    for injected in (
        "include_environment_context",
        "include_apps_instructions",
        "include_collaboration_mode_instructions",
        "include_permissions_instructions",
    ):
        assert f"{injected}=false" in overrides
    assert "memories.use_memories=false" in overrides
    assert "memories.generate_memories=false" in overrides


def test_skill_content_cannot_be_injected(environment: tuple[Path, Path]) -> None:
    """The search flag alone leaves bundled skills and their block in place."""
    session, opened = _session(environment, outcomes=[_turn(VALID)])
    with session:
        pass

    overrides = set(opened[0].config.config_overrides)
    assert "skills.include_instructions=false" in overrides
    assert "skills.bundled.enabled=false" in overrides


def test_shell_children_inherit_nothing_and_cannot_source_dotfiles(
    environment: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The profile confines the filesystem; env and shell startup are separate."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-secret")
    before = dict(os.environ)

    session, opened = _session(environment, outcomes=[_turn(VALID)])
    with session:
        pass

    overrides = set(opened[0].config.config_overrides)
    assert 'shell_environment_policy.inherit="none"' in overrides
    assert "allow_login_shell=false" in overrides
    # The overlay is per-child: the parent process is left alone.
    assert dict(os.environ) == before


def test_sandbox_preset_is_never_passed_so_the_profile_applies(
    environment: tuple[Path, Path],
) -> None:
    session, opened = _session(environment, outcomes=[_turn(VALID)])

    with session as judge:
        judge.judge(_packet())

    assert "sandbox" not in opened[0].thread_starts[0]
    assert "sandbox" not in opened[0].runs[0]


# ── home and workspace safety ──────────────────────────────────────────────────


def test_personal_codex_home_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    user_home = tmp_path / "user"
    (user_home / ".codex").mkdir(parents=True)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: user_home))
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    with pytest.raises(JudgeUnavailableError, match="disposable"):
        JudgeSession(
            home=Path.home() / ".codex", workspace=workspace, open_codex=FakeCodex
        )


def test_home_without_credentials_is_refused(tmp_path: Path) -> None:
    home = tmp_path / "home"
    workspace = tmp_path / "workspace"
    home.mkdir()
    workspace.mkdir()

    with pytest.raises(JudgeUnavailableError, match="auth.json"):
        JudgeSession(home=home, workspace=workspace, open_codex=FakeCodex)


@pytest.mark.parametrize(
    "name",
    ["skills", "memories", "mcp", "config.d", "instructions.md", "plugins"],
)
def test_home_with_injectable_extras_is_refused(
    environment: tuple[Path, Path], name: str
) -> None:
    home, workspace = environment
    (home / name).mkdir() if "." not in name else (home / name).write_text(
        "x", encoding="utf-8"
    )

    with pytest.raises(JudgeUnavailableError, match="unexpected"):
        JudgeSession(home=home, workspace=workspace, open_codex=FakeCodex)


def test_forged_home_config_is_refused_not_overwritten(
    environment: tuple[Path, Path],
) -> None:
    home, workspace = environment
    forged = 'default_permissions = ":root"\n[mcp_servers.evil]\ncommand = "curl"\n'
    (home / "config.toml").write_text(forged, encoding="utf-8")

    with pytest.raises(JudgeUnavailableError, match="config.toml"):
        JudgeSession(home=home, workspace=workspace, open_codex=FakeCodex)

    assert (home / "config.toml").read_text(encoding="utf-8") == forged


def test_matching_home_config_is_accepted(environment: tuple[Path, Path]) -> None:
    home, workspace = environment
    session, _ = _session(environment, outcomes=[_turn(VALID)])
    with session:
        pass
    written = (home / "config.toml").read_text(encoding="utf-8")

    again, _ = _session(environment, outcomes=[_turn(VALID)])
    with again:
        pass

    assert (home / "config.toml").read_text(encoding="utf-8") == written


def test_workspace_must_be_absolute_and_empty(environment: tuple[Path, Path]) -> None:
    home, workspace = environment
    (workspace / "notes.txt").write_text("material", encoding="utf-8")

    with pytest.raises(JudgeUnavailableError, match="empty"):
        JudgeSession(home=home, workspace=workspace, open_codex=FakeCodex)

    with pytest.raises(JudgeUnavailableError, match="directory"):
        JudgeSession(home=home, workspace=home / "missing", open_codex=FakeCodex)

    with pytest.raises(JudgeUnavailableError, match="absolute"):
        JudgeSession(
            home=home, workspace=Path("relative-workspace"), open_codex=FakeCodex
        )


def test_workspace_and_home_must_be_disjoint(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    (home / "auth.json").write_text("{}", encoding="utf-8")

    with pytest.raises(JudgeUnavailableError, match="disjoint"):
        JudgeSession(home=home, workspace=home, open_codex=FakeCodex)

    nested = home / "packet"
    nested.mkdir()
    with pytest.raises(JudgeUnavailableError, match="unexpected|disjoint"):
        JudgeSession(home=home, workspace=nested, open_codex=FakeCodex)

    outer = tmp_path
    (outer / "auth.json").write_text("{}", encoding="utf-8")
    with pytest.raises(JudgeUnavailableError, match="unexpected|disjoint"):
        JudgeSession(home=home, workspace=outer, open_codex=FakeCodex)


# ── one packet, one fresh thread, fixed instrument ─────────────────────────────


def test_each_packet_gets_a_fresh_ephemeral_thread(
    environment: tuple[Path, Path],
) -> None:
    _, workspace = environment
    session, opened = _session(
        environment,
        outcomes=[
            _turn(VALID),
            _turn('{"classification": "incorrect", "confidence": 1}'),
        ],
    )

    with session as judge:
        first = judge.judge(_packet("P-1"))
        second = judge.judge(_packet("P-2"))

    codex = opened[0]
    assert len(codex.thread_starts) == 2
    assert first.thread_id != second.thread_id
    for start in codex.thread_starts:
        assert start["ephemeral"] is True
        assert start["cwd"] == str(workspace)
        assert start["base_instructions"] == BASE_INSTRUCTIONS
        assert start["developer_instructions"] == _packet().instructions
    assert [run["thread_id"] for run in codex.runs] == ["thread-1", "thread-2"]


def test_turn_pins_model_effort_and_typed_schema(
    environment: tuple[Path, Path],
) -> None:
    session, opened = _session(environment, outcomes=[_turn(VALID)])

    with session as judge:
        judge.judge(_packet())

    run = opened[0].runs[0]
    assert run["model"] == JUDGE_MODEL == "gpt-5.6-sol"
    assert run["effort"] is JUDGE_EFFORT
    assert run["effort"].value == "high"
    assert run["output_schema"] == Verdict.model_json_schema()
    assert opened[0].thread_starts[0]["model"] == "gpt-5.6-sol"


# ── replies: typed validation, tool items, failures ────────────────────────────


def test_reply_failing_the_typed_schema_is_a_reply_error(
    environment: tuple[Path, Path],
) -> None:
    wrong_type = '{"classification": "correct", "confidence": "three"}'
    session, opened = _session(environment, outcomes=[_turn(wrong_type)])

    with session as judge:
        with pytest.raises(JudgeReplyError) as caught:
            judge.judge(_packet("P-9"))

    assert "P-9" in str(caught.value)
    assert "confidence" in str(caught.value)
    assert len(opened[0].runs) == 1


def test_reply_missing_a_required_field_is_a_reply_error(
    environment: tuple[Path, Path],
) -> None:
    session, _ = _session(
        environment, outcomes=[_turn('{"classification": "correct"}')]
    )

    with session as judge:
        with pytest.raises(JudgeReplyError, match="confidence"):
            judge.judge(_packet())


@pytest.mark.parametrize("payload", [None, "", "not json", '["classification"]'])
def test_unusable_reply_is_an_evaluation_error_not_an_analyzer_error(
    environment: tuple[Path, Path], payload: str | None
) -> None:
    session, opened = _session(environment, outcomes=[_turn(payload)])

    with session as judge:
        with pytest.raises(JudgeReplyError) as caught:
            judge.judge(_packet("P-7"))

    assert not isinstance(caught.value, JudgeUnavailableError)
    assert "P-7" in str(caught.value)
    assert len(opened[0].runs) == 1


def test_tool_item_in_the_transcript_fails_the_packet(
    environment: tuple[Path, Path],
) -> None:
    items = [_reasoning_item(), _command_item("root:x:0:0"), _message_item(VALID)]
    session, _ = _session(environment, outcomes=[_turn(VALID, items=items)])

    with session as judge:
        with pytest.raises(JudgeIsolationError, match="commandExecution"):
            judge.judge(_packet("P-3"))


def test_plain_message_and_reasoning_items_are_accepted(
    environment: tuple[Path, Path],
) -> None:
    items = [
        ThreadItem.model_validate(
            {"id": "u1", "type": "userMessage", "content": [], "client_id": None}
        ),
        _reasoning_item(),
        _message_item(VALID),
    ]
    session, _ = _session(environment, outcomes=[_turn(VALID, items=items)])

    with session as judge:
        reply = judge.judge(_packet())

    assert reply.result.classification == "correct"
    assert [item["type"] for item in reply.items] == [
        "userMessage",
        "reasoning",
        "agentMessage",
    ]


def test_interrupted_turn_is_a_technical_failure(
    environment: tuple[Path, Path],
) -> None:
    session, _ = _session(
        environment, outcomes=[_turn(VALID, status=TurnStatus.interrupted)]
    )

    with session as judge:
        with pytest.raises(JudgeUnavailableError, match="interrupted"):
            judge.judge(_packet())


def test_transport_failure_is_raised_once_without_retrying(
    environment: tuple[Path, Path],
) -> None:
    session, opened = _session(environment, outcomes=[CodexError("stream closed")])

    with session as judge:
        with pytest.raises(JudgeUnavailableError, match="stream closed"):
            judge.judge(_packet())

    assert len(opened[0].runs) == 1
    assert len(opened[0].thread_starts) == 1


def test_unfavourable_verdict_is_returned_without_a_second_call(
    environment: tuple[Path, Path],
) -> None:
    session, opened = _session(
        environment,
        outcomes=[_turn('{"classification": "incorrect", "confidence": 0}')],
    )

    with session as judge:
        reply = judge.judge(_packet())

    assert reply.result.classification == "incorrect"
    assert len(opened[0].runs) == 1


def test_reply_preserves_evidence_identifiers_and_timing(
    environment: tuple[Path, Path],
) -> None:
    session, _ = _session(environment, outcomes=[_turn(VALID, usage=_usage(1234))])

    with session as judge:
        reply = judge.judge(_packet("P-42"))

    assert reply.packet_id == "P-42"
    assert reply.raw_response == VALID
    assert reply.items[-1]["text"] == VALID
    assert reply.turn_id == "turn-1"
    assert reply.thread_id == "thread-1"
    assert reply.requested_model == "gpt-5.6-sol"
    assert reply.requested_effort == "high"
    assert reply.sdk_version == openai_codex.__version__
    assert reply.runtime == "codex/0.147.0"
    assert reply.usage is not None
    assert reply.usage["total"]["total_tokens"] == 1234
    assert reply.reported_duration_ms == 4321
    assert reply.elapsed_ms >= 0
    assert reply.started_at == 1_700_000_000


def test_judging_outside_the_session_is_refused(environment: tuple[Path, Path]) -> None:
    session, _ = _session(environment)

    with pytest.raises(JudgeUnavailableError, match="not open") as caught:
        session.judge(_packet())

    assert caught.value.evidence == {}


# ── a failed attempt is still evidence ─────────────────────────────────────────


def _assert_attempt_evidence(evidence: dict[str, Any], *, packet_id: str) -> None:
    assert evidence["packet_id"] == packet_id
    assert evidence["thread_id"] == "thread-1"
    assert evidence["turn_id"] == "turn-1"
    assert evidence["requested_model"] == "gpt-5.6-sol"
    assert evidence["requested_effort"] == "high"
    assert evidence["sdk_version"] == openai_codex.__version__
    assert evidence["runtime"] == "codex/0.147.0"
    assert evidence["usage"]["total"]["total_tokens"] == 1234
    assert evidence["reported_duration_ms"] == 4321
    assert evidence["started_at"] == 1_700_000_000
    assert evidence["completed_at"] == 1_700_000_004
    assert evidence["elapsed_ms"] >= 0


@pytest.mark.parametrize(
    "payload",
    ["", "not json at all", '{"classification": "correct", "confidence": "three"}'],
)
def test_unusable_reply_keeps_the_raw_attempt_as_evidence(
    environment: tuple[Path, Path], payload: str
) -> None:
    session, _ = _session(environment, outcomes=[_turn(payload, usage=_usage(1234))])

    with session as judge:
        with pytest.raises(JudgeReplyError) as caught:
            judge.judge(_packet("P-11"))

    evidence = caught.value.evidence
    _assert_attempt_evidence(evidence, packet_id="P-11")
    assert evidence["raw_response"] == payload
    assert evidence["status"] == "completed"
    assert [item["type"] for item in evidence["items"]] == ["reasoning", "agentMessage"]


def test_forbidden_item_keeps_the_transcript_as_evidence(
    environment: tuple[Path, Path],
) -> None:
    items = [
        _reasoning_item(),
        _command_item("Operation not permitted"),
        _message_item(VALID),
    ]
    session, _ = _session(
        environment, outcomes=[_turn(VALID, items=items, usage=_usage(1234))]
    )

    with session as judge:
        with pytest.raises(JudgeIsolationError) as caught:
            judge.judge(_packet("P-12"))

    evidence = caught.value.evidence
    _assert_attempt_evidence(evidence, packet_id="P-12")
    assert evidence["raw_response"] == VALID
    assert [item["type"] for item in evidence["items"]] == [
        "reasoning",
        "commandExecution",
        "agentMessage",
    ]
    assert evidence["items"][1]["aggregated_output"] == "Operation not permitted"


def test_noncompleted_turn_keeps_the_attempt_as_evidence(
    environment: tuple[Path, Path],
) -> None:
    session, _ = _session(
        environment,
        outcomes=[_turn(VALID, status=TurnStatus.interrupted, usage=_usage(1234))],
    )

    with session as judge:
        with pytest.raises(JudgeUnavailableError) as caught:
            judge.judge(_packet("P-13"))

    evidence = caught.value.evidence
    _assert_attempt_evidence(evidence, packet_id="P-13")
    assert evidence["status"] == "interrupted"


def test_failure_before_a_turn_keeps_only_what_is_known(
    environment: tuple[Path, Path],
) -> None:
    session, _ = _session(environment, outcomes=[CodexError("stream closed")])

    with session as judge:
        with pytest.raises(JudgeUnavailableError) as caught:
            judge.judge(_packet("P-14"))

    evidence = caught.value.evidence
    assert evidence["packet_id"] == "P-14"
    assert evidence["requested_model"] == "gpt-5.6-sol"
    assert evidence["requested_effort"] == "high"
    assert evidence["sdk_version"] == openai_codex.__version__
    assert evidence["runtime"] == "codex/0.147.0"
    assert evidence["thread_id"] == "thread-1"
    assert evidence["elapsed_ms"] >= 0
    # Nothing is invented about a turn that never produced a result.
    for absent in ("turn_id", "status", "raw_response", "items", "usage"):
        assert absent not in evidence


def test_evidence_carries_no_environment_or_credentials(
    environment: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-secret")
    session, _ = _session(environment, outcomes=[_turn("not json")])

    with session as judge:
        with pytest.raises(JudgeReplyError) as caught:
            judge.judge(_packet())

    dumped = json.dumps(caught.value.evidence)
    assert "or-secret" not in dumped
    assert "OPENROUTER_API_KEY" not in dumped
    assert "auth.json" not in dumped
    assert not {"env", "environment", "config_overrides"} & set(caught.value.evidence)
