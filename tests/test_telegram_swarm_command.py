from dataclasses import replace
from pathlib import Path

import pytest

from takopi.config import ProjectConfig, ProjectsConfig
from takopi.context import RunContext
from takopi.router import AutoRouter, RunnerEntry
from takopi.runners.mock import Return, ScriptRunner
from takopi.settings import TelegramTopicsSettings
from takopi.telegram.topic_state import TopicStateStore
from takopi.telegram.types import TelegramIncomingMessage
from takopi.transport_runtime import TransportRuntime
from tests.telegram_fakes import DEFAULT_ENGINE_ID, FakeTransport, make_cfg

from takopi.telegram.commands import swarm as swarm_commands


def _runtime(tmp_path: Path) -> TransportRuntime:
    runner = ScriptRunner([Return(answer="ok")], engine=DEFAULT_ENGINE_ID)
    router = AutoRouter(
        entries=[RunnerEntry(engine=runner.engine, runner=runner)],
        default_engine=runner.engine,
    )
    projects = ProjectsConfig(
        projects={
            "alpha": ProjectConfig(
                alias="Alpha",
                path=tmp_path,
                worktrees_dir=Path(".worktrees"),
            )
        },
        default_project="alpha",
        chat_map={123: "alpha"},
    )
    return TransportRuntime(router=router, projects=projects)


def _cfg(tmp_path: Path, transport: FakeTransport):
    return replace(
        make_cfg(transport),
        runtime=_runtime(tmp_path),
        topics=TelegramTopicsSettings(enabled=True, scope="all"),
    )


def _msg(
    text: str,
    *,
    chat_id: int = 123,
    message_id: int = 1,
    chat_type: str | None = "private",
    thread_id: int | None = None,
) -> TelegramIncomingMessage:
    return TelegramIncomingMessage(
        transport="telegram",
        chat_id=chat_id,
        message_id=message_id,
        text=text,
        reply_to_message_id=None,
        reply_to_text=None,
        sender_id=1,
        chat_type=chat_type,
        thread_id=thread_id,
    )


def test_plan_swarm_branches_starts_at_two() -> None:
    planned = swarm_commands._plan_swarm_branches(
        "feature",
        count=1,
        exists=lambda _name: False,
    )
    assert planned == ["feature-2"]


def test_plan_swarm_branches_fills_gaps() -> None:
    existing = {"feature-2", "feature-4"}
    planned = swarm_commands._plan_swarm_branches(
        "feature",
        count=2,
        exists=existing.__contains__,
    )
    assert planned == ["feature-3", "feature-5"]


@pytest.mark.anyio
async def test_swarm_command_requires_count(tmp_path: Path) -> None:
    transport = FakeTransport()
    cfg = _cfg(tmp_path, transport)
    store = TopicStateStore(tmp_path / "topics.json")
    msg = _msg("/swarm")

    await swarm_commands._handle_swarm_command(
        cfg,
        msg,
        args_text="",
        store=store,
        ambient_context=RunContext(project="alpha", branch="feature"),
        resolved_scope="all",
        scope_chat_ids=frozenset({msg.chat_id}),
    )

    text = transport.send_calls[-1]["message"].text
    assert "usage: /swarm" in text


@pytest.mark.anyio
@pytest.mark.parametrize("args_text", ["0", "-1", "nope", "1 2"])
async def test_swarm_command_rejects_invalid_count(
    tmp_path: Path, args_text: str
) -> None:
    transport = FakeTransport()
    cfg = _cfg(tmp_path, transport)
    store = TopicStateStore(tmp_path / "topics.json")
    msg = _msg(f"/swarm {args_text}")

    await swarm_commands._handle_swarm_command(
        cfg,
        msg,
        args_text=args_text,
        store=store,
        ambient_context=RunContext(project="alpha", branch="feature"),
        resolved_scope="all",
        scope_chat_ids=frozenset({msg.chat_id}),
    )

    text = transport.send_calls[-1]["message"].text
    assert "usage: /swarm" in text


@pytest.mark.anyio
async def test_swarm_command_requires_project(tmp_path: Path) -> None:
    transport = FakeTransport()
    cfg = replace(
        make_cfg(transport),
        topics=TelegramTopicsSettings(enabled=True, scope="all"),
    )
    store = TopicStateStore(tmp_path / "topics.json")
    msg = _msg("/swarm 1")

    await swarm_commands._handle_swarm_command(
        cfg,
        msg,
        args_text="1",
        store=store,
        ambient_context=None,
        resolved_scope="all",
        scope_chat_ids=frozenset({msg.chat_id}),
    )

    text = transport.send_calls[-1]["message"].text
    assert "project" in text


@pytest.mark.anyio
async def test_swarm_command_requires_branch(tmp_path: Path) -> None:
    transport = FakeTransport()
    cfg = _cfg(tmp_path, transport)
    store = TopicStateStore(tmp_path / "topics.json")
    msg = _msg("/swarm 1")

    await swarm_commands._handle_swarm_command(
        cfg,
        msg,
        args_text="1",
        store=store,
        ambient_context=RunContext(project="alpha", branch=None),
        resolved_scope="all",
        scope_chat_ids=frozenset({msg.chat_id}),
    )

    text = transport.send_calls[-1]["message"].text
    assert "branch" in text


@pytest.mark.anyio
async def test_swarm_command_creates_topics(tmp_path: Path, monkeypatch) -> None:
    transport = FakeTransport()
    cfg = _cfg(tmp_path, transport)
    store = TopicStateStore(tmp_path / "topics.json")
    msg = _msg("/swarm 2")
    planned_calls: list[tuple[str, int]] = []
    created_branches: list[tuple[str, str]] = []
    topic_calls: list[str] = []

    def _fake_plan(base_branch: str, *, count: int, exists):
        _ = exists
        planned_calls.append((base_branch, count))
        return ["feature-2", "feature-3"]

    def _fake_create(_root, *, new_branch: str, base_branch: str) -> None:
        created_branches.append((new_branch, base_branch))

    async def _fake_topic_command(_cfg, _msg, args_text: str, **_kwargs) -> None:
        topic_calls.append(args_text)

    monkeypatch.setattr(swarm_commands, "_plan_swarm_branches", _fake_plan)
    monkeypatch.setattr(swarm_commands, "_create_branch_from_base", _fake_create)
    monkeypatch.setattr(swarm_commands, "_handle_topic_command", _fake_topic_command)

    await swarm_commands._handle_swarm_command(
        cfg,
        msg,
        args_text="2",
        store=store,
        ambient_context=RunContext(project="alpha", branch="feature"),
        resolved_scope="all",
        scope_chat_ids=frozenset({msg.chat_id}),
    )

    assert planned_calls == [("feature", 2)]
    assert created_branches == [("feature-2", "feature"), ("feature-3", "feature")]
    assert topic_calls == ["Alpha @feature-2", "Alpha @feature-3"]


def test_create_branch_from_base_requires_base(tmp_path: Path, monkeypatch) -> None:
    def _fake_ok(*_args, **_kwargs) -> bool:
        return False

    monkeypatch.setattr(swarm_commands, "git_ok", _fake_ok)

    with pytest.raises(swarm_commands.SwarmError, match="base branch not found"):
        swarm_commands._create_branch_from_base(
            tmp_path,
            new_branch="feature-2",
            base_branch="feature",
        )
