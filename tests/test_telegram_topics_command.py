from dataclasses import replace
from pathlib import Path

import anyio
import pytest

from takopi.context import RunContext
from takopi.config import ProjectConfig, ProjectsConfig
from takopi.runners.mock import Return, ScriptRunner
from takopi.settings import TelegramTopicsSettings
from takopi.telegram.chat_sessions import ChatSessionStore
from takopi.telegram.chat_prefs import ChatPrefsStore, resolve_prefs_path
from takopi.telegram.commands.topics import (
    _handle_allrepos_command,
    _handle_chat_ctx_command,
    _handle_chat_new_command,
    _handle_ctx_command,
    _handle_new_command,
    _handle_pause_command,
    _handle_resume_command,
    _handle_topic_command,
)
from takopi.telegram.topic_state import TopicStateStore
from takopi.telegram.types import TelegramIncomingMessage
from tests.telegram_fakes import (
    DEFAULT_ENGINE_ID,
    FakeTransport,
    _make_router,
    make_cfg,
)
from takopi.transport_runtime import TransportRuntime


def _msg(
    text: str,
    *,
    chat_id: int = 123,
    message_id: int = 1,
    thread_id: int | None = None,
    chat_type: str | None = "private",
) -> TelegramIncomingMessage:
    return TelegramIncomingMessage(
        transport="telegram",
        chat_id=chat_id,
        message_id=message_id,
        text=text,
        reply_to_message_id=None,
        reply_to_text=None,
        sender_id=1,
        thread_id=thread_id,
        chat_type=chat_type,
    )


def _runtime(
    tmp_path: Path, projects: dict[str, ProjectConfig] | None = None
) -> tuple[TransportRuntime, Path]:
    runner = ScriptRunner([Return(answer="ok")], engine=DEFAULT_ENGINE_ID)
    if projects is None:
        projects = {
            "alpha": ProjectConfig(
                alias="Alpha",
                path=tmp_path,
                worktrees_dir=Path(".worktrees"),
            )
        }
    default_project = next(iter(projects), None)
    projects_cfg = ProjectsConfig(
        projects=projects,
        default_project=default_project,
    )
    state_path = tmp_path / "takopi.toml"
    runtime = TransportRuntime(
        router=_make_router(runner),
        projects=projects_cfg,
        config_path=state_path,
    )
    return runtime, state_path


def _cfg_with_projects(
    transport: FakeTransport,
    tmp_path: Path,
    projects: dict[str, ProjectConfig],
    *,
    scope: str = "all",
):
    runtime, _ = _runtime(tmp_path, projects)
    return replace(
        make_cfg(transport),
        runtime=runtime,
        topics=TelegramTopicsSettings(enabled=True, scope=scope),
    )


@pytest.mark.anyio
async def test_ctx_command_requires_topic(tmp_path: Path) -> None:
    transport = FakeTransport()
    cfg = replace(
        make_cfg(transport),
        topics=TelegramTopicsSettings(enabled=True, scope="all"),
    )
    store = TopicStateStore(tmp_path / "topics.json")
    msg = _msg("/ctx")

    await _handle_ctx_command(
        cfg,
        msg,
        args_text="",
        store=store,
        resolved_scope="all",
        scope_chat_ids=frozenset({msg.chat_id}),
    )

    text = transport.send_calls[-1]["message"].text
    assert "only works inside a topic" in text


@pytest.mark.anyio
async def test_chat_ctx_command_sets_binding(tmp_path: Path) -> None:
    transport = FakeTransport()
    runtime, state_path = _runtime(tmp_path)
    cfg = replace(make_cfg(transport), runtime=runtime, session_mode="chat")
    store = ChatPrefsStore(resolve_prefs_path(state_path))

    msg = _msg("/ctx set alpha @dev", chat_type="private")
    await _handle_chat_ctx_command(
        cfg,
        msg,
        args_text="set alpha @dev",
        chat_prefs=store,
    )

    msg_show = _msg("/ctx", chat_type="private")
    await _handle_chat_ctx_command(
        cfg,
        msg_show,
        args_text="",
        chat_prefs=store,
    )

    text = transport.send_calls[-1]["message"].text
    assert "bound ctx: Alpha @dev" in text


@pytest.mark.anyio
async def test_ctx_command_clear_disabled_keeps_context(tmp_path: Path) -> None:
    transport = FakeTransport()
    cfg = replace(
        make_cfg(transport),
        topics=TelegramTopicsSettings(enabled=True, scope="all"),
    )
    store = TopicStateStore(tmp_path / "topics.json")
    context = RunContext(project="alpha", branch="dev")
    msg = _msg("/ctx clear", thread_id=456, chat_type="supergroup")

    await store.set_context(msg.chat_id, msg.thread_id, context)
    await _handle_ctx_command(
        cfg,
        msg,
        args_text="clear",
        store=store,
        resolved_scope="all",
        scope_chat_ids=frozenset({msg.chat_id}),
    )

    text = transport.send_calls[-1]["message"].text
    assert (
        text.strip()
        == "Robob's context is locked per-topic. /ctx clear is disabled."
    )
    snapshot = await store.get_thread(msg.chat_id, msg.thread_id)
    assert snapshot is not None
    assert snapshot.context == context


@pytest.mark.anyio
async def test_chat_ctx_command_clear_disabled_keeps_context(tmp_path: Path) -> None:
    transport = FakeTransport()
    runtime, state_path = _runtime(tmp_path)
    cfg = replace(make_cfg(transport), runtime=runtime, session_mode="chat")
    store = ChatPrefsStore(resolve_prefs_path(state_path))
    context = RunContext(project="alpha", branch="dev")
    msg = _msg("/ctx clear", chat_type="private")

    await store.set_context(msg.chat_id, context)
    await _handle_chat_ctx_command(
        cfg,
        msg,
        args_text="clear",
        chat_prefs=store,
    )

    text = transport.send_calls[-1]["message"].text
    assert (
        text.strip()
        == "Robob's context is locked per-topic. /ctx clear is disabled."
    )
    bound = await store.get_context(msg.chat_id)
    assert bound == context


@pytest.mark.anyio
async def test_new_command_requires_topic(tmp_path: Path) -> None:
    transport = FakeTransport()
    cfg = replace(
        make_cfg(transport),
        topics=TelegramTopicsSettings(enabled=True, scope="all"),
    )
    store = TopicStateStore(tmp_path / "topics.json")
    msg = _msg("/new")

    await _handle_new_command(
        cfg,
        msg,
        store=store,
        resolved_scope="all",
        scope_chat_ids=frozenset({msg.chat_id}),
    )

    text = transport.send_calls[-1]["message"].text
    assert "only works inside a topic" in text


@pytest.mark.anyio
async def test_chat_new_command_no_sessions(tmp_path: Path) -> None:
    transport = FakeTransport()
    cfg = make_cfg(transport)
    store = ChatSessionStore(tmp_path / "sessions.json")
    msg = _msg("/new", chat_type="private")

    await _handle_chat_new_command(cfg, msg, store, session_key=None)

    text = transport.send_calls[-1]["message"].text
    assert "no stored sessions" in text


@pytest.mark.anyio
async def test_chat_new_command_group_clears(tmp_path: Path) -> None:
    transport = FakeTransport()
    cfg = make_cfg(transport)
    store = ChatSessionStore(tmp_path / "sessions.json")
    msg = _msg("/new", chat_type="supergroup")

    await _handle_chat_new_command(cfg, msg, store, session_key=(msg.chat_id, 1))

    text = transport.send_calls[-1]["message"].text
    assert "cleared stored sessions for you in this chat" in text


@pytest.mark.anyio
async def test_topic_command_requires_args(tmp_path: Path) -> None:
    transport = FakeTransport()
    cfg = replace(
        make_cfg(transport),
        topics=TelegramTopicsSettings(enabled=True, scope="all"),
    )
    store = TopicStateStore(tmp_path / "topics.json")
    msg = _msg("/topic")

    await _handle_topic_command(
        cfg,
        msg,
        args_text="",
        store=store,
        resolved_scope="all",
        scope_chat_ids=frozenset({msg.chat_id}),
    )

    text = transport.send_calls[-1]["message"].text
    assert "usage: /topic" in text


@pytest.mark.anyio
async def test_pause_command_requires_topic(tmp_path: Path) -> None:
    transport = FakeTransport()
    cfg = replace(
        make_cfg(transport),
        topics=TelegramTopicsSettings(enabled=True, scope="all"),
    )
    store = TopicStateStore(tmp_path / "topics.json")
    msg = _msg("/pause")

    await _handle_pause_command(
        cfg,
        msg,
        store=store,
        resolved_scope="all",
        scope_chat_ids=frozenset({msg.chat_id}),
    )

    text = transport.send_calls[-1]["message"].text
    assert "only works inside a topic" in text


@pytest.mark.anyio
async def test_pause_resume_command_toggle(tmp_path: Path) -> None:
    transport = FakeTransport()
    cfg = replace(
        make_cfg(transport),
        topics=TelegramTopicsSettings(enabled=True, scope="all"),
    )
    store = TopicStateStore(tmp_path / "topics.json")
    msg = _msg("/pause", thread_id=456, chat_type="supergroup")

    await _handle_pause_command(
        cfg,
        msg,
        store=store,
        resolved_scope="all",
        scope_chat_ids=frozenset({msg.chat_id}),
    )

    assert await store.get_paused(msg.chat_id, msg.thread_id)
    text = transport.send_calls[-1]["message"].text
    assert "paused this topic" in text

    msg_resume = _msg("/resume", thread_id=456, chat_type="supergroup")
    await _handle_resume_command(
        cfg,
        msg_resume,
        store=store,
        resolved_scope="all",
        scope_chat_ids=frozenset({msg.chat_id}),
    )

    assert not await store.get_paused(msg.chat_id, msg.thread_id)
    text = transport.send_calls[-1]["message"].text
    assert "resumed this topic" in text


@pytest.mark.anyio
async def test_allrepos_requires_query(tmp_path: Path) -> None:
    transport = FakeTransport()
    projects = {
        "alpha": ProjectConfig(
            alias="alpha",
            path=tmp_path,
            worktrees_dir=Path(".worktrees"),
        )
    }
    cfg = _cfg_with_projects(transport, tmp_path, projects)
    store = TopicStateStore(tmp_path / "topics.json")
    msg = _msg("/allrepos")

    async with anyio.create_task_group() as tg:
        await _handle_allrepos_command(
            cfg,
            msg,
            args_text="",
            store=store,
            chat_prefs=None,
            running_tasks={},
            task_group=tg,
            resolved_scope="all",
            scope_chat_ids=frozenset({msg.chat_id}),
        )

    text = transport.send_calls[-1]["message"].text
    assert "usage: /allrepos" in text


@pytest.mark.anyio
async def test_allrepos_rejects_directives(tmp_path: Path) -> None:
    transport = FakeTransport()
    projects = {
        "alpha": ProjectConfig(
            alias="alpha",
            path=tmp_path,
            worktrees_dir=Path(".worktrees"),
        )
    }
    cfg = _cfg_with_projects(transport, tmp_path, projects)
    store = TopicStateStore(tmp_path / "topics.json")
    msg = _msg("/allrepos")

    async with anyio.create_task_group() as tg:
        await _handle_allrepos_command(
            cfg,
            msg,
            args_text="/codex do work",
            store=store,
            chat_prefs=None,
            running_tasks={},
            task_group=tg,
            resolved_scope="all",
            scope_chat_ids=frozenset({msg.chat_id}),
        )

    text = transport.send_calls[-1]["message"].text
    assert "does not allow directives" in text


@pytest.mark.anyio
async def test_allrepos_fails_on_scope_mismatch(tmp_path: Path) -> None:
    transport = FakeTransport()
    projects = {
        "alpha": ProjectConfig(
            alias="alpha",
            path=tmp_path,
            worktrees_dir=Path(".worktrees"),
            chat_id=999,
        )
    }
    cfg = _cfg_with_projects(transport, tmp_path, projects, scope="main")
    store = TopicStateStore(tmp_path / "topics.json")
    msg = _msg("/allrepos")
    called = False

    async def create_forum_topic(chat_id: int, name: str):
        nonlocal called
        called = True
        return None

    cfg.bot.create_forum_topic = create_forum_topic

    async with anyio.create_task_group() as tg:
        await _handle_allrepos_command(
            cfg,
            msg,
            args_text="do work",
            store=store,
            chat_prefs=None,
            running_tasks={},
            task_group=tg,
            resolved_scope="main",
            scope_chat_ids=frozenset({msg.chat_id}),
        )

    text = transport.send_calls[-1]["message"].text
    assert "topics.scope" in text
    assert called is False


@pytest.mark.anyio
async def test_allrepos_starts_runs(tmp_path: Path, monkeypatch) -> None:
    transport = FakeTransport()
    projects = {
        "alpha": ProjectConfig(
            alias="alpha",
            path=tmp_path,
            worktrees_dir=Path(".worktrees"),
            chat_id=200,
        ),
        "beta": ProjectConfig(
            alias="beta",
            path=tmp_path,
            worktrees_dir=Path(".worktrees"),
            chat_id=201,
        ),
    }
    cfg = _cfg_with_projects(transport, tmp_path, projects, scope="all")
    store = TopicStateStore(tmp_path / "topics.json")
    msg = _msg("/allrepos")
    calls: list[dict[str, object]] = []

    async def fake_run_engine(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr("takopi.telegram.commands.topics._run_engine", fake_run_engine)

    async with anyio.create_task_group() as tg:
        await _handle_allrepos_command(
            cfg,
            msg,
            args_text="ship it",
            store=store,
            chat_prefs=None,
            running_tasks={},
            task_group=tg,
            resolved_scope="all",
            scope_chat_ids=frozenset({cfg.chat_id, 200, 201}),
        )

    assert len(calls) == 2
    assert {call["chat_id"] for call in calls} == {200, 201}
    assert {call["text"] for call in calls} == {"ship it"}
    contexts = {(call["context"].project, call["context"].branch) for call in calls}
    assert contexts == {("alpha", "origin/master"), ("beta", "origin/master")}


@pytest.mark.anyio
async def test_allrepos_no_projects(tmp_path: Path) -> None:
    transport = FakeTransport()
    cfg = _cfg_with_projects(transport, tmp_path, {})
    store = TopicStateStore(tmp_path / "topics.json")
    msg = _msg("/allrepos")

    async with anyio.create_task_group() as tg:
        await _handle_allrepos_command(
            cfg,
            msg,
            args_text="do work",
            store=store,
            chat_prefs=None,
            running_tasks={},
            task_group=tg,
            resolved_scope="all",
            scope_chat_ids=frozenset({msg.chat_id}),
        )

    text = transport.send_calls[-1]["message"].text
    assert "no projects configured" in text


@pytest.mark.anyio
async def test_allrepos_rejects_branch_directive(tmp_path: Path) -> None:
    transport = FakeTransport()
    projects = {
        "alpha": ProjectConfig(
            alias="alpha",
            path=tmp_path,
            worktrees_dir=Path(".worktrees"),
        )
    }
    cfg = _cfg_with_projects(transport, tmp_path, projects)
    store = TopicStateStore(tmp_path / "topics.json")
    msg = _msg("/allrepos")

    async with anyio.create_task_group() as tg:
        await _handle_allrepos_command(
            cfg,
            msg,
            args_text="@feature do work",
            store=store,
            chat_prefs=None,
            running_tasks={},
            task_group=tg,
            resolved_scope="all",
            scope_chat_ids=frozenset({msg.chat_id}),
        )

    text = transport.send_calls[-1]["message"].text
    assert "does not allow directives" in text


@pytest.mark.anyio
async def test_allrepos_uses_default_chat_id(tmp_path: Path, monkeypatch) -> None:
    transport = FakeTransport()
    projects = {
        "alpha": ProjectConfig(
            alias="alpha",
            path=tmp_path,
            worktrees_dir=Path(".worktrees"),
        )
    }
    cfg = _cfg_with_projects(transport, tmp_path, projects, scope="all")
    store = TopicStateStore(tmp_path / "topics.json")
    msg = _msg("/allrepos", chat_id=555)
    calls: list[dict[str, object]] = []

    async def fake_run_engine(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr("takopi.telegram.commands.topics._run_engine", fake_run_engine)

    async with anyio.create_task_group() as tg:
        await _handle_allrepos_command(
            cfg,
            msg,
            args_text="ship it",
            store=store,
            chat_prefs=None,
            running_tasks={},
            task_group=tg,
            resolved_scope="all",
            scope_chat_ids=frozenset({msg.chat_id}),
        )

    assert {call["chat_id"] for call in calls} == {msg.chat_id}


@pytest.mark.anyio
async def test_allrepos_skips_run_when_seed_missing(
    tmp_path: Path, monkeypatch
) -> None:
    transport = FakeTransport()
    projects = {
        "alpha": ProjectConfig(
            alias="alpha",
            path=tmp_path,
            worktrees_dir=Path(".worktrees"),
            chat_id=200,
        )
    }
    cfg = _cfg_with_projects(transport, tmp_path, projects, scope="all")
    store = TopicStateStore(tmp_path / "topics.json")
    msg = _msg("/allrepos")
    calls: list[dict[str, object]] = []

    async def fake_run_engine(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr("takopi.telegram.commands.topics._run_engine", fake_run_engine)
    original_send = transport.send

    async def send(*, channel_id, message, options=None):
        if options is not None and options.thread_id is not None:
            return None
        return await original_send(
            channel_id=channel_id,
            message=message,
            options=options,
        )

    monkeypatch.setattr(transport, "send", send)

    async with anyio.create_task_group() as tg:
        await _handle_allrepos_command(
            cfg,
            msg,
            args_text="ship it",
            store=store,
            chat_prefs=None,
            running_tasks={},
            task_group=tg,
            resolved_scope="all",
            scope_chat_ids=frozenset({cfg.chat_id, 200}),
        )

    assert calls == []


@pytest.mark.anyio
async def test_allrepos_reports_topic_creation_failure(
    tmp_path: Path, monkeypatch
) -> None:
    transport = FakeTransport()
    projects = {
        "alpha": ProjectConfig(
            alias="alpha",
            path=tmp_path,
            worktrees_dir=Path(".worktrees"),
        )
    }
    cfg = _cfg_with_projects(transport, tmp_path, projects, scope="all")
    store = TopicStateStore(tmp_path / "topics.json")
    msg = _msg("/allrepos")
    calls: list[dict[str, object]] = []

    async def create_forum_topic(chat_id: int, name: str):
        _ = chat_id
        _ = name
        return None

    async def fake_run_engine(**kwargs):
        calls.append(kwargs)

    cfg.bot.create_forum_topic = create_forum_topic
    monkeypatch.setattr("takopi.telegram.commands.topics._run_engine", fake_run_engine)

    async with anyio.create_task_group() as tg:
        await _handle_allrepos_command(
            cfg,
            msg,
            args_text="do work",
            store=store,
            chat_prefs=None,
            running_tasks={},
            task_group=tg,
            resolved_scope="all",
            scope_chat_ids=frozenset({msg.chat_id}),
        )

    text = transport.send_calls[-1]["message"].text
    assert "failed to create topics" in text
    assert calls == []
