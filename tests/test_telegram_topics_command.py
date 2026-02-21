from dataclasses import replace
from pathlib import Path

import pytest

from takopi.context import RunContext
from takopi.settings import TelegramTopicsSettings
from takopi.config import ProjectConfig, ProjectsConfig
from takopi.runners.mock import Return, ScriptRunner
from takopi.telegram.chat_sessions import ChatSessionStore
from takopi.telegram.chat_prefs import ChatPrefsStore, resolve_prefs_path
from takopi.telegram.commands.topics import (
    _handle_chat_ctx_command,
    _handle_chat_new_command,
    _handle_ctx_command,
    _handle_new_command,
    _handle_pause_command,
    _handle_resume_command,
    _handle_topic_command,
    _parse_chat_ctx_args,
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


def _runtime(tmp_path: Path) -> tuple[TransportRuntime, Path]:
    runner = ScriptRunner([Return(answer="ok")], engine=DEFAULT_ENGINE_ID)
    projects = ProjectsConfig(
        projects={
            "alpha": ProjectConfig(
                alias="Alpha",
                path=tmp_path,
                worktrees_dir=Path(".worktrees"),
            )
        },
        default_project="alpha",
    )
    state_path = tmp_path / "takopi.toml"
    runtime = TransportRuntime(
        router=_make_router(runner),
        projects=projects,
        config_path=state_path,
    )
    return runtime, state_path


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


def test_chat_ctx_args_accept_set_prefix(tmp_path: Path) -> None:
    runtime, _ = _runtime(tmp_path)
    context, error = _parse_chat_ctx_args(
        "set alpha @dev",
        runtime=runtime,
        default_project=runtime.default_project,
    )

    assert error is None
    assert context == RunContext(project="alpha", branch="dev")


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
async def test_pause_command_sets_state(tmp_path: Path) -> None:
    transport = FakeTransport()
    cfg = replace(
        make_cfg(transport),
        topics=TelegramTopicsSettings(enabled=True, scope="all"),
    )
    store = TopicStateStore(tmp_path / "topics.json")
    msg = _msg("/pause", thread_id=77)

    await _handle_pause_command(
        cfg,
        msg,
        store=store,
        resolved_scope="all",
        scope_chat_ids=frozenset({msg.chat_id}),
    )

    assert await store.get_paused(msg.chat_id, 77) is True
    text = transport.send_calls[-1]["message"].text
    assert "topic paused" in text


@pytest.mark.anyio
async def test_pause_command_reports_already_paused(tmp_path: Path) -> None:
    transport = FakeTransport()
    cfg = replace(
        make_cfg(transport),
        topics=TelegramTopicsSettings(enabled=True, scope="all"),
    )
    store = TopicStateStore(tmp_path / "topics.json")
    await store.set_paused(123, 77, True)
    msg = _msg("/pause", thread_id=77)

    await _handle_pause_command(
        cfg,
        msg,
        store=store,
        resolved_scope="all",
        scope_chat_ids=frozenset({msg.chat_id}),
    )

    text = transport.send_calls[-1]["message"].text
    assert "already paused" in text
    assert await store.get_paused(msg.chat_id, 77) is True


@pytest.mark.anyio
async def test_resume_command_reports_already_active(tmp_path: Path) -> None:
    transport = FakeTransport()
    cfg = replace(
        make_cfg(transport),
        topics=TelegramTopicsSettings(enabled=True, scope="all"),
    )
    store = TopicStateStore(tmp_path / "topics.json")
    msg = _msg("/resume", thread_id=77)

    await _handle_resume_command(
        cfg,
        msg,
        store=store,
        resolved_scope="all",
        scope_chat_ids=frozenset({msg.chat_id}),
    )

    text = transport.send_calls[-1]["message"].text
    assert "already active" in text
    assert await store.get_paused(msg.chat_id, 77) is False


@pytest.mark.anyio
async def test_resume_command_clears_state(tmp_path: Path) -> None:
    transport = FakeTransport()
    cfg = replace(
        make_cfg(transport),
        topics=TelegramTopicsSettings(enabled=True, scope="all"),
    )
    store = TopicStateStore(tmp_path / "topics.json")
    await store.set_paused(123, 77, True)
    msg = _msg("/resume", thread_id=77)

    await _handle_resume_command(
        cfg,
        msg,
        store=store,
        resolved_scope="all",
        scope_chat_ids=frozenset({msg.chat_id}),
    )

    text = transport.send_calls[-1]["message"].text
    assert "topic resumed" in text
    assert await store.get_paused(msg.chat_id, 77) is False


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
