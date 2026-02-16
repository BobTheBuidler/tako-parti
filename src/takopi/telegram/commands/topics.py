from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ...context import RunContext
from ...markdown import MarkdownParts
from ...runners.run_options import EngineRunOptions
from ...transport_runtime import TransportRuntime
from ...transport import RenderedMessage, SendOptions
from ..chat_prefs import ChatPrefsStore
from ..chat_sessions import ChatSessionStore
from ..context import (
    _format_context,
    _format_ctx_status,
    _merge_topic_context,
    _parse_project_branch_args,
    _usage_ctx_set,
    _usage_topic,
)
from ..engine_defaults import resolve_engine_for_message
from ..engine_overrides import merge_overrides
from ..files import split_command_args
from ..render import prepare_telegram
from ..topic_state import TopicStateStore
from ..topics import (
    _maybe_rename_topic,
    _resolve_topics_scope,
    _topic_key,
    _topic_title,
    _topics_chat_project,
    _topics_command_error,
    _topics_scope_label,
)
from ..types import TelegramIncomingMessage
from .executor import _run_engine
from .reply import make_reply

if TYPE_CHECKING:
    from ..bridge import TelegramBridgeConfig
    from ..chat_prefs import ChatPrefsStore
    from ...runner_bridge import RunningTasks


async def _handle_ctx_command(
    cfg: TelegramBridgeConfig,
    msg: TelegramIncomingMessage,
    args_text: str,
    store: TopicStateStore,
    *,
    resolved_scope: str | None = None,
    scope_chat_ids: frozenset[int] | None = None,
) -> None:
    reply = make_reply(cfg, msg)
    error = _topics_command_error(
        cfg,
        msg.chat_id,
        resolved_scope=resolved_scope,
        scope_chat_ids=scope_chat_ids,
    )
    if error is not None:
        await reply(text=error)
        return
    chat_project = _topics_chat_project(cfg, msg.chat_id)
    tkey = _topic_key(msg, cfg, scope_chat_ids=scope_chat_ids)
    if tkey is None:
        await reply(text="this command only works inside a topic.")
        return
    tokens = split_command_args(args_text)
    action = tokens[0].lower() if tokens else "show"
    if action in {"show", ""}:
        snapshot = await store.get_thread(*tkey)
        bound = snapshot.context if snapshot is not None else None
        ambient = _merge_topic_context(chat_project=chat_project, bound=bound)
        resolved = cfg.runtime.resolve_message(
            text="",
            reply_text=msg.reply_to_text,
            chat_id=msg.chat_id,
            ambient_context=ambient,
        )
        text = _format_ctx_status(
            cfg=cfg,
            runtime=cfg.runtime,
            bound=bound,
            resolved=resolved.context,
            context_source=resolved.context_source,
            snapshot=snapshot,
            chat_project=chat_project,
        )
        await reply(text=text)
        return
    if action == "set":
        rest = " ".join(tokens[1:])
        context, error = _parse_project_branch_args(
            rest,
            runtime=cfg.runtime,
            require_branch=False,
            chat_project=chat_project,
        )
        if error is not None:
            await reply(
                text=f"error:\n{error}\n{_usage_ctx_set(chat_project=chat_project)}",
            )
            return
        if context is None:
            await reply(
                text=f"error:\n{_usage_ctx_set(chat_project=chat_project)}",
            )
            return
        await store.set_context(*tkey, context)
        await _maybe_rename_topic(
            cfg,
            store,
            chat_id=tkey[0],
            thread_id=tkey[1],
            context=context,
        )
        await reply(
            text=f"topic bound to `{_format_context(cfg.runtime, context)}`",
        )
        return
    if action == "clear":
        await reply(
            text="Robob's context is locked per-topic. /ctx clear is disabled.",
        )
        return
    await reply(
        text="unknown `/ctx` command. use `/ctx` or `/ctx set`.",
    )


def _parse_chat_ctx_args(
    args_text: str,
    *,
    runtime: TransportRuntime,
    default_project: str | None,
) -> tuple[RunContext | None, str | None]:
    tokens = split_command_args(args_text)
    if not tokens:
        return None, _usage_ctx_set(chat_project=None)
    if len(tokens) > 2:
        return None, "too many arguments"
    project_token: str | None = None
    branch: str | None = None
    first = tokens[0]
    if first.startswith("@"):
        branch = first[1:] or None
    else:
        project_token = first
        if len(tokens) == 2:
            second = tokens[1]
            if not second.startswith("@"):
                return None, "branch must be prefixed with @"
            branch = second[1:] or None
    project_key: str | None = None
    if project_token is None:
        if default_project is None:
            return None, "project is required"
        project_key = default_project
    else:
        project_key = runtime.normalize_project_key(project_token)
        if project_key is None:
            return None, f"unknown project {project_token!r}"
    return RunContext(project=project_key, branch=branch), None


async def _handle_chat_ctx_command(
    cfg: TelegramBridgeConfig,
    msg: TelegramIncomingMessage,
    args_text: str,
    chat_prefs: ChatPrefsStore | None,
) -> None:
    reply = make_reply(cfg, msg)
    if chat_prefs is None:
        await reply(text="chat context unavailable; config path is not set.")
        return

    tokens = split_command_args(args_text)
    action = tokens[0].lower() if tokens else "show"
    if action in {"show", ""}:
        bound = await chat_prefs.get_context(msg.chat_id)
        resolved = cfg.runtime.resolve_message(
            text="",
            reply_text=msg.reply_to_text,
            chat_id=msg.chat_id,
            ambient_context=bound,
        )
        source = resolved.context_source
        if bound is not None and resolved.context_source == "ambient":
            source = "bound"
        lines = [
            f"bound ctx: {_format_context(cfg.runtime, bound)}",
            f"resolved ctx: {_format_context(cfg.runtime, resolved.context)} (source: {source})",
        ]
        if bound is None:
            ctx_usage = (
                _usage_ctx_set(chat_project=None).removeprefix("usage: ").strip()
            )
            lines.append(f"note: no bound context — bind with {ctx_usage}")
        await reply(text="\n".join(lines))
        return
    if action == "set":
        rest = " ".join(tokens[1:])
        context, error = _parse_chat_ctx_args(
            rest,
            runtime=cfg.runtime,
            default_project=cfg.runtime.default_project,
        )
        if error is not None:
            await reply(
                text=f"error:\n{error}\n{_usage_ctx_set(chat_project=None)}",
            )
            return
        if context is None:
            await reply(text=f"error:\n{_usage_ctx_set(chat_project=None)}")
            return
        await chat_prefs.set_context(msg.chat_id, context)
        await reply(
            text=f"chat bound to `{_format_context(cfg.runtime, context)}`",
        )
        return
    if action == "clear":
        await reply(
            text="Robob's context is locked per-topic. /ctx clear is disabled.",
        )
        return
    await reply(
        text="unknown `/ctx` command. use `/ctx` or `/ctx set`.",
    )


async def _handle_new_command(
    cfg: TelegramBridgeConfig,
    msg: TelegramIncomingMessage,
    store: TopicStateStore,
    *,
    resolved_scope: str | None = None,
    scope_chat_ids: frozenset[int] | None = None,
) -> None:
    reply = make_reply(cfg, msg)
    error = _topics_command_error(
        cfg,
        msg.chat_id,
        resolved_scope=resolved_scope,
        scope_chat_ids=scope_chat_ids,
    )
    if error is not None:
        await reply(text=error)
        return
    tkey = _topic_key(msg, cfg, scope_chat_ids=scope_chat_ids)
    if tkey is None:
        await reply(text="this command only works inside a topic.")
        return
    await store.clear_sessions(*tkey)
    await reply(text="cleared stored sessions for this topic.")


async def _handle_pause_command(
    cfg: TelegramBridgeConfig,
    msg: TelegramIncomingMessage,
    store: TopicStateStore,
    *,
    resolved_scope: str | None = None,
    scope_chat_ids: frozenset[int] | None = None,
) -> None:
    reply = make_reply(cfg, msg)
    error = _topics_command_error(
        cfg,
        msg.chat_id,
        resolved_scope=resolved_scope,
        scope_chat_ids=scope_chat_ids,
    )
    if error is not None:
        await reply(text=error)
        return
    tkey = _topic_key(msg, cfg, scope_chat_ids=scope_chat_ids)
    if tkey is None:
        await reply(text="this command only works inside a topic.")
        return
    if await store.get_paused(*tkey):
        await reply(text="topic already paused.")
        return
    await store.set_paused(*tkey, True)
    await reply(text="paused this topic. send `/resume` to continue.")


async def _handle_resume_command(
    cfg: TelegramBridgeConfig,
    msg: TelegramIncomingMessage,
    store: TopicStateStore,
    *,
    resolved_scope: str | None = None,
    scope_chat_ids: frozenset[int] | None = None,
) -> None:
    reply = make_reply(cfg, msg)
    error = _topics_command_error(
        cfg,
        msg.chat_id,
        resolved_scope=resolved_scope,
        scope_chat_ids=scope_chat_ids,
    )
    if error is not None:
        await reply(text=error)
        return
    tkey = _topic_key(msg, cfg, scope_chat_ids=scope_chat_ids)
    if tkey is None:
        await reply(text="this command only works inside a topic.")
        return
    if not await store.get_paused(*tkey):
        await reply(text="topic already running.")
        return
    await store.set_paused(*tkey, False)
    await reply(text="resumed this topic.")


async def _handle_chat_new_command(
    cfg: TelegramBridgeConfig,
    msg: TelegramIncomingMessage,
    store: ChatSessionStore,
    session_key: tuple[int, int | None] | None,
) -> None:
    reply = make_reply(cfg, msg)
    if session_key is None:
        await reply(text="no stored sessions to clear for this chat.")
        return
    await store.clear_sessions(session_key[0], session_key[1])
    if msg.chat_type == "private":
        text = "cleared stored sessions for this chat."
    else:
        text = "cleared stored sessions for you in this chat."
    await reply(text=text)


async def _handle_topic_command(
    cfg: TelegramBridgeConfig,
    msg: TelegramIncomingMessage,
    args_text: str,
    store: TopicStateStore,
    *,
    resolved_scope: str | None = None,
    scope_chat_ids: frozenset[int] | None = None,
) -> None:
    reply = make_reply(cfg, msg)
    error = _topics_command_error(
        cfg,
        msg.chat_id,
        resolved_scope=resolved_scope,
        scope_chat_ids=scope_chat_ids,
    )
    if error is not None:
        await reply(text=error)
        return
    chat_project = _topics_chat_project(cfg, msg.chat_id)
    context, error = _parse_project_branch_args(
        args_text,
        runtime=cfg.runtime,
        require_branch=True,
        chat_project=chat_project,
    )
    if error is not None or context is None:
        usage = _usage_topic(chat_project=chat_project)
        text = f"error:\n{error}\n{usage}" if error else usage
        await reply(text=text)
        return
    title = _topic_title(runtime=cfg.runtime, context=context)
    existing = await store.find_thread_for_context(msg.chat_id, context)
    stale_thread_id: int | None = None
    if existing is not None:
        updated = await cfg.bot.edit_forum_topic(
            chat_id=msg.chat_id,
            message_thread_id=existing,
            name=title,
        )
        if updated:
            await reply(
                text=f"topic already exists for {_format_context(cfg.runtime, context)} "
                "in this chat.",
            )
            return
        stale_thread_id = existing
    created = await cfg.bot.create_forum_topic(msg.chat_id, title)
    if created is None:
        await reply(text="failed to create topic.")
        return
    thread_id = created.message_thread_id
    if stale_thread_id is not None:
        await store.delete_thread(msg.chat_id, stale_thread_id)
    await store.set_context(
        msg.chat_id,
        thread_id,
        context,
        topic_title=title,
    )
    await reply(text=f"created topic `{title}`.")
    bound_text = f"topic bound to `{_format_context(cfg.runtime, context)}`"
    rendered_text, entities = prepare_telegram(MarkdownParts(header=bound_text))
    await cfg.exec_cfg.transport.send(
        channel_id=msg.chat_id,
        message=RenderedMessage(text=rendered_text, extra={"entities": entities}),
        options=SendOptions(thread_id=thread_id),
    )
ALLREPOS_BRANCH = "origin/master"
ALLREPOS_USAGE = "usage: `/allrepos <query>`"


@dataclass(slots=True)
class _AllReposTarget:
    project_key: str
    alias: str
    chat_id: int
    context: RunContext


@dataclass(slots=True)
class _AllReposTopic:
    target: _AllReposTarget
    thread_id: int


def _usage_allrepos() -> str:
    return ALLREPOS_USAGE


def _allrepos_has_directives(cfg: TelegramBridgeConfig, query: str) -> bool:
    if not query:
        return False
    lines = query.splitlines()
    idx = next((i for i, line in enumerate(lines) if line.strip()), None)
    if idx is None:
        return False
    tokens = lines[idx].lstrip().split()
    if not tokens:
        return False
    engine_ids = {engine.lower() for engine in cfg.runtime.engine_ids}
    project_aliases = {alias.lower() for alias in cfg.runtime.project_aliases()}
    for token in tokens:
        if token.startswith("/"):
            name = token[1:]
            if "@" in name:
                name = name.split("@", 1)[0]
            if not name:
                break
            key = name.lower()
            if key in engine_ids or key in project_aliases:
                return True
            break
        if token.startswith("@"):
            return bool(token[1:])
        break
    return False


def _allrepos_show_resume_line(cfg: TelegramBridgeConfig, context: RunContext) -> bool:
    if cfg.show_resume_line:
        return True
    return context.project is None


async def _ensure_topic_for_context(
    cfg: TelegramBridgeConfig,
    store: TopicStateStore,
    *,
    chat_id: int,
    context: RunContext,
) -> tuple[int | None, str | None]:
    title = _topic_title(runtime=cfg.runtime, context=context)
    existing = await store.find_thread_for_context(chat_id, context)
    stale_thread_id: int | None = None
    if existing is not None:
        updated = await cfg.bot.edit_forum_topic(
            chat_id=chat_id,
            message_thread_id=existing,
            name=title,
        )
        if updated:
            return existing, None
        stale_thread_id = existing
    created = await cfg.bot.create_forum_topic(chat_id, title)
    if created is None:
        return None, "failed to create topic"
    thread_id = created.message_thread_id
    if stale_thread_id is not None:
        await store.delete_thread(chat_id, stale_thread_id)
    await store.set_context(
        chat_id,
        thread_id,
        context,
        topic_title=title,
    )
    return thread_id, None


async def _resolve_allrepos_run_options(
    *,
    chat_id: int,
    thread_id: int,
    engine: str,
    topic_store: TopicStateStore | None,
    chat_prefs: ChatPrefsStore | None,
) -> EngineRunOptions | None:
    topic_override = None
    if topic_store is not None:
        topic_override = await topic_store.get_engine_override(
            chat_id, thread_id, engine
        )
    chat_override = None
    if chat_prefs is not None:
        chat_override = await chat_prefs.get_engine_override(chat_id, engine)
    merged = merge_overrides(topic_override, chat_override)
    if merged is None:
        return None
    return EngineRunOptions(model=merged.model, reasoning=merged.reasoning)


async def _run_allrepos_query(
    cfg: TelegramBridgeConfig,
    entry: _AllReposTopic,
    query: str,
    topic_store: TopicStateStore,
    chat_prefs: ChatPrefsStore | None,
    running_tasks: RunningTasks,
) -> None:
    chat_id = entry.target.chat_id
    thread_id = entry.thread_id
    context = entry.target.context
    seed = await cfg.exec_cfg.transport.send(
        channel_id=chat_id,
        message=RenderedMessage(text=query, extra={}),
        options=SendOptions(thread_id=thread_id),
    )
    if seed is None:
        return
    engine_resolution = await resolve_engine_for_message(
        runtime=cfg.runtime,
        context=context,
        explicit_engine=None,
        chat_id=chat_id,
        topic_key=(chat_id, thread_id),
        topic_store=topic_store,
        chat_prefs=chat_prefs,
    )
    run_options = await _resolve_allrepos_run_options(
        chat_id=chat_id,
        thread_id=thread_id,
        engine=engine_resolution.engine,
        topic_store=topic_store,
        chat_prefs=chat_prefs,
    )

    async def on_thread_known(token, done) -> None:
        await topic_store.set_session_resume(chat_id, thread_id, token)

    await _run_engine(
        exec_cfg=cfg.exec_cfg,
        runtime=cfg.runtime,
        running_tasks=running_tasks,
        chat_id=chat_id,
        user_msg_id=seed.message_id,
        text=query,
        resume_token=None,
        context=context,
        reply_ref=None,
        on_thread_known=on_thread_known,
        engine_override=engine_resolution.engine,
        thread_id=thread_id,
        show_resume_line=_allrepos_show_resume_line(cfg, context),
        progress_ref=None,
        run_options=run_options,
    )


async def _handle_allrepos_command(
    cfg: TelegramBridgeConfig,
    msg: TelegramIncomingMessage,
    args_text: str,
    store: TopicStateStore,
    chat_prefs: ChatPrefsStore | None,
    running_tasks: RunningTasks,
    task_group,
    *,
    resolved_scope: str | None = None,
    scope_chat_ids: frozenset[int] | None = None,
) -> None:
    reply = make_reply(cfg, msg)
    error = _topics_command_error(
        cfg,
        msg.chat_id,
        resolved_scope=resolved_scope,
        scope_chat_ids=scope_chat_ids,
    )
    if error is not None:
        await reply(text=error)
        return
    query = args_text.strip()
    if not query:
        await reply(text=_usage_allrepos())
        return
    if _allrepos_has_directives(cfg, query):
        await reply(
            text=(
                "error:\n"
                "/allrepos does not allow directives like /engine, /project, or "
                "@branch.\n"
                f"{_usage_allrepos()}"
            )
        )
        return
    targets: list[_AllReposTarget] = []
    for alias in cfg.runtime.project_aliases():
        project_key = cfg.runtime.normalize_project_key(alias)
        if project_key is None:
            continue
        project_cfg = cfg.runtime.project_config(project_key)
        if project_cfg is None:
            continue
        chat_id = project_cfg.chat_id or msg.chat_id
        targets.append(
            _AllReposTarget(
                project_key=project_key,
                alias=project_cfg.alias,
                chat_id=chat_id,
                context=RunContext(project=project_key, branch=ALLREPOS_BRANCH),
            )
        )
    if not targets:
        await reply(text="no projects configured.")
        return
    targets.sort(key=lambda target: target.alias.lower())
    if scope_chat_ids is None:
        _, scope_chat_ids = _resolve_topics_scope(cfg)
    invalid = [
        f"{target.alias} ({target.chat_id})"
        for target in targets
        if target.chat_id not in scope_chat_ids
    ]
    if invalid:
        scope_label = _topics_scope_label(cfg)
        await reply(
            text=(
                "error:\n"
                "/allrepos targets chats outside the configured topics.scope.\n"
                f"scope: {scope_label}\n"
                f"invalid: {', '.join(invalid)}"
            )
        )
        return
    failures: list[str] = []
    entries: list[_AllReposTopic] = []
    for target in targets:
        thread_id, error = await _ensure_topic_for_context(
            cfg,
            store,
            chat_id=target.chat_id,
            context=target.context,
        )
        if error is not None or thread_id is None:
            failures.append(f"{target.alias}: {error or 'failed to create topic'}")
            continue
        entries.append(_AllReposTopic(target=target, thread_id=thread_id))
    if failures:
        await reply(
            text="error:\nfailed to create topics:\n" + "\n".join(failures),
        )
        return
    for entry in entries:
        task_group.start_soon(
            _run_allrepos_query,
            cfg,
            entry,
            query,
            store,
            chat_prefs,
            running_tasks,
        )
    started = ", ".join(entry.target.alias for entry in entries)
    await reply(
        text=(
            f"started /allrepos for {len(entries)} projects "
            f"(@{ALLREPOS_BRANCH}): {started}"
        ),
    )
