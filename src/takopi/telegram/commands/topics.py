from __future__ import annotations

from typing import TYPE_CHECKING

from ...context import RunContext
from ...markdown import MarkdownParts
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
from ..files import split_command_args
from ..render import prepare_telegram
from ..topic_state import TopicStateStore
from ..topics import (
    _maybe_rename_topic,
    _topic_key,
    _topic_title,
    _topics_chat_project,
    _topics_command_error,
)
from ..types import TelegramIncomingMessage
from .reply import make_reply

if TYPE_CHECKING:
    from ..bridge import TelegramBridgeConfig


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
        await store.clear_context(*tkey)
        await reply(text="topic binding cleared.")
        return
    await reply(
        text="unknown `/ctx` command. use `/ctx`, `/ctx set`, or `/ctx clear`.",
    )


def _parse_chat_ctx_args(
    args_text: str,
    *,
    runtime: TransportRuntime,
    default_project: str | None,
) -> tuple[RunContext | None, str | None]:
    tokens = split_command_args(args_text)
    if not tokens:
        return None, None
    action = tokens[0].lower()
    if action != "set":
        return None, f"unknown `/ctx` command. use `/ctx` or `/ctx set`."
    rest = " ".join(tokens[1:])
    context, error = _parse_project_branch_args(
        rest,
        runtime=runtime,
        require_branch=True,
        chat_project=default_project,
    )
    if error is not None:
        return None, error
    if context is None:
        return None, _usage_ctx_set(chat_project=default_project)
    return context, None


async def _handle_chat_ctx_command(
    cfg: TelegramBridgeConfig,
    msg: TelegramIncomingMessage,
    args_text: str,
    chat_prefs: ChatPrefsStore,
) -> None:
    reply = make_reply(cfg, msg)
    tokens = split_command_args(args_text)
    action = tokens[0].lower() if tokens else "show"
    if action in {"show", ""}:
        snapshot = await chat_prefs.get_snapshot(msg.chat_id)
        bound = snapshot.context if snapshot is not None else None
        resolved = cfg.runtime.resolve_message(
            text="",
            reply_text=msg.reply_to_text,
            chat_id=msg.chat_id,
            ambient_context=bound,
        )
        text = _format_ctx_status(
            cfg=cfg,
            runtime=cfg.runtime,
            bound=bound,
            resolved=resolved.context,
            context_source=resolved.context_source,
            snapshot=snapshot,
            chat_project=None,
        )
        await reply(text=text)
        return
    if action == "set":
        context, error = _parse_chat_ctx_args(
            args_text,
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
        await chat_prefs.clear_context(msg.chat_id)
        await reply(text="chat context cleared.")
        return
    await reply(
        text="unknown `/ctx` command. use `/ctx`, `/ctx set`, or `/ctx clear`.",
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
    tokens = split_command_args(args_text)
    if not tokens:
        await reply(text=_usage_topic(chat_project=None))
        return
    response = await _handle_topic_command_inner(
        cfg,
        msg,
        tokens,
        store,
        resolved_scope=resolved_scope,
        scope_chat_ids=scope_chat_ids,
    )
    await reply(text=response)


def _topic_command_usage(aliases: str) -> str:
    return _usage_topic(chat_project=None) + "\n" + aliases


def _topic_helptext(runtime: TransportRuntime, aliases: list[str]) -> str:
    alias_text = " "
    if aliases:
        alias_text = "\n".join(f"- `/{alias}`" for alias in aliases)
    topic_help = (
        "`/topic <alias>`: create/bind a topic to a project." "\n" + alias_text
    )
    return topic_help


async def _handle_topic_command_inner(
    cfg: TelegramBridgeConfig,
    msg: TelegramIncomingMessage,
    tokens: list[str],
    store: TopicStateStore,
    *,
    resolved_scope: str | None,
    scope_chat_ids: frozenset[int] | None,
) -> str:
    chat_project = _topics_chat_project(cfg, msg.chat_id)
    aliases = list(cfg.runtime.project_aliases())
    if tokens[0] == "help":
        return _topic_helptext(cfg.runtime, aliases)
    alias, rest, error = split_command_args(
        " ".join(tokens),
        allow_empty=False,
        allow_comments=False,
    )
    if error is not None:
        return f"error:\n{error}\n{_usage_topic(chat_project=None)}"
    if alias is None:
        return _topic_command_usage(_topic_helptext(cfg.runtime, aliases))
    tkey = _topic_key(msg, cfg, scope_chat_ids=scope_chat_ids)
    if tkey is None:
        return "this command only works inside a topic."
    alias = alias.lower()
    alias_lookup = cfg.runtime.resolve_project_alias(alias)
    if alias_lookup is None:
        alias_help = _topic_helptext(cfg.runtime, aliases)
        return f"unknown project: {alias}\n\n{alias_help}"
    context = RunContext(project=alias_lookup, branch=rest or None)
    await store.set_context(*tkey, context)
    await _maybe_rename_topic(
        cfg,
        store,
        chat_id=tkey[0],
        thread_id=tkey[1],
        context=context,
    )
    return f"bound topic to `{_format_context(cfg.runtime, context)}`"


def _render_topic_prompt_header(
    runtime: TransportRuntime,
    prompt: str,
    chat_project: str | None,
) -> str:
    context = _topic_header_context(runtime, prompt, chat_project)
    if context is None:
        return prompt
    header = _topic_header(runtime, context)
    return f"{header}\n\n{prompt}"


def _topic_header_context(
    runtime: TransportRuntime,
    prompt: str,
    chat_project: str | None,
) -> RunContext | None:
    resolved = runtime.resolve_message(
        text=prompt,
        reply_text=None,
        chat_id=None,
        ambient_context=None,
    )
    if resolved.context is not None:
        return resolved.context
    if chat_project is None:
        return None
    return RunContext(project=chat_project, branch=None)


def _topic_header(runtime: TransportRuntime, context: RunContext) -> str:
    return f"Topic ({runtime.format_context_line(context)}):"


async def _ensure_topic_header(
    runtime: TransportRuntime,
    msg: TelegramIncomingMessage,
    *,
    response_text: str,
    chat_project: str | None,
) -> RenderedMessage:
    parts = MarkdownParts(header=response_text)
    text, entities = prepare_telegram(parts)
    return RenderedMessage(text=text, extra={"entities": entities})


def _build_topic_aliases(runtime: TransportRuntime) -> str:
    aliases = list(runtime.project_aliases())
    if not aliases:
        return ""
    sorted_aliases = sorted(alias.lower() for alias in aliases)
    return "available projects: " + ", ".join(sorted_aliases)


def _topic_bot_replies(text: str) -> bool:
    return text.startswith("/")


async def _handle_topic_command_inner_for_reply(
    cfg: TelegramBridgeConfig,
    msg: TelegramIncomingMessage,
    tokens: list[str],
    store: TopicStateStore,
    *,
    resolved_scope: str | None,
    scope_chat_ids: frozenset[int] | None,
) -> tuple[str, RenderedMessage]:
    response_text = await _handle_topic_command_inner(
        cfg,
        msg,
        tokens,
        store,
        resolved_scope=resolved_scope,
        scope_chat_ids=scope_chat_ids,
    )
    chat_project = _topics_chat_project(cfg, msg.chat_id)
    if not _topic_bot_replies(response_text):
        return response_text, RenderedMessage(text=response_text, extra={})
    alias_help = _build_topic_aliases(cfg.runtime)
    text = _render_topic_prompt_header(
        runtime=cfg.runtime,
        prompt=alias_help,
        chat_project=chat_project,
    )
    message = await _ensure_topic_header(
        runtime=cfg.runtime,
        msg=msg,
        response_text=text,
        chat_project=chat_project,
    )
    return response_text, message


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
    tokens = split_command_args(args_text)
    if not tokens:
        await reply(text=_usage_topic(chat_project=None))
        return
    response_text, message = await _handle_topic_command_inner_for_reply(
        cfg,
        msg,
        tokens,
        store,
        resolved_scope=resolved_scope,
        scope_chat_ids=scope_chat_ids,
    )
    rendered_text, entities = prepare_telegram(MarkdownParts(header=response_text))
    await cfg.exec_cfg.transport.send(
        channel_id=msg.chat_id,
        message=RenderedMessage(text=rendered_text, extra={"entities": entities}),
        options=SendOptions(thread_id=msg.thread_id),
    )
