from __future__ import annotations

from typing import TYPE_CHECKING

from ...context import RunContext
from ...transport_runtime import TransportRuntime
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
from ..topic_state import TopicStateStore
from ..topics import (
    _maybe_rename_topic,
    _topic_key,
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


def _topic_helptext(runtime: TransportRuntime, aliases: list[str]) -> str:
    alias_text = ""
    if aliases:
        alias_text = "\n".join(f"- `/{alias}`" for alias in aliases)
    topic_help = "`/topic <alias> [@branch]`: bind this topic to a project."
    if alias_text:
        topic_help = f"{topic_help}\n{alias_text}"
    return topic_help


def _resolve_project_alias(runtime: TransportRuntime, alias: str) -> str | None:
    key = runtime.normalize_project_key(alias)
    if key is not None:
        return key
    alias_key = alias.strip().lower()
    for project_key, project in runtime._projects.projects.items():
        if project.alias.lower() == alias_key:
            return project_key
    return None


async def _handle_topic_command_inner(
    cfg: TelegramBridgeConfig,
    msg: TelegramIncomingMessage,
    tokens: list[str],
    store: TopicStateStore,
    *,
    resolved_scope: str | None,
    scope_chat_ids: frozenset[int] | None,
) -> str:
    aliases = list(cfg.runtime.project_aliases())
    if tokens[0].lower() == "help":
        return _topic_helptext(cfg.runtime, aliases)
    alias = tokens[0]
    rest_tokens = tokens[1:]
    if len(rest_tokens) > 1:
        return f"error:\ntoo many arguments\n{_usage_topic(chat_project=None)}"
    branch: str | None = None
    if rest_tokens:
        branch_token = rest_tokens[0]
        if not branch_token.startswith("@"):
            return f"error:\nbranch must be prefixed with @\n{_usage_topic(chat_project=None)}"
        branch = branch_token[1:] or None
    tkey = _topic_key(msg, cfg, scope_chat_ids=scope_chat_ids)
    if tkey is None:
        return "this command only works inside a topic."
    alias_lookup = _resolve_project_alias(cfg.runtime, alias)
    if alias_lookup is None:
        alias_help = _topic_helptext(cfg.runtime, aliases)
        return f"unknown project: {alias}\n\n{alias_help}"
    context = RunContext(project=alias_lookup, branch=branch)
    await store.set_context(*tkey, context)
    await _maybe_rename_topic(
        cfg,
        store,
        chat_id=tkey[0],
        thread_id=tkey[1],
        context=context,
    )
    return f"bound topic to `{_format_context(cfg.runtime, context)}`"


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
