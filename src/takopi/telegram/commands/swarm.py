from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from ...config import ConfigError
from ...context import RunContext
from ...directives import DirectiveError
from ...transport_runtime import TransportRuntime
from ...utils.git import git_ok, git_run
from ..files import split_command_args
from ..topic_state import TopicStateStore
from ..topics import _topics_command_error
from ..types import TelegramIncomingMessage
from .reply import make_reply
from .topics import _handle_topic_command

SWARM_USAGE = "usage: `/swarm <count>`"


class SwarmError(RuntimeError):
    pass


def _plan_swarm_branches(
    base_branch: str,
    *,
    count: int,
    exists: Callable[[str], bool],
) -> list[str]:
    planned: list[str] = []
    idx = 2
    while len(planned) < count:
        candidate = f"{base_branch}-{idx}"
        if not exists(candidate):
            planned.append(candidate)
        idx += 1
    return planned


def _parse_swarm_count(args_text: str) -> tuple[int | None, str | None]:
    tokens = split_command_args(args_text)
    if len(tokens) != 1:
        return None, SWARM_USAGE
    try:
        count = int(tokens[0])
    except ValueError:
        return None, "count must be an integer"
    if count < 1:
        return None, "count must be >= 1"
    return count, None


def _branch_exists(root: Path, branch: str) -> bool:
    return git_ok(
        ["show-ref", "--verify", "--quiet", f"refs/heads/{branch}"],
        cwd=root,
    ) or git_ok(
        ["show-ref", "--verify", "--quiet", f"refs/remotes/origin/{branch}"],
        cwd=root,
    )


def _select_base_ref(root: Path, base_branch: str) -> str | None:
    if git_ok(
        ["show-ref", "--verify", "--quiet", f"refs/heads/{base_branch}"],
        cwd=root,
    ):
        return base_branch
    if git_ok(
        ["show-ref", "--verify", "--quiet", f"refs/remotes/origin/{base_branch}"],
        cwd=root,
    ):
        return f"origin/{base_branch}"
    return None


def _create_branch_from_base(
    root: Path,
    *,
    new_branch: str,
    base_branch: str,
) -> None:
    base_ref = _select_base_ref(root, base_branch)
    if base_ref is None:
        raise SwarmError(f"base branch not found: {base_branch}")
    result = git_run(["branch", new_branch, base_ref], cwd=root)
    if result is None:
        raise SwarmError("git not available on PATH")
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip()
        raise SwarmError(message or "git branch failed")


def _resolve_swarm_context(
    runtime: TransportRuntime,
    *,
    msg: TelegramIncomingMessage,
    ambient_context: RunContext | None,
) -> tuple[RunContext | None, str | None]:
    try:
        resolved = runtime.resolve_message(
            text="",
            reply_text=msg.reply_to_text,
            ambient_context=ambient_context,
            chat_id=msg.chat_id,
        )
    except DirectiveError as exc:
        return None, f"error:\n{exc}"
    context = resolved.context
    if context is None or context.project is None:
        return None, "error:\nproject is required"
    if context.branch is None:
        return None, "error:\nbranch is required"
    return context, None


def _resolve_project_root(
    runtime: TransportRuntime, *, context: RunContext
) -> tuple[Path | None, str | None]:
    try:
        root = runtime.resolve_run_cwd(RunContext(project=context.project, branch=None))
    except ConfigError as exc:
        return None, f"error:\n{exc}"
    if root is None:
        return None, "error:\nproject is required"
    return root, None


async def _handle_swarm_command(
    cfg,
    msg: TelegramIncomingMessage,
    args_text: str,
    store: TopicStateStore,
    *,
    ambient_context: RunContext | None,
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
    count, count_error = _parse_swarm_count(args_text)
    if count_error is not None or count is None:
        text = count_error
        if count_error != SWARM_USAGE:
            text = f"error:\n{count_error}\n{SWARM_USAGE}"
        await reply(text=text)
        return
    context, context_error = _resolve_swarm_context(
        cfg.runtime,
        msg=msg,
        ambient_context=ambient_context,
    )
    if context_error is not None or context is None:
        await reply(text=context_error or "error:\nproject is required")
        return
    assert context.project is not None
    assert context.branch is not None
    root, root_error = _resolve_project_root(cfg.runtime, context=context)
    if root_error is not None or root is None:
        await reply(text=root_error or "error:\nproject is required")
        return

    planned = _plan_swarm_branches(
        context.branch,
        count=count,
        exists=lambda branch: _branch_exists(root, branch),
    )
    project_alias = cfg.runtime.project_alias_for_key(context.project)

    for branch in planned:
        try:
            _create_branch_from_base(
                root, new_branch=branch, base_branch=context.branch
            )
        except SwarmError as exc:
            await reply(text=f"error:\n{exc}")
            return
        await _handle_topic_command(
            cfg,
            msg,
            args_text=f"{project_alias} @{branch}",
            store=store,
            resolved_scope=resolved_scope,
            scope_chat_ids=scope_chat_ids,
        )
