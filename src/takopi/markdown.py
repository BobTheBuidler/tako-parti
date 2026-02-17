from __future__ import annotations

import hashlib
import os
import textwrap
from dataclasses import dataclass
from pathlib import Path

from .model import Action, ActionEvent, StartedEvent, TakopiEvent
from .progress import ProgressState
from .transport import RenderedMessage
from .utils.paths import relativize_path

STATUS = {"running": "▸", "update": "↻", "done": "✓", "fail": "✗"}
STATUS_EMOJI = {
    "starting": "😮",
    "working": "🤖",
    "queued": "😴",
    "done": "💪",
    "error": "🤬",
    "cancelled": "🛑",
}
DEFAULT_STATUS_EMOJI = "ℹ️"
HEADER_SEP = " · "
HARD_BREAK = "  \n"

MAX_PROGRESS_CMD_LEN = 300
MAX_FILE_CHANGES_INLINE = 3
COMMAND_LOG_LINES = (
    "aligning circuits",
    "charging photons",
    "tuning antennas",
    "calibrating flux",
    "warming the cores",
    "dusting gears",
    "untangling wires",
    "rolling updates",
    "tightening bolts",
    "counting electrons",
    "polishing lenses",
    "booting widgets",
    "shuffling bytes",
    "checking relays",
    "sorting atoms",
    "spinning disks",
    "juggling threads",
    "patching rivets",
    "cooking kernels",
    "routing signals",
)


@dataclass(frozen=True, slots=True)
class MarkdownParts:
    header: str
    body: str | None = None
    footer: str | None = None


def assemble_markdown_parts(parts: MarkdownParts) -> str:
    return "\n\n".join(
        chunk for chunk in (parts.header, parts.body, parts.footer) if chunk
    )


def format_changed_file_path(path: str, *, base_dir: Path | None = None) -> str:
    return f"`{relativize_path(path, base_dir=base_dir)}`"


def format_elapsed(elapsed_s: float) -> str:
    total = max(0, int(elapsed_s))
    minutes, seconds = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def format_header(
    elapsed_s: float,
    item: int | None,
    *,
    emoji: str,
    status: str,
    scope: str,
) -> str:
    elapsed = format_elapsed(elapsed_s)
    parts = [emoji, status, scope]
    parts.append(elapsed)
    if item is not None:
        parts.append(f"step {item}")
    return HEADER_SEP.join(parts)


def _strip_wrapped_ticks(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("`") and stripped.endswith("`") and len(stripped) > 1:
        return stripped[1:-1].strip()
    return stripped


def _status_parts(label: str) -> tuple[str, str]:
    cleaned = _strip_wrapped_ticks(label)
    key = cleaned.lower()
    return STATUS_EMOJI.get(key, DEFAULT_STATUS_EMOJI), cleaned


def _extract_scope(context_line: str | None) -> str | None:
    if not context_line:
        return None
    for line in context_line.splitlines():
        stripped = _strip_wrapped_ticks(line)
        if not stripped.lower().startswith("ctx:"):
            continue
        content = stripped.split(":", 1)[1].strip()
        if not content:
            continue
        tokens = content.split()
        if not tokens:
            continue
        project = tokens[0]
        branch = None
        if len(tokens) >= 2:
            if tokens[1] == "@" and len(tokens) >= 3:
                branch = tokens[2]
            elif tokens[1].startswith("@"):
                branch = tokens[1][1:]
        return branch or project
    return None


def shorten(text: str, width: int | None) -> str:
    if width is None:
        return text
    if width <= 0:
        return ""
    if len(text) <= width:
        return text
    return textwrap.shorten(text, width=width, placeholder="…")


def action_status(action: Action, *, completed: bool, ok: bool | None = None) -> str:
    if not completed:
        return STATUS["running"]
    if ok is not None:
        return STATUS["done"] if ok else STATUS["fail"]
    detail = action.detail or {}
    exit_code = detail.get("exit_code")
    if isinstance(exit_code, int) and exit_code != 0:
        return STATUS["fail"]
    return STATUS["done"]


def action_suffix(action: Action) -> str:
    detail = action.detail or {}
    exit_code = detail.get("exit_code")
    if isinstance(exit_code, int) and exit_code != 0:
        return f" (exit {exit_code})"
    return ""


def format_file_change_title(action: Action, *, command_width: int | None) -> str:
    title = str(action.title or "")
    detail = action.detail or {}

    changes = detail.get("changes")
    if isinstance(changes, list) and changes:
        rendered: list[str] = []
        for raw in changes:
            path: str | None
            kind: str | None
            if isinstance(raw, dict):
                path = raw.get("path")
                kind = raw.get("kind")
            else:
                path = getattr(raw, "path", None)
                kind = getattr(raw, "kind", None)
            if not isinstance(path, str) or not path:
                continue
            verb = kind if isinstance(kind, str) and kind else "update"
            rendered.append(f"{verb} {format_changed_file_path(path)}")

        if rendered:
            if len(rendered) > MAX_FILE_CHANGES_INLINE:
                remaining = len(rendered) - MAX_FILE_CHANGES_INLINE
                rendered = rendered[:MAX_FILE_CHANGES_INLINE] + [f"…({remaining} more)"]
            inline = shorten(", ".join(rendered), command_width)
            return f"files: {inline}"

    fallback = title
    relativized = relativize_path(fallback)
    was_relativized = relativized != fallback
    if was_relativized:
        fallback = relativized
    if (
        fallback
        and not (fallback.startswith("`") and fallback.endswith("`"))
        and (was_relativized or os.sep in fallback or "/" in fallback)
    ):
        fallback = f"`{fallback}`"
    return f"files: {shorten(fallback, command_width)}"


def format_action_title(action: Action, *, command_width: int | None) -> str:
    title = str(action.title or "")
    kind = action.kind
    if kind == "command":
        title = shorten(title, command_width)
        return f"`{title}`"
    if kind == "tool":
        title = shorten(title, command_width)
        return f"tool: {title}"
    if kind == "web_search":
        title = shorten(title, command_width)
        return f"searched: {title}"
    if kind == "subagent":
        title = shorten(title, command_width)
        return f"subagent: {title}"
    if kind == "file_change":
        return format_file_change_title(action, command_width=command_width)
    if kind in {"note", "warning"}:
        return shorten(title, command_width)
    return shorten(title, command_width)


def format_action_line(
    action: Action,
    phase: str,
    ok: bool | None,
    *,
    command_width: int | None,
) -> str:
    if phase != "completed":
        status = STATUS["update"] if phase == "updated" else STATUS["running"]
        return f"{status} {format_action_title(action, command_width=command_width)}"
    status = action_status(action, completed=True, ok=ok)
    suffix = action_suffix(action)
    return (
        f"{status} {format_action_title(action, command_width=command_width)}{suffix}"
    )


def render_event_cli(event: TakopiEvent) -> list[str]:
    match event:
        case StartedEvent(engine=engine):
            return [str(engine)]
        case ActionEvent() as action_event:
            action = action_event.action
            if action.kind == "turn":
                return []
            if action.kind == "command":
                action = Action(
                    id=action.id,
                    kind=action.kind,
                    title=_command_log_line(action),
                    detail=action.detail,
                )
            return [
                format_action_line(
                    action,
                    action_event.phase,
                    action_event.ok,
                    command_width=MAX_PROGRESS_CMD_LEN,
                )
            ]
        case _:
            return []


def _command_log_line(action: Action) -> str:
    seed = action.id or action.title or "command"
    digest = hashlib.sha256(seed.encode("utf-8")).digest()
    idx = int.from_bytes(digest[:4], "big") % len(COMMAND_LOG_LINES)
    return COMMAND_LOG_LINES[idx]


class MarkdownFormatter:
    def __init__(
        self,
        *,
        max_actions: int = 5,
        command_width: int | None = MAX_PROGRESS_CMD_LEN,
    ) -> None:
        self.max_actions = max(0, int(max_actions))
        self.command_width = command_width

    def render_progress_parts(
        self,
        state: ProgressState,
        *,
        elapsed_s: float,
        label: str = "working",
    ) -> MarkdownParts:
        step = state.action_count or None
        emoji, status = _status_parts(label)
        scope = _extract_scope(state.context_line) or state.engine
        header = format_header(
            elapsed_s,
            step,
            emoji=emoji,
            status=status,
            scope=scope,
        )
        body = self._assemble_body(self._format_actions(state))
        return MarkdownParts(
            header=header, body=body, footer=self._format_footer(state)
        )

    def render_final_parts(
        self,
        state: ProgressState,
        *,
        elapsed_s: float,
        status: str,
        answer: str,
    ) -> MarkdownParts:
        step = state.action_count or None
        emoji, status_label = _status_parts(status)
        scope = _extract_scope(state.context_line) or state.engine
        header = format_header(
            elapsed_s,
            step,
            emoji=emoji,
            status=status_label,
            scope=scope,
        )
        answer = (answer or "").strip()
        body = answer if answer else None
        return MarkdownParts(
            header=header, body=body, footer=self._format_footer(state)
        )

    def _format_footer(self, state: ProgressState) -> str | None:
        lines: list[str] = []
        if state.context_line:
            lines.append(state.context_line)
        if state.resume_line:
            lines.append(state.resume_line)
        if not lines:
            return None
        return HARD_BREAK.join(lines)

    def _format_actions(self, state: ProgressState) -> list[str]:
        actions = list(state.actions)
        actions = [] if self.max_actions == 0 else actions[-self.max_actions :]
        return [
            format_action_line(
                action_state.action,
                action_state.display_phase,
                action_state.ok,
                command_width=self.command_width,
            )
            for action_state in actions
        ]

    @staticmethod
    def _assemble_body(lines: list[str]) -> str | None:
        if not lines:
            return None
        return HARD_BREAK.join(lines)


class MarkdownPresenter:
    def __init__(self, *, formatter: MarkdownFormatter | None = None) -> None:
        self._formatter = formatter or MarkdownFormatter()

    def render_progress(
        self,
        state: ProgressState,
        *,
        elapsed_s: float,
        label: str = "working",
    ) -> RenderedMessage:
        parts = self._formatter.render_progress_parts(
            state, elapsed_s=elapsed_s, label=label
        )
        return RenderedMessage(text=assemble_markdown_parts(parts))

    def render_final(
        self,
        state: ProgressState,
        *,
        elapsed_s: float,
        status: str,
        answer: str,
    ) -> RenderedMessage:
        parts = self._formatter.render_final_parts(
            state, elapsed_s=elapsed_s, status=status, answer=answer
        )
        return RenderedMessage(text=assemble_markdown_parts(parts))
