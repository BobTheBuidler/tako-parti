from collections.abc import AsyncIterator
from pathlib import Path
import re
from typing import cast

import anyio
import pytest

from takopi.config import ProjectConfig, ProjectsConfig
from takopi.context import RunContext
from takopi.markdown import MarkdownPresenter
from takopi.model import CompletedEvent, ResumeToken, StartedEvent
from takopi.router import AutoRouter, RunnerEntry
from takopi.runner_bridge import ExecBridgeConfig
from takopi.runners.mock import Return, ScriptRunner
from takopi.settings import TelegramTopicsSettings
from takopi.telegram.bridge import TelegramBridgeConfig, run_main_loop
from takopi.telegram.topic_state import TopicStateStore, resolve_state_path
from takopi.telegram.types import TelegramIncomingMessage
from takopi.transport import MessageRef
from takopi.transport_runtime import TransportRuntime
from tests.telegram_fakes import FakeBot, FakeTransport

CODEX_ENGINE = "codex"
FAST_FORWARD_COALESCE_S = 0.0
FAST_MEDIA_GROUP_DEBOUNCE_S = 0.0


class _TopicMultiSessionRunner:
    engine = CODEX_ENGINE
    title = "Codex"
    resume_re = re.compile(
        rf"(?im)^\s*`?{re.escape(CODEX_ENGINE)}\s+resume\s+(?P<token>[^`\s]+)`?\s*$"
    )

    def __init__(
        self,
        *,
        start_tokens: dict[str, str],
        hold_by_token: dict[str, anyio.Event],
        started_by_token: dict[str, anyio.Event],
        finished_by_token: dict[str, anyio.Event],
    ) -> None:
        self._start_tokens = start_tokens
        self._hold_by_token = hold_by_token
        self._started_by_token = started_by_token
        self._finished_by_token = finished_by_token
        self.calls: list[tuple[str, ResumeToken | None]] = []

    def format_resume(self, token: ResumeToken) -> str:
        return f"`{self.engine} resume {token.value}`"

    def is_resume_line(self, line: str) -> bool:
        return bool(self.resume_re.match(line))

    def extract_resume(self, text: str | None) -> ResumeToken | None:
        if not text:
            return None
        found: str | None = None
        for match in self.resume_re.finditer(text):
            token = match.group("token")
            if token:
                found = token
        if found is None:
            return None
        return ResumeToken(engine=self.engine, value=found)

    async def run(
        self,
        prompt: str,
        resume: ResumeToken | None,
    ) -> AsyncIterator[StartedEvent | CompletedEvent]:
        self.calls.append((prompt, resume))
        if resume is None:
            token_value = self._start_tokens[prompt]
            token = ResumeToken(engine=self.engine, value=token_value)
            self._started_by_token[token_value].set()
            yield StartedEvent(engine=self.engine, resume=token, title=self.title)
            await anyio.sleep(0)
            await self._hold_by_token[token_value].wait()
            self._finished_by_token[token_value].set()
            yield CompletedEvent(
                engine=self.engine,
                ok=True,
                answer=f"done-{prompt}",
                resume=token,
            )
            return

        yield StartedEvent(engine=self.engine, resume=resume, title=self.title)
        await anyio.sleep(0)
        yield CompletedEvent(
            engine=self.engine,
            ok=True,
            answer=f"done-{prompt}",
            resume=resume,
        )


async def _wait_for_progress_ref(
    transport: FakeTransport,
    *,
    reply_to_message_id: int,
) -> MessageRef:
    with anyio.fail_after(2):
        while True:
            for call in transport.send_calls:
                options = call["options"]
                if (
                    options is not None
                    and options.notify is False
                    and options.reply_to is not None
                    and options.reply_to.message_id == reply_to_message_id
                ):
                    return cast(MessageRef, call["ref"])
            await anyio.sleep(0)


@pytest.mark.anyio
@pytest.mark.parametrize(
    "reply_order",
    [("A", "B"), ("B", "A")],
    ids=["reply-A-then-B", "reply-B-then-A"],
)
@pytest.mark.parametrize(
    "completion_order",
    [("A", "B"), ("B", "A")],
    ids=["complete-A-first", "complete-B-first"],
)
@pytest.mark.parametrize(
    "fallback_present",
    [False, True],
    ids=["fallback-absent", "fallback-present"],
)
@pytest.mark.parametrize(
    "duplicate_update",
    [False, True],
    ids=["no-duplicate", "duplicate-update"],
)
async def test_run_main_loop_same_topic_multi_session_interleaving_matrix(
    tmp_path: Path,
    reply_order: tuple[str, str],
    completion_order: tuple[str, str],
    fallback_present: bool,
    duplicate_update: bool,
) -> None:
    """Contract: replies route by progress ref; ordering across sessions follows completion, not reply arrival."""

    token_for_session = {"A": "tok-a", "B": "tok-b"}
    hold_by_token = {token: anyio.Event() for token in ("tok-a", "tok-b")}
    started_by_token = {token: anyio.Event() for token in ("tok-a", "tok-b")}
    finished_by_token = {token: anyio.Event() for token in ("tok-a", "tok-b")}
    codex_runner = _TopicMultiSessionRunner(
        start_tokens={
            "seed-a": "tok-a",
            "seed-b": "tok-b",
        },
        hold_by_token=hold_by_token,
        started_by_token=started_by_token,
        finished_by_token=finished_by_token,
    )
    claude_runner = ScriptRunner([Return(answer="done-claude")], engine="claude")

    transport = FakeTransport()
    bot = FakeBot()
    projects = ProjectsConfig(
        projects={
            "proj": ProjectConfig(
                alias="proj",
                path=tmp_path,
                worktrees_dir=Path(".worktrees"),
                chat_id=123,
            )
        },
        default_project=None,
    )
    runtime = TransportRuntime(
        router=AutoRouter(
            entries=[
                RunnerEntry(engine=codex_runner.engine, runner=codex_runner),
                RunnerEntry(engine=claude_runner.engine, runner=claude_runner),
            ],
            default_engine=codex_runner.engine,
        ),
        projects=projects,
        config_path=tmp_path / "takopi.toml",
    )
    cfg = TelegramBridgeConfig(
        bot=bot,
        runtime=runtime,
        chat_id=123,
        startup_msg="",
        exec_cfg=ExecBridgeConfig(
            transport=transport,
            presenter=MarkdownPresenter(),
            final_notify=True,
        ),
        forward_coalesce_s=FAST_FORWARD_COALESCE_S,
        media_group_debounce_s=FAST_MEDIA_GROUP_DEBOUNCE_S,
        topics=TelegramTopicsSettings(enabled=True, scope="main"),
    )

    state_store = TopicStateStore(resolve_state_path(cast(Path, runtime.config_path)))
    await state_store.set_context(123, 77, RunContext(project="proj"))

    prompt_for_session = {"A": "follow-a", "B": "follow-b"}
    duplicate_prompt = f"duplicate-{reply_order[0].lower()}"

    async def poller(_cfg: TelegramBridgeConfig):
        next_message_id = 3
        next_update_id = 9000
        yield TelegramIncomingMessage(
            transport="telegram",
            chat_id=123,
            message_id=1,
            text="/proj seed-a",
            reply_to_message_id=None,
            reply_to_text=None,
            sender_id=123,
            thread_id=77,
            chat_type="supergroup",
            update_id=next_update_id,
        )
        next_update_id += 1
        yield TelegramIncomingMessage(
            transport="telegram",
            chat_id=123,
            message_id=2,
            text="seed-b",
            reply_to_message_id=None,
            reply_to_text=None,
            sender_id=123,
            thread_id=77,
            chat_type="supergroup",
            update_id=next_update_id,
        )
        next_update_id += 1

        with anyio.fail_after(2):
            await started_by_token["tok-a"].wait()
            await started_by_token["tok-b"].wait()

        progress_ref_a = await _wait_for_progress_ref(transport, reply_to_message_id=1)
        progress_ref_b = await _wait_for_progress_ref(transport, reply_to_message_id=2)
        progress_msg_id: dict[str, int] = {
            "A": int(progress_ref_a.message_id),
            "B": int(progress_ref_b.message_id),
        }

        for idx, session in enumerate(reply_order):
            update_id = next_update_id
            next_update_id += 1
            yield TelegramIncomingMessage(
                transport="telegram",
                chat_id=123,
                message_id=next_message_id + idx,
                text=prompt_for_session[session],
                reply_to_message_id=progress_msg_id[session],
                reply_to_text=None,
                sender_id=123,
                thread_id=77,
                chat_type="supergroup",
                update_id=update_id,
            )
            if duplicate_update and idx == 0:
                yield TelegramIncomingMessage(
                    transport="telegram",
                    chat_id=123,
                    message_id=next_message_id + idx + 100,
                    text=duplicate_prompt,
                    reply_to_message_id=progress_msg_id[session],
                    reply_to_text=None,
                    sender_id=123,
                    thread_id=77,
                    chat_type="supergroup",
                    update_id=update_id,
                )
        next_message_id += len(reply_order)

        yield TelegramIncomingMessage(
            transport="telegram",
            chat_id=123,
            message_id=next_message_id + 10,
            text="fallback-msg" if fallback_present else "/claude fallback-msg",
            reply_to_message_id=999,
            reply_to_text=None,
            sender_id=123,
            thread_id=77,
            chat_type="supergroup",
            update_id=next_update_id,
        )

        await anyio.sleep(0)
        for session in completion_order:
            token = token_for_session[session]
            hold_by_token[token].set()
            with anyio.fail_after(2):
                await finished_by_token[token].wait()

    with anyio.fail_after(4):
        await run_main_loop(cfg, poller)

    codex_calls_by_prompt = dict(codex_runner.calls)
    assert codex_calls_by_prompt["follow-a"] == ResumeToken(
        engine=CODEX_ENGINE, value="tok-a"
    )
    assert codex_calls_by_prompt["follow-b"] == ResumeToken(
        engine=CODEX_ENGINE, value="tok-b"
    )

    if fallback_present:
        assert codex_calls_by_prompt["fallback-msg"] == ResumeToken(
            engine=CODEX_ENGINE, value="tok-b"
        )
        assert "fallback-msg" not in {prompt for prompt, _ in claude_runner.calls}
    else:
        assert "fallback-msg" not in codex_calls_by_prompt
        assert ("fallback-msg", None) in claude_runner.calls

    assert duplicate_prompt not in codex_calls_by_prompt
    assert duplicate_prompt not in {prompt for prompt, _ in claude_runner.calls}

    follow_order = [
        prompt for prompt, _ in codex_runner.calls if prompt in {"follow-a", "follow-b"}
    ]
    expected_follow_order = [
        prompt_for_session[session] for session in completion_order
    ]
    assert follow_order == expected_follow_order
