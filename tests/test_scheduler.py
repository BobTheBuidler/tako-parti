import anyio
import pytest

from takopi.model import ResumeToken
from takopi.scheduler import ThreadJob, ThreadScheduler
from takopi.transport import MessageRef


class _TaskGroupAdapter:
    def __init__(self, task_group) -> None:
        self._task_group = task_group

    def start_soon(self, func, *args) -> None:
        self._task_group.start_soon(func, *args)


def _job(
    *,
    user_msg_id: int,
    resume_token: ResumeToken,
    progress_id: int,
) -> ThreadJob:
    return ThreadJob(
        chat_id=123,
        user_msg_id=user_msg_id,
        text=f"job-{user_msg_id}",
        resume_token=resume_token,
        progress_ref=MessageRef(channel_id=123, message_id=progress_id),
    )


@pytest.mark.anyio
async def test_scheduler_preserves_fifo_per_resume_key() -> None:
    token = ResumeToken(engine="codex", value="session-a")
    first_started = anyio.Event()
    second_started = anyio.Event()
    release_first = anyio.Event()
    all_done = anyio.Event()
    started: list[int | str] = []
    finished: list[int | str] = []

    async def run_job(job: ThreadJob) -> None:
        started.append(job.user_msg_id)
        if job.user_msg_id == 1:
            first_started.set()
            await release_first.wait()
        if job.user_msg_id == 2:
            second_started.set()
        finished.append(job.user_msg_id)
        if len(finished) == 2:
            all_done.set()

    async with anyio.create_task_group() as tg:
        scheduler = ThreadScheduler(task_group=_TaskGroupAdapter(tg), run_job=run_job)
        await scheduler.enqueue(_job(user_msg_id=1, resume_token=token, progress_id=11))
        await scheduler.enqueue(_job(user_msg_id=2, resume_token=token, progress_id=12))

        with anyio.fail_after(1):
            await first_started.wait()
        assert second_started.is_set() is False

        release_first.set()
        with anyio.fail_after(1):
            await second_started.wait()
            await all_done.wait()

    assert started == [1, 2]
    assert finished == [1, 2]


@pytest.mark.anyio
async def test_scheduler_allows_different_keys_to_progress_independently() -> None:
    token_a = ResumeToken(engine="codex", value="session-a")
    token_b = ResumeToken(engine="codex", value="session-b")
    release_a = anyio.Event()
    started_a = anyio.Event()
    started_b = anyio.Event()
    finished_b = anyio.Event()
    all_done = anyio.Event()
    order: list[str] = []

    async def run_job(job: ThreadJob) -> None:
        if job.resume_token == token_a:
            started_a.set()
            await release_a.wait()
            order.append("A")
        else:
            started_b.set()
            order.append("B")
            finished_b.set()
        if len(order) == 2:
            all_done.set()

    async with anyio.create_task_group() as tg:
        scheduler = ThreadScheduler(task_group=_TaskGroupAdapter(tg), run_job=run_job)
        await scheduler.enqueue(
            _job(user_msg_id=1, resume_token=token_a, progress_id=21),
        )
        await scheduler.enqueue(
            _job(user_msg_id=2, resume_token=token_b, progress_id=22),
        )

        with anyio.fail_after(1):
            await started_a.wait()
            await started_b.wait()
            await finished_b.wait()

        assert order == ["B"]

        release_a.set()
        with anyio.fail_after(1):
            await all_done.wait()

    assert order == ["B", "A"]


@pytest.mark.anyio
async def test_scheduler_busy_gate_waits_for_original_done_event() -> None:
    token = ResumeToken(engine="codex", value="session-a")
    done_first = anyio.Event()
    done_second = anyio.Event()
    started = anyio.Event()
    completed = anyio.Event()

    async def run_job(_: ThreadJob) -> None:
        started.set()
        completed.set()

    async with anyio.create_task_group() as tg:
        scheduler = ThreadScheduler(task_group=_TaskGroupAdapter(tg), run_job=run_job)
        await scheduler.note_thread_known(token, done_first)
        await scheduler.note_thread_known(token, done_second)
        await scheduler.enqueue(_job(user_msg_id=1, resume_token=token, progress_id=31))

        await anyio.sleep(0)
        assert started.is_set() is False

        done_second.set()
        await anyio.sleep(0)
        assert started.is_set() is False

        done_first.set()
        with anyio.fail_after(1):
            await completed.wait()


@pytest.mark.anyio
async def test_scheduler_cancel_queued_removes_only_targeted_job() -> None:
    token = ResumeToken(engine="codex", value="session-a")
    release_first = anyio.Event()
    first_started = anyio.Event()
    all_done = anyio.Event()
    executed: list[int | str] = []

    async def run_job(job: ThreadJob) -> None:
        if job.user_msg_id == 1:
            first_started.set()
            await release_first.wait()
        executed.append(job.user_msg_id)
        if len(executed) == 2:
            all_done.set()

    async with anyio.create_task_group() as tg:
        scheduler = ThreadScheduler(task_group=_TaskGroupAdapter(tg), run_job=run_job)
        await scheduler.enqueue(_job(user_msg_id=1, resume_token=token, progress_id=41))
        await scheduler.enqueue(_job(user_msg_id=2, resume_token=token, progress_id=42))
        await scheduler.enqueue(_job(user_msg_id=3, resume_token=token, progress_id=43))

        with anyio.fail_after(1):
            await first_started.wait()

        removed = await scheduler.cancel_queued(123, 42)
        assert removed is not None
        assert removed.user_msg_id == 2
        assert await scheduler.cancel_queued(123, 41) is None

        release_first.set()
        with anyio.fail_after(1):
            await all_done.wait()

    assert executed == [1, 3]
