import pytest

from takopi.telegram.topic_state import TopicStateStore


@pytest.mark.anyio
async def test_topic_state_paused_defaults_false(tmp_path) -> None:
    store = TopicStateStore(tmp_path / "topics.json")

    assert await store.get_paused(1, 2) is False


@pytest.mark.anyio
async def test_topic_state_paused_roundtrip(tmp_path) -> None:
    path = tmp_path / "topics.json"
    store = TopicStateStore(path)

    await store.set_paused(1, 2, True)
    assert await store.get_paused(1, 2) is True

    store2 = TopicStateStore(path)
    assert await store2.get_paused(1, 2) is True

    await store2.set_paused(1, 2, False)
    assert await store2.get_paused(1, 2) is False

    store3 = TopicStateStore(path)
    assert await store3.get_paused(1, 2) is False
