from dataclasses import replace

import pytest

from takopi.telegram.loop import _send_startup
from takopi.telegram.render import MAX_BODY_CHARS
from takopi.transport import RenderedMessage

from .telegram_fakes import FakeTransport, make_cfg


@pytest.mark.anyio
async def test_send_startup_single_message_no_followups() -> None:
    transport = FakeTransport()
    cfg = replace(make_cfg(transport), startup_msg="ready")

    await _send_startup(cfg)

    assert len(transport.send_calls) == 1
    message = transport.send_calls[0]["message"]
    assert "followups" not in message.extra
    assert "ready" in message.text


@pytest.mark.anyio
async def test_send_startup_splits_long_message_with_continued_header() -> None:
    transport = FakeTransport()
    long_body = "x" * (MAX_BODY_CHARS + 25)
    startup_msg = "**takopi is ready**\n\n" + long_body
    cfg = replace(make_cfg(transport), startup_msg=startup_msg)

    await _send_startup(cfg)

    assert len(transport.send_calls) == 1
    message = transport.send_calls[0]["message"]
    followups = message.extra.get("followups")
    assert followups
    assert all(isinstance(item, RenderedMessage) for item in followups)
    assert "takopi is ready" in message.text
    assert any("continued" in item.text for item in followups)
