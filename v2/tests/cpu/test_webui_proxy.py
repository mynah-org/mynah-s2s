"""Application bridge lifecycle: browser termination must close the upstream V2 session."""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "webui"))
from server import proxy_session  # noqa: E402


class _Browser:
    def __init__(self, messages=(), fail=False):
        self.messages = list(messages)
        self.fail = fail
        self.sent = []

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.messages:
            return self.messages.pop(0)
        if self.fail:
            raise ConnectionError("browser disconnected")
        raise StopAsyncIteration

    async def send(self, message):
        self.sent.append(message)


class _Upstream:
    def __init__(self, browser_disconnect=False):
        self.sent = []
        self.browser_disconnect = browser_disconnect
        self.index = 0

    async def send(self, message):
        self.sent.append(message)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.index == 0:
            self.index += 1
            return json.dumps({"type": "session.created", "backend_profile": "int4"})
        if self.index == 1:
            if self.browser_disconnect:
                for _ in range(100):
                    if any(json.loads(m).get("type") == "session.cancel" for m in self.sent):
                        break
                    await asyncio.sleep(0)
            self.index += 1
            return json.dumps({"type": "session.closed", "kind": "cancel"})
        raise StopAsyncIteration

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return False


def _run(browser, upstream):
    import types

    old = sys.modules.get("websockets")
    sys.modules["websockets"] = types.SimpleNamespace(connect=lambda *_a, **_k: upstream)
    try:
        asyncio.run(proxy_session(browser, json.dumps({"type": "session.start"}), "ws://v2", "int4"))
    finally:
        if old is None:
            sys.modules.pop("websockets", None)
        else:
            sys.modules["websockets"] = old


def test_proxy_waits_for_upstream_closed_after_normal_browser_end():
    browser = _Browser([json.dumps({"type": "session.end"})])
    upstream = _Upstream()
    _run(browser, upstream)
    sent = [json.loads(m).get("type") for m in upstream.sent]
    received = [json.loads(m).get("type") for m in browser.sent]
    assert sent == ["session.start", "session.end"]
    assert received == ["session.created", "session.closed"]


def test_proxy_cancels_upstream_when_browser_disconnects():
    browser = _Browser(fail=True)
    upstream = _Upstream(browser_disconnect=True)
    _run(browser, upstream)
    sent = [json.loads(m).get("type") for m in upstream.sent]
    assert sent == ["session.start", "session.cancel"]
