"""Inbound backlog of the WebSocket worker: overflow drops the oldest audio and is counted (session.closed fields)."""
from types import SimpleNamespace

from mynah_v2.transport.ws_server import SessionWorker


def test_backlog_overflow_drops_oldest_and_counts():
    notices = []
    sess = SimpleNamespace(on_audio=None)
    w = SessionWorker(sess, lambda b: None, max_backlog_bytes=4, on_drop=notices.append)
    for _ in range(5):
        w.push(b"\x00\x01")                       # 2 bytes each, cap 4 bytes -> 3 oldest packets dropped
    assert notices == [2, 2, 2]
    assert (w.inbound_dropped_bytes, w.inbound_dropped_events) == (6, 3)
    assert w.backlog == 4 and w.q.qsize() == 2


def test_no_overflow_no_count():
    w = SessionWorker(SimpleNamespace(on_audio=None), lambda b: None, max_backlog_bytes=100, on_drop=lambda n: None)
    w.push(b"\x00" * 10)
    assert (w.inbound_dropped_bytes, w.inbound_dropped_events) == (0, 0)
