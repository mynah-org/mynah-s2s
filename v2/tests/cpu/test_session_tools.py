"""Session ownership of tool handlers and of the FC background thread (concurrency and cancellation)."""

import threading
import time

import pytest

from mynah_v2.model.session_tools import SessionToolBinding, release_fc_thread


def test_bind_unbind_restores_base_and_refuses_stale_calls():
    reg = {"builtin": lambda a: "b"}
    b = SessionToolBinding(reg)
    calls = []
    b.bind({"get_order_status": lambda a: calls.append(a) or "ok", "builtin": lambda a: "override"})
    h = reg["get_order_status"]
    assert h({"order_id": "X"}) == "ok" and reg["builtin"]({}) == "override"
    b.unbind()
    assert set(reg) == {"builtin"} and reg["builtin"]({}) == "b"        # base restored
    assert h({"order_id": "late"}) is None                              # handler captured before close: refused
    assert calls == [{"order_id": "X"}] and b.stale_calls[0]["tool"] == "get_order_status"


def test_second_session_cannot_overwrite_a_bound_one():
    b = SessionToolBinding({})
    b.bind({"t": lambda a: "s1"})
    with pytest.raises(RuntimeError):
        b.bind({"t": lambda a: "s2"})
    assert b.registry["t"]({}) == "s1"


def test_concurrent_binds_exactly_one_wins():
    b = SessionToolBinding({})
    won, lost = [], []
    start = threading.Barrier(8)

    def go(i):
        start.wait()
        try:
            b.bind({"t": lambda a, i=i: i})
            won.append(i)
        except RuntimeError:
            lost.append(i)
    ts = [threading.Thread(target=go, args=(i,)) for i in range(8)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert len(won) == 1 and len(lost) == 7 and b.registry["t"]({}) == won[0]


def test_old_epoch_handler_refused_after_rebind():
    b = SessionToolBinding({})
    b.bind({"t": lambda a: "s1"})
    old = b.registry["t"]
    b.unbind()
    b.bind({"t": lambda a: "s2"})
    assert old({}) is None and b.registry["t"]({}) == "s2"


def _bg(stop_after_abort: bool, extra_s: float = 0.0):
    abort = threading.Event()

    def run():
        abort.wait()
        if not stop_after_abort:
            time.sleep(extra_s)
    th = threading.Thread(target=run, daemon=True)
    th.start()
    return {"thread": th, "abort_event": abort, "quit_async_event": threading.Event()}


def test_release_fc_thread_drops_handle_only_when_thread_exited():
    m = {0: _bg(True)}
    assert release_fc_thread(m, 0, 2.0) == "released" and 0 not in m
    assert release_fc_thread(m, 0, 2.0) == "none"
    m = {0: _bg(False, extra_s=0.5)}                                    # ignores abort for 0.5 s
    assert release_fc_thread(m, 0, 0.05) == "running" and 0 in m        # bounded join, handle KEPT
    assert release_fc_thread(m, 0, 2.0) == "released" and 0 not in m    # next open waits and reuses
