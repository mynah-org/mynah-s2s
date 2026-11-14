"""Session ownership for the two pieces of engine state that outlive a step: tool handlers and the FC background thread.

The NVIDIA pipeline keeps ONE tool registry (name -> handler) and ONE ``_fc_async_bg[stream_id]`` entry. Both are
process objects; a session must own them for exactly its lifetime:
  * ``SessionToolBinding`` — handlers bound by a session are wrapped with the session epoch; ``unbind`` (at close)
    restores the base registry and bumps the epoch, so a call that arrives later (an FC thread still finishing, a late
    executor) is refused and recorded instead of running against the next session's ledger. Binding while another
    session is bound is an error (no silent overwrite; one engine = one session, enforced, not assumed).
  * ``release_fc_thread`` — signal abort, join with a bound; drop the handle ONLY if the thread has exited. A thread
    still alive keeps its handle and is reported, and the next ``open`` must wait for it (bounded) or refuse.
"""

from __future__ import annotations

import threading
import time
from typing import Callable


class SessionToolBinding:
    def __init__(self, registry: dict):
        self.registry = registry
        self.base = dict(registry)
        self.epoch = 0
        self.active = False
        self.stale_calls: list[dict] = []
        self.names: list[str] = []              # the tools the current session bound (wrapped by a call guard, if any)
        self._lock = threading.Lock()

    def bind(self, tools: dict[str, Callable]) -> int:
        with self._lock:
            if self.active:
                raise RuntimeError("session tools already bound: unbind (close the session) first")
            self.epoch += 1
            ep = self.epoch
            for name, fn in tools.items():
                self.registry[name] = self._guard(name, fn, ep)
            self.names = list(tools)
            self.active = True
            return ep

    def _guard(self, name, fn, ep):
        def handler(*a, **kw):
            if self.epoch != ep or not self.active:
                self.stale_calls.append({"tool": name, "epoch": ep, "current_epoch": self.epoch,
                                         "t_ns": time.monotonic_ns()})
                return None                     # reference semantics for "no handler": nothing is injected
            return fn(*a, **kw)
        handler.__wrapped__ = fn
        return handler

    def unbind(self) -> None:
        with self._lock:
            self.epoch += 1
            self.registry.clear()
            self.registry.update(self.base)
            self.names = []
            self.active = False


def release_fc_thread(bg_map: dict, key, join_timeout_s: float) -> str:
    """'none' (nothing there), 'released' (thread exited, handle dropped) or 'running' (still alive, handle kept)."""
    bg = bg_map.get(key)
    if bg is None:
        return "none"
    for ev_name in ("abort_event", "quit_async_event"):
        ev = bg.get(ev_name)
        if ev is not None:
            ev.set()
    th = bg.get("thread")
    if th is not None:
        th.join(timeout=join_timeout_s)
        if th.is_alive():
            return "running"
    bg_map.pop(key, None)
    return "released"
