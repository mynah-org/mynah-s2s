"""Talker languages: which are resident on the GPU, and loading the others CONCURRENTLY (never on the scheduler).

Three tiers, each an LRU:
  VRAM   ``vram_langs`` ready models (soft cap: a language in use is never evicted; when every resident language is
         in use, a new one is loaded anyway and the cache shrinks back once they are free)
  host   ``host_langs`` packages kept in (pinned) host RAM, so a re-load skips the disk
  disk   the package directory (or a checkpoint)

The scheduler thread only calls ``want`` / ``get`` / ``poll``; they never block and never touch the disk.
A loader thread does ``fetch(name)`` (disk -> host) and ``build(name, host_obj)`` (host -> GPU, on its own CUDA
stream, plus the default voice state). ``poll`` publishes finished loads and evicts least-recently-used languages
that nothing uses. Every load is timed per tier; ``stats()`` reports them with the evictions.

Model-free (the loaders are injected), so the policy is unit-tested on fakes.
"""

from __future__ import annotations

import queue
import threading
import time
from collections import OrderedDict


class TalkerLangs:
    def __init__(self, fetch, build, vram_langs: int = 4, host_langs: int = 8, clock=time.monotonic,
                 on_event=None):
        """fetch(name) -> host object (package in RAM); build(name, host_obj) -> resident object (model + voices).
        on_event(dict): optional listener for load / evict events (tracing)."""
        self.fetch, self.build = fetch, build
        self.vram_langs, self.host_langs = max(1, vram_langs), max(0, host_langs)
        self.clock, self.on_event = clock, on_event
        self.resident: OrderedDict = OrderedDict()        # name -> built object (LRU order: oldest first)
        self.host: OrderedDict = OrderedDict()            # name -> host object (loader thread only)
        self.loading: dict = {}                           # name -> t_requested
        self.failed: dict = {}                            # name -> error text (last failure)
        self._q: queue.Queue = queue.Queue()
        self._done: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self.loads: list = []                             # per finished load: name, tier, fetch_ms, build_ms, wait_ms
        self.evictions = 0
        self.host_evictions = 0
        self._stop = False
        self._t = threading.Thread(target=self._run, name="f2-talker-loader", daemon=True)
        self._t.start()

    # ---------------------------------------------------------------- scheduler side (non-blocking)
    def get(self, name):
        """The resident object, or None (does not request a load)."""
        obj = self.resident.get(name)
        if obj is not None:
            self.resident.move_to_end(name)
        return obj

    def want(self, name) -> str:
        """'ready' | 'loading' (a load is queued or running) | 'failed: ...'."""
        if name in self.resident:
            self.resident.move_to_end(name)
            return "ready"
        if name in self.failed:
            return f"failed: {self.failed[name]}"
        if name not in self.loading:
            self.loading[name] = self.clock()
            self._q.put(name)
        return "loading"

    def poll(self, in_use=lambda name: False) -> list:
        """Publish finished loads and evict down to ``vram_langs``; -> the names that became ready."""
        ready = []
        while True:
            try:
                name, obj, info = self._done.get_nowait()
            except queue.Empty:
                break
            t_req = self.loading.pop(name, None)
            if obj is None:
                self.failed[name] = info.get("error", "error")
                self._emit({"ev": "load_failed", "name": name, **info})
                continue
            self.resident[name] = obj
            info["wait_ms"] = round((self.clock() - t_req) * 1000, 1) if t_req is not None else None
            self.loads.append(dict(info, name=name))
            self._emit({"ev": "loaded", "name": name, **info})
            ready.append(name)
        self._evict(in_use, keep=set(ready))
        return ready

    def _evict(self, in_use, keep=()):
        while len(self.resident) > self.vram_langs:
            victim = next((n for n in self.resident if n not in keep and not in_use(n)), None)
            if victim is None:
                return                                    # everything resident is in use: soft cap
            obj = self.resident.pop(victim)
            self.evictions += 1
            self._emit({"ev": "evicted", "name": victim})
            close = getattr(obj, "close", None)
            if close is not None:
                close()

    def ensure(self, name, timeout: float = 300.0, in_use=lambda name: False):
        """Blocking load (server start / preload only, never from a running session)."""
        t_end = self.clock() + timeout
        while True:
            st = self.want(name)
            if st == "ready":
                return self.resident[name]
            if st.startswith("failed"):
                raise RuntimeError(f"talker language {name}: {st}")
            if self.clock() > t_end:
                raise TimeoutError(f"talker language {name} not loaded in {timeout} s")
            self.poll(in_use)
            time.sleep(0.01)

    def stats(self) -> dict:
        def pct(xs, p):
            xs = sorted(x for x in xs if x is not None)
            return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else None
        by = {}
        for tier in ("host", "disk"):
            ls = [x for x in self.loads if x.get("tier") == tier]
            by[tier] = {"n": len(ls), "build_ms_p50": pct([x["build_ms"] for x in ls], .5),
                        "total_ms_p50": pct([x["fetch_ms"] + x["build_ms"] for x in ls], .5),
                        "total_ms_max": max([x["fetch_ms"] + x["build_ms"] for x in ls], default=None)}
        return {"resident": list(self.resident), "host": len(self.host), "loading": list(self.loading),
                "loads": len(self.loads), "evictions": self.evictions, "host_evictions": self.host_evictions,
                "failed": dict(self.failed), "by_tier": by,
                "wait_ms_p50": pct([x.get("wait_ms") for x in self.loads], .5),
                "wait_ms_max": max([x.get("wait_ms") or 0 for x in self.loads], default=None)}

    def close(self):
        self._stop = True
        self._q.put(None)

    # ---------------------------------------------------------------- loader thread
    def _emit(self, ev):
        if self.on_event is not None:
            try:
                self.on_event(ev)
            except Exception:  # noqa: BLE001
                pass

    def _run(self):
        while not self._stop:
            name = self._q.get()
            if name is None:
                return
            info = {"tier": "host"}
            try:
                t0 = self.clock()
                h = self.host.get(name)
                if h is None:
                    info["tier"] = "disk"
                    h = self.fetch(name)
                    if self.host_langs:
                        self.host[name] = h
                        while len(self.host) > self.host_langs:
                            self.host.popitem(last=False)
                            self.host_evictions += 1
                else:
                    self.host.move_to_end(name)
                t1 = self.clock()
                obj = self.build(name, h)
                t2 = self.clock()
                info.update(fetch_ms=round((t1 - t0) * 1000, 1), build_ms=round((t2 - t1) * 1000, 1))
                info.update(getattr(obj, "timing", None) or {})
                self._done.put((name, obj, info))
            except Exception as e:  # noqa: BLE001 - a broken package fails that language only
                info["error"] = f"{type(e).__name__}: {e}"
                self._done.put((name, None, info))
