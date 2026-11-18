"""Session -> language slot orchestration, independent of torch (the tensor work is done by a Backend).

Tiers per language, fastest first:
  vram    a slot of preallocated weight buffers (SlotPool; fixed addresses while allocated)
  pinned  the language package in page-locked host memory (fast, async host-to-device copies)
  ram     the package in ordinary (pageable) host memory
  disk    the package files (optionally zstd-compressed); always available
Every host tier is optional and has its own caps (bytes / items / idle seconds). An entry evicted from a host
tier is demoted to the next one if it fits there; a hit in a lower tier is promoted to the first one.
Binding a session to a language ref-counts its slot; a slot with live sessions is never evicted or rewritten.
Switching one session's language moves only that session; other sessions keep their slots, weights and states.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Hashable, Protocol

from .lru import AllSlotsBusy, CacheFull, RefLRU, Slot, SlotPool


class Backend(Protocol):
    def read_package(self, lang: str) -> tuple[Any, int]:
        """Read a language from disk into host memory; returns (package, nbytes)."""

    def to_tier(self, package: Any, tier: str) -> tuple[Any, int]:
        """Convert a host package for `tier` (e.g. pin / unpin its tensors); returns (package, nbytes)."""

    def allocate_slot(self, index: int) -> None: ...

    def free_slot(self, index: int) -> None: ...

    def materialise(self, index: int, lang: str, package: Any) -> None:
        """Write the language's weights into slot `index` in place (same addresses)."""

    def voice_state(self, index: int, lang: str, voice: Any) -> Any:
        """Prompt state for `voice` computed with the weights now in slot `index` (language `lang`)."""


@dataclass
class TierSpec:
    name: str                       # "pinned" or "ram" for the torch backend
    max_bytes: int | None = None
    max_items: int | None = None
    idle_s: float | None = None


@dataclass
class Binding:
    session: Hashable
    lang: str
    slot: int
    event: str                 # hot / reuse / alloc / evict
    source: str                # vram / <host tier name> / disk
    load_ms: float = 0.0
    voice_key: Hashable | None = None
    voice_state: Any = None
    alloc_gen: int = 0
    load_gen: int = 0


@dataclass
class CoreStats:
    binds: int = 0
    hot: int = 0
    sources: dict = field(default_factory=dict)
    slot_evictions: int = 0
    idle_frees: int = 0
    load_ms: list = field(default_factory=list)


class SwapCore:
    def __init__(self, backend: Backend, n_slots: int = 4, tiers: list[TierSpec] | None = None,
                 slot_idle_s: float | None = None, voice_cache_items: int = 64, clock=time.monotonic):
        self.backend = backend
        self.clock = clock
        self.pool = SlotPool(n_slots, idle_s=slot_idle_s, clock=clock)
        specs = [TierSpec("pinned")] if tiers is None else tiers
        self.tiers: list[tuple[str, RefLRU]] = []
        for i, t in enumerate(specs):
            self.tiers.append((t.name, RefLRU(max_items=t.max_items, max_bytes=t.max_bytes, idle_s=t.idle_s,
                                              clock=clock, on_evict=self._demoter(i))))
        self.voices = RefLRU(max_items=voice_cache_items, clock=clock)
        self.sessions: dict[Hashable, Binding] = {}
        self.stats = CoreStats()
        self._cv = threading.Condition(threading.RLock())
        self._promoting = False

    def tier(self, name: str) -> RefLRU:
        return dict(self.tiers)[name]

    # -- host tiers --------------------------------------------------------------------------------------
    def _demoter(self, i: int):
        def on_evict(lang, pkg):
            if self._promoting or i + 1 >= len(self.tiers):
                return  # promotion moves the entry up; the last tier drops to disk
            name, nxt = self.tiers[i + 1]
            p2, nb = self.backend.to_tier(pkg, name)
            try:
                nxt.put(lang, p2, nb)
            except CacheFull:
                pass
        return on_evict

    def _host_get(self, lang: str) -> tuple[Any, str] | None:
        for i, (name, lru) in enumerate(self.tiers):
            pkg = lru.get(lang)
            if pkg is None:
                continue
            if i > 0:  # promote to the first tier
                self._promoting = True
                try:
                    lru.pop(lang)
                finally:
                    self._promoting = False
                self._host_put(lang, pkg)
            return pkg, name
        return None

    def _host_put(self, lang: str, pkg: Any) -> Any:
        for name, lru in self.tiers:
            p2, nb = self.backend.to_tier(pkg, name)
            try:
                lru.put(lang, p2, nb)
                return p2
            except CacheFull:
                continue
        return pkg

    def _host_lru_of(self, lang: str) -> RefLRU | None:
        for _, lru in self.tiers:
            if lang in lru:
                return lru
        return None

    # -- binding -------------------------------------------------------------------------------------
    def bind(self, session: Hashable, lang: str, voice: Any = None, voice_key: Hashable | None = None,
             timeout: float | None = 0.0) -> Binding:
        """Bind (or re-bind) `session` to `lang`. Waits up to `timeout` s for a free slot (None = forever)."""
        deadline = None if timeout is None else self.clock() + timeout
        with self._cv:
            old = self.sessions.get(session)
            if old is not None and old.lang == lang and (voice_key is None or voice_key == old.voice_key):
                return old
            while True:
                try:
                    slot, event = self.pool.choose(lang)
                    break
                except AllSlotsBusy:
                    # the session's own old slot counts as free for itself
                    if old is not None and self.pool.slots[old.slot].refs == 1:
                        self._drop(session)
                        old = None
                        continue
                    remaining = None if deadline is None else deadline - self.clock()
                    if remaining is not None and remaining <= 0:
                        raise
                    self._cv.wait(remaining)
            source, ms = self._load(slot, lang, event)
            if old is not None:
                self._drop(session)
            self.pool.ref(slot, session)
            b = Binding(session, lang, slot.index, event, source, ms, alloc_gen=slot.alloc_gen, load_gen=slot.load_gen)
            if voice is not None or voice_key is not None:
                b.voice_key = voice_key if voice_key is not None else id(voice)
                b.voice_state = self._voice(slot, lang, voice, b.voice_key)
            self.sessions[session] = b
            self.stats.binds += 1
            return b

    def _load(self, slot: Slot, lang: str, event: str) -> tuple[str, float]:
        if event == "hot":
            self.stats.hot += 1
            self.pool.commit(slot, lang, event)
            return "vram", 0.0
        t0 = time.perf_counter()
        if event == "evict":
            self.stats.slot_evictions += 1
            self._forget_voices(slot.lang)
        hit = self._host_get(lang)
        if hit is None:
            pkg, _ = self.backend.read_package(lang)
            source = "disk"
            pkg = self._host_put(lang, pkg)
        else:
            pkg, source = hit
        if event == "alloc":
            self.backend.allocate_slot(slot.index)
        lru = self._host_lru_of(lang)
        if lru is not None:
            lru.acquire(lang)  # not evictable while it is being copied into the slot
        try:
            self.backend.materialise(slot.index, lang, pkg)
        except BaseException:
            slot.lang = None  # weights may be half-written: the slot holds no valid language any more
            raise
        finally:
            if lru is not None:
                lru.release(lang)
        self.pool.commit(slot, lang, event)
        ms = (time.perf_counter() - t0) * 1e3
        self.stats.load_ms.append(ms)
        self.stats.sources[source] = self.stats.sources.get(source, 0) + 1
        return source, ms

    def _voice(self, slot: Slot, lang: str, voice: Any, key: Hashable) -> Any:
        k = (lang, key)
        st = self.voices.get(k)
        if st is None:
            st = self.backend.voice_state(slot.index, lang, voice)
            try:
                self.voices.put(k, st, 1)
            except CacheFull:
                pass
        return st

    def _forget_voices(self, lang: str | None) -> None:
        # voice states depend only on (language weights, voice), not on the slot: they stay valid while the
        # language exists anywhere; we drop them when the language leaves VRAM to bound memory.
        if lang is None:
            return
        for k in [k for k in self.voices.keys() if k[0] == lang]:
            if self.voices.refs(k) == 0:
                self.voices.pop(k)

    def _drop(self, session: Hashable) -> None:
        b = self.sessions.pop(session)
        self.pool.unref(self.pool.slots[b.slot], session)
        self._cv.notify_all()

    def unbind(self, session: Hashable) -> None:
        with self._cv:
            if session in self.sessions:
                self._drop(session)

    # -- queries ---------------------------------------------------------------------------------------
    def binding(self, session: Hashable) -> Binding:
        return self.sessions[session]

    def group_by_slot(self, sessions) -> dict[int, list]:
        """Sessions that share a slot can be batched in one forward; different slots run separately."""
        out: dict[int, list] = {}
        with self._cv:
            for s in sessions:
                out.setdefault(self.sessions[s].slot, []).append(s)
        return out

    def resident(self) -> dict[int, str | None]:
        return {s.index: s.lang for s in self.pool.slots}

    def host_resident(self) -> dict[str, list]:
        return {name: lru.keys() for name, lru in self.tiers}

    # -- maintenance -----------------------------------------------------------------------------------
    def maintenance(self, now: float | None = None) -> dict:
        """Free idle unreferenced slots (VRAM) and idle host-tier entries. Call it from a timer."""
        with self._cv:
            freed = []
            for s in self.pool.idle_slots(now):
                self._forget_voices(s.lang)
                self.backend.free_slot(s.index)
                self.pool.free(s)
                freed.append(s.index)
                self.stats.idle_frees += 1
            host = {name: lru.evict_idle(now) for name, lru in self.tiers}
            return {"slots_freed": freed, "host_evicted": host}
