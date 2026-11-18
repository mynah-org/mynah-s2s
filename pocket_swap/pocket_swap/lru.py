"""Pure-python cache bookkeeping (no torch): a ref-counted LRU and the fixed pool of language slots.

Nothing here touches tensors. The caller supplies the bytes of each entry and does the actual
allocation / copy work; these classes decide what is resident, what can be evicted and when.
An entry or slot with refs > 0 is never evicted.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Hashable, Iterable


class CacheFull(RuntimeError):
    """Nothing evictable is left and the new entry does not fit."""


class AllSlotsBusy(RuntimeError):
    """Every language slot is bound to a session that is still using it."""


@dataclass
class _Entry:
    value: Any
    nbytes: int
    refs: int = 0
    last_used: float = 0.0


class RefLRU:
    """LRU keyed cache with optional caps on item count and bytes, ref-count pinning and an idle timer.

    `put` evicts least-recently-used unpinned entries until the new one fits; if pinned entries alone
    exceed a cap it raises CacheFull and leaves the cache unchanged. `on_evict(key, value)` is called for
    every eviction (capacity, idle or explicit).
    """

    def __init__(self, max_items: int | None = None, max_bytes: int | None = None,
                 idle_s: float | None = None, clock: Callable[[], float] = time.monotonic,
                 on_evict: Callable[[Hashable, Any], None] | None = None):
        self.max_items, self.max_bytes, self.idle_s = max_items, max_bytes, idle_s
        self.clock, self.on_evict = clock, on_evict
        self._d: "OrderedDict[Hashable, _Entry]" = OrderedDict()
        self.hits = self.misses = self.evictions = 0

    def __contains__(self, key) -> bool:
        return key in self._d

    def __len__(self) -> int:
        return len(self._d)

    @property
    def nbytes(self) -> int:
        return sum(e.nbytes for e in self._d.values())

    def keys(self) -> list:
        """Keys from least to most recently used."""
        return list(self._d)

    def get(self, key, default=None):
        e = self._d.get(key)
        if e is None:
            self.misses += 1
            return default
        self.hits += 1
        e.last_used = self.clock()
        self._d.move_to_end(key)
        return e.value

    def refs(self, key) -> int:
        e = self._d.get(key)
        return 0 if e is None else e.refs

    def acquire(self, key) -> None:
        self._d[key].refs += 1
        self._d[key].last_used = self.clock()
        self._d.move_to_end(key)

    def release(self, key) -> None:
        e = self._d[key]
        if e.refs <= 0:
            raise RuntimeError(f"release without acquire: {key!r}")
        e.refs -= 1
        e.last_used = self.clock()

    def _fits(self, extra_items: int, extra_bytes: int, skip: Iterable = ()) -> bool:
        skip = set(skip)
        n = sum(1 for k in self._d if k not in skip) + extra_items
        b = sum(e.nbytes for k, e in self._d.items() if k not in skip) + extra_bytes
        return (self.max_items is None or n <= self.max_items) and (self.max_bytes is None or b <= self.max_bytes)

    def put(self, key, value, nbytes: int) -> list:
        """Insert (or replace) `key`; returns the evicted keys."""
        if key in self._d:
            self._evict(key, call=True)
        victims = []
        for k, e in self._d.items():  # LRU order
            if self._fits(1, nbytes, victims):
                break
            if e.refs == 0:
                victims.append(k)
        if not self._fits(1, nbytes, victims):
            raise CacheFull(f"{key!r} ({nbytes} B) does not fit: pinned entries fill the cache")
        for k in victims:
            self._evict(k, call=True)
        self._d[key] = _Entry(value, nbytes, 0, self.clock())
        return victims

    def pop(self, key) -> None:
        if self._d[key].refs:
            raise RuntimeError(f"{key!r} is in use")
        self._evict(key, call=True)

    def _evict(self, key, call: bool) -> None:
        e = self._d.pop(key)
        self.evictions += 1
        if call and self.on_evict is not None:
            self.on_evict(key, e.value)

    def evict_idle(self, now: float | None = None) -> list:
        if self.idle_s is None:
            return []
        now = self.clock() if now is None else now
        out = [k for k, e in self._d.items() if e.refs == 0 and now - e.last_used >= self.idle_s]
        for k in out:
            self._evict(k, call=True)
        return out


@dataclass
class Slot:
    index: int
    lang: str | None = None
    refs: int = 0
    last_used: float = 0.0
    allocated: bool = False
    alloc_gen: int = 0      # bumps every time the slot storage is (re)allocated: captured graphs must be redone
    load_gen: int = 0       # bumps every time new weights are written into the slot
    sessions: set = field(default_factory=set)


class SlotPool:
    """A fixed number of language slots. Binding a language returns a slot and what had to happen:

    "hot"    the language is already resident in a slot (shared by its sessions)
    "alloc"  a never-used / freed slot was allocated and must be loaded
    "reuse"  an allocated free slot (no language) must be loaded
    "evict"  the least recently used unreferenced slot was taken from another language and must be reloaded
    A slot with refs > 0 is never taken; if every slot is referenced, AllSlotsBusy is raised.
    """

    def __init__(self, n_slots: int, idle_s: float | None = None, clock: Callable[[], float] = time.monotonic):
        if n_slots < 1:
            raise ValueError("need at least one slot")
        self.slots = [Slot(i) for i in range(n_slots)]
        self.idle_s, self.clock = idle_s, clock

    def find(self, lang: str) -> Slot | None:
        for s in self.slots:
            if s.lang == lang:
                return s
        return None

    def choose(self, lang: str) -> tuple[Slot, str]:
        """Pick the slot for `lang` without changing anything (the caller loads, then calls `commit`)."""
        s = self.find(lang)
        if s is not None:
            return s, "hot"
        free = [s for s in self.slots if s.lang is None]
        alloc_free = [s for s in free if s.allocated]
        if alloc_free:
            return alloc_free[0], "reuse"
        if free:
            return free[0], "alloc"
        idle = [s for s in self.slots if s.refs == 0]
        if not idle:
            raise AllSlotsBusy(f"all {len(self.slots)} slots are in use")
        return min(idle, key=lambda s: s.last_used), "evict"

    def commit(self, slot: Slot, lang: str, event: str) -> None:
        if event == "alloc":
            slot.allocated = True
            slot.alloc_gen += 1
        if event != "hot":
            slot.lang = lang
            slot.load_gen += 1
        slot.last_used = self.clock()

    def ref(self, slot: Slot, session) -> None:
        slot.refs += 1
        slot.sessions.add(session)
        slot.last_used = self.clock()

    def unref(self, slot: Slot, session) -> None:
        if slot.refs <= 0 or session not in slot.sessions:
            raise RuntimeError(f"session {session!r} does not hold slot {slot.index}")
        slot.refs -= 1
        slot.sessions.discard(session)
        slot.last_used = self.clock()

    def idle_slots(self, now: float | None = None) -> list[Slot]:
        """Unreferenced, allocated slots idle for at least idle_s (candidates to free)."""
        if self.idle_s is None:
            return []
        now = self.clock() if now is None else now
        return [s for s in self.slots if s.allocated and s.refs == 0 and now - s.last_used >= self.idle_s]

    def free(self, slot: Slot) -> None:
        if slot.refs:
            raise RuntimeError(f"slot {slot.index} is in use")
        slot.lang = None
        slot.allocated = False
