"""EXPERIMENTAL multi-session scheduler: N independent conversations sharing one loaded model.

Status: scheduler logic only, CPU-tested with fakes. Nothing here makes the engine support B>1. It conforms to the
shared contract ``mynah_v2.runtime.batch_contract`` (integrator-owned): the Thinker and ALL its recurrent state are
behind ``ThinkerBatchBackend`` (real one = ``mynah_v2.model.thinker_batch``), the non-Thinker per-session state behind
``SessionStateStore`` (real one = ``mynah_v2.model.session_state``). The single-session path (``runtime.session``
+ ``ReferenceEngine``) and the WebSocket server are untouched.

Time model (docs/v2-batching-design.md §3-4)
  * every session delivers one input chunk per ``chunk_s`` (160 ms) of wall time; one chunk = ``frames_per_chunk``
    (2) Thinker frames of 80 ms;
  * a ROUND takes the head chunk of every ready session (arrival <= now): per session ``store.restore(sid)`` +
    perception; then for each frame index k one ``backend.step`` per compatible DECODE group plus B=1
    ``backend.step_one`` calls for the other phases (I7); then per session ``restore`` + post-processing
    (EarTTS/codec) and delivery;
  * a chunk must complete within ``deadline_s`` (default 0.160 s = ``FrameRequest`` deadline) of its arrival;
    misses are recorded with their lateness, never hidden;
  * sessions join/leave only between rounds (= chunk boundaries); a round in progress is never re-planned.
    ``join(sid, cache=...)`` attaches the session's B=1 hybrid cache to the backend; leaving detaches it and keeps
    the returned cache in ``detached[sid]`` so the session can rejoin later (I3 sequence).

Grouping (I5/I6): DECODE frames are grouped by bucket class ``bucket_of(positions()[sid] + 1)``; the scheduler reads
only ``ThinkerBatchBackend.positions()``, phases and host bookkeeping, never tensors. Groups larger than
``max_rows`` are split. A custom ``group_key(sid, phase, pos)`` can replace the bucket rule.

Determinism: the scheduler never sleeps or reads time itself; ``clock``/``sleep`` are injectable (tests use a fake
clock that the fake backend advances by a cost model). Ordering inside a round is by join order.
"""

from __future__ import annotations

import collections
import math
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Hashable, Iterable, Mapping, Protocol, Sequence

from mynah_v2.runtime.batch_contract import (FrameRequest, Phase, SessionStateStore,  # noqa: F401 - re-exported
                                             ThinkerBatchBackend, ThinkerOut)


class Route(str, Enum):
    BATCH = "batch"                 # ThinkerBatchBackend.step with the compatible group (step_one if alone)
    SOLO = "solo"                   # always step_one, never grouped (I7)
    SKIP = "skip"                   # no Thinker work for this frame (post-processing receives None)


# I7: only DECODE is batched; everything else is B=1 (or not stepped) and never blocks the others.
DEFAULT_ROUTES: dict[Phase, Route] = {Phase.DECODE: Route.BATCH, Phase.PREFILL: Route.SOLO,
                                      Phase.FC_FORCED: Route.SOLO, Phase.TOOL_PENDING: Route.SOLO,
                                      Phase.CLOSED: Route.SKIP}


def pow2_bucket(n: int, lo: int = 64, lmax: int | None = None) -> int:
    """Same rule as ``mynah_v2.model.eartts_graph.bucket_for`` (copied: that module imports torch)."""
    b = lo
    while b < n:
        b *= 2
    return b if lmax is None else min(b, lmax)


class SessionPipeline(Protocol):
    """Per-session non-Thinker work around the shared Thinker (runs after ``SessionStateStore.restore(sid)``)."""

    def perceive(self, chunk: Any, is_last: bool) -> Sequence[Any]: ...   # -> frames_per_chunk Thinker payloads
    def phase(self) -> Phase: ...                                          # queried before every frame
    def postprocess(self, thinker_outs: list[Any], is_last: bool) -> Any: ...  # EarTTS/codec -> chunk result


class NullStateStore:
    def capture(self, sid): return None
    def restore(self, sid): pass
    def drop(self, sid): pass


class PassthroughPipeline:
    """Default pipeline: the chunk already is the list of frame payloads; post returns the Thinker outputs."""

    def perceive(self, chunk, is_last):
        return chunk

    def phase(self):
        return Phase.DECODE

    def postprocess(self, thinker_outs, is_last):
        return list(thinker_outs)


# ------------------------------------------------------------------------------------------------ records
@dataclass
class ChunkResult:
    sid: str
    seq: int
    arrival: float
    start: float
    done: float
    deadline: float
    output: Any
    routes: tuple[str, ...]           # per frame: batch / solo / skip
    batch_sizes: tuple[int, ...]      # per frame: size of the Thinker call this session was in (0 = skipped)
    is_last: bool = False

    @property
    def latency_s(self) -> float:     # arrival -> delivered (queueing + service)
        return self.done - self.arrival

    @property
    def service_s(self) -> float:     # round start -> delivered
        return self.done - self.start

    @property
    def late_s(self) -> float:
        return max(0.0, self.done - self.deadline)

    @property
    def missed(self) -> bool:
        return self.done > self.deadline + 1e-9


@dataclass
class ThinkerCall:
    frame: int                        # frame index inside the chunk
    route: str
    sids: tuple[str, ...]
    t0: float
    t1: float
    requests: tuple[FrameRequest, ...] = ()


@dataclass
class RoundRecord:
    """Audit trail of one round (design invariant 6: batch composition per step)."""
    index: int
    t_start: float
    t_end: float
    sids: tuple[str, ...]
    calls: list[ThinkerCall] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)


def percentile(xs: Sequence[float], p: float) -> float | None:
    """Linear interpolation between closest ranks (numpy's default); None for no data."""
    if not xs:
        return None
    s = sorted(xs)
    pos = (len(s) - 1) * p / 100.0
    lo, hi = math.floor(pos), math.ceil(pos)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


@dataclass
class SessionMetrics:
    latency_s: list[float] = field(default_factory=list)
    service_s: list[float] = field(default_factory=list)
    misses: list[tuple[int, float]] = field(default_factory=list)   # (seq, lateness s)
    chunks: int = 0
    dropped: int = 0                  # chunks discarded by leave(drain=False)
    routes: collections.Counter = field(default_factory=collections.Counter)
    error: str | None = None


@dataclass
class SchedulerMetrics:
    sessions: dict[str, SessionMetrics] = field(default_factory=dict)
    batch_hist: collections.Counter = field(default_factory=collections.Counter)   # Thinker-call size -> calls
    route_hist: collections.Counter = field(default_factory=collections.Counter)   # route -> calls
    rounds: int = 0
    completed: int = 0
    busy_s: float = 0.0               # sum of round durations
    first_arrival: float | None = None
    last_done: float | None = None

    def summary(self) -> dict:
        span = (self.last_done - self.first_arrival) if self.completed and self.first_arrival is not None else 0.0
        r3 = lambda x: None if x is None else round(x * 1000, 3)   # noqa: E731 - s -> ms
        per = {}
        for sid, m in self.sessions.items():
            per[sid] = {"chunks": m.chunks, "dropped": m.dropped, "error": m.error,
                        "latency_ms_p50": r3(percentile(m.latency_s, 50)),
                        "latency_ms_p95": r3(percentile(m.latency_s, 95)),
                        "service_ms_p50": r3(percentile(m.service_s, 50)),
                        "service_ms_p95": r3(percentile(m.service_s, 95)),
                        "deadline_misses": len(m.misses),
                        "max_late_ms": r3(max((late for _, late in m.misses), default=0.0)),
                        "routes": dict(m.routes)}
        return {"rounds": self.rounds, "completed_chunks": self.completed,
                "throughput_chunks_per_s": round(self.completed / span, 3) if span > 0 else None,
                "busy_s": round(self.busy_s, 6),
                "busy_chunks_per_s": round(self.completed / self.busy_s, 3) if self.busy_s > 0 else None,
                "batch_hist": dict(sorted(self.batch_hist.items())), "route_hist": dict(self.route_hist),
                "deadline_misses": sum(len(m.misses) for m in self.sessions.values()),
                "sessions": per}


# ------------------------------------------------------------------------------------------------ scheduler
@dataclass
class _Chunk:
    seq: int
    data: Any
    arrival: float
    is_last: bool


@dataclass
class _Slot:
    sid: str
    order: int
    pipeline: Any
    queue: collections.deque = field(default_factory=collections.deque)
    next_seq: int = 0
    last_arrival: float | None = None
    leaving: bool = False             # no more submits; removed at the next boundary once the queue is empty
    finished: bool = False            # served its is_last chunk / left / failed


@dataclass
class Event:
    """Simulation input for ``BatchScheduler.simulate``: kind = join | chunk | leave | call."""
    t: float
    kind: str
    sid: str = ""
    data: Any = None                  # chunk payload, (pipeline, cache) for join, or a callable for call
    is_last: bool = False
    drain: bool = True                # leave only


class BatchScheduler:
    def __init__(self, backend: ThinkerBatchBackend, *, state_store: SessionStateStore | None = None,
                 clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep,
                 chunk_s: float = 0.16, frames_per_chunk: int = 2, deadline_s: float | None = None,
                 bucket_of: Callable[[int], Hashable] = pow2_bucket,
                 group_key: Callable[[str, Phase, int | None], Hashable] | None = None,
                 routes: Mapping[Phase, Route] | None = None, coalesce_s: float = 0.0,
                 singleton_via_step_one: bool = True,
                 on_result: Callable[[ChunkResult], None] | None = None, keep_trace: bool = True):
        self.backend = backend
        self.store = state_store or NullStateStore()
        self.clock, self.sleep = clock, sleep
        self.chunk_s, self.frames_per_chunk = chunk_s, frames_per_chunk
        self.deadline_s = chunk_s if deadline_s is None else deadline_s
        self.bucket_of = bucket_of
        self.group_key = group_key
        self.routes = {**DEFAULT_ROUTES, **(routes or {})}
        # wait up to coalesce_s after the oldest ready chunk for sessions whose next chunk is due by then, so that
        # sessions with offset arrival phases share Thinker calls. 0 = serve whatever is ready (no added latency).
        self.coalesce_s = coalesce_s
        self.singleton_via_step_one = singleton_via_step_one
        self.on_result = on_result
        self.keep_trace = keep_trace
        self.metrics = SchedulerMetrics()
        self.trace: list[RoundRecord] = []
        self.results: dict[str, list[ChunkResult]] = collections.defaultdict(list)
        self.detached: dict[str, Any] = {}       # sid -> B=1 cache returned by backend.detach (for a rejoin)
        self._slots: dict[str, _Slot] = {}
        self._order = 0
        self._lock = threading.RLock()

    # -------------------------------------------------------------------------------------- membership / input
    def join(self, sid: str, pipeline: SessionPipeline | None = None, cache: Any = None) -> None:
        """Register a session; it takes part from the next round (chunk boundary). ``cache`` (the session's B=1
        hybrid cache after its prefill) is attached to the backend; None = the glue attached it already."""
        with self._lock:
            if sid in self._slots:
                raise ValueError(f"session {sid!r} already joined")
            if cache is not None:
                self.backend.attach(sid, cache)
            self.detached.pop(sid, None)
            self._slots[sid] = _Slot(sid=sid, order=self._order, pipeline=pipeline or PassthroughPipeline())
            self._order += 1
            self.metrics.sessions.setdefault(sid, SessionMetrics())

    def leave(self, sid: str, drain: bool = True) -> None:
        """Stop a session at a chunk boundary. drain=True serves its queued chunks first; False drops them (counted)."""
        with self._lock:
            slot = self._slots.get(sid)
            if slot is None or slot.finished:
                return
            slot.leaving = True
            if not drain:
                self.metrics.sessions[sid].dropped += len(slot.queue)
                slot.queue.clear()
            self._reap()

    def submit(self, sid: str, chunk: Any, is_last: bool = False, arrival: float | None = None) -> int:
        """Enqueue one input chunk (arrival defaults to now). Returns its per-session chunk index."""
        with self._lock:
            slot = self._slots.get(sid)
            if slot is None:
                raise KeyError(f"unknown session {sid!r}")
            if slot.leaving or slot.finished:
                raise ValueError(f"session {sid!r} is closed for input")
            t = self.clock() if arrival is None else arrival
            c = _Chunk(seq=slot.next_seq, data=chunk, arrival=t, is_last=is_last)
            slot.next_seq += 1
            slot.last_arrival = t
            slot.queue.append(c)
            if is_last:
                slot.leaving = True
            if self.metrics.first_arrival is None or t < self.metrics.first_arrival:
                self.metrics.first_arrival = t
            return c.seq

    @property
    def active_sessions(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(s.sid for s in self._ordered() if not s.finished)

    def pending(self) -> int:
        with self._lock:
            return sum(len(s.queue) for s in self._slots.values())

    # -------------------------------------------------------------------------------------- planning
    def _ordered(self) -> list[_Slot]:
        return sorted(self._slots.values(), key=lambda s: s.order)

    def _reap(self) -> None:
        for slot in list(self._slots.values()):
            if slot.leaving and not slot.queue and not slot.finished:
                self._finish(slot)

    def _finish(self, slot: _Slot) -> None:
        """Teardown at a boundary: detach the Thinker row (cache kept for a rejoin), drop non-Thinker state."""
        slot.finished = True
        try:
            if slot.sid in self.backend.positions():
                self.detached[slot.sid] = self.backend.detach(slot.sid)
            self.store.drop(slot.sid)
        finally:
            del self._slots[slot.sid]

    def _plan(self, now: float) -> tuple[list[_Slot], float | None]:
        """(ready slots in join order, coalescing wake-up time or None)."""
        ready = [s for s in self._ordered() if s.queue and s.queue[0].arrival <= now]
        if not ready or self.coalesce_s <= 0:
            return ready, None
        horizon = min(s.queue[0].arrival for s in ready) + self.coalesce_s
        due = []
        for s in self._ordered():
            if s in ready or s.leaving or s.last_arrival is None:
                continue
            nxt = s.queue[0].arrival if s.queue else s.last_arrival + self.chunk_s
            if now < nxt <= horizon:
                due.append(nxt)
        return ready, (max(due) if due else None)

    def next_wakeup(self) -> float | None:
        """When ``run_once`` can next do work: now (ready), a coalescing target, a queued future arrival, or None."""
        with self._lock:
            now = self.clock()
            ready, wake = self._plan(now)
            if ready:
                return wake if wake is not None else now
            future = [s.queue[0].arrival for s in self._slots.values() if s.queue]
            return min(future) if future else None

    def _group(self, sid: str, ph: Phase, pos: int | None) -> Hashable:
        if self.group_key is not None:
            return (ph, self.group_key(sid, ph, pos))
        return (ph, None if pos is None else self.bucket_of(pos + 1))

    # -------------------------------------------------------------------------------------- execution
    def run_once(self) -> RoundRecord | None:
        """Execute one round if any session is ready (and no coalescing wait applies). Synchronous."""
        with self._lock:
            now = self.clock()
            ready, wake = self._plan(now)
            if not ready or wake is not None:
                return None
            batch = [(s, s.queue.popleft()) for s in ready]
        return self._execute(batch)

    def _execute(self, batch: list[tuple[_Slot, _Chunk]]) -> RoundRecord:
        m = self.metrics
        t_start = self.clock()
        rec = RoundRecord(index=m.rounds, t_start=t_start, t_end=t_start, sids=tuple(s.sid for s, _ in batch))
        live: dict[str, tuple[_Slot, _Chunk]] = {s.sid: (s, c) for s, c in batch}
        frames: dict[str, list[Any]] = {}
        outs: dict[str, list[Any]] = {sid: [] for sid in live}
        routes: dict[str, list[str]] = {sid: [] for sid in live}
        sizes: dict[str, list[int]] = {sid: [] for sid in live}

        def fail(sid: str, err: BaseException) -> None:
            rec.errors[sid] = f"{type(err).__name__}: {err}"
            m.sessions[sid].error = rec.errors[sid]
            live.pop(sid, None)

        for sid, (slot, ch) in list(live.items()):          # perception (per session, its own state swapped in)
            try:
                self.store.restore(sid)
                fr = list(slot.pipeline.perceive(ch.data, ch.is_last))
                if len(fr) != self.frames_per_chunk:
                    raise ValueError(f"perceive returned {len(fr)} frames, expected {self.frames_per_chunk}")
                frames[sid] = fr
            except Exception as e:  # noqa: BLE001 - one session's failure never stops the others
                fail(sid, e)

        mb = max(1, int(getattr(self.backend, "max_rows", 1)))
        for k in range(self.frames_per_chunk):               # Thinker, one frame index at a time
            pos = dict(self.backend.positions())             # I6: the only backend state the scheduler reads
            groups: dict[Hashable, list[FrameRequest]] = {}
            solo: list[FrameRequest] = []
            for sid, (slot, ch) in list(live.items()):
                try:
                    ph = slot.pipeline.phase()
                    if not isinstance(ph, Phase):
                        ph = Phase(ph)
                    req = FrameRequest(sid=sid, chunk_idx=ch.seq, frame_idx=k, phase=ph, arrival_s=ch.arrival,
                                       payload=frames[sid][k])
                    route = self.routes[ph]
                    if route is Route.BATCH and sid not in pos:
                        route = Route.SOLO                   # not attached: cannot share a batched call
                    if route is Route.BATCH:
                        groups.setdefault(self._group(sid, ph, pos.get(sid)), []).append(req)
                    elif route is Route.SOLO:
                        solo.append(req)
                    else:
                        outs[sid].append(None)
                        routes[sid].append(Route.SKIP.value)
                        sizes[sid].append(0)
                except Exception as e:  # noqa: BLE001
                    fail(sid, e)
            calls: list[tuple[str, tuple[FrameRequest, ...]]] = []
            for members in groups.values():
                for i in range(0, len(members), mb):
                    calls.append((Route.BATCH.value, tuple(members[i:i + mb])))
            calls += [(Route.SOLO.value, (r,)) for r in solo]
            for route, reqs in calls:
                sids = tuple(r.sid for r in reqs)
                t0 = self.clock()
                if route == Route.SOLO.value or (len(reqs) == 1 and self.singleton_via_step_one):
                    try:
                        res = {sids[0]: self.backend.step_one(sids[0], reqs[0].payload)}
                    except Exception as e:  # noqa: BLE001 - a B=1 failure belongs to that session only
                        fail(sids[0], e)
                        continue
                else:                                        # a batched failure cannot be attributed: propagate
                    res = self.backend.step({r.sid: r.payload for r in reqs})
                    if set(res) != set(sids):
                        raise RuntimeError(f"step returned rows {sorted(res)} for {sorted(sids)}")
                t1 = self.clock()
                rec.calls.append(ThinkerCall(frame=k, route=route, sids=sids, t0=t0, t1=t1, requests=reqs))
                m.batch_hist[len(sids)] += 1
                m.route_hist[route] += 1
                for sid in sids:
                    outs[sid].append(res[sid])
                    routes[sid].append(route)
                    sizes[sid].append(len(sids))

        for sid, (slot, ch) in list(live.items()):          # post-processing + delivery (per session)
            try:
                self.store.restore(sid)
                output = slot.pipeline.postprocess(outs[sid], ch.is_last)
            except Exception as e:  # noqa: BLE001
                fail(sid, e)
                continue
            done = self.clock()
            r = ChunkResult(sid=sid, seq=ch.seq, arrival=ch.arrival, start=t_start, done=done,
                            deadline=ch.arrival + self.deadline_s, output=output, routes=tuple(routes[sid]),
                            batch_sizes=tuple(sizes[sid]), is_last=ch.is_last)
            sm = m.sessions[sid]
            sm.chunks += 1
            sm.latency_s.append(r.latency_s)
            sm.service_s.append(r.service_s)
            sm.routes.update(r.routes)
            if r.missed:
                sm.misses.append((r.seq, r.late_s))
            m.completed += 1
            m.last_done = done if m.last_done is None else max(m.last_done, done)
            self.results[sid].append(r)
            if self.on_result is not None:
                self.on_result(r)

        rec.t_end = self.clock()
        m.rounds += 1
        m.busy_s += rec.t_end - rec.t_start
        if self.keep_trace:
            self.trace.append(rec)
        with self._lock:                                    # boundary: failed sessions out, drained leavers out
            for sid in rec.errors:
                slot = self._slots.get(sid)
                if slot is not None:
                    m.sessions[sid].dropped += len(slot.queue)
                    slot.queue.clear()
                    self._finish(slot)
            self._reap()
        return rec

    # -------------------------------------------------------------------------------------- drivers
    def run(self, stop: Callable[[], bool], idle_s: float = 0.005) -> None:
        """Realtime loop for live producers calling ``submit`` from other threads; returns when ``stop()``."""
        while not stop():
            if self.run_once() is not None:
                continue
            wake = self.next_wakeup()
            now = self.clock()
            self.sleep(max(0.0, min(wake - now, idle_s)) if wake is not None else idle_s)

    def simulate(self, events: Iterable[Event]) -> SchedulerMetrics:
        """Deterministic driver: apply timed events at their time, run rounds as soon as work is ready, sleep
        (injectable) to the next event otherwise. Works with a fake clock (tests) or the real one (harness)."""
        evs = sorted(events, key=lambda e: e.t)   # stable: same-time events keep their given order
        i = 0
        while True:
            now = self.clock()
            while i < len(evs) and evs[i].t <= now:
                self._apply(evs[i])
                i += 1
            if self.run_once() is not None:
                continue
            cands = [w for w in (self.next_wakeup(),) if w is not None and w > now]
            if i < len(evs):
                cands.append(evs[i].t)
            if not cands:
                if self.pending() == 0:
                    return self.metrics
                raise RuntimeError("scheduler stalled with pending chunks")   # defensive; should not happen
            self.sleep(max(0.0, min(cands) - now))

    def _apply(self, e: Event) -> None:
        if e.kind == "join":
            pipeline, cache = e.data if isinstance(e.data, tuple) else (e.data, None)
            self.join(e.sid, pipeline, cache)
        elif e.kind == "chunk":
            sm = self.metrics.sessions.get(e.sid)
            if e.sid not in self._slots and sm is not None and sm.error is not None:
                sm.dropped += 1                 # the producer keeps sending after the session failed
                return
            self.submit(e.sid, e.data, is_last=e.is_last, arrival=e.t)
        elif e.kind == "leave":
            self.leave(e.sid, drain=e.drain)
        elif e.kind == "call":
            e.data()
        else:
            raise ValueError(f"unknown event kind {e.kind!r}")


def stream_events(sid: str, chunks: Sequence[Any], t0: float = 0.0, chunk_s: float = 0.16, join: bool = True,
                  pipeline: Any = None, cache: Any = None, last: bool = True) -> list[Event]:
    """Events for one realtime session: optional join at t0, chunk k at t0 + k*chunk_s (the last one flagged)."""
    ev = [Event(t0, "join", sid, (pipeline, cache))] if join else []
    n = len(chunks)
    ev += [Event(t0 + k * chunk_s, "chunk", sid, c, is_last=last and k == n - 1) for k, c in enumerate(chunks)]
    return ev
