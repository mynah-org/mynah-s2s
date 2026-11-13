"""Experimental multi-session scheduler against fakes (no model, no torch). These tests check OUR scheduling logic —
grouping, routing, deadlines, join/leave, isolation at the scheduler level — and the contract
(runtime/batch_contract.py I1/I3/I6/I7). They say nothing about batched-forward numerics (thinker_batch owns that).

Times use chunk_s = 0.125 (exact in binary) where round alignment matters, so arrivals of different sessions at
"the same" chunk boundary compare equal."""

import pytest

from mynah_v2.runtime.batch_contract import (FrameRequest, Phase, SessionStateStore, ThinkerBatchBackend,
                                             ThinkerOut)
from mynah_v2.runtime.batch_scheduler import (BatchScheduler, Event, Route, percentile, pow2_bucket,
                                              stream_events)

CH = 0.125


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def advance(self, dt):
        assert dt >= 0
        self.t += dt


class FakeBackend:
    """Owns per-row recurrent state (pos, acc). step() advances ONLY the rows passed in (I1)."""

    def __init__(self, clock, max_rows=4, cost_one=0.01, cost_row=0.002):
        self.clock, self.max_rows = clock, max_rows
        self.cost_one, self.cost_row = cost_one, cost_row
        self.rows: dict[str, list[int]] = {}
        self.calls: list[tuple[str, frozenset]] = []

    def attach(self, sid, cache):
        if sid in self.rows:
            raise ValueError(f"{sid} already attached")
        if len(self.rows) >= self.max_rows:
            raise RuntimeError("backend full")
        self.rows[sid] = list(cache)
        return len(self.rows) - 1

    def detach(self, sid):
        return tuple(self.rows.pop(sid))

    def positions(self):
        return {sid: r[0] for sid, r in self.rows.items()}

    def _advance(self, sid, frame):
        r = self.rows[sid]                       # KeyError if not attached: the scheduler routed a stranger
        r[0] += 1
        r[1] = (r[1] * 31 + frame) % 1_000_003
        return ThinkerOut(text_logits=(r[0], r[1]))

    def step(self, frames):
        assert 1 <= len(frames) <= self.max_rows
        self.calls.append(("step", frozenset(frames)))
        self.clock.advance(self.cost_one + self.cost_row * (len(frames) - 1))
        return {sid: self._advance(sid, f) for sid, f in frames.items()}

    def step_one(self, sid, frame):
        self.calls.append(("one", frozenset([sid])))
        self.clock.advance(self.cost_one)
        return self._advance(sid, frame)

    def memory(self):
        return None

    def close(self):
        self.rows.clear()


class FakeStore:
    def __init__(self):
        self.current = None
        self.dropped = set()
        self.restores = []

    def capture(self, sid):
        self.dropped.discard(sid)
        return sid

    def restore(self, sid):
        if sid in self.dropped:
            raise KeyError(f"{sid} dropped")
        self.current = sid
        self.restores.append(sid)

    def drop(self, sid):
        self.dropped.add(sid)


class FakePipeline:
    """Per-session non-Thinker work. ``phase_fn(chunk_no, frame_k) -> Phase``; checks its state is swapped in."""

    def __init__(self, sid, store, clock, phase_fn=None, perceive_cost=0.002, post_cost=0.002, fail_at=None):
        self.sid, self.store, self.clock = sid, store, clock
        self.phase_fn = phase_fn or (lambda n, k: Phase.DECODE)
        self.perceive_cost, self.post_cost, self.fail_at = perceive_cost, post_cost, fail_at
        self.n = -1        # chunks perceived so far - 1
        self.k = 0

    def perceive(self, chunk, is_last):
        assert self.store.current == self.sid, "non-Thinker work ran without the session's state restored"
        self.n += 1
        self.k = 0
        if self.fail_at is not None and self.n == self.fail_at:
            raise RuntimeError("perception exploded")
        self.clock.advance(self.perceive_cost)
        return list(chunk)

    def phase(self):
        ph = self.phase_fn(self.n, self.k)
        self.k += 1
        return ph

    def postprocess(self, outs, is_last):
        assert self.store.current == self.sid
        self.clock.advance(self.post_cost)
        return tuple(None if o is None else o.text_logits for o in outs)


def stream(sid, chunks, **kw):
    kw.setdefault("chunk_s", CH)
    return stream_events(sid, chunks, **kw)


def chunks_for(sid, n, base=0):
    s = sum(map(ord, sid)) * 101
    return [[s + 2 * (base + i), s + 2 * (base + i) + 1] for i in range(n)]


def make(chunk_s=CH, **kw):
    clock = FakeClock()
    store = FakeStore()
    bk = kw.pop("backend_kw", {})
    backend = FakeBackend(clock, **bk)
    sched = BatchScheduler(backend, state_store=store, clock=clock, sleep=clock.advance, chunk_s=chunk_s, **kw)
    return sched, backend, store, clock


def cache0(sid, pos=0):
    return (pos, sum(map(ord, sid)))


def solo_outputs(sid, n, t0=0.0, phase_fn=None, chunk_s=CH, pos=0, **kw):
    """The session alone through a fresh scheduler + same fake backend: the isolation oracle."""
    sched, _, store, clock = make(chunk_s=chunk_s, **kw)
    p = FakePipeline(sid, store, clock, phase_fn)
    sched.simulate(stream(sid, chunks_for(sid, n), t0=t0, chunk_s=chunk_s, pipeline=p,
                                 cache=cache0(sid, pos)))
    return [r.output for r in sched.results[sid]]


def outputs(sched, sid):
    return [r.output for r in sched.results[sid]]


def step_sets(backend):
    return [s for kind, s in backend.calls if kind == "step"]


# ------------------------------------------------------------------------------------------------ contract
def test_fakes_conform_to_contract():
    clock = FakeClock()
    assert isinstance(FakeBackend(clock), ThinkerBatchBackend)
    assert isinstance(FakeStore(), SessionStateStore)


def test_scheduler_reads_only_positions_and_host_api():
    """I6: the scheduler touches no backend attribute beyond the host API (no K/V, Mamba state, buffers)."""
    sched, backend, store, clock = make()
    seen = set()

    class Spy:
        def __getattr__(self, name):
            seen.add(name)
            return getattr(backend, name)

    sched.backend = Spy()
    ev = []
    for sid in ("A", "B"):
        ev += stream(sid, chunks_for(sid, 3), pipeline=FakePipeline(sid, store, clock), cache=cache0(sid))
    sched.simulate(ev)
    assert seen <= {"attach", "detach", "positions", "step", "step_one", "max_rows"}
    assert "rows" not in seen


# ------------------------------------------------------------------------------------------------ arrivals
def test_two_sessions_offset_arrivals_without_and_with_coalescing():
    n = 6
    for coalesce in (0.0, 0.06):
        sched, backend, store, clock = make(chunk_s=0.16, coalesce_s=coalesce)
        ev = stream("A", chunks_for("A", n), t0=0.0, chunk_s=0.16,
                           pipeline=FakePipeline("A", store, clock), cache=cache0("A"))
        ev += stream("B", chunks_for("B", n), t0=0.05, chunk_s=0.16,
                            pipeline=FakePipeline("B", store, clock), cache=cache0("B"))
        m = sched.simulate(ev)
        assert outputs(sched, "A") == solo_outputs("A", n, chunk_s=0.16)
        assert outputs(sched, "B") == solo_outputs("B", n, t0=0.05, chunk_s=0.16)
        s = m.summary()
        assert s["deadline_misses"] == 0
        assert s["completed_chunks"] == 2 * n
        if coalesce == 0.0:
            # A finishes each chunk (~28 ms) before B's arrives 50 ms later: never batched, never delayed
            assert dict(m.batch_hist) == {1: 4 * n}
            assert step_sets(backend) == []
            assert max(sched.metrics.sessions["A"].latency_s) < 0.03
        else:
            # first chunk of each alone (B's phase unknown yet), then A waits for B's due chunk: 5 shared rounds
            assert dict(m.batch_hist) == {1: 4, 2: 2 * (n - 1)}
            assert all(s == frozenset("AB") for s in step_sets(backend))
            # the wait is visible in A's latency (arrival -> done), not hidden
            # (A is post-processed before B inside the round: 2 perceptions, 2 shared frames, its own post)
            assert max(sched.metrics.sessions["A"].latency_s) == pytest.approx(0.05 + 0.002 * 2 + 0.012 * 2 + 0.002)


def test_batched_call_gets_exactly_the_ready_and_compatible_set():
    """Bucket classes from positions() (I5/I6) and max_rows splitting."""
    sched, backend, store, clock = make(backend_kw={"max_rows": 4})
    pos = {"A": 10, "B": 20, "C": 100, "D": 30}
    for sid, p in pos.items():
        sched.join(sid, FakePipeline(sid, store, clock), cache0(sid, p))
    sched.backend.max_rows = 2                      # grouping must split a 3-session bucket into 2 + 1
    for sid in ("A", "B", "C", "D"):
        sched.submit(sid, chunks_for(sid, 1)[0], arrival=0.0)
    rec = sched.run_once()
    frame0 = [(c.route, frozenset(c.sids)) for c in rec.calls if c.frame == 0]
    assert pow2_bucket(11) == pow2_bucket(31) == 64 and pow2_bucket(101) == 128
    assert frame0 == [("batch", frozenset("AB")), ("batch", frozenset("D")), ("batch", frozenset("C"))]
    assert backend.calls[0] == ("step", frozenset("AB"))
    assert backend.calls[1:3] == [("one", frozenset("D")), ("one", frozenset("C"))]   # singletons -> step_one
    reqs = rec.calls[0].requests
    assert all(isinstance(r, FrameRequest) and r.phase is Phase.DECODE and r.frame_idx == 0 for r in reqs)
    assert reqs[0].arrival_s == 0.0 and reqs[0].payload == chunks_for("A", 1)[0][0]


def test_custom_group_key_overrides_buckets():
    sched, backend, store, clock = make(group_key=lambda sid, ph, pos: sid in "AC")
    for sid in "ABCD":
        sched.join(sid, FakePipeline(sid, store, clock), cache0(sid))
        sched.submit(sid, chunks_for(sid, 1)[0], arrival=0.0)
    sched.run_once()
    assert step_sets(backend)[:2] == [frozenset("AC"), frozenset("BD")]


# ------------------------------------------------------------------------------------------------ tool pending
def _pending_b(n, k):
    if 2 <= n <= 4:
        return Phase.TOOL_PENDING
    if n == 5 and k == 0:
        return Phase.FC_FORCED
    return Phase.DECODE


def test_tool_pending_session_goes_solo_and_never_blocks_others():
    n = 9
    sched, backend, store, clock = make()
    ev = []
    for sid in "ABC":
        pf = _pending_b if sid == "B" else None
        ev += stream(sid, chunks_for(sid, n), pipeline=FakePipeline(sid, store, clock, pf), cache=cache0(sid))
    m = sched.simulate(ev)
    for rec in sched.trace:
        for c in rec.calls:
            if c.route == "batch" and len(c.sids) > 1:
                ph = {r.sid: r.phase for r in c.requests}
                assert all(p is Phase.DECODE for p in ph.values())       # I7: only DECODE is batched
    # chunks 2..4 (both frames) + chunk 5 frame 0: the batch is exactly {A, C}, B goes B=1
    sets = step_sets(backend)
    assert sets.count(frozenset("AC")) == 2 * 3 + 1
    assert sets.count(frozenset("ABC")) == 2 * n - 7
    rb = sched.results["B"]
    assert rb[3].routes == ("solo", "solo") and rb[5].routes == ("solo", "batch") and rb[6].batch_sizes == (3, 3)
    assert sched.results["A"][3].batch_sizes == (2, 2)
    for sid in "ABC":
        assert outputs(sched, sid) == solo_outputs(sid, n, phase_fn=_pending_b if sid == "B" else None)
    assert m.summary()["deadline_misses"] == 0


def test_tool_pending_skip_route_leaves_row_untouched():
    """Route TOOL_PENDING -> SKIP: no Thinker work, B's row does not move (I1), others unaffected."""
    sched, backend, store, clock = make(routes={Phase.TOOL_PENDING: Route.SKIP})
    seen_pos = []
    pf = lambda n, k: Phase.TOOL_PENDING if n in (1, 2) else Phase.DECODE   # noqa: E731
    ev = stream("A", chunks_for("A", 4), pipeline=FakePipeline("A", store, clock), cache=cache0("A"))
    ev += stream("B", chunks_for("B", 4), pipeline=FakePipeline("B", store, clock, pf), cache=cache0("B"))
    ev += [Event(t, "call", data=lambda: seen_pos.append(dict(backend.positions()))) for t in (0.1, 0.2, 0.3)]
    sched.simulate(ev)
    assert [p["B"] for p in seen_pos] == [2, 2, 2] and [p["A"] for p in seen_pos] == [2, 4, 6]
    assert sched.results["B"][1].output == (None, None) and sched.results["B"][1].batch_sizes == (0, 0)
    assert outputs(sched, "A") == solo_outputs("A", 4)


# ------------------------------------------------------------------------------------------------ join / leave
def test_join_leave_rejoin_slot_reuse_sequence():
    """I3 at the scheduler level: {A+B} -> A leaves, C joins -> {B+C} -> B ends, A rejoins -> {A+C};
    every session's outputs equal its solo run (A's two segments = one continuous solo run)."""
    sched, backend, store, clock = make(backend_kw={"max_rows": 2})
    a1, a2 = chunks_for("A", 3), chunks_for("A", 3, base=3)
    pa = FakePipeline("A", store, clock)
    ev = stream("A", a1, t0=0.0, pipeline=pa, cache=cache0("A"), last=False)
    ev += stream("B", chunks_for("B", 7), t0=0.0, pipeline=FakePipeline("B", store, clock), cache=cache0("B"))
    ev.append(Event(3 * CH - 0.01, "leave", "A"))
    ev += stream("C", chunks_for("C", 8), t0=3 * CH, pipeline=FakePipeline("C", store, clock),
                        cache=cache0("C"))
    def rejoin():
        store.capture("A")                      # the glue re-registers A's non-Thinker state before rejoining
        sched.join("A", pa, sched.detached["A"])
    ev.append(Event(7 * CH, "call", data=rejoin))
    ev += stream("A", a2, t0=7 * CH, join=False)
    m = sched.simulate(ev)

    dedup = []
    for s in step_sets(backend):
        if not dedup or dedup[-1] != s:
            dedup.append(s)
    assert dedup == [frozenset("AB"), frozenset("BC"), frozenset("AC")]
    assert step_sets(backend).count(frozenset("AB")) == 6
    assert step_sets(backend).count(frozenset("BC")) == 8
    assert step_sets(backend).count(frozenset("AC")) == 6
    assert outputs(sched, "A") == solo_outputs("A", 6)
    assert outputs(sched, "B") == solo_outputs("B", 7)
    assert outputs(sched, "C") == solo_outputs("C", 8, t0=3 * CH)
    assert m.summary()["deadline_misses"] == 0
    assert set(store.dropped) == {"A", "B", "C"} and backend.rows == {}


def test_leave_without_drain_drops_queued_chunks():
    sched, backend, store, clock = make()
    sched.join("A", FakePipeline("A", store, clock), cache0("A"))
    for c in chunks_for("A", 3):
        sched.submit("A", c, arrival=0.0)
    sched.run_once()                            # one round = the head chunk of each ready session
    assert len(sched.results["A"]) == 1
    sched.leave("A", drain=False)
    assert sched.metrics.sessions["A"].dropped == 2
    assert "A" in sched.detached and sched.active_sessions == ()
    with pytest.raises(KeyError):
        sched.submit("A", chunks_for("A", 1)[0])


def test_eos_is_per_session():
    sched, backend, store, clock = make()
    ev = stream("A", chunks_for("A", 2), pipeline=FakePipeline("A", store, clock), cache=cache0("A"))
    ev += stream("B", chunks_for("B", 6), pipeline=FakePipeline("B", store, clock), cache=cache0("B"))
    sched.simulate(ev)
    assert sched.results["A"][-1].is_last and len(sched.results["A"]) == 2
    assert "A" in store.dropped and "A" not in backend.rows
    assert outputs(sched, "B") == solo_outputs("B", 6)
    assert step_sets(backend).count(frozenset("AB")) == 4


def test_session_failure_is_isolated():
    sched, backend, store, clock = make()
    ev = stream("A", chunks_for("A", 5), pipeline=FakePipeline("A", store, clock), cache=cache0("A"))
    ev += stream("B", chunks_for("B", 5), pipeline=FakePipeline("B", store, clock, fail_at=2),
                        cache=cache0("B"))
    sched.simulate(ev)
    sm = sched.metrics.sessions["B"]
    assert "perception exploded" in sm.error and sm.chunks == 2 and sm.dropped == 2   # chunks 3, 4 after failure
    assert outputs(sched, "A") == solo_outputs("A", 5)
    assert "B" in store.dropped


# ------------------------------------------------------------------------------------------------ deadlines/metrics
def test_deadline_miss_accounting():
    # 2 frames x 0.125 s = 0.25 s per chunk arriving every 0.125 s: backlog grows by 0.125 s per chunk
    sched, backend, store, clock = make(backend_kw={"cost_one": 0.125})
    p = FakePipeline("A", store, clock, perceive_cost=0.0, post_cost=0.0)
    sched.simulate(stream("A", chunks_for("A", 4), pipeline=p, cache=cache0("A")))
    misses = sched.metrics.sessions["A"].misses
    assert [s for s, _ in misses] == [0, 1, 2, 3]
    assert [late for _, late in misses] == pytest.approx([0.125, 0.25, 0.375, 0.5])
    r = sched.results["A"][3]
    assert r.missed and r.deadline == pytest.approx(3 * CH + CH) and r.latency_s == pytest.approx(0.625)
    s = sched.metrics.summary()["sessions"]["A"]
    assert s["deadline_misses"] == 4 and s["max_late_ms"] == pytest.approx(500.0)


def test_metrics_summary():
    sched, backend, store, clock = make()
    ev = []
    for sid in "AB":
        ev += stream(sid, chunks_for(sid, 4), pipeline=FakePipeline(sid, store, clock), cache=cache0(sid))
    m = sched.simulate(ev).summary()
    assert m["completed_chunks"] == 8 and m["rounds"] == 4
    assert m["batch_hist"] == {2: 8} and m["route_hist"] == {"batch": 8}
    # each round: 2 perceptions + 2 batched frames (0.012) + 2 posts = 0.032 s
    # A is post-processed (and delivered) before B inside the round
    a, b = m["sessions"]["A"], m["sessions"]["B"]
    assert a["service_ms_p50"] == pytest.approx(30.0) and a["latency_ms_p95"] == pytest.approx(30.0)
    assert b["service_ms_p50"] == pytest.approx(32.0)
    assert m["busy_s"] == pytest.approx(4 * 0.032)
    assert m["throughput_chunks_per_s"] == pytest.approx(8 / (3 * CH + 0.032), abs=1e-3)


def test_percentile():
    assert percentile([], 50) is None
    assert percentile([3.0], 95) == 3.0
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile(list(range(101)), 95) == 95


def test_join_twice_and_backend_full_raise():
    sched, backend, store, clock = make(backend_kw={"max_rows": 1})
    sched.join("A", None, cache0("A"))
    with pytest.raises(ValueError):
        sched.join("A")
    with pytest.raises(RuntimeError):
        sched.join("B", None, cache0("B"))
    assert sched.active_sessions == ("A",)
