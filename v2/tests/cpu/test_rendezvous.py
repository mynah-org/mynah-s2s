"""Cooperative rendezvous (runtime/rendezvous.py) on the session_state fakes + a fake Thinker and a fake row backend.

The fake Thinker is recurrent (its output depends on the whole history of its cache), so any cross-session state leak,
lost update or double advance changes the tokens. Every session's outputs must equal its solo single-path run."""

import importlib
import random
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
T = importlib.import_module("test_session_state")
fake_nemo = T.fake_nemo                                  # autouse fixture (nemo stand-ins)

from mynah_v2.model.reference import ReferenceEngine  # noqa: E402
from mynah_v2.model.session_state import MultiSessionEngine  # noqa: E402
from mynah_v2.model.session_tools import SessionToolBinding  # noqa: E402
from mynah_v2.runtime.batch_contract import ThinkerBatchBackend, ThinkerOut  # noqa: E402
from mynah_v2.runtime.rendezvous import OwnershipError, RendezvousEngine, RendezvousError  # noqa: E402

V = 50
CALL_THREADS: list = []
CALL_GRAD: list = []


class Cache:
    def __init__(self):
        self.s, self._v2_pos = [], 0


def advance(state: list, x) -> int:
    for t in range(x.shape[1]):
        state.append(float(x[0, t, 0]))
    return int((sum(state) * 7 + len(state) * 3) % V)


def logits(v):
    out = torch.zeros(1, 1, V)
    out[0, 0, v] = 1.0
    return out


class FakeSTT:
    def forward(self, input_embeds, cache=None, **kw):
        CALL_THREADS.append(threading.current_thread())
        CALL_GRAD.append((torch.is_grad_enabled(), torch.is_inference_mode_enabled()))
        if cache.s is None:
            raise RuntimeError("stt.forward on a cache owned by the backend")
        v = advance(cache.s, input_embeds)
        cache._v2_pos += input_embeds.shape[1]
        return {"text_logits": logits(v), "function_logits": None, "cache": cache}

    def __call__(self, *a, **k):
        return self.forward(*a, **k)


class FakeBackend:
    """ThinkerBatchBackend over the same recurrence; rows own the state while attached (cache.s = None)."""

    def __init__(self, max_rows=2, lmax=10**9):
        self.max_rows, self.rows, self.caches, self.calls = max_rows, {}, {}, []
        self.lmax, self.evicted = lmax, {}

    def attach(self, sid, cache):
        if len(self.rows) >= self.max_rows:
            raise RuntimeError("full")
        self.rows[sid], self.caches[sid] = (list(cache.s), cache._v2_pos), cache
        cache.s = None
        return len(self.rows) - 1

    def detach(self, sid):
        (s, pos), c = self.rows.pop(sid), self.caches.pop(sid)
        c.s, c._v2_pos = s, pos
        return c

    def positions(self):
        return {sid: pos for sid, (_, pos) in self.rows.items()}

    def _one(self, sid, x):
        CALL_GRAD.append((torch.is_grad_enabled(), torch.is_inference_mode_enabled()))
        s, pos = self.rows[sid]
        v = advance(s, x)
        self.rows[sid] = (s, pos + 1)
        return ThinkerOut(text_logits=logits(v))

    def step(self, frames):
        self.calls.append(sorted(frames))
        out = {}
        for sid, x in frames.items():
            if self.rows[sid][1] + 1 > self.lmax:      # like BatchedThinker: back to its own cache, frame eager
                c = self.detach(sid)
                c._v2_past_lmax = True
                o = FakeSTT().forward(x, cache=c)
                out[sid] = ThinkerOut(text_logits=o["text_logits"])
                self.evicted[sid] = c
            else:
                out[sid] = self._one(sid, x)
        return out

    def step_one(self, sid, frame):
        return self.step({sid: frame})[sid]

    def memory(self):
        return None

    def close(self):
        pass


class Wrapper(T.FakeWrapper):
    def __init__(self):
        super().__init__()
        self.model = SimpleNamespace(stt_model=FakeSTT())

    def _run_fc_async_steps(self, **kw):              # FC thread: drives the Thinker of its session (B=1)
        c = kw.get("cache")
        if c is not None:
            self.model.stt_model(torch.full((1, 1, 1), 5.0), cache=c)
            if kw.get("hold") is not None:
                kw["hold"].wait(5)
        return super()._run_fc_async_steps(**kw)


HANG = threading.Event()


class Pipeline(T.FakePipeline):
    def generate_step(self, frames):
        f, w = frames[0], self.s2s_model
        pre = f.is_first and f.samples.numel() == 0
        if not pre and float(f.samples[0]) < 0:
            raise RuntimeError("injected session failure")
        if not pre and float(f.samples[0]) > 1e6:
            HANG.wait(10)
        super().generate_step(frames)
        ctx, _ = self.context_manager.get_context([f.stream_id])
        stt = w.model.stt_model
        if pre:
            ctx.cache = stt(torch.full((1, 3, 1), float(ctx.prompt)), cache=Cache())["cache"]
            return
        toks = []
        for k in range(2):                            # two Thinker frames per chunk
            out = stt(torch.full((1, 1, 1), float(ctx.frame_idx * 10 + k)), cache=ctx.cache)
            ctx.cache = out["cache"]
            toks.append(int(out["text_logits"].argmax()))
        self.get_or_create_state(f.stream_id).text += f" T{toks}"


def make_engine():
    w = Wrapper()
    p = Pipeline(w)
    eng = object.__new__(ReferenceEngine)
    eng.pipeline, eng.wrapper, eng.tokenizer = p, w, None
    eng.session_tools = SessionToolBinding(p.tool_registry)
    eng.fc_still_running, eng.fc_records, eng._fc_lock = 0, [], threading.Lock()
    eng._instrument_fc()
    eng.unknown_tool_handler = eng.multi_call_handler = None
    eng.unknown_tool_events, eng.multi_call_events = [], []
    eng._wrap_executor()
    eng.chunk_samples, eng.stream_id, eng._started = T.CHUNK, 0, False
    eng._pos, eng._last_frame_idx, eng._rec_cursor, eng.llm_cache_info = {"text": 0, "asr": 0, "func": 0}, 0, 0, None
    return eng


def obs(out):
    return (out.new_text_ids, out.text_delta, out.frame_idx, out.audio.tolist())


def solo(sid, n):
    eng = make_engine()
    torch.manual_seed(T.SESS[sid][2])
    random.seed(T.SESS[sid][2])
    eng.open(T.SESS[sid][0])
    res = [obs(eng.step(T.pcm(sid, k))) for k in range(n)]
    eng.close()
    return res


def rz_engine(max_rows=2, timeout=10.0, lmax=10**9):
    eng = make_engine()
    m = MultiSessionEngine(eng, cuda_rng=False)
    be = FakeBackend(max_rows, lmax)
    assert isinstance(be, ThinkerBatchBackend)
    return eng, m, be, RendezvousEngine(m, be, step_timeout_s=timeout)


def open_(rz, sid):
    rz.open(sid, T.SESS[sid][0], seed=T.SESS[sid][2])


def test_pair_equals_solo_with_real_batched_steps_and_thinker_only_on_the_coordinator():
    n = 6
    sa, sb = solo("A", n), solo("B", n)
    _, m, be, rz = rz_engine()
    open_(rz, "A")
    open_(rz, "B")
    CALL_THREADS.clear()
    pa, pb = [], []
    for k in range(n):
        r = rz.step_jobs({"A": T.pcm("A", k), "B": T.pcm("B", k)})
        pa.append(obs(r["A"]))
        pb.append(obs(r["B"]))
    assert pa == sa and pb == sb
    assert rz.stats["batch_hist"] == {2: 2 * n}                # every Thinker frame batched A+B
    assert all(c == ["A", "B"] for c in be.calls)
    assert all(t is threading.main_thread() for t in CALL_THREADS)   # only the coordinator ran the Thinker
    rz.close_all()


def test_stage_resume_keeps_current_session_first_to_avoid_redundant_state_swaps():
    _, m, be, rz = rz_engine()
    open_(rz, "A")
    open_(rz, "B")
    assert rz.multi.store.active == "B"
    assert rz._resume_order(["A", "B"]) == ["B", "A"]
    assert rz._resume_order(["B"]) == ["B"]
    assert rz._resume_order(["A"]) == ["A"]
    rz.close_all()


def test_one_session_ends_while_the_other_continues_unbatched():
    sa, sb = solo("A", 2), solo("B", 5)
    _, m, be, rz = rz_engine()
    open_(rz, "A")
    open_(rz, "B")
    pa, pb = [], []
    for k in range(5):
        jobs = {"B": T.pcm("B", k)} | ({"A": T.pcm("A", k)} if k < 2 else {})
        r = rz.step_jobs(jobs)
        if k < 2:
            pa.append(obs(r["A"]))
        pb.append(obs(r["B"]))
        if k == 1:
            rz.close("A")
    assert pa == sa and pb == sb
    assert rz.stats["batch_hist"] == {2: 4} and rz.stats["b1_calls"] >= 6   # B alone after A left: B=1
    rz.close_all()


def test_slot_reuse_AB_then_BC_then_AC_matches_continuous_solo_runs():
    sa, sb, sc = solo("A", 6), solo("B", 6), solo("C", 6)
    _, m, be, rz = rz_engine(max_rows=2)
    for s in "AB":
        open_(rz, s)
    got = {"A": [], "B": [], "C": []}
    k = {"A": 0, "B": 0, "C": 0}

    def rnd(sids):
        r = rz.step_jobs({s: T.pcm(s, k[s]) for s in sids})
        for s in sids:
            got[s].append(obs(r[s]))
            k[s] += 1
    for _ in range(3):
        rnd("AB")
    rz.pause("A")                                   # A leaves the batch (row freed), stays open
    open_(rz, "C")
    for _ in range(3):
        rnd("BC")
    rz.pause("B")
    for _ in range(3):
        rnd("AC")                                   # A rejoins (re-attached from its B=1 cache)
    assert got["A"] == sa and got["B"] == sb[:6] and got["C"] == sc
    assert {"A", "B"} in [set(c) for c in be.calls] and {"B", "C"} in [set(c) for c in be.calls]
    assert {"A", "C"} in [set(c) for c in be.calls]
    rz.close_all()


def test_failing_session_is_removed_and_the_other_is_unaffected():
    sa = solo("A", 4)
    _, m, be, rz = rz_engine()
    open_(rz, "A")
    open_(rz, "B")
    pa = []
    for k in range(4):
        jobs = {"A": T.pcm("A", k)} | ({"B": torch.full((T.CHUNK,), -1.0)} if k == 1 else
                                       ({"B": T.pcm("B", k)} if k == 0 else {}))
        r = rz.step_jobs(jobs)
        pa.append(obs(r["A"]))
        if k == 1:
            assert isinstance(r["B"], RuntimeError) and "B" in rz.failed
    assert pa == sa
    assert "B" not in be.positions()
    rz.close_all()


def test_hanging_session_times_out_and_does_not_block_the_other():
    sa = solo("A", 3)
    _, m, be, rz = rz_engine()
    open_(rz, "A")
    open_(rz, "B")
    w = rz.step_jobs({"A": T.pcm("A", 0), "B": T.pcm("B", 0)})      # warm (cold-start imports), normal timeout
    rz.timeout = 1.0
    HANG.clear()
    r0 = rz.step_jobs({"A": T.pcm("A", 1), "B": torch.full((T.CHUNK,), 2e6)})
    assert isinstance(r0["B"], RendezvousError) and "B" in rz.failed
    r1 = rz.step_jobs({"A": T.pcm("A", 2)})
    assert [obs(w["A"]), obs(r0["A"]), obs(r1["A"])] == sa
    HANG.set()
    rz.workers.pop("B")                              # the hung worker is abandoned (daemon), not joined
    rz.close_all()


def test_fc_thread_owns_its_session_thinker_and_a_main_thread_request_is_refused():
    _, m, be, rz = rz_engine()
    open_(rz, "A")
    open_(rz, "B")
    rz.step_jobs({"A": T.pcm("A", 0), "B": T.pcm("B", 0)})
    stream_a = m.store.record("A").fields["engine.stream_id"]
    cache_a = be.caches["A"]                          # A is attached: its FC call must go through step_one
    hold, box = threading.Event(), {}

    def fc():
        try:
            gt = torch.zeros(1, 8, dtype=torch.long)
            m.engine.wrapper._run_fc_async_steps(gen_text=gt, gen_function_text=gt, current_frame_idx=2,
                                                 stream_id=stream_a, cache=cache_a, hold=hold)
        except Exception as e:  # noqa: BLE001
            box["err"] = e
    t = threading.Thread(target=fc)
    t.start()
    deadline = time.monotonic() + 5
    while rz.thinker_owner.get("A") is None and time.monotonic() < deadline:
        rz.pump(until=time.monotonic() + 0.01)
    assert rz.thinker_owner.get("A") is t
    rz.pump(until=time.monotonic() + 0.2)            # the FC thread's Thinker call is served B=1
    assert rz.stats["fc_b1_calls"] == 1 and be.calls[-1] == ["A"]
    r = rz.step_jobs({"A": T.pcm("A", 1), "B": T.pcm("B", 1)})
    assert isinstance(r["A"], OwnershipError)          # main thread of A refused while its FC thread owns it
    assert not isinstance(r["B"], BaseException)       # B unaffected
    hold.set()
    t.join(5)
    assert "err" not in box, box.get("err")
    rz.close_all()


def test_close_all_restores_the_single_session_engine():
    eng = make_engine()
    m = MultiSessionEngine(eng, cuda_rng=False)
    stt = eng.wrapper.model.stt_model
    fc_before = eng.wrapper._run_fc_async_steps          # MultiSessionEngine's router, before the rendezvous
    rz = RendezvousEngine(m, FakeBackend())
    open_(rz, "A")
    rz.step_jobs({"A": T.pcm("A", 0)})
    rz.close_all()
    assert "forward" not in stt.__dict__
    assert eng.wrapper._run_fc_async_steps is fc_before
    m.close_all()
    assert solo("A", 2) == solo("A", 2)


def test_coordinator_runs_the_thinker_in_the_grad_mode_of_the_requesting_thread():
    """grad/inference mode are thread-local: a session stepping under no_grad must not get a grad-enabled Thinker
    forward on the coordinator (autograd graphs kept per step → OOM on the Ada, round 9b)."""
    _, m, be, rz = rz_engine()
    open_(rz, "A")
    open_(rz, "B")
    assert torch.is_grad_enabled()                     # the coordinator (test thread) has grad ON
    CALL_GRAD.clear()

    def nograd_step(sid, k):
        with torch.no_grad():
            return m.engine.step(T.pcm(sid, k))
    r = rz.run_round({"A": lambda: nograd_step("A", 0), "B": lambda: nograd_step("B", 0)})
    assert not any(isinstance(v, BaseException) for v in r.values())
    assert CALL_GRAD and all(g is False for g, _ in CALL_GRAD)
    CALL_GRAD.clear()

    def inf_step(sid, k):
        with torch.inference_mode():
            return m.engine.step(T.pcm(sid, k))
    rz.run_round({"A": lambda: inf_step("A", 1), "B": lambda: inf_step("B", 1)})
    assert CALL_GRAD and all(g is False and i is True for g, i in CALL_GRAD)
    rz.close_all()


def test_session_reaching_lmax_continues_b1_without_state_loss_and_close_drops_the_backend_reference():
    n = 6
    sa, sb = solo("A", n), solo("B", n)
    _, m, be, rz = rz_engine(lmax=8)                  # prefill 3 + 2 frames/chunk: both rows pass lmax at chunk 2
    open_(rz, "A")
    open_(rz, "B")
    pa, pb = [], []
    for k in range(n):
        r = rz.step_jobs({"A": T.pcm("A", k), "B": T.pcm("B", k)})
        pa.append(obs(r["A"]))
        pb.append(obs(r["B"]))
    assert pa == sa and pb == sb                      # no silent state loss across the eviction
    assert set(be.evicted) == {"A", "B"} and not be.positions()
    assert rz.stats["batch_hist"].get(2, 0) >= 2 and rz.stats["b1_calls"] > 0   # batched before, B=1 after
    rz.close_all()
    assert not be.evicted and rz.stats["evicted_closed"] == 2


def _fake_fc(rz, n):
    served, orig = [], rz._serve

    def serve(reqs):                                   # fake FC requests are recorded; real Thinker requests run
        fake = [r for r in reqs if isinstance(r, SimpleNamespace)]
        served.extend(fake)
        if len(fake) < len(reqs):
            orig([r for r in reqs if not isinstance(r, SimpleNamespace)])
    rz._serve = serve                                  # FC service itself is covered by the FC-thread test above
    for i in range(n):
        rz.fc_q.put(SimpleNamespace(i=i))
    return served


def test_fc_budget_caps_fc_service_summed_over_the_pumps_of_a_round():
    _, m, be, rz = rz_engine()
    rz.fc_merge = False                                 # the budget applies to the legacy FC service
    open_(rz, "A")
    rz.fc_budget = 2
    served = _fake_fc(rz, 5)
    rz._round_fc = {"served": 0, "forced": 0, "deferred_polls": 0, "q_start": 5, "q_max": 5, "between_before": 0}
    rz.pump()
    rz.pump()                                          # a second pump in the same round adds nothing
    assert [r.i for r in served] == [0, 1] and rz.fc_q.qsize() == 3
    rz.pump(force=True)                                # anti-deadlock escape: exactly one more, counted
    assert len(served) == 3 and rz._round_fc["forced"] == 1 and rz._round_fc["deferred_polls"] >= 1
    rz._round_fc = None
    rz.pump(until=time.monotonic() - 1)                # budgeted + round overran the tick: nothing between rounds
    assert len(served) == 3
    rz.pump(until=time.monotonic() + 0.05)             # idle time between rounds: served, FIFO order kept
    assert [r.i for r in served] == [0, 1, 2, 3, 4] and rz._between_fc == 2
    rz.close_all()


def test_fc_budget_round_accounting_and_unlimited_is_legacy():
    _, m, be, rz = rz_engine()
    rz.fc_merge = False                                 # the budget applies to the legacy FC service
    open_(rz, "A")
    served = _fake_fc(rz, 4)
    rz.fc_budget = 1
    rz.step_jobs({"A": T.pcm("A", 0)})                 # the round's own pumps are budgeted
    rf = rz.fc_rounds[-1]
    assert rf["served"] - rf["forced"] == 1 and rf["q_start"] == 4 and rf["q_end"] == 3 - rf["forced"]
    assert rz._round_fc is None
    rz.fc_budget = None                                # unlimited: one pump drains the queue, also past the tick
    rz.pump(until=time.monotonic() - 1)
    assert len(served) == 4 and rz.fc_q.empty()
    rz.close_all()


def _fc_merge_run(merge: bool):
    _, m, be, rz = rz_engine()
    rz.fc_merge = merge
    open_(rz, "A")
    open_(rz, "B")
    rz.step_jobs({"A": T.pcm("A", 0), "B": T.pcm("B", 0)})
    stream_a, cache_a = m.store.record("A").fields["engine.stream_id"], be.caches["A"]
    hold, box = threading.Event(), {}

    def fc():
        try:
            gt = torch.zeros(1, 8, dtype=torch.long)
            m.engine.wrapper._run_fc_async_steps(gen_text=gt, gen_function_text=gt, current_frame_idx=2,
                                                 stream_id=stream_a, cache=cache_a, hold=hold)
        except Exception as e:  # noqa: BLE001
            box["err"] = e
    t = threading.Thread(target=fc)
    t.start()
    deadline = time.monotonic() + 5
    while rz.fc_q.empty() and time.monotonic() < deadline:   # the FC Thinker request is queued, NOT pumped
        time.sleep(0.01)
    if not merge:
        rz.pump(until=time.monotonic() + 0.2)                  # legacy: served B=1 (step_one) by pump
    r = rz.step_jobs({"B": T.pcm("B", 1)})
    calls = [list(c) for c in be.calls]
    pos_a = be.positions()["A"]
    hold.set()
    t.join(5)
    assert "err" not in box, box.get("err")
    out = (obs(r["B"]), pos_a, calls, rz.stats.get("fc_merged", 0), rz.stats["fc_b1_calls"])
    rz.close_all()
    return out


def test_fc_merge_puts_the_queued_fc_step_in_the_rounds_batched_thinker_step():
    b_m, pos_m, calls_m, merged, fc_b1_m = _fc_merge_run(True)
    b_l, pos_l, calls_l, merged_l, fc_b1_l = _fc_merge_run(False)
    assert merged == 1 and fc_b1_m == 0 and ["A", "B"] in calls_m             # one batched step with the FC row
    assert merged_l == 0 and fc_b1_l == 1 and ["A"] in calls_l               # legacy: B=1 step_one for A
    assert b_m == b_l and pos_m == pos_l                                      # same results, same row positions
