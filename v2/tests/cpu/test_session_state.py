"""Per-session state isolation (session_state.py) against fakes. The engine under test is the REAL ReferenceEngine
(open/step/close/FC recorder/tool executor) on a fake wrapper/pipeline that mimics the upstream state layout: wrapper
turn-taking attributes mutated in place and reassigned, a context manager REPLACED at prefill, a bufferer and a state
pool reset wholesale by open_session, a tool registry read by the executor, RNG consumed every step."""

import gc
import random
import sys
import threading
import types
import weakref
from types import SimpleNamespace

import pytest
import torch

from mynah_v2.model.reference import ReferenceEngine
from mynah_v2.model.session_state import (FieldSpec, MultiSessionEngine, Poisoned, SessionStateError,
                                          SessionStateStore)
from mynah_v2.model.session_tools import SessionToolBinding
from mynah_v2.runtime.batch_contract import SessionStateStore as StoreProtocol

CHUNK = 160


# ---------------------------------------------------------------------------------------------------- plain store
class Owner:
    pass


class L(list):                                   # weak-referenceable list
    pass


def plain_store(rng=False):
    a, b = Owner(), Owner()
    a.x, a.lst, b.r = 0, [], "base"
    specs = [FieldSpec("a", "x", lambda ns: 0), FieldSpec("a", "lst", lambda ns: L()),
             FieldSpec("b", "r", lambda ns: "fresh", routed=True)]
    st = SessionStateStore({"a": a, "b": b}, specs, rng=rng, cuda_rng=False)
    st.install()
    return st, a, b


def test_store_implements_the_contract_protocol():
    st, _, _ = plain_store()
    assert isinstance(st, StoreProtocol)


def test_swap_round_trip_restores_every_registered_field_exactly():
    st, a, b = plain_store()
    st.open("A")
    st.open("B")
    st.restore("A")
    a.x, b.r = 7, "rA"
    a.lst.append(1)
    lst_a = a.lst
    snap = st.capture("A")
    assert isinstance(a.x, Poisoned) and isinstance(a.lst, Poisoned)
    st.restore("B")
    assert (a.x, a.lst, b.r) == (0, [], "fresh")
    a.x, b.r = 9, "rB"
    st.capture("B")
    st.restore("A")
    assert a.x == 7 and a.lst is lst_a and a.lst == [1] and b.r == "rA"
    assert snap["fields"] == {"a.x": 7, "a.lst": [1], "b.r": "rA"}


def test_rng_is_per_session_and_seeded():
    st, a, b = plain_store(rng=True)
    st.open("A", seed=1)
    st.open("B", seed=2)
    seq = {"A": [], "B": []}
    for sid in "ABABAB":
        st.restore(sid)
        seq[sid].append((torch.rand(()).item(), random.random()))
        st.capture(sid)
    torch.manual_seed(1)
    random.seed(1)
    solo_a = [(torch.rand(()).item(), random.random()) for _ in range(3)]
    assert seq["A"] == solo_a and seq["A"] != seq["B"]


def test_misuse_raises_instead_of_leaking():
    st, a, b = plain_store()
    st.open("A")
    with pytest.raises(SessionStateError):
        a.x + 1                                  # nothing swapped in: poisoned
    with pytest.raises(SessionStateError):
        b.r                                      # routed, no current session
    with pytest.raises(SessionStateError):
        st.open("A")                             # double open
    with pytest.raises(SessionStateError):
        st.open("Z", overrides={"a.nope": 1})    # unknown field
    with pytest.raises(SessionStateError):
        st.restore("nobody")
    st.open("B")
    st.restore("A")
    with pytest.raises(SessionStateError):
        st.restore("B")                          # A still in: capture first
    with pytest.raises(SessionStateError):
        st.capture("B")
    st.capture("A")
    st.record("B").fields["a.extra"] = 1         # corrupted record: unknown field
    with pytest.raises(SessionStateError):
        st.restore("B")
    del st.record("B").fields["a.extra"], st.record("B").fields["a.x"]   # missing field
    with pytest.raises(SessionStateError):
        st.restore("B")
    st.drop("A")
    with pytest.raises(SessionStateError, match="dropped"):
        st.restore("A")


def test_install_refuses_a_field_missing_on_its_owner():
    o = Owner()
    st = SessionStateStore({"o": o}, [FieldSpec("o", "ghost", lambda ns: 0)], rng=False)
    with pytest.raises(SessionStateError):
        st.install()


def test_uninstall_restores_owner_and_forgets_records():
    st, a, b = plain_store()
    assert type(b) is not Owner and isinstance(b, Owner)    # routed: per-instance subclass while installed
    st.open("A")
    st.restore("A")
    w = weakref.ref(a.lst)
    st.uninstall()
    assert a.x == 0 and a.lst == [] and b.r == "base" and type(b) is Owner
    gc.collect()
    assert w() is None
    with pytest.raises(SessionStateError):
        st.restore("A")


# ---------------------------------------------------------------------------------------------------- fake upstream
@pytest.fixture(autouse=True)
def fake_nemo(monkeypatch):
    """The two nemo classes ReferenceEngine.open/_frame import."""
    def mod(name, **attrs):
        m = types.ModuleType(name)
        m.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, m)

    class Frame(SimpleNamespace):
        pass

    class S2SRequestOptions(SimpleNamespace):
        pass
    for n in ("nemo", "nemo.collections", "nemo.collections.asr", "nemo.collections.asr.inference",
              "nemo.collections.asr.inference.streaming", "nemo.collections.asr.inference.streaming.framing",
              "nemo.collections.speechlm2", "nemo.collections.speechlm2.inference",
              "nemo.collections.speechlm2.inference.streaming",
              "nemo.collections.speechlm2.inference.streaming.framing"):
        mod(n)
    mod("nemo.collections.asr.inference.streaming.framing.request", Frame=Frame)
    mod("nemo.collections.speechlm2.inference.streaming.framing.s2s_request_options",
        S2SRequestOptions=S2SRequestOptions)


class FakeCM:
    live = weakref.WeakSet()

    def __init__(self, s2s_model, num_slots, max_len):
        self.max_len = max_len
        self.reset()
        FakeCM.live.add(self)

    def reset(self):
        self.streamidx2slotidx, self.slot_contexts = {}, [None]

    def get_context(self, ids):
        if ids[0] not in self.streamidx2slotidx:
            self.streamidx2slotidx[ids[0]] = 0
            self.slot_contexts[0] = SimpleNamespace(frame_idx=0, prompt=0, dynamic_cache=[],
                                                    gen_text=torch.zeros(1, self.max_len, dtype=torch.long),
                                                    gen_function_text=torch.zeros(1, self.max_len, dtype=torch.long))
        return self.slot_contexts[0], {}


class FakeBufferer:
    def __init__(self, sample_rate, buffer_size_in_secs):
        self.bufs = {}

    def reset(self):
        self.bufs = {}

    def update(self, frames):
        b = self.bufs.setdefault(frames[0].stream_id, [])
        b.append(float(frames[0].samples.sum()))
        return [sum(b[-3:])], [0]


class FakeState:
    def __init__(self):
        self.audio_buffer, self.text = torch.zeros(0), ""

    def get_output_text(self):
        return self.text

    def get_output_asr_text(self):
        return ""

    def get_output_function_text(self):
        return ""

    def cleanup_after_response(self):
        self.audio_buffer = torch.zeros(0)


class FakeWrapper:
    def __init__(self):
        self._post_tc_bos_exempt, self._redirect_tokens_queue = False, []
        self._agent_eos_just_fired, self._agent_idle, self._last_user_audio_emb = False, True, None

    def _run_fc_async_steps(self, **kw):           # what the FC thread does to wrapper state
        self._redirect_tokens_queue.append(99)
        self._agent_eos_just_fired = True
        return (1, None, None, None, None)


class FakePipeline:
    def __init__(self, w):
        self.s2s_model, self.batch_size, self.max_len = w, 1, 64
        self.input_sample_rate, self.buffer_size_in_secs, self.chunk_size_in_secs = 16000, 0.5, CHUNK / 16000
        self.bufferer = FakeBufferer(16000, 0.5)
        self.context_manager = FakeCM(w, 1, 64)
        self._stream_has_prompt, self._state_pool, self._fc_async_bg = False, {}, {}
        self.tool_registry = {"builtin": lambda args: "B"}
        w._streaming_state_getter = self.get_or_create_state

    def get_or_create_state(self, sid):
        if sid not in self._state_pool:
            self._state_pool[sid] = FakeState()
        return self._state_pool[sid]

    def reset_session(self):
        self.bufferer.reset()
        self.context_manager.reset()
        self._state_pool.clear()
    open_session = close_session = reset_session

    def _execute_tool_call(self, call_text):
        fn = self.tool_registry.get(call_text)
        return fn({}) if fn else None

    def generate_step(self, frames):
        f, w = frames[0], self.s2s_model
        if f.is_first and f.samples.numel() == 0:         # prefill: NEW context manager (P:2221)
            self.context_manager = FakeCM(w, 1, self.max_len)
            ctx, _ = self.context_manager.get_context([f.stream_id])
            ctx.prompt = len(f.options.system_prompt)
            w._redirect_tokens_queue, w._post_tc_bos_exempt, w._agent_eos_just_fired = [], False, False
            self._stream_has_prompt = True
            return
        bufs, _ = self.bufferer.update(frames)
        if f.is_first and not self._stream_has_prompt:
            self.context_manager = FakeCM(w, 1, self.max_len)
        self._stream_has_prompt = False
        ctx, _ = self.context_manager.get_context([f.stream_id])
        q = w._redirect_tokens_queue
        noise = int(torch.rand(()).item() * 97)                 # RNG (EarTTS / sampling stand-in)
        tok = 1 + (int(bufs[0]) * 31 + ctx.prompt * 7 + len(q) * 13 + ctx.frame_idx * 3 + noise
                   + 5 * int(w._post_tc_bos_exempt) + getattr(w, "_tts_in_turn_content", 0)) % 50
        if tok % 3 == 0:
            q.append(tok)                                       # in-place mutation
        elif q and tok % 3 == 1:
            q.pop(0)
        w._post_tc_bos_exempt = tok % 2 == 0                    # reassignment
        w._tts_in_turn_content = getattr(w, "_tts_in_turn_content", 0) + 1
        w._last_user_audio_emb = torch.tensor([bufs[0]])
        ctx.gen_text[0, ctx.frame_idx] = tok
        ctx.frame_idx += 1
        ctx.dynamic_cache.append(tok)                           # "Thinker state", reached by reference only
        st = self.get_or_create_state(f.stream_id)
        st.text += f" {tok}"
        st.audio_buffer = torch.rand(4)


def make_engine():
    w = FakeWrapper()
    p = FakePipeline(w)
    eng = object.__new__(ReferenceEngine)
    eng.pipeline, eng.wrapper, eng.tokenizer = p, w, None
    eng.session_tools = SessionToolBinding(p.tool_registry)
    eng.fc_still_running, eng.fc_records, eng._fc_lock = 0, [], threading.Lock()
    eng._instrument_fc()
    eng.unknown_tool_handler = eng.multi_call_handler = None
    eng.unknown_tool_events, eng.multi_call_events = [], []
    eng._wrap_executor()
    eng.chunk_samples, eng.stream_id, eng._started = CHUNK, 0, False
    eng._pos, eng._last_frame_idx, eng._rec_cursor, eng.llm_cache_info = {"text": 0, "asr": 0, "func": 0}, 0, 0, None
    return eng


SESS = {"A": ("prompt a", 0.01, 11), "B": ("prompt bb", 0.02, 22), "C": ("prompt ccc", 0.03, 33)}


def pcm(sid, k):
    return torch.full((CHUNK,), SESS[sid][1] * (k + 1))


def observe(eng, out):
    w = eng.wrapper
    return (out.new_text_ids, out.text_delta, out.frame_idx, out.audio.tolist(), list(w._redirect_tokens_queue),
            w._post_tc_bos_exempt, w._tts_in_turn_content, w._last_user_audio_emb.tolist())


def solo(sid, n):
    """Single-session path, untouched: plain ReferenceEngine, global seed at open."""
    eng = make_engine()
    torch.manual_seed(SESS[sid][2])
    random.seed(SESS[sid][2])
    eng.open(SESS[sid][0])
    res = [observe(eng, eng.step(pcm(sid, k))) for k in range(n)]
    eng.close()
    return res


def test_one_session_through_the_multi_engine_equals_the_single_path():
    eng = make_engine()
    m = MultiSessionEngine(eng, cuda_rng=False)
    m.open("A", SESS["A"][0], seed=SESS["A"][2])
    got = [observe(eng, m.step("A", pcm("A", k))) for k in range(6)]
    m.close_all()
    assert got == solo("A", 6)


def test_interleaving_ABAB_equals_each_session_alone():
    eng = make_engine()
    m = MultiSessionEngine(eng, cuda_rng=False)
    for s in "AB":
        m.open(s, SESS[s][0], seed=SESS[s][2])
    got = {"A": [], "B": []}
    order = "ABABBAAB" + "BA"
    k = {"A": 0, "B": 0}
    for s in order:
        got[s].append(observe(eng, m.step(s, pcm(s, k[s]))))
        k[s] += 1
    m.close_all()
    assert got["A"] == solo("A", 5) and got["B"] == solo("B", 5)
    assert got["A"] != got["B"]


def _record_view(m, sid):
    st = m.store
    cm = st.get(sid, "pipeline.context_manager")
    ctx = cm.slot_contexts[0]
    return (st.get(sid, "engine._pos"), st.get(sid, "engine._last_frame_idx"),
            list(st.record(sid).fields["wrapper._redirect_tokens_queue"]),
            st.record(sid).fields["wrapper._post_tc_bos_exempt"], ctx.frame_idx, ctx.gen_text.clone(),
            list(ctx.dynamic_cache), st.record(sid).rng["cpu"].clone() if st.record(sid).rng else None)


def _same(x, y):
    return all(torch.equal(a, b) if isinstance(a, torch.Tensor) else a == b for a, b in zip(x, y))


def test_slot_reuse_sequence_AB_then_BC_then_AC_matches_solo_and_leaves_no_trace():
    eng = make_engine()
    m = MultiSessionEngine(eng, cuda_rng=False)
    got = {s: [] for s in "ABC"}
    k = {s: 0 for s in "ABC"}

    def run(order):
        for s in order:
            got[s].append(observe(eng, m.step(s, pcm(s, k[s]))))
            k[s] += 1

    m.open("A", SESS["A"][0], seed=SESS["A"][2])
    m.open("B", SESS["B"][0], seed=SESS["B"][2])
    run("ABAB")
    cm_a = weakref.ref(m.store.get("A", "pipeline.context_manager"))
    m.close("A")                                          # drop A
    with pytest.raises(SessionStateError, match="dropped"):
        m.store.restore("A")
    with pytest.raises(SessionStateError):
        m.step("A", pcm("A", 0))
    m.open("C", SESS["C"][0], seed=SESS["C"][2])
    run("C")
    before_b = _record_view(m, "B")
    run("CC")                                             # B untouched while only C steps (I1 analog)
    assert _same(before_b, _record_view(m, "B"))
    run("BCB")
    gc.collect()
    assert cm_a() is None                                 # nothing keeps the dropped session alive
    first_a = got["A"]
    got["A"], k["A"] = [], 0
    m.open("A", SESS["A"][0], seed=SESS["A"][2])          # re-open A fresh
    m.close("B")
    run("ACAACA")
    m.close_all()
    assert first_a == solo("A", 2)
    assert got["A"] == solo("A", 4)
    assert got["B"] == solo("B", 4)
    assert got["C"] == solo("C", 6)


def test_teardown_restores_the_single_session_engine():
    eng = make_engine()
    w, p = eng.wrapper, eng.pipeline
    cls_w, cls_p, cls_e = type(w), type(p), type(eng)
    orig = {"cm": p.context_manager, "pool": p._state_pool, "reg": p.tool_registry, "tools": eng.session_tools,
            "fc": w._run_fc_async_steps, "q": w._redirect_tokens_queue, "recs": eng.fc_records}
    m = MultiSessionEngine(eng, cuda_rng=False)
    m.open("A", "x", seed=1)
    m.open("B", "y", seed=2)
    m.step("A", pcm("A", 0))
    m.step("B", pcm("B", 0))
    refs = [weakref.ref(m.store.get(s, "pipeline.context_manager")) for s in "AB"]
    m.close_all()
    del m
    gc.collect()
    assert all(r() is None for r in refs)
    assert (type(w), type(p), type(eng)) == (cls_w, cls_p, cls_e)
    assert p.context_manager is orig["cm"] and p._state_pool is orig["pool"] and p.tool_registry is orig["reg"]
    assert eng.session_tools is orig["tools"] and w._run_fc_async_steps is orig["fc"]
    assert w._redirect_tokens_queue is orig["q"] and eng.fc_records is orig["recs"] and eng.stream_id == 0
    eng.open("solo")                                     # the single path works again
    eng.step(pcm("A", 0))
    eng.close()


def test_fc_thread_of_A_writes_only_to_A_while_B_is_stepped():
    eng = make_engine()
    m = MultiSessionEngine(eng, cuda_rng=False)
    calls = []
    m.open("A", "pa", tools={"lookup": lambda args: calls.append("A") or "ra"}, seed=1)
    m.open("B", "pb", tools={"lookup": lambda args: calls.append("B") or "rb"}, seed=2)
    stream_a = m.store.record("A").fields["engine.stream_id"]
    q_b_before = list(m.store.record("B").fields["wrapper._redirect_tokens_queue"])
    go, done, box = threading.Event(), threading.Event(), {}

    def fc_thread():                                   # mimics _fc_async_thread_fn of stream A
        try:
            gt = torch.zeros(1, 8, dtype=torch.long)
            eng.wrapper._run_fc_async_steps(gen_text=gt, gen_function_text=gt, current_frame_idx=2,
                                            stream_id=stream_a)
            go.wait(5)
            box["tool"] = eng.pipeline._execute_tool_call("lookup")
            eng.pipeline.get_or_create_state(stream_a).text += " FC"
            eng.wrapper._post_tc_bos_exempt = True
        except Exception as e:  # noqa: BLE001
            box["err"] = e
        finally:
            done.set()

    m.step("B", pcm("B", 0))                           # B swapped in on the driver
    t = threading.Thread(target=fc_thread)
    t.start()
    m.step("B", pcm("B", 1))
    go.set()
    done.wait(5)
    t.join(5)
    assert "err" not in box, box.get("err")
    assert box["tool"] == "ra" and calls == ["A"]
    rec_a, rec_b = m.store.record("A").fields, m.store.record("B").fields
    assert rec_a["wrapper._redirect_tokens_queue"][-1] == 99 and rec_a["wrapper._agent_eos_just_fired"] is True
    assert rec_a["wrapper._post_tc_bos_exempt"] is True
    assert rec_a["pipeline._state_pool"][stream_a].text.endswith(" FC")
    assert 99 not in rec_b["wrapper._redirect_tokens_queue"] and 99 not in q_b_before  # fake tokens are 1..50
    assert rec_b["wrapper._agent_eos_just_fired"] is False
    assert len(m.store.get("A", "engine.fc_records")) == 1 and m.store.get("B", "engine.fc_records") == []
    out_a = m.step("A", pcm("A", 0))
    assert len(out_a.fc_records) == 1 and out_a.fc_records[0].start_frame == 2
    m.close_all()


def test_fc_call_for_an_unknown_or_dropped_stream_raises():
    eng = make_engine()
    m = MultiSessionEngine(eng, cuda_rng=False)
    m.open("A", "pa", seed=1)
    sa = m.store.record("A").fields["engine.stream_id"]
    with pytest.raises(SessionStateError):
        eng.wrapper._run_fc_async_steps(stream_id=12345)
    m.close("A")
    box = {}
    t = threading.Thread(target=lambda: box.setdefault("r", _try(lambda: eng.wrapper._run_fc_async_steps(
        stream_id=sa))))
    t.start()
    t.join(5)
    assert isinstance(box["r"], SessionStateError)
    m.close_all()


def _try(f):
    try:
        return f()
    except Exception as e:  # noqa: BLE001
        return e


def test_tool_executed_in_a_fresh_pool_thread_uses_the_owning_sessions_tools():
    """Upstream runs the tool in a NEW ThreadPoolExecutor worker (unbound thread) with the executor snapshotted when
    the FC thread was spawned (in the session's own step). While B is swapped in on the driver, A's call must still
    run A's tools (regression: round 11/13 PG-p1 logged A's call in B)."""
    import concurrent.futures
    eng = make_engine()
    m = MultiSessionEngine(eng, cuda_rng=False)
    calls = []
    m.open("A", "pa", tools={"lookup": lambda args: calls.append("A") or "ra"}, seed=1)
    m.open("B", "pb", tools={"lookup": lambda args: calls.append("B") or "rb"}, seed=2)
    m.step("A", pcm("A", 0))
    m.store.bind_thread("A")                       # the spawn happens in A's own (bound) thread
    snap = eng.pipeline._execute_tool_call         # pipeline :1603  _execute_tool_fn_snap = self._execute_tool_call
    m.store.unbind_thread()
    m.step("B", pcm("B", 0))                       # B swapped in on the driver when the tool finally runs
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        res = ex.submit(snap, "lookup").result(5)
        unbound_after = ex.submit(lambda: getattr(m.store._tl, "sid", None)).result(5)
    assert res == "ra" and calls == ["A"]
    assert unbound_after is None                   # the worker's binding is restored after the call
    m.close_all()


def _tools(tag, calls, gate=None):
    def lookup(args):
        if gate is not None:
            gate.wait(5)
        calls.append(tag)
        return "r" + tag
    return {"lookup": lookup}


def _snapshot(m, sid, eng):
    """The executor as the pipeline captures it at FC spawn, in the session's own (bound) thread."""
    m.store.bind_thread(sid)
    try:
        return eng.pipeline._execute_tool_call
    finally:
        m.store.unbind_thread()


def test_overlapping_tool_calls_of_two_sessions_each_use_their_own_tools_while_the_driver_switches():
    import concurrent.futures
    eng = make_engine()
    m = MultiSessionEngine(eng, cuda_rng=False)
    calls, gate = [], threading.Event()
    m.open("A", "pa", tools=_tools("A", calls, gate), seed=1)
    m.open("B", "pb", tools=_tools("B", calls, gate), seed=2)
    sa, sb = _snapshot(m, "A", eng), _snapshot(m, "B", eng)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        fa, fb = ex.submit(sa, "lookup"), ex.submit(sb, "lookup")   # both pending at once
        for k in range(3):                                            # driver alternates the swapped-in session
            m.step("A", pcm("A", k))
            m.step("B", pcm("B", k))
        gate.set()
        assert (fa.result(5), fb.result(5)) == ("rA", "rB")
    assert sorted(calls) == ["A", "B"]
    m.close_all()


def test_late_tool_callback_of_a_closed_session_is_refused_even_if_the_sid_or_slot_is_reused():
    import concurrent.futures
    from mynah_v2.model.session_state import StaleSessionError
    eng = make_engine()
    m = MultiSessionEngine(eng, cuda_rng=False)
    calls = []
    m.open("A", "pa", tools=_tools("A", calls), seed=1)
    m.open("B", "pb", tools=_tools("B", calls), seed=2)
    stale = _snapshot(m, "A", eng)                  # A spawned an FC, its tool is not run yet
    m.close("A")
    m.open("C", "pc", tools=_tools("C", calls), seed=3)   # C may take A's slot
    m.open("A", "pa2", tools=_tools("A2", calls), seed=4)  # and the sid "A" itself is reused by a NEW session
    m.step("B", pcm("B", 0))
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        with pytest.raises(StaleSessionError):
            ex.submit(stale, "lookup").result(5)
        assert ex.submit(_snapshot(m, "A", eng), "lookup").result(5) == "rA2"   # the new A works normally
    assert calls == ["A2"]                           # the stale callback ran nobody's tool
    m.close_all()
