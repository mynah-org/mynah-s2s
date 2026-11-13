"""Batched cache-aware perception (model/perception_batch.py) and its rendezvous integration, on CPU fakes.

The fake encoder is row-independent and recurrent in its per-row caches (channel / time / channel_len, batch dims 1, 1,
0 like NeMo's ConformerEncoder), the fake preprocessor is row-wise and length-dependent: any stacking / unstacking /
grouping / cache hand-back error changes the outputs. The real-model equivalence (FP32 kernels at another M) is a GPU
check (scripts/perception_batch_check.py)."""

import importlib
import random
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
R = importlib.import_module("test_rendezvous")
T = R.T
fake_nemo = T.fake_nemo                                  # autouse fixture (nemo stand-ins)

from mynah_v2.model.perception_batch import BatchedPerception, step_kwargs  # noqa: E402
from mynah_v2.model.reference import ReferenceEngine  # noqa: E402
from mynah_v2.model.session_state import MultiSessionEngine  # noqa: E402
from mynah_v2.model.session_tools import SessionToolBinding  # noqa: E402
from mynah_v2.runtime.rendezvous import RendezvousEngine  # noqa: E402

F, L, C, D, K = 3, 2, 3, 4, 2
ENC_CALLS: list = []


@dataclass
class PCState:
    cache_last_channel: Optional[torch.Tensor] = None
    cache_last_time: Optional[torch.Tensor] = None
    cache_last_channel_len: Optional[torch.Tensor] = None

    def is_initialized(self):
        return self.cache_last_channel is not None


class FakeEncoder:
    def __init__(self, lookahead=1):
        self._cfg = SimpleNamespace(att_context_size=[70, lookahead])

    def get_initial_cache_state(self, batch_size=1):
        return (torch.zeros(L, batch_size, C, D, dtype=torch.float64),
                torch.zeros(L, batch_size, D, K, dtype=torch.float64), torch.zeros(batch_size, dtype=torch.long))

    def cache_aware_stream_step(self, processed_signal, processed_signal_length, cache_last_channel, cache_last_time,
                                cache_last_channel_len, keep_all_outputs, drop_extra_pre_encoded):
        ENC_CALLS.append((threading.current_thread(), processed_signal.shape[0]))
        assert keep_all_outputs and processed_signal_length.tolist() == [processed_signal.shape[-1]] * processed_signal.shape[0]
        x = processed_signal.double()
        pos = torch.arange(1, x.shape[-1] + 1, dtype=x.dtype)
        # per-row reductions (a batched reduction may sum in another order: the fake must be exactly row-independent)
        feats = torch.stack([(x[b] * pos).sum() for b in range(x.shape[0])])               # [B], position-weighted
        hist = torch.stack([cache_last_channel[:, b].sum() + 0.5 * cache_last_time[:, b].sum()
                            for b in range(x.shape[0])])                                    # [B]
        v = feats + hist * 1.3 + cache_last_channel_len.double() * 0.7 + drop_extra_pre_encoded
        enc = v[:, None, None].expand(-1, D, 2) * torch.tensor([1.0, 2.0], dtype=x.dtype)    # [B, D, T=2]
        ch = torch.roll(cache_last_channel, 1, dims=2).clone()
        ch[:, :, 0, :] = v[None, :, None] * 0.01
        tm = cache_last_time * 0.5 + v[None, :, None, None] * 0.001
        ln = torch.clamp(cache_last_channel_len + 1, max=C)
        return enc, torch.full((x.shape[0],), 2, dtype=torch.long), ch, tm, ln


class FakePerception:
    def __init__(self, lookahead=1):
        self.encoder = FakeEncoder(lookahead)

    def modality_adapter(self, audio_signal, length):
        return audio_signal * 2.0 + 1.0, length

    def proj(self, x):
        return x * 3.0 - 0.25


def fake_pre(input_signal, length):
    """Row-wise mel stand-in: [B, N] -> [B, F, N // 10] (sum of 10-sample windows, scaled per feature)."""
    assert input_signal.dtype == torch.float32 and length.tolist() == [input_signal.shape[1]] * input_signal.shape[0]
    b, n = input_signal.shape
    w = input_signal[:, : (n // 10) * 10].double().reshape(b, n // 10, 10).sum(-1)   # exact (bf16 inputs)
    return torch.stack([w * (f + 1) for f in range(F)], dim=1).float(), None          # FP32 like NeMo


class FakeMgr:
    def __init__(self, lookahead=1, pad_and_drop=False):
        self.model = SimpleNamespace(stt_model=SimpleNamespace(perception=FakePerception(lookahead)))
        self.preprocessor = fake_pre
        self.streaming_cfg = SimpleNamespace(chunk_size=[9, 16], shift_size=[9, 16], pre_encode_cache_size=[0, 9],
                                             drop_extra_pre_encoded=2)
        self.input_features, self.device, self.pad_and_drop_preencoded = F, "cpu", pad_and_drop
        self._solo = None

    def get_initial_state(self, batch_size=1):
        return PCState(*self.model.stt_model.perception.encoder.get_initial_cache_state(batch_size))

    def step(self, audio_input, frame_idx, num_frames_per_chunk, perception_cache):
        """B=1 reference: the same code path with one row (stands for the upstream B=1 graph step)."""
        if self._solo is None:
            self._solo = BatchedPerception(self)
        return self._solo.run([dict(audio_input=audio_input, frame_idx=frame_idx,
                                    num_frames_per_chunk=num_frames_per_chunk, perception_cache=perception_cache)])[0]


def audio(seed, n):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, n, generator=g).to(torch.bfloat16)


def close(x, y):
    """Exact up to CPU float64 reduction-order ULPs (a batched slice can be summed in another SIMD order); a row or
    cache mix-up is orders of magnitude larger. Shapes and integer tensors must match exactly."""
    assert x.shape == y.shape and x.dtype == y.dtype
    return torch.equal(x, y) if not x.is_floating_point() else torch.allclose(x, y, rtol=1e-12, atol=1e-9)


def same(a, b):
    ea, ca, ra = a
    eb, cb, rb = b
    assert close(ea, eb) and close(ra, rb)
    for f in ("cache_last_channel", "cache_last_time", "cache_last_channel_len"):
        assert close(getattr(ca, f), getattr(cb, f)), f


def call(mgr, seed, n, frame_idx, nf=2, pc=None):
    return dict(audio_input=audio(seed, n), frame_idx=frame_idx, num_frames_per_chunk=nf,
                perception_cache=pc if pc is not None else mgr.get_initial_state())


def test_batch_dims_are_found_from_the_initial_cache_shapes():
    assert BatchedPerception(FakeMgr()).bdims == (1, 1, 0)


@pytest.mark.parametrize("lookahead", [1, 0])          # 1 sub-step / 2 sub-steps per 2-frame chunk
def test_mixed_rows_first_and_subsequent_and_different_buffer_lengths_equal_per_row_runs(lookahead):
    mgr = FakeMgr(lookahead)
    bp = BatchedPerception(mgr)
    calls = [call(mgr, 1, 160, 0), call(mgr, 2, 320, 2), call(mgr, 3, 400, 10), call(mgr, 4, 400, 24)]
    got = bp.run(calls)
    for c, g in zip(calls, got):
        same(g, mgr.step(**c))
    assert bp.stats["mel_per_row"] == 4 and bp.stats["mel_batched"] == 0
    assert bp.stats["hist"] == {4: 1}


def test_equal_buffers_use_one_stacked_mel_call_and_still_match():
    mgr = FakeMgr()
    bp = BatchedPerception(mgr)
    calls = [call(mgr, s, 400, 30) for s in (5, 6, 7)]
    got = bp.run(calls)
    for c, g in zip(calls, got):
        same(g, mgr.step(**c))
    assert bp.stats["mel_batched"] == 1 and bp.stats["groups"] == 1
    ENC_CALLS.clear()
    BatchedPerception(mgr, batch_mel=False).run(calls)
    assert [n for _, n in ENC_CALLS] == [3]              # per-row mel, still one encoder call


def test_cache_continuity_over_steps_with_a_late_joining_row():
    mgr = FakeMgr()
    bp = BatchedPerception(mgr)
    solo = {s: mgr.get_initial_state() for s in "ABC"}
    bat = {s: mgr.get_initial_state() for s in "ABC"}
    buf = {s: torch.zeros(1, 0, dtype=torch.bfloat16) for s in "ABC"}
    start = {"A": 0, "B": 0, "C": 3}                     # C joins at step 3: its first chunk batched with others' later
    for k in range(8):
        live = [s for s in "ABC" if k >= start[s]]
        kws = {}
        for s in live:
            buf[s] = torch.cat([buf[s], audio(100 * k + ord(s), 160)], dim=1)[:, -400:]
            kws[s] = dict(audio_input=buf[s], frame_idx=2 * (k - start[s]), num_frames_per_chunk=2)
        ref = {s: mgr.step(perception_cache=solo[s], **kws[s]) for s in live}
        got = bp.run([dict(perception_cache=bat[s], **kws[s]) for s in live])
        for s, g in zip(live, got):
            same(g, ref[s])
            solo[s], bat[s] = ref[s][1], g[1]


def test_rows_with_and_without_caches_cannot_be_mixed_and_frames_per_chunk_must_match():
    mgr = FakeMgr()
    bp = BatchedPerception(mgr)
    with pytest.raises(ValueError):
        bp.run([call(mgr, 1, 400, 4), call(mgr, 2, 400, 4, nf=4)])
    bad = call(mgr, 2, 400, 4)
    bad["perception_cache"] = PCState(None, bad["perception_cache"].cache_last_time,
                                      bad["perception_cache"].cache_last_channel_len)
    with pytest.raises(RuntimeError):
        bp.run([call(mgr, 1, 400, 4), bad])


def test_step_kwargs_normalizes_positional_and_keyword_calls():
    pc = object()
    assert step_kwargs((1, 2), {"num_frames_per_chunk": 3, "perception_cache": pc}) == dict(
        audio_input=1, frame_idx=2, num_frames_per_chunk=3, perception_cache=pc)
    assert step_kwargs((1,), {"audio_input": 1, "frame_idx": 2, "num_frames_per_chunk": 3, "perception_cache": pc}) is None
    assert step_kwargs((), {"audio_input": 1}) is None
    assert step_kwargs((), {"audio_input": 1, "frame_idx": 2, "num_frames_per_chunk": 3, "perception_cache": pc,
                            "x": 1}) is None


# ------------------------------------------------------------------------------------------------ rendezvous
class PWrapper(R.Wrapper):
    def __init__(self):
        super().__init__()
        self.perception_cache_mgr = FakeMgr()


class PPipeline(R.Pipeline):
    """Perception (cache-aware, per-session buffer + cache in the swapped context) before the Thinker frames; its
    output feeds the session's text, so a wrong row / cache shows up in the outputs."""

    def generate_step(self, frames):
        f, w = frames[0], self.s2s_model
        if not (f.is_first and f.samples.numel() == 0) and float(f.samples[0]) >= 0:
            ctx, _ = self.context_manager.get_context([f.stream_id])
            buf = torch.cat([getattr(ctx, "abuf", torch.zeros(1, 0)), f.samples.reshape(1, -1).float()], 1)[:, -400:]
            ctx.abuf = buf
            pc = getattr(ctx, "pc", None) or w.perception_cache_mgr.get_initial_state()
            enc, pc, raw = w.perception_cache_mgr.step(audio_input=buf, frame_idx=2 * ctx.frame_idx,
                                                       num_frames_per_chunk=2, perception_cache=pc)
            ctx.pc = pc
            self.get_or_create_state(f.stream_id).text += f" P{float(enc.sum()):.6f}/{float(raw.sum()):.6f}"
        super().generate_step(frames)


def make_engine():
    w = PWrapper()
    p = PPipeline(w)
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


def pcm(sid, k):
    g = torch.Generator().manual_seed(1000 * ord(sid) + k)
    return torch.rand(T.CHUNK, generator=g) * 0.1 + 0.001


def obs(out):
    return (out.new_text_ids, out.text_delta, out.frame_idx, out.audio.tolist())


def solo(sid, n):
    eng = make_engine()
    torch.manual_seed(T.SESS[sid][2])
    random.seed(T.SESS[sid][2])
    eng.open(T.SESS[sid][0])
    res = [obs(eng.step(pcm(sid, k))) for k in range(n)]
    eng.close()
    return res


def rz_engine(batch_perception=True):
    eng = make_engine()
    m = MultiSessionEngine(eng, cuda_rng=False)
    be = R.FakeBackend(3)
    return eng, m, be, RendezvousEngine(m, be, step_timeout_s=10.0, batch_perception=batch_perception)


def test_rendezvous_batches_perception_of_all_ready_sessions_and_outputs_equal_solo():
    n = 5
    ref = {s: solo(s, n) for s in "ABC"}
    assert all("P" in r[0][1] for r in ref.values())     # perception output reaches the text
    eng, m, be, rz = rz_engine()
    for s in "ABC":
        rz.open(s, T.SESS[s][0], seed=T.SESS[s][2])
    ENC_CALLS.clear()
    got = {s: [] for s in "ABC"}
    for k in range(n):
        r = rz.step_jobs({s: pcm(s, k) for s in "ABC"})
        for s in "ABC":
            got[s].append(obs(r[s]))
    assert got == ref
    ps = rz.stats["perception"]
    assert ps["batch_hist"] == {3: n} and ps["b1"] == 0 and ps["passthrough"] == 0
    assert rz.stats["batch_hist"] == {3: 2 * n}            # Thinker frames still batched A+B+C
    assert all(t is threading.main_thread() and b == 3 for t, b in ENC_CALLS)   # one batched encoder call, coordinator
    assert all(j.get("perc", 0) == 1 for j in rz.job_log[-3:])
    rz.close_all()
    assert "step" not in eng.wrapper.perception_cache_mgr.__dict__


def test_opportunistic_participation_lone_session_runs_the_installed_b1_step_and_late_session_joins():
    ref = {s: solo(s, 4) for s in "AB"}
    eng, m, be, rz = rz_engine()
    pcm_mgr = eng.wrapper.perception_cache_mgr
    inner_calls = []
    orig = rz._pcm_inner
    rz._pcm_inner = lambda *a, **k: (inner_calls.append(threading.current_thread()), orig(*a, **k))[1]
    rz.open("A", T.SESS["A"][0], seed=T.SESS["A"][2])
    got = {"A": [], "B": []}
    for k in range(2):                                    # A alone: B=1 through the installed step
        got["A"].append(obs(rz.step_jobs({"A": pcm("A", k)})["A"]))
    rz.open("B", T.SESS["B"][0], seed=T.SESS["B"][2])
    for k in range(2):                                    # B joins: its first chunk batched with A's third
        r = rz.step_jobs({"A": pcm("A", k + 2), "B": pcm("B", k)})
        got["A"].append(obs(r["A"]))
        got["B"].append(obs(r["B"]))
    for k in range(2, 4):                                 # A done: B alone again
        got["B"].append(obs(rz.step_jobs({"B": pcm("B", k)})["B"]))
    assert got == ref
    ps = rz.stats["perception"]
    assert ps["b1"] == 4 and ps["batch_hist"] == {2: 2}
    assert len(inner_calls) == 4 and all(t is threading.main_thread() for t in inner_calls)
    # a call from a thread that is not a session worker passes through unchanged
    out = pcm_mgr.step(audio_input=torch.ones(1, 320), frame_idx=0, num_frames_per_chunk=2,
                       perception_cache=pcm_mgr.get_initial_state())
    assert ps["passthrough"] == 1 and out[0].shape[0] == 1
    rz.close_all()


def test_failing_batched_perception_fails_its_members_only_and_default_off_never_touches_the_manager():
    eng, m, be, rz = rz_engine(batch_perception=False)
    assert "step" not in eng.wrapper.perception_cache_mgr.__dict__ and rz.pbatch is None
    rz.close_all()
    eng, m, be, rz = rz_engine()
    for s in "ABC":
        rz.open(s, T.SESS[s][0], seed=T.SESS[s][2])
    rz.step_jobs({s: pcm(s, 0) for s in "ABC"})

    def boom(calls):
        raise RuntimeError("batched perception failure")
    rz.pbatch.run = boom
    r = rz.step_jobs({"A": pcm("A", 1), "B": pcm("B", 1), "C": torch.full((T.CHUNK,), -1.0)})
    assert isinstance(r["A"], RuntimeError) and isinstance(r["B"], RuntimeError)   # A+B batched together → both fail
    assert isinstance(r["C"], RuntimeError) and "injected" in str(r["C"])           # C failed before perception
    assert set(rz.failed) == {"A", "B", "C"}
    rz.close_all()


# ------------------------------------------------------------------------------------------------ graphs per row count
def fake_capture(fn):
    """CUDA-graph stand-in with the same buffer semantics: outputs are STATIC tensors overwritten by every replay,
    inputs are read from the static buffers the closure captured."""
    out = tuple(None if t is None else t.clone() for t in fn())

    def replay():
        for s_, n_ in zip(out, fn()):
            if s_ is not None:
                s_.copy_(n_)
    return replay, out


def gbp(mgr, max_rows=4):
    bp = BatchedPerception(mgr, graphs=True, capture=fake_capture)
    bp.precapture(max_rows)
    return bp


def test_precapture_covers_the_steady_key_for_each_row_count_and_nothing_else():
    mgr = FakeMgr()
    bp = gbp(mgr, 4)
    assert bp.steady_key() == (False, 16 + 9, 2)
    assert sorted(bp.graphs) == [(n, False, 25, 2) for n in (2, 3, 4)]
    assert BatchedPerception(mgr).precapture(4) == {} and not BatchedPerception(mgr).graphs   # eager mode: no capture


def test_graph_replays_match_per_row_runs_and_outputs_survive_the_next_replay():
    mgr = FakeMgr()
    bp = gbp(mgr)
    st = {s: mgr.get_initial_state() for s in "ABC"}
    buf = {s: torch.zeros(1, 0, dtype=torch.bfloat16) for s in "ABC"}
    ref_st = dict(st)
    kept = []
    for k in range(6):
        kws = {}
        for s in "ABC":
            buf[s] = torch.cat([buf[s], audio(10 * k + ord(s), 160)], 1)[:, -400:]
            kws[s] = dict(audio_input=buf[s], frame_idx=2 * k, num_frames_per_chunk=2)
        ref = {s: mgr.step(perception_cache=ref_st[s], **kws[s]) for s in "ABC"}
        got = bp.run([dict(perception_cache=st[s], **kws[s]) for s in "ABC"])
        for s, g in zip("ABC", got):
            same(g, ref[s])
            st[s], ref_st[s] = g[1], ref[s][1]
        kept.append((got, ref))
    for got, ref in kept:                                   # nothing returned aliases the static graph buffers
        for s, g in zip("ABC", got):
            same(g, ref[s])
    assert bp.stats["graph_groups"] == 5 and bp.stats["eager_groups"] == 1      # chunk 0 = first-chunk key → eager
    assert bp.stats["eager_keys"] == {str((3, True, 9, 0)): 1}


def test_row_count_without_a_graph_runs_eager():
    mgr = FakeMgr()
    bp = gbp(mgr, max_rows=2)
    calls = [call(mgr, s, 400, 30) for s in (1, 2, 3)]
    for c, g in zip(calls, bp.run(calls)):
        same(g, mgr.step(**c))
    assert bp.stats["graph_groups"] == 0 and bp.stats["eager_keys"] == {str((3, False, 25, 2)): 1}


def test_rendezvous_precaptures_at_construction_and_replays_graphs(monkeypatch):
    import mynah_v2.model.perception_batch as PB
    monkeypatch.setattr(PB, "_cuda_capture", fake_capture)
    n = 4
    ref = {s: solo(s, n) for s in "ABC"}
    eng = make_engine()
    m = MultiSessionEngine(eng, cuda_rng=False)
    rz = RendezvousEngine(m, R.FakeBackend(3), step_timeout_s=10.0, batch_perception=True, perception_graphs=True)
    assert rz.pbatch.stats["captured"] == [str((k, False, 25, 2)) for k in (2, 3)] and not rz.workers
    for s in "ABC":
        rz.open(s, T.SESS[s][0], seed=T.SESS[s][2])
    got = {s: [] for s in "ABC"}
    for k in range(n):
        r = rz.step_jobs({s: pcm(s, k) for s in "ABC"})
        for s in "ABC":
            got[s].append(obs(r[s]))
    assert got == ref
    assert rz.pbatch.stats["graph_groups"] == n - 1 and rz.pbatch.stats["eager_groups"] == 1
    rz.close_all()
