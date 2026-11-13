"""Batched cross-session codec decode (model/codec_batch.py + RendezvousEngine(batch_codec=True)) on a fake causal
streaming codec with the upstream cache contract (CausalConv1dCache.update). Every session's audio must equal its solo
run (the fake is exact in float64; the real codec differs by ~1e-6, measured on GPU)."""

import importlib
import random
import sys
import threading
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
R = importlib.import_module("test_rendezvous")
T = R.T
fake_nemo = T.fake_nemo                                  # autouse fixture (nemo stand-ins)

from mynah_v2.model.codec_batch import batched_decode, decode_kwargs, stack_key  # noqa: E402
from mynah_v2.model.reference import ReferenceEngine  # noqa: E402
from mynah_v2.model.session_state import MultiSessionEngine  # noqa: E402
from mynah_v2.model.session_tools import SessionToolBinding  # noqa: E402
from mynah_v2.runtime.rendezvous import RendezvousEngine  # noqa: E402

DEC_CALLS: list = []


class CCache:
    """Same contract as ear_tts_vae_codec.CausalConv1dCache (update -> padded states, keeps the tail)."""

    def __init__(self):
        self.cache = {}

    def update(self, states, layer_id, padding, padding_value=0, flush=False):
        b, c, t = states.size()
        pad = self.cache.get(layer_id)
        if pad is None:
            pad = torch.zeros((b, c, padding), dtype=states.dtype) + padding_value
        assert pad.size(2) == padding
        padded = torch.cat([pad, states], dim=2)
        self.cache[layer_id] = padded[:, :, -padding:]
        if flush:
            del self.cache[layer_id]
        return padded


class FakeCodec:
    """Two causal layers with state (kernel 3 and 2) + a per-sample nonlinearity; 4 samples per code."""

    def decode(self, code, code_len=None, constrain_value_range=True, cache=None, flush=False):
        DEC_CALLS.append((threading.current_thread(), code.shape[0], torch.is_grad_enabled()))
        x = (code.double() * torch.arange(1, code.shape[-1] + 1, dtype=torch.float64)).sum(-1)       # [B, T]
        x = x.repeat_interleave(4, dim=1).unsqueeze(1) / 100.0                                       # [B, 1, 4T]
        p = cache.update(x, 0, 2, flush=flush) if cache is not None else torch.nn.functional.pad(x, [2, 0])
        y = torch.tanh(p[..., 2:] + 0.5 * p[..., 1:-1] - 0.25 * p[..., :-2])
        q = cache.update(y, "spec", 1, flush=flush) if cache is not None else torch.nn.functional.pad(y, [1, 0])
        out = q[..., 1:] * 0.7 + q[..., :-1] * 0.3
        if constrain_value_range:
            out = out.clamp(-0.9, 0.9)
        return out, (None if code_len is None else code_len * 4)


def codes(seed, k=2, q=5):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 9, (1, k, q), generator=g)


# ------------------------------------------------------------------------------------------------ unit
def test_decode_kwargs_normalizes_and_rejects():
    c = torch.zeros(1, 2, 3)
    d = decode_kwargs((c, None), {"cache": 7})
    assert d == {"code": c, "code_len": None, "constrain_value_range": True, "cache": 7, "flush": False}
    assert decode_kwargs((c,), {"bogus": 1}) is None
    assert decode_kwargs((c, None), {"code_len": None}) is None
    assert decode_kwargs((), {}) is None


@pytest.mark.parametrize("warm", [0, 3])
def test_batched_decode_equals_per_session_and_writes_back_each_cache(warm):
    codec = FakeCodec()
    seqs = {s: [codes(100 * s + i) for i in range(warm + 4)] for s in range(3)}
    solo_c, bat_c = {s: CCache() for s in seqs}, {s: CCache() for s in seqs}
    solo_out = {s: [] for s in seqs}
    for s, cs in seqs.items():
        for c in cs:
            solo_out[s].append(codec.decode(c, torch.tensor([2]), cache=solo_c[s]))
    bat_out = {s: [] for s in seqs}
    for i in range(warm + 4):
        reqs = [decode_kwargs((seqs[s][i], torch.tensor([2])), {"cache": bat_c[s]}) for s in seqs]
        if i < warm:                                     # sessions warmed alone, then batched
            outs = [codec.decode(**d) for d in reqs]
        else:
            assert len({stack_key(d) for d in reqs}) == 1
            outs = batched_decode(codec.decode, reqs)
        for s, o in zip(seqs, outs):
            bat_out[s].append(o)
    for s in seqs:
        for (wa, la), (wb, lb) in zip(solo_out[s], bat_out[s]):
            assert torch.equal(wa, wb) and torch.equal(la, lb) and wb.shape[0] == 1
        assert set(solo_c[s].cache) == set(bat_c[s].cache)
        for k in solo_c[s].cache:
            assert torch.equal(solo_c[s].cache[k], bat_c[s].cache[k]) and bat_c[s].cache[k].shape[0] == 1


def test_stack_key_separates_layouts_and_refuses_unstackable():
    a, b = CCache(), CCache()
    FakeCodec().decode(codes(1), cache=a)
    da = decode_kwargs((codes(2),), {"cache": a})
    db = decode_kwargs((codes(3),), {"cache": b})                 # empty cache: different layout
    assert stack_key(da) != stack_key(db) and stack_key(db) is not None
    assert stack_key(decode_kwargs((codes(2, k=1),), {"cache": a})) != stack_key(da)
    assert stack_key(decode_kwargs((codes(2),), {"cache": a, "flush": True})) is None
    assert stack_key(decode_kwargs((codes(2),), {})) is None
    assert stack_key(decode_kwargs((torch.zeros(2, 2, 5, dtype=torch.long),), {"cache": a})) is None


# ------------------------------------------------------------------------------------------------ rendezvous
class CWrapper(R.Wrapper):
    def __init__(self):
        super().__init__()
        self.model.tts_model = type("TTS", (), {})()
        self.model.tts_model.audio_codec = FakeCodec()


class CPipeline(R.Pipeline):
    """After the Thinker frames: one codec decode per chunk with the session's cache (in the swapped context), codes
    derived from the session's tokens, so a wrong row or cache shows up in the text."""

    def generate_step(self, frames):
        super().generate_step(frames)
        f, w = frames[0], self.s2s_model
        if (f.is_first and f.samples.numel() == 0) or float(f.samples[0]) < 0:
            return
        ctx, _ = self.context_manager.get_context([f.stream_id])
        cc = getattr(ctx, "codec_cache", None)
        if cc is None:
            cc = ctx.codec_cache = CCache()
        st = self.get_or_create_state(f.stream_id)
        seed = (sum(map(ord, st.text)) + ctx.frame_idx) % 10007
        with torch.no_grad():
            wav, _ = w.model.tts_model.audio_codec.decode(codes(seed), torch.tensor([2]), cache=cc)
        st.text += f" A{float(wav.sum()):.12f}"


def make_engine():
    w = CWrapper()
    p = CPipeline(w)
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


def solo(sid, n):
    eng = make_engine()
    torch.manual_seed(T.SESS[sid][2])
    random.seed(T.SESS[sid][2])
    eng.open(T.SESS[sid][0])
    res = [R.obs(eng.step(T.pcm(sid, k))) for k in range(n)]
    eng.close()
    return res


def rz_engine(batch_codec=True):
    eng = make_engine()
    m = MultiSessionEngine(eng, cuda_rng=False)
    return eng, m, RendezvousEngine(m, R.FakeBackend(3), step_timeout_s=10.0, batch_codec=batch_codec)


def test_rendezvous_batches_codec_of_all_sessions_and_outputs_equal_solo():
    n = 5
    ref = {s: solo(s, n) for s in "ABC"}
    assert all(" A" in r[0][1] for r in ref.values())
    eng, m, rz = rz_engine()
    for s in "ABC":
        rz.open(s, T.SESS[s][0], seed=T.SESS[s][2])
    DEC_CALLS.clear()
    got = {s: [] for s in "ABC"}
    for k in range(n):
        r = rz.step_jobs({s: T.pcm(s, k) for s in "ABC"})
        for s in "ABC":
            got[s].append(R.obs(r[s]))
    assert got == ref
    cs = rz.stats["codec"]
    assert cs["batch_hist"] == {3: n} and cs["b1"] == 0 and cs["passthrough"] == 0
    assert rz.stats["batch_hist"] == {3: 2 * n}            # Thinker frames still batched A+B+C
    assert DEC_CALLS == [(threading.main_thread(), 3, False)] * n   # one batched call per round, coordinator, no_grad
    assert all(j.get("codec", 0) == 1 for j in rz.job_log[-3:])
    rz.close_all()
    assert "decode" not in eng.wrapper.model.tts_model.audio_codec.__dict__


def test_lone_session_uses_installed_decode_late_session_joins_and_other_threads_pass_through():
    ref = {s: solo(s, 4) for s in "AB"}
    eng, m, rz = rz_engine()
    codec = eng.wrapper.model.tts_model.audio_codec
    rz.open("A", T.SESS["A"][0], seed=T.SESS["A"][2])
    got = {"A": [], "B": []}
    for k in range(2):
        got["A"].append(R.obs(rz.step_jobs({"A": T.pcm("A", k)})["A"]))
    rz.open("B", T.SESS["B"][0], seed=T.SESS["B"][2])
    for k in range(2):
        r = rz.step_jobs({"A": T.pcm("A", k + 2), "B": T.pcm("B", k)})
        got["A"].append(R.obs(r["A"]))
        got["B"].append(R.obs(r["B"]))
    for k in range(2, 4):
        got["B"].append(R.obs(rz.step_jobs({"B": T.pcm("B", k)})["B"]))
    assert got == ref
    cs = rz.stats["codec"]
    # B's first decode has an empty cache (different layout from A's): not stacked, both alone
    assert cs["b1"] == 6 and cs["batch_hist"] == {2: 1} and cs["unstackable"] == 0
    res = []
    t = threading.Thread(target=lambda: res.append(codec.decode(codes(1), cache=CCache())), name="Thread-9 (_fc_async_x)")
    t.start()
    t.join()
    assert cs["passthrough"] == 1 and res[0][0].shape[0] == 1
    rz.close_all()


def test_failing_batched_codec_fails_its_members_and_default_off_never_touches_the_codec():
    eng, m, rz = rz_engine(batch_codec=False)
    assert "decode" not in eng.wrapper.model.tts_model.audio_codec.__dict__ and "codec" not in rz.stats
    rz.close_all()
    eng, m, rz = rz_engine()
    for s in "AB":
        rz.open(s, T.SESS[s][0], seed=T.SESS[s][2])
    rz.step_jobs({s: T.pcm(s, 0) for s in "AB"})

    def boom(*a, **k):
        raise RuntimeError("batched codec failure")
    rz._codec_inner = boom
    r = rz.step_jobs({s: T.pcm(s, 1) for s in "AB"})
    assert all(isinstance(r[s], RuntimeError) and "codec failure" in str(r[s]) for s in "AB")
    assert set(rz.failed) == {"A", "B"}
    rz.close_all()
