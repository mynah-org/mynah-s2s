"""codec_fc_2codes: the FC thread decodes codes in pairs; audio (list + queue) in order, cache aligned, half the calls."""

import queue
from types import SimpleNamespace

import pytest
import torch

from mynah_v2.model import codec_async as ca
from mynah_v2.model import perf


class CCache:
    def __init__(self):
        self.cache = {}

    def update(self, x, k, pad):
        p = self.cache.get(k)
        p = torch.zeros(x.shape[0], x.shape[1], pad, dtype=x.dtype) if p is None else p
        y = torch.cat([p, x], 2)
        self.cache[k] = y[:, :, -pad:]
        return y


class Codec:
    def __init__(self):
        self.calls = []

    def decode(self, code, code_len=None, constrain_value_range=True, cache=None, flush=False):
        self.calls.append(code.shape[1])
        x = code.double().sum(-1).repeat_interleave(3, 1).unsqueeze(1) / 50.0
        p = cache.update(x, 0, 2)
        y = torch.tanh(p[..., 2:] - 0.5 * p[..., 1:-1] + 0.2 * p[..., :-2])
        return y, (None if code_len is None else code_len * 3)


class W:
    """Mimics the wrapper's FC loop (:1541-1564): one decode per step, .detach().cpu(), list + queue."""

    def __init__(self):
        self.model = SimpleNamespace(tts_model=SimpleNamespace(audio_codec=Codec()))

    def _run_fc_async_steps(self, n=0, cache=None, tts_audio_output_queue=None, seed=0, fail_at=None):
        chunks = []
        g = torch.Generator().manual_seed(seed)
        for t in range(n):
            if t == fail_at:
                raise RuntimeError("abort")
            code = torch.randint(0, 9, (1, 1, 4), generator=g)
            wav, _ = self.model.tts_model.audio_codec.decode(code, torch.tensor([1]), cache=cache)
            c = wav.detach().cpu()
            chunks.append(c)
            if tts_audio_output_queue is not None:
                tts_audio_output_queue.put(c)
        return n, None, None, chunks, None


def drain(q):
    out = []
    while not q.empty():
        out.append(q.get_nowait())
    return out


def cat(xs):
    return torch.cat(list(xs) + [torch.zeros(1, 1, 0, dtype=torch.float64)], -1)


@pytest.mark.parametrize("n", [0, 1, 6, 7])
def test_same_audio_in_order_cache_aligned_and_half_the_calls(n):
    res = {}
    for on in (False, True):
        e = SimpleNamespace(wrapper=W())
        if on:
            perf.enable("codec_fc_2codes", e)
        cache, q = CCache(), queue.Queue()
        _, _, _, chunks, _ = e.wrapper._run_fc_async_steps(n=n, cache=cache, tts_audio_output_queue=q, seed=n)
        qa = drain(q)
        # the main path afterwards on the same cache (2 codes) must continue identically
        nxt, _ = e.wrapper.model.tts_model.audio_codec.decode(torch.ones(1, 2, 4, dtype=torch.long), cache=cache)
        res[on] = (cat(chunks), cat(qa), nxt, list(e.wrapper.model.tts_model.audio_codec.calls))
        if on:
            st = e._v2_fc2_stats
            assert st["pairs"] == n // 2 and st["flushed"] == n % 2 and st["calls"] == 1
            perf.disable("codec_fc_2codes", e)
            assert "_run_fc_async_steps" not in e.wrapper.__dict__
            assert "decode" not in e.wrapper.model.tts_model.audio_codec.__dict__
    for i in range(3):
        assert res[False][i].shape == res[True][i].shape
        assert torch.allclose(res[False][i], res[True][i], atol=1e-12, rtol=0), i
    assert res[True][3] == [2] * (n // 2) + [1] * (n % 2) + [2]


def test_pending_code_decoded_on_abort_and_no_pairing_outside_an_fc_phase():
    e, ref = SimpleNamespace(wrapper=W()), SimpleNamespace(wrapper=W())
    perf.enable("codec_fc_2codes", e)
    c1, c2 = CCache(), CCache()
    with pytest.raises(RuntimeError):
        e.wrapper._run_fc_async_steps(n=9, cache=c1, seed=3, fail_at=5)
    with pytest.raises(RuntimeError):
        ref.wrapper._run_fc_async_steps(n=9, cache=c2, seed=3, fail_at=5)
    assert torch.allclose(c1.cache[0], c2.cache[0], atol=1e-12, rtol=0)      # 5 codes decoded in both
    codec = e.wrapper.model.tts_model.audio_codec
    n0 = len(codec.calls)
    w, _ = codec.decode(torch.ones(1, 1, 4, dtype=torch.long), cache=c1)   # main path: untouched
    assert w.shape[-1] == 3 and codec.calls[n0:] == [1]
    perf.disable("codec_fc_2codes", e)


def test_holds_its_own_copy_of_the_code_and_unpairable_calls_keep_order():
    e = SimpleNamespace(wrapper=W())
    perf.enable("codec_fc_2codes", e)
    codec = e.wrapper.model.tts_model.audio_codec
    buf = torch.full((1, 1, 4), 3, dtype=torch.long)       # a static output buffer reused by the next step
    cache, ref_cache = CCache(), CCache()

    def steps(**kw):
        outs = []
        for v in (3, 5, 7):
            buf.fill_(v)
            outs.append(codec.decode(buf, torch.tensor([1]), cache=cache)[0])
        outs.append(codec.decode(torch.full((1, 2, 4), 2, dtype=torch.long), cache=cache)[0])  # unpairable: 2 codes
        return 4, None, None, outs, None
    e.wrapper._run_fc_async_steps = ca.make_fc2_steps(steps, lambda: codec.decode.__v2_fc2__, {})
    _, _, _, outs, _ = e.wrapper._run_fc_async_steps()
    rc = Codec()
    ref = [rc.decode(torch.full((1, 1, 4), v, dtype=torch.long), cache=ref_cache)[0] for v in (3, 5, 7)]
    ref.append(rc.decode(torch.full((1, 2, 4), 2, dtype=torch.long), cache=ref_cache)[0])
    assert torch.allclose(cat(outs), cat(ref), atol=1e-12, rtol=0)


# ------------------------------------------------------------------------------------------------ engine chain
# Round 29 CB5: codec_fc_2codes wrapped _run_fc_async_steps ABOVE the ReferenceEngine FC recorder → MultiSessionEngine
# refused it. It must sit under the recorder (fc_ctx_fix pattern) and compose with fc_ctx_fix in either order.
import importlib  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
T = importlib.import_module("test_session_state")
fake_nemo = T.fake_nemo                                  # autouse fixture (nemo stand-ins)


def ctx_fix_standin(engine, calls):
    """Same chain surgery as fc_ctx_fix.enable (fc_ctx_fix.py:102-134): wrap the innermost call, re-instrument."""
    w = engine.wrapper
    cur = w._run_fc_async_steps
    inner = getattr(cur, "__wrapped_v2__", cur)

    def fc_fixed(*a, **kw):
        calls.append("fix")
        return inner(*a, **kw)
    w._run_fc_async_steps = fc_fixed
    engine._instrument_fc()
    return fc_fixed


def engine_with_codec():
    eng = T.make_engine()
    eng.wrapper.model = SimpleNamespace(tts_model=SimpleNamespace(audio_codec=Codec()))
    return eng


def fc_call(eng, stream):
    box = {}

    def run():
        try:
            gt = torch.zeros(1, 8, dtype=torch.long)
            box["out"] = eng.wrapper._run_fc_async_steps(gen_text=gt, gen_function_text=gt, current_frame_idx=2,
                                                         stream_id=stream)
        except Exception as e:  # noqa: BLE001
            box["err"] = e
    t = threading.Thread(target=run, name="Thread-7 (_fc_async_thread_fn)")
    t.start()
    t.join(5)
    assert "err" not in box, box.get("err")
    return box["out"]


@pytest.mark.parametrize("order", ["fix_then_fc2", "fc2_then_fix"])
def test_fc2_sits_under_the_recorder_and_multisession_routes_it_with_ctx_fix(order):
    from mynah_v2.model.session_state import MultiSessionEngine
    eng = engine_with_codec()
    calls = []
    if order == "fix_then_fc2":
        fix = ctx_fix_standin(eng, calls)
        perf.enable("codec_fc_2codes", eng)
    else:
        perf.enable("codec_fc_2codes", eng)
        fix = ctx_fix_standin(eng, calls)
    w = eng.wrapper
    top = w._run_fc_async_steps
    assert hasattr(top, "__wrapped_v2__")                      # the engine recorder is on top
    fc2 = w._v2_fc2_steps
    if order == "fix_then_fc2":                                # recorded → fc2 → fix → upstream
        assert top.__wrapped_v2__ is fc2 and fc2.__v2_fc2__ is fix
    else:                                                      # recorded → fix → fc2 → upstream
        assert top.__wrapped_v2__ is fix
    m = MultiSessionEngine(eng, cuda_rng=False)               # CB5 failed here
    m.open("A", "pa", seed=1)
    stream_a = m.store.record("A").fields["engine.stream_id"]
    out = fc_call(eng, stream_a)
    assert out[0] == 1 and calls == ["fix"] and eng._v2_fc2_stats["calls"] == 1
    assert len(m.store.get("A", "engine.fc_records")) == 1   # recorded and routed to A
    m.close_all()
    assert w._run_fc_async_steps is top
    perf.disable("codec_fc_2codes", eng)
    if order == "fix_then_fc2":                                # unwrapped: recorded → fix
        assert w._run_fc_async_steps.__wrapped_v2__ is fix and not hasattr(w, "_v2_fc2_steps")
    else:                                                      # buried under fix: pass-through
        assert fc2.__v2_fc2_on__ is False
    assert "decode" not in w.model.tts_model.audio_codec.__dict__
    calls.clear()
    eng.wrapper._run_fc_async_steps(stream_id=0)
    assert calls == ["fix"] and eng._v2_fc2_stats["calls"] == 1   # fc2 no longer counts
