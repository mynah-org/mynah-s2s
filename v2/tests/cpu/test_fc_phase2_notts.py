"""CPU tests of the experimental FC phase-2 no-TTS wrapper (fake engine; the real loop is GPU-only)."""
from types import SimpleNamespace

from mynah_v2.model import fc_phase2_notts as P


class FakeEngine:
    def __init__(self):
        self.calls = []

        def upstream(**kw):
            self.calls.append(dict(kw))
            return (34, "cache", kw.get("tts_state"), ["chunk"] if kw.get("tts_state") is not None else [], "rnnt")
        self.wrapper = SimpleNamespace(_run_fc_async_steps=upstream)
        self.recorded = []
        self._instrument_fc()

    def _instrument_fc(self):                       # same shape as ReferenceEngine._instrument_fc
        orig = self.wrapper._run_fc_async_steps

        def recorded(*a, **kw):
            out = orig(*a, **kw)
            self.recorded.append(out[0])
            return out
        recorded.__wrapped_v2__ = orig
        self.wrapper._run_fc_async_steps = recorded


def test_phase2_runs_without_tts_and_hands_the_state_back():
    e = FakeEngine()
    info = P.enable(e)
    ts = {"code": 1, "past_key_values": "kv"}
    out = e.wrapper._run_fc_async_steps(tool_response_text="<TOOL_RESPONSE>[{}]</TOOL_RESPONSE>", tts_state=ts,
                                        fc_state={"active": False})
    assert e.calls[-1]["tts_state"] is None                 # upstream skipped its TTS block
    assert out[2] is ts and out[0] == 34 and out[3] == []    # caller's TTS state back unchanged
    assert e.recorded == [34]                                # still under the engine's FC recorder
    assert info["stats"]["phase2_calls"] == 1 and info["stats"]["skipped_tts_steps"] == 34


def test_phase1_open_chain_and_no_tts_pass_through():
    e = FakeEngine()
    P.enable(e)
    ts = {"code": 1}
    for kw in ({"tool_response_text": None, "tts_state": ts},                                    # phase 1
               {"tool_response_text": "<TOOL_RESPONSE>[{}]</TOOL_RESPONSE>", "tts_state": ts,
                "fc_state": {"active": True}},                                                 # chain continuing
               {"tool_response_text": "<TOOL_RESPONSE>[{}]</TOOL_RESPONSE>", "tts_state": None}):  # no TTS at all
        out = e.wrapper._run_fc_async_steps(**kw)
        assert e.calls[-1]["tts_state"] is kw["tts_state"] and out[2] is kw["tts_state"]


def test_disable_passes_through_and_enable_is_idempotent():
    e = FakeEngine()
    P.enable(e)
    P.enable(e)
    ts = {"code": 1}
    P.disable(e)
    out = e.wrapper._run_fc_async_steps(tool_response_text="x", tts_state=ts, fc_state={})
    assert e.calls[-1]["tts_state"] is ts and out[3] == ["chunk"]
    P.enable(e)
    e.wrapper._run_fc_async_steps(tool_response_text="x", tts_state=ts, fc_state={})
    assert e.calls[-1]["tts_state"] is None


def test_registered_as_perf_experiment():
    from mynah_v2.model import perf
    assert "fc_phase2_notts" in perf.EXPERIMENTS


def test_prefill_dispatch_is_thread_local_across_overlapping_sessions():
    """Two sessions' phase-2 calls overlap on one shared wrapper (MultiSessionEngine): each FC thread sees only its own
    deferred interface; the main thread sees the real one throughout; the interface object is installed once."""
    import threading
    import torch
    from mynah_v2.model import fc_phase2_prefill as F
    seen, gate = {}, threading.Barrier(2)

    class Real:
        def __call__(self, x, cache=None, **kw):
            return {"real": True}

    def upstream(**kw):                         # phase 2: the loop calls the interface while forced tokens are queued
        gate.wait(5)
        ans = e.wrapper.model_llm_interface(torch.zeros(1, 1, 4), cache=kw["dynamic_cache"])
        seen[kw["sid"]] = ans
        main = []
        t = threading.Thread(target=lambda: main.append(e.wrapper.model_llm_interface(torch.zeros(1, 1, 4), cache=1)))
        t.start(); t.join()
        seen[kw["sid"] + "_other_thread"] = main[0]
        gate.wait(5)
        return (1, kw["dynamic_cache"], None, [], None)

    stt = SimpleNamespace(text_pad_id=12, llm=SimpleNamespace())
    e = SimpleNamespace(wrapper=SimpleNamespace(_run_fc_async_steps=upstream, model_llm_interface=Real(),
                                                model=SimpleNamespace(stt_model=stt)))
    import sys as _s
    _s.modules.setdefault(type(stt.llm).__module__, _s.modules[__name__])
    vars(_s.modules[type(stt.llm).__module__]).setdefault("causal_conv1d_update", None)
    F.enable(e)
    cache = SimpleNamespace(conv_states=[], _v2_pos=5)

    def phase2(sid):
        e.wrapper._run_fc_async_steps(tool_response_text="x", fc_state={"forced_function_tokens": [1, 2]},
                                      dynamic_cache=cache, sid=sid)
    ts = [threading.Thread(target=phase2, args=(s,)) for s in ("A", "B")]
    for t in ts:
        t.start()
    for t in ts:
        t.join(10)
    assert int(seen["A"]["function_predicted_token"]) == 12 and int(seen["B"]["function_predicted_token"]) == 12
    assert seen["A_other_thread"] == {"real": True} and seen["B_other_thread"] == {"real": True}
    assert isinstance(e.wrapper.model_llm_interface, F._Dispatch)
    assert e.wrapper.model_llm_interface(torch.zeros(1, 1, 4), cache=1) == {"real": True}
