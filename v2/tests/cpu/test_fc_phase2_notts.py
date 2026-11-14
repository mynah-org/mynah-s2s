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
