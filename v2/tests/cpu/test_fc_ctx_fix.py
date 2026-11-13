"""fc_ctx_fix: the non-blocking FC hand-back reproduced with fakes (no model).

Flow mirrored from the pinned NeMo pipeline: the main loop writes gen_*[:, t] and embeds gen_*[:, t-1]; on SOTC the FC
thread runs ``_run_fc_async_steps`` on CLONES (phase 1 call, phase 2 response), then the pipeline only advances
``frame_idx`` by the async steps. Without the fix the first resumed step reads the never-written position (fill 0);
with it, it reads what the FC loop wrote (text PAD, function EOTR)."""

import threading
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("nemo.collections.speechlm2.inference.streaming.state.s2s_context_manager")

from mynah_v2.model import fc_ctx_fix  # noqa: E402
from mynah_v2.model.reference import ReferenceEngine  # noqa: E402

PAD, EOTR, L = 12, 22, 64


class FakeWrapper:
    def __init__(self):
        self.model = SimpleNamespace(stt_model=SimpleNamespace(text_pad_id=PAD))
        self.reads = []                                       # (stream_id, frame_idx, text[t-1], func[t-1])

    def infer_one_step(self, audio_input=None, num_frames_per_chunk=1, frame_idx=0, gen_text=None,
                       gen_asr_text=None, gen_function_text=None, stream_id=None, **kw):
        t = frame_idx
        self.reads.append((stream_id, t, int(gen_text[0, t - 1]), int(gen_function_text[0, t - 1])))
        gen_text[0, t] = 5
        gen_asr_text[0, t] = PAD
        gen_function_text[0, t] = PAD
        return {}

    def _run_fc_async_steps(self, fc_state=None, gen_text=None, gen_asr_text=None, gen_function_text=None,
                            current_frame_idx=0, func=(), stream_id=None, **kw):
        for i, f in enumerate(func):
            t = current_frame_idx + 1 + i
            gen_text[0, t] = PAD                              # forced PAD during FC (wrapper :1579)
            gen_asr_text[0, t] = PAD
            gen_function_text[0, t] = f
        return (len(func), None, None, [], None)


class FakeEngine:
    _instrument_fc = ReferenceEngine._instrument_fc

    def __init__(self):
        self.wrapper = FakeWrapper()
        self.pipeline = SimpleNamespace(context_manager=None)
        self.fc_records, self._fc_lock = [], threading.Lock()
        self._instrument_fc()


class Ctx:
    def __init__(self):                                       # upstream fallback fill = 0
        self.gen_text, self.gen_asr_text, self.gen_function_text = (torch.zeros(1, L, dtype=torch.long)
                                                                    for _ in range(3))
        self.frame_idx = 0


def main_step(eng, ctx, sid):
    eng.wrapper.infer_one_step(frame_idx=ctx.frame_idx, gen_text=ctx.gen_text, gen_asr_text=ctx.gen_asr_text,
                               gen_function_text=ctx.gen_function_text, stream_id=sid)
    ctx.frame_idx += 1


def fc_cycle(eng, ctx, sid, p1, p2, applied=True):
    """SOTC at frame_idx-1: FC thread on clones (pipeline :1591), phase 1 + phase 2, then frame_idx += steps."""
    snap = [b.clone() for b in (ctx.gen_text, ctx.gen_asr_text, ctx.gen_function_text)]
    cfi, steps = ctx.frame_idx - 1, 0
    for func in (p1, p2):
        out = eng.wrapper._run_fc_async_steps(fc_state={}, gen_text=snap[0], gen_asr_text=snap[1],
                                              gen_function_text=snap[2], current_frame_idx=cfi + steps,
                                              func=func, stream_id=sid)
        steps += out[0]
    if applied:
        ctx.frame_idx += steps                                # pipeline :1060 (tokens NOT copied back)
    return steps


def run(eng, sid=0):
    ctx = Ctx()
    for _ in range(10):
        main_step(eng, ctx, sid)
    fc_cycle(eng, ctx, sid, p1=[30, 31, 32, 21], p2=[40, 41, EOTR])
    main_step(eng, ctx, sid)
    return ctx, eng.wrapper.reads[-1]


def test_without_fix_the_resumed_step_reads_token_zero():
    eng = FakeEngine()
    _, (_, t, txt, fn) = run(eng)
    assert t == 17 and (txt, fn) == (0, 0)                    # the upstream defect, reproduced


def test_with_fix_the_resumed_step_reads_the_fc_tokens_and_records_still_work():
    eng = FakeEngine()
    info = fc_ctx_fix.enable(eng)
    ctx, (_, t, txt, fn) = run(eng)
    assert t == 17 and (txt, fn) == (PAD, EOTR)
    assert ctx.gen_function_text[0, 10:17].tolist() == [30, 31, 32, 21, 40, 41, EOTR]
    assert ctx.gen_text[0, 10:17].tolist() == [PAD] * 7
    assert len(eng.fc_records) == 2                           # instrumentation chain intact (recorded → fixed)
    assert info["stats"]["copied_positions"] == 7 and info["stats"]["no_context"] == 0
    fc_ctx_fix.disable(eng)
    assert "infer_one_step" not in eng.wrapper.__dict__


def test_quit_and_abort_keep_upstreams_eos_written_after_the_thread_ends():
    eng = FakeEngine()
    fc_ctx_fix.enable(eng)
    ctx = Ctx()
    for _ in range(6):
        main_step(eng, ctx, 0)
    fc_cycle(eng, ctx, 0, p1=[30, 31], p2=[])                 # interrupted after 2 steps, result applied
    ctx.gen_text[0, ctx.frame_idx - 1] = 2                    # pipeline :1098 / :1137 (EOS), :1100 (func PAD)
    ctx.gen_function_text[0, ctx.frame_idx - 1] = PAD
    main_step(eng, ctx, 0)
    assert eng.wrapper.reads[-1][2:] == (2, PAD)              # upstream's deliberate tokens are not clobbered
    assert ctx.gen_function_text[0, 6] == 30                  # the rest of the FC span carries the real tokens


def test_unapplied_fc_is_harmless_and_consecutive_cycles_and_streams_stay_independent():
    eng = FakeEngine()
    info = fc_ctx_fix.enable(eng)
    a, b = Ctx(), Ctx()
    for _ in range(5):
        main_step(eng, a, 0)
        main_step(eng, b, 1)
    fc_cycle(eng, a, 0, p1=[30, 21], p2=[], applied=False)    # result not applied: frame_idx not advanced
    fc_cycle(eng, b, 1, p1=[30, 31, 21], p2=[40, EOTR])       # overlapping FC on the other session
    main_step(eng, a, 0)
    main_step(eng, b, 1)
    ra, rb = eng.wrapper.reads[-2], eng.wrapper.reads[-1]
    assert ra[2:] == (5, PAD)                                 # a reads its own written frame, then overwrites 5
    assert a.gen_function_text[0, 5] == PAD
    assert rb[2:] == (PAD, EOTR)
    assert b.gen_function_text[0, 5:10].tolist() == [30, 31, 21, 40, EOTR] and a.gen_function_text[0, 7] == 0
    # a second legitimate call on the same stream right after the first one
    fc_cycle(eng, b, 1, p1=[33, 21], p2=[42, EOTR])
    main_step(eng, b, 1)
    assert eng.wrapper.reads[-1][2:] == (PAD, EOTR)
    assert info["stats"]["copies"] == 5 and info["stats"]["errors"] == 0
    fc_ctx_fix.disable(eng)


def test_selective_variant_resumes_on_pad_pad_and_records_consumed_inputs():
    """fc_ctx_fix_sel (official checkpoint candidate): text/ASR copied, function channel PAD over the FC span, so the
    resumed step consumes (PAD, PAD) — never EOTR — and that is recorded from the step's real inputs."""
    eng = FakeEngine()
    info = fc_ctx_fix.enable(eng, pad_fill=True, channels=("gen_text", "gen_asr_text"), func_pad=True)
    ctx, (_, t, txt, fn) = run(eng)
    assert t == 17 and (txt, fn) == (PAD, PAD)
    assert ctx.gen_function_text[0, 10:17].tolist() == [PAD] * 7 and ctx.gen_text[0, 10:17].tolist() == [PAD] * 7
    assert info["stats"]["resume_inputs"] == [{"stream": 0, "frame_idx": 17, "text": PAD, "asr": PAD, "func": PAD}]
    # quit/abort: upstream writes EOS at frame_idx-1 after the thread ends → still what the resumed step consumes
    fc_cycle(eng, ctx, 0, p1=[30, 31], p2=[])
    ctx.gen_text[0, ctx.frame_idx - 1] = 2
    main_step(eng, ctx, 0)
    assert eng.wrapper.reads[-1][2:] == (2, PAD) and info["stats"]["resume_inputs"][-1]["text"] == 2
    # consecutive legitimate call and a second stream
    b = Ctx()
    for _ in range(3):
        main_step(eng, b, 1)
    fc_cycle(eng, ctx, 0, p1=[33, 21], p2=[42, EOTR])
    fc_cycle(eng, b, 1, p1=[30, 21], p2=[40, EOTR])
    main_step(eng, ctx, 0)
    main_step(eng, b, 1)
    assert eng.wrapper.reads[-2][2:] == (PAD, PAD) and eng.wrapper.reads[-1][2:] == (PAD, PAD)
    assert [r["stream"] for r in info["stats"]["resume_inputs"]] == [0, 0, 0, 1]
    fc_ctx_fix.disable(eng)


# ------------------------------------------------------------------ eotr_step (experiment, owner GO 2026-11-13)
SOTC = 20


class FakeWrapperE(FakeWrapper):
    """Adds the pieces the inserted EOTR slot uses: model_llm_interface (spied) and the NeMo loop's forced-token /
    max_async_steps behaviour (wrapper :1579-1606)."""

    def __init__(self, silent_pred=SOTC):
        super().__init__()
        self.silent_pred, self.calls = silent_pred, []

    def model_llm_interface(self, *a, **k):
        return {"predicted_token": torch.tensor([PAD]), "function_predicted_token": torch.tensor([self.silent_pred])}

    def _run_fc_async_steps(self, fc_state=None, gen_text=None, gen_asr_text=None, gen_function_text=None,
                            current_frame_idx=0, func=(), stream_id=None, max_async_steps=2000, **kw):
        self.calls.append(dict(kw, current_frame_idx=current_frame_idx, max_async_steps=max_async_steps,
                               fc_state={k: (list(v) if isinstance(v, list) else v) for k, v in (fc_state or {}).items()}))
        forced = (fc_state or {}).get("forced_function_tokens")
        if forced and max_async_steps == 1:                   # the inserted EOTR slot
            t = current_frame_idx + 1
            ans = self.model_llm_interface()
            gen_text[0, t] = PAD
            gen_asr_text[0, t] = PAD
            gen_function_text[0, t] = int(ans["function_predicted_token"][0])
            gen_function_text[0, t] = forced.pop(0)          # forced override, as the NeMo loop does
            return (1, "cache2", "tts2", ["chunk"], None)
        return super()._run_fc_async_steps(fc_state=fc_state, gen_text=gen_text, gen_asr_text=gen_asr_text,
                                           gen_function_text=gen_function_text, current_frame_idx=current_frame_idx,
                                           func=func, stream_id=stream_id)


class FakeEngineE(FakeEngine):
    def __init__(self, **kw):
        self.wrapper = FakeWrapperE(**kw)
        self.pipeline = SimpleNamespace(context_manager=None)
        self.fc_records, self._fc_lock = [], threading.Lock()
        self._instrument_fc()


def fc_cycle_e(eng, ctx, sid, p1, p2, phase2_kw=None):
    snap = [b.clone() for b in (ctx.gen_text, ctx.gen_asr_text, ctx.gen_function_text)]
    cfi, steps = ctx.frame_idx - 1, 0
    for func, extra in ((p1, {}), (p2, dict({"tool_response_text": "{}"}, **(phase2_kw or {})))):
        out = eng.wrapper._run_fc_async_steps(fc_state={}, gen_text=snap[0], gen_asr_text=snap[1],
                                              gen_function_text=snap[2], current_frame_idx=cfi + steps,
                                              func=func, stream_id=sid, **extra)
        steps += out[0]
    ctx.frame_idx += steps
    return steps, out


def test_eotr_step_inserts_the_training_slot_and_never_opens_a_call():
    eng, ctx = FakeEngineE(silent_pred=SOTC), Ctx()
    info = fc_ctx_fix.enable(eng, channels=fc_ctx_fix._CH, func_pad=False, eotr_step=True)
    for _ in range(4):
        main_step(eng, ctx, 1)
    p1, p2 = [30, 31, 21], [40, 41, EOTR]
    steps, out = fc_cycle_e(eng, ctx, 1, p1, p2)
    assert steps == len(p1) + len(p2) + 1                    # one inserted position more
    rec = info["stats"]["eotr_steps"][-1]
    last = 3 + len(p1) + len(p2)
    assert rec["consumed"] == {"text": PAD, "asr": PAD, "func": EOTR} and rec["frame"] == last + 1
    assert rec["pred_func"] == SOTC and rec["written"] == {"text": PAD, "func": PAD}   # predicted SOTC, written PAD
    call = eng.wrapper.calls[-1]
    assert call["max_async_steps"] == 1 and call["current_frame_idx"] == last and call["tool_response_text"] is None
    assert call["live_audio_queue"] is None and call["realtime_audio"] is None
    assert call["fc_state"] == {"forced_function_tokens": [PAD], "active": False}
    assert out[1:4] == ("cache2", "tts2", ["chunk"])          # cache / TTS state carried from the inserted step
    main_step(eng, ctx, 1)
    assert eng.wrapper.reads[-1][2:] == (PAD, PAD)            # the main loop resumes on (PAD, PAD)
    assert "model_llm_interface" not in eng.wrapper.__dict__  # the spy is gone, the method is the class one again
    assert info["stats"]["resume_inputs"][-1]["func"] == PAD


@pytest.mark.parametrize("p2,kw,why", [([40, 41, 7], {}, "no EOTR at the end"),
                                       ([40, 41, EOTR], {"abort_event": threading.Event()}, "abort/quit"),
                                       ([40, 41, EOTR], {"natural": True}, "natural interrupt")])
def test_eotr_step_skips_when_the_phase_did_not_end_on_eotr(p2, kw, why):
    eng, ctx = FakeEngineE(), Ctx()
    info = fc_ctx_fix.enable(eng, channels=fc_ctx_fix._CH, func_pad=False, eotr_step=True)
    for _ in range(4):
        main_step(eng, ctx, 1)
    extra = {}
    if "abort_event" in kw:
        kw["abort_event"].set()
        extra["abort_event"] = kw["abort_event"]
    snap = [b.clone() for b in (ctx.gen_text, ctx.gen_asr_text, ctx.gen_function_text)]
    st = {"natural_interrupt": True} if kw.get("natural") else {}
    out = eng.wrapper._run_fc_async_steps(fc_state=st, gen_text=snap[0], gen_asr_text=snap[1], gen_function_text=snap[2],
                                          current_frame_idx=3, func=p2, stream_id=1, tool_response_text="{}", **extra)
    assert out[0] == len(p2) and info["stats"]["eotr_steps"][-1]["skipped"] == why
    assert all(c["max_async_steps"] != 1 for c in eng.wrapper.calls)


def test_eotr_step_never_runs_in_phase_1_or_when_off():
    for on in (True, False):
        eng, ctx = FakeEngineE(), Ctx()
        info = fc_ctx_fix.enable(eng, channels=fc_ctx_fix._CH, func_pad=False, eotr_step=on)
        snap = [b.clone() for b in (ctx.gen_text, ctx.gen_asr_text, ctx.gen_function_text)]
        out = eng.wrapper._run_fc_async_steps(fc_state={}, gen_text=snap[0], gen_asr_text=snap[1],
                                              gen_function_text=snap[2], current_frame_idx=3, func=[30, EOTR],
                                              stream_id=1)            # phase 1: no tool_response_text
        assert out[0] == 2 and all(c["max_async_steps"] != 1 for c in eng.wrapper.calls)
        assert bool(info["stats"]["eotr_steps"]) == on        # recorded as skipped only when the option is on


def test_engine_modes_default_unchanged_and_eotr_step_explicit(monkeypatch):
    seen = []
    monkeypatch.setattr(fc_ctx_fix, "enable", lambda eng, **k: seen.append(k) or {"ok": True})
    monkeypatch.delenv("MYNAH_V2_FC_CTX_FIX", raising=False)
    from mynah_v2.model.reference import OFFICIAL_SNAPSHOT
    fake = SimpleNamespace(cfg=SimpleNamespace(s2s=SimpleNamespace(llm_checkpoint_path=f"/x/{OFFICIAL_SNAPSHOT}")))
    auto = ReferenceEngine._default_fc_ctx_fix(fake, None)
    assert auto["enabled"] and seen[-1] == {"pad_fill": True, "channels": ("gen_text", "gen_asr_text"), "func_pad": True}
    exp = ReferenceEngine._default_fc_ctx_fix(fake, "eotr_step")
    assert exp["enabled"] and seen[-1]["eotr_step"] is True and seen[-1]["func_pad"] is False
    other = SimpleNamespace(cfg=SimpleNamespace(s2s=SimpleNamespace(llm_checkpoint_path="/x/lab-ckpt")))
    assert not ReferenceEngine._default_fc_ctx_fix(other, None)["enabled"]   # non-official stays off by default
