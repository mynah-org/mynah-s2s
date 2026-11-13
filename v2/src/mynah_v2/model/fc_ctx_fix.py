"""Fix of the NeMo VoiceChat non-blocking FC context hand-back (pinned NeMo Speech @097dfe9).

Upstream defect (verified by reading the pinned source, file:line at 097dfe9; confirmed causally by the multilingual
lab with a logits replay: (0, 0) at the resume position reproduces the re-opened call, (PAD, EOTR) fixes it):
  1. ``inference/streaming/state/s2s_context_manager.py:80``: ``text_pad_id = getattr(s2s_model.model, "text_pad_id", 0)``
     — ``NemotronVoiceChat`` has no ``text_pad_id`` (only ``stt_model`` does), so every context token buffer
     (gen_text / gen_asr_text / gen_function_text) is pre-filled with id 0 instead of the real PAD.
  2. ``inference/pipelines/streaming_s2s_pipeline.py:1591-1593``: the non-blocking FC background thread runs
     ``_run_fc_async_steps`` on CLONES of those buffers; when the thread is done (``:1043-1062`` normal completion,
     ``:1080-1110`` quit-to-normal, ``:1118-1140`` abort) the pipeline copies back the LLM cache / TTS state and does
     ``frame_idx += async_steps`` but NOT the tokens the FC loop wrote. The main loop then embeds
     ``gen_*[:, frame_idx - 1]`` (wrapper ``:2399-2406``) = a never-written position → id 0 on the text, ASR and
     function channels instead of the PAD / EOTR the FC loop produced; the turn-taking lookback windows (wrapper
     ``:2812``, ``:3047``) also read zeros over the whole FC span. Quit/abort then overwrite ``gen_text[fi-1]`` with
     EOS (and ``gen_function_text[fi-1]`` with PAD on quit) on purpose. The blocking path (``:1932``) passes the
     context buffers themselves and is not affected.

Fix (isolated, no upstream file edited; experiment ``fc_ctx_fix``):
  * the context manager uses ``stt_model.text_pad_id`` (class patch + live instances; unwritten positions refilled);
  * the main thread's ``infer_one_step`` remembers, per ``stream_id``, the session's own context buffers (the same
    tensor objects the pipeline later advances);
  * the innermost ``_run_fc_async_steps`` (FC thread), when each phase returns, copies the range it wrote
    ``[cfi+1, cfi+1+steps)`` from its clones into those buffers. This happens BEFORE the FC thread ends, i.e. before
    the pipeline applies the result, so upstream's own quit/abort EOS/PAD writes at ``fi-1`` land after it and are
    kept. Positions beyond the applied ``frame_idx`` (result not applied) are overwritten by the main loop before
    they are read. The main loop never touches positions > its frame_idx of that stream while the FC thread is alive
    (non-blocking mode skips ``infer_one_step``), and the buffers are per session → no cross-session effect under
    state swapping or the batched Thinker backend (which owns only the LLM cache, never these token buffers).
"""

from __future__ import annotations

import threading

_CH = ("gen_text", "gen_asr_text", "gen_function_text")


def _pad_id(engine) -> int:
    return int(engine.wrapper.model.stt_model.text_pad_id)


def enable(engine, pad_fill: bool = True, channels=_CH, func_pad: bool = False, eotr_step: bool = False) -> dict:
    """pad_fill: real PAD in the context manager; channels: which FC-written buffers are copied back (diagnostic
    variants isolate which part changes the model's behaviour at the resume step); func_pad: write PAD over the FC span
    of the function channel (the official VoiceChat re-opens the call when EOTR is fed back at resume: FX5/round 12).
    eotr_step (EXPERIMENT, off by default; owner GO 2026-11-13): the training layout's inserted EOTR slot. After a
    phase-2 FC run that ends on the model's own EOTR, one more inserted FC step runs. It uses the same NeMo loop with
    ``max_async_steps=1`` and consumes (text PAD, function EOTR, silence embedding), with no real audio and the live
    queue detached. Its function prediction is recorded and overwritten with PAD (the training label at that slot), so
    the step can never open or run a call. The main loop then resumes on (PAD, PAD), and any further call is decided
    by the first real frame through the normal path. Cost: one inserted position (one more 80 ms TTS silence chunk);
    no real audio frame is consumed.
    Every first main step after an FC hand-back records the tokens it ACTUALLY consumes (``stats["resume_inputs"]``)."""
    w = engine.wrapper
    if getattr(w, "_v2_fc_ctx_fix", None) is not None:
        raise RuntimeError("fc_ctx_fix already enabled")
    channels = tuple(channels)
    pad = _pad_id(engine)
    from nemo.collections.speechlm2.inference.streaming.state import s2s_context_manager as scm
    cls = scm.S2SContextManager
    orig_cls_init = cls.__init__

    def __init__(self, *a, **kw):
        orig_cls_init(self, *a, **kw)
        stt = getattr(getattr(self.s2s_model, "model", None), "stt_model", None)
        if stt is not None and hasattr(stt, "text_pad_id"):
            self.text_pad_id = int(stt.text_pad_id)
    if pad_fill:
        cls.__init__ = __init__

    refilled = 0
    cm = getattr(engine.pipeline, "context_manager", None)
    if cm is not None and pad_fill:
        cm.text_pad_id = pad
        for ctx in getattr(cm, "slot_contexts", []) or []:
            if ctx is None:
                continue
            fi = int(ctx.frame_idx)
            for name in _CH:
                buf = getattr(ctx, name, None)
                if buf is not None:
                    buf[:, fi:] = pad                     # never-written positions only
                    refilled += 1

    bufs: dict = {}                                       # stream_id → the session's context buffers
    lock = threading.Lock()
    stats = {"fc_calls": 0, "copies": 0, "copied_positions": 0, "no_context": 0, "errors": 0, "pad_id": pad,
             "resume_inputs": [], "eotr_steps": []}
    resume_pending: set = set()

    orig_infer = w.infer_one_step

    def infer_one_step(*a, **kw):
        sid = kw.get("stream_id")
        gt = kw.get("gen_text", a[3] if len(a) > 3 else None)
        if gt is not None:
            with lock:
                bufs[sid] = (gt, kw.get("gen_asr_text"), kw.get("gen_function_text"))
                first = sid in resume_pending
                resume_pending.discard(sid)
            if first and len(stats["resume_inputs"]) < 500:   # what the resumed step really embeds at frame_idx-1
                fi = int(kw.get("frame_idx", a[2] if len(a) > 2 else 0))
                ga, gf = kw.get("gen_asr_text"), kw.get("gen_function_text")
                stats["resume_inputs"].append({"stream": sid, "frame_idx": fi, "text": int(gt[0, fi - 1]),
                                               "asr": int(ga[0, fi - 1]) if ga is not None else None,
                                               "func": int(gf[0, fi - 1]) if gf is not None else None})
        return orig_infer(*a, **kw)

    # innermost FC call: rebuild the instrumentation chain on top of it (recorded → fc_fixed → upstream)
    cur = w._run_fc_async_steps
    inner = getattr(cur, "__wrapped_v2__", cur)

    def fc_fixed(*args, **kw):
        out = inner(*args, **kw)
        if eotr_step:
            out = _eotr_silent_step(w, inner, kw, out, pad, stats)
        try:
            stats["fc_calls"] += 1
            steps = int(out[0])
            if steps > 0:
                lo = int(kw["current_frame_idx"]) + 1
                with lock:
                    dst = bufs.get(kw.get("stream_id"))
                if dst is None:
                    stats["no_context"] += 1
                else:
                    for d, name in zip(dst, _CH):
                        src = kw.get(name)
                        if name in channels and d is not None and src is not None and d is not src:
                            hi = min(lo + steps, d.shape[1], src.shape[1])
                            d[:, lo:hi] = src[:, lo:hi].to(d.device)
                        elif func_pad and name == "gen_function_text" and d is not None:
                            d[:, lo:min(lo + steps, d.shape[1])] = pad
                    stats["copies"] += 1
                    stats["copied_positions"] += steps
                with lock:
                    resume_pending.add(kw.get("stream_id"))
        except Exception:  # noqa: BLE001 - never break the FC path; counted
            stats["errors"] += 1
        return out

    w._run_fc_async_steps = fc_fixed
    engine._instrument_fc()                               # recorded(fc_fixed), __wrapped_v2__ = fc_fixed
    had_infer_attr = "infer_one_step" in w.__dict__
    w.infer_one_step = infer_one_step
    w._v2_fc_ctx_fix = {"cls_init": orig_cls_init, "cls": cls, "infer": orig_infer, "had_infer_attr": had_infer_attr,
                        "fc_inner": inner, "bufs": bufs, "stats": stats}
    return {"pad_id": pad, "pad_fill": pad_fill, "channels": channels, "func_pad": func_pad, "eotr_step": eotr_step,
            "refilled_buffers": refilled, "stats": stats}


EOTR_ID = 22


def _eotr_silent_step(w, inner, kw: dict, out, pad: int, stats: dict):
    """The inserted EOTR slot (see enable(eotr_step)). Returns the FC result extended by one step, or ``out`` unchanged
    (the reason is recorded) when the phase did not end on the model's own EOTR (abort, quit, interrupt, step cap)."""
    rec = {"stream": kw.get("stream_id")}
    try:
        steps = int(out[0])
        cfi = int(kw["current_frame_idx"])
        last = cfi + steps                                   # the position that holds the model's EOTR
        gt, ga, gf = kw.get("gen_text"), kw.get("gen_asr_text"), kw.get("gen_function_text")
        fc_state = kw.get("fc_state") or {}
        ev = [kw.get("abort_event"), kw.get("quit_async_event")]
        why = ("not phase 2" if not kw.get("tool_response_text") else "no steps" if steps <= 0
               else "no buffer" if gf is None or gt is None else "end of buffer" if last + 1 >= gf.shape[1]
               else "abort/quit" if any(e is not None and e.is_set() for e in ev)
               else "natural interrupt" if fc_state.get("natural_interrupt")
               else "no EOTR at the end" if int(gf[0, last]) != int(getattr(w, "_fc_eotr_id", None) or EOTR_ID) else None)
        rec.update(frame=last + 1, phase_steps=steps)
        if why:
            rec["skipped"] = why
            return out
        rec["consumed"] = {"text": int(gt[0, last]), "asr": int(ga[0, last]) if ga is not None else None,
                          "func": int(gf[0, last])}
        seen = {}
        llm = w.model_llm_interface
        had_attr = "model_llm_interface" in w.__dict__

        def spy(*a, **k):                                    # record what the silent step predicts (then overwritten)
            ans = llm(*a, **k)
            for key in ("predicted_token", "function_predicted_token"):
                v = ans.get(key) if isinstance(ans, dict) else None
                if v is not None:
                    seen[key] = int(v.reshape(-1)[0]) if hasattr(v, "reshape") else int(v)
            return ans

        kw2 = dict(kw, fc_state={"forced_function_tokens": [pad], "active": False}, current_frame_idx=last,
                   dynamic_cache=out[1], tts_state=out[2], rnnt_partial_hypotheses=out[4] if len(out) > 4 else None,
                   tool_response_text=None, max_async_steps=1, live_audio_queue=None, realtime_audio=None,
                   rnnt_text_queue=None, acknowledgement_tokens=None)
        w.model_llm_interface = spy
        try:
            o2 = inner(**kw2)
        finally:                                             # restore exactly what was there
            if had_attr:
                w.model_llm_interface = llm
            else:
                del w.model_llm_interface
        rec.update(steps=int(o2[0]), pred_text=seen.get("predicted_token"), pred_func=seen.get("function_predicted_token"),
                   written={"text": int(gt[0, last + 1]), "func": int(gf[0, last + 1])})
        out = (steps + int(o2[0]), o2[1], o2[2], list(out[3] or []) + list(o2[3] or []),
               *(o2[4:] if len(o2) > 4 else ()))
        return out
    except Exception as e:  # noqa: BLE001 - never break the FC path; recorded
        rec["error"] = repr(e)
        stats["errors"] += 1
        return out
    finally:
        if len(stats["eotr_steps"]) < 500:
            stats["eotr_steps"].append(rec)


def forget(engine, stream_id) -> None:
    """Drop a closed stream's buffer references (session teardown)."""
    st = getattr(engine.wrapper, "_v2_fc_ctx_fix", None)
    if st is not None:
        st["bufs"].pop(stream_id, None)


def disable(engine) -> None:
    w = engine.wrapper
    st = getattr(w, "_v2_fc_ctx_fix", None)
    if st is None:
        return
    st["cls"].__init__ = st["cls_init"]
    if st["had_infer_attr"]:
        w.infer_one_step = st["infer"]
    else:
        del w.infer_one_step
    w._run_fc_async_steps = st["fc_inner"]
    engine._instrument_fc()
    del w._v2_fc_ctx_fix
