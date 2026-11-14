"""EXPERIMENT (opt-in, perf experiment "fc_phase2_notts"): no EarTTS / codec work during FC phase 2.

Phase 2 of NeMo's non-blocking FC (`_run_fc_async_steps` called with `tool_response_text`) forces the
`<TOOL_RESPONSE>` tokens one per step and lets the model predict EOTR. Every forced step also runs one EarTTS step (CFG)
and one codec decode + `.cpu()` (wrapper ~:1471-1564), about 36 + ~30 ms of the ~165 ms step on GB10. None of that
reaches the caller: the pipeline passes no TTS audio queue to phase 2 ("TTS during forced-token injection produces
silence ... we do NOT pass tts_audio_output_queue here", pipeline ~:1658-1666) and the non-blocking hand-back never
reads phase-2 chunks. The text/ASR predictions are forced to PAD and the function prediction is overwritten by the
forced token on every step except the last (wrapper ~:1576-1593).

This wrapper calls phase 2 with `tts_state=None` (the upstream loop then skips its whole TTS block) and hands the
caller's `tts_state` back unchanged in the result, so the hand-back keeps the TTS state of the end of phase 1 (+ the
on-hold reminder). The Thinker sees exactly the same inputs: its tokens are bit-identical on the same timeline. What
changes, by design: the EarTTS context reaches the spoken answer without the N+1 PAD frames of phase 2 (audio after
the FC: to measure), and the FC returns sooner (the end-to-end timeline shifts).

Untouched: phase 1 (call decode + ack audio), a phase-2 call that continues an open chain (fc_state active), calls
without a TTS state. Wrapped under the engine's FC recorder like fc_ctx_fix / codec_fc_2codes, so traces still see it.
"""

from __future__ import annotations


def _wrap(inner, stats: dict):
    def run(*a, **kw):
        fs = kw.get("fc_state") or {}
        if (not getattr(run, "__v2_on__", True) or not kw.get("tool_response_text")
                or kw.get("tts_state") is None or fs.get("active")):
            stats["passthrough"] = stats.get("passthrough", 0) + 1
            return inner(*a, **kw)
        ts = kw["tts_state"]
        out = inner(*a, **dict(kw, tts_state=None))
        stats["phase2_calls"] = stats.get("phase2_calls", 0) + 1
        try:
            stats["skipped_tts_steps"] = stats.get("skipped_tts_steps", 0) + int(out[0])
        except (TypeError, ValueError, IndexError):
            pass
        if isinstance(out, tuple) and len(out) >= 3:
            out = (out[0], out[1], ts) + tuple(out[3:])
        return out
    run.__v2_p2notts__ = inner
    return run


def enable(engine) -> dict:
    w = engine.wrapper
    stats = getattr(engine, "_v2_p2notts_stats", None)
    if stats is None:
        stats = engine._v2_p2notts_stats = {}
    if getattr(w, "_v2_p2notts", None) is None:
        cur = w._run_fc_async_steps
        instrumented = hasattr(engine, "_instrument_fc") and hasattr(cur, "__wrapped_v2__")
        inner = cur.__wrapped_v2__ if instrumented else cur     # under the recorder: recorded -> this -> [...]
        f = _wrap(inner, stats)
        w._v2_p2notts = f
        w._v2_p2notts_had = "_run_fc_async_steps" in w.__dict__
        w._run_fc_async_steps = f
        if instrumented:
            engine._instrument_fc()
    w._v2_p2notts.__v2_on__ = True
    return {"note": "FC phase 2 without EarTTS/codec (the caller's TTS state is handed back unchanged)",
            "stats": stats}


def disable(engine) -> None:
    w = engine.wrapper
    f = getattr(w, "_v2_p2notts", None)
    if f is not None:
        f.__v2_on__ = False                 # keep the chain intact (the recorder may wrap it); just pass through
