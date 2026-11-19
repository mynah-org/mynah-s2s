"""Frankenstein v2 behind V2's Session / Engine contract (``chunk_samples``, ``open``, ``step``, ``fc_active``,
``close``), so V2's Session and WebSocket server run it unchanged (a separate engine: ENGINE=frankenstein2; v1 and
the VoiceChat engine are untouched).

``Resources`` = what is loaded once per server and shared by sessions:
    ears      the ears process (any number of sessions, each with its own language and state)
    llm       ``make_llm(session_id, lang, system_prompt) -> LLM plugin`` (Nano: one shared model, one worker per session)
    talker    ``make_talker(session_id, lang, voice) -> Talker plugin`` (Pocket: one process, per-session language/voice)
``Frankenstein2Engine`` = one session at a time on those resources (the V2 server's engine lock); all session state
lives in the ``Pipeline``, so a batching server can hold several.

Step: 1280 input samples (80 ms) -> the pipeline step -> output audio resampled 24 kHz -> 22.05 kHz (the V2 wire
rate), the user transcript delta, and BOS / EOS marks when an agent turn opens / closes (V2's turn.started).
Events go to the session tracer (frankenstein2.tracing) and, through ``attach_tracer``, to the V2 trace file and its
live listeners under their own names (the v1 BridgeTracer path).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

STEP = 1280
OUT_SR = 22050


@dataclass
class SessionOptions:
    language: str = "it"
    voice: str = ""


@dataclass
class Resources:
    ears: object
    make_llm: object
    make_talker: object
    languages: object = ("it", "en", "fr")            # a tuple, or a callable -> codes (the talker's package folder)
    voices: dict = field(default_factory=dict)          # name -> reference wav (or stock name)
    ears_thr: dict = field(default_factory=dict)        # language -> threshold overrides
    client_delay_steps: int = 1
    trace_dir: str | None = None
    sinks: list = field(default_factory=list)            # extra trace listeners (obs.TraceStore.sink, LogSink.sink)
    store_text: bool = True                              # False: user / agent text never reaches a trace or sink


def served_languages(res) -> tuple:
    """The languages a session may ask for: fixed, or what the talker's package folder holds right now (a folder
    added while the server runs is served from the next session on)."""
    return tuple(res.languages() if callable(res.languages) else res.languages)


def make_pipeline(res: Resources, sid: str, language: str, voice: str = "", system_prompt: str | None = None,
                  sink=None, trace: bool = True):
    """One session's Pipeline on the shared resources (language and voice fixed for the session)."""
    from frankenstein2.core import Pipeline, PipelineConfig
    from frankenstein2.ears.client import EarsLink
    from frankenstein2.tracing import Tracer
    served = served_languages(res)
    if language not in served:
        raise ValueError(f"language {language!r} not served; available {list(served)}")
    if voice and res.voices and voice not in res.voices:
        raise ValueError(f"unknown voice {voice!r}; available {sorted(res.voices)}")
    from frankenstein2.obs import fanout
    tr = Tracer(f"{res.trace_dir}/{sid}.f2.jsonl" if (res.trace_dir and trace) else None, session_id=sid, keep=False,
                sink=fanout(sink, *res.sinks), store_text=res.store_text)
    llm = res.make_llm(sid, language, system_prompt)
    talker = res.make_talker(sid, language, res.voices.get(voice, voice))
    ears = EarsLink(res.ears, sid, language, thr=res.ears_thr.get(language))
    p = Pipeline(sid, ears, llm, talker, PipelineConfig(lang=language, client_delay_steps=res.client_delay_steps), tr)
    p.open()
    orig = p.close

    def close():
        try:
            orig()
        finally:
            for x in (llm, talker):
                try:
                    x.close()
                except Exception:  # noqa: BLE001
                    pass
            tr.emit("engine.teardown", p.k, steps=p.k, cancels=len(p.stats["cancels"]),
                    endpoints=len(p.stats["endpoints"]), barges=len(p.stats["barges"]))
            tr.close()
    p.close = close
    return p


def _step_output():
    """V2's StepOutput; its module pulls VoiceChat's model stack (omegaconf, ...), which the one-process engine's
    venv does not carry: then a field-identical stand-in."""
    try:
        from mynah_v2.model.reference import StepOutput
        return StepOutput
    except ImportError:
        global _StepOutput
        if _StepOutput is None:
            from dataclasses import dataclass, field

            @dataclass
            class StepOutput:  # same fields as mynah_v2.model.reference.StepOutput
                audio: object
                text_delta: str
                asr_delta: str
                func_delta: str
                frame_idx: int
                new_text_ids: list
                new_func_ids: list
                fc_active: bool
                fc_records: list = field(default_factory=list)
                latency_ms: float = 0.0
            _StepOutput = StepOutput
        return _StepOutput


_StepOutput = None


class Frankenstein2Engine:
    chunk_samples = STEP

    def __init__(self, res: Resources):
        self.res = res
        self.pending = SessionOptions()
        self.v2tracer = None
        self.p = None
        self.n = 0
        self.wrapper = None

    def attach_tracer(self, tracer) -> None:
        self.v2tracer = tracer

    def set_session_options(self, **kw) -> None:
        self.pending = SessionOptions(**{k: v for k, v in kw.items() if v and k in ("language", "voice")})

    def fc_active(self) -> bool:
        return False

    def _bridge(self):
        t = self.v2tracer
        if t is None:
            return None
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "frankenstein"))
        from server.engine import BridgeTracer           # v1's bridge, read-only reuse
        b = BridgeTracer(t)

        def sink(name, d):
            # V2's tracer adds session_id / t_ns itself; send the fields only
            b.emit(name, d.get("model_step"), **{x: v for x, v in d.items()
                                                 if x not in ("ev", "t_ns", "session_id", "model_step")})
        return sink

    def open(self, system_prompt: str | None) -> None:
        from frankenstein2.core import Pipeline, PipelineConfig
        from frankenstein2.ears.client import EarsLink
        from frankenstein2.tracing import Tracer
        opts, self.pending = self.pending, SessionOptions()
        served = served_languages(self.res)
        if opts.language not in served:
            raise ValueError(f"language {opts.language!r} not served; available {list(served)}")
        if opts.voice and self.res.voices and opts.voice not in self.res.voices:
            raise ValueError(f"unknown voice {opts.voice!r}; available {sorted(self.res.voices)}")
        self.n += 1
        sid = getattr(self.v2tracer, "session_id", None) or f"f2-{self.n}"
        tr = Tracer(f"{self.res.trace_dir}/{sid}.f2.jsonl" if self.res.trace_dir else None, session_id=sid,
                    keep=False, sink=self._bridge())
        llm = self.res.make_llm(sid, opts.language, system_prompt)
        talker = self.res.make_talker(sid, opts.language, self.res.voices.get(opts.voice, opts.voice))
        ears = EarsLink(self.res.ears, sid, opts.language, thr=self.res.ears_thr.get(opts.language))
        from frankenstein2.audio.playout import Resampler
        self.rs = Resampler(talker.sr, OUT_SR)
        self.p = Pipeline(sid, ears, llm, talker, PipelineConfig(lang=opts.language,
                                                                 client_delay_steps=self.res.client_delay_steps), tr)
        self.tr, self.turn_open, self.frame = tr, False, 0
        self.p.open()

    def step(self, pcm16k, is_last: bool = False):
        import torch
        from mynah_v2.model import tokens as T
        StepOutput = _step_output()
        t0 = time.monotonic()
        x = pcm16k.detach().float().cpu().numpy() if hasattr(pcm16k, "detach") else np.asarray(pcm16k, np.float32)
        r = self.p.step(x)
        self.frame += 1
        marks = []
        speaking = r["state"] != "listening"
        if speaking and not self.turn_open:
            marks.append(T.BOS_ID)
            self.turn_open = True
        elif not speaking and self.turn_open:
            marks.append(T.EOS_ID)
            self.turn_open = False
        audio = torch.from_numpy(self.rs(r["audio"]).astype(np.float32))
        return StepOutput(audio=audio, text_delta="", asr_delta=r["asr_delta"], func_delta="",
                          frame_idx=self.frame, new_text_ids=marks, new_func_ids=[], fc_active=False,
                          latency_ms=(time.monotonic() - t0) * 1e3)

    def close(self) -> None:
        p, self.p = self.p, None
        if p is None:
            return
        p.close()
        try:
            p.llm.close()
            p.talker.close()
        finally:
            self.tr.emit("engine.teardown", p.k, steps=p.k, cancels=len(p.stats["cancels"]),
                         endpoints=len(p.stats["endpoints"]), barges=len(p.stats["barges"]))
            self.tr.close()
