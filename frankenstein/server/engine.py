"""The Frankenstein engine behind V2's Session / Engine contract (INTEGRATION.md step 1).

``FrankensteinEngine`` implements ``mynah_v2.runtime.session.Engine`` (``chunk_samples``, ``open``, ``step``,
``fc_active``, ``close``), so V2's ``Session`` and WebSocket server run it unchanged. Per session it owns:

  duplex Thinker (V2 ReferenceEngine, decode_audio off: timing shell)   + instance hooks of the controller driver
  live streaming ASR (e1/live_asr.py)                                     reset per session
  controller + driver (frankenstein_s2s/controller.py, e1/ctl_driver.py) new per session
  text LLM worker (e1/nano_worker.py, thread)                             new per session, joined on close
  talker adapter -> talker process (e1/flash_server.py, shared, reset per session: language + voice)
  output queue: talker PCM 24 kHz -> 22.05 kHz (the server's declared output rate), epoch-gated: a cancel drops
  everything not yet handed to the transport and emits ``audio.flush`` so a client can drop what it buffered.

Events go to the session's V2 tracer (``attach_tracer``): V2's own names where they exist (tool.* with call_id from
the ToolLedger, text.generated / asr.generated / turn.started from Session via the step output) plus the
controller / adapter / talker events under their own names; every event carries session_id (the V2 tracer adds it),
the controller's turn_id and request_id where they apply.

One engine serves one session at a time (the server's engine lock); sequential sessions share the loaded models and
must not leak state into each other (checked by the smoke test). Limits: ``max_session_s`` of input audio,
reply tokens (worker caps), talker frames (talker runaway cap), one text-LLM request in flight per session.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

TALKER_SR = 24000
OUT_SR = 22050


@dataclass
class SessionOptions:
    language: str = "it"
    voice: str = ""                      # "" = the talker's default voice; a clone name or a built-in voice


@dataclass
class _Out:
    """Model-free part of the output path: epoch gate + pending PCM + flush bookkeeping (unit-tested)."""
    epoch: int = 0
    pending: list = field(default_factory=list)
    flushed: int = 0
    stale: int = 0

    def audio(self, epoch: int, pcm) -> bool:
        if epoch != self.epoch:
            self.stale += 1
            return False
        self.pending.append(pcm)
        return True

    def cancel(self, new_epoch: int) -> int:
        n = sum(len(x) for x in self.pending)
        self.flushed += n
        self.pending = []
        self.epoch = new_epoch
        return n

    def take(self) -> list:
        out, self.pending = self.pending, []
        return out


class BridgeTracer:
    """frankenstein_s2s tracer interface (``emit(ev, k, **fields)``) onto a V2 tracer (``emit(name, **fields)``)."""

    def __init__(self, v2tracer=None):
        self.t = v2tracer

    def emit(self, ev: str, k=None, **fields) -> None:
        if self.t is None:
            return
        f = {x: v for x, v in fields.items() if v is not None}
        if k is not None:
            f["model_step"] = k
        if hasattr(self.t, "_subs") and hasattr(self.t, "_fh"):
            from mynah_v2.tracing import events as E
            if ev not in E.ALL_EVENTS:
                # V2's Tracer only accepts its canonical names (I1: the first custom event raised and the server
                # closed the session). Engine-specific events go through the SAME JSONL file and live listeners,
                # under their own names, without touching the V2 module.
                e = E.Event(name=ev, t_ns=self.t.clock(), session_id=self.t.session_id, fields=f)
                with self.t._lock:
                    if self.t.events is not None:
                        self.t.events.append(e)
                    if self.t._fh:
                        self.t._fh.write(e.to_json() + "\n")
                for fn in self.t._subs:
                    try:
                        fn(e)
                    except Exception:  # noqa: BLE001 - same policy as V2: a broken listener never breaks a session
                        pass
                return
        self.t.emit(ev, **f)

    def write(self, rec: dict) -> None:            # FrameLog-compatible (brain log) -> events too
        rec = dict(rec)
        ev = rec.pop("ev", "brain.log")
        self.emit(ev, rec.pop("k", None), **{x: v for x, v in rec.items() if isinstance(v, (str, int, float, bool))})

    def close(self) -> None:
        pass


class _NullLog:
    def write(self, rec):
        pass

    def close(self):
        pass


class FrankensteinEngine:  # pragma: no cover - GPU (the model-free parts are tested separately)
    def __init__(self, ref, tok, nano, spec, live, proc, lab_agentbench, policy: dict, brief: dict,
                 nano_graph=None, max_session_s: float = 1200.0, ids_per_frame: int = 8, holding: bool = True,
                 gate=None):
        import torch
        self.torch = torch
        self.ref, self.tok, self.nano, self.spec, self.live, self.proc = ref, tok, nano, spec, live, proc
        self.lab_agentbench, self.policy, self.brief = lab_agentbench, policy, brief
        self.nano_graph, self.ids_per_frame, self.holding = nano_graph, ids_per_frame, holding
        self.gate = gate                       # G7 GpuGate: frames and text-LLM units alternate on the GPU
        self.chunk_samples = ref.chunk_samples
        self.max_samples = int(max_session_s * 16000)
        self.buf = torch.zeros(self.max_samples)
        self.tracer = None
        self.pending_opts = SessionOptions()
        self.opts = None
        self.s = None                          # per-session objects
        self.lock = threading.Lock()
        self.sessions = 0
        self.wrapper = ref.wrapper             # Session / server look for a few engine attributes

    # ------------------------------------------------------------------ server hooks
    def attach_tracer(self, tracer) -> None:
        self.tracer = tracer

    def set_session_options(self, **kw) -> None:
        self.pending_opts = SessionOptions(**{k: v for k, v in kw.items() if v})

    def fc_active(self) -> bool:
        return False                           # tools run in the text-LLM worker, not in the Thinker's FC path

    # ------------------------------------------------------------------ lifecycle
    def open(self, system_prompt: str | None) -> None:
        import sys
        sys.path.insert(0, str(self.lab_agentbench))
        from m0_runner import render_prompt, set_model_cfg
        from tenant_backend import TenantBackend
        from mynah_v2.agent.tools import ToolLedger, ToolRegistry
        import llm_text_bench as B2
        from e1.ctl_driver import CtlDriver
        from e1.flash_client import Player
        from e1.nano_worker import NanoWorker
        from e1.tools_bench import tool_schemas
        from frankenstein_s2s.adapter import AdapterConfig, TalkerAdapter
        from frankenstein_s2s.controller import Controller, ControllerConfig
        from frankenstein_s2s.events import Cancel, TalkerDone
        from frankenstein_s2s.lang import session_language, with_language
        torch = self.torch
        opts, self.pending_opts = self.pending_opts, SessionOptions()
        L = session_language(opts.language)
        if L.talker_lang not in set(self.proc.ready.get("languages") or [L.talker_lang]):
            raise ValueError(f"the talker has no language {L.talker_lang!r}")
        voices = self.proc.ready.get("voices") or []
        if opts.voice and voices and opts.voice not in voices:
            raise ValueError(f"unknown voice {opts.voice!r}; available: {voices}")
        self.opts = opts
        tr = BridgeTracer(self.tracer)
        mem0 = torch.cuda.memory_allocated()
        self.proc.reset()
        self.proc.submit_raw({"op": "session", "lang": L.code, "voice": opts.voice})
        said: list = []
        backend = TenantBackend(self.spec, user_activity=lambda: len(said), user_transcript=lambda: said)
        backend._said = said
        reg = ToolRegistry()
        for t in self.spec["tools"]:
            reg.register(t["name"], backend.handlers()[t["name"]], t["description"], t["parameters"])
        led = ToolLedger(getattr(self.tracer, "session_id", "frank"), self.tracer)
        turn_of = {"ctl": None}                 # tool events carry the controller's turn id (I1b: they said 0)
        bound = reg.bind(led, lambda: turn_of["ctl"].turn if turn_of["ctl"] is not None else 0, lambda: None,
                         timeout_s=None)
        base = system_prompt or self.spec["system_prompt"]
        policy = self.policy.get(L.code) or self.policy["en"]
        system = with_language(base, L) + "\n" + policy + "\n" + self.brief.get(L.code, self.brief["en"]) + " /no_think"
        worker = NanoWorker(self.tok, self.nano, system, tool_schemas(reg), bound, backend, B2.parse_toolcall,
                            max_new=60, tool_step=True, constrained=True)
        worker.reply_soft_cap, worker.graph, worker.gate = 24, self.nano_graph, self.gate
        worker.graph_reply_only = self.nano_graph is not None      # the tool step always eager (G8)
        worker.start()
        out = _Out()
        player = Player(sr=TALKER_SR, prebuffer_s=0.0)
        import soxr
        rs = {"s": soxr.ResampleStream(TALKER_SR, OUT_SR, 1, dtype="float32", quality="HQ")}
        tracer = self.tracer
        sid = getattr(tracer, "session_id", None)

        class OutSink:
            def submit(self, cmd):
                if isinstance(cmd, Cancel):
                    n = out.cancel(cmd.new_epoch)
                    player.on_cancel(time.monotonic(), cmd.new_epoch)
                    rs["s"] = soxr.ResampleStream(TALKER_SR, OUT_SR, 1, dtype="float32", quality="HQ")
                    tr.emit("audio.flush", cmd.k, epoch=cmd.epoch, new_epoch=cmd.new_epoch, reason=cmd.reason,
                            dropped_samples=n)
        ad = TalkerAdapter([self.proc, OutSink()], decode=lambda ids: self.tok.decode(ids),
                           config=AdapterConfig(talker_lang=L.talker_lang, barge_in="thinker"), tracer=tr)
        ctl = Controller(ControllerConfig(min_blank=2, debounce=8, reply_timeout=100, tool_timeout=150,
                                          merge_frames=12, barge_guard=6, barge_in_frames=3, barge_after_open=True,
                                          holding_phrase=self.holding, audio_aware=True, barge_in_frames_audible=2),
                         tracer=tr)
        turn_of["ctl"] = ctl
        self.live.reset(window_s=12.0, every=8)
        self.live.lang_idx = __import__("e1.live_asr", fromlist=["LANG_IDX"]).LANG_IDX.get(L.code, 101)
        empty = {"words": [], "pieces": {}, "eou": []}
        d = CtlDriver(self.ref.wrapper, worker, self.tok, empty, ad, tr, _NullLog(), tr, ctl, lang=L.code,
                      live=self.live, wav=self.buf)
        d.fast_barge, d.holding_cut, d.ids_per_frame = True, "never", self.ids_per_frame
        d.audible = lambda: player.audible(time.monotonic()) or bool(ad.busy)
        state = {"spoken": 0, "words": 0}

        def pump():
            for ev in self.proc.poll():
                kind = ev.get("ev")
                if kind == "audio" and ev["epoch"] == ad.epoch:
                    pcm = ev["pcm"]
                    player.on_audio(time.monotonic(), ev["epoch"], ev["turn"], ev["seg"], ev["a"], ev["b"], len(pcm))
                    if out.audio(ev["epoch"], rs["s"].resample_chunk(pcm)):
                        tr.emit("talker.chunk", None, turn_id=ev["turn"], segment=ev["seg"], epoch=ev["epoch"],
                                frames=ev["b"] - ev["a"])
                elif kind == "audio":
                    out.stale += 1
                elif kind == "done":
                    ad.talker_done(TalkerDone(ev["epoch"], ev["turn"], ev["seg"]))
        d.pump, d.pump_every = pump, 1
        old = set_model_cfg(self.ref.wrapper.model_cfg, "force_turn_taking", False)
        uninstall = d.install()
        self.ref.bind_session_tools({})
        duplex_prompt = render_prompt(reg.schemas, with_language(self.spec["system_prompt"], L))
        self.s = dict(worker=worker, ad=ad, ctl=ctl, d=d, out=out, player=player, uninstall=uninstall, old=old,
                      led=led, state=state, pos=0, mem0=mem0, opened=time.monotonic(), turn_open=False,
                      set_model_cfg=set_model_cfg, said=said)
        self.ref.open(duplex_prompt)
        self.sessions += 1
        tr.emit("engine.session", None, engine="frankenstein", language=L.code, talker_lang=L.talker_lang,
                voice=opts.voice or self.proc.ready.get("voice"), session_index=self.sessions,
                voices=list(self.proc.ready.get("voices") or []),
                cuda_allocated_gb=round(mem0 / 2 ** 30, 2))

    def step(self, pcm16k, is_last: bool = False):
        from mynah_v2.model import tokens as T
        from mynah_v2.model.reference import StepOutput
        torch = self.torch
        s = self.s
        t0 = time.monotonic()
        if s["pos"] + pcm16k.numel() > self.max_samples:
            raise RuntimeError("session longer than max_session_s")
        self.buf[s["pos"]: s["pos"] + pcm16k.numel()] = pcm16k
        s["pos"] += pcm16k.numel()
        n_words = sum(len(v) for v in s["d"].words_at.values())
        n_spoken = len(s["d"].handed)
        if self.gate is not None:
            with self.gate.frame():
                r = self.ref.step(pcm16k, is_last=is_last)
        else:
            r = self.ref.step(pcm16k, is_last=is_last)
        d = s["d"]
        new_ids = d.handed[n_spoken:]
        text_delta = self.tok.decode(new_ids) if new_ids else ""
        words = [w for k in sorted(d.words_at) for w in d.words_at[k]][n_words:]
        marks = []
        st = s["ctl"].state.value
        if st != "listening" and not s["turn_open"]:
            marks.append(T.BOS_ID)
            s["turn_open"] = True
        elif st == "listening" and s["turn_open"]:
            marks.append(T.EOS_ID)
            s["turn_open"] = False
        pcm = s["out"].take()
        import numpy as np
        audio = torch.from_numpy(np.concatenate(pcm).astype(np.float32)) if pcm else torch.zeros(0)
        return StepOutput(audio=audio, text_delta=text_delta, asr_delta=" ".join(words), func_delta="",
                          frame_idx=r.frame_idx, new_text_ids=marks, new_func_ids=[], fc_active=False,
                          latency_ms=(time.monotonic() - t0) * 1e3)

    def close(self) -> None:
        s, self.s = self.s, None
        if s is None:
            return
        torch = self.torch
        try:
            s["ad"].cancel(0, "session_close", audible=True)
        except Exception:  # noqa: BLE001
            pass
        s["worker"].close(timeout=30)
        s["uninstall"]()
        s["set_model_cfg"](self.ref.wrapper.model_cfg, "force_turn_taking", s["old"])
        self.ref.close()
        self.proc.reset()
        self.live.reset()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        mem1 = torch.cuda.memory_allocated()
        BridgeTracer(self.tracer).emit(
            "engine.teardown", None, worker_alive=s["worker"].is_alive(), out_flushed=s["out"].flushed,
            out_stale=s["out"].stale, cuda_allocated_gb_open=round(s["mem0"] / 2 ** 30, 3),
            cuda_allocated_gb_close=round(mem1 / 2 ** 30, 3), cuda_delta_mb=round((mem1 - s["mem0"]) / 2 ** 20, 1),
            calls=[c["name"] for c in s["worker"].calls], session_s=round(time.monotonic() - s["opened"], 1))
