"""G0: the first real speaking loop. The E1l controller session with everything live:

  user audio (the 52 s Italian track, paced in real time)
    -> duplex Thinker (INT4, timing shell) + LIVE multilingual streaming ASR (e1/live_asr.py, GPU, causal)
    -> controller -> Nano worker (bf16, tool step, constrained calls, holding phrase)
    -> talker adapter -> the REAL Sophon Flash talker in its own process (e1/flash_server.py, zero-shot clone voice)
    -> Player (e1/flash_client.py): what a client would hear, on the wall clock.

Outputs in the run dir: the usual traces (frames, brain, ctl, adapter, V2 trace), talker events (talker.jsonl),
agent channel agent_24k.wav, user + agent stereo mix mix_stereo_24k.wav, and g0.json with the measurements.
"""

from __future__ import annotations

import json
import time
from pathlib import Path


def _q(xs, p):
    xs = sorted(xs)
    return round(xs[int(p * (len(xs) - 1))], 2) if xs else None


def _stats(xs):
    return {"n": len(xs), "p50": _q(xs, .5), "p95": _q(xs, .95), "mean": round(sum(xs) / len(xs), 2) if xs else None,
            "max": round(max(xs), 2) if xs else None}


BRIEF_IT = ("Rispondi come in una telefonata: al massimo due frasi brevi, senza elenchi puntati, senza markdown e "
            "senza simboli; dì i prezzi a parole.")


def replay_player(log, prebuffer_s, sr):
    from e1.flash_client import Player
    p = Player(sr=sr, prebuffer_s=prebuffer_s)
    for e in log:
        if e["kind"] == "audio":
            p.on_audio(e["t"], e["epoch"], e["turn"], e["seg"], e["a"], e["b"], e["n"])
        else:
            p.on_cancel(e["t"], e["new_epoch"])
    return {"prebuffer_s": prebuffer_s, "gaps": len(p.gaps), "gap_total_s": round(sum(g["gap_s"] for g in p.gaps), 3)}


def run(out: Path, od_name: str, eng, tok, model, spec, policy_it: str, asr_dir: Path, cut_s: float,
        lab_agentbench: Path, tool_schemas, live, proc, timer=None, prebuffer_s: float = 0.32,
        listen_dir: Path | None = None, drain_s: float = 15.0, holding_cut: str = "never",
        audio_aware: bool = False, reactive: bool = False, keep_proc: bool = False, brief: bool = False,
        fast_barge: bool = False, nano_graph=None, ids_per_frame: int = 1, gate=None) -> dict:  # pragma: no cover - GPU
    import sys
    import numpy as np
    import soundfile as sf
    sys.path.insert(0, str(lab_agentbench))
    from m0_runner import load_16k, render_prompt, set_model_cfg
    from tenant_backend import TenantBackend
    from mynah_v2.agent.tools import ToolLedger, ToolRegistry
    from mynah_v2.runtime.session import Mode, Session, run_paced
    from mynah_v2.tracing.events import Tracer as V2Tracer
    from e1.ctl_driver import CtlDriver
    from e1.ctl_session import _spoken_turns
    from e1.flash_client import Player
    from e1.m1force import FrameLog
    from e1.nano_worker import NanoWorker
    from e1.runner import adapter_summary, seed_all
    from frankenstein_s2s.adapter import AdapterConfig, TalkerAdapter
    from frankenstein_s2s.controller import Controller, ControllerConfig
    from frankenstein_s2s.events import Cancel, TalkerDone
    from frankenstein_s2s.lang import session_language, with_language
    from frankenstein_s2s.tracing import Tracer as FTracer
    w = eng.wrapper
    L = session_language("it")
    frozen = json.load(open(asr_dir / "asr_it.json"))
    wav = load_16k(asr_dir / "track_it.wav")[: int(cut_s * 16000)]
    track = wav
    if reactive:                                     # G3b arm B: user audio produced live, written into this buffer
        import torch
        wav = torch.zeros(int(240 * 16000))
    od = out / od_name
    od.mkdir()
    said = []
    backend = TenantBackend(spec, user_activity=lambda: len(said), user_transcript=lambda: said)
    backend._said = said
    reg = ToolRegistry()
    for t in spec["tools"]:
        reg.register(t["name"], backend.handlers()[t["name"]], t["description"], t["parameters"])
    tr = V2Tracer(od / "trace.jsonl", session_id=od_name)
    led = ToolLedger(od_name, tr)
    bound = reg.bind(led, lambda: 0, lambda: None, timeout_s=None)
    duplex_prompt = render_prompt(reg.schemas, with_language(spec["system_prompt"], L))
    system = spec["system_prompt"] + "\n" + policy_it + (("\n" + BRIEF_IT) if brief else "") + " /no_think"
    worker = NanoWorker(tok, model, system, tool_schemas(reg), bound, backend, None, max_new=80, tool_step=True,
                        constrained=True)
    import llm_text_bench as B2
    worker.parse = B2.parse_toolcall
    worker.graph = nano_graph                        # G6: Nano decode CUDA graph (None = eager)
    worker.gate = gate                               # G7: frame loop / text LLM time-slicing (None = off)
    if brief:                                        # G4: short spoken replies (prompt + sentence-aware cap)
        worker.max_new, worker.reply_soft_cap = 60, 24
    worker.start()
    ftr = FTracer(od / "fr_trace.jsonl", session_id=od_name)
    player = Player(sr=24000, prebuffer_s=prebuffer_s)
    plog: list = []
    tlog = FrameLog(od / "talker.jsonl")

    class PlayerSink:
        def submit(self, cmd):
            if isinstance(cmd, Cancel):
                t = time.monotonic()
                player.on_cancel(t, cmd.new_epoch)
                plog.append({"kind": "cancel", "t": t, "new_epoch": cmd.new_epoch})
    ad = TalkerAdapter([proc, PlayerSink()], decode=lambda ids: tok.decode(ids),
                       config=AdapterConfig(talker_lang=L.talker_lang, barge_in="thinker"), tracer=ftr)

    def pump():
        for ev in proc.poll():
            kind = ev.get("ev")
            if kind == "audio":
                t = ev["t_ns"] / 1e9
                n = len(ev["pcm"])
                it = player.on_audio(t, ev["epoch"], ev["turn"], ev["seg"], ev["a"], ev["b"], n, pcm=ev["pcm"])
                plog.append({"kind": "audio", "t": t, "epoch": ev["epoch"], "turn": ev["turn"], "seg": ev["seg"],
                             "a": ev["a"], "b": ev["b"], "n": n})
                tlog.write({"ev": "audio", **{k: v for k, v in ev.items() if k != "pcm"}, "n": n,
                            "t_play": it.t_start if it else None, "stale": it is None})
            elif kind == "done":
                ad.talker_done(TalkerDone(ev["epoch"], ev["turn"], ev["seg"]))
                tlog.write(ev)
            elif kind in ("cancelled", "stats", "eof"):
                tlog.write({k: v for k, v in ev.items() if k not in ("step_ms", "decode_ms")})
    ctl = Controller(ControllerConfig(min_blank=2, debounce=8, reply_timeout=100, tool_timeout=150, merge_frames=12,
                                      barge_guard=6, barge_in_frames=3, barge_after_open=True, holding_phrase=True,
                                      barge_in_frames_audible=2 if fast_barge else 0,
                                      audio_aware=audio_aware),
                     tracer=ftr)
    frames, blog = FrameLog(od / "frames.jsonl"), FrameLog(od / "brain.jsonl")
    seed_all()
    empty = {"words": [], "pieces": {}, "eou": []}
    d = CtlDriver(w, worker, tok, empty, ad, ftr, frames, blog, ctl, lang="it", live=live, wav=wav)
    d.pump, d.pump_every = pump, 1
    d.holding_cut = holding_cut
    d.fast_barge = fast_barge
    d.ids_per_frame = ids_per_frame
    d.audible = lambda: player.audible(time.monotonic()) or bool(ad.busy)
    old = set_model_cfg(w.model_cfg, "force_turn_taking", False)
    uninstall = d.install()
    eng.bind_session_tools({})
    if timer is not None:
        timer.reset()
    step_orig = None
    if gate is not None:                             # G7: every engine chunk is one frame-side unit of the gate
        step_orig = eng.step

        def gated_step(*a_, **kw_):
            with gate.frame():
                return step_orig(*a_, **kw_)
        eng.step = gated_step
    sess = Session(eng, tr, Mode.REFERENCE, None, on_audio=lambda x: None, controller=None)
    t_wall0 = time.monotonic()
    try:
        sess.open(duplex_prompt)
        t0 = time.monotonic()                       # user-audio sample 0 is "played" now
        if reactive:
            ru = _reactive_feed(sess, eng, wav, track, frozen, player, ad, ctl, d, holding_turns_fn=lambda: {
                h["adapter_turn"] for h in d.holding if h["ok"]})
            wav = wav[: ru.sent]
        else:
            run_paced(sess, wav, pace="realtime", tail_silence_s=1.0, max_extra_s=10.0, drain=True)
        t_end_loop = time.monotonic()
        sess.close("normal")
    finally:
        uninstall()
        if step_orig is not None:
            del eng.step                             # back to the class method
        tr.close()
        set_model_cfg(w.model_cfg, "force_turn_taking", old)
        worker.close(timeout=30)
    # let the talker finish what the adapter already sent (bounded)
    t_drain = time.monotonic()
    while time.monotonic() - t_drain < drain_s and (ad.busy or not proc.events.empty()):
        pump()
        time.sleep(0.05)
    pump()
    drained_s = round(time.monotonic() - t_drain, 1)
    for ev in (proc.reset() if keep_proc else proc.close()):   # stats arrive with quit / reset (G0: read too early)
        if ev.get("ev") not in ("audio",):
            tlog.write({k: v for k, v in ev.items() if k not in ("step_ms", "decode_ms", "pcm")})
    frames.close()
    blog.close()
    holding_turns = {h["adapter_turn"] for h in d.holding if h["ok"]}
    rec = dict(d.summary(), arm=od_name, lang="it", cut_s=cut_s, calls=worker.calls, backend_audit=backend.audit,
               ledger=[{"name": c.name, "status": c.status.value} for c in led.calls.values()],
               adapter={k: v for k, v in adapter_summary(ad, _NoGate()).items() if k != "gate"},
               wall_s=round(time.monotonic() - t_wall0, 1),
               comp_timer=timer.report() if timer is not None else None, worker_alive_after_close=worker.is_alive(),
               spoken_turns=_spoken_turns(ad, tok, holding_turns), holding_on=True, constrained=True,
               live_asr_pieces={str(k): v for k, v in live.per_frame.items()})
    ftr.close()
    tlog.close()
    (od / "summary.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False, default=str))

    # ---------------------------------------------------------------- measurements
    fr = [json.loads(x) for x in open(od / "frames.jsonl")]
    dt = [(b["t_ns"] - a["t_ns"]) / 1e6 for a, b in zip(fr, fr[1:])]
    frame_k_t = {r["k"]: r["t_ns"] / 1e9 for r in fr}
    ftl = {u["label"]: (i, u) for i, u in enumerate(frozen["timeline"])}
    tl = [{"label": x["label"], "start_k": x["start_k"], "end_k": x["end_k"]} for x in ru.timeline
          if "end_k" in x] if reactive else frozen["timeline"]
    # live vs frozen words, per user turn (frozen words in the frozen track's window of the same utterance)
    live_words = [{"k": k, "word": wd} for k, ws in sorted(d.words_at.items()) for wd in ws]
    by_turn = {}
    for i, u in enumerate(tl):
        hi = tl[i + 1]["start_k"] if i + 1 < len(tl) else 10 ** 9
        lw = [x for x in live_words if u["start_k"] <= x["k"] < hi]
        fi, fu = ftl[u["label"]]
        fhi = frozen["timeline"][fi + 1]["start_k"] if fi + 1 < len(frozen["timeline"]) else 10 ** 9
        fw = [x for x in frozen["words"] if fu["start_k"] <= x["k"] < fhi]
        by_turn[u["label"]] = {"live": " ".join(x["word"] for x in lw), "frozen": " ".join(x["word"] for x in fw),
                               "same_text": " ".join(x["word"] for x in lw) == " ".join(x["word"] for x in fw),
                               "last_word_k": {"live": lw[-1]["k"] if lw else None,
                                               "frozen": fw[-1]["k"] if fw else None},
                               "end_k": u["end_k"]}
    # first audio after the end of each user utterance: holding phrase vs real reply, played time
    pt = player.per_turn()
    first_audio = {}
    for u in tl:
        t_end_user = t0 + (u["end_k"] + 1) * 0.08
        nxt = [(v["first_play"], turn) for turn, v in pt.items() if v["first_play"] > t_end_user]
        hold = [x for x in nxt if x[1] in holding_turns]
        real = [x for x in nxt if x[1] not in holding_turns]
        first_audio[u["label"]] = {
            "any_s": round(min(nxt)[0] - t_end_user, 3) if nxt else None,
            "holding_s": round(min(hold)[0] - t_end_user, 3) if hold else None,
            "real_reply_s": round(min(real)[0] - t_end_user, 3) if real else None}
    # processing clock (G0 analysis): latencies from the wall time the loop PROCESSED the user's end frame, which
    # excludes the loop's accumulated lag behind real time; and the lag itself
    first_play, first_ready = {}, {}
    for it in player.items:
        if not it.dropped and it.keep != 0:
            first_play.setdefault(it.turn, it.t_start)
            first_ready.setdefault(it.turn, it.t_ready)
    seg_k = {}
    for c in ad.commands:
        if type(c).__name__ == "StartSegment":
            seg_k.setdefault(c.turn, c.k)
    first_audio_proc, lag = {}, {}
    for u in tl:
        e = u["end_k"]
        if e not in frame_k_t:
            continue
        lag[u["label"]] = round(frame_k_t[e] - (t0 + (e + 1) * 0.08), 2)
        row = {}
        for kind, want in (("holding", True), ("reply", False)):
            cand = sorted((k, turn) for turn, k in seg_k.items() if k > e and (turn in holding_turns) == want)
            if cand:
                k, turn = cand[0]
                row[kind] = {"turn": turn, "text_s": round(frame_k_t.get(k, 0) - frame_k_t[e], 2),
                             "first_play_s": round(first_play[turn] - frame_k_t[e], 2) if turn in first_play else None}
        first_audio_proc[u["label"]] = row
    last_k = max(frame_k_t)
    lag["end"] = round(frame_k_t[last_k] - (t0 + (last_k + 1) * 0.08), 2)
    # barge-in: user onset -> cancel (= playback stop in the model)
    barge = []
    for a in d.ctl.actions:
        if a.kind == "CANCEL" and a.reason == "barge_in":
            onset = max((u for u in tl if u["start_k"] <= a.k), key=lambda u: u["start_k"], default=None)
            # the playback cut of THIS cancel (same frame); none if no Cancel reached the player (G5 U2)
            cut = next((c for c in player.cuts if abs(c["t"] - frame_k_t.get(a.k, -1e9)) < 1.0), None)
            t_proc_onset = frame_k_t.get(onset["start_k"]) if onset else None
            barge.append({"k": a.k, "utt": onset["label"] if onset else None,
                          "onset_to_cancel_frames": a.k - onset["start_k"] if onset else None,
                          "proc_onset_to_audio_stop_s": round(cut["t"] - t_proc_onset, 3) if cut and t_proc_onset
                          else None,
                          "onset_to_cancel_s": round(cut["t"] - (t0 + onset["start_k"] * 0.08), 3) if onset and cut
                          else None, "audio_was_playing": cut["was_playing"] if cut else None})
    ts = proc.stats or {}
    g0 = {"frames": {"n": len(fr), "mean_ms": round(sum(dt) / len(dt), 1) if dt else None, **{
              f"{k}_ms": v for k, v in _stats(dt).items() if k in ("p50", "p95", "max")}},
          "loop_s": round(t_end_loop - t0, 1), "audio_s": round(len(wav) / 16000, 1),
          "asr": {"calls": live.stats["calls"], "ms": _stats(live.stats["ms"]), "resets": live.stats["resets"],
                  "algorithmic_lookahead_s": round(live.chunk * 0.08, 2), "by_turn": by_turn},
          "talker": {"ready": proc.ready, "step_ms": _stats(ts.get("step_ms", [])),
                     "prefill_ms": _stats(ts.get("prefill_ms", [])), "decode_ms": _stats(ts.get("decode_ms", [])),
                     "segments": ts.get("segments"), "cancels": ts.get("cancels")},
          "playback": {"prebuffer_s": prebuffer_s, "gaps": len(player.gaps),
                       "gap_total_s": round(sum(g["gap_s"] for g in player.gaps), 3), "gap_list": player.gaps[:40],
                       "stale_dropped": len(player.stale), "cuts": player.cuts,
                       "replay": [replay_player(plog, pb, 24000) for pb in (0.0, 0.16, 0.32, 0.64, 1.0, 1.5)]},
          "first_audio_after_user_end": first_audio, "first_audio_processing_clock": first_audio_proc,
          "loop_lag_s": lag, "barge_in": barge, "reactive": reactive, "brief": brief, "fast_barge": fast_barge,
          "gate": dict(gate.stats) if gate is not None else None,
          "user_timeline": tl, "reactive_waits": ru.waits if reactive else None, "drained_s": drained_s, "audio_aware": audio_aware,
          "holding": d.holding}
    (od / "g0.json").write_text(json.dumps(g0, indent=1, ensure_ascii=False, default=str))
    # ---------------------------------------------------------------- listening files
    total = max(len(wav) / 16000, (player.end or t0) - t0) + 0.5
    agent = player.render(t0, total)
    from scipy.signal import resample_poly
    user24 = resample_poly(np.asarray(wav, dtype=np.float32), 3, 2).astype(np.float32)
    n = max(len(agent), len(user24))
    st = np.zeros((n, 2), np.float32)
    st[: len(user24), 0] = user24
    st[: len(agent), 1] = agent
    sf.write(od / "agent_24k.wav", agent, 24000)
    sf.write(od / "user_fed_16k.wav", np.asarray(wav, dtype=np.float32), 16000)
    sf.write(od / "mix_stereo_24k.wav", st, 24000)
    if listen_dir is not None:
        import shutil
        listen_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(od / "agent_24k.wav", listen_dir / "agent_24k.wav")
        shutil.copy(od / "mix_stereo_24k.wav", listen_dir / "mix_stereo_24k.wav")
    return {"summary": rec, "g0": g0}


class _NoGate:
    dropped_stale = flushed_frames = played_frames = 0


def _reactive_feed(sess, eng, buf, track, frozen, player, ad, ctl, d, holding_turns_fn,
                   max_s: float = 200.0):  # pragma: no cover - GPU
    """G3b arm B: pace the reactive user's chunks in real time (as run_paced does), writing each chunk into ``buf``
    before the engine sees it (the live ASR reads the same buffer)."""
    import torch
    from e1.reactive_user import ReactiveUser, Utt
    from frankenstein_s2s.controller import St
    tl = frozen["timeline"]
    utts = [Utt(u["label"], track[u["start_k"] * 1280:(u["end_k"] + 1) * 1280], barge=(i == len(tl) - 1))
            for i, u in enumerate(tl)]
    gen = torch.Generator().manual_seed(0)

    class World:
        def audible(self, now):
            return player.audible(now)

        def agent_idle(self, now):
            return ctl.state is St.LISTENING and not ad.busy and not player.audible(now)

        def audio_starts_after(self, t, real_only):
            hold = holding_turns_fn() if real_only else set()
            xs = [it.t_start for it in player.items if not it.dropped and it.t_start > t and it.turn not in hold]
            return min(xs) if xs else None
    cs = eng.chunk_samples
    ru = ReactiveUser(utts, chunk_samples=cs, noise=lambda n: torch.randn(n, generator=gen) * 3e-4,
                      cat=lambda xs: torch.cat([torch.as_tensor(x, dtype=torch.float32) for x in xs]))
    w = World()
    dur = cs / 16000
    t0 = time.monotonic()
    c = 0
    while not ru.done and c * dur < max_s and (c + 1) * cs <= buf.numel():
        wait = t0 + c * dur - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        ch = ru.next_chunk(time.monotonic(), w)
        buf[c * cs:(c + 1) * cs] = ch
        sess.feed_chunk(ch)
        c += 1
    sess.drain_fc(post_fc_s=1.0, max_extra_s=2.0, realtime=True)
    sess.flush(0.0)
    ru.sent = c * cs
    return ru
