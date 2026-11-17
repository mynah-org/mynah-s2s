"""E1 runner (GPU job): M1-force vs M0 vs text cascade, Thinker + frozen multilingual-ASR tracks, no talker.

Arms (identical user track, seeds and frame clock):
  m1f  V2 ReferenceEngine (NVIDIA native pipeline, decode_audio=False, REFERENCE mode, realtime pacing) with the M1-force
       hooks (e1/m1force.py); brain = the same Thinker modules on the text path with their own cache; tools through
       the brain (T1) and the V2 ToolRegistry / ToolLedger handlers (tenant mock backend).
  m0   the same engine and session, observer hook only (no write): the shipped duplex behaviour, its own FC path.
  cas  no duplex loop: the brain answers at each ASR end-of-utterance; one id per frame to the adapter.
The TalkerAdapter (frankenstein_s2s) runs on CPU on every arm's agent stream with the fake talker + PlaybackGate.
Everything goes to a NEW --out directory; nothing is overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
sys.path.insert(0, str(PKG))

from frankenstein_s2s import tokens as T  # noqa: E402
from frankenstein_s2s.adapter import AdapterConfig, TalkerAdapter  # noqa: E402
from frankenstein_s2s.driver import FakeTalkerDriver, PlaybackGate  # noqa: E402
from frankenstein_s2s.events import Frame  # noqa: E402
from frankenstein_s2s.forcing import AgentChannelForcer, ForceState  # noqa: E402
from frankenstein_s2s.lang import session_language, with_language  # noqa: E402
from frankenstein_s2s.tracing import Tracer as FTracer  # noqa: E402

SEED = 1234


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def mem_available_gb() -> float:
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024 / 1024
    return -1.0


def seed_all():
    import numpy as np
    import torch
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


def make_adapter(tok, ftr, talker_lang):
    drv, gate = FakeTalkerDriver(tracer=ftr), PlaybackGate(tracer=ftr)
    ad = TalkerAdapter([drv, gate], decode=lambda ids: tok.decode(ids),
                       config=AdapterConfig(talker_lang=talker_lang, barge_in="immediate"), tracer=ftr)
    return ad, drv, gate


def pump(ad, drv, gate):
    drv.produce()
    for c in drv.take():
        gate.accept(c)
    for d in drv.done:
        ad.talker_done(d)
    drv.done.clear()
    gate.play_all()


def adapter_summary(ad, gate):
    cmds = [{"type": type(c).__name__, **{k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(c).items()}}
            for c in ad.commands]
    return {"commands": cmds, "log": [vars(e) for e in ad.log], "epoch": ad.epoch,
            "gate": {"dropped_stale": gate.dropped_stale, "flushed_frames": gate.flushed_frames,
                     "played_frames": gate.played_frames}}


def main():  # pragma: no cover - GPU
    import torch
    from transformers import AutoTokenizer
    ap = argparse.ArgumentParser()
    ap.add_argument("--asr-dir", required=True)
    ap.add_argument("--lab-agentbench", required=True, help="labs/multilingual/scripts/agentbench")
    ap.add_argument("--tokenizer", required=True, help="Nemotron Nano 9B v2 snapshot (chat template)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--langs", default="it,fr,en")
    ap.add_argument("--arms", default="m1f,m0,cas")
    ap.add_argument("--budget-s", type=float, default=720.0, help="do not START a session past this (job wall)")
    ap.add_argument("--decode-audio", action="store_true",
                    help="keep NVIDIA's EarTTS stage running (its audio is discarded). Needed on the streaming pipeline: "
                         "with decode_audio=False its log_output() fails on audio_wave=None (E1 job 3)")
    ap.add_argument("--smoke-s", type=float, default=20.0,
                    help="first run M1-force on the first N s of the first language and stop the job if the "
                         "TECHNICAL invariants fail (0 = no smoke)")
    a = ap.parse_args()
    T0 = time.monotonic()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, a.lab_agentbench)
    from tenant_backend import TenantBackend
    from m0_runner import load_16k, render_prompt
    from mynah_v2.agent.tools import ToolLedger, ToolRegistry
    from mynah_v2.model.reference import ReferenceEngine, build_cfg
    from mynah_v2.runtime.session import Mode, Session, run_paced
    from mynah_v2.tracing.events import Tracer as V2Tracer
    from e1.brain import TextBrain, fingerprint
    from e1.m1force import FrameLog, M1Force, UserSide, OPENER_USER
    spec = json.load(open(Path(a.lab_agentbench) / "tenant_furniture.json"))
    asr_dir = Path(a.asr_dir)

    # ------------------------------------------------------------------------------------------- freeze
    harness = {str(p.relative_to(PKG)): sha256_file(p) for p in sorted(PKG.rglob("*.py")) if "__pycache__" not in str(p)}
    freeze = {"commit": os.environ.get("FRANK_COMMIT"), "harness_sha256": harness,
              "lab_files_sha256": {n: sha256_file(Path(a.lab_agentbench) / n)
                                   for n in ("tenant_backend.py", "m0_runner.py", "tenant_furniture.json")},
              "asr_inputs": json.load(open(asr_dir / "inputs.json")),
              "asr_tracks": {p.name: sha256_file(p) for p in sorted(asr_dir.glob("asr_*.json"))},
              "seed": SEED, "argv": sys.argv, "mem_available_gb_before_load": round(mem_available_gb(), 1)}
    if freeze["mem_available_gb_before_load"] < 70:
        (out / "freeze.json").write_text(json.dumps(freeze, indent=1))
        raise SystemExit(f"not enough free memory to load: {freeze['mem_available_gb_before_load']} GB")

    # ------------------------------------------------------------------------------------------- load
    ov = {} if a.decode_audio else {"s2s.decode_audio": False}
    eng = ReferenceEngine(build_cfg(output_dir=str(out / "_nemo"), overrides=ov), llm_cache=True)
    w = eng.wrapper
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    freeze.update({"engine_load_s": round(eng.load_s, 1), "decode_audio": bool(w.decode_audio),
                   "checkpoint": str(eng.cfg.s2s.model_path), "tokenizer": a.tokenizer,
                   "v2_src": str(Path(sys.modules["mynah_v2"].__file__).parent),
                   "rnnt_loaded": getattr(w.model.stt_model, "_rnnt_decoder", None) is not None,
                   "mem_available_gb_after_load": round(mem_available_gb(), 1)})
    (out / "freeze.json").write_text(json.dumps(freeze, indent=1, default=str))
    print("LOADED", json.dumps({k: freeze[k] for k in ("engine_load_s", "decode_audio", "rnnt_loaded",
                                                         "mem_available_gb_after_load")}), flush=True)
    brain = TextBrain(eng, tok, "")
    durations = []

    def budget_ok(est):
        return time.monotonic() - T0 + est <= a.budget_s

    langs = a.langs.split(",")
    arms = a.arms.split(",")
    plan = [(langs[0], "m1f", a.smoke_s, "smoke")] if a.smoke_s else []
    plan += [(lang, arm, 0.0, "") for lang in langs for arm in arms]
    status = {"done": [], "skipped": []}
    for lang, arm, cut_s, tag in plan:
        L = session_language(lang)
        asr = json.load(open(asr_dir / f"asr_{lang}.json"))
        wav = load_16k(asr_dir / f"track_{lang}.wav")
        if cut_s:
            wav = wav[: int(cut_s * 16000)]
        if True:
            est = (max(durations) * 1.1) if durations else (wav.numel() / 16000 + 30)
            if not tag and not budget_ok(est):
                status["skipped"].append([lang, arm])
                print("BUDGET: skip", lang, arm, flush=True)
                continue
            sid = f"{tag}_{arm}_{lang}" if tag else f"{arm}_{lang}"
            od = out / sid
            od.mkdir()
            backend = TenantBackend(spec, user_activity=lambda: 0, user_transcript=lambda: [])
            reg = ToolRegistry()
            for t in spec["tools"]:
                reg.register(t["name"], backend.handlers()[t["name"]], t["description"], t["parameters"])
            prompt = render_prompt(reg.schemas, with_language(spec["system_prompt"], L))
            brain.system = prompt + " /no_think"
            ftr = FTracer(od / "fr_trace.jsonl", session_id=sid)
            ad, drv, gate = make_adapter(tok, ftr, L.talker_lang)
            frames, blog = FrameLog(od / "frames.jsonl"), FrameLog(od / "brain.jsonl")
            seed_all()
            t_s = time.monotonic()
            if arm in ("m1f", "m0"):
                tr = V2Tracer(od / "trace.jsonl", session_id=sid)
                led = ToolLedger(sid, tr)
                state = {"s": None}
                bound = reg.bind(led, lambda: state["s"].turn_id if state["s"] else 0,
                                 lambda: state["s"].model_step if state["s"] else None, timeout_s=None)
                eng.bind_session_tools(dict(bound))

                def execute_tool(parsed, _bound=bound):
                    t0 = time.monotonic()
                    if not parsed or parsed.get("name") not in _bound:
                        res = json.dumps({"error": f"unknown or unparsable call: {parsed}"})
                    else:
                        res = _bound[parsed["name"]](parsed.get("arguments") or {})
                    return res, round((time.monotonic() - t0) * 1e3, 1)
                hook = M1Force(w, brain, tok, asr, ad, ftr, frames, blog, execute_tool,
                               mode="force" if arm == "m1f" else "observe")
                hook.pump = lambda: pump(ad, drv, gate)
                uninstall = hook.install()
                sess = Session(eng, tr, Mode.REFERENCE, None, on_audio=lambda x: None, controller=None)
                state["s"] = sess
                try:
                    sess.open(prompt)
                    run_paced(sess, wav, pace="realtime", tail_silence_s=2.0, max_extra_s=20.0, drain=True)
                    pump(ad, drv, gate)
                    sess.close("normal")
                finally:
                    uninstall()
                    tr.close()
                v2_text = "".join(sess.text)
                rec = {"arm": arm, "lang": lang, "agent_text_v2": v2_text, "hook_calls": hook.n_hook,
                       "frames_last": hook.last_t, "mismatch": hook.mismatch[:50], "n_mismatch": len(hook.mismatch),
                       "gaps": hook.gaps[:50], "n_gaps": len(hook.gaps),
                       "fc_suppressed": hook.fc_suppressed, "cache_checks": hook.cache_checks,
                       "bos_suppressed": hook.suppressed, "brain_skipped": hook.skipped,
                       "forcer": {"overrides": hook.forcer.overrides, "agreements": hook.forcer.agreements,
                                  "log": [list(map(str, x)) for x in hook.forcer.log]},
                       "messages": hook.messages,
                       "calls": [{"name": c.name, "args": c.arguments, "status": c.status.value, "result": c.result,
                                  "model_step": c.model_step} for c in led.calls.values()],
                       "backend_audit": backend.audit}
            else:
                rec = run_cascade(asr, brain, tok, ad, drv, gate, ftr, frames, blog, reg, backend, wav)
                rec.update({"arm": arm, "lang": lang})
            frames.close()
            blog.close()
            rec["adapter"] = adapter_summary(ad, gate)
            rec["wall_s"] = round(time.monotonic() - t_s, 1)
            ftr.close()
            (od / "summary.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False, default=str))
            if tag == "smoke":
                ok = (rec.get("hook_calls", 0) > 0 and rec.get("n_gaps", 1) == 0
                      and all(c["same"] for c in rec.get("cache_checks", [])))
                status["smoke"] = {"ok": ok, "hook_calls": rec.get("hook_calls"), "n_gaps": rec.get("n_gaps"),
                                   "mismatch": rec.get("mismatch"),
                                   "cache_checks": rec.get("cache_checks"), "wall_s": rec["wall_s"]}
                print("SMOKE", json.dumps(status["smoke"], default=str), flush=True)
                if not ok:
                    (out / "status.json").write_text(json.dumps(status, indent=1, default=str))
                    raise SystemExit("smoke failed: TECHNICAL invariants")
                continue
            durations.append(rec["wall_s"])
            status["done"].append([lang, arm, rec["wall_s"]])
            print("DONE", sid, rec["wall_s"], "s |", json.dumps(rec.get("messages", [])[-6:], ensure_ascii=False)[:1500],
                  flush=True)
    status["elapsed_s"] = round(time.monotonic() - T0, 1)
    (out / "status.json").write_text(json.dumps(status, indent=1))
    print("STATUS", json.dumps(status), flush=True)


def run_cascade(asr, brain, tok, ad, drv, gate, ftr, frames, blog, reg, backend, wav):  # pragma: no cover - GPU
    """Text cascade on the same frame clock: brain at the opener (frame 0) and at every ASR EOU; reply ids reach the
    adapter one per frame from their measured availability; ASR speech during a reply = adapter barge-in."""
    from mynah_v2.agent.tools import ToolLedger
    from mynah_v2.tracing.events import Tracer as V2Tracer
    from e1.m1force import UserSide, OPENER_USER
    import time as _t
    user = UserSide(asr)
    n_frames = int(wav.numel() / 16000 / 0.08) + 25
    tr = V2Tracer(None, session_id="cas")
    led = ToolLedger("cas", tr)
    bound = reg.bind(led, lambda: 0, lambda: None, timeout_s=None)
    messages, queue, speaking, turn_ids, rec_turns = [], [], False, [], []
    pending_request = 0                       # frame at which a request is due (opener at 0)
    reply_end = None
    for k in range(n_frames):
        u = user.advance(k)
        if u["eou"]:
            ftr.emit("turn.eou", k, source="asr")
            pending_request = k
        text_id = T.PAD_ID
        if pending_request is not None and not speaking:
            t = pending_request
            pending_request = None
            utext = user.take_turn()
            if not utext and k > 0:                 # never an empty user turn after the opener
                ftr.emit("brain.skip", k, why="empty_user_turn")
                blog.write({"ev": "brain.skip", "k": k, "why": "empty_user_turn"})
                ad.feed(Frame(k, T.PAD_ID, user_speech=u["speech"]))
                frames.write({"k": k, "emitted": T.PAD_ID, "speech": u["speech"], "eou": u["eou"],
                              "new_words": u["new_words"] or None})
                continue
            messages.append({"role": "user", "content": utext or OPENER_USER})
            t0 = _t.monotonic()
            ftr.emit("brain.request", k, user_text=utext or OPENER_USER)
            r = brain.run(messages, decide_tool=bool(utext))
            row = {"ev": "brain.run", "k": k, "user": utext, "tool_margin": r["tool_margin"], "prefill_ms": r["prefill_ms"],
                   "decision": r.get("decision")}
            if r.get("tool"):
                call = r["tool"]
                parsed = call.get("parsed")
                t1 = _t.monotonic()
                res = (bound[parsed["name"]](parsed.get("arguments") or {}) if parsed and parsed.get("name") in bound
                       else json.dumps({"error": f"unknown or unparsable call: {parsed}"}))
                row.update(call=call, tool_result=res, tool_exec_ms=round((_t.monotonic() - t1) * 1e3, 1))
                messages.append({"role": "assistant", "content": call["text"]})
                messages.append({"role": "user", "content": f"<TOOL_RESPONSE>[{res}]</TOOL_RESPONSE>"})
                r = brain.run(messages, decide_tool=False)
                row["resume"] = {"text": r["text"]}
            ms = (_t.monotonic() - t0) * 1e3
            ids = [i for i in r["ids"] if i >= T.FIRST_TEXT_ID]
            per = r.get("decode_ms_per_tok", 0.0)
            first = max(ms - per * max(0, len(ids) - 1), r.get("prefill_ms", 0.0) + per)
            queue = [(k + math.ceil((first + per * i) / 80.0), tid) for i, tid in enumerate(ids)]
            row.update(reply=r["text"], wall_ms=round(ms, 1), first_avail_k=queue[0][0] if queue else None)
            blog.write(row)
            speaking, turn_ids = True, []
            text_id = T.BOS_ID
            rec_turns.append(row)
        elif speaking:
            if queue and queue[0][0] <= k:
                text_id = queue.pop(0)[1]
                turn_ids.append(text_id)
            elif not queue:
                if reply_end is None:
                    reply_end = k
                if k - reply_end >= 3:
                    text_id, speaking, reply_end = T.EOS_ID, False, None
                    messages.append({"role": "assistant", "content": tok.decode(turn_ids)})
            else:
                text_id = None
        n_cmd = len(ad.commands)
        ad.feed(Frame(k, text_id, user_speech=u["speech"]))
        if speaking and any(type(c).__name__ == "Cancel" for c in ad.commands[n_cmd:]):
            queue, speaking, reply_end = [], False, None
            messages.append({"role": "assistant", "content": tok.decode(turn_ids)})
            blog.write({"ev": "history.agent", "k": k, "spoken": tok.decode(turn_ids), "why": "barge_in"})
        frames.write({"k": k, "emitted": text_id, "speech": u["speech"], "eou": u["eou"],
                      "new_words": u["new_words"] or None})
        if k % 25 == 0:
            pump(ad, drv, gate)
    pump(ad, drv, gate)
    return {"messages": messages, "turns": rec_turns,
            "calls": [{"name": c.name, "args": c.arguments, "status": c.status.value, "result": c.result}
                      for c in led.calls.values()], "backend_audit": backend.audit}


if __name__ == "__main__":
    main()
