"""Frankenstein v2 end-to-end session runs (GPU host), the v1 G-run protocol on the v2 stack.

  arm A  the fixed 52 s Italian user track (v1's frozen session: U1 store question, U2 order question with name +
         postcode, U3 out of scope, U4 a barge-in over the agent), paced in real time
  arm B  reactive user: the same four utterances, each after the agent's audio ended + 0.5 s; the last one a
         deliberate barge-in 1 s into the agent's real reply (v1 e1/reactive_user.py, imported read-only)
Per arm: the session trace, measurements (lab/metrics.py), content check (each played agent turn transcribed by the
ears' ASR with the session language vs the text it spoke), per-process memory after load / after the arm (peak),
listening files: agent_24k.wav, mix_stream_24k.wav (L user, R agent on the session clock), mix_wall_24k.wav (both
on the wall clock).

``--llm scripted`` + ``--pace fast`` + ``--ears-sync`` = a CPU plumbing smoke (no Nano, no real time).
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "frankenstein"))

TOOLS_EXPECTED = [{"name": "search_catalog", "args": {"query": "divano"}},
                  {"name": "find_customer", "args": {"full_name": "Mario Rossi", "postcode": "10121"}}]


def mem_available_gb():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return round(int(line.split()[1]) / 2 ** 20, 2)
    return None


def cuda_snapshot(torch):
    if not torch.cuda.is_available():
        return {}
    free, total = torch.cuda.mem_get_info()
    return {"device_free_gb": round(free / 2 ** 30, 2), "device_total_gb": round(total / 2 ** 30, 2),
            "allocated_gb": round(torch.cuda.memory_allocated() / 2 ** 30, 3),
            "reserved_gb": round(torch.cuda.memory_reserved() / 2 ** 30, 3),
            "peak_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 3)}


def rss_gb(key="VmRSS"):
    for line in open("/proc/self/status"):
        if line.startswith(key + ":"):
            return round(int(line.split()[1]) / 2 ** 20, 3)
    return None


def proc_mem(ears, pocket):
    out = {}
    try:
        out["ears"] = ears.request({"op": "mem", "id": "m"}, "mem", 30)
    except Exception as e:  # noqa: BLE001
        out["ears"] = {"error": repr(e)}
    try:
        pocket.send({"op": "mem", "id": "m"})
        out["pocket"] = pocket.wait("mem", 60)
    except Exception as e:  # noqa: BLE001
        out["pocket"] = {"error": repr(e)}
    for d in out.values():
        for k in ("ev", "id", "t_recv_ns"):
            d.pop(k, None)
    return out


def load_track(asr_dir: Path, cut_s: float):
    import numpy as np
    import soundfile as sf
    frozen = json.load(open(asr_dir / "asr_it.json"))
    x, sr = sf.read(asr_dir / "track_it.wav", dtype="float32")
    if x.ndim > 1:
        x = x.mean(1)
    assert sr == 16000
    return frozen, x[: int(cut_s * 16000)].astype(np.float32)


def scripted_llm_factory():
    from frankenstein2.llm.base import ScriptedLLM

    def script(msgs):
        u = msgs[-1]["content"].lower()
        if "divano" in u:
            return [("decided",), ("tool", "search_catalog", {"query": "divano a tre posti"}),
                    ("text", "Abbiamo il divano Alba a tre posti, costa 899 €.")]
        if "rossi" in u:
            return [("decided",), ("tool", "find_customer", {"name": "Mario Rossi", "postcode": "uno zero uno due uno"}),
                    ("text", "Grazie signor Rossi, il suo ordine arriva il 14/11.")]
        return [("text", "Mi dispiace, posso aiutarla solo con il negozio.")]

    def make(sid, lang, system_prompt=None):
        return ScriptedLLM(script, tools={"search_catalog": lambda a: '{"items": 1}',
                                          "find_customer": lambda a: '{"found": true}'}, lang=lang)
    return make


def run_arm(name, eng, res, frozen, track, out: Path, a, reactive: bool):  # pragma: no cover - model run
    import numpy as np
    eng.set_session_options(language="it", voice=a.session_voice)
    eng.open(None)
    p = eng.p
    for k in ("ears", "llm", "talker"):
        pass
    od = out / name
    od.mkdir()
    from frankenstein2.tracing import Tracer
    tr = Tracer(od / "trace.jsonl", session_id=name)
    p.tr = tr
    p.ctl.tr = tr
    p.ears.sync, p.ears.sync_timeout_s = a.ears_sync, 120.0
    tl = frozen["timeline"]
    STEP = 1280
    users, outs, agents, wall = [], [], [], []
    orig = p.step

    def rec_step(x):
        r = orig(x)
        outs.append(r["audio"])
        agents.append(r["agent"])
        return r
    p.step = rec_step
    import torch
    t0 = time.monotonic()
    if not reactive:
        n_steps = len(track) // STEP + int(a.tail_s / 0.08)
        for k in range(n_steps):
            if a.pace == "realtime":
                w = t0 + k * 0.08 - time.monotonic()
                if w > 0:
                    time.sleep(w)
            wall.append(time.monotonic())
            x = track[k * STEP:(k + 1) * STEP]
            if len(x) < STEP:
                x = np.concatenate([x, np.zeros(STEP - len(x), np.float32)])
            users.append(x)
            eng.step(torch.from_numpy(x))
        timeline = [{"label": u["label"], "start_k": u["start_k"], "end_k": u["end_k"]} for u in tl]
        waits = None
    else:
        from e1.reactive_user import ReactiveUser, Utt
        utts = [Utt(u["label"], track[u["start_k"] * STEP:(u["end_k"] + 1) * STEP], barge=(i == len(tl) - 1))
                for i, u in enumerate(tl)]
        rng = np.random.default_rng(0)
        from frankenstein2.control.controller import St

        class World:
            def audible(self, now):
                return p.playout.audible()

            def agent_idle(self, now):
                return p.ctl.state is St.LISTENING and not p.playout.audible()

            def audio_starts_after(self, t, real_only):
                xs = []
                for s in p.playout.segs.values():
                    if s.lane != "agent" or s.first_step is None or (real_only and s.holding):
                        continue
                    if s.first_step < len(wall) and wall[s.first_step] > t:
                        xs.append(wall[s.first_step])
                return min(xs) if xs else None
        ru = ReactiveUser(utts, chunk_samples=STEP, noise=lambda n: (rng.standard_normal(n) * 3e-4).astype(np.float32),
                          cat=lambda xs: np.concatenate([np.asarray(x, np.float32) for x in xs]))
        w_ = World()
        k = 0
        while not ru.done and k * 0.08 < a.max_reactive_s:
            if a.pace == "realtime":
                w = t0 + k * 0.08 - time.monotonic()
                if w > 0:
                    time.sleep(w)
            wall.append(time.monotonic())
            x = ru.next_chunk(time.monotonic() if a.pace == "realtime" else t0 + k * 0.08, w_)
            users.append(x)
            eng.step(torch.from_numpy(x))
            k += 1
        for _ in range(int(a.tail_s / 0.08)):
            wall.append(time.monotonic())
            x = (rng.standard_normal(STEP) * 3e-4).astype(np.float32)
            users.append(x)
            eng.step(torch.from_numpy(x))
        timeline = [{"label": x["label"], "start_k": x["start_k"], "end_k": x["end_k"], "barge": x["barge"]}
                    for x in ru.timeline if "end_k" in x]
        waits = ru.waits
    t_end = time.monotonic()
    segs = [{"epoch": s.epoch, "turn": s.turn, "seg": s.seg, "lane": s.lane, "holding": s.holding,
             "first_step": s.first_step, "released": s.released, "queued": s.queued, "done": s.done, "text": s.text}
            for s in p.playout.segs.values()]
    llm = p.llm
    calls = getattr(llm, "calls", [])
    history = list(p.conv.messages)
    stats = dict(p.stats)
    talker_done = getattr(p.talker, "done_info", [])
    p.step = orig
    eng.close()
    tr.close()
    events = [json.loads(x) for x in open(od / "trace.jsonl")]
    from frankenstein2.lab.metrics import analyse, cer
    m = analyse(timeline, events, segs, wall, t0, client_delay_steps=1, tools_expected=TOOLS_EXPECTED)
    m["loop_s"] = round(t_end - t0, 1)
    m["steps"] = len(wall)
    st = [e for e in events if e["ev"] == "audio.release"]
    m["step_wall_ms"] = None
    # listening files + content check
    import soundfile as sf
    from scipy.signal import resample_poly
    agent = np.concatenate(agents) if agents else np.zeros(0, np.float32)
    mix_out = np.concatenate(outs) if outs else np.zeros(0, np.float32)
    user = np.concatenate(users)
    user24 = resample_poly(user, 3, 2).astype(np.float32)
    n = max(len(user24), len(mix_out) + 1920)
    stereo = np.zeros((n, 2), np.float32)
    stereo[:len(user24), 0] = user24
    stereo[1920:1920 + len(mix_out), 1] = mix_out                      # heard one step after release
    sf.write(od / "agent_24k.wav", agent, 24000, subtype="PCM_16")
    sf.write(od / "mix_stream_24k.wav", stereo, 24000, subtype="PCM_16")
    W = int((wall[-1] - t0 + 1.0) * 24000) + 1920
    wst = np.zeros((W, 2), np.float32)
    for k, wk in enumerate(wall):
        i = int((wk - t0) * 24000)
        u = user24[k * 1920:(k + 1) * 1920]
        wst[i:i + len(u), 0] += u[: max(0, W - i)]
        o = mix_out[k * 1920:(k + 1) * 1920]
        j = i + 1920
        wst[j:j + len(o), 1] += o[: max(0, W - j)]
    sf.write(od / "mix_wall_24k.wav", wst, 24000, subtype="PCM_16")
    content = []
    if a.content:
        by_turn: dict = {}
        for s in segs:
            if s["lane"] == "agent" and s["first_step"] is not None and s["released"] > 0:
                by_turn.setdefault((s["epoch"], s["turn"]), []).append(s)
        for (ep, turn), ss in sorted(by_turn.items()):
            ss.sort(key=lambda s: s["seg"])
            k0 = min(s["first_step"] for s in ss)
            k1 = max(s["first_step"] + (s["released"] + 1919) // 1920 for s in ss)
            seg_audio = agent[k0 * 1920:k1 * 1920]
            pcm16 = resample_poly(seg_audio, 2, 3).astype(np.float32)
            ref = " ".join(s["text"] for s in ss)
            spoken = [e for e in events if e["ev"] == "tts.segment" and e.get("turn_id") == turn
                      and e.get("epoch") == ep and e.get("lane") == "agent"]
            ref_speech = " ".join(e.get("speech", "") for e in spoken)
            from frankenstein2.ears.client import f32b64
            r = res.ears.request({"op": "transcribe", "id": f"{name}-{ep}-{turn}", "lang": "it",
                                  "pcm": f32b64(pcm16)}, "transcript", 120)
            complete = all(s["done"] and s["released"] >= s["queued"] for s in ss)
            content.append({"epoch": ep, "turn": turn, "played_s": round(len(seg_audio) / 24000, 2),
                            "complete": complete, "text": ref, "speech": ref_speech, "asr": r["text"],
                            "cer_vs_speech": cer(ref_speech, r["text"]) if complete else None})
            sf.write(od / f"agent_turn_e{ep}_t{turn}.wav", seg_audio, 24000, subtype="PCM_16")
    summary = {"arm": name, "reactive": reactive, "metrics": m, "content": content, "calls": calls,
               "history": history, "waits": waits, "timeline": timeline, "stats": {
                   "endpoints": stats["endpoints"], "barges": stats["barges"], "backchannels": stats["backchannels"],
                   "cancels": stats["cancels"]}, "talker_done": talker_done[:60]}
    (od / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False, default=str))
    return summary


def ws_phases(a, res, out: Path) -> dict:  # pragma: no cover - model run
    """The multi-session WebSocket server in this process + the web UI bridge as a subprocess; concurrent clients in
    different languages (phase 1: IT through the bridge + EN direct; phase 2: FR + IT direct)."""
    import asyncio
    import subprocess
    import threading
    from frankenstein2.lab.ws_sessions import ClientSpec, run_clients
    from frankenstein2.server.engine import make_pipeline
    from frankenstein2.server.multi import MultiServer
    port, bport, uport = 8791, 8798, 8797
    srv = MultiServer(lambda sid, lang, voice, prompt, sink: make_pipeline(res, sid, lang, voice, prompt, sink),
                      max_sessions=4)
    ready = threading.Event()
    th = threading.Thread(target=lambda: asyncio.run(srv.serve("127.0.0.1", port, ready=ready.set)), daemon=True)
    th.start()
    ready.wait(30)
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    bridge = subprocess.Popen([sys.executable, str(ROOT / "webui" / "server.py"), "--engine", "frankenstein2",
                               "--v2-ws", f"ws://127.0.0.1:{port}/v1/session", "--port", str(uport), "--ws-port",
                               str(bport), "--profile", "int4"], env=env, stdout=open(out / "bridge.log", "w"),
                              stderr=subprocess.STDOUT)
    time.sleep(3)
    direct, via = f"ws://127.0.0.1:{port}/v1/session", f"ws://127.0.0.1:{bport}/ws"
    tracks = {}
    for lang in ("it", "en", "fr"):
        fz = json.load(open(Path(a.asr_dir) / f"asr_{lang}.json"))
        import soundfile as sf
        x, _ = sf.read(Path(a.asr_dir) / f"track_{lang}.wav", dtype="float32")
        x = x[: int(a.cut_s * 16000)]
        tracks[lang] = (x, [u for u in fz["timeline"] if u["end_k"] * 0.08 < a.cut_s])
    res_ws = {}
    v = a.session_voice
    p1 = [ClientSpec("p1_it_bridge", via, "it", *tracks["it"], via_bridge=True, voice=v),
          ClientSpec("p1_en_direct", direct, "en", *tracks["en"], voice=v, start_delay_s=1.5)]
    res_ws["phase1"] = run_clients(p1, out / "ws")
    p2 = [ClientSpec("p2_fr_direct", direct, "fr", *tracks["fr"], voice=v),
          ClientSpec("p2_it_direct", direct, "it", *tracks["it"], voice=v, start_delay_s=2.0)]
    res_ws["phase2"] = run_clients(p2, out / "ws")
    bridge.terminate()
    if a.content:                                     # content of everything each client received (ears ASR)
        import soundfile as sf
        from scipy.signal import resample_poly
        from frankenstein2.ears.client import f32b64
        from frankenstein2.lab.metrics import cer
        for ph in res_ws.values():
            for name, r in ph.items():
                if "error" in r:
                    continue
                y, sr = sf.read(out / "ws" / name / "agent_client_22k.wav", dtype="float32")
                y16 = resample_poly(y, 320, 441).astype(np.float32)
                t = res.ears.request({"op": "transcribe", "id": name, "lang": r["language"], "pcm": f32b64(y16)},
                                     "transcript", 120)
                r["asr_all"] = t["text"]
                r["cer_all_vs_said"] = cer(" ".join(x for x in r["said"] if x), t["text"])
    return res_ws


def main():  # pragma: no cover - model run
    from frankenstein2.server import build as B
    ap = argparse.ArgumentParser()
    B.add_args(ap)
    ap.add_argument("--asr-dir", required=True, help="v1 frozen session dir (track_it.wav, asr_it.json)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--listen-dir", default="")
    ap.add_argument("--cut-s", type=float, default=52.0)
    ap.add_argument("--tail-s", type=float, default=10.0)
    ap.add_argument("--max-reactive-s", type=float, default=150.0)
    ap.add_argument("--arms", default="A,B")
    ap.add_argument("--pace", default="realtime", choices=["realtime", "fast"])
    ap.add_argument("--ears-sync", action="store_true")
    ap.add_argument("--llm", default="nano", choices=["nano", "scripted"])
    ap.add_argument("--bf16-arm", action="store_true", help="after the INT4 arms, arm A again with INT4 decode off")
    ap.add_argument("--session-voice", default="")
    ap.add_argument("--warm-llm", action="store_true", help="one request through the LLM before the arms")
    ap.add_argument("--ws-phases", action="store_true", help="after the arms: concurrent WebSocket sessions")
    ap.add_argument("--content", action="store_true")
    ap.add_argument("--budget-s", type=float, default=1e9)
    a = ap.parse_args()
    T0 = time.monotonic()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    import torch
    if a.threads:
        torch.set_num_threads(a.threads)
    res_mem = {"t0": {"mem_available_gb": mem_available_gb(), **cuda_snapshot(torch)}}
    info = {"argv": sys.argv, "commit": os.environ.get("F2_COMMIT"), "mem": res_mem}

    def save():
        (out / "run.json").write_text(json.dumps(info, indent=1, ensure_ascii=False, default=str))
    ears = B.start_ears(a, out)
    res_mem["after_ears"] = {"mem_available_gb": mem_available_gb(), **cuda_snapshot(torch), "ears": ears.ready}
    save()
    spec = B.tenant(a)
    # the talker first (small), then the text LLM: f2-001 timed out loading the Pocket voice on the GPU after Nano
    pocket = B.start_pocket(a, out)
    from frankenstein2.tts.procs import PocketTalker
    try:
        warm = PocketTalker(pocket, "warm", "it", B.voices(a).get(a.session_voice, a.session_voice))
    except Exception as e:  # noqa: BLE001 - fall back to Pocket on the CPU (verified in the CPU smoke)
        info["talker_gpu_error"] = repr(e)
        save()
        pocket.close()
        (out / f"{a.talker}_server.log").rename(out / f"{a.talker}_server.gpu_failed.log")
        a.talker_device = "cpu"
        pocket = B.start_pocket(a, out)
        warm = PocketTalker(pocket, "warm", "it", B.voices(a).get(a.session_voice, a.session_voice))
    info["talker_voice"] = warm.info
    warm.close()
    info["talker_device"] = a.talker_device
    res_mem["after_pocket"] = {"mem_available_gb": mem_available_gb(), **cuda_snapshot(torch),
                               "procs": proc_mem(ears, pocket)}
    save()
    shared = None
    if a.llm == "nano":
        from frankenstein2.llm.nano import NanoShared
        t = time.monotonic()
        shared = NanoShared(a.nano_dir, str(Path(a.lab_scripts) / "multilingual"), a.nano_precision, a.nano_graph)
        info["nano_load_s"] = round(time.monotonic() - t, 1)
        res_mem["after_nano"] = {"mem_available_gb": mem_available_gb(), **cuda_snapshot(torch), "nano": shared.mem,
                                 "engine_rss_gb": rss_gb(), "engine_rss_peak_gb": rss_gb("VmHWM"), "procs": proc_mem(ears, pocket)}
        make_llm = B.llm_factory(a, shared, spec)
        if a.warm_llm:                    # f2-003: the first request after load took ~7 s (cold kernels)
            t = time.monotonic()
            w = make_llm("warm", "it")
            w.submit(1, [{"role": "user", "content": "Buongiorno, cerco un divano a tre posti."}])
            done = False
            while not done and time.monotonic() - t < 90:
                done = any(e["ev"] == "done" for e in w.poll())
                time.sleep(0.05)
            w.close()
            info["llm_warm_s"] = round(time.monotonic() - t, 1)
    else:
        make_llm = scripted_llm_factory()
    save()
    print("LOADS", json.dumps(res_mem, default=str)[:3000], flush=True)
    from frankenstein2.server.engine import Frankenstein2Engine, Resources
    res = Resources(ears=ears, make_llm=make_llm, make_talker=B.talker_factory(a, pocket), voices=B.voices(a),
                    ears_thr=B.thresholds(a), trace_dir=str(out))
    eng = Frankenstein2Engine(res)
    frozen, track = load_track(Path(a.asr_dir), a.cut_s)
    info["arms"] = {}
    arms = [x for x in a.arms.split(",") if x]
    if a.bf16_arm and a.llm == "nano" and a.nano_precision == "int4-decode":
        arms.append("A_bf16")
    for arm in arms:
        if time.monotonic() - T0 > a.budget_s:
            info["arms"][arm] = {"status": "INCOMPLETE", "why": "time budget"}
            save()
            continue
        if arm == "A_bf16":
            sys.path.insert(0, str(ROOT / "frankenstein"))
            from e1.int4_dispatch import Int4DecodeOnly
            Int4DecodeOnly.active = False
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        s = run_arm(arm, eng, res, frozen, track, out, a, reactive=arm.startswith("B"))
        info["arms"][arm] = {"status": "DONE", "metrics": s["metrics"],
                             "content": [{k: c[k] for k in ("turn", "epoch", "complete", "cer_vs_speech", "played_s")}
                                         for c in s["content"]],
                             "mem_after": {"mem_available_gb": mem_available_gb(), **cuda_snapshot(torch),
                                           "engine_rss_gb": rss_gb(), "engine_rss_peak_gb": rss_gb("VmHWM"), "procs": proc_mem(ears, pocket)}}
        save()
        print("ARM", arm, json.dumps(info["arms"][arm], default=str)[:4000], flush=True)
        if a.listen_dir:
            import shutil
            ld = Path(a.listen_dir) / arm
            ld.mkdir(parents=True, exist_ok=True)
            for f in ("agent_24k.wav", "mix_stream_24k.wav", "mix_wall_24k.wav"):
                shutil.copy(out / arm / f, ld / f)
    if a.ws_phases and time.monotonic() - T0 < a.budget_s + 60:
        info["ws"] = ws_phases(a, res, out)
        save()
        print("WS", json.dumps(info["ws"], default=str)[:6000], flush=True)
    pocket.close()
    ears.close()
    info["total_s"] = round(time.monotonic() - T0, 1)
    save()
    print("DONE", info["total_s"], flush=True)


if __name__ == "__main__":
    main()
