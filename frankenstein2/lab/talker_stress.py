"""Stress test of the talker language cache under real end-to-end sessions (one process, models loaded once).

Waves of concurrent reactive WebSocket sessions (it / en / fr tracks, tools on) run back to back for ``--minutes``.
With ``--talker-rotate N`` every new session of a language is served by the next of N package copies
(``italian_24l@00`` ...), so with 3 languages and N=10 the talker sees 30 distinct languages: most sessions start
on a cold language while the others are talking. Per wave one ``WAVE`` line:

  replies / missing, first audio p50 / p95, reply p50, the talker's language waits (p50 / max), loads per tier
  (disk / host) with their times, evictions, resident languages, scheduler component times (max: a stall shows
  here), audio underruns, CUDA allocated / reserved and the process RSS (a leak shows as a trend across waves).

    python frankenstein2/lab/talker_stress.py --runtime single --pocket-packages PKGS --talker-rotate 10 \\
        --asr-dir RUN_WITH_TRACKS --out OUT --minutes 45 --wave-n 8   (+ the server's model options)
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "frankenstein"))


def pct(xs, p):
    xs = sorted(x for x in xs if x is not None)
    return round(xs[min(len(xs) - 1, int(p * len(xs)))], 3) if xs else None


def rss_gb():
    with open("/proc/self/statm") as f:
        return round(int(f.read().split()[1]) * resource.getpagesize() / 2 ** 30, 2)


def main():  # pragma: no cover - model run
    import asyncio
    import numpy as np
    import soundfile as sf
    import torch
    from frankenstein2.server import build as B
    ap = argparse.ArgumentParser()
    B.add_args(ap)
    ap.add_argument("--asr-dir", required=True, help="dir with track_{it,en,fr}.wav + asr_{lang}.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--minutes", type=float, default=30.0)
    ap.add_argument("--wave-n", type=int, default=8)
    ap.add_argument("--cut-s", type=float, default=52.0)
    ap.add_argument("--session-voice", default="")
    a = ap.parse_args()
    B.resolve_llm(a)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    t_load = time.monotonic()
    from frankenstein2.obs import TraceStore
    from frankenstein2.server.engine import Resources, make_pipeline
    from frankenstein2.server.multi import MultiServer
    from frankenstein2.lab.ws_sessions import ClientSpec, run_clients
    ears = B.start_ears(a, out)
    spec = B.tenant(a)
    pocket = B.start_pocket(a, out)
    qproc = B.start_qwen(a, out)
    make_llm = B.qwen_factory(a, qproc, spec)
    store = TraceStore(out / "f2_traces.sqlite")
    res = Resources(ears=ears, make_llm=make_llm, make_talker=B.talker_factory(a, pocket), voices=B.voices(a),
                    ears_thr=B.thresholds(a), trace_dir=str(out), sinks=[store.sink])
    sched = a._single["sched"]
    print("LOADED", json.dumps({"load_s": round(time.monotonic() - t_load, 1), **sched.mem(), "rss_gb": rss_gb(),
                                "talker": pocket.langs.stats()}, default=str), flush=True)
    port = 8793
    srv = MultiServer(lambda sid, lang, voice, prompt, sink: make_pipeline(res, sid, lang, voice, prompt, sink),
                      max_sessions=a.wave_n + 2)
    ready = threading.Event()
    threading.Thread(target=lambda: asyncio.run(srv.serve("127.0.0.1", port, ready=ready.set)), daemon=True).start()
    ready.wait(30)
    tracks = {}
    for lang in ("it", "en", "fr"):
        fz = json.load(open(Path(a.asr_dir) / f"asr_{lang}.json"))
        x, _ = sf.read(Path(a.asr_dir) / f"track_{lang}.wav", dtype="float32")
        x = x[: int(a.cut_s * 16000)]
        tracks[lang] = (x, [u for u in fz["timeline"] if u["end_k"] * 0.08 < a.cut_s])
    voice = B.voices(a).get(a.session_voice, a.session_voice)
    T0, wave, rows = time.monotonic(), 0, []
    seen = {k: 0 for k in ("ears_ms", "talker_ms", "llm_ms")}
    n_loads = n_wait = 0
    langs = ["it", "en", "fr"]
    while time.monotonic() - T0 < a.minutes * 60:
        specs = [ClientSpec(f"w{wave:03d}_{i:02d}_{langs[(wave + i) % 3]}", f"ws://127.0.0.1:{port}/v1/session",
                            langs[(wave + i) % 3], *tracks[langs[(wave + i) % 3]], voice=voice,
                            start_delay_s=0.7 * i, reactive=True) for i in range(a.wave_n)]
        tw = time.monotonic()
        r = run_clients(specs, out / f"wave{wave:03d}")
        rep, fa, miss, under = [], [], 0, 0
        for name, x in r.items():
            if "error" in x:
                miss += 4
                continue
            u = x.get("utterances", [])
            rep += [y.get("reply_text_s") for y in u if y.get("reply_text_s") is not None]
            fa += [y.get("first_audio_s") for y in u if y.get("first_audio_s") is not None]
            miss += sum(y.get("reply_text_s") is None for y in u)
        for f in (out / f"wave{wave:03d}").glob("*/events.jsonl"):
            under += sum(1 for line in open(f) if '"audio.underrun"' in line)
        st = pocket.langs.stats()
        loads = pocket.langs.loads[n_loads:]
        n_loads = len(pocket.langs.loads)
        waits = pocket.waits[n_wait:]
        n_wait = len(pocket.waits)
        comp = {}
        for k in seen:
            xs = sched.stats[k][seen[k]:]
            seen[k] = len(sched.stats[k])
            comp[k] = round(max(xs), 1) if xs else None
        row = {"wave": wave, "t_min": round((time.monotonic() - T0) / 60, 1), "wave_s": round(time.monotonic() - tw, 1),
               "replies": len(rep), "missing": miss, "first_audio_p50": pct(fa, .5), "first_audio_p95": pct(fa, .95),
               "reply_p50": pct(rep, .5), "lang_wait_ms_p50": pct(waits, .5), "lang_wait_ms_max": pct(waits, 1.0),
               "loads_disk": sum(1 for x in loads if x["tier"] == "disk"),
               "loads_host": sum(1 for x in loads if x["tier"] == "host"),
               "load_ms_max": max([x["fetch_ms"] + x["build_ms"] for x in loads], default=None),
               "evictions": st["evictions"], "resident": len(st["resident"]), "failed": len(st["failed"]),
               "sched_max_ms": comp, "underruns": under, **sched.mem(), "rss_gb": rss_gb()}
        rows.append(row)
        print("WAVE", json.dumps(row), flush=True)
        (out / "waves.json").write_text(json.dumps(rows, indent=1))
        wave += 1
    alloc = [x["cuda_allocated_gb"] for x in rows]
    rss = [x["rss_gb"] for x in rows]
    summary = {"waves": len(rows), "minutes": round((time.monotonic() - T0) / 60, 1),
               "replies": sum(x["replies"] for x in rows), "missing": sum(x["missing"] for x in rows),
               "talker": pocket.langs.stats(), "voice_ms": pocket.voice_ms[:20],
               "cuda_allocated_first_last_max": [alloc[0], alloc[-1], max(alloc)] if alloc else None,
               "rss_first_last_max": [rss[0], rss[-1], max(rss)] if rss else None,
               "scheduler": sched.summary()}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print("SUMMARY", json.dumps(summary, default=str)[:4000], flush=True)
    store.close()
    _ = np, torch


if __name__ == "__main__":
    main()
