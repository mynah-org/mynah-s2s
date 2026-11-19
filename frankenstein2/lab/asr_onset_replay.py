"""Why does the streaming ASR lose the start of some turns? Replay a recorded session's input through the ears, offline
and deterministic, in variants that isolate the cause:

  as_run    the same modes (listen / speaking) and turn resets at the same steps as in the session's trace
  no_reset  the same modes, no resets
  listen    always "listen", no resets
  offline   the whole input transcribed at once (op transcribe), and each caller line alone

    python frankenstein2/lab/asr_onset_replay.py --runs DIR [--tags de,pl,fr] --out DIR [server flags]

DIR holds demo_dialogue_check.py outputs (<tag>_dialogue.ogg: left = the session's input as streamed, 16 kHz) and the
traces dir of the serving engine (--traces). Prints ONSET lines: per variant, the text heard over the session.
"""

from __future__ import annotations

import argparse
import difflib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

STEP = 1280                         # 80 ms at 16 kHz


def session_plan(trace: Path) -> dict:
    """From a session trace: the mode of each 80 ms step and the steps where the ears were reset."""
    modes, resets, lang, heard = {}, [], None, []
    for line in trace.open():
        r = json.loads(line)
        if r["ev"] == "asr.turn":
            heard.append(r.get("text", ""))
        elif r["ev"] == "engine.session":
            lang = r.get("language")
        elif r["ev"] == "ears.frame" and "mode" in r:
            modes[r.get("model_step", 0)] = r["mode"]
        elif r["ev"] == "ears.reset":
            resets.append((r.get("model_step", 0), r.get("reason")))
    return {"lang": lang, "modes": modes, "resets": resets, "heard": " ".join(heard)}


def main():  # pragma: no cover - model run
    import numpy as np
    import soundfile as sf
    from frankenstein2.ears.client import EarsLink, f32b64
    from frankenstein2.server import build as B
    ap = argparse.ArgumentParser()
    B.add_args(ap)
    ap.add_argument("--runs", required=True, help="demo_dialogue_check.py --out dir")
    ap.add_argument("--traces", required=True, help="the engine's --trace-dir of that run")
    ap.add_argument("--tags", default="de,pl,fr")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    a.pocket_preload = ""
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    ears = B.start_ears(a, out)
    traces = {}
    for t in Path(a.traces).glob("*.f2.jsonl"):
        p = session_plan(t)
        if p["lang"]:
            traces.setdefault(p["lang"], []).append((t.stat().st_mtime, t, p))
    report = {}
    for tag in [x for x in a.tags.split(",") if x]:
        lang = tag.split("_")[-1]
        x, sr = sf.read(Path(a.runs) / f"{tag}_dialogue.ogg", dtype="float32")
        u = x[:, 0] if x.ndim > 1 else x
        assert sr == 16000, sr
        cand = traces.get(lang, [])
        key = f"{tag.rsplit('_', 1)[0]}:{lang}" if "_" in tag else f"store:{lang}"
        demo = json.loads((Path(a.runs) / "demo_dialogue.json").read_text()).get(key, {})
        want = " ".join(x.get("heard", "") for x in demo.get("lines", []))
        if not cand or not want:
            print("ONSET_NO_TRACE", tag, flush=True)
            continue
        # the session of this dialogue: the trace whose ASR turns match what the checker recorded as heard
        _, tpath, plan = max(cand, key=lambda c: difflib.SequenceMatcher(None, c[2]["heard"], want).ratio())
        reset_at = dict(plan["resets"])
        rep = {"trace": tpath.name, "resets": plan["resets"]}
        for variant in ("as_run", "no_reset", "listen"):
            sid = f"replay-{tag}-{variant}-{int(time.time())}"
            link = EarsLink(ears, sid, lang, thr=B.thresholds(a).get(lang), sync=True, sync_timeout_s=30)
            link.open()
            pieces = []
            n = len(u) // STEP
            for k in range(n):
                mode = "listen" if variant == "listen" else plan["modes"].get(k, "listen")
                if variant == "as_run" and k in reset_at:
                    link.request_reset(keep_turn=reset_at[k] == "user_continues")
                link.push(u[k * STEP:(k + 1) * STEP], np.zeros(STEP, np.float32), mode)
                for res in link.poll():
                    pieces += [(p["f"], p["p"]) for p in res.get("pieces", [])]
            t_end = time.monotonic() + 10
            while link.in_flight is not None and time.monotonic() < t_end:
                for res in link.poll():
                    pieces += [(p["f"], p["p"]) for p in res.get("pieces", [])]
                time.sleep(0.02)
            link.close()
            text = "".join(p for _, p in pieces).replace("▁", " ")
            rep[variant] = text
            print("ONSET", tag, variant, json.dumps(text, ensure_ascii=False), flush=True)
        whole = ears.request({"op": "transcribe", "id": f"off-{tag}", "lang": lang, "pcm": f32b64(u)}, "transcript",
                             300)["text"]
        rep["offline"] = whole
        print("ONSET", tag, "offline", json.dumps(whole, ensure_ascii=False), flush=True)
        report[tag] = rep
    (out / "onset.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print("ONSET_DONE", flush=True)


if __name__ == "__main__":
    main()
