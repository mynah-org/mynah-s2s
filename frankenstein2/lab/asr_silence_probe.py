"""Does the streaming ASR drop the first words of an utterance that follows a long silence, and does the kind of
silence matter? For each language: caller line A, then S seconds of silence (digital zeros, or a faint noise floor
like a real microphone's), then caller line B, streamed through the ears hop by hop ("listen"). Reports how much of
line B was heard (character match against its own transcript alone).

    python frankenstein2/lab/asr_silence_probe.py --samples DIR --langs de,pl,fr,en,it --out DIR [server flags]

DIR = a packed samples folder (frankenstein2/lab/pack_samples.py: manifest.json + lines/<lang>/store_<i>.ogg).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

STEP = 1280


def main():  # pragma: no cover - model run
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    from frankenstein2.ears.client import EarsLink, f32b64
    from frankenstein2.lab.lang_suite import char_ratio
    from frankenstein2.server import build as B
    ap = argparse.ArgumentParser()
    B.add_args(ap)
    ap.add_argument("--samples", required=True)
    ap.add_argument("--langs", default="de,pl,fr,en,it")
    ap.add_argument("--silences", default="0.5,1,2,3,4,6")
    ap.add_argument("--floors", default="zeros,-70,-55", help="silence kind: zeros, or noise RMS in dBFS")
    ap.add_argument("--prerolls", default="0", help="seconds of the stream before line A (line 1 of the script "
                    "repeated with silences, like a long conversation): does the stream's age matter?")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    a.pocket_preload = ""
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    ears = B.start_ears(a, out)
    man = json.loads((Path(a.samples) / "manifest.json").read_text())
    by = {}
    for text, f in man["lines"].items():
        parts = f.split("/")
        if parts[0] == "lines" and Path(f).stem.startswith("store_"):
            by.setdefault(parts[1], {})[int(Path(f).stem.split("_")[1])] = (text, Path(a.samples) / f)

    def load(p):
        x, sr = sf.read(p, dtype="float32")
        x = x if x.ndim == 1 else x[:, 0]
        return resample_poly(x, 2, 3).astype(np.float32) if sr == 24000 else x

    def stream(lang, u):
        link = EarsLink(ears, f"probe-{lang}-{time.monotonic_ns()}", lang, thr=B.thresholds(a).get(lang), sync=True,
                        sync_timeout_s=30)
        link.open()
        pieces = []
        u = np.concatenate([u, np.zeros((-len(u)) % STEP + STEP * 8, np.float32)])
        for k in range(len(u) // STEP):
            link.push(u[k * STEP:(k + 1) * STEP], np.zeros(STEP, np.float32), "listen")
            for r in link.poll():
                pieces += [(p["f"], p["p"]) for p in r.get("pieces", [])]
        t_end = time.monotonic() + 10
        while link.in_flight is not None and time.monotonic() < t_end:
            for r in link.poll():
                pieces += [(p["f"], p["p"]) for p in r.get("pieces", [])]
            time.sleep(0.02)
        link.close()
        return pieces

    def text(pieces, f0=0):
        from frankenstein2.ears.stream import is_tag
        return "".join(p for f, p in pieces if f >= f0 and not is_tag(p)).replace("▁", " ").strip()

    rng = np.random.default_rng(0)
    rows = []
    for lang in [x for x in a.langs.split(",") if x]:
        if lang not in by or 0 not in by[lang] or 2 not in by[lang]:
            print("PROBE_SKIP", lang, flush=True)
            continue
        (ta, pa), (tb, pb) = by[lang][0], by[lang][2]
        xa, xb = load(pa), load(pb)
        alone = ears.request({"op": "transcribe", "id": f"pb-{lang}", "lang": lang, "pcm": f32b64(xb)},
                             "transcript", 120)["text"]
        xm = load(by[lang][1][1]) if 1 in by[lang] else xa
        for pre, floor, s in [(p, f, s) for p in [float(x) for x in a.prerolls.split(",") if x]
                              for f in [x for x in a.floors.split(",") if x]
                              for s in [float(x) for x in a.silences.split(",") if x]]:
            if True:
                n = int(s * 16000)
                gap = np.zeros(n, np.float32) if floor == "zeros" else \
                    (rng.standard_normal(n) * 10 ** (float(floor) / 20)).astype(np.float32)
                lead = np.zeros(8000, np.float32) if floor == "zeros" else \
                    (rng.standard_normal(8000) * 10 ** (float(floor) / 20)).astype(np.float32)
                roll = []
                while sum(len(x) for x in roll) < pre * 16000:     # line 2, 6 s of silence, again
                    roll += [xm, np.zeros(6 * 16000, np.float32)]
                lead = np.concatenate(roll + [lead]) if roll else lead
                u = np.concatenate([lead, xa, gap, xb])
                pcs = stream(lang, u)
                fb = (len(lead) + len(xa) + n) // STEP   # encoder frames are 80 ms, like the steps
                hb = text(pcs, fb - 2)
                r = {"lang": lang, "preroll_s": pre, "floor": floor, "silence_s": s, "heard_b": hb, "ratio_b": char_ratio(hb, tb),
                     "alone_ratio": char_ratio(alone, tb)}
                rows.append(r)
                print("PROBE", json.dumps(r, ensure_ascii=False), flush=True)
    (out / "probe.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False))
    import collections
    agg = collections.defaultdict(list)
    for r in rows:
        agg[(r["preroll_s"], r["floor"], r["silence_s"])].append(r["ratio_b"])
    for k, v in sorted(agg.items()):
        print("PROBE_MEAN", *k, round(sum(v) / len(v), 3), len(v), flush=True)
    print("PROBE_DONE", flush=True)


if __name__ == "__main__":
    main()
