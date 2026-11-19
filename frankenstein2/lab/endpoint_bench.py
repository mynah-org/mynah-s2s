"""How well does the turn-taking GRU end turns, per language? The ears alone (no LLM, no talker) on streams built from
the recorded caller lines, whose speech start and end are known from their energy:

  single     silence, one line, 3 s of silence: the endpoint's latency after the speech ends, or a miss
  mid_S      one line cut between two words in its middle (the quietest gap between 30 % and 70 % of the speech)
             and that gap stretched to S seconds: the sentence is not finished, so an endpoint inside the pause is
             premature; S = 0.3, 0.6, 0.9
  pause_S    line A, S seconds, line B: A is a complete sentence, so ending the turn there is reasonable (reference:
             how long a pause between sentences the GRU waits through)

Per endpoint it also records whether the user channel was voiced in the 300 ms before it (an energy veto would have
held it), and what the continuation check (control/continuation.py) would do with the text heard so far: hold it,
and whether the next word arrives within the hold (a premature endpoint rescued) or not (a finished turn delayed).
--ep-thrs sweeps the GRU's endpoint threshold. Prints EP lines and a per-language summary (EP_LANG); JSON in OUT.

    python frankenstein2/lab/endpoint_bench.py --samples DIR [--langs it,en,...] [--ep-thrs 0.915,0.95] --out DIR
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

STEP = 1280                      # 80 ms at 16 kHz = one ears frame


def speech_span(x, sr=16000, db=-40.0):
    """(first, last) sample index above ``db`` dBFS in 20 ms windows, or None."""
    import numpy as np
    w = sr // 50
    n = len(x) // w
    if n == 0:
        return None
    e = 20 * np.log10(np.sqrt((x[: n * w].reshape(n, w) ** 2).mean(1)) + 1e-9)
    on = np.where(e > db)[0]
    if not len(on):
        return None
    return int(on[0] * w), int((on[-1] + 1) * w)


def mid_gap(x, sr=16000):
    """The quietest 20 ms window between 30 % and 70 % of the speech: (sample index) where to cut the sentence."""
    import numpy as np
    span = speech_span(x, sr)
    if span is None:
        return None
    a, b = span
    w = sr // 50
    lo, hi = a + int(0.3 * (b - a)), a + int(0.7 * (b - a))
    n = (hi - lo) // w
    if n < 3:
        return None
    e = (x[lo: lo + n * w].reshape(n, w) ** 2).mean(1)
    i = int(np.argmin(e))
    return lo + i * w + w // 2


def hold_effect(eps: list, pieces: list, lang: str, hold: int) -> list:
    """Per endpoint frame: (held, resumed). held = the text heard up to it wants a continuation; resumed = a new
    word's piece arrives within ``hold`` frames after it (the turn goes on)."""
    from frankenstein2.control.continuation import wants_more
    out = []
    for e in eps:
        text = "".join(p for f, p in pieces if f <= e and not p.strip().startswith("<")).replace("\u2581", " ")
        held = wants_more(text, lang)
        resumed = held and any(e < f <= e + hold and any(ch.isalnum() for ch in p) for f, p in pieces)
        out.append((held, resumed))
    return out


def summarize(rows: list) -> dict:
    """Per language (and threshold): endpoint latency (median, max, ms) on single lines, misses, premature endpoints
    per pause, and with the continuation hold: premature ones left, finished turns delayed."""
    import statistics
    out = {}
    keys = sorted({(r["lang"], r.get("thr")) for r in rows}, key=lambda x: (x[0], x[1] or 0))
    for lang, thr in keys:
        rs = [r for r in rows if r["lang"] == lang and r.get("thr") == thr]
        single = [r for r in rs if r["kind"] == "single"]
        lat = [r["latency_ms"] for r in single if r["latency_ms"] is not None]
        d = {"lines": len(single), "missed": sum(1 for r in single if r["latency_ms"] is None),
             "latency_ms_p50": round(statistics.median(lat)) if lat else None, "latency_ms_max": max(lat) if lat else None,
             "held_finished": sum(1 for r in single if r.get("held_final"))}
        for kind in sorted({r["kind"] for r in rs if r["kind"] != "single"}):
            ps = [r for r in rs if r["kind"] == kind]
            d[kind] = {"n": len(ps), "premature": sum(1 for r in ps if r["premature"]),
                       "veto_would_hold": sum(1 for r in ps if r["premature"] and r["voiced_before"]),
                       "premature_after_hold": sum(1 for r in ps if r["premature"] and not r.get("rescued"))}
        out[lang if thr is None else f"{lang}@{thr}"] = d
    return out


def main():  # pragma: no cover - model run
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    from frankenstein2.ears.client import EarsLink
    from frankenstein2.server import build as B
    ap = argparse.ArgumentParser()
    B.add_args(ap)
    ap.add_argument("--samples", required=True)
    ap.add_argument("--langs", default="")
    ap.add_argument("--pauses", default="0.3,0.6,0.9")
    ap.add_argument("--ep-thrs", default="", help="endpoint thresholds to sweep (default: the server's own)")
    ap.add_argument("--hold-frames", type=int, default=10, help="the continuation hold (core.py default 10)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    a.pocket_preload = ""
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    ears = B.start_ears(a, out)
    thr = B.thresholds(a)
    man = json.loads((Path(a.samples) / "manifest.json").read_text())
    by: dict = {}
    for text, f in man["lines"].items():
        parts = f.split("/")
        if parts[0] == "lines" and Path(f).stem.startswith("store_"):
            by.setdefault(parts[1], []).append((int(Path(f).stem.split("_")[1]), Path(a.samples) / f))
    langs = [x for x in a.langs.split(",") if x] or sorted(by)

    def load(p):
        x, sr = sf.read(p, dtype="float32")
        x = x if x.ndim == 1 else x[:, 0]
        return resample_poly(x, 2, 3).astype(np.float32) if sr == 24000 else x

    def stream(lang, u):
        """-> (endpoint frames, frame dB by frame index) for audio ``u`` streamed in listen mode; the pieces heard
        go to ``last_pieces``."""
        link = EarsLink(ears, f"ep-{lang}-{time.monotonic_ns()}", lang, thr=thr_now.get(lang), sync=True,
                        sync_timeout_s=30)
        link.open()
        eps, dbs = [], {}
        last_pieces.clear()
        u = np.concatenate([u, np.zeros((-len(u)) % STEP, np.float32)])

        def take(res):
            for r in res:
                for f in r.get("frames", []):
                    dbs[f.get("k")] = f.get("db")
                last_pieces.extend((p["f"], p["p"]) for p in r.get("pieces", []))
                eps.extend(e["k"] for e in r.get("events", []) if e["kind"] == "ENDPOINT")
        for k in range(len(u) // STEP):
            link.push(u[k * STEP:(k + 1) * STEP], np.zeros(STEP, np.float32), "listen")
            take(link.poll())
        t_end = time.monotonic() + 10
        while link.in_flight is not None and time.monotonic() < t_end:
            take(link.poll())
            time.sleep(0.02)
        link.close()
        return eps, dbs

    def voiced_before(dbs, k, frames=4, db=-42.0):
        return any((dbs.get(j) or -120) > db for j in range(k - frames, k))

    rows = []
    last_pieces: list = []
    sweep = [float(x) for x in a.ep_thrs.split(",") if x] or [None]
    for tv, lang in [(t, l) for t in sweep for l in langs]:
        thr_now = thr if tv is None else {**thr, lang: {**thr.get(lang, {}), "ep_thr": tv}}
        lines = [load(p) for _, p in sorted(by.get(lang, []))]
        if not lines:
            print("EP_SKIP", lang, flush=True)
            continue
        lead = np.zeros(16000, np.float32)
        for i, x in enumerate(lines):                              # single lines: latency
            span = speech_span(x)
            if span is None:
                continue
            u = np.concatenate([lead, x, np.zeros(3 * 16000, np.float32)])
            eps, dbs = stream(lang, u)
            end_f = (len(lead) + span[1]) / STEP
            after = [e for e in eps if e >= end_f - 1]
            he = hold_effect(after[:1], last_pieces, lang, a.hold_frames)
            r = {"lang": lang, "thr": tv, "kind": "single", "line": i,
                 "latency_ms": round((after[0] - end_f) * 80) if after else None,
                 "early": [e for e in eps if e < end_f - 1], "held_final": bool(he and he[0][0])}
            rows.append(r)
            print("EP", json.dumps(r), flush=True)
        for s in [float(x) for x in a.pauses.split(",") if x]:     # a pause in the middle of a sentence
            for i, x in enumerate(lines):
                cut = mid_gap(x)
                if cut is None:
                    continue
                u = np.concatenate([lead, x[:cut], np.zeros(int(s * 16000), np.float32), x[cut:],
                                    np.zeros(3 * 16000, np.float32)])
                eps, dbs = stream(lang, u)
                p0, p1 = (len(lead) + cut) / STEP, (len(lead) + cut + int(s * 16000)) / STEP
                inside = [e for e in eps if p0 - 1 <= e <= p1 + 1]
                he = hold_effect(inside[:1], last_pieces, lang, a.hold_frames)
                r = {"lang": lang, "thr": tv, "kind": f"mid_{s}", "line": i, "premature": bool(inside),
                     "rescued": bool(he and he[0][1]), "text_at_ep": "".join(
                         p for f, p in last_pieces if inside and f <= inside[0]).replace("\u2581", " ")[-40:],
                     "voiced_before": bool(inside) and voiced_before(dbs, inside[0]), "endpoints": eps}
                rows.append(r)
                print("EP", json.dumps(r), flush=True)
        for s in [float(x) for x in a.pauses.split(",") if x]:     # a pause between two sentences
            for i in range(len(lines)):
                xa, xb = lines[i], lines[(i + 1) % len(lines)]
                sa, sb = speech_span(xa), speech_span(xb)
                if sa is None or sb is None:
                    continue
                xa, xb = xa[: sa[1]], xb[sb[0]:]                   # exactly S seconds of silence between the words
                u = np.concatenate([lead, xa, np.zeros(int(s * 16000), np.float32), xb,
                                    np.zeros(3 * 16000, np.float32)])
                eps, dbs = stream(lang, u)
                p0, p1 = (len(lead) + len(xa)) / STEP, (len(lead) + len(xa) + int(s * 16000)) / STEP
                inside = [e for e in eps if p0 - 1 <= e <= p1 + 1]
                he = hold_effect(inside[:1], last_pieces, lang, a.hold_frames)
                r = {"lang": lang, "thr": tv, "kind": f"pause_{s}", "line": i, "premature": bool(inside),
                     "rescued": bool(he and he[0][1]), "text_at_ep": "".join(
                         p for f, p in last_pieces if inside and f <= inside[0]).replace("\u2581", " ")[-40:],
                     "voiced_before": bool(inside) and voiced_before(dbs, inside[0]), "endpoints": eps}
                rows.append(r)
                print("EP", json.dumps(r), flush=True)
    summ = summarize(rows)
    for lang, d in summ.items():
        print("EP_LANG", lang, json.dumps(d), flush=True)
    (out / "endpoint_bench.json").write_text(json.dumps({"rows": rows, "summary": summ}, indent=1))
    print("EP_DONE", flush=True)


if __name__ == "__main__":
    main()
