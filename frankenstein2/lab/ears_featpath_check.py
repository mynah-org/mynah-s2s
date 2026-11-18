"""How sensitive is the turn-taking GRU to the length of its feature window?

The GRU was trained on Tyger's regime: every 320 ms hop, the last 8 s of mic and agent audio are re-encoded with
lookahead 0 up to layer 15, and the last 4 frames are fed to the GRU. That re-encode is the largest cost of the ears.
With lookahead 0 the encoder is causal, so a longer window equals a cache-aware stream whose newest frames are
recomputed with the window's end-of-clip edge; a shorter window is plain truncation. This script runs the GRU + rules
on the same tracks with several window lengths and reports how far each one's outputs and events are from the 8 s
reference, with a silent agent channel and with an agent channel that carries speech.

usage: python ears_featpath_check.py --bundle DIR --asr-dir DIR --out out.json [--device cuda] [--cut-s 52]
                                     [--windows 8,4,2,16,30]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

HOP, FRAME, SR = 5120, 1280, 16000
NAMES = ["s0", "s1", "s2", "s3", "hazard", "yield", "bc", "p_end"]


def run_path(ears, x, agent, win_s, rules_cls, thr, energy_db, asr=None, tf32=False):  # pragma: no cover
    import numpy as np
    import torch
    asr, m, dev = asr or ears.asr, ears.gru, ears.dev

    def enc(xb):
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.backends.cudnn.allow_tf32 = tf32
        try:
            return asr.encode(asr.mel(xb, center=True), lookahead=0, stop_after=15)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
    L, H = m.core.num_layers, m.core.hidden_size
    W = int(win_s * SR)
    rules, h = rules_cls(thr=thr), None
    outs, events = [], []
    zero_cache = {}
    t_enc = 0.0
    for n in range(len(x) // HOP):
        end = (n + 1) * HOP
        u, a = x[max(0, end - W):end], agent[max(0, end - W):end]
        t0 = time.monotonic()
        rows = [u] + ([a] if a.any() else [])
        xb = torch.from_numpy(np.stack(rows)).to(dev)
        hb = enc(xb)[:, -4:]
        hu = hb[0]
        if a.any():
            ha = hb[1]
        else:
            if len(u) not in zero_cache:
                z = torch.zeros(1, len(u), device=dev)
                zero_cache[len(u)] = enc(z)[0, -4:]
            ha = zero_cache[len(u)]
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t_enc += time.monotonic() - t0
        for j in range(4):
            seg = x[end - HOP + j * FRAME: end - HOP + (j + 1) * FRAME]
            db = energy_db(seg)
            act = torch.tensor([[max(-3.0, min(3.0, (db + 50.0) / 25.0))]], device=dev)
            prev = torch.tensor([[rules.gru_prev_state()]], dtype=torch.long, device=dev)
            hh = h if h is not None else torch.zeros(L, 1, H, device=dev)
            o, h = m(hu[None, j:j + 1], ha[None, j:j + 1], act, prev, hh)
            row = o[0, 0].tolist()
            outs.append(row)
            evs, rst = rules.on_frame(row, db, left=3 - j)
            events += [{"frame": len(outs) - 1, **{k: v for k, v in e.items() if k != "kind"}, "kind": e["kind"]}
                       for e in evs if e["kind"] != "FRAME"]
            if rst:
                h = None
    return outs, events, t_enc


def compare(ref, alt):
    import numpy as np
    r, a = np.array(ref), np.array(alt)
    d = np.abs(r - a)
    return {NAMES[i]: {"max": round(float(d[:, i].max()), 4), "mean": round(float(d[:, i].mean()), 5)}
            for i in range(4, 8)}


def ev_key(e):
    return (e["kind"], e["frame"])


def main():  # pragma: no cover - model run
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--asr-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cut-s", type=float, default=52.0)
    ap.add_argument("--windows", default="8,4,2,16,30")
    ap.add_argument("--precisions", default="", help="e.g. fp32,tf32,bf16: compare encoder precisions at 8 s instead")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import numpy as np
    import soundfile as sf
    import torch
    from frankenstein2.ears.own import OwnEars
    from frankenstein2.ears.rules import RulesState, Thresholds, energy_db
    torch.set_grad_enabled(False)
    ears = OwnEars(a.bundle, device=a.device, slots=2)
    langs = ["it", "en", "fr"]
    tr = {l: sf.read(Path(a.asr_dir) / f"track_{l}.wav", dtype="float32")[0][: int(a.cut_s * SR)] for l in langs}
    wins = [float(w) for w in a.windows.split(",")]
    paths = [(f"win{w:g}", w, None, False) for w in wins]
    if a.precisions:
        from frankenstein2.ears.nemotron import NemotronASR
        bf = NemotronASR(a.bundle, a.device, dtype=torch.bfloat16) if "bf16" in a.precisions else None
        paths = [(p, 8.0, bf if p == "bf16" else None, p == "tf32") for p in a.precisions.split(",")]
    res = {}
    for li, l in enumerate(langs):
        x = tr[l]
        other = tr[langs[(li + 1) % 3]]
        agent_speech = np.zeros_like(x)
        n = min(len(x), len(other))
        agent_speech[:n] = 0.3 * other[:n]                       # an agent channel that carries speech throughout
        for arm, agent in (("silent", np.zeros_like(x)), ("speech", agent_speech)):
            outs = {}
            for name, w, asr2, tf32 in paths:
                outs[name] = run_path(ears, x, agent, w, RulesState, Thresholds(), energy_db, asr=asr2, tf32=tf32)
            ref_o, ref_ev, _ = outs[paths[0][0]]
            rk = sorted(ev_key(e) for e in ref_ev)
            for name, *_ in paths:
                o, ev, t = outs[name]
                k = sorted(ev_key(e) for e in ev)
                res[f"{l}/{arm}/{name}"] = {
                    "diff_vs_ref": compare(ref_o, o), "events": len(ev), "events_ref": len(ref_ev),
                    "events_equal": k == rk, "only_ref": sorted(set(rk) - set(k)), "only_this": sorted(set(k) - set(rk)),
                    "enc_s": round(t, 2)}
                print(l, arm, name, json.dumps(res[f"{l}/{arm}/{name}"]), flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
