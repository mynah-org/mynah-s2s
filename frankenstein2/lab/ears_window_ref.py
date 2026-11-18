"""Reference: the GRU on Tyger's own feature regime (8 s rolling window re-encoded every 320 ms hop with lookahead
0, layer 15 of the last 4 frames), same rules, on one user track with a silent agent channel. Compared with the
streaming ears (ears_check.py) it tells whether a difference in endpoints comes from the features (k=3 cache-aware
streaming vs the trained regime) or from somewhere else.

usage: python ears_window_ref.py --bundle DIR --wav track.wav --cut-s 52 --out out.json [--lookahead 0]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():  # pragma: no cover - model run
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--wav", required=True)
    ap.add_argument("--cut-s", type=float, default=0.0)
    ap.add_argument("--lookahead", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import numpy as np
    import soundfile as sf
    import torch
    from frankenstein2.ears.rules import RulesState, Thresholds, energy_db
    from frankenstein2.ears.turntaker import L15, GruSession, load_bundle
    torch.set_grad_enabled(False)
    model, proc, tt, _ = load_bundle(a.bundle)
    x, _ = sf.read(a.wav, dtype="float32")
    if a.cut_s:
        x = x[: int(a.cut_s * 16000)]
    HOP, W = 5120, 128000
    held = {}
    model.encoder.layers[L15 - 1].register_forward_hook(lambda m, i, o: held.__setitem__("h", o))
    zero_cache = {}

    def l15(win):
        key = len(win)
        if not win.any() and key in zero_cache:
            return zero_cache[key]
        f = proc.feature_extractor(win, sampling_rate=16000, return_tensors="pt")
        model.encoder(input_features=f["input_features"], num_lookahead_tokens=a.lookahead)
        h = held["h"]
        h = (h[0] if isinstance(h, tuple) else h)[0]
        if not win.any():
            zero_cache[key] = h
        return h
    gru, rules = GruSession(tt, "cpu"), RulesState(thr=Thresholds())
    p_end, events = [], []
    for n in range(len(x) // HOP):
        end = (n + 1) * HOP
        u = x[max(0, end - W):end]
        hu, ha = l15(u), l15(np.zeros_like(u))
        for j in range(4):
            fi = hu.shape[0] - 4 + j
            seg = x[end - HOP + j * 1280: end - HOP + (j + 1) * 1280]
            db = energy_db(seg)
            out = gru.step(hu[fi], ha[fi], db, rules.gru_prev_state())
            evs, rst = rules.on_frame(out, db, left=3 - j)
            p_end.append(out[7])
            events += [e for e in evs if e["kind"] != "FRAME"]
            if rst:
                gru.reset()
    Path(a.out).write_text(json.dumps({"events": events, "p_end": p_end}))
    print(json.dumps(events))
    for e in (149, 345, 533, 598):
        print(e, [round(v, 2) for v in p_end[e - 2:e + 20]])


if __name__ == "__main__":
    main()
