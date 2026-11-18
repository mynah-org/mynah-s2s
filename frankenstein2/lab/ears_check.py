"""CPU/GPU check of the streaming ears on one user track (agent channel silent, listen mode throughout, with the
rules' resets after each endpoint):

  * streaming (cache-aware, 320 ms chunks) vs offline (whole track, chunked attention of the same lookahead):
    max / mean abs difference of the last encoder layer and of layer 15, cosine of L15 per frame;
  * the streaming transcript vs the offline transcript;
  * the endpoint / backchannel events and the p_end trace around each end of speech.

usage: python ears_check.py --bundle DIR --wav track.wav [--cut-s 52] [--lang it] [--device cpu] --out out.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():  # pragma: no cover - model run
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--wav", required=True)
    ap.add_argument("--cut-s", type=float, default=0.0)
    ap.add_argument("--lang", default="it")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--lookahead", type=int, default=3)
    ap.add_argument("--gru-k", type=int, default=0, help="-1 = shared pass")
    ap.add_argument("--gru-features", default="window")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import numpy as np
    import soundfile as sf
    import torch
    from frankenstein2.ears.rules import Thresholds
    from frankenstein2.ears.server import PROMPTS
    from frankenstein2.ears.stream import HOP, EarsSession, RnntStream, is_tag
    from frankenstein2.ears.turntaker import L15, load_bundle
    torch.set_grad_enabled(False)
    t0 = time.monotonic()
    model, proc, tt, _ = load_bundle(a.bundle)
    model.to(a.device), tt.to(a.device)
    load_s = time.monotonic() - t0
    x, sr = sf.read(a.wav, dtype="float32")
    assert sr == 16000, sr
    if x.ndim > 1:
        x = x.mean(1)
    if a.cut_s:
        x = x[: int(a.cut_s * 16000)]
    x = x[: len(x) // HOP * HOP]
    pid = PROMPTS[a.lang]
    s = EarsSession(model, proc, tt, a.device, pid, Thresholds(), k=a.lookahead, gru_k=None if a.gru_k < 0 else a.gru_k,
                    gru_features=a.gru_features)
    l15s, lasts = [], []
    orig = s.pair.chunk

    def spy():
        l, h = orig()
        l15s.append(l[0].cpu() if l.shape[0] > 1 or a.gru_features != "window" else l[0].cpu())
        lasts.append(h[0].cpu())
        return l, h
    s.pair.chunk = spy
    hops, evs, frames, pieces = [], [], [], []
    for n in range(len(x) // HOP):
        r = s.hop(x[n * HOP:(n + 1) * HOP], np.zeros(HOP, np.float32), "listen")
        hops.append(r["ms"])
        evs += r["events"]
        frames += r["frames"]
        pieces += r["pieces"]
    stream_text = "".join(p["p"] for p in pieces if not is_tag(p["p"])).replace("▁", " ").strip()
    L_s = torch.cat(lasts, 0)
    l15_s = torch.cat(l15s, 0)
    # offline, same lookahead (chunked_limited mask over the whole track)
    f = proc.feature_extractor(x, sampling_rate=16000, return_tensors="pt")
    held = {}
    hd = model.encoder.layers[L15 - 1].register_forward_hook(lambda m, a_, o: held.__setitem__("h", o))
    o = model.encoder(input_features=f["input_features"].to(a.device), num_lookahead_tokens=a.lookahead)
    hd.remove()
    L_o, l15_o = o.last_hidden_state[0].cpu(), (held["h"][0] if isinstance(held["h"], tuple) else held["h"])[0].cpu()
    n = min(len(L_s), len(L_o))
    r = RnntStream(model, pid, a.device)
    toks = [t for _, t in r.decode(r.project(o.last_hidden_state)[0])]
    off_text = "".join(p for p in proc.tokenizer.convert_ids_to_tokens(toks) if not is_tag(p)).replace("▁",
                                                                                                       " ").strip()
    cos = torch.nn.functional.cosine_similarity(l15_s[:n], l15_o[:n], dim=-1)
    res = {"load_s": round(load_s, 1), "frames_stream": len(L_s), "frames_offline": len(L_o),
           "last_maxabs": float((L_s[:n] - L_o[:n]).abs().max()), "last_meanabs": float((L_s[:n] - L_o[:n]).abs().mean()),
           "l15_maxabs": float((l15_s[:n] - l15_o[:n]).abs().max()), "l15_cos_mean": float(cos.mean()),
           "l15_cos_min": float(cos.min()), "stream_text": stream_text, "offline_text": off_text,
           "same_text": stream_text == off_text, "hop_ms_p50": sorted(hops)[len(hops) // 2], "hop_ms_max": max(hops),
           "events": evs, "p_end": [f_["p_end"] for f_ in frames], "db": [f_["db"] for f_ in frames]}
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False))
    print(json.dumps({k: v for k, v in res.items() if k not in ("p_end", "db")}, ensure_ascii=False)[:4000])


if __name__ == "__main__":
    main()
