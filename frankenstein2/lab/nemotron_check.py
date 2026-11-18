"""Our Nemotron runtime (ears/nemotron.py) against the transformers reference (test only):
  1. log-mel (centred and streaming chunks);
  2. offline encode: the GRU's window regime (8 s, lookahead 0, layer 15) and lookahead 3 (last layer);
  3. streaming, batched: N sessions (IT/EN/FR, some joining later) through one pool vs the reference per-session
     cache-aware streaming; last layer per chunk and the greedy RNN-T transcript (identical pieces required).
usage: python nemotron_check.py --bundle DIR --asr-dir DIR [--device cpu] [--n 3] [--chunks 40] --out out.json
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
    ap.add_argument("--asr-dir", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--chunks", type=int, default=40)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import numpy as np
    import soundfile as sf
    import torch
    from frankenstein2.ears.nemotron import NemotronASR
    from frankenstein2.ears.stream import RnntStream, StreamChannelPair
    from frankenstein2.ears.turntaker import load_bundle
    torch.set_grad_enabled(False)
    dev = a.device
    ref, proc, _, _ = load_bundle(a.bundle, device=dev)
    own = NemotronASR(a.bundle, device=dev)
    langs = ["it", "en", "fr"]
    tr = {l: sf.read(Path(a.asr_dir) / f"track_{l}.wav", dtype="float32")[0] for l in langs}
    res = {}
    # 1. mel
    x = tr["it"][:128000]
    fr = proc.feature_extractor(x, sampling_rate=16000, return_tensors="pt")["input_features"][0].to(dev)
    fo = own.mel(torch.from_numpy(x)[None].to(dev), center=True)[0][:fr.shape[0]]
    res["mel_maxabs_center"] = float((fr - fo).abs().max())
    seg = tr["it"][4000:4000 + 5520]
    fr2 = proc.feature_extractor([seg], sampling_rate=16000, center=False, return_tensors="pt")["input_features"][0][:32]
    fo2 = own.mel(torch.from_numpy(seg)[None].to(dev), center=False)[0][:32]
    res["mel_maxabs_chunk"] = float((fr2.to(dev) - fo2).abs().max())
    # 2. offline
    held = {}
    hk = ref.encoder.layers[14].register_forward_hook(lambda m, i, o: held.__setitem__("h", o[0] if isinstance(o, tuple) else o))
    ref.encoder(input_features=fr[None], num_lookahead_tokens=0)
    hk.remove()
    o15 = own.encode(fr[None], lookahead=0, stop_after=15)
    res["window_l15_maxabs"] = float((held["h"] - o15).abs().max())
    res["window_l15_shape"] = [list(held["h"].shape), list(o15.shape)]
    r3 = ref.encoder(input_features=fr[None], num_lookahead_tokens=3).last_hidden_state
    o3 = own.encode(fr[None], lookahead=3)
    res["offline_k3_last_maxabs"] = float((r3 - o3).abs().max())
    # 3. streaming, batched
    start = [0, 0, 9][: a.n] + [5 * i for i in range(3, a.n)]
    pool = own.new_pool(a.n)
    pid = [own.prompt_id(langs[i % 3]) for i in range(a.n)]
    d, h, c = own.pred_start(a.n)
    refs = [StreamChannelPair(ref, proc, dev, 3, None) for _ in range(a.n)]
    for r in refs:
        r.user_only = True
    rr = [RnntStream(ref, pid[i], dev) for i in range(a.n)]
    pieces_ref = [[] for _ in range(a.n)]
    pieces_own = [[] for _ in range(a.n)]
    maxd = 0.0
    pos = [0] * a.n
    for step in range(a.chunks + max(start)):
        live = [i for i in range(a.n) if start[i] <= step < start[i] + a.chunks]
        if not live:
            continue
        lasts_ref = {}
        for i in live:
            xx = tr[langs[i % 3]]
            u = xx[pos[i]:pos[i] + 5120]
            refs[i].push(u, np.zeros_like(u))
            pos[i] += 5120
            while refs[i].ready():
                _, last = refs[i].chunk()
                lasts_ref[i] = last[:1]
                for t, tok in rr[i].decode(rr[i].project(last[:1])[0]):
                    pieces_ref[i].append(tok)
        # own: first chunks and later chunks in their own batched calls
        groups = {f: [i for i in live if i in lasts_ref and (pool["chunks"][i] == 0) == f] for f in (True, False)}
        for first in (True, False):
            grp = groups[first]
            if not grp:
                continue
            mels = []
            for i in grp:
                p = refs[i]
                xx = tr[langs[i % 3]]
                if first:
                    w = xx[:(own_first := (1 + 8 * 3 - 1) * 160 + 200)]
                    mels.append(own.mel(torch.from_numpy(w)[None].to(dev), center=True)[0][:25])
                else:
                    lo = (p.mel_done - 32) * 160 - 256
                    w = xx[lo: lo + 32 * 160 + 400]
                    mels.append(own.mel(torch.from_numpy(w)[None].to(dev), center=False)[0][:32])
            last = own.stream_step(pool, grp, torch.stack(mels), first=first)
            enc = own.project(last, [pid[i] for i in grp])
            sel = torch.tensor(grp, device=own.dev)
            toks, d2, h2, c2 = own.greedy(enc, d[sel], h[:, sel], c[:, sel])
            d[sel], h[:, sel], c[:, sel] = d2, h2, c2
            for j, i in enumerate(grp):
                maxd = max(maxd, float((last[j:j + 1] - lasts_ref[i]).abs().max()))
                pieces_own[i] += [tk for _, tk in toks[j]]
    res["stream_last_maxabs"] = maxd
    res["stream_same_pieces"] = [pieces_ref[i] == pieces_own[i] for i in range(a.n)]
    res["texts"] = ["".join(own.pieces(p)).replace("▁", " ") for p in pieces_own]
    res["texts_ref"] = ["".join(own.pieces(p)).replace("▁", " ") for p in pieces_ref]
    Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps(res, ensure_ascii=False)[:3000])


if __name__ == "__main__":
    main()
