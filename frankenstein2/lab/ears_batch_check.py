"""Correctness of the batched ears (hop_batch: windows + GRU + streaming ASR + RNN-T for all sessions at once)
against per-session hops: N sessions in IT/EN/FR, some joining later, mode switches and resets; transcripts and
events must be identical, p_end within float tolerance. Also times both paths (GPU: per round of hops).

usage: python ears_batch_check.py --bundle DIR --asr-dir DIR [--device cuda] [--n 4] [--hops 60] --out out.json
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
    ap.add_argument("--asr-dir", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--hops", type=int, default=60)
    ap.add_argument("--out", required=True)
    ap.add_argument("--impl", default="hf_batched", choices=["hf_batched", "own"],
                    help="batched path to check: hop_batch on the transformers modules, or our own runtime (OwnEars)")
    a = ap.parse_args()
    import numpy as np
    import soundfile as sf
    import torch
    from frankenstein2.ears.rules import Thresholds
    from frankenstein2.ears.server import PROMPTS
    from frankenstein2.ears.stream import HOP, EarsSession, hop_batch
    from frankenstein2.ears.turntaker import load_bundle
    torch.set_grad_enabled(False)
    m, p, t, _ = load_bundle(a.bundle, device=a.device)
    langs = ["it", "en", "fr"]
    tr = {l: sf.read(Path(a.asr_dir) / f"track_{l}.wav", dtype="float32")[0] for l in langs}
    start = [0 if i < a.n // 2 else 7 + 3 * i for i in range(a.n)]      # half join later
    mk = lambda i: EarsSession(m, p, t, a.device, PROMPTS[langs[i % 3]], Thresholds())  # noqa: E731
    ref = [mk(i) for i in range(a.n)]
    if a.impl == "own":
        from frankenstein2.ears.own import OwnEars
        own = OwnEars(a.bundle, device=a.device, slots=a.n + 1)
        for i in range(a.n):
            own.open(i, langs[i % 3])
    else:
        bat = [mk(i) for i in range(a.n)]
    z = np.zeros(HOP, np.float32)
    agent = (0.05 * np.sin(np.arange(HOP) * 0.07)).astype(np.float32)
    R = {i: {"p": [], "e": [], "pe": []} for i in range(a.n)}
    Bt = {i: {"p": [], "e": [], "pe": []} for i in range(a.n)}
    t_ref, t_bat = [], []
    for h in range(a.hops):
        items_r, items_b = [], []
        for i in range(a.n):
            if h < start[i]:
                continue
            k = h - start[i]
            x = tr[langs[i % 3]]
            u = x[k * HOP:(k + 1) * HOP]
            if len(u) < HOP:
                u = np.concatenate([u, np.zeros(HOP - len(u), np.float32)])
            md = "speaking" if 25 <= k < 33 else "listen"
            ag = agent if md == "speaking" else z
            items_r.append((i, u, ag, md, k == 33))
        t0 = time.monotonic()
        for i, u, ag, md, rs in items_r:
            r = ref[i].hop(u, ag, md, rs)
            R[i]["p"] += [q["p"] for q in r["pieces"]]
            R[i]["e"] += [(e["kind"], e["k"]) for e in r["events"]]
            R[i]["pe"] += [f["p_end"] for f in r["frames"]]
        if a.device != "cpu":
            torch.cuda.synchronize()
        t_ref.append(time.monotonic() - t0)
        t0 = time.monotonic()
        if a.impl == "own":
            rs_ = own.hop([(i, u, ag, md, rs) for i, u, ag, md, rs in items_r])
        else:
            rs_ = hop_batch([(bat[i], u, ag, md, rs) for i, u, ag, md, rs in items_r])
        if a.device != "cpu":
            torch.cuda.synchronize()
        t_bat.append(time.monotonic() - t0)
        for (i, *_), r in zip(items_r, rs_):
            Bt[i]["p"] += [q["p"] for q in r["pieces"]]
            Bt[i]["e"] += [(e["kind"], e["k"]) for e in r["events"]]
            Bt[i]["pe"] += [f["p_end"] for f in r["frames"]]
    res = {"n": a.n, "device": a.device, "sessions": []}
    ok = True
    for i in range(a.n):
        same_p, same_e = R[i]["p"] == Bt[i]["p"], R[i]["e"] == Bt[i]["e"]
        d = max([abs(x - y) for x, y in zip(R[i]["pe"], Bt[i]["pe"])] or [0])
        ok &= same_p and same_e and d < 2e-3
        res["sessions"].append({"lang": langs[i % 3], "start": start[i], "same_pieces": same_p, "same_events": same_e,
                                "max_dp_end": d, "text": "".join(R[i]["p"]).replace("▁", " ")[:200],
                                "text_batched": "".join(Bt[i]["p"]).replace("▁", " ")[:200]})
    q = lambda xs: (round(sorted(xs)[len(xs) // 2] * 1e3, 1), round(sorted(xs)[int(.95 * (len(xs) - 1))] * 1e3, 1))  # noqa
    res.update(ok=ok, round_ms_ref_p50_p95=q(t_ref[10:]), round_ms_batched_p50_p95=q(t_bat[10:]))
    Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in res.items() if k != "sessions"}))
    for x in res["sessions"]:
        print(json.dumps(x, ensure_ascii=False)[:400])


if __name__ == "__main__":
    main()
