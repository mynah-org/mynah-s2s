"""A2 reproduction — interface maps from a source encoder (ML/EN, any layer) to VoiceChat's encoder output space.

  extract : run encoders on LibriSpeech dev-clean (or any flac/wav tree), save frame features per utterance (fp16)
  fit     : speaker-held-out fits (identity / orthogonal Procrustes / ridge over an alpha sweep), scored in the 4480-d
            space VoiceChat's LLM actually reads (through the VC proj), saves the selected map {M, m} for e2e_qa.py

Row convention: y = x @ M + m (x: source frame [1024], y: VC encoder frame [1024]).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))


def cmd_extract(a):
    from encoders import SpeechEncoder
    files = sorted(Path(a.audio).rglob("*.flac")) + sorted(Path(a.audio).rglob("*.wav"))
    if a.max_files:
        random.Random(0).shuffle(files)
        files = sorted(files[: a.max_files])
    specs = [s.split(":") for s in a.encoders.split(",")]  # name[:layers e.g. 11-17]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = [{"path": str(p), "speaker": p.parent.parent.name} for p in files]
    for spec in specs:
        name = spec[0]
        layers = [int(x) for x in spec[1].split("-")] if len(spec) > 1 else []
        enc = SpeechEncoder(name)
        store = {"out": []} | {f"L{i}": [] for i in layers}
        for i, p in enumerate(files):
            w, sr = sf.read(p, dtype="float32")
            assert sr == 16000, p
            r = enc(torch.from_numpy(w), layers)
            for k in store:
                store[k].append(r[k].half().cpu())
            if i % 200 == 0:
                print(name, i, len(files), r["out"].shape, flush=True)
        for k, v in store.items():
            torch.save(v, out / f"{name}_{k}.pt")
        del enc
        torch.cuda.empty_cache()
    (out / "meta.json").write_text(json.dumps(meta))


def r2_cos(pred, tgt, W, b):
    """metrics in the 4480-d LLM input space: y -> y @ W.T + b"""
    P, T = pred @ W.T + b, tgt @ W.T + b
    ss_res = ((P - T) ** 2).sum(0)
    ss_tot = ((T - T.mean(0)) ** 2).sum(0)
    r2 = 1 - (ss_res.sum() / ss_tot.sum())
    cos = torch.nn.functional.cosine_similarity(P, T, dim=1).mean()
    return float(r2), float(cos)


def cmd_fit(a):
    from encoders import _vc_state
    d = Path(a.feats)
    meta = json.loads((d / "meta.json").read_text())
    X = torch.load(d / f"{a.src}.pt")
    Y = torch.load(d / "vc_out.pt")
    spk = sorted({m["speaker"] for m in meta})
    random.Random(0).shuffle(spk)
    held = set(spk[: max(1, len(spk) // 5)])
    tr = [i for i, m in enumerate(meta) if m["speaker"] not in held]
    te = [i for i, m in enumerate(meta) if m["speaker"] in held]

    def stack(idx):
        xs, ys = [], []
        for i in idx:
            n = min(len(X[i]), len(Y[i]))
            assert abs(len(X[i]) - len(Y[i])) <= 1, (i, len(X[i]), len(Y[i]))
            xs.append(X[i][:n]); ys.append(Y[i][:n])
        return torch.cat(xs).double(), torch.cat(ys).double()

    Xtr, Ytr = stack(tr)
    Xte, Yte = stack(te)
    *_, proj = _vc_state()
    W, b = proj["weight"].double(), proj["bias"].double()
    mx, my = Xtr.mean(0), Ytr.mean(0)
    Xc, Yc = Xtr - mx, Ytr - my
    Cxx, Cxy = Xc.T @ Xc, Xc.T @ Yc
    res = {"src": a.src, "train_frames": len(Xtr), "test_frames": len(Xte), "held_speakers": sorted(held), "maps": {}}
    maps = {"identity": (torch.eye(1024, dtype=torch.float64), torch.zeros(1024, dtype=torch.float64))}
    U, S, Vh = torch.linalg.svd(Cxy)
    R = U @ Vh
    scale = S.sum() / torch.trace(Cxx)
    maps["orthogonal"] = (R * scale, my - (mx @ R) * scale)
    for alpha in [1e-4, 1e-3, 1e-2, 1e-1, 1.0]:
        lam = alpha * torch.diagonal(Cxx).mean()
        M = torch.linalg.solve(Cxx + lam * torch.eye(1024, dtype=torch.float64), Cxy)
        maps[f"ridge_{alpha:g}"] = (M, my - mx @ M)
    best, best_r2 = None, -1e9
    for k, (M, m) in maps.items():
        r2, cos = r2_cos(Xte @ M + m, Yte, W, b)
        r2tr, _ = r2_cos(Xtr[:200000] @ M + m, Ytr[:200000], W, b)
        res["maps"][k] = {"test_r2": round(r2, 4), "test_cos": round(cos, 4), "train_r2": round(r2tr, 4)}
        print(k, res["maps"][k], flush=True)
        if k.startswith("ridge") and r2 > best_r2:
            best, best_r2 = k, r2
    res["selected"] = best
    M, m = maps[best]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"M": M.float(), "m": m.float(), "src": a.src, "fit": best}, out)
    (out.with_suffix(".json")).write_text(json.dumps(res, indent=1))
    print("selected", best, "->", out)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    e = sp.add_parser("extract")
    e.add_argument("--audio", required=True)
    e.add_argument("--encoders", default="vc,en,ml:11-17")
    e.add_argument("--max-files", type=int, default=0)
    e.add_argument("--out", required=True)
    f = sp.add_parser("fit")
    f.add_argument("--feats", required=True)
    f.add_argument("--src", default="ml_out")
    f.add_argument("--out", required=True)
    a = ap.parse_args()
    {"extract": cmd_extract, "fit": cmd_fit}[a.cmd](a)


if __name__ == "__main__":
    main()
