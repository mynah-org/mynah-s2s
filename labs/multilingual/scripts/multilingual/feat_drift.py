"""CPU-only offline diagnostics on existing artifacts (no model inference beyond a 5M-param adapter on cached features).

1. Weight drift: adapter affine vs its ridge init, MLP branch magnitude; trained encoder LayerNorms vs the ML checkpoint.
2. Paired features (LibriSpeech dev-clean, held-out speakers of the ridge fit): VC encoder output vs ML encoder output
   mapped by identity / ridge / adapters. Cosine, relative error and norm ratio, overall and on the first frames of each
   utterance (leading low-energy region, a weak proxy for silence) vs the rest.
   CAVEAT: cached ML features come from the ML encoder with ORIGINAL LayerNorms; E5/V2.1 adapters were trained on top
   of trained LNs, so their rows here are an approximation (labelled "*").
"""
import json
import sys

import torch

R = str(__import__("pathlib").Path.home() / "mynah-s2s/ml/results") + "/"
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from adapters import load_adapter  # noqa: E402

torch.set_grad_enabled(False)
meta = json.load(open(R + "feats-ls/meta.json"))
rj = json.load(open(R + "maps/ridge_ml_out.json"))
held = set(rj["held_speakers"])
vc = torch.load(R + "feats-ls/vc_out.pt", map_location="cpu", weights_only=False)
ml = torch.load(R + "feats-ls/ml_out.pt", map_location="cpu", weights_only=False)
idx = [i for i, m in enumerate(meta) if m["speaker"] in held]
print(f"utterances {len(vc)}, held-out {len(idx)}")
mp = torch.load(R + "maps/ridge_ml_out.pt", map_location="cpu", weights_only=False)
M, m0 = mp["M"].float(), mp["m"].float()

out = {"drift": {}, "features": {}}
ADS = {"a1r": "distill/a1r/adapter_best.pt", "e5*": "distill/e5/adapter_best.pt",
       "v21*": "v21/distill/v21/adapter_best.pt"}
ads = {k: load_adapter(R + p).float() for k, p in ADS.items()}
for k, ad in ads.items():
    W = ad.affine.weight.T
    d = {"affine_rel_change_vs_ridge": float((W - M).norm() / M.norm()),
         "bias_rel_change": float((ad.affine.bias - m0).norm() / m0.norm().clamp_min(1e-6)),
         "down_weight_norm": float(ad.down.weight.norm())}
    out["drift"][k] = d
    print("drift", k, d)

# LayerNorm drift (E5, V2.1) vs the ML checkpoint
sd = torch.load(str(__import__("pathlib").Path.home() / "mynah-s2s/ml/ckpt/asr-ml/model_weights.ckpt"), map_location="cpu", weights_only=True)
for k, p in {"e5": "distill/e5/encoder_ln_best.pt", "v21": "v21/distill/v21/encoder_ln_best.pt"}.items():
    ln = torch.load(R + p, map_location="cpu", weights_only=True)
    rel = []
    for n, v in ln.items():
        o = sd.get("encoder." + n)
        if o is None:
            continue
        rel.append((n, float((v.float() - o.float()).norm() / o.float().norm().clamp_min(1e-6))))
    rel.sort(key=lambda t: -t[1])
    tot = sum(r for _, r in rel) / max(1, len(rel))
    out["drift"][k + "_ln"] = {"tensors": len(rel), "mean_rel_change": tot, "top5": rel[:5]}
    print("LN", k, len(rel), f"mean rel change {tot:.4f}", rel[:5])


def stats(pred, tgt):
    cos = torch.nn.functional.cosine_similarity(pred, tgt, dim=-1)
    rel = (pred - tgt).norm(dim=-1) / tgt.norm(dim=-1).clamp_min(1e-6)
    nr = pred.norm(dim=-1) / tgt.norm(dim=-1).clamp_min(1e-6)
    return {"cos": float(cos.mean()), "rel_err": float(rel.mean()), "norm_ratio": float(nr.mean()), "n": int(len(cos))}


maps = {"identity": lambda x: x, "ridge": lambda x: x @ M + m0, **{k: ad for k, ad in ads.items()}}
X = [ml[i].float() for i in idx]
Y = [vc[i].float() for i in idx]
L = [min(a.shape[0], b.shape[0]) for a, b in zip(X, Y)]
for name, f in maps.items():
    P = [f(x[:n]) for x, n in zip(X, L)]
    head = stats(torch.cat([p[:3] for p in P]), torch.cat([y[:3] for y, n in zip(Y, L)]))
    rest = stats(torch.cat([p[3:] for p in P]), torch.cat([y[3:n] for y, n in zip(Y, L)]))
    allr = stats(torch.cat(P), torch.cat([y[:n] for y, n in zip(Y, L)]))
    out["features"][name] = {"all": allr, "first3": head, "rest": rest}
    print(f"{name:9} all cos {allr['cos']:.3f} rel {allr['rel_err']:.3f} norm {allr['norm_ratio']:.2f} | "
          f"first3 cos {head['cos']:.3f} norm {head['norm_ratio']:.2f} | rest cos {rest['cos']:.3f}")
# low-energy proxy: frames whose VC feature is closest to the VC "silence" centroid (mean of the first frame)
c0 = torch.stack([y[0] for y in Y]).mean(0)
print("VC first-frame centroid norm", float(c0.norm()))
json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else "feat_drift.json", "w"), indent=1)
