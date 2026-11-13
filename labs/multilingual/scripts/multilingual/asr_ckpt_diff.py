"""A1 — tensor-level diff of the speech encoders: VoiceChat-11B perception vs the standalone NVIDIA streaming ASRs.

CPU only. Reads the VoiceChat safetensors lazily (only perception/RNNT keys) and the two .nemo weight files
(already extracted with `tar xf`). For every tensor present in at least two of {VC, EN, ML} it records shape match,
exact equality, relative L2 distance and cosine similarity, plus the cosine between the two fine-tuning deltas
(VC-EN) and (ML-EN): if VoiceChat's adaptation and the multilingual training moved the English ancestor in
unrelated directions, task arithmetic (EN + d_VC + d_ML) is at least not self-cancelling.

Output: <out>/asr_ckpt_diff.json (per tensor) + <out>/asr_ckpt_diff_summary.json (per module/layer) on stdout too.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open

VC_PREFIX = {"stt_model.perception.encoder.": "encoder.", "stt_model.perception.preprocessor.": "preprocessor.",
             "stt_model.rnnt_decoder.": "decoder.", "stt_model.rnnt_joint.": "joint."}


def load_vc(path: Path) -> dict[str, torch.Tensor]:
    out = {}
    with safe_open(str(path), framework="pt", device="cpu") as f:
        for k in f.keys():
            for p, q in VC_PREFIX.items():
                if k.startswith(p):
                    out[q + k[len(p):]] = f.get_tensor(k)
            if k.startswith("stt_model.perception.proj."):
                out["vc_proj." + k.rsplit(".", 1)[1]] = f.get_tensor(k)
    return out


def load_nemo(d: Path) -> dict[str, torch.Tensor]:
    sd = torch.load(d / "model_weights.ckpt", map_location="cpu", weights_only=True)
    return {k: v for k, v in sd.items() if torch.is_tensor(v)}


def stats(a: torch.Tensor, b: torch.Tensor) -> dict:
    if a.shape != b.shape:
        return {"shape_match": False}
    if not a.is_floating_point():
        return {"shape_match": True, "exact": bool(torch.equal(a, b))}
    a, b = a.double().flatten(), b.double().flatten()
    nb = b.norm().item()
    return {"shape_match": True, "exact": bool(torch.equal(a, b)),
            "rel_l2": (a - b).norm().item() / nb if nb > 0 else None,
            "cos": torch.nn.functional.cosine_similarity(a, b, dim=0).item() if a.norm() > 0 and nb > 0 else None}


def group(name: str) -> str:
    m = re.match(r"encoder\.layers\.(\d+)\.(\w+)", name)
    if m:
        return f"encoder.L{int(m.group(1)):02d}.{m.group(2)}"
    return ".".join(name.split(".")[:2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vc", required=True, type=Path, help="VoiceChat model.safetensors")
    ap.add_argument("--en", required=True, type=Path, help="extracted nemotron-speech-streaming-en-0.6b .nemo dir")
    ap.add_argument("--ml", required=True, type=Path, help="extracted nemotron-3.5-asr-streaming-0.6b .nemo dir")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    W = {"VC": load_vc(a.vc), "EN": load_nemo(a.en), "ML": load_nemo(a.ml)}
    keys = sorted(set().union(*[set(w) for w in W.values()]))
    rows = []
    for k in keys:
        have = [n for n in W if k in W[n]]
        r = {"name": k, "in": have, "shape": {n: list(W[n][k].shape) for n in have}}
        for x, y in (("VC", "EN"), ("VC", "ML"), ("EN", "ML")):
            if x in have and y in have:
                r[f"{x}~{y}"] = stats(W[x][k], W[y][k])
        if all(n in have for n in W) and len({tuple(W[n][k].shape) for n in W}) == 1 and W["EN"][k].is_floating_point():
            dv = (W["VC"][k] - W["EN"][k]).double().flatten()
            dm = (W["ML"][k] - W["EN"][k]).double().flatten()
            if dv.norm() > 0 and dm.norm() > 0:
                r["cos(dVC,dML)"] = torch.nn.functional.cosine_similarity(dv, dm, dim=0).item()
                r["|dVC|/|EN|"] = dv.norm().item() / max(W["EN"][k].double().norm().item(), 1e-12)
                r["|dML|/|EN|"] = dm.norm().item() / max(W["EN"][k].double().norm().item(), 1e-12)
        rows.append(r)
    (a.out / "asr_ckpt_diff.json").write_text(json.dumps(rows, indent=1))

    # per-group summary (parameter-count-weighted means over float tensors with a common shape)
    agg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        g = group(r["name"])
        n = 1
        for s in r["shape"].values():
            n = max(n, int(torch.tensor(s).prod().item()) if s else 1)
        for pair in ("VC~EN", "VC~ML", "EN~ML"):
            s = r.get(pair)
            if s and s.get("cos") is not None:
                agg[g][pair].append((s["cos"], n))
        if "cos(dVC,dML)" in r:
            agg[g]["cos(dVC,dML)"].append((r["cos(dVC,dML)"], n))
    summary = {}
    for g, d in agg.items():
        summary[g] = {p: round(sum(c * n for c, n in v) / sum(n for _, n in v), 4) for p, v in d.items()}
    only = {n: sorted(k for k in W[n] if all(k not in W[m] for m in W if m != n)) for n in W}
    mism = [r["name"] for r in rows if any(r.get(p, {}).get("shape_match") is False for p in ("VC~EN", "VC~ML", "EN~ML"))]
    exact = {p: sum(1 for r in rows if r.get(p, {}).get("exact")) for p in ("VC~EN", "VC~ML", "EN~ML")}
    common = {p: sum(1 for r in rows if p in r) for p in ("VC~EN", "VC~ML", "EN~ML")}
    out = {"tensors": {n: len(W[n]) for n in W}, "only_in": only, "shape_mismatch": mism,
           "exact_equal": exact, "compared": common, "groups": summary}
    (a.out / "asr_ckpt_diff_summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "groups"}, indent=1))
    for g in sorted(summary):
        print(f"{g:40s}", "  ".join(f"{p}={v:+.3f}" for p, v in summary[g].items()))


if __name__ == "__main__":
    main()
