"""B1 — VoiceChat-11B Thinker LLM vs the official NVIDIA-Nemotron-Nano-9B-v2 checkpoint, tensor by tensor (CPU).

Mapping (verified on the safetensors headers, same shapes): stt_model.llm.X <-> backbone.X,
stt_model.embed_tokens.weight <-> backbone.embeddings.weight, stt_model.lm_head.weight <-> lm_head.weight.
Extra in VoiceChat: stt_model.function_head.weight (compared with lm_head and with Nano's lm_head).
VoiceChat stores FP32, Nano BF16: `bf16_exact` = VC tensor rounded to BF16 equals Nano bit-for-bit (i.e. the tensor
was only upcast, never trained). Also: relative L2 of the difference, cosine, fraction of changed elements, and for
the (131072 x 4480) matrices the per-row change so special/added tokens can be identified.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open


def open_nano(d: Path):
    idx = {}
    for p in sorted(glob.glob(str(d / "model-*.safetensors"))):
        with safe_open(p, framework="pt") as f:
            for k in f.keys():
                idx[k] = p
    return idx


def get(idx, k):
    with safe_open(idx[k], framework="pt") as f:
        return f.get_tensor(k)


def cmp(a: torch.Tensor, b: torch.Tensor, rows: bool = False) -> dict:
    a32, b32 = a.float(), b.float()
    d = a32 - b32
    # float64 chunked accumulation: float32 reductions over ~6e8 elements are not trustworthy (cos > 1 seen)
    dot = na = nb2 = nd = 0.0
    fa, fb = a32.flatten(), b32.flatten()
    for i in range(0, fa.numel(), 1 << 24):
        x, y = fa[i:i + (1 << 24)].double(), fb[i:i + (1 << 24)].double()
        dot += float(x @ y); na += float(x @ x); nb2 += float(y @ y); nd += float((x - y) @ (x - y))
    nb = nb2 ** 0.5
    r = {"bf16_exact": bool(torch.equal(a.to(torch.bfloat16), b.to(torch.bfloat16))),
         "frac_changed_bf16": (a.to(torch.bfloat16) != b.to(torch.bfloat16)).float().mean().item(),
         "rel_l2": nd ** 0.5 / nb if nb else None,
         "cos": dot / (na ** 0.5 * nb) if na and nb else None}
    if rows and a.dim() == 2:
        rn = d.norm(dim=1) / b32.norm(dim=1).clamp_min(1e-12)
        top = torch.topk(rn, 40)
        r["row_rel_l2"] = {"median": rn.median().item(), "p99": rn.quantile(0.99).item() if rn.numel() < 2**24 else None,
                           "max": rn.max().item(), "n_rows_changed": int((rn > 1e-3).sum()),
                           "top_rows": [[int(i), round(float(v), 4)] for v, i in zip(top.values, top.indices)]}
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vc", required=True, type=Path)
    ap.add_argument("--nano", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    nano = open_nano(a.nano)
    vcf = safe_open(str(a.vc), framework="pt")
    vck = [k for k in vcf.keys() if k.startswith("stt_model.") and not k.startswith(("stt_model.perception", "stt_model.rnnt"))]
    pattern = json.load(open(a.nano / "config.json"))["hybrid_override_pattern"]
    rows = []
    for k in sorted(vck):
        if k.startswith("stt_model.llm."):
            nk = "backbone." + k[len("stt_model.llm."):]
        elif k == "stt_model.embed_tokens.weight":
            nk = "backbone.embeddings.weight"
        elif k == "stt_model.lm_head.weight":
            nk = "lm_head.weight"
        else:
            nk = None
        a_t = vcf.get_tensor(k)
        r = {"vc": k, "nano": nk, "shape": list(a_t.shape)}
        if nk is None or nk not in nano:
            r["note"] = "no counterpart in Nano-9B-v2"
        else:
            b_t = get(nano, nk)
            if a_t.shape != b_t.shape:
                r["note"] = f"shape {list(a_t.shape)} vs {list(b_t.shape)}"
            else:
                r.update(cmp(a_t, b_t, rows=a_t.dim() == 2 and a_t.shape[0] == 131072))
        m = re.match(r"stt_model\.llm\.layers\.(\d+)\.", k)
        if m:
            r["layer"] = int(m.group(1))
            r["block"] = {"M": "mamba2", "-": "mlp", "*": "attention"}[pattern[int(m.group(1))]]
        rows.append(r)
        print(k, {x: r.get(x) for x in ("bf16_exact", "rel_l2", "cos", "frac_changed_bf16", "note")}, flush=True)
    # function head: compare with VC lm_head, Nano lm_head and embeddings
    fh = vcf.get_tensor("stt_model.function_head.weight")
    extra = {"function_head~vc_lm_head": cmp(fh, vcf.get_tensor("stt_model.lm_head.weight")),
             "function_head~nano_lm_head": cmp(fh, get(nano, "lm_head.weight")),
             "vc_lm_head~vc_embed(tied?)": cmp(vcf.get_tensor("stt_model.lm_head.weight"), vcf.get_tensor("stt_model.embed_tokens.weight"))}
    # VC tensors bf16-representable at all? (were weights ever trained in fp32 after the bf16 release)
    agg = defaultdict(lambda: {"n": 0, "params": 0, "bf16_exact": 0, "rel_l2_w": 0.0})
    for r in rows:
        if "rel_l2" not in r:
            continue
        key = r.get("block", r["vc"].split(".")[1])
        n = 1
        for s in r["shape"]:
            n *= s
        g = agg[key]
        g["n"] += 1
        g["params"] += n
        g["bf16_exact"] += int(r["bf16_exact"])
        g["rel_l2_w"] += r["rel_l2"] * n
    summary = {k: {"tensors": v["n"], "params_M": round(v["params"] / 1e6, 1), "bf16_exact_tensors": v["bf16_exact"],
                   "mean_rel_l2": round(v["rel_l2_w"] / v["params"], 5)} for k, v in agg.items()}
    per_layer = [{"layer": r["layer"], "block": r["block"], "t": r["vc"].split(".")[-2] + "." + r["vc"].split(".")[-1],
                  "rel_l2": round(r["rel_l2"], 5), "cos": round(r["cos"], 5)} for r in rows if "layer" in r and "rel_l2" in r]
    out = {"summary": summary, "extra": extra, "untrained_tensors": [r["vc"] for r in rows if r.get("bf16_exact")],
           "per_layer": per_layer}
    (a.out / "llm_weight_diff.json").write_text(json.dumps({"rows": rows, **out}, indent=1))
    print(json.dumps({k: out[k] for k in ("summary", "extra")}, indent=1))
    print("untrained:", len(out["untrained_tensors"]), out["untrained_tensors"][:20])


if __name__ == "__main__":
    main()
