"""Does an interface (ridge map / adapter) keep different utterances distinct, or collapse them onto the average
teacher representation? Encoders + interface only (no LLM).

For every QA item and language X, the utterance vector = mean over speech frames of the 4480-d LLM-input embedding.
Teacher = VC encoder + VC proj on the EN clip of the same item. Reported per (interface, X):
  * retrieval top-1 / median rank of X → EN-teacher among the N items (chance 1/N), centred cosine
  * spread: mean pairwise centred cosine between DIFFERENT items' vectors (high = collapse) and the ratio of between-item
    variance to the teacher's between-item variance
  * frame-level: mean per-frame cosine to the frame mean inside an utterance (temporal flatness)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import soundfile as sf
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa-root", required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--voice", default="aiden")
    ap.add_argument("--langs", default="en,it,fr,de,es")
    ap.add_argument("--map", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from adapters import load_adapter
    from encoders import SpeechEncoder, _vc_state

    rows = [json.loads(l) for l in open(Path(a.qa_root) / "manifest.jsonl")]
    rows = [r for r in rows if r["split"] == a.split and r["voice"] == a.voice]
    ids = sorted({r["id"] for r in rows})
    path = {(r["id"], r["lang"]): Path(a.qa_root) / r["path"] for r in rows}
    *_, proj = _vc_state()
    W, b = proj["weight"].cuda(), proj["bias"].cuda()
    m = torch.load(a.map, map_location="cuda", weights_only=True)
    ad = load_adapter(a.adapter).cuda()
    vc, ml = SpeechEncoder("vc"), SpeechEncoder("ml")

    def frames(enc, p):
        w = torch.from_numpy(sf.read(p, dtype="float32")[0])
        n = int(math.ceil(len(w) / 1280))
        return enc(w)["out"][:n]

    with torch.no_grad():
        T = torch.stack([(frames(vc, path[(i, "en")]) @ W.T + b).mean(0) for i in ids]).double()
        res = {"n_items": len(ids), "chance_top1": 1 / len(ids)}

        def stats(V, name):
            Tc, Vc = T - T.mean(0), V - V.mean(0)
            S = torch.nn.functional.normalize(Vc, dim=1) @ torch.nn.functional.normalize(Tc, dim=1).T
            ranks = (S > S.diag()[:, None]).sum(1) + 1
            Vn = torch.nn.functional.normalize(Vc, dim=1)
            off = (Vn @ Vn.T)[~torch.eye(len(ids), dtype=bool, device=V.device)].mean()
            return {"top1": float((ranks == 1).float().mean()), "median_rank": float(ranks.float().median()),
                    "diag_cos": float(S.diag().mean()), "between_item_cos": float(off),
                    "between_var_ratio": float(Vc.pow(2).sum() / Tc.pow(2).sum())}

        for lang in a.langs.split(","):
            raw, rid, ada, flat = [], [], [], []
            for i in ids:
                f = frames(ml, path[(i, lang)])
                raw.append((f @ W.T + b).mean(0))
                rid.append(((f @ m["M"] + m["m"]) @ W.T + b).mean(0))
                e = ad(f) @ W.T + b
                ada.append(e.mean(0))
                flat.append(float(torch.nn.functional.cosine_similarity(e, e.mean(0, keepdim=True)).mean()))
            res[lang] = {"ml_raw": stats(torch.stack(raw).double(), "raw"), "ridge": stats(torch.stack(rid).double(), "ridge"),
                         "adapter": stats(torch.stack(ada).double(), "adapter"), "adapter_frame_flatness": sum(flat) / len(flat)}
            if lang != "en":
                vcx = torch.stack([(frames(vc, path[(i, lang)]) @ W.T + b).mean(0) for i in ids]).double()
                res[lang]["vc_on_X (floor)"] = stats(vcx, "vc")
            print(lang, json.dumps(res[lang]), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    if Path(a.out).exists():
        raise SystemExit(f"{a.out} exists")
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
