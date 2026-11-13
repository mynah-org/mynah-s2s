"""A0 (owner GO 2026-11-13, CPU only): does the multilingual encoder still carry the spoken IT/FR slot values?

Offline greedy RNNT readout of G-LEX-dev + val clips with the ML ASR model's own decoder and joint
(`EncDecRNNTBPEModelWithPrompt` semantics: joint(prompt_kernel(cat[enc, onehot(lang)]), pred), as
perception_swap.PromptJoint and the wrapper's _rnnt_step, max 10 symbols per frame).
Factors kept separate (the owner: do not call the encoder poor from a wrong transcript alone):
  encoder   E0 = ML encoder, original LNs, att [56, 0] (the bridge's encoder config without its LN change)
            E13 = same, att [56, 13] (the model's own best lookahead: how much the zero-lookahead costs)
            S103 = the D-V3 step-103 LNs (the encoder the bridge actually reads; the RNNT was not trained on it)
  decoder   lang prompt auto (101) vs forced (en 0 / it 15 / fr 8); greedy only
  input     the bridge session (cs = 100 frames of digital silence, the clip, 3 s tail); frames from cs - 2 decoded
Scoring layers per clip (separately):
  entity     strict slot checks of verify_asr_v2.check (name / postcode / concept / attributes)
  meaning    lenient: name fuzzy 0.6; postcode = any 5-digit reading within 1 digit; concept = any IT/FR form or synonym
             at 0.7; attributes as strict
  canonical  the English canonical query string the API needs appears in the transcript (expected ~0 for IT/FR)
  CER        vs the TTS input text (manifest `text`), case/punctuation folded
CPU only: CUDA is hidden by the job; one JSON line per (clip, arm), fsync'd.
Usage: a0_rnnt_readout.py --sets free_sets.json --glex-data D --val-data D --ln103 F --out DIR [--sets-keep glex_dev,val]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lexicon_v2 as L  # noqa: E402
import verify_asr_v2 as V  # noqa: E402

LANG_IDX = {"auto": 101, "en": 0, "it": 15, "fr": 8}
TAG = re.compile(r"<[a-z]{2}(-[A-Z]{2})?>")


def detok(pieces: list[str]) -> str:
    return TAG.sub("", "".join(pieces).replace("▁", " ")).strip()


def cer(ref: str, hyp: str) -> float:
    r, h = V.fold(ref).replace(" ", ""), V.fold(hyp).replace(" ", "")
    d = list(range(len(h) + 1))
    for i, a in enumerate(r, 1):
        p, d[0] = d[0], i
        for j, b in enumerate(h, 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (a != b))
    return round(d[-1] / max(1, len(r)), 4)


def five_digit_readings(hyp: str, lang: str) -> set[str]:
    out = set()
    for gp in (False, True):
        for k in (3, 4):
            r = L.parse_spoken_digits(hyp, lang, gp, max_k=k)
            out |= {r[i:i + 5] for i in range(max(0, len(r) - 4))}
    out |= set(re.findall(r"\d{5}", hyp.replace(" ", "")))
    return out


def score(row: dict, hyp: str) -> dict:
    s, t, lang = row["slots"], row["target"], row["lang"]
    strict_bad = V.check(row, hyp)
    ent, mean = {}, {}
    if t["name"] == "find_customer":
        if s.get("name"):
            ent["name"] = "name" not in strict_bad
            mean["name"] = V.name_ok(hyp, s["name"], thr=0.6)
        pc = (t.get("arguments") or {}).get("postcode")
        if pc:
            ent["postcode"] = "postcode" not in strict_bad
            mean["postcode"] = any(sum(a != b for a, b in zip(r, pc)) <= 1 for r in five_digit_readings(hyp, lang))
    if t["name"] == "search_catalog" and s.get("concept"):
        c = next(x for x in L.CONCEPTS if x[0] == s["concept"])
        forms = [c[0]] if lang == "en" else ([c[2][0]] + list(c[3]) if lang == "it" else [c[4][0]] + list(c[5]))
        syn = (s.get("synonym") or {}).get(lang)
        forms += [syn] if syn else []
        ent["concept"] = "concept" not in strict_bad
        mean["concept"] = any(V.contains(hyp, f, thr=0.7) for f in forms if f)
        ent["attrs"] = not any(b.startswith("attr:") for b in strict_bad)
        mean["attrs"] = ent["attrs"]
    canon = None
    if t["name"] == "search_catalog":
        canon = V.fold(t["arguments"]["query"]) in V.fold(hyp)
    return {"entity": ent, "meaning": mean, "canonical_en": canon, "strict_bad": strict_bad}


def main():  # pragma: no cover - CPU model run
    import soundfile as sf
    import torch
    import yaml
    from omegaconf import OmegaConf
    from nemo.collections.asr.modules.rnnt import RNNTDecoder, RNNTJoint
    from encoders import ASR_DIRS, SpeechEncoder
    from session_distill import session_wav
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", required=True)
    ap.add_argument("--sets-keep", default="glex_dev,val")
    ap.add_argument("--glex-data", required=True)
    ap.add_argument("--val-data", required=True)
    ap.add_argument("--ln103", required=True)
    ap.add_argument("--cs", type=int, default=100)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    assert not torch.cuda.is_available(), "A0 is CPU-only: run with CUDA_VISIBLE_DEVICES="
    torch.set_num_threads(a.threads)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    jl = out / "a0_rows.jsonl"
    if jl.exists():
        raise SystemExit(f"{jl} exists")
    t0 = time.monotonic()
    rows = [r for r in json.load(open(a.sets))["rows"] if r["set"] in a.sets_keep.split(",")]
    for r in rows:
        if isinstance(r["slots"], str):
            r["slots"] = eval(r["slots"])  # noqa: S307 - our own plan files (as cand_test)
    text = {}
    for root in (a.glex_data, a.val_data):
        for m in map(json.loads, open(Path(root) / "manifest.jsonl")):
            text[(root, m["id"], m["lang"])] = m.get("text", "")
    d = ASR_DIRS["ml"]
    sd = torch.load(d / "model_weights.ckpt", map_location="cpu", weights_only=True)
    cfg = yaml.safe_load(open(d / "model_config.yaml"))
    dec = RNNTDecoder.from_config_dict(OmegaConf.create(cfg["decoder"]))
    jnt = RNNTJoint.from_config_dict(OmegaConf.create(cfg["joint"]))
    dec.load_state_dict({k[8:]: v for k, v in sd.items() if k.startswith("decoder.")})
    jnt.load_state_dict({k[6:]: v for k, v in sd.items() if k.startswith("joint.")})
    nd = cfg["model_defaults"]["enc_hidden"]
    pk = torch.nn.Sequential(torch.nn.Linear(nd + 128, 2 * nd), torch.nn.ReLU(), torch.nn.Linear(2 * nd, nd))
    pk.load_state_dict({k[len("prompt_kernel."):]: v for k, v in sd.items() if k.startswith("prompt_kernel.")})
    dec.eval(), jnt.eval(), pk.eval()
    vocab = list(jnt.vocabulary)
    blank = len(vocab)

    @torch.no_grad()
    def greedy(enc: torch.Tensor, lang_idx: int) -> str:
        oh = torch.zeros(enc.shape[0], 128)
        oh[:, lang_idx] = 1
        f = pk(torch.cat([enc, oh], -1))
        g, st = dec.predict(torch.tensor([[blank]]), None, add_sos=False, batch_size=1)
        hyp = []
        for t in range(f.shape[0]):
            for _ in range(10):
                k = int(jnt.joint(f[t:t + 1][None], g)[0, 0, 0].argmax())
                if k == blank:
                    break
                hyp.append(k)
                g, st = dec.predict(torch.tensor([[k]]), st, add_sos=False, batch_size=1)
        return detok([vocab[k] for k in hyp])

    encs = {"E0": SpeechEncoder("ml", att=[56, 0], device="cpu"), "E13": SpeechEncoder("ml", att=[56, 13], device="cpu")}
    s103 = SpeechEncoder("ml", att=[56, 0], device="cpu")
    st = torch.load(a.ln103, map_location="cpu", weights_only=False)
    prm = dict(s103.enc.named_parameters())
    with torch.no_grad():
        for k, v in st.items():
            prm[k].copy_(v.to(prm[k].dtype))
    encs["S103"] = s103
    arms = [("E0", "auto"), ("E0", "lang"), ("E13", "lang"), ("S103", "lang")]
    with open(jl, "a") as fh:
        fh.write(json.dumps({"meta": {"n": len(rows), "arms": arms, "cs": a.cs, "ln103": a.ln103,
                                      "asr": str(d), "threads": a.threads}}) + "\n")
    for n, r in enumerate(rows):
        root = a.glex_data if r["data"] == "glex" else a.val_data
        x = sf.read(Path(root) / r["path"], dtype="float32")[0]
        w = torch.from_numpy(session_wav(x, a.cs, 3.0))
        feats = {}
        for name, se in encs.items():
            f = se(w)["out"].float()
            feats[name] = f[max(0, a.cs - 2):]
        res = {}
        for name, mode in arms:
            hyp = greedy(feats[name], LANG_IDX["auto" if mode == "auto" else r["lang"]])
            res[f"{name}|{mode}"] = {"hyp": hyp, "cer": cer(text.get((root, r["id"], r["lang"]), ""), hyp),
                                     **score(r, hyp)}
        line = {"set": r["set"], "id": r["id"], "lang": r["lang"], "target": r["target"],
                "ref": text.get((root, r["id"], r["lang"]), ""), "res": res, "t": round(time.monotonic() - t0, 1)}
        with open(jl, "a") as fh:
            fh.write(json.dumps(line, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        print(n + 1, "/", len(rows), r["lang"], r["id"], round(time.monotonic() - t0, 1), "|",
              res["E0|lang"]["hyp"][:70], flush=True)
    print("done", round(time.monotonic() - t0, 1), flush=True)


if __name__ == "__main__":
    main()
