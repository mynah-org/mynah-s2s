"""D1 pilot, step 3 (GPU, ml env): frozen-model states at the bridge taps over slot regions, the VC EN reference, and
the Thinker's use of the slot (teacher-forced call). Read-only: no gradient, no training (owner GO 2026-11-13).

Session layout and encoder as in glex_probe.py (lead-in of cs frames, clip, 3 s tail; 80 ms frames; clip starts at
frame cs). Slot region frames = cs + floor(t0 / 0.08) .. cs + ceil(t1 / 0.08) from the forced alignment (≥ 1 frame).
Taps per arm (frame level, then pooled):
  T1 = ML encoder output (arm's encoder LNs)      T2 = adapter.affine(T1)      T3 = adapter(T1) = T2 + MLP branch
  T4 = perception.proj(T3)  (the Thinker's audio input)          REF = perception.proj(VC encoder) on EN clips
Pooled per region: mean over the region and the centre frame; per clip: the whole-utterance mean (cs .. x_end).
U (use): for up to --u-per-lang call clips per language, the teacher-forced call forward of glex_probe; per call token
the gold id, argmax, gold log-prob and margin (gold minus best other); tokens are tagged with the slot whose characters
they cover in the canonical call text (full_name / postcode / query).
Stages are written as they finish (npz per tap set, json for U), so a timeout keeps what is done.
Usage: d1_extract.py --sample s.json --align a.json --data <tts root> --teacher t.json --arm NAME=SPEC ... --out DIR
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from glex_probe import parse_spec, synthetic_record  # noqa: E402
from session_distill import BOS, EOTC, FRAME, PAD, SOTC, rule_call_text, session_wav, student_targets  # noqa: E402

FS = FRAME / 16000                                  # 0.08 s


def region_frames(t0: float, t1: float, cs: int) -> list[int]:
    a = cs + int(math.floor(t0 / FS))
    b = max(a + 1, cs + int(math.ceil(t1 / FS)))
    return list(range(a, b))


def slot_char_spans(call: str, target: dict) -> dict[str, tuple[int, int]]:
    out = {}
    for k, v in (target.get("arguments") or {}).items():
        s = json.dumps(v, ensure_ascii=False)[1:-1] if isinstance(v, str) else str(v)
        i = call.find(s)
        if i >= 0:
            out[k] = (i, i + len(s))
    return out


def main():  # pragma: no cover - GPU
    import soundfile as sf
    import torch
    from adapters import load_adapter
    from distill_adapter import Duplex, load_engine
    from encoders import SpeechEncoder
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True)
    ap.add_argument("--align", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--arm", action="append", required=True)
    ap.add_argument("--u-per-lang", type=int, default=20)
    ap.add_argument("--lead-extra", type=int, default=12)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    clips = json.load(open(a.sample))["clips"]
    ali = json.load(open(a.align))["alignments"]
    prep = json.loads(Path(a.teacher).read_text())
    opener = next(iter(prep["items"].values()))["main_text"][:prep["opener_end"] + 1]
    cs = prep["opener_end"] + 1 + a.lead_extra
    eng = load_engine(out / "_nemo")
    D = Duplex(eng, prep["prompt"])
    tok = D.w.tokenizer
    print("engine", round(time.monotonic() - t0, 1), flush=True)

    def enc(t):
        ids = list(tok.text_to_ids(t))
        return ids[1:] if ids and ids[0] == BOS else ids

    # regions (shared by every arm)
    meta, usable = [], []
    for ci, c in enumerate(clips):
        al = ali.get(f"{c['id']}|{c['lang']}", {})
        if "words" not in al:
            continue
        usable.append(ci)
        for s in c["spans"]:
            ws = al["words"][s["w0"]:s["w1"]]
            if any(w is None for w in ws):
                continue
            t_0, t_1 = ws[0][0], ws[-1][1]
            meta.append({"clip": ci, "id": c["id"], "lang": c["lang"], "voice": c.get("voice"), "task": s["task"],
                         "label": s["label"], "pos": s.get("pos"), "category": s.get("category"),
                         "t0": t_0, "t1": t_1, "frames": region_frames(t_0, t_1, cs)})
    (out / "regions.json").write_text(json.dumps({"cs": cs, "clips": usable, "regions": meta}, ensure_ascii=False))
    wav = {ci: sf.read(Path(a.data) / clips[ci]["path"], dtype="float32")[0] for ci in usable}
    encs = {}

    def taps(se, ad, ref=False):
        reg = {k: [] for k in (["REF"] if ref else ["T1", "T2", "T3", "T4"])}
        regc = {k: [] for k in reg}
        utt = {k: [] for k in reg}
        by_clip = {}
        for r in meta:
            by_clip.setdefault(r["clip"], []).append(r)
        order = [ci for ci in usable if (not ref or clips[ci]["lang"] == "en")]
        for ci in order:
            x = wav[ci]
            with torch.no_grad():
                f = se(torch.from_numpy(session_wav(x, cs, 3.0)))["out"].to(D.dev).float()
                x_end = cs + int(math.ceil(len(x) / FRAME))
                if ref:
                    st = {"REF": D.proj(f.to(D.proj.weight.dtype)).float()}
                else:
                    t2 = ad.affine(f)
                    t3 = ad(f)
                    st = {"T1": f, "T2": t2, "T3": t3, "T4": D.proj(t3.to(D.proj.weight.dtype)).float()}
            for k, m in st.items():
                m = m.cpu().numpy()
                utt[k].append(m[cs:x_end].mean(0))
                for r in by_clip.get(ci, []):
                    fr = [t for t in r["frames"] if t < m.shape[0]]
                    reg[k].append(m[fr].mean(0))
                    regc[k].append(m[fr[len(fr) // 2]])
        arrays = {}
        for k in reg:
            arrays[f"{k}_mean"] = np.stack(reg[k]).astype(np.float16)
            arrays[f"{k}_centre"] = np.stack(regc[k]).astype(np.float16)
            arrays[f"{k}_utt"] = np.stack(utt[k]).astype(np.float16)
        idx = [i for i, r in enumerate(meta) if not ref or clips[r["clip"]]["lang"] == "en"]
        return arrays, idx, order

    def use(se, ad, name):
        per = {}
        sel = []
        for ci in usable:
            c = clips[ci]
            if c["target"] and per.get(c["lang"], 0) < a.u_per_lang:
                per[c["lang"]] = per.get(c["lang"], 0) + 1
                sel.append(ci)
        res = []
        for b in range(0, len(sel), 6):
            seqs = []
            for ci in sel[b:b + 6]:
                c, x = clips[ci], wav[ci]
                with torch.no_grad():
                    f = se(torch.from_numpy(session_wav(x, cs, 3.0)))["out"].to(D.dev).float()
                x_end = cs + int(math.ceil(len(x) / FRAME))
                gold = enc(rule_call_text(c["target"])) + [EOTC]
                rec = synthetic_record(opener, cs, x_end, gold)
                stt = student_targets(rec, cs, x_end, x_end, f.shape[0], False)
                sil = f[2]
                src = torch.stack([f[p[1]] if p[0] == "a" else sil for p in stt["pos"]])
                seqs.append((ci, stt, src, gold))
            with torch.no_grad():
                X = torch.nn.utils.rnn.pad_sequence([s[2] for s in seqs], batch_first=True)
                audio = D.proj(ad(X).to(D.proj.weight.dtype))
                B, Tm = X.shape[0], X.shape[1]
                AI = torch.full((B, Tm), PAD, dtype=torch.long, device=D.dev)
                FI = AI.clone()
                for k, s in enumerate(seqs):
                    n = len(s[1]["pos"])
                    AI[k, :n] = torch.tensor(s[1]["agent_in"], device=D.dev)
                    FI[k, :n] = torch.tensor(s[1]["func_in"], device=D.dev)
                e = D.stt.embed_tokens
                fr = D.fusion(agent_text_embeds=e(AI).to(D.dtype), user_audio_embeds=audio.to(D.dtype),
                              user_text_embeds=None, function_embeds=e(FI).to(D.dtype))
                o = D.stt(torch.cat([D.prompt_emb.expand(B, -1, -1), fr], 1), cache=None)
                lf = torch.log_softmax(o["function_logits"][:, D.prompt_emb.shape[1]:].float(), -1)
            for k, (ci, stt, _, gold) in enumerate(seqs):
                c = clips[ci]
                call = rule_call_text(c["target"])
                spans = slot_char_spans(call, c["target"])
                fc_idx = [j for j, kd in enumerate(stt["kind"]) if kd == "fc"]
                toks = []
                for q, (j, g) in enumerate(zip(fc_idx, gold)):
                    if g == EOTC:
                        continue
                    lo = len(tok.ids_to_text(gold[:q]))
                    hi = len(tok.ids_to_text(gold[:q + 1]))
                    slot = next((n for n, (s0, s1) in spans.items() if lo < s1 and hi > s0), None)
                    row = lf[k, j]
                    gl = float(row[g])
                    other = float(torch.cat([row[:g], row[g + 1:]]).max())
                    toks.append({"slot": slot, "gold": int(g), "argmax": int(row.argmax()), "gold_lp": round(gl, 3),
                                 "margin": round(gl - other, 3)})
                res.append({"clip": ci, "id": c["id"], "lang": c["lang"], "kind": c["kind"], "call": call,
                            "tokens": toks})
        (out / f"use_{name}.json").write_text(json.dumps(res, ensure_ascii=False))
        print("use", name, len(res), round(time.monotonic() - t0, 1), flush=True)

    ref_done = False
    for spec in a.arm:
        name, sp = spec.split("=", 1)
        cfg = parse_spec(sp)
        if "ml" not in encs:
            encs["ml"] = SpeechEncoder("ml")
        se = encs["ml"]
        st = torch.load(cfg["ln"], map_location="cpu", weights_only=False)
        prm = dict(se.enc.named_parameters())
        saved = {k: prm[k].detach().clone() for k in st}
        with torch.no_grad():
            for k, v in st.items():
                prm[k].copy_(v.to(prm[k].device, prm[k].dtype))
        ad = load_adapter(cfg["adapter"]).to(D.dev).float()
        arr, idx, order = taps(se, ad)
        np.savez_compressed(out / f"taps_{name}.npz", region_index=np.array(idx), utt_clips=np.array(order), **arr)
        print("taps", name, {k: v.shape for k, v in arr.items()}, round(time.monotonic() - t0, 1), flush=True)
        if not ref_done:
            vc = SpeechEncoder("vc")
            arr, idx, order = taps(vc, None, ref=True)
            np.savez_compressed(out / "taps_REF.npz", region_index=np.array(idx), utt_clips=np.array(order), **arr)
            print("ref", {k: v.shape for k, v in arr.items()}, round(time.monotonic() - t0, 1), flush=True)
            del vc
            ref_done = True
        use(se, ad, name)
        with torch.no_grad():
            for k, v in saved.items():
                prm[k].copy_(v)
    print("done", round(time.monotonic() - t0, 1), flush=True)


if __name__ == "__main__":
    main()
