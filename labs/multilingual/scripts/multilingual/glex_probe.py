"""G-LEX probe: batched, teacher-forced measure of lexical slot transfer on held-out IT/FR caller turns.

Owner (2026-11-12): G-LEX-dev is the repeatable tracking probe (cheap; may guide checkpoint choice); G-LEX-test is frozen
and only used for the final evaluation. Final selection stays on LIVE generated calls; this probe is not a live metric.

Per clip, the session is: lead-in of silence (A's opener tokens on the agent channel, as in D-V3 training) + the caller
turn + tail. Inputs are teacher-forced: agent channel = opener then PAD; function channel = PAD, SOTC at x_end + offset,
then the canonical rule call (session_corpus_v2). One forward per batch (no cache, no grad). Metrics:
  sotc_margin     function logP(SOTC) minus the best other token at the call frame (decision readiness);
  call_tok_acc    argmax accuracy over the call tokens given the gold prefix;
  exact_call      every call token correct (the canonical query / name / postcode reproduced exactly);
  wanted          the argmax token at every call position, decoded (what the model "wanted" to write);
  listen          max text logP(BOS) inside the caller turn and the number of BOS-argmax frames there.
Usage: glex_probe.py --plan glex_dev.jsonl --data <tts root> --teacher teacher.json --arm NAME=SPEC ... --out f.json
  SPEC: enc=vc | enc=ml[;ln=<encoder_ln.pt>][;adapter=<adapter.pt>|;map=<ridge.pt>]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from session_distill import BOS, EOTC, FRAME, PAD, SOTC, rule_call_text, session_wav, student_targets  # noqa: E402

OFFSET = 2


def synthetic_record(opener: list[int], cs: int, x_end: int, call_tokens: list[int], offset: int = OFFSET) -> dict:
    s = x_end + offset
    tx = (opener + [PAD] * (s + 1))[:s + 1]
    tx = [x if t < cs else PAD for t, x in enumerate(tx)]
    fn = [PAD] * s + [SOTC]
    return {"main_text": tx, "main_func": fn, "fc": list(call_tokens), "topk": []}


def call_metrics(fc_pred: list[int], fc_gold: list[int]) -> dict:
    ok = [p == g for p, g in zip(fc_pred, fc_gold)]
    return {"call_tok_acc": round(sum(ok) / max(1, len(ok)), 3), "exact_call": all(ok), "n_call_tok": len(ok),
            "first_error": next((k for k, v in enumerate(ok) if not v), None)}


def parse_spec(spec: str) -> dict:
    return dict(kv.split("=", 1) for kv in spec.split(";"))


def main():  # pragma: no cover - GPU
    import soundfile as sf
    import torch
    from adapters import ResidualAdapter, load_adapter
    from distill_adapter import Duplex, load_engine
    from encoders import SpeechEncoder
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--data", required=True, help="TTS root with manifest.jsonl for the probe clips")
    ap.add_argument("--teacher", required=True, help="a teacher.json (prompt, opener_end, opener tokens)")
    ap.add_argument("--arm", action="append", required=True)
    ap.add_argument("--langs", default="it,fr")
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--lead-extra", type=int, default=12)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if Path(a.out).exists():
        raise SystemExit(f"{a.out} exists")
    prep = json.loads(Path(a.teacher).read_text())
    any_item = next(iter(prep["items"].values()))
    opener = any_item["main_text"][:prep["opener_end"] + 1]
    cs = prep["opener_end"] + 1 + a.lead_extra
    man = {(r["id"], r["lang"]): r for r in map(json.loads, open(Path(a.data) / "manifest.jsonl"))}
    rows = [r for r in map(json.loads, open(a.plan)) if r["lang"] in a.langs.split(",") and (r["id"], r["lang"]) in man]
    eng = load_engine(Path(a.out).parent / "_glex_nemo")
    D = Duplex(eng, prep["prompt"])
    tok = D.w.tokenizer

    def enc(t):
        ids = list(tok.text_to_ids(t))
        return ids[1:] if ids and ids[0] == BOS else ids
    encs = {}
    rep = {"cs": cs, "offset": OFFSET, "n_clips": len(rows), "arms": {}}
    for spec in a.arm:
        name, sp = spec.split("=", 1)
        cfg = parse_spec(sp)
        if cfg["enc"] not in encs:
            encs[cfg["enc"]] = SpeechEncoder(cfg["enc"])
        se = encs[cfg["enc"]]
        saved = None
        if cfg.get("ln"):
            st = torch.load(cfg["ln"], map_location="cpu", weights_only=False)
            prm = dict(se.enc.named_parameters())
            saved = {k: prm[k].detach().clone() for k in st}
            with torch.no_grad():
                for k, v in st.items():
                    prm[k].copy_(v.to(prm[k].device, prm[k].dtype))
        if cfg.get("adapter"):
            ad = load_adapter(cfg["adapter"]).to(D.dev).float()
        elif cfg.get("map"):
            m = torch.load(cfg["map"], map_location="cpu", weights_only=True)
            ad = ResidualAdapter().to(D.dev)
            ad.init_affine(m["M"].to(D.dev), m["m"].to(D.dev))            # MLP branch zero-init: exactly the ridge map
        else:
            ad = None
        res = []
        for b in range(0, len(rows), a.batch):
            chunk = rows[b:b + a.batch]
            seqs = []
            for r in chunk:
                x = sf.read(Path(a.data) / man[(r["id"], r["lang"])]["path"], dtype="float32")[0]
                w = session_wav(x, cs, 3.0)
                f = se(torch.from_numpy(w))["out"].to(D.dev).float()
                x_end = cs + int(math.ceil(len(x) / FRAME))
                gold = enc(rule_call_text(r["target"])) + [EOTC]
                rec = synthetic_record(opener, cs, x_end, gold)
                stt = student_targets(rec, cs, x_end, x_end, f.shape[0], False)
                sil = f[2]                                                  # encoded digital silence (lead frame)
                src = torch.stack([f[p[1]] if p[0] == "a" else sil for p in stt["pos"]])
                seqs.append((r, stt, src, x_end, gold))
            with torch.no_grad():
                X = torch.nn.utils.rnn.pad_sequence([s[2] for s in seqs], batch_first=True)
                Y = ad(X) if ad is not None else X
                audio = D.proj(Y.to(D.proj.weight.dtype))
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
                P = D.prompt_emb.shape[1]
                lt = torch.log_softmax(o["text_logits"][:, P:].float(), -1)
                lf = torch.log_softmax(o["function_logits"][:, P:].float(), -1)
            for k, (r, stt, _, x_end, gold) in enumerate(seqs):
                kinds = stt["kind"]
                si = stt["func_lbl"].index(SOTC)
                fc_idx = [j for j, kd in enumerate(kinds) if kd == "fc"]
                pred = [int(lf[k, j].argmax()) for j in fc_idx]
                lfs = lf[k, si]
                margin = float(lfs[SOTC] - torch.cat([lfs[:SOTC], lfs[SOTC + 1:]]).max())
                turn = [j for j, p in enumerate(stt["pos"]) if p[0] == "a" and cs + 2 <= p[1] < x_end - 4]
                res.append({"id": r["id"], "lang": r["lang"], "type": r["slots"].get("probe_type"),
                            "gold": rule_call_text(r["target"]), "wanted": tok.ids_to_text([t for t in pred if t != EOTC]),
                            "sotc_margin": round(margin, 3), **call_metrics(pred, gold),
                            "listen_max_bos": round(max(float(lt[k, j, BOS]) for j in turn), 3) if turn else None,
                            "listen_bos_argmax": sum(int(lt[k, j].argmax()) == BOS for j in turn)})
        if saved:                                                           # restore the encoder for the next arm
            prm = dict(se.enc.named_parameters())
            with torch.no_grad():
                for k_, v in saved.items():
                    prm[k_].copy_(v)
        summ = {}
        for key in ("all", "it", "fr", "new_concept", "new_combo", "new_name_digits"):
            sel = [x for x in res if key == "all" or x["lang"] == key or x["type"] == key]
            if sel:
                summ[key] = {"n": len(sel), "exact": round(float(np.mean([x["exact_call"] for x in sel])), 3),
                             "tok_acc": round(float(np.mean([x["call_tok_acc"] for x in sel])), 3),
                             "sotc_argmax": round(float(np.mean([x["sotc_margin"] > 0 for x in sel])), 3),
                             "bos_in_turn_clips": sum(x["listen_bos_argmax"] > 0 for x in sel)}
        rep["arms"][name] = {"spec": sp, "summary": summ, "rows": res}
        print(name, json.dumps(summ), flush=True)
        for x in res[:6]:
            print("   ", x["lang"], x["type"], "| wanted:", x["wanted"][:90], "| gold:", x["gold"][-60:], flush=True)
        Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
