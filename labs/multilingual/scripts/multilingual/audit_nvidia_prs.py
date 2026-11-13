"""CPU-only audit of the D-V3 / corpus-v2 path against NVIDIA NeMo Speech PRs #15558 / #16597 (issue #16142).

Read-only: loads only the tokenizer (no model weights) and the teacher.json files; writes one JSON report.
  1. effective token ids (tokenizer bos/eos, <SPECIAL_12> pad, <SPECIAL_20..22>, user '^'/'$') vs the constants the
     D-V3 trainer hard-codes (session_distill.PAD/BOS/EOS/SOTC/EOTC);
  2. the agent-channel EOS position in the teacher streams relative to the caller onset (EA places the agent EOS at
     user_bos + 8 frames, PR #15558 `eos_offset_frames=8`) and whether the teacher ever emits EOS elsewhere;
  3. how the loss weights map onto the label ids (text head: bos/eos/pad/text, function head), on a synthetic record.
Usage: audit_nvidia_prs.py --ckpt <snapshot dir> --teacher a.json [b.json ...] --out report.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import session_distill as S  # noqa: E402


def eos_offsets(items: dict, cs_of) -> dict:
    """Per teacher stream: frames from the caller onset (cs) to each agent EOS inside / after the caller turn."""
    first, other, n, no_eos = Counter(), Counter(), 0, 0
    for v in items.values():
        tx = v.get("main_text") or []
        cs = cs_of(v)
        if cs is None or not tx:
            continue
        n += 1
        eos = [t for t in range(cs, len(tx)) if tx[t] == S.EOS]
        if not eos:
            no_eos += 1
            continue
        first[eos[0] - cs] += 1
        for t in eos[1:]:
            other[t - cs] += 1
    return {"n_streams": n, "no_eos_after_onset": no_eos, "first_eos_offset_hist": dict(sorted(first.items())),
            "later_eos_offsets": dict(sorted(other.items())[:40]), "n_later_eos": sum(other.values())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--teacher", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if Path(a.out).exists():
        raise SystemExit(f"{a.out} exists")
    cfg = json.load(open(Path(a.ckpt) / "config.json"))
    stt = cfg["model"]["stt"]["model"]
    rep = {"config": {k: stt.get(k) for k in ("predict_user_text", "use_separate_asr_head", "use_numbers_norm",
                                               "use_nemo_text_norm", "nemo_text_norm_lang", "token_loss_weight",
                                               "function_token_loss_weight", "pad_token", "bos_token", "eos_token")}}
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["model"]["stt"]["model"].get("pretrained_tokenizer_name")
                                        or "nvidia/NVIDIA-Nemotron-Nano-9B-v2", local_files_only=True)
    ids = {"bos_token_id": tok.bos_token_id, "eos_token_id": tok.eos_token_id,
           "<s>": tok.convert_tokens_to_ids("<s>"), "</s>": tok.convert_tokens_to_ids("</s>")}
    for k in (12, 13, 14, 20, 21, 22):
        ids[f"<SPECIAL_{k}>"] = tok.convert_tokens_to_ids(f"<SPECIAL_{k}>")
    for c in ("^", "$"):
        ids[f"text_to_ids({c})"] = tok.encode(c, add_special_tokens=False)
    for t in ("<TOOLCALL>", "</TOOLCALL>"):
        ids[f"encode({t})"] = tok.encode(t, add_special_tokens=False)
    rep["token_ids"] = ids
    # the model OVERRIDES the HF specials (duplex_stt_model.py, Nemotron branch: bos '<s>', eos '</s>', pad
    # '<SPECIAL_12>'; config override_tokens); the raw HF eos_token_id is NOT the model's text_eos_id
    eff = {k: tok.convert_tokens_to_ids(stt[f"{k}_token"]) for k in ("bos", "eos", "pad")}
    rep["effective_ids_after_override"] = eff
    want = {"PAD": (S.PAD, eff["pad"]), "BOS": (S.BOS, eff["bos"]), "EOS": (S.EOS, eff["eos"]),
            "SOTC": (S.SOTC, ids["<SPECIAL_20>"]), "EOTC": (S.EOTC, ids["<SPECIAL_21>"])}
    rep["trainer_constants_match"] = {k: {"trainer": x, "tokenizer": y, "ok": x == y} for k, (x, y) in want.items()}
    rep["user_eos_distinct_from_text_eos"] = ids["text_to_ids($)"] != [eff["eos"]]
    # 2) EOS placement in the teacher streams
    rep["eos_placement"] = {}
    for p in a.teacher:
        d = json.load(open(p))
        oe = d.get("opener_end")
        def cs_of(v, oe=oe):
            return v.get("lead") if v.get("lead") is not None else (oe + 1 + 12 if oe is not None else None)
        rep["eos_placement"][p] = eos_offsets(d["items"], cs_of)
    # 3) loss weights on a synthetic record: every label id gets the intended class weight
    rec = {"main_text": [S.PAD] * 5 + [S.EOS] + [S.PAD] * 6 + [S.BOS, 1000, 1001, S.PAD],
           "main_func": [S.PAD] * 12 + [S.PAD] * 3 + [S.SOTC], "fc": [500, 501, S.EOTC], "topk": []}
    st = S.student_targets(rec, cs=2, en_end=10, x_end=10, n_audio=40, is_en=True)
    w = Counter()
    for lbl, tw in zip(st["text_lbl"], st["text_w"]):
        w[(int(lbl), float(tw))] += 1
    fw = Counter()
    for lbl, kd, f_w in zip(st["func_lbl"], st["kind"], st["func_w"]):
        fw[(int(lbl), kd, float(f_w))] += 1
    rep["synthetic_weights"] = {"text": {f"{k[0]}@{k[1]}": n for k, n in sorted(w.items())},
                                "func": {f"{k[0]}:{k[1]}@{k[2]}": n for k, n in sorted(fw.items())}}
    Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    print(json.dumps(rep, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
