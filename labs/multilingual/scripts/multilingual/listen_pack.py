"""Corpus v2 (CPU, lab-box side): a stratified pack of ~20 EXISTING clips for a targeted human listening check
(owner 2026-11-13). Audio is COPIED inside the lab-box output folder only (never to the Mac); the index (text, metrics)
is the only part that may be committed.

Strata (deterministic, seed 20261113; only the 2550 new clips, so the transcript is the one of the presented render):
  A  FR grouped postcode, ACOUSTICALLY AMBIGUOUS (postcode_v2b exhaustive parse of the v2 2-1-2 reading > 1)   3 fail + 1 pass
  B  FR grouped postcode with a vigesimal group 70-99, not ambiguous                                          4 fail + 2 pass
  C  homophone names: Margaux FR 2, Margaux IT 1, Faure FR 2                                                  all as found
  D  controls: FR grouped clear pass 2, FR digits pass 1, FR digits fail 1, IT ambiguous pass 1
The verifier class (uncertain / asr_probable / tts_verified) is reported as the AUTOMATIC classification only. It is
not a verdict on the TTS: a spelling mismatch is never called a TTS error here. Human columns are left empty.
The postcode / name position is an estimate (character offset x clip duration), labelled as such.
Usage: listen_pack.py --manifest m.jsonl --verify-dir results/v4 --out DIR
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lexicon_v2 as L  # noqa: E402
import postcode_v2b as P  # noqa: E402

SEED = 20261113


def stratum_pools(man: list[dict]) -> dict:
    def pcs(r):
        return (r["slots"] or {}).get("postcode_form"), (r["slots"] or {}).get("postcode")

    def amb(r):
        f, pc = pcs(r)
        return f == "grouped" and len(P.decode(pc, r["lang"], "v2_212")) > 1

    def vig(r):
        f, pc = pcs(r)
        return f == "grouped" and r["lang"] == "fr" and (int(pc[:2]) >= 70 or int(pc[3:]) >= 70) and not amb(r)

    def name_has(r, w):
        return w in ((r["slots"] or {}).get("name") or "").split()

    ok = lambda r: bool(r["asr_ok"])                                         # noqa: E731
    fr = [r for r in man if r["lang"] == "fr"]
    return {
        "A_fr_ambiguous_fail": ([r for r in fr if amb(r) and not ok(r)], 3),
        "A_fr_ambiguous_pass": ([r for r in fr if amb(r) and ok(r)], 1),
        "B_fr_vigesimal_fail": ([r for r in fr if vig(r) and not ok(r)], 4),
        "B_fr_vigesimal_pass": ([r for r in fr if vig(r) and ok(r)], 2),
        "C_margaux_fr": ([r for r in fr if name_has(r, "Margaux")], 2),
        "C_margaux_it": ([r for r in man if r["lang"] == "it" and name_has(r, "Margaux")], 1),
        "C_faure_fr": ([r for r in fr if name_has(r, "Faure")], 2),
        "D_fr_grouped_clear_pass": ([r for r in fr if pcs(r)[0] == "grouped" and not amb(r) and not vig(r) and ok(r)], 2),
        "D_fr_digits_pass": ([r for r in fr if pcs(r)[0] == "digits" and ok(r)], 1),
        "D_fr_digits_fail": ([r for r in fr if pcs(r)[0] == "digits" and not ok(r)
                              and not any(name_has(r, w) for w in ("Margaux", "Faure"))], 1),
        "D_it_ambiguous_pass": ([r for r in man if r["lang"] == "it" and amb(r) and ok(r)], 1),
    }


def approx_span(text: str, needle: str, dur: float) -> str:
    i = text.lower().find(needle.lower())
    if i < 0:
        return "n/a"
    a, b = i / len(text) * dur, (i + len(needle)) / len(text) * dur
    return f"~{max(0, a - 0.5):.1f}-{min(dur, b + 0.5):.1f} s (estimate)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--verify-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists")
    base = Path(a.manifest).parent
    ver = {k: json.load(open(Path(a.verify_dir) / f"verify_full_r{k}.json"))["rows"] for k in range(4)}
    man = [r for r in map(json.loads, open(a.manifest)) if f"{r['id']}|{r['lang']}" in ver[0]]
    rng = random.Random(SEED)
    rows, n = [], 0
    out.mkdir(parents=True)
    for st, (pool, k) in stratum_pools(man).items():
        pool = sorted(pool, key=lambda r: (r["id"], r["lang"]))
        for r in (rng.sample(pool, k) if len(pool) > k else pool):
            n += 1
            key = f"{r['id']}|{r['lang']}"
            v = ver[r.get("render_round", 0)].get(key, {})
            s = r["slots"] or {}
            spoken = L.spoken_postcode(s["postcode"], r["lang"], s["postcode_form"] == "grouped") \
                if s.get("postcode") else None
            fn = f"{n:02d}_{st}_{r['id']}_{r['lang']}.wav"
            shutil.copyfile(os.path.realpath(base / r["path"]), out / fn)
            rows.append({"n": n, "stratum": st, "key": key, "file": fn, "dur_s": r["dur"],
                         "render_round": r.get("render_round", 0), "expected_text": r["text"],
                         "expected_name": s.get("name"), "expected_postcode": s.get("postcode"),
                         "postcode_form": s.get("postcode_form"), "postcode_as_written": spoken,
                         "v2_reading_parses": sorted(P.decode(s["postcode"], r["lang"], "v2_212"))
                         if s.get("postcode_form") == "grouped" else None,
                         "postcode_span": approx_span(r["text"], spoken, r["dur"]) if spoken else None,
                         "name_span": approx_span(r["text"], s["name"], r["dur"]) if s.get("name") else None,
                         "asr_hyp": v.get("hyp"), "verifier_failed_slots": v.get("bad"),
                         "verifier_class_automatic": v.get("cause"), "strict_asr_ok": bool(r["asr_ok"]),
                         "excluded": r.get("excluded"),
                         "HUMAN_heard_postcode": "", "HUMAN_heard_name": "",
                         "HUMAN_verdict(tts_wrong|asr_wrong|audio_ambiguous|ok)": "", "HUMAN_notes": ""})
    (out / "index.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False))
    cols = ["n", "stratum", "file", "dur_s", "expected_postcode", "postcode_as_written", "v2_reading_parses",
            "postcode_span", "expected_name", "name_span", "asr_hyp", "verifier_failed_slots",
            "verifier_class_automatic", "strict_asr_ok", "HUMAN_heard_postcode", "HUMAN_heard_name",
            "HUMAN_verdict(tts_wrong|asr_wrong|audio_ambiguous|ok)", "HUMAN_notes"]
    with open(out / "index.tsv", "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(str(r[c]) if r[c] is not None else "" for c in cols) + "\n")
    print(json.dumps({"clips": len(rows), "by_stratum": {r["stratum"]: sum(x["stratum"] == r["stratum"] for x in rows)
                                                         for r in rows}}, indent=1))


if __name__ == "__main__":
    main()
