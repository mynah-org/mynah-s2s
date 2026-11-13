"""Score one run_stream run over the fixed corpus (eval/corpus_v1.json), optionally against a golden run.

Three evidences, kept separate (AGENTS.md):
  model  = the agent's internal text stream + tool calls (summary.json);
  asr    = OmniASR CTC 3B v2 of the audio actually spoken (asr_omni3b.json, asr_dirs.py layout);
  ground = SQLite / tool ground truth (corpus expectations: function, arguments, status, date).
Per category:
  tool_orders*   : expected function called, order_id argument (normalized) exact, status + date in model text / ASR,
                   no grounding violation; correction: final call uses the corrected id; follow-up: tracking code said.
  tool_random    : expected function, min/max arguments exact.
  reasoning      : no tool call, answer regex in model text / ASR.
  smalltalk      : no tool call, non-empty answer.
Golden comparison (--golden): per item token identity (text/func), first divergent frame, decoded-text similarity,
ASR-text similarity, run-on/self-talk vs golden (continuation turns after the answer, self-asked
questions; 'new' vs 'pre_existing' in the golden). Writes <run>/score.json (refuses to overwrite) and prints a compact table.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
from pathlib import Path

NUM = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
       "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty"]
ORD = ["", "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth", "eleventh",
       "twelfth", "thirteenth", "fourteenth", "fifteenth", "sixteenth", "seventeenth", "eighteenth", "nineteenth",
       "twentieth"]
STATUS_RX = {"shipped": r"ship", "processing": r"process", "delivered": r"deliver", "cancelled": r"cancel"}
NOT_FOUND_RX = r"not (be )?f(ou|i)nd|couldn.?t find|can.?t find|cannot find|no (such )?order|unable to (find|locate)|" \
               r"doesn.?t (seem to )?exist|don.?t see|no record|not (in|on) (our|the) (system|records)|invalid"


def day_rx(day: int) -> str:
    alts = [rf"\b{day}(st|nd|rd|th)?\b"]
    if day <= 20:
        alts += [rf"\b{NUM[day]}\b", rf"\b{ORD[day]}\b"]
    else:
        tens, unit = divmod(day, 10)
        t = {2: "twenty", 3: "thirty"}[tens]
        alts += [rf"\b{t}[- ]?{NUM[unit]}\b" if unit else rf"\b{t}\b", rf"\b{t}[- ]?{ORD[unit]}\b" if unit else rf"\b{t[:-1]}ieth\b"]
    return "|".join(alts)


def date_ok(text: str, date: str) -> bool:
    if not date:
        return True
    month, day = date.split(",")[0].split()
    return bool(re.search(month[:3].lower(), text)) and bool(re.search(day_rx(int(day)), text))


def norm_id(s) -> str:
    return "".join(ch for ch in str(s).upper() if ch.isalnum())


def load_asr(run: Path) -> dict:
    p = run / "asr_omni3b.json"
    if not p.exists():
        return {}
    d = json.loads(p.read_text())
    return {s["scenario"]: " ".join(seg.get("text", "") for seg in s.get("segments", [])).lower() for s in d["sessions"]}


def decoded(summ) -> str:
    return re.sub(r"<[^>]+>", " ", summ.get("text", "")).lower()


def score_item(item, summ, asr_text):
    e, cat = item["expect"], item["category"]
    calls = summ.get("tool_calls", [])
    text = decoded(summ)
    r = {"id": item["id"], "category": cat, "n_calls": len(calls),
         "grounding_violations": len(summ.get("grounding_violations") or [])}
    evid = {"model": text, "asr": asr_text}
    if cat.startswith("tool_") and cat not in ("tool_forbidden", "tool_missing_args"):
        good = [c for c in calls if c["name"] == e["function"]]
        r["function_ok"] = bool(good)
        last = good[-1] if good else None
        args = (last or {}).get("args") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except ValueError:
                args = {}
        if cat == "tool_random":
            r["args_ok"] = bool(last) and str(args.get("min")) == str(e["min"]) and str(args.get("max")) == str(e["max"])
            r["args"] = args
        else:
            r["args_ok"] = bool(last) and norm_id(args.get("order_id", "")) == e["order_id"]
            r["args"] = args.get("order_id") if args else None
            for src, t in evid.items():
                if t is None:
                    continue
                if e["found"]:
                    r[f"status_{src}"] = bool(re.search(STATUS_RX[e["status"]], t))
                    r[f"date_{src}"] = date_ok(t, e.get("date") or "")
                else:
                    r[f"status_{src}"] = bool(re.search(NOT_FOUND_RX, t))
                if "tracking" in e:
                    digits = re.sub(r"\D", "", t)
                    words = " ".join(NUM[int(ch)] for ch in e["tracking"])
                    r[f"tracking_{src}"] = e["tracking"] in digits or words in re.sub(r"[^a-z ]", " ", t)
        r["ok_model"] = bool(r["function_ok"] and r["args_ok"] and r.get("status_model", True)
                             and r.get("date_model", True) and r.get("tracking_model", True)
                             and r["grounding_violations"] == 0)
        if e.get("forbid_function"):                         # change of intent: the abandoned tool must not run
            r["forbidden_called"] = any(c["name"] == e["forbid_function"] for c in calls)
            r["ok_model"] = r["ok_model"] and not r["forbidden_called"]
    elif cat in ("instruction", "tool_forbidden", "tool_missing_args"):
        # capability_v1: the answer turn = first agent turn after the greeting (turn 0)
        turns = agent_turns(summ)
        ans = turns[1].lower() if len(turns) > 1 else ""
        r["answer_words"] = len(re.findall(r"[a-z0-9']+", ans))
        checks = [r["n_calls"] == 0] if e.get("no_tool_call") else []
        if "answer_regex" in e:
            r["answer_model"] = bool(re.search(e["answer_regex"], ans))
            checks.append(r["answer_model"])
            if asr_text is not None:
                r["answer_asr"] = bool(re.search(e["answer_regex"], asr_text.lower()))
        if "max_words" in e:
            r["format_ok"] = 0 < r["answer_words"] <= e["max_words"]
            checks.append(r["format_ok"])
        if "ask_regex" in e:                                 # missing argument: must ask for it, not invent it
            r["asked_model"] = bool(re.search(e["ask_regex"], text))
            checks.append(r["asked_model"])
        if cat == "tool_forbidden" and "answer_regex" not in e:
            checks.append(r["answer_words"] >= 3)
        r["ok_model"] = all(checks)
    elif cat == "reasoning":
        for src, t in evid.items():
            if t is not None:
                r[f"answer_{src}"] = bool(re.search(e["answer_regex"], t))
        r["ok_model"] = r["n_calls"] == 0 and r["answer_model"]
    else:
        r["answered"] = len(text.split()) >= 3
        r["ok_model"] = r["n_calls"] == 0 and r["answered"]
    if e.get("must_not_regex"):                       # (agentbench must_not idea) invented / forbidden content
        r["must_not_hit"] = bool(re.search(e["must_not_regex"], text))
        r["ok_model"] = bool(r["ok_model"]) and not r["must_not_hit"]
    if asr_text is not None:
        keys = [k for k in r if k.endswith("_asr")]
        r["ok_asr"] = all(r[k] for k in keys) if keys else len(asr_text.split()) >= 3
    return r


def agent_turns(summ) -> list[str]:
    """Agent turns from the internal text stream: segments opened by <s> (turn start), tags stripped, empty dropped.
    Turn 0 is the greeting the model says at session start."""
    out = []
    for seg in summ.get("text", "").split("<s>")[1:]:
        t = re.sub(r"<[^>]+>", " ", seg).strip()
        if t:
            out.append(t)
    return out


def turn_stats(summ) -> dict:
    """Run-on / self-talk indicators. answer = first turn after the greeting; everything after it is a continuation.
    question_turns = continuation turns ending with '?' (the agent asking itself — the q05 self-dialogue shape)."""
    t = agent_turns(summ)
    cont = t[2:]
    return {"agent_turns": len(t), "continuation_turns": len(cont),
            "continuation_words": sum(len(x.split()) for x in cont),
            "question_turns": sum(x.rstrip().endswith("?") for x in cont)}


RUNON_WORDS = 15        # continuation growth (words) that counts as a run-on, absent new question turns


def runon_vs_golden(r: dict, g: dict) -> str | None:
    """'new' = the run continues after its answer clearly more than the golden does (more turns AND new self-asked
    questions or >= RUNON_WORDS more continuation words); 'pre_existing' = the golden already continues/self-talks
    and the run does too; None otherwise. Golden-relative on purpose: a tail monologue the golden already has is a
    model behaviour, not a regression of the run."""
    more_turns = r["continuation_turns"] > g["continuation_turns"]
    new_q = r["question_turns"] > g["question_turns"]
    more_words = r["continuation_words"] - g["continuation_words"] >= RUNON_WORDS
    if more_turns and (new_q or more_words):
        return "new"
    if g["continuation_turns"] > 0 and r["continuation_turns"] > 0:
        return "pre_existing"
    return None


def first_div(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--corpus-dir", required=True, help="rendered corpus (sidecar JSONs)")
    ap.add_argument("--golden", default=None, help="golden run dir to compare tokens/text/ASR against")
    ap.add_argument("--tag", default="score")
    ap.add_argument("--reference-score", default=None,
                    help="score.json of the NVIDIA-faithful reference run on the same corpus: each item is then classed "
                         "regression (reference OK, run fails) / pre_existing (both fail: reference/model limitation) / "
                         "fixed (reference fails, run OK) / ok")
    a = ap.parse_args()
    run, cdir = Path(a.run), Path(a.corpus_dir)
    outp = run / f"{a.tag}.json"
    if outp.exists():
        raise SystemExit(f"refusing to overwrite {outp}")
    asr = load_asr(run)
    gasr = load_asr(Path(a.golden)) if a.golden else {}
    rows, missing = [], []
    for side in sorted(cdir.glob("*.json")):
        if side.name in ("orders.json", "manifest.json"):
            continue
        item = json.loads(side.read_text())
        sp = run / item["id"] / "summary.json"
        if not sp.exists():
            missing.append(item["id"])         # subset runs (mini gate) score only what they ran
            continue
        summ = json.loads(sp.read_text())
        r = score_item(item, summ, asr.get(item["id"]) if asr else None)
        r["step_ms_p50"], r["step_ms_p95"], r["rtf"] = summ["step_ms_p50"], summ["step_ms_p95"], summ["rtf"]
        words = decoded(summ).split()
        tri = [tuple(words[i:i + 3]) for i in range(len(words) - 2)]
        r["words"] = len(words)
        r["repeated_trigrams"] = len(tri) - len(set(tri))                 # repetition / loop detector
        ft = summ.get("final_tokens") or summ.get("token_ids") or {}
        r["eos_frames"] = [i for i, t in enumerate(ft.get("text") or []) if t == 2]
        r["audio_out_s"] = summ.get("audio_out_s")
        r.update(turn_stats(summ))
        if a.golden:
            gp = Path(a.golden) / item["id"] / "summary.json"
            if gp.exists():
                g = json.loads(gp.read_text())
                ft = summ.get("final_tokens") or summ.get("token_ids") or {}
                gt = g.get("final_tokens") or g.get("token_ids") or {}
                r["text_identical"] = ft.get("text") == gt.get("text")
                r["func_identical"] = ft.get("func") == gt.get("func")
                r["first_div_frame"] = first_div(ft.get("text") or [], gt.get("text") or [])
                r["text_sim"] = round(difflib.SequenceMatcher(None, decoded(summ), decoded(g)).ratio(), 3)
                r["n_tokens"] = [len(ft.get("text") or []), len(gt.get("text") or [])]
                r["eos_identical"] = r["eos_frames"] == [i for i, t in enumerate(gt.get("text") or []) if t == 2]
                r["audio_out_s_golden"] = g.get("audio_out_s")
                gs = turn_stats(g)
                r["golden_turns"] = {k: gs[k] for k in ("agent_turns", "continuation_turns", "question_turns",
                                                        "continuation_words")}
                r["runon_vs_golden"] = runon_vs_golden(r, gs)
                r["tool_args_identical"] = [(c["name"], c["args"]) for c in summ.get("tool_calls", [])] == \
                    [(c["name"], c["args"]) for c in g.get("tool_calls", [])]
                if asr and gasr:
                    r["asr_sim"] = round(difflib.SequenceMatcher(None, asr.get(item["id"], ""),
                                                                 gasr.get(item["id"], "")).ratio(), 3)
        rows.append(r)
    if a.reference_score:
        ref = {r["id"]: r for r in json.loads(Path(a.reference_score).read_text())["items"]}
        for r in rows:
            q = ref.get(r["id"])
            if q is None:
                continue
            ro, co = bool(q.get("ok_model")), bool(r.get("ok_model"))
            r["vs_reference"] = {(True, True): "ok", (True, False): "regression", (False, False): "pre_existing",
                                 (False, True): "fixed"}[(ro, co)]
    cats = sorted({r["category"] for r in rows})
    agg = {}
    for c in cats + ["ALL"]:
        rs = [r for r in rows if c == "ALL" or r["category"] == c]
        d = {"n": len(rs), "ok_model": sum(bool(r.get("ok_model")) for r in rs)}
        if any("ok_asr" in r for r in rs):
            d["ok_asr"] = sum(bool(r.get("ok_asr")) for r in rs)
        if any("args_ok" in r for r in rs):
            d["args_ok"] = sum(bool(r.get("args_ok")) for r in rs)
        if a.golden:
            d["text_identical"] = sum(bool(r.get("text_identical")) for r in rs)
            sims = [r["text_sim"] for r in rs if "text_sim" in r]
            d["text_sim_mean"] = round(sum(sims) / len(sims), 3) if sims else None
            d["eos_identical"] = sum(bool(r.get("eos_identical")) for r in rs)
            d["tool_args_identical"] = sum(bool(r.get("tool_args_identical")) for r in rs)
            fd = [r["first_div_frame"] for r in rs if r.get("first_div_frame") is not None]
            d["first_div_frame_median"] = sorted(fd)[len(fd) // 2] if fd else None
            d["runon_new"] = sum(r.get("runon_vs_golden") == "new" for r in rs)
            d["runon_pre_existing"] = sum(r.get("runon_vs_golden") == "pre_existing" for r in rs)
            d["extra_turns"] = sum(r["agent_turns"] - r["golden_turns"]["agent_turns"] for r in rs if "golden_turns" in r)
        d["repeated_trigrams"] = sum(r.get("repeated_trigrams", 0) for r in rs)
        d["question_turns"] = sum(r.get("question_turns", 0) for r in rs)
        d["continuation_turns"] = sum(r.get("continuation_turns", 0) for r in rs)
        d["audio_out_s"] = round(sum(r.get("audio_out_s") or 0 for r in rs), 1)
        if any("vs_reference" in r for r in rs):
            for k in ("regression", "pre_existing", "fixed", "ok"):
                d["ref_" + k] = sum(r.get("vs_reference") == k for r in rs)
        p50 = [r["step_ms_p50"] for r in rs if "step_ms_p50" in r]
        d["step_ms_p50_median"] = sorted(p50)[len(p50) // 2] if p50 else None
        agg[c] = d
    rep = {"run": str(run), "golden": a.golden, "aggregate": agg, "items": rows, "not_run": missing}
    outp.write_text(json.dumps(rep, indent=1))
    for c, d in agg.items():
        print(f"{c:24s} {json.dumps(d)}")
    for r in rows:
        if not r.get("ok_model") or (a.golden and not r.get("text_identical", True)) or r.get("runon_vs_golden") == "new":
            print("  ", json.dumps({k: v for k, v in r.items() if k not in ("category",)})[:400])


if __name__ == "__main__":
    main()
