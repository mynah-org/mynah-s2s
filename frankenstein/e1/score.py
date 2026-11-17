"""E1 scoring (pure Python, CPU): TECHNICAL / DUPLEX / LINGUISTIC evidence per arm and language from a run directory.

  python e1/score.py RUN_DIR ASR_DIR > RUN_DIR/score.json

Every number comes from the run's JSONL files (frames.jsonl, brain.jsonl, trace.jsonl = V2 engine trace,
fr_trace.jsonl = Frankenstein trace, summary.json). Gate verdicts are NOT computed here: this prints the evidence; the
report states PASS / FAIL / INCOMPLETE against it. Language ID is a crude function-word count (it / fr / en / es), not
a model; content checks are substring checks against the scenario gold (tool, slots, must-say), both are evidence for
a human read, not a score.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from e1.script import GOLD  # noqa: E402

FW = {
    "it": "il lo la le gli un una di che non per con sono è ho ha mi può posso suo sua mio grazie vorrei cosa come "
          "quando nostro nostra ordine abbiamo della del alla al".split(),
    "fr": "le la les un une de des du que je vous nous est pas pour avec votre mon ma merci bonjour commande "
          "quel quelle au aux et".split(),
    "en": "the a an of to and is are you your i we my can our for with have this that what order thank".split(),
    "es": "el la los las un una de que no por con es su sus mi gracias pedido usted".split(),
}


def lang_id(text: str) -> tuple:
    w = re.findall(r"[a-zàâäéèêëîïôöùûüç']+", text.lower())
    sc = {L: sum(1 for x in w if x in s) for L, s in FW.items()}
    best = max(sc, key=sc.get) if w else None
    return best if best and sc[best] > 0 else None, sc


def rows(p: Path) -> list:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def v2_emitted(trace: list) -> dict:
    out = {}
    for e in trace:
        if e.get("ev") == "thinker.step" and e.get("text_ids"):
            end, n = e["model_step"], len(e["text_ids"])
            for i, tid in enumerate(e["text_ids"]):
                out[end - n + i] = tid
    return out


def turn_of(k: int, tl: list) -> str | None:
    lab = None
    for t in tl:
        if k >= t["start_k"]:
            lab = t["label"]
    return lab


def content(label: str, reply: str, calls: list) -> dict:
    g = GOLD.get(label, {})
    r = reply.lower()
    res = {}
    if g.get("tool"):
        hit = [c for c in calls if (c.get("name") == g["tool"])]
        res["tool_called"] = bool(hit)
        if g["tool"] == "find_customer" and hit:
            a = json.dumps(hit[0].get("args") or {}).lower()
            res["name_ok"] = g["name"] in a
            res["postcode_ok"] = g["postcode"] in re.sub(r"\D", "", a)
        if g["tool"] == "search_catalog" and hit:
            q = json.dumps(hit[0].get("args") or {}).lower()
            res["query_ok"] = any(x in q for x in g["query_any"])
        for i, alts in enumerate(g.get("must_say_any", [])):
            res[f"must_say_{i}"] = any(x in r for x in alts)
    else:
        res["no_tool"] = not calls
    if g.get("ask_any"):
        res["asks_identity"] = any(x in r for x in g["ask_any"])
    return res


def score_session(d: Path, asr: dict) -> dict:
    s = json.loads((d / "summary.json").read_text())
    fr = rows(d / "frames.jsonl")
    br = rows(d / "brain.jsonl")
    v2 = rows(d / "trace.jsonl")
    ft = rows(d / "fr_trace.jsonl")
    tl = asr["timeline"]
    arm = s["arm"]
    out = {"arm": arm, "lang": s["lang"], "wall_s": s.get("wall_s")}
    # ---------------- TECHNICAL
    if arm in ("m1f", "m0"):
        em = v2_emitted(v2)
        hook_vs_v2 = [r["k"] for r in fr if r["k"] in em and em[r["k"]] != r["emitted"]]
        forced = [r for r in fr if r.get("written") is not None]
        overwritten = [{"k": r["k"], "written": r["written"], "emitted": r["emitted"]} for r in forced
                       if r["emitted"] != r["written"]]
        out["technical"] = {
            "frames_hooked": len(fr), "frames_in_v2_trace": len(em), "hook_vs_v2_emitted_mismatch": hook_vs_v2[:20],
            "n_forced_frames": len(forced), "forced_ids_emitted_exactly": len(forced) - len(overwritten),
            "forced_overwritten_by_wrapper": overwritten[:20], "n_gaps": s.get("n_gaps"),
            "cache_checks": [(c["k"], c["same"], c["distinct_object"]) for c in s.get("cache_checks", [])],
            "fc_suppressed": s.get("fc_suppressed", [])}
    # causality: every brain request used only words closed at <= its frame
    words = asr["words"]
    caus = []
    for b in [x for x in br if x.get("ev") == "brain.run"]:
        allowed = " ".join(w["word"] for w in words if w["k"] <= b["k"])
        caus.append(all(tok in allowed.split() for tok in (b.get("user") or "").split()))
    out.setdefault("technical", {})["brain_requests_causal"] = caus
    # reply availability vs first forced frame
    if arm == "m1f":
        firsts = []
        for b in [x for x in br if x.get("ev") == "brain.run"]:
            nxt = [r["k"] for r in fr if r["k"] > b["k"] and r.get("written") is not None and r["written"] >= 1000]
            firsts.append({"request_k": b["k"], "first_avail_k": b.get("first_avail_k"),
                           "first_forced_k": nxt[0] if nxt else None})
        out["technical"]["reply_not_before_available"] = all(
            f["first_forced_k"] is None or f["first_avail_k"] is None or f["first_forced_k"] >= f["first_avail_k"]
            for f in firsts)
        out["technical"]["first_forced"] = firsts
    # ---------------- DUPLEX: agent turns from the emitted stream
    emitted = [(r["k"], r["emitted"]) for r in fr]
    turns, cur = [], None
    for k, tid in emitted:
        if tid == 1:
            cur = {"bos_k": k, "ids": [], "eos_k": None}
            turns.append(cur)
        elif tid == 2 and cur is not None and cur["eos_k"] is None:
            cur["eos_k"] = k
            cur = None
        elif cur is not None and tid is not None and tid >= 1000:
            cur["ids"].append(tid)
    nb = set(asr["nonblank"])
    wk = sorted(w["k"] for w in asr["words"])
    ek = sorted(asr["eou"])

    def premature(k):
        if any((k - j) in nb for j in range(0, 5)):
            return "asr_speech_recent"
        lw = max([x for x in wk if x <= k], default=-1)
        le = max([x for x in ek if x <= k], default=-1)
        return "no_eou_after_last_word" if lw > le else None
    for t in turns:
        t["premature"] = premature(t["bos_k"])
    # model BOS predictions (wanted) while listening, attributed the same way (from the per-frame log)
    wanted_bos = [r["k"] for r in fr if r.get("predicted") == 1]
    for t in turns:
        t["after_user"] = turn_of(t["bos_k"], tl)
        t["n_text"] = len(t["ids"])
        del t["ids"]
    src = [e.get("source") for e in ft if e["ev"] == "force.turn_open"]
    closes = [(e.get("model_step"), e.get("why")) for e in ft if e["ev"] == "force.turn_close"]
    barge = [(e.get("model_step"), e.get("source", "adapter"), e.get("what")) for e in ft if e["ev"] == "adapter.barge_in"]
    u4 = next(t for t in tl if t["label"] == "U4")
    speaking_at_u4 = None
    for t in turns:
        if t["bos_k"] <= u4["start_k"] and (t["eos_k"] is None or t["eos_k"] > u4["start_k"]):
            speaking_at_u4 = t
    out["duplex"] = {"n_agent_turns": len(turns), "n_premature_emitted_bos": sum(1 for t in turns if t["premature"]),
                     "model_wanted_bos": [(k, premature(k)) for k in wanted_bos],
                     "bos_suppressed": s.get("bos_suppressed"), "brain_skipped": s.get("brain_skipped"),
                     "agent_turns": turns, "turn_open_sources": src, "turn_closes": closes, "barge_in_events": barge,
                     "u4_start_k": u4["start_k"], "agent_turn_open_at_u4": speaking_at_u4,
                     "adapter_cancels": [c for c in s["adapter"]["commands"] if c["type"] == "Cancel"],
                     "gate": s["adapter"]["gate"],
                     "forcer": {k: s.get("forcer", {}).get(k) for k in ("overrides", "agreements")}}
    # ---------------- LINGUISTIC: replies per user turn
    replies = []
    if arm in ("m1f", "cas"):
        for b in [x for x in br if x.get("ev") == "brain.run"]:
            label = turn_of(b["k"], tl) if b.get("user") else "opener"
            reply = (b.get("resume") or {}).get("text") or b.get("reply") or ""
            calls = [b["call"]["parsed"]] if b.get("call") and b["call"].get("parsed") else []
            calls = [{"name": c.get("name"), "args": c.get("arguments")} for c in calls]
            spoken = None
            replies.append({"k": b["k"], "label": label, "user": b.get("user"), "reply": reply,
                            "decision": b.get("decision"), "call_parsed": (b.get("call") or {}).get("parsed"),
                            "resume_text": (b.get("resume") or {}).get("text"),
                            "lang_id": lang_id(reply)[0], "tool_margin": b.get("tool_margin"),
                            "call_text": (b.get("call") or {}).get("text"), "tool_result": b.get("tool_result"),
                            "content": content(label, reply, calls) if label in GOLD else None,
                            "wall_ms": b.get("wall_ms"), "prefill_ms": b.get("prefill_ms"),
                            "decode_ms_per_tok": b.get("decode_ms_per_tok"), "spoken": spoken})
        hist = [x for x in br if x.get("ev") == "history.agent"]
        out["history_spoken"] = [(h["k"], h.get("why"), h.get("spoken")) for h in hist]
        out["brain_skips"] = [x for x in br if x.get("ev") == "brain.skip"]
    out["ledger_tool_events"] = [{k: e.get(k) for k in ("ev", "model_step", "name", "call_id", "status", "args")}
                                 for e in v2 if str(e.get("ev", "")).startswith(("tool.", "function."))]
    out["summary_calls"] = s.get("calls")
    if False:
        pass
    else:
        text = s.get("agent_text_v2", "")
        out["agent_text_v2"] = text
        for t in turns:
            pass
        calls = s.get("calls", [])
        out["m0_calls"] = calls
        replies.append({"label": "all", "reply": text, "lang_id": lang_id(text)[0]})
    out["linguistic"] = replies
    return out


def main():
    run, asr_dir = Path(sys.argv[1]), Path(sys.argv[2])
    res = {}
    for d in sorted(p for p in run.iterdir() if p.is_dir() and (p / "summary.json").exists()):
        lang = json.loads((d / "summary.json").read_text())["lang"]
        asr = json.loads((asr_dir / f"asr_{lang}.json").read_text())
        res[d.name] = score_session(d, asr)
    print(json.dumps(res, indent=1, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
