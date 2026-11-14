"""D-V3 — session-regime distillation of the ML-encoder interface through the frozen VoiceChat Thinker.

Design: .work/multilingual/transplant-diagnosis.md §7, §10 and the owner's addendum (teacher barge-in exclusion).

Front-end (student): ML encoder (trainable LayerNorms) -> ridge affine (C-ridge init, TRAINABLE, low LR) -> residual MLP
-> frozen VC proj. Thinker, text/function heads, fusion, RNNT, EarTTS frozen. No encoder LoRA.

Teacher: A (VC encoder + VC proj, same frozen Thinker) on the PARALLEL EN SESSION:
  lead-in of digital silence (A's opener happens there), the EN caller turn, then a tail.
  Greedy duplex decode with the function channel: main frames consume audio; when the function head emits SOTC the
  live FC fast path is reproduced (agent PAD, the A FC-silence embedding, previous function token) until EOTC; the
  first call ends the sample (no tool result, no post-result training).
Lead-in per family = A's opener end (measured once on silence) + U(0.5, 3.0) s, so the caller starts after the opener
(a shorter lead makes the teacher greet over the caller, which is excluded anyway).

Family filter (owner addendum): if the teacher speaks (non-PAD text) or calls (SOTC) inside the caller's EN turn before
en_end - TOL, or does not act within NO_RESP frames after en_end, the WHOLE family is excluded for every language and
every loss term (text CE, function CE, KL). Nothing is re-timed from a contaminated stream.

Student labels for language X (frames t; cs = caller start = lead frames; x_end / en_end = end of X / EN speech):
  t <  cs                : the teacher's own stream (opener), identical lead-in for EN and X;
  cs <= t < x_end - TOL  : PAD (listening), text and function;
  t >= x_end - TOL       : teacher frame t' = en_end + (t - x_end) (re-timed reply / call onset, same offset);
  after a teacher SOTC   : inserted FC positions (agent PAD, student FC-silence embedding) with the call tokens as
                           function labels up to EOTC, then the sample ends.
Inputs are the labels shifted by one position (teacher forcing), agent PAD at FC positions as in the live wrapper.
Loss = text CE (bos 12.5, eos 7.5, text 5, pad 1 incl. during speech) + function CE (pad 1, SOTC 12.5, call 5)
       + KL(A || student) on the text head, EN samples only, main frames, top-16 renormalised (frame-aligned: same wav)
       + anchor: relative MSE of the adapter output to the frozen ridge output on non-speech frames.
Each component is logged separately. Checkpoints every --save-every steps; resumable (optimizer + RNG + step).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from labpaths import resolve  # noqa: E402  (archived clips: absolute symlinks into the historical layout)

PAD, BOS, EOS, SOTC, EOTC = 12, 1, 2, 20, 21
FRAME = 1280
TOL = 4            # onset tolerance (frames): an action up to 4 frames before the speech end is boundary slop
NO_RESP = 50       # the teacher must act within 50 frames (4 s) of the EN end
TOPK = 16
W_TEXT = {"bos": 12.5, "eos": 7.5, "text": 5.0, "pad": 1.0}
W_FUNC = {"pad": 1.0, "sotc": 12.5, "call": 5.0}


# ----------------------------------------------------------------------------------------------- pure (CPU-tested)
def teacher_filter(rec: dict, cs: int, en_end: int) -> tuple[bool, str]:
    """rec: teacher record {'main_text': [...], 'main_func': [...], 'fc': [...]} indexed by audio frame.
    Returns (keep, reason). Rejects any teacher action inside the caller's turn (barge-in / early call), a teacher
    whose opener has not ended before the caller starts, and a teacher that does not act after the turn."""
    tx, fn = rec["main_text"], rec["main_func"]
    if any(t != PAD for t in tx[cs:cs + 1]):
        return False, "opener_into_caller"
    lo, hi = cs, max(cs, en_end - TOL)
    # a lone EOS ~9 frames after the caller's onset is A's turn-yield marker (311/334 teacher streams, job 285),
    # not speech: only tokens other than PAD/EOS count as talking over the caller
    if any(t not in (PAD, EOS) for t in tx[lo:hi]):
        return False, "talk_during_caller"
    if any(f == SOTC for f in fn[lo:hi]):
        return False, "call_during_caller"
    act = [t for t in range(hi, min(len(tx), en_end + NO_RESP)) if tx[t] != PAD or fn[t] == SOTC]
    if not act:
        return False, "no_response"
    return True, "ok"


def student_targets(rec: dict, cs: int, en_end: int, x_end: int, n_audio: int, is_en: bool) -> dict:
    """Per-position inputs, labels, weights and masks for one student sample (see the module docstring).
    Positions are ('a', frame) for audio frames or ('s', k) for inserted FC positions (student silence embedding)."""
    tx, fn, fc = rec["main_text"], rec["main_func"], rec["fc"]
    pos, tl, fl, kind = [], [], [], []
    start = x_end - TOL
    for t in range(n_audio):
        if t < cs:
            tt, ff = tx[t], fn[t]
        elif t < start:
            # listening; the teacher's onset EOS (yield marker) is kept at the same offset from the caller's onset
            tt, ff = (EOS if t < len(tx) and tx[t] == EOS else PAD), PAD
        else:
            tp = en_end + (t - x_end)
            if tp >= len(tx):
                break
            tt, ff = tx[tp], fn[tp]
        pos.append(("a", t))
        tl.append(tt)
        fl.append(ff)
        kind.append("lead" if t < cs else "speech" if t < x_end else "after")
        if ff == SOTC:
            for k, tok in enumerate(fc):          # the call, then stop (first call only)
                pos.append(("s", k))
                tl.append(PAD)
                fl.append(tok)
                kind.append("fc")
            break
    n = len(pos)
    agent_in = [PAD] + [PAD if kind[i + 1] == "fc" else tl[i] for i in range(n - 1)]
    func_in = [PAD] + fl[:-1]
    tw, fw = [], []
    for i in range(n):
        if kind[i] == "fc":
            tw.append(0.0)                         # agent text is forced PAD during the FC (not a model decision)
            fw.append(W_FUNC["call"])
            continue
        t = tl[i]
        tw.append(W_TEXT["bos"] if t == BOS else W_TEXT["eos"] if t == EOS else W_TEXT["pad"] if t == PAD
                  else W_TEXT["text"])
        fw.append(W_FUNC["sotc"] if fl[i] == SOTC else W_FUNC["pad"] if fl[i] == PAD else W_FUNC["call"])
    # KL to A only for EN students, on main frames whose teacher frame is the SAME audio frame (identity mapping)
    kl = []
    for i, (p, k) in enumerate(zip(pos, kind)):
        kl.append(bool(is_en and p[0] == "a" and x_end == en_end and k != "fc"))
    anchor = [p[0] == "a" and k in ("lead", "after") for p, k in zip(pos, kind)]
    tf = []
    for p in pos:                                  # teacher frame each audio position was labelled from
        if p[0] != "a":
            tf.append(None)
        elif p[1] < cs or is_en:
            tf.append(p[1])                        # EN: x_end == en_end, so every frame maps to itself
        elif p[1] < start:
            tf.append(None)
        else:
            tf.append(en_end + (p[1] - x_end))
    return {"pos": pos, "kind": kind, "text_lbl": tl, "func_lbl": fl, "agent_in": agent_in, "func_in": func_in,
            "text_w": tw, "func_w": fw, "kl": kl, "anchor": anchor, "teacher_frame": tf}


ADMIT_KEEP = ("agree", "teacher_wrong_tool", "teacher_no_call")
REGREET = re.compile(r"how can i help you today", re.I)


def regreet(rec: dict, lead: int, decode) -> bool:
    """Owner 2026-11-13: the SUPERVISED agent text after the caller onset (frames >= lead, PAD dropped; after admission,
    so a PADded teacher_no_call reply never counts) repeats the opener greeting. The opener itself (frames < lead) is
    never inspected or rewritten. Deterministic; the family is quarantined, no token is edited."""
    return bool(REGREET.search(decode([x for x in rec["main_text"][lead:] if x != PAD])))


def rule_call_text(target: dict) -> str:
    return "<TOOLCALL>" + json.dumps([target], ensure_ascii=False) + "</TOOLCALL>"


def admit(rec: dict, rule: dict | None, teacher_tool: str | None, cs: int, en_end: int, offset: int,
          rule_tokens: list[int] | None) -> tuple[str, dict | None]:
    """Corpus-v2 admission (owner, 2026-11-12): a teacher that disagrees with a valid deterministic target does NOT make
    the sample ambiguous. Returns (status, record used for the targets or None = quarantined).
      agree               teacher tool == rule tool (or both no call): teacher timing + RULE call tokens
      teacher_wrong_tool  teacher called another tool: teacher SOTC timing + RULE call tokens (wrong tool not supervised)
      teacher_no_call     rule has a call, teacher replied instead: SOTC at en_end + offset, every teacher text token in
                          [cs, SOTC] replaced by PAD (the onset EOS kept), RULE call tokens (wrong decision not supervised)
      quarantine:called_without_rule  rule has no call but the teacher called (no coherent reply target)
      quarantine:no_response          nothing to supervise after the turn
      quarantine:temporal             the synthetic SOTC would fall outside the decoded window
      quarantine:no_rule_args         free-text call (escalate) whose arguments cannot come from the teacher"""
    tx, fn = list(rec["main_text"]), list(rec["main_func"])
    s = next((t for t in range(cs, len(fn)) if fn[t] == SOTC), None)
    if rule is None:
        if s is not None:
            return "quarantine:called_without_rule", None
        acted = any(x not in (PAD, EOS) for x in tx[cs:])
        return ("agree", rec) if acted else ("quarantine:no_response", None)
    if rule_tokens is None:
        if s is not None and teacher_tool == rule["name"]:
            return "agree", rec                   # free-text arguments: the teacher's own call is kept
        return "quarantine:no_rule_args", None
    new = dict(rec, fc=list(rule_tokens), fc_teacher=rec.get("fc"))
    if s is not None:
        return ("agree" if teacher_tool == rule["name"] else "teacher_wrong_tool"), new
    s = en_end + offset
    if s >= len(tx):
        return "quarantine:temporal", None
    tx = [x if (t < cs or x == EOS) else PAD for t, x in enumerate(tx[:s])] + [PAD]
    fn = fn[:s] + [SOTC]
    new.update(main_text=tx, main_func=fn, topk=rec.get("topk", [])[:s + 1], synthetic_sotc=s)
    return "teacher_no_call", new


def usable_keys(man: dict) -> set:
    """Clips allowed in training: strict ASR verification ok (corpus v2) AND no data exclusion (owner 2026-11-12:
    `excluded: grammar_defect` = IT/FR synonym gender error, card §15.3). Only that clip goes; the family's other
    languages stay."""
    return {k for k, r in man.items() if r.get("asr_ok", True) and not r.get("excluded")}


def kept_keys(items: dict, man_keys: set, split: dict, langs: list[str]) -> tuple[list, list]:
    """(train, val) student keys: ONLY families whose teacher passed the filter; the exclusion is per family, so it
    removes every language and every loss term (CE, function CE, KL) of a contaminated family."""
    keys = [(i, l) for i in sorted(items) if items[i]["keep"] for l in langs if (i, l) in man_keys]
    return [k for k in keys if split[k[0]] == "train"], [k for k in keys if split[k[0]] == "val"]


def correct_call(call_text: str, ent: dict) -> tuple[str | None, list[str]]:
    """Grounded call target (owner goal: slot grounding). The teacher (A on EN audio) chooses the tool and the argument
    keys; slot VALUES come from what the caller actually said (the family's entities). Keys the caller never gave
    (e.g. a postcode in a name-only turn) are removed. Returns (corrected text or None if unparsable, changes)."""
    body = call_text.strip()
    if not (body.startswith("<TOOLCALL>") and body.endswith("</TOOLCALL>")):
        return None, ["unparsable"]
    try:
        calls = json.loads(body[len("<TOOLCALL>"):-len("</TOOLCALL>")])
    except ValueError:
        return None, ["unparsable"]
    changes = []
    gold = {"full_name": ent.get("name"), "postcode": ent.get("postcode")}
    for c in calls:
        args = c.get("arguments", {})
        for k in list(args):
            if k in gold:
                if gold[k] is None:
                    del args[k]
                    changes.append(f"drop:{k}")
                elif str(args[k]).strip() != gold[k]:
                    args[k] = gold[k]
                    changes.append(f"fix:{k}")
    return "<TOOLCALL>" + json.dumps(calls, ensure_ascii=False) + "</TOOLCALL>", changes


def text_class(lbl: int, kind: str) -> str:
    if kind == "fc":
        return "fc"
    if lbl == BOS:
        return "bos"
    if lbl == EOS:
        return "eos"
    if lbl != PAD:
        return "text"
    return "pad_speech" if kind == "speech" else "pad_other"


def param_groups(adapter, ln_params: list, lr: float, affine_mult: float, ln_lr: float) -> list[dict]:
    """Materialised lists (the V2.1 bug: a generator consumed before AdamW left the affine group EMPTY)."""
    aff = list(adapter.affine.parameters())
    rest = [p for n, p in adapter.named_parameters() if not n.startswith("affine.")]
    groups = [{"name": "affine", "params": aff, "lr": lr * affine_mult},
              {"name": "mlp", "params": rest, "lr": lr}]
    if ln_params:
        groups.append({"name": "encoder_ln", "params": list(ln_params), "lr": ln_lr})
    for g in groups:
        assert isinstance(g["params"], list) and g["params"], g["name"]
    return groups


def opener_end_frame(tx: list[int], quiet: int = 15) -> int | None:
    """Last non-PAD agent token of the opener on silence, followed by >= `quiet` PAD frames; None if still talking."""
    last = next((t for t in range(len(tx) - 1, -1, -1) if tx[t] != PAD), None)
    if last is None or len(tx) - 1 - last < quiet:
        return None
    return last


def constrain(t: int, f: int, frame: int, window: tuple | None) -> tuple[int, int, str | None]:
    """Constrained teacher (owner decision after the job-285 filter stop): inside the caller's turn window
    [cs, en_end - TOL) the teacher must listen. A BOS / text token becomes PAD and a SOTC becomes PAD; the onset EOS
    (turn-yield marker) passes. The suppressed token is NOT fed back (the inputs follow the constrained stream), so the
    reply after the turn is A's own reply in the listened-to-the-end context the student sees. Barge-in is never
    supervised. Returns (text, func, what was suppressed or None)."""
    if window is not None and len(window) > 2 and window[1] <= frame < window[2] and f == SOTC:
        return t, PAD, "sotc_after"               # rule says no call: the teacher must answer, not call (corpus v2)
    if window is None or not (window[0] <= frame < window[1]):
        return t, f, None
    why = []
    if t not in (PAD, EOS):
        why.append("text")
        t = PAD
    if f == SOTC:
        why.append("sotc")
        f = PAD
    return t, f, "+".join(why) or None


def lead_frames(opener_end: int, rng: random.Random) -> int:
    return opener_end + 1 + int(round(rng.uniform(0.5, 3.0) / 0.08))


# ----------------------------------------------------------------------------------------------- GPU: prep
def session_wav(clip: np.ndarray, lead: int, tail_s: float) -> np.ndarray:
    return np.concatenate([np.zeros(lead * FRAME, np.float32), clip.astype(np.float32),
                           np.zeros(int(tail_s * 16000), np.float32)])


def cmd_prep(a):  # pragma: no cover - GPU
    import soundfile as sf
    import torch
    from distill_adapter import Duplex, load_engine
    from encoders import SpeechEncoder
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "teacher.json").exists():
        raise SystemExit(f"{out}/teacher.json exists")
    root = Path(a.data)
    rows = [json.loads(l) for l in open(root / "manifest.jsonl")]
    fam = defaultdict(dict)
    for r in rows:
        fam[r["id"]][r["lang"]] = r
    ids = sorted(i for i in fam if "en" in fam[i] and (not a.splits or fam[i]["en"].get("split") in a.splits.split(",")))
    eng = load_engine(out)
    sys.path.insert(0, str(HERE.parent / "agentbench"))
    from m0_runner import render_prompt
    from tenant_backend import TenantBackend
    from mynah_v2.agent.tools import ToolRegistry
    spec = json.load(open(HERE.parent / "agentbench/tenant_furniture.json"))
    be, reg = TenantBackend(spec), ToolRegistry()
    for tl in spec["tools"]:
        reg.register(tl["name"], be.handlers()[tl["name"]], tl["description"], tl["parameters"])
    prompt = render_prompt(reg.schemas, spec["system_prompt"])
    assert len(prompt) == 2738, len(prompt)
    D = Duplex(eng, prompt)
    vc = SpeechEncoder("vc")
    sil_A = D.w._silence_embedding_cache.reshape(-1).to(D.dev, D.dtype)
    t0 = time.monotonic()

    def feats_A(wavs):
        L = max(len(w) for w in wavs)
        fs = [vc(torch.from_numpy(np.pad(w, (0, L - len(w)))))["out"] for w in wavs]
        T = min(f.shape[0] for f in fs)
        with torch.no_grad():
            return D.proj(torch.stack([f[:T] for f in fs]).to(D.dev, D.proj.weight.dtype)), [len(w) // FRAME for w in wavs]

    prev = json.loads((Path(a.merge_from)).read_text()) if a.merge_from else None
    if prev is not None:                       # constrained re-run of a subset: same opener end, same lead per family
        opener_end = prev["opener_end"]
    else:
        # 1) A's opener on pure silence (deterministic for the prompt): end = the last non-PAD agent token followed
        #    by >= 15 PAD frames (job 284: the second opener sentence ends close to 14 s, so 20 s of silence)
        A0, _ = feats_A([np.zeros(int(20 * 16000), np.float32)])
        r0 = greedy_fc(D, A0, sil_A, [A0.shape[1]], topk=0)[0]
        tx = r0["main_text"]
        opener_end = opener_end_frame(tx)
        if opener_end is None:
            raise SystemExit(f"no opener end found on silence: {D.w.tokenizer.ids_to_text([x for x in tx if x != PAD])[:200]}")
        print("opener_end", opener_end, D.w.tokenizer.ids_to_text([x for x in tx[:opener_end + 1] if x != PAD]), flush=True)
    rng = random.Random(a.seed)
    meta = {i: {"lead": lead_frames(opener_end, rng)} for i in ids}     # drawn over ALL ids: leads stay identical
    items = {}
    todo = ids
    redo = set(json.load(open(a.redo_ids))) if a.redo_ids else None
    if prev is not None and redo is not None:     # corpus v2 fixes: re-decode exactly these families, reuse the rest
        items = {i: v for i, v in prev["items"].items() if i not in redo and i in meta}
        for v in items.values():
            v["source"] = "reused"
        todo = [i for i in ids if i not in items]
        print(f"redo {len(todo)} families; {len(items)} reused", flush=True)
    elif prev is not None:
        items = {i: v for i, v in prev["items"].items() if v["keep"]}     # unchanged: the constraint never fires there
        for i, v in items.items():
            assert v["lead"] == meta[i]["lead"], (i, v["lead"], meta[i]["lead"])
            v["source"] = "job285_unconstrained_identical"
        todo = [i for i in ids if i not in items]
        print(f"constrained re-run of {len(todo)} families; {len(items)} reused", flush=True)
    if a.only_ids:                                # decision-checkpoint sample: decode only these, keep nothing else
        only = set(json.load(open(a.only_ids)))
        items = {i: v for i, v in items.items() if i in only}
        todo = [i for i in todo if i in only]
        print(f"only-ids: {len(todo)} to decode, {len(items)} reused", flush=True)
    # 2) teacher decode per family on the EN session
    for b in range(0, len(todo), a.batch):
        chunk = todo[b:b + a.batch]
        wavs, ends = [], []
        for i in chunk:
            x = sf.read(resolve(root / fam[i]["en"]["path"]), dtype="float32")[0]
            wavs.append(session_wav(x, meta[i]["lead"], a.tail_s))
            ends.append(meta[i]["lead"] + int(math.ceil(len(x) / FRAME)))
        Af, n = feats_A(wavs)
        win = [(meta[i]["lead"], e - TOL) + ((10 ** 9,) if a.v2_rules and not fam[i]["en"].get("target") else ())
               for i, e in zip(chunk, ends)]
        recs = greedy_fc(D, Af, sil_A, [min(k, Af.shape[1]) for k in n], topk=TOPK, windows=win)
        for i, rec, e in zip(chunk, recs, ends):
            keep, why = teacher_filter(rec, meta[i]["lead"], e)
            items[i] = {"lead": meta[i]["lead"], "en_end": e, "keep": keep, "reason": why, **rec,
                        "vc_frames": int(Af.shape[1]), "vc_frames_own": int(len(wavs[chunk.index(i)]) // FRAME),
                        "teacher_text": D.w.tokenizer.ids_to_text([x for x in rec["main_text"] if x not in (PAD,)]),
                        "call_text": D.w.tokenizer.ids_to_text([x for x in rec["fc"] if x != EOTC]),
                        "end": {l: meta[i]["lead"] + int(math.ceil(fam[i][l]["dur"] * 16000 / FRAME)) for l in fam[i]}}
        print(f"teacher {b + len(chunk)}/{len(todo)} {time.monotonic() - t0:.0f}s | {items[chunk[0]]['reason']} "
              f"{items[chunk[0]]['call_text'][:60]!r} {items[chunk[0]]['teacher_text'][-80:]!r}", flush=True)
    if a.v2_rules:                                   # corpus v2: rule-derived call targets + admission policy
        tok = D.w.tokenizer

        def enc(t):
            ids = list(tok.text_to_ids(t))
            return ids[1:] if ids and ids[0] == BOS else ids
        offs = [next(t for t in range(v["lead"], len(v["main_func"])) if v["main_func"][t] == SOTC) - v["en_end"]
                for v in items.values() if SOTC in v["main_func"][v["lead"]:] and "synthetic_sotc" not in v]
        off = int(np.median(offs)) if offs else 2
        for i, v in items.items():
            if v.get("source") == "reused":              # already admitted (and possibly rewritten): keep as is
                continue
            rule = fam[i]["en"].get("target")
            m_ = re.search(r'"name": "([a-z_]+)"', v.get("call_text") or "")
            ttool = m_.group(1) if m_ else None
            rt = enc(rule_call_text(rule)) + [EOTC] if rule and rule.get("arguments") is not None else None
            st, new = admit(v, rule, ttool, v["lead"], v["en_end"], off, rt)
            if new is not None:
                v.update(new)
            v["keep"], v["reason"], v["teacher_tool"] = st in ADMIT_KEEP, st, ttool
            v["rule_call_text"] = rule_call_text(rule) if rt is not None else None   # {} args (escalate) included
        for v in items.values():                     # every kept family, REUSED pilot families included
            if v["keep"] and regreet(v, v["lead"], tok.ids_to_text):
                v["keep"], v["reason"] = False, "quarantine:regreet"
        print("ADMISSION offset", off, dict(Counter(v["reason"] for v in items.values())), flush=True)
    for i, v in items.items():                       # per-language speech ends from THIS manifest (re-rendered clips)
        v["end"] = {l: v["lead"] + int(math.ceil(fam[i][l]["dur"] * 16000 / FRAME)) for l in fam[i]}
    rep = filter_report(items, fam)
    (out / "teacher.json").write_text(json.dumps({"prompt": prompt, "opener_end": opener_end, "tail_s": a.tail_s,
                                                  "items": items, "filter": rep}, ensure_ascii=False))
    print("FILTER", json.dumps(rep), flush=True)
    print("prep done", len(items), f"{time.monotonic() - t0:.0f}s")


def filter_report(items: dict, fam: dict) -> dict:
    """Exclusions per reason, intent, language duration bucket and phrase count (owner addendum item 4)."""
    rep = {"n": len(items), "kept": sum(v["keep"] for v in items.values()), "reason": dict(Counter(v["reason"] for v in items.values())),
           "constrained_families": sum(bool(v.get("suppressed")) for v in items.values()),
           "suppressed_kinds": dict(Counter(k for v in items.values() for _, k in v.get("suppressed", [])))}
    by = defaultdict(lambda: [0, 0])
    for i, v in items.items():
        r = fam[i]["en"]
        sl = r.get("slots") or {}
        diff = [f"attrs:{len(sl.get('attrs') or [])}"] if sl.get("concept") else []
        if sl.get("synonym") and any((sl.get("synonym") or {}).values()):
            diff.append("synonym:yes")
        if sl.get("postcode_form"):
            diff.append(f"postcode:{sl['postcode_form']}")
        for key in [f"intent:{r['cat']}", f"phrases:{r.get('n_phrases')}"] + diff:
            by[key][0] += 1
            by[key][1] += int(v["keep"])
        for lang, rr in fam[i].items():
            b = "<3s" if rr["dur"] < 3 else "3-5s" if rr["dur"] < 5 else "5-7s" if rr["dur"] < 7 else ">=7s"
            by[f"{lang}:{b}"][0] += 1
            by[f"{lang}:{b}"][1] += int(v["keep"])
    rep["by"] = {k: {"n": n, "kept": kk, "kept_frac": round(kk / n, 3)} for k, (n, kk) in sorted(by.items())}
    return rep


def greedy_fc(D, audio, sil, n_frames: list[int], topk: int = 0, max_fc: int = 64, windows=None,
              user_text=None):  # pragma: no cover - GPU
    """Batched duplex greedy decode with the function channel and the live FC fast path (see module docstring).
    user_text (V1, optional): [B, T, H] user-text channel embeddings indexed by audio frame, read in main mode only;
    FC fast-path steps get a zero contribution. None keeps the original code path."""
    import torch
    from mynah_v2.model.llm_cache import _hybrid_cache_cls
    B = audio.shape[0]
    cache = _hybrid_cache_cls(D.stt.llm)(D.stt.llm.config, B, dtype=D.dtype, device=D.dev, ssm_dtype=torch.float32)
    D.stt(D.prompt_emb.expand(B, -1, -1).contiguous(), cache=cache)
    e = D.stt.embed_tokens
    mode, ptr = ["main"] * B, [0] * B
    pt = torch.full((B,), PAD, dtype=torch.long, device=D.dev)
    pf = torch.full((B,), PAD, dtype=torch.long, device=D.dev)
    recs = [{"main_text": [], "main_func": [], "fc": [], "topk": [], "suppressed": []} for _ in range(B)]
    windows = windows or [None] * B
    with torch.no_grad():
        while any(m != "done" for m in mode):
            rows = torch.stack([(audio[i, min(ptr[i], audio.shape[1] - 1)] if mode[i] == "main" else sil).to(D.dtype)
                                for i in range(B)])[:, None]
            ut = None
            if user_text is not None:
                ut = torch.stack([(user_text[i, min(ptr[i], user_text.shape[1] - 1)] if mode[i] == "main"
                                   else torch.zeros_like(user_text[i, 0])).to(D.dtype) for i in range(B)])[:, None]
            x = D.fusion(agent_text_embeds=e(pt[:, None]).to(D.dtype), user_audio_embeds=rows.to(D.dtype),
                         user_text_embeds=ut, function_embeds=e(pf[:, None]).to(D.dtype))
            ans = D.stt(x, cache=cache)
            cache = ans.get("cache", cache)
            lt = torch.log_softmax(ans["text_logits"][:, -1].float(), -1)
            lf = ans["function_logits"][:, -1].float()
            ta, fa = lt.argmax(-1), lf.argmax(-1)
            for i in range(B):
                r = recs[i]
                if mode[i] == "main":
                    t, f = int(ta[i]), int(fa[i])
                    t, f, sup = constrain(t, f, ptr[i], windows[i])
                    if sup:
                        r["suppressed"].append([ptr[i], sup])
                    if f == SOTC:
                        t = PAD                              # the wrapper forces agent PAD at the FC entry
                    r["main_text"].append(t)
                    r["main_func"].append(f)
                    if topk:
                        v, k = lt[i].topk(topk)
                        r["topk"].append([k.tolist(), [round(float(z), 4) for z in v]])
                    ptr[i] += 1
                    pt[i], pf[i] = t, f
                    if f == SOTC:
                        mode[i] = "fc"
                    elif ptr[i] >= n_frames[i]:
                        mode[i] = "done"
                elif mode[i] == "fc":
                    f = int(fa[i])
                    r["fc"].append(f)
                    pt[i], pf[i] = PAD, f
                    if f == EOTC or len(r["fc"]) >= max_fc:
                        mode[i] = "done"
    return recs


# ----------------------------------------------------------------------------------------------- GPU: train
def pick_distractor(fid: str, target: dict, slots: dict) -> dict | None:
    """L2 (pre-registered 2026-11-13): candidates from cand_test.distractors restricted to TRAIN-lexicon kinds
    (full_name 'train', postcode 'last', query 'train'); one per family, chosen by a stable hash of the family id."""
    from cand_test import distractors
    c = [d for d in distractors(target, slots or {}) if (d["slot"], d["kind"]) in
         (("full_name", "train"), ("postcode", "last"), ("query", "train"))]
    if not c:
        return None
    h = int(hashlib.sha256(fid.encode()).hexdigest(), 16)
    return c[h % len(c)]


def contrastive_targets(items: dict, man: dict, enc, train_ids: set) -> tuple[dict, dict]:
    out, st = {}, Counter()
    for i, v in items.items():
        if i not in train_ids or not v.get("fc"):
            continue
        m = man.get((i, "en"), {})
        t = m.get("target")
        if not t or not t.get("arguments") or enc(rule_call_text(t)) + [EOTC] != v["fc"]:
            st["skip_no_canonical_target"] += 1
            continue
        d = pick_distractor(i, t, m.get("slots"))
        if d is None:
            st["skip_no_distractor"] += 1
            continue
        out[i] = enc(rule_call_text(d["target"])) + [EOTC]
        st[f"{d['slot']}|{d['kind']}"] += 1
    return out, dict(st)


def cmd_train(a):  # pragma: no cover - GPU
    import torch
    from adapters import ResidualAdapter, load_adapter, save_adapter
    from distill_adapter import Duplex, LiveML, load_engine
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    prep = json.loads((Path(a.prep) / "teacher.json").read_text())
    man = {(r["id"], r["lang"]): r for r in map(json.loads, open(Path(a.data) / "manifest.jsonl"))}
    split = {r["id"]: r["split"] for r in man.values()}
    usable = usable_keys(man)
    tr, va = kept_keys(prep["items"], usable, split, a.langs.split(","))
    assert not any(split[k[0]] not in ("train", "val") for k in tr + va)   # idev / G-LEX never trained
    items = {i: v for i, v in prep["items"].items() if v["keep"]}
    eng = load_engine(out)
    D = Duplex(eng, prep["prompt"])
    # grounded call targets: teacher tool + keys, caller's slot values (job 287 stop: A misspelled 79/125 names)
    tok = D.w.tokenizer

    def enc(t):
        ids = list(tok.text_to_ids(t))
        return ids[1:] if ids and ids[0] == BOS else ids
    fixes, same_tok = Counter(), [0, 0]
    for i, v in items.items():
        if not v["fc"]:
            continue
        if "fc_teacher" in v:      # corpus v2: fc = the canonical RULE call (admit); never re-derived from the teacher
            fixes["rule_target_kept"] += 1
            continue
        ent = man[(i, "en")].get("entities") or {}
        new, ch = correct_call(v["call_text"], ent)
        if new is None:
            fixes["unparsable"] += 1
            continue
        if not ch:                                      # tokenisation sanity: an unchanged call must round-trip
            same_tok[0] += 1
            same_tok[1] += int(enc(new) == v["fc"][:-1])
            continue
        v["fc_teacher"] = v["fc"]
        v["fc"] = enc(new) + [EOTC]
        v["call_text_grounded"] = new
        fixes.update(ch)
    print("CALL TARGETS", dict(fixes), "unchanged round-trip", same_tok, flush=True)
    if same_tok[0] and same_tok[1] < 0.9 * same_tok[0]:
        raise SystemExit("call re-tokenisation does not reproduce the teacher's tokens")
    tkl = None
    if a.kl_teacher:                                 # B1: masked text-teacher KL on tool / full_name / postcode tokens
        tkl = torch.load(a.kl_teacher, map_location="cpu", weights_only=False)
        print("KL-TEACHER", a.kl_teacher, len(tkl), "keys,", sum(1 for v in tkl.values() if v["pos"]),
              "with distilled positions; weight", a.kl_call_w, flush=True)
    dist_fc = {}
    if a.contrastive_w > 0:                          # L2: one pre-registered TRAIN-lexicon distractor per call family
        dist_fc, dstat = contrastive_targets(items, man, enc, set(k[0] for k in tr))
        print("CONTRASTIVE", dstat, flush=True)
        (Path(a.out) / "contrastive_targets.json").write_text(json.dumps(
            {i: tok.ids_to_text([t for t in v if t != EOTC]) for i, v in dist_fc.items()}, ensure_ascii=False))
    torch.manual_seed(a.seed)
    lora = None
    if a.lora:                       # L1: bridge FROZEN at the given checkpoint, only the Thinker LoRA trains
        import thinker_lora as TL
        assert a.bridge_adapter and a.bridge_ln and not a.init_from and a.anchor_w == 0.0
        ad = load_adapter(a.bridge_adapter).to(D.dev).float()
        for p in ad.parameters():
            p.requires_grad_(False)
        M0 = m0 = None
        live = SessionML(a.data, items, prep["tail_s"], D.dev)
        st = torch.load(a.bridge_ln, map_location="cpu", weights_only=False)
        assert set(st) == set(live.ln), "bridge LN state does not match the encoder LayerNorms"
        for n, p in live.ln.items():
            p.data.copy_(st[n].to(p.device, p.dtype))
            p.requires_grad_(False)
        lora = {"layers": [int(x) for x in a.lora_layers.split(",")], "targets": a.lora_targets.split(","),
                "r": a.lora_r, "alpha": a.lora_alpha}
        lora["mods"] = TL.install(D.stt.llm, layers=tuple(lora["layers"]), targets=tuple(lora["targets"]),
                                  r=a.lora_r, alpha=a.lora_alpha, seed=a.seed)
        groups = [{"name": "lora", "params": TL.lora_parameters(lora["mods"]), "lr": a.lr}]
        if a.lora_frozen:            # B0 ablation: LoRA installed at B = 0 (exact identity) but NOT trained
            for p in TL.lora_parameters(lora["mods"]):
                p.requires_grad_(False)
            groups = []
        lora["joint"] = bool(a.lora_joint)
        if a.lora_joint:             # B0: encoder LNs + adapter (affine + MLP) + LoRA trained JOINTLY (D-V3 bridge LRs)
            for p in ad.parameters():
                p.requires_grad_(True)
            for p in live.ln.values():
                p.requires_grad_(True)
            groups = param_groups(ad, list(live.ln.values()), a.lr, a.affine_lr_mult, a.ln_lr) + groups
        assert groups, "nothing to train"
        print("LORA", {k: v for k, v in lora.items() if k != "mods"}, TL.n_params(lora["mods"]), "params; bridge",
              a.bridge_adapter, a.bridge_ln, flush=True)
    else:
        ad = ResidualAdapter().to(D.dev)
        ridge = torch.load(a.init_map, map_location="cpu", weights_only=True)
        ad.init_affine(ridge["M"].to(D.dev), ridge["m"].to(D.dev))
        M0, m0 = ridge["M"].float().to(D.dev), ridge["m"].float().to(D.dev)
        live = SessionML(a.data, items, prep["tail_s"], D.dev)
        groups = param_groups(ad, list(live.ln.values()), a.lr, a.affine_lr_mult, a.ln_lr)
    trainable = [p for g in groups for p in g["params"]]
    opt = torch.optim.AdamW([{k: v for k, v in g.items() if k != "name"} for g in groups], weight_decay=0.0)
    step, rng = 0, random.Random(1)
    if a.init_from and not a.resume:              # warm start (weights only): fresh AdamW, step 0, fresh sampler RNG
        ck = torch.load(Path(a.init_from), map_location="cpu", weights_only=False)
        ad.load_state_dict(ck["adapter"])
        for n, p in live.ln.items():
            p.data.copy_(ck["ln"][n].to(p.device))
        print("init from", a.init_from, "(weights of step", ck.get("step"), "); optimizer fresh", flush=True)
    if a.resume:
        ck = torch.load(Path(a.resume), map_location="cpu", weights_only=False)
        if lora:
            TL.load_state(lora["mods"], ck["lora"])
            if lora.get("joint"):
                ad.load_state_dict(ck["adapter"])
                for n, p in live.ln.items():
                    p.data.copy_(ck["ln"][n].to(p.device))
        else:
            ad.load_state_dict(ck["adapter"])
            for n, p in live.ln.items():
                p.data.copy_(ck["ln"][n].to(p.device))
        opt.load_state_dict(ck["opt"])
        step = ck["step"]
        rng.setstate(ck["rng"])
        print("resumed at step", step, flush=True)
    print(f"train keys {len(tr)} {dict(Counter(k[1] for k in tr))} | val {len(va)} | groups "
          f"{[(g['name'], sum(p.numel() for p in g['params'])) for g in groups]}", flush=True)

    def batch_loss(ks, comps, contrast=False):
        feats = live.feats(ks)                                     # list of [T, 1024], grads into the LNs
        sil = live.silence()                                       # [1024], 1 s of zeros through the encoder
        tot_text, tot_func, tot_kl, tot_anc = 0.0, 0.0, 0.0, 0.0
        wsum_t = wsum_f = n_kl = n_anc = 0.0
        seqs, pairs = [], []
        for (i, l), f in zip(ks, feats):
            it = items[i]
            st = student_targets(it, it["lead"], it["en_end"], it["end"][l], f.shape[0], l == "en")
            src = torch.stack([f[p[1]] if p[0] == "a" else sil for p in st["pos"]])
            seqs.append((st, src))
        B0 = len(seqs)
        if contrast:                                               # same audio and prefix, distractor call tokens
            for b, ((i, l), f) in enumerate(zip(ks, feats)):
                if i in dist_fc and SOTC in seqs[b][0]["func_lbl"]:
                    it = {**items[i], "fc": dist_fc[i]}
                    st = student_targets(it, it["lead"], it["en_end"], it["end"][l], f.shape[0], l == "en")
                    src = torch.stack([f[p[1]] if p[0] == "a" else sil for p in st["pos"]])
                    pairs.append((b, len(seqs)))
                    seqs.append((st, src))
        B = len(seqs)
        X = torch.nn.utils.rnn.pad_sequence([src.float() for _, src in seqs], batch_first=True)
        rows = [st for st, _ in seqs]
        Tm = X.shape[1]
        Y = ad(X)                                                  # [B, T, 1024]
        R = X @ M0 + m0 if M0 is not None else None                # frozen ridge reference (none with --lora)
        audio = D.proj(Y.to(D.proj.weight.dtype))
        e = D.stt.embed_tokens
        AI = torch.full((B, Tm), PAD, dtype=torch.long, device=D.dev)
        FI = AI.clone()
        for b, st in enumerate(rows):
            AI[b, :len(st["pos"])] = torch.tensor(st["agent_in"], device=D.dev)
            FI[b, :len(st["pos"])] = torch.tensor(st["func_in"], device=D.dev)
        fr = D.fusion(agent_text_embeds=e(AI).to(D.dtype), user_audio_embeds=audio.to(D.dtype), user_text_embeds=None,
                      function_embeds=e(FI).to(D.dtype))
        x = torch.cat([D.prompt_emb.expand(B, -1, -1), fr], 1)
        o = D.stt(x, cache=None)
        P = D.prompt_emb.shape[1]
        lt, lf = o["text_logits"][:, P:].float(), o["function_logits"][:, P:].float()
        tot_klc, n_klc = 0.0, 0
        if tkl is not None:                                        # B1 masked KL(p_teacher^{top32, renorm} || p_student)
            for b, (i, l) in enumerate(ks[:B0]):
                ent = tkl.get(f"{i}|{l}")
                if not ent or not ent["pos"]:
                    continue
                fcx = [j for j, kd in enumerate(rows[b]["kind"]) if kd == "fc"]
                if len(fcx) <= max(ent["pos"]):
                    continue
                pos = [fcx[j] for j in ent["pos"]]
                tp = torch.softmax(ent["lp"].float().to(D.dev), -1)
                sl = torch.log_softmax(lf[b, pos], -1).gather(1, ent["ids"].long().to(D.dev))
                kl = (tp * (tp.clamp_min(1e-9).log() - sl)).sum(1)
                tot_klc = tot_klc + kl.sum()
                n_klc += len(pos)
                comps["kl_call"].extend(kl.detach().tolist())
        tot_con = 0.0
        for bg, bd in pairs:                                       # L2: softplus(m - [log P(y+) - log P(y-)])
            lps = []
            for b in (bg, bd):
                st = rows[b]
                idx = [j for j, k in enumerate(st["kind"]) if k == "fc"]
                lab = torch.tensor([st["func_lbl"][j] for j in idx], device=D.dev)
                lps.append(torch.log_softmax(lf[b, idx], -1).gather(1, lab[:, None]).sum())
            delta = lps[0] - lps[1]
            tot_con = tot_con + torch.nn.functional.softplus(a.contrastive_margin - delta)
            comps["contrast_delta"].append(float(delta.detach()))
        if pairs:
            comps["contrast_loss"].append(float(tot_con.detach()) / len(pairs))
        rows = rows[:B0]
        for b, st in enumerate(rows):
            n = len(st["pos"])
            tl = torch.tensor(st["text_lbl"], device=D.dev)
            fl = torch.tensor(st["func_lbl"], device=D.dev)
            tw = torch.tensor(st["text_w"], device=D.dev)
            fw = torch.tensor(st["func_w"], device=D.dev)
            ct = torch.nn.functional.cross_entropy(lt[b, :n], tl, reduction="none")
            cf = torch.nn.functional.cross_entropy(lf[b, :n], fl, reduction="none")
            tot_text = tot_text + (ct * tw).sum()
            tot_func = tot_func + (cf * fw).sum()
            wsum_t += float(tw.sum())
            wsum_f += float(fw.sum())
            ctc, cfc = ct.detach().cpu().tolist(), cf.detach().cpu().tolist()   # one sync per sample
            for j in range(n):                                    # per-component bookkeeping (detached)
                comps[f"text:{text_class(st['text_lbl'][j], st['kind'][j])}"].append(ctc[j])
                fcl = "sotc" if st["func_lbl"][j] == SOTC else "call" if st["kind"][j] == "fc" else "pad"
                comps[f"func:{fcl}"].append(cfc[j])
            km = torch.tensor(st["kl"], device=D.dev)
            if km.any():
                it = items[ks[b][0]]
                idx = [j for j in range(n) if st["kl"][j]]
                tf = [st["teacher_frame"][j] for j in idx]
                top = [it["topk"][t] for t in tf if t is not None and t < len(it["topk"])]
                idx = idx[:len(top)]
                tk = torch.tensor([t[0] for t in top], device=D.dev)
                tp = torch.softmax(torch.tensor([t[1] for t in top], device=D.dev), -1)    # renormalised top-k
                sl = torch.log_softmax(lt[b, idx], -1).gather(1, tk)
                kl = (tp * (tp.clamp_min(1e-9).log() - sl)).sum(1)
                tot_kl = tot_kl + kl.sum()
                n_kl += len(idx)
                comps["kl_en"].extend(kl.detach().tolist())
            am = torch.tensor(st["anchor"] + [False] * 0, device=D.dev)
            if am.any() and R is not None:
                d = ((Y[b, :n][am] - R[b, :n][am]) ** 2).mean(1) / (R[b, :n][am] ** 2).mean(1).clamp_min(1e-6)
                tot_anc = tot_anc + d.sum()
                n_anc += int(am.sum())
                comps["anchor"].extend(d.detach().tolist())
        loss = (tot_text / max(wsum_t, 1e-6) + tot_func / max(wsum_f, 1e-6) + a.kl_w * tot_kl / max(n_kl, 1)
                + a.anchor_w * tot_anc / max(n_anc, 1))
        if pairs:
            loss = loss + a.contrastive_w * tot_con / len(pairs)
        if n_klc:
            loss = loss + a.kl_call_w * tot_klc / n_klc
        return loss

    def save_ckpt(tag):
        if lora:
            meta = {k: v for k, v in lora.items() if k != "mods"}
            torch.save({**meta, "state": TL.state(lora["mods"]), "step": step, "bridge_adapter": a.bridge_adapter,
                        "bridge_ln": a.bridge_ln}, out / f"lora_{tag}.pt")
            extra = {}
            if lora.get("joint"):                  # B0: the bridge moved too; save it in the usual D-V3 formats
                save_adapter(ad, out / f"adapter_{tag}.pt", {"step": step})
                torch.save(live.ln_state(), out / f"encoder_ln_{tag}.pt")
                extra = {"adapter": ad.state_dict(), "ln": {n: p.detach().cpu() for n, p in live.ln.items()}}
            torch.save({"lora": TL.state(lora["mods"]), "opt": opt.state_dict(), "step": step, "rng": rng.getstate(),
                        **extra}, out / "resume.pt")
            return
        save_adapter(ad, out / f"adapter_{tag}.pt", {"step": step})
        torch.save(live.ln_state(), out / f"encoder_ln_{tag}.pt")
        torch.save({"adapter": ad.state_dict(), "ln": {n: p.detach().cpu() for n, p in live.ln.items()},
                    "opt": opt.state_dict(), "step": step, "rng": rng.getstate()}, out / "resume.pt")

    if lora and (a.smoke or a.smoke_first):
        import l1_smoke
        l1_smoke.run(a, out, D, ad, live, items, tr, batch_loss, opt, lora)    # measures only; never picks the LR
        opt.zero_grad(set_to_none=True)
        if a.smoke:
            return
    elif a.smoke or a.smoke_first:
        smoke_checks(a, out, D, ad, live, groups, tr, batch_loss, M0, m0)   # raises SystemExit if a check fails
        opt.zero_grad(set_to_none=True)
        if a.smoke:
            return
    log = []
    t0 = time.monotonic()
    while time.monotonic() - t0 < a.max_minutes * 60 and step < a.max_steps:
        comps = defaultdict(list)
        opt.zero_grad()
        tot = 0.0
        batch_ids = []
        for _ in range(a.accum):                 # job 288: batch 8 exceeded unified memory (smoke at 4: 55.4 GiB)
            ks = rng.sample(tr, a.batch)         # same sampler draws as L1 in every arm
            batch_ids.append([f"{i}|{l}" for i, l in ks])
            if a.contrastive_w > 0:              # L2: gold + distractor sequences -> two halves of <= 4 sequences
                h = len(ks) // 2
                for part in (ks[:h], ks[h:]):
                    loss = batch_loss(part, comps, contrast=True) / (a.accum * 2)
                    loss.backward()
                    tot += float(loss)
                continue
            loss = batch_loss(ks, comps) / a.accum
            loss.backward()
            tot += float(loss)
        loss = torch.tensor(tot)
        gn = {g["name"]: float(torch.sqrt(sum((p.grad.float() ** 2).sum() for p in g["params"] if p.grad is not None)))
              for g in groups}
        if tkl is not None and step == 0:                          # B1 inline check (owner): stop before any update
            bad = [n for n, p in D.stt.named_parameters() if p.grad is not None]
            klv = comps.get("kl_call", [])
            chk = {"thinker_params_with_grad": bad[:5], "n_kl_call": len(klv),
                   "kl_call_finite": bool(klv) and all(math.isfinite(x) for x in klv),
                   "bridge_grads": {k: round(v, 4) for k, v in gn.items()},
                   "comp_step1": {k: round(float(np.mean(v)), 4) for k, v in comps.items()}}
            print("B1-CHECK", json.dumps(chk), flush=True)
            (out / "b1_step1_check.json").write_text(json.dumps(chk, indent=1))
            if bad or not chk["kl_call_finite"] or not all(math.isfinite(v) and v > 0 for v in gn.values()):
                raise SystemExit("B1 step-1 check failed: stop before any update")
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        step += 1
        row = {"step": step, "batch_ids": batch_ids, "loss": round(float(loss), 4), "grad": {k: round(v, 4) for k, v in gn.items()},
               "comp": {k: round(float(np.mean(v)), 4) for k, v in comps.items()}, "t": round(time.monotonic() - t0)}
        log.append(row)
        if step % 10 == 0:
            print(json.dumps(row), flush=True)
        if a.save_every and step % a.save_every == 0:
            save_ckpt(f"step{step}")
            with torch.no_grad():
                vc = defaultdict(list)
                for k in range(0, min(len(va), 48), a.batch):
                    batch_loss(va[k:k + a.batch], vc)
            log.append({"step": step, "val_comp": {k: round(float(np.mean(v)), 4) for k, v in vc.items()}})
            print("VAL", json.dumps(log[-1]), flush=True)
    save_ckpt(f"step{step}")
    with open(out / "train_log.jsonl", "a") as f:
        for r in log:
            f.write(json.dumps(r) + "\n")
    print("done", step, flush=True)


def smoke_checks(a, out, D, ad, live, groups, tr, batch_loss, M0, m0):  # pragma: no cover - GPU
    """Owner checks 1 and 3 on the real model: every trainable group gets a non-zero gradient; lead-in and pause frames
    are ENCODED by the ML encoder (non-zero, equal to the encoder's own output on zeros), never zero vectors."""
    import torch
    rep = {}
    ks = tr[:4]
    feats = live.feats(ks)
    for (i, l), f in zip(ks, feats):
        lead = f[: live.items[i]["lead"]]
        rep.setdefault("lead_norm_min", []).append(round(float(lead.norm(dim=1).min()), 4))
        rep.setdefault("lead_zero_rows", []).append(int((lead.abs().sum(1) == 0).sum()))
    rep["silence_emb_norm"] = round(float(live.silence().norm()), 4)
    en = [k for k in tr if k[1] == "en"][:8]
    fe = live.feats(en)
    rep["en_frame_diff_vs_teacher"] = [int(f.shape[0]) - int(math.ceil(len(live._wav(k)) / FRAME)) for k, f in zip(en, fe)]
    rep["en_teacher_main_len_le_frames"] = all(len(live.items[k[0]]["main_text"]) <= f.shape[0] + 1 for k, f in zip(en, fe))
    comps = defaultdict(list)
    loss = batch_loss(ks, comps)
    loss.backward()
    rep["grad_norm"] = {g["name"]: float(torch.sqrt(sum((p.grad.float() ** 2).sum() for p in g["params"] if p.grad is not None)))
                        for g in groups}
    rep["grad_none"] = {g["name"]: sum(p.grad is None for p in g["params"]) for g in groups}
    rep["comp"] = {k: round(float(np.mean(v)), 4) for k, v in comps.items()}
    rep["checks"] = {"all_groups_nonzero_grad": all(v > 0 for v in rep["grad_norm"].values()),
                     "no_zero_lead_rows": all(z == 0 for z in rep["lead_zero_rows"]),
                     "lead_encoded": all(v > 0 for v in rep["lead_norm_min"]),
                     "en_aligned": all(abs(d) <= 1 for d in rep["en_frame_diff_vs_teacher"]) and rep["en_teacher_main_len_le_frames"]}
    rep["peak_cuda_gib"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
    (out / "smoke.json").write_text(json.dumps(rep, indent=1))
    print("SMOKE", json.dumps(rep), flush=True)
    if not all(rep["checks"].values()):
        raise SystemExit("smoke checks failed")


class SessionML:  # pragma: no cover - GPU
    """ML encoder with trainable LayerNorms (as E5/V2.1 LiveML), on the SESSION wav: lead-in zeros + clip + tail,
    so silence is encoded by the encoder (never a zero vector)."""

    def __init__(self, data_root, items, tail_s, dev):
        import torch
        from encoders import SpeechEncoder
        self.root = Path(data_root)
        self.items = items
        self.path = {(r["id"], r["lang"]): self.root / r["path"] for r in map(json.loads, open(self.root / "manifest.jsonl"))}
        self.tail_s = tail_s
        self.se = SpeechEncoder("ml")
        self.se.eval()
        for p in self.se.parameters():
            p.requires_grad_(False)
        self.ln = {n: p for n, p in self.se.enc.named_parameters()
                   if isinstance(self.se.enc.get_submodule(n.rsplit(".", 1)[0]), torch.nn.LayerNorm)}
        for p in self.ln.values():
            p.requires_grad_(True)
        self.cache = {}

    def _wav(self, key):
        import soundfile as sf
        import torch
        if key not in self.cache:
            x = sf.read(resolve(self.path[key]), dtype="float32")[0]
            self.cache[key] = torch.from_numpy(session_wav(x, self.items[key[0]]["lead"], self.tail_s))
        return self.cache[key]

    def _enc(self, ws):
        import torch
        n = torch.tensor([len(w) for w in ws], device=self.se.device)
        x = torch.stack([torch.nn.functional.pad(w, (0, int(n.max()) - len(w))) for w in ws]).to(self.se.device)
        mel, ml = self.se.pre(input_signal=x, length=n)
        e, el = self.se.enc(audio_signal=mel, length=ml)
        return [e[b, :, : int(el[b])].transpose(0, 1) for b in range(len(ws))]

    def feats(self, keys):
        return self._enc([self._wav(k) for k in keys])

    def silence(self):
        import torch
        return self._enc([torch.zeros(16000)])[0][-1]           # 1 s of digital silence, as the wrapper's cache

    def ln_state(self):
        return {n: p.detach().cpu().clone() for n, p in self.ln.items()}


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep")
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--batch", type=int, default=24)
    p.add_argument("--tail-s", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=20261112)
    p.add_argument("--splits", default="", help="only families of these splits (e.g. train,val: no probe families)")
    p.add_argument("--v2-rules", action="store_true", help="corpus v2: rule-derived call targets + admission policy")
    p.add_argument("--redo-ids", default="", help="json list of family ids to re-decode (with --merge-from)")
    p.add_argument("--only-ids", default="", help="json list: decode only these families (leads still drawn over all ids)")
    p.add_argument("--merge-from", default="", help="previous teacher.json: reuse its kept families, re-decode the rest "
                                                   "with the constrained teacher")
    t = sub.add_parser("train")
    t.add_argument("--prep", required=True)
    t.add_argument("--data", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--init-map", default="", help="ridge map (bridge training; not used with --lora)")
    t.add_argument("--langs", default="en,it,fr")
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--affine-lr-mult", type=float, default=0.1)
    t.add_argument("--ln-lr", type=float, default=5e-5)
    t.add_argument("--kl-w", type=float, default=1.0)
    t.add_argument("--anchor-w", type=float, default=1.0)
    t.add_argument("--batch", type=int, default=4)
    t.add_argument("--accum", type=int, default=2, help="micro-batches per optimizer step (effective batch = batch*accum)")
    t.add_argument("--max-minutes", type=float, default=12.0)
    t.add_argument("--max-steps", type=int, default=100000)
    t.add_argument("--save-every", type=int, default=100)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--resume", default="")
    t.add_argument("--init-from", default="", help="resume.pt of a previous run: adapter + encoder LN weights only "
                   "(fresh optimizer and step 0); ignored when --resume is given")
    t.add_argument("--lora", action="store_true", help="L1: Thinker LoRA, bridge frozen (--bridge-adapter/--bridge-ln)")
    t.add_argument("--bridge-adapter", default="")
    t.add_argument("--bridge-ln", default="")
    t.add_argument("--lora-joint", action="store_true", help="B0: also train the encoder LNs + adapter with the LoRA")
    t.add_argument("--lora-frozen", action="store_true", help="B0 ablation: keep the LoRA at B = 0, untrained (bridge only)")
    t.add_argument("--lora-layers", default="14,21,30,39")
    t.add_argument("--lora-targets", default="q_proj,v_proj")
    t.add_argument("--lora-r", type=int, default=8)
    t.add_argument("--lora-alpha", type=float, default=16.0)
    t.add_argument("--smoke-steps", type=int, default=10, help="--lora smoke: optimiser steps measured, then discarded")
    t.add_argument("--smoke-sigma", type=float, default=0.01, help="--lora smoke: std of the temporary B perturbation")
    t.add_argument("--smoke-lrs", default="", help="--lora smoke: comma-separated LRs, each measured from the same state")
    t.add_argument("--contrastive-w", type=float, default=0.0, help="L2: weight of the slot contrastive loss (0 = L1)")
    t.add_argument("--kl-teacher", default="", help="B1: teacher_kl.pt (masked text-teacher KL on call tokens)")
    t.add_argument("--kl-call-w", type=float, default=1.0)
    t.add_argument("--contrastive-margin", type=float, default=2.0)
    t.add_argument("--smoke", action="store_true")
    t.add_argument("--smoke-first", action="store_true", help="run the smoke checks, then train in the same process")
    a = ap.parse_args()
    {"prep": cmd_prep, "train": cmd_train}[a.cmd](a)


if __name__ == "__main__":
    main()
