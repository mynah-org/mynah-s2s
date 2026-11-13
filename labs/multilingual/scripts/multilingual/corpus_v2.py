"""Corpus V2 for the multilingual VoiceChat interface (design: .work/multilingual/corpus-v2-design.md).

Subcommands (all outputs under ml/data/v2 and never overwritten):
  cv     (CPU)  pick Common Voice IT/FR clips from the owner's existing CV pools on the lab box (16 kHz mono already):
                one clip per speaker, 2.5-8 s, 4-20 words, train/val speakers disjoint; copies + sha256.
  minds  (CPU)  MINDS-14 it/fr/en natural-speech EVAL set (eval-only, as data/eval/v1 already treats it):
                8 kHz telephone audio decoded and resampled to 16 kHz (no bandwidth recovery: flagged
                `band_limited_8k`), intent label -> deterministic English keyword regex for scoring.
  gen    (GPU)  Nano-9B: semantic families per (phenomenon, topic) as JSON {q, a, kw}; answer validation by
                Nano itself (kept only if its own brief answer contains a keyword); dedup; translations
                to it/fr/de/es; English translations of the selected CV transcripts. Batched, left padding.
  build  (CPU)  family-level splits (test/dev/val/train), leakage checks, QA v2 corpus json, TTS render plans.
  assemble (CPU) after rendering: training manifest (synthetic + CV students + EN teacher clips, absolute paths).
  teacher_filter (CPU) after distill prep: drop items whose teacher reply is empty/refusal or misses the answer
                keyword (families with keywords) -> keep_ids.json for the trainer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

HOME = Path.home()
import os  # noqa: E402
V2 = Path(os.environ.get("V2_DIR", HOME / "mynah-s2s/ml/data/v2"))
HERE = Path(__file__).resolve().parent
LANGS = {"it": "Italian", "fr": "French", "de": "German", "es": "Spanish"}

# ------------------------------------------------------------------------------------------------ text utils
STOP = set("the a an of to in on at is are was were be what which who how many much do does did you your my me i "
           "and or for with it this that there please can could would tell about from by as".split())
ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen " \
       "seventeen eighteen nineteen".split()
TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def num_words(n: int) -> str:
    if n < 20:
        return ONES[n]
    if n < 100:
        return TENS[n // 10] + ("" if n % 10 == 0 else " " + ONES[n % 10])
    if n < 1000:
        return ONES[n // 100] + " hundred" + ("" if n % 100 == 0 else " and " + num_words(n % 100))
    if n < 10000:
        if 1100 <= n < 2100 and n % 100 != 0:  # years read as "nineteen eighty nine"
            return num_words(n // 100) + " " + (num_words(n % 100) if n % 100 >= 10 else "oh " + ONES[n % 100])
        return num_words(n // 1000) + " thousand" + ("" if n % 1000 == 0 else " " + num_words(n % 1000))
    return str(n)


def norm(s: str) -> str:
    s = "".join(c for c in unicodedata.normalize("NFKD", s.lower()) if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s)).strip()


def content(s: str) -> set:
    return {w for w in norm(s).split() if w not in STOP and len(w) > 1}


def jacc(a: set, b: set) -> float:
    return len(a & b) / max(1, len(a | b))


def kw_regexes(kws: list[str]) -> list[str]:
    """Answer keywords -> regexes on the normalised reply (lower-case, accents stripped, as e2e_qa.score).
    Numbers also match their English words (VoiceChat verbalises numbers in English)."""
    out = []
    for k in kws:
        k = norm(str(k))
        if not k:
            continue
        if k.isdigit():
            out.append(rf"\b{k}\b")
            out.append(r"\b" + re.escape(num_words(int(k))).replace(r"\ ", r"[ -]") + r"\b")
        else:
            out.append(r"\b" + re.escape(k).replace(r"\ ", r"[ -]?") + r"s?\b")
    return sorted(set(out))


REFUSAL = re.compile(r"\b(i cannot|i can ?not|i can't|i am unable|i'm unable|i am not able|i'm not able|"
                     r"i do not have access|i don't have access|i cannot provide|cannot fulfill|cannot engage|"
                     r"i'm sorry, but i|i am sorry, but i)\b")


def answer_kws(f: dict) -> list[str]:
    """Answer keywords that do not echo the EN question (a reply repeating the question must not count as correct).
    Phenomenon recall keeps all keywords: repeating the user's words is the task there."""
    if f.get("phen", f.get("cat")) == "recall":
        return list(f["kw"])
    q = norm(f["en"])
    return [k for k in f["kw"] if not re.search(r"\b" + re.escape(norm(str(k))) + r"s?\b", q)]


def is_refusal(t: str) -> bool:
    return bool(REFUSAL.search(t.lower()))


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write_new(p: Path, rows=None, obj=None):
    if p.exists():
        raise SystemExit(f"{p} exists; never overwritten")
    p.parent.mkdir(parents=True, exist_ok=True)
    if obj is not None:
        p.write_text(json.dumps(obj, indent=1, ensure_ascii=False))
    else:
        with open(p, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ------------------------------------------------------------------------------------------------ cv
CV_POOLS = {
    "it": (HOME / "itsnap/root/ft/mix/it/train_cv.json", HOME / "itsnap"),
    "fr": (HOME / "ftfr/mix/fr/train_cv.json", HOME / "ftfr/snap"),
}


def eval_texts() -> list[str]:
    """Every evaluation text we must never train on: QA v1, FLEURS/MINDS-14 eval v1 transcripts."""
    t = []
    qa = json.load(open(HERE / "qa_corpus.json"))
    for it in qa["items"]:
        t += list(it["text"].values())
    for l in open(HOME / "mynah-s2s/data/eval/v1/manifest.jsonl"):
        r = json.loads(l)
        t.append(r.get("transcript", ""))
    return [x for x in t if x]


def cmd_cv(a):
    import soundfile as sf
    rnd = random.Random(20261111)
    ev = {norm(x) for x in eval_texts()}
    out_rows = []
    for lang, (man, root) in CV_POOLS.items():
        rows = [json.loads(l) for l in open(man)]
        by_spk = defaultdict(list)
        for r in rows:
            n = len(r["text"].split())
            if 2.5 <= r["duration"] <= 8.0 and 4 <= n <= 20 and norm(r["text"]) not in ev:
                by_spk[r["speaker"]].append(r)
        spks = sorted(by_spk)
        rnd.shuffle(spks)
        need = a.per_lang_train + a.per_lang_val
        if len(spks) < need:
            print(f"WARNING {lang}: only {len(spks)} eligible speakers for {need}", flush=True)
        for k, s in enumerate(spks[:need]):
            r = rnd.choice(by_spk[s])
            src = Path(str(root) + r["audio_filepath"]) if r["audio_filepath"].startswith("/dev/shm") \
                else Path(r["audio_filepath"])
            info = sf.info(src)
            if info.samplerate != 16000 or info.channels != 1:
                print("skip (format)", src, flush=True)
                continue
            uid = Path(src).stem
            dst = V2 / "cv" / lang / f"{uid}.wav"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            out_rows.append({"id": f"cv-{lang}-{uid.split('_')[-1]}", "lang": lang, "source": "commonvoice",
                             "split": "train" if k < a.per_lang_train else "val", "speaker": s,
                             "text": r["text"], "path": str(dst), "dur": round(info.duration, 3), "sha256": sha(dst),
                             "provenance": f"owner CV pool {man} (CC0; release not recorded in the pool manifest)"})
        print(lang, sum(1 for r in out_rows if r["lang"] == lang), "clips", flush=True)
    write_new(V2 / "cv/manifest.jsonl", out_rows)


# ------------------------------------------------------------------------------------------------ minds14
MINDS_INTENTS = ["abroad", "address", "app_error", "atm_limit", "balance", "business_loan", "card_issues",
                 "cash_deposit", "direct_debit", "freeze", "high_value_payment", "joint_account",
                 "latest_transactions", "pay_bill"]
MINDS_KW = {
    "abroad": [r"abroad", r"overseas", r"travel", r"foreign", r"international", r"another country"],
    "address": [r"address", r"\bmov(e|ed|ing)\b", r"relocat"],
    "app_error": [r"\bapp\b", r"application", r"error", r"crash", r"log ?in", r"sign ?in"],
    "atm_limit": [r"\batm\b", r"limit", r"withdraw", r"cash machine"],
    "balance": [r"balance", r"how much (money )?(you have|is in)", r"funds"],
    "business_loan": [r"business", r"\bloans?\b"],
    "card_issues": [r"\bcards?\b"],
    "cash_deposit": [r"deposit", r"\bcash\b"],
    "direct_debit": [r"direct debit", r"\bdebit", r"standing order"],
    "freeze": [r"freez", r"\bblock", r"\block", r"suspend", r"stolen", r"\blost\b"],
    "high_value_payment": [r"large", r"high.?value", r"\bbig\b", r"payment", r"transfer"],
    "joint_account": [r"\bjoint\b", r"shared account", r"partner", r"spouse"],
    "latest_transactions": [r"transaction", r"recent", r"history", r"statement"],
    "pay_bill": [r"\bbills?\b", r"\bpay"],
}


def cmd_minds(a):
    import io

    import numpy as np
    import pyarrow.parquet as pq
    import soundfile as sf
    sys.path.insert(0, str(HERE))
    from render_qa import resample
    raw = HOME / "mynah-s2s/data/eval/raw/minds14"
    rnd = random.Random(20261111)
    rows, corpus = [], []
    for lang, code, n in (("it", "it-IT", a.n_it), ("fr", "fr-FR", a.n_fr), ("en", "en-US", a.n_en)):
        t = pq.read_table(raw / code / "train-00000-of-00001.parquet").to_pylist()
        by_int = defaultdict(list)
        for r in t:
            by_int[r["intent_class"]].append(r)
        pick = []
        for k in sorted(by_int):  # balanced over intents
            rnd.shuffle(by_int[k])
        while len(pick) < n and any(by_int.values()):
            for k in sorted(by_int):
                if by_int[k] and len(pick) < n:
                    pick.append(by_int[k].pop())
        for r in pick:
            x, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
            if x.ndim > 1:
                x = x.mean(1)
            y = resample(x, sr, 16000)
            y = y / max(1e-6, float(np.abs(y).max())) * 0.7
            uid = hashlib.sha1(r["path"].encode()).hexdigest()[:10]
            p = V2 / "minds14" / lang / f"m14-{lang}-{uid}.wav"
            p.parent.mkdir(parents=True, exist_ok=True)
            sf.write(p, y, 16000, subtype="PCM_16")
            intent = MINDS_INTENTS[r["intent_class"]]
            iid = f"m14-{lang}-{uid}"
            rows.append({"id": iid, "split": "minds14", "cat": intent, "lang": lang, "voice": "natural",
                         "text": r["transcription"], "english": r["english_transcription"],
                         "path": str(p.relative_to(V2 / "minds14")), "dur": round(len(y) / 16000, 3),
                         "src_sr": sr, "band_limited_8k": sr <= 8000, "sha256": sha(p),
                         "provenance": "PolyAI/minds14@40ce77cb (CC-BY-4.0), eval-only"})
            corpus.append({"id": iid, "cat": intent, "answers": MINDS_KW[intent],
                           "text": {lang: r["transcription"], "en_ref": r["english_transcription"]}})
        print(lang, len(pick), Counter(MINDS_INTENTS[r["intent_class"]] for r in pick).most_common(3), flush=True)
    write_new(V2 / "minds14/manifest.jsonl", rows)
    write_new(V2 / "minds14/corpus.json", obj={"version": "minds14-eval-v2", "items": corpus,
                                                "note": "answers = intent keyword regexes on the English reply"})


# ------------------------------------------------------------------------------------------------ gen (Nano)
PHENOMENA = {  # name: (share, instruction, has_keywords)
    "number": (0.12, "a factual question whose answer is a NUMBER (counts, sizes, quantities of everyday or "
                     "general-knowledge things)", True),
    "arithmetic": (0.10, "a simple arithmetic question with small numbers (addition, subtraction, multiplication "
                         "or division up to 100), phrased naturally, numbers written as digits", True),
    "entity": (0.20, "a factual question whose answer is a named entity (a city, country, planet, ocean, river, "
                     "language, animal, food, landmark)", True),
    "date_time": (0.10, "a question whose answer is a day, month, year, season or time span", True),
    "negation": (0.10, "a question that contains a negation or exclusion (e.g. 'which of these is NOT ...', "
                       "'I don't eat meat, which of these can I eat ...'), with a single correct answer", True),
    "recall": (0.10, "the user states a personal fact (name, favourite colour, pet name, city, age, number) "
                     "and in the same turn asks the assistant to repeat it back", True),
    "factual": (0.10, "a general-knowledge question (science, nature, everyday life) with a short answer", True),
    "request": (0.18, "a natural request or conversational turn to a voice assistant (advice, suggestion, small "
                      "talk, planning), no single correct answer", False),
}
TOPICS = ["food and cooking", "travel and places", "weather and seasons", "shopping and money", "health and sport",
          "music, films and books", "science and space", "history", "animals and nature", "home and family",
          "work and school", "geography", "technology", "transport", "holidays and festivals", "languages",
          "the human body", "sports", "art and culture", "everyday objects"]
SYS = "/no_think"


QWEN_TR = "Qwen/Qwen2.5-7B-Instruct"


class Nano:
    """Batched greedy chat generation (left padding; identical to one-by-one, results/v2/probe-nano). Default model =
    Nemotron-Nano-9B-v2 (/no_think); `path`/`system` select another local model (Qwen2.5-7B-Instruct translator)."""

    def __init__(self, batch, path=None, system=SYS):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        sys.path.insert(0, str(HERE))
        from gen_train_text import NANO
        if path is None:
            path = NANO
        elif not Path(path).exists():
            from huggingface_hub import snapshot_download
            path = snapshot_download(path, local_files_only=True)
        NANO = path
        self.system = system
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(NANO)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:  # Nano ships no pad token; id 0 is the pad used by generate()
            self.tok.pad_token = self.tok.convert_ids_to_tokens(0)
        self.m = AutoModelForCausalLM.from_pretrained(NANO, torch_dtype=torch.bfloat16, trust_remote_code=True,
                                                      device_map="cuda").eval()
        self.batch = batch
        self.rev = str(NANO)

    def run(self, prompts, max_new):
        outs = []
        torch = self.torch
        for i in range(0, len(prompts), self.batch):
            chunk = [self.tok.apply_chat_template([{"role": "system", "content": self.system}, {"role": "user", "content": p}],
                                                  add_generation_prompt=True, tokenize=False)
                     for p in prompts[i:i + self.batch]]
            b = self.tok(chunk, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            with torch.no_grad():
                # pad with the tokenizer's own pad (Nano: id 0, unchanged). A hard-coded id 0 is '!' for Qwen2.5 and
                # leaked as '!!!!' tails into 1229 of the 1375 V2 CV->EN translations (night run 2026-11-11)
                g = self.m.generate(**b, max_new_tokens=max_new, do_sample=False, pad_token_id=self.tok.pad_token_id,
                                    eos_token_id=self.tok.eos_token_id)
            outs += [self.tok.decode(x[b["input_ids"].shape[1]:], skip_special_tokens=True).strip() for x in g]
            print(f"  llm batch {i // self.batch + 1}/{-(-len(prompts) // self.batch)} | {outs[-1][:80]!r}", flush=True)
        return outs


def parse_json(s, kind):
    m = re.search(r"\[.*\]" if kind == "list" else r"\{.*\}", s, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:  # noqa: BLE001
        return None


def cmd_gen(a):
    t0 = time.monotonic()
    # Generation by Qwen2.5-7B (standard KV cache). Nano-9B through HF remote code generates WITHOUT a cache
    # ("NemotronH requires an initialized NemotronHHybridDynamicCache"), i.e. quadratic recompute: batch 16 x 1100
    # tokens exhausted unified memory (mem_guard kill, job 160 first attempt). Nano only validates short answers.
    import torch
    gen = Nano(a.batch, path=QWEN_TR, system="You write realistic test items for a multilingual voice assistant.")
    total = a.families
    cells = []
    for ph, (share, instr, haskw) in PHENOMENA.items():
        n_ph = int(round(total * share * a.overgen))
        per = max(1, -(-n_ph // len(TOPICS)))
        for tp in TOPICS:
            cells.append((ph, tp, per, instr, haskw))
    prompts = []
    for ph, tp, per, instr, haskw in cells:
        spec = ('{"q": <the user\'s spoken question, 5 to 18 words, English>, "a": <short correct answer>, '
                '"kw": [<1 to 3 short lowercase English keywords or digits that any correct answer must contain>]}'
                if haskw else '{"q": <the user\'s spoken turn, 5 to 18 words, English>}')
        prompts.append(f"Write {per} different items for a voice-assistant test set. Each item is {instr}, about "
                       f"the topic '{tp}'. Vary wording and sentence structure; do not reuse templates; avoid "
                       f"ambiguous or opinion-dependent answers. Answer ONLY with a JSON list of objects {spec}.")
    raw = gen.run(prompts, a.max_new_gen)
    gen_rev = gen.rev
    del gen
    torch.cuda.empty_cache()
    fams, seen = [], []
    evq = [content(x) for x in eval_texts()]
    for (ph, tp, per, instr, haskw), out in zip(cells, raw):
        lst = parse_json(out, "list") or []
        for d in lst[:per]:
            if not isinstance(d, dict) or not isinstance(d.get("q"), str):
                continue
            q = d["q"].strip()
            c = content(q)
            if not (4 <= len(q.split()) <= 24) or not c:
                continue
            if any(jacc(c, s) >= 0.6 for s in seen) or any(jacc(c, s) >= 0.6 for s in evq):
                continue
            kws = [str(k) for k in d.get("kw", [])][:3] if haskw else []
            if haskw and not kws:
                continue
            seen.append(c)
            fams.append({"fid": f"f{len(fams):05d}", "phen": ph, "topic": tp, "en": q,
                         "answer": str(d.get("a", "")), "kw": kws})
    print(f"generated {len(fams)} candidate families in {time.monotonic() - t0:.0f}s", flush=True)
    # validation: Nano's own brief answer must contain a keyword
    need = [f for f in fams if f["kw"]]
    nano = Nano(a.val_batch)
    ans = nano.run([f"Answer briefly in English: {f['en']}" for f in need], 40)
    for f, x in zip(need, ans):
        f["nano_answer"] = x
        f["valid"] = any(re.search(r, norm(x)) for r in kw_regexes(f["kw"]))
    for f in fams:
        f.setdefault("valid", True)
    fams = [f for f in fams if f["valid"]]
    print(f"{len(fams)} families after answer validation ({time.monotonic() - t0:.0f}s)", flush=True)
    write_new(V2 / "families_en.jsonl", fams)
    write_new(V2 / "gen_info.json", obj={"generator": gen_rev, "validator": nano.rev, "batch": a.batch,
                                          "families_target": total,
                                          "overgen": a.overgen, "phenomena": {k: v[0] for k, v in PHENOMENA.items()},
                                          "n_families_en": len(fams), "seconds": round(time.monotonic() - t0)})


def cmd_translate(a):
    """Qwen2.5-7B-Instruct (Apache-2.0) translates the EN families to it/fr/de/es and the CV transcripts to EN
    (Nano's IT/FR translations had meaning errors in the probe: 'Mi ricordo di chiamare' for 'Remind me to call')."""
    t0 = time.monotonic()
    tr = Nano(a.batch, path=QWEN_TR, system="You are a professional translator for spoken voice-assistant requests.")
    fams = [json.loads(l) for l in open(V2 / "families_en.jsonl")]
    outs = tr.run(['Translate this English sentence, spoken by a user to a voice assistant, into natural spoken '
                   'Italian, French, German and Spanish. Preserve the exact meaning, the grammatical person, '
                   'questions vs commands, names, and keep numbers as digits. Answer only with a JSON object '
                   '{"it": "...", "fr": "...", "de": "...", "es": "..."}.\n\nSentence: ' + f["en"] for f in fams],
                  a.max_new_tr)
    keep = []
    for f, out in zip(fams, outs):
        d = parse_json(out, "obj")
        if not d or not all(isinstance(d.get(l), str) and d[l].strip() for l in LANGS):
            continue
        f.update({l: d[l].strip() for l in LANGS})
        keep.append(f)
    print(f"{len(keep)}/{len(fams)} families translated ({time.monotonic() - t0:.0f}s)", flush=True)
    cv = [json.loads(l) for l in open(V2 / "cv/manifest.jsonl")]
    cvt = tr.run([f"Translate this {LANGS[r['lang']]} sentence into natural English. Answer only with the English "
                  f"sentence.\n\n{r['text']}" for r in cv], 80)
    for r, x in zip(cv, cvt):
        r["en"] = x.strip().strip('"').splitlines()[0] if x.strip() else ""
    del tr
    import torch
    torch.cuda.empty_cache()
    # independent meaning check by Nano (short YES/NO): IT and FR of every family (trained languages), and the EN
    # translation of every CV transcript. A NO drops the family / CV clip.
    judge = Nano(a.check_batch)
    q = []
    for f in keep:
        for l in ("it", "fr"):
            q.append(("fam", f, l, f"Do these two sentences have exactly the same meaning (same request, same "
                                   f"numbers, names and negations)? Answer YES or NO.\nEnglish: {f['en']}\n"
                                   f"{LANGS[l]}: {f[l]}"))
    for r in cv:
        q.append(("cv", r, r["lang"], f"Do these two sentences have exactly the same meaning? Answer YES or NO.\n"
                                      f"{LANGS[r['lang']]}: {r['text']}\nEnglish: {r['en']}"))
    outs = judge.run([x[3] for x in q], 3)
    bad_f, bad_cv = set(), set()
    for (kind, obj, l, _), o in zip(q, outs):
        if not o.strip().upper().startswith("YES"):
            (bad_f if kind == "fam" else bad_cv).add(obj["fid"] if kind == "fam" else obj["id"])
            obj.setdefault("meaning_check_failed", []).append(l)
    print(f"meaning check: {len(bad_f)} families and {len(bad_cv)} CV clips dropped", flush=True)
    dropped = [{"fid": f["fid"], "en": f["en"], "it": f["it"], "fr": f["fr"], "failed": f["meaning_check_failed"]}
               for f in keep if f["fid"] in bad_f]
    keep = [f for f in keep if f["fid"] not in bad_f]
    for r in cv:
        if r["id"] in bad_cv:
            r["en"] = ""  # build drops CV clips without a usable translation
    write_new(V2 / "families.jsonl", keep)
    write_new(V2 / "cv/translations.jsonl", [{"id": r["id"], "en": r["en"], "src": r["text"]} for r in cv])
    write_new(V2 / "translate_dropped_meaning.jsonl", dropped)
    write_new(V2 / "translate_info.json", obj={"model": QWEN_TR, "checker": judge.rev, "batch": a.batch,
                                                "n_families": len(keep), "n_dropped_meaning": len(bad_f),
                                                "n_cv_dropped_meaning": len(bad_cv), "n_cv": len(cv),
                                                "seconds": round(time.monotonic() - t0)})


# ------------------------------------------------------------------------------------------------ build
OPS = [(r"\b(plus|add(ed)?|sum of)\b", lambda x, y: x + y), (r"\b(minus|subtract(ed)?|take away|less|give away|gave away|left)\b",
                                                                   lambda x, y: x - y),
       (r"\b(times|multipl(y|ied)|product of|x)\b", lambda x, y: x * y),
       (r"\b(divided|divide|split|over|shared)\b", lambda x, y: x / y if y else None)]


def det_check(f):
    """Deterministic consistency checks (None = OK, else the reason). arithmetic: two numbers + one operator in the
    question -> computed result must be among the keywords. recall: a keyword must literally occur in the question."""
    q = norm(f["en"])
    kw = [norm(k) for k in f["kw"]]
    if f["phen"] == "arithmetic":
        nums = [int(x) for x in re.findall(r"\b\d+\b", q)]
        ops = [fn for pat, fn in OPS if re.search(pat, q)]
        if len(nums) == 2 and len(ops) == 1:
            v = ops[0](*nums)
            if v is None or v != int(v) or (str(int(v)) not in kw and num_words(int(v)) not in kw):
                return f"arithmetic {nums} -> {v} not in {kw}"
        elif len(nums) != 2:
            return f"arithmetic not checkable ({len(nums)} numbers)"
    if f["phen"] == "recall" and not any(k and k in q for k in kw):
        return "recall keyword not stated in the question"
    return None


def cmd_build(a):
    rnd = random.Random(20261111)
    fams = [json.loads(l) for l in open(V2 / "families.jsonl")]
    det_bad = []
    for f in fams:
        r = det_check(f)
        if r:
            det_bad.append({"fid": f["fid"], "phen": f["phen"], "en": f["en"], "kw": f["kw"], "reason": r})
    fams = [f for f in fams if f["fid"] not in {x["fid"] for x in det_bad}]
    by_ph = defaultdict(list)
    for f in fams:
        by_ph[f["phen"]].append(f)
    kw_ph = [p for p, v in PHENOMENA.items() if v[2]]
    split = {}
    # eval families only from phenomena with checkable answers, stratified by phenomenon
    for ph, lst in by_ph.items():
        rnd.shuffle(lst)
        if ph in kw_ph:
            nt = round(a.test * len(lst) / sum(len(by_ph[p]) for p in kw_ph))
            nd = round(a.dev * len(lst) / sum(len(by_ph[p]) for p in kw_ph))
        else:
            nt = nd = 0
        nv = round(a.val * len(lst) / len(fams))
        for i, f in enumerate(lst):
            split[f["fid"]] = "test" if i < nt else "dev" if i < nt + nd else "val" if i < nt + nd + nv else "train"
    # cap train families
    tr = [f for f in fams if split[f["fid"]] == "train"]
    if len(tr) > a.max_train:
        rnd.shuffle(tr)
        for f in tr[a.max_train:]:
            split[f["fid"]] = "unused"
    # leakage: any train/val family too close to an eval family (any language) -> moved to unused
    ev = [f for f in fams if split[f["fid"]] in ("test", "dev")]
    evc = [(f, {l: content(f[l]) for l in ["en", *LANGS]}) for f in ev]
    moved = []
    for f in fams:
        if split[f["fid"]] not in ("train", "val"):
            continue
        cf = {l: content(f[l]) for l in ["en", *LANGS]}
        for g, cg in evc:
            if any(jacc(cf[l], cg[l]) >= 0.5 for l in cf):
                moved.append((f["fid"], g["fid"], f["en"], g["en"]))
                split[f["fid"]] = "unused"
                break
    cv = {json.loads(l)["id"]: json.loads(l) for l in open(V2 / "cv/manifest.jsonl")}
    cvt = {json.loads(l)["id"]: json.loads(l)["en"] for l in open(V2 / "cv/translations.jsonl")}
    cv_bad = [i for i, e in cvt.items() if not e or len(e.split()) < 3]
    for f in fams:
        f["split"] = split[f["fid"]]
    # QA v2 corpus (dev/test) in e2e_qa format
    items = [{"id": f["fid"], "cat": f["phen"], "split": "dev" if f["split"] == "dev" else "heldout",
              "text": {"en": f["en"], **{l: f[l] for l in LANGS}}, "answers": kw_regexes(f["kw"]),
              "answer_text": f["answer"], "kw": f["kw"]} for f in fams if f["split"] in ("dev", "test")]
    write_new(V2 / "qa_v2.json", obj={"version": "qa-v2", "items": items})
    voices = a.voices.split(",")
    plan_eval, plan_train = [], []
    for it in items:
        for l in ("en", "it", "fr", "de", "es"):
            plan_eval.append({"id": it["id"], "split": it["split"], "cat": it["cat"], "lang": l, "voice": a.eval_voice,
                              "text": it["text"][l]})
    for k, f in enumerate(sorted((f for f in fams if f["split"] in ("train", "val")), key=lambda x: x["fid"])):
        v0, v1 = voices[k % len(voices)], voices[(k + 1) % len(voices)]
        for l, v in (("en", v0), ("it", v0), ("fr", v1)):
            plan_train.append({"id": f["fid"], "split": f["split"], "cat": f["phen"], "lang": l, "voice": v,
                               "text": f[l]})
    for k, (i, r) in enumerate(sorted(cv.items())):
        if i in cv_bad:
            continue
        plan_train.append({"id": i, "split": r["split"], "cat": "cv", "lang": "en", "voice": voices[k % len(voices)],
                           "text": cvt[i], "teacher_only": True})
    write_new(V2 / "plan_eval.jsonl", plan_eval)
    write_new(V2 / "plan_train.jsonl", plan_train)
    rep = {"families": len(fams), "splits": Counter(f["split"] for f in fams),
           "eval_by_phen": {s: Counter(f["phen"] for f in fams if f["split"] == s) for s in ("dev", "test")},
           "train_by_phen": Counter(f["phen"] for f in fams if f["split"] == "train"),
           "leakage_moved_to_unused": len(moved), "leakage_examples": moved[:20],
           "deterministic_check_dropped": len(det_bad), "deterministic_check_examples": det_bad[:30],
           "cv_clips": Counter((r["lang"], r["split"]) for r in cv.values()).most_common(),
           "cv_dropped_bad_translation": cv_bad,
           "render_eval_clips": len(plan_eval), "render_train_clips": len(plan_train),
           "voices_train": voices, "voice_eval": a.eval_voice}
    rep["splits"] = dict(rep["splits"])
    rep["cv_clips"] = [[list(k), v] for k, v in rep["cv_clips"]]
    write_new(V2 / "build_report.json", obj=rep)
    write_new(V2 / "families_split.jsonl", fams)
    print(json.dumps({k: v for k, v in rep.items() if k != "leakage_examples"}, indent=1, default=str))


# ------------------------------------------------------------------------------------------------ assemble
def cmd_assemble(a):
    """Training manifest for distill_adapter prep: one row per (id, lang) with absolute paths. EN rows are the teacher
    inputs (and EN rehearsal students for synthetic families); CV students are the natural IT/FR clips."""
    rend = [json.loads(l) for l in open(V2 / "tts_train/manifest.jsonl")]
    cv = [json.loads(l) for l in open(V2 / "cv/manifest.jsonl")]
    rows = []
    for r in rend:
        rows.append({**r, "path": str(V2 / "tts_train" / r["path"]), "source": "qwen3tts"})
    have_en = {r["id"] for r in rend if r["lang"] == "en"}
    for r in cv:
        if r["id"] in have_en:
            rows.append({"id": r["id"], "split": r["split"], "cat": "cv", "lang": r["lang"], "voice": r["speaker"][:16],
                         "text": r["text"], "path": r["path"], "dur": r["dur"], "sha256": r["sha256"],
                         "source": "commonvoice"})
    out = V2 / "train"
    write_new(out / "manifest.jsonl", rows)
    c = Counter((r["source"], r["lang"], r.get("teacher_only", False)) for r in rows)
    print(json.dumps([[list(k), v] for k, v in c.most_common()], indent=1))


# ------------------------------------------------------------------------------------------------ eval review
def cmd_fix_eval(a):
    """Apply the echo rule + the manual review to the eval set. Originals are kept as *.orig (never overwritten)."""
    rev = json.load(open(a.review))
    qa_p, plan_p = V2 / "qa_v2.json", V2 / "plan_eval.jsonl"
    for p in (qa_p, plan_p):
        o = p.with_name(p.name + ".orig")
        if o.exists():
            raise SystemExit(f"{o} exists: fix-eval already applied")
        o.write_bytes(p.read_bytes())
    qa = json.load(open(qa_p))
    kept, excl, log = [], [], Counter()
    for it in qa["items"]:
        if "kw" not in it:  # MINDS-14 items: intent regexes, untouched
            kept.append(it)
            continue
        f = {"en": it["text"]["en"], "kw": it["kw"], "cat": it["cat"]}
        if it["id"] in rev["exclude"]:
            excl.append({**it, "excluded": rev["exclude"][it["id"]]})
            log["exclude_manual"] += 1
            continue
        kws = rev["fix"].get(it["id"]) or answer_kws(f)
        log["fix_manual" if it["id"] in rev["fix"] else ("echo_removed" if kws != it["kw"] else "unchanged")] += 1
        if not kws:
            excl.append({**it, "excluded": "no answer keyword left after removing echoes"})
            log["exclude_no_answer"] += 1
            continue
        at = ", ".join(kws) if it["id"] in rev["fix"] else it["answer_text"]  # the judge reads answer_text
        kept.append({**it, "kw_orig": it["kw"], "kw": kws, "answers": kw_regexes(kws), "answer_text": at})
    ids = {it["id"] for it in kept}
    qa["items"] = kept
    qa["eval_review"] = {"review": a.review, "sha256": sha(Path(a.review)), "log": dict(log), "n_excluded": len(excl)}
    qa_p.write_text(json.dumps(qa, indent=1, ensure_ascii=False))
    (V2 / "qa_v2_excluded.json").write_text(json.dumps(excl, indent=1, ensure_ascii=False))
    plan = [json.loads(l) for l in open(plan_p.with_name(plan_p.name + ".orig"))]
    plan = [r for r in plan if r["id"] in ids]
    plan_p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in plan))
    by = Counter((it.get("split"), it["cat"]) for it in kept)
    print(json.dumps({"log": dict(log), "kept_items": len(kept), "excluded": len(excl), "plan_rows": len(plan),
                      "kept_by_split_cat": {f"{k[0]}|{k[1]}": v for k, v in sorted(by.items(), key=str)}}, indent=1))


def cmd_fix_translations(a):
    """Replace reviewed IT/FR eval texts in qa_v2.json and plan_eval.jsonl (pre-fix copies kept as *.pre_tr)."""
    tr = json.load(open(a.review))
    qa_p, plan_p = V2 / "qa_v2.json", V2 / "plan_eval.jsonl"
    for p in (qa_p, plan_p):
        o = p.with_name(p.name + ".pre_tr")
        if o.exists():
            raise SystemExit(f"{o} exists: fix_translations already applied")
        o.write_bytes(p.read_bytes())
    qa = json.load(open(qa_p))
    n = Counter()
    ids = {it["id"] for it in qa["items"]}
    for lang in ("it", "fr"):
        miss = set(tr[lang]) - ids
        if miss:
            raise SystemExit(f"{lang}: ids not in the eval set {sorted(miss)}")
    for it in qa["items"]:
        if "kw" not in it:
            continue
        for lang in ("it", "fr"):
            if it["id"] in tr[lang]:
                it.setdefault("text_mt", {})[lang] = it["text"][lang]
                it["text"][lang] = tr[lang][it["id"]]
                n[lang] += 1
            it.setdefault("text_reviewed", ["it", "fr"])
    qa["translation_review"] = {"review": a.review, "sha256": sha(Path(a.review)), "replaced": dict(n),
                                "reviewed_langs": ["it", "fr"], "not_reviewed": ["de", "es"]}
    qa_p.write_text(json.dumps(qa, indent=1, ensure_ascii=False))
    rows = [json.loads(l) for l in open(plan_p)]
    m = 0
    for r in rows:
        if r["lang"] in ("it", "fr") and r["id"] in tr[r["lang"]]:
            r["text"] = tr[r["lang"]][r["id"]]
            m += 1
    plan_p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(json.dumps({"qa_replaced": dict(n), "plan_rows_replaced": m}))


# ------------------------------------------------------------------------------------------------ teacher filter
def cmd_teacher_filter(a):
    prep = json.loads((Path(a.prep) / "teacher.json").read_text())["items"]
    fams = {json.loads(l)["fid"]: json.loads(l) for l in open(V2 / "families_split.jsonl")}
    cvt = {json.loads(l)["id"]: json.loads(l)["en"] for l in open(V2 / "cv/translations.jsonl")}
    keep, drop = [], []
    for i, it in prep.items():
        t = it["teacher_text"]
        reason = None
        if len(norm(t).split()) < 2:
            reason = "empty"
        elif is_refusal(t):
            reason = "refusal"
        elif i in fams and answer_kws(fams[i]) and not any(re.search(r, norm(t)) for r in kw_regexes(answer_kws(fams[i]))):
            reason = "teacher_misses_answer"
        elif i in cvt and not (content(t) & content(cvt[i])):
            reason = "cv_teacher_not_about_the_sentence"
        (drop if reason else keep).append(i if not reason else {"id": i, "reason": reason, "teacher": t[:160]})
    write_new(Path(a.out), obj={"keep": sorted(keep), "drop": drop,
                                "drop_reasons": dict(Counter(d["reason"] for d in drop))})
    print(len(keep), "kept;", dict(Counter(d["reason"] for d in drop)))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("cv")
    p.add_argument("--per-lang-train", type=int, default=700)
    p.add_argument("--per-lang-val", type=int, default=60)
    p = sp.add_parser("minds")
    p.add_argument("--n-it", type=int, default=42)
    p.add_argument("--n-fr", type=int, default=42)
    p.add_argument("--n-en", type=int, default=28)
    p = sp.add_parser("gen")
    p.add_argument("--families", type=int, default=1100)
    p.add_argument("--overgen", type=float, default=1.35)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--max-new-gen", type=int, default=1100)
    p.add_argument("--val-batch", type=int, default=8)
    p = sp.add_parser("translate")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--max-new-tr", type=int, default=260)
    p.add_argument("--check-batch", type=int, default=16)
    p = sp.add_parser("build")
    p.add_argument("--test", type=int, default=100)
    p.add_argument("--dev", type=int, default=40)
    p.add_argument("--val", type=int, default=80)
    p.add_argument("--max-train", type=int, default=800)
    p.add_argument("--voices", default="ryan,vivian")
    p.add_argument("--eval-voice", default="ryan")
    sp.add_parser("assemble")
    p = sp.add_parser("fix_eval")
    p.add_argument("--review", required=True)
    p = sp.add_parser("fix_translations")
    p.add_argument("--review", required=True)
    p = sp.add_parser("teacher_filter")
    p.add_argument("--prep", required=True)
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    {"cv": cmd_cv, "minds": cmd_minds, "gen": cmd_gen, "translate": cmd_translate, "build": cmd_build,
     "assemble": cmd_assemble,
     "fix_eval": cmd_fix_eval, "fix_translations": cmd_fix_translations, "teacher_filter": cmd_teacher_filter}[a.cmd](a)


if __name__ == "__main__":
    main()
