"""Parallel training utterances for interface adapters (never the QA eval items).

Step 1: Nemotron-Nano-9B-v2 (/no_think, greedy) writes short voice-assistant user turns (questions, requests,
statements) in English per topic; step 2: the same model translates each into it/fr/de/es. Items whose text overlaps
the QA corpus (same normalised question, or containing a dev/held-out answer entity in a question about the same
fact) are dropped. Output: JSONL {id, topic, en, it, fr, de, es}.
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

import torch

NANO = Path.home() / "mynah-s2s/hf-cache/models--nvidia--NVIDIA-Nemotron-Nano-9B-v2/snapshots/6533e8de2c68e4536bf7c411d7a3ce5734111476"
TOPICS = ["cooking and recipes", "travel and directions", "weather and seasons", "shopping and orders", "health and sport",
          "music and films", "science for kids", "history", "animals and nature", "numbers and arithmetic",
          "personal facts the user tells about themselves (name, age, city, job, pets) followed by a question about them",
          "home and appliances", "work and schedules", "geography of countries and rivers", "technology and phones",
          "banking and payments", "restaurants and booking", "transport and trains", "school and language learning",
          "small talk and feelings"]
LANGS = {"it": "Italian", "fr": "French", "de": "German", "es": "Spanish"}


def norm(s):
    s = "".join(c for c in unicodedata.normalize("NFKD", s.lower()) if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9 ]", "", s).strip()


@torch.no_grad()
def chat(tok, m, user, max_new=900):
    msgs = [{"role": "system", "content": "/no_think"}, {"role": "user", "content": user}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to("cuda")
    g = m.generate(ids, max_new_tokens=max_new, do_sample=False, eos_token_id=tok.eos_token_id, pad_token_id=0)
    return tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--per-topic", type=int, default=20)
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(NANO)
    m = AutoModelForCausalLM.from_pretrained(NANO, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="cuda").eval()
    evalq = {norm(it["text"]["en"]) for it in json.load(open(a.corpus))["items"]}
    seen, items = set(), []
    for ti, topic in enumerate(TOPICS):
        txt = chat(tok, m, f"Write {a.per_topic} different things a user might say to a voice assistant about {topic}. "
                           "Each one is a single short spoken sentence or two (4 to 18 words), natural and varied: "
                           "questions, requests, and statements. One per line, no numbering, no quotes.")
        for line in txt.splitlines():
            s = re.sub(r"^[\-\*\d\.\)\s]+", "", line).strip().strip('"')
            n = norm(s)
            if not (3 <= len(n.split()) <= 25) or n in seen or n in evalq:
                continue
            seen.add(n)
            items.append({"id": f"t{len(items):04d}", "topic": topic, "en": s})
        print(topic, len(items), flush=True)
    keep = []
    for it in items:
        out = chat(tok, m, "Translate the sentence into Italian, French, German and Spanish. Answer only with a JSON "
                           'object {"it": ..., "fr": ..., "de": ..., "es": ...}.\n\nSentence: ' + it["en"], 300)
        try:
            d = json.loads(re.search(r"\{.*\}", out, re.S).group(0))
            it.update({l: str(d[l]).strip() for l in LANGS})
            keep.append(it)
            print(it["id"], it["en"], "|", it["it"], flush=True)
        except Exception:  # noqa: BLE001
            print("DROP (bad JSON)", it["id"], out[:120], flush=True)
    items = keep
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print("wrote", len(items))


if __name__ == "__main__":
    main()
