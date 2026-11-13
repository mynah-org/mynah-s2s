"""B2 — text-only multilingual capability: official Nemotron-Nano-9B-v2 vs the VoiceChat Thinker weights.

Model B ("vc-text") = Nano-9B-v2 architecture/config/tokenizer with every backbone tensor, the input embeddings and
the LM head replaced by VoiceChat's `stt_model.llm.*`, `stt_model.embed_tokens`, `stt_model.lm_head` (cast FP32 -> BF16).
WHAT IS BYPASSED for B (documented, not hidden): the duplex frame loop (one input embedding per 80 ms frame =
agent-text emb + user-audio emb + function emb), the system-prompt-as-user-channel prefill, the function head, the
PAD/BOS/EOS turn protocol and delays. B is driven exactly like A: Nano chat template, standard autoregressive text.
So a drop on B mixes (i) lost language/knowledge with (ii) template/inference-path mismatch (VoiceChat never saw the
chat template during S2S training). To separate them, every item is also scored with lenient criteria and the raw
outputs are kept for inspection; the "think" mode is off (`/no_think`) for both.

Greedy decoding, max 200 new tokens, same prompts for both models. Scores per language/category -> JSON.
"""
from __future__ import annotations

import argparse
import json
import re
import time
import unicodedata
from pathlib import Path

import torch

NANO = Path.home() / "mynah-s2s/hf-cache/models--nvidia--NVIDIA-Nemotron-Nano-9B-v2/snapshots/6533e8de2c68e4536bf7c411d7a3ce5734111476"
VC = Path.home() / "mynah-s2s/hf-cache/models--nvidia--NVIDIA-NemotronLabs-VoiceChat-11B/snapshots/443794ea956ef0065f001967ffd00e77f519cb39/model.safetensors"

TOOLS = [
    {"type": "function", "function": {"name": "get_order_status", "description": "Look up the status of an order by its ID.",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string", "description": "Order ID, e.g. AB1234"}}, "required": ["order_id"]}}},
    {"type": "function", "function": {"name": "get_weather", "description": "Get the weather forecast for a city.",
     "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "day": {"type": "string", "description": "today or tomorrow"}}, "required": ["city"]}}},
]

L = ["en", "it", "fr", "es", "de"]
T = {  # per item: text per language
    "qa": {"en": "What is the capital of Australia? Answer with one word.",
           "it": "Qual è la capitale dell'Australia? Rispondi con una sola parola.",
           "fr": "Quelle est la capitale de l'Australie ? Réponds en un seul mot.",
           "es": "¿Cuál es la capital de Australia? Responde con una sola palabra.",
           "de": "Was ist die Hauptstadt von Australien? Antworte mit einem Wort."},
    "comprehension": {"en": "Marco has three sisters and one brother. How many siblings does Marco have? Answer with a number only.",
                      "it": "Marco ha tre sorelle e un fratello. Quanti fratelli e sorelle ha Marco in tutto? Rispondi solo con un numero.",
                      "fr": "Marco a trois sœurs et un frère. Combien de frères et sœurs Marco a-t-il en tout ? Réponds uniquement par un nombre.",
                      "es": "Marco tiene tres hermanas y un hermano. ¿Cuántos hermanos tiene Marco en total? Responde solo con un número.",
                      "de": "Marco hat drei Schwestern und einen Bruder. Wie viele Geschwister hat Marco insgesamt? Antworte nur mit einer Zahl."},
    "reasoning": {"en": "A train leaves at 14:20 and the trip takes 1 hour and 55 minutes. At what time does it arrive? Answer in HH:MM format only.",
                  "it": "Un treno parte alle 14:20 e il viaggio dura 1 ora e 55 minuti. A che ora arriva? Rispondi solo nel formato HH:MM.",
                  "fr": "Un train part à 14:20 et le trajet dure 1 heure et 55 minutes. À quelle heure arrive-t-il ? Réponds uniquement au format HH:MM.",
                  "es": "Un tren sale a las 14:20 y el viaje dura 1 hora y 55 minutos. ¿A qué hora llega? Responde solo en formato HH:MM.",
                  "de": "Ein Zug fährt um 14:20 ab und die Fahrt dauert 1 Stunde und 55 Minuten. Wann kommt er an? Antworte nur im Format HH:MM."},
    "instruction": {"en": "Reply with the single word OK in capital letters and nothing else.",
                    "it": "Rispondi con la sola parola OK in maiuscolo e nient'altro.",
                    "fr": "Réponds par le seul mot OK en majuscules et rien d'autre.",
                    "es": "Responde solo con la palabra OK en mayúsculas y nada más.",
                    "de": "Antworte nur mit dem Wort OK in Großbuchstaben und sonst nichts."},
    "json": {"en": "Extract the data from this sentence: 'Giulia is 34 years old and lives in Turin.' Return only a JSON object with the keys name, age, city.",
             "it": "Estrai i dati da questa frase: 'Giulia ha 34 anni e vive a Torino.' Restituisci solo un oggetto JSON con le chiavi name, age, city.",
             "fr": "Extrais les données de cette phrase : 'Giulia a 34 ans et habite à Turin.' Renvoie uniquement un objet JSON avec les clés name, age, city.",
             "es": "Extrae los datos de esta frase: 'Giulia tiene 34 años y vive en Turín.' Devuelve solo un objeto JSON con las claves name, age, city.",
             "de": "Extrahiere die Daten aus diesem Satz: 'Giulia ist 34 Jahre alt und wohnt in Turin.' Gib nur ein JSON-Objekt mit den Schlüsseln name, age, city zurück."},
    "tool_order": {"en": "Hi, can you tell me where my order XYZ123 is?",
                   "it": "Ciao, mi puoi dire dov'è il mio ordine XYZ123?",
                   "fr": "Bonjour, pouvez-vous me dire où en est ma commande XYZ123 ?",
                   "es": "Hola, ¿me puedes decir dónde está mi pedido XYZ123?",
                   "de": "Hallo, kannst du mir sagen, wo meine Bestellung XYZ123 ist?"},
    "tool_weather": {"en": "What will the weather be like in Milan tomorrow?",
                     "it": "Che tempo farà domani a Milano?",
                     "fr": "Quel temps fera-t-il demain à Milan ?",
                     "es": "¿Qué tiempo hará mañana en Milán?",
                     "de": "Wie wird das Wetter morgen in Mailand?"},
    "tool_result": {"en": "Where is my order XYZ123?", "it": "Dov'è il mio ordine XYZ123?", "fr": "Où est ma commande XYZ123 ?",
                    "es": "¿Dónde está mi pedido XYZ123?", "de": "Wo ist meine Bestellung XYZ123?"},
    "multiturn": {"en": ["My favourite colour is green.", "Got it.", "What is my favourite colour? Answer with one word."],
                  "it": ["Il mio colore preferito è il verde.", "Ricevuto.", "Qual è il mio colore preferito? Rispondi con una parola."],
                  "fr": ["Ma couleur préférée est le vert.", "C'est noté.", "Quelle est ma couleur préférée ? Réponds en un mot."],
                  "es": ["Mi color favorito es el verde.", "Entendido.", "¿Cuál es mi color favorito? Responde con una palabra."],
                  "de": ["Meine Lieblingsfarbe ist Grün.", "Verstanden.", "Was ist meine Lieblingsfarbe? Antworte mit einem Wort."]},
}
GREEN = r"green|verde|vert|grun"
SHIPPED = r"shipped|spedit|expedi|envi|versandt|verschickt|unterwegs"
# function-word sets for a crude reply-language check
STOP = {"en": {"the", "is", "and", "your", "it", "you", "was"}, "it": {"il", "che", "è", "di", "ordine", "stato", "suo", "tuo"},
        "fr": {"le", "la", "est", "votre", "commande", "été", "et"}, "es": {"el", "su", "está", "pedido", "ha", "sido", "tu"},
        "de": {"die", "ist", "ihre", "bestellung", "wurde", "und", "deine"}}


def norm(s):
    return "".join(c for c in unicodedata.normalize("NFKD", s.lower()) if not unicodedata.combining(c))


def parse_toolcall(s):
    m = re.search(r"<TOOLCALL>(.*?)</TOOLCALL>", s, re.S)
    if not m:
        return None
    try:
        v = json.loads(m.group(1))
        return v if isinstance(v, list) else [v]
    except Exception:  # noqa: BLE001
        return "unparsable"


def build_items():
    items = []
    for l in L:
        for cat in ("qa", "comprehension", "reasoning", "instruction", "json", "tool_order", "tool_weather", "tool_result", "multiturn"):
            t = T[cat][l]
            if cat == "multiturn":
                msgs = [{"role": "user", "content": t[0]}, {"role": "assistant", "content": t[1]}, {"role": "user", "content": t[2]}]
            elif cat == "tool_result":
                call = [{"name": "get_order_status", "arguments": {"order_id": "XYZ123"}}]
                msgs = [{"role": "user", "content": t},
                        {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": "get_order_status", "arguments": {"order_id": "XYZ123"}}}]},
                        {"role": "tool", "content": json.dumps({"order_id": "XYZ123", "status": "shipped", "estimated_delivery": "2026-11-12"})}]
            else:
                msgs = [{"role": "user", "content": t}]
            items.append({"lang": l, "cat": cat, "messages": msgs, "tools": cat.startswith("tool")})
    return items


def score(it, out):
    o = out.strip()
    n = norm(o)
    c = it["cat"]
    if c == "qa":
        return bool(re.search("canberra", n))
    if c == "comprehension":
        return bool(re.search(r"\b4\b|four|quattro|quatre|cuatro|vier", n))
    if c == "reasoning":
        return "16:15" in o
    if c == "instruction":
        return re.fullmatch(r"\W*OK\W*", o) is not None
    if c == "json":
        m = re.search(r"\{.*\}", o, re.S)
        try:
            d = json.loads(m.group(0))
            return norm(str(d.get("name"))) == "giulia" and str(d.get("age")) == "34" and norm(str(d.get("city"))) in ("turin", "torino", "turin")
        except Exception:  # noqa: BLE001
            return False
    if c == "tool_order":
        tc = parse_toolcall(o)
        return isinstance(tc, list) and any(x.get("name") == "get_order_status" and str(x.get("arguments", {}).get("order_id", "")).upper() == "XYZ123" for x in tc)
    if c == "tool_weather":
        tc = parse_toolcall(o)
        return isinstance(tc, list) and any(x.get("name") == "get_weather" and norm(str(x.get("arguments", {}).get("city", ""))) in ("milan", "milano", "mailand", "milan") for x in tc)
    if c == "tool_result":
        return bool(re.search(SHIPPED, n)) and bool(re.search(r"12|twelfth|dodici|douze|doce|zwolf", n))
    if c == "multiturn":
        return bool(re.search(GREEN, n))
    return False


def reply_lang(out):
    w = set(re.findall(r"[a-zàèéìòùçäöüßñ]+", out.lower()))
    s = {l: len(w & v) for l, v in STOP.items()}
    best = max(s, key=s.get)
    return best if s[best] > 0 else "?"


def load(which):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(NANO)
    m = AutoModelForCausalLM.from_pretrained(NANO, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="cuda")
    info = {"model": which, "nano": str(NANO)}
    if which in ("vc", "vcsup"):
        from safetensors import safe_open
        sd = m.state_dict()
        n = 0
        with safe_open(str(VC), framework="pt") as f, torch.no_grad():
            for k in f.keys():
                if k.startswith("stt_model.llm."):
                    t = "backbone." + k[len("stt_model.llm."):]
                elif k == "stt_model.embed_tokens.weight":
                    t = "backbone.embeddings.weight"
                elif k == "stt_model.lm_head.weight":
                    t = "lm_head.weight"
                else:
                    continue
                sd[t].copy_(f.get_tensor(k).to(sd[t].dtype))
                n += 1
        info["replaced_tensors"] = n
        if n != len(sd):
            raise RuntimeError(f"replaced {n} of {len(sd)} tensors")
    m.eval()
    return tok, m, info


@torch.no_grad()
def run(which, items, out_dir):
    tok, m, info = load(which)
    rows = []
    for it in items:
        msgs = [{"role": "system", "content": "/no_think"}] + it["messages"]
        ids = tok.apply_chat_template(msgs, tools=TOOLS if it["tools"] else None, add_generation_prompt=True, return_tensors="pt").to("cuda")
        t0 = time.monotonic()
        kw = {}
        if which == "vcsup":  # VoiceChat repurposed id 12 (<SPECIAL_12>, Nano's end-of-turn) as its frame PAD
            kw = {"bad_words_ids": [[12]], "eos_token_id": [11, 2]}
        else:
            kw = {"eos_token_id": tok.eos_token_id}
        g = m.generate(ids, max_new_tokens=200, do_sample=False, pad_token_id=0, **kw)
        out = tok.decode(g[0, ids.shape[1]:], skip_special_tokens=False)
        out = re.split(r"<SPECIAL_1[12]>|</s>|<extra_id_1>", out)[0].strip()
        out_vis = re.sub(r"<think>.*?</think>", "", out, flags=re.S).strip()
        r = {"model": which, "lang": it["lang"], "cat": it["cat"], "ok": score(it, out_vis), "reply_lang": reply_lang(out_vis),
             "out": out_vis[:600], "s": round(time.monotonic() - t0, 2)}
        rows.append(r)
        with open(out_dir / f"rows_{which}.jsonl", "a") as fh:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[{which}] {it['lang']} {it['cat']:14s} {'OK' if r['ok'] else 'NO'} {r['reply_lang']} | {out_vis[:140]!r}", flush=True)
    del m
    torch.cuda.empty_cache()
    return rows, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default="nano,vc,vcsup")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    items = build_items()
    allrows, infos = [], {}
    if a.models != "summarize":
        for w in a.models.split(","):
            (out / f"rows_{w}.jsonl").unlink(missing_ok=True)
            rows, info = run(w, items, out)
            (out / f"info_{w}.json").write_text(json.dumps(info))
        return
    for p in sorted(out.glob("rows_*.jsonl")):
        allrows += [json.loads(l) for l in open(p)]
        w = p.stem[5:]
        infos[w] = json.loads((out / f"info_{w}.json").read_text())
    summ = {}
    for r in allrows:
        s = summ.setdefault(r["model"], {}).setdefault(r["lang"], {"ok": 0, "n": 0})
        s["ok"] += r["ok"]
        s["n"] += 1
    bycat = {}
    for r in allrows:
        s = bycat.setdefault(r["model"], {}).setdefault(r["cat"], {"ok": 0, "n": 0})
        s["ok"] += r["ok"]
        s["n"] += 1
    (out / "rows.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in allrows) + "\n")
    (out / "summary.json").write_text(json.dumps({"by_lang": summ, "by_cat": bycat, "models": infos}, indent=1))
    print(json.dumps({"by_lang": summ, "by_cat": bycat}, indent=1))


if __name__ == "__main__":
    main()
