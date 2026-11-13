"""D-V3 corpus v2 (CPU): compositional EN/IT/FR caller turns with rule-derived, verifiable function targets.

Card: .work/multilingual/dv3-corpus-v2-card.md. Lexicon and canonicalisation rules: lexicon_v2.py.
Every family = one caller turn in EN/IT/FR built from paraphrase templates x noun phrases (concept + attributes, with IT/FR
gender agreement) x names x postcodes (digit-by-digit or grouped). The FIRST-CALL TARGET is derived by rule from what
the caller says (never from the teacher's free text); the teacher only provides timing and replies, and its tool choice
must agree with the rule (otherwise the family is dropped as ambiguous). Splits are by concept, combination and name.
Usage: session_corpus_v2.py build --n 150 --out plan.jsonl [--probe-out probe.jsonl] [--stats stats.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lexicon_v2 as L  # noqa: E402

LANGS = ("en", "it", "fr")
VOWEL = tuple("aeiouàèéìòùhâêîôûœ")


# ----------------------------------------------------------------------------------------------- noun phrases
def _h(s: str) -> int:
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)


def it_ind(gender: str, nxt: str) -> str:
    n = nxt.lower()
    if gender == "m":
        return "uno" if re.match(r"(s[^aeiou]|z|gn|ps|x|y)", n) else "un"
    return "un'" if n.startswith(VOWEL[:5]) else "una"


def it_def(gender: str, nxt: str) -> str:
    n = nxt.lower()
    if n.startswith(VOWEL[:5]):
        return "l'"
    if gender == "m":
        return "lo" if re.match(r"(s[^aeiou]|z|gn|ps|x|y)", n) else "il"
    return "la"


def fr_ind(gender: str) -> str:
    return "un" if gender == "m" else "une"


def fr_def(gender: str, nxt: str) -> str:
    return "l'" if nxt.lower().startswith(VOWEL) else ("le" if gender == "m" else "la")


def fr_poss(gender: str, nxt: str) -> str:
    return "mon" if gender == "m" or nxt.lower().startswith(VOWEL) else "ma"


def join_art(art: str, rest: str) -> str:
    return art + rest if art.endswith("'") else f"{art} {rest}"


def noun_phrase(concept: tuple, attrs: list[tuple], lang: str, surface: str | None = None,
                legacy_gender: bool = False) -> dict:
    """Bare NP (no article) + gender + forms with articles; attrs sorted in canonical order.
    Gender = the SPOKEN noun's (synonym gender from L.SYN_GENDER). legacy_gender=True reproduces the frozen v4 plans,
    which used the base noun's gender for synonyms too (card §15.3: 40 IT/FR clips excluded as grammar_defect)."""
    canon, cat, it_n, it_syn, fr_n, fr_syn = concept
    if lang == "en":
        pre = [a[0] for a in attrs]
        bare = " ".join(pre + [canon])
        art = "an" if bare[0].lower() in "aeiou" else "a"
        return {"bare": bare, "ind": f"{art} {bare}", "def": f"the {bare}", "poss": f"my {bare}", "gender": "m"}
    noun, g = (surface or (it_n if lang == "it" else fr_n)[0]), (it_n if lang == "it" else fr_n)[1]
    if surface and not legacy_gender:
        g = L.SYN_GENDER[lang][surface]
    idx = 0 if g == "m" else 1
    prenom = [a for a in attrs if a[0] in (L.IT_PRENOMINAL if lang == "it" else L.FR_PRENOMINAL)]
    post = [a for a in attrs if a not in prenom]
    post = sorted(post, key=lambda a: {1: 0, 0: 1, 2: 2}[a[1]])     # colour, size/seats, material after the noun
    form = lambda a: (a[2] if lang == "it" else a[3])[idx]          # noqa: E731
    words = [form(a) for a in prenom] + [noun] + [form(a) for a in post]
    bare = " ".join(words)
    if lang == "it":
        return {"bare": bare, "ind": join_art(it_ind(g, bare), bare), "def": join_art(it_def(g, bare), bare),
                "poss": f"{'il mio' if g == 'm' else 'la mia'} {bare}", "gender": g}
    return {"bare": bare, "ind": f"{fr_ind(g)} {bare}", "def": join_art(fr_def(g, bare), bare),
            "poss": f"{fr_poss(g, bare)} {bare}", "gender": g}


def canonical_query(concept: tuple, attrs: list[tuple]) -> str:
    """The catalog query target: the attributes the caller SAID (canonical EN, fixed order) + the canonical noun."""
    return " ".join([a[0] for a in sorted(attrs, key=lambda a: a[1])] + [concept[0]])


# ----------------------------------------------------------------------------------------------- templates
# (en, it, fr); slots {ind} {def} {poss} {name} {pc}; agreement slots {e_it}/{e_fr}: feminine endings
CAT = [("I'm looking for {ind}.", "Cerco {ind}.", "Je cherche {ind}."),
       ("Do you sell {ind}?", "Vendete {ind}?", "Est-ce que vous vendez {ind} ?"),
       ("Have you got {ind} in stock?", "Avete {ind} disponibile?", "Avez-vous {ind} en stock ?"),
       ("I need {ind} for the living room.", "Mi serve {ind} per il soggiorno.", "Il me faudrait {ind} pour le salon."),
       ("How much does {def} cost?", "Quanto costa {def}?", "Combien coûte {def} ?"),
       ("I saw {ind} on your website.", "Ho visto {ind} sul vostro sito.", "J'ai vu {ind} sur votre site.")]
RET = [("I'd like to return {def} I bought last week.", "Vorrei restituire {def} che ho comprato la settimana scorsa.",
        "Je voudrais retourner {def} que j'ai acheté{e_fr} la semaine dernière."),
       ("{Def} arrived damaged.", "{Def} è arrivat{o_it} rovinat{o_it}.", "{Def} est arrivé{e_fr} abîmé{e_fr}."),
       ("I want to send back {poss}, it doesn't fit.", "Voglio rimandare indietro {poss}, non ci sta.",
        "Je veux renvoyer {poss}, {il_fr} ne rentre pas.")]
ID = [("My name is {name}, and my postcode is {pc}.", "Mi chiamo {name}, e il mio CAP è {pc}.",
       "Je m'appelle {name}, et mon code postal est {pc}."),
      ("This is {name}, postcode {pc}.", "Sono {name}, CAP {pc}.", "Ici {name}, code postal {pc}."),
      ("The order is under {name}, postcode {pc}.", "L'ordine è a nome di {name}, CAP {pc}.",
       "La commande est au nom de {name}, code postal {pc}.")]
NAME = [("My name is {name}.", "Mi chiamo {name}.", "Je m'appelle {name}."),
        ("It's {name} speaking.", "Sono {name}.", "C'est {name} à l'appareil.")]
STAT = [("Where is my order?", "Dov'è il mio ordine?", "Où en est ma commande ?"),
        ("When will {poss} arrive?", "Quando arriva {poss}?", "Quand est-ce que {poss} arrive ?"),
        ("I'm calling about a delivery.", "Chiamo per una consegna.", "J'appelle pour une livraison.")]
NOID = [("I don't have the order details with me.", "Non ho con me i dati dell'ordine.",
         "Je n'ai pas les détails de la commande sous la main."),
        ("I ordered something weeks ago and nothing has arrived.", "Ho ordinato qualcosa settimane fa e non è arrivato niente.",
         "J'ai commandé quelque chose il y a des semaines et rien n'est arrivé.")]
TRANSFER = [("Can I talk to a real person, please?", "Posso parlare con una persona, per favore?",
             "Je peux parler à une vraie personne, s'il vous plaît ?")]
OUT = [("Can you book me a train ticket?", "Mi prenota un biglietto del treno?", "Vous pouvez me réserver un billet de train ?"),
       ("What time does the pharmacy close?", "A che ora chiude la farmacia?", "À quelle heure ferme la pharmacie ?")]
GREET = [("Hi there.", "Buongiorno.", "Bonjour."), ("Hello, good afternoon.", "Salve, buon pomeriggio.", "Bonsoir."),
         ("Hi, sorry to bother you.", "Salve, mi scusi.", "Bonjour, excusez-moi.")]
FILL = [("Let me think.", "Mi faccia pensare.", "Attendez, je réfléchis."),
        ("It's a bit urgent.", "È un po' urgente.", "C'est un peu urgent."),
        ("Thanks in advance.", "Grazie in anticipo.", "Merci d'avance.")]

INTENTS = {   # intent: (phrase sequences, rule tool, weight)
    "catalog": ([[CAT], [CAT, FILL]], "search_catalog", 5),
    "status_id": ([[ID, STAT], [STAT, ID]], "find_customer", 3),
    "status_name_only": ([[NAME, STAT]], "find_customer", 2),
    "return_id": ([[RET, ID], [ID, RET]], "find_customer", 3),
    "return_no_id": ([[RET]], None, 1),
    "status_no_id": ([[STAT, NOID], [NOID]], None, 1),
    "transfer": ([[TRANSFER]], "escalate_to_human", 1),
    "out_of_scope": ([[OUT]], None, 1),
}


# ----------------------------------------------------------------------------------------------- splits
def split_concepts(seed: int = 7) -> dict[str, str]:
    """EVAL_CONCEPTS -> 'agentbench' (never trained); about 20 % of the rest -> 'heldout' (stratified by category)."""
    out, by = {}, defaultdict(list)
    for c in L.CONCEPTS:
        if c[0] in L.EVAL_CONCEPTS:
            out[c[0]] = "agentbench"
        else:
            by[c[1]].append(c[0])
    for cat, names in sorted(by.items()):
        names = sorted(names, key=lambda n: _h(f"{seed}|{n}"))
        k = max(1, round(0.2 * len(names))) if len(names) >= 3 else 0
        for i, n in enumerate(names):
            out[n] = "heldout" if i < k else "train"
    return out


def heldout_part(key: str, seed: int = 7) -> str:
    """Held-out pools are split in two halves: 'dev' (repeatable G-LEX-dev probe, may guide checkpoint choice) and
    'test' (frozen G-LEX-test, final evaluation only, never used to choose checkpoints or edit the corpus)."""
    return "dev" if _h(f"part|{seed}|{key}") % 2 == 0 else "test"


def applicable(attr: tuple, concept: tuple) -> bool:
    return attr[4] is None or concept[1] in attr[4]


def heldout_combo(attr: tuple, concept: tuple, seed: int = 7) -> bool:
    return _h(f"{seed}|{attr[0]}|{concept[0]}") % 10 == 0           # ~10 % of attribute x concept pairs


def name_split(first: str, last: str) -> str:
    fi, la = L.FIRST.index(first), L.LAST.index(last)
    return "heldout" if fi >= int(0.8 * len(L.FIRST)) or la >= int(0.8 * len(L.LAST)) else "train"


# ----------------------------------------------------------------------------------------------- build
def make_postcode(rng) -> str:
    while True:
        pc = "".join(rng.choice("0123456789") for _ in range(5))
        if pc not in L.EXCL_PC:
            return pc


def fill(t: str, lang: str, np_: dict | None, name: str, pc_spoken: str, legacy: bool = False) -> str:
    """legacy=True reproduces the frozen v4 FR pronoun defect (feminine 'ille' instead of 'elle', card §17)."""
    f = {"name": name, "pc": pc_spoken}
    if np_ is not None:
        g = np_["gender"]
        f.update(ind=np_["ind"], **{"def": np_["def"]}, poss=np_["poss"],
                 Def=np_["def"][0].upper() + np_["def"][1:], e_fr="e" if g == "f" else "", o_it="a" if g == "f" else "o",
                 il_fr=("ille" if legacy else "elle") if g == "f" else "il")
    else:
        f.update(ind="", poss="", Def="", e_fr="", o_it="o", il_fr="il", **{"def": ""})
    s = t.format(**f)
    s = re.sub(r"\bde ([AEIOUÉÈ])", r"d'\1", s)          # FR: "au nom de Omar" -> "au nom d'Omar"
    s = re.sub(r"\bdi ([AEIOU])", r"di \1", s)
    return s


def build(n: int, seed: int, split: str = "train", legacy_gender: bool = False) -> list[dict]:
    """split='train': train concepts / names / non-held-out combinations only. split='dev' or 'test': the matching half
    of the held-out concepts, combinations and names (G-LEX-dev / G-LEX-test)."""
    rng = random.Random(seed)
    cs = split_concepts()
    part = None if split == "train" else split
    concepts = [c for c in L.CONCEPTS if cs[c[0]] == "train"] if split == "train" else \
        [c for c in L.CONCEPTS if cs[c[0]] == "heldout" and heldout_part(c[0]) == part]
    keys = [k for k, v in INTENTS.items() for _ in range(v[2])]
    out, seen = [], set()
    tries = 0
    while len(out) < n and tries < 100000:
        tries += 1
        ptype = None
        if split == "train":
            intent = keys[rng.randrange(len(keys))]
        else:                    # G-LEX dev/test: 1/2 new concepts, 1/4 new attribute x concept pairs, 1/4 new names+digits
            ptype = ["new_concept", "new_concept", "new_combo", "new_name_digits"][len(out) % 4]
            intent = "catalog" if ptype != "new_name_digits" else "status_id"
        seqs, tool, _ = INTENTS[intent]
        seq = [CAT] if ptype in ("new_concept", "new_combo") else rng.choice(seqs)
        if ptype == "new_combo":
            pool_c = [x for x in L.CONCEPTS if cs[x[0]] == "train"]
            c = rng.choice(pool_c)
        elif ptype == "new_name_digits":
            c = rng.choice([x for x in L.CONCEPTS if cs[x[0]] == "train"])
        else:
            c = rng.choice(concepts)
        pool = [a for a in L.ATTRS if applicable(a, c)]
        k = rng.choice([0, 0, 1, 1, 2]) if intent == "catalog" else rng.choice([0, 1])
        attrs = sorted(rng.sample(pool, min(k, len(pool))), key=lambda a: a[1])
        if len({a[1] for a in attrs}) < len(attrs):
            continue                                                 # one attribute per slot type
        if ptype == "new_combo":
            hp = [a for a in pool if heldout_combo(a, c) and heldout_part(f"{a[0]}|{c[0]}") == part]
            if not hp:
                continue
            attrs = [rng.choice(hp)]
        held = any(heldout_combo(a, c) for a in attrs)
        if split == "train" and held:
            continue
        while True:
            first, last = rng.choice(L.FIRST), rng.choice(L.LAST)
            want = "train" if split == "train" or ptype in ("new_concept", "new_combo") else "heldout"
            if f"{first} {last}".lower() not in L.EXCL_NAMES and name_split(first, last) == want and \
                    (want == "train" or heldout_part(f"{first} {last}") == part):
                break
        name = f"{first} {last}"
        pc = make_postcode(rng)
        grouped = rng.random() < (0.35 if ptype is None else 0.5)
        phrases = []
        for pool_ in seq:
            cand = [x for x in pool_ if x not in phrases]
            phrases.append(rng.choice(cand))
        if rng.random() < 0.4:
            phrases = [rng.choice(GREET)] + phrases
        syn_pick = {l: None for l in LANGS}
        for l, syn in (("it", c[3]), ("fr", c[5])):
            if syn and rng.random() < 0.3:
                syn_pick[l] = rng.choice(syn)                       # lawful synonym, same canonical concept
        text = {}
        uses_np = any(("{ind}" in p[0] or "{def}" in p[0] or "{poss}" in p[0] or "{Def}" in p[0]) for p in phrases)
        for i, l in enumerate(LANGS):
            np_ = noun_phrase(c, attrs, l, syn_pick[l], legacy_gender) if uses_np else None
            text[l] = " ".join(fill(p[i], l, np_, name, L.spoken_postcode(pc, l, grouped), legacy_gender)
                               for p in phrases)
        if text["en"] in seen:
            continue
        seen.add(text["en"])
        says_name = any("{name}" in p[0] for p in phrases)
        says_pc = any("{pc}" in p[0] for p in phrases)
        target = None
        if tool == "search_catalog" and uses_np:
            target = {"name": "search_catalog", "arguments": {"query": canonical_query(c, attrs)}}
        elif tool == "find_customer" and says_name:
            target = {"name": "find_customer", "arguments": {"full_name": name, **({"postcode": pc} if says_pc else {})}}
        elif tool == "escalate_to_human":
            # pilot finding (card §12): the optional free-text reason was invented by the teacher -> never supervised;
            # the call carries EMPTY arguments (schema: reason optional)
            target = {"name": "escalate_to_human", "arguments": {}}
        out.append({"id": f"{ {'train': 's4', 'dev': 'gd4', 'test': 'gt4'}[split] }-{len(out):04d}", "intent": intent, "text": text,
                    "n_phrases": len(phrases), "target": target,
                    "slots": {"concept": c[0] if uses_np else None, "category": c[1] if uses_np else None,
                              "attrs": [a[0] for a in attrs] if uses_np else [], "synonym": syn_pick,
                              "name": name if says_name else None, "postcode": pc if says_pc else None,
                              "postcode_form": ("grouped" if grouped else "digits") if says_pc else None,
                              "heldout_combo": held, "concept_split": cs[c[0]] if uses_np else None, "probe_type": ptype},
                    "voice": rng.choice(["ryan", "vivian"]),
                    "split": split if split != "train" else ("val" if rng.random() < 0.1 else "train")})
    return out


# ----------------------------------------------------------------------------------------------- checks
def derivable(item: dict) -> list[str]:
    """Label audit (CPU): every target slot must be recoverable from the caller's text in EVERY language through the
    lexicon (surface or synonym for the concept, attribute surface forms, the name string, the spoken postcode)."""
    bad = []
    t, s = item["target"], item["slots"]
    if not t or not t.get("arguments"):
        return bad
    for l in LANGS:
        tx = item["text"][l].lower()
        if t["name"] == "search_catalog":
            c = next(x for x in L.CONCEPTS if x[0] == s["concept"])
            forms = [c[0]] if l == "en" else ([c[2][0]] + c[3] if l == "it" else [c[4][0]] + c[5])
            if not any(f.lower() in tx for f in forms):
                bad.append(f"{l}:concept")
            for an in s["attrs"]:
                a = next(x for x in L.ATTRS if x[0] == an)
                af = [a[0]] if l == "en" else list(a[2] if l == "it" else a[3])
                if not any(f.lower() in tx for f in af):
                    bad.append(f"{l}:attr:{an}")
            said = [a for a in L.ATTRS if applicable(a, c) and a[0] not in s["attrs"] and
                    any(re.search(rf"\b{re.escape(f.lower())}\b", tx) for f in
                        ([a[0]] if l == "en" else list(a[2] if l == "it" else a[3])))]
            if said:                                    # completeness: no said attribute missing from the query
                bad.append(f"{l}:implicit_or_missing_attr:{[a[0] for a in said]}")
        if t["name"] == "find_customer":
            if "postcode" in t["arguments"] and L.parse_spoken_digits(tx, l).find(t["arguments"]["postcode"]) < 0:
                bad.append(f"{l}:postcode_roundtrip")
            if s["name"].lower() not in tx:
                bad.append(f"{l}:name")
            if "postcode" in t["arguments"]:
                sp = L.spoken_postcode(s["postcode"], l, s["postcode_form"] == "grouped").lower()
                if sp not in tx:
                    bad.append(f"{l}:postcode")
    return bad


def stats(items: list[dict]) -> dict:
    words = {l: Counter(w for it in items for w in re.findall(r"[a-zà-ü']+", it["text"][l].lower())) for l in LANGS}
    est = {l: sorted(len(it["text"][l]) / 15.0 for it in items) for l in LANGS}
    q = lambda xs, p: round(xs[int(p * (len(xs) - 1))], 1)          # noqa: E731
    return {"families": len(items), "intents": dict(Counter(i["intent"] for i in items)),
            "distinct_words": {l: len(w) for l, w in words.items()},
            "concepts": len({i["slots"]["concept"] for i in items if i["slots"]["concept"]}),
            "concept_mentions": sum(bool(i["slots"]["concept"]) for i in items),
            "with_attrs": dict(Counter(len(i["slots"]["attrs"]) for i in items if i["slots"]["concept"])),
            "synonym_used": {l: sum(bool(i["slots"]["synonym"][l]) for i in items) for l in ("it", "fr")},
            "names": len({i["slots"]["name"] for i in items if i["slots"]["name"]}),
            "postcode_forms": dict(Counter(i["slots"]["postcode_form"] for i in items if i["slots"]["postcode_form"])),
            "targets": dict(Counter((i["target"] or {}).get("name", "none") for i in items)),
            "phrases": dict(Counter(i["n_phrases"] for i in items)),
            "est_speech_s_p10_p50_p90": {l: [q(v, .1), q(v, .5), q(v, .9)] for l, v in est.items()},
            "split": dict(Counter(i["split"] for i in items))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build"])
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=20261113)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dev-n", type=int, default=12)
    ap.add_argument("--test-n", type=int, default=24)
    ap.add_argument("--dev-out", default="")
    ap.add_argument("--test-out", default="")
    ap.add_argument("--stats", default="")
    ap.add_argument("--legacy-synonym-gender", action="store_true",
                    help="reproduce the frozen v4 plans byte for byte (synonyms agreed with the base noun's gender; FR feminine "
                         "pronoun 'ille')")
    a = ap.parse_args()
    lg = a.legacy_synonym_gender
    tr = build(a.n, a.seed, "train", lg)
    dv = build(a.dev_n, a.seed + 1, "dev", lg) if a.dev_out else []
    te = build(a.test_n, a.seed + 2, "test", lg) if a.test_out else []
    bad = {i["id"]: derivable(i) for i in tr + dv + te if derivable(i)}
    rep = {"train": stats(tr), "dev": stats(dv) if dv else None, "test": stats(te) if te else None,
           "not_derivable": bad, "concept_split": dict(Counter(split_concepts().values())),
           "test_sha256": hashlib.sha256(json.dumps([i["text"] for i in te], ensure_ascii=False).encode()).hexdigest()}
    for path, items in ((a.out, tr), (a.dev_out, dv), (a.test_out, te)):
        if not path:
            continue
        with open(path, "w") as f:
            for it in items:
                for l in LANGS:
                    f.write(json.dumps({"id": it["id"], "split": it["split"], "cat": f"s4:{it['intent']}", "lang": l,
                                        "voice": it["voice"] if it["split"] in ("train", "val") or l == "en" else "ryan",
                                        "text": it["text"][l], "target": it["target"], "slots": it["slots"],
                                        "n_phrases": it["n_phrases"]}, ensure_ascii=False) + "\n")
    if a.stats:
        Path(a.stats).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    print(json.dumps(rep, ensure_ascii=False)[:3000])


if __name__ == "__main__":
    main()
