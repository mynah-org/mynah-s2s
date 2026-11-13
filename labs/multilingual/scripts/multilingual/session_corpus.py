"""D-V3 training corpus: parallel EN/IT/FR caller turns for the furniture tenant, disjoint from the eval.

Each family is one caller turn in three languages (same content), built from 1-3 phrases so that spoken durations
spread over roughly 2-8 s (no single turn length to learn as an end-of-turn cue). Intents mirror the tenant's
tools (order status with/without credentials, catalog, return, transfer, out of scope, short greetings), but
templates and entities are DISJOINT from the agentbench eval:
  - no fixture/scenario names (Mario Rossi, Giulia Bianchi, Marco Rizzi/Ricci, Siobhan Okonkwo-Laurent, Anna Verdi),
  - no fixture/scenario postcodes,
  - no eval products or their keywords (sofa/couch/divano/canape, bookcase/bookshelf/shelf/libreria, wardrobe/closet/
    armadio/armoire, lamp/light/lampada/lampe, desk, chair),
  - no scenario sentence is reused (checked by `check_disjoint`).
Usage: session_corpus.py build --out plan.jsonl [--n 300 --seed 20261112]
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

FIRST = ["Luca", "Elena", "Paolo", "Chiara", "Sophie", "Julien", "Thomas", "Emma", "Laura", "Davide", "Sara",
         "Nicolas", "Claire", "Matteo", "Francesca", "Hugo", "Peter", "Hannah", "Omar", "Ines", "Karim", "Alice"]
LAST = ["Ferrari", "Romano", "Colombo", "Moreau", "Lefevre", "Martin", "Bernard", "Esposito", "Greco", "Conti",
        "Fontaine", "Weber", "Novak", "Marchetti", "Gallo", "Dubois", "Petit", "Mancini", "Lambert", "Costa"]
# masculine singular, consonant-initial in IT ("il/un/mio") and FR ("le/un/mon"); EN takes "a"
PRODUCTS = [("rug", "tappeto", "tapis"), ("mattress", "materasso", "matelas"), ("TV stand", "mobile TV", "meuble TV"),
            ("footstool", "pouf", "pouf"), ("shoe cabinet", "portascarpe", "meuble à chaussures"),
            ("coffee table", "tavolino", "guéridon"), ("chest of drawers", "comò", "chiffonnier"),
            ("nightstand", "comodino", "chevet"), ("bed base", "telaio del letto", "sommier")]
DIG = {"en": "zero one two three four five six seven eight nine".split(),
       "it": "zero uno due tre quattro cinque sei sette otto nove".split(),
       "fr": "zéro un deux trois quatre cinq six sept huit neuf".split()}
LANGS = ("en", "it", "fr")

EXCL_NAMES = {"mario rossi", "giulia bianchi", "marco rizzi", "marco ricci", "siobhan okonkwo-laurent", "anna verdi"}
EXCL_PC = {"10121", "20135", "00184", "75011", "10125", "20121", "50123", "99999"}
EXCL_WORDS = {"sofa", "couch", "divano", "canape", "canapé", "bookcase", "bookshelf", "shelf", "libreria",
              "bibliotheque", "bibliothèque", "wardrobe", "closet", "armadio", "armoire", "lamp", "light", "lampada",
              "lampe", "desk", "chair", "sedia", "chaise", "fjordby", "lysa", "holm", "brisk", "tarn", "vika",
              "operator", "operatore", "opérateur", "restaurant", "ristorante", "neighbour", "vicina", "voisine"}

# phrase pools: (en, it, fr). Slots: {name} {pc} {p} (product, per language)
GREET = [("Hi there.", "Ciao.", "Bonjour."), ("Good afternoon.", "Buon pomeriggio.", "Bon après-midi."),
         ("Hello, good morning.", "Salve, buongiorno.", "Bonjour, bonne journée."),
         ("Hi, sorry to bother you.", "Salve, scusi il disturbo.", "Bonjour, excusez-moi de vous déranger.")]
ID = [("My name is {name}, and my postcode is {pc}.", "Mi chiamo {name}, e il mio CAP è {pc}.",
       "Je m'appelle {name}, et mon code postal est {pc}."),
      ("This is {name}. Postcode {pc}.", "Sono {name}. CAP {pc}.", "Ici {name}. Code postal {pc}."),
      ("I'm {name}, the postcode is {pc}.", "Sono {name}, il CAP è {pc}.", "Je suis {name}, le code postal est {pc}.")]
NAME_ONLY = [("This is {name} calling.", "Sono {name}.", "C'est {name} à l'appareil."),
             ("My name is {name}.", "Mi chiamo {name}.", "Je m'appelle {name}.")]
STATUS = [("I'm calling to check on a delivery.", "Chiamo per controllare una consegna.",
           "J'appelle pour vérifier une livraison."),
          ("Could you tell me when my {p} will be delivered?", "Mi può dire quando verrà consegnato il mio {p}?",
           "Pouvez-vous me dire quand mon {p} sera livré ?"),
          ("I ordered a {p} and it still hasn't arrived.", "Ho ordinato un {p} e non è ancora arrivato.",
           "J'ai commandé un {p} et il n'est toujours pas arrivé.")]
NO_ID = [("I bought something from you a while ago, and I have no idea where it is now.",
          "Ho comprato qualcosa da voi tempo fa, e non ho idea di dove sia adesso.",
          "J'ai acheté quelque chose chez vous il y a un moment, et je ne sais pas où il en est."),
         ("I can't find my confirmation email anywhere.", "Non trovo da nessuna parte la mail di conferma.",
          "Je ne trouve nulle part mon e-mail de confirmation.")]
CATALOG = [("Do you have a {p} in stock?", "Avete un {p} disponibile?", "Avez-vous un {p} en stock ?"),
           ("I'm furnishing a new flat and I need a {p}.", "Sto arredando un appartamento nuovo e mi serve un {p}.",
            "J'aménage un nouvel appartement et j'ai besoin d'un {p}."),
           ("How much does your {p} cost?", "Quanto costa il vostro {p}?", "Combien coûte votre {p} ?")]
RETURN = [("I'd like to send back the {p} I received.", "Vorrei rendere il {p} che ho ricevuto.",
           "Je voudrais renvoyer le {p} que j'ai reçu."),
          ("The {p} arrived damaged and I want to return it.", "Il {p} è arrivato rovinato e voglio restituirlo.",
           "Le {p} est arrivé abîmé et je veux le retourner.")]
TRANSFER = [("Could you put me through to someone from customer service?",
             "Mi può passare qualcuno del servizio clienti?",
             "Pouvez-vous me passer quelqu'un du service client ?")]
OUT = [("Can you call me a taxi to the station?", "Mi può chiamare un taxi per la stazione?",
        "Pouvez-vous m'appeler un taxi pour la gare ?"),
       ("What's the weather going to be like tomorrow?", "Che tempo farà domani?", "Quel temps fera-t-il demain ?")]
FILL = [("Let me think for a second.", "Mi faccia pensare un attimo.", "Laissez-moi réfléchir une seconde."),
        ("It's a bit urgent, actually.", "In realtà è un po' urgente.", "C'est un peu urgent, en fait."),
        ("I hope you can help me.", "Spero che mi possa aiutare.", "J'espère que vous pouvez m'aider.")]
SHORT = [("Hello? Can you hear me?", "Pronto? Mi sente?", "Allô ? Vous m'entendez ?"),
         ("Hi, is this the furniture store?", "Salve, parlo con il negozio di mobili?", "Bonjour, c'est bien le magasin de meubles ?"),
         ("Yes, hello. I have a question.", "Sì, salve. Ho una domanda.", "Oui, bonjour. J'ai une question.")]

INTENTS = {  # intent -> list of phrase-slot sequences (optional greeting / filler added at build time)
    "status_id": [[ID, STATUS], [STATUS, ID]],
    "status_name_only": [[NAME_ONLY, STATUS]],
    "status_no_id": [[STATUS, NO_ID], [NO_ID]],
    "catalog": [[CATALOG], [CATALOG, CATALOG]],
    "return_id": [[RETURN, ID], [ID, RETURN]],
    "return_no_id": [[RETURN]],
    "transfer": [[TRANSFER]],
    "out_of_scope": [[OUT]],
    "short": [[SHORT]],
}
WEIGHTS = {"status_id": 6, "status_name_only": 2, "status_no_id": 2, "catalog": 2, "return_id": 6, "return_no_id": 2,
           "transfer": 1, "out_of_scope": 1, "short": 2}
BUCKETS = [(20, 50), (50, 75), (75, 100), (100, 125)]   # EN chars ~ 15/s -> about 1.5-3.3, 3.3-5, 5-6.7, 6.7-8.3 s


def spoken(pc: str, lang: str) -> str:
    return " ".join(DIG[lang][int(c)] for c in pc)


def fill(t: str, lang: str, ent: dict) -> str:
    li = LANGS.index(lang)
    return t.format(name=ent["name"], pc=spoken(ent["pc"], lang), p=ent["product"][li] if ent.get("product") else "")


def build(n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    names = [f"{f} {l}" for f in FIRST for l in LAST if f"{f} {l}".lower() not in EXCL_NAMES]
    keys = [k for k, w in WEIGHTS.items() for _ in range(w)]
    out, seen, fill_b = [], set(), [0] * len(BUCKETS)
    per_b = -(-n // len(BUCKETS))
    tries = 0
    while len(out) < n and tries < 200000:
        tries += 1
        intent = rng.choice(keys)
        seq = rng.choice(INTENTS[intent])
        ent = {"name": rng.choice(names), "pc": None, "product": rng.choice(PRODUCTS)}
        while ent["pc"] is None or ent["pc"] in EXCL_PC:
            ent["pc"] = "".join(rng.choice("0123456789") for _ in range(5))
        phrases = []
        for pool in seq:                       # no phrase repeated inside one turn
            cand = [x for x in pool if x not in phrases]
            phrases.append(rng.choice(cand or pool))
        if intent != "short" and rng.random() < 0.5:
            phrases = [rng.choice(GREET)] + phrases
        if intent not in ("short",) and rng.random() < 0.35:
            phrases.insert(rng.randrange(1, len(phrases) + 1), rng.choice(FILL))
        text = {lang: " ".join(fill(p[LANGS.index(lang)], lang, ent) for p in phrases) for lang in LANGS}
        if text["en"] in seen:
            continue
        b = next((k for k, (lo, hi) in enumerate(BUCKETS) if lo <= len(text["en"]) < hi), None)
        if b is None or fill_b[b] >= per_b:      # flat length distribution (no learnable turn length)
            continue
        fill_b[b] += 1
        seen.add(text["en"])
        uses = {"name": any("{name}" in p[0] for p in phrases), "pc": any("{pc}" in p[0] for p in phrases),
                "product": any("{p}" in p[0] for p in phrases)}
        out.append({"id": f"s3-{len(out):04d}", "intent": intent, "text": text, "n_phrases": len(phrases),
                    "entities": {"name": ent["name"] if uses["name"] else None,
                                 "postcode": ent["pc"] if uses["pc"] else None,
                                 "product": ent["product"][0] if uses["product"] else None},
                    "voice": rng.choice(["ryan", "vivian"]), "split": "val" if rng.random() < 0.1 else "train"})
    return out


def check_disjoint(items: list[dict], scenarios: list[dict]) -> list[str]:
    """Violations: excluded names/postcodes/product keywords, or any eval scenario sentence, inside a training text."""
    bad = []
    scen = {t.get(l, "").lower() for s in scenarios for t in s.get("turns", []) for l in LANGS if t.get(l)}
    for it in items:
        for lang, tx in it["text"].items():
            low = tx.lower()
            words = set(re.findall(r"[a-zà-ü'-]+", low))
            hit = [w for w in EXCL_WORDS if w in words] + [nm for nm in EXCL_NAMES if nm in low]
            if it["entities"]["postcode"] in EXCL_PC:
                hit.append(it["entities"]["postcode"])
            hit += [s for s in scen if s and s in low]
            if hit:
                bad.append(f"{it['id']}|{lang}: {hit}")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20261112)
    ap.add_argument("--tenant", default=str(Path(__file__).resolve().parent.parent / "agentbench/tenant_furniture.json"))
    a = ap.parse_args()
    items = build(a.n, a.seed)
    bad = check_disjoint(items, json.load(open(a.tenant))["scenarios"])
    if bad:
        sys.exit("disjointness violations:\n" + "\n".join(bad[:20]))
    with open(a.out, "w") as f:
        for it in items:
            for lang in LANGS:   # render_qa --plan rows: one clip per (family, language)
                f.write(json.dumps({"id": it["id"], "split": it["split"], "cat": f"s3:{it['intent']}", "lang": lang,
                                    "voice": it["voice"], "text": it["text"][lang], "entities": it["entities"],
                                    "n_phrases": it["n_phrases"]}, ensure_ascii=False) + "\n")
    from collections import Counter
    print(len(items), "families", dict(Counter(i["intent"] for i in items)), dict(Counter(i["split"] for i in items)))


if __name__ == "__main__":
    main()
