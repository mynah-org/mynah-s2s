"""Corpus v2 SUPPLEMENT (CPU, PROPOSAL; owner 2026-11-12): targeted families for the intents/attributes the full plan
covers too thinly to call them learned (transfer 4/1000, out_of_scope 8/1000, two-seat 1, three-seat 1).

Tracked SEPARATELY from the frozen corpus: ids `s4x-NNNN`, own seed, own plan file; `session_corpus_v2.build()` and the
pilot / full plan are not touched (no change in RNG consumption, splits or G-LEX-dev/test). The canonical target rules are
the ones of session_corpus_v2 (canonical_query, escalate_to_human with EMPTY arguments, no call for out-of-scope).
  transfer      8 TRAIN phrasings (+ optional greeting / filler)       -> escalate_to_human {}
  out_of_scope 10 TRAIN phrasings, none naming furniture               -> no call (reply must decline / state scope)
  capacity      two-seat / three-seat on TRAIN concepts whose capacity is natural (bench, garden bench, sofa bed),
                only (attribute, concept) pairs NOT held out by heldout_combo; never sofa/chair (agentbench) nor
                armchair/deck chair (G-LEX held out)                   -> search_catalog {query}
Intent-generalisation probe (split `idev`, NEVER trained): held-out PHRASINGS of transfer (2) and out_of_scope (3), so
"transfer works" is measured on wordings the student never saw, with its own n (not a statistical validation of the
intent by the 4 pilot families).
Usage: session_corpus_v2_supp.py --out supp.jsonl --stats supp_stats.json [--n-transfer 60 --n-oos 60 --n-cap 60]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lexicon_v2 as L  # noqa: E402
import session_corpus_v2 as C  # noqa: E402

LANGS = C.LANGS
TRANSFER_TRAIN = [
    ("Can I talk to a real person, please?", "Posso parlare con una persona, per favore?",
     "Je peux parler à une vraie personne, s'il vous plaît ?"),
    ("I'd like to speak to a human agent.", "Vorrei parlare con un operatore.", "Je voudrais parler à un conseiller."),
    ("Please put me through to someone in customer service.", "Mi passi qualcuno del servizio clienti, per favore.",
     "Passez-moi quelqu'un du service client, s'il vous plaît."),
    ("I don't want to talk to a machine.", "Non voglio parlare con una macchina.", "Je ne veux pas parler à une machine."),
    ("Is there an operator I can speak with?", "C'è un operatore con cui posso parlare?",
     "Il y a un opérateur avec qui je peux parler ?"),
    ("I want to make a complaint to a manager.", "Voglio fare un reclamo con un responsabile.",
     "Je veux faire une réclamation auprès d'un responsable."),
    ("Transfer me to a colleague, please.", "Mi trasferisca a un collega, per favore.",
     "Transférez-moi à un collègue, s'il vous plaît."),
    ("Could I speak to somebody from your team?", "Potrei parlare con qualcuno del vostro staff?",
     "Est-ce que je pourrais parler à quelqu'un de votre équipe ?"),
]
TRANSFER_HELDOUT = [
    ("Get me a human, please.", "Mi faccia parlare con un essere umano.", "Je veux parler à un être humain."),
    ("I need to talk to an actual person about this.", "Ho bisogno di parlarne con una persona vera.",
     "J'ai besoin d'en parler avec une vraie personne."),
]
OOS_TRAIN = [
    ("Can you book me a train ticket?", "Mi prenota un biglietto del treno?", "Vous pouvez me réserver un billet de train ?"),
    ("What time does the pharmacy close?", "A che ora chiude la farmacia?", "À quelle heure ferme la pharmacie ?"),
    ("What's the weather like tomorrow?", "Che tempo fa domani?", "Quel temps fera-t-il demain ?"),
    ("Can you call me a taxi?", "Mi chiama un taxi?", "Vous pouvez m'appeler un taxi ?"),
    ("Can you book me a restaurant for tonight?", "Mi prenota un ristorante per stasera?",
     "Vous pouvez me réserver un restaurant pour ce soir ?"),
    ("What's my bank balance?", "Qual è il saldo del mio conto?", "Quel est le solde de mon compte ?"),
    ("Can you make me a doctor's appointment?", "Mi fissa un appuntamento dal medico?",
     "Vous pouvez me prendre un rendez-vous chez le médecin ?"),
    ("Who won the football match yesterday?", "Chi ha vinto la partita ieri?", "Qui a gagné le match hier ?"),
    ("Can you top up my phone credit?", "Mi ricarica il credito del telefono?", "Vous pouvez recharger mon forfait téléphonique ?"),
    ("What films are on at the cinema tonight?", "Che film danno al cinema stasera?", "Quels films passent au cinéma ce soir ?"),
]
OOS_HELDOUT = [
    ("Can you order me a pizza?", "Mi ordina una pizza?", "Vous pouvez me commander une pizza ?"),
    ("What time is the next bus to the station?", "A che ora passa il prossimo autobus per la stazione?",
     "À quelle heure passe le prochain bus pour la gare ?"),
    ("Can you renew my passport?", "Mi rinnova il passaporto?", "Vous pouvez renouveler mon passeport ?"),
]
CAP_CONCEPTS = ("bench", "garden bench", "sofa bed")
CAP_ATTRS = ("two-seat", "three-seat")
# synonyms whose grammatical gender differs from the base noun (card §15.3). Since the generator fix, noun_phrase() agrees
# with the SPOKEN noun (L.SYN_GENDER), so the supplement uses them with correct agreement; the set is kept for the audit.
GENDER_MISMATCH_SYN = {"it": {"lettino", "comò", "cassapanca", "plaid"},
                       "fr": {"banquette", "table d'appoint", "chevet", "malle", "plaid", "lustre", "glace",
                              "chaise longue"}}
FURNITURE_WORDS = {w for c in L.CONCEPTS for w in [c[0], c[2][0], c[4][0]] + c[3] + c[5]}


def cap_pairs() -> list[tuple]:
    cs = C.split_concepts()
    out = []
    for cn in CAP_CONCEPTS:
        c = next(x for x in L.CONCEPTS if x[0] == cn)
        for an in CAP_ATTRS:
            a = next(x for x in L.ATTRS if x[0] == an)
            if cs[cn] == "train" and not C.heldout_combo(a, c):
                out.append((a, c))
    return out


def _extra_attrs(rng, c, cap) -> list[tuple]:
    """0 or 1 further attribute (colour or material), applicable to the concept and never a held-out combination."""
    pool = [a for a in L.ATTRS if a[1] in (1, 2) and C.applicable(a, c) and not C.heldout_combo(a, c)]
    k = rng.choice([0, 1, 1])
    return sorted([cap] + rng.sample(pool, k), key=lambda a: a[1])


def build_supp(n_tr: int, n_oos: int, n_cap: int, n_idev: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    out, seen = [], set()

    def add(intent, phrases, target, slots, split):
        text = {l: " ".join(p[i] for p in phrases) for i, l in enumerate(LANGS)}
        if text["en"] in seen:
            return False
        seen.add(text["en"])
        if split == "train" and rng.random() < 0.1:
            split = "val"
        out.append({"id": f"s4x-{len(out):04d}", "intent": intent, "text": text, "n_phrases": len(phrases),
                    "target": target, "slots": {"concept": None, "category": None, "attrs": [], "synonym":
                                                {l: None for l in LANGS}, "name": None, "postcode": None,
                                                "postcode_form": None, "heldout_combo": False, "concept_split": None,
                                                "probe_type": None, **slots},
                    "voice": rng.choice(["ryan", "vivian"]), "split": split, "supplement": True})
        return True

    def frame(core):
        ph = [core]
        if rng.random() < 0.5:
            ph = [rng.choice(C.GREET)] + ph
        if rng.random() < 0.3:
            ph = ph + [rng.choice(C.FILL)]
        return ph

    for intent, pool, n, target, split in (
            ("transfer", TRANSFER_TRAIN, n_tr, {"name": "escalate_to_human", "arguments": {}}, "train"),
            ("out_of_scope", OOS_TRAIN, n_oos, None, "train"),
            ("transfer", TRANSFER_HELDOUT, n_idev // 2, {"name": "escalate_to_human", "arguments": {}}, "idev"),
            ("out_of_scope", OOS_HELDOUT, n_idev - n_idev // 2, None, "idev")):
        k, tries = 0, 0
        while k < n and tries < 5000:
            tries += 1
            core = pool[k % len(pool)] if tries <= len(pool) else rng.choice(pool)   # every phrasing at least once
            if add(intent, frame(core), target, {"phrasing": pool.index(core)}, split):
                k += 1
    pairs = cap_pairs()
    k, tries = 0, 0
    while k < n_cap and tries < 5000:
        tries += 1
        cap, c = pairs[k % len(pairs)]
        attrs = _extra_attrs(rng, c, cap)
        syn = {l: None for l in LANGS}
        for l, s_ in (("it", c[3]), ("fr", c[5])):
            if s_ and rng.random() < 0.3:
                syn[l] = rng.choice(s_)
        tpl = rng.choice([t for t in C.CAT if not (c[1] == "outdoor" and "living room" in t[0])])
        ph = ([rng.choice(C.GREET)] if rng.random() < 0.4 else []) + [tpl]
        text_ph = []
        for p in ph:
            text_ph.append(tuple(C.fill(p[i], l, C.noun_phrase(c, attrs, l, syn[l]), "", "")
                                 for i, l in enumerate(LANGS)))
        if add("catalog_capacity", text_ph,
               {"name": "search_catalog", "arguments": {"query": C.canonical_query(c, attrs)}},
               {"concept": c[0], "category": c[1], "attrs": [a[0] for a in attrs], "synonym": syn,
                "concept_split": "train"}, "train"):
            k += 1
    return out


def audit(items: list[dict]) -> dict:
    """Derivability, isolation from G-LEX / agentbench, and no furniture word in out-of-scope turns."""
    cs = C.split_concepts()
    bad, leaks, oos_furn = {}, [], []
    for it in items:
        if it["intent"] == "catalog_capacity":
            b = C.derivable({**it, "slots": {**it["slots"]}})
            for l in LANGS:                                            # capacity completeness (outside `applicable`)
                tx = it["text"][l].lower()
                for an in CAP_ATTRS:
                    a = next(x for x in L.ATTRS if x[0] == an)
                    forms = [a[0]] if l == "en" else list(a[2] if l == "it" else a[3])
                    if an not in it["slots"]["attrs"] and any(f.lower() in tx for f in forms):
                        b.append(f"{l}:capacity_said_not_in_query:{an}")
            if b:
                bad[it["id"]] = b
            c = it["slots"]["concept"]
            if cs[c] != "train" or any(C.heldout_combo(next(x for x in L.ATTRS if x[0] == an),
                                                       next(x for x in L.CONCEPTS if x[0] == c))
                                       for an in it["slots"]["attrs"]):
                leaks.append(it["id"])
        if it["intent"] == "out_of_scope":
            for l in LANGS:
                tx = " " + it["text"][l].lower() + " "
                hit = [w for w in FURNITURE_WORDS if re.search(rf"\b{re.escape(w.lower())}\b", tx)]
                if hit:
                    oos_furn.append((it["id"], l, hit))
    def cnt(sel):
        return dict(Counter(f"{i['intent']}|{i['split']}" for i in sel))
    return {"families": len(items), "by_intent_split": cnt(items),
            "phrasings": {f"{i}|{s}": len({it['slots'].get('phrasing') for it in items if it['intent'] == i and
                                           it['split'] == s}) for i, s in {(it['intent'], it['split']) for it in items
                                                                           if it['intent'] != 'catalog_capacity'}},
            "capacity_pairs": dict(Counter(f"{[a for a in i['slots']['attrs'] if a in CAP_ATTRS][0]}|{i['slots']['concept']}"
                                           for i in items if i["intent"] == "catalog_capacity")),
            "capacity_queries": sorted({i["target"]["arguments"]["query"] for i in items if i["intent"] == "catalog_capacity"}),
            "not_derivable": bad, "leaks": leaks, "oos_with_furniture_word": oos_furn,
            "sha256_texts": hashlib.sha256(json.dumps([i["text"] for i in items], ensure_ascii=False).encode()).hexdigest()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-transfer", type=int, default=60)
    ap.add_argument("--n-oos", type=int, default=60)
    ap.add_argument("--n-cap", type=int, default=60)
    ap.add_argument("--n-idev", type=int, default=24)
    ap.add_argument("--seed", type=int, default=20261114)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stats", required=True)
    a = ap.parse_args()
    for p in (a.out, a.stats):
        if Path(p).exists():
            raise SystemExit(f"{p} exists")
    items = build_supp(a.n_transfer, a.n_oos, a.n_cap, a.n_idev, a.seed)
    rep = audit(items)
    with open(a.out, "w") as f:
        for it in items:
            for l in LANGS:
                f.write(json.dumps({"id": it["id"], "split": it["split"], "cat": f"s4:{it['intent']}", "lang": l,
                                    "voice": it["voice"], "text": it["text"][l], "target": it["target"],
                                    "slots": it["slots"], "n_phrases": it["n_phrases"], "supplement": True},
                                   ensure_ascii=False) + "\n")
    Path(a.stats).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in rep.items() if k != "capacity_queries"}, ensure_ascii=False)[:2500])
    print("queries:", rep["capacity_queries"])


if __name__ == "__main__":
    main()
