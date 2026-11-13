"""D-V3 corpus v2 lexicon: product concepts, attributes, names, number forms (EN/IT/FR), with canonical EN forms.

Each concept: canonical EN query word(s) + IT/FR surface nouns WITH grammatical gender (m/f) and synonyms that map to
the same concept. Rules for the canonical query (owner: derivable from speech + catalog, never arbitrary):
  - query = [attributes the caller SAID, canonical EN, fixed order: size/seats, colour, material] + canonical noun;
  - synonyms and register variants normalise to the canonical noun (divano / sofà -> sofa) = lawful normalisation;
  - nothing the caller did not say is added (no brand, no inferred attribute, no catalog name);
  - a concept absent from the tenant catalog keeps its canonical query (the backend answers "no results").
Split (by concept and by entity, never by clip):
  - EVAL_CONCEPTS: the agentbench products (sofa, bookcase, wardrobe, lamp, desk, chair) are NEVER in training;
  - HELDOUT: about 20 % of the other concepts are held out for the lexical generalisation probe (G-LEX);
  - held-out attribute x concept combinations (both parts seen, the pair never) test composition;
  - names: first/last-name lists split 80/20 (eval names never in training).
"""
from __future__ import annotations

import re

# canonical EN, category, (IT noun, gender), [IT synonyms], (FR noun, gender), [FR synonyms]
CONCEPTS = [
    ("sofa", "seat", ("divano", "m"), ["sofà"], ("canapé", "m"), ["sofa"]),
    ("bookcase", "storage", ("libreria", "f"), [], ("bibliothèque", "f"), []),
    ("wardrobe", "storage", ("armadio", "m"), ["guardaroba"], ("armoire", "f"), ["penderie"]),
    ("lamp", "lighting", ("lampada", "f"), [], ("lampe", "f"), []),
    ("desk", "table", ("scrivania", "f"), [], ("bureau", "m"), []),
    ("chair", "seat", ("sedia", "f"), [], ("chaise", "f"), []),
    ("armchair", "seat", ("poltrona", "f"), [], ("fauteuil", "m"), []),
    ("stool", "seat", ("sgabello", "m"), [], ("tabouret", "m"), []),
    ("bench", "seat", ("panca", "f"), ["panchina"], ("banc", "m"), ["banquette"]),
    ("footstool", "seat", ("pouf", "m"), ["poggiapiedi"], ("pouf", "m"), ["repose-pieds"]),
    ("rocking chair", "seat", ("sedia a dondolo", "f"), [], ("fauteuil à bascule", "m"), []),
    ("dining table", "table", ("tavolo da pranzo", "m"), [], ("table à manger", "f"), []),
    ("coffee table", "table", ("tavolino", "m"), ["tavolino da salotto"], ("table basse", "f"), []),
    ("side table", "table", ("tavolino laterale", "m"), [], ("bout de canapé", "m"), ["table d'appoint"]),
    ("console table", "table", ("consolle", "f"), [], ("console", "f"), []),
    ("folding table", "table", ("tavolo pieghevole", "m"), [], ("table pliante", "f"), []),
    ("bed", "bed", ("letto", "m"), [], ("lit", "m"), []),
    ("bunk bed", "bed", ("letto a castello", "m"), [], ("lit superposé", "m"), []),
    ("sofa bed", "bed", ("divano letto", "m"), [], ("canapé-lit", "m"), ["clic-clac"]),
    ("mattress", "bed", ("materasso", "m"), [], ("matelas", "m"), []),
    ("pillow", "textile", ("cuscino", "m"), ["guanciale"], ("oreiller", "m"), []),
    ("duvet", "textile", ("piumone", "m"), ["piumino"], ("couette", "f"), []),
    ("bed base", "bed", ("rete del letto", "f"), [], ("sommier", "m"), []),
    ("headboard", "bed", ("testiera", "f"), [], ("tête de lit", "f"), []),
    ("crib", "kids", ("culla", "f"), ["lettino"], ("berceau", "m"), ["lit bébé"]),
    ("high chair", "kids", ("seggiolone", "m"), [], ("chaise haute", "f"), []),
    ("changing table", "kids", ("fasciatoio", "m"), [], ("table à langer", "f"), []),
    ("toy box", "kids", ("cesta dei giochi", "f"), [], ("coffre à jouets", "m"), []),
    ("chest of drawers", "storage", ("cassettiera", "f"), ["comò"], ("commode", "f"), []),
    ("nightstand", "storage", ("comodino", "m"), [], ("table de chevet", "f"), ["chevet"]),
    ("sideboard", "storage", ("credenza", "f"), [], ("buffet", "m"), ["bahut"]),
    ("display cabinet", "storage", ("vetrina", "f"), [], ("vitrine", "f"), []),
    ("shoe cabinet", "storage", ("scarpiera", "f"), [], ("meuble à chaussures", "m"), []),
    ("TV stand", "storage", ("mobile TV", "m"), ["porta TV"], ("meuble TV", "m"), []),
    ("wall shelf", "storage", ("mensola", "f"), [], ("étagère murale", "f"), []),
    ("storage box", "storage", ("scatola portaoggetti", "f"), [], ("boîte de rangement", "f"), []),
    ("coat rack", "storage", ("appendiabiti", "m"), [], ("portemanteau", "m"), []),
    ("chest", "storage", ("baule", "m"), ["cassapanca"], ("coffre", "m"), ["malle"]),
    ("filing cabinet", "storage", ("classificatore", "m"), ["schedario"], ("classeur", "m"), []),
    ("kitchen cabinet", "kitchen", ("pensile", "m"), [], ("placard de cuisine", "m"), []),
    ("kitchen island", "kitchen", ("isola da cucina", "f"), [], ("îlot de cuisine", "m"), []),
    ("bar stool", "kitchen", ("sgabello da bar", "m"), [], ("tabouret de bar", "m"), []),
    ("spice rack", "kitchen", ("portaspezie", "m"), [], ("étagère à épices", "f"), []),
    ("dish rack", "kitchen", ("scolapiatti", "m"), [], ("égouttoir", "m"), []),
    ("trolley", "kitchen", ("carrello", "m"), [], ("desserte", "f"), []),
    ("rug", "textile", ("tappeto", "m"), [], ("tapis", "m"), []),
    ("curtain", "textile", ("tenda", "f"), [], ("rideau", "m"), []),
    ("blanket", "textile", ("coperta", "f"), ["plaid"], ("couverture", "f"), ["plaid"]),
    ("cushion", "textile", ("cuscino decorativo", "m"), [], ("coussin", "m"), []),
    ("tablecloth", "textile", ("tovaglia", "f"), [], ("nappe", "f"), []),
    ("towel", "textile", ("asciugamano", "m"), [], ("serviette de bain", "f"), []),
    ("ceiling light", "lighting", ("plafoniera", "f"), [], ("plafonnier", "m"), []),
    ("pendant light", "lighting", ("lampadario", "m"), [], ("suspension", "f"), ["lustre"]),
    ("wall light", "lighting", ("applique", "f"), [], ("applique", "f"), []),
    ("candle holder", "decor", ("portacandele", "m"), [], ("bougeoir", "m"), []),
    ("mirror", "decor", ("specchio", "m"), [], ("miroir", "m"), ["glace"]),
    ("picture frame", "decor", ("cornice", "f"), [], ("cadre photo", "m"), []),
    ("vase", "decor", ("vaso", "m"), [], ("vase", "m"), []),
    ("wall clock", "decor", ("orologio da parete", "m"), [], ("horloge murale", "f"), ["pendule"]),
    ("plant pot", "decor", ("vaso per piante", "m"), [], ("pot de fleurs", "m"), []),
    ("room divider", "decor", ("separè", "m"), ["paravento"], ("paravent", "m"), []),
    ("laundry basket", "bathroom", ("cesto della biancheria", "m"), [], ("panier à linge", "m"), []),
    ("bathroom cabinet", "bathroom", ("mobile da bagno", "m"), [], ("meuble de salle de bain", "m"), []),
    ("towel rail", "bathroom", ("portasciugamani", "m"), [], ("porte-serviettes", "m"), []),
    ("shower curtain", "bathroom", ("tenda da doccia", "f"), [], ("rideau de douche", "m"), []),
    ("garden table", "outdoor", ("tavolo da giardino", "m"), [], ("table de jardin", "f"), []),
    ("deck chair", "outdoor", ("sdraio", "f"), [], ("transat", "m"), ["chaise longue"]),
    ("parasol", "outdoor", ("ombrellone", "m"), [], ("parasol", "m"), []),
    ("hammock", "outdoor", ("amaca", "f"), [], ("hamac", "m"), []),
    ("garden bench", "outdoor", ("panchina da giardino", "f"), [], ("banc de jardin", "m"), []),
    ("office chair", "office", ("sedia da ufficio", "f"), ["poltroncina da ufficio"], ("chaise de bureau", "f"), []),
    ("monitor stand", "office", ("supporto per monitor", "m"), [], ("support d'écran", "m"), []),
    ("drawer unit", "office", ("cassettiera da ufficio", "f"), [], ("caisson de bureau", "m"), []),
]
# grammatical gender of every IT/FR synonym (owner 2026-11-12, card §15.3): articles and adjectives agree with the noun
# actually SPOKEN, not with the concept's base noun ("une banquette bleue", not "un banquette bleu")
SYN_GENDER = {
    "it": {"sofà": "m", "guardaroba": "m", "panchina": "f", "poggiapiedi": "m", "tavolino da salotto": "m",
           "guanciale": "m", "piumino": "m", "lettino": "m", "comò": "m", "porta TV": "m", "cassapanca": "f",
           "schedario": "m", "plaid": "m", "paravento": "m", "poltroncina da ufficio": "f"},
    "fr": {"sofa": "m", "penderie": "f", "banquette": "f", "repose-pieds": "m", "table d'appoint": "f", "clic-clac": "m",
           "lit bébé": "m", "chevet": "m", "bahut": "m", "malle": "f", "plaid": "m", "lustre": "m", "glace": "f",
           "pendule": "f", "chaise longue": "f"},
}
EVAL_CONCEPTS = {"sofa", "bookcase", "wardrobe", "lamp", "desk", "chair"}

# attributes: canonical EN, slot order, IT (m form, f form), FR (m form, f form), applies-to categories (None = all)
ATTRS = [
    ("two-seat", 0, ("a due posti", "a due posti"), ("deux places", "deux places"), {"seat"}),
    ("three-seat", 0, ("a tre posti", "a tre posti"), ("trois places", "trois places"), {"seat"}),
    ("double", 0, ("matrimoniale", "matrimoniale"), ("double", "double"), {"bed"}),
    ("single", 0, ("singolo", "singola"), ("simple", "simple"), {"bed"}),
    ("small", 0, ("piccolo", "piccola"), ("petit", "petite"), None),
    ("large", 0, ("grande", "grande"), ("grand", "grande"), None),
    ("white", 1, ("bianco", "bianca"), ("blanc", "blanche"), None),
    ("black", 1, ("nero", "nera"), ("noir", "noire"), None),
    ("grey", 1, ("grigio", "grigia"), ("gris", "grise"), None),
    ("green", 1, ("verde", "verde"), ("vert", "verte"), None),
    ("blue", 1, ("blu", "blu"), ("bleu", "bleue"), None),
    ("red", 1, ("rosso", "rossa"), ("rouge", "rouge"), None),
    ("beige", 1, ("beige", "beige"), ("beige", "beige"), None),
    ("wooden", 2, ("in legno", "in legno"), ("en bois", "en bois"), None),
    ("metal", 2, ("in metallo", "in metallo"), ("en métal", "en métal"), None),
    ("glass", 2, ("in vetro", "in vetro"), ("en verre", "en verre"), {"table", "storage", "decor", "lighting"}),
    ("leather", 2, ("in pelle", "in pelle"), ("en cuir", "en cuir"), {"seat"}),
    ("velvet", 2, ("in velluto", "in velluto"), ("en velours", "en velours"), {"seat", "textile"}),
    ("oak", 2, ("in rovere", "in rovere"), ("en chêne", "en chêne"), {"table", "storage", "bed"}),
]
# pre-nominal in FR (petit/grand); all others post-nominal
FR_PRENOMINAL = {"small", "large"}
IT_PRENOMINAL = {"small", "large"}

FIRST = ["Luca", "Elena", "Paolo", "Chiara", "Sophie", "Julien", "Thomas", "Emma", "Laura", "Davide", "Sara", "Nicolas",
         "Claire", "Matteo", "Francesca", "Hugo", "Peter", "Hannah", "Omar", "Ines", "Karim", "Alice", "Giorgio", "Marta",
         "Camille", "Antoine", "Lucie", "Federico", "Valentina", "Stefano", "Manon", "Pierre", "Giulio", "Aurélie",
         "Samuel", "Nadia", "Riccardo", "Margaux", "Tommaso", "Leila", "Bruno", "Agnese", "Yann", "Irene", "Noah",
         "Beatrice", "Rémi", "Sofia", "Kevin", "Anaïs"]
LAST = ["Ferrari", "Romano", "Colombo", "Moreau", "Lefevre", "Martin", "Bernard", "Esposito", "Greco", "Conti",
        "Fontaine", "Weber", "Novak", "Marchetti", "Gallo", "Dubois", "Petit", "Mancini", "Lambert", "Costa", "Ricciardi",
        "Bonnet", "Girard", "Fabbri", "Leroy", "Barbieri", "Rousseau", "Caruso", "Mercier", "Lombardi", "Garnier",
        "Moretti", "Faure", "De Luca", "Chevalier", "Rinaldi", "Blanc", "Santoro", "Perrin", "Vitale", "Morel",
        "Testa", "Fournier", "Pellegrini", "Durand", "Gentile", "Roux", "Ferraro", "Vincent", "Martini"]
EXCL_NAMES = {"mario rossi", "giulia bianchi", "marco rizzi", "marco ricci", "siobhan okonkwo-laurent", "anna verdi"}
EXCL_PC = {"10121", "20135", "00184", "75011", "10125", "20121", "50123", "99999"}

DIG = {"en": "zero one two three four five six seven eight nine".split(),
       "it": "zero uno due tre quattro cinque sei sette otto nove".split(),
       "fr": "zéro un deux trois quatre cinq six sept huit neuf".split()}
FR_TENS = {10: "dix", 11: "onze", 12: "douze", 13: "treize", 14: "quatorze", 15: "quinze", 16: "seize", 20: "vingt",
           30: "trente", 40: "quarante", 50: "cinquante", 60: "soixante"}
IT_UNITS = "zero uno due tre quattro cinque sei sette otto nove dieci undici dodici tredici quattordici quindici sedici " \
           "diciassette diciotto diciannove".split()
IT_TENS = {2: "venti", 3: "trenta", 4: "quaranta", 5: "cinquanta", 6: "sessanta", 7: "settanta", 8: "ottanta",
           9: "novanta"}


def fr_number(n: int) -> str:
    """0-99 in French words (as spoken in grouped postcodes: 75 -> soixante-quinze)."""
    if n < 10:
        return DIG["fr"][n]
    if n in FR_TENS:
        return FR_TENS[n]
    if n < 17:
        return FR_TENS[n]
    if n < 20:
        return "dix-" + DIG["fr"][n - 10]
    if n < 70:
        t, u = divmod(n, 10)
        base = FR_TENS[t * 10]
        return base if u == 0 else base + ("-et-un" if u == 1 else "-" + DIG["fr"][u])
    if n < 80:
        return "soixante-" + ("et-onze" if n == 71 else fr_number(n - 60))
    if n == 80:
        return "quatre-vingts"
    return "quatre-vingt-" + fr_number(n - 80)


def it_number(n: int) -> str:
    if n < 20:
        return IT_UNITS[n]
    t, u = divmod(n, 10)
    base = IT_TENS[t]
    if u in (1, 8):
        base = base[:-1]
    return base + ("" if u == 0 else IT_UNITS[u])


def en_number(n: int) -> str:
    units = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen " \
            "seventeen eighteen nineteen".split()
    tens = {2: "twenty", 3: "thirty", 4: "forty", 5: "fifty", 6: "sixty", 7: "seventy", 8: "eighty", 9: "ninety"}
    if n < 20:
        return units[n]
    t, u = divmod(n, 10)
    return tens[t] + ("" if u == 0 else "-" + units[u])


def spoken_postcode(pc: str, lang: str, grouped: bool) -> str:
    """digit by digit, or grouped as people say them (FR: 75 011 -> soixante-quinze zéro onze; IT/EN: pairs)."""
    if not grouped:
        return " ".join(DIG[lang][int(c)] for c in pc)
    parts = [pc[:2], pc[2], pc[3:]]                       # e.g. 41 5 40
    f = {"fr": fr_number, "it": it_number, "en": en_number}[lang]
    out = []
    for p in parts:
        if len(p) == 2 and p[0] == "0":
            out += [DIG[lang][0], DIG[lang][int(p[1])]]   # leading zero is said
        else:
            out.append(f(int(p)))
    return " ".join(out)


WORD2DIG = {}
for _lang, _ws in DIG.items():
    for _i, _w in enumerate(_ws):
        WORD2DIG.setdefault(_w, str(_i))


def _num_words(lang: str) -> dict[str, str]:
    f = {"fr": fr_number, "it": it_number, "en": en_number}[lang]
    return {f(n): f"{n:02d}" if n >= 10 else str(n) for n in range(100)}


def parse_spoken_digits(text: str, lang: str, multi_first: bool = False, max_k: int = 3) -> str:
    """Digits a caller said, digit by digit or grouped ('soixante-quinze zéro onze' -> 75011), plus any digits the ASR
    wrote as numerals. Greedy longest match on the language's 0-99 words.
    max_k = longest unhyphenated multi-token number tried. The frozen corpus-v2 verification used 3, which never
    matches the 4-token FR forms ("quatre vingt dix sept"); max_k=4 covers every FR 0-99 form (finding 2026-11-13,
    card §21). The default stays 3 so the v2 verification reports stay reproducible."""
    nums = _num_words(lang)
    toks = re.findall(r"[0-9]+|[a-zà-üœ'-]+", text.lower())
    out, i = [], 0
    while i < len(toks):
        t = toks[i]
        if t.isdigit():
            out.append(t)
            i += 1
            continue
        hit = (nums[t], 1) if t in nums and ("-" in t or not multi_first) else None   # hyphenated token = one number
        if hit is None:
            for k in range(max_k, 1, -1):             # unhyphenated multi-token forms ("soixante dix sept")
                cand = "-".join(toks[i:i + k])
                if cand in nums:
                    hit = (nums[cand], k)
                    break
        if hit is None and t in nums:
            hit = (nums[t], 1)
        if hit is None and t in WORD2DIG:
            hit = (WORD2DIG[t], 1)
        if hit:
            out.append(hit[0])
            i += hit[1]
        else:
            i += 1
    return "".join(out)
