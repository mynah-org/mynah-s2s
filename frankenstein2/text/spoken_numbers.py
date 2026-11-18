"""Spoken numbers -> digits in tool arguments (it / fr / en).

The LLM reads a transcript; callers spell postcodes, order ids and phone numbers as words ("uno zero uno due uno",
"X Y Z one two three"). An LLM may pass them to a tool verbatim (D report: Gemma sent "one zero one two one" as a
postcode) and the backend cannot match. ``normalize_args`` rewrites string arguments before a tool runs:

  * a run of >= 2 tokens made of digit words / digits / single UPPER-case letters becomes one code
    ("X Y Z uno due tre" -> "XYZ123", "uno zero uno due uno" -> "10121", "tre tre tre 12" -> "33312");
  * in a field whose name says it is a number or code (postcode, cap, zip, code, id, phone, number, order...), a
    whole value that is one spoken number becomes digits ("ventitré" -> "23", "dieci" -> "10");
  * nothing else changes ("divano a tre posti" keeps its words: lower-case single letters are words, and one digit
    word alone outside a code-like field stays a word).
Model-free; unit-tested.
"""

from __future__ import annotations

import re

_DIG = {
    "it": {"zero": 0, "uno": 1, "una": 1, "un": 1, "due": 2, "tre": 3, "tré": 3, "quattro": 4, "cinque": 5, "sei": 6,
           "sette": 7, "otto": 8, "nove": 9},
    "fr": {"zéro": 0, "zero": 0, "un": 1, "une": 1, "deux": 2, "trois": 3, "quatre": 4, "cinq": 5, "six": 6,
           "sept": 7, "huit": 8, "neuf": 9},
    "en": {"zero": 0, "oh": 0, "o": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
           "eight": 8, "nine": 9},
}
_TEENS = {
    "it": {"dieci": 10, "undici": 11, "dodici": 12, "tredici": 13, "quattordici": 14, "quindici": 15, "sedici": 16,
           "diciassette": 17, "diciotto": 18, "diciannove": 19},
    "fr": {"dix": 10, "onze": 11, "douze": 12, "treize": 13, "quatorze": 14, "quinze": 15, "seize": 16,
           "dix-sept": 17, "dix-huit": 18, "dix-neuf": 19},
    "en": {"ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
           "seventeen": 17, "eighteen": 18, "nineteen": 19},
}
_TENS = {
    "it": {"venti": 20, "trenta": 30, "quaranta": 40, "cinquanta": 50, "sessanta": 60, "settanta": 70, "ottanta": 80,
           "novanta": 90},
    "fr": {"vingt": 20, "trente": 30, "quarante": 40, "cinquante": 50, "soixante": 60},
    "en": {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80,
           "ninety": 90},
}
CODE_KEYS = re.compile(r"(post|cap|zip|code|codice|id\b|_id|phone|tel|number|numero|num\b|order|ordine|pin|otp)",
                       re.I)


def _it_compound(w: str):
    """Italian compound below 100: ventitré, trentuno, quarantotto, ..."""
    for t, tv in _TENS["it"].items():
        for u, uv in _DIG["it"].items():
            if uv == 0 or u in ("una", "un", "tré"):
                continue
            forms = {t + u}
            if u in ("uno", "otto"):
                forms.add(t[:-1] + u)
            if u == "tre":
                forms.add(t + "tré")
            if w in forms:
                return tv + uv
    return None


def word_number(w: str, lang: str):
    """One spoken token -> int (0..99) or None."""
    w = w.lower().strip(".,;:!?")
    if lang not in _DIG:
        return None
    if w in _DIG[lang]:
        return _DIG[lang][w]
    if w in _TEENS[lang]:
        return _TEENS[lang][w]
    if w in _TENS[lang]:
        return _TENS[lang][w]
    if lang == "it":
        return _it_compound(w)
    if lang == "en" and "-" in w:
        a, _, b = w.partition("-")
        if a in _TENS["en"] and b in _DIG["en"] and _DIG["en"][b]:
            return _TENS["en"][a] + _DIG["en"][b]
    if lang == "fr" and "-" in w:
        a, _, b = w.partition("-")
        if a in _TENS["fr"] and b in _DIG["fr"] and _DIG["fr"][b]:
            return _TENS["fr"][a] + _DIG["fr"][b]
    return None


AMBIGUOUS = {"o", "un", "una", "une", "oh"}


def _code_token(tok: str, lang: str):
    """-> (code text, ambiguous) or None."""
    t = tok.strip(".,;:")
    if t.isdigit():
        return t, False
    if len(t) == 1 and t.isalpha() and t.isupper():
        return t, False
    d = _DIG.get(lang, {}).get(t.lower())
    if d is not None:
        return str(d), t.lower() in AMBIGUOUS
    return None


def collapse_codes(text: str, lang: str) -> str:
    """Runs of >= 2 code tokens (digits, digit words, single capitals) with at least one digit and at least two
    unambiguous tokens become one code; everything else is kept as it is."""
    toks = text.split()
    out, i = [], 0
    while i < len(toks):
        j, run = i, []
        while j < len(toks):
            c = _code_token(toks[j], lang)
            if c is None:
                break
            run.append(c)
            j += 1
        firm = sum(1 for _, amb in run if not amb)
        if len(run) >= 2 and firm >= 2 and any(c.isdigit() for c, _ in run):
            out.append("".join(c for c, _ in run))
            i = j
        else:
            out.append(toks[i])
            i += 1
    return " ".join(out)


def normalize_value(key: str, value: str, lang: str) -> str:
    if not isinstance(value, str) or not value.strip():
        return value
    v = collapse_codes(value, lang)
    if CODE_KEYS.search(key or ""):
        toks = v.split()
        nums = [word_number(t, lang) for t in toks]
        if toks and all(n is not None for n in nums):
            v = "".join(str(n) for n in nums)
        if re.fullmatch(r"[A-Za-z0-9 ]+", v) and any(c.isdigit() for c in v) and " " in v:
            v = v.replace(" ", "")
    return v


def _lev(a: str, b: str) -> int:
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, y in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (x != y))
    return d[len(b)]


def codes_in(text: str, lang: str) -> list:
    """Code-like tokens (with at least one digit) the caller said, after collapsing spoken digits / letters."""
    return [t.strip(".,;:!?") for t in collapse_codes(text or "", lang).split()
            if any(c.isdigit() for c in t) and len(t.strip(".,;:!?")) >= 3]


def ground_codes(args: dict, user_texts: list, lang: str) -> tuple:
    """Every model we tried garbles Italian spoken postcodes on its own ("uno zero uno due uno" -> "01201", E report).
    For a code-like argument whose value is NOT among the codes the caller said (transcripts of this session), take the
    caller's code that is an anagram of it or within edit distance 2 (unique match only). -> (args, fixes)."""
    cands = []
    for t in user_texts or []:
        for c in codes_in(t, lang):
            if c not in cands:
                cands.append(c)
    out, fixes = dict(args), []
    for k, v in args.items():
        if not isinstance(v, str) or not CODE_KEYS.search(k) or not any(c.isdigit() for c in v):
            continue
        val = v.replace(" ", "").upper()
        if not cands or val in (c.upper() for c in cands):
            continue
        near = [c for c in cands if sorted(c.upper()) == sorted(val) or
                (abs(len(c) - len(val)) <= 1 and _lev(c.upper(), val) <= 2)]
        if len(near) == 1:
            out[k] = near[0]
            fixes.append({"key": k, "from": v, "to": near[0]})
    return out, fixes


def normalize_args(args: dict, lang: str) -> dict:
    """A copy of tool arguments with spoken numbers turned into digits (strings only, nested dicts included)."""
    out = {}
    for k, v in (args or {}).items():
        if isinstance(v, str):
            out[k] = normalize_value(k, v, lang)
        elif isinstance(v, dict):
            out[k] = normalize_args(v, lang)
        else:
            out[k] = v
    return out
