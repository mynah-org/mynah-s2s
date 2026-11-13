"""Corpus v2b (CPU): postcode verbalisations and an exhaustive ACOUSTIC ambiguity audit (owner 2026-11-13).

Why: in corpus v2 the grouped 2-1-2 reading ("vingt six dix-huit" = 20 6 18) sounds exactly like "vingt-six dix-huit"
(26 18); strict ASR rejected 0/22 of those clips after three renders (card §19). A v2b reading must be decodable by a
listener WITHOUT punctuation, hyphens or pauses.

Acoustic token model (deliberately pessimistic):
  - punctuation and hyphens are dropped; every number word is split into its spoken morphemes
    (FR "quatre-vingt-dix-sept" -> quatre vingt dix sept; "quatre-vingts" -> quatre vingt);
  - IT compounds are split too, with elision ignored ("ventuno" ~ "venti uno", "seicento" -> sei cento);
  - the listener knows every number 0..999 and that a postcode has exactly 5 digits (structure, not punctuation).
A reading is UNAMBIGUOUS for a postcode when exactly one 5-digit string can be segmented out of its token sequence, and
it equals the postcode.

Readings:
  digits  every digit on its own                          20618 -> deux zéro six un huit
  v2_212  corpus v2 grouped (pc[:2], pc[2], pc[3:])        20618 -> vingt six dix-huit           (ambiguous)
  h23     2 + 3 with the hundreds spoken                  20618 -> vingt, six cent dix-huit     (candidate)
  d3      department as a number + 3 single digits        20618 -> vingt, six un huit           (candidate)
  d3s     d3, or every digit when d3 is ambiguous         80200 -> huit zéro deux zéro zéro     (v2b choice)
          (exhaustive audit: d3 is ambiguous only for FR department 80 + a digit 2..9, "quatre-vingts deux" ~
          "quatre vingt-deux"; d3s has 0 ambiguous postcodes in every language)
Leading zeros of a group are spoken as "zéro/zero" digits (05 -> zéro cinq; 018 -> zéro dix-huit).
"""
from __future__ import annotations

import re
import sys
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lexicon_v2 as L  # noqa: E402

LANGS = ("en", "it", "fr")
READINGS = ("digits", "v2_212", "h23", "d3", "d3s")


# ------------------------------------------------------------------------------------------ orthographic numbers
def hundreds_text(n: int, lang: str) -> str:
    """100..999 as written for TTS (FR 'six cent dix-huit', IT 'seicentodiciotto', EN 'six hundred eighteen')."""
    h, r = divmod(n, 100)
    if lang == "fr":
        head = "cent" if h == 1 else f"{L.DIG['fr'][h]} cent" + ("s" if r == 0 else "")
        return head if r == 0 else f"{head} {L.fr_number(r)}"
    if lang == "it":
        head = "cento" if h == 1 else L.IT_UNITS[h] + "cento"
        return head if r == 0 else head + L.it_number(r)
    head = f"{L.DIG['en'][h]} hundred"
    return head if r == 0 else f"{head} {L.en_number(r)}"


def num_text(s: str, lang: str) -> str:
    """A digit group as a number; leading zeros are spoken as zero digits."""
    out = []
    while len(s) > 1 and s[0] == "0":
        out.append(L.DIG[lang][0])
        s = s[1:]
    n = int(s)
    if n >= 100:
        out.append(hundreds_text(n, lang))
    elif len(s) == 1:
        out.append(L.DIG[lang][n])
    else:
        out.append({"fr": L.fr_number, "it": L.it_number, "en": L.en_number}[lang](n))
    return " ".join(out)


def spoken(pc: str, lang: str, reading: str) -> str:
    """Orthographic text given to the TTS. Commas are a TTS hint only; the audit never relies on them."""
    d = L.DIG[lang]
    if reading == "digits":
        return " ".join(d[int(c)] for c in pc)
    if reading == "v2_212":
        return L.spoken_postcode(pc, lang, True)
    if reading == "h23":
        return f"{num_text(pc[:2], lang)}, {num_text(pc[2:], lang)}"
    if reading == "d3":
        return f"{num_text(pc[:2], lang)}, " + " ".join(d[int(c)] for c in pc[2:])
    if reading == "d3s":
        return spoken(pc, lang, "d3") if len(decode(pc, lang, "d3")) == 1 else spoken(pc, lang, "digits")
    raise ValueError(reading)


# ------------------------------------------------------------------------------------------ acoustic tokens
IT_TENS_FULL = {t: L.IT_TENS[t] for t in L.IT_TENS}


def _it_tokens(n: int) -> list[str]:
    if n < 20:
        return [L.IT_UNITS[n]]
    if n < 100:
        t, u = divmod(n, 10)
        return [IT_TENS_FULL[t]] + ([L.IT_UNITS[u]] if u else [])
    h, r = divmod(n, 100)
    return ([] if h == 1 else [L.IT_UNITS[h]]) + ["cento"] + (_it_tokens(r) if r else [])


def _norm(w: str, lang: str) -> str:
    w = w.lower().strip(",.")
    if lang == "fr":
        w = {"vingts": "vingt", "cents": "cent"}.get(w, w)
    return w


def number_tokens(n: int, lang: str) -> tuple[str, ...]:
    """Spoken morphemes of the number n (0..999)."""
    if lang == "it":
        return tuple(_it_tokens(n))
    txt = hundreds_text(n, lang) if n >= 100 else (L.DIG[lang][n] if n < 10 else
                                                     {"fr": L.fr_number, "en": L.en_number}[lang](n))
    return tuple(_norm(w, lang) for w in txt.replace("-", " ").split())


def text_tokens(text: str, lang: str) -> tuple[str, ...]:
    """Orthographic postcode text -> acoustic tokens (IT compounds re-split through the number lexicon)."""
    words = [_norm(w, lang) for w in text.replace("-", " ").replace(",", " ").split()]
    if lang != "it":
        return tuple(words)
    inv = {L.it_number(n) if n < 100 else hundreds_text(n, "it"): n for n in range(1000)}
    out = []
    for w in words:
        out += list(_it_tokens(inv[w])) if w in inv else [w]
    return tuple(out)


@lru_cache(maxsize=None)
def lexicon(lang: str) -> dict[tuple[str, ...], set[str]]:
    lex: dict[tuple[str, ...], set[str]] = {}
    for n in range(1000):
        lex.setdefault(number_tokens(n, lang), set()).add(str(n) if n >= 10 else str(n))
    return lex


def parses(tokens: tuple[str, ...], lang: str, length: int = 5) -> set[str]:
    """Every digit string of exactly `length` digits a listener could segment out of the tokens."""
    lex = lexicon(lang)
    maxk = max(len(k) for k in lex)

    @lru_cache(maxsize=None)
    def go(i: int) -> frozenset:
        if i == len(tokens):
            return frozenset({""})
        res = set()
        if tokens[i].isdigit():                       # a numeral the ASR committed to is one unit, as written
            return frozenset(tokens[i] + r for r in go(i + 1) if len(tokens[i] + r) <= length)
        for k in range(1, min(maxk, len(tokens) - i) + 1):
            for val in lex.get(tokens[i:i + k], ()):
                for rest in go(i + k):
                    s = val + rest
                    if len(s) <= length:
                        res.add(s)
        return frozenset(res)
    return {s for s in go(0) if len(s) == length}


def hyp_has_postcode(hyp: str, lang: str, pc: str) -> bool:
    """v2b strict verification: some maximal run of number words / numerals in the ASR hypothesis has exactly ONE
    5-digit parse, and it is the target postcode. A run with several 5-digit readings is a failure."""
    vocab = {w for k in lexicon(lang) for w in k}
    toks = [_norm(w, lang) for w in re.findall(r"[0-9]+|[a-zà-üœ'-]+", hyp.lower().replace("-", " "))]
    if lang == "it":
        toks = list(text_tokens(" ".join(toks), lang))
    runs, cur = [], []
    for t in toks + [None]:
        if t is not None and (t.isdigit() or t in vocab):
            cur.append(t)
        elif cur:
            runs.append(tuple(cur))
            cur = []
    return any(parses(r, lang) == {pc} for r in runs)


def decode(pc: str, lang: str, reading: str) -> set[str]:
    return parses(text_tokens(spoken(pc, lang, reading), lang), lang)


def audit(lang: str, reading: str, pcs=None) -> dict:
    """Exhaustive round trip over all postcodes (default 00000..99999)."""
    pcs = pcs or [f"{i:05d}" for i in range(100000)]
    bad, amb, examples = 0, 0, []
    for pc in pcs:
        got = decode(pc, lang, reading)
        if pc not in got:
            bad += 1
        if len(got) != 1:
            amb += 1
            if len(examples) < 5:
                examples.append({"pc": pc, "text": spoken(pc, lang, reading), "parses": sorted(got)})
    return {"n": len(pcs), "roundtrip_fail": bad, "ambiguous": amb, "examples": examples}


if __name__ == "__main__":
    import json
    rep = {f"{l}|{r}": audit(l, r) for r in READINGS for l in LANGS}
    print(json.dumps({k: {x: v[x] for x in ("n", "roundtrip_fail", "ambiguous")} for k, v in rep.items()}, indent=1))
    out = sys.argv[1] if len(sys.argv) > 1 else None
    if out:
        Path(out).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
