"""Gold matcher v2 for QA v2.1 (CPU). Replaces the regex any-match of corpus_v21.gold_hit for scoring.

Fixes found by the blinded manual check of 2026-11-12 (9/30 test hits were false):
  * spelled numbers are parsed into integers first, so "four" inside "three hundred and twenty four" is 324, not 4;
  * number_with_unit: the gold unit must follow the number (within 3 tokens): "eleven months" != "11 years";
  * arithmetic (SVAMP): the answer is the LAST number in the reply (VC states the result last), not any mention;
  * dates: year / month / day from the canonical text (ordinals parsed: "twenty fifth" -> 25), every component the
    gold gives must appear (unchanged rule);
  * entities / short phrases: gold text or alias as a whole phrase (unchanged), article dropped;
  * leakage: an item whose gold already occurs in the EN question is flagged `echo` and excluded from scoring.
Known residual false positives (left to the manual audit): hedged enumerations ("states like Mississippi,
Louisiana and Texas"), self-contradicting replies, a gold entity used in a wrong sentence ("the first Sunday of
the seventh month is Sunday").
"""
from __future__ import annotations

import re
import unicodedata

ONES = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen "
                                   "fourteen fifteen sixteen seventeen eighteen nineteen".split())}
TENS = {w: 10 * i for i, w in enumerate("_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()) if w != "_"}
ORDS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8,
        "ninth": 9, "tenth": 10, "eleventh": 11, "twelfth": 12, "thirteenth": 13, "fourteenth": 14, "fifteenth": 15,
        "sixteenth": 16, "seventeenth": 17, "eighteenth": 18, "nineteenth": 19, "twentieth": 20, "thirtieth": 30}
SCALES = {"hundred": 100, "thousand": 1000, "million": 1_000_000, "billion": 1_000_000_000}
MONTHS = "january february march april may june july august september october november december".split()
PRON_DET = {"the", "this", "that", "each", "every", "which", "any", "no", "another", "spanish", "chinese", "english",
            "other", "last", "only", "big", "small", "new", "old", "first", "second"}
NUMWORD = set(ONES) | set(TENS) | set(SCALES) | set(ORDS) | {"and", "oh", "a"}
UNIT_ALIASES = {"°f": ["degrees", "fahrenheit"], "°c": ["degrees", "celsius"], "km": ["kilometers", "kilometres", "km"],
                "mi": ["miles"], "m": ["meters", "metres"], "kg": ["kilograms", "kilos", "kg"], "lb": ["pounds"],
                "ft": ["feet", "foot"], "%": ["percent", "per cent"]}


def norm(s: str) -> str:
    s = "".join(c for c in unicodedata.normalize("NFKD", s.lower()) if not unicodedata.combining(c))
    s = re.sub(r"</?s>", " ", s)
    s = re.sub(r"(?<=\d)[,.](?=\d{3}\b)", "", s)  # 384,400 -> 384400
    s = re.sub(r"(\d)(st|nd|rd|th)\b", r"\1", s)  # 25th -> 25
    s = s.replace("-", " ").replace("‑", " ")
    s = re.sub(r"\.(?!\d)|(?<!\d)\.", " ", s)  # keep only decimal points
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9.% ]", " ", s)).strip()


def _ord_value(w: str):
    if w in ORDS:
        return ORDS[w]
    return None


def canon_tokens(text: str) -> list[str]:
    """Tokens with every spelled number replaced by its digits. Year-style pairs ("nineteen seventy seven",
    "seventeen oh seven", "twenty twelve") become one 4-digit token; "twenty fifth" -> "25"."""
    toks = norm(text).split()
    out, i = [], 0
    while i < len(toks):
        w = toks[i]
        if w == "one" and i > 0 and toks[i - 1] in PRON_DET and not (i + 1 < len(toks) and toks[i + 1] in NUMWORD):
            out.append(w)  # pronoun "one" ("the Spanish one", "each one")
            i += 1
            continue
        if w in NUMWORD and w not in ("and", "oh", "a") or (w == "a" and i + 1 < len(toks) and toks[i + 1] in SCALES):
            j, words = i, []
            while j < len(toks) and toks[j] in NUMWORD:
                if toks[j] == "and" and not (j + 1 < len(toks) and toks[j + 1] in NUMWORD and toks[j + 1] not in ("and", "a")):
                    break
                if toks[j] == "a" and not (j + 1 < len(toks) and toks[j + 1] in SCALES):
                    break
                words.append(toks[j])
                j += 1
                if words[-1] in ORDS:  # an ordinal closes the number
                    break
            v = parse_number(words)
            if v is None:
                out.append(w)
                i += 1
                continue
            out.append(str(v))
            i = j
            continue
        out.append(w)
        i += 1
    return out


def _cardinal(words: list[str]):
    total, cur, seen = 0, 0, False
    for w in words:
        if w in ("and",):
            continue
        if w == "a":
            cur = cur or 1
            continue
        if w in ONES:
            cur += ONES[w]
        elif w in TENS:
            cur += TENS[w]
        elif w in ORDS:
            cur += ORDS[w]
        elif w == "hundred":
            cur = (cur or 1) * 100
        elif w in SCALES:
            total += (cur or 1) * SCALES[w]
            cur = 0
        else:
            return None
        seen = True
    return total + cur if seen else None


def parse_number(words: list[str]):
    if not words:
        return None
    # year style: <teen|tens[+one]> (oh)? <tens[+one]|teen|one> with no scale word: 19 77 -> 1977, 17 oh 7 -> 1707
    if not any(w in SCALES for w in words) and "and" not in words:
        groups, cur = [], []
        for w in words:
            if w == "oh":
                cur.append(0)
                continue
            if w in TENS and cur and not (len(cur) == 1 and cur[0] in TENS.values()):
                groups.append(cur)
                cur = [TENS[w]]
                continue
            if w in TENS:
                if cur:
                    groups.append(cur)
                cur = [TENS[w]]
            elif w in ONES:
                if cur and len(cur) == 1 and cur[0] in TENS.values() and ONES[w] < 10:
                    cur.append(ONES[w])
                else:
                    if cur:
                        groups.append(cur)
                    cur = [ONES[w]]
            elif w in ORDS:
                if cur and len(cur) == 1 and cur[0] in TENS.values() and ORDS[w] < 10:
                    cur.append(ORDS[w])
                else:
                    if cur:
                        groups.append(cur)
                    cur = [ORDS[w]]
            else:
                return None
        if cur:
            groups.append(cur)
        vals = [sum(g) for g in groups]
        if len(vals) == 2 and 10 <= vals[0] <= 20 and 0 <= vals[1] <= 99 and words[0] not in ORDS:
            return vals[0] * 100 + vals[1]  # year reading
        if len(vals) == 1:
            return vals[0]
        if len(vals) == 3 and words[1] == "oh":
            return vals[0] * 100 + vals[2]
        return None
    return _cardinal(words)


def numbers_with_pos(text: str):
    toks = canon_tokens(text)
    return toks, [(k, float(t)) for k, t in enumerate(toks) if re.fullmatch(r"\d+(\.\d+)?", t)]


def gold_number(text: str):
    m = re.match(r"^\s*(-?\d+(?:\.\d+)?)\s*(.*)$", (text or "").replace(",", ""))
    if not m:
        return None, None
    v = float(m.group(1))
    unit = m.group(2).strip().lower() or None
    return v, unit


def unit_words(unit: str | None) -> list[str]:
    if not unit:
        return []
    if unit in UNIT_ALIASES:
        return UNIT_ALIASES[unit]
    w = norm(unit).split()
    if not w:
        return []
    head = w[-1]
    stem = head[:-1] if head.endswith("s") else head
    return [stem, stem + "s"] if len(stem) > 1 else []


def parse_date(text: str):
    t = (text or "").strip().lower()
    m = re.fullmatch(r"(\d{4})(?:-(\d{2}))?(?:-(\d{2}))?", t)
    if m:
        return int(m.group(1)), int(m.group(2)) if m.group(2) else None, int(m.group(3)) if m.group(3) else None
    y = re.search(r"\b(\d{4})\b", t)
    mo = next((i + 1 for i, n in enumerate(MONTHS) if re.search(rf"\b{n}\b", t)), None)
    if not y:
        return None
    rest = re.sub(r"\b\d{4}\b", " ", t)
    dd = re.search(r"\b(\d{1,2})\b", rest)
    return int(y.group(1)), mo, int(dd.group(1)) if (dd and mo) else None


def phrase(p: str):
    p = " ".join(norm(p).split())
    p = re.sub(r"^(the|a|an) ", "", p)
    return p if len(p) >= 3 else None


def spec(cat: str, gold_text: str, aliases: list[str]) -> dict:
    """cat: 'mkqa:<type>' or 'svamp:arithmetic'."""
    src, typ = cat.split(":", 1)
    if src == "svamp":
        v, _ = gold_number(gold_text)
        return {"kind": "last_number", "value": v}
    if typ in ("number", "number_with_unit"):
        v, u = gold_number(gold_text)
        if v is None:
            for a in aliases:
                v, u = gold_number(a)
                if v is not None:
                    break
        return {"kind": "number", "value": v, "unit": unit_words(u) if typ == "number_with_unit" else []}
    if typ == "date":
        return {"kind": "date", "ymd": parse_date(gold_text)}
    return {"kind": "phrase", "phrases": sorted({p for p in [phrase(gold_text)] + [phrase(a) for a in aliases] if p})}


UNITS = {u + s for u in ("year", "month", "week", "day", "hour", "minute", "second", "meter", "metre", "kilometer",
                          "kilometre", "mile", "foot", "inch", "pound", "kilogram", "gram", "degree", "percent",
                          "season", "episode", "time", "dollar", "point", "goal", "chapter", "book", "film", "movie",
                          "song", "album", "game", "player", "member", "person", "state", "country")
         for s in ("", "s")} | {"feet", "inches", "people", "km", "kg"}


def hit(reply: str, sp: dict) -> bool:
    toks, nums = numbers_with_pos(reply)
    if sp["kind"] == "last_number":
        # the answer is the last number of the first reply segment that states any number ("<s>...</s>" segments:
        # VC often keeps talking after the answer, and a later segment may restart with other numbers)
        if sp["value"] is None:
            return False
        for seg in re.split(r"</s>", reply):
            _, sn = numbers_with_pos(seg)
            if sn:
                return sn[-1][1] == sp["value"]
        return False
    if sp["kind"] == "number":
        if sp["value"] is None:
            return False
        for k, v in nums:
            if v == sp["value"]:
                nxt = [t for t in toks[k + 1:k + 3]]
                unit_here = next((t for t in nxt if t in UNITS), None)
                # unit check: reject only a DIFFERENT unit right after the number ("eleven months" for 11 years);
                # a bare number ("the drinking age is eighteen") is accepted
                if not sp["unit"] or unit_here is None or unit_here in sp["unit"]:
                    return True
        return False
    if sp["kind"] == "date":
        if not sp["ymd"]:
            return False
        y, mo, d = sp["ymd"]
        s = " " + " ".join(toks) + " "
        if f" {y} " not in s:
            return False
        if mo and f" {MONTHS[mo - 1]} " not in s:
            return False
        if d and f" {d} " not in s:
            return False
        return True
    s = " " + " ".join(norm(reply).split()) + " "
    return any(f" {p} " in s for p in sp["phrases"])


def echo(question_en: str, sp: dict) -> bool:
    """The gold is already in the question (a reply repeating the question would score)."""
    if sp["kind"] == "phrase":
        return hit(question_en, sp)
    if sp["kind"] in ("number", "last_number"):
        _, nums = numbers_with_pos(question_en)
        return sp["value"] is not None and any(v == sp["value"] for _, v in nums)
    if sp["kind"] == "date":
        return bool(sp["ymd"]) and str(sp["ymd"][0]) in " ".join(canon_tokens(question_en))
    return False
