"""Deterministic spoken-form text normaliser for TTS input, Italian / French / English. No dependencies.

`normalize(text, lang)` rewrites what a TTS cannot be trusted to read into words, sentence-chunk by sentence-chunk:

  phone numbers     "+39 333 123 4567"   -> digits one by one per group ("più tre nove, tre tre tre, ...")
  times             "10:30", "9.05 pm"   -> "dieci e trenta" / "dix heures trente" / "ten thirty"
  dates             "14/11", "14/11/2026", "14 novembre 2026", "1° maggio", "1er mai", "November 14th"
  prices            "49,90 €", "€49.90", "$5", "12 EUR"  -> "quarantanove euro e novanta" / ...
  percentages       "15%", "2,5 %"       -> "quindici per cento" / "quinze pour cent" / "fifteen percent"
  ordinals          "3°", "2ª", "1er", "2e", "3rd", "21st"
  codes             "XYZ123", "A1B2", "10121", "0039"  -> letters spaced + digits one by one
  decimals          "3,5" (it/fr), "3.5" (en)  with the locale's separators; "1.234" / "1 234" / "1,234" thousands
  integers          "42" -> words

Rules that matter for streaming: `split_safe(text)` tells how much of a growing text can be committed without cutting a
number, time, code or price in two (a progressive talker must commit only up to that point).

The module is standalone (stdlib `re` only) so other engines can reuse it. Ported unchanged from the Tyger lab
harness of this project (labs/tyger/textnorm on the lab branch).
"""
from __future__ import annotations

import re

__all__ = ["normalize", "number_words", "ordinal_words", "split_safe", "LANGS"]
LANGS = ("it", "fr", "en")

# ---------------------------------------------------------------------------------------------------------------
# cardinals
_EN_U = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen " \
        "seventeen eighteen nineteen".split()
_EN_T = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
_IT_U = "zero uno due tre quattro cinque sei sette otto nove dieci undici dodici tredici quattordici quindici " \
        "sedici diciassette diciotto diciannove".split()
_IT_T = "_ _ venti trenta quaranta cinquanta sessanta settanta ottanta novanta".split()
_FR_U = "zéro un deux trois quatre cinq six sept huit neuf dix onze douze treize quatorze quinze seize dix-sept " \
        "dix-huit dix-neuf".split()
_FR_T = "_ _ vingt trente quarante cinquante soixante".split()


def _en(n: int) -> str:
    if n < 20:
        return _EN_U[n]
    if n < 100:
        t, u = divmod(n, 10)
        return _EN_T[t] + ("-" + _EN_U[u] if u else "")
    if n < 1000:
        h, r = divmod(n, 100)
        return _EN_U[h] + " hundred" + (" " + _en(r) if r else "")
    for div, name in ((10 ** 9, "billion"), (10 ** 6, "million"), (1000, "thousand")):
        if n >= div:
            q, r = divmod(n, div)
            return _en(q) + " " + name + (" " + _en(r) if r else "")
    raise ValueError(n)


def _it(n: int) -> str:
    if n < 20:
        return _IT_U[n]
    if n < 100:
        t, u = divmod(n, 10)
        tens = _IT_T[t]
        if u in (1, 8):
            tens = tens[:-1]                        # ventuno, ventotto
        return tens + ("tré" if u == 3 else _IT_U[u] if u else "")
    if n < 1000:
        h, r = divmod(n, 100)
        head = "cento" if h == 1 else _IT_U[h] + "cento"
        if r and r // 10 == 8:                      # centottanta, trecentottanta
            head = head[:-1]
        return head + (_it(r) if r else "")
    if n < 1_000_000:
        q, r = divmod(n, 1000)
        head = "mille" if q == 1 else _it(q) + "mila"
        return head + (_it(r) if r else "")
    for div, one, many in ((10 ** 9, "un miliardo", "miliardi"), (10 ** 6, "un milione", "milioni")):
        if n >= div:
            q, r = divmod(n, div)
            head = one if q == 1 else _it(q) + " " + many
            return head + (" " + _it(r) if r else "")
    raise ValueError(n)


def _fr_below_100(n: int) -> str:
    if n < 20:
        return _FR_U[n]
    t, u = divmod(n, 10)
    if t in (7, 9):                                 # soixante-dix, quatre-vingt-dix
        base = "soixante" if t == 7 else "quatre-vingt"
        r = 10 + u
        if t == 7 and u == 1:
            return "soixante et onze"
        return base + "-" + _FR_U[r]
    if t == 8:
        return "quatre-vingts" if u == 0 else "quatre-vingt-" + _FR_U[u]
    tens = _FR_T[t]
    if u == 0:
        return tens
    if u == 1:
        return tens + " et un"
    return tens + "-" + _FR_U[u]


def _fr(n: int) -> str:
    if n < 100:
        return _fr_below_100(n)
    if n < 1000:
        h, r = divmod(n, 100)
        head = "cent" if h == 1 else _FR_U[h] + " cent" + ("s" if r == 0 else "")
        return head + (" " + _fr_below_100(r) if r else "")
    if n < 1_000_000:
        q, r = divmod(n, 1000)
        w = _fr(q)
        if w.endswith(("cents", "vingts")):
            w = w[:-1]                              # deux cent mille, quatre-vingt mille
        head = "mille" if q == 1 else w + " mille"
        return head + (" " + _fr(r) if r else "")
    for div, name in ((10 ** 9, "milliard"), (10 ** 6, "million")):
        if n >= div:
            q, r = divmod(n, div)
            head = ("un " + name) if q == 1 else _fr(q) + " " + name + "s"
            return head + (" " + _fr(r) if r else "")
    raise ValueError(n)


_CARD = {"en": _en, "it": _it, "fr": _fr}


def number_words(n: int, lang: str) -> str:
    if n < 0:
        return {"en": "minus ", "it": "meno ", "fr": "moins "}[lang] + _CARD[lang](-n)
    return _CARD[lang](n)


def _digits(s: str, lang: str) -> str:
    return " ".join(_CARD[lang](int(c)) for c in s if c.isdigit())


# ---------------------------------------------------------------------------------------------------------------
# ordinals
_EN_ORD = {"one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth", "nine": "ninth",
           "twelve": "twelfth"}
_IT_ORD = ["", "primo", "secondo", "terzo", "quarto", "quinto", "sesto", "settimo", "ottavo", "nono", "decimo"]


def ordinal_words(n: int, lang: str, fem: bool = False) -> str:
    if lang == "en":
        w = _en(n)
        head, _, last = w.rpartition("-") if "-" in w.split(" ")[-1] else w.rpartition(" ")
        sep = "-" if "-" in w.split(" ")[-1] else " "
        if last in _EN_ORD:
            last = _EN_ORD[last]
        elif last.endswith("y"):
            last = last[:-1] + "ieth"
        else:
            last = last + "th"
        return (head + sep + last) if head else last
    if lang == "it":
        if n <= 10:
            w = _IT_ORD[n]
        else:
            c = _it(n)
            if c.endswith("tré"):
                w = c[:-1] + "eesimo"               # ventitreesimo
            elif c.endswith("sei"):
                w = c + "esimo"                     # ventiseiesimo
            else:
                w = c[:-1] + "esimo"                # undicesimo, ventesimo, centesimo
        return w[:-1] + "a" if fem else w
    if lang == "fr":
        if n == 1:
            return "première" if fem else "premier"
        c = _fr(n)
        if c.endswith("cinq"):
            return c + "uième"
        if c.endswith("neuf"):
            return c[:-1] + "vième"
        if c.endswith("e"):
            return c[:-1] + "ième"
        if c.endswith("s") and (c.endswith("vingts") or c.endswith("cents")):
            return c[:-1] + "ième"
        return c + "ième"
    raise ValueError(lang)


# ---------------------------------------------------------------------------------------------------------------
# decimals / amounts
def _dec_sep(lang):
    return "." if lang == "en" else ","


def _parse_number(s: str, lang: str):
    """'1.234,5' (it/fr), '1 234,5', '1,234.5' (en) -> (int part, decimal digits str or '') or None."""
    s = s.replace(" ", " ").replace(" ", " ")
    ds = _dec_sep(lang)
    ts = "," if lang == "en" else "."
    m = re.search(r"[.,](\d+)$", s)
    if m and len(m.group(1)) != 3 and s[m.start()] == ts:
        ds, ts = ts, ds                  # "49,90" in English text, "3.5" in Italian text: a decimal, not thousands
    if ds in s:
        ip, _, dp = s.rpartition(ds)
    else:
        ip, dp = s, ""
    ip = ip.replace(ts, "").replace(" ", "").replace("'", "")
    if not ip.isdigit() or (dp and not dp.isdigit()):
        return None
    return int(ip), dp


def _decimal_words(ip: int, dp: str, lang: str) -> str:
    w = number_words(ip, lang)
    if not dp:
        return w
    point = {"en": "point", "it": "virgola", "fr": "virgule"}[lang]
    if lang == "en" or dp.startswith("0") or len(dp) > 2:
        tail = _digits(dp, lang)
    else:
        tail = number_words(int(dp), lang)
    return f"{w} {point} {tail}"


_CUR = {"€": "eur", "eur": "eur", "euro": "eur", "euros": "eur", "$": "usd", "usd": "usd", "£": "gbp", "gbp": "gbp"}
_CUR_NAMES = {
    "it": {"eur": ("euro", "euro", "centesimo", "centesimi"), "usd": ("dollaro", "dollari", "centesimo", "centesimi"),
           "gbp": ("sterlina", "sterline", "penny", "pence")},
    "fr": {"eur": ("euro", "euros", "centime", "centimes"), "usd": ("dollar", "dollars", "cent", "cents"),
           "gbp": ("livre", "livres", "penny", "pence")},
    "en": {"eur": ("euro", "euros", "cent", "cents"), "usd": ("dollar", "dollars", "cent", "cents"),
           "gbp": ("pound", "pounds", "penny", "pence")},
}


def _money(num: str, cur: str, lang: str):
    p = _parse_number(num, lang)
    if p is None:
        return None
    ip, dp = p
    one, many, c1, cn = _CUR_NAMES[lang][_CUR[cur.lower()]]
    unit = one if ip == 1 else many
    if lang == "it" and ip == 1:
        head = "un " + unit
    elif lang == "fr" and ip == 1:
        head = "un " + unit
    elif lang == "en" and ip == 1:
        head = "one " + unit
    else:
        head = number_words(ip, lang) + " " + unit
    if not dp or int(dp) == 0:
        return head
    cents = int((dp + "0")[:2])
    if lang == "it":
        return f"{head} e {number_words(cents, lang)}"          # "quarantanove euro e novanta"
    if lang == "fr":
        return f"{head} {number_words(cents, lang)}"            # "quarante-neuf euros quatre-vingt-dix"
    return f"{head} and {number_words(cents, lang)} {c1 if cents == 1 else cn}"


# ---------------------------------------------------------------------------------------------------------------
# times and dates
_MONTHS = {
    "it": "gennaio febbraio marzo aprile maggio giugno luglio agosto settembre ottobre novembre dicembre".split(),
    "fr": "janvier février mars avril mai juin juillet août septembre octobre novembre décembre".split(),
    "en": "January February March April May June July August September October November December".split(),
}


def _time(h: int, m: int, lang: str, ampm: str = "") -> str | None:
    if h > 24 or m > 59:
        return None
    if lang == "it":
        hw = "mezzanotte" if h in (0, 24) else ("una" if h == 1 else number_words(h, lang))
        return hw if m == 0 else f"{hw} e {number_words(m, lang)}"
    if lang == "fr":
        hw = "minuit" if h in (0, 24) else ("midi" if h == 12 and not ampm else
                                            ("une heure" if h == 1 else number_words(h, lang) + " heures"))
        return hw if m == 0 else f"{hw} {number_words(m, lang)}"
    suffix = {"a": " a m", "p": " p m"}.get(ampm[:1].lower(), "") if ampm else ""
    hw = number_words((h % 12 or 12) if ampm else h, lang)
    if m == 0:
        return f"{hw}{suffix}" if ampm else f"{hw} o'clock"
    mw = ("oh " + number_words(m, lang)) if m < 10 else number_words(m, lang)
    return f"{hw} {mw}{suffix}"


def _year(y: int, lang: str) -> str:
    if lang == "en" and 1100 <= y <= 2099 and not (2000 <= y <= 2009):
        hi, lo = divmod(y, 100)
        if lo == 0:
            return _en(hi) + " hundred"
        return _en(hi) + " " + (("oh " + _en(lo)) if lo < 10 else _en(lo))
    return number_words(y, lang)


def _date(d: int, mo: int, y: int | None, lang: str) -> str | None:
    if not (1 <= d <= 31 and 1 <= mo <= 12):
        return None
    month = _MONTHS[lang][mo - 1]
    if lang == "it":
        dw = "primo" if d == 1 else number_words(d, lang)
        s = f"{dw} {month}"
    elif lang == "fr":
        dw = "premier" if d == 1 else number_words(d, lang)
        s = f"{dw} {month}"
    else:
        s = f"{month} {ordinal_words(d, lang)}"
    if y is not None:
        s += (", " if lang == "en" else " ") + _year(y, lang)
    return s


# ---------------------------------------------------------------------------------------------------------------
# the rewrite passes
_NUM = r"\d{1,3}(?:[.,'   ]\d{3})*(?:[.,]\d+)?|\d+(?:[.,]\d+)?"
_MONTH_RE = {k: "|".join(v) for k, v in _MONTHS.items()}


def _sub(pattern, fn, text, flags=0):
    def rep(m):
        r = fn(m)
        return m.group(0) if r is None else r
    return re.sub(pattern, rep, text, flags=flags)


def normalize(text: str, lang: str) -> str:
    """The spoken form of `text` in `lang` (it / fr / en). Idempotent on text without digits or symbols."""
    if lang not in LANGS:
        raise ValueError(f"unsupported language {lang!r}")
    t = text
    plus = {"it": "più", "fr": "plus", "en": "plus"}[lang]

    # ISO dates (2026-11-14): LLMs copy them from tool results; read as a date, not as phone-number groups
    t = _sub(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b",
             lambda m: _date(int(m.group(3)), int(m.group(2)), int(m.group(1)), lang), t)

    # phone numbers: +CC and groups, or 3+ groups of digits joined by spaces / dashes / dots
    def phone(m):
        s = m.group(0)
        groups = re.findall(r"\+?\d+", s)
        if sum(len(g.lstrip("+")) for g in groups) < 8:
            return None
        return ", ".join(((plus + " ") if g.startswith("+") else "") + _digits(g, lang) for g in groups)
    t = _sub(r"(?<![\w.,])\+\d{1,3}(?:[ \-.]?\d{1,4}){2,6}(?![\w])|(?<![\w.,])\d{2,4}(?:[ \-.]\d{2,4}){2,4}(?![\w.,]\d)",
             phone, t)

    # times: 10:30, 9:05 pm; it/fr also 10.30 when followed by an hour word is ambiguous: only ':' and 'h' (fr)
    def tm(m):
        h, mi = int(m.group(1)), int(m.group(2) or 0)
        return _time(h, mi, lang, (m.group(3) or "").replace(".", "").strip())
    t = _sub(r"\b(\d{1,2}):(\d{2})(?:\s*([ap]\.?m\.?))?(?!\d)", tm, t, re.I)
    if lang == "fr":
        t = _sub(r"\b(\d{1,2})\s?h\s?(\d{2})?\b()", tm, t)
    if lang == "en":
        t = _sub(r"\b(\d{1,2})()\s*([ap]\.?m\.?)", tm, t, re.I)

    # numeric dates: 14/11, 14/11/2026, 14-11-2026 (day first in it/fr; month first in en)
    def nd(m):
        a, b = int(m.group(1)), int(m.group(2))
        y = m.group(3)
        y = int(y) + (2000 if y and len(y) == 2 else 0) if y else None
        d, mo = (b, a) if lang == "en" and a <= 12 else (a, b)   # en: month first unless it can't be a month
        return _date(d, mo, y, lang)
    t = _sub(r"\b(\d{1,2})[/](\d{1,2})(?:[/](\d{4}|\d{2}))?\b", nd, t)
    t = _sub(r"\b(\d{1,2})-(\d{1,2})-(\d{4})\b", nd, t)

    # written dates: "14 novembre 2026", "1° maggio", "1er mai", "November 14(th), 2026", "14th of November"
    mre = _MONTH_RE[lang]
    if lang in ("it", "fr"):
        def wd(m):
            d = int(m.group(1)); y = m.group(3)
            mo = [x.lower() for x in _MONTHS[lang]].index(m.group(2).lower()) + 1
            return _date(d, mo, int(y) if y else None, lang)
        t = _sub(rf"\b(\d{{1,2}})(?:°|º|er|re)?\s+({mre})(?:\s+(\d{{4}}))?\b", wd, t, re.I)
    else:
        def wd_en(m):
            mo = [x.lower() for x in _MONTHS["en"]].index(m.group(1).lower()) + 1
            y = m.group(3)
            return _date(int(m.group(2)), mo, int(y) if y else None, lang)
        t = _sub(rf"\b({mre})\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:,?\s+(\d{{4}}))?\b", wd_en, t, re.I)

        def wd_en2(m):
            mo = [x.lower() for x in _MONTHS["en"]].index(m.group(2).lower()) + 1
            y = m.group(3)
            return (ordinal_words(int(m.group(1)), lang) + " of " + _MONTHS["en"][mo - 1]
                    + (", " + _year(int(y), lang) if y else ""))
        t = _sub(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({mre})(?:,?\s+(\d{{4}}))?\b", wd_en2, t, re.I)

    # prices: "49,90 €", "49.90€", "€ 49,90", "$5", "12 EUR", "12 euro"
    cur_after = r"(€|EUR|euros?|\$|USD|£|GBP)"
    t = _sub(rf"(?<![\w.,])({_NUM})\s?{cur_after}(?![\w])", lambda m: _money(m.group(1), m.group(2), lang), t, re.I)
    t = _sub(rf"(€|\$|£)\s?({_NUM})(?![\d])", lambda m: _money(m.group(2), m.group(1), lang), t)

    # percentages
    pct = {"it": "per cento", "fr": "pour cent", "en": "percent"}[lang]

    def pc(m):
        p = _parse_number(m.group(1), lang)
        return None if p is None else _decimal_words(*p, lang) + " " + pct
    t = _sub(rf"(?<![\w.,])({_NUM})\s?%", pc, t)

    # ordinals
    if lang == "it":
        t = _sub(r"\b(\d{1,4})\s?([°º])", lambda m: ordinal_words(int(m.group(1)), lang), t)
        t = _sub(r"\b(\d{1,4})\s?ª", lambda m: ordinal_words(int(m.group(1)), lang, fem=True), t)
    elif lang == "fr":
        t = _sub(r"\b1(er|re)\b", lambda m: "premier" if m.group(1) == "er" else "première", t)
        t = _sub(r"\b(\d{1,4})(?:e|ème|eme)\b", lambda m: ordinal_words(int(m.group(1)), lang), t)
    else:
        t = _sub(r"\b(\d{1,4})(?:st|nd|rd|th)\b", lambda m: ordinal_words(int(m.group(1)), lang), t, re.I)

    # numbers with thousands separators ("1.234", "200 000", "1,234") before codes, so "000" is not a code
    ts = r"[.\u202f\u00a0 ']" if lang != "en" else r"[,\u202f\u00a0]"

    def thou(m):
        p = _parse_number(m.group(0), lang)
        return None if p is None else _decimal_words(*p, lang)
    t = _sub(rf"(?<![\w.,])\d{{1,3}}(?:{ts}\d{{3}})+(?:{'[.]' if lang == 'en' else ','}\d+)?(?![\w]|[.,]\d)", thou, t)

    # codes: letters and digits mixed (XYZ123, A1B2, 2B), or digit strings that are not quantities (leading zero,
    # 5+ digits without separators): letters spaced, digits one by one
    def code(m):
        s = m.group(0)
        out = []
        for ch in s:
            if ch.isdigit():
                out.append(_CARD[lang](int(ch)))
            elif ch.isalpha():
                out.append(ch.upper())
        return " ".join(out)
    t = _sub(r"\b(?=[A-Za-z]*\d)(?=\d*[A-Za-z])[A-Za-z0-9]*[A-Z0-9][A-Za-z0-9]*\b",
             lambda m: code(m) if (any(c.isupper() for c in m.group(0)) or len(m.group(0)) <= 6) else None, t)
    t = _sub(r"(?<![\w.,])(?:0\d+|\d{5,})(?![\w.,]\d)", code, t)

    # decimals and integers
    def num(m):
        p = _parse_number(m.group(0), lang)
        return None if p is None else _decimal_words(*p, lang)
    t = _sub(rf"(?<![\w])({_NUM})(?![\w])", num, t)
    t = re.sub(r"\s{2,}", " ", t)
    return t


def split_safe(text: str) -> int:
    """How many characters of a GROWING text can be committed to a progressive talker without cutting a number,
    price, time, date, code or phone number: up to the last whitespace that is not followed by more of a numeric
    expression still being written. Text after the returned index must wait for more input (or the end)."""
    i = len(text)
    while i > 0 and not text[i - 1].isspace():
        i -= 1
    head = text[:i]
    # a trailing token like "49," "10:" "+39" "14/" "3" may continue across a space: hold back the last numeric run
    m = re.search(r"(?:[+€$£]?\d[\d.,:/'\-]*\s?(?:€|EUR|%|°)?\s*)+$", head)
    return m.start() if m else i
