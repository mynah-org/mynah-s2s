"""Does the user's text so far want a continuation? A second check on the turn-taking GRU's endpoint. Model-free;
unit-tested.

The GRU ends a turn on a pause of 0.6-0.9 s inside a sentence about half of the time (frankenstein2/lab/
endpoint_bench.py, all 15 languages). A text that ends with a comma, or with a word that cannot end a sentence (a
conjunction, an article, a preposition: "e", "und", "and", "the", "de"...), is most likely cut in the middle: the
pipeline then holds the endpoint for a short while and lets the turn go on if more words come (core.py).

Japanese and Korean are left out: their sentence-final particles would read as unfinished.
"""

from __future__ import annotations

WORDS = {
    "it": "e ed o oppure ma però perché quindi che di del dello della dei degli delle a al allo alla ai agli alle da "
          "dal dalla in nel nella con su sul sulla per tra fra il lo la i gli le un uno una se come anche",
    "en": "and or but so because that which who the a an of to with for in on at from by as if than my your our "
          "their his her its",
    "fr": "et ou mais donc car parce que qui de du des le la les un une à au aux pour avec en sur dans par si mon ma "
          "mes ton ta votre notre",
    "de": "und oder aber denn weil dass ob wenn der die das den dem des ein eine einen einem einer mit für von zu "
          "zum zur in im auf an am bei nach aus über unter mein meine ihr ihre",
    "es": "y e o u pero porque que de del el la los las un una unos unas a al con para en por si mi mis tu su sus "
          "como",
    "pt": "e ou mas porque que de do da dos das o a os as um uma com para em no na nos nas por pelo pela se meu "
          "minha seu sua como",
    "nl": "en of maar want omdat dat die de het een met voor van naar in op aan bij als mijn uw",
    "pl": "i a ale lub albo czy że bo w we z ze na do od dla o po przy za mój moja",
    "cs": "a i ale nebo že protože když v ve na s se z ze do pro od o k ke po u můj moje",
    "ru": "и а но или что чтобы потому если в во на с со к ко по для от о об у из мой моя",
    "et": "ja või aga et sest kui ning ka",
    "tr": "ve veya ama fakat çünkü ile için bir bu şu",
    "hi": "और या लेकिन कि क्योंकि के की का में से को पर एक",
}
_SETS = {k: set(v.split()) for k, v in WORDS.items()}


def wants_more(text: str, lang: str) -> bool:
    """True when ``text`` ends with a comma, a colon or a dash, or with a word of ``lang`` that cannot end a
    sentence. A final full stop, question or exclamation mark means the speaker closed it: False."""
    t = (text or "").strip()
    if not t:
        return False
    if t[-1] in ",;:-–—":
        return True
    if t[-1] in ".?!。？！।":
        return False
    words = _SETS.get(lang)
    if not words:
        return False
    last = t.lower().split()[-1].strip("\"'«»“”„()[]")      # split on spaces: \w misses Devanagari vowel signs
    return last in words
