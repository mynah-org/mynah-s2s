"""Session languages: everything the engine needs per language, in one table that a package folder can extend.

A language is a ``LangSpec``: ISO 639-1 code, English name (for the LLM's "the caller speaks X"), ASR locale, the
Pocket 24-layer model it comes from, and the fixed phrases the engine says itself (holding phrase, on-hold reminder,
tool-timeout apology, backchannel). The built-in table holds Kyutai's seven official Pocket languages. A language
folder of the talker packages (``<packages>/<code>/language.json``, see docs/frankenstein2-languages.md) registers a
new language or overrides a built-in one, so a language can be added without touching code:

    {"code": "nl", "name": "Dutch", "asr_locale": "nl-NL", "pocket": "dutch_24l",
     "phrases": {"holding": "...", "reminder": "...", "apology": "...", "backchannel": "Mm-hm."}}

Missing phrases fall back to English. What a JSON file cannot add: the number normaliser and the spoken-number
parser (text/), which exist for it / en / fr; other languages speak digits as the talker reads them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

PHRASES = ("holding", "reminder", "apology", "backchannel")


@dataclass
class LangSpec:
    code: str
    name: str
    asr_locale: str
    pocket: str
    phrases: dict = field(default_factory=dict)
    remove_semicolons: bool = False
    tested: bool = False                    # wired and tested end to end in this repository

    def phrase(self, kind: str) -> str:
        return self.phrases.get(kind) or BUILTIN["en"].phrases[kind]

    def to_json(self) -> dict:
        return {"code": self.code, "name": self.name, "asr_locale": self.asr_locale, "pocket": self.pocket,
                "phrases": dict(self.phrases), "remove_semicolons": self.remove_semicolons}


def _b(code, name, loc, pocket, holding, reminder, apology, bc, semi=False, tested=False):
    return LangSpec(code, name, loc, pocket, {"holding": holding, "reminder": reminder, "apology": apology,
                                              "backchannel": bc}, semi, tested)


BUILTIN = {x.code: x for x in (
    _b("it", "Italian", "it-IT", "italian_24l", "Un momento, controllo.", "Ci sono quasi, ancora un attimo.",
       "Mi dispiace, non sono riuscito a recuperare l'informazione in tempo. Può riprovare tra poco?", "Mh-mh.",
       tested=True),
    _b("en", "English", "en-US", "english_2026-09_24l", "One moment, let me check.",
       "Almost there, just a moment more.",
       "I'm sorry, I couldn't retrieve that information in time. Could you try again in a moment?", "Mm-hmm.",
       tested=True),
    _b("fr", "French", "fr-FR", "french_24l", "Un instant, je vérifie.", "J'y suis presque, encore un instant.",
       "Je suis désolé, je n'ai pas pu récupérer l'information à temps. Pouvez-vous réessayer dans un instant ?",
       "Mh-mh.", semi=True, tested=True),
    _b("de", "German", "de-DE", "german_24l", "Einen Moment, ich schaue nach.",
       "Gleich habe ich es, noch einen Moment.",
       "Es tut mir leid, ich konnte die Information nicht rechtzeitig abrufen. Versuchen Sie es bitte gleich noch "
       "einmal.", "Mh-mh."),
    _b("es", "Spanish", "es-ES", "spanish_24l", "Un momento, lo compruebo.", "Ya casi, un momento más.",
       "Lo siento, no he podido obtener la información a tiempo. ¿Puede volver a intentarlo en un momento?",
       "Mh-mh."),
    _b("pt", "Portuguese", "pt-PT", "portuguese_24l", "Um momento, vou verificar.", "Quase lá, só mais um momento.",
       "Desculpe, não consegui obter a informação a tempo. Pode tentar novamente daqui a pouco?", "Hum-hum."),
    _b("nl", "Dutch", "nl-NL", "dutch_24l", "Een moment, ik kijk het even na.", "Bijna klaar, nog heel even.",
       "Sorry, ik kon de informatie niet op tijd ophalen. Kunt u het zo nog eens proberen?", "Mm-hm."),
    # community Pocket voices (frankenstein2/scripts/build_packages.py SOURCES); phrases not reviewed by a native
    # speaker yet
    _b("ko", "Korean", "ko-KR", "", "잠시만요, 확인해 보겠습니다.", "거의 다 됐어요, 조금만 기다려 주세요.",
       "죄송합니다, 정보를 제때 가져오지 못했습니다. 잠시 후 다시 시도해 주시겠어요?", "네."),
    _b("tr", "Turkish", "tr-TR", "", "Bir dakika, kontrol ediyorum.", "Neredeyse bitti, bir dakika daha.",
       "Üzgünüm, bilgiyi zamanında alamadım. Birazdan tekrar dener misiniz?", "Hı-hı."),
    _b("ja", "Japanese", "ja-JP", "", "少々お待ちください、確認します。", "もうすぐです、もう少々お待ちください。",
       "申し訳ありません、時間内に情報を取得できませんでした。少し後でもう一度お試しいただけますか？", "はい。"),
    _b("cs", "Czech", "cs-CZ", "", "Okamžik, ověřím to.", "Už to skoro mám, ještě chvilku.",
       "Omlouvám se, informaci se nepodařilo včas získat. Můžete to za chvíli zkusit znovu?", "Mhm."),
    _b("hi", "Hindi", "hi-IN", "", "एक क्षण, मैं जाँच करता हूँ।", "बस हो गया, थोड़ा और रुकिए।",
       "माफ़ कीजिए, मैं समय पर जानकारी नहीं ला सका। क्या आप थोड़ी देर में फिर से कोशिश करेंगे?", "हम्म।"),
    _b("pl", "Polish", "pl-PL", "", "Chwileczkę, sprawdzam.", "Już prawie, jeszcze chwilka.",
       "Przepraszam, nie udało mi się na czas pobrać informacji. Czy może Pan spróbować ponownie za chwilę?",
       "Mhm."),
    _b("ru", "Russian", "ru-RU", "", "Минутку, сейчас проверю.", "Почти готово, ещё минутку.",
       "Извините, не удалось вовремя получить информацию. Попробуйте, пожалуйста, ещё раз чуть позже.", "Угу."),
    _b("et", "Estonian", "et-EE", "", "Üks hetk, ma kontrollin.", "Peaaegu valmis, veel üks hetk.",
       "Vabandust, ma ei saanud teavet õigeks ajaks kätte. Kas proovite mõne aja pärast uuesti?", "Mhm."),
)}

_REG: dict = dict(BUILTIN)


# What was checked per language (docs/frankenstein2-languages.md): "e2e" = the end-to-end test from recorded caller
# samples passed (make test-f2-e2e: the caller's spoken lines heard by the ASR, every line answered aloud, the
# catalogue tool called; several languages at once, the talker's packages swapping); it/en/fr also have the tool-call
# suites and multi-session gates. "suite" = the language suite only (talker -> ASR, LLM alone, one spoken session);
# "weak-voice" = the suite's voice check failed. A language folder added later is "unchecked".
STATUS = {c: "e2e" for c in ("it", "en", "fr", "de", "es", "pt", "nl", "ko", "tr", "ja", "cs", "hi", "pl", "ru", "et")}


# The turn-taking GRU's endpoint threshold per language (ears/rules.py Thresholds.ep_thr, default 0.915), from
# frankenstein2/lab/endpoint_bench.py on the recorded caller lines (f2c-070): raising it for these languages cut the
# turns ended inside a 0.3-0.9 s pause in the middle of a sentence by about 40 %, for 100-320 ms more end-of-turn
# latency. Italian keeps 0.85 (half the latency; its early endpoints are recovered by the continuation guard), English
# and French keep the default: those three have the tool-call suites and multi-session gates, measured with these
# values. --ep-thr LANG=x overrides.
EP_THR = {"it": 0.85, "de": 0.95, "es": 0.95, "nl": 0.95, "ko": 0.95, "pl": 0.95, "ru": 0.95, "tr": 0.95,
          "et": 0.975}


def status(code: str) -> str:
    return STATUS.get(code, "unchecked")


def get(code: str) -> LangSpec:
    """The language's spec; an unknown code gets English phrases and its code as locale (ASR auto-detect)."""
    s = _REG.get(code)
    if s is None:
        return LangSpec(code, code, code, "", {})
    return s


def phrase(code: str, kind: str) -> str:
    return get(code).phrase(kind)


def known() -> list:
    return sorted(_REG)


def register(spec: LangSpec) -> LangSpec:
    """Add or override a language (an override keeps the built-in phrases it does not redefine)."""
    old = _REG.get(spec.code)
    if old is not None:
        spec.phrases = {**old.phrases, **{k: v for k, v in spec.phrases.items() if v}}
        spec.tested = old.tested
    _REG[spec.code] = spec
    return spec


def from_json(path) -> LangSpec:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    code = d["code"]
    base = BUILTIN.get(code)
    return LangSpec(code=code, name=d.get("name") or (base.name if base else code),
                    asr_locale=d.get("asr_locale") or (base.asr_locale if base else code),
                    pocket=d.get("pocket") or (base.pocket if base else ""),
                    phrases={k: v for k, v in (d.get("phrases") or {}).items() if k in PHRASES},
                    remove_semicolons=bool(d.get("remove_semicolons", base.remove_semicolons if base else False)))


def reset() -> None:
    """Tests: back to the built-in table."""
    _REG.clear()
    _REG.update(BUILTIN)
