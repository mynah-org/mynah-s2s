"""Session language: one setting, translated for each component.

    session language "it"
      -> streaming ASR prompt     target_lang "it-IT"   (prompt-conditioned multilingual RNNT; "auto" = detect)
      -> Thinker system prompt    "... The caller speaks Italian. Always answer in Italian. ..."
      -> talker prefill           codec language "italian" (Qwen3-TTS codec_language_id table)

The ASR locales come from the ASR model's processor prompt dictionary; the talker names from the Qwen3-TTS
``codec_language_id`` table. Only the languages below are wired; adding one = one row.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SessionLanguage:
    code: str           # ISO 639-1, the session setting
    asr_locale: str     # streaming ASR target_lang
    name: str           # English name used in the Thinker prompt
    talker_lang: str    # Qwen3-TTS codec language key


LANGUAGES = {
    "en": SessionLanguage("en", "en-US", "English", "english"),
    "it": SessionLanguage("it", "it-IT", "Italian", "italian"),
    "fr": SessionLanguage("fr", "fr-FR", "French", "french"),
    "de": SessionLanguage("de", "de-DE", "German", "german"),
    "es": SessionLanguage("es", "es-ES", "Spanish", "spanish"),
    # the rest of the Qwen3-TTS codec language table (talker); the ASR runs language-auto for these
    "pt": SessionLanguage("pt", "pt-PT", "Portuguese", "portuguese"),
    "ru": SessionLanguage("ru", "ru-RU", "Russian", "russian"),
    "zh": SessionLanguage("zh", "zh-CN", "Chinese", "chinese"),
    "ja": SessionLanguage("ja", "ja-JP", "Japanese", "japanese"),
    "ko": SessionLanguage("ko", "ko-KR", "Korean", "korean"),
}


def session_language(code: str) -> SessionLanguage:
    try:
        return LANGUAGES[code.lower()]
    except KeyError:
        raise ValueError(f"unsupported session language {code!r}; wired: {sorted(LANGUAGES)}") from None


def language_instruction(lang: SessionLanguage) -> str:
    """The sentence added to the Thinker's system prompt. English on purpose: the duplex training prompts were
    English; whether the Thinker then WRITES the target language in duplex mode is NOT VERIFIED (DESIGN.md, E1)."""
    if lang.code == "en":
        return "The caller speaks English. Always answer in English."
    return (f"The caller speaks {lang.name}. Their words reach you as a {lang.name} transcript. "
            f"Always answer in {lang.name}, with short spoken sentences.")


def with_language(system_prompt: str | None, lang: SessionLanguage) -> str:
    base = (system_prompt or "").rstrip()
    line = language_instruction(lang)
    return f"{base}\n{line}" if base else line
