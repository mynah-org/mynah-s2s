"""Per-language holding phrases (E1l, PERF R1/G1): said once per turn as soon as the text LLM decides to call a tool,
while the call is generated and executed. Neutral on purpose: they never promise a result."""

from __future__ import annotations

PHRASES = {
    "it": "Un momento, controllo.",
    "fr": "Un instant, je vérifie.",
    "en": "One moment, let me check.",
}


def holding_phrase(lang: str) -> str:
    return PHRASES.get((lang or "en").split("-")[0].lower(), PHRASES["en"])
