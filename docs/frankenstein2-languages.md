# Frankenstein v2: languages

A session has one language, chosen at start. The ears, the LLM and the talker are separate models, so a language
works when all three cover it and the engine has its small per-language pieces. In short:

- **Tested end to end:** Italian, English, French.
- **Possible with official models, not wired yet:** German, Spanish, Portuguese, Dutch.
- **Possible with community voices, not verified:** Czech, Polish, Estonian, Hindi, Korean.
- **Not possible with this ASR:** languages outside its 40 locales.

## What each model covers

**Ears: NVIDIA Nemotron 3.5 ASR Streaming 0.6B (40 locales).** Source:
[model card](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b). The engine sets the locale from the
session language and never auto-detects.

| tier (model card) | locales |
|---|---|
| transcription-ready | en-US, en-GB, es-US, es-ES, fr-FR, fr-CA, it-IT, pt-BR, pt-PT, nl-NL, de-DE, tr-TR, ru-RU, ar-AR, hi-IN, ja-JP, ko-KR, vi-VN, uk-UA |
| broad coverage | pl-PL, sv-SE, cs-CZ, nb-NO, da-DK, bg-BG, fi-FI, hr-HR, sk-SK, zh-CN, hu-HU, ro-RO, et-EE |
| adaptation-ready (needs fine-tuning) | el-GR, lt-LT, lv-LV, mt-MT, sl-SI, he-IL, th-TH, nn-NO |

The turn-taking GRU in the same bundle was trained and evaluated on English (Tyger model card). In our tests it also
ends Italian and French turns correctly, with a per-language threshold. Any other language needs the same check.

**Brain: Qwen3.5-4B (201 languages and dialects).** Source: [model card](https://huggingface.co/Qwen/Qwen3.5-4B).
We measured the tool-call quality in Italian, English and French only.

**Talker: Kyutai Pocket TTS.**

- **Official** ([kyutai/pocket-tts](https://huggingface.co/kyutai/pocket-tts), CC-BY-4.0): English, French, German,
  Portuguese, Italian, Spanish, Dutch. Each has a 6-layer and a 24-layer model; the engine uses the 24-layer ones.
  The voice-cloning repository is gated (accept the terms); tokenizers come from
  [kyutai/pocket-tts-without-voice-cloning](https://huggingface.co/kyutai/pocket-tts-without-voice-cloning).
  Reference voices: [kyutai/tts-voices](https://huggingface.co/kyutai/tts-voices). Each folder has its own licence;
  some are non-commercial. The setup script takes one CC-BY-4.0 clip.
- **Community models** listed on Kyutai's model card, not by Kyutai, not tested here:
  [Czech](https://huggingface.co/vvolhejn/pocket-tts-czech), [Hindi](https://huggingface.co/saryps-labs/pocket-tts-hindi),
  [Korean](https://huggingface.co/seastar105/pocket-tts-korean-300m), [Persian](https://huggingface.co/mehdi-hf/pocket-tts-farsi),
  [Indonesian](https://huggingface.co/anak10thn/pocket-tts-indonesian), [Estonian](https://huggingface.co/cbentes/pocket-tts-estonian),
  [Welsh](https://huggingface.co/EryriLabs/pocket-tts-cymraeg), [Polish](https://huggingface.co/shefowl/pocket-tts-polish-6l).
  For more, search the Hub for "pocket-tts" and check each licence.

## Combinations

| language | ears (locale tier) | LLM | talker | engine pieces | status |
|---|---|---|---|---|---|
| Italian | it-IT, ready | ✓ | official | ✓ | **tested** |
| English | en-US / en-GB, ready | ✓ | official | ✓ | **tested** |
| French | fr-FR / fr-CA, ready | ✓ | official | ✓ | **tested** |
| German | de-DE, ready | ✓ | official | partial (holding, reminder, apology) | possible, not wired |
| Spanish | es-ES / es-US, ready | ✓ | official | partial (holding, reminder, apology) | possible, not wired |
| Portuguese | pt-PT / pt-BR, ready | ✓ | official | – | possible, not wired |
| Dutch | nl-NL, ready | ✓ | official | – | possible, not wired |
| Czech, Polish, Estonian | broad coverage | ✓ | community | – | not verified |
| Hindi, Korean | ready | ✓ | community | – | not verified |
| Persian, Indonesian, Welsh | not in the ASR | ✓ | community | – | not possible with this ASR |
| Turkish, Russian, Arabic, Japanese, Vietnamese, Ukrainian, Chinese, … | ready / broad | ✓ | none found | – | needs a TTS |

"Community" means: the model exists, but nobody has checked here that our Pocket runtime loads it (some are 6-layer
or a different size) or how it sounds.

## Downloading languages

```bash
python frankenstein2/scripts/setup_models.py --langs it,en,fr              # default
python frankenstein2/scripts/setup_models.py --langs de,es,pt,nl --dry-run # files and sizes, nothing downloaded
```

Each official 24-layer Pocket language is 0.7–0.9 GB. The ears bundle (about 2.4 GB) and the LLM (about 8.7 GB
in bf16 on disk; the engine keeps int8 weights) are shared by every language.

**Many languages on one GPU: `pocket_swap`.** Each loaded Pocket language holds its own weights. `pocket_swap/`
keeps one shared base checkpoint plus a small package per language: a quantised delta or quantised weights. It moves
languages between GPU, host RAM and disk as sessions need them. Packages are built locally from the official
checkpoints (`python -m pocket_swap.cli build ...`, see [`pocket_swap/README.md`](../pocket_swap/README.md)).
Prebuilt packages are not published yet.

## Adding a language

Each step is one row or table entry:

1. **Talker:** the Pocket folder name in `frankenstein2/runtime.py` (`LANGS`) and in
   `frankenstein2/scripts/setup_models.py` (`POCKET`).
2. **Ears:** the locale prompt, and an end-of-turn threshold checked on real speech (`--ep-thr xx=...`).
3. **Phrases:** the holding phrase, reminder and backchannel in `frankenstein2/core.py`; the timeout apology and the
   "I'm checking" pattern in `frankenstein2/llm/qwen.py`.
4. **Prompts:** the reply brief and tool policy in `frankenstein2/llm/nano.py`, and the language row in
   `frankenstein/frankenstein_s2s/lang.py`.
5. **Text:** numbers, dates and currencies to words for the talker (`frankenstein2/text/normalize.py`), and spoken
   numbers to digits for tool arguments (`frankenstein2/text/spoken_numbers.py`).
6. **Serve it:** the session language list (`Resources.languages`). Then run a gate in that language, checking the
   model text, an ASR of the spoken audio and the tool ground truth.
