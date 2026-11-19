# Frankenstein v2: languages

A session has one language, chosen at start. The ears, the LLM and the talker are separate models, so a language
works when all three cover it. The ASR covers 40 locales and Qwen3.5 201 languages; the talker, Kyutai Pocket TTS,
is the limit.

- **15 languages have a talker voice with an open licence** that the ASR and the LLM also cover:
  - Kyutai's official seven: Italian, English, French, German, Spanish, Portuguese, Dutch;
  - eight community models: Korean, Turkish, Japanese, Czech, Hindi, Polish, Russian, Estonian.
- **All 15 pass the end-to-end test from recorded samples** (`make test-f2-e2e`, opt-in, GPU). A simulated caller's
  recorded lines (`webui/f2/samples`, the store scenario, three lines per language) are streamed in real time, three
  languages at once, while the talker's VRAM cache swaps language packages. Per language: the ASR heard the lines,
  every line got a spoken answer, the catalogue tool was called. Each language speaks with its own voices
  (`--voice-map`): Kyutai's predefined voices for the official seven, the community models' published voices
  otherwise, picked by how well the ASR understands them (`frankenstein2/lab/voice_audition.py`).
- **Italian, English and French** also have tool-call suites and multi-session gates, and the other demo scenarios
  (weather, travel, knowledge, restaurant; `frankenstein2/scenarios.py`). Known LLM limits there, with Qwen3.5-4B: in
  French the restaurant agent sometimes books with a name the caller never said; a booking repeated with the same
  details keeps one table, and a date in the past is refused by the tool.
- The streaming ASR sometimes drops the first word or two of a short turn ("Perfekt, vielen Dank" heard as "vielen
  Dank"): sporadic, in any language, not caused by silence, turn resets or the stream's age (replays in
  `frankenstein2/lab/asr_onset_replay.py`, `asr_silence_probe.py`). The engine notices it (voice before the first
  transcribed word) and hears such a turn again offline before the LLM gets it (`--asr-second-pass`, on by default;
  `frankenstein2/DESIGN.md`).
- The language suite (`make f2-lang-suite`) checks the parts one by one: the talker speaks a question and the ears
  must understand it; the LLM alone must call the catalogue tool and answer in the language; one spoken session.
- **Adding a language is copying a folder**, see "Adding a language" below.

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
ends turns in the other 14 languages (the end-to-end test above). `frankenstein2/lab/endpoint_bench.py` measures it
per language on the recorded caller lines: no missed endpoint, a median latency after the speech ends of 300-640 ms
(English and Portuguese slower, about 1 s: there the GRU rarely reaches its threshold and the 1.2 s silence rule ends
the turn), and a pause of 0.6-0.9 s inside a sentence often ends the turn too early. Per-language thresholds
(`frankenstein2/langs.py` EP_THR) cut those early ends by about 40 % in the languages they apply to; when the user goes
on speaking within 2 s the engine cancels the reply and merges the two parts of the turn.

**Brain: Qwen3.5-4B (201 languages and dialects).** Source: [model card](https://huggingface.co/Qwen/Qwen3.5-4B).
We measured tool-call quality in Italian, English and French only.

**Talker: Kyutai Pocket TTS.** Two model families share one architecture per family: **24l** (24 layers, the
engine's default) and **6l** (6-layer students: smaller and faster, lower quality).

| code | language | ASR tier | talker source | family | licence | status |
|---|---|---|---|---|---|---|
| it | Italian | ready | Kyutai official | 24l, 6l | CC-BY-4.0 | **tested** |
| en | English | ready | Kyutai official (the base) | 24l, 6l | CC-BY-4.0 | **tested** |
| fr | French | ready | Kyutai official | 24l, 6l | CC-BY-4.0 | **tested** |
| de | German | ready | Kyutai official | 24l, 6l | CC-BY-4.0 | tested end to end (recorded samples) |
| es | Spanish | ready | Kyutai official | 24l, 6l | CC-BY-4.0 | tested end to end (recorded samples) |
| pt | Portuguese | ready | Kyutai official | 24l, 6l | CC-BY-4.0 | tested end to end (recorded samples) |
| nl | Dutch | ready | Kyutai official | 24l, 6l | CC-BY-4.0 | tested end to end (recorded samples) |
| ko | Korean | ready | [seastar105/pocket-tts-korean-300m](https://huggingface.co/seastar105/pocket-tts-korean-300m) | 24l | CC-BY-4.0 | tested end to end (recorded samples) |
| tr | Turkish | ready | [alperiox/pocket-tts-turkish](https://huggingface.co/alperiox/pocket-tts-turkish) | 24l, 6l | MIT | tested end to end (recorded samples) |
| ja | Japanese | ready | [takuma104/pocket-tts-ja-cc](https://huggingface.co/takuma104/pocket-tts-ja-cc) | 24l, 6l | CC-BY-4.0 | tested end to end (recorded samples) |
| cs | Czech | broad | [vvolhejn/pocket-tts-czech](https://huggingface.co/vvolhejn/pocket-tts-czech) | 6l | CC-BY-4.0 | tested end to end (recorded samples) |
| hi | Hindi | ready | [saryps-labs/pocket-tts-hindi](https://huggingface.co/saryps-labs/pocket-tts-hindi) | 6l | CC-BY-4.0 | tested end to end (recorded samples) |
| pl | Polish | broad | [shefowl/pocket-tts-polish-6l](https://huggingface.co/shefowl/pocket-tts-polish-6l) | 6l | CC-BY-4.0 | tested end to end (recorded samples) |
| ru | Russian | ready | [ArtShtorm/Shtorm-PocketTTS-RU](https://huggingface.co/ArtShtorm/Shtorm-PocketTTS-RU) | 6l | CC-BY-4.0 | tested end to end (recorded samples) |
| et | Estonian | broad | [cbentes/pocket-tts-estonian](https://huggingface.co/cbentes/pocket-tts-estonian) | 6l | CC-BY-SA-4.0 | tested end to end (recorded samples) |

**Left out:**
- Persian, Indonesian, Welsh, Gujarati and Tamil have community voices but are not in the ASR.
- A Persian and a Japanese model are non-commercial (CC-BY-NC).
- Arabic, Vietnamese, Ukrainian, Chinese, Swedish and others are in the ASR, but we found no Pocket voice for them.
- Several community repositories are ONNX / CoreML / MLX exports only, which the torch runtime does not load.

In the suite, the catalogue search of the test scenario only knows the sofa under a few names, so in Japanese,
Korean, Dutch, Estonian and Turkish the tool call is right but the product is "not found". That is the test data, not
the engine. The LLM gets the session language again after each tool result. Without that reminder, a Russian
session once answered in English after the (English) tool result.

The fixed phrases of the community languages (holding, reminder, apology, backchannel) were written by us, not by
native speakers. Each one can be corrected in its `language.json` without touching code.

## The language folder

The talker reads every language from one folder. The engine scans it; there is no configuration:

```
pocket-tts-deltas/
  en/
    language.json
    24l/base/                  English 24-layer, verbatim: the shared base of every 24l language (and English)
    6l/base/                   English 6-layer: the base of the 6l languages
  it/
    language.json              name, ASR locale, fixed phrases
    24l/int4/  24l/int8/  24l/bf16/     Italian as a delta against en/24l/base, at three precisions
    6l/int4/   6l/int8/   6l/bf16/      the same for the 6-layer family
  cs/
    language.json
    6l/int4/ ...               a language that only exists as a 6-layer model
```

- **Packages.** Each `<code>/<family>/<precision>/` is a `pocket_swap` package. It holds `manifest.json` (source,
  licence and base identity, sizes, error), the transformer backbone as a delta against the family's English base,
  the language's own text embedding and head (stored as they are: vocabularies differ), and its tokenizer
  (`tokenizer.json`, or `tokenizer.model` for SentencePiece).
- **Precisions.** `int4` is the default: about the size of a 4-bit model, and much more accurate, because only the
  difference from English is quantised. `int8` is closer still. `bf16` is the exact difference.
- **At run time.** The GPU holds one copy of each family's English backbone. Every language adds only its delta;
  the fused kernel computes x·(W_en + Δ) without ever writing the language's full weights.
- **Picking a package.** For each language the engine uses the first family in `--pocket-families` (default
  `24l,6l`) that has the precision `--pocket-precision` (default `int4`).
- **Mimi.** The Mimi codec and its voice-cloning encoder are identical in every language. They come from Kyutai's
  English checkpoint, which is gated: accept Kyutai's terms on the model page.

**Download.** The packages are published at
[pocket-tts-deltas](https://huggingface.co/gabrione/pocket-tts-deltas); `F2_PACKAGES_REPO` uses another repo.
`make f2-setup F2_LANGS=it,en,fr,ko` fetches the languages you list in their best family, the needed English bases,
and the gated Mimi checkpoint. `make f2-plan ...` shows the sizes first.

**Build it yourself.** `python frankenstein2/scripts/build_packages.py --out pocket-tts-deltas` rebuilds the whole
folder from the sources in its `SOURCES` table:
- Kyutai's official languages come from the non-gated `kyutai/pocket-tts-without-voice-cloning`. Their weights are
  identical to the gated ones, and no package contains Mimi, so no voice-cloning weights are redistributed.
- Each package records its source's licence and attribution, and is verified against its source after the build.

**The cache at run time.** The engine keeps up to `--talker-vram-langs` languages on the GPU (default 4) and
`--talker-host-langs` packages in pinned host RAM (default 8). The rest stay on disk.
- A session whose language is not loaded waits for it; every other session keeps talking. The load runs on its own
  thread and CUDA stream, never on the scheduler. A language comes from disk in well under a second; only the
  very first one takes longer, because it also encodes the cloning voice, which every later language reuses.
- The language of an open session is never evicted, also between its replies. The VRAM limit is therefore soft:
  with more concurrent languages than `--talker-vram-langs`, all of them stay loaded until their sessions close.
- Sessions in different languages share one talker step (`frankenstein2/tts/cross_step.py`):
  - **One launch per op for all rows.** The small per-language tensors (norms, flow head, ...) are copied into
    a language table in VRAM the first time a language speaks. Each row reads its own language's tensors by index.
  - **Backbone.** Each backbone linear is one launch of a grouped kernel: the English base tile is read once for
    all languages, and each language's rows add their own delta.
  - **Attention and Mimi** run once for all rows.
  - **CUDA graphs.** The frame is captured as a CUDA graph per row count (2, 4, 8, 16).
  - **Speed.** Under the talker stress (8 sessions, 30 rotating languages) talker turns are shorter and audio gaps
    fewer than with one step per language.
  - **Correctness.** The result is not bit-identical to each language stepping alone (gathered norms, batched
    small matmuls). The GPU test checks that the difference is small, that the graph equals the eager step, and
    that the ears understand eight languages spoken at once.
  - **Other modes.** `POCKET_TALKER_STEP=many` steps the languages together with bit-identical latents (per-language
    small ops); `own` steps each language on its own.
- All languages share one pool of per-session talker state per family.
- `frankenstein2/lab/talker_stress.py` stresses it with end-to-end sessions over many languages
  (`--talker-rotate`).

## Adding a language

Copy a folder with this layout into the language folder; the server picks it up at the next session that asks for
it, without a restart:

1. **Choose the code.** Use its ISO 639-1 code (`sv`, `ko`, ...), and check that the ASR has its locale (table
   above).
2. **Build the package.** You need a Pocket TTS model of that language in one of the two families: a 24-layer or
   6-layer model with the same architecture as Kyutai's English one (fine-tuned from it, or not).
   - Add a row to `SOURCES` in `frankenstein2/scripts/build_packages.py` (repo, file, tokenizer, licence) and
     run it with `--langs <code>`.
   - Or call `pocket_swap`'s `build_package` directly, with `en/<family>/base` as the base.
3. **Write `<code>/language.json`:**

   ```json
   {"code": "sv", "name": "Swedish", "asr_locale": "sv-SE",
    "phrases": {"holding": "Ett ögonblick, jag kollar.", "reminder": "Snart klart, ett ögonblick till.",
                "apology": "Tyvärr kunde jag inte hämta informationen i tid. Kan du försöka igen om en stund?",
                "backchannel": "Mm."}}
   ```

   `name` goes into the LLM's instruction ("the caller speaks Swedish"). `asr_locale` selects the ASR prompt.
   Missing phrases fall back to English.
4. **Optional, for quality.** A per-language end-of-turn threshold (`--ep-thr sv=0.88`), checked on real speech.
   The number normaliser (`frankenstein2/text/normalize.py`, digits to words for the talker) exists for it / en /
   fr only. Other languages pass digits to the talker as they are.
5. **Before calling it supported**, run a gate in that language: the model's text, an ASR of the spoken audio and
   the tool ground truth.
