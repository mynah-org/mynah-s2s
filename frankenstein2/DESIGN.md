# Frankenstein v2: design

Frankenstein v2 is a multilingual full-duplex speech-to-speech engine built from separate parts:
- **ears**: one Nemotron 3.5 ASR streaming encoder plus the Tyger turn-taking GRU;
- **brain**: a text LLM with tools;
- **talker**: a swappable TTS.

There is no duplex VoiceChat Thinker. It is a separate engine (`ENGINE=frankenstein2`), so Frankenstein v1
(`frankenstein/`) and the VoiceChat engine (`v2/`) are unchanged: v2 imports their modules read-only and edits
no shared code.

```
mic 16 kHz ─► EarsLink (320 ms hops, 1 in flight) ─► ears process ─► ENDPOINT / BARGE / BACKCHANNEL + transcript
    ▲                                                     │
    └── 2nd ear: agent audio as released (delayed, 16 kHz)│
                                                          ▼
Playout ◄─ Talker plugin ◄─ normaliser ◄─ Chunker ◄─ LLM plugin ◄─ Controller (single writer, epochs)
 (real-time release, flush on cancel, played cursor)
```

## 1. Ears (`ears/`)

**One model, two uses** (`stream.py`, `window.py`). The Nemotron 3.5 ASR 0.6B weights come from the Tyger bundle.
They run in their own process (`server.py`), because that model needs a newer transformers than the engine's
environment. Each session has its own language, encoder caches, GRU state and rules.

- **Transcript.** True cache-aware streaming with lookahead k=3: 320 ms chunks, `chunked_limited` attention, and
  causal conv caches carried across chunks.
  - Mel frames per chunk: 25 for the first chunk, then 32 per chunk.
  - Only the user row is decoded. The greedy RNN-T keeps its predictor state across hops. A language tag such as
    `<it-IT>` is dropped from the text without resetting the predictor: a reset there deleted the next words.
  - The language prompt is always explicit (`it` / `fr` / `en`). There is no auto-detection by default.
  - Checked against an offline pass over the whole track with the same lookahead: the last layer matches to within
    4e-6 (max abs) and layer 15 has cosine 1.0. The transcripts are equal except for one repeated short word.
- **GRU features: the trained regime.** Every hop, the last 8 s of each channel is re-encoded to layer 15 with
  lookahead 0. The GPU log-mel, the all-zero-window cache and the hook that stops the encoder at layer 15 are
  ported from Tyger.
  - The plan was to feed the GRU from the streaming encoder's layer 15. **That was measured and rejected.** On the
    52 s Italian session track, streaming features at k=3 (and at k=0) keep `p_end` at 0.72–0.88 after three of the
    four turn ends: 1 endpoint out of 4 at the default threshold.
  - The window regime reaches 0.89–0.98 there, which gives 4 out of 4, identical to Tyger's own regime.
  - The earlier offline estimate ("small drift at k=3") did not hold for this continuous track. A GRU fine-tune on
    streaming features would remove the second pass; that is NOT DONE.
- **GRU** (`turntaker.py`): Tyger's module, fp32, TF32 off. Its inputs are the user and agent layer-15 features, the
  user frame energy, and the previous state forced by the controller's mode (`listen` or `speaking`).
- **Rules** (`rules.py`): Tyger's per-frame rules in pure Python.
  - ENDPOINT: `p_end ≥ ep_thr` (0.915) for 2 frames, after at least 350 ms of voiced speech.
  - BARGE: `yield ≥ 0.4` for 2 frames, with `onset_s`.
  - BACKCHANNEL: `bc ≥ 0.4` for 2 frames, then a 1.2 s refractory period.
  - 6 warm-up frames after each reset or mode switch.
  - Thresholds can be set per language (`--ep-thr it=0.85`). On the Italian session, 0.85 roughly halves the
    endpoint latency; the extra early fires fall in inner pauses and are recovered by the continuation guard and
    the unheard-barge merge.
- **2nd ear.** The agent channel is the agent audio exactly as the playout released it, delayed by the client delay
  and resampled 24→16 kHz with a zero-latency resampler. Backchannel audio is never in it.

## 2. Controller (`control/`)

The controller is the single writer of turn state. Sources post intents stamped with the 80 ms step. Each tick
sorts the due intents by priority, then source, then step, then arrival order, and applies them, so permutations
give the same actions. Priority: barge > cancel / user-continues > tool > text > done > played-out > endpoint >
backchannel. Answers that belong to a cancelled epoch or request are dropped and traced.

The states are LISTENING, THINKING and SPEAKING. The ears are in `speaking` mode from the turn open until the turn's
audio has played out, as in Tyger.

| intent | action |
|---|---|
| ENDPOINT | open the turn and request a reply. The user text is the streaming transcript since the last boundary. |
| BARGE | cancel: new epoch, LLM cancel, talker cancel, playout flush, ears reset. The history keeps only the heard text, marked ` —`. |
| USER_CONTINUES | ASR words after the endpoint frame, within 2 s and before any reply audio, mean the endpoint was early (a pause inside the turn): cancel and merge the text into the next turn. Added because the GRU fired inside U1 of the Italian track. |
| LLM_DECIDED | play the holding phrase, once per turn and only if nothing was spoken yet. |
| LLM_TOOL | write the tool record into the history. It stays even after a barge, because the tool ran. |
| PLAYED_OUT | close the turn and reset the ears. |
| BACKCHANNEL | play a short audio while listening. It does not go into the 2nd ear or the history. |

Words the user speaks over the agent are kept only if they are part of a barge, from its onset. Otherwise they are
dropped and traced as `asr.drop`.

## 3. LLM plugins (`llm/`)

The interface is `submit(rid, messages)`, `cancel(rid)`, `poll()` and `close()`. `poll()` returns events:
`text` / `decided` / `tool` / `done`. There is one instance per session; instances may share one loaded model.

- **Nano 9B v2** (`nano.py`): v1's worker, with the tool step, schema-constrained calls and brief replies.
  - Precision `bf16` or `int4-decode`. `int4-decode` adds INT4 copies to the bf16 weights for the one-row decode
    steps (v1 G6/G8).
  - A lock serialises requests across sessions, because the INT4 prefill scratch is shared.
- **Qwen3.5 2B/4B** (`qwen.py` + `qwen_server.py`), for an A/B later: XML tool-call parser, a splitter that keeps
  call markup out of speech, and tool rounds. The process path is NOT VERIFIED.
- **No serving framework in the engine.** vLLM is used only in the lab, to compare models. The engine runs torch,
  our own code, and HF transformers / tokenizers for model definitions and loading.
  - **Today:** Nano runs on v1's own path: explicit hybrid KV / SSM cache, a one-row decode CUDA graph, the INT4
    decode dispatch, and constrained calls. The Qwen3.5 process uses plain HF `generate` as a stand-in.
  - **What our own path still needs, per candidate:**
    - *Qwen3.5 dense, 2B / 4B* (hybrid: gated-delta-net linear attention plus full-attention layers): a per-session
      state for the linear-attention layers and a KV cache for the attention layers (the same split as Nano's
      Mamba + attention cache); a decode graph per batch bucket; weight-only INT8 / INT4 / FP8 linears behind one
      dispatch (one-row GEMV, multi-row dequant + GEMM, as v1's `int4_dispatch`); the XML call grammar in the
      constrained decoder.
    - *Qwen MoE*: top-k router; grouped expert GEMMs over the tokens routed to each expert (batched across
      sessions), with a portable kernel (Triton or torch `_grouped_mm` where available) and a reference loop fallback.
      FP8 experts are native on sm_89 and sm_12x; INT4 / INT8 weight-only need dequant-in-kernel. Each path is
      chosen by arch detection, with the reference implementation as the fallback, and each is verified on both sm_89
      and sm_121 before it is promoted.
    - *All candidates*: batching across sessions (one row per active request, continuous decode), a per-session
      KV / state slot, and the tool-suite identity check as the promotion gate for every precision.
- **Tool arguments** (`text/spoken_numbers.py`), in the session language:
  1. Spoken digits and capital letters become codes: "X Y Z uno due tre" → "XYZ123", "uno zero uno due uno" → "10121".
  2. Code-like values are then grounded on what the caller said. Every model we tried garbles "uno zero uno due uno"
     on its own, for example to "01201". A value that is not among the caller's codes is replaced by the caller's
     code that is an anagram of it or within edit distance 2, if exactly one such code exists.

## 4. Text to speech (`text/`, `tts/`, `audio/`)

- **Chunker policy per talker:**
  - `sentence`: whole sentences, never cut inside a number. For Pocket.
  - `whole`: the whole reply at the end.
  - `progressive`: commit text up to `split_safe`. For Flash.
- **Normaliser** (ported from the lab): numbers, prices, times, dates, codes and phone numbers become words in the
  session language. It runs before any other text clean-up.
- **Talkers:**
  - Pocket (`pocket_server.py`): one process for all sessions, language models kept in an LRU cache, and a
    per-session voice (zero-shot clone from a reference wav).
  - Flash (v1's process, one session at a time).
  - pocket_swap (language slots) plugs in later behind the same interface. VoxCPM2 is not wired.
- **Playout:** the engine releases exactly 80 ms of audio per input step.
  - A cancel only flushes this queue, so no audio is generated ahead in the transport.
  - The played cursor is exact up to the client delay. It drives the 2nd ear and the heard text after a barge (whole
    segments played, plus the played share of the current segment, cut at a word).

## 5. Server, sessions, languages

- `server/engine.py` implements V2's Engine contract, so V2's Session and WebSocket server run it unchanged.
- `session.start` carries `language` and `voice`; that one value drives the whole chain for the session.
- `server/serve.py` reuses v1's start-field wrapper.
- All state is per session (`Pipeline`):
  - ears session and language;
  - controller, epochs and conversation;
  - LLM worker, tool registry and ledger;
  - chunker and normaliser locale;
  - talker language and voice;
  - playout.
- `tests/test_multi_session.py` runs IT, EN and FR sessions interleaved on shared backends, with a barge in one of
  them. Each session's history, LLM requests, tool calls, talker requests, audio bytes and ears hops come out equal
  to the same session run alone.

## 6. Placement and memory (`server/build.py`)

Every placement choice is an option: ears device, Nano precision, talker device, Pocket languages kept loaded, and
CPU threads per process. Each process reports its own CUDA allocated / reserved / peak and host RSS. The engine does
not assume unified memory.

There are no custom kernels in v2 itself. Arch-specific paths are inherited only from v1's optional Nano INT4
(torch tinygemm, sm80+) and decode graph. Measured footprints and which targets fit are in the local performance
notes, not here.

## 7. Tracing (`tracing.py`)

The trace records:
- every GRU frame (`p_end`, `yield`, `bc`, `hazard`, energy, mode);
- hops with compute time and lag;
- streaming ASR pieces;
- every controller intent and action;
- LLM text and tool events (raw, normalised and grounded arguments);
- talker segments (text and spoken form), first audio, cancels and flushes;
- what was heard after a barge.

## 8. Known limits
- End-to-end runs (Italian session, fixed and reactive user): tools with parameters right for `find_customer`, but
  the text LLM answers the catalogue question without calling `search_catalog`; barge-in by the yield head is slow
  (seconds, Tyger's known weakness off English); cache-aware GRU features would need a fine-tune.

- The GRU was trained on English. In synthetic tests, Italian and French endpointing work; barge-in recall is
  weaker off English.
- There is no AEC. The 2nd ear assumes the client cancels the echo.
- Words spoken over the agent without a detected barge are dropped. This is the same as Tyger.
- Sessions are served one hop or one request at a time. There is no cross-session batching yet, but all state is
  already per session.
