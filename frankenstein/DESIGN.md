# Frankenstein S2S: design

A separate multilingual speech-to-speech engine assembled from existing parts, with no training:

- a NeMo multilingual streaming ASR, which gives the authoritative transcript;
- the VoiceChat Thinker, kept frozen. It is used twice: its duplex frame loop on the original audio decides timing,
  and a text-mode pass of the same weights decides content;
- a talker adapter;
- the Sophon Flash talker (Qwen3-TTS-12Hz-1.7B structure, fine-tuned weights).

The V2 engine (`v2/`) and the labs (`labs/`) are not modified. Nothing here was run on a GPU.

Status words used below:

- **VERIFIED in source**: checked in the files cited, at the lines given.
- **Measured earlier**: an earlier internal experiment. Its harness is cited where it is public (mostly
  `labs/multilingual/scripts/multilingual/`); its result tables are not in this repository.
- **NOT VERIFIED**: a hypothesis, to be tested by the gates in §10.

## 0. Decision summary

1. **Primary: M1-force.** Audio decides timing, a text-mode pass decides content, and the agent channel is
   teacher-forced. All three parts already exist.
   - **Timing.** The duplex frame loop runs on the **original user audio**. It alone decides when to speak (BOS),
     pause (PAD), stop (EOS) and yield to barge-in. Its own words are not used.
   - **Content.** At its BOS, the **same Thinker weights in text mode** write the reply. Text mode = the
     `text_to_text` path the model was co-trained on: Nano chat template, `/no_think`, id 12 banned. The history is
     the authoritative multilingual transcripts plus the replies actually spoken.
   - **Forcing.** The reply is **teacher-forced into the agent text channel**, one id per 80 ms frame. NVIDIA's
     wrapper already does this for its own "redirect" phrase (§4.1), and training feeds that channel gold text. The
     same burst goes to the adapter and the talker, with the session language.
2. **Measured basis for the content path.** All of these come from earlier internal runs.
   - The Thinker on **text** answers IT/FR/DE/ES at the English level: 15-16/16 from reference text. Fed the donor
     streaming-ASR transcript, it scored EN 14, IT 13, FR 12, DE 14, ES 10 out of 16.
   - The original **speech** path scores IT/FR 0/16 on the same items.
   - So the Thinker is not the bottleneck; the acoustic mapping is. M1-force sidesteps the acoustic mapping
     **without training**.
3. **Negative result: the user-text channel (`M1-chan`).** The released checkpoint has no `asr_head`, no
   `embed_asr_tokens`, and fuses that term with weight 0.0 (§3). Forcing it on zero-shot was measured earlier as
   **noise**. It is documented, not proposed.
4. **Alternative: M1-ctx** (NOT VERIFIED). Keep the duplex loop fully in charge, and insert the transcript into its
   context as prompt-layout text at extra steps. This is the in-loop variant if M1-force's split brain causes
   problems (§5).
5. **Fallback: text cascade.** ASR, then an external EOU / VAD, then the text-mode brain, then the adapter and talker.
   It loses in-model timing (§8).
6. **ASR:** `nvidia/nemotron-3.5-asr-streaming-0.6b@ea30d66`. This is the same donor the earlier multilingual
   work pinned; reuse its loading and RNNT code (§4.5). Its encoder frame is 80 ms, the Thinker step.
7. **Tracing:** every decision is one JSONL line with both clocks (§9). `trace_view` prints a per-turn timeline.
8. **First GPU gate, E1** (§10): M1-force vs M0 vs cascade. Per language (EN/IT/FR): an opener, two Q/A turns (one
   tool) and one barge-in. Thinker and ASR only, no talker. Cap **15 GPU-min**.

## 1. Architecture (M1-force)

```
                 user audio (16 kHz PCM, 80 ms frames)
                      |
       +--------------+------------------------------------+
       v                                                   v
 [A] multilingual streaming ASR (donor RNNT)        [B] VoiceChat perception + English RNNT (original)
     target_lang = session locale / auto                 |
       |  pieces per frame                               v
       v                                          [D] DUPLEX LOOP (frozen Thinker, decode_audio=False)
 [C] UserTextScheduler -> closed words                 real frame k: fuse(agent[k-1], audio[k], func[k-1])
     TurnTaker -> EOU / BOU (source="asr")              predicts agent id  ->  [F] AgentChannelForcer
       |                                                     ^   BOS => request reply            |
       | authoritative user turn text                        |   write gen_text[k] (forced id)   |
       v                                                     +-----------------------------------+
 [E] TEXT-MODE BRAIN (same weights, own cache)                         | to_adapter id per frame
     chat history (session language), /no_think, ban id 12             v
     reply ids ---------------------------------------------> [G] TalkerAdapter (word commits, burst commit,
     tool decision: function head on the text context                 EOS drain/cancel, barge-in, tool wait,
                                                                       segments, epochs)
                                                                         |  StartSegment / AppendText /
                                                                         v  FinalizeSegment / Cancel
                                                               [H] Flash talker: fresh prefill per segment,
                                                                   codec language = session language, BLOCK
                                                                         |  AudioChunk(epoch, ...)
                                                                         v
                                                               [I] PlaybackGate -> transport -> client
 every box -> Tracer (JSONL: ev, t_ns, model_step, turn_id, epoch, ...)
```

### Clocks

There are three clocks. They are never mixed.

| clock | unit | advanced by | notes |
|---|---|---|---|
| model frame `k` | 80 ms of user audio | the duplex loop, one real step per input frame | the text-mode brain and inserted steps (M1-ctx) do not advance `k` |
| talker codec frame | 80 ms of agent audio | the talker, one step per text embedding | BLOCK: no frame without its text; independent of wall time |
| wall clock / playback | seconds | scheduler, transport, client | late frames = underrun gaps |

The ASR runs on the same 80 ms clock (encoder subsampling x8 of a 10 ms hop). A word is available `r` frames of
right context after it is heard, plus up to `close_after` (3) frames for word closure.

### Components (pure Python here; GPU parts are interfaces)

| component | contract | code |
|---|---|---|
| ASR [A] | PCM 16 kHz + `target_lang` → pieces per frame, monotonic (never retracted) | interface; reuse the lab loader (§4.5) |
| UserTextScheduler / TurnTaker [C] | pieces → closed words (+ Thinker ids, specials < 1000 and language tags dropped); per frame non-blank → `bou` / `eou` | `usertext.py` |
| duplex runner [D] | per frame: predicted agent id and function id; per-frame hook that may overwrite `gen_text[:, k]` | interface (§4.1) |
| AgentChannelForcer [F] | per frame `step(k, predicted) → ForcedStep(write, to_adapter)`; `add_reply`, `cut`, `on_eou` | `forcing.py` |
| text-mode brain [E] | history → reply ids (no specials, no id 12), streaming; tool decision | interface (§4.3) |
| TalkerAdapter [G] | `Frame(k, text_id, func_id, user_speech)`, `ToolEvent`, `TalkerDone` → epoch-tagged commands | `adapter.py` |
| talker driver [H] / PlaybackGate [I] | commands → `AudioChunk`s; drop stale epochs | `driver.py` (protocol, fake, Flash stub) |
| Tracer | `emit(ev, k, **fields)` → JSONL; `trace_view` CLI | `tracing.py`, `trace_view.py` |

### Language hint, end to end (`lang.py`)

| step | value for `it` | mechanism | status |
|---|---|---|---|
| ASR | `target_lang="it-IT"` (prompt id 15) or `auto` | the language one-hot enters the RNNT joint through the prompt MLP (the lab's `PromptJoint`, `perception_swap.py:34-51`) | measured earlier with `auto` |
| brain | system line "The caller speaks Italian ... Always answer in Italian." plus Italian user turns | chat template | measured earlier: text-mode replies follow the question language |
| duplex loop | same system prompt (timing only) | prompt prefill in the user-audio slot (W:1774-1838) | timing only; content not used |
| talker | codec language `italian` | prefill of every segment | an earlier smoke: IT/FR/EN intelligible; no formal gate |

## 2. Sources

Abbreviations, NeMo Speech files:

- **W** = `nemo/collections/speechlm2/inference/model_wrappers/nemotron_voicechat_inference_wrapper.py`
- **M** = `nemo/collections/speechlm2/models/duplex_stt_model.py`
- **F** = `nemo/collections/speechlm2/parts/fusion.py`
- **S** = `nemo/collections/speechlm2/inference/pipelines/streaming_s2s_pipeline.py`
- **Y** = `examples/speechlm2/nemo_inference_pipelines/conf/s2s_streaming.yaml`

These come from branch `nemotron-labs-voicechat`: the source the V2 engine wraps (revision in
`v2/src/mynah_v2/model/reference.py`). Line numbers refer to the pinned copies used for this audit. That they are
byte-identical to that revision is NOT VERIFIED.

**P** = `nemo/collections/speechlm2/parts/label_prep.py`, from a NeMo main checkout.

## 3. Audit: what a frame input is made of, and where text can enter

**Per real frame** (W:2370-2414), the input is the `AddFusion` sum (F:100-106; plain float weights):

```
x_t = 1.0 * embed_tokens(gen_text[t-1])             agent text fed back                         W:2397
    + 1.0 * perception(audio)[t]                    user audio                                  W:2373
    + w_ut * embed_asr_tokens(gen_asr_text[t-1])    user-text channel (only with an asr_head)   W:2398-2403
    + 2.0 * embed_tokens(gen_func[t-1])             function channel                            W:2404-2408
```

The weights 1.0 / 1.0 / 2.0 are the ones in use (`v2/src/mynah_v2/model/tokens.py` `FUSION_WEIGHTS`).

### Where the duplex loop's state is written each frame (the M1-force intervention points)

| order in a frame | what | file:line |
|---|---|---|
| 1 | LLM forward on `x_t` → `predicted_token`, `function_predicted_token` | W:2418-2452 |
| 2 | `gen_text[:, t] = predicted_token` | W:2454 |
| 3 | FC state machine on `gen_function_text[:, t]` (SOTC starts the FC) | W:2466-2477 |
| 4 | FC in progress → `gen_text[:, t] = PAD` | W:2494-2496 |
| 5 | forced turn-taking on the internal ASR channel (inactive here: no head → `gen_asr` stays PAD) | W:2498-2500, W:2789-2845 |
| 6 | RNNT step, then `_apply_rnnt_turn_taking`: (a) **redirect queue: `gen_text[b, t] = next_tok`, one pre-tokenized id per frame, then `continue`**; (b) self-play suppression (off in the shipped config, W:543); (c) max-response / ratio caps → forced EOS + redirect; (d) **barge-in → forced EOS** | W:2503-2513; (a) W:3034-3044; (b) W:3077-3096; (c) W:3200-3240; (d) W:3249-3259 |
| 7 | `predicted_tokens[:, offset] = gen_text[:, t]`: the written value is the output | W:2502, W:2514 |
| next frame | `embed_tokens(gen_text[:, t])` is the agent-channel input | W:2397 |

Training feeds the agent channel with **gold** text (teacher forcing): `text_inputs = target_tokens[:, :-1]`
(P:229-230), `agent_text_embeds = embed_tokens(text_inputs)` (M:2113). A forced agent channel is therefore the
training regime. What is new is the **content and language** of the forced text, not the mechanism.

### Every place where text enters the duplex model

| path | layout | trained? | file:line |
|---|---|---|---|
| agent channel (fed back) | `embed_tokens(agent id)` | **yes**: teacher-forced gold text | M:2113, P:229-230, W:2397 |
| system prompt | `embed_tokens([bos] + prompt + [eos])` in the **user-audio slot**, agent PAD | **yes** | M:1807-1845, W:1774-1838, S:2930ff |
| tool response | forced ids on the function channel at **inserted** steps, silence audio, agent PAD | **yes** | M:1561-1584, M:1038-1073, W:1250-1300, W:2124-2135 |
| user-text channel | `embed_asr_tokens(user text)`, 4th term | **no** in this checkpoint | W:763-771, W:1833-1838, W:2398-2403 |
| plain text LM | `llm(embed_tokens(text)) → lm_head`, outside the loop | yes: `text_to_text_loss` (weight 0.5 per the recipe audit) | M:2624-2648 |

### The user-text channel in the released checkpoint

- `predict_user_text` is false. This was read from the checkpoint config by an earlier audit; it was not re-read
  here.
- So the checkpoint has no `asr_head` and no `embed_asr_tokens` (M:274-283). The wrapper builds the fusion with
  `user_text_weight = duplex_asr_text_weight if self._has_asr_head else 0.0` (W:763-771). It passes `None` on every
  frame and in the prompt prefill.
- The NeMo design this channel comes from:
  - training feeds the channel the frame-aligned **gold** transcript, optionally delayed (M:2116-2118, P:204-244);
  - inference feeds back the model's own `asr_head` prediction (W:2457).
- The released Thinker therefore never learned to read the channel. "The Thinker already reads user text" is false
  for this checkpoint.
- The RNNT transcript is auxiliary: it drives the turn-taking counters and the display, and never enters the LLM
  (W:2505-2513).

## 4. Answers

### 4.1 Forcing the content (M1-force), and where to hook it without touching `v2/`

The hook is a **per-frame instance-level wrap of `_apply_rnnt_turn_taking`**. This is the same technique V2 already
uses for `_rnnt_step` (`v2/src/mynah_v2/runtime/rendezvous.py:347-372`); it lives in the Frankenstein runner. Per
frame:

1. Read `p = gen_text[b, t]`: the model's prediction, after the FC and forced-turn-taking steps.
2. Call `s = forcer.step(t, p)`. If `s.write` is set, then `gen_text[b, t] = s.write`.
3. Call the original `_apply_rnnt_turn_taking`. It keeps the RNNT EOU forced BOS and the RNNT barge-in forced EOS.
4. Read `gen_text[b, t]` again. If the original turned it into EOS while the forcer was speaking, that is an
   audio-side barge-in: call `forcer.cut()` and `adapter`-cancel, with source `duplex_rnnt`.
5. Feed the adapter `Frame(t, s.to_adapter, func, user_speech)`. Trace `thinker.frame` (predicted, fed) and
   `force.*`.

Details checked in source:

- The wrapper processes **2 frames per call**: 0.16 s chunks (Y:172) in V2. So the forcing must be per frame inside
  the loop. Writing `gen_text` after `step()` returns would be one frame late for the second frame.
- The hook runs only when the RNNT is loaded and no FC is in progress (W:2503-2513). That is true in the shipped
  configuration, and M1-force suppresses duplex FCs (§4.3).
- Do **not** use the wrapper's own `_redirect_tokens_queue`. While it is non-empty, the per-frame logic `continue`s
  before the barge-in check (W:3036-3044), which disables RNNT barge-in for the whole forced reply.
- The max-response cap forces EOS plus a redirect phrase after 15 s of agent speech (W:3200-3210). Long forced
  replies must stay under it, or the cap must be raised in the Frankenstein config. The adapter's segmenting already
  cuts long replies.
- `decode_audio=False` (W:188) removes EarTTS and the codec from the duplex loop. Whether every V2 path honours it
  (FC phase 2, rehydration) is NOT VERIFIED.

The forcer's per-frame states (`forcing.py`):

| state | model predicts | written / to adapter | why |
|---|---|---|---|
| LISTEN | BOS | keep BOS / BOS; request a reply | the audio decided to speak |
| LISTEN | text without BOS | PAD / PAD | the duplex loop's own content is never spoken |
| LISTEN | anything, `bos_fallback_frames` after the ASR EOU | BOS / BOS | timing fallback when the audio path stays silent (e.g. it does not "hear" IT end-of-turn) |
| WAIT_TEXT | anything | PAD / *no text* | the reply is not ready; *no text* never triggers a burst commit; cap `wait_max` → EOS |
| FORCING | anything | next reply id / same | one id per frame, like the natural burst; overrides are counted and traced |
| FORCING | (queue empty, generation running) | PAD / *no text* | generation underrun; traced `force.underrun` |
| FORCING | (barge-in: adapter or duplex RNNT) | EOS / EOS; drop the rest | both brains stop; the history keeps what was forced and played |
| TAIL | PAD / EOS | keep | the model chooses its PAD tail and EOS after "its" reply |
| TAIL | other text | PAD / PAD | no extra content; cap `tail_max` → EOS |

### 4.2 What the duplex loop does when its preference is overridden for many frames (risk analysis)

- **In distribution:** a forced agent channel is how the model was trained (teacher forcing on gold agent text).
  After a few forced ids the duplex state "has said" the reply. Its PAD tail and EOS should then be paced on that
  text, the way it paces its own replies.
- **Out of distribution:**
  - IT/FR agent text in the channel; duplex training data was English.
  - Agent text that disagrees with what the audio context suggests. The model heard misrecognised IT audio and now
    "says" a sensible Italian answer.
  - Effect on later frames (BOS/EOS timing, spurious EOS, early SOTC) is NOT VERIFIED.
- **Measured per frame:** `force.override` count, agreement rate, model EOS frame vs end of the forced text, spurious
  EOS during forcing (ignored, traced).
- **Gate:** E1-G3.

### 4.3 Text-mode brain, process sharing, tools

Text mode means:

- `stt_model.llm(inputs_embeds=embed_tokens(ids), past_key_values=<own cache>) → lm_head`. This is exactly the
  training `text_to_text` path (M:2628-2634).
- The Nano chat template, `/no_think`, greedy decoding.
- **Ban id 12.** It is Nano's chat end-of-turn, but VoiceChat's frame PAD. Stop on `<SPECIAL_11>` / `</s>`.
- This is the earlier harness `labs/multilingual/scripts/multilingual/text_oracle.py` (`ask`) and
  `llm_text_bench.py`. Measured earlier: without the ban, replies came back empty.

Memory and process:

- The same modules and weights in the same process, so no extra weight memory. It needs a **separate** hybrid cache
  (Mamba state + attention KV) for the text history.
- How this interacts with V2's captured graphs and LLM cache (`v2/src/mynah_v2/model/thinker_graph.py`,
  `llm_cache.py`) is NOT VERIFIED. The safe first version runs the brain eagerly, outside the graphs, serialised with
  the duplex steps on one CUDA stream.

Timing: reply text must exist when the duplex loop says BOS.

- Start the brain at the **earliest** of the ASR EOU and the duplex BOS. Use the closed words so far, plus a flush of
  the open word.
- In the measured duplex behaviour the agent's decision lands around the end of speech, so the duplex BOS will often
  come first.
- Cache the system prompt and history prefix once. Each turn then appends only the new user turn and the last spoken
  reply, and decodes.
- The forcer covers the gap with PAD (WAIT_TEXT). Target: first forced id ≤ 4 frames (320 ms) after BOS, measured as
  `brain.request` → `brain.first_token` → first `force.override`.
- The brain must out-decode one id per 80 ms frame while sharing the GPU with the duplex step. Underruns are traced,
  and they do not break the talker (*no text* never commits).
- Speculative starts (on short pauses) are possible later, at the cost of discarded work.

History:

- User turns are the ASR text, in the session language.
- Agent turns are the **spoken** prefix. A barge-in cuts the reply. The forcer keeps `turn_ids` (what was forced) and
  the adapter / talker traces give what was played. The history stores what the caller actually heard.

**Tools: who decides when.**

| option | decides | content | verdict |
|---|---|---|---|
| T1 (proposed) | brain: a function-head probe on the brain's text context at the user's turn end (SOTC vs PAD margin); if call → the function head writes the call text (the earlier forced-call harness, `a1b_forced_call.py`) | from the authoritative transcript | measured earlier: from gold IT/FR text the function head wrote names 21/21, postcodes 17/19, but canonicalised concepts only 2/27. Exact names from the RNNT transcript capped at 9/21. The **decision** margin on text context is NOT VERIFIED (E1 has 2 tool items per language). |
| T2 | duplex function channel (SOTC from audio) | from the duplex context (misheard IT/FR audio) | rejected for IT/FR: the content would come from audio that the measurements show it does not understand |

The `lm_head` text path never writes `<TOOLCALL>` text (measured earlier: 0 of all tries). Tools go through the
function head.

During the tool:

- The duplex SOTC is suppressed by wrapping `_apply_fc_state_machine` (W:2466-2477): set `gen_function_text[:, t]`
  to PAD first, so the duplex never starts its own FC.
- The forcer keeps the agent channel at PAD. The adapter goes to TOOL_WAIT and may speak an on-hold line
  (`inject_utterance`).
- The result enters the brain history as `<TOOL_RESPONSE>[...]</TOOL_RESPONSE>` (the reference format,
  `mynah_v2.model.tokens.format_tool_response`). The brain writes the answer, which is forced at the next duplex BOS
  (or by the EOU fallback).
- The duplex KV never sees the call. Replaying call and response through the duplex function channel (the wrapper
  supports forced function tokens) is a later option, if E1 shows the duplex state drifts after tools.

### 4.4 Session language and output language

- The language line in both system prompts (`lang.with_language`). V2 already carries a per-session system prompt
  (`session.start.system_prompt`; `prompt_chars` in `session.opened`).
- Measured earlier on the text path: replies come back in the question's language, IT/FR/ES/DE at the English level.
  There was one IT→ES slip.
- **Known defect:** numbers and dates come out as English words inside IT/FR sentences ("Tre più quattro fa seven").
  The talker then speaks English numbers.
  - Mitigations to measure, never silent post-editing in evaluation: a prompt line "write numbers as digits", and
    digits read by the talker in the codec language.
  - Talker reading of digits in IT/FR: NOT VERIFIED.
- In M1-force the output language is the brain's, so the open question "does the duplex Thinker write Italian" goes
  away. What remains is whether **forced** Italian keeps the duplex timing healthy (§4.2).

### 4.5 Streaming ASR (default and reuse)

**`nvidia/nemotron-3.5-asr-streaming-0.6b`**, HF revision `ea30d66`. This is the donor ASR pinned by the earlier
multilingual work; the English sibling is `nemotron-speech-streaming-en-0.6b@ebe59e5`.

| property | value | source |
|---|---|---|
| architecture | cache-aware FastConformer, 24 layers, hidden 1024; RNNT (2-layer predictor, 640); prompt-conditioned; 600M | card, `config.json` |
| frame | 128 mel, 10 ms hop at 16 kHz, subsampling x8 = **80 ms** | `processor_config.json`, `config.json` |
| latency | `att_context_size [56, r]`, r ∈ {0, 1, 3, 6, 13} = 80 / 160 / 320 / 560 / 1120 ms chunks; ≤ 10 symbols per frame | card, `config.json` |
| languages | 40 locales: 19 transcription-ready (en, it, fr, de, es, pt, nl, ...), 13 broad-coverage, 8 that need fine-tuning | card |
| language hint | `target_lang=<locale>` or `auto`; prompt ids e.g. `it-IT` 15, `fr-FR` 8, `en-US` 0, `auto` 101 | card, `processor_config.json` |
| output | cased, punctuated; with `auto`, a `<xx-XX>` tag after the final punctuation | card |
| tokenizer | 13087 pieces + blank, plus 39 locale tags | `config.json`, `tokenizer_config.json` |
| license / runtime | OpenMDW-1.1; NeMo 26.06 (card: NeMo main) or Transformers ≥ 5.13 | card |

Card WER for it-IT across the five latencies: 5.23 → 4.25. We have not measured it.

**Reuse, not re-derivation.** The earlier lab code already loads this checkpoint and decodes its RNNT with the
language prompt on the 80 ms clock:

- `perception_swap.load_nemo_dir`;
- `PromptJoint` (language one-hot → prompt MLP → joint, `perception_swap.py:34-51`);
- `set_rnnt_ml(sd, cfg, lang="auto")` (`perception_swap.py:272-286`);
- the encoder attention-context handling in `encoders.py`.

Note that the lab runs it **inside** the VoiceChat perception (a swap). Frankenstein runs it **beside** the original
perception, on the same audio. Only the loader and the RNNT greedy step are reused.

Measured earlier with this donor:

- its RNNT readout recovered names 18/21, postcodes 13/19 and concepts 27/27 on the agent lexicon set (IT postcodes
  weakest);
- typical cascade errors: "dieci meno 6" → "dix point six", and script artefacts in ES transcripts.

Defaults: the earlier tests used `[56, 0]` and `auto`. Frankenstein uses the session locale when known, with `r` to
tune in E2.

### 4.6 Alignment, tokenization, revisions, EOU, causality

| topic | rule | where |
|---|---|---|
| tokenization | pieces → words (a word-start marker opens a word) → `" " + word` → Thinker tokenizer; ids ≥ 1000 only | `UserTextScheduler` |
| causality | a word exists from the frame it closed (next word, `close_after` idle frames, or EOU flush); the brain only ever sees closed words; M1-ctx spans only after `k` of their last word; no transcript is placed before the audio it came from | `usertext.py` (tested) |
| revisions | greedy cache-aware RNNT never retracts; a revising ASR would need a stable-prefix policy; nothing given to the brain or inserted is taken back | design rule |
| user EOU | `TurnTaker`: `eou_frames` blank ASR frames after confirmed speech (code default 10 = 0.8 s; the shipped VoiceChat config uses 40 for its RNNT, Y:84) | `usertext.py` |
| agent BOS | from the duplex audio path; ASR EOU + `bos_fallback_frames` as the fallback | `forcing.py` |
| barge-in | adapter (`user_speech` from the ASR `TurnTaker`), or the duplex RNNT forced EOS; both cut forcer and talker | `adapter.py`, `forcing.py` |

## 5. Controlled modes

| mode | real frames | content source | timing source | status |
|---|---|---|---|---|
| **M0** baseline | original audio | duplex loop (its own words) | duplex loop | IT/FR 0/16 on QA (measured earlier) |
| **M1-force** (primary) | original audio | text-mode brain, forced into the agent channel | duplex loop (+ ASR EOU fallback) | NOT VERIFIED: E1 |
| M1-ctx (alternative) | original audio + transcript **inserted** after the frame its words closed (prompt layout: `embed_tokens` ids in the user-audio slot, agent PAD; no audio consumed, like FC steps) | duplex loop reading the inserted text | duplex loop; agent BOS held until the span is in (`TranscriptInserter.hold_bos`) | NOT VERIFIED; E2 if M1-force fails G3 |
| M1-chan | original audio + `embed_asr_tokens` term x 1.0 | duplex loop | duplex loop | **negative zero-shot** (measured earlier: strong logit moves, no slot gain, hallucinated names, EN tool use worse; harness `v0_user_text.py`, `v1_gate.py`). That run used a bridge encoder, not the original perception; not re-run as-is |
| M2 | M1-ctx with the perception term damped | as M1-ctx | as M1-ctx | only if M1-ctx shows audio interference |
| M3 bypass | silence embedding (W:1028-1049) | brain (forced) or M1-ctx inserts | **external** only: ASR EOU → forced BOS, ASR BOU → forced EOS | experimental: measures how much timing survives without audio. Measured earlier: with silent audio and nothing else, the model never took a turn |
| cascade (fallback) | no duplex loop | brain | ASR `TurnTaker` / VAD | content measured earlier (E3) |

Context only, not in this project: training an adapter for the acoustic route did help. A trained residual adapter
on a multilingual encoder lifted non-English QA from 17 to 32 of 124, still far below the cascade. The acoustic route
needs training; M1-force does not.

## 6. Adapter state machine (`frankenstein_s2s/adapter.py`)

States:

- `IDLE`
- `ARMED`: BOS seen, no complete word yet
- `SPEAKING`
- `FINAL`: text final, `tts_eos` fed
- `DRAINING`
- `TOOL_WAIT`

Commands carry `epoch`; `Cancel` bumps it.

| case | event (frame k) | condition | action | test |
|---|---|---|---|---|
| burst | text ids | — | commit **complete words only**; the first opens a segment (`StartSegment` = fresh prefill with language + voice); then `AppendText` with prefix-stable Qwen ids | `test_burst_word_commit…`, `test_partial_words…`, `test_qwen_ids…` |
| burst commit | PAD | `commit_pads` (2) after text | flush the tail, `FinalizeSegment(burst_commit)` (`tts_eos`) instead of waiting for EOS | same |
| short pause | 1 PAD | < `commit_pads` | nothing (the talker BLOCKs on underrun) | `test_single_pad_pause…` |
| natural EOS | EOS | text already final | drain; log the PAD-tail length | `test_burst…` |
| early EOS, no speech | EOS | not final, no recent user speech | treated as natural: finalize (logged as ambiguous) | `test_eos_before_commit_without_speech…` |
| early EOS, speech | EOS | not final, recent user speech | interruption: `Cancel` | `test_eos_before_commit_with_recent_speech…` |
| barge-in | `user_speech` x `barge_in_frames` (3) | reply open or audible, `immediate` | `Cancel(barge_in)`; the gate drops in-flight and queued audio; later Thinker text of that turn dropped | `test_barge_in…`, `test_short_backchannel…` |
| Thinker-led barge-in | — | `barge_in="thinker"` | wait for the EOS (+ the early-EOS rule) | `test_thinker_mode…` |
| stale audio | chunk of an old epoch | — | dropped by `PlaybackGate`; `Cancel` flushes the queue | `test_barge_in…`, `test_no_command_ever_targets_an_old_epoch` |
| tool call | function SOTC | reply open | finalize the preamble, close the turn → `TOOL_WAIT` | `test_tool_call…` |
| tool wait / result / resume | `ToolEvent`, EOTR | — | silent (no `tts_pad`); optional `inject_utterance` on-hold line; EOTR / resumed / timeout / cancelled leave the state | `test_tool_call…`, `test_inject_on_hold…` |
| answer after tool | BOS or text without BOS | — | new turn and segment | `test_text_without_bos_after_tool…` |
| second turn | BOS after cancel / EOS | — | fresh prefill, new epoch if cancelled | `test_second_turn_after_cancel…` |
| BOS in an open turn | BOS | text not final | `Cancel(bos_restart)` | `test_bos_inside_open_turn` |
| | BOS | text final | close the previous turn, let it drain | `test_bos_after_commit…` |
| multi-burst | text after the commit | — | new segment, same turn | `test_long_pause_then_more_text…` |
| long reply | ≥ `segment_max_chars` at a sentence end | — | `FinalizeSegment(segment_length)`, new segment next | `test_long_reply…` |
| missing text / no text | `text_id=None` | — | neither text nor PAD (no commit); the forcer uses this for brain underruns | `test_missing_frames…`, `test_generation_underrun…` |
| specials / stray text | specials on the text channel; text outside a turn | — | dropped, traced | `test_specials…`, `test_empty_turn` |
| cascade source | `begin_turn` / `add_text` / `end_turn` | — | same rules on text deltas | `test_cascade…` |

`PAD → tts_pad` is never done.

- An underrun is the talker's BLOCK. An earlier talker smoke found BLOCK bit-exact with the unstalled run, and found
  that `tts_pad` mid-sentence destroyed content.
- With one id per frame, BLOCK leaves zero real-time margin. Playback needs a prebuffer, or the talker a faster
  runtime.

## 7. Duplex runner without EarTTS (reusing V2)

- V2's `ReferenceEngine` (`v2/src/mynah_v2/model/reference.py`) builds NVIDIA's config through
  `build_cfg(overrides=…)`. Its `step()` returns `new_text_ids` / `new_func_ids`.
- The Frankenstein runner imports `mynah_v2` **read-only** and builds it with `s2s.decode_audio=false`.
- It installs three instance-level wraps:
  1. `_apply_rnnt_turn_taking`: forcing, §4.1;
  2. `_apply_fc_state_machine`: duplex SOTC suppression, §4.3;
  3. for M1-ctx only, an inserted-step hook after a frame.
- Each step is mapped to `adapter.Frame` / `ToolEvent`, and traced.
- Nothing in `v2/` changes. All three wraps are on the wrapper instance, as V2's rendezvous does for `_rnnt_step`.

## 8. Fallback: text cascade

```
ASR [A] -> TurnTaker EOU (or VAD) -> text-mode brain -> adapter (begin_turn / add_text / end_turn) -> talker
```

- **Content:** measured earlier as E3. Donor transcript to the Thinker text path gave EN 14, IT 13, FR 12, DE 14,
  ES 10 out of 16 on dev. Held-out: EN 12, IT 12, FR 13, DE 14, ES 10. Reference text: 15-16.
- **Loses:**
  - the duplex loop's acoustic timing (BOS at the right moment, backchannels, overlap);
  - in-model barge-in;
  - endpointing becomes a threshold: EOU latency against cutting the user off.
- **Tools:** the same T1 path.

## 9. Tracing (`tracing.py`, `trace_view.py`)

Each event is one JSONL line in the V2 format: `{"ev", "t_ns", "session_id", "model_step" (= k), "turn_id",
"epoch", ...}`. V2 names are reused where the meaning is equal (`thinker.step`, `turn.started`, `function.*`,
`tool.*`, `asr.generated`, `audio.*`). Unknown names are rejected, as in V2.

| family | events | emitted by |
|---|---|---|
| duplex Thinker | `thinker.frame` (predicted id, fed id, `forced`, function id, decoded piece, source) | runner (`tracing.thinker_frame`) |
| forcing | `force.turn_open` (source: model_bos / eou_fallback), `force.override` (both ids, why), `force.underrun`, `force.turn_close` (why, dropped ids) | `AgentChannelForcer` |
| M1-ctx | `insert.span` (after_k, ids, words) | `TranscriptInserter` |
| ASR / turn-taking | `asr.partial`, `asr.word`, `asr.revised`, `asr.final`, `asr.drop`, `turn.bou` / `turn.eou` (source="asr") | `UserTextScheduler`, `TurnTaker`, runner |
| brain | `brain.request`, `brain.first_token`, `brain.last_token`, `brain.text`, `brain.tool_decision` | runner |
| adapter | `adapter.turn_open` / `turn_close` / `turn_restart` / `turn_implicit`, `commit_word`, `segment_start`, `segment_restart`, `burst_commit`, `finalize`, `eos_drain`, `eos_ambiguous`, `barge_in`, `cancel`, `epoch`, `tool_wait`, `drop`, `inject`, `retokenize`, `stale_drop`, `talker_done` | `TalkerAdapter`, `PlaybackGate` |
| talker / audio | `talker.prefill`, `append`, `step`, `block`, `finalize`, `codec_eos`, `cancel`, `chunk`; `audio.delivered`, `flush`, `underrun`, `played`, `playback.started` | driver, gate |

The source of every turn decision is in the event:

- `force.turn_open.source`;
- `turn.eou.source`;
- `adapter.barge_in` (the adapter heuristic);
- a duplex RNNT EOS seen as `thinker.frame` with `fed = 2` and `force.turn_close(why=barge_in)`.

User text is stored by default; pass `store_text=False` to keep it out.

To view a run:

```
python -m frankenstein_s2s.trace_view run/trace.jsonl [--turn N]
```

It prints a per-turn timeline with the columns `k | t ms | USER | THINKER | FORCE | ADAPTER | TALKER | AUDIO`.

## 10. Staged plan (GPU experiments listed, NOT run)

| stage | what | GPU | cap | pass / fail |
|---|---|---|---|---|
| S0 (done, CPU) | audit, design, adapter, forcer, user-text scheduling, tracer + viewer; tests | none | 0 | 49 pure-Python tests pass |
| **E1: M1-force vs M0 vs cascade** | Per language EN / IT / FR, one scripted session (synthetic agent corpus, store prompt): opener; Q/A 1 (store question); Q/A 2 (tool request, T1); a barge-in 1 s into the reply to Q/A 1 (repeat item). Duplex loop with `decode_audio=false`, deterministic, cache on, eager brain. ASR = donor with `target_lang` = session locale. Arms: **M0** (as shipped), **M1-force**, **cascade** (brain + `TurnTaker`, no duplex). The adapter runs on CPU on each arm's stream with the fake driver, so stale-audio and commit traces exist without a talker. Full traces. | Thinker 9B (one copy) + ASR 0.6B | **15 GPU-min** (load ~4, 3 sessions × 3 arms ~8, slack 3) | **G1 language:** M1-force agent text in the session language in ≥ 7/8 IT+FR replies (opener + Q/A). **G2 content:** IT+FR Q/A correct, M1-force ≥ cascade − 1 and ≥ M0 + 2 (of 4); for reference, the E3 cascade gave IT 13/16, FR 12/16. **G3 duplex timing under forcing:** BOS from the audio path (not the fallback) in ≥ 5/6 IT+FR turns; median first forced id ≤ 4 frames after BOS; 0 spurious EOS that cut a forced reply; model EOS within `tail_max`. **G4 barge-in:** forced reply and talker commands stop ≤ 3 frames after the decision; 0 stale chunks accepted after `Cancel`; the history holds only the spoken prefix. **G5 EN:** EN content ≥ M0 − 1. **G6 tools (diagnostic):** the brain's function-head decision and the call fields correct in ≥ 4/6 tool items; three evidences (model text, call, tool ground truth). Pass G1-G5 → M1-force is the engine path. G3 fails while G1/G2 pass → try M1-ctx (E2) or the cascade. G1/G2 fail in M1-force **and** the cascade → brain issue (prompt, numbers), not an architecture issue. |
| E2: M1-ctx / M3 / ASR latency | only if needed after E1: M1-ctx (eou policy, two wrappings); M3 timing survival; ASR `r` ∈ {0, 3} | same | 10 GPU-min | M1-ctx: content ≥ M0 + 2 with no BOS regression; M3: completed turns and barge-in vs M1-force |
| E3: talker in the loop | adapter + Flash talker (BLOCK) on E1 traces (replay) and live; IT/FR/EN; barge-in, tool wait, second turn | Thinker + ASR + talker | 30 GPU-min | ASR of the spoken audio matches the agent text (CER per language, numbers normalised for scoring only); 0 stale chunks played; gaps with a prebuffer within budget |
| E4: live runner | streaming end to end, one session, latency profile | same | 20 GPU-min | first audio after the user's EOU, barge-in reaction, three evidences on a tool call |

E1 is the cheapest decisive test. It needs one model copy, no talker and 9 short sessions. It measures the three
open questions at once: in-language content via the brain; whether the duplex timing survives forced content; and
barge-in through both brains.

## 11. Integration later (web UI, quantization): interfaces only

### Web UI

The existing web UI (`webui/server.py` + `static/app.js` on the web-UI branch) proxies the browser to the V2
WebSocket protocol (`v2/src/mynah_v2/transport/ws_server.py` header). The Frankenstein server should speak the
**same protocol**:

- `session.start` (`system_prompt`, `mode`, `input_rate`, plus a new optional `language`);
- binary PCM16 in;
- `session.end` / `session.cancel`;
- out: `session.created` (`output_rate`), binary PCM16, `{"type":"event","ev":…}` (the trace events of §9, filtered
  like V2's `verbose`), and `session.closed`.

Needed UI changes:

1. **Output rate.** `app.js` plays at a hard-coded 22050 Hz, but the talker outputs 24 kHz. Either resample on the
   server, or honour `output_rate` in the client.
2. **Client-side flush.** The client schedules buffers back to back and has no flush. On `adapter.cancel` it must
   stop the scheduled sources: the server-side gate cannot recall bytes already sent.

Steps:

1. the runner behind the `Session` interface;
2. `language` in `session.start`;
3. the two client fixes;
4. a guided sample per language.

### Quantization

The V2 Thinker quantization applies unchanged to the shared Thinker weights, for both the duplex and the brain passes:

- `mynah_v2.model.quant.quantize_thinker_int4`;
- the Q8 / INT4 weight-only GEMV packs (`mynah_v2.model.wo_gemv`, `--int8-pack` / `--int4-pack`);
- evaluation with `quant_eval.py` / `quant_microbench.py` from the web-UI branch.

What does not apply:

- the EarTTS, codec and RVQ scopes;
- the Q8 RNNT path, unless VoiceChat's RNNT stays for timing (it does in M0/M1).

The ASR (the card lists a q8_0 GGUF) and the Qwen3-TTS talker are outside V2's tooling.

Gate: only after E1, using token agreement on the E1 sessions for both passes.

## 12. NOT VERIFIED (consolidated)

1. That the duplex timing stays healthy under long forced IT/FR agent text (BOS/EOS, spurious EOS, SOTC). This is
   E1-G3.
2. The brain's tool **decision** from the function head on the text context. The call fields were measured; the
   decision was not.
3. That a second cache can share the loaded modules alongside V2's graphs and LLM cache.
4. Brain latency to the first token at the duplex BOS, and decode rate while sharing the GPU.
5. That `decode_audio=false` holds on every V2 path.
6. The M1-ctx reading of mid-conversation prompt-layout text, and the position semantics of inserted steps in
   Nemotron-H.
7. That the scratch line numbers are identical to the pinned NeMo revision. The checkpoint's
   `predict_user_text: false` was not re-read here.
8. The ASR card claims (WER, latency, language ID).
9. That the English RNNT works as a VAD on IT/FR speech, for the duplex loop's own BOS.
10. Flash talker IT/FR quality under an ASR gate; digit reading in IT/FR; duration against VoiceChat pacing.
