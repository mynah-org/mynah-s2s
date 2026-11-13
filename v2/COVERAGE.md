# V2 paper-fidelity matrix — VoiceChat 11B (single source; replaces the earlier 27-row checklist)

The spec is the paper, arXiv 2609.21967 (v2, re-read section by section on 2026-11-11). The reference is NVIDIA NeMo
Speech `097dfe9e`, and the checkpoint is `443794ea`. Path prefixes:
- **P** = `inference/pipelines/streaming_s2s_pipeline.py`
- **W** = `inference/model_wrappers/nemotron_voicechat_inference_wrapper.py`
- **M** = `models/duplex_stt_model.py`
- **Y** = `examples/.../conf/s2s_streaming.yaml`

V2 runs NVIDIA's native PyTorch engine behind `ReferenceEngine`, so **model components are NVIDIA's code**. Their
"parity" therefore checks how we integrate and drive them (driver, chunking, state, cache), not a reimplementation.

## Status labels
Each row carries one status, the highest level it has reached:
- **IMPLEMENTED**: the code path exists in V2 or is integrated from NVIDIA.
- **TESTED**: an automated test exercises it.
- **PARITY VERIFIED**: measured identical to the NVIDIA reference, by the method stated.
- **PARTIAL**: some part is missing or unverified (named in the gap column).
- **MISSING**: not implemented, or not checked at all.

Parity methods (`scripts/parity_driver.py`, `scripts/cache_parity.py`, `tests/gpu/test_llm_cache_gpu.py`):
- **driver**: V2 `Session.step` vs NVIDIA `run()`. Same process and input, deterministic decoding, token-identical.
  It only works on inputs without a tool call, because `run()` uses the blocking real-audio FC path while V2 and
  NVIDIA's Triton backend use the live non-blocking one.
- **cache**: incremental Thinker vs NVIDIA's shipped full-history recompute, token-identical.

| # | Paper mechanism (section) | NVIDIA reference | V2 implementation | Test | Parity evidence | Status / gap |
|---|---|---|---|---|---|---|
| 1 | 16 kHz mono in, 128 log-Mel 25/10 ms (§2.1) | wrapper preprocessor | via engine; `runtime.session` chunker (160 ms = 2×80 ms) | cpu chunking; gpu shapes (2 steps/chunk) | driver: identical downstream tokens, 3 inputs | **PARITY VERIFIED (driver)** |
| 2 | Streaming FastConformer 600M: 24 layers, ×8 causal subsampling, 70-frame left context, cache-aware, CUDA graphs (§2.1, App. B) | W perception cache + CUDA graphs | via engine | gpu runs (30 ms per 2 steps) | driver (indirect, via tokens) | **PARITY VERIFIED (driver, indirect)**. Features/embeddings never compared numerically |
| 3 | Identity adapter + projection into the LLM (§2.1) | wrapper | via engine | — | driver (indirect) | **PARITY VERIFIED (driver, indirect)** |
| 4 | Per-frame fusion user/text/function = 1/1/2 (§2.2; paper prints "11,11,22") | W `fusion_module`; ckpt config 1/1/2 | `tokens.FUSION_WEIGHTS` | gpu `test_fusion_weights_match_checkpoint_config` | driver | **PARITY VERIFIED** (live module = ckpt config; driver tokens) |
| 5 | Thinker Nemotron-Nano-9B-v2 (Mamba2 + 4 attention layers), BF16 (§2.1, App. C) | W `infer_one_step` | via engine | gpu dtype test | driver: text + function tokens, 3 inputs up to 1102 frames | **PARITY VERIFIED (driver)** |
| 6 | Incremental per-step LLM state; "cached Mamba recurrent states in FP32" (App. C) | **native engine disables the cache for Nemotron** (`s2s_context_manager.py:65-76`); NVIDIA serves with vLLM | `model.llm_cache`, **ON by default**: hybrid cache, FP32 SSM state (= paper), `cache_position`, 3 remote-code fixes applied as subclass + instance wrappers (no file edited) | gpu `test_llm_cache_gpu` (+2 positions per chunk, attention KV = position, FP32 SSM state evolving, fresh cache per session, ON/OFF identity on text/function/RNN-T); cpu `test_llm_cache` | cache: identical on 3 inputs up to 1102 frames + gpu test; FC path with cache: 0 violations | **PARITY VERIFIED (cache, deterministic)**. Identity under default (stochastic) sampling cannot be measured |
| 7 | Agent text head: BOS / subwords / EOS / PAD per 80 ms (§2.1) | W | traced per step | gpu runs | driver | **PARITY VERIFIED (driver)** |
| 8 | RNN-T branch (2×640, 1024 BPE + blank), auxiliary, never fed to the LLM (§2.1) | W `_rnnt_step` | via engine (`asr.generated`) | gpu cache test (ON/OFF identical) | driver B5: RNN-T/ASR channel token-identical to NVIDIA `run()` on 2 inputs, cache ON | **PARITY VERIFIED (driver)** |
| 9 | Function head, previous function token fed back (§2.2) | W `function_predicted_token` | traced (`func_ids`), FC records | gpu: every live call (≥15 runs), right names/args; **E6/E7 independent NVIDIA oracle** (offline training-format FC inference, `run_fc_offline_inference`, BF16) vs live path on sample_fc with the same fixed payload: **call identical** (`generate_random_number {"min": 1, "max": 50}`), SOTC frame 95 vs 97, response consumed with EOTR in both (34 vs 37 tokens: live rewrites numbers to words), same answer "Your random number is forty-two/forty two" | driver: function channel identical on no-FC inputs | **TESTED (independent oracle, semantic, 1 input)**: not token identity by design (silence conditioning, number words, ±2 frames) |
| 10 | Special tokens SOTC 20 / EOTC 21 / EOTR 22, PAD 12 (§2.2) | W:787-801 | `model.tokens` + live check at load | cpu + gpu `test_special_token_ids_live_tokenizer` | live tokenizer | **PARITY VERIFIED** |
| 11 | Payload = JSON list of {name, arguments}; parallel calls (§2.2; §7 "not reliable") | **live executor runs only the first call** (P:790 `call = parsed[0]`), the rest is dropped; offline path also stops at the first | REFERENCE = NVIDIA. ENHANCED: controller `execute_calls` runs every call (parallel, ledger entry each, per-slot unknown/timeout errors), responses in call order → `<TOOL_RESPONSE>[r1, r2]` as in the template | cpu: order kept under out-of-order completion, unknown + timeout slots, adapter routing, mode wiring | — | **TESTED (cpu)**; model never emitted a multi-call in our GPU runs (no such scenario audio); reference limit documented |
| 12 | FC timeline insertion: SOTC → args → EOTC → response 1 token/step → EOTR; agent text PAD; user = silence (§2.2, App. B) | M:936-1131 (training), W `_run_fc_async_steps` (live) | `runtime.fc_timeline` checker on the live records | cpu `test_fc_timeline` (6 cases), `test_fc_records` (2 measured geometries); gpu `test_fc_sample_…` | every GPU FC run: 0 violations | **TESTED (structural invariants)**. Not a token parity (no oracle, see #9) |
| 13 | Multi-step: after EOTR the model may call again (§2.2) | live path | — | gpu s2 ENHANCED: second call ABC456 after the first answer | — | **TESTED (1 behavioural case)** |
| 14 | Tool response = `<TOOL_RESPONSE>[…]</TOOL_RESPONSE>`, numbers to words (§2.2) | P:1794-1800 (`fc_convert_num_to_text`) | `tokens.format_tool_response`; ledger-wrapped handlers | cpu; gpu: spoken facts = SQLite row (3 evidences) | — | **TESTED** |
| 15 | System prompt `<AVAILABLE_TOOLS>` template (App. D.2) | `function_calling/template.jinja` | NVIDIA renderer (`run_stream.render_prompt`) | cpu `test_paper_reuse`: App. D.2 structure, tool JSON round-trip, identical to NVIDIA's renderer for the default tools | — | **TESTED** |
| 16 | Fast-decode at SOTC, tool invoked asynchronously (App. B) | W two-phase async, P:1600-1800 | via engine | gpu FC tests | — | **TESTED** |
| 17 | Fast-inject at return, then generation resumes (App. B) | W:1283-1297, P:1806 | via engine; consumption linked to call_id at EOTR (controller) | gpu FC tests; cpu `test_result_is_consumed_only_at_eotr…` | — | **TESTED** |
| 18 | Per-tool `ack_message`, spoken while the tool runs (§2.2, §4, App. D.1); paper: agent-text channel overridden | **TTS only, text stays PAD**: (a) random generic ack pool (20 phrases), (b) per-tool Layer-1 on-hold via `fc_on_hold_messages_path` (Y:132, default null), (c) generic reminders | registry acks exported in the reference JSON format (`--on-hold`) | cpu format test; **gpu B6 onhold + ASR**: "happy to assist" (generic ack), then "let me check that order for you" (per-tool), then the correct facts | — | **TESTED (gpu, 3 evidences)**. Text-channel override (paper wording) ≠ reference (TTS only); we follow the reference |
| 19 | Tool timeout (App. B: not specified) | `fc_tool_timeout_sec` 15 s after the on-hold messages → error TOOL_RESPONSE (P:1771-1779). **Bug confirmed on GPU (B6 timeout_ref)**: its `with ThreadPoolExecutor` joins the tool thread, so with a 25 s tool the timeout error is injected only at 25 s and the real result is thrown away | REFERENCE unchanged; ENHANCED: bounded wait in our handler, explicit timeout error, late result recorded and never applied | cpu `test_tool_lifecycle`; **gpu B6 timeout_enh + ASR**: timed_out at 3.0 s, "I am sorry, there was a technical issue…" (no invented facts) | — | **TESTED (gpu)** |
| 20 | Unknown tool / invalid call (not in paper) | executor → None → **nothing injected** (P:1795); **gpu B6 unknown_ref + ASR: the call never closes and the agent states an invented delivery date** | ENHANCED: controller policy → error JSON + FAILED call in the ledger; grounding audit flags unanswered requests | cpu policy + audit tests; **gpu B6 unknown_enh + ASR**: "I do not have access to order tracking" | — | **TESTED (gpu)**; REFERENCE hallucination documented |
| 21 | Endpointing fallback: RNN-T + activity force BOS/EOS (§4) | W forced turn-taking; Y:84-101 (EOU/BOU 40 frames) | via engine | **gpu C0 probe** (deterministic, input timeline): user end → agent BOS 0.16 / 2.72 / 0.48 s, → first audio 0.64 / 3.12 / 0.88 s on sample_general (paper FDB 1.0 smooth turn-taking: 448 ms) | driver: BOS/EOS identical to `run()` on 2 inputs | **TESTED (1 input)**: no multi-scenario latency distribution |
| 22 | RNN-T barge-in detection outside FC (App. B; FDB interruption latency 480 ms, TOR 100%) | W `_apply_rnnt_turn_taking` (BOU 40 frames) + the LLM's own EOS | via engine; ENHANCED feeds the engine identically outside FC (cpu test) | **gpu C0 + E3**: 2 interrupting utterances × 3 offsets into the agent's answer: the agent yields in **6/6**, EOS +0.68-1.16 s (median ~0.8), audio silent +0.76-1.32 s; it takes the new turn in **1/6** ("Of course. I can check that for you…"), stays silent in 5/6 | — | **TESTED**: stopping = paper (TOR 100 %), slower than 480 ms; not answering the interruption is reference/model behaviour (same in both modes outside FC). During FC: REFERENCE ignores, ENHANCED buffers + replays (#23) |
| 23 | No barge-in during tool execution: audio transcribed, not conditioning (App. B, §7) | RNN-T only; FC interrupt `rnnt_fc_interrupt_ms` 3200 effectively unreachable (audit §11) | REFERENCE = same. ENHANCED extension: speech buffered during the call and replayed after it | cpu buffer tests; gpu A/B job 91 (3 evidences) | REFERENCE behaviour = paper | **TESTED** (REFERENCE faithful; ENHANCED extension) |
| 24 | EarTTS 778M + 199M codec, 22 kHz, 12.5 Hz, 31 RVQ, MoG head, FP32 TTS (§2.3, App. C) | W TTS path, Y:46 FP32 | via engine; PCM out | gpu dtype test | **driver C1, same RNG seed**: agent waveform vs NVIDIA `run()` max |diff| 3.05e-5 = 1 LSB of the PCM16 file NVIDIA writes (mean 1.8e-5), same length; unseeded B5 differs (EarTTS sampling RNG) | **PARITY VERIFIED (driver, seeded, within PCM16 quantization)** |
| 25 | Speaker prompt 3 s (§2.3) | checkpoint's pre-baked latent `inference_speaker_name` | `speaker_name="Aria"` (same latent), no reference clip | cpu `test_paper_reuse` (cfg == checkpoint config) | — | **TESTED** (config level) |
| 26 | TTS interruption embedding + fixed silence frame (§2.3) | codec silence codes substituted on EOS (ckpt `inference_force_speech_silence_on_eos: true`) and optionally on idle PAD (`inference_force_speech_silence_on_pad`, env-gated, off) | via engine; `set_tts_pad_silence` per session (ENHANCED opt-in) | cpu knob test; **gpu B8 + ASR**: knob ON cut the end of an answer ("how about you" lost); REFERENCE run without babble this time | — | **TESTED (1 run per arm)**: knob not adopted; babble rate not measured |
| 27 | Causal codec with CUDA graphs, 22.05 kHz out (§2.3) | wrapper codec | via engine | gpu runs | see #24 (C1) | **PARITY VERIFIED (driver, seeded)** |
| 28 | Client: 80 ms PCM chunks, resample to 16 kHz, 22.05 kHz out, chunk = k×80 ms (App. B) | Triton + FastAPI WebSocket (not in repo) | `transport.ws_server`: PCM16 in at 8-48 kHz (`input_rate`, streaming soxr to 16 kHz), 22.05 kHz out, 160 ms engine chunks | cpu `test_ws` (resampler tone/length, 48 kHz session, cancel on hang-up); gpu ws demo with cache (2 sessions, 16 kHz) | — | **TESTED**; engine chunk fixed at 160 ms (not configurable) |
| 29 | Streaming latency: p95 118 ms per 160 ms chunk, 4 streams, H100 (App. C) | vLLM engine (+ CUDA graphs for TTS/codec in the paper runtime) | native + cache: p50 345 / p95 354 ms per chunk, 1 stream, GB10 (RTF 2.16) | gpu C2 profile per component: Thinker 147 (43 %), EarTTS 86-98 (26 %), codec 48 (14 %), perception 30, stt rest 27, glue 2 | — | **MISSING (performance)**: P1 targets in `.work` (quantize the Thinker, graphs for EarTTS/codec) |
| 30 | Precision: perception + LLM BF16, TTS FP32, Mamba cache FP32, "TF32" (App. C) | Y: compute bf16, TTS fp32, `matmul_precision: medium` (code) | same as reference; SSM state FP32 in our cache | gpu dtype test | — | **TESTED**. Paper says TF32, reference sets `medium`; we follow the reference |
| 31 | ~2 min trained audio context (§7) | 6144 frames ≈ 8.2 min allowed (Y:175) | ENHANCED rehydration (controller facts + history summary → new model stream, fresh cache); replay hold keeps answers intact | cpu `test_controller` (+ turn markers stripped); **gpu E4/E5** (spliced s1 request → 30 s → "just tell me the delivery date"): refresh at 42.7 s, summary carries the verified XYZ123 row, follow-up answered "scheduled to arrive on November twelfth, twenty twenty six" with **no new tool call**, 0 grounding violations, ASR agrees | — | **TESTED (gpu, 3 evidences)**; summary history is fragmented (user/agent deltas interleave) |
| 32 | ≤ 5 tools per session (§7) | not enforced | `ToolRegistry` warns above 5 | cpu `test_registry_warns_beyond_paper_tool_limit` | — | **TESTED** (warning, not a hard limit) |
| 33 | Long tool responses delay speech (§7) | response forced 1 token/step (W:1283-1297) | traced (`tool.result.injected` steps) | **gpu F1** `test_fc_sample_…`: phase-2 length = tokenized (number-converted) response (+ EOTR) | — | **TESTED (gpu)** |
| 34 | Session lifecycle (not in paper) | **FC background state leaks across close/open** (P:399, 990, 1078) | drain (wall-paced) / close kinds normal-cancel-force, `release_fc`, controller cancels open calls | cpu drain/release/close-kind/ws tests; gpu ws demo with cache (2 sessions); **gpu B6 cancel**: call cancelled at hang-up (2.8 s), next session clean | cache: fresh state per session (gpu) | **TESTED (gpu)** |

## Application layer (not in the paper: V2 extensions, controller-owned)
Controller-owned items:
- call / turn / intent / session ids;
- ledger states pending → running → completed / failed / timed_out / cancelled / stale;
- late and duplicate deliveries;
- epoch fence for corrections;
- facts only from completed calls;
- grounding audit (agent words while a call is open);
- consumption linked to the call at EOTR.

Tested on CPU in `tests/cpu/test_tool_lifecycle.py` and `test_controller.py`. GPU scenarios are pending in the
`.work` log.

## Counts (2026-11-11 evening, after jobs B1-H on GB10; GPU suite 10/10, CPU suite 91/91)
All 34 mechanisms are accounted for:
- **PARITY VERIFIED: 11** (#1-8, #10, #24, #27). Identical to NVIDIA via the driver, seeded driver or cache
  methods: tokens on every channel, waveform within 1 PCM16 LSB.
- **TESTED: 22**. Each has an automated test or GPU evidence (internal text + OmniASR + SQLite where facts are
  involved). #9 is checked against an independent NVIDIA oracle.
- **MISSING: 1** (#29). Realtime on GB10 is a separate performance goal. Measured RTF: 2.16 BF16, 1.88 with
  `cudnn.benchmark`, 1.31 with the experimental int4 Thinker.

Limits that belong to the model or reference, and are therefore not runtime bugs:
- **#11:** the reference executes only the first call of a list. ENHANCED executes all of them.
- **#19:** the reference timeout joins the tool thread before injecting. ENHANCED bounds the wait.
- **#20:** the reference injects nothing for an unknown tool, and the agent invents facts (GPU + ASR). ENHANCED
  answers with an error.
- **#22:** the agent yields to an interruption, but rarely takes the new turn.
- **#26:** occasional babble. NVIDIA's own silence knob cuts the end of answers.

Weakest evidence: #9 (one input, semantic), #11 (CPU only: no multi-call audio), #22 (6 samples on one base input).
