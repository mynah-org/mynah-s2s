# Frankenstein S2S: performance plan (proposals)

This file contains proposals only. Nothing below was run because of this document. Every GPU validation listed here
has a time cap and waits for an explicit GO. Measured timings from the E1 runs are kept in the local run reports and
are not copied here: this tree does not publish performance numbers for any particular machine. The text below uses
counts that do not depend on hardware (frames, tokens, steps, rows), plus orderings ("larger than", "dominates").

Scope and priority:
1. **Nano-9B-v2 text LLM ("Nano")**: hybrid Mamba/attention cache, prefill vs decode, the INT4 multi-token path,
   prefix/KV reuse, and the worker/stream model.
2. **Sophon Flash / Qwen3-TTS-12Hz talker**: the 12.5 Hz frame step, the 15-step code predictor, the codec decode,
   graph capture, chunked decode, and batching across sessions.

The duplex Thinker and perception come third. They already have most of the optimizations they need in the web UI
tree.

## 1. What runs, on which clock

| Component | Clock | Work per tick | In the E1 loop? |
|---|---|---|---|
| Perception (FastConformer + adapter) | one call every 2 model frames (160 ms of audio) | full-window encode | yes |
| Duplex Thinker step (NemotronH, INT4) | every model frame (80 ms) | one decode row plus heads; the controller hooks run in Python after the forward | yes |
| RNNT step + turn-taking hooks | every model frame | tiny | yes |
| Controller | every model frame | Python, no GPU work | yes |
| Streaming ASR (multilingual RNNT) | 80 ms encoder frame | encoder chunk + greedy RNNT | **no**: E1 replays a frozen CPU-built track |
| Nano worker | asynchronous, per request | prefill, then the decide (3 tokens), call (27-35 tokens) and reply (25-54 tokens) steps | yes, same GPU and same CUDA stream |
| Talker (Flash) | 12.5 Hz codec frame (80 ms) | talker LM step + 15-step code predictor + codec decode | **no**: FakeTalkerDriver |

Consequences:
- Every budget below leaves out the ASR and the talker. Both run on the same 80 ms clock, so both will compete for
  the same frame time. They have to be measured before any optimization is ranked as "sufficient" (G0 below).
- The Nano worker has to run on the default CUDA stream. NemotronH's remote code wraps every block in
  `torch.cuda.stream(default_stream)` (modeling_nemotron_h.py:767). On a side stream, the block bodies race with the
  embeddings and the head (found in E1h). As a result, Nano kernels and duplex-frame kernels serialize on one queue:
  - each Nano token waits behind frame work;
  - each frame waits behind Nano work.

  E1k shows both effects. Nano decode inside the session is slower than the same steps run standalone. Frame-time
  p95 goes above the 80 ms period while Nano decodes.

## 2. Budget (E1i / E1j / E1k)

### 2.1 Per frame (budget = 80 ms, real time)

| Item | Calls per 662 frames | Ranking (E1j/E1k, bf16 Nano) |
|---|---|---|
| Thinker step | 663 | largest per-frame item. GPU time is above host time, so the default stream backs up when Nano is active |
| Perception | 331 | second. Runs every 2 frames, so it alternates heavy and light frames |
| RNNT | 662 | negligible |
| Nano (async) | per request | adds directly to the frames it overlaps (same stream) |

Observations:
- **bf16 Nano**: the frame mean sits slightly above the period and p95 is well above it. The worst frames line up with
  Nano prefills.
- **INT4 Nano, e1k**: every per-frame component got slower, including perception and RNNT, which do not use Nano.
  The frame mean was several times the period. That fits GPU contention: the dequant path ran on every 16-row
  continuation chunk (§2.3), so it was not a per-frame regression. It is fixed on the CPU side by `min_rows` and is
  NOT VERIFIED on the GPU.

### 2.2 Per turn (first spoken token after the request)

| Turn type | Nano work before the first reply token |
|---|---|
| Greeting / no tool | prefill (744 prompt tokens), then the first decode |
| Out-of-scope (decide = REPLY) | prefill + decide (3 tokens on a 52-token branch) + reply branch (7 tokens) + first decode |
| Tool turn | prefill + decide (3 tokens) + **call (27-35 generated tokens)** + tool execution + reply branch (109-132 suffix tokens) + first decode |

Findings:

- **The call step dominates tool turns.** The call is decoded token by token: JSON names, keys, punctuation and
  values. Tool execution itself is negligible. A tool turn waits for decide + call + reply prefill before saying
  anything. That is why the first reply on tool turns is several seconds in E1j and E1k: tool-turn latency FAIL.

- **KV reuse inside a turn (E1k) did not pay off in the session.**
  - V2's `continuation_forward` runs the GEMMs on all T rows but advances the Mamba recurrence one token at a time.
    For the reply branch (109-132 suffix tokens), deepcopying the base cache and continuing it costs more than a
    fresh ~900-token prefill, which uses the chunked scan.
  - E1k tool turns got slower than E1j (first reply token later by about a tenth).
  - The standalone `reuse_check` reported a large first-token gain, but it is **confounded**: the no-reuse run went
    first and paid a cold start inside its decide step. NOT VERIFIED as a gain.
  - Decision and call were identical with and without reuse. The reply text differed by a few words (expected: the
    numerics are not bit-exact).

- **INT4 Nano**:
  - The M == 1 linear (tinygemm) is clearly faster than bf16 in the microbench, so decode should win.
  - tinygemm at M = 800 is an order of magnitude slower than bf16, so prefill loses.
  - The frankenstein dequant path (`e1/int4_dispatch.py`) beats tinygemm at M = 800 only modestly. It is far from
    bf16, because the dequant is written as unfused PyTorch ops with fp32 temporaries (about 4·N·K bytes per call).
  - Correctness: M == 1 is bit-identical to the old path. For M > 1, the max abs diff against the fp32 dequantized
    reference equals the old path's max abs diff.
  - In the session, INT4 Nano was much worse (§2.1), so per-token decode showed no INT4 gain.

### 2.3 What the numbers say, in one line each
1. Tool-turn latency is decode-count bound (call tokens) and serial (decide → call → reply).
2. Nano and the duplex loop serialize on one stream, so every Nano optimization also buys back frame time.
3. Reuse needs a chunked-scan continuation. Without one, re-prefilling is better for suffixes in the hundreds.
4. INT4 for Nano is a decode-only win until a prefill kernel exists. The current dequant path is a stopgap with a row
   threshold.

## 3. Web UI tree inventory (v2/src/mynah_v2, branch webui-v1; read-only)

Legend:
- **U**: applies unchanged to the mix.
- **A**: needs adapting.
- **C**: conflicts with the mix as built.
- **N/A**: not on the mix's hot path.

| Module / experiment | What it does | Mix | Notes |
|---|---|---|---|
| `model/llm_cache.py` | hybrid NemotronH cache (position injection, cache-passing attention, `_hybrid_cache_cls`) | U | Already used by the Nano worker on the bare HF model |
| `perf.mamba_host_pos` | removes the per-Mamba-layer host sync (`cache_position[0] > 0`) | U (Thinker) / A (Nano) | Hooks `engine.wrapper.model.stt_model.llm`. Nano needs the same patch on its own `llm` handle. Prerequisite for any graph |
| `perf.no_timer_sync` | removes the wrapper's timing `synchronize()` calls | U | Pure win for the duplex loop. It also stops the loop from draining Nano's queued work |
| `perception_graph`, `perception_nosync` | exact CUDA graph / sync removal for full-window perception | U | Perception code is unchanged in the mix |
| `thinker_graph.ThinkerGraph` | Thinker decode step as one CUDA graph per KV bucket; `_stream_neutral` block forward | U (Thinker) / A (Nano) | Thinker: the controller hooks (`_apply_rnnt_turn_taking`, `_apply_fc_state_machine`) and forced agent-channel writes run after the graphed forward, and the forced id only enters through the next frame's input embeddings, so they are outside the graph. Nano: same NemotronH remote code revision (6533e8de). Needs a constructor on a bare `llm` + `lm_head` instead of `engine` |
| `capture_guard.CaptureGuard` | safe lazy captures with helper threads | U, required | The Nano worker is exactly a helper thread that syncs (`int(argmax)`, `synchronize`). Captures must be pre-taken at load, before the worker starts, or guarded |
| `quant.quantize_thinker_int4` / `Int4Linear` | tinygemm INT4 for the NemotronH linears | U (Thinker) / A (Nano) | In use. M > 1 weakness → `Int4Dispatch` (`min_rows`) |
| `quant.quantize_thinker_w8`, `wo_gemv.WOLinear` (W8/W4/NVFP4 Triton, ≤ 16 rows) | weight-only GEMV for batched decode; `min_rows_kernel` pins 16-row chunks | A | Matches the Nano decode (M = 1) and the 16-row continuation chunks. Its own packing, so it cannot share weights with `Int4Linear` |
| `precision.SensitivityProbe` | teacher-forced per-layer-group sensitivity of a weight format | A | For Nano, the sensitive step is the tool-call arguments, not speech text |
| `fc_phase2_prefill.continuation_forward` | multi-token continuation on a live hybrid cache | U, but slow for long suffixes | Mamba advanced per token. See R6 |
| `fc_phase2_notts`, `fc_ctx_fix`, `fc_profile`, `codec_fc_*` | the Thinker's own function-call path | N/A | Tools go through Nano. The Thinker's FC tokens are suppressed |
| `eartts_*` (graph, q4, w8, bf16, rvq, mog, host_schedule), `codec_*`, `depthwise_q4`, `codec_cpu` | EarTTS + VAE codec | N/A on the hot path (`decode_audio=False`) | **Patterns to reuse for the Flash talker**: `StaticKV` + `FullStepGraph` (eartts_graph), `DeferredD2H` / `codec_nosync` (codec_async) |
| `rnnt_decoder_q4/w8`, `heads_w8`, `extended_w8`, `thinker_w8` | smaller Thinker-side formats | U | Low gain next to the Nano items |
| `session_state.SessionStateStore`, `MultiSessionEngine` | per-session swap of non-Thinker state | A | New per-session state in the mix: controller, Nano cache/worker queue, talker state, ASR state |
| `thinker_batch.BatchedThinker`, `runtime/rendezvous.py`, `batch_scheduler.py`, `batch_contract.py` | real batched Thinker across sessions | A | The duplex Thinker plugs in as is. Nano and the talker need their own batch backends with the same contract |
| `runtime/comp_timer.py`, `op_profiler.py` | per-component GPU/host timing | U | Used in E1j/E1k |

There are two direct conflicts, both manageable:
1. **Graph capture vs the Nano worker thread.** A capture breaks if another thread syncs during it, and the worker
   syncs on every token. Fix: capture the Thinker (and later the Nano) buckets at load, or take the CaptureGuard
   lock around worker steps while a capture is open.
2. **`_stream_neutral` is graph-mode only.** In eager mode the upstream stream switch stays on, so the Nano worker
   stays pinned to the default stream (§1). Moving Nano off the default stream needs the stream-neutral body in eager
   mode for the Nano model only (R5).

## 4. A shared, model-agnostic optimization layer

**Goal.** One layer that applies the same services (quantization, cache handling, graph capture, streams, batching,
profiling) to any of the engine's models through small per-model adapters. The web UI tree can adopt it later
without rewriting its modules. The layer wraps those modules rather than replacing them.

### 4.1 Per-model adapter (one per model family)

```python
class ModelAdapter(Protocol):
    name: str                                   # "nemotronh-thinker", "nemotronh-nano", "qwen3tts-talker", ...
    def linears(self) -> Iterable[LinearSite]   # (path, module, role: "mixer"|"mlp"|"head"|"codepred", shape)
    def cache_spec(self) -> CacheSpec           # attention layers (kvh, hd), Mamba layers (conv, ssm shapes), position
    def new_cache(self, batch: int, lmax: int) -> Any
    def clone_cache(self, cache) -> Any          # branch for a prefix store (deepcopy today)
    def prefill(self, cache, ids_or_embeds) -> Hidden            # T rows, chunked scan for Mamba
    def continue_(self, cache, ids_or_embeds) -> Hidden          # T rows on a live cache
    def step(self, cache, one) -> StepOut                        # one row: the unit graph capture works on
    def step_io(self) -> StepIO                  # static input/output tensors + bucket function for capture
    def stream_patch(self) -> ContextManager     # e.g. the stream-neutral block forward for NemotronH
    def host_syncs(self) -> list[PatchSpec]      # e.g. mamba_host_pos
```

Adapters:

- **NemotronH**: one implementation serves both the duplex Thinker (handle `engine.wrapper.model.stt_model.llm` plus
  its heads) and Nano (bare HF model plus `lm_head`). It reuses the code of `llm_cache`, `mamba_host_pos`,
  `ThinkerGraph` and `continuation_forward`.
- **Qwen3-TTS talker**: two parts.
  - The talker LM step (attention-only decoder, one row per 80 ms frame).
  - The code predictor, which runs **15 sequential steps per frame** on a short local context. Its natural graph
    unit is the whole 15-step loop: static KV for the local context, as `eartts_graph.StaticKV` does.
  - Codec decode is a separate `CodecAdapter`: `decode(codes[F], state) -> pcm`, with chunk size and overlap.
- **EarTTS / codec**: wraps the existing `eartts_*` / `codec_*` experiments unchanged, so the web UI loses nothing.

### 4.2 Services (model-agnostic, driven by the adapter)

| Service | Responsibility | Built from (web UI module) |
|---|---|---|
| `QuantService` | picks a kernel per (linear, row regime): M = 1 decode, M ≤ 16 batched/chunk, M large prefill; keeps one packed weight per site and dispatches by M | `quant.Int4Linear`, `wo_gemv.WOLinear` + `min_rows_kernel`, frankenstein `Int4Dispatch` |
| `GraphService` | captures `adapter.step` per (bucket, rows); replays; eager fallback with reasons | `thinker_graph.ThinkerGraph` (generalized), `eartts_graph.FullStepGraph`, `capture_guard` |
| `CacheService` | allocate/clone/branch caches; a prefix store keyed by token-prefix hash; chunked-scan continuation | `llm_cache`, `fc_phase2_prefill`, `session_state` (per-session ownership) |
| `StreamService` | one stream per worker with priority; events instead of host syncs; stream-neutral patches | `thinker_graph._stream_neutral`, `codec_async.DeferredD2H`, `perf.no_timer_sync` |
| `BatchService` | the `ThinkerBatchBackend` contract generalized to any adapter: attach/detach/positions/step/step_one/memory | `runtime/batch_contract.py`, `thinker_batch`, `rendezvous`, `batch_scheduler` |
| `ProfileService` | named, reversible profiles; A/B gate (token identity for exact items, corpus + ASR for the rest) | `perf.py` registry (`experiment`, `enable`, `disable`, `PROFILES`), `comp_timer` |

### 4.3 What the web UI modules would need to expose (no edits now)

1. `ThinkerGraph(llm, heads, lmax)`: a constructor on a bare NemotronH backbone plus a heads callable, next to the
   current `ThinkerGraph(engine)`.
2. `mamba_host_pos` and `no_timer_sync` taking the `llm` handle instead of finding it through `engine`.
3. `quantize_thinker_int4` / `quantize_thinker_w8` taking a module tree and a site list (TARGETS already exists).
4. `batch_contract`: `ThinkerBatchBackend` renamed or aliased to a model-keyed `StepBatchBackend`;
   `SessionStateStore` field specs registrable by an adapter (controller, Nano queue, talker state).
5. `CaptureGuard`: register extra helper threads (the Nano worker, a talker thread). `helper_threads` exists; it
   needs a public registration call.
6. `perf.py`: accept experiments contributed by an adapter, so `PROFILES` can name "nano-graph", "talker-graph", and
   so on.

## 5. Ranked optimizations for the mix

Order: expected user-visible gain (first audio on tool turns, real time), then risk.

| # | Optimization | Expected gain | Risk | CPU work first | GPU validation (cap) |
|---|---|---|---|---|---|
| R1 | **Speak a short per-language holding phrase as soon as decide = TOOL**, while the call is generated and executed. The controller opens the turn and speaks it; the reply continues the same turn | First audio on tool turns moves from "after call + reply prefill" to "after decide" (the 27-35 call tokens leave the critical path) | Double-talk or awkward phrasing; the phrase must be in the session language and must not promise a result | controller action + tests (stale/barge-in during the phrase) | G1: E1 session, 6 min |
| R2 | **Constrained tool-call decoding**: force `{"name": <enum>, "arguments": {` and the argument keys from the schema; generate only the values | Call tokens drop to the value tokens (names, keys and punctuation are forced) | Grammar bugs; values that need free text still decode | grammar from the bound schemas + CPU tests on recorded calls | G2: tool bench IT/EN (13 prompts each): decision, params, execution identity, 8 min |
| R3 | **Nano decode as a CUDA graph** (GraphService on the NemotronH adapter; `mamba_host_pos` on Nano; captured at load before the worker starts) | Less host launch overhead per token: faster decide, call and reply, and less time holding the shared stream | Capture vs worker races; bucket transitions | adapter constructor; parity test harness | G4: greedy token identity vs eager on 13 prompts + per-token time, 8 min |
| R4 | **INT4 Nano, done per regime**: tinygemm for M < `min_rows` (decode, chunks), a fused Triton dequant (one pass, nibble → bf16, no fp32 temporaries) or an int4 GEMM for prefill | Faster decode (fewer weight bytes); prefill at least as fast as bf16 | Tool-argument quality (probe the call step first); memory (nibbles + scratch) | kernel + CPU reference tests; `min_rows` is already in | G3: microbench sweep M ∈ {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 800}, crossover + max diff, 4 min; then the session, 8 min |
| R5 | **Nano on its own stream with lower priority** (stream-neutral block forward in eager mode, Nano model only) | Frames no longer wait behind Nano tokens; the loop keeps real time | The upstream comment cites NaNs on multi-GPU; E1h showed a race when only part of the model moved | patch + token-identity test harness | G5: 13 prompts token identity default vs side stream, then the session frame p95, 6 min |
| R6 | **Prefix cache across turns + chunked-scan continuation** (Mamba `mamba_chunk_scan_combined` with initial states instead of per-token steps) | No re-prefill of the shared system/tools prefix (744+ tokens) per request; the reply branch costs one chunked continuation | Not bit-exact vs a fresh prefill; must hold decision/call identity | CacheService prefix store; continuation in the adapter | G6: decision/call identity on 13 prompts + first-token comparison with the cold start excluded (warm both arms), 6 min |
| R7 | **Duplex Thinker "exact" profile**: `mamba_host_pos`, `no_timer_sync`, `perception_graph`, `thinker_graph` | Lower per-frame host time; more room for Nano and the talker | Graph vs the controller hooks (outside the graph by construction) | none (modules exist) | G7: E1 session with the controller, token identity on all channels vs eager, 6 min |
| R8 | **Talker per-frame graph**: talker LM step + the 15-step code predictor as one graph per bucket (StaticKV pattern) | The talker is fifteen-plus small steps per 80 ms frame, so it is likely launch-bound | Correctness of static KV for the local predictor context | adapter + CPU shape tests | G8: standalone talker, frames per second vs eager + code identity, 8 min (after G0) |
| R9 | **Chunked codec decode** for the talker (decode every N frames with overlap; deferred D2H) | Removes per-frame codec syncs; better GPU occupancy | Audio seams at chunk edges; first-audio latency grows with N | chunk/overlap scheduler + CPU tests on synthetic codes | G9: ASR + listening check on 5 replies, 6 min |
| R10 | **Batching across sessions**: duplex Thinker via `thinker_batch`; Nano requests from several sessions in one decode batch; talker steps batched | Throughput per GPU | Scheduling complexity; per-session state isolation | BatchService contract tests (fakes) | later; no cap requested yet |

Not recommended now:
- KV reuse through the current `continuation_forward` for long suffixes (E1k).
- The dequant path below a few hundred rows.
- More INT4 on the duplex Thinker: it is already INT4, and the frame cost is now dominated by contention.

## 6. GPU validations (none run; each needs a GO)

| Id | What | Cap | Pass rule |
|---|---|---|---|
| G0 | Measure the missing loads: live streaming ASR + the Flash talker in the loop with the controller (bf16 Nano) | 8 min | frame p50/p95 with all components; no new leaks or stale audio |
| G1 | R1 holding phrase | 6 min | first audio on tool turns after decide; spoken text = phrase + Nano reply in order; tools 3/3 |
| G2 | R2 constrained calls | 8 min | decision and params identical or better than unconstrained on 13 IT + 13 EN prompts |
| G3 | R4 INT4 crossover sweep | 4 min | crossover row count; M == 1 bit-identical; max diff ≤ the old path's |
| G4 | R3 Nano decode graph | 8 min | greedy token identity vs eager on 13 prompts |
| G5 | R5 Nano side stream | 6 min | token identity vs the default stream; frame p95 lower |
| G6 | R6 prefix cache + chunked continuation | 6 min | decision/call identity; warm-vs-warm first-token comparison |
| G7 | R7 Thinker exact profile | 6 min | token identity on all channels vs eager |
| G8 | R8 talker graph | 8 min | code identity vs eager; frames per second |
| G9 | R9 chunked codec | 6 min | ASR of the replies equals the reply text; no audible seams in a listening check |

## 7. NOT VERIFIED
- Any gain of R1-R10. All are proposals.
- The INT4 `min_rows` crossover (default 256 is a placeholder between the two measured points, M = 1 and M = 800).
- The talker's real per-frame cost and the live ASR cost. Neither has run in the E1 loop.
- Whether deepcopying the hybrid cache is itself a material cost. It was not timed separately from the continuation.
- The upstream multi-GPU NaN concern behind the stream switch, on a single GPU.
