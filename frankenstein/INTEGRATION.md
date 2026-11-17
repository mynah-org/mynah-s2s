# Frankenstein S2S: integration plan into the main tree (proposal)

This is a plan only: nothing is merged and the web UI branch is not edited. It covers the server side, the
launcher and the UI.

**Target.** One tree on `main` where `make webui-gpu` starts either:
- the base VoiceChat model (as today), or
- the Frankenstein engine: streaming multilingual ASR → duplex Thinker as a timing shell → text LLM (tools, holding
  phrase, constrained calls) → talker (Qwen3-TTS structure, zero-shot clone voice).

The UI stays the current one, slightly adapted (language selector, voice choice). Quantization tooling is shared.
Frankenstein must meet the V2 server-side bar:
- batching across sessions;
- identifiers in every event;
- the same WebSocket protocol;
- resource limits;
- clean teardown.

## 1. What exists and is reused

### V2 server side (`main`, `v2/src/mynah_v2`)

| Piece | What it gives | Use for Frankenstein |
|---|---|---|
| `transport/ws_server.py` | one connection = one session. `session.start` / binary PCM16 in / `session.end` / `session.cancel`; out: `session.created`, binary PCM16, `{"type": "event", "ev": ...}`, `session.closed`. Bounded inbound backlog, idle timeout, close timeout → unhealthy, `?probe=1` | **Unchanged** if the engine implements the same `Engine` protocol (below). Output rate must become a field the engine declares (talker 24 kHz vs EarTTS 22.05 kHz) |
| `runtime/session.py` `Session` / `Engine` protocol (`open`, `step(pcm16k)`, `fc_active`, `close`) | chunking, pacing, drain, flush, tracing hooks | Frankenstein becomes an `Engine`. Its `step` runs one 160 ms chunk: perception + Thinker frames + live ASR + controller tick, plus non-blocking hand-off to the text-LLM worker and the talker |
| `tracing/events.py` | canonical JSONL events with `session_id`, `turn_id`, `call_id`, model step, audio frame | Frankenstein already emits `ctl.intent` / `ctl.action` / `adapter.*` / `talker.*` / `brain.*` with session and turn ids (`frankenstein_s2s/tracing.py`). They need mapping to canonical names: `text.generated`, `tool.pending` / `running` / `completed`, `audio.generated`, `audio.playback.started` |
| `agent/tools.py` (`ToolRegistry`, `ToolLedger`, `bind`) | call ids, timeouts, ledger states | already used by the Frankenstein text-LLM worker |
| `agent/call_guard.py` | admit / refuse a call before execution | reusable as is in the worker, before execution |
| `presets.py` and `model/perf.py` profiles | named, reported configurations, fallback vs strict policy | add Frankenstein presets (e.g. `frank-int4`) with the same "report what is really active" rule |

### Web UI branch (`webui-v1`, not on `main`)

| Piece | Use |
|---|---|
| `webui/server.py` bridge (static UI, browser WebSocket ↔ V2 WebSocket proxy, demo mode, profiles `int4` / `q8` / `bf16` / `demo`) | **Reused.** Add an `engine` field (`voicechat` / `frankenstein`) next to `profile`, and pass `language` / `voice` through in `session.start` |
| `webui/index.html`, `static/app.js` (scenario, backend and input selectors; event stream; session/call ids) | **Reused.** Add LANGUAGE and VOICE selectors. Show the holding phrase and the barge-in cut in the event stream |
| `Makefile` `webui-gpu` (starts `serve`, waits for `READY`, then the bridge) | add `ENGINE=frankenstein`, which selects the Frankenstein server entry point and its preset |
| `model/quant.py` (INT4 tinygemm `Int4Linear`, W8), `wo_gemv.py` (Triton W8/W4/NVFP4 GEMV), `precision.py` (sensitivity probe) | **Shared quantization layer.** Frankenstein already uses `Int4Linear` for the Thinker and, decode-only, for the text LLM. The W8/NVFP4 Triton kernels are candidates for the text LLM's decode |
| `model/thinker_graph.py`, `capture_guard.py` | already reused unchanged: the Thinker graph, and the text LLM's decode graph through a shim (`frankenstein/e1/nano_graph.py`) |
| `runtime/batch_contract.py`, `thinker_batch.py`, `session_state.py`, `rendezvous.py`, `batch_scheduler.py` | the multi-session model. Frankenstein needs one backend per grafted component under the same contract (§3) |

## 2. Gaps

1. **Engine entry point.** Frankenstein currently runs from experiment scripts (`frankenstein/e1/*`) that wrap the
   NVIDIA wrapper's per-frame hooks at instance level. It needs a `FrankensteinEngine(Engine)` class that owns:
   - the duplex model (Thinker only, `decode_audio=False`);
   - the live ASR, the controller and the text-LLM worker;
   - the talker client;
   - per-session state.
2. **Output audio path.** V2 sends talker PCM from inside `step`. Frankenstein's talker audio arrives
   asynchronously from another process. The engine must forward delivered chunks to the transport, epoch-gated,
   with cancel = stop + flush (the playback model in `e1/flash_client.py` becomes the real output queue).
3. **Event names and ids.** Map Frankenstein events to the canonical set. Every event must carry `session_id`, and
   `turn_id` / `call_id` where they apply. The controller's request ids and the talker's epochs go in as extra
   fields.
4. **Language and voice.** `session.start` gains `language` (session language: ASR prompt, text-LLM instruction,
   talker codec language) and `voice` (a built-in voice name or a reference clip id).
   - Talker languages are the Qwen3-TTS codec language table: Chinese, English, German, Italian, Portuguese,
     Spanish, Japanese, Korean, French, Russian.
   - The ASR prompt covers a superset.
   - The UI offers the intersection.
5. **Batching across sessions.** Missing for every grafted component (§3).
6. **Resource limits.** Per-session caps are needed for:
   - the text LLM's prompt length and reply tokens (the brief-reply cap exists);
   - talker frames per segment (a runaway cap exists);
   - queue depths between controller, worker and talker;
   - a global admission limit (`max_sessions`) based on measured memory.
7. **Teardown.** On close or cancel:
   - stop the worker request;
   - cancel the talker epoch;
   - drop the talker process's per-session state;
   - release the graph buffers (the decode graph binds one cache at a time);
   - verify nothing of the session is left in any queue (a teardown test like V2's).

## 3. Batching design (per component)

| Component | Today | Batched form | Notes |
|---|---|---|---|
| Duplex Thinker + perception | V2's `thinker_batch` / `rendezvous` (works for VoiceChat) | the same: Frankenstein only reads the Thinker's BOS/EOS and suppresses its function channel | instance hooks must become per-session, behind the rendezvous baton |
| Live ASR | one stream, sliding-window re-encode every 4–8 frames | batch the per-session windows (same shape when aligned to chunk boundaries) into one encoder call per round; RNNT greedy stays per session (small) | or the real cache-aware streaming step, which is cheaper per call |
| Text LLM (Nano-9B structure) | one worker thread, default CUDA stream, one request at a time | a request queue shared by all sessions with continuous batching of decode steps (one row per active request). The INT4 decode path and the decode graph need batched variants: tinygemm at M = rows, a graph per row-count bucket (V2's `thinker_batch` already does this for the same NemotronH structure) | the default-stream constraint (§5) holds per process, so batching in one thread is the natural fit |
| Talker | separate process (its own environment), one segment at a time | one talker server for N sessions: batch the per-frame talker step and the code-predictor graph over active segments (same shapes per step); codec decode batched per chunk | see the process decision in §5 |

## 4. Step order with gates

1. **Engine class, one session, behind a flag.**
   - `FrankensteinEngine` with the existing components.
   - The WebSocket server unchanged except the declared output rate.
   - Gate: `ws-check` round trip on an Italian item; the trace contains the canonical events with ids; no stale
     audio after a cancel.
2. **Language / voice in the protocol and the UI.**
   - Gate: IT / EN / FR sessions from the UI. The session language is reported in `session.created`, and ASR,
     text and talker all use it.
3. **Presets and quantization sharing.**
   - `frank-int4` = INT4 Thinker + text-LLM INT4 decode + decode graphs + code-predictor graph.
   - The preset reports what is really active.
   - Gate: the same tool-suite identity checks as G6, run as a CPU-reported, GPU-executed preset check.
4. **Teardown and limits.**
   - Gate: 50 open/cancel cycles without memory growth or leftover queue items; idle-timeout and close-timeout
     behaviour equal to V2.
5. **Multi-session, by component.** In order: Thinker via the rendezvous, then text-LLM batching, then the talker
   server, then ASR batching.
   - Gate per step: isolation (no cross-session audio or text), per-session latency vs single-session, the V2
     `multi` targets adapted.
6. **`make webui-gpu ENGINE=frankenstein`.**
   - Gate: the V2 corpus and FC suites in the session language (EN gates as in V2; IT/FR as Frankenstein gates);
     ASR of the spoken audio; tool ground truth.
7. **Merge to `main`** only after all gates, with the owner's GO.

## 5. Risks

| Risk | Why | Mitigation |
|---|---|---|
| Text LLM pinned to the default CUDA stream | the NemotronH remote code runs every block on the default stream; a side stream raced (E1h). Stream priorities therefore cannot be used for it, and its kernels serialize with the Thinker's | one model-executor thread per process for both NemotronH models (Thinker and text LLM) interleaving their steps; or the stream-neutral block forward (as the graphs use) validated in eager mode too; or the text LLM in its own process |
| Talker in a separate process | the talker package needs transformers 4.57 while the engine's NeMo stack pins 4.56: importing it into the engine process fails | keep the process, give it a real protocol (the JSON-lines driver here becomes shared memory or a pipe for PCM) and supervise it (restart, health); or vendor the talker's model code against 4.56 (a port, with code-identity checks) and run it in-process |
| Cross-process GPU contention | the talker runs about 2x slower in the loop than alone; stream priority does not cross processes (G4) | batching (fewer, larger kernels), the code-predictor graph (G5), an MPS daemon (process priorities), or a dedicated GPU for the talker |
| Text-LLM decode speed | it is the critical path of tool turns | INT4 decode-only, decode graph, batched decode (G6 onwards), shorter replies, earlier hand-off to the talker |
| Unified-memory hosts | a second CUDA context alive during a big load ended in a CUDA OOM (G5) | load order (engine first, talker last), admission check on free memory, never two big loads at once |
| Audio clock vs text clock | the turn now closes at the end of its audio; a slow talker stretches turns and makes the next caller speech a barge-in | measure audio-stop latency and end-of-audio in every gate, with the reactive user track |
| Quant drift on tool arguments | INT4 / NVFP4 change logits | the tool-suite identity check (decisions + calls) as a promotion gate for every format |

## 6. What not to do

- Do not fork the WebSocket protocol: extend `session.start` and `session.created` with optional fields only.
- Do not copy V2 modules into `frankenstein/`. Import them; where an entry point is engine-specific (the Thinker
  graph), wrap it with a shim, as `e1/nano_graph.py` does.
- Do not put a second transformers version into the engine's venv.
