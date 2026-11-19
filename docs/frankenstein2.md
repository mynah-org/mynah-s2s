# Frankenstein v2: the recommended engine

**Status: experimental, work in progress.** Frankenstein v2 is the engine we recommend for working with this
repository today. It is multilingual and light: about 14 GB of open weights for three languages, one GPU process.

It is a full-duplex voice agent assembled from separate open models, with no training:

| part | model | licence |
|---|---|---|
| **ears**: streaming transcript + turn-taking (end of turn, barge-in, backchannel) | [NVIDIA Nemotron 3.5 ASR Streaming 0.6B](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) with the [Tyger](https://huggingface.co/nineninesix/tyger) turn-taking GRU (one bundle) | OpenMDW-1.1 |
| **brain**: the reply and the tool calls | [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B), int8 weights on our own inference engine | Apache-2.0 |
| **talker**: the voice | [Kyutai Pocket TTS](https://huggingface.co/kyutai/pocket-tts), 24-layer language models, voice cloning from one reference clip, 4-bit weights | CC-BY-4.0 |

The VoiceChat engine (`v2/`, one 11B duplex model, English only) and Frankenstein v1 (`frankenstein/`) stay in
the repository unchanged. Frankenstein v2 lives in `frankenstein2/`. Its design notes are in
[`frankenstein2/DESIGN.md`](../frankenstein2/DESIGN.md).

## Why it is the default

- **Languages.** Italian, English and French are wired and tested end to end. More are possible with the same
  models: see [`frankenstein2-languages.md`](frankenstein2-languages.md).
- **Light.** One process holds the three models (`--runtime single`), several sessions share them, and each session
  has its own language and voice.
- **Tools are first-class.** Calls are parsed from the LLM's own format and run asynchronously. Arguments are checked
  against what the caller said. Every call is traced (see below).
- **Inspectable.** Every decision is an event with a session, turn and call id. The events go to a short log, a
  SQLite store and the web UI's debug view.

## Quick start

```bash
# 1. a Python environment with the runtime (torch with CUDA first, then the rest)
pip install -r frankenstein2/requirements.txt
# 2. the weights: Italian, English, French (the Pocket TTS repository is gated: accept its terms on the model page)
export HF_TOKEN=...            # a Hugging Face read token
make f2-setup                  # = python frankenstein2/scripts/setup_models.py --langs it,en,fr
#    other languages: make f2-setup F2_LANGS=it,en,fr,de   (dry run: add --dry-run to the script)
# 3. run
make f2-webui                  # server + web UI on http://127.0.0.1:8787 (pick GPU backend, a language, Connect)
make f2-serve                  # or only the WebSocket server on :8772
```

`f2-setup` writes the model paths to `~/mynah-s2s/hf/f2.env`. The other `f2-*` targets read that file
(`F2_ENV=...` to move it, `F2_PYTHON=...` to choose the interpreter, `F2_ARGS="..."` for extra server flags).
Server options are listed by `python frankenstein2/server/serve.py --help`.

The default tool back end is the example furniture store in `labs/multilingual/scripts/agentbench/` (product search,
customer and order lookup, escalation), over a small local table. `--tenant` selects another tenant file.

## Tool calls

The LLM writes a call in Qwen3.5's tool format. The engine never lets the call markup reach the voice. What the
caller hears:

1. **A holding phrase at once** ("Un momento, controllo." / "One moment, let me check."). It is spoken as soon as
   the model decides to call a tool, and when the model says something like that itself.
2. **The calls run asynchronously.** All calls of a round start together on a thread pool. The engine keeps
   listening and speaking for every session, and nothing sleeps while it waits. The caller can interrupt; the
   results of an interrupted request are dropped. Arguments are normalised first (spoken numbers to digits; codes
   such as postcodes checked against what the caller said).
3. **A reminder if it takes long.** When a call is still running and the agent has been silent for about a second,
   it says a reminder ("Ci sono quasi, ancora un attimo." / "Almost there, just a moment more."). That is at most
   twice per turn, 4 s apart, and only in silence.
4. **A timeout.** A call still running after `--tool-timeout` seconds (default 10; 30 for slow back ends) is
   stopped. If every call of the round timed out, the caller hears a fixed apology in the session language ("Mi
   dispiace, non sono riuscito a recuperare l'informazione in tempo...") without another LLM round. A late result
   is recorded as late and never used. `--tool-timeout-reply model` lets the LLM answer from the timeout error
   instead.
5. **The answer.** The results go back to the LLM in call order and the reply is spoken. A turn that only
   announced a lookup, without making the call, is continued with the call opened, so the model completes the call
   it announced.

Holding phrases and reminders never enter the conversation history; completed calls and their results do. A call
still running when the caller interrupts is dropped from the history (the trace keeps it).

## Tracing and debugging

Every session emits events (`frankenstein2/tracing.py`): ears, transcript, controller intents and actions, LLM
requests, tool calls, talker segments and audio. Three listeners are on by default in the server
(`frankenstein2/obs.py`):

- **Short log lines** in `TRACE_DIR/events.log` (`--log-events stdout` for the console). Texts are cut and
  per-frame noise is skipped:

  ```
  [3f2a turn=2 +12.48s] asr: "vorrei sapere dove è il mio ordine" (endpoint)
  [3f2a turn=2 +12.61s] llm: request r2 (5 messages)
  [3f2a turn=2 call=call1 +14.70s] tool: calling find_customer(full_name="Mario Rossi", postcode="10121")
  [3f2a turn=2 call=call1 +15.02s] tool: find_customer -> completed in 318 ms {"found": true, ...}
  [3f2a turn=2 +15.40s] tts: "Il suo ordine è in consegna."
  ```

- **A SQLite trace store**, `TRACE_DIR/f2_traces.sqlite`, one file per server. It has three tables:
  - `sessions`: one row per session;
  - `events`: wall time, monotonic time, the 80 ms step, the turn / request / call ids, the short line and the full
    JSON;
  - `spans`: intervals with a duration and a status, for each turn, LLM request, tool call and time to first audio.

  The full texts, tool arguments and results stay in the store for debugging. `--trace-no-text` drops user and
  agent text from every trace. The span layout follows OpenTelemetry's model (session = trace); exporting to an
  OTLP collector is not implemented. To view it:

  ```bash
  make f2-traces                          # the sessions
  make f2-traces SESSION=3f2a             # one session's timeline (an id suffix is enough)
  make f2-traces SESSION=3f2a SPANS=1     # its spans with durations
  sqlite3 .f2-traces/f2_traces.sqlite "select name, dur_ms, status from spans where kind='tool'"
  ```

- **The web UI.** The Activity stream shows the events. The **Debug log** switch under it shows the short lines
  live; it is off by default and the browser remembers it.

The per-session JSONL trace (`TRACE_DIR/<session>.f2.jsonl`, every event) is still written, for offline analysis.

## Tests

| command | what | needs |
|---|---|---|
| `make test-f2` | unit tests: control, pipeline, tools (async, timeouts, reminders), tracing, text, setup plan | Python, pytest, numpy, websockets; **no torch, no GPU, no weights**. This is what CI runs (`.github/workflows/frankenstein2-tests.yml`) |
| `make test-f2-torch` | + CPU tests of our Qwen engine pieces on tiny random weights | torch |
| `make test-f2-gpu` | + GPU smoke tests on the real models: Pocket speaks each language and the ears hear it, Qwen makes the right tool call | CUDA, `make f2-setup` |

A plain `pytest frankenstein2/tests` runs the first tier on any machine. The GPU and torch tiers are opt-in
(`--gpu`, `--torch`, or `F2_GPU_TESTS=1` / `F2_TORCH_TESTS=1`); without them those files are not even imported.

## Limits (read before relying on it)

- Languages beyond it / en / fr are not wired yet (holding phrases, number normaliser, prompts): see the language
  page.
- The turn-taking GRU was trained on English. Italian and French work in our tests with per-language end-of-turn
  thresholds (`--ep-thr it=0.85`).
- Concurrency: several sessions share one GPU. How many stay in real time depends on the GPU. We have not published
  measured capacity for this engine yet.
- The tool back end in the repository is an example tenant over a local table, not a production integration.
- Voice cloning: Kyutai's terms forbid cloning a voice without the speaker's explicit and lawful consent.
