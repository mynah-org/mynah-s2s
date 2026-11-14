# Running and reproducing the tests

Everything below is driven by `make` (see `make help`), which calls `v2/scripts/mynah.sh`. All state lives under
`MYNAH_HOME` (default `~/mynah-s2s`): `.venv-v2` (engine), `.venv-asr` (evaluation ASR), `cache/pylib-tts` (test
TTS), `hf-cache/` (weights), `data/` (rendered test audio), `results/` (runs).

## 1. Install and check

```bash
make setup      # core runtime only: ~20-40 min the first time (weights ~42 GB, CUDA kernels compiled for your GPU);
                # a few minutes with MYNAH_HF_CACHE pointing at a cache that already has the weights (setup links it
                # as $MYNAH_HOME/hf-cache and may write refs / locks into it)
make setup-all  # core runtime + synthetic test-audio TTS + evaluation ASR (what the test suites use)
make doctor     # core lines must be OK (else MISSING + the make target that fixes it); TTS / ASR / audio are optional
```

`make setup-tts` / `make setup-asr` add or repair only those parts; `make reproduce` installs whichever is missing.
Without the ASR, everything still runs; only the audio check is skipped (the scripts say so). On aarch64 the ASR
does not install out of the box (its `fairseq2n` dependency has no wheels there); `make setup-all` then ends with
`SETUP_DONE (PARTIAL ...)` and `make doctor` lists the ASR as ABSENT. Typical durations on a 46-48 GB GPU: `make
test-cpu` < 1 min, `make test-gpu` 15-20 min, `make corpus` ~3 min, `make e2e-fc` ~10 min, `make multi` ~10-15 min.

## 2. Test audio

`make corpus` renders the caller side of every test with a synthetic voice (Qwen3-TTS 0.6B CustomVoice, pinned
revision, fixed seeds): `data/corpus_v1` (40 items), `data/capability_v1` (9), `data/fcmulti_v1` (7 multi-call
conversations spliced from the corpus clips). Item definitions are in `v2/eval/*.json`.

| prefix | what the caller does | what is checked |
|---|---|---|
| `q01-q12` | arithmetic, reasoning, general knowledge ("What is the capital of France?") | answer, no tool call |
| `s01-s10` | small talk | a sensible answer, no tool call |
| `r01-r04` | asks for a random number in a range | `generate_random_number` with the right min / max |
| `t01-t12` | asks for an order by a spelled ID (two IDs do not exist) | `get_order_status` argument, status and date spoken |
| `t13` | gives an order ID, then corrects it while the call is running | the second call uses the corrected ID |
| `t14` | asks for an order, then for its tracking code | follow-up answered from the same result |
| `m01, m02, long01, i01-i04` | consecutive calls, a long session, speech during an open call | calls, grounding, no crash |

## 3. Single-session runs

- `make demo ITEM=t02` prints the agent's internal text, the tool calls and (if the ASR is installed) the ASR of what
  it actually said; `combined.wav` lets you listen to both sides.
- `make e2e-fc` runs the tool-call items above; `make e2e-corpus` runs all 40 items. Both finish with
  `score_corpus.py`: per category, model text vs expectations, tool arguments, and the ASR of the spoken audio.
  The scores cover the single-call corpus items; the multi-call conversations of `e2e-fc` (`m01`, `i01`) are run and
  listed (calls and text in their `summary.json`) but not scored automatically.

Three evidences are kept separate on purpose: (1) the model's internal text and tool calls (`summary.json`), (2) the
ASR of the spoken audio (`asr_omni3b.txt`), (3) the tool ground truth (`orders.sqlite`, corpus `expect`). The
internal text can be right while the audio is wrong, and the opposite.

What to expect from the model (our run, details in `docs/benchmarks.md` and the research summary): short factual
and arithmetic answers are reliable; spelled IDs are the main source of tool errors (about 7 of 12 order IDs heard
correctly); a misheard ID leads to an honest "I could not find an order", not an invented one.

## 4. Several sessions at once

`make multi` runs `rendezvous_pair.py` on two groups of four sessions (`m4_notools`, `m4_mixed`: two of the four
make a tool call). Each session runs once alone and once concurrently; the gate checks that tool calls, arguments
and order are identical, that no data leaks across sessions, and reports round latency (p50 / p95 / max against the
160 ms budget), per-session lag, throughput and the call cycle (`summary.txt`), then prints one verdict line per
group: `GATE PASS` when every session made the same calls alone and together, with no leak and no error. Latency is
reported, not judged (real time needs every round ≤ 160 ms; on GPUs slower than real time every round misses).
`ok_solo` / `ok_pair` are the model's answer checks for each run and can differ between runs of the same item: the
concurrent run is not token-identical to the solo one (batched kernels), so the wording may differ.

## 5. Your own scenario and persona

A scenario is a JSON file in the corpus format plus an optional system message:

```json
{"name": "my-shop", "lang": "en", "tts": {...copy from v2/eval/corpus_v1.json...},
 "orders": [["OAK417", "shipped", "November 20, 2026", "5512"]],
 "items": [{"id": "c01", "category": "tool_orders", "speaker": "ryan", "tail_s": 20.0,
            "utterances": [{"offset": 3.0, "text": "Where is my order O A K four one seven?"},
                           {"offset": 6.0, "text": "And when will it arrive?"}],
            "expect": {"function": "get_order_status", "order_id": "OAK417"}}]}
```

`utterances` are what the caller says, in order; `offset` is the silence before each one (seconds), so several
utterances make a multi-turn conversation between the synthetic caller and the agent. Speakers: `ryan`, `vivian`
(Qwen3-TTS presets). `orders` seeds the mock order database.

```bash
make custom                                                   # the shipped example: v2/eval/examples/furniture_store*
make custom SPEC=path/my_shop.json MSG=path/my_system.txt     # yours (paths relative to where you run make)
```

The caller audio of a scenario is rendered once and cached by file name and content
(`$MYNAH_HOME/data/custom-<name>-<hash>`): editing the JSON renders it again.

The system message is placed in NVIDIA's function-calling prompt template together with the tool schemas. The same
option exists on the server: `make serve ARGS="--system-message @path/my_system.txt"`; a WebSocket client may also
send its own full `system_prompt` in `session.start`. New tools need code: see `v2/src/mynah_v2/agent/tools.py`
(`ToolRegistry.register`, and `run_stream.py --tools`).

## 6. Realtime server

```bash
make serve                                   # ws://127.0.0.1:8765/v1/session, one session at a time, enhanced mode
make client WAV=$MYNAH_HOME/data/corpus_v1/t02.wav   # in another shell: streams at real-time pace, saves agent audio + events
```

`make serve` listens on localhost; `ARGS="--host 0.0.0.0"` exposes it on the network. On a GPU slower than real
time the server queues at most 10 s of inbound audio and then drops the oldest (`audio.dropped` events; the client
prints a WARNING): raise it with `make serve ARGS="--max-backlog 120"`. `make ws-check` (server + client, one item,
`WS_ITEM=q03` by default) uses a large backlog and fails if audio was dropped or the agent did not answer.

The protocol (JSON control messages, binary PCM16 in at 8-48 kHz, 22.05 kHz agent audio out) is documented at the top
of `v2/src/mynah_v2/transport/ws_server.py`. Multi-session batching is not exposed by the server yet; it is
exercised by `make multi`.

## 7. Configuration

| preset | what it is | outputs |
|---|---|---|
| `release` (default) | CUDA graphs, batched stages (multi-session), fc_merge, RNN-T overlap (multi-session: the user-side RNN-T decode runs while the batched Thinker does), fused EarTTS RMSNorm, int8 weight-only Thinker MLP, codec async copies, cached RVQ tables (`eartts_rvq_cache`), deterministic mode without the NaN fill of uninitialized memory (`--fill-uninit off`), flash / memory-efficient attention allowed (`--sdpa fast`) | reproducible run-to-run on the same GPU and software; not bit-identical to `release-exact` (attention sums in another order) |
| `release-exact` | the same with `--sdpa default` | bit-identical to the deterministic reference line |
| `reference` | NVIDIA's reference step, no V2 execution optimisation | NVIDIA's numerics (slower; parity checks) |

Tool-call policy (ENHANCED, on by default, also in `serve.py` and the multi-session engines): the per-session call
guard and the outcome-driven resume contract (`architecture.md`, "Function calls"). `run_stream.py` options:
`--no-call-guard` (guard off, the engine's own resume contract), `--guard-resume by_outcome|pad_eotr|pad_pad|engine`,
`--guard-echo` (a refused repeat restates the earlier result; experimental). The run summary records every guard
decision and the contract used at each resume (`call_guard`).

Any single flag can be overridden, e.g. `make multi ARGS="--fc-merge off"`, `make demo ARGS="--sdpa default"`,
`ARGS="--deterministic off"` (NeMo's serving default: not reproducible run-to-run). The determinism switches are plain
PyTorch settings and work on any GPU: where flash attention is not available PyTorch picks the next attention kernel.
The effective state is printed (`DETERMINISM {...}`) and stored in each run summary (`preset`, `determinism`,
`perf_profile`).

If an optimisation cannot be enabled on your hardware:
- **demos and serving** (`run_stream.py`, `serve.py`): the run prints `WARNING optimisation ... skipped` and
  `OPTIMISATIONS active=... skipped=...` and continues; the summary records it (`--strict` makes it an error);
- **regression / performance gates** (`rendezvous_pair.py`, `make multi`): strict by default, the run stops, so a
  comparison never runs on a silently different configuration (`--no-strict` measures the fallback on purpose).
