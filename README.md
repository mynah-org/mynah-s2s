# mynah-s2s

**Status: work in progress, experimental research code.** Not a production system.

mynah-s2s is an open research project on **open-weight full-duplex speech-to-speech (S2S) voice agents**: models
that listen while they speak, take turns on their own, and call tools in the middle of a conversation. We surveyed
the open S2S ecosystem (Moshi-family models, PersonaPlex, MoshiRAG, NVIDIA VoiceChat, omni models), studied why tool
calls fail or are grounded wrongly, and built **V2**, an experimental runtime for
[NVIDIA NemotronLabs VoiceChat 11B](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-VoiceChat-11B).

## What V2 is

V2 splits the problem in two, on purpose:

- **Protocol and model: NVIDIA's reference, unchanged.** The per-frame step logic (input fusion, sampling,
  turn-taking, the function-call state machine with forced tool-response injection, PAD handling, end-of-response)
  and the model modules come from NVIDIA NeMo Speech, used without modifying upstream files. Keeping the reference
  step lets us check token-level parity with NVIDIA's own driver.
- **Execution engine: new.** How that step executes is ours: a multi-session coordinator that turns per-session
  steps into batched GPU work, a batched Thinker backend with static state and CUDA graphs, int8 weight-only GEMV
  kernels, a full-step CUDA-graphed EarTTS with pooled per-session KV slots and per-session random streams, batched
  perception and codec, capture-safety logic, per-session state isolation, and a tool controller (call ledger,
  timeouts, cancellation, grounding audit).

See [`docs/research-summary.md`](docs/research-summary.md) (findings and ideas),
[`docs/architecture.md`](docs/architecture.md), [`docs/benchmarks.md`](docs/benchmarks.md) and
[`docs/limitations.md`](docs/limitations.md).

## Current status and limits (read this first)

**Work in progress.** What is measured today (details in `docs/benchmarks.md`):

- **One session runs in real time** on an RTX 6000 Ada (BF16, RTF 0.71-0.74, including a tool call). One session on
  an L40S has not been benchmarked separately.
- **GB10 (128 GB unified memory) runs everything but is not real time**, even for one session (each 80 ms step
  takes longer than 80 ms); use it to test and develop, not to judge latency. Its runs are not part of our
  benchmarks.
- **Several concurrent sessions are not yet real time.** Four sessions on an L40S: round p95 173 ms without tool
  calls and 322 ms with two calls running (budget 160 ms per round). On an RTX 6000 Ada, four sessions without
  tools are close (throughput 25.9 session-chunks/s, lag ≤ 1.1 s) but still above the budget at p95.
- **Function calls work but are slow**: tool name, arguments and order are identical to a single session in our
  multi-session gate, but a call cycle takes ~13-15 s end to end with four sessions (fc_merge, the default, trades
  the per-call latency for far smaller tail-latency spikes for everyone).
- **Model limits** (NVIDIA's card and our tests): English only; spelled IDs and numbers are often misheard; no
  barge-in while a tool runs; about two minutes of trained context.
- **All numbers come from specific GPUs and synthetic English test speech**; expect different results elsewhere.

## Requirements

- Linux (x86_64 or aarch64), NVIDIA GPU with ≥ 46 GB memory for one VoiceChat instance (tested: RTX 6000 Ada 48 GB,
  L40S 46 GB, GB10 128 GB unified memory). The 11B model in BF16 needs ~42 GB at peak for one session.
- CUDA 13.x toolkit, Python 3.12, [uv](https://docs.astral.sh/uv/), git, `make`, ~70 GB of disk; `ffmpeg` is
  optional (silences an audio-library warning).
- The evaluation ASR (spoken-output check) installs out of the box on x86_64 only; on aarch64 the suites run and
  report that the audio check was skipped.
- Hugging Face access to the model weights (read the model license first, see `NOTICE.md`).

## Run it

```bash
git clone https://github.com/mynah-org/mynah-s2s.git && cd mynah-s2s
export MYNAH_HOME=$HOME/mynah-s2s-home   # venvs, weights, test audio and results go here (default ~/mynah-s2s)
# optional: reuse a Hugging Face cache that already holds the weights (setup links it and writes into it)
# export MYNAH_HF_CACHE=/path/to/hf-cache
make setup-all  # one time: core runtime + synthetic test-audio TTS + evaluation ASR (`make setup` = core only)
make doctor     # checks everything; tells you the exact fix for anything missing
make corpus     # renders the synthetic test audio (a few minutes)
make demo       # one call: "Can you look up order A B C four five six?" -> tool call -> spoken answer
make help       # all targets
```

## Web UI

The local HTML5 voice-agent UI lives in `webui/`. It runs with Python and `uv`,
without Node.js or a frontend build step. The default demo uses simulated
scenarios and tool calls, so it does not require a model or GPU:

```bash
make webui
```

Open <http://127.0.0.1:8787>. To start the V2 runtime and connect the UI in one
command, use:

```bash
make webui-gpu
```

For the GB10 realtime experiment, the lighter INT4 path is opt-in. The short
form is one switch (the equivalent environment variable is `MYNAH_INT4=1`):

```bash
make webui-gpu V2_SERVE_ARGS="--int4 --cudnn-benchmark"
```

This quantizes all measured standard Linear paths: Thinker projections and
heads, 218 perception Conformer linears, RNNT joint projections, the EarTTS
Gemma backbone, and compatible CAS/fusion/MoG linears. Dense codec decoder
`Conv1d(kernel=1)` projections reuse the same INT4 kernel. Recurrent/KV caches,
temporal/grouped convolution weights, embeddings/codebooks, RNNT decoder LSTM,
MoG `low_mat` by default, and the remaining codec paths stay BF16. The longer flags
(`--thinker-int4`, `--int4-heads`, `--int4-group`, `--int4-clip`) remain
available for experiments. `--max-backlog 120` can be added when testing a
slower-than-realtime build so input is not dropped while measuring it.

Dynamic `--int4` also enables the measured selected-MoG Q4 kernel and RVQ
gather/sum Q4 kernel. The streaming codec's small depthwise ConvNeXt filters
are available as an additional fused experiment with
`--exp codec_depthwise_q4`. Set `MYNAH_INT4_SPECIALIZED=0` alongside
`MYNAH_INT4=1` to keep only the standard INT4 paths. The second selected MoG
matrix (`low_mat`) is available for experiments with `MYNAH_INT4_MOG_LOW=1`;
the current GB10 measurement keeps it BF16 because its dequant launch adds
overhead. The fused RVQ argmin encoder and RNNT LSTM are also opt-in safety
experiments: use `MYNAH_INT4_RVQ_ENCODE=1` and `MYNAH_INT4_RNNT=1`
respectively; they can change quantized RVQ codes and are not part of the
default realtime path until a longer audio parity run validates them.
The RNNT Q4 path is batch-safe for `B=1` and `B>1`; its optional fully fused
gate/activation kernel can be profiled with `MYNAH_INT4_RNNT_FUSED=1`, but the
measured GB10 path keeps the smaller two-launch kernels because they are faster.

The legacy `--eartts-int4` flag remains available when testing only the TTS
backbone. It uses the native tinygemm path; the earlier custom Triton Q4 kernel
is retained only as a kernel experiment.

Compared with the external GGUF Q8 deployment, our coverage is now aligned on
the standard matrix paths plus dense codec projections; the remaining pieces
are temporal convolution/codebook and stateful/custom-kernel paths. The reference project emits
separate LLM, function-head, perception, and TTS/codec artifacts; its README
also says Q4 converts but the supported measurements are Q8. See the
[reference layout](https://github.com/mkotyushev/nemotron-voicechat-realtime-gguf#why-gguf).

To avoid quantizing at every server start, export one validated offline bundle once:

```bash
make int4-pack INT4_PACK=/path/to/voicechat-int4.safetensors
make webui-gpu V2_SERVE_ARGS="--int4-pack /path/to/voicechat-int4.safetensors"
```

The exporter writes the sibling `.json` manifest. It records the native
tinygemm packing, module shapes, group size, and explicitly lists the custom
audio runtime kernels that are not serialized in the bundle (selected MoG/RVQ
adapters and temporal codec paths). The server validates the whole manifest before replacing any module;
the default remains BF16 when no INT4 flag or environment variable is set.

The UI is in English and shows the User/Agent conversation, microphone and
speaker controls, model activity, tool calls, and an optional editable system
prompt. More details are in [`webui/README.md`](webui/README.md).

If the UI is opened from another machine and the browser does not offer a
microphone permission, use an SSH port-forward and open the local URL instead:

```bash
ssh -N -L 8787:127.0.0.1:8787 -L 8788:127.0.0.1:8788 <user>@<host>
```

Then visit <http://127.0.0.1:8787>. Browsers require `localhost` or HTTPS for
microphone capture; speakers normally do not show a separate permission prompt.

| goal | command |
|---|---|
| one conversation, any corpus item (q03 general knowledge, t13 correction mid-call, s05 small talk) | `make demo ITEM=t13` |
| the tool-call suite (lookups, not found, correction, follow-up, consecutive calls) + ASR + score | `make e2e-fc` |
| all 40 corpus items (reasoning, small talk, tools) + ASR + score | `make e2e-corpus` |
| four concurrent sessions, without tools and with two tool calls | `make multi` |
| your own scenario and persona prompt (example: a furniture-store assistant) | `make custom` |
| realtime WebSocket server, and a client streaming a WAV | `make serve` / `make client WAV=...` |
| unit tests | `make test-cpu`, `make test-gpu` |
| non-interactive server + client round trip | `make ws-check` |
| everything above except the full corpus (~1.5 h; installs missing TTS / ASR first) | `make reproduce` |

Three separable parts:

| part | installed by | what it is for |
|---|---|---|
| **core runtime** | `make setup` | the engine: NeMo Speech, CUDA kernels, VoiceChat weights; `run_stream.py`, `serve.py`, `rendezvous_pair.py` on your own audio |
| **synthetic test audio** | `make setup-tts`, then `make corpus` | the caller's voice for the test suites and custom scenarios (Qwen3-TTS 0.6B) |
| **evaluation ASR** | `make setup-asr` | transcribes what the agent actually said (Omnilingual ASR CTC 3B v2, own venv); suites skip the audio check without it. x86_64 only out of the box: on aarch64 its `fairseq2n` dependency has no PyPI wheels and must be built from source |

The scripts default to `--preset release`, the fastest configuration that passed every gate (CUDA graphs, batched
stages, fc_merge, fused EarTTS norm, int8 Thinker MLP weights, cached RVQ tables in the EarTTS sampler, deterministic
mode with fast attention kernels): reproducible run-to-run, not bit-identical to NVIDIA's step. `--preset release-exact`
is the same without the attention change, bit-identical to the deterministic reference line. Every run prints
`PRESET`, `OPTIMISATIONS active=... skipped=...` and `DETERMINISM {...}` and records them in its summary. If an optimisation cannot be enabled on your GPU,
demos and the server continue without it and print a WARNING plus the active list, while the multi-session gate
(`make multi`, a regression benchmark) stops with an error unless you pass `ARGS="--no-strict"`.
`ARGS="--preset reference"` runs NVIDIA's plain reference step instead. Each run writes to `$MYNAH_HOME/results/<suite>-<timestamp>/`; `combined.wav` has the
caller on the left channel and the agent on the right. How to write scenarios, what each suite checks and what to
expect: [`docs/running.md`](docs/running.md).

## Evaluation conventions

English synthetic test speech; tool back ends are deterministic mocks over a small SQLite table; spoken output is
checked with an open-source ASR (Omnilingual ASR CTC 3B v2), kept separate from the model's internal text and from
the tool ground truth. Results are labelled VERIFIED / PARTIAL / CLAIMED / SPECULATIVE.

## License

Original code: MIT (see `LICENSE`). A few portions are modified copies of Apache-2.0 code from NVIDIA NeMo Speech
and the NemotronH model code; they stay under Apache-2.0 (`LICENSES/Apache-2.0.txt`), marked with SPDX headers and
listed in `NOTICE.md`. Model weights and upstream components keep their own licenses (VoiceChat model card:
OpenMDW 1.1; NeMo Speech: Apache-2.0); this repository does not redistribute them.
