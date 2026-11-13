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

- **One session runs in real time** (RTX 6000 Ada, BF16, including a tool call).
- **Four concurrent sessions are not yet stably real time.** Without tool calls they are close on an RTX 6000 Ada
  (throughput 25.9 session-chunks/s, lag ≤ 1.1 s, but round p95 above the 160 ms budget); on an L40S calm rounds are
  already at ~160-170 ms.
- **Function calls work** (tool name, arguments and order identical to a single session in our multi-session gate)
  **but with high latency**: a call cycle takes ~11-15 s end to end in our 4-session workload, and fc_merge trades
  tail-latency bursts for ~3 s more per call.
- **All numbers are from specific hardware and synthetic English workloads** (RTX 6000 Ada 48 GB, L40S 46 GB);
  expect different results elsewhere. See `docs/benchmarks.md` and `docs/limitations.md`.

## Requirements

- Linux, NVIDIA GPU with ≥ 46 GB memory for one VoiceChat instance (tested: RTX 6000 Ada 48 GB, L40S 46 GB, and
  GB10 128 GB unified memory). The 11B model in BF16 needs ~42 GB at peak for one session.
- CUDA 13.x toolkit, Python 3.12, [uv](https://docs.astral.sh/uv/).
- The model weights from Hugging Face (not included here; check the model license first, see `NOTICE.md`).

## Quickstart (x86_64 + CUDA, e.g. Ada / L40S)

```bash
git clone <this repo> && cd mynah-s2s
export MYNAH_HOME=$HOME/mynah-s2s   # default; caches, venv, NeMo checkout, weights (clone the repo elsewhere)
bash v2/scripts/setup_cuda.sh                      # venv, NeMo Speech @097dfe9, mamba kernels, HF weights (~42 GB)
source v2/scripts/env.sh
cd v2 && PYTHONPATH=src $PY -m pytest -q tests/cpu  # CPU tests (no GPU, no weights)
```

`v2/scripts/run_stream.py` streams a WAV file through one session; `v2/scripts/rendezvous_pair.py` runs the
multi-session gate (solo vs concurrent sessions: tool calls, isolation, round latency, backlog). Both print their
options with `--help`.

## Evaluation conventions

English synthetic test speech; tool back ends are deterministic mocks over a small SQLite table; spoken output is
checked with an open-source ASR (Omnilingual ASR CTC 3B v2), kept separate from the model's internal text and from
the tool ground truth. Results are labelled VERIFIED / PARTIAL / CLAIMED / SPECULATIVE.

## License

Original code: MIT (see `LICENSE`). A few portions are modified copies of Apache-2.0 code from NVIDIA NeMo Speech
and the NemotronH model code; they stay under Apache-2.0 (`LICENSES/Apache-2.0.txt`), marked with SPDX headers and
listed in `NOTICE.md`. Model weights and upstream components keep their own licenses (VoiceChat model card:
OpenMDW 1.1; NeMo Speech: Apache-2.0); this repository does not redistribute them.
