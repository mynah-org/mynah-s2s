# V2 architecture

## The model: what is inside VoiceChat 11B

One checkpoint (11.1 B parameters, stored in FP32) made of distinct networks that run one after the other every
80 ms frame: user audio (16 kHz) → perception → Thinker → EarTTS → codec decoder (22.05 kHz). Parameter counts are
summed from the released weights; module classes are read in NVIDIA NeMo Speech.

| module | what it is | parameters |
|---|---|---|
| **Perception** | log-mel front end (128 bins, 10 ms hop) and a cache-aware streaming FastConformer (24 layers, width 1024, 8× subsampling, causal convolutions, limited left context): one embedding per 80 ms frame, projected to the LLM width | 0.61 B |
| **Thinker** | NVIDIA Nemotron Nano 9B v2, hybrid Mamba-2 / Transformer (56 blocks: Mamba-2, MLP and 4 attention blocks), consuming the audio embeddings and its own text | 7.71 B blocks + 0.59 B input embeddings |
| Thinker heads | **text head** (what the agent says, subword tokens), **function head** (tool-call JSON and tool responses, a separate channel), **RNN-T head** on the user side (turn-taking signals) | 0.59 B + 0.59 B + 0.01 B |
| **EarTTS** | a separate streaming TTS decoder (`RVQEARTTSModel` in NeMo `speechlm2`, not Magpie TTS): reads the Thinker's text subwords and its own previous audio codes; Gemma-3-style transformer (28 layers, width 1152) plus a mixture-of-Gaussians head (1024 components, low rank 64); each frame is decoded into 31 residual-VQ codes in 8 masked refinement passes with classifier-free guidance; one fixed speaker ("Aria") | 0.59 B backbone + 0.16 B head + 0.04 B embeddings |
| **Codec** | RVQ-VAE audio codec, 31 codebooks × 1024 codes, 1764 samples per frame (80 ms at 22.05 kHz); the decoder turns EarTTS codes into the waveform; frozen during VoiceChat training | 0.18 B + 0.02 B quantizer |

Consequences used throughout this repository:
- the agent's words exist as text *before* they are spoken (the Thinker writes them, EarTTS voices them): internal
  text and spoken audio are separate evidences and can disagree, so the test suites check both;
- tool calls live on their own channel and never reach EarTTS; the agent keeps talking ("just a moment") while a call
  runs;
- the Thinker holds most of the weights (its GPU time is bound by reading them), while EarTTS, with 8 refinement passes
  per frame, issues most of the small GPU kernels; on an L40S both take a comparable share of a round.

## One session, one 160 ms chunk (two 80 ms frames)

| stage | what runs | V2 execution |
|---|---|---|
| perception | NeMo FastConformer cache-aware encoder | batched across sessions, CUDA graphs per row count |
| input fusion | text / ASR / function-token embeddings + audio embedding | per session (small) |
| Thinker | NemotronH 9B (Mamba2 + attention + MLP), text / ASR / function heads, one decode step per frame | ONE batched step for all sessions at the rendezvous; static recurrent/attention state; CUDA graphs per (rows, bucket); int8 weight-only MLP GEMV kernels (Triton) |
| EarTTS | Gemma3 backbone + MoG head, one step per frame, RVQ codes | full-step CUDA graph per KV bucket; pooled per-session KV slots; ONE batched step for the sessions parked at it; per-session counter-based random streams |
| codec | streaming RVQ codec decode | ONE stacked decode per round |

The step logic in between (sampling, turn-taking, the function-call state machine) is NVIDIA's reference code,
unchanged.

## Multi-session rendezvous

Each session runs its unchanged reference step in its own worker thread, but only one session executes wrapper code
at a time (a baton held by a coordinator). Sessions **park** at the Thinker, EarTTS, perception and codec calls; the
coordinator serves all parked requests of a round as one batched call, then resumes them. Guarantees (CPU-tested):
the Thinker runs only on the coordinator; explicit Thinker ownership per session (a session's function-call thread
owns its Thinker during a call); no waiting for absent sessions; per-session state, random streams and tool
bindings are isolated.

## Function calls

NVIDIA's reference runs a function-call cycle in a background thread: call tokens on the function channel, tool
execution, forced token-by-token injection of the tool response, then the model resumes speaking. Each async step
also runs one EarTTS step and one codec decode to keep the speech caches aligned. In V2:

- the function-call thread's Thinker steps ride the round's batched Thinker step as extra rows (**fc_merge**,
  default on; `--fc-merge off` restores the batch-size-1 service). This removed the tail-latency bursts that
  occurred when a whole call cycle was served inside a few rounds (see `benchmarks.md`); the trade-off is a longer
  call cycle (~3 s on L40S in our workload);
- a tool controller keeps a call ledger, call/session epochs, timeouts, cancellation and a grounding audit.

## Correctness checks

- Parity with NVIDIA's own driver: identical text and function tokens on three inputs (196-1102 frames) and a
  seeded waveform within one PCM16 step.
- Multi-session gate (`v2/scripts/rendezvous_pair.py`): each session solo, then all concurrently; tool name,
  arguments and order identical to solo, no cross-session leak, no errors, finite audio; spoken output checked by ASR.
- Batched EarTTS with per-session random streams: row results independent of row order and neighbours; with
  batching off, concurrent sessions are bit-identical to solo.
