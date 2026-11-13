# V2 architecture

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
