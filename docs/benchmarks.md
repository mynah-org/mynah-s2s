# Benchmarks (experimental cutoff 13 November 2026)

Definitions: **round** = one 160 ms chunk for every active session (wall time of the coordinator round);
**miss** = round > 160 ms; **lag** = how late a session's chunk starts vs its real-time arrival; **throughput** =
session-chunks per busy second. Real time for N sessions requires sustained rounds ≤ 160 ms (lag bounded).
All runs: English synthetic speech, Thinker BF16 with int8 weight-only MLP, EarTTS FP32 unless stated.

## One session, RTX 6000 Ada 48 GB (BF16, no quantization)

Real time including a tool call: step p50 113 ms, RTF 0.71-0.74 (optimized graph profile, token-identical to the
default configuration on the tested items).

## Four sessions, RTX 6000 Ada 48 GB (mixed workload: 2 sessions with one tool call each + 2 plain)

| configuration (cumulative) | round p50 / p95 ms | throughput |
|---|---|---|
| Thinker batched, other stages per session | 248 / 645 | 9.8 |
| + batched perception (CUDA graphs) | 219 / 552 | 10.5 |
| + batched codec | 208 / 540 | 10.8 |
| + cheaper codec decode on the tool-call path | 201 / 552 | 11.1 |
| graphs captured before sessions start | 209 / 471 | 14.6 (measurement fix: excludes capture time) |
| + batched EarTTS | 141 / 502 | 17.8 |
| 4 sessions, no tools | 131 / 203 | 25.9 (lag ≤ 1.1 s) |

Single runs per row (run-to-run noise ≈ 1 % on p50, ±20 ms on p95). **Neither workload is stably real time:** the
no-tools mix has p95 above the 160 ms budget, and in the mixed workload tool-call cycles produced bursts of
0.7-1.1 s rounds and a per-session lag of 7-10 s (see below).

## Function-call bursts and fc_merge, NVIDIA L40S 46 GB (same mixed workload, batched EarTTS FP32)

Two runs per configuration; the third fc_merge value is the release code (fc_merge default, fused RMSNorm).

| | baseline (call steps at batch size 1) | fc_merge |
|---|---|---|
| round p95 ms | 683 / 621 | 330 / 323 / 322 |
| round max ms | 1880 / 1333 | 500 / 518 / 503 |
| call cycle (2 sessions) | 11.1-12.2 s | 13.1-14.6 s |
| tool calls vs solo | identical | identical |

Reproducible: p95 -50 %, max -61…-73 %. GPU busy time, throughput and max lag stayed within run-to-run noise.
The L40S is slower than the RTX 6000 Ada here: calm rounds ~160-170 ms, so four sessions are not real time on it.

Serving one or two extra call steps per round on top of fc_merge did not shorten the call cycle (14.4-15.2 s vs
14.7-14.8 s for plain fc_merge in the same harness) and raised p95 to 384 ms with two steps, so it is not included.

## EarTTS precision on L40S

BF16 EarTTS reduces memory (-1.1 GB weights only, -3.9 GB KV only, 4 sessions) but not time: the FP32 GEMMs already
run on TF32 tensor cores and are ~30 % of EarTTS kernel time; casts around norms and the head cancel the saving, and
mixed weight/KV dtypes add a full KV-window cast per step. FP32 remains the default.

## Fused RMSNorm in the speech decoder (default in the graph profiles)

The EarTTS backbone runs 28 layers × 6 RMSNorms per step, ~7 small kernels each. Running each as one fused
`rms_norm` with the same FP32 formula reduced EarTTS GPU time on L40S (4 sessions, no tools, CUDA graphs, 20 steady
rounds, single runs) by **2.0 ms per round in FP32** (35.8 → 33.7 ms) and **2.75 ms with a BF16 backbone**; spoken
output checked with ASR was unchanged. This is GPU kernel time of one component, not a demonstrated end-to-end p95
improvement (round p50/p95 vary by ±10-25 ms between runs). Fallback: `V2_FUSED_RMSNORM=0`.

Release-code check (L40S, 4 sessions, end-to-end runs, single run each vs the previous fc_merge runs): EarTTS GPU
time -4 to -6 % (no tools 21.5 → 20.3 s per run; mixed 21.1 → 20.2 s), tool calls identical to solo, no errors.
No-tools rounds p50/p95/max 164/173/240 ms vs 171/178/249 ms before, within the range of run-to-run noise.

## Release preset: deterministic mode without NaN fills, fast attention, cached RVQ tables (L40S)

`release` (the default since this update) adds three changes to the line above; `release-exact` keeps the
bit-exact attention. GPU kernel time per 4-session round (profiled rounds 100-119), rounds in end-to-end runs:

| configuration | GPU kernels ms/round | no tools p50 / p95 / max ms (misses of 188) | mixed p50 / p95 / max ms (misses of 227) | outputs |
|---|---|---|---|---|
| previous release line | 113.8 | 164 / 173 / 240 | 174 / 322 / 503 (161) | reference |
| `release-exact` (+ RVQ cache, no NaN fill) | 103.3 | 156 / 160 / 232 (12); 131 / 151 / 224 (3) | 154 / 316 / 484 (92); 150 / 307 / 448 (66) | bit-identical to the reference line |
| `release` (+ flash / memory-efficient attention) | 96.6 | 148 / 153 / 226 (2); 147 / 155 / 246 (5) | 150 / 314 / 442 (52); 133 / 299 / 455 (53); 137 / 295 / 436 (51) | reproducible run-to-run, not bit-identical |

- The NaN fill: in deterministic mode PyTorch fills every newly allocated tensor (~3000 fills per round here); the
  outputs never read uninitialized memory, so turning the fill off is bit-exact.
- The RVQ cache computes the codebook padding and per-code norms once instead of at each of the 8 sampler
  iterations of every EarTTS step (bit-exact).
- Fast attention: the deterministic mode had also disabled flash / memory-efficient attention; inference is
  forward-only, so they are allowed again. Same outputs on every rerun on the same GPU and software, small
  numerical differences vs `release-exact`.
- Quality checks (`release` vs the deterministic reference line, one session, 40-item corpus): reasoning 12/12,
  small talk 10/10, tool-call arguments unchanged, model text identical on 35/40 items (the 5 differences reviewed:
  equivalent wording, one answer completed where the reference stopped early); spoken output checked with ASR, no
  garbled audio. 4 sessions: tool calls identical to solo, no cross-session leak, memory unchanged.
- Rejected: running without the deterministic mode at all (GPU 93-96 ms/round, but not reproducible and a garbled
  greeting in 2/2 runs of one session); a fused norm in the sampler head (not bit-exact, no gain over the RVQ cache).
- Run-to-run noise without tools is large (the same `release-exact` configuration measured p50 156 and 131 ms);
  GPU kernel time and the mixed-workload misses are the stable signals.

## Host side of a 4-session round: RNN-T overlap and cheaper session switches (L40S)

GPU kernels take ~96 ms of a ~137 ms round; most of the rest was the host running NVIDIA's per-session code after
each batched Thinker step, one session at a time, with the GPU idle (~2.5 ms per session per step). Two changes,
both bit-exact (same outputs; codes identical without tools):

- **RNN-T overlap** (default in the release presets, `--rnnt-overlap off` to disable): the user-side RNN-T decode
  step of each session depends only on the perception frame and its own state, so it now runs on a side CUDA stream
  while the batched Thinker runs, instead of after it. Host tail per session after the Thinker 3.1 → 1.7 ms.
- **Session switch**: the per-session state swap (several per round) resolves its fields once (−0.07 ms per switch).

| 4 sessions, `release` | round p50 / p95 / max ms (misses) |
|---|---|
| no tools, before | 137 / 149 / 218 (4 of 188) |
| no tools, with both | 121 / 139 / 195 (3); 139 / 144 / 188 (2) |
| mixed, before | 137 / 295 / 436 (51 of 227) |
| mixed, with both (3 runs) | 129 / 289 / 430 (64); 128 / 297 / 424 (55); 128 / 299 / 418 (52) |

The host part depends on the CPU of the machine: on the same GPU and code, the gaps between GPU stages varied from
run to run, which is most of the no-tools noise. Tool calls identical in every run, no cross-session leak, spoken
output checked with ASR unchanged.

