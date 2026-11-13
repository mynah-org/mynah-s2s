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
