# Research summary: open-weight full-duplex S2S agents (cutoff 13 November 2026)

A short, public summary of what we explored, what we found and what is still open. Evidence labels:
**VERIFIED** (measured by us or read in code), **PARTIAL** (small sample or open gap), **CLAIMED** (stated by
authors, not reproduced), **SPECULATIVE** (hypothesis).

## Question

Can an open-weight full-duplex S2S model act as a voice agent that listens while it speaks, calls a tool, and then
*says the tool's result correctly*? Which model is the best base, and what runtime does it need?

## Models we looked at

| model | duplex | tools | our depth |
|---|---|---|---|
| Moshi / Moshiko / Moshika (Kyutai) | full-duplex, text "inner monologue" | none | run |
| PersonaPlex (NVIDIA, Moshi-derived) | full-duplex, role/voice conditioning | none official | run |
| MoshiRAG (Kyutai) | full-duplex, learned retrieval token | free-text reference, no schema | run |
| NVIDIA NemotronLabs VoiceChat 11B | full-duplex, 80 ms frames | **native function channel** | run (three runtimes) |
| KAME, K-Moshi, J-Moshi, MultiTalk | full-duplex designs / adaptations | none | paper / code / card |
| MiniCPM-o 4.5, Qwen3-Omni | omni models (time-division duplex / turn-based) | injected text / examples | run |
| caramel-1 | audio-only full-duplex, no text stream | none | run (small probe) |

## Findings

- **Valid tool-call JSON is not enough** [VERIFIED]. The spoken answer must use the actual tool result. On a
  re-implemented runtime, VoiceChat answered from its priors in 5/5 sessions before the tool response was consumed and
  grounded correctly in 5/5 once it was: the apparent "hallucinations" were a serving-timing effect (a backlog of
  ~140 frames between the end of the call and the injected result), not a model limit [PARTIAL: n = 5].
- **Runtime fidelity matters.** Keeping NVIDIA's reference step and function-call protocol unchanged, and replacing
  only how it executes, gave token-level parity with NVIDIA's own driver [VERIFIED].
- **Batching only the language model does not scale end to end** [VERIFIED]: perception, speech decoder (EarTTS) and
  codec must be batched across sessions too.
- **Per-session random streams are required under batching** [VERIFIED]: with one shared generator a session's audio
  depended on which other sessions were in the batch (a reproducible garbled greeting); per-session counter-based
  streams fixed it (batching off: bit-identical to a solo run).
- **Function-call bursts drive tail latency** [VERIFIED]. The reference runs one language-model step, one speech step
  and one codec decode per async call step; serving them greedily compressed a whole call cycle into a few slow
  rounds. Capping steps per round only spread the work (call latency 2-3× higher, other sessions' backlog grew);
  running the call steps as extra rows of the batched step (fc_merge) halved round p95 on an L40S at the cost of
  ~3 s longer calls.
- **BF16 does not speed up the speech decoder on L40S** [VERIFIED]: its FP32 GEMMs already run on TF32 tensor cores
  and are ~30 % of its GPU time; the rest is many small elementwise/norm/sampling kernels. BF16 saves memory, not time.
- **Language: understanding ≠ speaking** [VERIFIED, small n]: Moshiko answered Italian and French questions in
  English (0/5 in-language per language); a turn-based omni model answered in-language 5/5.
- **Audio-only training learns timing, not content** [PARTIAL]: caramel-1 yields quickly when interrupted but
  answered 0/18 questions.

## Ideas and open questions [SPECULATIVE]

- Recover function-call latency without bursts: batch the call's speech-decoder and codec steps with the round.
- Fewer, larger kernels in the speech decoder (fused norms, head and sampling).
- Multilingual VoiceChat: which components to adapt (language backbone, speech decoder, interaction behaviour), with
  which data, while keeping full-duplex turn-taking and reliable tool calls. Not solved; not validated.
