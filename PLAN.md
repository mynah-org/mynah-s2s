# Roadmap (public)

- [x] Survey of open-weight full-duplex S2S models and tool-calling mechanisms.
- [x] V2 runtime on NVIDIA's reference step: parity, tool controller, single-session real time (RTX 6000 Ada).
- [x] Multi-session engine: batched Thinker, perception, codec, EarTTS; capture safety; per-session isolation.
- [x] Per-session EarTTS random streams (audio independent of batch composition).
- [x] fc_merge: function-call Thinker steps in the batched step (tail latency -50 % p95 on L40S), default on.
- [ ] Recover the function-call latency added by fc_merge without reintroducing bursts.
- [ ] Kernel-level hotspots from the steady-state profile: EarTTS small-kernel overhead (norms, head, fills),
      Thinker weight bandwidth (roofline), codec.
- [ ] End-to-end round latency breakdown (kernel time vs host gaps and synchronisation).
- [ ] Harness: drain phase after the scripted input (response completion metrics).
- [ ] Multilingual research directions (Italian, French): which components to adapt while keeping full-duplex
      behaviour and tool calls.
- [ ] Later: a small local web demo (live conversation, scenario presets, tool calls, latency panel).
