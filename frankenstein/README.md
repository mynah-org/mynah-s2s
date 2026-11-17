# Frankenstein S2S

A separate, experimental multilingual speech-to-speech engine. It is built from existing parts, with no training.

- **Timing:** the VoiceChat duplex loop listens to the original audio and decides when to speak, pause, stop or yield.
- **Content:** the same Thinker weights, in text mode, write the reply from a multilingual streaming-ASR transcript
  in the session language. The reply is teacher-forced into the duplex loop's agent channel.
- **Voice:** a talker adapter turns that token stream into commands for the Sophon Flash (Qwen3-TTS 12 Hz) talker,
  with the session language.

The design, source audit, controlled modes and GPU gates are in [`DESIGN.md`](DESIGN.md).

## Status

Design and pure-Python control layer only. **Nothing has been run on a GPU; no model is loaded here.**

| module | what | state |
|---|---|---|
| `frankenstein_s2s/adapter.py` | talker adapter state machine (word commits, burst commit, EOS drain / cancel, barge-in, epochs, tool wait, segments) | implemented, tested |
| `frankenstein_s2s/forcing.py` | per-frame agent-channel forcing (M1-force) | implemented, tested |
| `frankenstein_s2s/usertext.py` | ASR pieces → Thinker ids, causal; EOU / BOU; M1-ctx insert spans; mode contracts | implemented, tested |
| `frankenstein_s2s/driver.py` | `TalkerDriver` protocol, fake driver, `PlaybackGate`; `FlashTalkerDriver` | fake + gate tested; Flash driver is an interface stub |
| `frankenstein_s2s/tracing.py`, `trace_view.py` | JSONL events (V2 format, both clocks) and a per-turn timeline viewer | implemented, tested |
| `frankenstein_s2s/lang.py`, `replay.py` | session language mapping; `vc_replay_v1` reader | implemented, tested |
| duplex runner, text-mode brain, ASR, Flash talker | GPU parts | design only (DESIGN.md §4, §7, §10) |

The package has no third-party runtime dependency and never imports torch. The V2 engine (`v2/`) is not modified;
the planned runner imports it read-only.

## Run the tests

From the repository root (Python ≥ 3.10, pytest):

```bash
python -m pytest -q frankenstein/tests
```

To view a trace:

```bash
cd frankenstein && python -m frankenstein_s2s.trace_view path/to/trace.jsonl [--turn N]
```
