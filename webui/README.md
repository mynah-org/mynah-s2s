# Mynah S2S web UI

Minimal HTML5 client and Python bridge for the V2 realtime server.

## Local demo

From the repository root:

```bash
uv run --project webui --no-sync python webui/server.py
```

Open <http://127.0.0.1:8787>. The demo mode uses the three scenario presets,
simulates a tool call, and uses the browser's speech synthesis for the spoken
answer. It does not require a checkpoint or a GPU. Click `Connect`, then
`Run guided example`; `Reset` clears the transcript and event stream.

## GPU-backed session

Start the existing V2 server first, then point the bridge at its WebSocket. The
GPU extra installs the optional upstream client library:

```bash
python v2/scripts/serve.py --trace-dir /tmp/mynah-traces --host 127.0.0.1
uv run --project webui --extra gpu python webui/server.py --v2-ws ws://127.0.0.1:8765/v1/session
```

To try the experimental lower-latency V2 path, pass the single opt-in switch
through the Make target (or set `MYNAH_INT4=1`):

```bash
make webui-gpu V2_SERVE_ARGS="--int4 --cudnn-benchmark"
```

The INT4 mode quantizes Thinker projections, text/function heads, and the
graph-safe EarTTS Gemma backbone with native tinygemm. It keeps recurrent/KV
caches, perception/RNNT, CAS/subword, MoG, RVQ, and codec components unchanged.

The legacy TTS-only switch is still available for isolated tests:

```bash
make webui-gpu V2_SERVE_ARGS="--eartts-int4 --perf-profile optimized-graph"
```

It quantizes only the Gemma backbone linears. CAS/subword embeddings, MoG
projections, RVQ, codec, and perception remain unchanged. The combined
`--int4` path has passed a real DGX WebSocket turn with zero dropped audio.

For repeatable startup, export the native packed weights once and pass the pack
instead:

```bash
make int4-pack INT4_PACK=/path/to/voicechat-thinker-int4.safetensors
make webui-gpu V2_SERVE_ARGS="--thinker-int4-pack /path/to/voicechat-thinker-int4.safetensors"
```

This writes `.safetensors` plus a sibling `.json` manifest. No INT4 option means
the normal BF16 server path.

To open it from another device on the LAN, use `--host 0.0.0.0` and pass the
machine address so the bridge prints a copy/pasteable URL.

```bash
uv run --project webui --no-sync python webui/server.py --host 0.0.0.0 --advertise-host 192.168.1.20
```

The top bar shows bridge and voice-agent state separately. The button becomes
`Disconnect` during a session and `Reconnect` after a dropped socket. The GPU
status changes to `V2 GPU connected` only after the bridge has opened a session
with the V2 server.

Select `GPU · v2 server`, click `Connect`, then `Start microphone`. The bridge
forwards browser PCM16 frames and relays V2's PCM16 22.05 kHz audio plus its
JSON event stream. The UI surfaces `thinker.step`, `text.generated`, ASR and
all `tool.*` events without exposing hidden chain-of-thought text.

### Microphone from another machine

Browsers allow microphone capture only from `localhost`, HTTPS, or another
secure origin. If the UI is running on a remote machine, forward its ports to
your computer and open the local URL instead:

```bash
ssh -N -L 8787:127.0.0.1:8787 -L 8788:127.0.0.1:8788 <user>@<host>
```

Then open <http://127.0.0.1:8787>. The browser will be able to request the
microphone after you click `Start microphone`. Speaker output normally does not
have a separate permission prompt; audio playback starts after that user click.

The bridge listens on two small localhost ports: HTTP `8787`, WebSocket `8788`.
It deliberately has no persistence or auth and is intended as a local lab
prototype, not an internet-facing deployment.
