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

The WebUI GPU target starts V2 with the measured lower-latency INT4 path by
default (and enables cuDNN benchmarking):

```bash
make webui-gpu
```

To override that default, pass an explicit `V2_SERVE_ARGS` value, for example
`make webui-gpu V2_SERVE_ARGS="--cudnn-benchmark"` for BF16. Direct
`make serve` remains BF16 unless `--int4` or `MYNAH_INT4=1` is supplied.

For an explicit BF16 EarTTS measurement, use
`V2_SERVE_ARGS="--perf-profile optimized-bf16"`. The profile converts the
backbone to BF16 and enables the fused norms and cached RVQ constants before
graph capture; it is opt-in because FP32 EarTTS remains the quality reference.

The INT4 mode quantizes all measured standard Linear paths: Thinker/head,
perception Conformer, RNNT joint, EarTTS Gemma, CAS/fusion, and compatible MoG
linears. Dense codec decoder `Conv1d(kernel=1)` projections use the same
INT4 kernel; temporal/grouped convolution, codebooks, recurrent/KV caches,
RNNT decoder LSTM, MoG `low_mat` by default, and remaining codec components stay BF16.
Dynamic mode also enables the selected-MoG and RVQ gather/sum Q4 kernels;
`MYNAH_INT4_SPECIALIZED=0` disables those two while keeping standard INT4.
The codec depthwise ConvNeXt path is available explicitly with
`--exp codec_depthwise_q4`. The fused RVQ argmin encoder and RNNT prediction
LSTM are safety-gated experiments via `MYNAH_INT4_RVQ_ENCODE=1` and
`MYNAH_INT4_RNNT=1`; they are not enabled by default because Q4 code choices
and end-to-end latency still need longer target-GPU validation.
The additional selected `low_mat` MoG kernel is available with
`MYNAH_INT4_MOG_LOW=1`, but is not enabled by default because its current GB10
measurement is slower than the BF16 path.

The legacy TTS-only switch is still available for isolated tests:

```bash
make webui-gpu V2_SERVE_ARGS="--eartts-int4 --perf-profile optimized-graph"
```

It quantizes only the Gemma backbone linears. CAS/subword embeddings, MoG
direct projections, RVQ, codec, and perception remain unchanged. The combined
`--int4` path has passed a real DGX WebSocket turn with zero dropped audio.

For an explicit Q8 coverage run, use `--int8` (or `MYNAH_INT8=1`). This enables
Q8 counterparts for the standard matrix paths plus selected MoG, RVQ, codec
depthwise, and RNNT prediction paths. It is currently an experimental
parity/memory mode, not the WebUI default: the full all-Q8 corpus smoke is
semantically correct but measured `RTF 10.38` on GB10.

Export the standard Q8 scopes once with `make w8-pack W8_PACK=/path/to/voicechat-w8.safetensors`,
then start with `V2_SERVE_ARGS="--int8-pack /path/to/voicechat-w8.safetensors"`.
The sibling JSON manifest is validated before replacement; selected MoG/RVQ,
depthwise codec, and RNNT runtime adapters remain opt-in startup work.

For repeatable startup, export the native packed weights once and pass the bundle
instead:

```bash
make int4-pack INT4_PACK=/path/to/voicechat-int4.safetensors
make webui-gpu V2_SERVE_ARGS="--int4-pack /path/to/voicechat-int4.safetensors"
```

This writes `.safetensors` plus a sibling `.json` manifest. The bundle includes
all measured standard Linear scopes and pointwise codec/Conformer adapters, and
records the remaining runtime-only BF16/custom scopes. No INT4 option means
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
