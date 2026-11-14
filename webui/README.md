# Mynah S2S web UI

Minimal HTML5 client and Python bridge for the V2 realtime server.

## Local demo

From the repository root:

```bash
uv run --project webui --no-sync python webui/server.py
```

Open <http://127.0.0.1:8787>. The demo mode uses the three scenario presets,
simulates a tool call, and uses the browser's speech synthesis for the spoken
answer. It does not require a checkpoint or a GPU.

## GPU-backed session

Start the existing V2 server first, then point the bridge at its WebSocket. The
GPU extra installs the optional upstream client library:

```bash
python v2/scripts/serve.py --trace-dir /tmp/mynah-traces --host 127.0.0.1
uv run --project webui --extra gpu python webui/server.py --v2-ws ws://127.0.0.1:8765/v1/session
```

To open it from another device on the LAN, use `--host 0.0.0.0` and pass the
machine address so the bridge prints a copy/pasteable URL.

```bash
uv run --project webui --no-sync python webui/server.py --host 0.0.0.0 --advertise-host 192.168.1.20
```

The top bar shows bridge and voice-agent state separately. The button becomes
`Disconnect` during a session and `Reconnect` after a dropped socket.

Select `GPU · v2 server`, click `Connetti`, then `Avvia microfono`. The bridge
forwards browser PCM16 frames and relays V2's PCM16 22.05 kHz audio plus its
JSON event stream. The UI surfaces `thinker.step`, `text.generated`, ASR and
all `tool.*` events without exposing hidden chain-of-thought text.

The bridge listens on two small localhost ports: HTTP `8787`, WebSocket `8788`.
It deliberately has no persistence or auth and is intended as a local lab
prototype, not an internet-facing deployment.
