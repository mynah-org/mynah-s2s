"""Realtime WebSocket test client: streams a WAV at real-time pace (80 ms packets), records agent audio + events."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf


def load_16k_int16(path: str) -> np.ndarray:
    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = wav.mean(1)
    if sr != 16000:
        import torch
        import torchaudio.functional as AF
        wav = AF.resample(torch.from_numpy(wav).unsqueeze(0), sr, 16000)[0].numpy()
    return (np.clip(wav, -1, 1) * 32767).astype("<i2")


async def run(url: str, wav: str, out: Path, tail_s: float, mode: str | None, idle_s: float,
              close_timeout: float = 300.0) -> dict:
    import websockets
    pcm = load_16k_int16(wav)
    pcm = np.concatenate([pcm, np.zeros(int(tail_s * 16000), dtype="<i2")])
    out.mkdir(parents=True, exist_ok=True)
    agent = bytearray()
    events = []
    t0 = time.monotonic()
    first_audio = None
    async with websockets.connect(url, max_size=2**24) as ws:
        start = {"type": "session.start"}
        if mode:
            start["mode"] = mode
        await ws.send(json.dumps(start))
        created = json.loads(await ws.recv())
        events.append(created)

        async def reader():
            nonlocal first_audio
            async for m in ws:
                if isinstance(m, bytes):
                    if first_audio is None:
                        first_audio = time.monotonic() - t0
                    agent.extend(m)
                else:
                    d = json.loads(m)
                    d["t_client"] = round(time.monotonic() - t0, 3)
                    events.append(d)
                    if d.get("type") == "session.closed":
                        return

        rt = asyncio.create_task(reader())
        step = 1280  # 80 ms
        for k in range(0, len(pcm), step):
            target = t0 + (k / 16000)
            await asyncio.sleep(max(0.0, target - time.monotonic()))
            await ws.send(pcm[k:k + step].tobytes())
        await asyncio.sleep(idle_s)
        await ws.send(json.dumps({"type": "session.end"}))
        await asyncio.wait_for(rt, timeout=close_timeout)
    a = np.frombuffer(bytes(agent), dtype="<i2").astype(np.float32) / 32767
    sf.write(out / "agent.wav", a, 22050)
    (out / "events.jsonl").write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in events) + "\n")
    res = {"wav": wav, "agent_s": round(len(a) / 22050, 2), "first_audio_s": first_audio,
           "n_events": len(events), "event_types": sorted({e.get("ev", e.get("type")) for e in events}),
           "text": "".join(e.get("text", "") for e in events if e.get("ev") == "text.generated"),
           "tools": [e for e in events if e.get("ev", "").startswith("tool.")],
           "closed": next((e for e in events if e.get("type") == "session.closed"), None),
           "dropped_notices": sum(1 for e in events if e.get("type") == "audio.dropped"),
           "wall_s": round(time.monotonic() - t0, 2)}
    (out / "client_result.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://127.0.0.1:8765/v1/session")
    ap.add_argument("--wav", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tail", type=float, default=4.0)
    ap.add_argument("--idle", type=float, default=2.0)
    ap.add_argument("--mode", default=None)
    ap.add_argument("--close-timeout", type=float, default=300.0,
                    help="max wait for session.closed after session.end (server drains its backlog first)")
    a = ap.parse_args()
    print(json.dumps(asyncio.run(run(a.url, a.wav, Path(a.out), a.tail, a.mode, a.idle, a.close_timeout)),
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
