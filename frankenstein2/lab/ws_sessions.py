"""Concurrent WebSocket client sessions against a Frankenstein v2 server (directly or through the web UI bridge).

Each client streams a 16 kHz track in real time (80 ms PCM16 chunks, then tail silence), with its own language, and
records: the events, the agent audio received (22.05 kHz) with arrival times, its own send times. From those it
measures, per user utterance of the track's timeline: end of speech -> first agent audio received (client clock),
the tools called (from tool.completed events), barge-ins / flushes, and writes the stereo listening file
(L user, R agent as received, on the client clock).

  ClientSpec(name, url, language, wav16, timeline, via_bridge)
  run_clients([...], out_dir) -> {name: result}
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class ClientSpec:
    name: str
    url: str
    language: str
    wav16: object
    timeline: list = field(default_factory=list)
    via_bridge: bool = False
    voice: str = ""
    tail_s: float = 12.0
    start_delay_s: float = 0.0


async def run_client(c: ClientSpec):  # pragma: no cover - needs a server
    import websockets
    events, ctrl, audio = [], [], []
    await asyncio.sleep(c.start_delay_s)
    t0 = None
    start = {"type": "session.start", "language": c.language, "voice": c.voice or None}
    if c.via_bridge:
        start.update(transport="gpu", scenario="ikea-orders")
    async with websockets.connect(c.url, max_size=2 ** 24) as ws:
        await ws.send(json.dumps(start))

        async def reader():
            async for m in ws:
                now = time.monotonic()
                if isinstance(m, bytes):
                    audio.append((now, m))
                else:
                    d = json.loads(m)
                    d["_t"] = now
                    (events if d.get("type") == "event" else ctrl).append(d)
                    if d.get("type") == "session.closed":
                        return
        rt = asyncio.create_task(reader())
        while not any(x.get("type") == "session.created" for x in ctrl):
            await asyncio.sleep(0.02)
            if rt.done():
                raise RuntimeError(f"{c.name}: {ctrl}")
        x = np.asarray(c.wav16, np.float32)
        x = np.concatenate([x, np.zeros(int(c.tail_s * 16000), np.float32)])
        pcm = (np.clip(x, -1, 1) * 32767).astype("<i2")
        t0 = time.monotonic()
        for i in range(0, len(pcm) // 1280):
            w = t0 + i * 0.08 - time.monotonic()
            if w > 0:
                await asyncio.sleep(w)
            await ws.send(pcm[i * 1280:(i + 1) * 1280].tobytes())
        await ws.send(json.dumps({"type": "session.end"}))
        await asyncio.wait_for(rt, 120)
    return {"t0": t0, "events": events, "ctrl": ctrl, "audio": audio}


def analyse(c: ClientSpec, r: dict, out: Path) -> dict:  # pragma: no cover - needs a run
    import soundfile as sf
    from scipy.signal import resample_poly
    t0 = r["t0"]
    out.mkdir(parents=True, exist_ok=True)
    sr = 22050
    n_tot = int((max([t for t, _ in r["audio"]] + [t0 + len(c.wav16) / 16000]) - t0 + 2) * sr)
    agent = np.zeros(n_tot, np.float32)
    first_heard = []
    cursor = 0                                        # the client plays chunks back to back from their arrival
    for t, b in r["audio"]:
        y = np.frombuffer(b, "<i2").astype(np.float32) / 32767
        i = max(cursor, int((t - t0) * sr))
        if i + len(y) > len(agent):
            agent = np.concatenate([agent, np.zeros(i + len(y) - len(agent), np.float32)])
        agent[i:i + len(y)] = y
        cursor = i + len(y)
        if np.abs(y).max() > 0.01:
            first_heard.append((i + int(np.argmax(np.abs(y) > 0.01))) / sr)
    rows = []
    for j, u in enumerate(c.timeline):
        end_s = (u["end_k"] + 1) * 0.08
        nxt = c.timeline[j + 1]["start_k"] * 0.08 if j + 1 < len(c.timeline) else 1e9
        fa = [t for t in first_heard if end_s < t < nxt]
        rows.append({"label": u["label"], "first_audio_s": round(min(fa) - end_s, 3) if fa else None})
    ev = r["events"]
    tools = [{"name": e.get("name"), "args": e.get("args")} for e in ev if e.get("ev") == "tool.completed"]
    turns = [e.get("text") for e in ev if e.get("ev") == "asr.turn"]
    said = [e.get("speech") for e in ev if e.get("ev") == "tts.segment" and e.get("lane") == "agent"]
    barges = [{k: e.get(k) for k in ("model_step", "by", "onset_s", "onset_frame")} for e in ev
              if e.get("ev") == "ears.event" and e.get("kind") == "BARGE"]
    flushes = [{k: e.get(k) for k in ("model_step", "reason", "dropped_samples")} for e in ev
               if e.get("ev") == "audio.flush"]
    eps = [{k: e.get(k) for k in ("model_step", "by", "p_end")} for e in ev
           if e.get("ev") == "ears.event" and e.get("kind") == "ENDPOINT"]
    user22 = resample_poly(np.asarray(c.wav16, np.float32), 441, 320).astype(np.float32)
    n = max(len(user22), len(agent))
    st = np.zeros((n, 2), np.float32)
    st[:len(user22), 0] = user22
    st[:len(agent), 1] = agent
    sf.write(out / "mix_client_22k.wav", st, sr, subtype="PCM_16")
    sf.write(out / "agent_client_22k.wav", agent, sr, subtype="PCM_16")
    res = {"name": c.name, "language": c.language, "via_bridge": c.via_bridge, "utterances": rows, "tools": tools,
           "turns": turns, "said": said, "barges": barges, "flushes": flushes, "endpoints": eps,
           "session": [x for x in r["ctrl"] if x.get("type") in ("session.created", "session.closed", "error")]}
    (out / "client.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str))
    (out / "events.jsonl").write_text("\n".join(json.dumps(e, ensure_ascii=False, default=str) for e in ev))
    return res


def run_clients(specs: list, out: Path) -> dict:  # pragma: no cover - needs a server
    async def main():
        return await asyncio.gather(*(run_client(c) for c in specs), return_exceptions=True)
    rs = asyncio.run(main())
    res = {}
    for c, r in zip(specs, rs):
        if isinstance(r, Exception):
            res[c.name] = {"error": f"{type(r).__name__}: {r}"}
        else:
            res[c.name] = analyse(c, r, out / c.name)
    return res
