"""The web UI's demo dialogue, end to end against a running engine (what webui/f2/f2.js does, in Python):

    python frankenstein2/lab/demo_dialogue_check.py --url ws://127.0.0.1:8772/v1/session --langs it,en,fr --out DIR

For each language: open a session, ask the engine to speak each caller line (demo.say), stream the returned audio
back as the session's input at real-time pace (silence in between), wait for the agent's answer, next line. Judges:
the ASR heard each line (character match), the agent answered every line with audio, and the first line (a
catalogue question) led to a tool call. Prints one DEMO line per language; WAVs of both sides in DIR.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import difflib
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "webui"))


def ratio(a: str, b: str) -> float:
    a, b = "".join(re.findall(r"\w", a.lower())), "".join(re.findall(r"\w", b.lower()))
    return round(difflib.SequenceMatcher(None, a, b).ratio(), 3)


async def run_lang(url, lang, lines, out):  # pragma: no cover - needs a running engine
    import numpy as np
    import soundfile as sf
    import websockets
    from scipy.signal import resample_poly
    r = {"lang": lang, "lines": [], "tools": [], "errors": []}
    agent = []
    caller_wav = []
    async with websockets.connect(url, max_size=2 ** 24) as ws:
        await ws.send(json.dumps({"type": "session.start", "language": lang, "input_rate": 16000}))
        state = {"asr": [], "done": 0, "loud": 0.0, "lines": {}, "pending_tools": 0, "t_tool_done": 0.0}
        queue: list = []

        async def reader():
            async for m in ws:
                if isinstance(m, bytes):
                    pcm = np.frombuffer(m, "<i2")
                    agent.append(pcm)
                    if len(pcm) and np.abs(pcm).max() > 300:
                        state["loud"] = time.monotonic()
                    continue
                d = json.loads(m)
                t = d.get("type")
                if t == "demo.audio":
                    state["lines"].setdefault(d["id"], []).append(np.frombuffer(base64.b64decode(d["pcm"]), "<i2"))
                elif t == "demo.done":
                    state["lines"].setdefault(d["id"], [])
                    state[f"done{d['id']}"] = True
                elif t == "error":
                    r["errors"].append(d.get("error"))
                elif t == "session.closed":
                    return
                elif t == "event":
                    ev = d.get("ev")
                    if ev == "asr.turn":
                        state["asr"].append((time.monotonic(), d.get("text", "")))
                    elif ev == "llm.done":
                        state["done"] = len(state["asr"])
                    elif ev == "tool.started":
                        state["pending_tools"] += 1
                        r["tools"].append(d.get("name"))
                    elif ev == "tool.completed":
                        state["pending_tools"] = max(0, state["pending_tools"] - 1)
                        state["t_tool_done"] = time.monotonic()

        async def pump():
            t0 = time.monotonic()
            n = 0
            while True:
                chunk = np.zeros(1280, np.float32)
                off = 0
                while off < 1280 and queue:
                    head = queue[0]
                    k = min(1280 - off, len(head))
                    chunk[off:off + k] = head[:k]
                    off += k
                    queue[0] = head[k:]
                    if not len(queue[0]):
                        queue.pop(0)
                await ws.send((np.clip(chunk, -1, 1) * 32767).astype("<i2").tobytes())
                n += 1
                await asyncio.sleep(max(0.0, t0 + n * 0.08 - time.monotonic()))

        rd = asyncio.create_task(reader())
        pm = asyncio.create_task(pump())
        await asyncio.sleep(1.0)
        for i, text in enumerate(lines):
            before = len(state["asr"])
            await ws.send(json.dumps({"type": "demo.say", "id": i, "text": text}))
            t_end = time.monotonic() + 30
            while not state.get(f"done{i}") and time.monotonic() < t_end:
                await asyncio.sleep(0.05)
            pcm24 = np.concatenate(state["lines"].get(i) or [np.zeros(0, "<i2")]).astype(np.float32) / 32768
            caller_wav.append(pcm24)
            if not len(pcm24):
                r["lines"].append({"text": text, "error": "no caller audio"})
                continue
            x16 = resample_poly(pcm24, 2, 3).astype(np.float32)
            queue.append(x16)
            t_send = time.monotonic()
            await asyncio.sleep(len(x16) / 16000 + 0.3)
            t_end = time.monotonic() + 30
            answered = False
            while time.monotonic() < t_end:
                await asyncio.sleep(0.2)
                heard = len(state["asr"]) > before
                spoke = state["loud"] > max(t_send, state["t_tool_done"])
                if heard and state["done"] >= len(state["asr"]) and state["pending_tools"] == 0 and spoke \
                        and time.monotonic() - state["loud"] > 1.4:
                    answered = True
                    break
            got = " ".join(t for _, t in state["asr"][before:])
            r["lines"].append({"text": text, "heard": got, "ratio": ratio(got, text), "answered": answered,
                               "answer_s": round(state["loud"] - t_send, 1) if answered else None})
        await ws.send(json.dumps({"type": "session.end"}))
        try:
            await asyncio.wait_for(rd, 30)
        except asyncio.TimeoutError:
            rd.cancel()
        pm.cancel()
    sf.write(out / f"{lang}_agent_22k.wav", np.concatenate(agent or [np.zeros(1, "<i2")]).astype(np.float32) / 32768,
             22050, subtype="PCM_16")
    sf.write(out / f"{lang}_caller_24k.wav", np.concatenate(caller_wav or [np.zeros(1, np.float32)]), 24000,
             subtype="PCM_16")
    hard = 0.6 if lang in ("ja", "ko", "hi") else 0.7
    r["pass"] = (not r["errors"] and all(x.get("answered") and x.get("ratio", 0) >= hard for x in r["lines"])
                 and bool(r["tools"]))
    return r


def main():  # pragma: no cover - needs a running engine
    from server import DEMO_SCRIPT
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://127.0.0.1:8772/v1/session")
    ap.add_argument("--langs", default="it,en,fr")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    res = {}
    for lang in [x for x in a.langs.split(",") if x]:
        r = asyncio.run(run_lang(a.url, lang, DEMO_SCRIPT[lang], out))
        res[lang] = r
        print("DEMO", json.dumps(r, ensure_ascii=False), flush=True)
    (out / "demo_dialogue.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print("DEMO_SUITE", " ".join(f"{k}={'PASS' if v['pass'] else 'FAIL'}" for k, v in res.items()), flush=True)


if __name__ == "__main__":
    main()
