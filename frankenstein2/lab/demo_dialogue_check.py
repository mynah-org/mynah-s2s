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


def save_samples(out, tag, sent, agent, lines):  # pragma: no cover - needs a run
    """Small Opus files for the web UI: each caller line (24 kHz mono) and the whole dialogue as heard (caller left,
    agent right, 16 kHz), both streams real-time from the session start."""
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    for i, x in enumerate(lines):
        if len(x):
            sf.write(out / f"{tag}_line{i}.ogg", x.astype(np.float32), 24000, format="OGG", subtype="OPUS")
    if sent and agent:
        u = np.concatenate(sent).astype(np.float32)
        a = resample_poly(np.concatenate(agent).astype(np.float32) / 32768, 320, 441).astype(np.float32)
        n = max(len(u), len(a))
        st = np.zeros((n, 2), np.float32)
        st[:len(u), 0], st[:len(a), 1] = u, a
        sf.write(out / f"{tag}_dialogue.ogg", st, 16000, format="OGG", subtype="OPUS")


def ratio(a: str, b: str) -> float:
    a, b = "".join(re.findall(r"\w", a.lower())), "".join(re.findall(r"\w", b.lower()))
    return round(difflib.SequenceMatcher(None, a, b).ratio(), 3)


async def run_lang(url, lang, lines, out, scenario=None, recorded=None):
    """``recorded``: {line text: 24 kHz float32 audio} to stream instead of asking the engine to speak the line."""  # pragma: no cover - needs a running engine
    import numpy as np
    import soundfile as sf
    import websockets
    from scipy.signal import resample_poly
    r = {"lang": lang, "lines": [], "tools": [], "errors": []}
    agent = []
    caller_wav = []
    sent: list = []                     # the session's input as streamed (silence + caller lines), 16 kHz
    async with websockets.connect(url, max_size=2 ** 24) as ws:
        await ws.send(json.dumps({"type": "session.start", "language": lang, "input_rate": 16000,
                                  **({"scenario": scenario} if scenario else {})}))
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
                        if d.get("reason") == "endpoint_merged" and state["asr"]:
                            state["asr"].pop()      # the engine cancelled that turn and re-heard it whole
                        state["asr"].append((time.monotonic(), d.get("text", "")))
                    elif ev == "llm.done":
                        state["done"] = len(state["asr"])
                    elif ev == "tool.started":
                        state["pending_tools"] += 1
                        r["tools"].append(d.get("name"))
                    elif ev == "tool.completed":
                        state["pending_tools"] = max(0, state["pending_tools"] - 1)
                        state["t_tool_done"] = time.monotonic()
                        r.setdefault("results", []).append({"name": d.get("name"), "status": d.get("status"),
                                                            "ms": d.get("ms"),
                                                            "result": str(d.get("result"))[:200]})
                    elif ev == "tts.segment" and not d.get("holding"):
                        r.setdefault("said", []).append(d.get("text"))

        async def pump():
            t0 = time.monotonic()
            n = 0
            state["t0"] = t0
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
                sent.append(chunk)
                await ws.send((np.clip(chunk, -1, 1) * 32767).astype("<i2").tobytes())
                n += 1
                await asyncio.sleep(max(0.0, t0 + n * 0.08 - time.monotonic()))

        rd = asyncio.create_task(reader())
        pm = asyncio.create_task(pump())
        await asyncio.sleep(1.0)
        for i, text in enumerate(lines):
            before = len(state["asr"])
            if recorded and text in recorded:
                pcm24 = np.asarray(recorded[text], np.float32)
            else:
                await ws.send(json.dumps({"type": "demo.say", "id": i, "text": text}))
                t_end = time.monotonic() + 30
                while not state.get(f"done{i}") and time.monotonic() < t_end:
                    await asyncio.sleep(0.05)
                pcm24 = np.concatenate(state["lines"].get(i) or [np.zeros(0, "<i2")]).astype(np.float32) / 32768
            if len(pcm24) and np.abs(pcm24).max() > 0:    # a steady caller level (the talker's varies a lot)
                pcm24 = (pcm24 / np.abs(pcm24).max() * 0.7).astype(np.float32)
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
    tag = f"{scenario}_{lang}" if scenario else lang
    save_samples(out, tag, sent, agent, caller_wav)
    sf.write(out / f"{tag}_agent_22k.wav", np.concatenate(agent or [np.zeros(1, "<i2")]).astype(np.float32) / 32768,
             22050, subtype="PCM_16")
    sf.write(out / f"{tag}_caller_24k.wav", np.concatenate(caller_wav or [np.zeros(1, np.float32)]), 24000,
             subtype="PCM_16")
    hard = 0.6 if lang in ("ja", "ko", "hi") else 0.7
    r["pass"] = (not r["errors"] and all(x.get("answered") and x.get("ratio", 0) >= hard for x in r["lines"])
                 and bool(r["tools"]))
    return r


async def run_many(url, jobs, out, parallel=1, recorded=None, on_result=None):
    """jobs: [(lang, lines, scenario)] -> {"<scenario|store>:<lang>": result}. Up to ``parallel`` sessions at once
    (the engine batches them: cross-language talker steps, one LLM batch); a session starts as soon as a slot frees.
    recorded: {lang: {line text: 24 kHz audio}} to stream instead of asking the engine to speak."""  # pragma: no cover
    sem = asyncio.Semaphore(max(1, int(parallel)))
    res = {}

    async def one(lang, lines, sc):
        async with sem:
            t0 = time.monotonic()
            try:
                r = await run_lang(url, lang, lines, out, sc, (recorded or {}).get(lang))
            except Exception as e:  # noqa: BLE001 - one language's failure must not stop the others
                r = {"lang": lang, "lines": [], "tools": [], "errors": [f"{type(e).__name__}: {e}"[:300]],
                     "pass": False}
            r["scenario"] = sc or "store"
            r["wall_s"] = round(time.monotonic() - t0, 1)
            res[f"{sc or 'store'}:{lang}"] = r
            if on_result:
                on_result(r)
    await asyncio.gather(*(one(*j) for j in jobs))
    return {f"{sc or 'store'}:{lang}": res[f"{sc or 'store'}:{lang}"] for lang, _, sc in jobs}


def main():  # pragma: no cover - needs a running engine
    from server import DEMO_SCRIPT
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://127.0.0.1:8772/v1/session")
    ap.add_argument("--langs", default="it,en,fr")
    ap.add_argument("--out", required=True)
    ap.add_argument("--scenarios", default="", help="scenario ids (frankenstein2/scenarios.py); none = the store")
    ap.add_argument("--parallel", type=int, default=1, help="sessions at once (<= the server's --max-sessions)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from frankenstein2 import scenarios as SC
    jobs = []
    for sc in [x for x in a.scenarios.split(",") if x] or [None]:
        for lang in [x for x in a.langs.split(",") if x]:
            lines = SC.SCENARIOS[sc].demo.get(lang) if sc else DEMO_SCRIPT[lang]
            if lines:
                jobs.append((lang, lines, sc))
    t0 = time.monotonic()
    res = asyncio.run(run_many(a.url, jobs, out, a.parallel,
                               on_result=lambda r: print("DEMO", json.dumps(r, ensure_ascii=False), flush=True)))
    (out / "demo_dialogue.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print("DEMO_SUITE", " ".join(f"{k}={'PASS' if v['pass'] else 'FAIL'}" for k, v in res.items()),
          f"wall_s={round(time.monotonic() - t0)} parallel={a.parallel}", flush=True)


if __name__ == "__main__":
    main()
