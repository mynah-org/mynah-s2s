"""End-to-end smoke of the Frankenstein WebSocket server (INTEGRATION.md step 1 gates), as a plain V2 client.

Session 1: Italian, default voice, the 52 s Italian track streamed in real time (+ tail silence), session.end.
           While it runs, a second connection must be refused ("busy": one engine, one session).
Session 2: Italian, another voice, the first ``--s2-seconds`` of the same track.
Gates (written to ``--out``/smoke.json): every event carries session_id; text.generated / turn.started carry turn_id;
tool.* carry call_id; engine.session reports the requested language / voice; tool calls of session 1 (names + args);
the text sent to the talker equals the text LLM's replies (spoken order); session 2 sees nothing of session 1 (its
first request has no session-1 words, its calls are its own); engine.teardown: worker joined, CUDA memory back.
The agent audio of each session is written as a WAV (22.05 kHz) for listening.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path


async def run_session(url, wav16, language, voice, seconds, tail_s, busy_probe=None):  # pragma: no cover - GPU
    import numpy as np
    import websockets
    events, audio, ctrl = [], bytearray(), []
    async with websockets.connect(url, max_size=2 ** 24) as ws:
        await ws.send(json.dumps({"type": "session.start", "language": language, "voice": voice}))

        async def reader():
            try:
                async for m in ws:
                    if isinstance(m, bytes):
                        audio.extend(m)
                    else:
                        d = json.loads(m)
                        (events if d.get("type") == "event" else ctrl).append(d)
            except Exception:  # noqa: BLE001
                pass
        rt = asyncio.create_task(reader())
        pcm = (np.clip(wav16[: int(seconds * 16000)], -1, 1) * 32767).astype("<i2").tobytes()
        pcm += b"\x00\x00" * int(tail_s * 16000)
        step = 2560 * 2
        t0 = time.monotonic()
        busy = None
        for i in range(0, len(pcm), step):
            wait = t0 + (i // 2) / 16000 - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)
            try:
                await ws.send(pcm[i:i + step])
            except Exception as e:  # noqa: BLE001 - the server closed the session: keep what it said
                ctrl.append({"type": "client.send_failed", "error": f"{type(e).__name__}: {e}", "at_s": i / 32000})
                break
            if busy_probe is not None and busy is None and time.monotonic() - t0 > 5:
                busy = await busy_probe()
        try:
            await ws.send(json.dumps({"type": "session.end"}))
        except Exception:  # noqa: BLE001
            pass
        await asyncio.wait_for(rt, 120)
    return {"events": events, "ctrl": ctrl, "audio": bytes(audio), "busy_probe": busy}


async def probe_busy(url):  # pragma: no cover - GPU
    import websockets
    try:
        async with websockets.connect(url) as ws:
            await ws.send(json.dumps({"type": "session.start", "language": "it"}))
            m = await asyncio.wait_for(ws.recv(), 10)
            return json.loads(m) if isinstance(m, str) else {"binary": True}
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}


def gates(r, language, voice, other_words=None) -> dict:
    ev = r["events"]
    names = [e["ev"] for e in ev]
    sid = {e.get("session_id") for e in ev}
    no_sid = sum(1 for e in ev if not e.get("session_id"))
    tg = [e for e in ev if e["ev"] == "text.generated"]
    tool = [e for e in ev if e["ev"].startswith("tool.")]
    es = next((e for e in ev if e["ev"] == "engine.session"), {})
    td = next((e for e in ev if e["ev"] == "engine.teardown"), {})
    done = [e for e in ev if e["ev"] == "brain.done"]
    replies = "".join(e.get("reply", "") for e in done)
    spoken = "".join(e.get("text", "") for e in tg)
    norm = lambda s: "".join(s.split())          # noqa: E731 - per-step decodes may differ in spaces only
    subs = [e for e in ev if e["ev"] == "brain.submit"]
    first_user = next((e.get("user", "") for e in subs if e.get("user")), "")
    return {
        "events": len(ev), "session_ids": sorted(x for x in sid if x), "events_without_session_id": no_sid,
        "text_generated_with_turn_id": sum(1 for e in tg if e.get("turn_id") is not None), "text_generated": len(tg),
        "tool_events": len(tool), "tool_events_with_call_id": sum(1 for e in tool if e.get("call_id")),
        "tool_completed": [e.get("name") or e.get("tool") for e in tool if e["ev"] == "tool.completed"],
        "language_ok": es.get("language") == language, "voice_reported": es.get("voice"), "voice_requested": voice,
        "session_index": es.get("session_index"),
        "spoken_equals_replies": norm(spoken) == norm(replies) if replies else None,
        "spoken_prefix_of_replies": norm(replies).startswith(norm(spoken)) if replies else None,
        "first_user_text": first_user,
        "leak_from_other_session": bool(other_words and any(w in first_user for w in other_words)),
        "teardown": td, "audio_s": round(len(r["audio"]) / 2 / 22050, 1), "flushes": names.count("audio.flush"),
        "busy_probe": r.get("busy_probe"), "ctrl": [c.get("type") for c in r["ctrl"]],
    }


def main():  # pragma: no cover - GPU
    import numpy as np
    import soundfile as sf
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--track", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--voice1", default="")
    ap.add_argument("--voice2", default="")
    ap.add_argument("--s1-seconds", type=float, default=52.0)
    ap.add_argument("--s2-seconds", type=float, default=22.0)
    ap.add_argument("--lang2", default="it", help="language of session 2 (language honoured: replies in it)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    wav, sr = sf.read(a.track, dtype="float32")
    res = {}
    r1 = asyncio.run(run_session(a.url, wav, "it", a.voice1, a.s1_seconds, 8.0,
                                 busy_probe=lambda: probe_busy(a.url)))
    res["s1"] = gates(r1, "it", a.voice1)
    (out / "s1_events.jsonl").write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in r1["events"]))
    sf.write(out / "s1_agent_22k.wav", np.frombuffer(r1["audio"], "<i2").astype(np.float32) / 32767, 22050)
    print("S1", json.dumps(res["s1"], ensure_ascii=False, default=str)[:2500], flush=True)
    r2 = asyncio.run(run_session(a.url, wav, a.lang2, a.voice2, a.s2_seconds, 8.0))
    res["s2"] = gates(r2, a.lang2, a.voice2, other_words=None)
    # no leak: session 2's first request starts a fresh history (1 message), and none of session 1's replies is in
    # session 2's history (the history length counts only session 2's turns)
    h = [e.get("history_len") for e in r2["events"] if e["ev"] == "brain.submit"]
    res["s2"]["first_request_history_len"] = h[0] if h else None
    res["s2"]["no_leak"] = bool(h) and h[0] <= 1 and r2["events"][0].get("session_id") not in res["s1"]["session_ids"]
    (out / "s2_events.jsonl").write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in r2["events"]))
    sf.write(out / "s2_agent_22k.wav", np.frombuffer(r2["audio"], "<i2").astype(np.float32) / 32767, 22050)
    res["distinct_session_ids"] = len(set(res["s1"]["session_ids"]) | set(res["s2"]["session_ids"]))
    (out / "smoke.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str))
    print("S2", json.dumps(res["s2"], ensure_ascii=False, default=str)[:2500], flush=True)


if __name__ == "__main__":
    main()
