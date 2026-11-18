"""Listening page for a run: per arm the three mixes and the agent turns, with the key measurements.

usage: python listen_index.py RUN_DIR LISTEN_DIR   (copies the per-turn wavs, writes LISTEN_DIR/index.html)
"""

from __future__ import annotations

import html
import json
import shutil
import sys
from pathlib import Path


def main():  # pragma: no cover - file plumbing
    run, ld = Path(sys.argv[1]), Path(sys.argv[2])
    ld.mkdir(parents=True, exist_ok=True)
    info = json.load(open(run / "run.json"))
    rows = [f"<h1>Frankenstein v2 — {html.escape(run.name)}</h1><p>Mixes: <b>stream</b> = session clock (L user, "
            "R agent as heard), <b>wall</b> = wall clock (shows the loop lag), <b>agent</b> = agent channel only.</p>"]
    for arm, a in info.get("arms", {}).items():
        if a.get("status") != "DONE":
            rows.append(f"<h2>{arm}: {html.escape(str(a))}</h2>")
            continue
        s = json.load(open(run / arm / "summary.json"))
        rows.append(f"<h2>Arm {arm} ({'reactive' if s['reactive'] else 'fixed track'})</h2>")
        for f in ("mix_stream_24k.wav", "mix_wall_24k.wav", "agent_24k.wav"):
            rows.append(f"<div>{f} <audio controls preload=none src='{arm}/{f}'></audio></div>")
        rows.append("<table border=1 cellpadding=3><tr><th>utt</th><th>endpoint s</th><th>holding s</th>"
                    "<th>reply s</th><th>early endpoints</th></tr>")
        for u in s["metrics"]["utterances"]:
            rows.append(f"<tr><td>{u['label']}</td><td>{u.get('endpoint_s')}</td><td>{u.get('first_holding_s')}</td>"
                        f"<td>{u.get('first_reply_s')}</td><td>{u.get('early_endpoints')}</td></tr>")
        rows.append("</table><h3>Agent turns (ASR of the played audio)</h3><ul>")
        for c in s["content"]:
            f = f"agent_turn_e{c['epoch']}_t{c['turn']}.wav"
            if (run / arm / f).exists():
                shutil.copy(run / arm / f, ld / arm / f)
            rows.append(f"<li>turn {c['turn']} (epoch {c['epoch']}, {c['played_s']} s, complete={c['complete']}, "
                        f"CER={c['cer_vs_speech']}) <audio controls preload=none src='{arm}/{f}'></audio><br>"
                        f"said: {html.escape(c['speech'])}<br>heard (ASR): {html.escape(c['asr'])}</li>")
        rows.append("</ul><h3>History</h3><pre>" + html.escape(json.dumps(s["history"], indent=1,
                                                                            ensure_ascii=False)) + "</pre>")
    ws = run / "ws"
    if ws.is_dir():
        rows.append("<h2>WebSocket sessions (client side: L user, R agent as received)</h2>")
        for d in sorted(ws.iterdir()):
            if not (d / "client.json").exists():
                continue
            c = json.load(open(d / "client.json"))
            (ld / "ws" / d.name).mkdir(parents=True, exist_ok=True)
            shutil.copy(d / "mix_client_22k.wav", ld / "ws" / d.name / "mix_client_22k.wav")
            fa = ", ".join(f"{u['label']} {u['first_audio_s']}" for u in c["utterances"])
            rows.append(f"<h3>{d.name} ({c['language']}{', via the web UI bridge' if c['via_bridge'] else ''})</h3>"
                        f"<audio controls preload=none src='ws/{d.name}/mix_client_22k.wav'></audio>"
                        f"<p>first agent audio after each utterance (s): {html.escape(fa)}<br>tools: "
                        f"{html.escape(json.dumps(c['tools'], ensure_ascii=False))}<br>user turns: "
                        f"{html.escape(' | '.join(t or '' for t in c['turns']))}<br>agent said: "
                        f"{html.escape(' | '.join(x or '' for x in c['said']))}</p>")
    (ld / "index.html").write_text("<meta charset=utf-8><body style='font-family:sans-serif'>" + "\n".join(rows))


if __name__ == "__main__":
    main()
