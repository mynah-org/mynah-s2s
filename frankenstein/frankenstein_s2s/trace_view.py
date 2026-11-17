"""Print a Frankenstein trace as a per-turn timeline (frames x channels), readable at a glance.

    python -m frankenstein_s2s.trace_view TRACE.jsonl [--turn N] [--width 34]

Columns: k (model frame, 80 ms) | t (ms since the first event) | USER (ASR words, EOU/BOU, inserts) | THINKER
(predicted -> fed, function id) | FORCE | ADAPTER | TALKER | AUDIO. Events without a model frame (talker chunks,
deliveries, playback) are shown at their position in time with k = "·". A new block starts at every agent turn.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

COLUMNS = ("USER", "THINKER", "FORCE", "ADAPTER", "TALKER", "AUDIO")
SKIP = {"ev", "t_ns", "session_id", "model_step", "turn_id", "epoch"}


def column(ev: str) -> str:
    fam = ev.split(".", 1)[0]
    if fam in ("asr", "insert") or ev in ("turn.bou", "turn.eou"):
        return "USER"
    if fam in ("thinker", "function", "brain", "tool"):
        return "THINKER"
    if fam == "force":
        return "FORCE"
    if fam in ("adapter", "turn"):
        return "AUDIO" if ev == "adapter.stale_drop" else "ADAPTER"
    if fam == "talker":
        return "TALKER"
    return "AUDIO"


def cell(e: dict) -> str:
    ev = e["ev"]
    short = ev.split(".", 1)[1] if "." in ev else ev
    if ev == "thinker.frame":
        txt = f"{e.get('predicted')}->{e.get('fed')}" if e.get("forced") else f"{e.get('predicted')}"
        if e.get("func_id") not in (None, 12):
            txt += f" f{e['func_id']}"
        return txt
    if ev == "asr.word":
        return f"'{e.get('word', '?')}'"
    if ev == "adapter.commit_word":
        return f"+'{e.get('text', '')}'"
    if ev == "force.override":
        return f"{e.get('predicted')}=>{e.get('forced')}"
    extra = [f"{k}={v}" for k, v in e.items() if k not in SKIP and k not in ("what", "action", "qwen_ids", "ids")]
    if "what" in e:
        extra.insert(0, str(e["what"]))
    return short + ("(" + ",".join(extra) + ")" if extra else "")


def render(events: list, turn: int | None = None, width: int = 34) -> str:
    if not events:
        return "(empty trace)"
    t0 = events[0]["t_ns"]
    out, cur_turn, last_k = [], None, None
    head = f"{'k':>5} {'t ms':>7}  " + " | ".join(f"{c:<{width}}" for c in COLUMNS)
    for e in events:
        tid = e.get("turn_id")
        if e["ev"] in ("turn.started", "adapter.turn_open") and tid != cur_turn:
            cur_turn = tid
            if turn is None or tid == turn:
                out.append(f"\n=== agent turn {tid} (epoch {e.get('epoch')}) ===")
                out.append(head)
        if turn is not None and cur_turn != turn:
            continue
        k = e.get("model_step")
        last_k = k if k is not None else last_k
        cols = {c: "" for c in COLUMNS}
        cols[column(e["ev"])] = cell(e)[:width]
        ks = f"{k:>5}" if k is not None else f"{'·':>5}"
        out.append(f"{ks} {(e['t_ns'] - t0) / 1e6:>7.1f}  " + " | ".join(f"{cols[c]:<{width}}" for c in COLUMNS))
    if not out or not out[0].startswith("\n==="):
        out.insert(0, head)
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("trace", type=Path)
    ap.add_argument("--turn", type=int)
    ap.add_argument("--width", type=int, default=34)
    a = ap.parse_args(argv)
    events = [json.loads(line) for line in a.trace.read_text().splitlines() if line.strip()]
    print(render(events, a.turn, a.width))
    return 0


if __name__ == "__main__":
    sys.exit(main())
