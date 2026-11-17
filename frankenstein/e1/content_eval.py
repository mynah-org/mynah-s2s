"""Content evidence for a speaking session (G0 / G3): the ASR transcript of what the talker actually played, per agent
turn, against the text that turn was given. CPU is enough (a minute of audio).

Per adapter turn: the played audio span is cut from agent_24k.wav using the talker events (chunk delivery and
playback times in talker.jsonl); the turn's reference text is the adapter's text for it (summary.json
spoken_turns). A turn is COMPLETE when its talker segment ended with codec EOS and nothing of it was cancelled;
otherwise it is CUT and only an upper bound of the error is meaningful.

  python e1/content_eval.py RUN_IT_DIR --lab-multilingual DIR [--device cpu]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():  # pragma: no cover - model run on the GPU host (CPU)
    import numpy as np
    import soundfile as sf
    import torch
    from scipy.signal import resample_poly
    from e1.g23 import wer_cer
    from e1.live_asr import LiveASR
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--lab-multilingual", required=True)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    run = Path(a.run)
    s = json.load(open(run / "summary.json"))
    fr = [json.loads(x) for x in open(run / "frames.jsonl")]
    t0 = fr[0]["t_ns"] / 1e9 - 0.16                       # user sample 0 (first 160 ms chunk precedes frame 0)
    ev = [json.loads(x) for x in open(run / "talker.jsonl")]
    agent, sr = sf.read(run / "agent_24k.wav", dtype="float32")
    spans, ended, cancelled = {}, {}, set()
    for e in ev:
        if e.get("ev") == "audio" and e.get("t_play") is not None:
            lo, hi = e["t_play"], e["t_play"] + e["n"] / sr
            a0, a1 = spans.get(e["turn"], (lo, hi))
            spans[e["turn"]] = (min(a0, lo), max(a1, hi))
        elif e.get("ev") == "done":
            ended[e["turn"]] = e.get("ended")
        elif e.get("ev") == "cancelled":
            cancelled.add(e["turn"])
    torch.set_num_threads(16)
    live = LiveASR(a.lab_multilingual, lang="it", device=a.device)
    rows = []
    for t in s["spoken_turns"]:
        turn = t["turn"]
        if turn not in spans:
            rows.append({"turn": turn, "holding": t.get("holding"), "ref": t["text"], "played": False})
            continue
        lo, hi = spans[turn]
        i0, i1 = max(0, int((lo - t0 - 0.1) * sr)), min(len(agent), int((hi - t0 + 0.2) * sr))
        x = agent[i0:i1]
        x16 = resample_poly(x, 2, 3).astype(np.float32)
        x16 = np.pad(x16, (8000, 16000))                  # >= 1 s and a little silence around
        hyp = live.transcribe(torch.from_numpy(x16), chunk_s=60.0)
        complete = ended.get(turn) == "eos" and turn not in cancelled
        rows.append({"turn": turn, "holding": t.get("holding"), "played": True, "complete": complete,
                     "audio_s": round(len(x) / sr, 2), "ref": t["text"], "hyp": hyp, **wer_cer(t["text"], hyp)})
        print(turn, "COMPLETE" if complete else "CUT", rows[-1]["cer"], "|", t["text"][:60], "|", hyp[:60], flush=True)
    comp = [r for r in rows if r.get("complete")]
    ref = " ".join(r["ref"] for r in comp)
    hyp = " ".join(r["hyp"] for r in comp)
    res = {"turns": rows, "complete_turns": len(comp), "complete_total": wer_cer(ref, hyp)}
    (run / "content_eval.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print("CONTENT_COMPLETE", json.dumps(res["complete_total"]), "turns", len(comp), flush=True)


if __name__ == "__main__":
    main()


def played_spans(run: Path):
    """turn -> (t_lo, t_hi) wall-clock span of the audio the player scheduled, plus ended / cancelled sets."""
    spans, ended, cancelled = {}, {}, set()
    for e in map(json.loads, open(run / "talker.jsonl")):
        if e.get("ev") == "audio" and e.get("t_play") is not None:
            lo, hi = e["t_play"], e["t_play"] + e["n"] / 24000
            a0, a1 = spans.get(e["turn"], (lo, hi))
            spans[e["turn"]] = (min(a0, lo), max(a1, hi))
        elif e.get("ev") == "done":
            ended[e["turn"]] = e.get("ended")
        elif e.get("ev") == "cancelled":
            cancelled.add(e["turn"])
    return spans, ended, cancelled


def build_items(run: Path, items_dir: Path, prefix: str) -> dict:
    """G3b: one <items_dir>/<prefix>_t<turn>/output.wav (16-bit PCM, 24 kHz) per played agent turn, for the offline
    evaluation ASR (v2/scripts/asr_eval.py layout). Returns {item: {ref, complete, holding}}."""
    import numpy as np
    import soundfile as sf
    s = json.load(open(run / "summary.json"))
    fr = [json.loads(x) for x in open(run / "frames.jsonl")]
    t0 = fr[0]["t_ns"] / 1e9 - 0.16
    agent, sr = sf.read(run / "agent_24k.wav", dtype="float32")
    spans, ended, cancelled = played_spans(run)
    refs = {}
    for t in s["spoken_turns"]:
        turn = t["turn"]
        if turn not in spans:
            continue
        lo, hi = spans[turn]
        x = agent[max(0, int((lo - t0 - 0.1) * sr)): min(len(agent), int((hi - t0 + 0.2) * sr))]
        if len(x) < sr // 4:
            continue
        name = f"{prefix}_t{turn:02d}"
        (items_dir / name).mkdir(parents=True, exist_ok=False)
        sf.write(items_dir / name / "output.wav", np.clip(x, -1, 1), sr, subtype="PCM_16")
        refs[name] = {"ref": t["text"], "holding": bool(t.get("holding")),
                      "complete": ended.get(turn) == "eos" and turn not in cancelled}
    return refs


def score_items(items_dir: Path, refs: dict, tag: str = "omni3b") -> dict:
    from e1.g23 import wer_cer
    asr = {r["scenario"]: " ".join(x["text"] for x in r["segments"])
           for r in json.load(open(items_dir / f"asr_{tag}.json"))["sessions"]}
    rows = {k: {**v, "hyp": asr.get(k, ""), **wer_cer(v["ref"], asr.get(k, ""))} for k, v in refs.items()}
    out = {"items": rows}
    for prefix in sorted({k.rsplit("_t", 1)[0] for k in rows}):
        comp = [r for k, r in rows.items() if k.startswith(prefix + "_t") and r["complete"]]
        out[prefix] = {"complete_turns": len(comp), **wer_cer(" ".join(r["ref"] for r in comp),
                                                              " ".join(r["hyp"] for r in comp))}
    return out
