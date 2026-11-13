"""Compare two run_stream runs item by item (baseline vs candidate): text / function text / token ids identity and
agent.wav samples (max |diff|, first differing sample, and max |diff| in a +-0.5 s window around each FC resume).

  python scripts/codec_smoke_compare.py BASE_RUN CAND_RUN
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 22050


def item(base: Path, cand: Path) -> dict:
    sb, sc = (json.loads((d / "summary.json").read_text()) for d in (base, cand))
    ab, _ = sf.read(base / "agent.wav", dtype="float32")
    ac, _ = sf.read(cand / "agent.wav", dtype="float32")
    m = min(len(ab), len(ac))
    d = np.abs(ab[:m] - ac[:m])
    nz = np.nonzero(d)[0]
    r = {"item": base.name, "text_identical": sb.get("text") == sc.get("text"),
         "function_text_identical": sb.get("function_text") == sc.get("function_text"),
         "token_ids_identical": sb.get("token_ids") == sc.get("token_ids"),
         "fc_steps": [[(c.get("phase1_steps"), c.get("phase2_steps")) for c in s.get("fc_timeline", {}).get("calls", [])]
                      for s in (sb, sc)],
         "audio_len": [len(ab), len(ac)], "audio_max_abs_diff": float(d.max()) if m else 0.0,
         "first_diff_sample": int(nz[0]) if len(nz) else None,
         "first_diff_s": round(float(nz[0]) / SR, 3) if len(nz) else None}
    # FC resume: the agent stream frame where phase 2 ended (start + phase1 + phase2 steps), 80 ms per frame
    res = []
    for c in sb.get("fc_timeline", {}).get("calls", []):
        f = (c.get("start_frame") or 0) + (c.get("phase1_steps") or 0) + (c.get("phase2_steps") or 0)
        s = int(f * 0.08 * SR)
        lo, hi = max(0, s - SR // 2), min(m, s + SR // 2)
        res.append({"resume_frame": f, "resume_s": round(s / SR, 2),
                    "window_max_abs_diff": float(d[lo:hi].max()) if hi > lo else None,
                    "base_window_rms": float(np.sqrt(np.mean(ab[lo:hi] ** 2))) if hi > lo else None,
                    "cand_window_rms": float(np.sqrt(np.mean(ac[lo:hi] ** 2))) if hi > lo else None})
    r["fc_resume"] = res
    r["identical"] = bool(r["text_identical"] and r["function_text_identical"] and r["token_ids_identical"]
                          and len(ab) == len(ac) and r["audio_max_abs_diff"] == 0.0)
    return r


def main():
    base, cand = Path(sys.argv[1]), Path(sys.argv[2])
    for b in sorted(p for p in base.iterdir() if (p / "summary.json").exists()):
        c = cand / b.name
        if (c / "summary.json").exists():
            print("CMP", json.dumps(item(b, c)), flush=True)


if __name__ == "__main__":
    main()
