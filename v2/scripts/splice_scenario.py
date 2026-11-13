"""Build a test input by splicing utterances of the existing synthetic scenario WAVs (no new TTS):
    splice_scenario.py OUT.wav  WAV:start,end  @gap_seconds  WAV:start,end  ...  [@tail]
Writes OUT.wav (16 kHz) and OUT.json with the utterance timeline (ground truth for latency/grounding checks)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent))
from run_stream import load_16k  # noqa: E402


def main():
    out = Path(sys.argv[1])
    parts, meta, t = [], [], 0.0
    for tok in sys.argv[2:]:
        if tok.startswith("@"):
            n = int(float(tok[1:]) * 16000)
            parts.append(np.zeros(n, np.float32))
            t += n / 16000
            continue
        w, seg = tok.rsplit(":", 1)
        s0, s1 = (float(v) for v in seg.split(","))
        x = load_16k(w)[int(s0 * 16000):int(s1 * 16000)].numpy()
        parts.append(x)
        meta.append({"start": round(t, 3), "end": round(t + len(x) / 16000, 3), "src": f"{Path(w).name}:{seg}"})
        t += len(x) / 16000
    y = np.concatenate(parts) if parts else np.zeros(0, np.float32)
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out, y, 16000)
    out.with_suffix(".json").write_text(json.dumps({"duration": round(t, 2), "utterances": meta}, indent=1))
    print(out, round(t, 2), meta)


if __name__ == "__main__":
    torch.set_num_threads(4)
    main()
