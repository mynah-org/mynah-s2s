"""Build eval/fcmulti_v1.json items from existing EN clips (no TTS): concatenation with silence gaps, optional overlay of
a second utterance a few seconds after the first ends (user speech during an open FC cycle). Keeps the source rate.
Usage: make_fcmulti.py <data/agent root> <out dir> [spec]   (refuses to overwrite existing wavs)"""

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf


def main():
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    spec = json.loads(Path(sys.argv[3] if len(sys.argv) > 3 else Path(__file__).parents[1] / "eval/fcmulti_v1.json").read_text())
    out.mkdir(parents=True, exist_ok=True)
    sr = None

    def load(rel):
        nonlocal sr
        x, r = sf.read(root / f"{rel}.wav", dtype="float32")
        sr = sr or r
        assert r == sr, (rel, r, sr)
        return x if x.ndim == 1 else x.mean(1)

    made = {}
    for k, it in spec["items"].items():
        f = out / f"{k}.wav"
        if f.exists():
            made[k] = "exists"
            continue
        xs = [load(p) for p in it["parts"]]
        gap = np.zeros(int(sr * it["gap_s"]), dtype="float32")
        y = np.concatenate([np.concatenate([x, gap]) for x in xs])
        if "overlay" in it:
            o = load(it["overlay"]["clip"])
            ov = it["overlay"]
            start = int(sr * ov["at_abs_s"]) if "at_abs_s" in ov else len(xs[0]) + int(sr * ov["at_s"])
            y = np.concatenate([y, np.zeros(max(0, start + len(o) - len(y)), dtype="float32")])
            y[start:start + len(o)] += o
        sf.write(f, y, sr)
        made[k] = round(len(y) / sr, 1)
    (out / "manifest.json").write_text(json.dumps({"spec": spec, "rate": sr, "seconds": made}, indent=1))
    print(json.dumps(made))


if __name__ == "__main__":
    main()
