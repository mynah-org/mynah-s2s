"""Bytes per component (precision research track, step 1): parameter bytes by module group and dtype, as loaded by the
native engine. Weight bytes read per 80 ms step ≈ the bytes of every module that runs once per step (the Thinker
backbone, its heads, EarTTS backbone + MoG head per refinement pass, RNN-T joint/pred), which together with the measured
GPU time per component (profile_components / profile_kernels) gives the effective bandwidth per component.
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import torch

from mynah_v2.model.reference import ReferenceEngine, build_cfg


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=False)
    eng = ReferenceEngine(build_cfg(output_dir=str(out / "_nemo")))
    m = eng.wrapper.model
    groups = collections.defaultdict(lambda: collections.Counter())
    for name, p in m.named_parameters():
        parts = name.split(".")
        if parts[0] == "stt_model" and parts[1] == "llm":
            sub = "thinker." + ("layers." + {"mamba": "mamba", "attention": "attention", "mlp": "mlp"}.get(
                getattr(m.stt_model.llm.layers[int(parts[3])], "block_type", "?"), "?") if parts[2] == "layers" else parts[2])
            if parts[2] == "layers":
                sub += "." + parts[-2]
        elif parts[0] == "stt_model":
            sub = "stt." + parts[1]
        elif parts[0] == "tts_model":
            sub = "tts." + ".".join(parts[1:3])
        else:
            sub = parts[0]
        groups[sub][str(p.dtype)] += p.numel() * p.element_size()
    rows = sorted(({"group": g, "MB": round(sum(c.values()) / 2**20, 1), "dtypes": {k: round(v / 2**20, 1) for k, v in c.items()}}
                   for g, c in groups.items()), key=lambda r: -r["MB"])
    tot = round(sum(r["MB"] for r in rows) / 1024, 2)
    (out / "bytes.json").write_text(json.dumps({"total_GB": tot, "groups": rows}, indent=1))
    print(json.dumps({"total_GB": tot}))
    for r in rows[:60]:
        print(f'{r["MB"]:10.1f} MB  {r["group"]:45s} {r["dtypes"]}')


if __name__ == "__main__":
    main()
