"""Export an offline native-tinygemm INT4 bundle for the V2 runtime.

The default output covers the Thinker, output heads, and graph-safe EarTTS
Gemma backbone.  ``NAME.safetensors`` plus ``NAME.json`` are deliberately
weight-only: the server still loads the normal checkpoint, validates the whole
bundle, then swaps only the declared linears before any CUDA graph is captured.
"""

from __future__ import annotations

import argparse

from mynah_v2.model.quant import save_int4_bundle, save_thinker_int4_pack
from mynah_v2.model.reference import DEFAULT_CKPT, DEFAULT_NEMO, ReferenceEngine, build_cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output", help="output .safetensors path; a sibling .json manifest is written")
    ap.add_argument("--checkpoint", default=str(DEFAULT_CKPT))
    ap.add_argument("--nemo-dir", default=str(DEFAULT_NEMO))
    ap.add_argument("--output-dir", default="/tmp/mynah-v2-int4-export")
    ap.add_argument("--int4-group", type=int, default=128, choices=[32, 64, 128])
    ap.add_argument("--int4-clip", action="store_true", help="per-group clipping search; slower export, lower MSE")
    ap.add_argument("--no-heads", action="store_true", help="exclude text/function output heads from the pack")
    ap.add_argument("--thinker-only", action="store_true",
                    help="write the legacy Thinker-only pack instead of the combined bundle")
    ap.add_argument("--no-eartts", action="store_true",
                    help="exclude the EarTTS Gemma backbone from the combined bundle")
    a = ap.parse_args()
    if not a.output.endswith(".safetensors"):
        ap.error("output must end in .safetensors")
    cfg = build_cfg(ckpt=a.checkpoint, nemo_dir=a.nemo_dir, output_dir=a.output_dir)
    eng = ReferenceEngine(cfg)
    if a.thinker_only:
        info = save_thinker_int4_pack(eng, a.output, max_group=a.int4_group, clip=a.int4_clip,
                                      include_heads=not a.no_heads)
    else:
        info = save_int4_bundle(eng, a.output, max_group=a.int4_group, clip=a.int4_clip,
                                include_heads=not a.no_heads, include_eartts=not a.no_eartts)
    print("INT4_PACK", info, flush=True)


if __name__ == "__main__":
    main()
