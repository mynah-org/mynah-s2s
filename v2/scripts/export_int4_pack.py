"""Export an offline native-tinygemm INT4 pack for the V2 Thinker.

The output is ``NAME.safetensors`` plus ``NAME.json``.  It is deliberately
weight-only: the server still loads the normal checkpoint, validates the pack,
then swaps only the declared Thinker linears before any CUDA graph is captured.
"""

from __future__ import annotations

import argparse

from mynah_v2.model.quant import save_thinker_int4_pack
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
    a = ap.parse_args()
    if not a.output.endswith(".safetensors"):
        ap.error("output must end in .safetensors")
    cfg = build_cfg(ckpt=a.checkpoint, nemo_dir=a.nemo_dir, output_dir=a.output_dir)
    eng = ReferenceEngine(cfg)
    info = save_thinker_int4_pack(eng, a.output, max_group=a.int4_group, clip=a.int4_clip,
                                  include_heads=not a.no_heads)
    print("INT4_PACK", info, flush=True)


if __name__ == "__main__":
    main()
