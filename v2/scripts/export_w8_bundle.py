"""Export the standard Q8 runtime scopes to a safetensors bundle."""

from __future__ import annotations

import argparse

from mynah_v2.model.quant import save_w8_bundle
from mynah_v2.model.reference import DEFAULT_CKPT, DEFAULT_NEMO, ReferenceEngine, build_cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output", help="output .safetensors path; a sibling .json manifest is written")
    ap.add_argument("--checkpoint", default=str(DEFAULT_CKPT))
    ap.add_argument("--nemo-dir", default=str(DEFAULT_NEMO))
    ap.add_argument("--output-dir", default="/tmp/mynah-v2-w8-export")
    ap.add_argument("--no-heads", action="store_true", help="exclude text/function output heads")
    ap.add_argument("--no-eartts", action="store_true", help="exclude EarTTS Gemma backbone")
    ap.add_argument("--no-extended", action="store_true", help="exclude perception/RNNT-joint/TTS auxiliary scopes")
    a = ap.parse_args()
    if not a.output.endswith(".safetensors"):
        ap.error("output must end in .safetensors")
    cfg = build_cfg(ckpt=a.checkpoint, nemo_dir=a.nemo_dir, output_dir=a.output_dir)
    eng = ReferenceEngine(cfg)
    print("W8_PACK", save_w8_bundle(
        eng, a.output, include_heads=not a.no_heads, include_eartts=not a.no_eartts,
        include_extended=not a.no_extended,
    ), flush=True)


if __name__ == "__main__":
    main()
