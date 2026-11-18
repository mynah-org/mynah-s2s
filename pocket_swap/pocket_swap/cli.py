"""pocket-deltas: build / verify / inspect language packages.

    python -m pocket_swap.cli build --base EN.safetensors --base-id kyutai/pocket-tts@<rev>:languages/<lang>/model.safetensors \
        --target IT.safetensors --target-id ... --lang italian_24l --tokenizer tokenizer.json \
        --config italian_24l.yaml --precision int8 --out pocket-tts-deltas/
    python -m pocket_swap.cli build --base EN.safetensors --base-id ... --lang english_2026-09_24l \
        --tokenizer en_tokenizer.json --base-language           # the base language itself (no delta)
    python -m pocket_swap.cli verify --base EN.safetensors --package pocket-tts-deltas/italian_24l --target IT.safetensors
    python -m pocket_swap.cli card --root pocket-tts-deltas      # README.md for a model repo (nothing is uploaded)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .package import Source, build_package, verify_package, write_repo_card


def _source(path: str, ident: str | None) -> Source:
    repo = rev = file = ""
    if ident:
        head, _, file = ident.partition(":")
        repo, _, rev = head.partition("@")
    return Source(path=path, repo=repo, revision=rev, file=file)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="pocket-deltas")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--base", required=True)
    b.add_argument("--base-id", help="repo@revision:file of the base (recorded in the manifest)")
    b.add_argument("--target")
    b.add_argument("--target-id")
    b.add_argument("--base-language", action="store_true", help="package the base's own language (no delta)")
    b.add_argument("--lang", required=True)
    b.add_argument("--tokenizer", required=True)
    b.add_argument("--config", help="upstream language yaml (preprocessing, defaults)")
    b.add_argument("--precision", default="int8", choices=["exact", "bf16", "int8", "int4"])
    b.add_argument("--group", type=int, default=128, help="quantisation group along input dim (0 = per row)")
    b.add_argument("--zstd", type=int, default=None, metavar="LEVEL", help="zstd-compress tensor files")
    b.add_argument("--out", required=True)
    v = sub.add_parser("verify")
    v.add_argument("--base", required=True)
    v.add_argument("--package", required=True)
    v.add_argument("--target")
    c = sub.add_parser("card")
    c.add_argument("--root", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "build":
        if not a.base_language and not a.target:
            ap.error("--target is required unless --base-language")
        m = build_package(_source(a.base, a.base_id), None if a.base_language else _source(a.target, a.target_id),
                          a.lang, a.out, a.tokenizer, a.precision, a.group, a.zstd, a.config)
        print(json.dumps({k: m[k] for k in ("language", "precision", "total_bytes", "backbone_rel_err")}))
    elif a.cmd == "verify":
        r = verify_package(a.base, a.package, a.target)
        print(json.dumps(r, indent=1))
        return 0 if r.get("base_matches") else 1
    else:
        ms = [json.loads(p.read_text()) for p in sorted(Path(a.root).glob("*/manifest.json"))]
        print(write_repo_card(a.root, ms))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
