"""Build Pocket TTS language packages (pocket_swap format) from Kyutai's official checkpoints, optionally upload.

    python frankenstein2/scripts/build_packages.py --out pkgs --langs it,en,fr            # w4, from the Hub
    python frankenstein2/scripts/build_packages.py --out pkgs --src SNAPSHOT_DIR --langs all
    python frankenstein2/scripts/build_packages.py --out pkgs --upload USER/pocket-tts-deltas   # needs HF_TOKEN

Source: ``kyutai/pocket-tts-without-voice-cloning`` (not gated). Its language FlowLM weights are identical to the
voice-cloning repository's (only Mimi's encoder differs), and packages never contain Mimi, so a package carries no
voice-cloning weights: cloning still needs Kyutai's gated base checkpoint.

The base is ``english_2026-09_24l``; English itself becomes a base-language package. ``w4`` (default) stores each
language's own backbone in 4-bit (what the engine runs); ``int4`` / ``int8`` store deltas against the base (more
accurate, but the engine does not load deltas yet). Every package is verified against its source after the build.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pocket_swap"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from frankenstein2.scripts.setup_models import POCKET  # noqa: E402

SRC_REPO = "kyutai/pocket-tts-without-voice-cloning"
BASE = "english_2026-09_24l"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output root: <out>/<precision>/<language>/")
    ap.add_argument("--langs", default="it,en,fr", help=f"comma list from {sorted(POCKET)}, or 'all'")
    ap.add_argument("--precision", default="w4", choices=["w4", "w8", "wfp8", "int4", "int8", "bf16"])
    ap.add_argument("--src", default="", help=f"a local {SRC_REPO} snapshot dir (default: download the files)")
    ap.add_argument("--hf-home", default=os.environ.get("HF_HOME", ""))
    ap.add_argument("--upload", default="", help="HF model repo to upload <out>/<precision> into (created public)")
    a = ap.parse_args(argv)
    from pocket_swap.package import Source, build_package, verify_package, write_repo_card
    langs = list(POCKET) if a.langs == "all" else [x.strip() for x in a.langs.split(",") if x.strip()]
    names = [POCKET[x] for x in langs]
    src, rev = Path(a.src) if a.src else None, ""
    if src is None:
        from huggingface_hub import snapshot_download
        pats = [f"languages/{n}/model.safetensors" for n in {BASE, *names}] + \
               [f"languages/{n}/tokenizer.json" for n in {BASE, *names}]
        src = Path(snapshot_download(SRC_REPO, allow_patterns=pats,
                                     cache_dir=str(Path(a.hf_home) / "hub") if a.hf_home else None))
    rev = src.name if len(src.name) == 40 else ""
    root = Path(a.out) / a.precision
    root.mkdir(parents=True, exist_ok=True)

    def source(n):
        return Source(path=str(src / "languages" / n / "model.safetensors"), repo=SRC_REPO, revision=rev,
                      file=f"languages/{n}/model.safetensors")
    rows = []
    for n in names:
        tok = src / "languages" / n / "tokenizer.json"
        m = build_package(source(BASE), None if n == BASE else source(n), n, root, tok, a.precision)
        v = verify_package(source(BASE).path, str(root / n), None if n == BASE else source(n).path)
        rows.append({"language": n, "precision": m["precision"], "MB": round(m["total_bytes"] / 1e6, 1),
                     "backbone_rel_err": round(m["backbone_rel_err"], 5), "verify": {k: v[k] for k in v
                                                                                   if not isinstance(v[k], dict)}})
        print("PKG", json.dumps(rows[-1]), flush=True)
    card = write_repo_card(root, [json.loads((root / n / "manifest.json").read_text()) for n in
                                  sorted(p.name for p in root.iterdir() if (p / "manifest.json").exists())])
    print("CARD", card, flush=True)
    if a.upload:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(a.upload, repo_type="model", private=False, exist_ok=True)
        api.upload_folder(repo_id=a.upload, folder_path=str(root), path_in_repo=a.precision,
                          commit_message=f"{a.precision} packages: {', '.join(names)}")
        api.upload_file(path_or_fileobj=str(card), path_in_repo="README.md", repo_id=a.upload,
                        commit_message="model card")
        print("UPLOADED", a.upload, a.precision, names, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
