"""Build the talker's language folder (deltas against the English base, per model family), verify it, upload it.

    python frankenstein2/scripts/build_packages.py --out pocket-tts-deltas                 # every source below
    python frankenstein2/scripts/build_packages.py --out pkgs --langs it,fr --families 24l
    python frankenstein2/scripts/build_packages.py --out pocket-tts-deltas --upload USER/pocket-tts-deltas

Layout written (the engine scans it; docs/frankenstein2-languages.md):

    <out>/<code>/language.json
    <out>/<code>/<family>/<precision>/      precision int4 / int8 / bf16: the language's FlowLM backbone as a delta
                                            against <out>/en/<family>/base, plus its own embedding / head / tokenizer
    <out>/en/<family>/base/                 the English base model verbatim (the shared backbone, and English itself)

Families: ``24l`` (the 24-layer models) and ``6l`` (the 6-layer students). Sources: Kyutai's official languages from
``kyutai/pocket-tts-without-voice-cloning`` (not gated; their FlowLM weights are identical to the voice-cloning
repository's, and packages never contain Mimi, so no voice-cloning weights are redistributed) and community models
with an open licence, listed in SOURCES with their licence, which each package's manifest carries.

Adding a language tomorrow = one SOURCES row (or any folder in this layout built with pocket_swap).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pocket_swap"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from frankenstein2 import langs  # noqa: E402

KY = "kyutai/pocket-tts-without-voice-cloning"
KY_LIC = ("CC-BY-4.0", None)          # None = pocket_swap's Kyutai notice


def _ky(code, fam, name):
    return {"code": code, "family": fam, "repo": KY, "file": f"languages/{name}/model.safetensors",
            "tokenizer": f"languages/{name}/tokenizer.json", "licence": "CC-BY-4.0", "official": True}


def _cm(code, fam, repo, file, tok, lic):
    return {"code": code, "family": fam, "repo": repo, "file": file, "tokenizer": tok, "licence": lic,
            "official": False}


OFFICIAL = {"en": ("english_2026-09_24l", "english_2026-09"), "it": ("italian_24l", "italian"),
            "fr": ("french_24l", "french"), "de": ("german_24l", "german"), "es": ("spanish_24l", "spanish"),
            "pt": ("portuguese_24l", "portuguese"), "nl": ("dutch_24l", "dutch")}
SOURCES = [s for c, (n24, n6) in OFFICIAL.items() for s in (_ky(c, "24l", n24), _ky(c, "6l", n6))] + [
    # community models, open licences only (non-commercial ones are left out); not by Kyutai
    _cm("ko", "24l", "seastar105/pocket-tts-korean-300m", "model.safetensors", "tokenizer.model", "CC-BY-4.0"),
    _cm("tr", "24l", "alperiox/pocket-tts-turkish", "teacher.safetensors", "tokenizer.json", "MIT"),
    _cm("tr", "6l", "alperiox/pocket-tts-turkish", "model.safetensors", "tokenizer.json", "MIT"),
    _cm("ja", "24l", "takuma104/pocket-tts-ja-cc", "japanese_24l/model.safetensors", "tokenizer.model", "CC-BY-4.0"),
    _cm("ja", "6l", "takuma104/pocket-tts-ja-cc", "japanese/model.safetensors", "tokenizer.model", "CC-BY-4.0"),
    _cm("cs", "6l", "vvolhejn/pocket-tts-czech", "model.safetensors", "tokenizer.model", "CC-BY-4.0"),
    _cm("hi", "6l", "saryps-labs/pocket-tts-hindi", "model.safetensors", "tokenizer.model", "CC-BY-4.0"),
    _cm("pl", "6l", "shefowl/pocket-tts-polish-6l", "model.safetensors", "pl_bpe4000.model", "CC-BY-4.0"),
    _cm("ru", "6l", "ArtShtorm/Shtorm-PocketTTS-RU", "model_fast.safetensors", "tokenizer.model", "CC-BY-4.0"),
    _cm("et", "6l", "cbentes/pocket-tts-estonian", "model.safetensors", "tokenizer.model", "CC-BY-SA-4.0"),
]


def notice(src: dict, rev: str) -> str | None:
    if src["official"]:
        return None
    return (f"The tensors in this package are derived from {src['repo']}@{rev[:10]} ({src['file']}), released under "
            f"{src['licence']} by its authors, itself derived from Kyutai's Pocket TTS (CC-BY-4.0, "
            "https://huggingface.co/kyutai). Changes: re-expressed as a difference from the English base named in "
            "manifest.json (quantised as stated in 'precision'); the Mimi codec is omitted and comes from Kyutai's "
            "base checkpoint. Use is subject to the source's licence and to Kyutai's terms on the original model "
            "pages, including the prohibited uses listed there (e.g. voice cloning without explicit and lawful "
            "consent). Neither Kyutai nor the source's authors endorse this package.")


def fetch(src: dict, hf_home: str):
    from huggingface_hub import HfApi, hf_hub_download
    kw = {"cache_dir": str(Path(hf_home) / "hub")} if hf_home else {}
    rev = HfApi().model_info(src["repo"]).sha
    w = hf_hub_download(src["repo"], src["file"], revision=rev, **kw)
    t = hf_hub_download(src["repo"], src["tokenizer"], revision=rev, **kw)
    return w, t, rev


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--langs", default="all", help="comma list of codes, or 'all' (every SOURCES row)")
    ap.add_argument("--families", default="24l,6l")
    ap.add_argument("--precisions", default="int4,int8,bf16", help="delta precisions: int4 / int8 / bf16")
    ap.add_argument("--hf-home", default=os.environ.get("HF_HOME", ""))
    ap.add_argument("--upload", default="", help="HF model repo to upload <out> into (created public)")
    ap.add_argument("--upload-only", action="store_true", help="upload an existing <out> without building")
    a = ap.parse_args(argv)
    from pocket_swap.package import Source, build_package, verify_package
    root = Path(a.out)
    fams = [x for x in a.families.split(",") if x]
    precs = [x for x in a.precisions.split(",") if x]
    want = None if a.langs == "all" else {x.strip() for x in a.langs.split(",") if x.strip()}
    rows = []
    for fam in ([] if a.upload_only else fams):
        bsrc = next(s for s in SOURCES if s["code"] == "en" and s["family"] == fam)
        bw, btok, brev = fetch(bsrc, a.hf_home)
        base = Source(path=bw, repo=bsrc["repo"], revision=brev, file=bsrc["file"])
        todo = [s for s in SOURCES if s["family"] == fam and (want is None or s["code"] in want or s["code"] == "en")]
        for src in todo:
            code = src["code"]
            try:
                if code == "en":
                    w, tok, rev = bw, btok, brev
                    jobs = [("base", "exact")]
                else:
                    w, tok, rev = fetch(src, a.hf_home)
                    jobs = [(p, p) for p in precs]
                tgt = Source(path=w, repo=src["repo"], revision=rev, file=src["file"])
                for folder, prec in jobs:
                    m = build_package(base, tgt, folder, root / code / fam, tok, prec, licence=src["licence"],
                                      notice=notice(src, rev))
                    d = root / code / fam / folder
                    m.update(language=code, family=fam, source={k: src[k] for k in ("repo", "file", "licence")})
                    (d / "manifest.json").write_text(json.dumps(m, indent=1))
                    v = verify_package(base.path, str(d), tgt.path)
                    rows.append({"code": code, "family": fam, "precision": folder, "MB": round(m["total_bytes"] / 1e6, 1),
                                 "backbone_rel_err": round(m["backbone_rel_err"], 5), "licence": src["licence"],
                                 "source": src["repo"], "base_matches": v.get("base_matches")})
                    print("PKG", json.dumps(rows[-1]), flush=True)
                spec = langs.get(code).to_json()
                spec["pocket"] = src["file"] if not src["official"] else spec["pocket"]
                (root / code).mkdir(parents=True, exist_ok=True)
                (root / code / "language.json").write_text(json.dumps(spec, indent=1, ensure_ascii=False))
            except Exception as e:  # noqa: BLE001 - one bad source does not stop the others
                print("PKG_FAILED", json.dumps({"code": code, "family": fam, "error": f"{type(e).__name__}: {e}"[:300]}),
                      flush=True)
    if rows:
        write_card(root, rows)
    if a.upload:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(a.upload, repo_type="model", private=False, exist_ok=True)
        api.upload_large_folder(repo_id=a.upload, folder_path=str(root), repo_type="model")
        print("UPLOADED", a.upload, flush=True)
    return 0


def write_card(root: Path, rows: list) -> None:
    table = "\n".join(f"| {r['code']} | {r['family']} | {r['precision']} | {r['MB']} MB | {r['backbone_rel_err']} | "
                      f"[{r['source']}](https://huggingface.co/{r['source']}) | {r['licence']} |" for r in rows)
    (root / "README.md").write_text(f"""---
license: other
license_name: per-package
license_link: LICENSE.md
library_name: pocket-tts
tags: [text-to-speech, pocket-tts, weight-delta, multilingual]
---
# Pocket TTS language deltas

Language packages for [Kyutai Pocket TTS](https://huggingface.co/kyutai/pocket-tts), as used by the Frankenstein v2
engine of [mynah-s2s](https://github.com/mynah-org/mynah-s2s). Each language is a small delta against one English
base model per family (24-layer and 6-layer), stored in this repository as `en/<family>/base`.

    <code>/language.json                    name, ASR locale, fixed phrases
    <code>/<family>/<precision>/            precision int4 / int8 / bf16 (delta vs en/<family>/base)
    en/<family>/base/                       the English base model (the shared backbone)

The Mimi codec and its voice-cloning encoder are not included: they come from Kyutai's base checkpoint
(`kyutai/pocket-tts`, gated: accept Kyutai's terms). `backbone_rel_err` is the relative error of the reconstructed
transformer weights.

| code | family | precision | size | backbone_rel_err | source | licence |
|---|---|---|---|---|---|---|
{table}

## Licences and attribution

Every package's `manifest.json` names its source and licence and carries its attribution notice. Kyutai's official
languages: CC-BY-4.0. Community languages: the licence of their source repository (CC-BY-4.0, CC-BY-SA-4.0 or MIT,
as listed). Use stays subject to Kyutai's terms on the original model pages, including the prohibited uses listed
there (for example voice cloning without explicit and lawful consent). Neither Kyutai nor the community authors
endorse these packages.
""")
    (root / "LICENSE.md").write_text("Per package: see each <code>/<family>/<precision>/manifest.json ('licence', "
                                     "'source'). Kyutai Pocket TTS: CC-BY-4.0. Community sources: as listed in "
                                     "README.md.\n")


if __name__ == "__main__":
    raise SystemExit(main())
