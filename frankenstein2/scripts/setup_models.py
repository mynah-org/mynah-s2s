"""Download the Frankenstein v2 models from Hugging Face for a set of languages, and write the paths to an env file.

    python frankenstein2/scripts/setup_models.py --langs it,en,fr            # the tested languages (default)
    python frankenstein2/scripts/setup_models.py --langs de,es --dry-run    # what would be fetched, and its size

What it fetches (official repositories, nothing re-hosted):
  ears    nineninesix/tyger                       Nemotron 3.5 ASR streaming 0.6B + turn-taking GRU (OpenMDW-1.1)
  brain   Qwen/Qwen3.5-4B                         the text LLM with tools (Apache-2.0)
  talker  kyutai/pocket-tts                       languages/<lang>/model.safetensors only (voice cloning; CC-BY-4.0,
                                                  gated: accept the terms on the model page and set HF_TOKEN)
          kyutai/pocket-tts-without-voice-cloning languages/<lang>/tokenizer.* (CC-BY-4.0, not gated)
  voice   kyutai/tts-voices                       one reference WAV for cloning (default alba-mackenna/casual.wav,
                                                  CC-BY-4.0; other folders there have other licences)

Languages with an official Pocket TTS model: it, en, fr, de, es, pt, nl (the 24-layer variants are used). it/en/fr
are the ones wired and tested end to end; the others download and load, but the engine's per-language pieces
(normaliser, holding phrases, prompts) are not written for them yet: see docs/frankenstein2-languages.md.

Weights go to <hf-home>/hub (a normal Hugging Face cache; HF_HOME=<hf-home> reuses it). The env file holds F2_*
variables that the server command line and the opt-in GPU tests read.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

POCKET = {"it": "italian_24l", "en": "english_2026-09_24l", "fr": "french_24l", "de": "german_24l",
          "es": "spanish_24l", "pt": "portuguese_24l", "nl": "dutch_24l"}
TESTED = ("it", "en", "fr")
EARS = "nineninesix/tyger"
POCKET_CLONE = "kyutai/pocket-tts"
POCKET_TOK = "kyutai/pocket-tts-without-voice-cloning"
VOICES = "kyutai/tts-voices"
LLM_DEFAULT = "Qwen/Qwen3.5-4B"


def plan(langs: list, llm: str, voice: str) -> list:
    """[(repo, allow_patterns or None = whole repo, role)]."""
    return [
        (EARS, None, "ears"),
        (llm, ["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja", "LICENSE*"], "llm"),
        (POCKET_CLONE, [f"languages/{POCKET[x]}/model.safetensors" for x in langs], "talker weights (gated)"),
        (POCKET_TOK, [f"languages/{POCKET[x]}/tokenizer*" for x in langs], "talker tokenizers"),
        (VOICES, [voice], "reference voice"),
    ]


def check_langs(raw: str) -> list:
    langs = [x.strip().lower() for x in raw.split(",") if x.strip()]
    bad = [x for x in langs if x not in POCKET]
    if bad:
        sys.exit(f"no official Pocket TTS model for {bad}; official: {sorted(POCKET)}. Community models for other "
                 "languages are listed in docs/frankenstein2-languages.md (not wired into this script).")
    return langs


def sizes(api, repo: str, patterns, token) -> int:
    import fnmatch
    info = api.model_info(repo, files_metadata=True, token=token)
    tot = 0
    for s in info.siblings or []:
        if patterns is None or any(fnmatch.fnmatch(s.rfilename, p) for p in patterns):
            tot += s.size or 0
    return tot


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--langs", default=",".join(TESTED), help=f"comma list from {sorted(POCKET)}")
    ap.add_argument("--hf-home", default=os.environ.get("HF_HOME") or str(Path(os.environ.get(
        "MYNAH_HOME", Path.home() / "mynah-s2s")) / "hf"), help="Hugging Face home (weights in <hf-home>/hub)")
    ap.add_argument("--llm", default=LLM_DEFAULT, help="the LLM repository (Qwen3.5 family)")
    ap.add_argument("--voice", default="alba-mackenna/casual.wav", help=f"reference WAV inside {VOICES}")
    ap.add_argument("--env-file", default="", help="where to write the F2_* paths (default <hf-home>/f2.env)")
    ap.add_argument("--dry-run", action="store_true", help="only list what would be downloaded and its size")
    a = ap.parse_args(argv)
    langs = check_langs(a.langs)
    untested = [x for x in langs if x not in TESTED]
    if untested:
        print(f"NOTE: {untested} have official models but are not wired / tested in the engine yet "
              "(docs/frankenstein2-languages.md).")
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError:
        sys.exit("pip install huggingface_hub (or run inside the engine's environment)")
    token = os.environ.get("HF_TOKEN") or None
    cache = Path(a.hf_home) / "hub"
    steps = plan(langs, a.llm, a.voice)
    if a.dry_run:
        api = HfApi()
        total = 0
        for repo, pats, role in steps:
            try:
                n = sizes(api, repo, pats, token)
                total += n
                print(f"{role:<24} {repo:<42} {n / 2 ** 30:7.2f} GiB  {pats or 'all files'}")
            except Exception as e:  # noqa: BLE001 - gated / offline: say so, keep listing
                print(f"{role:<24} {repo:<42}    ?      ({type(e).__name__}: {str(e).splitlines()[0][:80]})")
        print(f"total (known) {total / 2 ** 30:.2f} GiB -> {cache}")
        return 0
    paths = {}
    for repo, pats, role in steps:
        print(f"[setup] {role}: {repo}", flush=True)
        try:
            paths[repo] = snapshot_download(repo, allow_patterns=pats, cache_dir=str(cache), token=token)
        except Exception as e:  # noqa: BLE001
            hint = (" (gated: open https://huggingface.co/kyutai/pocket-tts, accept the terms, then export "
                    "HF_TOKEN=<a read token>)") if repo == POCKET_CLONE else ""
            sys.exit(f"download failed for {repo}: {type(e).__name__}: {e}{hint}")
    env = {
        "F2_EARS_BUNDLE": paths[EARS],
        "F2_QWEN_MODEL": paths[a.llm],
        "F2_POCKET_WEIGHTS": paths[POCKET_CLONE],
        "F2_POCKET_TOKENIZERS": paths[POCKET_TOK],
        "F2_VOICE": str(Path(paths[VOICES]) / a.voice),
        "F2_LANGS": ",".join(langs),
        "HF_HOME": a.hf_home,
    }
    out = Path(a.env_file or Path(a.hf_home) / "f2.env")
    out.write_text("".join(f"export {k}={v}\n" for k, v in env.items()))
    print(f"[setup] wrote {out}\n\nNext:\n  source {out}\n  make f2-serve        # or: make webui-gpu ENGINE=frankenstein2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
