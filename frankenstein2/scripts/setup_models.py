"""Download the Frankenstein v2 models from Hugging Face for a set of languages, and write the paths to an env file.

    python frankenstein2/scripts/setup_models.py --langs it,en,fr            # the tested languages (default)
    python frankenstein2/scripts/setup_models.py --langs de,es --dry-run    # what would be fetched, and its size

    python frankenstein2/scripts/setup_models.py --llm moe                   # the MoE brain instead of Qwen3.5-4B

What it fetches:
  ears    nineninesix/tyger                 Nemotron 3.5 ASR streaming 0.6B + turn-taking GRU (OpenMDW-1.1)
  brain   --llm qwen4b (default)            Qwen/Qwen3.5-4B (Apache-2.0)
          --llm moe                         nvidia/Qwen3.6-35B-A3B-NVFP4 (Apache-2.0; FP4 weights, Blackwell GPUs)
          --llm moe-awq                     cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit (Apache-2.0; int4, any recent GPU)
          --llm nano                        nvidia/NVIDIA-Nemotron-Nano-9B-v2 (NVIDIA open model licence; legacy)
  talker  --talker packages (default):
            kyutai/pocket-tts               languages/english_2026-09_24l/model.safetensors: the base checkpoint, whose
                                            Mimi codec and voice-cloning encoder every language shares (CC-BY-4.0,
                                            gated: accept the terms on the model page and set HF_TOKEN)
            <packages repo>                 <code>/<family>/<precision>/: each language's delta against the English
                                            base of its family (en/<family>/base), int4 by default
                                            (frankenstein2/scripts/build_packages.py builds them)
          --talker full:
            kyutai/pocket-tts               languages/<lang>/model.safetensors per language (gated, as above)
            kyutai/pocket-tts-without-voice-cloning   languages/<lang>/tokenizer.* (not gated)
  voice   kyutai/tts-voices                 one reference WAV for cloning (default alba-mackenna/casual.wav,
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
BASE = "english_2026-09_24l"
PACKAGES_REPO = os.environ.get("F2_PACKAGES_REPO", "gabrione/pocket-tts-deltas")
LLMS = {"qwen4b": ("Qwen/Qwen3.5-4B", "qwen"), "moe": ("nvidia/Qwen3.6-35B-A3B-NVFP4", "qwen"),
        "moe-awq": ("cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit", "qwen"),
        "nano": ("nvidia/NVIDIA-Nemotron-Nano-9B-v2", "nano")}
POCKET_TOK = "kyutai/pocket-tts-without-voice-cloning"
VOICES = "kyutai/tts-voices"
LLM_DEFAULT = "qwen4b"


def llm_repo(llm: str) -> tuple:
    """--llm alias or repository -> (repo, backend)."""
    if llm in LLMS:
        return LLMS[llm]
    return llm, ("nano" if "nemotron-nano" in llm.lower() else "qwen")


def pick_families(files: list, langs: list, families=("24l", "6l"), precision: str = "int4") -> dict:
    """code -> the first family the packages repo has for it at ``precision`` (English: its base)."""
    have = set(files)
    out = {}
    for c in langs:
        for fam in families:
            prec = "base" if c == "en" else precision
            if f"{c}/{fam}/{prec}/manifest.json" in have:
                out[c] = fam
                break
    return out


def plan(langs: list, llm: str, voice: str, talker: str = "packages", packages_repo: str = PACKAGES_REPO,
         precision: str = "int4", families: dict | None = None) -> list:
    """[(repo, allow_patterns or None = whole repo, role)]. ``families``: code -> family (pick_families); without
    it every family of each language is fetched."""
    repo, _ = llm_repo(llm)
    steps = [(EARS, None, "ears"),
             (repo, ["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja", "*.py", "LICENSE*"], "llm")]
    if talker == "packages":
        fams = families or {}
        pats = [f"{c}/language.json" for c in langs] + ["en/language.json", "README.md", "LICENSE.md"]
        for c in langs:
            for fam in ([fams[c]] if c in fams else ["24l", "6l"]):
                pats.append(f"{c}/{fam}/{'base' if c == 'en' else precision}/*")
        for fam in sorted(set(fams.values()) if fams else {"24l", "6l"}):
            pats.append(f"en/{fam}/base/*")                      # the family's shared base backbone
        steps += [(POCKET_CLONE, [f"languages/{BASE}/model.safetensors"], "talker Mimi + voice cloning (gated)"),
                  (packages_repo, sorted(set(pats)), "talker language packages")]
    else:
        steps += [(POCKET_CLONE, [f"languages/{POCKET[x]}/model.safetensors" for x in langs], "talker weights (gated)"),
                  (POCKET_TOK, [f"languages/{POCKET[x]}/tokenizer*" for x in langs], "talker tokenizers")]
    return steps + [(VOICES, [voice], "reference voice")]


def check_langs(raw: str, talker: str = "packages") -> list:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from frankenstein2 import langs as LG
    codes = [x.strip().lower() for x in raw.split(",") if x.strip()]
    ok = LG.known() if talker == "packages" else list(POCKET)
    bad = [x for x in codes if x not in ok]
    if bad:
        sys.exit(f"no talker language for {bad}; available: {sorted(ok)} (docs/frankenstein2-languages.md: a new "
                 "language is a folder in the packages layout)")
    return codes


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
    ap.add_argument("--llm", default=LLM_DEFAULT, help=f"{' | '.join(LLMS)} (or a repository id)")
    ap.add_argument("--talker", default="packages", choices=["packages", "full"],
                    help="packages = base checkpoint + per-language 4-bit packages (default); full = one full "
                         "checkpoint per language")
    ap.add_argument("--packages-repo", default=PACKAGES_REPO, help="Hugging Face repo with the language packages")
    ap.add_argument("--precision", default="int4", choices=["int4", "int8", "bf16"],
                    help="which delta of each language to fetch (int4 is the engine's default)")
    ap.add_argument("--voice", default="alba-mackenna/casual.wav", help=f"reference WAV inside {VOICES}")
    ap.add_argument("--env-file", default="", help="where to write the F2_* paths (default <hf-home>/f2.env)")
    ap.add_argument("--dry-run", action="store_true", help="only list what would be downloaded and its size")
    a = ap.parse_args(argv)
    langs = check_langs(a.langs, a.talker)
    untested = [x for x in langs if x not in TESTED]
    if untested:
        print(f"NOTE: {untested} are not tested end to end yet (docs/frankenstein2-languages.md).")
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError:
        sys.exit("pip install huggingface_hub (or run inside the engine's environment)")
    token = os.environ.get("HF_TOKEN") or None
    cache = Path(a.hf_home) / "hub"
    fams = None
    if a.talker == "packages":
        try:
            fams = pick_families(HfApi().list_repo_files(a.packages_repo, token=token), langs, precision=a.precision)
            missing = [c for c in langs if c not in fams]
            if missing:
                print(f"NOTE: {a.packages_repo} has no {a.precision} package for {missing}")
        except Exception as e:  # noqa: BLE001 - listing failed: fetch every family
            print(f"NOTE: could not list {a.packages_repo} ({type(e).__name__}); fetching every family")
    steps = plan(langs, a.llm, a.voice, a.talker, a.packages_repo, a.precision, fams)
    llm, backend = llm_repo(a.llm)
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
    env = {"F2_EARS_BUNDLE": paths[EARS], "F2_LLM": a.llm, "F2_LLM_BACKEND": backend,
           ("F2_NANO_DIR" if backend == "nano" else "F2_QWEN_MODEL"): paths[llm],
           "F2_POCKET_WEIGHTS": paths[POCKET_CLONE], "F2_VOICE": str(Path(paths[VOICES]) / a.voice),
           "F2_LANGS": ",".join(langs), "HF_HOME": a.hf_home}
    if a.talker == "packages":
        env["F2_POCKET_PACKAGES"] = str(Path(paths[a.packages_repo]))
        env["F2_POCKET_PRECISION"] = a.precision
        env["F2_POCKET_BASE"] = str(Path(paths[POCKET_CLONE]) / "languages" / BASE / "model.safetensors")
    else:
        env["F2_POCKET_TOKENIZERS"] = paths[POCKET_TOK]
    out = Path(a.env_file or Path(a.hf_home) / "f2.env")
    out.write_text("".join(f"export {k}={v}\n" for k, v in env.items()))
    print(f"[setup] wrote {out}\n\nNext:\n  source {out}\n  make f2-serve        # or: make webui-gpu ENGINE=frankenstein2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
