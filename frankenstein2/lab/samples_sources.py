"""SOURCES.md for the web UI's samples folder: per language, which voices spoke (agent and demo caller), where they
come from and under which licence, and which talker model made the audio.

    python frankenstein2/lab/samples_sources.py --voice-map MAP.json --sources REFS/sources.json --out SAMPLES_DIR
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

KYUTAI = "Kyutai Pocket TTS predefined voice (kyutai/pocket-tts-without-voice-cloning, CC-BY-4.0)"


def describe(spec: str, sources: dict) -> tuple:
    """-> (voice, source, licence) of a voice-map entry. Cloned voices are named by kind, not by file: their file
    names are people's names or handles, which neither licence asks for (CC0; CC-BY is met by the source link)."""
    if spec.startswith("native:"):
        return spec.split(":", 1)[1], KYUTAI, "CC-BY-4.0"
    p = Path(spec)
    s = sources.get(f"{p.parent.name}/{p.stem}") or {}
    src = s.get("source", "?").split("/blob/")[0]
    if "/voice-donations/" in s.get("source", ""):
        return "a donated voice", src + " (voice-donations)", s.get("licence", "?")
    return "a published voice prompt", src, s.get("licence", "?")


def render(vmap: dict, sources: dict, talkers: dict) -> str:
    rows = []
    for lang in sorted(vmap):
        for role in ("default", "caller"):
            if role in vmap[lang]:
                name, src, lic = describe(vmap[lang][role], sources)
                rows.append(f"| {lang} | {'agent' if role == 'default' else 'caller'} | {name} | {src} | {lic} |")
    tal = "\n".join(f"| {k} | {v['source']} | {v['licence']} |" for k, v in sorted(talkers.items()))
    return f"""# Demo samples: sources and licences

Recorded with the Frankenstein v2 engine (frankenstein2/lab/demo_dialogue_check.py, packed by pack_samples.py):
`lines/` are the simulated caller's lines, `dialogues/` whole demo dialogues (caller and agent). Every voice is either
a predefined voice of the talker model or a reference voice offered for speech synthesis, cloned by the talker; no
voice was cloned from a recording made for another purpose.

The audio of a language is a derivative of its talker model and of the voices below: it is shared under the same
terms (CC-BY-4.0; CC-BY-SA-4.0 where a source is share-alike, such as the Estonian voices and talker).

## Voices

| lang | role | voice | source | licence |
|---|---|---|---|---|
{chr(10).join(rows)}

## Talker models

| lang | model | licence |
|---|---|---|
{tal}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice-map", required=True)
    ap.add_argument("--sources", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from frankenstein2.scripts.build_packages import SOURCES
    talkers = {}
    for s in SOURCES:
        talkers.setdefault(s["code"], {"source": f"https://huggingface.co/{s['repo']}", "licence": s["licence"]})
    vmap = json.loads(Path(a.voice_map).read_text())
    md = render(vmap, json.loads(Path(a.sources).read_text()), talkers)
    (Path(a.out) / "SOURCES.md").write_text(md)
    print("SOURCES", Path(a.out) / "SOURCES.md", len(vmap), "languages")


if __name__ == "__main__":
    main()
