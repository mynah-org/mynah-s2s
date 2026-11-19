"""Pack demo_dialogue_check.py runs into the web UI's samples folder (webui/f2/samples): the caller lines (the page
plays them instead of asking the engine to speak them) and one recorded dialogue per language (listen without a GPU),
with manifest.json. Only dialogues that passed their checks are kept; a later RUN_DIR's dialogue replaces an earlier
one. Everything is re-encoded as small Opus files: caller lines mono at about 19 kbps (they are also the e2e test's
input, so not lower), dialogues mono at about 10 kbps (libsndfile's compression level, 0..1 = high..low bitrate).

    python frankenstein2/lab/pack_samples.py RUN_DIR [RUN_DIR ...] --out webui/f2/samples
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def reencode(src: Path, dst: Path, level: float, mono: bool = False) -> None:
    """Opus re-encode at libsndfile's compression level (0.95 ~ 19 kbps, 0.97 ~ 15 kbps mono speech)."""
    import soundfile as sf
    x, sr = sf.read(src, dtype="float32")
    if mono and x.ndim > 1:
        x = x.mean(1)
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(dst, x, sr, format="OGG", subtype="OPUS", compression_level=level)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--line-level", type=float, default=0.95)
    ap.add_argument("--dialogue-level", type=float, default=0.97)
    a = ap.parse_args(argv)
    out = Path(a.out)
    (out / "lines").mkdir(parents=True, exist_ok=True)
    (out / "dialogues").mkdir(parents=True, exist_ok=True)
    mf = out / "manifest.json"
    man = json.loads(mf.read_text()) if mf.exists() else {"lines": {}, "dialogues": {}, "verified": {}}
    for run in map(Path, a.runs):
        res = json.loads((run / "demo_dialogue.json").read_text())
        for key, r in res.items():
            sc, lang = r.get("scenario", "store"), r["lang"]
            tag = lang if sc == "store" else f"{sc}_{lang}"
            for i, line in enumerate(r["lines"]):
                src = run / f"{tag}_line{i}.ogg"
                if src.exists() and line.get("text"):
                    dst = f"lines/{lang}/{sc}_{i}.ogg"
                    reencode(src, out / dst, a.line_level)
                    man["lines"][line["text"]] = dst
            v = man["verified"].setdefault(lang, {})
            v[sc] = bool(r.get("pass")) or bool(v.get(sc))       # passed in any of the runs
            src = run / f"{tag}_dialogue.ogg"
            if r.get("pass") and src.exists():
                dst = f"dialogues/{lang}_{sc}.ogg"
                reencode(src, out / dst, a.dialogue_level, mono=True)
                man["dialogues"].setdefault(lang, {})[sc] = dst
    mf.write_text(json.dumps(man, indent=1, ensure_ascii=False, sort_keys=True))
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print("PACKED", len(man["lines"]), "lines", sum(len(v) for v in man["dialogues"].values()), "dialogues",
          f"{size / 1e6:.2f} MB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
