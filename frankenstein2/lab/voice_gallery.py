"""Listen to the talker's voices in one language: Kyutai's predefined voices of that language (native:<name>) and
the cloned reference voices, each saying the same sentences. Writes WAVs and an index.html to compare them.

    python frankenstein2/lab/voice_gallery.py --pocket-packages DIR --pocket-base CKPT --pocket-voices SNAPSHOT \\
        --lang it --clone default=ref.wav --clone caller=ref2.wav --out DIR
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

TEXT = {"it": ["Buongiorno, sono l'assistente di Mynah Home. Come posso aiutarla oggi?",
               "Il nostro divano a tre posti si chiama Fjordby e costa quattrocentonovantanove euro."],
        "fr": ["Bonjour, je suis l'assistant de Mynah Home. Comment puis-je vous aider ?",
               "Notre canapé trois places s'appelle Fjordby et coûte quatre cent quatre-vingt-dix-neuf euros."],
        "en": ["Hello, this is the Mynah Home assistant. How can I help you today?",
               "Our three-seat sofa is called Fjordby and costs four hundred ninety-nine euros."]}


def main():  # pragma: no cover - model run
    import numpy as np
    import soundfile as sf
    import torch
    from frankenstein2.tts.catalog import PackageCatalog
    from frankenstein2.tts.pocket_own import BaseBackbone, PocketOwn, _pocket_swap
    from frankenstein2 import langs as LG
    _pocket_swap()
    from pocket_swap.package import load_package
    ap = argparse.ArgumentParser()
    ap.add_argument("--pocket-packages", required=True)
    ap.add_argument("--pocket-base", required=True)
    ap.add_argument("--pocket-voices", required=True)
    ap.add_argument("--pocket-precision", default="int4")
    ap.add_argument("--lang", default="it")
    ap.add_argument("--clone", action="append", default=[], help="NAME=wav (cloned voices to compare)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    cat = PackageCatalog(a.pocket_packages, a.pocket_precision, families=("24l",))
    base = BaseBackbone.from_package(load_package(cat.base_dir("24l")), device="cuda")
    m = PocketOwn.from_package(load_package(cat.path(a.lang)), a.lang, a.pocket_base, device="cuda", base=base)
    folder = LG.get(a.lang).pocket
    emb = Path(a.pocket_voices) / "languages" / folder / "embeddings"
    voices = [(f"native:{p.stem}", lambda p=p: m.voice_from_state(p)) for p in sorted(emb.glob("*.safetensors"))]
    voices += [(f"clone:{n}", lambda w=w: m.voice_from_file(w)) for n, w in (x.split("=", 1) for x in a.clone)]
    rows = []
    for name, make in voices:
        torch.manual_seed(0)
        try:
            v = make()
            with torch.no_grad():
                y = torch.cat([torch.cat(list(m.stream(v, t))) for t in TEXT[a.lang]]).float().cpu().numpy()
            f = f"{name.replace(':', '_')}.wav"
            sf.write(out / f, y, 24000, subtype="PCM_16")
            rows.append({"voice": name, "file": f, "s": round(len(y) / 24000, 1)})
            print("VOICE", json.dumps(rows[-1]), flush=True)
        except Exception as e:  # noqa: BLE001
            print("VOICE_FAILED", name, f"{type(e).__name__}: {e}"[:200], flush=True)
    items = "\n".join(f'<li><b>{html.escape(r["voice"])}</b> <small>{r["s"]} s</small><br>'
                      f'<audio controls preload="none" src="{html.escape(r["file"])}"></audio></li>' for r in rows)
    (out / "index.html").write_text(f"""<!doctype html><meta charset=utf-8><title>Voices · {a.lang}</title>
<style>body{{font:15px system-ui;background:#111;color:#eee;max-width:900px;margin:30px auto;padding:0 16px}}
li{{margin:12px 0;list-style:none}}small{{color:#888}}audio{{width:100%}}</style>
<h2>Talker voices · {a.lang} ({a.pocket_precision} package)</h2>
<p>native:* = Kyutai's predefined voices of this language; clone:* = cloned from a reference WAV.</p>
<p>{html.escape(" / ".join(TEXT[a.lang]))}</p><ul>{items}</ul>""")
    print("GALLERY", out / "index.html", len(rows), flush=True)


if __name__ == "__main__":
    main()
