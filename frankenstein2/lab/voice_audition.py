"""Which voices does the ASR understand best, per language? Every candidate voice of a language says the demo
caller's lines and one of the agent's phrases; the engine's own ASR (the ears) transcribes them; candidates are ranked
by character match (mean, then worst line). The best becomes the agent's voice ("default"), the next one the demo
caller's ("caller"): a voice map for the server (--voice-map). WAVs, scores and an index.html to listen and override.

Candidates: Kyutai's predefined voices of the language (native:<name>, --pocket-voices), the references of --refs
(frankenstein2/lab/voice_refs.py: DIR/<lang>/*.wav), the shared references of --shared-refs (every language, e.g.
DIR/_donated), and the server's own --voice references as a baseline (ranked, never picked while another candidate
exists). --exclude drops references by name prefix (fleurs_: lab-only voices). With --screen N every candidate first
says one line only, and the N best say them all (many candidates, little GPU time).

    python frankenstein2/lab/voice_audition.py --pocket-voices SNAP --refs DIR [--langs ko,tr] --out DIR \\
        [server flags as for serve.py]
"""

from __future__ import annotations

import argparse
import html
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "webui"))


def rank(rows: list) -> list:
    """Best first: mean character match, then the worst line, then the shorter audio (fewer stalls)."""
    return sorted(rows, key=lambda r: (-r["mean"], -r["min"], r["s"]))


def picks(rows: list, n: int = 2) -> list:
    """The n best distinct candidates, native ones (not the --voice baseline) when there are any."""
    good = [r for r in rank(rows) if not r["baseline"]] or rank(rows)
    return [r["voice"] for r in good[:n]]


def main():  # pragma: no cover - model run
    import numpy as np
    import soundfile as sf
    import torch
    from scipy.signal import resample_poly
    from frankenstein2 import langs as LG
    from frankenstein2.ears.client import f32b64
    from frankenstein2.lab.lang_suite import char_ratio
    from frankenstein2.server import build as B
    from server import DEMO_SCRIPT
    ap = argparse.ArgumentParser()
    B.add_args(ap)
    ap.add_argument("--langs", default="", help="codes (default: every language of the demo script)")
    ap.add_argument("--refs", default="", help="native-speaker references: DIR/<lang>/*.wav")
    ap.add_argument("--max-native", type=int, default=0, help="at most N predefined voices per language (0 = all)")
    ap.add_argument("--shared-refs", default="", help="references tried in every language: DIR/*.wav")
    ap.add_argument("--exclude", default="", help="comma list of reference name prefixes to leave out (fleurs_)")
    ap.add_argument("--screen", type=int, default=0, help="first pass on one line, then all lines for the N best")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    a.pocket_preload = ""
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    ears = B.start_ears(a, out)
    talker = B.start_pocket(a, out)
    sched = a._single["sched"]
    base_voices = B.voices(a)
    langs = [x for x in a.langs.split(",") if x] or list(DEMO_SCRIPT)
    report = {}
    vmap = {}
    for lang in langs:
        lines = list(DEMO_SCRIPT.get(lang, [])) + [LG.get(lang).phrase("holding")]
        lines = [x for x in lines if x]
        cands = []                                         # (name, spec, baseline)
        folder = LG.get(lang).pocket or ""
        if a.pocket_voices and folder:
            L = sched.call(lambda: talker.langs.ensure(talker.names(lang, None), 600, talker._in_use), timeout=900)
            fold = folder if L.model.n_layers == 24 else folder.replace("_24l", "")
            emb = Path(a.pocket_voices) / "languages" / fold / "embeddings"
            nat = sorted(emb.glob("*.safetensors"))
            if a.max_native:
                nat = nat[: a.max_native]
            cands += [(f"native:{p.stem}", f"native:{p.stem}", False) for p in nat]
        excl = tuple(x for x in a.exclude.split(",") if x)
        if a.refs and (Path(a.refs) / lang).is_dir():
            cands += [(f"ref:{p.stem}", str(p.resolve()), False) for p in sorted((Path(a.refs) / lang).glob("*.wav"))
                      if not (excl and p.stem.startswith(excl))]
        if a.shared_refs:
            cands += [(f"shared:{p.stem}", str(p.resolve()), False) for p in sorted(Path(a.shared_refs).glob("*.wav"))
                      if not (excl and p.stem.startswith(excl))]
        cands += [(f"server:{n}", w, True) for n, w in base_voices.items()]
        d = out / lang
        d.mkdir()
        if a.screen and len(cands) > a.screen:
            scored = []
            for name, spec, baseline in cands:
                def one(spec=spec):
                    L = talker.langs.ensure(talker.names(lang, None), 600, talker._in_use)
                    v = talker._voice(L, spec)
                    torch.manual_seed(0)
                    with torch.no_grad():
                        y = torch.cat(list(L.model.stream(v, lines[0]))).float().cpu().numpy()
                    L.voices.pop(spec, None)
                    return y
                try:
                    y = sched.call(one, timeout=600)
                except Exception as e:  # noqa: BLE001
                    print("SCREEN_FAILED", lang, name, f"{type(e).__name__}: {e}"[:160], flush=True)
                    continue
                h = ears.request({"op": "transcribe", "id": f"sc_{lang}", "lang": lang,
                                  "pcm": f32b64(resample_poly(y, 2, 3).astype(np.float32))}, "transcript", 120)["text"]
                scored.append((char_ratio(h, lines[0]), -len(y), name, spec, baseline))
            scored.sort(reverse=True)
            keep = {x[2] for x in scored[: a.screen]} | {n for n, _, b in cands if b}
            print("SCREEN", lang, len(cands), "->", len(keep), json.dumps([(x[2], x[0]) for x in scored[:a.screen]]),
                  flush=True)
            cands = [c for c in cands if c[0] in keep]
        rows = []
        for name, spec, baseline in cands:
            t0 = time.monotonic()

            def job(spec=spec):
                L = talker.langs.ensure(talker.names(lang, None), 600, talker._in_use)
                v = talker._voice(L, spec)
                ys = []
                for i, t in enumerate(lines):
                    torch.manual_seed(i)
                    with torch.no_grad():
                        ys.append(torch.cat(list(L.model.stream(v, t))).float().cpu().numpy())
                L.voices.pop(spec, None)
                return ys
            try:
                ys = sched.call(job, timeout=900)
            except Exception as e:  # noqa: BLE001
                print("AUDITION_FAILED", lang, name, f"{type(e).__name__}: {e}"[:200], flush=True)
                continue
            heard, ratios = [], []
            for i, (t, y) in enumerate(zip(lines, ys)):
                x16 = resample_poly(y, 2, 3).astype(np.float32)
                h = ears.request({"op": "transcribe", "id": f"au_{lang}_{i}", "lang": lang, "pcm": f32b64(x16)},
                                 "transcript", 120)["text"]
                heard.append(h)
                ratios.append(char_ratio(h, t))
            gap = np.zeros(int(0.4 * 24000), np.float32)
            y = np.concatenate([np.concatenate([x, gap]) for x in ys])
            f = f"{name.replace(':', '_')}.wav"
            sf.write(d / f, y, 24000, subtype="PCM_16")
            r = {"voice": name, "spec": spec, "baseline": baseline, "file": f"{lang}/{f}",
                 "mean": round(float(np.mean(ratios)), 3), "min": round(float(np.min(ratios)), 3),
                 "s": round(len(y) / 24000, 1), "ratios": ratios, "heard": heard,
                 "took_s": round(time.monotonic() - t0, 1)}
            rows.append(r)
            print("AUDITION", json.dumps({k: r[k] for k in ("voice", "mean", "min", "s", "took_s")}), lang,
                  flush=True)
        if not rows:
            continue
        best = picks(rows)
        spec_of = {r["voice"]: r["spec"] for r in rows}
        vmap[lang] = {"default": spec_of[best[0]], **({"caller": spec_of[best[1]]} if len(best) > 1 else {})}
        report[lang] = {"lines": lines, "rows": rank(rows), "picks": best}
        print("PICK", lang, json.dumps(best), json.dumps([r["mean"] for r in rank(rows)[:3]]), flush=True)
        (out / "audition.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
        (out / "voice_map.json").write_text(json.dumps(vmap, indent=1, ensure_ascii=False))
    sec = []
    for lang, r in report.items():
        items = "".join(
            f'<li class="{"pick" if x["voice"] in r["picks"] else ""}"><b>{html.escape(x["voice"])}</b> '
            f'<small>match {x["mean"]:.2f} (worst {x["min"]:.2f}) · {x["s"]} s'
            f'{" · baseline" if x["baseline"] else ""}'
            f'{" · AGENT" if x["voice"] == r["picks"][0] else ""}'
            f'{" · CALLER" if len(r["picks"]) > 1 and x["voice"] == r["picks"][1] else ""}</small><br>'
            f'<audio controls preload="none" src="{html.escape(x["file"])}"></audio>'
            f'<div class="heard">{html.escape(" / ".join(x["heard"]))}</div></li>' for x in r["rows"])
        sec.append(f'<h3 id="{lang}">{html.escape(LG.get(lang).name)} ({lang})</h3>'
                   f'<p class="lines">{html.escape(" / ".join(r["lines"]))}</p><ul>{items}</ul>')
    nav = " · ".join(f'<a href="#{x}">{x}</a>' for x in report)
    (out / "index.html").write_text(f"""<!doctype html><meta charset=utf-8><title>Voice audition</title>
<style>body{{font:15px system-ui;background:#111;color:#eee;max-width:960px;margin:30px auto;padding:0 16px}}
a{{color:#8cf}}li{{margin:12px 0;list-style:none;padding:6px 10px;border-radius:8px}}li.pick{{background:#1d2a1d}}
small{{color:#999}}audio{{width:100%}}.heard,.lines{{color:#aaa;font-size:13px}}</style>
<h2>Talker voices per language, ranked by how well the engine's ASR understands them</h2>
<p>{nav}</p><p>native:* = Kyutai's predefined voices; ref:* = cloned from a native speaker's reference
(frankenstein2/lab/voice_refs.py); server:* = the server's own reference voices (baseline). Green = picked
(agent, caller).</p>{"".join(sec)}""")
    print("AUDITION_DONE", out / "index.html", json.dumps({k: v["picks"] for k, v in report.items()}), flush=True)


if __name__ == "__main__":
    main()
