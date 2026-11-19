"""Opt-in language suite: does each language work, part by part and end to end? (GPU, the real models)

    make f2-lang-suite F2_LANGS=de,ko,cs        # or: python frankenstein2/lab/lang_suite.py --langs de,ko --out DIR

The engine is loaded once (one process, as served). For each language, three checks with independent judges:

  talker   the talker speaks a sentence; the ears (the engine's own ASR) must understand it (character match)
  llm      the LLM alone gets a catalogue question written in the language: it must call search_catalog and its
           reply must be in that language (script / stop-word check)
  e2e      the same question SPOKEN (synthesised by the talker) is streamed to a real session: the session must call
           the tool and answer with audio; the answer's text must be in the language and the ears must transcribe
           the answer audio

Report: one line per language (``LANG``), ``<out>/lang_suite.json``, and the WAVs of every e2e session. A few
minutes for three languages; nothing is downloaded (the languages come from the talker's folder).
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "frankenstein"))

QUESTION = {
    "it": "Buongiorno, avete un divano a tre posti? Quanto costa?",
    "en": "Hello, do you have a three-seat sofa? How much is it?",
    "fr": "Bonjour, avez-vous un canapé trois places ? Combien coûte-t-il ?",
    "de": "Guten Tag, haben Sie ein dreisitziges Sofa? Was kostet es?",
    "es": "Hola, ¿tienen un sofá de tres plazas? ¿Cuánto cuesta?",
    "pt": "Olá, têm um sofá de três lugares? Quanto custa?",
    "nl": "Hallo, hebben jullie een driezitsbank? Wat kost die?",
    "ko": "안녕하세요, 3인용 소파 있나요? 가격이 얼마예요?",
    "tr": "Merhaba, üç kişilik kanepeniz var mı? Fiyatı ne kadar?",
    "ja": "こんにちは、三人掛けのソファはありますか？おいくらですか？",
    "cs": "Dobrý den, máte trojmístnou pohovku? Kolik stojí?",
    "hi": "नमस्ते, क्या आपके पास तीन सीटों वाला सोफ़ा है? इसकी कीमत क्या है?",
    "pl": "Dzień dobry, czy macie trzyosobową sofę? Ile kosztuje?",
    "ru": "Здравствуйте, у вас есть трёхместный диван? Сколько он стоит?",
    "et": "Tere, kas teil on kolmekohaline diivan? Kui palju see maksab?",
}
SCRIPT = {"ko": r"[가-힣]", "ja": r"[぀-ヿ一-鿿]", "hi": r"[ऀ-ॿ]",
          "ru": r"[Ѐ-ӿ]"}
LETTERS = {"cs": r"[ěščřžýůťďň]", "tr": r"[ışğçöü]", "pl": r"[ąćęłńśźż]", "et": r"[õäöü]", "de": r"[äöüß]",
           "es": r"[ñáéíóú¿¡]", "pt": r"[ãõçâêô]", "fr": r"[éèêàçùœ]", "it": r"[àèéìòù]"}
STOP = {
    "it": "il la di che è un una per non con sono abbiamo costa euro ha le",
    "en": "the a is we have it for you and costs our this",
    "fr": "le la de est un une nous avons pour vous coûte les des",
    "de": "der die das ist ein eine wir haben für sie kostet und",
    "es": "el la de es un una tenemos por para usted cuesta los",
    "pt": "o a de é um uma temos para você custa não os",
    "nl": "de het een is we hebben voor u kost niet en",
    "tr": "bir ve bu için var mı fiyatı evet değil size",
    "cs": "je a na to máme za stojí ano se v",
    "pl": "jest i na to mamy za kosztuje tak się w",
    "et": "on ja see meil maksab jah kas ei teil",
}


def in_language(text: str, code: str) -> bool:
    """Script share for non-Latin languages; for Latin ones, more of its stop words than English ones."""
    t = (text or "").lower()
    if not t.strip():
        return False
    if code in SCRIPT:
        letters = re.findall(r"\w", t)
        return bool(letters) and len(re.findall(SCRIPT[code], t)) / len(letters) > 0.4
    words = re.findall(r"\w+", t)
    own = sum(w in set(STOP.get(code, "").split()) for w in words)
    eng = sum(w in set(STOP["en"].split()) for w in words)
    if code != "en" and own == 0 and LETTERS.get(code):         # few stop words in a short reply: its own letters
        own = len(re.findall(LETTERS[code], t))
    return own > 0 and (code == "en" or own > eng)


def char_ratio(a: str, b: str) -> float:
    a, b = "".join(re.findall(r"\w", a.lower())), "".join(re.findall(r"\w", b.lower()))
    return round(difflib.SequenceMatcher(None, a, b).ratio(), 3)


def main():  # pragma: no cover - model run
    import numpy as np
    import soundfile as sf
    import torch
    from scipy.signal import resample_poly
    from frankenstein2.server import build as B
    ap = argparse.ArgumentParser()
    B.add_args(ap)
    ap.add_argument("--langs", default="", help="codes (default: every language the talker folder serves)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--session-voice", default="")
    a = ap.parse_args()
    B.resolve_llm(a)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    from frankenstein2.ears.client import f32b64
    from frankenstein2.lab.ws_sessions import ClientSpec, run_clients
    from frankenstein2.server.engine import Resources, make_pipeline, served_languages
    from frankenstein2.server.multi import MultiServer
    a.pocket_preload = ""
    ears = B.start_ears(a, out)
    spec = B.tenant(a)
    talker = B.start_pocket(a, out)
    qproc = B.start_qwen(a, out)
    make_llm = B.qwen_factory(a, qproc, spec)
    res = Resources(ears=ears, make_llm=make_llm, make_talker=B.talker_factory(a, talker), voices=B.voices(a),
                    ears_thr=B.thresholds(a), trace_dir=str(out))
    if getattr(talker, "catalog", None) is not None:
        res.languages = talker.catalog.languages
    sched = a._single["sched"]
    langs = [x for x in a.langs.split(",") if x] or [c for c in served_languages(res) if c in QUESTION]
    voice_name = a.session_voice or next(iter(B.voices(a)), "")
    vwav = B.voices(a).get(voice_name, voice_name)

    def synth(lang, text):
        def job():
            L = talker.langs.ensure(talker.names(lang, None), 300, talker._in_use)
            v = talker._voice(L, vwav)
            with torch.no_grad():
                return torch.cat(list(L.model.stream(v, text))).float().cpu().numpy()
        return sched.call(job, timeout=600)

    def transcribe(x16, lang, name):
        t = ears.request({"op": "transcribe", "id": name, "lang": lang, "pcm": f32b64(x16.astype(np.float32))},
                         "transcript", 120)
        return t["text"]

    import asyncio
    import threading
    port = 8794
    srv = MultiServer(lambda sid, lang, voice, prompt, sink: make_pipeline(res, sid, lang, voice, prompt, sink),
                      max_sessions=2)
    ready = threading.Event()
    threading.Thread(target=lambda: asyncio.run(srv.serve("127.0.0.1", port, ready=ready.set)), daemon=True).start()
    ready.wait(30)
    report = {}
    for lang in langs:
        r = {"lang": lang}
        q = QUESTION.get(lang)
        try:
            t0 = time.monotonic()
            y = synth(lang, q)                                            # talker -> ears
            heard = transcribe(resample_poly(y, 2, 3), lang, f"t_{lang}")
            r["talker"] = {"heard": heard, "ratio": char_ratio(heard, q), "s": round(time.monotonic() - t0, 1)}
            r["talker"]["pass"] = r["talker"]["ratio"] >= (0.6 if lang in SCRIPT else 0.75)
            sf.write(out / f"{lang}_question.wav", y, 24000, subtype="PCM_16")
            llm = make_llm(f"llm_{lang}", lang, None)                     # the LLM alone
            llm.submit(1, [{"role": "user", "content": q}])
            evs, t_end = [], time.monotonic() + 120
            while time.monotonic() < t_end and not any(e["ev"] == "done" for e in evs):
                evs += llm.poll()
                time.sleep(0.02)
            llm.close()
            calls = [e["name"] for e in evs if e["ev"] == "tool"]
            reply = "".join(e["text"] for e in evs if e["ev"] == "text")
            r["llm"] = {"calls": calls, "reply": reply[:300], "in_language": in_language(reply, lang),
                        "pass": "search_catalog" in calls and in_language(reply, lang)}
            x16 = resample_poly(y, 2, 3).astype(np.float32)               # end to end: the spoken question
            x16 = np.concatenate([np.zeros(8000, np.float32), x16, np.zeros(4000, np.float32)])
            tl = [{"label": "U1", "start_k": 0, "end_k": len(x16) // 1280 - 1}]
            c = ClientSpec(f"e2e_{lang}", f"ws://127.0.0.1:{port}/v1/session", lang, x16, tl, voice=voice_name,
                           reactive=True, tail_s=4.0)
            e = run_clients([c], out / "e2e")[c.name]
            if "error" in e:
                raise RuntimeError(e["error"])
            said = " ".join(s for s in e.get("said", []) if s)
            wav = out / "e2e" / c.name / "agent_client_22k.wav"
            asr = ""
            if wav.exists():
                ya, sr = sf.read(wav, dtype="float32")
                asr = transcribe(resample_poly(ya, 320, 441), lang, f"a_{lang}")
            u = (e.get("utterances") or [{}])[0]
            r["e2e"] = {"tools": [t.get("name") for t in e.get("tools", [])], "said": said[:300], "asr": asr[:300],
                        "first_audio_s": u.get("first_audio_s"), "reply_s": u.get("reply_text_s"),
                        "in_language": in_language(said, lang)}
            r["e2e"]["pass"] = bool(r["e2e"]["tools"]) and r["e2e"]["in_language"] and len(asr) > 3
        except Exception as ex:  # noqa: BLE001 - one language failing does not stop the suite
            r["error"] = f"{type(ex).__name__}: {ex}"[:300]
        r["pass"] = all(r.get(k, {}).get("pass") for k in ("talker", "llm", "e2e"))
        report[lang] = r
        print("LANG", json.dumps(r, ensure_ascii=False), flush=True)
        (out / "lang_suite.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print("SUITE", " ".join(f"{k}={'PASS' if v['pass'] else 'FAIL'}"
                            f"[{''.join('T' if v.get(p, {}).get('pass') else '-' for p in ('talker', 'llm', 'e2e'))}]"
                            for k, v in report.items()), flush=True)
    print("TALKER", json.dumps(talker.langs.stats(), default=str)[:1500], flush=True)


if __name__ == "__main__":
    main()
