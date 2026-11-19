"""End-to-end conversation per language from the recorded demo samples (opt-in: ``make test-f2-e2e``, i.e.
``pytest frankenstein2/tests/gpu --gpu`` with F2_E2E=1; GPU, the real models, several minutes).

The engine runs in this process as it is served (one MultiServer, the talker's language folder with a small VRAM
cache, so walking through the languages loads and evicts their delta packages). For each language of the samples'
manifest (webui/f2/samples, or F2_SAMPLES), the recorded caller lines are streamed in real time as a session's input,
exactly as a microphone would. F2_E2E_PARALLEL languages talk at once (default 3: the engine batches their sessions,
cross-language talker steps included, and the VRAM cache must swap packages while others speak). Per language: the
ASR heard the first line, the agent answered every line with audio, and the catalogue question led to a tool call. A
summary line (E2E) per language gives the talker's loads and waits."""

import asyncio
import json
import os
import threading
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("no CUDA device", allow_module_level=True)
if os.environ.get("F2_E2E", "0") != "1":
    pytest.skip("opt-in: F2_E2E=1 (make test-f2-e2e)", allow_module_level=True)

ROOT = Path(__file__).resolve().parents[3]
SAMPLES = Path(os.environ.get("F2_SAMPLES", ROOT / "webui" / "f2" / "samples"))
MAN = json.loads((SAMPLES / "manifest.json").read_text()) if (SAMPLES / "manifest.json").exists() else None
if MAN is None:
    pytest.skip(f"no samples manifest in {SAMPLES}", allow_module_level=True)


def _lines_by_lang():
    out: dict = {}
    for text, f in MAN["lines"].items():
        lang, name = f.split("/")[1], Path(f).stem          # lines/<lang>/<scenario>_<i>.ogg
        sc, i = name.rsplit("_", 1)
        if sc == "store":
            out.setdefault(lang, []).append((int(i), text, SAMPLES / f))
    return {k: [(t, p) for _, t, p in sorted(v)] for k, v in out.items()}


LINES = _lines_by_lang()
LANGS = [x for x in os.environ.get("F2_E2E_LANGS", "").split(",") if x] or sorted(LINES)
PARALLEL = int(os.environ.get("F2_E2E_PARALLEL", "3"))
_ENGINE: dict = {}
_RESULTS: dict = {}


def engine():
    if _ENGINE:
        return _ENGINE
    import argparse
    import sys
    sys.path.insert(0, str(ROOT))
    from frankenstein2.server import build as B
    from frankenstein2.server.engine import Resources, make_pipeline
    from frankenstein2.server.multi import MultiServer
    out = Path(os.environ.get("F2_E2E_OUT", "/tmp/f2-e2e"))
    out.mkdir(parents=True, exist_ok=True)
    ap = argparse.ArgumentParser()
    B.add_args(ap)
    a = ap.parse_args(["--talker-vram-langs", os.environ.get("F2_E2E_VRAM_LANGS", "3"), "--pocket-preload", ""]
                      + os.environ.get("F2_E2E_ARGS", "").split())      # e.g. --lab-scripts DIR on a lab box
    B.resolve_llm(a)
    ears = B.start_ears(a, out)
    spec = B.tenant(a)
    talker = B.start_pocket(a, out)
    qproc = B.start_qwen(a, out)
    res = Resources(ears=ears, make_llm=B.qwen_factory(a, qproc, spec), make_talker=B.talker_factory(a, talker),
                    voices=B.voices(a), voice_map=B.voice_map(a), second_pass=getattr(a, "asr_second_pass", "on") == "on", ears_thr=B.thresholds(a),
                    trace_dir=str(out))
    if getattr(talker, "catalog", None) is not None:
        res.languages = talker.catalog.languages
    srv = MultiServer(lambda sid, lang, voice, prompt, sink, **kw: make_pipeline(res, sid, lang, voice, prompt, sink,
                                                                                 **kw), max_sessions=PARALLEL + 1)
    ready = threading.Event()
    port = int(os.environ.get("F2_E2E_PORT", "8796"))
    threading.Thread(target=lambda: asyncio.run(srv.serve("127.0.0.1", port, ready=ready.set)), daemon=True).start()
    ready.wait(60)
    _ENGINE.update(url=f"ws://127.0.0.1:{port}/v1/session", talker=talker, out=out)
    return _ENGINE


def results():
    """Every language once, PARALLEL sessions at a time (computed on the first test, shared by all)."""
    if _RESULTS:
        return _RESULTS
    import soundfile as sf
    from frankenstein2.lab.demo_dialogue_check import run_many
    e = engine()
    rec, jobs = {}, []
    for lang in LANGS:
        if lang not in LINES:
            continue
        rec[lang] = {}
        for text, path in LINES[lang]:
            x, sr = sf.read(path, dtype="float32")
            assert sr == 24000, (path, sr)
            rec[lang][text] = x if x.ndim == 1 else x[:, 0]
        jobs.append((lang, [t for t, _ in LINES[lang]], None))
    out = e["out"] / "e2e"
    out.mkdir(parents=True, exist_ok=True)

    def report(r):
        st = e["talker"].langs.stats()
        print("E2E", json.dumps({"lang": r["lang"], "pass": r["pass"], "tools": r["tools"],
                                 "lines": [(x.get("ratio"), x.get("answered")) for x in r["lines"]],
                                 "wall_s": r.get("wall_s"), "errors": r["errors"],
                                 "second_pass": r.get("second_pass", []),
                                 "talker_loads": st.get("loads"), "evictions": st.get("evictions"),
                                 "wait_ms_max": st.get("wait_ms_max")}, ensure_ascii=False), flush=True)
    _RESULTS.update(asyncio.run(run_many(e["url"], jobs, out, PARALLEL, recorded=rec, on_result=report)))
    (out / "demo_dialogue.json").write_text(json.dumps(_RESULTS, indent=1, ensure_ascii=False))   # pack_samples.py
    return _RESULTS


@pytest.mark.gpu
@pytest.mark.parametrize("lang", LANGS)
def test_language_end_to_end_from_recorded_caller(lang):
    if lang not in LINES:
        pytest.skip(f"no recorded lines for {lang}")
    r = results()[f"store:{lang}"]
    hard = 0.6 if lang in ("ja", "ko", "hi") else 0.7
    assert not r["errors"], r["errors"]
    assert r["lines"][0].get("ratio", 0) >= hard, ("ASR", r["lines"][0])
    assert all(x.get("answered") for x in r["lines"]), ("answers", r["lines"])
    assert r["tools"], "no tool call"
