"""GPU smoke tests on the real models (opt-in: ``pytest frankenstein2/tests --gpu`` or ``make test-f2-gpu``).

Paths come from the env file written by ``frankenstein2/scripts/setup_models.py`` (F2_EARS_BUNDLE, F2_QWEN_MODEL,
F2_POCKET_WEIGHTS, F2_POCKET_TOKENIZERS, F2_VOICE, F2_LANGS). Each check is a round trip with an independent judge:
  * talker -> ears: Pocket speaks a sentence in each language; the ears' own ASR must hear most of its words;
  * brain: Qwen3.5 (own engine, int8) answers a tool request with a well-formed call of the right tool.
About 2-4 minutes on one GPU, ~12 GB of device memory.
"""

import os
import re
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("no CUDA device", allow_module_level=True)
NEED = ("F2_EARS_BUNDLE", "F2_QWEN_MODEL", "F2_POCKET_WEIGHTS", "F2_POCKET_TOKENIZERS", "F2_VOICE")
missing = [k for k in NEED if not os.environ.get(k)]
if missing:
    pytest.skip(f"{missing} not set: run frankenstein2/scripts/setup_models.py and source its env file",
                allow_module_level=True)

from frankenstein2.scripts.setup_models import POCKET  # noqa: E402

LANGS = [x for x in os.environ.get("F2_LANGS", "it,en,fr").split(",") if x in POCKET]
SAY = {"it": "Buongiorno, il suo ordine arriverà domani mattina.",
       "en": "Good morning, your order will arrive tomorrow morning.",
       "fr": "Bonjour, votre commande arrivera demain matin.",
       "de": "Guten Morgen, Ihre Bestellung kommt morgen früh an.",
       "es": "Buenos días, su pedido llegará mañana por la mañana.",
       "pt": "Bom dia, a sua encomenda chega amanhã de manhã.",
       "nl": "Goedemorgen, uw bestelling komt morgenochtend aan."}


def find(root, rel):
    """As the runtime: a snapshot dir, a snapshots dir, or any sibling snapshot of the repository."""
    r = Path(root)
    hits = sorted(r.glob(rel)) or sorted(r.glob(f"*/{rel}")) or sorted(r.parent.glob(f"*/{rel}"))
    assert hits, f"{rel} not under {root}"
    return hits[0]


def heard_ok(heard, ref, min_ratio=0.75):
    """Character-level match on letters only (compounds such as "morgenochtend" / "morgen ochtend" are the same)."""
    import difflib
    a, b = "".join(re.findall(r"\w", heard.lower())), "".join(re.findall(r"\w", ref.lower()))
    return difflib.SequenceMatcher(None, a, b).ratio() >= min_ratio


def words(s):
    return re.findall(r"\w+", s.lower())


@pytest.fixture(scope="module")
def ears():
    from frankenstein2.ears.own import OwnEars
    torch.set_grad_enabled(False)
    return OwnEars(os.environ["F2_EARS_BUNDLE"], device="cuda", slots=2, window_precision="bf16")


@pytest.mark.gpu
@pytest.mark.parametrize("lang", LANGS)
def test_pocket_speaks_and_the_ears_hear_it(ears, lang):
    import numpy as np
    from scipy.signal import resample_poly
    from frankenstein2.tts.pocket_own import PocketOwn
    name = POCKET[lang]
    try:
        w = find(os.environ["F2_POCKET_WEIGHTS"], f"languages/{name}/model.safetensors")
        tk = find(os.environ["F2_POCKET_TOKENIZERS"], f"languages/{name}/tokenizer.json")
    except AssertionError:
        pytest.skip(f"no full checkpoint for {name} (packages only)")
    m = PocketOwn(w, tk, lang=lang, device="cuda", dtype=torch.bfloat16, wq="w4")
    torch.manual_seed(0)
    voice = m.voice_from_file(os.environ["F2_VOICE"])
    x = torch.cat(list(m.stream(voice, SAY[lang]))).float().cpu().numpy()
    dur = len(x) / 24000
    assert np.isfinite(x).all() and 1.0 < dur < 10.0, dur
    heard = ears.transcribe(resample_poly(x, 2, 3).astype(np.float32), lang)   # unknown prompt -> auto
    assert heard_ok(heard, SAY[lang]), (heard, SAY[lang])
    del m
    torch.cuda.empty_cache()


@pytest.mark.gpu
def test_qwen_calls_the_right_tool():
    from frankenstein2.llm.qwen import parse_calls
    from frankenstein2.llm.qwen_own.engine import QwenOwn
    q = QwenOwn(os.environ["F2_QWEN_MODEL"], device="cuda", mode="int8", slots=2, max_len=4096)
    tools = [{"type": "function", "function": {
        "name": "find_customer", "description": "Find a customer by full name and postcode.",
        "parameters": {"type": "object", "properties": {"full_name": {"type": "string"},
                                                        "postcode": {"type": "string"}},
                       "required": ["full_name", "postcode"]}}}]
    msgs = [{"role": "system", "content": "You are a furniture store's phone assistant. Use the tools to look up "
                                          "customers before answering about their orders."},
            {"role": "user", "content": "Hi, I'm Mario Rossi, postcode 10121. Where is my order?"}]
    s = q.open()
    text = ""
    for ev in q.generate(s, q.chat_ids(msgs, tools), max_new=96):
        text += ev.get("text", "")
    q.close(s)
    calls = parse_calls(text)
    assert calls and calls[0]["name"] == "find_customer", text
    assert calls[0]["arguments"].get("postcode") == "10121" and "Rossi" in calls[0]["arguments"].get("full_name", "")


SAY.update({"ko": "안녕하세요, 주문하신 상품은 내일 아침에 도착합니다.",
            "tr": "Günaydın, siparişiniz yarın sabah gelecek.",
            "ja": "おはようございます。ご注文の商品は明日の朝に届きます。",
            "cs": "Dobré ráno, vaše objednávka dorazí zítra ráno.",
            "hi": "नमस्ते, आपका ऑर्डर कल सुबह पहुँच जाएगा।",
            "pl": "Dzień dobry, pana zamówienie dotrze jutro rano.",
            "ru": "Доброе утро, ваш заказ прибудет завтра утром.",
            "et": "Tere hommikust, teie tellimus saabub homme hommikul."})
PKG_ROOT = os.environ.get("F2_POCKET_PACKAGES", "")
PKG_LANGS = [x for x in os.environ.get("F2_PKG_LANGS", "").split(",") if x] or (
    sorted(p.name for p in Path(PKG_ROOT).iterdir() if p.is_dir() and "@" not in p.name) if PKG_ROOT and
    Path(PKG_ROOT).is_dir() else [])
_BASES: dict = {}


@pytest.mark.gpu
@pytest.mark.parametrize("lang", PKG_LANGS or ["none"])
def test_language_package_speaks_and_the_ears_hear_it(ears, lang):
    """A language from the talker folder (F2_POCKET_PACKAGES, F2_POCKET_PRECISION), as the engine loads it: its
    delta on the family's shared English base, Mimi from the base checkpoint. The ears must understand it."""
    import numpy as np
    from scipy.signal import resample_poly
    from frankenstein2.tts.catalog import PackageCatalog
    from frankenstein2.tts.pocket_own import BaseBackbone, PocketOwn, _pocket_swap
    _pocket_swap()                                   # puts the repository's pocket_swap on sys.path
    from pocket_swap.package import load_package
    if lang == "none":
        pytest.skip("no packages (F2_POCKET_PACKAGES)")
    cat = PackageCatalog(PKG_ROOT, os.environ.get("F2_POCKET_PRECISION", "int4"))
    d, fam = cat.path(lang), cat.family(lang)
    if d is None:
        pytest.skip(f"no {cat.precision} package for {lang}")
    if fam not in _BASES:
        _BASES[fam] = BaseBackbone.from_package(load_package(cat.base_dir(fam)), device="cuda")
    mimi = os.environ.get("F2_POCKET_BASE") or find(os.environ["F2_POCKET_WEIGHTS"],
                                                    f"languages/{POCKET['en']}/model.safetensors")
    m = PocketOwn.from_package(load_package(d), lang, mimi, device="cuda", base=_BASES[fam])
    torch.manual_seed(0)
    voice = m.voice_from_file(os.environ["F2_VOICE"])
    x = torch.cat(list(m.stream(voice, SAY[lang]))).float().cpu().numpy()
    assert np.isfinite(x).all() and 1.0 < len(x) / 24000 < 12.0, len(x) / 24000
    heard = ears.transcribe(resample_poly(x, 2, 3).astype(np.float32), lang)
    print(f"HEARD {lang} {fam} {d.name}: {heard!r} (said {SAY[lang]!r})")
    assert heard_ok(heard, SAY[lang], 0.6 if lang in ("ja", "ko", "hi") else 0.75), (heard, SAY[lang])


XL_LANGS = [x for x in os.environ.get("F2_XL_LANGS", "it,fr,de,es,pt,nl,ko,en").split(",") if x]


@pytest.mark.gpu
def test_cross_language_step_equals_each_language_alone():
    """Rows of several languages stepped together (pocket_own.step_many: one grouped kernel per backbone linear,
    one batched Mimi decode) give, row by row, the same latents as each language's own batch, and audio equal up to
    float rounding. Also prints the time per frame of both ways (TIMING)."""
    import time as _t
    from frankenstein2.tts.catalog import PackageCatalog
    from frankenstein2.tts.pocket_own import BaseBackbone, PocketBatch, PocketOwn, SlotPool, _pocket_swap, step_many
    _pocket_swap()
    from pocket_swap.package import load_package
    if not PKG_ROOT:
        pytest.skip("no packages (F2_POCKET_PACKAGES)")
    cat = PackageCatalog(PKG_ROOT, os.environ.get("F2_POCKET_PRECISION", "int4"), families=("24l",))
    langs = [x for x in XL_LANGS if cat.family(x) == "24l"]
    if len(langs) < 3:
        pytest.skip(f"fewer than 3 24l languages in {PKG_ROOT}")
    if "24l" not in _BASES:
        _BASES["24l"] = BaseBackbone.from_package(load_package(cat.base_dir("24l")), device="cuda")
    first = os.environ.get("F2_POCKET_BASE") or find(os.environ["F2_POCKET_WEIGHTS"],
                                                     f"languages/{POCKET['en']}/model.safetensors")
    models = {}
    for x in langs:
        models[x] = PocketOwn.from_package(load_package(cat.path(x)), x, next(iter(models.values()), None) or first,
                                           device="cuda", base=_BASES["24l"])
    fused, _ = _pocket_swap()
    dm = [m for m in models.values() if m.QB]                 # first the kernel alone: grouped == per language
    for key in ("flow_lm.transformer.layers.0.self_attn.in_proj.weight", "flow_lm.transformer.layers.7.linear2.weight"):
        base = _BASES["24l"].terms[key]
        gt = fused.GroupedTerms([m.Q[key][:3] for m in dm], base, "cuda")
        cnt = [2] + [1] * (len(dm) - 1)
        x = torch.randn(sum(cnt), base[1].shape[1] if base[0] == "bf16" else _BASES["24l"].cols[key],
                        device="cuda", dtype=torch.bfloat16)
        st = [sum(cnt[:i]) for i in range(len(cnt))]
        y = fused.fused_linear_grouped(x, dm[0].Q[key][1].shape[0], gt, torch.tensor(st, dtype=torch.int32,
                                       device="cuda"), torch.tensor(cnt, dtype=torch.int32, device="cuda"), 128)
        for m, a, n in zip(dm, st, cnt):
            ref = fused.fused_linear2(x[a:a + n], m.Q[key][1].shape[0], m.Q[key][:3], m.QB[key], 128)
            assert torch.equal(y[a:a + n], ref), (key, m.lang, (y[a:a + n] - ref).abs().max().item())
    voices = {x: m.voice_from_file(os.environ["F2_VOICE"]) for x, m in models.items()}
    text = {x: models[x].chunks(SAY.get(x, SAY["en"]))[0] for x in langs}

    def run(cross, rows_per_lang=(2, 1), frames=60, timed=False):
        pool = SlotPool(models[langs[0]], slots=16, max_len=512)
        bs = {x: PocketBatch(models[x], pool=pool) for x in langs}
        keys = {}
        for i, x in enumerate(langs):
            n = rows_per_lang[0] if i == 0 else rows_per_lang[1]
            keys[x] = [f"{x}{j}" for j in range(n)]
            for j, k in enumerate(keys[x]):
                g = torch.Generator(device="cuda").manual_seed(100 * i + j)
                bs[x].start(k, voices[x], text[x][0], text[x][1], gen=g)
        lat, aud = {k: [] for ks in keys.values() for k in ks}, {k: [] for ks in keys.values() for k in ks}
        torch.cuda.synchronize()
        t0, n_steps = _t.monotonic(), 0
        for _ in range(frames):
            groups = [(bs[x], [k for k in keys[x] if k in bs[x].rows]) for x in langs]
            groups = [g for g in groups if g[1]]
            if not groups:
                break
            if cross:
                res = step_many(groups)
            else:
                res = {}
                for b, ks in groups:
                    res.update(b.step(ks))
            n_steps += 1
            for x in langs:
                for k in list(keys[x]):
                    if k not in res:
                        continue
                    if res[k] is None:
                        bs[x].stop(k)
                    else:
                        aud[k].append(res[k].float().cpu())
                        lat[k].append(bs[x].rows[k]["x"].clone())
        torch.cuda.synchronize()
        ms = (_t.monotonic() - t0) * 1000 / max(n_steps, 1)
        return lat, aud, ms

    lat_a, aud_a, _ = run(False)
    lat_b, aud_b, _ = run(True)
    for k in lat_a:
        assert len(lat_a[k]) == len(lat_b[k]) > 5, (k, len(lat_a[k]), len(lat_b[k]))
        for i, (p, q) in enumerate(zip(lat_a[k], lat_b[k])):
            assert torch.equal(p, q), (k, i, (p - q).abs().max().item())
        da = max((p - q).abs().max().item() for p, q in zip(aud_a[k], aud_b[k]))
        assert da < 1e-2, (k, da)
    run(False, (1, 1), 10)                                   # warm both paths, then time one row per language
    run(True, (1, 1), 10)
    _, _, ms_a = run(False, (1, 1), 40)
    _, _, ms_b = run(True, (1, 1), 40)
    print(f"TIMING {len(langs)} languages x 1 row: per-language batches {ms_a:.1f} ms/frame, "
          f"cross-language {ms_b:.1f} ms/frame ({ms_a / ms_b:.2f}x)")
