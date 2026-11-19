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
    ref = words(SAY[lang])
    hit = sum(1 for t in ref if t in set(words(heard)))
    assert hit >= 0.6 * len(ref), (heard, SAY[lang])
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


@pytest.mark.gpu
@pytest.mark.parametrize("lang", LANGS)
def test_language_package_speaks_and_the_ears_hear_it(ears, lang):
    """A pocket_swap w4 package (F2_POCKET_PACKAGES) + the base checkpoint's Mimi, as the engine loads them."""
    import numpy as np
    from scipy.signal import resample_poly
    from frankenstein2.tts.pocket_own import PocketOwn
    root = os.environ.get("F2_POCKET_PACKAGES")
    if not root or not (Path(root) / POCKET[lang] / "manifest.json").exists():
        pytest.skip("no package for this language (F2_POCKET_PACKAGES)")
    base = os.environ.get("F2_POCKET_BASE") or find(os.environ["F2_POCKET_WEIGHTS"],
                                                    f"languages/{POCKET['en']}/model.safetensors")
    m = PocketOwn.from_package(Path(root) / POCKET[lang], lang, base, device="cuda")
    torch.manual_seed(0)
    voice = m.voice_from_file(os.environ["F2_VOICE"])
    x = torch.cat(list(m.stream(voice, SAY[lang]))).float().cpu().numpy()
    assert np.isfinite(x).all() and 1.0 < len(x) / 24000 < 10.0
    heard = ears.transcribe(resample_poly(x, 2, 3).astype(np.float32), lang)
    ref = words(SAY[lang])
    assert sum(1 for t in ref if t in set(words(heard))) >= 0.6 * len(ref), (heard, SAY[lang])
