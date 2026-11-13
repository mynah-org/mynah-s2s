"""GPU regression for the default-ON incremental Thinker (mynah_v2.model.llm_cache) on the real checkpoint.

Deterministic decoding (``s2s.deterministic``: torch deterministic algorithms, math SDPA), so it runs in its OWN pytest
process (the flag is process-wide): ``pytest -m gpu tests/gpu/test_llm_cache_gpu.py``. The no-cache path is the oracle.
"""

import os
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.gpu

CKPT = Path(os.environ.get("V2_CKPT", Path(os.environ.get("MYNAH_HOME", Path.home() / "mynah-s2s")) / "hf-cache/models--nvidia--NVIDIA-NemotronLabs-VoiceChat-11B/snapshots/443794ea956ef0065f001967ffd00e77f519cb39"))
NEMO = Path(os.environ.get("MYNAH_HOME", Path.home() / "mynah-s2s")) / "third_party/agent/nemo-speech"
SAMPLE = NEMO / "examples/speechlm2/sample_audio/sample_general.wav"
PROMPT = "You are a helpful assistant."
N_CHUNKS = 60   # 9.6 s of the sample: speech + an answer start, short enough for the no-cache oracle


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from mynah_v2.model.reference import ReferenceEngine, build_cfg
    eng = ReferenceEngine(build_cfg(ckpt=CKPT, output_dir=str(tmp_path_factory.mktemp("nemo")),
                                    overrides={"s2s.deterministic": True}))
    prof = os.environ.get("V2_PERF_PROFILE")           # run the same suite under a validated perf profile
    if prof:
        from mynah_v2.model import perf
        perf.apply_profile(prof, eng)
    yield eng
    eng.close()
    torch.use_deterministic_algorithms(False)


@pytest.fixture(scope="module")
def chunks():
    import sys
    sys.path.insert(0, str(Path(__file__).parents[2] / "scripts"))
    from run_stream import load_16k
    from mynah_v2.runtime.session import iter_chunks
    return list(iter_chunks(load_16k(str(SAMPLE)), 2560))[:N_CHUNKS]


def drive(eng, chunks, probe=None):
    eng.open(PROMPT)
    for i, c in enumerate(chunks):
        eng.step(c, is_last=(i == len(chunks) - 1))
        if probe is not None and i < len(chunks) - 1:      # the last (is_last) step releases the context
            probe(i, eng)
    tt = eng.token_tensors()
    assert tt is not None, "no token tensors saved after the is_last step"
    gt, ga, n, gf = tt
    out = {"text": gt[0, :n].tolist(), "asr": ga[0, :n].tolist() if ga is not None else None,
           "func": gf[0, :n].tolist() if gf is not None else None}
    eng.close()
    return out


def test_cache_is_on_by_default(engine):
    info = engine.llm_cache_info
    assert info is not None and info["attention_blocks_patched"] == 4 and info["cache_key"] == "cache_params"
    assert engine.wrapper.model.stt_model.cfg.cache_key == "cache_params"
    assert engine.pipeline.context_manager.cache_disabled is False


def test_cache_state_advances_one_frame_per_thinker_step(engine, chunks):
    """Mamba conv/SSM state and attention KV are updated every 80 ms step: position +2 per 160 ms chunk, attention
    KV length == position, SSM state FP32 and changing."""
    seen = []

    def probe(i, eng):
        c = eng._ctx().dynamic_cache
        attn = [k for k, layer in enumerate(eng.wrapper.model.stt_model.llm.layers) if layer.block_type == "attention"]
        mamba = [k for k, layer in enumerate(eng.wrapper.model.stt_model.llm.layers) if layer.block_type == "mamba"]
        seen.append({"id": id(c), "cls": type(c).__name__, "pos": c._v2_pos,
                     "kv": [int(c.key_cache[k].shape[-2]) for k in attn],
                     "ssm_dtype": c.ssm_states[mamba[0]].dtype,
                     "ssm_sum": float(c.ssm_states[mamba[0]].float().abs().sum()),
                     "conv_sum": float(c.conv_states[mamba[0]].float().abs().sum())})

    drive(engine, chunks[:13], probe)
    print("CACHE_PROBE", [{k: v for k, v in s_.items() if k != "id"} for s_ in seen])
    assert {s["cls"] for s in seen} == {"V2HybridMambaAttentionCache"}
    assert len({s["id"] for s in seen}) == 1
    pos = [s["pos"] for s in seen]
    assert all(b - a == 2 for a, b in zip(pos[1:], pos[2:])), pos          # 2 Thinker frames per chunk
    assert all(set(s["kv"]) == {s["pos"]} for s in seen), seen              # attention KV covers every position
    assert all(s["ssm_dtype"] == torch.float32 for s in seen)
    assert len({round(s["ssm_sum"], 3) for s in seen[1:]}) > 1             # recurrent state actually evolves
    assert len({round(s["conv_sum"], 3) for s in seen[1:]}) > 1


def test_sessions_are_isolated(engine, chunks):
    """A fresh cache per session: same input twice -> identical tokens, different cache objects, position restarts."""
    starts = []
    ids = []

    def probe(i, eng):
        c = eng._ctx().dynamic_cache
        if i == 0:
            starts.append(c._v2_pos)
            ids.append(id(c))

    a = drive(engine, chunks, probe)
    b = drive(engine, chunks, probe)
    assert a == b
    assert starts[0] == starts[1] and ids[0] != ids[1]


def test_cache_on_off_token_identity(engine, chunks):
    """Text, function and RNN-T (user ASR) channels identical with the cache ON vs the reference no-cache path."""
    on = drive(engine, chunks)
    engine.set_llm_cache(False)
    try:
        assert engine.pipeline.context_manager.cache_disabled is True
        off = drive(engine, chunks)
    finally:
        engine.set_llm_cache(True)
    assert any(t not in (0, 1, 2, 12) for t in off["text"]), "oracle produced no agent text: input too short"
    assert on["text"] == off["text"]
    assert on["func"] == off["func"]
    assert on["asr"] == off["asr"]
