"""GPU tests on the real checkpoint (NVIDIA native engine behind ReferenceEngine). Run as a queue job:
    pytest -m gpu tests/gpu -q
They load the model once per module (~1-2 min). Not run in the CPU suite."""

import json
import os
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

pytestmark = pytest.mark.gpu

from mynah_v2.model import tokens as T  # noqa: E402

CKPT = Path(os.environ.get("V2_CKPT", Path(os.environ.get("MYNAH_HOME", Path.home() / "mynah-s2s")) / "hf-cache/models--nvidia--NVIDIA-NemotronLabs-VoiceChat-11B/snapshots/443794ea956ef0065f001967ffd00e77f519cb39"))
NEMO = Path(os.environ.get("MYNAH_HOME", Path.home() / "mynah-s2s")) / "third_party/agent/nemo-speech"


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from mynah_v2.model.reference import ReferenceEngine, build_cfg
    # V2_LLM_CACHE=0 runs the suite on NVIDIA's shipped no-cache Thinker (diagnostic); default = cache ON
    eng = ReferenceEngine(build_cfg(ckpt=CKPT, output_dir=str(tmp_path_factory.mktemp("nemo"))),
                          llm_cache=os.environ.get("V2_LLM_CACHE", "1") != "0")
    prof = os.environ.get("V2_PERF_PROFILE")           # run the same suite under a validated perf profile
    if prof:
        from mynah_v2.model import perf
        perf.apply_profile(prof, eng)
    yield eng
    eng.close()


def test_lean_loading_used_both_wrapper_calls(engine):
    assert len(engine.lean_load_keys) == 2 and all(n > 0 for n in engine.lean_load_keys)


def test_fusion_weights_match_checkpoint_config(engine):
    cfg = json.loads((CKPT / "config.json").read_text())["model"]["stt"]["model"]
    fm = engine.wrapper.fusion_module
    got = {"user": float(fm.user_audio_weight), "text": float(fm.agent_text_weight),
           "function": float(fm.function_weight)}
    assert got == T.FUSION_WEIGHTS
    assert got == {"user": cfg["duplex_user_channel_weight"], "text": cfg["duplex_text_channel_weight"],
                   "function": cfg["duplex_function_channel_weight"]}


def test_special_token_ids_live_tokenizer(engine):
    assert T.verify_against(engine.tokenizer) == {}
    w = engine.wrapper
    assert (w._fc_sotc_id, w._fc_eotc_id, w._fc_eotr_id) == (T.SOTC_ID, T.EOTC_ID, T.EOTR_ID)
    assert engine.wrapper.model.stt_model.text_pad_id == T.PAD_ID


def test_dtypes_reference_config(engine):
    m = engine.wrapper.model
    assert next(m.stt_model.llm.parameters()).dtype == torch.bfloat16      # wrapper:691 (compute_dtype bf16)
    assert next(m.tts_model.parameters()).dtype == torch.float32           # EarTTS kept FP32 (yaml: bf16 hallucinates)


def test_streaming_steps_silence_shapes_and_state_advance(engine):
    engine.open("You are a helpful assistant.")
    outs = [engine.step(torch.zeros(engine.chunk_samples)) for _ in range(5)]
    assert engine.chunk_samples == 2560
    assert outs[-1].frame_idx - outs[0].frame_idx == 8                    # 2 Thinker steps per 160 ms chunk
    assert all(len(o.new_text_ids) == 2 for o in outs[1:])
    total_audio = sum(o.audio.numel() for o in outs)
    assert total_audio % T.SAMPLES_PER_FRAME_OUT == 0                       # whole 80 ms output frames
    engine.close()


def test_fc_sample_produces_valid_reference_fc_timeline(engine, tmp_path):
    """NVIDIA's own FC sample through the live (non-blocking) path: a call must fire and its sub-timeline must
    satisfy the training invariants (PAD text, no free step EOTC->response, EOTR right after)."""
    import sys
    sys.path.insert(0, str(Path(__file__).parents[2] / "scripts"))
    from run_stream import fc_timeline_from_records, load_16k, render_prompt
    import offline_voicechat_fc_infer as fcmod
    from mynah_v2.runtime.session import Mode, Session, run_paced
    from mynah_v2.tracing.events import Tracer
    tr = Tracer(tmp_path / "t.jsonl")
    s = Session(engine, tr, Mode.REFERENCE)
    s.open(render_prompt(fcmod.DEFAULT_TOOLS))
    run_paced(s, load_16k(str(NEMO / "examples/speechlm2/sample_audio/sample_fc.wav")), pace="realtime",
              tail_silence_s=6)
    recs = list(engine.fc_records)
    s.close()
    assert any(r.phase == 1 for r in recs), "no tool call fired on NVIDIA's FC sample"
    tl = fc_timeline_from_records(recs)
    out = Path(os.environ.get("MYNAH_HOME", Path.home() / "mynah-s2s")) / "results/gpu-tests"
    out.mkdir(parents=True, exist_ok=True)
    (out / "fc_sample_records.json").write_text(json.dumps(
        {"timeline": tl, "raw": [{k: v for k, v in vars(r).items()} for r in recs], "text": "".join(s.text),
         "func_text": "".join(s.func_text)}, indent=1, default=str))
    assert tl["calls"] and all(not c["violations"] for c in tl["calls"]), tl
    # #33 (paper §7 "long tool responses delay speech"): the response is forced one token per 80 ms step, so the
    # phase-2 length must equal the tokenized (number-converted) response, plus the EOTR step
    p2 = [r for r in recs if r.phase == 2 and r.tool_response_text]
    assert p2
    for r in p2:
        n_tok = len(engine.tokenizer.text_to_ids(r.tool_response_text))
        assert r.steps in (n_tok, n_tok + 1), (r.steps, n_tok, r.tool_response_text)
