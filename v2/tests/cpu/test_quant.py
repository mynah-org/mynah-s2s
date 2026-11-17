"""int4 quantizer (CPU parts only; the tinygemm kernel is CUDA-only and is exercised by scripts/quant_eval.py)."""

import torch
import torch.nn as nn
import json

from mynah_v2.model.quant import (BUNDLE_FORMAT, W8_BUNDLE_FORMAT, Int4Linear, TARGETS, _pick,
                                  dequantize_int4_reference, load_int4_pack, load_w8_bundle)
from mynah_v2.model.wo_gemv import WOLinear


def test_group_and_tiles_fit_every_nemotron_k():
    for k in (4480, 15680, 10240, 5120):
        g, t = _pick(k)
        assert k % g == 0 and k % (16 * t) == 0
    assert _pick(15680) == (64, 4)


def test_minmax_int4_error_is_bounded_by_half_a_step():
    torch.manual_seed(0)
    w = torch.randn(64, 256) * 0.02
    d = dequantize_int4_reference(w, 128)
    step = (w.reshape(64, 2, 128).amax(-1) - w.reshape(64, 2, 128).amin(-1)) / 15
    err = (d - w).abs().reshape(64, 2, 128).amax(-1)
    assert torch.all(err <= step / 2 + 1e-7)


def test_targets_are_backbone_linears_only():
    assert TARGETS == {"in_proj", "out_proj", "up_proj", "down_proj", "q_proj", "k_proj", "v_proj", "o_proj"}
    assert isinstance(nn.Linear(2, 2), nn.Linear)


def test_offline_packed_linear_validates_layout_without_requantizing():
    lin = nn.Linear(128, 4, bias=True, dtype=torch.bfloat16)
    packed = torch.empty(4, 64, dtype=torch.int32)
    scales_zeros = torch.empty(1, 4, 2, dtype=torch.bfloat16)
    q = Int4Linear.from_packed(lin, packed, scales_zeros, group=128, tiles=8)
    assert q.packed.data_ptr() == packed.data_ptr()
    assert q.scales_zeros.data_ptr() == scales_zeros.data_ptr()
    assert q.bias is lin.bias


def test_offline_bundle_loads_thinker_and_eartts_scopes_atomically(tmp_path):
    class FakeStt(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([nn.Linear(128, 4, dtype=torch.bfloat16)])

    class FakeBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([nn.Linear(128, 4, dtype=torch.bfloat16)])

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.stt_model = FakeStt()
            self.tts_model = nn.Module()
            self.tts_model.tts_model = nn.Module()
            self.tts_model.tts_model.backbone = FakeBackbone()

    class FakeEngine:
        def __init__(self):
            self.wrapper = type("Wrapper", (), {"model": FakeModel()})()

    tensors = {}
    modules = []
    for root, name in (("stt_model", "layers.0"), ("eartts_backbone", "layers.0")):
        prefix = f"{root}.{name}"
        pk, sk = f"{prefix}.packed", f"{prefix}.scales_zeros"
        tensors[pk] = torch.zeros(4, 64, dtype=torch.int32)
        tensors[sk] = torch.zeros(1, 4, 2, dtype=torch.bfloat16)
        modules.append({"root": root, "name": name, "in_features": 128, "out_features": 4,
                        "group": 128, "tiles": 8, "packed_key": pk, "scales_zeros_key": sk})
    from safetensors.torch import save_file
    pack = tmp_path / "bundle.safetensors"
    save_file(tensors, str(pack))
    (tmp_path / "bundle.json").write_text(json.dumps({"format": BUNDLE_FORMAT, "modules": modules}))

    engine = FakeEngine()
    info = load_int4_pack(engine, pack)
    assert info["modules"] == 2
    assert isinstance(engine.wrapper.model.stt_model.layers[0], Int4Linear)
    assert isinstance(engine.wrapper.model.tts_model.tts_model.backbone.layers[0], Int4Linear)


def test_offline_w8_bundle_loads_without_requantizing(tmp_path):
    class FakeStt(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([nn.Linear(7, 4, dtype=torch.bfloat16)])
            self.perception = nn.Identity()
            self.rnnt_joint = nn.Identity()

    class FakeBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([nn.Linear(7, 4, dtype=torch.bfloat16)])

    class FakeTTS(nn.Module):
        def __init__(self):
            super().__init__()
            self.tts_model = nn.Module()
            self.tts_model.backbone = FakeBackbone()

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.stt_model = FakeStt()
            self.tts_model = FakeTTS()

    class FakeEngine:
        def __init__(self):
            self.wrapper = type("Wrapper", (), {"model": FakeModel()})()

    tensors, modules = {}, []
    for root, name in (("stt_model", "layers.0"), ("eartts_backbone", "layers.0")):
        prefix = f"{root}.{name}"
        qk, sk = f"{prefix}.q", f"{prefix}.scale"
        tensors[qk] = torch.ones(4, 7, dtype=torch.int8)
        tensors[sk] = torch.ones(4, dtype=torch.float32)
        modules.append({"root": root, "name": name, "kind": "linear", "fmt": "w8",
                        "in_features": 7, "out_features": 4, "q_key": qk, "scale_key": sk})
    from safetensors.torch import save_file
    pack = tmp_path / "w8.safetensors"
    save_file(tensors, str(pack))
    (tmp_path / "w8.json").write_text(json.dumps({"format": W8_BUNDLE_FORMAT, "modules": modules}))

    engine = FakeEngine()
    info = load_w8_bundle(engine, pack)
    assert info["modules"] == 2
    assert isinstance(engine.wrapper.model.stt_model.layers[0], WOLinear)
    assert isinstance(engine.wrapper.model.tts_model.tts_model.backbone.layers[0], WOLinear)
    assert torch.equal(engine.wrapper.model.stt_model.layers[0].p_q, tensors["stt_model.layers.0.q"])
