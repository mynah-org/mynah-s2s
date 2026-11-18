"""Torch CPU tests on synthetic checkpoints: quantisation, package build/load/verify round trip, manifest."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from safetensors.torch import save_file  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pocket_swap.package import (Source, backbone_fingerprint, build_package, load_package,  # noqa: E402
                                 reconstruct_state, verify_package, write_repo_card)
from pocket_swap.quant import dequantize, packed_nbytes, quantize  # noqa: E402


@pytest.mark.parametrize("bits,group", [(8, 128), (4, 128), (8, 0), (4, 64)])
def test_quant_roundtrip_error_bound(bits, group):
    g = torch.Generator().manual_seed(0)
    x = torch.randn(48, 256, generator=g)
    q, s = quantize(x, bits, group)
    y = dequantize(q, s, bits, 256)
    gs = 256 if group == 0 else group
    step = x.reshape(48, -1, gs).abs().amax(-1, keepdim=True) / (127 if bits == 8 else 7)
    assert ((y - x).reshape(48, -1, gs).abs() <= step * 0.5 + 1e-2 * step + 1e-6).all()
    assert q.numel() * q.element_size() + s.numel() * 2 == packed_nbytes(48, 256, bits, group)


def test_int4_packing_layout():
    x = torch.tensor([[7.0, -7.0, 0.0, 1.0]])
    q, s = quantize(x, 4, 4)
    assert q.dtype == torch.uint8 and q.shape == (1, 2)
    assert torch.equal(dequantize(q, s, 4, 4), torch.tensor([[7.0, -7.0, 0.0, 1.0]]))


def _ckpt(path, seed, vocab_seed=None):
    g = torch.Generator().manual_seed(seed)
    sd = {
        "flow_lm.transformer.layers.0.linear1.weight": torch.randn(64, 32, generator=g),
        "flow_lm.transformer.layers.0.linear2.weight": torch.randn(32, 64, generator=g),
        "flow_lm.transformer.layers.0.norm1.weight": torch.randn(32, generator=g),
        "flow_lm.conditioner.embed.weight": torch.randn(11, 32, generator=torch.Generator().manual_seed(vocab_seed or seed)),
        "flow_lm.flow_net.res_blocks.0.mlp.0.weight": torch.randn(16, 16, generator=g),
        "mimi.decoder.weight": torch.ones(8, 8),
    }
    sd = {k: v.to(torch.bfloat16) for k, v in sd.items()}
    save_file(sd, str(path))
    return sd


@pytest.fixture()
def ckpts(tmp_path):
    base = _ckpt(tmp_path / "base.safetensors", 1)
    tgt = _ckpt(tmp_path / "tgt.safetensors", 2, vocab_seed=9)
    tok = tmp_path / "tokenizer.json"
    tok.write_text("{}")
    cfg = tmp_path / "lang.yaml"
    cfg.write_text("weights_path: hf://x\nweights_path_without_voice_cloning: hf://y\nremove_semicolons: true\n")
    return tmp_path, base, tgt, tok, cfg


@pytest.mark.parametrize("precision,exact", [("exact", True), ("bf16", False), ("int8", False), ("int4", False)])
def test_build_load_reconstruct(ckpts, precision, exact):
    d, base, tgt, tok, cfg = ckpts
    m = build_package(Source(str(d / "base.safetensors"), "kyutai/x", "rev1", "a.safetensors"),
                      Source(str(d / "tgt.safetensors"), "kyutai/y", "rev2", "b.safetensors"),
                      "it", d / "out", tok, precision, group=16, config_yaml=cfg)
    assert m["base"]["revision"] == "rev1" and len(m["base"]["sha256"]) == 64 and m["licence"]["weights"] == "CC-BY-4.0"
    assert set(m["files"]) == {"delta", "lang", "tokenizer", "config"}
    assert "weights_path" not in (d / "out/it/config.yaml").read_text()
    pkg = load_package(d / "out/it", check_hashes=True)
    assert "mimi.decoder.weight" not in pkg.verbatim
    assert "flow_lm.conditioner.embed.weight" in pkg.verbatim  # different vocabulary: shipped whole
    rec = reconstruct_state(base, pkg)
    for k, y in tgt.items():
        if not k.startswith("flow_lm."):
            continue
        if k.startswith("flow_lm.transformer.") and y.dim() == 2:
            err = (rec[k].float() - y.float()).norm() / y.float().norm()
            assert (err == 0) if exact else (err < {"bf16": 0.01, "int8": 0.02, "int4": 0.2}[precision])
        else:
            assert torch.equal(rec[k], y)
    r = verify_package(str(d / "base.safetensors"), str(d / "out/it"), str(d / "tgt.safetensors"))
    assert r["base_matches"] and r["hashes"] == "ok"
    if exact:
        assert r["tensors_bit_exact"] == "5/5"


def test_base_language_package_has_no_delta(ckpts):
    d, base, tgt, tok, _ = ckpts
    m = build_package(Source(str(d / "base.safetensors")), None, "en", d / "out", tok)
    assert m["precision"] == "base" and "delta" not in m["files"]
    pkg = load_package(d / "out/en")
    rec = reconstruct_state(base, pkg)
    assert all(torch.equal(rec[k], v) for k, v in base.items() if k.startswith("flow_lm."))


def test_verify_detects_wrong_base_and_tampering(ckpts):
    d, base, tgt, tok, _ = ckpts
    build_package(Source(str(d / "base.safetensors")), Source(str(d / "tgt.safetensors")), "it", d / "out", tok, "int8", 16)
    assert not verify_package(str(d / "tgt.safetensors"), str(d / "out/it"), None)["base_matches"]
    assert backbone_fingerprint(base) != backbone_fingerprint(tgt)
    p = d / "out/it/tokenizer.json"
    p.write_text('{"x": 1}')
    with pytest.raises(ValueError, match="sha256"):
        load_package(d / "out/it", check_hashes=True)


def test_zstd_roundtrip_and_card(ckpts):
    pytest.importorskip("zstandard")
    d, base, tgt, tok, _ = ckpts
    m = build_package(Source(str(d / "base.safetensors"), "kyutai/x", "rev1", "a"), Source(str(d / "tgt.safetensors"), "kyutai/y", "r", "b"),
                      "it", d / "out", tok, "int4", 16, zstd_level=3)
    assert m["files"]["delta"]["name"].endswith(".zst")
    pkg = load_package(d / "out/it", check_hashes=True)
    assert pkg.manifest["precision"] == "int4" and pkg.delta
    card = write_repo_card(d / "out", [json.loads((d / "out/it/manifest.json").read_text())])
    assert "cc-by-4.0" in card.read_text()


@pytest.mark.parametrize("precision", ["bf16", "int8", "int4"])
def test_ondemand_delta_linear_matches_materialised(ckpts, precision):
    from pocket_swap.ondemand import DeltaLinear
    from pocket_swap.package import reconstruct_tensor

    d, base, tgt, tok, _ = ckpts
    build_package(Source(str(d / "base.safetensors")), Source(str(d / "tgt.safetensors")), "it", d / "out", tok,
                  precision, 16)
    pkg = load_package(d / "out/it")
    k = "flow_lm.transformer.layers.0.linear1.weight"
    b = base[k].clone()
    dl = DeltaLinear(b, precision, 16)
    if precision == "bf16":
        dl.load(pkg.delta[k])
    else:
        dl.load(pkg.delta[k + ".q"], pkg.delta[k + ".s"])
    w_mat = reconstruct_tensor(k, b, pkg, torch.bfloat16)
    assert (dl.weight_now().float() - w_mat.float()).abs().max() <= 1e-2 * w_mat.float().abs().max()
    assert torch.equal(b, base[k])  # the shared base is never written
    x = torch.randn(3, 32, dtype=torch.bfloat16)
    assert torch.allclose(dl(x).float(), (x.float() @ w_mat.float().t()), rtol=0.05, atol=0.05)
    dl.load(None)
    assert torch.equal(dl.weight_now(), b)  # zero delta = the base language


@pytest.mark.parametrize("precision", ["bf16", "int8", "int4"])
@pytest.mark.parametrize("rows", [1, 4, 16])
def test_fused_kernel_matches_torch_reference(precision, rows, monkeypatch):
    """Runs the Triton kernel in interpreter mode on CPU (TRITON_INTERPRET=1) against the torch path."""
    import os

    if os.environ.get("TRITON_INTERPRET") != "1":
        pytest.skip("set TRITON_INTERPRET=1 to run the kernel on CPU")
    from pocket_swap import fused
    from pocket_swap.ondemand import DeltaLinear
    from pocket_swap.quant import quantize

    if not fused.HAVE_TRITON:
        pytest.skip("triton not installed")
    g = torch.Generator().manual_seed(rows)
    base = (torch.randn(48, 256, generator=g) * 0.05).to(torch.bfloat16)
    delta = torch.randn(48, 256, generator=g) * 0.05
    dl = DeltaLinear(base, precision, 128, fused=False)
    if precision == "bf16":
        dl.load(delta.to(torch.bfloat16))
    else:
        dl.load(*quantize(delta, 8 if precision == "int8" else 4, 128))
    x = torch.randn(rows, 256, generator=g).to(torch.bfloat16)
    ref = x.float() @ dl.weight_now().float().t()
    codes = dl.d if precision == "bf16" else dl.q
    y = fused.fused_linear(x, base, precision, codes, None if precision == "bf16" else dl.s, 128)
    assert y.shape == (rows, 48)
    assert torch.allclose(y.float(), ref, rtol=2e-2, atol=2e-2), (y.float() - ref).abs().max()


@pytest.mark.parametrize("precision,tol", [("int8", 0.02), ("int4", 0.2), ("fp8", 0.05)])
def test_quantize_any_roundtrip(precision, tol):
    from pocket_swap.quant import dequant_any, quantize_any, term_nbytes

    x = torch.randn(32, 256, generator=torch.Generator().manual_seed(3))
    q, s = quantize_any(x, precision, 128)
    y = dequant_any(q, s, precision, 256)
    assert ((y - x).norm() / x.norm()) < tol
    assert q.numel() * q.element_size() + s.numel() * 2 == term_nbytes(32, 256, precision, 128)


@pytest.mark.parametrize("precision,tol", [("w8", 0.02), ("w4", 0.2), ("wfp8", 0.05)])
def test_weight_only_packages(ckpts, precision, tol):
    d, base, tgt, tok, _ = ckpts
    m = build_package(Source(str(d / "base.safetensors")), Source(str(d / "tgt.safetensors")), "it", d / "out", tok,
                      precision, 16)
    assert m["precision"] == precision and 0 < m["backbone_rel_err"] < tol
    mb = build_package(Source(str(d / "base.safetensors")), None, "en", d / "out", tok, precision, 16)
    assert mb["precision"] == precision and "delta" in mb["files"]  # the base language is quantised too
    for lang, ref in (("it", tgt), ("en", base)):
        pkg = load_package(d / "out" / lang)
        rec = reconstruct_state(base, pkg)
        k = "flow_lm.transformer.layers.0.linear1.weight"
        assert (rec[k].float() - ref[k].float()).norm() / ref[k].float().norm() < tol
        assert torch.equal(rec["flow_lm.conditioner.embed.weight"], ref["flow_lm.conditioner.embed.weight"])


@pytest.mark.parametrize("bp,lp", [("int8", "int8"), ("int4", "int4"), (None, "int8"), (None, "int4"), (None, "fp8"),
                                   ("bf16", "int8")])
def test_delta_linear_terms_torch_and_fused(bp, lp):
    """Quantised base + quantised delta, and weight-only terms: torch path == dequant reference, and the Triton
    kernel (interpreter, when TRITON_INTERPRET=1) == torch path."""
    import os

    from pocket_swap import fused
    from pocket_swap.ondemand import DeltaLinear
    from pocket_swap.quant import dequant_any, quantize_any

    g = torch.Generator().manual_seed(11)
    W0 = torch.randn(48, 256, generator=g) * 0.05
    D = torch.randn(48, 256, generator=g) * 0.05
    if bp is None:
        dl = DeltaLinear(None, lp, 128, fused=False, shape=(48, 256), device="cpu")
        q, s = quantize_any(W0 + D, lp, 128)
        dl.load(q, s)
        ref = dequant_any(q, s, lp, 256)
    else:
        bterm = (bp,) + quantize_any(W0, bp, 128)
        dl = DeltaLinear(None, lp, 128, fused=False, base_term=bterm, shape=(48, 256), device="cpu")
        q, s = quantize_any(D, lp, 128)
        dl.load(q, s)
        ref = dequant_any(*bterm[1:], bp, 256) + dequant_any(q, s, lp, 256)
    assert torch.equal(dl.weight_now(), ref.to(torch.bfloat16))
    x = torch.randn(4, 256, generator=g).to(torch.bfloat16)
    y_t = dl(x).float()
    assert torch.allclose(y_t, x.float() @ ref.to(torch.bfloat16).float().t(), rtol=2e-2, atol=2e-2)
    if os.environ.get("TRITON_INTERPRET") == "1" and fused.available(x, (lp,) + ((bp,) if bp else ())):
        dl.fused = True
        y_f = dl(x).float()
        assert torch.allclose(y_f, y_t, rtol=2e-2, atol=2e-2), (y_f - y_t).abs().max()
