"""EXPERIMENTAL weight-only INT4 quantization for the graph-safe V2 paths. Never used by REFERENCE.

Why int4 tinygemm: the Thinker step is bandwidth-bound (profile C2/C6: 68-74 ms/step, gemv kernels; 14.1 GB of BF16
linear weights read per step). On GB10/sm_121 with torch 2.10 native kernels (microbench D1, M=1):
bf16 64.4 ms/step, int8 weight-only ``_weight_int8pack_mm`` 86.5 ms (slower), fp8 row-wise ``_scaled_mm`` not supported
by cuBLASLt, int4 ``_weight_int4pack_mm`` 11.9 ms (5.4×). Only int4 actually pays off, hence this module.

Scope: Thinker Mamba/attention/MLP projections and optional text/function heads. The bundle API also stores the
separately discovered EarTTS Gemma backbone linears. Embeddings, RNN-T, perception, CAS/MoG/RVQ and codec stay
outside this module until they have a compatible graph-safe backend. Asymmetric min-max per group
(torchao/tinygemm convention: w ≈ (q - 8) · scale + zero). Group 128, or 64/32 when K is not a multiple.
The replaced module keeps an empty ``weight`` buffer so the remote code's device/dtype checks still work
(``in_proj.weight.device`` for the fast-path dispatch, ``q_proj.weight.dtype``). Only the cached (incremental) path is
supported: the training-only fused kernel reads ``out_proj.weight`` directly.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn

TARGETS = {"in_proj", "out_proj", "up_proj", "down_proj", "q_proj", "k_proj", "v_proj", "o_proj"}


def _pick(k: int, max_group: int = 128) -> tuple[int, int]:
    """(group size, inner-k tiles) satisfying tinygemm: K % group == 0 and K % (16 × inner tiles) == 0."""
    group = next(g for g in (128, 64, 32) if g <= max_group and k % g == 0)
    tiles = next(t for t in (8, 4, 2) if k % (16 * t) == 0)
    return group, tiles


def _clip_range(wf: torch.Tensor, steps=(1.0, 0.95, 0.9, 0.85, 0.8, 0.75)):
    """Per-group [min, max] shrink that minimises the int4 reconstruction MSE (HQQ/AWQ-style range search)."""
    mn0, mx0 = wf.amin(-1), wf.amax(-1)
    best_mn, best_mx = mn0.clone(), mx0.clone()
    best = torch.full_like(mn0, float("inf"))
    for f in steps:
        mid = (mx0 + mn0) / 2
        half = (mx0 - mn0) / 2 * f
        mn, mx = mid - half, mid + half
        scale = (mx - mn).clamp_min(1e-8) / 15
        q = torch.round((wf - mn[..., None]) / scale[..., None]).clamp(0, 15)
        err = ((q * scale[..., None] + mn[..., None]) - wf).pow(2).sum(-1)
        upd = err < best
        best = torch.where(upd, err, best)
        best_mn, best_mx = torch.where(upd, mn, best_mn), torch.where(upd, mx, best_mx)
    return best_mn, best_mx


def quantize_int4(w: torch.Tensor, group: int, tiles: int, clip: bool = False):
    """bf16 (N, K) -> (packed int4 weight, scales_and_zeros (K/group, N, 2) bf16)."""
    n, k = w.shape
    wf = w.float().reshape(n, k // group, group)
    mn, mx = _clip_range(wf) if clip else (wf.amin(-1), wf.amax(-1))
    scale = (mx - mn).clamp_min(1e-8) / 15
    zero = mn + scale * 8
    q = torch.round((wf - mn[..., None]) / scale[..., None]).clamp(0, 15).to(torch.int32).reshape(n, k)
    packed = torch._convert_weight_to_int4pack(((q[:, ::2] << 4) | q[:, 1::2]).to(torch.uint8), tiles)
    sz = torch.stack([scale, zero], -1).transpose(0, 1).contiguous().to(torch.bfloat16)
    return packed, sz


def dequantize_int4_reference(w: torch.Tensor, group: int) -> torch.Tensor:
    """Same quantizer, dequantized in torch (for error checks without the kernel)."""
    n, k = w.shape
    wf = w.float().reshape(n, k // group, group)
    mn, mx = wf.amin(-1), wf.amax(-1)
    scale = (mx - mn).clamp_min(1e-8) / 15
    q = torch.round((wf - mn[..., None]) / scale[..., None]).clamp(0, 15)
    return (q * scale[..., None] + mn[..., None]).reshape(n, k)


class Int4Linear(nn.Module):
    def __init__(self, lin: nn.Linear, max_group: int = 128, clip: bool = False):
        super().__init__()
        w = lin.weight.data
        self.in_features, self.out_features = lin.in_features, lin.out_features
        self.group, self.tiles = _pick(self.in_features, max_group)
        packed, sz = quantize_int4(w.to(torch.bfloat16), self.group, self.tiles, clip=clip)
        self.register_buffer("packed", packed)
        self.register_buffer("scales_zeros", sz)
        self.register_buffer("weight", torch.empty(0, dtype=w.dtype, device=w.device), persistent=False)
        self.bias = lin.bias

    @classmethod
    def from_packed(cls, lin: nn.Linear, packed: torch.Tensor, scales_zeros: torch.Tensor,
                    group: int, tiles: int):
        """Build a tinygemm linear from an offline pack without re-quantizing the checkpoint.

        The original bias is deliberately retained from ``lin``.  This keeps the pack
        format weight-only and makes loading safe against accidentally changing the
        model's non-quantized state (biases, caches, embeddings, and audio modules).
        """
        if packed.device != lin.weight.device or scales_zeros.device != lin.weight.device:
            raise ValueError("offline int4 tensors must be loaded on the same device as the model")
        # Torch's tinygemm pack dtype is device/version dependent: GB10 with
        # torch 2.10 returns int32, while some builds expose uint8 storage.
        if packed.dtype not in (torch.int32, torch.uint8):
            raise ValueError(f"offline int4 packed weights must be int32 or uint8, got {packed.dtype}")
        if scales_zeros.dtype != torch.bfloat16:
            raise ValueError(f"offline int4 scales_zeros must be bfloat16, got {scales_zeros.dtype}")
        expected = _pick(lin.in_features, max_group=group)
        if expected != (group, tiles):
            raise ValueError(f"invalid int4 layout for {lin.in_features=} expected {expected}, got {(group, tiles)}")
        out = cls.__new__(cls)
        nn.Module.__init__(out)
        out.in_features, out.out_features = lin.in_features, lin.out_features
        out.group, out.tiles = group, tiles
        out.register_buffer("packed", packed.contiguous())
        out.register_buffer("scales_zeros", scales_zeros.contiguous())
        out.register_buffer("weight", torch.empty(0, dtype=lin.weight.dtype, device=lin.weight.device),
                            persistent=False)
        out.bias = lin.bias
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shp = x.shape
        x2 = x.reshape(-1, shp[-1])
        dt = x2.dtype
        y = torch._weight_int4pack_mm(x2.to(torch.bfloat16).contiguous(), self.packed, self.group, self.scales_zeros)
        if self.bias is not None:
            y = y + self.bias.to(y.dtype)
        return y.to(dt).reshape(*shp[:-1], self.out_features)


def quantize_thinker_int4(engine, max_group: int = 128, clip: bool = False, skip_layers: set[int] | None = None,
                          keep_original: bool = False, include_heads: bool = False):
    """Replace Thinker projections in place, optionally including text/function heads.

    ``include_heads`` is deliberately separate from the backbone switch: these two
    large vocab projections are read on every decode step, but the recurrent/KV
    caches and all non-Linear audio state remain untouched. With ``keep_original``
    the replaced modules are returned so ``restore_thinker`` can undo an A/B run.
    """
    stt = engine.wrapper.model.stt_model
    llm = stt.llm
    before = after = 0
    n = 0
    originals = []
    skip_layers = skip_layers or set()
    if include_heads:
        for name in ("lm_head", "function_head"):
            mod = getattr(stt, name, None)
            if isinstance(mod, nn.Linear):
                before += mod.weight.numel() * mod.weight.element_size()
                q = Int4Linear(mod, max_group=max_group, clip=clip)
                after += q.packed.numel() * q.packed.element_size() + q.scales_zeros.numel() * 2
                setattr(stt, name, q)
                if keep_original:
                    originals.append((stt, name, mod))
                n += 1
    for li, layer in enumerate(llm.layers):
        mixer = getattr(layer, "mixer", None)
        if mixer is None or li in skip_layers:
            continue
        for name, mod in list(mixer.named_children()):
            if name in TARGETS and isinstance(mod, nn.Linear):
                before += mod.weight.numel() * mod.weight.element_size()
                q = Int4Linear(mod, max_group=max_group, clip=clip)
                after += q.packed.numel() * q.packed.element_size() + q.scales_zeros.numel() * 2
                setattr(mixer, name, q)
                if keep_original:
                    originals.append((mixer, name, mod))
                n += 1
    torch.cuda.empty_cache()
    info = {"thinker_quant": "int4_tinygemm", "linears": n, "include_heads": include_heads,
            "max_group": max_group, "clip_search": clip,
            "skip_layers": sorted(skip_layers), "bf16_GB": round(before / 2**30, 2), "int4_GB": round(after / 2**30, 2)}
    return (info, originals) if keep_original else info


def restore_thinker(originals) -> None:
    for mixer, name, mod in originals:
        setattr(mixer, name, mod)
    torch.cuda.empty_cache()


PACK_FORMAT = "mynah.int4.tinygemm.v1"
BUNDLE_FORMAT = "mynah.int4.bundle.v1"


def _replace_child(root: nn.Module, name: str, replacement: nn.Module) -> None:
    parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
    parent = root.get_submodule(parent_name) if parent_name else root
    setattr(parent, child_name, replacement)


def save_thinker_int4_pack(engine, path: str | Path, max_group: int = 128, clip: bool = False,
                           include_heads: bool = True) -> dict:
    """Quantize Thinker weights once and export a portable safetensors pack + manifest.

    This is intentionally weight-only.  The manifest records every replaced module
    and tinygemm layout; the runtime can validate it before swapping modules.  Audio
    and recurrent state are not included because their CUDA-graph contracts are
    model-specific and cannot be made safe by replacing arbitrary ``nn.Linear``
    objects.
    """
    path = Path(path)
    if path.suffix != ".safetensors":
        raise ValueError("int4 pack path must end in .safetensors")
    path.parent.mkdir(parents=True, exist_ok=True)
    info = quantize_thinker_int4(engine, max_group=max_group, clip=clip, include_heads=include_heads)
    stt = engine.wrapper.model.stt_model
    tensors = {}
    modules = []
    for name, mod in stt.named_modules():
        if not isinstance(mod, Int4Linear):
            continue
        pk, sk = f"{name}.packed", f"{name}.scales_zeros"
        tensors[pk] = mod.packed.detach().cpu().contiguous()
        tensors[sk] = mod.scales_zeros.detach().cpu().contiguous()
        modules.append({"name": name, "in_features": mod.in_features, "out_features": mod.out_features,
                        "group": mod.group, "tiles": mod.tiles, "packed_dtype": str(mod.packed.dtype),
                        "scales_dtype": str(mod.scales_zeros.dtype), "packed_key": pk, "scales_zeros_key": sk,
                        "has_bias": mod.bias is not None})
    if not modules:
        raise RuntimeError("no Thinker int4 modules were produced")
    manifest = {
        "format": PACK_FORMAT,
        "backend": "torch_native_weight_int4pack_mm",
        "packing": "torch._convert_weight_to_int4pack",
        "compute_dtype": "bfloat16",
        "root": "stt_model",
        "weight_only": True,
        "audio_modules": "excluded_until_graph_safe_backend",
        "quantizer": "asymmetric_minmax_tinygemm",
        "max_group": max_group,
        "clip_search": clip,
        "modules": modules,
        "quant_info": info,
    }
    from safetensors.torch import save_file
    tmp_pack = path.with_name(path.name + ".tmp")
    tmp_manifest = path.with_suffix(".json.tmp")
    save_file(tensors, str(tmp_pack), metadata={"format": PACK_FORMAT})
    tmp_pack.replace(path)
    tmp_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_manifest.replace(path.with_suffix(".json"))
    return {"path": str(path), "manifest": str(path.with_suffix(".json")), "modules": len(modules), **info}


def _bundle_module_spec(root_name: str, name: str, mod: Int4Linear, tensors: dict) -> dict:
    """Copy one native INT4 module into a bundle and return its portable spec."""
    prefix = f"{root_name}.{name}" if name else root_name
    pk, sk = f"{prefix}.packed", f"{prefix}.scales_zeros"
    tensors[pk] = mod.packed.detach().cpu().contiguous()
    tensors[sk] = mod.scales_zeros.detach().cpu().contiguous()
    return {"root": root_name, "name": name, "in_features": mod.in_features,
            "out_features": mod.out_features, "group": mod.group, "tiles": mod.tiles,
            "packed_dtype": str(mod.packed.dtype), "scales_dtype": str(mod.scales_zeros.dtype),
            "packed_key": pk, "scales_zeros_key": sk, "has_bias": mod.bias is not None}


def save_int4_bundle(engine, path: str | Path, max_group: int = 128, clip: bool = False,
                     include_heads: bool = True, include_eartts: bool = True) -> dict:
    """Export one offline INT4 bundle for every currently graph-safe weight family.

    The bundle deliberately uses the same Torch tinygemm representation for the
    Thinker and the EarTTS Gemma backbone.  That means the server can load one
    file, validate every module, and only then swap anything in the live model.
    Unsupported audio pieces (perception, CAS/MoG/RVQ, and codec) are not put in
    the bundle until they have their own measured graph-safe implementation.
    """
    path = Path(path)
    if path.suffix != ".safetensors":
        raise ValueError("int4 bundle path must end in .safetensors")
    path.parent.mkdir(parents=True, exist_ok=True)
    thinker_info = quantize_thinker_int4(engine, max_group=max_group, clip=clip,
                                         include_heads=include_heads)
    eartts_info = None
    if include_eartts:
        from mynah_v2.model.eartts_q4 import quantize_eartts_int4
        eartts_info = quantize_eartts_int4(engine, group=max_group, backend="native")

    model = engine.wrapper.model
    tensors = {}
    modules = []
    stt = model.stt_model
    for name, mod in stt.named_modules():
        if isinstance(mod, Int4Linear):
            modules.append(_bundle_module_spec("stt_model", name, mod, tensors))
    if include_eartts:
        backbone = model.tts_model.tts_model.backbone
        for name, mod in backbone.named_modules():
            if isinstance(mod, Int4Linear):
                modules.append(_bundle_module_spec("eartts_backbone", name, mod, tensors))
    if not modules:
        raise RuntimeError("no native INT4 modules were produced")
    manifest = {
        "format": BUNDLE_FORMAT,
        "backend": "torch_native_weight_int4pack_mm",
        "packing": "torch._convert_weight_to_int4pack",
        "compute_dtype": "bfloat16",
        "weight_only": True,
        "scopes": {
            "stt_model": "Thinker projections and optional text/function heads",
            "eartts_backbone": "EarTTS Gemma backbone linears",
        },
        "unsupported_scopes": ["perception", "rnnt", "cas_subword", "mog", "rvq", "codec"],
        "quantizer": "asymmetric_minmax_tinygemm",
        "max_group": max_group,
        "clip_search": clip,
        "modules": modules,
        "quant_info": {"thinker": thinker_info, "eartts": eartts_info},
    }
    from safetensors.torch import save_file
    tmp_pack = path.with_name(path.name + ".tmp")
    tmp_manifest = path.with_suffix(".json.tmp")
    save_file(tensors, str(tmp_pack), metadata={"format": BUNDLE_FORMAT})
    tmp_pack.replace(path)
    tmp_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_manifest.replace(path.with_suffix(".json"))
    return {"path": str(path), "manifest": str(path.with_suffix(".json")), "modules": len(modules),
            "format": BUNDLE_FORMAT, "thinker": thinker_info, "eartts": eartts_info}


def load_thinker_int4_pack(engine, path: str | Path, strict: bool = True) -> dict:
    """Load and validate an offline pack before replacing Thinker linears.

    ``strict=True`` rejects missing/mismatched modules rather than silently running
    a partially quantized model.  The caller can opt out only for an explicit A/B
    experiment; the server uses strict loading.
    """
    path = Path(path)
    manifest_path = path.with_suffix(".json")
    if not path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"int4 pack requires {path} and {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != PACK_FORMAT:
        raise ValueError(f"unsupported int4 pack format: {manifest.get('format')!r}")
    stt = engine.wrapper.model.stt_model
    device = str(next(stt.parameters()).device)
    from safetensors.torch import load_file
    tensors = load_file(str(path), device=device)
    missing = []
    replacements = []
    for spec in manifest.get("modules", []):
        name = spec["name"]
        try:
            current = stt.get_submodule(name)
        except AttributeError:
            missing.append(name)
            continue
        if not isinstance(current, nn.Linear):
            missing.append(f"{name} (not nn.Linear)")
            continue
        shape = (current.out_features, current.in_features)
        if shape != (spec["out_features"], spec["in_features"]):
            missing.append(f"{name} (shape {shape} != manifest)")
            continue
        try:
            packed = tensors[spec["packed_key"]]
            scales = tensors[spec["scales_zeros_key"]]
            replacement = Int4Linear.from_packed(current, packed, scales, spec["group"], spec["tiles"])
        except (KeyError, ValueError) as exc:
            missing.append(f"{name} ({exc})")
            continue
        replacements.append((name, replacement))
    if missing and strict:
        raise RuntimeError("int4 pack validation failed: " + "; ".join(missing[:8]))
    for name, replacement in replacements:
        _replace_child(stt, name, replacement)
    loaded = len(replacements)
    if loaded == 0:
        raise RuntimeError("int4 pack did not replace any module")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"path": str(path), "modules": loaded, "skipped": missing, "format": PACK_FORMAT,
            "weight_only": manifest.get("weight_only", False), "audio_modules": manifest.get("audio_modules")}


def load_int4_pack(engine, path: str | Path, strict: bool = True) -> dict:
    """Load either the legacy Thinker pack or the combined INT4 bundle."""
    path = Path(path)
    manifest_path = path.with_suffix(".json")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"int4 pack requires {path} and {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") == PACK_FORMAT:
        return load_thinker_int4_pack(engine, path, strict=strict)
    if manifest.get("format") != BUNDLE_FORMAT:
        raise ValueError(f"unsupported int4 pack format: {manifest.get('format')!r}")
    model = engine.wrapper.model
    roots = {"stt_model": model.stt_model, "eartts_backbone": model.tts_model.tts_model.backbone}
    device = str(next(model.parameters()).device)
    from safetensors.torch import load_file
    tensors = load_file(str(path), device=device)
    missing = []
    replacements = []
    for spec in manifest.get("modules", []):
        root_name, name = spec.get("root"), spec.get("name", "")
        root = roots.get(root_name)
        if root is None:
            missing.append(f"{root_name}.{name} (unknown root)")
            continue
        try:
            current = root.get_submodule(name) if name else root
        except AttributeError:
            missing.append(f"{root_name}.{name}")
            continue
        if not isinstance(current, nn.Linear):
            missing.append(f"{root_name}.{name} (not nn.Linear)")
            continue
        shape = (current.out_features, current.in_features)
        if shape != (spec["out_features"], spec["in_features"]):
            missing.append(f"{root_name}.{name} (shape {shape} != manifest)")
            continue
        try:
            replacement = Int4Linear.from_packed(current, tensors[spec["packed_key"]],
                                                 tensors[spec["scales_zeros_key"]],
                                                 spec["group"], spec["tiles"])
        except (KeyError, ValueError) as exc:
            missing.append(f"{root_name}.{name} ({exc})")
            continue
        replacements.append((root, name, replacement))
    if missing and strict:
        raise RuntimeError("int4 bundle validation failed: " + "; ".join(missing[:8]))
    for root, name, replacement in replacements:
        _replace_child(root, name, replacement)
    if not replacements:
        raise RuntimeError("int4 bundle did not replace any module")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"path": str(path), "modules": len(replacements), "skipped": missing,
            "format": BUNDLE_FORMAT, "weight_only": manifest.get("weight_only", False),
            "unsupported_scopes": manifest.get("unsupported_scopes", [])}
# End of quantization helpers.
