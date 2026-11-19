"""Language packages: one base Pocket checkpoint + per-language (text embedding, tokenizer, backbone delta).

Layout of a package directory (one per language)::

    <lang>/manifest.json            format, base and target identity (repo, revision, sha256), precision, sizes
    <lang>/delta.safetensors[.zst]  backbone deltas vs the base (absent for the base language itself)
    <lang>/lang.safetensors[.zst]   verbatim FlowLM tensors that are not delta-coded (text embedding, head, ...)
    <lang>/tokenizer.json           the language's tokenizer (vocabularies differ: embeddings are not diffable);
                                    tokenizer.model when the source ships SentencePiece only
    <lang>/config.yaml              the upstream language config (text preprocessing, defaults), weights paths blanked

Precisions of the backbone part:
  exact  target weights stored verbatim in bf16 (no delta; bit-exact for a bf16 target)
  bf16   delta in bf16 (near-exact: rounding of the delta)
  int8   delta, symmetric per-group int8 codes + bf16 scales
  int4   delta, symmetric per-group int4 codes (packed) + bf16 scales
  w8 / w4 / wfp8   the language's own backbone weights quantised (int8 / int4 / fp8 e4m3, per-group scales): no
         delta, no base needed at run time (weight-only quantised end to end); the base language gets one too
Mimi is never stored in a package: it is identical in every language and comes from the base checkpoint.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load as st_load, load_file, save as st_save

from .quant import dequant_any, dequantize, quantize, quantize_any

FORMAT = "pocket-swap-package/1"
PRECISIONS = ("exact", "bf16", "int8", "int4", "w8", "w4", "wfp8")
WEIGHT_PRECISIONS = {"w8": "int8", "w4": "int4", "wfp8": "fp8"}
BACKBONE_PREFIX = "flow_lm.transformer."
LICENCE_NOTICE = (
    "The tensors in this package are derived from Kyutai's Pocket TTS weights (https://huggingface.co/kyutai), "
    "released under CC-BY-4.0 (https://creativecommons.org/licenses/by/4.0/). Changes: the language checkpoint "
    "was re-expressed as a difference from the base checkpoint named in manifest.json (quantised as stated in "
    "'precision'); tensors that are identical in every language (the Mimi codec) are omitted and must be taken "
    "from the original base checkpoint. Use is subject to Kyutai's terms on the original model pages, including "
    "the prohibited uses listed there (e.g. voice cloning without explicit and lawful consent). Kyutai does not "
    "endorse this package."
)


def sha256_file(path: str | os.PathLike, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def is_delta_key(key: str, t: torch.Tensor) -> bool:
    return key.startswith(BACKBONE_PREFIX) and t.dim() == 2


def backbone_fingerprint(state: dict[str, torch.Tensor]) -> str:
    """sha256 over the bf16 bytes of the delta-coded tensors in sorted order (the base the deltas assume)."""
    h = hashlib.sha256()
    for k in sorted(state):
        t = state[k]
        if is_delta_key(k, t):
            h.update(k.encode())
            h.update(t.to(torch.bfloat16).contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def backbone_fingerprint_file(path: str | os.PathLike) -> str:
    """Same as backbone_fingerprint, reading one tensor at a time (no full checkpoint in host RAM)."""
    from safetensors import safe_open

    h = hashlib.sha256()
    with safe_open(str(path), "pt") as f:
        for k in sorted(f.keys()):
            sl = f.get_slice(k)
            if k.startswith(BACKBONE_PREFIX) and len(sl.get_shape()) == 2:
                h.update(k.encode())
                h.update(f.get_tensor(k).to(torch.bfloat16).contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def _write_st(tensors: dict[str, torch.Tensor], path: Path, zstd_level: int | None) -> Path:
    data = st_save({k: v.contiguous() for k, v in tensors.items()})
    if zstd_level is None:
        path.write_bytes(data)
        return path
    import zstandard  # optional dependency

    out = path.with_name(path.name + ".zst")
    out.write_bytes(zstandard.ZstdCompressor(level=zstd_level, threads=-1).compress(data))
    return out


def _read_st(path: Path) -> dict[str, torch.Tensor]:
    if path.name.endswith(".zst"):
        import zstandard

        return st_load(zstandard.ZstdDecompressor().decompress(path.read_bytes()))
    return load_file(str(path))


def _find(dirpath: Path, stem: str) -> Path | None:
    for name in (stem, stem + ".zst"):
        if (dirpath / name).exists():
            return dirpath / name
    return None


@dataclass
class Source:
    """Identity of a checkpoint, recorded in the manifest (repo/revision/file are informational)."""

    path: str
    repo: str = ""
    revision: str = ""
    file: str = ""

    def describe(self, with_hash: bool = True) -> dict:
        d = {"repo": self.repo, "revision": self.revision, "file": self.file}
        if with_hash:
            d["sha256"] = sha256_file(self.path)
        return d


def build_package(base: Source, target: Source | None, lang: str, out_dir: str | os.PathLike,
                  tokenizer: str | os.PathLike, precision: str = "int8", group: int = 128,
                  zstd_level: int | None = None, config_yaml: str | os.PathLike | None = None,
                  licence: str = "CC-BY-4.0", notice: str | None = None) -> dict:
    """Write the package for `lang`. target=None builds the base language package (no delta file).
    licence / notice: the target's licence and attribution (a community model's own terms; default Kyutai's)."""
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}")
    out = Path(out_dir) / lang
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    bsd = load_file(base.path)
    tsd = bsd if target is None else load_file(target.path)
    missing = [k for k in tsd if k.startswith("flow_lm.") and k not in bsd]
    if missing:
        raise ValueError(f"target has FlowLM tensors the base lacks: {missing[:5]}")
    delta: dict[str, torch.Tensor] = {}
    verbatim: dict[str, torch.Tensor] = {}
    err2 = w2 = 0.0
    n_delta = 0
    for k, t in tsd.items():
        if not k.startswith("flow_lm."):
            continue  # Mimi: shared, taken from the base checkpoint
        if precision in WEIGHT_PRECISIONS and is_delta_key(k, t):
            y = t.to(torch.bfloat16).float()
            wp = WEIGHT_PRECISIONS[precision]
            q, sc = quantize_any(y, wp, group)
            delta[k + ".q"], delta[k + ".s"] = q, sc
            rec = dequant_any(q, sc, wp, y.shape[1]).to(torch.bfloat16).float()
            err2 += (rec - y).pow(2).sum().item()
            w2 += y.pow(2).sum().item()
            n_delta += t.numel()
        elif target is not None and is_delta_key(k, t) and bsd[k].shape == t.shape:
            y = t.to(torch.bfloat16).float()
            if precision == "exact":
                delta[k] = t.to(torch.bfloat16)
                rec = y
            else:
                d = y - bsd[k].to(torch.bfloat16).float()
                if precision == "bf16":
                    delta[k] = d.to(torch.bfloat16)
                    rec_d = delta[k].float()
                else:
                    bits = 8 if precision == "int8" else 4
                    q, s = quantize(d, bits, group)
                    delta[k + ".q"], delta[k + ".s"] = q, s
                    rec_d = dequantize(q, s, bits, d.shape[1])
                rec = (bsd[k].to(torch.bfloat16).float() + rec_d).to(torch.bfloat16).float()
            err2 += (rec - y).pow(2).sum().item()
            w2 += y.pow(2).sum().item()
            n_delta += t.numel()
        elif target is None and is_delta_key(k, t):
            continue  # the base language uses the base backbone as is
        else:
            verbatim[k] = t  # original dtype (an fp32 head stays fp32)
    files = {}
    if delta:
        files["delta"] = _write_st(delta, out / "delta.safetensors", zstd_level)
    files["lang"] = _write_st(verbatim, out / "lang.safetensors", zstd_level)
    tok_name = "tokenizer.model" if str(tokenizer).endswith(".model") else "tokenizer.json"   # SentencePiece or HF
    shutil.copyfile(tokenizer, out / tok_name)
    files["tokenizer"] = out / tok_name
    if config_yaml is not None:
        import yaml

        cfg = yaml.safe_load(Path(config_yaml).read_text())
        for k in ("weights_path", "weights_path_without_voice_cloning"):
            cfg.pop(k, None)
        (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        files["config"] = out / "config.yaml"
    manifest = {
        "format": FORMAT,
        "language": lang,
        "precision": precision if (target is not None or precision in WEIGHT_PRECISIONS) else "base",
        "group": group,
        "zstd_level": zstd_level,
        "base": base.describe(),
        "base_backbone_bf16_sha256": backbone_fingerprint(bsd),
        "target": None if target is None else target.describe(),
        "delta_params": n_delta,
        "backbone_rel_err": (err2 / w2) ** 0.5 if w2 else 0.0,
        "files": {k: {"name": p.name, "bytes": p.stat().st_size, "sha256": sha256_file(p)} for k, p in files.items()},
        "licence": {"weights": licence, "notice": notice or LICENCE_NOTICE},
    }
    manifest["total_bytes"] = sum(f["bytes"] for f in manifest["files"].values())
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


@dataclass
class Package:
    """A language package in host memory (tensors possibly pinned)."""

    lang: str
    manifest: dict
    delta: dict[str, torch.Tensor] = field(default_factory=dict)
    verbatim: dict[str, torch.Tensor] = field(default_factory=dict)
    tokenizer_path: str = ""
    config_path: str | None = None

    @property
    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for d in (self.delta, self.verbatim) for t in d.values())

    def delta_keys(self) -> list[str]:
        return sorted({k[:-2] if k.endswith((".q", ".s")) else k for k in self.delta})

    def pin(self) -> "Package":
        if torch.cuda.is_available():
            self.delta = {k: v.pin_memory() for k, v in self.delta.items()}
            self.verbatim = {k: v.pin_memory() for k, v in self.verbatim.items()}
        return self


def load_package(pkg_dir: str | os.PathLike, check_hashes: bool = False, pin: bool = False) -> Package:
    d = Path(pkg_dir)
    m = json.loads((d / "manifest.json").read_text())
    if m.get("format") != FORMAT:
        raise ValueError(f"unknown package format {m.get('format')!r}")
    if check_hashes:
        for key, f in m["files"].items():
            got = sha256_file(d / f["name"])
            if got != f["sha256"]:
                raise ValueError(f"{d.name}/{f['name']}: sha256 mismatch")
    tok = d / "tokenizer.json" if (d / "tokenizer.json").exists() else d / "tokenizer.model"
    p = Package(lang=m["language"], manifest=m, tokenizer_path=str(tok),
                config_path=str(d / "config.yaml") if (d / "config.yaml").exists() else None)
    df = _find(d, "delta.safetensors")
    if df is not None:
        p.delta = _read_st(df)
    p.verbatim = _read_st(_find(d, "lang.safetensors"))
    return p.pin() if pin else p


def reconstruct_tensor(key: str, base_w: torch.Tensor, pkg: Package, dtype: torch.dtype = torch.bfloat16,
                       out: torch.Tensor | None = None) -> torch.Tensor:
    """W_lang for one delta-coded key, computed on base_w's device; written into `out` when given."""
    prec = pkg.manifest["precision"]
    dev = base_w.device
    if prec == "exact":
        w = pkg.delta[key].to(dev, non_blocking=True)
    elif prec == "bf16":
        w = base_w.float() + pkg.delta[key].to(dev, non_blocking=True).float()
    elif prec in ("int8", "int4"):
        q = pkg.delta[key + ".q"].to(dev, non_blocking=True)
        s = pkg.delta[key + ".s"].to(dev, non_blocking=True)
        w = base_w.float() + dequantize(q, s, 8 if prec == "int8" else 4, base_w.shape[1])
    elif prec in WEIGHT_PRECISIONS:
        q = pkg.delta[key + ".q"].to(dev, non_blocking=True)
        s = pkg.delta[key + ".s"].to(dev, non_blocking=True)
        w = dequant_any(q, s, WEIGHT_PRECISIONS[prec], base_w.shape[1])
    else:
        raise ValueError(f"no delta for precision {prec!r}")
    if out is None:
        return w.to(dtype)
    out.copy_(w)
    return out


def reconstruct_state(base_sd: dict[str, torch.Tensor], pkg: Package, dtype: torch.dtype = torch.bfloat16) -> dict:
    """Full FlowLM state (flow_lm.* keys) for the package's language, on the base tensors' device."""
    out = {}
    dk = set(pkg.delta_keys())
    for k, b in base_sd.items():
        if not k.startswith("flow_lm."):
            continue
        if k in dk:
            out[k] = reconstruct_tensor(k, b, pkg, dtype)
        elif k in pkg.verbatim:
            out[k] = pkg.verbatim[k].to(b.device, dtype) if pkg.verbatim[k].is_floating_point() else pkg.verbatim[k]
        else:
            out[k] = b.to(dtype)
    return out


def verify_package(base_path: str, pkg_dir: str, target_path: str | None) -> dict[str, Any]:
    """Hashes, base fingerprint, and (if a target is given) a tensor comparison of the reconstruction."""
    pkg = load_package(pkg_dir, check_hashes=True)
    bsd = load_file(base_path)
    report: dict[str, Any] = {"language": pkg.lang, "precision": pkg.manifest["precision"], "hashes": "ok"}
    fp = backbone_fingerprint(bsd)
    report["base_matches"] = fp == pkg.manifest["base_backbone_bf16_sha256"]
    if not report["base_matches"]:
        return report
    if target_path:
        tsd = load_file(target_path)
        rec = reconstruct_state(bsd, pkg)
        exact = total = 0
        err2 = w2 = 0.0
        worst = (0.0, "")
        for k, y in tsd.items():
            if not k.startswith("flow_lm."):
                continue
            yb = y.to(torch.bfloat16) if y.is_floating_point() else y
            r = rec[k]
            total += 1
            exact += int(torch.equal(r, yb))
            if y.is_floating_point():
                e = (r.float() - yb.float()).pow(2).sum().item()
                n = yb.float().pow(2).sum().item()
                err2, w2 = err2 + e, w2 + n
                rel = (e / n) ** 0.5 if n else 0.0
                worst = max(worst, (rel, k))
        report.update(tensors_bit_exact=f"{exact}/{total}", flowlm_rel_err=(err2 / w2) ** 0.5,
                      worst_tensor=worst[1], worst_rel_err=worst[0])
    return report


def write_repo_card(root: str | os.PathLike, languages: list[dict]) -> Path:
    """README.md for a Hugging Face model repo holding several packages (not uploaded by this tool)."""
    rows = "\n".join(
        f"| {m['language']} | {m['precision']} | {m['total_bytes'] / 1e6:.1f} MB | "
        f"{(m['target'] or m['base'])['repo']}@{(m['target'] or m['base'])['revision'][:8]} |" for m in languages)
    base = languages[0]["base"]
    weight_only = all(m["precision"] in WEIGHT_PRECISIONS for m in languages)
    how = ("its transformer backbone as the language's own weights quantised to "
           + "/".join(sorted({m['precision'] for m in languages})) + " (per-group scales); no base backbone is "
           "needed at run time" if weight_only else
           "its transformer backbone as a difference from the base")
    text = f"""---
license: cc-by-4.0
library_name: pocket-tts
tags: [text-to-speech, pocket-tts, {'quantized' if weight_only else 'weight-delta'}]
---
# Pocket TTS language packages

Per-language packages for [Kyutai Pocket TTS](https://huggingface.co/kyutai/pocket-tts), built against one base
checkpoint: `{base['repo']}` @ `{base['revision']}` (`{base['file']}`, sha256 `{base['sha256'][:16]}…`).
Each package holds the language's tokenizer, text embedding and non-backbone tensors verbatim, and {how}.
The Mimi codec (and its voice-cloning encoder) is not included: it comes from Kyutai's base checkpoint, which you
download from Kyutai (accepting their terms).

| language | precision | size | derived from |
|---|---|---|---|
{rows}

## Licence and attribution

{LICENCE_NOTICE}
"""
    p = Path(root) / "README.md"
    p.write_text(text)
    return p
