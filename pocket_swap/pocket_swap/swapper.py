"""Multi-language Pocket TTS on one GPU: shared Mimi + base backbone, language slots materialised on request.

    sw = PocketSwapper(base_weights="model.safetensors", base_language="english_2026-09_24l",
                       packages="pocket-deltas/", n_slots=4, device="cuda", dtype=torch.bfloat16)
    sw.open("call-1", "italian_24l", voice="ref.wav")
    audio = sw.generate("call-1", "Buongiorno.")
    sw.open("call-1", "french_24l")          # switch only this session; its voice state is rebuilt
    sw.close("call-1")

Each slot is a full TTSModel whose FlowLM parameters are allocated once (fixed addresses while the slot stays
allocated, so CUDA graphs captured on a slot survive language swaps; `binding.alloc_gen` changes only if the slot is
freed by the idle timer and allocated again). The Mimi module is shared by all slots. Plain torch only: works on any
CUDA architecture (unified or discrete memory) and on CPU.
"""

from __future__ import annotations

import os
import tempfile
import threading
from pathlib import Path
from typing import Any, Hashable

import torch
from safetensors.torch import load_file

from .core import Binding, SwapCore, TierSpec
from .package import Package, backbone_fingerprint, is_delta_key, load_package, reconstruct_tensor


def _host_ram_bytes() -> int:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return 0


def resolve_cap(cap: int | float | None, total: int) -> int | None:
    """An int is bytes; a float in (0, 1] is a fraction of `total`; None means no cap."""
    if cap is None:
        return None
    if isinstance(cap, float) and 0 < cap <= 1:
        return int(cap * total)
    return int(cap)


class PocketBackend:
    """Torch side of SwapCore: slot allocation, package reads, in-place materialisation, voice states."""

    def __init__(self, base_weights: str, base_language: str, packages: str, device: str = "cuda",
                 dtype: torch.dtype = torch.bfloat16, ram_form: str = "packed", check_hashes: bool = False,
                 mode: str = "materialise", delta_precision: str | None = None, group: int = 128,
                 fused: bool | None = None):
        """mode "materialise": a slot holds a full backbone W_base + dequant(delta) (paid once per load).
        mode "ondemand": a slot holds only the compressed delta; every forward computes W_base + dequant(delta)
        (paid every step). delta_precision (bf16/int8/int4) must match the packages in ondemand mode."""
        from pocket_tts import TTSModel  # noqa: F401  (import check)
        from pocket_tts.utils.config import CONFIGS_DIR, load_config

        if ram_form not in ("packed", "materialised"):
            raise ValueError("ram_form is 'packed' or 'materialised'")
        if mode not in ("materialise", "ondemand"):
            raise ValueError("mode is 'materialise' or 'ondemand'")
        if mode == "ondemand" and (delta_precision not in ("bf16", "int8", "int4") or ram_form != "packed"):
            raise ValueError("ondemand mode needs delta_precision bf16/int8/int4 and packed host copies")
        self.mode, self.delta_precision, self.group, self.fused = mode, delta_precision, group, fused
        self.device, self.dtype, self.ram_form = torch.device(device), dtype, ram_form
        self.packages_root = Path(packages)
        self.check_hashes = check_hashes
        sd = load_file(base_weights)
        self.base_fingerprint = backbone_fingerprint(sd)
        # base backbone resident on the device (the source of every delta), in the slot dtype
        self.base = {k: v.to(self.device, dtype) for k, v in sd.items() if is_delta_key(k, v)}
        cfg_yaml = Path(base_language) if str(base_language).endswith((".yaml", ".yml")) else \
            Path(CONFIGS_DIR) / f"{base_language}.yaml"
        self._tmp = tempfile.TemporaryDirectory(prefix="pocket_swap_")
        self.base_config_path = self._local_config(cfg_yaml, base_weights)
        self.base_config = load_config(self.base_config_path)
        self.mimi = self._build_shared_mimi()
        self.slots: dict[int, Any] = {}
        self.lang_meta: dict[str, dict] = {}
        esz = torch.empty((), dtype=dtype).element_size()
        if mode == "materialise":
            self.slot_bytes = sum(v.numel() * (4 if (".flow_net." in k or ".out_eos." in k) else esz)
                                  for k, v in sd.items() if k.startswith("flow_lm."))
        else:
            from .quant import packed_nbytes

            other = sum(v.numel() * (4 if (".flow_net." in k or ".out_eos." in k) else esz)
                        for k, v in sd.items() if k.startswith("flow_lm.") and k not in self.base)
            bits = {"bf16": 16, "int8": 8, "int4": 4}[delta_precision]
            self.slot_bytes = other + sum(v.numel() * 2 if bits == 16 else packed_nbytes(*v.shape, bits, group)
                                          for v in self.base.values())
        del sd

    # -- construction helpers --------------------------------------------------------------------------
    def _local_config(self, yaml_path: Path, weights: str) -> str:
        import yaml

        cfg = yaml.safe_load(Path(yaml_path).read_text())
        cfg["weights_path"] = cfg["weights_path_without_voice_cloning"] = str(weights)
        tok = self.packages_root / self._base_lang_name(yaml_path) / "tokenizer.json"
        if tok.exists():
            cfg["flow_lm"]["lookup_table"]["tokenizer_path"] = str(tok)
        out = Path(self._tmp.name) / "base.yaml"
        out.write_text(yaml.safe_dump(cfg, sort_keys=False))
        return str(out)

    @staticmethod
    def _base_lang_name(yaml_path: Path) -> str:
        return Path(yaml_path).stem

    def _build_shared_mimi(self):
        from pocket_tts import TTSModel

        m = TTSModel.load_model(config=self.base_config_path)
        mimi = m.mimi.to(self.device).eval()
        mimi_dtype = next(mimi.parameters()).dtype
        dev_type = self.device.type
        orig_decode, orig_encode = mimi.decode_from_latent, mimi.encode_to_latent

        # Mimi always runs in its own dtype (fp32), outside any autocast region of the FlowLM.
        def decode_from_latent(latent, state, *a, **k):
            with torch.autocast(dev_type, enabled=False):
                return orig_decode(latent.to(mimi_dtype), state, *a, **k)

        def encode_to_latent(audio, *a, **k):
            with torch.autocast(dev_type, enabled=False):
                return orig_encode(audio.to(mimi_dtype), *a, **k)

        mimi.decode_from_latent = decode_from_latent
        mimi.encode_to_latent = encode_to_latent
        self._template_has_cloning = m.has_voice_cloning
        del m
        return mimi

    def _new_slot_model(self):
        from pocket_tts import TTSModel

        cfg = self.base_config

        dev_type, low = self.device.type, self.dtype != torch.float32

        class SlotTTS(TTSModel):
            """Backbone + text embedding in the slot dtype under autocast; sampler head and EOS head in fp32
            (upstream upcasts the backbone output to fp32 for them); Mimi in fp32."""

            def _encode_audio(self, audio):
                encoded = self.mimi.encode_to_latent(audio)
                w = self.flow_lm.speaker_proj_weight
                return torch.nn.functional.linear(encoded.to(w.dtype), w)

            def _run_flow_lm(self, *args, **kw):
                with torch.autocast(dev_type, dtype=self.flow_lm.dtype, enabled=low):
                    return super()._run_flow_lm(*args, **kw)

        def fp32_island(mod):
            mod.float()
            orig = mod.forward

            def forward(*a, **k):
                f = lambda x: x.float() if torch.is_tensor(x) and x.is_floating_point() else x  # noqa: E731
                with torch.autocast(dev_type, enabled=False):
                    return orig(*[f(x) for x in a], **{n: f(v) for n, v in k.items()})

            mod.forward = forward

        t = SlotTTS._from_pydantic_config(cfg, cfg.default_temperature, 1, None, cfg.default_eos_threshold, None)
        t.flow_lm.speaker_proj_weight = torch.nn.Parameter(torch.zeros(
            (cfg.flow_lm.transformer.d_model, cfg.mimi.inner_dim or cfg.mimi.seanet.dimension)))
        t.mimi = self.mimi
        self._deltas = getattr(self, "_deltas", {})
        if self.mode == "ondemand":
            from .ondemand import convert_backbone

            # drop the per-slot backbone weights before moving to the device: the slot keeps only deltas
            self._pending_deltas = convert_backbone(t.flow_lm, self.base, self.delta_precision, self.group,
                                                   fused=self.fused)
        t.flow_lm.to(self.device, self.dtype)
        t.flow_lm.dtype = self.dtype
        if low:
            fp32_island(t.flow_lm.flow_net)
            fp32_island(t.flow_lm.out_eos)
        t.has_voice_cloning = self._template_has_cloning
        t.eval()
        from pocket_tts.models.tts_model import stamp_state_names

        stamp_state_names(t)  # stateful modules need their absolute names (same tree in every slot)
        for p in t.flow_lm.parameters():
            p.requires_grad_(False)
        return t

    # -- Backend protocol -------------------------------------------------------------------------------
    def read_package(self, lang: str) -> tuple[Any, int]:
        pkg = load_package(self.packages_root / lang, check_hashes=self.check_hashes)
        if pkg.manifest["base_backbone_bf16_sha256"] != self.base_fingerprint:
            raise ValueError(f"package {lang} was built against a different base checkpoint")
        self._lang_meta(lang, pkg)
        if self.ram_form == "materialised":
            pkg = self._materialise_on_host(pkg)
        return pkg, pkg.nbytes

    def to_tier(self, pkg: Package, tier: str) -> tuple[Package, int]:
        if tier == "pinned":
            if self.device.type == "cuda":
                pkg = Package(pkg.lang, pkg.manifest, {k: v if v.is_pinned() else v.pin_memory() for k, v in pkg.delta.items()},
                              {k: v if v.is_pinned() else v.pin_memory() for k, v in pkg.verbatim.items()},
                              pkg.tokenizer_path, pkg.config_path)
        elif tier == "ram":
            if any(v.is_pinned() for v in pkg.delta.values()):
                pkg = Package(pkg.lang, pkg.manifest, {k: v.clone() for k, v in pkg.delta.items()},
                              {k: v.clone() for k, v in pkg.verbatim.items()}, pkg.tokenizer_path, pkg.config_path)
        else:
            raise ValueError(f"unknown host tier {tier!r}")
        return pkg, pkg.nbytes

    def _materialise_on_host(self, pkg: Package) -> Package:
        full = Package(lang=pkg.lang, manifest=dict(pkg.manifest, precision="exact"),
                       tokenizer_path=pkg.tokenizer_path, config_path=pkg.config_path)
        for k in pkg.delta_keys():
            full.delta[k] = reconstruct_tensor(k, self.base[k], pkg, self.dtype).cpu()
        full.verbatim = pkg.verbatim
        return full

    def _lang_meta(self, lang: str, pkg: Package) -> None:
        if lang in self.lang_meta:
            return
        from pocket_tts.modules.text_conditioner import JsonTokenizer
        from pocket_tts.utils.config import load_config

        nbins = self.base_config.flow_lm.lookup_table.n_bins
        meta = {"tokenizer": JsonTokenizer(nbins, pkg.tokenizer_path)}
        if pkg.config_path:
            import yaml

            raw = yaml.safe_load(Path(pkg.config_path).read_text())
            raw["weights_path"] = None
            raw["flow_lm"]["lookup_table"]["tokenizer_path"] = pkg.tokenizer_path
            p = Path(self._tmp.name) / f"{lang}.yaml"
            p.write_text(yaml.safe_dump(raw, sort_keys=False))
            c = load_config(p)
            meta.update(remove_semicolons=c.remove_semicolons, replace_characters=c.replace_characters or {},
                        append_terminal_punctuation=c.append_terminal_punctuation,
                        capitalize_first_letter=c.capitalize_first_letter,
                        pad_with_spaces_for_short_inputs=c.pad_with_spaces_for_short_inputs,
                        model_recommended_frames_after_eos=c.model_recommended_frames_after_eos,
                        temp=c.default_temperature, eos_threshold=c.default_eos_threshold)
        self.lang_meta[lang] = meta

    def allocate_slot(self, index: int) -> None:
        self.slots[index] = self._new_slot_model()
        if self.mode == "ondemand":
            self._deltas[index] = self._pending_deltas

    def free_slot(self, index: int) -> None:
        self.slots.pop(index, None)
        self._deltas.pop(index, None) if hasattr(self, "_deltas") else None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    @torch.no_grad()
    def materialise(self, index: int, lang: str, package: Package) -> None:
        tts = self.slots[index]
        dk = set(package.delta_keys())
        if self.mode == "ondemand":
            prec = package.manifest["precision"]
            if prec not in ("base", self.delta_precision):
                raise ValueError(f"{lang}: package precision {prec} != ondemand precision {self.delta_precision}")
            for k, dl in self._deltas[index].items():
                if k not in dk:
                    dl.load(None)
                elif prec == "bf16":
                    dl.load(package.delta[k])
                else:
                    dl.load(package.delta[k + ".q"], package.delta[k + ".s"])
        for name, p in tts.flow_lm.state_dict(keep_vars=True).items():  # persistent tensors = checkpoint keys
            k = "flow_lm." + name
            if k in dk:
                reconstruct_tensor(k, self.base[k], package, self.dtype, out=p.data)
            elif k in package.verbatim:
                p.data.copy_(package.verbatim[k])
            elif k in self.base:
                p.data.copy_(self.base[k])
            else:
                raise KeyError(f"{lang}: no source for {k}")
        meta = self.lang_meta[lang]
        tts.flow_lm.conditioner.tokenizer = meta["tokenizer"]
        for attr in ("remove_semicolons", "replace_characters", "append_terminal_punctuation",
                     "capitalize_first_letter", "pad_with_spaces_for_short_inputs",
                     "model_recommended_frames_after_eos", "temp", "eos_threshold"):
            if attr in meta:
                setattr(tts, attr, meta[attr])
        tts._cached_get_state_for_audio_prompt.cache_clear()  # its cache is keyed by model instance
        if self.device.type == "cuda":
            torch.cuda.current_stream(self.device).synchronize()

    def voice_state(self, index: int, lang: str, voice: Any) -> Any:
        return self.slots[index].get_state_for_audio_prompt(voice)


class PocketSwapper:
    """Public API: sessions bound to languages; generation serialised per slot, parallel across slots."""

    def __init__(self, base_weights: str, base_language: str, packages: str, n_slots: int = 4,
                 device: str = "cuda", dtype: torch.dtype = torch.bfloat16,
                 vram_cap: int | float | None = None,
                 pinned_cap: int | float | None = None, pinned_max_languages: int | None = 8,
                 ram_cap: int | float | None = None, ram_max_languages: int | None = 0,
                 host_idle_s: float | None = None, ram_form: str = "packed",
                 slot_idle_s: float | None = None, check_hashes: bool = False,
                 mode: str = "materialise", delta_precision: str | None = None, fused: bool | None = None):
        """Caps: an int is bytes, a float in (0, 1] a fraction of VRAM / host RAM, None = unlimited.
        A tier with max_languages = 0 is disabled (pinned: page-locked host copies; ram: pageable copies)."""
        self.backend = PocketBackend(base_weights, base_language, packages, device, dtype, ram_form, check_hashes,
                                     mode, delta_precision, fused=fused)
        if self.backend.device.type == "cuda":
            free, total = torch.cuda.mem_get_info(self.backend.device)
            cap = resolve_cap(vram_cap, total)
            if cap is not None:
                n_slots = max(1, min(n_slots, cap // self.backend.slot_bytes))
        host = _host_ram_bytes()
        tiers = []
        if pinned_max_languages != 0 and pinned_cap != 0:
            tiers.append(TierSpec("pinned", resolve_cap(pinned_cap, host), pinned_max_languages, host_idle_s))
        if ram_max_languages != 0 and ram_cap != 0:
            tiers.append(TierSpec("ram", resolve_cap(ram_cap, host), ram_max_languages, host_idle_s))
        self.core = SwapCore(self.backend, n_slots=n_slots, tiers=tiers, slot_idle_s=slot_idle_s)
        self._slot_locks = [threading.Lock() for _ in range(n_slots)]
        self._voices: dict[Hashable, tuple[Any, Hashable]] = {}

    def open(self, session: Hashable, lang: str, voice: Any = None, voice_key: Hashable | None = None,
             timeout: float | None = 0.0) -> Binding:
        """Bind a session to a language (switches it if already bound). Keeps the session's voice on switch."""
        if voice is None and session in self._voices:
            voice, voice_key = self._voices[session]
        if voice is not None:
            voice_key = voice_key if voice_key is not None else (str(voice) if isinstance(voice, (str, Path)) else id(voice))
            self._voices[session] = (voice, voice_key)
        return self.core.bind(session, lang, voice=voice, voice_key=voice_key, timeout=timeout)

    def close(self, session: Hashable) -> None:
        self.core.unbind(session)
        self._voices.pop(session, None)

    def model(self, session: Hashable):
        return self.backend.slots[self.core.binding(session).slot]

    def lock(self, session: Hashable) -> threading.Lock:
        return self._slot_locks[self.core.binding(session).slot]

    def generate(self, session: Hashable, text: str, **kw) -> torch.Tensor:
        b = self.core.binding(session)
        with self._slot_locks[b.slot]:
            kw.setdefault("copy_state", True)  # the cached voice state is a shared template: never mutate it
            if not kw["copy_state"]:
                raise ValueError("copy_state=False would mutate a voice state shared by sessions")
            return self.backend.slots[b.slot].generate_audio(b.voice_state, text, **kw)

    def maintenance(self) -> dict:
        return self.core.maintenance()
