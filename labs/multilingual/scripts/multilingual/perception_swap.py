"""Swap VoiceChat's speech front end inside a live NVIDIA streaming wrapper (V2 ReferenceEngine), in place.

What can be swapped (all observational of the rest of the model: the LLM, fusion, FC and TTS are untouched):
  * encoder   : the 24-layer FastConformer weights (VC <-> standalone EN / multilingual ML), optionally with a
                different left attention context (ML was trained with 56, VC with 70). Changing the context rebuilds
                the wrapper's cache-aware perception manager (and its CUDA graphs).
  * interface : `perception.proj` (1024 -> 4480) replaced by (a) VC proj, (b) an affine map folded into proj,
                (c) an arbitrary torch module (adapter) placed in front of proj.
  * rnnt      : the user-ASR branch that drives turn-taking (EOU/BOU barge-in, wrapper:_rnnt_step). With the ML
                encoder, VC's RNNT reads features it was not trained on, so the ML decoder/joint (13087 tokens) can be
                transplanted together with the ML language-prompt MLP (`prompt_kernel`) in front of the joint.

Weights are copied with `param.data.copy_` so tensors keep their storage (CUDA graphs captured on them stay
valid). `snapshot()` / `restore()` bring the original VoiceChat front end back.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch
import yaml
from omegaconf import OmegaConf


def load_nemo_dir(d: str | os.PathLike) -> tuple[dict, dict]:
    d = Path(d)
    sd = torch.load(d / "model_weights.ckpt", map_location="cpu", weights_only=True)
    cfg = yaml.safe_load(open(d / "model_config.yaml"))
    return sd, cfg


class PromptJoint(torch.nn.Module):
    """RNNT joint with the multilingual language prompt applied to the encoder frame first
    (rnnt_bpe_models_prompt.py forward: cat([enc, onehot(lang)]) -> prompt_kernel -> joint)."""

    def __init__(self, joint, prompt_kernel, num_prompts: int, lang_idx: int):
        super().__init__()
        self.inner = joint
        self.prompt_kernel = prompt_kernel
        self.num_prompts = num_prompts
        self.lang_idx = lang_idx
        self.vocabulary = joint.vocabulary
        self.take = None  # dual-encoder mode: keep only the first `take` channels (= the ORIGINAL encoder output)

    def joint(self, f: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        if self.take is not None:
            f = f[..., : self.take]
        oh = torch.zeros(*f.shape[:-1], self.num_prompts, dtype=f.dtype, device=f.device)
        oh[..., self.lang_idx] = 1
        return self.inner.joint(self.prompt_kernel(torch.cat([f, oh], -1)), g)


from nemo.collections.asr.parts.mixins.streaming import StreamingEncoder  # noqa: E402


class DualEncoder(torch.nn.Module, StreamingEncoder):
    """Two cache-aware encoders run side by side (E5): `orig` = the frozen ML encoder whose output feeds the RNNT
    (turn-taking / user ASR unchanged by construction), `mod` = the same encoder with trained LayerNorms whose output
    feeds the LLM path. Subclasses StreamingEncoder because PerceptionCacheManager.setup() checks isinstance
    (perception_cache.py:131). Output = cat([orig, mod], channel dim) -> SliceConnector (modality_adapter) keeps `mod` for
    proj; PromptJoint.take keeps `orig` for the RNNT. Streaming caches are concatenated on the layer dim."""

    def __init__(self, orig, mod):
        super().__init__()
        self.orig, self.mod = orig, mod
        self.L = len(orig.layers)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__("orig"), name)

    def set_default_att_context_size(self, ctx):
        self.orig.set_default_att_context_size(ctx)
        self.mod.set_default_att_context_size(ctx)

    def setup_streaming_params(self, *a, **k):
        self.orig.setup_streaming_params(*a, **k)
        self.mod.setup_streaming_params(*a, **k)

    def get_initial_cache_state(self, *a, **k):
        c1, t1, l1 = self.orig.get_initial_cache_state(*a, **k)
        c2, t2, _ = self.mod.get_initial_cache_state(*a, **k)
        return torch.cat([c1, c2], 0), torch.cat([t1, t2], 0), l1

    def _split(self, x):
        return (None, None) if x is None else (x[: self.L], x[self.L:])

    def forward(self, audio_signal, length=None, cache_last_channel=None, cache_last_time=None,
                cache_last_channel_len=None, **kw):
        c1, c2 = self._split(cache_last_channel)
        t1, t2 = self._split(cache_last_time)
        r1 = self.orig(audio_signal=audio_signal, length=length, cache_last_channel=c1, cache_last_time=t1,
                       cache_last_channel_len=cache_last_channel_len, **kw)
        r2 = self.mod(audio_signal=audio_signal, length=length, cache_last_channel=c2, cache_last_time=t2,
                      cache_last_channel_len=cache_last_channel_len, **kw)
        if len(r1) == 2:
            return torch.cat([r1[0], r2[0]], 1), r1[1]
        return (torch.cat([r1[0], r2[0]], 1), r1[1], torch.cat([r1[2], r2[2]], 0), torch.cat([r1[3], r2[3]], 0),
                *r1[4:])

    def cache_aware_stream_step(self, processed_signal, processed_signal_length=None, cache_last_channel=None,
                                cache_last_time=None, cache_last_channel_len=None, keep_all_outputs=True,
                                drop_extra_pre_encoded=None):
        c1, c2 = self._split(cache_last_channel)
        t1, t2 = self._split(cache_last_time)
        kw = dict(processed_signal=processed_signal, processed_signal_length=processed_signal_length,
                  cache_last_channel_len=cache_last_channel_len, keep_all_outputs=keep_all_outputs,
                  drop_extra_pre_encoded=drop_extra_pre_encoded)
        r1 = self.orig.cache_aware_stream_step(cache_last_channel=c1, cache_last_time=t1, **kw)
        r2 = self.mod.cache_aware_stream_step(cache_last_channel=c2, cache_last_time=t2, **kw)
        cat0 = lambda a, b: None if a is None else torch.cat([a, b], 0)  # noqa: E731
        return torch.cat([r1[0], r2[0]], 1), r1[1], cat0(r1[2], r2[2]), cat0(r1[3], r2[3]), r1[4]


class SliceConnector(torch.nn.Module):
    """modality_adapter for DualEncoder: [B, 2D, T] -> the `mod` half [B, D, T] (IdentityConnector otherwise)."""

    def __init__(self, d):
        super().__init__()
        self.d = d

    def forward(self, audio_signal, length):
        return audio_signal[:, self.d:], length


class FrontEnd:
    def __init__(self, engine):
        self.eng = engine
        self.w = engine.wrapper
        self.stt = self.w.model.stt_model
        self.enc = self.stt.perception.encoder
        self.dev = next(self.enc.parameters()).device
        self._orig = None
        self.adapter = None
        self.state = {"encoder": "vc", "proj": "vc", "rnnt": "vc", "att_context": list(self.enc.att_context_size)}

    # ------------------------------------------------------------------ snapshot
    def snapshot(self):
        self._orig = {
            "enc": {k: v.detach().cpu().clone() for k, v in self.enc.state_dict().items()},
            "proj": {k: v.detach().cpu().clone() for k, v in self.stt.perception.proj.state_dict().items()},
            "rnnt": (self.stt._rnnt_decoder, self.stt._rnnt_joint, self.stt._rnnt_blank_id),
            "att": list(self.enc.att_context_size),
            "proj_mod": self.stt.perception.proj,
            "enc_mod": self.stt.perception.encoder,
            "ma_mod": self.stt.perception.modality_adapter,
        }

    def restore(self):
        o = self._orig
        dual = self.stt.perception.encoder is not o["enc_mod"]
        self.stt.perception.encoder = o["enc_mod"]
        self.stt.perception.modality_adapter = o["ma_mod"]
        self._copy(self.enc, o["enc"])
        swapped_mod = self.stt.perception.proj is not o["proj_mod"]
        self.stt.perception.proj = o["proj_mod"]
        self._copy(self.stt.perception.proj, o["proj"])
        self._set_rnnt(*o["rnnt"])
        if list(self.enc.att_context_size) != o["att"]:
            self.set_att_context(o["att"])
        elif swapped_mod or dual:
            self._recapture()
        self.state = {"encoder": "vc", "proj": "vc", "rnnt": "vc", "att_context": o["att"]}

    @staticmethod
    def _copy(mod: torch.nn.Module, sd: dict):
        own = mod.state_dict()
        missing = sorted(set(own) - set(sd))
        extra = sorted(set(sd) - set(own))
        if missing or extra:
            raise KeyError(f"state mismatch: missing={missing[:5]} extra={extra[:5]}")
        with torch.no_grad():
            for k, v in own.items():
                if v.shape != sd[k].shape:
                    raise ValueError(f"{k}: {tuple(v.shape)} vs {tuple(sd[k].shape)}")
                v.copy_(sd[k].to(v.dtype))

    # ------------------------------------------------------------------ encoder
    def set_encoder(self, sd_full: dict, tag: str):
        sd = {k[len("encoder."):]: v for k, v in sd_full.items() if k.startswith("encoder.")}
        self._copy(self.enc, sd)
        self.state["encoder"] = tag

    def set_att_context(self, ctx: list[int]):
        """Change the encoder's left/right attention context and rebuild the cache-aware perception manager."""
        self.enc.set_default_att_context_size(list(ctx))
        self.enc.setup_streaming_params()
        self._recapture()
        self.state["att_context"] = list(ctx)

    def _recapture(self):
        """Rebuild the cache-aware perception manager. Its CUDA graphs capture encoder + modality_adapter + **proj**
        (perception_cache.py:257-311), so a replaced proj module or a new attention context needs a new capture."""
        from nemo.collections.speechlm2.inference.model_wrappers.perception_cache import PerceptionCacheManager
        if getattr(self.w, "use_perception_cache", False):
            mgr = PerceptionCacheManager(model=self.w.model, device=self.w.device, dtype=self.w.dtype,
                                         use_cudagraph=self.w._use_perception_cudagraph,
                                         pad_and_drop_preencoded=self.w._pad_and_drop_preencoded)
            if not mgr.setup():
                raise RuntimeError("perception cache setup failed after att_context change")
            self.w.perception_cache_mgr = mgr

    def set_dual_ln(self, ln_sd: dict | None, tag: str):
        """E5: after set_encoder(ML) (+ att context), keep that encoder for the RNNT and add a copy whose LayerNorm
        params are replaced by `ln_sd` (None = identical copy, the parity control) for the LLM path."""
        import copy
        orig = self.stt.perception.encoder
        mod = copy.deepcopy(orig)
        if ln_sd:
            own = mod.state_dict()
            bad = [k for k in ln_sd if k not in own]
            if bad:
                raise KeyError(f"unknown LN keys {bad[:5]}")
            with torch.no_grad():
                for k, v in ln_sd.items():
                    own[k].copy_(v.to(own[k].dtype))
        self.stt.perception.encoder = DualEncoder(orig, mod).eval()
        self.stt.perception.modality_adapter = SliceConnector(orig.d_model)
        j = self.stt._rnnt_joint
        if isinstance(j, PromptJoint):
            j.take = orig.d_model
        else:
            raise RuntimeError("dual encoder needs the ML RNNT (PromptJoint) set first")
        self._recapture()
        self.state["encoder"] += f"+dual_ln:{tag}"

    def refresh_silence_embedding(self) -> str | None:
        """The FC fast path feeds `wrapper._silence_embedding_cache` (1 s of digital silence through the CURRENT
        perception, computed once at engine init, wrapper:1028-1053) instead of user audio. After a front-end swap it
        would still hold the original VC front-end's embedding -> recompute it through the active front-end.
        Returns a short hash of the new embedding (recorded per session)."""
        if not hasattr(self.w, "_create_silence_embedding_cache") or getattr(self.w, "_silence_embedding_cache", None) is None:
            return None
        self.w._create_silence_embedding_cache()
        import hashlib
        v = self.w._silence_embedding_cache.detach().float().cpu().numpy().tobytes()
        h = hashlib.sha256(v).hexdigest()[:12]
        self.state["silence_emb"] = h
        return h

    # ------------------------------------------------------------------ interface
    def set_proj_affine(self, M: torch.Tensor | None, m: torch.Tensor | None, tag: str):
        """Fold an affine map into proj: with y = x @ M + m (row vectors, source space -> VC encoder space),
        proj(y) = x @ (W0 M^T)^T + (W0 m + b0). Copied in place, so the captured CUDA graphs stay valid."""
        o = self._orig["proj"]
        W0, b0 = o["weight"].double(), o["bias"].double()
        W = W0 if M is None else W0 @ M.double().T
        b = b0 if m is None else W0 @ m.double() + b0
        self.stt.perception.proj = self._orig["proj_mod"]
        self._copy(self.stt.perception.proj, {"weight": W.float(), "bias": b.float()})
        self.state["proj"] = tag

    def set_adapter(self, module: torch.nn.Module, tag: str):
        """Insert `module` (1024 -> 1024, applied per frame on [.., 1024]) before the original VC proj."""
        self._copy(self._orig["proj_mod"], self._orig["proj"])
        mod = torch.nn.Sequential(module, self._orig["proj_mod"]).to(self.dev, next(self._orig["proj_mod"].parameters()).dtype).eval()
        self.stt.perception.proj = mod
        self._recapture()
        self.state["proj"] = tag

    # ------------------------------------------------------------------ rnnt
    def _set_rnnt(self, dec, joint, blank):
        object.__setattr__(self.stt, "_rnnt_decoder", dec)
        object.__setattr__(self.stt, "_rnnt_joint", joint)
        object.__setattr__(self.stt, "_rnnt_blank_id", blank)
        self.stt.rnnt_decoder, self.stt.rnnt_joint = dec, joint
        self._rebuild_vocab_tables(joint.vocabulary)

    def set_rnnt_ml(self, sd: dict, cfg: dict, lang: str = "auto"):
        from nemo.collections.asr.modules.rnnt import RNNTDecoder, RNNTJoint
        dcfg = OmegaConf.create(cfg["decoder"])
        jcfg = OmegaConf.create(cfg["joint"])
        dec = RNNTDecoder.from_config_dict(dcfg)
        jnt = RNNTJoint.from_config_dict(jcfg)
        self._copy(dec, {k[len("decoder."):]: v for k, v in sd.items() if k.startswith("decoder.")})
        self._copy(jnt, {k[len("joint."):]: v for k, v in sd.items() if k.startswith("joint.")})
        nprompt = int(cfg.get("num_prompts", cfg["model_defaults"].get("num_prompts", 128)))
        d = cfg["model_defaults"]["enc_hidden"]
        pk = torch.nn.Sequential(torch.nn.Linear(d + nprompt, 2 * d), torch.nn.ReLU(), torch.nn.Linear(2 * d, d))
        self._copy(pk, {k[len("prompt_kernel."):]: v for k, v in sd.items() if k.startswith("prompt_kernel.")})
        lang_idx = int(cfg["model_defaults"]["prompt_dictionary"][lang])
        dt = next(self._orig["rnnt"][1].parameters()).dtype
        pj = PromptJoint(jnt, pk, nprompt, lang_idx).to(self.dev, dt).eval()
        dec = dec.to(self.dev, dt).eval()
        self._set_rnnt(dec, pj, len(jnt.vocabulary))
        self.state["rnnt"] = f"ml:{lang}"

    def _rebuild_vocab_tables(self, vocab):
        """Mirror wrapper.__init__ (nemotron_voicechat_inference_wrapper.py:946-985) for a new RNNT vocabulary."""
        w = self.w
        vocab = list(vocab)
        w._rnnt_is_start_tokens = [t.startswith("▁") for t in vocab]
        w._rnnt_unk_id = next((i for i, t in enumerate(vocab) if t in ("<unk>", "⁇")), None)
        w._rnnt_punct_ids, w._rnnt_punct_ids_set, w._rnnt_punct_bias_increments = [], set(), {}
        mc = w.model_cfg
        if mc.get("rnnt_punct_bias_enabled", False):
            inc_cfg = mc.get("rnnt_punct_bias_increments", None)
            inc_def = float(mc.get("rnnt_punct_bias_increment", 2.5))
            for pt in mc.get("rnnt_punct_bias_tokens", [".", ",", "?", "!"]):
                for var in (pt, "▁" + pt):
                    if var in vocab:
                        pid = vocab.index(var)
                        if pid not in w._rnnt_punct_ids_set:
                            w._rnnt_punct_ids.append(pid)
                            w._rnnt_punct_ids_set.add(pid)
                            w._rnnt_punct_bias_increments[pid] = float(inc_cfg[pt]) if (inc_cfg and pt in inc_cfg) else inc_def
