"""REFERENCE backend: the NVIDIA NeMo Speech streaming pipeline (native PyTorch engine), isolated behind our API.

Source: NeMo Speech branch nemotron-labs-voicechat @ 097dfe9e
  nemo/collections/speechlm2/inference/pipelines/streaming_s2s_pipeline.py   (StreamingS2SPipeline)
  nemo/collections/speechlm2/inference/model_wrappers/nemotron_voicechat_inference_wrapper.py
Driven exactly like the NVIDIA Triton backend (examples/.../triton/model_repo_s2s/voicechat/1/infer_streaming.py):
prefill-only frame with the system prompt, then one ``generate_step([Frame])`` per audio chunk, output read from
the per-stream state and ``cleanup_after_response()``.

Nothing here changes model inputs. Instrumentation is observational only:
  * ``_run_fc_async_steps`` is wrapped at instance level to copy the FC sub-timeline tokens it writes into the
    thread-local snapshot tensors (the main context never receives them, streaming_s2s_pipeline.py:1040-1062);
  * the tool registry entries are our ledger-wrapped handlers with the reference signature ``f(args) -> str``.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch
from omegaconf import OmegaConf

from mynah_v2.model import tokens as T

DEFAULT_CKPT = Path(os.environ.get("MYNAH_HOME", Path.home() / "mynah-s2s")) / "hf-cache/models--nvidia--NVIDIA-NemotronLabs-VoiceChat-11B/snapshots/443794ea956ef0065f001967ffd00e77f519cb39"
DEFAULT_NEMO = Path(os.environ.get("MYNAH_HOME", Path.home() / "mynah-s2s")) / "third_party/agent/nemo-speech"
OFFICIAL_SNAPSHOT = "443794ea956ef0065f001967ffd00e77f519cb39"   # NVIDIA-NemotronLabs-VoiceChat-11B (HF revision)


def build_cfg(ckpt: str | os.PathLike = DEFAULT_CKPT, nemo_dir: str | os.PathLike = DEFAULT_NEMO,
              output_dir: str = "/tmp/mynah-v2-ref", overrides: dict | None = None):
    """NVIDIA's own streaming config (s2s_streaming.yaml) with the minimum needed for the native engine."""
    base = OmegaConf.load(Path(nemo_dir) / "examples/speechlm2/nemo_inference_pipelines/conf/s2s_streaming.yaml")
    OmegaConf.set_struct(base, False)
    ckpt = str(ckpt)
    base.audio_file = ""
    base.output_dir = output_dir
    s = base.s2s
    s.model_path = ckpt
    s.llm_checkpoint_path = ckpt
    s.speaker_reference = ""
    s.speaker_name = "Aria"            # pre-baked latent in the checkpoint (config.json inference_speaker_name)
    s.engine_type = "native"
    s.system_prompt = ""
    for k, v in (overrides or {}).items():
        OmegaConf.update(base, k, v, force_add=True)
    return base


class lean_checkpoint_loading:
    """Context manager for the reference build on GB10 unified memory (121 GB shared CPU/GPU).

    The NVIDIA wrapper (`_initialize_model`, wrapper:572-615) calls safetensors ``load_file`` on the same 42 GB
    FP32 checkpoint twice: call 1 keeps only keys NOT under tts_model./speech_generation. (native engine), call 2
    keeps only keys under tts_model.; the second full dict stays alive through ``model.to(device)`` (wrapper:685).
    Peak = model 42 + dict 42 + GPU copy 42 GB -> CUDA OOM on GB10 (H100 hosts have separate RAM).
    Here each call loads only the keys the wrapper retains, which is exactly the subset it would keep after its
    own filtering, so the resulting model is identical. Any third call or other path falls back to the original.
    """

    def __init__(self, ckpt: str):
        self.path = os.path.realpath(os.path.join(ckpt, "model.safetensors"))
        self.calls = 0
        self.loaded = []

    def __enter__(self):
        import safetensors.torch as st
        self._st = st
        self._orig = st.load_file

        def lean(filename, device="cpu"):
            if os.path.realpath(str(filename)) != self.path or self.calls >= 2:
                return self._orig(filename, device=device)
            self.calls += 1
            keep_tts = self.calls == 2
            from safetensors import safe_open
            out = {}
            with safe_open(filename, framework="pt", device=str(device)) as f:
                for k in f.keys():
                    is_tts = k.startswith("tts_model.") or k.startswith("speech_generation.")
                    if (keep_tts and k.startswith("tts_model.")) or (not keep_tts and not is_tts):
                        out[k] = f.get_tensor(k)
            self.loaded.append(len(out))
            return out

        st.load_file = lean
        return self

    def __exit__(self, *exc):
        self._st.load_file = self._orig
        try:  # drop the checkpoint's page cache (we own the file; no root needed)
            fd = os.open(self.path, os.O_RDONLY)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            os.close(fd)
        except OSError:
            pass
        return False


@dataclass
class FCRecord:
    """One call to the wrapper's FC fast path (phase 1 = call tokens, phase 2 = response injection)."""
    phase: int
    start_frame: int
    steps: int
    text: list[int]
    func: list[int]
    tool_response_text: str | None
    t_start_ns: int
    t_end_ns: int
    # tokens at start_frame-1. SOTC is sometimes predicted by the regular step there, before the fast path is entered
    # (measured: NVIDIA FC sample SOTC 96 / current_frame_idx 97), sometimes it is the fast path's first frame (en_s2/s4)
    pre_text: int | None = None
    pre_func: int | None = None


@dataclass
class StepOutput:
    audio: torch.Tensor                     # [N] float32 @ 22.05 kHz, new audio since previous step
    text_delta: str
    asr_delta: str
    func_delta: str
    frame_idx: int                          # Thinker frame index after this step (main timeline)
    new_text_ids: list[int]
    new_func_ids: list[int]
    fc_active: bool                         # FC background thread running after this step
    fc_records: list[FCRecord] = field(default_factory=list)
    latency_ms: float = 0.0


def fc_record(args: tuple, kw: dict, out, t0_ns: int) -> FCRecord | None:
    """FCRecord of one ``_run_fc_async_steps`` call (``out`` = its return value); None if it cannot be read.
    Instrumentation must never break the reference, hence the blanket except."""
    try:
        steps = int(out[0])
        start = int(kw.get("current_frame_idx", args[4] if len(args) > 4 else 0))
        gt, gf = kw.get("gen_text"), kw.get("gen_function_text")
        lo, hi = start, start + steps + 1
        return FCRecord(
            phase=2 if kw.get("tool_response_text") else 1, start_frame=start, steps=steps,
            text=gt[0, lo:hi].tolist() if gt is not None else [],
            func=gf[0, lo:hi].tolist() if gf is not None else [],
            tool_response_text=kw.get("tool_response_text"), t_start_ns=t0_ns, t_end_ns=time.monotonic_ns(),
            pre_text=int(gt[0, lo - 1]) if gt is not None and lo > 0 else None,
            pre_func=int(gf[0, lo - 1]) if gf is not None and lo > 0 else None)
    except Exception:  # noqa: BLE001
        return None


class ReferenceEngine:
    """Single-stream NVIDIA reference pipeline. One instance holds the model (~11B params)."""

    def __init__(self, cfg=None, tools: dict[str, Callable[[dict], str]] | None = None, keep_builtin_tools=True,
                 llm_cache: bool = True, fc_ctx_fix: str | None = None):
        from nemo.collections.speechlm2.inference.factory.s2s_pipeline_builder import S2SPipelineBuilder

        self.cfg = cfg if cfg is not None else build_cfg()
        torch.set_float32_matmul_precision(self.cfg.get("matmul_precision", "medium"))
        t0 = time.monotonic()
        with lean_checkpoint_loading(str(self.cfg.s2s.llm_checkpoint_path)) as lean:
            self.pipeline = S2SPipelineBuilder.build_pipeline(self.cfg)
        self.lean_load_keys = lean.loaded
        self.load_s = time.monotonic() - t0
        self.wrapper = self.pipeline.s2s_model
        self.tokenizer = self.wrapper.tokenizer
        bad = T.verify_against(self.tokenizer)
        if bad:
            raise RuntimeError(f"special token ids differ from mynah_v2.model.tokens: {bad}")
        if not keep_builtin_tools:
            self.pipeline.tool_registry.clear()
        if tools:
            self.pipeline.tool_registry.update(tools)
        from mynah_v2.model.session_tools import SessionToolBinding
        # base registry = builtins + constructor tools; per-session handlers are bound/unbound around each session
        self.session_tools = SessionToolBinding(self.pipeline.tool_registry)
        # per-session call guard (agent.call_guard; attached by the session layer, default ON in ENHANCED) and the FC
        # resume contract of guarded streams (read by fc_ctx_fix from the FC thread, keyed by stream id)
        self._v2_guard_by_stream: dict = {}
        self._v2_resume_by_stream: dict = {}
        self._v2_guard_summaries: dict = {}
        self.fc_still_running = 0
        self.fc_records: list[FCRecord] = []
        self._fc_lock = threading.Lock()
        self._instrument_fc()
        # Unknown tool / unparsable call: the reference executor returns None and NOTHING is injected
        # (streaming_s2s_pipeline.py:767-822 + 1795 `if tool_response:`). What to answer instead is application
        # policy, owned by the controller: it sets ``unknown_tool_handler(call_text) -> str | None`` (ENHANCED).
        # The adapter only routes the model's request and records it.
        self.unknown_tool_handler: Callable[[str], str | None] | None = None
        # A call text holding SEVERAL calls: the reference executes only the first one (P:790 `call = parsed[0]`) and
        # the others are dropped. ``multi_call_handler(call_text) -> str`` (controller policy, ENHANCED) gets the
        # whole list instead.
        self.multi_call_handler: Callable[[str], str | None] | None = None
        self.unknown_tool_events: list[dict] = []
        self.multi_call_events: list[dict] = []
        self._wrap_executor()
        self.fc_ctx_fix_info = self._default_fc_ctx_fix(fc_ctx_fix)
        self.chunk_samples = int(self.pipeline.chunk_size_in_secs * self.pipeline.input_sample_rate)
        self.stream_id = 0
        self._started = False
        self._pos = {"text": 0, "asr": 0, "func": 0}
        self._last_frame_idx = 0
        self._rec_cursor = 0
        # Incremental Thinker (mynah_v2.model.llm_cache): ON by default since 2026-11-11 — token-identical to the
        # reference full-history recompute under deterministic decoding (scripts/cache_parity.py, 3 inputs up to
        # 1102 frames). ``llm_cache=False`` restores NVIDIA's shipped no-cache path, kept as the diagnostic oracle.
        self.llm_cache_info: dict | None = None
        if llm_cache:
            from mynah_v2.model.llm_cache import enable_hybrid_llm_cache
            self.llm_cache_info = enable_hybrid_llm_cache(self)

    def set_tts_pad_silence(self, on: bool) -> None:
        """NVIDIA's own idle-silence substitution (wrapper ``inference_force_speech_silence_on_pad``, normally enabled
        by the env var S2S_INFERENCE_FORCE_SPEECH_SILENCE_ON_PAD; off in the default config): while the agent has no
        open turn, PAD text frames get the codec's fixed silence codes (paper §2.3 "fixed silence frame") instead
        of EarTTS output, which otherwise can babble after long PAD stretches. Read per step from ``model.cfg``."""
        cfg = self.wrapper.model.cfg
        from omegaconf import open_dict
        try:
            with open_dict(cfg):
                cfg["inference_force_speech_silence_on_pad"] = bool(on)
        except Exception:  # noqa: BLE001 - plain dict config
            cfg["inference_force_speech_silence_on_pad"] = bool(on)

    def tts_pad_silence(self) -> bool:
        return bool(self.wrapper.model.cfg.get("inference_force_speech_silence_on_pad", False))

    def set_llm_cache(self, on: bool) -> dict | None:
        """Switch the Thinker cache for the next session (between sessions only)."""
        from mynah_v2.model.llm_cache import disable_hybrid_llm_cache, enable_hybrid_llm_cache
        if on:
            self.llm_cache_info = enable_hybrid_llm_cache(self)
        else:
            disable_hybrid_llm_cache(self)
            self.llm_cache_info = None
        return self.llm_cache_info

    # -------------------------------------------------------------------------------------------------
    def _instrument_fc(self) -> None:
        orig = self.wrapper._run_fc_async_steps
        records, lock = self.fc_records, self._fc_lock

        def recorded(*args, **kw):
            t0 = time.monotonic_ns()
            out = orig(*args, **kw)
            rec = fc_record(args, kw, out, t0)
            if rec is not None:
                with lock:
                    records.append(rec)
            return out

        recorded.__wrapped_v2__ = orig      # the un-instrumented call (session_state.MultiSessionEngine re-routes it)
        self.wrapper._run_fc_async_steps = recorded

    def _default_fc_ctx_fix(self, mode: str | None) -> dict:
        """FC hand-back fix default (owner GO 2026-11-12): ON for the OFFICIAL VoiceChat checkpoint only, selective
        semantics (real PAD + text/ASR copied back, function channel PAD → the resumed step consumes (PAD, PAD), never
        EOTR: EOTR re-opens the call on this model, FX5/round 12). Other checkpoints (e.g. the multilingual lab's
        D-V2.1) stay OFF: their hand-back semantics differ. mode: "auto" (default) | "on" | "off"; env
        MYNAH_V2_FC_CTX_FIX overrides the default (rollback without code change). "eotr_step" (explicit only, never
        a default) = the experimental training-layout EOTR slot, see fc_ctx_fix.enable(eotr_step)."""
        mode = (mode or os.environ.get("MYNAH_V2_FC_CTX_FIX") or "auto").lower()
        ckpt = os.path.realpath(str(self.cfg.s2s.llm_checkpoint_path))
        official = os.path.basename(ckpt.rstrip("/")) == OFFICIAL_SNAPSHOT
        on = mode in ("on", "eotr_step") or (mode == "auto" and official)
        info = {"mode": mode, "official_checkpoint": official, "enabled": on}
        if mode == "eotr_step":                  # EXPERIMENT (explicit only): the training layout's inserted EOTR slot
            from mynah_v2.model import fc_ctx_fix
            info["fix"] = fc_ctx_fix.enable(self, pad_fill=True, channels=fc_ctx_fix._CH, func_pad=False, eotr_step=True)
        elif on:
            from mynah_v2.model import fc_ctx_fix
            info["fix"] = fc_ctx_fix.enable(self, pad_fill=True, channels=("gen_text", "gen_asr_text"), func_pad=True)
        return info

    def _wrap_executor(self) -> None:
        orig = self.pipeline._execute_tool_call

        def execute(call_text: str):
            mh = self.multi_call_handler
            if mh is not None:
                parsed = T.parse_call_text(call_text)
                if len(parsed) > 1:
                    self.multi_call_events.append({"n": len(parsed), "t_ns": time.monotonic_ns()})
                    return mh(call_text)
            res = orig(call_text)
            if res is None:
                h = self.unknown_tool_handler
                self.unknown_tool_events.append({"call_text": call_text[:300], "handled": h is not None,
                                                 "t_ns": time.monotonic_ns()})
                if h is not None:
                    return h(call_text)
            return res

        self.pipeline._execute_tool_call = execute

    def _frame(self, samples: torch.Tensor, is_first=False, is_last=False, options=None):
        from nemo.collections.asr.inference.streaming.framing.request import Frame
        return Frame(samples=samples, stream_id=self.stream_id, is_first=is_first, is_last=is_last,
                     options=options)

    def _ctx(self):
        cm = self.pipeline.context_manager
        slot = cm.streamidx2slotidx.get(self.stream_id)
        return None if slot is None else cm.slot_contexts[slot]

    # -------------------------------------------------------------------------------------------------
    def bind_session_tools(self, tools: dict) -> int:
        """Tool handlers owned by the next/current session (unbound by ``close``); see session_tools.py."""
        return self.session_tools.bind(tools)

    def set_session_guard(self, guard, resume: str | None = "by_outcome", tracer=None) -> None:
        """Guard the current session's bound tools (agent.call_guard.attach_session_guard); cleared by ``close``."""
        from mynah_v2.agent.call_guard import attach_session_guard
        attach_session_guard(self, guard, resume=resume, tracer=tracer)

    def open(self, system_prompt: str | None, fc_wait_s: float = 10.0) -> None:
        from nemo.collections.speechlm2.inference.streaming.framing.s2s_request_options import S2SRequestOptions
        from mynah_v2.model.session_tools import release_fc_thread
        # a previous session's FC thread that outlived its close must finish before this stream id is reused
        if release_fc_thread(getattr(self.pipeline, "_fc_async_bg", {}), self.stream_id, fc_wait_s) == "running":
            raise RuntimeError("previous session's FC thread is still running; refusing to reuse the stream")
        self.pipeline.open_session()
        self._opts = S2SRequestOptions(system_prompt=system_prompt) if system_prompt else None
        if self._opts is not None:
            self.pipeline.generate_step([self._frame(torch.empty(0), is_first=True, options=self._opts)])
        self._started = False
        self._pos = {"text": 0, "asr": 0, "func": 0}
        self._last_frame_idx = 0
        with self._fc_lock:
            self.fc_records.clear()
            self._rec_cursor = 0

    def fc_active(self) -> bool:
        return self.stream_id in getattr(self.pipeline, "_fc_async_bg", {})

    def step(self, pcm16k: torch.Tensor, is_last: bool = False) -> StepOutput:
        """Feed one chunk (``chunk_samples`` float32 samples in [-1, 1] at 16 kHz)."""
        if pcm16k.numel() != self.chunk_samples:
            raise ValueError(f"chunk must have {self.chunk_samples} samples, got {pcm16k.numel()}")
        first = not self._started
        self._started = True
        t0 = time.monotonic_ns()
        self.pipeline.generate_step([self._frame(pcm16k.float().cpu(), is_first=first, is_last=is_last,
                                                 options=self._opts if first else None)])
        lat = (time.monotonic_ns() - t0) / 1e6
        st = self.pipeline.get_or_create_state(self.stream_id)
        audio = st.audio_buffer.detach().float().reshape(-1).cpu() if st.audio_buffer.numel() else torch.zeros(0)
        full_t, full_a, full_f = st.get_output_text(), st.get_output_asr_text(), st.get_output_function_text()
        td = full_t[self._pos["text"]:]
        ad = full_a[self._pos["asr"]:] if len(full_a) >= self._pos["asr"] else " "
        fd = full_f[self._pos["func"]:]
        self._pos = {"text": len(full_t), "asr": len(full_a), "func": len(full_f)}
        gb = getattr(self, "_v2_guard_by_stream", None)
        g = gb.get(self.stream_id) if gb else None
        if g is not None and ad and ad.strip():           # the caller's speech = the guard's input epoch
            g.on_user_text(ad)
        st.cleanup_after_response()
        ctx = self._ctx()
        new_t, new_f, fi = [], [], self._last_frame_idx
        if ctx is not None:
            fi = int(ctx.frame_idx)
            lo = self._last_frame_idx
            if fi > lo:
                new_t = ctx.gen_text[0, lo:fi].tolist()
                if getattr(ctx, "gen_function_text", None) is not None:
                    new_f = ctx.gen_function_text[0, lo:fi].tolist()
            self._last_frame_idx = fi
        with self._fc_lock:
            recs = list(self.fc_records[self._rec_cursor:])
            self._rec_cursor = len(self.fc_records)
        return StepOutput(audio=audio, text_delta=td, asr_delta=ad, func_delta=fd, frame_idx=fi,
                          new_text_ids=new_t, new_func_ids=new_f, fc_active=self.fc_active(), fc_records=recs,
                          latency_ms=lat)

    def token_tensors(self) -> tuple | None:
        st = self.pipeline.get_or_create_state(self.stream_id)
        return st.get_token_tensors()

    def release_fc(self, join_timeout_s: float = 60.0) -> bool:
        """Stop and forget a still-running FC fast path of this stream. The reference pipeline keeps it in
        ``_fc_async_bg[stream_id]`` across ``close_session``/``open_session`` (streaming_s2s_pipeline.py:399, 990,
        1078: only a later generate_step consumes it), so a session closed mid-call would leak it into the next
        session on the same stream id. Returns True if something was released."""
        from mynah_v2.model.session_tools import release_fc_thread
        st = release_fc_thread(getattr(self.pipeline, "_fc_async_bg", {}), self.stream_id, join_timeout_s)
        if st == "running":                     # handle kept: the next open() waits for it or refuses
            self.fc_still_running = getattr(self, "fc_still_running", 0) + 1
            return False
        if st == "released":
            self.fc_released = getattr(self, "fc_released", 0) + 1
            return True
        return False

    def close(self) -> None:
        try:
            from mynah_v2.agent.call_guard import detach_session_guard
            detach_session_guard(self)
            self.release_fc()
            self.pipeline.close_session()
        finally:
            st = getattr(self, "session_tools", None)
            if st is not None and st.active:
                st.unbind()
            fx = getattr(getattr(self, "wrapper", None), "_v2_fc_ctx_fix", None)
            if fx is not None:                       # fc_ctx_fix: drop this stream's buffer references
                fx["bufs"].pop(self.stream_id, None)
            torch.cuda.empty_cache()
