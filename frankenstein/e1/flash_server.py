"""G0: the Sophon Flash talker as a separate process, driven by the talker adapter's commands (stdin -> stdout).

Why a process: the official ``qwen_tts`` package requires transformers 4.57 while the engine's venv (NeMo + V2) pins
4.56; importing it into the engine process fails (``check_model_inputs`` signature). The talker therefore runs in the
talker's own, unmodified environment (python of that env), and talks JSON lines over pipes. Nothing is installed or
changed in either environment.

Model, prompt layout, sampling and per-frame seeding are the transplant harness's ``flash_rt.FlashTalker`` (imported
read-only from ``--harness``): Qwen3-TTS-12Hz structure, zero-shot clone voice (x-vector of a reference wav), block
mode (no step without text: the paused state is kept), codec EOS masked while the text is open, runaway cap.
What is new here is only the INCREMENTAL driving (text arrives while frames are generated) and the chunked delivery
on the wall clock:

  stdin   {"op": "start",  "epoch", "turn", "seg", "text"}   new segment: fresh prefill (first text id)
          {"op": "append", "epoch", "turn", "seg", "text"}   more committed text for that segment
          {"op": "final",  "epoch", "turn", "seg"}           text complete: tts_eos, let the codec EOS end it
          {"op": "cancel", "epoch", "new_epoch"}             drop every segment of that epoch (and older)
          {"op": "quit"}
  stdout  {"ev": "ready", ...} | {"ev": "audio", "epoch", "turn", "seg", "a", "b", "t_ns", "sr", "pcm16": b64}
          | {"ev": "done", ..., "frames", "ended"} | {"ev": "cancelled", ...} | {"ev": "stats", ...}

G1 options: ``--fast-cp`` replaces the code predictor's HF ``generate`` (called once per codec frame for the 15
residual codebooks) by a plain loop over the same forward with the same sampling (float32 logits, temperature, top-k,
softmax, multinomial: the HF warper order) -- instance-level, the package is not edited; codes must stay identical
(checked by e1/g1_talker.py). ``--profile`` times the code predictor separately (extra syncs). ``done`` events
carry the segment's codes so two runs can be compared.

Segments of one epoch are generated in order, one at a time. Delivery: chunk ramp 2, 4, 8, 12, 16, 25 frames with
25 frames of codec left context (flash_rt RAMP / LEFT_CTX). ``t_ns`` is CLOCK_MONOTONIC (shared with the engine
process on Linux).
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import queue
import sys
import threading
import time
from collections import deque

OUT = None


def emit(obj):
    OUT.write(json.dumps(obj) + "\n")
    OUT.flush()


class Gen:
    def __init__(self, epoch, turn, seg, ids):
        self.epoch, self.turn, self.seg = epoch, turn, seg
        self.ids, self.nxt, self.final, self.eos_fed = list(ids), 0, False, False
        self.past = self.past_hidden = self.logits = None
        self.hist, self.codes, self.k = [], [], 0
        self.delivered, self.ramp_i = 0, 0
        self.t_start_ns = time.monotonic_ns()
        self.first_audio_ns = None
        self.stalled_ms = 0.0
        self.started = False


def fast_generate(cp, torch, stats=None, profile=False):
    """Drop-in for ``code_predictor.generate`` as called by the talker forward (inputs_embeds of 2 positions,
    max_new_tokens = 15, sampling with top_p = 1.0)."""
    from types import SimpleNamespace
    from transformers.cache_utils import DynamicCache

    def gen(inputs_embeds=None, max_new_tokens=15, do_sample=True, top_p=1.0, top_k=50, temperature=1.0, **kw):
        if profile:
            torch.cuda.synchronize()
            c0 = time.perf_counter()
        cache = DynamicCache()
        out = cp(inputs_embeds=inputs_embeds, past_key_values=cache, use_cache=True)
        gs, seq = out.generation_steps, []
        for i in range(max_new_tokens):
            x = out.logits[:, -1, :].to(dtype=torch.float32)
            if do_sample:
                x = x / temperature
                x = x.masked_fill(x < torch.topk(x, top_k)[0][..., -1, None], -float("inf"))
                nxt = torch.multinomial(torch.softmax(x, -1), num_samples=1)
            else:
                nxt = x.argmax(-1, keepdim=True)
            seq.append(nxt)
            if i + 1 == max_new_tokens:
                break
            out = cp(input_ids=nxt, past_key_values=cache, use_cache=True, generation_steps=gs)
            gs = out.generation_steps
        if profile:
            torch.cuda.synchronize()
            stats["cp_ms"].append(round((time.perf_counter() - c0) * 1e3, 2))
        return SimpleNamespace(sequences=torch.cat(seq, 1))
    return gen


def timed_generate(orig, torch, stats):
    def gen(*a, **kw):
        torch.cuda.synchronize()
        c0 = time.perf_counter()
        r = orig(*a, **kw)
        torch.cuda.synchronize()
        stats["cp_ms"].append(round((time.perf_counter() - c0) * 1e3, 2))
        return r
    return gen


def main():  # pragma: no cover - GPU, talker env
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--harness", required=True, help="directory of flash_rt.py (read-only import)")
    ap.add_argument("--root", required=True, help="Flash snapshot (contains talker/)")
    ap.add_argument("--voice", required=True, help="reference wav (zero-shot clone) or a built-in voice name")
    ap.add_argument("--lang", default="it")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fast-cp", action="store_true", help="plain loop instead of HF generate for the code predictor")
    ap.add_argument("--profile", action="store_true", help="time the code predictor separately (extra syncs)")
    a = ap.parse_args()
    OUT = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    sys.stdout = sys.stderr                                 # library prints never corrupt the protocol stream
    sys.path.insert(0, a.harness)
    import numpy as np
    import torch
    from flash_rt import LEFT_CTX, RAMP, FRAME_S, FlashTalker, Sampling
    t0 = time.monotonic()
    ft = FlashTalker(a.root)
    xvec, vname = ft.voice(a.voice, a.lang)
    s = Sampling()
    load_s = round(time.monotonic() - t0, 1)
    emit({"ev": "ready", "load_s": load_s, "voice": vname, "lang": a.lang,
          "mem_alloc_gb": round(torch.cuda.memory_allocated() / 2 ** 30, 2)})

    cmds: queue.Queue = queue.Queue()

    def reader():
        for line in sys.stdin:
            line = line.strip()
            if line:
                cmds.put(json.loads(line))
        cmds.put({"op": "quit"})
    threading.Thread(target=reader, daemon=True).start()

    pending: deque = deque()          # Gen objects not finished, in order
    by_key: dict = {}
    dead_epochs: set = set()
    stats = {"step_ms": [], "prefill_ms": [], "decode_ms": [], "cp_ms": [], "segments": [], "cancels": 0,
             "fast_cp": a.fast_cp, "profile": a.profile}
    cp = ft.tk.code_predictor
    if a.fast_cp:
        cp.generate = fast_generate(cp, torch, stats, a.profile)
    elif a.profile:
        cp.generate = timed_generate(cp.generate, torch, stats)
    sync = torch.cuda.synchronize
    V = ft.tc.codec_eos_token_id

    def deliver(g, last=False):
        while g.delivered < len(g.codes):
            n = RAMP[min(g.ramp_i, len(RAMP) - 1)]
            b = g.delivered + n
            if b > len(g.codes):
                if not last:
                    return
                b = len(g.codes)
            a_, lo = g.delivered, max(0, g.delivered - LEFT_CTX)
            sync()
            c0 = time.perf_counter()
            w, sr = ft.decode(torch.tensor(g.codes[lo:b], dtype=torch.long))
            stats["decode_ms"].append(round((time.perf_counter() - c0) * 1e3, 2))
            spf = sr * FRAME_S
            piece = w[int(round((a_ - lo) * spf)):int(round((b - lo) * spf))]
            pcm16 = (np.clip(piece, -1, 1) * 32767).astype(np.int16).tobytes()
            t = time.monotonic_ns()
            if g.first_audio_ns is None:
                g.first_audio_ns = t
            emit({"ev": "audio", "epoch": g.epoch, "turn": g.turn, "seg": g.seg, "a": a_, "b": b, "t_ns": t, "sr": sr,
                  "pcm16": base64.b64encode(pcm16).decode()})
            g.delivered, g.ramp_i = b, g.ramp_i + 1

    def finish(g, ended):
        deliver(g, last=True)
        rec = {"epoch": g.epoch, "turn": g.turn, "seg": g.seg, "frames": len(g.codes), "ended": ended,
               "text_ids": len(g.ids), "first_audio_ms": round((g.first_audio_ns - g.t_start_ns) / 1e6, 1)
               if g.first_audio_ns else None}
        stats["segments"].append(rec)
        emit({"ev": "done", **rec, "t_ns": time.monotonic_ns(), "codes": g.codes})
        pending.remove(g)
        by_key.pop((g.epoch, g.turn, g.seg), None)

    def can_step(g):
        return g.nxt < len(g.ids) or g.final or g.eos_fed

    with torch.inference_mode():
        while True:
            g = pending[0] if pending else None
            idle = g is None or (g.started and not can_step(g)) or (not g.started and not g.ids)
            try:
                cmd = cmds.get(timeout=0.2) if idle else cmds.get_nowait()
            except queue.Empty:
                cmd = None
            while cmd is not None:
                op = cmd["op"]
                if op == "quit":
                    emit({"ev": "stats", **stats})
                    return
                key = (cmd.get("epoch"), cmd.get("turn"), cmd.get("seg"))
                if op == "cancel":
                    stats["cancels"] += 1
                    dead_epochs.update(range(0, cmd["new_epoch"]))
                    for x in [x for x in pending if x.epoch < cmd["new_epoch"]]:
                        pending.remove(x)
                        by_key.pop((x.epoch, x.turn, x.seg), None)
                        emit({"ev": "cancelled", "epoch": x.epoch, "turn": x.turn, "seg": x.seg,
                              "frames": len(x.codes), "t_ns": time.monotonic_ns()})
                elif key[0] in dead_epochs:
                    pass
                elif op == "start":
                    gg = Gen(*key, ft.encode_text(cmd["text"]))
                    pending.append(gg)
                    by_key[key] = gg
                elif op == "append" and key in by_key:
                    by_key[key].ids += ft.encode_text(cmd["text"])
                elif op == "final" and key in by_key:
                    by_key[key].final = True
                try:
                    cmd = cmds.get_nowait()
                except queue.Empty:
                    cmd = None
            g = pending[0] if pending else None
            if g is None:
                continue
            if not g.started:
                if not g.ids:
                    if g.final:
                        finish(g, "empty")
                    continue
                sync()
                c0 = time.perf_counter()
                out = ft.tk(inputs_embeds=ft.prefill_embeds(g.ids[0], a.lang, xvec), use_cache=True,
                            attention_mask=None, generation_step=None)
                sync()
                stats["prefill_ms"].append(round((time.perf_counter() - c0) * 1e3, 2))
                g.past, g.past_hidden, g.logits = out.past_key_values, out.past_hidden, out.logits
                g.nxt, g.started = 1, True
            if not can_step(g):
                continue
            c = ft._sample0(g.logits, g.hist, g.k, not g.eos_fed, s, a.seed)
            if c == V:
                finish(g, "eos")
                continue
            if g.nxt < len(g.ids):
                add = ft.te([g.ids[g.nxt]])
                g.nxt += 1
            elif not g.eos_fed:
                add, g.eos_fed = ft.eos, True
            else:
                add = ft.pad
            torch.manual_seed(a.seed * 100003 + g.k + 7919)
            sync()
            c0 = time.perf_counter()
            out = ft.tk(input_ids=torch.tensor([[c]], device=ft.dev), past_key_values=g.past,
                        past_hidden=g.past_hidden, trailing_text_hidden=add, tts_pad_embed=ft.pad, generation_step=0,
                        use_cache=True, subtalker_dosample=True, subtalker_top_k=s.sub_top_k, subtalker_top_p=1.0,
                        subtalker_temperature=s.sub_temperature)
            g.codes.append(out.hidden_states[1][0].tolist())
            sync()
            stats["step_ms"].append(round((time.perf_counter() - c0) * 1e3, 2))
            g.past, g.past_hidden, g.logits = out.past_key_values, out.past_hidden, out.logits
            g.hist.append(c)
            g.k += 1
            deliver(g)
            if g.eos_fed and g.k > s.frames_per_token_cap * len(g.ids) + 60:
                finish(g, "cap")
            elif g.k >= 1500:
                finish(g, "max")


if __name__ == "__main__":
    main()
