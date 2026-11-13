"""Opt-in diagnostics for the multi-session gate (V2_DIAG=1; token-neutral, read-only observers).

* codes: every ``audio_codec.decode`` call's input codes, attributed to a session: the rendezvous worker thread
  ``rz-<sid>`` in the multi-session phase, ``solo_sid`` (set by the runner) on the main thread in the solo phase,
  ``fc`` for the upstream FC threads. Installed as the OUTERMOST wrapper so it sees the per-session request before
  any rendezvous parking / batching. Codes are what the decoder will speak (after control-code replacement).
* syncs: every ``torch.cuda.synchronize()`` call (thread, host start, host wait ms). Upstream calls it after each
  EarTTS step and each codec decode, also inside the FC threads; a device-wide sync from one thread waits for the
  work every other thread queued on the shared stream.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict

import torch


class DiagRecorder:
    def __init__(self):
        self.lock = threading.Lock()
        self.phase = "solo"
        self.solo_sid = None
        self.t_ref = time.perf_counter()
        self.codes = defaultdict(list)          # (phase, key) -> [(t_ms, codes int32 [k, Q])]
        self.syncs = []                          # (phase, thread, t_ms, wait_ms)
        self._undo = []

    def key(self) -> str:
        name = threading.current_thread().name
        if name.startswith("rz-"):
            return name[3:]
        if "_fc_async" in name:
            return "fc"
        return self.solo_sid or name

    def now_ms(self) -> float:
        return (time.perf_counter() - self.t_ref) * 1e3

    def install_codes(self, codec) -> None:
        from mynah_v2.model.codec_batch import decode_kwargs
        orig, had = codec.decode, "decode" in codec.__dict__
        rec = self

        def decode(*args, **kw):
            d = decode_kwargs(args, kw)
            code = None if d is None else d.get("code")
            if isinstance(code, torch.Tensor) and code.dim() == 3:
                c = code.detach()[0].to("cpu", torch.int32)
                with rec.lock:
                    rec.codes[(rec.phase, rec.key())].append((rec.now_ms(), c))
            return orig(*args, **kw)
        codec.decode = decode
        self._undo.append((codec, "decode", orig, had))

    def install_sync(self) -> None:
        orig = torch.cuda.synchronize
        rec = self

        def synchronize(device=None):
            t0 = time.perf_counter()
            r = orig(device)
            with rec.lock:
                rec.syncs.append((rec.phase, threading.current_thread().name, (t0 - rec.t_ref) * 1e3,
                                  (time.perf_counter() - t0) * 1e3))
            return r
        torch.cuda.synchronize = synchronize
        self._undo.append((torch.cuda, "synchronize", orig, True))

    def uninstall_codes(self) -> None:
        keep = []
        for obj, attr, orig, had in reversed(self._undo):
            if attr != "decode":
                keep.append((obj, attr, orig, had))
                continue
            if had:
                setattr(obj, attr, orig)
            else:
                try:
                    delattr(obj, attr)
                except AttributeError:
                    pass
        self._undo = list(reversed(keep))

    def uninstall(self) -> None:
        for obj, attr, orig, had in reversed(self._undo):
            if had:
                setattr(obj, attr, orig)
            else:
                try:
                    delattr(obj, attr)
                except AttributeError:
                    pass
        self._undo.clear()

    def save(self, out) -> dict:
        """codes.npz (one [n_frames, Q] array + per-frame call index / time per (phase, key)) and syncs list."""
        import numpy as np
        arrs, info = {}, {}
        with self.lock:
            items = sorted(self.codes.items())
            syncs = list(self.syncs)
        for (ph, k), lst in items:
            if not lst:
                continue
            name = f"{ph}__{k}"
            arrs[name] = np.concatenate([c.numpy() for _, c in lst], axis=0)
            arrs[name + "__call"] = np.concatenate([np.full(c.shape[0], i, np.int32) for i, (_, c) in enumerate(lst)])
            arrs[name + "__t_ms"] = np.array([t for t, _ in lst], np.float64)
            info[name] = {"calls": len(lst), "frames": int(arrs[name].shape[0]), "q": int(arrs[name].shape[1])}
        np.savez_compressed(out / "codes.npz", **arrs)
        return {"codes": info, "syncs": syncs}
