# SPDX-License-Identifier: Apache-2.0
# Derived from tyger-s2s `app/plugins/ears/tyger/asr.py` @97d011d (NemotronAsr.features / encode: log-mel on the
# device, layer-15 encode stopped by a hook, all-zero windows cached; Copyright the tyger-s2s authors, Apache License
# 2.0, see LICENSES/Apache-2.0.txt). Modified for Frankenstein v2: one class per session holding the two rolling
# windows, single-device code path (CPU or any CUDA GPU).
"""The GRU's trained feature regime: every 320 ms hop, the last 8 s of each channel re-encoded to layer 15 with
lookahead 0; the hop's 4 newest frames feed the GRU.

Why not the streaming ASR encoder's layer 15 (one pass for both): measured on the 52 s Italian session track
(ears_check / ears_window_ref), cache-aware streaming features (k=3, and k=0 too) keep the endpoint head's p_end at
0.72-0.88 after three of the four turn ends where this regime reaches 0.89-0.98: 1 endpoint of 4 at the default
threshold instead of 4 of 4. The head was trained on this regime; until it is fine-tuned on streaming features
the ears use the same encoder weights twice (streaming for the transcript, windows for the GRU).
"""

from __future__ import annotations

import numpy as np
import torch

from .turntaker import L15

WINDOW = 128000          # 8 s
HOP = 5120


class _Stop(Exception):
    pass


class WindowL15:
    def __init__(self, model, proc, device):
        self.enc, self.fe, self.dev = model.encoder, proc.feature_extractor, device
        if device != "cpu":
            self.fe.mel_filters = self.fe.mel_filters.to(device) if hasattr(self.fe.mel_filters, "to") else \
                torch.as_tensor(self.fe.mel_filters, device=device)
        self.win = np.zeros((2, 0), np.float32)
        self._zero = {}
        self._h = None
        self.enc.layers[L15 - 1].register_forward_hook(self._hook) if not getattr(self.enc, "_f2_hooked", False) \
            else None
        self.enc._f2_hooked = True
        WindowL15._active = self

    @staticmethod
    def _hook(mod, args, out):
        w = getattr(WindowL15, "_active", None)
        if w is not None and w._stop:
            w._h = out[0] if isinstance(out, tuple) else out
            raise _Stop
    _stop = False

    @torch.no_grad()
    def features(self, wavs):
        """Log-mel on the device (the processor's steps: preemphasis, STFT, mel, mask) for equal-length clips."""
        fe, dev = self.fe, self.dev
        x = torch.from_numpy(np.stack(wavs)).to(dev)
        n = x.shape[1]
        x = torch.cat([x[:, :1], x[:, 1:] - fe.preemphasis * x[:, :-1]], 1)
        f = fe._torch_extract_fbank_features(x, dev, center=True)
        flen = (n + fe.n_fft // 2 * 2 - fe.n_fft) // fe.hop_length
        am = torch.arange(f.shape[1], device=dev)[None] < flen
        return f * am.unsqueeze(-1)

    def push(self, user: np.ndarray, agent: np.ndarray) -> list:
        """Add a hop; -> the windows this session needs encoded: [user] or [user, agent] (an all-zero agent window
        of a length already seen is served from the cache)."""
        self.win = np.concatenate([self.win, np.stack([user, agent]).astype(np.float32)], 1)[:, -WINDOW:]
        n = self.win.shape[1]
        if not self.win[1].any() and n in self._zero:
            return [self.win[0]]
        return [self.win[0], self.win[1]]

    def take(self, encoded: list):
        """``encoded``: the L15 (T, 1024) of the windows ``push`` returned, in order -> (user L15 (4,1024), agent)."""
        n = self.win.shape[1]
        hu = encoded[0][-4:]
        if len(encoded) == 1:
            return hu, self._zero[n]
        ha = encoded[1][-4:]
        if not self.win[1].any():
            self._zero[n] = ha.clone()
        return hu, ha

    @torch.no_grad()
    def encode(self, wavs: list) -> list:
        """Layer 15 of several windows (any sessions; equal lengths batched in one encoder call)."""
        out = [None] * len(wavs)
        groups: dict = {}
        for i, w in enumerate(wavs):
            groups.setdefault(len(w), []).append(i)
        for n, idx in groups.items():
            feats = self.features([wavs[i] for i in idx])
            WindowL15._active, self._stop = self, True
            try:
                self.enc(input_features=feats, num_lookahead_tokens=0)
            except _Stop:
                pass
            finally:
                self._stop = False
            h = self._h.float()
            for j, i in enumerate(idx):
                out[i] = h[j]
        return out

    @torch.no_grad()
    def hop(self, user: np.ndarray, agent: np.ndarray):
        """-> (user L15 (4, 1024), agent L15 (4, 1024)) for the hop just added (one session)."""
        return self.take(self.encode(self.push(user, agent)))
