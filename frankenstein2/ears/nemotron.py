"""Our own runtime for the Nemotron 3.5 ASR streaming model (cache-aware FastConformer encoder + language-prompt
projector + RNN-T predictor / joint), in plain torch, for many sessions at once.

Architecture (as the published checkpoint defines it; the module layout follows the reference implementation in
transformers' `nemotron_asr_streaming` / `nemotron3_5_asr` models, Apache-2.0, which we keep only as a test
reference):
  log-mel (128 bins, n_fft 512, win 400 Hann, hop 160, pre-emphasis 0.97, log(x + 2^-24), no normalisation)
  -> causal Conv2d subsampling x8 (3x3 stride-2 stem + 2 depthwise-separable stride-2 stages, ReLU) -> Linear 4352->1024
  -> 24 conformer blocks: 0.5*FF(SiLU) / rel-pos MHSA (8 heads, Transformer-XL biases u, v) / conv module (pointwise
     GLU, causal depthwise k=9, LayerNorm, SiLU, pointwise) / 0.5*FF / LayerNorm
  attention: chunked-limited (chunk = lookahead+1 frames, left context 56 frames = sliding_window - 1)
  -> prompt projector (concat one-hot language prompt, 1152->2048->1024, ReLU) -> encoder projector 1024->640
  RNN-T: embedding + 2-layer LSTM(640) + projector; joint = head(ReLU(enc + dec)); greedy, max 10 symbols per frame.

Streaming state (one row per session, in fixed-size pools so every session is stacked into one batch and the step has
static shapes): per block the last 56 frames of K and V (right-aligned; a per-row count of valid frames masks the
rest, so sessions at different ages share a batch), the depthwise-conv left context (8 frames), and the subsampling
convolutions' left context (1 mel / 1 frame per stage). A first chunk (25 mel frames, two zero frames of left context)
runs in its own call; every later chunk is 32 mel frames -> 4 encoder frames.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..gpu import host

SR = 16000
N_FFT, WIN, HOP_LEN, N_MELS = 512, 400, 160, 128
LOG_GUARD = 2.0 ** -24
PREEMPH = 0.97


def slaney_mel(sr=SR, n_fft=N_FFT, n_mels=N_MELS, fmin=0.0, fmax=None) -> np.ndarray:
    """Slaney-style mel filterbank with slaney area normalisation (the librosa default), float32 [n_mels, 1+n_fft/2]."""
    fmax = fmax or sr / 2

    def hz_to_mel(f):
        f = np.asarray(f, np.float64)
        f_sp = 200.0 / 3
        mels = f / f_sp
        min_log_hz, min_log_mel, logstep = 1000.0, 1000.0 / f_sp, np.log(6.4) / 27.0
        return np.where(f >= min_log_hz, min_log_mel + np.log(np.maximum(f, 1e-10) / min_log_hz) / logstep, mels)

    def mel_to_hz(m):
        m = np.asarray(m, np.float64)
        f_sp = 200.0 / 3
        freqs = f_sp * m
        min_log_hz, min_log_mel, logstep = 1000.0, 1000.0 / f_sp, np.log(6.4) / 27.0
        return np.where(m >= min_log_mel, min_log_hz * np.exp(logstep * (m - min_log_mel)), freqs)
    fftfreqs = np.linspace(0, sr / 2, 1 + n_fft // 2)
    mel_f = mel_to_hz(np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2))
    fdiff = np.diff(mel_f)
    ramps = mel_f[:, None] - fftfreqs[None, :]
    lower = -ramps[:-2] / fdiff[:-1, None]
    upper = ramps[2:] / fdiff[1:, None]
    w = np.maximum(0, np.minimum(lower, upper))
    enorm = 2.0 / (mel_f[2:n_mels + 2] - mel_f[:n_mels])
    return (w * enorm[:, None]).astype(np.float32)


class NemotronASR:
    def __init__(self, bundle: str, device="cuda", prefix="nemotron.", dtype=torch.float32):
        from safetensors import safe_open
        self.dev = torch.device(device)
        self.dtype = dtype
        folder = Path(bundle)
        cfg = json.loads((folder / "nemotron" / "config.json").read_text())
        ec = cfg["encoder_config"]
        self.cfg, self.ec = cfg, ec
        self.H, self.nh = ec["hidden_size"], ec["num_attention_heads"]
        self.hd = self.H // self.nh
        self.n_layers = ec["num_hidden_layers"]
        self.left = ec["sliding_window"] - 1                         # 56 frames of left context
        self.kconv = ec["conv_kernel_size"]
        self.blank, self.max_sym = cfg["blank_token_id"], cfg["max_symbols_per_step"]
        self.num_prompts = cfg["num_prompts"]
        W = {}
        with safe_open(str(folder / "model.safetensors"), framework="pt", device=str(self.dev)) as f:
            for k in f.keys():
                if k.startswith(prefix):
                    W[k[len(prefix):]] = f.get_tensor(k).to(dtype)
        self.W = W
        self.mel_fb = torch.from_numpy(slaney_mel()).to(self.dev)
        self.window = torch.hann_window(WIN, periodic=False, device=self.dev)
        inv = 1.0 / (10000.0 ** (torch.arange(0, self.H, 2, dtype=torch.float32, device=self.dev) / self.H))
        self.inv_freq = inv
        pc = json.loads((folder / "nemotron" / "processor_config.json").read_text())
        self.prompts = pc.get("prompt_dictionary", {})
        from tokenizers import Tokenizer
        self.tokenizer = Tokenizer.from_file(str(folder / "nemotron" / "tokenizer.json"))
        self.L = [{n: W[f"encoder.layers.{i}.{n}"] for n in (
            "feed_forward1.linear1.weight", "feed_forward1.linear2.weight", "feed_forward2.linear1.weight",
            "feed_forward2.linear2.weight", "norm_feed_forward1.weight", "norm_feed_forward1.bias",
            "norm_feed_forward2.weight", "norm_feed_forward2.bias", "norm_self_att.weight", "norm_self_att.bias",
            "norm_conv.weight", "norm_conv.bias", "norm_out.weight", "norm_out.bias", "self_attn.q_proj.weight",
            "self_attn.k_proj.weight", "self_attn.v_proj.weight", "self_attn.o_proj.weight",
            "self_attn.relative_k_proj.weight", "self_attn.bias_u", "self_attn.bias_v", "conv.pointwise_conv1.weight",
            "conv.depthwise_conv.weight", "conv.norm.weight", "conv.norm.bias", "conv.pointwise_conv2.weight")}
            for i in range(self.n_layers)]

    # ------------------------------------------------------------------ features
    @torch.no_grad()
    def mel(self, wav: torch.Tensor, center: bool) -> torch.Tensor:
        """wav (B, N) float on the device -> log-mel (B, T, 128). Pre-emphasis inside the given clip (as the reference
        feature extractor does per call)."""
        x = torch.cat([wav[:, :1], wav[:, 1:] - PREEMPH * wav[:, :-1]], 1)
        st = torch.stft(x, N_FFT, hop_length=HOP_LEN, win_length=WIN, window=self.window, return_complex=True,
                        pad_mode="constant", center=center)
        mag = torch.view_as_real(st)
        mag = torch.sqrt(mag.pow(2).sum(-1)).pow(2)
        m = torch.log(self.mel_fb @ mag + LOG_GUARD).permute(0, 2, 1)
        if center:                                   # frames past floor(N / hop) are padding (zeroed, as the reference)
            n = wav.shape[1] // HOP_LEN
            m[:, n:] = 0.0
        return m

    # ------------------------------------------------------------------ pieces
    def _ln(self, x, w, b):
        return F.layer_norm(x, (x.shape[-1],), w, b, 1e-5)

    def _subsample(self, mel: torch.Tensor, caches=None, first: bool = False):
        """mel (B, T, 128) -> (B, T', 1024). caches: None (offline: causal time pad 2 / 1) or a list of 3 tensors
        (left context per stage, updated in place)."""
        W = self.W
        x = mel.unsqueeze(1)                                            # (B, 1, T, F)
        convs = [("encoder.subsampling.conv_in", 1)] + [(f"encoder.subsampling.layers.{i}.depthwise_conv", 256)
                                                        for i in range(2)]
        new = []
        for si, (name, groups) in enumerate(convs):
            x = F.pad(x, (2, 1))                                        # frequency: (k - 1, s - 1)
            if caches is None:
                x = F.pad(x, (0, 0, 2, 1))                              # time: (k - 1, s - 1)
            else:
                left = caches[si]
                if first:
                    left = torch.cat([torch.zeros_like(left), left], 2)  # + init pad (k - 1) - (k - s) = 1 frame
                new.append(x[:, :, -1:])
                x = torch.cat([left, x], 2)
            x = F.conv2d(x, W[name + ".weight"], W[name + ".bias"], stride=2, groups=groups)
            if si > 0:
                p = f"encoder.subsampling.layers.{si - 1}.pointwise_conv"
                x = F.conv2d(x, W[p + ".weight"], W[p + ".bias"])
            x = F.relu(x)
        if caches is not None:
            for c, n in zip(caches, new):
                c.copy_(n)
        B, C, T, Fq = x.shape
        x = x.transpose(1, 2).reshape(B, T, C * Fq)
        return F.linear(x, W["encoder.subsampling.linear.weight"], W["encoder.subsampling.linear.bias"])

    def _pos(self, L: int):
        pos = torch.arange(L - 1, -L, -1, device=self.dev, dtype=torch.float32)
        f = pos[:, None] * self.inv_freq[None, :]
        return torch.stack([f.sin(), f.cos()], -1).reshape(2 * L - 1, self.H)

    def _attn(self, l, x, mask, pos, kv_past=None):
        """x (B, q, H); mask (B, 1|q, L) bool; pos (2L-1, H); kv_past None or (k, v) (B, nh, P, hd)."""
        B, q, _ = x.shape
        nh, hd = self.nh, self.hd
        Q = F.linear(x, l["self_attn.q_proj.weight"]).view(B, q, nh, hd).transpose(1, 2)
        K = F.linear(x, l["self_attn.k_proj.weight"]).view(B, q, nh, hd).transpose(1, 2)
        V = F.linear(x, l["self_attn.v_proj.weight"]).view(B, q, nh, hd).transpose(1, 2)
        if kv_past is not None:
            K = torch.cat([kv_past[0], K], 2)
            V = torch.cat([kv_past[1], V], 2)
        L = K.shape[2]
        R = F.linear(pos, l["self_attn.relative_k_proj.weight"]).view(1, 2 * L - 1, nh, hd).permute(0, 2, 3, 1)
        bd = (Q + l["self_attn.bias_v"].view(1, nh, 1, hd)) @ R                  # (B, nh, q, 2L-1)
        bd = F.pad(bd, (1, 0)).view(B, nh, -1, q)[:, :, 1:].reshape(B, nh, q, 2 * L - 1)[..., :L]
        scale = hd ** -0.5
        bd = (bd * scale).masked_fill(~mask[:, None], float("-inf"))
        ac = ((Q + l["self_attn.bias_u"].view(1, nh, 1, hd)) @ K.transpose(2, 3)) * scale
        w = torch.softmax(ac + bd, -1, dtype=torch.float32).to(Q.dtype)
        o = (w @ V).transpose(1, 2).reshape(B, q, nh * hd)
        return F.linear(o, l["self_attn.o_proj.weight"]), K, V

    def _conv(self, l, x, left=None):
        """Conv module; x (B, T, H); left (B, H, 8) causal context (None: zeros), updated copy returned."""
        h = F.conv1d(x.transpose(1, 2), l["conv.pointwise_conv1.weight"])
        h = F.glu(h, dim=1)
        pad = self.kconv - 1
        hp = torch.cat([left if left is not None else h.new_zeros(h.shape[0], h.shape[1], pad), h], 2)
        new_left = hp[:, :, -pad:]
        h = F.conv1d(hp, l["conv.depthwise_conv.weight"], groups=h.shape[1])
        h = self._ln(h.transpose(1, 2), l["conv.norm.weight"], l["conv.norm.bias"])
        h = F.silu(h).transpose(1, 2)
        return F.conv1d(h, l["conv.pointwise_conv2.weight"]).transpose(1, 2), new_left

    def _ff(self, l, x, n):
        return F.linear(F.silu(F.linear(x, l[f"feed_forward{n}.linear1.weight"])), l[f"feed_forward{n}.linear2.weight"])

    # ------------------------------------------------------------------ offline (window regime, transcription)
    @torch.no_grad()
    def encode(self, mel: torch.Tensor, lookahead: int, stop_after: int | None = None) -> torch.Tensor:
        """Whole clips (B, T, 128) -> hidden (B, T', 1024) after block ``stop_after`` (1-based; None = all 24), with
        the chunked-limited attention of ``lookahead``."""
        x = self._subsample(mel.to(self.dtype))
        T = x.shape[1]
        chunk = lookahead + 1
        lc = self.left // chunk
        i = torch.arange(T, device=self.dev)
        d = (i[:, None] // chunk) - (i[None, :] // chunk)
        mask = ((d >= 0) & (d <= lc))[None].expand(x.shape[0], -1, -1)
        pos = self._pos(T).to(self.dtype)
        for n, l in enumerate(self.L[:stop_after or self.n_layers]):
            x = x + 0.5 * self._ff(l, self._ln(x, l["norm_feed_forward1.weight"], l["norm_feed_forward1.bias"]), 1)
            a, _, _ = self._attn(l, self._ln(x, l["norm_self_att.weight"], l["norm_self_att.bias"]), mask, pos)
            x = x + a
            c, _ = self._conv(l, self._ln(x, l["norm_conv.weight"], l["norm_conv.bias"]))
            x = x + c
            x = x + 0.5 * self._ff(l, self._ln(x, l["norm_feed_forward2.weight"], l["norm_feed_forward2.bias"]), 2)
            x = self._ln(x, l["norm_out.weight"], l["norm_out.bias"])
        return x.float()

    # ------------------------------------------------------------------ streaming
    def new_pool(self, slots: int, lookahead: int = 3) -> dict:
        """State pools for ``slots`` sessions (zeros = a fresh stream)."""
        dev, H, nh, hd = self.dev, self.H, self.nh, self.hd
        return {"k": torch.zeros(self.n_layers, slots, nh, self.left, hd, device=dev),
                "v": torch.zeros(self.n_layers, slots, nh, self.left, hd, device=dev),
                "n": torch.zeros(slots, dtype=torch.long, device=dev),          # valid cached frames (<= 56)
                "conv": torch.zeros(self.n_layers, slots, H, self.kconv - 1, device=dev),
                "sub": [torch.zeros(slots, 1, 1, N_MELS + 3, device=dev),
                        torch.zeros(slots, 256, 1, (N_MELS + 3 - 3) // 2 + 1 + 3, device=dev),
                        torch.zeros(slots, 256, 1, ((N_MELS + 3 - 3) // 2 + 1 + 3 - 3) // 2 + 1 + 3, device=dev)],
                "lookahead": lookahead, "chunks": [0] * slots}

    def reset_slot(self, pool: dict, s: int) -> None:
        for t in (pool["k"], pool["v"], pool["conv"]):
            t[:, s].zero_()
        for t in pool["sub"]:
            t[s].zero_()
        pool["n"][s] = 0
        pool["chunks"][s] = 0

    @torch.no_grad()
    def stream_step(self, pool: dict, slots: list, mel: torch.Tensor, first: bool = False,
                    l15_out: bool = False):
        """One chunk for the sessions in ``slots`` (all first chunks or all later chunks): mel (B, 25|32, 128) ->
        last hidden (B, 4, 1024) [and layer 15]. Gathers the slots' state, runs one batched step, writes it back."""
        idx = torch.tensor(slots, device=self.dev)
        B = len(slots)
        subc = [t.index_select(0, idx) for t in pool["sub"]]
        x = self._subsample(mel, subc, first=first)
        for t, c in zip(pool["sub"], subc):
            t.index_copy_(0, idx, c)
        q = x.shape[1]
        P = self.left
        n = pool["n"].index_select(0, idx)                                      # valid past frames per row
        valid = torch.arange(P, device=self.dev)[None] >= (P - n)[:, None]       # right-aligned
        mask = torch.cat([valid, torch.ones(B, q, dtype=torch.bool, device=self.dev)], 1)[:, None, :].expand(-1, q, -1)
        pos = self._pos(P + q)
        Kp, Vp, Cp = pool["k"].index_select(1, idx), pool["v"].index_select(1, idx), pool["conv"].index_select(1, idx)
        l15 = None
        for li, l in enumerate(self.L):
            x = x + 0.5 * self._ff(l, self._ln(x, l["norm_feed_forward1.weight"], l["norm_feed_forward1.bias"]), 1)
            a, K, V = self._attn(l, self._ln(x, l["norm_self_att.weight"], l["norm_self_att.bias"]), mask, pos,
                                 (Kp[li], Vp[li]))
            Kp[li], Vp[li] = K[:, :, -P:], V[:, :, -P:]
            x = x + a
            c, Cp[li] = self._conv(l, self._ln(x, l["norm_conv.weight"], l["norm_conv.bias"]), Cp[li])
            x = x + c
            x = x + 0.5 * self._ff(l, self._ln(x, l["norm_feed_forward2.weight"], l["norm_feed_forward2.bias"]), 2)
            x = self._ln(x, l["norm_out.weight"], l["norm_out.bias"])
            if li == 14:
                l15 = x
        pool["k"].index_copy_(1, idx, Kp)
        pool["v"].index_copy_(1, idx, Vp)
        pool["conv"].index_copy_(1, idx, Cp)
        pool["n"].index_copy_(0, idx, (n + q).clamp(max=P))
        for s in slots:
            pool["chunks"][s] += 1
        return (x, l15) if l15_out else x

    # ------------------------------------------------------------------ RNN-T
    def project(self, last: torch.Tensor, prompt_ids: list) -> torch.Tensor:
        """last (B, T, 1024) + one language prompt per row -> joint inputs (B, T, 640)."""
        W = self.W
        oh = F.one_hot(torch.tensor(prompt_ids, device=self.dev), self.num_prompts).to(last.dtype)
        oh = oh[:, None, :].expand(-1, last.shape[1], -1)
        h = F.linear(torch.cat([last, oh], -1), W["prompt_projector.linear_1.weight"], W["prompt_projector.linear_1.bias"])
        h = F.linear(F.relu(h), W["prompt_projector.linear_2.weight"], W["prompt_projector.linear_2.bias"])
        return F.linear(h, W["encoder_projector.weight"], W["encoder_projector.bias"])

    def _lstm(self, x, h, c):
        """2-layer LSTM, one step. x (B, 640); h, c (2, B, 640)."""
        W = self.W
        hs, cs = [], []
        for i in range(2):
            g = F.linear(x, W[f"decoder.lstm.weight_ih_l{i}"], W[f"decoder.lstm.bias_ih_l{i}"]) + \
                F.linear(h[i], W[f"decoder.lstm.weight_hh_l{i}"], W[f"decoder.lstm.bias_hh_l{i}"])
            ig, fg, gg, og = g.chunk(4, -1)
            ci = torch.sigmoid(fg) * c[i] + torch.sigmoid(ig) * torch.tanh(gg)
            hi = torch.sigmoid(og) * torch.tanh(ci)
            hs.append(hi)
            cs.append(ci)
            x = hi
        return F.linear(x, W["decoder.decoder_projector.weight"], W["decoder.decoder_projector.bias"]), \
            torch.stack(hs), torch.stack(cs)

    def pred_start(self, B: int):
        """Predictor state after the blank start token: (d (B, 640), h, c (2, B, 640))."""
        W = self.W
        e = W["decoder.embedding.weight"][torch.full((B,), self.blank, device=self.dev)]
        z = torch.zeros(2, B, 640, device=self.dev)
        return self._lstm(e, z, z.clone())

    @torch.no_grad()
    def greedy(self, enc: torch.Tensor, d, h, c, sync_every: int = 4):
        """Greedy RNN-T over enc (B, T, 640) for B rows with their predictor states -> ([[(t, tok)]], d, h, c)."""
        W = self.W
        B, T, _ = enc.shape
        ar = torch.arange(B, device=self.dev)
        idx = torch.zeros(B, dtype=torch.long, device=self.dev)
        sym = torch.zeros(B, dtype=torch.long, device=self.dev)
        toks, frames, valid = [], [], []
        for step in range(T * (self.max_sym + 1)):
            done = idx >= T
            e = enc[ar, idx.clamp(max=T - 1)]
            tok = F.linear(F.relu(e + d), W["joint.head.weight"], W["joint.head.bias"]).argmax(-1)
            is_blank = (tok == self.blank) | done
            emit = ~is_blank
            toks.append(tok)
            frames.append(idx)
            valid.append(emit)
            sym = torch.where(is_blank, torch.zeros_like(sym), sym + 1)
            force = sym >= self.max_sym
            idx = idx + (is_blank | force).long()
            sym = torch.where(force, torch.zeros_like(sym), sym)
            d2, h2, c2 = self._lstm(W["decoder.embedding.weight"][tok], h, c)
            u = emit[:, None]
            d = torch.where(u, d2, d)
            h = torch.where(u[None], h2, h)
            c = torch.where(u[None], c2, c)
            if (step + 1) % sync_every == 0 and bool(host((idx >= T).all())):
                break
        tk, fr, va = host(torch.stack(toks, 1)), host(torch.stack(frames, 1)), host(torch.stack(valid, 1))
        return [[(fr[i][s], tk[i][s]) for s in range(len(tk[i])) if va[i][s]] for i in range(B)], d, h, c

    # ---- greedy RNN-T as one CUDA graph per (row bucket, frames): the eager loop issues ~15 tiny kernels per step
    # plus a host sync every 4 steps; next to another model's kernels on the GPU each of them waits its turn, so the
    # loop's latency grows with contention. The graph runs the full T * (max_sym + 1) steps (the eager loop's extra
    # steps after every row is done are no-ops: blank, no emit, state unchanged), one launch, one sync at the end.
    GREEDY_B = (1, 2, 4, 8, 12, 16)

    def _greedy_steps(self, enc, d, h, c, steps):
        W = self.W
        B, T, _ = enc.shape
        ar = torch.arange(B, device=self.dev)
        idx = torch.zeros(B, dtype=torch.long, device=self.dev)
        sym = torch.zeros(B, dtype=torch.long, device=self.dev)
        toks, frames, valid = [], [], []
        for _ in range(steps):
            done = idx >= T
            e = enc[ar, idx.clamp(max=T - 1)]
            tok = F.linear(F.relu(e + d), W["joint.head.weight"], W["joint.head.bias"]).argmax(-1)
            is_blank = (tok == self.blank) | done
            emit = ~is_blank
            toks.append(tok)
            frames.append(idx)
            valid.append(emit)
            sym = torch.where(is_blank, torch.zeros_like(sym), sym + 1)
            force = sym >= self.max_sym
            idx = idx + (is_blank | force).long()
            sym = torch.where(force, torch.zeros_like(sym), sym)
            d2, h2, c2 = self._lstm(W["decoder.embedding.weight"][tok], h, c)
            u = emit[:, None]
            d = torch.where(u, d2, d)
            h = torch.where(u[None], h2, h)
            c = torch.where(u[None], c2, c)
        return torch.stack(toks, 1), torch.stack(frames, 1), torch.stack(valid, 1), d, h, c

    def warm_greedy_graphs(self, frames=(3, 4)) -> int:
        """Capture the greedy graphs for every row bucket and the given chunk lengths (call before other threads use
        the GPU: a capture fails if another thread synchronises meanwhile)."""
        self._gg = getattr(self, "_gg", {})
        n = 0
        for T in frames:
            for B in self.GREEDY_B:
                key = (B, T)
                if key in self._gg:
                    continue
                d0, h0, c0 = self.pred_start(B)
                g = {"enc": torch.zeros(B, T, self.W["joint.head.weight"].shape[1], device=self.dev,
                                        dtype=self.W["joint.head.weight"].dtype),
                     "d": d0.clone(), "h": h0.clone(), "c": c0.clone()}
                run = lambda g=g, T=T: self._greedy_steps(g["enc"], g["d"], g["h"], g["c"],   # noqa: E731
                                                          T * (self.max_sym + 1))
                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    run()
                torch.cuda.current_stream().wait_stream(side)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, capture_error_mode="thread_local"):
                    g["out"] = run()
                g["graph"] = graph
                self._gg[key] = g
                n += 1
        return n

    def greedy_fast(self, enc, d, h, c):
        """greedy() through a captured graph when one fits (same results), else the eager loop."""
        B, T, _ = enc.shape
        gg = getattr(self, "_gg", {})
        Bb = next((x for x in self.GREEDY_B if x >= B), None)
        g = gg.get((Bb, T)) if Bb is not None else None
        if g is None:
            return self.greedy(enc, d, h, c)
        g["enc"].zero_()
        g["enc"][:B] = enc
        g["d"][:B], g["h"][:, :B], g["c"][:, :B] = d, h, c
        g["graph"].replay()
        tk, fr, va, d2, h2, c2 = g["out"]
        tk, fr, va = tk[:B].tolist(), fr[:B].tolist(), va[:B].tolist()
        out = [[(fr[i][s], tk[i][s]) for s in range(len(tk[i])) if va[i][s]] for i in range(B)]
        return out, d2[:B].clone(), h2[:, :B].clone(), c2[:, :B].clone()

    def pieces(self, ids: list) -> list:
        return [self.tokenizer.id_to_token(i) for i in ids]

    def prompt_id(self, lang: str) -> int:
        if lang in self.prompts:
            return int(self.prompts[lang])
        from .. import langs                     # a language.json's asr_locale (e.g. "sv-SE"), else auto-detect
        loc = langs.get(lang).asr_locale
        return int(self.prompts.get(loc, {"auto": 101, "en": 0, "it": 15, "fr": 8}.get(lang, 101)))
