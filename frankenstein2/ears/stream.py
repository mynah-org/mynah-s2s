"""The ears: ONE cache-aware streaming Nemotron 3.5 ASR encoder (lookahead k=3: 320 ms chunks) for two channels.

Per 320 ms hop:
  * the user mic and the agent "2nd ear" channel go through the encoder as one batch of 2, each row with its own
    attention K/V and convolution caches (transformers' cache-aware streaming: ``use_cache=True``,
    ``chunked_limited`` attention [sliding_window - 1, k]); no 8 s window re-encode;
  * a forward hook keeps layer 15 (the turn-taking GRU's input) of both rows;
  * the user row's last layer + the explicit language prompt go through the RNN-T head, decoded greedily and
    incrementally (the predictor state carries over hops): the live transcript;
  * the GRU steps 4 frames with the rules (rules.py) under the mode the controller forces.

Chunking (from the processor): the first chunk is 1 + 8k mel frames (center=True), later chunks 8(k+1) mel frames
fed from audio[f0*hop - n_fft/2 : ...] with center=False. With k=3: 25 then 32 mel frames, 4 encoder frames per
chunk; hop n needs audio up to sample 5120 n + 4144, so every hop of 5120 new samples completes exactly one chunk.
GPU or CPU; fp32, TF32 off (the GRU was validated so). Runs in the ears process (its own environment).
"""

from __future__ import annotations

import time

import numpy as np
import torch

from .rules import HOP_FRAMES, RulesState, Thresholds, energy_db
from .turntaker import L15, GruSession

HOP = 5120
FRAME = 1280


def is_tag(p: str) -> bool:
    s = p.strip().replace("▁", "")
    return s.startswith("<") and s.endswith(">")


class StreamChannelPair:
    """Streaming encoder state for (user, agent) of one session."""

    def __init__(self, model, proc, device, k: int = 3, gru_k: int | None = 0):
        """k: the ASR lookahead (chunk = k + 1 frames). gru_k: the lookahead of the GRU's layer-15 features; None or k
        = one shared pass (L15 of the ASR pass for both channels); otherwise a second cache-aware pass with that
        lookahead over the same mel chunk, stopped at layer 15, for both channels (the ASR pass then runs the user
        row only). Chunk sizes are set by k; a smaller gru_k only changes the attention mask (causal within the
        chunk), so both passes advance together."""
        from transformers import DynamicCache
        from transformers.models.nemotron_asr_streaming.modeling_nemotron_asr_streaming import \
            NemotronAsrStreamingEncoderCausalConvPaddingCache as PC
        self.model, self.proc, self.dev, self.k = model, proc, device, k
        self.gru_k = None if gru_k is None or gru_k == k else gru_k
        self.fe = proc.feature_extractor
        self.enc = model.encoder
        self.pkv, self.pc = DynamicCache(config=self.enc.config), PC()
        self.pkv_g, self.pc_g = DynamicCache(config=self.enc.config), PC()
        sub = self.enc.config.subsampling_factor
        self.first_mel = 1 + sub * k
        self.mel_per = sub * (k + 1)
        self.n_fft, self.hop_len, self.win = self.fe.n_fft, self.fe.hop_length, self.fe.win_length
        self.chunks = 0
        self.mel_done = 0
        self.buf = np.zeros((2, 0), np.float32)
        self.buf0 = 0                         # absolute sample index of buf[:, 0]

    def need(self) -> int:
        """Absolute sample index the next chunk needs audio up to (exclusive)."""
        if self.chunks == 0:
            return (self.first_mel - 1) * self.hop_len + self.win // 2
        return self.mel_done * self.hop_len - self.n_fft // 2 + self.mel_per * self.hop_len + self.win

    def push(self, user: np.ndarray, agent: np.ndarray) -> None:
        self.buf = np.concatenate([self.buf, np.stack([user, agent]).astype(np.float32)], 1)

    def ready(self) -> bool:
        return self.buf0 + self.buf.shape[1] >= self.need()

    @torch.no_grad()
    def chunk(self):
        """-> (L15 (2, T, 1024), last layer (2, T, 1024)) for the next chunk."""
        if self.chunks == 0:
            lo, hi, center = 0, self.need(), True
        else:
            lo = self.mel_done * self.hop_len - self.n_fft // 2
            hi = lo + self.mel_per * self.hop_len + self.win
            center = False
        x = self.buf[:, lo - self.buf0: hi - self.buf0]
        f = self.fe(list(x), sampling_rate=16000, center=center, return_tensors="pt")
        feats = f["input_features"].to(self.dev)
        want = self.first_mel if self.chunks == 0 else self.mel_per
        feats = feats[:, :want]
        held = {}

        def hook(mod, args, out):
            held["h"] = out[0] if isinstance(out, tuple) else out
            if held.get("stop"):
                raise _StopAtL15
        hd = self.enc.layers[L15 - 1].register_forward_hook(hook)
        try:
            one = self.gru_k is not None or getattr(self, "user_only", False)
            o = self.enc(input_features=feats[:1] if one else feats, past_key_values=self.pkv,
                         padding_cache=self.pc, use_cache=True, num_lookahead_tokens=self.k,
                         output_attention_mask=False)
            l15 = held["h"]
            if self.gru_k is not None and not getattr(self, "user_only", False):
                held["stop"] = True
                try:
                    self.enc(input_features=feats, past_key_values=self.pkv_g, padding_cache=self.pc_g,
                             use_cache=True, num_lookahead_tokens=self.gru_k, output_attention_mask=False)
                except _StopAtL15:
                    pass
                l15 = held["h"]
        finally:
            hd.remove()
        self.pkv, self.pc = o.past_key_values, o.padding_cache
        self.mel_done += feats.shape[1]
        self.chunks += 1
        keep_from = self.mel_done * self.hop_len - self.n_fft // 2  # drop audio no later chunk reads
        if keep_from > self.buf0:
            self.buf = self.buf[:, keep_from - self.buf0:]
            self.buf0 = keep_from
        return l15.float(), o.last_hidden_state


class _StopAtL15(Exception):
    pass


class RnntStream:
    """Greedy RNN-T over a stream of encoder frames for one channel, predictor state carried over calls."""

    def __init__(self, model, prompt_id: int, device):
        self.m, self.dev = model, device
        cfg = model.config
        self.blank, self.max_sym, self.num_prompts = cfg.blank_token_id, model.max_symbols_per_step, cfg.num_prompts
        self.prompt_id = prompt_id
        self.tok = None
        self.fresh()

    def _dec(self, tok, hc):
        dec = self.m.decoder
        out, (h, c) = dec.lstm(dec.embedding(torch.tensor([[tok]], device=self.dev)), hc)
        return dec.decoder_projector(out)[:, 0], (h, c)

    def fresh(self):
        self.d, self.hc = self._dec(self.blank, None)
        self.resets = getattr(self, "resets", 0)

    @torch.no_grad()
    def project(self, last):
        oh = torch.nn.functional.one_hot(torch.tensor([self.prompt_id], device=self.dev),
                                         num_classes=self.num_prompts).to(last.dtype)
        oh = oh[:, None, :].expand(-1, last.shape[1], -1)
        return self.m.encoder_projector(self.m.prompt_projector(torch.cat([last, oh], -1)))

    @torch.no_grad()
    def decode(self, enc):
        """enc: (T, H) projected frames -> list of (frame offset, token id)."""
        out = []
        for t in range(enc.shape[0]):
            for _ in range(self.max_sym):
                tok = int(self.m.joint(decoder_hidden_states=self.d, encoder_hidden_states=enc[t:t + 1]).argmax(-1))
                if tok == self.blank:
                    break
                out.append((t, tok))
                self.d, self.hc = self._dec(tok, self.hc)
        return out


class EarsSession:
    """Everything the ears keep for one conversation.

    gru_features: "window" (default: Tyger's trained regime, window.py) or "stream" (layer 15 of the streaming
    passes; gru_k sets their lookahead; measured worse for endpoints, kept for experiments)."""

    def __init__(self, model, proc, turntaker, device, lang_prompt: int, thr: Thresholds, k: int = 3,
                 gru_k: int | None = 0, gru_features: str = "window"):
        self.mode_feat = gru_features
        if gru_features == "window":
            from .window import WindowL15
            self.window = WindowL15(model, proc, device)
            self.pair = StreamChannelPair(model, proc, device, k, None)
            self.pair.user_only = True
        else:
            self.window = None
            self.pair = StreamChannelPair(model, proc, device, k, gru_k)
        self.rnnt = RnntStream(model, lang_prompt, device)
        self.gru = GruSession(turntaker, device)
        self.rules = RulesState(thr=thr)
        self.proc = proc
        self.dev = device
        self.hops = 0
        self.frames_in = 0                    # ASR encoder frames produced
        self.gru_frames = 0
        self.user = np.zeros(0, np.float32)   # user samples not yet consumed by the energy frames (stream mode)
        self.e_frames = 0

    def begin(self, mode: str, reset: bool) -> None:
        if reset:
            self.rules.reset(keep_turn=reset == "turn")
            self.gru.reset()
        if self.rules.switch_mode(mode):
            self.gru.reset()

    def _gru_frames(self, l15u, l15a, energies, frames, events):
        T = len(energies)
        for j in range(T):
            out = self.gru.step(l15u[j], l15a[j], energies[j], self.rules.gru_prev_state())
            evs, rst = self.rules.on_frame(out, energies[j], left=T - 1 - j)
            for e in evs:
                (frames if e["kind"] == "FRAME" else events).append(e)
            if rst:
                self.gru.reset()
        self.gru_frames += T

    def _sync(self):
        if self.dev != "cpu":
            torch.cuda.synchronize()

    def hop(self, user: np.ndarray, agent: np.ndarray, mode: str, reset: bool = False, l15=None,
            gru_done: tuple | None = None) -> dict:
        """One 320 ms hop of both channels. Returns frames (head outputs + rules view), events, ASR pieces.
        ``l15``: the window features already computed for this hop by a batched encode (hop_batch)."""
        t0 = time.monotonic()
        if gru_done is None:
            self.begin(mode, reset)
        user = user.astype(np.float32)
        self.pair.push(user, agent)
        frames, events, pieces = [], [], []
        t_enc = t_win = 0.0
        if gru_done is not None:                 # hop_batch ran the GRU frames of all sessions together
            frames, events = gru_done
        elif self.window is not None:
            tw = time.monotonic()
            hu, ha = l15 if l15 is not None else self.window.hop(user, agent.astype(np.float32))
            self._sync()
            t_win = time.monotonic() - tw
            dbs = [energy_db(user[j * FRAME:(j + 1) * FRAME]) for j in range(len(user) // FRAME)]
            self._gru_frames(hu, ha, dbs, frames, events)
        else:
            self.user = np.concatenate([self.user, user])
        while self.pair.ready():
            te = time.monotonic()
            l15, last = self.pair.chunk()
            self._sync()
            t_enc += time.monotonic() - te
            T = last.shape[1]
            for t, tok in self.rnnt.decode(self.rnnt.project(last[:1])[0]):
                p = self.proc.tokenizer.convert_ids_to_tokens(tok)
                pieces.append({"f": self.frames_in + t, "p": p, "id": tok})
                # A language tag (e.g. <it-IT> after a sentence) is dropped from the text but does NOT reset the
                # predictor: f2-003 showed a reset there deleting the next words ("Mi [chiamo Mario] Rossi").
            if self.window is None:
                dbs = []
                for j in range(T):
                    seg = self.user[j * FRAME:(j + 1) * FRAME]
                    dbs.append(energy_db(seg) if len(seg) == FRAME else -120.0)
                self._gru_frames(l15[0], l15[1], dbs, frames, events)
                self.user = self.user[T * FRAME:]
            self.frames_in += T
        self.hops += 1
        return {"frames": frames, "events": events, "pieces": pieces, "enc_ms": round(t_enc * 1e3, 2),
                "win_ms": round(t_win * 1e3, 2), "ms": round((time.monotonic() - t0) * 1e3, 2),
                "frames_total": self.frames_in}


@torch.no_grad()
def gru_batch_frames(sessions: list, hus: list, has: list, dbs: list) -> list:
    """The GRU frames of several sessions as one batch per frame (each session keeps its own state row and rules).
    -> [(frames, events)] per session."""
    m = sessions[0].gru.m
    dev = sessions[0].dev
    L, H = m.core.num_layers, m.core.hidden_size
    out = [([], []) for _ in sessions]
    T = len(dbs[0])
    hu = torch.stack([x[:T] for x in hus]).float()           # (B, T, 1024)
    ha = torch.stack([x[:T] for x in has]).float()
    for j in range(T):
        h = torch.cat([s.gru.h if s.gru.h is not None else torch.zeros(L, 1, H, device=dev) for s in sessions], 1)
        act = torch.tensor([[max(-3.0, min(3.0, (d[j] + 50.0) / 25.0))] for d in dbs], dtype=torch.float32,
                           device=dev)
        prev = torch.tensor([[s.rules.gru_prev_state()] for s in sessions], dtype=torch.long, device=dev)
        o, h2 = m(hu[:, j:j + 1], ha[:, j:j + 1], act, prev, h)
        rows = o[:, 0].tolist()
        for i, s in enumerate(sessions):
            s.gru.h = h2[:, i:i + 1].contiguous()
            evs, rst = s.rules.on_frame(rows[i], dbs[i][j], left=T - 1 - j)
            for e in evs:
                (out[i][0] if e["kind"] == "FRAME" else out[i][1]).append(e)
            if rst:
                s.gru.reset()
    for s in sessions:
        s.gru_frames += T
    return out


def hop_batch(items: list, batched_asr: bool = True) -> list:
    """items: [(EarsSession, user, agent, mode, reset)] of DIFFERENT sessions -> their hop results, everything batched
    across sessions: the GRU's window encodes, the GRU frames, the streaming ASR chunks (one encoder forward, per-row
    language prompt, per-session caches: ears/batched.py) and the greedy RNN-T. Results equal per-session hops."""
    from .batched import rnnt_batch, stream_chunks
    t0 = time.monotonic()
    for s, u, a, m, r in items:
        s.begin(m, r)
    win = [(s, s.window.push(u.astype(np.float32), a.astype(np.float32))) for s, u, a, m, r in items
           if s.window is not None]
    feats = {}
    if win:
        flat = [w for _, ws in win for w in ws]
        enc = win[0][0].window.encode(flat)
        i = 0
        for s, ws in win:
            feats[id(s)] = s.window.take(enc[i:i + len(ws)])
            i += len(ws)
    gru = {}
    ws_ = [x for x in items if id(x[0]) in feats]
    if ws_:
        dbs = [[energy_db(u[j * FRAME:(j + 1) * FRAME]) for j in range(len(u) // FRAME)] for _, u, *_ in ws_]
        res = gru_batch_frames([x[0] for x in ws_], [feats[id(x[0])][0] for x in ws_],
                               [feats[id(x[0])][1] for x in ws_], dbs)
        for x, r in zip(ws_, res):
            gru[id(x[0])] = r
    t_win = time.monotonic() - t0
    if not batched_asr:
        out = []
        for s, u, a, m, r in items:
            res = s.hop(u, a, m, r, gru_done=gru.get(id(s), ([], [])))
            res["win_ms"] = round(t_win * 1e3 / max(1, len(items)), 2)
            res["batch"] = len(items)
            out.append(res)
        return out
    t1 = time.monotonic()
    pieces = {id(s): [] for s, *_ in items}
    for s, u, a, m, r in items:
        s.pair.push(u.astype(np.float32), a)
    while True:
        ready = [s for s, *_ in items if s.pair.ready()]
        if not ready:
            break
        lasts = stream_chunks([s.pair for s in ready])
        toks = rnnt_batch([s.rnnt for s in ready], lasts)
        for s, last, tk in zip(ready, lasts, toks):
            for t, tok in tk:
                pieces[id(s)].append({"f": s.frames_in + t, "p": s.proc.tokenizer.convert_ids_to_tokens(tok),
                                      "id": tok})
            s.frames_in += last.shape[1]
    if items[0][0].dev != "cpu":
        torch.cuda.synchronize()
    t_asr = time.monotonic() - t1
    out = []
    for s, u, a, m, r in items:
        fr, ev = gru.get(id(s), ([], []))
        s.hops += 1
        out.append({"frames": fr, "events": ev, "pieces": pieces[id(s)], "enc_ms": round(t_asr * 1e3 / len(items), 2),
                    "win_ms": round(t_win * 1e3 / len(items), 2), "ms": round((time.monotonic() - t0) * 1e3, 2),
                    "frames_total": s.frames_in, "batch": len(items)})
    return out


@torch.no_grad()
def transcribe_offline(model, proc, wav: np.ndarray, prompt_id: int, device, k: int = 3) -> str:
    """Whole-utterance transcript (content evidence): the same encoder with the chunked attention of lookahead k,
    greedy RNN-T with the explicit language prompt."""
    f = proc.feature_extractor(wav.astype(np.float32), sampling_rate=16000, return_tensors="pt")
    o = model.encoder(input_features=f["input_features"].to(device), num_lookahead_tokens=k)
    r = RnntStream(model, prompt_id, device)
    toks = [tok for _, tok in r.decode(r.project(o.last_hidden_state)[0])]
    pieces = proc.tokenizer.convert_ids_to_tokens(toks)
    return "".join(p for p in pieces if not is_tag(p)).replace("▁", " ").strip()
