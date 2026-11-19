"""The ears on our own runtime (ears/nemotron.py + ears/turntaker.py + ears/rules.py), for many sessions at once.

``OwnEars`` holds the model once and a pool of session slots. ``hop(items)`` runs one 320 ms hop for several sessions
as a handful of batched GPU calls, whatever the number of sessions:
  1. the GRU's window features: the last 8 s of each session's mic and 2nd ear, log-mel + encoder to layer 15 with
     lookahead 0 (all-zero agent windows served from a cache), one call per window length;
  2. the GRU frames: one batched step per frame (each session its own state row and rules);
  3. the streaming ASR: one cache-aware encoder step for every session with a chunk ready (first chunks in their own
     call), the language prompt per row, one batched greedy RNN-T loop.
Results equal per-session runs of the reference implementation (lab/nemotron_check.py, lab/ears_own_check.py).
"""

from __future__ import annotations

import time

import numpy as np
import torch

from ..gpu import host, ready
from .nemotron import HOP_LEN, NemotronASR
from .rules import RulesState, Thresholds, energy_db

HOP = 5120
FRAME = 1280
WINDOW = 128000
FIRST_MEL, MEL_PER, N_FFT, WIN = 25, 32, 512, 400


class _Sess:
    def __init__(self, slot: int, prompt: int, thr: Thresholds):
        self.slot, self.prompt = slot, prompt
        self.rules = RulesState(thr=thr)
        self.win = np.zeros((2, 0), np.float32)
        self.buf = np.zeros(0, np.float32)              # mic samples not consumed by the streaming ASR
        self.buf0 = 0
        self.mel_done = 0
        self.chunks = 0
        self.frames_in = 0
        self.gru_h = None
        self.hops = 0

    def need(self) -> int:
        if self.chunks == 0:
            return (FIRST_MEL - 1) * HOP_LEN + WIN // 2
        return self.mel_done * HOP_LEN - N_FFT // 2 + MEL_PER * HOP_LEN + WIN

    def ready(self) -> bool:
        return self.buf0 + len(self.buf) >= self.need()


class OwnEars:
    def __init__(self, bundle: str, device="cuda", slots: int = 16, window_precision: str = "tf32"):
        from .turntaker import TurnTaker
        import json
        from pathlib import Path
        from safetensors import safe_open
        self.asr = NemotronASR(bundle, device)
        self.dev = self.asr.dev
        # the GRU's 8 s window encode (layers 1-15, lookahead 0): fp32 | tf32 (events identical to fp32) | bf16 (own
        # bf16 copy of the encoder; ~2.4x faster, a few events move by one 80 ms frame: lab/ears_featpath_check.py)
        self.window_precision = window_precision
        self.window_tf32 = window_precision == "tf32"
        self.asr_win = NemotronASR(bundle, device, dtype=torch.bfloat16) if window_precision == "bf16" else self.asr
        cfg = json.loads((Path(bundle) / "config.json").read_text())
        arch = cfg["controller"]["architecture"]
        with torch.device(self.dev):
            self.gru = TurnTaker(arch["width"], arch["layers"]).eval()
        sd = {}
        with safe_open(str(Path(bundle) / "model.safetensors"), framework="pt", device=str(self.dev)) as f:
            for k in f.keys():
                if k.startswith("turn_taker."):
                    sd[k[len("turn_taker."):]] = f.get_tensor(k)
        self.gru.load_state_dict(sd, strict=True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.pool = self.asr.new_pool(slots)
        d, h, c = self.asr.pred_start(slots)
        self.pred = {"d": d, "h": h, "c": c}
        self.free = list(range(slots))
        self.sess: dict = {}
        self._zero_l15: dict = {}
        self.greedy_graphs = False                      # warm_greedy_graphs() turns the graph path on
        self.prof = None                                # dict: per-phase GPU-synchronised times (lab profiling)

    def warm_greedy_graphs(self) -> int:
        n = self.asr.warm_greedy_graphs()
        self.greedy_graphs = n > 0
        return n

    # ------------------------------------------------------------------ sessions
    def open(self, sid, lang: str, thr: Thresholds | None = None) -> None:
        if sid in self.sess:
            self.close(sid)
        if not self.free:
            raise RuntimeError(f"ears: no free slot ({len(self.sess)} sessions open)")
        s = self.free.pop(0)
        self.asr.reset_slot(self.pool, s)
        d, h, c = self.asr.pred_start(1)
        self.pred["d"][s], self.pred["h"][:, s], self.pred["c"][:, s] = d[0], h[:, 0], c[:, 0]
        self.sess[sid] = _Sess(s, self.asr.prompt_id(lang), thr or Thresholds())

    def close(self, sid) -> None:
        s = self.sess.pop(sid, None)
        if s is not None:
            self.free.append(s.slot)

    # ------------------------------------------------------------------ one hop for many sessions
    @torch.no_grad()
    def hop(self, items: list) -> list:
        """items: [(sid, user (5120,), agent (5120,), mode, reset)] of different sessions -> results (as
        EarsSession.hop: frames, events, pieces, timings)."""
        t0 = time.monotonic()
        S = [self.sess[x[0]] for x in items]
        for s, (sid, u, a, mode, reset) in zip(S, items):
            if reset:
                s.rules.reset(keep_turn=reset == "turn")
                s.gru_h = None
            if s.rules.switch_mode(mode):
                s.gru_h = None
            s.win = np.concatenate([s.win, np.stack([u, a]).astype(np.float32)], 1)[:, -WINDOW:]
            s.buf = np.concatenate([s.buf, u.astype(np.float32)])
        prof = self.prof
        hu, ha = self._windows(S)
        if prof is not None:
            torch.cuda.synchronize()
            tw = time.monotonic()
            prof.setdefault("win_ms", []).append((tw - t0) * 1e3)
        t_win = time.monotonic() - t0
        frames, events = self._gru(S, hu, ha, [x[1] for x in items])
        t1 = time.monotonic()
        if prof is not None:
            torch.cuda.synchronize()
            t1 = time.monotonic()
            prof.setdefault("gru_ms", []).append((t1 - tw) * 1e3)
        pieces = self._stream(S)
        if self.dev.type == "cuda":
            ready()
        t_asr = time.monotonic() - t1
        if prof is not None:
            prof.setdefault("asr_ms", []).append(t_asr * 1e3)
            prof.setdefault("hop_ms", []).append((time.monotonic() - t0) * 1e3)
        out = []
        for i, s in enumerate(S):
            s.hops += 1
            out.append({"frames": frames[i], "events": events[i], "pieces": pieces[i], "frames_total": s.frames_in,
                        "win_ms": round(t_win * 1e3 / len(S), 2), "enc_ms": round(t_asr * 1e3 / len(S), 2),
                        "ms": round((time.monotonic() - t0) * 1e3, 2), "batch": len(S)})
        return out

    def _windows(self, S):
        rows, owner = [], []
        out_u, out_a = [None] * len(S), [None] * len(S)
        for i, s in enumerate(S):
            n = s.win.shape[1]
            rows.append(s.win[0])
            owner.append((i, 0))
            if not s.win[1].any() and n in self._zero_l15:
                out_a[i] = self._zero_l15[n]
            else:
                rows.append(s.win[1])
                owner.append((i, 1))
        groups: dict = {}
        for r, o in zip(rows, owner):
            groups.setdefault(len(r), []).append((r, o))
        for n, grp in groups.items():
            x = torch.from_numpy(np.stack([r for r, _ in grp])).to(self.dev)
            h = self._encode_window(x)
            for j, (_, (i, ch)) in enumerate(grp):
                if ch == 0:
                    out_u[i] = h[j]
                else:
                    out_a[i] = h[j]
                    if not S[i].win[1].any():
                        self._zero_l15[n] = h[j].clone()
        return out_u, out_a

    # TF32 for the 8 s window encode (the largest ears cost): GRU outputs within 0.003 of fp32 and identical events on
    # IT/EN/FR with silent and speaking agent channels (lab/ears_featpath_check.py --precisions). The GRU itself and
    # the streaming ASR stay fp32 with TF32 off, as the bundle requires. window_tf32=False restores plain fp32.
    WIN_B = (1, 2, 4, 8, 12, 16, 24, 32)       # row buckets of the full-window encode graphs (rows = mic + agent)

    def warm_window_graphs(self, max_rows: int | None = None) -> int:
        """Capture the full 8 s window encode (fixed shape) for every row bucket now, then freeze (a later miss runs
        eager): captures must not happen while other threads use the GPU (the one-process engine's LLM thread)."""
        if self.dev.type != "cuda":
            return 0
        self._wg = getattr(self, "_wg", {})
        top = max_rows or self.WIN_B[-1]
        n = 0
        for b in self.WIN_B:
            if b > top or b in self._wg:
                continue
            xs = torch.zeros(b, WINDOW, device=self.dev)
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(2):
                    self._encode_window_eager(xs)
            torch.cuda.current_stream().wait_stream(side)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=self._wg.get("pool"), capture_error_mode="thread_local"):
                out = self._encode_window_eager(xs)
            self._wg.setdefault("pool", g.pool())
            self._wg[b] = {"x": xs, "out": out, "graph": g}
            n += 1
        self._wg_frozen = True
        return n

    def _encode_window(self, x):
        wg = getattr(self, "_wg", None)
        if wg and x.shape[1] == WINDOW and x.shape[0] <= self.WIN_B[-1]:
            b = next(v for v in self.WIN_B if v >= x.shape[0])
            g = wg.get(b)
            if g is not None:
                g["x"][:x.shape[0]].copy_(x)
                if x.shape[0] < b:
                    g["x"][x.shape[0]:].zero_()
                g["graph"].replay()
                return g["out"][:x.shape[0]].clone()
        return self._encode_window_eager(x)

    def _encode_window_eager(self, x):
        tf = self.window_tf32 and self.dev.type == "cuda"
        torch.backends.cuda.matmul.allow_tf32 = tf
        torch.backends.cudnn.allow_tf32 = tf
        try:
            w = self.asr_win
            return w.encode(w.mel(x, center=True), lookahead=0, stop_after=15)[:, -4:]
        finally:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False

    def _gru(self, S, hu, ha, users):
        m = self.gru
        L, H = m.core.num_layers, m.core.hidden_size
        frames, events = [[] for _ in S], [[] for _ in S]
        dbs = [[energy_db(u[j * FRAME:(j + 1) * FRAME]) for j in range(4)] for u in users]
        xu, xa = torch.stack(hu), torch.stack(ha)
        for j in range(4):
            h = torch.cat([s.gru_h if s.gru_h is not None else torch.zeros(L, 1, H, device=self.dev) for s in S], 1)
            act = torch.tensor([[max(-3.0, min(3.0, (d[j] + 50.0) / 25.0))] for d in dbs], device=self.dev)
            prev = torch.tensor([[s.rules.gru_prev_state()] for s in S], dtype=torch.long, device=self.dev)
            o, h2 = m(xu[:, j:j + 1], xa[:, j:j + 1], act, prev, h)
            rows = host(o[:, 0])
            for i, s in enumerate(S):
                s.gru_h = h2[:, i:i + 1].contiguous()
                evs, rst = s.rules.on_frame(rows[i], dbs[i][j], left=3 - j)
                for e in evs:
                    (frames[i] if e["kind"] == "FRAME" else events[i]).append(e)
                if rst:
                    s.gru_h = None
        return frames, events

    def _stream(self, S):
        asr = self.asr
        pieces = [[] for _ in S]
        while True:
            ready = [i for i, s in enumerate(S) if s.ready()]
            if not ready:
                return pieces
            groups = {f: [i for i in ready if (S[i].chunks == 0) == f] for f in (True, False)}
            for first in (True, False):
                grp = groups[first]
                if not grp:
                    continue
                xs = []
                for i in grp:
                    s = S[i]
                    if first:
                        xs.append(s.buf[:s.need() - s.buf0])
                    else:
                        lo = s.mel_done * HOP_LEN - N_FFT // 2 - s.buf0
                        xs.append(s.buf[lo: lo + MEL_PER * HOP_LEN + WIN])
                x = torch.from_numpy(np.stack(xs)).to(self.dev)
                mel = asr.mel(x, center=first)[:, :FIRST_MEL if first else MEL_PER]
                slots = [S[i].slot for i in grp]
                last = asr.stream_step(self.pool, slots, mel, first=first)
                enc = asr.project(last, [S[i].prompt for i in grp])
                sl = torch.tensor(slots, device=self.dev)
                greedy = asr.greedy_fast if self.greedy_graphs else asr.greedy
                toks, d, h, c = greedy(enc, self.pred["d"][sl], self.pred["h"][:, sl], self.pred["c"][:, sl])
                self.pred["d"][sl], self.pred["h"][:, sl], self.pred["c"][:, sl] = d, h, c
                for j, i in enumerate(grp):
                    s = S[i]
                    for t, tok in toks[j]:
                        pieces[i].append({"f": s.frames_in + t, "p": asr.tokenizer.id_to_token(tok), "id": tok})
                    s.frames_in += last.shape[1]
                    s.mel_done += FIRST_MEL if first else MEL_PER
                    s.chunks += 1
                    keep = s.mel_done * HOP_LEN - N_FFT // 2
                    if keep > s.buf0:
                        s.buf = s.buf[keep - s.buf0:]
                        s.buf0 = keep

    @torch.no_grad()
    def transcribe(self, wav: np.ndarray, lang: str, lookahead: int = 3) -> str:
        """Offline transcript of a clip (barge-in history = heard text, lab checks)."""
        from .stream import is_tag
        asr = self.asr
        x = torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32))[None].to(self.dev)
        enc = asr.project(asr.encode(asr.mel(x, center=True), lookahead=lookahead), [asr.prompt_id(lang)])
        d, h, c = asr.pred_start(1)
        toks, *_ = asr.greedy(enc, d, h, c)
        pieces = asr.pieces([t for _, t in toks[0]])
        return "".join(p for p in pieces if not is_tag(p)).replace("▁", " ").strip()
