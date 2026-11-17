"""G0: the multilingual streaming ASR live in the frame loop (GPU), replacing the frozen track of E1.

Same model and decoding as e1/asr_track.py (nvidia/nemotron-3.5-asr-streaming-0.6b, cache-aware FastConformer with
chunked-limited attention [56, 3] and causal convolutions, greedy RNNT with the session-language prompt, predictor
reset after a language tag), but computed DURING the session from the audio received so far:

  * the encoder chunk is att_right + 1 = 4 frames (320 ms). When the audio of a whole chunk is in, the encoder runs
    over a sliding window that ends at that chunk (window start aligned to a chunk boundary, ``window_s`` of left
    audio as context) and the chunk's 4 new frames are decoded. Their pieces become available at the CURRENT model
    frame: causal by construction (no audio after the current frame is ever read).
  * the window replaces the encoder's streaming cache (equivalent up to the truncated left context; NOT bit-exact
    with the offline track, compared in the G0 report).

``ASRPost`` turns the pieces into the events the controller driver reads (closed words, activity, end of utterance)
with the same rules as the frozen track builder; it is model-free and unit-tested.
"""

from __future__ import annotations

import time

from frankenstein_s2s.usertext import TurnTaker, UserTextScheduler

LANG_IDX = {"auto": 101, "en": 0, "it": 15, "fr": 8}
FRAME = 1280                     # samples per 80 ms encoder frame at 16 kHz


def is_tag(p: str) -> bool:
    s = p.strip()
    return s.startswith("<") and s.endswith(">")


class ASRPost:
    """Pieces at frame k -> closed words, ASR activity, EOU (asr_track.py rules, incremental)."""

    def __init__(self, close_after: int = 3, eou_frames: int = 10, bou_frames: int = 3, smooth: int = 3):
        self.words: list = []
        self.sched = UserTextScheduler(lambda w: [1000], close_after=close_after,
                                       on_close=lambda k, w, ids: self.words.append({"k": k, "word": w}))
        self.tt = TurnTaker(eou_frames=eou_frames, bou_frames=bou_frames)
        self.raw: set = set()
        self.smooth = smooth            # activity = a piece in the last `smooth` frames (>= the ASR stride, G3)

    def step(self, k: int, pieces: list) -> dict:
        n0 = len(self.words)
        real = [p for p in pieces if not is_tag(p)]
        if real:
            self.raw.add(k)
        self.sched.on_asr(k, pieces)
        self.sched.step(k)
        nb = any(j in self.raw for j in range(k - self.smooth + 1, k + 1))
        eou = False
        for ev in self.tt.step(k, nb):
            if ev == "eou":
                eou = True
                self.sched.flush(k)
        return {"k": k, "pieces": real, "words": [w["word"] for w in self.words[n0:]], "eou": eou, "active": nb}


class LiveASR:  # pragma: no cover - GPU
    def __init__(self, lab_multilingual: str, lang: str = "it", att_right: int = 3, window_s: float = 12.0,
                 device: str = "cuda", every: int = 0):
        import sys
        import torch
        import yaml
        from omegaconf import OmegaConf
        sys.path.insert(0, lab_multilingual)
        from encoders import ASR_DIRS, SpeechEncoder
        from nemo.collections.asr.modules.rnnt import RNNTDecoder, RNNTJoint
        self.torch = torch
        d = ASR_DIRS["ml"]
        sd = torch.load(d / "model_weights.ckpt", map_location="cpu", weights_only=True)
        cfg = yaml.safe_load(open(d / "model_config.yaml"))
        self.dec = RNNTDecoder.from_config_dict(OmegaConf.create(cfg["decoder"]))
        self.jnt = RNNTJoint.from_config_dict(OmegaConf.create(cfg["joint"]))
        self.dec.load_state_dict({k[8:]: v for k, v in sd.items() if k.startswith("decoder.")})
        self.jnt.load_state_dict({k[6:]: v for k, v in sd.items() if k.startswith("joint.")})
        nd = cfg["model_defaults"]["enc_hidden"]
        self.pk = torch.nn.Sequential(torch.nn.Linear(nd + 128, 2 * nd), torch.nn.ReLU(), torch.nn.Linear(2 * nd, nd))
        self.pk.load_state_dict({k[len("prompt_kernel."):]: v for k, v in sd.items() if k.startswith("prompt_kernel.")})
        for m in (self.dec, self.jnt, self.pk):
            m.to(device).eval()
        self.enc = SpeechEncoder("ml", att=[56, att_right], device=device)
        self.vocab = list(self.jnt.vocabulary)
        self.blank = len(self.vocab)
        self.dev = device
        self.chunk = att_right + 1
        self.win = int(window_s / 0.08) // self.chunk * self.chunk
        self.every = max(self.chunk, every // self.chunk * self.chunk) if every else self.chunk   # G2: run every N frames
        self.lang_idx = LANG_IDX[lang]
        self.done_frames = 0                       # encoder frames decoded so far (multiple of chunk)
        self.g = self.st = None
        self._fresh()
        # pieces arrive in 4-frame batches: a word may only close after a gap longer than one chunk (G0: with
        # close_after 3 words were split at chunk boundaries, "diva no", "ave te")
        # G3: with a stride of 8 frames the 3-frame activity window had 5-frame holes inside speech: a false end of
        # utterance opened a turn mid-sentence (frame 139 of U1) and the caller's next words cancelled it
        self.post = ASRPost(close_after=self.every + 1, smooth=max(3, self.every))
        self.stats = {"calls": 0, "ms": [], "frames": 0, "resets": 0}
        self.per_frame: dict = {}                  # model frame k -> pieces (incl. tags), for the record

    def _fresh(self):
        torch = self.torch
        self.g, self.st = self.dec.predict(torch.tensor([[self.blank]], device=self.dev), None, add_sos=False,
                                           batch_size=1)

    def _decode(self, e):
        torch = self.torch
        oh = torch.zeros(e.shape[0], 128, device=self.dev)
        oh[:, self.lang_idx] = 1
        f = self.pk(torch.cat([e, oh], -1))
        out = []
        for t in range(f.shape[0]):
            for _ in range(10):
                k = int(self.jnt.joint(f[t:t + 1][None], self.g)[0, 0, 0].argmax())
                if k == self.blank:
                    break
                out.append(self.vocab[k])
                self.g, self.st = self.dec.predict(torch.tensor([[k]], device=self.dev), self.st, add_sos=False,
                                                   batch_size=1)
                if is_tag(self.vocab[k]):
                    self._fresh()
                    self.stats["resets"] += 1
                    break
        return out

    def step(self, k: int, wav) -> dict:
        """Model frame k: audio wav[: (k + 1) * FRAME] is available. Decodes every newly complete chunk."""
        torch = self.torch
        pieces = []
        have = (k + 1) * FRAME // FRAME                             # complete encoder frames of audio
        ready = have // self.chunk * self.chunk
        if ready - self.done_frames >= self.every:
            t0 = time.monotonic_ns()
            with torch.inference_mode():
                start = max(0, ready - self.win)
                e = self.enc(wav[start * FRAME: ready * FRAME])["out"]          # [F, 1024]
                lo = self.done_frames - start
                pieces = self._decode(e[lo: ready - start])
            torch.cuda.current_stream().synchronize()
            self.stats["calls"] += 1
            self.stats["frames"] += ready - self.done_frames
            self.stats["ms"].append(round((time.monotonic_ns() - t0) / 1e6, 2))
            self.done_frames = ready
        if pieces:
            self.per_frame[k] = pieces
        return self.post.step(k, pieces)

    def reset(self, window_s: float | None = None, every: int | None = None) -> None:
        """A fresh stream (G2: several configurations on one loaded model)."""
        if window_s is not None:
            self.win = int(window_s / 0.08) // self.chunk * self.chunk
        if every is not None:
            self.every = max(self.chunk, every // self.chunk * self.chunk)
        self.done_frames = 0
        self._fresh()
        self.post = ASRPost(close_after=self.every + 1, smooth=max(3, self.every))
        self.stats = {"calls": 0, "ms": [], "frames": 0, "resets": 0}
        self.per_frame = {}

    def transcribe(self, wav16, chunk_s: float = 30.0) -> str:
        """Offline transcript of a whole 16 kHz signal (content evidence for the talker's audio), in 30 s pieces."""
        torch = self.torch
        self._fresh()
        out = []
        n = int(chunk_s * 16000)
        with torch.inference_mode():
            for i in range(0, len(wav16), n):
                piece = wav16[i:i + n]
                if len(piece) < 16000:
                    break
                out += self._decode(self.enc(piece)["out"])
        self._fresh()
        return "".join(p for p in out if not is_tag(p)).replace("\u2581", " ").strip()
