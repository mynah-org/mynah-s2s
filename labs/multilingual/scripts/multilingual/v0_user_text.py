"""V0 (owner GO 2026-11-13, CPU): RNNT -> Nano token stream for the duplex user-text channel, with emission frames.

Production-like RNNT: the ML ASR model's own decoder / joint / prompt kernel on the ML encoder with its ORIGINAL LNs,
att [56, 0], language prompt 'auto' (as perception_swap.set_rnnt_ml default), greedy, <= 10 symbols per frame, on the
bridge session (cs = 100 frames of silence, the clip, 5 s tail). Every emitted RNNT piece keeps its frame.
Injection policy (pure functions below, CPU-tested):
  words    RNNT pieces are grouped into words ('▁' starts a word); a word is complete when the next word starts
           (its frame) or at the end of the stream (last emission frame + 1). Language tags <xx-XX> are dropped.
  tokens   each complete word is re-tokenised with the Nano tokenizer as ' word' (leading space) and appended to a FIFO
           queue at its completion frame; the queue releases ONE Nano token per frame, never earlier than completion.
  filter   ids < FIRST_TEXT_ID (control / special range: BOS 1, EOS 2, PAD 12, SOTC 20, EOTC 21, EOTR 22, <SPECIAL_n>)
           are never injected.
  masking  frames with no released token carry NO user-text contribution (zero vector, not PAD), so the opener region
           and silences are untouched; FC fast-path steps never consume a user-text token (greedy_fc reads the stream
           by audio frame index only in main mode).
Output: {"stream": {"set|id|lang": [[frame, nano_id], ...]}, "words": ..., "latency": ...}.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

TAG = re.compile(r"<[a-z]{2}(-[A-Z]{2})?>")
FIRST_TEXT_ID = 1000


def word_events(pieces: list[tuple[int, str]], hold: int = 3) -> list[tuple[int, str]]:
    """[(frame, rnnt piece)] -> [(completion frame, word)]. A word is complete at the first piece of the next word, or
    `hold` frames after its own last piece, whichever comes first ('▁' or a bare space piece starts a word)."""
    words, cur, first, last, boundary = [], "", None, None, False
    for fr, p in pieces:
        p = TAG.sub("", p)
        if p.startswith("▁"):
            boundary = True                       # a bare space piece is a boundary too (V0 merge bug, fixed)
        body = p.replace("▁", "")
        if not any(ch.isalnum() for ch in body):  # punctuation-only pieces (often emitted late) neither form nor extend words
            continue
        if boundary and cur:
            words.append([first, last, cur])
            cur = ""
        if not cur:
            first = fr
        boundary = False
        cur += body
        last = fr
    if cur:
        words.append([first, last, cur])
    out = []
    for k, (f0, f1, w) in enumerate(words):
        nxt = words[k + 1][0] if k + 1 < len(words) else None
        out.append((min(f1 + hold, nxt) if nxt is not None else f1 + hold, w))
    return out


def frame_tokens(events: list[tuple[int, str]], encode) -> list[tuple[int, int]]:
    """FIFO release, one token per frame, never before the word's completion frame; special ids filtered."""
    out, t = [], -1
    for fr, w in events:
        for tid in encode(" " + w):
            if tid < FIRST_TEXT_ID:
                continue
            t = max(t + 1, fr)
            out.append((t, tid))
    return out


def main():  # pragma: no cover - CPU model run
    import soundfile as sf
    import torch
    import yaml
    from omegaconf import OmegaConf
    from nemo.collections.asr.modules.rnnt import RNNTDecoder, RNNTJoint
    from transformers import AutoTokenizer
    from encoders import ASR_DIRS, SpeechEncoder
    from session_distill import session_wav
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", required=True)
    ap.add_argument("--keep", default="glex_dev")
    ap.add_argument("--glex-data", required=True)
    ap.add_argument("--val-data", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--cs", type=int, default=100)
    ap.add_argument("--tail-s", type=float, default=5.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    assert not torch.cuda.is_available(), "V0 is CPU-only"
    torch.set_num_threads(16)
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    enc = lambda s: tok.encode(s, add_special_tokens=False)  # noqa: E731
    rows = [r for r in json.load(open(a.sets))["rows"] if r["set"] in a.keep.split(",")]
    d = ASR_DIRS["ml"]
    sd = torch.load(d / "model_weights.ckpt", map_location="cpu", weights_only=True)
    cfg = yaml.safe_load(open(d / "model_config.yaml"))
    dec = RNNTDecoder.from_config_dict(OmegaConf.create(cfg["decoder"]))
    jnt = RNNTJoint.from_config_dict(OmegaConf.create(cfg["joint"]))
    dec.load_state_dict({k[8:]: v for k, v in sd.items() if k.startswith("decoder.")})
    jnt.load_state_dict({k[6:]: v for k, v in sd.items() if k.startswith("joint.")})
    nd = cfg["model_defaults"]["enc_hidden"]
    pk = torch.nn.Sequential(torch.nn.Linear(nd + 128, 2 * nd), torch.nn.ReLU(), torch.nn.Linear(2 * nd, nd))
    pk.load_state_dict({k[len("prompt_kernel."):]: v for k, v in sd.items() if k.startswith("prompt_kernel.")})
    dec.eval(), jnt.eval(), pk.eval()
    vocab, blank = list(jnt.vocabulary), len(jnt.vocabulary)
    lang_idx = int(cfg["model_defaults"]["prompt_dictionary"]["auto"])
    se = SpeechEncoder("ml", att=[56, 0], device="cpu")
    res, t0 = {"stream": {}, "words": {}, "latency": {}, "policy": __doc__}, time.monotonic()
    for r in rows:
        root = a.glex_data if r["data"] == "glex" else a.val_data
        x = sf.read(Path(root) / r["path"], dtype="float32")[0]
        with torch.no_grad():
            f = se(torch.from_numpy(session_wav(x, a.cs, a.tail_s)))["out"].float()
            oh = torch.zeros(f.shape[0], 128)
            oh[:, lang_idx] = 1
            ff = pk(torch.cat([f, oh], -1))
            g, st = dec.predict(torch.tensor([[blank]]), None, add_sos=False, batch_size=1)
            pieces = []
            for t in range(f.shape[0]):
                for _ in range(10):
                    k = int(jnt.joint(ff[t:t + 1][None], g)[0, 0, 0].argmax())
                    if k == blank:
                        break
                    pieces.append((t, vocab[k]))
                    g, st = dec.predict(torch.tensor([[k]]), st, add_sos=False, batch_size=1)
        ev = word_events(pieces)
        key = f"{r['set']}|{r['id']}|{r['lang']}"
        res["words"][key] = ev
        res.setdefault("pieces", {})[key] = pieces
        res["stream"][key] = frame_tokens(ev, enc)
        x_end = a.cs + int(len(x) / 1280 + 0.999)
        res["latency"][key] = {"x_end": x_end, "last_word_frame": ev[-1][0] if ev else None,
                               "last_token_frame": res["stream"][key][-1][0] if res["stream"][key] else None}
        print(key, round(time.monotonic() - t0, 1), "|", " ".join(w for _, w in ev)[:80], flush=True)
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False))
    print("done", round(time.monotonic() - t0, 1))


if __name__ == "__main__":
    main()
