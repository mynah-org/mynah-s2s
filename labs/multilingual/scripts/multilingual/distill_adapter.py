"""A3 — cross-lingual distillation of an interface adapter THROUGH the frozen VoiceChat Thinker.

Idea: VoiceChat already understands English speech. For a parallel utterance (same sentence in EN and in X),
the teacher is VoiceChat (VC encoder + VC proj + LLM) hearing the EN clip; the student is the ML encoder + adapter +
VC proj + the *same frozen* LLM hearing the X clip (X in {en, it, fr}; DE/ES are never trained → transfer controls).
The student is trained to emit the teacher's agent-text tokens, re-timed to start at the same offset after the end of
its own speech. Only the adapter (≈ 5 M params) is trained; encoder, proj, fusion, LLM and heads are frozen.

  prep  : teacher decode (batched, hybrid LLM cache, greedy) on EN clips with offline VC features + ML features of
          every student clip (speech + trailing silence). The teacher loop is the per-frame duplex input of the NVIDIA
          wrapper (infer_one_step: fusion(agent emb of previous token, user-audio emb, function emb)), system prompt
          prefilled with the wrapper's own `_prepare_system_prompt_embeddings`; it has NO RNNT/forced turn-taking —
          those only matter at evaluation, which goes through the real pipeline (e2e_qa.py).
  train : teacher-forced CE on the student's text logits; wall-clock cap; best checkpoint on held-out items.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

FRAME = 1280  # samples per 80 ms frame


def load_engine(out):
    from mynah_v2.model.llm_cache import enable_hybrid_llm_cache
    from mynah_v2.model.reference import ReferenceEngine, build_cfg
    eng = ReferenceEngine(build_cfg(output_dir=str(Path(out) / "_nemo"),
                                    overrides={"s2s.deterministic": False, "s2s.decode_audio": False}))
    enable_hybrid_llm_cache(eng)
    return eng


class Duplex:
    """Thin, differentiable view of the frozen VoiceChat Thinker as the NVIDIA wrapper drives it."""

    def __init__(self, eng, prompt):
        self.w = eng.wrapper
        self.stt = self.w.model.stt_model
        for p in self.stt.parameters():
            p.requires_grad_(False)
        self.fusion = self.w.fusion_module
        for p in self.fusion.parameters():
            p.requires_grad_(False)
        self.dtype = next(self.stt.llm.parameters()).dtype
        self.dev = next(self.stt.llm.parameters()).device
        self.pad = int(self.stt.text_pad_id)
        with torch.no_grad():
            pe, _ = self.w._prepare_system_prompt_embeddings(prompt)
        self.prompt_emb = pe.to(self.dtype)  # [1, P, H] (already fused)
        self.proj = self.stt.perception.proj  # VC proj (frozen)

    def frames(self, audio_emb, prev_tokens):
        """audio_emb [B, T, H] (already through proj), prev_tokens [B, T] (agent token fed at each frame, pad at t=0)."""
        e = self.stt.embed_tokens
        agent = e(prev_tokens).to(self.dtype)
        func = e(torch.full_like(prev_tokens, self.pad)).to(self.dtype)
        return self.fusion(agent_text_embeds=agent, user_audio_embeds=audio_emb.to(self.dtype),
                           user_text_embeds=None, function_embeds=func)

    def logits(self, frame_embeds):
        B = frame_embeds.shape[0]
        x = torch.cat([self.prompt_emb.expand(B, -1, -1), frame_embeds], 1)
        out = self.stt(x, cache=None)
        return out["text_logits"][:, self.prompt_emb.shape[1]:]

    @torch.no_grad()
    def greedy(self, audio_emb):
        """Batched incremental decode (hybrid cache), returns agent text tokens [B, T]."""
        from mynah_v2.model.llm_cache import _hybrid_cache_cls
        B, T, _ = audio_emb.shape
        cache = _hybrid_cache_cls(self.stt.llm)(self.stt.llm.config, B, dtype=self.dtype, device=self.dev,
                                                ssm_dtype=torch.float32)
        self.stt(self.prompt_emb.expand(B, -1, -1).contiguous(), cache=cache)
        toks = torch.full((B, T), self.pad, dtype=torch.long, device=self.dev)
        prev = torch.full((B,), self.pad, dtype=torch.long, device=self.dev)
        for t in range(T):
            x = self.frames(audio_emb[:, t:t + 1], prev[:, None])
            ans = self.stt(x, cache=cache)
            cache = ans.get("cache", cache)
            prev = ans["text_logits"][:, -1].float().argmax(-1)
            toks[:, t] = prev
        return toks


def pad_to(x, n):
    return np.pad(x, (0, max(0, n - len(x))))


def cmd_prep(a):
    from encoders import SpeechEncoder
    root = Path(a.data)
    rows = [json.loads(l) for l in open(root / "manifest.jsonl")]
    by = defaultdict(dict)
    for r in rows:
        by[r["id"]][r["lang"]] = r
    ids = sorted(i for i in by if "en" in by[i])
    if a.limit:
        ids = ids[: a.limit]
    eng = load_engine(a.out)
    D = Duplex(eng, a.prompt)
    vc = SpeechEncoder("vc")
    ml = SpeechEncoder("ml")
    tail = int(a.tail_s * 16000)
    out = {"prompt": a.prompt, "tail_s": a.tail_s, "items": {}}
    feats = {}
    t0 = time.monotonic()
    for b in range(0, len(ids), a.batch):
        chunk = ids[b:b + a.batch]
        wavs = [sf.read(root / by[i]["en"]["path"], dtype="float32")[0] for i in chunk]
        L = int(math.ceil((max(len(w) for w in wavs) + tail) / FRAME) * FRAME)
        embs, ends = [], []
        for w in wavs:
            f = vc(torch.from_numpy(pad_to(w, L)))["out"]
            embs.append(f)
            ends.append(int(math.ceil(len(w) / FRAME)))
        T = min(e.shape[0] for e in embs)
        with torch.no_grad():
            A = D.proj(torch.stack([e[:T] for e in embs]).to(D.dev, D.proj.weight.dtype))
        toks = D.greedy(A).cpu()
        for k, i in enumerate(chunk):
            tt = toks[k].tolist()
            out["items"][i] = {"teacher": tt, "en_end": ends[k], "T": T,
                               "teacher_text": D.w.tokenizer.ids_to_text([x for x in tt if x != D.pad])}
        print(f"teacher {b + len(chunk)}/{len(ids)} T={T} {time.monotonic() - t0:.0f}s | "
              f"{out['items'][chunk[0]]['teacher_text'][:90]!r}", flush=True)
    for i in ids:
        for lang, r in by[i].items():
            w = sf.read(root / r["path"], dtype="float32")[0]
            L = int(math.ceil((len(w) + tail + 2 * FRAME) / FRAME) * FRAME)
            f = ml(torch.from_numpy(pad_to(w, L)))["out"].half().cpu()
            feats[f"{i}|{lang}"] = f
            out["items"][i].setdefault("end", {})[lang] = int(math.ceil(len(w) / FRAME))
    torch.save(feats, Path(a.out) / "ml_feats.pt")
    (Path(a.out) / "teacher.json").write_text(json.dumps(out, ensure_ascii=False))
    print("prep done", len(ids), f"{time.monotonic() - t0:.0f}s")


def build_batch(D, items, feats, keys, adapter, live=None):
    """keys: list of (item_id, lang). Returns frame embeds, labels [B,T], loss weights [B,T].
    live: optional LiveML (E5) -> features recomputed by the (partly trainable) ML encoder instead of the cache."""
    seqs = []
    fl = live.feats(keys) if live is not None else None
    for k0, (i, lang) in enumerate(keys):
        it = items[i]
        tt, e_en, T = it["teacher"], it["en_end"], it["T"]
        e_x = it["end"][lang]
        n = e_x + (T - e_en)
        f = feats[f"{i}|{lang}"] if fl is None else fl[k0]
        n = min(n, f.shape[0])
        lab = [D.pad] * e_x + tt[e_en:e_en + (n - e_x)]
        lab = lab[:n]
        w = [0.3] * e_x + [5.0 if x != D.pad else 1.0 for x in lab[e_x:]]
        seqs.append((f[:n], lab, w))
    Tm = max(len(s[1]) for s in seqs)
    B = len(seqs)
    X = torch.zeros(B, Tm, 1024)
    Y = torch.full((B, Tm), D.pad, dtype=torch.long)
    W = torch.zeros(B, Tm)
    if fl is not None:  # differentiable path: no in-place copy into a leaf buffer
        X = torch.stack([torch.nn.functional.pad(f[:len(lab)].float(), (0, 0, 0, Tm - len(lab)))
                         for f, lab, _ in seqs]).to(D.dev)
    for k, (f, lab, w) in enumerate(seqs):
        if fl is None:
            X[k, :len(lab)] = f[:len(lab)].float()
        Y[k, :len(lab)] = torch.tensor(lab)
        W[k, :len(lab)] = torch.tensor(w)
    X, Y, W = X.to(D.dev), Y.to(D.dev), W.to(D.dev)
    prev = torch.cat([torch.full((B, 1), D.pad, device=D.dev, dtype=torch.long), Y[:, :-1]], 1)
    audio = D.proj(adapter(X).to(D.proj.weight.dtype))
    return D.frames(audio, prev), Y, W


class LoRALinear(torch.nn.Module):
    """y = base(x) + (x A^T B^T) * alpha/r, base frozen, B zero-init (starts as the identity of base)."""

    def __init__(self, base: torch.nn.Linear, r: int, alpha: float):
        super().__init__()
        self.base = base
        self.A = torch.nn.Parameter(torch.empty(r, base.in_features, device=base.weight.device))
        self.B = torch.nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device))
        torch.nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + (x @ self.A.t() @ self.B.t()) * self.scale

    def merged_weight(self):
        return (self.base.weight + (self.B @ self.A) * self.scale).detach()


class LiveML:
    """E5: the ML encoder inside the training loop, only its LayerNorm affine params trainable (fp32, eval mode so
    dropout/BatchNorm statistics are unchanged). Input padding reproduces prep exactly (speech + tail + 2 frames)."""

    def __init__(self, data_root, tail_s, dev, train_ln=True, lora_top=0, lora_rank=8):
        from encoders import SpeechEncoder
        self.root = Path(data_root)
        rows = [json.loads(l) for l in open(self.root / "manifest.jsonl")]
        self.path = {(r["id"], r["lang"]): self.root / r["path"] for r in rows}
        self.tail = int(tail_s * 16000)
        self.se = SpeechEncoder("ml")
        self.se.eval()
        for p in self.se.parameters():
            p.requires_grad_(False)
        self.ln = {n: p for n, p in self.se.enc.named_parameters()
                   if isinstance(self.se.enc.get_submodule(n.rsplit(".", 1)[0]), torch.nn.LayerNorm)} if train_ln else {}
        for p in self.ln.values():
            p.requires_grad_(True)
        self.ln0 = {n: p.detach().clone() for n, p in self.ln.items()}
        self.lora = {}
        if lora_top:
            L = len(self.se.enc.layers)
            for li in range(L - lora_top, L):
                layer = self.se.enc.layers[li]
                for name, mod in list(layer.named_modules()):
                    if isinstance(mod, torch.nn.Linear) and any(k in name for k in ("self_attn.linear_", "feed_forward")):
                        parent = layer.get_submodule(name.rsplit(".", 1)[0]) if "." in name else layer
                        w = LoRALinear(mod, lora_rank, 2.0 * lora_rank)
                        setattr(parent, name.rsplit(".", 1)[-1], w)
                        self.lora[f"layers.{li}.{name}.weight"] = w
        self.params = list(self.ln.values()) + [p for w in self.lora.values() for p in (w.A, w.B)]
        self.wav = {}

    def _wav(self, key):
        if key not in self.wav:
            w = sf.read(self.path[key], dtype="float32")[0]
            L = int(math.ceil((len(w) + self.tail + 2 * FRAME) / FRAME) * FRAME)
            self.wav[key] = torch.from_numpy(pad_to(w, L))
        return self.wav[key]

    def feats(self, keys):
        ws = [self._wav(k) for k in keys]
        n = torch.tensor([len(w) for w in ws], device=self.se.device)
        x = torch.stack([torch.nn.functional.pad(w, (0, int(n.max()) - len(w))) for w in ws]).to(self.se.device)
        mel, ml = self.se.pre(input_signal=x, length=n)
        e, el = self.se.enc(audio_signal=mel, length=ml)
        return [e[b, :, : int(el[b])].transpose(0, 1) for b in range(len(keys))]

    def ln_state(self):
        """Encoder delta for evaluation (perception_swap.set_dual_ln loads it by original key names): trained
        LayerNorm params + LoRA-merged Linear weights."""
        d = {n: p.detach().cpu().clone() for n, p in self.ln.items()}
        d.update({k: w.merged_weight().cpu().clone() for k, w in self.lora.items()})
        return d


def cmd_train(a):
    from adapters import ResidualAdapter, save_adapter
    out = Path(a.out)
    prep = json.loads((Path(a.prep) / "teacher.json").read_text())
    feats = torch.load(Path(a.prep) / "ml_feats.pt")
    items = prep["items"]
    if a.keep_ids:  # teacher filter (corpus_v2.py teacher_filter)
        keep = set(json.load(open(a.keep_ids))["keep"])
        items = {i: v for i, v in items.items() if i in keep}
    ids = sorted(items)
    man = [json.loads(l) for l in open(Path(a.data) / "manifest.jsonl")] if a.data else []
    excl = {(r["id"], r["lang"]) for r in man if r.get("teacher_only")}
    mval = sorted({r["id"] for r in man if r.get("split") == "val"} & set(ids))
    if mval:  # family-level validation split defined by the corpus
        rv = random.Random(0)
        rv.shuffle(mval)
        val_ids = sorted(mval[: a.max_val_items]) if a.max_val_items else sorted(mval)
        tr_ids = [i for i in ids if i not in set(mval)]
    else:
        random.Random(0).shuffle(ids)
        nval = max(8, len(ids) // 10)
        val_ids, tr_ids = ids[:nval], ids[nval:]
    langs = a.langs.split(",")
    eng = load_engine(out)
    D = Duplex(eng, prep["prompt"])
    # The engine load re-seeds torch's default generator randomly (measured: a different initial_seed per process),
    # so a1's init is not reproducible. From now on the seed is fixed BEFORE the adapter is built.
    torch.manual_seed(a.seed)
    ad = ResidualAdapter(temporal_k=a.temporal_k).to(D.dev)
    if a.init_map:
        m = torch.load(a.init_map, map_location="cpu", weights_only=True)
        ad.init_affine(m["M"].to(D.dev), m["m"].to(D.dev))
    import hashlib
    init_hash = hashlib.sha256(b"".join(v.detach().float().cpu().numpy().tobytes()
                                        for _, v in sorted(ad.state_dict().items()))).hexdigest()[:16]
    print(f"adapter init sha256[:16] {init_hash} (torch seed {torch.initial_seed()})", flush=True)
    if a.init_hash_only:
        (out / "init_hash.json").write_text(json.dumps({"init_hash": init_hash, "seed": torch.initial_seed()}))
        return
    params = [{"params": ad.affine.parameters(), "lr": a.lr * a.affine_lr_mult},
              {"params": [p for n, p in ad.named_parameters() if not n.startswith("affine.")], "lr": a.lr}]
    live = None
    if a.train_ln or a.lora_top:
        live = LiveML(a.data, prep["tail_s"], D.dev, train_ln=a.train_ln, lora_top=a.lora_top, lora_rank=a.lora_rank)
        if live.ln:
            params.append({"params": list(live.ln.values()), "lr": a.ln_lr})
        if live.lora:
            params.append({"params": [p for w in live.lora.values() for p in (w.A, w.B)], "lr": a.lora_lr})
        print(f"encoder trainables: {len(live.ln)} LayerNorm tensors ({sum(p.numel() for p in live.ln.values())}), "
              f"{len(live.lora)} LoRA linears ({sum(w.A.numel() + w.B.numel() for w in live.lora.values())})",
              flush=True)
    trainable = [p for g in params for p in g["params"]]
    opt = torch.optim.AdamW(params, weight_decay=0.0)
    tr_keys = [(i, l) for i in tr_ids for l in langs if l in items[i]["end"] and (i, l) not in excl]
    va_keys = [(i, l) for i in val_ids for l in langs if l in items[i]["end"] and (i, l) not in excl]
    print(f"train keys {len(tr_keys)} {dict(Counter(k[1] for k in tr_keys))} | val keys {len(va_keys)} "
          f"{dict(Counter(k[1] for k in va_keys))}", flush=True)

    def loss_on(keys, per_sample=False):
        fe, Y, W = build_batch(D, items, feats, keys, ad, live)
        lg = D.logits(fe).float()
        ce = torch.nn.functional.cross_entropy(lg.reshape(-1, lg.shape[-1]), Y.reshape(-1), reduction="none")
        hit = (lg.argmax(-1) == Y).float() * (W >= 5)
        if per_sample:  # [B] weighted CE and token accuracy per key (same normalisation as the batch loss)
            cs = (ce.view_as(W) * W).sum(1) / W.sum(1).clamp_min(1e-6)
            return cs, hit.sum(1) / (W >= 5).sum(1).clamp_min(1)
        acc = hit.sum() / (W >= 5).sum().clamp_min(1)
        return (ce * W.reshape(-1)).sum() / W.sum(), acc

    @torch.no_grad()
    def validate():
        ad.eval()
        tot = defaultdict(list)
        for k in range(0, len(va_keys), a.batch):
            ks = va_keys[k:k + a.batch]
            ls, accs = loss_on(ks, per_sample=True)
            for key, l, acc in zip(ks, ls.tolist(), accs.tolist()):
                tot[key[1]].append((l, acc))
        ad.train()
        return {l: [round(float(np.mean([x[0] for x in v])), 4), round(float(np.mean([x[1] for x in v])), 4)] for l, v in tot.items()}

    if a.smoke_steps:
        return smoke(a, out, D, items, feats, tr_keys, ad, live, opt, loss_on, trainable)
    log = []
    v0 = validate()
    print("val@0", v0, flush=True)
    log.append({"step": 0, "val": v0})
    best = sum(v[0] for v in v0.values())
    save_adapter(ad, out / "adapter_best.pt", {"step": 0, "val": v0})
    if live is not None:
        torch.save(live.ln_state(), out / "encoder_ln_best.pt")
    t0 = time.monotonic()
    step = 0
    rng = random.Random(1)
    since_best = 0
    while time.monotonic() - t0 < a.max_minutes * 60 and step < a.max_steps:
        ks = rng.sample(tr_keys, a.batch)
        l, acc = loss_on(ks)
        opt.zero_grad()
        l.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        step += 1
        if step % 10 == 0:
            print(f"step {step} loss {l.item():.4f} tokacc {acc.item():.3f} {time.monotonic() - t0:.0f}s", flush=True)
        if step % a.eval_every == 0:
            v = validate()
            s = sum(x[0] for x in v.values())
            log.append({"step": step, "val": v, "t": round(time.monotonic() - t0)})
            print("val", step, v, flush=True)
            if s < best:
                best, since_best = s, 0
                save_adapter(ad, out / "adapter_best.pt", {"step": step, "val": v})
                if live is not None:
                    torch.save(live.ln_state(), out / "encoder_ln_best.pt")
            else:
                since_best += 1
            if a.save_every and step % a.save_every == 0:
                save_adapter(ad, out / f"adapter_step{step}.pt", {"step": step, "val": v})
                if live is not None:
                    torch.save(live.ln_state(), out / f"encoder_ln_step{step}.pt")
            if a.patience and since_best >= a.patience:
                print(f"early stop at step {step}: no val improvement for {a.patience} evaluations", flush=True)
                log.append({"step": step, "early_stop": True})
                break
    v = validate()
    log.append({"step": step, "val": v, "t": round(time.monotonic() - t0), "final": True})
    save_adapter(ad, out / "adapter_last.pt", {"step": step, "val": v})
    if live is not None:
        torch.save(live.ln_state(), out / "encoder_ln_last.pt")
    (out / "train_log.json").write_text(json.dumps({"args": vars(a), "init_hash": init_hash, "n_train_items": len(tr_ids),
                                                    "n_train_keys": len(tr_keys), "n_val_keys": len(va_keys),
                                                    "train_s": round(time.monotonic() - t0), "val_ids": val_ids,
                                                    "log": log}, indent=1))
    print("done", step, "steps", v)


def smoke(a, out, D, items, feats, tr_keys, ad, live, opt, loss_on, trainable):
    """GO checks for E5 (no checkpoint written): (1) live features == cached prep features at step 0;
    (2) non-zero LayerNorm gradients, features carry grad; (3) encoder output moves after N steps;
    (4) step time, peak CUDA memory, minimum MemAvailable."""
    def avail():
        return next(int(l.split()[1]) // 1024 for l in open("/proc/meminfo") if l.startswith("MemAvailable"))
    rep = {"steps": a.smoke_steps, "batch": a.batch}
    probe = tr_keys[:4]
    with torch.no_grad():
        f0 = live.feats(probe)
        rep["parity_vs_cached_max_abs"] = max(float((f0[k][: feats[f"{i}|{l}"].shape[0]].float().cpu()
                                                     - feats[f"{i}|{l}"].float()).abs().max())
                                              for k, (i, l) in enumerate(probe))
        rep["feat_abs_mean"] = float(torch.cat([x.abs().flatten() for x in f0]).mean())
    torch.cuda.reset_peak_memory_stats()
    rng, times, mins = random.Random(1), [], [avail()]
    for st in range(a.smoke_steps):
        ks = rng.sample(tr_keys, a.batch)
        torch.cuda.synchronize()
        t = time.monotonic()
        l, acc = loss_on(ks)
        opt.zero_grad()
        l.backward()
        if st == 0:
            g = [p.grad for p in live.params]
            rep["ln_grad_none"] = sum(x is None for x in g)
            rep["ln_grad_zero"] = sum(x is not None and float(x.abs().max()) == 0 for x in g)
            rep["ln_grad_norm"] = float(torch.sqrt(sum((x.float() ** 2).sum() for x in g if x is not None)))
            rep["adapter_grad_norm"] = float(torch.sqrt(sum((p.grad.float() ** 2).sum() for p in ad.parameters()
                                                            if p.grad is not None)))
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        torch.cuda.synchronize()
        times.append(time.monotonic() - t)
        mins.append(avail())
        print(f"smoke step {st} loss {l.item():.4f} {times[-1]:.2f}s", flush=True)
    with torch.no_grad():
        f1 = live.feats(probe)
        rep["feat_rel_change"] = float(torch.cat([(x1 - x0).flatten() for x0, x1 in zip(f0, f1)]).norm()
                                       / torch.cat([x.flatten() for x in f0]).norm())
        if live.ln:
            rep["ln_rel_change"] = float(torch.sqrt(sum(((p.detach() - live.ln0[n]) ** 2).sum() for n, p in live.ln.items()))
                                         / torch.sqrt(sum((v ** 2).sum() for v in live.ln0.values())))
    rep["step_s_median"] = float(np.median(times[1:] if len(times) > 1 else times))
    rep["peak_cuda_gib"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
    rep["min_memavailable_mib"] = min(mins)
    rep["go_checks"] = {"parity": rep["parity_vs_cached_max_abs"] < 2e-2,
                        "ln_grads": rep["ln_grad_none"] == 0 and rep["ln_grad_norm"] > 0,
                        "encoder_moves": rep["feat_rel_change"] > 0}
    (out / "smoke.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("prep")
    p.add_argument("--data", required=True, help="dir with manifest.jsonl from render_qa.py --train-jsonl")
    p.add_argument("--out", required=True)
    p.add_argument("--prompt", default="You are a helpful assistant.")
    p.add_argument("--tail-s", type=float, default=5.0)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--limit", type=int, default=0)
    t = sp.add_parser("train")
    t.add_argument("--prep", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--langs", default="en,it,fr")
    t.add_argument("--init-map", default=None)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--affine-lr-mult", type=float, default=0.3)
    t.add_argument("--batch", type=int, default=8)
    t.add_argument("--temporal-k", type=int, default=0)
    t.add_argument("--max-minutes", type=float, default=25)
    t.add_argument("--max-steps", type=int, default=100000)
    t.add_argument("--eval-every", type=int, default=50)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--init-hash-only", action="store_true", help="print/save the adapter init hash and exit")
    t.add_argument("--train-ln", action="store_true", help="E5: ML-encoder LayerNorms trainable, encoder in the loop")
    t.add_argument("--data", default=None, help="E5: training audio dir (same as prep --data)")
    t.add_argument("--ln-lr", type=float, default=5e-5)
    t.add_argument("--smoke-steps", type=int, default=0, help="E5 GO checks only (no checkpoints)")
    t.add_argument("--keep-ids", default=None, help="teacher_filter output; only these items are used")
    t.add_argument("--max-val-items", type=int, default=0, help="cap on manifest val families (0 = all)")
    t.add_argument("--patience", type=int, default=0, help="early stop after N evaluations without improvement")
    t.add_argument("--save-every", type=int, default=0, help="also keep a checkpoint every N steps")
    t.add_argument("--lora-top", type=int, default=0, help="LoRA on the top K encoder blocks (attention + FFN)")
    t.add_argument("--lora-rank", type=int, default=8)
    t.add_argument("--lora-lr", type=float, default=1e-4)
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    {"prep": cmd_prep, "train": cmd_train}[a.cmd](a)


if __name__ == "__main__":
    main()
