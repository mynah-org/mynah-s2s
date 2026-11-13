"""L1 smoke on the real model (GPU; called by `session_distill.py train --lora --smoke`). MEASURES ONLY: it never picks
or changes the learning rate (owner 2026-11-13); every LoRA change it makes is undone before it returns.
Train clips only (no held-out data is looked at, so the smoke is not an implicit tuning on held-out).

S1 audit       target modules, shapes, LoRA parameter count, nothing else trainable anywhere (Thinker, heads, fusion,
               adapter, encoder LNs)
S2 identity    B = 0 vs LoRA disabled: teacher-forced text / function logits bit-identical; greedy_fc (hybrid cache,
               CUDA Mamba kernels) identical tokens
S3 gradients   one backward of the training loss: every B gradient non-zero, A gradients 0 (B = 0), no gradient on
               any frozen tensor; the function loss ALONE also gives non-zero B gradients
S4 perturbed   B ~ N(0, sigma) temporarily: logit change, argmax change share, bf16 branch survival on the real q/v
               inputs; cache vs no-cache: greedy_fc top-k text log-probs against one full forward over the generated
               tokens, LoRA off and LoRA on (the LoRA must not add cache error)
S5 memory      peak CUDA allocated, MemAvailable
S6 calibration --smoke-steps optimiser steps per LR of --smoke-lrs (default --lr), same sampled batches, on train clips: per-step loss components, grad norms, update norms
               per LoRA tensor, branch survival, and on 8 fixed train call clips (teacher-forced): mean gold call
               log-prob, call-token argmax accuracy, mean slot margin, SOTC log-prob, BOS argmax frames while
               listening, before vs after. Then the LoRA and the optimiser state are reset.
Writes smoke_l1.json.
"""
from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from session_distill import BOS, PAD, SOTC, greedy_fc, student_targets


def capture_hook(store: dict, name: str):
    """Forward hook that records the module INPUT once and returns None (a non-None return would replace the output:
    job 352 bug, the q_proj output became its 4480-wide input)."""
    def hook(mod, inp, outp):
        if name not in store:
            store[name] = inp[0].detach()
        return None
    return hook


def _mem():
    for ln in open("/proc/meminfo"):
        if ln.startswith("MemAvailable"):
            return int(ln.split()[1]) // 1024
    return -1


def run(a, out, D, ad, live, items, tr, batch_loss, opt, lora):  # pragma: no cover - GPU
    import torch
    import thinker_lora as TL
    mods = lora["mods"]
    rep = {"checks": {}}

    def tf(ks, grad=False):
        """Teacher-forced forward, exactly the training layout. Returns text / function logits [B, T, V] and rows."""
        feats = live.feats(ks)
        sil = live.silence()
        seqs = []
        for (i, l), f in zip(ks, feats):
            it = items[i]
            st = student_targets(it, it["lead"], it["en_end"], it["end"][l], f.shape[0], l == "en")
            seqs.append((st, torch.stack([f[p[1]] if p[0] == "a" else sil for p in st["pos"]])))
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            X = torch.nn.utils.rnn.pad_sequence([s[1].float() for s in seqs], batch_first=True)
            audio = D.proj(ad(X).to(D.proj.weight.dtype))
            B, Tm = X.shape[0], X.shape[1]
            AI = torch.full((B, Tm), PAD, dtype=torch.long, device=D.dev)
            FI = AI.clone()
            for b, (st, _) in enumerate(seqs):
                AI[b, :len(st["pos"])] = torch.tensor(st["agent_in"], device=D.dev)
                FI[b, :len(st["pos"])] = torch.tensor(st["func_in"], device=D.dev)
            e = D.stt.embed_tokens
            fr = D.fusion(agent_text_embeds=e(AI).to(D.dtype), user_audio_embeds=audio.to(D.dtype), user_text_embeds=None,
                          function_embeds=e(FI).to(D.dtype))
            o = D.stt(torch.cat([D.prompt_emb.expand(B, -1, -1), fr], 1), cache=None)
            P = D.prompt_emb.shape[1]
            return o["text_logits"][:, P:], o["function_logits"][:, P:], [s[0] for s in seqs]

    def probe_metrics(ks):
        lt, lf, rows = tf(ks)
        lt, lf = torch.log_softmax(lt.float(), -1), torch.log_softmax(lf.float(), -1)
        m = defaultdict(list)
        for b, st in enumerate(rows):
            for j, (k, g) in enumerate(zip(st["kind"], st["func_lbl"])):
                if k == "fc":
                    row = lf[b, j]
                    m["call_logp"].append(float(row[g]))
                    m["call_argmax_ok"].append(int(row.argmax()) == g)
                    m["call_margin"].append(float(row[g] - torch.cat([row[:g], row[g + 1:]]).max()))
                elif g == SOTC:
                    m["sotc_logp"].append(float(lf[b, j, SOTC]))
            lis = [j for j, (p, k) in enumerate(zip(st["pos"], st["kind"])) if k == "speech" and st["text_lbl"][j] == PAD]
            m["bos_argmax_listen"].append(sum(int(lt[b, j].argmax()) == BOS for j in lis))
        return {k: round(float(np.mean(v)), 4) for k, v in m.items()} | {"n_call_tokens": len(m["call_logp"])}

    # ---------------- S1 audit
    exp = {n: m.base.weight.shape for n, m in mods.items()}
    trainable = [n for mod in (D.stt, D.fusion, ad, live.se) for n, p in mod.named_parameters() if p.requires_grad]
    rep["S1"] = {"targets": {n: list(s) for n, s in exp.items()}, "lora_params": TL.n_params(mods),
                 "trainable_names_non_lora": [n for n in trainable if "lora_" not in n],
                 "n_trainable_lora_tensors": sum("lora_" in n for n in trainable)}
    if lora.get("joint"):                # B0: LoRA + adapter + encoder LayerNorms, nothing in the Thinker base / heads / proj
        ok_ids = {id(p) for p in ad.parameters()} | {id(p) for p in live.ln.values()}
        bad = [n for mod in (D.stt, D.fusion, ad, live.se) for n, p in mod.named_parameters()
               if p.requires_grad and "lora_" not in n and id(p) not in ok_ids]
        rep["S1"]["joint_trainable_counts"] = {"adapter": sum(p.numel() for p in ad.parameters()),
                                               "encoder_ln": sum(p.numel() for p in live.ln.values())}
        rep["checks"]["S1_only_lora_bridge_trainable"] = not bad and rep["S1"]["n_trainable_lora_tensors"] == 2 * len(mods)
        rep["S1"]["unexpected_trainable"] = bad[:10]
    else:
        rep["checks"]["S1_only_lora_trainable"] = not rep["S1"]["trainable_names_non_lora"] and \
            rep["S1"]["n_trainable_lora_tensors"] == 2 * len(mods)
    rep["checks"]["S1_dtypes_fp32"] = all(p.dtype == torch.float32 for p in TL.lora_parameters(mods))
    # fixed train clips: the first call clips of the (seeded) train key list, 2 per language for the probe set
    call_keys = [k for k in tr if items[k[0]].get("fc")]
    probe = []
    for lang in ("en", "it", "fr"):
        probe += [k for k in call_keys if k[1] == lang][:3 if lang != "en" else 2]
    rep["probe_keys"] = [f"{i}|{l}" for i, l in probe]
    ks2 = probe[:2]
    # ---------------- S2 identity
    TL.set_enabled(mods, False)
    t0, f0, _ = tf(ks2)
    TL.set_enabled(mods, True)
    t1, f1, _ = tf(ks2)
    rep["checks"]["S2_tf_identity_text"] = bool(torch.equal(t0, t1))
    rep["checks"]["S2_tf_identity_func"] = bool(torch.equal(f0, f1))

    def greedy_one(k, topk=0):
        f = live.feats([k])[0]
        with torch.no_grad():
            audio = D.proj(ad(f.float()).to(D.proj.weight.dtype))[None]
            sil = D.proj(ad(live.silence().float()[None]).to(D.proj.weight.dtype))[0]
        it = items[k[0]]
        n = min(audio.shape[1], it["end"][k[1]] + 60)
        return greedy_fc(D, audio[:, :n], sil, [n], topk=topk)[0], audio[0, :n], sil
    TL.set_enabled(mods, False)
    g0, _, _ = greedy_one(ks2[0])
    TL.set_enabled(mods, True)
    g1, _, _ = greedy_one(ks2[0])
    rep["checks"]["S2_greedy_identity"] = g0["main_text"] == g1["main_text"] and g0["main_func"] == g1["main_func"] \
        and g0["fc"] == g1["fc"]
    # ---------------- S3 gradients
    opt.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    comps = defaultdict(list)
    batch_loss(probe[:4], comps).backward()
    gn = {n: {"A": float(m.lora_A.grad.norm()) if m.lora_A.grad is not None else None,
              "B": float(m.lora_B.grad.norm()) if m.lora_B.grad is not None else None} for n, m in mods.items()}
    trainable_ids = {id(p) for p in TL.lora_parameters(mods)}
    if lora.get("joint"):
        trainable_ids |= {id(p) for p in ad.parameters()} | {id(p) for p in live.ln.values()}
    frozen_with_grad = [n for mod in (D.stt, D.fusion, ad, live.se) for n, p in mod.named_parameters()
                        if id(p) not in trainable_ids and p.grad is not None]
    if lora.get("joint"):
        nz = lambda ps: float(torch.sqrt(sum((p.grad.float() ** 2).sum() for p in ps if p.grad is not None)))  # noqa: E731
        rep["S3_joint_grad"] = {"adapter_affine": nz(list(ad.affine.parameters())),
                                "adapter_mlp": nz([p for n, p in ad.named_parameters() if not n.startswith("affine.")]),
                                "encoder_ln": nz(list(live.ln.values()))}
        rep["checks"]["S3_joint_bridge_grads_nonzero"] = all(v > 0 for v in rep["S3_joint_grad"].values())
    rep["S3"] = {"grad_norm": gn, "frozen_with_grad": frozen_with_grad[:10], "comp": {k: round(float(np.mean(v)), 4)
                                                                                       for k, v in comps.items()}}
    rep["checks"]["S3_all_B_grad_nonzero"] = all((v["B"] or 0) > 0 for v in gn.values())
    rep["checks"]["S3_A_grad_zero_at_B0"] = all((v["A"] or 0) == 0 for v in gn.values())
    rep["checks"]["S3_no_grad_on_frozen"] = not frozen_with_grad
    opt.zero_grad(set_to_none=True)
    _, lf, rows = tf(probe[:4], grad=True)
    loss_f = 0.0
    for b, st in enumerate(rows):
        n = len(st["pos"])
        ce = torch.nn.functional.cross_entropy(lf[b, :n].float(), torch.tensor(st["func_lbl"], device=D.dev), reduction="none")
        loss_f = loss_f + (ce * torch.tensor(st["func_w"], device=D.dev)).sum() / max(sum(st["func_w"]), 1e-6)
    loss_f.backward()
    rep["S3"]["func_only_B_grad"] = {n: float(m.lora_B.grad.norm()) for n, m in mods.items()}
    rep["checks"]["S3_func_loss_alone_reaches_B"] = all(v > 0 for v in rep["S3"]["func_only_B_grad"].values())
    opt.zero_grad(set_to_none=True)
    rep["S5_after_backward"] = {"peak_cuda_gib": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2), "mem_avail_mib": _mem()}
    # ---------------- S4 perturbed LoRA (temporary)
    saved = TL.state(mods)
    gen = torch.Generator(device="cpu").manual_seed(7)
    with torch.no_grad():
        for m in mods.values():
            m.lora_B.copy_(torch.randn(m.lora_B.shape, generator=gen).to(m.lora_B.device) * a.smoke_sigma)
    captured = {}
    hooks = [m.register_forward_hook(capture_hook(captured, n)) for n, m in mods.items()]
    t2, f2, rows2 = tf(ks2)
    for h in hooks:
        h.remove()
    fcpos = [(b, j) for b, st in enumerate(rows2) for j, k in enumerate(st["kind"]) if k == "fc"]
    rep["S4"] = {"sigma": a.smoke_sigma,
                 "func_logit_max_abs_change_fc": max(float((f2[b, j].float() - f1[b, j].float()).abs().max()) for b, j in fcpos),
                 "func_argmax_changed_fc": f"{sum(int(f2[b, j].argmax()) != int(f1[b, j].argmax()) for b, j in fcpos)}/{len(fcpos)}",
                 "survival": {n: TL.branch_survival(mods[n], x) for n, x in captured.items()}}
    rep["checks"]["S4_lora_changes_logits"] = rep["S4"]["func_logit_max_abs_change_fc"] > 0

    def cache_gap(on):
        TL.set_enabled(mods, on)
        rec, audio, sil = greedy_one(ks2[0], topk=8)
        tx, fn = rec["main_text"], rec["main_func"]
        T = len(tx)
        AI = torch.tensor([[PAD] + tx[:-1]], device=D.dev)
        FI = torch.tensor([[PAD] + fn[:-1]], device=D.dev)
        with torch.no_grad():
            e = D.stt.embed_tokens
            fr = D.fusion(agent_text_embeds=e(AI).to(D.dtype), user_audio_embeds=audio[None, :T].to(D.dtype),
                          user_text_embeds=None, function_embeds=e(FI).to(D.dtype))
            o = D.stt(torch.cat([D.prompt_emb, fr], 1), cache=None)
            lt = torch.log_softmax(o["text_logits"][0, D.prompt_emb.shape[1]:].float(), -1)
        gaps = [max(abs(float(lt[t, k]) - v) for k, v in zip(*rec["topk"][t])) for t in range(min(T, len(rec["topk"])))]
        agree = sum(int(lt[t].argmax()) == tx[t] or fn[t] == SOTC for t in range(T))
        return {"frames": T, "max_abs_topk_logp_gap": round(max(gaps), 4), "mean_gap": round(float(np.mean(gaps)), 5),
                "argmax_agree": f"{agree}/{T}"}
    rep["S4"]["cache_vs_full_lora_off"] = cache_gap(False)
    rep["S4"]["cache_vs_full_lora_on"] = cache_gap(True)
    off, on = rep["S4"]["cache_vs_full_lora_off"], rep["S4"]["cache_vs_full_lora_on"]
    rep["checks"]["S4_cache_gap_not_increased"] = on["max_abs_topk_logp_gap"] <= 2 * off["max_abs_topk_logp_gap"] + 0.05
    TL.load_state(mods, saved)
    TL.set_enabled(mods, True)
    # ---------------- S6 calibration (train clips only, discarded): one paired run per LR (same sampled batches)
    before = probe_metrics(probe)
    p0 = TL.state(mods)
    opt_state = opt.state_dict()
    lr0 = [g["lr"] for g in opt.param_groups]          # restored at the end (joint mode has per-group LRs)
    lrs = [float(x) for x in (a.smoke_lrs.split(",") if a.smoke_lrs else [a.lr])] if a.smoke_steps > 0 else []
    rep["S6"] = {"probe_before": before, "runs": {}, "note": "measurement only; no automatic LR choice"}
    for lr in lrs:
        TL.load_state(mods, p0)
        opt.load_state_dict(opt_state)
        for g in opt.param_groups:
            g["lr"] = lr
        rng = random.Random(123)
        steps = []
        for s in range(a.smoke_steps):
            opt.zero_grad(set_to_none=True)
            comps = defaultdict(list)
            tot = 0.0
            for _ in range(a.accum):
                ls = batch_loss(rng.sample(tr, a.batch), comps) / a.accum
                ls.backward()
                tot += float(ls)
            g = float(torch.sqrt(sum((p.grad.float() ** 2).sum() for p in TL.lora_parameters(mods) if p.grad is not None)))
            prev = TL.state(mods)
            torch.nn.utils.clip_grad_norm_(TL.lora_parameters(mods), 1.0)
            opt.step()
            u = TL.update_norms(prev, mods)
            steps.append({"step": s + 1, "loss": round(tot, 4), "grad_norm": round(g, 5),
                          "upd_B_total": round(math.sqrt(sum(v["upd"] ** 2 for k, v in u.items() if k.endswith("lora_B"))), 7),
                          "comp": {k: round(float(np.mean(v)), 4) for k, v in comps.items()}})
            print("SMOKE-CAL", lr, json.dumps(steps[-1]), flush=True)
        after = probe_metrics(probe)
        captured.clear()
        hooks = [m.register_forward_hook(capture_hook(captured, n)) for n, m in mods.items()]
        tf(probe[:2])
        for h in hooks:
            h.remove()
        rep["S6"]["runs"][str(lr)] = {
            "steps": steps, "probe_after": after,
            "probe_delta": {k: round(after[k] - before[k], 5) for k in before if isinstance(before[k], float)},
            "update_norms_total": TL.update_norms(p0, mods),
            "survival_after": {n: TL.branch_survival(mods[n], x) for n, x in captured.items()}}
        print("SMOKE-CAL-DONE", lr, json.dumps(rep["S6"]["runs"][str(lr)]["probe_delta"]), flush=True)
    TL.load_state(mods, p0)                          # discard the calibration weights and optimiser state
    opt.load_state_dict(opt_state)
    for g, lr in zip(opt.param_groups, lr0):
        g["lr"] = lr
    opt.zero_grad(set_to_none=True)
    rep["S5"] = {"peak_cuda_gib": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2), "mem_avail_mib": _mem()}
    rep["pass"] = all(rep["checks"].values())
    Path(out, "smoke_l1.json").write_text(json.dumps(rep, indent=1, default=str))
    print("SMOKE-L1", json.dumps({"checks": rep["checks"], "S5": rep["S5"],
                                  "probe_delta": {k: v["probe_delta"] for k, v in rep["S6"]["runs"].items()}}), flush=True)
    if not rep["pass"]:
        raise SystemExit("L1 smoke checks failed")
