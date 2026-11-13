"""B0 CPU check (owner 2026-11-13, before any GPU): joint surface = ML-encoder LayerNorms + ResidualAdapter + Thinker q/v LoRA.
Same autograd structure as `session_distill train --lora --lora-joint`, at toy scale on the lab-box CPU:
  real ML encoder (SpeechEncoder 'ml', CPU) with the D-V3 step-103 LNs, only the LNs trainable (SessionML rule:
  isinstance LayerNorm) -> real step-103 ResidualAdapter (trainable) -> a frozen Linear 1024->64 (stand-in for VC proj
  1024->4480) -> tiny NemotronH from the real remote code (thinker_lora_cpu_check.prepare) with the q/v LoRA ->
  frozen head on the last hidden state -> CE.
Checks:
  freeze      trainable = LN + adapter + LoRA only; proj / backbone base / head frozen
  parity      at init (LoRA B = 0) the joint graph output equals the frozen-bridge graph output bit for bit, and the
              step-103 encoder features equal the eval path (glex_free: SpeechEncoder + ln state copy)
  gradients   one backward reaches the LNs, the adapter affine and MLP, and every LoRA B; none on frozen tensors
  EN path     an EN clip goes through the identical code (no language branch); its features at init equal step 103
Usage: b0_cpu_check.py --snapshot <Nano-9B-v2 snapshot> --ln103 F --adapter103 F --wav-en F --wav-it F --out F
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--ln103", required=True)
    ap.add_argument("--adapter103", required=True)
    ap.add_argument("--wav-en", required=True)
    ap.add_argument("--wav-it", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lora-frozen", action="store_true", help="ablation: LoRA installed at B = 0 but not trainable")
    a = ap.parse_args()
    assert not torch.cuda.is_available()
    torch.cuda.default_stream = lambda *x, **k: None
    torch.cuda.stream = lambda *x, **k: contextlib.nullcontext()
    import soundfile as sf
    import thinker_lora as TL
    from adapters import load_adapter
    from encoders import SpeechEncoder
    from session_distill import session_wav
    from thinker_lora_cpu_check import prepare
    ok, rep = {}, {}
    se = SpeechEncoder("ml", att=[56, 0], device="cpu")
    se.eval()
    for p in se.parameters():
        p.requires_grad_(False)
    ln = {n: p for n, p in se.enc.named_parameters()
          if isinstance(se.enc.get_submodule(n.rsplit(".", 1)[0]), torch.nn.LayerNorm)}
    st = torch.load(a.ln103, map_location="cpu", weights_only=False)
    ok["ln_state_matches"] = set(st) == set(ln)
    with torch.no_grad():
        for n, p in ln.items():
            p.copy_(st[n])
    ref = SpeechEncoder("ml", att=[56, 0], device="cpu")                   # the eval path (glex_free arm)
    prm = dict(ref.enc.named_parameters())
    with torch.no_grad():
        for k, v in st.items():
            prm[k].copy_(v)
    ad = load_adapter(a.adapter103).float()
    proj = torch.nn.Linear(1024, 64, bias=False)
    for p in proj.parameters():
        p.requires_grad_(False)
    cfg, llm, attn = prepare(a.snapshot, torch.float32, "M-M*M-*-")
    head = torch.nn.Linear(64, 32, bias=False)
    for p in head.parameters():
        p.requires_grad_(False)
    mods = TL.install(llm, layers=tuple(attn))
    if a.lora_frozen:
        for p in TL.lora_parameters(mods):
            p.requires_grad_(False)

    def enc(path, model, grad):
        x = sf.read(path, dtype="float32")[0]
        w = torch.from_numpy(session_wav(x, 20, 1.0))
        if grad:                                   # SessionML._enc: pre + enc called directly, grads into the LNs
            n = torch.tensor([len(w)])
            mel, ml = model.pre(input_signal=w[None], length=n)
            e, el = model.enc(audio_signal=mel, length=ml)
            return e[0, :, : int(el[0])].transpose(0, 1)
        return model(w)["out"]

    with torch.no_grad():
        f_eval = enc(a.wav_it, ref, False)
        f_en_eval = enc(a.wav_en, ref, False)
    for p in ln.values():
        p.requires_grad_(True)
    for p in ad.parameters():
        p.requires_grad_(True)
    f = enc(a.wav_it, se, True)
    f_en = enc(a.wav_en, se, True)
    rep["feat_maxdiff_it"] = float((f.detach() - f_eval).abs().max())
    rep["feat_maxdiff_en"] = float((f_en.detach() - f_en_eval).abs().max())
    ok["parity_features_it"] = rep["feat_maxdiff_it"] < 1e-5
    ok["parity_features_en"] = rep["feat_maxdiff_en"] < 1e-5

    def thinker(feats, lora_on=True):
        TL.set_enabled(mods, lora_on)
        x = proj(ad(feats))[None]
        return head(llm(inputs_embeds=x, return_dict=True)["last_hidden_state"])

    with torch.no_grad():
        y_frozen = thinker(f.detach(), lora_on=False)
    y_joint = thinker(f, lora_on=True)
    ok["parity_init_joint_vs_frozen"] = bool(torch.equal(y_joint.detach(), y_frozen))
    torch.nn.functional.cross_entropy(y_joint[0].float(), torch.randint(0, 32, (y_joint.shape[1],))).backward()
    g = lambda ps: float(torch.sqrt(sum((p.grad ** 2).sum() for p in ps if p.grad is not None)))  # noqa: E731
    rep["grad"] = {"encoder_ln": g(list(ln.values())), "adapter_affine": g(list(ad.affine.parameters())),
                   "adapter_mlp": g([p for n, p in ad.named_parameters() if not n.startswith("affine.")])}
    if a.lora_frozen:
        ok["lora_has_no_grad"] = all(m.lora_A.grad is None and m.lora_B.grad is None for m in mods.values())
    else:
        rep["grad"]["lora_B_min"] = min(float(m.lora_B.grad.norm()) for m in mods.values())
    ok["grads_reach_all_trainable"] = all(v > 0 for v in rep["grad"].values())
    tr_ids = {id(p) for p in ln.values()} | {id(p) for p in ad.parameters()}
    if not a.lora_frozen:
        tr_ids |= {id(p) for p in TL.lora_parameters(mods)}
    frozen_grad = [n for mod, tag in ((se, "se"), (proj, "proj"), (llm, "llm"), (head, "head"))
                   for n, p in mod.named_parameters() if id(p) not in tr_ids and p.grad is not None]
    ok["no_grad_on_frozen"] = not frozen_grad
    rep["trainable_counts"] = {"encoder_ln": sum(p.numel() for p in ln.values()),
                               "adapter": sum(p.numel() for p in ad.parameters()), "lora": TL.n_params(mods)}
    rep["checks"] = ok
    rep["pass"] = all(ok.values())
    Path(a.out).write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    sys.exit(0 if rep["pass"] else 1)


if __name__ == "__main__":
    main()
