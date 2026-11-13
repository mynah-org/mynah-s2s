"""V1 corrected: mandatory CPU gate (owner 2026-11-13) before any GPU. Uses NeMo's real create_fusion_module / AddFusion
and the real session_distill.greedy_fc with a recording fake Thinker (no weights, no GPU).
G1 our own fusion instance carries user_text_weight 1.0; the wrapper-style instance (no asr_head) carries 0.0; separate
   objects (the wrapper / NeMo / V2 code is not modified).
G2 synthetic non-zero text: fusion(audio + text) - fusion(audio) == text (ours); == 0 with the wrapper-style instance
   (the cause of the invalid V1); and the difference reaches the Thinker input that greedy_fc passes to stt.
G3 zero contribution: Thinker inputs bit-identical to user_text=None.
G4 timing per clip: RNNT pieces, injected token frames, SOTC, end of speech (baseline job 349), tokens before / after SOTC.
G5 FC masking: a user-text token scheduled during FC fast-path steps never reaches the Thinker input; the V0 stream has
   no id < 1000 (no BOS / EOS / PAD / SOTC / EOTC / EOTR / <SPECIAL_n>).
Exit code 1 if any gate fails. Usage: v1_gate.py --v0 v0_user_text.json --baseline free_rows.jsonl --out gate.json
"""
from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from session_distill import EOTC, PAD, SOTC  # noqa: E402

H, V = 64, 40


class FakeSTT:
    """Records every input; function head emits SOTC at main step `sotc_at`, then 2 call tokens and EOTC."""

    def __init__(self, sotc_at: int):
        self.inputs, self.calls, self.sotc_at, self.fc = [], 0, sotc_at, 0
        self.llm = types.SimpleNamespace(config=None)
        self.embed_tokens = torch.nn.Embedding(V, H)

    def __call__(self, x, cache=None):
        self.inputs.append(x.detach().clone())
        B, T = x.shape[:2]
        tl = torch.full((B, T, V), -10.0)
        tl[..., PAD] = 0.0
        fl = torch.full((B, T, V), -10.0)
        step = self.calls - 1                       # call 0 is the prompt prefill
        self.calls += 1
        if step >= 0 and step == self.sotc_at:
            fl[..., SOTC] = 0.0
        elif step > self.sotc_at and self.fc < 3:
            fl[..., (30, 31, EOTC)[self.fc]] = 0.0
            self.fc += 1
        else:
            fl[..., PAD] = 0.0
        return {"text_logits": tl, "function_logits": fl, "cache": cache}


def fake_D(fusion, sotc_at):
    stt = FakeSTT(sotc_at)
    return types.SimpleNamespace(stt=stt, fusion=fusion, dtype=torch.float32, dev="cpu",
                                 prompt_emb=torch.zeros(1, 2, H))


def run(fusion, audio, sil, ut, sotc_at):
    from session_distill import greedy_fc
    D = fake_D(fusion, sotc_at)
    torch.manual_seed(0)
    D.stt.embed_tokens.weight.data = torch.randn(V, H)
    recs = greedy_fc(D, audio, sil, [audio.shape[1]], user_text=ut)
    return D.stt.inputs, recs[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v0", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.modules["mynah_v2.model.llm_cache"] = types.SimpleNamespace(
        _hybrid_cache_cls=lambda llm: (lambda *x, **k: object()))           # the cache is irrelevant to a fake Thinker
    from nemo.collections.speechlm2.parts.fusion import create_fusion_module
    rep, ok = {}, {}
    wrapper_like = create_fusion_module(None, H, 1.0, 1.0, 0.0, 2.0)        # what wrapper:765-771 builds without asr_head
    ours = create_fusion_module(None, H, 1.0, 1.0, 1.0, 2.0)
    ok["G1_ours_weight_1"] = ours.user_text_weight == 1.0 and type(ours).__name__ == "AddFusion"
    ok["G1_wrapper_like_weight_0"] = wrapper_like.user_text_weight == 0.0 and ours is not wrapper_like
    g = torch.Generator().manual_seed(1)
    ag, au, fu, tx = (torch.randn(1, 5, H, generator=g) for _ in range(4))
    d_ours = ours(ag, au, tx, fu) - ours(ag, au, None, fu)
    d_wrap = wrapper_like(ag, au, tx, fu) - wrapper_like(ag, au, None, fu)
    ok["G2_ours_adds_text"] = bool(torch.allclose(d_ours, tx, atol=1e-5))
    ok["G2_wrapper_like_drops_text"] = bool(d_wrap.abs().max() == 0)
    T, k_sotc = 12, 7
    audio = torch.randn(1, T, H, generator=g)
    sil = torch.randn(H, generator=g)
    ut = torch.zeros(1, T, H)
    tok_frames = [3, 5, 9, 10]                       # 3, 5 before SOTC (main); 9, 10 would fall during FC
    for f in tok_frames:
        ut[0, f] = torch.randn(H, generator=g)
    ins_none, rec_none = run(ours, audio, sil, None, k_sotc)
    ins_zero, rec_zero = run(ours, audio, sil, torch.zeros_like(ut), k_sotc)
    ins_tok, rec_tok = run(ours, audio, sil, ut, k_sotc)
    ok["G3_zero_bitwise"] = len(ins_none) == len(ins_zero) and all(torch.equal(x, y) for x, y in zip(ins_none, ins_zero))
    main_steps = list(range(1, 1 + k_sotc + 1))      # input index i (1-based after prefill) <-> main frame i-1
    diff = {i - 1: float((ins_tok[i] - ins_none[i]).abs().max()) for i in main_steps}
    ok["G2_reaches_thinker_on_token_frames"] = all(diff[f] > 0 for f in tok_frames if f <= k_sotc)
    ok["G2_no_change_off_token_frames"] = all(diff[f] == 0 for f in diff if f not in tok_frames)
    fc_idx = range(1 + k_sotc + 1, len(ins_tok))
    ok["G5_fc_steps_unchanged"] = all(torch.equal(ins_tok[i], ins_none[i]) for i in fc_idx) and len(fc_idx) == 3
    ok["G5_same_call"] = rec_tok["fc"] == rec_none["fc"] == [30, 31, EOTC]
    rep["G2_main_frame_diffs"] = diff
    v0 = json.load(open(a.v0))
    ok["G5_no_special_ids_in_stream"] = all(t >= 1000 for v in v0["stream"].values() for _, t in v)
    base = {f"{r['set']}|{r['id']}|{r['lang']}": r for r in map(json.loads, open(a.baseline))
            if "score" in r and r["arm"] == "s103_last" and r["set"] == "glex_dev"}
    trace, nb, nt = {}, 0, 0
    for key, r in base.items():
        st, sotc = v0["stream"].get(key, []), r["beh"]["sotc_frame"]
        pcs = [p for p in v0["pieces"].get(key, []) if any(c.isalnum() for c in p[1]) and not p[1].startswith("<")]
        trace[key] = {"x_end": r["x_end"], "sotc": sotc, "fc_len": r["beh"]["fc_len"],
                      "rnnt_first_last": [pcs[0][0], pcs[-1][0]] if pcs else None,
                      "inject_first_last": [st[0][0], st[-1][0]] if st else None, "n_tokens": len(st),
                      "before_sotc": sum(1 for f, _ in st if sotc is not None and f < sotc) if sotc is not None else None}
        if sotc is not None:
            nb += trace[key]["before_sotc"]
            nt += len(st)
    rep["G4_trace"] = trace
    rep["G4_summary"] = {"clips": len(trace), "with_sotc": sum(v["sotc"] is not None for v in trace.values()),
                         "tokens_before_sotc": nb, "tokens_total_called_clips": nt}
    ok["G4_trace_complete"] = len(trace) == 36
    rep["checks"] = ok
    rep["pass"] = all(ok.values())
    Path(a.out).write_text(json.dumps(rep, indent=1))
    print(json.dumps({"checks": ok, "G4": rep["G4_summary"], "pass": rep["pass"]}, indent=1))
    sys.exit(0 if rep["pass"] else 1)


if __name__ == "__main__":
    main()
