"""M4 (owner GO 2026-11-13): does the lab run identically on the canonical engine?

The same lab command runs under two stacks on the same machine:
- old = the lab code at the freeze tag + the pinned engine snapshot it was trained with;
- new = labs/multilingual + this repository's v2.
This wrapper records what the engine builds and what the Thinker sees, then runs the unchanged lab entry point in the
same process.

  m4_parity.py c1 --lab-dir D --out O [--compare-to OLD/construction.json] -- <glex_free.py args>
  m4_parity.py c2 --lab-dir D --out O [--compare-to OLD/construction.json] -- <session_distill.py train args>
  m4_parity.py compare --old O1 --new O2 --mode c1|c2 --out report.json

Recorded (construction.json, written right after the engine load, before anything else):
- module tree (names, types);
- every parameter and buffer: name / shape / dtype / device / sha256 of the bytes;
- engine cfg (yaml, output_dir normalised);
- fusion module type and weights, text_pad_id, the silence-embedding cache, the duplex prompt embedding (sha256);
- active patches: `fc_ctx_fix_info` if the engine has it, the qualnames of the wrapper / context-manager / attention
  forwards that patches replace;
- versions: torch / CUDA / cuDNN / transformers / NeMo commit; TF32 and determinism flags.
With --compare-to, the new run compares its construction record with the old one **before** running the lab command,
and exits 3 on any difference outside the declared ones (DECLARED_DIFF). The owner's stop rule: no C1 / C2 if
construction parity fails.

Recorded during the run (calls.jsonl):
- every Thinker call (`stt_model` forward): sha256 of the input embeddings and of the text / function logits;
- c2 only: every `cross_entropy` call (sha of input and target, value) and, at `clip_grad_norm_` (pre-clip), every
  trainable tensor in order (shape, dtype, sha of value and grad, grad norm).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

# Differences allowed between old and new construction records, declared before any run (owner 2026-11-13):
# the new engine's FC hand-back fix exists only in the new engine and patches only the NeMo streaming FC hand-back
# (context manager + wrapper FC thread), which the lab's training and free-running paths never call.
DECLARED_DIFF = ("fc_ctx_fix_info", "patches.S2SContextManager.__init__", "patches.wrapper.infer_one_step",
                 "patches.wrapper._run_fc_async_steps", "versions.engine_path", "versions.lab_dir")
# C2 tolerances, declared before any run: identical inputs / targets / trainable values are required (sha);
# loss components and total |rel| <= 1e-3; pre-clip gradient norm per tensor |rel| <= 1e-2. Grad sha equality is
# reported, never required (bf16 backward reductions on GPU are not guaranteed bit-reproducible).
C2_TOL = {"loss_rel": 1e-3, "grad_norm_rel": 1e-2}


def sha(t) -> str:
    import torch
    x = t.detach()
    if x.dtype == torch.bfloat16:
        x = x.view(torch.int16)
    return hashlib.sha256(x.contiguous().cpu().numpy().tobytes()).hexdigest()


def qual(f) -> str:
    f = getattr(f, "__func__", f)
    return f"{getattr(f, '__module__', '?')}.{getattr(f, '__qualname__', repr(f))}"


def construction(eng, prompt: str, lab_dir: str, light: bool = False) -> dict:
    import torch
    import transformers
    from omegaconf import OmegaConf
    from distill_adapter import Duplex
    m = eng.wrapper.model
    rec = {"modules": [[n, type(x).__name__] for n, x in m.named_modules()], "tensors": {}}
    for kind, it in (("param", m.named_parameters()), ("buffer", m.named_buffers())):
        for n, t in it:
            rec["tensors"][f"{kind}:{n}"] = [list(t.shape), str(t.dtype), str(t.device), None if light else sha(t)]
    cfg = OmegaConf.to_container(eng.cfg, resolve=True)
    if isinstance(cfg, dict):
        cfg.get("s2s", {}).pop("output_dir", None)
        cfg.pop("output_dir", None)
    rec["cfg"] = cfg
    D = Duplex(eng, prompt)
    f = D.fusion
    rec["fusion"] = {"type": type(f).__name__, **{k: getattr(f, k) for k in dir(f) if k.endswith("_weight")
                                                  and isinstance(getattr(f, k), (int, float))}}
    rec["text_pad_id"] = int(D.pad)
    rec["silence_sha"] = sha(D.w._silence_embedding_cache)
    rec["prompt_emb"] = [list(D.prompt_emb.shape), str(D.prompt_emb.dtype), sha(D.prompt_emb)]
    from nemo.collections.speechlm2.inference.streaming.state import s2s_context_manager as scm
    w = eng.wrapper
    rec["patches"] = {"S2SContextManager.__init__": qual(scm.S2SContextManager.__init__),
                      "wrapper.infer_one_step": qual(getattr(w, "infer_one_step", None)),
                      "wrapper._run_fc_async_steps": qual(getattr(w, "_run_fc_async_steps", None)),
                      "attention_forwards": sorted({qual(b.mixer.forward) for b in D.stt.llm.layers
                                                    if getattr(b, "block_type", "") == "attention"}),
                      "llm.forward": qual(D.stt.llm.forward)}
    rec["fc_ctx_fix_info"] = getattr(eng, "fc_ctx_fix_info", None)
    import nemo
    nemo_dir = Path(nemo.__file__).resolve().parents[1]
    try:
        nemo_commit = subprocess.run(["git", "-C", str(nemo_dir), "rev-parse", "HEAD"], capture_output=True,
                                     text=True).stdout.strip()
    except Exception:  # noqa: BLE001
        nemo_commit = "?"
    import mynah_v2
    rec["versions"] = {"torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                       "transformers": transformers.__version__, "nemo_commit": nemo_commit,
                       "tf32_matmul": torch.backends.cuda.matmul.allow_tf32, "tf32_cudnn": torch.backends.cudnn.allow_tf32,
                       "cudnn_benchmark": torch.backends.cudnn.benchmark,
                       "deterministic": torch.are_deterministic_algorithms_enabled(),
                       "engine_path": str(Path(mynah_v2.__file__).resolve().parent), "lab_dir": lab_dir}
    return rec


def flat(d, pre=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict) and k != "tensors":
            out.update(flat(v, f"{pre}{k}."))
        else:
            out[f"{pre}{k}"] = v
    return out


def diff_construction(old: dict, new: dict) -> dict:
    fo, fn = flat(old), flat(new)
    keys = sorted(set(fo) | set(fn))
    bad, declared = [], []
    for k in keys:
        if k == "tensors":
            continue
        if fo.get(k) != fn.get(k):
            (declared if any(k == d or k.startswith(d + ".") for d in DECLARED_DIFF) else bad).append(k)
    to, tn = old["tensors"], new["tensors"]
    tdiff = sorted(k for k in set(to) | set(tn) if to.get(k) != tn.get(k))
    return {"n_tensors": [len(to), len(tn)], "tensor_diff": tdiff[:50], "n_tensor_diff": len(tdiff),
            "field_diff": bad, "declared_diff": declared, "pass": not bad and not tdiff}


def run(mode: str, a, rest: list[str]):
    sys.path.insert(0, a.lab_dir)
    import torch
    import distill_adapter
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    calls = open(out / "calls.jsonl", "w")
    orig_load, cache = distill_adapter.load_engine, {}

    def log(row):
        calls.write(json.dumps(row) + "\n")

    def load_engine(o):
        if "eng" in cache:
            return cache["eng"]
        eng = orig_load(o)
        cache["eng"] = eng
        prompt = json.loads(Path(a.prompt_from).read_text())["prompt"]
        rec = construction(eng, prompt, a.lab_dir, light=a.light)
        (out / "construction.json").write_text(json.dumps(rec))
        print("M4 construction:", len(rec["tensors"]), "tensors recorded", flush=True)
        if a.compare_to:
            rep = diff_construction(json.load(open(a.compare_to)), rec)
            (out / "construction_vs_old.json").write_text(json.dumps(rep, indent=1))
            print("M4 construction parity:", json.dumps({k: rep[k] for k in ("pass", "n_tensors", "n_tensor_diff",
                                                                              "field_diff", "declared_diff")}), flush=True)
            if not rep["pass"]:
                calls.close()
                raise SystemExit(3)
        stt = eng.wrapper.model.stt_model
        n = {"i": 0}

        def pre(mod, args, kwargs):
            x = args[0] if args else kwargs.get("inputs_embeds")
            n["x"] = sha(x) if x is not None else None
            n["shape"] = list(x.shape) if x is not None else None

        def post(mod, args, kwargs, res):
            row = {"call": n["i"], "x": n["x"], "shape": n["shape"]}
            if isinstance(res, dict):
                for k in ("text_logits", "function_logits"):
                    if k in res and res[k] is not None:
                        row[k] = sha(res[k])
            log(row)
            n["i"] += 1

        stt.register_forward_pre_hook(pre, with_kwargs=True)
        stt.register_forward_hook(post, with_kwargs=True)
        return eng

    distill_adapter.load_engine = load_engine
    if mode == "c2":
        F = torch.nn.functional
        ce0, clip0 = F.cross_entropy, torch.nn.utils.clip_grad_norm_
        k = {"ce": 0}

        def ce(inp, target, *x, **kw):
            r = ce0(inp, target, *x, **kw)
            log({"ce": k["ce"], "inp": sha(inp), "target": sha(target), "out": sha(r), "sum": float(r.float().sum())})
            k["ce"] += 1
            return r

        def clip(params, *x, **kw):
            ps = list(params)
            log({"clip": [[list(p.shape), str(p.dtype), sha(p), sha(p.grad) if p.grad is not None else None,
                           float(p.grad.float().norm()) if p.grad is not None else None] for p in ps]})
            return clip0(ps, *x, **kw)

        F.cross_entropy = ce
        torch.nn.utils.clip_grad_norm_ = clip
        import session_distill as entry
        sys.argv = ["session_distill.py"] + rest
    else:
        import glex_free as entry
        sys.argv = ["glex_free.py"] + rest
    try:
        entry.main()
    finally:
        calls.close()


def compare(a):
    o, n = Path(a.old), Path(a.new)
    rep = {"construction": diff_construction(json.load(open(o / "construction.json")),
                                             json.load(open(n / "construction.json")))}
    L = lambda p: [json.loads(x) for x in open(p / "calls.jsonl")]  # noqa: E731
    co, cn = L(o), L(n)
    th_o = [r for r in co if "call" in r]
    th_n = [r for r in cn if "call" in r]
    first = next((i for i, (x, y) in enumerate(zip(th_o, th_n)) if x != y), None)
    rep["thinker_calls"] = {"n": [len(th_o), len(th_n)], "identical": th_o == th_n, "first_diff": first}
    if a.mode == "c1":
        rows = lambda p: {(r["arm"], r["set"], r["id"], r["lang"]): r for r in map(json.loads, open(p / "free_rows.jsonl"))  # noqa: E731
                          if "score" in r}
        ro, rn = rows(o), rows(n)
        keys = sorted(set(ro) | set(rn))
        diff = [list(k) for k in keys if k not in ro or k not in rn or ro[k] != rn[k]]      # whole row: tokens, call, text, behaviour, score
        rep["c1"] = {"rows": [len(ro), len(rn)], "rows_differing": diff, "tokens_bit_exact": not diff and len(ro) == len(rn) > 0}
        rep["pass"] = rep["construction"]["pass"] and rep["c1"]["tokens_bit_exact"]
    else:
        ce_o = [r for r in co if "ce" in r]
        ce_n = [r for r in cn if "ce" in r]
        same_targets = [x["target"] for x in ce_o] == [y["target"] for y in ce_n]
        same_inp = [x["inp"] for x in ce_o] == [y["inp"] for y in ce_n]
        cl_o = next(r["clip"] for r in co if "clip" in r)
        cl_n = next(r["clip"] for r in cn if "clip" in r)
        vals_same = [x[:3] for x in cl_o] == [y[:3] for y in cl_n]
        gn = [(x[4], y[4]) for x, y in zip(cl_o, cl_n) if x[4] is not None and y[4] is not None]
        grel = max((abs(x - y) / max(abs(x), 1e-12) for x, y in gn), default=0.0)
        gsha = sum(x[3] == y[3] for x, y in zip(cl_o, cl_n))
        lo = [json.loads(x) for x in open(o / "c2_train" / "train_log.jsonl") if '"comp"' in x][0]
        ln = [json.loads(x) for x in open(n / "c2_train" / "train_log.jsonl") if '"comp"' in x][0]
        lrel = max([abs(lo["loss"] - ln["loss"]) / max(abs(lo["loss"]), 1e-12)] +
                   [abs(lo["comp"][k] - ln["comp"].get(k, float("nan"))) / max(abs(lo["comp"][k]), 1e-12) for k in lo["comp"]])
        rep["c2"] = {"ce_calls": [len(ce_o), len(ce_n)], "ce_targets_identical": same_targets,
                     "ce_inputs_identical": same_inp, "trainable_values_identical": vals_same,
                     "n_trainable": [len(cl_o), len(cl_n)], "grad_sha_identical": f"{gsha}/{len(cl_o)}",
                     "grad_norm_max_rel": grel, "loss_max_rel": lrel, "batch_ids_identical": lo["batch_ids"] == ln["batch_ids"],
                     "log_old": {k: lo[k] for k in ("loss", "grad", "comp")}, "log_new": {k: ln[k] for k in ("loss", "grad", "comp")},
                     "tolerances": C2_TOL}
        rep["pass"] = (rep["construction"]["pass"] and same_targets and vals_same and rep["c2"]["batch_ids_identical"]
                       and len(cl_o) == len(cl_n) and lrel <= C2_TOL["loss_rel"] and grel <= C2_TOL["grad_norm_rel"])
    Path(a.out).write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k != "construction"} | {"construction_pass": rep["construction"]["pass"]},
                     indent=1)[:4000])
    sys.exit(0 if rep["pass"] else 1)


def main():
    argv = sys.argv[1:]
    rest = []
    if "--" in argv:
        i = argv.index("--")
        argv, rest = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["c1", "c2", "compare"])
    ap.add_argument("--lab-dir", default=str(Path(__file__).resolve().parent))
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt-from", default="", help="teacher.json whose 'prompt' builds the duplex prompt embedding")
    ap.add_argument("--compare-to", default="")
    ap.add_argument("--light", action="store_true", help="tensor metadata without the sha256 of the bytes (c2: saves time)")
    ap.add_argument("--old", default="")
    ap.add_argument("--new", default="")
    a = ap.parse_args(argv)
    if a.mode == "compare":
        compare(a)
    else:
        run(a.mode, a, rest)


if __name__ == "__main__":
    main()
