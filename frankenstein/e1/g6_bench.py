"""G6 bench (GPU, inside the G6 job, before the session): Nano bf16 vs INT4 decode-only vs INT4 + decode CUDA graph,
and the NVFP4 probe.

Suite = the first turn of every tenant scenario (E1c, IT + EN) plus the tool prompts of the recorded E1k session (E1l
call A/B), each served by the session's worker configuration (tool step, constrained call, brief replies). Per prompt:
the decision, the parsed call (name + args), the reply token ids, per-step timings. A variant is acceptable when every
decision and call equals the bf16 reference; reply token identity is reported (not required: INT4 is not bit-exact).

NVFP4 probe: what this torch can do on this GPU (dtype present, torch._scaled_mm with FP4 operands) and, if the web UI
tree's Triton NVFP4 GEMV file is given, a one-linear microbench vs bf16 and INT4 tinygemm (M = 1) with the error vs the
fp32 dequantized reference.
"""

from __future__ import annotations

import json
import time


def suite_items(spec, session_summary=None, langs=("it", "en")) -> list:
    items = []
    for lang in langs:
        for sc in spec["scenarios"]:
            t = sc["turns"][0].get(lang)
            if t:
                items.append({"id": f"{sc['id']}_{lang}", "lang": lang, "msgs": [{"role": "user", "content": t}]})
    if session_summary:
        msgs = json.load(open(session_summary))["messages"]
        for i, m in enumerate(msgs):
            if m["role"] == "user" and i + 1 < len(msgs) and msgs[i + 1]["content"].startswith("<TOOLCALL>"):
                items.append({"id": f"session_{i}", "lang": "it", "msgs": msgs[: i + 1]})
    return items


def run_suite(make_worker, items, label) -> dict:  # pragma: no cover - GPU
    rows = []
    for n, it in enumerate(items):
        wk = make_worker(it["lang"])
        t0 = time.monotonic_ns()
        wk._serve_job(n + 1, [dict(x) for x in it["msgs"]], t0)
        ev = wk.poll()
        done = next(e for e in ev if e[0] == "done")
        steps = done[4].get("steps", [])
        dec = next((s.get("out") for s in steps if s["step"] == "decide"), None)
        rep = next((s for s in steps if s["step"] == "reply"), {})
        rows.append({"id": it["id"], "decision": (dec or "").strip(),
                     "calls": [c["name"] + json.dumps(c["args"], sort_keys=True, ensure_ascii=False) for c in wk.calls],
                     "reply_ids": [e[2] for e in ev if e[0] == "tok"], "reply": done[2],
                     "reply_n": rep.get("n"), "reply_ms": rep.get("ms"),
                     "steps": [{k: s.get(k) for k in ("step", "n", "ms", "first_ms", "generated")} for s in steps]})
    ms = [r["reply_ms"] / r["reply_n"] for r in rows if r.get("reply_n")]
    return {"label": label, "rows": rows, "per_token_ms_mean": round(sum(ms) / len(ms), 1) if ms else None}


def compare(ref: dict, var: dict) -> dict:
    a, b = ref["rows"], var["rows"]
    same_dec = sum(x["decision"] == y["decision"] for x, y in zip(a, b))
    same_call = sum(x["calls"] == y["calls"] for x, y in zip(a, b))
    same_reply = sum(x["reply_ids"] == y["reply_ids"] for x, y in zip(a, b))
    diffs = [{"id": x["id"], "ref": (x["decision"], x["calls"]), "var": (y["decision"], y["calls"])}
             for x, y in zip(a, b) if (x["decision"], x["calls"]) != (y["decision"], y["calls"])]
    return {"n": len(a), "same_decision": same_dec, "same_calls": same_call, "same_reply_tokens": same_reply,
            "tools_ok": same_dec == len(a) and same_call == len(a), "diffs": diffs[:10]}


def per_token_bench(wk, n_tokens=48) -> dict:  # pragma: no cover - GPU
    """Plain greedy decode of n tokens on a fixed prompt: per-token ms of the decode loop only."""
    import torch
    with torch.inference_mode():
        ids = wk._ids([{"role": "user", "content": "Raccontami in dettaglio come funziona la consegna di un divano."}])
        cache, h = wk._prefill(ids)
        logits = wk.m.lm_head(h[:, -1]).float()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_tokens):
            nxt = int(logits.argmax(-1))
            h = wk._step(wk.llm.embeddings(torch.tensor([[nxt]], device=wk.dev)), cache)
            logits = wk.m.lm_head(h[:, -1]).float()
        torch.cuda.synchronize()
    return {"tokens": n_tokens, "per_token_ms": round((time.perf_counter() - t0) * 1e3 / n_tokens, 2)}


def nvfp4_probe(m, wo_gemv_path: str | None) -> dict:  # pragma: no cover - GPU
    import importlib.util
    import torch
    out = {"torch": torch.__version__, "capability": list(torch.cuda.get_device_capability()),
           "float4_dtype": hasattr(torch, "float4_e2m1fn_x2")}
    lin = None
    for layer in m.backbone.layers:
        mx = getattr(layer, "mixer", None)
        up = getattr(mx, "up_proj", None) if mx is not None else None
        if up is not None:
            lin = getattr(up, "lin", up)
            break
    w = lin.weight.data
    n, k = w.shape
    out["shape"] = [n, k]
    x = torch.randn(1, k, device=w.device, dtype=torch.bfloat16)

    def bench(fn, reps=50):
        fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        torch.cuda.synchronize()
        return round((time.perf_counter() - t0) * 1e3 / reps, 4)
    ref = x.float() @ w.float().t()
    out["bf16_ms"] = bench(lambda: torch.nn.functional.linear(x, w))
    try:                                                   # cuBLASLt block-scaled FP4 through torch, if wired here
        a4 = torch.zeros(16, k // 2, dtype=torch.float4_e2m1fn_x2, device=w.device)
        b4 = torch.zeros(n, k // 2, dtype=torch.float4_e2m1fn_x2, device=w.device)
        sa = torch.ones(16 * (k // 16), dtype=torch.float8_e4m3fn, device=w.device)
        sb = torch.ones(n * (k // 16), dtype=torch.float8_e4m3fn, device=w.device)
        torch._scaled_mm(a4, b4.t(), sa, sb, out_dtype=torch.bfloat16)
        out["torch_scaled_mm_fp4"] = "ran"
    except Exception as e:  # noqa: BLE001
        out["torch_scaled_mm_fp4"] = f"unavailable: {type(e).__name__}: {str(e)[:200]}"
    if wo_gemv_path:
        spec = importlib.util.spec_from_file_location("webui_wo_gemv", wo_gemv_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for fmt in ("nvf4", "w4"):
            try:
                p = getattr(mod, f"pack_{fmt}")(w)
                y = mod.gemv(x, p)
                out[f"triton_{fmt}_ms"] = bench(lambda: mod.gemv(x, p))
                out[f"triton_{fmt}_maxdiff_vs_fp32"] = round(float((y.float() - ref).abs().max()), 4)
            except Exception as e:  # noqa: BLE001
                out[f"triton_{fmt}"] = f"error: {type(e).__name__}: {str(e)[:200]}"
    try:
        from mynah_v2.model.quant import Int4Linear
        q4 = Int4Linear(lin, max_group=128, clip=False)
        y = q4(x)
        out["int4_tinygemm_ms"] = bench(lambda: q4(x))
        out["int4_maxdiff_vs_fp32"] = round(float((y.float() - ref).abs().max()), 4)
        del q4
    except Exception as e:  # noqa: BLE001
        out["int4_tinygemm"] = f"error: {type(e).__name__}: {str(e)[:200]}"
    out["ref_absmax"] = round(float(ref.abs().max()), 3)
    return out
