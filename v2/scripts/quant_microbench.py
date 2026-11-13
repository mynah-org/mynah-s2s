"""P1 quantization research: which weight formats actually run fast on GB10 (sm_121) for the Thinker's batch-1 decode?

The Thinker step is bandwidth-bound (C2/C6: 68-74 ms/step, `gemv` kernels). This microbench times, for every linear
shape of NemotronH-9B (MLP up/down, Mamba in/out proj, attention q/kv/o) at M=1 (one 80 ms frame), the kernels that
torch 2.10 ships natively — no new dependency:
  * bf16      F.linear (baseline, what the native engine runs)
  * int8_wo   torch._weight_int8pack_mm (per-channel int8 weights, bf16 activations), if the CUDA kernel exists
  * int4_wo   torch._weight_int4pack_mm (tinygemm, group 128, bf16 scales/zeros), if the CUDA kernel exists
  * fp8_row   torch._scaled_mm with per-row fp8 e4m3 weights and per-token fp8 activations, if supported
For each: µs per call (CUDA events, median of 200), relative output error vs bf16, and the estimated Thinker linear
time per step (shape time × layer count). "Format available" ≠ "fast on sm_121": this is the check.
"""

from __future__ import annotations

import json
import statistics as S
import sys

import torch
import torch.nn.functional as F

H, I, DI, NG, ST, NH = 4480, 15680, 10240, 8, 128, 128
SHAPES = {  # name: (in, out, layers)
    "mlp_up": (H, I, 24), "mlp_down": (I, H, 24),
    "mamba_in": (H, 2 * DI + 2 * NG * ST + NH, 27), "mamba_out": (DI, H, 27),
    "attn_q": (H, 5120, 4), "attn_kv": (H, 2048, 4), "attn_o": (5120, H, 4),
}


def timeit(fn, n=200, warm=20):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        e.synchronize()
        ts.append(s.elapsed_time(e) * 1000)
    return S.median(ts)


def rel_err(y, ref):
    return float((y.float() - ref.float()).norm() / ref.float().norm())


def main():
    dev = "cuda"
    torch.manual_seed(0)
    out = {"device": torch.cuda.get_device_name(), "capability": torch.cuda.get_device_capability(),
           "torch": torch.__version__, "shapes": {}}
    tot = {}
    for name, (k, n, layers) in SHAPES.items():
        w = (torch.randn(n, k, device=dev) * 0.02).to(torch.bfloat16)
        x = torch.randn(1, k, device=dev).to(torch.bfloat16)
        ref = F.linear(x, w)
        r = {"in": k, "out": n, "layers": layers, "bf16_MB": round(w.numel() * 2 / 2**20, 1)}
        r["bf16_us"] = timeit(lambda: F.linear(x, w))
        # int8 weight-only, per output channel
        try:
            sc = w.float().abs().amax(1).clamp_min(1e-8) / 127
            wq = torch.round(w.float() / sc[:, None]).clamp(-127, 127).to(torch.int8).contiguous()
            scb = sc.to(torch.bfloat16)
            y = torch._weight_int8pack_mm(x, wq, scb)
            r["int8_wo_us"], r["int8_wo_err"] = timeit(lambda: torch._weight_int8pack_mm(x, wq, scb)), rel_err(y, ref)
        except Exception as e:  # noqa: BLE001
            r["int8_wo"] = f"unavailable: {type(e).__name__}: {str(e)[:120]}"
        # int4 weight-only, tinygemm group 128
        try:
            g = 128
            wf = w.float().reshape(n, k // g, g)
            mn, mx = wf.amin(-1), wf.amax(-1)
            scale = ((mx - mn).clamp_min(1e-8) / 15)
            zero = mn + scale * 8
            q = torch.round((wf - mn[..., None]) / scale[..., None]).clamp(0, 15).to(torch.int32).reshape(n, k)
            packed = torch._convert_weight_to_int4pack(((q[:, ::2] << 4) | q[:, 1::2]).to(torch.uint8), 8)
            sz = torch.stack([scale, zero], -1).transpose(0, 1).contiguous().to(torch.bfloat16)  # (k/g, n, 2)
            y = torch._weight_int4pack_mm(x, packed, g, sz)
            r["int4_wo_us"], r["int4_wo_err"] = timeit(lambda: torch._weight_int4pack_mm(x, packed, g, sz)), rel_err(y, ref)
        except Exception as e:  # noqa: BLE001
            r["int4_wo"] = f"unavailable: {type(e).__name__}: {str(e)[:120]}"
        # fp8 rowwise
        try:
            f8 = torch.float8_e4m3fn
            ws = (w.float().abs().amax(1, keepdim=True) / 448).clamp_min(1e-12)
            w8 = (w.float() / ws).to(f8)
            xp = torch.zeros(16, k, device=dev, dtype=torch.bfloat16)   # _scaled_mm wants M multiple of 16
            xp[:1] = x
            xs = (xp.float().abs().amax(1, keepdim=True) / 448).clamp_min(1e-12)
            x8 = (xp.float() / xs).to(f8)
            def f():
                return torch._scaled_mm(x8, w8.t(), scale_a=xs, scale_b=ws.t(), out_dtype=torch.bfloat16)
            y = f()[:1]
            r["fp8_row_us"], r["fp8_row_err"] = timeit(f), rel_err(y, ref)
        except Exception as e:  # noqa: BLE001
            r["fp8_row"] = f"unavailable: {type(e).__name__}: {str(e)[:120]}"
        out["shapes"][name] = r
        for key in ("bf16_us", "int8_wo_us", "int4_wo_us", "fp8_row_us"):
            if key in r:
                tot[key] = tot.get(key, 0.0) + r[key] * layers
        print(json.dumps({name: r}), flush=True)
    out["thinker_linear_ms_per_step"] = {k: round(v / 1000, 2) for k, v in tot.items()}
    out["bandwidth_floor_note"] = "bf16 weights read per step ≈ sum(bf16_MB × layers)"
    out["bf16_GB_per_step"] = round(sum(r["bf16_MB"] * r["layers"] for r in out["shapes"].values()) / 1024, 2)
    print(json.dumps({"thinker_linear_ms_per_step": out["thinker_linear_ms_per_step"],
                      "bf16_GB_per_step": out["bf16_GB_per_step"]}), flush=True)
    if len(sys.argv) > 1:
        open(sys.argv[1], "w").write(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
