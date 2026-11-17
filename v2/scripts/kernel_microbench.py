"""Precision track step 4: which weight formats have a FAST kernel for the batch-1 Thinker decode on this GPU?

For every NemotronH-9B backbone linear shape at M=1 (and M=2, the size a 2-session batch would have): BF16 F.linear
(baseline), int4 tinygemm (torch native), and our Triton weight-only GEMVs (mynah_v2.model.wo_gemv: w8, f8, w4 g128,
NVFP4). Per kernel: µs/call (CUDA events, median), weight bytes, effective GB/s, kernel error vs the fp32 product with
the SAME dequantized weights (kernel correctness) and format error vs the BF16 product (what the format costs), and the
projected Thinker linear time per step (µs × layer count). Tensor-core FP4/FP8 GEMMs are not the right tool at M=1
(no reuse of the weights, the step is a bandwidth-bound GEMV), so NVFP4 is measured as weight-only dequant here.
"""

from __future__ import annotations

import argparse
import json
import statistics as S
from pathlib import Path

import torch
import torch.nn.functional as F

from mynah_v2.model import wo_gemv as W

H, I, DI, NG, ST, NH = 4480, 15680, 10240, 8, 128, 128
SHAPES = {"mlp_up": (H, I, 24), "mlp_down": (I, H, 24), "mamba_in": (H, 2 * DI + 2 * NG * ST + NH, 27),
          "mamba_out": (DI, H, 27), "attn_q": (H, 5120, 4), "attn_kv": (H, 2048, 4), "attn_o": (5120, H, 4)}


def timeit(fn, n=30, warm=10, inner=10):
    """µs per call: median over n event pairs, each timing `inner` back-to-back calls (event overhead amortized).
    Note: the same weights are re-read every call, so tensors that fit in L2 (small attention shapes, packed int4)
    look faster than in the real step, where 120 different linears stream through."""
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(inner):
            fn()
        e.record()
        e.synchronize()
        ts.append(s.elapsed_time(e) * 1000 / inner)
    return S.median(ts)


def rel(y, ref):
    return float((y.float() - ref.float()).norm() / ref.float().norm())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--m", default="1,2,4")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(0)
    rep = {"device": torch.cuda.get_device_name(), "capability": torch.cuda.get_device_capability(),
           "torch": torch.__version__, "shapes": {}, "projected_thinker_linear_ms": {}}
    import triton
    rep["triton"] = triton.__version__
    for m in map(int, a.m.split(",")):
        tot = {}
        for name, (k, n, layers) in SHAPES.items():
            w = (torch.randn(n, k, device="cuda") * 0.02).to(torch.bfloat16)
            x = torch.randn(m, k, device="cuda").to(torch.bfloat16)
            ref = F.linear(x.float(), w.float())
            r = {"K": k, "N": n, "layers": layers}
            r["bf16"] = {"us": timeit(lambda: F.linear(x, w)), "bytes": w.numel() * 2, "fmt_err": rel(F.linear(x, w), ref)}
            try:                                                   # torch-native int4 tinygemm
                from mynah_v2.model.quant import _pick, quantize_int4
                group, tiles = _pick(k, 128)
                packed, sz = quantize_int4(w, group, tiles)
                y = torch._weight_int4pack_mm(x, packed, group, sz)
                r["int4_tinygemm"] = {"group": group, "us": timeit(lambda: torch._weight_int4pack_mm(x, packed, group, sz)),
                                      "bytes": packed.numel() * packed.element_size() + sz.numel() * 2,
                                      "fmt_err": rel(y, ref)}
            except Exception as e:  # noqa: BLE001
                r["int4_tinygemm"] = {"error": f"{type(e).__name__}: {str(e)[:160]}"}
            for fmt, pack in W.PACKERS.items():
                try:
                    p = pack(w)
                    y = W.gemv(x, p)
                    deq = F.linear(x.float(), W.dequant(p))
                    r[fmt] = {"us": timeit(lambda: W.gemv(x, p)), "bytes": W.nbytes(p),
                              "kernel_err": rel(y, deq), "fmt_err": rel(y, ref)}
                except Exception as e:  # noqa: BLE001
                    r[fmt] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
            for key, v in r.items():
                if isinstance(v, dict) and "us" in v:
                    v["GBps"] = round(v["bytes"] / v["us"] / 1e3, 1)
                    v["us"] = round(v["us"], 1)
                    tot[key] = tot.get(key, 0.0) + v["us"] * layers
            rep["shapes"][f"{name}/M{m}"] = r
            print(json.dumps({f"{name}/M{m}": {k2: (v if not isinstance(v, dict) else
                                                    {kk: (round(vv, 5) if isinstance(vv, float) else vv) for kk, vv in v.items()})
                                               for k2, v in r.items()}}), flush=True)
        rep["projected_thinker_linear_ms"][f"M{m}"] = {k: round(v / 1000, 2) for k, v in tot.items()}
        print(json.dumps({f"projected_thinker_linear_ms/M{m}": rep["projected_thinker_linear_ms"][f"M{m}"]}), flush=True)
    (out / "kernels.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
