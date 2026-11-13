"""Can Thinker(t+1) and EarTTS(t) usefully run concurrently on GB10? (P1 overlap feasibility, before touching the loop)

Both are memory-traffic heavy, so concurrency may buy little. This times, on the real shapes:
  A = one Thinker decode step's linears (NemotronH-9B, BF16, M=1: 24 MLP up/down, 27 Mamba in/out, 4 attention q/kv/o)
  B = one EarTTS backbone step's linears (Gemma3, 28 layers, hidden 1152, MLP 4608 gated, FP32, batch 2 for CFG)
alone, back to back on one stream (A+B serial), and concurrently on two CUDA streams (A ‖ B). Reports ms and the
achieved effective bandwidth. The overlap gain upper bound for the real loop is (serial − concurrent).
"""

from __future__ import annotations

import json
import statistics as S
import sys

import torch
import torch.nn.functional as F

H, I, DI, NG, ST, NH = 4480, 15680, 10240, 8, 128, 128
THINKER = [(H, I, 24), (I, H, 24), (H, 2 * DI + 2 * NG * ST + NH, 27), (DI, H, 27), (H, 5120, 4), (H, 2048, 4), (5120, H, 4)]
EH, EI, EL = 1152, 4608, 28
EARTTS = [(EH, 3 * EH, EL), (EH, EH, EL), (EH, 2 * EI, EL), (EI, EH, EL)]      # qkv, o, gate+up, down


def build(spec, dtype, m):
    ws, xs = [], []
    for k, n, layers in spec:
        for _ in range(layers):
            ws.append((torch.randn(n, k, device="cuda") * 0.02).to(dtype))
        xs.append(torch.randn(m, k, device="cuda").to(dtype))
    return ws, {k: x for (k, _, _), x in zip(spec, xs)}


def run(ws, xs):
    for w in ws:
        F.linear(xs[w.shape[1]], w)


def timeit(fn, n=30):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return S.median(ts)


def main():
    a_w, a_x = build(THINKER, torch.bfloat16, 1)
    b_w, b_x = build(EARTTS, torch.float32, 2)
    gb_a = sum(w.numel() * w.element_size() for w in a_w) / 1e9
    gb_b = sum(w.numel() * w.element_size() for w in b_w) / 1e9
    sa, sb = torch.cuda.Stream(), torch.cuda.Stream()
    t_a = timeit(lambda: run(a_w, a_x))
    t_b = timeit(lambda: run(b_w, b_x))
    t_serial = timeit(lambda: (run(a_w, a_x), run(b_w, b_x)))

    def conc():
        cur = torch.cuda.current_stream()
        sa.wait_stream(cur)
        sb.wait_stream(cur)
        with torch.cuda.stream(sa):
            run(a_w, a_x)
        with torch.cuda.stream(sb):
            run(b_w, b_x)
        cur.wait_stream(sa)
        cur.wait_stream(sb)
    t_conc = timeit(conc)
    # interleaved launches (one host thread, alternating the two streams layer by layer)
    pairs = list(zip(a_w, b_w)) + [(w, None) for w in a_w[len(b_w):]]

    def inter():
        cur = torch.cuda.current_stream()
        sa.wait_stream(cur)
        sb.wait_stream(cur)
        for wa, wb in pairs:
            with torch.cuda.stream(sa):
                F.linear(a_x[wa.shape[1]], wa)
            if wb is not None:
                with torch.cuda.stream(sb):
                    F.linear(b_x[wb.shape[1]], wb)
        cur.wait_stream(sa)
        cur.wait_stream(sb)
    t_inter = timeit(inter)
    res = {"thinker_GB": round(gb_a, 2), "eartts_GB": round(gb_b, 2),
           "thinker_ms": round(t_a, 2), "eartts_ms": round(t_b, 2), "serial_ms": round(t_serial, 2),
           "concurrent_ms": round(t_conc, 2), "interleaved_ms": round(t_inter, 2),
           "thinker_GBps": round(gb_a / t_a * 1000, 1), "eartts_GBps": round(gb_b / t_b * 1000, 1),
           "serial_GBps": round((gb_a + gb_b) / t_serial * 1000, 1),
           "best_overlap_gain_ms": round(t_serial - min(t_conc, t_inter), 2)}
    print(json.dumps(res), flush=True)
    if len(sys.argv) > 1:
        open(sys.argv[1], "w").write(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
