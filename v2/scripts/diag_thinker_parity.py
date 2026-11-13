"""Thinker graph parity diagnosis (funnel level 2): run the same short input twice in one process — baseline
(mamba_host_pos + no_timer_sync) and with thinker_graph — recording every Thinker call (stt_model.forward): input
fingerprint, text/function argmax, and for the first calls the full text logits. Reports the first call whose INPUT
differs and the first call whose OUTPUT differs, with the graph's action on that call (eager / capture / replay), so a
wrong patched-eager path (mask / K-V update) is told apart from a wrong replay (stale buffers / baked state).
    diag_thinker_parity.py WAV OUT [--chunks 40] [--full 16]"""

import argparse
import json
import sys
from pathlib import Path

import torch

from mynah_v2.model import perf
from mynah_v2.model.reference import ReferenceEngine, build_cfg
from mynah_v2.runtime.session import iter_chunks

sys.path.insert(0, str(Path(__file__).parent))
from run_stream import load_16k  # noqa: E402


class Recorder:
    def __init__(self, stt, tg=None, full=16):
        self.stt, self.tg, self.full = stt, tg, full
        self.orig = stt.forward
        self.calls = []
        stt.forward = self.forward

    def forward(self, input_embeds, cache=None, **kw):
        before = dict(self.tg.stats) if self.tg is not None else None
        out = self.orig(input_embeds, cache=cache, **kw)
        x = input_embeds.float()
        tl = out["text_logits"][:, -1].float()
        rec = {"T": int(input_embeds.shape[1]), "in_sum": float(x.sum()), "in_absmax": float(x.abs().max()),
               "text_top": int(tl.argmax(-1)[0]), "pos": int(getattr(cache, "_v2_pos", -1)) if cache is not None else None}
        fl = out.get("function_logits")
        if fl is not None:
            rec["func_top"] = int(fl[:, -1].argmax(-1)[0])
        if self.tg is not None:
            a = self.tg.stats
            rec["action"] = ("capture" if a["captures"] > before["captures"] else
                             "replay" if a["replays"] > before["replays"] else "eager")
        if len(self.calls) < self.full:
            rec["_logits"] = tl[0].cpu()
        self.calls.append(rec)
        return out

    def remove(self):
        self.stt.forward = self.orig


def session(eng, x, chunks):
    eng.open("You are a helpful assistant.")
    for i, c in enumerate(iter_chunks(x, eng.chunk_samples)):
        eng.step(c)
        if i + 1 >= chunks:
            break
    eng.close()
    eng.release_fc()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("out")
    ap.add_argument("--chunks", type=int, default=40)
    ap.add_argument("--full", type=int, default=16)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    eng = ReferenceEngine(build_cfg(output_dir=str(out / "_nemo"), overrides={"s2s.deterministic": True}))
    for n in ("mamba_host_pos", "no_timer_sync"):
        perf.enable(n, eng)
    stt = eng.wrapper.model.stt_model
    x = torch.cat([load_16k(a.wav), torch.zeros(16000 * 5)])

    torch.manual_seed(1234)
    rb = Recorder(stt, full=a.full)
    session(eng, x, a.chunks)
    rb.remove()

    info = perf.enable("thinker_graph", eng)
    tg = eng._v2_thinker_graph
    torch.manual_seed(1234)
    rg = Recorder(stt, tg=tg, full=a.full)
    session(eng, x, a.chunks)
    rg.remove()

    B, G = rb.calls, rg.calls
    n = min(len(B), len(G))
    first_in = next((i for i in range(n) if (B[i]["T"], round(B[i]["in_sum"], 2)) != (G[i]["T"], round(G[i]["in_sum"], 2))), None)
    first_out = next((i for i in range(n) if B[i]["text_top"] != G[i]["text_top"] or B[i].get("func_top") != G[i].get("func_top")), None)
    rows = []
    for i in range(min(n, a.full)):
        d = float((B[i]["_logits"] - G[i]["_logits"]).abs().max())
        rows.append({"call": i, "T": B[i]["T"], "pos_b": B[i]["pos"], "pos_g": G[i]["pos"], "action": G[i].get("action"),
                     "in_equal": round(B[i]["in_sum"], 3) == round(G[i]["in_sum"], 3),
                     "top_b": B[i]["text_top"], "top_g": G[i]["text_top"], "logit_maxdiff": round(d, 4)})
    rep = {"calls": [len(B), len(G)], "first_input_diff": first_in, "first_output_diff": first_out,
           "action_at_first_output_diff": G[first_out].get("action") if first_out is not None else None,
           "graph_stats": {k: v for k, v in tg.stats.items() if k != "sync_sites"}, "enable": {k: v for k, v in info.items() if k != "stats"},
           "first_calls": rows}
    (out / "parity.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k != "first_calls"}), flush=True)
    for r in rows:
        print(json.dumps(r), flush=True)


if __name__ == "__main__":
    main()
