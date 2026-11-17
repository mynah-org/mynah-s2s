"""G6: Nano's decode step as a CUDA graph, reusing V2's ThinkerGraph (mynah_v2.model.thinker_graph) unchanged.

Nano-9B-v2 and the duplex Thinker share the NemotronH remote code (snapshot 6533e8de), so the Thinker's graph machinery
applies as is: attention K/V in static buffers per length bucket with an additive mask, Mamba conv / SSM states in fixed
buffers updated in place by the decode kernels, the stream-neutral block forward in graph mode, mamba_host_pos for the
host-side position. Only the entry point differs: ThinkerGraph wraps ``engine.wrapper.model.stt_model.forward``; here a
small shim plays that role (``forward(input_embeds, cache) -> {"h": last_hidden_state}``) on Nano's backbone. A
multi-position call on a converted cache is not supported by the worker (it feeds single positions when the graph is
on). Correctness is checked by the caller (tool decisions / calls and reply tokens vs eager).
"""

from __future__ import annotations

from types import SimpleNamespace


class _Shim:
    def __init__(self, llm):
        self.llm = llm

    def forward(self, input_embeds, cache=None, **kw):
        h = self.llm(inputs_embeds=input_embeds, cache_params=cache, use_cache=True).last_hidden_state
        return {"h": h, "cache": cache}


class NanoGraph:  # pragma: no cover - GPU
    def __init__(self, llm, lmax: int = 4096):
        from mynah_v2.model import perf
        from mynah_v2.model.thinker_graph import ThinkerGraph
        self.shim = _Shim(llm)
        eng = SimpleNamespace(wrapper=SimpleNamespace(model=SimpleNamespace(stt_model=self.shim)), llm_cache_info=True)
        self.host_pos = perf.enable("mamba_host_pos", eng)
        self.tg = ThinkerGraph(eng, lmax=lmax)

    def step(self, e, cache):
        """One decode position (e: [1, 1, H]) -> last hidden [1, 1, H]."""
        return self.shim.forward(e, cache=cache)["h"]

    def stats(self) -> dict:
        return dict(self.tg.stats)
