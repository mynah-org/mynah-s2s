"""The talker's frame step for every language of one family at once, with few kernel launches and CUDA graphs.

``pocket_own.step_many`` already shares the backbone across languages (one grouped kernel per linear), but the
small per-language parts (norms, input projection, flow head, EOS) still run once per language, and every op is a
Python-launched kernel: with 8 languages a frame is hundreds of launches, so the talker is bound by the CPU and the
GIL it shares with the ears and the LLM thread, not by the GPU.

``CrossStep`` keeps a **language table** per slot pool (one model family):

* every small per-language tensor (layer norms, ``bos_emb``, ``input_linear``, ``out_norm``, ``out_eos``, the flow
  head, ``emb_mean`` / ``emb_std``) is copied into slot ``g`` of a stacked tensor ``[G, ...]`` the first time the
  language is stepped; a row reads its language's parameters by index (``gid``), so each op is ONE launch for all
  rows whatever their languages;
* the backbone linears use pocket_swap's grouped kernel in table mode: per slot, pointers to the language's delta
  codes and scales (device arrays), and a flag for the base language (no delta);
* the whole FlowLM part of a frame (input -> latent, EOS) is captured as a CUDA graph per row bucket (2 / 4 / 8 /
  16 rows); padding rows run on a scratch KV slot reserved from the pool. The noise is drawn outside the graph,
  so per-row generators still work.

Not bit-identical to each language's own step (layer norms with a gathered affine, batched small matmuls): the GPU
test compares it with the per-language step (small difference), checks graph == eager exactly, and has the ears
transcribe what it says. Mimi decoding stays as in ``step_many`` (one batched call, eager).
"""

from __future__ import annotations

import os
import weakref

import torch
import torch.nn.functional as F

from . import pocket_own as P

GRAPHS = os.environ.get("POCKET_CROSS_GRAPHS", "1") != "0"
BUCKETS = (2, 4, 8, 16)
COUNTS = {"cross": 0, "fallback": 0}          # frames stepped by CrossStep / handed to step_many (lab reports)
_ALL: "weakref.WeakSet" = weakref.WeakSet()   # every CrossStep (report())


def small_keys(m) -> list:
    """The per-language tensors the frame step reads besides the backbone linears and Mimi."""
    ks = ["flow_lm.bos_emb", "flow_lm.input_linear.weight", "flow_lm.out_norm.weight", "flow_lm.out_norm.bias",
          "flow_lm.out_eos.weight", "flow_lm.out_eos.bias", "flow_lm.emb_mean", "flow_lm.emb_std"]
    for li in range(m.n_layers):
        for n in ("norm1", "norm2"):
            ks += [f"flow_lm.transformer.layers.{li}.{n}.weight", f"flow_lm.transformer.layers.{li}.{n}.bias"]
    ks += sorted(k for k in m.W if k.startswith("flow_lm.flow_net."))
    return ks


class _Terms:
    """One backbone linear in table mode: per slot, delta code / scale pointers (what fused_linear_grouped reads)."""

    def __init__(self, prec, base, G, dev):
        self.prec, self.base = prec, base
        self.qp = torch.zeros(G, dtype=torch.int64, device=dev)
        self.sp = torch.zeros(G, dtype=torch.int64, device=dev)
        self.sqn = self.ssn = 0


class CrossStep:
    def __init__(self, pool, model, G: int = 16, graphs: bool = GRAPHS):
        self.pool, self.G = pool, G
        self.base = model.base
        self.dev = model.dev
        self.n_layers, self.d, self.nh, self.ldim = model.n_layers, model.d, model.nh, model.ldim
        self.n_res = model.n_res
        self.keys = small_keys(model)
        self.T = {k: torch.zeros((G, *model.W[k].shape), dtype=model.W[k].dtype, device=self.dev) for k in self.keys}
        self.lin_keys = list(model.QB) if model.QB else [k for k in self.base.terms]
        self.terms = {}
        self.prec = None
        self.has = torch.zeros(G, dtype=torch.int32, device=self.dev)
        self.owner: list = [None] * G            # weakref to the model in each slot
        self.last_use = [0] * G
        self.tick = 0
        self.scratch = pool.free.pop()           # padding rows of a graph bucket write their KV here
        self.graphs = graphs and self.dev.type == "cuda"
        self.g: dict = {}                         # bucket -> (graph, static inputs, static outputs)
        self.mempool = None
        self.fused, _ = P._pocket_swap()
        self.stats = {"captures": 0, "replays": 0, "eager": 0, "assigns": 0}
        _ALL.add(self)

    # ------------------------------------------------------------------ the language table
    def fits(self, m) -> bool:
        """Same shared base backbone, same architecture and small tensors, same delta precision."""
        return (m.base is self.base and (m.n_layers, m.d, m.nh, m.ldim, m.n_res) ==
                (self.n_layers, self.d, self.nh, self.ldim, self.n_res) and
                all(k in m.W and m.W[k].shape == self.T[k].shape[1:] for k in self.keys) and
                (not m.QB or self.prec in (None, next(iter(m.Q.values()))[0])))

    def slot(self, m) -> int:
        for g, r in enumerate(self.owner):
            if r is not None and r() is m:
                return g
        live = [g for g, r in enumerate(self.owner) if r is not None and r() is not None]
        free = [g for g in range(self.G) if g not in live]
        g = free[0] if free else min(live, key=lambda i: self.last_use[i])   # the least recently stepped
        with torch.no_grad():
            for k in self.keys:
                self.T[k][g].copy_(m.W[k])
            for k in self.lin_keys:
                t = self.terms.get(k)
                if t is None:
                    t = self.terms[k] = _Terms(None, self.base.terms[k], self.G, self.dev)
                if m.QB:
                    prec, q, sc, _ = m.Q[k]
                    if t.prec != prec:            # the kernel's precision is compiled into the captured graphs
                        self.g.clear()
                    self.prec = t.prec = prec
                    sc = sc if sc is not None else q
                    t.qp[g], t.sp[g] = q.data_ptr(), sc.data_ptr()
                    t.sqn, t.ssn = q.stride(0), sc.stride(0)
            self.has[g] = 1 if m.QB else 0
        self.owner[g] = weakref.ref(m)
        self.stats["assigns"] += 1
        return g

    # ------------------------------------------------------------------ the frame (pure tensor ops: capturable)
    def _ln(self, x, key, gid):
        y = F.layer_norm(x.float(), (x.shape[-1],), None, None, 1e-5)
        return (y * self.T[key + ".weight"][gid][:, None].float() + self.T[key + ".bias"][gid][:, None].float()
                ).to(x.dtype)

    def _blin(self, x, key, gid, bias=True):
        """Per-row linear with the row's language weights: x [B, in] -> [B, out]."""
        y = torch.bmm(self.T[key + ".weight"][gid], x[:, :, None])[:, :, 0]
        return y + self.T[key + ".bias"][gid] if bias else y

    def _flow_net(self, c, x, gid):
        p = "flow_lm.flow_net"
        T = self.T
        x = self._blin(x, p + ".input_proj", gid)
        y = self._blin(c, p + ".cond_embed", gid)
        temb = 0
        for i, tval in enumerate((0.0, 1.0)):
            q = f"{p}.time_embed.{i}"
            args = tval * T[q + ".freqs"][gid]
            e = torch.cat([torch.cos(args), torch.sin(args)], -1)
            e = self._blin(F.silu(self._blin(e, q + ".mlp.0", gid)), q + ".mlp.2", gid)
            var = 1e-5 + e.var(dim=-1, keepdim=True)
            e = e * (T[q + ".mlp.3.alpha"][gid].reshape(e.shape[0], -1) * torch.rsqrt(var))
            temb = temb + e
        y = y + temb / 2
        for i in range(self.n_res):
            q = f"{p}.res_blocks.{i}"
            sh, sc, g = self._blin(F.silu(y), q + ".adaLN_modulation.1", gid).chunk(3, -1)
            mean = x.mean(-1, keepdim=True)
            var = x.var(-1, unbiased=False, keepdim=True)
            h = (x - mean) / torch.sqrt(var + 1e-6) * T[q + ".in_ln.weight"][gid] + T[q + ".in_ln.bias"][gid]
            h = h * (1 + sc) + sh
            h = self._blin(F.silu(self._blin(h, q + ".mlp.0", gid)), q + ".mlp.2", gid)
            x = x + g * h
        q = f"{p}.final_layer"
        sh, sc = self._blin(F.silu(y), q + ".adaLN_modulation.1", gid).chunk(2, -1)
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        x = (x - mean) / torch.sqrt(var + 1e-6)
        x = x * (1 + sc) + sh
        return self._blin(x, q + ".linear", gid)

    def _core(self, lat_in, gid, slots, offs, r0, rc, noise):
        """lat_in [B, ldim] (NaN = BOS), gid / slots / offs [B], r0 / rc [G], noise [B, ldim]
        -> (latent [B, ldim], de-normalised latent [B, ldim], eos [B] bool). No host sync."""
        from ..llm.qwen_own.kernels import decode_attention
        T, B, d, nh = self.T, lat_in.shape[0], self.d, self.nh
        kv = self.pool.kv
        bos = T["flow_lm.bos_emb"][gid]
        seq = torch.where(torch.isnan(lat_in), bos, lat_in)
        x = self._blin(seq, "flow_lm.input_linear", gid, bias=False).to(torch.bfloat16)[:, None]   # [B, 1, d]
        fz = self.fused

        def lin(h, key):
            t = self.terms[key]
            return fz.fused_linear_grouped(h.reshape(B, -1).to(torch.bfloat16), t.base[1].shape[0], t, r0, rc, 128,
                                           has=self.has).view(B, 1, -1)

        for li in range(self.n_layers):
            p = f"flow_lm.transformer.layers.{li}"
            proj = lin(self._ln(x, p + ".norm1", gid), p + ".self_attn.in_proj.weight")
            q = proj[..., :d].view(B, 1, nh, d // nh)
            k, v = proj[..., d:].view(B, 1, 2, nh, d // nh).unbind(2)
            q, k = P._rope(q, k, offs)
            kv[li, 0, slots, offs] = k[:, 0]
            kv[li, 1, slots, offs] = v[:, 0]
            o = decode_attention(q[:, 0], kv[li, 0].permute(0, 2, 1, 3), kv[li, 1].permute(0, 2, 1, 3),
                                 slots, offs + 1)[:, None]
            x = x + lin(o.reshape(B, 1, d), p + ".self_attn.out_proj.weight")
            h = self._ln(x, p + ".norm2", gid)
            x = x + lin(F.gelu(lin(h, p + ".linear1.weight"), approximate="tanh"), p + ".linear2.weight")
        y = self._ln(x.float(), "flow_lm.out_norm", gid)[:, -1]
        eos = (self._blin(y, "flow_lm.out_eos", gid) > P.EOS_THR)[:, 0]
        lat = noise + self._flow_net(y, noise, gid)
        lat_d = lat * T["flow_lm.emb_std"][gid] + T["flow_lm.emb_mean"][gid]
        return lat, lat_d, eos

    # ------------------------------------------------------------------ graphs
    def _static(self, n):
        dev = self.dev
        return {"lat_in": torch.zeros(n, self.ldim, device=dev), "gid": torch.zeros(n, dtype=torch.long, device=dev),
                "slots": torch.full((n,), self.scratch, dtype=torch.long, device=dev),
                "offs": torch.zeros(n, dtype=torch.long, device=dev),
                "r0": torch.zeros(self.G, dtype=torch.int32, device=dev),
                "rc": torch.zeros(self.G, dtype=torch.int32, device=dev),
                "noise": torch.zeros(n, self.ldim, device=dev)}

    def _graph(self, n):
        hit = self.g.get(n)
        if hit is not None:
            return hit
        st = self._static(n)
        s = torch.cuda.Stream(device=self.dev)
        s.wait_stream(torch.cuda.current_stream(self.dev))
        with torch.cuda.stream(s):
            for _ in range(2):                    # warm-up (allocator, Triton autotune) on the scratch slot
                self._core(**st)
        torch.cuda.current_stream(self.dev).wait_stream(s)
        if self.mempool is None:
            self.mempool = torch.cuda.graph_pool_handle()
        g = torch.cuda.CUDAGraph()
        # thread_local: the LLM thread, the language loader and the ears keep using the GPU while this captures
        with torch.cuda.graph(g, pool=self.mempool, capture_error_mode="thread_local"):
            out = self._core(**st)
        self.g[n] = (g, st, out)
        self.stats["captures"] += 1
        return self.g[n]

    # ------------------------------------------------------------------ one frame
    @torch.no_grad()
    def step(self, groups: list) -> dict:
        """[(PocketBatch, keys)] of languages in this family -> {key: audio (1920,) | None}, like step_many."""
        rows, order = [], []
        for b, ks in groups:
            if not self.fits(b.m):
                raise ValueError(f"{b.m.lang}: not in this table's family / precision")
            g = self.slot(b.m)
            self.tick += 1
            self.last_use[g] = self.tick
            for k in ks:
                rows.append((g, b, k, b.rows[k]))
        rows.sort(key=lambda t: t[0])             # grouped kernel: rows sorted by slot
        B = len(rows)
        counts = [0] * self.G
        for g, *_ in rows:
            counts[g] += 1
        if B == 0:
            return {}
        r0 = [sum(counts[:g]) for g in range(self.G)]
        dev = self.dev
        lat_in = torch.cat([r["x"] for *_, r in rows], 0)
        gens = [r["gen"] for *_, r in rows]
        if all(x is not None for x in gens):
            noise = torch.cat([rows[i][1].m._noise(1, gens[i]) for i in range(B)], 0)
        else:
            noise = rows[0][1].m._noise(B)
        ints = torch.tensor([g for g, *_ in rows] + [r["slot"] for *_, r in rows] + [r["kv"].off for *_, r in rows],
                            dtype=torch.long).to(dev, non_blocking=True)
        grc = torch.tensor(r0 + counts, dtype=torch.int32).to(dev, non_blocking=True)
        n = next((x for x in BUCKETS if x >= B), None) if self.graphs else None
        if n is not None and n not in self.g:
            try:
                self._graph(n)
            except Exception as e:  # noqa: BLE001 - a failed capture: this table steps eagerly from now on
                import traceback
                traceback.print_exc()
                torch.cuda.synchronize(self.dev)
                self.graphs, n = False, None
                self.stats["capture_failed"] = repr(e)[:200]
        if n is not None:
            graph, st, out = self.g[n]
            st["lat_in"][:B].copy_(lat_in)
            st["noise"][:B].copy_(noise)
            st["gid"][:B].copy_(ints[:B])
            st["slots"][:B].copy_(ints[B:2 * B])
            st["offs"][:B].copy_(ints[2 * B:])
            if B < n:
                st["gid"][B:].zero_()
                st["slots"][B:].fill_(self.scratch)
                st["offs"][B:].zero_()
                st["lat_in"][B:].zero_()
            st["r0"].copy_(grc[:self.G])
            st["rc"].copy_(grc[self.G:])
            graph.replay()
            self.stats["replays"] += 1
            lat, lat_d, eos = (t[:B] for t in out)
        else:
            lat, lat_d, eos = self._core(lat_in, ints[:B], ints[B:2 * B], ints[2 * B:], grc[:self.G], grc[self.G:],
                                         noise)
            self.stats["eager"] += 1
        for *_, r in rows:
            r["kv"].off += 1
        eos_l = P.host(eos)                       # the one host sync per frame
        out_, dec = {}, []
        for i, (g, b, k, r) in enumerate(rows):
            if eos_l[i] and r["eos"] is None and r["step"] >= P.MIN_FRAMES_BEFORE_EOS:
                r["eos"] = r["step"]
            if (r["eos"] is not None and r["step"] >= r["eos"] + r["after"]) or r["step"] >= r["max_len"]:
                r["done"] = True
                out_[k] = None
                continue
            r["step"] += 1
            r["x"] = lat[i:i + 1].clone()
            dec.append((k, r, lat_d[i:i + 1]))
        if dec:
            b0 = rows[0][1]
            idx = torch.tensor([r["slot"] for _, r, _ in dec], device=dev)
            moff = torch.tensor([r["moff"] for _, r, _ in dec], device=dev)
            wav, nn = b0._mimi_decode(torch.cat([l_ for _, _, l_ in dec], 0)[:, None], idx, moff)
            for i, (k, r, _) in enumerate(dec):
                audio = wav[i, 0]
                r["moff"] += nn
                if r["fade"]:
                    m = min(P.SR // 200, audio.shape[-1])
                    audio[:m] *= torch.linspace(0, 1, P.SR // 200, device=dev)[:m]
                    r["fade"] = False
                out_[k] = audio
        return out_


def cross_step(groups: list) -> dict:
    """The talker's multi-language frame: a CrossStep per slot pool (created on first use), falling back to
    ``pocket_own.step_many`` when the languages cannot share a table (other precision or family, > 16 rows)."""
    pool = groups[0][0].pool
    cs = getattr(pool, "cross", None)
    if cs is None:
        cs = pool.cross = CrossStep(pool, groups[0][0].m)
    B = sum(len(ks) for _, ks in groups)
    if B > BUCKETS[-1] or len(groups) > cs.G or not all(cs.fits(b.m) for b, _ in groups):
        COUNTS["fallback"] += 1
        return P.step_many(groups)
    COUNTS["cross"] += 1
    return cs.step(groups)


def report() -> dict:
    """Frames per path and, per CrossStep, its graph captures / replays / eager frames / table assignments."""
    out = dict(COUNTS)
    for o in list(_ALL):
        for k, v in o.stats.items():
            out[k] = v if isinstance(v, str) else out.get(k, 0) + v
    return out
