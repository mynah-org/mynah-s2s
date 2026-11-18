"""Qwen3.5 / Qwen3.6 text model in plain torch (no serving framework): hybrid gated DeltaNet + gated full attention,
dense MLP or top-k MoE with a shared expert. Batched decode across sessions ("slots"), each slot with its own KV cache
(full-attention layers) and DeltaNet state (conv window + fp32 recurrent state).

Numerics follow the HF transformers reference (modeling_qwen3_5*.py, torch fallback paths): RMSNorm with (1 + w),
norm-then-gate RMSNorm in DeltaNet, l2-normalised q/k, fp32 delta-rule state, partial rotary (first `rotary_dim`
dims, rotate-half), output gate sigmoid on attention, softmax router in fp32 with renormalised top-k.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .config import QwenCfg
from .kernels import awq_expert_gemv
from .quant import QLinear, awq_dequant


# ---------------------------------------------------------------------------------------------------------------
# small ops
def rms_norm(x, w, eps):
    """Qwen3.5 RMSNorm: (x / rms(x)) * (1 + w), computed in fp32, cast back."""
    y = x.float()
    y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + eps)
    return (y * (1.0 + w.float())).to(x.dtype)


def gated_rms_norm(x, gate, w, eps):
    """DeltaNet output norm: normalise, cast, scale by w, then multiply by silu(gate) in fp32."""
    dt = x.dtype
    y = x.float()
    y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + eps)
    y = w * y.to(dt)
    return (y * F.silu(gate.float())).to(dt)


def l2norm(x, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)


def rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


class Rope:
    def __init__(self, cfg: QwenCfg, device):
        d = cfg.rotary_dim
        self.inv = 1.0 / (cfg.rope_theta ** (torch.arange(0, d, 2, dtype=torch.int64, device=device).float() / d))
        self.d = d

    def cos_sin(self, pos: torch.Tensor, dtype):
        """pos [B, T] int -> cos, sin [B, T, d] (text-only positions: the interleaved mRoPE reduces to plain RoPE)."""
        f = pos.float()[..., None] * self.inv
        emb = torch.cat((f, f), dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)

    def apply(self, x, cos, sin):
        """x [B, H, T, hd]; cos/sin [B, T, d]."""
        c, s = cos[:, None], sin[:, None]
        xr, xp = x[..., :self.d], x[..., self.d:]
        return torch.cat((xr * c + rotate_half(xr) * s, xp), dim=-1)


# ---------------------------------------------------------------------------------------------------------------
# gated delta rule
def chunk_delta_rule(q, k, v, g, beta, state, chunk=64):
    """Prefill. q, k [B, T, H, dk] (already l2-normalised), v [B, T, H, dv], g, beta [B, T, H]; state [B, H, dk, dv]
    fp32 (updated in place is NOT done: the new state is returned). Returns (out [B, T, H, dv] fp32, state)."""
    q, k, v, beta, g = [x.transpose(1, 2).contiguous().float() for x in (q, k, v, beta, g)]
    B, H, T, dk = k.shape
    dv = v.shape[-1]
    pad = (chunk - T % chunk) % chunk
    q, k, v = (F.pad(x, (0, 0, 0, pad)) for x in (q, k, v))
    beta, g = F.pad(beta, (0, pad)), F.pad(g, (0, pad))
    n = (T + pad) // chunk
    q = q * dk ** -0.5
    vb, kb = v * beta[..., None], k * beta[..., None]
    q, k, v, kb, vb = (x.reshape(B, H, n, chunk, x.shape[-1]) for x in (q, k, v, kb, vb))
    g = g.reshape(B, H, n, chunk).cumsum(-1)
    decay = (g[..., :, None] - g[..., None, :]).tril().exp().tril()                    # [B, H, n, C, C]
    eye = torch.eye(chunk, device=q.device, dtype=q.dtype)
    a = (kb @ k.transpose(-1, -2)) * decay
    a = a.tril(-1)                                                                     # strictly lower
    tri = torch.linalg.solve_triangular(eye + a, eye.expand_as(a), upper=False)       # (I + A)^-1
    vv = tri @ vb
    kcum = tri @ (kb * g.exp()[..., None])
    out = torch.empty_like(vv)
    s = state.float()
    for i in range(n):
        qi, ki = q[:, :, i], k[:, :, i]
        att = (qi @ ki.transpose(-1, -2)) * decay[:, :, i]
        att = att.tril()
        vnew = vv[:, :, i] - kcum[:, :, i] @ s
        out[:, :, i] = (qi * g[:, :, i, :, None].exp()) @ s + att @ vnew
        gl = g[:, :, i, -1]
        s = s * gl[..., None, None].exp() + (ki * (gl[..., None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ vnew
    out = out.reshape(B, H, n * chunk, dv)[:, :, :T].transpose(1, 2)
    return out, s


def step_delta_rule(q, k, v, g, beta, state):
    """Decode, one token. q, k [B, H, dk] (l2-normalised), v [B, H, dv], g, beta [B, H]; state [B, H, dk, dv] fp32.
    Returns (out [B, H, dv] fp32, new state)."""
    q, k, v, g, beta = (x.float() for x in (q, k, v, g, beta))
    q = q * q.shape[-1] ** -0.5
    s = state * g.exp()[..., None, None]
    kv = (s * k[..., None]).sum(-2)
    delta = (v - kv) * beta[..., None]
    s = s + k[..., None] * delta[..., None, :]
    out = (s * q[..., None]).sum(-2)
    return out, s


# ---------------------------------------------------------------------------------------------------------------
class Model:
    """Weights + forward. `W` maps HF tensor names (text model, prefix stripped) to tensors already on the device."""

    def __init__(self, cfg: QwenCfg, W: dict, device, mode="bf16", group=128, awq_group=None):
        self.cfg, self.device = cfg, device
        self.use_kernels = device.type == "cuda"
        self.rope = Rope(cfg, device)
        L = lambda n: QLinear(W.pop(n), mode=mode, group=group)          # noqa: E731
        self.embed = W.pop("embed_tokens.weight").to(torch.bfloat16)
        self.norm = W.pop("norm.weight")
        self.lm_head = QLinear(self.embed if cfg.tie_embeddings or "lm_head.weight" not in W else W.pop("lm_head.weight"),
                               mode="bf16")
        self.layers = []
        for i, kind in enumerate(cfg.layer_types):
            p = f"layers.{i}."
            d = {"kind": kind, "ln1": W.pop(p + "input_layernorm.weight"), "ln2": W.pop(p + "post_attention_layernorm.weight")}
            if kind == "linear_attention":
                a = p + "linear_attn."
                d.update(qkv=L(a + "in_proj_qkv.weight"), z=L(a + "in_proj_z.weight"),
                         b=QLinear(W.pop(a + "in_proj_b.weight")), a=QLinear(W.pop(a + "in_proj_a.weight")),
                         conv=W.pop(a + "conv1d.weight"), A_log=W.pop(a + "A_log"), dt_bias=W.pop(a + "dt_bias"),
                         gnorm=W.pop(a + "norm.weight"), out=L(a + "out_proj.weight"))
            else:
                a = p + "self_attn."
                d.update(q=L(a + "q_proj.weight"), k=L(a + "k_proj.weight"), v=L(a + "v_proj.weight"),
                         o=L(a + "o_proj.weight"), qn=W.pop(a + "q_norm.weight"), kn=W.pop(a + "k_norm.weight"))
            m = p + "mlp."
            if cfg.moe:
                d.update(router=W.pop(m + "gate.weight"), sh_gate=W.pop(m + "shared_expert_gate.weight"),
                         sh_g=L(m + "shared_expert.gate_proj.weight"), sh_u=L(m + "shared_expert.up_proj.weight"),
                         sh_d=L(m + "shared_expert.down_proj.weight"))
                d["experts"] = {k: W.pop(m + "experts." + k) for k in list(_expert_keys(W, m))}
                d["awq_group"] = awq_group
            else:
                d.update(g=L(m + "gate_proj.weight"), u=L(m + "up_proj.weight"), dn=L(m + "down_proj.weight"))
            self.layers.append(d)
        self.lin_idx = [i for i, k in enumerate(cfg.layer_types) if k == "linear_attention"]
        self.att_idx = [i for i, k in enumerate(cfg.layer_types) if k != "linear_attention"]

    # ---- state ----
    def new_state(self, slots: int, max_len: int):
        c = self.cfg
        conv_dim = 2 * c.lin_k_heads * c.lin_k_dim + c.lin_v_heads * c.lin_v_dim
        dev = self.device
        return {
            "conv": {i: torch.zeros(slots, conv_dim, c.conv_k - 1, dtype=torch.bfloat16, device=dev) for i in self.lin_idx},
            "rec": {i: torch.zeros(slots, c.lin_v_heads, c.lin_k_dim, c.lin_v_dim, dtype=torch.float32, device=dev)
                    for i in self.lin_idx},
            "k": {i: torch.zeros(slots, c.kv_heads, max_len, c.head_dim, dtype=torch.bfloat16, device=dev) for i in self.att_idx},
            "v": {i: torch.zeros(slots, c.kv_heads, max_len, c.head_dim, dtype=torch.bfloat16, device=dev) for i in self.att_idx},
            "max_len": max_len,
        }

    # ---- blocks ----
    def _deltanet(self, d, x, st, slots, prefill):
        c = self.cfg
        B, T, _ = x.shape
        mixed = d["qkv"](x).transpose(1, 2)                                     # [B, C, T]
        z = d["z"](x).reshape(B, T, c.lin_v_heads, c.lin_v_dim)
        bb, aa = d["b"](x), d["a"](x)
        conv_state = st["conv"][d["i"]][slots]                                  # [B, C, K-1]
        full = torch.cat([conv_state, mixed.to(conv_state.dtype)], dim=-1)
        st["conv"][d["i"]][slots] = full[:, :, -(c.conv_k - 1):]
        w = d["conv"]                                                           # [C, 1, K]
        mixed = F.silu(F.conv1d(full.to(w.dtype), w, groups=w.shape[0]))[:, :, -T:].to(x.dtype)
        mixed = mixed.transpose(1, 2)
        kd = c.lin_k_heads * c.lin_k_dim
        q, k, v = torch.split(mixed, [kd, kd, c.lin_v_heads * c.lin_v_dim], dim=-1)
        q = q.reshape(B, T, c.lin_k_heads, c.lin_k_dim)
        k = k.reshape(B, T, c.lin_k_heads, c.lin_k_dim)
        v = v.reshape(B, T, c.lin_v_heads, c.lin_v_dim)
        beta = bb.sigmoid()
        g = -d["A_log"].float().exp() * F.softplus(aa.float() + d["dt_bias"])
        rep = c.lin_v_heads // c.lin_k_heads
        if rep > 1:
            q, k = q.repeat_interleave(rep, dim=2), k.repeat_interleave(rep, dim=2)
        q, k = l2norm(q), l2norm(k)
        state = st["rec"][d["i"]][slots]
        if prefill:
            out, s = chunk_delta_rule(q, k, v, g, beta, state)
        else:
            o, s = step_delta_rule(q[:, 0], k[:, 0], v[:, 0], g[:, 0], beta[:, 0], state)
            out = o[:, None]
        st["rec"][d["i"]][slots] = s
        out = out.to(x.dtype).reshape(-1, c.lin_v_dim)
        out = gated_rms_norm(out, z.reshape(-1, c.lin_v_dim), d["gnorm"], c.eps).reshape(B, T, -1)
        return d["out"](out)

    def _attention(self, d, x, st, slots, pos, lens, prefill, kv_len=None):
        """pos [B, T] absolute positions; lens [B] valid cache length after this step."""
        c = self.cfg
        B, T, _ = x.shape
        qg = d["q"](x).view(B, T, c.heads, 2 * c.head_dim)
        q, gate = qg[..., :c.head_dim], qg[..., c.head_dim:]
        gate = gate.reshape(B, T, -1)
        q = rms_norm(q, d["qn"], c.eps).transpose(1, 2)
        k = rms_norm(d["k"](x).view(B, T, c.kv_heads, c.head_dim), d["kn"], c.eps).transpose(1, 2)
        v = d["v"](x).view(B, T, c.kv_heads, c.head_dim).transpose(1, 2)
        cos, sin = self.rope.cos_sin(pos, x.dtype)
        q, k = self.rope.apply(q, cos, sin), self.rope.apply(k, cos, sin)
        K, V = st["k"][d["i"]], st["v"][d["i"]]
        if prefill:                                                            # one slot, positions p0..p0+T-1
            s, p0 = int(slots[0]), int(pos[0, 0])
            K[s, :, p0:p0 + T] = k[0]
            V[s, :, p0:p0 + T] = v[0]
            kk, vv = K[s:s + 1, :, :p0 + T], V[s:s + 1, :, :p0 + T]
            if p0 == 0:
                o = F.scaled_dot_product_attention(q, kk, vv, is_causal=True, enable_gqa=True)
            else:
                m = torch.arange(p0 + T, device=x.device)[None, :] <= (p0 + torch.arange(T, device=x.device))[:, None]
                o = F.scaled_dot_product_attention(q, kk, vv, attn_mask=m[None, None], enable_gqa=True)
        else:                                                                  # one token per slot
            K[slots, :, pos[:, 0]] = k[:, :, 0]
            V[slots, :, pos[:, 0]] = v[:, :, 0]
            L = kv_len or K.shape[2]                                             # a static bucket >= every length
            kk, vv = K[slots, :, :L], V[slots, :, :L]
            m = torch.arange(L, device=x.device)[None, :] < lens[:, None]
            o = F.scaled_dot_product_attention(q, kk, vv, attn_mask=m[:, None, None, :], enable_gqa=True)
        o = o.transpose(1, 2).reshape(B, T, -1)
        o = o * torch.sigmoid(gate)
        return d["o"](o)

    def _mlp(self, d, x):
        c = self.cfg
        if not c.moe:
            return d["dn"](F.silu(d["g"](x)) * d["u"](x))
        B, T, H = x.shape
        h = x.reshape(-1, H)
        sh = d["sh_d"](F.silu(d["sh_g"](h)) * d["sh_u"](h)) * torch.sigmoid(F.linear(h, d["sh_gate"]))
        logits = F.linear(h, d["router"])
        p = torch.softmax(logits, dim=-1, dtype=torch.float)
        topv, topi = torch.topk(p, c.top_k, dim=-1)
        topv = (topv / topv.sum(-1, keepdim=True)).to(h.dtype)
        out = torch.zeros_like(h)
        ex = d["experts"]
        grp = d["awq_group"]
        if h.shape[0] * c.top_k <= 256:                                       # decode (graph-safe: no host sync)
            # few tokens (decode): gather the selected experts' weights and run them as a batched matmul
            idx = topi.reshape(-1)                                             # [N*k]
            if "gate_proj.qweight" in ex and self.use_kernels:                 # AWQ: fused dequant GEMV (Triton)
                hh = h.repeat_interleave(c.top_k, dim=0)
                gv = lambda nm, xx: awq_expert_gemv(xx, ex[nm + ".qweight"], ex[nm + ".qzeros"],   # noqa: E731
                                                    ex[nm + ".scales"], idx, grp)
                a = F.silu(gv("gate_proj", hh)) * gv("up_proj", hh)
                y = gv("down_proj", a) * topv.reshape(-1, 1)
                return (y.reshape(-1, c.top_k, H).sum(1) + sh).reshape(B, T, H)
            wg = _expert_weight(ex, "gate_proj", idx, grp)                     # [N*k, I, H]
            wu = _expert_weight(ex, "up_proj", idx, grp)
            wd = _expert_weight(ex, "down_proj", idx, grp)                     # [N*k, H, I]
            hh = h.repeat_interleave(c.top_k, dim=0)[:, None, :]               # [N*k, 1, H]
            a = F.silu(hh @ wg.transpose(-1, -2)) * (hh @ wu.transpose(-1, -2))
            y = (a @ wd.transpose(-1, -2))[:, 0] * topv.reshape(-1, 1)
            out = y.reshape(-1, c.top_k, H).sum(1)
        else:
            # many tokens (prefill): group the tokens per expert, dequantise each hit expert once
            flat = topi.reshape(-1)
            order = torch.argsort(flat)
            counts = torch.bincount(flat, minlength=c.num_experts).tolist()
            tok = order // c.top_k
            start = 0
            for e, n in enumerate(counts):
                if n == 0:
                    continue
                sel = order[start:start + n]
                ti = tok[start:start + n]
                start += n
                e_t = torch.tensor([e], device=h.device)
                wg = _expert_weight(ex, "gate_proj", e_t, grp)[0]
                wu = _expert_weight(ex, "up_proj", e_t, grp)[0]
                wd = _expert_weight(ex, "down_proj", e_t, grp)[0]
                xs = h[ti]
                y = F.linear(F.silu(F.linear(xs, wg)) * F.linear(xs, wu), wd)
                out.index_add_(0, ti, y * topv.reshape(-1)[sel, None])
        return (out + sh).reshape(B, T, H)

    # ---- forward ----
    @torch.no_grad()
    def forward(self, ids, st, slots, pos, lens, prefill, kv_len=None):
        """ids [B, T]; slots [B] long; pos [B, T]; lens [B] (cache length after this call). Returns logits of the last
        position [B, V] (fp32)."""
        x = F.embedding(ids, self.embed)
        for i, d in enumerate(self.layers):
            d["i"] = i
            h = rms_norm(x, d["ln1"], self.cfg.eps)
            if d["kind"] == "linear_attention":
                h = self._deltanet(d, h, st, slots, prefill)
            else:
                h = self._attention(d, h, st, slots, pos, lens, prefill, kv_len)
            x = x + h
            x = x + self._mlp(d, rms_norm(x, d["ln2"], self.cfg.eps))
        x = rms_norm(x[:, -1], self.norm, self.cfg.eps)
        return self.lm_head(x).float()


def _expert_keys(W, m):
    pre = m + "experts."
    return {k[len(pre):] for k in W if k.startswith(pre)}


def _expert_weight(ex, name, idx, group):
    """Selected experts' weight [n, out, in] bf16. `ex` holds stacked tensors: either "<name>" bf16 [E, out, in]
    or AWQ "<name>.qweight/.qzeros/.scales" stacked over E."""
    if name in ex:
        return ex[name][idx]
    return awq_dequant(ex[name + ".qweight"][idx], ex[name + ".qzeros"][idx], ex[name + ".scales"][idx], group)
