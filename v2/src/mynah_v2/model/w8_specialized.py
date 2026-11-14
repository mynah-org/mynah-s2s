"""Q8 adapters for the non-standard INT4 inference paths.

The ordinary ``WOLinear`` covers dense Linear modules.  These small adapters
cover the remaining weight-only paths whose callers index selected matrices,
codebooks, depthwise filters, or recurrent gates directly.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from mynah_v2.model.wo_gemv import WOLinear, pack_w8


_W8_LSTM_KERNEL = None


def _w8_lstm_kernel():
    """Build the one-launch Q8 RNNT gate/activation kernel lazily on CUDA."""
    global _W8_LSTM_KERNEL
    if _W8_LSTM_KERNEL is not None:
        return _W8_LSTM_KERNEL
    import triton
    import triton.language as tl

    @triton.jit
    def fused(x, h, c, qih, sih, qhh, shh, bias, out, c_new, I, H, BLOCK_H: tl.constexpr,
              BLOCK_K: tl.constexpr):
        b = tl.program_id(0)
        j = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
        valid = j < H
        gate = tl.arange(0, 4)[:, None]
        n = gate * H + j[None, :]
        acc = tl.load(bias + n, mask=valid[None, :], other=0.0).to(tl.float32)
        for k0 in tl.range(0, I, BLOCK_K):
            k = k0 + tl.arange(0, BLOCK_K)
            mk = k < I
            w = tl.load(qih + n[:, :, None] * I + k[None, None, :],
                        mask=valid[None, :, None] & mk[None, None, :], other=0).to(tl.float32)
            xv = tl.load(x + b * I + k, mask=mk, other=0.0).to(tl.float32)
            acc += tl.sum(w * xv[None, None, :], axis=2) * tl.load(sih + n, mask=valid[None, :], other=0.0).to(tl.float32)
        for k0 in tl.range(0, H, BLOCK_K):
            k = k0 + tl.arange(0, BLOCK_K)
            mk = k < H
            w = tl.load(qhh + n[:, :, None] * H + k[None, None, :],
                        mask=valid[None, :, None] & mk[None, None, :], other=0).to(tl.float32)
            hv = tl.load(h + b * H + k, mask=mk, other=0.0).to(tl.float32)
            acc += tl.sum(w * hv[None, None, :], axis=2) * tl.load(shh + n, mask=valid[None, :], other=0.0).to(tl.float32)
        gate_idx = tl.arange(0, 4)[:, None]
        gi = tl.sum(tl.where(gate_idx == 0, acc, 0.0), axis=0)
        gf = tl.sum(tl.where(gate_idx == 1, acc, 0.0), axis=0)
        gg = tl.sum(tl.where(gate_idx == 2, acc, 0.0), axis=0)
        go = tl.sum(tl.where(gate_idx == 3, acc, 0.0), axis=0)
        fi = 1.0 / (1.0 + tl.exp(-gi))
        ff = 1.0 / (1.0 + tl.exp(-gf))
        fo = 1.0 / (1.0 + tl.exp(-go))
        gg = 2.0 / (1.0 + tl.exp(-2.0 * gg)) - 1.0
        old_c = tl.load(c + b * H + j, mask=valid, other=0.0).to(tl.float32)
        cn = ff * old_c + fi * gg
        hn = fo * (2.0 / (1.0 + tl.exp(-2.0 * cn)) - 1.0)
        tl.store(out + b * H + j, hn.to(out.dtype.element_ty), mask=valid)
        tl.store(c_new + b * H + j, cn.to(c_new.dtype.element_ty), mask=valid)

    _W8_LSTM_KERNEL = (fused, triton.cdiv)
    return _W8_LSTM_KERNEL


class W8PointwiseConv1d(nn.Module):
    """Q8 replacement for a dense, non-dilated Conv1d with kernel size 1."""

    def __init__(self, conv: nn.Conv1d):
        super().__init__()
        if not isinstance(conv, nn.Conv1d):
            raise TypeError("W8PointwiseConv1d expects nn.Conv1d")
        if (tuple(conv.kernel_size), tuple(conv.stride), tuple(conv.padding), tuple(conv.dilation), conv.groups) != (
            (1,), (1,), (0,), (1,), 1
        ):
            raise ValueError("W8PointwiseConv1d requires dense kernel_size=1, stride=1, padding=0")
        lin = nn.Linear(conv.in_channels, conv.out_channels, bias=conv.bias is not None,
                        device=conv.weight.device, dtype=conv.weight.dtype)
        with torch.no_grad():
            lin.weight.copy_(conv.weight[:, :, 0])
            if conv.bias is not None:
                lin.bias.copy_(conv.bias)
        self.linear = WOLinear(lin, "w8")
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != self.in_channels:
            raise RuntimeError(f"Q8 pointwise codec expects [B,{self.in_channels},T], got {tuple(x.shape)}")
        y = self.linear(x.transpose(1, 2).contiguous())
        return y.transpose(1, 2).contiguous()

    def nbytes(self) -> int:
        return self.linear.nbytes()


class Int8LSTM(nn.Module):
    """Q8 RNNT prediction LSTM using batched weight-only GEMV matrices."""

    def __init__(self, rnn_module: nn.Module, fused: bool = False):
        super().__init__()
        lstm = getattr(rnn_module, "lstm", None)
        if not isinstance(lstm, nn.LSTM) or lstm.proj_size or lstm.bidirectional:
            raise ValueError("only unidirectional unprojected LSTM is supported")
        self.input_size, self.hidden_size = lstm.input_size, lstm.hidden_size
        self.num_layers = lstm.num_layers
        self.fused = bool(fused)
        self.ih = nn.ModuleList()
        self.hh = nn.ModuleList()
        self.bias = nn.ParameterList()
        for layer in range(self.num_layers):
            wi = getattr(lstm, f"weight_ih_l{layer}").detach()
            wh = getattr(lstm, f"weight_hh_l{layer}").detach()
            li = nn.Linear(wi.shape[1], wi.shape[0], bias=False, device=wi.device, dtype=wi.dtype)
            lh = nn.Linear(wh.shape[1], wh.shape[0], bias=False, device=wh.device, dtype=wh.dtype)
            li.weight.data.copy_(wi)
            lh.weight.data.copy_(wh)
            self.ih.append(WOLinear(li, "w8"))
            self.hh.append(WOLinear(lh, "w8"))
            bi = getattr(lstm, f"bias_ih_l{layer}").detach()
            bh = getattr(lstm, f"bias_hh_l{layer}").detach()
            self.bias.append(nn.Parameter((bi + bh).contiguous(), requires_grad=False))

    def forward(self, x: torch.Tensor, state=None):
        if x.ndim != 3 or x.shape[-1] != self.input_size:
            raise ValueError(f"RNNT Q8 expects [T,B,{self.input_size}], got {tuple(x.shape)}")
        t, b, _ = x.shape
        if state is None:
            hs = [torch.zeros(b, self.hidden_size, dtype=x.dtype, device=x.device) for _ in range(self.num_layers)]
            cs = [torch.zeros_like(h) for h in hs]
        else:
            if len(state) != 2 or state[0].shape != (self.num_layers, b, self.hidden_size) \
                    or state[1].shape != (self.num_layers, b, self.hidden_size):
                raise ValueError("RNNT Q8 state shape does not match the current batch")
            hs = [state[0][i].contiguous() for i in range(self.num_layers)]
            cs = [state[1][i].contiguous() for i in range(self.num_layers)]
        outputs = x.new_empty((t, b, self.hidden_size))
        if x.device.type == "cuda" and self.fused and t and b:
            import triton

            kernel, cdiv = _w8_lstm_kernel()
            # h is read across all output gates, so each layer uses ping-pong
            # buffers instead of writing its recurrent input in place.  The
            # cell state is also double-buffered to keep graph/capture and
            # concurrent session ownership explicit.
            hbuf = [[torch.empty((b, self.hidden_size), dtype=x.dtype, device=x.device) for _ in range(2)]
                    for _ in range(self.num_layers)]
            cbuf = [[torch.empty((b, self.hidden_size), dtype=x.dtype, device=x.device) for _ in range(2)]
                    for _ in range(self.num_layers)]
            for ti in range(t):
                y = x[ti].contiguous()
                for li in range(self.num_layers):
                    slot = ti & 1
                    hnew, cnew = hbuf[li][slot], cbuf[li][slot]
                    ih, hh = self.ih[li], self.hh[li]
                    kernel[(b, cdiv(self.hidden_size, 128))](
                        y, hs[li], cs[li], ih.p_q, ih.p_scale, hh.p_q, hh.p_scale,
                        self.bias[li], hnew, cnew, self.input_size if li == 0 else self.hidden_size,
                        self.hidden_size, BLOCK_H=128, BLOCK_K=128,
                    )
                    hs[li], cs[li], y = hnew, cnew, hnew
                outputs[ti].copy_(y)
            return outputs, (torch.stack(hs), torch.stack(cs))
        for ti in range(t):
            y = x[ti]
            for li in range(self.num_layers):
                gates = self.ih[li](y) + self.hh[li](hs[li]) + self.bias[li].to(y.dtype)
                i, f, g, o = gates.chunk(4, -1)
                cs[li] = torch.sigmoid(f) * cs[li] + torch.sigmoid(i) * torch.tanh(g)
                hs[li] = torch.sigmoid(o) * torch.tanh(cs[li])
                y = hs[li]
            outputs[ti].copy_(y)
        return outputs, (torch.stack(hs), torch.stack(cs))

    def nbytes(self) -> int:
        return sum(m.nbytes() for m in list(self.ih) + list(self.hh))


class SelectedMoGW8(nn.Module):
    """Per-output-channel Q8 matrices with NeMo's selected-mixture contract."""

    def __init__(self, lin: nn.Linear, num_predictions: int, out_size: int):
        super().__init__()
        if lin.out_features != num_predictions * out_size:
            raise ValueError("MoG projection shape is not num_predictions * out_size")
        n, d, k = num_predictions, out_size, lin.in_features
        p = pack_w8(lin.weight.data.reshape(n * d, k))
        self.num_predictions, self.out_size, self.in_features = n, d, k
        self.register_buffer("q", p["q"].reshape(n, d, k).contiguous())
        self.register_buffer("scale", p["scale"].reshape(n, d).contiguous())
        self.register_buffer("bias", lin.bias.detach().clone() if lin.bias is not None else torch.empty(0),
                             persistent=False)

    @classmethod
    def from_tensor(cls, weight: torch.Tensor) -> "SelectedMoGW8":
        if weight.ndim != 3:
            raise ValueError("selected MoG weight must be [num_predictions,out_size,in_features]")
        out = cls.__new__(cls)
        nn.Module.__init__(out)
        n, d, k = weight.shape
        p = pack_w8(weight.reshape(n * d, k))
        out.num_predictions, out.out_size, out.in_features = n, d, k
        out.register_buffer("q", p["q"].reshape(n, d, k).contiguous())
        out.register_buffer("scale", p["scale"].reshape(n, d).contiguous())
        out.register_buffer("bias", torch.empty(0, device=weight.device), persistent=False)
        return out

    def selected(self, x: torch.Tensor, mixture_indices: torch.Tensor) -> torch.Tensor:
        shape = x.shape[:-1]
        x2 = x.reshape(-1, self.in_features).contiguous()
        ids = mixture_indices.reshape(-1).contiguous()
        if x2.shape[0] != ids.shape[0]:
            raise ValueError("MoG Q8 input and mixture index row counts differ")
        if x2.device.type != "cuda":
            q = self.q[ids].float() * self.scale[ids].float().unsqueeze(-1)
            y = torch.einsum("bk,bdk->bd", x2.float(), q)
            if self.bias.numel():
                y = y + self.bias.view(self.num_predictions, self.out_size)[ids].float()
            return y.to(x.dtype).reshape(*shape, self.out_size)
        import triton
        import triton.language as tl

        @triton.jit
        def kernel(xp, qp, sp, bp, ip, yp, B, D, K, HAS_BIAS: tl.constexpr,
                   BLOCK_D: tl.constexpr, BLOCK_K: tl.constexpr):
            bid = tl.program_id(0)
            pd = tl.program_id(1)
            d = pd * BLOCK_D + tl.arange(0, BLOCK_D)
            md = d < D
            idx = tl.load(ip + bid).to(tl.int32)
            acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
            scale = tl.load(sp + idx * D + d, mask=md, other=0.0).to(tl.float32)
            for k0 in tl.range(0, K, BLOCK_K):
                k = k0 + tl.arange(0, BLOCK_K)
                mk = k < K
                w = tl.load(qp + idx * D * K + d[:, None] * K + k[None, :],
                            mask=md[:, None] & mk[None, :], other=0).to(tl.float32)
                xv = tl.load(xp + bid * K + k, mask=mk, other=0.0).to(tl.float32)
                acc += tl.sum(w * xv[None, :], axis=1) * scale
            if HAS_BIAS:
                acc += tl.load(bp + idx * D + d, mask=md, other=0.0).to(tl.float32)
            tl.store(yp + bid * D + d, acc.to(yp.dtype.element_ty), mask=md)

        y = torch.empty((x2.shape[0], self.out_size), device=x2.device, dtype=x2.dtype)
        kernel[(x2.shape[0], triton.cdiv(self.out_size, 64))](
            x2, self.q, self.scale, self.bias, ids, y, x2.shape[0], self.out_size, self.in_features,
            HAS_BIAS=self.bias.numel() > 0, BLOCK_D=64, BLOCK_K=128,
        )
        return y.reshape(*shape, self.out_size)

    def nbytes(self) -> int:
        return self.q.numel() * self.q.element_size() + self.scale.numel() * self.scale.element_size()


class DepthwiseConv1dW8(nn.Module):
    """Per-channel symmetric Q8 replacement for streaming depthwise Conv1d."""

    def __init__(self, conv: nn.Conv1d):
        super().__init__()
        if not isinstance(conv, nn.Conv1d):
            raise TypeError("DepthwiseConv1dW8 expects nn.Conv1d")
        if not (conv.in_channels == conv.out_channels == conv.groups and conv.stride == (1,)
                and conv.dilation == (1,) and conv.padding == (0,)):
            raise ValueError("only dense stride-1 depthwise Conv1d is supported")
        w = conv.weight.detach().float().reshape(conv.in_channels, conv.kernel_size[0])
        scale = w.abs().amax(-1).clamp_min(1e-8) / 127
        q = torch.round(w / scale[:, None]).clamp(-127, 127).to(torch.int8)
        self.channels = conv.in_channels
        self.kernel_size = conv.kernel_size[0]
        self.register_buffer("q", q.contiguous())
        self.register_buffer("scale", scale.contiguous())
        self.bias = conv.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != self.channels:
            raise RuntimeError(f"depthwise Q8 expected [B,{self.channels},T], got {tuple(x.shape)}")
        if x.shape[-1] < self.kernel_size:
            x = F.pad(x, (self.kernel_size - 1, 0))
        if x.device.type != "cuda":
            w = self.q.float() * self.scale[:, None]
            return F.conv1d(x, w[:, None, :].to(x.dtype), self.bias, groups=self.channels)
        if not x.is_contiguous():
            x = x.contiguous()
        import triton
        import triton.language as tl

        @triton.jit
        def kernel(xp, qp, sp, bp, yp, C, T, O, K: tl.constexpr,
                   BLOCK_T: tl.constexpr, HAS_BIAS: tl.constexpr):
            pc = tl.program_id(0)
            pt = tl.program_id(1)
            c = pc % C
            b = pc // C
            t = pt * BLOCK_T + tl.arange(0, BLOCK_T)
            mt = t < O
            acc = tl.zeros((BLOCK_T,), dtype=tl.float32)
            sc = tl.load(sp + c).to(tl.float32)
            for k in tl.static_range(0, K):
                xv = tl.load(xp + (b * C + c) * T + t + k, mask=mt, other=0.0).to(tl.float32)
                w = tl.load(qp + c * K + k).to(tl.float32) * sc
                acc += xv * w
            if HAS_BIAS:
                acc += tl.load(bp + c).to(tl.float32)
            tl.store(yp + (b * C + c) * O + t, acc.to(yp.dtype.element_ty), mask=mt)

        out_len = x.shape[-1] - self.kernel_size + 1
        y = torch.empty((x.shape[0], self.channels, out_len), dtype=x.dtype, device=x.device)
        bias = self.bias if self.bias is not None else self.scale
        kernel[(x.shape[0] * self.channels, triton.cdiv(out_len, 128))](
            x, self.q, self.scale, bias, y, self.channels, x.shape[-1], out_len,
            K=self.kernel_size, BLOCK_T=128, HAS_BIAS=self.bias is not None,
        )
        return y

    def nbytes(self) -> int:
        return self.q.numel() * self.q.element_size() + self.scale.numel() * self.scale.element_size()


class RVQInt8Embedding(nn.Module):
    """Q8 RVQ gather/sum with the same padded-code contract as the Q4 path."""

    def __init__(self, rvq_embs: torch.Tensor):
        super().__init__()
        if rvq_embs.ndim != 3:
            raise ValueError(f"RVQ table must be [D,V,H], got {tuple(rvq_embs.shape)}")
        d, v, h = rvq_embs.shape
        real = pack_w8(rvq_embs.reshape(d * v, h))
        q = torch.cat([real["q"].reshape(d, v, h), torch.zeros(d, 1, h, dtype=real["q"].dtype,
                                                                device=rvq_embs.device)], dim=1)
        scale = torch.cat([real["scale"].reshape(d, v), torch.zeros(d, 1, dtype=real["scale"].dtype,
                                                                      device=rvq_embs.device)], dim=1)
        self.depth, self.vocab, self.hidden = d, v, h
        self.register_buffer("q", q.contiguous())
        self.register_buffer("scale", scale.contiguous())

    def forward(self, code: torch.Tensor) -> torch.Tensor:
        shape = code.shape[:-1]
        if code.shape[-1] != self.depth:
            raise ValueError(f"RVQ code depth {code.shape[-1]} != {self.depth}")
        flat = code.reshape(-1, self.depth).contiguous()
        if flat.device.type != "cuda":
            out = torch.zeros(flat.shape[0], self.hidden, dtype=torch.float32, device=flat.device)
            for i in range(self.depth):
                idx = flat[:, i]
                out += self.q[i, idx].float() * self.scale[i, idx, None].float()
            return out.reshape(*shape, self.hidden)
        import triton
        import triton.language as tl

        @triton.jit
        def kernel(cp, qp, sp, yp, P, D, V, H, BLOCK_H: tl.constexpr):
            pid = tl.program_id(0)
            ph = tl.program_id(1)
            h = ph * BLOCK_H + tl.arange(0, BLOCK_H)
            mh = h < H
            acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
            for qi in tl.range(0, D):
                idx = tl.load(cp + pid * D + qi).to(tl.int32)
                qbase = qi * V * H + idx * H + h
                sc = tl.load(sp + qi * V + idx).to(tl.float32)
                acc += tl.load(qp + qbase, mask=mh, other=0).to(tl.float32) * sc
            tl.store(yp + pid * H + h, acc, mask=mh)

        out = torch.empty(flat.shape[0], self.hidden, dtype=torch.float32, device=flat.device)
        kernel[(flat.shape[0], triton.cdiv(self.hidden, 128))](
            flat, self.q, self.scale, out, flat.shape[0], self.depth, self.vocab + 1, self.hidden, BLOCK_H=128
        )
        return out.reshape(*shape, self.hidden)

    def nbytes(self) -> int:
        return self.q.numel() * self.q.element_size() + self.scale.numel() * self.scale.element_size()


class RVQInt8Encoder(nn.Module):
    """Q8 residual encoder matching ``RVQInt4Encoder`` semantics."""

    def __init__(self, embedding: RVQInt8Embedding, block_v: int = 32, block_h: int = 128):
        super().__init__()
        self.depth, self.vocab, self.hidden = embedding.depth, embedding.vocab, embedding.hidden
        self.block_v, self.block_h = block_v, min(block_h, self.hidden)
        self.register_buffer("q", embedding.q[:, :self.vocab].contiguous())
        self.register_buffer("scale", embedding.scale[:, :self.vocab].contiguous())

    def forward(self, residual: torch.Tensor, code: torch.Tensor, depth_str: int = 0, k: int = 72):
        if residual.shape[-1] != self.hidden or code.shape[-1] != self.depth:
            raise ValueError("RVQ Q8 encoder shape mismatch")
        if residual.device != code.device:
            raise ValueError("RVQ residual and code must share a device")
        r = residual.reshape(-1, self.hidden).contiguous()
        flat_code = code.reshape(-1, self.depth).contiguous()
        end = min(self.depth, depth_str + k)
        if r.shape[0] == 0 or end == depth_str:
            return code
        if r.device.type != "cuda":
            for qi in range(depth_str, end):
                w = self.q[qi].float() * self.scale[qi, :, None].float()
                idx = (r[:, None, :] - w[None, :, :]).square().sum(-1).argmin(-1)
                flat_code[:, qi] = idx
                r -= w[idx].to(r.dtype)
            code.reshape(-1, self.depth).copy_(flat_code)
            return code
        import triton
        import triton.language as tl

        @triton.jit
        def distance(rp, qp, sp, vp, ip, P, V, H, BLOCK_V: tl.constexpr, BLOCK_H: tl.constexpr):
            pid = tl.program_id(0)
            row = pid // tl.cdiv(V, BLOCK_V)
            vb = pid % tl.cdiv(V, BLOCK_V)
            v = vb * BLOCK_V + tl.arange(0, BLOCK_V)
            mv = v < V
            acc = tl.zeros((BLOCK_V,), dtype=tl.float32)
            for h0 in tl.range(0, H, BLOCK_H):
                h = h0 + tl.arange(0, BLOCK_H)
                mh = h < H
                rv = tl.load(rp + row * H + h, mask=mh, other=0.0).to(tl.float32)
                sc = tl.load(sp + v, mask=mv, other=0.0).to(tl.float32)
                w = tl.load(qp + v[:, None] * H + h[None, :], mask=mv[:, None] & mh[None, :], other=0).to(tl.float32)
                diff = rv[None, :] - w * sc[:, None]
                acc += tl.sum(diff * diff, axis=1)
            acc = tl.where(mv, acc, float("inf"))
            pos = tl.argmin(acc, axis=0)
            tl.store(vp + row * tl.cdiv(V, BLOCK_V) + vb, tl.sum(tl.where(tl.arange(0, BLOCK_V) == pos, acc, 0.0)), mask=row < P)
            tl.store(ip + row * tl.cdiv(V, BLOCK_V) + vb, tl.sum(tl.where(tl.arange(0, BLOCK_V) == pos, v, 0)), mask=row < P)

        @triton.jit
        def reduce_update(vp, ip, rp, cp, qp, sp, P, D, depth, H, NB: tl.constexpr, BLOCK_H: tl.constexpr):
            row = tl.program_id(0)
            best, idx = float("inf"), 0
            for j in tl.static_range(0, NB):
                val = tl.load(vp + row * NB + j)
                take = val < best
                best = tl.where(take, val, best)
                idx = tl.where(take, tl.load(ip + row * NB + j).to(tl.int32), idx)
            for h0 in tl.range(0, H, BLOCK_H):
                h = h0 + tl.arange(0, BLOCK_H)
                mh = h < H
                old = tl.load(rp + row * H + h, mask=mh, other=0.0).to(tl.float32)
                w = tl.load(qp + idx * H + h, mask=mh, other=0).to(tl.float32)
                sc = tl.load(sp + idx).to(tl.float32)
                tl.store(rp + row * H + h, (old - w * sc).to(rp.dtype.element_ty), mask=mh)
            tl.store(cp + row * D + depth, idx)

        p = r.shape[0]
        nb = triton.cdiv(self.vocab, self.block_v)
        values = torch.empty((p, nb), dtype=torch.float32, device=r.device)
        indices = torch.empty((p, nb), dtype=torch.int32, device=r.device)
        for qi in range(depth_str, end):
            distance[(p * nb,)](r, self.q[qi], self.scale[qi], values, indices, p, self.vocab, self.hidden,
                                BLOCK_V=self.block_v, BLOCK_H=self.block_h)
            reduce_update[(p,)](values, indices, r, flat_code, self.q[qi], self.scale[qi],
                                p, self.depth, qi, self.hidden, NB=nb, BLOCK_H=self.block_h)
        code.reshape(-1, self.depth).copy_(flat_code)
        return code

    def nbytes(self) -> int:
        return self.q.numel() * self.q.element_size() + self.scale.numel() * self.scale.element_size()
