"""Optional fused Q4 single-step LSTM for the RNNT prediction network."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from mynah_v2.model.wo_gemv import pack_w4


_KERNELS = None


def _kernels():
    global _KERNELS
    if _KERNELS is not None:
        return _KERNELS
    import triton
    import triton.language as tl

    @triton.jit
    def gate_mm(x, h, qih, sih, mih, qhh, shh, mhh, bias, gates, B, I, H, G: tl.constexpr,
                BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
        b = tl.program_id(0)
        pn = tl.program_id(1)
        n = pn * BLOCK_N + tl.arange(0, BLOCK_N)
        valid_n = n < 4 * H
        acc = tl.load(bias + n, mask=valid_n, other=0.0).to(tl.float32)
        half = I // 2
        groups = I // G
        for k0 in tl.range(0, I, BLOCK_K):
            rh = k0 // 2 + tl.arange(0, BLOCK_K // 2)
            valid = (2 * rh) < I
            packed = tl.load(qih + n[:, None] * half + rh[None, :],
                             mask=valid_n[:, None] & valid[None, :], other=0)
            xe = tl.load(x + b * I + 2 * rh, mask=valid, other=0.0).to(tl.float32)
            xo = tl.load(x + b * I + 2 * rh + 1, mask=valid & (2 * rh + 1 < I), other=0.0).to(tl.float32)
            gi = (2 * rh) // G
            sc = tl.load(sih + n[:, None] * groups + gi[None, :],
                         mask=valid_n[:, None] & valid[None, :], other=0.0).to(tl.float32)
            mn = tl.load(mih + n[:, None] * groups + gi[None, :],
                         mask=valid_n[:, None] & valid[None, :], other=0.0).to(tl.float32)
            acc += tl.sum((packed & 0xF).to(tl.float32) * sc * xe[None, :] +
                          ((packed >> 4).to(tl.float32) * sc) * xo[None, :] +
                          mn * (xe[None, :] + xo[None, :]), axis=1)
        half = H // 2
        groups = H // G
        for k0 in tl.range(0, H, BLOCK_K):
            rh = k0 // 2 + tl.arange(0, BLOCK_K // 2)
            valid = (2 * rh) < H
            packed = tl.load(qhh + n[:, None] * half + rh[None, :],
                             mask=valid_n[:, None] & valid[None, :], other=0)
            he = tl.load(h + b * H + 2 * rh, mask=valid, other=0.0).to(tl.float32)
            ho = tl.load(h + b * H + 2 * rh + 1, mask=valid & (2 * rh + 1 < H), other=0.0).to(tl.float32)
            gi = (2 * rh) // G
            sc = tl.load(shh + n[:, None] * groups + gi[None, :],
                         mask=valid_n[:, None] & valid[None, :], other=0.0).to(tl.float32)
            mn = tl.load(mhh + n[:, None] * groups + gi[None, :],
                         mask=valid_n[:, None] & valid[None, :], other=0.0).to(tl.float32)
            acc += tl.sum((packed & 0xF).to(tl.float32) * sc * he[None, :] +
                          ((packed >> 4).to(tl.float32) * sc) * ho[None, :] +
                          mn * (he[None, :] + ho[None, :]), axis=1)
        tl.store(gates + b * (4 * H) + n, acc.to(gates.dtype.element_ty), mask=valid_n)

    @triton.jit
    def activate(gates, h, c, out, h_new, c_new, H, BLOCK_H: tl.constexpr):
        b = tl.program_id(0)
        j = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
        valid = j < H
        base = b * H + j
        gi = tl.load(gates + base, mask=valid, other=0.0).to(tl.float32)
        gf = tl.load(gates + base + H, mask=valid, other=0.0).to(tl.float32)
        gg = tl.load(gates + base + 2 * H, mask=valid, other=0.0).to(tl.float32)
        go = tl.load(gates + base + 3 * H, mask=valid, other=0.0).to(tl.float32)
        fi = 1.0 / (1.0 + tl.exp(-gi))
        ff = 1.0 / (1.0 + tl.exp(-gf))
        fo = 1.0 / (1.0 + tl.exp(-go))
        tanh_g = 2.0 / (1.0 + tl.exp(-2.0 * gg)) - 1.0
        cn = ff * tl.load(c + base, mask=valid, other=0.0).to(tl.float32) + fi * tanh_g
        tanh_c = 2.0 / (1.0 + tl.exp(-2.0 * cn)) - 1.0
        hn = fo * tanh_c
        tl.store(c_new + base, cn.to(c_new.dtype.element_ty), mask=valid)
        tl.store(h_new + base, hn.to(h_new.dtype.element_ty), mask=valid)
        tl.store(out + base, hn.to(out.dtype.element_ty), mask=valid)

    _KERNELS = (gate_mm, activate)
    return _KERNELS


class Int4LSTM(nn.Module):
    """Q4 LSTM matching NeMo's inference-only ``LSTMDropout`` contract.

    It supports the actual RNNT configuration (two unprojected layers, batch
    first only through the public ``[T,B,H]`` call).  The gate matmul and gate
    update are separate fused Triton kernels; CPU uses a faithful dequantized
    fallback for tests.
    """

    def __init__(self, rnn_module: nn.Module, group: int = 128):
        super().__init__()
        lstm = getattr(rnn_module, "lstm", None)
        if not isinstance(lstm, nn.LSTM) or lstm.proj_size or lstm.bidirectional:
            raise ValueError("only unidirectional unprojected LSTM is supported")
        if getattr(rnn_module, "dropout", None) is not None and rnn_module.training:
            raise ValueError("Q4 RNNT LSTM is inference-only")
        self.input_size, self.hidden_size = lstm.input_size, lstm.hidden_size
        self.num_layers = lstm.num_layers
        self.group = group
        self.qih = nn.ParameterList()
        self.sih = nn.ParameterList()
        self.mih = nn.ParameterList()
        self.qhh = nn.ParameterList()
        self.shh = nn.ParameterList()
        self.mhh = nn.ParameterList()
        self.bias = nn.ParameterList()
        for layer in range(self.num_layers):
            wi = getattr(lstm, f"weight_ih_l{layer}").detach()
            wh = getattr(lstm, f"weight_hh_l{layer}").detach()
            pi, ph = pack_w4(wi, group=group), pack_w4(wh, group=group)
            self.qih.append(nn.Parameter(pi["q"], requires_grad=False))
            self.sih.append(nn.Parameter(pi["scale"], requires_grad=False))
            self.mih.append(nn.Parameter(pi["min"], requires_grad=False))
            self.qhh.append(nn.Parameter(ph["q"], requires_grad=False))
            self.shh.append(nn.Parameter(ph["scale"], requires_grad=False))
            self.mhh.append(nn.Parameter(ph["min"], requires_grad=False))
            bi = getattr(lstm, f"bias_ih_l{layer}").detach()
            bh = getattr(lstm, f"bias_hh_l{layer}").detach()
            self.bias.append(nn.Parameter((bi + bh).contiguous(), requires_grad=False))

    def _weights(self, layer):
        qi = self.qih[layer]
        qh = self.qhh[layer]
        return qi, self.sih[layer], self.mih[layer], qh, self.shh[layer], self.mhh[layer], self.bias[layer]

    def forward(self, x: torch.Tensor, state=None):
        if x.ndim != 3 or x.shape[-1] != self.input_size:
            raise ValueError(f"RNNT Q4 expects [T,B,{self.input_size}], got {tuple(x.shape)}")
        if not x.is_floating_point():
            raise ValueError("RNNT Q4 activations must be floating point")
        t, b, _ = x.shape
        if state is None:
            hs = [torch.zeros(b, self.hidden_size, dtype=x.dtype, device=x.device) for _ in range(self.num_layers)]
            cs = [torch.zeros_like(h) for h in hs]
        else:
            if len(state) != 2 or any(s.ndim != 3 for s in state):
                raise ValueError("RNNT Q4 state must be (h, c) with [layers,batch,hidden] tensors")
            if (state[0].shape != (self.num_layers, b, self.hidden_size)
                    or state[1].shape != (self.num_layers, b, self.hidden_size)):
                raise ValueError("RNNT Q4 state shape does not match the current batch")
            if state[0].device != x.device or state[1].device != x.device:
                raise ValueError("RNNT Q4 state and activations must share a device")
            hs = [state[0][i].contiguous() for i in range(self.num_layers)]
            cs = [state[1][i].contiguous() for i in range(self.num_layers)]
        if t == 0 or b == 0:
            return x.new_empty((t, b, self.hidden_size)), (torch.stack(hs), torch.stack(cs))
        if x.device.type != "cuda":
            weights = [(
                self._dequant(self.qih[li], self.sih[li], self.mih[li], self.input_size if li == 0 else self.hidden_size),
                self._dequant(self.qhh[li], self.shh[li], self.mhh[li], self.hidden_size),
                self.bias[li],
            ) for li in range(self.num_layers)]
            outputs = x.new_empty((t, b, self.hidden_size))
            for ti in range(t):
                y = x[ti]
                for li in range(self.num_layers):
                    wi, wh, bias = weights[li]
                    gates = F.linear(y, wi) + F.linear(hs[li], wh) + bias
                    i, f, g, o = gates.chunk(4, -1)
                    cs[li] = torch.sigmoid(f) * cs[li] + torch.sigmoid(i) * torch.tanh(g)
                    hs[li] = torch.sigmoid(o) * torch.tanh(cs[li])
                    y = hs[li]
                outputs[ti].copy_(y)
            return outputs, (torch.stack(hs), torch.stack(cs))
        import triton
        mm, act = _kernels()
        # Workspace belongs to this invocation, never the module: concurrent
        # sessions/streams cannot overwrite each other's gates or states.
        outputs = x.new_empty((t, b, self.hidden_size))
        gates_buf = [torch.empty((b, 4 * self.hidden_size), dtype=x.dtype, device=x.device)
                     for _ in range(self.num_layers)]
        h_buf = [torch.empty_like(hs[li]) for li in range(self.num_layers)]
        c_buf = [torch.empty_like(cs[li]) for li in range(self.num_layers)]
        for ti in range(t):
            y = x[ti].contiguous()
            for li in range(self.num_layers):
                gates = gates_buf[li]
                hn = h_buf[li]
                cn = c_buf[li]
                qi, si, mi, qh, sh, mh, bias = self._weights(li)
                mm[(b, triton.cdiv(4 * self.hidden_size, 128))](
                    y, hs[li], qi, si, mi, qh, sh, mh, bias, gates,
                    b, self.input_size if li == 0 else self.hidden_size, self.hidden_size,
                    G=self.group, BLOCK_N=128, BLOCK_K=128,
                )
                act[(b, triton.cdiv(self.hidden_size, 128))](gates, hs[li], cs[li], hn, hn, cn,
                                                               self.hidden_size, BLOCK_H=128)
                hs[li], cs[li], y = hn, cn, hn
            outputs[ti].copy_(y)
        return outputs, (torch.stack(hs), torch.stack(cs))

    @staticmethod
    def _dequant(q, scale, minimum, k):
        lo, hi = q & 0xF, q >> 4
        vals = torch.stack((lo, hi), -1).reshape(q.shape[0], k).float()
        g = k // scale.shape[-1]
        return (vals.reshape(q.shape[0], -1, g) * scale.float()[..., None] + minimum.float()[..., None]).reshape(q.shape[0], k)

    def nbytes(self) -> int:
        tensors = list(self.qih) + list(self.sih) + list(self.mih) + list(self.qhh) + list(self.shh) + list(self.mhh) + list(self.bias)
        return sum(t.numel() * t.element_size() for t in tensors)

