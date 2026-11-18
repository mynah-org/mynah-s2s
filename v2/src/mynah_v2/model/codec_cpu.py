"""CPU codec building blocks for an opt-in EarTTS offload experiment.

The serving path does not import this module by default. These adapters keep
the codec weights in FP32 and rewrite common depthwise/pointwise Conv1d
operations so CPU execution can be measured before adding a worker process.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SlicedDepthwiseConv1d(nn.Module):
    """Depthwise Conv1d expressed as a channel-wise sliding-window reduction."""

    def __init__(self, conv: nn.Conv1d):
        if conv.groups != conv.in_channels or conv.out_channels != conv.in_channels:
            raise ValueError("expected a depthwise Conv1d")
        if conv.padding_mode != "zeros":
            raise ValueError("only zero-padded depthwise Conv1d is supported")
        super().__init__()
        self.channels = conv.in_channels
        self.kernel_size = tuple(conv.kernel_size)
        self.stride = tuple(conv.stride)
        self.padding = tuple(conv.padding)
        self.dilation = tuple(conv.dilation)
        self.register_buffer("weight", conv.weight.detach().clone())
        self.register_buffer("bias", None if conv.bias is None else conv.bias.detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dilation != (1,) or self.stride != (1,) or self.padding[0] < 0:
            return F.conv1d(x, self.weight, self.bias, self.stride, self.padding,
                            self.dilation, self.channels)
        pad = self.padding[0]
        if pad:
            x = F.pad(x, (pad, pad))
        k = self.kernel_size[0]
        windows = x.unfold(-1, k, self.stride[0])
        y = (windows * self.weight[:, 0, :].view(1, self.channels, 1, k)).sum(-1)
        if self.bias is not None:
            y = y + self.bias.view(1, self.channels, 1)
        return y


class LinearPointwiseConv1d(nn.Module):
    """Pointwise Conv1d as one linear transform per time position."""

    def __init__(self, conv: nn.Conv1d):
        if conv.kernel_size != (1,) or conv.stride != (1,) or conv.padding != (0,) \
                or conv.dilation != (1,) or conv.groups != 1:
            raise ValueError("expected a dense 1x1 Conv1d")
        super().__init__()
        self.register_buffer("weight", conv.weight.detach()[:, :, 0].clone())
        self.register_buffer("bias", None if conv.bias is None else conv.bias.detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x.transpose(1, 2), self.weight, self.bias).transpose(1, 2)


def rewrite_codec(codec: nn.Module, *, pointwise: bool = True) -> dict:
    """Replace safe Conv1d children in-place and return a replacement count."""
    depthwise = pointwise_count = 0
    for name, module in list(codec.named_modules()):
        if not isinstance(module, nn.Conv1d):
            continue
        replacement = None
        if module.groups == module.in_channels == module.out_channels:
            try:
                replacement = SlicedDepthwiseConv1d(module)
                depthwise += 1
            except ValueError:
                replacement = None
        elif pointwise and module.kernel_size == (1,) and module.groups == 1:
            try:
                replacement = LinearPointwiseConv1d(module)
                pointwise_count += 1
            except ValueError:
                replacement = None
        if replacement is None:
            continue
        parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
        parent = codec.get_submodule(parent_name) if parent_name else codec
        setattr(parent, child_name, replacement)
    return {"depthwise": depthwise, "pointwise": pointwise_count}
