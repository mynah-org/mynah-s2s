import torch
from torch import nn

from mynah_v2.model.codec_cpu import LinearPointwiseConv1d, SlicedDepthwiseConv1d


def test_sliced_depthwise_matches_conv1d():
    for channels, kernel, padding in ((4, 3, 1), (8, 5, 2)):
        ref = nn.Conv1d(channels, channels, kernel, padding=padding, groups=channels).eval()
        cand = SlicedDepthwiseConv1d(ref).eval()
        x = torch.randn(2, channels, 32)
        torch.testing.assert_close(cand(x), ref(x), rtol=1e-5, atol=1e-5)


def test_linear_pointwise_matches_conv1d():
    ref = nn.Conv1d(7, 11, 1).eval()
    cand = LinearPointwiseConv1d(ref).eval()
    x = torch.randn(2, 7, 32)
    torch.testing.assert_close(cand(x), ref(x), rtol=1e-5, atol=1e-5)
