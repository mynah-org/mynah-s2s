# SPDX-License-Identifier: MIT AND Apache-2.0
# Original code: MIT (see LICENSE). _upstream_spec_to_wav is an unmodified copy of NeMo Speech ear_tts_vae_codec.py
# spec_to_wav (Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES, Apache License 2.0, LICENSES/Apache-2.0.txt), kept as
# the reference of a bit-identity test. See NOTICE.md.
"""codec_async: cached-envelope spec_to_wav is bit-identical to upstream; PendingHost waits only on first data read."""

import threading

import pytest
import torch
import torch.nn.functional as F

from mynah_v2.model import codec_async as ca
from mynah_v2.model import perf


def _upstream_spec_to_wav(spec, n_fft, hop_length, win_length, window_fn=torch.hann_window,
                          constrain_value_range=False):
    # verbatim ops of NeMo 097dfe9 ear_tts_vae_codec.py:166-251
    with torch.autocast(device_type=spec.device.type, enabled=False):
        pad = (win_length - hop_length) // 2
        T = spec.size(-1)
        window = window_fn(win_length, device=spec.device)
        ifft = torch.fft.irfft(spec, n=n_fft, dim=-2, norm="backward")
        window_unsqz = window.unsqueeze(-1)
        if constrain_value_range:
            ifft = torch.where(ifft >= 0, torch.minimum(ifft, window_unsqz), torch.maximum(ifft, -window_unsqz))
        ifft = ifft * window_unsqz
        output_size = (T - 1) * hop_length + win_length
        wav = F.fold(ifft, output_size=(1, output_size), kernel_size=(1, win_length),
                     stride=(1, hop_length))[..., 0, 0, pad:-pad]
        window_sq = window.square().expand(T, -1).transpose(0, 1)
        window_envelope = F.fold(window_sq, output_size=(1, output_size), kernel_size=(1, win_length),
                                 stride=(1, hop_length)).squeeze()[pad:-pad]
        assert (window_envelope > 1e-11).all()
        return wav / window_envelope


@pytest.mark.parametrize("T", [3, 4, 5, 6])
@pytest.mark.parametrize("cvr", [True, False])
def test_cached_spec_to_wav_bit_identical(T, cvr):
    g = torch.Generator().manual_seed(T)
    cache = {}
    f = ca.make_cached_spec_to_wav(_upstream_spec_to_wav, cache, threading.Lock())
    for _ in range(3):                                   # first call fills the cache, later calls reuse it
        spec = torch.complex(torch.randn(2, 17, T, generator=g), torch.randn(2, 17, T, generator=g))
        a = _upstream_spec_to_wav(spec, 32, 8, 32, constrain_value_range=cvr)
        b = f(spec, 32, 8, 32, constrain_value_range=cvr)
        assert a.shape == b.shape and torch.equal(a, b)
    assert len(cache) == 1


def test_cached_spec_to_wav_rejects_other_window():
    f = ca.make_cached_spec_to_wav(None, {}, threading.Lock())
    with pytest.raises(ValueError):
        f(torch.zeros(1, 17, 3, dtype=torch.complex64), 32, 8, 32, window_fn=torch.hamming_window)


class FakeEvent:
    def __init__(self):
        self.syncs = 0

    def synchronize(self):
        self.syncs += 1


def test_pending_host_waits_once_on_data_and_never_on_metadata():
    ev = FakeEvent()
    keep = torch.ones(3)
    p = ca.PendingHost.wrap(torch.arange(6.).reshape(1, 1, 6), ev, keepalive=keep)
    assert p.shape[-1] == 6 and p.dim() == 3 and p.numel() == 6 and p.dtype == torch.float32
    assert ev.syncs == 0 and not p.ready
    out = torch.cat([p, torch.zeros(1, 1, 2)], dim=-1).to("cpu", dtype=torch.float32)
    assert ev.syncs == 1 and p.ready
    assert type(out) is torch.Tensor and out.shape[-1] == 8 and out[0, 0, 5] == 5
    (p * 2).sum()
    assert ev.syncs == 1                                  # already materialized
    assert p.__dict__.get("_v2_keep") is None             # device source released after the wait


def test_pending_host_waits_inside_list_args():
    evs = [FakeEvent(), FakeEvent()]
    ps = [ca.PendingHost.wrap(torch.ones(1, 1, 3) * i, e) for i, e in enumerate(evs)]
    c = torch.cat(ps, dim=-1)
    assert [e.syncs for e in evs] == [1, 1] and c.tolist() == [[[0, 0, 0, 1, 1, 1]]]


def test_deferred_d2h_cpu_on_cpu_tensor_is_plain_copy():
    d = torch.ones(1, 1, 4).as_subclass(ca.DeferredD2H)
    out = d.detach().cpu()
    assert type(out) is torch.Tensor and torch.equal(out, torch.ones(1, 1, 4))


def test_fc_decode_wraps_only_on_predicate():
    wav, ln = torch.ones(1, 1, 4), torch.tensor([4])
    orig = lambda *a, **kw: (wav, ln)
    on = ca.make_fc_decode(orig, predicate=lambda: True)
    off = ca.make_fc_decode(orig, predicate=lambda: False)
    w1, l1 = on(None)
    assert isinstance(w1, ca.DeferredD2H) and l1 is ln and torch.equal(w1.as_subclass(torch.Tensor), wav)
    w2, _ = off(None)
    assert type(w2) is torch.Tensor


def test_fc_thread_name_detection():
    seen = []

    def _fc_async_thread_fn():
        seen.append(ca.is_fc_thread())

    t = threading.Thread(target=_fc_async_thread_fn)
    t.start()
    t.join()
    assert seen == [True] and not ca.is_fc_thread()


def test_fc_d2h_enable_disable_restores_codec():
    class Codec:
        def decode(self, *a, **kw):
            return torch.zeros(1, 1, 2), None

    class E:
        pass
    e = E()
    e.wrapper = E()
    e.wrapper.model = E()
    e.wrapper.model.tts_model = E()
    codec = e.wrapper.model.tts_model.audio_codec = Codec()
    perf.enable("codec_fc_d2h", e)
    assert hasattr(codec.decode, "__v2_fcd2h__")
    perf.enable("codec_fc_d2h", e)                      # idempotent
    assert not hasattr(codec.decode.__v2_fcd2h__, "__v2_fcd2h__")
    perf.disable("codec_fc_d2h", e)
    assert "decode" not in codec.__dict__
