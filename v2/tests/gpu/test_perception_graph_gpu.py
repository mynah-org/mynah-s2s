"""Full-buffer perception CUDA graph (mynah_v2.model.perception_graph) on a tiny module built from the REAL NeMo pieces
(AudioToMelSpectrogramPreprocessor with per_feature normalization + ConformerEncoder rel_pos): the host check in
normalize_batch, the encoder pos-enc resize and the mask from the length tensor all run through NeMo's own code.
Gate: graph outputs BIT-identical (torch.equal) to eager over a session (filling buffer → full sliding window)."""

import threading

import pytest
import torch

pytestmark = pytest.mark.gpu
FULL = 71 * 1280


@pytest.fixture(scope="module", autouse=True)
def cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")


class TinyPerception(torch.nn.Module):
    def __init__(self):
        super().__init__()
        from nemo.collections.asr.modules import AudioToMelSpectrogramPreprocessor, ConformerEncoder
        self.pre = AudioToMelSpectrogramPreprocessor(sample_rate=16000, normalize="per_feature", window_size=0.025,
                                                     window_stride=0.01, features=80, n_fft=512, dither=1e-5,
                                                     pad_to=0)
        self.enc = ConformerEncoder(feat_in=80, n_layers=2, d_model=64, n_heads=4, subsampling="dw_striding",
                                    subsampling_factor=8, subsampling_conv_channels=32, ff_expansion_factor=2,
                                    conv_kernel_size=9, self_attention_model="rel_pos")
        self.proj = torch.nn.Linear(64, 32)

    def forward(self, input_signal=None, input_signal_length=None, return_encoder_emb=False):
        x, xl = self.pre(input_signal=input_signal, length=input_signal_length)
        e, el = self.enc(audio_signal=x, length=xl)
        out = self.proj(e.transpose(1, 2))
        return (out, el, e.transpose(1, 2)) if return_encoder_emb else (out, el)


def _model(seed=0):
    torch.manual_seed(seed)
    return TinyPerception().cuda().eval()


def _session(n_chunks, seed):
    """Sliding buffer like the wrapper: grows by 2 frames per chunk, then stays at FULL samples."""
    g = torch.Generator().manual_seed(seed)
    audio = torch.randn(1, (n_chunks + 2) * 2 * 1280, generator=g) * 0.1
    out = []
    for c in range(n_chunks):
        end = (c + 1) * 2 * 1280
        out.append(audio[:, max(0, end - FULL):end].contiguous())
    return out


def _call(m, buf):
    sig = buf.cuda()
    ln = torch.tensor([sig.shape[1]], dtype=torch.long, device="cuda")
    return m(input_signal=sig, input_signal_length=ln, return_encoder_emb=True)


def _eager_ref(bufs):
    m = _model()
    with torch.inference_mode():
        return [tuple(t.clone() for t in _call(m, b)) for b in bufs]


def _assert_equal(a, b, tag):
    for i, (x, y) in enumerate(zip(a, b)):
        assert x.shape == y.shape and x.dtype == y.dtype, f"{tag} out{i} shape/dtype"
        assert torch.equal(x, y), f"{tag} out{i} max|diff|={(x.float() - y.float()).abs().max().item()}"


def test_session_bit_identical_and_only_full_window_is_graphed():
    from mynah_v2.model.perception_graph import PerceptionGraph
    bufs = _session(50, seed=1)                     # 35 filling chunks (eager) + 15 full-window chunks (graph)
    ref = _eager_ref(bufs)
    m = _model()
    pg = PerceptionGraph(m, FULL)
    with torch.inference_mode():
        got = [tuple(t.clone() for t in _call(m, b)) for b in bufs]
    for c, (r, o) in enumerate(zip(ref, got)):
        _assert_equal(r, o, f"chunk {c}")
    n_full = sum(b.shape[1] == FULL for b in bufs)
    assert n_full >= 10
    assert pg.stats == {"captures": 1, "replays": n_full - 1, "eager": len(bufs) - n_full}


def test_returned_outputs_are_not_overwritten_by_the_next_replay():
    from mynah_v2.model.perception_graph import PerceptionGraph
    bufs = [b for b in _session(45, seed=2) if b.shape[1] == FULL]
    ref = _eager_ref(bufs)
    m = _model()
    PerceptionGraph(m, FULL)
    with torch.inference_mode():
        held = [_call(m, b) for b in bufs]          # no clone on the caller side: must still be correct at the end
    torch.cuda.synchronize()
    for c, (r, o) in enumerate(zip(ref, held)):
        _assert_equal(r, o, f"held {c}")


def test_two_sessions_share_the_graph_without_state():
    from mynah_v2.model.perception_graph import PerceptionGraph
    a, b = _session(45, seed=3), _session(45, seed=4)
    ref = _eager_ref(a) + _eager_ref(b)
    m = _model()
    pg = PerceptionGraph(m, FULL)
    with torch.inference_mode():
        got = [tuple(t.clone() for t in _call(m, x)) for x in a + b]
    for c, (r, o) in enumerate(zip(ref, got)):
        _assert_equal(r, o, f"call {c}")
    assert pg.stats["captures"] == 1


def test_capture_in_worker_while_main_syncs_then_replay_in_main():
    from mynah_v2.model.perception_graph import PerceptionGraph
    bufs = [b for b in _session(45, seed=5) if b.shape[1] == FULL]
    ref = _eager_ref(bufs)
    m = _model()
    pg = PerceptionGraph(m, FULL)
    res, err, stop = {}, [], threading.Event()

    def worker():
        try:
            with torch.inference_mode():
                res[0] = tuple(t.clone() for t in _call(m, bufs[0]))      # first full call → capture here
        except Exception as e:                                           # noqa: BLE001
            err.append(e)
        finally:
            stop.set()

    t = threading.Thread(target=worker)
    t.start()
    x = torch.randn(256, 256, device="cuda")
    while not stop.is_set():                                             # concurrent CUDA work + host syncs
        (x @ x).sum().item()
    t.join()
    assert not err, err
    with torch.inference_mode():
        for i in range(1, len(bufs)):
            res[i] = tuple(t.clone() for t in _call(m, bufs[i]))
    for i in range(len(bufs)):
        _assert_equal(ref[i], res[i], f"call {i}")
    assert pg.stats["captures"] == 1 and pg.stats["replays"] == len(bufs) - 1


def test_non_matching_calls_stay_eager_and_remove_restores():
    from mynah_v2.model.perception_graph import PerceptionGraph
    m = _model()
    cls_forward = type(m).forward
    pg = PerceptionGraph(m, FULL)
    sig = torch.randn(1, FULL, device="cuda") * 0.1
    ln = torch.tensor([FULL], device="cuda")
    with torch.inference_mode():
        m(input_signal=sig, input_signal_length=ln)                      # full: graphed (capture)
        m(input_signal=sig[:, :-1280], input_signal_length=ln - 1280)    # shorter: eager
        m(input_signal=sig.cpu().cuda(), input_signal_length=ln, return_encoder_emb=True)  # new key: capture
    m(input_signal=sig, input_signal_length=ln)                          # grad enabled: eager
    assert pg.stats["eager"] == 2 and pg.stats["captures"] == 2
    pg.remove()
    assert "forward" not in m.__dict__ and m.forward.__func__ is cls_forward


def test_negative_control_stale_input_is_caught():
    """If the graph input were NOT refreshed, the parity gate above must fail: replay without copy → wrong output."""
    from mynah_v2.model.perception_graph import PerceptionGraph
    bufs = [b for b in _session(45, seed=6) if b.shape[1] == FULL][:3]
    ref = _eager_ref(bufs)
    m = _model()
    pg = PerceptionGraph(m, FULL)
    with torch.inference_mode():
        _call(m, bufs[0])                                                # capture with bufs[0]
        graph, static_sig, static_ln, static_out = next(iter(pg.graphs.values()))
        graph.replay()                                                   # inputs NOT updated to bufs[1]
        stale = tuple(t.clone() for t in static_out)
    assert not torch.equal(stale[0], ref[1][0])
