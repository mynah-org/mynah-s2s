"""eartts_rvq_cache: the hoisted RVQ constants give bit-identical depthsum embedding / encoding results."""
import types

import torch
import torch.nn.functional as F

from mynah_v2.model import perf


def _ref_depthsum_embedding(rvq_embs, code):          # formula of NeMo ear_tts_model.depthsum_embedding
    b, t, d = code.size()
    ret = torch.zeros((b, t, rvq_embs.shape[-1]))
    embs = F.pad(rvq_embs, [0, 0, 0, 1])
    for i in range(d):
        ret = ret + F.embedding(code[..., i], embs[i])
    return ret


def _ref_encoding_step(embs, r, code, depth_str, k):  # formula of NeMo ear_tts_model.depthsum_encoding_step
    for i in range(depth_str, depth_str + k):
        idx_sel = (embs[i].pow(2).sum(-1) - 2 * (r.unsqueeze(-2) @ embs[i].transpose(-1, -2)).squeeze(-2)).argmin(-1)
        r = r - F.embedding(idx_sel, embs[i])
        code[..., i] = idx_sel
    return code


def _mod(d=5, v=16, h=8):
    torch.manual_seed(0)
    m = types.SimpleNamespace()
    m.rvq_embs = torch.randn(d, v, h)
    return m


def test_depthsum_embedding_bit_identical():
    m = _mod()
    code = torch.randint(0, 17, (3, 1, 5))             # includes the padding index v
    assert torch.equal(perf._depthsum_embedding_cached(m, code), _ref_depthsum_embedding(m.rvq_embs, code))


def test_encoding_step_bit_identical_and_cache_refresh():
    m = _mod()
    step = perf._make_depthsum_step_cached(m)
    r = torch.randn(3, 1, 8)
    for s, k in ((0, 2), (2, 3)):
        a = step(m.rvq_embs, r.clone(), torch.full((3, 1, 5), 16), s, k)
        b = _ref_encoding_step(m.rvq_embs, r.clone(), torch.full((3, 1, 5), 16), s, k)
        assert torch.equal(a, b)
    m.rvq_embs = torch.randn(5, 16, 8)                  # set_rvq_embs replaces the table: constants follow
    a = step(m.rvq_embs, r.clone(), torch.full((3, 1, 5), 16), 0, 5)
    b = _ref_encoding_step(m.rvq_embs, r.clone(), torch.full((3, 1, 5), 16), 0, 5)
    assert torch.equal(a, b)
