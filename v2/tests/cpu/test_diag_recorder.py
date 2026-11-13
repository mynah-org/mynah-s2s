import threading

import numpy as np
import torch

from mynah_v2.runtime.diag_recorder import DiagRecorder


class FakeCodec:
    def decode(self, code, code_len=None, constrain_value_range=True, cache=None, flush=False):
        return code.float(), code_len


def test_codes_attributed_per_session_and_uninstalled(tmp_path):
    codec, d = FakeCodec(), DiagRecorder()
    d.install_codes(codec)
    d.solo_sid = "a"
    codec.decode(torch.full((1, 2, 3), 7))
    d.uninstall_codes()
    assert "decode" not in codec.__dict__
    d.phase, d.solo_sid = "pair", None
    d.install_codes(codec)
    th = [threading.Thread(target=lambda v=v: codec.decode(torch.full((1, 1, 3), v), cache=None), name=f"rz-{s}")
          for s, v in (("b", 1), ("c", 2))]
    th.append(threading.Thread(target=lambda: codec.decode(torch.zeros(1, 1, 3)), name="Thread-9 (_fc_async_thread_fn)"))
    for t in th:
        t.start()
    for t in th:
        t.join()
    out, _ = codec.decode(torch.ones(1, 1, 3))              # main thread, no solo sid: keyed by thread name
    assert out.shape == (1, 1, 3)
    d.uninstall()
    info = d.save(tmp_path)["codes"]
    z = np.load(tmp_path / "codes.npz")
    assert info["solo__a"]["frames"] == 2 and (z["solo__a"] == 7).all()
    assert (z["pair__b"] == 1).all() and (z["pair__c"] == 2).all() and "pair__fc" in info
    assert "decode" not in codec.__dict__


def test_sync_recorded_and_restored(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device=None: calls.append(device))
    orig = torch.cuda.synchronize
    d = DiagRecorder()
    d.install_sync()
    torch.cuda.synchronize()
    d.uninstall()
    assert torch.cuda.synchronize is orig and calls == [None] and len(d.syncs) == 1 and d.syncs[0][0] == "solo"
