"""The single-process engine: one GPU thread runs every model for every session, under one scheduler.

Models (all ours, torch): the ears (ears/own.py: Nemotron streaming ASR + window encoder + GRU), the LLM (qwen_own),
the talker (tts/pocket_own.py, one model per language). Sessions (core.Pipeline) talk to them through in-process
backends that keep the interfaces of the old subprocess plugins (EarsLink, QwenLLM, PocketTalker work unchanged):

  InProcEars    send/poll like the ears process; hops wait in a queue
  InProcLLM     send/poll like the Qwen process; requests are prefilled in chunks and decoded in batches
  InProcTalker  send/poll/wait like the Pocket process; one generator per session, stepped by deadline

Scheduler (one thread, the only one issuing GPU work), in priority order every turn:
  1. ears: all waiting hops of all sessions as one batched hop (hard real time: a hop every 320 ms per session);
  2. talker: one latent frame (80 ms of audio) for each session whose generated audio is less than ``ahead_s`` ahead
     of real time, most urgent first, within ``talker_budget_ms``;
  3. LLM: one batched decode step for every decoding request, or one prefill chunk of the oldest waiting request
     (fair: decode and prefill alternate when both wait).
"""

from __future__ import annotations

import queue
import threading
import time
from collections import deque


class _Boxes:
    def __init__(self):
        self.boxes: dict = {}
        self.misc: queue.Queue = queue.Queue()
        self.lock = threading.Lock()

    def put(self, key, m):
        if key is None:
            self.misc.put(m)
            return
        with self.lock:
            b = self.boxes.setdefault(key, deque())
            b.append(m)

    def poll(self, key, block=False):
        with self.lock:
            b = self.boxes.get(key)
            if not b:
                out = []
            else:
                out = list(b)
                b.clear()
        if not out and block:
            time.sleep(0.005)
        return out

    def wait(self, ev, timeout=300.0, pred=None):
        t_end = time.monotonic() + timeout
        held = []
        try:
            while time.monotonic() < t_end:
                try:
                    m = self.misc.get(timeout=0.2)
                except queue.Empty:
                    continue
                if m.get("ev") == ev and (pred is None or pred(m)):
                    return m
                if m.get("ev") == "error":
                    raise RuntimeError(m.get("error"))
                held.append(m)
            raise TimeoutError(ev)
        finally:
            for m in held:
                self.misc.put(m)


class InProcEars:
    def __init__(self, ears, sched):
        self.e, self.s = ears, sched
        self.box = _Boxes()
        self.pending: dict = {}
        self.lock = threading.Lock()
        self.ready = {"ev": "ready", "inproc": True}

    def send(self, m):
        op = m.get("op")
        if op == "hop":
            import base64
            import numpy as np
            m = dict(m, u=np.frombuffer(base64.b64decode(m["u"]), "<f4"),
                     a=np.frombuffer(base64.b64decode(m["a"]), "<f4"))
            m["t_arr"] = time.monotonic()
            with self.lock:
                self.pending[m["sid"]] = m
            self.s.kick()
        else:
            self.s.submit(("ears", m))

    def poll(self, sid, block=False):
        return self.box.poll(sid, block)

    def request(self, m, ev, timeout=60.0):
        self.s.submit(("ears", m))
        return self.box.wait(ev, timeout)

    def close(self):
        pass

    # scheduler side
    def control(self, m):
        from .ears.rules import Thresholds
        op = m.get("op")
        if op == "open":
            self.e.open(m["sid"], m.get("lang", "it"), Thresholds(**(m.get("thr") or {})))
            self.box.put(m["sid"], {"ev": "opened", "sid": m["sid"]})
        elif op == "close":
            self.e.close(m["sid"])
            with self.lock:
                self.pending.pop(m["sid"], None)
        elif op == "transcribe":
            import base64
            import numpy as np
            x = np.frombuffer(base64.b64decode(m["pcm"]), "<f4")
            self.box.put(None, {"ev": "transcript", "id": m.get("id"), "text": self.e.transcribe(x, m.get("lang", "it"))})
        elif op == "mem":
            self.box.put(None, {"ev": "mem", "id": m.get("id"), **self.s.mem()})

    gather_ms = 120.0    # cohort window (mynah-asr design): hops of sessions out of phase wait for each other

    def due(self) -> bool:
        with self.lock:
            if not self.pending:
                return False
            if len(self.pending) >= len(self.e.sess):
                return True
            return (time.monotonic() - min(m["t_arr"] for m in self.pending.values())) * 1e3 >= self.gather_ms

    def run(self) -> bool:
        if not self.due():
            return False
        with self.lock:
            items, self.pending = list(self.pending.values()), {}
        if not items:
            return False
        res = self.e.hop([(m["sid"], m["u"], m["a"], m["mode"], bool(m.get("reset"))) for m in items])
        self.e.batch_sizes = getattr(self.e, "batch_sizes", []) + [len(items)]
        for m, r in zip(items, res):
            self.box.put(m["sid"], {"ev": "hop", "sid": m["sid"], "n": m["n"], "t_ns": time.monotonic_ns(), **r})
        return True


class InProcTalker:
    """Pocket talker for every session (our runtime, tts/pocket_own.py): one model + one PocketBatch per language,
    voice states per (lang, voice); each session has at most one row in its language's batch, and every scheduler
    turn steps, in one batched FlowLM call per language, the rows whose audio is less than ``ahead_s`` ahead."""

    def __init__(self, sched, make_model, ahead_s: float = 1.2, slots: int = 16, default_voice: str = ""):
        self.s, self.make_model, self.default_voice = sched, make_model, default_voice
        self.models: dict = {}
        self.batches: dict = {}
        self.voices: dict = {}
        self.box = _Boxes()
        self.jobs: dict = {}            # ss -> deque of pending (sid, epoch, lang, voice, prepared, after, last)
        self.cur: dict = {}             # ss -> row info
        self.floor: dict = {}
        self.ahead_s, self.slots = ahead_s, slots
        self.open_lock = threading.Lock()
        self.ready = {"ev": "ready", "inproc": True}
        self.lock = threading.Lock()

    # client side (PocketProc interface)
    def send(self, m):
        if m.get("op") == "cancel":
            with self.lock:
                self.floor[m["ss"]] = max(self.floor.get(m["ss"], 0), int(m["epoch"]))
            self.s.kick()
            return
        self.s.submit(("talker", m))

    def poll(self, ss):
        return self.box.poll(ss)

    def wait(self, ev, timeout=300.0, pred=None):
        return self.box.wait(ev, timeout, pred)

    def close(self):
        pass

    # scheduler side
    def _model(self, lang):
        if lang not in self.models:
            self.models[lang] = self.make_model(lang)
            from .tts.pocket_own import PocketBatch
            self.batches[lang] = PocketBatch(self.models[lang], slots=self.slots, max_len=512)
        return self.models[lang]

    def _voice(self, lang, voice):
        voice = voice or self.default_voice
        key = (lang, voice)
        if key not in self.voices:
            self.voices[key] = self._model(lang).voice_from_file(voice)
        return self.voices[key]

    def control(self, m):
        op = m.get("op")
        if op == "voice":
            t0 = time.monotonic()
            self._voice(m["lang"], m.get("voice"))
            self.box.put(None, {"ev": "voice", "id": m.get("id"), "lang": m["lang"],
                                "load_s": round(time.monotonic() - t0, 2)})
        elif op == "say":
            model = self._model(m["lang"])
            ch = model.chunks(m["text"])
            q = self.jobs.setdefault(m["ss"], deque())
            for i, (prep, after) in enumerate(ch):
                q.append({"sid": m["sid"], "epoch": int(m["epoch"]), "lang": m["lang"], "voice": m.get("voice"),
                          "text": prep, "after": after, "last": i == len(ch) - 1, "t_req": time.monotonic()})
        elif op == "mem":
            self.box.put(None, {"ev": "mem", "id": m.get("id"), **self.s.mem(), "langs": sorted(self.models)})
        elif op == "close":
            self.jobs.pop(m["ss"], None)
            c = self.cur.pop(m["ss"], None)
            if c is not None:
                self.batches[c["lang"]].stop(m["ss"])

    def _finish(self, ss, c, cancelled=False):
        self.batches[c["lang"]].stop(ss)
        del self.cur[ss]
        if c["last"] or cancelled:
            self.box.put(ss, {"ev": "done", "ss": ss, "sid": c["sid"], "cancelled": cancelled,
                              "audio_s": round(c["samples"] / 24000, 2), "first_audio_ms": c.get("first_ms")})

    def run(self, budget_ms: float) -> bool:
        did = False
        now = time.monotonic()
        with self.lock:
            floor = dict(self.floor)
        for ss in list(self.cur):                       # cancelled rows stop at once
            if self.cur[ss]["epoch"] < floor.get(ss, 0):
                self._finish(ss, self.cur[ss], cancelled=True)
        for ss, q in self.jobs.items():
            while q and q[0]["epoch"] < floor.get(ss, 0):
                q.popleft()
            if ss not in self.cur and q:
                j = q.popleft()
                v = self._voice(j["lang"], j["voice"])
                self.batches[j["lang"]].start(ss, v, j["text"], j["after"])
                self.cur[ss] = dict(j, t0=None, samples=0, start=now, first_ms=None)
                did = True
        by_lang: dict = {}
        for ss, c in self.cur.items():
            if self._ahead(c, now) <= self.ahead_s:
                by_lang.setdefault(c["lang"], []).append(ss)
        for lang, keys in by_lang.items():
            res = self.batches[lang].step(keys)
            did = True
            for ss, audio in res.items():
                c = self.cur[ss]
                if audio is None:
                    self._finish(ss, c)
                    continue
                if c["t0"] is None:
                    c["t0"] = time.monotonic()
                    c["first_ms"] = round((c["t0"] - c["t_req"]) * 1e3, 1)
                pcm = audio.float().cpu().numpy()
                c["samples"] += len(pcm)
                self.box.put(ss, {"ev": "audio", "ss": ss, "sid": c["sid"], "pcm": pcm})
        return did

    @staticmethod
    def _ahead(c, now):
        if c["t0"] is None:
            return -1.0
        return c["samples"] / 24000 - (now - c["t0"])


class InProcLLM:
    """Qwen (qwen_own) for every session: chunked prefill + batched greedy decode; the QwenLLM client protocol."""
    releases_slots = True

    def __init__(self, sched, eng, max_new: int = 160, prefill_chunk: int = 256):
        self.s, self.eng = sched, eng
        self.max_new, self.chunk = max_new, prefill_chunk
        self.box = _Boxes()
        self.slots: dict = {}
        self.wait_q: deque = deque()       # requests to prefill
        self.active: dict = {}             # (ss, rid) -> decoding state
        self.cancelled: set = set()
        self.lock = threading.Lock()
        self.turn = 0
        self.prefix_keys: set = set()
        self.log: list = []

    def send(self, m):
        op = m.get("op")
        if op == "cancel":
            with self.lock:
                self.cancelled.add((m["ss"], m["rid"]))
            return
        self.s.submit(("llm", m))

    def poll(self, ss):
        return self.box.poll(ss)

    def close(self):
        pass

    def control(self, m):
        if m.get("op") == "close":
            ss = m["ss"]
            for k in [k for k in self.active if k[0] == ss]:
                del self.active[k]
            self.wait_q = deque(r for r in self.wait_q if r["ss"] != ss)
            if ss in self.slots:
                self.eng.close(self.slots.pop(ss))
            return
        if m.get("op") == "gen":
            ss = m["ss"]
            if ss not in self.slots:
                self.slots[ss] = self.eng.open()
            ids = self.eng.chat_ids(m["messages"], m.get("tools"))
            msgs = m["messages"]
            if msgs and msgs[0].get("role") == "system" and hasattr(self.eng, "add_prefix"):
                key = (msgs[0].get("content"), repr(m.get("tools")))
                if key not in self.prefix_keys:           # once per distinct system prompt + tools
                    self.prefix_keys.add(key)
                    pre = self.eng.shared_prefix(msgs[0], m.get("tools"))
                    if pre and ids[:len(pre)] == pre:
                        self.eng.add_prefix(pre)      # later sessions copy this state instead of prefilling it
            for k in [k for k in self.active if k[0] == ss]:            # one generation per session slot
                self.box.put(ss, {"ev": "end", "ss": ss, "rid": k[1], "n": 0, "cancelled": True})
                del self.active[k]
            self.wait_q = deque(r for r in self.wait_q if r["ss"] != ss)
            self.wait_max = max(getattr(self, "wait_max", 0), len(self.wait_q) + 1)
            req = {"ss": ss, "rid": m["rid"], "ids": ids, "slot": self.slots[ss], "begun": False,
                   "t0": time.monotonic(), "after_tool": bool(msgs) and msgs[-1].get("role") == "tool"}
            if req["after_tool"]:          # a reply after a tool result goes first: its caller is already waiting
                pos = next((i for i, r in enumerate(self.wait_q) if not r["begun"] and not r.get("after_tool")),
                           len(self.wait_q))
                self.wait_q.insert(pos, req)
            else:
                self.wait_q.append(req)

    def run(self, budget_ms: float = 0.0, stop=None) -> bool:
        """One LLM action (a batched decode step or one prefill chunk); with a budget, keep going until it is spent
        or ``stop()`` says another component is due (the ears' cohort): decode needs many short steps and taking only
        one per scheduler round left the GPU idle between rounds."""
        t_end = time.monotonic() + budget_ms / 1e3
        did = False
        while self._run_one():
            did = True
            if time.monotonic() >= t_end or (stop is not None and stop()):
                break
        return did

    def _run_one(self) -> bool:
        import torch
        with self.lock:
            cancelled = set(self.cancelled)
        for key in [k for k in self.active if k in cancelled]:
            self.box.put(key[0], {"ev": "end", "ss": key[0], "rid": key[1], "n": 0, "cancelled": True})
            del self.active[key]
        self.wait_q = deque(r for r in self.wait_q if (r["ss"], r["rid"]) not in cancelled)
        self.turn += 1
        if self.active and (not self.wait_q or self.turn % 2 == 0):
            keys = list(self.active)
            logits = self.eng.decode_batch([self.active[k]["slot"] for k in keys], [self.active[k]["next"] for k in keys])
            toks = torch.argmax(logits, -1).tolist()
            for k, t in zip(keys, toks):
                self._emit(k, t)
            return True
        if self.wait_q:
            r = self.wait_q[0]
            if not r["begun"]:
                r["n_prefill"] = self.eng.prefill_begin(r["slot"], r["ids"])
                r["begun"] = True
                r["t_pf"] = time.monotonic()
            out = self.eng.prefill_step(r["slot"], self.chunk)
            if out is not None:
                self.wait_q.popleft()
                key = (r["ss"], r["rid"])
                self.active[key] = {"slot": r["slot"], "out": [], "text": "", "sent": 0, "next": None, "n": 0,
                                    "t0": r["t0"], "trace": {"n_prompt": len(r["ids"]), "n_prefill": r.get("n_prefill"),
                                                             "after_tool": r.get("after_tool"),
                                                             "queue_ms": round((r["t_pf"] - r["t0"]) * 1e3, 1),
                                                             "prefill_ms": round((time.monotonic() - r["t_pf"]) * 1e3, 1)}}
                self._emit(key, int(out.argmax()))
            return True
        return False

    def _emit(self, key, tok):
        a = self.active[key]
        if a["n"] == 0:
            a["first_ms"] = round((time.monotonic() - a["t0"]) * 1e3, 1)
        if tok in set(self.eng.eos_ids) or a["n"] >= self.max_new:
            self._log_end(a, "eos")
            self.box.put(key[0], {"ev": "end", "ss": key[0], "rid": key[1], "n": a["n"], "first_ms": a.get("first_ms")})
            del self.active[key]
            return
        a["out"].append(tok)
        a["n"] += 1
        a["next"] = tok
        text = self.eng.tok.decode(a["out"], skip_special_tokens=False)
        if not text.endswith("�") and len(text) > a["sent"]:
            self.box.put(key[0], {"ev": "delta", "ss": key[0], "rid": key[1], "text": text[a["sent"]:]})
            a["sent"] = len(text)
        if "</tool_call>" in text:
            self._log_end(a, "call")
            self.box.put(key[0], {"ev": "end", "ss": key[0], "rid": key[1], "n": a["n"], "first_ms": a.get("first_ms")})
            del self.active[key]

    def _log_end(self, a, kind):
        """Per-request timeline (submitted -> prefill start -> first token -> end), for the scheduler summary."""
        tr = dict(a.get("trace", {}))
        tr.update(kind=kind, first_ms=a.get("first_ms"), total_ms=round((time.monotonic() - a["t0"]) * 1e3, 1),
                  n_tok=a["n"])
        self.log.append(tr)


def _req_stats(log):
    import numpy as np
    out = {}
    for kind, sel in (("call", lambda r: r.get("kind") == "call"),
                      ("reply_after_tool", lambda r: r.get("kind") == "eos" and r.get("after_tool")),
                      ("reply", lambda r: r.get("kind") == "eos" and not r.get("after_tool"))):
        rs = [r for r in log if sel(r)]
        if not rs:
            continue
        d = {"n": len(rs)}
        for f in ("queue_ms", "prefill_ms", "first_ms", "total_ms", "n_prefill", "n_tok"):
            v = [r[f] for r in rs if r.get(f) is not None]
            if v:
                d[f] = {"p50": round(float(np.percentile(v, 50)), 1), "p95": round(float(np.percentile(v, 95)), 1)}
        out[kind] = d
    return out


class ThreadedLLM(threading.Thread):
    """The LLM part of the one-process engine on its own thread and CUDA stream: requests and cancels go to its queue,
    it runs batched decode steps (CUDA graphs) and prefill chunks back to back. Ears and talker keep the scheduler
    thread and the default stream; the GPU interleaves both streams, so decode fills the gaps the ears' host-side
    work leaves (with one thread the LLM got a step only every few hundred ms under 10 sessions). Same process, same
    weights, same per-session isolation; the scheduler still owns ears and talker deadlines."""
    def __init__(self, inner: "InProcLLM", budget_ms: float = 25.0):
        import torch
        self.__dict__["inner"] = inner
        super().__init__(daemon=True, name="f2-llm")
        self.budget_ms = budget_ms
        self.q: queue.Queue = queue.Queue()
        self.ev = threading.Event()
        # own stream, default priority: stream priorities (ears high, LLM low) were tried on GB10 and made the ears'
        # rounds slower, not faster
        self.stream = torch.cuda.Stream()
        self.stop_flag = False
        self.err = None
        self.busy_s = 0.0

    def __getattr__(self, k):              # log, wait_max, slots, box, ... of the wrapped InProcLLM
        inner = self.__dict__.get("inner")
        if inner is None:
            raise AttributeError(k)
        return getattr(inner, k)

    def send(self, m):
        if m.get("op") == "cancel":
            with self.inner.lock:
                self.inner.cancelled.add((m["ss"], m["rid"]))
        else:
            self.q.put(m)
        self.ev.set()

    def control(self, m):
        self.q.put(m)
        self.ev.set()

    def poll(self, ss):
        return self.inner.poll(ss)

    def close(self):
        self.stop_flag = True
        self.ev.set()

    def run(self, *args, **kw):            # Thread.run: the worker loop (the scheduler never calls this)
        import torch
        with torch.inference_mode(), torch.cuda.stream(self.stream):
            while not self.stop_flag:
                try:
                    while True:
                        try:
                            self.inner.control(self.q.get_nowait())
                        except queue.Empty:
                            break
                    t = time.monotonic()
                    did = self.inner.run(self.budget_ms)
                    if did:
                        self.busy_s += time.monotonic() - t
                    else:
                        self.ev.wait(0.004)
                        self.ev.clear()
                except Exception as e:  # noqa: BLE001 - keep the LLM thread alive, report
                    import traceback
                    traceback.print_exc()
                    self.err = repr(e)


class Scheduler(threading.Thread):
    def __init__(self, talker_budget_ms: float = 40.0, llm_budget_ms: float = 60.0):
        super().__init__(daemon=True, name="f2-gpu")
        self.llm_budget_ms = llm_budget_ms
        self.inbox: queue.Queue = queue.Queue()
        self.cv = threading.Event()
        self.ears = self.talker = self.llm = None
        self.talker_budget_ms = talker_budget_ms
        self.stop_flag = False
        self.stats = {"turns": 0, "ears": 0, "talker": 0, "llm": 0, "ears_ms": [], "llm_ms": [], "talker_ms": []}
        self.err = None

    def submit(self, item):
        self.inbox.put(item)
        self.cv.set()

    def kick(self):
        self.cv.set()

    def summary(self) -> dict:
        import numpy as np

        def pc(x):
            return {"n": len(x), "p50": round(float(np.percentile(x, 50)), 1), "p95": round(float(np.percentile(x, 95)), 1),
                    "max": round(float(np.max(x)), 1)} if x else {"n": 0}
        e = self.ears.e if self.ears is not None else None
        tot = {k: round(sum(self.stats[k]) / 1e3, 1) for k in ("ears_ms", "talker_ms", "llm_ms")}
        return {"turns": self.stats["turns"], "busy_s": tot, "ears_ms": pc(self.stats["ears_ms"]), "talker_ms": pc(self.stats["talker_ms"]),
                "llm_ms": pc(self.stats["llm_ms"]), "idle_s": round(self.stats.get("idle_s", 0.0), 1),
                "ctl": {k: {"n": len(v), "sum_s": round(sum(v) / 1e3, 2), "max_ms": round(max(v), 1)}
                        for k, v in self.stats.items() if k.startswith("ctl_")},
                "llm_wait_q_max": getattr(self.llm, "wait_max", None),
                "llm_thread_busy_s": round(getattr(self.llm, "busy_s", 0.0), 1) if isinstance(self.llm, ThreadedLLM) else None,
                "llm_requests": _req_stats(getattr(self.llm, "log", [])), "ears_batch": pc(getattr(e, "batch_sizes", [])), "error": self.err,
                **self.mem()}

    def mem(self) -> dict:
        import torch
        if not torch.cuda.is_available():
            return {}
        return {"cuda_allocated_gb": round(torch.cuda.memory_allocated() / 2 ** 30, 3),
                "cuda_reserved_gb": round(torch.cuda.memory_reserved() / 2 ** 30, 3),
                "cuda_peak_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 3)}

    def call(self, fn, timeout=600.0):
        """Run ``fn()`` on the GPU thread (model loads, voice states) and return its result."""
        done = threading.Event()
        box = {}

        def job():
            try:
                box["r"] = fn()
            except Exception as e:  # noqa: BLE001
                box["e"] = e
            done.set()
        self.submit(("call", job))
        if not done.wait(timeout):
            raise TimeoutError("scheduler call")
        if "e" in box:
            raise box["e"]
        return box.get("r")

    def run(self):
        import torch
        with torch.inference_mode():
            while not self.stop_flag:
                try:
                    did = False
                    while True:
                        try:
                            kind, m = self.inbox.get_nowait()
                        except queue.Empty:
                            break
                        tc = time.monotonic()
                        if kind == "call":
                            m()
                        else:
                            getattr(self, kind).control(m)
                        key = f"ctl_{kind}_{m.get('op') if isinstance(m, dict) else 'call'}"
                        self.stats.setdefault(key, []).append((time.monotonic() - tc) * 1e3)
                        did = True
                    t = time.monotonic()
                    if self.ears is not None and self.ears.run():
                        self.stats["ears"] += 1
                        self.stats["ears_ms"].append((time.monotonic() - t) * 1e3)
                        did = True
                    t = time.monotonic()
                    if self.talker is not None and self.talker.run(self.talker_budget_ms):
                        self.stats["talker"] += 1
                        self.stats["talker_ms"].append((time.monotonic() - t) * 1e3)
                        did = True
                    if self.ears is not None and self.ears.due():
                        continue                              # a cohort of hops is due: ears first
                    t = time.monotonic()
                    if (self.llm is not None and not isinstance(self.llm, ThreadedLLM)
                            and self.llm.run(self.llm_budget_ms, stop=self.ears.due if self.ears else None)):
                        self.stats["llm"] += 1
                        self.stats["llm_ms"].append((time.monotonic() - t) * 1e3)
                        did = True
                    self.stats["turns"] += 1
                    if not did:
                        ti = time.monotonic()
                        self.cv.wait(0.003)
                        self.cv.clear()
                        self.stats["idle_s"] = self.stats.get("idle_s", 0.0) + time.monotonic() - ti
                except Exception as e:  # noqa: BLE001 - keep the GPU thread alive, report
                    import traceback
                    traceback.print_exc()
                    self.err = repr(e)


# ------------------------------------------------------------------------------------------------ building it
def start(a, log_dir=None) -> dict:  # pragma: no cover - GPU
    """Load every model in THIS process, on the scheduler's thread, and return the in-process backends:
    {"ears": InProcEars, "talker": InProcTalker, "llm": InProcLLM | None, "sched": Scheduler, "ready": {...}}.
    Arguments (server/build.py): --ears-bundle, --pocket-weights / --pocket-tokenizers, --qwen-model / --qwen-mode /
    --qwen-slots, --max-sessions."""
    import resource
    import torch
    from pathlib import Path
    torch.backends.mkldnn.enabled = False        # aarch64 oneDNN aborts the process on some Pocket CPU convs
    sched = Scheduler(talker_budget_ms=getattr(a, "talker_budget_ms", 40.0))
    sched.start()
    dev = getattr(a, "ears_device", "cuda")
    slots = max(4, int(getattr(a, "max_sessions", 4)), int(getattr(a, "ws_n", 0) or 0) + 2)
    ready: dict = {}

    def rss():
        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20, 2)

    def load_ears():
        from .ears.own import OwnEars
        t0 = time.monotonic()
        e = OwnEars(a.ears_bundle, device=dev, slots=slots, window_precision=getattr(a, "ears_window", "bf16"))
        t1 = time.monotonic()
        ng = e.warm_window_graphs(2 * slots) if str(getattr(a, "ears_graphs", "off")) == "on" else 0
        return e, {"load_s": round(time.monotonic() - t0, 1), "window_graphs": ng,
                   "window_graphs_s": round(time.monotonic() - t1, 1), **sched.mem(), "rss_peak_gb": rss()}
    ears_m, ready["ears"] = sched.call(load_ears)
    ears = InProcEars(ears_m, sched)
    ears.ready = {"ev": "ready", **ready["ears"]}
    sched.ears = ears

    LANGS = {"it": "italian_24l", "fr": "french_24l", "en": "english_2026-09_24l", "es": "spanish_24l",
             "de": "german_24l", "pt": "portuguese_24l"}

    def make_pocket(lang):
        from .tts.pocket_own import PocketOwn
        root = Path(a.pocket_weights)                    # a snapshot dir, or the repo's snapshots dir (any snapshot)
        w = next(iter(sorted(root.glob(f"languages/{LANGS[lang]}/model.safetensors"))
                      or sorted(root.glob(f"*/languages/{LANGS[lang]}/model.safetensors"))
                      or sorted(root.parent.glob(f"*/languages/{LANGS[lang]}/model.safetensors"))))
        tk = sorted(Path(a.pocket_tokenizers).glob(f"*/languages/{LANGS[lang]}/tokenizer.json")) or \
            sorted(Path(a.pocket_tokenizers).glob(f"languages/{LANGS[lang]}/tokenizer.json"))
        dt = {"bf16": torch.bfloat16, "fp32": torch.float32}[getattr(a, "pocket_dtype", "bf16")]
        wq = getattr(a, "pocket_wq", "w4")
        m = PocketOwn(w, tk[0], lang=lang, device=getattr(a, "talker_device", "cuda"), dtype=dt,
                      wq="" if wq in ("", "none") else wq, share_mimi=pocket_first.get("m"))
        pocket_first.setdefault("m", m)                  # Mimi (codec + voice encoder) loaded once for all languages
        return m
    pocket_first: dict = {}
    vs = dict(x.split("=", 1) for x in getattr(a, "voice", []) or [])
    talker = InProcTalker(sched, make_pocket, slots=slots, default_voice=next(iter(vs.values()), ""))
    sched.talker = talker
    for lang in [x for x in getattr(a, "pocket_preload", "").split(",") if x]:   # models + default voice, now
        t0 = time.monotonic()
        sched.call(lambda lang=lang: talker._voice(lang, ""))
        ready[f"talker_{lang}"] = {"load_s": round(time.monotonic() - t0, 1), **sched.mem(), "rss_peak_gb": rss()}
    llm = None
    if getattr(a, "qwen_model", ""):
        def load_llm():
            from .llm.qwen_own.engine import QwenOwn
            t0 = time.monotonic()
            q = QwenOwn(a.qwen_model, device=dev, mode=a.qwen_mode, slots=max(slots, int(getattr(a, "qwen_slots", 0))), max_len=getattr(a, "qwen_max_len", 8192))
            return q, {"load_s": round(time.monotonic() - t0, 1), **q.footprint(), **sched.mem(), "rss_peak_gb": rss()}
        q, ready["llm"] = sched.call(load_llm, timeout=1800)
        inner = InProcLLM(sched, q)
        if str(getattr(a, "llm_thread", "on")) != "off":
            t0 = time.monotonic()
            n = sched.call(lambda: q.warm_graphs(), timeout=1800)     # all captures before any session runs
            ready["llm"].update(graphs_captured=n, graphs_s=round(time.monotonic() - t0, 1))
            llm = ThreadedLLM(inner)
            q.stream = llm.stream          # all of qwen_own's GPU work on the LLM's own stream
            llm.start()
        else:
            llm = inner
        llm.ready = {"ev": "ready", **ready["llm"]}
        sched.llm = llm
    talker.ready = {"ev": "ready", "inproc": True}
    return {"ears": ears, "talker": talker, "llm": llm, "sched": sched, "ready": ready}
