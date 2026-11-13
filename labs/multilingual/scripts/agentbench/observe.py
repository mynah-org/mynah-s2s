"""Observational instrumentation of the NVIDIA VoiceChat wrapper (instance-level wrappers; no V2/NeMo file edited).

Records, per model frame:
  * forcing  : agent token BEFORE and AFTER `_maybe_apply_forced_turn_taking` / `_apply_rnnt_turn_taking`, the RNNT
               blank flag and the RNNT turn-taking counters (blank_count, nonblank_consec, speech_confirmed,
               agent_speaking, first_turn) -> which BOS/EOS were written by the controller, not by the Thinker;
  * logits   : log-probabilities of the control tokens (text: PAD/BOS/EOS; function: PAD/SOTC/EOTC) and the top-k of
               both heads, for every single-frame Thinker call (main loop AND FC fast path), keyed by `current_step`;
  * features : norm of the Thinker's user-audio embedding and its cosine to the session's FC silence embedding.

Raw signals only: none of them is a calibrated probability of understanding.
`install(wrapper, rec, what=...)` returns an `uninstall()` callable; `rec` is any object with `.append(dict)`.
"""
from __future__ import annotations

import torch

PAD, BOS, EOS, SOTC, EOTC = 12, 1, 2, 20, 21
RNNT_KEYS = ("blank_count", "nonblank_consec", "nonblank_total", "speech_confirmed", "agent_speaking", "first_turn")


def _scalar(v):
    try:
        return v[0].item() if hasattr(v, "__getitem__") and hasattr(v[0], "item") else (v.item() if hasattr(v, "item") else v)
    except Exception:  # noqa: BLE001 - observation must never break the run
        return None


def control_logprobs(text_logits: torch.Tensor, func_logits: torch.Tensor | None, topk: int = 5) -> dict:
    """text_logits / func_logits: [V] for one frame."""
    lt = torch.log_softmax(text_logits.float(), -1)
    out = {"text": {"PAD": float(lt[PAD]), "BOS": float(lt[BOS]), "EOS": float(lt[EOS])},
           "text_top": [[int(i), round(float(v), 3)] for v, i in zip(*lt.topk(topk))]}
    if func_logits is not None:
        lf = torch.log_softmax(func_logits.float(), -1)
        out["func"] = {"PAD": float(lf[PAD]), "SOTC": float(lf[SOTC]), "EOTC": float(lf[EOTC])}
        out["func_top"] = [[int(i), round(float(v), 3)] for v, i in zip(*lf.topk(topk))]
    return out


def install(w, rec, what=("forcing", "logits", "features")) -> callable:
    """w = the NVIDIA inference wrapper (engine.wrapper)."""
    undo = []
    if "forcing" in what:
        for name in ("_maybe_apply_forced_turn_taking", "_apply_rnnt_turn_taking"):
            orig = getattr(w, name)

            def wrapped(t, gen_text, *args, _orig=orig, _name=name, **kw):
                pre = int(gen_text[0, t])
                out = _orig(t, gen_text, *args, **kw)
                try:
                    post = int(gen_text[0, t])
                    row = {"kind": "forcing", "fn": _name, "t": int(t), "pre": pre, "post": post,
                           "forced": pre != post}
                    st = kw.get("rnnt_state") if _name.startswith("_maybe") else (args[1] if len(args) > 1 else None)
                    if _name == "_apply_rnnt_turn_taking" and args:
                        row["is_blank"] = bool(_scalar(args[0]))
                    if isinstance(st, dict):
                        row["rnnt"] = {k: _scalar(st[k]) for k in RNNT_KEYS if k in st}
                    rec.append(row)
                except Exception:  # noqa: BLE001
                    pass
                return out
            setattr(w, name, wrapped)
            undo.append(lambda n=name, o=orig: setattr(w, n, o))
    if "logits" in what:
        stt = w.model.stt_model
        holder = {}
        orig_fwd = stt.forward

        def fwd(*a, **kw):
            ans = orig_fwd(*a, **kw)
            try:
                tl = ans["text_logits"]
                if tl is not None and tl.shape[1] == 1:
                    fl = ans.get("function_logits")
                    holder["last"] = (tl[0, -1].detach(), None if fl is None else fl[0, -1].detach())
            except Exception:  # noqa: BLE001
                pass
            return ans
        stt.forward = fwd
        undo.append(lambda: setattr(stt, "forward", orig_fwd))
        orig_if = w.model_llm_interface

        def llm_if(*a, _o=orig_if, **kw):
            holder.pop("last", None)
            ans = _o(*a, **kw)
            try:
                if "last" in holder:
                    t, f = holder.pop("last")
                    rec.append({"kind": "logits", "t": kw.get("current_step"),
                                "pred_text": int(ans["predicted_token"][0]),
                                "pred_func": int(ans["function_predicted_token"][0]) if "function_predicted_token" in ans else None,
                                **control_logprobs(t, f)})
            except Exception:  # noqa: BLE001
                pass
            return ans
        w.model_llm_interface = llm_if
        undo.append(lambda: setattr(w, "model_llm_interface", orig_if))
    if "features" in what:
        fm = w.fusion_module
        orig_fu = fm.forward
        cnt = {"n": 0}

        def fuse(*a, **kw):
            try:
                ua = kw.get("user_audio_embeds")
                if ua is not None and ua.dim() == 3 and ua.shape[1] == 1:
                    sil = getattr(w, "_silence_embedding_cache", None)
                    v = ua[0, 0].float()
                    row = {"kind": "feature", "i": cnt["n"], "norm": float(v.norm())}
                    if sil is not None:
                        row["cos_silence"] = float(torch.nn.functional.cosine_similarity(v, sil.reshape(-1).float(), 0))
                    rec.append(row)
                    cnt["n"] += 1
            except Exception:  # noqa: BLE001
                pass
            return orig_fu(*a, **kw)
        fm.forward = fuse
        undo.append(lambda: setattr(fm, "forward", orig_fu))

    def uninstall():
        for u in reversed(undo):
            u()
    return uninstall
