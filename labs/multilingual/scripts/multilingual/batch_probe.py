"""Throughput probe for corpus V2 data prep (no outputs kept beyond a JSON report + 8 sample WAVs):
(1) Qwen3-TTS CustomVoice: sequential vs batched generate_custom_voice (Ryan, IT/FR/EN);
(2) Nano-9B: batched greedy generation with left padding vs one-by-one (output identity check)."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
out = Path(sys.argv[1])
if out.exists():
    raise SystemExit("exists")
out.mkdir(parents=True)
rep = {}
mode = sys.argv[2]
if mode == "tts":
    from huggingface_hub import snapshot_download
    from qwen_tts import Qwen3TTSModel
    from render_qa import REPO, REV, LANG
    snap = snapshot_download(REPO, revision=REV, local_files_only=True)
    m = Qwen3TTSModel.from_pretrained(snap, device_map="cuda", dtype=torch.bfloat16, attn_implementation="sdpa")
    texts = ["Quante zampe ha un ragno?", "Qual è la capitale della Spagna e perché è famosa?",
             "Quelle est la plus grande planète du système solaire ?", "Combien font douze fois trois ?",
             "Ricordami di comprare il latte domani alle otto.", "Je m'appelle Claire, quel est mon prénom ?",
             "What year did the Berlin Wall fall?", "Non voglio il caffè, preferisco il tè: cosa mi consigli?"]
    langs = ["it", "it", "fr", "fr", "it", "fr", "en", "it"]
    torch.manual_seed(0)
    m.generate_custom_voice(text=texts[0], speaker="ryan", language=LANG["it"])  # warm-up
    torch.cuda.synchronize(); t = time.monotonic()
    seq = []
    for x, l in zip(texts, langs):
        torch.manual_seed(1)
        w, sr = m.generate_custom_voice(text=x, speaker="ryan", language=LANG[l])
        seq.append(len(w[0]) / sr)
    torch.cuda.synchronize(); rep["tts_seq_s"] = time.monotonic() - t
    import os
    for bs in [int(x) for x in os.environ.get("BATCHES", "8").split(",")]:
        k = bs // 8
        T, Lg = texts * k, langs * k
        torch.cuda.synchronize(); t = time.monotonic()
        torch.manual_seed(1)
        w, sr = m.generate_custom_voice(text=T, speaker=["ryan"] * bs, language=[LANG[l] for l in Lg])
        torch.cuda.synchronize(); rep[f"tts_batch{bs}_s"] = time.monotonic() - t
        rep[f"tts_batch{bs}_s_per_clip"] = rep[f"tts_batch{bs}_s"] / bs
        rep["batch_durs"] = [round(len(x) / sr, 2) for x in w[:8]]
        rep[f"batch{bs}_max_mem_gib"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
        for i, x in enumerate(w[:8]):
            sf.write(out / f"b{bs}_{i}_{langs[i]}.wav", np.asarray(x, dtype=np.float32), sr)
    rep["seq_durs"] = [round(d, 2) for d in seq]
    rep["audio_s"] = sum(seq)
else:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from gen_train_text import NANO
    tok = AutoTokenizer.from_pretrained(NANO)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.convert_ids_to_tokens(0)
    m = AutoModelForCausalLM.from_pretrained(NANO, torch_dtype=torch.bfloat16, trust_remote_code=True,
                                             device_map="cuda").eval()
    qs = ["Translate into Italian and French as JSON {\"it\":..,\"fr\":..}: How many legs does a spider have?",
          "Translate into Italian and French as JSON {\"it\":..,\"fr\":..}: Remind me to call my sister at six.",
          "Answer briefly: what is the capital of Spain?",
          "Answer briefly: how much is twelve times three?",
          "Write one short question a user could ask a voice assistant about trains.",
          "Answer briefly: which planet is the largest in the solar system?",
          "Translate into Italian and French as JSON {\"it\":..,\"fr\":..}: I don't want coffee, I prefer tea.",
          "Answer briefly: in which country is Lyon?"]
    def enc(q):
        return tok.apply_chat_template([{"role": "system", "content": "/no_think"}, {"role": "user", "content": q}],
                                       add_generation_prompt=True, tokenize=False)
    single = []
    torch.cuda.synchronize(); t = time.monotonic()
    for q in qs:
        ids = tok(enc(q), return_tensors="pt", add_special_tokens=False).to("cuda")
        g = m.generate(**ids, max_new_tokens=80, do_sample=False, pad_token_id=0, eos_token_id=tok.eos_token_id)
        single.append(tok.decode(g[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).strip())
    torch.cuda.synchronize(); rep["nano_single_s"] = time.monotonic() - t
    t = time.monotonic()
    b = tok([enc(q) for q in qs], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
    g = m.generate(**b, max_new_tokens=80, do_sample=False, pad_token_id=0, eos_token_id=tok.eos_token_id)
    batch = [tok.decode(x[b["input_ids"].shape[1]:], skip_special_tokens=True).strip() for x in g]
    torch.cuda.synchronize(); rep["nano_batch8_s"] = time.monotonic() - t
    rep["identical"] = [s == x for s, x in zip(single, batch)]
    rep["pairs"] = list(zip(single, batch))
(out / f"probe_{mode}.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
print(json.dumps(rep, indent=1, ensure_ascii=False))
