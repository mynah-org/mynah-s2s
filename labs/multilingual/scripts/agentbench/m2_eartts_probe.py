"""M2: zero-shot EarTTS probe in EN / IT / FR (GPU, <= 15 min with load).

Fixed agent texts are forced through VoiceChat's own EarTTS + codec exactly the way NVIDIA renders scripted speech
(on-hold reminders, `NemotronVoicechatInferenceWrapper._run_tts_reminder`): [BOS] + tokenizer ids + PAD tail
(NVIDIA's per-character rule) + [EOS], one token per 80 ms frame, starting from the checkpoint's pre-baked speaker
latent and TTS warm-up state, with a fresh codec streaming cache per clip and a fixed seed. No Thinker step is run,
so this isolates speech generation. Outputs <out>/<lang>_<id>/output.wav (OmniASR asr_dirs layout) + texts.json.
"""
from __future__ import annotations

import argparse
import json
import math
import queue
import time
from pathlib import Path

import soundfile as sf
import torch

from mynah_v2.model.reference import ReferenceEngine, build_cfg

TEXTS = {
 "greet": {"en": "Hello, thank you for calling Lindvik Home. How can I help you today?",
           "it": "Buongiorno, grazie per aver chiamato Lindvik Home. Come posso aiutarla oggi?",
           "fr": "Bonjour, merci d'avoir appelé Lindvik Home. Comment puis-je vous aider aujourd'hui ?"},
 "name": {"en": "Thank you, Mario Rossi. Could you also tell me your postcode, please?",
          "it": "Grazie, signor Mario Rossi. Mi può dire anche il suo codice postale, per favore?",
          "fr": "Merci, monsieur Mario Rossi. Pouvez-vous aussi me donner votre code postal, s'il vous plaît ?"},
 "rare_name": {"en": "Thank you, Siobhan Okonkwo-Laurent, I found your order.",
               "it": "Grazie, Siobhan Okonkwo-Laurent, ho trovato il suo ordine.",
               "fr": "Merci, Siobhan Okonkwo-Laurent, j'ai trouvé votre commande."},
 "date_tool": {"en": "Your Fjordby three-seat sofa is out for delivery and will arrive on Friday, November fourteenth.",
               "it": "Il suo divano Fjordby a tre posti è in consegna e arriverà venerdì quattordici novembre.",
               "fr": "Votre canapé trois places Fjordby est en cours de livraison et arrivera le vendredi quatorze novembre."},
 "price": {"en": "We have the Fjordby three-seat sofa for four hundred and ninety-nine euros.",
           "it": "Abbiamo il divano Fjordby a tre posti a quattrocentonovantanove euro.",
           "fr": "Nous avons le canapé trois places Fjordby à quatre cent quatre-vingt-dix-neuf euros."},
 "place": {"en": "The parcel is in our warehouse in Turin and will reach Milan tomorrow.",
           "it": "Il pacco è nel nostro magazzino di Torino e arriverà a Milano domani.",
           "fr": "Le colis est dans notre entrepôt de Lyon et arrivera à Paris demain."},
 "policy": {"en": "I'm sorry, I can't share information about other customers.",
            "it": "Mi dispiace, non posso condividere informazioni su altri clienti.",
            "fr": "Je suis désolée, je ne peux pas partager d'informations sur d'autres clients."},
 "transfer": {"en": "Of course, I'm transferring you to one of our colleagues now. Please stay on the line.",
              "it": "Certo, la sto trasferendo a un nostro collega. Resti in linea, per favore.",
              "fr": "Bien sûr, je vous transfère à un de nos collègues. Restez en ligne, s'il vous plaît."},
 "foreign": {"en": "You can track it on the website under My Orders, or reply to the SMS we sent you.",
             "it": "Può seguirlo sul sito nella sezione My Orders, oppure rispondere all'SMS che le abbiamo inviato.",
             "fr": "Vous pouvez le suivre sur le site dans la rubrique My Orders, ou répondre au SMS que nous vous avons envoyé."},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()
    out = Path(a.out)
    if (out / "texts.json").exists():
        raise SystemExit(f"{out} already holds results")
    out.mkdir(parents=True, exist_ok=True)
    eng = ReferenceEngine(build_cfg(output_dir=str(out / "_nemo")), llm_cache=True)
    w = eng.wrapper
    stt = w.model.stt_model
    from nemo.collections.speechlm2.inference.pipelines import streaming_s2s_pipeline as P
    from nemo.collections.speechlm2.modules.ear_tts_vae_codec import CausalConv1dCache
    # NVIDIA's constants are LOCAL to the pipeline constructor (streaming_s2s_pipeline.py:195-196: MIN 17, 0.5 per
    # char), not module attributes: the first M2 run read 0.0 and gave every clip only 17 PADs -> truncated speech.
    # 0.7/char is used here (deviation from 0.5, documented) so long IT/FR sentences are not cut by the probe itself.
    del P
    pad_min, pad_per_char = 17, 0.7
    sr = getattr(eng.pipeline, "output_sample_rate", 22050)
    meta = {"engine_load_s": eng.load_s, "pad_rule": [pad_min, pad_per_char], "seed": a.seed, "speaker": w.speaker_name,
            "clips": []}
    for tid, by_lang in TEXTS.items():
        for lang, text in by_lang.items():
            ids = list(w.tokenizer.text_to_ids(text))
            n_pad = max(pad_min, math.ceil(pad_per_char * len(text)))  # NVIDIA's rule for scripted speech
            toks = [stt.text_bos_id] + ids + [stt.text_pad_id] * n_pad + [stt.text_eos_id]
            torch.manual_seed(a.seed)
            torch.cuda.manual_seed_all(a.seed)
            st = {"code": w.first_tts_code_input.clone(), "past_key_values": w._clone_cache(w.first_tts_past_key_values_input),
                  "subword_mask": None, "codec_cache": CausalConv1dCache()}
            q = queue.Queue()
            t0 = time.monotonic()
            with torch.no_grad():
                n = w._run_tts_reminder(toks, st, tts_audio_output_queue=q)
            dt = time.monotonic() - t0
            chunks = []
            while not q.empty():
                chunks.append(q.get().reshape(-1))
            audio = torch.cat(chunks).float().numpy() if chunks else []
            d = out / f"{lang}_{tid}"
            d.mkdir(parents=True, exist_ok=True)
            sf.write(d / "output.wav", audio, sr)
            meta["clips"].append({"id": tid, "lang": lang, "text": text, "n_text_tokens": len(ids), "n_pad": n_pad,
                                  "samples": int(n), "audio_s": round(len(audio) / sr, 2), "gen_s": round(dt, 2)})
            print(lang, tid, len(ids), "tokens", round(len(audio) / sr, 2), "s audio", round(dt, 1), "s gen", flush=True)
    (out / "texts.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
