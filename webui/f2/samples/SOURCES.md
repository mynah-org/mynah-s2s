# Demo samples: sources and licences

Recorded with the Frankenstein v2 engine (frankenstein2/lab/demo_dialogue_check.py, packed by pack_samples.py):
`lines/` are the simulated caller's lines, `dialogues/` whole demo dialogues (caller and agent). Every voice is either
a predefined voice of the talker model or a reference voice offered for speech synthesis, cloned by the talker; no
voice was cloned from a recording made for another purpose.

The audio of a language is a derivative of its talker model and of the voices below: it is shared under the same
terms (CC-BY-4.0; CC-BY-SA-4.0 where a source is share-alike, such as the Estonian voices and talker).

## Voices

| lang | role | voice | source | licence |
|---|---|---|---|---|
| cs | agent | a published voice prompt | https://huggingface.co/vvolhejn/pocket-tts-czech | CC-BY-4.0 |
| cs | caller | a published voice prompt | https://huggingface.co/vvolhejn/pocket-tts-czech | CC-BY-4.0 |
| de | agent | peter_yearsley | Kyutai Pocket TTS predefined voice (kyutai/pocket-tts-without-voice-cloning, CC-BY-4.0) | CC-BY-4.0 |
| de | caller | paul | Kyutai Pocket TTS predefined voice (kyutai/pocket-tts-without-voice-cloning, CC-BY-4.0) | CC-BY-4.0 |
| en | agent | george | Kyutai Pocket TTS predefined voice (kyutai/pocket-tts-without-voice-cloning, CC-BY-4.0) | CC-BY-4.0 |
| en | caller | marius | Kyutai Pocket TTS predefined voice (kyutai/pocket-tts-without-voice-cloning, CC-BY-4.0) | CC-BY-4.0 |
| es | agent | rafael | Kyutai Pocket TTS predefined voice (kyutai/pocket-tts-without-voice-cloning, CC-BY-4.0) | CC-BY-4.0 |
| es | caller | cosette | Kyutai Pocket TTS predefined voice (kyutai/pocket-tts-without-voice-cloning, CC-BY-4.0) | CC-BY-4.0 |
| et | agent | a published voice prompt | https://huggingface.co/cbentes/pocket-tts-estonian | CC-BY-SA-4.0 |
| et | caller | a published voice prompt | https://huggingface.co/cbentes/pocket-tts-estonian | CC-BY-SA-4.0 |
| fr | agent | eve | Kyutai Pocket TTS predefined voice (kyutai/pocket-tts-without-voice-cloning, CC-BY-4.0) | CC-BY-4.0 |
| fr | caller | fantine | Kyutai Pocket TTS predefined voice (kyutai/pocket-tts-without-voice-cloning, CC-BY-4.0) | CC-BY-4.0 |
| hi | agent | a donated voice | https://huggingface.co/kyutai/tts-voices (voice-donations) | CC0-1.0 |
| hi | caller | a donated voice | https://huggingface.co/kyutai/tts-voices (voice-donations) | CC0-1.0 |
| it | agent | giovanni | Kyutai Pocket TTS predefined voice (kyutai/pocket-tts-without-voice-cloning, CC-BY-4.0) | CC-BY-4.0 |
| it | caller | charles | Kyutai Pocket TTS predefined voice (kyutai/pocket-tts-without-voice-cloning, CC-BY-4.0) | CC-BY-4.0 |
| ja | agent | a published voice prompt | https://huggingface.co/takuma104/pocket-tts-ja-cc | CC-BY-4.0 |
| ja | caller | a published voice prompt | https://huggingface.co/takuma104/pocket-tts-ja-cc | CC-BY-4.0 |
| ko | agent | a donated voice | https://huggingface.co/kyutai/tts-voices (voice-donations) | CC0-1.0 |
| ko | caller | a donated voice | https://huggingface.co/kyutai/tts-voices (voice-donations) | CC0-1.0 |
| nl | agent | caro_davy | Kyutai Pocket TTS predefined voice (kyutai/pocket-tts-without-voice-cloning, CC-BY-4.0) | CC-BY-4.0 |
| nl | caller | rafael | Kyutai Pocket TTS predefined voice (kyutai/pocket-tts-without-voice-cloning, CC-BY-4.0) | CC-BY-4.0 |
| pl | agent | a donated voice | https://huggingface.co/kyutai/tts-voices (voice-donations) | CC0-1.0 |
| pl | caller | a donated voice | https://huggingface.co/kyutai/tts-voices (voice-donations) | CC0-1.0 |
| pt | agent | paul | Kyutai Pocket TTS predefined voice (kyutai/pocket-tts-without-voice-cloning, CC-BY-4.0) | CC-BY-4.0 |
| pt | caller | giovanni | Kyutai Pocket TTS predefined voice (kyutai/pocket-tts-without-voice-cloning, CC-BY-4.0) | CC-BY-4.0 |
| ru | agent | a donated voice | https://huggingface.co/kyutai/tts-voices (voice-donations) | CC0-1.0 |
| ru | caller | a donated voice | https://huggingface.co/kyutai/tts-voices (voice-donations) | CC0-1.0 |
| tr | agent | a donated voice | https://huggingface.co/kyutai/tts-voices (voice-donations) | CC0-1.0 |
| tr | caller | a donated voice | https://huggingface.co/kyutai/tts-voices (voice-donations) | CC0-1.0 |

## Talker models

| lang | model | licence |
|---|---|---|
| cs | https://huggingface.co/vvolhejn/pocket-tts-czech | CC-BY-4.0 |
| de | https://huggingface.co/kyutai/pocket-tts-without-voice-cloning | CC-BY-4.0 |
| en | https://huggingface.co/kyutai/pocket-tts-without-voice-cloning | CC-BY-4.0 |
| es | https://huggingface.co/kyutai/pocket-tts-without-voice-cloning | CC-BY-4.0 |
| et | https://huggingface.co/cbentes/pocket-tts-estonian | CC-BY-SA-4.0 |
| fr | https://huggingface.co/kyutai/pocket-tts-without-voice-cloning | CC-BY-4.0 |
| hi | https://huggingface.co/saryps-labs/pocket-tts-hindi | CC-BY-4.0 |
| it | https://huggingface.co/kyutai/pocket-tts-without-voice-cloning | CC-BY-4.0 |
| ja | https://huggingface.co/takuma104/pocket-tts-ja-cc | CC-BY-4.0 |
| ko | https://huggingface.co/seastar105/pocket-tts-korean-300m | CC-BY-4.0 |
| nl | https://huggingface.co/kyutai/pocket-tts-without-voice-cloning | CC-BY-4.0 |
| pl | https://huggingface.co/shefowl/pocket-tts-polish-6l | CC-BY-4.0 |
| pt | https://huggingface.co/kyutai/pocket-tts-without-voice-cloning | CC-BY-4.0 |
| ru | https://huggingface.co/ArtShtorm/Shtorm-PocketTTS-RU | CC-BY-4.0 |
| tr | https://huggingface.co/alperiox/pocket-tts-turkish | MIT |
