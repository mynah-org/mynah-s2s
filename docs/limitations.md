# Limitations

- **Experimental.** Not production-ready; APIs and flags change.
- **English only** for VoiceChat; multilingual adaptation (e.g. Italian, French) is an open research question.
- **Model limits** (from the model card and our tests): short trained context (~2 minutes of audio), few tools per
  session, argument extraction errors on IDs and numbers, no barge-in while a tool runs.
- **Answers cut short after a tool call.** Occasionally the agent starts the answer after a lookup and stops mid-sentence (e.g. "Your order has been"), with the correct call and result. Seen in single and multi-session runs, with every preset including NVIDIA's reference step; under investigation.
- **Order IDs heard wrong.** Spelled IDs are sometimes misheard (an extra or missing letter); the tool then correctly reports "not found". Results depend on the synthetic voice, so a re-rendered test set can score differently item by item.
- **Spoken vs internal text.** Correct internal text does not guarantee correct speech; always check the audio.
- **Small samples, synthetic speech, mock tools.** Our gates use small synthetic sets; treat results as indicative.
- **Multi-session real time** is reached only without tool calls on an RTX 6000 Ada; tool-call cycles still cost
  latency (fc_merge trades tail latency for a longer call cycle).
- **Quantization** below BF16 beyond the int8 MLP showed behavioural regressions in our tests and is not used.
- **Hardware.** One VoiceChat instance needs ~42 GB GPU memory at peak; it does not fit 24 GB cards.
