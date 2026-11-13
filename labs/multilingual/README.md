# labs/multilingual — VoiceChat IT/FR adaptation lab

Research lab that adapts NVIDIA VoiceChat-11B (EN full-duplex speech-to-speech with function calling) to Italian and
French. The goal is to keep duplex behaviour, barge-in, turn-taking, function calls and speech output. The lab
**consumes** the engine in `v2/`; it does not contain a copy of it and does not patch it.

Status: research code, migrated as-is from the lab's working tree. Not a supported part of the engine. Results,
notes and checkpoints are not in this repository.

## What is here
```
labs/multilingual/
  scripts/multilingual/   training, evaluation, diagnostics, data / scenario generators (flat, module names unchanged)
  scripts/agentbench/     voice-agent product benchmark: tenant scenarios, mock backend, deterministic scoring
  tests/                  checks of this directory itself (provenance)
  PROVENANCE.tsv          every file: source path and blob in the lab history, sha256, identical / edited (and why)
  requirements-lab.txt    optional extras for the lab only
  Makefile                `make -C labs/multilingual test`
```
The two script directories must stay siblings: `agentbench` imports `../multilingual` and vice versa.

### Component index (`scripts/multilingual/` unless noted)
| area | files |
|---|---|
| bridge / encoder transplant | `encoders`, `perception_swap`, `adapters`, `fit_map`, `asr_ckpt_diff`, `feat_drift`, `probe_adapter` |
| training | `session_distill` (bridge, Thinker q/v LoRA, joint, masked teacher KL), `thinker_lora`, `distill_adapter`, `teacher_kl`, `batch_sequence` |
| free-running evaluation | `glex_free` (duplex greedy decode with the FC fast path, call scoring, opener / BOS / SOTC behaviour), `glex_free_report`, `cycle_report`, `entity_table`, `glex_probe`, `cand_test`, `cand_report*` |
| diagnostics | `a0_rnnt_readout`, `a0_report`, `a1_text_oracle`, `a1_report`, `a1b_forced_call`, `v0_user_text`, `v1_gate`, `opener_probe`, `l1_smoke`, `b0_cpu_check`, `thinker_lora_cpu_check`, `d1_*`, `batch_probe`, `text_oracle`, `e5_gate` |
| data / scenario generators | `session_corpus*`, `corpus_v2*`, `corpus_v21`, `lexicon_v2`, `postcode_v2b`, `gen_train_text`, `union_manifest`, `merge_*`, `apply_round`, `sample_ids`, `render_qa`, `listen_pack`, `coverage_audit` |
| QA era (kept, not maintained) | `gold_v21`, `score_v2*`, `judge_v2`, `verify_asr_v2`, `asr_qa`, `e2e_qa`, `qa_corpus.json`, `v2*_eval_*.json`, `v2_report`, `pretrain_report` |
| audits | `audit_nvidia_prs`, `audit_transfer_timing`, `llm_weight_diff`, `llm_text_bench`, `check_nocall_replies` |
| ops | `drop_cache` (page-cache drop before a large load) |
| `scripts/agentbench/` | `agentbench`, `domains.json`, `tenant_*`, `m0_runner`, `observe`, `live_timeline`, `trace_*`, `t1_probe`, `t2_*`, `guided_*`, `ecal_probe`, `m1_parity_gate`, `m2_eartts_probe`, `rebaseline_report`, `fix_eval`, `dv3_gates` |

## Engine contract
- The lab imports only public engine APIs: `mynah_v2.model.reference` (`ReferenceEngine`, `build_cfg`,
  `DEFAULT_NEMO`), `mynah_v2.model.llm_cache`, `mynah_v2.model.tokens`, `mynah_v2.runtime.session`,
  `mynah_v2.agent.{tools,controller}` and `mynah_v2.tracing.events`; plus NeMo Speech at the commit the engine pins.
- **Training and free-running evaluation** load the engine and then call the Thinker directly. They never use
  the NeMo streaming FC hand-back.
- **Live sessions** (`agentbench/m0_runner` and the probes that use `Session`) run through the engine's FC hand-back
  fix `fc_ctx_fix`, which resumes after a call with (PAD, PAD). The lab's older live results used a NeMo patch that
  resumed with (PAD, EOTR). They are kept apart and are not comparable.
  - `m0_runner`'s `S2S_FIX_*` flags are **legacy**: they act only on that frozen patch and do nothing on a clean NeMo.
    On this engine, run it with `--no-fc-fix`.
  - Every live record now carries `runtime_fix.engine_fc_ctx_fix` and `runtime_fix.resume_contract`
    (`agentbench/live_gates.py`). A run with both fixes active is refused.
  - `live_gates.m07_chain` is the mandatory live regression: EN m07 must keep the legitimate second call
    (`find_customer` → `prepare_return`).
  - **The live path is not yet re-validated on this engine.**
- **Adapters and checkpoints** trained before the migration were produced with an earlier engine snapshot. Each one is
  declared compatible only after the construction-parity and bit-exact checks pass.

## Paths
Defaults follow the lab's machine layout under `~/mynah-s2s` (`ml/data`, `ml/ckpt`, `ml/results`, `hf-cache`). Most
scripts take explicit `--data` / `--out` / checkpoint arguments. The defaults do **not** read `MYNAH_HOME` yet.

## Install and test
Inference-only users install nothing from here: the engine's `v2/pyproject.toml` is unchanged.
Lab extras: `pip install -r labs/multilingual/requirements-lab.txt` into the engine venv. Speech rendering and
evaluation ASR reuse the engine's optional environments (`v2/requirements-tts.txt`, `v2/requirements-asr.txt`).

```
make -C labs/multilingual test     # CPU unit tests of the lab + provenance check (engine venv, PYTHONPATH = v2/src)
```
Some tests need the engine venv with NeMo Speech importable. Tests that need private artefacts skip themselves.
