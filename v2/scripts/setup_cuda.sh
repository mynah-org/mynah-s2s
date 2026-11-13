#!/usr/bin/env bash
# One-time setup on a Linux x86_64 or aarch64 machine with an NVIDIA GPU and a CUDA 13.x toolkit.
# Creates the MYNAH_HOME layout used by env.sh and the code defaults (see env.sh). Versions are the ones we tested
# (requirements-pinned.txt). Downloads ~42 GB of model weights from Hugging Face: read NOTICE.md (model license) first.
#   usage: setup_cuda.sh [--with-tts] [--with-asr] [--all] [--only tts|asr]
#     default         core runtime only: engine venv, NeMo Speech, CUDA kernels, VoiceChat weights
#     --with-tts      + synthetic test-audio TTS (Qwen3-TTS 0.6B; needed by make corpus / demo / e2e / multi / custom)
#     --with-asr      + evaluation ASR (Omnilingual ASR CTC 3B v2, own venv; the audio check of the test suites)
#     --all           all three;  --only tts|asr  add just that part to an existing setup (no venv rebuild)
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
DO_ENGINE=1 DO_TTS=0 DO_ASR=0
while [ $# -gt 0 ]; do
  case $1 in
    --with-tts) DO_TTS=1 ;;
    --with-asr) DO_ASR=1 ;;
    --all) DO_TTS=1 DO_ASR=1 ;;
    --only) shift; DO_ENGINE=0 DO_TTS=0 DO_ASR=0; case ${1:-} in tts) DO_TTS=1 ;; asr) DO_ASR=1 ;; *) echo "--only tts|asr"; exit 2 ;; esac ;;
    *) echo "unknown option $1 (see the header of this script)"; exit 2 ;;
  esac; shift
done
export MYNAH_HOME=${MYNAH_HOME:-$HOME/mynah-s2s}
mkdir -p $MYNAH_HOME/third_party/agent $MYNAH_HOME/build $MYNAH_HOME/data/agent
cd $MYNAH_HOME
export HF_HUB_CACHE=$MYNAH_HOME/hf-cache UV_CACHE_DIR=$MYNAH_HOME/uv-cache
CUDA_HOME=${CUDA_HOME:-$(ls -d /usr/local/cuda-13* 2>/dev/null | sort -V | tail -1)}
export CUDA_HOME PATH=$CUDA_HOME/bin:$HOME/.local/bin:$PATH        # ~/.local/bin = default location of the uv installer
for tool in uv uvx git nvcc nvidia-smi python3; do
  command -v $tool >/dev/null || { echo "missing prerequisite: $tool (see README Requirements)"; exit 1; }
done
ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)   # e.g. 89 (Ada/L40S), 121 (GB10)
echo "MYNAH_HOME=$MYNAH_HOME CUDA_HOME=$CUDA_HOME sm_$ARCH"

HFC="uvx --from huggingface_hub[cli]==0.34.4 huggingface-cli"
if [ $DO_ENGINE = 1 ]; then
echo "== NVIDIA NeMo Speech @097dfe9 (Apache-2.0)"
N=third_party/agent/nemo-speech
[ -d $N ] || git clone -q https://github.com/NVIDIA-NeMo/Speech.git $N
git -C $N checkout -q 097dfe9e2f55baf653b83035868bdc89849f1b47

echo "== weights: VoiceChat-11B @443794ea + Nano-9B-v2 remote code/tokenizer @6533e8d"
if [ -n "${MYNAH_HF_CACHE:-}" ] && [ ! -e hf-cache ]; then ln -s "$MYNAH_HF_CACHE" hf-cache; fi   # reuse an existing HF cache
$HFC download nvidia/NVIDIA-NemotronLabs-VoiceChat-11B --revision 443794ea956ef0065f001967ffd00e77f519cb39 >/dev/null
$HFC download nvidia/NVIDIA-Nemotron-Nano-9B-v2 --revision 6533e8de2c68e4536bf7c411d7a3ce5734111476 --exclude "*.safetensors" >/dev/null
# offline resolution of the remote code by name needs refs/main
for m in "NVIDIA-NemotronLabs-VoiceChat-11B 443794ea956ef0065f001967ffd00e77f519cb39" "NVIDIA-Nemotron-Nano-9B-v2 6533e8de2c68e4536bf7c411d7a3ce5734111476"; do
  set -- $m; mkdir -p hf-cache/models--nvidia--$1/refs; echo -n $2 > hf-cache/models--nvidia--$1/refs/main
done

echo "== venv (Python 3.12)"
rm -rf .venv-v2   # re-runnable: the venv lives in MYNAH_HOME and is rebuilt from scratch
uv venv -q --seed --python 3.12 .venv-v2
P=$MYNAH_HOME/.venv-v2/bin/python
uv pip install -q --python $P torch==2.10.0+cu130 torchaudio==2.10.0+cu130 triton==3.6.0 \
  --index-url https://download.pytorch.org/whl/cu130 --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match
uv pip install -q --python $P -r $REPO/v2/requirements-pinned.txt --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu130
uv pip install -q --python $P --no-deps -e $N -e $REPO/v2

echo "== mamba-ssm 2.3.2.post1 / causal-conv1d 1.6.2.post1 from source, for sm_$ARCH only"
# Their setup.py hard-codes a gencode list (and may miss your GPU); build a copy restricted to this GPU's arch.
uv pip install -q --python $P ninja packaging wheel setuptools
mkdir -p build/src && cd build/src
for pkg in "causal-conv1d 1.6.2.post1" "mamba-ssm 2.3.2.post1"; do   # sdists straight from PyPI (no metadata build)
  set -- $pkg
  url=$(python3 -c "import json,sys,urllib.request as u; d=json.load(u.urlopen('https://pypi.org/pypi/$1/$2/json')); print([f['url'] for f in d['urls'] if f['packagetype']=='sdist'][0])")
  curl -sSLO "$url"
done
for t in causal_conv1d-1.6.2.post1.tar.gz mamba_ssm-2.3.2.post1.tar.gz; do tar xzf $t; done
for f in causal_conv1d-1.6.2.post1/setup.py mamba_ssm-2.3.2.post1/setup.py; do
  ARCH=$ARCH python3 - "$f" <<'PY'
import os, re, sys
p, a = sys.argv[1], os.environ["ARCH"]; s = open(p).read()
s = re.sub(r'cc_flag\.append\("-gencode"\)\s*\n\s*cc_flag\.append\("arch=compute_\d+,code=sm_\d+"\)', 'pass', s)
s = s.replace('cc_flag = []', f'cc_flag = ["-gencode", "arch=compute_{a},code=sm_{a}"]', 1)
open(p, "w").write(s)
PY
done
export MAX_JOBS=${MAX_JOBS:-32} CAUSAL_CONV1D_FORCE_BUILD=TRUE MAMBA_FORCE_BUILD=TRUE
uv pip install --python $P --no-build-isolation --no-deps --no-cache ./causal_conv1d-1.6.2.post1
uv pip install --python $P --no-build-isolation --no-deps --no-cache ./mamba_ssm-2.3.2.post1
$P -c "import torch, mamba_ssm, causal_conv1d; print('torch', torch.__version__, torch.cuda.get_device_name())"
cd $MYNAH_HOME
fi
P=$MYNAH_HOME/.venv-v2/bin/python
[ -x $P ] || { echo "engine venv missing: run setup_cuda.sh without --only first"; exit 1; }

if [ $DO_TTS = 1 ]; then
  # Synthetic caller voice for the test corpus (scripts/make_corpus.py): Qwen3-TTS 0.6B CustomVoice, pinned revision.
  # Installed OUTSIDE the engine venv (its transformers version differs); env.sh puts it on PYTHONPATH only for TTS.
  echo "== test-audio TTS: requirements-tts.txt (isolated, no deps) and Qwen3-TTS 0.6B CustomVoice"
  rm -rf $MYNAH_HOME/cache/pylib-tts     # re-runnable; --no-deps: torch / numpy / librosa come from the engine venv
  uv pip install -q --python $P --target $MYNAH_HOME/cache/pylib-tts --no-deps -r $REPO/v2/requirements-tts.txt
  read TTS_MODEL TTS_REV < <(python3 -c "import json; t=json.load(open('$REPO/v2/eval/corpus_v1.json'))['tts']; print(t['model'], t['revision'])")
  $HFC download $TTS_MODEL --revision $TTS_REV >/dev/null
  mkdir -p hf-cache/models--${TTS_MODEL/\//--}/refs; echo -n $TTS_REV > hf-cache/models--${TTS_MODEL/\//--}/refs/main
fi

if [ $DO_ASR = 1 ] && [ "$(uname -m)" != "x86_64" ]; then
  echo "== evaluation ASR: NOT installed on $(uname -m)"
  echo "   omnilingual-asr needs fairseq2n, which has no $(uname -m) wheels on PyPI. Build fairseq2n 0.6 from source"
  echo "   (https://github.com/facebookresearch/fairseq2, native library) into \$MYNAH_HOME/.venv-asr and pip install"
  echo "   omnilingual-asr there, or run the ASR on an x86_64 machine. Test suites still run; they skip the audio check."
  [ $DO_ENGINE = 1 ] || [ $DO_TTS = 1 ] || exit 1      # --only asr: report failure
  DO_ASR=0
  ASR_PARTIAL=1
fi
if [ $DO_ASR = 1 ]; then
  # Evaluation ASR (scripts/asr_eval.py): Omnilingual ASR CTC 3B v2 through the official omnilingual_asr pipeline, in
  # its own venv (it pins its own torch / fairseq2). The checkpoint is downloaded on first use into cache/fairseq2.
  echo "== evaluation ASR: omnilingual-asr in .venv-asr"
  rm -rf .venv-asr
  uv venv -q --seed --python 3.12 .venv-asr
  uv pip install -q --python $MYNAH_HOME/.venv-asr/bin/python -r $REPO/v2/requirements-asr.txt   # pinned, see the file
  $MYNAH_HOME/.venv-asr/bin/python -c "import omnilingual_asr, fairseq2; print('omnilingual_asr ok, fairseq2', fairseq2.__version__)"
fi
if [ "${ASR_PARTIAL:-0}" = 1 ]; then
  echo "SETUP_DONE (PARTIAL: evaluation ASR not installed on $(uname -m); suites will skip the spoken-audio check) — next: make doctor"
else
  echo "SETUP_DONE — next: make doctor"
fi
