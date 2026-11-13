#!/usr/bin/env bash
# One-time setup on a Linux x86_64 or aarch64 machine with an NVIDIA GPU and a CUDA 13.x toolkit.
# Creates the MYNAH_HOME layout used by env.sh and the code defaults (see env.sh). Versions are the ones we tested
# (requirements-pinned.txt). Downloads ~42 GB of model weights from Hugging Face: read NOTICE.md (model license) first.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
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

echo "== NVIDIA NeMo Speech @097dfe9 (Apache-2.0)"
N=third_party/agent/nemo-speech
[ -d $N ] || git clone -q https://github.com/NVIDIA-NeMo/Speech.git $N
git -C $N checkout -q 097dfe9e2f55baf653b83035868bdc89849f1b47

echo "== weights: VoiceChat-11B @443794ea + Nano-9B-v2 remote code/tokenizer @6533e8d"
if [ -n "${MYNAH_HF_CACHE:-}" ] && [ ! -e hf-cache ]; then ln -s "$MYNAH_HF_CACHE" hf-cache; fi   # reuse an existing HF cache
HFC="uvx --from huggingface_hub[cli]==0.34.4 huggingface-cli"
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
echo "SETUP_DONE — next: source $REPO/v2/scripts/env.sh"
