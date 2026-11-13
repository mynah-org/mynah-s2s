# Environment for V2 runs. Layout under MYNAH_HOME (default ~/mynah-s2s, the path the code defaults use):
#   hf-cache/ (HF hub cache: VoiceChat-11B + Nano-9B-v2 remote code), third_party/agent/nemo-speech (NeMo Speech
#   @097dfe9), data/agent (your test audio), cache/v2/* (Triton / inductor / HF modules), .venv-v2 (Python 3.12).
export MYNAH_HOME=${MYNAH_HOME:-$HOME/mynah-s2s}
export HF_HUB_CACHE=$MYNAH_HOME/hf-cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONHASHSEED=0
export TRITON_CACHE_DIR=$MYNAH_HOME/cache/v2/triton TORCHINDUCTOR_CACHE_DIR=$MYNAH_HOME/cache/v2/inductor
export HF_MODULES_CACHE=$MYNAH_HOME/cache/v2/hf-modules XDG_CACHE_HOME=$MYNAH_HOME/cache/v2/xdg
if [ -z "${CUDA_HOME:-}" ]; then CUDA_HOME=$(ls -d /usr/local/cuda-13* 2>/dev/null | sort -V | tail -1); fi
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda} PATH=${CUDA_HOME:-/usr/local/cuda}/bin:$PATH
mkdir -p $TRITON_CACHE_DIR $TORCHINDUCTOR_CACHE_DIR $HF_MODULES_CACHE $XDG_CACHE_HOME
PY=$MYNAH_HOME/.venv-v2/bin/python
N=$MYNAH_HOME/third_party/agent/nemo-speech
C=$MYNAH_HOME/hf-cache/models--nvidia--NVIDIA-NemotronLabs-VoiceChat-11B/snapshots/443794ea956ef0065f001967ffd00e77f519cb39
# Optional guard for unified-memory machines (e.g. GB10): kill this job's process group if MemAvailable drops below
# MEM_GUARD_MIB, before the whole box runs out of memory.
mem_guard() {
  local pg parent; pg=$(ps -o pgid= $$ | tr -d " "); parent=$$
  ( exec 9>&-
    while kill -0 "$parent" 2>/dev/null; do
      a=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
      if [ "$a" -lt "${MEM_GUARD_MIB:-8000}" ]; then
        echo "MEM_GUARD: MemAvailable=${a}MiB < ${MEM_GUARD_MIB:-8000}MiB, killing job process group $pg" >&2
        kill -TERM -- -"$pg"; sleep 3; kill -KILL -- -"$pg"
      fi
      sleep 1
    done ) &
}
