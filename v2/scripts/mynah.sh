#!/usr/bin/env bash
# One entry point for checks and test suites; the Makefile targets call it. Run `make help` for the list.
#   mynah.sh doctor                 check core runtime (required), synthetic test audio and evaluation ASR (optional)
#   mynah.sh extras                 install the test-audio TTS and the evaluation ASR if missing (used by reproduce)
#   mynah.sh corpus                 render the synthetic test audio (corpus_v1, capability_v1, fcmulti_v1)
#   mynah.sh demo [ITEM]            one session on one corpus item (default t02: an order lookup with a tool call)
#   mynah.sh e2e-fc | e2e-corpus    single-session suites (tool calls / all 40 items) + ASR + score
#   mynah.sh multi [GROUP]          multi-session gate, 4 concurrent sessions (m4_notools, m4_mixed or both)
#   mynah.sh custom [SPEC] [MSG]    a custom scenario with its own system message (default: eval/examples/furniture_*)
#   mynah.sh serve | client WAV     realtime WebSocket server / test client
#   mynah.sh ws-check [ITEM]        start the server, stream one corpus item with the client, stop the server
#   mynah.sh asr DIR | score DIR    evaluation ASR / corpus scoring of an existing run directory
#   mynah.sh test-cpu | test-gpu    unit tests
# Outputs: $MYNAH_HOME/results/<suite>-<timestamp>/ (never overwritten). Extra run_stream / gate flags: ARGS="...".
set -uo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
V2=$REPO/v2
source $V2/scripts/env.sh
export PYTHONPATH=$V2/src${PYTHONPATH:+:$PYTHONPATH}
CALLER_DIR=$PWD                      # user paths (SPEC=, MSG=, WAV=) are relative to where make / mynah.sh was run
cd $V2
abspath() { case "$1" in /*) echo "$1" ;; *) if [ -e "$CALLER_DIR/$1" ]; then echo "$CALLER_DIR/$1"; else echo "$V2/$1"; fi ;; esac; }
STAMP=$(date +%Y%m%d-%H%M%S)
ARGS=${ARGS:-}
RED=$'\e[31m' GRN=$'\e[32m' YEL=$'\e[33m' RST=$'\e[0m'
ok()   { echo "  ${GRN}OK${RST}      $*"; }
warn() { echo "  ${YEL}WARN${RST}    $*"; }
miss() { echo "  ${RED}MISSING${RST} $*"; FAIL=1; }
opt()  { echo "  ${YEL}ABSENT${RST}  $*"; }
die()  { echo "${RED}error:${RST} $*" >&2; exit 1; }

need_engine() { [ -x "$PY" ] || die "engine not installed: run \`make setup\` (see README)"; }
need_corpus() { [ -f $DATA/corpus_v1/manifest.json ] || die "test audio not rendered: run \`make corpus\` (needs \`make setup-tts\`)"; }
have_tts()    { PYTHONPATH=$TTS_PYTHONPATH "$PY" -c "import qwen_tts" 2>/dev/null && ls -d $MYNAH_HOME/hf-cache/models--Qwen--Qwen3-TTS-12Hz-0.6B-CustomVoice/snapshots/* >/dev/null 2>&1; }
have_asr()    { "$PY_ASR" -c "import omnilingual_asr" 2>/dev/null; }
need_asr()    { "$PY_ASR" -c "import omnilingual_asr" 2>/dev/null || { warn "evaluation ASR not installed (\`make setup-asr\`): skipping the audio check"; return 1; }; }

doctor() {
  FAIL=0
  echo "== GPU / CUDA"
  if command -v nvidia-smi >/dev/null; then
    read -r name mem cc < <(nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader,nounits | head -1 | awk -F', ' '{gsub(/ /,"_",$1); print $1, $2, $3}')
    if [[ "$mem" =~ ^[0-9]+$ ]]; then
      ok "GPU $name, ${mem} MiB, compute capability $cc"
      [ "$mem" -ge 46000 ] || warn "less than 46 GB GPU memory: one VoiceChat session needs ~42 GB at peak"
    else
      um=$(awk '/MemTotal/{print int($2/1024/1024)}' /proc/meminfo)
      ok "GPU $name, unified memory (system RAM ${um} GB), compute capability $cc"
    fi
  else miss "nvidia-smi not found (NVIDIA driver)"; fi
  command -v nvcc >/dev/null && ok "nvcc $(nvcc --version | grep -o 'release [0-9.]*')" || miss "nvcc not found (CUDA 13.x toolkit, set CUDA_HOME)"
  echo "== core runtime (make setup) — required"
  if [ -x "$PY" ]; then
    out=$("$PY" -c "import torch, mamba_ssm, causal_conv1d, nemo, mynah_v2; print(torch.__version__, torch.cuda.is_available())" 2>&1 | tail -1)
    case "$out" in *True) ok "venv: torch $out (mamba_ssm, causal_conv1d, nemo, mynah_v2 import)";; *) miss "venv imports failed: $out  -> make setup";; esac
  else miss ".venv-v2 not found -> make setup"; fi
  [ "$(git -C $N rev-parse --short=7 HEAD 2>/dev/null)" = "097dfe9" ] && ok "NeMo Speech @097dfe9" || miss "NeMo Speech checkout @097dfe9 -> make setup"
  w=$(du -L -sm $C/model.safetensors 2>/dev/null | cut -f1)
  [ "${w:-0}" -gt 40000 ] && ok "VoiceChat-11B weights ($w MB)" || miss "VoiceChat-11B weights not downloaded -> make setup"
  ls $MYNAH_HOME/hf-cache/models--nvidia--NVIDIA-Nemotron-Nano-9B-v2/snapshots/*/config.json >/dev/null 2>&1 && ok "Nemotron Nano 9B v2 remote code" || miss "Nemotron Nano 9B v2 remote code -> make setup"
  echo "== synthetic test audio (make setup-tts, make corpus) — optional, needed by demo / e2e / multi / custom"
  PYTHONPATH=$TTS_PYTHONPATH "$PY" -c "import qwen_tts" 2>/dev/null && ok "qwen-tts" || opt "qwen-tts not installed -> make setup-tts"
  ls -d $MYNAH_HOME/hf-cache/models--Qwen--Qwen3-TTS-12Hz-0.6B-CustomVoice/snapshots/* >/dev/null 2>&1 && ok "Qwen3-TTS 0.6B CustomVoice" || opt "Qwen3-TTS model not downloaded -> make setup-tts"
  for d in corpus_v1:40 capability_v1:9 fcmulti_v1:7; do n=$(ls $DATA/${d%%:*}/*.wav 2>/dev/null | wc -l)
    [ "$n" -ge "${d##*:}" ] && ok "${d%%:*}: $n WAVs $( [ -f $V2/eval/reference/${d%%:*}.manifest.json ] && ref_match ${d%%:*})" || opt "${d%%:*}: $n/${d##*:} WAVs -> make corpus"; done
  echo "== evaluation ASR (make setup-asr) — optional, the audio check of the test suites"
  if have_asr; then ok "omnilingual-asr (checkpoint is fetched into $FAIRSEQ2_CACHE_DIR on first use)"
  elif [ "$(uname -m)" != "x86_64" ]; then opt "omnilingual-asr not installed: no $(uname -m) wheels for fairseq2n (see make setup-asr); the audio check is skipped"
  else opt "omnilingual-asr not installed -> make setup-asr (runs still work; the audio check is skipped)"; fi
  [ $FAIL = 0 ] && echo "${GRN}core runtime OK${RST} (optional parts: see above)" || echo "${RED}core runtime incomplete${RST}: run the suggested make targets"
  return $FAIL
}

corpus() {
  need_engine; have_tts || die "the test-audio TTS is not installed: run \`make setup-tts\`"
  local tp="PYTHONPATH=$V2/src:$TTS_PYTHONPATH"
  [ -f $DATA/corpus_v1/manifest.json ] || env $tp $PY scripts/make_corpus.py --out $DATA/corpus_v1 || exit 1
  [ -f $DATA/capability_v1/manifest.json ] || env $tp $PY scripts/make_corpus.py --corpus eval/capability_v1.json --out $DATA/capability_v1 || exit 1
  $PY scripts/make_fcmulti.py $DATA $DATA/fcmulti_v1 || exit 1
  echo "test audio in $DATA: $(ls $DATA/corpus_v1/*.wav | wc -l) corpus, $(ls $DATA/capability_v1/*.wav | wc -l) capability, $(ls $DATA/fcmulti_v1/*.wav | wc -l) multi-call"
  echo "corpus_v1: $(ref_match corpus_v1)"; echo "capability_v1: $(ref_match capability_v1)"
}

asr()   { need_asr && "$PY_ASR" scripts/asr_eval.py "$@"; }
ref_match() {  # ref_match <set>: how many rendered WAVs are byte-identical to the reference render (eval/reference)
  python3 - "$DATA/$1/manifest.json" "$V2/eval/reference/$1.manifest.json" <<'PYM'
import json, sys
try:
    got, ref = json.load(open(sys.argv[1]))["items"], json.load(open(sys.argv[2]))["items"]
except FileNotFoundError:
    sys.exit(0)
same = sum(1 for k in ref if k in got and got[k]["sha256"] == ref[k]["sha256"])
print(f"{same}/{len(ref)} identical to the reference render" + ("" if same == len(ref) else
      " (different TTS stack or GPU: results stay valid but are not item-for-item comparable with ours)"))
PYM
}
score() { $PY scripts/score_corpus.py "$1" --corpus-dir $DATA/corpus_v1 | tail -12; }

stream() {  # stream <suite-name> <wav...>; sets O
  local name=$1; shift; O=$RESULTS/$name-$STAMP
  $PY scripts/run_stream.py "$@" --out $O --orders-json $DATA/corpus_v1/orders.json --seed 0 $ARGS || exit 1
  asr $O || true
}

ws_check() {
  need_engine; need_corpus; local it=${1:-q03} T=$RESULTS/ws-check-$STAMP port=8799; mkdir -p $T
  # a large inbound backlog: on a GPU slower than realtime the check still tests the full protocol round trip
  # (no audio dropped); the realtime margin is reported separately below
  $PY scripts/serve.py --trace-dir $T/server --port $port --max-backlog 600 $ARGS > $T/server.log 2>&1 & local pid=$!
  for i in $(seq 1 90); do grep -q "^READY" $T/server.log && break; kill -0 $pid 2>/dev/null || { tail -20 $T/server.log; die "server exited"; }; sleep 5; done
  grep -q "^READY" $T/server.log || { kill $pid; die "server not ready after 450 s (see $T/server.log)"; }
  $PY scripts/ws_client.py --url ws://127.0.0.1:$port/v1/session --wav $DATA/corpus_v1/$it.wav --out $T/client; rc=$?
  kill $pid; wait $pid 2>/dev/null
  [ $rc = 0 ] || { echo "results: $T"; die "client failed (rc=$rc)"; }
  # PASS = no inbound audio dropped AND an agent answer after the greeting (a protocol round trip that loses the
  # question is not a pass); realtime = wall time vs input duration
  $PY - $T/client/client_result.json $DATA/corpus_v1/$it.wav <<'PY'; rc=$?
import json, sys
import soundfile as sf
r = json.load(open(sys.argv[1]))
in_s = sf.info(sys.argv[2]).duration
turns = [t.strip() for t in r["text"].replace("</s>", "").split("<s>") if t.strip()]
print("agent turns:", turns)
print(f"input {in_s:.1f} s, wall {r['wall_s']:.1f} s, first agent audio {r['first_audio_s'] or 0:.1f} s, "
      f"dropped notices {r['dropped_notices']}")
if r["wall_s"] > in_s + 20:
    print("NOTE: this GPU is slower than realtime for one session (the server queued the audio); "
          "see README 'Status' for the realtime GPUs")
ok = r["dropped_notices"] == 0 and len(turns) >= 2
print("ws-check:", "PASS" if ok else "FAIL (" + ("audio dropped" if r["dropped_notices"] else "no answer after the greeting") + ")")
sys.exit(0 if ok else 1)
PY
  echo "results: $T"; return $rc
}

case "${1:-help}" in
  doctor) doctor ;;
  extras)
    need_engine
    have_tts || bash $V2/scripts/setup_cuda.sh --only tts || exit 1
    have_asr || bash $V2/scripts/setup_cuda.sh --only asr || echo "continuing without the evaluation ASR (audio check skipped)" ;;
  ws-check) ws_check "${2:-}" ;;
  corpus) corpus ;;
  demo)
    need_engine; need_corpus; it=${2:-t02}; stream demo $DATA/corpus_v1/$it.wav
    echo "== agent text (internal)"; $PY -c "import json,sys; print(json.load(open('$O/$it/summary.json'))['text'])"
    echo "== tool calls"; $PY -c "import json; print(json.load(open('$O/$it/summary.json'))['tool_calls'])"
    [ -f $O/asr_omni3b.txt ] && { echo "== ASR of the spoken audio"; cat $O/asr_omni3b.txt; }
    echo "listen: $O/$it/combined.wav (user left, agent right)" ;;
  e2e-fc)
    need_engine; need_corpus
    W="$DATA/corpus_v1/t01.wav $DATA/corpus_v1/t02.wav $DATA/corpus_v1/t11.wav $DATA/corpus_v1/t13.wav $DATA/corpus_v1/t14.wav $DATA/corpus_v1/r01.wav $DATA/fcmulti_v1/m01.wav $DATA/fcmulti_v1/i01.wav"
    stream e2e-fc $W; score $O
    echo "note: m01 / i01 (multi-call conversations) are run but not scored; see $O/m01 and $O/i01 summary.json"
    echo "results: $O" ;;
  e2e-corpus)
    need_engine; need_corpus; stream e2e-corpus $DATA/corpus_v1/*.wav; score $O; echo "results: $O" ;;
  multi)
    need_engine; need_corpus
    for g in ${2:-m4_notools m4_mixed}; do O=$RESULTS/multi-$g-$STAMP
      $PY scripts/rendezvous_pair.py --spec eval/multi_session_v1.json --group $g --audio-root $DATA --out $O $ARGS || exit 1
      $PY scripts/val3_summary.py $O --json $O/summary.json | tee $O/summary.txt | cut -c1-220
      asr $O/wav || true
      # verdict: same calls alone and together, no leak, no error (latency is reported above, not judged here)
      $PY - $O/summary.txt <<'PYV' || exit 1
import json, sys
bad = []
for line in open(sys.argv[1]):
    if line.startswith("SESS "):
        _, sid, js = line.split(" ", 2)
        d = json.loads(js)
        for k, ok in (("calls_identical", True), ("leak", False), ("errors", False)):
            if d.get(k) != ok:
                bad.append(f"{sid}:{k}={d.get(k)}")
print("GATE", "PASS" if not bad else "FAIL " + ", ".join(bad), "(calls identical alone vs together, no leak, no error)")
sys.exit(1 if bad else 0)
PYV
      echo "results: $O"; done ;;
  custom)
    need_engine; spec=$(abspath "${2:-eval/examples/furniture_store.json}"); msg=$(abspath "${3:-eval/examples/furniture_store_system.txt}")
    [ -f "$spec" ] || die "scenario not found: $spec"; [ -f "$msg" ] || die "system message not found: $msg"
    name=$(basename $spec .json); D=$DATA/custom-$name-$(sha256sum "$spec" | cut -c1-8)   # re-rendered when the JSON changes
    [ -f $D/manifest.json ] || env PYTHONPATH=$V2/src:$TTS_PYTHONPATH $PY scripts/make_corpus.py --corpus $spec --out $D || exit 1
    O=$RESULTS/custom-$name-$STAMP
    $PY scripts/run_stream.py $D/*.wav --out $O --orders-json $D/orders.json --seed 0 --system-message @$msg $ARGS || exit 1
    asr $O || true
    for s in $O/*/summary.json; do echo "== $(basename $(dirname $s))"; $PY -c "import json; d=json.load(open('$s')); print('text :', d['text'][:400]); print('tools:', d['tool_calls'][:300])"; done
    [ -f $O/asr_omni3b.txt ] && { echo "== ASR"; cat $O/asr_omni3b.txt; }; echo "results: $O" ;;
  serve)  need_engine; exec $PY scripts/serve.py --host 127.0.0.1 --trace-dir $RESULTS/serve-$STAMP ${ARGS} ;;
  client)
    [ -n "${2:-}" ] || die "usage: mynah.sh client WAV"; C=$RESULTS/client-$STAMP
    $PY scripts/ws_client.py --wav "$(abspath "$2")" --out $C $ARGS || exit 1
    n=$($PY -c "import json; print(json.load(open('$C/client_result.json'))['dropped_notices'])")
    [ "$n" = 0 ] || warn "the server dropped inbound audio $n times (slower than real time?): restart it with ARGS=\"--max-backlog 120\""
    echo "results: $C" ;;
  asr)    [ -n "${2:-}" ] || die "usage: mynah.sh asr RUN_DIR"; shift; asr "$@" ;;
  score)  [ -n "${2:-}" ] || die "usage: mynah.sh score RUN_DIR"; score "$2" ;;
  test-cpu) need_engine; $PY -m pytest -q tests/cpu ;;
  test-gpu) need_engine; $PY -m pytest -q tests/gpu ;;
  help|*) sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//' ;;
esac
