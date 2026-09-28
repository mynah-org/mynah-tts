#!/bin/bash
# Qualify one concurrency level: two independent 30-minute closed-loop soaks on
# the v2 corpus with captured audio and a GPU-memory log, recogniser WER on the
# captures (server stopped), an optional unloaded control on the same request
# ids, and the evidence bundle with capped listening ZIPs.
# usage: qualify.sh <name> <model-dir> <C> [options as env]
#   MYNAH_Q_SECONDS   soak length (1800)          MYNAH_Q_SEEDS  "1234 5678"
#   MYNAH_Q_CONTROL   1 = also run the C8 control on the first soak's ids (0)
#   MYNAH_Q_ASR_EVERY transcribe every Nth capture (1; 4 halves the wait on a small model)
#   MYNAH_Q_ASR_WORKERS recogniser processes (default nproc/2, one core each)
#   MYNAH_L4_SAVE_EVERY capture every Nth request (10)   MYNAH_L4_LISTEN_MB  ZIP cap per soak (100)
# Run it under tools/l4/detach.sh so it finishes without the ssh session.
set -u
name=$1; model=$2; C=$3
root="${MYNAH_L4_ROOT:-/root/mynah-head}"; ev="${MYNAH_L4_EVIDENCE:-/root/evidence}"
cd "$root" || exit 1
secs="${MYNAH_Q_SECONDS:-1800}"; seeds=(${MYNAH_Q_SEEDS:-1234 5678})
every="${MYNAH_Q_ASR_EVERY:-1}"; workers="${MYNAH_Q_ASR_WORKERS:-$(( $(nproc) / 2 ))}"
py="${MYNAH_L4_PY:-/venv/main/bin/python}"; [ -x "$py" ] || py=python3
export MYNAH_L4_MODEL="$model" MYNAH_L4_BATCH="$C"
export MYNAH_L4_CORPUS="${MYNAH_L4_CORPUS:-tools/corpus/pocket_v2_en.jsonl}"
export MYNAH_L4_VOICES="${MYNAH_L4_VOICES:-alba,marius,javert,jean}"
export MYNAH_L4_SAVE_EVERY="${MYNAH_L4_SAVE_EVERY:-10}"

tags=()
for i in "${!seeds[@]}"; do
  t="$name-$(printf "\\x$(printf %x $((65 + i)))")"   # <name>-A, <name>-B, ...
  tags+=("$t")
  "$root/tools/l4/vram_log.sh" "$ev/$t-vram.log" 30 & mon=$!
  MYNAH_L4_SEED="${seeds[$i]}" MYNAH_L4_SAVE_AUDIO="$ev/$t/audio" "$root/tools/l4/soak.sh" "$t" "$C" "$secs"
  kill "$mon" 2>/dev/null
  echo "$t-SOAK-DONE"
done

asr() {  # asr <jsonl> <out-dir>
  local j=$1 src=$1
  if [ "$every" -gt 1 ]; then
    src="${j%.jsonl}-asr-sample.jsonl"
    python3 - "$j" "$every" > "$src" <<'PY'
import json, sys
n = 0
for line in open(sys.argv[1]):
    r = json.loads(line)
    if r.get("wav"):
        n += 1
        if n % int(sys.argv[2]):
            continue
    print(line.rstrip())
PY
  fi
  flock /root/gpu.lock "$py" tools/pocket_quality.py --from-jsonl "$src" --out "$2" \
    --asr-workers "$workers" --asr-threads 1 2>&1 | grep -v -E "FLAG req|BAD req|analysed"
}

for t in "${tags[@]}"; do asr "$ev/$t/$t-c$C.jsonl" "$ev/$t/quality"; done

if [ "${MYNAH_Q_CONTROL:-0}" = 1 ]; then
  ct="$name-ctrl"
  MYNAH_L4_BATCH=8 MYNAH_L4_SEED="${seeds[0]}" MYNAH_L4_SAVE_AUDIO="$ev/$ct/audio" \
    "$root/tools/l4/soak.sh" "$ct" 8 900
  asr "$ev/$ct/$ct-c8.jsonl" "$ev/$ct/quality"
  tags+=("$ct")
fi

"$root/tools/l4/bundle.sh" "$name" "${tags[@]}" 2>&1 | tail -3
ls -la "$ev/bundles/$name"/*/audio-*.zip 2>/dev/null
echo "QUALIFY-DONE $name C$C"
