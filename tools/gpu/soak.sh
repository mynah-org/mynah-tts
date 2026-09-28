#!/bin/bash
. "$(dirname "$0")/compat.sh"
# Closed-loop or Poisson soak with the per-window report.
# usage: soak.sh <tag> <level> <seconds> [poisson-rate] [VAR=value ...]
# (pass "" as the rate to keep the closed loop and still give VAR=value pairs)
#
# Optional, for qualification runs (paths relative to MYNAH_GPU_ROOT or absolute):
#   MYNAH_GPU_CORPUS=tools/corpus/pocket_v2_en.jsonl  corpus instead of built-in texts
#   MYNAH_GPU_VOICES=alba,marius,javert,jean           voices drawn per request
#   MYNAH_GPU_SAVE_AUDIO=/root/evidence/<tag>/audio    keep the streamed audio as WAV
#   MYNAH_GPU_SAVE_EVERY=10                            every Nth request (default 1)
#   MYNAH_GPU_SAVE_MAX_MB=4000                         disk cap for the WAVs
#   MYNAH_GPU_SEED=1234                                ladder seed (texts, voices, seeds)
tag=$1; lvl=$2; secs=$3; rate=${4:-}
shift $(( $# < 4 ? $# : 4 ))
mode="--mode closed"; [ -n "$rate" ] && mode="--mode poisson --rate $rate --max-open 64"
extra=""
[ -n "$MYNAH_GPU_CORPUS" ] && extra="$extra --corpus $MYNAH_GPU_CORPUS"
[ -n "$MYNAH_GPU_VOICES" ] && extra="$extra --voices $MYNAH_GPU_VOICES"
[ -n "$MYNAH_GPU_SEED" ] && extra="$extra --seed $MYNAH_GPU_SEED"
if [ -n "$MYNAH_GPU_SAVE_AUDIO" ]; then
  extra="$extra --save-audio $MYNAH_GPU_SAVE_AUDIO --save-every ${MYNAH_GPU_SAVE_EVERY:-1}"
  [ -n "$MYNAH_GPU_SAVE_MAX_MB" ] && extra="$extra --save-max-mb $MYNAH_GPU_SAVE_MAX_MB"
fi
export MYNAH_GPU_LADDER_ARGS="--timeout 300 $mode$extra"
root="${MYNAH_GPU_ROOT:-/root/mynah-head}"; ev="${MYNAH_GPU_EVIDENCE:-/root/evidence}"
"$root/tools/gpu/ab.sh" "$tag" "${MYNAH_GPU_MODEL:-models/pocket-english-24l}" "$lvl" 30 "$secs" "$@"
/venv/main/bin/python "$root/tools/pocket_soak_report.py" "$ev/$tag/$tag-c$lvl.jsonl" --windows 10 \
  | tee "$ev/$tag/$tag-report.txt"
