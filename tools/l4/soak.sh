#!/bin/bash
# Closed-loop or Poisson soak with the per-window report.
# usage: soak.sh <tag> <level> <seconds> [poisson-rate] [VAR=value ...]
tag=$1; lvl=$2; secs=$3; rate=$4; shift 4
mode="--mode closed"; [ -n "$rate" ] && mode="--mode poisson --rate $rate --max-open 64"
export MYNAH_L4_LADDER_ARGS="--timeout 300 $mode"
root="${MYNAH_L4_ROOT:-/root/mynah-head}"; ev="${MYNAH_L4_EVIDENCE:-/root/evidence}"
"$root/tools/l4/ab.sh" "$tag" models/pocket-english-24l "$lvl" 30 "$secs" "$@"
/venv/main/bin/python "$root/tools/pocket_soak_report.py" "$ev/$tag/$tag-c$lvl.jsonl" --windows 10
