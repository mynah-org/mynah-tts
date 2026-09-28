#!/bin/bash
# Screen a concurrency ladder and pick the qualification level.
# usage: knee.sh <tag> <model-dir> <levels> [rtf-gate] [seconds-per-level]
# Runs ab.sh with max-batch = the highest level, then prints the highest level
# with 0 failures, 0 stalls at 250 ms, TTFA p95 < 500 ms and stream RTF p95
# below the gate (default 0.88: real margin under the 0.90 production gate).
# Last line: "KNEE C<n>" (C0 when nothing passed).
set -u
tag=$1; model=$2; lvls=$3; gate="${4:-0.88}"; secs="${5:-25}"
root="${MYNAH_L4_ROOT:-/root/mynah-head}"; ev="${MYNAH_L4_EVIDENCE:-/root/evidence}"
top=$(echo "$lvls" | tr ',' '\n' | sort -n | tail -1)
MYNAH_L4_BATCH="${MYNAH_L4_BATCH:-$top}" "$root/tools/l4/ab.sh" "$tag" "$model" "$lvls" 5 "$secs"
python3 - "$ev/$tag/$tag-summary.jsonl" "$gate" <<'PY'
import json, sys
best = 0
for line in open(sys.argv[1]):
    s = json.loads(line)
    ok = (s["failed"] == 0 and s["stalls_250"] == 0 and s["completed"] > 0
          and (s["ttfa"]["p95"] or 9) < 0.5
          and s["rtf_stream"]["p95"] is not None and s["rtf_stream"]["p95"] < float(sys.argv[2]))
    print("C%-4d rtf_p95 %s ttfa_p95 %s stalls %d failed %d -> %s" % (
        s["offered"], s["rtf_stream"]["p95"], s["ttfa"]["p95"], s["stalls_250"], s["failed"],
        "pass" if ok else "fail"))
    if ok:
        best = max(best, s["offered"])
print("KNEE C%d" % best)
PY
