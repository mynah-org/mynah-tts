#!/bin/bash
# Closed-loop knee for the Pocket CUDA server: one fresh server, a list of
# concurrency levels, one summary line per level. The method is described in
# docs/benchmarking.md.
#
# usage: [VAR=value ...] tools/gpu/knee_closed.sh
#
#   TAG      run name; results go to $OUT/$TAG                  (default: run)
#   LEVELS   space-separated levels, run in this order          (default: 768)
#   DUR      measured seconds per level                         (default: 120)
#   WARM     warm-up seconds per level, discarded               (default: 15)
#   PRE      optional warm-up level run first (45 s, not reported), e.g. 640
#   PROCS    load-generator processes (pocket_ladder --client-procs) (default: 4)
#   ENVS     extra server environment, e.g. "MYNAH_CTX_HOST_POOL=1"
#   PIN      taskset CPU list for the server (the GPU's NUMA node), e.g. 32-63,96-127
#   CLIPIN   taskset CPU list for the clients (another node)
#   THERM    1 = log temperature, SM clock, power and throttle reasons every 15 s
#   MODEL    model pack                          (default: models/pocket-english-24l)
#   ROOT     repository with build/cuda/          (default: the current directory)
#   OUT      results directory                    (default: ./bench-res)
#   PORT     server port                           (default: 18080)
#   MAXB     --max-batch / --max-inflight           (default: the highest level)
#   WORKERS  HTTP workers (-w)                      (default: 8)
#   PY       python interpreter                     (default: python3)
#
# Per level it prints:
#   C<n> aps <audio-s/s> rtf95 <stream RTF p95> ttfa95 <ms> gap95 <ms> st250 <stalls> fail <n>
#        | cli_cpu <%> | gpu <W> sm <%>
# "gpu" and "sm" are the means of `nvidia-smi dmon` samples with sm > 5 %.
# At the end it stops the server with SIGINT, so MYNAH_SERVE_PROFILE prints its
# report into $OUT/$TAG/server.log, and shows the [SERVE] device wait and loop lines.
set -u
TAG=${TAG:-run}; LEVELS=${LEVELS:-768}; DUR=${DUR:-120}; WARM=${WARM:-15}; PROCS=${PROCS:-4}
ENVS=${ENVS:-}; MODEL=${MODEL:-models/pocket-english-24l}; ROOT=${ROOT:-$PWD}
OUT=${OUT:-$PWD/bench-res}; PORT=${PORT:-18080}; WORKERS=${WORKERS:-8}; PY=${PY:-python3}
SP=""; [ -n "${PIN:-}" ] && SP="taskset -c $PIN"
CP=""; [ -n "${CLIPIN:-}" ] && CP="taskset -c $CLIPIN"
MAXB=${MAXB:-$(printf '%s\n' $LEVELS | sort -n | tail -1)}
R=$OUT/$TAG; mkdir -p "$R" && R=$(cd "$R" && pwd); cd "$ROOT" || exit 1
LADDER="$PY tools/pocket_ladder.py --port $PORT --mode closed --timeout 300 \
  --corpus tools/corpus/pocket_v2_en.jsonl --voices alba,marius,javert,jean --client-procs $PROCS"

curl -fsS "localhost:$PORT/health" >/dev/null 2>&1 && { echo "$TAG: port $PORT is busy"; exit 1; }
t0=$(date +%s)
( exec env MYNAH_QUANT_GROUPS=none MYNAH_SERVE_PROFILE=1 MYNAH_THREADS=1 $ENVS $SP \
    ./build/cuda/mynah-tts-server --device cuda -w "$WORKERS" --max-batch "$MAXB" \
    --max-inflight "$MAXB" --max-pending 2048 -p "$PORT" -m "$MODEL" ) > "$R/server.log" 2>&1 &
SPID=$!
stop_server() {
  kill -INT "$SPID" 2>/dev/null
  for _ in $(seq 1 30); do kill -0 "$SPID" 2>/dev/null || return; sleep 1; done
  kill -9 "$SPID" 2>/dev/null
}
trap 'stop_server; [ -n "${TP:-}" ] && kill $TP 2>/dev/null' EXIT
ok=0
for _ in $(seq 1 300); do
  sleep 2; kill -0 "$SPID" 2>/dev/null || break
  curl -fsS "localhost:$PORT/health" >/dev/null 2>&1 && { ok=1; break; }
done
[ $ok = 1 ] || { echo "$TAG: server did not start"; tail -5 "$R/server.log"; exit 1; }
vram() { nvidia-smi --query-gpu=memory.used --format=csv,noheader | head -1; }
echo "== $TAG [$ENVS] procs=$PROCS pin=${PIN:-none} ready in $(( $(date +%s) - t0 )) s, VRAM $(vram)"
if [ "${THERM:-0}" = 1 ]; then
  nvidia-smi --query-gpu=temperature.gpu,clocks.sm,power.draw,clocks_throttle_reasons.active \
    --format=csv,noheader -l 15 > "$R/therm.log" 2>&1 & TP=$!
fi
if [ -n "${PRE:-}" ]; then
  timeout 300 $CP $LADDER --levels "$PRE" --warmup 5 --duration 45 --seed 99 --out "$R/pre" --tag pre \
    > "$R/pre.log" 2>&1
fi
for C in $LEVELS; do
  kill -0 "$SPID" 2>/dev/null || { echo "  C$C: SERVER DIED before this level"; tail -3 "$R/server.log"; break; }
  nvidia-smi dmon -s pu -d 1 > "$R/dmon-c$C.log" 2>&1 & DM=$!
  timeout $((DUR + WARM + 400)) $CP $LADDER --levels "$C" --warmup "$WARM" --duration "$DUR" --seed 1234 \
    --server-pid "$SPID" --out "$R" --tag "c$C" > "$R/ladder-c$C.log" 2>&1
  kill $DM 2>/dev/null
  $PY - "$R/c$C-summary.jsonl" "$R/dmon-c$C.log" "$C" <<'PYEOF'
import json, sys
try:
    s = json.loads(open(sys.argv[1]).read().strip().splitlines()[-1])
except Exception:
    print("  C%s: no summary (see ladder-c%s.log)" % (sys.argv[3], sys.argv[3])); sys.exit(0)
pw, sm = [], []
for line in open(sys.argv[2]):
    f = line.split()
    if not f or f[0].startswith("#"): continue
    try: p, u = float(f[1]), float(f[4])
    except (ValueError, IndexError): continue
    if u > 5: pw.append(p); sm.append(u)
p95 = lambda k: s[k]["p95"] if s[k]["p95"] is not None else float("nan")
print("  C%s aps %.0f rtf95 %.3f ttfa95 %.0fms gap95 %.0fms st250 %d fail %d | cli_cpu %.0f%% | gpu %.0fW sm %.0f%%" % (
    sys.argv[3], s["audio_s_per_s"], p95("rtf_stream"), 1e3 * p95("ttfa"), 1e3 * p95("max_gap"),
    s["stalls_250"], s["failed"], s["client_cpu_pct"],
    sum(pw) / max(1, len(pw)), sum(sm) / max(1, len(sm))))
PYEOF
done
echo "  VRAM end $(vram)"
stop_server
if [ -n "${TP:-}" ]; then
  kill $TP 2>/dev/null; TP=""
  echo "  therm (temp C, SM clock, power, throttle reasons), every 45 s:"
  awk 'NR % 3 == 0 { print "    " $0 }' "$R/therm.log"
fi
grep -E "\[SERVE\] (device wait|loop)" "$R/server.log" | sed 's/^/  /'
