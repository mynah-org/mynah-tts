#!/bin/bash
# Cold burst: fresh server, C$C closed loop for DUR s starting at ready (no warm-up),
# /metrics snapshot at ready and after. usage: TREE= TAG= ENVS= C=320 DUR=30 cold.sh
set -u
TREE=${TREE:-/root/warmup2}; TAG=${TAG:?}; ENVS=${ENVS:-}; C=${C:-320}; DUR=${DUR:-30}
MAXB=${MAXB:-$C}; PORT=${PORT:-18080}; PY=/venv/main/bin/python
R=/root/res/warmup2/$TAG; rm -rf $R; mkdir -p $R; cd $TREE || exit 1
t0=$(date +%s.%N)
( exec env MYNAH_QUANT_GROUPS=none MYNAH_SERVE_PROFILE=1 MYNAH_THREADS=1 $ENVS ./build/cuda/mynah-tts-server --device cuda -w 8 \
   --max-batch $MAXB --max-inflight $MAXB --max-pending 2048 -p $PORT -m models/pocket-english-24l ) > $R/server.log 2>&1 &
SPID=$!
trap 'kill -INT $SPID 2>/dev/null; for _ in $(seq 1 30); do kill -0 $SPID 2>/dev/null || break; sleep 1; done; kill -9 $SPID 2>/dev/null' EXIT
ok=0
for _ in $(seq 1 600); do sleep 1; kill -0 $SPID 2>/dev/null || break
  curl -m 5 -fsS localhost:$PORT/health >/dev/null 2>&1 && { ok=1; break; }; done
t1=$(date +%s.%N)
[ $ok = 1 ] || { echo "$TAG not ready"; tail -5 $R/server.log; exit 1; }
V=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
curl -m 5 -s localhost:$PORT/metrics > $R/m0.txt
timeout $((DUR + 400)) $PY tools/pocket_ladder.py --port $PORT --mode closed --timeout 300 \
  --corpus tools/corpus/pocket_v2_en.jsonl --voices alba,marius,javert,jean --client-procs 4 \
  --levels $C --warmup 0 --duration $DUR --seed 1234 --server-pid $SPID --out $R --tag c$C > $R/ladder.log 2>&1
curl -m 5 -s localhost:$PORT/metrics > $R/m1.txt
$PY - $R/c$C-summary.jsonl $C "$TAG [$ENVS]" "$(echo "$t1 - $t0" | bc)" "$V" <<'PYEOF'
import json, sys
s = json.loads(open(sys.argv[1]).read().strip().splitlines()[-1])
p95 = lambda k: s[k]["p95"] if s[k]["p95"] is not None else float("nan")
print("== %s ready %.1fs VRAM %s MiB | C%s aps %.0f rtf95 %.3f ttfa95 %.0fms gap95 %.0fms st250 %d fail %d" % (
    sys.argv[3], float(sys.argv[4]), sys.argv[5], sys.argv[2], s["audio_s_per_s"], p95("rtf_stream"), 1e3 * p95("ttfa"),
    1e3 * p95("max_gap"), s["stalls_250"], s["failed"]))
PYEOF
grep "start-up phases" $R/server.log | cut -c1-400
# counters that moved during the burst
join <(grep -v '^#' $R/m0.txt | sort) <(grep -v '^#' $R/m1.txt | sort) | awk '$2!=$3 && /graph|capture|alloc|fallback|grow|pool|slot|replan|realloc|malloc/ {print "   ", $1, $2, "->", $3}'
