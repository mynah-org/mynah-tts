#!/bin/bash
# One server start-up, measured: time to /health, VRAM at ready, peak VRAM
# during start-up, the server's own start-up lines. Then a stop.
# usage: TREE=/root/warmup/base-384 MAXB=320 MODEL=models/pocket-english-24l \
#        TAG=name ENVS="..." LIMIT=840 startup.sh
# Run it under: flock /root/box.lock timeout <secs> startup.sh
set -u
TREE=${TREE:?}; MAXB=${MAXB:-320}; MODEL=${MODEL:-models/pocket-english-24l}
TAG=${TAG:-run}; ENVS=${ENVS:-}; LIMIT=${LIMIT:-840}; PORT=${PORT:-18090}
R=/root/res/warmup/$TAG; mkdir -p $R; cd $TREE || exit 1
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 500 > $R/vram.log 2>&1 & VP=$!
t0=$(date +%s.%N)
( exec env MYNAH_QUANT_GROUPS=none MYNAH_THREADS=1 $ENVS ./build/cuda/mynah-tts-server --device cuda -w 8 \
    --max-batch $MAXB --max-inflight $MAXB --max-pending 2048 -p $PORT -m $MODEL ) > $R/server.log 2>&1 &
SPID=$!
stop_server() {
  kill -INT $SPID 2>/dev/null
  for _ in $(seq 1 30); do kill -0 $SPID 2>/dev/null || break; sleep 1; done
  kill -9 $SPID 2>/dev/null; kill $VP 2>/dev/null
}
trap stop_server EXIT
ok=0
end=$(( $(date +%s) + LIMIT ))
while [ $(date +%s) -lt $end ]; do
  sleep 1; kill -0 $SPID 2>/dev/null || break
  curl -m 5 -fsS localhost:$PORT/health >/dev/null 2>&1 && { ok=1; break; }
done
t1=$(date +%s.%N)
V=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
PEAK=$(sort -n $R/vram.log | tail -1)
if [ $ok = 1 ]; then
  printf '== %s [%s] MAXB=%s %s ready in %.1f s, VRAM at ready %s MiB, peak %s MiB\n' \
    "$TAG" "$ENVS" "$MAXB" "$MODEL" "$(echo "$t1 - $t0" | bc)" "$V" "$PEAK"
else
  printf '== %s [%s] MAXB=%s NOT READY after %.0f s, VRAM %s MiB, peak %s MiB\n' \
    "$TAG" "$ENVS" "$MAXB" "$(echo "$t1 - $t0" | bc)" "$V" "$PEAK"
fi
grep -h "warm-up\|prefill:\|start-up\|SLOT_FIXED\|re-plan\|replan\|out of memory\|graphs captured" $R/server.log | cut -c1-400 | sed 's/^/    /'
