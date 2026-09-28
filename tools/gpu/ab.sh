#!/bin/bash
# One serving measurement: start the server, run the ladder under the GPU
# lock, stop the server, print the summary and the [SERVE] loop profile.
# usage: ab.sh <tag> <model-dir> <levels> <warmup-s> <duration-s> [VAR=value ...]
tag=$1; model=$2; lvls=$3; wu=$4; du=$5; shift 5
mb="${MYNAH_L4_BATCH:-64}"; mi="${MYNAH_L4_INFLIGHT:-$mb}"
root="${MYNAH_L4_ROOT:-/root/mynah-head}"; ev="${MYNAH_L4_EVIDENCE:-/root/evidence}"
port="${MYNAH_L4_PORT:-18080}"
mkdir -p "$ev"
pkill -f "mynah-tts-server.*-p $port"; sleep 2
tmux kill-session -t "srv-$port" 2>/dev/null
tmux new-session -d -s "srv-$port" "MYNAH_L4_CPUS=$MYNAH_L4_CPUS MYNAH_L4_WORKERS=$MYNAH_L4_WORKERS $root/tools/l4/serve.sh $model $mb $mi $port $* > $ev/$tag-server.log 2>&1"
for i in $(seq 90); do curl -sf "localhost:$port/health" >/dev/null && break; sleep 2; done
P=$(pgrep -f "mynah-tts-server.*-p $port" | head -1)
echo "affinity: $(taskset -cp "$P" 2>&1)" >> "$ev/$tag-server.log"
cd "$root" && flock "${MYNAH_L4_LOCK:-/root/gpu.lock}" /venv/main/bin/python tools/pocket_ladder.py --port "$port" \
  --levels "$lvls" --warmup "$wu" --duration "$du" --server-pid "$P" --stop-rtf-p95 50 \
  ${MYNAH_L4_LADDER_ARGS} --out "$ev/$tag" --tag "$tag" > "$ev/$tag-ladder.out" 2>&1
pkill -INT -f "mynah-tts-server.*-p $port"; sleep 4
echo "=== $tag $* cpus=${MYNAH_L4_CPUS:-all} workers=${MYNAH_L4_WORKERS:-8}"
grep -E "affinity:|quant=|resident\{|error|WARN" "$ev/$tag-server.log" | head -4
python3 "$root/tools/l4/summ.py" "$ev/$tag/$tag-summary.jsonl"
echo "stream_failed=$(grep -c 'stream failed' "$ev/$tag-server.log")"
grep -E "SERVE\]   B(1|8|16|32|48|64) |SERVE\] loop" "$ev/$tag-server.log"
