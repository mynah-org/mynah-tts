#!/bin/bash
. "$(dirname "$0")/compat.sh"
# Quality gate against a fresh server: quality.sh <tag> <concurrency> [VAR=value ...]
tag=$1; conc=$2; shift 2
root="${MYNAH_GPU_ROOT:-/root/mynah-head}"; ev="${MYNAH_GPU_EVIDENCE:-/root/evidence}"
port="${MYNAH_GPU_PORT:-18080}"
pkill -f "mynah-tts-server.*-p $port"; sleep 2; tmux kill-session -t "srv-$port" 2>/dev/null
tmux new-session -d -s "srv-$port" "MYNAH_GPU_CPUS=$MYNAH_GPU_CPUS MYNAH_GPU_WORKERS=$MYNAH_GPU_WORKERS $root/tools/gpu/serve.sh ${MYNAH_GPU_MODEL:-models/pocket-english-24l} 64 64 $port $* > $ev/$tag-server.log 2>&1"
for i in $(seq 90); do curl -sf "localhost:$port/health" >/dev/null && break; sleep 2; done
py="/venv/main/bin/python $root/tools/pocket_quality.py --out $ev/$tag"
flock /root/gpu.lock $py --phase synth --port "$port" --concurrency "$conc" > "$ev/$tag-quality.out" 2>&1
# The GPU is idle during ASR: stop the server first, then transcribe on CPU.
pkill -INT -f "mynah-tts-server.*-p $port"; sleep 4
flock /root/gpu.lock $py --phase asr >> "$ev/$tag-quality.out" 2>&1
echo "=== $tag $*"; grep -E "QUALITY|BAD" "$ev/$tag-quality.out"; tail -2 "$ev/$tag-quality.out" | grep -v QUALITY
echo stream_failed=$(grep -c "stream failed" "$ev/$tag-server.log")
