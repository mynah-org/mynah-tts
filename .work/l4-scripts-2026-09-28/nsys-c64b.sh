#!/bin/bash
while ! grep -q ABTW-DONE /root/evidence/ab-tilews.out 2>/dev/null; do sleep 10; done
cd /root/mynah-head; ev=/root/evidence/nsys; mkdir -p $ev
export MYNAH_THREADS=1 MYNAH_CUDA_KV_DTYPE=bf16 MYNAH_QUANT_GROUPS=none
exec 9>/root/gpu.lock; flock 9
tmux new-session -d -s nsrv "cd /root/mynah-head; MYNAH_THREADS=1 MYNAH_CUDA_KV_DTYPE=bf16 MYNAH_QUANT_GROUPS=none nsys profile -o $ev/c64b -f true --cuda-graph-trace=node --delay 25 --duration 12 -t cuda,nvtx,osrt ./build/cuda/mynah-tts-server --device cuda -w 8 --max-batch 64 --max-inflight 64 -p 18082 -m models/pocket-english-24l > $ev/server.log 2>&1"
for i in $(seq 90); do curl -sf localhost:18082/health >/dev/null && break; sleep 2; done
/venv/main/bin/python tools/pocket_ladder.py --port 18082 --levels 64 --warmup 10 --duration 30 --out $ev/ladder --tag nsys > $ev/ladder.out 2>&1
pkill -INT -f "[m]ynah-tts-server.*-p 18082"; sleep 20
nsys stats -r cuda_gpu_kern_sum,cuda_api_sum --format csv -o $ev/c64b $ev/c64b.nsys-rep > /dev/null 2>&1
python3 - <<PY
import csv,glob
for f in sorted(glob.glob("$ev/c64b_cuda_gpu_kern_sum*.csv"))+sorted(glob.glob("$ev/c64b_cuda_api_sum*.csv")):
    rows=list(csv.DictReader(open(f)))
    print("##", f.split("/")[-1])
    for r in rows[:18]:
        print("%6s%% %12s ns %8s x  %s" % (r.get("Time (%)"), r.get("Total Time (ns)"), r.get("Instances", r.get("Num Calls")), r.get("Name")[:90]))
PY
echo NSYS2-DONE
