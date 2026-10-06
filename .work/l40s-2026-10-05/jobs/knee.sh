#!/bin/bash
# usage: TREE=/root/m768 TAG=x LEVELS="384 512" PROCS=1 DUR=120 ENVS="A=1 B=2" knee.sh
set -u
W=${TREE:-/root/m768}; TAG=${TAG:-run}; LEVELS=${LEVELS:-384}; PROCS=${PROCS:-1}; DUR=${DUR:-120}; ENVS=${ENVS:-}
PY=/venv/main/bin/python; [ -x $PY ] || PY=python3
R=/root/res/$TAG; mkdir -p $R; PORT=18080
MAXC=0; for C in $LEVELS; do [ $C -gt $MAXC ] && MAXC=$C; done
pkill -f "[b]uild/cuda/mynah-tts-server"; sleep 2
cd $W
t0=$(date +%s)
tmux new-session -d -s srv "cd $W && env MYNAH_QUANT_GROUPS=none MYNAH_SERVE_PROFILE=1 MYNAH_THREADS=1 $ENVS ./build/cuda/mynah-tts-server --device cuda -w 8 --max-batch $MAXC --max-inflight $MAXC --max-pending 2048 -p $PORT -m models/pocket-english-24l > $R/server.log 2>&1"
ok=0; for _ in $(seq 1 300); do sleep 2; curl -fsS localhost:$PORT/health >/dev/null 2>&1 && { ok=1; break; }; done
[ $ok = 1 ] || { echo "$TAG: server did not start"; tail -5 $R/server.log; exit 1; }
SPID=$(pgrep -f "mynah-tts-server --device cuda" | head -1)
echo "== $TAG [$ENVS] procs=$PROCS ready in $(( $(date +%s) - t0 )) s, VRAM $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
for C in $LEVELS; do
  pgrep -f "mynah-tts-server --device cuda" >/dev/null || { echo "  C$C: SERVER DIED before this level"; tail -3 $R/server.log; break; }
  nvidia-smi dmon -s pu -d 1 > $R/dmon-c$C.log 2>&1 & DM=$!
  timeout $((DUR + 400)) $PY tools/pocket_ladder.py --port $PORT --levels $C --mode closed --warmup 15 --duration $DUR --timeout 300 \
    --corpus tools/corpus/pocket_v2_en.jsonl --voices alba,marius,javert,jean --seed 1234 --client-procs $PROCS \
    --server-pid $SPID --out $R --tag c$C > $R/ladder-c$C.log 2>&1
  kill $DM 2>/dev/null
  $PY - $R/c$C-summary.jsonl $R/dmon-c$C.log $C <<"PY"
import json,sys
s=json.loads(open(sys.argv[1]).read().strip().splitlines()[-1])
p=[];u=[]
for l in open(sys.argv[2]):
    f=l.split()
    if not f or f[0].startswith("#"): continue
    try: pw=float(f[1]); sm=float(f[4])
    except: continue
    if sm>5: p.append(pw); u.append(sm)
g=lambda k:s[k]["p95"]
d=s.get("metrics_delta",{}); ex=" ".join("%s=%d"%(k.replace("mynah_backend_decoder_graph_","dg_"),v) for k,v in d.items() if "rerecord" in k or "table_patch" in k)
print("  C%s aps %.0f rtf95 %.3f ttfa95 %.0fms gap95 %.0fms st250 %d fail %d | srv_cpu %.0f%% cli_cpu %.0f%% | gpu %.0fW sm %.0f%%" % (sys.argv[3], s["audio_s_per_s"], g("rtf_stream"), 1e3*g("ttfa"), 1e3*g("max_gap"), s["stalls_250"], s["failed"], s["server_cpu_pct_mean"] or -1, s["client_cpu_pct"], sum(p)/max(1,len(p)), sum(u)/max(1,len(u))) + ((" | "+ex) if ex else ""))
PY
done
tmux send-keys -t srv C-c; for _ in $(seq 1 30); do pgrep -f "mynah-tts-server --device" >/dev/null || break; sleep 1; done
pkill -f "[b]uild/cuda/mynah-tts-server"; tmux kill-session -t srv 2>/dev/null
grep -E "\[SERVE\] (device|loop)|step cost|B(384|512|640|768) " $R/server.log | head -12
