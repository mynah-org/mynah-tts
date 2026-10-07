#!/bin/bash
# usage: TREE=/root/m768b TAG=x ROWS=320 DUR=120 PROCS=2 twoeng.sh   (two servers, two clients, same GPU)
set -u
W=${TREE:-/root/m768b}; TAG=${TAG:-two}; ROWS=${ROWS:-320}; DUR=${DUR:-120}; PROCS=${PROCS:-2}
PY=/venv/main/bin/python; [ -x $PY ] || PY=python3
R=/root/res/$TAG; mkdir -p $R
pkill -f "[b]uild/cuda/mynah-tts-server"; sleep 2
cd $W
for P in 18080 18081; do
  tmux new-session -d -s srv$P "cd $W && env MYNAH_QUANT_GROUPS=none MYNAH_SERVE_PROFILE=1 MYNAH_THREADS=1 ./build/cuda/mynah-tts-server --device cuda -w 8 --max-batch $ROWS --max-inflight $ROWS --max-pending 2048 -p $P -m models/pocket-english-24l > $R/server-$P.log 2>&1"
  ok=0; for _ in $(seq 1 300); do sleep 2; curl -fsS localhost:$P/health >/dev/null 2>&1 && { ok=1; break; }; done
  [ $ok = 1 ] || { echo "$TAG: server $P did not start"; tail -5 $R/server-$P.log; exit 1; }
done
echo "== $TAG two servers x $ROWS rows ready, VRAM $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
nvidia-smi dmon -s pu -d 1 > $R/dmon.log 2>&1 & DM=$!
for P in 18080 18081; do
  SEED=$([ $P = 18080 ] && echo 1234 || echo 5678)
  timeout $((DUR + 400)) $PY tools/pocket_ladder.py --port $P --levels $ROWS --mode closed --warmup 15 --duration $DUR --timeout 300 \
    --corpus tools/corpus/pocket_v2_en.jsonl --voices alba,marius,javert,jean --seed $SEED --client-procs $PROCS --out $R --tag p$P > $R/ladder-$P.log 2>&1 &
done
wait $(jobs -p | grep -v "^$DM$") 2>/dev/null; sleep 1; kill $DM 2>/dev/null
$PY - $R <<"PY"
import json,sys,glob
R=sys.argv[1]; tot=0
for f in sorted(glob.glob(R+"/p*-summary.jsonl")):
    s=json.loads(open(f).read().strip().splitlines()[-1]); tot+=s["audio_s_per_s"]
    print("  %s C%d aps %.0f rtf95 %.3f ttfa95 %.0fms st250 %d fail %d cli_cpu %.0f%%" % (f.split("/")[-1][:6], s["offered"], s["audio_s_per_s"], s["rtf_stream"]["p95"], 1e3*s["ttfa"]["p95"], s["stalls_250"], s["failed"], s["client_cpu_pct"]))
p=[];u=[]
for l in open(R+"/dmon.log"):
    x=l.split()
    if not x or x[0].startswith("#"): continue
    try: pw=float(x[1]); sm=float(x[4])
    except: continue
    if sm>5: p.append(pw); u.append(sm)
print("  TOTAL aps %.0f | gpu %.0fW sm %.0f%%" % (tot, sum(p)/max(1,len(p)), sum(u)/max(1,len(u))))
PY
for P in 18080 18081; do tmux send-keys -t srv$P C-c; done; sleep 10
pkill -f "[b]uild/cuda/mynah-tts-server"; for P in 18080 18081; do tmux kill-session -t srv$P 2>/dev/null; done
for P in 18080 18081; do grep -E "\[SERVE\] device" $R/server-$P.log; done
