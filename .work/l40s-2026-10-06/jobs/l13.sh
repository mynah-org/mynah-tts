#!/bin/bash
# After the L4 chain: build the L13 tree, identity check (ALL vs ALL+L13), then A/B at C288/C320.
while tmux has-session -t l4 2>/dev/null; do sleep 20; done
exec >> /root/res/l13.log 2>&1
echo "== l13 start $(date -u +%T)"
mkdir -p /root/ml13 && cd /root/ml13 && tar xzf /root/ml13.tgz 2>/dev/null && ln -sfn /root/mflags/models models
timeout 1800 make cuda cuda-server CUDA_ARCH=sm_89 ROW_CAP=384 -j24 > /root/res/build-l13.log 2>&1 && echo "BUILD OK" || { echo "BUILD FAILED"; tail -20 /root/res/build-l13.log; exit 1; }
source <(sed -n "/^declare -A F/,/^ALL=/p" /root/jobs/day2.sh)
PY=/venv/main/bin/python; PORT=18080; M=models/pocket-english-24l
TEXT="The train leaves the central station at a quarter past eight and arrives just before noon, so there is plenty of time for lunch before the meeting starts."
cmp_py() { $PY - "$@" <<"PYEOF"
import os,sys,hashlib
d,b,label,kind=sys.argv[1:5]
def load(x):
    return {f.split("-",1)[-1] if kind=="srv" else f:hashlib.sha256(open(os.path.join(x,f),"rb").read()).hexdigest() for f in os.listdir(x)} if os.path.isdir(x) else {}
a=load(d); r=load(b); c=sorted(set(a)&set(r))
print("  %-4s %-6s files %d common %d identical %d" % (kind,label,len(a),len(c),sum(a[k]==r[k] for k in c)))
PYEOF
}
for arm in "ALL|$ALL" "ALL2|$ALL" "L13|$ALL MYNAH_CUDA_STEP_OVERLAP=1"; do
  L="${arm%%|*}"; E="${arm#*|}"; R=/root/res/ident13/$L; rm -rf $R; mkdir -p $R/cli $R/srv
  echo "== ident $L $(date -u +%T)"
  env MYNAH_QUANT_GROUPS=none MYNAH_THREADS=1 MYNAH_SERVE_PROFILE=1 $E timeout 600 ./build/cuda/mynah-tts --synthesize $M --text "$TEXT" --lang en --speaker 0 --seed 1000 --batch 32 --device cuda --output $R/cli/out.wav > $R/cli.log 2>&1 || { echo "  cli FAILED"; tail -5 $R/cli.log; }
  cmp_py $R/cli /root/res/ident13/ALL/cli $L cli
  grep -h "step overlap\|discard" $R/cli.log | head -3 | sed "s/^/    /"
  pkill -f "[b]uild/cuda/mynah-tts-server"; sleep 2
  tmux new-session -d -s isrv "cd /root/ml13 && env MYNAH_QUANT_GROUPS=none MYNAH_THREADS=1 MYNAH_SERVE_PROFILE=1 $E ./build/cuda/mynah-tts-server --device cuda -w 8 --max-batch 16 --max-inflight 16 -p $PORT -m $M > $R/server.log 2>&1"
  ok=0; for _ in $(seq 1 150); do sleep 2; curl -fsS localhost:$PORT/health >/dev/null 2>&1 && { ok=1; break; }; done
  if [ $ok = 1 ]; then
    timeout 300 $PY tools/pocket_ladder.py --port $PORT --levels 1 --mode closed --warmup 0 --duration 40 --timeout 120 \
      --corpus tools/corpus/pocket_v2_en.jsonl --voices alba,marius --seed 1234 --save-audio $R/srvwav --out $R --tag x > $R/ladder.log 2>&1
    tmux send-keys -t isrv C-c; sleep 4; pkill -f "[b]uild/cuda/mynah-tts-server"; tmux kill-session -t isrv 2>/dev/null
    cmp_py $R/srvwav /root/res/ident13/ALL/srvwav $L srv
    grep -h "step overlap\|discard" $R/server.log | head -3 | sed "s/^/    /"
  else echo "  server did not start"; tail -5 $R/server.log; tmux kill-session -t isrv 2>/dev/null; fi
done
K=/root/jobs/knee.sh
for r in 1 2; do
  TREE=/root/ml13 TAG=l13-all$r LEVELS="288 320 352" PRE=256 PROCS=4 DUR=120 ENVS="$ALL" timeout 1200 $K
  TREE=/root/ml13 TAG=l13-on$r  LEVELS="288 320 352" PRE=256 PROCS=4 DUR=120 ENVS="$ALL MYNAH_CUDA_STEP_OVERLAP=1" timeout 1200 $K
  grep -h "step overlap" /root/res/l13-on$r/server.log | tail -1
done
echo "== l13 done $(date -u +%T)"
