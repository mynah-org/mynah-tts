#!/bin/bash
# A1a host-context pool (+ the fd-limit fix): build, identity at MYNAH_CTX_HOST_POOL=0/1/2, A/B at C768/C896/C1024.
# The knee copy used here does NOT raise ulimit: the server must raise its own fd limit now.
while tmux has-session -t l13b 2>/dev/null; do sleep 20; done
exec >> /root/res/a1a.log 2>&1
echo "== a1a start $(date -u +%T)"
T=/root/ma1a; mkdir -p $T && cd $T && tar xzf /root/ma1a.tgz 2>/dev/null && ln -sfn /root/mflags/models models
timeout 1800 make cuda cuda-server CUDA_ARCH=sm_89 ROW_CAP=1024 -j24 > /root/res/build-a1a.log 2>&1 && echo "BUILD OK" || { echo "BUILD FAILED"; tail -20 /root/res/build-a1a.log; exit 1; }
sed 's/ulimit -n 65536; //' /root/jobs/knee.sh > /root/jobs/knee-noul.sh; chmod +x /root/jobs/knee-noul.sh
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
I=/root/res/identa1a
for arm in "P0|$ALL" "P1|$ALL MYNAH_CTX_HOST_POOL=1" "P2|$ALL MYNAH_CTX_HOST_POOL=2"; do
  L="${arm%%|*}"; E="${arm#*|}"; R=$I/$L; rm -rf $R; mkdir -p $R/cli
  echo "== ident $L $(date -u +%T)"
  env MYNAH_QUANT_GROUPS=none MYNAH_THREADS=1 $E timeout 600 ./build/cuda/mynah-tts --synthesize $M --text "$TEXT" --lang en --speaker 0 --seed 1000 --batch 32 --device cuda --output $R/cli/out.wav > $R/cli.log 2>&1 || { echo "  cli FAILED"; tail -5 $R/cli.log; }
  cmp_py $R/cli $I/P0/cli $L cli
  pkill -f "[b]uild/cuda/mynah-tts-server"; sleep 2
  tmux new-session -d -s isrv "cd $T && env MYNAH_QUANT_GROUPS=none MYNAH_THREADS=1 MYNAH_SERVE_PROFILE=1 $E ./build/cuda/mynah-tts-server --device cuda -w 8 --max-batch 16 --max-inflight 16 -p $PORT -m $M > $R/server.log 2>&1"
  ok=0; for _ in $(seq 1 150); do sleep 2; curl -fsS localhost:$PORT/health >/dev/null 2>&1 && { ok=1; break; }; done
  if [ $ok = 1 ]; then
    timeout 300 $PY tools/pocket_ladder.py --port $PORT --levels 1 --mode closed --warmup 0 --duration 40 --timeout 120 \
      --corpus tools/corpus/pocket_v2_en.jsonl --voices alba,marius --seed 1234 --save-audio $R/srvwav --out $R --tag x > $R/ladder.log 2>&1
    tmux send-keys -t isrv C-c; sleep 4; pkill -f "[b]uild/cuda/mynah-tts-server"; tmux kill-session -t isrv 2>/dev/null
    cmp_py $R/srvwav $I/P0/srvwav $L srv
    grep -h "host_pool\|open-file" $R/server.log | head -2 | cut -c1-200 | sed "s/^/    /"
  else echo "  server did not start"; tail -5 $R/server.log; tmux kill-session -t isrv 2>/dev/null; fi
done
export PIN=32-63,96-127 CLIPIN=0-31,64-95
K=/root/jobs/knee-noul.sh
TREE=$T TAG=a1a-on  LEVELS="768 896 1024" PRE=640 PROCS=4 DUR=120 ENVS="$ALL MYNAH_CTX_HOST_POOL=1" timeout 1800 $K
grep -h "open-file\|accept:\|\[CTX\]" /root/res/a1a-on/server.log | cut -c1-260 | sed "s/^/    /"
TREE=$T TAG=a1a-off LEVELS="768 896 1024" PRE=640 PROCS=4 DUR=120 ENVS="$ALL" timeout 1800 $K
grep -h "open-file\|accept:\|\[CTX\]" /root/res/a1a-off/server.log | cut -c1-260 | sed "s/^/    /"
echo "== a1a done $(date -u +%T)"
