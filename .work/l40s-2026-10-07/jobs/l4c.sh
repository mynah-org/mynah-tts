#!/bin/bash
# L4 regression check before promoting A1a, L13b/L13d and A1b to default.
until grep -q "PROV-DONE" /root/res/prov.log 2>/dev/null; do sleep 10; done
exec >> /root/res/l4c.log 2>&1
grep -q "self-check: PASS" /root/res/prov.log && grep -q "BUILD OK" /root/res/prov.log || { echo "== PROVISIONING FAILED"; exit 1; }
cd /root/mflags; PY=/venv/main/bin/python; PORT=18080; M=models/pocket-english-24l
A1A="MYNAH_CTX_HOST_POOL=1"; OVL="MYNAH_CUDA_STEP_OVERLAP=1 MYNAH_CUDA_DECODE_OVERLAP=1 MYNAH_CUDA_FIRST_FRAME_FIRST=1"; A1B="MYNAH_CUDA_SLOT_FIXED=1"
TEXT="The train leaves the central station at a quarter past eight and arrives just before noon, so there is plenty of time for lunch before the meeting starts."
cmp_py() { $PY - "$@" <<"PYEOF"
import os,sys,hashlib
d,b,label,kind=sys.argv[1:5]
def load(x):
    return {f.split("-",1)[-1] if kind=="srv" else f:hashlib.sha256(open(os.path.join(x,f),"rb").read()).hexdigest() for f in os.listdir(x)} if os.path.isdir(x) else {}
a=load(d); r=load(b); c=sorted(set(a)&set(r))
print("  %-6s %-6s files %d common %d identical %d" % (kind,label,len(a),len(c),sum(a[k]==r[k] for k in c)))
PYEOF
}
echo "== l4c start $(date -u +%T)"
I=/root/res/idl4
for arm in "DEF|" "ALL|$A1A $OVL $A1B"; do
  L="${arm%%|*}"; E="${arm#*|}"; R=$I/$L; rm -rf $R; mkdir -p $R/cli $R/str
  echo "== ident $L $(date -u +%T)"
  env MYNAH_QUANT_GROUPS=none MYNAH_THREADS=1 $E timeout 600 ./build/cuda/mynah-tts --synthesize $M --text "$TEXT" --lang en --speaker 0 --seed 1000 --batch 32 --device cuda --output $R/cli/out.wav > $R/cli.log 2>&1 || echo "  cli FAILED"
  cmp_py $R/cli $I/DEF/cli $L cli
  env MYNAH_QUANT_GROUPS=none MYNAH_THREADS=1 $E timeout 600 ./build/cuda/mynah-tts --synthesize $M --text "$TEXT" --lang en --speaker 0 --seed 1000 --batch 32 --stream --device cuda --output $R/str/out.wav > $R/str.log 2>&1 || echo "  stream FAILED"
  cmp_py $R/str $I/DEF/str $L stream
  tmux new-session -d -s isrv "cd /root/mflags && env MYNAH_QUANT_GROUPS=none MYNAH_THREADS=1 $E ./build/cuda/mynah-tts-server --device cuda -w 8 --max-batch 16 --max-inflight 16 -p $PORT -m $M > $R/server.log 2>&1"
  ok=0; for _ in $(seq 1 150); do sleep 2; curl -m 5 -fsS localhost:$PORT/health >/dev/null 2>&1 && { ok=1; break; }; done
  if [ $ok = 1 ]; then
    timeout 300 $PY tools/pocket_ladder.py --port $PORT --levels 1 --mode closed --warmup 0 --duration 40 --timeout 120 --corpus tools/corpus/pocket_v2_en.jsonl --voices alba,marius --seed 1234 --save-audio $R/srvwav --out $R --tag x > $R/ladder.log 2>&1
    tmux send-keys -t isrv C-c; sleep 4; tmux kill-session -t isrv 2>/dev/null
    cmp_py $R/srvwav $I/DEF/srvwav $L srv
  else echo "  server did not start"; tail -5 $R/server.log; tmux kill-session -t isrv 2>/dev/null; fi
done
K=/root/jobs/knee.sh
TAG=l4-def  LEVELS="256 288 320" PRE=224 PROCS=4 DUR=120                      timeout 1500 $K
TAG=l4-all  LEVELS="256 288 320" PRE=224 PROCS=4 DUR=120 ENVS="$A1A $OVL $A1B" timeout 1500 $K
grep -h "SLOT_FIXED\|\[CTX\]" /root/res/l4-all/server.log | tail -2 | cut -c1-300
TAG=l4-pool LEVELS="256 288 320" PRE=224 PROCS=4 DUR=120 ENVS="$A1A $A1B"      timeout 1500 $K
TAG=l4-def2 LEVELS="256 288 320" PRE=224 PROCS=4 DUR=120                      timeout 1500 $K
echo "== l4c done $(date -u +%T)"
