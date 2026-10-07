#!/bin/bash
# A1b fixed-size device request sets (MYNAH_CUDA_SLOT_FIXED) on a single-NUMA
# box (RTX 6000 Ada host or similar), ROW_CAP 1024 tree. Sizing v2: the cap is
# re-planned after the start-up walk and prefill (two "re-planned" lines).
# 1. build; 2. identity (CLI offline burst, CLI streaming burst, server C1)
# flag off vs on, plus the leak A/B (ZERO_KV) and a tiny fixed size that forces
# short takes and growth; 3. speed, knee at C768/C896/C1024, arms A (defaults + A1a)
# and B (A + A1b), order A B A, one repetition (time-boxed), [CTX] lines.
# Design and pass criteria: .work/pocket-l40s-1024-host-profile.md, A1b design.
exec >> /root/res/a1b.log 2>&1
echo "== a1b start $(date -u +%T)"
T=/root/ma1b
mkdir -p $T && cd $T && tar xzf /root/ma1b.tgz 2>/dev/null && ln -sfn /root/mflags/models models
timeout 1800 make cuda cuda-server CUDA_ARCH=sm_89 ROW_CAP=1024 -j$(nproc) > /root/res/build-a1b.log 2>&1 && echo "BUILD OK" || { echo "BUILD FAILED"; tail -20 /root/res/build-a1b.log; exit 1; }
PY=/venv/main/bin/python; [ -x $PY ] || PY=python3
PORT=18080; M=models/pocket-english-24l
TEXT="The train leaves the central station at a quarter past eight and arrives just before noon, so there is plenty of time for lunch before the meeting starts."
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used --format=csv,noheader

cmp_py() { $PY - "$@" <<"PYEOF"
import os,sys,hashlib
d,b,label,kind=sys.argv[1:5]
def load(x):
    return {f.split("-",1)[-1] if kind=="srv" else f:hashlib.sha256(open(os.path.join(x,f),"rb").read()).hexdigest() for f in os.listdir(x)} if os.path.isdir(x) else {}
a=load(d); r=load(b); c=sorted(set(a)&set(r))
print("  %-5s %-7s files %d common %d identical %d" % (kind,label,len(a),len(c),sum(a[k]==r[k] for k in c)))
PYEOF
}

A1A="MYNAH_CTX_HOST_POOL=1"
FIX="$A1A MYNAH_CUDA_SLOT_FIXED=1"
I=/root/res/identa1b
for arm in "OFF|$A1A" "OFF2|$A1A" "FIX|$FIX" "FIXZ|$FIX MYNAH_CUDA_SLOT_POOL_ZERO_KV=1" "FIX64|$FIX MYNAH_CUDA_SLOT_FIXED_POSITIONS=64 MYNAH_CUDA_KV_GROW_LOG=1"; do
  L="${arm%%|*}"; E="${arm#*|}"; R=$I/$L; rm -rf $R; mkdir -p $R/cli $R/str $R/srv
  echo "== ident $L [$E] $(date -u +%T)"
  env MYNAH_QUANT_GROUPS=none MYNAH_THREADS=1 MYNAH_SERVE_PROFILE=1 $E timeout 600 ./build/cuda/mynah-tts --synthesize $M --text "$TEXT" --lang en --speaker 0 --seed 1000 --batch 32 --device cuda --output $R/cli/out.wav > $R/cli.log 2>&1 || { echo "  cli FAILED"; tail -5 $R/cli.log; }
  cmp_py $R/cli $I/OFF/cli $L cli
  env MYNAH_QUANT_GROUPS=none MYNAH_THREADS=1 MYNAH_SERVE_PROFILE=1 $E timeout 600 ./build/cuda/mynah-tts --synthesize $M --text "$TEXT" --lang en --speaker 0 --seed 1000 --batch 32 --stream --device cuda --output $R/str/out.wav > $R/str.log 2>&1 || { echo "  stream FAILED"; tail -5 $R/str.log; }
  cmp_py $R/str $I/OFF/str $L stream
  grep -h "MYNAH_CUDA_SLOT_FIXED" $R/cli.log | head -2 | sed "s/^/    /"
  pkill -f "[b]uild/cuda/mynah-tts-server"; sleep 2
  tmux new-session -d -s isrv "cd $T && env MYNAH_QUANT_GROUPS=none MYNAH_THREADS=1 MYNAH_SERVE_PROFILE=1 $E ./build/cuda/mynah-tts-server --device cuda -w 8 --max-batch 16 --max-inflight 16 -p $PORT -m $M > $R/server.log 2>&1"
  ok=0; for _ in $(seq 1 150); do sleep 2; curl -fsS localhost:$PORT/health >/dev/null 2>&1 && { ok=1; break; }; done
  if [ $ok = 1 ]; then
    timeout 300 $PY tools/pocket_ladder.py --port $PORT --levels 1 --mode closed --warmup 0 --duration 40 --timeout 120 \
      --corpus tools/corpus/pocket_v2_en.jsonl --voices alba,marius --seed 1234 --save-audio $R/srvwav --out $R --tag x > $R/ladder.log 2>&1
    tmux send-keys -t isrv C-c; sleep 4; pkill -f "[b]uild/cuda/mynah-tts-server"; tmux kill-session -t isrv 2>/dev/null
    cmp_py $R/srvwav $I/OFF/srvwav $L srv
    grep -h "MYNAH_CUDA_SLOT_FIXED" $R/server.log | head -3 | sed "s/^/    /"
    grep -h "^\[CTX\]" $R/server.log | tail -1 | sed "s/^/    /"
    grep -c "KV grew" $R/server.log | sed "s/^/    KV growths: /"
  else echo "  server did not start"; tail -5 $R/server.log; tmux kill-session -t isrv 2>/dev/null; fi
done

K=/root/jobs/knee.sh
unset PIN CLIPIN   # single NUMA node: no pinning
A="$A1A"; B="$FIX"
for r in 1; do
  for a in B A; do
    n=1; while [ -d /root/res/a1b-$a$r-$n ]; do n=$((n+1)); done
    TAG=a1b-$a$r-$n
    TREE=$T TAG=$TAG LEVELS="768 896 1024" PRE=640 PROCS=4 DUR=120 ENVS="${!a}" timeout 2100 $K
    S=/root/res/$TAG/server.log
    grep -h "MYNAH_CUDA_SLOT_FIXED" $S | head -3 | sed "s/^/    /"
    grep -h "^\[CTX\]" $S | tail -1 | sed "s/^/    /"
    grep -h "width-bucket graph warm-up\|slot-pool prefill" $S | sed "s/^/    /"
    grep -h "\[SERVE\] device wait" $S | tail -1 | sed "s/^/    /"
    grep -h "out of memory" $S | head -3 | sed "s/^/    /"
  done
done
echo "== a1b done $(date -u +%T)"
