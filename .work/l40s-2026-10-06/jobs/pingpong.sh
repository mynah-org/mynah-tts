#!/bin/bash
# L26 ping-pong (MYNAH_CUDA_PINGPONG=2) on the L40S, ROW_CAP 1024 tree.
# 1. build; 2. identity: CLI offline and streaming bursts of 32 below the
# threshold (vs ALL, expect 32/32 and group B never used), the split
# (MYNAH_CUDA_PINGPONG_MIN=2, --batch 32 --seed 1000 vs two --batch 16 runs,
# seeds 1000 and 1016, expect 32/32 with pipelined steps), the split with
# MYNAH_CUDA_DECODE_CHECK=1 (race check: identical to the split), server C1
# ladder (identical); 3. speed, one repetition, knee at C768/C896/C1024:
# arm B = ALL+L13+L13b vs arm PP = ALL+PINGPONG. Every step is time-boxed.
# Design, status and pass criteria: .work/pocket-l40s-1024-pingpong.md
# sections 12-13.
exec >> /root/res/pingpong.log 2>&1
echo "== pingpong start $(date -u +%T)"
T=/root/mpp
mkdir -p $T && cd $T && tar xzf /root/mpp.tgz 2>/dev/null && ln -sfn /root/mflags/models models
timeout 1800 make cuda cuda-server CUDA_ARCH=sm_89 ROW_CAP=1024 -j24 > /root/res/build-pp.log 2>&1 && echo "BUILD OK" || { echo "BUILD FAILED"; tail -20 /root/res/build-pp.log; exit 1; }
source <(sed -n "/^declare -A F/,/^ALL=/p" /root/jobs/day2.sh)
# Common to every arm (design section 12): KV growth without a stream drain,
# and one explicit width-bucket list with half-width buckets, so that the
# groups (~half the rows each) do not pad to the next full-width bucket.
COMMON="MYNAH_CUDA_KV_VMM=1 MYNAH_CUDA_WIDTH_BUCKETS=1,2,4,8,16,24,32,48,64,96,128,160,192,256,320,384,448,512,576,640,704,768,896,1024"
BASE="$ALL $COMMON"
L13B="MYNAH_CUDA_STEP_OVERLAP=1 MYNAH_CUDA_DECODE_OVERLAP=1"
PP="MYNAH_CUDA_PINGPONG=2"
SPLIT="$PP MYNAH_CUDA_PINGPONG_MIN=2"
PY=/venv/main/bin/python; PORT=18080; M=models/pocket-english-24l
TEXT="The train leaves the central station at a quarter past eight and arrives just before noon, so there is plenty of time for lunch before the meeting starts."

cmp_py() { $PY - "$@" <<"PYEOF"
import os,sys,hashlib
d,b,label,kind=sys.argv[1:5]
def load(x):
    return {f.split("-",1)[-1] if kind=="srv" else f:hashlib.sha256(open(os.path.join(x,f),"rb").read()).hexdigest() for f in os.listdir(x)} if os.path.isdir(x) else {}
a=load(d); r=load(b); c=sorted(set(a)&set(r))
print("  %-5s %-9s files %d common %d identical %d" % (kind,label,len(a),len(c),sum(a[k]==r[k] for k in c)))
PYEOF
}
# The split run's requests 0-15 against the first half run, 16-31 against
# the second (CLI names: out.wav, out.wav.1, ...).
cmp_split() { $PY - "$@" <<"PYEOF"
import os,sys,hashlib
got,h1,h2,label=sys.argv[1:5]
half=16
def name(i): return "out.wav" if i==0 else "out.wav.%d" % i
def h(p): return hashlib.sha256(open(p,"rb").read()).hexdigest() if os.path.isfile(p) else None
same=0
for i in range(2*half):
    ref=os.path.join(h1 if i<half else h2, name(i if i<half else i-half))
    x=h(os.path.join(got,name(i))); same+=(x is not None and x==h(ref))
print("  %-9s %d/%d identical to the two halves served alone" % (label,same,2*half))
PYEOF
}
I=/root/res/identpp; rm -rf $I; mkdir -p $I
cli() { # label env batch seed sub(cli|str)
  local L=$1 E=$2 N=$3 S=$4 SUB=$5 X=""; [ $SUB = str ] && X=--stream
  local R=$I/$L/$SUB; rm -rf $R; mkdir -p $R
  env MYNAH_QUANT_GROUPS=none MYNAH_THREADS=1 MYNAH_SERVE_PROFILE=1 $E timeout 600 ./build/cuda/mynah-tts --synthesize $M --text "$TEXT" --lang en --speaker 0 --seed $S --batch $N $X --device cuda --output $R/out.wav > $I/$L/$SUB.log 2>&1 || { echo "  $L $SUB FAILED"; tail -5 $I/$L/$SUB.log; }
}
ppgrep() { grep -h "\[SERVE\] pingpong\|\[SERVE\]   group\|ignored\|discarded\|inside its submission" "$@" | cut -c1-400 | sed "s/^/    /"; }
# Start-up memory and the recoverable-failure path (section 13.7): group B's
# scratch line, the width-bucket warm-up delta (compare arm B with arm PP:
# same build, same flags but the ping-pong one), and any stale-error clear,
# one-sync retry or failed device-owned step.
memgrep() { grep -h "ping-pong groups\|width-bucket graph warm-up\|stale device error\|one-sync frame failed\|ONE_SYNC disabled\|device-owned row\|out of memory" "$@" | sort | uniq -c | sort -rn | head -12 | cut -c1-400 | sed "s/^/    /"; }
for SUB in cli str; do
  echo "== ident $SUB $(date -u +%T)"
  cli REF "$BASE" 32 1000 $SUB
  cli PP "$BASE $PP" 32 1000 $SUB
  cmp_py $I/PP/$SUB $I/REF/$SUB PP/REF $SUB
  ppgrep $I/PP/$SUB.log
  cli H1 "$BASE" 16 1000 $SUB
  cli H2 "$BASE" 16 1016 $SUB
  cli SPLIT "$BASE $SPLIT" 32 1000 $SUB
  cmp_split $I/SPLIT/$SUB $I/H1/$SUB $I/H2/$SUB SPLIT
  ppgrep $I/SPLIT/$SUB.log
  grep -h "while queued" $I/SPLIT/$SUB.log | head -6 | sed "s/^/    /"
  cli SPLITCHK "$BASE $SPLIT MYNAH_CUDA_DECODE_CHECK=1" 32 1000 $SUB
  cmp_py $I/SPLITCHK/$SUB $I/SPLIT/$SUB CHK/SPLIT $SUB
done
for arm in "SREF|$BASE" "SPP|$BASE $PP"; do
  L="${arm%%|*}"; E="${arm#*|}"; R=$I/$L; mkdir -p $R
  echo "== ident server $L $(date -u +%T)"
  pkill -f "[b]uild/cuda/mynah-tts-server"; sleep 2
  tmux new-session -d -s isrv "cd $T && env MYNAH_QUANT_GROUPS=none MYNAH_THREADS=1 MYNAH_SERVE_PROFILE=1 $E ./build/cuda/mynah-tts-server --device cuda -w 8 --max-batch 16 --max-inflight 16 -p $PORT -m $M > $R/server.log 2>&1"
  ok=0; for _ in $(seq 1 150); do sleep 2; curl -fsS localhost:$PORT/health >/dev/null 2>&1 && { ok=1; break; }; done
  if [ $ok = 1 ]; then
    timeout 300 $PY tools/pocket_ladder.py --port $PORT --levels 1 --mode closed --warmup 0 --duration 40 --timeout 120 \
      --corpus tools/corpus/pocket_v2_en.jsonl --voices alba,marius --seed 1234 --save-audio $R/srvwav --out $R --tag x > $R/ladder.log 2>&1
    tmux send-keys -t isrv C-c; sleep 4; pkill -f "[b]uild/cuda/mynah-tts-server"; tmux kill-session -t isrv 2>/dev/null
    [ $L = SPP ] && cmp_py $R/srvwav $I/SREF/srvwav SPP/SREF srv
    ppgrep $R/server.log
    memgrep $R/server.log
  else echo "  server did not start"; tail -5 $R/server.log; tmux kill-session -t isrv 2>/dev/null; fi
done

K=/root/jobs/knee.sh
export PIN=32-63,96-127 CLIPIN=0-31,64-95
B="$BASE $L13B"; P="$BASE $PP"
for r in 1; do
  for a in B P; do
    TREE=$T TAG=pp-$a$r LEVELS="768 896 1024" PRE=512 PROCS=4 DUR=120 ENVS="${!a}" timeout 1800 $K
    S=/root/res/pp-$a$r/server.log
    grep -h "\[SERVE\] device wait\|\[SERVE\] step overlap\|\[SERVE\] decode overlap" $S | tail -3 | cut -c1-400 | sed "s/^/    /"
    ppgrep $S
    memgrep $S
    grep -h "while queued" $S | sed "s/^/    /" | head -8
  done
done
echo "== pingpong done $(date -u +%T)"
