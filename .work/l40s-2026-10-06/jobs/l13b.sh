#!/bin/bash
# L13b decode-ahead (+ L13d first frame first) on the L40S, ROW_CAP 1024 tree.
# 1. build; 2. Pocket gang self-check through the decode split; 3. identity
# (CLI offline burst, CLI streaming burst, server C1) against ALL; 4. speed,
# knee at C768/C896/C1024, arms B (ALL+L13), C0 (+L13b), E (+L13b, late wait
# 3000 us, + L13d), one repetition (time-boxed).
# Design and pass criteria: .work/pocket-l40s-1024-overlap-decode.md section 13.
exec >> /root/res/l13b.log 2>&1
echo "== l13b start $(date -u +%T)"
T=/root/ml13b
mkdir -p $T && cd $T && tar xzf /root/ml13b.tgz 2>/dev/null && ln -sfn /root/mflags/models models
timeout 1800 make cuda cuda-server CUDA_ARCH=sm_89 ROW_CAP=1024 -j24 > /root/res/build-l13b.log 2>&1 && echo "BUILD OK" || { echo "BUILD FAILED"; tail -20 /root/res/build-l13b.log; exit 1; }
source <(sed -n "/^declare -A F/,/^ALL=/p" /root/jobs/day2.sh)
L13="MYNAH_CUDA_STEP_OVERLAP=1"
L13B="$L13 MYNAH_CUDA_DECODE_OVERLAP=1"
PY=/venv/main/bin/python; PORT=18080; M=models/pocket-english-24l
TEXT="The train leaves the central station at a quarter past eight and arrives just before noon, so there is plenty of time for lunch before the meeting starts."

echo "== self-check through the split $(date -u +%T)"
env MYNAH_QUANT_GROUPS=none $ALL MYNAH_CUDA_DECODE_OVERLAP=1 timeout 900 ./build/cuda/mynah-tts --pocket-self-check $M --device cuda 2>&1 | grep -E "PASS|FAIL|failed|DECODE_OVERLAP" | sed "s/^/  /"

cmp_py() { $PY - "$@" <<"PYEOF"
import os,sys,hashlib
d,b,label,kind=sys.argv[1:5]
def load(x):
    return {f.split("-",1)[-1] if kind=="srv" else f:hashlib.sha256(open(os.path.join(x,f),"rb").read()).hexdigest() for f in os.listdir(x)} if os.path.isdir(x) else {}
a=load(d); r=load(b); c=sorted(set(a)&set(r))
print("  %-5s %-7s files %d common %d identical %d" % (kind,label,len(a),len(c),sum(a[k]==r[k] for k in c)))
PYEOF
}
I=/root/res/ident13b
for arm in "ALL|$ALL" "ALL2|$ALL" "L13|$ALL $L13" "L13B|$ALL $L13B" "L13BD|$ALL $L13B MYNAH_CUDA_FIRST_FRAME_FIRST=1" "L13BCHK|$ALL $L13B MYNAH_CUDA_DECODE_CHECK=1"; do
  L="${arm%%|*}"; E="${arm#*|}"; R=$I/$L; rm -rf $R; mkdir -p $R/cli $R/str $R/srv
  echo "== ident $L $(date -u +%T)"
  env MYNAH_QUANT_GROUPS=none MYNAH_THREADS=1 MYNAH_SERVE_PROFILE=1 $E timeout 600 ./build/cuda/mynah-tts --synthesize $M --text "$TEXT" --lang en --speaker 0 --seed 1000 --batch 32 --device cuda --output $R/cli/out.wav > $R/cli.log 2>&1 || { echo "  cli FAILED"; tail -5 $R/cli.log; }
  cmp_py $R/cli $I/ALL/cli $L cli
  env MYNAH_QUANT_GROUPS=none MYNAH_THREADS=1 MYNAH_SERVE_PROFILE=1 $E timeout 600 ./build/cuda/mynah-tts --synthesize $M --text "$TEXT" --lang en --speaker 0 --seed 1000 --batch 32 --stream --device cuda --output $R/str/out.wav > $R/str.log 2>&1 || { echo "  stream FAILED"; tail -5 $R/str.log; }
  cmp_py $R/str $I/ALL/str $L stream
  [ "$L" != L13 ] && cmp_py $R/str $I/L13/str "$L/L13" stream
  grep -h "step overlap\|decode overlap\|discard\|inside its submission" $R/str.log | head -4 | sed "s/^/    /"
  pkill -f "[b]uild/cuda/mynah-tts-server"; sleep 2
  tmux new-session -d -s isrv "cd $T && env MYNAH_QUANT_GROUPS=none MYNAH_THREADS=1 MYNAH_SERVE_PROFILE=1 $E ./build/cuda/mynah-tts-server --device cuda -w 8 --max-batch 16 --max-inflight 16 -p $PORT -m $M > $R/server.log 2>&1"
  ok=0; for _ in $(seq 1 150); do sleep 2; curl -fsS localhost:$PORT/health >/dev/null 2>&1 && { ok=1; break; }; done
  if [ $ok = 1 ]; then
    timeout 300 $PY tools/pocket_ladder.py --port $PORT --levels 1 --mode closed --warmup 0 --duration 40 --timeout 120 \
      --corpus tools/corpus/pocket_v2_en.jsonl --voices alba,marius --seed 1234 --save-audio $R/srvwav --out $R --tag x > $R/ladder.log 2>&1
    tmux send-keys -t isrv C-c; sleep 4; pkill -f "[b]uild/cuda/mynah-tts-server"; tmux kill-session -t isrv 2>/dev/null
    cmp_py $R/srvwav $I/ALL/srvwav $L srv
    grep -h "decode overlap\|discard\|inside its submission" $R/server.log | head -3 | sed "s/^/    /"
  else echo "  server did not start"; tail -5 $R/server.log; tmux kill-session -t isrv 2>/dev/null; fi
done

K=/root/jobs/knee.sh
export PIN=32-63,96-127 CLIPIN=0-31,64-95
B="$ALL $L13"; C0="$ALL $L13B"; C="$C0 MYNAH_CUDA_FAST_FIRST_CHUNK_WAIT_US=3000"; E="$C MYNAH_CUDA_FIRST_FRAME_FIRST=1"
for r in 1; do
  ORDER="B C0 E"
  for a in $ORDER; do
    TREE=$T TAG=l13b-$a$r LEVELS="768 896 1024" PRE=512 PROCS=4 DUR=120 ENVS="${!a}" timeout 1800 $K
    S=/root/res/l13b-$a$r/server.log
    grep -h "\[SERVE\] device wait\|\[SERVE\] step overlap\|\[SERVE\] decode overlap" $S | tail -3 | sed "s/^/    /"
    grep -h "while queued" $S | sed "s/^/    /" | head -8
  done
done
echo "== l13b done $(date -u +%T)"
