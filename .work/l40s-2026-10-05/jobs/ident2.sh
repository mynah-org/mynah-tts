#!/bin/bash
# Bit-identity A/B with deterministic batch composition:
#  A) CLI --batch 32 (one burst, fixed seeds): exercises the batched AR step + retire
#  B) server at C1 (one request at a time, deterministic ids/texts/seeds): exercises the streaming path
exec > /root/res/ident2.log 2>&1
W=/root/mflags; PY=/venv/main/bin/python; [ -x $PY ] || PY=python3; PORT=18080; M=models/pocket-english-24l
TEXT="The train leaves the central station at a quarter past eight and arrives just before noon, so there is plenty of time for lunch before the meeting starts."
ALL="MYNAH_CANCEL_CHECK_EVERY=4 MYNAH_STREAM_OUT_WRITEV=1 MYNAH_DUP_CHECK_EPOCH=1 MYNAH_CUDA_DECODER_TABLE_PATCH=1 MYNAH_CUDA_ONESYNC_SUBSET=1 MYNAH_CUDA_DEFERRED_RELEASE=1 MYNAH_STREAM_DELIVER_THREADS=4 MYNAH_CUDA_PCM_DIRECT=1 MYNAH_CUDA_HIDDEN_LAZY=1 MYNAH_CUDA_KV_TABLE_CACHE=1 MYNAH_CUDA_DECODER_VALIDATE_ONCE=1"
ARMS=("base|" "base2|" "ALL|$ALL" "L6|MYNAH_CUDA_DEFERRED_RELEASE=1" "L7L8|MYNAH_CANCEL_CHECK_EVERY=4 MYNAH_STREAM_OUT_WRITEV=1" "L10|MYNAH_DUP_CHECK_EPOCH=1" "L11|MYNAH_CUDA_DECODER_TABLE_PATCH=1" "L12|MYNAH_CUDA_ONESYNC_SUBSET=1" "L19|MYNAH_STREAM_DELIVER_THREADS=4" "L20|MYNAH_CUDA_PCM_DIRECT=1" "L21|MYNAH_CUDA_HIDDEN_LAZY=1" "L22|MYNAH_CUDA_KV_TABLE_CACHE=1" "L24|MYNAH_CUDA_DECODER_VALIDATE_ONCE=1")
cd $W
cmp_py() { $PY - "$@" <<"PYEOF"
import os,sys,hashlib
d,b,label,kind=sys.argv[1:5]
def load(x):
    return {f.split("-",1)[-1] if kind=="srv" else f:hashlib.sha256(open(os.path.join(x,f),"rb").read()).hexdigest() for f in os.listdir(x)} if os.path.isdir(x) else {}
a=load(d); r=load(b); c=sorted(set(a)&set(r))
print("  %-4s %-5s files %d common %d identical %d" % (kind,label,len(a),len(c),sum(a[k]==r[k] for k in c)))
PYEOF
}
for arm in "${ARMS[@]}"; do
  L="${arm%%|*}"; E="${arm#*|}"; R=/root/res/ident2/$L; rm -rf $R; mkdir -p $R/cli $R/srv
  echo "== $L [$E] $(date -u +%T)"
  env MYNAH_QUANT_GROUPS=none MYNAH_THREADS=1 $E timeout 600 ./build/cuda/mynah-tts --synthesize $M --text "$TEXT" --lang en --speaker 0 --seed 1000 --batch 32 --device cuda --output $R/cli/out.wav > $R/cli.log 2>&1 || { echo "  cli FAILED"; tail -3 $R/cli.log; }
  cmp_py $R/cli /root/res/ident2/base/cli $L cli
  pkill -f "[b]uild/cuda/mynah-tts-server"; sleep 2
  tmux new-session -d -s isrv "cd $W && env MYNAH_QUANT_GROUPS=none MYNAH_THREADS=1 $E ./build/cuda/mynah-tts-server --device cuda -w 8 --max-batch 16 --max-inflight 16 -p $PORT -m $M > $R/server.log 2>&1"
  ok=0; for _ in $(seq 1 150); do sleep 2; curl -fsS localhost:$PORT/health >/dev/null 2>&1 && { ok=1; break; }; done
  if [ $ok = 1 ]; then
    timeout 300 $PY tools/pocket_ladder.py --port $PORT --levels 1 --mode closed --warmup 0 --duration 40 --timeout 120 \
      --corpus tools/corpus/pocket_v2_en.jsonl --voices alba,marius --seed 1234 --save-audio $R/srvwav --out $R --tag x > $R/ladder.log 2>&1
    tmux send-keys -t isrv C-c; sleep 4; pkill -f "[b]uild/cuda/mynah-tts-server"; tmux kill-session -t isrv 2>/dev/null
    cmp_py $R/srvwav /root/res/ident2/base/srvwav $L srv
  else echo "  server did not start"; tail -3 $R/server.log; tmux kill-session -t isrv 2>/dev/null; fi
  grep -hiE "mynah-tts: .*(MYNAH_|epoch|patch|subset|lazy|deferred|deliver|direct|validate|table cache)|disabled|mismatch|error" $R/cli.log $R/server.log 2>/dev/null | sort -u | head -4 | sed "s/^/    /"
done
echo IDENT2-DONE
