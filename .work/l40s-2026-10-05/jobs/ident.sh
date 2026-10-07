#!/bin/bash
# Bit-identity A/B in pedantic (batch-invariant) mode: C8, fixed seed, WAVs per request id vs the base arm.
exec > /root/res/ident.log 2>&1
W=/root/mflags; PY=/venv/main/bin/python; [ -x $PY ] || PY=python3; PORT=18080
PED="MYNAH_CUDA_TF32=0 MYNAH_CUDA_SEANET_BF16=0 MYNAH_CUDA_QUANT=f32 MYNAH_CUDA_ATTN_SPLIT=0 MYNAH_CUDA_KV_DTYPE=f32"
ARMS=("base|" "base2|" "L7L8|MYNAH_CANCEL_CHECK_EVERY=4 MYNAH_STREAM_OUT_WRITEV=1" "L10|MYNAH_DUP_CHECK_EPOCH=1" "L11|MYNAH_CUDA_DECODER_TABLE_PATCH=1" "L12|MYNAH_CUDA_ONESYNC_SUBSET=1" "L6|MYNAH_CUDA_DEFERRED_RELEASE=1" "L19|MYNAH_STREAM_DELIVER_THREADS=4" "L20|MYNAH_CUDA_PCM_DIRECT=1" "L21|MYNAH_CUDA_HIDDEN_LAZY=1" "L22|MYNAH_CUDA_KV_TABLE_CACHE=1" "L24|MYNAH_CUDA_DECODER_VALIDATE_ONCE=1")
ALL="MYNAH_CANCEL_CHECK_EVERY=4 MYNAH_STREAM_OUT_WRITEV=1 MYNAH_DUP_CHECK_EPOCH=1 MYNAH_CUDA_DECODER_TABLE_PATCH=1 MYNAH_CUDA_ONESYNC_SUBSET=1 MYNAH_CUDA_DEFERRED_RELEASE=1 MYNAH_STREAM_DELIVER_THREADS=4 MYNAH_CUDA_PCM_DIRECT=1 MYNAH_CUDA_HIDDEN_LAZY=1 MYNAH_CUDA_KV_TABLE_CACHE=1 MYNAH_CUDA_DECODER_VALIDATE_ONCE=1"
ARMS+=("ALL|$ALL")
cd $W
for arm in "${ARMS[@]}"; do
  L="${arm%%|*}"; E="${arm#*|}"; R=/root/res/ident/$L; rm -rf $R; mkdir -p $R
  pkill -f "[b]uild/cuda/mynah-tts-server"; sleep 2
  tmux new-session -d -s isrv "cd $W && env MYNAH_QUANT_GROUPS=none MYNAH_THREADS=1 $PED $E ./build/cuda/mynah-tts-server --device cuda -w 8 --max-batch 64 --max-inflight 64 -p $PORT -m models/pocket-english-24l > $R/server.log 2>&1"
  ok=0; for _ in $(seq 1 150); do sleep 2; curl -fsS localhost:$PORT/health >/dev/null 2>&1 && { ok=1; break; }; done
  [ $ok = 1 ] || { echo "$L: server did not start"; tail -5 $R/server.log; tmux kill-session -t isrv; continue; }
  timeout 300 $PY tools/pocket_ladder.py --port $PORT --levels 8 --mode closed --warmup 0 --duration 30 --timeout 120 \
    --corpus tools/corpus/pocket_v2_en.jsonl --voices alba,marius --seed 1234 --save-audio $R/wav --out $R --tag x > $R/ladder.log 2>&1
  tmux send-keys -t isrv C-c; sleep 5; pkill -f "[b]uild/cuda/mynah-tts-server"; tmux kill-session -t isrv 2>/dev/null
  $PY - $R /root/res/ident/base "$L" <<"PY"
import os,sys,hashlib,json
R,B,L=sys.argv[1:4]
def load(d):
    d=os.path.join(d,"wav")
    return {f:hashlib.sha256(open(os.path.join(d,f),"rb").read()).hexdigest() for f in os.listdir(d)} if os.path.isdir(d) else {}
a=load(R); b=load(B); c=sorted(set(a)&set(b))
s=json.loads(open(R+"/x-summary.jsonl").read().strip().splitlines()[-1])
lines=[l.strip() for l in open(R+"/server.log") if "mynah-tts:" in l and ("MYNAH_" in l or "flag" in l.lower() or "on" in l.split())][-0:]
print("%-5s wavs %d common-with-base %d identical %d | done %d fail %d" % (L,len(a),len(c),sum(a[k]==b[k] for k in c),s["completed"],s["failed"]))
PY
  grep -iE "error|warn|disabled|mismatch" $R/server.log | head -3
done
echo IDENT-DONE
