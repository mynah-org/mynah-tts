#!/bin/bash
# English 6-layer Pocket on the L40S, every win on: ROW_CAP 4096 build, coarse screen, refine near the knee,
# then a 30-min soak near the highest safe C (RTF p95 < 0.85, 0 stalls, 0 failures).

exec >> /root/res/six.log 2>&1
echo "== six start $(date -u +%T)"
T=/root/m6l2; mkdir -p $T && cd $T && tar xzf /root/ma1b2.tgz 2>/dev/null && ln -sfn /root/mflags/models models
timeout 1800 make cuda cuda-server CUDA_ARCH=sm_89 ROW_CAP=2048 -j24 > /root/res/build-6l.log 2>&1 && echo "BUILD OK" || { echo "BUILD FAILED"; tail -20 /root/res/build-6l.log; exit 1; }
MYNAH_QUANT_GROUPS=none timeout 900 ./build/cuda/mynah-tts --pocket-self-check models/pocket-english-6l --device cuda 2>&1 | tail -1
WIN="MYNAH_CTX_HOST_POOL=1 MYNAH_CUDA_STEP_OVERLAP=1 MYNAH_CUDA_DECODE_OVERLAP=1 MYNAH_CUDA_FIRST_FRAME_FIRST=1 MYNAH_CUDA_SLOT_FIXED=1 MYNAH_CUDA_WIDTH_BUCKETS=1,2,4,8,16,24,32,48,64,96,128,160,192,256,320,384,448,512,640,768,896,1024,1280,1536,1792,2048"
export MODEL=pocket-english-6l PIN=32-63,96-127 CLIPIN=0-31,64-95
pass() { /venv/main/bin/python -c "import json,sys;s=json.loads(open(sys.argv[1]).read().strip().splitlines()[-1]);print(1 if s[\"rtf_stream\"][\"p95\"]<0.85 and s[\"stalls_250\"]==0 and s[\"failed\"]==0 else 0)" "$1" 2>/dev/null; }
TREE=$T TAG=six-coarse LEVELS="1024 1536 2048" PRE=768 PROCS=4 DUR=120 ENVS="$WIN" timeout 5000 /root/jobs/knee6.sh
BEST=""; FAIL=""; for C in 1024 1536 2048; do f=/root/res/six-coarse/c$C-summary.jsonl; [ -s $f ] || { FAIL=$C; break; }; [ "$(pass $f)" = 1 ] && BEST=$C || { FAIL=$C; break; }; done
echo "== 6L coarse: highest pass C${BEST:-none}, first fail C${FAIL:-none (cap 2048 reached)}"
if [ -n "$BEST" ] && [ -n "$FAIL" ]; then
  MID=$(( (BEST + FAIL) / 2 / 128 * 128 )); if [ $MID -gt $BEST ]; then
    TREE=$T TAG=six-fine LEVELS="$MID" PRE=768 PROCS=4 DUR=120 ENVS="$WIN" timeout 2500 /root/jobs/knee6.sh
    [ "$(pass /root/res/six-fine/c$MID-summary.jsonl)" = 1 ] && BEST=$MID; fi
fi
echo "== 6L soak level: C${BEST:-none}"
[ -n "$BEST" ] || exit 0
TREE=$T TAG=six-soak LEVELS="$BEST" PRE=768 PROCS=4 DUR=1800 ENVS="$WIN" timeout 4500 /root/jobs/knee6.sh
/root/jobs/windows.sh /root/res/six-soak c$BEST
echo "== six done $(date -u +%T)"
