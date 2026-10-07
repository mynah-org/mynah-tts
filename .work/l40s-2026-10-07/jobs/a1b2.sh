#!/bin/bash
# Replace the old-A1b arm of jpc with A1b v2: stop jpc once the 4-vCPU arm has reported, then run v2.
until [ $(grep -c "helper thread" /root/res/jpc.log 2>/dev/null) -ge 2 ]; do sleep 5; done
tmux kill-session -t jpc; pkill -f "[j]pc.sh"; sleep 1; pkill -f "timeout 1500 /root/jobs/knee"; sleep 1; pkill -f "[p]ocket_ladder.py"; pkill -f "[n]vidia-smi dmon"
tmux send-keys -t srv C-c 2>/dev/null; sleep 4; pkill -f "[b]uild/cuda/mynah-tts-server"; tmux kill-session -t srv 2>/dev/null
exec >> /root/res/jpc.log 2>&1
echo "== a1b v2 start $(date -u +%T)"
T=/root/ma1b2; mkdir -p $T && cd $T && tar xzf /root/ma1b2.tgz 2>/dev/null && ln -sfn /root/mflags/models models
timeout 1800 make cuda cuda-server CUDA_ARCH=sm_89 ROW_CAP=1024 -j24 > /root/res/build-a1b2.log 2>&1 && echo "BUILD OK" || { echo "BUILD FAILED"; tail -20 /root/res/build-a1b2.log; exit 1; }
BEST="MYNAH_CUDA_STEP_OVERLAP=1 MYNAH_CUDA_DECODE_OVERLAP=1 MYNAH_CUDA_FAST_FIRST_CHUNK_WAIT_US=3000 MYNAH_CUDA_FIRST_FRAME_FIRST=1 MYNAH_CTX_HOST_POOL=1"
PIN=32-63,96-127 CLIPIN=0-31,64-95 TREE=$T TAG=jpc-a1b2 LEVELS="896 1024" PRE=768 PROCS=4 DUR=120 ENVS="$BEST MYNAH_CUDA_SLOT_FIXED=1" timeout 2100 /root/jobs/knee.sh
grep -h "SLOT_FIXED\|startup mark\|fixed cap\|\[CTX\]" /root/res/jpc-a1b2/server.log | tail -4 | cut -c1-420
echo "== a1b v2 done $(date -u +%T)"
