#!/bin/bash
# After the regression A/B: ROW_CAP 512 build, all new wins on, 2-minute screen above the old 384 cap.
while tmux has-session -t l4c 2>/dev/null; do sleep 20; done
exec >> /root/res/l4c.log 2>&1
echo "== l4hi start $(date -u +%T)"
T=/root/m512; mkdir -p $T && cd $T && tar xzf /root/ml4.tgz 2>/dev/null && ln -sfn /root/mflags/models models
timeout 1800 make cuda cuda-server CUDA_ARCH=sm_89 ROW_CAP=512 -j24 > /root/res/build-512.log 2>&1 && echo "BUILD OK" || { echo "BUILD FAILED"; tail -20 /root/res/build-512.log; exit 1; }
ALL="MYNAH_CTX_HOST_POOL=1 MYNAH_CUDA_STEP_OVERLAP=1 MYNAH_CUDA_DECODE_OVERLAP=1 MYNAH_CUDA_FIRST_FRAME_FIRST=1 MYNAH_CUDA_SLOT_FIXED=1"
TREE=$T TAG=l4-hi LEVELS="352 384 416 448" PRE=320 PROCS=4 DUR=120 ENVS="$ALL" timeout 2400 /root/jobs/knee.sh
grep -h "SLOT_FIXED" /root/res/l4-hi/server.log | head -1 | cut -c1-260
echo "== l4hi done $(date -u +%T)"
