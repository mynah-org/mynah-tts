#!/bin/bash
# Japan L40S, defaults tree (8c8cc74): C1024 confirm, 4-vCPU threshold, A1b at 1024 rows.
until grep -q "quick done" /root/res/quick.log 2>/dev/null; do sleep 10; done
exec >> /root/res/jpc.log 2>&1
BEST="MYNAH_CUDA_STEP_OVERLAP=1 MYNAH_CUDA_DECODE_OVERLAP=1 MYNAH_CUDA_FAST_FIRST_CHUNK_WAIT_US=3000 MYNAH_CUDA_FIRST_FRAME_FIRST=1 MYNAH_CTX_HOST_POOL=1"
K=/root/jobs/knee.sh
echo "== jpc start $(date -u +%T)"
PIN=32-63,96-127 CLIPIN=0-31,64-95 TAG=jpc-best LEVELS="896 1024" PRE=768 PROCS=4 DUR=120 ENVS="$BEST" timeout 1500 $K
grep -h "helper thread" /root/res/jpc-best/server.log | head -1 | cut -c1-160
PIN=32,33,96,97 CLIPIN=0-31,64-95 TAG=jpc-v4 LEVELS="512 640 768" PRE=384 PROCS=4 DUR=120 ENVS="$BEST" timeout 1500 $K
grep -h "helper thread" /root/res/jpc-v4/server.log | head -1 | cut -c1-160
PIN=32-63,96-127 CLIPIN=0-31,64-95 TAG=jpc-a1b LEVELS="896 1024" PRE=768 PROCS=4 DUR=120 ENVS="$BEST MYNAH_CUDA_SLOT_FIXED=1" timeout 1500 $K
grep -h "SLOT_FIXED\|\[CTX\]" /root/res/jpc-a1b/server.log | tail -2 | cut -c1-400
echo "== jpc done $(date -u +%T)"
