#!/bin/bash
# Vast.ai L40S (Xeon Gold 6430, 2 NUMA nodes): leave-one-out at C768/C896, then L13 on/off.
# Server pinned to the GPU NUMA node, clients on the other node.
exec >> /root/res/jp.log 2>&1
source <(sed -n "/^declare -A F/,/^ALL=/p" /root/jobs/day2.sh)
export PIN=32-63,96-127 CLIPIN=0-31,64-95
ORDER="L6 L19 L21 L11 L12 L20 L22 L24 L10 L7 L8"
LV="768 896"; K=/root/jobs/knee.sh
echo "== jp start $(date -u +%T)"
TAG=jp-ref  LEVELS="$LV" PRE=640 PROCS=4 DUR=120 ENVS="$ALL" timeout 900 $K
TAG=jp-base LEVELS="$LV" PRE=640 PROCS=4 DUR=120             timeout 900 $K
for k in $ORDER; do
  E=""; for j in $ORDER; do [ $j = $k ] || E="$E ${F[$j]}"; done
  TAG=jp-no$k LEVELS="$LV" PRE=640 PROCS=4 DUR=120 ENVS="$E" timeout 900 $K
done
TAG=jp-ref2  LEVELS="$LV" PRE=640 PROCS=4 DUR=120 ENVS="$ALL" timeout 900 $K
TAG=jp-base2 LEVELS="$LV" PRE=640 PROCS=4 DUR=120             timeout 900 $K
TAG=jp-l13   LEVELS="768 896 1024" PRE=640 PROCS=4 DUR=120 ENVS="$ALL MYNAH_CUDA_STEP_OVERLAP=1" timeout 1200 $K
TAG=jp-all3  LEVELS="768 896 1024" PRE=640 PROCS=4 DUR=120 ENVS="$ALL" timeout 1200 $K
grep -h "step overlap" /root/res/jp-l13/server.log | tail -1
echo "== jp done $(date -u +%T)"
