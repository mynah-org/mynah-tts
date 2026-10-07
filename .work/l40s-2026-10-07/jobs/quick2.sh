#!/bin/bash
# 3-minute thermal / sanity check with the combined best config at C768.
exec >> /root/res/quick.log 2>&1
source <(sed -n "/^declare -A F/,/^ALL=/p" /root/jobs/day2.sh)
BEST="$ALL MYNAH_CUDA_STEP_OVERLAP=1 MYNAH_CUDA_DECODE_OVERLAP=1 MYNAH_CUDA_FAST_FIRST_CHUNK_WAIT_US=3000 MYNAH_CUDA_FIRST_FRAME_FIRST=1 MYNAH_CTX_HOST_POOL=1"
nvidia-smi --query-gpu=temperature.gpu,clocks.sm,power.draw,clocks_throttle_reasons.active --format=csv,noheader -l 15 > /root/res/quick-therm.log 2>&1 & T=$!
TAG=quick-best LEVELS="768" PROCS=4 DUR=180 ENVS="$BEST" timeout 900 /root/jobs/knee.sh
kill $T; echo "therm (temp, sm clock, power, reasons):"; awk 'NR%3==0' /root/res/quick-therm.log
echo "== quick done $(date -u +%T)"
