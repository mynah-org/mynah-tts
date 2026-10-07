#!/bin/bash
# Why is this host 4x slower? Same C768, 2-minute levels: 11 flags only, then + A1a, then the combined config.
exec >> /root/res/diag.log 2>&1
source <(sed -n "/^declare -A F/,/^ALL=/p" /root/jobs/day2.sh)
BEST="$ALL MYNAH_CUDA_STEP_OVERLAP=1 MYNAH_CUDA_DECODE_OVERLAP=1 MYNAH_CUDA_FAST_FIRST_CHUNK_WAIT_US=3000 MYNAH_CUDA_FIRST_FRAME_FIRST=1 MYNAH_CTX_HOST_POOL=1"
for arm in "all|$ALL" "a1a|$ALL MYNAH_CTX_HOST_POOL=1" "best|$BEST"; do L=${arm%%|*}; E=${arm#*|}
  TAG=diag-$L LEVELS="768" PRE=640 PROCS=4 DUR=120 ENVS="$E" timeout 900 /root/jobs/knee.sh
  grep -h "\[CTX\]" /root/res/diag-$L/server.log | tail -1 | cut -c1-300
done
echo "== diag done $(date -u +%T)"
