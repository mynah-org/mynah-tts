#!/bin/bash
# 2026-10-06: soak at C832, then leave-one-out attribution at C768/C896.
exec >> /root/res/day2.log 2>&1
K=/root/jobs/knee.sh
declare -A F=( [L6]=MYNAH_CUDA_DEFERRED_RELEASE=1 [L19]=MYNAH_STREAM_DELIVER_THREADS=4 [L21]=MYNAH_CUDA_HIDDEN_LAZY=1
  [L11]=MYNAH_CUDA_DECODER_TABLE_PATCH=1 [L12]=MYNAH_CUDA_ONESYNC_SUBSET=1 [L20]=MYNAH_CUDA_PCM_DIRECT=1
  [L22]=MYNAH_CUDA_KV_TABLE_CACHE=1 [L24]=MYNAH_CUDA_DECODER_VALIDATE_ONCE=1 [L10]=MYNAH_DUP_CHECK_EPOCH=1
  [L7]=MYNAH_CANCEL_CHECK_EVERY=4 [L8]=MYNAH_STREAM_OUT_WRITEV=1 )
ORDER="L6 L19 L21 L11 L12 L20 L22 L24 L10 L7 L8"
ALL=""; for k in $ORDER; do ALL="$ALL ${F[$k]}"; done
STEP=${STEP:-all}
echo "== day2 start $(date -u +%T) step=$STEP"
if [ $STEP = all ] || [ $STEP = soak ]; then
  TAG=soak-all  LEVELS=832 PRE=640 PROCS=4 DUR=600 ENVS="$ALL" timeout 1500 $K
  TAG=soak-base LEVELS=832 PRE=640 PROCS=4 DUR=600            timeout 1500 $K
fi
if [ $STEP = all ] || [ $STEP = loo ]; then
  TAG=loo-ref LEVELS="768 896" PRE=640 PROCS=4 DUR=120 ENVS="$ALL" timeout 900 $K
  for k in $ORDER; do
    E=""; for j in $ORDER; do [ $j = $k ] || E="$E ${F[$j]}"; done
    TAG=loo-no$k LEVELS="768 896" PRE=640 PROCS=4 DUR=120 ENVS="$E" timeout 900 $K
  done
  TAG=loo-ref2 LEVELS="768 896" PRE=640 PROCS=4 DUR=120 ENVS="$ALL" timeout 900 $K
fi
echo "== day2 done $(date -u +%T)"
