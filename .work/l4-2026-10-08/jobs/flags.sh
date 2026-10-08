#!/bin/bash
# A/B of the opt-in host flags at C320 on the L4, one flag at a time, one lock per run.
set -u
mkdir -p /root/res/flags; exec >> /root/res/flags/flags.log 2>&1
echo "== flags start $(date -u +%T)"
T=/root/merged; mkdir -p $T && tar xzf /root/merged.tgz -C $T && ln -sfn /root/mt/models $T/models
cd $T && timeout 1800 make cuda cuda-server CUDA_ARCH=sm_89 -j12 > /root/res/flags/build.log 2>&1 && echo "BUILD OK" || { echo "BUILD FAILED"; tail -20 /root/res/flags/build.log; exit 1; }
run() { # tag envs
  flock /root/box.lock timeout 600 env ROOT=$T OUT=/root/res/flags PY=/venv/main/bin/python PROCS=4 TAG=$1 LEVELS=320 DUR=120 ENVS="$2" THERM=1 $T/tools/gpu/knee_closed.sh 2>&1 | grep -E "^ *C320|did not|FAIL|error" | sed "s/^/$1 /"
}
for r in 1 2; do
  run off$r ""
  run pin$r "MYNAH_CUDA_PREFILL_PINNED=1"
  run stale$r "MYNAH_CUDA_MIMI_STALE_WINDOW=1"
  run both$r "MYNAH_CUDA_PREFILL_PINNED=1 MYNAH_CUDA_MIMI_STALE_WINDOW=1"
done
echo "== flags done $(date -u +%T)"
