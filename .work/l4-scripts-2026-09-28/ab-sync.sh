#!/bin/bash
while ! grep -q G6BASE-DONE /root/evidence/g6base.out 2>/dev/null; do sleep 10; done
cd /root/mynah-head
flock /root/gpu.lock make cuda-server cuda CUDA_ARCH=sm_89 -j16 > /root/evidence/build-sync.log 2>&1; echo BUILD:$?
T=/root/mynah-head/tools/l4/ab.sh
for m in blocking yield; do
  MYNAH_L4_CPUS=0-3 MYNAH_L4_WORKERS=4 $T g6-sync-$m models/pocket-english-24l 16,32,48,64 5 20 MYNAH_CUDA_SYNC=$m
done
echo ABSYNC-DONE
