#!/bin/bash
while ! grep -q ABPF-DONE /root/evidence/ab-prefill.out 2>/dev/null; do sleep 10; done
cd /root/mynah-head
flock /root/gpu.lock make cuda-server cuda CUDA_ARCH=sm_89 -j16 > /root/evidence/build-tilews.log 2>&1; echo BUILD:$?
T=/root/mynah-head/tools/l4/ab.sh
$T tilews models/pocket-english-24l 32,48,64 5 20
echo ABTW-DONE
/root/evidence/nsys-c64b.sh
