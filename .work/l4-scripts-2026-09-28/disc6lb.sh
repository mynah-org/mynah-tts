#!/bin/bash
cd /root/mynah-head
flock /root/gpu.lock make cuda-server cuda CUDA_ARCH=sm_89 -j16 > /root/evidence/build-6lb.log 2>&1; echo BUILD:$?
MYNAH_L4_BATCH=352 /root/mynah-head/tools/l4/ab.sh disc6lb models/pocket-english-6l 256,288,320,352 5 25
echo DISC6LB-DONE
