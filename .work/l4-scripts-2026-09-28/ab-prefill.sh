#!/bin/bash
cd /root/mynah-head
flock /root/gpu.lock make cuda-server cuda CUDA_ARCH=sm_89 -j16 > /root/evidence/build-prefill.log 2>&1; echo BUILD:$?
T=/root/mynah-head/tools/l4/ab.sh
$T pf-cublas models/pocket-english-24l 32,48,64,96 5 20 MYNAH_CUDA_PREFILL_FIXED=0
MYNAH_L4_BATCH=96 $T pf-fixed96 models/pocket-english-24l 96 5 20
MYNAH_L4_BATCH=96 $T pf-cublas96 models/pocket-english-24l 96 5 20 MYNAH_CUDA_PREFILL_FIXED=0
$T pf-fixed models/pocket-english-24l 32,48,64 5 20
/root/mynah-head/tools/l4/quality.sh q-pfcublas 64 MYNAH_CUDA_PREFILL_FIXED=0
echo ABPF-DONE
