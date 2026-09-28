#!/bin/bash
while ! grep -q ABDECG-DONE /root/evidence/ab-decg.out 2>/dev/null; do sleep 10; done
/venv/main/bin/python -m pip install -q faster-whisper jiwer 2>&1 | tail -1
cd /root/mynah-head
flock /root/gpu.lock make cuda-server cuda CUDA_ARCH=sm_89 -j16 > /root/evidence/build-reuse.log 2>&1; echo BUILD:$?
grep -E "error|warning: unused" /root/evidence/build-reuse.log | head
Q=/root/mynah-head/tools/l4/quality.sh
$Q q-reuse1 64
$Q q-reuse0 64 MYNAH_CUDA_DECODER_GRAPH_REUSE=0
T=/root/mynah-head/tools/l4/ab.sh
MYNAH_L4_CPUS=0-3 MYNAH_L4_WORKERS=4 $T g6-reuse1 models/pocket-english-24l 32,48,64 5 20
MYNAH_L4_CPUS=0-3 MYNAH_L4_WORKERS=4 $T g6-reuse0 models/pocket-english-24l 32,48,64 5 20 MYNAH_CUDA_DECODER_GRAPH_REUSE=0
echo ABREUSE-DONE
