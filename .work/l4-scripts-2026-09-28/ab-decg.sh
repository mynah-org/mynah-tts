#!/bin/bash
while ! grep -q ABSYNC-DONE /root/evidence/ab-sync.out 2>/dev/null; do sleep 10; done
T=/root/mynah-head/tools/l4/ab.sh
MYNAH_L4_CPUS=0-3 MYNAH_L4_WORKERS=4 $T g6-decg0 models/pocket-english-24l 32,48,64 5 20 MYNAH_CUDA_DECODER_GRAPHS=0
echo ABDECG-DONE
