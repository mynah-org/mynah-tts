#!/bin/bash
while ! grep -q MODELS2-DONE /root/evidence/models2.out 2>/dev/null; do sleep 10; done
nproc; lscpu | grep -E "Model name|^CPU\(s\)"
cd /root/mynah-head
T=/root/mynah-head/tools/l4/ab.sh
$T g6-full models/pocket-english-24l 16,32,48,64 5 20
MYNAH_L4_CPUS=0-3 MYNAH_L4_WORKERS=4 $T g6-cpu4 models/pocket-english-24l 16,32,48,64 5 20
echo G6BASE-DONE
