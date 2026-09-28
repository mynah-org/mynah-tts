#!/bin/bash
T=/root/mynah-head/tools/l4/ab.sh
MYNAH_L4_BATCH=96 MYNAH_L4_CPUS=0-3 MYNAH_L4_WORKERS=4 $T g6-b96 models/pocket-english-24l 64,80,96 5 20
MYNAH_L4_BATCH=128 MYNAH_L4_CPUS=0-3 MYNAH_L4_WORKERS=4 $T g6-b128 models/pocket-english-24l 112,128 5 20
echo ABWIDE-DONE
