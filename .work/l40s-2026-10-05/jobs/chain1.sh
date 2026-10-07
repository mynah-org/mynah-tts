#!/bin/bash
exec > /root/res/chain1.log 2>&1
K=/root/jobs/knee.sh
echo "== chain1 start $(date -u +%T)"
TREE=/root/m768 TAG=e0-p1 LEVELS=640 PROCS=1 timeout 1500 $K
TREE=/root/m768 TAG=e0-p4 LEVELS=640 PROCS=4 timeout 1500 $K
TREE=/root/m768 TAG=e1-768 LEVELS="384 512 640 768" PROCS=4 timeout 3000 $K
TREE=/root/m384 TAG=e1-384 LEVELS=384 PROCS=4 timeout 1500 $K
TREE=/root/m768 TAG=e2-flags LEVELS=640 PROCS=4 ENVS="MYNAH_STREAM_OUT_WRITEV=1 MYNAH_CANCEL_CHECK_EVERY=8" timeout 1500 $K
echo "== chain1 done $(date -u +%T)"
