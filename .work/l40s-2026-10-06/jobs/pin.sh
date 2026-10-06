#!/bin/bash
# NUMA pin check: server on the GPU's node (48-71,144-167), clients on node 0.
exec >> /root/res/day2.log 2>&1
source <(sed -n '/^declare -A F/,/^ALL=/p' /root/jobs/day2.sh)
G=48-71,144-167; C=0-47,96-143
TAG=pin-all  LEVELS="832" PRE=640 PROCS=4 DUR=120 ENVS="$ALL" PIN=$G CLIPIN=$C timeout 900 /root/jobs/knee.sh
TAG=pin-base LEVELS="832" PRE=640 PROCS=4 DUR=120             PIN=$G CLIPIN=$C timeout 900 /root/jobs/knee.sh
echo "== pin done $(date -u +%T)"
