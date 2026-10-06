#!/bin/bash
exec > /root/res/chain2.log 2>&1
while tmux has-session -t chain1 2>/dev/null; do sleep 10; done
echo "== chain2 start $(date -u +%T)"
TREE=/root/m768b TAG=e3-sites LEVELS=640 PROCS=1 timeout 1500 /root/jobs/knee.sh
grep -E "\[SERVE\]   sync" /root/res/e3-sites/server.log | sort -t"s" -k1 | head -40
echo "== chain2 done $(date -u +%T)"
