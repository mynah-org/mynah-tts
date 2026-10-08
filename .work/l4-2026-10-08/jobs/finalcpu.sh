#!/bin/bash
# Final CPU check of the merged branch: identity against the x86 branch tree, then two 2-min streaming-server screens.
set -u
mkdir -p /root/res/final; exec >> /root/res/final/cpu.log 2>&1
echo "== finalcpu start $(date -u +%T)"
T=/root/final; rm -rf $T; mkdir -p $T && tar xzf /root/final.tgz -C $T && ln -sfn /root/mt/models $T/models
cd $T && timeout 1800 make all server -j12 > /root/res/final/build-cpu.log 2>&1 && echo "CPU BUILD OK" || { echo "CPU BUILD FAILED"; tail -20 /root/res/final/build-cpu.log; exit 1; }
timeout 600 make test-c > /root/res/final/test-c.log 2>&1 && echo "test-c OK: $(grep -c PASS /root/res/final/test-c.log) PASS lines" || { echo "test-c FAILED"; tail -15 /root/res/final/test-c.log; }
for tr in x86final final; do /root/x86/cliident.sh $tr ref-$tr; done
flock /root/box.lock timeout 600 /root/x86/serve2.sh final fin24 pocket-english-24l 24 120 40 "--prefork 6 --prefork-threads 4 --max-batch 8"
flock /root/box.lock timeout 600 /root/x86/serve2.sh final fin6 pocket-english-base 36 120 40 "--prefork 11 --prefork-threads 2 --max-batch 8"
for f in /root/res/x86/srv-fin24-c24.txt /root/res/x86/srv-fin6-c36.txt; do grep -E "^ *[0-9]+ +[0-9]+/[0-9]+ " $f | tail -1; done
echo "== finalcpu done $(date -u +%T)"
