#!/bin/bash
while ! grep -q GROWALL-DONE /root/evidence/ab-grow.out 2>/dev/null; do sleep 10; done
flock /root/gpu.lock /root/evidence/growtest2.sh
echo GROW2-DONE
