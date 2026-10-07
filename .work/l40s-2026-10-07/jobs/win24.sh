#!/bin/bash
until grep -q "4-vCPU soak level" /root/res/soak.log 2>/dev/null; do sleep 15; done
/root/jobs/windows.sh /root/res/soak30 c1024 >> /root/res/soak.log 2>&1
