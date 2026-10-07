#!/bin/bash
exec > /root/res/build2.log 2>&1
( cd /root/m768b && timeout 1800 make cuda cuda-server CUDA_ARCH=sm_89 ROW_CAP=768 -j24 > /root/res/build-768b.log 2>&1 && echo "BUILD 768b OK" || { echo "BUILD 768b FAILED"; tail -15 /root/res/build-768b.log; } ) &
( cd /root/m1024 && timeout 1800 make cuda cuda-server CUDA_ARCH=sm_89 ROW_CAP=1024 -j24 > /root/res/build-1024.log 2>&1 && echo "BUILD 1024 OK" || { echo "BUILD 1024 FAILED"; tail -15 /root/res/build-1024.log; } ) &
wait; echo BUILD2-DONE
