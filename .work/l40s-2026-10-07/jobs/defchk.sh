#!/bin/bash
# New defaults with NO env set: identity vs the explicit-env run of the old tree, then C288/C320.
while tmux has-session -t l4hi 2>/dev/null; do sleep 20; done
exec >> /root/res/l4c.log 2>&1
echo "== defchk start $(date -u +%T)"
T=/root/mdef2; mkdir -p $T && cd $T && tar xzf /root/mdef2.tgz 2>/dev/null && ln -sfn /root/mflags/models models
timeout 1800 make cuda cuda-server CUDA_ARCH=sm_89 ROW_CAP=384 -j24 > /root/res/build-def2.log 2>&1 && echo "BUILD OK" || { echo "BUILD FAILED"; tail -20 /root/res/build-def2.log; exit 1; }
MYNAH_QUANT_GROUPS=none timeout 900 ./build/cuda/mynah-tts --pocket-self-check models/pocket-english-24l --device cuda 2>&1 | tail -1
PY=/venv/main/bin/python; M=models/pocket-english-24l
TEXT="The train leaves the central station at a quarter past eight and arrives just before noon, so there is plenty of time for lunch before the meeting starts."
R=/root/res/idl4/NEWDEF; rm -rf $R; mkdir -p $R/cli $R/str
env MYNAH_QUANT_GROUPS=none MYNAH_THREADS=1 timeout 600 ./build/cuda/mynah-tts --synthesize $M --text "$TEXT" --lang en --speaker 0 --seed 1000 --batch 32 --device cuda --output $R/cli/out.wav > $R/cli.log 2>&1 || echo "  cli FAILED"
env MYNAH_QUANT_GROUPS=none MYNAH_THREADS=1 timeout 600 ./build/cuda/mynah-tts --synthesize $M --text "$TEXT" --lang en --speaker 0 --seed 1000 --batch 32 --stream --device cuda --output $R/str/out.wav > $R/str.log 2>&1 || echo "  stream FAILED"
for k in cli str; do $PY - $R/$k /root/res/idl4/ALL/$k $k <<"PYEOF"
import os,sys,hashlib
d,b,k=sys.argv[1:4]
h=lambda x:{f:hashlib.sha256(open(os.path.join(x,f),"rb").read()).hexdigest() for f in os.listdir(x)}
a=h(d); r=h(b); c=set(a)&set(r); print("  %-4s NEWDEF vs ALL(explicit) files %d identical %d"%(k,len(c),sum(a[x]==r[x] for x in c)))
PYEOF
done
grep -h "(default)" $R/str.log | cut -c1-140 | head -8
TREE=$T TAG=l4-newdef LEVELS="288 320" PRE=224 PROCS=4 DUR=120 timeout 1500 /root/jobs/knee.sh
echo "== defchk done $(date -u +%T)"
