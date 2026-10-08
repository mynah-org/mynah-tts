#!/bin/bash
# The three audio-identity checks of docs/benchmarking.md section 3 for one arm.
# usage: ARM=name TREE=/root/warmup/new-384 ENVS="..." MAXB=16 ident.sh
# Run it under: flock /root/box.lock timeout 900 ident.sh
set -u
ARM=${ARM:?}; TREE=${TREE:?}; ENVS=${ENVS:-}; MAXB=${MAXB:-16}; PORT=${PORT:-18091}
PY=/venv/main/bin/python; [ -x $PY ] || PY=python3
O=/root/res/warmup/ident/$ARM; rm -rf $O; mkdir -p $O/cli $O/str $O/srv
cd $TREE || exit 1
M=models/pocket-english-24l
T="The train leaves the central station at a quarter past eight and arrives just before noon, so there is plenty of time for lunch before the meeting starts."
export MYNAH_QUANT_GROUPS=none MYNAH_THREADS=1
timeout 300 env $ENVS ./build/cuda/mynah-tts --synthesize $M --text "$T" --lang en --speaker 0 --seed 1000 \
  --batch 32 --device cuda --output $O/cli/out.wav > $O/cli.log 2>&1 || echo "$ARM: cli failed"
timeout 300 env $ENVS ./build/cuda/mynah-tts --synthesize $M --text "$T" --lang en --speaker 0 --seed 1000 \
  --batch 32 --stream --device cuda --output $O/str/out.wav > $O/str.log 2>&1 || echo "$ARM: stream failed"
( exec env $ENVS ./build/cuda/mynah-tts-server --device cuda -w 8 --max-batch $MAXB --max-inflight $MAXB \
    -p $PORT -m $M ) > $O/server.log 2>&1 &
SPID=$!
trap 'kill -INT $SPID 2>/dev/null; sleep 3; kill -9 $SPID 2>/dev/null' EXIT
for _ in $(seq 1 300); do sleep 1; kill -0 $SPID 2>/dev/null || break
  curl -m 5 -fsS localhost:$PORT/health >/dev/null 2>&1 && break; done
timeout 200 $PY tools/pocket_ladder.py --port $PORT --levels 1 --mode closed --warmup 0 --duration 40 \
  --timeout 120 --corpus tools/corpus/pocket_v2_en.jsonl --voices alba,marius --seed 1234 \
  --save-audio $O/srv --out $O --tag ident > $O/ladder.log 2>&1 || echo "$ARM: ladder failed"
echo "== ident $ARM [$ENVS] maxb=$MAXB: cli $(ls $O/cli | wc -l) str $(ls $O/str | wc -l) srv $(ls $O/srv | wc -l) wavs"
grep -h "width-bucket graph warm-up\|start-up phases" $O/server.log | cut -c1-300
