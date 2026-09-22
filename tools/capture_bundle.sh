#!/bin/bash
# Capture a client listening bundle from a server that is genuinely under load.
#
# The whole point of the bundle is that these are NOT quiet-machine renders. A
# load generator saturates the box at the certified concurrency for the entire
# capture, and every sample is pulled through the same streaming route a real
# client uses, while the other 126 streams are in flight. Audio rendered on an
# idle machine and presented as "under load" would be a lie the listener cannot
# detect, which is exactly why it must not be produced that way.
#
#   bash tools/capture_bundle.sh [CONCURRENCY] [PORT] [SECONDS]
#   PREFORK_W=12 PREFORK_T=2 bash tools/capture_bundle.sh 80
#
# Run it from the repo root on the serving host. Produces ~/bundle<C>/ with the
# WAVs, a manifest naming every sentence, and the load conditions they were
# captured under.
#
# THE TOPOLOGY IS AN ARGUMENT, and it had to become one. This script described
# itself as generic while hardcoding `--prefork 16 --prefork-threads 2`: the
# qualified cut of a 32-core Axion. Run unchanged on the 24-core EPYC it would
# have oversubscribed the box by a third and captured audio under a load
# nobody certified -- which is precisely the lie the header above says this
# file exists to avoid.
set -u
C="${1:-126}"
PORT="${2:-9200}"
SECONDS_OF_LOAD="${3:-480}"
PREFORK_W="${PREFORK_W:-16}"
PREFORK_T="${PREFORK_T:-2}"

cd "$(dirname "$0")/.." || exit 1
OUT="$HOME/bundle$C"
rm -rf "$OUT"; mkdir -p "$OUT/streaming-under-load"
LOG="$HOME/capture-$(date +%Y%m%d-%H%M).log"
exec > >(tee -a "$LOG") 2>&1
echo "=== $(date -Is) | $(git log --oneline -1) | C$C port $PORT ==="

clear_box() {
  # NOT pkill -x: Linux truncates /proc/<pid>/comm to 15 chars, so the 16-char
  # name never matches and pkill reports success while the servers keep running.
  for _ in 1 2 3; do
    local pids; pids=$(ps -C mynah-tts-server -o pid= 2>/dev/null | tr -d ' ')
    [ -z "$pids" ] && break
    for p in $pids; do kill -9 "$p" 2>/dev/null; done
    sleep 3
  done
  echo "box clear: $(ps -C mynah-tts-server -o pid= 2>/dev/null | wc -l) servers"
}

clear_box
echo "topology: prefork ${PREFORK_W} x ${PREFORK_T} threads = $((PREFORK_W*PREFORK_T)) cores, \
$((PREFORK_W*8)) admission slots"
./build/cpu/mynah-tts-server -m models/pocket-en -p "$PORT" \
    --prefork "$PREFORK_W" --prefork-threads "$PREFORK_T" --max-batch 8 \
    >"$HOME/srv-bundle.log" 2>&1 &
for _ in $(seq 40); do curl -sf "localhost:$PORT/health" >/dev/null && break; sleep 1; done
curl -sf "localhost:$PORT/health" >"$OUT/health-before.json" || { echo "server never came up"; exit 1; }
echo "server up: $(cat "$OUT/health-before.json")"

# Saturate the box for the whole capture. --url attaches to the server we just
# started rather than launching a second one.
# C-1, not C. The capture pulls one stream of its own, so a generator at C
# would make the real instantaneous concurrency C+1 -- above the level the
# bundle claims to represent. The first run of this script did exactly that and
# the generator came back NOT STREAMABLE, which is the correct verdict for C127
# and the wrong condition for the bundle.
python3 tools/serving_profile.py --url "http://127.0.0.1:$PORT" \
    --mode soak --bank tests/load_texts_en_v2.txt --levels "$((C-1))" \
    --soak-seconds "$SECONDS_OF_LOAD" --warmup-seconds 20 \
    --window-seconds "$((SECONDS_OF_LOAD / 4))" \
    >"$HOME/loadgen.log" 2>&1 &
LOADPID=$!
sleep 45

# THE CHECK THAT WAS MISSING, AND THE ONE FAILURE THIS SCRIPT CANNOT SURVIVE.
#
# The generator is a background process. On 2026-09-22 it REFUSED to start --
# `--soak-seconds 480 cannot contain the 3 windows of 180 s the drift gate
# needs` -- and this script sailed past it and captured forty-eight files on an
# idle box, then printed "load at C80, loadavg 1.73". The number that
# contradicted the sentence was on the same line as the sentence.
#
# Everything else here can degrade and still leave a usable artefact. This
# cannot: audio rendered on a quiet machine and labelled "under load" is
# indistinguishable to the listener from the real thing, which is the whole
# reason the header of this file exists. So it is a refusal, not a warning, and
# it tests the FACT (what the machine is doing) and not only the intent (that a
# process was launched).
if ! kill -0 "$LOADPID" 2>/dev/null; then
  echo "REFUSING: the load generator exited during warm-up. Its log says:"
  tail -5 "$HOME/loadgen.log"
  clear_box; exit 1
fi
LOADAVG=$(cut -d' ' -f1 /proc/loadavg)
CORES=$(nproc)
# Half the cores busy is a floor, not a target: a server certified at C80 on 24
# cores runs at a load average near 24, and anything under 12 means the streams
# the bundle claims to be competing with are not there.
if [ "$(echo "$LOADAVG" | cut -d. -f1)" -lt "$((CORES / 2))" ]; then
  echo "REFUSING: loadavg $LOADAVG on $CORES cores after 45 s of warm-up."
  echo "          The generator is alive but the box is idle, so these samples"
  echo "          would not be 'under load' whatever the manifest says."
  tail -5 "$HOME/loadgen.log"
  clear_box; kill "$LOADPID" 2>/dev/null; exit 1
fi
echo "load at C$C CONFIRMED: loadavg $LOADAVG on $CORES cores, generator pid $LOADPID alive"

# The bank is TAB-SEPARATED: "class<TAB>text". Splitting on it is not cosmetic --
# feeding the whole line to the server would make it speak the word "medium"
# before every sentence. Take a proportional spread of the four English classes
# rather than the first N lines, which are all short ones.
mapfile -t ROWS < <(python3 - tests/load_texts_en_v2.txt <<'PY'
import sys, collections
want = {"short": 12, "medium": 16, "conversational": 11, "long": 9}
by = collections.defaultdict(list)
for line in open(sys.argv[1], encoding="utf-8"):
    line = line.rstrip("\n")
    if not line.strip() or line.lstrip().startswith("#"):
        continue
    parts = line.split("\t", 1)
    if len(parts) != 2:
        continue
    cls, text = parts[0].strip(), parts[1].strip()
    if cls in want and len(text) > 12:
        by[cls].append(text)
for cls, n in want.items():
    rows = by.get(cls, [])
    if not rows:
        continue
    step = max(1, len(rows) // n)          # spread across the class, not its head
    for text in rows[::step][:n]:
        print(f"{cls}\t{text}")
PY
)
echo "selected ${#ROWS[@]} sentences from a bank of $(grep -v '^#' tests/load_texts_en_v2.txt | grep -cv '^[[:space:]]*$')"

# Build every JSON body NOW, in one python process, before a single sample is
# pulled. The first version spawned two python3 interpreters per sample inside
# the capture loop -- on a box already saturated at C126 that is CPU stolen from
# the thing being measured, and it showed up as stalls the server did not cause.
BODYDIR=$(mktemp -d)
ROWFILE="$BODYDIR/rows.tsv"
printf '%s\n' "${ROWS[@]}" > "$ROWFILE"
# The rows go in through a FILE, not stdin. `python3 - <<PY` already uses stdin
# for the program itself, so a second stdin redirect silently replaces the
# program with the data: the first attempt printed "prebuilt 0 request bodies"
# and every curl then failed on a body file that was never written.
python3 - "$BODYDIR" "$ROWFILE" <<'PY'
import json, os, sys
outdir, rowfile = sys.argv[1], sys.argv[2]
with open(rowfile, encoding="utf-8") as fh:
    rows = [r for r in fh.read().splitlines() if r.strip()]
for i, row in enumerate(rows):
    cls, text = row.split("\t", 1)
    with open(os.path.join(outdir, f"{i:02d}.json"), "w", encoding="utf-8") as f:
        json.dump({"model": "pocket-en", "input": text,
                   "voice": "alba", "response_format": "wav"}, f)
print(f"prebuilt {len(rows)} request bodies", file=sys.stderr)
PY
built=$(ls "$BODYDIR"/[0-9][0-9].json 2>/dev/null | wc -l)
echo "prebuilt $built request bodies"
[ "$built" -eq "${#ROWS[@]}" ] || { echo "REFUSING: $built bodies for ${#ROWS[@]} sentences"; clear_box; exit 1; }

printf 'file|class|bytes|seconds|http_s|text\n' > "$OUT/manifest.psv"
i=0
for row in "${ROWS[@]}"; do
  cls=${row%%$'\t'*}                  # the bank's own class, not a length guess
  t=${row#*$'\t'}
  n=$(printf '%02d' "$i")
  f="$OUT/streaming-under-load/${n}_${cls}.wav"
  t0=$EPOCHREALTIME                    # bash builtin: no process spawned
  # --limit-rate paces the read at 24 kHz x 16-bit = 48000 B/s, i.e. realtime,
  # which is what an actual listener does. WITHOUT IT THE CAPTURE PERTURBS THE
  # THING IT RECORDS: a reader that drains the socket flat out never applies
  # backpressure, so its worker produces as fast as it can instead of pacing,
  # and one such stream costs the other 125 real capacity. Measured -- same
  # parameters, same box, one greedy reader added: stall@250 went from
  # 4/23,212 to 46/23,078, and stall@500 from 0 to 30.
  curl -sf -m 300 --limit-rate 48k -X POST "localhost:$PORT/v1/audio/speech" \
       -H 'content-type: application/json' --data-binary "@$BODYDIR/$n.json" -o "$f"
  rc=$?; t1=$EPOCHREALTIME
  sz=$(stat -c%s "$f" 2>/dev/null || echo 0)
  # Integer arithmetic in the shell; the manifest is rewritten exactly once at
  # the end, so nothing but curl runs while the box is under load.
  secs=$(( (sz > 44 ? sz - 44 : 0) / 48000 ))
  printf '%s|%s|%s|%s|%s|%s\n' "${n}_${cls}.wav" "$cls" "$sz" "$secs" \
         "$(( ${t1%%.*} - ${t0%%.*} ))" "$t" >> "$OUT/manifest.psv"
  [ "$rc" -ne 0 ] && echo "  WARN $n rc=$rc"
  echo "  $n $cls ~${secs}s ($sz B)"
  i=$((i+1))
done

echo "capture done; letting the load generator finish so its verdict is real"
wait "$LOADPID" 2>/dev/null
curl -sf "localhost:$PORT/health" >"$OUT/health-after.json" 2>/dev/null || true
{
  echo "Load conditions during the capture above"
  echo "----------------------------------------"
  grep -E "^ *$C |PASS (mandatory|preferred)|FAIL " "$HOME/loadgen.log" | tail -14
} > "$OUT/load-conditions.txt" 2>/dev/null || true
rm -rf "$BODYDIR"
clear_box
echo "CAPTURE DONE $(date -Is)  files=$(ls "$OUT/streaming-under-load" | wc -l)  -> $OUT"
