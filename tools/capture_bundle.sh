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
#   bash tools/capture_bundle.sh [CONCURRENCY] [PORT]
#
# Run it from the repo root on the serving host. Produces ~/bundle<C>/ with the
# WAVs, a manifest naming every sentence, and the load conditions they were
# captured under.
set -u
C="${1:-126}"
PORT="${2:-9200}"
SECONDS_OF_LOAD="${3:-480}"

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
./build/cpu/mynah-tts-server -m models/pocket-en -p "$PORT" \
    --prefork 16 --prefork-threads 2 --max-batch 8 >"$HOME/srv-bundle.log" 2>&1 &
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
    --soak-seconds "$SECONDS_OF_LOAD" --warmup-seconds 20 --window-seconds 180 \
    >"$HOME/loadgen.log" 2>&1 &
LOADPID=$!
sleep 45
echo "load at C$C, loadavg $(cut -d' ' -f1 /proc/loadavg)"

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
python3 - "$BODYDIR" <<'PY'
import json, sys, os
d = sys.argv[1]
for i, row in enumerate(sys.stdin.read().splitlines()):
    if not row.strip():
        continue
    cls, text = row.split("\t", 1)
    with open(os.path.join(d, f"{i:02d}.json"), "w", encoding="utf-8") as f:
        json.dump({"model": "pocket-en", "input": text,
                   "voice": "alba", "response_format": "wav"}, f)
PY
<<< "$(printf '%s\n' "${ROWS[@]}")"
echo "prebuilt $(ls "$BODYDIR" | wc -l) request bodies"

printf 'file|class|bytes|seconds|http_s|text\n' > "$OUT/manifest.psv"
i=0
for row in "${ROWS[@]}"; do
  cls=${row%%$'\t'*}                  # the bank's own class, not a length guess
  t=${row#*$'\t'}
  n=$(printf '%02d' "$i")
  f="$OUT/streaming-under-load/${n}_${cls}.wav"
  t0=$EPOCHREALTIME                    # bash builtin: no process spawned
  curl -sf -m 180 -X POST "localhost:$PORT/v1/audio/speech" \
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
