#!/bin/sh
# FAST screen: one pass over a concurrency ladder, in the style of the reference
# engine's Tier-A wave. It is a SCREEN, not a qualification — a wave may
# disqualify a configuration but may never promote one. Only a SOAK promotes,
# and .work/streaming-cadence.md §5 says why: a configuration there passed the
# wave screen at 0.919 and failed a thirty-minute soak at 1.004 with 596
# rejects.
#
# Deciding metric is NOT aggregate RTF. STREAM_RTF is a capacity number; what
# gates is required_prebuffer p95 and stall@250, because a stream can average
# under real time and still deliver in bursts a player cannot absorb.
#
# Usage: tools/fast-suite.sh MODEL_DIR [LEVELS] [TOPOLOGY...]
#   tools/fast-suite.sh models/pocket-en "1 2 4 8 12 16 20 24 32"
set -eu

MODEL="${1:?usage: fast-suite.sh MODEL_DIR [LEVELS] [SERVER_ARGS...]}"
LEVELS="${2:-1 2 4 8 12 16 20 24 32}"
shift 2 2>/dev/null || shift 1 2>/dev/null || true
SERVER_ARGS="${*:-}"

BIN=build/cpu/mynah-tts-server
OUT="${FAST_OUT:-/tmp/fast-suite-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$OUT"

echo "FAST screen"
echo "  host      $(uname -srm)  $(getconf _NPROCESSORS_ONLN 2>/dev/null || echo '?') cpus"
echo "  build     $(git rev-parse --short HEAD 2>/dev/null || echo 'no git')"
echo "  model     $MODEL"
echo "  levels    $LEVELS"
echo "  topology  ${SERVER_ARGS:-single process}"
echo "  out       $OUT"
echo

# The machine must be quiet. A capacity number taken next to a compile is a
# number about the compile. This refuses rather than warns, because the whole
# point of the run is that the answer is trustworthy.
LOAD=$(uptime | sed 's/.*load averages*: *//' | awk '{print int($1)}')
if [ "${FAST_ALLOW_LOAD:-0}" -eq 0 ] && [ "$LOAD" -gt 2 ]; then
    echo "REFUSED: load average is $LOAD. Wait for the box to go quiet, or set" >&2
    echo "FAST_ALLOW_LOAD=1 and know that the numbers describe a loaded machine." >&2
    exit 3
fi

for c in $LEVELS; do
    echo "=== C$c ==="
    python3 tools/serving_profile.py \
        --server-bin "$BIN" --model "$MODEL" \
        ${SERVER_ARGS:+--server-args "$SERVER_ARGS"} \
        --mode wave --levels "$c" --waves 3 \
        --server-log "$OUT/server-c$c.log" \
        2>&1 | tee "$OUT/c$c.txt" | grep -E "GOOD|MARGINAL|NOT STREAMABLE|STREAM_RTF|prebuffer|stall|TTFA|REFUS" || true
    echo
done

echo "screen written to $OUT"
echo "Capacity is the highest GOOD level with margin. It is discovered, not"
echo "prescribed: if C12 is GOOD and C16 is MARGINAL, the operating point is C12."
echo "Nothing here promotes anything — run a SOAK at the winner before that."
