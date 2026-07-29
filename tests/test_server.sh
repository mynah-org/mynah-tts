#!/bin/sh
# End-to-end check of the mynah-tts HTTP server.
#
#   make server-test MODEL_DIR=models/magpie-v2607-pack
#
# Starts a server on a scratch port, exercises every route, and verifies that a
# streamed response carries the same samples as the batch one for the same
# request -- the claim docs/server.md makes.
set -eu

SERVER="${SERVER:-build/cpu/mynah-tts-server}"
MODEL_DIR="${MODEL_DIR:?set MODEL_DIR to a model pack}"
PORT="${PORT:-8973}"
BASE="http://127.0.0.1:$PORT"
TMP="$(mktemp -d)"
PID=""

cleanup() {
    [ -n "$PID" ] && kill "$PID" 2>/dev/null || true
    rm -rf "$TMP"
}
trap cleanup EXIT INT TERM

fail() { echo "FAIL: $*" >&2; exit 1; }

"$SERVER" -m "$MODEL_DIR" -p "$PORT" > "$TMP/server.log" 2>&1 &
PID=$!

# Wait for readiness rather than sleeping a guessed amount.
i=0
while [ "$i" -lt 60 ]; do
    if curl -sf --max-time 2 "$BASE/health" > /dev/null 2>&1; then break; fi
    kill -0 "$PID" 2>/dev/null || { cat "$TMP/server.log" >&2; fail "server exited"; }
    i=$((i + 1))
    sleep 1
done
[ "$i" -lt 60 ] || fail "server did not become ready"

echo "health      $(curl -s "$BASE/health")"
curl -s "$BASE/health" | grep -q '"status":"ok"' || fail "/health"

curl -s "$BASE/v1/voices" | grep -q '"voices"' || fail "/v1/voices"
echo "voices      ok"

curl -s "$BASE/v1/models" | grep -q '"object":"list"' || fail "/v1/models"
echo "models      ok"

code=$(curl -s -o /dev/null -w '%{http_code}' "$BASE/nope")
[ "$code" = "404" ] || fail "unknown route returned $code, expected 404"
echo "404         ok"

code=$(curl -s -o "$TMP/err.json" -w '%{http_code}' -X POST "$BASE/v1/audio/speech" \
    -H 'Content-Type: application/json' -d '{"voice":"Sofia"}')
[ "$code" = "400" ] || fail "missing input returned $code, expected 400"
grep -q 'invalid_request_error' "$TMP/err.json" || fail "error body not OpenAI-shaped"
echo "missing in  ok (400)"

code=$(curl -s -o /dev/null -w '%{http_code}' -X POST "$BASE/v1/audio/speech" \
    -H 'Content-Type: application/json' -d '{"input":"x","response_format":"mp3"}')
[ "$code" = "400" ] || fail "mp3 returned $code, expected 400"
echo "mp3 reject  ok (400)"

REQ='{"input":"server parity check","voice":"Sofia","seed":7}'

curl -s --max-time 600 -X POST "$BASE/v1/audio/speech" \
    -H 'Content-Type: application/json' -d "$REQ" -o "$TMP/batch.wav"
[ -s "$TMP/batch.wav" ] || fail "empty WAV"
head -c 4 "$TMP/batch.wav" | grep -q RIFF || fail "not a RIFF file"
echo "speech wav  ok ($(wc -c < "$TMP/batch.wav" | tr -d ' ') bytes)"

curl -s --max-time 600 -X POST "$BASE/v1/audio/speech" \
    -H 'Content-Type: application/json' \
    -d "$(printf '%s' "$REQ" | sed 's/}$/,"stream":true}/')" -o "$TMP/stream.pcm"
[ -s "$TMP/stream.pcm" ] || fail "empty stream"
echo "speech pcm  ok ($(wc -c < "$TMP/stream.pcm" | tr -d ' ') bytes)"

# The WAV header is 44 bytes; the rest must match the streamed PCM exactly.
tail -c +45 "$TMP/batch.wav" > "$TMP/batch.pcm"
if cmp -s "$TMP/batch.pcm" "$TMP/stream.pcm"; then
    echo "stream==batch ok (byte-identical)"
else
    a=$(wc -c < "$TMP/batch.pcm" | tr -d ' ')
    b=$(wc -c < "$TMP/stream.pcm" | tr -d ' ')
    fail "streamed audio differs from batch (batch $a bytes, stream $b bytes)"
fi

# --- Reproducibility -------------------------------------------------------
# Three identical requests must return identical audio. This is the highest
# value check for this codebase: mynah_tts_model carries mutable caches (qmat,
# codec filters, local projection cache) that synthesis populates, so a stale
# entry surviving between requests would show up exactly here. qwen-tts shipped
# this test after hitting that bug for real.
for n in 1 2 3; do
    curl -s --max-time 600 -X POST "$BASE/v1/audio/speech" \
        -H 'Content-Type: application/json' -d "$REQ" -o "$TMP/repro$n.wav"
    [ -s "$TMP/repro$n.wav" ] || fail "repro request $n returned nothing"
done
cmp -s "$TMP/repro1.wav" "$TMP/repro2.wav" || fail "request 2 differs from request 1"
cmp -s "$TMP/repro1.wav" "$TMP/repro3.wav" || fail "request 3 differs from request 1"
cmp -s "$TMP/repro1.wav" "$TMP/batch.wav"  || fail "repro differs from the first batch call"
echo "repro x3    ok (byte-identical, and equal to the earlier call)"

# --- Concurrency -----------------------------------------------------------
# Synthesis is serialized behind a mutex; this asserts the lock actually holds
# instead of assuming it. Both clients must get complete, correct audio.
curl -s --max-time 600 -X POST "$BASE/v1/audio/speech" \
    -H 'Content-Type: application/json' -d "$REQ" -o "$TMP/c1.wav" &
p1=$!
curl -s --max-time 600 -X POST "$BASE/v1/audio/speech" \
    -H 'Content-Type: application/json' \
    -d '{"input":"a different concurrent request","voice":"Leo","seed":9}' -o "$TMP/c2.wav" &
p2=$!
wait $p1 || fail "concurrent client 1 failed"
wait $p2 || fail "concurrent client 2 failed"
[ -s "$TMP/c1.wav" ] && [ -s "$TMP/c2.wav" ] || fail "a concurrent client got no audio"
head -c 4 "$TMP/c1.wav" | grep -q RIFF || fail "concurrent client 1 got a corrupt WAV"
head -c 4 "$TMP/c2.wav" | grep -q RIFF || fail "concurrent client 2 got a corrupt WAV"
cmp -s "$TMP/c1.wav" "$TMP/repro1.wav" || fail "concurrency perturbed the identical request"
echo "concurrent  ok (both complete; the repeated request is still identical)"

# --- Native route ----------------------------------------------------------
curl -s --max-time 600 -X POST "$BASE/v1/tts" \
    -H 'Content-Type: application/json' \
    -d '{"text":"server parity check","speaker":"Sofia","seed":7}' -o "$TMP/native.wav"
cmp -s "$TMP/native.wav" "$TMP/repro1.wav" || fail "/v1/tts differs from /v1/audio/speech"
echo "native /tts ok (same audio as the OpenAI route)"

# --- Batching under load ---------------------------------------------------
# Four clients at once must each get exactly the audio they would have got
# alone, and the batch must not be slower than the same four one after another.
# The identity check is the one that matters: batching reorders independent
# work, so a difference here means a slot leaked state into its neighbour.
#
# Every wait names its PIDs: the server itself is a background child of this
# script, so a bare `wait` would block on it forever.
serial_start=$(date +%s)
for i in 1 2 3 4; do
    curl -s --max-time 600 -X POST "$BASE/v1/audio/speech" \
        -H 'Content-Type: application/json' \
        -d "{\"input\":\"batching under load check\",\"voice\":\"Sofia\",\"seed\":$i}" \
        -o "$TMP/serial$i.wav" || fail "serial client $i failed"
done
serial_end=$(date +%s)

pids=""
batch_start=$(date +%s)
for i in 1 2 3 4; do
    curl -s --max-time 600 -X POST "$BASE/v1/audio/speech" \
        -H 'Content-Type: application/json' \
        -d "{\"input\":\"batching under load check\",\"voice\":\"Sofia\",\"seed\":$i}" \
        -o "$TMP/p$i.wav" &
    pids="$pids $!"
done
for p in $pids; do
    wait "$p" || fail "concurrent batching client failed"
done
batch_end=$(date +%s)

for i in 1 2 3 4; do
    [ -s "$TMP/p$i.wav" ] || fail "batched client $i got no audio"
    cmp -s "$TMP/p$i.wav" "$TMP/serial$i.wav" ||
        fail "batched client $i differs from the same request run alone"
done
serial_s=$((serial_end - serial_start))
batch_s=$((batch_end - batch_start))
[ "$batch_s" -le "$serial_s" ] ||
    fail "four concurrent requests ($batch_s s) were slower than four serial ($serial_s s)"
echo "batching    ok (4 concurrent ${batch_s}s vs 4 serial ${serial_s}s, all byte-identical)"

echo "server test: PASS"
