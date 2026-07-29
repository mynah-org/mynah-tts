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

echo "server test: PASS"
