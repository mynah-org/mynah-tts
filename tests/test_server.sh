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
# The warm-up check (E5-20) needs servers that have never served anything, so
# it starts its own on scratch ports. Tracked here rather than killed only on
# the happy path: a `fail` between start and stop would otherwise leave a
# process holding the whole pack.
AUX_PID=""

# A server holds the whole model resident (~2 GB once it has synthesized
# anything), so a leaked one is expensive on a small-memory machine. SIGTERM
# first, then make sure: a server killed mid-synthesis can take a moment, and
# an escaped one would still be holding those 2 GB an hour later.
cleanup() {
    if [ -n "$AUX_PID" ]; then
        kill "$AUX_PID" 2>/dev/null || true
        kill -9 "$AUX_PID" 2>/dev/null || true
        wait "$AUX_PID" 2>/dev/null || true
        AUX_PID=""
    fi
    if [ -n "$PID" ]; then
        kill "$PID" 2>/dev/null || true
        i=0
        while kill -0 "$PID" 2>/dev/null && [ "$i" -lt 20 ]; do
            i=$((i + 1))
            sleep 0.25
        done
        kill -9 "$PID" 2>/dev/null || true
        wait "$PID" 2>/dev/null || true
    fi
    rm -rf "$TMP"
}
trap cleanup EXIT INT TERM

fail() { echo "FAIL: $*" >&2; exit 1; }

# Refuse to pile a second 2 GB server on top of one left behind by an earlier
# run: two of these is most of a small machine's memory.
if curl -sf --max-time 2 "$BASE/health" > /dev/null 2>&1; then
    fail "something is already serving on port $PORT; stop it first (each server holds ~2 GB)"
fi

# SERVER_ARGS lets the same 15 checks run against a different serving topology
# without a second copy of the script. The prefork gate is exactly this file
# with SERVER_ARGS="--prefork 2": every route, the parity check and the
# concurrency checks must behave identically whether one process serves them or
# a router hands each connection to a worker. A separate script would drift.
# shellcheck disable=SC2086
"$SERVER" -m "$MODEL_DIR" -p "$PORT" ${SERVER_ARGS:-} > "$TMP/server.log" 2>&1 &
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
# alone. Batching reorders independent work, so a difference here means a slot
# leaked state into its neighbour.
#
# E5-10: this check used to ALSO assert that four concurrent requests finish no
# slower than four serial ones, timed with `date +%s` -- one-second granularity
# around a workload of roughly 200 ms. Two samples from the same distribution
# land on either side of a second boundary often enough that the check failed
# on a correct server, which is worse than no check: a gate that cries wolf
# gets ignored, and then the identity comparison next to it gets ignored too.
# It is gone rather than made finer-grained. Throughput is not this file's
# question -- `make serving-wave` and `make serving-soak` (tools/serving_profile.py)
# measure it properly, with warm-up discarded and a drift gate, and
# .work/serving-doctrine.md is explicit that a screen may disqualify a
# configuration but never promote one. What stays here is the part that is
# decidable from a single run: the bytes.
#
# Every wait names its PIDs: the server itself is a background child of this
# script, so a bare `wait` would block on it forever.
for i in 1 2 3 4; do
    curl -s --max-time 600 -X POST "$BASE/v1/audio/speech" \
        -H 'Content-Type: application/json' \
        -d "{\"input\":\"batching under load check\",\"voice\":\"Sofia\",\"seed\":$i}" \
        -o "$TMP/serial$i.wav" || fail "serial client $i failed"
done

pids=""
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

# The four requests differ only in seed, so if the seed were ignored all four
# would be identical and every comparison below would be trivially true. Assert
# they are distinct BEFORE asserting they match their serial twins: this check
# is what keeps the one after it from being decoration.
cmp -s "$TMP/serial1.wav" "$TMP/serial2.wav" &&
    fail "seeds 1 and 2 produced identical audio; the identity check below would be vacuous"
for i in 1 2 3 4; do
    [ -s "$TMP/p$i.wav" ] || fail "batched client $i got no audio"
    cmp -s "$TMP/p$i.wav" "$TMP/serial$i.wav" ||
        fail "batched client $i differs from the same request run alone"
done
echo "batching    ok (4 concurrent == 4 serial, byte-identical and pairwise distinct)"

# --- Continuous admission --------------------------------------------------
# A request that arrives while another is mid-synthesis must be parked by the
# scheduler, admitted into the next batch, and still return exactly the audio
# it would get alone. The batching test above cannot see this path: there every
# client arrives on an idle server, so nothing is running when they queue.
LONG='{"input":"this deliberately longer sentence keeps the decoder busy for a while so that a later request truly arrives in the middle of a running synthesis","voice":"Sofia","seed":21}'
SHORT='{"input":"late arrival","voice":"Leo","seed":22}'
curl -s --max-time 600 -X POST "$BASE/v1/audio/speech" \
    -H 'Content-Type: application/json' -d "$LONG" -o "$TMP/adm_long_solo.wav"
curl -s --max-time 600 -X POST "$BASE/v1/audio/speech" \
    -H 'Content-Type: application/json' -d "$SHORT" -o "$TMP/adm_short_solo.wav"
curl -s --max-time 600 -X POST "$BASE/v1/audio/speech" \
    -H 'Content-Type: application/json' -d "$LONG" -o "$TMP/adm_long.wav" &
pa=$!
sleep 1
curl -s --max-time 600 -X POST "$BASE/v1/audio/speech" \
    -H 'Content-Type: application/json' -d "$SHORT" -o "$TMP/adm_short.wav" &
pb=$!
wait $pa || fail "the running request failed when a late one arrived"
wait $pb || fail "the late-arriving request failed"
cmp -s "$TMP/adm_long.wav" "$TMP/adm_long_solo.wav" ||
    fail "mid-flight arrival perturbed the running request"
cmp -s "$TMP/adm_short.wav" "$TMP/adm_short_solo.wav" ||
    fail "late-arriving request differs from the same request run alone"
echo "admission   ok (request arriving mid-synthesis is served and byte-identical)"

# --- Mixed streaming + batch ----------------------------------------------
# Streaming runs alone behind the synthesis lock by design; this asserts that
# streams and batches interleaved under load neither deadlock nor perturb each
# other: every client must get exactly the audio of the same request run alone.
SREQ='{"input":"streaming under load","voice":"Leo","seed":31,"stream":true}'
BREQ='{"input":"mixed load check","voice":"Sofia","seed":32}'
curl -s --max-time 600 -X POST "$BASE/v1/audio/speech" \
    -H 'Content-Type: application/json' -d "$BREQ" -o "$TMP/mix_b_solo.wav"
curl -s --max-time 600 -X POST "$BASE/v1/audio/speech" \
    -H 'Content-Type: application/json' -d "$SREQ" -o "$TMP/mix_s_solo.pcm"
pids=""
for i in 1 2; do
    curl -s --max-time 600 -X POST "$BASE/v1/audio/speech" \
        -H 'Content-Type: application/json' -d "$BREQ" -o "$TMP/mix_b$i.wav" &
    pids="$pids $!"
    curl -s --max-time 600 -X POST "$BASE/v1/audio/speech" \
        -H 'Content-Type: application/json' -d "$SREQ" -o "$TMP/mix_s$i.pcm" &
    pids="$pids $!"
done
for p in $pids; do
    wait "$p" || fail "a mixed stream+batch client failed"
done
for i in 1 2; do
    cmp -s "$TMP/mix_b$i.wav" "$TMP/mix_b_solo.wav" ||
        fail "batch client $i under mixed load differs from solo"
    cmp -s "$TMP/mix_s$i.pcm" "$TMP/mix_s_solo.pcm" ||
        fail "stream client $i under mixed load differs from solo"
done
echo "mixed       ok (2 streams + 2 batches concurrent, all byte-identical to solo)"

# --- JSON body handling ----------------------------------------------------
# server/main.c parses the body ONCE with src/json.c and hands the parsed root
# to the route, instead of re-parsing it per key through the compatibility
# wrappers. Three behaviours came with that and all three are asserted here,
# because each is the kind of thing a future "optimisation" would quietly undo:
#
#   - a refusal names the byte offset and the expectation. "invalid JSON" with
#     no locus is not actionable for whoever is holding the request, and the
#     old brace-sniff could not produce one at all -- a body that was not JSON
#     but started with `{` got through and failed later somewhere that could
#     only say "missing 'input'".
#   - a nested key does not satisfy a top-level lookup.
#   - surrogate pairs decode, so anything outside the BMP reaches the
#     tokenizer instead of being rejected.
J="$BASE/v1/audio/speech"

curl -s -X POST "$J" -H 'Content-Type: application/json' \
    -d '{"input":"hi", "voice":' > "$TMP/j_trunc.json"
grep -q 'not valid JSON at byte' "$TMP/j_trunc.json" ||
    fail "a truncated body did not report the offset: $(cat "$TMP/j_trunc.json")"

curl -s -X POST "$J" -H 'Content-Type: application/json' -d '[1,2,3]' \
    > "$TMP/j_arr.json"
grep -q 'not an object' "$TMP/j_arr.json" ||
    fail "a JSON array body was not refused as a non-object"

curl -s -X POST "$J" -H 'Content-Type: application/json' \
    -d '{"outer":{"input":"nested"}}' > "$TMP/j_nest.json"
grep -q "missing 'input'" "$TMP/j_nest.json" ||
    fail "a nested \"input\" satisfied the top-level lookup"

code=$(curl -s -o "$TMP/j_emoji.wav" -w '%{http_code}' -X POST "$J" \
    -H 'Content-Type: application/json' \
    -d '{"input":"hello \ud83d\ude00 world","voice":"Sofia","seed":7}')
[ "$code" = "200" ] || fail "a surrogate pair was refused ($code)"
head -c 4 "$TMP/j_emoji.wav" | grep -q RIFF || fail "the emoji request returned no audio"
echo "json        ok (offset in the error, non-object refused, nested key is not"
echo "            top-level, surrogate pair reaches the tokenizer)"

# --- Warm-up: no trajectory fork at request one (E5-20) --------------------
#
# The bug this guards against is the reference implementation's, and it is the
# hardest kind to find later: its warm-up ran on whatever state the CLI had
# left behind, primed per-request state for a configuration no request uses,
# and THE FIRST REAL REQUEST CAME OUT DIFFERENT FROM EVERY ONE AFTER IT
# (.work/serving-design.md §7). It reproduces only on a fresh process, so no
# load test ever sees it, and every A/B measured afterwards quietly contains it.
#
# The assertion is deliberately three-cornered, because two of the corners
# would each pass for the wrong reason on their own:
#
#   warm1  first request to a server that warmed up
#   cold1  first request to a server that did NOT warm up
#   cold2  second request to that same server
#
# warm1 == cold1  says the warm-up changed no audio at all.
# cold1 == cold2  says the first request equals the second without a warm-up --
#                 i.e. the property holds on its own merits and the warm-up is
#                 an optimisation, not a patch over a divergence.
# Both together are what "the warm-up goes through the request path's own
# reset" means operationally. A warm-up that left per-request state behind
# breaks the first; a first request that differed for any other reason breaks
# the second.
#
# Every check above this one runs against a server that has already served
# dozens of requests, so none of them can see request one.
AUX_PORT=$((PORT + 11))

start_aux() {   # $1 = port, $2... = extra server args
    _p=$1
    shift
    # shellcheck disable=SC2086
    "$SERVER" -m "$MODEL_DIR" -p "$_p" ${SERVER_ARGS:-} "$@" \
        > "$TMP/aux.log" 2>&1 &
    AUX_PID=$!
    _i=0
    while [ "$_i" -lt 90 ]; do
        if curl -sf --max-time 2 "http://127.0.0.1:$_p/health" > /dev/null 2>&1; then
            return 0
        fi
        kill -0 "$AUX_PID" 2>/dev/null || { cat "$TMP/aux.log" >&2; fail "aux server exited"; }
        _i=$((_i + 1))
        sleep 1
    done
    cat "$TMP/aux.log" >&2
    fail "aux server on port $_p did not become ready"
}

stop_aux() {
    [ -n "$AUX_PID" ] || return 0
    kill "$AUX_PID" 2>/dev/null || true
    _i=0
    while kill -0 "$AUX_PID" 2>/dev/null && [ "$_i" -lt 40 ]; do
        _i=$((_i + 1))
        sleep 0.25
    done
    kill -9 "$AUX_PID" 2>/dev/null || true
    wait "$AUX_PID" 2>/dev/null || true
    AUX_PID=""
}

if curl -sf --max-time 2 "http://127.0.0.1:$AUX_PORT/health" > /dev/null 2>&1; then
    fail "something is already serving on port $AUX_PORT"
fi

WREQ='{"input":"warm up parity","voice":"Sofia","seed":11}'
WREQ_OTHER='{"input":"warm up parity control","voice":"Sofia","seed":11}'

start_aux "$AUX_PORT" --warmup 1
curl -s "http://127.0.0.1:$AUX_PORT/health" | grep -q '"warmups":{"requested":1,"done":1}' ||
    fail "a --warmup 1 server does not report a completed warm-up in /health"
curl -s --max-time 600 -X POST "http://127.0.0.1:$AUX_PORT/v1/audio/speech" \
    -H 'Content-Type: application/json' -d "$WREQ" -o "$TMP/warm1.wav"
stop_aux
[ -s "$TMP/warm1.wav" ] || fail "the warmed server returned nothing"

start_aux "$AUX_PORT" --warmup 0
curl -s "http://127.0.0.1:$AUX_PORT/health" | grep -q '"warmups":{"requested":0,"done":0}' ||
    fail "--warmup 0 still reports a warm-up"
curl -s --max-time 600 -X POST "http://127.0.0.1:$AUX_PORT/v1/audio/speech" \
    -H 'Content-Type: application/json' -d "$WREQ" -o "$TMP/cold1.wav"
curl -s --max-time 600 -X POST "http://127.0.0.1:$AUX_PORT/v1/audio/speech" \
    -H 'Content-Type: application/json' -d "$WREQ" -o "$TMP/cold2.wav"
# The anti-vacuity control, on the same process: if this server returned the
# same bytes for every input, all three comparisons above would pass while
# proving nothing.
curl -s --max-time 600 -X POST "http://127.0.0.1:$AUX_PORT/v1/audio/speech" \
    -H 'Content-Type: application/json' -d "$WREQ_OTHER" -o "$TMP/cold_other.wav"
stop_aux

[ -s "$TMP/cold1.wav" ] && [ -s "$TMP/cold2.wav" ] || fail "the unwarmed server returned nothing"
cmp -s "$TMP/cold1.wav" "$TMP/cold_other.wav" &&
    fail "a different input produced identical audio; the warm-up check would be vacuous"
cmp -s "$TMP/cold1.wav" "$TMP/cold2.wav" ||
    fail "request 1 differs from request 2 on an unwarmed server (trajectory fork)"
cmp -s "$TMP/warm1.wav" "$TMP/cold1.wav" ||
    fail "the first request to a warmed server differs from the first to an unwarmed one"
echo "warmup      ok (warm-first == cold-first == cold-second, and not vacuous)"

echo "server test: PASS"
