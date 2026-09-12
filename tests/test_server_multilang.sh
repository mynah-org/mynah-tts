#!/bin/sh
# Multi-language serving (E5-9): one worker process per language.
#
#   make server-multilang-test MODEL_DIR=models/pocket-en MODEL_DIR_B=models/pocket-it
#
# A separate script from tests/test_server.sh on purpose. That one asserts the
# behaviour of ONE pack and must keep asserting exactly that: it is the gate
# that says the single-language server did not change. This one asserts what
# happens when there are two, which is a different server shape.
#
# What is actually being checked, in the order the failures matter:
#
#   1. both languages are served, and each from its OWN weights -- the same
#      text in two languages must not come back as the same bytes, or the
#      routing did nothing and one model answered twice;
#   2. a language nobody holds is REFUSED, with the ladder's own machine
#      readable code, in both topologies -- and not mis-served, which is what
#      happened before E5-9;
#   3. THE PROPERTY THAT MATTERS MOST: a request's bytes do not depend on what
#      else was in flight. Taken alone on a single-pack server, alone on a
#      two-language server, and concurrently with a pile of the other
#      language's requests, the same request must produce the same bytes.
#
# If a second real pack is not available, the script FABRICATES one by
# symlinking the first pack's weights and rewriting model.json's language. Then
# check 1's "the bytes differ" half is skipped and says so, because a synthetic
# pack shares the weights it was copied from -- the routing is still exercised,
# the divergence is not.
set -eu

SERVER="${SERVER:-build/cpu/mynah-tts-server}"
MODEL_DIR="${MODEL_DIR:?set MODEL_DIR to a language-bound model pack}"
MODEL_DIR_B="${MODEL_DIR_B:-}"
PORT="${PORT:-8975}"
PORT_SOLO="${PORT_SOLO:-8976}"
BASE="http://127.0.0.1:$PORT"
SOLO="http://127.0.0.1:$PORT_SOLO"
TMP="$(mktemp -d)"
PID=""
PID_SOLO=""
SYNTHETIC=0
PASS=0

# Every server here holds a whole model resident. A leaked one is expensive, and
# two of them is most of a small machine, so the cleanup is unconditional and
# escalates rather than hoping SIGTERM lands.
stop_one() {
    [ -n "$1" ] || return 0
    kill "$1" 2>/dev/null || true
    i=0
    while kill -0 "$1" 2>/dev/null && [ "$i" -lt 20 ]; do
        i=$((i + 1))
        sleep 0.25
    done
    kill -9 "$1" 2>/dev/null || true
    wait "$1" 2>/dev/null || true
}
cleanup() {
    stop_one "$PID"
    stop_one "$PID_SOLO"
    rm -rf "$TMP"
}
trap cleanup EXIT INT TERM

fail() { echo "FAIL: $*" >&2; exit 1; }
ok()   { PASS=$((PASS + 1)); echo "ok          $*"; }

json_field() {
    # $1 = file, $2 = python expression over the parsed object `d`
    python3 -c "import json,sys; d=json.load(open('$1')); print($2)"
}

pack_language() {
    python3 -c "import json;print(json.load(open('$1/model.json')).get('language',''))"
}

LANG_A="$(pack_language "$MODEL_DIR")"
[ -n "$LANG_A" ] || fail "$MODEL_DIR declares no \"language\" in model.json; this test needs a language-bound pack (a Magpie pack is not one)"

if [ -n "$MODEL_DIR_B" ]; then
    LANG_B="$(pack_language "$MODEL_DIR_B")"
    [ -n "$LANG_B" ] || fail "$MODEL_DIR_B declares no \"language\""
    [ "$LANG_A" != "$LANG_B" ] || fail "both packs claim '$LANG_A'"
else
    # No second pack on this machine. Say so loudly and fabricate one rather
    # than skipping the whole file: routing, refusal and the byte-stability
    # property are all still testable, and a silently skipped case is worse
    # than a clearly reduced one.
    SYNTHETIC=1
    LANG_B="synthetic_second_language"
    MODEL_DIR_B="$TMP/pack-b"
    mkdir -p "$MODEL_DIR_B"
    for f in "$MODEL_DIR"/*; do
        base="$(basename "$f")"
        [ "$base" = "model.json" ] && continue
        ln -s "$(cd "$(dirname "$f")" && pwd)/$base" "$MODEL_DIR_B/$base"
    done
    python3 - "$MODEL_DIR/model.json" "$MODEL_DIR_B/model.json" "$LANG_B" <<'PY'
import json, sys
src, dst, lang = sys.argv[1], sys.argv[2], sys.argv[3]
m = json.load(open(src))
m["language"] = lang
json.dump(m, open(dst, "w"), indent=2, sort_keys=True)
PY
    echo "NOTE        only one language pack was given, so the second is SYNTHETIC:"
    echo "NOTE        $MODEL_DIR_B shares $MODEL_DIR's weights and only renames the"
    echo "NOTE        language. Routing and refusal are exercised; 'the two languages"
    echo "NOTE        produce different audio' is NOT, and is skipped below."
fi

echo "packs       $LANG_A=$MODEL_DIR  $LANG_B=$MODEL_DIR_B"

for p in "$PORT" "$PORT_SOLO"; do
    if curl -sf --max-time 2 "http://127.0.0.1:$p/health" > /dev/null 2>&1; then
        fail "something is already serving on port $p; stop it first"
    fi
done

wait_ready() {
    i=0
    while [ "$i" -lt 90 ]; do
        if curl -sf --max-time 2 "$1/health" > /dev/null 2>&1; then return 0; fi
        kill -0 "$2" 2>/dev/null || { cat "$3" >&2; fail "server exited"; }
        i=$((i + 1))
        sleep 1
    done
    cat "$3" >&2
    fail "server did not become ready"
}

# The two-language fleet. No --prefork: the server must enable it itself and say
# so, because "several packs implies one worker per language" is a decision this
# binary makes and a decision it makes has to be printed.
"$SERVER" -m "$MODEL_DIR" -m "$MODEL_DIR_B" -p "$PORT" --max-batch 4 \
    > "$TMP/fleet.log" 2>&1 &
PID=$!
wait_ready "$BASE" "$PID" "$TMP/fleet.log"

grep -q "enabling --prefork 2" "$TMP/fleet.log" \
    || fail "two packs did not enable prefork, or did not say so"
ok "two packs enabled prefork and printed it"

grep -q "one pack per worker" "$TMP/fleet.log" || fail "no language table in the banner"
grep -q "$LANG_A" "$TMP/fleet.log" || fail "banner does not name $LANG_A"
grep -q "$LANG_B" "$TMP/fleet.log" || fail "banner does not name $LANG_B"
ok "the banner prints the capacity split per language"

curl -s "$BASE/health" -o "$TMP/health.json"
plan="$(json_field "$TMP/health.json" "d['languages']['fleet']")"
bound="$(json_field "$TMP/health.json" "d['languages']['bound']")"
[ "$bound" = "True" ] || fail "/health says bound=$bound for a language-bound pack"
case "$plan" in
    *"$LANG_A="*) : ;;
    *) fail "/health fleet plan '$plan' does not name $LANG_A" ;;
esac
case "$plan" in
    *"$LANG_B="*) : ;;
    *) fail "/health fleet plan '$plan' does not name $LANG_B" ;;
esac
ok "/health reports the resident languages and the capacity split ($plan)"

# ---------------------------------------------------------------- serving

say() {
    # $1 = base url, $2 = language ("" for none), $3 = output file, $4 = steps
    if [ -z "$2" ]; then
        body="{\"input\":\"Questo e un test di instradamento multilingua.\",\"seed\":42,\"max_steps\":$4}"
    else
        body="{\"input\":\"Questo e un test di instradamento multilingua.\",\"language\":\"$2\",\"seed\":42,\"max_steps\":$4}"
    fi
    curl -s -o "$3" -w '%{http_code}' --max-time 120 -X POST "$1/v1/audio/speech" \
        -H 'Content-Type: application/json' -d "$body"
}

code="$(say "$BASE" "$LANG_A" "$TMP/a.wav" 24)"
[ "$code" = "200" ] || fail "$LANG_A returned $code"
[ -s "$TMP/a.wav" ] || fail "$LANG_A produced no audio"
code="$(say "$BASE" "$LANG_B" "$TMP/b.wav" 24)"
[ "$code" = "200" ] || fail "$LANG_B returned $code"
[ -s "$TMP/b.wav" ] || fail "$LANG_B produced no audio"
ok "both languages served from one endpoint"

if [ "$SYNTHETIC" = "1" ]; then
    echo "skip        'the two languages differ' — the second pack is synthetic and"
    echo "skip        shares the first's weights, so identical bytes are CORRECT here"
else
    if cmp -s "$TMP/a.wav" "$TMP/b.wav"; then
        fail "$LANG_A and $LANG_B returned identical bytes for the same text: one model answered both, so the routing did nothing"
    fi
    ok "the two languages came from different weights (bytes differ)"
fi

# The prefix rule: `it` must reach `italian`. Same request otherwise, so the
# bytes must match exactly -- a prefix that reached a different pack, or a
# different code path, would show up here rather than as a vague 400 later.
short_a="$(printf '%s' "$LANG_A" | cut -c1-2)"
code="$(say "$BASE" "$short_a" "$TMP/a-short.wav" 24)"
[ "$code" = "200" ] || fail "prefix '$short_a' returned $code"
cmp -s "$TMP/a.wav" "$TMP/a-short.wav" \
    || fail "prefix '$short_a' reached different weights than '$LANG_A'"
ok "an unambiguous prefix ('$short_a') resolves to '$LANG_A'"

# A request naming no language lands on the FIRST pack, which is the documented
# default and the reason pack order is a decision rather than an accident.
code="$(say "$BASE" "" "$TMP/a-default.wav" 24)"
[ "$code" = "200" ] || fail "a request with no language returned $code"
cmp -s "$TMP/a.wav" "$TMP/a-default.wav" \
    || fail "a request with no language did not land on the first pack"
ok "a request naming no language goes to the default pack ($LANG_A)"

# ---------------------------------------------------------------- refusal

refuse_check() {
    # $1 = base url, $2 = label
    code="$(curl -s -o "$TMP/refused.json" -w '%{http_code}' --max-time 30 \
        -X POST "$1/v1/audio/speech" -H 'Content-Type: application/json' \
        -d '{"input":"hallo","language":"klingon","max_steps":8}')"
    [ "$code" = "400" ] || fail "$2: an absent language returned $code, expected 400"
    got="$(json_field "$TMP/refused.json" "d['error']['code']")"
    [ "$got" = "language_not_served" ] \
        || fail "$2: error.code was '$got', expected language_not_served"
    type="$(json_field "$TMP/refused.json" "d['error']['type']")"
    [ "$type" = "invalid_request_error" ] \
        || fail "$2: error.type was '$type', expected invalid_request_error"
}

refuse_check "$BASE" "fleet"
ok "the fleet refuses an absent language 400 language_not_served"

# And it must still be SERVING afterwards: a refusal that wedges the router, or
# that leaks the slot it refused, is worse than a mis-service.
code="$(say "$BASE" "$LANG_B" "$TMP/b2.wav" 24)"
[ "$code" = "200" ] || fail "the fleet stopped serving after a refusal ($code)"
cmp -s "$TMP/b.wav" "$TMP/b2.wav" || fail "$LANG_B bytes changed after a refusal"
ok "the fleet still serves, byte-identically, after a refusal"

# ------------------------------------------------- the single-process server
#
# The same refusal from the topology that has no router at all. This is the case
# that was silently MIS-SERVED before E5-9: a bound pack ignored `language`
# entirely and answered 200 with audio from the wrong model.
"$SERVER" -m "$MODEL_DIR" -p "$PORT_SOLO" --max-batch 4 > "$TMP/solo.log" 2>&1 &
PID_SOLO=$!
wait_ready "$SOLO" "$PID_SOLO" "$TMP/solo.log"

refuse_check "$SOLO" "single-process"
curl -s "$SOLO/health" -o "$TMP/solo-health.json"
n="$(json_field "$TMP/solo-health.json" "d['jobs']['language_refused']")"
[ "$n" -ge 1 ] || fail "single-process: language_refused counter stayed at $n"
ok "the single-process server refuses too, and counts it separately ($n)"

# ------------------------------------------------------- the real property
#
# A request's bytes must not depend on what else was in flight. Three samples of
# the same request: alone on a single-pack server, alone on the fleet, and on
# the fleet while the OTHER language is saturating its own worker. If routing,
# admission or the shared scheduler ever let one language's work perturb
# another's, this is where it shows.
code="$(say "$SOLO" "$LANG_A" "$TMP/solo-a.wav" 24)"
[ "$code" = "200" ] || fail "single-pack $LANG_A returned $code"
cmp -s "$TMP/a.wav" "$TMP/solo-a.wav" \
    || fail "$LANG_A differs between a single-pack server and the two-language fleet"
ok "$LANG_A is byte-identical on a single-pack server and in the fleet"

i=0
while [ "$i" -lt 6 ]; do
    i=$((i + 1))
    say "$BASE" "$LANG_B" "$TMP/noise-$i.wav" 96 > "$TMP/noise-$i.code" &
done
sleep 1
code="$(say "$BASE" "$LANG_A" "$TMP/a-under-load.wav" 24)"
wait
[ "$code" = "200" ] || fail "$LANG_A returned $code while $LANG_B was in flight"
cmp -s "$TMP/a.wav" "$TMP/a-under-load.wav" \
    || fail "$LANG_A's bytes changed while $LANG_B was in flight — a batch mixed languages, or the scheduler is not isolated"
ok "$LANG_A is byte-identical with six $LANG_B requests in flight"

# Some of the six are expected to be REFUSED: six requests against one
# language's slots plus its queue bound is past capacity by design, and rung 1+2
# answering 503 is the correct outcome. So the reference for the comparison is
# the first one that was actually SERVED -- using noise-1 unconditionally would
# compare against a 503 body on any run where it was the one refused.
ref=""
served=0
i=0
while [ "$i" -lt 6 ]; do
    i=$((i + 1))
    [ "$(cat "$TMP/noise-$i.code")" = "200" ] || continue
    served=$((served + 1))
    if [ -z "$ref" ]; then
        ref="$TMP/noise-$i.wav"
        continue
    fi
    cmp -s "$ref" "$TMP/noise-$i.wav" \
        || fail "two identical $LANG_B requests returned different bytes"
done
[ "$served" -ge 1 ] || fail "no $LANG_B request survived the concurrent run"
ok "$served/6 concurrent $LANG_B requests served, all byte-identical to each other"

echo
echo "PASS        $PASS checks"
