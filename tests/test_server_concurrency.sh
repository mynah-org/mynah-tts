#!/bin/sh
# PLAN.md E5-8 — N concurrent requests, each byte-identical to the same
# request run alone, THROUGH THE HTTP SERVER.
#
#   make server-concurrency-test MODEL_DIR=models/fake-magpie
#   make server-concurrency-test MODEL_DIR=models/pocket-en SERVER_ARGS="--prefork 4"
#   SERVER=tests/prefork_server.sh make server-concurrency-test MODEL_DIR=models/pocket-en
#
# The engine already proves batch-equals-solo bit for bit at widths 2/4/8/16.
# This is the other claim: that accept, queue, scheduler, slot assignment, the
# per-slot RNG, the sampler state, the streaming writer's framing and the
# socket add nothing of their own. A server can pass every engine test and
# still hand request A a buffer that request B wrote.
#
# What it checks:
#   1  identity   C concurrent requests == the same requests run alone,
#                 for C = 2, 4, 8, on the batch route and the streaming route
#   2  order      the same C requests in a different arrival order == solo
#   3  shapes     long and short interleaved (a ragged batch), and streaming
#                 and non-streaming mixed inside one burst
#   4  stream==batch per request WHILE other requests are in flight
#   5  all of the above again under --prefork, where requests are spread
#                 across processes (drive it with SERVER_ARGS or the wrapper)
#
# What it deliberately does NOT do: assert on wall-clock. The pre-existing
# `batching` check in tests/test_server.sh compares one-second-granularity
# timestamps around a 200 ms workload (PLAN.md E5-10) and is flaky for that
# reason. Concurrency correctness does not need a stopwatch, so there is not
# one here, and there is no sleep-poll anywhere either.
#
# Anti-vacuity. A gate that cannot fail is worse than no gate. Three things
# here can fail on their own:
#   * the comparator self-tests before it is trusted (pcm_diff.py --self-test);
#   * every run asserts that requests which MUST differ do differ, so a server
#     returning a constant, or a corpus that collapsed to one utterance, is
#     caught instead of silently making every identity check trivially true;
#   * the number of comparisons actually executed is counted and asserted
#     against a minimum, so a loop that quietly iterates zero times fails.
set -eu

SERVER="${SERVER:-build/cpu/mynah-tts-server}"
MODEL_DIR="${MODEL_DIR:?set MODEL_DIR to a model pack (models/fake-magpie is the deterministic default)}"
PORT="${PORT:-8987}"
BASE="http://127.0.0.1:$PORT"
# Bounded so a wedged request fails the gate instead of hanging CI; never used
# as a measurement.
CURL_TIMEOUT="${CURL_TIMEOUT:-600}"
# Concurrency widths. 2, 4 and 8 by default: 8 is also this server's default
# --max-batch, so the widest burst is the one that fills a batch exactly.
LEVELS="${LEVELS:-2 4 8}"
# Bounds decode length so the gate stays a correctness gate and the machine
# stays quiet. Identical on both sides of every comparison, which is all that
# matters for an A/B.
MAX_STEPS="${MAX_STEPS:-48}"
REQ_LANGUAGE="${REQ_LANGUAGE:-en}"

ROOT="$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)"
PCM_DIFF="$ROOT/tools/pcm_diff.py"
TMP="$(mktemp -d)"
PID=""
CHECKS=0
NEG_CHECKS=0
REQUESTS=0

# A server holds the whole pack resident, so a leaked one is expensive. SIGTERM
# first, then make sure: a server killed mid-synthesis can take a moment.
cleanup() {
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

# A transport failure (an empty reply, a reset) is only actionable next to what
# the server said while it happened, and $TMP goes away on exit -- so the tail
# of the log is part of the failure message, not something to go looking for.
fail() {
    echo "FAIL: $*" >&2
    if [ -s "$TMP/server.log" ]; then
        echo "--- last 40 lines of the server log ---" >&2
        tail -40 "$TMP/server.log" >&2
        echo "---------------------------------------" >&2
    fi
    if [ -n "${KEEP_TMP:-}" ]; then
        echo "artifacts kept in $TMP (KEEP_TMP was set)" >&2
        trap - EXIT INT TERM
        if [ -n "$PID" ]; then kill "$PID" 2>/dev/null || true; fi
    fi
    exit 1
}
note() { echo "  $*"; }

# ------------------------------------------------------------------ the corpus
#
# Distinct text AND distinct voice AND distinct seed per request: if any one of
# the three leaked between slots the identity check has to notice, and a corpus
# that varied only in seed could not tell a leaked speaker embedding from a
# leaked prompt. Lengths are interleaved so that a prefix of 2, 4 or 8 is
# already a ragged mix -- the shape that exposes a shared buffer or a stale
# length, and the one a corpus of eight equal-length sentences never produces.
#
# No double quotes, backslashes or pipes in the texts: the first two would have
# to be JSON-escaped, the third is the field separator.
write_corpus() {
    printf '%s\n' $1 > "$TMP/voices"
    nvoices=$(wc -l < "$TMP/voices" | tr -d ' ')
    [ "$nvoices" -ge 1 ] || fail "no voices to build the corpus from"
    i=0
    : > "$TMP/corpus"
    printf '%s\n' \
        'short|Yes.' \
        'long|This deliberately long sentence exists so that the batch it lands in is ragged, with neighbours that finish long before it does, which is where a shared buffer or a stale length would show itself.' \
        'short|No thanks, not today.' \
        'long|Another long one, different in wording and in length from the first, so that two long requests inside one batch cannot be confused with each other by a comparison that only looks at duration.' \
        'medium|A medium length sentence, long enough to take several decoder steps.' \
        'long|A third long sentence rounds out the corpus, because eight concurrent requests with only two distinct shapes would not be a ragged batch at all.' \
        'short|Go.' \
        'medium|The quick brown fox jumps over the lazy dog, twice.' \
    | while IFS='|' read -r shape text; do
        i=$((i + 1))
        # Cycle the voice list; a pack with two voices still gets a different
        # voice in adjacent slots, which is where a swap would show.
        voice=$(sed -n "$(( (i - 1) % nvoices + 1 ))p" "$TMP/voices")
        printf '%s|%s|%s|%s|%s\n' "$i" "$voice" "$((100 + i))" "$shape" "$text" >> "$TMP/corpus"
    done
}

# body <id> <stream 0|1>
body() {
    b_line=$(grep "^$1|" "$TMP/corpus") || fail "no corpus entry $1"
    b_voice=$(printf '%s' "$b_line" | cut -d'|' -f2)
    b_seed=$(printf '%s' "$b_line" | cut -d'|' -f3)
    b_text=$(printf '%s' "$b_line" | cut -d'|' -f5)
    if [ "$2" = 1 ]; then b_stream=',"stream":true'; else b_stream=''; fi
    # Every knob explicit, on both sides of every comparison: the repo's own
    # A/B rule is model revision, language, speaker, text, seed, temperature,
    # top-k and thread count. Thread count and revision are fixed by using one
    # server process (or one prefork pool) for solo and concurrent alike.
    printf '{"input":"%s","voice":"%s","seed":%s,"temperature":%s,"top_k":%s,"max_steps":%s,"language":"%s","response_format":"%s"%s}' \
        "$b_text" "$b_voice" "$b_seed" "$TEMPERATURE" "$TOPK" "$MAX_STEPS" \
        "$REQ_LANGUAGE" "$([ "$2" = 1 ] && echo pcm || echo wav)" "$b_stream"
}

# fire <id> <stream> <out>  -- one request, foreground. The status code lands
# next to the payload: a server under load can refuse with a perfectly valid
# 503 JSON body, and comparing that body to a WAV would report "not
# byte-identical" for something that is not a correctness failure at all.
fire() {
    f_code=$(curl -sS --max-time "$CURL_TIMEOUT" -o "$3" -w '%{http_code}' \
        -X POST "$BASE/v1/audio/speech" \
        -H 'Content-Type: application/json' -d "$(body "$1" "$2")")
    f_rc=$?
    printf '%s' "$f_code" > "$3.code"
    return $f_rc
}

# expect_200 <out> <label>
expect_200() {
    e_code=$(cat "$1.code" 2>/dev/null || echo 000)
    [ "$e_code" = 200 ] && return 0
    echo "the server answered HTTP $e_code:" >&2
    head -c 400 "$1" >&2
    echo "" >&2
    fail "$2: HTTP $e_code, not 200"
}

# --------------------------------------------------------------- comparators
#
# same() and differs() both bump CHECKS. The count is asserted at the end: a
# `for` that silently iterates zero times is exactly how a gate ends up
# comparing nothing, which is what happened to the `batching` check.
same() { # a b label
    CHECKS=$((CHECKS + 1))
    [ -s "$1" ] || fail "$3: $1 is empty"
    [ -s "$2" ] || fail "$3: $2 is empty"
    if cmp -s "$1" "$2"; then return 0; fi
    echo "" >&2
    echo "NOT BYTE-IDENTICAL: $3" >&2
    python3 "$PCM_DIFF" --rate "$SAMPLE_RATE" --label-a solo --label-b "under load" \
        "$1" "$2" >&2 || true
    echo "" >&2
    fail "$3"
}

differs() { # a b label -- in-band negative control
    CHECKS=$((CHECKS + 1))
    NEG_CHECKS=$((NEG_CHECKS + 1))
    [ -s "$1" ] && [ -s "$2" ] || fail "negative control $3: an input is empty"
    if cmp -s "$1" "$2"; then
        fail "negative control failed: $3 -- two requests that must differ are
      byte-identical, so every identity check in this file is vacuous. Either
      the server is returning a constant or the corpus collapsed."
    fi
}

# ------------------------------------------------------------------- bursts
#
# run_burst <tag> <arrival order> <mode: batch|stream|alt>
# Fires every client at once and waits for all of them. The arrival order is
# the order the clients are started in, which is the only handle a test has on
# which request the scheduler sees first.
run_burst() {
    rb_tag="$1"; rb_order="$2"; rb_mode="$3"
    mkdir -p "$TMP/$rb_tag"
    # A starting gate, because "fired concurrently" was a lie without one.
    # Forking eight subshells that each then fork and exec curl staggers the
    # arrivals by however long a fork/exec takes on a loaded machine, which was
    # enough for the earliest requests to be admitted and finished before the
    # last one arrived -- a burst that never overlapped, verifying nothing. So
    # every client forks first and blocks opening this FIFO for reading, which
    # cannot complete until a writer appears; the parent opening the write end
    # releases all of them in one go. No sleep, nothing timed, no assertion on
    # when anything happened: a rendezvous, not a stopwatch.
    #
    # The write end stays open until the burst is finished on purpose. A client
    # that is late to the gate then opens immediately instead of blocking
    # forever on a writer that already left.
    mkfifo "$TMP/$rb_tag/gate" || fail "cannot create the burst gate"
    rb_pids=""
    rb_n=0
    for rb_id in $rb_order; do
        case "$rb_mode" in
            stream) rb_s=1 ;;
            batch)  rb_s=0 ;;
            alt)    rb_s=$((rb_id % 2)) ;;
            *) fail "bad burst mode $rb_mode" ;;
        esac
        if [ "$rb_s" = 1 ]; then rb_ext=pcm; else rb_ext=wav; fi
        printf '%s %s\n' "$rb_id" "$rb_ext" >> "$TMP/$rb_tag/plan"
        (
            set +e
            exec 3< "$TMP/$rb_tag/gate"
            fire "$rb_id" "$rb_s" "$TMP/$rb_tag/$rb_id.$rb_ext"
            echo $? > "$TMP/$rb_tag/$rb_id.rc"
        ) &
        rb_pids="$rb_pids $!"
        rb_n=$((rb_n + 1))
        REQUESTS=$((REQUESTS + 1))
    done
    [ "$rb_n" -gt 0 ] || fail "burst $rb_tag fired no requests"
    exec 7> "$TMP/$rb_tag/gate"    # open the gate: every client is released here
    for rb_p in $rb_pids; do
        wait "$rb_p" || fail "a client process in burst $rb_tag died"
    done
    exec 7>&-
    rm -f "$TMP/$rb_tag/gate"
    while read -r rb_id rb_ext; do
        rb_rc=$(cat "$TMP/$rb_tag/$rb_id.rc" 2>/dev/null || echo 99)
        [ "$rb_rc" = 0 ] || fail "burst $rb_tag: client $rb_id curl exited $rb_rc
      (52 is an empty reply: the connection closed before any byte arrived,
       which is not a refusal -- a refused request carries a 503 body)"
        expect_200 "$TMP/$rb_tag/$rb_id.$rb_ext" "burst $rb_tag: client $rb_id"
    done < "$TMP/$rb_tag/plan"
}

# verify_burst <tag> -- every client got exactly its solo bytes
verify_burst() {
    vb_tag="$1"
    vb_n=0
    while read -r vb_id vb_ext; do
        vb_n=$((vb_n + 1))
        case "$vb_ext" in
            wav)
                same "$TMP/solo/$vb_id.wav" "$TMP/$vb_tag/$vb_id.wav" \
                     "$vb_tag: request $vb_id (batch) differs from the same request run alone"
                ;;
            pcm)
                same "$TMP/solo/$vb_id.pcm" "$TMP/$vb_tag/$vb_id.pcm" \
                     "$vb_tag: request $vb_id (stream) differs from the same request streamed alone"
                # The claim item 4 asks for directly: the concatenated chunks of
                # a streamed request equal that request's NON-streamed bytes,
                # while other requests are in flight. Checked against the solo
                # WAV payload, not derived by transitivity from the line above.
                same "$TMP/solo/$vb_id.payload" "$TMP/$vb_tag/$vb_id.pcm" \
                     "$vb_tag: request $vb_id streamed under load differs from its own batch audio"
                ;;
        esac
    done < "$TMP/$vb_tag/plan"
    [ "$vb_n" -gt 0 ] || fail "verify_burst $vb_tag verified nothing"
    echo "$vb_n" > "$TMP/$vb_tag/verified"
}

# ------------------------------------------------------------------- preflight

command -v python3 > /dev/null 2>&1 || fail "python3 is required (tools/pcm_diff.py)"
[ -f "$PCM_DIFF" ] || fail "missing $PCM_DIFF"
# Trust nothing that has not shown it can say no.
python3 "$PCM_DIFF" --self-test || fail "the comparator failed its own self-test"

if curl -sf --max-time 2 "$BASE/health" > /dev/null 2>&1; then
    fail "something is already serving on port $PORT; stop it first"
fi

# Sampler defaults come from the pack, not from this script: the point is to
# pass the SAME explicit values on both sides, not to impose magpie's numbers
# on pocket. Overridable, and printed, so a run is reproducible from its log.
PACK_DEFAULTS=$(python3 - "$MODEL_DIR/model.json" <<'PY'
import json, sys
try:
    with open(sys.argv[1]) as fh:
        m = json.load(fh)
except Exception:
    m = {}
print(m.get("temperature", 0.7), m.get("topk", 80))
PY
)
TEMPERATURE="${TEMPERATURE:-$(printf '%s' "$PACK_DEFAULTS" | cut -d' ' -f1)}"
TOPK="${TOPK:-$(printf '%s' "$PACK_DEFAULTS" | cut -d' ' -f2)}"

# shellcheck disable=SC2086
"$SERVER" -m "$MODEL_DIR" -p "$PORT" ${SERVER_ARGS:-} > "$TMP/server.log" 2>&1 &
PID=$!

i=0
while [ "$i" -lt 120 ]; do
    if curl -sf --max-time 2 "$BASE/health" > /dev/null 2>&1; then break; fi
    kill -0 "$PID" 2>/dev/null || { cat "$TMP/server.log" >&2; fail "server exited"; }
    i=$((i + 1))
    sleep 1
done
[ "$i" -lt 120 ] || { cat "$TMP/server.log" >&2; fail "server did not become ready"; }

curl -s "$BASE/health" > "$TMP/health0.json"
read_health() { # key
    python3 - "$TMP/$2" "$1" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
path = sys.argv[2].split('.')
for p in path:
    d = d[p]
print(d)
PY
}
SAMPLE_RATE=$(read_health sample_rate health0.json)
MAX_BATCH=$(read_health limits.max_batch health0.json)
THREADS=$(read_health process.synthesis_threads health0.json)
PREFORK=$(read_health process.prefork_worker health0.json)
ENGINE=$(read_health engine health0.json)
MODEL_ID=$(read_health model health0.json)
COMPLETED0=$(read_health jobs.completed health0.json)

VOICES=$(curl -s "$BASE/v1/voices" | python3 -c '
import json,sys
v=[x["name"] for x in json.load(sys.stdin)["voices"]]
print(" ".join(v[:8]))')
[ -n "$VOICES" ] || fail "/v1/voices returned no voices"

echo "E5-8 concurrency identity gate"
note "model      $MODEL_ID  engine=$ENGINE  rate=${SAMPLE_RATE}Hz"
note "server     $SERVER ${SERVER_ARGS:-}  max_batch=$MAX_BATCH  threads=$THREADS  prefork_worker=$PREFORK"
note "params     temperature=$TEMPERATURE top_k=$TOPK max_steps=$MAX_STEPS language=$REQ_LANGUAGE"
note "voices     $VOICES"
note "levels     $LEVELS"
echo ""

write_corpus "$VOICES"
N=$(wc -l < "$TMP/corpus" | tr -d ' ')
ALL=$(cut -d'|' -f1 "$TMP/corpus" | tr '\n' ' ')
[ "$N" -ge 8 ] || fail "corpus has $N entries, need at least 8"

# ------------------------------------------------- phase 1: the solo reference
#
# Serial, on a quiet server: this is what "the same request run alone" means.
mkdir -p "$TMP/solo"
for id in $ALL; do
    fire "$id" 0 "$TMP/solo/$id.wav" || fail "solo request $id (batch) failed"
    fire "$id" 1 "$TMP/solo/$id.pcm" || fail "solo request $id (stream) failed"
    REQUESTS=$((REQUESTS + 2))
    expect_200 "$TMP/solo/$id.wav" "solo request $id (batch)"
    expect_200 "$TMP/solo/$id.pcm" "solo request $id (stream)"
    [ -s "$TMP/solo/$id.wav" ] || fail "solo request $id returned no audio"
    head -c 4 "$TMP/solo/$id.wav" | grep -q RIFF || fail "solo request $id is not a RIFF file"
    python3 "$PCM_DIFF" "$TMP/solo/$id.wav" --extract "$TMP/solo/$id.payload" \
        || fail "cannot extract the pcm payload of solo request $id"
    same "$TMP/solo/$id.payload" "$TMP/solo/$id.pcm" \
         "solo: request $id streamed differs from the same request batched"
done

# The corpus must be discriminating, or every identity check below is a
# tautology. All pairs, not a sample: with N=8 it is 28 cheap comparisons and
# it also catches "voice was ignored" and "seed was ignored".
for a in $ALL; do
    for b in $ALL; do
        [ "$a" -lt "$b" ] || continue
        differs "$TMP/solo/$a.wav" "$TMP/solo/$b.wav" \
                "solo requests $a and $b (different text, voice and seed)"
    done
done
LONGEST=$(for id in $ALL; do printf '%s %s\n' "$(wc -c < "$TMP/solo/$id.wav" | tr -d ' ')" "$id"; done | sort -n | tail -1 | cut -d' ' -f2)
SHORTEST=$(for id in $ALL; do printf '%s %s\n' "$(wc -c < "$TMP/solo/$id.wav" | tr -d ' ')" "$id"; done | sort -n | head -1 | cut -d' ' -f2)
note "solo        $N requests, batch and stream, all distinct pairwise"
note "            shortest #$SHORTEST $(wc -c < "$TMP/solo/$SHORTEST.wav" | tr -d ' ') B, longest #$LONGEST $(wc -c < "$TMP/solo/$LONGEST.wav" | tr -d ' ') B"
if [ "$(wc -c < "$TMP/solo/$SHORTEST.wav" | tr -d ' ')" = "$(wc -c < "$TMP/solo/$LONGEST.wav" | tr -d ' ')" ]; then
    note "            NOTE this pack produces a fixed duration, so the batches below"
    note "            are not ragged in length. Run the gate against a pack that"
    note "            stops on EOS (models/pocket-en) for the ragged case."
fi
echo ""

# ---------------------------------------- phase 2..4: identity under concurrency
for C in $LEVELS; do
    [ "$C" -le "$N" ] || fail "level $C exceeds the corpus size $N"
    FWD=$(printf '%s ' $ALL | cut -d' ' -f1-"$C")
    REV=$(printf '%s\n' $FWD | sed '1!G;h;$!d' | tr '\n' ' ')
    # Ragged on purpose: the longest and the shortest request of the corpus
    # lead, so the batch has to carry two very different lengths side by side,
    # and the rest follows in corpus order.
    RAGGED="$LONGEST $SHORTEST"
    for id in $ALL; do
        [ "$id" = "$LONGEST" ] && continue
        [ "$id" = "$SHORTEST" ] && continue
        RAGGED="$RAGGED $id"
    done
    RAGGED=$(printf '%s ' $RAGGED | cut -d' ' -f1-"$C")

    run_burst "c${C}_batch" "$FWD" batch
    verify_burst "c${C}_batch"
    note "C=$C batch    $C concurrent, arrival order [$FWD] — all byte-identical to solo"

    run_burst "c${C}_stream" "$FWD" stream
    verify_burst "c${C}_stream"
    note "C=$C stream   $C concurrent streams — identical to solo stream AND to solo batch"

    run_burst "c${C}_rev" "$REV" batch
    verify_burst "c${C}_rev"
    note "C=$C order    same $C requests, arrival order [$REV] — still identical"

    run_burst "c${C}_mixed" "$RAGGED" alt
    verify_burst "c${C}_mixed"
    note "C=$C mixed    long+short, streaming+batch in one burst [$RAGGED] — identical"
done
echo ""

# --------------------------------------------------------------- accounting
#
# Proof the requests reached synthesis rather than being answered by something
# cheaper: the server's own completed counter has to have moved by exactly the
# number of requests this script issued. Under prefork each /health lands on
# one worker and reports only that worker's counters, so the check is an
# inequality there instead of an equality.
curl -s "$BASE/health" > "$TMP/health1.json"
COMPLETED1=$(read_health jobs.completed health1.json)
FAILED=$(read_health jobs.failed health1.json)
REJECTED=$(read_health jobs.rejected health1.json)
TIMEDOUT=$(read_health jobs.timed_out health1.json)
DELTA=$((COMPLETED1 - COMPLETED0))
if [ "$PREFORK" = "-1" ]; then
    [ "$DELTA" = "$REQUESTS" ] ||
        fail "the server completed $DELTA jobs but this script issued $REQUESTS;
      some request did not reach synthesis"
    [ "$FAILED" = 0 ] && [ "$REJECTED" = 0 ] && [ "$TIMEDOUT" = 0 ] ||
        fail "server counters: failed=$FAILED rejected=$REJECTED timed_out=$TIMEDOUT"
    note "accounting  $DELTA jobs completed == $REQUESTS requests issued, 0 failed/rejected/timed out"
else
    [ "$DELTA" -gt 0 ] ||
        fail "the worker answering /health completed nothing; no request reached synthesis"
    note "accounting  $REQUESTS requests issued; worker $PREFORK alone completed $DELTA (per-worker counters)"
    # Worth saying out loud: the gap between the two numbers is the only
    # evidence in this run that the router actually spread the work. If it
    # closed, the prefork lane degenerated into the single-process lane and
    # proved nothing extra -- which is not a correctness failure, but it is
    # not the check anyone thought they ran either.
    if [ "$DELTA" = "$REQUESTS" ]; then
        note "            WARNING one worker served every request: this run did NOT"
        note "            exercise cross-process routing."
    else
        note "            $((REQUESTS - DELTA)) of them were served by other processes, so the"
        note "            identity above holds across workers, not just within one"
    fi
fi

# ------------------------------------------------------ the gate on the gate
#
# If any of this had quietly compared nothing, the counts below are what says
# so. Minimum, not exact, so adding a check does not mean editing a constant.
EXPECTED_MIN=$((N + 28))
for C in $LEVELS; do
    # batch C, stream 2C (vs solo stream and vs solo batch), rev C,
    # mixed at least C (more when the burst contains streaming clients).
    EXPECTED_MIN=$((EXPECTED_MIN + 5 * C))
done
[ "$CHECKS" -ge "$EXPECTED_MIN" ] ||
    fail "only $CHECKS comparisons ran, expected at least $EXPECTED_MIN — this gate
      compared less than it claims to"
[ "$NEG_CHECKS" -ge 28 ] ||
    fail "only $NEG_CHECKS negative controls ran; without them the identity checks
      could all be trivially true"

echo ""
note "$CHECKS byte comparisons, of which $NEG_CHECKS are must-differ controls"
note "$REQUESTS requests issued in total"
echo "server concurrency identity (E5-8): PASS"
