#!/bin/sh
# The refusals a WORKER cannot see, and an operator must.
#
#   make server-refusal-test MODEL_DIR=models/pocket-en
#
# WHY THIS EXISTS. On 2026-09-22 a 2x12 prefork server on an EPYC 9254 answered
# 18 of 72 requests with HTTP 503 server_at_capacity while every worker's
# /health reported "rejected": 0. Nothing was lying: the refusal happens in the
# ROUTER, which is the parent, which serves no HTTP, and its counters went to
# stderr and nowhere else. A monitoring system polling /health saw a perfect
# server dropping a quarter of its traffic.
#
# ANTI-VACUITY, and it is the whole design of this file. A test that only
# checked "router.refused_total > 0 after overload" would pass against a server
# that hardcoded a large number, and -- worse -- would say nothing about the
# case that actually misleads. So:
#
#   1 a SINGLE-PROCESS server must report router: null, not zeros. "No
#     refusals" and "nobody is counting" are different facts and a dashboard
#     built on the second one is built on sand.
#   2 a prefork server before any load must report a zero block, so the null
#     above is a real distinction and not just an unimplemented field.
#   3 under overload the count must MATCH the 503s the client actually got --
#     not merely be nonzero -- and be split by reason.
#   4 the worker's own jobs.rejected must stay 0, because it genuinely never
#     saw those requests. If that number moved, the fix would have been to
#     double-count somewhere.
set -eu

SERVER="${SERVER:-build/cpu/mynah-tts-server}"
MODEL_DIR="${MODEL_DIR:?set MODEL_DIR to a model pack}"
PORT="${PORT:-8991}"
BURST="${BURST:-40}"
STEPS="${STEPS:-120}"
PID=""
fail=0
say() { printf '  %-5s %s\n' "$1" "$2"; [ "$1" = "ok" ] || fail=1; }
cleanup() { [ -n "$PID" ] && kill "$PID" 2>/dev/null; wait "$PID" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

start() {  # start <port> [extra args...]
    p=$1; shift
    "$SERVER" -m "$MODEL_DIR" -p "$p" --warmup 0 "$@" >"/tmp/refusals-$p.log" 2>&1 &
    PID=$!
    i=0
    while [ $i -lt 120 ]; do
        curl -sf "http://127.0.0.1:$p/health" >/dev/null 2>&1 && return 0
        kill -0 "$PID" 2>/dev/null || { echo "server died:"; tail -5 "/tmp/refusals-$p.log"; return 1; }
        i=$((i+1)); sleep 1
    done
    echo "server never became healthy"; return 1
}

field() {  # field <port> <python expression over the health doc `d`>
    curl -s "http://127.0.0.1:$1/health" | python3 -c "import json,sys;d=json.load(sys.stdin);print($2)"
}

echo "== 1. a single process has no router, and must not pretend to have one =="
start "$PORT" || exit 1
R=$(field "$PORT" 'json.dumps(d.get("router"))')
[ "$R" = "null" ] \
  && say ok "router: null -- UNKNOWN, not a zero that would read as a measurement" \
  || say FAIL "single process reported router=$R; zeros here are a false negative"
cleanup; PID=""; sleep 1

echo
echo "== 2. a prefork server carries the block, at zero, before any load =="
PORT2=$((PORT+1))
start "$PORT2" --prefork 2 --prefork-threads 2 || exit 1
T0=$(field "$PORT2" 'd["router"]["refused_total"]')
[ "$T0" = "0" ] \
  && say ok "refused_total 0 before load, so the block is real and not a constant" \
  || say FAIL "a freshly started server already reports $T0 refusals"

echo
echo "== 3. overload: the count must MATCH the 503s the client was given =="
codes=$(mktemp)
i=0
burst_pids=""
while [ $i -lt "$BURST" ]; do
    ( curl -s -o /dev/null -w '%{http_code}\n' --max-time 120 \
        -X POST "http://127.0.0.1:$PORT2/v1/audio/speech" \
        -H 'Content-Type: application/json' \
        -d "{\"input\":\"the quick brown fox jumps over the lazy dog\",\"voice\":\"0\",\"max_steps\":$STEPS}" \
        >>"$codes" ) &
    burst_pids="$burst_pids $!"
    i=$((i+1))
done
# NAMED pids, never a bare `wait`. A bare wait also waits for $PID -- the
# SERVER, started in the background by start() -- which never exits, so the
# gate hung for thirty-four minutes with every curl already finished. The
# symptom was indistinguishable from a wedged server, which is the worst kind
# of test bug: it accuses the thing it is testing.
for bp in $burst_pids; do wait "$bp" 2>/dev/null || true; done
GOT503=$(grep -c '^503$' "$codes" || true)
GOT200=$(grep -c '^200$' "$codes" || true)
rm -f "$codes"
echo "  client saw: $GOT200 x 200, $GOT503 x 503 out of $BURST"
[ "$GOT503" -gt 0 ] \
  || say FAIL "nothing was refused at all -- raise BURST or lower the slot cap;
      this gate cannot prove anything about a server that never hit its limit"

T1=$(field "$PORT2" 'd["router"]["refused_total"]')
[ "$T1" = "$GOT503" ] \
  && say ok "router.refused_total = $T1 = the 503s the client counted" \
  || say FAIL "client saw $GOT503 refusals, /health reports $T1"

BY=$(field "$PORT2" 'json.dumps(d["router"]["refused"])')
echo "$BY" | grep -q 'server_at_capacity' \
  && say ok "split by reason: $BY" \
  || say FAIL "no per-reason breakdown: $BY"

echo
echo "== 4. the worker's own counter must NOT move: it never saw them =="
WR=$(field "$PORT2" 'd["jobs"]["rejected"]')
[ "$WR" = "0" ] \
  && say ok "jobs.rejected stayed 0 -- the router's refusals are not double counted" \
  || say FAIL "a worker claims $WR rejections it could not have seen"

echo
if [ "$fail" -eq 0 ]; then
    echo "server refusals visible to /health: PASS"
else
    echo "server refusals visible to /health: FAIL"
fi
exit "$fail"
