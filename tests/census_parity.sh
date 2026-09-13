#!/usr/bin/env bash
# census_parity.sh — the proof that the instrumentation does not change what it
# measures, and the measurement of what it costs.
#
# Without this, every number the census and the cost map produce is
# unfalsifiable. .work/engineering-method.md §4:
#
#   "parity proof: the load is run twice with the same binary — census only,
#    then census + cost map — and the two census reports are diffed. Without
#    that proof the numbers would be unfalsifiable."
#
#   "overhead measured A/B/B/A interleaved, never a clean run followed by an
#    instrumented one: drift between two identical runs can be larger than the
#    effect."
#
# Three modes:
#
#   parity    same workload, census-only vs census+costmap. The two censuses
#             must agree on every path id, every count and every fallback, and
#             the AUDIO must be byte-identical — instrumentation that moves the
#             audio changed the engine.
#   overhead  A/B/B/A interleaved arms (A = clean, B = instrumented), reporting
#             the spread of the clean arms alongside the effect, because a
#             effect smaller than the drift is not an effect.
#   alloc     the allocation count is CONSTANT across --max-steps, which is the
#             evidence that the autoregressive loop allocates nothing.
#
# Usage: tests/census_parity.sh <mode> <binary> <model-dir> [reps]
set -u

MODE="${1:-parity}"
BIN="${2:-build/cpu/mynah-tts}"
MODEL="${3:-models/pocket-en}"
REPS="${4:-3}"

TEXT="the quick brown fox jumps over the lazy dog"
STEPS=48
COMMON=(--text "$TEXT" --lang en --seed 1234 --max-steps "$STEPS")

if [ ! -x "$BIN" ]; then
    echo "SKIP: no binary at $BIN (build it with make)"; exit 0
fi
if [ ! -d "$MODEL" ]; then
    echo "SKIP: no model pack at $MODEL — reporting the skip rather than"
    echo "      substituting an unrecorded claim (CLAUDE.md testing checklist)"
    exit 0
fi

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# `env -u` on every steering variable, not just the ones we set. A leaked
# MYNAH_QUANT from an earlier command in the same shell once made an
# instrumented arm look as though the instrumentation had moved the audio; the
# arms must differ ONLY in what this function is asked to differ in.
run_arm() {   # run_arm <tag> <env...> -- ; writes $WORK/<tag>.wav
    local tag="$1"; shift
    env -u MYNAH_QUANT -u MYNAH_QUANT_GROUPS -u MYNAH_CENSUS -u MYNAH_CENSUS_JSON \
        -u MYNAH_COST_MAP -u MYNAH_COSTMAP_JSON -u MYNAH_CENSUS_STRICT \
        -u MYNAH_COST_MAP_STRICT -u MYNAH_THREADS \
        "$@" "$BIN" --synthesize "$MODEL" "${COMMON[@]}" \
        --output "$WORK/$tag.wav" >"$WORK/$tag.out" 2>"$WORK/$tag.err"
    return $?
}

# Compare the two census JSONs on everything that describes WHAT RAN. Wall
# time is expected to differ; path ids, counts and fallbacks are not.
census_key() {  # census_key <json> -> sorted "block kind k n rows path enc calls"
    python3 - "$1" <<'PY'
import json, sys
doc = json.load(open(sys.argv[1]))
rows = []
for o in doc.get("ops", []):
    rows.append("%s %s %s %s %s %s %s %s" % (
        o["block"], o["kind"], o["k"], o["n"], o["rows"],
        o["path"], o["encoding"], o["calls"]))
for line in sorted(rows):
    print(line)
print("BATCHED", doc.get("batched_calls"))
print("ROWLOOP", doc.get("rowloop_calls"))
print("CALLS", doc.get("calls"))
PY
}

# ------------------------------------------------------------------ parity
if [ "$MODE" = "parity" ]; then
    echo "[PARITY] same binary, same workload, census-only vs census+costmap"

    # The CLEAN arm. The census cannot be compared against it -- a clean run
    # produces no census -- but the AUDIO can, and that is the stronger claim:
    # instrumentation that moves the audio changed the engine, and comparing
    # only the two instrumented arms would never see it.
    run_arm clean \
        || { echo "FAIL: clean arm exited $?"; cat "$WORK/clean.err"; exit 1; }

    run_arm censusonly MYNAH_CENSUS=1 MYNAH_CENSUS_JSON="$WORK/a.json" \
        || { echo "FAIL: census-only arm exited $?"; cat "$WORK/censusonly.err"; exit 1; }
    run_arm bothon MYNAH_CENSUS=1 MYNAH_CENSUS_JSON="$WORK/b.json" \
        MYNAH_COST_MAP=2 MYNAH_COSTMAP_JSON="$WORK/b.cost.json" \
        || { echo "FAIL: census+costmap arm exited $?"; cat "$WORK/bothon.err"; exit 1; }

    for f in "$WORK/a.json" "$WORK/b.json"; do
        [ -s "$f" ] || { echo "FAIL: $f was not written"; exit 1; }
    done

    census_key "$WORK/a.json" > "$WORK/a.key"
    census_key "$WORK/b.json" > "$WORK/b.key"

    if ! diff -u "$WORK/a.key" "$WORK/b.key" > "$WORK/census.diff"; then
        echo "FAIL: the two censuses disagree. The instrumentation changed"
        echo "      what it measures, so every number it produces is void."
        cat "$WORK/census.diff"
        exit 1
    fi
    echo "  census identical: same path ids, same counts, same fallbacks"
    echo "  ($(grep -c . "$WORK/a.key") comparable lines)"

    # The stronger claim: the AUDIO did not move.
    for arm in censusonly bothon; do
        if ! cmp -s "$WORK/clean.wav" "$WORK/$arm.wav"; then
            echo "FAIL: arm '$arm' produced DIFFERENT AUDIO from the CLEAN run."
            echo "      Instrumentation that moves the audio is instrumentation"
            echo "      that changed the engine."
            exit 1
        fi
    done
    echo "  audio byte-identical: clean == census-only == census+costmap"

    # The cost map must also not have refused, or the pair is unreadable.
    if grep -q '"nest_mismatch": [1-9]' "$WORK/b.cost.json" 2>/dev/null; then
        echo "FAIL: the cost map recorded a nest mismatch in the paired run"
        exit 1
    fi
    echo "  cost map clean (nest_mismatch=0)"
    echo "PARITY PASS"
    exit 0
fi

# ---------------------------------------------------------------- overhead
if [ "$MODE" = "overhead" ]; then
    echo "[OVERHEAD] interleaved palindrome arms, ${REPS} cycles"
    echo "  Each cycle runs  clean census cost1 cost2 cost2 cost1 census clean,"
    echo "  so every arm sits at both ends of the cycle and a monotonic drift"
    echo "  cancels instead of landing on whichever arm ran last. A clean run"
    echo "  followed by an instrumented one would measure the drift."
    echo
    echo "  The arms are reported SEPARATELY. Lumping census+level2 into one"
    echo "  'instrumented' number hides which half costs what, and level 1 is"
    echo "  the one that is meant to be cheap enough to leave on."

    for arm in clean census cost1 cost2; do
        : > "$WORK/$arm.times"; : > "$WORK/$arm.synth"
    done

    # TWO numbers per arm, and conflating them gave a wrong answer once
    # already:
    #
    #   synth  the engine's own reported synthesis time. This is what the
    #          discipline is about -- instrumentation must not change what it
    #          MEASURES, and this is the thing being measured.
    #   wall   whole-process time, which also contains the report written at
    #          EXIT. The census's refusal check collects the dispatch table,
    #          and that opens backends and runs kernel self-tests (~200 ms).
    #          Real, but paid after the audio is written and charged to no
    #          measurement.
    #
    # Reporting only `wall` made the census look like a 31% hot-path tax when
    # its in-loop cost was under the noise floor.
    time_arm() {   # time_arm <arm>
        local arm="$1" t0 t1 s
        t0=$(python3 -c 'import time;print(time.monotonic())')
        case "$arm" in
          clean)  run_arm "x_$arm" ;;
          census) run_arm "x_$arm" MYNAH_CENSUS=1 MYNAH_CENSUS_JSON=/dev/null ;;
          cost1)  run_arm "x_$arm" MYNAH_COST_MAP=1 MYNAH_COSTMAP_JSON=/dev/null ;;
          cost2)  run_arm "x_$arm" MYNAH_COST_MAP=2 MYNAH_COSTMAP_JSON=/dev/null ;;
        esac
        t1=$(python3 -c 'import time;print(time.monotonic())')
        python3 -c "print($t1-$t0)" >> "$WORK/$arm.times"
        s=$(grep -o 'synth=[0-9.]*' "$WORK/x_$arm.out" 2>/dev/null | head -1 | cut -d= -f2)
        [ -n "$s" ] && echo "$s" >> "$WORK/$arm.synth"
    }

    for i in $(seq 1 "$REPS"); do
        for arm in clean census cost1 cost2 cost2 cost1 census clean; do
            time_arm "$arm"
        done
    done

    python3 - "$WORK" <<'PY'
import statistics as st, sys, os
work = sys.argv[1]
def load(a, ext):
    p = os.path.join(work, a + ext)
    try:
        return [float(x) for x in open(p) if x.strip()]
    except OSError:
        return []

ARMS = (("census", "census only"), ("cost1", "cost map L1"), ("cost2", "cost map L2"))

for ext, title, note in (
    (".synth", "IN-RUN SYNTHESIS TIME -- the thing being measured",
     "This is the number the discipline is about. An instrument that moves "
     "this changed the engine."),
    (".times", "WHOLE-PROCESS WALL -- includes the report written at EXIT",
     "The census writes its report and runs its refusal check here, after the "
     "audio. Real cost, but charged to no measurement."),
):
    clean = load("clean", ext)
    print()
    print(f"  {title}")
    print(f"  {note}")
    if len(clean) < 2:
        print("    [UNKNOWN] too few clean arms to say anything")
        continue
    cm = st.median(clean)
    spread = (max(clean) - min(clean)) / cm * 100.0
    print(f"    clean median {cm*1000:8.1f} ms  n={len(clean)}  "
          f"spread {spread:5.1f}%  <- noise floor")
    for name, label in ARMS:
        v = load(name, ext)
        if len(v) < 2:
            print(f"    {label:<12} [UNKNOWN]")
            continue
        m = st.median(v)
        eff = (m - cm) / cm * 100.0
        verdict = (f"BELOW the {spread:.1f}% noise floor -- bounded, not resolved"
                   if abs(eff) < spread
                   else f"RESOLVED, above the {spread:.1f}% noise floor")
        print(f"    {label:<12} {m*1000:8.1f} ms  {eff:+7.1f}%   {verdict}")
print()
print("  An arm whose effect is below the noise floor has NOT been measured at")
print("  this rep count; it has been bounded. Raise REPS to resolve it, or")
print("  quote the bound and not a number.")
PY
    echo "OVERHEAD DONE"
    exit 0
fi

# ------------------------------------------------------------------- alloc
if [ "$MODE" = "alloc" ]; then
    SHIM="${SHIM:-build/cpu/alloc_shim.so}"
    if [ ! -f "$SHIM" ]; then
        echo "SKIP: no shim at $SHIM (make alloc-shim)"; exit 0
    fi
    echo "[ALLOC] the count must be CONSTANT across --max-steps:"
    echo "        that is the evidence the AR loop allocates nothing."

    # LINUX ONLY, and that is a finding rather than a shortcut.
    #
    # DYLD_FORCE_FLAT_NAMESPACE interposes malloc for EVERY library in the
    # process, including the ones Accelerate pulls in, and a 24-step synthesis
    # that takes 0.7 s natively did not finish in five minutes under it. A
    # target that hangs is worse than one that says why it cannot run, and
    # production is Linux (CLAUDE.md): LD_PRELOAD is the mechanism the original
    # 2,786-allocation measurement used, and it is the one this checks.
    if [ "$(uname -s)" = "Darwin" ]; then
        echo "SKIP: macOS. Counting allocations here needs"
        echo "      DYLD_FORCE_FLAT_NAMESPACE, which interposes malloc for the"
        echo "      whole process including Accelerate, and makes the run too"
        echo "      slow to finish. Exact command skipped:"
        echo "        DYLD_INSERT_LIBRARIES=$SHIM DYLD_FORCE_FLAT_NAMESPACE=1 \\"
        echo "          $BIN --synthesize $MODEL --max-steps {24,96}"
        echo "      Run it on Linux, where LD_PRELOAD costs nothing."
        exit 0
    fi
    PRE=(LD_PRELOAD="$SHIM")

    for steps in 24 96; do
        env "${PRE[@]}" MYNAH_ALLOC_COUNT_FILE="$WORK/alloc.$steps.json" \
            "$BIN" --synthesize "$MODEL" --text "$TEXT" --lang en --seed 1234 \
            --max-steps "$steps" --output "$WORK/alloc.$steps.wav" \
            >/dev/null 2>"$WORK/alloc.$steps.err"
        if [ ! -s "$WORK/alloc.$steps.json" ]; then
            echo "SKIP: the shim did not report (preloading unavailable here)."
            echo "      stderr: $(tail -1 "$WORK/alloc.$steps.err" 2>/dev/null)"
            exit 0
        fi
    done

    python3 - "$WORK/alloc.24.json" "$WORK/alloc.96.json" <<'PY'
import json, sys
a = json.load(open(sys.argv[1])); b = json.load(open(sys.argv[2]))
print(f"  --max-steps 24: total={a['total']} malloc={a['malloc']} calloc={a['calloc']}")
print(f"  --max-steps 96: total={b['total']} malloc={b['malloc']} calloc={b['calloc']}")
if a["total"] != b["total"]:
    d = b["total"] - a["total"]
    print(f"  FAIL: {d:+d} allocations for 4x the steps. Something in the")
    print(f"        autoregressive loop allocates (CLAUDE.md rule 4).")
    raise SystemExit(1)
print(f"  PASS: {a['total']} allocations either way — the AR loop allocates nothing.")
PY
    rc=$?
    [ $rc -eq 0 ] && echo "ALLOC PASS"
    exit $rc
fi

echo "unknown mode: $MODE (parity|overhead|alloc)"
exit 2
