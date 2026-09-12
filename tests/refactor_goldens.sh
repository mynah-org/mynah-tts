#!/usr/bin/env bash
# Capture or verify the refactor goldens for the synthetic pack.
#
# The graph.c split (PLAN.md E1) has to prove exactly one thing: that it did not
# change the numbers. This is that proof. Capture before touching src/, verify
# after every step.
#
#   tests/refactor_goldens.sh capture [PACK]
#   tests/refactor_goldens.sh verify  [PACK]
#
# Covers the combinations that catch different classes of mistake:
#   f32 vs int8      - int8 is what catches a mangled qmat cache key
#   batch 1 vs 4     - batch N must equal batch 1 (promised in src/mynah_tts.h)
#   offline vs stream - the shared state machine, via tests/test_stream.c
set -u

MODE="${1:-verify}"
PACK="${2:-models/fake-magpie}"
BIN="${BIN:-build/cpu/mynah-tts}"
GOLDEN="${GOLDEN:-tests/goldens/fake-magpie.sha256}"
TOKENS="7,42,13,99,64,5,88,21"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

[ -x "$BIN" ] || { echo "missing binary: $BIN (run make)" >&2; exit 2; }
[ -f "$PACK/model.json" ] || {
    echo "missing pack: $PACK" >&2
    echo "  generate it with: make fake-pack" >&2
    exit 2
}

# Record the pack's own hashes. If the generator or numpy's bit stream ever
# changes, the goldens are stale and this says so instead of reporting a
# refactor regression that is not one.
pack_hashes() {
    printf '%s  %s\n' "$(shasum -a 256 "$PACK/tts.safetensors" | cut -d' ' -f1)" "pack:tts"
    printf '%s  %s\n' "$(shasum -a 256 "$PACK/codec.safetensors" | cut -d' ' -f1)" "pack:codec"
}

emit() {  # name, quant, speaker, seed
    local name="$1" quant="$2" speaker="$3" seed="$4"
    local out="$WORK/$name.wav"
    if ! env MYNAH_QUANT="$quant" "$BIN" --synthesize "$PACK" --tokens "$TOKENS" \
            --output "$out" --speaker "$speaker" --seed "$seed" --max-steps 6 \
            >"$WORK/$name.log" 2>&1; then
        echo "FAIL: $name did not run" >&2
        sed 's/^/    /' "$WORK/$name.log" >&2
        exit 1
    fi
    printf '%s  %s\n' "$(shasum -a 256 "$out" | cut -d' ' -f1)" "$name"
}

{
    pack_hashes
    emit offline-f32   f32  3 1234
    emit offline-int8  int8 3 1234
    emit offline-f16   f16  3 1234
    emit speaker0      f32  0 1234   # different baked context row
    emit seed99        f32  3 99     # different RNG stream
} > "$WORK/actual.sha256"

# Batch 4 writes OUTPUT.1..3 alongside OUTPUT; job 0 must equal the solo run.
env MYNAH_QUANT=f32 "$BIN" --synthesize "$PACK" --tokens "$TOKENS" \
    --output "$WORK/batch.wav" --speaker 3 --seed 1234 --max-steps 6 --batch 4 \
    >"$WORK/batch.log" 2>&1 || { echo "FAIL: batch run" >&2; cat "$WORK/batch.log" >&2; exit 1; }
printf '%s  %s\n' "$(shasum -a 256 "$WORK/batch.wav" | cut -d' ' -f1)" "batch4-job0" \
    >> "$WORK/actual.sha256"

if [ "$MODE" = "capture" ]; then
    mkdir -p "$(dirname "$GOLDEN")"
    cp "$WORK/actual.sha256" "$GOLDEN"
    echo "captured $(wc -l < "$GOLDEN" | tr -d ' ') goldens -> $GOLDEN"
    cat "$GOLDEN"
    exit 0
fi

[ -f "$GOLDEN" ] || { echo "no goldens at $GOLDEN; run: $0 capture" >&2; exit 2; }

if ! diff <(grep '^.* pack:' "$GOLDEN") <(pack_hashes) > "$WORK/packdiff" 2>&1; then
    echo "goldens: STALE - the synthetic pack is not the one they were captured from" >&2
    cat "$WORK/packdiff" >&2
    echo "  regenerate with 'make fake-pack', or re-capture if the generator changed" >&2
    exit 2
fi

if diff -u "$GOLDEN" "$WORK/actual.sha256" > "$WORK/diff"; then
    echo "goldens: PASS ($(wc -l < "$GOLDEN" | tr -d ' ') checks)"
else
    echo "goldens: FAIL - the refactor changed the audio" >&2
    cat "$WORK/diff" >&2
    exit 1
fi

# batch 4 job 0 must be bit-identical to the solo run, not merely stable
solo=$(grep ' offline-f32$' "$GOLDEN" | cut -d' ' -f1)
b0=$(grep ' batch4-job0$' "$WORK/actual.sha256" | cut -d' ' -f1)
if [ "$solo" != "$b0" ]; then
    echo "goldens: FAIL - batch 4 job 0 differs from the solo run" >&2
    exit 1
fi
echo "batch parity: PASS"
