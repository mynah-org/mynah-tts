#!/bin/sh
# Negative control for tests/test_kernels.c and mynah_vecmath_self_test().
#
# A suite that has only ever passed is not evidence. Three gates shipped in
# this repository in one week that passed while testing nothing, and one of
# them had been passing for weeks. So before trusting the kernel suite, break
# the kernels it watches -- one deliberate defect per property it claims to
# police -- and require it to notice every time.
#
# The nine breaks, and what each one is standing in for:
#
#   1  sin-reduction  strip the exactness out of the argument reduction --
#                     the f32 Cody-Waite subtraction, written the way
#                     -ffast-math is PERMITTED to rewrite it. THE REAL BUG
#                     this suite found on its first run. It is written as the
#                     already-reassociated form on purpose: the natural
#                     spelling is a catastrophe under clang and harmless under
#                     gcc 15.2, so a break that relies on the compiler doing
#                     the rewrite is only a defect on one of the two machines,
#                     and this script would then demand a catch that is not
#                     owed. Measured, clang -O3 -ffast-math: 2.28e-04 worst
#                     absolute error over [0, 8192] against 6.9e-08 for the
#                     double reduction.
#   2  sin-handover   raise the reduction limit past infinity, so the libm
#                     hand-off for huge arguments never happens
#   3  sin-sign       drop the sign of x, so sin stops being odd and sin(-0)
#                     comes back as +0
#   4  tanh-knee      saturate at 6 instead of 9.010913, so tanh(7) returns
#                     exactly 1 when it should not
#   5  tanh-exp       drop the exp branch and extend the small polynomial over
#                     the whole line
#   6  sign-lost      stop applying the sign of x at all, so tanh and sin
#                     lose oddness and tanh(-0) comes back as +0
#
# One break was TRIED AND WITHDRAWN, and it is worth recording why. Applying
# the sign with a multiply by -1.0f instead of with a bit XOR is, on paper,
# exactly what -fno-signed-zeros is allowed to fold away -- and on clang 17 /
# arm64 it does not: the suite could not tell the two spellings apart. The bit
# form stays, because "this compiler happens not to" is not a guarantee, but
# it is not listed here as a caught defect, because on this machine it is not
# a defect. A negative control that claims a catch it did not make is worse
# than no negative control.
#   7  ln-unbiased    give the flow-head LayerNorm an unbiased variance, so
#                     the two LayerNorms stop being the same function
#   8  rms-collapse   give the flow-head variance RMSNorm the mean-square
#                     formula, i.e. exactly the substitution E3-6 exists to
#                     forbid
#   9  snake-tail     drop the scalar tail of the Snake row, so every length
#                     that is not a multiple of the vector width is wrong
#
# Every one MUST make `make kernels-test` fail. A break that slips through is
# reported by name and this script exits non-zero -- which means the SUITE,
# not the kernel, is what needs fixing.
#
# It is slow (nine rebuilds of the core), so it is deliberately NOT in
# `make test`. Run it when the kernels or this suite change.
#
#   sh tests/kernels_negative_control.sh     (or: make kernels-negative-control)
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

BREAKS="sin-reduction sin-handover sin-sign tanh-knee tanh-exp sign-lost ln-unbiased rms-collapse snake-tail"

# Apply one named break. Refuses to report success if the text it expected is
# not present: a patch that silently matched nothing would look exactly like a
# defect the suite caught, which is the one way this script could lie.
patch_break() {
    python3 - "$WORK/tree" "$1" <<'PYEOF'
import sys, os
root, which = sys.argv[1], sys.argv[2]

BREAKS = {
 "sin-reduction": ("src/kernels.c",
   """    const float z = (float)((d - y * MYNAH_SIN_PIO4_HI)
                            - y * MYNAH_SIN_PIO4_LO);""",
   """    const float yf = (float)y;
    const float z = ax - (yf * 0.78515625f
                          + yf * 2.4187564849853515625e-4f
                          + yf * 3.77489497744594108e-8f);"""),
 "sin-handover": ("src/kernels.c",
   "#define MYNAH_SIN_LIMIT 65536.0f",
   "#define MYNAH_SIN_LIMIT 1.0e30f"),
 "sin-sign": ("src/kernels.c",
   """    return mynah_apply_sign(r, x, negate);""",
   """    return mynah_apply_sign(r, 1.0f, negate);"""),
 "tanh-knee": ("src/kernels.c",
   "#define MYNAH_TANH_KNEE   9.010913f",
   "#define MYNAH_TANH_KNEE   6.0f"),
 "tanh-exp": ("src/kernels.c",
   "    if (ax < MYNAH_TANH_SMALL) {\n        /* Denormal in",
   "    if (ax < 1.0e30f) {\n        /* Denormal in"),
 "sign-lost": ("src/kernels.c",
   "    rb ^= (lb & 0x80000000u);",
   "    rb ^= (lb & 0x00000000u); /* break: never take the sign of x */"),
 "ln-unbiased": ("src/flow_head.c",
   """    /* Biased variance: torch var(unbiased=False) inside the custom LayerNorm. */
    variance /= (float)n;""",
   """    variance /= (float)(n > 1u ? n - 1u : 1u);"""),
 "rms-collapse": ("src/flow_head.c",
   """    /* Unbiased: torch.var() defaults to correction=1, so divide by n-1. */
    variance /= (float)(n - 1u);""",
   """    variance = 0.0f;
    for (size_t i = 0; i < n; ++i) variance += input[i] * input[i];
    variance /= (float)n;"""),
 "snake-tail": ("src/kernels.c",
   """    for (; t < length; ++t) {
        const float v = row[t];
        const float s = mynah_sin_core(alpha * v);
        row[t] = v + s * s * inverse_alpha;
    }""",
   """    for (; t < length; ++t) {
        row[t] = row[t];
    }"""),
}

if which not in BREAKS:
    sys.exit("unknown break: " + which)
rel, old, new = BREAKS[which]
path = os.path.join(root, rel)
s = open(path).read()
if old not in s:
    sys.exit("break '%s': the text it patches is no longer in %s -- this "
             "script would silently prove nothing" % (which, rel))
open(path, "w").write(s.replace(old, new, 1))
PYEOF
}

fresh_tree() {
    rm -rf "$WORK/tree"
    mkdir -p "$WORK/tree"
    # Everything the build needs, and nothing that carries a stale object.
    ( cd "$ROOT" && tar -cf - Makefile src cli tests server gpu third_party \
        2>/dev/null ) | ( cd "$WORK/tree" && tar -xf - )
    rm -rf "$WORK/tree/build"
}

echo "kernels negative control: nine deliberate defects, each must be caught"
echo

# Sanity first: the unmodified tree must PASS, or every "caught" below would
# be meaningless.
fresh_tree
if ! ( cd "$WORK/tree" && make kernels-test ) >"$WORK/base.log" 2>&1; then
    echo "FAIL: the unmodified tree does not pass kernels-test." >&2
    tail -30 "$WORK/base.log" >&2
    exit 1
fi
echo "  baseline (no break)            PASS, as it must be"

missed=""
for b in $BREAKS; do
    fresh_tree
    if ! patch_break "$b" >"$WORK/patch.log" 2>&1; then
        echo "  $b" >&2
        cat "$WORK/patch.log" >&2
        exit 1
    fi
    if ( cd "$WORK/tree" && make kernels-test ) >"$WORK/$b.log" 2>&1; then
        printf '  %-30s NOT CAUGHT\n' "$b"
        missed="$missed $b"
    else
        why="$(grep -m1 -E '^  FAIL|self_test:' "$WORK/$b.log" | \
               sed 's/^  FAIL [a-z_]*:[0-9]*  //' | cut -c1-72)"
        printf '  %-30s caught: %s\n' "$b" "$why"
    fi
done

echo
if [ -n "$missed" ]; then
    echo "kernels negative control: FAIL -- these breaks slipped through:$missed" >&2
    echo "The suite, not the kernel, is what needs fixing." >&2
    exit 1
fi
echo "kernels negative control: PASS (9 of 9 caught)"
