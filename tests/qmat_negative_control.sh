#!/bin/sh
# Negative control for tests/test_qmat.c and mynah_qmat_self_test().
#
#   sh tests/qmat_negative_control.sh      (or: make qmat-negative-control)
#
# WHY. The determinism gates in src/qmat.c exist because a gate that had only
# ever passed licensed shipping a broken kernel: self_test_i8mm_identity ran at
# 13x37, which is too small a shape for the epilogue's reassociation to show,
# passed, and was read as "SMMLA is bit-identical to SDOT". It was not. So
# before trusting this suite, break the epilogue the ways it has actually been
# broken and require the suite to notice every time.
#
# THE FOUR BREAKS, AND WHAT EACH ONE STANDS IN FOR:
#
#   1  freeze-off      put the epilogue on its no-hardware-FMA branch AND
#                      delete the value barrier, leaving `(float)s * ws * sx`
#                      for -fassociative-math to group per site. This is two
#                      edits on purpose, and the reason is a measurement worth
#                      keeping: removing the barrier ALONE is not a defect,
#                      because fmaf()'s second operand has to be materialised
#                      as a value, so the fma pins the grouping by itself.
#                      Verified by disassembly on this tree, GCC 15.2/aarch64:
#                      with the barrier gone, SIMD=auto and SIMD=portable both
#                      still emit `fmul sx,scale` then `fmadd`. The barrier is
#                      load-bearing only where there is no hardware FMA -- x86
#                      built without -mfma -- and this break compiles that
#                      configuration so the suite is asked the real question.
#   2  contract-split  give dot_q8() back its old shape -- return the product
#                      and let each caller write `value += bias[row]` -- so the
#                      tail rounds twice where the quad epilogue rounds once.
#                      THE SECOND HALF of the same defect, and the one that was
#                      still there after the grouping was fixed: it was worth
#                      exactly 1 ULP at 96x256 under SIMD=portable.
#   3  smmla-row-scale use scales[row] for both rows of the SMMLA tile, i.e.
#                      apply the wrong weight row's scale to the second output.
#                      A gross error, here to prove the shapes reach the kernel
#                      at all.
#   4  smmla-sx-swap   swap the two activation scales in one of the four tile
#                      writes, so out1's second row is scaled by out0's sx.
#                      This is the one a careless 2x2 transcription makes, and
#                      it is invisible to any test that uses a single
#                      activation or two identical ones.
#
# WHICH PROFILE CATCHES WHICH, AND WHY IT IS NOT THE SAME ONE. These breaks are
# defects only where two int8 kernels are compiled for the same arithmetic, and
# which kernels those are is a property of the build:
#
#   SIMD=auto      aarch64 with -march=native: SDOT quad + SMMLA pair.
#   SIMD=portable  aarch64 baseline: NO dotprod, so matvec_q8 is all scalar
#                  tail, against the SMMLA pair -- which is why break 2 shows
#                  up here and can hide under auto, where both sides contract.
#
# So each break is required to be caught under at least one profile, and this
# script says which one caught it. A break caught under NO profile is reported
# by name and fails the script -- that means the SUITE needs fixing, not the
# kernel.
#
# ON A HOST WITH NO SMMLA (x86, or aarch64 built SIMD=scalar) breaks 3 and 4
# patch code that is not compiled and break 1 has only the u8-vs-signed pair to
# show up in. The script detects that from --dispatch-map and SKIPS what it
# cannot prove, by name, rather than reporting a catch it did not make.
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

PROFILES="auto portable"

patch_break() {
    python3 - "$WORK/tree" "$1" <<'PYEOF'
import sys, os
root, which = sys.argv[1], sys.argv[2]

BREAKS = {
 "freeze-off": ("src/qmat.c",
   """static inline float qmat_row_epilogue(int32_t s, float rs, float bias) {
#if defined(__FP_FAST_FMAF)
    return fmaf((float)s, rs, bias);
#else
    float p = (float)s * rs;
    QMAT_FREEZE_F32(p);
    return p + bias;
#endif
}

static float dot_q8""",
   """static inline float qmat_row_epilogue(int32_t s, float rs, float bias) {
    /* break: the no-hardware-FMA branch, with nothing holding the grouping --
     * which is exactly what x86 without -mfma compiles, minus the barrier. */
    return (float)s * rs + bias;
}

static float dot_q8"""),
 "contract-split": ("src/qmat.c",
   """static float dot_q8(const void *qa, float sx, const int8_t *w, float ws,
                    int32_t rowsum, size_t k, int level, float bias) {
    return qmat_row_epilogue(dot_q8_i32(qa, w, rowsum, k, level),
                             qmat_row_scale(ws, sx), bias);
}""",
   """static float dot_q8(const void *qa, float sx, const int8_t *w, float ws,
                    int32_t rowsum, size_t k, int level, float bias) {
    /* break: the product finishes on its own and the bias is a separate
     * rounding, which is the shape this file had before E4-20b. */
    const float p = qmat_row_epilogue(dot_q8_i32(qa, w, rowsum, k, level),
                                      qmat_row_scale(ws, sx), 0.0f);
    return p + bias;
}"""),
 "smmla-row-scale": ("src/qmat.c",
   """        out0[row + 1u] = qmat_row_epilogue(s10, qmat_row_scale(scales[row + 1u], sx0),
                                           bias == NULL ? 0.0f : bias[row + 1u]);""",
   """        out0[row + 1u] = qmat_row_epilogue(s10, qmat_row_scale(scales[row], sx0),
                                           bias == NULL ? 0.0f : bias[row + 1u]);"""),
 "smmla-sx-swap": ("src/qmat.c",
   """        out1[row + 1u] = qmat_row_epilogue(s11, qmat_row_scale(scales[row + 1u], sx1),
                                           bias == NULL ? 0.0f : bias[row + 1u]);""",
   """        out1[row + 1u] = qmat_row_epilogue(s11, qmat_row_scale(scales[row + 1u], sx0),
                                           bias == NULL ? 0.0f : bias[row + 1u]);"""),
}

if which not in BREAKS:
    sys.exit("unknown break: " + which)
rel, old, new = BREAKS[which]
path = os.path.join(root, rel)
s = open(path).read()
if old not in s:
    sys.exit("break '%s': the text it patches is no longer in %s -- this "
             "script would silently prove nothing" % (which, rel))
s = s.replace(old, new, 1)

# freeze-off is two edits, not one: the barrier AND the fmaf that also pins
# the operand grouping. Removing only one of them is not a defect, which is
# the whole finding this break records.
if which == "freeze-off":
    bar = """static inline float qmat_row_scale(float ws, float sx) {
    float rs = ws * sx;
    QMAT_FREEZE_F32(rs);
    return rs;
}"""
    if bar not in s:
        sys.exit("break 'freeze-off': qmat_row_scale is no longer the text "
                 "this script expects -- it would prove nothing")
    s = s.replace(bar, """static inline float qmat_row_scale(float ws, float sx) {
    return ws * sx;   /* break: nothing left holding the grouping */
}""", 1)
open(path, "w").write(s)
PYEOF
}

fresh_tree() {
    rm -rf "$WORK/tree"
    mkdir -p "$WORK/tree"
    ( cd "$ROOT" && tar -cf - Makefile src cli tests server gpu third_party \
        2>/dev/null ) | ( cd "$WORK/tree" && tar -xf - )
    rm -rf "$WORK/tree/build"
}

echo "qmat negative control: four deliberate defects, each must be caught"
echo

# Does this build have a second int8 kernel to disagree with itself? If not,
# say what can and cannot be proven here instead of proving nothing quietly.
fresh_tree
if ! ( cd "$WORK/tree" && make qmat-test ) >"$WORK/base.log" 2>&1; then
    echo "FAIL: the unmodified tree does not pass qmat-test." >&2
    tail -40 "$WORK/base.log" >&2
    exit 1
fi
echo "  baseline (no break)            PASS, as it must be"
if grep -q "smmla wiring  : on" "$WORK/base.log"; then
    HAVE_SMMLA=1
else
    HAVE_SMMLA=0
    echo "  NOTE: this host has no SMMLA wiring. Breaks 3 and 4 patch code that"
    echo "        is not compiled here and are SKIPPED, not passed."
fi
echo

missed=""
skipped=""
for b in freeze-off contract-split smmla-row-scale smmla-sx-swap; do
    case "$b" in
      smmla-*) if [ "$HAVE_SMMLA" -eq 0 ]; then
                   skipped="$skipped $b"
                   printf '  %-18s SKIPPED (no SMMLA kernel on this host)\n' "$b"
                   continue
               fi ;;
    esac
    caught=""
    for p in $PROFILES; do
        fresh_tree
        if ! patch_break "$b" >"$WORK/patch.log" 2>&1; then
            echo "  $b" >&2
            cat "$WORK/patch.log" >&2
            exit 1
        fi
        if ! ( cd "$WORK/tree" && make SIMD="$p" qmat-test ) \
                >"$WORK/$b-$p.log" 2>&1; then
            caught="$caught $p"
        fi
    done
    if [ -n "$caught" ]; then
        printf '  %-18s caught under:%s\n' "$b" "$caught"
    else
        printf '  %-18s NOT CAUGHT under any profile\n' "$b"
        missed="$missed $b"
    fi
done

echo
if [ -n "$missed" ]; then
    echo "qmat negative control: FAIL -- these breaks slipped through:$missed" >&2
    echo "  The suite is what needs fixing, not the kernel." >&2
    exit 1
fi
if [ -n "$skipped" ]; then
    echo "qmat negative control: PASS for what this host can run; SKIPPED:$skipped"
    exit 0
fi
echo "qmat negative control: PASS -- every break was caught"
