#!/bin/sh
# Cross-build mynah-tts for x86_64 on an Apple Silicon Mac and RUN it.
#
# The ISA audit of 2026-09-21 recorded "cross-compilation to x86 is not
# available on this machine (no x86_64 toolchain or SDK; `cc -arch x86_64`
# fails)". That is wrong, and the way it was wrong is worth keeping: `cc` is
# shadowed in the developer's interactive shell, so the probe failed for a
# reason that had nothing to do with the toolchain. /usr/bin/clang cross-compiles
# x86_64 out of the box, the macOS SDK ships both slices, and Rosetta 2 runs the
# result.
#
# WHAT ROSETTA IS, FOR THIS PURPOSE. It emulates an x86-64 CPU with SSE4.2 and
# NO AVX -- so __builtin_cpu_supports("avx2") answers 0. That is not a
# limitation here, it is the test: it is an OLD x86 HOST, the tier we have no
# hardware for and were about to rent one to reach. It proves two things no
# other machine in this project can prove today:
#
#   1. a portable x86 binary RUNS CORRECTLY on a CPU with no AVX2 -- the scalar
#      half of the runtime dispatch, executed, not just compiled;
#   2. an AVX2 binary REFUSES to start there with the ISA guard's message
#      instead of taking a SIGILL. That guard has never been executed on a real
#      no-AVX2 host; every CI runner has AVX2.
#
# What it does NOT prove: anything about the AVX2 kernels executing (Rosetta
# cannot run them) and any performance number at all (it is an emulator). Those
# need real x86 silicon. CI covers the first -- its x86 runners have AVX2 and
# run both halves via MYNAH_KERNELS_X86.
set -eu
cd "$(dirname "$0")/.."

case "$(uname -s)" in
  Darwin) ;;
  *) echo "SKIP x86-cross: macOS only (it needs Rosetta 2 to execute the artifact)"; exit 0 ;;
esac
[ "$(uname -m)" = "arm64" ] || { echo "SKIP x86-cross: already an x86 host, use the native build"; exit 0; }
[ -x /usr/bin/clang ] || { echo "SKIP x86-cross: /usr/bin/clang is missing"; exit 0; }
if ! /usr/bin/clang -arch x86_64 -x c -c /dev/null -o /dev/null 2>/dev/null; then
    echo "SKIP x86-cross: this clang cannot target x86_64"
    exit 0
fi

OUT=build/x86-cross
XCC="/usr/bin/clang -arch x86_64"
rm -rf "$OUT"
mkdir -p "$OUT/ingot"

echo "== ingot, x86_64 =="
for f in third_party/ingot/src/*.c; do
    $XCC -std=c11 -O2 -Ithird_party/ingot/include -c "$f" \
         -o "$OUT/ingot/$(basename "$f" .c).o"
done
ar rcs "$OUT/libingot.a" "$OUT"/ingot/*.o

# Two profiles, because the whole point is that they now differ at RUNTIME and
# not at build time. portable must run here; avx2 must refuse to.
for profile in portable avx2; do
    echo "== mynah-tts, x86_64, SIMD=$profile =="
    make --no-print-directory BUILD_DIR="$OUT/$profile" CC="$XCC" \
         SIMD="$profile" BLAS=none INGOT_LIB="$PWD/$OUT/libingot.a" \
         "$OUT/$profile/mynah-tts" >/dev/null
done

fail=0
say() { printf '  %-5s %s\n' "$1" "$2"; [ "$1" = "ok" ] || fail=1; }

echo "== portable: it must RUN on a host with no AVX2 =="
BIN="$OUT/portable/mynah-tts"
if "$BIN" --self-test >"$OUT/selftest.log" 2>&1; then
    say ok "--self-test passes under Rosetta (no AVX2 anywhere on this host)"
else
    say FAIL "--self-test failed: $(tail -3 "$OUT/selftest.log" | tr '\n' ' ')"
fi

MAP="$OUT/dispatch.txt"
"$BIN" --dispatch-map >"$MAP" 2>&1 || true
grep -q 'arch=x86_64' "$MAP" && say ok "reports arch=x86_64" || say FAIL "not an x86_64 report"
if grep -q 'isa.x86.avx2.*run SCALAR' "$MAP"; then
    say ok "isa.x86.avx2 resolves SCALAR at runtime, and says so"
else
    say FAIL "isa.x86.avx2 row does not state the runtime consequence: $(grep 'isa.x86.avx2' "$MAP" | head -1)"
fi
grep -q 'ISA GUARD: ok' "$MAP" && say ok "the ISA guard accepts this binary here" \
                               || say FAIL "the guard refused a portable binary"

# The env override has to be exercised even where it cannot change the answer:
# a typo in the parsing would otherwise only ever be found on real x86.
if MYNAH_KERNELS_X86=scalar "$BIN" --self-test >/dev/null 2>&1; then
    say ok "MYNAH_KERNELS_X86=scalar is accepted and self-tests clean"
else
    say FAIL "MYNAH_KERNELS_X86=scalar broke the self-test"
fi

echo "== avx2: it must REFUSE to start, with the guard's message =="
set +e
OUTPUT=$("$OUT/avx2/mynah-tts" --version 2>&1); RC=$?
set -e
if [ "$RC" -eq 132 ]; then
    say FAIL "SIGILL -- the guard did not fire before the first AVX2 instruction"
elif [ "$RC" -ne 0 ] && printf '%s' "$OUTPUT" | grep -q 'does not have it'; then
    say ok "refused with the guard's own message, exit $RC (not a SIGILL)"
else
    say FAIL "expected the ISA guard to refuse; got exit $RC: $(printf '%s' "$OUTPUT" | head -1)"
fi

echo
if [ "$fail" -eq 0 ]; then echo "x86-cross: PASS"; else echo "x86-cross: FAIL"; fi
exit "$fail"
