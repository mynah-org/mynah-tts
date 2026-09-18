#!/bin/sh
# test_simd_auto.sh — the SIMD=auto resolution table, checked without the host.
#
# PLAN.md E4-15.  This project owns one Linux box and it is ARM, so every x86
# claim `tools/simd-auto.sh` makes is otherwise unfalsifiable here.  The
# resolver reads the cpuinfo path it is given, so the table can be exercised
# against captured feature lines from parts we do not have.
#
# --no-cc-check is passed on purpose: the compiler half of the double test can
# only be exercised by a compiler that targets the fixture's architecture, and
# pretending otherwise is exactly the kind of claim this repo keeps catching.
# What is tested here is the /proc/cpuinfo half and the resolution table.

set -u
root=$(cd "$(dirname "$0")/.." && pwd)
resolver="$root/tools/simd-auto.sh"
fixtures="$root/tests/fixtures/cpuinfo"
failures=0
checks=0

resolve() {  # resolve <fixture> <arch> [os]
    "$resolver" --cpuinfo "$fixtures/$1" --arch "$2" --os "${3:-Linux}" --no-cc-check
}

expect() {   # expect <label> <output> <variable> <substring>
    checks=$((checks + 1))
    line=$(printf '%s\n' "$2" | grep "^$3 :=" | sed "s/^$3 := //")
    case "$line" in
        *"$4"*) printf '  ok   %-34s %s contains %s\n' "$1" "$3" "$4" ;;
        *) printf '  FAIL %-34s %s = "%s", expected to contain "%s"\n' \
                  "$1" "$3" "$line" "$4"; failures=$((failures + 1)) ;;
    esac
}

expect_exact() {
    checks=$((checks + 1))
    line=$(printf '%s\n' "$2" | grep "^$3 :=" | sed "s/^$3 := //")
    if [ "$line" = "$4" ]; then
        printf '  ok   %-34s %s = "%s"\n' "$1" "$3" "$4"
    else
        printf '  FAIL %-34s %s = "%s", expected "%s"\n' "$1" "$3" "$line" "$4"
        failures=$((failures + 1))
    fi
}

printf 'SIMD=auto resolution (PLAN.md E4-15)\n'

# ---- Zen 5 EPYC: the x86 production target named in .work/linux-production.md
o=$(resolve x86-zen5-epyc.txt x86_64)
expect_exact "zen5/flags"     "$o" SIMD_AUTO_FLAGS    "-mavx2 -mfma -mf16c"
expect_exact "zen5/profile"   "$o" SIMD_AUTO_NAME     "x86-64/avx2+fma+vnni512"
expect      "zen5/vnni"       "$o" SIMD_AUTO_DETECTED "avx512_vnni(runtime)"
expect      "zen5/avx512-not-a-flag" "$o" SIMD_AUTO_REJECTED "avx512f:no-kernel-dispatches-on-it"
expect      "zen5/bf16-absent"       "$o" SIMD_AUTO_REJECTED "avx512_bf16:NOT-IMPLEMENTED"

# ---- Sapphire Rapids: AMX is detected and deliberately not used.  Their AMX
# 8-core host qualified C2 while the VNNI 32-core did C12; AMX is E4-7, last.
o=$(resolve x86-sapphire-rapids-amx.txt x86_64)
expect_exact "amx/profile"    "$o" SIMD_AUTO_NAME     "x86-64/avx2+fma+vnni512"
expect      "amx/amx-int8"    "$o" SIMD_AUTO_REJECTED "amx_int8:NOT-IMPLEMENTED"
expect      "amx/amx-bf16"    "$o" SIMD_AUTO_REJECTED "amx_bf16:NOT-IMPLEMENTED"
expect      "amx/avx-vnni"    "$o" SIMD_AUTO_DETECTED "avx_vnni(runtime)"

# ---- Haswell: what the old hardcoded "-mavx2 -mfma" assumed every x86 was
o=$(resolve x86-haswell.txt x86_64)
expect_exact "haswell/flags"  "$o" SIMD_AUTO_FLAGS    "-mavx2 -mfma -mf16c"
expect_exact "haswell/profile" "$o" SIMD_AUTO_NAME    "x86-64/avx2+fma"

# ---- Westmere: THE regression.  Before E4-15 this host got -mavx2 -mfma and a
# binary that dies on its first vpaddd with no message.  It must now get none.
o=$(resolve x86-westmere-no-avx2.txt x86_64)
expect_exact "westmere/flags"   "$o" SIMD_AUTO_FLAGS  ""
expect_exact "westmere/profile" "$o" SIMD_AUTO_NAME   "x86-64/baseline"
expect      "westmere/why"      "$o" SIMD_AUTO_REJECTED "avx2:absent-in-cpuinfo"

# ---- Skylake-SP: has AVX-512 but no VNNI.  The profile name must not claim it.
o=$(resolve x86-skylake-avx512-no-vnni.txt x86_64)
expect_exact "skylake/profile" "$o" SIMD_AUTO_NAME    "x86-64/avx2+fma"
expect      "skylake/no-vnni"  "$o" SIMD_AUTO_DETECTED "f16c"

# ---- The box this project actually owns
o=$(resolve arm-neoverse-v2-gcp-c4a.txt aarch64)
expect_exact "neoverse-v2/flags" "$o" SIMD_AUTO_FLAGS "-march=native"
expect      "neoverse-v2/i8mm"   "$o" SIMD_AUTO_DETECTED "i8mm"
expect      "neoverse-v2/sve2"   "$o" SIMD_AUTO_DETECTED "sve2"
expect      "neoverse-v2/sve-unused"     "$o" SIMD_AUTO_REJECTED "sve2:NOT-IMPLEMENTED"
expect      "neoverse-v2/svebf16-unused" "$o" SIMD_AUTO_REJECTED "svebf16:NOT-IMPLEMENTED"

# ---- A machine with no cpuinfo at all: must resolve, not crash
o=$("$resolver" --cpuinfo /nonexistent/cpuinfo --arch x86_64 --os Linux --no-cc-check)
expect_exact "no-cpuinfo/flags"   "$o" SIMD_AUTO_FLAGS ""
expect_exact "no-cpuinfo/profile" "$o" SIMD_AUTO_NAME  "x86-64/baseline"

printf '\n%d checks, %d failures\n' "$checks" "$failures"
[ "$failures" -eq 0 ] || exit 1
printf 'simd-auto: PASS\n'
