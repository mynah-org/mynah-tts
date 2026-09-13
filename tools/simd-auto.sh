#!/bin/sh
# simd-auto.sh — resolve SIMD=auto against the HOST, and say what it resolved.
#
# PLAN.md E4-15.  Before this script, `SIMD=auto` on Linux x86 appended a fixed
# `-mavx2 -mfma` to CFLAGS without asking the CPU anything: a host without AVX2
# got a binary that dies with an opaque SIGILL (that is E4-12's guard), and a
# host WITH avx512/vnni/bf16 got no acknowledgement that those units exist, so a
# future flag-gated kernel could never be selected in production however well it
# was implemented.
#
# The rule copied from .work/linux-production.md trap 6 is the DOUBLE TEST: a
# feature counts only when the kernel advertises it in /proc/cpuinfo *and* the
# compiler accepts the corresponding flag on a program that uses the intrinsic.
# Either alone is a guess — a new kernel on an old gcc, or a new gcc on an old
# CPU, both produce a binary that does not run.
#
# WHAT IT DOES NOT DO.  It does not pass -mavx512f/-mavx512bw/-mavx512vl even
# when the host has them.  No f32 kernel in src/ dispatches on AVX-512 (see the
# isa.x86.avx512f row in src/dispatch.c), so the only effect would be to widen
# autovectorization — an unmeasured change that also makes the artifact
# non-portable.  `SIMD=avx512` remains the explicit opt-in.  Likewise VNNI: the
# VPDPBUSD kernels in src/qmat.c carry __attribute__((target(...))) and are
# selected by a runtime CPUID probe, so they need no build flag and asking for
# one would imply a kernel choice the flag does not make.  Both are REPORTED,
# which is the part that was missing.
#
# TESTABILITY.  There is no x86 host in this project's fleet, so the resolution
# logic is parameterised on the cpuinfo path (--cpuinfo / $MYNAH_CPUINFO) and on
# the reported machine (--arch), and tests/fixtures/cpuinfo/ holds captured
# /proc/cpuinfo files from real parts.  `make simd-auto-test` runs the table
# against those on any host.  Only the compiler half of the double test needs
# the real toolchain; --no-cc-check reports it as skipped rather than silently
# passing.
#
# Usage:
#   tools/simd-auto.sh [--cpuinfo PATH] [--cc CC] [--arch ARCH] [--os OS]
#                      [--no-cc-check] [--out FILE] [--human]
#
# --out writes a makefile fragment (SIMD_AUTO_FLAGS / _NAME / _DETECTED /
# _REJECTED / _NOTE); with no --out the same fragment goes to stdout.  --human
# prints the resolved profile as a table instead, which is what `make info` and
# `make simd-auto` show.

set -u

cpuinfo="${MYNAH_CPUINFO:-/proc/cpuinfo}"
cc="${CC:-cc}"
arch=""
os=""
out=""
human=0
cc_check=1

while [ $# -gt 0 ]; do
    case "$1" in
        --cpuinfo) cpuinfo="$2"; shift 2 ;;
        --cc)      cc="$2"; shift 2 ;;
        --arch)    arch="$2"; shift 2 ;;
        --os)      os="$2"; shift 2 ;;
        --out)     out="$2"; shift 2 ;;
        --human)   human=1; shift ;;
        --no-cc-check) cc_check=0; shift ;;
        *) echo "simd-auto.sh: unknown argument: $1" >&2; exit 2 ;;
    esac
done

[ -n "$arch" ] || arch="$(uname -m 2>/dev/null || echo unknown)"
[ -n "$os" ]   || os="$(uname -s 2>/dev/null || echo unknown)"

# ---------------------------------------------------------------------------
# half one: what the kernel says this CPU has
#
# /proc/cpuinfo spells the feature set differently on the two architectures —
# "flags" on x86, "Features" on aarch64 — and repeats it per logical CPU.  The
# first block is enough: a heterogeneous ISA across cores would be a far larger
# problem than this script, and Linux does not boot such a machine with
# different feature masks visible per core.
# ---------------------------------------------------------------------------
cpu_flags=""
if [ -r "$cpuinfo" ]; then
    cpu_flags=$(awk -F: '
        /^flags[ \t]*:/    { print $2; exit }
        /^Features[ \t]*:/ { print $2; exit }
    ' "$cpuinfo" 2>/dev/null)
fi
cpu_flags=" $(echo "$cpu_flags" | tr -s ' \t' '  ') "

has_flag() {
    case "$cpu_flags" in
        *" $1 "*) return 0 ;;
        *)        return 1 ;;
    esac
}

# ---------------------------------------------------------------------------
# half two: what this compiler will actually accept
#
# The flag alone is not the test.  gcc accepts -mavx2 while refusing an
# intrinsic its headers do not carry, and a cross-compiler can accept both and
# emit for a different target.  So each probe compiles a program that USES the
# unit, and only -c: linking would drag in a libc we may be cross-building for.
# ---------------------------------------------------------------------------
cc_probe_count=0
cc_ok() {   # cc_ok "<flags>" "<body>"
    [ "$cc_check" -eq 1 ] || return 0
    cc_probe_count=$((cc_probe_count + 1))
    printf '%s\n' "$2" | $cc $1 -x c -c -o /dev/null - >/dev/null 2>&1
}

AVX2_BODY='#include <immintrin.h>
int main(void){ __m256i a=_mm256_set1_epi32(1); a=_mm256_add_epi32(a,a);
  return _mm256_extract_epi32(a,0)!=2; }'
FMA_BODY='#include <immintrin.h>
int main(void){ __m256 a=_mm256_set1_ps(1.0f); a=_mm256_fmadd_ps(a,a,a);
  return _mm256_cvtss_f32(a)!=2.0f; }'
F16C_BODY='#include <immintrin.h>
int main(void){ __m128i h=_mm_set1_epi16(0x3c00); __m128 f=_mm_cvtph_ps(h);
  return _mm_cvtss_f32(f)!=1.0f; }'
NEON_BODY='#include <arm_neon.h>
int main(void){ float32x4_t a=vdupq_n_f32(1.0f); return vaddvq_f32(a)!=4.0f; }'

detected=""       # kernel says yes AND compiler agrees
rejected=""       # one half said no, or we deliberately do not pass the flag
flags=""
note=""

add_detected() { detected="$detected${detected:+ }$1"; }
add_rejected() { rejected="$rejected${rejected:+ }$1"; }
add_flag()     { flags="$flags${flags:+ }$1"; }

case "$arch" in
x86_64|amd64|i386|i686)
    # Kernel-gating flags: a kernel in src/ is compiled out without them.
    if has_flag avx2; then
        if cc_ok "-mavx2" "$AVX2_BODY"; then add_flag -mavx2; add_detected avx2
        else add_rejected "avx2:cc-rejects-intrinsic"; fi
    else
        add_rejected "avx2:absent-in-cpuinfo"
    fi
    if has_flag fma; then
        if cc_ok "-mavx2 -mfma" "$FMA_BODY"; then add_flag -mfma; add_detected fma
        else add_rejected "fma:cc-rejects-intrinsic"; fi
    else
        add_rejected "fma:absent-in-cpuinfo"
    fi
    if has_flag f16c; then
        if cc_ok "-mf16c" "$F16C_BODY"; then add_flag -mf16c; add_detected f16c
        else add_rejected "f16c:cc-rejects-intrinsic"; fi
    else
        add_rejected "f16c:absent-in-cpuinfo"
    fi

    # Reported, deliberately not turned into build flags.  See the header:
    # these are either runtime-dispatched already or have no kernel at all, and
    # a flag that implies neither is how a false ISA claim gets made.
    has_flag avx512f      && add_rejected "avx512f:no-kernel-dispatches-on-it"
    has_flag avx512bw     && add_rejected "avx512bw:no-kernel-dispatches-on-it"
    has_flag avx512vl     && add_rejected "avx512vl:no-kernel-dispatches-on-it"
    has_flag avx512_vnni  && add_detected "avx512_vnni(runtime)"
    has_flag avx_vnni     && add_detected "avx_vnni(runtime)"
    has_flag avx512_bf16  && add_rejected "avx512_bf16:NOT-IMPLEMENTED"
    has_flag amx_int8     && add_rejected "amx_int8:NOT-IMPLEMENTED"
    has_flag amx_bf16     && add_rejected "amx_bf16:NOT-IMPLEMENTED"

    case "$detected" in
        *avx512_vnni*) name="x86-64/avx2+fma+vnni512" ;;
        *avx_vnni*)    name="x86-64/avx2+fma+vnni256" ;;
        *avx2*)        name="x86-64/avx2+fma" ;;
        *)             name="x86-64/baseline" ;;
    esac
    note="resolved for THIS host; use SIMD=avx2 or SIMD=portable for a travelling binary"
    ;;
aarch64|arm64)
    # -march=native is safe here per .work/linux-production.md item 6, and it is
    # what we already did.  What was missing is the report, so the features are
    # enumerated even though the flag does not change.
    add_flag -march=native
    if [ "$os" = "Darwin" ]; then
        name="arm64/native"
        note="macOS: -march=native, features from the compiler not from cpuinfo"
    else
        for f in asimd asimdhp asimddp i8mm bf16 sve sve2 svei8mm svebf16; do
            has_flag "$f" && add_detected "$f"
        done
        # These are advertised by the hardware and used by nothing in src/.
        # The dispatch report says the same thing in its own rows; saying it at
        # build time too is how someone notices before they benchmark.
        has_flag sve       && add_rejected "sve:NOT-IMPLEMENTED"
        has_flag sve2      && add_rejected "sve2:NOT-IMPLEMENTED"
        has_flag svei8mm   && add_rejected "svei8mm:NOT-IMPLEMENTED"
        has_flag svebf16   && add_rejected "svebf16:NOT-IMPLEMENTED"
        has_flag bf16      && add_rejected "bf16:NOT-IMPLEMENTED"
        name="arm64/native"
        note="-march=native pins this artifact to this core; SIMD=neon for a travelling binary"
    fi
    if [ "$cc_check" -eq 1 ] && ! cc_ok "" "$NEON_BODY"; then
        add_rejected "neon:cc-rejects-intrinsic"
    fi
    ;;
*)
    name="$arch/portable"
    note="no ISA profile for this machine; scalar C only"
    ;;
esac

[ -n "$detected" ] || detected="(none)"
[ -n "$rejected" ] || rejected="(none)"
[ "$cc_check" -eq 1 ] || note="$note [cc check SKIPPED]"

if [ "$human" -eq 1 ]; then
    printf 'SIMD=auto resolved\n'
    printf '  host        %s %s (cpuinfo: %s)\n' "$os" "$arch" "$cpuinfo"
    printf '  compiler    %s (%d capability probes)\n' "$cc" "$cc_probe_count"
    printf '  profile     %s\n' "$name"
    printf '  flags       %s\n' "${flags:-(none)}"
    printf '  used        %s\n' "$detected"
    printf '  not used    %s\n' "$rejected"
    printf '  note        %s\n' "$note"
else
    fragment=$(printf 'SIMD_AUTO_FLAGS := %s\nSIMD_AUTO_NAME := %s\nSIMD_AUTO_DETECTED := %s\nSIMD_AUTO_REJECTED := %s\nSIMD_AUTO_NOTE := %s\n' \
        "$flags" "$name" "$detected" "$rejected" "$note")
    if [ -n "$out" ]; then
        mkdir -p "$(dirname "$out")" 2>/dev/null
        printf '%s\n' "$fragment" > "$out"
    else
        printf '%s\n' "$fragment"
    fi
fi
