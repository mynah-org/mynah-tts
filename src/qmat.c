#include "qmat.h"
#include "dispatch.h"
#include "kernels.h"
#include "threads.h"

#include <pthread.h>

#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if !defined(MYNAH_DISABLE_SIMD) && defined(__ARM_FEATURE_DOTPROD)
#include <arm_neon.h>
#define MYNAH_QMAT_DOTPROD 1
#endif
#if !defined(MYNAH_DISABLE_SIMD) && defined(__AVX2__)
#include <immintrin.h>
#define MYNAH_QMAT_AVX2 1
#endif
/* IEEE half weights.  ARM reaches them through NEON's f16<->f32 converts; the
 * x86 gate is further down, next to the other intrinsics it needs.  Where
 * neither exists the F16 cache refuses the tensor and the caller falls back to
 * exact f32, the same contract INT4 uses for a shape it cannot represent. */
#if !defined(MYNAH_DISABLE_SIMD) && defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>
#define MYNAH_QMAT_F16_NEON 1
#endif

/* ---------------------------------------------------------------- x86 VNNI
 *
 * VPDPBUSD is the only int8 multiply-accumulate x86 has that is not a
 * two-instruction widen-then-madd, and it is the instruction the README once
 * claimed this runtime used.  It exists in two encodings that must BOTH be
 * reachable: EVEX `_mm512_dpbusd_epi32` (AVX512-VNNI) and VEX
 * `_mm256_dpbusd_avx_epi32` (AVX-VNNI).  The 256-bit form is not a fallback
 * for weak machines -- on Zen 4/5 it is full-rate, and on Alder Lake and later
 * client parts it is the ONLY VNNI available because AVX-512 is fused off.
 *
 * The operand constraint is u8 x s8, and the cheap side to flip is the
 * activation: one vector per call, against a whole matrix of weights.  So the
 * activation quantizer writes x+128 directly (no add inside the loop, variant
 * (b) of the two exact choices) and the +128 bias is removed afterwards with
 *
 *     sum_j w[j]*(x[j]+128) - 128 * sum_j w[j]
 *
 * where the row sum is computed once per weight row at quantization time and
 * cached next to the weights.  Every term is an exact int32, so the result is
 * BIT-IDENTICAL to the signed path -- see self_test_u8_identity().
 *
 * The k tail below the 64-element block stays on the signed path, so no
 * partial correction arithmetic is needed: the cached row sum covers exactly
 * the prefix the VNNI loop consumed.
 *
 * DISPATCH.  mynah-tts has no runtime ISA dispatch on x86: every kernel is a
 * compile-time #if and the default Linux build is -mavx2 -mfma, so a VPDPBUSD
 * emitted unconditionally would SIGILL on any pre-Ice-Lake / pre-Alder-Lake
 * host.  Rather than a separate translation unit and new Makefile flags, the
 * two kernels carry __attribute__((target(...))): only those functions are
 * compiled for VNNI, the rest of this file stays at the build's baseline, and
 * qmat_u8_level() (CPUID leaf 7 + XGETBV, not __builtin_cpu_supports, which
 * did not learn "avxvnni" until GCC 11) decides at runtime which one may run.
 */
#if !defined(MYNAH_DISABLE_SIMD) && (defined(__x86_64__) || defined(__i386__)) && \
    (defined(__clang__) || (defined(__GNUC__) && __GNUC__ >= 11))
#include <cpuid.h>
#include <immintrin.h>
#define MYNAH_QMAT_X86_VNNI 1
#endif

/* ----------------------------------------------------------------- x86 F16
 *
 * Until this existed, `MYNAH_QUANT=f16` on x86 was a no-op: the gate above was
 * aarch64-only, mynah_qmat_cache_new() silently rewrote QMAT_F16 to QMAT_F32,
 * and the run went through the exact f32 matvec at f32 speed.  On ARM f16 is
 * measured at 1.96x f32 end to end -- almost exactly the 2.00x the halved
 * weight bytes predict, which is the proof that decode is bound by weight
 * traffic and that the in-loop half->float convert is free.  x86 was paying
 * the full f32 traffic for no stated reason, on the target that is production.
 *
 * VCVTPH2PS (F16C) is the same deal as VPDPBUSD one block up: an instruction
 * the default Linux build (-mavx2 -mfma) does not enable, present on every
 * Intel since Ivy Bridge and every AMD since Bulldozer.  It is reached the same
 * way -- __attribute__((target(...))) on the two functions that need it, the
 * rest of the file at the build's baseline, and CPUID+XGETBV deciding at
 * runtime -- so no Makefile flag changes and no pre-2012 host gets a SIGILL.
 *
 * WEIGHT STORAGE IS uint16_t, NOT A HALF TYPE.  __fp16 is ARM-only and
 * _Float16 on x86 needs GCC >= 12, so the cache holds raw IEEE-754 binary16
 * bit patterns and each kernel reinterprets them.  On aarch64 that is a cast
 * to __fp16*, which is the identical storage, so the ARM path is unchanged
 * bit for bit. */
#if !defined(MYNAH_DISABLE_SIMD) && (defined(__x86_64__) || defined(__i386__)) && \
    (defined(__clang__) || (defined(__GNUC__) && __GNUC__ >= 11))
#define MYNAH_QMAT_F16_X86 1
#endif

#if defined(MYNAH_QMAT_F16_NEON) || defined(MYNAH_QMAT_F16_X86)
#define MYNAH_QMAT_F16 1
#endif

/* How the f16 weights of this process are multiplied.  Three answers, not two:
 * a compiled-but-unsupported host falls to the scalar kernel rather than to
 * f32, because half the weight bytes is worth having even without the
 * instruction -- but that IS a fallback, so it is named. */
enum { QMAT_F16K_OFF = 0, QMAT_F16K_NEON = 1, QMAT_F16K_F16C = 2,
       QMAT_F16K_SCALAR = 3 };

/* count at/below this uses the native int dot; above it falls back to the f32
 * BLAS matmul (the prefill, already fast and kept bit-exact). */
#define QMAT_SMALL_COUNT 16
#define QMAT_K_MAX 8192
#define QMAT_Q4_GROUP 32

/* The VNNI loops consume this many k elements per iteration; the row sums
 * cached for the +128 correction cover exactly `k - k % QMAT_U8_BLOCK`.  Both
 * the 512-bit and the 256-bit kernel use it (the latter as two passes of 32)
 * so one cached prefix sum serves either dispatch. */
#define QMAT_U8_BLOCK 64u

/* How the int8 activation is encoded, and which kernel consumes it.
 *   OFF     signed int8, the SDOT / AVX2 / scalar path (the reference)
 *   SCALAR  unsigned x+128, portable C -- the algebra without the intrinsics,
 *           so the correction can be proven bit-identical on any host
 *   VEX     unsigned x+128, _mm256_dpbusd_avx_epi32
 *   EVEX    unsigned x+128, _mm512_dpbusd_epi32 */
enum {
    QMAT_U8_OFF = 0,
    QMAT_U8_SCALAR = 1,
    QMAT_U8_VEX = 2,
    QMAT_U8_EVEX = 3
};

enum { QMAT_F32 = 0, QMAT_INT8 = 1, QMAT_INT4 = 2, QMAT_F16 = 3 };

/* ------------------------------------------------------- runtime ISA gates
 *
 * Both gates below resolve ONCE and are then immutable for the process: they
 * describe the machine, not a request, so there is no per-request state here
 * and nothing to lock.  Each honours an environment override, which is how a
 * suspected kernel gets bisected without a rebuild -- and the override can
 * only ever narrow what the CPU already supports, never widen it. */

#if defined(MYNAH_QMAT_X86_VNNI)
/* CPUID leaf 7 plus XGETBV, so a feature the OS has not enabled for XSAVE is
 * reported as absent instead of being claimed from a CPUID bit alone. */
static unsigned long long qmat_xcr0(void) {
    unsigned a = 0, b = 0, c = 0, d = 0;
    if (!__get_cpuid(1, &a, &b, &c, &d)) return 0;
    if (!((c >> 27) & 1u)) return 0;            /* OSXSAVE */
    unsigned lo = 0, hi = 0;
    __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    return ((unsigned long long)hi << 32) | lo;
}

/* EVEX and VEX are INDEPENDENT features, not a ladder.  Ice Lake server has
 * AVX512-VNNI and no AVX-VNNI; Alder Lake has AVX-VNNI and no AVX-512; Zen 4/5
 * have both.  They are probed separately so the self-test can exercise every
 * kernel a host can actually run, instead of only the fastest one -- otherwise
 * the VEX kernel would ship untested on exactly the machines that benchmark it. */
static void qmat_x86_probe(int *evex, int *vex) {
    unsigned a = 0, b = 0, c = 0, d = 0;
    *evex = 0;
    *vex = 0;
    const unsigned long long xcr0 = qmat_xcr0();
    const int os_ymm = (xcr0 & 0x6ull) == 0x6ull;
    const int os_zmm = os_ymm && (xcr0 & 0xe0ull) == 0xe0ull;
    if (!__get_cpuid_count(7, 0, &a, &b, &c, &d)) return;
    const int avx2 = (b >> 5) & 1u;
    const int f512 = (b >> 16) & 1u;
    const int bw512 = (b >> 30) & 1u;
    const int vl512 = (b >> 31) & 1u;
    const int vnni512 = (c >> 11) & 1u;
    if (os_zmm && f512 && bw512 && vl512 && vnni512) *evex = 1;
    /* AVX-VNNI is leaf 7 SUBLEAF 1, EAX bit 4 -- a different subleaf from the
     * AVX-512 bits above, which is the usual place this check goes wrong. */
    if (os_ymm && avx2 && __get_cpuid_count(7, 1, &a, &b, &c, &d) &&
        ((a >> 4) & 1u)) {
        *vex = 1;
    }
}
#endif

/* QMAT_U8_VEX when this host can run the 256-bit kernel, whether or not the
 * 512-bit one also resolved; QMAT_U8_OFF otherwise. */
static int qmat_u8_vex_level(void) {
#if defined(MYNAH_QMAT_X86_VNNI)
    static int cached = -1;
    if (cached < 0) {
        int evex = 0, vex = 0;
        qmat_x86_probe(&evex, &vex);
        cached = vex ? QMAT_U8_VEX : QMAT_U8_OFF;
    }
    return cached;
#else
    return QMAT_U8_OFF;
#endif
}

/* MYNAH_QMAT_VNNI: off | scalar | 256 | 512 | auto (default).  A level the CPU
 * cannot run is clamped down, never up; "scalar" forces the portable unsigned
 * kernel on any host, which is what makes the correction algebra testable on a
 * machine with no VNNI at all. */
static int qmat_u8_level_uncached(void) {
    int detected = QMAT_U8_OFF;
#if defined(MYNAH_QMAT_X86_VNNI)
    int evex = 0, vex = 0;
    qmat_x86_probe(&evex, &vex);
    detected = evex ? QMAT_U8_EVEX : (vex ? QMAT_U8_VEX : QMAT_U8_OFF);
#endif
    const char *env = getenv("MYNAH_QMAT_VNNI");
    if (env == NULL || strcmp(env, "auto") == 0) return detected;
    if (strcmp(env, "off") == 0 || strcmp(env, "0") == 0) return QMAT_U8_OFF;
    if (strcmp(env, "scalar") == 0) return QMAT_U8_SCALAR;
    if (strcmp(env, "256") == 0) return detected >= QMAT_U8_VEX ? QMAT_U8_VEX
                                                               : detected;
    if (strcmp(env, "512") == 0) return detected;
    return detected;
}

static int qmat_u8_level(void) {
    static int cached = -1;
    if (cached < 0) cached = qmat_u8_level_uncached();
    return cached;
}

/* ---------------------------------------------------------------- f16 gates
 *
 * Probed and cached once, like the VNNI level above: it describes the machine,
 * not a request.  MYNAH_QMAT_F16C narrows only -- "0"/"off" forces the scalar
 * half kernel on a host that has VCVTPH2PS, which is how the scalar reference
 * gets exercised on the machine that benchmarks the vector one. */
#if defined(MYNAH_QMAT_F16_X86)
__attribute__((target("f16c")))
static void qmat_f16c_touch(void) { }   /* keeps the target attr referenced */

static int qmat_f16c_probe(void) {
    unsigned a = 0, b = 0, c = 0, d = 0;
    const unsigned long long xcr0 = qmat_xcr0();
    if ((xcr0 & 0x6ull) != 0x6ull) return 0;           /* OS has not enabled YMM */
    if (!__get_cpuid(1, &a, &b, &c, &d)) return 0;
    if (!((c >> 29) & 1u)) return 0;                    /* F16C */
    if (!__get_cpuid_count(7, 0, &a, &b, &c, &d)) return 0;
    if (!((b >> 5) & 1u)) return 0;                     /* AVX2 */
    (void)qmat_f16c_touch;
    return 1;
}
#endif

/* Which f16 kernel this process will run.  QMAT_F16K_OFF only when no half
 * weight type is compiled at all -- and then mynah_qmat_cache_new() has
 * already refused QMAT_F16, so the two answers cannot disagree. */
static int qmat_f16_kernel_uncached(void) {
#if defined(MYNAH_QMAT_F16_NEON)
    return QMAT_F16K_NEON;
#elif defined(MYNAH_QMAT_F16_X86)
    const char *env = getenv("MYNAH_QMAT_F16C");
    if (env != NULL && (strcmp(env, "0") == 0 || strcmp(env, "off") == 0))
        return QMAT_F16K_SCALAR;
    return qmat_f16c_probe() ? QMAT_F16K_F16C : QMAT_F16K_SCALAR;
#else
    return QMAT_F16K_OFF;
#endif
}

static int qmat_f16_kernel(void) {
    static int cached = -1;
    if (cached < 0) cached = qmat_f16_kernel_uncached();
    return cached;
}

#if defined(MYNAH_QMAT_F16)
/* ------------------------------------------------------- half <-> float, exact
 *
 * The scalar reference.  It is round-to-nearest-even, which is what both the
 * ARM `(__fp16)` cast and VCVTPS2PH with _MM_FROUND_TO_NEAREST_INT do, so the
 * three producers agree bit for bit -- self_test_f16_convert() asserts exactly
 * that over a sweep that includes the subnormal band, the 65504/65520 rounding
 * cliff and the tie-to-even cases, because "close enough" here would mean the
 * scalar host and the vector host quantize the same checkpoint differently. */
static uint16_t qmat_f16_from_f32(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof bits);
    const uint32_t sign = (bits >> 16) & 0x8000u;
    const uint32_t rest = bits & 0x7fffffffu;
    if (rest >= 0x7f800000u) {                       /* Inf, or NaN kept a NaN */
        return (uint16_t)(sign | (rest > 0x7f800000u ? 0x7e00u : 0x7c00u));
    }
    /* 0x47800000 is 65536.0f.  Everything at or above it is Inf; everything
     * below goes through the normal path, where a carry out of the mantissa
     * walks into the exponent and turns 65520 and up into Inf by itself.
     * Without this branch the shift below wraps and a large finite float comes
     * back as a plausible small half -- which is exactly how this test earned
     * its place: the F16C hardware was right and the reference was not. */
    if (rest >= 0x47800000u) return (uint16_t)(sign | 0x7c00u);
    if (rest < 0x33000000u) return (uint16_t)sign;   /* below 2^-25: zero */
    if (rest < 0x38800000u) {                        /* half subnormal */
        const uint32_t shift = 126u - (rest >> 23);  /* 14 .. 24 */
        const uint32_t mant = (rest & 0x007fffffu) | 0x00800000u;
        uint32_t half = mant >> shift;
        const uint32_t remainder = mant & ((1u << shift) - 1u);
        const uint32_t halfway = 1u << (shift - 1u);
        if (remainder > halfway || (remainder == halfway && (half & 1u))) half += 1u;
        return (uint16_t)(sign | half);
    }
    /* Normal.  A carry out of the mantissa walks into the exponent by itself,
     * which is also how 65520 and above become Inf without a second branch. */
    uint32_t half = (rest - 0x38000000u) >> 13;
    const uint32_t remainder = rest & 0x1fffu;
    if (remainder > 0x1000u || (remainder == 0x1000u && (half & 1u))) half += 1u;
    return (uint16_t)(sign | half);
}

static float qmat_f16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    const uint32_t exponent = ((uint32_t)h >> 10) & 0x1fu;
    uint32_t mantissa = (uint32_t)h & 0x3ffu;
    uint32_t bits;
    if (exponent == 0u) {
        if (mantissa == 0u) {
            bits = sign;
        } else {
            uint32_t e = 127u - 15u + 1u;
            while ((mantissa & 0x400u) == 0u) { mantissa <<= 1; e -= 1u; }
            bits = sign | (e << 23) | ((mantissa & 0x3ffu) << 13);
        }
    } else if (exponent == 31u) {
        bits = sign | 0x7f800000u | (mantissa << 13);
    } else {
        bits = sign | ((exponent + 112u) << 23) | (mantissa << 13);
    }
    float out;
    memcpy(&out, &bits, sizeof out);
    return out;
}

#if defined(MYNAH_QMAT_F16_X86)
__attribute__((target("avx2,f16c")))
static void qmat_f16_pack_f16c(uint16_t *dst, const float *src, size_t n) {
    size_t i = 0;
    for (; i + 8u <= n; i += 8u) {
        _mm_storeu_si128((__m128i *)(void *)(dst + i),
                         _mm256_cvtps_ph(_mm256_loadu_ps(src + i),
                                         _MM_FROUND_TO_NEAREST_INT |
                                         _MM_FROUND_NO_EXC));
    }
    for (; i < n; ++i) dst[i] = qmat_f16_from_f32(src[i]);
}
#endif

/* One pass over the weight at cache-insert time, never in the decode loop. */
static void qmat_f16_pack(uint16_t *dst, const float *src, size_t n) {
#if defined(MYNAH_QMAT_F16_NEON)
    /* Convert into an __fp16 and memcpy the two bytes out.  Writing through an
     * `__fp16 *` aliased onto this `uint16_t` storage is what the code used to
     * do, and the (void *) cast silenced the diagnostic without removing the
     * undefined behaviour: every other access to the cache reads these bytes as
     * uint16_t, so under -fstrict-aliasing -- the default at -O2 and above --
     * the compiler is entitled to assume the two never overlap.
     *
     * It is not theoretical.  On Linux/gcc 15 every non-native ARM profile
     * packed WRONG WEIGHTS: 66088.48, which is above the f16 maximum and must
     * saturate to 0x7c00 (+inf), came out as 0xb01a, a small negative finite
     * number -- a value structurally unrelated to its input rather than a
     * rounding difference.  -march=native happened to hide it, which is why
     * every build anyone runs by hand was fine and the portable profile was
     * not.  memcpy is the standard, always-legal spelling of this store and
     * both compilers fold it to a single 16-bit write. */
    for (size_t i = 0; i < n; ++i) {
        const __fp16 h = (__fp16)src[i];
        memcpy(&dst[i], &h, sizeof dst[i]);
    }
#elif defined(MYNAH_QMAT_F16_X86)
    if (qmat_f16_kernel() == QMAT_F16K_F16C) {
        qmat_f16_pack_f16c(dst, src, n);
        return;
    }
    for (size_t i = 0; i < n; ++i) dst[i] = qmat_f16_from_f32(src[i]);
#else
    for (size_t i = 0; i < n; ++i) dst[i] = qmat_f16_from_f32(src[i]);
#endif
}
#endif /* MYNAH_QMAT_F16 */

/* ARM i8mm (SMMLA).  Unlike SDOT this is a 2x2 outer kernel: it is worth an
 * instruction only when TWO activation vectors share the weight rows, which in
 * this runtime means the weight-stationary batched linear, not the single-row
 * decode matvec.  Claiming it for GEMV would be the same kind of unearned ISA
 * attribution the dispatch report exists to catch. */
#if !defined(MYNAH_DISABLE_SIMD) && defined(__aarch64__) && \
    (defined(__clang__) || (defined(__GNUC__) && __GNUC__ >= 10))
#define MYNAH_QMAT_ARM_I8MM 1
#if defined(__APPLE__)
#include <sys/sysctl.h>
#elif defined(__linux__) && defined(__has_include)
#if __has_include(<sys/auxv.h>)
#include <sys/auxv.h>
#define MYNAH_QMAT_HAVE_AUXV 1
#endif
#endif
#endif

#if defined(MYNAH_QMAT_ARM_I8MM)
static int qmat_i8mm(void) {
    static int cached = -1;
    if (cached < 0) {
        int detected = 0;
#if defined(__APPLE__)
        int value = 0;
        size_t size = sizeof(value);
        if (sysctlbyname("hw.optional.arm.FEAT_I8MM", &value, &size, NULL, 0) == 0)
            detected = value != 0;
#elif defined(MYNAH_QMAT_HAVE_AUXV) && defined(AT_HWCAP2)
        detected = (getauxval(AT_HWCAP2) & (1UL << 13)) != 0;   /* HWCAP2_I8MM */
#endif
        const char *env = getenv("MYNAH_QMAT_I8MM");
        if (env != NULL && strcmp(env, "0") == 0) detected = 0;
        cached = detected;
    }
    return cached;
}

/* Whether the SMMLA kernel is WIRED INTO the batched linear, which is a
 * different question from whether the CPU has the instruction.
 *
 * It is ON by default as of E4-20b, and the history is worth keeping because
 * it is what the gate now exists to prevent.  SMMLA's int32 tile is
 * bit-identical to two SDOT rows -- verified over every row of a 96x256 matrix,
 * 0 of 96 differ -- but the kernel also has to turn that int32 into a float,
 * and the epilogue used to be written out longhand as a three-factor product
 * in each kernel.  Under -ffast-math the compiler grouped those products
 * differently per site, and that made a row's answer depend on WHERE IT SAT IN
 * THE BATCH: one activation row against fixed weights produced four different
 * results -- lead of an SMMLA pair, follower, SDOT tail of an odd batch, alone
 * -- differing in up to 39 of 96 columns by up to 2 ULP.  Under concurrent
 * serving the position is decided by arrival order, so the same request got a
 * different answer depending on who else was in flight.
 *
 * The cause was never in the integer math.  It is removed at the source now:
 * every int8 kernel here ends through qmat_row_scale()/qmat_row_epilogue(),
 * whose value barrier leaves a two-factor product with exactly one grouping,
 * so the kernels agree by construction instead of by luck.  See the long
 * comment above qmat_row_scale().
 *
 * MYNAH_QMAT_I8MM=0 disables the wiring (the kernel stays compiled and
 * self-tested); =1 is the default and is accepted for symmetry.
 * mynah_qmat_i8mm_force() is the in-process override the self-test uses to run
 * the same shapes both ways and require bit-identical output; it is a test
 * hook, not a runtime knob, and the env variable is resolved once so nothing
 * in the decode loop calls getenv(). */
static int qmat_i8mm_override = -1;   /* -1 env/default, 0 off, 1 on */

static int qmat_i8mm_batched(void) {
    if (!qmat_i8mm()) return 0;
    if (qmat_i8mm_override >= 0) return qmat_i8mm_override;
    static int cached = -1;
    if (cached < 0) {
        const char *env = getenv("MYNAH_QMAT_I8MM");
        cached = (env != NULL && strcmp(env, "0") == 0) ? 0 : 1;
    }
    return cached;
}
#endif

/* ------------------------------------------------------------- quantizers */
static void quantize_weight_int8(const float *w, size_t n, size_t k,
                                 int8_t *q, float *scales) {
    for (size_t i = 0; i < n; ++i) {
        const float *row = w + i * k;
        float amax = 0.0f;
        for (size_t j = 0; j < k; ++j) {
            const float a = fabsf(row[j]);
            if (a > amax) amax = a;
        }
        const float scale = amax > 0.0f ? amax / 127.0f : 1.0f;
        scales[i] = scale;
        const float inv = 1.0f / scale;
        int8_t *qrow = q + i * k;
        for (size_t j = 0; j < k; ++j) {
            const float v = row[j] * inv;
            qrow[j] = (int8_t)(v >= 0.0f ? v + 0.5f : v - 0.5f);
        }
    }
}

/* Per-group-of-32 symmetric INT4 (Q4_0 style): nibbles offset by +8, low nibble
 * = even index, high nibble = odd index; scales[i * k/32 + g]. */
/* MYNAH_QMAT_Q4_NAIVE=1 restores the absmax/7 scale the int4 quantizer shipped
 * with, so a quality A/B attributes its result to the quantizer and not to a
 * neighbouring change.  Read once; it cannot change after start. */
static int q4_naive_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *e = getenv("MYNAH_QMAT_Q4_NAIVE");
        cached = (e != NULL && e[0] != '0');
    }
    return cached;
}

/* mode 0 = seed from the signed extreme and then solve the scale (shipped),
 * 1 = the absmax/7 scale this quantizer shipped with, 2 = seeded but NOT
 * solved.  Mode 2 exists only so the self-test can price the two halves
 * separately: without it, disabling the solve still beats absmax on the
 * seeding alone and a gate on the pair cannot see it. */
static void quantize_weight_int4_mode(const float *w, size_t n, size_t k,
                                      uint8_t *q, float *scales, int mode) {
    const size_t groups = k / QMAT_Q4_GROUP;
    const int g_q4_naive = (mode == 1);
    for (size_t i = 0; i < n; ++i) {
        const float *row = w + i * k;
        uint8_t *qrow = q + i * (k / 2u);
        float *srow = scales + i * groups;
        for (size_t g = 0; g < groups; ++g) {
            const float *grp = row + g * QMAT_Q4_GROUP;
            float amax = 0.0f;
            for (size_t j = 0; j < QMAT_Q4_GROUP; ++j) {
                const float a = fabsf(grp[j]);
                if (a > amax) amax = a;
            }
            /* TWO CHANGES TO A ROUND-TO-NEAREST QUANTIZER, both taken from
             * the reference engine's measured version, both free at runtime.
             *
             * 1. SEED FROM THE SIGNED EXTREME, NOT THE MAGNITUDE.  int4 here is
             *    [-8, 7]: eight negative levels and seven positive.  amax/7
             *    throws the -8 away, so a block whose largest element is
             *    negative is quantized with one level less than it has.
             *    Mapping that element onto -8 instead uses the range the
             *    format actually offers.
             *
             * 2. THEN SOLVE THE SCALE, RATHER THAN ASSUME IT.  With the
             *    integers fixed, the scale that minimises the weighted error
             *    sum w_j (v_j - s q_j)^2 has a closed form,
             *    s = sum(w v q) / sum(w q^2), and weighting by w = v^2 asks the
             *    block to be accurate where its energy is, which is what a
             *    downstream matmul is sensitive to.  One pass, no search.
             *
             * Neither changes the format, the layout, the kernels or the number
             * of bytes: same nibbles, same per-group scale, same dequantization.
             * MYNAH_QMAT_Q4_NAIVE=1 restores the absmax scale for an A/B, which
             * is the only way to attribute a quality result to this and not to
             * something else that moved. */
            float scale;
            if (amax <= 0.0f) {
                scale = 1.0f;
            } else {
                float extreme = 0.0f;
                for (size_t j = 0; j < QMAT_Q4_GROUP; ++j) {
                    if (fabsf(grp[j]) == amax) { extreme = grp[j]; break; }
                }
                scale = (extreme < 0.0f) ? (extreme / -8.0f) : (amax / 7.0f);
            }
            if (g_q4_naive) scale = amax > 0.0f ? amax / 7.0f : 1.0f;
            float inv = 1.0f / scale;
            int qv[QMAT_Q4_GROUP];
            for (size_t j = 0; j < QMAT_Q4_GROUP; ++j) {
                const float v = grp[j] * inv;
                int qj = (int)(v >= 0.0f ? v + 0.5f : v - 0.5f);
                if (qj < -8) qj = -8;
                if (qj > 7) qj = 7;
                qv[j] = qj;
            }
            if (mode == 0) {
                double num = 0.0, den = 0.0;
                for (size_t j = 0; j < QMAT_Q4_GROUP; ++j) {
                    const double v = (double)grp[j];
                    const double w = v * v;            /* weight by energy */
                    num += w * v * (double)qv[j];
                    den += w * (double)qv[j] * (double)qv[j];
                }
                /* A degenerate block -- every integer zero, or a solved scale
                 * that is not finite or not positive -- keeps the seed.  A
                 * refinement that cannot be trusted is not applied. */
                if (den > 0.0) {
                    const float refined = (float)(num / den);
                    if (isfinite(refined) && refined > 0.0f) scale = refined;
                }
            }
            srow[g] = scale;
            for (size_t j = 0; j < QMAT_Q4_GROUP; j += 2) {
                const int q0 = qv[j];
                const int q1 = qv[j + 1];
                qrow[(g * QMAT_Q4_GROUP + j) / 2] = (uint8_t)((q0 + 8) | ((q1 + 8) << 4));
            }
            (void)inv;
        }
    }
}

static void quantize_weight_int4(const float *w, size_t n, size_t k,
                                 uint8_t *q, float *scales) {
    quantize_weight_int4_mode(w, n, k, q, scales, q4_naive_enabled() ? 1 : 0);
}

/* Per-vector absmax activation quantization; returns the activation scale.
 *
 * THE SCALAR FORM IS THE REFERENCE and stays compiled: the vector paths below
 * are asserted BYTE-IDENTICAL to it, not close to it, because an activation
 * byte that depended on the ISA would make a request's audio depend on which
 * machine served it.  self_test_act_quantize() does that over both encodings.
 *
 * It is worth vectorising because it is not only the decode loop's pass: the
 * codec conv stack (src/convq8.c) quantizes a whole window of activations per
 * convolution, where this was measured at 40.8% of the int8 path's cost --
 * more than the SDOT it feeds. */
static float quantize_act_int8_scalar(int8_t *qx, const float *x, size_t k) {
    float amax = 0.0f;
    for (size_t i = 0; i < k; ++i) {
        const float a = fabsf(x[i]);
        if (a > amax) amax = a;
    }
    if (amax == 0.0f) {
        memset(qx, 0, k);
        return 0.0f;
    }
    const float inv = 127.0f / amax;
    for (size_t i = 0; i < k; ++i) {
        const float v = x[i] * inv;
        int q = (int)(v >= 0.0f ? v + 0.5f : v - 0.5f);
        if (q > 127) q = 127;
        if (q < -127) q = -127;
        qx[i] = (int8_t)q;
    }
    return amax / 127.0f;
}

/* The absmax, which is where half the scalar time went.  A tree reduction of
 * a max is bit-identical to a sequential one -- max is associative and exact,
 * with no rounding to reassociate -- so this is the one part of the pass that
 * needs no argument beyond that sentence. */
static float qmat_absmax(const float *x, size_t k) {
    size_t i = 0;
    float amax = 0.0f;
#if defined(MYNAH_QMAT_DOTPROD) || defined(MYNAH_QMAT_F16_NEON)
    float32x4_t m0 = vdupq_n_f32(0.0f), m1 = vdupq_n_f32(0.0f);
    for (; i + 8u <= k; i += 8u) {
        m0 = vmaxq_f32(m0, vabsq_f32(vld1q_f32(x + i)));
        m1 = vmaxq_f32(m1, vabsq_f32(vld1q_f32(x + i + 4u)));
    }
    amax = vmaxvq_f32(vmaxq_f32(m0, m1));
#elif defined(MYNAH_QMAT_AVX2)
    __m256 m = _mm256_setzero_ps();
    const __m256 sign = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    for (; i + 8u <= k; i += 8u) {
        m = _mm256_max_ps(m, _mm256_and_ps(_mm256_loadu_ps(x + i), sign));
    }
    float lanes[8];
    _mm256_storeu_ps(lanes, m);
    for (int l = 0; l < 8; ++l) if (lanes[l] > amax) amax = lanes[l];
#endif
    for (; i < k; ++i) {
        const float a = fabsf(x[i]);
        if (a > amax) amax = a;
    }
    return amax;
}

static float quantize_act_int8(int8_t *qx, const float *x, size_t k) {
    const float amax = qmat_absmax(x, k);
    if (amax == 0.0f) {
        memset(qx, 0, k);
        return 0.0f;
    }
    const float inv = 127.0f / amax;
    size_t i = 0;
#if defined(MYNAH_QMAT_DOTPROD) || defined(MYNAH_QMAT_F16_NEON)
    /* vcvtaq_s32_f32 rounds to nearest with TIES AWAY FROM ZERO, which is
     * exactly what `(int)(v >= 0 ? v + 0.5f : v - 0.5f)` spells out; and for
     * |v| < 128 the scalar form's `v + 0.5f` is exact, so the two agree on
     * every input this function can see rather than on almost all of them. */
    const float32x4_t vinv = vdupq_n_f32(inv);
    const int32x4_t lo = vdupq_n_s32(-127), hi = vdupq_n_s32(127);
    for (; i + 16u <= k; i += 16u) {
        int32x4_t a = vcvtaq_s32_f32(vmulq_f32(vld1q_f32(x + i), vinv));
        int32x4_t b2 = vcvtaq_s32_f32(vmulq_f32(vld1q_f32(x + i + 4u), vinv));
        int32x4_t c = vcvtaq_s32_f32(vmulq_f32(vld1q_f32(x + i + 8u), vinv));
        int32x4_t d = vcvtaq_s32_f32(vmulq_f32(vld1q_f32(x + i + 12u), vinv));
        a = vminq_s32(vmaxq_s32(a, lo), hi);
        b2 = vminq_s32(vmaxq_s32(b2, lo), hi);
        c = vminq_s32(vmaxq_s32(c, lo), hi);
        d = vminq_s32(vmaxq_s32(d, lo), hi);
        const int16x8_t p0 = vcombine_s16(vmovn_s32(a), vmovn_s32(b2));
        const int16x8_t p1 = vcombine_s16(vmovn_s32(c), vmovn_s32(d));
        vst1q_s8(qx + i, vcombine_s8(vmovn_s16(p0), vmovn_s16(p1)));
    }
#endif
    for (; i < k; ++i) {
        const float v = x[i] * inv;
        int q = (int)(v >= 0.0f ? v + 0.5f : v - 0.5f);
        if (q > 127) q = 127;
        if (q < -127) q = -127;
        qx[i] = (int8_t)q;
    }
    return amax / 127.0f;
}

/* The same quantizer with the +128 already applied, so the VNNI loop never
 * pays for the bias.  It must agree with quantize_act_int8 element for element
 * -- qu[i] == (uint8_t)(qx[i] + 128) -- which is why the rounding and clamping
 * are written here identically rather than shared through a callback. */
static float quantize_act_u8_scalar(uint8_t *qu, const float *x, size_t k) {
    float amax = 0.0f;
    for (size_t i = 0; i < k; ++i) {
        const float a = fabsf(x[i]);
        if (a > amax) amax = a;
    }
    if (amax == 0.0f) {
        memset(qu, 128, k);
        return 0.0f;
    }
    const float inv = 127.0f / amax;
    for (size_t i = 0; i < k; ++i) {
        const float v = x[i] * inv;
        int q = (int)(v >= 0.0f ? v + 0.5f : v - 0.5f);
        if (q > 127) q = 127;
        if (q < -127) q = -127;
        qu[i] = (uint8_t)(q + 128);
    }
    return amax / 127.0f;
}

/* The unsigned twin.  It shares the absmax and then adds 128, which on ARM is
 * one instruction on the narrowed bytes; the invariant it must keep is
 * qu[i] == (uint8_t)(qx[i] + 128) for every i, and the self-test checks that
 * against the scalar form rather than against its sibling. */
static float quantize_act_u8(uint8_t *qu, const float *x, size_t k) {
    const float amax = qmat_absmax(x, k);
    if (amax == 0.0f) {
        memset(qu, 128, k);
        return 0.0f;
    }
    const float inv = 127.0f / amax;
    size_t i = 0;
#if defined(MYNAH_QMAT_DOTPROD) || defined(MYNAH_QMAT_F16_NEON)
    const float32x4_t vinv = vdupq_n_f32(inv);
    const int32x4_t lo = vdupq_n_s32(-127), hi = vdupq_n_s32(127);
    const uint8x16_t bias = vdupq_n_u8(128u);
    for (; i + 16u <= k; i += 16u) {
        int32x4_t a = vcvtaq_s32_f32(vmulq_f32(vld1q_f32(x + i), vinv));
        int32x4_t b2 = vcvtaq_s32_f32(vmulq_f32(vld1q_f32(x + i + 4u), vinv));
        int32x4_t c = vcvtaq_s32_f32(vmulq_f32(vld1q_f32(x + i + 8u), vinv));
        int32x4_t d = vcvtaq_s32_f32(vmulq_f32(vld1q_f32(x + i + 12u), vinv));
        a = vminq_s32(vmaxq_s32(a, lo), hi);
        b2 = vminq_s32(vmaxq_s32(b2, lo), hi);
        c = vminq_s32(vmaxq_s32(c, lo), hi);
        d = vminq_s32(vmaxq_s32(d, lo), hi);
        const int16x8_t p0 = vcombine_s16(vmovn_s32(a), vmovn_s32(b2));
        const int16x8_t p1 = vcombine_s16(vmovn_s32(c), vmovn_s32(d));
        const int8x16_t packed = vcombine_s8(vmovn_s16(p0), vmovn_s16(p1));
        vst1q_u8(qu + i, vaddq_u8(vreinterpretq_u8_s8(packed), bias));
    }
#endif
    for (; i < k; ++i) {
        const float v = x[i] * inv;
        int q = (int)(v >= 0.0f ? v + 0.5f : v - 0.5f);
        if (q > 127) q = 127;
        if (q < -127) q = -127;
        qu[i] = (uint8_t)(q + 128);
    }
    return amax / 127.0f;
}

/* Picks the encoding the resolved kernel wants.  `q` is uint8_t* above OFF and
 * int8_t* at OFF; both are character types, so the cast aliases legally. */
static float quantize_act(void *q, const float *x, size_t k, int level) {
    return level == QMAT_U8_OFF ? quantize_act_int8((int8_t *)q, x, k)
                                : quantize_act_u8((uint8_t *)q, x, k);
}

/* Number of k elements the VNNI loop covers; the remainder runs signed. */
static size_t qmat_u8_main(size_t k) { return k - (k % QMAT_U8_BLOCK); }

/* sum of w[row][0 .. qmat_u8_main(k)) for every row, in int32.  |w| <= 127 and
 * k <= QMAT_K_MAX, so the widest possible sum is 127 * 8192 ~ 1.04e6: no
 * overflow, and no need for a wider accumulator. */
static void weight_rowsum_prefix(const int8_t *q, size_t n, size_t k,
                                 int32_t *rowsum) {
    const size_t main = qmat_u8_main(k);
    for (size_t i = 0; i < n; ++i) {
        const int8_t *row = q + i * k;
        int32_t sum = 0;
        for (size_t j = 0; j < main; ++j) sum += (int32_t)row[j];
        rowsum[i] = sum;
    }
}

/* ---------------------------------------------------------- int dot kernels */

/* ---- x86 VNNI: exact int32, u8 activation x s8 weight --------------------
 *
 * VPDPBUSD multiplies four u8 x s8 pairs into int16 intermediates and adds all
 * four to the int32 accumulator.  The widest intermediate here is 255 * 127 =
 * 32385, inside int16, and the widest accumulator is 8192 * 32385 ~ 2.65e8,
 * inside int32 -- so the instruction is exact for this operand range and the
 * saturating VPDPBUSDS variant is not needed. */
#if defined(MYNAH_QMAT_X86_VNNI)
__attribute__((target("avx512f,avx512bw,avx512vl,avx512vnni")))
static void dot4_u8_evex(const uint8_t *xu, const int8_t *w0, const int8_t *w1,
                         const int8_t *w2, const int8_t *w3, size_t main,
                         int32_t out[4]) {
    __m512i a0 = _mm512_setzero_si512();
    __m512i a1 = _mm512_setzero_si512();
    __m512i a2 = _mm512_setzero_si512();
    __m512i a3 = _mm512_setzero_si512();
    for (size_t j = 0; j < main; j += 64u) {
        const __m512i x = _mm512_loadu_si512((const void *)(xu + j));
        a0 = _mm512_dpbusd_epi32(a0, x, _mm512_loadu_si512((const void *)(w0 + j)));
        a1 = _mm512_dpbusd_epi32(a1, x, _mm512_loadu_si512((const void *)(w1 + j)));
        a2 = _mm512_dpbusd_epi32(a2, x, _mm512_loadu_si512((const void *)(w2 + j)));
        a3 = _mm512_dpbusd_epi32(a3, x, _mm512_loadu_si512((const void *)(w3 + j)));
    }
    out[0] = _mm512_reduce_add_epi32(a0);
    out[1] = _mm512_reduce_add_epi32(a1);
    out[2] = _mm512_reduce_add_epi32(a2);
    out[3] = _mm512_reduce_add_epi32(a3);
}

__attribute__((target("avx512f,avx512bw,avx512vl,avx512vnni")))
static int32_t dot1_u8_evex(const uint8_t *xu, const int8_t *w, size_t main) {
    __m512i acc = _mm512_setzero_si512();
    for (size_t j = 0; j < main; j += 64u) {
        acc = _mm512_dpbusd_epi32(acc, _mm512_loadu_si512((const void *)(xu + j)),
                                  _mm512_loadu_si512((const void *)(w + j)));
    }
    return _mm512_reduce_add_epi32(acc);
}

__attribute__((target("avx2,avxvnni")))
static int32_t hsum256(__m256i v) {
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(v),
                              _mm256_extracti128_si256(v, 1));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4e));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xb1));
    return _mm_cvtsi128_si32(s);
}

__attribute__((target("avx2,avxvnni")))
static void dot4_u8_vex(const uint8_t *xu, const int8_t *w0, const int8_t *w1,
                        const int8_t *w2, const int8_t *w3, size_t main,
                        int32_t out[4]) {
    __m256i a0 = _mm256_setzero_si256();
    __m256i a1 = _mm256_setzero_si256();
    __m256i a2 = _mm256_setzero_si256();
    __m256i a3 = _mm256_setzero_si256();
    for (size_t j = 0; j < main; j += 32u) {
        const __m256i x = _mm256_loadu_si256((const __m256i *)(xu + j));
        a0 = _mm256_dpbusd_avx_epi32(a0, x, _mm256_loadu_si256((const __m256i *)(w0 + j)));
        a1 = _mm256_dpbusd_avx_epi32(a1, x, _mm256_loadu_si256((const __m256i *)(w1 + j)));
        a2 = _mm256_dpbusd_avx_epi32(a2, x, _mm256_loadu_si256((const __m256i *)(w2 + j)));
        a3 = _mm256_dpbusd_avx_epi32(a3, x, _mm256_loadu_si256((const __m256i *)(w3 + j)));
    }
    out[0] = hsum256(a0);
    out[1] = hsum256(a1);
    out[2] = hsum256(a2);
    out[3] = hsum256(a3);
}

__attribute__((target("avx2,avxvnni")))
static int32_t dot1_u8_vex(const uint8_t *xu, const int8_t *w, size_t main) {
    __m256i acc = _mm256_setzero_si256();
    for (size_t j = 0; j < main; j += 32u) {
        acc = _mm256_dpbusd_avx_epi32(acc,
                                      _mm256_loadu_si256((const __m256i *)(xu + j)),
                                      _mm256_loadu_si256((const __m256i *)(w + j)));
    }
    return hsum256(acc);
}
#endif /* MYNAH_QMAT_X86_VNNI */

/* The same arithmetic with no intrinsic at all.  It is not a performance path:
 * it is the executable statement of what VPDPBUSD computes, so the +128
 * correction, the cached row sum and the tail split can be proven
 * bit-identical to the signed path on a host that has no VNNI unit. */
static int32_t dot1_u8_scalar(const uint8_t *xu, const int8_t *w, size_t main) {
    int32_t s = 0;
    for (size_t j = 0; j < main; ++j) s += (int32_t)xu[j] * (int32_t)w[j];
    return s;
}

/* One row, unsigned activation: VNNI prefix + signed tail + the -128*rowsum
 * correction.  Returns the exact int32 the signed kernel would have produced. */
static int32_t dot_u8_i32(const uint8_t *xu, const int8_t *w, int32_t rowsum,
                          size_t k, int level) {
    const size_t main = qmat_u8_main(k);
    int32_t s;
    switch (level) {
#if defined(MYNAH_QMAT_X86_VNNI)
    case QMAT_U8_EVEX: s = dot1_u8_evex(xu, w, main); break;
    case QMAT_U8_VEX:  s = dot1_u8_vex(xu, w, main);  break;
#endif
    default:           s = dot1_u8_scalar(xu, w, main); break;
    }
    s -= 128 * rowsum;
    /* The tail never entered the unsigned domain, so it needs no correction:
     * recover the signed activation from the same buffer. */
    for (size_t j = main; j < k; ++j)
        s += ((int32_t)xu[j] - 128) * (int32_t)w[j];
    return s;
}

/* Four rows, unsigned activation: one activation load feeds four weight rows,
 * mirroring the SDOT unroll below. */
static void dot4_u8_i32(const uint8_t *xu, const int8_t *w, size_t cols,
                        const int32_t *rowsum, size_t row, int level,
                        int32_t out[4]) {
    const size_t main = qmat_u8_main(cols);
    const int8_t *w0 = w + row * cols;
    const int8_t *w1 = w0 + cols;
    const int8_t *w2 = w1 + cols;
    const int8_t *w3 = w2 + cols;
    switch (level) {
#if defined(MYNAH_QMAT_X86_VNNI)
    case QMAT_U8_EVEX: dot4_u8_evex(xu, w0, w1, w2, w3, main, out); break;
    case QMAT_U8_VEX:  dot4_u8_vex(xu, w0, w1, w2, w3, main, out);  break;
#endif
    default:
        out[0] = dot1_u8_scalar(xu, w0, main);
        out[1] = dot1_u8_scalar(xu, w1, main);
        out[2] = dot1_u8_scalar(xu, w2, main);
        out[3] = dot1_u8_scalar(xu, w3, main);
        break;
    }
    for (int r = 0; r < 4; ++r) {
        const int8_t *wr = w + (row + (size_t)r) * cols;
        out[r] -= 128 * rowsum[row + (size_t)r];
        for (size_t j = main; j < cols; ++j)
            out[r] += ((int32_t)xu[j] - 128) * (int32_t)wr[j];
    }
}
#if defined(MYNAH_QMAT_AVX2)
static int32_t dot_q8_i32_avx2(const int8_t *qx, const int8_t *w, size_t k) {
    __m256i acc = _mm256_setzero_si256();
    size_t j = 0;
    for (; j + 16u <= k; j += 16u) {
        const __m128i x0 = _mm_loadu_si128((const __m128i *)(qx + j));
        const __m128i w0 = _mm_loadu_si128((const __m128i *)(w + j));
        const __m256i x16_0 = _mm256_cvtepi8_epi16(x0);
        const __m256i w16_0 = _mm256_cvtepi8_epi16(w0);
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(x16_0, w16_0));
    }
    __m128i sum = _mm_add_epi32(_mm256_castsi256_si128(acc),
                                _mm256_extracti128_si256(acc, 1));
    sum = _mm_hadd_epi32(sum, sum);
    sum = _mm_hadd_epi32(sum, sum);
    int32_t result = _mm_cvtsi128_si32(sum);
    for (; j < k; ++j) result += (int32_t)qx[j] * (int32_t)w[j];
    return result;
}
#endif
/* `qa` is int8_t* at QMAT_U8_OFF and uint8_t* (x+128) above it; `rowsum` is
 * this row's cached prefix sum and is read only in the unsigned case. */
static int32_t dot_q8_i32(const void *qa, const int8_t *w, int32_t rowsum,
                          size_t k, int level) {
    if (level != QMAT_U8_OFF)
        return dot_u8_i32((const uint8_t *)qa, w, rowsum, k, level);
    const int8_t *qx = (const int8_t *)qa;
#if defined(MYNAH_QMAT_DOTPROD)
    int32x4_t acc = vdupq_n_s32(0);
    size_t j = 0;
    for (; j + 16 <= k; j += 16) acc = vdotq_s32(acc, vld1q_s8(w + j), vld1q_s8(qx + j));
    int32_t s = vaddvq_s32(acc);
    for (; j < k; ++j) s += (int32_t)w[j] * (int32_t)qx[j];
    return s;
#elif defined(MYNAH_QMAT_AVX2)
    return dot_q8_i32_avx2(qx, w, k);
#else
    int32_t s = 0;
    for (size_t j = 0; j < k; ++j) s += (int32_t)w[j] * (int32_t)qx[j];
    return s;
#endif
}

/* ------------------------------------------------ THE INT8 FLOAT EPILOGUE
 *
 * Every int8 kernel in this file ends the same way: an exact int32 dot product
 * becomes a float.  Written the obvious way that is
 *
 *     out[row] = (float)s * scales[row] * sx + bias[row];
 *
 * a THREE-factor product, and -ffast-math (-fassociative-math, see the
 * Makefile) lets the compiler pick any of its three groupings:
 *
 *     ((float)s * scales[row]) * sx
 *     (float)s * (scales[row] * sx)
 *     ((float)s * sx) * scales[row]
 *
 * They do not round alike, and the compiler chooses PER SITE -- per textual
 * copy of the epilogue, and also per INLINED COPY of the same textual one.
 * Measured at 10f8f48 on this tree with GCC 15.2/aarch64, by disassembling
 * src/qmat.c: under SIMD=auto (-march=native) matvec_q8's out-of-line copy
 * emits `fmul s31, s0, s31` (sx * scale) then `fmul s31, s31, s27` -- the
 * middle grouping -- while under SIMD=portable the SAME SOURCE emits
 * `fmul s31, s31, s0` (s * sx) then `fmul s31, s31, s24` -- the last one.
 * Two profiles of one compiler, one source line, two different answers.
 *
 * That is not a cosmetic ULP.  It is what made a row's answer depend on WHERE
 * IT SAT IN THE BATCH: matvec_q8_pair_i8mm (SMMLA) and matvec_q8 (SDOT) are
 * two textual copies, and the weight-stationary batched linear routes a row
 * through one or the other according to the batch's size and the row's
 * position in it -- which under concurrent serving is decided by arrival
 * order.  One activation row against fixed weights produced four different
 * results: lead of an SMMLA pair, follower, SDOT tail of an odd batch, alone.
 *
 * E4-20 tried the obvious repair and it failed: hoisting the epilogue into a
 * `static inline` helper changes nothing, because the compiler inlines the
 * helper and then reassociates the resulting expression per site exactly as
 * before.  A source-level hoist is a HINT.  The optimizer works on the whole
 * expression DAG, not on statement boundaries, so `float rs = ws * sx;` is
 * substituted straight back into `(float)s * (ws * sx)` and regrouped.
 *
 * So the hoist has to be enforced, not suggested, and that is what this is.
 * The empty `__asm__` with `rs` as an in-out operand is a value barrier: it
 * emits NO instruction, but after it the compiler no longer knows that `rs`
 * came from `ws * sx`, so `(float)s * rs` is a two-factor product with exactly
 * one grouping and nothing left to reassociate.  Every int8 kernel here --
 * SDOT quad, AVX2 quad, u8/VNNI quad, the scalar tails, and the SMMLA pair --
 * goes through this one function, so they agree BY CONSTRUCTION rather than by
 * the compiler happening to choose alike.
 *
 * HOW MUCH OF THE DEFECT THE GROUPING ACTUALLY WAS: less than it looks, and
 * the measurement is worth keeping because the obvious reading of the
 * disassembly overstates it.  The two profiles above really do group
 * differently -- but they are two different BINARIES, and a row's answer
 * cannot depend on a build it was not built by.  Within one binary GCC 15.2
 * was consistent, and tests/qmat_negative_control.sh proves it: removing this
 * barrier on its own is NOT caught by any gate here, because on a target with
 * a hardware FMA the epilogue below is written with fmaf(), whose second
 * operand must be materialised as a value -- so the fma pins `ws * sx` as a
 * unit all by itself.  The barrier is what holds the grouping on the OTHER
 * branch, where there is no fma to do it: x86 built without -mfma, which is
 * SIMD=portable on the production target.  Both halves are needed and neither
 * is decoration; the negative control breaks them together for that reason.
 *
 * The canonical grouping is therefore `(float)s * (scales[row] * sx)`, and it
 * was not chosen for elegance: it is the one SIMD=auto -- the production
 * profile, -march=native on the Neoverse-V2 -- already emitted for matvec_q8
 * before this change, so the default output does not move and only the other
 * kernels come to meet it.  Verified: the Magpie int8 golden (offline-int8 in
 * tests/goldens/fake-magpie.sha256) is byte-identical before and after, under
 * SIMD=auto, SIMD=portable and SIMD=scalar.
 *
 * THE SECOND FREEDOM, AND IT IS THE ONE THAT ACTUALLY BIT.  Fixing the
 * grouping is only half of it, and measured here, the smaller half.  `(float)s * rs + bias` may also CONTRACT into a single-rounding
 * FMA, and `-ffp-contract=fast` is the default, so whether a site rounds once
 * or twice depends on whether the product and the add ended up adjacent in that
 * site's expression tree.  They did not, everywhere: `dot_q8()` used to return
 * the product and let each caller write `value += bias[row]` itself, which is
 * two roundings, while the quad epilogues wrote `... * sx + bias[row]`, which
 * is one.  Measured here at 10f8f48 + the grouping fix, GCC 15.2/aarch64: with
 * the groupings already identical, SIMD=portable still failed
 * self_test_i8mm_identity by exactly 1 ULP at 96x256, and rebuilding the very
 * same source with -ffp-contract=off turned every profile green.  That is the
 * whole of the residual, and it is why this function takes the bias: the
 * epilogue is ONE expression with one rounding decision, not a product that
 * each caller finishes in its own way.
 *
 * So the contraction is settled by construction too, rather than by hoping the
 * optimizer contracts alike at every site.  Where the target has a hardware FMA
 * (__FP_FAST_FMAF: aarch64 always, x86 with -mfma) fmaf() IS the contraction,
 * named explicitly, which is also what SIMD=auto already emitted for the quad
 * epilogue before this change -- so the production numerics do not move.  Where
 * it does not, the product is frozen before the add so nothing can fuse it
 * later.  Either way every int8 kernel in this file rounds the same number of
 * times in the same places. */
/* A value barrier on a float: after it the compiler knows nothing about where
 * the value came from, so it can neither regroup the product that produced it
 * nor fuse it into what consumes it.  It emits no instruction; the constraint
 * just names a register class the value is already in. */
#if defined(__GNUC__) && defined(__aarch64__)
#define QMAT_FREEZE_F32(v) __asm__("" : "+w"(v))
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#define QMAT_FREEZE_F32(v) __asm__("" : "+x"(v))
#elif defined(__GNUC__)
#define QMAT_FREEZE_F32(v) __asm__("" : "+m"(v))
#else
#define QMAT_FREEZE_F32(v) do { volatile float qmat_frz_ = (v); (v) = qmat_frz_; } while (0)
#endif

static inline float qmat_row_scale(float ws, float sx) {
    float rs = ws * sx;
    QMAT_FREEZE_F32(rs);
    return rs;
}

/* The epilogue itself, so the one expression shape has one name.  `rs` must
 * come from qmat_row_scale(); passing `ws * sx` directly would hand the
 * optimizer the three-factor product back.  `bias` is a value, not a pointer:
 * a caller with no bias passes 0.0f, which keeps the rounding count the same
 * for biased and unbiased rows (fmaf(a, b, 0) is a*b rounded once, and the
 * frozen fallback's `p + 0.0f` is exact). */
static inline float qmat_row_epilogue(int32_t s, float rs, float bias) {
#if defined(__FP_FAST_FMAF)
    return fmaf((float)s, rs, bias);
#else
    float p = (float)s * rs;
    QMAT_FREEZE_F32(p);
    return p + bias;
#endif
}

static float dot_q8(const void *qa, float sx, const int8_t *w, float ws,
                    int32_t rowsum, size_t k, int level, float bias) {
    return qmat_row_epilogue(dot_q8_i32(qa, w, rowsum, k, level),
                             qmat_row_scale(ws, sx), bias);
}

/* The int4 group accumulation, and it is the int8 epilogue's problem again in
 * a different costume.  `acc += (float)gi * scales[g]` is a multiply-add whose
 * rounding count -- fused or not -- the compiler decides PER SITE, and int4 has
 * four sites: dot_q4's three branches and matvec_q4's quad macro.  Nothing
 * caught it, because int4's existing gates compare against an f32 dot with a
 * relative tolerance and a 1 ULP accumulation difference disappears into it;
 * self_test_q4_identity() found it the moment it asserted equality instead.
 *
 * It is the same hazard for the same reason: matvec_q4 sends rows 0..3 through
 * the quad macro and the remainder through dot_q4, so two spellings of one sum
 * decide a row's answer.  Today that is position-INdependent, because a row's
 * index does not change with the batch -- but "it happens not to be reachable"
 * is what was said about the SMMLA pair too.  One helper, one shape, one
 * rounding decision, and self_test_q4_identity() asserts it rather than
 * assuming it. */
static inline float qmat_q4_accum(float acc, int32_t gi, float scale) {
#if defined(__FP_FAST_FMAF)
    return fmaf((float)gi, scale, acc);
#else
    float p = (float)gi * scale;
    QMAT_FREEZE_F32(p);
    return acc + p;
#endif
}

/* And the row's last step, `acc * sx + bias`, for the same reason and with the
 * same answer.  The bias is a value here rather than something each caller
 * adds afterwards, because `acc * sx` followed by a separate `+= bias[row]`
 * rounds twice where the inlined `acc * sx + bias[row]` rounds once -- which
 * is precisely the asymmetry that made dot_q8 and the quad epilogues disagree
 * by 1 ULP before E4-20b. */
static inline float qmat_q4_finish(float acc, float sx, float bias) {
#if defined(__FP_FAST_FMAF)
    return fmaf(acc, sx, bias);
#else
    float p = acc * sx;
    QMAT_FREEZE_F32(p);
    return p + bias;
#endif
}

/* qx is the int8 activation; q is the packed INT4 weight group row; scales has
 * one entry per group of 32.  k must be a multiple of 32. */
/* ------------------------------------------------------- INT4 on x86 (E4-21)
 *
 * Until this existed, int4 on the x86 production target had NO VECTOR PATH AT
 * ALL: `dot_q4` had a NEON SDOT branch and, for everything else, a scalar
 * nibble loop -- while int8 next door had AVX2 madd, AVX-VNNI and AVX-512
 * VNNI.  int4 therefore carried half the weight bytes and computed them one
 * nibble at a time, and would lose to int8 on every shape.  It also reframes
 * the one number this repo had recorded: "int4 buys only 3% over int8 on the
 * codec" was taken on ARM, where int4 at least has SDOT.
 *
 * WHY AVX2 AND NOT VNNI, WHICH IS THE INTERESTING PART.  int8 carries ONE
 * SCALE PER ROW, so its VNNI loop accumulates int32 across a 64-element block
 * -- QMAT_U8_BLOCK -- and in fact across the whole row, touching a float once
 * at the end: about 2 instructions per 64 MACs.  int4 carries ONE SCALE PER
 * GROUP OF 32, so the int32 accumulator MUST be flushed to float every 32
 * elements.  That flush -- a horizontal reduce plus a convert, a multiply and
 * an add -- is about 6 of the ~18 instructions this kernel spends per group,
 * and the nibble unpack is another 6.  The dot itself is 3.
 *
 * So VPDPBUSD, which would replace those 3 with 2, moves the whole kernel by
 * roughly a tenth.  The 32-element group really is too short for VNNI to be
 * the story here, and the prize is simply HAVING a vector path: ~18
 * instructions per 32 MACs against the scalar loop's ~140.  An int4 VNNI
 * variant is a small, well-understood delta on top of this one and is
 * deliberately NOT written yet -- it should be measured on the machine that
 * would benefit, not guessed at from here.
 *
 * THE ALGEBRA, AND THE CORRECTION THAT IS NOT A ROW SUM.  Q4_0 nibbles are
 * stored offset by +8, so the weight is `n - 8` for a nibble n in [0, 15].
 * _mm256_maddubs_epi16 wants its FIRST operand unsigned, and the nibble
 * already is, so the weights go in unsigned and the activation stays signed:
 *
 *     sum (n - 8) * x  ==  sum n * x  -  8 * sum x
 *
 * and BOTH terms are computed with maddubs against the same activation, so
 * the correction is subtracted in int16 before the widening and there is no
 * precomputed sum of anything.  That is on purpose: the int8 path's +128
 * correction needs a cached per-row prefix sum, and the mistake its own
 * self-test exists to catch is taking that sum over the wrong extent.  Here
 * the equivalent term is per group AND depends only on the activation, so
 * rather than add a second table with a second extent to get wrong, it is
 * recomputed in two instructions from the operand that defines it.
 *
 * No saturation: n in [0,15] and x in [-128,127] give pairwise sums in
 * [-3840, 3810]; the correction term is in [-2048, 2032]; their difference is
 * in [-5888, 5858], all far inside int16.  self_test_q4_x86_identity() asserts
 * the bound as well as the result.
 *
 * BIT-IDENTICAL, NOT MERELY CLOSE.  The group loop keeps the scalar
 * reference's order, and each group's int32 is exact, so `acc += (float)gi *
 * scales[g]` sees the same sequence of floats in the same order.  The gates
 * assert equality, not a tolerance. */
/* The scalar group, always compiled: it is the #else branch below AND the
 * reference self_test_q4_identity() holds every vector variant to.  Neither
 * the NEON nor the x86 int4 kernel had ever been compared against it. */
static int32_t q4_group_i32_scalar(const uint8_t *q, const int8_t *x) {
    int32_t gi = 0;
    for (size_t j = 0; j < QMAT_Q4_GROUP; j += 2) {
        const uint8_t b = q[j / 2];
        const int lo = (int)(b & 0x0F) - 8;
        const int hi = (int)(b >> 4) - 8;
        gi += lo * (int32_t)x[j] + hi * (int32_t)x[j + 1];
    }
    return gi;
}

#if defined(MYNAH_QMAT_AVX2)
/* 32 unsigned nibbles in natural index order from 16 packed bytes.  Low nibble
 * is the even index and high nibble the odd one, so interleaving lo and hi
 * with two unpacks puts them back in order -- which is cheaper than
 * deinterleaving the activation, and leaves the activation a plain load. */
static inline __m256i q4_unpack_u8(const uint8_t *q) {
    const __m128i b = _mm_loadu_si128((const __m128i *)q);
    const __m128i mask = _mm_set1_epi8(0x0F);
    const __m128i lo = _mm_and_si128(b, mask);
    const __m128i hi = _mm_and_si128(_mm_srli_epi16(b, 4), mask);
    return _mm256_set_m128i(_mm_unpackhi_epi8(lo, hi), _mm_unpacklo_epi8(lo, hi));
}

/* sum over one group of 32 of (nibble - 8) * x, exactly. */
static inline int32_t q4_group_i32_avx2(const uint8_t *q, const int8_t *x) {
    const __m256i w = q4_unpack_u8(q);
    const __m256i xv = _mm256_loadu_si256((const __m256i *)x);
    const __m256i eight = _mm256_set1_epi8(8);
    /* maddubs: unsigned x signed, pairwise, into int16. */
    const __m256i t = _mm256_sub_epi16(_mm256_maddubs_epi16(w, xv),
                                       _mm256_maddubs_epi16(eight, xv));
    __m256i acc = _mm256_madd_epi16(t, _mm256_set1_epi16(1));
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(acc),
                              _mm256_extracti128_si256(acc, 1));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
    return _mm_cvtsi128_si32(s);
}
#endif

static float dot_q4(const int8_t *qx, float sx, const uint8_t *q,
                    const float *scales, size_t k, float bias) {
    const size_t groups = k / QMAT_Q4_GROUP;
#if defined(MYNAH_QMAT_DOTPROD)
    const int8x16_t off = vdupq_n_s8(8);
    const uint8x16_t maskv = vdupq_n_u8(0x0F);
    float acc = 0.0f;
    for (size_t g = 0; g < groups; ++g) {
        const uint8x16_t b = vld1q_u8(q + g * 16);
        const int8x16_t lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(b, maskv)), off);
        const int8x16_t hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(b, 4)), off);
        const int8x16x2_t xg = vld2q_s8(qx + g * 32); /* val[0]=even, val[1]=odd */
        int32x4_t ig = vdotq_s32(vdotq_s32(vdupq_n_s32(0), lo, xg.val[0]), hi, xg.val[1]);
        acc = qmat_q4_accum(acc, vaddvq_s32(ig), scales[g]);
    }
    return qmat_q4_finish(acc, sx, bias);
#elif defined(MYNAH_QMAT_AVX2)
    float acc = 0.0f;
    for (size_t g = 0; g < groups; ++g) {
        acc = qmat_q4_accum(acc, q4_group_i32_avx2(q + g * 16, qx + g * 32),
                            scales[g]);
    }
    return qmat_q4_finish(acc, sx, bias);
#else
    float acc = 0.0f;
    for (size_t g = 0; g < groups; ++g) {
        acc = qmat_q4_accum(acc, q4_group_i32_scalar(q + g * 16, qx + g * 32),
                            scales[g]);
    }
    return qmat_q4_finish(acc, sx, bias);
#endif
}

/* Decode is a stream of matrix-vector products.  On ARM, keep four independent
 * output rows in flight so SDOT latency is hidden and the quantized activation
 * vector is loaded once for four weight rows.  Each row retains the same
 * accumulation order as dot_q8/dot_q4; the scalar tail is the reference path. */
#if defined(MYNAH_QMAT_DOTPROD)
/* Two weight rows, four activations, one weight load -- the int8 counterpart of
 * matvec_f16_neon_x4, and it exists for the same reason: without it the batched
 * path walks the batch and calls the single-activation kernel, so a 16-row
 * prefill tile reads every weight block sixteen times.
 *
 * BIT-IDENTICAL, and here that is not an argument, it is arithmetic. The
 * accumulation is int32: vdotq_s32 sums exactly, with no rounding to reassociate
 * and no order to preserve. Only the epilogue is float, and each (activation,
 * row) pair runs exactly the epilogue the row-at-a-time kernel would run, on
 * exactly the same integer. An int32 overflow would change the answer, but it
 * would change it identically in both kernels -- the existing one already
 * assumes it cannot happen for these shapes.
 *
 * SIGNED ONLY, deliberately. The u8 encoding (level != QMAT_U8_OFF) carries the
 * +128 bias and its row-sum correction, and it is the encoding x86 uses for
 * VPDPBUSD. The batched x86 hole is real and bigger than this one -- there the
 * fall-through is the ONLY path -- but it cannot be executed, let alone
 * measured, on an arm64 development machine, so it is not written here on
 * faith. E10-4b.
 *
 * Registers: 8 accumulators, 2 weight vectors, 4 activation vectors. */
static void matvec_q8_neon_x4(float *o0, float *o1, float *o2, float *o3,
                              const int8_t *x0, const int8_t *x1,
                              const int8_t *x2, const int8_t *x3,
                              float s0f, float s1f, float s2f, float s3f,
                              const int8_t *weights, const float *scales,
                              const float *bias, size_t rows, size_t cols) {
    size_t row = 0;
    for (; row + 2u <= rows; row += 2u) {
        const int8_t *wa = weights + row * cols;
        const int8_t *wb = wa + cols;
        int32x4_t a0 = vdupq_n_s32(0), a1 = vdupq_n_s32(0);
        int32x4_t a2 = vdupq_n_s32(0), a3 = vdupq_n_s32(0);
        int32x4_t b0 = vdupq_n_s32(0), b1 = vdupq_n_s32(0);
        int32x4_t b2 = vdupq_n_s32(0), b3 = vdupq_n_s32(0);
        size_t j = 0;
        for (; j + 16u <= cols; j += 16u) {
            const int8x16_t va = vld1q_s8(wa + j);
            const int8x16_t vb = vld1q_s8(wb + j);
            const int8x16_t v0 = vld1q_s8(x0 + j);
            a0 = vdotq_s32(a0, va, v0); b0 = vdotq_s32(b0, vb, v0);
            const int8x16_t v1 = vld1q_s8(x1 + j);
            a1 = vdotq_s32(a1, va, v1); b1 = vdotq_s32(b1, vb, v1);
            const int8x16_t v2 = vld1q_s8(x2 + j);
            a2 = vdotq_s32(a2, va, v2); b2 = vdotq_s32(b2, vb, v2);
            const int8x16_t v3 = vld1q_s8(x3 + j);
            a3 = vdotq_s32(a3, va, v3); b3 = vdotq_s32(b3, vb, v3);
        }
        int32_t sa0 = vaddvq_s32(a0), sa1 = vaddvq_s32(a1);
        int32_t sa2 = vaddvq_s32(a2), sa3 = vaddvq_s32(a3);
        int32_t sb0 = vaddvq_s32(b0), sb1 = vaddvq_s32(b1);
        int32_t sb2 = vaddvq_s32(b2), sb3 = vaddvq_s32(b3);
        for (; j < cols; ++j) {
            const int32_t wav = wa[j], wbv = wb[j];
            sa0 += wav * (int32_t)x0[j]; sb0 += wbv * (int32_t)x0[j];
            sa1 += wav * (int32_t)x1[j]; sb1 += wbv * (int32_t)x1[j];
            sa2 += wav * (int32_t)x2[j]; sb2 += wbv * (int32_t)x2[j];
            sa3 += wav * (int32_t)x3[j]; sb3 += wbv * (int32_t)x3[j];
        }
        const float ba = (bias == NULL) ? 0.0f : bias[row];
        const float bb = (bias == NULL) ? 0.0f : bias[row + 1u];
        const float ka = scales[row], kb = scales[row + 1u];
        o0[row] = qmat_row_epilogue(sa0, qmat_row_scale(ka, s0f), ba);
        o1[row] = qmat_row_epilogue(sa1, qmat_row_scale(ka, s1f), ba);
        o2[row] = qmat_row_epilogue(sa2, qmat_row_scale(ka, s2f), ba);
        o3[row] = qmat_row_epilogue(sa3, qmat_row_scale(ka, s3f), ba);
        o0[row + 1u] = qmat_row_epilogue(sb0, qmat_row_scale(kb, s0f), bb);
        o1[row + 1u] = qmat_row_epilogue(sb1, qmat_row_scale(kb, s1f), bb);
        o2[row + 1u] = qmat_row_epilogue(sb2, qmat_row_scale(kb, s2f), bb);
        o3[row + 1u] = qmat_row_epilogue(sb3, qmat_row_scale(kb, s3f), bb);
    }
    for (; row < rows; ++row) {
        const int8_t *w = weights + row * cols;
        int32x4_t a0 = vdupq_n_s32(0), a1 = vdupq_n_s32(0);
        int32x4_t a2 = vdupq_n_s32(0), a3 = vdupq_n_s32(0);
        size_t j = 0;
        for (; j + 16u <= cols; j += 16u) {
            const int8x16_t v = vld1q_s8(w + j);
            a0 = vdotq_s32(a0, v, vld1q_s8(x0 + j));
            a1 = vdotq_s32(a1, v, vld1q_s8(x1 + j));
            a2 = vdotq_s32(a2, v, vld1q_s8(x2 + j));
            a3 = vdotq_s32(a3, v, vld1q_s8(x3 + j));
        }
        int32_t t0 = vaddvq_s32(a0), t1 = vaddvq_s32(a1);
        int32_t t2 = vaddvq_s32(a2), t3 = vaddvq_s32(a3);
        for (; j < cols; ++j) {
            const int32_t wv = w[j];
            t0 += wv * (int32_t)x0[j]; t1 += wv * (int32_t)x1[j];
            t2 += wv * (int32_t)x2[j]; t3 += wv * (int32_t)x3[j];
        }
        const float bv = (bias == NULL) ? 0.0f : bias[row];
        const float kv = scales[row];
        o0[row] = qmat_row_epilogue(t0, qmat_row_scale(kv, s0f), bv);
        o1[row] = qmat_row_epilogue(t1, qmat_row_scale(kv, s1f), bv);
        o2[row] = qmat_row_epilogue(t2, qmat_row_scale(kv, s2f), bv);
        o3[row] = qmat_row_epilogue(t3, qmat_row_scale(kv, s3f), bv);
    }
}
#endif /* MYNAH_QMAT_DOTPROD */


/* ======================================================================
 * THE INT8 PRIMITIVES, EXPORTED
 *
 * A second consumer exists: the SEANet conv stack (src/convq8.c, E10-5),
 * whose tap GEMMs are 42% of `codec.conv_stack` in the configuration that
 * ships.  It needs int8 arithmetic but not this file's cache -- it has no
 * tensor name to key on, by design (src/seanet.h: "this module never formats
 * a tensor name"), and its weight is a permutation of a weight this file
 * never sees.
 *
 * WHY EXPORTED RATHER THAN REWRITTEN THERE.  The activation encoding is a
 * property of the HOST, not of the caller: signed int8 where SDOT exists,
 * unsigned x+128 with a row-sum correction where VPDPBUSD does.  A second
 * copy of that dispatch is a second thing to get wrong, and it would get it
 * wrong in the direction that matters -- a caller that assumed signed would
 * silently give up VNNI on the x86 half of production, which is where the
 * only committed int8 speedup we have was measured (0.427 vs 0.806 RTF on
 * EPYC Zen 5).
 *
 * WHAT THE CALLER OWNS: the float epilogue.  These return exact int32, and
 * mynah_qmat_epilogue() is the one expression shape that turns one into a
 * float -- see the long comment above qmat_row_scale().  The conv stack
 * accumulates over kernel taps, which this file has no concept of.
 *
 * DETERMINISM.  Integer accumulation is exact and order-independent, so every
 * path below -- SDOT, VNNI, AVX2, scalar, any batch width -- produces the
 * SAME int32 for the same (weight row, activation).  That is not a tolerance,
 * it is an identity, and self_test_dots_i8() asserts it with ==.
 * ====================================================================== */

size_t mynah_qmat_act_bytes(size_t k) { return k; }

float mynah_qmat_act_quantize(void *dst, const float *x, size_t k) {
    if (dst == NULL || x == NULL || k == 0u) return 0.0f;
    return quantize_act(dst, x, k, qmat_u8_level());
}

int mynah_qmat_pack_q8(const float *w, size_t rows, size_t cols, int8_t *q,
                       float *scale, int32_t *rowsum) {
    if (w == NULL || q == NULL || scale == NULL || rowsum == NULL) return -1;
    if (rows == 0u || cols == 0u) return -1;
    if (cols > QMAT_K_MAX) return -1;   /* the rowsum's overflow argument */
    quantize_weight_int8(w, rows, cols, q, scale);
    weight_rowsum_prefix(q, rows, cols, rowsum);
    return 0;
}

size_t mynah_qmat_dots_max_batch(void) { return 4u; }

void mynah_qmat_dots_i8(const int8_t *w, size_t rows, size_t cols,
                        const int32_t *rowsum, const void *const *xq,
                        size_t batch, int32_t *out, size_t out_stride) {
    if (w == NULL || xq == NULL || out == NULL || rows == 0u || batch == 0u) {
        return;
    }
    const int level = qmat_u8_level();
    if (level != QMAT_U8_OFF) {
        /* Activation-stationary, four weight rows at a time: the VNNI unroll.
         * dot4_u8_i32 already folds the -128*rowsum correction, so the int32
         * it returns is the signed kernel's, exactly. */
        for (size_t b = 0; b < batch; ++b) {
            const uint8_t *xu = (const uint8_t *)xq[b];
            int32_t *ob = out + b * out_stride;
            size_t row = 0;
            for (; row + 4u <= rows; row += 4u) {
                int32_t s[4];
                dot4_u8_i32(xu, w, cols, rowsum, row, level, s);
                ob[row] = s[0]; ob[row + 1u] = s[1];
                ob[row + 2u] = s[2]; ob[row + 3u] = s[3];
            }
            for (; row < rows; ++row) {
                ob[row] = dot_u8_i32(xu, w + row * cols, rowsum[row], cols, level);
            }
        }
        return;
    }
#if defined(MYNAH_QMAT_DOTPROD)
    if (batch == 4u) {
        /* Two weight rows, four activations, one weight load -- the integer
         * half of matvec_q8_neon_x4, which cannot be called here because it
         * ends in that kernel's float epilogue. */
        const int8_t *x0 = (const int8_t *)xq[0], *x1 = (const int8_t *)xq[1];
        const int8_t *x2 = (const int8_t *)xq[2], *x3 = (const int8_t *)xq[3];
        int32_t *o0 = out, *o1 = out + out_stride;
        int32_t *o2 = out + 2u * out_stride, *o3 = out + 3u * out_stride;
        size_t row = 0;
        for (; row + 2u <= rows; row += 2u) {
            const int8_t *wa = w + row * cols;
            const int8_t *wb = wa + cols;
            int32x4_t a0 = vdupq_n_s32(0), a1 = vdupq_n_s32(0);
            int32x4_t a2 = vdupq_n_s32(0), a3 = vdupq_n_s32(0);
            int32x4_t b0 = vdupq_n_s32(0), b1 = vdupq_n_s32(0);
            int32x4_t b2 = vdupq_n_s32(0), b3 = vdupq_n_s32(0);
            size_t j = 0;
            for (; j + 16u <= cols; j += 16u) {
                const int8x16_t va = vld1q_s8(wa + j);
                const int8x16_t vb = vld1q_s8(wb + j);
                const int8x16_t v0 = vld1q_s8(x0 + j);
                a0 = vdotq_s32(a0, va, v0); b0 = vdotq_s32(b0, vb, v0);
                const int8x16_t v1 = vld1q_s8(x1 + j);
                a1 = vdotq_s32(a1, va, v1); b1 = vdotq_s32(b1, vb, v1);
                const int8x16_t v2 = vld1q_s8(x2 + j);
                a2 = vdotq_s32(a2, va, v2); b2 = vdotq_s32(b2, vb, v2);
                const int8x16_t v3 = vld1q_s8(x3 + j);
                a3 = vdotq_s32(a3, va, v3); b3 = vdotq_s32(b3, vb, v3);
            }
            int32_t sa0 = vaddvq_s32(a0), sa1 = vaddvq_s32(a1);
            int32_t sa2 = vaddvq_s32(a2), sa3 = vaddvq_s32(a3);
            int32_t sb0 = vaddvq_s32(b0), sb1 = vaddvq_s32(b1);
            int32_t sb2 = vaddvq_s32(b2), sb3 = vaddvq_s32(b3);
            for (; j < cols; ++j) {
                const int32_t wav = wa[j], wbv = wb[j];
                sa0 += wav * (int32_t)x0[j]; sb0 += wbv * (int32_t)x0[j];
                sa1 += wav * (int32_t)x1[j]; sb1 += wbv * (int32_t)x1[j];
                sa2 += wav * (int32_t)x2[j]; sb2 += wbv * (int32_t)x2[j];
                sa3 += wav * (int32_t)x3[j]; sb3 += wbv * (int32_t)x3[j];
            }
            o0[row] = sa0; o1[row] = sa1; o2[row] = sa2; o3[row] = sa3;
            o0[row + 1u] = sb0; o1[row + 1u] = sb1;
            o2[row + 1u] = sb2; o3[row + 1u] = sb3;
        }
        for (; row < rows; ++row) {
            const int8_t *wr = w + row * cols;
            o0[row] = dot_q8_i32(x0, wr, 0, cols, QMAT_U8_OFF);
            o1[row] = dot_q8_i32(x1, wr, 0, cols, QMAT_U8_OFF);
            o2[row] = dot_q8_i32(x2, wr, 0, cols, QMAT_U8_OFF);
            o3[row] = dot_q8_i32(x3, wr, 0, cols, QMAT_U8_OFF);
        }
        return;
    }
#endif
    for (size_t b = 0; b < batch; ++b) {
        const int8_t *xb = (const int8_t *)xq[b];
        int32_t *ob = out + b * out_stride;
        for (size_t row = 0; row < rows; ++row) {
            ob[row] = dot_q8_i32(xb, w + row * cols, 0, cols, QMAT_U8_OFF);
        }
    }
}

static void matvec_q8(float *out, const void *qa, float sx,
                      const int8_t *weights, const float *scales,
                      const int32_t *rowsum, const float *bias,
                      size_t rows, size_t cols, int level) {
    size_t row = 0;
    if (level != QMAT_U8_OFF) {
        const uint8_t *xu = (const uint8_t *)qa;
        for (; row + 4u <= rows; row += 4u) {
            int32_t s[4];
            dot4_u8_i32(xu, weights, cols, rowsum, row, level, s);
            for (size_t r = 0; r < 4u; ++r) {
                out[row + r] = qmat_row_epilogue(
                    s[r], qmat_row_scale(scales[row + r], sx),
                    bias == NULL ? 0.0f : bias[row + r]);
            }
        }
        for (; row < rows; ++row) {
            out[row] = dot_q8(xu, sx, weights + row * cols, scales[row],
                              rowsum[row], cols, level,
                              bias == NULL ? 0.0f : bias[row]);
        }
        return;
    }
    const int8_t *qx = (const int8_t *)qa;
#if defined(MYNAH_QMAT_DOTPROD)
    for (; row + 4u <= rows; row += 4u) {
        const int8_t *w0 = weights + row * cols;
        const int8_t *w1 = w0 + cols;
        const int8_t *w2 = w1 + cols;
        const int8_t *w3 = w2 + cols;
        int32x4_t a0 = vdupq_n_s32(0);
        int32x4_t a1 = vdupq_n_s32(0);
        int32x4_t a2 = vdupq_n_s32(0);
        int32x4_t a3 = vdupq_n_s32(0);
        size_t j = 0;
        for (; j + 16u <= cols; j += 16u) {
            const int8x16_t x = vld1q_s8(qx + j);
            a0 = vdotq_s32(a0, vld1q_s8(w0 + j), x);
            a1 = vdotq_s32(a1, vld1q_s8(w1 + j), x);
            a2 = vdotq_s32(a2, vld1q_s8(w2 + j), x);
            a3 = vdotq_s32(a3, vld1q_s8(w3 + j), x);
        }
        int32_t s0 = vaddvq_s32(a0);
        int32_t s1 = vaddvq_s32(a1);
        int32_t s2 = vaddvq_s32(a2);
        int32_t s3 = vaddvq_s32(a3);
        for (; j < cols; ++j) {
            const int32_t x = qx[j];
            s0 += (int32_t)w0[j] * x;
            s1 += (int32_t)w1[j] * x;
            s2 += (int32_t)w2[j] * x;
            s3 += (int32_t)w3[j] * x;
        }
        out[row] = qmat_row_epilogue(s0, qmat_row_scale(scales[row], sx),
                                     bias == NULL ? 0.0f : bias[row]);
        out[row + 1u] = qmat_row_epilogue(s1, qmat_row_scale(scales[row + 1u], sx),
                                          bias == NULL ? 0.0f : bias[row + 1u]);
        out[row + 2u] = qmat_row_epilogue(s2, qmat_row_scale(scales[row + 2u], sx),
                                          bias == NULL ? 0.0f : bias[row + 2u]);
        out[row + 3u] = qmat_row_epilogue(s3, qmat_row_scale(scales[row + 3u], sx),
                                          bias == NULL ? 0.0f : bias[row + 3u]);
    }
#elif defined(MYNAH_QMAT_AVX2)
    for (; row + 4u <= rows; row += 4u) {
        const int32_t s0 = dot_q8_i32_avx2(qx, weights + row * cols, cols);
        const int32_t s1 = dot_q8_i32_avx2(qx, weights + (row + 1u) * cols, cols);
        const int32_t s2 = dot_q8_i32_avx2(qx, weights + (row + 2u) * cols, cols);
        const int32_t s3 = dot_q8_i32_avx2(qx, weights + (row + 3u) * cols, cols);
        out[row] = qmat_row_epilogue(s0, qmat_row_scale(scales[row], sx),
                                     bias == NULL ? 0.0f : bias[row]);
        out[row + 1u] = qmat_row_epilogue(s1, qmat_row_scale(scales[row + 1u], sx),
                                          bias == NULL ? 0.0f : bias[row + 1u]);
        out[row + 2u] = qmat_row_epilogue(s2, qmat_row_scale(scales[row + 2u], sx),
                                          bias == NULL ? 0.0f : bias[row + 2u]);
        out[row + 3u] = qmat_row_epilogue(s3, qmat_row_scale(scales[row + 3u], sx),
                                          bias == NULL ? 0.0f : bias[row + 3u]);
    }
#endif
    for (; row < rows; ++row) {
        out[row] = dot_q8(qx, sx, weights + row * cols, scales[row], 0,
                          cols, QMAT_U8_OFF, bias == NULL ? 0.0f : bias[row]);
    }
}

static void matvec_q4(float *out, const int8_t *qx, float sx,
                      const uint8_t *weights, const float *scales,
                      const float *bias, size_t rows, size_t cols) {
    size_t row = 0;
#if defined(MYNAH_QMAT_DOTPROD)
    const size_t groups = cols / QMAT_Q4_GROUP;
    const size_t packed_row = cols / 2u;
    const int8x16_t off = vdupq_n_s8(8);
    const uint8x16_t mask = vdupq_n_u8(0x0F);
    for (; row + 4u <= rows; row += 4u) {
        const uint8_t *w0 = weights + row * packed_row;
        const uint8_t *w1 = w0 + packed_row;
        const uint8_t *w2 = w1 + packed_row;
        const uint8_t *w3 = w2 + packed_row;
        const float *s0 = scales + row * groups;
        const float *s1 = s0 + groups;
        const float *s2 = s1 + groups;
        const float *s3 = s2 + groups;
        float a0 = 0.0f;
        float a1 = 0.0f;
        float a2 = 0.0f;
        float a3 = 0.0f;
        for (size_t group = 0; group < groups; ++group) {
            const int8x16x2_t x = vld2q_s8(qx + group * QMAT_Q4_GROUP);
#define Q4_DOT_ROW(weight, scale, accumulator) do { \
                const uint8x16_t packed = vld1q_u8((weight) + group * 16u); \
                const int8x16_t lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(packed, mask)), off); \
                const int8x16_t hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(packed, 4)), off); \
                const int32x4_t dot = vdotq_s32(vdotq_s32(vdupq_n_s32(0), lo, x.val[0]), \
                                                hi, x.val[1]); \
                (accumulator) = qmat_q4_accum((accumulator), vaddvq_s32(dot), \
                                             (scale)[group]); \
            } while (0)
            Q4_DOT_ROW(w0, s0, a0);
            Q4_DOT_ROW(w1, s1, a1);
            Q4_DOT_ROW(w2, s2, a2);
            Q4_DOT_ROW(w3, s3, a3);
#undef Q4_DOT_ROW
        }
        out[row] = qmat_q4_finish(a0, sx, bias == NULL ? 0.0f : bias[row]);
        out[row + 1u] = qmat_q4_finish(a1, sx,
                                       bias == NULL ? 0.0f : bias[row + 1u]);
        out[row + 2u] = qmat_q4_finish(a2, sx,
                                       bias == NULL ? 0.0f : bias[row + 2u]);
        out[row + 3u] = qmat_q4_finish(a3, sx,
                                       bias == NULL ? 0.0f : bias[row + 3u]);
    }
#endif
    for (; row < rows; ++row) {
        out[row] = dot_q4(qx, sx, weights + row * (cols / 2u),
                          scales + row * (cols / QMAT_Q4_GROUP), cols,
                          bias == NULL ? 0.0f : bias[row]);
    }
}

#if defined(MYNAH_QMAT_F16)
/* Weights as IEEE half; activation and accumulation stay f32.  Decode is bound
 * by weight bytes, so halving them is close to halving the time -- measured
 * 2.4-2.8x against Accelerate sgemv on a working set too large to cache, and
 * 1.96x end to end on PocketTTS, against the 2.00x the byte ratio predicts.
 * That gap being ~0 is the measurement that says the in-loop half->float
 * convert costs nothing: it is issued in the shadow of the loads it feeds.
 *
 * Unlike INT8/INT4 this does not quantize the activation, and it needs no
 * scales: f16 carries its own exponent.  On Magpie weights the mean relative
 * error is 1.8e-4 against 1.7e-2 for INT8 (~96x more accurate), and the largest
 * weight is about 6 against the 65504 f16 limit, so overflow is not a concern.
 *
 * Four rows in flight so the activation is read once per four weight rows,
 * matching matvec_q8.  Each row accumulates into its own pair of vectors, so a
 * row's reduction order never depends on how the rows are partitioned.
 *
 * The three kernels below are NOT bit-identical to one another -- they reduce
 * in different orders, which AGENTS.md's numerical rules allow across ISAs and
 * which self_test_f16() bounds against an exact f64 dot.  What they ARE is
 * fed by bit-identical weights: see qmat_f16_pack(). */
#if defined(MYNAH_QMAT_F16_NEON)
static void matvec_f16_neon(float *out, const float *x, const __fp16 *weights,
                            const float *bias, size_t rows, size_t cols) {
    size_t row = 0;
    for (; row + 4u <= rows; row += 4u) {
        const __fp16 *w0 = weights + row * cols;
        const __fp16 *w1 = w0 + cols;
        const __fp16 *w2 = w1 + cols;
        const __fp16 *w3 = w2 + cols;
        float32x4_t a0 = vdupq_n_f32(0.0f), b0 = vdupq_n_f32(0.0f);
        float32x4_t a1 = vdupq_n_f32(0.0f), b1 = vdupq_n_f32(0.0f);
        float32x4_t a2 = vdupq_n_f32(0.0f), b2 = vdupq_n_f32(0.0f);
        float32x4_t a3 = vdupq_n_f32(0.0f), b3 = vdupq_n_f32(0.0f);
        size_t j = 0;
        for (; j + 8u <= cols; j += 8u) {
            const float32x4_t xl = vld1q_f32(x + j);
            const float32x4_t xh = vld1q_f32(x + j + 4u);
            const float16x8_t v0 = vld1q_f16(w0 + j);
            const float16x8_t v1 = vld1q_f16(w1 + j);
            const float16x8_t v2 = vld1q_f16(w2 + j);
            const float16x8_t v3 = vld1q_f16(w3 + j);
            a0 = vfmaq_f32(a0, vcvt_f32_f16(vget_low_f16(v0)), xl);
            b0 = vfmaq_f32(b0, vcvt_f32_f16(vget_high_f16(v0)), xh);
            a1 = vfmaq_f32(a1, vcvt_f32_f16(vget_low_f16(v1)), xl);
            b1 = vfmaq_f32(b1, vcvt_f32_f16(vget_high_f16(v1)), xh);
            a2 = vfmaq_f32(a2, vcvt_f32_f16(vget_low_f16(v2)), xl);
            b2 = vfmaq_f32(b2, vcvt_f32_f16(vget_high_f16(v2)), xh);
            a3 = vfmaq_f32(a3, vcvt_f32_f16(vget_low_f16(v3)), xl);
            b3 = vfmaq_f32(b3, vcvt_f32_f16(vget_high_f16(v3)), xh);
        }
        float s0 = vaddvq_f32(vaddq_f32(a0, b0));
        float s1 = vaddvq_f32(vaddq_f32(a1, b1));
        float s2 = vaddvq_f32(vaddq_f32(a2, b2));
        float s3 = vaddvq_f32(vaddq_f32(a3, b3));
        for (; j < cols; ++j) {
            const float xv = x[j];
            s0 += (float)w0[j] * xv;
            s1 += (float)w1[j] * xv;
            s2 += (float)w2[j] * xv;
            s3 += (float)w3[j] * xv;
        }
        out[row] = s0 + (bias == NULL ? 0.0f : bias[row]);
        out[row + 1u] = s1 + (bias == NULL ? 0.0f : bias[row + 1u]);
        out[row + 2u] = s2 + (bias == NULL ? 0.0f : bias[row + 2u]);
        out[row + 3u] = s3 + (bias == NULL ? 0.0f : bias[row + 3u]);
    }
    for (; row < rows; ++row) {
        const __fp16 *w = weights + row * cols;
        float32x4_t a = vdupq_n_f32(0.0f), b = vdupq_n_f32(0.0f);
        size_t j = 0;
        for (; j + 8u <= cols; j += 8u) {
            const float16x8_t v = vld1q_f16(w + j);
            a = vfmaq_f32(a, vcvt_f32_f16(vget_low_f16(v)), vld1q_f32(x + j));
            b = vfmaq_f32(b, vcvt_f32_f16(vget_high_f16(v)), vld1q_f32(x + j + 4u));
        }
        float s = vaddvq_f32(vaddq_f32(a, b));
        for (; j < cols; ++j) s += (float)w[j] * x[j];
        out[row] = s + (bias == NULL ? 0.0f : bias[row]);
    }
}
/* The same arithmetic as matvec_f16_neon, with the activation loop pulled
 * INSIDE the column loop so one weight load serves four activations.
 *
 * WHY: the batched path used to walk the batch and call the single-activation
 * kernel once per row, so a 16-row prefill tile read every weight block
 * sixteen times.  matvec_f16_neon is a good GEMV -- four weight rows against
 * one activation, eight FMAs per 64 bytes of f16 weights -- but 0.5 FLOP per
 * weight byte is memory-bound by construction, and a GEMM run as B GEMVs
 * inherits that ceiling B times over.  Here two weight rows are loaded once
 * and multiplied into four activations: sixteen FMAs per 32 weight bytes, or
 * 2 FLOP/byte, four times the arithmetic intensity.
 *
 * BIT-IDENTICAL, and that is the point rather than a hope.  Each (activation,
 * row) pair keeps its own pair of accumulators, visits j in the same order,
 * reduces with the same vaddvq_f32(vaddq_f32(lo, hi)), runs the same scalar
 * tail and adds the same bias.  Nothing is reassociated: the only thing that
 * changed is the interleaving of chains that never interacted.  The batch
 * remainder falls back to the row-at-a-time kernel, which is the same
 * arithmetic again.
 *
 * Register budget on aarch64 (32 vectors): 16 accumulators, 4 converted
 * weight vectors, 8 activation vectors. */
#define QMAT_F16_BATCH_LANES 4u

static void matvec_f16_neon_x4(float *o0, float *o1, float *o2, float *o3,
                               const float *x0, const float *x1,
                               const float *x2, const float *x3,
                               const __fp16 *weights, const float *bias,
                               size_t rows, size_t cols) {
    size_t row = 0;
    for (; row + 2u <= rows; row += 2u) {
        const __fp16 *wa = weights + row * cols;
        const __fp16 *wb = wa + cols;
        float32x4_t a0l = vdupq_n_f32(0.0f), a0h = vdupq_n_f32(0.0f);
        float32x4_t a1l = vdupq_n_f32(0.0f), a1h = vdupq_n_f32(0.0f);
        float32x4_t a2l = vdupq_n_f32(0.0f), a2h = vdupq_n_f32(0.0f);
        float32x4_t a3l = vdupq_n_f32(0.0f), a3h = vdupq_n_f32(0.0f);
        float32x4_t b0l = vdupq_n_f32(0.0f), b0h = vdupq_n_f32(0.0f);
        float32x4_t b1l = vdupq_n_f32(0.0f), b1h = vdupq_n_f32(0.0f);
        float32x4_t b2l = vdupq_n_f32(0.0f), b2h = vdupq_n_f32(0.0f);
        float32x4_t b3l = vdupq_n_f32(0.0f), b3h = vdupq_n_f32(0.0f);
        size_t j = 0;
        for (; j + 8u <= cols; j += 8u) {
            const float16x8_t va = vld1q_f16(wa + j);
            const float16x8_t vb = vld1q_f16(wb + j);
            const float32x4_t wal = vcvt_f32_f16(vget_low_f16(va));
            const float32x4_t wah = vcvt_f32_f16(vget_high_f16(va));
            const float32x4_t wbl = vcvt_f32_f16(vget_low_f16(vb));
            const float32x4_t wbh = vcvt_f32_f16(vget_high_f16(vb));

            const float32x4_t x0l = vld1q_f32(x0 + j), x0h = vld1q_f32(x0 + j + 4u);
            a0l = vfmaq_f32(a0l, wal, x0l); a0h = vfmaq_f32(a0h, wah, x0h);
            b0l = vfmaq_f32(b0l, wbl, x0l); b0h = vfmaq_f32(b0h, wbh, x0h);

            const float32x4_t x1l = vld1q_f32(x1 + j), x1h = vld1q_f32(x1 + j + 4u);
            a1l = vfmaq_f32(a1l, wal, x1l); a1h = vfmaq_f32(a1h, wah, x1h);
            b1l = vfmaq_f32(b1l, wbl, x1l); b1h = vfmaq_f32(b1h, wbh, x1h);

            const float32x4_t x2l = vld1q_f32(x2 + j), x2h = vld1q_f32(x2 + j + 4u);
            a2l = vfmaq_f32(a2l, wal, x2l); a2h = vfmaq_f32(a2h, wah, x2h);
            b2l = vfmaq_f32(b2l, wbl, x2l); b2h = vfmaq_f32(b2h, wbh, x2h);

            const float32x4_t x3l = vld1q_f32(x3 + j), x3h = vld1q_f32(x3 + j + 4u);
            a3l = vfmaq_f32(a3l, wal, x3l); a3h = vfmaq_f32(a3h, wah, x3h);
            b3l = vfmaq_f32(b3l, wbl, x3l); b3h = vfmaq_f32(b3h, wbh, x3h);
        }
        float sa0 = vaddvq_f32(vaddq_f32(a0l, a0h));
        float sa1 = vaddvq_f32(vaddq_f32(a1l, a1h));
        float sa2 = vaddvq_f32(vaddq_f32(a2l, a2h));
        float sa3 = vaddvq_f32(vaddq_f32(a3l, a3h));
        float sb0 = vaddvq_f32(vaddq_f32(b0l, b0h));
        float sb1 = vaddvq_f32(vaddq_f32(b1l, b1h));
        float sb2 = vaddvq_f32(vaddq_f32(b2l, b2h));
        float sb3 = vaddvq_f32(vaddq_f32(b3l, b3h));
        for (; j < cols; ++j) {
            const float wav = (float)wa[j], wbv = (float)wb[j];
            sa0 += wav * x0[j]; sb0 += wbv * x0[j];
            sa1 += wav * x1[j]; sb1 += wbv * x1[j];
            sa2 += wav * x2[j]; sb2 += wbv * x2[j];
            sa3 += wav * x3[j]; sb3 += wbv * x3[j];
        }
        const float ba = (bias == NULL) ? 0.0f : bias[row];
        const float bb = (bias == NULL) ? 0.0f : bias[row + 1u];
        o0[row] = sa0 + ba; o0[row + 1u] = sb0 + bb;
        o1[row] = sa1 + ba; o1[row + 1u] = sb1 + bb;
        o2[row] = sa2 + ba; o2[row + 1u] = sb2 + bb;
        o3[row] = sa3 + ba; o3[row + 1u] = sb3 + bb;
    }
    /* An odd last row: the one-row shape of the same accumulation. */
    for (; row < rows; ++row) {
        const __fp16 *w = weights + row * cols;
        float32x4_t a0 = vdupq_n_f32(0.0f), h0 = vdupq_n_f32(0.0f);
        float32x4_t a1 = vdupq_n_f32(0.0f), h1 = vdupq_n_f32(0.0f);
        float32x4_t a2 = vdupq_n_f32(0.0f), h2 = vdupq_n_f32(0.0f);
        float32x4_t a3 = vdupq_n_f32(0.0f), h3 = vdupq_n_f32(0.0f);
        size_t j = 0;
        for (; j + 8u <= cols; j += 8u) {
            const float16x8_t v = vld1q_f16(w + j);
            const float32x4_t wl = vcvt_f32_f16(vget_low_f16(v));
            const float32x4_t wh = vcvt_f32_f16(vget_high_f16(v));
            a0 = vfmaq_f32(a0, wl, vld1q_f32(x0 + j)); h0 = vfmaq_f32(h0, wh, vld1q_f32(x0 + j + 4u));
            a1 = vfmaq_f32(a1, wl, vld1q_f32(x1 + j)); h1 = vfmaq_f32(h1, wh, vld1q_f32(x1 + j + 4u));
            a2 = vfmaq_f32(a2, wl, vld1q_f32(x2 + j)); h2 = vfmaq_f32(h2, wh, vld1q_f32(x2 + j + 4u));
            a3 = vfmaq_f32(a3, wl, vld1q_f32(x3 + j)); h3 = vfmaq_f32(h3, wh, vld1q_f32(x3 + j + 4u));
        }
        float s0 = vaddvq_f32(vaddq_f32(a0, h0));
        float s1 = vaddvq_f32(vaddq_f32(a1, h1));
        float s2 = vaddvq_f32(vaddq_f32(a2, h2));
        float s3 = vaddvq_f32(vaddq_f32(a3, h3));
        for (; j < cols; ++j) {
            const float wv = (float)w[j];
            s0 += wv * x0[j]; s1 += wv * x1[j];
            s2 += wv * x2[j]; s3 += wv * x3[j];
        }
        const float bv = (bias == NULL) ? 0.0f : bias[row];
        o0[row] = s0 + bv; o1[row] = s1 + bv;
        o2[row] = s2 + bv; o3[row] = s3 + bv;
    }
}

/* Two activations, for the batch sizes the four-wide kernel cannot serve.
 *
 * This is not a rounding-out of the API: at the shipping topology a worker
 * holds C/workers live slots, which is two or three at the concurrencies that
 * matter, and the AR step batches exactly those. Without this, the production
 * case is the one case that gets no weight reuse at all -- measured as
 * step.total 496.7 ms at batch 3 against 414.2 at batch 4, where the larger
 * batch does a third more work and still costs less. It also serves the
 * remainder of any batch that is not a multiple of four.
 *
 * Same construction, same bits: one accumulator pair per (activation, row),
 * same column order, same reduction, same tail, same bias. */
static void matvec_f16_neon_x2(float *o0, float *o1,
                               const float *x0, const float *x1,
                               const __fp16 *weights, const float *bias,
                               size_t rows, size_t cols) {
    size_t row = 0;
    for (; row + 2u <= rows; row += 2u) {
        const __fp16 *wa = weights + row * cols;
        const __fp16 *wb = wa + cols;
        float32x4_t a0l = vdupq_n_f32(0.0f), a0h = vdupq_n_f32(0.0f);
        float32x4_t a1l = vdupq_n_f32(0.0f), a1h = vdupq_n_f32(0.0f);
        float32x4_t b0l = vdupq_n_f32(0.0f), b0h = vdupq_n_f32(0.0f);
        float32x4_t b1l = vdupq_n_f32(0.0f), b1h = vdupq_n_f32(0.0f);
        size_t j = 0;
        for (; j + 8u <= cols; j += 8u) {
            const float16x8_t va = vld1q_f16(wa + j);
            const float16x8_t vb = vld1q_f16(wb + j);
            const float32x4_t wal = vcvt_f32_f16(vget_low_f16(va));
            const float32x4_t wah = vcvt_f32_f16(vget_high_f16(va));
            const float32x4_t wbl = vcvt_f32_f16(vget_low_f16(vb));
            const float32x4_t wbh = vcvt_f32_f16(vget_high_f16(vb));
            const float32x4_t x0l = vld1q_f32(x0 + j), x0h = vld1q_f32(x0 + j + 4u);
            a0l = vfmaq_f32(a0l, wal, x0l); a0h = vfmaq_f32(a0h, wah, x0h);
            b0l = vfmaq_f32(b0l, wbl, x0l); b0h = vfmaq_f32(b0h, wbh, x0h);
            const float32x4_t x1l = vld1q_f32(x1 + j), x1h = vld1q_f32(x1 + j + 4u);
            a1l = vfmaq_f32(a1l, wal, x1l); a1h = vfmaq_f32(a1h, wah, x1h);
            b1l = vfmaq_f32(b1l, wbl, x1l); b1h = vfmaq_f32(b1h, wbh, x1h);
        }
        float sa0 = vaddvq_f32(vaddq_f32(a0l, a0h));
        float sa1 = vaddvq_f32(vaddq_f32(a1l, a1h));
        float sb0 = vaddvq_f32(vaddq_f32(b0l, b0h));
        float sb1 = vaddvq_f32(vaddq_f32(b1l, b1h));
        for (; j < cols; ++j) {
            const float wav = (float)wa[j], wbv = (float)wb[j];
            sa0 += wav * x0[j]; sb0 += wbv * x0[j];
            sa1 += wav * x1[j]; sb1 += wbv * x1[j];
        }
        const float ba = (bias == NULL) ? 0.0f : bias[row];
        const float bb = (bias == NULL) ? 0.0f : bias[row + 1u];
        o0[row] = sa0 + ba; o0[row + 1u] = sb0 + bb;
        o1[row] = sa1 + ba; o1[row + 1u] = sb1 + bb;
    }
    for (; row < rows; ++row) {
        const __fp16 *w = weights + row * cols;
        float32x4_t a0 = vdupq_n_f32(0.0f), h0 = vdupq_n_f32(0.0f);
        float32x4_t a1 = vdupq_n_f32(0.0f), h1 = vdupq_n_f32(0.0f);
        size_t j = 0;
        for (; j + 8u <= cols; j += 8u) {
            const float16x8_t v = vld1q_f16(w + j);
            const float32x4_t wl = vcvt_f32_f16(vget_low_f16(v));
            const float32x4_t wh = vcvt_f32_f16(vget_high_f16(v));
            a0 = vfmaq_f32(a0, wl, vld1q_f32(x0 + j)); h0 = vfmaq_f32(h0, wh, vld1q_f32(x0 + j + 4u));
            a1 = vfmaq_f32(a1, wl, vld1q_f32(x1 + j)); h1 = vfmaq_f32(h1, wh, vld1q_f32(x1 + j + 4u));
        }
        float s0 = vaddvq_f32(vaddq_f32(a0, h0));
        float s1 = vaddvq_f32(vaddq_f32(a1, h1));
        for (; j < cols; ++j) {
            const float wv = (float)w[j];
            s0 += wv * x0[j]; s1 += wv * x1[j];
        }
        const float bv = (bias == NULL) ? 0.0f : bias[row];
        o0[row] = s0 + bv; o1[row] = s1 + bv;
    }
}

#endif /* MYNAH_QMAT_F16_NEON */

#if defined(MYNAH_QMAT_F16_X86)
__attribute__((target("avx2,f16c,fma")))
static float qmat_hsum256(__m256 v) {
    const __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), hi);
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 0x55));
    return _mm_cvtss_f32(s);
}

/* VCVTPH2PS widens eight halves into a YMM with no scratch and no shuffle, so
 * the loop body is one 128-bit load, one convert and one FMA per weight row --
 * the same shape as the NEON kernel, four rows deep for the same reason. */
__attribute__((target("avx2,f16c,fma")))
static void matvec_f16_f16c(float *out, const float *x, const uint16_t *weights,
                            const float *bias, size_t rows, size_t cols) {
    size_t row = 0;
    for (; row + 4u <= rows; row += 4u) {
        const uint16_t *w0 = weights + row * cols;
        const uint16_t *w1 = w0 + cols;
        const uint16_t *w2 = w1 + cols;
        const uint16_t *w3 = w2 + cols;
        __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
        __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
        size_t j = 0;
        for (; j + 8u <= cols; j += 8u) {
            const __m256 xv = _mm256_loadu_ps(x + j);
            a0 = _mm256_fmadd_ps(
                _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(const void *)(w0 + j))), xv, a0);
            a1 = _mm256_fmadd_ps(
                _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(const void *)(w1 + j))), xv, a1);
            a2 = _mm256_fmadd_ps(
                _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(const void *)(w2 + j))), xv, a2);
            a3 = _mm256_fmadd_ps(
                _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(const void *)(w3 + j))), xv, a3);
        }
        float s0 = qmat_hsum256(a0), s1 = qmat_hsum256(a1);
        float s2 = qmat_hsum256(a2), s3 = qmat_hsum256(a3);
        for (; j < cols; ++j) {
            const float xv = x[j];
            s0 += qmat_f16_to_f32(w0[j]) * xv;
            s1 += qmat_f16_to_f32(w1[j]) * xv;
            s2 += qmat_f16_to_f32(w2[j]) * xv;
            s3 += qmat_f16_to_f32(w3[j]) * xv;
        }
        out[row]      = s0 + (bias == NULL ? 0.0f : bias[row]);
        out[row + 1u] = s1 + (bias == NULL ? 0.0f : bias[row + 1u]);
        out[row + 2u] = s2 + (bias == NULL ? 0.0f : bias[row + 2u]);
        out[row + 3u] = s3 + (bias == NULL ? 0.0f : bias[row + 3u]);
    }
    for (; row < rows; ++row) {
        const uint16_t *w = weights + row * cols;
        __m256 a = _mm256_setzero_ps();
        size_t j = 0;
        for (; j + 8u <= cols; j += 8u) {
            a = _mm256_fmadd_ps(
                _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(const void *)(w + j))),
                _mm256_loadu_ps(x + j), a);
        }
        float s = qmat_hsum256(a);
        for (; j < cols; ++j) s += qmat_f16_to_f32(w[j]) * x[j];
        out[row] = s + (bias == NULL ? 0.0f : bias[row]);
    }
}
/* The x86 counterpart of matvec_f16_neon_x4, and the same bargain: two weight
 * rows converted once, multiplied into four activations, so the batched path
 * stops re-reading the weight block once per activation.  Sixteen FMAs per 32
 * bytes of f16 weights instead of four.
 *
 * Bit-identical for the same reason and by the same construction: each
 * (activation, row) pair keeps ONE __m256 accumulator, visits j in the same
 * order, reduces with the same qmat_hsum256(), runs the same scalar tail and
 * adds the same bias.  Nothing is reassociated.
 *
 * Register budget on x86-64 (16 YMM): 8 accumulators, 2 converted weight
 * vectors, 4 activation vectors.
 *
 * NOT EXECUTED ON THE MACHINE THAT WROTE IT -- this was developed on arm64.
 * It is gated by self_test_batched_qt(), which memcmps the batched result
 * against the row-at-a-time reference for every cache/group encoding including
 * F16, and which asserts the weight-stationary path actually ran rather than
 * being quietly skipped.  Run `--self-test` on x86 before trusting the speed
 * claim; the correctness claim is the test's to make, not this comment's. */
__attribute__((target("avx2,f16c,fma")))
static void matvec_f16_f16c_x4(float *o0, float *o1, float *o2, float *o3,
                               const float *x0, const float *x1,
                               const float *x2, const float *x3,
                               const uint16_t *weights, const float *bias,
                               size_t rows, size_t cols) {
    size_t row = 0;
    for (; row + 2u <= rows; row += 2u) {
        const uint16_t *wa = weights + row * cols;
        const uint16_t *wb = wa + cols;
        __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
        __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
        __m256 b0 = _mm256_setzero_ps(), b1 = _mm256_setzero_ps();
        __m256 b2 = _mm256_setzero_ps(), b3 = _mm256_setzero_ps();
        size_t j = 0;
        for (; j + 8u <= cols; j += 8u) {
            const __m256 wav = _mm256_cvtph_ps(
                _mm_loadu_si128((const __m128i *)(const void *)(wa + j)));
            const __m256 wbv = _mm256_cvtph_ps(
                _mm_loadu_si128((const __m128i *)(const void *)(wb + j)));
            const __m256 v0 = _mm256_loadu_ps(x0 + j);
            a0 = _mm256_fmadd_ps(wav, v0, a0);
            b0 = _mm256_fmadd_ps(wbv, v0, b0);
            const __m256 v1 = _mm256_loadu_ps(x1 + j);
            a1 = _mm256_fmadd_ps(wav, v1, a1);
            b1 = _mm256_fmadd_ps(wbv, v1, b1);
            const __m256 v2 = _mm256_loadu_ps(x2 + j);
            a2 = _mm256_fmadd_ps(wav, v2, a2);
            b2 = _mm256_fmadd_ps(wbv, v2, b2);
            const __m256 v3 = _mm256_loadu_ps(x3 + j);
            a3 = _mm256_fmadd_ps(wav, v3, a3);
            b3 = _mm256_fmadd_ps(wbv, v3, b3);
        }
        float sa0 = qmat_hsum256(a0), sa1 = qmat_hsum256(a1);
        float sa2 = qmat_hsum256(a2), sa3 = qmat_hsum256(a3);
        float sb0 = qmat_hsum256(b0), sb1 = qmat_hsum256(b1);
        float sb2 = qmat_hsum256(b2), sb3 = qmat_hsum256(b3);
        for (; j < cols; ++j) {
            const float wav = qmat_f16_to_f32(wa[j]);
            const float wbv = qmat_f16_to_f32(wb[j]);
            sa0 += wav * x0[j]; sb0 += wbv * x0[j];
            sa1 += wav * x1[j]; sb1 += wbv * x1[j];
            sa2 += wav * x2[j]; sb2 += wbv * x2[j];
            sa3 += wav * x3[j]; sb3 += wbv * x3[j];
        }
        const float ba = (bias == NULL) ? 0.0f : bias[row];
        const float bb = (bias == NULL) ? 0.0f : bias[row + 1u];
        o0[row] = sa0 + ba; o0[row + 1u] = sb0 + bb;
        o1[row] = sa1 + ba; o1[row + 1u] = sb1 + bb;
        o2[row] = sa2 + ba; o2[row + 1u] = sb2 + bb;
        o3[row] = sa3 + ba; o3[row + 1u] = sb3 + bb;
    }
    for (; row < rows; ++row) {
        const uint16_t *w = weights + row * cols;
        __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
        __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
        size_t j = 0;
        for (; j + 8u <= cols; j += 8u) {
            const __m256 wv = _mm256_cvtph_ps(
                _mm_loadu_si128((const __m128i *)(const void *)(w + j)));
            a0 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(x0 + j), a0);
            a1 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(x1 + j), a1);
            a2 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(x2 + j), a2);
            a3 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(x3 + j), a3);
        }
        float s0 = qmat_hsum256(a0), s1 = qmat_hsum256(a1);
        float s2 = qmat_hsum256(a2), s3 = qmat_hsum256(a3);
        for (; j < cols; ++j) {
            const float wv = qmat_f16_to_f32(w[j]);
            s0 += wv * x0[j]; s1 += wv * x1[j];
            s2 += wv * x2[j]; s3 += wv * x3[j];
        }
        const float bv = (bias == NULL) ? 0.0f : bias[row];
        o0[row] = s0 + bv; o1[row] = s1 + bv;
        o2[row] = s2 + bv; o3[row] = s3 + bv;
    }
}

/* The x86 two-activation kernel; see matvec_f16_neon_x2 for why it exists.
 * Written on arm64 and never executed, like its four-wide sibling, and gated by
 * the same self_test_batched_qt() byte compare. */
__attribute__((target("avx2,f16c,fma")))
static void matvec_f16_f16c_x2(float *o0, float *o1,
                               const float *x0, const float *x1,
                               const uint16_t *weights, const float *bias,
                               size_t rows, size_t cols) {
    size_t row = 0;
    for (; row + 2u <= rows; row += 2u) {
        const uint16_t *wa = weights + row * cols;
        const uint16_t *wb = wa + cols;
        __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
        __m256 b0 = _mm256_setzero_ps(), b1 = _mm256_setzero_ps();
        size_t j = 0;
        for (; j + 8u <= cols; j += 8u) {
            const __m256 wav = _mm256_cvtph_ps(
                _mm_loadu_si128((const __m128i *)(const void *)(wa + j)));
            const __m256 wbv = _mm256_cvtph_ps(
                _mm_loadu_si128((const __m128i *)(const void *)(wb + j)));
            const __m256 v0 = _mm256_loadu_ps(x0 + j);
            a0 = _mm256_fmadd_ps(wav, v0, a0);
            b0 = _mm256_fmadd_ps(wbv, v0, b0);
            const __m256 v1 = _mm256_loadu_ps(x1 + j);
            a1 = _mm256_fmadd_ps(wav, v1, a1);
            b1 = _mm256_fmadd_ps(wbv, v1, b1);
        }
        float sa0 = qmat_hsum256(a0), sa1 = qmat_hsum256(a1);
        float sb0 = qmat_hsum256(b0), sb1 = qmat_hsum256(b1);
        for (; j < cols; ++j) {
            const float wav = qmat_f16_to_f32(wa[j]);
            const float wbv = qmat_f16_to_f32(wb[j]);
            sa0 += wav * x0[j]; sb0 += wbv * x0[j];
            sa1 += wav * x1[j]; sb1 += wbv * x1[j];
        }
        const float ba = (bias == NULL) ? 0.0f : bias[row];
        const float bb = (bias == NULL) ? 0.0f : bias[row + 1u];
        o0[row] = sa0 + ba; o0[row + 1u] = sb0 + bb;
        o1[row] = sa1 + ba; o1[row + 1u] = sb1 + bb;
    }
    for (; row < rows; ++row) {
        const uint16_t *w = weights + row * cols;
        __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
        size_t j = 0;
        for (; j + 8u <= cols; j += 8u) {
            const __m256 wv = _mm256_cvtph_ps(
                _mm_loadu_si128((const __m128i *)(const void *)(w + j)));
            a0 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(x0 + j), a0);
            a1 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(x1 + j), a1);
        }
        float s0 = qmat_hsum256(a0), s1 = qmat_hsum256(a1);
        for (; j < cols; ++j) {
            const float wv = qmat_f16_to_f32(w[j]);
            s0 += wv * x0[j]; s1 += wv * x1[j];
        }
        const float bv = (bias == NULL) ? 0.0f : bias[row];
        o0[row] = s0 + bv; o1[row] = s1 + bv;
    }
}

#endif /* MYNAH_QMAT_F16_X86 */

/* The portable kernel.  It is the correctness reference for the two above, and
 * it is also what actually runs on an x86 host with no F16C -- half the weight
 * bytes at scalar convert cost, rather than a silent retreat to f32. */
static void matvec_f16_scalar(float *out, const float *x, const uint16_t *weights,
                              const float *bias, size_t rows, size_t cols) {
    for (size_t row = 0; row < rows; ++row) {
        const uint16_t *w = weights + row * cols;
        float s = 0.0f;
        for (size_t j = 0; j < cols; ++j) s += qmat_f16_to_f32(w[j]) * x[j];
        out[row] = s + (bias == NULL ? 0.0f : bias[row]);
    }
}

static void matvec_f16(float *out, const float *x, const uint16_t *weights,
                       const float *bias, size_t rows, size_t cols) {
    switch (qmat_f16_kernel()) {
#if defined(MYNAH_QMAT_F16_NEON)
    case QMAT_F16K_NEON:
        matvec_f16_neon(out, x, (const __fp16 *)(const void *)weights, bias,
                        rows, cols);
        return;
#endif
#if defined(MYNAH_QMAT_F16_X86)
    case QMAT_F16K_F16C:
        matvec_f16_f16c(out, x, weights, bias, rows, cols);
        return;
#endif
    default:
        matvec_f16_scalar(out, x, weights, bias, rows, cols);
        return;
    }
}
#endif /* MYNAH_QMAT_F16 */

/* Quantized decode is memory bound on the packed weights exactly like the f32
 * path, and one core only reaches a fraction of DRAM bandwidth.  Every output
 * row is an independent dot product, so splitting the row range over the pool
 * reorders no reduction and stays bit-exact.  Blocks are multiples of the
 * four-row unroll so each worker still runs the row4 kernel, and the last block
 * absorbs the remainder exactly as the serial call did. */
#define QMAT_ROW_BLOCK 32u
/* Below this many weight bytes the pool dispatch costs more than it saves. */
#define QMAT_THREAD_MIN_BYTES 262144u

typedef struct {
    float *out;
    const void *qx;     /* INT8: int8 or (VNNI) uint8 activation; INT4: int8 */
    const float *x;     /* F16: the activation is not quantized */
    float sx;
    const void *weights;
    const float *scales;
    const int32_t *rowsum; /* INT8 + unsigned activation: the +128 correction */
    const float *bias;
    size_t rows;
    size_t cols;
    int qtype;
    int level;          /* QMAT_U8_* -- how qx is encoded, for INT8 only */
} qmat_rows_job;

static void qmat_rows_dispatch(const qmat_rows_job *j, size_t row0, size_t count) {
    const float *bias = j->bias == NULL ? NULL : j->bias + row0;
    if (j->qtype == QMAT_INT8) {
        matvec_q8(j->out + row0, j->qx, j->sx,
                  (const int8_t *)j->weights + row0 * j->cols,
                  j->scales + row0,
                  j->rowsum == NULL ? NULL : j->rowsum + row0,
                  bias, count, j->cols, j->level);
#if defined(MYNAH_QMAT_F16)
    } else if (j->qtype == QMAT_F16) {
        matvec_f16(j->out + row0, j->x,
                   (const uint16_t *)j->weights + row0 * j->cols,
                   bias, count, j->cols);
#endif
    } else {
        matvec_q4(j->out + row0, (const int8_t *)j->qx, j->sx,
                  (const uint8_t *)j->weights + row0 * (j->cols / 2u),
                  j->scales + row0 * (j->cols / QMAT_Q4_GROUP),
                  bias, count, j->cols);
    }
}

static void qmat_rows_block(void *opaque, int block_index) {
    const qmat_rows_job *j = (const qmat_rows_job *)opaque;
    const size_t row0 = (size_t)block_index * QMAT_ROW_BLOCK;
    if (row0 >= j->rows) return;
    size_t count = j->rows - row0;
    if (count > QMAT_ROW_BLOCK) count = QMAT_ROW_BLOCK;
    qmat_rows_dispatch(j, row0, count);
}

static void matvec_q_rows(float *out, const void *qx, const float *x, float sx,
                          const void *weights, const float *scales,
                          const int32_t *rowsum, const float *bias,
                          size_t rows, size_t cols, int qtype, int level) {
    size_t weight_bytes;
    if (qtype == QMAT_INT8) weight_bytes = rows * cols;
    else if (qtype == QMAT_F16) weight_bytes = rows * cols * 2u;
    else weight_bytes = rows * cols / 2u;
    const qmat_rows_job job = {out, qx, x, sx, weights, scales, rowsum, bias,
                               rows, cols, qtype, level};
    if (mynah_num_threads() > 1 && weight_bytes >= QMAT_THREAD_MIN_BYTES &&
        rows > QMAT_ROW_BLOCK) {
        const size_t blocks = (rows + QMAT_ROW_BLOCK - 1u) / QMAT_ROW_BLOCK;
        if (blocks <= (size_t)INT_MAX) {
            mynah_parallel_for((int)blocks, qmat_rows_block, (void *)&job);
            return;
        }
    }
    qmat_rows_dispatch(&job, 0u, rows);
}

/* ------------------------------------------------ threaded greedy argmax (f32)
 *
 * The fused local-transformer output projection walks the full f32 projection
 * matrix per stream, and one core only reaches a fraction of DRAM bandwidth
 * (see the row split above).  Every candidate row is an independent dot
 * product, so each block computes a private best and the reduction visits
 * blocks in ascending row order with a strict >, which preserves the serial
 * loop's first-on-ties winner exactly.  A block whose rows are all masked or
 * all -inf keeps -inf and never wins, matching the serial kernel's behavior. */
#define QMAT_ARGMAX_MAX_BLOCKS 512u

typedef struct {
    const float *weights;
    const float *in;
    const float *bias;
    size_t rows;
    size_t cols;
    size_t block;
    size_t allowed_rows;
    size_t extra_row;
    int allow_extra;
    float *best_value;   /* per block */
    unsigned *best_row;  /* per block */
} argmax_rows_job;

static void argmax_rows_block(void *opaque, int block_index) {
    const argmax_rows_job *j = (const argmax_rows_job *)opaque;
    const size_t row0 = (size_t)block_index * j->block;
    size_t end = row0 + j->block;
    if (end > j->rows) end = j->rows;
    float best = -INFINITY;
    size_t winner = 0;
    for (size_t row = row0; row < end; ++row) {
        const int allowed = row < j->allowed_rows ||
                            (j->allow_extra && row == j->extra_row);
        if (!allowed) continue;
        float value = mynah_dot_f32(j->weights + row * j->cols, j->in, j->cols);
        if (j->bias != NULL) value += j->bias[row];
        if (value > best) {
            best = value;
            winner = row;
        }
    }
    j->best_value[block_index] = best;
    j->best_row[block_index] = (unsigned)winner;
}

/* The threaded-argmax gate, exported so the dispatch report can CALL it
 * instead of restating it.  It is shape-dependent -- that is the point: the
 * same binary threads a 4096-row projection and does not thread a 64-row one,
 * and a report that printed one bit for both would be lying about one of
 * them.  `why` names the clause that decided. */
int mynah_qmat_argmax_mt_resolved(size_t rows, size_t cols, const char **why) {
    const char *env = getenv("MYNAH_ARGMAX_MT");
    const char *reason = NULL;
    int on = 1;
    if (env != NULL && strcmp(env, "0") == 0) {
        on = 0; reason = "MYNAH_ARGMAX_MT=0";
    } else if (mynah_num_threads() <= 1) {
        on = 0; reason = "mynah_num_threads() <= 1";
    } else if (rows <= 2u * QMAT_ROW_BLOCK) {
        on = 0; reason = "rows <= 2 * QMAT_ROW_BLOCK (64)";
    } else if (cols == 0u || rows > SIZE_MAX / cols) {
        on = 0; reason = "degenerate or overflowing shape";
    } else if (rows * cols * sizeof(float) < (size_t)QMAT_THREAD_MIN_BYTES) {
        on = 0; reason = "weight bytes < QMAT_THREAD_MIN_BYTES (256 KiB)";
    } else {
        reason = "threads > 1 and the projection is over 256 KiB";
    }
    if (why != NULL) *why = reason;
    return on;
}

static int matvec_argmax_f32_mt(const float *weights, const float *in,
                                const float *bias, size_t rows, size_t cols,
                                size_t allowed_rows, size_t extra_row,
                                int allow_extra, unsigned *argmax) {
    if (!mynah_qmat_argmax_mt_resolved(rows, cols, NULL)) {
        return mynah_matvec_argmax_f32(weights, in, bias, rows, cols,
                                       allowed_rows, extra_row, allow_extra,
                                       argmax);
    }
    if (weights == NULL || in == NULL || argmax == NULL ||
        allowed_rows > rows || (allow_extra && extra_row >= rows)) {
        return -1;
    }
    size_t blocks = (rows + QMAT_ROW_BLOCK - 1u) / QMAT_ROW_BLOCK;
    if (blocks > QMAT_ARGMAX_MAX_BLOCKS) blocks = QMAT_ARGMAX_MAX_BLOCKS;
    const size_t block = (rows + blocks - 1u) / blocks;
    blocks = (rows + block - 1u) / block;
    float best_value[QMAT_ARGMAX_MAX_BLOCKS];
    unsigned best_row[QMAT_ARGMAX_MAX_BLOCKS];
    argmax_rows_job job = {weights, in, bias, rows, cols, block,
                           allowed_rows, extra_row, allow_extra,
                           best_value, best_row};
    mynah_parallel_for((int)blocks, argmax_rows_block, &job);
    float best = -INFINITY;
    size_t winner = 0;
    for (size_t i = 0; i < blocks; ++i) {
        if (best_value[i] > best) {
            best = best_value[i];
            winner = best_row[i];
        }
    }
    *argmax = (unsigned)winner;
    return 0;
}

/* --------------------------------------------------------------- weight cache */
typedef struct {
    char *name;
    int qtype;
    int8_t *q8;      /* INT8 */
    uint8_t *q4;     /* INT4 packed */
#if defined(MYNAH_QMAT_F16)
    uint16_t *f16;   /* F16: [n * k] raw binary16 bit patterns, no scales */
#endif
    float *scales;   /* INT8: [n]; INT4: [n * k/32] */
    /* INT8 only: sum of each weight row over the prefix the VNNI loop covers.
     * It is the whole cost of the u8 x s8 constraint, paid once per tensor at
     * quantization time instead of once per matvec -- which is why it lives in
     * the per-tensor cache and not in a separate structure with its own lock. */
    int32_t *rowsum;
    size_t n;
    size_t k;
} qmat_entry;

/* Entries are held by pointer, not by value.  A value array would be moved by
 * realloc on growth, and another thread is meanwhile reading an entry for the
 * whole duration of its matvec -- the pointer it holds would dangle.  Indirect
 * storage keeps every entry at a fixed address for its lifetime, so a lock
 * around get-or-create is enough and readers need no lock at all: an entry is
 * fully built before it is published and never mutated afterwards. */
struct mynah_qmat_cache {
    int qtype; /* QMAT_F32 (off) / QMAT_INT8 / QMAT_INT4 / QMAT_F16 */
    int use_row4;
    /* Batched calls that actually reached the weight-stationary kernel, as
     * opposed to falling back to one `_resolved_qt` per row.  It exists because
     * that difference is invisible from the outside -- the fallback produces
     * the same bytes AND leaves the same cache entry behind -- so without a
     * counter, a self-test asserting "the batched path ran" would be asserting
     * nothing.  E8-5's whole claim is that a group with its own encoding now
     * reaches this path, and this is what makes that claim falsifiable.
     * Written only under `mutex`, which the same call already holds. */
    size_t batched_calls;
    qmat_entry **entries;
    size_t count;
    size_t capacity;
    pthread_mutex_t mutex;
};

/* The one place a requested qtype becomes a usable one.  F16 has no scalar
 * fallback -- it needs NEON's half converts -- so asking for it where it is not
 * compiled resolves to exact f32 rather than to a silent approximation. */
static int qmat_qtype_available(int qtype) {
#if !defined(MYNAH_QMAT_F16)
    if (qtype == QMAT_F16) return QMAT_F32;
#endif
    if (qtype != QMAT_INT8 && qtype != QMAT_INT4 && qtype != QMAT_F16) {
        return QMAT_F32;
    }
    return qtype;
}

int mynah_qmat_qtype_from_name(const char *name) {
    if (name == NULL) return -1;
    if (strcmp(name, "f32") == 0 || strcmp(name, "off") == 0) return QMAT_F32;
    if (strcmp(name, "int8") == 0) return QMAT_INT8;
    if (strcmp(name, "int4") == 0) return QMAT_INT4;
    if (strcmp(name, "f16") == 0) return QMAT_F16;
    return -1;
}

int mynah_qmat_qtype_resolved(int qtype) { return qmat_qtype_available(qtype); }

/* MYNAH_QUANT, resolved once, with no third meaning.
 *
 * It used to have one.  `MYNAH_QUANT` unset reaches src/mynah_tts.c:402, which
 * asks for f16 on every non-Magpie engine.  `MYNAH_QUANT=` -- empty, which is
 * what a shell writes for MYNAH_QUANT="$SOMETHING" when SOMETHING is not set --
 * is a non-NULL string that matches none of the three names, so it fell through
 * to f32.  Unset and empty therefore selected DIFFERENT encodings, ~1.8x apart
 * in wall clock and different in audio, and a measurement harness that wrote the
 * empty form put both arms of a paired A/B on the wrong workload and read the
 * result as machine contention.  A typo -- `int-8` -- did the same thing
 * silently.
 *
 * Now: empty means unset, because that is what a shell means by it, and an
 * unrecognised value says so on stderr and then means unset too.  Neither can
 * still select a third encoding nobody named.  The group-spec parser next door
 * already fails a model load on an unknown group name, for the reason that
 * applies here -- "a typo that silently quantizes nothing would show up as a
 * quality result, which is the worst place to discover it".  This cannot fail
 * the load from inside a cache constructor with no error channel, so it is loud
 * instead of fatal, and it says it once.
 *
 * Returns a QMAT_* code, or -1 for "nothing was asked for". */
int mynah_qmat_qtype_from_env(void) {
    static int cached = -2;
    if (cached != -2) return cached;
    const char *env = getenv("MYNAH_QUANT");
    if (env == NULL || env[0] == '\0') {
        cached = -1;
        return cached;
    }
    if (strcmp(env, "int8") == 0) cached = QMAT_INT8;
    else if (strcmp(env, "int4") == 0) cached = QMAT_INT4;
    else if (strcmp(env, "f16") == 0) cached = QMAT_F16;
    else if (strcmp(env, "f32") == 0 || strcmp(env, "off") == 0) cached = QMAT_F32;
    else {
        fprintf(stderr,
                "MYNAH_QUANT=\"%s\" is not an encoding (int8, int4, f16, f32/off). "
                "Ignoring it and taking the default, which is NOT the same as "
                "turning quantization off -- set MYNAH_QUANT=f32 if that is what "
                "you meant.\n", env);
        cached = -1;
    }
    return cached;
}

mynah_qmat_cache *mynah_qmat_cache_new(int enabled) {
    mynah_qmat_cache *c = (mynah_qmat_cache *)calloc(1, sizeof(*c));
    if (c == NULL) return NULL;
    int qtype = QMAT_F32;
    if (enabled < 0) {
        const int from_env = mynah_qmat_qtype_from_env();
        if (from_env >= 0) qtype = from_env;
    } else if (enabled == QMAT_INT8 || enabled == QMAT_INT4 ||
               enabled == QMAT_F16) {
        qtype = enabled;
    }
#if !defined(MYNAH_QMAT_F16)
    /* No half converts on this target: stay on the exact f32 path. */
    if (qtype == QMAT_F16) qtype = QMAT_F32;
#endif
    c->qtype = qtype;
    c->use_row4 = getenv("MYNAH_QMAT_SINGLE_ROW") == NULL;
    if (pthread_mutex_init(&c->mutex, NULL) != 0) {
        free(c);
        return NULL;
    }
    return c;
}

int mynah_qmat_cache_enabled(const mynah_qmat_cache *cache) {
    return cache != NULL && cache->qtype != QMAT_F32;
}

int mynah_qmat_cache_qtype(const mynah_qmat_cache *cache) {
    return cache == NULL ? QMAT_F32 : cache->qtype;
}

const char *mynah_qmat_qtype_name(int qtype) {
    switch (qtype) {
    case QMAT_INT8: return "int8";
    case QMAT_INT4: return "int4";
    case QMAT_F16:  return "f16";
    default:        return "f32";
    }
}

int mynah_qmat_cache_row4(const mynah_qmat_cache *cache) {
    return cache != NULL && cache->use_row4;
}

void mynah_qmat_cache_free(mynah_qmat_cache *cache) {
    if (cache == NULL) return;
    for (size_t i = 0; i < cache->count; ++i) {
        qmat_entry *e = cache->entries[i];
        if (e == NULL) continue;
        free(e->name);
        free(e->q8);
        free(e->q4);
#if defined(MYNAH_QMAT_F16)
        free(e->f16);
#endif
        free(e->scales);
        free(e->rowsum);
        free(e);
    }
    free(cache->entries);
    pthread_mutex_destroy(&cache->mutex);
    free(cache);
}

/* Caller holds cache->mutex. */
static const qmat_entry *cache_lookup(const mynah_qmat_cache *cache, const char *name) {
    for (size_t i = 0; i < cache->count; ++i) {
        if (strcmp(cache->entries[i]->name, name) == 0) return cache->entries[i];
    }
    return NULL;
}

/* Quantize `w` [n, k] under the cache's qtype and store under `name`.  Returns
 * the entry, or NULL on OOM or an INT4 shape it cannot represent (k not a
 * multiple of 32) so the caller can fall back to f32. */
static const qmat_entry *cache_insert(mynah_qmat_cache *cache, const char *name,
                                      const float *w, size_t n, size_t k,
                                      int qtype) {
    if (k == 0 || n > SIZE_MAX / k || n > SIZE_MAX / sizeof(float)) return NULL;
    if (qtype == QMAT_INT4 && (k % QMAT_Q4_GROUP) != 0) return NULL;
    if (cache->count == cache->capacity) {
        const size_t next = cache->capacity == 0 ? 16u : cache->capacity * 2u;
        qmat_entry **grown = (qmat_entry **)realloc(cache->entries, next * sizeof(*grown));
        if (grown == NULL) return NULL;
        cache->entries = grown;
        cache->capacity = next;
    }
    qmat_entry *e = (qmat_entry *)calloc(1, sizeof(*e));
    if (e == NULL) return NULL;
    e->qtype = qtype;
    e->n = n;
    e->k = k;
    e->name = (char *)malloc(strlen(name) + 1u);
    /* `goto fail`, not `return NULL`: this is the one allocation failure here
     * that used to leak the entry it had just calloc'd.  The label frees every
     * member and the entry, and calloc left the rest NULL, so it is correct on
     * the earliest failure as well as the latest.  Found by the same
     * clang-analyzer unix.Malloc check the Code Quality job runs. */
    if (e->name == NULL) goto fail;
    if (qtype == QMAT_INT8) {
        e->q8 = (int8_t *)malloc(n * k);
        e->scales = (float *)malloc(n * sizeof(float));
        e->rowsum = (int32_t *)malloc(n * sizeof(int32_t));
        if (e->q8 == NULL || e->scales == NULL || e->rowsum == NULL) goto fail;
        quantize_weight_int8(w, n, k, e->q8, e->scales);
        /* 4 bytes per output row, computed once for the tensor's lifetime.
         * Always built, not only where VNNI is compiled: it costs a single
         * pass over weights that were just written, and it is what lets the
         * unsigned kernel be self-tested on a host with no VNNI at all. */
        weight_rowsum_prefix(e->q8, n, k, e->rowsum);
#if defined(MYNAH_QMAT_F16)
    } else if (qtype == QMAT_F16) {
        if (n > SIZE_MAX / k / sizeof(uint16_t)) goto fail;
        e->f16 = (uint16_t *)malloc(n * k * sizeof(uint16_t));
        if (e->f16 == NULL) goto fail;
        qmat_f16_pack(e->f16, w, n * k);
#endif
    } else {
        const size_t groups = k / QMAT_Q4_GROUP;
        e->q4 = (uint8_t *)malloc(n * (k / 2u));
        e->scales = (float *)malloc(n * groups * sizeof(float));
        if (e->q4 == NULL || e->scales == NULL) goto fail;
        quantize_weight_int4(w, n, k, e->q4, e->scales);
    }
    memcpy(e->name, name, strlen(name) + 1u);
    /* Published only once fully built: a reader that finds it can use it
     * without a lock, because nothing mutates an entry after this point. */
    cache->entries[cache->count++] = e;
    return e;
fail:
    free(e->name);
    free(e->q8);
    free(e->q4);
#if defined(MYNAH_QMAT_F16)
    free(e->f16);
#endif
    free(e->scales);
    free(e->rowsum);
    free(e);
    return NULL;
}

/* ------------------------------------------------------------ ARM i8mm/SMMLA
 *
 * SMMLA is a 2x2x8 outer product: one instruction consumes two weight rows and
 * TWO activation vectors and does 32 MACs, against SDOT's 16.  That second
 * activation is the whole point, and it is also the honest limit of the
 * instruction here: a single-vector decode matvec has nothing to put in the
 * other half, so vmmlaq_s32 would do the same work as vdotq_s32 at half the
 * lane utilisation.  It is therefore wired into the weight-stationary batched
 * linear -- the one path in this runtime that really does hold B activations
 * against one weight matrix -- and nowhere else.
 *
 * vmmlaq_s32(r, a, b) fills r with the four row-by-row dot products of the two
 * 8-byte halves of `a` against the two of `b`:
 *     r[0] += a.lo . b.lo   r[1] += a.lo . b.hi
 *     r[2] += a.hi . b.lo   r[3] += a.hi . b.hi
 * so packing a = (w[row], w[row+1]) and b = (x[b], x[b+1]) yields the whole
 * 2x2 tile with two 64-bit loads per operand and no weight repacking: the
 * existing row-major int8 layout is already what SMMLA wants.
 *
 * The accumulation is exact int32 and IS bit-identical to two SDOT rows: that
 * half of the original claim is measured and is what self_test_i8mm_identity
 * now checks.  The other half was wrong.  The float epilogue is written
 * textually as matvec_q8's, but "textually the same" is not "rounds the same"
 * under -ffast-math, and it does not: see qmat_i8mm_batched() for the
 * measurement and for why this kernel is not wired into the batched linear by
 * default. */
#if defined(MYNAH_QMAT_ARM_I8MM)
__attribute__((target("+i8mm")))
static void matvec_q8_pair_i8mm(float *out0, float *out1,
                                const int8_t *x0, const int8_t *x1,
                                float sx0, float sx1,
                                const int8_t *weights, const float *scales,
                                const float *bias, size_t rows, size_t cols) {
    size_t row = 0;
    for (; row + 2u <= rows; row += 2u) {
        const int8_t *w0 = weights + row * cols;
        const int8_t *w1 = w0 + cols;
        int32x4_t acc_a = vdupq_n_s32(0);
        int32x4_t acc_b = vdupq_n_s32(0);
        size_t j = 0;
        for (; j + 16u <= cols; j += 16u) {
            const int8x16_t wa = vcombine_s8(vld1_s8(w0 + j), vld1_s8(w1 + j));
            const int8x16_t xa = vcombine_s8(vld1_s8(x0 + j), vld1_s8(x1 + j));
            const int8x16_t wb = vcombine_s8(vld1_s8(w0 + j + 8u), vld1_s8(w1 + j + 8u));
            const int8x16_t xb = vcombine_s8(vld1_s8(x0 + j + 8u), vld1_s8(x1 + j + 8u));
            acc_a = vmmlaq_s32(acc_a, wa, xa);
            acc_b = vmmlaq_s32(acc_b, wb, xb);
        }
        for (; j + 8u <= cols; j += 8u) {
            const int8x16_t wa = vcombine_s8(vld1_s8(w0 + j), vld1_s8(w1 + j));
            const int8x16_t xa = vcombine_s8(vld1_s8(x0 + j), vld1_s8(x1 + j));
            acc_a = vmmlaq_s32(acc_a, wa, xa);
        }
        const int32x4_t acc = vaddq_s32(acc_a, acc_b);
        int32_t s00 = vgetq_lane_s32(acc, 0);   /* w0 . x0 */
        int32_t s01 = vgetq_lane_s32(acc, 1);   /* w0 . x1 */
        int32_t s10 = vgetq_lane_s32(acc, 2);   /* w1 . x0 */
        int32_t s11 = vgetq_lane_s32(acc, 3);   /* w1 . x1 */
        for (; j < cols; ++j) {
            const int32_t a0 = x0[j];
            const int32_t a1 = x1[j];
            s00 += (int32_t)w0[j] * a0;
            s01 += (int32_t)w0[j] * a1;
            s10 += (int32_t)w1[j] * a0;
            s11 += (int32_t)w1[j] * a1;
        }
        out0[row]      = qmat_row_epilogue(s00, qmat_row_scale(scales[row], sx0),
                                           bias == NULL ? 0.0f : bias[row]);
        out0[row + 1u] = qmat_row_epilogue(s10, qmat_row_scale(scales[row + 1u], sx0),
                                           bias == NULL ? 0.0f : bias[row + 1u]);
        out1[row]      = qmat_row_epilogue(s01, qmat_row_scale(scales[row], sx1),
                                           bias == NULL ? 0.0f : bias[row]);
        out1[row + 1u] = qmat_row_epilogue(s11, qmat_row_scale(scales[row + 1u], sx1),
                                           bias == NULL ? 0.0f : bias[row + 1u]);
    }
    if (row < rows) {
        matvec_q8(out0 + row, x0, sx0, weights + row * cols, scales + row, NULL,
                  bias == NULL ? NULL : bias + row, rows - row, cols, QMAT_U8_OFF);
        matvec_q8(out1 + row, x1, sx1, weights + row * cols, scales + row, NULL,
                  bias == NULL ? NULL : bias + row, rows - row, cols, QMAT_U8_OFF);
    }
}
#endif /* MYNAH_QMAT_ARM_I8MM */

/* ------------------------------------------------- weight-stationary batching
 *
 * Decode is bound by the weight bytes, not by the arithmetic: one activation
 * row touches the whole matrix, so B concurrent requests that each walk it in
 * turn pay B trips to DRAM for the same bytes.  Inverting the loops -- outer
 * over row blocks, inner over the batch -- reads each block once and serves
 * every row from cache.  A block is QMAT_ROW_BLOCK * cols bytes (24 KB for a
 * 768-wide int8 weight), so it stays resident in L1 across the batch.
 *
 * This must be bit-exact against the unbatched path, or the audio a request
 * gets would depend on which other requests it happened to batch with.  It is,
 * and the reason used to be narrower than it looks: every (weight row,
 * activation) pair ran THE SAME COMPILED KERNEL over the same k in the same
 * order, and only the order of independent pairs changed.  "The same
 * arithmetic" was not enough -- a second kernel computing the same products,
 * however carefully transcribed, rounded its float epilogue differently under
 * -ffast-math and broke the guarantee.  That is what kept the SMMLA pair
 * kernel out of this path until E4-20b.
 *
 * It is wired in now, and the guarantee no longer rests on there being only
 * one kernel.  The int32 side was always exact -- SMMLA's 2x2 tile is
 * bit-identical to two SDOT rows -- and the float side now goes through
 * qmat_row_scale()/qmat_row_epilogue(), one two-factor product with one
 * grouping, so a row rounds the same whether it arrived as the lead of a pair,
 * the follower, the SDOT tail of an odd batch, or alone.  The structure is
 * still not what is trusted: self_test_batch_membership() asserts the property
 * directly, and mynah_qmat_self_test() additionally runs every shape with the
 * SMMLA wiring forced on and forced off and requires the two to agree bit for
 * bit.  The f32 path has no such guarantee -- sgemm may accumulate differently
 * for M=B than for M=1 -- so it stays one call per row. */
typedef struct {
    const qmat_entry *entry;
    const void *qx;         /* batch * cols, INT8 (signed or u8) / INT4 */
    const float *const *x;  /* batch pointers, F16 */
    const float *sx;        /* batch */
    float *const *out;      /* batch pointers */
    const float *bias;
    size_t batch;
    size_t rows;
    size_t cols;
    int level;              /* QMAT_U8_* -- how qx is encoded, INT8 only */
} qmat_batch_job;

/* MYNAH_QMAT_U8_BATCH=0 restores the per-activation loop for the unsigned
 * encoding, so the change below can be A/B'd in one command on the machine
 * that can actually measure it.  Read ONCE -- a getenv here would run per row
 * block per call, which is the exact defect memoised away in three other
 * files. */
static int qmat_u8_batch_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *e = getenv("MYNAH_QMAT_U8_BATCH");
        cached = (e != NULL && e[0] == '0') ? 0 : 1;
    }
    return cached;
}

static void qmat_batch_rows(const qmat_batch_job *j, size_t row0, size_t count) {
    const qmat_entry *e = j->entry;
    const void *weights;
    if (e->qtype == QMAT_INT8) weights = (const void *)e->q8;
#if defined(MYNAH_QMAT_F16)
    else if (e->qtype == QMAT_F16) weights = (const void *)e->f16;
#endif
    else weights = (const void *)e->q4;
#if defined(MYNAH_QMAT_ARM_I8MM)
    /* Two activations per SMMLA: only INT8, only the signed encoding (the
     * unsigned one belongs to x86), only when the CPU really has FEAT_I8MM. */
    if (e->qtype == QMAT_INT8 && j->level == QMAT_U8_OFF && j->batch >= 2u &&
        j->qx != NULL && qmat_i8mm_batched()) {
        const int8_t *qx = (const int8_t *)j->qx;
        const int8_t *wb = (const int8_t *)weights + row0 * j->cols;
        const float *sc = e->scales + row0;
        const float *bs = j->bias == NULL ? NULL : j->bias + row0;
        size_t b = 0;
        for (; b + 2u <= j->batch; b += 2u) {
            matvec_q8_pair_i8mm(j->out[b] + row0, j->out[b + 1u] + row0,
                                qx + b * j->cols, qx + (b + 1u) * j->cols,
                                j->sx[b], j->sx[b + 1u], wb, sc, bs,
                                count, j->cols);
        }
        for (; b < j->batch; ++b) {
            matvec_q8(j->out[b] + row0, qx + b * j->cols, j->sx[b], wb, sc,
                      NULL, bs, count, j->cols, QMAT_U8_OFF);
        }
        return;
    }
#endif
#if defined(MYNAH_QMAT_DOTPROD)
    /* The same bargain for signed int8, and the one that matters most: this is
     * where a 16-row prefill tile of the codec transformer lands, and it was
     * reading the weight block once per activation.  Signed only -- the u8
     * encoding carries the +128 row-sum correction and is x86's, which cannot
     * be executed here (E10-4b). */
    if (e->qtype == QMAT_INT8 && j->level == QMAT_U8_OFF && j->qx != NULL &&
        j->batch >= 2u) {   /* DOTPROD is a compile gate here, as in matvec_q8 */
        const int8_t *qx = (const int8_t *)j->qx;
        const int8_t *wb = (const int8_t *)weights + row0 * j->cols;
        const float *sc = e->scales + row0;
        const float *bs = (j->bias == NULL) ? NULL : j->bias + row0;
        size_t b = 0;
        for (; b + 4u <= j->batch; b += 4u) {
            matvec_q8_neon_x4(j->out[b] + row0, j->out[b + 1u] + row0,
                              j->out[b + 2u] + row0, j->out[b + 3u] + row0,
                              qx + b * j->cols, qx + (b + 1u) * j->cols,
                              qx + (b + 2u) * j->cols, qx + (b + 3u) * j->cols,
                              j->sx[b], j->sx[b + 1u], j->sx[b + 2u],
                              j->sx[b + 3u], wb, sc, bs, count, j->cols);
        }
        if (j->batch - b == 3u) {
            /* Three left: the four-lane kernel with the last activation
             * repeated, spending a quarter more arithmetic to load the weight
             * block once instead of twice.  The repeated lane writes the same
             * address twice with the same integer, so the bytes cannot differ;
             * see the f16 twin below for the same reasoning and the lane-width
             * self-test that gates both. */
            matvec_q8_neon_x4(j->out[b] + row0, j->out[b + 1u] + row0,
                              j->out[b + 2u] + row0, j->out[b + 2u] + row0,
                              qx + b * j->cols, qx + (b + 1u) * j->cols,
                              qx + (b + 2u) * j->cols, qx + (b + 2u) * j->cols,
                              j->sx[b], j->sx[b + 1u], j->sx[b + 2u],
                              j->sx[b + 2u], wb, sc, bs, count, j->cols);
            return;
        }
        for (; b < j->batch; ++b) {
            matvec_q8(j->out[b] + row0, qx + b * j->cols, j->sx[b], wb, sc,
                      NULL, bs, count, j->cols, QMAT_U8_OFF);
        }
        return;
    }
#endif
#if defined(MYNAH_QMAT_F16)
    /* Four activations per weight load.  This is the f16 counterpart of the
     * SMMLA pair above and exists for the same reason: without it the batch is
     * walked one activation at a time and a 16-row prefill tile reads every
     * weight block sixteen times.  Same arithmetic, same order, same bits --
     * see matvec_f16_neon_x4.  The scalar half kernel is deliberately excluded:
     * it has no vector registers to keep a weight block in, so the four-wide
     * shape would buy it nothing and cost it clarity. */
    {
        const int f16k = qmat_f16_kernel();
        const int wide_f16 =
            e->qtype == QMAT_F16 && j->x != NULL && j->batch >= 2u &&
            (f16k == QMAT_F16K_NEON || f16k == QMAT_F16K_F16C);
        if (wide_f16) {
            const float *bs = (j->bias == NULL) ? NULL : j->bias + row0;
            size_t b = 0;
            for (; b + QMAT_F16_BATCH_LANES <= j->batch;
                 b += QMAT_F16_BATCH_LANES) {
#if defined(MYNAH_QMAT_F16_NEON)
                matvec_f16_neon_x4(
                    j->out[b] + row0, j->out[b + 1u] + row0,
                    j->out[b + 2u] + row0, j->out[b + 3u] + row0,
                    j->x[b], j->x[b + 1u], j->x[b + 2u], j->x[b + 3u],
                    (const __fp16 *)weights + row0 * j->cols, bs, count, j->cols);
#else
                matvec_f16_f16c_x4(
                    j->out[b] + row0, j->out[b + 1u] + row0,
                    j->out[b + 2u] + row0, j->out[b + 3u] + row0,
                    j->x[b], j->x[b + 1u], j->x[b + 2u], j->x[b + 3u],
                    (const uint16_t *)weights + row0 * j->cols, bs, count, j->cols);
#endif
            }
            /* Two at a time for what is left, and this is the case that
             * matters in production rather than a tidy-up: at the shipping
             * topology a worker holds C/workers live slots -- three at C48 --
             * and the AR step batches exactly those. */
            if (j->batch - b == 3u) {
                /* EXPERIMENT: three left, served by the four-lane kernel with
                 * the last activation repeated. */
#if defined(MYNAH_QMAT_F16_NEON)
                matvec_f16_neon_x4(
                    j->out[b] + row0, j->out[b + 1u] + row0,
                    j->out[b + 2u] + row0, j->out[b + 2u] + row0,
                    j->x[b], j->x[b + 1u], j->x[b + 2u], j->x[b + 2u],
                    (const __fp16 *)weights + row0 * j->cols, bs, count, j->cols);
#else
                matvec_f16_f16c_x4(
                    j->out[b] + row0, j->out[b + 1u] + row0,
                    j->out[b + 2u] + row0, j->out[b + 2u] + row0,
                    j->x[b], j->x[b + 1u], j->x[b + 2u], j->x[b + 2u],
                    (const uint16_t *)weights + row0 * j->cols, bs, count, j->cols);
#endif
                return;
            }
            if (b + 2u <= j->batch) {
#if defined(MYNAH_QMAT_F16_NEON)
                matvec_f16_neon_x2(
                    j->out[b] + row0, j->out[b + 1u] + row0,
                    j->x[b], j->x[b + 1u],
                    (const __fp16 *)weights + row0 * j->cols, bs, count, j->cols);
#else
                matvec_f16_f16c_x2(
                    j->out[b] + row0, j->out[b + 1u] + row0,
                    j->x[b], j->x[b + 1u],
                    (const uint16_t *)weights + row0 * j->cols, bs, count, j->cols);
#endif
                b += 2u;
            }
            for (; b < j->batch; ++b) {
                const qmat_rows_job rj = {
                    j->out[b], NULL, j->x[b], 0.0f,
                    weights, e->scales, e->rowsum, j->bias, j->rows, j->cols,
                    e->qtype, j->level
                };
                qmat_rows_dispatch(&rj, row0, count);
            }
            return;
        }
    }
#endif
    /* ------------------------------------------------ the x86 batched hole
     *
     * Every weight-stationary path above -- SMMLA, SDOT, the f16 lanes -- is
     * gated on the SIGNED activation encoding, because the unsigned x+128 form
     * and its row-sum correction are x86's.  So on the half of production that
     * runs VPDPBUSD the batch fell through to the loop below, which walks the
     * activations and reads the whole weight block once for each: a 16-row
     * prefill tile of the codec transformer read every block sixteen times.
     * That is E10-4b, and it is the same defect the SDOT path was written to
     * fix, left standing on the other architecture.
     *
     * This adds no instruction and no kernel.  It swaps the loop order -- row
     * block outer, batch inner -- so four weight rows are loaded once and stay
     * in L1 for the whole batch.  dot4_u8_i32 is the same function the
     * per-activation path calls, on the same bytes, so the int32 is identical
     * BY CONSTRUCTION rather than by tolerance, and the float epilogue is the
     * shared helper.
     *
     * WHAT IS AND IS NOT CLAIMED.  Correctness is gated everywhere:
     * MYNAH_QMAT_VNNI=scalar forces exactly this encoding and this access
     * pattern on any host, self_test_lane_widths compares the result against
     * the row-at-a-time reference with memcmp at every width from 1 to 9, and
     * two mutations of this block are caught by it -- and by nothing else,
     * since the default ARM run never reaches here.
     *
     * SPEED IS NOT CLAIMED.  A/B'd on this arm64 machine under
     * MYNAH_QMAT_VNNI=scalar, batch 4: 1.050 / 1.022 / 1.004 -- noise.  That
     * is the expected null result and not a refutation: the scalar unsigned
     * kernel is arithmetic-bound, so reordering weight traffic cannot move it,
     * which is the same reason a key-stationary variant lost in
     * src/transformer_ar.c.  The reuse only pays where VPDPBUSD makes the
     * arithmetic cheap enough for the traffic to matter, and that machine is
     * not this one.  MYNAH_QMAT_U8_BATCH=0 restores the old loop so the box
     * can settle it in one command. */
    if (e->qtype == QMAT_INT8 && j->level != QMAT_U8_OFF && j->qx != NULL &&
        j->batch >= 2u && qmat_u8_batch_enabled()) {
        const unsigned char *qx = (const unsigned char *)j->qx;
        const int8_t *wb = (const int8_t *)weights + row0 * j->cols;
        const float *sc = e->scales + row0;
        const int32_t *rs = e->rowsum + row0;
        const float *bs = (j->bias == NULL) ? NULL : j->bias + row0;
        size_t row = 0;
        for (; row + 4u <= count; row += 4u) {
            for (size_t b = 0; b < j->batch; ++b) {
                int32_t acc[4];
                dot4_u8_i32(qx + b * j->cols, wb, j->cols, rs, row, j->level,
                            acc);
                float *out = j->out[b] + row0;
                for (size_t r = 0; r < 4u; ++r) {
                    out[row + r] = qmat_row_epilogue(
                        acc[r], qmat_row_scale(sc[row + r], j->sx[b]),
                        bs == NULL ? 0.0f : bs[row + r]);
                }
            }
        }
        for (; row < count; ++row) {
            for (size_t b = 0; b < j->batch; ++b) {
                const int32_t acc = dot_u8_i32(qx + b * j->cols,
                                               wb + row * j->cols, rs[row],
                                               j->cols, j->level);
                j->out[b][row0 + row] = qmat_row_epilogue(
                    acc, qmat_row_scale(sc[row], j->sx[b]),
                    bs == NULL ? 0.0f : bs[row]);
            }
        }
        return;
    }

    for (size_t b = 0; b < j->batch; ++b) {
        const qmat_rows_job rj = {
            j->out[b],
            j->qx == NULL ? NULL : (const void *)((const unsigned char *)j->qx +
                                                  b * j->cols),
            j->x == NULL ? NULL : j->x[b],
            j->sx == NULL ? 0.0f : j->sx[b],
            weights, e->scales, e->rowsum, j->bias, j->rows, j->cols,
            e->qtype, j->level
        };
        qmat_rows_dispatch(&rj, row0, count);
    }
}

static void qmat_batch_block(void *opaque, int block_index) {
    const qmat_batch_job *j = (const qmat_batch_job *)opaque;
    const size_t row0 = (size_t)block_index * QMAT_ROW_BLOCK;
    if (row0 >= j->rows) return;
    size_t count = j->rows - row0;
    if (count > QMAT_ROW_BLOCK) count = QMAT_ROW_BLOCK;
    qmat_batch_rows(j, row0, count);
}

/* --------------------------------------------------------------- public API */
int mynah_qmat_greedy_argmax_resolved(mynah_qmat_cache *cache, const char *name,
                             const float *weight_data, const float *in, size_t k, size_t n,
                             const float *bias, size_t allowed_rows,
                             unsigned extra_row, int allow_extra, unsigned *argmax,
                             char *error, size_t error_capacity) {
    if (cache == NULL || cache->qtype != QMAT_F32) return 1;
    if (matvec_argmax_f32_mt(weight_data, in, bias, n, k,
                             allowed_rows, (size_t)extra_row,
                             allow_extra, argmax) != 0) {
        snprintf(error, error_capacity, "invalid greedy projection shape: %s", name);
        return -1;
    }
    return 0;
}

int mynah_qmat_greedy_argmax(mynah_qmat_cache *cache, const mynah_weights *file,
                             const char *name, const float *in, size_t k, size_t n,
                             const float *bias, size_t allowed_rows,
                             unsigned extra_row, int allow_extra, unsigned *argmax,
                             char *error, size_t error_capacity) {
    mynah_tensor weight;
    if (mynah_weights_get(file, name, &weight) != 0) {
        snprintf(error, error_capacity, "model tensor is missing: %s", name);
        return -1;
    }
    return mynah_qmat_greedy_argmax_resolved(cache, name, weight.data, in, k, n,
                                             bias, allowed_rows, extra_row,
                                             allow_extra, argmax, error, error_capacity);
}

/* ------------------------------------------------- where the error comes from
 *
 * "int8 broke parity" is not actionable; "the activation entering this one
 * projection loses 4.4% and its weight loses 0.3%" is.  The two sides of a
 * w8a8 product fail for different reasons and have different fixes -- a weight
 * outlier wants a finer scale granularity, an activation outlier wants a
 * different quantizer or an exclusion -- so they are measured separately,
 * per tensor, instead of being inferred from the end-to-end number.
 *
 * OFF unless MYNAH_QMAT_ACT_STATS is set; when off this costs one load of a
 * static int per call. */
#define QMAT_STATS_MAX 128u

typedef struct {
    char name[64]; /* copied: the engine's key table dies before atexit runs */
    unsigned long long calls;
    double act_rel_sum;  /* ||x - dequant(quant(x))|| / ||x||, summed         */
    double act_peak_sum; /* max|x| / rms(x), summed                          */
    double w_rel;        /* ||W - dequant(W)||_F / ||W||_F, once             */
    size_t rows, cols;
} qmat_stats_entry;

static qmat_stats_entry g_stats[QMAT_STATS_MAX];
static size_t g_stats_count;
static pthread_mutex_t g_stats_mutex = PTHREAD_MUTEX_INITIALIZER;

/* fork() keeps only the calling thread, so a mutex a pool thread held at that
 * instant stays locked forever in the child: its owner is not there to unlock
 * it.  The prefork server forks with the pool already running -- the model has
 * to be open before the fork so the weights are one physical copy -- so this
 * is reachable, not theoretical.  Registered once, lazily, from the first
 * stats call; pthread_atfork handlers survive fork and run in the child. */
static void qmat_stats_after_fork(void) {
    pthread_mutex_init(&g_stats_mutex, NULL);
}

static void qmat_stats_register_atfork(void) {
    pthread_atfork(NULL, NULL, qmat_stats_after_fork);
}

/* Registered once from the first stats call.  pthread_atfork handlers survive
 * fork and run in the child; registering lazily keeps a process that never
 * touches qmat from paying for a handler it cannot need. */
static void qmat_stats_atfork_once(void) {
    static pthread_once_t once = PTHREAD_ONCE_INIT;
    pthread_once(&once, qmat_stats_register_atfork);
}
static int g_stats_on = -1;
static int g_stats_registered;

static void qmat_stats_report(void) {
    pthread_mutex_lock(&g_stats_mutex);
    if (g_stats_count == 0) {
        pthread_mutex_unlock(&g_stats_mutex);
        return;
    }
    fprintf(stderr,
            "\nMYNAH_QMAT_ACT_STATS -- per-tensor quantization error, the two "
            "sides kept apart\n%-26s %8s %8s %10s %10s %10s\n", "tensor", "rows",
            "cols", "calls", "act_rel", "w_rel");
    for (size_t i = 0; i < g_stats_count; ++i) {
        const qmat_stats_entry *e = &g_stats[i];
        const double n = (double)(e->calls ? e->calls : 1u);
        fprintf(stderr, "%-26s %8zu %8zu %10llu %10.3e %10.3e   peak/rms %6.2f\n",
                e->name, e->rows, e->cols, e->calls, e->act_rel_sum / n, e->w_rel,
                e->act_peak_sum / n);
    }
    pthread_mutex_unlock(&g_stats_mutex);
}

static int qmat_stats_enabled(void) {
    if (g_stats_on < 0) {
        const char *env = getenv("MYNAH_QMAT_ACT_STATS");
        g_stats_on = (env != NULL && env[0] != '\0' && strcmp(env, "0") != 0);
    }
    return g_stats_on;
}

/* Relative Frobenius error of the cached weight against the f32 original. */
static double qmat_weight_rel(const qmat_entry *e, const float *w, size_t n,
                              size_t k) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < k; ++j) {
            const double ref = (double)w[i * k + j];
            double got = ref;
            if (e->qtype == QMAT_INT8) {
                got = (double)e->q8[i * k + j] * (double)e->scales[i];
            } else if (e->qtype == QMAT_INT4) {
                const size_t g = j / QMAT_Q4_GROUP;
                const uint8_t byte = e->q4[i * (k / 2u) + j / 2u];
                const int nib = (j % 2u == 0) ? (byte & 0x0fu) : (byte >> 4);
                got = (double)(nib - 8) *
                      (double)e->scales[i * (k / QMAT_Q4_GROUP) + g];
#if defined(MYNAH_QMAT_F16)
            } else if (e->qtype == QMAT_F16) {
                got = (double)qmat_f16_to_f32(e->f16[i * k + j]);
#endif
            }
            num += (ref - got) * (ref - got);
            den += ref * ref;
        }
    }
    return den > 0.0 ? sqrt(num / den) : 0.0;
}

static void qmat_stats_record(const char *name, const qmat_entry *e,
                              const float *weight, const float *x, size_t k,
                              size_t n) {
    double amax = 0.0, sq = 0.0;
    for (size_t j = 0; j < k; ++j) {
        const double v = fabs((double)x[j]);
        if (v > amax) amax = v;
        sq += (double)x[j] * (double)x[j];
    }
    const double scale = amax > 0.0 ? amax / 127.0 : 1.0;
    double num = 0.0;
    for (size_t j = 0; j < k; ++j) {
        const double q = (double)(int)lrint((double)x[j] / scale) * scale;
        num += ((double)x[j] - q) * ((double)x[j] - q);
    }
    const double rms = sqrt(sq / (double)k);
    pthread_mutex_lock(&g_stats_mutex);
    if (!g_stats_registered) {
        atexit(qmat_stats_report);
        qmat_stats_atfork_once();
        g_stats_registered = 1;
    }
    qmat_stats_entry *slot = NULL;
    for (size_t i = 0; i < g_stats_count; ++i) {
        if (strcmp(g_stats[i].name, name) == 0) {
            slot = &g_stats[i];
            break;
        }
    }
    if (slot == NULL && g_stats_count < QMAT_STATS_MAX) {
        slot = &g_stats[g_stats_count++];
        snprintf(slot->name, sizeof(slot->name), "%s", name);
        slot->rows = n;
        slot->cols = k;
        slot->w_rel = qmat_weight_rel(e, weight, n, k);
    }
    if (slot != NULL) {
        ++slot->calls;
        slot->act_rel_sum += (sq > 0.0) ? sqrt(num / sq) : 0.0;
        slot->act_peak_sum += (rms > 0.0) ? amax / rms : 0.0;
    }
    pthread_mutex_unlock(&g_stats_mutex);
}

/* Scratch for turning a contiguous multi-row call into a weight-stationary one.
 *
 * Per thread, grown once, and freed by a pthread_key destructor rather than
 * leaked: `make leaks` is a gate here and a cache that cannot be reclaimed is a
 * leak with a good excuse.  Sized for the largest call this path accepts --
 * QMAT_SMALL_COUNT rows of QMAT_K_MAX -- which is 128 KB of int8 plus a scale
 * per row, too much to put on a stack that pool workers also use. */
typedef struct {
    int8_t *qx;
    float *sx;
    size_t rows, cols;
} qmat_row_scratch;

static pthread_key_t g_row_scratch_key;
static pthread_once_t g_row_scratch_once = PTHREAD_ONCE_INIT;

static void row_scratch_free(void *p) {
    qmat_row_scratch *sc = (qmat_row_scratch *)p;
    if (sc == NULL) return;
    free(sc->qx);
    free(sc->sx);
    free(sc);
}

static void row_scratch_init(void) {
    (void)pthread_key_create(&g_row_scratch_key, row_scratch_free);
}

/* 0 and both pointers set, or -1 and the caller keeps its per-row loop. */
static int row_scratch_get(size_t rows, size_t cols, int8_t **qx, float **sx) {
    pthread_once(&g_row_scratch_once, row_scratch_init);
    qmat_row_scratch *sc = (qmat_row_scratch *)pthread_getspecific(g_row_scratch_key);
    if (sc == NULL) {
        sc = (qmat_row_scratch *)calloc(1, sizeof(*sc));
        if (sc == NULL) return -1;
        if (pthread_setspecific(g_row_scratch_key, sc) != 0) { free(sc); return -1; }
    }
    if (rows > sc->rows || cols > sc->cols) {
        const size_t r = rows > sc->rows ? rows : sc->rows;
        const size_t c = cols > sc->cols ? cols : sc->cols;
        if (r > SIZE_MAX / c) return -1;
        int8_t *nq = (int8_t *)realloc(sc->qx, r * c);
        if (nq == NULL) return -1;
        sc->qx = nq;
        float *ns = (float *)realloc(sc->sx, r * sizeof(float));
        if (ns == NULL) return -1;
        sc->sx = ns;
        sc->rows = r;
        sc->cols = c;
    }
    *qx = sc->qx;
    *sx = sc->sx;
    return 0;
}

int mynah_qmat_linear_resolved_qt(mynah_qmat_cache *cache,
                      const mynah_backend *backend, const char *name,
                      const float *weight_data,
                      const float *in, float *out, size_t count, size_t k, size_t n,
                      const float *bias, int qtype, char *error,
                      size_t error_capacity) {
    /* qtype < 0 means "whatever the cache resolved to", which is every caller
     * that predates per-group precision.  A caller that names one gets exactly
     * that tensor in that encoding: entries are keyed by name, and a name
     * belongs to one group, so an entry's encoding never changes under it. */
    const int want = (qtype < 0) ? (cache == NULL ? QMAT_F32 : cache->qtype)
                                 : qmat_qtype_available(qtype);
    const int use_q = cache != NULL && want != QMAT_F32 &&
                      count <= QMAT_SMALL_COUNT && k <= QMAT_K_MAX;
    const qmat_entry *e = NULL;
    if (use_q) {
        /* Get-or-create is one critical section; the returned entry then stays
         * valid and immutable, so the matvec below runs outside the lock. */
        pthread_mutex_lock(&cache->mutex);
        e = cache_lookup(cache, name);
        if (e == NULL) e = cache_insert(cache, name, weight_data, n, k, want);
        pthread_mutex_unlock(&cache->mutex);
    }
    if (e == NULL) {
        /* Disabled, too large, unrepresentable, or OOM: exact f32 matmul. */
        return mynah_backend_matmul(backend, in, out, count, k, n, weight_data, bias,
                                    error, error_capacity);
    }
    /* MORE THAN ONE ROW IS A GEMM, and it was being run as `count` GEMVs.
     *
     * The loop below walks the activation rows and sweeps the whole weight for
     * each one, so a 16-row prefill tile read every weight block sixteen times.
     * The weight-stationary path already exists, already blocks over rows for
     * the pool, and already carries the four- and two-activation kernels; all
     * that was missing was handing it these rows.  The pool meter says what
     * this is worth: qmat_rows_block was 72.5% of all regions against
     * qmat_batch_block's 14.5%.
     *
     * Byte-identical, and gated rather than argued:
     * self_test_lane_widths() compares this exact delegation against the
     * row-at-a-time reference with memcmp, at every width from 1 to 9, for
     * every batched encoding.
     *
     * A refusal -- no scratch, too many rows -- falls through to the loop
     * below, which is a slower call and never a different sample. */
    if (count >= 2u) {
        const float *in_rows[QMAT_SMALL_COUNT];
        float *out_rows[QMAT_SMALL_COUNT];
        int8_t *qs = NULL;
        float *ss = NULL;
        if (count <= (size_t)QMAT_SMALL_COUNT &&
            row_scratch_get(count, k, &qs, &ss) == 0) {
            for (size_t t = 0; t < count; ++t) {
                in_rows[t] = in + t * k;
                out_rows[t] = out + t * n;
            }
            return mynah_qmat_linear_batched_qt(cache, backend, name, weight_data,
                                                in_rows, out_rows, count, k, n,
                                                bias, qs, ss, qtype, error,
                                                error_capacity);
        }
    }

    /* uint8_t because the INT8 activation may be the unsigned x+128 encoding
     * the VNNI kernels need; both are character types, so the int8 paths read
     * the same storage through an int8_t* without an aliasing violation. */
    uint8_t qx[QMAT_K_MAX];
    const int level = e->qtype == QMAT_INT8 ? qmat_u8_level() : QMAT_U8_OFF;
    for (size_t t = 0; t < count; ++t) {
        const float *xr = in + t * k;
        float *orow = out + t * n;
#if defined(MYNAH_QMAT_F16)
        /* F16 carries its own exponent, so the activation stays f32 and there
         * is no activation-quantization pass at all. */
        if (e->qtype == QMAT_F16) {
            matvec_q_rows(orow, NULL, xr, 0.0f, e->f16, NULL, NULL, bias, n, k,
                          QMAT_F16, QMAT_U8_OFF);
            continue;
        }
#endif
        if (qmat_stats_enabled()) {
            qmat_stats_record(name, e, weight_data, xr, k, n);
        }
        const float sx = quantize_act(qx, xr, k, level);
        if (!cache->use_row4) {
            for (size_t row = 0; row < n; ++row) {
                const float b = bias == NULL ? 0.0f : bias[row];
                orow[row] = e->qtype == QMAT_INT8
                                ? dot_q8(qx, sx, e->q8 + row * k, e->scales[row],
                                         e->rowsum[row], k, level, b)
                                : dot_q4((const int8_t *)qx, sx,
                                         e->q4 + row * (k / 2u),
                                         e->scales + row * (k / QMAT_Q4_GROUP),
                                         k, b);
            }
        } else if (e->qtype == QMAT_INT8) {
            matvec_q_rows(orow, qx, xr, sx, e->q8, e->scales, e->rowsum, bias,
                          n, k, QMAT_INT8, level);
        } else {
            matvec_q_rows(orow, qx, xr, sx, e->q4, e->scales, NULL, bias,
                          n, k, QMAT_INT4, QMAT_U8_OFF);
        }
    }
    return 0;
}

int mynah_qmat_linear_resolved(mynah_qmat_cache *cache,
                      const mynah_backend *backend, const char *name,
                      const float *weight_data,
                      const float *in, float *out, size_t count, size_t k, size_t n,
                      const float *bias, char *error, size_t error_capacity) {
    return mynah_qmat_linear_resolved_qt(cache, backend, name, weight_data, in, out,
                                         count, k, n, bias, -1, error,
                                         error_capacity);
}

/*
 * The batched linear with the encoding named per tensor instead of per cache --
 * the weight-stationary twin of `mynah_qmat_linear_resolved_qt`, and the same
 * relationship to `mynah_qmat_linear_batched` that that function has to
 * `mynah_qmat_linear_resolved`.
 *
 * WHY IT HAD TO EXIST (E8-5).  `mynah_qmat_linear_batched` takes no qtype: it
 * gates on `cache->qtype` and, when it is the first caller to touch a tensor,
 * creates the cache entry in the CACHE's encoding.  For a group carrying an
 * explicit encoding that differs, that is a first-touch race deciding the
 * group's precision -- so `engine_pocket.c` kept such a group off the batched
 * path entirely rather than allow it.  The consequence was concrete: under
 * `MYNAH_QUANT=int8` the PocketTTS backbone and flow head, which carry `:f16`
 * in the shipped group spec, fell off the weight-stationary path and read the
 * weight once per row.
 *
 * The property that made the refusal necessary is KEPT, not traded away.
 * `want` is resolved from the group spec before anything touches the cache, and
 * it is what both the gate and `cache_insert` use, so precision is decided by
 * the spec and never by whichever caller arrived first.  A tensor name belongs
 * to exactly one group, so an entry's encoding still never changes under it.
 *
 * `qtype`: 0 f32 (exact, no cache entry), 1 int8, 2 int4, 3 f16, or -1 for
 * "whatever the cache resolved to", which is what `mynah_qmat_linear_batched`
 * passes -- so every existing caller keeps its exact behaviour.
 *
 * The lookup deliberately uses the entry it FINDS rather than insisting it
 * match `want`: `_resolved_qt` does the same, and row b must come out bit
 * identical to row b computed alone.  Agreeing with the unbatched path matters
 * more than being right about a disagreement that the one-name-one-group
 * invariant already rules out.
 */
int mynah_qmat_linear_batched_qt(mynah_qmat_cache *cache,
                                 const mynah_backend *backend, const char *name,
                                 const float *weight_data,
                                 const float *const *in_rows,
                                 float *const *out_rows, size_t batch, size_t k,
                                 size_t n, const float *bias, int8_t *qx_scratch,
                                 float *sx_scratch, int qtype, char *error,
                                 size_t error_capacity) {
    if (batch == 0u) return 0;
    if (batch == 1u) {
        return mynah_qmat_linear_resolved_qt(cache, backend, name, weight_data,
                                             in_rows[0], out_rows[0], 1u, k, n,
                                             bias, qtype, error, error_capacity);
    }
    /* Resolved BEFORE the cache is touched, and used for both the gate and the
     * insert: that pair is the whole of E8-5. */
    const int want = (qtype < 0) ? (cache == NULL ? QMAT_F32 : cache->qtype)
                                 : qmat_qtype_available(qtype);
    const int use_q = cache != NULL && want != QMAT_F32 && k <= QMAT_K_MAX &&
                      cache->use_row4 && qx_scratch != NULL && sx_scratch != NULL;
    const qmat_entry *e = NULL;
    if (use_q) {
        pthread_mutex_lock(&cache->mutex);
        e = cache_lookup(cache, name);
        if (e == NULL) e = cache_insert(cache, name, weight_data, n, k, want);
        if (e != NULL) ++cache->batched_calls;
        pthread_mutex_unlock(&cache->mutex);
    }
    if (e == NULL) {
        /* No bit-exact batching available here: keep every row on the exact
         * path it would have taken alone, in the encoding its group asked for. */
        for (size_t b = 0; b < batch; ++b) {
            if (mynah_qmat_linear_resolved_qt(cache, backend, name, weight_data,
                                              in_rows[b], out_rows[b], 1u, k, n,
                                              bias, qtype, error,
                                              error_capacity) != 0) {
                return -1;
            }
        }
        return 0;
    }
#if defined(MYNAH_QMAT_F16)
    const int is_f16 = e->qtype == QMAT_F16;
#else
    const int is_f16 = 0;
#endif
    /* The same resolution the unbatched path makes, so a row cannot change
     * encoding just because it was batched. */
    const int level = e->qtype == QMAT_INT8 ? qmat_u8_level() : QMAT_U8_OFF;
    if (!is_f16) {
        /* F16 keeps the activation in f32 and needs no quantization pass. */
        for (size_t b = 0; b < batch; ++b) {
            sx_scratch[b] = quantize_act(qx_scratch + b * k, in_rows[b], k, level);
        }
    }
    const qmat_batch_job job = {
        e, is_f16 ? NULL : (const void *)qx_scratch, is_f16 ? in_rows : NULL,
        is_f16 ? NULL : sx_scratch, out_rows, bias, batch, n, k, level
    };
    size_t weight_bytes;
    if (e->qtype == QMAT_INT8) weight_bytes = n * k;
    else if (is_f16) weight_bytes = n * k * 2u;
    else weight_bytes = n * k / 2u;
    if (mynah_num_threads() > 1 && weight_bytes >= QMAT_THREAD_MIN_BYTES &&
        n > QMAT_ROW_BLOCK) {
        const size_t blocks = (n + QMAT_ROW_BLOCK - 1u) / QMAT_ROW_BLOCK;
        if (blocks <= (size_t)INT_MAX) {
            mynah_parallel_for((int)blocks, qmat_batch_block, (void *)&job);
            return 0;
        }
    }
    /* Serial, but still blocked: walking the whole matrix once per row would
     * defeat the point of batching. */
    for (size_t row0 = 0; row0 < n; row0 += QMAT_ROW_BLOCK) {
        size_t count = n - row0;
        if (count > QMAT_ROW_BLOCK) count = QMAT_ROW_BLOCK;
        qmat_batch_rows(&job, row0, count);
    }
    return 0;
}

int mynah_qmat_linear_batched(mynah_qmat_cache *cache, const mynah_backend *backend,
                              const char *name, const float *weight_data,
                              const float *const *in_rows, float *const *out_rows,
                              size_t batch, size_t k, size_t n, const float *bias,
                              int8_t *qx_scratch, float *sx_scratch,
                              char *error, size_t error_capacity) {
    return mynah_qmat_linear_batched_qt(cache, backend, name, weight_data, in_rows,
                                        out_rows, batch, k, n, bias, qx_scratch,
                                        sx_scratch, -1, error, error_capacity);
}

int mynah_qmat_linear(mynah_qmat_cache *cache, const mynah_weights *file,
                      const mynah_backend *backend, const char *name,
                      const float *in, float *out, size_t count, size_t k, size_t n,
                      const float *bias, char *error, size_t error_capacity) {
    mynah_tensor weight;
    if (mynah_weights_get(file, name, &weight) != 0) {
        snprintf(error, error_capacity, "model tensor is missing: %s", name);
        return -1;
    }
    return mynah_qmat_linear_resolved(cache, backend, name, weight.data,
                                      in, out, count, k, n, bias,
                                      error, error_capacity);
}

static int self_test_one(int qtype, char *error, size_t error_capacity) {
    enum { N = 48, K = 768, MATVEC_ROWS = 5 };
    static float w[N * K];
    static float x[K];
    static int8_t q8[N * K];
    static uint8_t q4[N * K / 2];
    static float scales8[N];
    static float scales4[N * K / QMAT_Q4_GROUP];
    static int8_t qx[K];
    float bias[MATVEC_ROWS];
    float matvec[MATVEC_ROWS];
    for (size_t i = 0; i < (size_t)N; ++i) {
        for (size_t j = 0; j < (size_t)K; ++j) {
            w[i * K + j] = sinf(0.017f * (float)(i * 7u + j)) * (0.5f + 0.5f * cosf(0.003f * (float)j));
        }
    }
    for (size_t j = 0; j < (size_t)K; ++j) x[j] = cosf(0.011f * (float)j) - 0.3f;
    const float sx = quantize_act_int8(qx, x, K);
    for (size_t i = 0; i < MATVEC_ROWS; ++i) bias[i] = (float)i * 0.125f - 0.25f;
    if (qtype == QMAT_INT8) {
        quantize_weight_int8(w, N, K, q8, scales8);
        matvec_q8(matvec, qx, sx, q8, scales8, NULL, bias, MATVEC_ROWS, K,
                  QMAT_U8_OFF);
    } else {
        quantize_weight_int4(w, N, K, q4, scales4);
        matvec_q4(matvec, qx, sx, q4, scales4, bias, MATVEC_ROWS, K);
    }
    float max_rel = 0.0f;
    float ref_energy = 0.0f;
    for (size_t i = 0; i < (size_t)N; ++i) {
        float ref = 0.0f;
        for (size_t j = 0; j < (size_t)K; ++j) ref += w[i * K + j] * x[j];
        const float got = qtype == QMAT_INT8
                              ? dot_q8(qx, sx, q8 + i * K, scales8[i], 0, K,
                                       QMAT_U8_OFF, 0.0f)
                              : dot_q4(qx, sx, q4 + i * (K / 2),
                                       scales4 + i * (K / QMAT_Q4_GROUP), K, 0.0f);
        if (i < MATVEC_ROWS) {
            const float expected = got + bias[i];
            const float tolerance = 1.0e-6f * (1.0f + fabsf(expected));
            if (fabsf(matvec[i] - expected) > tolerance) {
                if (error != NULL && error_capacity > 0) {
                    snprintf(error, error_capacity,
                             "qmat %s four-row matvec mismatch at row %zu",
                             qtype == QMAT_INT8 ? "int8" : "int4", i);
                }
                return -1;
            }
        }
        ref_energy += ref * ref;
        const float denom = fabsf(ref) > 1.0e-3f ? fabsf(ref) : 1.0e-3f;
        const float rel = fabsf(got - ref) / denom;
        if (rel > max_rel) max_rel = rel;
    }
    /* INT8 ~ a few percent; INT4 is coarser (16 levels/group). */
    const float limit = qtype == QMAT_INT8 ? 0.05f : 0.20f;
    if (!(ref_energy > 0.0f) || max_rel > limit) {
        if (error != NULL && error_capacity > 0) {
            snprintf(error, error_capacity, "qmat %s relative error too large: %.4f",
                     qtype == QMAT_INT8 ? "int8" : "int4", (double)max_rel);
        }
        return -1;
    }
    return 0;
}

/* The row-blocked dispatch must be bit-exact against a single serial call:
 * same kernel, same four-row unroll inside every block, no reduction split
 * across a block boundary.  The blocks are walked directly rather than through
 * the pool so the partition arithmetic (offsets into weights/scales/bias and
 * the short trailing block) is checked at any thread count.  N is deliberately
 * not a multiple of QMAT_ROW_BLOCK so the remainder path is exercised. */
static int self_test_rows_blocked(int qtype, char *error, size_t error_capacity) {
    enum { N = 200, K = 256 };
    int status = -1;
    float *w = malloc((size_t)N * K * sizeof(float));
    float *x = malloc((size_t)K * sizeof(float));
    float *bias = malloc((size_t)N * sizeof(float));
    float *serial = malloc((size_t)N * sizeof(float));
    float *blocked = malloc((size_t)N * sizeof(float));
    int8_t *qx = malloc((size_t)K);
    int8_t *q8 = malloc((size_t)N * K);
    uint8_t *q4 = malloc((size_t)N * K / 2u);
    float *scales8 = malloc((size_t)N * sizeof(float));
    float *scales4 = malloc((size_t)N * (K / QMAT_Q4_GROUP) * sizeof(float));
#if defined(MYNAH_QMAT_F16)
    uint16_t *h16 = malloc((size_t)N * K * sizeof(uint16_t));
    const int h16_ok = h16 != NULL;
#else
    void *h16 = NULL;
    const int h16_ok = 1;   /* unused on targets without half converts */
#endif
    if (w == NULL || x == NULL || bias == NULL || serial == NULL || blocked == NULL ||
        qx == NULL || q8 == NULL || q4 == NULL || scales8 == NULL || scales4 == NULL ||
        !h16_ok) {
        if (error != NULL && error_capacity > 0)
            snprintf(error, error_capacity, "qmat blocked self-test out of memory");
        goto done;
    }
    for (size_t i = 0; i < (size_t)N * K; ++i)
        w[i] = sinf(0.013f * (float)i) * (0.5f + 0.5f * cosf(0.0007f * (float)i));
    for (size_t j = 0; j < (size_t)K; ++j) x[j] = cosf(0.011f * (float)j) - 0.3f;
    for (size_t i = 0; i < (size_t)N; ++i) bias[i] = (float)i * 0.03125f - 0.5f;
    const float sx = quantize_act_int8(qx, x, K);
    const void *weights;
    const float *scales;
    if (qtype == QMAT_INT8) {
        quantize_weight_int8(w, N, K, q8, scales8);
        weights = q8;
        scales = scales8;
        matvec_q8(serial, qx, sx, q8, scales8, NULL, bias, N, K, QMAT_U8_OFF);
#if defined(MYNAH_QMAT_F16)
    } else if (qtype == QMAT_F16) {
        qmat_f16_pack(h16, w, (size_t)N * K);
        weights = h16;
        scales = NULL;
        matvec_f16(serial, x, h16, bias, N, K);
#endif
    } else {
        quantize_weight_int4(w, N, K, q4, scales4);
        weights = q4;
        scales = scales4;
        matvec_q4(serial, qx, sx, q4, scales4, bias, N, K);
    }
    for (size_t i = 0; i < (size_t)N; ++i) blocked[i] = 0.0f;
    const qmat_rows_job job = {blocked, qx, x, sx, weights, scales, NULL, bias,
                               (size_t)N, (size_t)K, qtype, QMAT_U8_OFF};
    const int blocks = (int)(((size_t)N + QMAT_ROW_BLOCK - 1u) / QMAT_ROW_BLOCK);
    for (int b = 0; b < blocks; ++b) qmat_rows_block((void *)&job, b);
    /* Compare with a tight tolerance rather than memcmp.  The split performs
     * the same arithmetic per row -- a wrong partition would offset weights or
     * scales and be off by a wide margin, which this still catches -- but the
     * build uses -ffast-math (and -mfma on x86), so the compiler may contract
     * or reassociate the final scale/bias differently for a block of 32 rows
     * than for one call of 200, giving last-bit differences.  Bit-identical
     * output was verified end-to-end on macOS/Accelerate; it is not something
     * these flags guarantee across compilers. */
    for (size_t i = 0; i < (size_t)N; ++i) {
        const float denom = fabsf(serial[i]) > 1.0e-3f ? fabsf(serial[i]) : 1.0e-3f;
        if (fabsf(serial[i] - blocked[i]) / denom > 1.0e-6f) {
            if (error != NULL && error_capacity > 0) {
                snprintf(error, error_capacity,
                         "qmat %s blocked matvec differs from serial at row %zu"
                         " (%.9g vs %.9g)",
                         qtype == QMAT_INT8 ? "int8"
                             : (qtype == QMAT_F16 ? "f16" : "int4"), i,
                         (double)serial[i], (double)blocked[i]);
            }
            goto done;
        }
    }
    status = 0;
done:
    free(w); free(x); free(bias); free(serial); free(blocked);
    free(qx); free(q8); free(q4); free(scales8); free(scales4); free(h16);
    return status;
}

#if defined(MYNAH_QMAT_F16)
/* The weights a host produces must not depend on which convert instruction it
 * has.  A scalar-only x86 box and an F16C one load the same checkpoint; if
 * their rounding disagreed, the same pack would quantize two ways and a parity
 * run would chase a phantom.  So this asserts BIT equality between
 * qmat_f16_pack() and the scalar reference, over a sweep chosen for the places
 * naive converters break: the subnormal band, the 2^-25 flush point, the
 * 65504/65520 overflow cliff and exact ties. */
static int self_test_f16_convert(char *error, size_t error_capacity) {
    static const float cases[] = {
        0.0f, -0.0f, 1.0f, -1.0f, 0.5f, 2.0f, 65504.0f, -65504.0f,
        65519.0f, 65520.0f, 65536.0f, 1.0e30f, -1.0e30f,
        6.10352e-05f, 6.09756e-05f, 3.0517578125e-05f, 1.52587890625e-05f,
        5.960464477539063e-08f, 2.980232238769531e-08f, 2.9802320e-08f,
        1.0e-10f, -1.0e-10f, 1.0009765625f, 1.00048828125f, 1.0004883f,
        2049.0f, 2050.0f, 2051.0f, 0.333333343f, -0.333333343f
    };
    enum { EXTRA = 4096 };
    const size_t fixed = sizeof cases / sizeof cases[0];
    const size_t total = fixed + (size_t)EXTRA;
    float *src = (float *)malloc(total * sizeof(float));
    uint16_t *packed = (uint16_t *)malloc(total * sizeof(uint16_t));
    int status = -1;
    if (src == NULL || packed == NULL) {
        if (error != NULL && error_capacity > 0)
            snprintf(error, error_capacity, "qmat f16 convert self-test OOM");
        goto done;
    }
    for (size_t i = 0; i < fixed; ++i) src[i] = cases[i];
    for (size_t i = 0; i < (size_t)EXTRA; ++i) {
        /* Sweeps fourteen decades, both signs, so the normal band, the
         * subnormal band and the ties between them are all crossed. */
        const float t = (float)i / (float)EXTRA;
        src[fixed + i] = (i & 1u ? -1.0f : 1.0f) *
                         (float)pow(10.0, -9.0 + 14.0 * (double)t) *
                         (1.0f + 0.5f * sinf(37.0f * t));
    }
    qmat_f16_pack(packed, src, total);
    for (size_t i = 0; i < total; ++i) {
        const uint16_t want = qmat_f16_from_f32(src[i]);
        if (packed[i] != want) {
            if (error != NULL && error_capacity > 0) {
                snprintf(error, error_capacity,
                         "qmat f16 pack disagrees with the scalar reference at "
                         "%zu: %.9g -> 0x%04x, reference 0x%04x",
                         i, (double)src[i], (unsigned)packed[i], (unsigned)want);
            }
            goto done;
        }
        /* And the round trip must recover the value the half actually holds,
         * so a broken widening cannot hide behind a matching narrowing. */
        const float back = qmat_f16_to_f32(packed[i]);
        if (fabsf(src[i]) <= 65504.0f && fabsf(src[i]) >= 1.0e-4f) {
            if (fabsf(back - src[i]) / fabsf(src[i]) > 1.0e-2f) {
                if (error != NULL && error_capacity > 0) {
                    snprintf(error, error_capacity,
                             "qmat f16 round trip lost too much at %zu: "
                             "%.9g -> %.9g", i, (double)src[i], (double)back);
                }
                goto done;
            }
        }
    }
    status = 0;
done:
    free(src);
    free(packed);
    return status;
}

/* F16 keeps the activation exact and only rounds the weights, so it must land
 * far closer to the f32 reference than INT8 does.  The bound below is ~50x
 * tighter than the INT8 one; if a change loosens it, the accuracy argument for
 * preferring F16 over INT8 no longer holds. */
static int self_test_f16(char *error, size_t error_capacity) {
    enum { N = 64, K = 512 };
    int status = -1;
    float *w = malloc((size_t)N * K * sizeof(float));
    float *x = malloc((size_t)K * sizeof(float));
    float *bias = malloc((size_t)N * sizeof(float));
    float *got = malloc((size_t)N * sizeof(float));
    uint16_t *h = malloc((size_t)N * K * sizeof(uint16_t));
    if (w == NULL || x == NULL || bias == NULL || got == NULL || h == NULL) {
        if (error != NULL && error_capacity > 0)
            snprintf(error, error_capacity, "qmat f16 self-test out of memory");
        goto done;
    }
    for (size_t i = 0; i < (size_t)N * K; ++i) {
        w[i] = sinf(0.017f * (float)i) * (0.5f + 0.5f * cosf(0.003f * (float)i));
    }
    qmat_f16_pack(h, w, (size_t)N * K);
    for (size_t j = 0; j < (size_t)K; ++j) x[j] = cosf(0.011f * (float)j) - 0.3f;
    for (size_t i = 0; i < (size_t)N; ++i) bias[i] = (float)i * 0.125f - 0.25f;
    matvec_f16(got, x, h, bias, N, K);
    float max_rel = 0.0f;
    for (size_t i = 0; i < (size_t)N; ++i) {
        double ref = 0.0;
        for (size_t j = 0; j < (size_t)K; ++j) ref += (double)w[i * K + j] * x[j];
        ref += bias[i];
        const double denom = fabs(ref) > 1.0e-3 ? fabs(ref) : 1.0e-3;
        const float rel = (float)(fabs(got[i] - ref) / denom);
        if (rel > max_rel) max_rel = rel;
    }
    if (!(max_rel <= 1.0e-3f)) {
        if (error != NULL && error_capacity > 0) {
            snprintf(error, error_capacity,
                     "qmat f16 relative error too large: %.5f", (double)max_rel);
        }
        goto done;
    }
    status = 0;
done:
    free(w); free(x); free(bias); free(got); free(h);
    return status;
}
#endif

/* The batching contract: batching must not change a row's result.  If it did,
 * a request's audio would depend on which other requests happened to be in
 * flight with it, which is a nondeterminism regression no tolerance can excuse.
 * So this asserts bit-equality, not closeness, against the same rows computed
 * one at a time. */
static int self_test_batched(int qtype, char *error, size_t error_capacity) {
    enum { N = 96, K = 256, B = 5 };
    int status = -1;
    mynah_qmat_cache *cache = mynah_qmat_cache_new(qtype);
    if (cache == NULL) {
        if (error != NULL && error_capacity > 0)
            snprintf(error, error_capacity, "qmat batched self-test out of memory");
        return -1;
    }
    if (cache->qtype != qtype) {
        /* Unsupported on this target (no half converts): nothing to check. */
        mynah_qmat_cache_free(cache);
        return 0;
    }
    float *w = (float *)malloc((size_t)N * K * sizeof(float));
    float *x = (float *)malloc((size_t)B * K * sizeof(float));
    float *bias = (float *)malloc((size_t)N * sizeof(float));
    float *ref = (float *)malloc((size_t)B * N * sizeof(float));
    float *got = (float *)malloc((size_t)B * N * sizeof(float));
    int8_t *qx = (int8_t *)malloc((size_t)B * K);
    float *sx = (float *)malloc((size_t)B * sizeof(float));
    if (w == NULL || x == NULL || bias == NULL || ref == NULL || got == NULL ||
        qx == NULL || sx == NULL) {
        if (error != NULL && error_capacity > 0)
            snprintf(error, error_capacity, "qmat batched self-test out of memory");
        goto done;
    }
    for (size_t i = 0; i < (size_t)N * K; ++i)
        w[i] = sinf(0.017f * (float)i) * (0.5f + 0.5f * cosf(0.0011f * (float)i));
    for (size_t i = 0; i < (size_t)B * K; ++i) x[i] = cosf(0.019f * (float)i) - 0.25f;
    for (size_t i = 0; i < (size_t)N; ++i) bias[i] = (float)i * 0.015625f - 0.75f;
    const float *in_rows[B];
    float *out_rows[B];
    for (size_t b = 0; b < (size_t)B; ++b) {
        in_rows[b] = x + b * K;
        out_rows[b] = got + b * N;
    }
    for (size_t b = 0; b < (size_t)B; ++b) {
        if (mynah_qmat_linear_resolved(cache, NULL, "batched.self.test", w,
                                       x + b * K, ref + b * N, 1u, K, N, bias,
                                       error, error_capacity) != 0) {
            goto done;
        }
    }
    if (mynah_qmat_linear_batched(cache, NULL, "batched.self.test", w,
                                  in_rows, out_rows, B, K, N, bias, qx, sx,
                                  error, error_capacity) != 0) {
        goto done;
    }
    for (size_t b = 0; b < (size_t)B; ++b) {
        for (size_t i = 0; i < (size_t)N; ++i) {
            const size_t at = b * N + i;
            if (memcmp(&ref[at], &got[at], sizeof(float)) != 0) {
                if (error != NULL && error_capacity > 0)
                    snprintf(error, error_capacity,
                             "qmat batched qtype=%d differs at row %zu col %zu: "
                             "%.9g vs %.9g", qtype, b, i,
                             (double)ref[at], (double)got[at]);
                goto done;
            }
        }
    }
    status = 0;
done:
    free(w); free(x); free(bias); free(ref); free(got); free(qx); free(sx);
    mynah_qmat_cache_free(cache);
    return status;
}

/*
 * E8-5.  The same contract for a group that names its OWN encoding, plus the
 * one hazard that kept such a group off the batched path until now.
 *
 * The experiment is built so that the batched call is the FIRST toucher of its
 * tensor: `batched.qt.ref` is quantized by the per-row reference,
 * `batched.qt.first` by the batched call alone.  If the batched path created
 * its entry in the CACHE's encoding rather than the GROUP's -- which is exactly
 * what `mynah_qmat_linear_batched` did, and why a group with an explicit
 * encoding had to be refused the batched path -- the two tensors come out in
 * different encodings and this fires.
 *
 * TWO ASSERTIONS, AND THE SECOND ONE IS THE LOAD-BEARING ONE.  Comparing the
 * two outputs alone would pass VACUOUSLY: when `_batched_qt` declines the
 * weight-stationary path it falls back to `_resolved_qt` per row, which is
 * literally the reference, so a byte compare cannot distinguish "batched and
 * correct" from "batched never ran".  So the entry the batched call left in the
 * cache is inspected directly -- it must exist and it must carry `want` -- and
 * that is both the anti-vacuity check and the direct statement of the property.
 * It is skipped only under the two conditions that legitimately disable the
 * path (`MYNAH_QMAT_SINGLE_ROW`, or k over QMAT_K_MAX), which this shape is
 * chosen to stay inside.
 *
 * Both cache profiles the runtime ships are covered: f16, which
 * `mynah_tts.c` selects for a non-Magpie pack when MYNAH_QUANT is unset, and
 * int8, which is the case E8-5 exists for -- there the shipped `:f16` groups
 * differ from the cache and are exactly what used to be refused.  f32 is run
 * too, because a cache that is off is the case where the two encodings differ
 * most.
 */
/* EVERY batch width, because the f16 batched path now has four of them and the
 * existing B=5 test exercises exactly two.
 *
 * The widths are not decoration.  A worker at the shipping topology holds
 * C/workers live slots -- two or three at the concurrencies that matter -- and
 * the AR step batches precisely those, so the remainders are the production
 * case and not the leftovers.  One of them, the three-remainder, is served by
 * the four-lane kernel with the last activation REPEATED: it spends a quarter
 * more arithmetic to load the weight block once instead of twice, which wins
 * because the kernel is memory-bound, and it writes one output address twice
 * with bits that are identical by construction.  That is subtle enough that it
 * must be tested rather than argued, and B=5 would never have reached it.
 *
 * The reference is the row-at-a-time path, compared with memcmp: the claim is
 * bit-identity, not closeness. */
/* The int4 block scale is seeded from the signed extreme and then solved, and
 * this prices BOTH halves separately rather than the pair.
 *
 * Measured over the shapes the suite quantizes: relative reconstruction error
 * 4.182% with the absmax scale against 3.824% solved -- an 8.6% relative
 * reduction for the same bytes, the same layout, the same kernels and the same
 * runtime cost.
 *
 * A first version of this test asserted only solved <= naive, and a mutation
 * that disabled the solve PASSED it: the seeding alone already beats absmax,
 * because int4 here is [-8, 7] and amax/7 throws the -8 away. Hence the middle
 * mode and the two-sided assertion.
 *
 * WHY A WEIGHT METRIC AND NOT AUDIO. Over six seeds the waveform correlation
 * against the f16 gold is 3-3 between the two quantizers, mean +0.0008, inside
 * its own spread -- because POCKET_QG_DEFAULT_SPEC keeps int4 out of the
 * autoregressive loop, so it perturbs only the feed-forward codec and the frame
 * count cannot move. The reference engine's 7-point word-accuracy win came from
 * a configuration where int4 DOES reach the sampler; that does not transfer and
 * is not claimed here. What transfers is a strictly better approximation at no
 * cost, which is what this asserts. */
/* The activation quantizer's vector paths against the scalar reference, BYTE
 * FOR BYTE.  Not a tolerance: an activation byte that depended on the ISA
 * would make a request's audio depend on which machine served it, and the
 * rounding rules were chosen (vcvta = ties away from zero) so that equality is
 * achievable rather than approximately true.
 *
 * The lengths straddle the 16-element vector body on both sides, and the data
 * deliberately includes an exact .5 tie, a value that clamps, a zero vector
 * and a vector whose amax is its last element -- the four inputs where a
 * reduction or a rounding rule can differ and nothing else would show it. */
static int self_test_act_quantize(char *error, size_t error_capacity) {
    static const size_t lens[] = { 1, 4, 15, 16, 17, 31, 32, 33, 64, 127, 512 };
    float x[512];
    int8_t got_s[512], want_s[512];
    uint8_t got_u[512], want_u[512];
    for (size_t li = 0; li < sizeof lens / sizeof lens[0]; ++li) {
        const size_t k = lens[li];
        for (int variant = 0; variant < 4; ++variant) {
            for (size_t i = 0; i < k; ++i) {
                switch (variant) {
                case 0: x[i] = (float)((int)(i % 37u) - 18) * 0.125f; break;
                case 1: x[i] = 0.0f; break;
                /* amax is the LAST element: a tree reduction that dropped the
                 * tail would pass every other variant. */
                case 2: x[i] = (i + 1u == k) ? 9.0f : 0.5f; break;
                /* EXACT TIES.  One element is 127, so amax is 127 and the
                 * scale is exactly 1.0; every other value is a half-integer
                 * and therefore lands exactly on a rounding tie.  Without
                 * this the ties-away-from-zero rule is never exercised and
                 * swapping vcvta for vcvtn passes the whole suite -- it did,
                 * the first time this test was written. */
                default:
                    x[i] = (i + 1u == k)
                               ? 127.0f
                               : ((float)((int)(i % 5u) - 2) + 0.5f);
                    break;
                }
            }
            const float ss = quantize_act_int8_scalar(want_s, x, k);
            const float vs = quantize_act_int8(got_s, x, k);
            if (memcmp(&ss, &vs, sizeof ss) != 0) {
                snprintf(error, error_capacity,
                         "qmat: int8 activation scale differs at k=%zu "
                         "variant %d: scalar %.9g vector %.9g",
                         k, variant, (double)ss, (double)vs);
                return -1;
            }
            if (memcmp(got_s, want_s, k) != 0) {
                /* memcmp said they differ, so this stops before `k` -- but the
                 * bound is written down rather than reasoned about, because the
                 * next line indexes with it. */
                size_t at = 0;
                while (at + 1u < k && got_s[at] == want_s[at]) ++at;
                snprintf(error, error_capacity,
                         "qmat: int8 activation byte %zu of %zu differs "
                         "(variant %d): scalar %d vector %d",
                         at, k, variant, (int)want_s[at], (int)got_s[at]);
                return -1;
            }
            const float su = quantize_act_u8_scalar(want_u, x, k);
            const float vu = quantize_act_u8(got_u, x, k);
            if (memcmp(&su, &vu, sizeof su) != 0 ||
                memcmp(got_u, want_u, k) != 0) {
                size_t at = 0;
                while (at + 1u < k && got_u[at] == want_u[at]) ++at;
                snprintf(error, error_capacity,
                         "qmat: u8 activation differs at %zu of %zu "
                         "(variant %d): scalar %u vector %u",
                         at, k, variant, (unsigned)want_u[at],
                         (unsigned)got_u[at]);
                return -1;
            }
            /* And the invariant that ties the two encodings together, which
             * is what the -128*rowsum correction assumes. */
            for (size_t i = 0; i < k; ++i) {
                if (got_u[i] != (uint8_t)((int)got_s[i] + 128)) {
                    snprintf(error, error_capacity,
                             "qmat: u8 is not int8+128 at %zu of %zu "
                             "(variant %d): %d vs %u",
                             i, k, variant, (int)got_s[i], (unsigned)got_u[i]);
                    return -1;
                }
            }
        }
    }
    return 0;
}

static int self_test_q4_scale_solve(char *error, size_t error_capacity) {
    enum { N = 8, K = 256 };
    if (qmat_qtype_available(QMAT_INT4) != QMAT_INT4) return 0;
    int status = -1;
    const size_t groups = (size_t)K / QMAT_Q4_GROUP;
    float *w = (float *)malloc((size_t)N * K * sizeof(float));
    uint8_t *qq[3] = {NULL, NULL, NULL};
    float *ss[3] = {NULL, NULL, NULL};
    double err[3] = {0.0, 0.0, 0.0};
    if (w == NULL) goto oom;
    for (int m = 0; m < 3; ++m) {
        qq[m] = (uint8_t *)malloc((size_t)N * K / 2u);
        ss[m] = (float *)malloc((size_t)N * groups * sizeof(float));
        if (qq[m] == NULL || ss[m] == NULL) goto oom;
    }
    /* Asymmetric on purpose: a block whose extreme is negative is exactly what
     * amax/7 quantizes with one level fewer than the format offers. */
    for (size_t i = 0; i < (size_t)N * K; ++i) {
        const float t = 0.017f * (float)i;
        w[i] = sinf(t) * (1.0f + cosf(0.37f * t)) - 0.35f;
    }
    for (int m = 0; m < 3; ++m) {
        quantize_weight_int4_mode(w, N, K, qq[m], ss[m], m);
        for (size_t i = 0; i < (size_t)N; ++i) {
            for (size_t g = 0; g < groups; ++g) {
                for (size_t j = 0; j < QMAT_Q4_GROUP; ++j) {
                    const size_t at = g * QMAT_Q4_GROUP + j;
                    const size_t byte = i * ((size_t)K / 2u) + at / 2u;
                    const int nib = (int)((at % 2u) ? (qq[m][byte] >> 4)
                                                    : (qq[m][byte] & 0x0fu)) - 8;
                    const double d = (double)w[i * (size_t)K + at] -
                                     (double)ss[m][i * groups + g] * (double)nib;
                    err[m] += d * d;
                }
            }
        }
    }
    /* ABSOLUTE BOUNDS FIRST, and they are the half that works.
     *
     * The ordering checks below compare the three modes against each other, and
     * a mutation that degrades all three symmetrically is invisible to them --
     * measured, not supposed: disabling the solve and dropping the seeding each
     * passed an ordering-only version of this test, because they moved the
     * baseline along with the subject. Pinning the numbers closes that.
     *
     * On this fixed synthetic tensor the three are 3.9429% / 4.0516% / 4.3156%
     * relative reconstruction error. The bounds sit just above the first two
     * with roughly 1.5% of headroom, which is deterministic arithmetic on a
     * deterministic input and not a tolerance for noise. */
    {
        double ref = 0.0;
        for (size_t i = 0; i < (size_t)N * K; ++i) ref += (double)w[i] * (double)w[i];
        const double solved = 100.0 * sqrt(err[0] / ref);
        const double seeded = 100.0 * sqrt(err[2] / ref);
        if (!(solved <= 4.00)) {
            if (error != NULL && error_capacity > 0)
                snprintf(error, error_capacity,
                         "qmat int4: solved block scale reconstructs at %.4f%%, "
                         "above the 4.00%% this tensor has measured at 3.9429%% "
                         "-- the closed form or its seeding regressed", solved);
            goto done;
        }
        if (!(seeded <= 4.20)) {
            if (error != NULL && error_capacity > 0)
                snprintf(error, error_capacity,
                         "qmat int4: seeding from the signed extreme reconstructs "
                         "at %.4f%%, above the 4.20%% this tensor has measured at "
                         "4.0516%% -- the -8 level is being thrown away again",
                         seeded);
            goto done;
        }
    }
    /* Then the ordering: solved <= seeded <= naive, each half earning its place. */
    if (!(err[2] <= err[1])) {
        if (error != NULL && error_capacity > 0)
            snprintf(error, error_capacity,
                     "qmat int4: seeding from the signed extreme reconstructs "
                     "WORSE than absmax (%.9g vs %.9g)", err[2], err[1]);
        goto done;
    }
    if (!(err[0] <= err[2])) {
        if (error != NULL && error_capacity > 0)
            snprintf(error, error_capacity,
                     "qmat int4: solving the block scale reconstructs WORSE than "
                     "the seed alone (%.9g vs %.9g) -- the closed form regressed",
                     err[0], err[2]);
        goto done;
    }
    status = 0;
    goto done;
oom:
    if (error != NULL && error_capacity > 0)
        snprintf(error, error_capacity, "qmat q4-scale self-test out of memory");
done:
    free(w);
    for (int m = 0; m < 3; ++m) { free(qq[m]); free(ss[m]); }
    return status;
}

static int self_test_lane_widths(int qtype, char *error, size_t error_capacity) {
    enum { N = 96, K = 256, BMAX = 9 };
    if (qmat_qtype_available(qtype) != qtype) return 0;
    int status = -1;
    mynah_qmat_cache *cache = mynah_qmat_cache_new(qtype);
    if (cache == NULL) return -1;
    if (cache->qtype != qtype) { mynah_qmat_cache_free(cache); return 0; }

    float *w = (float *)malloc((size_t)N * K * sizeof(float));
    float *x = (float *)malloc((size_t)BMAX * K * sizeof(float));
    float *bias = (float *)malloc((size_t)N * sizeof(float));
    float *ref = (float *)malloc((size_t)BMAX * N * sizeof(float));
    float *got = (float *)malloc((size_t)BMAX * N * sizeof(float));
    int8_t *qx = (int8_t *)malloc((size_t)BMAX * K);
    float *sx = (float *)malloc((size_t)BMAX * sizeof(float));
    if (w == NULL || x == NULL || bias == NULL || ref == NULL || got == NULL ||
        qx == NULL || sx == NULL) {
        if (error != NULL && error_capacity > 0)
            snprintf(error, error_capacity, "qmat lane-width self-test out of memory");
        goto done;
    }
    for (size_t i = 0; i < (size_t)N * K; ++i)
        w[i] = sinf(0.011f * (float)i) * (0.5f + 0.5f * cosf(0.0009f * (float)i));
    for (size_t i = 0; i < (size_t)BMAX * K; ++i)
        x[i] = cosf(0.019f * (float)i) - 0.0625f;
    for (size_t i = 0; i < (size_t)N; ++i) bias[i] = 0.25f - (float)i * 0.00390625f;

    for (size_t b = 0; b < (size_t)BMAX; ++b) {
        if (mynah_qmat_linear_resolved_qt(cache, NULL, "lanes.ref", w,
                                          x + b * K, ref + b * N, 1u, K, N, bias,
                                          qtype, error, error_capacity) != 0) {
            goto done;
        }
    }
    for (size_t batch = 1; batch <= (size_t)BMAX; ++batch) {
        const float *in_rows[BMAX];
        float *out_rows[BMAX];
        for (size_t b = 0; b < batch; ++b) {
            in_rows[b] = x + b * K;
            out_rows[b] = got + b * N;
        }
        memset(got, 0, (size_t)BMAX * N * sizeof(float));
        if (mynah_qmat_linear_batched_qt(cache, NULL, "lanes.ref", w, in_rows,
                                         out_rows, batch, K, N, bias, qx, sx,
                                         qtype, error, error_capacity) != 0) {
            goto done;
        }
        for (size_t b = 0; b < batch; ++b) {
            for (size_t i = 0; i < (size_t)N; ++i) {
                const size_t at = b * N + i;
                if (memcmp(&ref[at], &got[at], sizeof(float)) != 0) {
                    if (error != NULL && error_capacity > 0)
                        snprintf(error, error_capacity,
                                 "qmat %s batch=%zu differs from the row-at-a-time "
                                 "reference at row %zu col %zu: %.9g vs %.9g -- a "
                                 "lane width changed a row's answer",
                                 mynah_qmat_qtype_name(qtype), batch, b, i,
                                 (double)ref[at], (double)got[at]);
                    goto done;
                }
            }
        }
    }
    status = 0;
done:
    free(w); free(x); free(bias); free(ref); free(got); free(qx); free(sx);
    mynah_qmat_cache_free(cache);
    return status;
}

static int self_test_batched_qt(int cache_qtype, int group_qtype, char *error,
                                size_t error_capacity) {
    enum { N = 96, K = 256, B = 5 };
    int status = -1;
    const int want = qmat_qtype_available(group_qtype);
    if (want != group_qtype) return 0; /* this build cannot honour it */
    mynah_qmat_cache *cache = mynah_qmat_cache_new(cache_qtype);
    if (cache == NULL) {
        if (error != NULL && error_capacity > 0)
            snprintf(error, error_capacity, "qmat batched-qt self-test out of memory");
        return -1;
    }
    if (cache->qtype != cache_qtype) {
        /* Unsupported on this target (no half converts): nothing to check. */
        mynah_qmat_cache_free(cache);
        return 0;
    }
    float *w = (float *)malloc((size_t)N * K * sizeof(float));
    float *x = (float *)malloc((size_t)B * K * sizeof(float));
    float *bias = (float *)malloc((size_t)N * sizeof(float));
    float *ref = (float *)malloc((size_t)B * N * sizeof(float));
    float *got = (float *)malloc((size_t)B * N * sizeof(float));
    int8_t *qx = (int8_t *)malloc((size_t)B * K);
    float *sx = (float *)malloc((size_t)B * sizeof(float));
    if (w == NULL || x == NULL || bias == NULL || ref == NULL || got == NULL ||
        qx == NULL || sx == NULL) {
        if (error != NULL && error_capacity > 0)
            snprintf(error, error_capacity, "qmat batched-qt self-test out of memory");
        goto done;
    }
    for (size_t i = 0; i < (size_t)N * K; ++i)
        w[i] = sinf(0.013f * (float)i) * (0.5f + 0.5f * cosf(0.0007f * (float)i));
    for (size_t i = 0; i < (size_t)B * K; ++i) x[i] = cosf(0.023f * (float)i) + 0.125f;
    for (size_t i = 0; i < (size_t)N; ++i) bias[i] = 0.5f - (float)i * 0.0078125f;

    const float *in_rows[B];
    float *out_rows[B];
    for (size_t b = 0; b < (size_t)B; ++b) {
        in_rows[b] = x + b * K;
        out_rows[b] = got + b * N;
    }
    for (size_t b = 0; b < (size_t)B; ++b) {
        if (mynah_qmat_linear_resolved_qt(cache, NULL, "batched.qt.ref", w,
                                          x + b * K, ref + b * N, 1u, K, N, bias,
                                          group_qtype, error, error_capacity) != 0) {
            goto done;
        }
    }
    /*
     * THREE SHAPES, because `_batched_qt` has three exits and only one of them
     * is the weight-stationary kernel.  A test that exercised the wide call
     * alone would pass while the qtype was dropped from either of the other
     * two -- both measured: mutating the batch==1 delegate or the fallback loop
     * to pass -1 leaves a B-row-only test green.
     *
     *   wide      B rows with scratch  -> the weight-stationary kernel
     *   fallback  B rows, scratch NULL -> one `_resolved_qt` per row
     *   single    1 row                -> the batch==1 delegate
     *
     * Each gets its own tensor name so that in every one of them the call under
     * test is the tensor's FIRST toucher, which is the condition the whole item
     * is about.
     */
    static const struct {
        const char *name;
        size_t rows;
        int stationary;
    } shapes[3] = {
        {"batched.qt.wide", (size_t)B, 1},
        {"batched.qt.fallback", (size_t)B, 0},
        {"batched.qt.single", 1u, 0},
    };
    for (size_t sh = 0; sh < 3u; ++sh) {
        pthread_mutex_lock(&cache->mutex);
        const size_t before = cache->batched_calls;
        pthread_mutex_unlock(&cache->mutex);
        memset(got, 0, (size_t)B * N * sizeof(float));
        if (mynah_qmat_linear_batched_qt(cache, NULL, shapes[sh].name, w, in_rows,
                                         out_rows, shapes[sh].rows, K, N, bias,
                                         shapes[sh].stationary ? qx : NULL,
                                         shapes[sh].stationary ? sx : NULL,
                                         group_qtype, error, error_capacity) != 0) {
            goto done;
        }
        /* Anti-vacuity, and the property itself.  The byte compare below cannot
         * tell "batched and correct" from "batched never ran" -- the fallback
         * produces identical bytes AND leaves an identically-encoded entry --
         * so which path ran is asserted directly, in both directions. */
        pthread_mutex_lock(&cache->mutex);
        const qmat_entry *made = cache_lookup(cache, shapes[sh].name);
        const int made_qtype = (made == NULL) ? -1 : made->qtype;
        const size_t engaged = cache->batched_calls - before;
        pthread_mutex_unlock(&cache->mutex);
        const int reachable = cache->use_row4 && (size_t)K <= (size_t)QMAT_K_MAX;
        if (reachable && shapes[sh].stationary && engaged == 0u) {
            if (error != NULL && error_capacity > 0)
                snprintf(error, error_capacity,
                         "qmat batched_qt cache=%s group=%s (%s): the "
                         "weight-stationary path never ran -- the group was "
                         "refused the batched path and every row read the weight "
                         "on its own, which is the regression E8-5 removed",
                         mynah_qmat_qtype_name(cache_qtype),
                         mynah_qmat_qtype_name(want), shapes[sh].name);
            goto done;
        }
        if (!shapes[sh].stationary && engaged != 0u) {
            if (error != NULL && error_capacity > 0)
                snprintf(error, error_capacity,
                         "qmat batched_qt cache=%s group=%s (%s): took the "
                         "weight-stationary path with no scratch to do it with",
                         mynah_qmat_qtype_name(cache_qtype),
                         mynah_qmat_qtype_name(want), shapes[sh].name);
            goto done;
        }
        if (made_qtype != want) {
            if (error != NULL && error_capacity > 0)
                snprintf(error, error_capacity,
                         "qmat batched_qt cache=%s group=%s (%s): the call left "
                         "the tensor in encoding %s, not the group's -- precision "
                         "was decided by the cache, not by the spec",
                         mynah_qmat_qtype_name(cache_qtype),
                         mynah_qmat_qtype_name(want), shapes[sh].name,
                         made == NULL ? "none" : mynah_qmat_qtype_name(made_qtype));
            goto done;
        }
        for (size_t b = 0; b < shapes[sh].rows; ++b) {
            for (size_t i = 0; i < (size_t)N; ++i) {
                const size_t at = b * N + i;
                if (memcmp(&ref[at], &got[at], sizeof(float)) != 0) {
                    if (error != NULL && error_capacity > 0)
                        snprintf(error, error_capacity,
                                 "qmat batched_qt cache=%s group=%s (%s) differs at "
                                 "row %zu col %zu: %.9g vs %.9g -- a row's answer "
                                 "changed because of how it was called",
                                 mynah_qmat_qtype_name(cache_qtype),
                                 mynah_qmat_qtype_name(want), shapes[sh].name, b, i,
                                 (double)ref[at], (double)got[at]);
                    goto done;
                }
            }
        }
    }
    status = 0;
done:
    free(w); free(x); free(bias); free(ref); free(got); free(qx); free(sx);
    mynah_qmat_cache_free(cache);
    return status;
}

/* The property self_test_batched cannot see, and the one the serving path
 * actually depends on: a row's answer must not change when the BATCH AROUND IT
 * changes.  self_test_batched compares a fixed batch against the single-row
 * path, so it catches a kernel that is uniformly wrong -- but a kernel whose
 * rounding depends on a row's POSITION can still pass it on a lucky shape, and
 * one did.  On a Neoverse-V2 the SMMLA wiring gave one activation row four
 * different answers (lead of a pair, follower of a pair, SDOT tail of an odd
 * batch, alone) and self_test_batched's first mismatch looked like an ordinary
 * 1 ULP epilogue difference.
 *
 * So this varies the two things the caller does not control: how many rows are
 * in flight, and where this row landed among them.  The subject row is held
 * fixed and its companions are changed, which is the literal statement of "a
 * request's audio may not depend on who it was batched with".  Bit-equality,
 * not a tolerance -- a tolerance here would be a promise that the audio only
 * changes a little depending on the traffic, which is not a promise worth
 * making. */
static int self_test_batch_membership(int qtype, char *error,
                                      size_t error_capacity) {
    enum { N = 96, K = 256, BMAX = 6, SUBJECTS = 3 };
    int status = -1;
    mynah_qmat_cache *cache = mynah_qmat_cache_new(qtype);
    if (cache == NULL) {
        if (error != NULL && error_capacity > 0)
            snprintf(error, error_capacity, "qmat membership self-test OOM");
        return -1;
    }
    if (cache->qtype != qtype) {   /* unsupported here: nothing to check */
        mynah_qmat_cache_free(cache);
        return 0;
    }
    float *w = (float *)malloc((size_t)N * K * sizeof(float));
    float *x = (float *)malloc((size_t)BMAX * K * sizeof(float));
    float *bias = (float *)malloc((size_t)N * sizeof(float));
    float *alone = (float *)malloc((size_t)BMAX * N * sizeof(float));
    float *got = (float *)malloc((size_t)BMAX * N * sizeof(float));
    int8_t *qx = (int8_t *)malloc((size_t)BMAX * K);
    float *sx = (float *)malloc((size_t)BMAX * sizeof(float));
    if (w == NULL || x == NULL || bias == NULL || alone == NULL || got == NULL ||
        qx == NULL || sx == NULL) {
        if (error != NULL && error_capacity > 0)
            snprintf(error, error_capacity, "qmat membership self-test OOM");
        goto done;
    }
    for (size_t i = 0; i < (size_t)N * K; ++i)
        w[i] = sinf(0.017f * (float)i) * (0.5f + 0.5f * cosf(0.0011f * (float)i));
    for (size_t i = 0; i < (size_t)BMAX * K; ++i)
        x[i] = cosf(0.019f * (float)i) - 0.25f;
    for (size_t i = 0; i < (size_t)N; ++i) bias[i] = (float)i * 0.015625f - 0.75f;

    /* The answer every batch must reproduce: the row on its own. */
    for (size_t r = 0; r < (size_t)BMAX; ++r) {
        if (mynah_qmat_linear_resolved(cache, NULL, "membership.self.test", w,
                                       x + r * K, alone + r * N, 1u, K, N, bias,
                                       error, error_capacity) != 0) {
            goto done;
        }
    }
    for (size_t subject = 0; subject < (size_t)SUBJECTS; ++subject) {
        for (size_t batch = 1u; batch <= (size_t)BMAX; ++batch) {
            for (size_t pos = 0; pos < batch; ++pos) {
                /* Two different sets of companions for the same (batch, pos),
                 * so a pass cannot come from one lucky arrangement. */
                for (size_t variant = 0; variant < 2u; ++variant) {
                    const float *in_rows[BMAX];
                    float *out_rows[BMAX];
                    for (size_t b = 0; b < batch; ++b) {
                        size_t which = subject;
                        if (b != pos) {
                            which = (b + variant * 2u + 1u) % (size_t)BMAX;
                            if (which == subject)
                                which = (which + 1u) % (size_t)BMAX;
                        }
                        in_rows[b] = x + which * K;
                        out_rows[b] = got + b * N;
                    }
                    if (mynah_qmat_linear_batched(cache, NULL,
                                                  "membership.self.test", w,
                                                  in_rows, out_rows, batch, K, N,
                                                  bias, qx, sx, error,
                                                  error_capacity) != 0) {
                        goto done;
                    }
                    for (size_t i = 0; i < (size_t)N; ++i) {
                        const float *ref = alone + subject * N;
                        const float *mine = got + pos * N;
                        if (memcmp(&ref[i], &mine[i], sizeof(float)) == 0)
                            continue;
                        if (error != NULL && error_capacity > 0) {
                            snprintf(error, error_capacity,
                                     "qmat batch membership qtype=%d: row %zu "
                                     "at position %zu of %zu (variant %zu) "
                                     "differs from the same row alone at col "
                                     "%zu: %.9g vs %.9g -- a row's answer must "
                                     "not depend on who it was batched with",
                                     qtype, subject, pos, batch, variant, i,
                                     (double)ref[i], (double)mine[i]);
                        }
                        goto done;
                    }
                }
            }
        }
    }
    status = 0;
done:
    free(w); free(x); free(bias); free(alone); free(got); free(qx); free(sx);
    mynah_qmat_cache_free(cache);
    return status;
}

/* ------------------------------------------------- the VNNI identity test
 *
 * The u8 x s8 rewrite is integer algebra, not an approximation:
 *
 *     sum_j w[j]*x[j]  ==  sum_j w[j]*(x[j]+128) - 128 * sum_j w[j]
 *
 * over the prefix the VNNI loop covers, plus an untouched signed tail.  So the
 * only acceptable result is BIT-IDENTICAL output, and this asserts exactly
 * that with memcmp rather than a tolerance.  A tolerance here would hide the
 * two mistakes that are actually easy to make: a row sum taken over the whole
 * row instead of the VNNI prefix, and a tail that gets corrected twice.
 *
 * It runs the portable QMAT_U8_SCALAR kernel on every host, which is what
 * makes the correction provable on a machine with no VNNI unit at all, and
 * additionally runs whichever of QMAT_U8_VEX / QMAT_U8_EVEX this CPU resolves
 * to.  K is chosen so the block, the tail and the row remainder are all
 * exercised: 200 = 3*64 + 8 with 13 rows = 3*4 + 1.
 *
 * WHAT IS ASSERTED EXACTLY, AND WHAT IS NOT (E4-20).
 *
 * The int32 dot products are compared with `==`.  That is the whole of the u8
 * rewrite -- the +128 correction, the cached row-sum prefix and the signed
 * tail are integer algebra, every term fits int32, and there is no rounding
 * anywhere in it.  Both mistakes this test was written to catch live here: a
 * row sum taken over the whole row instead of the VNNI prefix, and a tail
 * corrected twice.  Either moves the int32 by a multiple of 128 * a weight
 * sum, i.e. by ~1e5, so an exact integer comparison cannot be fooled.
 *
 * The FLOATS are compared to within 2 ULP, not bit-identically, and the
 * reason is a property of the build rather than of the kernel.  The epilogue
 * `(float)s * scales[row] * sx + bias[row]` is a three-factor product, this
 * file compiles with -ffast-math (hence -fassociative-math), and the
 * optimizer regroups it as `(s*scale)*sx` or `(s*sx)*scale` independently at
 * each of the three places the quad epilogue appears -- and, when the three
 * were merged into one `static inline` helper, independently at each inlined
 * site as well.  The two groupings round differently.  This test used to
 * demand bit-identity of the floats and therefore FAILED on x86 at rows 3, 7
 * and 11 -- lane r == 3 of each quad -- while every integer it compared was
 * already identical.  The failure was manufactured by the assertion, not by
 * the kernel.
 *
 * 2 ULP is not a tolerance on the quantity being tested: a wrong row sum lands
 * ~1e5 int32 units away, which is millions of ULP.  It is a bound on
 * reassociation of a product that the language permits the compiler to
 * reorder.  AGENTS.md's numerical rule allows exactly this ("do not require
 * byte-identical audio across different floating-point orderings"), and no
 * process ever mixes the two kernels: qmat_u8_level() resolves once and is
 * immutable for the life of the process. */

/* Distance in representable floats.  The two operands are finite and carry the
 * same sign -- they are one integer scaled by the same two positive factors --
 * so their bit patterns are monotone and the difference counts ULP. */
static long u8_ulp_gap(float a, float b) {
    int32_t ia = 0, ib = 0;
    memcpy(&ia, &a, sizeof ia);
    memcpy(&ib, &b, sizeof ib);
    if ((ia < 0) != (ib < 0)) return (a == b) ? 0L : LONG_MAX;
    const long d = (long)ia - (long)ib;
    return d < 0 ? -d : d;
}

static int u8_identity_one(size_t n, size_t k, int level,
                           char *error, size_t error_capacity) {
    int status = -1;
    float *w = (float *)malloc(n * k * sizeof(float));
    float *x = (float *)malloc(k * sizeof(float));
    float *bias = (float *)malloc(n * sizeof(float));
    float *ref = (float *)malloc(n * sizeof(float));
    float *got = (float *)malloc(n * sizeof(float));
    int8_t *q8 = (int8_t *)malloc(n * k);
    float *scales = (float *)malloc(n * sizeof(float));
    int32_t *rowsum = (int32_t *)malloc(n * sizeof(int32_t));
    int8_t *qs = (int8_t *)malloc(k);
    uint8_t *qu = (uint8_t *)malloc(k);
    if (w == NULL || x == NULL || bias == NULL || ref == NULL || got == NULL ||
        q8 == NULL || scales == NULL || rowsum == NULL || qs == NULL || qu == NULL) {
        if (error != NULL && error_capacity > 0)
            snprintf(error, error_capacity, "qmat u8 identity out of memory");
        goto done;
    }
    for (size_t i = 0; i < n * k; ++i)
        w[i] = sinf(0.0173f * (float)i) * (0.5f + 0.5f * cosf(0.0009f * (float)i));
    for (size_t j = 0; j < k; ++j) x[j] = cosf(0.0111f * (float)j) - 0.3f;
    for (size_t i = 0; i < n; ++i) bias[i] = (float)i * 0.0625f - 0.5f;
    quantize_weight_int8(w, n, k, q8, scales);
    weight_rowsum_prefix(q8, n, k, rowsum);

    const float ss = quantize_act_int8(qs, x, k);
    const float su = quantize_act_u8(qu, x, k);
    /* The two quantizers must agree element for element, or the identity
     * below would be comparing two different activations. */
    if (memcmp(&ss, &su, sizeof(float)) != 0) {
        if (error != NULL && error_capacity > 0)
            snprintf(error, error_capacity,
                     "qmat u8 activation scale differs: %.9g vs %.9g",
                     (double)ss, (double)su);
        goto done;
    }
    for (size_t j = 0; j < k; ++j) {
        if ((int)qu[j] - 128 != (int)qs[j]) {
            if (error != NULL && error_capacity > 0)
                snprintf(error, error_capacity,
                         "qmat u8 activation differs at %zu: %d vs %d",
                         j, (int)qu[j] - 128, (int)qs[j]);
            goto done;
        }
    }

    /* The exact assertion: the integer the two kernels reduce to, row by row.
     * This is the u8 rewrite in full, and it admits no rounding. */
    for (size_t i = 0; i < n; ++i) {
        const int32_t want = dot_q8_i32(qs, q8 + i * k, 0, k, QMAT_U8_OFF);
        const int32_t have = dot_q8_i32(qu, q8 + i * k, rowsum[i], k, level);
        if (want != have) {
            if (error != NULL && error_capacity > 0) {
                snprintf(error, error_capacity,
                         "qmat u8 level=%d int32 differs at row %zu of %zu "
                         "(k=%zu, main=%zu): %d vs %d -- the +128 correction, "
                         "the row-sum prefix or the signed tail is wrong",
                         level, i, n, k, qmat_u8_main(k), want, have);
            }
            goto done;
        }
    }

    /* And the assembled matvec, BIT FOR BIT.  This used to be a 2 ULP bound,
     * and the bound was a symptom: the signed quad epilogue and the u8 quad
     * epilogue were two textual three-factor products that -ffast-math grouped
     * independently, so they disagreed at lane r == 3 of each quad on x86 while
     * every integer above was already identical.  Both now end through
     * qmat_row_scale()/qmat_row_epilogue(), which is one grouping, so the only
     * honest assertion is equality -- and asserting it here is what makes the
     * repair visible on a machine with a VNNI unit, which is where the ULP
     * disagreement was originally seen. */
    matvec_q8(ref, qs, ss, q8, scales, NULL, bias, n, k, QMAT_U8_OFF);
    matvec_q8(got, qu, su, q8, scales, rowsum, bias, n, k, level);
    for (size_t i = 0; i < n; ++i) {
        if (memcmp(&ref[i], &got[i], sizeof(float)) != 0) {
            if (error != NULL && error_capacity > 0) {
                snprintf(error, error_capacity,
                         "qmat u8 level=%d matvec row %zu of %zu (k=%zu) "
                         "differs from the signed kernel by %ld ULP: %.9g vs "
                         "%.9g -- the int32 above is identical, so this is the "
                         "float epilogue regrouping per site again",
                         level, i, n, k, u8_ulp_gap(ref[i], got[i]),
                         (double)ref[i], (double)got[i]);
            }
            goto done;
        }
    }
    /* And the single-row entry, which the non-row4 path uses.  Same reason and
     * the same assertion: dot_q8's `* ws * sx` is now one call to
     * qmat_row_scale(), so the two levels share one compiled epilogue. */
    for (size_t i = 0; i < n; ++i) {
        const float a = dot_q8(qs, ss, q8 + i * k, scales[i], 0, k,
                               QMAT_U8_OFF, bias[i]);
        const float b = dot_q8(qu, su, q8 + i * k, scales[i], rowsum[i], k,
                               level, bias[i]);
        if (memcmp(&a, &b, sizeof(float)) != 0) {
            if (error != NULL && error_capacity > 0) {
                snprintf(error, error_capacity,
                         "qmat u8 level=%d single-row %zu differs by %ld ULP: "
                         "%.9g vs %.9g", level, i, u8_ulp_gap(a, b),
                         (double)a, (double)b);
            }
            goto done;
        }
    }
    status = 0;
done:
    free(w); free(x); free(bias); free(ref); free(got);
    free(q8); free(scales); free(rowsum); free(qs); free(qu);
    return status;
}

static int self_test_u8_identity(char *error, size_t error_capacity) {
    /* k = 200 (block + tail), 64 (no tail), 40 (tail only, no VNNI block). */
    static const size_t widths[3] = { 200u, 64u, 40u };
    /* Every kernel this host can run, not just the one it would choose: on a
     * Zen 4/5 both VPDPBUSD encodings are reachable, and testing only the
     * faster one would ship the other untested on the very machine used to
     * benchmark it.  QMAT_U8_SCALAR always runs, which is what gives an
     * ARM-only CI a real check of the +128 correction. */
    int levels[3] = { QMAT_U8_SCALAR, qmat_u8_level(), qmat_u8_vex_level() };
    for (int l = 0; l < 3; ++l) {
        if (levels[l] == QMAT_U8_OFF) continue;
        int already = 0;
        for (int p = 0; p < l; ++p) if (levels[p] == levels[l]) already = 1;
        if (already) continue;
        for (size_t i = 0; i < 3u; ++i) {
            if (u8_identity_one(13u, widths[i], levels[l], error,
                                error_capacity) != 0) {
                return -1;
            }
        }
    }
    return 0;
}

#if defined(MYNAH_QMAT_ARM_I8MM)
/* SMMLA accumulates the same products in a different order, and integer
 * addition does not care about order, so the batched 2x2 tile must land on the
 * SAME INT32 as two separate SDOT rows.  That was always true and is checked
 * first, with scales and sx forced to 1 and bias NULL, so the float epilogue is
 * the identity on an int32 far inside the 2^24 a float represents exactly and
 * any surviving difference is an integer difference.
 *
 * The second pass is the one that is new, and it is the whole point of E4-20b.
 * It runs the same shapes with REAL per-row scales, a real activation scale and
 * a real bias -- the arithmetic the serving path actually does -- and demands
 * BIT-IDENTICAL floats.  The previous version of this test deliberately did not
 * assert that, because it was not true: 71 of 192 rows differed by up to 2 ULP
 * at 96x256 under GCC 15, the two kernels' three-factor epilogues having been
 * grouped differently by the optimizer.  Now both end through
 * qmat_row_scale()/qmat_row_epilogue() and the difference is gone by
 * construction, so anything but equality is a regression in the thing the
 * barrier exists to hold.
 *
 * The shapes are chosen to REACH the epilogue's reassociation rather than to
 * be convenient: one large enough that the old defect was visible (96x256), one
 * with an odd row count and an odd k so both tails are covered (13x37).  An
 * earlier version used only 13x37, passed, and thereby licensed wiring the
 * broken kernel into the batched linear -- a gate that passes because it never
 * reaches the road is worse than no gate. */
static int self_test_i8mm_identity_shape(size_t N, size_t K,
                                         char *error, size_t error_capacity) {
    int status = -1;
    float *w = (float *)malloc(N * K * sizeof(float));
    float *x = (float *)malloc(2u * K * sizeof(float));
    float *unit = (float *)malloc(N * sizeof(float));
    float *bias = (float *)malloc(N * sizeof(float));
    float *ref = (float *)malloc(2u * N * sizeof(float));
    float *got = (float *)malloc(2u * N * sizeof(float));
    int8_t *q8 = (int8_t *)malloc(N * K);
    float *scales = (float *)malloc(N * sizeof(float));
    int8_t *qx = (int8_t *)malloc(2u * K);
    if (w == NULL || x == NULL || unit == NULL || bias == NULL || ref == NULL ||
        got == NULL || q8 == NULL || scales == NULL || qx == NULL) {
        if (error != NULL && error_capacity > 0)
            snprintf(error, error_capacity, "qmat i8mm identity out of memory");
        goto done;
    }
    for (size_t i = 0; i < N * K; ++i)
        w[i] = sinf(0.021f * (float)i) * (0.5f + 0.5f * cosf(0.0013f * (float)i));
    for (size_t i = 0; i < 2u * K; ++i) x[i] = cosf(0.017f * (float)i) - 0.2f;
    for (size_t i = 0; i < N; ++i) unit[i] = 1.0f;
    for (size_t i = 0; i < N; ++i) bias[i] = (float)i * 0.03125f - 0.6f;
    quantize_weight_int8(w, N, K, q8, scales);
    const float sx0 = quantize_act_int8(qx, x, K);
    const float sx1 = quantize_act_int8(qx + K, x + K, K);

    /* Pass 1: scale = sx = 1, bias = NULL.  out is (float)s, exact for
     * |s| < 2^24, so this is an assertion about the integers. */
    matvec_q8(ref, qx, 1.0f, q8, unit, NULL, NULL, N, K, QMAT_U8_OFF);
    matvec_q8(ref + N, qx + K, 1.0f, q8, unit, NULL, NULL, N, K, QMAT_U8_OFF);
    matvec_q8_pair_i8mm(got, got + N, qx, qx + K, 1.0f, 1.0f, q8, unit, NULL,
                        N, K);
    for (size_t i = 0; i < 2u * N; ++i) {
        if (memcmp(&ref[i], &got[i], sizeof(float)) != 0) {
            if (error != NULL && error_capacity > 0) {
                snprintf(error, error_capacity,
                         "qmat i8mm int32 differs at %zu of %zux%zu: "
                         "sdot %.1f vs smmla %.1f", i, N, K,
                         (double)ref[i], (double)got[i]);
            }
            goto done;
        }
        if (fabsf(ref[i]) >= 16777216.0f) {
            if (error != NULL && error_capacity > 0) {
                snprintf(error, error_capacity,
                         "qmat i8mm identity test is out of the exact-float "
                         "range at %zu: %.1f", i, (double)ref[i]);
            }
            goto done;
        }
    }

    /* Pass 2: the real epilogue -- per-row scales, two different activation
     * scales, a bias.  Bit-identical, not bounded. */
    matvec_q8(ref, qx, sx0, q8, scales, NULL, bias, N, K, QMAT_U8_OFF);
    matvec_q8(ref + N, qx + K, sx1, q8, scales, NULL, bias, N, K, QMAT_U8_OFF);
    matvec_q8_pair_i8mm(got, got + N, qx, qx + K, sx0, sx1, q8, scales, bias,
                        N, K);
    for (size_t i = 0; i < 2u * N; ++i) {
        if (memcmp(&ref[i], &got[i], sizeof(float)) != 0) {
            if (error != NULL && error_capacity > 0) {
                snprintf(error, error_capacity,
                         "qmat i8mm float epilogue differs at row %zu of "
                         "%zux%zu (activation %zu) by %ld ULP: sdot %.9g vs "
                         "smmla %.9g -- the int32 above is identical, so the "
                         "two kernels are grouping the scale product "
                         "differently again",
                         i % N, N, K, i / N, u8_ulp_gap(ref[i], got[i]),
                         (double)ref[i], (double)got[i]);
            }
            goto done;
        }
    }
    status = 0;
done:
    free(w); free(x); free(unit); free(bias); free(ref); free(got);
    free(q8); free(scales); free(qx);
    return status;
}

static int self_test_i8mm_identity(char *error, size_t error_capacity) {
    if (!qmat_i8mm()) return 0;
    /* Large enough that the float epilogue would diverge if it were asserted;
     * then the two-tail edge case (odd rows, odd k). */
    if (self_test_i8mm_identity_shape(96u, 256u, error, error_capacity) != 0)
        return -1;
    return self_test_i8mm_identity_shape(13u, 37u, error, error_capacity);
}
#endif

/* ---------------------------------------------- the INT4 identity (E4-21)
 *
 * int4's vector kernels compute the same integers as the scalar nibble loop or
 * they are wrong.  There is no rounding anywhere in a group: nibbles are
 * exact, the activation is exact int8, and the products and their sum all fit
 * int32 with room to spare.  So the only acceptable result is BIT-IDENTICAL
 * output, and this asserts exactly that -- first per group, in the integer
 * domain where the claim is unambiguous, then through the assembled dot_q4()
 * and matvec_q4() where a wrong group index or a wrong scale would show.
 *
 * IT IS NOT ONLY AN x86 TEST.  Nothing in this tree had ever compared the NEON
 * int4 kernel against the scalar one either: the existing int4 gates compare
 * the quantized result against an f32 dot with a RELATIVE TOLERANCE, which int4
 * needs (it is a lossy format) but which is far too loose to notice a kernel
 * that has, say, swapped the even and odd nibbles -- that lands inside the
 * tolerance on smooth data and is the exact mistake the interleave in
 * q4_unpack_u8() could make.  So this runs everywhere and compares the kernel
 * this build actually uses against the reference it is supposed to reproduce.
 *
 * THE SATURATION BOUND is asserted too, not just assumed.  The x86 kernel
 * leans on _mm256_maddubs_epi16 not saturating: nibble in [0,15] against an
 * int8 activation gives pairwise sums in [-3840, 3810], the `- 8 * sum x`
 * correction term is in [-2048, 2032], and the int16 difference is in
 * [-5888, 5858].  The data below is built to reach the extremes (an all-0x0F
 * group against an all -128 activation and the mirror of it), because a bound
 * that is only ever exercised at a tenth of its range is not evidence. */
static int self_test_q4_identity(char *error, size_t error_capacity) {
    enum { GROUPS = 9, K = GROUPS * QMAT_Q4_GROUP, N = 7 };
    int status = -1;
    uint8_t *q = (uint8_t *)malloc((size_t)N * (K / 2u));
    int8_t *x = (int8_t *)malloc(K);
    float *scales = (float *)malloc((size_t)N * GROUPS * sizeof(float));
    float *bias = (float *)malloc((size_t)N * sizeof(float));
    float *ref = (float *)malloc((size_t)N * sizeof(float));
    float *acc_ref = (float *)malloc((size_t)N * sizeof(float));
    float *got = (float *)malloc((size_t)N * sizeof(float));
    if (q == NULL || x == NULL || scales == NULL || bias == NULL || ref == NULL ||
        acc_ref == NULL || got == NULL) {
        if (error != NULL && error_capacity > 0)
            snprintf(error, error_capacity, "qmat q4 identity out of memory");
        goto done;
    }
    /* Group 0 of every row is all 0x0F against an all -128 activation, and
     * group 1 is all 0x00, so the two int16 extremes are actually reached. */
    for (size_t i = 0; i < (size_t)N * (K / 2u); ++i) {
        const size_t g = (i % (K / 2u)) / 16u;
        if (g == 0u) q[i] = 0xFFu;
        else if (g == 1u) q[i] = 0x00u;
        else q[i] = (uint8_t)((i * 37u + 11u) & 0xFFu);
    }
    for (size_t j = 0; j < (size_t)K; ++j) {
        if (j < QMAT_Q4_GROUP) x[j] = -128;
        else if (j < 2u * QMAT_Q4_GROUP) x[j] = 127;
        else x[j] = (int8_t)((int)((j * 53u + 7u) % 255u) - 127);
    }
    for (size_t i = 0; i < (size_t)N * GROUPS; ++i)
        scales[i] = 1.0e-3f * (1.0f + (float)(i % 13) * 0.37f);
    for (size_t i = 0; i < (size_t)N; ++i) bias[i] = (float)i * 0.0625f - 0.25f;

    /* 1. Per group, in the integer domain, plus the int16 bound the x86
     *    kernel's lack of saturation depends on. */
    for (size_t i = 0; i < (size_t)N; ++i) {
        for (size_t g = 0; g < (size_t)GROUPS; ++g) {
            const uint8_t *qg = q + i * (K / 2u) + g * 16u;
            const int8_t *xg = x + g * 32u;
            const int32_t want = q4_group_i32_scalar(qg, xg);
            int32_t have = want;
#if defined(MYNAH_QMAT_AVX2)
            have = q4_group_i32_avx2(qg, xg);
#endif
            if (want != have) {
                if (error != NULL && error_capacity > 0) {
                    snprintf(error, error_capacity,
                             "qmat q4 group int32 differs at row %zu group %zu: "
                             "scalar %d vs vector %d -- the nibble order, the "
                             "-8 offset or the correction term is wrong",
                             i, g, want, have);
                }
                goto done;
            }
            /* Pairwise int16 partials must not have saturated on the way. */
            for (size_t j = 0; j < QMAT_Q4_GROUP; j += 2) {
                const uint8_t b = qg[j / 2];
                const long p = (long)((int)(b & 0x0F) - 8) * (long)xg[j] +
                               (long)((int)(b >> 4) - 8) * (long)xg[j + 1];
                if (p < -32768L || p > 32767L) {
                    if (error != NULL && error_capacity > 0) {
                        snprintf(error, error_capacity,
                                 "qmat q4 pairwise partial %ld at row %zu group "
                                 "%zu pair %zu is outside int16 -- the x86 "
                                 "kernel's maddubs would saturate", p, i, g,
                                 j / 2u);
                    }
                    goto done;
                }
            }
        }
    }

    /* 2. The assembled row, bit for bit, against the same group order. */
    const float sx = 0.0072f;
    for (size_t i = 0; i < (size_t)N; ++i) {
        float acc = 0.0f;
        for (size_t g = 0; g < (size_t)GROUPS; ++g) {
            acc = qmat_q4_accum(acc,
                                q4_group_i32_scalar(q + i * (K / 2u) + g * 16u,
                                                    x + g * 32u),
                                scales[i * GROUPS + g]);
        }
        acc_ref[i] = acc;
        ref[i] = qmat_q4_finish(acc, sx, 0.0f);
        got[i] = dot_q4(x, sx, q + i * (K / 2u), scales + i * GROUPS, K, 0.0f);
        if (memcmp(&ref[i], &got[i], sizeof(float)) != 0) {
            if (error != NULL && error_capacity > 0) {
                snprintf(error, error_capacity,
                         "qmat q4 dot row %zu differs: reference %.9g vs kernel "
                         "%.9g -- the integers matched, so this is the group "
                         "order or a scale index", i, (double)ref[i],
                         (double)got[i]);
            }
            goto done;
        }
    }

    /* 3. And matvec_q4, which has its own quad unroll on ARM and falls to the
     *    dot above on x86 -- both must land on the same floats as the row loop
     *    plus the bias. */
    matvec_q4(got, x, sx, q, scales, bias, (size_t)N, (size_t)K);
    for (size_t i = 0; i < (size_t)N; ++i) {
        const float want = qmat_q4_finish(acc_ref[i], sx, bias[i]);
        if (memcmp(&want, &got[i], sizeof(float)) != 0) {
            if (error != NULL && error_capacity > 0) {
                snprintf(error, error_capacity,
                         "qmat q4 matvec row %zu of %d differs from the row "
                         "loop: %.9g vs %.9g", i, (int)N, (double)want,
                         (double)got[i]);
            }
            goto done;
        }
    }
    status = 0;
done:
    free(q); free(x); free(scales); free(bias); free(ref); free(acc_ref);
    free(got);
    return status;
}

/* ------------------------------------------- the SMMLA on/off A/B (E4-20b)
 *
 * self_test_batch_membership asks whether a row's answer survives a change of
 * companions.  This asks the other half of the same question: whether it
 * survives a change of KERNEL.  The batched linear is run twice over identical
 * inputs, once with the SMMLA wiring forced off and once forced on, and the
 * two must be bit-identical -- not close, identical, because the two runs
 * differ only in which instruction did the integer accumulation and integer
 * accumulation has no rounding to differ about.
 *
 * It is the direct statement of the promise that was broken before E4-20b: a
 * deployment that flips MYNAH_QMAT_I8MM must not change one sample of audio.
 * Membership alone would not catch a kernel that is consistently wrong in the
 * same way for every position, and the goldens would not catch it either on a
 * machine where the wiring is off.
 *
 * The shapes deliberately straddle the kernel's seams: an odd BATCH leaves one
 * activation to the SDOT fallback, and an odd ROW COUNT leaves one weight row
 * to matvec_q8 inside the SMMLA path, so both tails are compared as well as the
 * 2x2 body.  On a build or CPU without SMMLA the force is a no-op and this
 * degenerates to running the same thing twice, which costs a little time and
 * asserts something trivially true -- preferable to an #if that would stop the
 * test existing on the machines where it is cheapest to run. */
static int self_test_i8mm_ab(char *error, size_t error_capacity) {
    /* Odd and even on both axes; K covers a k with and without a 16-byte tail. */
    static const size_t rows[4] = { 96u, 13u, 64u, 7u };
    static const size_t cols[4] = { 256u, 37u, 128u, 40u };
    enum { BMAX = 5 };
    int status = -1;
    const int saved = mynah_qmat_i8mm_force(-1);
    mynah_qmat_cache *cache = mynah_qmat_cache_new(QMAT_INT8);
    float *w = NULL, *x = NULL, *bias = NULL, *a = NULL, *b = NULL, *sx = NULL;
    int8_t *qx = NULL;
    if (cache == NULL || cache->qtype != QMAT_INT8) {
        if (cache != NULL) { mynah_qmat_cache_free(cache); return 0; }
        if (error != NULL && error_capacity > 0)
            snprintf(error, error_capacity, "qmat i8mm A/B self-test OOM");
        return -1;
    }
    for (size_t sh = 0; sh < 4u; ++sh) {
        const size_t N = rows[sh], K = cols[sh];
        free(w); free(x); free(bias); free(a); free(b); free(qx); free(sx);
        w = (float *)malloc(N * K * sizeof(float));
        x = (float *)malloc((size_t)BMAX * K * sizeof(float));
        bias = (float *)malloc(N * sizeof(float));
        a = (float *)malloc((size_t)BMAX * N * sizeof(float));
        b = (float *)malloc((size_t)BMAX * N * sizeof(float));
        qx = (int8_t *)malloc((size_t)BMAX * K);
        sx = (float *)malloc((size_t)BMAX * sizeof(float));
        if (w == NULL || x == NULL || bias == NULL || a == NULL || b == NULL ||
            qx == NULL || sx == NULL) {
            if (error != NULL && error_capacity > 0)
                snprintf(error, error_capacity, "qmat i8mm A/B self-test OOM");
            goto done;
        }
        for (size_t i = 0; i < N * K; ++i)
            w[i] = sinf(0.013f * (float)i) * (0.5f + 0.5f * cosf(0.0007f * (float)i));
        for (size_t i = 0; i < (size_t)BMAX * K; ++i)
            x[i] = cosf(0.023f * (float)i) - 0.15f;
        for (size_t i = 0; i < N; ++i) bias[i] = (float)i * 0.0078125f - 0.4f;
        for (size_t batch = 1u; batch <= (size_t)BMAX; ++batch) {
            const float *in_rows[BMAX];
            float *out_a[BMAX];
            float *out_b[BMAX];
            for (size_t j = 0; j < batch; ++j) {
                in_rows[j] = x + j * K;
                out_a[j] = a + j * N;
                out_b[j] = b + j * N;
            }
            for (int pass = 0; pass < 2; ++pass) {
                mynah_qmat_i8mm_force(pass);
                if (mynah_qmat_linear_batched(cache, NULL, "i8mm.ab.self.test",
                                              w, in_rows,
                                              pass == 0 ? out_a : out_b, batch,
                                              K, N, bias, qx, sx, error,
                                              error_capacity) != 0) {
                    goto done;
                }
            }
            for (size_t j = 0; j < batch; ++j) {
                for (size_t i = 0; i < N; ++i) {
                    const size_t at = j * N + i;
                    if (memcmp(&a[at], &b[at], sizeof(float)) == 0) continue;
                    if (error != NULL && error_capacity > 0) {
                        snprintf(error, error_capacity,
                                 "qmat i8mm A/B %zux%zu batch %zu: activation "
                                 "%zu row %zu differs by %ld ULP with the SMMLA "
                                 "wiring off (%.9g) vs on (%.9g) -- flipping "
                                 "MYNAH_QMAT_I8MM must not change one sample",
                                 N, K, batch, j, i, u8_ulp_gap(a[at], b[at]),
                                 (double)a[at], (double)b[at]);
                    }
                    goto done;
                }
            }
        }
    }
    status = 0;
done:
    free(w); free(x); free(bias); free(a); free(b); free(qx); free(sx);
    mynah_qmat_cache_free(cache);
    mynah_qmat_i8mm_force(saved);
    return status;
}

int mynah_qmat_self_test(char *error, size_t error_capacity) {
    if (self_test_u8_identity(error, error_capacity) != 0) return -1;
#if defined(MYNAH_QMAT_ARM_I8MM)
    if (self_test_i8mm_identity(error, error_capacity) != 0) return -1;
#endif
    if (self_test_q4_identity(error, error_capacity) != 0) return -1;
    if (self_test_one(QMAT_INT8, error, error_capacity) != 0) return -1;
    if (self_test_one(QMAT_INT4, error, error_capacity) != 0) return -1;
    if (self_test_rows_blocked(QMAT_INT8, error, error_capacity) != 0) return -1;
    if (self_test_rows_blocked(QMAT_INT4, error, error_capacity) != 0) return -1;
    if (self_test_batched(QMAT_INT8, error, error_capacity) != 0) return -1;
    if (self_test_batched(QMAT_INT4, error, error_capacity) != 0) return -1;
    /* Every lane width of every batched encoding that has one.  f16 and int8
     * each have a four-wide kernel, a two-wide one and a three-remainder served
     * by the four-wide kernel with an activation repeated; B=5 in the test above
     * reaches two of those six paths. */
    if (self_test_act_quantize(error, error_capacity) != 0) return -1;
    if (self_test_q4_scale_solve(error, error_capacity) != 0) return -1;
    if (self_test_lane_widths(QMAT_F16, error, error_capacity) != 0) return -1;
    if (self_test_lane_widths(QMAT_INT8, error, error_capacity) != 0) return -1;
    /* E8-5: a group that names its own encoding, with the batched call as the
     * tensor's first toucher.  Every cache profile the runtime ships crossed
     * with every encoding a group spec can name. */
    {
        static const int caches[3] = {QMAT_F32, QMAT_INT8, QMAT_F16};
        static const int groups[3] = {QMAT_INT8, QMAT_INT4, QMAT_F16};
        for (size_t c = 0; c < 3u; ++c) {
            for (size_t g = 0; g < 3u; ++g) {
                if (self_test_batched_qt(caches[c], groups[g], error,
                                         error_capacity) != 0) {
                    return -1;
                }
            }
        }
    }
    /* INT8 membership is asserted against BOTH int8 kernels, not just the one
     * this host would pick: the SMMLA wiring is what made the property fail,
     * and a suite that only ever ran the default would have gone quiet again
     * the moment the default changed. */
    {
        const int saved = mynah_qmat_i8mm_force(-1);
        for (int mode = 0; mode < 2; ++mode) {
            mynah_qmat_i8mm_force(mode);
            if (self_test_batch_membership(QMAT_INT8, error,
                                           error_capacity) != 0) {
                mynah_qmat_i8mm_force(saved);
                return -1;
            }
        }
        mynah_qmat_i8mm_force(saved);
    }
    if (self_test_i8mm_ab(error, error_capacity) != 0) return -1;
    if (self_test_batch_membership(QMAT_INT4, error, error_capacity) != 0) return -1;
#if defined(MYNAH_QMAT_F16)
    if (self_test_f16_convert(error, error_capacity) != 0) return -1;
    if (self_test_f16(error, error_capacity) != 0) return -1;
    if (self_test_rows_blocked(QMAT_F16, error, error_capacity) != 0) return -1;
    if (self_test_batched(QMAT_F16, error, error_capacity) != 0) return -1;
    if (self_test_batch_membership(QMAT_F16, error, error_capacity) != 0) return -1;
#endif
    return 0;
}

/* ======================================================================
 * Dispatch predicates
 *
 * Everything below exists so src/dispatch.c can CALL the decision instead of
 * restating it.  Each function is the single definition used by the hot path
 * too -- a predicate that only the report consults would drift from the code
 * within one commit, which is exactly the failure the report was written to
 * catch.
 * ====================================================================== */

/* Names the int8 kernel this host+binary pair actually resolves to. */
const char *mynah_qmat_int8_kernel(const char **why) {
    const int level = qmat_u8_level();
    switch (level) {
    case QMAT_U8_EVEX:
        if (why != NULL)
            *why = "[predicate] src/qmat.c dot4_u8_evex: VPDPBUSD (EVEX, "
                   "_mm512_dpbusd_epi32) on a u8 activation, with the +128 "
                   "bias removed by the per-row weight sum cached at "
                   "quantization time. Bit-identical to the signed path";
        return "avx512vnni";
    case QMAT_U8_VEX:
        if (why != NULL)
            *why = "[predicate] src/qmat.c dot4_u8_vex: VPDPBUSD (VEX, "
                   "_mm256_dpbusd_avx_epi32). AVX-512 is absent or off, which "
                   "on Alder Lake and later client parts is the normal case; "
                   "on Zen 4/5 the 256-bit form is full rate anyway";
        return "avxvnni";
    case QMAT_U8_SCALAR:
        if (why != NULL)
            *why = "[predicate] MYNAH_QMAT_VNNI=scalar: the portable unsigned "
                   "kernel. It is the algebra without the intrinsics, for "
                   "proving the +128 correction, not for speed";
        return "u8-scalar";
    default:
        break;
    }
#if defined(MYNAH_QMAT_DOTPROD)
    if (why != NULL)
        *why = "[predicate] src/qmat.c matvec_q8: ARM SDOT (vdotq_s32), four "
               "weight rows in flight per activation load";
    return "neon-sdot";
#elif defined(MYNAH_QMAT_AVX2)
    if (why != NULL)
        *why = "[predicate] src/qmat.c dot_q8_i32_avx2: no usable VPDPBUSD on "
               "this CPU, so the int8 dot is the AVX2 widen-then-madd pair "
               "(_mm256_cvtepi8_epi16 + _mm256_madd_epi16)";
    return "avx2";
#else
    if (why != NULL)
        *why = "[predicate] src/qmat.c: scalar int32 accumulation, the "
               "correctness reference";
    return "scalar";
#endif
}

/* The canonical epilogue, exported.  It is the same inline pair every kernel
 * in this file ends with, so a test that pins this pins them. */
float mynah_qmat_epilogue(int32_t s, float ws, float sx, float bias) {
    return qmat_row_epilogue(s, qmat_row_scale(ws, sx), bias);
}

/* Test hook: force the SMMLA wiring on (1) or off (0) for the rest of this
 * process, or return to the env/default resolution (-1).  Returns what was in
 * effect before, so a caller can restore it.  It exists so a test can run the
 * SAME shapes through both kernels in one process and require bit-identical
 * output -- the property the batched linear promises and the one a
 * position-dependent epilogue breaks.  On a build or a CPU with no SMMLA it is
 * a no-op that reports -1, which is what lets the test run everywhere. */
int mynah_qmat_i8mm_force(int mode) {
#if defined(MYNAH_QMAT_ARM_I8MM)
    if (!qmat_i8mm()) return -1;
    const int before = qmat_i8mm_override;
    qmat_i8mm_override = (mode < 0) ? -1 : (mode != 0);
    return before;
#else
    (void)mode;
    return -1;
#endif
}

/* Names the int4 kernel this host resolves to: "neon-sdot", "avx2" or
 * "scalar".  It exists because int4's x86 path was INVISIBLE: until E4-21
 * there was no vector kernel there at all, the dispatch report had no int4 row
 * to say so, and the one recorded int4 measurement ("only 3% over int8 on the
 * codec") was taken on ARM, where int4 at least had SDOT.  A format that is
 * slower than the wider one it is supposed to beat should be discoverable by
 * reading the report, not by reading the source. */
const char *mynah_qmat_int4_kernel(const char **why) {
#if defined(MYNAH_QMAT_DOTPROD)
    if (why != NULL)
        *why = "[predicate] src/qmat.c dot_q4/matvec_q4: ARM SDOT (vdotq_s32) "
               "on nibbles unpacked to int8, with vld2q_s8 splitting the "
               "activation into the even and odd halves the two nibbles need";
    return "neon-sdot";
#elif defined(MYNAH_QMAT_AVX2)
    if (why != NULL)
        *why = "[predicate] src/qmat.c q4_group_i32_avx2: AVX2 "
               "_mm256_maddubs_epi16 on UNSIGNED nibbles against the signed "
               "activation, with the -8 offset applied as a second maddubs "
               "rather than a table. NOT VNNI on purpose: int4's per-group "
               "scale forces a float flush every 32 elements, so the dot is "
               "~3 of ~18 instructions per group and VPDPBUSD would move the "
               "kernel by about a tenth";
    return "avx2";
#else
    if (why != NULL)
        *why = "[predicate] src/qmat.c: scalar nibble loop, the correctness "
               "reference -- roughly eight times the instructions per MAC of "
               "the vector paths, so int4 here will lose to int8 despite half "
               "the weight bytes";
    return "scalar";
#endif
}

int mynah_qmat_i8mm_enabled(const char **why) {
#if defined(MYNAH_QMAT_ARM_I8MM)
    const int hw = qmat_i8mm();
    const int on = qmat_i8mm_batched();
    if (why != NULL) {
        *why = on ? "[predicate] src/qmat.c matvec_q8_pair_i8mm: SMMLA "
                    "(vmmlaq_s32), two activations x two weight rows x eight k "
                    "per instruction, wired into the weight-stationary batched "
                    "linear -- and ONLY there, because a single-vector decode "
                    "matvec has nothing to put in the other half of the tile. "
                    "Bit-identical to the SDOT path: the int32 tile always was, "
                    "and the float epilogue now goes through qmat_row_scale()'s "
                    "value barrier, so both kernels round alike by construction"
            : hw ? "[predicate] src/qmat.c: this CPU HAS FEAT_I8MM and the "
                   "SMMLA kernel is compiled and self-tested, but the batched "
                   "linear was told to stay on SDOT by MYNAH_QMAT_I8MM=0"
                 : "[predicate] src/qmat.c compiled the SMMLA kernel, but this "
                   "CPU reports no FEAT_I8MM (hw.optional.arm.FEAT_I8MM / "
                   "HWCAP2_I8MM), so the batched linear stays on SDOT";
    }
    return on;
#else
    if (why != NULL)
        *why = "[predicate] src/qmat.c: no SMMLA kernel compiled for this "
               "target (aarch64 with clang or GCC >= 10 only)";
    return 0;
#endif
}

/* MYNAH_FUSED_GREEDY steers whether the engine fuses the head projection with
 * the constrained argmax.  The fused kernel is mynah_qmat_greedy_argmax*, so
 * the switch belongs to this module and the engine asks rather than re-reading
 * the environment behind the report's back. */
int mynah_qmat_fused_greedy_enabled(void) {
    const char *env = getenv("MYNAH_FUSED_GREEDY");
    return env == NULL || strcmp(env, "0") != 0;
}

/* Which half kernel is about to run, in the same shape as
 * mynah_qmat_int8_kernel(): the module's own answer, not "is f16 compiled".
 * The difference matters most where it used to be invisible -- on x86, where
 * before this existed the honest answer was "none, and the cache quietly
 * downgraded you to f32". */
const char *mynah_qmat_f16_kernel(const char **why) {
    switch (qmat_f16_kernel()) {
    case QMAT_F16K_NEON:
        if (why != NULL)
            *why = "[predicate] mynah_qmat_f16_kernel(): NEON vcvt_f32_f16, "
                   "four weight rows per activation load";
        return "neon";
    case QMAT_F16K_F16C:
        if (why != NULL)
            *why = "[predicate] mynah_qmat_f16_kernel(): AVX2 + F16C "
                   "VCVTPH2PS, four weight rows per activation load (target "
                   "attribute, no build flag needed)";
        return "f16c";
    case QMAT_F16K_SCALAR:
        if (why != NULL)
            *why = "[predicate] mynah_qmat_f16_kernel(): the half kernel is "
                   "compiled but this host has no F16C (or MYNAH_QMAT_F16C=0), "
                   "so the portable bit-twiddling convert runs -- still half "
                   "the weight bytes, at scalar convert cost";
        return "scalar";
    default:
        if (why != NULL)
            *why = "[predicate] mynah_qmat_f16_kernel(): no half weight type "
                   "compiled for this target, so MYNAH_QUANT=f16 is downgraded "
                   "to exact f32 by mynah_qmat_cache_new()";
        return "off";
    }
}

static int probe_row4(const char **why) {
    mynah_qmat_cache *c = mynah_qmat_cache_new(-1);
    if (c == NULL) return -1;
    const int on = mynah_qmat_cache_row4(c);
    mynah_qmat_cache_free(c);
    if (why != NULL) {
        *why = on ? "[predicate] mynah_qmat_cache_row4(): four output rows per "
                    "activation load in matvec_q8/matvec_q4/matvec_f16"
                  : "[predicate] mynah_qmat_cache_row4(): MYNAH_QMAT_SINGLE_ROW "
                    "is set, so every row runs its own dot -- the rollback path";
    }
    return on;
}

/* The report has no shape of its own, so it asks about the shape that decides
 * something: a full-size greedy projection.  4096 x 768 f32 is 12 MB, far over
 * the 256 KiB threshold, so the answer isolates the env + thread-count half of
 * the gate, and the reason states the threshold the other half applies. */
static int probe_argmax_mt(const char **why) {
    static char text[240];
    const char *clause = NULL;
    const int on = mynah_qmat_argmax_mt_resolved(4096u, 768u, &clause);
    snprintf(text, sizeof text,
             "[predicate] mynah_qmat_argmax_mt_resolved(4096, 768): %s -- %s. "
             "Shape-dependent by design; a smaller projection still runs "
             "serial on the same binary",
             on ? "threaded" : "serial", clause);
    if (why != NULL) *why = text;
    return on;
}

static int probe_quant_type(char *out, size_t capacity, const char **why) {
    static char text[240];
    mynah_qmat_cache *c = mynah_qmat_cache_new(-1);
    if (c == NULL) return -1;
    const int qtype = mynah_qmat_cache_qtype(c);
    mynah_qmat_cache_free(c);
    snprintf(out, capacity, "%s", qtype == QMAT_F32 ? "off"
                                                    : mynah_qmat_qtype_name(qtype));
    snprintf(text, sizeof text,
             "[predicate] mynah_qmat_cache_new(-1) + mynah_qmat_cache_qtype(): "
             "the module's own answer, named rather than reduced to ON. A type "
             "this build cannot represent is silently downgraded to f32 here, "
             "and this row is where that shows");
    if (why != NULL) *why = text;
    return 0;
}

static int probe_f16_kernel(char *out, size_t capacity, const char **why) {
    const char *reason = NULL;
    snprintf(out, capacity, "%s", mynah_qmat_f16_kernel(&reason));
    if (why != NULL) *why = reason;
    return 0;
}

static int probe_int8_kernel(char *out, size_t capacity, const char **why) {
    const char *reason = NULL;
    snprintf(out, capacity, "%s", mynah_qmat_int8_kernel(&reason));
    if (why != NULL) *why = reason;
    return 0;
}

static int probe_avx512vnni(const char **why) {
    const char *reason = NULL;
    const int on = strcmp(mynah_qmat_int8_kernel(&reason), "avx512vnni") == 0;
    if (why != NULL) *why = on ? reason
        : "[predicate] mynah_qmat_int8_kernel(): the EVEX VPDPBUSD kernel is "
          "compiled (target attribute, no build flag needed) but this host or "
          "MYNAH_QMAT_VNNI did not resolve to it";
    return on;
}

static int probe_avxvnni(const char **why) {
    const char *reason = NULL;
    const int on = strcmp(mynah_qmat_int8_kernel(&reason), "avxvnni") == 0;
    if (why != NULL) *why = on ? reason
        : "[predicate] mynah_qmat_int8_kernel(): the VEX VPDPBUSD kernel is "
          "compiled but not resolved -- either AVX-512 VNNI won, or this CPU "
          "has neither";
    return on;
}

static int probe_i8mm(const char **why) { return mynah_qmat_i8mm_enabled(why); }

static int probe_fused_greedy(const char **why) {
    const int on = mynah_qmat_fused_greedy_enabled();
    if (why != NULL) {
        *why = on ? "[predicate] mynah_qmat_fused_greedy_enabled(): the engine "
                    "fuses the head projection with the constrained argmax when "
                    "one greedy stream is live and no logits are needed"
                  : "[predicate] mynah_qmat_fused_greedy_enabled(): "
                    "MYNAH_FUSED_GREEDY=0, so the projection and the argmax run "
                    "as two passes over the weight";
    }
    return on;
}

/* ------------------------------------------------------- quantization groups
 *
 * WHICH weights are quantized is an engine decision, not a kernel one, so the
 * names live with the engine (see src/engine_pocket.c).  What lives here is
 * the single reading of the environment, so the dispatch report and the engine
 * cannot disagree about what was asked for -- the same reason quant.requested
 * is answered by this file rather than re-derived by dispatch.c.
 *
 * The string is deliberately NOT parsed here: an unknown group name has to
 * fail the engine that owns the names, loudly, instead of being silently
 * dropped by a parser that does not know them. */
const char *mynah_qmat_groups_spec(void) {
    const char *env = getenv("MYNAH_QUANT_GROUPS");
    if (env == NULL || env[0] == '\0') return "default";
    return env;
}

static int probe_quant_groups(char *out, size_t capacity, const char **reason) {
    const char *spec = mynah_qmat_groups_spec();
    snprintf(out, capacity, "%s", spec);
    *reason = (getenv("MYNAH_QUANT_GROUPS") == NULL)
                  ? "[predicate] mynah_qmat_groups_spec(): MYNAH_QUANT_GROUPS "
                    "unset, the engine's measured default set applies"
                  : "[predicate] mynah_qmat_groups_spec(): MYNAH_QUANT_GROUPS, "
                    "parsed and validated by the engine that owns the names";
    return 0;
}

void mynah_qmat_dispatch_probes(void) {
    mynah_dispatch_register_probe("quant.row4", probe_row4);
    mynah_dispatch_register_probe("quant.argmax_mt", probe_argmax_mt);
    mynah_dispatch_register_probe("isa.x86.avx512vnni", probe_avx512vnni);
    mynah_dispatch_register_probe("isa.x86.avxvnni", probe_avxvnni);
    mynah_dispatch_register_probe("isa.arm.i8mm", probe_i8mm);
    mynah_dispatch_register_probe("kernel.fused_greedy", probe_fused_greedy);
    mynah_dispatch_register_value_probe("quant.requested", probe_quant_type);
    mynah_dispatch_register_value_probe("quant.int8_kernel", probe_int8_kernel);
    /* quant.f16 was a boolean whose ON hid which of three kernels ran, and
     * whose OFF on x86 was the whole reason f16 was dead there.  A value probe
     * needs no row of its own: dispatch.c already consults one per id. */
    mynah_dispatch_register_value_probe("quant.f16", probe_f16_kernel);
    mynah_dispatch_register_value_probe("quant.groups", probe_quant_groups);
}
