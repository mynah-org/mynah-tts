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
    __fp16 *h = (__fp16 *)(void *)dst;
    for (size_t i = 0; i < n; ++i) h[i] = (__fp16)src[i];
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
static void quantize_weight_int4(const float *w, size_t n, size_t k,
                                 uint8_t *q, float *scales) {
    const size_t groups = k / QMAT_Q4_GROUP;
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
            const float scale = amax > 0.0f ? amax / 7.0f : 1.0f;
            srow[g] = scale;
            const float inv = 1.0f / scale;
            for (size_t j = 0; j < QMAT_Q4_GROUP; j += 2) {
                const float v0 = grp[j] * inv;
                const float v1 = grp[j + 1] * inv;
                int q0 = (int)(v0 >= 0.0f ? v0 + 0.5f : v0 - 0.5f);
                int q1 = (int)(v1 >= 0.0f ? v1 + 0.5f : v1 - 0.5f);
                if (q0 < -8) q0 = -8;
                if (q0 > 7) q0 = 7;
                if (q1 < -8) q1 = -8;
                if (q1 > 7) q1 = 7;
                qrow[(g * QMAT_Q4_GROUP + j) / 2] = (uint8_t)((q0 + 8) | ((q1 + 8) << 4));
            }
        }
    }
}

/* Per-vector absmax activation quantization; returns the activation scale. */
static float quantize_act_int8(int8_t *qx, const float *x, size_t k) {
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

/* The same quantizer with the +128 already applied, so the VNNI loop never
 * pays for the bias.  It must agree with quantize_act_int8 element for element
 * -- qu[i] == (uint8_t)(qx[i] + 128) -- which is why the rounding and clamping
 * are written here identically rather than shared through a callback. */
static float quantize_act_u8(uint8_t *qu, const float *x, size_t k) {
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

static float dot_q8(const void *qa, float sx, const int8_t *w, float ws,
                    int32_t rowsum, size_t k, int level) {
    return (float)dot_q8_i32(qa, w, rowsum, k, level) * ws * sx;
}

/* qx is the int8 activation; q is the packed INT4 weight group row; scales has
 * one entry per group of 32.  k must be a multiple of 32. */
static float dot_q4(const int8_t *qx, float sx, const uint8_t *q, const float *scales, size_t k) {
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
        acc += (float)vaddvq_s32(ig) * scales[g];
    }
    return acc * sx;
#else
    float acc = 0.0f;
    for (size_t g = 0; g < groups; ++g) {
        int32_t gi = 0;
        for (size_t j = 0; j < QMAT_Q4_GROUP; j += 2) {
            const uint8_t b = q[g * 16 + j / 2];
            const int lo = (int)(b & 0x0F) - 8;
            const int hi = (int)(b >> 4) - 8;
            gi += lo * (int32_t)qx[g * 32 + j] + hi * (int32_t)qx[g * 32 + j + 1];
        }
        acc += (float)gi * scales[g];
    }
    return acc * sx;
#endif
}

/* Decode is a stream of matrix-vector products.  On ARM, keep four independent
 * output rows in flight so SDOT latency is hidden and the quantized activation
 * vector is loaded once for four weight rows.  Each row retains the same
 * accumulation order as dot_q8/dot_q4; the scalar tail is the reference path. */
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
                out[row + r] = (float)s[r] * scales[row + r] * sx +
                               (bias == NULL ? 0.0f : bias[row + r]);
            }
        }
        for (; row < rows; ++row) {
            float value = dot_q8(xu, sx, weights + row * cols, scales[row],
                                 rowsum[row], cols, level);
            if (bias != NULL) value += bias[row];
            out[row] = value;
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
        out[row] = (float)s0 * scales[row] * sx + (bias == NULL ? 0.0f : bias[row]);
        out[row + 1u] = (float)s1 * scales[row + 1u] * sx +
                        (bias == NULL ? 0.0f : bias[row + 1u]);
        out[row + 2u] = (float)s2 * scales[row + 2u] * sx +
                        (bias == NULL ? 0.0f : bias[row + 2u]);
        out[row + 3u] = (float)s3 * scales[row + 3u] * sx +
                        (bias == NULL ? 0.0f : bias[row + 3u]);
    }
#elif defined(MYNAH_QMAT_AVX2)
    for (; row + 4u <= rows; row += 4u) {
        const int32_t s0 = dot_q8_i32_avx2(qx, weights + row * cols, cols);
        const int32_t s1 = dot_q8_i32_avx2(qx, weights + (row + 1u) * cols, cols);
        const int32_t s2 = dot_q8_i32_avx2(qx, weights + (row + 2u) * cols, cols);
        const int32_t s3 = dot_q8_i32_avx2(qx, weights + (row + 3u) * cols, cols);
        out[row] = (float)s0 * scales[row] * sx + (bias == NULL ? 0.0f : bias[row]);
        out[row + 1u] = (float)s1 * scales[row + 1u] * sx +
                        (bias == NULL ? 0.0f : bias[row + 1u]);
        out[row + 2u] = (float)s2 * scales[row + 2u] * sx +
                        (bias == NULL ? 0.0f : bias[row + 2u]);
        out[row + 3u] = (float)s3 * scales[row + 3u] * sx +
                        (bias == NULL ? 0.0f : bias[row + 3u]);
    }
#endif
    for (; row < rows; ++row) {
        float value = dot_q8(qx, sx, weights + row * cols, scales[row], 0,
                             cols, QMAT_U8_OFF);
        if (bias != NULL) value += bias[row];
        out[row] = value;
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
                (accumulator) += (float)vaddvq_s32(dot) * (scale)[group]; \
            } while (0)
            Q4_DOT_ROW(w0, s0, a0);
            Q4_DOT_ROW(w1, s1, a1);
            Q4_DOT_ROW(w2, s2, a2);
            Q4_DOT_ROW(w3, s3, a3);
#undef Q4_DOT_ROW
        }
        out[row] = a0 * sx + (bias == NULL ? 0.0f : bias[row]);
        out[row + 1u] = a1 * sx + (bias == NULL ? 0.0f : bias[row + 1u]);
        out[row + 2u] = a2 * sx + (bias == NULL ? 0.0f : bias[row + 2u]);
        out[row + 3u] = a3 * sx + (bias == NULL ? 0.0f : bias[row + 3u]);
    }
#endif
    for (; row < rows; ++row) {
        float value = dot_q4(qx, sx, weights + row * (cols / 2u),
                             scales + row * (cols / QMAT_Q4_GROUP), cols);
        if (bias != NULL) value += bias[row];
        out[row] = value;
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
 * in different orders, which CLAUDE.md's numerical rules allow across ISAs and
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

mynah_qmat_cache *mynah_qmat_cache_new(int enabled) {
    mynah_qmat_cache *c = (mynah_qmat_cache *)calloc(1, sizeof(*c));
    if (c == NULL) return NULL;
    int qtype = QMAT_F32;
    if (enabled < 0) {
        const char *env = getenv("MYNAH_QUANT");
        if (env != NULL && strcmp(env, "int8") == 0) qtype = QMAT_INT8;
        else if (env != NULL && strcmp(env, "int4") == 0) qtype = QMAT_INT4;
        else if (env != NULL && strcmp(env, "f16") == 0) qtype = QMAT_F16;
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
    if (e->name == NULL) return NULL;
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
 * The accumulation is exact int32, and the float epilogue is written exactly
 * as matvec_q8's, so a batched row is BIT-IDENTICAL to the same row computed
 * alone -- which self_test_batched already asserts with memcmp. */
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
        out0[row]      = (float)s00 * scales[row] * sx0 +
                         (bias == NULL ? 0.0f : bias[row]);
        out0[row + 1u] = (float)s10 * scales[row + 1u] * sx0 +
                         (bias == NULL ? 0.0f : bias[row + 1u]);
        out1[row]      = (float)s01 * scales[row] * sx1 +
                         (bias == NULL ? 0.0f : bias[row]);
        out1[row + 1u] = (float)s11 * scales[row + 1u] * sx1 +
                         (bias == NULL ? 0.0f : bias[row + 1u]);
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
 * gets would depend on which other requests it happened to batch with.  It is:
 * every (weight row, activation) pair runs the same kernel over the same k in
 * the same order, and only the order of independent pairs changes.  The f32
 * path has no such guarantee -- sgemm may accumulate differently for M=B than
 * for M=1 -- so it stays one call per row. */
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
        j->qx != NULL && qmat_i8mm()) {
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
            } else if (e->qtype == QMAT_F16) {
                got = (double)qmat_f16_to_f32(e->f16[i * k + j]);
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
                float value = e->qtype == QMAT_INT8
                                  ? dot_q8(qx, sx, e->q8 + row * k, e->scales[row],
                                           e->rowsum[row], k, level)
                                  : dot_q4((const int8_t *)qx, sx,
                                           e->q4 + row * (k / 2u),
                                           e->scales + row * (k / QMAT_Q4_GROUP), k);
                if (bias != NULL) value += bias[row];
                orow[row] = value;
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

int mynah_qmat_linear_batched(mynah_qmat_cache *cache, const mynah_backend *backend,
                              const char *name, const float *weight_data,
                              const float *const *in_rows, float *const *out_rows,
                              size_t batch, size_t k, size_t n, const float *bias,
                              int8_t *qx_scratch, float *sx_scratch,
                              char *error, size_t error_capacity) {
    if (batch == 0u) return 0;
    if (batch == 1u) {
        return mynah_qmat_linear_resolved(cache, backend, name, weight_data,
                                          in_rows[0], out_rows[0], 1u, k, n, bias,
                                          error, error_capacity);
    }
    const int use_q = cache != NULL && cache->qtype != QMAT_F32 && k <= QMAT_K_MAX &&
                      cache->use_row4 && qx_scratch != NULL && sx_scratch != NULL;
    const qmat_entry *e = NULL;
    if (use_q) {
        pthread_mutex_lock(&cache->mutex);
        e = cache_lookup(cache, name);
        if (e == NULL) e = cache_insert(cache, name, weight_data, n, k, cache->qtype);
        pthread_mutex_unlock(&cache->mutex);
    }
    if (e == NULL) {
        /* No bit-exact batching available here: keep every row on the exact
         * path it would have taken alone. */
        for (size_t b = 0; b < batch; ++b) {
            if (mynah_qmat_linear_resolved(cache, backend, name, weight_data,
                                           in_rows[b], out_rows[b], 1u, k, n, bias,
                                           error, error_capacity) != 0) {
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
                                       QMAT_U8_OFF)
                              : dot_q4(qx, sx, q4 + i * (K / 2), scales4 + i * (K / QMAT_Q4_GROUP), K);
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
 * exercised: 200 = 3*64 + 8 with 13 rows = 3*4 + 1. */
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

    matvec_q8(ref, qs, ss, q8, scales, NULL, bias, n, k, QMAT_U8_OFF);
    matvec_q8(got, qu, su, q8, scales, rowsum, bias, n, k, level);
    for (size_t i = 0; i < n; ++i) {
        if (memcmp(&ref[i], &got[i], sizeof(float)) != 0) {
            if (error != NULL && error_capacity > 0) {
                snprintf(error, error_capacity,
                         "qmat u8 level=%d not bit-identical at row %zu of %zu "
                         "(k=%zu): %.9g vs %.9g",
                         level, i, n, k, (double)ref[i], (double)got[i]);
            }
            goto done;
        }
    }
    /* And the single-row entry, which the non-row4 path uses. */
    for (size_t i = 0; i < n; ++i) {
        const float a = dot_q8(qs, ss, q8 + i * k, scales[i], 0, k, QMAT_U8_OFF);
        const float b = dot_q8(qu, su, q8 + i * k, scales[i], rowsum[i], k, level);
        if (memcmp(&a, &b, sizeof(float)) != 0) {
            if (error != NULL && error_capacity > 0) {
                snprintf(error, error_capacity,
                         "qmat u8 level=%d single-row differs at %zu: %.9g vs %.9g",
                         level, i, (double)a, (double)b);
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
 * same int32 as two separate SDOT rows -- and therefore on the same float.
 * Bit-identity is the contract self_test_batched already relies on; this
 * checks the kernel directly, including the odd row and the k tail. */
static int self_test_i8mm_identity(char *error, size_t error_capacity) {
    if (!qmat_i8mm()) return 0;
    enum { N = 13, K = 37 };
    int status = -1;
    float *w = (float *)malloc((size_t)N * K * sizeof(float));
    float *x = (float *)malloc(2u * K * sizeof(float));
    float *bias = (float *)malloc((size_t)N * sizeof(float));
    float *ref = (float *)malloc(2u * N * sizeof(float));
    float *got = (float *)malloc(2u * N * sizeof(float));
    int8_t *q8 = (int8_t *)malloc((size_t)N * K);
    float *scales = (float *)malloc((size_t)N * sizeof(float));
    int8_t *qx = (int8_t *)malloc(2u * K);
    if (w == NULL || x == NULL || bias == NULL || ref == NULL || got == NULL ||
        q8 == NULL || scales == NULL || qx == NULL) {
        if (error != NULL && error_capacity > 0)
            snprintf(error, error_capacity, "qmat i8mm identity out of memory");
        goto done;
    }
    for (size_t i = 0; i < (size_t)N * K; ++i)
        w[i] = sinf(0.021f * (float)i) * (0.5f + 0.5f * cosf(0.0013f * (float)i));
    for (size_t i = 0; i < 2u * K; ++i) x[i] = cosf(0.017f * (float)i) - 0.2f;
    for (size_t i = 0; i < (size_t)N; ++i) bias[i] = (float)i * 0.03125f - 0.25f;
    quantize_weight_int8(w, N, K, q8, scales);
    const float s0 = quantize_act_int8(qx, x, K);
    const float s1 = quantize_act_int8(qx + K, x + K, K);
    matvec_q8(ref, qx, s0, q8, scales, NULL, bias, N, K, QMAT_U8_OFF);
    matvec_q8(ref + N, qx + K, s1, q8, scales, NULL, bias, N, K, QMAT_U8_OFF);
    matvec_q8_pair_i8mm(got, got + N, qx, qx + K, s0, s1, q8, scales, bias, N, K);
    for (size_t i = 0; i < 2u * N; ++i) {
        if (memcmp(&ref[i], &got[i], sizeof(float)) != 0) {
            if (error != NULL && error_capacity > 0) {
                snprintf(error, error_capacity,
                         "qmat i8mm not bit-identical at %zu: %.9g vs %.9g",
                         i, (double)ref[i], (double)got[i]);
            }
            goto done;
        }
    }
    status = 0;
done:
    free(w); free(x); free(bias); free(ref); free(got);
    free(q8); free(scales); free(qx);
    return status;
}
#endif

int mynah_qmat_self_test(char *error, size_t error_capacity) {
    if (self_test_u8_identity(error, error_capacity) != 0) return -1;
#if defined(MYNAH_QMAT_ARM_I8MM)
    if (self_test_i8mm_identity(error, error_capacity) != 0) return -1;
#endif
    if (self_test_one(QMAT_INT8, error, error_capacity) != 0) return -1;
    if (self_test_one(QMAT_INT4, error, error_capacity) != 0) return -1;
    if (self_test_rows_blocked(QMAT_INT8, error, error_capacity) != 0) return -1;
    if (self_test_rows_blocked(QMAT_INT4, error, error_capacity) != 0) return -1;
    if (self_test_batched(QMAT_INT8, error, error_capacity) != 0) return -1;
    if (self_test_batched(QMAT_INT4, error, error_capacity) != 0) return -1;
#if defined(MYNAH_QMAT_F16)
    if (self_test_f16_convert(error, error_capacity) != 0) return -1;
    if (self_test_f16(error, error_capacity) != 0) return -1;
    if (self_test_rows_blocked(QMAT_F16, error, error_capacity) != 0) return -1;
    if (self_test_batched(QMAT_F16, error, error_capacity) != 0) return -1;
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

int mynah_qmat_i8mm_enabled(const char **why) {
#if defined(MYNAH_QMAT_ARM_I8MM)
    const int on = qmat_i8mm();
    if (why != NULL) {
        *why = on ? "[predicate] src/qmat.c matvec_q8_pair_i8mm: SMMLA "
                    "(vmmlaq_s32), two activations x two weight rows x eight k "
                    "per instruction, in the weight-stationary batched linear. "
                    "Single-vector decode stays on SDOT, where SMMLA would "
                    "waste half its lanes"
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
