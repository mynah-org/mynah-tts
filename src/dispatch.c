/* dispatch.c — the dispatch report.  See dispatch.h for why it exists and for
 * the rule that governs every row: `resolved` is obtained by CALLING something,
 * never by re-deriving it from `compiled && supported`.
 *
 * Three things in here are worth reading before trusting a row.
 *
 * 1. The CPU probes are real.  On arm64 Darwin they are sysctlbyname()
 *    ("hw.optional.arm.FEAT_DotProd" and friends); on arm64 Linux they are
 *    getauxval(AT_HWCAP/AT_HWCAP2); on x86 they are CPUID leaf 7 plus XGETBV,
 *    so a feature the OS has not enabled for XSAVE reports supported=no rather
 *    than being claimed from a CPUID bit the process cannot use.
 *
 * 2. Several rows exist only to say that something is NOT there.  AMX and
 *    BF16 still have rows whose `compiled` is "no" and whose reason names the
 *    intrinsic that is missing from src/.  Those rows are the reason this file
 *    was written: the absence has to be as loud as a presence, or the next
 *    benchmark write-up invents the attribution again.  VNNI and i8mm have
 *    since acquired real kernels, and the difference is visible here: their
 *    rows moved from [gate] "NOT IMPLEMENTED" to [predicate], resolved by
 *    src/qmat.c's own CPUID / sysctl probe rather than by a build flag.
 *
 * 3. The int8/int4/f16 rows observe src/qmat.c rather than guessing at it.
 *    mynah_qmat_cache_new(qtype) followed by mynah_qmat_cache_enabled() is the
 *    module's own resolution: ask for f16 on a build without NEON half
 *    converts and the cache silently downgrades itself to f32, which is
 *    exactly the kind of silent fallback the report must surface.  The qtype
 *    codes are the documented ones of src/qmat.c:35 (1=int8, 2=int4, 3=f16);
 *    the f16 row doubles as the DRIFT canary for the mirrored compile gates.
 */
#include "dispatch.h"

#include "backend.h"
#include "kernels.h"
#include "mynah_tts.h"
#include "qmat.h"
#include "threads.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif

#if defined(__aarch64__) && defined(__linux__)
#if defined(__has_include)
#if __has_include(<sys/auxv.h>)
#include <sys/auxv.h>
#define MYNAH_DISPATCH_HAVE_AUXV 1
#endif
#endif
#endif

#if defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>
#endif

/* The qtype codes src/qmat.c documents for mynah_qmat_cache_new().  Named here
 * so the coupling is visible instead of being three magic integers. */
enum { QMAT_PROBE_INT8 = 1, QMAT_PROBE_INT4 = 2, QMAT_PROBE_F16 = 3 };

#ifndef MYNAH_GIT_REV
#define MYNAH_GIT_REV "unknown"
#endif
#ifndef MYNAH_SIMD_PROFILE
#define MYNAH_SIMD_PROFILE "unset"
#endif

/* ======================================================================
 * Runtime CPU probes
 * ====================================================================== */

/* Tri-state: 1 yes, 0 no, -1 could not be determined on this platform. */
typedef int tri;

#if defined(__APPLE__) && defined(__aarch64__)
/* Only the AArch64 probes read boolean sysctls; an Intel Mac build must not
 * carry an unused one. */
static tri sysctl_flag(const char *name) {
    int value = 0;
    size_t size = sizeof(value);
    if (sysctlbyname(name, &value, &size, NULL, 0) != 0) return -1;
    return value != 0 ? 1 : 0;
}
#endif

#if defined(__APPLE__)
static long sysctl_long(const char *name) {
    int64_t v64 = 0;
    size_t size = sizeof(v64);
    if (sysctlbyname(name, &v64, &size, NULL, 0) == 0 && size == sizeof(v64)) {
        return (long)v64;
    }
    int v32 = 0;
    size = sizeof(v32);
    if (sysctlbyname(name, &v32, &size, NULL, 0) == 0) return (long)v32;
    return -1;
}
#endif

#if defined(MYNAH_DISPATCH_HAVE_AUXV)
/* Defined locally so the build does not depend on which kernel headers are
 * installed; these bit positions are ABI and cannot move. */
#define MYNAH_HWCAP_ASIMD    (1UL << 1)
#define MYNAH_HWCAP_ASIMDHP  (1UL << 10)
#define MYNAH_HWCAP_ASIMDDP  (1UL << 20)
#define MYNAH_HWCAP_SVE      (1UL << 22)
#define MYNAH_HWCAP2_SVE2    (1UL << 1)
#define MYNAH_HWCAP2_SVEI8MM (1UL << 9)
#define MYNAH_HWCAP2_SVEBF16 (1UL << 12)
#define MYNAH_HWCAP2_I8MM    (1UL << 13)
#define MYNAH_HWCAP2_BF16    (1UL << 14)
#endif

#if defined(__x86_64__) || defined(__i386__)
static int x86_cpuid(unsigned leaf, unsigned sub, unsigned regs[4]) {
    unsigned max = __get_cpuid_max(leaf & 0x80000000u, NULL);
    if (leaf > max) return 0;
    __cpuid_count(leaf, sub, regs[0], regs[1], regs[2], regs[3]);
    return 1;
}

/* XCR0, so a CPUID bit the OS has not enabled is not reported as usable. */
static unsigned long long x86_xcr0(void) {
    unsigned regs[4];
    if (!x86_cpuid(1, 0, regs)) return 0;
    if (!(regs[2] & (1u << 27))) return 0; /* OSXSAVE */
    unsigned lo = 0, hi = 0;
    __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    return ((unsigned long long)hi << 32) | lo;
}

static int x86_os_avx(void) {
    const unsigned long long x = x86_xcr0();
    return (x & 0x6ull) == 0x6ull;                /* XMM | YMM */
}
static int x86_os_avx512(void) {
    const unsigned long long x = x86_xcr0();
    return (x & 0xe6ull) == 0xe6ull;              /* + opmask, ZMM hi256, hi16 */
}
static int x86_os_amx(void) {
    const unsigned long long x = x86_xcr0();
    return (x & 0x60000ull) == 0x60000ull;        /* XTILECFG | XTILEDATA */
}

static tri x86_feature(unsigned leaf, unsigned sub, int reg, int bit, int os_ok) {
    unsigned regs[4];
    if (!x86_cpuid(leaf, sub, regs)) return -1;
    if (!(regs[reg] & (1u << bit))) return 0;
    return os_ok ? 1 : 0;
}
#endif

static tri cpu_has_neon(void) {
#if defined(__aarch64__)
#if defined(__APPLE__)
    const tri t = sysctl_flag("hw.optional.neon");
    return t < 0 ? 1 : t;   /* AdvSIMD is mandatory in AArch64 */
#elif defined(MYNAH_DISPATCH_HAVE_AUXV)
    return (getauxval(AT_HWCAP) & MYNAH_HWCAP_ASIMD) ? 1 : 0;
#else
    return 1;
#endif
#elif defined(__ARM_NEON)
    return 1;
#else
    return 0;
#endif
}

static tri cpu_has_dotprod(void) {
#if defined(__aarch64__) && defined(__APPLE__)
    tri t = sysctl_flag("hw.optional.arm.FEAT_DotProd");
    if (t < 0) t = sysctl_flag("hw.optional.armv8_3_compnum") >= 0 ? 1 : -1;
    return t;
#elif defined(MYNAH_DISPATCH_HAVE_AUXV)
    return (getauxval(AT_HWCAP) & MYNAH_HWCAP_ASIMDDP) ? 1 : 0;
#else
    return -1;
#endif
}

static tri cpu_has_fp16(void) {
#if defined(__aarch64__) && defined(__APPLE__)
    return sysctl_flag("hw.optional.arm.FEAT_FP16");
#elif defined(MYNAH_DISPATCH_HAVE_AUXV)
    return (getauxval(AT_HWCAP) & MYNAH_HWCAP_ASIMDHP) ? 1 : 0;
#else
    return -1;
#endif
}

static tri cpu_has_i8mm(void) {
#if defined(__aarch64__) && defined(__APPLE__)
    return sysctl_flag("hw.optional.arm.FEAT_I8MM");
#elif defined(MYNAH_DISPATCH_HAVE_AUXV)
    return (getauxval(AT_HWCAP2) & MYNAH_HWCAP2_I8MM) ? 1 : 0;
#else
    return -1;
#endif
}

/* SVE and its three extensions.  Apple silicon has none of them and exposes no
 * sysctl for them, so the Darwin answer is a definite 0 rather than "?": a
 * question mark there would suggest a probe worth writing, and there is not
 * one.  On Linux they are HWCAP bits, which is also how the kernel decides
 * what to print in /proc/cpuinfo -- the same source tools/simd-auto.sh reads
 * at build time, so the two halves of the E4-15 double test agree by
 * construction. */
static tri cpu_has_sve(void) {
#if defined(__aarch64__) && defined(__APPLE__)
    return 0;
#elif defined(MYNAH_DISPATCH_HAVE_AUXV)
    return (getauxval(AT_HWCAP) & MYNAH_HWCAP_SVE) ? 1 : 0;
#elif defined(__aarch64__)
    return -1;
#else
    return 0;
#endif
}

static tri cpu_has_sve2(void) {
#if defined(__aarch64__) && defined(__APPLE__)
    return 0;
#elif defined(MYNAH_DISPATCH_HAVE_AUXV)
    return (getauxval(AT_HWCAP2) & MYNAH_HWCAP2_SVE2) ? 1 : 0;
#elif defined(__aarch64__)
    return -1;
#else
    return 0;
#endif
}

static tri cpu_has_svei8mm(void) {
#if defined(__aarch64__) && defined(__APPLE__)
    return 0;
#elif defined(MYNAH_DISPATCH_HAVE_AUXV)
    return (getauxval(AT_HWCAP2) & MYNAH_HWCAP2_SVEI8MM) ? 1 : 0;
#elif defined(__aarch64__)
    return -1;
#else
    return 0;
#endif
}

static tri cpu_has_svebf16(void) {
#if defined(__aarch64__) && defined(__APPLE__)
    return 0;
#elif defined(MYNAH_DISPATCH_HAVE_AUXV)
    return (getauxval(AT_HWCAP2) & MYNAH_HWCAP2_SVEBF16) ? 1 : 0;
#elif defined(__aarch64__)
    return -1;
#else
    return 0;
#endif
}

static tri cpu_has_bf16(void) {
#if defined(__aarch64__) && defined(__APPLE__)
    return sysctl_flag("hw.optional.arm.FEAT_BF16");
#elif defined(MYNAH_DISPATCH_HAVE_AUXV)
    return (getauxval(AT_HWCAP2) & MYNAH_HWCAP2_BF16) ? 1 : 0;
#elif defined(__x86_64__) || defined(__i386__)
    return x86_feature(7, 1, 0, 5, x86_os_avx512());   /* AVX512-BF16, EAX[5] */
#else
    return -1;
#endif
}

#if defined(__x86_64__) || defined(__i386__)
static tri cpu_has_avx2(void)        { return x86_feature(7, 0, 1,  5, x86_os_avx()); }
static tri cpu_has_fma(void)         { unsigned r[4]; if (!x86_cpuid(1, 0, r)) return -1;
                                       return (r[2] & (1u << 12)) && x86_os_avx() ? 1 : 0; }
static tri cpu_has_avx512f(void)     { return x86_feature(7, 0, 1, 16, x86_os_avx512()); }
static tri cpu_has_avx512bw(void)    { return x86_feature(7, 0, 1, 30, x86_os_avx512()); }
static tri cpu_has_avx512vl(void)    { return x86_feature(7, 0, 1, 31, x86_os_avx512()); }
static tri cpu_has_avx512vnni(void)  { return x86_feature(7, 0, 2, 11, x86_os_avx512()); }
static tri cpu_has_avxvnni(void)     { return x86_feature(7, 1, 0,  4, x86_os_avx()); }
static tri cpu_has_amx_int8(void)    { return x86_feature(7, 0, 3, 25, x86_os_amx()); }
/* AVX512-BF16 is leaf 7 SUBLEAF 1, EAX bit 5 -- the same subleaf as AVX-VNNI
 * and a different one from the F/BW/VL bits, which is where this check is
 * usually got wrong. */
static tri cpu_has_avx512bf16(void)  { return x86_feature(7, 1, 0,  5, x86_os_avx512()); }
#else
static tri cpu_has_avx2(void)        { return 0; }
static tri cpu_has_fma(void)         { return 0; }
static tri cpu_has_avx512f(void)     { return 0; }
static tri cpu_has_avx512bw(void)    { return 0; }
static tri cpu_has_avx512vl(void)    { return 0; }
static tri cpu_has_avx512vnni(void)  { return 0; }
static tri cpu_has_avxvnni(void)     { return 0; }
static tri cpu_has_amx_int8(void)    { return 0; }
static tri cpu_has_avx512bf16(void)  { return 0; }
#endif

/* `isa.arm.bf16` asks whether a bf16 WEIGHT gets a vector path, and on x86 that
 * question is not cpu_has_bf16().  The x86 kernel widens bf16 to f32 with a
 * 16-bit shift and multiplies, so what it needs is AVX2 -- AVX512-BF16 only
 * decides WHICH x86 tier runs, and that is isa.x86.avx512bf16's row.  Asking
 * cpu_has_bf16() here produced `supported no, resolved ON` on every AVX2 host
 * without VDPBF16PS: a row contradicting itself. */
static tri cpu_has_bf16_path(void) {
#if defined(__x86_64__) || defined(__i386__)
    return cpu_has_avx2();
#else
    return cpu_has_bf16();
#endif
}

/* ======================================================================
 * E4-12: the fatal ISA guard
 *
 * A binary compiled with -mavx2 and started on a CPU without AVX2 dies with
 * SIGILL at the first vpaddd.  There is no message, no exit code that means
 * anything, and no line in any log that names the cause; on a fleet it looks
 * like a crash loop on some hosts and not others.  The whole diagnosis is
 * fifteen lines of CPUID, and it has to run before anything else so that what
 * the operator sees is the mismatch rather than a signal.
 *
 * WHICH MACROS THIS TESTS, AND WHY THEY ARE NOT THE ROW MACROS.  The rows
 * above ask "which kernel will dispatch", and answer through
 * MYNAH_DISPATCH_HAS_*, which folds in MYNAH_DISABLE_SIMD.  That is the wrong
 * question here.  SIMD=scalar compiles out our intrinsics but does not stop
 * the compiler from autovectorizing with whatever -march allowed -- so
 * `make SIMD=scalar` on a host whose -march=native implies SVE still produces
 * a binary full of SVE, and it is the raw predefined macros, not our gates,
 * that say so.  This is the one place in this file entitled to read them.
 *
 * ONLY A DEFINITE ABSENCE IS FATAL.  Each probe returns 1 / 0 / -1 and the
 * guard fires on 0 alone.  A "?" means we could not ask -- an OS with no
 * getauxval, a CPUID leaf the hypervisor hid -- and turning that into an exit
 * would take a working process down over our own ignorance.  It is exactly the
 * emulator case from .work/dtype-and-fallbacks.md, where the layer did not
 * expose F16C through CPUID.
 * ====================================================================== */

/* Defined with the rest of the report below; the guard runs before any of it. */
static const char *arch_name(void);

typedef struct {
    const char *name;
    tri (*probe)(void);
} isa_requirement;

static const isa_requirement *isa_requirements(void) {
    /* Listed only when the compiler was allowed to emit the unit anywhere in
     * this translation unit's build.  The sentinel is unconditional so the
     * array is never zero-length, which -Wpedantic rejects. */
    static const isa_requirement table[] = {
#if defined(__AVX512F__)
        { "AVX-512F", cpu_has_avx512f },
#endif
#if defined(__AVX512BW__)
        { "AVX-512BW", cpu_has_avx512bw },
#endif
#if defined(__AVX512VL__)
        { "AVX-512VL", cpu_has_avx512vl },
#endif
#if defined(__AVX2__)
        { "AVX2", cpu_has_avx2 },
#endif
#if defined(__FMA__)
        { "FMA", cpu_has_fma },
#endif
#if defined(__ARM_FEATURE_SVE)
        { "SVE", cpu_has_sve },
#endif
#if defined(__ARM_FEATURE_SVE2)
        { "SVE2", cpu_has_sve2 },
#endif
#if defined(__ARM_FEATURE_MATMUL_INT8)
        { "i8mm", cpu_has_i8mm },
#endif
#if defined(__ARM_FEATURE_DOTPROD)
        { "dotprod", cpu_has_dotprod },
#endif
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
        { "fp16", cpu_has_fp16 },
#endif
#if defined(__ARM_NEON) || defined(__aarch64__)
        { "AdvSIMD", cpu_has_neon },
#endif
        { NULL, NULL }
    };
    return table;
}

/* E14-1.  THE GUARD IS COMPILED FOR THE BASELINE, ON PURPOSE.
 *
 * Everything else in this translation unit is built with whatever the profile
 * asked for -- `-mavx2 -mfma`, or `-march=native` under SIMD=auto. The guard
 * may not be, and the reason is circular in a way that is easy to miss: it
 * exists to say "this binary needs an instruction your CPU does not have", and
 * if the compiler puts one of those instructions INSIDE THE GUARD, the process
 * dies on it before the message is printed. The operator then gets a SIGILL
 * with no text, which is precisely the outcome the guard was written to
 * replace.
 *
 * `target("arch=x86-64")` constrains only the instructions generated for these
 * two functions; the feature macros still describe the build, so what the guard
 * CHECKS is unchanged. Borrowed from ../qwen-tts, which carries the same
 * attribute on qwen_check_runtime_isa for the same reason
 * (.work/qwen-tts-kernel-reuse.md).
 *
 * It has never bitten us -- tests/x86_cross.sh proves the guard fires correctly
 * on a real no-AVX2 host -- and that is an argument for keeping it that way,
 * not for leaving it to luck: the failure mode is invisible until the one day
 * it is a silent crash on a customer's machine. */
#if (defined(__x86_64__) || defined(__i386__)) && \
    (defined(__GNUC__) || defined(__clang__))
#define MYNAH_ISA_GUARD_BASELINE __attribute__((target("arch=x86-64"), noinline))
#else
#define MYNAH_ISA_GUARD_BASELINE
#endif

/* AND THEREFORE NO snprintf IN THESE TWO FUNCTIONS, which is not a style
 * choice. Found by gcc 13 on an EPYC 9254, first build of this code on real
 * x86 hardware:
 *
 *   bits/stdio2.h: error: inlining failed in call to 'always_inline'
 *   'snprintf': target specific option mismatch
 *
 * With _FORTIFY_SOURCE on, glibc's snprintf is an __always_inline wrapper
 * compiled for the translation unit's target -- here -mavx2 -mfma -mf16c -- and
 * gcc refuses, correctly, to inline it into a function that has promised to
 * emit only baseline instructions. clang accepts it silently, which is why this
 * survived a full macOS build and a cross-compile and died on the first machine
 * that would actually have run it.
 *
 * The fix is the honest one rather than the quiet one. Turning fortification
 * off for this file would trade real hardening for a compile; formatting the
 * message in an unrestricted helper would put AVX2 back on the failure path,
 * which is the exact path this guard exists to survive. So the message is built
 * with bounded copies and no libc formatting at all -- it is only string
 * substitution, there is not one number in it, and a function that must run on
 * any x86-64 has no business calling into a header inlined for a wider one. */
MYNAH_ISA_GUARD_BASELINE
static size_t isa_guard_put(char *out, size_t cap, size_t o, const char *text) {
    if (out == NULL || cap == 0u || text == NULL) return o;
    while (text[0] != '\0' && o + 1u < cap) out[o++] = *text++;
    out[o] = '\0';
    return o;
}

/* What this CPU does have, for the second half of the message.  A mismatch
 * report that names only what is missing leaves the operator to guess which
 * build to fetch instead. */
MYNAH_ISA_GUARD_BASELINE
static void isa_guard_host(char *out, size_t cap) {
    static const isa_requirement known[] = {
#if defined(__x86_64__) || defined(__i386__)
        { "avx2", cpu_has_avx2 },       { "fma", cpu_has_fma },
        { "avx512f", cpu_has_avx512f }, { "avx512bw", cpu_has_avx512bw },
        { "avx512vl", cpu_has_avx512vl },
        { "avx512vnni", cpu_has_avx512vnni }, { "avxvnni", cpu_has_avxvnni },
        { "amx_int8", cpu_has_amx_int8 },
#else
        { "asimd", cpu_has_neon },      { "asimdhp", cpu_has_fp16 },
        { "asimddp", cpu_has_dotprod }, { "i8mm", cpu_has_i8mm },
        { "bf16", cpu_has_bf16 },       { "sve", cpu_has_sve },
        { "sve2", cpu_has_sve2 },       { "svei8mm", cpu_has_svei8mm },
        { "svebf16", cpu_has_svebf16 },
#endif
        { NULL, NULL }
    };
    size_t o = 0;
    out[0] = '\0';
    for (size_t i = 0; known[i].name != NULL && o + 1 < cap; ++i) {
        if (known[i].probe() != 1) continue;
        if (o > 0) o = isa_guard_put(out, cap, o, " ");
        o = isa_guard_put(out, cap, o, known[i].name);
    }
    if (out[0] == '\0')
        isa_guard_put(out, cap, 0, "(nothing this build knows how to probe)");
}

MYNAH_ISA_GUARD_BASELINE
int mynah_dispatch_isa_guard(char *error, size_t error_capacity) {
    const isa_requirement *req = isa_requirements();
    for (size_t i = 0; req[i].name != NULL; ++i) {
        if (req[i].probe() != 0) continue;     /* 1 = have it, -1 = cannot ask */
        if (error != NULL && error_capacity > 0) {
            char host[256];
            isa_guard_host(host, sizeof host);
            size_t o = 0;
            o = isa_guard_put(error, error_capacity, o, "this binary requires ");
            o = isa_guard_put(error, error_capacity, o, req[i].name);
            o = isa_guard_put(error, error_capacity, o,
                              " and this CPU does not have it. Built as SIMD=");
            o = isa_guard_put(error, error_capacity, o, MYNAH_SIMD_PROFILE);
            o = isa_guard_put(error, error_capacity, o, " (");
            o = isa_guard_put(error, error_capacity, o, arch_name());
            o = isa_guard_put(error, error_capacity, o, ", ");
            o = isa_guard_put(error, error_capacity, o, MYNAH_GIT_REV);
            o = isa_guard_put(error, error_capacity, o, "); the CPU reports: ");
            o = isa_guard_put(error, error_capacity, o, host);
            o = isa_guard_put(error, error_capacity, o,
                              ". Rebuild with a profile this host supports "
                              "(make SIMD=portable, or SIMD=avx2 for a "
                              "travelling x86 binary) -- without this check the "
                              "next instruction would have been SIGILL.");
            (void)o;
        }
        return -1;
    }
    return 0;
}

/* ======================================================================
 * Probe registry
 * ====================================================================== */

typedef struct {
    const char *id;
    mynah_dispatch_probe_fn fn;
} probe_slot;

static probe_slot g_probes[MYNAH_DISPATCH_MAX_ROWS];
static int g_probe_count;

typedef struct {
    const char *id;
    mynah_dispatch_value_probe_fn fn;
} value_probe_slot;

static value_probe_slot g_value_probes[MYNAH_DISPATCH_MAX_ROWS];
static int g_value_probe_count;

int mynah_dispatch_register_probe(const char *id, mynah_dispatch_probe_fn fn) {
    if (id == NULL || fn == NULL) return -1;
    for (int i = 0; i < g_probe_count; ++i) {
        if (strcmp(g_probes[i].id, id) == 0) { g_probes[i].fn = fn; return 0; }
    }
    if (g_probe_count >= MYNAH_DISPATCH_MAX_ROWS) return -1;
    g_probes[g_probe_count].id = id;
    g_probes[g_probe_count].fn = fn;
    ++g_probe_count;
    return 0;
}

int mynah_dispatch_register_value_probe(const char *id,
                                        mynah_dispatch_value_probe_fn fn) {
    if (id == NULL || fn == NULL) return -1;
    for (int i = 0; i < g_value_probe_count; ++i) {
        if (strcmp(g_value_probes[i].id, id) == 0) {
            g_value_probes[i].fn = fn;
            return 0;
        }
    }
    if (g_value_probe_count >= MYNAH_DISPATCH_MAX_ROWS) return -1;
    g_value_probes[g_value_probe_count].id = id;
    g_value_probes[g_value_probe_count].fn = fn;
    ++g_value_probe_count;
    return 0;
}

static mynah_dispatch_probe_fn probe_for(const char *id) {
    for (int i = 0; i < g_probe_count; ++i) {
        if (strcmp(g_probes[i].id, id) == 0) return g_probes[i].fn;
    }
    return NULL;
}

static mynah_dispatch_value_probe_fn value_probe_for(const char *id) {
    for (int i = 0; i < g_value_probe_count; ++i) {
        if (strcmp(g_value_probes[i].id, id) == 0) return g_value_probes[i].fn;
    }
    return NULL;
}

/* Every module that owns a dispatch decision, asked to register before the
 * table is built.  An explicit list rather than constructors: these objects go
 * into libmynah_tts.a, and a linker is entitled to drop an object whose only
 * contribution is a constructor nobody references -- which would silently turn
 * [predicate] rows back into UNKNOWN in exactly the builds least likely to be
 * looked at. */
static void register_module_probes(void) {
    mynah_qmat_dispatch_probes();
    mynah_kernels_dispatch_probes();
    mynah_backend_dispatch_probes();
    mynah_threads_dispatch_probes();
    mynah_conv1d_dispatch_probes();
    mynah_codec_dispatch_probes();
    mynah_seanet_dispatch_probes();
    mynah_sgemm_dispatch_probes();
    mynah_convq8_dispatch_probes();
}

/* ======================================================================
 * Row construction
 * ====================================================================== */

static const char *yn(int v)    { return v ? "yes" : "no"; }
static const char *yn3(tri v)   { return v < 0 ? "?" : (v ? "yes" : "no"); }
static const char *onoff(int v) { return v ? "ON" : "OFF"; }

typedef struct {
    mynah_dispatch_row *rows;
    int capacity;
    int count;
} row_sink;

static void row_env(mynah_dispatch_row *r, const char *env_name) {
    r->env_name = env_name;
    const char *e = env_name != NULL ? getenv(env_name) : NULL;
    if (e == NULL)      snprintf(r->env_value, sizeof r->env_value, "unset");
    else if (e[0] == 0) snprintf(r->env_value, sizeof r->env_value, "\"\"");
    else                snprintf(r->env_value, sizeof r->env_value, "%s", e);
}

/* Add one row.  A registered probe ALWAYS overrides the caller's resolved
 * value: that is the whole point of the registry, and it is the only path by
 * which a row can be promoted to [predicate]. */
static void add_row(row_sink *sink, const char *id, const char *compiled,
                    const char *supported, const char *env_name,
                    const char *resolved, mynah_dispatch_source source,
                    const char *reason) {
    if (sink->count >= sink->capacity) return;
    mynah_dispatch_row *r = &sink->rows[sink->count];
    memset(r, 0, sizeof *r);
    r->id = id;
    r->compiled = compiled;
    r->supported = supported;
    row_env(r, env_name);
    r->source = source;
    snprintf(r->resolved, sizeof r->resolved, "%s", resolved);
    snprintf(r->reason, sizeof r->reason, "%s", reason);

    const mynah_dispatch_value_probe_fn vfn = value_probe_for(id);
    if (vfn != NULL) {
        const char *why = NULL;
        char value[sizeof r->resolved];
        value[0] = 0;
        if (vfn(value, sizeof value, &why) == 0 && value[0] != 0) {
            snprintf(r->resolved, sizeof r->resolved, "%s", value);
            r->source = MYNAH_DISPATCH_SRC_PREDICATE;
        } else {
            snprintf(r->resolved, sizeof r->resolved, "UNKNOWN");
            r->source = MYNAH_DISPATCH_SRC_UNKNOWN;
        }
        if (why != NULL && why[0] != 0) snprintf(r->reason, sizeof r->reason, "%s", why);
        ++sink->count;
        return;
    }

    const mynah_dispatch_probe_fn fn = probe_for(id);
    if (fn != NULL) {
        const char *why = NULL;
        const int v = fn(&why);
        if (v < 0) {
            snprintf(r->resolved, sizeof r->resolved, "UNKNOWN");
            r->source = MYNAH_DISPATCH_SRC_UNKNOWN;
        } else {
            snprintf(r->resolved, sizeof r->resolved, "%s", onoff(v));
            r->source = MYNAH_DISPATCH_SRC_PREDICATE;
        }
        if (why != NULL && why[0] != 0) snprintf(r->reason, sizeof r->reason, "%s", why);
    }
    ++sink->count;
}

/* A feature nothing in src/ implements.  compiled is "no" by construction and
 * the reason must name the missing intrinsic, so the row reads as evidence. */
static void add_absent(row_sink *sink, const char *id, tri supported,
                       const char *reason) {
    add_row(sink, id, "no", yn3(supported), NULL, "OFF",
            MYNAH_DISPATCH_SRC_GATE, reason);
}

/* A pure compile gate with no runtime fallback. */
static void add_gate(row_sink *sink, const char *id, int gate, tri supported,
                     const char *reason) {
    add_row(sink, id, yn(gate), yn3(supported), NULL, onoff(gate),
            MYNAH_DISPATCH_SRC_GATE, reason);
}

/* Nobody exports the predicate that decides this.  Name the wrapper that would
 * make the row truthful: the footer turns these into a work list. */
static void add_unknown(row_sink *sink, const char *id, const char *compiled,
                        const char *supported, const char *env_name,
                        const char *reason) {
    add_row(sink, id, compiled, supported, env_name, "UNKNOWN",
            MYNAH_DISPATCH_SRC_UNKNOWN, reason);
}

/* ======================================================================
 * The table
 * ====================================================================== */

const char *mynah_dispatch_isa_class(void) {
#if defined(__aarch64__) || defined(__ARM_NEON)
    if (!MYNAH_DISPATCH_HAS_NEON) return "arm_scalar";
    if (MYNAH_DISPATCH_HAS_DOTPROD) return "arm_neon_dotprod";
    return "arm_neon";
#elif defined(__x86_64__) || defined(__i386__)
    /* Asked of the kernels, not of CFLAGS: SIMD=avx512 still says nothing about
     * which int8 kernel runs, and that gap is what produced the false claim.
     *
     * AND NOT OF THE INT8 KERNEL ALONE, which is what it used to ask. That was
     * a fair proxy while every other x86 kernel was a compile-time #if and the
     * int8 one was the only runtime choice. It stopped being fair the day the
     * f32 kernels and sgemm gained their own runtime dispatch: a sanitizer
     * build, whose CFLAGS replace the SIMD flags entirely, compiles no AVX2
     * int8 dot and still runs AVX2 f32 and an AVX2 sgemm through their target
     * attributes. On a CI runner with AVX2 and no VNNI that binary called
     * itself `x86_scalar` while two of its three kernel families were
     * vectorised.
     *
     * Found by tools/dispatch_gate.py on its first run in CI -- MISMATCH
     * isa.x86.avx2 expected OFF, got ON -- which is the kind of disagreement
     * nobody reads out of a 53-row table by eye, and is the reason that gate
     * exists. `x86_scalar` now means what a reader takes it to mean: nothing
     * here is vectorised. */
    {
        const char *k = mynah_qmat_int8_kernel(NULL);
        if (strcmp(k, "avx512vnni") == 0) return "x86_avx512_vnni";
        if (strcmp(k, "avxvnni") == 0) return "x86_avx_vnni";
        if (mynah_kernels_x86_avx2() || strcmp(k, "avx512bw") == 0 ||
            strcmp(k, "avx2") == 0 || MYNAH_DISPATCH_HAS_AVX2) {
            return "x86_avx2";
        }
    }
    return "x86_scalar";
#else
    return "portable_scalar";
#endif
}

static const char *os_name(void) {
#if defined(__APPLE__)
    return "Darwin";
#elif defined(__linux__)
    return "Linux";
#else
    return "unknown";
#endif
}

static const char *arch_name(void) {
#if defined(__aarch64__)
    return "arm64";
#elif defined(__x86_64__)
    return "x86_64";
#elif defined(__i386__)
    return "i386";
#elif defined(__ARM_NEON)
    return "arm";
#else
    return "unknown";
#endif
}

/* Observe src/qmat.c's own resolution of a quantization type.  Returns 1 when
 * the module accepts it, 0 when it silently falls back to f32, -1 on OOM. */
static int qmat_probe(int qtype) {
    mynah_qmat_cache *c = mynah_qmat_cache_new(qtype);
    if (c == NULL) return -1;
    const int on = mynah_qmat_cache_enabled(c);
    mynah_qmat_cache_free(c);
    return on;
}

static void collect_isa(row_sink *s) {
    add_gate(s, "isa.arm.neon", MYNAH_DISPATCH_HAS_NEON, cpu_has_neon(),
             "[gate] src/kernels.c NEON dot/matvec/rmsnorm/layernorm/gelu/axpy; "
             "compile-time selection, there is no runtime fallback path");
    add_gate(s, "isa.arm.dotprod", MYNAH_DISPATCH_HAS_DOTPROD, cpu_has_dotprod(),
             "[gate] src/qmat.c int8 dot via vdotq_s32 (SDOT); without it the "
             "int8 matvec runs the scalar int32 accumulation");
    add_unknown(s, "isa.arm.i8mm", yn(MYNAH_DISPATCH_HAS_I8MM_KERNEL),
                yn3(cpu_has_i8mm()), "MYNAH_QMAT_I8MM",
                "[UNKNOWN] src/qmat.c did not register "
                "mynah_qmat_i8mm_enabled() -- the SMMLA row cannot be resolved "
                "from the compile gate, because the kernel is compiled "
                "unconditionally and chosen by a runtime CPU probe");

    /* SVE and BF16: the rows that were missing, and the reason they are here.
     *
     * On the project's production box -- GCP Axion, Neoverse V2 -- the kernel
     * advertises `sve sve2 svei8mm svebf16 i8mm bf16` and this report printed
     * ONE of those six.  There was no isa.arm.sve row at all, so a reader of a
     * clean report had no way to learn that four vector units on the target CPU
     * are idle; and isa.arm.bf16 said compiled=no with a reason that never
     * mentioned that the hardware in front of it has the unit.  A report whose
     * silence has to be interpreted is not doing the job this file exists for.
     *
     * `resolved` for all five comes from src/kernels.c's exported inventory,
     * never from a #if here -- add_row promotes them to [predicate] and the
     * reason each one prints is written next to where the kernel would live.
     * The supported column is the hardware half, and `compiled=no` beside
     * `supported=yes` is the finding; the footer counts those pairs. */
    add_absent(s, "isa.arm.sve", cpu_has_sve(),
               "[gate] NOT IMPLEMENTED (no predicate registered by "
               "src/kernels.c)");
    add_absent(s, "isa.arm.sve2", cpu_has_sve2(),
               "[gate] NOT IMPLEMENTED (no predicate registered by "
               "src/kernels.c)");
    add_absent(s, "isa.arm.svei8mm", cpu_has_svei8mm(),
               "[gate] NOT IMPLEMENTED (no predicate registered by "
               "src/kernels.c)");
    add_absent(s, "isa.arm.svebf16", cpu_has_svebf16(),
               "[gate] NOT IMPLEMENTED (no predicate registered by "
               "src/kernels.c)");
    /* NOT add_absent: src/qmat.c compiles a bf16 kernel on BOTH architectures,
     * so `compiled no` here was false on x86 from the day E14-3 landed, and it
     * was false in the direction that hides work -- the row said no kernel
     * while VDPBF16PS was executing two rows below it.  And the env column said
     * `-` while MYNAH_QMAT_BF16 turned the row OFF, which is the dispatch map's
     * one promise: the env column names the flag that moves the row. */
    add_unknown(s, "isa.arm.bf16", yn(MYNAH_DISPATCH_HAS_BF16_KERNEL),
                yn3(cpu_has_bf16_path()), "MYNAH_QMAT_BF16",
                "[UNKNOWN] src/qmat.c did not register the bf16 predicate over "
                "this id. The id names ARM and the kernel exists on x86 too -- "
                "isa.x86.avx512bf16 carries the x86 tier");
    /* NOT add_gate any more, and the change is the finding it used to hide.
     * This row read `compiled no` on an x86 portable build and stopped
     * there -- true, and it let a reader conclude the whole binary was
     * scalar when src/qmat.c was still dispatching to VNNI/F16C at
     * runtime. Since the f32 kernels grew their own runtime dispatch the
     * compile flag decides nothing at all here, so the row is answered by
     * src/kernels.c's predicate and carries the env that overrides it. */
    add_row(s, "isa.x86.avx2", yn(MYNAH_DISPATCH_HAS_AVX2),
            yn3(cpu_has_avx2()), "MYNAH_KERNELS_X86", onoff(MYNAH_DISPATCH_HAS_AVX2),
            MYNAH_DISPATCH_SRC_GATE,
            "[gate] src/kernels.c f32 kernels (RUNTIME-selected since the x86 "
            "dispatch landed -- this row is normally answered by that "
            "predicate) and src/qmat.c dot_q8_i32_avx2 "
            "(_mm256_cvtepi8_epi16 + _mm256_madd_epi16), the int8 dot for "
            "every x86 CPU without VPDPBUSD");
    /* Answered by the same predicate as isa.x86.avx2, and for the same reason
     * that row stopped being a gate: the f32 kernels carry target("avx2,fma")
     * and emit _mm256_fmadd_ps whether or not the BUILD was given -mfma. A
     * sanitizer build replaces CFLAGS entirely, so it compiles no -mfma and
     * runs FMA anyway -- and this row said OFF, and the footer counted the
     * host's FMA unit as idle hardware. Caught by tools/dispatch_gate.py in
     * CI, one commit after the identical mistake on isa.x86.avx2. */
    add_unknown(s, "isa.x86.fma", yn(MYNAH_DISPATCH_HAS_FMA), yn3(cpu_has_fma()),
                "MYNAH_KERNELS_X86",
                "[UNKNOWN] src/kernels.c did not register the f32 predicate "
                "over this id");

    /* The three rows the false README claim needed.  compiled can be yes here
     * (SIMD=avx512 passes the flags) while resolved is OFF, because no kernel
     * dispatches on it.  That gap is the finding, so it is stated twice. */
    /* These three used to say "COMPILER FLAG ONLY ... no f32 kernel dispatches
     * on it. The one _mm512_* kernel in src/ is the VNNI int8 dot." That was
     * true until E14-2 and E14-3 and is not any more: there are three now --
     * the VNNI int8 dot, the AVX-512BW int8 dot for hosts WITHOUT VNNI, and
     * VDPBF16PS. Two of them need F/BW/VL, so this trio is reached by a kernel
     * and is answered by src/qmat.c's predicate rather than by CFLAGS.
     *
     * `compiled` still reports the build flag, because that is what it means
     * everywhere else in this table, and it is still not what decides: all
     * three kernels carry their own target attribute and need no flag. */
    /* `compiled` is the KERNEL, not the build flag -- the same correction these
     * rows' text already carries. The footer counts `compiled=no,
     * supported=yes` as idle hardware, so reading CFLAGS here told an operator
     * on a VNNI host that their AVX-512 had no kernel, when it has one that
     * merely lost the job to a wider one. Second time this exact confusion has
     * shipped; `compiled` means "it is in this binary" everywhere now. */
    add_unknown(s, "isa.x86.avx512f", yn(MYNAH_DISPATCH_HAS_AVX512BW_KERNEL),
                yn3(cpu_has_avx512f()), "MYNAH_QMAT_AVX512",
                "[UNKNOWN] src/qmat.c did not register the AVX-512BW int8 "
                "predicate over this id");
    add_unknown(s, "isa.x86.avx512bw", yn(MYNAH_DISPATCH_HAS_AVX512BW_KERNEL),
                yn3(cpu_has_avx512bw()), "MYNAH_QMAT_AVX512",
                "[UNKNOWN] src/qmat.c did not register the AVX-512BW int8 "
                "predicate over this id");
    add_unknown(s, "isa.x86.avx512vl", yn(MYNAH_DISPATCH_HAS_AVX512BW_KERNEL),
                yn3(cpu_has_avx512vl()), NULL,
                "[UNKNOWN] src/qmat.c did not register the prerequisite "
                "predicate over this id");
    add_unknown(s, "isa.x86.avx512bf16", yn(MYNAH_DISPATCH_HAS_AVX512BF16_KERNEL),
                yn3(cpu_has_avx512bf16()), "MYNAH_QMAT_BF16DOT",
                "[UNKNOWN] src/qmat.c did not register the VDPBF16PS "
                "predicate. bf16 is the dtype this backbone ships, so a "
                "supported=yes with resolved=OFF here is a real gap, not a "
                "curiosity");
    add_unknown(s, "isa.x86.avx512vnni", yn(MYNAH_DISPATCH_HAS_AVX512VNNI_KERNEL),
                yn3(cpu_has_avx512vnni()), "MYNAH_QMAT_VNNI",
                "[UNKNOWN] src/qmat.c did not register the VPDPBUSD predicate. "
                "Historical note: the EPYC Zen 5 int8 RTF 0.427 predates this "
                "kernel and was produced by the AVX2 int8 dot, not by VNNI");
    add_unknown(s, "isa.x86.avxvnni", yn(MYNAH_DISPATCH_HAS_AVXVNNI_KERNEL),
                yn3(cpu_has_avxvnni()), "MYNAH_QMAT_VNNI",
                "[UNKNOWN] src/qmat.c did not register the VEX VPDPBUSD "
                "predicate");
    add_absent(s, "isa.x86.amx_int8", cpu_has_amx_int8(),
               "[gate] NOT IMPLEMENTED: no tile configuration or _tile_* op in "
               "src/; E4 lists AMX last, as the narrowest-platform item");
}

/* E4-16.  THERE ARE NO blas.* ROWS ANY MORE, and their absence is the finding.
 *
 * There used to be five: blas.accelerate, blas.openblas, blas.scalar_fallback,
 * blas.threads_owned and blas.thread_timeout.  The first three asked "which
 * library multiplies two f32 matrices", which is exactly what sgemm.provider
 * answers -- and answers better, because it names the provider that actually
 * ran instead of listing the ones that were compiled.  The last two were not
 * about GEMM at all: they reported whether we had managed to wrestle control
 * of a foreign thread pool, and one of them had to be declared UNKNOWN
 * because the honest answer was "we cannot tell from here".  With BLAS=none
 * the default on Linux there is no foreign pool in the process, so there is
 * nothing to report and no row that can go stale.
 *
 * Accelerate has NOT left this report: it is still the macOS default, and the
 * parts of it that are not a GEMM keep their own rows -- kernel.gelu_vector
 * for the vForce tanh, codec.sgemm_conv for the BNNS convolution, and
 * codec.seanet_gemm for the decoder fast path. */
static void collect_sgemm(row_sink *s) {
    /* E4-16, .work/no-blas.md.  One BLAS function was ever used --
     * cblas_sgemm, three call sites -- and src/sgemm.c replaces it.  These
     * four rows are declared UNKNOWN and then overridden by the predicates
     * src/backend.c and src/sgemm.c register, so if either file stops
     * registering they read UNKNOWN again instead of reverting to a
     * compile-time guess.
     *
     * `compiled` is "yes" unconditionally and that is not sloppiness:
     * src/sgemm.c is in CORE_SOURCES for every build, including the
     * Accelerate and OpenBLAS comparison builds where it is self-tested but
     * not wired in.  sgemm.provider is the row that says which one runs. */
    add_unknown(s, "sgemm.provider", "yes", "-", NULL,
                "[UNKNOWN] src/backend.c did not register "
                "backend_sgemm_provider(). This is the row that says whether "
                "an external BLAS is in this process at all");
    add_unknown(s, "sgemm.kernel", "yes", "-", NULL,
                "[UNKNOWN] src/sgemm.c did not register "
                "mynah_sgemm_isa_name()");
    add_unknown(s, "sgemm.family", "yes", "-", NULL,
                "[UNKNOWN] src/sgemm.c did not register its family counters");
    add_unknown(s, "sgemm.selftest", "yes", "-", NULL,
                "[UNKNOWN] src/sgemm.c did not register "
                "mynah_sgemm_self_test()");
}

static void collect_pool(row_sink *s) {
    char buf[24];
    const int threads = mynah_num_threads();     /* the real predicate */
    snprintf(buf, sizeof buf, "%d", threads);
    add_row(s, "pool.threads", "yes", "-", "MYNAH_THREADS", buf,
            MYNAH_DISPATCH_SRC_RUNTIME,
            "[runtime] mynah_num_threads(); default is hw.perflevel0.logicalcpu "
            "on Apple Silicon (decode is DRAM-bound, E-cores lengthen the "
            "barrier) and the online CPU count elsewhere");

#if defined(__APPLE__)
    {
        char cores[64];
        const long perf = sysctl_long("hw.perflevel0.logicalcpu");
        const long all  = sysctl_long("hw.logicalcpu");
        snprintf(cores, sizeof cores, "%ldP/%ld", perf, all);
        add_row(s, "pool.cpu_topology", "-", "-", NULL, cores,
                MYNAH_DISPATCH_SRC_RUNTIME,
                "[runtime] sysctl hw.perflevel0.logicalcpu / hw.logicalcpu: "
                "performance cores over logical cores");
    }
#endif

    /* This row used to describe the pool as "one dispatch at a time; a nested
     * dispatch runs inline serial".  That stopped being true when the pool
     * grew concurrent submit and safe nesting, and a stale reason is worse
     * than no row: it reads as a measurement.  So the three capabilities are
     * now ASKED of the pool -- mynah_pool_concurrent_submit_ok() /
     * _nested_dispatch_ok() / _priority_ok() are the pool's own answers, not a
     * sentence in this file that has to be remembered. */
    {
        static char text[240];
        const int concurrent = mynah_pool_concurrent_submit_ok();
        const int nested = mynah_pool_nested_dispatch_ok();
        const int priority = mynah_pool_priority_ok();
        snprintf(text, sizeof text,
                 "[runtime] mynah_num_threads() + the pool's own capability "
                 "predicates: %d job slot%s, concurrent submit %s, nested "
                 "dispatch %s, deadline priority %s",
                 mynah_pool_max_jobs(), mynah_pool_max_jobs() == 1 ? "" : "s",
                 concurrent ? "OK" : "NO", nested ? "OK" : "NO",
                 priority ? "OK" : "no-op");
        add_row(s, "pool.parallel_for", "yes", "-", NULL, onoff(threads > 1),
                MYNAH_DISPATCH_SRC_RUNTIME, text);
    }

    /* E5-22. A VALUE row, not a boolean: 65536 and 4096 are different servers
     * (their sweep moved STREAM p95 0.893 -> 0.808 and context switches 38k ->
     * 7.6k/s), and a row saying ON would hide which one is running. The
     * default is TRANSFERRED, never measured here, and src/threads.c's probe
     * says so in the reason. */
    add_unknown(s, "pool.spin", "yes", "-", "MYNAH_POOL_SPIN",
                "[UNKNOWN] src/threads.c did not register "
                "mynah_pool_spin_budget()");

    /* E5-21. OFF, or the split actually in force. A lane that refused to
     * engage -- too narrow, or a platform that cannot pin -- reads OFF here
     * and the reason says which, because "I asked for a lane" and "I have a
     * lane" are different claims and the reference lost a campaign to exactly
     * that distinction. */
    add_unknown(s, "pool.decoder_lane", "yes", "-", "MYNAH_LANE_SPLIT",
                "[UNKNOWN] src/threads.c did not register mynah_lane_width()");

    /* E9-P1. Whether anybody is counting, and what the count says if they are.
     * The serving sweep that put 16x2 at STREAM p95 0.736 and 1x32 at "did not
     * complete" was explained by a barrier nobody had measured; this row is
     * where that stops being an explanation and starts being a number. */
    add_unknown(s, "pool.meter", "yes", "-", "MYNAH_POOL_METER",
                "[UNKNOWN] src/threads.c did not register "
                "mynah_pool_meter_read()");

    /* E9-P2. The narrowing levers change WHO runs a chunk. This row is the
     * evidence that they did not change WHAT is computed -- a short form of the
     * litmus, run inside the report itself, because a knob reported as ON that
     * has never been tested in this binary is a claim and not a fact. */
    add_unknown(s, "pool.litmus", "yes", "-", NULL,
                "[UNKNOWN] src/threads.c did not register "
                "mynah_threads_self_test()");

}

static void collect_quant(row_sink *s) {
    const int requested = qmat_probe(-1);        /* reads MYNAH_QUANT itself */
    const int int8_ok   = qmat_probe(QMAT_PROBE_INT8);
    const int int4_ok   = qmat_probe(QMAT_PROBE_INT4);
    const int f16_ok    = qmat_probe(QMAT_PROBE_F16);

    add_row(s, "quant.requested", "yes", "-", "MYNAH_QUANT",
            requested > 0 ? "ON" : (requested == 0 ? "OFF" : "UNKNOWN"),
            requested < 0 ? MYNAH_DISPATCH_SRC_UNKNOWN : MYNAH_DISPATCH_SRC_RUNTIME,
            "[runtime] mynah_qmat_cache_new(-1) + mynah_qmat_cache_enabled(); "
            "src/qmat.c's value probe replaces this with the resolved type");

    add_row(s, "quant.int8", yn(int8_ok > 0), "-", NULL,
            onoff(int8_ok > 0), MYNAH_DISPATCH_SRC_RUNTIME,
            "[runtime] qmat cache accepts int8: per-row symmetric absmax, "
            "activations quantized per row; count<=16 takes the integer dot, "
            "a larger count falls back to the bit-exact f32 BLAS prefill");
    add_row(s, "quant.int4", yn(int4_ok > 0), "-", NULL,
            onoff(int4_ok > 0), MYNAH_DISPATCH_SRC_RUNTIME,
            "[runtime] qmat cache accepts int4: Q4_0-style, group 32, per-group "
            "scale; a shape it cannot represent stays f32");
    add_row(s, "quant.f16", yn(f16_ok > 0), yn3(cpu_has_fp16()), NULL,
            onoff(f16_ok > 0), MYNAH_DISPATCH_SRC_RUNTIME,
            f16_ok > 0
              ? "[runtime] qmat cache accepts f16 (__fp16 weights, NEON half "
                "converts). Also the DRIFT canary for the mirrored gates"
              : "[runtime] qmat cache SILENTLY DOWNGRADES f16 to f32 on this "
                "build: MYNAH_QUANT=f16 would run exact f32 and say nothing");

    add_unknown(s, "quant.row4", "yes", "-", "MYNAH_QMAT_SINGLE_ROW",
                "[UNKNOWN] src/qmat.c did not register mynah_qmat_cache_row4()");
    add_unknown(s, "quant.argmax_mt", "yes", "-", "MYNAH_ARGMAX_MT",
                "[UNKNOWN] src/qmat.c did not register "
                "mynah_qmat_argmax_mt_resolved()");

    /* The row the AVX-512 claim needed and did not have: not "is VNNI
     * compiled" but "which int8 kernel is this process about to run". */
    add_unknown(s, "quant.int8_kernel", "yes", "-", "MYNAH_QMAT_VNNI",
                "[UNKNOWN] src/qmat.c did not register "
                "mynah_qmat_int8_kernel()");
    /* WHICH weights the requested encoding is allowed to touch.  The engine
     * owns the group names; src/qmat.c owns the single reading of the
     * variable and answers this row, so the report and the engine cannot
     * disagree about what was asked for. */
    add_unknown(s, "quant.groups", "yes", "-", "MYNAH_QUANT_GROUPS",
                "[UNKNOWN] src/qmat.c did not register "
                "mynah_qmat_groups_spec()");
}

static void collect_backends(row_sink *s) {
    static char cpu_reason[240];
    static char metal_reason[240];
    static char cuda_reason[240];
    char err[192];
    mynah_backend *b = NULL;

    err[0] = 0;
    if (mynah_backend_open(MYNAH_TTS_DEVICE_CPU, &b, err, sizeof err) == 0) {
        snprintf(cpu_reason, sizeof cpu_reason,
                 "[runtime] mynah_backend_open(CPU) succeeded, name=\"%s\"; cpu "
                 "is always available by contract", mynah_backend_name(b));
        mynah_backend_close(b);
        add_row(s, "backend.cpu", "yes", "yes", NULL, "ON",
                MYNAH_DISPATCH_SRC_RUNTIME, cpu_reason);
    } else {
        snprintf(cpu_reason, sizeof cpu_reason,
                 "[runtime] mynah_backend_open(CPU) FAILED: %s", err);
        add_row(s, "backend.cpu", "yes", "no", NULL, "OFF",
                MYNAH_DISPATCH_SRC_RUNTIME, cpu_reason);
    }

    b = NULL; err[0] = 0;
    if (mynah_backend_open(MYNAH_TTS_DEVICE_METAL, &b, err, sizeof err) == 0) {
        snprintf(metal_reason, sizeof metal_reason,
                 "[runtime] mynah_backend_open(METAL) succeeded, name=\"%s\"",
                 mynah_backend_name(b));
        mynah_backend_close(b);
        add_row(s, "backend.metal", yn(MYNAH_DISPATCH_HAS_METAL), "yes", NULL,
                "ON", MYNAH_DISPATCH_SRC_RUNTIME, metal_reason);
    } else {
        snprintf(metal_reason, sizeof metal_reason,
                 "[runtime] mynah_backend_open(METAL) declined: %s", err);
        add_row(s, "backend.metal", yn(MYNAH_DISPATCH_HAS_METAL), "-", NULL,
                "OFF", MYNAH_DISPATCH_SRC_RUNTIME, metal_reason);
    }

    b = NULL; err[0] = 0;
    if (mynah_backend_open(MYNAH_TTS_DEVICE_CUDA, &b, err, sizeof err) == 0) {
        snprintf(cuda_reason, sizeof cuda_reason,
                 "[runtime] mynah_backend_open(CUDA) succeeded, name=\"%s\"",
                 mynah_backend_name(b));
        mynah_backend_close(b);
        add_row(s, "backend.cuda", yn(MYNAH_DISPATCH_HAS_CUDA), "yes", NULL,
                "ON", MYNAH_DISPATCH_SRC_RUNTIME, cuda_reason);
    } else {
        snprintf(cuda_reason, sizeof cuda_reason,
                 "[runtime] mynah_backend_open(CUDA) declined: %s", err);
        add_row(s, "backend.cuda", yn(MYNAH_DISPATCH_HAS_CUDA), "-", NULL,
                "OFF", MYNAH_DISPATCH_SRC_RUNTIME, cuda_reason);
    }

    add_unknown(s, "cpu.matvec_policy", "yes", "-", "MYNAH_CPU_MATVEC",
                "[UNKNOWN] src/backend.c did not register "
                "mynah_cpu_matvec_mode(). This is the row that hides a rows=1 "
                "regression");
    add_unknown(s, "kernel.gelu_vector", "yes", "-", "MYNAH_GELU_SCALAR",
                "[UNKNOWN] src/kernels.c did not register "
                "mynah_gelu_vector_enabled()");
    add_unknown(s, "kernel.fused_greedy", "yes", "-", "MYNAH_FUSED_GREEDY",
                "[UNKNOWN] src/qmat.c did not register "
                "mynah_qmat_fused_greedy_enabled()");
    add_unknown(s, "codec.sgemm_conv", "yes", "-", "MYNAH_CODEC_SGEMM",
                "[UNKNOWN] src/conv1d.c did not register "
                "mynah_conv1d_sgemm_enabled()");

    /* E4-21: the PocketTTS codec conv stack.  src/seanet.c had twelve fallback
     * paths and no row here at all -- `grep seanet src/dispatch.c` returned
     * nothing -- while the build gate behind them decides a 36x difference
     * (8570 ms of scalar loops against 237 ms of sgemm, .work/no-blas.md §2).
     * These four are declared as UNKNOWN and then overridden by the predicates
     * src/seanet.c registers, so if that file ever stops registering they read
     * UNKNOWN again instead of silently reverting to a compile-time guess. */
    add_unknown(s, "codec.seanet_blas", "yes", "-", NULL,
                "[UNKNOWN] src/seanet.c did not register "
                "mynah_seanet_blas_name()");
    /* The GEMM fold needs an sgemm, not specifically an external one: under
     * BLAS=none src/sgemm.c provides it and the fold is compiled.  Reading
     * only the two vendor macros made a BLAS=none build report compiled=no on
     * a row that then resolved ON -- seen on Linux ARM, 2026-09-13. */
    add_unknown(s, "codec.seanet_gemm", yn(MYNAH_DISPATCH_HAS_ACCELERATE ||
                                           MYNAH_DISPATCH_HAS_OPENBLAS ||
                                           MYNAH_DISPATCH_HAS_OWN_SGEMM),
                "-", "MYNAH_SEANET_GEMM",
                "[UNKNOWN] src/seanet.c did not register "
                "mynah_seanet_gemm_enabled()");
    add_unknown(s, "codec.seanet_conv_path", "yes", "-", NULL,
                "[UNKNOWN] src/seanet.c did not register its conv1d path "
                "counters");
    add_unknown(s, "codec.seanet_convtr_path", "yes", "-", NULL,
                "[UNKNOWN] src/seanet.c did not register its convtranspose "
                "path counters");
    add_unknown(s, "codec.conv_int8_host", "yes", "-", "MYNAH_CODEC_CONV_Q8",
                "[UNKNOWN] src/convq8.c did not register "
                "mynah_convq8_host_ok()");
    add_unknown(s, "codec.conv_int8_path", "yes", "-", NULL,
                "[UNKNOWN] src/convq8.c did not register its tap GEMM "
                "counters");
    add_unknown(s, "codec.snake_vector", "yes", "-", "MYNAH_SNAKE_SCALAR",
                "[UNKNOWN] src/codec_nanocodec.c did not register "
                "mynah_snake_vector_enabled()");
}

static void collect_selftests(row_sink *s) {
    static char kern[240], gelu[240], qmat[240];
    char err[160];

    err[0] = 0;
    const int k = mynah_kernels_self_test(err, sizeof err);
    snprintf(kern, sizeof kern,
             "[runtime] mynah_kernels_self_test(): %s. Proves the COMPILED "
             "dot/matvec/norm path is numerically correct; it cannot name the "
             "ISA, which is what the gate rows above are for",
             k == 0 ? "PASS" : err);
    add_row(s, "selftest.kernels", "yes", "-", NULL, k == 0 ? "PASS" : "FAIL",
            MYNAH_DISPATCH_SRC_RUNTIME, kern);

    err[0] = 0;
    const int g = mynah_gelu_self_test(err, sizeof err);
    snprintf(gelu, sizeof gelu, "[runtime] mynah_gelu_self_test(): %s",
             g == 0 ? "PASS" : err);
    add_row(s, "selftest.gelu", "yes", "-", NULL, g == 0 ? "PASS" : "FAIL",
            MYNAH_DISPATCH_SRC_RUNTIME, gelu);

    err[0] = 0;
    const int q = mynah_qmat_self_test(err, sizeof err);
    snprintf(qmat, sizeof qmat,
             "[runtime] mynah_qmat_self_test(): %s. Exercises int8/int4%s "
             "against an exact f32 dot with a bounded relative error",
             q == 0 ? "PASS" : err,
             MYNAH_DISPATCH_HAS_QMAT_F16 ? "/f16" : "");
    add_row(s, "selftest.qmat", "yes", "-", NULL, q == 0 ? "PASS" : "FAIL",
            MYNAH_DISPATCH_SRC_RUNTIME, qmat);
}


/* Collected LAST, after the self-tests, because these are process totals: run
 * from collect_pool() they would all read 0 and say nothing.  Placed here they
 * at least cover the pool traffic this report itself generated (the qmat
 * self-test dispatches row blocks), and in a long-lived server process they
 * cover everything since start. */
static void collect_pool_stats(row_sink *s) {
    {
        /* 240 was too small: the "dispatched nothing" arm alone is 321 bytes
         * worst case, and gcc's -Wformat-truncation said so on every Linux
         * build. A truncated row cuts the sentence that tells the reader the
         * report measured nothing -- the one thing it must not lose. */
        static char text[384];
        char value[24];
        long long dispatches = 0, serial = 0, inline_fallbacks = 0, joins = 0;
        mynah_parallel_stats(&dispatches, &serial, &inline_fallbacks, &joins);
        /* "0" with no dispatches behind it is not a clean bill of health, so
         * the value says which of the two it is. */
        if (dispatches == 0) snprintf(value, sizeof value, "n/a");
        else snprintf(value, sizeof value, "%lld", inline_fallbacks);
        snprintf(text, sizeof text,
                 "[runtime] mynah_parallel_stats(): %lld dispatch%s, %lld "
                 "serial, %lld inline fallback%s (all job slots were busy -- "
                 "must stay 0), %lld helper join%s. %s",
                 dispatches, dispatches == 1 ? "" : "es", serial,
                 inline_fallbacks, inline_fallbacks == 1 ? "" : "s",
                 joins, joins == 1 ? "" : "s",
                 dispatches == 0
                   ? "n/a: this report dispatched nothing, so it has measured "
                     "nothing. Read it from a synthesis or a server process"
                   : "Process totals since start");
        add_row(s, "pool.inline_fallbacks", "yes", "-", NULL, value,
                MYNAH_DISPATCH_SRC_RUNTIME, text);
    }
}

int mynah_dispatch_collect(mynah_dispatch_row *rows, int capacity) {
    if (rows == NULL || capacity <= 0) return -1;
    row_sink sink = { rows, capacity, 0 };
    register_module_probes();
    collect_isa(&sink);
    collect_sgemm(&sink);
    collect_pool(&sink);
    collect_quant(&sink);
    collect_backends(&sink);
    collect_selftests(&sink);
    collect_pool_stats(&sink);
    return sink.count;
}

/* ======================================================================
 * The drift canary
 *
 * The [gate] rows mirror gates that live in other translation units.  This
 * checks the one mirrored gate whose truth is observable at runtime.  If the
 * assumption "one CFLAGS for every file in src/" ever breaks, this is what
 * says so, instead of the report quietly reporting a build that does not exist.
 * ====================================================================== */
static int drift_f16(const char **detail) {
    const int observed = qmat_probe(QMAT_PROBE_F16);
    if (observed < 0) { *detail = "qmat probe could not allocate"; return 0; }
    if ((observed != 0) == (MYNAH_DISPATCH_HAS_QMAT_F16 != 0)) {
        *detail = "MYNAH_DISPATCH_HAS_QMAT_F16 agrees with src/qmat.c";
        return 0;
    }
    *detail = "MYNAH_DISPATCH_HAS_QMAT_F16 DISAGREES with src/qmat.c: this "
              "file and qmat.c were compiled with different flags, so every "
              "[gate] row is suspect";
    return 1;
}

/* ======================================================================
 * Output
 * ====================================================================== */

static void json_str(FILE *j, const char *s) {
    fputc('"', j);
    for (; s != NULL && *s; ++s) {
        const unsigned char c = (unsigned char)*s;
        if (c == '"' || c == '\\')      { fputc('\\', j); fputc((int)c, j); }
        else if (c == '\n')             { fputs("\\n", j); }
        else if (c < 0x20)              { fprintf(j, "\\u%04x", c); }
        else                            { fputc((int)c, j); }
    }
    fputc('"', j);
}

static const char *source_name(mynah_dispatch_source s) {
    switch (s) {
    case MYNAH_DISPATCH_SRC_PREDICATE: return "predicate";
    case MYNAH_DISPATCH_SRC_RUNTIME:   return "runtime";
    case MYNAH_DISPATCH_SRC_GATE:      return "gate";
    default:                           return "unknown";
    }
}

static int write_report(FILE *f, int as_json, mynah_dispatch_row *rows, int n) {
    const char *drift_detail = "";
    const int drift = drift_f16(&drift_detail);
    int unknown = 0, predicate = 0, runtime = 0, gate = 0;
    /* Units this CPU advertises and this binary contains no kernel for.  It is
     * derived from the rows rather than listed by hand -- supported=yes with
     * compiled=no is exactly that pair -- so a feature can never be idle and
     * unmentioned, which is how four SVE units on the production box went
     * unreported for as long as there was no isa.arm.sve row to be silent in. */
    int idle = 0;
    char idle_list[256];
    size_t idle_used = 0;
    idle_list[0] = '\0';
    for (int i = 0; i < n; ++i) {
        switch (rows[i].source) {
        case MYNAH_DISPATCH_SRC_PREDICATE: ++predicate; break;
        case MYNAH_DISPATCH_SRC_RUNTIME:   ++runtime;   break;
        case MYNAH_DISPATCH_SRC_GATE:      ++gate;      break;
        default:                           ++unknown;   break;
        }
        if (strcmp(rows[i].supported, "yes") != 0) continue;
        if (strcmp(rows[i].compiled, "no") != 0) continue;
        /* A row that RESOLVED ON is not idle hardware, whatever the `compiled`
         * column inherited. That column is filled by whoever seeded the row;
         * `resolved` is filled by the module that owns the kernel, through its
         * registered predicate, and it wins. Without this test the summary
         * contradicted the row directly beneath it -- isa.arm.bf16 printed
         * `resolved ON ... BFDOT` and was then listed as a unit with no kernel,
         * which is exactly the kind of silence-to-be-interpreted this report
         * exists to remove. */
        if (strcmp(rows[i].resolved, "ON") == 0) continue;
        ++idle;
        if (idle_used + 1 < sizeof idle_list) {
            const int k = snprintf(idle_list + idle_used,
                                   sizeof idle_list - idle_used, "%s%s",
                                   idle_used > 0 ? " " : "", rows[i].id);
            if (k > 0) idle_used += (size_t)k;
        }
    }

    if (!as_json) {
        fprintf(f, "[DISPATCH] v=1 pid=%d os=%s arch=%s isa_class=%s simd=%s "
                   "build=%s rows=%d\n",
                (int)getpid(), os_name(), arch_name(), mynah_dispatch_isa_class(),
                MYNAH_SIMD_PROFILE, MYNAH_GIT_REV, n);
        fprintf(f, "  %-22s %-8s %-9s %-28s %-8s %s\n",
                "feature", "compiled", "supported", "env", "resolved", "reason");
        for (int i = 0; i < n; ++i) {
            char envc[96];
            if (rows[i].env_name != NULL) {
                snprintf(envc, sizeof envc, "%s=%s", rows[i].env_name,
                         rows[i].env_value);
            } else {
                snprintf(envc, sizeof envc, "-");
            }
            fprintf(f, "  %-22s %-8s %-9s %-28s %-8s %s\n",
                    rows[i].id, rows[i].compiled, rows[i].supported, envc,
                    rows[i].resolved, rows[i].reason);
        }
        fprintf(f, "  resolved from: %d predicate, %d runtime, %d gate, "
                   "%d UNKNOWN\n", predicate, runtime, gate, unknown);
        if (unknown > 0) {
            fprintf(f, "  NOTE: %d row%s resolve to UNKNOWN because the owning "
                       "module exports no predicate. Each reason names the "
                       "wrapper to add; UNKNOWN is never replaced by a guess.\n",
                    unknown, unknown == 1 ? "" : "s");
        }
        if (idle > 0) {
            fprintf(f, "  IDLE HARDWARE: %d unit%s this CPU has and this "
                       "binary has no kernel for: %s. Each row's reason names "
                       "what would have to be written and where.\n",
                    idle, idle == 1 ? "" : "s", idle_list);
        }
        {
            char guard[512];
            if (mynah_dispatch_isa_guard(guard, sizeof guard) != 0) {
                fprintf(f, "  ISA GUARD: FATAL -- %s\n", guard);
            } else {
                fprintf(f, "  ISA GUARD: ok -- every instruction set this "
                           "build may emit is present on this CPU\n");
            }
        }
        fprintf(f, "  gate canary: %s -- %s\n", drift ? "DRIFT" : "ok",
                drift_detail);
        fflush(f);
        return unknown;
    }

    fprintf(f, "{\n  \"v\": 1,\n  \"pid\": %d,\n", (int)getpid());
    fprintf(f, "  \"os\": ");        json_str(f, os_name());
    fprintf(f, ",\n  \"arch\": ");   json_str(f, arch_name());
    fprintf(f, ",\n  \"isa_class\": "); json_str(f, mynah_dispatch_isa_class());
    fprintf(f, ",\n  \"simd_profile\": "); json_str(f, MYNAH_SIMD_PROFILE);
    fprintf(f, ",\n  \"build\": ");  json_str(f, MYNAH_GIT_REV);
    fprintf(f, ",\n  \"threads\": %d", mynah_num_threads());
    fprintf(f, ",\n  \"gate_drift\": %s", drift ? "true" : "false");
    fprintf(f, ",\n  \"gate_drift_detail\": "); json_str(f, drift_detail);
    fprintf(f, ",\n  \"counts\": {\"predicate\": %d, \"runtime\": %d, "
               "\"gate\": %d, \"unknown\": %d}", predicate, runtime, gate, unknown);
    fprintf(f, ",\n  \"idle_hardware_count\": %d", idle);
    fprintf(f, ",\n  \"idle_hardware\": "); json_str(f, idle_list);
    {
        char guard[512];
        const int bad = mynah_dispatch_isa_guard(guard, sizeof guard) != 0;
        fprintf(f, ",\n  \"isa_guard_ok\": %s", bad ? "false" : "true");
        fprintf(f, ",\n  \"isa_guard_detail\": "); json_str(f, bad ? guard : "");
    }
    fprintf(f, ",\n  \"features\": [\n");
    for (int i = 0; i < n; ++i) {
        fprintf(f, "    {\"id\": ");        json_str(f, rows[i].id);
        fprintf(f, ", \"compiled\": ");     json_str(f, rows[i].compiled);
        fprintf(f, ", \"supported\": ");    json_str(f, rows[i].supported);
        fprintf(f, ", \"env\": ");          json_str(f, rows[i].env_name ? rows[i].env_name : "");
        fprintf(f, ", \"env_value\": ");    json_str(f, rows[i].env_name ? rows[i].env_value : "");
        fprintf(f, ", \"resolved\": ");     json_str(f, rows[i].resolved);
        fprintf(f, ", \"source\": ");       json_str(f, source_name(rows[i].source));
        fprintf(f, ", \"reason\": ");       json_str(f, rows[i].reason);
        fprintf(f, "}%s\n", i + 1 < n ? "," : "");
    }
    fprintf(f, "  ]\n}\n");
    fflush(f);
    return unknown;
}

int mynah_dispatch_report(void *out_file, int as_json) {
    FILE *f = out_file != NULL ? (FILE *)out_file : stderr;
    mynah_dispatch_row rows[MYNAH_DISPATCH_MAX_ROWS];
    const int n = mynah_dispatch_collect(rows, MYNAH_DISPATCH_MAX_ROWS);
    if (n < 0) return -1;
    return write_report(f, as_json, rows, n);
}

static void expand_pid(char *dst, size_t cap, const char *src) {
    size_t o = 0;
    for (size_t i = 0; src[i] != 0 && o + 1 < cap; ++i) {
        if (src[i] == '%' && src[i + 1] == 'd') {
            const int k = snprintf(dst + o, cap - o, "%d", (int)getpid());
            if (k > 0) o += (size_t)k;
            ++i;
        } else {
            dst[o++] = src[i];
        }
    }
    dst[o < cap ? o : cap - 1] = '\0';
}

int mynah_dispatch_report_json_path(const char *path) {
    if (path == NULL || path[0] == 0) path = getenv("MYNAH_DISPATCH_JSON");
    if (path == NULL || path[0] == 0) return 0;
    char real[1024];
    expand_pid(real, sizeof real, path);
    FILE *f = fopen(real, "w");
    if (f == NULL) return -1;
    const int r = mynah_dispatch_report(f, 1);
    fclose(f);
    return r;
}

/* ======================================================================
 * Self-test
 * ====================================================================== */

static int st_fail(char *error, size_t cap, const char *message) {
    if (error != NULL && cap > 0) snprintf(error, cap, "%s", message);
    return -1;
}

static int st_probe_on(const char **why) {
    *why = "[predicate] registered by mynah_dispatch_self_test";
    return 1;
}

int mynah_dispatch_self_test(char *error, size_t error_capacity) {
    /* E9-P2. The pool's narrowing levers decide WHO runs a chunk, and the claim
     * that this cannot change WHAT is computed is the gate the whole item rests
     * on. It is checked here, first and unconditionally, because this is the
     * one self-test that every build and both sanitizers already run. */
    if (mynah_threads_self_test(error, error_capacity) != 0) return -1;

    mynah_dispatch_row rows[MYNAH_DISPATCH_MAX_ROWS];
    int n = mynah_dispatch_collect(rows, MYNAH_DISPATCH_MAX_ROWS);
    if (n <= 0) return st_fail(error, error_capacity, "dispatch: no rows collected");
    if (n >= MYNAH_DISPATCH_MAX_ROWS) {
        return st_fail(error, error_capacity,
                       "dispatch: row table is full, raise MYNAH_DISPATCH_MAX_ROWS");
    }

    for (int i = 0; i < n; ++i) {
        if (rows[i].id == NULL || rows[i].id[0] == 0) {
            return st_fail(error, error_capacity, "dispatch: row with no id");
        }
        if (rows[i].resolved[0] == 0) {
            return st_fail(error, error_capacity, "dispatch: row with no resolved value");
        }
        if (rows[i].reason[0] == 0) {
            return st_fail(error, error_capacity, "dispatch: row with no reason");
        }
        for (int j = i + 1; j < n; ++j) {
            if (strcmp(rows[i].id, rows[j].id) == 0) {
                return st_fail(error, error_capacity, "dispatch: duplicate row id");
            }
        }
    }

    /* No id may carry both kinds of probe: the value probe would win and the
     * boolean one would be dead code that still reads as authoritative. */
    for (int i = 0; i < g_value_probe_count; ++i) {
        for (int j = 0; j < g_probe_count; ++j) {
            if (strcmp(g_value_probes[i].id, g_probes[j].id) == 0) {
                return st_fail(error, error_capacity,
                               "dispatch: id has both a value probe and a "
                               "boolean probe");
            }
        }
    }

    /* A registered predicate must override the gate: that is the mechanism the
     * whole "never re-derive" rule rests on.  AMX is the target because it is
     * the one remaining row with no owner -- hijacking a row that a module
     * really does register would leave the process with the test's probe. */
    const char *target = "isa.x86.amx_int8";   /* compiled=no, gate says OFF */
    if (mynah_dispatch_register_probe(target, st_probe_on) != 0) {
        return st_fail(error, error_capacity, "dispatch: probe registration failed");
    }
    n = mynah_dispatch_collect(rows, MYNAH_DISPATCH_MAX_ROWS);
    int seen = 0;
    for (int i = 0; i < n; ++i) {
        if (strcmp(rows[i].id, target) != 0) continue;
        seen = 1;
        if (rows[i].source != MYNAH_DISPATCH_SRC_PREDICATE ||
            strcmp(rows[i].resolved, "ON") != 0) {
            return st_fail(error, error_capacity,
                           "dispatch: registered predicate did not override the gate");
        }
    }
    /* Unregister by replacing with the gate again is not supported; drop the
     * entry so a later report is not poisoned by the test. */
    for (int i = 0; i < g_probe_count; ++i) {
        if (strcmp(g_probes[i].id, target) == 0) {
            g_probes[i] = g_probes[--g_probe_count];
            break;
        }
    }
    if (!seen) return st_fail(error, error_capacity, "dispatch: probe target row missing");

    const char *detail = "";
    if (drift_f16(&detail) != 0) return st_fail(error, error_capacity, detail);

    /* The census is the empirical half of this file and is checked with it:
     * a dispatch report whose census machinery is broken can still claim a
     * kernel ran. */
    if (mynah_census_self_test(error, error_capacity) != 0) return -1;
    return 0;
}

/* ======================================================================
 * THE SHAPE AND KERNEL CENSUS — see dispatch.h for why a count and not a
 * banner.  Same no-allocation, no-shared-atomic discipline as costmap.c: if
 * the census changed what it measures, every number it produced would be
 * about a program that does not ship.
 * ====================================================================== */

#include "costmap.h"

#include <stdatomic.h>
#include <stdint.h>
#include <pthread.h>

int mynah_census_on_v = 0;

const char *mynah_census_path_name(int path) {
    switch (path) {
        case MYNAH_CENSUS_PATH_MATVEC_F32: return "matvec-f32";
        case MYNAH_CENSUS_PATH_MATVEC_Q:   return "matvec-q";
        case MYNAH_CENSUS_PATH_GEMM:       return "gemm";
        case MYNAH_CENSUS_PATH_BATCHED_Q:  return "batched-q";
        case MYNAH_CENSUS_PATH_ROWLOOP:    return "row-loop";
        default:                           return "UNKNOWN";
    }
}

static const char *census_qname(int qtype) {
    switch (qtype) {
        case 0:  return "f32";
        case 1:  return "int8";
        case 2:  return "int4";
        case 3:  return "f16";
        default: return "f32";       /* -1: the group was not selected */
    }
}

/* One distinct (block, kind, shape, width, path, encoding).  `block` and
 * `kind` are static strings, so the key is a pointer compare. */
typedef struct {
    const char *block;
    const char *kind;
    uint32_t    k, n, rows;
    int16_t     path;
    int16_t     qtype;
    uint64_t    calls;
    uint64_t    row_total;
} census_op;

#define CENSUS_SLOTS_PER_THREAD 192
#define CENSUS_TLS_SLOTS        68   /* as costmap.c: pool + caller + server */

typedef struct {
    census_op ops[CENSUS_SLOTS_PER_THREAD];
    int       used;
    uint64_t  dropped;     /* table full: counted, never silently discarded */
    int       in_use;
} census_tls;

static census_tls       g_cs[CENSUS_TLS_SLOTS];
static atomic_int       g_cs_next;
static atomic_ullong    g_cs_tls_overflow;
static atomic_ullong    g_cs_dropped;
static _Thread_local census_tls *t_cs;

static census_tls *census_self(void) {
    census_tls *t = t_cs;
    if (t != NULL) return t;
    const int slot = atomic_fetch_add_explicit(&g_cs_next, 1, memory_order_relaxed);
    if (slot >= CENSUS_TLS_SLOTS) {
        atomic_fetch_add_explicit(&g_cs_tls_overflow, 1ull, memory_order_relaxed);
        return NULL;
    }
    t = &g_cs[slot];
    t->in_use = 1;
    t_cs = t;
    return t;
}

void mynah_census_init(void) {
    static int done = 0;
    if (done) return;
    done = 1;
    const char *e = getenv("MYNAH_CENSUS");
    const char *j = getenv("MYNAH_CENSUS_JSON");
    int on = (e != NULL && e[0] != 0 && e[0] != '0');
    if (!on && j != NULL && j[0] != 0) on = 1;
    mynah_census_on_v = on;
}

int mynah_census_enabled(void) { return mynah_census_on_v; }

void mynah_census_op_(const char *block, const char *kind, size_t k, size_t n,
                      size_t rows, int path, int qtype) {
    census_tls *t = census_self();
    if (t == NULL || block == NULL || kind == NULL) return;
    /* Linear scan over this thread's own table.  It is short (one entry per
     * distinct shape, not per layer) and it is thread-local, so there is no
     * lock and no shared line to bounce. */
    for (int i = 0; i < t->used; ++i) {
        census_op *o = &t->ops[i];
        if (o->block == block && o->kind == kind && o->k == (uint32_t)k &&
            o->n == (uint32_t)n && o->rows == (uint32_t)rows &&
            o->path == (int16_t)path && o->qtype == (int16_t)qtype) {
            ++o->calls;
            o->row_total += (uint64_t)rows;
            return;
        }
    }
    if (t->used >= CENSUS_SLOTS_PER_THREAD) {
        ++t->dropped;
        atomic_fetch_add_explicit(&g_cs_dropped, 1ull, memory_order_relaxed);
        return;
    }
    census_op *o = &t->ops[t->used++];
    o->block = block;
    o->kind = kind;
    o->k = (uint32_t)k;
    o->n = (uint32_t)n;
    o->rows = (uint32_t)rows;
    o->path = (int16_t)path;
    o->qtype = (int16_t)qtype;
    o->calls = 1;
    o->row_total = (uint64_t)rows;
}

void mynah_census_reset(void) {
    for (int i = 0; i < CENSUS_TLS_SLOTS; ++i) {
        g_cs[i].used = 0;
        g_cs[i].dropped = 0;
        memset(g_cs[i].ops, 0, sizeof(g_cs[i].ops));
    }
    atomic_store_explicit(&g_cs_dropped, 0ull, memory_order_relaxed);
    atomic_store_explicit(&g_cs_tls_overflow, 0ull, memory_order_relaxed);
}

static int census_tls_used(void) {
    const int n = atomic_load_explicit(&g_cs_next, memory_order_relaxed);
    return n > CENSUS_TLS_SLOTS ? CENSUS_TLS_SLOTS : n;
}

int mynah_census_merge(mynah_census_row *out, int capacity) {
    if (out == NULL || capacity <= 0) return -1;
    int n = 0;
    const int used = census_tls_used();
    for (int s = 0; s < used; ++s) {
        const census_tls *t = &g_cs[s];
        if (!t->in_use) continue;
        for (int i = 0; i < t->used; ++i) {
            const census_op *o = &t->ops[i];
            int found = -1;
            for (int j = 0; j < n; ++j) {
                if (out[j].block == o->block && out[j].kind == o->kind &&
                    out[j].k == o->k && out[j].n == o->n &&
                    out[j].rows == o->rows && out[j].path == o->path &&
                    out[j].qtype == o->qtype) { found = j; break; }
            }
            if (found < 0) {
                if (n >= capacity) return n;
                found = n++;
                memset(&out[found], 0, sizeof(out[found]));
                out[found].block = o->block;
                out[found].kind = o->kind;
                out[found].k = o->k;
                out[found].n = o->n;
                out[found].rows = o->rows;
                out[found].path = o->path;
                out[found].qtype = o->qtype;
            }
            out[found].calls += o->calls;
            out[found].row_total += o->row_total;
            ++out[found].threads_seen;
        }
    }
    return n;
}

/* The allocation counter lives in tests/alloc_shim.c and is preloaded, never
 * linked: with no shim this stays null and the census says [UNKNOWN] rather
 * than printing a number it did not measure. */
static unsigned long long (*g_alloc_source)(void) = NULL;

void mynah_census_set_alloc_source(unsigned long long (*fn)(void)) {
    g_alloc_source = fn;
}

unsigned long long mynah_census_alloc_count(void) {
    return g_alloc_source != NULL ? g_alloc_source() : 0ull;
}

/* ---- the refusals -------------------------------------------------------
 *
 * Each of these is a sentence the census would otherwise let somebody write
 * and be wrong about.  They are checked against the DISPATCH table collected
 * in the same process, which is the only scope in which "resolved ON" and
 * "never executed" are about the same binary on the same host. */

static const char *row_resolved(const mynah_dispatch_row *rows, int n,
                                const char *id) {
    for (int i = 0; i < n; ++i) if (strcmp(rows[i].id, id) == 0) return rows[i].resolved;
    return NULL;
}

/* Did any op carry this encoding?  `want_batched` narrows it to the
 * weight-stationary path, which is the only place the SMMLA kernel lives. */
static int census_saw_qtype(const mynah_census_row *rows, int n, int qtype,
                            int want_batched) {
    for (int i = 0; i < n; ++i) {
        if (rows[i].qtype != qtype) continue;
        if (want_batched && rows[i].path != MYNAH_CENSUS_PATH_BATCHED_Q) continue;
        if (rows[i].calls > 0) return 1;
    }
    return 0;
}

int mynah_census_refusals(char *reason, size_t capacity) {
    mynah_census_row rows[MYNAH_CENSUS_MAX_ROWS];
    const int n = mynah_census_merge(rows, MYNAH_CENSUS_MAX_ROWS);
    int refusals = 0;
    char first[512];
    first[0] = 0;

#define CENSUS_REFUSE(...)                                                     \
    do {                                                                       \
        if (refusals++ == 0) snprintf(first, sizeof first, __VA_ARGS__);        \
    } while (0)

    if (n < 0) {
        CENSUS_REFUSE("the census could not be merged");
        goto done;
    }

    /* R1 -- a hole in the census.  A table with an unattributed op cannot
     * support "these are the kernels that ran", so it is not printed as if it
     * could. */
    for (int i = 0; i < n; ++i) {
        if (rows[i].path == MYNAH_CENSUS_PATH_UNKNOWN) {
            CENSUS_REFUSE("R1 UNKNOWN operation: %s.%s [%llux%llu] ran %llu times "
                          "on a path no hook named. The census has a hole in it "
                          "and cannot say which kernels ran",
                          rows[i].block, rows[i].kind,
                          (unsigned long long)rows[i].k,
                          (unsigned long long)rows[i].n,
                          (unsigned long long)rows[i].calls);
        }
    }
    if (atomic_load_explicit(&g_cs_dropped, memory_order_relaxed) != 0ull) {
        CENSUS_REFUSE("R1 the census table was full and %llu distinct operations "
                      "were dropped: the table is incomplete",
                      (unsigned long long)atomic_load_explicit(&g_cs_dropped,
                                                              memory_order_relaxed));
    }
    if (atomic_load_explicit(&g_cs_tls_overflow, memory_order_relaxed) != 0ull) {
        CENSUS_REFUSE("R1 a thread could not claim a census block, so its "
                      "operations were not counted at all");
    }

    /* R2 and R3 compare the census against the dispatch table, and collecting
     * that table is NOT free: mynah_dispatch_collect() opens the backends and
     * runs the model-free kernel self-tests (~200 ms on a 32-core Neoverse).
     * Two consequences, both measured rather than assumed:
     *
     *   - it is collected ONCE per process and cached, not per refusal check;
     *   - it is not collected at all when the census is empty, because a
     *     census with no rows has nothing to cross-check.
     *
     * This cost is entirely at EXIT, after the audio is written. The in-run
     * synthesis time is unaffected -- measured clean 0.664/0.686/0.698 s
     * against census 0.681/0.671/0.678 s on the same host -- which is the
     * property that matters: the instrumentation does not change what it
     * measures. `make census-overhead` reports the two separately for exactly
     * this reason. */
    if (n > 0) {
        static mynah_dispatch_row drows[MYNAH_DISPATCH_MAX_ROWS];
        static int dn = -1;
        if (dn < 0) dn = mynah_dispatch_collect(drows, MYNAH_DISPATCH_MAX_ROWS);
        if (dn > 0) {
            /* R2 -- an encoding was asked for and resolved, and nothing used
             * it.  This is the "I set MYNAH_QUANT=int8 and measured f32"
             * failure, which otherwise looks exactly like a fast run. */
            const char *req = row_resolved(drows, dn, "quant.requested");
            if (req != NULL && strcmp(req, "off") != 0 && n > 0) {
                const int qt = mynah_qmat_qtype_from_name(req);
                if (qt > 0 && !census_saw_qtype(rows, n, qt, 0)) {
                    CENSUS_REFUSE("R2 quant.requested resolved to '%s' and NOT ONE "
                                  "projection carried that encoding. Whatever was "
                                  "measured, it was not %s", req, req);
                }
            }
            /* R3 -- a kernel class resolved ON that never ran.  The SMMLA
             * kernel exists only on the weight-stationary path, so i8mm ON
             * with zero batched int8 calls means the feature was reported and
             * idled. */
            const char *i8mm = row_resolved(drows, dn, "isa.arm.i8mm");
            if (i8mm != NULL && strcmp(i8mm, "ON") == 0 && n > 0 &&
                census_saw_qtype(rows, n, 1, 0) &&
                !census_saw_qtype(rows, n, 1, 1)) {
                CENSUS_REFUSE("R3 isa.arm.i8mm resolved ON, int8 projections ran, "
                              "and none of them took the batched path -- the "
                              "SMMLA kernel is only reachable there, so the "
                              "feature was reported and idled");
            }
            const char *k8 = row_resolved(drows, dn, "quant.int8_kernel");
            if (k8 != NULL && (strstr(k8, "vnni") != NULL) && n > 0 &&
                !census_saw_qtype(rows, n, 1, 0)) {
                CENSUS_REFUSE("R3 quant.int8_kernel resolved to '%s' and no int8 "
                              "projection executed: the kernel named in this "
                              "run's report never ran in it", k8);
            }
        }
    }

    /* R4 -- the cost map was on and rejected its own numbers.  A census taken
     * beside an untrustworthy profile is still a true census, but the pair is
     * what gets read, and the pair is not trustworthy. */
    if (mynah_costmap_level() != 0) {
        char why[256];
        if (mynah_costmap_trustworthy(why, sizeof why) != 0) {
            CENSUS_REFUSE("R4 the cost map refused its own numbers in this run "
                          "(%s), so this census and that profile must not be "
                          "read together", why);
        }
    }

    /* R5 -- per-request scope.  The census is process-wide and summed over
     * threads. Dividing it by a request count the process does not have is the
     * WORKER-by-HOST division the roofs tool refuses; MYNAH_CENSUS_PER_REQUEST
     * is the explicit request for that number and the cost map is the only
     * thing that counts requests. */
    {
        const char *want = getenv("MYNAH_CENSUS_PER_REQUEST");
        if (want != NULL && want[0] != 0 && want[0] != '0') {
            mynah_costmap_health h;
            mynah_costmap_health_get(&h);
            if (mynah_costmap_level() == 0 || h.requests == 0u) {
                CENSUS_REFUSE("R5 a per-request figure was asked for, and this "
                              "process has no request count to divide by "
                              "(MYNAH_COST_MAP is off, or no request completed). "
                              "The scopes do not match and no number is printed");
            }
        }
    }

done:
#undef CENSUS_REFUSE
    if (refusals > 0 && reason != NULL && capacity > 0) {
        snprintf(reason, capacity, "%s", first);
    }
    return refusals;
}

/* ---- the report ---------------------------------------------------------- */

/* Which ISA kernel a row's encoding resolves to in THIS process.  Taken from
 * src/qmat.c's own predicate, never from a compile gate: that is the same rule
 * the dispatch table lives by, and the reason an attribution here is checkable
 * rather than plausible. */
static const char *census_kernel_for(int qtype, int path) {
    const char *why = NULL;
    switch (qtype) {
        case 1:
            if (path == MYNAH_CENSUS_PATH_BATCHED_Q && mynah_qmat_i8mm_enabled(&why)) {
                return "arm-smmla";
            }
            return mynah_qmat_int8_kernel(&why);
        case 2:  return mynah_qmat_int4_kernel(&why);
        case 3:  return mynah_qmat_f16_kernel(&why);
        default: return "f32";
    }
}

static int census_cmp(const void *a, const void *b) {
    const mynah_census_row *x = (const mynah_census_row *)a;
    const mynah_census_row *y = (const mynah_census_row *)b;
    const int c = strcmp(x->block, y->block);
    if (c != 0) return c;
    if (x->calls != y->calls) return x->calls < y->calls ? 1 : -1;
    return strcmp(x->kind, y->kind);
}

int mynah_census_report(void *out_file, int as_json) {
    FILE *f = out_file != NULL ? (FILE *)out_file : stderr;
    mynah_census_row rows[MYNAH_CENSUS_MAX_ROWS];
    int n = mynah_census_merge(rows, MYNAH_CENSUS_MAX_ROWS);
    if (n < 0) return -1;
    qsort(rows, (size_t)n, sizeof(rows[0]), census_cmp);

    char refusal[512];
    refusal[0] = 0;
    const int refusals = mynah_census_refusals(refusal, sizeof refusal);

    unsigned long long total_calls = 0, batched_calls = 0, rowloop_calls = 0;
    for (int i = 0; i < n; ++i) {
        total_calls += rows[i].calls;
        if (rows[i].path == MYNAH_CENSUS_PATH_BATCHED_Q) batched_calls += rows[i].calls;
        if (rows[i].path == MYNAH_CENSUS_PATH_ROWLOOP)    rowloop_calls += rows[i].calls;
    }
    const unsigned long long allocs = mynah_census_alloc_count();
    mynah_costmap_health h;
    mynah_costmap_health_get(&h);
    const int have_requests = (mynah_costmap_level() != 0 && h.requests > 0u);

    if (!as_json) {
        fprintf(f, "[CENSUS] v=1 pid=%d rows=%d calls=%llu build=%s simd=%s\n",
                (int)getpid(), n, (unsigned long long)total_calls,
                MYNAH_GIT_REV, MYNAH_SIMD_PROFILE);
        fprintf(f, "  every PocketTTS projection that executed, by shape and by "
                   "the path it ACTUALLY took. `batched-q` is micro-batching "
                   "that engaged; `row-loop` is a call that asked for several "
                   "rows and got a per-row loop, which is the fallback a banner "
                   "cannot see.\n");
        if (n == 0) {
            fprintf(f, "  (no operations recorded: nothing ran, or MYNAH_CENSUS "
                       "was set after the work)\n");
        } else {
            fprintf(f, "  %-14s %-8s %12s %6s %-11s %-5s %-12s %10s\n",
                    "block", "kind", "shape k x n", "rows", "path", "enc",
                    "kernel", "calls");
            for (int i = 0; i < n; ++i) {
                char shape[24];
                snprintf(shape, sizeof shape, "%llux%llu",
                         (unsigned long long)rows[i].k,
                         (unsigned long long)rows[i].n);
                fprintf(f, "  %-14s %-8s %12s %6llu %-11s %-5s %-12s %10llu\n",
                        rows[i].block, rows[i].kind, shape,
                        (unsigned long long)rows[i].rows,
                        mynah_census_path_name(rows[i].path),
                        census_qname(rows[i].qtype),
                        census_kernel_for(rows[i].qtype, rows[i].path),
                        (unsigned long long)rows[i].calls);
            }
        }
        /* The count, not the banner. */
        fprintf(f, "  micro-batching: %llu of %llu calls took the batched path, "
                   "%llu took a per-row loop. %s\n",
                (unsigned long long)batched_calls, (unsigned long long)total_calls,
                (unsigned long long)rowloop_calls,
                batched_calls == 0
                    ? "IT DID NOT ENGAGE in this run -- whatever the build says."
                    : "It engaged.");
        if (allocs > 0ull) {
            fprintf(f, "  allocations: %llu [MEASURED, tests/alloc_shim.c]",
                    (unsigned long long)allocs);
            if (have_requests) {
                fprintf(f, ", %.1f per request over %llu requests\n",
                        (double)allocs / (double)h.requests,
                        (unsigned long long)h.requests);
            } else {
                fprintf(f, "; per request [UNKNOWN]: no request count in this "
                           "process (set MYNAH_COST_MAP=1)\n");
            }
        } else {
            fprintf(f, "  allocations: [UNKNOWN] -- no counter is preloaded. "
                       "The runtime does not interpose its own allocator; "
                       "run under tests/alloc_shim to measure it.\n");
        }
        if (refusals > 0) {
            fprintf(f, "  CENSUS REFUSED (%d): %s\n", refusals, refusal);
            fprintf(f, "  A refusing census does not print a verdict. Fix the "
                       "cause or stop quoting the run.\n");
        } else {
            fprintf(f, "  census: OK -- every operation is attributed, and every "
                       "kernel class this run resolved ON executed at least "
                       "once.\n");
        }
        fflush(f);
        return n;
    }

    fprintf(f, "{\n  \"v\": 1,\n  \"pid\": %d,\n", (int)getpid());
    fprintf(f, "  \"build\": ");  json_str(f, MYNAH_GIT_REV);
    fprintf(f, ",\n  \"simd\": "); json_str(f, MYNAH_SIMD_PROFILE);
    fprintf(f, ",\n  \"calls\": %llu", (unsigned long long)total_calls);
    fprintf(f, ",\n  \"batched_calls\": %llu", (unsigned long long)batched_calls);
    fprintf(f, ",\n  \"rowloop_calls\": %llu", (unsigned long long)rowloop_calls);
    fprintf(f, ",\n  \"micro_batching_engaged\": %s",
            batched_calls > 0 ? "true" : "false");
    if (allocs > 0ull) {
        fprintf(f, ",\n  \"allocations\": %llu", (unsigned long long)allocs);
        fprintf(f, ",\n  \"allocations_label\": \"MEASURED\"");
        if (have_requests) {
            fprintf(f, ",\n  \"requests\": %llu", (unsigned long long)h.requests);
            fprintf(f, ",\n  \"allocations_per_request\": %.3f",
                    (double)allocs / (double)h.requests);
        } else {
            fprintf(f, ",\n  \"allocations_per_request\": null");
        }
    } else {
        fprintf(f, ",\n  \"allocations\": null,\n  \"allocations_label\": \"UNKNOWN\"");
    }
    fprintf(f, ",\n  \"refusals\": %d", refusals);
    fprintf(f, ",\n  \"refusal\": "); json_str(f, refusal);
    fprintf(f, ",\n  \"ops\": [\n");
    for (int i = 0; i < n; ++i) {
        fprintf(f, "    {\"block\": ");  json_str(f, rows[i].block);
        fprintf(f, ", \"kind\": ");      json_str(f, rows[i].kind);
        fprintf(f, ", \"k\": %llu, \"n\": %llu, \"rows\": %llu",
                (unsigned long long)rows[i].k, (unsigned long long)rows[i].n,
                (unsigned long long)rows[i].rows);
        fprintf(f, ", \"path\": ");      json_str(f, mynah_census_path_name(rows[i].path));
        fprintf(f, ", \"encoding\": ");  json_str(f, census_qname(rows[i].qtype));
        fprintf(f, ", \"kernel\": ");    json_str(f, census_kernel_for(rows[i].qtype, rows[i].path));
        fprintf(f, ", \"calls\": %llu, \"row_total\": %llu, \"threads\": %u}%s\n",
                (unsigned long long)rows[i].calls,
                (unsigned long long)rows[i].row_total, rows[i].threads_seen,
                i + 1 < n ? "," : "");
    }
    fprintf(f, "  ]\n}\n");
    fflush(f);
    return n;
}

static void census_atexit(void) {
    if (!mynah_census_on_v) return;
    const char *path = getenv("MYNAH_CENSUS_JSON");
    (void)0;
    if (path != NULL && path[0] != 0) {
        char real[1024];
        expand_pid(real, sizeof real, path);
        FILE *j = fopen(real, "w");
        if (j != NULL) { mynah_census_report(j, 1); fclose(j); }
    } else {
        mynah_census_report(stderr, 0);
    }
    const char *strict = getenv("MYNAH_CENSUS_STRICT");
    if (strict != NULL && strict[0] != 0 && strict[0] != '0') {
        char reason[512];
        const int refusals = mynah_census_refusals(reason, sizeof reason);
        if (refusals > 0) {
            fprintf(stderr,
                    "[CENSUS] REFUSED (%d): %s.\n"
                    "         MYNAH_CENSUS_STRICT is set, so this run exits "
                    "non-zero rather than print a number nobody should trust.\n",
                    refusals, reason);
            fflush(stderr);
            _exit(5);
        }
    }
}

__attribute__((constructor)) static void census_ctor(void) {
    mynah_census_init();
    if (mynah_census_on_v) atexit(census_atexit);
}

/* ---- census self-test ---------------------------------------------------
 *
 * Model-free.  It asserts the properties the census's own claims rest on:
 * identical keys merge, different shapes do NOT merge, an off census records
 * nothing, a second thread accumulates into its own block and merges exactly,
 * and an UNKNOWN op is REFUSED rather than printed.  It resets the table on
 * entry and on exit, so it must never run inside a profiled synthesis. */

static int cs_fail(char *error, size_t cap, const char *message) {
    if (error != NULL && cap > 0) snprintf(error, cap, "%s", message);
    return -1;
}

#define CS_THREAD_ITERS 100
static const char cs_block_a[] = "selftest";
static const char cs_kind_a[]  = "alpha";
static const char cs_kind_b[]  = "beta";

static void *cs_worker(void *arg) {
    (void)arg;
    for (int i = 0; i < CS_THREAD_ITERS; ++i) {
        mynah_census_op(cs_block_a, cs_kind_a, 64, 128, 1,
                        MYNAH_CENSUS_PATH_MATVEC_Q, 1);
    }
    return NULL;
}

static const mynah_census_row *cs_find(const mynah_census_row *r, int n,
                                       const char *kind, int path) {
    for (int i = 0; i < n; ++i) {
        if (strcmp(r[i].kind, kind) == 0 && r[i].path == path) return &r[i];
    }
    return NULL;
}

int mynah_census_self_test(char *error, size_t error_capacity) {
    const int saved = mynah_census_on_v;
    mynah_census_row rows[MYNAH_CENSUS_MAX_ROWS];
    int rc = 0;
    int n;

    mynah_census_on_v = 1;
    mynah_census_reset();

    /* 1. identical keys merge; a different shape does not. */
    for (int i = 0; i < 5; ++i) {
        mynah_census_op(cs_block_a, cs_kind_a, 64, 128, 1,
                        MYNAH_CENSUS_PATH_MATVEC_Q, 1);
    }
    mynah_census_op(cs_block_a, cs_kind_a, 64, 256, 1,
                    MYNAH_CENSUS_PATH_MATVEC_Q, 1);
    n = mynah_census_merge(rows, MYNAH_CENSUS_MAX_ROWS);
    if (n != 2) { rc = cs_fail(error, error_capacity, "census: shapes did not separate"); goto done; }
    {
        unsigned long long five = 0, one = 0;
        for (int i = 0; i < n; ++i) {
            if (rows[i].n == 128) five = rows[i].calls;
            if (rows[i].n == 256) one = rows[i].calls;
        }
        if (five != 5 || one != 1) {
            rc = cs_fail(error, error_capacity, "census: identical keys did not merge");
            goto done;
        }
    }

    /* 2. the path is part of the key: the same shape on two paths must stay
     *    two rows, or "did micro-batching engage" could never be answered. */
    mynah_census_reset();
    mynah_census_op(cs_block_a, cs_kind_b, 64, 128, 4,
                    MYNAH_CENSUS_PATH_BATCHED_Q, 1);
    mynah_census_op(cs_block_a, cs_kind_b, 64, 128, 4,
                    MYNAH_CENSUS_PATH_ROWLOOP, 1);
    n = mynah_census_merge(rows, MYNAH_CENSUS_MAX_ROWS);
    if (n != 2 || cs_find(rows, n, cs_kind_b, MYNAH_CENSUS_PATH_BATCHED_Q) == NULL ||
        cs_find(rows, n, cs_kind_b, MYNAH_CENSUS_PATH_ROWLOOP) == NULL) {
        rc = cs_fail(error, error_capacity,
                     "census: a batched call and a row-loop of the same shape merged");
        goto done;
    }
    {
        const mynah_census_row *b = cs_find(rows, n, cs_kind_b, MYNAH_CENSUS_PATH_BATCHED_Q);
        if (b->row_total != 4) {
            rc = cs_fail(error, error_capacity, "census: row_total is not the width");
            goto done;
        }
    }

    /* 3. an UNKNOWN op REFUSES. */
    mynah_census_reset();
    mynah_census_op(cs_block_a, cs_kind_a, 8, 8, 1, MYNAH_CENSUS_PATH_UNKNOWN, -1);
    {
        char why[512];
        if (mynah_census_refusals(why, sizeof why) == 0) {
            rc = cs_fail(error, error_capacity,
                         "census: an UNKNOWN operation was NOT refused");
            goto done;
        }
        if (strstr(why, "R1") == NULL) {
            rc = cs_fail(error, error_capacity,
                         "census: the UNKNOWN refusal is not reported as R1");
            goto done;
        }
    }

    /* 4. a clean table does not refuse.
     *
     * THE REASON IS CARRIED, not discarded.  This used to pass NULL, so when
     * it failed on a Linux sanitizer build it could say "a clean table
     * refused" and not which of R1..R4 did it -- the one fact needed to fix
     * it, thrown away by the call itself.  A gate that detects a problem it
     * cannot describe costs a round trip through CI for every guess. */
    mynah_census_reset();
    mynah_census_op(cs_block_a, cs_kind_a, 8, 8, 1, MYNAH_CENSUS_PATH_MATVEC_F32, -1);
    /* AND AN INT8 OP ON THE BATCHED PATH, because "clean" is a property of the
     * table RELATIVE TO THE HOST and the old table was only clean on some
     * hosts.
     *
     * R3 refuses a census whose int8 kernel resolved to something and no int8
     * projection carried it -- "the kernel named in this run's report never
     * ran in it" -- and it is right to. A table holding one f32 matvec is
     * exactly that census on any machine whose int8 kernel resolves: it
     * failed on a GitHub x86 runner with `R3 quant.int8_kernel resolved to
     * 'avx512vnni' and no int8 projection executed`, and passed on this arm64
     * laptop and on the arm runner only because "neon-sdot" does not match the
     * VNNI clause. Two runners of the same `ubuntu-latest` label differ in
     * whether they have AVX-512 VNNI at all, so the old case was not merely
     * platform-dependent, it was FLAKY BY MACHINE.
     *
     * The batched path specifically: it satisfies both R3 clauses at once, the
     * VNNI one and the i8mm one (SMMLA is only reachable from the batched
     * linear, so i8mm ON with no batched int8 call is its own refusal). */
    mynah_census_op(cs_block_a, cs_kind_a, 8, 8, 1, MYNAH_CENSUS_PATH_BATCHED_Q, 1);
    {
        char why4[512];
        why4[0] = 0;
        if (mynah_census_refusals(why4, sizeof why4) != 0) {
            char msg4[600];
            snprintf(msg4, sizeof msg4, "census: a clean table refused: %s",
                     why4[0] ? why4 : "(no reason reported)");
            rc = cs_fail(error, error_capacity, msg4);
            goto done;
        }
    }

    /* 5. two threads accumulate independently and merge exactly. */
    mynah_census_reset();
    {
        pthread_t a, b;
        if (pthread_create(&a, NULL, cs_worker, NULL) != 0 ||
            pthread_create(&b, NULL, cs_worker, NULL) != 0) {
            rc = cs_fail(error, error_capacity, "census: cannot create threads");
            goto done;
        }
        pthread_join(a, NULL);
        pthread_join(b, NULL);
        n = mynah_census_merge(rows, MYNAH_CENSUS_MAX_ROWS);
        if (n != 1 || rows[0].calls != 2u * CS_THREAD_ITERS) {
            rc = cs_fail(error, error_capacity,
                         "census: merged call count is wrong across two threads");
            goto done;
        }
        if (rows[0].threads_seen != 2) {
            rc = cs_fail(error, error_capacity,
                         "census: the two threads did not get separate blocks");
            goto done;
        }
    }

    /* 6. off means off -- the property the whole parity proof rests on. */
    mynah_census_reset();
    mynah_census_on_v = 0;
    mynah_census_op(cs_block_a, cs_kind_a, 1, 1, 1, MYNAH_CENSUS_PATH_MATVEC_Q, 1);
    mynah_census_on_v = 1;
    n = mynah_census_merge(rows, MYNAH_CENSUS_MAX_ROWS);
    if (n != 0) {
        rc = cs_fail(error, error_capacity, "census: an op recorded while off");
        goto done;
    }

done:
    mynah_census_reset();
    mynah_census_on_v = saved;
    return rc;
}
