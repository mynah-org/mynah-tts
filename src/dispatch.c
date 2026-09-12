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
#else
static tri cpu_has_avx2(void)        { return 0; }
static tri cpu_has_fma(void)         { return 0; }
static tri cpu_has_avx512f(void)     { return 0; }
static tri cpu_has_avx512bw(void)    { return 0; }
static tri cpu_has_avx512vl(void)    { return 0; }
static tri cpu_has_avx512vnni(void)  { return 0; }
static tri cpu_has_avxvnni(void)     { return 0; }
static tri cpu_has_amx_int8(void)    { return 0; }
#endif

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
    /* Asked of the kernel, not of CFLAGS: SIMD=avx512 still says nothing about
     * which int8 kernel runs, and that gap is what produced the false claim. */
    {
        const char *k = mynah_qmat_int8_kernel(NULL);
        if (strcmp(k, "avx512vnni") == 0) return "x86_avx512_vnni";
        if (strcmp(k, "avxvnni") == 0) return "x86_avx_vnni";
    }
    if (!MYNAH_DISPATCH_HAS_AVX2) return "x86_scalar";
    return "x86_avx2";
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

#if MYNAH_DISPATCH_HAS_OPENBLAS && defined(__GNUC__) && !defined(__APPLE__)
/* The same weak reference src/threads.c:105 uses: it answers "did this process
 * actually link a threaded OpenBLAS", which no #define can. */
extern void openblas_set_num_threads(int) __attribute__((weak));
#endif

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
    add_absent(s, "isa.arm.bf16", cpu_has_bf16(),
               "[gate] NOT IMPLEMENTED: no bfdot/bfmmla and no bf16 weight "
               "type in src/qmat.c; every f32 path stays f32");
    add_gate(s, "isa.x86.avx2", MYNAH_DISPATCH_HAS_AVX2, cpu_has_avx2(),
             "[gate] src/kernels.c AVX2 kernels and src/qmat.c "
             "dot_q8_i32_avx2 (_mm256_cvtepi8_epi16 + _mm256_madd_epi16), the "
             "int8 dot for every x86 CPU without VPDPBUSD");
    add_gate(s, "isa.x86.fma", MYNAH_DISPATCH_HAS_FMA, cpu_has_fma(),
             "[gate] _mm256_fmadd_ps in the AVX2 dot/matvec; Makefile passes "
             "-mfma with -mavx2");

    /* The three rows the false README claim needed.  compiled can be yes here
     * (SIMD=avx512 passes the flags) while resolved is OFF, because no kernel
     * dispatches on it.  That gap is the finding, so it is stated twice. */
    add_row(s, "isa.x86.avx512f", yn(MYNAH_DISPATCH_HAS_AVX512F),
            yn3(cpu_has_avx512f()), NULL, "OFF", MYNAH_DISPATCH_SRC_GATE,
            "[gate] COMPILER FLAG ONLY. SIMD=avx512 widens autovectorization; "
            "no f32 kernel dispatches on it. The one _mm512_* kernel in src/ "
            "is the VNNI int8 dot, which carries its own target attribute and "
            "needs no build flag -- see isa.x86.avx512vnni, not this row");
    add_row(s, "isa.x86.avx512bw", yn(MYNAH_DISPATCH_HAS_AVX512BW),
            yn3(cpu_has_avx512bw()), NULL, "OFF", MYNAH_DISPATCH_SRC_GATE,
            "[gate] compiler flag only, as isa.x86.avx512f");
    add_row(s, "isa.x86.avx512vl", yn(MYNAH_DISPATCH_HAS_AVX512VL),
            yn3(cpu_has_avx512vl()), NULL, "OFF", MYNAH_DISPATCH_SRC_GATE,
            "[gate] compiler flag only, as isa.x86.avx512f");
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

static void collect_blas(row_sink *s) {
    tri accel_ok = -1;
#if defined(__APPLE__)
    accel_ok = 1;
#else
    accel_ok = 0;
#endif
    add_gate(s, "blas.accelerate", MYNAH_DISPATCH_HAS_ACCELERATE, accel_ok,
             "[gate] macOS Accelerate cblas_sgemm for the multi-row prefill and "
             "vvtanhf for the array GELU (src/backend.c, src/kernels.c)");

#if MYNAH_DISPATCH_HAS_OPENBLAS && defined(__GNUC__) && !defined(__APPLE__)
    {
        const int linked = (openblas_set_num_threads != NULL);
        add_row(s, "blas.openblas", "yes", yn(linked), NULL, onoff(linked),
                MYNAH_DISPATCH_SRC_RUNTIME,
                linked
                  ? "[runtime] weak openblas_set_num_threads resolved: a "
                    "threaded OpenBLAS is linked and src/threads.c can hold it "
                    "at one thread inside a parallel region"
                  : "[runtime] compiled against cblas.h but the weak symbol did "
                    "NOT resolve: no thread control, so the vendor BLAS keeps "
                    "its own team and nests under the pool");
    }
#else
    add_row(s, "blas.openblas", yn(MYNAH_DISPATCH_HAS_OPENBLAS), "-", NULL,
            onoff(MYNAH_DISPATCH_HAS_OPENBLAS), MYNAH_DISPATCH_SRC_GATE,
            MYNAH_DISPATCH_HAS_OPENBLAS
              ? "[gate] linked, but this platform has no weak-symbol thread "
                "control probe"
              : "[gate] not compiled (Linux BLAS=openblas selects it)");
#endif

    {   /* Not a CPU feature, so `supported` is "-" rather than a probe. */
        const int scalar_only =
            !(MYNAH_DISPATCH_HAS_ACCELERATE || MYNAH_DISPATCH_HAS_OPENBLAS);
        add_row(s, "blas.scalar_fallback", yn(scalar_only), "-", NULL,
                onoff(scalar_only), MYNAH_DISPATCH_SRC_GATE,
                "[gate] BLAS=scalar: every matmul falls back to the in-tree "
                "matvec loop. It is the correctness oracle, never the "
                "performance target");
    }

    add_unknown(s, "blas.threads_owned", "-", "-", "OPENBLAS_NUM_THREADS",
                "[UNKNOWN] src/threads.c did not register mynah_blas_owned()");
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
        static char text[240];
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
    collect_blas(&sink);
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
    for (int i = 0; i < n; ++i) {
        switch (rows[i].source) {
        case MYNAH_DISPATCH_SRC_PREDICATE: ++predicate; break;
        case MYNAH_DISPATCH_SRC_RUNTIME:   ++runtime;   break;
        case MYNAH_DISPATCH_SRC_GATE:      ++gate;      break;
        default:                           ++unknown;   break;
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
    return 0;
}
