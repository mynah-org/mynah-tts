/* dispatch.h — the dispatch report: every ISA / BLAS / quantization / pool
 * decision this binary can make, RESOLVED for this host and this environment.
 *
 * WHY THIS EXISTS.  Without it no ISA claim about this runtime is verifiable,
 * and an unverifiable claim eventually becomes a false one.  It already did:
 * `README.md` and commit 724d677 attributed the EPYC Zen 5 int8 result (0.427
 * RTF) to "AVX-512 VNNI" at a time when `src/qmat.c` contained no `_mm512_*`
 * and no `_mm256_dpbusd_epi32` at all — the int8 dot was the AVX2
 * widen-then-madd pair, and the Linux build did not even pass `-mavx512f`
 * unless `SIMD=avx512` was asked for.  The number was real; the attribution
 * was invented.  In qwen-tts the same report was written after a dispatch bug
 * hid ~400 ms of TTFA.
 *
 * src/qmat.c has since grown the real VPDPBUSD kernels, which changes nothing
 * about the rule: the `quant.int8_kernel` row names the kernel this process
 * will actually run, so "0.427 was VNNI" is now a checkable sentence rather
 * than a plausible one.  A number is only ever attributed to the kernel that
 * row printed in the same run.
 *
 * FIVE COLUMNS, per logical feature:
 *
 *   compiled   the implementation is in THIS binary (the build-time gate)
 *   supported  this CPU actually has the instructions/units (runtime probe)
 *   env        the raw environment variable that steers it, or "unset"
 *   resolved   what the runtime decision is NOW
 *   reason     which branch decided it, and where that branch lives
 *
 * THE CENTRAL RULE: `resolved` is never re-derived here.  `compiled &&
 * supported` is NOT an answer — that is precisely the guess that lets a report
 * agree with a README and both be wrong.  A row's `resolved` may come from
 * exactly three places, and the row says which one in `reason`:
 *
 *   [predicate]  a runtime predicate registered by the owning translation unit
 *                through mynah_dispatch_register_probe().  Always preferred.
 *   [runtime]    a real exported call this file makes and believes:
 *                mynah_num_threads(), mynah_qmat_cache_new()+_enabled(),
 *                mynah_backend_open(), mynah_*_self_test(), sysctl, cpuid.
 *   [gate]       a pure compile-time gate with NO runtime fallback, whose
 *                single definition is the MYNAH_DISPATCH_HAS_* macro below.
 *
 * Anything else resolves to UNKNOWN and names the wrapper that would fix it.
 * UNKNOWN is a finding, not a failure: the footer counts those rows because
 * they are the list of predicates E4 still has to export.
 *
 * THE [gate] ROWS AND DRIFT.  A compile gate is a fact about another
 * translation unit, so mirroring it here could drift.  Two things stop that:
 * the Makefile compiles every file in src/ with one CFLAGS (so the predefined
 * macros are identical by construction), and this file verifies that claim at
 * runtime with a canary — MYNAH_DISPATCH_HAS_QMAT_F16 is checked against what
 * src/qmat.c actually does, observed through mynah_qmat_cache_new(F16), and a
 * disagreement is printed as DRIFT.  The intended end state is that kernels.c
 * and qmat.c include this header and use these macros as their own gates, so
 * there is one definition rather than two that agree.
 */
#ifndef MYNAH_TTS_DISPATCH_H
#define MYNAH_TTS_DISPATCH_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------------
 * The compile gates of this runtime, in one place.
 *
 * Each macro is 1 or 0 and mirrors, expression for expression, the gate named
 * in its comment.  Nothing else in this header may test a raw __ARM_* or
 * __AVX* macro.
 * ------------------------------------------------------------------------ */

/* src/kernels.c:15 — NEON dot/matvec/rmsnorm/layernorm/gelu/axpy */
#if !defined(MYNAH_DISABLE_SIMD) && (defined(__ARM_NEON) || defined(__aarch64__))
#define MYNAH_DISPATCH_HAS_NEON 1
#else
#define MYNAH_DISPATCH_HAS_NEON 0
#endif

/* src/kernels.c:18 and src/qmat.c:18 — the AVX2 twins of the same kernels */
#if !defined(MYNAH_DISABLE_SIMD) && defined(__AVX2__)
#define MYNAH_DISPATCH_HAS_AVX2 1
#else
#define MYNAH_DISPATCH_HAS_AVX2 0
#endif

/* implied by -mfma; kernels.c uses _mm256_fmadd_ps unconditionally on AVX2 */
#if !defined(MYNAH_DISABLE_SIMD) && defined(__FMA__)
#define MYNAH_DISPATCH_HAS_FMA 1
#else
#define MYNAH_DISPATCH_HAS_FMA 0
#endif

/* src/qmat.c:14 — int8 dot via vdotq_s32 (ARM SDOT) */
#if !defined(MYNAH_DISABLE_SIMD) && defined(__ARM_FEATURE_DOTPROD)
#define MYNAH_DISPATCH_HAS_DOTPROD 1
#else
#define MYNAH_DISPATCH_HAS_DOTPROD 0
#endif

/* src/qmat.c:25 — __fp16 weight cache (needs NEON's half converts) */
#if !defined(MYNAH_DISABLE_SIMD) && defined(__aarch64__) && defined(__ARM_NEON)
#define MYNAH_DISPATCH_HAS_QMAT_F16 1
#else
#define MYNAH_DISPATCH_HAS_QMAT_F16 0
#endif

/* Makefile SIMD=avx512 passes -mavx512f -mavx512bw -mavx512vl.  These say the
 * compiler was allowed to use AVX-512; they do NOT say a kernel does.  No
 * _mm512_* intrinsic exists anywhere under src/ — see the report's reason. */
#if !defined(MYNAH_DISABLE_SIMD) && defined(__AVX512F__)
#define MYNAH_DISPATCH_HAS_AVX512F 1
#else
#define MYNAH_DISPATCH_HAS_AVX512F 0
#endif
#if !defined(MYNAH_DISABLE_SIMD) && defined(__AVX512BW__)
#define MYNAH_DISPATCH_HAS_AVX512BW 1
#else
#define MYNAH_DISPATCH_HAS_AVX512BW 0
#endif
#if !defined(MYNAH_DISABLE_SIMD) && defined(__AVX512VL__)
#define MYNAH_DISPATCH_HAS_AVX512VL 1
#else
#define MYNAH_DISPATCH_HAS_AVX512VL 0
#endif

/* Kernels that are compiled unconditionally and selected at RUNTIME, plus the
 * two features this repo still has no kernel for.  The latter are compiled=no
 * by construction and the report says so in words, rather than leaving a blank
 * the reader fills in optimistically. */
/* src/qmat.c carries the VPDPBUSD kernels behind __attribute__((target(...)))
 * rather than behind a build flag, so on x86 they are compiled into EVERY
 * build and selected at runtime by CPUID.  "compiled" is therefore a property
 * of the target and the compiler, not of CFLAGS -- which is the whole reason
 * SIMD=avx512 could once imply a kernel that did not exist. */
#if !defined(MYNAH_DISABLE_SIMD) && (defined(__x86_64__) || defined(__i386__)) && \
    (defined(__clang__) || (defined(__GNUC__) && __GNUC__ >= 11))
#define MYNAH_DISPATCH_HAS_AVX512VNNI_KERNEL 1  /* qmat.c dot4_u8_evex */
#define MYNAH_DISPATCH_HAS_AVXVNNI_KERNEL    1  /* qmat.c dot4_u8_vex  */
#else
#define MYNAH_DISPATCH_HAS_AVX512VNNI_KERNEL 0
#define MYNAH_DISPATCH_HAS_AVXVNNI_KERNEL    0
#endif

/* src/qmat.c matvec_q8_pair_i8mm, likewise target-attributed. */
#if !defined(MYNAH_DISABLE_SIMD) && defined(__aarch64__) && \
    (defined(__clang__) || (defined(__GNUC__) && __GNUC__ >= 10))
#define MYNAH_DISPATCH_HAS_I8MM_KERNEL 1
#else
#define MYNAH_DISPATCH_HAS_I8MM_KERNEL 0
#endif

#define MYNAH_DISPATCH_HAS_AMX_KERNEL        0   /* no tile ops                */
#define MYNAH_DISPATCH_HAS_BF16_KERNEL       0   /* no bfdot / bfmmla          */

#if defined(MYNAH_USE_ACCELERATE)
#define MYNAH_DISPATCH_HAS_ACCELERATE 1
#else
#define MYNAH_DISPATCH_HAS_ACCELERATE 0
#endif
#if defined(MYNAH_USE_OPENBLAS)
#define MYNAH_DISPATCH_HAS_OPENBLAS 1
#else
#define MYNAH_DISPATCH_HAS_OPENBLAS 0
#endif
#if defined(MYNAH_ENABLE_METAL)
#define MYNAH_DISPATCH_HAS_METAL 1
#else
#define MYNAH_DISPATCH_HAS_METAL 0
#endif
#if defined(MYNAH_ENABLE_CUDA)
#define MYNAH_DISPATCH_HAS_CUDA 1
#else
#define MYNAH_DISPATCH_HAS_CUDA 0
#endif

/* ------------------------------------------------------------------------
 * Rows
 * ------------------------------------------------------------------------ */

/* Where `resolved` came from.  Printed as a tag inside `reason`, and counted
 * in the footer, so a reader never has to assume. */
typedef enum {
    MYNAH_DISPATCH_SRC_PREDICATE = 0, /* registered runtime predicate         */
    MYNAH_DISPATCH_SRC_RUNTIME   = 1, /* a real exported call made from here  */
    MYNAH_DISPATCH_SRC_GATE      = 2, /* compile gate, no runtime fallback    */
    MYNAH_DISPATCH_SRC_UNKNOWN   = 3  /* nobody exports the predicate         */
} mynah_dispatch_source;

#define MYNAH_DISPATCH_MAX_ROWS 64

typedef struct {
    const char *id;             /* stable, dotted: component.feature[.impl]   */
    const char *compiled;       /* "yes" / "no" / "-"                          */
    const char *supported;      /* "yes" / "no" / "-" (unknown probes say "?") */
    const char *env_name;       /* the steering variable, or NULL              */
    char        env_value[64];  /* raw value, or "unset"                       */
    char        resolved[24];   /* ON / OFF / UNKNOWN / a value                */
    char        reason[240];
    mynah_dispatch_source source;
} mynah_dispatch_row;

/* ------------------------------------------------------------------------
 * Runtime predicates, registered by their owner
 *
 * A translation unit that owns a dispatch decision registers the predicate
 * that decides it; the report then CALLS it instead of guessing.  Register
 * from a constructor or from the module's init — order does not matter, the
 * table is consulted only when the report is built.
 *
 *   fn(&reason) returns  1 = on, 0 = off, -1 = the predicate itself cannot
 *   tell (it stays UNKNOWN).  *reason may be set to a static string.
 *
 * Registering an id that already has a probe replaces it.  Returns 0 on
 * success, -1 when the table is full or the arguments are NULL.
 * ------------------------------------------------------------------------ */
typedef int (*mynah_dispatch_probe_fn)(const char **reason);
int mynah_dispatch_register_probe(const char *id, mynah_dispatch_probe_fn fn);

/* Some decisions are not booleans.  "which int8 kernel resolved" and "which
 * matvec policy fired" have three or more answers, and squashing them into
 * ON/OFF is the same loss of information that let the AVX-512 claim survive:
 * a row saying ON next to a reason mentioning VNNI reads as VNNI even when the
 * kernel was AVX2.  A value probe writes the answer itself.
 *
 *   fn(out, capacity, &reason) returns 0 after writing `out`, or -1 when the
 *   predicate cannot tell (the row stays UNKNOWN).
 *
 * A value probe and a boolean probe on the same id is a programming error; the
 * value probe wins and mynah_dispatch_self_test() reports the collision. */
typedef int (*mynah_dispatch_value_probe_fn)(char *out, size_t capacity,
                                             const char **reason);
int mynah_dispatch_register_value_probe(const char *id,
                                        mynah_dispatch_value_probe_fn fn);

/* ------------------------------------------------------------------------
 * Where the predicates come from
 *
 * A module that owns a dispatch decision exports one of these and registers
 * its predicates inside it; mynah_dispatch_collect() calls all of them before
 * building the table.  An explicit call rather than a constructor, so the
 * registration cannot be dropped when this code is consumed as a static
 * library and the linker discards an object nothing else referenced.
 *
 * Every entry here is the answer to a row that used to read UNKNOWN.  The
 * fallback is still UNKNOWN: if a module stops registering, its rows go back
 * to saying so instead of silently reverting to a compile-time guess.
 * ------------------------------------------------------------------------ */
void mynah_qmat_dispatch_probes(void);      /* src/qmat.c    */
void mynah_kernels_dispatch_probes(void);   /* src/kernels.c */
void mynah_backend_dispatch_probes(void);   /* src/backend.c */
void mynah_threads_dispatch_probes(void);   /* src/threads.c */
void mynah_conv1d_dispatch_probes(void);    /* src/conv1d.c  */
void mynah_codec_dispatch_probes(void);     /* src/codec_nanocodec.c */

/* ------------------------------------------------------------------------
 * Report
 * ------------------------------------------------------------------------ */

/* Fill `rows` with the resolved table.  Returns the number written, or -1.
 * Collecting opens and closes the CPU/Metal/CUDA backends and runs the
 * model-free kernel self-tests, so it is not free — call it once. */
int mynah_dispatch_collect(mynah_dispatch_row *rows, int capacity);

/* Print the table.  `out_file` is a FILE* (NULL = stderr).  as_json != 0
 * writes the same rows as one JSON document instead of the human table.
 * Returns the number of rows whose `resolved` is UNKNOWN — i.e. the number of
 * predicates still missing — or -1 on error. */
int mynah_dispatch_report(void *out_file, int as_json);

/* Write the JSON form to `path`; "%d" in the path becomes the pid.  When
 * `path` is NULL the environment variable MYNAH_DISPATCH_JSON is used, and the
 * call is a no-op when that is unset too. */
int mynah_dispatch_report_json_path(const char *path);

/* The coarse class of the best matrix lever this host+binary pair can reach.
 * Keyed by expected-vs-observed tooling; deliberately coarse. */
const char *mynah_dispatch_isa_class(void);

/* Model-free check of the report machinery itself: the table is collectible,
 * ids are unique, every row has a source, a registered probe wins over the
 * gate, and the F16 drift canary agrees.  0 = ok, -1 = error. */
int mynah_dispatch_self_test(char *error, size_t error_capacity);

#ifdef __cplusplus
}
#endif
#endif /* MYNAH_TTS_DISPATCH_H */
