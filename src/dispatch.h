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

/* src/qmat.c — half weight cache: NEON converts on aarch64, F16C on x86.
 * Mirror of MYNAH_QMAT_F16 in src/qmat.c; the drift canary checks it. */
#if !defined(MYNAH_DISABLE_SIMD) && \
    ((defined(__aarch64__) && defined(__ARM_NEON)) || \
     ((defined(__x86_64__) || defined(__i386__)) && \
      (defined(__clang__) || (defined(__GNUC__) && __GNUC__ >= 11))))
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
/* NOT a macro any more: whether a bf16 or SVE kernel exists is a fact about
 * src/kernels.c and src/qmat.c, and mirroring it here is the drift this header
 * spends a paragraph warning about.  mynah_kernels_isa_kernels() (kernels.h)
 * is the single definition, and the isa.arm.{sve,sve2,svei8mm,svebf16,bf16}
 * rows resolve by calling it. */

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
/* BLAS=none is not "no GEMM": src/sgemm.c serves it.  The distinction matters
 * for the compiled column of any row whose fast path only needs SOME sgemm,
 * which is not the same set as "an external BLAS is linked". */
#if defined(MYNAH_USE_OWN_SGEMM)
#define MYNAH_DISPATCH_HAS_OWN_SGEMM 1
#else
#define MYNAH_DISPATCH_HAS_OWN_SGEMM 0
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
void mynah_seanet_dispatch_probes(void);    /* src/seanet.c  */
void mynah_sgemm_dispatch_probes(void);     /* src/sgemm.c   */

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

/* E4-12: does this CPU have every instruction set this binary may contain?
 *
 * Call it FIRST, before any allocation and before any model is opened.  It
 * returns 0 when the binary can run here, or -1 after writing a message that
 * names the instruction set the binary needs, the profile it was built with,
 * and the feature list this CPU reports -- which is the whole diagnosis that an
 * -mavx2 binary on a pre-Haswell host otherwise delivers as a bare SIGILL.
 *
 * It fires only on a DEFINITE absence.  A feature the process cannot probe
 * (no getauxval, a leaf the hypervisor hid, an emulator) is not an absence, and
 * refusing to start over our own ignorance would be a worse bug than the one
 * this prevents.  It allocates nothing and opens nothing.
 */
int mynah_dispatch_isa_guard(char *error, size_t error_capacity);

/* The coarse class of the best matrix lever this host+binary pair can reach.
 * Keyed by expected-vs-observed tooling; deliberately coarse. */
const char *mynah_dispatch_isa_class(void);

/* ========================================================================
 * THE SHAPE AND KERNEL CENSUS
 *
 * The dispatch table above says what this binary WOULD choose.  The census
 * says what it DID.  They are different questions and the gap between them is
 * where this project has lost time twice:
 *
 *   - 724d677 attributed an int8 result to "AVX-512 VNNI" on a build whose
 *     int8 dot was the AVX2 widen-then-madd pair.  The dispatch table now
 *     answers that one.
 *   - 2026-09: a banner announced a batched decoder while `pocket_proj_tile`
 *     fell through to a per-row loop for every group whose encoding the
 *     batched entry point could not honour.  The dispatch table CANNOT answer
 *     that one -- the feature was compiled, supported and resolved ON, and the
 *     code still took the other branch.  Only a count can tell you, and this
 *     is the count.
 *
 * So the unit of the census is not a feature, it is a CALL: one projection,
 * with its shape, the path it actually took, the encoding it actually carried,
 * and how many times.  "Did micro-batching engage" is answered by the `calls`
 * column of the rows whose path is `batched`, and by the existence of rows
 * whose path is `row-loop` -- which is the fallback wearing its own name
 * instead of hiding inside the batched row.
 *
 * WHERE THE HOOKS ARE.  At the four funnels in src/engine_pocket.c that every
 * PocketTTS projection passes through (`pocket_proj_row`, `_tile`, `_rows`,
 * `_batched`), never inside a kernel.  The engine is the only layer that knows
 * both the shape and the policy decision, which is exactly what has to be
 * recorded together.
 *
 * IT MUST NOT CHANGE WHAT IT MEASURES.  Same discipline as costmap.c: no
 * malloc anywhere, no atomic read-modify-write on the hot path, per-thread
 * blocks from a static array claimed with one relaxed fetch_add, merged only
 * at report time.  When off, a marker is one relaxed load and one predictable
 * branch.  The proof that this held is `make census-parity`, which runs the
 * same workload census-only and census+costmap and diffs the two censuses: the
 * same path ids, the same counts, the same fallbacks.  Without that diff the
 * numbers are unfalsifiable.
 *
 * ENV
 *   MYNAH_CENSUS=1          collect, and print the table on stderr at exit
 *   MYNAH_CENSUS_JSON=path  write JSON instead ("%d" becomes the pid)
 *   MYNAH_CENSUS_STRICT=1   exit non-zero when the census refuses (below)
 * ======================================================================== */

/* Which branch the projection actually took.  UNKNOWN is not a spare value:
 * it is what an unhooked or a future call site records, and its presence is
 * itself a refusal, because a census with a hole in it cannot support the
 * sentence "these are the kernels that ran". */
typedef enum {
    MYNAH_CENSUS_PATH_UNKNOWN = 0,
    MYNAH_CENSUS_PATH_MATVEC_F32,  /* mynah_matvec_bias_f32, one row         */
    MYNAH_CENSUS_PATH_MATVEC_Q,    /* mynah_qmat_linear_resolved_qt, one row */
    MYNAH_CENSUS_PATH_GEMM,        /* mynah_backend_matmul, whole tile       */
    MYNAH_CENSUS_PATH_BATCHED_Q,   /* mynah_qmat_linear_batched_qt           */
    MYNAH_CENSUS_PATH_ROWLOOP,     /* asked for >1 row, GOT a per-row loop   */
    MYNAH_CENSUS_PATH_COUNT
} mynah_census_path;

const char *mynah_census_path_name(int path);

/* 0 = off.  Read directly by the inline marker, as costmap's level is. */
extern int mynah_census_on_v;
void mynah_census_init(void);
int  mynah_census_enabled(void);

/* `block` and `kind` MUST be static strings: the table keys on the pointers,
 * so a new entry is formatted once and a repeat call is a pointer compare.
 * `rows` is the width the call site ASKED for; the path says what it got. */
void mynah_census_op_(const char *block, const char *kind, size_t k, size_t n,
                      size_t rows, int path, int qtype);

/* A MACRO, not an inline function, and that is not a style choice.
 *
 * C evaluates a call's arguments whether or not the body runs, so an inline
 * `if (on) record(...)` still pays for every argument EXPRESSION in a build
 * with the census off.  The qtype argument at these call sites is
 * mynah_qmat_qtype_resolved(mynah_qmat_cache_qtype(...)) -- two real calls, on
 * every projection, in the hot loop, for a census nobody asked for.  Measured:
 * that is a cost the CLEAN path was paying, which is the exact failure mode
 * this whole file is supposed to prevent.
 *
 * The macro short-circuits before the arguments exist.  Off, it is one relaxed
 * load of an int and one predictable branch, and nothing else is evaluated. */
#define MYNAH_CENSUS_OP(block, kind, k, n, rows, path, qtype)                  \
    do {                                                                       \
        if (mynah_census_on_v) {                                               \
            mynah_census_op_((block), (kind), (k), (n), (rows), (path),         \
                             (qtype));                                         \
        }                                                                      \
    } while (0)

/* Kept for callers that genuinely have no side-effecting arguments. */
static inline void mynah_census_op(const char *block, const char *kind, size_t k,
                                   size_t n, size_t rows, int path, int qtype) {
    if (mynah_census_on_v) mynah_census_op_(block, kind, k, n, rows, path, qtype);
}

/* One merged row. */
typedef struct {
    const char        *block;
    const char        *kind;
    unsigned long long k, n, rows;
    int                path;
    int                qtype;        /* -1 = f32 / not quantized            */
    unsigned long long calls;
    unsigned long long row_total;    /* rows summed over calls              */
    unsigned           threads_seen;
} mynah_census_row;

#define MYNAH_CENSUS_MAX_ROWS 256

int mynah_census_merge(mynah_census_row *out, int capacity);
void mynah_census_reset(void);

/* ---- allocations per request --------------------------------------------
 *
 * .work/linux-build-and-dispatch.md measured 2,786 allocations per request
 * (2,608 malloc + 178 calloc) with an LD_PRELOAD counter, and the decisive
 * part was not the number: it was that the number is CONSTANT across
 * --max-steps, which is the evidence that the autoregressive loop allocates
 * nothing (CLAUDE.md rule 4).  That property is worth a permanent check, so
 * `make alloc-constant-test` asserts it rather than trusting the comment.
 *
 * The counter itself lives OUTSIDE the runtime, in tests/alloc_shim.c, which
 * is preloaded and never linked.  The dependency is inverted on purpose: the
 * runtime exports a registration hook and the shim calls it from its own
 * constructor (dlsym(RTLD_DEFAULT, ...)), so the runtime needs no weak symbol,
 * no -ldl and no link-flag change, and above all never interposes an allocator
 * of its own.  With no shim the source is null and the census prints
 * [UNKNOWN], which is the honest answer and not a zero. */
void mynah_census_set_alloc_source(unsigned long long (*fn)(void));
unsigned long long mynah_census_alloc_count(void);

/* ---- the refusals -------------------------------------------------------
 *
 * .work/engineering-method.md 4: "the shape census exits non-zero if any
 * operation resolved UNKNOWN, or if a feature resolved ON whose class never
 * executed."  Plus the two this engine needs:
 *
 *   R1  an op recorded path=UNKNOWN
 *   R2  quant.requested resolved to an encoding, and NO op carried it
 *   R3  a kernel-class feature resolved ON and its class never executed
 *       (isa.arm.i8mm without a single batched int8 call; VNNI without a
 *       single int8 call)
 *   R4  the cost map was on and refused its own numbers
 *   R5  a per-request figure was asked for against a scope that does not
 *       match -- the census is process-wide and summed over threads, so
 *       dividing it by a request count the process does not have is the
 *       WORKER-by-HOST division the roofs tool refuses.  It prints
 *       [UNKNOWN] and, in strict mode, refuses.
 *
 * Returns 0 when the census may be read, or the number of refusals after
 * writing the first one into `reason`.
 *
 * It consults the dispatch table, and mynah_dispatch_collect() opens backends
 * and runs the kernel self-tests -- it is emphatically not free.  The table is
 * therefore collected ONCE per process and cached: calling this twice at exit
 * (the report, then the strict gate) was costing a full collect each time and
 * showed up as a third of the census's measured overhead. */
int mynah_census_refusals(char *reason, size_t capacity);

/* Print the census.  `out_file` is a FILE* (NULL = stderr); as_json != 0
 * writes JSON.  Returns the number of rows, or -1. */
int mynah_census_report(void *out_file, int as_json);

/* Model-free check of the census machinery: recording, merging, key identity,
 * the UNKNOWN refusal and that an off census records nothing.  0 = ok. */
int mynah_census_self_test(char *error, size_t error_capacity);

/* Model-free check of the report machinery itself: the table is collectible,
 * ids are unique, every row has a source, a registered probe wins over the
 * gate, and the F16 drift canary agrees.  0 = ok, -1 = error. */
int mynah_dispatch_self_test(char *error, size_t error_capacity);

#ifdef __cplusplus
}
#endif
#endif /* MYNAH_TTS_DISPATCH_H */
