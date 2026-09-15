/*
 * Causal SEANet decoder and Mimi up/downsample.  See seanet.h for the
 * streaming contract and for why the position counter is part of the state.
 *
 * Layout: channel major [channels][length], i.e. torch [1, C, T].
 */
#include "seanet.h"

#include <limits.h>
#include <math.h>
#include <stdarg.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "dispatch.h"
#include "kernels.h"
#include "sgemm.h"
#include "threads.h"

/* Which GEMM the two fast paths below call.  MYNAH_SEANET_BLAS says "a GEMM
 * exists", not "a vendor BLAS exists": with BLAS=none that GEMM is our own
 * (src/sgemm.c) and the fast paths are compiled exactly as before.  These two
 * call sites are the ENTIRE BLAS dependency of the PocketTTS production path
 * -- the backbone and the flow head already run our quantized kernels
 * (.work/no-blas.md §1). */
#if defined(MYNAH_USE_ACCELERATE)
#include <Accelerate/Accelerate.h>
#define MYNAH_SEANET_BLAS 1
#define MYNAH_SEANET_BLAS_NAME "Accelerate"
#elif defined(MYNAH_USE_OPENBLAS)
#include <cblas.h>
#define MYNAH_SEANET_BLAS 1
#define MYNAH_SEANET_BLAS_NAME "OpenBLAS"
#elif defined(MYNAH_USE_OWN_SGEMM)
#define MYNAH_SEANET_BLAS 1
#define MYNAH_SEANET_OWN_SGEMM 1
#define MYNAH_SEANET_BLAS_NAME "mynah-sgemm"
#else
#define MYNAH_SEANET_BLAS_NAME "none"
#endif

/* ------------------------------------------------------------- GEMM paths
 *
 * Measured on this file, not assumed: with the scalar loops below, the SEANet
 * decoder was 76% of one PocketTTS synthesis (MYNAH_COST_MAP=2,
 * codec.conv_stack 8570 ms of 11224 ms) and `sample` put 88% of the wall in
 * `mynah_dot_f32` + the two conv bodies.  The reason is structural: a decoder
 * convolution has kernel 1, 3 or 7, so the innermost reduction is 1-7 long and
 * every output element pays a call and a serial accumulator chain.
 *
 * Both fast paths below are the SAME arithmetic expressed as one GEMM per
 * kernel tap, which is what BLAS is for.  They are NOT bit-identical to the
 * scalar loops: the reduction over input channels is reassociated, so the two
 * agree to ~1e-6 relative, not exactly.  CLAUDE.md's numerical rule allows
 * that ("do not require byte-identical audio across different floating-point
 * orderings"); the parity gate against the oracle is what checks it, and the
 * scalar loop stays the reference implementation for every shape or build the
 * fast path refuses.
 *
 * A shape qualifies only when the B operand of the GEMM is a real matrix:
 *   conv1d          stride == 1 and groups == 1 (any kernel, any dilation)
 *   convtranspose   groups == 1
 * plus a BLAS build and a non-NULL tap buffer.  Everything else falls through.
 */

/* ------------------------------------------------------- dispatch counters
 *
 * E4-21.  Before these existed this file had twelve fallback paths and ZERO
 * rows in `--dispatch-map`: `grep seanet src/dispatch.c` returned nothing.
 * The one that matters is the build gate right above -- with `BLAS=scalar`
 * neither GEMM below is even compiled, the whole codec conv stack runs the
 * hand-written scalar loops, and that is the path that measured 8570 ms
 * against 237 ms for the GEMM path (.work/no-blas.md §2).  A 36x regression
 * that nothing in the binary announced.
 *
 * They are COUNTERS, not predicates about shapes, because a shape predicate
 * would have to lie: the report is built before a model is loaded, so at that
 * moment no SEANet convolution has run and nothing is known about the shapes
 * this pack will present.  Counting what actually executed lets the row say
 * "no SEANet conv has run in this process yet" instead of implying health --
 * .work/engineering-method.md §4, every tool declares a refusal.
 *
 * Relaxed atomics: one increment per convolution call (order tens per frame,
 * against a GEMM each), never per element, and never inside an inner loop. */
typedef struct {
    atomic_ullong conv_calls;
    atomic_ullong conv_gemm;
    atomic_ullong conv_scalar_stride;
    atomic_ullong conv_scalar_groups;
    atomic_ullong conv_scalar_taps;
    atomic_ullong conv_scalar_narrow;
    atomic_ullong conv_scalar_nogemm;
    atomic_ullong convtr_calls;
    atomic_ullong convtr_gemm;
    atomic_ullong convtr_scalar_groups;
    atomic_ullong convtr_scalar_taps;
    atomic_ullong convtr_scalar_narrow;
    atomic_ullong convtr_scalar_nogemm;
} sea_counters;

static sea_counters g_sea;

static void sea_bump(atomic_ullong *c) {
    atomic_fetch_add_explicit(c, 1ull, memory_order_relaxed);
}

static unsigned long long sea_read(atomic_ullong *c) {
    return atomic_load_explicit(c, memory_order_relaxed);
}

/* MYNAH_SEANET_GEMM: "0"/"off" forces the scalar reference loops on a build
 * that HAS a BLAS.  It narrows only -- it can never switch a GEMM on in a
 * build where one was not compiled -- which is the same contract
 * MYNAH_QMAT_VNNI and MYNAH_QMAT_F16C use in src/qmat.c, and for the same
 * reason: without it the scalar convolution reference is unreachable on every
 * machine anybody develops on, so nothing ever executes it. Resolved once and
 * then immutable: it describes the process, not a request. */
static int sea_gemm_enabled(void) {
#if defined(MYNAH_SEANET_BLAS)
    static int cached = -1;
    if (cached < 0) {
        const char *env = getenv("MYNAH_SEANET_GEMM");
        cached = (env != NULL && (strcmp(env, "0") == 0 ||
                                  strcmp(env, "off") == 0)) ? 0 : 1;
    }
    return cached;
#else
    return 0;
#endif
}

#if defined(MYNAH_SEANET_BLAS)
/* Every argument cblas_sgemm takes is an `int`, and all six of the ones below
 * come from `size_t` dimensions.  conv1d.c guards its narrowing
 * (conv1d.c:464); this file did not, and silently truncated six values per
 * call (E4-21, item H).  The check is made ONCE per convolution rather than
 * per tap, because every tap of one call has the same shape and because a
 * fast path must be refused before it starts accumulating, not halfway
 * through. */
static int sea_dims_fit(size_t m, size_t n, size_t k, size_t lda, size_t ldb,
                        size_t ldc) {
#if defined(MYNAH_SEANET_OWN_SGEMM)
    /* mynah_sgemm_f32 takes size_t, so there is nothing to narrow.  The check
     * stays in the dispatch chain (and keeps its counter) so that the report
     * reads the same on every build instead of a row quietly disappearing. */
    (void)m; (void)n; (void)k; (void)lda; (void)ldb; (void)ldc;
    return 1;
#else
    return m <= (size_t)INT_MAX && n <= (size_t)INT_MAX &&
           k <= (size_t)INT_MAX && lda <= (size_t)INT_MAX &&
           ldb <= (size_t)INT_MAX && ldc <= (size_t)INT_MAX;
#endif
}

static void sea_sgemm(int trans_a, size_t m, size_t n, size_t k,
                      const float *a, size_t lda, const float *b, size_t ldb,
                      float beta, float *c, size_t ldc) {
#if defined(MYNAH_SEANET_OWN_SGEMM)
    (void)mynah_sgemm_f32(trans_a, 0, m, n, k, 1.0f, a, lda, b, ldb, beta, c,
                          ldc);
#else
    cblas_sgemm(CblasRowMajor, trans_a ? CblasTrans : CblasNoTrans, CblasNoTrans,
                (int)m, (int)n, (int)k, 1.0f, a, (int)lda, b, (int)ldb, beta, c,
                (int)ldc);
#endif
}
#endif

/* ------------------------------------------------------------------ utils */

static void sea_set_error(char *error, size_t capacity, const char *format, ...)
    __attribute__((format(printf, 3, 4)));

static void sea_set_error(char *error, size_t capacity, const char *format,
                          ...) {
    if (error == NULL || capacity == 0) return;
    va_list args;
    va_start(args, format);
    (void)vsnprintf(error, capacity, format, args);
    va_end(args);
}

static int sea_mul(size_t a, size_t b, size_t *out) {
    if (a != 0 && b > (size_t)-1 / a) return -1;
    *out = a * b;
    return 0;
}

static int sea_add(size_t a, size_t b, size_t *out) {
    if (b > (size_t)-1 - a) return -1;
    *out = a + b;
    return 0;
}

static size_t sea_max(size_t a, size_t b) { return (a > b) ? a : b; }

/* ---------------------------------------------------------- phase profiler
 *
 * WHY.  codec.conv_stack scales 2.6x on sixteen cores where codec.transformer
 * gets 4.5x and step.backbone 4.7x, and a region timer cannot say why: it
 * measures the sum of a GEMM that has a thread pool behind it and a dozen
 * element-wise loops that do not.  Amdahl on the region total says "about a
 * third of this is serial" without naming one line.  These counters split the
 * region into phases whose parallelism is a property of the phase, so the
 * serial fraction is READ OFF instead of inferred.
 *
 * The split is by WHO RUNS THE LOOP, not by what the loop computes:
 *
 *   *.gemm      dispatches to mynah_parallel_for   -> scales
 *   everything else runs on the calling thread     -> does not
 *
 * so the table is a direct measurement of the ceiling.
 *
 * OFF unless MYNAH_SEANET_PROFILE is set to something other than "0": one
 * relaxed load of an int and a predictable branch per phase boundary, never
 * inside an inner loop.  Same level-0 contract as costmap.h.  Two clock reads
 * per phase, so the numbers are wall time on the calling thread and are NOT
 * summed over pool workers -- a `*.gemm` row is the caller's view of the
 * dispatch (wake-up, its own share of the work, and the barrier), which is
 * exactly the quantity that has to shrink for the region to scale. */
enum {
    SEA_PH_DECODE = 0,      /* whole mynah_seanet_decode: == codec.conv_stack */
    SEA_PH_ELU,             /* mynah_seanet_elu_f32                           */
    SEA_PH_RESADD,          /* the residual skip add                          */
    SEA_PH_CONV_WINDOW,     /* prepend the carried tail                       */
    SEA_PH_CONV_GATHER,     /* gather one kernel tap into a dense matrix      */
    SEA_PH_CONV_GEMM,       /* sea_sgemm, conv1d                              */
    SEA_PH_CONV_TAPS,       /* all taps fused into one pool region             */
    SEA_PH_CONV_BIAS,       /* broadcast the bias over the output             */
    SEA_PH_CONV_CARRY,      /* save the new tail                              */
    SEA_PH_CONV_SCALAR,     /* the reference conv1d loop                      */
    SEA_PH_CONVTR_FILL,     /* initialise `full` with the bias                */
    SEA_PH_CONVTR_GEMM,     /* sea_sgemm, convtranspose                       */
    SEA_PH_CONVTR_SCATTER,  /* taps -> full, stride-spaced                    */
    SEA_PH_CONVTR_SCALAR,   /* the reference scatter (grouped/depthwise)      */
    SEA_PH_CONVTR_TAIL,     /* fold the carried head, take the new tail       */
    SEA_PH_CONVTR_COPY,     /* full -> output                                 */
    SEA_PH_COUNT
};

static const char *const g_sea_phase_name[SEA_PH_COUNT] = {
    "decode.total",   "elu",            "residual_add",  "conv.window",
    "conv.gather",    "conv.gemm",      "conv.taps",     "conv.bias",
    "conv.carry",     "conv.scalar",    "convtr.fill",   "convtr.gemm",
    "convtr.scatter", "convtr.scalar",  "convtr.tail",   "convtr.copy"
};

/* 1 when the phase dispatches to the thread pool; 0 when it is a plain loop on
 * the calling thread.  The report prints this column because the whole point
 * of the table is that split, and a reader should not have to know which name
 * means which. */
static const int g_sea_phase_par[SEA_PH_COUNT] = {
    0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0
};

typedef struct {
    atomic_ullong ns;
    atomic_ullong calls;
    atomic_ullong elems;
} sea_phase_acc;

static sea_phase_acc g_sea_phase[SEA_PH_COUNT];
static atomic_int g_sea_prof_state = -1; /* -1 unresolved, 0 off, 1 on */
static atomic_int g_sea_prof_hooked;

static void sea_prof_report(void);

static int sea_prof_on(void) {
    int state = atomic_load_explicit(&g_sea_prof_state, memory_order_relaxed);
    if (state >= 0) return state;
    const char *env = getenv("MYNAH_SEANET_PROFILE");
    state = (env != NULL && env[0] != '\0' && strcmp(env, "0") != 0) ? 1 : 0;
    atomic_store_explicit(&g_sea_prof_state, state, memory_order_relaxed);
    if (state) {
        int expected = 0;
        if (atomic_compare_exchange_strong(&g_sea_prof_hooked, &expected, 1)) {
            (void)atexit(sea_prof_report);
        }
    }
    return state;
}

/* 0 means "not profiling": the caller skips the matching _add.  CLOCK_MONOTONIC
 * is time since boot and is never 0 on a running process, so the sentinel
 * cannot collide with a real reading. */
static unsigned long long sea_prof_now(void) {
    if (!sea_prof_on()) return 0ull;
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (unsigned long long)ts.tv_sec * 1000000000ull +
           (unsigned long long)ts.tv_nsec;
}

static void sea_prof_add(int phase, unsigned long long start, size_t elems) {
    if (start == 0ull) return;
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    const unsigned long long now =
        (unsigned long long)ts.tv_sec * 1000000000ull +
        (unsigned long long)ts.tv_nsec;
    sea_phase_acc *acc = &g_sea_phase[phase];
    atomic_fetch_add_explicit(&acc->ns, now - start, memory_order_relaxed);
    atomic_fetch_add_explicit(&acc->calls, 1ull, memory_order_relaxed);
    atomic_fetch_add_explicit(&acc->elems, (unsigned long long)elems,
                              memory_order_relaxed);
}

static void sea_prof_report(void) {
    const unsigned long long total = atomic_load_explicit(
        &g_sea_phase[SEA_PH_DECODE].ns, memory_order_relaxed);
    fprintf(stderr,
            "[SEANET-PHASES] decode.total is codec.conv_stack; every other row "
            "is a phase inside it.\n"
            "  par=1 dispatches to the pool, par=0 runs on the calling "
            "thread.  The par=0 rows are the scaling ceiling.\n");
    fprintf(stderr, "  %-16s %3s %10s %12s %14s %8s\n", "phase", "par", "calls",
            "ms", "elements", "%decode");
    unsigned long long serial = 0, parallel = 0;
    for (int i = 0; i < SEA_PH_COUNT; ++i) {
        const unsigned long long ns =
            atomic_load_explicit(&g_sea_phase[i].ns, memory_order_relaxed);
        const unsigned long long calls =
            atomic_load_explicit(&g_sea_phase[i].calls, memory_order_relaxed);
        if (calls == 0ull) continue;
        const unsigned long long el =
            atomic_load_explicit(&g_sea_phase[i].elems, memory_order_relaxed);
        if (i != SEA_PH_DECODE) {
            if (g_sea_phase_par[i]) parallel += ns; else serial += ns;
        }
        fprintf(stderr, "  %-16s %3d %10llu %12.3f %14llu %7.2f%%\n",
                g_sea_phase_name[i], g_sea_phase_par[i], calls,
                (double)ns / 1e6, el,
                total ? 100.0 * (double)ns / (double)total : 0.0);
    }
    const double t = (double)(total ? total : 1ull);
    fprintf(stderr,
            "  sum: parallel %.3f ms (%.2f%%), serial %.3f ms (%.2f%%), "
            "unattributed %.3f ms (%.2f%%)\n",
            (double)parallel / 1e6, 100.0 * (double)parallel / t,
            (double)serial / 1e6, 100.0 * (double)serial / t,
            ((double)total - (double)parallel - (double)serial) / 1e6,
            100.0 * ((double)total - (double)parallel - (double)serial) / t);
}

/* ELU is element-wise, so splitting it over the pool is bit-identical by
 * construction -- threads.h's contract is "tasks write to disjoint regions,
 * the result is BIT-IDENTICAL to the serial loop" -- and there is no reduction
 * here to reassociate.  It needed splitting: measured at sixteen threads it
 * was 77.0 ms of the 234.7 ms codec.conv_stack (32.8%) and its speedup over
 * one thread was 1.00x, because mynah_seanet_decode calls it ten times per
 * frame on the calling thread and nothing ever dispatched it.
 *
 * THE THRESHOLDS ARE MEASURED, not chosen.  On the 32-core Neoverse-V2 box a
 * pool region costs ~35 us of wake-up and barrier (fitted from the per-shape
 * GEMM histogram: t16 - t1/16 over eleven shapes) and this loop runs at
 * ~2.5 ns per element, so a region only starts to pay at ~16k elements and a
 * task below ~8k elements is barrier, not work.  Both numbers belong to this
 * box; they do not port.
 *
 * CHUNKS ARE WHOLE CACHE LINES.  Sixteen floats is one 64-byte line; the
 * round-up to 64 floats keeps every task's stores four lines clear of its
 * neighbour's and keeps each chunk a multiple of any vector width the compiler
 * might use, so no element moves between a vector body and a scalar
 * remainder. */
#define SEA_ELU_MIN_PARALLEL 16384u
#define SEA_ELU_MIN_CHUNK     8192u

typedef struct {
    const float *in;
    float *out;
    float alpha;
    size_t n;
    size_t chunk;
} sea_elu_job;

/* ------------------------------------------------------- exp on (-inf, 0]
 *
 * ELU needs exp only where x is negative, and that is a far easier function
 * than exp in general: the result lives in (0, 1], there is nothing to
 * overflow, and everything below about -88 has already flushed to zero.
 *
 * The method is the standard range reduction.  n = round(x * log2(e)), then
 * r = x - n*ln2 with ln2 split into a high part that is exact in binary32 and
 * a low correction, so the subtraction loses nothing; r then lies in
 * [-ln2/2, ln2/2] and a degree-5 polynomial covers it.  2^n is built directly
 * into the exponent field rather than called for.
 *
 * ACCURACY IS ASSERTED, NOT ASSUMED.  sea_exp_self_test() below sweeps
 * [-88, 0] against libm and fails above 1e-7 ABSOLUTE.  Measured: max absolute
 * error 5.96e-08 on exp, and the same on ELU itself at alpha = 1 -- about half
 * an ulp of binary32 near one.
 *
 * Absolute rather than relative on purpose.  Below about -87 the true value is
 * subnormal (expf(-88) is 6e-39) and this flushes it to zero, so the relative
 * error there is 1.0 and means nothing: ELU returns -alpha either way.  A
 * relative bound would have to carve out that tail; an absolute one is the
 * quantity that actually reaches the waveform.
 *
 * This follows the precedent src/kernels.c set for tanh (E4-16d): a Padé we
 * own, on both ISAs, with a stated bound -- rather than a libm call per
 * element that cannot be vectorised.  The scalar version here IS the
 * reference, and the two vector versions are checked against it. */
#define SEA_EXP_LO (-88.0f)   /* expf underflows to zero below this */

static float sea_exp_neg_scalar(float x) {
    if (x <= SEA_EXP_LO) return 0.0f;
    const float log2e = 1.44269504088896340736f;
    const float ln2_hi = 0.693359375f;        /* exact in binary32 */
    const float ln2_lo = -2.12194440e-4f;
    /* Round to nearest, ties to even -- the same rule vrndnq_f32 and
     * _mm256_round_ps(NEAREST_INT) apply, so the reference and the two vector
     * paths reduce the argument identically.  An earlier version open-coded
     * this and got it wrong: the absolute error was 1.9e-05, which is what a
     * bad range reduction looks like. */
    const float nf = rintf(x * log2e);
    const int n = (int)nf;
    const float r = (x - nf * ln2_hi) - nf * ln2_lo;
    /* exp(r) on [-ln2/2, ln2/2], Horner. */
    float p = 1.98756912e-4f;
    p = p * r + 1.39819618e-3f;
    p = p * r + 8.33345973e-3f;
    p = p * r + 4.16666418e-2f;
    p = p * r + 1.66666657e-1f;
    p = p * r + 5.00000000e-1f;
    p = p * r + 1.0f;
    p = p * r + 1.0f;
    /* 2^n by construction; n is in [-127, 0] here. */
    union { uint32_t u; float f; } scale;
    int e = n + 127;
    if (e < 1) return 0.0f;
    scale.u = (uint32_t)e << 23;
    return p * scale.f;
}

static void sea_elu_range(const float *in, float *out, size_t n, float alpha) {
    size_t i = 0;
#if defined(__aarch64__) || defined(__ARM_NEON)
    {
        const float32x4_t log2e = vdupq_n_f32(1.44269504088896340736f);
        const float32x4_t ln2_hi = vdupq_n_f32(0.693359375f);
        const float32x4_t ln2_lo = vdupq_n_f32(-2.12194440e-4f);
        const float32x4_t zero = vdupq_n_f32(0.0f);
        const float32x4_t one = vdupq_n_f32(1.0f);
        const float32x4_t va = vdupq_n_f32(alpha);
        const float32x4_t lo = vdupq_n_f32(SEA_EXP_LO);
        for (; i + 4u <= n; i += 4u) {
            const float32x4_t x = vld1q_f32(in + i);
            /* Only the negative lane's exp is used, so clamp the argument into
             * the range this approximation covers and let the select discard
             * the rest.  Nothing here can overflow. */
            const float32x4_t xn = vmaxq_f32(vminq_f32(x, zero), lo);
            const float32x4_t fn = vmulq_f32(xn, log2e);
            const float32x4_t nf = vrndnq_f32(fn);
            float32x4_t r = vsubq_f32(xn, vmulq_f32(nf, ln2_hi));
            r = vsubq_f32(r, vmulq_f32(nf, ln2_lo));
            float32x4_t p = vdupq_n_f32(1.98756912e-4f);
            p = vfmaq_f32(vdupq_n_f32(1.39819618e-3f), p, r);
            p = vfmaq_f32(vdupq_n_f32(8.33345973e-3f), p, r);
            p = vfmaq_f32(vdupq_n_f32(4.16666418e-2f), p, r);
            p = vfmaq_f32(vdupq_n_f32(1.66666657e-1f), p, r);
            p = vfmaq_f32(vdupq_n_f32(5.00000000e-1f), p, r);
            p = vfmaq_f32(one, p, r);
            p = vfmaq_f32(one, p, r);
            const int32x4_t e = vaddq_s32(vcvtq_s32_f32(nf), vdupq_n_s32(127));
            const int32x4_t ec = vmaxq_s32(e, vdupq_n_s32(0));
            const float32x4_t sc = vreinterpretq_f32_s32(vshlq_n_s32(ec, 23));
            const float32x4_t ex = vmulq_f32(p, sc);
            const float32x4_t neg = vmulq_f32(va, vsubq_f32(ex, one));
            vst1q_f32(out + i, vbslq_f32(vcgtq_f32(x, zero), x, neg));
        }
    }
#endif
    for (; i < n; ++i) {
        const float x = in[i];
        out[i] = (x > 0.0f) ? x : alpha * (sea_exp_neg_scalar(x) - 1.0f);
    }
}

static void sea_elu_task(void *ctx, int index) {
    const sea_elu_job *j = (const sea_elu_job *)ctx;
    const size_t start = (size_t)index * j->chunk;
    if (start >= j->n) return;
    size_t len = j->chunk;
    if (start + len > j->n) len = j->n - start;
    sea_elu_range(j->in + start, j->out + start, len, j->alpha);
}

void mynah_seanet_elu_f32(const float *input, float *output, size_t n,
                          float alpha) {
    if (input == NULL || output == NULL) return;
    const unsigned long long t0 = sea_prof_now();
    const int threads = mynah_num_threads();
    if (threads > 1 && n >= SEA_ELU_MIN_PARALLEL) {
        size_t tasks = n / SEA_ELU_MIN_CHUNK;
        if (tasks > (size_t)threads) tasks = (size_t)threads;
        if (tasks > 1u) {
            size_t chunk = (n + tasks - 1u) / tasks;
            chunk += (64u - chunk % 64u) % 64u;
            tasks = (n + chunk - 1u) / chunk;
            sea_elu_job job;
            job.in = input; job.out = output; job.alpha = alpha;
            job.n = n; job.chunk = chunk;
            mynah_parallel_for((int)tasks, sea_elu_task, &job);
            sea_prof_add(SEA_PH_ELU, t0, n);
            return;
        }
    }
    sea_elu_range(input, output, n, alpha);
    sea_prof_add(SEA_PH_ELU, t0, n);
}

/* ------------------------------------------------ causal streaming conv1d */

static int conv_effective_kernel(const mynah_conv1d_spec *spec, size_t *out) {
    size_t span = 0;
    if (sea_mul(spec->kernel_size - 1u, spec->dilation, &span) != 0) return -1;
    return sea_add(span, 1u, out);
}

static int conv_validate(const mynah_conv1d_spec *spec, size_t *effective,
                         char *error, size_t error_capacity) {
    if (spec == NULL) {
        sea_set_error(error, error_capacity, "conv1d: null spec");
        return -1;
    }
    if (spec->in_channels == 0 || spec->out_channels == 0 ||
        spec->kernel_size == 0 || spec->stride == 0 || spec->dilation == 0 ||
        spec->groups == 0) {
        sea_set_error(error, error_capacity, "conv1d: zero-valued dimension");
        return -1;
    }
    if ((spec->in_channels % spec->groups) != 0 ||
        (spec->out_channels % spec->groups) != 0) {
        sea_set_error(error, error_capacity,
                      "conv1d: channels %zu/%zu not divisible by groups %zu",
                      spec->in_channels, spec->out_channels, spec->groups);
        return -1;
    }
    if (conv_effective_kernel(spec, effective) != 0) {
        sea_set_error(error, error_capacity, "conv1d: kernel span overflow");
        return -1;
    }
    if (*effective < spec->stride) {
        sea_set_error(error, error_capacity,
                      "conv1d: effective kernel %zu < stride %zu", *effective,
                      spec->stride);
        return -1;
    }
    return 0;
}

/* Floats the GEMM fast path needs on top of the ring buffers: one kernel tap
 * of the weight, gathered dense.  Zero when the shape does not qualify, so a
 * grouped or strided conv costs exactly what it did before. */
/* ------------------------------------------------- the tap permutation memo
 *
 * `out[oc][n] = sum_k sum_j W[oc][j][k] * win[j][n + k*d]` is run as one GEMM
 * per tap k, and for a fixed k the left operand is W[.][.][k] -- a slice of
 * the weight strided by `kernel`.  Gathering it into a dense
 * [out_channels][in_channels] matrix is what `conv->taps` was for, and it was
 * being redone on EVERY FRAME for a weight that never changes: measured at
 * 1,964,224 floats per frame, 1.43 GB per request, 31.7% of the conv stack.
 *
 * The permutation is the whole of it.  It is NOT arithmetic -- `dst[j] =
 * src[j*kernel]` is a copy -- so hoisting it is byte-identical on every build,
 * which is why this memo sits outside the `MYNAH_SEANET_OWN_SGEMM` guard that
 * (correctly) protects the FUSED path.  That guard exists because routing the
 * taps through our own GEMM kernels would change the codec's f32 output on a
 * build that links Accelerate or OpenBLAS.  Moving a copy earlier changes
 * nothing at all, and the guard was covering both.
 *
 * SHARED, not per-state.  The permuted copy depends only on the weight
 * pointer, and weights are immutable and mmapped for the life of the process,
 * while a `mynah_seanet_state` is per request: a per-state copy would cost
 * ~6.5 MB times max_batch times workers, which is the wrong trade against a
 * one-off 6.5 MB here.  Keyed on the weight pointer because that is what
 * identifies the tensor -- this module never sees a tensor name, by design.
 *
 * Freed at exit rather than never: `make leaks` is a gate on this project and
 * a cache that cannot be reclaimed is a leak with a good excuse. */
typedef struct {
    const float *weight;
    size_t in_channels, out_channels, kernel;
    float *permuted;        /* [kernel][out_channels][in_channels] */
} sea_tap_entry;

static struct {
    sea_tap_entry *entries;
    size_t count, capacity;
    pthread_mutex_t mutex;
    int atexit_registered;
} g_sea_taps = { NULL, 0, 0, PTHREAD_MUTEX_INITIALIZER, 0 };

static void sea_taps_release(void) {
    pthread_mutex_lock(&g_sea_taps.mutex);
    for (size_t i = 0; i < g_sea_taps.count; ++i) free(g_sea_taps.entries[i].permuted);
    free(g_sea_taps.entries);
    g_sea_taps.entries = NULL;
    g_sea_taps.count = g_sea_taps.capacity = 0;
    pthread_mutex_unlock(&g_sea_taps.mutex);
}

/* Returns [kernel][out_channels][in_channels], or NULL -- in which case the
 * caller gathers into its own scratch exactly as before.  A NULL here is a
 * slower frame and never a different sample. */
static const float *sea_taps_all(const float *weight, size_t in_channels,
                                 size_t out_channels, size_t kernel) {
    if (weight == NULL || in_channels == 0u || out_channels == 0u ||
        kernel <= 1u) return NULL;
    pthread_mutex_lock(&g_sea_taps.mutex);
    for (size_t i = 0; i < g_sea_taps.count; ++i) {
        const sea_tap_entry *e = &g_sea_taps.entries[i];
        if (e->weight == weight && e->in_channels == in_channels &&
            e->out_channels == out_channels && e->kernel == kernel) {
            const float *p = e->permuted;
            pthread_mutex_unlock(&g_sea_taps.mutex);
            return p;
        }
    }
    if (g_sea_taps.count == g_sea_taps.capacity) {
        const size_t cap = (g_sea_taps.capacity == 0u) ? 16u : g_sea_taps.capacity * 2u;
        sea_tap_entry *grown = (sea_tap_entry *)realloc(g_sea_taps.entries,
                                                        cap * sizeof(*grown));
        if (grown == NULL) { pthread_mutex_unlock(&g_sea_taps.mutex); return NULL; }
        g_sea_taps.entries = grown;
        g_sea_taps.capacity = cap;
    }
    /* Checked, because these three come from a model file. */
    size_t floats = 0u;
    if (out_channels > SIZE_MAX / in_channels) goto refuse;
    floats = out_channels * in_channels;
    if (floats > SIZE_MAX / kernel) goto refuse;
    floats *= kernel;
    if (floats > SIZE_MAX / sizeof(float)) goto refuse;
    float *permuted = (float *)malloc(floats * sizeof(float));
    if (permuted == NULL) { pthread_mutex_unlock(&g_sea_taps.mutex); return NULL; }
    for (size_t k = 0; k < kernel; ++k) {
        float *dst_k = permuted + k * out_channels * in_channels;
        for (size_t oc = 0; oc < out_channels; ++oc) {
            const float *src = weight + oc * in_channels * kernel + k;
            float *dst = dst_k + oc * in_channels;
            for (size_t j = 0; j < in_channels; ++j) dst[j] = src[j * kernel];
        }
    }
    g_sea_taps.entries[g_sea_taps.count++] = (sea_tap_entry){
        weight, in_channels, out_channels, kernel, permuted };
    if (!g_sea_taps.atexit_registered) {
        g_sea_taps.atexit_registered = 1;
        atexit(sea_taps_release);
    }
    pthread_mutex_unlock(&g_sea_taps.mutex);
    return permuted;
refuse:
    pthread_mutex_unlock(&g_sea_taps.mutex);
    return NULL;
}

static size_t conv_taps_floats(const mynah_conv1d_spec *spec) {
    if (spec->stride != 1u || spec->groups != 1u || spec->kernel_size <= 1u) {
        return 0;
    }
    size_t taps = 0;
    if (sea_mul(spec->out_channels, spec->in_channels, &taps) != 0) return 0;
    return taps;
}

size_t mynah_causal_conv1d_scratch(const mynah_conv1d_spec *spec,
                                   size_t max_in_len) {
    size_t effective = 0;
    if (conv_validate(spec, &effective, NULL, 0) != 0) return 0;
    const size_t tail = effective - spec->stride;
    const size_t taps = conv_taps_floats(spec);
    if (tail == 0) return taps;
    size_t previous = 0;
    size_t window = 0;
    size_t span = 0;
    if (sea_mul(spec->in_channels, tail, &previous) != 0) return 0;
    if (sea_add(tail, max_in_len, &span) != 0) return 0;
    if (sea_mul(spec->in_channels, span, &window) != 0) return 0;
    size_t total = 0;
    if (sea_add(previous, window, &total) != 0) return 0;
    if (sea_add(total, taps, &total) != 0) return 0;
    return total;
}

int mynah_causal_conv1d_init(mynah_causal_conv1d *conv,
                             const mynah_conv1d_spec *spec, size_t max_in_len,
                             float *scratch, size_t scratch_floats, char *error,
                             size_t error_capacity) {
    if (conv == NULL) {
        sea_set_error(error, error_capacity, "conv1d: null object");
        return -1;
    }
    size_t effective = 0;
    if (conv_validate(spec, &effective, error, error_capacity) != 0) return -1;
    if (max_in_len == 0 || (max_in_len % spec->stride) != 0) {
        sea_set_error(error, error_capacity,
                      "conv1d: max_in_len %zu must be a positive multiple of "
                      "stride %zu",
                      max_in_len, spec->stride);
        return -1;
    }
    memset(conv, 0, sizeof(*conv));
    conv->spec = *spec;
    conv->tail = effective - spec->stride;
    conv->max_in_len = max_in_len;
    const size_t needed = mynah_causal_conv1d_scratch(spec, max_in_len);
    if (needed > scratch_floats) {
        sea_set_error(error, error_capacity,
                      "conv1d: scratch too small (%zu < %zu)", scratch_floats,
                      needed);
        return -1;
    }
    size_t used = 0;
    if (conv->tail > 0) {
        if (scratch == NULL) {
            sea_set_error(error, error_capacity, "conv1d: null scratch");
            return -1;
        }
        conv->previous = scratch;
        conv->window = scratch + spec->in_channels * conv->tail;
        used = spec->in_channels * conv->tail +
               spec->in_channels * (conv->tail + max_in_len);
    }
    if (conv_taps_floats(spec) > 0 && scratch != NULL) conv->taps = scratch + used;
    mynah_causal_conv1d_reset(conv);
    return 0;
}

void mynah_causal_conv1d_reset(mynah_causal_conv1d *conv) {
    if (conv == NULL) return;
    if (conv->previous != NULL && conv->tail > 0) {
        memset(conv->previous, 0,
               conv->spec.in_channels * conv->tail * sizeof(float));
    }
    conv->primed = 0;
}

int mynah_causal_conv1d_apply(mynah_causal_conv1d *conv,
                              const mynah_conv_weights *weights,
                              const float *input, size_t in_len,
                              float *output) {
    if (conv == NULL || weights == NULL || weights->weight == NULL ||
        input == NULL || output == NULL) {
        return -1;
    }
    const mynah_conv1d_spec *spec = &conv->spec;
    if (in_len == 0 || in_len > conv->max_in_len ||
        (in_len % spec->stride) != 0) {
        return -1;
    }
    const size_t tail = conv->tail;
    if (spec->pad_mode == MYNAH_CONV_PAD_REPLICATE && tail > 0 &&
        in_len < tail) {
        /* Upstream asserts the same thing: replicate padding needs at least a
         * full tail of content to keep the ring buffer meaningful. */
        return -1;
    }

    const size_t in_channels = spec->in_channels;
    const float *win = input;
    size_t window_len = in_len;

    if (tail > 0) {
        const unsigned long long t_win = sea_prof_now();
        if (spec->pad_mode == MYNAH_CONV_PAD_REPLICATE && !conv->primed) {
            for (size_t c = 0; c < in_channels; ++c) {
                const float first = input[c * in_len];
                float *dst = conv->previous + c * tail;
                for (size_t i = 0; i < tail; ++i) dst[i] = first;
            }
        }
        window_len = tail + in_len;
        for (size_t c = 0; c < in_channels; ++c) {
            float *dst = conv->window + c * window_len;
            memcpy(dst, conv->previous + c * tail, tail * sizeof(float));
            memcpy(dst + tail, input + c * in_len, in_len * sizeof(float));
        }
        win = conv->window;
        sea_prof_add(SEA_PH_CONV_WINDOW, t_win, in_channels * window_len);
    }

    const size_t kernel = spec->kernel_size;
    const size_t dilation = spec->dilation;
    const size_t stride = spec->stride;
    const size_t groups = spec->groups;
    const size_t in_per_group = in_channels / groups;
    const size_t out_per_group = spec->out_channels / groups;
    const size_t out_len = in_len / stride;

    /* Which of the conv1d paths this call takes, counted for the dispatch
     * report.  Exactly one counter fires per call, so the rows add up and a
     * reader can tell "the fast path never qualified" from "the fast path was
     * never compiled".  See the sea_counters comment. */
    sea_bump(&g_sea.conv_calls);
    int conv_use_gemm = 0;
#if !defined(MYNAH_SEANET_BLAS)
    (void)conv_use_gemm;
    sea_bump(&g_sea.conv_scalar_nogemm);
#else
    if (!sea_gemm_enabled()) {
        sea_bump(&g_sea.conv_scalar_nogemm);
    } else if (spec->stride != 1u) {
        sea_bump(&g_sea.conv_scalar_stride);
    } else if (groups != 1u) {
        sea_bump(&g_sea.conv_scalar_groups);
    } else if (kernel != 1u && conv->taps == NULL) {
        sea_bump(&g_sea.conv_scalar_taps);
    } else if (!sea_dims_fit(spec->out_channels, out_len, in_channels,
                             in_channels, window_len, out_len)) {
        sea_bump(&g_sea.conv_scalar_narrow);
    } else {
        conv_use_gemm = 1;
    }
#endif

#if defined(MYNAH_SEANET_BLAS)
    /* One GEMM per kernel tap, accumulating into the output:
     *   out[oc][n] = sum_k sum_j W[oc][j][k] * win[j][n + k*dilation]
     * For a fixed k the right operand is win[0:C][k*d : k*d + out_len], a real
     * row-major submatrix (ldb = window_len), which is why stride must be 1:
     * a stride > 1 would space the columns and there would be no matrix to
     * hand to BLAS.  `taps` gathers W[.][.][k], whose natural layout is
     * strided by `kernel`; for kernel == 1 the weight already is the dense
     * matrix and no gather happens at all. */
    if (conv_use_gemm) {
        sea_bump(&g_sea.conv_gemm);
        const size_t oc_count = spec->out_channels;
        /* All `kernel` taps in ONE pool region when the plan allows it: the
         * gather lands on the thread that is about to multiply by it, and the
         * dispatch is paid once instead of `kernel` times.  Byte-identical to
         * the loop below -- see the contract in sgemm.h -- and it refuses
         * rather than approximates, in which case the loop runs unchanged. */
        int fused = 0;
#if defined(MYNAH_SEANET_OWN_SGEMM)
        /* ONLY when sea_sgemm above is mynah_sgemm_f32.  On a build that links
         * Accelerate or OpenBLAS, sea_sgemm is THAT library's cblas_sgemm, and
         * routing these taps through our own kernels instead would change the
         * codec's f32 output -- a numerical change wearing a scheduling
         * change's clothes, on the one platform (macOS) where nobody would be
         * looking for it.  Production is Linux with BLAS=none, which is
         * exactly where this is compiled in. */
        if (kernel > 1u) {
            const unsigned long long t_fused = sea_prof_now();
            if (mynah_sgemm_f32_conv_taps(oc_count, out_len, in_channels,
                                          kernel, weights->weight, conv->taps,
                                          win, window_len, dilation, 0.0f,
                                          output, out_len) == 0) {
                fused = 1;
                sea_prof_add(SEA_PH_CONV_TAPS, t_fused,
                             oc_count * out_len * in_channels * kernel);
            }
        }
#endif
        /* Built once for this weight, on any build; NULL falls back to the
         * per-frame gather below, which is the same bytes either way. */
        const float *all_taps =
            fused ? NULL : sea_taps_all(weights->weight, in_channels, oc_count, kernel);
        for (size_t k = 0; !fused && k < kernel; ++k) {
            const float *a;
            if (kernel == 1u) {
                a = weights->weight;
            } else if (all_taps != NULL) {
                a = all_taps + k * oc_count * in_channels;
            } else {
                const unsigned long long t_gather = sea_prof_now();
                float *taps = conv->taps;
                for (size_t oc = 0; oc < oc_count; ++oc) {
                    const float *src = weights->weight + oc * in_channels * kernel + k;
                    float *dst = taps + oc * in_channels;
                    for (size_t j = 0; j < in_channels; ++j) dst[j] = src[j * kernel];
                }
                a = taps;
                sea_prof_add(SEA_PH_CONV_GATHER, t_gather,
                             oc_count * in_channels);
            }
            const unsigned long long t_gemm = sea_prof_now();
            sea_sgemm(0, oc_count, out_len, in_channels, a, in_channels,
                      win + k * dilation, window_len, (k == 0) ? 0.0f : 1.0f,
                      output, out_len);
            sea_prof_add(SEA_PH_CONV_GEMM, t_gemm,
                         oc_count * out_len * in_channels);
        }
        if (weights->bias != NULL) {
            const unsigned long long t_bias = sea_prof_now();
            for (size_t oc = 0; oc < oc_count; ++oc) {
                float *out_row = output + oc * out_len;
                const float bias = weights->bias[oc];
                for (size_t n = 0; n < out_len; ++n) out_row[n] += bias;
            }
            sea_prof_add(SEA_PH_CONV_BIAS, t_bias, oc_count * out_len);
        }
        if (tail > 0) {
            const unsigned long long t_carry = sea_prof_now();
            for (size_t c = 0; c < in_channels; ++c) {
                memcpy(conv->previous + c * tail,
                       conv->window + c * window_len + (window_len - tail),
                       tail * sizeof(float));
            }
            conv->primed = 1;
            sea_prof_add(SEA_PH_CONV_CARRY, t_carry, in_channels * tail);
        }
        return 0;
    }
#endif

    const unsigned long long t_scalar = sea_prof_now();
    for (size_t oc = 0; oc < spec->out_channels; ++oc) {
        const size_t group = oc / out_per_group;
        const float *weight_row = weights->weight + oc * in_per_group * kernel;
        float *out_row = output + oc * out_len;
        const float bias = (weights->bias != NULL) ? weights->bias[oc] : 0.0f;
        for (size_t n = 0; n < out_len; ++n) {
            const size_t base = n * stride;
            float acc = 0.0f;
            for (size_t j = 0; j < in_per_group; ++j) {
                const float *w = weight_row + j * kernel;
                const float *x =
                    win + (group * in_per_group + j) * window_len + base;
                if (dilation == 1u) {
                    acc += mynah_dot_f32(w, x, kernel);
                } else {
                    for (size_t k = 0; k < kernel; ++k) {
                        acc += w[k] * x[k * dilation];
                    }
                }
            }
            out_row[n] = acc + bias;
        }
    }
    sea_prof_add(SEA_PH_CONV_SCALAR, t_scalar,
                 spec->out_channels * out_len * in_per_group * kernel);

    if (tail > 0) {
        /* previous := last `tail` samples of the concatenated window. */
        const unsigned long long t_carry = sea_prof_now();
        for (size_t c = 0; c < in_channels; ++c) {
            memcpy(conv->previous + c * tail,
                   conv->window + c * window_len + (window_len - tail),
                   tail * sizeof(float));
        }
        conv->primed = 1;
        sea_prof_add(SEA_PH_CONV_CARRY, t_carry, in_channels * tail);
    }
    return 0;
}

/* --------------------------------------- causal streaming convtranspose1d */

static int convtr_validate(const mynah_convtr1d_spec *spec, char *error,
                           size_t error_capacity) {
    if (spec == NULL) {
        sea_set_error(error, error_capacity, "convtr1d: null spec");
        return -1;
    }
    if (spec->in_channels == 0 || spec->out_channels == 0 ||
        spec->kernel_size == 0 || spec->stride == 0 || spec->groups == 0) {
        sea_set_error(error, error_capacity, "convtr1d: zero-valued dimension");
        return -1;
    }
    if ((spec->in_channels % spec->groups) != 0 ||
        (spec->out_channels % spec->groups) != 0) {
        sea_set_error(error, error_capacity,
                      "convtr1d: channels %zu/%zu not divisible by groups %zu",
                      spec->in_channels, spec->out_channels, spec->groups);
        return -1;
    }
    if (spec->kernel_size < spec->stride) {
        sea_set_error(error, error_capacity,
                      "convtr1d: kernel %zu < stride %zu", spec->kernel_size,
                      spec->stride);
        return -1;
    }
    return 0;
}

static int convtr_full_len(const mynah_convtr1d_spec *spec, size_t in_len,
                           size_t *out) {
    size_t span = 0;
    if (sea_mul(in_len - 1u, spec->stride, &span) != 0) return -1;
    return sea_add(span, spec->kernel_size, out);
}

/* The GEMM fast path's un-scattered product: [out_channels * kernel][in_len].
 * Zero floats for a grouped convtranspose, which keeps the scalar path. */
static int convtr_taps_floats(const mynah_convtr1d_spec *spec,
                              size_t max_in_len, size_t *out) {
    *out = 0;
    if (spec->groups != 1u) return 0;
    size_t rows = 0;
    if (sea_mul(spec->out_channels, spec->kernel_size, &rows) != 0) return -1;
    return sea_mul(rows, max_in_len, out);
}

size_t mynah_causal_convtr1d_scratch(const mynah_convtr1d_spec *spec,
                                     size_t max_in_len) {
    if (convtr_validate(spec, NULL, 0) != 0 || max_in_len == 0) return 0;
    const size_t tail = spec->kernel_size - spec->stride;
    size_t full_len = 0;
    if (convtr_full_len(spec, max_in_len, &full_len) != 0) return 0;
    size_t partial = 0;
    size_t full = 0;
    if (sea_mul(spec->out_channels, tail, &partial) != 0) return 0;
    if (sea_mul(spec->out_channels, full_len, &full) != 0) return 0;
    size_t total = 0;
    if (sea_add(partial, full, &total) != 0) return 0;
    size_t taps = 0;
    if (convtr_taps_floats(spec, max_in_len, &taps) != 0) return 0;
    if (sea_add(total, taps, &total) != 0) return 0;
    return total;
}

int mynah_causal_convtr1d_init(mynah_causal_convtr1d *convtr,
                               const mynah_convtr1d_spec *spec,
                               size_t max_in_len, float *scratch,
                               size_t scratch_floats, char *error,
                               size_t error_capacity) {
    if (convtr == NULL) {
        sea_set_error(error, error_capacity, "convtr1d: null object");
        return -1;
    }
    if (convtr_validate(spec, error, error_capacity) != 0) return -1;
    if (max_in_len == 0) {
        sea_set_error(error, error_capacity, "convtr1d: max_in_len is zero");
        return -1;
    }
    const size_t needed = mynah_causal_convtr1d_scratch(spec, max_in_len);
    if (needed == 0 || needed > scratch_floats || scratch == NULL) {
        sea_set_error(error, error_capacity,
                      "convtr1d: scratch too small (%zu < %zu)", scratch_floats,
                      needed);
        return -1;
    }
    memset(convtr, 0, sizeof(*convtr));
    convtr->spec = *spec;
    convtr->tail = spec->kernel_size - spec->stride;
    convtr->max_in_len = max_in_len;
    convtr->partial = scratch;
    convtr->full = scratch + spec->out_channels * convtr->tail;
    size_t full_len = 0;
    size_t taps = 0;
    if (convtr_full_len(spec, max_in_len, &full_len) != 0 ||
        convtr_taps_floats(spec, max_in_len, &taps) != 0) {
        sea_set_error(error, error_capacity, "convtr1d: shape overflow");
        return -1;
    }
    if (taps > 0) {
        convtr->taps = convtr->full + spec->out_channels * full_len;
    }
    mynah_causal_convtr1d_reset(convtr);
    return 0;
}

void mynah_causal_convtr1d_reset(mynah_causal_convtr1d *convtr) {
    if (convtr == NULL || convtr->partial == NULL) return;
    if (convtr->tail > 0) {
        memset(convtr->partial, 0,
               convtr->spec.out_channels * convtr->tail * sizeof(float));
    }
}

/* The reference accumulation: input stationary, one kernel-length axpy per
 * (in channel, out channel, position).  Kept as the fallback for a grouped
 * convtranspose and for a build with no BLAS. */
static void convtr_scatter_scalar(const mynah_convtr1d_spec *spec,
                                  const mynah_conv_weights *weights,
                                  const float *input, size_t in_len, float *full,
                                  size_t full_len) {
    const size_t kernel = spec->kernel_size;
    const size_t stride = spec->stride;
    const size_t in_per_group = spec->in_channels / spec->groups;
    const size_t out_per_group = spec->out_channels / spec->groups;
    for (size_t ic = 0; ic < spec->in_channels; ++ic) {
        const size_t group = ic / in_per_group;
        const float *weight_row = weights->weight + ic * out_per_group * kernel;
        const float *in_row = input + ic * in_len;
        for (size_t j = 0; j < out_per_group; ++j) {
            const float *w = weight_row + j * kernel;
            float *out_base = full + (group * out_per_group + j) * full_len;
            for (size_t t = 0; t < in_len; ++t) {
                const float value = in_row[t];
                if (value == 0.0f) continue;
                float *dst = out_base + t * stride;
                for (size_t k = 0; k < kernel; ++k) dst[k] += w[k] * value;
            }
        }
    }
}

int mynah_causal_convtr1d_apply(mynah_causal_convtr1d *convtr,
                                const mynah_conv_weights *weights,
                                const float *input, size_t in_len,
                                float *output) {
    if (convtr == NULL || weights == NULL || weights->weight == NULL ||
        input == NULL || output == NULL) {
        return -1;
    }
    if (in_len == 0 || in_len > convtr->max_in_len) return -1;

    const mynah_convtr1d_spec *spec = &convtr->spec;
    size_t full_len = 0;
    if (convtr_full_len(spec, in_len, &full_len) != 0) return -1;
    const size_t tail = convtr->tail;
    const size_t out_len = in_len * spec->stride;
    float *full = convtr->full;

    /* PyTorch adds the bias to every output position. */
    const unsigned long long t_fill = sea_prof_now();
    for (size_t oc = 0; oc < spec->out_channels; ++oc) {
        float *row = full + oc * full_len;
        const float bias = (weights->bias != NULL) ? weights->bias[oc] : 0.0f;
        if (bias == 0.0f) {
            memset(row, 0, full_len * sizeof(float));
        } else {
            for (size_t i = 0; i < full_len; ++i) row[i] = bias;
        }
    }
    sea_prof_add(SEA_PH_CONVTR_FILL, t_fill, spec->out_channels * full_len);

    const size_t kernel = spec->kernel_size;
    const size_t stride = spec->stride;

    int folded = 0;

    /* Same accounting as conv1d above, and this is the one that carries the
     * finding from .work/dtype-and-fallbacks.md item E: the Mimi depthwise
     * `upsample` runs with groups == out_channels (512 on the pinned pack), so
     * it can NEVER qualify -- the trailing two weight axes are not one dense
     * matrix when the convolution is grouped -- and it has therefore taken
     * convtr_scatter_scalar silently on every frame since the GEMM work
     * landed.  Counting it is what turns "always scalar" from a thing someone
     * had to read the source to learn into a row in the report. */
    sea_bump(&g_sea.convtr_calls);
#if !defined(MYNAH_SEANET_BLAS)
    sea_bump(&g_sea.convtr_scalar_nogemm);
#else
    const size_t gemm_rows = spec->out_channels * kernel;
    if (!sea_gemm_enabled()) {
        sea_bump(&g_sea.convtr_scalar_nogemm);
    } else if (spec->groups != 1u) {
        sea_bump(&g_sea.convtr_scalar_groups);
    } else if (convtr->taps == NULL) {
        sea_bump(&g_sea.convtr_scalar_taps);
    } else if (!sea_dims_fit(gemm_rows, in_len, spec->in_channels, gemm_rows,
                             in_len, in_len)) {
        sea_bump(&g_sea.convtr_scalar_narrow);
    } else {
        folded = 1;
    }
#endif

#if defined(MYNAH_SEANET_BLAS)
    /* PyTorch stores a ConvTranspose1d weight as [in_channels][out_channels][kernel],
     * so with groups == 1 the trailing two axes are already one dense
     * [in_channels][out_channels * kernel] matrix.  Then
     *   taps[oc*kernel + k][t] = sum_ic W[ic][oc*kernel + k] * input[ic][t]
     * is a single GEMM with A transposed, and the only thing left is the
     * scatter taps -> full[oc][t*stride + k], which costs out_channels * kernel
     * * in_len adds against the GEMM's in_channels times that. */
    if (folded) {
        sea_bump(&g_sea.convtr_gemm);
        const size_t rows = gemm_rows;
        const unsigned long long t_gemm = sea_prof_now();
        sea_sgemm(1, rows, in_len, spec->in_channels, weights->weight, rows,
                  input, in_len, 0.0f, convtr->taps, in_len);
        sea_prof_add(SEA_PH_CONVTR_GEMM, t_gemm,
                     rows * in_len * spec->in_channels);
        const unsigned long long t_scatter = sea_prof_now();
        for (size_t oc = 0; oc < spec->out_channels; ++oc) {
            float *row = full + oc * full_len;
            for (size_t k = 0; k < kernel; ++k) {
                const float *src = convtr->taps + (oc * kernel + k) * in_len;
                float *dst = row + k;
                for (size_t t = 0; t < in_len; ++t) dst[t * stride] += src[t];
            }
        }
        sea_prof_add(SEA_PH_CONVTR_SCATTER, t_scatter, rows * in_len);
    }
#endif
    if (!folded) {
        const unsigned long long t_sc = sea_prof_now();
        convtr_scatter_scalar(spec, weights, input, in_len, full, full_len);
        sea_prof_add(SEA_PH_CONVTR_SCALAR, t_sc,
                     spec->out_channels * kernel * in_len);
    }

    if (tail > 0) {
        /* Faithful to upstream StreamingConvTranspose1d.forward: the carried
         * tail is folded into the head first, and only then is the new tail
         * taken (with the bias removed, because the next call re-adds it). */
        const unsigned long long t_tail = sea_prof_now();
        for (size_t oc = 0; oc < spec->out_channels; ++oc) {
            float *row = full + oc * full_len;
            const float *carry = convtr->partial + oc * tail;
            for (size_t i = 0; i < tail; ++i) row[i] += carry[i];
        }
        for (size_t oc = 0; oc < spec->out_channels; ++oc) {
            const float *row = full + oc * full_len;
            float *carry = convtr->partial + oc * tail;
            const float bias =
                (weights->bias != NULL) ? weights->bias[oc] : 0.0f;
            for (size_t i = 0; i < tail; ++i) {
                carry[i] = row[full_len - tail + i] - bias;
            }
        }
        sea_prof_add(SEA_PH_CONVTR_TAIL, t_tail,
                     2u * spec->out_channels * tail);
    }

    const unsigned long long t_copy = sea_prof_now();
    for (size_t oc = 0; oc < spec->out_channels; ++oc) {
        memcpy(output + oc * out_len, full + oc * full_len,
               out_len * sizeof(float));
    }
    sea_prof_add(SEA_PH_CONVTR_COPY, t_copy, spec->out_channels * out_len);
    return 0;
}

/* ------------------------------------------------------------ SEANet ops */

enum {
    SEA_OP_CONV = 0,
    SEA_OP_CONVTR = 1,
    SEA_OP_RESBLOCK = 2
};

enum {
    SEA_W_FIRST = 0,
    SEA_W_CONVTR = 1,
    SEA_W_BLOCK = 2,
    SEA_W_LAST = 3
};

typedef struct {
    int kind;
    int pre_elu;
    int weight_kind;
    size_t weight_index;
    size_t in_len;  /* maximum input length for this op */
    size_t out_len; /* maximum output length for this op */
    mynah_conv1d_spec conv_spec;
    mynah_convtr1d_spec convtr_spec;
    mynah_conv1d_spec rb1_spec;
    mynah_conv1d_spec rb2_spec;
    mynah_causal_conv1d conv;
    mynah_causal_convtr1d convtr;
    mynah_causal_conv1d rb1;
    mynah_causal_conv1d rb2;
} sea_op;

struct mynah_seanet_state {
    mynah_seanet_config config;
    size_t *ratios; /* owned copy; config.ratios points at it */
    mynah_resample_config up_config;
    int has_upsample;
    size_t max_latent_frames;
    size_t max_encoder_frames;
    size_t encoder_stride;
    size_t hop_length;
    size_t position;

    sea_op *ops;
    size_t n_ops;

    mynah_causal_convtr1d upsample;

    float *arena;
    size_t arena_floats;
    float *work[3];
    size_t work_floats;
};

struct mynah_seanet_downsample {
    mynah_resample_config config;
    mynah_causal_conv1d conv;
    float *arena;
};

static mynah_conv1d_spec sea_conv_spec(size_t in_channels, size_t out_channels,
                                       size_t kernel, size_t stride,
                                       size_t dilation, size_t groups,
                                       mynah_conv_pad_mode pad_mode) {
    mynah_conv1d_spec spec;
    spec.in_channels = in_channels;
    spec.out_channels = out_channels;
    spec.kernel_size = kernel;
    spec.stride = stride;
    spec.dilation = dilation;
    spec.groups = groups;
    spec.pad_mode = pad_mode;
    return spec;
}

/*
 * Walks the decoder topology exactly once.  With `ops == NULL` it only counts
 * and measures, which is how the single arena gets sized before anything is
 * allocated.
 */
static int sea_build_ops(const mynah_seanet_config *config,
                         size_t max_encoder_frames, sea_op *ops,
                         size_t *n_ops_out, size_t *scratch_out,
                         size_t *max_elems_out, char *error,
                         size_t error_capacity) {
    size_t mult = 1u;
    for (size_t i = 0; i < config->n_ratios; ++i) {
        if (sea_mul(mult, 2u, &mult) != 0) {
            sea_set_error(error, error_capacity, "seanet: ratio count overflow");
            return -1;
        }
    }
    size_t channels_here = 0;
    if (sea_mul(mult, config->n_filters, &channels_here) != 0) {
        sea_set_error(error, error_capacity, "seanet: filter count overflow");
        return -1;
    }

    size_t n_ops = 0;
    size_t scratch = 0;
    size_t max_elems = 0;
    size_t len = max_encoder_frames;

    size_t elems = 0;
    if (sea_mul(config->dimension, len, &elems) != 0) {
        sea_set_error(error, error_capacity, "seanet: activation overflow");
        return -1;
    }
    max_elems = sea_max(max_elems, elems);

#define SEA_EMIT_CONV(SPECVAR, MAXLEN, TARGET)                                \
    do {                                                                     \
        const size_t need = mynah_causal_conv1d_scratch(&(SPECVAR), (MAXLEN)); \
        size_t eff = 0;                                                      \
        if (conv_validate(&(SPECVAR), &eff, error, error_capacity) != 0)      \
            return -1;                                                       \
        if (sea_add(scratch, need, &scratch) != 0) {                          \
            sea_set_error(error, error_capacity, "seanet: scratch overflow");  \
            return -1;                                                       \
        }                                                                    \
        (void)(TARGET);                                                      \
    } while (0)

    /* index 0: the entry convolution */
    {
        mynah_conv1d_spec spec =
            sea_conv_spec(config->dimension, channels_here, config->kernel_size,
                          1u, 1u, 1u, MYNAH_CONV_PAD_ZERO);
        SEA_EMIT_CONV(spec, len, 0);
        if (ops != NULL) {
            sea_op *op = &ops[n_ops];
            memset(op, 0, sizeof(*op));
            op->kind = SEA_OP_CONV;
            op->pre_elu = 0;
            op->weight_kind = SEA_W_FIRST;
            op->weight_index = 0;
            op->in_len = len;
            op->out_len = len;
            op->conv_spec = spec;
        }
        ++n_ops;
        if (sea_mul(channels_here, len, &elems) != 0) {
            sea_set_error(error, error_capacity, "seanet: activation overflow");
            return -1;
        }
        max_elems = sea_max(max_elems, elems);
    }

    for (size_t stage = 0; stage < config->n_ratios; ++stage) {
        const size_t ratio = config->ratios[stage];
        if (ratio == 0) {
            sea_set_error(error, error_capacity, "seanet: ratio %zu is zero",
                          stage);
            return -1;
        }
        const size_t out_channels = channels_here / 2u;
        if (out_channels == 0) {
            sea_set_error(error, error_capacity,
                          "seanet: stage %zu collapses to zero channels",
                          stage);
            return -1;
        }
        mynah_convtr1d_spec tspec;
        tspec.in_channels = channels_here;
        tspec.out_channels = out_channels;
        tspec.kernel_size = ratio * 2u;
        tspec.stride = ratio;
        tspec.groups = 1u;
        if (convtr_validate(&tspec, error, error_capacity) != 0) return -1;
        {
            const size_t need = mynah_causal_convtr1d_scratch(&tspec, len);
            if (need == 0 || sea_add(scratch, need, &scratch) != 0) {
                sea_set_error(error, error_capacity,
                              "seanet: convtr scratch overflow at stage %zu",
                              stage);
                return -1;
            }
        }
        const size_t in_len_here = len;
        if (sea_mul(len, ratio, &len) != 0) {
            sea_set_error(error, error_capacity, "seanet: length overflow");
            return -1;
        }
        if (ops != NULL) {
            sea_op *op = &ops[n_ops];
            memset(op, 0, sizeof(*op));
            op->kind = SEA_OP_CONVTR;
            op->pre_elu = 1;
            op->weight_kind = SEA_W_CONVTR;
            op->weight_index = stage;
            op->in_len = in_len_here;
            op->out_len = len;
            op->convtr_spec = tspec;
        }
        ++n_ops;
        if (sea_mul(out_channels, len, &elems) != 0) {
            sea_set_error(error, error_capacity, "seanet: activation overflow");
            return -1;
        }
        max_elems = sea_max(max_elems, elems);

        for (size_t j = 0; j < config->n_residual_layers; ++j) {
            if (config->compress == 0) {
                sea_set_error(error, error_capacity, "seanet: compress is 0");
                return -1;
            }
            const size_t hidden = out_channels / config->compress;
            if (hidden == 0) {
                sea_set_error(error, error_capacity,
                              "seanet: compress %zu collapses stage %zu",
                              config->compress, stage);
                return -1;
            }
            size_t dilation = 1u;
            for (size_t d = 0; d < j; ++d) {
                if (sea_mul(dilation, config->dilation_base, &dilation) != 0) {
                    sea_set_error(error, error_capacity,
                                  "seanet: dilation overflow");
                    return -1;
                }
            }
            mynah_conv1d_spec s1 = sea_conv_spec(
                out_channels, hidden, config->residual_kernel_size, 1u,
                dilation, 1u, MYNAH_CONV_PAD_ZERO);
            mynah_conv1d_spec s2 = sea_conv_spec(hidden, out_channels, 1u, 1u,
                                                 1u, 1u, MYNAH_CONV_PAD_ZERO);
            SEA_EMIT_CONV(s1, len, 0);
            SEA_EMIT_CONV(s2, len, 0);
            if (ops != NULL) {
                sea_op *op = &ops[n_ops];
                memset(op, 0, sizeof(*op));
                op->kind = SEA_OP_RESBLOCK;
                op->pre_elu = 0;
                op->weight_kind = SEA_W_BLOCK;
                op->weight_index = stage * config->n_residual_layers + j;
                op->in_len = len;
                op->out_len = len;
                op->rb1_spec = s1;
                op->rb2_spec = s2;
            }
            ++n_ops;
            if (sea_mul(hidden, len, &elems) != 0) {
                sea_set_error(error, error_capacity,
                              "seanet: activation overflow");
                return -1;
            }
            max_elems = sea_max(max_elems, elems);
        }
        channels_here = out_channels;
    }

    if (channels_here != config->n_filters) {
        sea_set_error(error, error_capacity,
                      "seanet: last stage has %zu channels, expected "
                      "n_filters %zu",
                      channels_here, config->n_filters);
        return -1;
    }

    {
        mynah_conv1d_spec spec =
            sea_conv_spec(config->n_filters, config->channels,
                          config->last_kernel_size, 1u, 1u, 1u,
                          MYNAH_CONV_PAD_ZERO);
        SEA_EMIT_CONV(spec, len, 0);
        if (ops != NULL) {
            sea_op *op = &ops[n_ops];
            memset(op, 0, sizeof(*op));
            op->kind = SEA_OP_CONV;
            op->pre_elu = 1;
            op->weight_kind = SEA_W_LAST;
            op->weight_index = 0;
            op->in_len = len;
            op->out_len = len;
            op->conv_spec = spec;
        }
        ++n_ops;
        if (sea_mul(config->channels, len, &elems) != 0) {
            sea_set_error(error, error_capacity, "seanet: activation overflow");
            return -1;
        }
        max_elems = sea_max(max_elems, elems);
    }

#undef SEA_EMIT_CONV

    if (n_ops_out != NULL) *n_ops_out = n_ops;
    if (scratch_out != NULL) *scratch_out = scratch;
    if (max_elems_out != NULL) *max_elems_out = max_elems;
    return 0;
}

static int sea_config_valid(const mynah_seanet_config *config, char *error,
                            size_t error_capacity) {
    if (config == NULL) {
        sea_set_error(error, error_capacity, "seanet: null config");
        return -1;
    }
    if (config->channels == 0 || config->dimension == 0 ||
        config->n_filters == 0 || config->n_ratios == 0 ||
        config->ratios == NULL || config->kernel_size == 0 ||
        config->last_kernel_size == 0 || config->compress == 0) {
        sea_set_error(error, error_capacity, "seanet: incomplete config");
        return -1;
    }
    if (config->n_residual_layers > 0 &&
        (config->residual_kernel_size == 0 || config->dilation_base == 0)) {
        sea_set_error(error, error_capacity,
                      "seanet: residual layers need kernel and dilation base");
        return -1;
    }
    return 0;
}

mynah_seanet_state *mynah_seanet_state_create(const mynah_seanet_config *config,
                                              const mynah_resample_config *up,
                                              size_t max_latent_frames,
                                              char *error,
                                              size_t error_capacity) {
    if (error != NULL && error_capacity > 0) error[0] = '\0';
    if (sea_config_valid(config, error, error_capacity) != 0) return NULL;
    if (max_latent_frames == 0) {
        sea_set_error(error, error_capacity,
                      "seanet: max_latent_frames is zero");
        return NULL;
    }
    size_t encoder_stride = 1u;
    if (up != NULL) {
        if (up->stride == 0 || up->in_channels == 0 || up->out_channels == 0 ||
            up->groups == 0) {
            sea_set_error(error, error_capacity, "seanet: bad resample config");
            return NULL;
        }
        encoder_stride = up->stride;
    }
    size_t max_encoder_frames = 0;
    if (sea_mul(max_latent_frames, encoder_stride, &max_encoder_frames) != 0) {
        sea_set_error(error, error_capacity, "seanet: frame count overflow");
        return NULL;
    }

    size_t hop = 1u;
    for (size_t i = 0; i < config->n_ratios; ++i) {
        if (config->ratios[i] == 0 ||
            sea_mul(hop, config->ratios[i], &hop) != 0) {
            sea_set_error(error, error_capacity, "seanet: bad ratios");
            return NULL;
        }
    }

    size_t n_ops = 0;
    size_t conv_scratch = 0;
    size_t max_elems = 0;
    if (sea_build_ops(config, max_encoder_frames, NULL, &n_ops, &conv_scratch,
                      &max_elems, error, error_capacity) != 0) {
        return NULL;
    }

    mynah_seanet_state *state = calloc(1u, sizeof(*state));
    if (state == NULL) {
        sea_set_error(error, error_capacity, "seanet: out of memory");
        return NULL;
    }
    state->config = *config;
    state->ratios = calloc(config->n_ratios, sizeof(size_t));
    if (state->ratios == NULL) {
        sea_set_error(error, error_capacity, "seanet: out of memory");
        free(state);
        return NULL;
    }
    memcpy(state->ratios, config->ratios, config->n_ratios * sizeof(size_t));
    state->config.ratios = state->ratios;
    state->max_latent_frames = max_latent_frames;
    state->max_encoder_frames = max_encoder_frames;
    state->encoder_stride = encoder_stride;
    state->hop_length = hop;
    state->position = 0;
    state->n_ops = n_ops;
    state->work_floats = max_elems;

    size_t up_scratch = 0;
    if (up != NULL) {
        mynah_convtr1d_spec uspec;
        uspec.in_channels = up->in_channels;
        uspec.out_channels = up->out_channels;
        uspec.kernel_size = up->stride * 2u;
        uspec.stride = up->stride;
        uspec.groups = up->groups;
        up_scratch = mynah_causal_convtr1d_scratch(&uspec, max_latent_frames);
        if (up_scratch == 0) {
            sea_set_error(error, error_capacity, "seanet: bad upsample config");
            free(state->ratios);
            free(state);
            return NULL;
        }
        state->up_config = *up;
        state->has_upsample = 1;
    }

    size_t work_total = 0;
    size_t arena_floats = 0;
    if (sea_mul(max_elems, 3u, &work_total) != 0 ||
        sea_add(conv_scratch, up_scratch, &arena_floats) != 0 ||
        sea_add(arena_floats, work_total, &arena_floats) != 0) {
        sea_set_error(error, error_capacity, "seanet: arena overflow");
        free(state->ratios);
        free(state);
        return NULL;
    }

    state->ops = calloc(n_ops, sizeof(sea_op));
    state->arena = calloc(arena_floats ? arena_floats : 1u, sizeof(float));
    if (state->ops == NULL || state->arena == NULL) {
        sea_set_error(error, error_capacity, "seanet: out of memory");
        mynah_seanet_state_destroy(state);
        return NULL;
    }
    state->arena_floats = arena_floats;

    size_t counted_ops = 0;
    if (sea_build_ops(&state->config, max_encoder_frames, state->ops,
                      &counted_ops, NULL, NULL, error, error_capacity) != 0 ||
        counted_ops != n_ops) {
        sea_set_error(error, error_capacity, "seanet: topology mismatch");
        mynah_seanet_state_destroy(state);
        return NULL;
    }

    float *cursor = state->arena;
    size_t remaining = arena_floats;

#define SEA_TAKE(N)                                  \
    do {                                             \
        if ((N) > remaining) {                       \
            sea_set_error(error, error_capacity,     \
                          "seanet: arena exhausted"); \
            mynah_seanet_state_destroy(state);       \
            return NULL;                             \
        }                                            \
        remaining -= (N);                            \
    } while (0)

    for (size_t i = 0; i < n_ops; ++i) {
        sea_op *op = &state->ops[i];
        if (op->kind == SEA_OP_CONV) {
            const size_t need =
                mynah_causal_conv1d_scratch(&op->conv_spec, op->in_len);
            SEA_TAKE(need);
            if (mynah_causal_conv1d_init(&op->conv, &op->conv_spec, op->in_len,
                                         cursor, need, error,
                                         error_capacity) != 0) {
                mynah_seanet_state_destroy(state);
                return NULL;
            }
            cursor += need;
        } else if (op->kind == SEA_OP_CONVTR) {
            const size_t need =
                mynah_causal_convtr1d_scratch(&op->convtr_spec, op->in_len);
            SEA_TAKE(need);
            if (mynah_causal_convtr1d_init(&op->convtr, &op->convtr_spec,
                                           op->in_len, cursor, need, error,
                                           error_capacity) != 0) {
                mynah_seanet_state_destroy(state);
                return NULL;
            }
            cursor += need;
        } else {
            const size_t need1 =
                mynah_causal_conv1d_scratch(&op->rb1_spec, op->in_len);
            SEA_TAKE(need1);
            if (mynah_causal_conv1d_init(&op->rb1, &op->rb1_spec, op->in_len,
                                         cursor, need1, error,
                                         error_capacity) != 0) {
                mynah_seanet_state_destroy(state);
                return NULL;
            }
            cursor += need1;
            const size_t need2 =
                mynah_causal_conv1d_scratch(&op->rb2_spec, op->in_len);
            SEA_TAKE(need2);
            if (mynah_causal_conv1d_init(&op->rb2, &op->rb2_spec, op->in_len,
                                         cursor, need2, error,
                                         error_capacity) != 0) {
                mynah_seanet_state_destroy(state);
                return NULL;
            }
            cursor += need2;
        }
    }

    if (state->has_upsample) {
        mynah_convtr1d_spec uspec;
        uspec.in_channels = state->up_config.in_channels;
        uspec.out_channels = state->up_config.out_channels;
        uspec.kernel_size = state->up_config.stride * 2u;
        uspec.stride = state->up_config.stride;
        uspec.groups = state->up_config.groups;
        SEA_TAKE(up_scratch);
        if (mynah_causal_convtr1d_init(&state->upsample, &uspec,
                                       max_latent_frames, cursor, up_scratch,
                                       error, error_capacity) != 0) {
            mynah_seanet_state_destroy(state);
            return NULL;
        }
        cursor += up_scratch;
    }

    for (size_t i = 0; i < 3u; ++i) {
        SEA_TAKE(max_elems);
        state->work[i] = cursor;
        cursor += max_elems;
    }

#undef SEA_TAKE

    return state;
}

void mynah_seanet_state_destroy(mynah_seanet_state *state) {
    if (state == NULL) return;
    free(state->ops);
    free(state->arena);
    free(state->ratios);
    free(state);
}

void mynah_seanet_state_reset(mynah_seanet_state *state) {
    if (state == NULL) return;
    for (size_t i = 0; i < state->n_ops; ++i) {
        sea_op *op = &state->ops[i];
        if (op->kind == SEA_OP_CONV) {
            mynah_causal_conv1d_reset(&op->conv);
        } else if (op->kind == SEA_OP_CONVTR) {
            mynah_causal_convtr1d_reset(&op->convtr);
        } else {
            mynah_causal_conv1d_reset(&op->rb1);
            mynah_causal_conv1d_reset(&op->rb2);
        }
    }
    if (state->has_upsample) mynah_causal_convtr1d_reset(&state->upsample);
    state->position = 0; /* the half everybody forgets */
}

size_t mynah_seanet_state_position(const mynah_seanet_state *state) {
    return (state == NULL) ? 0 : state->position;
}

void mynah_seanet_state_advance(mynah_seanet_state *state,
                                size_t n_latent_frames) {
    if (state == NULL) return;
    size_t delta = 0;
    if (sea_mul(n_latent_frames, state->encoder_stride, &delta) != 0) return;
    size_t next = 0;
    if (sea_add(state->position, delta, &next) != 0) return;
    state->position = next;
}

size_t mynah_seanet_state_encoder_stride(const mynah_seanet_state *state) {
    return (state == NULL) ? 0 : state->encoder_stride;
}

size_t mynah_seanet_state_hop_length(const mynah_seanet_state *state) {
    return (state == NULL) ? 0 : state->hop_length;
}

size_t mynah_seanet_state_samples_per_latent(const mynah_seanet_state *state) {
    if (state == NULL) return 0;
    size_t out = 0;
    if (sea_mul(state->encoder_stride, state->hop_length, &out) != 0) return 0;
    return out;
}

size_t mynah_seanet_state_max_latent_frames(const mynah_seanet_state *state) {
    return (state == NULL) ? 0 : state->max_latent_frames;
}

int mynah_seanet_check_decoder_weights(const mynah_seanet_state *state,
                                       const mynah_seanet_decoder_weights *w,
                                       char *error, size_t error_capacity) {
    if (state == NULL || w == NULL) {
        sea_set_error(error, error_capacity, "seanet: null argument");
        return -1;
    }
    if (w->first.weight == NULL || w->last.weight == NULL) {
        sea_set_error(error, error_capacity,
                      "seanet: first/last conv weights missing");
        return -1;
    }
    if (w->convtr == NULL) {
        sea_set_error(error, error_capacity, "seanet: convtr array missing");
        return -1;
    }
    for (size_t i = 0; i < state->config.n_ratios; ++i) {
        if (w->convtr[i].weight == NULL) {
            sea_set_error(error, error_capacity,
                          "seanet: convtr[%zu] weight missing", i);
            return -1;
        }
    }
    const size_t n_blocks =
        state->config.n_ratios * state->config.n_residual_layers;
    if (n_blocks > 0) {
        if (w->blocks == NULL) {
            sea_set_error(error, error_capacity,
                          "seanet: resblock array missing");
            return -1;
        }
        for (size_t i = 0; i < n_blocks; ++i) {
            if (w->blocks[i].conv1.weight == NULL ||
                w->blocks[i].conv2.weight == NULL) {
                sea_set_error(error, error_capacity,
                              "seanet: resblock[%zu] weights missing", i);
                return -1;
            }
        }
    }
    return 0;
}

int mynah_seanet_upsample(mynah_seanet_state *state,
                          const mynah_conv_weights *weights,
                          const float *input, size_t n_latent_frames,
                          float *output) {
    if (state == NULL || !state->has_upsample) return -1;
    if (n_latent_frames == 0 || n_latent_frames > state->max_latent_frames) {
        return -1;
    }
    return mynah_causal_convtr1d_apply(&state->upsample, weights, input,
                                       n_latent_frames, output);
}

int mynah_seanet_decode(mynah_seanet_state *state,
                        const mynah_seanet_decoder_weights *weights,
                        const float *input, size_t n_encoder_frames,
                        float *output) {
    if (state == NULL || weights == NULL || input == NULL || output == NULL) {
        return -1;
    }
    if (n_encoder_frames == 0 ||
        n_encoder_frames > state->max_encoder_frames) {
        return -1;
    }
    const unsigned long long t_decode = sea_prof_now();
    const float alpha = state->config.elu_alpha;
    float *a = state->work[0];
    float *b = state->work[1];
    float *c = state->work[2];

    const float *cur_in = input;
    size_t len = n_encoder_frames;
    size_t channels = state->config.dimension;
    float *cur = NULL;

    for (size_t i = 0; i < state->n_ops; ++i) {
        sea_op *op = &state->ops[i];
        const int last_op = (i + 1u == state->n_ops);

        if (op->kind == SEA_OP_CONV) {
            const mynah_conv_weights *cw = (op->weight_kind == SEA_W_FIRST)
                                               ? &weights->first
                                               : &weights->last;
            const float *src;
            if (cur == NULL) {
                /* The entry convolution reads the caller's buffer and never
                 * has a pre-activation, so the input is never written to. */
                src = cur_in;
            } else {
                if (op->pre_elu) {
                    mynah_seanet_elu_f32(cur, cur, channels * len, alpha);
                }
                src = cur;
            }
            float *dst = last_op ? output : ((src == a) ? b : a);
            if (mynah_causal_conv1d_apply(&op->conv, cw, src, len, dst) != 0) {
                return -1;
            }
            cur = dst;
            channels = op->conv_spec.out_channels;
        } else if (op->kind == SEA_OP_CONVTR) {
            if (cur == NULL) return -1; /* topology always starts with a conv */
            float *src = cur;
            if (op->pre_elu) {
                mynah_seanet_elu_f32(src, src, channels * len, alpha);
            }
            float *dst = (src == a) ? b : a;
            if (mynah_causal_convtr1d_apply(&op->convtr,
                                            &weights->convtr[op->weight_index],
                                            src, len, dst) != 0) {
                return -1;
            }
            cur = dst;
            channels = op->convtr_spec.out_channels;
            len *= op->convtr_spec.stride;
        } else {
            if (cur == NULL) return -1;
            const mynah_seanet_resblock_weights *bw =
                &weights->blocks[op->weight_index];
            float *other = (cur == a) ? b : a;
            /* v = conv2(elu(conv1(elu(x)))); x = x + v  (true_skip shortcut) */
            mynah_seanet_elu_f32(cur, c, channels * len, alpha);
            if (mynah_causal_conv1d_apply(&op->rb1, &bw->conv1, c, len,
                                          other) != 0) {
                return -1;
            }
            const size_t hidden = op->rb1_spec.out_channels;
            mynah_seanet_elu_f32(other, other, hidden * len, alpha);
            if (mynah_causal_conv1d_apply(&op->rb2, &bw->conv2, other, len,
                                          c) != 0) {
                return -1;
            }
            const unsigned long long t_add = sea_prof_now();
            for (size_t k = 0; k < channels * len; ++k) cur[k] += c[k];
            sea_prof_add(SEA_PH_RESADD, t_add, channels * len);
        }
    }
    sea_prof_add(SEA_PH_DECODE, t_decode, n_encoder_frames);
    return 0;
}

/* ---------------------------------------------------------- downsample */

mynah_seanet_downsample *mynah_seanet_downsample_create(
    const mynah_resample_config *config, size_t max_in_len, char *error,
    size_t error_capacity) {
    if (error != NULL && error_capacity > 0) error[0] = '\0';
    if (config == NULL || config->stride == 0 || config->in_channels == 0 ||
        config->out_channels == 0 || config->groups == 0) {
        sea_set_error(error, error_capacity, "downsample: bad config");
        return NULL;
    }
    if (max_in_len == 0 || (max_in_len % config->stride) != 0) {
        sea_set_error(error, error_capacity,
                      "downsample: max_in_len %zu must be a multiple of "
                      "stride %zu",
                      max_in_len, config->stride);
        return NULL;
    }
    mynah_conv1d_spec spec = sea_conv_spec(
        config->in_channels, config->out_channels, config->stride * 2u,
        config->stride, 1u, config->groups, MYNAH_CONV_PAD_REPLICATE);
    const size_t need = mynah_causal_conv1d_scratch(&spec, max_in_len);
    mynah_seanet_downsample *down = calloc(1u, sizeof(*down));
    if (down == NULL) {
        sea_set_error(error, error_capacity, "downsample: out of memory");
        return NULL;
    }
    down->config = *config;
    down->arena = calloc(need ? need : 1u, sizeof(float));
    if (down->arena == NULL) {
        sea_set_error(error, error_capacity, "downsample: out of memory");
        free(down);
        return NULL;
    }
    if (mynah_causal_conv1d_init(&down->conv, &spec, max_in_len, down->arena,
                                 need, error, error_capacity) != 0) {
        free(down->arena);
        free(down);
        return NULL;
    }
    return down;
}

void mynah_seanet_downsample_destroy(mynah_seanet_downsample *down) {
    if (down == NULL) return;
    free(down->arena);
    free(down);
}

void mynah_seanet_downsample_reset(mynah_seanet_downsample *down) {
    if (down == NULL) return;
    mynah_causal_conv1d_reset(&down->conv);
}

int mynah_seanet_downsample_apply(mynah_seanet_downsample *down,
                                  const mynah_conv_weights *weights,
                                  const float *input, size_t in_len,
                                  float *output) {
    if (down == NULL) return -1;
    return mynah_causal_conv1d_apply(&down->conv, weights, input, in_len,
                                     output);
}

/* -------------------------------------------------------------- self test */

static float sea_fake(size_t index, size_t salt) {
    const double v = sin((double)(index * 11u + salt * 7u + 1u) * 0.41) * 0.6 +
                     cos((double)(index * 5u + salt * 3u + 2u) * 0.17) * 0.3;
    return (float)v;
}

/* One-shot causal conv reference, written straight off StreamingConv1d with
 * explicit left padding.  Double precision, no shared code. */
static void sea_ref_conv1d(const mynah_conv1d_spec *spec, const float *weight,
                           const float *bias, const float *input, size_t in_len,
                           double *padded, double *out) {
    const size_t eff = (spec->kernel_size - 1u) * spec->dilation + 1u;
    const size_t tail = eff - spec->stride;
    const size_t total = tail + in_len;
    for (size_t c = 0; c < spec->in_channels; ++c) {
        const double pad = (spec->pad_mode == MYNAH_CONV_PAD_REPLICATE)
                               ? (double)input[c * in_len]
                               : 0.0;
        for (size_t i = 0; i < tail; ++i) padded[c * total + i] = pad;
        for (size_t i = 0; i < in_len; ++i) {
            padded[c * total + tail + i] = (double)input[c * in_len + i];
        }
    }
    const size_t out_len = in_len / spec->stride;
    const size_t in_per_group = spec->in_channels / spec->groups;
    const size_t out_per_group = spec->out_channels / spec->groups;
    for (size_t oc = 0; oc < spec->out_channels; ++oc) {
        const size_t group = oc / out_per_group;
        for (size_t n = 0; n < out_len; ++n) {
            double acc = (bias != NULL) ? (double)bias[oc] : 0.0;
            for (size_t j = 0; j < in_per_group; ++j) {
                for (size_t k = 0; k < spec->kernel_size; ++k) {
                    const double w =
                        (double)weight[(oc * in_per_group + j) *
                                           spec->kernel_size +
                                       k];
                    acc += w * padded[(group * in_per_group + j) * total +
                                      n * spec->stride + k * spec->dilation];
                }
            }
            out[oc * out_len + n] = acc;
        }
    }
}

/* One-shot causal transposed conv reference: full convolution, trailing
 * (kernel - stride) samples removed. */
static void sea_ref_convtr1d(const mynah_convtr1d_spec *spec,
                             const float *weight, const float *bias,
                             const float *input, size_t in_len, double *full,
                             double *out) {
    const size_t full_len = (in_len - 1u) * spec->stride + spec->kernel_size;
    const size_t in_per_group = spec->in_channels / spec->groups;
    const size_t out_per_group = spec->out_channels / spec->groups;
    for (size_t oc = 0; oc < spec->out_channels; ++oc) {
        const double b = (bias != NULL) ? (double)bias[oc] : 0.0;
        for (size_t i = 0; i < full_len; ++i) full[oc * full_len + i] = b;
    }
    for (size_t ic = 0; ic < spec->in_channels; ++ic) {
        const size_t group = ic / in_per_group;
        for (size_t j = 0; j < out_per_group; ++j) {
            const size_t oc = group * out_per_group + j;
            for (size_t t = 0; t < in_len; ++t) {
                const double value = (double)input[ic * in_len + t];
                for (size_t k = 0; k < spec->kernel_size; ++k) {
                    const double w =
                        (double)weight[(ic * out_per_group + j) *
                                           spec->kernel_size +
                                       k];
                    full[oc * full_len + t * spec->stride + k] += w * value;
                }
            }
        }
    }
    const size_t out_len = in_len * spec->stride;
    for (size_t oc = 0; oc < spec->out_channels; ++oc) {
        for (size_t i = 0; i < out_len; ++i) {
            out[oc * out_len + i] = full[oc * full_len + i];
        }
    }
}

static int sea_close(float a, double b, double tolerance) {
    const double d = (double)a - b;
    return (d < 0 ? -d : d) <= tolerance;
}

static int sea_test_one_conv(const mynah_conv1d_spec *spec, size_t in_len,
                             size_t salt, int with_bias, const char *label,
                             char *error, size_t error_capacity) {
    const size_t eff = (spec->kernel_size - 1u) * spec->dilation + 1u;
    const size_t tail = eff - spec->stride;
    const size_t out_len = in_len / spec->stride;
    const size_t weight_count =
        spec->out_channels * (spec->in_channels / spec->groups) *
        spec->kernel_size;

    float *weight = calloc(weight_count, sizeof(float));
    float *bias = calloc(spec->out_channels, sizeof(float));
    float *input = calloc(spec->in_channels * in_len, sizeof(float));
    float *got = calloc(spec->out_channels * out_len, sizeof(float));
    double *padded =
        calloc(spec->in_channels * (tail + in_len) + 1u, sizeof(double));
    double *want = calloc(spec->out_channels * out_len, sizeof(double));
    float *scratch = NULL;
    int rc = -1;

    if (weight == NULL || bias == NULL || input == NULL || got == NULL ||
        padded == NULL || want == NULL) {
        sea_set_error(error, error_capacity, "%s: out of memory", label);
        goto fail;
    }
    for (size_t i = 0; i < weight_count; ++i) weight[i] = sea_fake(i, salt);
    for (size_t i = 0; i < spec->out_channels; ++i) {
        bias[i] = with_bias ? sea_fake(i, salt + 1u) : 0.0f;
    }
    for (size_t i = 0; i < spec->in_channels * in_len; ++i) {
        input[i] = sea_fake(i, salt + 2u);
    }

    const size_t need = mynah_causal_conv1d_scratch(spec, in_len);
    scratch = calloc(need ? need : 1u, sizeof(float));
    if (scratch == NULL) {
        sea_set_error(error, error_capacity, "%s: out of memory", label);
        goto fail;
    }

    mynah_conv_weights w;
    w.weight = weight;
    w.bias = with_bias ? bias : NULL;

    mynah_causal_conv1d conv;
    if (mynah_causal_conv1d_init(&conv, spec, in_len, scratch, need, error,
                                 error_capacity) != 0) {
        goto fail;
    }
    if (mynah_causal_conv1d_apply(&conv, &w, input, in_len, got) != 0) {
        sea_set_error(error, error_capacity, "%s: apply failed", label);
        goto fail;
    }
    sea_ref_conv1d(spec, weight, w.bias, input, in_len, padded, want);
    for (size_t i = 0; i < spec->out_channels * out_len; ++i) {
        if (!sea_close(got[i], want[i], 1e-5)) {
            sea_set_error(error, error_capacity,
                          "%s: element %zu = %.9g want %.9g", label,
                          i, (double)got[i], want[i]);
            goto fail;
        }
    }

    /* Streaming continuity: chunked decode must equal the one-shot decode.
     *
     * This used to be an exact `!=`.  It is a bound now, because the GEMM fast
     * path hands the reduction over input channels to BLAS and BLAS blocks it
     * by the number of columns, i.e. by the chunk length -- so a 1-column call
     * and a 16-column call reassociate differently and land ~1e-8 apart.  What
     * the check is for is the ring buffer: a conv that carries the wrong left
     * context is wrong by O(1), not by 1e-8, so the bound catches exactly what
     * the equality caught.  The scalar path (grouped, strided, or a non-BLAS
     * build) is still bit-exact here and passes the same bound trivially.
     *
     * This costs nothing in the product: PocketTTS decodes one latent frame per
     * call whether it is streaming or offline, so the real pipeline never varies
     * the chunk length and stream == offline stays byte-identical. */
    {
        const size_t chunk = spec->stride;
        if (in_len % chunk == 0 &&
            (spec->pad_mode != MYNAH_CONV_PAD_REPLICATE || chunk >= tail)) {
            float *chunked = calloc(spec->out_channels * out_len,
                                    sizeof(float));
            float *piece_in = calloc(spec->in_channels * chunk, sizeof(float));
            float *piece_out =
                calloc(spec->out_channels * (chunk / spec->stride),
                       sizeof(float));
            if (chunked == NULL || piece_in == NULL || piece_out == NULL) {
                free(chunked);
                free(piece_in);
                free(piece_out);
                sea_set_error(error, error_capacity, "%s: out of memory",
                              label);
                goto fail;
            }
            mynah_causal_conv1d_reset(&conv);
            const size_t piece_out_len = chunk / spec->stride;
            int failed = 0;
            for (size_t off = 0; off < in_len && !failed; off += chunk) {
                for (size_t c = 0; c < spec->in_channels; ++c) {
                    memcpy(piece_in + c * chunk, input + c * in_len + off,
                           chunk * sizeof(float));
                }
                if (mynah_causal_conv1d_apply(&conv, &w, piece_in, chunk,
                                              piece_out) != 0) {
                    failed = 1;
                    break;
                }
                for (size_t c = 0; c < spec->out_channels; ++c) {
                    memcpy(chunked + c * out_len + off / spec->stride,
                           piece_out + c * piece_out_len,
                           piece_out_len * sizeof(float));
                }
            }
            if (!failed) {
                for (size_t i = 0; i < spec->out_channels * out_len; ++i) {
                    if (!sea_close(chunked[i], (double)got[i], 1e-5)) {
                        failed = 2;
                        sea_set_error(error, error_capacity,
                                      "%s: streaming mismatch at %zu: "
                                      "%.9g vs %.9g",
                                      label, i, (double)chunked[i],
                                      (double)got[i]);
                        break;
                    }
                }
            } else {
                sea_set_error(error, error_capacity, "%s: chunked apply failed",
                              label);
            }
            free(chunked);
            free(piece_in);
            free(piece_out);
            if (failed) goto fail;
        }
    }

    rc = 0;
fail:
    free(weight);
    free(bias);
    free(input);
    free(got);
    free(padded);
    free(want);
    free(scratch);
    return rc;
}

static int sea_test_one_convtr(const mynah_convtr1d_spec *spec, size_t in_len,
                               size_t salt, int with_bias, const char *label,
                               char *error, size_t error_capacity) {
    const size_t out_len = in_len * spec->stride;
    const size_t full_len = (in_len - 1u) * spec->stride + spec->kernel_size;
    const size_t weight_count = spec->in_channels *
                                (spec->out_channels / spec->groups) *
                                spec->kernel_size;
    float *weight = calloc(weight_count, sizeof(float));
    float *bias = calloc(spec->out_channels, sizeof(float));
    float *input = calloc(spec->in_channels * in_len, sizeof(float));
    float *got = calloc(spec->out_channels * out_len, sizeof(float));
    double *full = calloc(spec->out_channels * full_len, sizeof(double));
    double *want = calloc(spec->out_channels * out_len, sizeof(double));
    float *scratch = NULL;
    int rc = -1;

    if (weight == NULL || bias == NULL || input == NULL || got == NULL ||
        full == NULL || want == NULL) {
        sea_set_error(error, error_capacity, "%s: out of memory", label);
        goto fail;
    }
    for (size_t i = 0; i < weight_count; ++i) weight[i] = sea_fake(i, salt);
    for (size_t i = 0; i < spec->out_channels; ++i) {
        bias[i] = with_bias ? sea_fake(i, salt + 1u) : 0.0f;
    }
    for (size_t i = 0; i < spec->in_channels * in_len; ++i) {
        input[i] = sea_fake(i, salt + 2u);
    }

    const size_t need = mynah_causal_convtr1d_scratch(spec, in_len);
    scratch = calloc(need ? need : 1u, sizeof(float));
    if (scratch == NULL) {
        sea_set_error(error, error_capacity, "%s: out of memory", label);
        goto fail;
    }
    mynah_conv_weights w;
    w.weight = weight;
    w.bias = with_bias ? bias : NULL;

    mynah_causal_convtr1d convtr;
    if (mynah_causal_convtr1d_init(&convtr, spec, in_len, scratch, need, error,
                                   error_capacity) != 0) {
        goto fail;
    }
    if (mynah_causal_convtr1d_apply(&convtr, &w, input, in_len, got) != 0) {
        sea_set_error(error, error_capacity, "%s: apply failed", label);
        goto fail;
    }
    sea_ref_convtr1d(spec, weight, w.bias, input, in_len, full, want);
    for (size_t i = 0; i < spec->out_channels * out_len; ++i) {
        if (!sea_close(got[i], want[i], 1e-5)) {
            sea_set_error(error, error_capacity,
                          "%s: element %zu = %.9g want %.9g", label, i,
                          (double)got[i], want[i]);
            goto fail;
        }
    }

    /* Frame-by-frame streaming must reproduce the one-shot result. */
    {
        float *chunked = calloc(spec->out_channels * out_len, sizeof(float));
        float *piece_in = calloc(spec->in_channels, sizeof(float));
        float *piece_out = calloc(spec->out_channels * spec->stride,
                                  sizeof(float));
        if (chunked == NULL || piece_in == NULL || piece_out == NULL) {
            free(chunked);
            free(piece_in);
            free(piece_out);
            sea_set_error(error, error_capacity, "%s: out of memory", label);
            goto fail;
        }
        mynah_causal_convtr1d_reset(&convtr);
        int failed = 0;
        for (size_t t = 0; t < in_len && !failed; ++t) {
            for (size_t c = 0; c < spec->in_channels; ++c) {
                piece_in[c] = input[c * in_len + t];
            }
            if (mynah_causal_convtr1d_apply(&convtr, &w, piece_in, 1u,
                                            piece_out) != 0) {
                failed = 1;
                break;
            }
            for (size_t c = 0; c < spec->out_channels; ++c) {
                memcpy(chunked + c * out_len + t * spec->stride,
                       piece_out + c * spec->stride,
                       spec->stride * sizeof(float));
            }
        }
        if (!failed) {
            for (size_t i = 0; i < spec->out_channels * out_len; ++i) {
                if (!sea_close(chunked[i], (double)got[i], 1e-5)) {
                    failed = 2;
                    sea_set_error(error, error_capacity,
                                  "%s: streaming mismatch at %zu: %.9g vs %.9g",
                                  label, i, (double)chunked[i],
                                  (double)got[i]);
                    break;
                }
            }
        } else {
            sea_set_error(error, error_capacity, "%s: chunked apply failed",
                          label);
        }
        free(chunked);
        free(piece_in);
        free(piece_out);
        if (failed) goto fail;
    }

    rc = 0;
fail:
    free(weight);
    free(bias);
    free(input);
    free(got);
    free(full);
    free(want);
    free(scratch);
    return rc;
}

/* The bound the exp above claims, checked rather than believed, and checked on
 * whichever path this build compiled: the vector one where it exists, the
 * scalar reference otherwise.  Both go through mynah_seanet_elu_f32, so a
 * vector path that drifts from the reference fails here and not in a waveform
 * three modules downstream. */
static int sea_exp_self_test(char *error, size_t error_capacity) {
    enum { N = 4096 };
    static float in[N], out[N];
    double worst = 0.0;
    float worst_x = 0.0f;
    for (int block = 0; block < 22; ++block) {
        const float base = -88.0f + (float)block * 4.0f;
        for (int i = 0; i < N; ++i) in[i] = base + 4.0f * (float)i / (float)N;
        mynah_seanet_elu_f32(in, out, (size_t)N, 1.0f);
        for (int i = 0; i < N; ++i) {
            const float x = in[i];
            const float want = (x > 0.0f) ? x : (expf(x) - 1.0f);
            const double d = fabs((double)out[i] - (double)want);
            if (d > worst) { worst = d; worst_x = x; }
        }
    }
    if (!(worst <= 1e-7)) {
        if (error != NULL && error_capacity > 0)
            snprintf(error, error_capacity,
                     "seanet ELU: absolute error %.3e at x=%.6f exceeds 1e-7 "
                     "(measured 5.96e-08 when this landed) -- the range "
                     "reduction or the polynomial regressed",
                     worst, (double)worst_x);
        return -1;
    }
    return 0;
}

int mynah_seanet_self_test(char *error, size_t error_capacity) {
    if (sea_exp_self_test(error, error_capacity) != 0) return -1;
    if (error != NULL && error_capacity > 0) error[0] = '\0';

    /* --- ELU --------------------------------------------------------- */
    {
        const float in[4] = {1.5f, 0.0f, -1.0f, -3.0f};
        const double want[4] = {1.5, 0.0, -0.6321205588285577,
                                -0.950212931632136};
        float got[4];
        mynah_seanet_elu_f32(in, got, 4u, 1.0f);
        for (size_t i = 0; i < 4u; ++i) {
            if (!sea_close(got[i], want[i], 1e-6)) {
                sea_set_error(error, error_capacity,
                              "elu[%zu] = %.9g want %.9g", i, (double)got[i],
                              want[i]);
                return -1;
            }
        }
        float half[2] = {-1.0f, 2.0f};
        mynah_seanet_elu_f32(half, half, 2u, 0.5f);
        if (!sea_close(half[0], -0.5 * 0.6321205588285577, 1e-6) ||
            half[1] != 2.0f) {
            sea_set_error(error, error_capacity, "elu alpha not honoured");
            return -1;
        }
    }

    /* --- causal conv1d, every shape that matters --------------------- */
    {
        struct {
            mynah_conv1d_spec spec;
            size_t in_len;
            int bias;
            const char *label;
        } cases[] = {
            {sea_conv_spec(3u, 2u, 3u, 1u, 1u, 1u, MYNAH_CONV_PAD_ZERO), 8u, 1,
             "conv k3"},
            {sea_conv_spec(2u, 3u, 1u, 1u, 1u, 1u, MYNAH_CONV_PAD_ZERO), 6u, 1,
             "conv k1 (no state)"},
            {sea_conv_spec(2u, 2u, 3u, 1u, 2u, 1u, MYNAH_CONV_PAD_ZERO), 9u, 1,
             "conv dilated"},
            {sea_conv_spec(4u, 4u, 4u, 2u, 1u, 1u, MYNAH_CONV_PAD_ZERO), 12u, 1,
             "conv strided"},
            {sea_conv_spec(4u, 4u, 3u, 1u, 1u, 4u, MYNAH_CONV_PAD_ZERO), 8u, 0,
             "conv depthwise"},
            {sea_conv_spec(3u, 2u, 6u, 3u, 1u, 1u, MYNAH_CONV_PAD_REPLICATE),
             12u, 0, "conv replicate"},
        };
        for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
            if (sea_test_one_conv(&cases[i].spec, cases[i].in_len, i + 1u,
                                  cases[i].bias, cases[i].label, error,
                                  error_capacity) != 0) {
                return -1;
            }
        }
    }

    /* --- causal transposed conv, dense and depthwise ------------------ */
    {
        mynah_convtr1d_spec dense = {3u, 2u, 4u, 2u, 1u};
        mynah_convtr1d_spec depthwise = {4u, 4u, 8u, 4u, 4u};
        mynah_convtr1d_spec mimi_like = {6u, 6u, 8u, 4u, 6u};
        if (sea_test_one_convtr(&dense, 5u, 11u, 1, "convtr dense", error,
                                error_capacity) != 0) {
            return -1;
        }
        if (sea_test_one_convtr(&depthwise, 4u, 13u, 0, "convtr depthwise",
                                error, error_capacity) != 0) {
            return -1;
        }
        if (sea_test_one_convtr(&mimi_like, 3u, 17u, 0,
                                "convtr depthwise (upsample shape)", error,
                                error_capacity) != 0) {
            return -1;
        }
    }

    /* --- a whole small SEANet decoder: one shot vs chunked ------------ */
    {
        static const size_t ratios[2] = {2u, 2u};
        mynah_seanet_config config;
        config.channels = 1u;
        config.dimension = 8u;
        config.n_filters = 2u;
        config.n_residual_layers = 1u;
        config.ratios = ratios;
        config.n_ratios = 2u;
        config.kernel_size = 3u;
        config.residual_kernel_size = 3u;
        config.last_kernel_size = 3u;
        config.dilation_base = 2u;
        config.compress = 2u;
        config.elu_alpha = 1.0f;

        mynah_resample_config up;
        up.stride = 2u;
        up.in_channels = 8u;
        up.out_channels = 8u;
        up.groups = 8u;

        const size_t max_latent = 4u;
        const size_t hop = 4u;          /* prod(ratios) */
        const size_t enc_frames = 8u;   /* 4 latent frames * stride 2 */
        const size_t out_len = enc_frames * hop;

        char local[256];
        mynah_seanet_state *one =
            mynah_seanet_state_create(&config, &up, max_latent, local,
                                      sizeof(local));
        mynah_seanet_state *many =
            mynah_seanet_state_create(&config, &up, max_latent, local,
                                      sizeof(local));
        float *input = calloc(config.dimension * enc_frames, sizeof(float));
        float *out_one = calloc(out_len, sizeof(float));
        float *out_many = calloc(out_len, sizeof(float));
        float *piece = calloc(config.dimension * 2u, sizeof(float));
        float *piece_out = calloc(2u * hop, sizeof(float));
        /* decoder weights */
        const size_t mult = 4u; /* 2^n_ratios */
        const size_t c0 = mult * config.n_filters; /* 8 */
        float *w_first = calloc(c0 * config.dimension * config.kernel_size,
                                sizeof(float));
        float *b_first = calloc(c0, sizeof(float));
        float *w_last =
            calloc(config.channels * config.n_filters * config.last_kernel_size,
                   sizeof(float));
        float *b_last = calloc(config.channels, sizeof(float));
        /* stage 0: 8 -> 4, stage 1: 4 -> 2 */
        float *w_ct0 = calloc(8u * 4u * 4u, sizeof(float));
        float *b_ct0 = calloc(4u, sizeof(float));
        float *w_ct1 = calloc(4u * 2u * 4u, sizeof(float));
        float *b_ct1 = calloc(2u, sizeof(float));
        float *w_b0a = calloc(2u * 4u * 3u, sizeof(float)); /* 4 -> 2, k3 */
        float *b_b0a = calloc(2u, sizeof(float));
        float *w_b0b = calloc(4u * 2u * 1u, sizeof(float)); /* 2 -> 4, k1 */
        float *b_b0b = calloc(4u, sizeof(float));
        float *w_b1a = calloc(1u * 2u * 3u, sizeof(float)); /* 2 -> 1, k3 */
        float *b_b1a = calloc(1u, sizeof(float));
        float *w_b1b = calloc(2u * 1u * 1u, sizeof(float)); /* 1 -> 2, k1 */
        float *b_b1b = calloc(2u, sizeof(float));
        int rc = -1;

        if (one == NULL || many == NULL) {
            sea_set_error(error, error_capacity, "seanet decoder create: %s",
                          local);
            goto decoder_done;
        }
        if (input == NULL || out_one == NULL || out_many == NULL ||
            piece == NULL || piece_out == NULL || w_first == NULL ||
            b_first == NULL || w_last == NULL || b_last == NULL ||
            w_ct0 == NULL || b_ct0 == NULL || w_ct1 == NULL || b_ct1 == NULL ||
            w_b0a == NULL || b_b0a == NULL || w_b0b == NULL || b_b0b == NULL ||
            w_b1a == NULL || b_b1a == NULL || w_b1b == NULL || b_b1b == NULL) {
            sea_set_error(error, error_capacity, "seanet decoder: out of "
                                                 "memory");
            goto decoder_done;
        }

#define SEA_FILL(PTR, COUNT, SALT)                            \
    for (size_t fi = 0; fi < (COUNT); ++fi)                   \
        (PTR)[fi] = sea_fake(fi, (SALT));
        SEA_FILL(input, config.dimension * enc_frames, 21u)
        SEA_FILL(w_first, c0 * config.dimension * config.kernel_size, 22u)
        SEA_FILL(b_first, c0, 23u)
        SEA_FILL(w_last,
                 config.channels * config.n_filters * config.last_kernel_size,
                 24u)
        SEA_FILL(b_last, config.channels, 25u)
        SEA_FILL(w_ct0, 8u * 4u * 4u, 26u)
        SEA_FILL(b_ct0, 4u, 27u)
        SEA_FILL(w_ct1, 4u * 2u * 4u, 28u)
        SEA_FILL(b_ct1, 2u, 29u)
        SEA_FILL(w_b0a, 2u * 4u * 3u, 30u)
        SEA_FILL(b_b0a, 2u, 31u)
        SEA_FILL(w_b0b, 4u * 2u, 32u)
        SEA_FILL(b_b0b, 4u, 33u)
        SEA_FILL(w_b1a, 1u * 2u * 3u, 34u)
        SEA_FILL(b_b1a, 1u, 35u)
        SEA_FILL(w_b1b, 2u * 1u, 36u)
        SEA_FILL(b_b1b, 2u, 37u)
#undef SEA_FILL

        mynah_conv_weights convtr_w[2];
        convtr_w[0].weight = w_ct0;
        convtr_w[0].bias = b_ct0;
        convtr_w[1].weight = w_ct1;
        convtr_w[1].bias = b_ct1;
        mynah_seanet_resblock_weights blocks[2];
        blocks[0].conv1.weight = w_b0a;
        blocks[0].conv1.bias = b_b0a;
        blocks[0].conv2.weight = w_b0b;
        blocks[0].conv2.bias = b_b0b;
        blocks[1].conv1.weight = w_b1a;
        blocks[1].conv1.bias = b_b1a;
        blocks[1].conv2.weight = w_b1b;
        blocks[1].conv2.bias = b_b1b;

        mynah_seanet_decoder_weights dw;
        dw.first.weight = w_first;
        dw.first.bias = b_first;
        dw.convtr = convtr_w;
        dw.blocks = blocks;
        dw.last.weight = w_last;
        dw.last.bias = b_last;

        if (mynah_seanet_check_decoder_weights(one, &dw, error,
                                               error_capacity) != 0) {
            goto decoder_done;
        }
        if (mynah_seanet_state_hop_length(one) != hop ||
            mynah_seanet_state_encoder_stride(one) != 2u ||
            mynah_seanet_state_samples_per_latent(one) != 2u * hop) {
            sea_set_error(error, error_capacity,
                          "seanet geometry wrong: hop=%zu stride=%zu",
                          mynah_seanet_state_hop_length(one),
                          mynah_seanet_state_encoder_stride(one));
            goto decoder_done;
        }
        if (mynah_seanet_decode(one, &dw, input, enc_frames, out_one) != 0) {
            sea_set_error(error, error_capacity, "seanet one-shot decode "
                                                 "failed");
            goto decoder_done;
        }
        /* Chunked: one latent frame (2 encoder frames) at a time. */
        {
            int failed = 0;
            for (size_t off = 0; off < enc_frames; off += 2u) {
                for (size_t c = 0; c < config.dimension; ++c) {
                    memcpy(piece + c * 2u, input + c * enc_frames + off,
                           2u * sizeof(float));
                }
                if (mynah_seanet_decode(many, &dw, piece, 2u, piece_out) != 0) {
                    failed = 1;
                    break;
                }
                memcpy(out_many + off * hop, piece_out, 2u * hop *
                                                            sizeof(float));
                mynah_seanet_state_advance(many, 1u);
            }
            if (failed) {
                sea_set_error(error, error_capacity,
                              "seanet chunked decode failed");
                goto decoder_done;
            }
        }
        for (size_t i = 0; i < out_len; ++i) {
            if (!sea_close(out_many[i], (double)out_one[i], 1e-4)) {
                sea_set_error(error, error_capacity,
                              "seanet streaming mismatch at %zu: %.9g vs %.9g",
                              i, (double)out_many[i], (double)out_one[i]);
                goto decoder_done;
            }
        }
        /* The position counter is the other half of the state. */
        if (mynah_seanet_state_position(many) != 4u * 2u) {
            sea_set_error(error, error_capacity,
                          "position counter = %zu, expected %u",
                          mynah_seanet_state_position(many), 8u);
            goto decoder_done;
        }
        if (mynah_seanet_state_position(one) != 0) {
            sea_set_error(error, error_capacity,
                          "position counter advanced without a call");
            goto decoder_done;
        }
        mynah_seanet_state_reset(many);
        if (mynah_seanet_state_position(many) != 0) {
            sea_set_error(error, error_capacity,
                          "reset left the position counter at %zu",
                          mynah_seanet_state_position(many));
            goto decoder_done;
        }

        /* Upsample: frame-by-frame must match the batched call. */
        {
            float *u_in = calloc(up.in_channels * 4u, sizeof(float));
            float *u_batch = calloc(up.out_channels * 4u * up.stride,
                                    sizeof(float));
            float *u_step = calloc(up.out_channels * 4u * up.stride,
                                   sizeof(float));
            float *u_piece_in = calloc(up.in_channels, sizeof(float));
            float *u_piece_out = calloc(up.out_channels * up.stride,
                                        sizeof(float));
            float *u_w = calloc(up.in_channels * (up.out_channels / up.groups) *
                                    up.stride * 2u,
                                sizeof(float));
            int ufail = 0;
            if (u_in == NULL || u_batch == NULL || u_step == NULL ||
                u_piece_in == NULL || u_piece_out == NULL || u_w == NULL) {
                sea_set_error(error, error_capacity, "upsample: out of memory");
                ufail = 1;
            } else {
                for (size_t i = 0; i < up.in_channels * 4u; ++i) {
                    u_in[i] = sea_fake(i, 41u);
                }
                const size_t uw =
                    up.in_channels * (up.out_channels / up.groups) *
                    up.stride * 2u;
                for (size_t i = 0; i < uw; ++i) u_w[i] = sea_fake(i, 42u);
                mynah_conv_weights uww;
                uww.weight = u_w;
                uww.bias = NULL;
                mynah_seanet_state_reset(one);
                if (mynah_seanet_upsample(one, &uww, u_in, 4u, u_batch) != 0) {
                    sea_set_error(error, error_capacity,
                                  "upsample batched failed");
                    ufail = 1;
                }
                if (!ufail) {
                    mynah_seanet_state_reset(many);
                    for (size_t t = 0; t < 4u && !ufail; ++t) {
                        for (size_t c = 0; c < up.in_channels; ++c) {
                            u_piece_in[c] = u_in[c * 4u + t];
                        }
                        if (mynah_seanet_upsample(many, &uww, u_piece_in, 1u,
                                                  u_piece_out) != 0) {
                            sea_set_error(error, error_capacity,
                                          "upsample step failed");
                            ufail = 1;
                            break;
                        }
                        for (size_t c = 0; c < up.out_channels; ++c) {
                            memcpy(u_step + c * 4u * up.stride +
                                       t * up.stride,
                                   u_piece_out + c * up.stride,
                                   up.stride * sizeof(float));
                        }
                    }
                }
                if (!ufail) {
                    for (size_t i = 0; i < up.out_channels * 4u * up.stride;
                         ++i) {
                        if (!sea_close(u_step[i], (double)u_batch[i], 1e-5)) {
                            sea_set_error(error, error_capacity,
                                          "upsample streaming mismatch at %zu: "
                                          "%.9g vs %.9g",
                                          i, (double)u_step[i],
                                          (double)u_batch[i]);
                            ufail = 1;
                            break;
                        }
                    }
                }
            }
            free(u_in);
            free(u_batch);
            free(u_step);
            free(u_piece_in);
            free(u_piece_out);
            free(u_w);
            if (ufail) goto decoder_done;
        }

        rc = 0;
    decoder_done:
        mynah_seanet_state_destroy(one);
        mynah_seanet_state_destroy(many);
        free(input);
        free(out_one);
        free(out_many);
        free(piece);
        free(piece_out);
        free(w_first);
        free(b_first);
        free(w_last);
        free(b_last);
        free(w_ct0);
        free(b_ct0);
        free(w_ct1);
        free(b_ct1);
        free(w_b0a);
        free(b_b0a);
        free(w_b0b);
        free(b_b0b);
        free(w_b1a);
        free(b_b1a);
        free(w_b1b);
        free(b_b1b);
        if (rc != 0) return -1;
    }

    /* --- downsample: replicate padding, streaming ------------------- */
    {
        mynah_resample_config down_cfg;
        down_cfg.stride = 2u;
        down_cfg.in_channels = 3u;
        down_cfg.out_channels = 2u;
        down_cfg.groups = 1u;
        char local[256];
        mynah_seanet_downsample *down =
            mynah_seanet_downsample_create(&down_cfg, 8u, local, sizeof(local));
        if (down == NULL) {
            sea_set_error(error, error_capacity, "downsample create: %s",
                          local);
            return -1;
        }
        const size_t kernel = down_cfg.stride * 2u;
        const size_t wc = down_cfg.out_channels * down_cfg.in_channels * kernel;
        float *w = calloc(wc, sizeof(float));
        float *input = calloc(down_cfg.in_channels * 8u, sizeof(float));
        float *got = calloc(down_cfg.out_channels * 4u, sizeof(float));
        double *padded = calloc(down_cfg.in_channels * (2u + 8u),
                                sizeof(double));
        double *want = calloc(down_cfg.out_channels * 4u, sizeof(double));
        int ok = (w != NULL && input != NULL && got != NULL &&
                  padded != NULL && want != NULL);
        if (ok) {
            for (size_t i = 0; i < wc; ++i) w[i] = sea_fake(i, 51u);
            for (size_t i = 0; i < down_cfg.in_channels * 8u; ++i) {
                input[i] = sea_fake(i, 52u);
            }
            mynah_conv_weights dw;
            dw.weight = w;
            dw.bias = NULL;
            if (mynah_seanet_downsample_apply(down, &dw, input, 8u, got) != 0) {
                sea_set_error(error, error_capacity, "downsample apply failed");
                ok = 0;
            }
            if (ok) {
                mynah_conv1d_spec spec = sea_conv_spec(
                    down_cfg.in_channels, down_cfg.out_channels, kernel,
                    down_cfg.stride, 1u, 1u, MYNAH_CONV_PAD_REPLICATE);
                sea_ref_conv1d(&spec, w, NULL, input, 8u, padded, want);
                for (size_t i = 0; i < down_cfg.out_channels * 4u; ++i) {
                    if (!sea_close(got[i], want[i], 1e-5)) {
                        sea_set_error(error, error_capacity,
                                      "downsample[%zu] = %.9g want %.9g", i,
                                      (double)got[i], want[i]);
                        ok = 0;
                        break;
                    }
                }
            }
        } else {
            sea_set_error(error, error_capacity, "downsample: out of memory");
        }
        mynah_seanet_downsample_destroy(down);
        free(w);
        free(input);
        free(got);
        free(padded);
        free(want);
        if (!ok) return -1;
    }

    return 0;
}

/* ======================================================================
 * Dispatch predicates
 *
 * E4-21.  Every `resolved` below comes from calling something in this file --
 * sea_gemm_enabled(), or the counters the two apply functions actually
 * incremented -- never from re-deriving MYNAH_SEANET_BLAS inside dispatch.c.
 * That is the central rule in dispatch.h: a report that recomputes the
 * condition can agree with the source and both be wrong, which is exactly how
 * "0.427 RTF was VNNI" survived in the README.
 *
 * The counters are zero until a model runs, and `--dispatch-map` loads no
 * model, so these rows say "no SEANet convolution has run in this process"
 * instead of presenting a clean zero as health.  A refusal, per
 * .work/engineering-method.md §4.
 * ====================================================================== */

void mynah_seanet_dispatch_stats_get(mynah_seanet_dispatch_stats *out) {
    if (out == NULL) return;
    out->conv_calls         = sea_read(&g_sea.conv_calls);
    out->conv_gemm          = sea_read(&g_sea.conv_gemm);
    out->conv_scalar_stride = sea_read(&g_sea.conv_scalar_stride);
    out->conv_scalar_groups = sea_read(&g_sea.conv_scalar_groups);
    out->conv_scalar_taps   = sea_read(&g_sea.conv_scalar_taps);
    out->conv_scalar_narrow = sea_read(&g_sea.conv_scalar_narrow);
    out->conv_scalar_nogemm = sea_read(&g_sea.conv_scalar_nogemm);
    out->convtr_calls         = sea_read(&g_sea.convtr_calls);
    out->convtr_gemm          = sea_read(&g_sea.convtr_gemm);
    out->convtr_scalar_groups = sea_read(&g_sea.convtr_scalar_groups);
    out->convtr_scalar_taps   = sea_read(&g_sea.convtr_scalar_taps);
    out->convtr_scalar_narrow = sea_read(&g_sea.convtr_scalar_narrow);
    out->convtr_scalar_nogemm = sea_read(&g_sea.convtr_scalar_nogemm);
}

const char *mynah_seanet_blas_name(void) { return MYNAH_SEANET_BLAS_NAME; }

int mynah_seanet_gemm_enabled(void) { return sea_gemm_enabled(); }

/* codec.seanet_blas -- which BLAS this object linked, and therefore whether
 * the fast paths exist at all.  A value row, not a boolean: "none" and
 * "OpenBLAS" are different facts and ON/OFF would erase the difference. */
static int probe_seanet_blas(char *out, size_t capacity, const char **why) {
#if defined(MYNAH_SEANET_BLAS)
    static char text[240];
#endif
    snprintf(out, capacity, "%s", MYNAH_SEANET_BLAS_NAME);
#if defined(MYNAH_SEANET_OWN_SGEMM)
    snprintf(text, sizeof text,
             "[predicate] src/seanet.c mynah_seanet_blas_name(): sea_sgemm is "
             "mynah_sgemm_f32 (%s kernels, our pool). NO external BLAS is in "
             "this process, and the two call sites here were its entire "
             "PocketTTS surface. The GEMM fold is still the 36x path",
             mynah_sgemm_isa_name());
    *why = text;
#elif defined(MYNAH_SEANET_BLAS)
    snprintf(text, sizeof text,
             "[predicate] src/seanet.c mynah_seanet_blas_name(): sea_sgemm is "
             "%s, so conv1d and convtranspose fold to one GEMM per kernel tap. "
             "This is the 36x path (8570 ms -> 237 ms). A COMPARISON build: "
             "BLAS=none swaps in mynah_sgemm_f32",
             MYNAH_SEANET_BLAS_NAME);
    *why = text;
#else
    *why = "[predicate] src/seanet.c mynah_seanet_blas_name(): NO GEMM here, so "
           "BOTH SEANet fast paths are compiled out and the whole codec conv "
           "stack runs the scalar loops -- 8570 ms vs 237 ms, 36x. Rebuild "
           "with BLAS=none (ours), BLAS=auto or BLAS=openblas";
#endif
    return 0;
}

/* codec.seanet_gemm -- the runtime answer, environment included. */
static int probe_seanet_gemm(const char **why) {
    const int on = mynah_seanet_gemm_enabled();
#if defined(MYNAH_SEANET_BLAS)
    *why = on ? "[predicate] src/seanet.c sea_gemm_enabled(): ON. A call still "
                "needs stride==1 and groups==1 (conv1d) or groups==1 "
                "(convtranspose) plus a tap buffer; codec.seanet_conv_path "
                "counts how many actually qualified"
              : "[predicate] src/seanet.c sea_gemm_enabled(): MYNAH_SEANET_GEMM "
                "turned the GEMM off, so every SEANet convolution takes the "
                "scalar reference. Deliberate, and 36x slower -- unset it";
#else
    *why = "[predicate] src/seanet.c sea_gemm_enabled(): OFF because no BLAS "
           "was compiled in (BLAS=scalar). Not an env choice; a build choice";
#endif
    return on;
}

/* One row per call site, saying what RAN rather than what could run:
 * `gemm/total`, plus the reason each refusal fired. */
static int probe_seanet_conv_path(char *out, size_t capacity, const char **why) {
    static char text[240];
    mynah_seanet_dispatch_stats st;
    mynah_seanet_dispatch_stats_get(&st);
    if (st.conv_calls == 0) {
        snprintf(out, capacity, "n/a");
        *why = "[predicate] src/seanet.c: no causal conv1d has run in this "
               "process yet, so there is nothing to report. Read this row "
               "after a synthesis; --dispatch-map alone loads no model";
        return 0;
    }
    snprintf(out, capacity, "%llu/%llu", st.conv_gemm, st.conv_calls);
    snprintf(text, sizeof text,
             "[predicate] src/seanet.c mynah_causal_conv1d_apply(): %llu of "
             "%llu calls folded to sgemm. Scalar: %llu stride!=1, %llu "
             "grouped, %llu no tap buffer, %llu dim>INT_MAX, %llu no GEMM "
             "compiled or enabled",
             st.conv_gemm, st.conv_calls, st.conv_scalar_stride,
             st.conv_scalar_groups, st.conv_scalar_taps, st.conv_scalar_narrow,
             st.conv_scalar_nogemm);
    *why = text;
    return 0;
}

static int probe_seanet_convtr_path(char *out, size_t capacity,
                                    const char **why) {
    static char text[240];
    mynah_seanet_dispatch_stats st;
    mynah_seanet_dispatch_stats_get(&st);
    if (st.convtr_calls == 0) {
        snprintf(out, capacity, "n/a");
        *why = "[predicate] src/seanet.c: no causal convtranspose1d has run in "
               "this process yet. Read this row after a synthesis";
        return 0;
    }
    snprintf(out, capacity, "%llu/%llu", st.convtr_gemm, st.convtr_calls);
    snprintf(text, sizeof text,
             "[predicate] src/seanet.c mynah_causal_convtr1d_apply(): %llu of "
             "%llu folded to sgemm. Scalar: %llu grouped (a grouped "
             "convtranspose -- the Mimi depthwise upsample -- can never fold, "
             "so it scatters every frame), %llu no taps, %llu dim>INT_MAX, "
             "%llu no GEMM",
             st.convtr_gemm, st.convtr_calls, st.convtr_scalar_groups,
             st.convtr_scalar_taps, st.convtr_scalar_narrow,
             st.convtr_scalar_nogemm);
    *why = text;
    return 0;
}

void mynah_seanet_dispatch_probes(void) {
    mynah_dispatch_register_value_probe("codec.seanet_blas", probe_seanet_blas);
    mynah_dispatch_register_probe("codec.seanet_gemm", probe_seanet_gemm);
    mynah_dispatch_register_value_probe("codec.seanet_conv_path",
                                        probe_seanet_conv_path);
    mynah_dispatch_register_value_probe("codec.seanet_convtr_path",
                                        probe_seanet_convtr_path);
}
