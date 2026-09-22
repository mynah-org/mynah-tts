/* E14-4.  THIS TRANSLATION UNIT IS COMPILED MORE THAN ONCE ON x86.
 *
 * Every other runtime kernel choice in this project is one function with a
 * target attribute and a CPUID probe.  That does not work here, and the reason
 * is in the file rather than in the loop: SG_LANES is not confined to the inner
 * kernel.  It reaches SG_NV_MAX, SG_NARROW_MAX, the packed panel geometry and
 * the PUBLIC mynah_sgemm_narrow_max(), so an AVX2 micro-kernel wearing a target
 * attribute would still be fed panels packed for the baseline shape.  The ISA
 * is in the DATA LAYOUT here, not only in the instructions.
 *
 * So on x86 the whole file is built twice -- once at the build's baseline and
 * once with -mavx2 -mfma -- and src/sgemm_rt.c owns the public names and picks
 * between them once per process.  Each variant's constants are its own, its
 * packing matches its own kernel, and no panel crosses from one to the other.
 * Everything that made the single-TU design good survives: one micro-kernel
 * body, one accumulation order, scalar as the reference rather than a second
 * algorithm.
 *
 * On aarch64 nothing changes and nothing is built twice.  AdvSIMD is
 * architecturally guaranteed, so the compile-time choice is the right one and
 * there is no second variant to select.
 *
 * MYNAH_SGEMM_VARIANT renames the twelve public symbols.  It is defined for
 * BOTH x86 variants -- `base` and `avx2` -- because if the baseline build kept
 * the plain names it would collide with the dispatcher that has to own them. */
#if defined(MYNAH_SGEMM_VARIANT)
#define MYNAH_SGEMM_J2(a, b) a##_##b
#define MYNAH_SGEMM_J1(a, b) MYNAH_SGEMM_J2(a, b)
#define MYNAH_SGEMM_SYM(n)   MYNAH_SGEMM_J1(n, MYNAH_SGEMM_VARIANT)
#define mynah_sgemm_f32                    MYNAH_SGEMM_SYM(mynah_sgemm_f32)
#define mynah_sgemm_self_test              MYNAH_SGEMM_SYM(mynah_sgemm_self_test)
#define mynah_sgemm_f32_conv_taps          MYNAH_SGEMM_SYM(mynah_sgemm_f32_conv_taps)
#define mynah_sgemm_f32_reference          MYNAH_SGEMM_SYM(mynah_sgemm_f32_reference)
#define mynah_sgemm_narrow_max             MYNAH_SGEMM_SYM(mynah_sgemm_narrow_max)
#define mynah_sgemm_family_for             MYNAH_SGEMM_SYM(mynah_sgemm_family_for)
#define mynah_sgemm_family_name            MYNAH_SGEMM_SYM(mynah_sgemm_family_name)
#define mynah_sgemm_isa_name               MYNAH_SGEMM_SYM(mynah_sgemm_isa_name)
#define mynah_sgemm_stats_get              MYNAH_SGEMM_SYM(mynah_sgemm_stats_get)
#define mynah_sgemm_stats_reset            MYNAH_SGEMM_SYM(mynah_sgemm_stats_reset)
#define mynah_sgemm_f32_forced             MYNAH_SGEMM_SYM(mynah_sgemm_f32_forced)
#define mynah_sgemm_dispatch_probes        MYNAH_SGEMM_SYM(mynah_sgemm_dispatch_probes)
#endif

/*
 * mynah_sgemm_f32 -- our own f32 GEMM.  See sgemm.h for why it exists and for
 * the measured shape histogram that shaped it.
 *
 * ------------------------------------------------------------------ layout
 *
 * Everything here is row-major and follows cblas_sgemm's argument contract
 * exactly, because the point is to be droppable into the three call sites that
 * used it.  op(A) is m x k, op(B) is k x n, C is m x n.
 *
 * -------------------------------------------------------------- determinism
 *
 * THE RESULT DOES NOT DEPEND ON THE THREAD COUNT, by construction, not by
 * tolerance.  Three properties give that, and all three are load-bearing:
 *
 *  1. The reduction over k is NEVER split.  One output element is accumulated
 *     by exactly one micro-kernel invocation, over p = 0..k-1 in order, in a
 *     register.  There is no k-blocking anywhere in this file.
 *  2. The COLUMN blocking depends only on the shape (n, k) and the compiled
 *     ISA -- never on mynah_num_threads().  So which column group, and hence
 *     which micro-kernel instantiation, covers a given column is fixed.
 *  3. The ROW blocking is the only axis the thread count touches, and the row
 *     block is always rounded UP to SG_MR.  Every block therefore starts at a
 *     multiple of SG_MR, only the final block can carry a ragged remainder,
 *     and that remainder is m % SG_MR whatever the thread count is.  So a
 *     given row is handled by the 4-row strip or the 1-row strip regardless of
 *     how the work was split.
 *
 * Property 3 is the subtle one.  Without the round-up, splitting m=6 into two
 * blocks of 3 would send rows 0..2 through the 1-row kernel where a single
 * block sent them through the 4-row kernel -- two different function
 * instantiations of the same macro, which clang is free to contract
 * differently under -ffast-math.  The self-test checks the property directly.
 *
 * ------------------------------------------------------- numerical contract
 *
 * Against mynah_sgemm_f32_reference() the kernels here differ only by FMA
 * contraction: the accumulation order over p is identical (sequential, one
 * accumulator per output element).  They are NOT asserted bit-identical to it,
 * because -ffast-math implies -fassociative-math and clang regroups a
 * multi-factor float product differently in two textually identical code
 * paths; the self-test uses a stated relative bound instead.  Two invocations
 * of the SAME instantiation with different block offsets ARE asserted
 * bit-identical, because that is the same machine code with different
 * arguments, and that assertion is the proof of the determinism claim above.
 *
 * Against Accelerate or OpenBLAS the difference is larger, because those DO
 * reassociate: replacing them changes the codec's f32 output slightly.  That
 * is a numerical change and is qualified as one -- against the SEANet oracle
 * tolerance, never waved through.
 */
#include "sgemm.h"

#include "dispatch.h"
#include "kernels.h"
#include "threads.h"

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ======================================================================
 * ISA abstraction
 *
 * One micro-kernel body, three instruction sets.  The scalar build is not a
 * separate algorithm: SG_LANES == 1 makes a "vector" one float, so the scalar
 * path executes the same loop nest with the same accumulation order.  That is
 * AGENTS.md coding rule 5 expressed in code rather than in a comment -- there
 * is no second formula to keep in sync.
 * ====================================================================== */
#if !defined(MYNAH_DISABLE_SIMD) && (defined(__ARM_NEON) || defined(__aarch64__))
#include <arm_neon.h>
#define SG_ISA_NAME "neon"
#define SG_LANES 4
typedef float32x4_t sg_vec;
#define sg_zero()        vdupq_n_f32(0.0f)
#define sg_load(p)       vld1q_f32(p)
#define sg_store(p, v)   vst1q_f32((p), (v))
#define sg_dup(x)        vdupq_n_f32(x)
#define sg_mul(a, b)     vmulq_f32((a), (b))
#define sg_fma(acc, a, b) vfmaq_f32((acc), (a), (b))
/* 32 architectural v-registers: half of them may hold accumulators. */
#define SG_ACC_VECS 16

#elif !defined(MYNAH_DISABLE_SIMD) && defined(__AVX2__)
#include <immintrin.h>
#define SG_ISA_NAME "avx2"
#define SG_LANES 8
typedef __m256 sg_vec;
#define sg_zero()        _mm256_setzero_ps()
#define sg_load(p)       _mm256_loadu_ps(p)
#define sg_store(p, v)   _mm256_storeu_ps((p), (v))
#define sg_dup(x)        _mm256_set1_ps(x)
#define sg_mul(a, b)     _mm256_mul_ps((a), (b))
#if defined(__FMA__)
#define sg_fma(acc, a, b) _mm256_fmadd_ps((a), (b), (acc))
#else
#define sg_fma(acc, a, b) _mm256_add_ps((acc), _mm256_mul_ps((a), (b)))
#endif
/* 16 architectural ymm registers: half of them may hold accumulators. */
#define SG_ACC_VECS 8

#else
#define SG_ISA_NAME "scalar"
#define SG_LANES 1
typedef float sg_vec;
#define sg_zero()        0.0f
#define sg_load(p)       (*(p))
#define sg_store(p, v)   (*(p) = (v))
#define sg_dup(x)        (x)
#define sg_mul(a, b)     ((a) * (b))
#define sg_fma(acc, a, b) ((acc) + (a) * (b))
#define SG_ACC_VECS 16
#endif

/* Micro-kernel rows.  Four independent accumulator chains per column vector is
 * enough to cover the FMA latency on every core this runtime targets, and it
 * keeps the b operand loaded once per four FMAs. */
#define SG_MR 4

/* The narrow/panel boundary, DERIVED (see mynah_sgemm_narrow_max in sgemm.h):
 * the widest n whose whole C row block still fits in the accumulator budget.
 * NEON 16, AVX2 16, scalar 4.  The measured 28%-of-calls shape has n = 16. */
#define SG_NV_MAX     (SG_ACC_VECS / SG_MR)
#define SG_NARROW_MAX (SG_NV_MAX * SG_LANES)

/* Panel family: accumulator vectors per column group.  On AVX2 this is forced
 * -- SG_NV_MAX is already 2 with 16 ymm registers.  On NEON it is a CHOICE
 * between two effects pulling opposite ways, and it has NOT been measured:
 *
 *   wider groups  -> fewer passes over op(A), and a better FMA-per-load ratio
 *   narrower ones -> a smaller ragged remainder when n % (NV*LANES) != 0, and
 *                    more registers free for the op(B) stream
 *
 * 2 is the conservative end. The sweep belongs on the Linux hosts with the
 * default flip, not on a development Mac; until then this is a cost-model
 * guess and is labelled as one rather than presented as a tuned constant. */
#define SG_PANEL_NV 2
#define SG_NR       (SG_PANEL_NV * SG_LANES)

/* op(B) panel budget for the panel family, in floats: the panel is k x nc and
 * we want it to stay in L2 while the rows of op(A) sweep past.  128 KiB leaves
 * room for the A strip and the C block in a 256-512 KiB private L2.  This is a
 * COST-MODEL estimate, not a measurement: it has never been swept on the
 * production Linux hosts, and the sweep belongs with the default flip. */
#define SG_PANEL_FLOATS 32768u

/* Task granularity for the two families that have no row axis to split.
 * m == 1 (129 calls per utterance of m=1 n=1920 k=64) parallelises over
 * columns only; the dot family is coarse because each task is n*m calls into
 * mynah_dot_f32. */
#define SG_MATVEC_NC 256u
#define SG_DOT_NC    64u

/* Below this much work the blocked path's own bookkeeping -- the plan, the job
 * struct, the pool's dispatch check -- is a visible fraction of the
 * arithmetic, so the reference is both simpler and faster.  4096 MACs is a
 * 16x16x16 GEMM.  Cost-model estimate; the self-test only depends on it
 * through the coverage refusal, never on its exact value. */
#define SG_MIN_BLOCKED_WORK 4096u

/* Below this much work the GEMM runs as ONE task.  A pool dispatch costs on
 * the order of a microsecond and 131072 MACs is a few tens of microseconds of
 * arithmetic, so this is the point where splitting starts to pay.  Also a
 * cost-model estimate; it must be re-derived on the Linux hosts, where both
 * the dispatch cost and the core count differ. */
#define SG_PARALLEL_MIN_WORK 131072u

/* ======================================================================
 * Counters
 *
 * What RAN, not what could run.  Relaxed atomics, one increment per GEMM call
 * (order tens per frame), never per element and never inside a loop -- the
 * same contract src/seanet.c's counters use.
 * ====================================================================== */
typedef struct {
    atomic_ullong calls;
    atomic_ullong reference;
    atomic_ullong dot;
    atomic_ullong matvec;
    atomic_ullong narrow;
    atomic_ullong panel;
    atomic_ullong refused;
    atomic_ullong fused;
    atomic_ullong fused_taps;
    atomic_ullong fused_refused;
} sg_counters;

static sg_counters g_sg;

static void sg_bump(atomic_ullong *c) {
    atomic_fetch_add_explicit(c, 1ull, memory_order_relaxed);
}

static unsigned long long sg_read(atomic_ullong *c) {
    return atomic_load_explicit(c, memory_order_relaxed);
}

void mynah_sgemm_stats_get(mynah_sgemm_stats *out) {
    if (out == NULL) return;
    out->calls     = sg_read(&g_sg.calls);
    out->reference = sg_read(&g_sg.reference);
    out->dot       = sg_read(&g_sg.dot);
    out->matvec    = sg_read(&g_sg.matvec);
    out->narrow    = sg_read(&g_sg.narrow);
    out->panel     = sg_read(&g_sg.panel);
    out->refused   = sg_read(&g_sg.refused);
    out->fused         = sg_read(&g_sg.fused);
    out->fused_taps    = sg_read(&g_sg.fused_taps);
    out->fused_refused = sg_read(&g_sg.fused_refused);
}

void mynah_sgemm_stats_reset(void) {
    atomic_store_explicit(&g_sg.calls, 0ull, memory_order_relaxed);
    atomic_store_explicit(&g_sg.reference, 0ull, memory_order_relaxed);
    atomic_store_explicit(&g_sg.dot, 0ull, memory_order_relaxed);
    atomic_store_explicit(&g_sg.matvec, 0ull, memory_order_relaxed);
    atomic_store_explicit(&g_sg.narrow, 0ull, memory_order_relaxed);
    atomic_store_explicit(&g_sg.panel, 0ull, memory_order_relaxed);
    atomic_store_explicit(&g_sg.refused, 0ull, memory_order_relaxed);
    atomic_store_explicit(&g_sg.fused, 0ull, memory_order_relaxed);
    atomic_store_explicit(&g_sg.fused_taps, 0ull, memory_order_relaxed);
    atomic_store_explicit(&g_sg.fused_refused, 0ull, memory_order_relaxed);
}

/* ======================================================================
 * Shape histogram -- E: what ACTUALLY ran, per shape
 *
 * sgemm.h states a measured histogram ("28% of calls are m=512 n=16 k=512")
 * that was collected once, by hand, on a different question.  A scaling
 * investigation needs the histogram AS EXECUTED, with per-shape wall time and
 * the task count each call asked the pool for, because "how many dispatches"
 * and "how long is one dispatch" are the two numbers that decide whether a
 * barrier is the ceiling.  Deriving them from the source is exactly the
 * mistake dispatch.h exists to prevent.
 *
 * A fixed table, no allocation, linear probe on a 5-tuple key.  OFF unless
 * MYNAH_SGEMM_PROFILE is set to something other than "0"; when off this costs
 * one relaxed load and a predictable branch per GEMM CALL (not per element,
 * not per task).  A full table stops recording and says so rather than
 * evicting, because a histogram that silently drops its tail is worse than
 * one that refuses.
 * ====================================================================== */
#define SG_HIST_MAX 96

typedef struct {
    atomic_int used;
    int trans_a, trans_b;
    size_t m, n, k;
    unsigned long long calls;
    unsigned long long tasks;     /* summed: tasks asked of the pool         */
    unsigned long long ns;        /* summed wall, caller's view              */
    unsigned long long ns_min;
    unsigned long long ns_max;
    int family;
} sg_hist_row;

static sg_hist_row g_sg_hist[SG_HIST_MAX];
static atomic_int g_sg_hist_full;
static atomic_flag g_sg_hist_lock = ATOMIC_FLAG_INIT;
static atomic_int g_sg_prof_state = -1;
static atomic_int g_sg_prof_hooked;

static void sg_hist_report(void);

static int sg_prof_on(void) {
    int state = atomic_load_explicit(&g_sg_prof_state, memory_order_relaxed);
    if (state >= 0) return state;
    const char *env = getenv("MYNAH_SGEMM_PROFILE");
    state = (env != NULL && env[0] != '\0' && strcmp(env, "0") != 0) ? 1 : 0;
    atomic_store_explicit(&g_sg_prof_state, state, memory_order_relaxed);
    if (state) {
        int expected = 0;
        if (atomic_compare_exchange_strong(&g_sg_prof_hooked, &expected, 1))
            (void)atexit(sg_hist_report);
    }
    return state;
}

static unsigned long long sg_now(void) {
    if (!sg_prof_on()) return 0ull;
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (unsigned long long)ts.tv_sec * 1000000000ull +
           (unsigned long long)ts.tv_nsec;
}

/* One spinlock, taken once per GEMM call and only while profiling.  The alt-
 * ernative -- per-row atomics -- would make the min/max racy for no benefit:
 * this path never runs in production. */
static void sg_hist_add(unsigned long long start, int trans_a, int trans_b,
                        size_t m, size_t n, size_t k, int family,
                        size_t tasks) {
    if (start == 0ull) return;
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    const unsigned long long dt =
        (unsigned long long)ts.tv_sec * 1000000000ull +
        (unsigned long long)ts.tv_nsec - start;
    while (atomic_flag_test_and_set_explicit(&g_sg_hist_lock,
                                             memory_order_acquire)) { }
    sg_hist_row *slot = NULL;
    for (int i = 0; i < SG_HIST_MAX; ++i) {
        sg_hist_row *r = &g_sg_hist[i];
        if (!atomic_load_explicit(&r->used, memory_order_relaxed)) {
            slot = r;
            break;
        }
        if (r->trans_a == trans_a && r->trans_b == trans_b && r->m == m &&
            r->n == n && r->k == k) {
            slot = r;
            break;
        }
    }
    if (slot == NULL) {
        atomic_store_explicit(&g_sg_hist_full, 1, memory_order_relaxed);
    } else if (!atomic_load_explicit(&slot->used, memory_order_relaxed)) {
        slot->trans_a = trans_a; slot->trans_b = trans_b;
        slot->m = m; slot->n = n; slot->k = k;
        slot->calls = 1ull; slot->tasks = (unsigned long long)tasks;
        slot->ns = dt; slot->ns_min = dt; slot->ns_max = dt;
        slot->family = family;
        atomic_store_explicit(&slot->used, 1, memory_order_relaxed);
    } else {
        slot->calls += 1ull;
        slot->tasks += (unsigned long long)tasks;
        slot->ns += dt;
        if (dt < slot->ns_min) slot->ns_min = dt;
        if (dt > slot->ns_max) slot->ns_max = dt;
    }
    atomic_flag_clear_explicit(&g_sg_hist_lock, memory_order_release);
}

static void sg_hist_report(void) {
    unsigned long long calls = 0, ns = 0, tasks = 0;
    for (int i = 0; i < SG_HIST_MAX; ++i) {
        if (!atomic_load_explicit(&g_sg_hist[i].used, memory_order_relaxed))
            continue;
        calls += g_sg_hist[i].calls;
        ns    += g_sg_hist[i].ns;
        tasks += g_sg_hist[i].tasks;
    }
    fprintf(stderr,
            "[SGEMM-SHAPES] isa=%s threads=%d narrow_max=%zu -- AS EXECUTED, "
            "not as predicted.\n"
            "  ns is the CALLER's wall for the whole call: wake-up + its own "
            "share + barrier.  tasks/call is what the pool was asked for.\n",
            SG_ISA_NAME, mynah_num_threads(), (size_t)SG_NARROW_MAX);
    if (atomic_load_explicit(&g_sg_hist_full, memory_order_relaxed))
        fprintf(stderr, "  WARNING: table full, some shapes not recorded.\n");
    fprintf(stderr, "  %2s %2s %6s %6s %6s %8s %9s %7s %10s %9s %9s %7s\n",
            "tA", "tB", "m", "n", "k", "calls", "family", "tasks", "total_ms",
            "mean_us", "min_us", "%ns");
    for (int i = 0; i < SG_HIST_MAX; ++i) {
        sg_hist_row *r = &g_sg_hist[i];
        if (!atomic_load_explicit(&r->used, memory_order_relaxed)) continue;
        fprintf(stderr,
                "  %2d %2d %6zu %6zu %6zu %8llu %9s %7.1f %10.3f %9.2f %9.2f "
                "%6.2f%%\n",
                r->trans_a, r->trans_b, r->m, r->n, r->k, r->calls,
                mynah_sgemm_family_name((mynah_sgemm_family)r->family),
                (double)r->tasks / (double)r->calls, (double)r->ns / 1e6,
                (double)r->ns / (double)r->calls / 1e3,
                (double)r->ns_min / 1e3,
                ns ? 100.0 * (double)r->ns / (double)ns : 0.0);
    }
    long long disp = 0, serial = 0, inl = 0, joins = 0;
    mynah_parallel_stats(&disp, &serial, &inl, &joins);
    fprintf(stderr,
            "  totals: calls=%llu wall=%.3f ms tasks=%llu (%.1f/call)  "
            "pool: dispatches=%lld serial=%lld inline_fallbacks=%lld "
            "helper_joins=%lld\n",
            calls, (double)ns / 1e6, tasks,
            calls ? (double)tasks / (double)calls : 0.0, disp, serial, inl,
            joins);
}

/* ======================================================================
 * The reference -- the definition of correctness
 * ====================================================================== */
void mynah_sgemm_f32_reference(int trans_a, int trans_b,
                               size_t m, size_t n, size_t k,
                               float alpha,
                               const float *a, size_t lda,
                               const float *b, size_t ldb,
                               float beta,
                               float *c, size_t ldc) {
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (size_t p = 0; p < k; ++p) {
                const float av = trans_a ? a[p * lda + i] : a[i * lda + p];
                const float bv = trans_b ? b[j * ldb + p] : b[p * ldb + j];
                sum += av * bv;
            }
            float *cp = c + i * ldc + j;
            /* beta == 0 must not READ C: cblas guarantees an uninitialised or
             * NaN-carrying C is overwritten, and the triple loop this replaces
             * in backend.c did read it. */
            *cp = (beta == 0.0f) ? alpha * sum : alpha * sum + beta * *cp;
        }
    }
}

/* ======================================================================
 * Micro-kernels
 *
 * One body, instantiated for {1, 4} rows x {1, 2, 4} column vectors.  Each
 * invocation owns MROWS * NVEC * SG_LANES output elements, accumulates every
 * one of them over the whole of k in registers, and stores once.
 *
 * `ap` walks op(A): ars is the step between rows, acs the step along k.  With
 * trans_a == 0 that is (lda, 1) and with trans_a == 1 it is (1, lda), so no
 * operand is ever copied or transposed -- the transpose is two strides.
 * ====================================================================== */

#define SG_STORE_TILE(MROWS, NVEC, ACC, ALPHA, BETA, C, LDC)                   \
    do {                                                                       \
        const sg_vec sg_a_ = sg_dup(ALPHA);                                    \
        if ((BETA) == 0.0f) {                                                  \
            for (int r_ = 0; r_ < (MROWS); ++r_)                               \
                for (int v_ = 0; v_ < (NVEC); ++v_)                            \
                    sg_store((C) + (size_t)r_ * (LDC) +                        \
                                 (size_t)v_ * SG_LANES,                        \
                             sg_mul((ACC)[r_][v_], sg_a_));                    \
        } else {                                                               \
            const sg_vec sg_b_ = sg_dup(BETA);                                 \
            for (int r_ = 0; r_ < (MROWS); ++r_)                               \
                for (int v_ = 0; v_ < (NVEC); ++v_) {                          \
                    float *cp_ = (C) + (size_t)r_ * (LDC) +                    \
                                 (size_t)v_ * SG_LANES;                        \
                    sg_store(cp_, sg_fma(sg_mul((ACC)[r_][v_], sg_a_),         \
                                         sg_b_, sg_load(cp_)));                \
                }                                                              \
        }                                                                      \
    } while (0)

#define SG_DEFINE_MICRO(NAME, MROWS, NVEC)                                     \
    static void NAME(size_t k, const float *ap, size_t ars, size_t acs,        \
                     const float *bp, size_t ldbp, float alpha, float beta,    \
                     float *c, size_t ldc) {                                   \
        sg_vec acc[MROWS][NVEC];                                               \
        const float *arow[MROWS];                                              \
        for (int r = 0; r < (MROWS); ++r) {                                    \
            arow[r] = ap + (size_t)r * ars;                                    \
            for (int v = 0; v < (NVEC); ++v) acc[r][v] = sg_zero();            \
        }                                                                      \
        for (size_t p = 0; p < k; ++p) {                                       \
            sg_vec bv[NVEC];                                                   \
            for (int v = 0; v < (NVEC); ++v)                                   \
                bv[v] = sg_load(bp + (size_t)v * SG_LANES);                    \
            for (int r = 0; r < (MROWS); ++r) {                                \
                const sg_vec av = sg_dup(*arow[r]);                            \
                arow[r] += acs;                                                \
                for (int v = 0; v < (NVEC); ++v)                               \
                    acc[r][v] = sg_fma(acc[r][v], av, bv[v]);                  \
            }                                                                  \
            bp += ldbp;                                                        \
        }                                                                      \
        SG_STORE_TILE(MROWS, NVEC, acc, alpha, beta, c, ldc);                  \
    }

SG_DEFINE_MICRO(sg_micro4_1, SG_MR, 1)
SG_DEFINE_MICRO(sg_micro4_2, SG_MR, 2)
SG_DEFINE_MICRO(sg_micro4_4, SG_MR, 4)
SG_DEFINE_MICRO(sg_micro1_1, 1, 1)
SG_DEFINE_MICRO(sg_micro1_2, 1, 2)
SG_DEFINE_MICRO(sg_micro1_4, 1, 4)

/* Column remainder: fewer than SG_LANES columns left.  Scalar, same
 * accumulation order as the vector bodies, so its results are consistent with
 * the rest of the row.  Never reached on the scalar build, where SG_LANES is
 * 1 and there is no remainder. */
static void sg_micro_tail(size_t rows, size_t cols, size_t k, const float *ap,
                          size_t ars, size_t acs, const float *bp, size_t ldbp,
                          float alpha, float beta, float *c, size_t ldc) {
    float acc[SG_MR][SG_LANES];
    for (size_t r = 0; r < rows; ++r)
        for (size_t j = 0; j < cols; ++j) acc[r][j] = 0.0f;
    for (size_t p = 0; p < k; ++p) {
        for (size_t r = 0; r < rows; ++r) {
            const float av = ap[r * ars + p * acs];
            for (size_t j = 0; j < cols; ++j)
                acc[r][j] += av * bp[p * ldbp + j];
        }
    }
    for (size_t r = 0; r < rows; ++r) {
        for (size_t j = 0; j < cols; ++j) {
            float *cp = c + r * ldc + j;
            *cp = (beta == 0.0f) ? alpha * acc[r][j]
                                 : alpha * acc[r][j] + beta * *cp;
        }
    }
}

/* A strip of MROWS rows across `cols` columns.  The column decomposition is
 * greedy over the available instantiations and depends only on (cols, nv_max),
 * which is why it is thread-count independent. */
#define SG_DEFINE_STRIP(NAME, MROWS, M4, M2, M1)                               \
    static void NAME(size_t k, size_t cols, size_t nv_max, const float *ap,    \
                     size_t ars, size_t acs, const float *bp, size_t ldbp,     \
                     float alpha, float beta, float *c, size_t ldc) {          \
        size_t j = 0;                                                          \
        if (nv_max >= 4u)                                                      \
            for (; j + 4u * SG_LANES <= cols; j += 4u * SG_LANES)              \
                M4(k, ap, ars, acs, bp + j, ldbp, alpha, beta, c + j, ldc);    \
        if (nv_max >= 2u)                                                      \
            for (; j + 2u * SG_LANES <= cols; j += 2u * SG_LANES)              \
                M2(k, ap, ars, acs, bp + j, ldbp, alpha, beta, c + j, ldc);    \
        for (; j + SG_LANES <= cols; j += SG_LANES)                            \
            M1(k, ap, ars, acs, bp + j, ldbp, alpha, beta, c + j, ldc);        \
        if (j < cols)                                                          \
            sg_micro_tail(MROWS, cols - j, k, ap, ars, acs, bp + j, ldbp,      \
                          alpha, beta, c + j, ldc);                            \
    }

SG_DEFINE_STRIP(sg_strip4, SG_MR, sg_micro4_4, sg_micro4_2, sg_micro4_1)
SG_DEFINE_STRIP(sg_strip1, 1, sg_micro1_4, sg_micro1_2, sg_micro1_1)

/* ======================================================================
 * The job and its tiles
 * ====================================================================== */
typedef struct {
    int trans_a;
    int trans_b;
    size_t m, n, k;
    float alpha, beta;
    const float *a;
    size_t lda;
    const float *b;
    size_t ldb;
    float *c;
    size_t ldc;
    /* plan */
    mynah_sgemm_family family;
    size_t nv_max;    /* accumulator vectors per column group */
    size_t nc;        /* columns per task                     */
    size_t row_block; /* rows per task, a multiple of SG_MR    */
    size_t grid_n;    /* column blocks                        */
    size_t grid_m;    /* row blocks                           */
    /* conv-tap fusion; see mynah_sgemm_f32_conv_taps in sgemm.h.  taps == 0
     * is an ordinary GEMM and none of the rest is read. */
    int serial;       /* the plan asked for one task: run inline */
    size_t taps;
    const float *w_taps;    /* [m][k][taps]                   */
    float *gather;          /* [m][k] scratch, row-block private */
    size_t b_tap_stride;
} sg_job;

/* op(B) not transposed: columns of op(B) are contiguous, so the micro-kernels
 * vectorise across n and no operand is copied. */
static void sg_tile_nn(const sg_job *j, size_t i0, size_t rows, size_t j0,
                       size_t cols) {
    const size_t ars = j->trans_a ? 1u : j->lda;
    const size_t acs = j->trans_a ? j->lda : 1u;
    const float *abase = j->a + (j->trans_a ? i0 : i0 * j->lda);
    const float *bbase = j->b + j0;
    float *cbase = j->c + i0 * j->ldc + j0;

    size_t i = 0;
    for (; i + SG_MR <= rows; i += SG_MR)
        sg_strip4(j->k, cols, j->nv_max, abase + i * ars, ars, acs, bbase,
                  j->ldb, j->alpha, j->beta, cbase + i * j->ldc, j->ldc);
    for (; i < rows; ++i)
        sg_strip1(j->k, cols, j->nv_max, abase + i * ars, ars, acs, bbase,
                  j->ldb, j->alpha, j->beta, cbase + i * j->ldc, j->ldc);
}

/* op(B) transposed: op(B)'s column j is B's ROW j, contiguous in k.  There is
 * nothing to vectorise across n, so each output element is one dot product --
 * over the existing, already-tuned mynah_dot_f32 rather than a fourth kernel
 * of our own.  Only reached with trans_a == 0, where op(A)'s rows are
 * contiguous too; the transposed-both case has no contiguous axis at all and
 * the predicate sends it to the reference. */
static void sg_tile_dot(const sg_job *j, size_t i0, size_t rows, size_t j0,
                        size_t cols) {
    for (size_t i = 0; i < rows; ++i) {
        const float *arow = j->a + (i0 + i) * j->lda;
        float *crow = j->c + (i0 + i) * j->ldc + j0;
        for (size_t jj = 0; jj < cols; ++jj) {
            const float s = mynah_dot_f32(arow, j->b + (j0 + jj) * j->ldb, j->k);
            crow[jj] = (j->beta == 0.0f) ? j->alpha * s
                                         : j->alpha * s + j->beta * crow[jj];
        }
    }
}

static void sg_task(void *ctx, int index) {
    const sg_job *j = (const sg_job *)ctx;
    const size_t bi = (size_t)index / j->grid_n;
    const size_t bj = (size_t)index % j->grid_n;
    const size_t i0 = bi * j->row_block;
    const size_t j0 = bj * j->nc;
    if (i0 >= j->m || j0 >= j->n) return;
    size_t rows = j->row_block;
    if (i0 + rows > j->m) rows = j->m - i0;
    size_t cols = j->nc;
    if (j0 + cols > j->n) cols = j->n - j0;
    if (j->family == MYNAH_SGEMM_FAMILY_DOT) sg_tile_dot(j, i0, rows, j0, cols);
    else sg_tile_nn(j, i0, rows, j0, cols);
}

/* One task of the fused conv-tap region.
 *
 * A TASK IS A ROW BLOCK, not a (row block, column group) tile, and that is the
 * one structural difference from sg_task.  The gather is per-ROW: if two tasks
 * shared a row block they would race to gather the same rows of the weight
 * into the same scratch.  Owning the whole row -- all column groups -- removes
 * the sharing instead of synchronising it, and costs nothing numerically,
 * because which thread computes a tile has never affected its value.  The
 * COLUMN plan is untouched, so every tile is still the same tile, on the same
 * micro-kernel instantiation, as the per-tap calls produced.
 *
 * The tap loop is outermost so the beta chain still rounds each tap's partial
 * sum to f32 before the next is added -- the property that makes this
 * byte-identical rather than merely equivalent. */
static void sg_task_taps(void *ctx, int index) {
    const sg_job *j = (const sg_job *)ctx;
    const size_t i0 = (size_t)index * j->row_block;
    if (i0 >= j->m) return;
    size_t rows = j->row_block;
    if (i0 + rows > j->m) rows = j->m - i0;

    sg_job local = *j;
    local.trans_a = 0;
    local.a = j->gather;
    local.lda = j->k;
    for (size_t t = 0; t < j->taps; ++t) {
        float *dst = j->gather + i0 * j->k;
        const float *src = j->w_taps + i0 * j->k * j->taps + t;
        for (size_t r = 0; r < rows; ++r) {
            for (size_t p = 0; p < j->k; ++p) dst[p] = src[p * j->taps];
            dst += j->k;
            src += j->k * j->taps;
        }
        local.b = j->b + t * j->b_tap_stride;
        local.beta = (t == 0u) ? j->beta : 1.0f;
        for (size_t bj = 0; bj < j->grid_n; ++bj) {
            const size_t j0 = bj * j->nc;
            if (j0 >= j->n) break;
            size_t cols = j->nc;
            if (j0 + cols > j->n) cols = j->n - j0;
            sg_tile_nn(&local, i0, rows, j0, cols);
        }
    }
}

/* ======================================================================
 * The predicate
 *
 * One implementation, called by the runtime AND by the dispatch report, so
 * the two cannot disagree.  dispatch.h's central rule: a report that
 * recomputes the condition can agree with the source and both be wrong.
 * ====================================================================== */
static int sg_work(size_t m, size_t n, size_t k, size_t *out) {
    if (m != 0 && n > (size_t)-1 / m) return -1;
    size_t mn = m * n;
    if (mn != 0 && k > (size_t)-1 / mn) return -1;
    *out = mn * k;
    return 0;
}

mynah_sgemm_family mynah_sgemm_family_for(int trans_a, int trans_b, size_t m,
                                          size_t n, size_t k,
                                          const char **why) {
    const char *reason = "";
    mynah_sgemm_family family = MYNAH_SGEMM_FAMILY_REFERENCE;
    size_t work = 0;
    const int huge = sg_work(m, n, k, &work) != 0;

    if (m == 0u || n == 0u || k == 0u) {
        reason = "degenerate shape: nothing to accumulate, the reference "
                 "applies beta and returns";
    } else if (!huge && work < SG_MIN_BLOCKED_WORK) {
        reason = "below SG_MIN_BLOCKED_WORK (4096 MACs): the plan and the "
                 "pool check cost more than the arithmetic";
    } else if (trans_a && trans_b) {
        reason = "op(A) and op(B) both transposed: neither operand has a "
                 "contiguous axis a kernel here can use. Not reached by any "
                 "call site in this repo";
    } else if (trans_b) {
        family = MYNAH_SGEMM_FAMILY_DOT;
        reason = "op(B) transposed: its columns are B's rows, contiguous in "
                 "k, so each output element is one mynah_dot_f32";
    } else if (m == 1u) {
        family = MYNAH_SGEMM_FAMILY_MATVEC;
        reason = "m == 1: one output row, no row axis to block. This is the "
                 "measured m=1 n=1920 k=64, 129 calls per utterance";
    } else if (n <= SG_NARROW_MAX) {
        family = MYNAH_SGEMM_FAMILY_NARROW;
        reason = "n fits the accumulator budget, so the whole C row block "
                 "stays in registers and op(A) is streamed once. This is the "
                 "measured n=16 frame batch, 28% of calls";
    } else {
        family = MYNAH_SGEMM_FAMILY_PANEL;
        reason = "n exceeds the accumulator budget: ordinary panel GEMM, "
                 "column panels sized to keep the op(B) panel in L2";
    }
    if (why != NULL) *why = reason;
    return family;
}

const char *mynah_sgemm_family_name(mynah_sgemm_family family) {
    switch (family) {
    case MYNAH_SGEMM_FAMILY_REFERENCE: return "reference";
    case MYNAH_SGEMM_FAMILY_DOT:       return "dot";
    case MYNAH_SGEMM_FAMILY_MATVEC:    return "matvec";
    case MYNAH_SGEMM_FAMILY_NARROW:    return "narrow";
    case MYNAH_SGEMM_FAMILY_PANEL:     return "panel";
    default:                           return "?";
    }
}

const char *mynah_sgemm_isa_name(void) { return SG_ISA_NAME; }

size_t mynah_sgemm_narrow_max(void) { return (size_t)SG_NARROW_MAX; }

/* ======================================================================
 * Planning and dispatch
 * ====================================================================== */

/* Column blocking: a function of the shape and the compiled ISA only.  If this
 * ever learns about mynah_num_threads() the determinism argument above is
 * void. */
static void sg_plan_columns(sg_job *j) {
    switch (j->family) {
    case MYNAH_SGEMM_FAMILY_NARROW:
        j->nv_max = SG_NV_MAX;
        j->nc = j->n; /* <= SG_NARROW_MAX by the predicate: one group */
        break;
    case MYNAH_SGEMM_FAMILY_MATVEC:
        j->nv_max = SG_NV_MAX;
        j->nc = SG_MATVEC_NC;
        break;
    case MYNAH_SGEMM_FAMILY_DOT:
        j->nv_max = 0u; /* unused: sg_tile_dot has no column groups */
        j->nc = SG_DOT_NC;
        break;
    case MYNAH_SGEMM_FAMILY_PANEL:
    default: {
        j->nv_max = SG_PANEL_NV;
        size_t nc = (j->k != 0u) ? SG_PANEL_FLOATS / j->k : j->n;
        nc -= nc % SG_NR;
        if (nc < SG_NR) nc = SG_NR;
        j->nc = nc;
        break;
    }
    }
    if (j->nc == 0u || j->nc > j->n) j->nc = j->n;
    j->grid_n = (j->n + j->nc - 1u) / j->nc;
}

/* Row blocking: the ONLY axis the thread count touches, and always rounded up
 * to SG_MR so which strip covers a row never moves.  `force_tasks` is the
 * self-test's handle on this; 0 means "ask the pool". */
static void sg_plan_rows(sg_job *j, size_t force_tasks) {
    size_t work = 0;
    const int huge = sg_work(j->m, j->n, j->k, &work) != 0;
    size_t want = 1u;
    if (force_tasks > 0u) {
        want = force_tasks;
    } else {
        const int threads = mynah_num_threads();
        if (threads > 1 && (huge || work >= SG_PARALLEL_MIN_WORK))
            want = (size_t)threads * 2u;
    }
    /* `want == 1` means the work is below SG_PARALLEL_MIN_WORK and this GEMM
     * should not touch the pool at all.  That intent used to be lost: the
     * COLUMN grid is planned from the shape alone, so a matvec with nc = 256
     * over n = 1920 still produced grid_n = 8 tasks and dispatched them.
     * Measured, the 177 calls of m=1 n=1920 k=64 per utterance took 11.1 us
     * each on one thread and 21.2 us on sixteen -- the pool made them twice as
     * slow.  Running the same tasks inline, in order, is bit-identical (same
     * task code, disjoint outputs) and is what `want == 1` always meant. */
    j->serial = (want == 1u);

    size_t grid_m = (want + j->grid_n - 1u) / j->grid_n;
    if (grid_m == 0u) grid_m = 1u;
    const size_t max_row_blocks = (j->m + SG_MR - 1u) / SG_MR;
    if (grid_m > max_row_blocks) grid_m = max_row_blocks;
    if (grid_m == 0u) grid_m = 1u;

    size_t row_block = (j->m + grid_m - 1u) / grid_m;
    row_block += (SG_MR - row_block % SG_MR) % SG_MR; /* round up to SG_MR */
    if (row_block == 0u) row_block = SG_MR;
    j->row_block = row_block;
    j->grid_m = (j->m + row_block - 1u) / row_block;
}

static int sg_validate(int trans_a, int trans_b, size_t m, size_t n, size_t k,
                       const float *a, size_t lda, const float *b, size_t ldb,
                       const float *c, size_t ldc) {
    if (m == 0u || n == 0u) return 0; /* no output elements: nothing to check */
    if (c == NULL || ldc < n) return -1;
    if (k == 0u) return 0;            /* a and b are not dereferenced */
    if (a == NULL || b == NULL) return -1;
    if (trans_a ? (lda < m) : (lda < k)) return -1;
    if (trans_b ? (ldb < k) : (ldb < n)) return -1;
    return 0;
}

static void sg_count(mynah_sgemm_family family) {
    switch (family) {
    case MYNAH_SGEMM_FAMILY_REFERENCE: sg_bump(&g_sg.reference); break;
    case MYNAH_SGEMM_FAMILY_DOT:       sg_bump(&g_sg.dot);       break;
    case MYNAH_SGEMM_FAMILY_MATVEC:    sg_bump(&g_sg.matvec);    break;
    case MYNAH_SGEMM_FAMILY_NARROW:    sg_bump(&g_sg.narrow);    break;
    case MYNAH_SGEMM_FAMILY_PANEL:     sg_bump(&g_sg.panel);     break;
    default: break;
    }
}

/* Can this family serve this operand layout at all?  A forced family that
 * silently degrades would make the self-test's comparison vacuous, so the
 * degradation is reported through `ran` rather than hidden. */
static int sg_family_fits(mynah_sgemm_family family, int trans_a, int trans_b,
                          size_t m, size_t n, size_t k) {
    if (m == 0u || n == 0u || k == 0u) return family == MYNAH_SGEMM_FAMILY_REFERENCE;
    switch (family) {
    case MYNAH_SGEMM_FAMILY_REFERENCE:
        return 1;
    case MYNAH_SGEMM_FAMILY_DOT:
        return trans_b != 0 && trans_a == 0;
    case MYNAH_SGEMM_FAMILY_MATVEC:
        return trans_b == 0 && m == 1u;
    case MYNAH_SGEMM_FAMILY_NARROW:
        return trans_b == 0 && n <= SG_NARROW_MAX;
    case MYNAH_SGEMM_FAMILY_PANEL:
        return trans_b == 0;
    default:
        return 0;
    }
}

static int sg_dispatch(mynah_sgemm_family want, int forced, size_t force_tasks,
                       mynah_sgemm_family *ran, int trans_a, int trans_b,
                       size_t m, size_t n, size_t k, float alpha,
                       const float *a, size_t lda, const float *b, size_t ldb,
                       float beta, float *c, size_t ldc) {
    if (ran != NULL) *ran = MYNAH_SGEMM_FAMILY_REFERENCE;
    const unsigned long long t_prof = sg_now();
    sg_bump(&g_sg.calls);
    if (sg_validate(trans_a, trans_b, m, n, k, a, lda, b, ldb, c, ldc) != 0) {
        sg_bump(&g_sg.refused);
        return -1;
    }
    if (m == 0u || n == 0u) return 0;

    mynah_sgemm_family family =
        forced ? want
               : mynah_sgemm_family_for(trans_a, trans_b, m, n, k, NULL);
    if (!sg_family_fits(family, trans_a, trans_b, m, n, k))
        family = MYNAH_SGEMM_FAMILY_REFERENCE;
    if (ran != NULL) *ran = family;
    sg_count(family);

    if (family == MYNAH_SGEMM_FAMILY_REFERENCE) {
        mynah_sgemm_f32_reference(trans_a, trans_b, m, n, k, alpha, a, lda, b,
                                  ldb, beta, c, ldc);
        sg_hist_add(t_prof, trans_a, trans_b, m, n, k, (int)family, 1u);
        return 0;
    }

    sg_job job;
    memset(&job, 0, sizeof job);
    job.trans_a = trans_a;
    job.trans_b = trans_b;
    job.m = m; job.n = n; job.k = k;
    job.alpha = alpha; job.beta = beta;
    job.a = a; job.lda = lda;
    job.b = b; job.ldb = ldb;
    job.c = c; job.ldc = ldc;
    job.family = family;
    sg_plan_columns(&job);
    sg_plan_rows(&job, force_tasks);

    /* A clamped task count would leave output blocks uncomputed, so an
     * absurd grid collapses to a single tile instead of being truncated.
     * Unreachable for any shape this runtime produces; it is here because
     * "clamp and hope" is how a silently wrong result gets shipped. */
    size_t tasks = job.grid_m * job.grid_n;
    if (job.grid_m != 0u && tasks / job.grid_m != job.grid_n) tasks = 0u;
    if (tasks == 0u || tasks > (size_t)INT_MAX) {
        job.nc = job.n;
        job.grid_n = 1u;
        job.row_block = job.m + (SG_MR - job.m % SG_MR) % SG_MR;
        job.grid_m = 1u;
        tasks = 1u;
    }
    if (job.serial) {
        for (size_t t = 0; t < tasks; ++t) sg_task(&job, (int)t);
    } else {
        mynah_parallel_for((int)tasks, sg_task, &job);
    }
    sg_hist_add(t_prof, trans_a, trans_b, m, n, k, (int)family, tasks);
    return 0;
}

int mynah_sgemm_f32(int trans_a, int trans_b, size_t m, size_t n, size_t k,
                    float alpha, const float *a, size_t lda, const float *b,
                    size_t ldb, float beta, float *c, size_t ldc) {
    return sg_dispatch(MYNAH_SGEMM_FAMILY_REFERENCE, 0, 0u, NULL, trans_a,
                       trans_b, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

int mynah_sgemm_f32_conv_taps(size_t m, size_t n, size_t k, size_t taps,
                              const float *weight, float *gather,
                              const float *b, size_t ldb, size_t b_tap_stride,
                              float beta, float *c, size_t ldc) {
    if (taps == 0u || weight == NULL || gather == NULL || b == NULL ||
        c == NULL) {
        return -1;
    }
    if (m == 0u || n == 0u || k == 0u) return 1;
    if (ldb < n || ldc < n) return -1;
    /* The weight is indexed as one [m][k][taps] object and the gather as one
     * [m][k] object, so m*k*taps must not wrap.  Refuse rather than clamp: the
     * caller's per-tap loop handles any shape this declines. */
    size_t weight_elems = 0;
    if (sg_work(m, k, taps, &weight_elems) != 0) {
        sg_bump(&g_sg.fused_refused);
        return 1;
    }

    /* The plan is made for ONE tap, exactly as the taps-many separate calls
     * would have made it, because that is what keeps every output element on
     * the micro-kernel instantiation it had before. */
    const mynah_sgemm_family family =
        mynah_sgemm_family_for(0, 0, m, n, k, NULL);
    if (family != MYNAH_SGEMM_FAMILY_NARROW &&
        family != MYNAH_SGEMM_FAMILY_PANEL) {
        sg_bump(&g_sg.fused_refused);
        return 1;
    }

    sg_job job;
    memset(&job, 0, sizeof job);
    job.m = m; job.n = n; job.k = k;
    job.alpha = 1.0f; job.beta = beta;
    job.b = b; job.ldb = ldb;
    job.c = c; job.ldc = ldc;
    job.family = family;
    sg_plan_columns(&job);
    sg_plan_rows(&job, 0u);
    if (job.grid_m == 0u || job.grid_m > (size_t)INT_MAX) {
        sg_bump(&g_sg.fused_refused);
        return 1;
    }
    job.taps = taps;
    job.w_taps = weight;
    job.gather = gather;
    job.b_tap_stride = b_tap_stride;

    /* The arithmetic is `taps` GEMMs, so it is counted as `taps` GEMMs: the
     * family histogram must not change meaning because the dispatch did. */
    const unsigned long long t_prof = sg_now();
    for (size_t t = 0; t < taps; ++t) {
        sg_bump(&g_sg.calls);
        sg_count(family);
    }
    sg_bump(&g_sg.fused);
    atomic_fetch_add_explicit(&g_sg.fused_taps, (unsigned long long)taps,
                              memory_order_relaxed);

    if (job.serial || job.grid_m == 1u) {
        for (size_t t = 0; t < job.grid_m; ++t) sg_task_taps(&job, (int)t);
    } else {
        mynah_parallel_for((int)job.grid_m, sg_task_taps, &job);
    }
    sg_hist_add(t_prof, 0, 0, m, n, k, (int)family, job.grid_m);
    return 0;
}

int mynah_sgemm_f32_forced(mynah_sgemm_family want, mynah_sgemm_family *ran,
                           int trans_a, int trans_b, size_t m, size_t n,
                           size_t k, float alpha, const float *a, size_t lda,
                           const float *b, size_t ldb, float beta, float *c,
                           size_t ldc) {
    return sg_dispatch(want, 1, 0u, ran, trans_a, trans_b, m, n, k, alpha, a,
                       lda, b, ldb, beta, c, ldc);
}

/* ======================================================================
 * Self-test
 *
 * Model-free.  The reference is the oracle; every compiled path is compared
 * against it over the eleven MEASURED shapes and over the edge cases the
 * blocking can get wrong.
 *
 * TOLERANCE.  Relative, not bit-identical, and the reason is in the build
 * flags rather than in the algorithm: -ffast-math implies -fassociative-math,
 * so clang may contract a*b+c into one FMA in the kernels and not in the
 * reference (or regroup either).  The accumulation ORDER over p is the same in
 * both, so the residual is one rounding per term at worst; 1e-4 relative is
 * three orders of magnitude tighter than any indexing bug and three orders
 * looser than the contraction noise.  The observed maximum is reported in the
 * error string when the bound is exceeded, so a tightening is a measurement
 * rather than a guess.
 *
 * DETERMINISM.  The thread-count sweep IS asserted bit-identical, because
 * there the two arms are the same machine code with different block offsets,
 * not two textually identical source paths compiled twice.
 * ====================================================================== */

#define SG_TEST_TOL 1.0e-4f

/* Largest relative deviation the last self-test observed.  Written once at the
 * end of a run that has no other writer; read only by the dispatch probe. */
static float g_sg_worst = -1.0f;

typedef struct {
    size_t m, n, k;
    int trans_a;
    const char *note;
} sg_case;

/* .work/no-blas.md §3, instrumented over one PocketTTS utterance: 1075 calls,
 * 11 distinct shapes.  Ordered by call count. */
static const sg_case g_measured[] = {
    {512,  16,   512, 0, "conv1d, 301 calls, 28% of all GEMMs"},
    {64,   480,  128, 0, "conv1d, 129 calls"},
    {32,   1920, 64,  0, "conv1d, 129 calls"},
    {128,  96,   256, 0, "conv1d, 129 calls"},
    {1,    1920, 64,  0, "conv1d, 129 calls, a matvec in a GEMM's clothes"},
    {512,  480,  128, 1, "convtranspose, 43 calls"},
    {3072, 16,   512, 1, "convtranspose, 43 calls, the widest"},
    {1280, 96,   256, 1, "convtranspose, 43 calls"},
    {64,   1920, 32,  0, "conv1d, 43 calls"},
    {256,  96,   128, 0, "conv1d, 43 calls"},
    {128,  480,  64,  0, "conv1d, 43 calls"},
};

/* Edge cases: k not a multiple of the unroll, m or n equal to 1, shapes that
 * straddle SG_LANES and SG_MR, and a transposed-B pair (the DOT family, which
 * no measured shape reaches). */
static const sg_case g_edges[] = {
    {1,  1,   1,   0, "1x1x1"},
    {1,  17,  13,  0, "single row, ragged n and k"},
    {17, 1,   13,  0, "single column"},
    {17, 13,  1,   0, "k == 1"},
    {5,  7,   11,  0, "every dimension prime"},
    {33, 31,  29,  0, "all three straddle SG_MR and SG_LANES"},
    {64, 16,  63,  0, "narrow family, k not a multiple of 4"},
    {64, 17,  63,  0, "one column past the narrow boundary"},
    {4,  64,  64,  0, "exactly SG_MR rows"},
    {6,  64,  64,  0, "SG_MR + 2 rows: the ragged row tail"},
    {70, 130, 70,  1, "transposed A, all ragged"},
    {33, 31,  29,  1, "transposed A, prime"},
};

static void sg_fill(float *p, size_t n, unsigned seed) {
    /* xorshift32: reproducible across platforms and libcs, which a rand()
     * based fixture is not. */
    unsigned s = seed | 1u;
    for (size_t i = 0; i < n; ++i) {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        p[i] = (float)((double)(s >> 8) / 8388608.0 - 1.0); /* [-1, 1) */
    }
}

/* Returns the largest relative deviation, or a negative value on a non-finite
 * result (which no tolerance may excuse). */
static float sg_maxdev(const float *got, const float *want, size_t n) {
    float worst = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        if (!isfinite(got[i])) return -1.0f;
        const float d = fabsf(got[i] - want[i]);
        const float scale = 1.0f + fabsf(want[i]);
        const float rel = d / scale;
        if (rel > worst) worst = rel;
    }
    return worst;
}

typedef struct {
    float *a, *b, *c, *ref;
    size_t cap_a, cap_b, cap_c;
    unsigned long long ran[5];
    float worst;
} sg_test_ctx;

static int sg_case_run(sg_test_ctx *t, const sg_case *sc, int trans_b,
                       float alpha, float beta, char *error, size_t cap) {
    const size_t m = sc->m, n = sc->n, k = sc->k;
    const size_t na = m * k, nb = n * k, nc = m * n;
    if (na > t->cap_a || nb > t->cap_b || nc > t->cap_c) {
        snprintf(error, cap, "sgemm self-test fixture too small for %zux%zux%zu",
                 m, n, k);
        return -1;
    }
    const size_t lda = sc->trans_a ? m : k;
    const size_t ldb = trans_b ? k : n;

    sg_fill(t->a, na, (unsigned)(m * 7919u + k));
    sg_fill(t->b, nb, (unsigned)(n * 104729u + k * 31u));
    sg_fill(t->c, nc, 0xC0FFEEu);
    memcpy(t->ref, t->c, nc * sizeof(float));
    mynah_sgemm_f32_reference(sc->trans_a, trans_b, m, n, k, alpha, t->a, lda,
                              t->b, ldb, beta, t->ref, n);

    /* The predicate's own choice, plus every family forced.  A family that
     * cannot serve this layout reports REFERENCE through `ran` and is not
     * counted as coverage, so a vacuous comparison cannot look like a pass. */
    for (int f = -1; f <= (int)MYNAH_SGEMM_FAMILY_PANEL; ++f) {
        /* Asked BEFORE the call, not after: a family that cannot serve this
         * layout degrades to the reference inside sg_dispatch, and comparing
         * the reference against itself is both vacuous and, on the 3072x512
         * shape, the single most expensive thing this test could do. */
        if (f == (int)MYNAH_SGEMM_FAMILY_REFERENCE)
            continue; /* the oracle: comparing it with itself proves nothing.
                       * Its coverage comes from the degenerate and below-
                       * threshold cases, where the dispatcher picks it. */
        if (f >= 0 && !sg_family_fits((mynah_sgemm_family)f, sc->trans_a,
                                      trans_b, m, n, k))
            continue;
        float *out = t->c;
        /* Restore C: beta != 0 reads it, so every arm must start from the
         * same bytes the reference started from. */
        sg_fill(out, nc, 0xC0FFEEu);
        mynah_sgemm_family ran = MYNAH_SGEMM_FAMILY_REFERENCE;
        int rc;
        if (f < 0) {
            rc = mynah_sgemm_f32(sc->trans_a, trans_b, m, n, k, alpha, t->a,
                                 lda, t->b, ldb, beta, out, n);
            ran = mynah_sgemm_family_for(sc->trans_a, trans_b, m, n, k, NULL);
            if (!sg_family_fits(ran, sc->trans_a, trans_b, m, n, k))
                ran = MYNAH_SGEMM_FAMILY_REFERENCE;
        } else {
            rc = mynah_sgemm_f32_forced((mynah_sgemm_family)f, &ran,
                                        sc->trans_a, trans_b, m, n, k, alpha,
                                        t->a, lda, t->b, ldb, beta, out, n);
            if ((int)ran != f) continue; /* layout refused: not coverage */
        }
        if (rc != 0) {
            snprintf(error, cap, "sgemm %s returned -1 on %zux%zux%zu (%s)",
                     f < 0 ? "dispatch" : mynah_sgemm_family_name(
                                              (mynah_sgemm_family)f),
                     m, n, k, sc->note);
            return -1;
        }
        t->ran[(int)ran]++;
        const float dev = sg_maxdev(out, t->ref, nc);
        if (dev < 0.0f) {
            snprintf(error, cap,
                     "sgemm %s produced a non-finite value on %zux%zux%zu (%s)",
                     mynah_sgemm_family_name(ran), m, n, k, sc->note);
            return -1;
        }
        if (dev > t->worst) t->worst = dev;
        if (dev > SG_TEST_TOL) {
            snprintf(error, cap,
                     "sgemm %s deviates %.3g (bound %.3g) on m=%zu n=%zu k=%zu "
                     "transA=%d transB=%d alpha=%g beta=%g (%s)",
                     mynah_sgemm_family_name(ran), (double)dev,
                     (double)SG_TEST_TOL, m, n, k, sc->trans_a, trans_b,
                     (double)alpha, (double)beta, sc->note);
            return -1;
        }
    }
    return 0;
}

/* The determinism proof: the same GEMM planned as one task and as many, byte
 * for byte.  Legitimate as a bit-identity assertion because both arms enter
 * the SAME compiled micro-kernel instantiations -- only the block offsets
 * differ -- which is exactly the property the SG_MR round-up in sg_plan_rows
 * exists to guarantee. */
static int sg_case_threads(sg_test_ctx *t, const sg_case *sc, char *error,
                           size_t cap) {
    const size_t m = sc->m, n = sc->n, k = sc->k;
    const size_t nc = m * n;
    const size_t lda = sc->trans_a ? m : k;
    sg_fill(t->a, m * k, (unsigned)(m * 7919u + k));
    sg_fill(t->b, n * k, (unsigned)(n * 104729u + k * 31u));

    static const size_t task_counts[] = {1u, 2u, 3u, 5u, 64u};
    for (size_t ti = 0; ti < sizeof task_counts / sizeof task_counts[0]; ++ti) {
        float *out = (ti == 0u) ? t->ref : t->c;
        sg_fill(out, nc, 0x5EEDu);
        if (sg_dispatch(MYNAH_SGEMM_FAMILY_REFERENCE, 0, task_counts[ti], NULL,
                        sc->trans_a, 0, m, n, k, 1.0f, t->a, lda, t->b, n,
                        0.5f, out, n) != 0) {
            snprintf(error, cap, "sgemm task sweep failed on %zux%zux%zu", m, n, k);
            return -1;
        }
        if (ti == 0u) continue;
        if (memcmp(t->ref, t->c, nc * sizeof(float)) != 0) {
            snprintf(error, cap,
                     "sgemm is not thread-count independent: %zu tasks differ "
                     "from 1 task on m=%zu n=%zu k=%zu (%s)",
                     task_counts[ti], m, n, k, sc->note);
            return -1;
        }
    }
    return 0;
}

/* The conv-tap fusion against the sequence of calls it replaces.  The
 * assertion is memcmp, not a tolerance: the whole claim of
 * mynah_sgemm_f32_conv_taps is that it is the SAME arithmetic in the SAME
 * order, so anything short of byte-identical is a failure.  A refusal (rc 1)
 * is a pass -- the caller then runs the old loop, which is the thing this is
 * being compared against -- but it is counted, because a test in which the
 * fused path never ran would prove nothing. */
static int sg_taps_case(size_t m, size_t n, size_t k, size_t taps, float beta,
                        unsigned long long *ran, char *error,
                        size_t error_capacity) {
    const size_t ldb = n + 3u; /* a window is wider than the output */
    float *w = (float *)malloc(m * k * taps * sizeof(float));
    float *gather = (float *)malloc(m * k * sizeof(float));
    float *b = (float *)malloc((k + taps) * ldb * sizeof(float));
    float *c_fused = (float *)malloc(m * n * sizeof(float));
    float *c_loop = (float *)malloc(m * n * sizeof(float));
    if (w == NULL || gather == NULL || b == NULL || c_fused == NULL ||
        c_loop == NULL) {
        free(w); free(gather); free(b); free(c_fused); free(c_loop);
        snprintf(error, error_capacity, "sgemm taps self-test: out of memory");
        return -1;
    }
    for (size_t i = 0; i < m * k * taps; ++i)
        w[i] = (float)(((i * 1103515245u + 12345u) >> 9) % 2003u) / 1000.0f - 1.0f;
    for (size_t i = 0; i < (k + taps) * ldb; ++i)
        b[i] = (float)(((i * 22695477u + 1u) >> 11) % 1999u) / 997.0f - 1.0f;
    for (size_t i = 0; i < m * n; ++i) {
        c_fused[i] = 0.25f * (float)(i % 7u);
        c_loop[i] = c_fused[i];
    }

    const int rc = mynah_sgemm_f32_conv_taps(m, n, k, taps, w, gather, b, ldb,
                                             1u, beta, c_fused, n);
    if (rc < 0) {
        free(w); free(gather); free(b); free(c_fused); free(c_loop);
        snprintf(error, error_capacity,
                 "sgemm taps self-test: refused m=%zu n=%zu k=%zu", m, n, k);
        return -1;
    }
    for (size_t t = 0; t < taps; ++t) {
        for (size_t i = 0; i < m; ++i)
            for (size_t p = 0; p < k; ++p)
                gather[i * k + p] = w[i * k * taps + p * taps + t];
        (void)mynah_sgemm_f32(0, 0, m, n, k, 1.0f, gather, k, b + t, ldb,
                              (t == 0u) ? beta : 1.0f, c_loop, n);
    }
    int bad = (rc == 0) && memcmp(c_fused, c_loop, m * n * sizeof(float)) != 0;
    if (bad) {
        size_t at = 0;
        for (; at < m * n; ++at) if (c_fused[at] != c_loop[at]) break;
        snprintf(error, error_capacity,
                 "sgemm taps self-test: m=%zu n=%zu k=%zu taps=%zu beta=%g "
                 "differs at %zu: fused %.9g vs per-tap %.9g -- the fusion is "
                 "only allowed to be byte-identical",
                 m, n, k, taps, (double)beta, at, (double)c_fused[at],
                 (double)c_loop[at]);
    } else if (rc == 0 && ran != NULL) {
        *ran += 1ull;
    }
    free(w); free(gather); free(b); free(c_fused); free(c_loop);
    return bad ? -1 : 0;
}

int mynah_sgemm_self_test(char *error, size_t error_capacity) {
    char scratch[256];
    if (error == NULL || error_capacity == 0) {
        error = scratch;
        error_capacity = sizeof scratch;
    }
    error[0] = '\0';

    size_t max_a = 0, max_b = 0, max_c = 0;
    const size_t n_meas = sizeof g_measured / sizeof g_measured[0];
    const size_t n_edge = sizeof g_edges / sizeof g_edges[0];
    for (size_t i = 0; i < n_meas + n_edge; ++i) {
        const sg_case *sc = (i < n_meas) ? &g_measured[i] : &g_edges[i - n_meas];
        if (sc->m * sc->k > max_a) max_a = sc->m * sc->k;
        if (sc->n * sc->k > max_b) max_b = sc->n * sc->k;
        if (sc->m * sc->n > max_c) max_c = sc->m * sc->n;
    }

    sg_test_ctx t;
    memset(&t, 0, sizeof t);
    t.cap_a = max_a; t.cap_b = max_b; t.cap_c = max_c;
    t.a = (float *)malloc(max_a * sizeof(float));
    t.b = (float *)malloc(max_b * sizeof(float));
    t.c = (float *)malloc(max_c * sizeof(float));
    t.ref = (float *)malloc(max_c * sizeof(float));
    if (t.a == NULL || t.b == NULL || t.c == NULL || t.ref == NULL) {
        free(t.a); free(t.b); free(t.c); free(t.ref);
        snprintf(error, error_capacity, "sgemm self-test: out of memory");
        return -1;
    }

    int rc = 0;
    /* The eleven measured shapes, alpha/beta as the call sites use them
     * (seanet.c alternates beta 0 and 1 across kernel taps). */
    for (size_t i = 0; i < n_meas && rc == 0; ++i) {
        rc = sg_case_run(&t, &g_measured[i], 0, 1.0f, 0.0f, error, error_capacity);
        if (rc == 0)
            rc = sg_case_run(&t, &g_measured[i], 0, 1.0f, 1.0f, error, error_capacity);
    }
    /* Edge cases across both transpose flags and a non-unit alpha/beta. */
    for (size_t i = 0; i < n_edge && rc == 0; ++i) {
        rc = sg_case_run(&t, &g_edges[i], 0, 1.0f, 0.0f, error, error_capacity);
        if (rc == 0)
            rc = sg_case_run(&t, &g_edges[i], 0, -0.75f, 2.5f, error, error_capacity);
        if (rc == 0 && !g_edges[i].trans_a)
            rc = sg_case_run(&t, &g_edges[i], 1, 1.0f, 0.0f, error, error_capacity);
        if (rc == 0 && !g_edges[i].trans_a)
            rc = sg_case_run(&t, &g_edges[i], 1, 0.5f, -1.25f, error, error_capacity);
    }
    /* Thread-count independence, on shapes that actually block in both axes. */
    for (size_t i = 0; i < n_edge && rc == 0; ++i) {
        if (g_edges[i].m * g_edges[i].n * g_edges[i].k < 1024u) continue;
        rc = sg_case_threads(&t, &g_edges[i], error, error_capacity);
    }
    if (rc == 0) {
        static const sg_case sweep[] = {
            {512, 16, 512, 0, "the 28% shape"},
            {64, 480, 128, 0, "a panel shape"},
        };
        for (size_t i = 0; i < 2u && rc == 0; ++i)
            rc = sg_case_threads(&t, &sweep[i], error, error_capacity);
    }

    /* The conv-tap fusion.  The first two are the shapes the PocketTTS SEANet
     * decoder actually fuses (entry conv, and the first residual block's
     * conv1); the rest exercise ragged m, taps == 1, and a non-zero beta. */
    if (rc == 0) {
        static const size_t taps_cases[][4] = {
            {512, 16, 512, 7}, {128, 96, 256, 3}, {64, 480, 128, 3},
            {32, 1920, 64, 3}, {13, 16, 37, 5},   {7, 9, 5, 1},
            {1, 1920, 64, 3},  {4, 16, 16, 2}
        };
        unsigned long long taps_ran = 0;
        for (size_t i = 0; i < sizeof taps_cases / sizeof taps_cases[0] &&
                           rc == 0; ++i) {
            rc = sg_taps_case(taps_cases[i][0], taps_cases[i][1],
                              taps_cases[i][2], taps_cases[i][3], 0.0f,
                              &taps_ran, error, error_capacity);
            if (rc == 0)
                rc = sg_taps_case(taps_cases[i][0], taps_cases[i][1],
                                  taps_cases[i][2], taps_cases[i][3], 1.0f,
                                  &taps_ran, error, error_capacity);
        }
        if (rc == 0 && taps_ran == 0ull) {
            snprintf(error, error_capacity,
                     "sgemm self-test: the conv-tap fusion refused every case, "
                     "so the byte-identity it claims was never checked");
            rc = -1;
        }
    }

    /* Coverage refusal.  A self-test in which a family never ran proves
     * nothing about that family, and reporting PASS would be the silent
     * fallback .work/engineering-method.md §4 exists to forbid. */
    if (rc == 0) {
        static const char *names[5] = {"reference", "dot", "matvec", "narrow",
                                       "panel"};
        for (int f = 0; f <= (int)MYNAH_SGEMM_FAMILY_PANEL; ++f) {
            if (t.ran[f] == 0ull) {
                snprintf(error, error_capacity,
                         "sgemm self-test covered no %s case: the comparison "
                         "would be vacuous, so this is a failure and not a pass",
                         names[f]);
                rc = -1;
                break;
            }
        }
    }

    /* Kept so the dispatch row can print the MEASURED margin rather than just
     * PASS: "7.1e-07 against a 1e-04 bound" is a number a future tightening
     * can be argued from; "PASS" is not. */
    g_sg_worst = t.worst;
    free(t.a); free(t.b); free(t.c); free(t.ref);
    return rc;
}

/* ======================================================================
 * Dispatch predicates
 *
 * Every `resolved` below is produced by calling something in this file -- the
 * predicate itself, the ISA name, or the counters the dispatcher actually
 * incremented.  Nothing here re-derives "n <= 16".
 * ====================================================================== */

static int probe_sgemm_kernel(char *out, size_t capacity, const char **why) {
    static char text[240];
    /* The predicate is CALLED, on the measured headline shape, rather than
     * restated: .work/no-blas.md §3, m=512 n=16 k=512, 28% of all GEMM calls
     * in one PocketTTS utterance. */
    const mynah_sgemm_family head =
        mynah_sgemm_family_for(0, 0, 512u, 16u, 512u, NULL);
    snprintf(out, capacity, "%s", mynah_sgemm_isa_name());
    snprintf(text, sizeof text,
             "[predicate] src/sgemm.c mynah_sgemm_isa_name(): %s micro-kernels, "
             "%d rows x %d accumulator vectors, narrow/panel boundary at n=%zu "
             "(derived from the register file, not chosen). "
             "mynah_sgemm_family_for(512,16,512) says '%s'",
             mynah_sgemm_isa_name(), SG_MR, SG_NV_MAX,
             mynah_sgemm_narrow_max(), mynah_sgemm_family_name(head));
    *why = text;
    return 0;
}

/* The conv-tap fusion is reported HERE, on the family row, and not on a row of
 * its own: src/dispatch.c declares every row id it will print, and a value
 * probe registered under an id that table does not carry is silently dropped.
 * Adding `sgemm.conv_taps` there is a one-line change in a file this lane does
 * not own; until it lands, the counters ride the row that exists rather than
 * living nowhere. */
static int probe_sgemm_family(char *out, size_t capacity, const char **why) {
    static char text[400];
    mynah_sgemm_stats st;
    mynah_sgemm_stats_get(&st);
    if (st.calls == 0ull) {
        snprintf(out, capacity, "n/a");
        *why = "[predicate] src/sgemm.c: mynah_sgemm_f32 has not been called in "
               "this process, so there is nothing to report. Read this row "
               "after a synthesis; --dispatch-map loads no model";
        return 0;
    }
    snprintf(out, capacity, "%llu calls", st.calls);
    snprintf(text, sizeof text,
             "[predicate] src/sgemm.c counters: %llu narrow, %llu panel, %llu "
             "matvec, %llu dot, %llu reference, %llu refused. narrow is the "
             "n<=%zu frame-batch family; reference is degenerate or too small "
             "to block. conv-tap fusion: %llu regions carrying %llu of those "
             "calls in one dispatch each, %llu refused and left to the "
             "caller's per-tap loop",
             st.narrow, st.panel, st.matvec, st.dot, st.reference, st.refused,
             mynah_sgemm_narrow_max(), st.fused, st.fused_taps,
             st.fused_refused);
    *why = text;
    return 0;
}

static int probe_sgemm_selftest(char *out, size_t capacity, const char **why) {
    static char text[480];
    char err[192];
    err[0] = '\0';
    const int ok = mynah_sgemm_self_test(err, sizeof err) == 0;
    snprintf(out, capacity, "%s", ok ? "PASS" : "FAIL");
    snprintf(text, sizeof text,
             "[predicate] src/sgemm.c mynah_sgemm_self_test(): %s. Every "
             "compiled family against the scalar reference over the eleven "
             "measured shapes plus the edge cases, and one-task vs many-task "
             "byte equality. Worst relative deviation %.2g against a %.0e "
             "bound",
             ok ? "PASS" : err, (double)g_sg_worst, (double)SG_TEST_TOL);
    *why = text;
    return 0;
}

void mynah_sgemm_dispatch_probes(void) {
    mynah_dispatch_register_value_probe("sgemm.kernel", probe_sgemm_kernel);
    mynah_dispatch_register_value_probe("sgemm.family", probe_sgemm_family);
    mynah_dispatch_register_value_probe("sgemm.selftest", probe_sgemm_selftest);
}
