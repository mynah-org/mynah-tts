/* sgemm.h -- mynah's own f32 GEMM, so that no external BLAS has to be in the
 * process.
 *
 * WHY THIS EXISTS.  The reason is ownership, not speed (.work/no-blas.md).
 * OpenBLAS brings its own thread pool with its own policies -- an idle spin
 * that doubled TTFA and made it bimodal, a team size that ignored our own
 * pool, an env var that had to be ABSENT for a profile to be valid -- and
 * every one of those is a trap that has to be rechecked on every host,
 * forever.  Production is Linux, where the BLAS is OpenBLAS, so removing it is
 * a production decision.
 *
 * THE SURFACE IS ONE FUNCTION.  `cblas_sgemm`, three call sites: the backend
 * vtable's sgemm and the row-blocked matmul_block (src/backend.c), and
 * sea_sgemm (src/seanet.c).  The PocketTTS production path reaches only the
 * last of those -- its backbone and flow head already run our own quantized
 * kernels -- so two call sites in seanet.c are the entire production
 * dependency.  This file replaces all three.
 *
 * WHAT THE SHAPES ACTUALLY ARE.  Measured, not assumed: 1075 GEMM calls over
 * one PocketTTS utterance, 11 distinct shapes (.work/no-blas.md §3).  The
 * single most frequent -- 28% of calls -- is m=512 n=16 k=512, and the widest
 * is m=3072 n=16 k=512.  That n=16 is the FRAME BATCH, not a long axis.  No
 * packing cost amortises over sixteen columns, so this is much closer to a
 * handful of matvecs than to a panel GEMM, and n=16 and n=1920 are two
 * different kernels rather than one tuned shape.  m=1 n=1920 k=64 appears 129
 * times: a matvec wearing a GEMM's clothes, which never reaches a blocked
 * kernel at all.
 *
 * The families below are exactly those three answers plus a transposed-B one.
 * The threshold between narrow and panel is DERIVED from the register file
 * (see mynah_sgemm_narrow_max), not chosen.
 */
#ifndef MYNAH_TTS_SGEMM_H
#define MYNAH_TTS_SGEMM_H

#include <stddef.h>

/* "A REAL f32 GEMM EXISTS IN THIS BUILD" -- Accelerate, OpenBLAS, or ours.
 *
 * NOT "a vendor BLAS is linked", and the distinction is the one the default
 * flip turned into a bug hazard.  Several fast paths outside this file were
 * keyed on MYNAH_USE_OPENBLAS when what they actually needed was "some GEMM":
 * the conv1d tap-GEMM accumulation and its packed-tap cache, and the rows=1
 * parallel matvec on x86.  With `none` as the Linux default those would have
 * switched themselves off on the production target -- silently, and with no
 * test that could see it, because the PocketTTS path goes through seanet.c
 * and never touches them.  One macro, so the question is asked once. */
#if defined(MYNAH_USE_ACCELERATE) || defined(MYNAH_USE_OPENBLAS) || \
    defined(MYNAH_USE_OWN_SGEMM)
#define MYNAH_HAVE_SGEMM 1
#else
#define MYNAH_HAVE_SGEMM 0
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------------
 * The call
 * ------------------------------------------------------------------------ */

/* C[m,n] = alpha * op(A) * op(B) + beta * C, row-major.
 *
 * Identical semantics to cblas_sgemm(CblasRowMajor, ...) for the arguments
 * this repo uses, including both transpose flags, alpha, beta, and the
 * leading dimensions:
 *
 *   trans_a == 0  op(A) is A, m x k, row stride lda   (lda >= k)
 *   trans_a != 0  op(A) is A^T, A is k x m, lda >= m
 *   trans_b == 0  op(B) is B, k x n, ldb >= n
 *   trans_b != 0  op(B) is B^T, B is n x k, ldb >= k
 *
 * beta == 0 means C is WRITTEN, never read: an uninitialised or NaN-carrying
 * C must not poison the result.  That is cblas' contract and the old triple
 * loop in backend.c did not honour it.
 *
 * Work is split over disjoint output blocks on the mynah thread pool.  The
 * reduction over k is never split, and the row-block size is always rounded up
 * to the micro-kernel's row count, so which micro-kernel computes a given
 * output element does not depend on the thread count -- see the determinism
 * note in sgemm.c.
 *
 * Returns 0, or -1 on a null pointer or a leading dimension too small for the
 * declared shape (the caller's error string is its own business; this function
 * has no state to report).
 */
int mynah_sgemm_f32(int trans_a, int trans_b,
                    size_t m, size_t n, size_t k,
                    float alpha,
                    const float *a, size_t lda,
                    const float *b, size_t ldb,
                    float beta,
                    float *c, size_t ldc);

/* ------------------------------------------------------------------------
 * Conv taps: several GEMMs in ONE pool region
 * ------------------------------------------------------------------------
 *
 * WHY THIS EXISTS.  A causal conv1d with kernel K is K GEMMs that accumulate
 * into the same C, one per kernel tap (src/seanet.c).  Written as K calls to
 * mynah_sgemm_f32 that is K pool dispatches, and each one is far too small to
 * pay for one: measured on the 32-core Neoverse-V2 box, m=512 n=16 k=512 --
 * 28% of every GEMM the PocketTTS codec issues -- takes 115 us on one thread
 * and 47 us on sixteen.  A 2.4x from 16x the cores, because ~40 us of that
 * 47 is the dispatch itself.  The entry convolution alone pays that seven
 * times per frame.
 *
 * Worse, each of those K calls needs op(A) as a DENSE [m][k] matrix, while the
 * weight is stored [m][k][taps] -- so the caller gathered one tap out of the
 * weight, on its own thread, before every dispatch.  That gather was 24.6% of
 * codec.conv_stack at sixteen threads and scaled 1.2x.
 *
 * THE FIX IS ONE REGION, NOT MORE THREADS.  A task owns a block of C's ROWS.
 * Tap t only ever accumulates C rows into themselves, and the gather for tap t
 * only ever needs the matching rows of the weight, so a task can gather ITS
 * rows and run ITS strip for every tap without looking at any other task's
 * rows.  K dispatches become one, and the gather lands in the thread that is
 * about to read it, while it is still in that core's L1.
 *
 * BYTE-IDENTICAL, NOT "AGREES TO 1e-6".  This is the same arithmetic in the
 * same order, and the gate is memcmp, not a tolerance:
 *   - the gathered values are a copy, not a computation;
 *   - the row blocking is planned from (m, n, k) for ONE tap, exactly as the K
 *     separate calls planned it, so every output element is computed by the
 *     same micro-kernel instantiation as before;
 *   - the reduction over k is still never split, and the taps are still
 *     applied in order 0..taps-1 with the beta chain, so each tap's partial
 *     sum is still rounded to f32 before the next one is added.
 * mynah_sgemm_self_test() asserts the memcmp against the K-call sequence.
 *
 * REFUSAL.  Returns 1 -- "not applicable, do it yourself" -- when the planned
 * column grid is wider than one block, because then two tasks share a row
 * block and would race to gather the same rows.  The caller then runs its
 * ordinary K-call loop, which is why this can never change a result: the
 * refused path is the old path.
 *
 *   C[m][n] = sum over t of A_t * B_t, chained through beta,
 *     A_t[i][p] = weight[i * k * taps + p * taps + t]   (gathered into scratch)
 *     B_t       = b + t * b_tap_stride, row stride ldb
 *
 * `gather` is [m][k] caller-owned scratch; its contents after the call are
 * unspecified.  Returns 0 done, 1 refused, -1 bad arguments. */
int mynah_sgemm_f32_conv_taps(size_t m, size_t n, size_t k, size_t taps,
                              const float *weight, float *gather,
                              const float *b, size_t ldb, size_t b_tap_stride,
                              float beta, float *c, size_t ldc);

/* The definition of correctness: the naive i,j,p triple loop, always
 * compiled, never vectorised, never threaded.  Every kernel in sgemm.c is
 * checked against THIS by mynah_sgemm_self_test(), and src/backend.c uses it
 * as the BLAS=scalar fallback so that the reference exists once rather than
 * twice.  Same arguments and same semantics as mynah_sgemm_f32. */
void mynah_sgemm_f32_reference(int trans_a, int trans_b,
                               size_t m, size_t n, size_t k,
                               float alpha,
                               const float *a, size_t lda,
                               const float *b, size_t ldb,
                               float beta,
                               float *c, size_t ldc);

/* ------------------------------------------------------------------------
 * Families
 *
 * A family is a RESIDENCY STRATEGY, not a tuning constant.  The micro-kernel
 * is shared on purpose (AGENTS.md coding rule 5: one formula, one reference);
 * what differs is which operand stays in registers and how many times the
 * other one is streamed.
 * ------------------------------------------------------------------------ */
typedef enum {
    /* Degenerate (m, n or k == 0), too small to pay for blocking, or a
     * transpose combination no kernel here handles.  Runs the reference. */
    MYNAH_SGEMM_FAMILY_REFERENCE = 0,
    /* op(B) transposed: op(B)'s columns are B's ROWS, contiguous in k.  There
     * is nothing to vectorise across the n axis, so this is n*m dot products
     * over the existing mynah_dot_f32 kernel. */
    MYNAH_SGEMM_FAMILY_DOT = 1,
    /* m == 1.  One output row: no row blocking exists, and the 129 calls of
     * m=1 n=1920 k=64 per utterance must not pay for a blocked kernel. */
    MYNAH_SGEMM_FAMILY_MATVEC = 2,
    /* n <= mynah_sgemm_narrow_max(): the ENTIRE C row block fits in the
     * register file, so op(A) -- the weights -- is streamed exactly once and
     * the n columns never leave registers.  This is the 28% shape. */
    MYNAH_SGEMM_FAMILY_NARROW = 3,
    /* Everything wider: an ordinary panel GEMM, column panels sized so the
     * op(B) panel stays in cache while the rows of op(A) sweep past it. */
    MYNAH_SGEMM_FAMILY_PANEL = 4
} mynah_sgemm_family;

/* The predicate.  Exported so the dispatch report can CALL it instead of
 * restating "n <= 16" -- a report that recomputes the condition can agree
 * with the source and both be wrong (dispatch.h, the central rule).
 * `why` (optional) receives a static string naming the clause that decided. */
mynah_sgemm_family mynah_sgemm_family_for(int trans_a, int trans_b,
                                          size_t m, size_t n, size_t k,
                                          const char **why);

const char *mynah_sgemm_family_name(mynah_sgemm_family family);

/* Which micro-kernel ISA this translation unit compiled: "neon", "avx2" or
 * "scalar".  A fact about the build, answered by the file that owns it. */
const char *mynah_sgemm_isa_name(void);

/* The narrow/panel boundary, in columns.  DERIVED, not chosen:
 *
 *     narrow_max = lanes * (accumulator budget / micro-kernel rows)
 *
 * with the accumulator budget set to half the architectural vector register
 * file so the b operand and the broadcast a values still have somewhere to
 * live.  NEON (32 regs, 4 lanes): 4 * (16/4) = 16.  AVX2 (16 regs, 8 lanes):
 * 8 * (8/4) = 16.  Scalar: 1 * (16/4) = 4.  The measured 28% shape has n=16
 * and lands inside it on both vector ISAs by arithmetic, not by fiat. */
size_t mynah_sgemm_narrow_max(void);

/* ------------------------------------------------------------------------
 * What actually ran
 * ------------------------------------------------------------------------ */

/* Per-family call counts since process start.  Counters, not predicates:
 * --dispatch-map builds its table before any model is loaded, so at that
 * moment no GEMM has run and the row must say so rather than present a clean
 * zero as health (.work/engineering-method.md §4). */
typedef struct {
    unsigned long long calls;
    unsigned long long reference;
    unsigned long long dot;
    unsigned long long matvec;
    unsigned long long narrow;
    unsigned long long panel;
    unsigned long long refused;   /* bad arguments, nothing computed */
    /* Conv-tap regions: `fused` is the number of mynah_sgemm_f32_conv_taps
     * calls that ran as ONE pool region, `fused_taps` the GEMMs those regions
     * carried (also counted in `calls` and in the family row, because the
     * arithmetic is the same), and `fused_refused` the calls that fell back to
     * the caller's own per-tap loop.  Three numbers rather than one because
     * "the fusion never applied" and "the fusion was never reached" are
     * different failures and a single counter cannot tell them apart. */
    unsigned long long fused;
    unsigned long long fused_taps;
    unsigned long long fused_refused;
} mynah_sgemm_stats;

void mynah_sgemm_stats_get(mynah_sgemm_stats *out);
void mynah_sgemm_stats_reset(void);

/* ------------------------------------------------------------------------
 * Self-test
 * ------------------------------------------------------------------------ */

/* Model-free.  Runs every COMPILED path against mynah_sgemm_f32_reference()
 * over the eleven measured shapes plus the edge cases the blocking can get
 * wrong: k not a multiple of the unroll, m == 1, n == 1, beta zero and
 * non-zero, alpha != 1, both transpose flags, and a thread-count sweep.
 * Refuses (returns -1) if a family never actually ran, because a forced path
 * that silently degraded to the reference would make the comparison vacuous.
 * 0 = pass. */
int mynah_sgemm_self_test(char *error, size_t error_capacity);

/* Force one family regardless of the predicate, for the self-test.  `ran`
 * (optional) receives the family that was actually executed, which may be
 * REFERENCE when the requested one cannot serve this operand layout.  Not for
 * production callers: mynah_sgemm_f32 is the entry point. */
int mynah_sgemm_f32_forced(mynah_sgemm_family want, mynah_sgemm_family *ran,
                           int trans_a, int trans_b,
                           size_t m, size_t n, size_t k,
                           float alpha,
                           const float *a, size_t lda,
                           const float *b, size_t ldb,
                           float beta,
                           float *c, size_t ldc);

/* Registers this file's dispatch predicates.  See dispatch.h. */
void mynah_sgemm_dispatch_probes(void);

#ifdef __cplusplus
}
#endif
#endif /* MYNAH_TTS_SGEMM_H */
