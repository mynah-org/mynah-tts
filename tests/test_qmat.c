/*
 * The int8/int4 determinism suite (PLAN.md E4-20b).
 *
 *     make qmat-test
 *
 * WHY THIS EXISTS
 *
 * src/qmat.c holds more than one kernel for the same arithmetic -- SDOT and
 * SMMLA on ARM, scalar / AVX2 / VEX-VNNI / EVEX-VNNI on x86 -- and the
 * weight-stationary batched linear routes a row through one or another
 * according to the size of the batch and the row's position in it.  Under
 * concurrent serving that position is decided by arrival order.  So "these
 * kernels compute the same thing" is not a style preference here; it is the
 * difference between a request's audio being a function of its input and being
 * a function of who else happened to be in flight.
 *
 * `--self-test` carries the pass/fail gates for that property.  This binary
 * carries the gates PLUS the two things a pass/fail cannot say:
 *
 *   1. WHICH KERNEL ACTUALLY RAN.  Printed, every time, for the int8 dot, the
 *      f16 path and the SMMLA wiring.  A green suite on a machine that
 *      silently fell back to the scalar reference is not evidence about the
 *      vector kernel, and on a CI runner nobody can log into it is the only
 *      way to find out.  This is what makes `make qmat-test` under
 *      MYNAH_QMAT_VNNI=256 / =512 a real observation of x86 silicon rather
 *      than a claim about it.
 *
 *   2. THE GROUPING PIN.  The float epilogue is `(float)s * ws * sx + bias`,
 *      a three-factor product that -ffast-math may group three ways, and the
 *      compiler chooses per site -- per inlined copy, even.  That is the
 *      defect this suite was written after: it made one activation row produce
 *      four different answers depending on where it sat in the batch.  §2
 *      below does not ask whether the kernels agree with each other (the
 *      self-test does that); it asks WHICH grouping and how many roundings the
 *      compiler actually emitted, against references this translation unit
 *      computes with its own value barriers.  If a future compiler, flag or
 *      -march regroups src/qmat.c's epilogue, §2 names the change instead of
 *      leaving it to be discovered as a 1 ULP audio difference under load.
 *
 * Model-free and millisecond-fast, so it runs inside `make test` and therefore
 * inside ubsan and asan.
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "qmat.h"

static int checks = 0;
static int failures = 0;

static void ok(const char *what, int good, const char *detail) {
    ++checks;
    if (good) {
        printf("  ok   %s\n", what);
        return;
    }
    ++failures;
    printf("  FAIL %s: %s\n", what, detail == NULL ? "(no detail)" : detail);
}

/* The same value barrier src/qmat.c uses, defined independently here: the
 * references below must be computed in an order the optimizer cannot change,
 * or this file would be pinning nothing.  It emits no instruction. */
#if defined(__GNUC__) && defined(__aarch64__)
#define FREEZE(v) __asm__("" : "+w"(v))
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#define FREEZE(v) __asm__("" : "+x"(v))
#elif defined(__GNUC__)
#define FREEZE(v) __asm__("" : "+m"(v))
#else
#define FREEZE(v) do { volatile float frz_ = (v); (v) = frz_; } while (0)
#endif

static float frozen(float v) { FREEZE(v); return v; }

/* The three groupings of `(float)s * ws * sx`, each pinned. */
static float group_ws_first(int32_t s, float ws, float sx) {
    return frozen(frozen((float)s * ws) * sx);
}
static float group_scale_first(int32_t s, float ws, float sx) {
    return frozen((float)s * frozen(ws * sx));
}
static float group_sx_first(int32_t s, float ws, float sx) {
    return frozen(frozen((float)s * sx) * ws);
}

/* The two ways `p + bias` can round: fused (one rounding) or not (two). */
static float epi_fused(int32_t s, float ws, float sx, float bias) {
    return fmaf((float)s, frozen(ws * sx), bias);
}
static float epi_split(int32_t s, float ws, float sx, float bias) {
    return frozen((float)s * frozen(ws * sx)) + bias;
}

/* -------------------------------------------------------------------------
 * 1. What resolved on this host.  Printed, not asserted: the point is the
 *    record.  A CI log with these four lines in it is evidence about the
 *    runner's silicon; a "PASS" on its own is not.
 * ---------------------------------------------------------------------- */
static void report_resolution(void) {
    const char *why = NULL;
    printf("\n1. WHAT RESOLVED ON THIS HOST\n");
    const char *int8 = mynah_qmat_int8_kernel(&why);
    printf("   int8 kernel   : %-12s %s\n", int8, why == NULL ? "" : why);
    why = NULL;
    const char *f16 = mynah_qmat_f16_kernel(&why);
    printf("   f16 kernel    : %-12s %s\n", f16, why == NULL ? "" : why);
    why = NULL;
    const int i8mm = mynah_qmat_i8mm_enabled(&why);
    printf("   smmla wiring  : %-12s %s\n", i8mm ? "on" : "off",
           why == NULL ? "" : why);
    const char *vnni = getenv("MYNAH_QMAT_VNNI");
    const char *force = getenv("MYNAH_QMAT_I8MM");
    printf("   env           : MYNAH_QMAT_VNNI=%s MYNAH_QMAT_I8MM=%s\n",
           vnni == NULL ? "unset" : vnni, force == NULL ? "unset" : force);
    /* A requested VNNI level that did not resolve is the single most
     * misleading way for an x86 measurement to be wrong, so say it plainly. */
    if (vnni != NULL && (strcmp(vnni, "256") == 0 || strcmp(vnni, "512") == 0)) {
        const int got = (strcmp(vnni, "512") == 0)
                            ? (strcmp(int8, "avx512vnni") == 0)
                            : (strcmp(int8, "avxvnni") == 0);
        printf("   VNNI REQUEST  : %s -> %s\n", vnni,
               got ? "RESOLVED, the VPDPBUSD kernel below really executed"
                   : "NOT RESOLVED -- this CPU does not have that unit, so "
                     "nothing below is evidence about VPDPBUSD");
    }
}

/* -------------------------------------------------------------------------
 * 2. The grouping pin.
 * ---------------------------------------------------------------------- */
static void grouping_pin(void) {
    printf("\n2. THE FLOAT EPILOGUE, PINNED\n");
    /* Adversarial-ish but deterministic: a spread of magnitudes so that some
     * triple somewhere lands on a rounding boundary in one grouping and not
     * in another.  If none did, the pin below would be vacuous, which is
     * exactly what `discriminating` checks. */
    enum { TRIPLES = 4096 };
    int mismatch_canonical = -1;
    int differ_ws_first = 0, differ_sx_first = 0, differ_rounding = 0;
    int fused_here = 0, split_here = 0;

    for (int t = 0; t < TRIPLES; ++t) {
        const int32_t s = (int32_t)((t * 2654435761u) % 3000001u) - 1500000;
        const float ws = 1.0e-5f * (1.0f + (float)(t % 977) * 0.013f);
        const float sx = 3.0e-3f * (1.0f + (float)(t % 811) * 0.017f);
        const float bias = (float)((t % 401) - 200) * 0.0078125f;

        const float canonical = mynah_qmat_epilogue(s, ws, sx, 0.0f);
        const float want = group_scale_first(s, ws, sx);
        if (mismatch_canonical < 0 && memcmp(&canonical, &want, sizeof want) != 0)
            mismatch_canonical = t;
        const float other_a = group_ws_first(s, ws, sx);
        const float other_c = group_sx_first(s, ws, sx);
        if (memcmp(&want, &other_a, sizeof want) != 0) ++differ_ws_first;
        if (memcmp(&want, &other_c, sizeof want) != 0) ++differ_sx_first;

        /* How many times does the bias add round? */
        const float withbias = mynah_qmat_epilogue(s, ws, sx, bias);
        const float f = epi_fused(s, ws, sx, bias);
        const float g = epi_split(s, ws, sx, bias);
        if (memcmp(&f, &g, sizeof f) != 0) {
            ++differ_rounding;
            if (memcmp(&withbias, &f, sizeof f) == 0) ++fused_here;
            else if (memcmp(&withbias, &g, sizeof g) == 0) ++split_here;
        }
    }

    printf("   canonical grouping   : (float)s * (ws * sx)\n");
    printf("   triples where the other groupings differ: ws-first %d/%d, "
           "sx-first %d/%d\n", differ_ws_first, (int)TRIPLES, differ_sx_first,
           (int)TRIPLES);

    /* The pin is only worth anything if the three groupings are actually
     * distinguishable on this host.  Say so either way. */
    const int discriminating = differ_ws_first > 0 && differ_sx_first > 0;
    ok("the grouping pin discriminates (the three groupings are not all equal "
       "on this host)", discriminating,
       "all three groupings agreed on every triple -- this pin proves nothing "
       "here and the sweep needs harder inputs");

    char detail[256];
    snprintf(detail, sizeof detail,
             "src/qmat.c's epilogue first disagreed with (float)s * (ws * sx) "
             "at triple %d -- the compiler regrouped it", mismatch_canonical);
    ok("src/qmat.c groups the epilogue as (float)s * (ws * sx)",
       mismatch_canonical < 0, detail);

    printf("   triples where fused and split rounding differ: %d/%d "
           "(matched fused %d, matched split %d)\n",
           differ_rounding, (int)TRIPLES, fused_here, split_here);
    if (differ_rounding == 0) {
        printf("   (fused and split agree on every triple here, so there is "
               "nothing for this pin to distinguish)\n");
    } else {
        ok("the bias add rounds the same way at every input (all fused or all "
           "split, never a mixture)",
           fused_here == differ_rounding || split_here == differ_rounding,
           "src/qmat.c's epilogue contracts into an FMA at some inputs and not "
           "at others, which is how a row's answer starts depending on its "
           "call site");
        printf("   src/qmat.c rounds the bias add: %s\n",
               fused_here == differ_rounding ? "ONCE (fused into an FMA)"
                                             : "TWICE (multiply, then add)");
    }
}

/* -------------------------------------------------------------------------
 * 3. The suite the runtime carries, run here too so ubsan/asan see it.
 * ---------------------------------------------------------------------- */
static void runtime_self_test(void) {
    printf("\n3. mynah_qmat_self_test()\n");
    char error[512] = {0};
    ok("mynah_qmat_self_test (u8 identity, SMMLA identity, batched, "
       "batch membership at INT8/INT4/F16, SMMLA on-vs-off)",
       mynah_qmat_self_test(error, sizeof error) == 0, error);
}

/* -------------------------------------------------------------------------
 * 4. The public batched API, both wirings, shapes the self-test does not use.
 *
 *    The self-test's shapes are square-ish and even.  These are the awkward
 *    ones: one row, two rows, a prime row count, a k that is not a multiple of
 *    the vector width, and batches that leave an odd activation for the
 *    fallback.  Bit-equality against the same row computed alone, which is the
 *    literal promise in src/mynah_tts.h.
 * ---------------------------------------------------------------------- */
static void awkward_shapes(void) {
    printf("\n4. AWKWARD SHAPES, SMMLA FORCED OFF AND ON\n");
    static const size_t rows[6] = { 1u, 2u, 3u, 17u, 31u, 64u };
    static const size_t cols[6] = { 32u, 33u, 64u, 40u, 96u, 8u };
    enum { BMAX = 4 };
    const int saved = mynah_qmat_i8mm_force(-1);
    int bad = 0;
    char detail[320] = {0};

    for (size_t sh = 0; sh < 6u && bad == 0; ++sh) {
        const size_t N = rows[sh], K = cols[sh];
        float *w = (float *)malloc(N * K * sizeof(float));
        float *x = (float *)malloc((size_t)BMAX * K * sizeof(float));
        float *bias = (float *)malloc(N * sizeof(float));
        float *alone = (float *)malloc((size_t)BMAX * N * sizeof(float));
        float *got = (float *)malloc((size_t)BMAX * N * sizeof(float));
        int8_t *qx = (int8_t *)malloc((size_t)BMAX * K);
        float *sx = (float *)malloc((size_t)BMAX * sizeof(float));
        mynah_qmat_cache *cache = mynah_qmat_cache_new(1 /* int8 */);
        if (w == NULL || x == NULL || bias == NULL || alone == NULL ||
            got == NULL || qx == NULL || sx == NULL || cache == NULL) {
            snprintf(detail, sizeof detail, "out of memory at shape %zux%zu", N, K);
            bad = 1;
            goto next;
        }
        if (mynah_qmat_cache_qtype(cache) != 1) goto next;  /* int8 unavailable */
        for (size_t i = 0; i < N * K; ++i)
            w[i] = sinf(0.031f * (float)i) * (0.5f + 0.5f * cosf(0.0023f * (float)i));
        for (size_t i = 0; i < (size_t)BMAX * K; ++i)
            x[i] = cosf(0.029f * (float)i) - 0.35f;
        for (size_t i = 0; i < N; ++i) bias[i] = (float)i * 0.03125f - 0.5f;

        for (int mode = 0; mode < 2 && bad == 0; ++mode) {
            mynah_qmat_i8mm_force(mode);
            char err[256] = {0};
            for (size_t r = 0; r < (size_t)BMAX; ++r) {
                if (mynah_qmat_linear_resolved(cache, NULL, "awkward.shapes", w,
                                               x + r * K, alone + r * N, 1u, K,
                                               N, bias, err, sizeof err) != 0) {
                    snprintf(detail, sizeof detail, "%zux%zu solo: %s", N, K, err);
                    bad = 1;
                    break;
                }
            }
            for (size_t batch = 1u; batch <= (size_t)BMAX && bad == 0; ++batch) {
                const float *in_rows[BMAX];
                float *out_rows[BMAX];
                for (size_t b = 0; b < batch; ++b) {
                    in_rows[b] = x + b * K;
                    out_rows[b] = got + b * N;
                }
                if (mynah_qmat_linear_batched(cache, NULL, "awkward.shapes", w,
                                              in_rows, out_rows, batch, K, N,
                                              bias, qx, sx, err,
                                              sizeof err) != 0) {
                    snprintf(detail, sizeof detail, "%zux%zu batch %zu: %s",
                             N, K, batch, err);
                    bad = 1;
                    break;
                }
                for (size_t b = 0; b < batch && bad == 0; ++b) {
                    for (size_t i = 0; i < N; ++i) {
                        if (memcmp(&alone[b * N + i], &got[b * N + i],
                                   sizeof(float)) == 0) {
                            continue;
                        }
                        snprintf(detail, sizeof detail,
                                 "%zux%zu smmla=%d batch %zu row %zu col %zu: "
                                 "alone %.9g vs batched %.9g",
                                 N, K, mode, batch, b, i,
                                 (double)alone[b * N + i],
                                 (double)got[b * N + i]);
                        bad = 1;
                        break;
                    }
                }
            }
        }
    next:
        mynah_qmat_cache_free(cache);
        free(w); free(x); free(bias); free(alone); free(got); free(qx); free(sx);
    }
    mynah_qmat_i8mm_force(saved);
    ok("every awkward shape is bit-identical batched and alone, with the SMMLA "
       "wiring off and on", bad == 0, detail);
}

int main(void) {
    printf("qmat determinism suite\n");
    report_resolution();
    grouping_pin();
    runtime_self_test();
    awkward_shapes();
    printf("\nqmat: %d checks, %d failures\n", checks, failures);
    if (failures != 0) {
        printf("qmat: FAIL\n");
        return 1;
    }
    printf("qmat: PASS\n");
    return 0;
}
