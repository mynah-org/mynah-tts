/*
 * Model-free self tests for the hot kernels (PLAN.md E3-6), and the numerical
 * qualification for the Accelerate replacements (PLAN.md E4-16d).
 *
 *     make kernels-test
 *
 * WHY THIS EXISTS AND WHAT IT IS FOR
 *
 * A kernel without a model-free self test cannot be trusted on a machine we
 * have not built on, and this project now has two of those.  `--self-test`
 * carries the pass/fail gates; this binary carries the same gates PLUS the
 * measurements behind them, printed, because "it passes" and "here is how far
 * it is from the true value" are different claims and only the second one
 * survives being moved to another compiler.
 *
 * Three things live here that live nowhere else:
 *
 *   1. THE ULP TABLE.  Every transcendental we now own, plus libm, plus --
 *      on macOS -- Accelerate's vvtanhf and vvsinf, all measured against a
 *      double-precision evaluation of the same function.  This is the
 *      evidence for whether replacing a vendor routine fixed the development
 *      platform or degraded it.  It is not a benchmark and prints no time.
 *
 *   2. THE TWO LAYERNORMS.  src/kernels.c and src/flow_head.c each carry a
 *      LayerNorm with bias, at two different epsilons, written by different
 *      hands.  Nothing in this tree had ever compared them.  §3 below either
 *      shows them to be the same function or prints exactly where they part.
 *
 *   3. THE TAIL SWEEP.  Every vectorised kernel in src/kernels.h run at every
 *      length from 0 to 40 against an independent f64 reference.  Almost all
 *      the existing coverage uses three-element arrays, which never reach a
 *      vector body at all, so a tail bug -- the classic SIMD bug -- could not
 *      have been caught.
 *
 * It links $(CORE_OBJECTS) because the kernels are spread over kernels.c,
 * flow_head.c and conv1d.c and the point is to compare them with each other.
 */
#include "kernels.h"
#include "flow_head.h"
#include "conv1d.h"
#include "dispatch.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(MYNAH_USE_ACCELERATE)
#include <Accelerate/Accelerate.h>
#endif

double mynah_vecmath_ulp(float got, double want); /* src/kernels.c */
int mynah_vecmath_denormals_flush(void);                 /* src/kernels.c */

static int failures = 0;
static int checks = 0;

#define CHECK(cond, ...)                                                   \
    do {                                                                   \
        ++checks;                                                          \
        if (!(cond)) {                                                     \
            ++failures;                                                    \
            printf("  FAIL %s:%d  ", __func__, __LINE__);                  \
            printf(__VA_ARGS__);                                           \
            printf("\n");                                                  \
        }                                                                  \
    } while (0)

/* Error relative to the SCALE of the problem, not to the size of the answer.
 *
 * Every norm and every GELU in this file has a zero: a row element that sits
 * on the mean, an input far enough negative that the GELU cancels.  At such a
 * point the ULP of the RESULT is unbounded for any implementation, including
 * a correct one, so a ULP gate there measures the conditioning of the input
 * and not the quality of the kernel.  These functions are measured against
 * the scale they live on instead -- the row's largest output, or 1 + |x| for
 * a GELU, which is the bound on |GELU(x)| itself. */
static double rel_to_scale(float got, double want, double scale) {
    if (!(scale > 1.0e-30)) scale = 1.0e-30;
    return fabs((double)got - want) / scale;
}

static int is_zero_bits(float x) {
    uint32_t b;
    memcpy(&b, &x, sizeof b);
    return (b & 0x7fffffffu) == 0u;
}

static int is_subnormal(float x) {
    uint32_t b;
    memcpy(&b, &x, sizeof b);
    return (b & 0x7f800000u) == 0u && (b & 0x007fffffu) != 0u;
}

static int same_bits(float a, float b) {
    uint32_t x, y;
    memcpy(&x, &a, sizeof x);
    memcpy(&y, &b, sizeof y);
    return x == y;
}

/* A deterministic pseudo-random float in [lo, hi].  Not rand(): the whole
 * value of this file is that a failure is reproducible on another machine. */
static uint32_t rng_state = 0x9e3779b9u;
static void rng_seed(uint32_t s) { rng_state = s | 1u; }
static float rng_float(float lo, float hi) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    const double u = (double)(rng_state >> 8) / (double)(1u << 24);
    return (float)((double)lo + ((double)hi - (double)lo) * u);
}

/* ======================================================================
 * 1. The ULP table
 * ====================================================================== */
typedef struct {
    const char *name;
    double worst_ulp;
    double worst_abs;
    float at;
} accuracy;

static void accuracy_note(accuracy *a, float got, double truth, float x) {
    const double u = mynah_vecmath_ulp(got, truth);
    const double e = fabs((double)got - truth);
    if (u > a->worst_ulp) { a->worst_ulp = u; a->at = x; }
    if (e > a->worst_abs) a->worst_abs = e;
}

static void ulp_table(void) {
    enum { N = 4096, SWEEPS = 240 };
    static float in[N], out[N];
    printf("\n-- accuracy against a double-precision reference "
           "(vector ISA: %s) --\n", mynah_vecmath_isa());

    /* ---- tanh, over the whole line including both sides of the knee ---- */
    {
        accuracy ours = {"mynah_tanh_f32", 0, 0, 0};
        accuracy ours_s = {"mynah_tanh_f32_scalar", 0, 0, 0};
        accuracy libm = {"libm tanhf", 0, 0, 0};
#if defined(MYNAH_USE_ACCELERATE)
        accuracy vf = {"Accelerate vvtanhf", 0, 0, 0};
        static float acc_out[N];
#endif
        rng_seed(0xA11CE);
        for (int s = 0; s < SWEEPS; ++s) {
            /* Spend most of the sweeps where the model lives (|x| < 4) and
             * the rest on the knee and the saturated tail. */
            const float span = (s % 4 == 3) ? 40.0f : 4.0f;
            for (int i = 0; i < N; ++i) in[i] = rng_float(-span, span);
            mynah_tanh_f32(in, out, N);
            for (int i = 0; i < N; ++i)
                accuracy_note(&ours, out[i], tanh((double)in[i]), in[i]);
            mynah_tanh_f32_scalar(in, out, N);
            for (int i = 0; i < N; ++i)
                accuracy_note(&ours_s, out[i], tanh((double)in[i]), in[i]);
            for (int i = 0; i < N; ++i)
                accuracy_note(&libm, tanhf(in[i]), tanh((double)in[i]), in[i]);
#if defined(MYNAH_USE_ACCELERATE)
            {
                const int n = N;
                vvtanhf(acc_out, in, &n);
                for (int i = 0; i < N; ++i)
                    accuracy_note(&vf, acc_out[i], tanh((double)in[i]), in[i]);
            }
#endif
        }
        printf("   tanh, %d samples over [-4,4] and [-40,40]\n", N * SWEEPS);
        printf("     %-24s %8.2f ulp  %.3e abs (worst at %+.6f)\n",
               ours.name, ours.worst_ulp, ours.worst_abs, (double)ours.at);
        printf("     %-24s %8.2f ulp  %.3e abs\n", ours_s.name,
               ours_s.worst_ulp, ours_s.worst_abs);
        printf("     %-24s %8.2f ulp  %.3e abs\n", libm.name, libm.worst_ulp,
               libm.worst_abs);
#if defined(MYNAH_USE_ACCELERATE)
        printf("     %-24s %8.2f ulp  %.3e abs\n", vf.name, vf.worst_ulp,
               vf.worst_abs);
        CHECK(ours.worst_ulp <= vf.worst_ulp * 4.0,
              "our tanh (%.2f ulp) is more than 4x worse than vvtanhf "
              "(%.2f ulp) -- this substitution would be a degradation",
              ours.worst_ulp, vf.worst_ulp);
#endif
        CHECK(ours.worst_ulp <= 2.0, "tanh vector %.2f ulp", ours.worst_ulp);
        CHECK(ours_s.worst_ulp <= 2.0, "tanh scalar %.2f ulp",
              ours_s.worst_ulp);
    }

    /* ---- sin, including far from the origin ------------------------- */
    {
        static const struct { float lo, hi; const char *label; } bands[] = {
            {-6.3f, 6.3f, "[-2pi, 2pi]"},
            {-400.0f, 400.0f, "[-400, 400]"},
            {-65000.0f, 65000.0f, "[-65000, 65000]"},
        };
        for (size_t b = 0; b < sizeof(bands) / sizeof(bands[0]); ++b) {
            accuracy ours = {"mynah_sin_f32", 0, 0, 0};
            accuracy ours_s = {"mynah_sin_f32_scalar", 0, 0, 0};
            accuracy libm = {"libm sinf", 0, 0, 0};
#if defined(MYNAH_USE_ACCELERATE)
            accuracy vf = {"Accelerate vvsinf", 0, 0, 0};
            static float acc_out[N];
#endif
            rng_seed(0x5115u + (uint32_t)b);
            for (int s = 0; s < SWEEPS; ++s) {
                for (int i = 0; i < N; ++i)
                    in[i] = rng_float(bands[b].lo, bands[b].hi);
                mynah_sin_f32(in, out, N);
                for (int i = 0; i < N; ++i)
                    accuracy_note(&ours, out[i], sin((double)in[i]), in[i]);
                mynah_sin_f32_scalar(in, out, N);
                for (int i = 0; i < N; ++i)
                    accuracy_note(&ours_s, out[i], sin((double)in[i]), in[i]);
                for (int i = 0; i < N; ++i)
                    accuracy_note(&libm, sinf(in[i]), sin((double)in[i]),
                                  in[i]);
#if defined(MYNAH_USE_ACCELERATE)
                {
                    const int n = N;
                    vvsinf(acc_out, in, &n);
                    for (int i = 0; i < N; ++i)
                        accuracy_note(&vf, acc_out[i], sin((double)in[i]),
                                      in[i]);
                }
#endif
            }
            /* Near a zero of sin the ULP of the result is unbounded for every
             * implementation, so the meaningful column here is absolute. */
            printf("   sin over %s, %d samples (absolute error is the "
                   "meaningful column)\n", bands[b].label, N * SWEEPS);
            printf("     %-24s %.3e abs\n", ours.name, ours.worst_abs);
            printf("     %-24s %.3e abs\n", ours_s.name, ours_s.worst_abs);
            printf("     %-24s %.3e abs\n", libm.name, libm.worst_abs);
#if defined(MYNAH_USE_ACCELERATE)
            printf("     %-24s %.3e abs\n", vf.name, vf.worst_abs);
            CHECK(ours.worst_abs <= vf.worst_abs * 4.0,
                  "our sin (%.3e) is more than 4x worse than vvsinf (%.3e) "
                  "over %s", ours.worst_abs, vf.worst_abs, bands[b].label);
#endif
            CHECK(ours.worst_abs <= 2.0e-7,
                  "sin vector worst absolute error %.3e over %s",
                  ours.worst_abs, bands[b].label);
            CHECK(ours_s.worst_abs <= 2.0e-7,
                  "sin scalar worst absolute error %.3e over %s",
                  ours_s.worst_abs, bands[b].label);
            CHECK(fabs(ours.worst_abs - ours_s.worst_abs) < 1.0e-7,
                  "sin scalar and %s have different error profiles over %s "
                  "(%.3e vs %.3e) -- they are not the same algorithm",
                  mynah_vecmath_isa(), bands[b].label, ours_s.worst_abs,
                  ours.worst_abs);
        }
    }
}

/* ======================================================================
 * 2. Transcendentals at the places they go wrong
 * ====================================================================== */
static void transcendental_edges(void) {
    printf("\n-- transcendental edge cases --\n");
    /* Buffers long enough that the VECTOR body runs on these, not the tail. */
    enum { N = 64 };
    float in[N], got[N], got_s[N];
    const int ftz = mynah_vecmath_denormals_flush();

    const float tanh_specials[] = {
        0.0f, -0.0f, 1.0e-40f, -1.0e-40f, 5.0e-45f, -5.0e-45f,
        1.0e-20f, -1.0e-20f, 0.624999f, 0.625f, 0.625001f,
        9.0f, 9.010913f, 9.010914f, 1.0e30f, -1.0e30f,
        INFINITY, -INFINITY, NAN, -NAN, 3.4028235e38f, -3.4028235e38f
    };
    const size_t tc = sizeof(tanh_specials) / sizeof(tanh_specials[0]);
    printf("   denormal inputs: this build %s (a property of the compiler "
           "driver, not of the kernels)\n",
           ftz ? "FLUSHES them to zero before the kernels see them"
               : "preserves them");
    for (size_t i = 0; i < N; ++i) in[i] = tanh_specials[i % tc];
    mynah_tanh_f32(in, got, N);
    mynah_tanh_f32_scalar(in, got_s, N);
    for (size_t i = 0; i < N; ++i) {
        if (in[i] != in[i]) {
            CHECK(got[i] != got[i], "tanh(NaN) = %.9g", (double)got[i]);
            continue;
        }
        /* A denormal on a flush-to-zero build is not comparable between the
         * two bodies -- the vector one runs with FPCR.FZ and the scalar one
         * may be constant-folded at compile time, where FZ does not exist.
         * See the long note in mynah_vecmath_self_test. */
        if (ftz && is_subnormal(in[i])) continue;
        /* Bits where the answer comes from a selection; 2 ulp where it comes
         * from a polynomial the compiler may contract differently in C than
         * in intrinsics. See the note in mynah_vecmath_self_test. */
        if (is_zero_bits(in[i]) || fabsf(in[i]) > 9.010913f) {
            CHECK(same_bits(got[i], got_s[i]),
                  "tanh(%.9g): %s gives %.9g, scalar gives %.9g (bits differ "
                  "in a case decided by selection)", (double)in[i],
                  mynah_vecmath_isa(), (double)got[i], (double)got_s[i]);
        } else {
            CHECK(mynah_vecmath_ulp(got[i], (double)got_s[i]) <= 2.0,
                  "tanh(%.9g): %s gives %.9g, scalar gives %.9g",
                  (double)in[i], mynah_vecmath_isa(), (double)got[i],
                  (double)got_s[i]);
        }
    }
    CHECK(same_bits(got[0], 0.0f), "tanh(+0) lost its sign");
    CHECK(same_bits(got[1], -0.0f), "tanh(-0) = %.9g, want -0",
          (double)got[1]);
    if (ftz) {
        CHECK((is_zero_bits(got[2]) || same_bits(got[2], 1.0e-40f)) &&
              (is_zero_bits(got[3]) || same_bits(got[3], -1.0e-40f)),
              "flush-to-zero build: tanh of a denormal must be that denormal "
              "or a zero, got %.9g / %.9g", (double)got[2], (double)got[3]);
        CHECK(is_zero_bits(got[4]) || same_bits(got[4], 5.0e-45f),
              "flush-to-zero build: tanh of the smallest denormal must be "
              "that denormal or a zero");
    } else {
        CHECK(same_bits(got[2], 1.0e-40f),
              "tanh(denormal) must be the denormal");
        CHECK(same_bits(got[4], 5.0e-45f),
              "tanh(smallest denormal) = %.9g", (double)got[4]);
        CHECK(same_bits(got[5], -5.0e-45f),
              "tanh(-smallest denormal) = %.9g", (double)got[5]);
    }
    CHECK(got[16] == 1.0f && got[17] == -1.0f, "tanh(+-Inf) must be +-1");
    /* The knee is at 9.010913, not at 9: tanh(9) = 1 - 3.05e-8 and the float
     * below 1.0 is 1 - 5.96e-8, so 9 still rounds DOWN.  Asserting both sides
     * is what keeps the constant honest -- a knee set too low saturates
     * values that should not saturate, and nothing else in the suite would
     * notice a 1-ulp saturation. */
    CHECK(got[11] == (float)tanh(9.0),
          "tanh(9) = %.9g, the correctly rounded value is %.9g -- the "
          "saturation knee is too low", (double)got[11], (double)(float)tanh(9.0));
    CHECK(got[11] != 1.0f, "tanh(9) must not saturate; the knee is 9.010913");
    CHECK(got[13] == 1.0f, "tanh(9.010914) must be exactly 1.0f");
    /* Monotone and bounded across the knee: the branch must not step. */
    {
        float prev = -2.0f;
        for (double x = 0.6f; x <= 9.2; x += 0.0009765625) {
            const float fx = (float)x;
            float v;
            mynah_tanh_f32_scalar(&fx, &v, 1);
            CHECK(v >= prev - 1.0e-7f && v <= 1.0f,
                  "tanh non-monotone or out of range at %.7f: %.9g after "
                  "%.9g", x, (double)v, (double)prev);
            if (v > prev) prev = v;
        }
    }

    const float sin_specials[] = {
        0.0f, -0.0f, 1.0e-40f, -1.0e-40f,
        3.14159265f, -3.14159265f, 1.5707963f, -1.5707963f,
        6.2831853f, 65535.9f, -65535.9f, 65536.0f, -65536.0f,
        99999.0f, -1.0e12f, 1.0e30f, INFINITY, -INFINITY, NAN
    };
    const size_t sc = sizeof(sin_specials) / sizeof(sin_specials[0]);
    for (size_t i = 0; i < N; ++i) in[i] = sin_specials[i % sc];
    mynah_sin_f32(in, got, N);
    mynah_sin_f32_scalar(in, got_s, N);
    for (size_t i = 0; i < N; ++i) {
        if (in[i] != in[i] || fabsf(in[i]) > 3.0e38f) {
            CHECK(got[i] != got[i], "sin(%.9g) = %.9g, want NaN",
                  (double)in[i], (double)got[i]);
            continue;
        }
        if (ftz && is_subnormal(in[i])) continue;
        if (fabsf(in[i]) >= 65536.0f || is_zero_bits(in[i])) {
            CHECK(same_bits(got[i], got_s[i]),
                  "sin(%.9g): %s gives %.9g, scalar gives %.9g -- both must "
                  "come from libm here", (double)in[i], mynah_vecmath_isa(),
                  (double)got[i], (double)got_s[i]);
        } else {
            CHECK(fabsf(got[i] - got_s[i]) <= 3.0e-7f,
                  "sin(%.9g): %s gives %.9g, scalar gives %.9g", (double)in[i],
                  mynah_vecmath_isa(), (double)got[i], (double)got_s[i]);
        }
    }
    CHECK(same_bits(got[0], 0.0f), "sin(+0) lost its sign");
    CHECK(same_bits(got[1], -0.0f), "sin(-0) = %.9g, want -0", (double)got[1]);
    /* Past the reduction limit the lane is handed to libm, so this is an
     * identity and not an approximation.  If the handover ever stops
     * happening this is the check that notices. */
    for (size_t i = 11; i <= 14; ++i) {
        const float x = sin_specials[i];
        float one[1] = {x}, r[1];
        mynah_sin_f32(one, r, 1);
        CHECK(same_bits(r[0], sinf(x)),
              "sin(%.9g) = %.9g but libm says %.9g -- the handover past the "
              "argument-reduction limit is gone", (double)x, (double)r[0],
              (double)sinf(x));
    }
    /* Oddness, by bit, at eight magnitudes.  A reduction that mishandles the
     * sign shows up here and nowhere else. */
    for (int k = 0; k < 8; ++k) {
        const float x = (float)ldexp(1.7, k * 3);
        float pos[1] = {x}, neg[1] = {-x}, rp[1], rn[1];
        mynah_sin_f32(pos, rp, 1);
        mynah_sin_f32(neg, rn, 1);
        CHECK(same_bits(rp[0], -rn[0]), "sin is not odd at %.9g: %.9g / %.9g",
              (double)x, (double)rp[0], (double)rn[0]);
        CHECK(rp[0] == 0.0f || (rp[0] < 0.0f) != (rn[0] < 0.0f),
              "sin lost its antisymmetry at %.9g", (double)x);
    }
    printf("   tanh and sin: zero, negative zero, denormals, the knee, "
           "+-Inf, NaN, oddness, libm handover\n");
}

/* ======================================================================
 * 3. The two LayerNorms.  E3-6 asks for them to be shown equivalent or
 *    shown to differ, and until now nothing had compared them.
 * ====================================================================== */
static void layernorm_two_ways(void) {
    printf("\n-- LayerNorm with bias: src/kernels.c vs src/flow_head.c --\n");
    enum { W = 512 };
    float x[W], w[W], b[W], a[W], c[W];
    double ref[W];
    /* Both epsilons that ship: the backbone LayerNorm runs at 1e-5, the flow
     * head's in_ln/norm_final at 1e-6. */
    const float epsilons[2] = {1.0e-5f, 1.0e-6f};
    double worst_pair = 0.0, worst_ref = 0.0;

    rng_seed(0x1AE12);
    for (int trial = 0; trial < 400; ++trial) {
        const size_t width = (size_t)(4 + (trial % 61));
        const float epsilon = epsilons[trial & 1];
        /* Well conditioned on purpose: the mean does not dwarf the spread.
         * The ill-conditioned case is measured separately below, because it
         * is a property of f32 LayerNorm and not a property of either of
         * these two implementations. */
        const float spread = (trial % 7 == 0) ? 0.01f : 3.0f;
        for (size_t i = 0; i < width; ++i) {
            x[i] = rng_float(-spread, spread);
            w[i] = rng_float(0.2f, 2.0f);
            b[i] = rng_float(-1.0f, 1.0f);
        }
        mynah_layernorm_f32(x, w, b, a, 1u, width, epsilon);
        mynah_flow_layernorm_f32(x, w, b, c, width, epsilon);

        /* An independent f64 statement of the contract, written from the
         * definition and not from either implementation. */
        double mean = 0.0;
        for (size_t i = 0; i < width; ++i) mean += (double)x[i];
        mean /= (double)width;
        double var = 0.0;
        for (size_t i = 0; i < width; ++i) {
            const double d = (double)x[i] - mean;
            var += d * d;
        }
        var /= (double)width;           /* biased, torch var(unbiased=False) */
        const double scale = 1.0 / sqrt(var + (double)epsilon);
        double row_scale = 0.0;
        for (size_t i = 0; i < width; ++i) {
            ref[i] = ((double)x[i] - mean) * scale * (double)w[i] + (double)b[i];
            if (fabs(ref[i]) > row_scale) row_scale = fabs(ref[i]);
        }

        for (size_t i = 0; i < width; ++i) {
            const double pair =
                rel_to_scale(a[i], (double)c[i], row_scale);
            const double r1 = rel_to_scale(a[i], ref[i], row_scale);
            const double r2 = rel_to_scale(c[i], ref[i], row_scale);
            if (pair > worst_pair) worst_pair = pair;
            if (r1 > worst_ref) worst_ref = r1;
            if (r2 > worst_ref) worst_ref = r2;
        }
    }
    printf("   400 rows, widths 4..64, epsilon 1e-5 and 1e-6, well conditioned\n");
    printf("     the two implementations agree to     %.2e of the row scale\n",
           worst_pair);
    printf("     both against an f64 reference within %.2e of the row scale\n",
           worst_ref);
    /* THE E3-6 ANSWER: they are the SAME function.  Same biased variance,
     * same epsilon inside the sqrt, same affine order.  What is left is the
     * accumulation order -- kernels.c vectorises the variance sum four or
     * eight lanes at a time, flow_head.c sums it straight -- and that is
     * worth a few f32 ulp, not a difference in definition. */
    CHECK(worst_pair <= 1.0e-5,
          "the two LayerNorms differ by %.2e of the row scale: that is a "
          "difference in DEFINITION, not in accumulation order", worst_pair);
    CHECK(worst_ref <= 1.0e-5,
          "a LayerNorm is %.2e of the row scale from the f64 reference",
          worst_ref);

    /* The ill-conditioned case, measured and recorded rather than asserted
     * away: when the mean dwarfs the spread, x[i] - mean cancels in f32 and
     * BOTH implementations lose digits.  Anyone normalising a row with a
     * large DC offset needs this number. */
    {
        double worst_cond = 0.0, worst_cond_pair = 0.0;
        for (int trial = 0; trial < 60; ++trial) {
            const size_t width = 32u;
            for (size_t i = 0; i < width; ++i) {
                x[i] = 10.0f + rng_float(-1.0e-4f, 1.0e-4f);
                w[i] = 1.0f;
                b[i] = 0.0f;
            }
            mynah_layernorm_f32(x, w, b, a, 1u, width, 1.0e-5f);
            mynah_flow_layernorm_f32(x, w, b, c, width, 1.0e-5f);
            double mean = 0.0;
            for (size_t i = 0; i < width; ++i) mean += (double)x[i];
            mean /= (double)width;
            double var = 0.0;
            for (size_t i = 0; i < width; ++i) {
                const double d = (double)x[i] - mean;
                var += d * d;
            }
            const double scale = 1.0 / sqrt(var / (double)width + 1.0e-5);
            double row_scale = 0.0;
            for (size_t i = 0; i < width; ++i) {
                ref[i] = ((double)x[i] - mean) * scale;
                if (fabs(ref[i]) > row_scale) row_scale = fabs(ref[i]);
            }
            for (size_t i = 0; i < width; ++i) {
                const double e1 = rel_to_scale(a[i], ref[i], row_scale);
                const double e2 = rel_to_scale(c[i], ref[i], row_scale);
                const double p = rel_to_scale(a[i], (double)c[i], row_scale);
                if (e1 > worst_cond) worst_cond = e1;
                if (e2 > worst_cond) worst_cond = e2;
                if (p > worst_cond_pair) worst_cond_pair = p;
            }
        }
        printf("   ill conditioned (mean 10, spread 1e-4): BOTH lose %.1f%% "
               "of the row scale in f32; they still track each other to "
               "%.1f%%\n", worst_cond * 100.0, worst_cond_pair * 100.0);
        CHECK(worst_cond_pair <= 0.35,
              "on an ill-conditioned row the two LayerNorms parted by %.1f%%; "
              "that is more than a summation order can explain",
              worst_cond_pair * 100.0);
    }

    /* The differences that are real, and that a caller has to know about. */
    {
        const float xs[4] = {1.0f, 2.0f, 3.0f, 4.0f};
        const float bs[4] = {0.5f, 0.5f, 0.5f, 0.5f};
        float only_flow[4];
        /* flow_head's accepts weight == NULL; kernels.c's dereferences it
         * unconditionally, including inside the NEON body.  Documented here
         * so the next caller does not find out the hard way. */
        mynah_flow_layernorm_f32(xs, NULL, bs, only_flow, 4u, 1.0e-6f);
        CHECK(fabsf(only_flow[0] - (-1.34164f + 0.5f)) < 1.0e-4f,
              "flow LayerNorm with weight == NULL = %.9g",
              (double)only_flow[0]);
        /* kernels.c's takes rows; flow_head's takes exactly one row.  Assert
         * that N rows through the first equal N calls to the second. */
        float many_in[3 * 8], many_out[3 * 8], one_out[8];
        const float ws[8] = {1.0f, 2.0f, 0.5f, 1.5f, 1.0f, 0.25f, 3.0f, 1.0f};
        const float bb[8] = {0.0f, 1.0f, -1.0f, 0.5f, 0.0f, 0.0f, 2.0f, -0.5f};
        for (size_t i = 0; i < 24u; ++i) many_in[i] = rng_float(-2.0f, 2.0f);
        mynah_layernorm_f32(many_in, ws, bb, many_out, 3u, 8u, 1.0e-5f);
        for (size_t r = 0; r < 3u; ++r) {
            mynah_flow_layernorm_f32(many_in + r * 8u, ws, bb, one_out, 8u,
                                     1.0e-5f);
            for (size_t i = 0; i < 8u; ++i)
                CHECK(fabsf(many_out[r * 8u + i] - one_out[i]) < 1.0e-5f,
                      "row %zu element %zu: multi-row %.9g vs single-row %.9g",
                      r, i, (double)many_out[r * 8u + i], (double)one_out[i]);
        }
    }
}

/* ======================================================================
 * 4. The two RMSNorms, which are NOT the same function, and must not be
 *    allowed to become each other by a careless substitution.
 * ====================================================================== */
static void rmsnorm_two_ways(void) {
    printf("\n-- RMSNorm: mean-square (kernels.c) vs variance, unbiased "
           "(flow_head.c) --\n");
    enum { W = 64 };
    float x[W], g[W], a[W], c[W];
    double worst_ms = 0.0, worst_var = 0.0, closest_gap = 1.0e30;
    rng_seed(0x2334);
    for (int trial = 0; trial < 200; ++trial) {
        const size_t width = (size_t)(2 + (trial % 61));
        /* A non-zero mean is what separates the two definitions; with a mean
         * of zero and a large N they converge, and a test that only ever used
         * zero-mean data would not be able to tell them apart. */
        const float offset = (trial % 3 == 0) ? 0.0f : 4.0f;
        for (size_t i = 0; i < width; ++i) {
            x[i] = rng_float(-2.0f, 2.0f) + offset;
            g[i] = rng_float(0.3f, 1.8f);
        }
        mynah_rmsnorm_f32(x, g, a, width, 1.0e-5f);
        mynah_flow_var_rmsnorm_f32(x, g, c, width, 1.0e-5f);

        double sumsq = 0.0, mean = 0.0;
        for (size_t i = 0; i < width; ++i) {
            sumsq += (double)x[i] * (double)x[i];
            mean += (double)x[i];
        }
        mean /= (double)width;
        double var = 0.0;
        for (size_t i = 0; i < width; ++i) {
            const double d = (double)x[i] - mean;
            var += d * d;
        }
        const double ms_scale = 1.0 / sqrt(sumsq / (double)width + 1.0e-5);
        const double var_scale =
            (width >= 2u) ? 1.0 / sqrt(1.0e-5 + var / (double)(width - 1u))
                          : 0.0;
        for (size_t i = 0; i < width; ++i) {
            const double ms_ref = (double)x[i] * (double)g[i] * ms_scale;
            const double v_ref = (double)x[i] * ((double)g[i] * var_scale);
            const double e1 = mynah_vecmath_ulp(a[i], ms_ref);
            if (e1 > worst_ms) worst_ms = e1;
            if (width >= 2u) {
                const double e2 = mynah_vecmath_ulp(c[i], v_ref);
                if (e2 > worst_var) worst_var = e2;
            }
        }
        if (offset != 0.0f && width >= 8u) {
            double gap = 0.0;
            for (size_t i = 0; i < width; ++i)
                gap += fabs((double)a[i] - (double)c[i]) /
                       (fabs((double)a[i]) + 1.0e-9);
            gap /= (double)width;
            if (gap < closest_gap) closest_gap = gap;
        }
    }
    printf("   mean-square form within %.1f ulp of its f64 reference\n",
           worst_ms);
    printf("   variance form   within %.1f ulp of its f64 reference\n",
           worst_var);
    printf("   on offset data the two differ by at least %.1f%% relative\n",
           closest_gap * 100.0);
    CHECK(worst_ms <= 32.0, "mean-square rmsnorm %.1f ulp", worst_ms);
    CHECK(worst_var <= 32.0, "variance rmsnorm %.1f ulp", worst_var);
    CHECK(closest_gap > 0.05,
          "the two RMSNorms came within %.3f%% of each other on offset data; "
          "this test could no longer catch one being substituted for the "
          "other", closest_gap * 100.0);
    /* n < 2 is undefined for an unbiased variance; flow_head's contract is
     * that it leaves the output alone rather than dividing by zero. */
    {
        const float one_in[1] = {3.0f};
        float one_out[1] = {-7.0f};
        mynah_flow_var_rmsnorm_f32(one_in, NULL, one_out, 1u, 1.0e-5f);
        CHECK(one_out[0] == -7.0f,
              "variance rmsnorm at n=1 wrote %.9g; an unbiased variance has "
              "no value at n=1 and the kernel must decline", (double)one_out[0]);
    }
}

/* ======================================================================
 * 5. GELU-tanh: three spellings in this tree, and what they owe each other
 * ====================================================================== */
static void gelu_three_ways(void) {
    printf("\n-- GELU-tanh: elementwise, array, and the Pade mynah_gelu_f32 --\n");
    enum { N = 1024 };
    float xs[N], arrayform[N], pade[N];
    double worst_pair = 0.0, worst_ref = 0.0, worst_pade = 0.0;
    float pade_at = 0.0f, pair_at = 0.0f;

    rng_seed(0x6E10u);
    for (int trial = 0; trial < 120; ++trial) {
        const float span = (trial % 3 == 0) ? 12.0f : 3.0f;
        for (int i = 0; i < N; ++i) xs[i] = rng_float(-span, span);
        memcpy(arrayform, xs, sizeof xs);
        memcpy(pade, xs, sizeof xs);
        mynah_gelu_tanh_array(arrayform, N, NULL);
        mynah_gelu_f32(pade, N);
        for (int i = 0; i < N; ++i) {
            const double x = (double)xs[i];
            /* |GELU(x)| <= |x|, so 1 + |x| is the scale the answer lives on.
             * Measuring in ULP of the result instead would report thousands
             * of ulp at x = -7, where the true GELU is 6e-6 and a correctly
             * rounded tanh still moves the answer by a large fraction of it. */
            const double scale = 1.0 + fabs(x);
            const double inner =
                0.7978845608028654 * (x + 0.044715 * x * x * x);
            const double ref = 0.5 * x * (1.0 + tanh(inner));
            const double e0 = rel_to_scale(arrayform[i],
                                           (double)mynah_gelu_tanh(xs[i]),
                                           scale);
            const double e1 = rel_to_scale(arrayform[i], ref, scale);
            const double e2 = rel_to_scale(pade[i], ref, scale);
            if (e0 > worst_pair) { worst_pair = e0; pair_at = xs[i]; }
            if (e1 > worst_ref) worst_ref = e1;
            if (e2 > worst_pade) { worst_pade = e2; pade_at = xs[i]; }
        }
    }
    printf("   elementwise vs array agree to %.2e of (1+|x|) (worst at %+.4f)\n",
           worst_pair, (double)pair_at);
    printf("   array form within %.2e of (1+|x|) of the f64 GELU definition\n",
           worst_ref);
    printf("   mynah_gelu_f32 (Pade [5/5], tanh argument clamped to +-6) "
           "within %.2e (worst at %+.4f)\n", worst_pade, (double)pade_at);
    CHECK(worst_pair <= 1.0e-6,
          "the elementwise and array GELU have drifted apart by %.2e; "
          "engine_magpie.c calls both", worst_pair);
    CHECK(worst_ref <= 1.0e-6, "array GELU %.2e from the definition",
          worst_ref);
    /* Not a failure -- a recorded fact, and the reason mynah_gelu_f32 must
     * never be swapped for the array form on the strength of "they are both
     * the tanh GELU".  The clamp makes it a different function past
     * |x| ~ 4.2, by roughly the factor printed above. */
    if (mynah_gelu_vector_enabled()) {
        CHECK(worst_pade > worst_ref,
              "the Pade GELU is no longer measurably different from the exact "
              "one; this check exists to keep that difference visible");
        printf("   note: the clamp makes the Pade GELU a different function "
               "past |x| ~ 4.2, by %.0fx here -- by design, but not "
               "interchangeable\n",
               worst_ref > 0.0 ? worst_pade / worst_ref : 0.0);
    } else {
        /* On a scalar build mynah_gelu_f32 falls through to its libm loop and
         * there is no clamp, so the two ARE the same function here.  That is
         * the correct behaviour, not a lost difference -- asserting otherwise
         * would fail a SIMD=scalar build for being right. */
        printf("   note: no vector Pade GELU is compiled here, so "
               "mynah_gelu_f32 is the libm form and the clamp is absent "
               "(%.2e vs %.2e)\n", worst_pade, worst_ref);
    }

    /* Non-finite handling, which the array form has to preserve. */
    {
        float edge[8] = {-INFINITY, INFINITY, NAN, -0.0f, 0.0f,
                         1.0e-40f, 1.0e30f, -1.0e30f};
        float want[8];
        for (int i = 0; i < 8; ++i) want[i] = mynah_gelu_tanh(edge[i]);
        mynah_gelu_tanh_array(edge, 8u, NULL);
        for (int i = 0; i < 8; ++i) {
            if (want[i] != want[i]) {
                CHECK(edge[i] != edge[i], "array GELU lost a NaN at %d", i);
            } else {
                CHECK(same_bits(edge[i], want[i]),
                      "array GELU edge %d: %.9g vs elementwise %.9g", i,
                      (double)edge[i], (double)want[i]);
            }
        }
    }
}

/* ======================================================================
 * 6. The SEANet Snake -- what codec_nanocodec.c actually calls
 * ====================================================================== */
static void snake_rows(void) {
    printf("\n-- SEANet Snake row --\n");
    enum { N = 1021 };  /* prime: guarantees a tail on 4-wide and 8-wide */
    float vec[N], sca[N], src[N];
    double truth[N], scale[N];
    static const float alphas[] = {1.0e-6f, 1.0e-3f, 0.05f, 0.5f, 1.0f,
                                   3.7f, 40.0f, 900.0f};
    double worst = 0.0, worst_pair = 0.0;
    rng_seed(0x54AE);
    for (size_t a = 0; a < sizeof(alphas) / sizeof(alphas[0]); ++a) {
        for (int i = 0; i < N; ++i) src[i] = rng_float(-9.0f, 9.0f);
        memcpy(vec, src, sizeof src);
        memcpy(sca, src, sizeof src);
        for (int i = 0; i < N; ++i) {
            const double s = sin((double)alphas[a] * (double)src[i]);
            truth[i] = (double)src[i] + s * s / ((double)alphas[a] + 1.0e-9);
            /* v and s^2/alpha can cancel; measure against the scale of the
             * terms, not the size of the answer.  See rel_to_scale. */
            scale[i] = fabs((double)src[i]) +
                       s * s / ((double)alphas[a] + 1.0e-9);
        }
        mynah_snake_row_f32(vec, N, alphas[a]);
        mynah_snake_row_f32_scalar(sca, N, alphas[a]);
        for (int i = 0; i < N; ++i) {
            const double uv = rel_to_scale(vec[i], truth[i], scale[i]);
            const double us = rel_to_scale(sca[i], truth[i], scale[i]);
            const double up = rel_to_scale(vec[i], (double)sca[i], scale[i]);
            if (uv > worst) worst = uv;
            if (us > worst) worst = us;
            if (up > worst_pair) worst_pair = up;
        }
    }
    printf("   8 alphas from 1e-6 to 900, %d samples each over [-9, 9]\n", N);
    printf("     within %.2e of the scale of its own terms\n", worst);
    printf("     %s and scalar agree to %.2e\n", mynah_vecmath_isa(),
           worst_pair);
    CHECK(worst <= 4.0e-7, "snake %.2e from the reference", worst);
    CHECK(worst_pair <= 4.0e-7, "snake vector/scalar disagree by %.2e",
          worst_pair);
    /* Alpha = 0 is the degenerate case the 1e-9 in the denominator exists
     * for: it must be finite, and it must be the identity. */
    {
        float row[8] = {1.0f, -1.0f, 0.5f, 0.0f, -0.0f, 3.0f, -2.5f, 7.0f};
        float expect[8];
        memcpy(expect, row, sizeof expect);
        mynah_snake_row_f32(row, 8u, 0.0f);
        for (int i = 0; i < 8; ++i)
            CHECK(row[i] == expect[i],
                  "snake at alpha=0 must be the identity; row[%d] = %.9g",
                  i, (double)row[i]);
    }
    /* Every length from 0 to 40, so the tail loop is exercised at each
     * offset against the vector body. */
    for (size_t len = 0; len <= 40u; ++len) {
        float v[40], s[40];
        for (size_t i = 0; i < len; ++i) v[i] = s[i] = rng_float(-5.0f, 5.0f);
        mynah_snake_row_f32(v, len, 0.61f);
        mynah_snake_row_f32_scalar(s, len, 0.61f);
        for (size_t i = 0; i < len; ++i)
            CHECK(fabs((double)v[i] - (double)s[i]) <=
                      4.0e-7 * (fabs((double)s[i]) + 1.0),
                  "snake length %zu differs at %zu: %.9g vs %.9g", len, i,
                  (double)v[i], (double)s[i]);
    }
}

/* ======================================================================
 * 7. The tail sweep: every vectorised kernel in kernels.h at every length
 *    from 0 to 40, against an independent f64 reference.
 * ====================================================================== */
static void tail_sweep(void) {
    printf("\n-- every kernel at every length 0..40 against an f64 reference "
           "--\n");
    enum { M = 40 };
    float a[M], b[M], out[M], w[M], bias[M];
    double worst = 0.0, worst_rel = 0.0;
    /* TWO METRICS, TWO NAMES.  These shared one `worst_name`, so whichever
     * kernel updated last named BOTH lines of the report: the run that found
     * this printed "worst reducing kernel: 1.05e-07 (axpy)" while axpy is an
     * elementwise kernel, and the elementwise failure named no kernel at all.
     * A report that can attribute a number to the wrong kernel is worse than
     * one that says nothing, because it is acted on. */
    const char *worst_name = "none";
    const char *worst_elem = "none";

    rng_seed(0x7A11);
    for (size_t n = 0; n <= M; ++n) {
        for (size_t i = 0; i < n; ++i) {
            a[i] = rng_float(-3.0f, 3.0f);
            b[i] = rng_float(-3.0f, 3.0f);
            w[i] = rng_float(0.2f, 2.0f);
            bias[i] = rng_float(-1.0f, 1.0f);
        }
        /* dot.  A dot product can cancel to nothing, so the scale is the sum
         * of the magnitudes of its terms -- the standard conditioning of a
         * sum -- and not the size of the answer. */
        {
            double ref = 0.0, mag = 0.0;
            for (size_t i = 0; i < n; ++i) {
                ref += (double)a[i] * (double)b[i];
                mag += fabs((double)a[i] * (double)b[i]);
            }
            const double e = rel_to_scale(mynah_dot_f32(a, b, n), ref, mag);
            if (e > worst_rel && n > 0) { worst_rel = e; worst_name = "dot"; }
        }
        /* residual add */
        if (n > 0) {
            float acc[M];
            memcpy(acc, a, sizeof(float) * n);
            mynah_residual_add_f32(acc, b, n);
            for (size_t i = 0; i < n; ++i) {
                /* CONDITIONED, not ULP -- the same argument the dot product
                 * above is measured by, and for the same reason: `a + b` is a
                 * two-term sum and it can cancel to nothing, so the scale it
                 * lives on is |a| + |b| and not the size of what came out. */
                const double e = rel_to_scale(acc[i],
                                              (double)a[i] + (double)b[i],
                                              fabs((double)a[i]) +
                                                  fabs((double)b[i]));
                if (e > worst_rel) { worst_rel = e; worst_name = "residual_add"; }
            }
        }
        /* axpy
         *
         * MEASURED IN ULP UNTIL NOW, AND THAT IS WHAT BROKE.  `a + 0.375*b`
         * cancels exactly like the dot above, and a cancelled result has so
         * few significant bits left that one rounding of an INPUT is a
         * thousand ULP of the OUTPUT.  Which rounding you get is an ISA
         * property: with FMA the kernel rounds once and lands near the double
         * reference, without it the multiply and the add round separately.
         *
         * So this passed on aarch64, where FMA always exists, and on an x86
         * build with -mfma -- and failed only on a baseline x86 build, at
         * 1024.0 ULP, on a Linux sanitizer job. The absolute error was never
         * more than about one ULP of the inputs. */
        if (n > 0) {
            float acc[M];
            memcpy(acc, a, sizeof(float) * n);
            mynah_axpy_f32(acc, b, 0.375f, n);
            for (size_t i = 0; i < n; ++i) {
                const double e = rel_to_scale(
                    acc[i], (double)a[i] + 0.375 * (double)b[i],
                    fabs((double)a[i]) + fabs(0.375 * (double)b[i]));
                if (e > worst_rel) { worst_rel = e; worst_name = "axpy"; }
            }
        }
        /* rmsnorm (mean square) */
        if (n > 0) {
            double sumsq = 0.0;
            for (size_t i = 0; i < n; ++i)
                sumsq += (double)a[i] * (double)a[i];
            const double scale = 1.0 / sqrt(sumsq / (double)n + 1.0e-6);
            mynah_rmsnorm_f32(a, w, out, n, 1.0e-6f);
            for (size_t i = 0; i < n; ++i) {
                const double e = mynah_vecmath_ulp(
                    out[i], (double)a[i] * scale * (double)w[i]);
                if (e > worst) { worst = e; worst_elem = "rmsnorm"; }
            }
        }
        /* layernorm, both forms, measured against the row scale -- see
         * rel_to_scale: an element that sits on the mean has an output of
         * zero and no ULP bound applies to it. */
        if (n > 0) {
            double mean = 0.0;
            for (size_t i = 0; i < n; ++i) mean += (double)a[i];
            mean /= (double)n;
            double var = 0.0;
            for (size_t i = 0; i < n; ++i) {
                const double d = (double)a[i] - mean;
                var += d * d;
            }
            const double scale = 1.0 / sqrt(var / (double)n + 1.0e-5);
            double row_scale = 0.0;
            double lref[M];
            for (size_t i = 0; i < n; ++i) {
                lref[i] = ((double)a[i] - mean) * scale * (double)w[i] +
                          (double)bias[i];
                if (fabs(lref[i]) > row_scale) row_scale = fabs(lref[i]);
            }
            float one_row[M];
            mynah_layernorm_f32(a, w, bias, out, 1u, n, 1.0e-5f);
            mynah_flow_layernorm_f32(a, w, bias, one_row, n, 1.0e-5f);
            for (size_t i = 0; i < n; ++i) {
                CHECK(rel_to_scale(out[i], lref[i], row_scale) <= 1.0e-5,
                      "layernorm n=%zu i=%zu off the f64 reference", n, i);
                CHECK(rel_to_scale(one_row[i], (double)out[i], row_scale)
                          <= 1.0e-5,
                      "the two layernorms differ at n=%zu i=%zu", n, i);
            }
        }
        /* matvec and matvec_bias, rows x n */
        if (n > 0) {
            float matrix[3 * M];
            float mv[3];
            for (size_t i = 0; i < 3u * n; ++i)
                matrix[i] = rng_float(-2.0f, 2.0f);
            mynah_matvec_f32(matrix, a, mv, 3u, n);
            for (size_t r = 0; r < 3u; ++r) {
                double ref = 0.0, mag = 0.0;
                for (size_t i = 0; i < n; ++i) {
                    ref += (double)matrix[r * n + i] * (double)a[i];
                    mag += fabs((double)matrix[r * n + i] * (double)a[i]);
                }
                const double e = rel_to_scale(mv[r], ref, mag);
                if (e > worst_rel) { worst_rel = e; worst_name = "matvec"; }
            }
            mynah_matvec_bias_f32(matrix, a, bias, mv, 3u, n);
            for (size_t r = 0; r < 3u; ++r) {
                double ref = (double)bias[r], mag = fabs((double)bias[r]);
                for (size_t i = 0; i < n; ++i) {
                    ref += (double)matrix[r * n + i] * (double)a[i];
                    mag += fabs((double)matrix[r * n + i] * (double)a[i]);
                }
                const double e = rel_to_scale(mv[r], ref, mag);
                if (e > worst_rel) { worst_rel = e; worst_name = "matvec_bias"; }
            }
        }
        /* softmax: must sum to one and preserve the argmax */
        if (n > 0) {
            float p[M];
            if (mynah_softmax_f32(a, p, n) == 0) {
                double total = 0.0;
                for (size_t i = 0; i < n; ++i) total += (double)p[i];
                CHECK(fabs(total - 1.0) < 1.0e-5,
                      "softmax at n=%zu sums to %.9g", n, total);
                size_t hi = 0;
                for (size_t i = 1; i < n; ++i) if (a[i] > a[hi]) hi = i;
                CHECK(mynah_argmax_f32(p, n) == hi,
                      "softmax at n=%zu moved the argmax", n);
            }
        }
        /* the tanh/sin arrays at the same lengths */
        if (n > 0) {
            float tv[M], ts[M];
            mynah_tanh_f32(a, tv, n);
            mynah_tanh_f32_scalar(a, ts, n);
            for (size_t i = 0; i < n; ++i)
                CHECK(mynah_vecmath_ulp(tv[i], (double)ts[i]) <= 2.0,
                      "tanh length %zu differs at %zu", n, i);
            mynah_sin_f32(a, tv, n);
            mynah_sin_f32_scalar(a, ts, n);
            for (size_t i = 0; i < n; ++i)
                CHECK(fabsf(tv[i] - ts[i]) <= 2.0e-7f,
                      "sin length %zu differs at %zu", n, i);
        }
    }
    printf("   worst elementwise kernel: %.1f ulp (%s)\n", worst, worst_elem);
    printf("   worst reducing kernel: %.2e relative to the sum of the "
           "magnitudes of its terms (%s)\n", worst_rel, worst_name);
    CHECK(worst <= 8.0, "the elementwise kernel %s is %.1f ulp out", worst_elem,
          worst);
    CHECK(worst_rel <= 2.0e-6,
          "a reducing kernel is %.2e out relative to its own conditioning "
          "(%s)", worst_rel, worst_name);
    /* n = 0 must be a no-op everywhere, not a one-element write. */
    {
        float guard[2] = {12345.0f, -12345.0f};
        mynah_residual_add_f32(guard, guard, 0);
        mynah_axpy_f32(guard, guard, 2.0f, 0);
        mynah_tanh_f32(guard, guard, 0);
        mynah_sin_f32(guard, guard, 0);
        mynah_snake_row_f32(guard, 0, 1.0f);
        mynah_layernorm_f32(guard, guard, NULL, guard, 0u, 2u, 1.0e-5f);
        CHECK(guard[0] == 12345.0f && guard[1] == -12345.0f,
              "a kernel wrote at n = 0");
        CHECK(mynah_softmax_f32(guard, guard, 0) != 0,
              "softmax accepted n = 0");
        CHECK(mynah_argmax_f32(guard, 0) == (size_t)-1,
              "argmax at n = 0 did not report empty");
    }
}

/* ======================================================================
 * 8. Causal unfold -- the primitive under every conv1d in this tree
 * ====================================================================== */
static void causal_unfold(void) {
    printf("\n-- causal unfold (the shared conv1d primitive) --\n");
    enum { C = 5, L = 9, K = 4 };
    float input[C * L];
    float col[L * C * K];
    for (size_t i = 0; i < C * L; ++i) input[i] = (float)(i + 1);
    memset(col, 0x7f, sizeof col);
    mynah_unfold_causal(input, col, L, C, K);
    /* The contract, restated from conv1d.h: col[t][i*kernel + k] is channel i
     * at time t - (kernel-1) + k, and zero before the start.  Causality is
     * the claim that nothing at t reads past t -- which is exactly what
     * k == kernel-1 mapping to t asserts.
     *
     * ONE THING conv1d.h DOES NOT SAY, and this test had to find out by
     * failing: the INPUT is time major, [length][channels], while the OUTPUT
     * column is channel major within each row, [channels][kernel].  The
     * function transposes.  Anyone writing a second engine against this
     * primitive needs that sentence, so it is recorded here. */
    for (size_t t = 0; t < L; ++t) {
        for (size_t i = 0; i < C; ++i) {
            for (size_t k = 0; k < K; ++k) {
                const long src = (long)t - (long)(K - 1) + (long)k;
                const float want =
                    (src < 0) ? 0.0f : input[(size_t)src * C + i];
                const float got = col[t * (C * K) + i * K + k];
                CHECK(got == want,
                      "unfold[t=%zu][ch=%zu][k=%zu] = %.9g want %.9g", t, i, k,
                      (double)got, (double)want);
            }
        }
    }
    CHECK(col[0 * (C * K) + 0 * K + (K - 1)] == input[0],
          "the last tap at t=0 must be the sample at t=0, or the conv is not "
          "causal");
    /* The taps before the start at t=0 must be zero and not stale memory --
     * col was filled with 0x7f bytes above, so a missing zero-fill shows. */
    for (size_t k = 0; k + 1 < K; ++k)
        for (size_t i = 0; i < C; ++i)
            CHECK(col[0 * (C * K) + i * K + k] == 0.0f,
                  "unfold left t=0 tap %zu of channel %zu unzeroed (%.9g)", k,
                  i, (double)col[0 * (C * K) + i * K + k]);
    printf("   %d channels x %d positions x kernel %d, time-major in, "
           "channel-major out, zero padded before the start\n", C, L, K);
}

int main(void) {
    char error[512];
    printf("kernels: model-free self tests (PLAN.md E3-6, E4-16d)\n");
    printf("  vector ISA for the transcendentals: %s\n", mynah_vecmath_isa());

    if (mynah_kernels_self_test(error, sizeof error) != 0) {
        printf("  FAIL mynah_kernels_self_test: %s\n", error);
        ++failures;
    }
    ++checks;
    if (mynah_vecmath_self_test(error, sizeof error) != 0) {
        printf("  FAIL mynah_vecmath_self_test: %s\n", error);
        ++failures;
    }
    ++checks;
    if (mynah_gelu_self_test(error, sizeof error) != 0) {
        printf("  FAIL mynah_gelu_self_test: %s\n", error);
        ++failures;
    }
    ++checks;
    if (mynah_flow_head_self_test(error, sizeof error) != 0) {
        printf("  FAIL mynah_flow_head_self_test: %s\n", error);
        ++failures;
    }
    ++checks;

    ulp_table();
    transcendental_edges();
    layernorm_two_ways();
    rmsnorm_two_ways();
    gelu_three_ways();
    snake_rows();
    tail_sweep();
    causal_unfold();

    printf("\nkernels: %d checks, %d failures\n", checks, failures);
    if (failures != 0) {
        printf("kernels: FAIL\n");
        return 1;
    }
    printf("kernels: PASS\n");
    return 0;
}
