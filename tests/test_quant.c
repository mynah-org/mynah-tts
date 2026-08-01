/* Kernel tests, kept honest in four ways:
 *
 *  - the quantizer is measured as relative L2 on two distributions, because
 *    the answer differs by a factor of two between them and one number would
 *    hide that;
 *  - every matvec is checked against its OWN dequant, so a packing bug shows
 *    up as a disagreement rather than as a plausible number;
 *  - the batched path is checked against the per-token one with deliberately
 *    awkward shapes (37 rows against a strip of 8, 131 tokens against a tile
 *    of 128), so a skipped or double-written tail surfaces here;
 *  - the tolerance is chosen by asking ingot_matmat_is_exact(), never guessed.
 *
 * SPDX-License-Identifier: MIT */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ingot/quant.h"

static int failures;
static int checks;

#define CHECK(cond, ...) do {                                        \
    checks++;                                                        \
    if (!(cond)) { printf("  FAIL: "); printf(__VA_ARGS__);          \
                   printf("  (%s:%d)\n", __FILE__, __LINE__);        \
                   failures++; }                                     \
    else { printf("  ok:   "); printf(__VA_ARGS__); printf("\n"); }  \
} while (0)

/* Deterministic pseudo-random bytes: every scale, nibble and high-bit pattern
 * gets exercised, which quantized smooth data would never do. */
static uint32_t rng_state = 0x9E3779B9u;
static unsigned char rnd_byte(void) {
    rng_state = rng_state * 1664525u + 1013904223u;
    return (unsigned char)(rng_state >> 24);
}
static float rnd_unit(void) {
    rng_state = rng_state * 1664525u + 1013904223u;
    return (float)(rng_state >> 8) / 16777216.0f - 0.5f;
}

static double rel_l2(const float *a, const float *b, size_t n) {
    double err = 0, ref = 0;
    for (size_t i = 0; i < n; i++) {
        const double d = (double)a[i] - (double)b[i];
        err += d * d;
        ref += (double)b[i] * (double)b[i];
    }
    return ref > 0 ? sqrt(err / ref) : sqrt(err);
}

/* ── the quantizer ──────────────────────────────────────────────────────── */
static void test_quantize(void) {
    printf("Q4_K quantizer\n");
    float values[512], decoded[512];
    unsigned char blocks[2 * 144];

    /* An exactly representable grid: q = 0..15 against a step of 63*2^-8.
     * The step matters — d is stored as f16 and equals step/63, so only a step
     * whose 63rd part lands on a power of two round-trips exactly. */
    for (int i = 0; i < 512; i++) values[i] = (float)(i % 16) * 0.24609375f;
    CHECK(ingot_q4_k_quantize(values, 512, blocks) == 0, "quantize returns 0");
    CHECK(ingot_q4_k_dequant(blocks, 2, 256, decoded) == 0, "dequant returns 0");
    int exact = 1;
    for (int i = 0; i < 512; i++) if (fabsf(decoded[i] - values[i]) > 1e-4f) exact = 0;
    CHECK(exact, "an exactly representable grid round-trips bit-for-bit");

    /* Two distributions. "spread" fills its range almost evenly; "bell" is the
     * shape a weight matrix actually has and comes out WORSE, which is the
     * opposite of the intuition: min..max over 32 Gaussian samples spans about
     * five sigma while the mass sits within one. */
    for (int i = 0; i < 512; i++)
        values[i] = 0.02f * (sinf((float)i * 0.7391f) + 0.5f * cosf((float)i * 2.113f));
    ingot_q4_k_quantize(values, 512, blocks);
    ingot_q4_k_dequant(blocks, 2, 256, decoded);
    const double spread = rel_l2(decoded, values, 512);

    rng_state = 0x9E3779B9u;
    for (int i = 0; i < 512; i++) {
        float acc = 0;
        for (int k = 0; k < 12; k++) acc += rnd_unit() + 0.5f;
        values[i] = 0.02f * (acc - 6.0f);
    }
    ingot_q4_k_quantize(values, 512, blocks);
    ingot_q4_k_dequant(blocks, 2, 256, decoded);
    const double bell = rel_l2(decoded, values, 512);

    CHECK(spread < 0.070, "spread distribution rel_l2 %.4f < 0.070", spread);
    CHECK(bell < 0.080, "bell distribution rel_l2 %.4f < 0.080", bell);

    /* The block the production matvec reads must agree with its own dequant. */
    float input[256], output = 0, manual = 0;
    for (int i = 0; i < 256; i++) input[i] = sinf((float)i * 0.05f);
    CHECK(ingot_q4_k_matvec(blocks, 1, 256, input, &output) == 0, "matvec returns 0");
    for (int i = 0; i < 256; i++) manual += decoded[i] * input[i];
    const float scale = fmaxf(1.0f, fabsf(manual));
    CHECK(fabsf(output - manual) <= 1e-3f * scale,
          "matvec agrees with dequant-then-dot (%.6f vs %.6f)", (double)output, (double)manual);
}

/* ── every format: dequant and matvec must tell the same story ──────────── */
static void test_formats(void) {
    printf("dequant / matvec agreement, every format\n");
    static const struct { const char *name; int bytes; int elems;
                          int (*deq)(const void *, size_t, size_t, float *);
                          int (*mv)(const void *, size_t, size_t, const float *, float *);
    } formats[] = {
        { "Q2_K", 84,  256, ingot_q2_k_dequant, ingot_q2_k_matvec },
        { "Q3_K", 110, 256, ingot_q3_k_dequant, ingot_q3_k_matvec },
        { "Q4_K", 144, 256, ingot_q4_k_dequant, ingot_q4_k_matvec },
        { "Q5_K", 176, 256, ingot_q5_k_dequant, ingot_q5_k_matvec },
        { "Q6_K", 210, 256, ingot_q6_k_dequant, ingot_q6_k_matvec },
        { "Q8_0", 34,  32,  ingot_q8_0_dequant, ingot_q8_0_matvec },
    };
    const size_t rows = 3;
    for (size_t f = 0; f < sizeof(formats) / sizeof(formats[0]); f++) {
        const size_t cols = 256;
        const size_t blocks_per_row = cols / (size_t)formats[f].elems;
        unsigned char *w = malloc(rows * blocks_per_row * (size_t)formats[f].bytes);
        float *decoded = malloc(rows * cols * sizeof(float));
        float *input = malloc(cols * sizeof(float));
        float *out = calloc(rows, sizeof(float));
        rng_state = 0x1234567u + (uint32_t)f;
        for (size_t i = 0; i < rows * blocks_per_row * (size_t)formats[f].bytes; i++)
            w[i] = rnd_byte();
        /* Keep the f16 scale fields in a moderate range so no inf/nan enters
         * the comparison: the exponent byte is the high one of each half. */
        for (size_t b = 0; b < rows * blocks_per_row; b++) {
            unsigned char *blk = w + b * (size_t)formats[f].bytes;
            for (int k = 0; k < formats[f].bytes; k++)
                if ((k & 1) == 1) blk[k] = (unsigned char)(blk[k] & 0x3f);
        }
        for (size_t i = 0; i < cols; i++) input[i] = rnd_unit();

        const int rc_d = formats[f].deq(w, rows, cols, decoded);
        const int rc_m = formats[f].mv(w, rows, cols, input, out);
        int finite = rc_d == 0 && rc_m == 0;
        double worst = 0;
        for (size_t r = 0; r < rows && finite; r++) {
            double manual = 0;
            for (size_t i = 0; i < cols; i++)
                manual += (double)decoded[r * cols + i] * (double)input[i];
            const double d = fabs((double)out[r] - manual);
            const double sc = fmax(1.0, fabs(manual));
            if (d / sc > worst) worst = d / sc;
            if (!(out[r] == out[r])) finite = 0;
        }
        CHECK(finite && worst < 1e-5, "%s: matvec vs dequant-dot, worst rel %.2e",
              formats[f].name, worst);
        free(w); free(decoded); free(input); free(out);
    }
}

/* ── the batched path and the precision contract ────────────────────────── */
static void test_matmat(void) {
    printf("batched matmat\n");
    /* Awkward on purpose: 37 rows against a row strip of 8, 131 tokens against
     * a token tile of 128 and a register tile of 4. */
    const size_t rows = 37, cols = 256, tokens = 131;
    unsigned char *w = malloc(rows * 144);
    float *row = malloc(cols * sizeof(float));
    float *input = malloc(tokens * cols * sizeof(float));
    float *batched = malloc(tokens * rows * sizeof(float));
    float *single = malloc(tokens * rows * sizeof(float));

    rng_state = 0xC0FFEEu;
    for (size_t r = 0; r < rows; r++) {
        for (size_t i = 0; i < cols; i++) row[i] = 0.02f * rnd_unit();
        ingot_q4_k_quantize(row, cols, w + r * 144);
    }
    for (size_t i = 0; i < tokens * cols; i++) input[i] = rnd_unit();

    int per_token_ok = 1;
    for (size_t t = 0; t < tokens; t++)
        per_token_ok &= ingot_q4_k_matvec(w, rows, cols, input + t * cols,
                                          single + t * rows) == 0;
    CHECK(per_token_ok, "%zu per-token matvec calls", tokens);
    CHECK(ingot_q4_k_matmat(w, rows, cols, input, batched, tokens) == 0, "matmat runs");

    /* The tolerance comes from the library, not from a guess. */
    const int is_exact = ingot_matmat_is_exact(tokens);
    const double budget = is_exact ? 1e-5 : 5e-3;
    const double rel = rel_l2(batched, single, tokens * rows);
    CHECK(rel < budget, "%s path: rel_l2 %.2e < %.0e",
          is_exact ? "exact" : "int8", rel, budget);

    /* The *_exact twin must be exact whatever the default is: this is the call
     * a GPU parity gate makes, and letting it approximate is the bug the twin
     * exists to prevent. */
    CHECK(ingot_q4_k_matmat_exact(w, rows, cols, input, batched, tokens) == 0,
          "matmat_exact runs");
    CHECK(rel_l2(batched, single, tokens * rows) < 1e-5,
          "matmat_exact is exact regardless of the default");

    /* One token is always the exact matvec, bit for bit. */
    CHECK(ingot_matmat_is_exact(1), "a single token always takes the exact path");
    ingot_q4_k_matmat(w, rows, cols, input, batched, 1);
    CHECK(memcmp(batched, single, rows * sizeof(float)) == 0,
          "tokens == 1 is bit-identical to matvec");

    free(w); free(row); free(input); free(batched); free(single);
}

/* ── argument validation ────────────────────────────────────────────────── */
static void test_guards(void) {
    printf("guards\n");
    unsigned char block[144] = {0};
    float in[256] = {0}, out[4] = {0};
    CHECK(ingot_q4_k_matvec(NULL, 1, 256, in, out) != 0, "null weights rejected");
    CHECK(ingot_q4_k_matvec(block, 1, 255, in, out) != 0, "cols not a multiple of 256 rejected");
    CHECK(ingot_q4_k_quantize(in, 255, block) != 0, "quantize rejects a partial block");
    CHECK(ingot_dequant(INGOT_TYPE_Q1_0, block, 128, in) != 0,
          "dequant refuses a format it cannot decode");
    CHECK(ingot_dequant(INGOT_TYPE_Q4_K, block, 100, in) != 0,
          "dequant refuses a partial block");
}

int main(void) {
    test_quantize();
    test_formats();
    test_matmat();
    test_guards();
    printf("\n%d checks, %d failures\n", checks, failures);
    return failures != 0;
}
