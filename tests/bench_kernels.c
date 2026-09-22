/* Model-free kernel micro-benchmark.
 *
 * WHY THIS EXISTS. Every performance number in this repository needs a model
 * pack, which means the kernels landed for x86 could be proven correct on a
 * rented box and still not be measured on one -- the pack is gitignored and a
 * 600 MB download is not something a five-minute check does. The idea is
 * borrowed from ../qwen-tts's bench_simd.c: time the hot loops at the model's
 * SHAPES without the model's WEIGHTS.
 *
 * WHAT IT IS NOT. It is not an RTF, it is not a serving number, and a kernel
 * that is 2x faster here is not a synthesis that is 2x faster -- the AR step is
 * bound by weight traffic and by the pool, and E12 measured the weight pass at
 * 5% of the step at B=8. Read it as "which kernel is executing and how fast is
 * THAT loop", which is the question the dispatch work raised and could not
 * answer.
 *
 * ATTRIBUTION IS PART OF THE OUTPUT. It prints which kernel resolved beside
 * every number, because this binary picks at runtime and a bare millisecond
 * from it would be unattributable -- the repo's own rule that a benchmark is
 * invalid until dispatch is proven.
 */
#include "kernels.h"
#include "qmat.h"
#include "dispatch.h"
#include "sgemm.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* PocketTTS backbone: hidden 1024, ffn 4096. The two matvec shapes below are
 * the FFN's, which is where the backbone's weight bytes are. */
enum { HID = 1024, FFN = 4096 };

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1.0e6;
}

/* Median of a small odd sample, not a mean: one descheduled run should not
 * move the answer, and on a shared box it will happen. */
static int cmp_d(const void *a, const void *b) {
    const double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}
static double median(double *v, int n) {
    qsort(v, (size_t)n, sizeof *v, cmp_d);
    return v[n / 2];
}

#define REPS 9

static void report(const char *name, double ms, double bytes, const char *kern) {
    printf("  %-26s %8.3f ms  %7.1f GB/s   %s\n", name, ms,
           bytes / (ms * 1.0e6), kern);
}

int main(void) {
    static float wf[FFN * HID];
    static float x[FFN], y[FFN], bias[FFN];
    static uint16_t wbf[FFN * HID];
    static int8_t wq[FFN * HID];
    static float wscale[FFN];
    static int32_t rowsum[FFN];
    static int32_t iout[FFN];
    double t[REPS];

    for (size_t i = 0; i < (size_t)FFN * HID; ++i)
        wf[i] = 0.02f * (float)((i * 37u) % 101u) - 1.0f;
    for (size_t i = 0; i < FFN; ++i) {
        x[i] = 0.01f * (float)((i * 17u) % 211u) - 1.0f;
        bias[i] = 0.001f * (float)i;
    }
    for (size_t i = 0; i < (size_t)FFN * HID; ++i) {
        uint32_t b;
        memcpy(&b, &wf[i], 4);
        const uint32_t lsb = (b >> 16) & 1u;
        wbf[i] = (uint16_t)((b + 0x7fffu + lsb) >> 16);
    }
    if (mynah_qmat_pack_q8(wf, FFN, HID, wq, wscale, rowsum) != 0) {
        fprintf(stderr, "pack_q8 refused %dx%d\n", FFN, HID);
        return 1;
    }
    void *xq = malloc(mynah_qmat_act_bytes(HID));
    if (xq == NULL) return 1;
    (void)mynah_qmat_act_quantize(xq, x, HID);
    const void *xqs[1] = { xq };

    const char *k_f32 = mynah_kernels_x86_avx2() ? "avx2+fma" : "scalar-or-neon";
    const char *k_i8 = mynah_qmat_int8_kernel(NULL);
    const char *why = NULL;
    (void)mynah_qmat_bf16_enabled(&why);
    const char *k_bf = strstr(why ? why : "", "VDPBF16PS") ? "vdpbf16ps"
                     : strstr(why ? why : "", "BFMMLA")    ? "bfmmla"
                     : strstr(why ? why : "", "BFDOT")     ? "bfdot"
                     : strstr(why ? why : "", "avx2")      ? "avx2-widen"
                                                           : "scalar";

    printf("kernel micro-bench -- shapes only, no model. %s, %s\n",
           mynah_dispatch_isa_class(), mynah_sgemm_isa_name());
    printf("  f32 rows=%d cols=%d, %d reps, median\n\n", FFN, HID, REPS);

    for (int r = 0; r < REPS; ++r) {
        const double t0 = now_ms();
        mynah_matvec_bias_f32(wf, x, bias, y, FFN, HID);
        t[r] = now_ms() - t0;
    }
    report("f32 matvec_bias", median(t, REPS),
           (double)FFN * HID * 4.0, k_f32);

    for (int r = 0; r < REPS; ++r) {
        const double t0 = now_ms();
        for (int i = 0; i < 64; ++i) mynah_layernorm_f32(x, x, bias, y, 1, HID, 1e-5f);
        t[r] = now_ms() - t0;
    }
    report("f32 layernorm x64", median(t, REPS), (double)HID * 4.0 * 64.0, k_f32);

    for (int r = 0; r < REPS; ++r) {
        const double t0 = now_ms();
        for (int i = 0; i < 64; ++i) mynah_rmsnorm_f32(x, bias, y, HID, 1e-5f);
        t[r] = now_ms() - t0;
    }
    report("f32 rmsnorm x64", median(t, REPS), (double)HID * 4.0 * 64.0, k_f32);

    for (int r = 0; r < REPS; ++r) {
        const double t0 = now_ms();
        for (int i = 0; i < 256; ++i) mynah_axpy_f32(y, x, 0.5f, HID);
        t[r] = now_ms() - t0;
    }
    report("f32 axpy x256", median(t, REPS), (double)HID * 4.0 * 256.0, k_f32);

    for (int r = 0; r < REPS; ++r) {
        const double t0 = now_ms();
        mynah_qmat_dots_i8(wq, FFN, HID, rowsum, xqs, 1, iout, FFN);
        t[r] = now_ms() - t0;
    }
    report("int8 dots (B=1)", median(t, REPS), (double)FFN * HID, k_i8);

    for (int r = 0; r < REPS; ++r) {
        const double t0 = now_ms();
        mynah_qmat_matvec_bf16(y, x, wbf, bias, FFN, HID);
        t[r] = now_ms() - t0;
    }
    report("bf16 matvec", median(t, REPS), (double)FFN * HID * 2.0, k_bf);

    for (int r = 0; r < REPS; ++r) {
        const double t0 = now_ms();
        mynah_sgemm_f32(0, 0, 16, HID, HID, 1.0f, wf, HID, wf, HID, 0.0f, y, HID);
        t[r] = now_ms() - t0;
    }
    report("sgemm 16x1024x1024", median(t, REPS),
           (double)16 * HID * HID * 4.0, mynah_sgemm_isa_name());

    free(xq);
    return 0;
}
