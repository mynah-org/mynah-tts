/* src/sgemm_rt.c -- which sgemm this process runs, decided once (E14-4).
 *
 * src/sgemm.c is the kernel; this file is the choice between two builds of it.
 * The reason the choice cannot live inside that file is written at the top of
 * it: SG_LANES reaches the packed panel geometry and the public
 * mynah_sgemm_narrow_max(), so the ISA is in the data layout and not only in
 * the inner loop.  A target attribute on the micro-kernel would leave it eating
 * panels packed for the other shape.
 *
 * ON aarch64 AND EVERYWHERE ELSE THIS FILE IS EMPTY.  AdvSIMD is
 * architecturally guaranteed, sgemm.c keeps the public names, and nothing is
 * built twice.  The empty case still needs a declaration, because -Wpedantic
 * rejects an empty translation unit.
 *
 * ON x86 the public names are ours and forward to one of two variants:
 *
 *   mynah_sgemm_*_base   sgemm.c at the build's baseline ISA
 *   mynah_sgemm_*_avx2   sgemm.c with -mavx2 -mfma
 *
 * The forward is a call, not a copy: statistics counters, the self-test and the
 * family planner are all per-variant, and only the resolved variant ever runs
 * in a process, so its counters are the whole story.
 *
 * WHICH ONE, and why it is not a new probe.  mynah_kernels_x86_avx2() already
 * answers "does the f32 half of this binary run AVX2 on this host", it is
 * memoised, and MYNAH_KERNELS_X86=scalar already forces it down.  A second
 * probe with a second env would let the two halves of the f32 path disagree,
 * and then a bisect would have to name which one it meant.  One question, one
 * answer, one switch.
 */

#include "sgemm.h"

#if defined(__x86_64__) || defined(__i386__)

#include "kernels.h"    /* mynah_kernels_x86_avx2 */

/* Implemented by src/sgemm.c compiled with MYNAH_SGEMM_VARIANT=base. */
int mynah_sgemm_f32_base(int trans_a, int trans_b, size_t m, size_t n, size_t k, float alpha, const float *a, size_t lda, const float *b, size_t ldb, float beta, float *c, size_t ldc);
int mynah_sgemm_f32_conv_taps_base(size_t m, size_t n, size_t k, size_t taps, const float *weight, float *gather, const float *b, size_t ldb, size_t b_tap_stride, float beta, float *c, size_t ldc);
void mynah_sgemm_f32_reference_base(int trans_a, int trans_b, size_t m, size_t n, size_t k, float alpha, const float *a, size_t lda, const float *b, size_t ldb, float beta, float *c, size_t ldc);
mynah_sgemm_family mynah_sgemm_family_for_base(int trans_a, int trans_b, size_t m, size_t n, size_t k, const char **why);
const char *mynah_sgemm_family_name_base(mynah_sgemm_family family);
const char *mynah_sgemm_isa_name_base(void);
size_t mynah_sgemm_narrow_max_base(void);
void mynah_sgemm_stats_get_base(mynah_sgemm_stats *out);
void mynah_sgemm_stats_reset_base(void);
int mynah_sgemm_self_test_base(char *error, size_t error_capacity);
int mynah_sgemm_f32_forced_base(mynah_sgemm_family want, mynah_sgemm_family *ran, int trans_a, int trans_b, size_t m, size_t n, size_t k, float alpha, const float *a, size_t lda, const float *b, size_t ldb, float beta, float *c, size_t ldc);
void mynah_sgemm_dispatch_probes_base(void);

/* Implemented by src/sgemm.c compiled with MYNAH_SGEMM_VARIANT=avx2. */
int mynah_sgemm_f32_avx2(int trans_a, int trans_b, size_t m, size_t n, size_t k, float alpha, const float *a, size_t lda, const float *b, size_t ldb, float beta, float *c, size_t ldc);
int mynah_sgemm_f32_conv_taps_avx2(size_t m, size_t n, size_t k, size_t taps, const float *weight, float *gather, const float *b, size_t ldb, size_t b_tap_stride, float beta, float *c, size_t ldc);
void mynah_sgemm_f32_reference_avx2(int trans_a, int trans_b, size_t m, size_t n, size_t k, float alpha, const float *a, size_t lda, const float *b, size_t ldb, float beta, float *c, size_t ldc);
mynah_sgemm_family mynah_sgemm_family_for_avx2(int trans_a, int trans_b, size_t m, size_t n, size_t k, const char **why);
const char *mynah_sgemm_family_name_avx2(mynah_sgemm_family family);
const char *mynah_sgemm_isa_name_avx2(void);
size_t mynah_sgemm_narrow_max_avx2(void);
void mynah_sgemm_stats_get_avx2(mynah_sgemm_stats *out);
void mynah_sgemm_stats_reset_avx2(void);
int mynah_sgemm_self_test_avx2(char *error, size_t error_capacity);
int mynah_sgemm_f32_forced_avx2(mynah_sgemm_family want, mynah_sgemm_family *ran, int trans_a, int trans_b, size_t m, size_t n, size_t k, float alpha, const float *a, size_t lda, const float *b, size_t ldb, float beta, float *c, size_t ldc);
void mynah_sgemm_dispatch_probes_avx2(void);

static int sgemm_use_avx2(void) { return mynah_kernels_x86_avx2(); }

int mynah_sgemm_f32(int trans_a, int trans_b, size_t m, size_t n, size_t k, float alpha, const float *a, size_t lda, const float *b, size_t ldb, float beta, float *c, size_t ldc) {
    return sgemm_use_avx2() ? mynah_sgemm_f32_avx2(trans_a, trans_b, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
                            : mynah_sgemm_f32_base(trans_a, trans_b, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

int mynah_sgemm_f32_conv_taps(size_t m, size_t n, size_t k, size_t taps, const float *weight, float *gather, const float *b, size_t ldb, size_t b_tap_stride, float beta, float *c, size_t ldc) {
    return sgemm_use_avx2() ? mynah_sgemm_f32_conv_taps_avx2(m, n, k, taps, weight, gather, b, ldb, b_tap_stride, beta, c, ldc)
                            : mynah_sgemm_f32_conv_taps_base(m, n, k, taps, weight, gather, b, ldb, b_tap_stride, beta, c, ldc);
}

void mynah_sgemm_f32_reference(int trans_a, int trans_b, size_t m, size_t n, size_t k, float alpha, const float *a, size_t lda, const float *b, size_t ldb, float beta, float *c, size_t ldc) {
    if (sgemm_use_avx2()) { mynah_sgemm_f32_reference_avx2(trans_a, trans_b, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc); return; }
    mynah_sgemm_f32_reference_base(trans_a, trans_b, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

mynah_sgemm_family mynah_sgemm_family_for(int trans_a, int trans_b, size_t m, size_t n, size_t k, const char **why) {
    return sgemm_use_avx2() ? mynah_sgemm_family_for_avx2(trans_a, trans_b, m, n, k, why)
                            : mynah_sgemm_family_for_base(trans_a, trans_b, m, n, k, why);
}

const char *mynah_sgemm_family_name(mynah_sgemm_family family) {
    return sgemm_use_avx2() ? mynah_sgemm_family_name_avx2(family)
                            : mynah_sgemm_family_name_base(family);
}

const char *mynah_sgemm_isa_name(void) {
    return sgemm_use_avx2() ? mynah_sgemm_isa_name_avx2()
                            : mynah_sgemm_isa_name_base();
}

size_t mynah_sgemm_narrow_max(void) {
    return sgemm_use_avx2() ? mynah_sgemm_narrow_max_avx2()
                            : mynah_sgemm_narrow_max_base();
}

void mynah_sgemm_stats_get(mynah_sgemm_stats *out) {
    if (sgemm_use_avx2()) { mynah_sgemm_stats_get_avx2(out); return; }
    mynah_sgemm_stats_get_base(out);
}

void mynah_sgemm_stats_reset(void) {
    if (sgemm_use_avx2()) { mynah_sgemm_stats_reset_avx2(); return; }
    mynah_sgemm_stats_reset_base();
}

int mynah_sgemm_self_test(char *error, size_t error_capacity) {
    return sgemm_use_avx2() ? mynah_sgemm_self_test_avx2(error, error_capacity)
                            : mynah_sgemm_self_test_base(error, error_capacity);
}

int mynah_sgemm_f32_forced(mynah_sgemm_family want, mynah_sgemm_family *ran, int trans_a, int trans_b, size_t m, size_t n, size_t k, float alpha, const float *a, size_t lda, const float *b, size_t ldb, float beta, float *c, size_t ldc) {
    return sgemm_use_avx2() ? mynah_sgemm_f32_forced_avx2(want, ran, trans_a, trans_b, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
                            : mynah_sgemm_f32_forced_base(want, ran, trans_a, trans_b, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void mynah_sgemm_dispatch_probes(void) {
    if (sgemm_use_avx2()) { mynah_sgemm_dispatch_probes_avx2(); return; }
    mynah_sgemm_dispatch_probes_base();
}

#else

/* Not x86: src/sgemm.c defines the public names itself and there is no variant
 * to choose.  This typedef exists only so the translation unit is not empty. */
typedef int mynah_sgemm_rt_not_needed_on_this_target;

#endif
