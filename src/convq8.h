/*
 * INT8 tap GEMM for the SEANet codec conv stack (E10-5).
 *
 * WHY THIS FILE EXISTS.  `codec.conv_stack` is a quarter of the wall, and in
 * the configuration production ships (`BLAS=none`, our own sgemm) its two
 * GEMM families are 81.5% of the region.  One of the two -- the causal
 * conv1d's per-tap GEMM -- is 42% and is what this replaces.  The other, the
 * three transposed convolutions, is deliberately NOT here: the reference
 * implementation measured its int8 convtranspose SLOWER than the f32 sgemm
 * and ships it off, and re-deriving that result is not worth the weight
 * memory it would cost to try.
 *
 * WHY NOT src/qmat.c.  That file's entry points are keyed by TENSOR NAME,
 * because its consumer is an engine that reads a `model.json`.  This one's
 * consumer is src/seanet.c, which "never formats a tensor name" by design and
 * whose operand is a permutation of a weight qmat never sees.  What is shared
 * is the arithmetic: the int8 primitives are qmat's, exported, so the
 * activation encoding (signed for SDOT, unsigned x+128 for VPDPBUSD) has ONE
 * definition rather than two that agree.
 *
 * WHAT IT CHANGES.  Output, by about a percent per operand -- int8 is an
 * approximation and no tolerance argument makes it otherwise.  It is reached
 * only when the engine's `codec_conv` quantization group resolves to int8,
 * which is the same sentence that already decides the codec transformer's
 * encoding, so there is one vocabulary and not a second env variable.
 */
#ifndef MYNAH_TTS_CONVQ8_H
#define MYNAH_TTS_CONVQ8_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* c[i][j] = bias[i] + sum_t sum_p weight[i][p][t] * b[p * ldb + j + t * tap_stride]
 *
 * The same operand shape as mynah_sgemm_f32_conv_taps, which is the f32 path
 * this stands in for, PLUS the bias: it is folded into the int8 epilogue here,
 * so a caller that gets 0 back must not run its own bias pass.
 *
 * `weight` is the f32 [m][k][taps] tensor, used as the identity of the weight
 * and quantized once per process behind this call.  It is never read again
 * after the first call for a given pointer.
 *
 * 0  = it ran and `c` is written.
 * 1  = refused; the caller runs its f32 path unchanged and `c` is untouched.
 * -1 = a caller error (null pointer, impossible shape).
 */
int mynah_convq8_conv_taps(size_t m, size_t n, size_t k, size_t taps,
                           const float *weight, const float *b, size_t ldb,
                           size_t tap_stride, const float *bias, float *c,
                           size_t ldc);

/* c[i][j] = sum_p weight[p * ldw + i] * b[p * ldb + j] -- the SAME product
 * with op(A) transposed, which is the shape a causal ConvTranspose1d takes.
 * PyTorch stores that weight as [in_channels][out_channels * kernel], so with
 * groups == 1 the logical row i is a COLUMN of the stored tensor and `ldw` is
 * out_channels * kernel.  No bias: the transposed path adds it when it fills
 * the output buffer, before the scatter, and folding it in here would move a
 * rounding the caller has already paid.
 *
 * Same return contract: 0 ran, 1 refused, -1 caller error.
 */
int mynah_convq8_gemm_tn(size_t m, size_t n, size_t k, const float *weight,
                         size_t ldw, const float *b, size_t ldb, float *c,
                         size_t ldc);

/* Would this host run the path at all?  Separate from the shape gate above
 * because "this CPU has no int8 unit" and "this shape is too small" are
 * different facts.  `why` (optional) receives a static reason string. */
int mynah_convq8_host_ok(const char **why);

/* TEST HOOK, not a runtime knob: forces the host gate on (1) or off (0) for
 * the rest of the process, or restores the resolution (-1); returns the mode
 * that was in effect before.  The self-test needs the arithmetic to run on a
 * host whose int8 kernel is the scalar fallback, where the gate would
 * (correctly) refuse. */
int mynah_convq8_force(int mode);

/* Counted rather than argued: a refusal is invisible from the outside -- the
 * f32 path produces valid audio -- so without these a claim that the int8
 * path ran would be unfalsifiable. */
typedef struct {
    unsigned long long calls;
    unsigned long long ran;
    unsigned long long refused_host;     /* no int8 unit worth using        */
    unsigned long long refused_shape;    /* too small, too wide, degenerate */
    unsigned long long refused_pack;     /* quantization or memo failed     */
    unsigned long long refused_scratch;  /* activation scratch failed       */
    unsigned long long packed_tensors;
    unsigned long long packed_bytes;
} mynah_convq8_stats;
void mynah_convq8_stats_get(mynah_convq8_stats *out);

/* Model-free: the int8 tap GEMM against an exact f32 triple loop over the
 * measured conv shapes, with ABSOLUTE error bounds (a relative-to-baseline
 * gate passes mutations, see E10-6), plus determinism across thread counts
 * and across batch widths.  0 = pass. */
int mynah_convq8_self_test(char *error, size_t error_capacity);

void mynah_convq8_dispatch_probes(void);

#ifdef __cplusplus
}
#endif

#endif
