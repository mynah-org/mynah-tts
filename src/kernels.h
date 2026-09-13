#ifndef MYNAH_TTS_KERNELS_H
#define MYNAH_TTS_KERNELS_H

#include <stddef.h>

float mynah_dot_f32(const float *a, const float *b, size_t n);
void mynah_matvec_f32(const float *weights, const float *input, float *output,
                      size_t rows, size_t cols);
void mynah_matvec_bias_f32(const float *weights, const float *input,
                           const float *bias, float *output, size_t rows,
                           size_t cols);
int mynah_matvec_argmax_f32(const float *weights, const float *input,
                            const float *bias, size_t rows, size_t cols,
                            size_t allowed_rows, size_t extra_row,
                            int allow_extra, unsigned *argmax);
void mynah_rmsnorm_f32(const float *input, const float *weight, float *output,
                       size_t n, float epsilon);
void mynah_layernorm_f32(const float *input, const float *weight,
                         const float *bias, float *output, size_t rows,
                         size_t width, float epsilon);
void mynah_residual_add_f32(float *output, const float *input, size_t n);
void mynah_gelu_f32(float *data, size_t n);
void mynah_gelu_f32_scalar(float *data, size_t n);

/* Is the vectorized GELU (NEON/AVX2 Pade tanh) the one that will run?
 * 0 means the libm scalar reference, either because MYNAH_GELU_SCALAR is set
 * or because no vector kernel is compiled for this target. */
int mynah_gelu_vector_enabled(void);

/* tanh-approximation GELU, the form the Magpie conv-FFN and the PocketTTS
 * backbone both use. The array form takes optional scratch and uses vForce
 * when Accelerate is present; pass NULL for the scalar loop. */
float mynah_gelu_tanh(float x);
void  mynah_gelu_tanh_array(float *values, size_t length, float *scratch);
int   mynah_gelu_self_test(char *error, size_t error_capacity);

/* out[0..n) += weight * src[0..n) */
void mynah_axpy_f32(float *out, const float *src, float weight, size_t n);
int mynah_softmax_f32(const float *logits, float *probabilities, size_t n);
size_t mynah_argmax_f32(const float *values, size_t n);
int mynah_kernels_self_test(char *error, size_t error_capacity);

/* ------------------------------------------------------------------------
 * The ISA kernel inventory
 *
 * Which vector units does this binary actually contain a kernel for?  The
 * dispatch report needs the answer for its `resolved` column and is forbidden
 * from re-deriving it from a #if of its own (src/dispatch.h, "THE CENTRAL
 * RULE"), so the answer is exported from the file that would hold the kernel.
 *
 * WHY THIS EXISTS.  On the project's own Linux box -- Neoverse V2, whose
 * /proc/cpuinfo advertises `sve sve2 svei8mm svebf16 i8mm bf16` -- the report
 * printed no SVE row at all.  Silence read as "nothing to see", when what it
 * meant was "four vector units on the production CPU that this runtime does
 * not touch".  That is precisely the failure the report exists to catch, so
 * each of them now has a row that resolves OFF through this function.
 *
 * Every bit is 0 today and each 0 is a statement, not an oversight.  When a
 * kernel lands it flips its bit HERE, in the same edit, and the row follows.
 * A module that grows its own kernel for one of these (an SVE int8 matvec
 * belongs in src/qmat.c, not here) registers its own probe over the same row
 * id -- mynah_dispatch_register_probe() replaces by id, by design.
 * ------------------------------------------------------------------------ */
#define MYNAH_KERNELS_ISA_SVE      (1u << 0)  /* any SVE-predicated f32 kernel */
#define MYNAH_KERNELS_ISA_SVE2     (1u << 1)
#define MYNAH_KERNELS_ISA_SVEI8MM  (1u << 2)  /* SMMLA under an SVE predicate  */
#define MYNAH_KERNELS_ISA_SVEBF16  (1u << 3)  /* BFMMLA/BFDOT, SVE form        */
#define MYNAH_KERNELS_ISA_BF16     (1u << 4)  /* NEON bfdot/bfmmla, or a bf16
                                               * weight type kept as bf16      */
unsigned mynah_kernels_isa_kernels(void);

/* Human text for one of the bits above: what would have to be written, and
 * where, for it to become 1.  Never NULL. */
const char *mynah_kernels_isa_missing_reason(unsigned bit);

#endif
