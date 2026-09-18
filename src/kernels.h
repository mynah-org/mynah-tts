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
 * backbone both use.  Both forms now call mynah_tanh_f32 below, so the
 * elementwise and array spellings are the same arithmetic on every target.
 * `scratch` is accepted and ignored: the vector tanh works in registers and
 * needs no staging buffer.  Callers may keep passing NULL. */
float mynah_gelu_tanh(float x);
void  mynah_gelu_tanh_array(float *values, size_t length, float *scratch);
int   mynah_gelu_self_test(char *error, size_t error_capacity);

/* ------------------------------------------------------------------------
 * Vector transcendentals -- ours, on every target (PLAN.md E4-16d)
 *
 * These replace Accelerate's vvtanhf and vvsinf/vDSP Snake, which existed on
 * macOS only.  The point is not speed: it is that the development platform
 * and the production target must execute the same arithmetic, instead of two
 * implementations that agree by luck.  Detail and the ULP measurements:
 * .work/accelerate-only-kernels.md.
 *
 * For every function the `_scalar` spelling is the DEFINITION OF CORRECTNESS
 * and is always compiled.  The unsuffixed spelling dispatches to NEON or AVX2
 * where one exists and to the scalar body otherwise.  Both are exported so
 * that mynah_vecmath_self_test() can compare them inside one binary with no
 * environment variable and no model -- if the two ever disagree by more than
 * a stated ULP bound on this machine, the build fails here rather than in an
 * utterance three hundred steps long.
 *
 * Domain notes that are part of the contract:
 *   tanh  -- exact on the whole line.  +-Inf -> +-1, NaN passes through,
 *            -0.0 stays -0.0, denormals return themselves.
 *   sin   -- the vector argument reduction is valid for |x| < 8192.  Outside
 *            that, and for every non-finite input, the lane is recomputed
 *            with libm sinf, so the vector and scalar paths are identical
 *            there by construction.
 * ------------------------------------------------------------------------ */
void mynah_tanh_f32(const float *input, float *output, size_t n);
void mynah_tanh_f32_scalar(const float *input, float *output, size_t n);
void mynah_sin_f32(const float *input, float *output, size_t n);
void mynah_sin_f32_scalar(const float *input, float *output, size_t n);

/* One SEANet Snake channel row, fused and in place:
 *     row[t] = row[t] + sin(alpha * row[t])^2 / (alpha + 1e-9)
 * Fused on purpose.  The Accelerate spelling it replaces staged the sines
 * through a malloc'd array on every call, which is an allocation on the codec
 * path (coding rule 4); this one works a vector at a time in registers. */
void mynah_snake_row_f32(float *row, size_t length, float alpha);
void mynah_snake_row_f32_scalar(float *row, size_t length, float alpha);

/* Which transcription of the above will actually run here?  Returns a static
 * string: "NEON", "AVX2" or "scalar". */
const char *mynah_vecmath_isa(void);

/* Does this BUILD flush denormals to zero before any kernel sees them?  A
 * property of the compiler driver, not of these kernels: -ffast-math sets the
 * FPCR flush-to-zero bit through a startup object on gcc/aarch64 and does not
 * on Apple clang, so the development platform preserves a denormal argument
 * and the production one does not.  Reported so that a self test can assert
 * the right thing instead of asserting the compiler's behaviour and calling
 * it a kernel bug. */
int mynah_vecmath_denormals_flush(void);

/* Model-free: scalar vs vector agreement, and both against a double-precision
 * reference, across zero, negative zero, denormals, the tanh saturation knee,
 * infinities, NaN, and the sine argument reduction far from the origin. */
int mynah_vecmath_self_test(char *error, size_t error_capacity);

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
