/* INT8 weight quantization for the autoregressive hot loop.
 *
 * Policy from the ../mynah / ../qwen-tts house pattern: only the big linear
 * weights consumed by the decode-step matvecs are quantized (per-row symmetric
 * absmax, scale[n]); conv/norm/bias/embedding stay f32.  The dispatch is:
 *   - small count (decode step, count <= threshold): native int8xint8 dot
 *     (ARM SDOT when available, scalar otherwise), activations quantized per row;
 *   - large count (context prefill): fall back to the f32 BLAS matmul, which is
 *     already fast and keeps the prefill bit-exact.
 *
 * Weights are quantized once and cached by tensor name for the model's lifetime.
 * INT8 changes the numerics: output is close to f32, not bit-identical. */
#ifndef MYNAH_TTS_QMAT_H
#define MYNAH_TTS_QMAT_H

#include <stddef.h>
#include <stdint.h>

#include "backend.h"
#include "weights.h"

typedef struct mynah_qmat_cache mynah_qmat_cache;

/* enabled: 0 = always f32 (cache is a no-op passthrough), 1 = int8 for small
 * count.  Reads env MYNAH_QUANT ("int8"/"f32") when passed -1. */
/* MYNAH_QUANT resolved to a QMAT_* code, or -1 when nothing was asked for.
 * Empty is unset; an unrecognised value is reported once and is unset too.  Use
 * this rather than reading the variable: two call sites reading it differently
 * is what made `MYNAH_QUANT=` mean a third thing. */
int mynah_qmat_qtype_from_env(void);

mynah_qmat_cache *mynah_qmat_cache_new(int enabled);
void mynah_qmat_cache_free(mynah_qmat_cache *cache);
int mynah_qmat_cache_enabled(const mynah_qmat_cache *cache);

/* The cache's own resolution, for the dispatch report and for anyone who needs
 * to know what actually got selected rather than what was asked for.  The
 * qtype codes are 0 f32 (off), 1 int8, 2 int4, 3 f16; mynah_qmat_qtype_name
 * turns one into the string the report prints. */
int mynah_qmat_cache_qtype(const mynah_qmat_cache *cache);
const char *mynah_qmat_qtype_name(int qtype);
/* 1 = the four-row unrolled matvec, 0 = MYNAH_QMAT_SINGLE_ROW rollback. */
int mynah_qmat_cache_row4(const mynah_qmat_cache *cache);

/* TEST HOOK, not a runtime knob.  Forces the ARM SMMLA wiring in the batched
 * linear on (1) or off (0) for the rest of the process, or restores the
 * env/default resolution (-1); returns the mode that was in effect before.
 * Reports -1 and does nothing where SMMLA is not compiled or the CPU has no
 * FEAT_I8MM, so a test can call it unconditionally.
 *
 * WHY IT IS PUBLIC: the two int8 kernels must produce bit-identical output for
 * the same (weight row, activation), or a request's audio would depend on who
 * it was batched with.  Proving that needs both kernels in ONE process over the
 * same data; an env variable can only pick one per run. */
int mynah_qmat_i8mm_force(int mode);

/* The canonical int8 float epilogue, exported so a test can pin the grouping
 * the kernels compiled to.  `ws` is the weight row's scale and `sx` the
 * activation's; the result is `(float)s * (ws * sx) + bias`, with the inner
 * product forced to be computed first.  See the long comment above
 * qmat_row_scale() in src/qmat.c for why that forcing is load-bearing. */
float mynah_qmat_epilogue(int32_t s, float ws, float sx, float bias);

/* Would a greedy projection of this shape be split over the thread pool?
 * Shape-dependent on purpose: the same binary threads a big projection and
 * runs a small one serially.  `why` (optional) receives a static string
 * naming the clause that decided. */
int mynah_qmat_argmax_mt_resolved(size_t rows, size_t cols, const char **why);

/* Names the int8 kernel this host resolves to: "avx512vnni", "avxvnni",
 * "u8-scalar", "neon-sdot", "avx2" or "scalar".  `why` (optional) receives a
 * static reason string. */
const char *mynah_qmat_int8_kernel(const char **why);
/* 1 when the SMMLA path is compiled AND the CPU reports FEAT_I8MM. */
int mynah_qmat_i8mm_enabled(const char **why);

/* Names the int4 kernel this build resolves to: "neon-sdot", "avx2" or
 * "scalar".  Unlike int8 this is a compile-time choice, not a CPUID one.
 * `why` (optional) receives a static reason string. */
const char *mynah_qmat_int4_kernel(const char **why);

/* Names the f16 kernel this host resolves to: "neon", "f16c", "scalar" or
 * "off".  "off" is the one that used to be silent: on a build with no half
 * weight type, mynah_qmat_cache_new() rewrites QMAT_F16 to QMAT_F32 and the
 * run proceeds at f32 speed with nothing said.  `why` (optional) receives a
 * static reason string. */
const char *mynah_qmat_f16_kernel(const char **why);
/* MYNAH_FUSED_GREEDY: whether the engine may fuse head projection + argmax. */
int mynah_qmat_fused_greedy_enabled(void);

/* The raw MYNAH_QUANT_GROUPS request, or "default" when it is unset.  WHICH
 * weight groups exist is the engine's vocabulary, so this file reads the
 * variable once (so the dispatch report and the engine agree on what was
 * asked) and leaves parsing -- and rejecting an unknown name -- to the engine.
 * Never NULL. */
const char *mynah_qmat_groups_spec(void);

/* out[count, n] = in[count, k] @ W[n, k]^T (+ bias), where W is the tensor
 * `name` in `file`.  Uses the cached int8 weight when the cache is enabled and
 * count is small; otherwise the f32 backend matmul.  0 = ok, -1 = error. */
int mynah_qmat_linear(mynah_qmat_cache *cache, const mynah_weights *file,
                      const mynah_backend *backend, const char *name,
                      const float *in, float *out, size_t count, size_t k, size_t n,
                      const float *bias, char *error, size_t error_capacity);
int mynah_qmat_linear_resolved(mynah_qmat_cache *cache, const mynah_backend *backend,
                               const char *name, const float *weight,
                               const float *in, float *out, size_t count, size_t k,
                               size_t n, const float *bias,
                               char *error, size_t error_capacity);

/* The same call with the encoding named per tensor instead of per cache.
 *
 * WHY: f16 and int8 fail differently.  f16 is numerically exact here (measured
 * 1.2e-06 per PocketTTS backbone step) at half the weight bytes; int8 is 1-3%
 * per operand at a quarter, and is the encoding SDOT, VNNI and AMX accelerate.
 * Which one is right is therefore a property of the *tensor*, not of the
 * process, and a single MYNAH_QUANT cannot express "f16 where the error feeds
 * back through an AR loop, int8 where it cannot".
 *
 * `qtype`: 0 f32 (exact, no cache entry), 1 int8, 2 int4, 3 f16, or -1 for
 * "whatever the cache resolved to" -- which is what mynah_qmat_linear_resolved
 * passes, so every existing caller keeps its exact behaviour.  A qtype this
 * build cannot honour (f16 off ARM) resolves to f32, never to a substitute. */
int mynah_qmat_linear_resolved_qt(mynah_qmat_cache *cache,
                                  const mynah_backend *backend, const char *name,
                                  const float *weight, const float *in, float *out,
                                  size_t count, size_t k, size_t n,
                                  const float *bias, int qtype, char *error,
                                  size_t error_capacity);

/* "int8" -> 1, "int4" -> 2, "f16" -> 3, "f32"/"off" -> 0, anything else -1. */
int mynah_qmat_qtype_from_name(const char *name);
/* What this build can actually honour for `qtype` (f16 -> f32 off ARM). */
int mynah_qmat_qtype_resolved(int qtype);

/* Weight-stationary batched linear: `batch` single-row activations that belong
 * to different requests, computed with one pass over the weight instead of one
 * pass each.  `in_rows[b]` and `out_rows[b]` need not be contiguous with one
 * another, so slots keep their own scratch.
 *
 * Bit-exact against `batch` separate mynah_qmat_linear_resolved calls: without
 * that, a request's audio would depend on which requests it batched with.
 *
 * The caller owns the scratch (`batch * k` int8 and `batch` floats) so the
 * decode loop stays allocation-free.  0 = ok, -1 = error. */
int mynah_qmat_linear_batched(mynah_qmat_cache *cache, const mynah_backend *backend,
                              const char *name, const float *weight,
                              const float *const *in_rows, float *const *out_rows,
                              size_t batch, size_t k, size_t n, const float *bias,
                              int8_t *qx_scratch, float *sx_scratch,
                              char *error, size_t error_capacity);

/* The same call with the encoding named per tensor instead of per cache -- what
 * `mynah_qmat_linear_resolved_qt` is to `mynah_qmat_linear_resolved`.
 *
 * WHY: `mynah_qmat_linear_batched` takes no qtype, so it both gates on the
 * cache's own encoding and, on a first touch, creates the cache entry in it.
 * A group carrying an explicit encoding therefore could not use the
 * weight-stationary path at all without a first-touch race deciding its
 * precision -- which under `MYNAH_QUANT=int8` pushed every `:f16` group (the
 * PocketTTS backbone and flow head) back onto one weight pass per row.  Here
 * `qtype` decides both the gate and the entry, so precision comes from the
 * group spec and never from whichever caller arrived first.
 *
 * `qtype`: 0 f32, 1 int8, 2 int4, 3 f16, or -1 for "whatever the cache
 * resolved to" -- which is what `mynah_qmat_linear_batched` now passes, so
 * every existing caller keeps its exact behaviour.  Row b stays bit-exact
 * against `mynah_qmat_linear_resolved_qt(..., 1, ..., qtype)`; checked by
 * mynah_qmat_self_test over every cache profile crossed with every encoding a
 * group spec can name. */
int mynah_qmat_linear_batched_qt(mynah_qmat_cache *cache,
                                 const mynah_backend *backend, const char *name,
                                 const float *weight,
                                 const float *const *in_rows,
                                 float *const *out_rows, size_t batch, size_t k,
                                 size_t n, const float *bias, int8_t *qx_scratch,
                                 float *sx_scratch, int qtype, char *error,
                                 size_t error_capacity);

/* Greedy f32 projection fused with the constrained argmax.  Returns 0 when
 * fused, 1 when the cache/backend is not eligible and the caller should use
 * mynah_qmat_linear, or -1 on a model/input error. */
int mynah_qmat_greedy_argmax(mynah_qmat_cache *cache, const mynah_weights *file,
                             const char *name, const float *in, size_t k, size_t n,
                             const float *bias, size_t allowed_rows,
                             unsigned extra_row, int allow_extra, unsigned *argmax,
                             char *error, size_t error_capacity);
int mynah_qmat_greedy_argmax_resolved(mynah_qmat_cache *cache, const char *name,
                                      const float *weight, const float *in,
                                      size_t k, size_t n, const float *bias,
                                      size_t allowed_rows, unsigned extra_row,
                                      int allow_extra, unsigned *argmax,
                                      char *error, size_t error_capacity);

/* ------------------------------------------------- the int8 primitives
 *
 * Exported for a SECOND consumer with no tensor name to key a cache on: the
 * SEANet conv stack (src/convq8.c).  The long comment in src/qmat.c says why
 * they are exported rather than rewritten there -- in one line, the
 * activation encoding is a property of the host (signed for SDOT, unsigned
 * x+128 for VPDPBUSD) and a second copy of that dispatch would give up VNNI
 * on the x86 half of production without saying so.
 *
 * The caller owns the float epilogue: these return exact int32, and
 * mynah_qmat_epilogue() above is the one expression shape that turns one into
 * a float. */

/* Bytes of activation scratch one vector of length k needs.  It is k, and it
 * is a function so a caller cannot assume the element type: above
 * QMAT_U8_OFF the bytes are unsigned. */
size_t mynah_qmat_act_bytes(size_t k);

/* Quantizes one f32 vector into this host's activation encoding and returns
 * its scale.  `dst` holds mynah_qmat_act_bytes(k) bytes. */
float mynah_qmat_act_quantize(void *dst, const float *x, size_t k);

/* Packs an f32 [rows][cols] block into per-row symmetric absmax int8 plus the
 * row sums the unsigned encoding needs.  The row sums are written on every
 * host, so a packed block does not depend on who packed it.  `cols` must be
 * at most the k bound the row-sum's no-overflow argument assumes; -1 says so.
 * Sizes: rows*cols int8, rows floats, rows int32. */
int mynah_qmat_pack_q8(const float *w, size_t rows, size_t cols, int8_t *q,
                       float *scale, int32_t *rowsum);

/* Activation vectors one mynah_qmat_dots_i8 call may carry. */
size_t mynah_qmat_dots_max_batch(void);

/* out[b * out_stride + row] = exact int32 inner product of weight row `row`
 * (int8, [rows][cols], contiguous in cols) with activation `xq[b]`, which was
 * produced by mynah_qmat_act_quantize.  `rowsum` is mynah_qmat_pack_q8's and
 * is read only where the host uses the unsigned encoding.
 *
 * The result does not depend on `batch`, on the ISA, or on the encoding:
 * integer accumulation is exact, so every compiled path returns the same
 * int32.  Asserted with == by the self-test, not with a tolerance. */
void mynah_qmat_dots_i8(const int8_t *w, size_t rows, size_t cols,
                        const int32_t *rowsum, const void *const *xq,
                        size_t batch, int32_t *out, size_t out_stride);

/* Model-free numeric check: int8 matvec vs an exact f32 dot on deterministic
 * data, asserting a bounded relative error.  0 = ok, -1 = error. */
/* Model-free bf16 matvec: out[row] = sum_j bf16(w[row][cols]) * bf16(x[j]),
 * plus bias, through whichever bf16 kernel this host resolved (BFDOT/BFMMLA on
 * Arm, VDPBF16PS or the AVX2 widening form on x86, scalar otherwise).
 *
 * Public because a kernel that cannot be called without a model pack cannot be
 * BENCHMARKED without one either, and the x86 bf16 tiers landed with no way to
 * compare them on a host that has both. Weights are raw IEEE-754 bf16 bit
 * patterns, row-major, exactly as the cache holds them. */
void mynah_qmat_matvec_bf16(float *out, const float *x, const uint16_t *w,
                            const float *bias, size_t rows, size_t cols);

/* Is the BFDOT kernel the one that will run?  `why` receives a [predicate]
 * string for the dispatch report.  See src/qmat.c and
 * .work/bf16-native-weights.md. */
int mynah_qmat_bf16_enabled(const char **why);

int mynah_qmat_self_test(char *error, size_t error_capacity);

#endif
