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

#include "backend.h"
#include "safetensors.h"

typedef struct mynah_qmat_cache mynah_qmat_cache;

/* enabled: 0 = always f32 (cache is a no-op passthrough), 1 = int8 for small
 * count.  Reads env MYNAH_QUANT ("int8"/"f32") when passed -1. */
mynah_qmat_cache *mynah_qmat_cache_new(int enabled);
void mynah_qmat_cache_free(mynah_qmat_cache *cache);
int mynah_qmat_cache_enabled(const mynah_qmat_cache *cache);

/* out[count, n] = in[count, k] @ W[n, k]^T (+ bias), where W is the tensor
 * `name` in `file`.  Uses the cached int8 weight when the cache is enabled and
 * count is small; otherwise the f32 backend matmul.  0 = ok, -1 = error. */
int mynah_qmat_linear(mynah_qmat_cache *cache, const mynah_safetensors *file,
                      const mynah_backend *backend, const char *name,
                      const float *in, float *out, size_t count, size_t k, size_t n,
                      const float *bias, char *error, size_t error_capacity);

/* Greedy f32 projection fused with the constrained argmax.  Returns 0 when
 * fused, 1 when the cache/backend is not eligible and the caller should use
 * mynah_qmat_linear, or -1 on a model/input error. */
int mynah_qmat_greedy_argmax(mynah_qmat_cache *cache, const mynah_safetensors *file,
                             const char *name, const float *in, size_t k, size_t n,
                             const float *bias, size_t allowed_rows,
                             unsigned extra_row, int allow_extra, unsigned *argmax,
                             char *error, size_t error_capacity);

/* Model-free numeric check: int8 matvec vs an exact f32 dot on deterministic
 * data, asserting a bounded relative error.  0 = ok, -1 = error. */
int mynah_qmat_self_test(char *error, size_t error_capacity);

#endif
