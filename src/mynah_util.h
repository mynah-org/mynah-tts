/* Small services shared by every graph module.
 *
 * These lived as file-static helpers in graph.c. They are here because the
 * split (PLAN.md E1) gives each of transformer, codec, inference and the
 * engines its own translation unit, and all of them need the same five things:
 * a uniform error string, a backend-agnostic sgemm, a monotonic clock for the
 * MYNAH_TIMING paths, a tensor lookup that reports the missing name, and a
 * checked float allocation.
 *
 * Nothing here knows about a model, an engine, or a request.
 */
#ifndef MYNAH_TTS_UTIL_H
#define MYNAH_TTS_UTIL_H

#include <stddef.h>

#include "backend.h"
#include "weights.h"

/* Write `message` into `error` when both are usable. */
void mynah_graph_error(char *error, size_t capacity, const char *message);

/* Backend-agnostic sgemm, so a GPU backend can take every matmul rather than
 * only the ones that happen to go through linear(). */
int mynah_graph_sgemm(const mynah_backend *backend,
                      int trans_a, int trans_b,
                      size_t m, size_t n, size_t k,
                      float alpha,
                      const float *a, size_t lda,
                      const float *b, size_t ldb,
                      float beta,
                      float *c, size_t ldc,
                      char *error, size_t error_capacity);

/* Monotonic seconds. Only used on the MYNAH_TIMING paths. */
double mynah_phase_seconds(void);

/* Look up a tensor, reporting the name that was missing. */
int mynah_tensor_get(const mynah_weights *file, const char *name,
                     mynah_tensor *out, char *error, size_t error_capacity);

/* calloc of `count` floats with the overflow check that a raw calloc lacks. */
float *mynah_alloc_floats(size_t count, char *error, size_t error_capacity);

#endif
