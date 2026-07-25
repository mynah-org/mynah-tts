#ifndef MYNAH_BACKEND_H
#define MYNAH_BACKEND_H

#include "mynah_tts.h"

#include <stddef.h>

typedef struct mynah_backend mynah_backend;

typedef int (*mynah_backend_matmul_fn)(void *, const float *, float *, size_t, size_t,
                                       size_t, const float *, const float *, char *, size_t);
typedef void (*mynah_backend_close_fn)(void *);
typedef int (*mynah_backend_self_test_fn)(void *, char *, size_t);

/* Generic sgemm mirroring cblas_sgemm semantics (row-major).
 * C[m,n] = alpha * op(A) * op(B) + beta * C
 * trans_a/trans_b: 0 = NoTrans, 1 = Trans. */
typedef int (*mynah_backend_sgemm_fn)(void *, int trans_a, int trans_b,
                                      size_t m, size_t n, size_t k,
                                      float alpha,
                                      const float *a, size_t lda,
                                      const float *b, size_t ldb,
                                      float beta,
                                      float *c, size_t ldc,
                                      char *, size_t);

int mynah_backend_open(mynah_tts_device device, mynah_backend **out,
                       char *error, size_t error_capacity);
void mynah_backend_close(mynah_backend *backend);
const char *mynah_backend_name(const mynah_backend *backend);

int mynah_backend_matmul(const mynah_backend *backend, const float *input,
                         float *output, size_t rows, size_t input_width,
                         size_t output_width, const float *weight,
                         const float *bias, char *error, size_t error_capacity);

int mynah_backend_sgemm(const mynah_backend *backend,
                        int trans_a, int trans_b,
                        size_t m, size_t n, size_t k,
                        float alpha,
                        const float *a, size_t lda,
                        const float *b, size_t ldb,
                        float beta,
                        float *c, size_t ldc,
                        char *error, size_t error_capacity);

int mynah_backend_self_test(mynah_tts_device device, char *error,
                            size_t error_capacity);

#endif
