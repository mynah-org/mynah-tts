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

/* Query: does this backend support device-side matmul (resident GPU)? */
int mynah_backend_has_dev_ops(const mynah_backend *backend);

/* Inplace device ops (no copy, no sync — caller syncs when needed). */
int mynah_backend_gelu_inplace(const mynah_backend *, float *, size_t, char *, size_t);
int mynah_backend_residual_inplace(const mynah_backend *, float *, const float *, size_t, char *, size_t);
int mynah_backend_layer_norm_inplace(const mynah_backend *, const float *, float *, const float *, size_t, size_t, char *, size_t);
int mynah_backend_matmul_to_dev(const mynah_backend *, const float *, float *, size_t, size_t, size_t, const float *, const float *, char *, size_t);
int mynah_backend_matmul_d2d(const mynah_backend *, const float *, float *, size_t, size_t, size_t, const float *, const float *, char *, size_t);
int mynah_backend_im2col(const mynah_backend *, const float *, float *, int, int, int, int, char *, size_t);
int mynah_backend_conv1d(const mynah_backend *, const float *, float *, int, int, int, int, int, const float *, const float *, char *, size_t);
int mynah_backend_gelu_host(const mynah_backend *, float *, size_t, char *, size_t);
int mynah_backend_gelu_host_f64(const mynah_backend *, float *, size_t, char *, size_t);

/* Device-side matvec: out[N] = in[K] @ W[N,K]^T + bias. No sync. */
int mynah_backend_matvec_dev(const mynah_backend *backend,
                             const float *dev_in, float *dev_out,
                             size_t K, size_t N,
                             const float *weight, const float *bias,
                             char *error, size_t error_capacity);

/* ---- Device-side operations (resident GPU inference) ----
 * These keep activations on the device between calls, eliminating
 * per-op H2D/D2H copies.  On CPU backends they are trivial wrappers.
 * The caller uploads input once, chains ops, then downloads output. */

/* Upload n floats from host to a device buffer; returns device pointer. */
int mynah_backend_upload(const mynah_backend *backend, const float *host,
                         size_t n, float **dev_ptr,
                         char *error, size_t error_capacity);
/* Download n floats from device to host. */
int mynah_backend_download(const mynah_backend *backend, const float *dev_ptr,
                           float *host, size_t n,
                           char *error, size_t error_capacity);
/* Block until all queued device work completes. */
int mynah_backend_sync(const mynah_backend *backend,
                       char *error, size_t error_capacity);

/* Device-side matmul: out[rows,ow] = in[rows,iw] @ W[ow,iw]^T + bias.
 * Weight/bias are host pointers (cached on device internally). */
int mynah_backend_matmul_dev(const mynah_backend *backend,
                             const float *dev_in, float *dev_out,
                             size_t rows, size_t iw, size_t ow,
                             const float *weight, const float *bias,
                             char *error, size_t error_capacity);
/* Device-side sgemm (row-major, same semantics as mynah_backend_sgemm). */
int mynah_backend_sgemm_dev(const mynah_backend *backend,
                            int trans_a, int trans_b,
                            size_t m, size_t n, size_t k,
                            float alpha,
                            const float *dev_a, size_t lda,
                            const float *dev_b, size_t ldb,
                            float beta,
                            float *dev_c, size_t ldc,
                            char *error, size_t error_capacity);
/* Element-wise ops on device buffers. */
int mynah_backend_gelu_dev(const mynah_backend *backend,
                           float *dev_data, size_t n,
                           char *error, size_t error_capacity);
int mynah_backend_layer_norm_dev(const mynah_backend *backend,
                                 const float *dev_in, float *dev_out,
                                 const float *gain, const float *bias,
                                 size_t rows, size_t width,
                                 char *error, size_t error_capacity);
int mynah_backend_softmax_dev(const mynah_backend *backend,
                              float *dev_data, size_t rows, size_t cols,
                              size_t valid,
                              char *error, size_t error_capacity);
int mynah_backend_residual_add_dev(const mynah_backend *backend,
                                   float *dev_out, const float *dev_in,
                                   size_t n,
                                   char *error, size_t error_capacity);
int mynah_backend_snake_dev(const mynah_backend *backend,
                            float *dev_data, const float *alpha,
                            size_t channels, size_t length,
                            size_t snake_channels,
                            char *error, size_t error_capacity);

/* Copy n floats from host to a specific device buffer (no scratch). */
int mynah_backend_h2d(const mynah_backend *backend, const float *host,
                      float *dev_ptr, size_t n,
                      char *error, size_t error_capacity);
/* Copy n floats from a specific device buffer to host (no scratch). */
int mynah_backend_d2h(const mynah_backend *backend, const float *dev_ptr,
                      float *host, size_t n,
                      char *error, size_t error_capacity);

/* Allocate a persistent device buffer of n floats.  On CPU returns a
 * malloc'd host buffer; on CUDA a cudaMalloc'd device buffer.
 * The caller owns the buffer and must free it with mynah_backend_dev_free. */
int mynah_backend_dev_alloc(const mynah_backend *backend, size_t n,
                            float **dev_ptr, char *error, size_t error_capacity);
void mynah_backend_dev_free(const mynah_backend *backend, float *dev_ptr);

#endif
