#include "backend.h"
#include "threads.h"

#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(MYNAH_USE_ACCELERATE)
#include <Accelerate/Accelerate.h>
#elif defined(MYNAH_USE_OPENBLAS)
#include <cblas.h>
#endif

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif
#if defined(__AVX2__)
#include <immintrin.h>
#endif

struct mynah_backend {
    mynah_tts_device device;
    const char *name;
    void *state;
    mynah_backend_matmul_fn matmul;
    mynah_backend_sgemm_fn sgemm;
    mynah_backend_close_fn close;
    mynah_backend_self_test_fn self_test;
    /* Device-side ops (NULL = CPU fallback in backend.c). */
    int (*upload)(void *, const float *, size_t, float **, char *, size_t);
    int (*download)(void *, const float *, float *, size_t, char *, size_t);
    int (*sync)(void *, char *, size_t);
    int (*snake_dev)(void *, float *, const float *, size_t, size_t, size_t, char *, size_t);
    int (*gelu_dev)(void *, float *, size_t, char *, size_t);
    int (*layer_norm_dev)(void *, const float *, float *, const float *, const float *, size_t, size_t, char *, size_t);
    int (*residual_add_dev)(void *, float *, const float *, size_t, char *, size_t);
    int (*matmul_dev)(void *, const float *, float *, size_t, size_t, size_t, const float *, const float *, char *, size_t);
    int (*sgemm_dev)(void *, int, int, size_t, size_t, size_t, float, const float *, size_t, const float *, size_t, float, float *, size_t, char *, size_t);
    int (*dev_alloc)(void *, size_t, float **, char *, size_t);
    void (*dev_free)(void *, float *);
    int (*h2d)(void *, const float *, float *, size_t, char *, size_t);
    int (*d2h)(void *, const float *, float *, size_t, char *, size_t);
    int (*matvec_dev)(void *, const float *, float *, size_t, size_t, const float *, const float *, char *, size_t);
    int (*gelu_inplace)(void *, float *, size_t, char *, size_t);
    int (*residual_inplace)(void *, float *, const float *, size_t, char *, size_t);
    int (*layer_norm_inplace)(void *, const float *, float *, const float *, size_t, size_t, char *, size_t);
    int (*matmul_to_dev)(void *, const float *, float *, size_t, size_t, size_t, const float *, const float *, char *, size_t);
    int (*matmul_d2d)(void *, const float *, float *, size_t, size_t, size_t, const float *, const float *, char *, size_t);
    int (*im2col)(void *, const float *, float *, int, int, int, int, char *, size_t);
    int (*conv1d)(void *, const float *, float *, int, int, int, int, int, const float *, const float *, char *, size_t);
    int (*gelu_host)(void *, float *, size_t, char *, size_t);
    int (*gelu_host_f64)(void *, float *, size_t, char *, size_t);
    int (*matmul_graph)(void *, const float *, float *, size_t, size_t, size_t, const float *, const float *, char *, size_t);
};

#if defined(MYNAH_ENABLE_METAL)
int mynah_backend_metal_open(void **state, mynah_backend_matmul_fn *matmul,
                             mynah_backend_sgemm_fn *sgemm,
                             mynah_backend_close_fn *close, mynah_backend_self_test_fn *self_test,
                             char *error, size_t error_capacity);
#endif
#if defined(MYNAH_ENABLE_CUDA)
int mynah_backend_cuda_open(void **state, mynah_backend_matmul_fn *matmul,
                            mynah_backend_sgemm_fn *sgemm,
                            mynah_backend_close_fn *close, mynah_backend_self_test_fn *self_test,
                            char *error, size_t error_capacity);
/* Device-side ops set after open. */
extern int mynah_cuda_upload(void *, const float *, size_t, float **, char *, size_t);
extern int mynah_cuda_download(void *, const float *, float *, size_t, char *, size_t);
extern int mynah_cuda_sync(void *, char *, size_t);
extern int mynah_cuda_snake_dev(void *, float *, const float *, size_t, size_t, size_t, char *, size_t);
extern int mynah_cuda_gelu_dev(void *, float *, size_t, char *, size_t);
extern int mynah_cuda_layer_norm_dev(void *, const float *, float *, const float *, const float *, size_t, size_t, char *, size_t);
extern int mynah_cuda_residual_add_dev(void *, float *, const float *, size_t, char *, size_t);
extern int mynah_cuda_matmul_dev(void *, const float *, float *, size_t, size_t, size_t, const float *, const float *, char *, size_t);
extern int mynah_cuda_sgemm_dev(void *, int, int, size_t, size_t, size_t, float, const float *, size_t, const float *, size_t, float, float *, size_t, char *, size_t);
extern int mynah_cuda_dev_alloc(void *, size_t, float **, char *, size_t);
extern void mynah_cuda_dev_free(void *, float *);
extern int mynah_cuda_h2d(void *, const float *, float *, size_t, char *, size_t);
extern int mynah_cuda_d2h(void *, const float *, float *, size_t, char *, size_t);
extern int mynah_cuda_matvec_dev(void *, const float *, float *, size_t, size_t, const float *, const float *, char *, size_t);
extern int mynah_cuda_gelu_inplace(void *, float *, size_t, char *, size_t);
extern int mynah_cuda_residual_inplace(void *, float *, const float *, size_t, char *, size_t);
extern int mynah_cuda_layer_norm_inplace(void *, const float *, float *, const float *, size_t, size_t, char *, size_t);
extern int mynah_cuda_matmul_to_dev(void *, const float *, float *, size_t, size_t, size_t, const float *, const float *, char *, size_t);
extern int mynah_cuda_matmul_d2d(void *, const float *, float *, size_t, size_t, size_t, const float *, const float *, char *, size_t);
extern int mynah_cuda_im2col(void *, const float *, float *, int, int, int, int, char *, size_t);
extern int mynah_cuda_conv1d(void *, const float *, float *, int, int, int, int, int, const float *, const float *, char *, size_t);
extern int mynah_cuda_gelu_host(void *, float *, size_t, char *, size_t);
extern int mynah_cuda_gelu_host_f64(void *, float *, size_t, char *, size_t);
extern int mynah_cuda_matmul_graph(void *, const float *, float *, size_t, size_t, size_t, const float *, const float *, char *, size_t);
#endif

static void set_error(char *error, size_t capacity, const char *message) {
    if (error != NULL && capacity > 0) snprintf(error, capacity, "%s", message);
}

static float dot_product(const float *left, const float *right, size_t count) {
    size_t i = 0;
    float sum = 0.0f;
#if defined(__AVX2__)
    __m256 accumulator = _mm256_setzero_ps();
    for (; i + 8u <= count; i += 8u) {
        accumulator = _mm256_add_ps(accumulator,
                                    _mm256_mul_ps(_mm256_loadu_ps(left + i),
                                                  _mm256_loadu_ps(right + i)));
    }
    float lanes[8];
    _mm256_storeu_ps(lanes, accumulator);
    for (size_t lane = 0; lane < 8u; ++lane) sum += lanes[lane];
#elif defined(__ARM_NEON)
    float32x4_t accumulator = vdupq_n_f32(0.0f);
    for (; i + 4u <= count; i += 4u) {
        accumulator = vmlaq_f32(accumulator, vld1q_f32(left + i), vld1q_f32(right + i));
    }
    sum += vaddvq_f32(accumulator);
#endif
    for (; i < count; ++i) sum += left[i] * right[i];
    return sum;
}

#if defined(MYNAH_USE_ACCELERATE) || defined(MYNAH_USE_OPENBLAS)
/* out[rows, output_width] = input[rows, input_width] @ weight[output_width,
 * input_width]^T (+ bias).  A multi-row call (prefill/encoder) is split across
 * worker threads over disjoint row blocks — each block is one sgemm, so the
 * result is bit-identical to the single-call form.  A single-row decode step
 * has one block and runs inline. */
typedef struct {
    const float *input;
    float *output;
    const float *weight;
    const float *bias;
    size_t rows;
    size_t input_width;
    size_t output_width;
    int blocks;
} matmul_job;

static void matmul_block(void *ctx, int b) {
    const matmul_job *j = (const matmul_job *)ctx;
    const size_t per = (j->rows + (size_t)j->blocks - 1u) / (size_t)j->blocks;
    const size_t r0 = (size_t)b * per;
    if (r0 >= j->rows) return;
    size_t r1 = r0 + per;
    if (r1 > j->rows) r1 = j->rows;
    const size_t m = r1 - r0;
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                (int)m, (int)j->output_width, (int)j->input_width,
                1.0f, j->input + r0 * j->input_width, (int)j->input_width,
                j->weight, (int)j->input_width,
                0.0f, j->output + r0 * j->output_width, (int)j->output_width);
    if (j->bias != NULL) {
        for (size_t row = r0; row < r1; ++row) {
            for (size_t column = 0; column < j->output_width; ++column) {
                j->output[row * j->output_width + column] += j->bias[column];
            }
        }
    }
}
#endif

static int cpu_matmul(void *state, const float *input, float *output, size_t rows,
                      size_t input_width, size_t output_width, const float *weight,
                      const float *bias, char *error, size_t error_capacity) {
    (void)state;
    (void)error;
    (void)error_capacity;
#if defined(MYNAH_USE_ACCELERATE) || defined(MYNAH_USE_OPENBLAS)
    if (rows <= (size_t)INT_MAX && input_width <= (size_t)INT_MAX &&
        output_width <= (size_t)INT_MAX) {
        int blocks = 1;
#if defined(MYNAH_USE_OPENBLAS)
        /* Accelerate (macOS) threads sgemm internally, so splitting there only
         * oversubscribes; on OpenBLAS we drive the parallelism ourselves (the
         * pool forces BLAS to one thread per block).  Thread only matmuls big
         * enough to amortize dispatch. */
        const int threads = mynah_num_threads();
        if (threads > 1 && rows >= 8u && input_width * output_width >= 65536u) {
            blocks = (int)rows < threads ? (int)rows : threads;
        }
#endif
        matmul_job job = {input, output, weight, bias, rows, input_width, output_width, blocks};
        mynah_parallel_for(blocks, matmul_block, &job);
        return 0;
    }
#endif
    for (size_t row = 0; row < rows; ++row) {
        const float *input_row = input + row * input_width;
        float *output_row = output + row * output_width;
        for (size_t column = 0; column < output_width; ++column) {
            output_row[column] = (bias == NULL ? 0.0f : bias[column]) +
                                dot_product(input_row, weight + column * input_width,
                                            input_width);
        }
    }
    return 0;
}

static int cpu_self_test(void *state, char *error, size_t error_capacity) {
    (void)state;
    float input[6] = {1.0f, 2.0f, 3.0f, -1.0f, 0.5f, 2.0f};
    float weight[12] = {1.0f, 0.0f, 0.0f,
                        0.0f, 1.0f, 0.0f,
                        0.0f, 0.0f, 1.0f,
                        1.0f, 1.0f, 1.0f};
    float bias[4] = {0.5f, -0.5f, 1.0f, 2.0f};
    float output[8] = {0};
    if (cpu_matmul(NULL, input, output, 2u, 3u, 4u, weight, bias, error, error_capacity) != 0) {
        return -1;
    }
    const float expected[8] = {1.5f, 1.5f, 4.0f, 8.0f, -0.5f, 0.0f, 3.0f, 3.5f};
    for (size_t i = 0; i < 8u; ++i) {
        if (fabsf(output[i] - expected[i]) > 1.0e-5f) {
            snprintf(error, error_capacity, "CPU backend self-test mismatch at %zu", i);
            return -1;
        }
    }
    return 0;
}

/* Generic sgemm: C[m,n] = alpha * op(A) * op(B) + beta * C  (row-major).
 * On BLAS builds this delegates to cblas_sgemm; otherwise a scalar
 * triple loop. */
static int cpu_sgemm(void *state, int trans_a, int trans_b,
                     size_t m, size_t n, size_t k,
                     float alpha,
                     const float *a, size_t lda,
                     const float *b, size_t ldb,
                     float beta,
                     float *c, size_t ldc,
                     char *error, size_t error_capacity) {
    (void)state;
    (void)error;
    (void)error_capacity;
#if defined(MYNAH_USE_ACCELERATE) || defined(MYNAH_USE_OPENBLAS)
    if (m <= (size_t)INT_MAX && n <= (size_t)INT_MAX && k <= (size_t)INT_MAX &&
        lda <= (size_t)INT_MAX && ldb <= (size_t)INT_MAX && ldc <= (size_t)INT_MAX) {
        cblas_sgemm(CblasRowMajor,
                    trans_a ? CblasTrans : CblasNoTrans,
                    trans_b ? CblasTrans : CblasNoTrans,
                    (int)m, (int)n, (int)k,
                    alpha, a, (int)lda, b, (int)ldb,
                    beta, c, (int)ldc);
        return 0;
    }
#endif
    /* Scalar fallback. */
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (size_t p = 0; p < k; ++p) {
                const float a_val = trans_a ? a[p * lda + i] : a[i * lda + p];
                const float b_val = trans_b ? b[j * ldb + p] : b[p * ldb + j];
                sum += a_val * b_val;
            }
            c[i * ldc + j] = alpha * sum + beta * c[i * ldc + j];
        }
    }
    return 0;
}

const char *mynah_tts_device_name(mynah_tts_device device) {
    switch (device) {
        case MYNAH_TTS_DEVICE_CPU: return "cpu";
        case MYNAH_TTS_DEVICE_METAL: return "metal";
        case MYNAH_TTS_DEVICE_CUDA: return "cuda";
        default: return "unknown";
    }
}

int mynah_backend_open(mynah_tts_device device, mynah_backend **out,
                       char *error, size_t error_capacity) {
    if (out != NULL) *out = NULL;
    if (out == NULL) {
        set_error(error, error_capacity, "invalid backend output pointer");
        return -1;
    }
    if (device != MYNAH_TTS_DEVICE_CPU && device != MYNAH_TTS_DEVICE_METAL &&
        device != MYNAH_TTS_DEVICE_CUDA) {
        set_error(error, error_capacity, "unknown backend device");
        return -1;
    }
    mynah_backend *backend = (mynah_backend *)calloc(1, sizeof(*backend));
    if (backend == NULL) {
        set_error(error, error_capacity, "out of memory creating backend");
        return -1;
    }
    backend->device = device;
    backend->name = mynah_tts_device_name(device);
    backend->matmul = cpu_matmul;
    backend->sgemm = cpu_sgemm;
    backend->self_test = cpu_self_test;
    if (device == MYNAH_TTS_DEVICE_METAL) {
#if defined(MYNAH_ENABLE_METAL)
        if (mynah_backend_metal_open(&backend->state, &backend->matmul, &backend->sgemm,
                                     &backend->close, &backend->self_test,
                                     error, error_capacity) != 0) {
            free(backend);
            return -1;
        }
        if (backend->sgemm == NULL) backend->sgemm = cpu_sgemm;
#else
        free(backend);
        set_error(error, error_capacity, "Metal backend is not compiled; use make metal");
        return -1;
#endif
    } else if (device == MYNAH_TTS_DEVICE_CUDA) {
#if defined(MYNAH_ENABLE_CUDA)
        if (mynah_backend_cuda_open(&backend->state, &backend->matmul, &backend->sgemm,
                                    &backend->close, &backend->self_test,
                                    error, error_capacity) != 0) {
            free(backend);
            return -1;
        }
        backend->upload = mynah_cuda_upload;
        backend->download = mynah_cuda_download;
        backend->sync = mynah_cuda_sync;
        backend->snake_dev = mynah_cuda_snake_dev;
        backend->gelu_dev = mynah_cuda_gelu_dev;
        backend->layer_norm_dev = mynah_cuda_layer_norm_dev;
        backend->residual_add_dev = mynah_cuda_residual_add_dev;
        backend->matmul_dev = mynah_cuda_matmul_dev;
        backend->sgemm_dev = mynah_cuda_sgemm_dev;
        backend->dev_alloc = mynah_cuda_dev_alloc;
        backend->dev_free = mynah_cuda_dev_free;
        backend->h2d = mynah_cuda_h2d;
        backend->d2h = mynah_cuda_d2h;
        backend->matvec_dev = mynah_cuda_matvec_dev;
        backend->gelu_inplace = mynah_cuda_gelu_inplace;
        backend->residual_inplace = mynah_cuda_residual_inplace;
        backend->layer_norm_inplace = mynah_cuda_layer_norm_inplace;
        backend->matmul_to_dev = mynah_cuda_matmul_to_dev;
        backend->matmul_d2d = mynah_cuda_matmul_d2d;
        backend->im2col = mynah_cuda_im2col;
        backend->conv1d = mynah_cuda_conv1d;
        backend->gelu_host = mynah_cuda_gelu_host;
        backend->gelu_host_f64 = mynah_cuda_gelu_host_f64;
        backend->matmul_graph = mynah_cuda_matmul_graph;
#else
        free(backend);
        set_error(error, error_capacity, "CUDA backend is not compiled; use make cuda");
        return -1;
#endif
    }
    if (error != NULL && error_capacity > 0) error[0] = '\0';
    *out = backend;
    return 0;
}

void mynah_backend_close(mynah_backend *backend) {
    if (backend == NULL) return;
    if (backend->close != NULL) backend->close(backend->state);
    free(backend);
}

const char *mynah_backend_name(const mynah_backend *backend) {
    return backend == NULL ? "unknown" : backend->name;
}

int mynah_backend_matmul(const mynah_backend *backend, const float *input,
                         float *output, size_t rows, size_t input_width,
                         size_t output_width, const float *weight,
                         const float *bias, char *error, size_t error_capacity) {
    if (backend == NULL || backend->matmul == NULL || input == NULL || output == NULL ||
        weight == NULL || rows == 0 || input_width == 0 || output_width == 0) {
        set_error(error, error_capacity, "invalid backend matmul request");
        return -1;
    }
    return backend->matmul(backend->state, input, output, rows, input_width, output_width,
                           weight, bias, error, error_capacity);
}

int mynah_backend_sgemm(const mynah_backend *backend,
                        int trans_a, int trans_b,
                        size_t m, size_t n, size_t k,
                        float alpha,
                        const float *a, size_t lda,
                        const float *b, size_t ldb,
                        float beta,
                        float *c, size_t ldc,
                        char *error, size_t error_capacity) {
    if (backend == NULL || backend->sgemm == NULL || a == NULL || b == NULL || c == NULL ||
        m == 0 || n == 0 || k == 0) {
        set_error(error, error_capacity, "invalid backend sgemm request");
        return -1;
    }
    return backend->sgemm(backend->state, trans_a, trans_b, m, n, k,
                          alpha, a, lda, b, ldb, beta, c, ldc,
                          error, error_capacity);
}

int mynah_backend_self_test(mynah_tts_device device, char *error, size_t error_capacity) {
    mynah_backend *backend = NULL;
    if (mynah_backend_open(device, &backend, error, error_capacity) != 0) return -1;
    const int result = backend->self_test == NULL ? 0 :
        backend->self_test(backend->state, error, error_capacity);
    mynah_backend_close(backend);
    return result;
}

/* ---- Device-side operations ----
 * CPU fallback: "device" pointers are host pointers; upload/download/sync
 * are no-ops or memcpy.  CUDA overrides these in backend_cuda.cu. */

int mynah_backend_upload(const mynah_backend *backend, const float *host,
                         size_t n, float **dev_ptr,
                         char *error, size_t error_capacity) {
    if (backend == NULL || host == NULL || dev_ptr == NULL) return -1;
    if (backend->upload != NULL)
        return backend->upload(backend->state, host, n, dev_ptr, error, error_capacity);
    *dev_ptr = (float *)host; /* CPU: same pointer */
    return 0;
}

int mynah_backend_download(const mynah_backend *backend, const float *dev_ptr,
                           float *host, size_t n,
                           char *error, size_t error_capacity) {
    if (backend == NULL || dev_ptr == NULL || host == NULL) return -1;
    if (backend->download != NULL)
        return backend->download(backend->state, dev_ptr, host, n, error, error_capacity);
    if (dev_ptr != host) memcpy(host, dev_ptr, n * sizeof(float));
    return 0;
}

int mynah_backend_sync(const mynah_backend *backend,
                       char *error, size_t error_capacity) {
    if (backend == NULL) return -1;
    if (backend->sync != NULL)
        return backend->sync(backend->state, error, error_capacity);
    return 0; /* CPU: nothing to sync */
}

int mynah_backend_gelu_dev(const mynah_backend *backend,
                           float *dev_data, size_t n,
                           char *error, size_t error_capacity) {
    if (backend == NULL) return -1;
    if (backend->gelu_dev != NULL)
        return backend->gelu_dev(backend->state, dev_data, n, error, error_capacity);
    for (size_t i = 0; i < n; ++i) {
        float x = dev_data[i];
        float c = 0.7978845608f * (x + 0.044715f * x * x * x);
        dev_data[i] = 0.5f * x * (1.0f + tanhf(c));
    }
    return 0;
}

int mynah_backend_layer_norm_dev(const mynah_backend *backend,
                                 const float *dev_in, float *dev_out,
                                 const float *gain, const float *bias,
                                 size_t rows, size_t width,
                                 char *error, size_t error_capacity) {
    if (backend == NULL) return -1;
    if (backend->layer_norm_dev != NULL)
        return backend->layer_norm_dev(backend->state, dev_in, dev_out, gain, bias,
                                       rows, width, error, error_capacity);
    for (size_t r = 0; r < rows; ++r) {
        const float *x = dev_in + r * width;
        float *y = dev_out + r * width;
        float mean = 0.0f;
        for (size_t i = 0; i < width; ++i) mean += x[i];
        mean /= (float)width;
        float var = 0.0f;
        for (size_t i = 0; i < width; ++i) { float d = x[i] - mean; var += d * d; }
        float inv = 1.0f / sqrtf(var / (float)width + 1e-5f);
        for (size_t i = 0; i < width; ++i)
            y[i] = (x[i] - mean) * inv * gain[i] + bias[i];
    }
    return 0;
}

int mynah_backend_softmax_dev(const mynah_backend *backend,
                              float *dev_data, size_t rows, size_t cols,
                              size_t valid,
                              char *error, size_t error_capacity) {
    (void)backend; (void)error; (void)error_capacity;
    for (size_t r = 0; r < rows; ++r) {
        float *row = dev_data + r * cols;
        size_t v = valid < cols ? valid : cols;
        float mx = -1e30f;
        for (size_t i = 0; i < v; ++i) mx = fmaxf(mx, row[i]);
        float sum = 0.0f;
        for (size_t i = 0; i < v; ++i) { row[i] = expf(row[i] - mx); sum += row[i]; }
        float inv = 1.0f / sum;
        for (size_t i = 0; i < v; ++i) row[i] *= inv;
        for (size_t i = v; i < cols; ++i) row[i] = 0.0f;
    }
    return 0;
}

int mynah_backend_residual_add_dev(const mynah_backend *backend,
                                   float *dev_out, const float *dev_in,
                                   size_t n,
                                   char *error, size_t error_capacity) {
    if (backend == NULL) return -1;
    if (backend->residual_add_dev != NULL)
        return backend->residual_add_dev(backend->state, dev_out, dev_in, n,
                                         error, error_capacity);
    for (size_t i = 0; i < n; ++i) dev_out[i] += dev_in[i];
    return 0;
}

int mynah_backend_snake_dev(const mynah_backend *backend,
                            float *dev_data, const float *alpha,
                            size_t channels, size_t length,
                            size_t snake_channels,
                            char *error, size_t error_capacity) {
    if (backend == NULL) return -1;
    if (backend->snake_dev != NULL)
        return backend->snake_dev(backend->state, dev_data, alpha,
                                  channels, length, snake_channels,
                                  error, error_capacity);
    /* CPU fallback. */
    for (size_t ch = 0; ch < channels; ++ch) {
        float *row = dev_data + ch * length;
        if (ch < snake_channels) {
            float a = alpha[ch];
            float inv_a = 1.0f / (a + 1e-9f);
            for (size_t t = 0; t < length; ++t) {
                float v = row[t];
                float sn = sinf(a * v);
                row[t] = v + sn * sn * inv_a;
            }
        } else {
            for (size_t t = 0; t < length; ++t)
                if (row[t] < 0.0f) row[t] *= 0.01f;
        }
    }
    return 0;
}

int mynah_backend_matmul_dev(const mynah_backend *backend,
                             const float *dev_in, float *dev_out,
                             size_t rows, size_t iw, size_t ow,
                             const float *weight, const float *bias,
                             char *error, size_t error_capacity) {
    if (backend == NULL) return -1;
    if (backend->matmul_dev != NULL)
        return backend->matmul_dev(backend->state, dev_in, dev_out, rows, iw, ow,
                                   weight, bias, error, error_capacity);
    return mynah_backend_matmul(backend, dev_in, dev_out, rows, iw, ow,
                                weight, bias, error, error_capacity);
}

int mynah_backend_sgemm_dev(const mynah_backend *backend,
                            int trans_a, int trans_b,
                            size_t m, size_t n, size_t k,
                            float alpha,
                            const float *dev_a, size_t lda,
                            const float *dev_b, size_t ldb,
                            float beta,
                            float *dev_c, size_t ldc,
                            char *error, size_t error_capacity) {
    if (backend == NULL) return -1;
    if (backend->sgemm_dev != NULL)
        return backend->sgemm_dev(backend->state, trans_a, trans_b, m, n, k,
                                  alpha, dev_a, lda, dev_b, ldb,
                                  beta, dev_c, ldc, error, error_capacity);
    return mynah_backend_sgemm(backend, trans_a, trans_b, m, n, k,
                               alpha, dev_a, lda, dev_b, ldb,
                               beta, dev_c, ldc, error, error_capacity);
}

int mynah_backend_dev_alloc(const mynah_backend *backend, size_t n,
                            float **dev_ptr, char *error, size_t error_capacity) {
    if (backend == NULL || dev_ptr == NULL) return -1;
    if (backend->dev_alloc != NULL)
        return backend->dev_alloc(backend->state, n, dev_ptr, error, error_capacity);
    *dev_ptr = (float *)malloc(n * sizeof(float));
    return *dev_ptr == NULL ? -1 : 0;
}

void mynah_backend_dev_free(const mynah_backend *backend, float *dev_ptr) {
    if (backend == NULL || dev_ptr == NULL) return;
    if (backend->dev_free != NULL) { backend->dev_free(backend->state, dev_ptr); return; }
    free(dev_ptr);
}

int mynah_backend_has_dev_ops(const mynah_backend *backend) {
    return backend != NULL && backend->matmul_dev != NULL;
}

int mynah_backend_h2d(const mynah_backend *backend, const float *host,
                      float *dev_ptr, size_t n,
                      char *error, size_t error_capacity) {
    if (backend == NULL) return -1;
    if (backend->h2d != NULL)
        return backend->h2d(backend->state, host, dev_ptr, n, error, error_capacity);
    memcpy(dev_ptr, host, n * sizeof(float));
    return 0;
}

int mynah_backend_d2h(const mynah_backend *backend, const float *dev_ptr,
                      float *host, size_t n,
                      char *error, size_t error_capacity) {
    if (backend == NULL) return -1;
    if (backend->d2h != NULL)
        return backend->d2h(backend->state, dev_ptr, host, n, error, error_capacity);
    memcpy(host, dev_ptr, n * sizeof(float));
    return 0;
}

int mynah_backend_matvec_dev(const mynah_backend *backend,
                             const float *dev_in, float *dev_out,
                             size_t K, size_t N,
                             const float *weight, const float *bias,
                             char *error, size_t error_capacity) {
    if (backend == NULL) return -1;
    if (backend->matvec_dev != NULL)
        return backend->matvec_dev(backend->state, dev_in, dev_out, K, N,
                                   weight, bias, error, error_capacity);
    /* CPU fallback */
    return mynah_backend_matmul(backend, dev_in, dev_out, 1u, K, N,
                                weight, bias, error, error_capacity);
}

int mynah_backend_gelu_inplace(const mynah_backend *bk, float *dev, size_t n,
                               char *e, size_t ec) {
    if (bk && bk->gelu_inplace) return bk->gelu_inplace(bk->state, dev, n, e, ec);
    return -1;
}
int mynah_backend_residual_inplace(const mynah_backend *bk, float *dev_out,
                                   const float *dev_in, size_t n,
                                   char *e, size_t ec) {
    if (bk && bk->residual_inplace) return bk->residual_inplace(bk->state, dev_out, dev_in, n, e, ec);
    return -1;
}
int mynah_backend_layer_norm_inplace(const mynah_backend *bk, const float *dev_in,
                                     float *dev_out, const float *gain,
                                     size_t rows, size_t width,
                                     char *e, size_t ec) {
    if (bk && bk->layer_norm_inplace) return bk->layer_norm_inplace(bk->state, dev_in, dev_out, gain, rows, width, e, ec);
    return -1;
}

int mynah_backend_matmul_to_dev(const mynah_backend *bk, const float *in, float *dout,
                                size_t rows, size_t iw, size_t ow,
                                const float *w, const float *b,
                                char *e, size_t ec) {
    if (bk && bk->matmul_to_dev)
        return bk->matmul_to_dev(bk->state, in, dout, rows, iw, ow, w, b, e, ec);
    return -1;
}

int mynah_backend_matmul_d2d(const mynah_backend *bk, const float *din, float *dout,
                             size_t rows, size_t iw, size_t ow,
                             const float *w, const float *b,
                             char *e, size_t ec) {
    if (bk && bk->matmul_d2d)
        return bk->matmul_d2d(bk->state, din, dout, rows, iw, ow, w, b, e, ec);
    return -1;
}

int mynah_backend_im2col(const mynah_backend *bk, const float *input, float *columns,
                         int in_ch, int length, int kernel, int dilation,
                         char *e, size_t ec) {
    if (bk && bk->im2col)
        return bk->im2col(bk->state, input, columns, in_ch, length, kernel, dilation, e, ec);
    return -1;
}

int mynah_backend_conv1d(const mynah_backend *bk,
                         const float *input, float *output,
                         int in_ch, int out_ch, int length,
                         int kernel, int dilation,
                         const float *weight, const float *bias,
                         char *e, size_t ec) {
    if (bk && bk->conv1d)
        return bk->conv1d(bk->state, input, output, in_ch, out_ch, length,
                          kernel, dilation, weight, bias, e, ec);
    return -1; /* no GPU conv1d — caller falls back to CPU */
}

int mynah_backend_gelu_host(const mynah_backend *bk, float *data, size_t n,
                            char *e, size_t ec) {
    if (bk && bk->gelu_host) return bk->gelu_host(bk->state, data, n, e, ec);
    return -1;
}

int mynah_backend_gelu_host_f64(const mynah_backend *bk, float *data, size_t n,
                                char *e, size_t ec) {
    if (bk && bk->gelu_host_f64) return bk->gelu_host_f64(bk->state, data, n, e, ec);
    return -1;
}

int mynah_backend_matmul_graph(const mynah_backend *bk,
                               const float *in, float *out,
                               size_t rows, size_t iw, size_t ow,
                               const float *w, const float *b,
                               char *e, size_t ec) {
    if (bk && bk->matmul_graph)
        return bk->matmul_graph(bk->state, in, out, rows, iw, ow, w, b, e, ec);
    return mynah_backend_matmul(bk, in, out, rows, iw, ow, w, b, e, ec);
}
