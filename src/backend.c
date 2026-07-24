#include "backend.h"
#include "threads.h"

#include <errno.h>
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
    mynah_backend_close_fn close;
    mynah_backend_self_test_fn self_test;
};

#if defined(MYNAH_ENABLE_METAL)
int mynah_backend_metal_open(void **state, mynah_backend_matmul_fn *matmul,
                             mynah_backend_close_fn *close, mynah_backend_self_test_fn *self_test,
                             char *error, size_t error_capacity);
#endif
#if defined(MYNAH_ENABLE_CUDA)
int mynah_backend_cuda_open(void **state, mynah_backend_matmul_fn *matmul,
                            mynah_backend_close_fn *close, mynah_backend_self_test_fn *self_test,
                            char *error, size_t error_capacity);
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
    backend->self_test = cpu_self_test;
    if (device == MYNAH_TTS_DEVICE_METAL) {
#if defined(MYNAH_ENABLE_METAL)
        if (mynah_backend_metal_open(&backend->state, &backend->matmul, &backend->close,
                                     &backend->self_test, error, error_capacity) != 0) {
            free(backend);
            return -1;
        }
#else
        free(backend);
        set_error(error, error_capacity, "Metal backend is not compiled; use make metal");
        return -1;
#endif
    } else if (device == MYNAH_TTS_DEVICE_CUDA) {
#if defined(MYNAH_ENABLE_CUDA)
        if (mynah_backend_cuda_open(&backend->state, &backend->matmul, &backend->close,
                                    &backend->self_test, error, error_capacity) != 0) {
            free(backend);
            return -1;
        }
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

int mynah_backend_self_test(mynah_tts_device device, char *error, size_t error_capacity) {
    mynah_backend *backend = NULL;
    if (mynah_backend_open(device, &backend, error, error_capacity) != 0) return -1;
    const int result = backend->self_test == NULL ? 0 :
        backend->self_test(backend->state, error, error_capacity);
    mynah_backend_close(backend);
    return result;
}
