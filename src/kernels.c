#include "kernels.h"

#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>

#if !defined(MYNAH_DISABLE_SIMD) && (defined(__ARM_NEON) || defined(__aarch64__))
#include <arm_neon.h>
#define MYNAH_KERNELS_NEON 1
#elif !defined(MYNAH_DISABLE_SIMD) && defined(__AVX2__)
#include <immintrin.h>
#define MYNAH_KERNELS_AVX2 1
#endif

float mynah_dot_f32(const float *a, const float *b, size_t n) {
#if defined(MYNAH_KERNELS_NEON)
    float32x4_t accumulator = vdupq_n_f32(0.0f);
    size_t i = 0;
    for (; i + 4u <= n; i += 4u) {
        accumulator = vmlaq_f32(accumulator, vld1q_f32(a + i), vld1q_f32(b + i));
    }
    float sum = vaddvq_f32(accumulator);
    for (; i < n; ++i) sum += a[i] * b[i];
    return sum;
#elif defined(MYNAH_KERNELS_AVX2)
    __m256 accumulator = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8u <= n; i += 8u) {
        accumulator = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), accumulator);
    }
    float partial[8];
    _mm256_storeu_ps(partial, accumulator);
    float sum = partial[0] + partial[1] + partial[2] + partial[3] +
                partial[4] + partial[5] + partial[6] + partial[7];
    for (; i < n; ++i) sum += a[i] * b[i];
    return sum;
#else
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
#endif
}

void mynah_matvec_f32(const float *weights, const float *input, float *output,
                      size_t rows, size_t cols) {
    for (size_t row = 0; row < rows; ++row) {
        output[row] = mynah_dot_f32(weights + row * cols, input, cols);
    }
}

void mynah_rmsnorm_f32(const float *input, const float *weight, float *output,
                       size_t n, float epsilon) {
    float mean_square = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        mean_square += input[i] * input[i];
    }
    mean_square /= (float)n;
    const float scale = 1.0f / sqrtf(mean_square + epsilon);
    for (size_t i = 0; i < n; ++i) {
        output[i] = input[i] * scale * weight[i];
    }
}

int mynah_softmax_f32(const float *logits, float *probabilities, size_t n) {
    if (n == 0) {
        return -1;
    }
    float maximum = logits[0];
    if (!isfinite(maximum)) {
        return -1;
    }
    for (size_t i = 1; i < n; ++i) {
        if (!isfinite(logits[i])) {
            return -1;
        }
        if (logits[i] > maximum) {
            maximum = logits[i];
        }
    }
    float total = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        probabilities[i] = expf(logits[i] - maximum);
        total += probabilities[i];
    }
    if (!(total > 0.0f) || !isfinite(total)) {
        return -1;
    }
    const float inverse_total = 1.0f / total;
    for (size_t i = 0; i < n; ++i) {
        probabilities[i] *= inverse_total;
    }
    return 0;
}

size_t mynah_argmax_f32(const float *values, size_t n) {
    if (n == 0) {
        return SIZE_MAX;
    }
    size_t best = 0;
    for (size_t i = 1; i < n; ++i) {
        if (values[i] > values[best]) {
            best = i;
        }
    }
    return best;
}

static int close_enough(float actual, float expected, float tolerance) {
    return fabsf(actual - expected) <= tolerance;
}

int mynah_kernels_self_test(char *error, size_t error_capacity) {
    const float a[] = {1.0f, 2.0f, 3.0f};
    const float b[] = {4.0f, -2.0f, 0.5f};
    if (!close_enough(mynah_dot_f32(a, b, 3), 1.5f, 1e-6f)) {
        snprintf(error, error_capacity, "dot product mismatch");
        return -1;
    }

    const float matrix[] = {1.0f, 2.0f, 3.0f, -1.0f, 0.0f, 2.0f};
    float matvec[2] = {0.0f, 0.0f};
    mynah_matvec_f32(matrix, a, matvec, 2, 3);
    if (!close_enough(matvec[0], 14.0f, 1e-6f) ||
        !close_enough(matvec[1], 5.0f, 1e-6f)) {
        snprintf(error, error_capacity, "matvec mismatch");
        return -1;
    }

    const float norm_input[] = {3.0f, 4.0f};
    const float norm_weight[] = {1.0f, 2.0f};
    float norm_output[2] = {0.0f, 0.0f};
    mynah_rmsnorm_f32(norm_input, norm_weight, norm_output, 2, 1e-6f);
    if (!close_enough(norm_output[0], 0.848528f, 1e-5f) ||
        !close_enough(norm_output[1], 2.262741f, 1e-5f)) {
        snprintf(error, error_capacity, "rmsnorm mismatch");
        return -1;
    }

    const float logits[] = {1.0f, 2.0f, 3.0f};
    float probabilities[3] = {0.0f, 0.0f, 0.0f};
    if (mynah_softmax_f32(logits, probabilities, 3) != 0 ||
        mynah_argmax_f32(probabilities, 3) != 2 ||
        !close_enough(probabilities[0] + probabilities[1] + probabilities[2],
                      1.0f, 1e-6f)) {
        snprintf(error, error_capacity, "softmax mismatch");
        return -1;
    }

    if (mynah_softmax_f32(logits, probabilities, 0) == 0) {
        snprintf(error, error_capacity, "empty softmax accepted");
        return -1;
    }
    error[0] = '\0';
    return 0;
}
