#include "kernels.h"

#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

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
    size_t row = 0;
#if defined(MYNAH_KERNELS_NEON)
    for (; row + 1u < rows; row += 2u) {
        const float *w0 = weights + row * cols;
        const float *w1 = w0 + cols;
        float32x4_t a0 = vdupq_n_f32(0.0f);
        float32x4_t a1 = vdupq_n_f32(0.0f);
        size_t i = 0;
        for (; i + 4u <= cols; i += 4u) {
            float32x4_t x = vld1q_f32(input + i);
            a0 = vmlaq_f32(a0, vld1q_f32(w0 + i), x);
            a1 = vmlaq_f32(a1, vld1q_f32(w1 + i), x);
        }
        float s0 = vaddvq_f32(a0);
        float s1 = vaddvq_f32(a1);
        for (; i < cols; ++i) {
            s0 += w0[i] * input[i];
            s1 += w1[i] * input[i];
        }
        output[row] = s0;
        output[row + 1u] = s1;
    }
#elif defined(MYNAH_KERNELS_AVX2)
    for (; row + 1u < rows; row += 2u) {
        const float *w0 = weights + row * cols;
        const float *w1 = w0 + cols;
        __m256 a0 = _mm256_setzero_ps();
        __m256 a1 = _mm256_setzero_ps();
        size_t i = 0;
        for (; i + 8u <= cols; i += 8u) {
            const __m256 x = _mm256_loadu_ps(input + i);
            a0 = _mm256_fmadd_ps(_mm256_loadu_ps(w0 + i), x, a0);
            a1 = _mm256_fmadd_ps(_mm256_loadu_ps(w1 + i), x, a1);
        }
        float lanes0[8], lanes1[8];
        _mm256_storeu_ps(lanes0, a0);
        _mm256_storeu_ps(lanes1, a1);
        float s0 = 0.0f, s1 = 0.0f;
        for (size_t lane = 0; lane < 8u; ++lane) {
            s0 += lanes0[lane];
            s1 += lanes1[lane];
        }
        for (; i < cols; ++i) {
            s0 += w0[i] * input[i];
            s1 += w1[i] * input[i];
        }
        output[row] = s0;
        output[row + 1u] = s1;
    }
#endif
    for (; row < rows; ++row) {
        output[row] = mynah_dot_f32(weights + row * cols, input, cols);
    }
}

void mynah_matvec_bias_f32(const float *weights, const float *input,
                           const float *bias, float *output, size_t rows,
                           size_t cols) {
    mynah_matvec_f32(weights, input, output, rows, cols);
    if (bias != NULL) {
        for (size_t row = 0; row < rows; ++row) {
            output[row] += bias[row];
        }
    }
}

int mynah_matvec_argmax_f32(const float *weights, const float *input,
                            const float *bias, size_t rows, size_t cols,
                            size_t allowed_rows, size_t extra_row,
                            int allow_extra, unsigned *argmax) {
    if (weights == NULL || input == NULL || argmax == NULL || rows == 0 ||
        allowed_rows > rows || (allow_extra && extra_row >= rows)) {
        return -1;
    }
    size_t best_row = 0;
    float best_value = -INFINITY;
    for (size_t row = 0; row < rows; ++row) {
        const int allowed = row < allowed_rows ||
                            (allow_extra && row == extra_row);
        if (!allowed) continue;
        float value = mynah_dot_f32(weights + row * cols, input, cols);
        if (bias != NULL) value += bias[row];
        if (value > best_value) {
            best_value = value;
            best_row = row;
        }
    }
    *argmax = (unsigned)best_row;
    return 0;
}

void mynah_rmsnorm_f32(const float *input, const float *weight, float *output,
                       size_t n, float epsilon) {
    float mean_square = 0.0f;
    size_t i = 0;
#if defined(MYNAH_KERNELS_NEON)
    float32x4_t sum = vdupq_n_f32(0.0f);
    for (; i + 4u <= n; i += 4u) {
        const float32x4_t x = vld1q_f32(input + i);
        sum = vmlaq_f32(sum, x, x);
    }
    mean_square = vaddvq_f32(sum);
#elif defined(MYNAH_KERNELS_AVX2)
    __m256 sum = _mm256_setzero_ps();
    for (; i + 8u <= n; i += 8u) {
        const __m256 x = _mm256_loadu_ps(input + i);
        sum = _mm256_fmadd_ps(x, x, sum);
    }
    float lanes[8];
    _mm256_storeu_ps(lanes, sum);
    for (size_t lane = 0; lane < 8u; ++lane) mean_square += lanes[lane];
#endif
    for (; i < n; ++i) mean_square += input[i] * input[i];
    mean_square /= (float)n;
    const float scale = 1.0f / sqrtf(mean_square + epsilon);
    i = 0;
#if defined(MYNAH_KERNELS_NEON)
    const float32x4_t vscale = vdupq_n_f32(scale);
    for (; i + 4u <= n; i += 4u) {
        const float32x4_t x = vld1q_f32(input + i);
        const float32x4_t w = vld1q_f32(weight + i);
        vst1q_f32(output + i, vmulq_f32(vmulq_f32(x, vscale), w));
    }
#elif defined(MYNAH_KERNELS_AVX2)
    const __m256 vscale = _mm256_set1_ps(scale);
    for (; i + 8u <= n; i += 8u) {
        const __m256 x = _mm256_loadu_ps(input + i);
        const __m256 w = _mm256_loadu_ps(weight + i);
        _mm256_storeu_ps(output + i, _mm256_mul_ps(_mm256_mul_ps(x, vscale), w));
    }
#endif
    for (; i < n; ++i) {
        output[i] = input[i] * scale * weight[i];
    }
}

void mynah_layernorm_f32(const float *input, const float *weight,
                         const float *bias, float *output, size_t rows,
                         size_t width, float epsilon) {
    for (size_t row = 0; row < rows; ++row) {
        const float *x = input + row * width;
        float *y = output + row * width;
        float mean = 0.0f;
        for (size_t i = 0; i < width; ++i) mean += x[i];
        mean /= (float)width;
        float variance = 0.0f;
        size_t i = 0;
#if defined(MYNAH_KERNELS_NEON)
        float32x4_t sum = vdupq_n_f32(0.0f);
        const float32x4_t vmean = vdupq_n_f32(mean);
        for (; i + 4u <= width; i += 4u) {
            const float32x4_t d = vsubq_f32(vld1q_f32(x + i), vmean);
            sum = vmlaq_f32(sum, d, d);
        }
        variance = vaddvq_f32(sum);
#elif defined(MYNAH_KERNELS_AVX2)
        __m256 sum = _mm256_setzero_ps();
        const __m256 vmean = _mm256_set1_ps(mean);
        for (; i + 8u <= width; i += 8u) {
            const __m256 d = _mm256_sub_ps(_mm256_loadu_ps(x + i), vmean);
            sum = _mm256_fmadd_ps(d, d, sum);
        }
        float lanes[8];
        _mm256_storeu_ps(lanes, sum);
        for (size_t lane = 0; lane < 8u; ++lane) variance += lanes[lane];
#endif
        for (; i < width; ++i) {
            const float d = x[i] - mean;
            variance += d * d;
        }
        const float scale = 1.0f / sqrtf(variance / (float)width + epsilon);
        i = 0;
#if defined(MYNAH_KERNELS_NEON)
        const float32x4_t vscale = vdupq_n_f32(scale);
        for (; i + 4u <= width; i += 4u) {
            const float32x4_t d = vsubq_f32(vld1q_f32(x + i), vmean);
            float32x4_t z = vmulq_f32(vmulq_f32(d, vscale), vld1q_f32(weight + i));
            if (bias != NULL) z = vaddq_f32(z, vld1q_f32(bias + i));
            vst1q_f32(y + i, z);
        }
#elif defined(MYNAH_KERNELS_AVX2)
        const __m256 vscale = _mm256_set1_ps(scale);
        for (; i + 8u <= width; i += 8u) {
            const __m256 d = _mm256_sub_ps(_mm256_loadu_ps(x + i), vmean);
            __m256 z = _mm256_mul_ps(_mm256_mul_ps(d, vscale), _mm256_loadu_ps(weight + i));
            if (bias != NULL) z = _mm256_add_ps(z, _mm256_loadu_ps(bias + i));
            _mm256_storeu_ps(y + i, z);
        }
#endif
        for (; i < width; ++i) {
            y[i] = (x[i] - mean) * scale * weight[i] +
                   (bias == NULL ? 0.0f : bias[i]);
        }
    }
}

void mynah_residual_add_f32(float *output, const float *input, size_t n) {
    size_t i = 0;
#if defined(MYNAH_KERNELS_NEON)
    for (; i + 4u <= n; i += 4u)
        vst1q_f32(output + i, vaddq_f32(vld1q_f32(output + i), vld1q_f32(input + i)));
#elif defined(MYNAH_KERNELS_AVX2)
    for (; i + 8u <= n; i += 8u)
        _mm256_storeu_ps(output + i, _mm256_add_ps(_mm256_loadu_ps(output + i),
                                                   _mm256_loadu_ps(input + i)));
#endif
    for (; i < n; ++i) output[i] += input[i];
}

void mynah_gelu_f32(float *data, size_t n) {
#if defined(MYNAH_KERNELS_AVX2)
    /* Padé [5/5] tanh is accurate to about 2e-8 on [-6, 6].
     * MYNAH_GELU_SCALAR keeps the libm reference path for rollback. */
    if (getenv("MYNAH_GELU_SCALAR") == NULL) {
        const __m256 half = _mm256_set1_ps(0.5f);
        const __m256 one = _mm256_set1_ps(1.0f);
        const __m256 scale = _mm256_set1_ps(0.7978845608f);
        const __m256 cubic_scale = _mm256_set1_ps(0.044715f);
        const __m256 p0 = _mm256_set1_ps(1.0f);
        const __m256 p1 = _mm256_set1_ps(1.0f / 7.0f);
        const __m256 p2 = _mm256_set1_ps(4.0f / 855.0f);
        const __m256 p3 = _mm256_set1_ps(1.0f / 20349.0f);
        const __m256 p4 = _mm256_set1_ps(1.0f / 6409935.0f);
        const __m256 p5 = _mm256_set1_ps(1.0f / 13749310575.0f);
        const __m256 q1 = _mm256_set1_ps(10.0f / 21.0f);
        const __m256 q2 = _mm256_set1_ps(4.0f / 133.0f);
        const __m256 q3 = _mm256_set1_ps(8.0f / 14535.0f);
        const __m256 q4 = _mm256_set1_ps(1.0f / 305235.0f);
        const __m256 q5 = _mm256_set1_ps(2.0f / 416645775.0f);
        const __m256 lower = _mm256_set1_ps(-6.0f);
        const __m256 upper = _mm256_set1_ps(6.0f);
        size_t i = 0;
        for (; i + 8u <= n; i += 8u) {
            const __m256 x = _mm256_loadu_ps(data + i);
            const __m256 x2 = _mm256_mul_ps(x, x);
            const __m256 x3 = _mm256_mul_ps(x2, x);
            const __m256 inner = _mm256_mul_ps(
                scale, _mm256_add_ps(x, _mm256_mul_ps(cubic_scale, x3)));
            const __m256 u = _mm256_min_ps(_mm256_max_ps(inner, lower), upper);
            const __m256 u2 = _mm256_mul_ps(u, u);
            __m256 numerator = p5;
            numerator = _mm256_fmadd_ps(numerator, u2, p4);
            numerator = _mm256_fmadd_ps(numerator, u2, p3);
            numerator = _mm256_fmadd_ps(numerator, u2, p2);
            numerator = _mm256_fmadd_ps(numerator, u2, p1);
            numerator = _mm256_fmadd_ps(numerator, u2, p0);
            __m256 denominator = q5;
            denominator = _mm256_fmadd_ps(denominator, u2, q4);
            denominator = _mm256_fmadd_ps(denominator, u2, q3);
            denominator = _mm256_fmadd_ps(denominator, u2, q2);
            denominator = _mm256_fmadd_ps(denominator, u2, q1);
            denominator = _mm256_add_ps(_mm256_mul_ps(denominator, u2), one);
            const __m256 tanh_u = _mm256_mul_ps(
                u, _mm256_div_ps(numerator, denominator));
            _mm256_storeu_ps(data + i, _mm256_mul_ps(
                _mm256_mul_ps(half, x), _mm256_add_ps(one, tanh_u)));
        }
        for (; i < n; ++i) {
            const float x = data[i];
            const float cubic = x * x * x;
            const float c = 0.7978845608f * (x + 0.044715f * cubic);
            data[i] = 0.5f * x * (1.0f + tanhf(c));
        }
        return;
    }
#endif
#if defined(MYNAH_KERNELS_NEON)
    if (getenv("MYNAH_GELU_SCALAR") == NULL) {
        const float32x4_t half = vdupq_n_f32(0.5f);
        const float32x4_t one = vdupq_n_f32(1.0f);
        const float32x4_t scale = vdupq_n_f32(0.7978845608f);
        const float32x4_t cubic_scale = vdupq_n_f32(0.044715f);
        const float32x4_t p0 = vdupq_n_f32(1.0f);
        const float32x4_t p1 = vdupq_n_f32(1.0f / 7.0f);
        const float32x4_t p2 = vdupq_n_f32(4.0f / 855.0f);
        const float32x4_t p3 = vdupq_n_f32(1.0f / 20349.0f);
        const float32x4_t p4 = vdupq_n_f32(1.0f / 6409935.0f);
        const float32x4_t p5 = vdupq_n_f32(1.0f / 13749310575.0f);
        const float32x4_t q1 = vdupq_n_f32(10.0f / 21.0f);
        const float32x4_t q2 = vdupq_n_f32(4.0f / 133.0f);
        const float32x4_t q3 = vdupq_n_f32(8.0f / 14535.0f);
        const float32x4_t q4 = vdupq_n_f32(1.0f / 305235.0f);
        const float32x4_t q5 = vdupq_n_f32(2.0f / 416645775.0f);
        const float32x4_t lower = vdupq_n_f32(-6.0f);
        const float32x4_t upper = vdupq_n_f32(6.0f);
        size_t i = 0;
        for (; i + 4u <= n; i += 4u) {
            const float32x4_t x = vld1q_f32(data + i);
            const float32x4_t x2 = vmulq_f32(x, x);
            const float32x4_t x3 = vmulq_f32(x2, x);
            const float32x4_t inner = vmulq_f32(
                scale, vaddq_f32(x, vmulq_f32(cubic_scale, x3)));
            const float32x4_t u = vminq_f32(vmaxq_f32(inner, lower), upper);
            const float32x4_t u2 = vmulq_f32(u, u);
            float32x4_t numerator = p5;
            numerator = vmlaq_f32(p4, numerator, u2);
            numerator = vmlaq_f32(p3, numerator, u2);
            numerator = vmlaq_f32(p2, numerator, u2);
            numerator = vmlaq_f32(p1, numerator, u2);
            numerator = vmlaq_f32(p0, numerator, u2);
            float32x4_t denominator = q5;
            denominator = vmlaq_f32(q4, denominator, u2);
            denominator = vmlaq_f32(q3, denominator, u2);
            denominator = vmlaq_f32(q2, denominator, u2);
            denominator = vmlaq_f32(q1, denominator, u2);
#if defined(__aarch64__)
            denominator = vmlaq_f32(one, denominator, u2);
            const float32x4_t tanh_u = vmulq_f32(u, vdivq_f32(numerator, denominator));
            vst1q_f32(data + i, vmulq_f32(vmulq_f32(half, x), vaddq_f32(one, tanh_u)));
#else
            (void)denominator;
            break;
#endif
        }
        for (; i < n; ++i) {
            const float x = data[i];
            const float cubic = x * x * x;
            const float c = 0.7978845608f * (x + 0.044715f * cubic);
            data[i] = 0.5f * x * (1.0f + tanhf(c));
        }
        return;
    }
#endif
    /* Keep the calibrated CUDA/CPU formula as the scalar reference. */
    for (size_t i = 0; i < n; ++i) {
        const float x = data[i];
        const float cubic = x * x * x;
        const float c = 0.7978845608f * (x + 0.044715f * cubic);
        data[i] = 0.5f * x * (1.0f + tanhf(c));
    }
}

void mynah_gelu_f32_scalar(float *data, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        const float x = data[i];
        const float cubic = x * x * x;
        const float c = 0.7978845608f * (x + 0.044715f * cubic);
        data[i] = 0.5f * x * (1.0f + tanhf(c));
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
    unsigned fused_argmax = 0;
    if (mynah_matvec_argmax_f32(matrix, a, NULL, 2u, 3u, 2u,
                                SIZE_MAX, 0, &fused_argmax) != 0 ||
        fused_argmax != 0u) {
        snprintf(error, error_capacity, "fused matvec argmax mismatch");
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

    const float ln_input[] = {1.0f, 2.0f, 3.0f, 4.0f};
    const float ln_weight[] = {1.0f, 1.0f, 1.0f, 1.0f};
    const float ln_bias[] = {0.0f, 0.0f, 0.0f, 0.0f};
    float ln_output[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    mynah_layernorm_f32(ln_input, ln_weight, ln_bias, ln_output, 1u, 4u, 1e-6f);
    if (!close_enough(ln_output[0], -1.34164f, 1e-4f) ||
        !close_enough(ln_output[3], 1.34164f, 1e-4f)) {
        snprintf(error, error_capacity, "layernorm mismatch");
        return -1;
    }

    float residual[] = {1.0f, 2.0f, 3.0f, 4.0f};
    const float residual_input[] = {3.0f, -1.0f, 2.0f, -2.0f};
    mynah_residual_add_f32(residual, residual_input, 4u);
    if (!close_enough(residual[0], 4.0f, 1e-6f) ||
        !close_enough(residual[3], 2.0f, 1e-6f)) {
        snprintf(error, error_capacity, "residual mismatch");
        return -1;
    }

    float gelu[] = {-1.0f, 0.0f, 1.0f};
    mynah_gelu_f32(gelu, 3u);
    if (!close_enough(gelu[1], 0.0f, 1e-6f) ||
        !close_enough(gelu[2], 0.8411920f, 1e-5f)) {
        snprintf(error, error_capacity, "gelu mismatch");
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
