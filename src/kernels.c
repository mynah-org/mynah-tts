#include "kernels.h"

#include "dispatch.h"

#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

#if defined(__aarch64__) && defined(__linux__)
#include <sys/prctl.h>
#endif

/* No <Accelerate/Accelerate.h> here any more.  The array GELU used to call
 * vvtanhf, which exists only on macOS, so the production target ran a scalar
 * libm loop instead and the two platforms executed different arithmetic.  The
 * tanh below is ours, on both ISAs.  PLAN.md E4-16d,
 * .work/accelerate-only-kernels.md. */

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

/* MYNAH_GELU_SCALAR keeps the libm reference GELU for rollback.  Two things
 * changed when this became a function: the decision is now readable by the
 * dispatch report, and it is read ONCE instead of on every call -- getenv on
 * the hot path is a strlen-per-entry walk of environ that the array GELU was
 * paying for per activation block. */
int mynah_gelu_vector_enabled(void) {
#if defined(MYNAH_KERNELS_NEON) || defined(MYNAH_KERNELS_AVX2)
    static int cached = -1;
    if (cached < 0) cached = getenv("MYNAH_GELU_SCALAR") == NULL;
    return cached;
#else
    return 0;
#endif
}

void mynah_gelu_f32(float *data, size_t n) {
#if defined(MYNAH_KERNELS_AVX2)
    /* Padé [5/5] tanh is accurate to about 2e-8 on [-6, 6]. */
    if (mynah_gelu_vector_enabled()) {
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
    if (mynah_gelu_vector_enabled()) {
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

/* ==========================================================================
 * Vector transcendentals -- ours, on every target (PLAN.md E4-16d)
 *
 * What used to be here: Accelerate's vvtanhf, and in codec_nanocodec.c the
 * vDSP/vForce Snake.  Both are macOS-only, so the production target ran a
 * scalar libm loop instead and the two platforms agreed by luck rather than
 * by construction.  These kernels are the construction.
 *
 * Layout of this section, and it is deliberate:
 *
 *   1. a scalar core, written once as static inline functions.  This is the
 *      DEFINITION OF CORRECTNESS.  Everything below transcribes it.
 *   2. mynah_*_scalar(), which is that core over an array and nothing else.
 *   3. the NEON and AVX2 transcriptions, guarded, with a scalar tail.
 *   4. mynah_vecmath_self_test(), which compares 2 against 3 inside one
 *      binary and both against a double-precision reference.
 *
 * A NOTE ON -ffast-math, which this project builds with.  It permits
 * reassociation, FMA contraction and reciprocal substitution in the C bodies
 * below, and it does NOT touch the intrinsics, which name their operations.
 * So the scalar and vector paths are NOT expected to be bit-identical and the
 * self test does not assert that; it asserts a ULP bound, and the bound it
 * asserts is the one that was measured.  -fno-signed-zeros is also in force,
 * which is exactly why every sign here is applied with a bit mask instead of
 * a multiply or a branch on x < 0.
 * ========================================================================== */

/* ---- tanh --------------------------------------------------------------
 * Cephes' split, with the exp written for the only range this reduction can
 * hand it.
 *
 *   |x| < 0.625              odd minimax polynomial in x^2
 *   0.625 <= |x| <= 9.010913 1 - 2/(exp(2|x|) + 1),  exp argument in [1.25, 18.03]
 *   |x| > 9.010913           +-1
 *
 * The last line is not a shortcut, it is the correctly rounded answer:
 * tanh(9) = 1 - 3.07e-8 and the largest float below 1.0 is 1 - 5.96e-8, so
 * every float argument past 9 rounds to exactly 1.0f in f32.
 *
 * Because the exp argument is confined to [1.25, 18.03] there is no overflow
 * branch, no denormal branch and no general-purpose expf here to get wrong:
 * n = round(x*log2e) lies in [2, 26], so the 2^n scale is always a normal
 * float built by one integer shift.
 * ---------------------------------------------------------------------- */
#define MYNAH_TANH_KNEE   9.010913f  /* past this, tanh(x) rounds to 1.0f    */
#define MYNAH_TANH_SMALL  0.625f     /* below this, the polynomial branch    */

/* exp(x) for x in [1.25, 18.03].  Degree-7 Taylor on the reduced argument,
 * |r| <= ln2/2 = 0.3466, so the first dropped term is r^8/40320 = 5.2e-9 --
 * about 1/20 ulp.  LN2_HI has 9 significant bits and n at most 5, so n*LN2_HI
 * is exact and the reduction loses nothing. */
#define MYNAH_EXP_LOG2E   1.4426950408889634f
#define MYNAH_EXP_LN2_HI  0.693359375f
#define MYNAH_EXP_LN2_LO (-2.12194440e-4f)

static inline float mynah_exp_poly7(float r) {
    float p = 1.0f / 5040.0f;
    p = p * r + 1.0f / 720.0f;
    p = p * r + 1.0f / 120.0f;
    p = p * r + 1.0f / 24.0f;
    p = p * r + 1.0f / 6.0f;
    p = p * r + 0.5f;
    p = p * r + 1.0f;
    p = p * r + 1.0f;
    return p;
}

static inline float mynah_ldexp_int(int n) {
    /* 2^n for n in [2, 26]; no denormal or overflow case can reach here.
     * memcpy, not a union and not a cast: the commit this branch sits on is
     * a strict-aliasing miscompile that only gcc on Linux produced. */
    const uint32_t u = (uint32_t)(n + 127) << 23;
    float f;
    memcpy(&f, &u, sizeof f);
    return f;
}

/* Sign handling, by bit, because the build carries -fno-signed-zeros: under
 * it a negation or a multiply by -1.0f is not a reliable way to move a sign.
 *
 * TWO forms, and the difference between them cost a Linux gate failure.
 *
 * mynah_set_sign REPLACES the sign: for an odd function whose magnitude was
 * computed from |x|, the magnitude carries no meaning in its sign bit, and a
 * polynomial that evaluates to -0.0 there -- which gcc's reassociation
 * produces and clang's does not -- must not flip the answer.
 *
 * mynah_apply_sign XORS the sign, which is what a sine needs, because its
 * polynomial's own sign is real: it follows the reduced argument.  For that
 * one the zero case has to be handled explicitly instead, by normalising a
 * zero magnitude to +0 before the XOR.  Without that, sin(+0.0) came back as
 * +0 from the NEON body and -0 from the scalar body on gcc 15.2 and agreed
 * on clang -- the same code, two compilers, two answers. */
static inline float mynah_set_sign(float magnitude, float like) {
    uint32_t mb, lb;
    memcpy(&mb, &magnitude, sizeof mb);
    memcpy(&lb, &like, sizeof lb);
    mb = (mb & 0x7fffffffu) | (lb & 0x80000000u);
    memcpy(&magnitude, &mb, sizeof magnitude);
    return magnitude;
}

static inline float mynah_apply_sign(float r, float like, int extra_flip) {
    uint32_t rb, lb;
    memcpy(&rb, &r, sizeof rb);
    memcpy(&lb, &like, sizeof lb);
    if ((rb & 0x7fffffffu) == 0u) rb = 0u;   /* a zero is +0 before signing */
    rb ^= (lb & 0x80000000u);
    if (extra_flip) rb ^= 0x80000000u;
    memcpy(&r, &rb, sizeof r);
    return r;
}

static inline float mynah_exp_narrow(float x) {
    /* rintf, not floorf(y + 0.5f): the default rounding mode is
     * to-nearest-even, which is what vrndnq_f32 and _mm256_round_ps do.  The
     * two spellings differ only on exact ties, and matching them there costs
     * nothing and removes a class of scalar/vector disagreement. */
    const float fn = rintf(x * MYNAH_EXP_LOG2E);
    const float r = (x - fn * MYNAH_EXP_LN2_HI) - fn * MYNAH_EXP_LN2_LO;
    return mynah_exp_poly7(r) * mynah_ldexp_int((int)fn);
}

static inline float mynah_tanh_small_poly(float x) {
    const float z = x * x;
    return x + x * z * ((((-5.70498872745e-3f * z + 2.06390887954e-2f) * z
                          - 5.37397155531e-2f) * z + 1.33314422036e-1f) * z
                        - 3.33332819422e-1f);
}

static inline float mynah_tanh_core(float x) {
    const float ax = fabsf(x);
    /* Written as !(ax <= knee) so that NaN takes this branch. */
    if (!(ax <= MYNAH_TANH_KNEE)) {
        if (ax != ax) return x + x;           /* NaN in, quiet NaN out */
        return copysignf(1.0f, x);            /* +-Inf and everything past 9 */
    }
    /* Both branches compute |tanh(x)| and then take the sign from x by bit.
     * That is what makes tanh(-0.0) = -0.0 survive -fno-signed-zeros, and it
     * is also what the NEON and AVX2 transcriptions below do, so the three
     * agree on the sign of a zero without a special case anywhere. */
    if (ax < MYNAH_TANH_SMALL) {
        /* Denormal in, the same denormal out: ax*ax underflows to 0 and the
         * polynomial collapses to its linear term. */
        return mynah_set_sign(mynah_tanh_small_poly(ax), x);
    }
    const float e = mynah_exp_narrow(ax + ax);
    return mynah_set_sign(1.0f - 2.0f / (e + 1.0f), x);
}

/* ---- sin ---------------------------------------------------------------
 * Cephes' octant scheme -- j = floor(4|x|/pi) picks an octant, the polynomial
 * is a sine or a cosine depending on which -- with ONE deliberate departure:
 * the argument reduction is done in DOUBLE, not in float.
 *
 * THAT DEPARTURE IS NOT TASTE, IT IS A MEASUREMENT.  The textbook float
 * spelling subtracts pi/4 in three pieces, `((ax - y*DP1) - y*DP2) - y*DP3`,
 * and it depends entirely on those subtractions happening in that order.
 * This project builds with -ffast-math, which is allowed to reassociate them
 * into `ax - (y*DP1 + y*DP2 + y*DP3)` -- and that single change throws away
 * every bit the split existed to protect.  Measured on this machine, clang
 * -O3 -ffast-math -march=native, maximum absolute error against a
 * double-precision sine:
 *
 *     range        float Cody-Waite   double reduction   libm sinf
 *     [0, 2pi]        1.85e-07           6.23e-08         3.24e-08
 *     [0, 1e3]        2.78e-05           6.97e-08         5.04e-08
 *     [0, 8192]       2.28e-04           6.91e-08         5.04e-08
 *     [0, 65536]      1.82e-03           6.90e-08         5.04e-08
 *
 * Three thousand times worse at 8190, and silent.  Without -ffast-math the
 * float spelling matches the double one exactly (7.7e-08 in both), which is
 * why this is the kind of bug that survives review: the algorithm is right
 * and the build flag is what breaks it.
 *
 * The double reduction has no such dependency.  y is at most 83446 and
 * y*PIO4_HI is at most 65536, so even a compiler that folds PIO4_HI and
 * PIO4_LO together leaves an error of order ulp(65536) in double = 1.5e-11,
 * which is eleven orders below the f32 result's own ulp.  It costs two
 * widening converts and one narrowing convert per vector.
 *
 * Past MYNAH_SIN_LIMIT, and for every non-finite input, the lane is handed to
 * libm sinf, so the vector path, the scalar path and libm are the same
 * function out there rather than three approximations of it.
 * ---------------------------------------------------------------------- */
#define MYNAH_SIN_LIMIT 65536.0f
#define MYNAH_SIN_FOPI  1.2732395447351626862  /* 4/pi, double              */
#define MYNAH_SIN_PIO4_HI 7.85398163397448279e-1   /* pi/4 to 53 bits       */
#define MYNAH_SIN_PIO4_LO 3.061616997868383018e-17 /* ...and the next 53    */

static inline float mynah_sin_poly(float zz, float z) {
    return ((-1.9515295891e-4f * zz + 8.3321608736e-3f) * zz
            - 1.6666654611e-1f) * zz * z + z;
}

static inline float mynah_cos_poly(float zz) {
    float y = ((2.443315711809948e-5f * zz - 1.388731625493765e-3f) * zz
               + 4.166664568298827e-2f) * zz * zz;
    y -= 0.5f * zz;
    return y + 1.0f;
}

static inline float mynah_sin_core(float x) {
    const float ax = fabsf(x);
    /* !(ax < limit) catches NaN and +-Inf as well as the large arguments. */
    if (!(ax < MYNAH_SIN_LIMIT)) return sinf(x);

    const double d = (double)ax;
    double y = floor(d * MYNAH_SIN_FOPI);
    int j = (int)y;
    /* Round the octant index up to an even one, so the reduced argument
     * lands in [-pi/4, pi/4] and one of two polynomials covers it. */
    if (j & 1) { ++j; y += 1.0; }
    int negate = 0;
    j &= 7;
    if (j > 3) { negate = 1; j -= 4; }

    const float z = (float)((d - y * MYNAH_SIN_PIO4_HI)
                            - y * MYNAH_SIN_PIO4_LO);
    const float zz = z * z;
    const float r = (j == 1 || j == 2) ? mynah_cos_poly(zz)
                                      : mynah_sin_poly(zz, z);
    /* sin is odd, so the sign of x and the octant flip are two XORs of one
     * bit.  sin(-0.0) = -0.0 comes out of this for free. */
    return mynah_apply_sign(r, x, negate);
}

/* ---- the scalar spellings: the core over an array, and nothing else ---- */
void mynah_tanh_f32_scalar(const float *input, float *output, size_t n) {
    for (size_t i = 0; i < n; ++i) output[i] = mynah_tanh_core(input[i]);
}

void mynah_sin_f32_scalar(const float *input, float *output, size_t n) {
    for (size_t i = 0; i < n; ++i) output[i] = mynah_sin_core(input[i]);
}

void mynah_snake_row_f32_scalar(float *row, size_t length, float alpha) {
    /* ONE division, hoisted, then a multiply -- not `s*s / (alpha + 1e-9f)`
     * per element.  The build carries -freciprocal-math, under which gcc
     * rewrites the per-element divide as a multiply by an estimate while the
     * NEON intrinsic stays a true divide, and the two bodies then disagreed
     * by up to 4 ulp on gcc 15.2 while agreeing bit for bit on clang.  A
     * hoisted reciprocal is one correctly rounded division and one multiply
     * on every compiler and every ISA, so there is nothing left to rewrite. */
    const float inverse_alpha = 1.0f / (alpha + 1.0e-9f);
    for (size_t t = 0; t < length; ++t) {
        const float v = row[t];
        const float s = mynah_sin_core(alpha * v);
        row[t] = v + s * s * inverse_alpha;
    }
}

/* ---- NEON transcription ------------------------------------------------
 * Same algorithm, same constants, same order of operations.  Every branch of
 * the scalar core becomes a select, so all lanes evaluate both sides; that is
 * safe here because neither side can trap -- the exp of a NaN or an infinity
 * produces a garbage float, never a signal, and it is selected away.
 * ---------------------------------------------------------------------- */
#if defined(MYNAH_KERNELS_NEON) && defined(__aarch64__)
#define MYNAH_VECMATH_NEON 1

static inline float32x4_t neon_exp_narrow(float32x4_t x) {
    const float32x4_t fn = vrndnq_f32(vmulq_f32(x, vdupq_n_f32(MYNAH_EXP_LOG2E)));
    float32x4_t r = vsubq_f32(x, vmulq_f32(fn, vdupq_n_f32(MYNAH_EXP_LN2_HI)));
    r = vsubq_f32(r, vmulq_f32(fn, vdupq_n_f32(MYNAH_EXP_LN2_LO)));
    float32x4_t p = vdupq_n_f32(1.0f / 5040.0f);
    p = vmlaq_f32(vdupq_n_f32(1.0f / 720.0f), p, r);
    p = vmlaq_f32(vdupq_n_f32(1.0f / 120.0f), p, r);
    p = vmlaq_f32(vdupq_n_f32(1.0f / 24.0f), p, r);
    p = vmlaq_f32(vdupq_n_f32(1.0f / 6.0f), p, r);
    p = vmlaq_f32(vdupq_n_f32(0.5f), p, r);
    p = vmlaq_f32(vdupq_n_f32(1.0f), p, r);
    p = vmlaq_f32(vdupq_n_f32(1.0f), p, r);
    /* 2^n by integer shift.  vcvtq_s32_f32 truncates but fn is already a
     * whole number, and the [2, 26] range means no saturation for any input
     * that will SURVIVE the select in neon_tanh.  Lanes that will not survive
     * -- NaN, +-Inf -- convert to INT32_MAX, and `INT32_MAX + 127` is signed
     * overflow: undefined behaviour in C even though the instruction wraps.
     * gcc's arm_neon.h spells vaddq_s32 as `a + b`, so UBSan on the Linux box
     * reported it; clang's builtin spelling hid it entirely.  The bias and
     * the shift are therefore done unsigned, where wrapping is defined, and
     * the garbage those lanes produce is discarded by the select as before. */
    const uint32x4_t n = vreinterpretq_u32_s32(vcvtq_s32_f32(fn));
    const uint32x4_t biased = vaddq_u32(n, vdupq_n_u32(127u));
    const float32x4_t scale =
        vreinterpretq_f32_u32(vshlq_n_u32(biased, 23));
    return vmulq_f32(p, scale);
}

static inline float32x4_t neon_tanh(float32x4_t x) {
    const uint32x4_t sign_mask = vdupq_n_u32(0x80000000u);
    const float32x4_t ax = vabsq_f32(x);

    /* small branch */
    const float32x4_t z = vmulq_f32(ax, ax);
    float32x4_t poly = vdupq_n_f32(-5.70498872745e-3f);
    poly = vmlaq_f32(vdupq_n_f32(2.06390887954e-2f), poly, z);
    poly = vmlaq_f32(vdupq_n_f32(-5.37397155531e-2f), poly, z);
    poly = vmlaq_f32(vdupq_n_f32(1.33314422036e-1f), poly, z);
    poly = vmlaq_f32(vdupq_n_f32(-3.33332819422e-1f), poly, z);
    const float32x4_t small = vmlaq_f32(ax, vmulq_f32(ax, z), poly);

    /* exp branch */
    const float32x4_t e = neon_exp_narrow(vaddq_f32(ax, ax));
    const float32x4_t big = vsubq_f32(
        vdupq_n_f32(1.0f),
        vdivq_f32(vdupq_n_f32(2.0f), vaddq_f32(e, vdupq_n_f32(1.0f))));

    float32x4_t r = vbslq_f32(vcltq_f32(ax, vdupq_n_f32(MYNAH_TANH_SMALL)),
                              small, big);
    /* Past the knee -- and for +-Inf, which compares greater -- the answer is
     * exactly 1.  NaN compares false here and keeps the garbage from `big`,
     * which the NaN select below then overwrites. */
    r = vbslq_f32(vcgtq_f32(ax, vdupq_n_f32(MYNAH_TANH_KNEE)),
                  vdupq_n_f32(1.0f), r);
    /* sign of x, by bit */
    r = vreinterpretq_f32_u32(vorrq_u32(
        vbicq_u32(vreinterpretq_u32_f32(r), sign_mask),
        vandq_u32(vreinterpretq_u32_f32(x), sign_mask)));
    /* NaN in, the same NaN out.  vceqq_f32(x, x) is false only for NaN. */
    return vbslq_f32(vceqq_f32(x, x), r, vaddq_f32(x, x));
}

static inline float32x4_t neon_sin_reduced(float32x4_t ax, uint32x4_t *negate) {
    /* Reduction in f64, two lanes at a time -- see the measurement above. */
    const float64x2_t fopi = vdupq_n_f64(MYNAH_SIN_FOPI);
    const float64x2_t hi = vdupq_n_f64(MYNAH_SIN_PIO4_HI);
    const float64x2_t lo = vdupq_n_f64(MYNAH_SIN_PIO4_LO);
    const uint64x2_t one = vdupq_n_u64(1u);

    float64x2_t d0 = vcvt_f64_f32(vget_low_f32(ax));
    float64x2_t d1 = vcvt_high_f64_f32(ax);
    float64x2_t y0 = vrndmq_f64(vmulq_f64(d0, fopi));   /* floor */
    float64x2_t y1 = vrndmq_f64(vmulq_f64(d1, fopi));
    /* Unsigned for the same reason as neon_exp_narrow above: a NaN or an
     * infinity converts to INT64_MAX and `INT64_MAX + 1` is UB in C, which
     * gcc's arm_neon.h exposes to UBSan.  Those lanes are handed to libm by
     * the caller, so the wrapped value is never read. */
    uint64x2_t j0 = vreinterpretq_u64_s64(vcvtq_s64_f64(y0));
    uint64x2_t j1 = vreinterpretq_u64_s64(vcvtq_s64_f64(y1));
    const uint64x2_t odd0 = vandq_u64(j0, one);
    const uint64x2_t odd1 = vandq_u64(j1, one);
    j0 = vaddq_u64(j0, odd0);
    j1 = vaddq_u64(j1, odd1);
    y0 = vaddq_f64(y0, vcvtq_f64_s64(vreinterpretq_s64_u64(odd0)));
    y1 = vaddq_f64(y1, vcvtq_f64_s64(vreinterpretq_s64_u64(odd1)));
    const float64x2_t z0 = vsubq_f64(vsubq_f64(d0, vmulq_f64(y0, hi)),
                                     vmulq_f64(y0, lo));
    const float64x2_t z1 = vsubq_f64(vsubq_f64(d1, vmulq_f64(y1, hi)),
                                     vmulq_f64(y1, lo));
    const float32x4_t z = vcombine_f32(vcvt_f32_f64(z0), vcvt_f32_f64(z1));

    int32x4_t j = vreinterpretq_s32_u32(
        vcombine_u32(vmovn_u64(j0), vmovn_u64(j1)));
    j = vandq_s32(j, vdupq_n_s32(7));
    const uint32x4_t high = vcgtq_s32(j, vdupq_n_s32(3));
    const int32x4_t jq = vsubq_s32(j, vandq_s32(vreinterpretq_s32_u32(high),
                                                vdupq_n_s32(4)));
    *negate = high;

    const float32x4_t zz = vmulq_f32(z, z);

    float32x4_t sp = vdupq_n_f32(-1.9515295891e-4f);
    sp = vmlaq_f32(vdupq_n_f32(8.3321608736e-3f), sp, zz);
    sp = vmlaq_f32(vdupq_n_f32(-1.6666654611e-1f), sp, zz);
    const float32x4_t sin_v = vmlaq_f32(z, vmulq_f32(zz, z), sp);

    float32x4_t cp = vdupq_n_f32(2.443315711809948e-5f);
    cp = vmlaq_f32(vdupq_n_f32(-1.388731625493765e-3f), cp, zz);
    cp = vmlaq_f32(vdupq_n_f32(4.166664568298827e-2f), cp, zz);
    float32x4_t cos_v = vmulq_f32(vmulq_f32(zz, zz), cp);
    cos_v = vsubq_f32(cos_v, vmulq_f32(vdupq_n_f32(0.5f), zz));
    cos_v = vaddq_f32(cos_v, vdupq_n_f32(1.0f));

    /* the cosine polynomial serves octants 1 and 2 */
    const uint32x4_t use_cos = vorrq_u32(vceqq_s32(jq, vdupq_n_s32(1)),
                                         vceqq_s32(jq, vdupq_n_s32(2)));
    return vbslq_f32(use_cos, cos_v, sin_v);
}

/* Sets the sign bits: the sign of x, XOR the octant flip. */
static inline float32x4_t neon_sin_sign(float32x4_t r, float32x4_t x,
                                        uint32x4_t negate) {
    const uint32x4_t sign_mask = vdupq_n_u32(0x80000000u);
    uint32x4_t bits = vreinterpretq_u32_f32(r);
    /* A zero magnitude is +0 before any sign is XORed in.  See the note on
     * mynah_apply_sign: the polynomial can produce -0.0 on one compiler and
     * +0.0 on another, and the XOR would then carry that through. */
    bits = vbicq_u32(bits, vceqzq_u32(vandq_u32(bits, vdupq_n_u32(0x7fffffffu))));
    bits = veorq_u32(bits, vandq_u32(vreinterpretq_u32_f32(x), sign_mask));
    bits = veorq_u32(bits, vandq_u32(negate, sign_mask));
    return vreinterpretq_f32_u32(bits);
}
#endif /* MYNAH_VECMATH_NEON */

/* ---- AVX2 transcription ------------------------------------------------
 * Line for line the same as the NEON block above; the only differences are
 * spelling and eight lanes instead of four.  Kept beside it on purpose --
 * coding rule 5 is that the ISA kernels stay numerically equivalent, and the
 * cheapest way to keep two of them equivalent is to keep them adjacent.
 * ---------------------------------------------------------------------- */
#if defined(MYNAH_KERNELS_AVX2)
#define MYNAH_VECMATH_AVX2 1

static inline __m256 avx2_exp_narrow(__m256 x) {
    const __m256 fn = _mm256_round_ps(
        _mm256_mul_ps(x, _mm256_set1_ps(MYNAH_EXP_LOG2E)),
        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256 r = _mm256_sub_ps(x, _mm256_mul_ps(fn, _mm256_set1_ps(MYNAH_EXP_LN2_HI)));
    r = _mm256_sub_ps(r, _mm256_mul_ps(fn, _mm256_set1_ps(MYNAH_EXP_LN2_LO)));
    __m256 p = _mm256_set1_ps(1.0f / 5040.0f);
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f / 720.0f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f / 120.0f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f / 24.0f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f / 6.0f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(0.5f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f));
    const __m256i n = _mm256_cvttps_epi32(fn);
    const __m256i biased = _mm256_add_epi32(n, _mm256_set1_epi32(127));
    const __m256 scale = _mm256_castsi256_ps(_mm256_slli_epi32(biased, 23));
    return _mm256_mul_ps(p, scale);
}

static inline __m256 avx2_tanh(__m256 x) {
    const __m256 sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32((int)0x80000000));
    const __m256 ax = _mm256_andnot_ps(sign_mask, x);

    const __m256 z = _mm256_mul_ps(ax, ax);
    __m256 poly = _mm256_set1_ps(-5.70498872745e-3f);
    poly = _mm256_fmadd_ps(poly, z, _mm256_set1_ps(2.06390887954e-2f));
    poly = _mm256_fmadd_ps(poly, z, _mm256_set1_ps(-5.37397155531e-2f));
    poly = _mm256_fmadd_ps(poly, z, _mm256_set1_ps(1.33314422036e-1f));
    poly = _mm256_fmadd_ps(poly, z, _mm256_set1_ps(-3.33332819422e-1f));
    const __m256 small = _mm256_fmadd_ps(_mm256_mul_ps(ax, z), poly, ax);

    const __m256 e = avx2_exp_narrow(_mm256_add_ps(ax, ax));
    const __m256 big = _mm256_sub_ps(
        _mm256_set1_ps(1.0f),
        _mm256_div_ps(_mm256_set1_ps(2.0f),
                      _mm256_add_ps(e, _mm256_set1_ps(1.0f))));

    __m256 r = _mm256_blendv_ps(
        big, small,
        _mm256_cmp_ps(ax, _mm256_set1_ps(MYNAH_TANH_SMALL), _CMP_LT_OQ));
    r = _mm256_blendv_ps(
        r, _mm256_set1_ps(1.0f),
        _mm256_cmp_ps(ax, _mm256_set1_ps(MYNAH_TANH_KNEE), _CMP_GT_OQ));
    r = _mm256_or_ps(_mm256_andnot_ps(sign_mask, r), _mm256_and_ps(x, sign_mask));
    /* _CMP_EQ_OQ against itself is false only for NaN. */
    return _mm256_blendv_ps(_mm256_add_ps(x, x), r,
                            _mm256_cmp_ps(x, x, _CMP_EQ_OQ));
}

static inline __m256 avx2_sin_reduced(__m256 ax, __m256 *negate) {
    /* Reduction in f64, four lanes at a time -- see the measurement above. */
    const __m256d fopi = _mm256_set1_pd(MYNAH_SIN_FOPI);
    const __m256d hi = _mm256_set1_pd(MYNAH_SIN_PIO4_HI);
    const __m256d lo = _mm256_set1_pd(MYNAH_SIN_PIO4_LO);

    const __m256d d0 = _mm256_cvtps_pd(_mm256_castps256_ps128(ax));
    const __m256d d1 = _mm256_cvtps_pd(_mm256_extractf128_ps(ax, 1));
    __m256d y0 = _mm256_floor_pd(_mm256_mul_pd(d0, fopi));
    __m256d y1 = _mm256_floor_pd(_mm256_mul_pd(d1, fopi));
    __m128i j0 = _mm256_cvttpd_epi32(y0);
    __m128i j1 = _mm256_cvttpd_epi32(y1);
    const __m128i odd0 = _mm_and_si128(j0, _mm_set1_epi32(1));
    const __m128i odd1 = _mm_and_si128(j1, _mm_set1_epi32(1));
    j0 = _mm_add_epi32(j0, odd0);
    j1 = _mm_add_epi32(j1, odd1);
    y0 = _mm256_add_pd(y0, _mm256_cvtepi32_pd(odd0));
    y1 = _mm256_add_pd(y1, _mm256_cvtepi32_pd(odd1));
    const __m256d z0 = _mm256_sub_pd(_mm256_sub_pd(d0, _mm256_mul_pd(y0, hi)),
                                     _mm256_mul_pd(y0, lo));
    const __m256d z1 = _mm256_sub_pd(_mm256_sub_pd(d1, _mm256_mul_pd(y1, hi)),
                                     _mm256_mul_pd(y1, lo));
    const __m256 z = _mm256_set_m128(_mm256_cvtpd_ps(z1), _mm256_cvtpd_ps(z0));

    __m256i j = _mm256_set_m128i(j1, j0);
    j = _mm256_and_si256(j, _mm256_set1_epi32(7));
    const __m256i high = _mm256_cmpgt_epi32(j, _mm256_set1_epi32(3));
    const __m256i jq = _mm256_sub_epi32(
        j, _mm256_and_si256(high, _mm256_set1_epi32(4)));
    *negate = _mm256_castsi256_ps(high);

    const __m256 zz = _mm256_mul_ps(z, z);

    __m256 sp = _mm256_set1_ps(-1.9515295891e-4f);
    sp = _mm256_fmadd_ps(sp, zz, _mm256_set1_ps(8.3321608736e-3f));
    sp = _mm256_fmadd_ps(sp, zz, _mm256_set1_ps(-1.6666654611e-1f));
    const __m256 sin_v = _mm256_fmadd_ps(_mm256_mul_ps(zz, z), sp, z);

    __m256 cp = _mm256_set1_ps(2.443315711809948e-5f);
    cp = _mm256_fmadd_ps(cp, zz, _mm256_set1_ps(-1.388731625493765e-3f));
    cp = _mm256_fmadd_ps(cp, zz, _mm256_set1_ps(4.166664568298827e-2f));
    __m256 cos_v = _mm256_mul_ps(_mm256_mul_ps(zz, zz), cp);
    cos_v = _mm256_sub_ps(cos_v, _mm256_mul_ps(_mm256_set1_ps(0.5f), zz));
    cos_v = _mm256_add_ps(cos_v, _mm256_set1_ps(1.0f));

    const __m256i use_cos = _mm256_or_si256(
        _mm256_cmpeq_epi32(jq, _mm256_set1_epi32(1)),
        _mm256_cmpeq_epi32(jq, _mm256_set1_epi32(2)));
    return _mm256_blendv_ps(sin_v, cos_v, _mm256_castsi256_ps(use_cos));
}

static inline __m256 avx2_sin_sign(__m256 r, __m256 x, __m256 negate) {
    const __m256 sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32((int)0x80000000));
    /* A zero magnitude is +0 before any sign is XORed in -- see the note on
     * mynah_apply_sign. */
    const __m256i mag = _mm256_and_si256(_mm256_castps_si256(r),
                                         _mm256_set1_epi32(0x7fffffff));
    r = _mm256_castsi256_ps(_mm256_andnot_si256(
        _mm256_cmpeq_epi32(mag, _mm256_setzero_si256()),
        _mm256_castps_si256(r)));
    __m256 out = _mm256_xor_ps(r, _mm256_and_ps(x, sign_mask));
    return _mm256_xor_ps(out, _mm256_and_ps(negate, sign_mask));
}
#endif /* MYNAH_VECMATH_AVX2 */

/* ---- the dispatching spellings ---------------------------------------- */
const char *mynah_vecmath_isa(void) {
#if defined(MYNAH_VECMATH_NEON)
    return "NEON";
#elif defined(MYNAH_VECMATH_AVX2)
    return "AVX2";
#else
    return "scalar";
#endif
}

void mynah_tanh_f32(const float *input, float *output, size_t n) {
    size_t i = 0;
#if defined(MYNAH_VECMATH_NEON)
    for (; i + 4u <= n; i += 4u)
        vst1q_f32(output + i, neon_tanh(vld1q_f32(input + i)));
#elif defined(MYNAH_VECMATH_AVX2)
    for (; i + 8u <= n; i += 8u)
        _mm256_storeu_ps(output + i, avx2_tanh(_mm256_loadu_ps(input + i)));
#endif
    for (; i < n; ++i) output[i] = mynah_tanh_core(input[i]);
}

void mynah_sin_f32(const float *input, float *output, size_t n) {
    size_t i = 0;
#if defined(MYNAH_VECMATH_NEON)
    const float32x4_t limit = vdupq_n_f32(MYNAH_SIN_LIMIT);
    for (; i + 4u <= n; i += 4u) {
        const float32x4_t x = vld1q_f32(input + i);
        const float32x4_t ax = vabsq_f32(x);
        uint32x4_t negate;
        const float32x4_t r = neon_sin_reduced(ax, &negate);
        vst1q_f32(output + i, neon_sin_sign(r, x, negate));
        /* NOT (ax < limit) is true for NaN, +-Inf and every argument the
         * Cody-Waite reduction can no longer split exactly.  Those lanes are
         * redone with libm, so the vector and scalar paths are the same
         * function out there rather than two approximations of it. */
        const uint32x4_t hard = vmvnq_u32(vcltq_f32(ax, limit));
        if (vmaxvq_u32(hard) != 0u) {
            uint32_t lanes[4];
            vst1q_u32(lanes, hard);
            for (size_t k = 0; k < 4u; ++k)
                if (lanes[k] != 0u) output[i + k] = sinf(input[i + k]);
        }
    }
#elif defined(MYNAH_VECMATH_AVX2)
    const __m256 limit = _mm256_set1_ps(MYNAH_SIN_LIMIT);
    const __m256 sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32((int)0x80000000));
    for (; i + 8u <= n; i += 8u) {
        const __m256 x = _mm256_loadu_ps(input + i);
        const __m256 ax = _mm256_andnot_ps(sign_mask, x);
        __m256 negate;
        const __m256 r = avx2_sin_reduced(ax, &negate);
        _mm256_storeu_ps(output + i, avx2_sin_sign(r, x, negate));
        /* _CMP_NLT_UQ is "not less than, unordered true": NaN included. */
        const int hard = _mm256_movemask_ps(
            _mm256_cmp_ps(ax, limit, _CMP_NLT_UQ));
        if (hard != 0) {
            for (size_t k = 0; k < 8u; ++k)
                if ((hard >> k) & 1) output[i + k] = sinf(input[i + k]);
        }
    }
#endif
    for (; i < n; ++i) output[i] = mynah_sin_core(input[i]);
}

void mynah_snake_row_f32(float *row, size_t length, float alpha) {
    /* See mynah_snake_row_f32_scalar: one hoisted reciprocal, then a
     * multiply, so the scalar body and both vector bodies are the same
     * arithmetic under every compiler. */
    const float inverse_alpha = 1.0f / (alpha + 1.0e-9f);
    size_t t = 0;
#if defined(MYNAH_VECMATH_NEON)
    {
        const float32x4_t va = vdupq_n_f32(alpha);
        const float32x4_t vd = vdupq_n_f32(inverse_alpha);
        const float32x4_t limit = vdupq_n_f32(MYNAH_SIN_LIMIT);
        for (; t + 4u <= length; t += 4u) {
            const float32x4_t v = vld1q_f32(row + t);
            const float32x4_t u = vmulq_f32(va, v);
            const float32x4_t au = vabsq_f32(u);
            uint32x4_t negate;
            const float32x4_t sr = neon_sin_reduced(au, &negate);
            /* The Snake only ever squares the sine, so the sign never
             * reaches the output -- but it is applied anyway, because the
             * kernel must stay the same function as mynah_sin_f32 for the
             * self test to be able to compare them. */
            float32x4_t s = neon_sin_sign(sr, u, negate);
            const uint32x4_t hard = vmvnq_u32(vcltq_f32(au, limit));
            if (vmaxvq_u32(hard) != 0u) {
                uint32_t lanes[4];
                float sv[4], uv[4];
                vst1q_u32(lanes, hard);
                vst1q_f32(sv, s);
                vst1q_f32(uv, u);
                for (size_t k = 0; k < 4u; ++k)
                    if (lanes[k] != 0u) sv[k] = sinf(uv[k]);
                s = vld1q_f32(sv);
            }
            vst1q_f32(row + t, vaddq_f32(v, vmulq_f32(vmulq_f32(s, s), vd)));
        }
    }
#elif defined(MYNAH_VECMATH_AVX2)
    {
        const __m256 va = _mm256_set1_ps(alpha);
        const __m256 vd = _mm256_set1_ps(inverse_alpha);
        const __m256 limit = _mm256_set1_ps(MYNAH_SIN_LIMIT);
        const __m256 sign_mask =
            _mm256_castsi256_ps(_mm256_set1_epi32((int)0x80000000));
        for (; t + 8u <= length; t += 8u) {
            const __m256 v = _mm256_loadu_ps(row + t);
            const __m256 u = _mm256_mul_ps(va, v);
            const __m256 au = _mm256_andnot_ps(sign_mask, u);
            __m256 negate;
            const __m256 sr = avx2_sin_reduced(au, &negate);
            __m256 s = avx2_sin_sign(sr, u, negate);
            const int hard = _mm256_movemask_ps(
                _mm256_cmp_ps(au, limit, _CMP_NLT_UQ));
            if (hard != 0) {
                float sv[8], uv[8];
                _mm256_storeu_ps(sv, s);
                _mm256_storeu_ps(uv, u);
                for (size_t k = 0; k < 8u; ++k)
                    if ((hard >> k) & 1) sv[k] = sinf(uv[k]);
                s = _mm256_loadu_ps(sv);
            }
            _mm256_storeu_ps(row + t,
                _mm256_add_ps(v, _mm256_mul_ps(_mm256_mul_ps(s, s), vd)));
        }
    }
#endif
    for (; t < length; ++t) {
        const float v = row[t];
        const float s = mynah_sin_core(alpha * v);
        row[t] = v + s * s * inverse_alpha;
    }
}

/* ---- the model-free self test (PLAN.md E3-6) ---------------------------
 * Three things are checked, and they are different things:
 *
 *   a. ACCURACY.  Every kernel against a double-precision evaluation of the
 *      same mathematical function, in ULP of the true value.  This is what
 *      says whether replacing a vendor routine fixed macOS or degraded it,
 *      and it is measurable on a machine with no vendor routine on it.
 *   b. AGREEMENT.  The scalar body against the vector body, inside one
 *      binary, no environment variable, no model.  This is what says that the
 *      two ISAs are the same function -- coding rule 5.
 *   c. THE PLACES TRANSCENDENTALS GO WRONG.  Zero and negative zero,
 *      denormals, the tanh saturation knee from both sides, +-Inf, NaN, and
 *      the sine's argument reduction far from the origin, which is the one
 *      that is wrong in most hand-written vector sines.
 *
 * The bounds asserted below are the bounds that were MEASURED, rounded up to
 * the next sensible number.  They are deliberately tight: a bound of "10 ulp"
 * would pass for a kernel that had quietly lost half its precision.
 * ---------------------------------------------------------------------- */

/* ULP distance between a float result and the true value, measured in ulp of
 * the true value's binade.  Denormals are measured in ulp of the smallest
 * normal's binade, which is the right unit there. */
double mynah_vecmath_ulp(float got, double want);
double mynah_vecmath_ulp(float got, double want) {
    if (got != got || want != want)
        return (got != got && want != want) ? 0.0 : 1.0e30;
    const double g = (double)got;
    if (g == want) return 0.0;
    if (got > 3.0e38f || got < -3.0e38f || want > 3.0e38 || want < -3.0e38) {
        /* one is infinite and the other is not, or they straddle the range */
        if ((g > 3.0e38) != (want > 3.0e38)) return 1.0e30;
        if ((g < -3.0e38) != (want < -3.0e38)) return 1.0e30;
    }
    const double magnitude = (want != 0.0) ? want : g;
    int exponent = 0;
    frexp(magnitude, &exponent);
    double ulp = ldexp(1.0, exponent - 24);
    const double smallest = ldexp(1.0, -149);
    if (!(ulp > smallest)) ulp = smallest;
    return fabs(g - want) / ulp;
}

#define VEC_FAIL(...)                                                  \
    do {                                                               \
        if (error != NULL && error_capacity > 0)                       \
            snprintf(error, error_capacity, __VA_ARGS__);              \
        return -1;                                                     \
    } while (0)

/* Does this BUILD flush denormals to zero?
 *
 * Not a property of these kernels: -ffast-math links a startup object that
 * sets the flush-to-zero bit in FPCR on gcc/aarch64 and does not on Apple
 * clang, so a denormal argument survives on the development machine and
 * becomes zero on the production one -- before any of our code runs.  The
 * self test has to know which platform it is on, or it is asserting a
 * property of the compiler driver and calling it a kernel bug.  This is
 * reported by tests/test_kernels.c and is worth knowing independently of
 * anything here. */
int mynah_vecmath_denormals_flush(void);
int mynah_vecmath_denormals_flush(void) {
    volatile float tiny = 1.0e-40f;
    volatile float one = 1.0f;
    const float product = tiny * one;
    return product == 0.0f;
}

/* True zero, by bits.  `fabsf(x) == 0.0f` is NOT this: on a flush-to-zero
 * build the comparison flushes a denormal operand and answers true for a
 * denormal, which is how a denormal ended up inside a bit-identity assertion
 * meant for zeros. */
static int vec_is_zero(float x) {
    uint32_t b;
    memcpy(&b, &x, sizeof b);
    return (b & 0x7fffffffu) == 0u;
}

static int vec_is_subnormal(float x) {
    uint32_t b;
    memcpy(&b, &x, sizeof b);
    return (b & 0x7f800000u) == 0u && (b & 0x007fffffu) != 0u;
}

/* Same value, same bits -- used for the sign of zero, where == is blind. */
static int vec_same_bits(float a, float b) {
    uint32_t x, y;
    memcpy(&x, &a, sizeof x);
    memcpy(&y, &b, sizeof y);
    return x == y;
}

/* A deterministic, badly-behaved-on-purpose spread of inputs: it must not be
 * a uniform grid, because a uniform grid can miss an octant boundary for a
 * whole run and then the test has only ever seen the easy case. */
static float vec_sample(size_t i, float lo, float hi, size_t count) {
    const double u = ((double)i + 0.37631) / (double)count;
    return (float)((double)lo + ((double)hi - (double)lo) * u);
}

int mynah_vecmath_self_test(char *error, size_t error_capacity) {
    if (error != NULL && error_capacity > 0) error[0] = '\0';
    enum { BLOCK = 1024 };
    float in[BLOCK], vec[BLOCK], ref[BLOCK];

    /* --- a/b/c for tanh ------------------------------------------------ */
    {
        /* Ranges chosen to sit on top of every branch and every boundary:
         * the polynomial region, the exp region, both sides of the knee, and
         * a long tail where the answer must be exactly 1. */
        static const float lo[] = {-1.0e-30f, -1.0e-6f, -0.7f,  -0.625f,
                                   -9.02f,    -60.0f,   0.0f,   0.62f,
                                   0.624f,    8.9f,     9.01f,  -3.0f,
                                   -9.0f,     3.0f};
        static const float hi[] = {1.0e-30f,  1.0e-6f,  0.7f,   0.625f,
                                   -8.99f,    -9.0f,    1.0e-8f, 0.63f,
                                   0.626f,    9.1f,     40.0f,  3.0f,
                                   9.0f,      9.0f};
        /* The last two ranges cover 0.7 to 8.9, which nothing else did.  The
         * hole was found by the negative control: a knee moved from 9.010913
         * down to 6 stayed inside every budget here, because no sample fell
         * between the polynomial branch and the saturated tail. */
        double worst_accuracy = 0.0, worst_agreement = 0.0;
        for (size_t r = 0; r < sizeof(lo) / sizeof(lo[0]); ++r) {
            for (size_t i = 0; i < BLOCK; ++i)
                in[i] = vec_sample(i, lo[r], hi[r], BLOCK);
            mynah_tanh_f32(in, vec, BLOCK);
            mynah_tanh_f32_scalar(in, ref, BLOCK);
            for (size_t i = 0; i < BLOCK; ++i) {
                const double truth = tanh((double)in[i]);
                const double acc_v = mynah_vecmath_ulp(vec[i], truth);
                const double acc_s = mynah_vecmath_ulp(ref[i], truth);
                const double agree = mynah_vecmath_ulp(vec[i], (double)ref[i]);
                if (acc_v > worst_accuracy) worst_accuracy = acc_v;
                if (acc_s > worst_accuracy) worst_accuracy = acc_s;
                if (agree > worst_agreement) worst_agreement = agree;
            }
        }
        if (worst_accuracy > 2.0)
            VEC_FAIL("tanh accuracy %.2f ulp against the double reference, "
                     "budget 2.0 (%s)", worst_accuracy, mynah_vecmath_isa());
        if (worst_agreement > 2.0)
            VEC_FAIL("tanh scalar and %s disagree by %.2f ulp, budget 2.0",
                     mynah_vecmath_isa(), worst_agreement);
    }

    /* tanh special values, by bits where the sign of a zero matters. */
    {
        const float specials[] = {0.0f, -0.0f, 1.0e-40f, -1.0e-40f,
                                  MYNAH_TANH_KNEE, -MYNAH_TANH_KNEE,
                                  9.0109f, 9.0111f, 1.0e30f, -1.0e30f,
                                  INFINITY, -INFINITY, NAN};
        const size_t count = sizeof(specials) / sizeof(specials[0]);
        float got[sizeof(specials) / sizeof(specials[0])];
        float got_scalar[sizeof(specials) / sizeof(specials[0])];
        /* A BLOCK-sized buffer so the vector body -- not the scalar tail --
         * is what runs on these: a special-value test that only ever reaches
         * the tail loop proves nothing about the kernel. */
        float wide_in[BLOCK], wide_out[BLOCK];
        for (size_t i = 0; i < BLOCK; ++i) wide_in[i] = specials[i % count];
        mynah_tanh_f32(wide_in, wide_out, BLOCK);
        mynah_tanh_f32(specials, got, count);
        mynah_tanh_f32_scalar(specials, got_scalar, count);
        /* WHERE BIT IDENTITY IS OWED, AND WHERE IT IS NOT.
         *
         * The vector body and the scalar body evaluate the same polynomial
         * with the same constants, but the compiler is free to contract the
         * C one into an FMA and is not free to touch the intrinsics, so in
         * the polynomial and exp regions they may land one ulp apart.  They
         * were bit identical on clang and one ulp apart on gcc 15.2 -- which
         * is exactly why a blanket bit-identity assertion is the wrong gate:
         * it passes on the machine you wrote it on and fails on the one you
         * ship to, for no defect.
         *
         * Bits ARE owed where the answer is produced by selection rather than
         * by arithmetic: both zeros, the saturated region past the knee, the
         * infinities and NaN.  Those are the cases a sign or a branch bug
         * shows up in, and they are asserted to the bit. */
        for (size_t i = 0; i < BLOCK; ++i) {
            const float x = specials[i % count];
            if (x != x) {                              /* NaN */
                if (wide_out[i] == wide_out[i])
                    VEC_FAIL("tanh(NaN) = %.9g, want NaN", (double)wide_out[i]);
                continue;
            }
            /* A DENORMAL ON A FLUSH-TO-ZERO BUILD IS NOT COMPARABLE, and
             * this is not a defect in either body.  The vector body runs at
             * run time with FPCR.FZ set and sees zero; the scalar body is in
             * this same translation unit, so the compiler may constant-fold
             * it at COMPILE time, where flush-to-zero does not exist, and
             * return the denormal.  Whether a denormal survives therefore
             * depends on whether the optimiser could fold it -- which no
             * kernel can control.  Both answers are accepted, and the only
             * thing asserted is that each is one of the two. */
            if (vec_is_subnormal(x) && mynah_vecmath_denormals_flush()) {
                const int ok_v = vec_is_zero(wide_out[i]) ||
                                 wide_out[i] == x;
                const int ok_s = vec_is_zero(got_scalar[i % count]) ||
                                 got_scalar[i % count] == x;
                if (!ok_v || !ok_s)
                    VEC_FAIL("tanh of a denormal on a flush-to-zero build "
                             "must be that denormal or a zero; got %.9g / "
                             "%.9g", (double)wide_out[i],
                             (double)got_scalar[i % count]);
                continue;
            }
            const float ax = fabsf(x);
            const int exact = vec_is_zero(x) || (ax > MYNAH_TANH_KNEE);
            if (exact) {
                if (!vec_same_bits(wide_out[i], got_scalar[i % count])) {
                    uint32_t vb, sb;
                    memcpy(&vb, &wide_out[i], sizeof vb);
                    memcpy(&sb, &got_scalar[i % count], sizeof sb);
                    VEC_FAIL("tanh(%.9g): vector body 0x%08x, scalar 0x%08x -- "
                             "this case is produced by selection, not "
                             "arithmetic, so the bits must match",
                             (double)x, vb, sb);
                }
            } else if (mynah_vecmath_ulp(wide_out[i],
                                        (double)got_scalar[i % count]) > 2.0) {
                VEC_FAIL("tanh(%.9g): vector %.9g, scalar %.9g, more than "
                         "2 ulp apart", (double)x, (double)wide_out[i],
                         (double)got_scalar[i % count]);
            }
        }
        if (!vec_same_bits(got_scalar[0], 0.0f))
            VEC_FAIL("tanh(+0) is not +0");
        if (!vec_same_bits(got_scalar[1], -0.0f))
            VEC_FAIL("tanh(-0) = %.9g, want -0 (sign of zero lost)",
                     (double)got_scalar[1]);
        /* tanh of a denormal is the denormal -- unless the BUILD flushes
         * denormals, in which case the argument is already zero before the
         * kernel sees it and the only correct answer is a zero of the same
         * sign.  Both are asserted, against whichever the platform does. */
        if (mynah_vecmath_denormals_flush()) {
            if (!(vec_is_zero(got_scalar[2]) ||
                  vec_same_bits(got_scalar[2], 1.0e-40f)) ||
                !(vec_is_zero(got_scalar[3]) ||
                  vec_same_bits(got_scalar[3], -1.0e-40f)))
                VEC_FAIL("this build flushes denormals, so tanh of one must "
                         "be that denormal or a zero; got %.9g / %.9g",
                         (double)got_scalar[2], (double)got_scalar[3]);
        } else if (!vec_same_bits(got_scalar[2], 1.0e-40f) ||
                   !vec_same_bits(got_scalar[3], -1.0e-40f)) {
            VEC_FAIL("tanh of a denormal must return the denormal");
        }
        if (got_scalar[10] != 1.0f || got_scalar[11] != -1.0f)
            VEC_FAIL("tanh(+-Inf) = %.9g / %.9g, want +-1",
                     (double)got_scalar[10], (double)got_scalar[11]);
        if (got_scalar[8] != 1.0f || got_scalar[9] != -1.0f)
            VEC_FAIL("tanh saturation lost past the knee");
    }

    /* --- a/b/c for sin ------------------------------------------------- */
    {
        /* The last three ranges are the point of this block.  A vector sine
         * that never reduces past a few radians looks perfect on [-pi, pi]
         * and is worthless at 6000. */
        static const float lo[] = {-1.0e-20f, -3.14159f, -6.3f,  -20.0f,
                                   -400.0f,   -8191.0f,  6283.0f, 8100.0f,
                                   1000.0f,   -1.0e-6f, 60000.0f, -65000.0f};
        static const float hi[] = {1.0e-20f,  3.14159f,  6.3f,   20.0f,
                                   400.0f,    8191.0f,   6284.0f, 8191.9f,
                                   1010.0f,   1.0e-6f, 60010.0f, -64990.0f};
        double worst_accuracy = 0.0, worst_agreement = 0.0;
        float worst_at = 0.0f;
        for (size_t r = 0; r < sizeof(lo) / sizeof(lo[0]); ++r) {
            for (size_t i = 0; i < BLOCK; ++i)
                in[i] = vec_sample(i, lo[r], hi[r], BLOCK);
            mynah_sin_f32(in, vec, BLOCK);
            mynah_sin_f32_scalar(in, ref, BLOCK);
            for (size_t i = 0; i < BLOCK; ++i) {
                const double truth = sin((double)in[i]);
                /* Near a zero of sin the ULP of the RESULT explodes for any
                 * implementation, because the true value is tiny while the
                 * argument is not; the meaningful bound there is absolute.
                 * Both are checked, and the looser of the two must hold. */
                const double acc_v = mynah_vecmath_ulp(vec[i], truth);
                const double acc_s = mynah_vecmath_ulp(ref[i], truth);
                const double abs_v = fabs((double)vec[i] - truth);
                const double abs_s = fabs((double)ref[i] - truth);
                const double v = (abs_v <= 1.0e-7) ? 0.0 : acc_v;
                const double sc = (abs_s <= 1.0e-7) ? 0.0 : acc_s;
                if (v > worst_accuracy) { worst_accuracy = v; worst_at = in[i]; }
                if (sc > worst_accuracy) { worst_accuracy = sc; worst_at = in[i]; }
                const double agree = fabs((double)vec[i] - (double)ref[i]);
                if (agree > worst_agreement) worst_agreement = agree;
            }
        }
        if (worst_accuracy > 3.0)
            VEC_FAIL("sin accuracy %.2f ulp near x = %.9g, budget 3.0 (%s)",
                     worst_accuracy, (double)worst_at, mynah_vecmath_isa());
        if (worst_agreement > 3.0e-7)
            VEC_FAIL("sin scalar and %s disagree by %.3g absolute, budget 3e-7",
                     mynah_vecmath_isa(), worst_agreement);
    }

    /* sin special values and the libm handover past the reduction limit. */
    {
        const float specials[] = {0.0f, -0.0f, 1.0e-40f, -1.0e-40f,
                                  65535.9f, -65535.9f, 65536.0f, -65536.0f,
                                  100000.0f, -1.0e12f, INFINITY, NAN};
        const size_t count = sizeof(specials) / sizeof(specials[0]);
        float wide_in[BLOCK], wide_out[BLOCK], wide_ref[BLOCK];
        for (size_t i = 0; i < BLOCK; ++i) wide_in[i] = specials[i % count];
        mynah_sin_f32(wide_in, wide_out, BLOCK);
        mynah_sin_f32_scalar(wide_in, wide_ref, BLOCK);
        for (size_t i = 0; i < BLOCK; ++i) {
            if (wide_in[i] != wide_in[i] || wide_in[i] > 3.0e38f) {
                /* sin(NaN) and sin(Inf) are both NaN and both come from libm */
                if (wide_out[i] == wide_out[i])
                    VEC_FAIL("sin(%.9g) = %.9g, want NaN", (double)wide_in[i],
                             (double)wide_out[i]);
                continue;
            }
            /* Past the reduction limit BOTH paths call libm, so those must
             * be bit identical; so must a zero, which is produced by sign
             * selection.  In between, the polynomial may be contracted
             * differently in C than in intrinsics -- see the tanh note -- so
             * the bound there is absolute. */
            if (vec_is_subnormal(wide_in[i]) &&
                mynah_vecmath_denormals_flush()) {
                continue;   /* see the tanh note above */
            }
            const float ax = fabsf(wide_in[i]);
            if (ax >= MYNAH_SIN_LIMIT || vec_is_zero(wide_in[i])) {
                if (!vec_same_bits(wide_out[i], wide_ref[i]))
                    VEC_FAIL("sin(%.9g): vector %.9g vs scalar %.9g -- both "
                             "must come from the same place here",
                             (double)wide_in[i], (double)wide_out[i],
                             (double)wide_ref[i]);
            } else if (fabsf(wide_out[i] - wide_ref[i]) > 3.0e-7f) {
                VEC_FAIL("sin(%.9g): vector %.9g vs scalar %.9g",
                         (double)wide_in[i], (double)wide_out[i],
                         (double)wide_ref[i]);
            }
        }
        if (!vec_same_bits(wide_ref[0], 0.0f))
            VEC_FAIL("sin(+0) is not +0");
        if (!vec_same_bits(wide_ref[1], -0.0f))
            VEC_FAIL("sin(-0) = %.9g, want -0", (double)wide_ref[1]);
        /* The handover itself: 100000 is past the limit, and libm's answer
         * must be reproduced exactly by both of our paths. */
        float one_in[1] = {100000.0f}, one_out[1];
        mynah_sin_f32(one_in, one_out, 1u);
        if (!vec_same_bits(one_out[0], sinf(100000.0f)))
            VEC_FAIL("sin(1e5) = %.9g, libm says %.9g -- the handover past "
                     "the argument-reduction limit is not happening",
                     (double)one_out[0], (double)sinf(100000.0f));
    }

    /* --- the Snake row, which is what the codec actually calls --------- */
    {
        /* Alphas from the small values a learned Snake actually carries, and
         * one large one, because 1/(alpha + 1e-9) is the term that blows up
         * when alpha is small and the test must have seen that. */
        static const float alphas[] = {1.0e-3f, 0.05f, 0.5f, 1.0f, 3.7f, 40.0f};
        double worst = 0.0;
        for (size_t a = 0; a < sizeof(alphas) / sizeof(alphas[0]); ++a) {
            float row_v[BLOCK], row_s[BLOCK];
            double truth[BLOCK], scale[BLOCK];
            for (size_t i = 0; i < BLOCK; ++i) {
                in[i] = vec_sample(i, -8.0f, 8.0f, BLOCK);
                row_v[i] = in[i];
                row_s[i] = in[i];
                const double s = sin((double)alphas[a] * (double)in[i]);
                truth[i] = (double)in[i] +
                           s * s / ((double)alphas[a] + 1.0e-9);
                /* v and s^2/alpha can cancel, so measure against the scale
                 * of the terms rather than the size of the answer. */
                scale[i] = fabs((double)in[i]) +
                           s * s / ((double)alphas[a] + 1.0e-9);
                if (scale[i] < 1.0e-30) scale[i] = 1.0e-30;
            }
            mynah_snake_row_f32(row_v, BLOCK, alphas[a]);
            mynah_snake_row_f32_scalar(row_s, BLOCK, alphas[a]);
            for (size_t i = 0; i < BLOCK; ++i) {
                const double uv = fabs((double)row_v[i] - truth[i]) / scale[i];
                const double us = fabs((double)row_s[i] - truth[i]) / scale[i];
                if (uv > worst) worst = uv;
                if (us > worst) worst = us;
                if (fabs((double)row_v[i] - (double)row_s[i]) / scale[i] > 4.0e-7)
                    VEC_FAIL("snake alpha=%.6g: %s row[%zu] = %.9g, scalar "
                             "row = %.9g", (double)alphas[a],
                             mynah_vecmath_isa(), i, (double)row_v[i],
                             (double)row_s[i]);
            }
        }
        if (worst > 4.0e-7)
            VEC_FAIL("snake accuracy %.2e relative to the scale of its terms, "
                     "budget 4e-7", worst);
        /* A length that is not a multiple of the vector width, so the tail
         * loop is exercised and must agree with the body. */
        for (size_t len = 1; len <= 19u; ++len) {
            float row_v[19], row_s[19];
            for (size_t i = 0; i < len; ++i) {
                row_v[i] = row_s[i] = vec_sample(i, -4.0f, 4.0f, 19u);
            }
            mynah_snake_row_f32(row_v, len, 0.7f);
            mynah_snake_row_f32_scalar(row_s, len, 0.7f);
            for (size_t i = 0; i < len; ++i)
                if (fabs((double)row_v[i] - (double)row_s[i]) >
                    4.0e-7 * (fabs((double)row_s[i]) + 1.0))
                    VEC_FAIL("snake tail length %zu differs at %zu", len, i);
        }
    }

    /* --- GELU: the array form must be the elementwise form ------------- */
    {
        float values[BLOCK], expect[BLOCK], source[BLOCK];
        for (size_t i = 0; i < BLOCK; ++i) {
            source[i] = vec_sample(i, -12.0f, 12.0f, BLOCK);
            values[i] = source[i];
            expect[i] = mynah_gelu_tanh(source[i]);
        }
        mynah_gelu_tanh_array(values, BLOCK, NULL);
        for (size_t i = 0; i < BLOCK; ++i) {
            /* |GELU(x)| <= |x|, so 1 + |x| is the scale the answer lives on.
             * The ULP of the RESULT is the wrong unit: around x = -0.28 the
             * GELU is an order of magnitude smaller than x, so a correctly
             * rounded tanh still moves the result several ulp.  Found by
             * running this very check under the AVX2 body, where the FMA
             * spelling lands a few ulp from the C one and a 4-ulp budget
             * failed for no defect. */
            const double scale = 1.0 + fabs((double)source[i]);
            if (fabs((double)values[i] - (double)expect[i]) > 1.0e-6 * scale)
                VEC_FAIL("gelu array[%zu] = %.9g, elementwise = %.9g (x = "
                         "%.9g) -- the two spellings of the same GELU have "
                         "drifted apart", i, (double)values[i],
                         (double)expect[i], (double)source[i]);
        }
    }
    return 0;
}
#undef VEC_FAIL

/* ---- moved out of graph.c by the E1 split ---------------------------------
 * These are kernels, not graph structure: a tanh-approximation GELU built on
 * the tanh above, and the NEON axpy that the single-row attention weighted
 * sum uses.
 *
 * WHAT CHANGED AND WHY IT MATTERS.  The array form used to call Accelerate's
 * vvtanhf when it was given scratch, and the elementwise form has always
 * called libm tanhf.  engine_magpie.c calls BOTH (:177 elementwise, :1064 and
 * :1997 array), so one binary held two different GELUs, and on Linux it held
 * a third -- the scalar loop, because vvtanhf does not exist there.  All
 * three are now mynah_tanh_core, so the elementwise form, the array form,
 * macOS and Linux are one function.  `scratch` is accepted and ignored: the
 * vector tanh works in registers and never needed a staging buffer.
 * .work/accelerate-only-kernels.md. */

float mynah_gelu_tanh(float x) {
    const float cubic = x * x * x;
    const float inner = 0.7978845608028654f * (x + 0.044715f * cubic);
    return 0.5f * x * (1.0f + mynah_tanh_core(inner));
}

void mynah_gelu_tanh_array(float *values, size_t length, float *scratch) {
    (void)scratch; /* kept in the signature; no longer read.  See above. */
    size_t i = 0;
#if defined(MYNAH_VECMATH_NEON)
    {
        const float32x4_t half = vdupq_n_f32(0.5f);
        const float32x4_t one = vdupq_n_f32(1.0f);
        const float32x4_t k = vdupq_n_f32(0.7978845608028654f);
        const float32x4_t c = vdupq_n_f32(0.044715f);
        for (; i + 4u <= length; i += 4u) {
            const float32x4_t x = vld1q_f32(values + i);
            const float32x4_t cubic = vmulq_f32(vmulq_f32(x, x), x);
            const float32x4_t inner =
                vmulq_f32(k, vmlaq_f32(x, c, cubic));
            const float32x4_t t = neon_tanh(inner);
            vst1q_f32(values + i,
                      vmulq_f32(vmulq_f32(half, x), vaddq_f32(one, t)));
        }
    }
#elif defined(MYNAH_VECMATH_AVX2)
    {
        const __m256 half = _mm256_set1_ps(0.5f);
        const __m256 one = _mm256_set1_ps(1.0f);
        const __m256 k = _mm256_set1_ps(0.7978845608028654f);
        const __m256 c = _mm256_set1_ps(0.044715f);
        for (; i + 8u <= length; i += 8u) {
            const __m256 x = _mm256_loadu_ps(values + i);
            const __m256 cubic = _mm256_mul_ps(_mm256_mul_ps(x, x), x);
            const __m256 inner =
                _mm256_mul_ps(k, _mm256_fmadd_ps(c, cubic, x));
            const __m256 t = avx2_tanh(inner);
            _mm256_storeu_ps(values + i,
                _mm256_mul_ps(_mm256_mul_ps(half, x), _mm256_add_ps(one, t)));
        }
    }
#endif
    for (; i < length; ++i) values[i] = mynah_gelu_tanh(values[i]);
}

/* ---- NEON-accelerated attention primitives --------------------------------
 * The single-row decoder attention attn·V weighted sum (hw = 64 elements per
 * position) is vectorised with 4-wide NEON FMA.  The Q·K dot product stays
 * scalar to preserve the exact greedy argmax path (NEON lane reordering
 * changes softmax scores enough to flip tokens).  The axpy accumulation
 * order change is absorbed by downstream layers without affecting EOS.
 * A scalar fallback is always compiled for portability. */

/* out[0..n) += weight * src[0..n) */
void mynah_axpy_f32(float *out, const float *src, float weight, size_t n) {
#if defined(MYNAH_KERNELS_NEON)
    const float32x4_t w = vdupq_n_f32(weight);
    size_t i = 0;
    for (; i + 4u <= n; i += 4u) {
        float32x4_t o = vld1q_f32(out + i);
        o = vfmaq_f32(o, w, vld1q_f32(src + i));
        vst1q_f32(out + i, o);
    }
    for (; i < n; ++i) out[i] += weight * src[i];
#else
    for (size_t i = 0; i < n; ++i) out[i] += weight * src[i];
#endif
}

int mynah_gelu_self_test(char *error, size_t error_capacity) {
    /* This used to be a no-op without Accelerate, which meant the gate that
     * existed to police the array GELU did nothing at all on the production
     * target.  It runs everywhere now, and it compares the array form against
     * the elementwise form -- the two spellings engine_magpie.c actually
     * calls -- across the infinities, the NaN and the negative zero. */
    float values[] = {
        -INFINITY, -10.0f, -3.0f, -1.0f, -0.25f, -0.0f, 0.0f,
        0.125f, 0.5f, 1.0f, 2.0f, 3.0f, 8.0f, INFINITY, NAN,
        -6.75f, 0.03125f, 4.5f, -2.125f
    };
    const size_t count = sizeof(values) / sizeof(values[0]);
    float expected[sizeof(values) / sizeof(values[0])];
    float scratch[sizeof(values) / sizeof(values[0])];
    for (size_t i = 0; i < count; ++i) expected[i] = mynah_gelu_tanh(values[i]);
    /* Passing scratch deliberately: the parameter is ignored now, and this
     * asserts that it is ignored rather than silently re-read. */
    mynah_gelu_tanh_array(values, count, scratch);
    for (size_t i = 0; i < count; ++i) {
        if (isnan(expected[i])) {
            if (isnan(values[i])) continue;
        } else if (isinf(expected[i])) {
            if (isinf(values[i]) && signbit(values[i]) == signbit(expected[i])) continue;
        } else {
            const float tolerance = 2.0e-6f * (1.0f + fabsf(expected[i]));
            if (fabsf(values[i] - expected[i]) <= tolerance) continue;
        }
        if (error != NULL && error_capacity > 0) {
            snprintf(error, error_capacity,
                     "array GELU mismatch at %zu: got %.9g expected %.9g",
                     i, (double)values[i], (double)expected[i]);
        }
        return -1;
    }
    return mynah_vecmath_self_test(error, error_capacity);
}

/* ======================================================================
 * Dispatch predicates
 * ====================================================================== */
static int probe_gelu_vector(const char **why) {
    const int on = mynah_gelu_vector_enabled();
#if defined(MYNAH_KERNELS_NEON)
    const char *impl = "NEON Pade [5/5] tanh";
#elif defined(MYNAH_KERNELS_AVX2)
    const char *impl = "AVX2 Pade [5/5] tanh";
#else
    const char *impl = "no vector GELU is compiled for this target";
#endif
    if (why != NULL) {
        static char text[300];
        snprintf(text, sizeof text,
                 "[predicate] mynah_gelu_vector_enabled(): %s (%s). This is "
                 "mynah_gelu_f32 ONLY -- the clamped Pade form. The array "
                 "GELU is kernel.transcendental below and is a different, "
                 "unclamped function", on ? "vector" : "scalar libm tanhf",
                 impl);
        *why = text;
    }
    return on;
}

/* ----------------------------------------------------------------------
 * The ISA kernel inventory (kernels.h)
 *
 * This is a fact about this translation unit and its siblings, so it is
 * written once, here, and the report calls it.  Today every bit is 0: there
 * is no `svfloat32_t` anywhere under src/, no `svmmla`, no `bfdot`/`bfmmla`,
 * and src/qmat.c stores weights as f32, f16, int8 or int4 but never bf16
 * (src/weights.c converts bf16 to f32 once at load).
 *
 * Note what is deliberately NOT tested here: __ARM_FEATURE_SVE.  That macro
 * says the compiler was allowed to emit SVE -- which -march=native does on the
 * Neoverse V2 box, for autovectorization -- and says nothing about whether a
 * hand-written kernel exists.  Confusing "the flag was passed" with "a kernel
 * dispatches on it" is the exact error that put an AVX-512 VNNI claim in
 * README.md next to a file with no _mm512_* in it.  So the answer is a
 * constant that a future kernel author edits, not a macro test.
 * ---------------------------------------------------------------------- */
unsigned mynah_kernels_isa_kernels(void) {
    return 0u;
}

const char *mynah_kernels_isa_missing_reason(unsigned bit) {
    switch (bit) {
    case MYNAH_KERNELS_ISA_SVE:
        return "no svfloat32_t kernel; src/kernels.c is NEON-only and "
               "fixed at 4 lanes";
    case MYNAH_KERNELS_ISA_SVE2:
        return "no SVE2 kernel and no vector-length-agnostic path to widen";
    case MYNAH_KERNELS_ISA_SVEI8MM:
        return "no SVE SMMLA kernel; src/qmat.c reaches i8mm only through "
               "the fixed-width NEON matvec_q8_pair_i8mm";
    case MYNAH_KERNELS_ISA_SVEBF16:
        return "no SVE BFMMLA/BFDOT kernel, and no bf16 weight type to feed it";
    case MYNAH_KERNELS_ISA_BF16:
        return "no bfdot/bfmmla and no bf16 weight type; src/weights.c "
               "widens bf16 to f32 at load and f32 paths stay f32";
    default:
        return "unknown ISA bit";
    }
}

/* How wide this CPU's SVE vector actually is, in bits, or 0 when it cannot be
 * asked.  Read with prctl rather than svcntb() so that querying the width costs
 * nothing at build time: svcntb() would require the translation unit to be
 * compiled with SVE enabled, which is a bigger commitment than a report line
 * deserves.
 *
 * WHY THE REPORT NEEDS IT.  "supported=yes and we idle it" invites the reader to
 * assume idle width.  On Neoverse-V2 -- this project's production CPU -- the SVE
 * vector measures 128 bits, exactly NEON's, so an SVE kernel there would buy
 * predication and vector-length-agnostic code and NOT throughput.  Three of the
 * five idle rows are that case, and a reader who cannot see the width has no way
 * to tell them from the one that is a real 4x (bf16 -> BFMMLA).  See
 * .work/bf16-native-weights.md. */
static unsigned sve_vector_bits(void) {
#if defined(__aarch64__) && defined(__linux__)
    /* PR_SVE_GET_VL; the length in bytes is the low 16 bits of the result. */
    const int vl = prctl(51);
    if (vl < 0) return 0u;
    return (unsigned)(vl & 0xffff) * 8u;
#else
    return 0u;
#endif
}

/* One probe body for all five rows.  It answers from the inventory above, so
 * `resolved` can never disagree with what the binary contains. */
static int probe_isa_bit(unsigned bit, const char **why) {
    const int on = (mynah_kernels_isa_kernels() & bit) != 0u;
    if (why != NULL) {
        static char text[5][360];
        static const unsigned bits[5] = {
            MYNAH_KERNELS_ISA_SVE, MYNAH_KERNELS_ISA_SVE2,
            MYNAH_KERNELS_ISA_SVEI8MM, MYNAH_KERNELS_ISA_SVEBF16,
            MYNAH_KERNELS_ISA_BF16
        };
        int slot = 0;
        for (int i = 0; i < 5; ++i) if (bits[i] == bit) slot = i;
        /* The width note goes on the SVE rows only: bf16 is reachable from NEON
         * (BFMMLA), so its width is not the question there. */
        const int is_sve = (bit == MYNAH_KERNELS_ISA_SVE ||
                            bit == MYNAH_KERNELS_ISA_SVE2 ||
                            bit == MYNAH_KERNELS_ISA_SVEI8MM ||
                            bit == MYNAH_KERNELS_ISA_SVEBF16);
        const unsigned vl = is_sve ? sve_vector_bits() : 0u;
        char width[160];
        width[0] = '\0';
        if (vl == 128u) {
            snprintf(width, sizeof width,
                     " This CPU's SVE vector is 128 bits -- the same width as "
                     "NEON -- so a kernel here buys predication and "
                     "vector-length-agnostic code, NOT throughput");
        } else if (vl > 128u) {
            snprintf(width, sizeof width,
                     " This CPU's SVE vector is %u bits against NEON's 128, so "
                     "a kernel here is a %ux widening", vl, vl / 128u);
        }
        snprintf(text[slot], sizeof text[slot],
                 "[predicate] mynah_kernels_isa_kernels(): %s -- %s. "
                 "supported=yes here means the CPU has the unit and we idle it.%s",
                 on ? "kernel present" : "NOT IMPLEMENTED",
                 mynah_kernels_isa_missing_reason(bit), width);
        *why = text[slot];
    }
    return on;
}

static int probe_sve(const char **why)     { return probe_isa_bit(MYNAH_KERNELS_ISA_SVE, why); }
static int probe_sve2(const char **why)    { return probe_isa_bit(MYNAH_KERNELS_ISA_SVE2, why); }
static int probe_svei8mm(const char **why) { return probe_isa_bit(MYNAH_KERNELS_ISA_SVEI8MM, why); }
static int probe_svebf16(const char **why) { return probe_isa_bit(MYNAH_KERNELS_ISA_SVEBF16, why); }
static int probe_bf16(const char **why)    { return probe_isa_bit(MYNAH_KERNELS_ISA_BF16, why); }

void mynah_kernels_dispatch_probes(void) {
    /* NOTE FOR THE DISPATCH LANE (src/dispatch.c is not this lane's file):
     * there is no row id for the transcendentals, and a probe registered
     * under a new id is silently dropped because the row table lives there.
     * Two one-line changes are owed over there after this change:
     *   - add a `kernel.transcendental` row, whose answer is
     *     mynah_vecmath_isa() plus mynah_vecmath_denormals_flush();
     *   - `blas.accelerate` at dispatch.c:754 still claims Accelerate
     *     supplies "vvtanhf for the array GELU".  It no longer does; after
     *     E4-16d Accelerate supplies cblas_sgemm and nothing else in
     *     src/kernels.c.  The row now overstates the dependency it exists to
     *     make visible.
     * Until then the facts are carried by the two rows below. */
    mynah_dispatch_register_probe("kernel.gelu_vector", probe_gelu_vector);
    mynah_dispatch_register_probe("isa.arm.sve", probe_sve);
    mynah_dispatch_register_probe("isa.arm.sve2", probe_sve2);
    mynah_dispatch_register_probe("isa.arm.svei8mm", probe_svei8mm);
    mynah_dispatch_register_probe("isa.arm.svebf16", probe_svebf16);
    mynah_dispatch_register_probe("isa.arm.bf16", probe_bf16);
}
