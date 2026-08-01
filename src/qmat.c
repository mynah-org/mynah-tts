#include "qmat.h"
#include "kernels.h"
#include "threads.h"

#include <pthread.h>

#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if !defined(MYNAH_DISABLE_SIMD) && defined(__ARM_FEATURE_DOTPROD)
#include <arm_neon.h>
#define MYNAH_QMAT_DOTPROD 1
#endif
#if !defined(MYNAH_DISABLE_SIMD) && defined(__AVX2__)
#include <immintrin.h>
#define MYNAH_QMAT_AVX2 1
#endif
/* IEEE half weights need NEON's f16<->f32 converts.  Where they are missing the
 * F16 cache simply refuses the tensor and the caller falls back to exact f32,
 * the same contract INT4 uses for a shape it cannot represent. */
#if !defined(MYNAH_DISABLE_SIMD) && defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>
#define MYNAH_QMAT_F16 1
#endif

/* count at/below this uses the native int dot; above it falls back to the f32
 * BLAS matmul (the prefill, already fast and kept bit-exact). */
#define QMAT_SMALL_COUNT 16
#define QMAT_K_MAX 8192
#define QMAT_Q4_GROUP 32

enum { QMAT_F32 = 0, QMAT_INT8 = 1, QMAT_INT4 = 2, QMAT_F16 = 3 };

/* ------------------------------------------------------------- quantizers */
static void quantize_weight_int8(const float *w, size_t n, size_t k,
                                 int8_t *q, float *scales) {
    for (size_t i = 0; i < n; ++i) {
        const float *row = w + i * k;
        float amax = 0.0f;
        for (size_t j = 0; j < k; ++j) {
            const float a = fabsf(row[j]);
            if (a > amax) amax = a;
        }
        const float scale = amax > 0.0f ? amax / 127.0f : 1.0f;
        scales[i] = scale;
        const float inv = 1.0f / scale;
        int8_t *qrow = q + i * k;
        for (size_t j = 0; j < k; ++j) {
            const float v = row[j] * inv;
            qrow[j] = (int8_t)(v >= 0.0f ? v + 0.5f : v - 0.5f);
        }
    }
}

/* Per-group-of-32 symmetric INT4 (Q4_0 style): nibbles offset by +8, low nibble
 * = even index, high nibble = odd index; scales[i * k/32 + g]. */
static void quantize_weight_int4(const float *w, size_t n, size_t k,
                                 uint8_t *q, float *scales) {
    const size_t groups = k / QMAT_Q4_GROUP;
    for (size_t i = 0; i < n; ++i) {
        const float *row = w + i * k;
        uint8_t *qrow = q + i * (k / 2u);
        float *srow = scales + i * groups;
        for (size_t g = 0; g < groups; ++g) {
            const float *grp = row + g * QMAT_Q4_GROUP;
            float amax = 0.0f;
            for (size_t j = 0; j < QMAT_Q4_GROUP; ++j) {
                const float a = fabsf(grp[j]);
                if (a > amax) amax = a;
            }
            const float scale = amax > 0.0f ? amax / 7.0f : 1.0f;
            srow[g] = scale;
            const float inv = 1.0f / scale;
            for (size_t j = 0; j < QMAT_Q4_GROUP; j += 2) {
                const float v0 = grp[j] * inv;
                const float v1 = grp[j + 1] * inv;
                int q0 = (int)(v0 >= 0.0f ? v0 + 0.5f : v0 - 0.5f);
                int q1 = (int)(v1 >= 0.0f ? v1 + 0.5f : v1 - 0.5f);
                if (q0 < -8) q0 = -8;
                if (q0 > 7) q0 = 7;
                if (q1 < -8) q1 = -8;
                if (q1 > 7) q1 = 7;
                qrow[(g * QMAT_Q4_GROUP + j) / 2] = (uint8_t)((q0 + 8) | ((q1 + 8) << 4));
            }
        }
    }
}

/* Per-vector absmax activation quantization; returns the activation scale. */
static float quantize_act_int8(int8_t *qx, const float *x, size_t k) {
    float amax = 0.0f;
    for (size_t i = 0; i < k; ++i) {
        const float a = fabsf(x[i]);
        if (a > amax) amax = a;
    }
    if (amax == 0.0f) {
        memset(qx, 0, k);
        return 0.0f;
    }
    const float inv = 127.0f / amax;
    for (size_t i = 0; i < k; ++i) {
        const float v = x[i] * inv;
        int q = (int)(v >= 0.0f ? v + 0.5f : v - 0.5f);
        if (q > 127) q = 127;
        if (q < -127) q = -127;
        qx[i] = (int8_t)q;
    }
    return amax / 127.0f;
}

/* ---------------------------------------------------------- int dot kernels */
#if defined(MYNAH_QMAT_AVX2)
static int32_t dot_q8_i32_avx2(const int8_t *qx, const int8_t *w, size_t k) {
    __m256i acc = _mm256_setzero_si256();
    size_t j = 0;
    for (; j + 16u <= k; j += 16u) {
        const __m128i x0 = _mm_loadu_si128((const __m128i *)(qx + j));
        const __m128i w0 = _mm_loadu_si128((const __m128i *)(w + j));
        const __m256i x16_0 = _mm256_cvtepi8_epi16(x0);
        const __m256i w16_0 = _mm256_cvtepi8_epi16(w0);
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(x16_0, w16_0));
    }
    __m128i sum = _mm_add_epi32(_mm256_castsi256_si128(acc),
                                _mm256_extracti128_si256(acc, 1));
    sum = _mm_hadd_epi32(sum, sum);
    sum = _mm_hadd_epi32(sum, sum);
    int32_t result = _mm_cvtsi128_si32(sum);
    for (; j < k; ++j) result += (int32_t)qx[j] * (int32_t)w[j];
    return result;
}
#endif
static float dot_q8(const int8_t *qx, float sx, const int8_t *w, float ws, size_t k) {
#if defined(MYNAH_QMAT_DOTPROD)
    int32x4_t acc = vdupq_n_s32(0);
    size_t j = 0;
    for (; j + 16 <= k; j += 16) acc = vdotq_s32(acc, vld1q_s8(w + j), vld1q_s8(qx + j));
    int32_t s = vaddvq_s32(acc);
    for (; j < k; ++j) s += (int32_t)w[j] * (int32_t)qx[j];
    return (float)s * ws * sx;
#elif defined(MYNAH_QMAT_AVX2)
    const int32_t s = dot_q8_i32_avx2(qx, w, k);
    return (float)s * ws * sx;
#else
    int32_t s = 0;
    for (size_t j = 0; j < k; ++j) s += (int32_t)w[j] * (int32_t)qx[j];
    return (float)s * ws * sx;
#endif
}

/* qx is the int8 activation; q is the packed INT4 weight group row; scales has
 * one entry per group of 32.  k must be a multiple of 32. */
static float dot_q4(const int8_t *qx, float sx, const uint8_t *q, const float *scales, size_t k) {
    const size_t groups = k / QMAT_Q4_GROUP;
#if defined(MYNAH_QMAT_DOTPROD)
    const int8x16_t off = vdupq_n_s8(8);
    const uint8x16_t maskv = vdupq_n_u8(0x0F);
    float acc = 0.0f;
    for (size_t g = 0; g < groups; ++g) {
        const uint8x16_t b = vld1q_u8(q + g * 16);
        const int8x16_t lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(b, maskv)), off);
        const int8x16_t hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(b, 4)), off);
        const int8x16x2_t xg = vld2q_s8(qx + g * 32); /* val[0]=even, val[1]=odd */
        int32x4_t ig = vdotq_s32(vdotq_s32(vdupq_n_s32(0), lo, xg.val[0]), hi, xg.val[1]);
        acc += (float)vaddvq_s32(ig) * scales[g];
    }
    return acc * sx;
#else
    float acc = 0.0f;
    for (size_t g = 0; g < groups; ++g) {
        int32_t gi = 0;
        for (size_t j = 0; j < QMAT_Q4_GROUP; j += 2) {
            const uint8_t b = q[g * 16 + j / 2];
            const int lo = (int)(b & 0x0F) - 8;
            const int hi = (int)(b >> 4) - 8;
            gi += lo * (int32_t)qx[g * 32 + j] + hi * (int32_t)qx[g * 32 + j + 1];
        }
        acc += (float)gi * scales[g];
    }
    return acc * sx;
#endif
}

/* Decode is a stream of matrix-vector products.  On ARM, keep four independent
 * output rows in flight so SDOT latency is hidden and the quantized activation
 * vector is loaded once for four weight rows.  Each row retains the same
 * accumulation order as dot_q8/dot_q4; the scalar tail is the reference path. */
static void matvec_q8(float *out, const int8_t *qx, float sx,
                      const int8_t *weights, const float *scales,
                      const float *bias, size_t rows, size_t cols) {
    size_t row = 0;
#if defined(MYNAH_QMAT_DOTPROD)
    for (; row + 4u <= rows; row += 4u) {
        const int8_t *w0 = weights + row * cols;
        const int8_t *w1 = w0 + cols;
        const int8_t *w2 = w1 + cols;
        const int8_t *w3 = w2 + cols;
        int32x4_t a0 = vdupq_n_s32(0);
        int32x4_t a1 = vdupq_n_s32(0);
        int32x4_t a2 = vdupq_n_s32(0);
        int32x4_t a3 = vdupq_n_s32(0);
        size_t j = 0;
        for (; j + 16u <= cols; j += 16u) {
            const int8x16_t x = vld1q_s8(qx + j);
            a0 = vdotq_s32(a0, vld1q_s8(w0 + j), x);
            a1 = vdotq_s32(a1, vld1q_s8(w1 + j), x);
            a2 = vdotq_s32(a2, vld1q_s8(w2 + j), x);
            a3 = vdotq_s32(a3, vld1q_s8(w3 + j), x);
        }
        int32_t s0 = vaddvq_s32(a0);
        int32_t s1 = vaddvq_s32(a1);
        int32_t s2 = vaddvq_s32(a2);
        int32_t s3 = vaddvq_s32(a3);
        for (; j < cols; ++j) {
            const int32_t x = qx[j];
            s0 += (int32_t)w0[j] * x;
            s1 += (int32_t)w1[j] * x;
            s2 += (int32_t)w2[j] * x;
            s3 += (int32_t)w3[j] * x;
        }
        out[row] = (float)s0 * scales[row] * sx + (bias == NULL ? 0.0f : bias[row]);
        out[row + 1u] = (float)s1 * scales[row + 1u] * sx +
                        (bias == NULL ? 0.0f : bias[row + 1u]);
        out[row + 2u] = (float)s2 * scales[row + 2u] * sx +
                        (bias == NULL ? 0.0f : bias[row + 2u]);
        out[row + 3u] = (float)s3 * scales[row + 3u] * sx +
                        (bias == NULL ? 0.0f : bias[row + 3u]);
    }
#elif defined(MYNAH_QMAT_AVX2)
    for (; row + 4u <= rows; row += 4u) {
        const int32_t s0 = dot_q8_i32_avx2(qx, weights + row * cols, cols);
        const int32_t s1 = dot_q8_i32_avx2(qx, weights + (row + 1u) * cols, cols);
        const int32_t s2 = dot_q8_i32_avx2(qx, weights + (row + 2u) * cols, cols);
        const int32_t s3 = dot_q8_i32_avx2(qx, weights + (row + 3u) * cols, cols);
        out[row] = (float)s0 * scales[row] * sx + (bias == NULL ? 0.0f : bias[row]);
        out[row + 1u] = (float)s1 * scales[row + 1u] * sx +
                        (bias == NULL ? 0.0f : bias[row + 1u]);
        out[row + 2u] = (float)s2 * scales[row + 2u] * sx +
                        (bias == NULL ? 0.0f : bias[row + 2u]);
        out[row + 3u] = (float)s3 * scales[row + 3u] * sx +
                        (bias == NULL ? 0.0f : bias[row + 3u]);
    }
#endif
    for (; row < rows; ++row) {
        float value = dot_q8(qx, sx, weights + row * cols, scales[row], cols);
        if (bias != NULL) value += bias[row];
        out[row] = value;
    }
}

static void matvec_q4(float *out, const int8_t *qx, float sx,
                      const uint8_t *weights, const float *scales,
                      const float *bias, size_t rows, size_t cols) {
    size_t row = 0;
#if defined(MYNAH_QMAT_DOTPROD)
    const size_t groups = cols / QMAT_Q4_GROUP;
    const size_t packed_row = cols / 2u;
    const int8x16_t off = vdupq_n_s8(8);
    const uint8x16_t mask = vdupq_n_u8(0x0F);
    for (; row + 4u <= rows; row += 4u) {
        const uint8_t *w0 = weights + row * packed_row;
        const uint8_t *w1 = w0 + packed_row;
        const uint8_t *w2 = w1 + packed_row;
        const uint8_t *w3 = w2 + packed_row;
        const float *s0 = scales + row * groups;
        const float *s1 = s0 + groups;
        const float *s2 = s1 + groups;
        const float *s3 = s2 + groups;
        float a0 = 0.0f;
        float a1 = 0.0f;
        float a2 = 0.0f;
        float a3 = 0.0f;
        for (size_t group = 0; group < groups; ++group) {
            const int8x16x2_t x = vld2q_s8(qx + group * QMAT_Q4_GROUP);
#define Q4_DOT_ROW(weight, scale, accumulator) do { \
                const uint8x16_t packed = vld1q_u8((weight) + group * 16u); \
                const int8x16_t lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(packed, mask)), off); \
                const int8x16_t hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(packed, 4)), off); \
                const int32x4_t dot = vdotq_s32(vdotq_s32(vdupq_n_s32(0), lo, x.val[0]), \
                                                hi, x.val[1]); \
                (accumulator) += (float)vaddvq_s32(dot) * (scale)[group]; \
            } while (0)
            Q4_DOT_ROW(w0, s0, a0);
            Q4_DOT_ROW(w1, s1, a1);
            Q4_DOT_ROW(w2, s2, a2);
            Q4_DOT_ROW(w3, s3, a3);
#undef Q4_DOT_ROW
        }
        out[row] = a0 * sx + (bias == NULL ? 0.0f : bias[row]);
        out[row + 1u] = a1 * sx + (bias == NULL ? 0.0f : bias[row + 1u]);
        out[row + 2u] = a2 * sx + (bias == NULL ? 0.0f : bias[row + 2u]);
        out[row + 3u] = a3 * sx + (bias == NULL ? 0.0f : bias[row + 3u]);
    }
#endif
    for (; row < rows; ++row) {
        float value = dot_q4(qx, sx, weights + row * (cols / 2u),
                             scales + row * (cols / QMAT_Q4_GROUP), cols);
        if (bias != NULL) value += bias[row];
        out[row] = value;
    }
}

#if defined(MYNAH_QMAT_F16)
/* Weights as IEEE half; activation and accumulation stay f32.  Decode is bound
 * by weight bytes, so halving them is close to halving the time -- measured
 * 2.4-2.8x against Accelerate sgemv on a working set too large to cache.
 *
 * Unlike INT8/INT4 this does not quantize the activation, and it needs no
 * scales: f16 carries its own exponent.  On Magpie weights the mean relative
 * error is 1.8e-4 against 1.7e-2 for INT8 (~96x more accurate), and the largest
 * weight is about 6 against the 65504 f16 limit, so overflow is not a concern.
 *
 * Four rows in flight so the activation is read once per four weight rows,
 * matching matvec_q8.  Each row accumulates into its own pair of vectors, so a
 * row's reduction order never depends on how the rows are partitioned. */
static void matvec_f16(float *out, const float *x, const __fp16 *weights,
                       const float *bias, size_t rows, size_t cols) {
    size_t row = 0;
    for (; row + 4u <= rows; row += 4u) {
        const __fp16 *w0 = weights + row * cols;
        const __fp16 *w1 = w0 + cols;
        const __fp16 *w2 = w1 + cols;
        const __fp16 *w3 = w2 + cols;
        float32x4_t a0 = vdupq_n_f32(0.0f), b0 = vdupq_n_f32(0.0f);
        float32x4_t a1 = vdupq_n_f32(0.0f), b1 = vdupq_n_f32(0.0f);
        float32x4_t a2 = vdupq_n_f32(0.0f), b2 = vdupq_n_f32(0.0f);
        float32x4_t a3 = vdupq_n_f32(0.0f), b3 = vdupq_n_f32(0.0f);
        size_t j = 0;
        for (; j + 8u <= cols; j += 8u) {
            const float32x4_t xl = vld1q_f32(x + j);
            const float32x4_t xh = vld1q_f32(x + j + 4u);
            const float16x8_t v0 = vld1q_f16(w0 + j);
            const float16x8_t v1 = vld1q_f16(w1 + j);
            const float16x8_t v2 = vld1q_f16(w2 + j);
            const float16x8_t v3 = vld1q_f16(w3 + j);
            a0 = vfmaq_f32(a0, vcvt_f32_f16(vget_low_f16(v0)), xl);
            b0 = vfmaq_f32(b0, vcvt_f32_f16(vget_high_f16(v0)), xh);
            a1 = vfmaq_f32(a1, vcvt_f32_f16(vget_low_f16(v1)), xl);
            b1 = vfmaq_f32(b1, vcvt_f32_f16(vget_high_f16(v1)), xh);
            a2 = vfmaq_f32(a2, vcvt_f32_f16(vget_low_f16(v2)), xl);
            b2 = vfmaq_f32(b2, vcvt_f32_f16(vget_high_f16(v2)), xh);
            a3 = vfmaq_f32(a3, vcvt_f32_f16(vget_low_f16(v3)), xl);
            b3 = vfmaq_f32(b3, vcvt_f32_f16(vget_high_f16(v3)), xh);
        }
        float s0 = vaddvq_f32(vaddq_f32(a0, b0));
        float s1 = vaddvq_f32(vaddq_f32(a1, b1));
        float s2 = vaddvq_f32(vaddq_f32(a2, b2));
        float s3 = vaddvq_f32(vaddq_f32(a3, b3));
        for (; j < cols; ++j) {
            const float xv = x[j];
            s0 += (float)w0[j] * xv;
            s1 += (float)w1[j] * xv;
            s2 += (float)w2[j] * xv;
            s3 += (float)w3[j] * xv;
        }
        out[row] = s0 + (bias == NULL ? 0.0f : bias[row]);
        out[row + 1u] = s1 + (bias == NULL ? 0.0f : bias[row + 1u]);
        out[row + 2u] = s2 + (bias == NULL ? 0.0f : bias[row + 2u]);
        out[row + 3u] = s3 + (bias == NULL ? 0.0f : bias[row + 3u]);
    }
    for (; row < rows; ++row) {
        const __fp16 *w = weights + row * cols;
        float32x4_t a = vdupq_n_f32(0.0f), b = vdupq_n_f32(0.0f);
        size_t j = 0;
        for (; j + 8u <= cols; j += 8u) {
            const float16x8_t v = vld1q_f16(w + j);
            a = vfmaq_f32(a, vcvt_f32_f16(vget_low_f16(v)), vld1q_f32(x + j));
            b = vfmaq_f32(b, vcvt_f32_f16(vget_high_f16(v)), vld1q_f32(x + j + 4u));
        }
        float s = vaddvq_f32(vaddq_f32(a, b));
        for (; j < cols; ++j) s += (float)w[j] * x[j];
        out[row] = s + (bias == NULL ? 0.0f : bias[row]);
    }
}
#endif

/* Quantized decode is memory bound on the packed weights exactly like the f32
 * path, and one core only reaches a fraction of DRAM bandwidth.  Every output
 * row is an independent dot product, so splitting the row range over the pool
 * reorders no reduction and stays bit-exact.  Blocks are multiples of the
 * four-row unroll so each worker still runs the row4 kernel, and the last block
 * absorbs the remainder exactly as the serial call did. */
#define QMAT_ROW_BLOCK 32u
/* Below this many weight bytes the pool dispatch costs more than it saves. */
#define QMAT_THREAD_MIN_BYTES 262144u

typedef struct {
    float *out;
    const int8_t *qx;   /* INT8/INT4: quantized activation */
    const float *x;     /* F16: the activation is not quantized */
    float sx;
    const void *weights;
    const float *scales;
    const float *bias;
    size_t rows;
    size_t cols;
    int qtype;
} qmat_rows_job;

static void qmat_rows_dispatch(const qmat_rows_job *j, size_t row0, size_t count) {
    const float *bias = j->bias == NULL ? NULL : j->bias + row0;
    if (j->qtype == QMAT_INT8) {
        matvec_q8(j->out + row0, j->qx, j->sx,
                  (const int8_t *)j->weights + row0 * j->cols,
                  j->scales + row0, bias, count, j->cols);
#if defined(MYNAH_QMAT_F16)
    } else if (j->qtype == QMAT_F16) {
        matvec_f16(j->out + row0, j->x,
                   (const __fp16 *)j->weights + row0 * j->cols,
                   bias, count, j->cols);
#endif
    } else {
        matvec_q4(j->out + row0, j->qx, j->sx,
                  (const uint8_t *)j->weights + row0 * (j->cols / 2u),
                  j->scales + row0 * (j->cols / QMAT_Q4_GROUP),
                  bias, count, j->cols);
    }
}

static void qmat_rows_block(void *opaque, int block_index) {
    const qmat_rows_job *j = (const qmat_rows_job *)opaque;
    const size_t row0 = (size_t)block_index * QMAT_ROW_BLOCK;
    if (row0 >= j->rows) return;
    size_t count = j->rows - row0;
    if (count > QMAT_ROW_BLOCK) count = QMAT_ROW_BLOCK;
    qmat_rows_dispatch(j, row0, count);
}

static void matvec_q_rows(float *out, const int8_t *qx, const float *x, float sx,
                          const void *weights, const float *scales,
                          const float *bias, size_t rows, size_t cols,
                          int qtype) {
    size_t weight_bytes;
    if (qtype == QMAT_INT8) weight_bytes = rows * cols;
    else if (qtype == QMAT_F16) weight_bytes = rows * cols * 2u;
    else weight_bytes = rows * cols / 2u;
    const qmat_rows_job job = {out, qx, x, sx, weights, scales, bias,
                               rows, cols, qtype};
    if (mynah_num_threads() > 1 && weight_bytes >= QMAT_THREAD_MIN_BYTES &&
        rows > QMAT_ROW_BLOCK) {
        const size_t blocks = (rows + QMAT_ROW_BLOCK - 1u) / QMAT_ROW_BLOCK;
        if (blocks <= (size_t)INT_MAX) {
            mynah_parallel_for((int)blocks, qmat_rows_block, (void *)&job);
            return;
        }
    }
    qmat_rows_dispatch(&job, 0u, rows);
}

/* ------------------------------------------------ threaded greedy argmax (f32)
 *
 * The fused local-transformer output projection walks the full f32 projection
 * matrix per stream, and one core only reaches a fraction of DRAM bandwidth
 * (see the row split above).  Every candidate row is an independent dot
 * product, so each block computes a private best and the reduction visits
 * blocks in ascending row order with a strict >, which preserves the serial
 * loop's first-on-ties winner exactly.  A block whose rows are all masked or
 * all -inf keeps -inf and never wins, matching the serial kernel's behavior. */
#define QMAT_ARGMAX_MAX_BLOCKS 512u

typedef struct {
    const float *weights;
    const float *in;
    const float *bias;
    size_t rows;
    size_t cols;
    size_t block;
    size_t allowed_rows;
    size_t extra_row;
    int allow_extra;
    float *best_value;   /* per block */
    unsigned *best_row;  /* per block */
} argmax_rows_job;

static void argmax_rows_block(void *opaque, int block_index) {
    const argmax_rows_job *j = (const argmax_rows_job *)opaque;
    const size_t row0 = (size_t)block_index * j->block;
    size_t end = row0 + j->block;
    if (end > j->rows) end = j->rows;
    float best = -INFINITY;
    size_t winner = 0;
    for (size_t row = row0; row < end; ++row) {
        const int allowed = row < j->allowed_rows ||
                            (j->allow_extra && row == j->extra_row);
        if (!allowed) continue;
        float value = mynah_dot_f32(j->weights + row * j->cols, j->in, j->cols);
        if (j->bias != NULL) value += j->bias[row];
        if (value > best) {
            best = value;
            winner = row;
        }
    }
    j->best_value[block_index] = best;
    j->best_row[block_index] = (unsigned)winner;
}

static int matvec_argmax_f32_mt(const float *weights, const float *in,
                                const float *bias, size_t rows, size_t cols,
                                size_t allowed_rows, size_t extra_row,
                                int allow_extra, unsigned *argmax) {
    const char *env = getenv("MYNAH_ARGMAX_MT");
    if ((env != NULL && strcmp(env, "0") == 0) || mynah_num_threads() <= 1 ||
        rows <= 2u * QMAT_ROW_BLOCK || cols == 0u || rows > SIZE_MAX / cols ||
        rows * cols * sizeof(float) < (size_t)QMAT_THREAD_MIN_BYTES) {
        return mynah_matvec_argmax_f32(weights, in, bias, rows, cols,
                                       allowed_rows, extra_row, allow_extra,
                                       argmax);
    }
    if (weights == NULL || in == NULL || argmax == NULL ||
        allowed_rows > rows || (allow_extra && extra_row >= rows)) {
        return -1;
    }
    size_t blocks = (rows + QMAT_ROW_BLOCK - 1u) / QMAT_ROW_BLOCK;
    if (blocks > QMAT_ARGMAX_MAX_BLOCKS) blocks = QMAT_ARGMAX_MAX_BLOCKS;
    const size_t block = (rows + blocks - 1u) / blocks;
    blocks = (rows + block - 1u) / block;
    float best_value[QMAT_ARGMAX_MAX_BLOCKS];
    unsigned best_row[QMAT_ARGMAX_MAX_BLOCKS];
    argmax_rows_job job = {weights, in, bias, rows, cols, block,
                           allowed_rows, extra_row, allow_extra,
                           best_value, best_row};
    mynah_parallel_for((int)blocks, argmax_rows_block, &job);
    float best = -INFINITY;
    size_t winner = 0;
    for (size_t i = 0; i < blocks; ++i) {
        if (best_value[i] > best) {
            best = best_value[i];
            winner = best_row[i];
        }
    }
    *argmax = (unsigned)winner;
    return 0;
}

/* --------------------------------------------------------------- weight cache */
typedef struct {
    char *name;
    int qtype;
    int8_t *q8;      /* INT8 */
    uint8_t *q4;     /* INT4 packed */
#if defined(MYNAH_QMAT_F16)
    __fp16 *f16;     /* F16: [n * k], no scales */
#endif
    float *scales;   /* INT8: [n]; INT4: [n * k/32] */
    size_t n;
    size_t k;
} qmat_entry;

/* Entries are held by pointer, not by value.  A value array would be moved by
 * realloc on growth, and another thread is meanwhile reading an entry for the
 * whole duration of its matvec -- the pointer it holds would dangle.  Indirect
 * storage keeps every entry at a fixed address for its lifetime, so a lock
 * around get-or-create is enough and readers need no lock at all: an entry is
 * fully built before it is published and never mutated afterwards. */
struct mynah_qmat_cache {
    int qtype; /* QMAT_F32 (off) / QMAT_INT8 / QMAT_INT4 / QMAT_F16 */
    int use_row4;
    qmat_entry **entries;
    size_t count;
    size_t capacity;
    pthread_mutex_t mutex;
};

mynah_qmat_cache *mynah_qmat_cache_new(int enabled) {
    mynah_qmat_cache *c = (mynah_qmat_cache *)calloc(1, sizeof(*c));
    if (c == NULL) return NULL;
    int qtype = QMAT_F32;
    if (enabled < 0) {
        const char *env = getenv("MYNAH_QUANT");
        if (env != NULL && strcmp(env, "int8") == 0) qtype = QMAT_INT8;
        else if (env != NULL && strcmp(env, "int4") == 0) qtype = QMAT_INT4;
        else if (env != NULL && strcmp(env, "f16") == 0) qtype = QMAT_F16;
    } else if (enabled == QMAT_INT8 || enabled == QMAT_INT4 ||
               enabled == QMAT_F16) {
        qtype = enabled;
    }
#if !defined(MYNAH_QMAT_F16)
    /* No half converts on this target: stay on the exact f32 path. */
    if (qtype == QMAT_F16) qtype = QMAT_F32;
#endif
    c->qtype = qtype;
    c->use_row4 = getenv("MYNAH_QMAT_SINGLE_ROW") == NULL;
    if (pthread_mutex_init(&c->mutex, NULL) != 0) {
        free(c);
        return NULL;
    }
    return c;
}

int mynah_qmat_cache_enabled(const mynah_qmat_cache *cache) {
    return cache != NULL && cache->qtype != QMAT_F32;
}

void mynah_qmat_cache_free(mynah_qmat_cache *cache) {
    if (cache == NULL) return;
    for (size_t i = 0; i < cache->count; ++i) {
        qmat_entry *e = cache->entries[i];
        if (e == NULL) continue;
        free(e->name);
        free(e->q8);
        free(e->q4);
#if defined(MYNAH_QMAT_F16)
        free(e->f16);
#endif
        free(e->scales);
        free(e);
    }
    free(cache->entries);
    pthread_mutex_destroy(&cache->mutex);
    free(cache);
}

/* Caller holds cache->mutex. */
static const qmat_entry *cache_lookup(const mynah_qmat_cache *cache, const char *name) {
    for (size_t i = 0; i < cache->count; ++i) {
        if (strcmp(cache->entries[i]->name, name) == 0) return cache->entries[i];
    }
    return NULL;
}

/* Quantize `w` [n, k] under the cache's qtype and store under `name`.  Returns
 * the entry, or NULL on OOM or an INT4 shape it cannot represent (k not a
 * multiple of 32) so the caller can fall back to f32. */
static const qmat_entry *cache_insert(mynah_qmat_cache *cache, const char *name,
                                      const float *w, size_t n, size_t k) {
    if (k == 0 || n > SIZE_MAX / k || n > SIZE_MAX / sizeof(float)) return NULL;
    if (cache->qtype == QMAT_INT4 && (k % QMAT_Q4_GROUP) != 0) return NULL;
    if (cache->count == cache->capacity) {
        const size_t next = cache->capacity == 0 ? 16u : cache->capacity * 2u;
        qmat_entry **grown = (qmat_entry **)realloc(cache->entries, next * sizeof(*grown));
        if (grown == NULL) return NULL;
        cache->entries = grown;
        cache->capacity = next;
    }
    qmat_entry *e = (qmat_entry *)calloc(1, sizeof(*e));
    if (e == NULL) return NULL;
    e->qtype = cache->qtype;
    e->n = n;
    e->k = k;
    e->name = (char *)malloc(strlen(name) + 1u);
    if (e->name == NULL) return NULL;
    if (cache->qtype == QMAT_INT8) {
        e->q8 = (int8_t *)malloc(n * k);
        e->scales = (float *)malloc(n * sizeof(float));
        if (e->q8 == NULL || e->scales == NULL) goto fail;
        quantize_weight_int8(w, n, k, e->q8, e->scales);
#if defined(MYNAH_QMAT_F16)
    } else if (cache->qtype == QMAT_F16) {
        if (n > SIZE_MAX / k / sizeof(__fp16)) goto fail;
        e->f16 = (__fp16 *)malloc(n * k * sizeof(__fp16));
        if (e->f16 == NULL) goto fail;
        for (size_t i = 0; i < n * k; ++i) e->f16[i] = (__fp16)w[i];
#endif
    } else {
        const size_t groups = k / QMAT_Q4_GROUP;
        e->q4 = (uint8_t *)malloc(n * (k / 2u));
        e->scales = (float *)malloc(n * groups * sizeof(float));
        if (e->q4 == NULL || e->scales == NULL) goto fail;
        quantize_weight_int4(w, n, k, e->q4, e->scales);
    }
    memcpy(e->name, name, strlen(name) + 1u);
    /* Published only once fully built: a reader that finds it can use it
     * without a lock, because nothing mutates an entry after this point. */
    cache->entries[cache->count++] = e;
    return e;
fail:
    free(e->name);
    free(e->q8);
    free(e->q4);
#if defined(MYNAH_QMAT_F16)
    free(e->f16);
#endif
    free(e->scales);
    free(e);
    return NULL;
}

/* ------------------------------------------------- weight-stationary batching
 *
 * Decode is bound by the weight bytes, not by the arithmetic: one activation
 * row touches the whole matrix, so B concurrent requests that each walk it in
 * turn pay B trips to DRAM for the same bytes.  Inverting the loops -- outer
 * over row blocks, inner over the batch -- reads each block once and serves
 * every row from cache.  A block is QMAT_ROW_BLOCK * cols bytes (24 KB for a
 * 768-wide int8 weight), so it stays resident in L1 across the batch.
 *
 * This must be bit-exact against the unbatched path, or the audio a request
 * gets would depend on which other requests it happened to batch with.  It is:
 * every (weight row, activation) pair runs the same kernel over the same k in
 * the same order, and only the order of independent pairs changes.  The f32
 * path has no such guarantee -- sgemm may accumulate differently for M=B than
 * for M=1 -- so it stays one call per row. */
typedef struct {
    const qmat_entry *entry;
    const int8_t *qx;       /* batch * cols, INT8/INT4 */
    const float *const *x;  /* batch pointers, F16 */
    const float *sx;        /* batch */
    float *const *out;      /* batch pointers */
    const float *bias;
    size_t batch;
    size_t rows;
    size_t cols;
} qmat_batch_job;

static void qmat_batch_rows(const qmat_batch_job *j, size_t row0, size_t count) {
    const qmat_entry *e = j->entry;
    const void *weights;
    if (e->qtype == QMAT_INT8) weights = (const void *)e->q8;
#if defined(MYNAH_QMAT_F16)
    else if (e->qtype == QMAT_F16) weights = (const void *)e->f16;
#endif
    else weights = (const void *)e->q4;
    for (size_t b = 0; b < j->batch; ++b) {
        const qmat_rows_job rj = {
            j->out[b],
            j->qx == NULL ? NULL : j->qx + b * j->cols,
            j->x == NULL ? NULL : j->x[b],
            j->sx == NULL ? 0.0f : j->sx[b],
            weights, e->scales, j->bias, j->rows, j->cols, e->qtype
        };
        qmat_rows_dispatch(&rj, row0, count);
    }
}

static void qmat_batch_block(void *opaque, int block_index) {
    const qmat_batch_job *j = (const qmat_batch_job *)opaque;
    const size_t row0 = (size_t)block_index * QMAT_ROW_BLOCK;
    if (row0 >= j->rows) return;
    size_t count = j->rows - row0;
    if (count > QMAT_ROW_BLOCK) count = QMAT_ROW_BLOCK;
    qmat_batch_rows(j, row0, count);
}

/* --------------------------------------------------------------- public API */
int mynah_qmat_greedy_argmax_resolved(mynah_qmat_cache *cache, const char *name,
                             const float *weight_data, const float *in, size_t k, size_t n,
                             const float *bias, size_t allowed_rows,
                             unsigned extra_row, int allow_extra, unsigned *argmax,
                             char *error, size_t error_capacity) {
    if (cache == NULL || cache->qtype != QMAT_F32) return 1;
    if (matvec_argmax_f32_mt(weight_data, in, bias, n, k,
                             allowed_rows, (size_t)extra_row,
                             allow_extra, argmax) != 0) {
        snprintf(error, error_capacity, "invalid greedy projection shape: %s", name);
        return -1;
    }
    return 0;
}

int mynah_qmat_greedy_argmax(mynah_qmat_cache *cache, const mynah_weights *file,
                             const char *name, const float *in, size_t k, size_t n,
                             const float *bias, size_t allowed_rows,
                             unsigned extra_row, int allow_extra, unsigned *argmax,
                             char *error, size_t error_capacity) {
    mynah_tensor weight;
    if (mynah_weights_get(file, name, &weight) != 0) {
        snprintf(error, error_capacity, "model tensor is missing: %s", name);
        return -1;
    }
    return mynah_qmat_greedy_argmax_resolved(cache, name, weight.data, in, k, n,
                                             bias, allowed_rows, extra_row,
                                             allow_extra, argmax, error, error_capacity);
}

int mynah_qmat_linear_resolved(mynah_qmat_cache *cache,
                      const mynah_backend *backend, const char *name,
                      const float *weight_data,
                      const float *in, float *out, size_t count, size_t k, size_t n,
                      const float *bias, char *error, size_t error_capacity) {
    const int use_q = cache != NULL && cache->qtype != QMAT_F32 &&
                      count <= QMAT_SMALL_COUNT && k <= QMAT_K_MAX;
    const qmat_entry *e = NULL;
    if (use_q) {
        /* Get-or-create is one critical section; the returned entry then stays
         * valid and immutable, so the matvec below runs outside the lock. */
        pthread_mutex_lock(&cache->mutex);
        e = cache_lookup(cache, name);
        if (e == NULL) e = cache_insert(cache, name, weight_data, n, k);
        pthread_mutex_unlock(&cache->mutex);
    }
    if (e == NULL) {
        /* Disabled, too large, unrepresentable, or OOM: exact f32 matmul. */
        return mynah_backend_matmul(backend, in, out, count, k, n, weight_data, bias,
                                    error, error_capacity);
    }
    int8_t qx[QMAT_K_MAX];
    for (size_t t = 0; t < count; ++t) {
        const float *xr = in + t * k;
        float *orow = out + t * n;
#if defined(MYNAH_QMAT_F16)
        /* F16 carries its own exponent, so the activation stays f32 and there
         * is no activation-quantization pass at all. */
        if (e->qtype == QMAT_F16) {
            matvec_q_rows(orow, NULL, xr, 0.0f, e->f16, NULL, bias, n, k, QMAT_F16);
            continue;
        }
#endif
        const float sx = quantize_act_int8(qx, xr, k);
        if (!cache->use_row4) {
            for (size_t row = 0; row < n; ++row) {
                float value = e->qtype == QMAT_INT8
                                  ? dot_q8(qx, sx, e->q8 + row * k, e->scales[row], k)
                                  : dot_q4(qx, sx, e->q4 + row * (k / 2u),
                                           e->scales + row * (k / QMAT_Q4_GROUP), k);
                if (bias != NULL) value += bias[row];
                orow[row] = value;
            }
        } else if (e->qtype == QMAT_INT8) {
            matvec_q_rows(orow, qx, xr, sx, e->q8, e->scales, bias, n, k, QMAT_INT8);
        } else {
            matvec_q_rows(orow, qx, xr, sx, e->q4, e->scales, bias, n, k, QMAT_INT4);
        }
    }
    return 0;
}

int mynah_qmat_linear_batched(mynah_qmat_cache *cache, const mynah_backend *backend,
                              const char *name, const float *weight_data,
                              const float *const *in_rows, float *const *out_rows,
                              size_t batch, size_t k, size_t n, const float *bias,
                              int8_t *qx_scratch, float *sx_scratch,
                              char *error, size_t error_capacity) {
    if (batch == 0u) return 0;
    if (batch == 1u) {
        return mynah_qmat_linear_resolved(cache, backend, name, weight_data,
                                          in_rows[0], out_rows[0], 1u, k, n, bias,
                                          error, error_capacity);
    }
    const int use_q = cache != NULL && cache->qtype != QMAT_F32 && k <= QMAT_K_MAX &&
                      cache->use_row4 && qx_scratch != NULL && sx_scratch != NULL;
    const qmat_entry *e = NULL;
    if (use_q) {
        pthread_mutex_lock(&cache->mutex);
        e = cache_lookup(cache, name);
        if (e == NULL) e = cache_insert(cache, name, weight_data, n, k);
        pthread_mutex_unlock(&cache->mutex);
    }
    if (e == NULL) {
        /* No bit-exact batching available here: keep every row on the exact
         * path it would have taken alone. */
        for (size_t b = 0; b < batch; ++b) {
            if (mynah_qmat_linear_resolved(cache, backend, name, weight_data,
                                           in_rows[b], out_rows[b], 1u, k, n, bias,
                                           error, error_capacity) != 0) {
                return -1;
            }
        }
        return 0;
    }
#if defined(MYNAH_QMAT_F16)
    const int is_f16 = e->qtype == QMAT_F16;
#else
    const int is_f16 = 0;
#endif
    if (!is_f16) {
        /* F16 keeps the activation in f32 and needs no quantization pass. */
        for (size_t b = 0; b < batch; ++b) {
            sx_scratch[b] = quantize_act_int8(qx_scratch + b * k, in_rows[b], k);
        }
    }
    const qmat_batch_job job = {
        e, is_f16 ? NULL : qx_scratch, is_f16 ? in_rows : NULL,
        is_f16 ? NULL : sx_scratch, out_rows, bias, batch, n, k
    };
    size_t weight_bytes;
    if (e->qtype == QMAT_INT8) weight_bytes = n * k;
    else if (is_f16) weight_bytes = n * k * 2u;
    else weight_bytes = n * k / 2u;
    if (mynah_num_threads() > 1 && weight_bytes >= QMAT_THREAD_MIN_BYTES &&
        n > QMAT_ROW_BLOCK) {
        const size_t blocks = (n + QMAT_ROW_BLOCK - 1u) / QMAT_ROW_BLOCK;
        if (blocks <= (size_t)INT_MAX) {
            mynah_parallel_for((int)blocks, qmat_batch_block, (void *)&job);
            return 0;
        }
    }
    /* Serial, but still blocked: walking the whole matrix once per row would
     * defeat the point of batching. */
    for (size_t row0 = 0; row0 < n; row0 += QMAT_ROW_BLOCK) {
        size_t count = n - row0;
        if (count > QMAT_ROW_BLOCK) count = QMAT_ROW_BLOCK;
        qmat_batch_rows(&job, row0, count);
    }
    return 0;
}

int mynah_qmat_linear(mynah_qmat_cache *cache, const mynah_weights *file,
                      const mynah_backend *backend, const char *name,
                      const float *in, float *out, size_t count, size_t k, size_t n,
                      const float *bias, char *error, size_t error_capacity) {
    mynah_tensor weight;
    if (mynah_weights_get(file, name, &weight) != 0) {
        snprintf(error, error_capacity, "model tensor is missing: %s", name);
        return -1;
    }
    return mynah_qmat_linear_resolved(cache, backend, name, weight.data,
                                      in, out, count, k, n, bias,
                                      error, error_capacity);
}

static int self_test_one(int qtype, char *error, size_t error_capacity) {
    enum { N = 48, K = 768, MATVEC_ROWS = 5 };
    static float w[N * K];
    static float x[K];
    static int8_t q8[N * K];
    static uint8_t q4[N * K / 2];
    static float scales8[N];
    static float scales4[N * K / QMAT_Q4_GROUP];
    static int8_t qx[K];
    float bias[MATVEC_ROWS];
    float matvec[MATVEC_ROWS];
    for (size_t i = 0; i < (size_t)N; ++i) {
        for (size_t j = 0; j < (size_t)K; ++j) {
            w[i * K + j] = sinf(0.017f * (float)(i * 7u + j)) * (0.5f + 0.5f * cosf(0.003f * (float)j));
        }
    }
    for (size_t j = 0; j < (size_t)K; ++j) x[j] = cosf(0.011f * (float)j) - 0.3f;
    const float sx = quantize_act_int8(qx, x, K);
    for (size_t i = 0; i < MATVEC_ROWS; ++i) bias[i] = (float)i * 0.125f - 0.25f;
    if (qtype == QMAT_INT8) {
        quantize_weight_int8(w, N, K, q8, scales8);
        matvec_q8(matvec, qx, sx, q8, scales8, bias, MATVEC_ROWS, K);
    } else {
        quantize_weight_int4(w, N, K, q4, scales4);
        matvec_q4(matvec, qx, sx, q4, scales4, bias, MATVEC_ROWS, K);
    }
    float max_rel = 0.0f;
    float ref_energy = 0.0f;
    for (size_t i = 0; i < (size_t)N; ++i) {
        float ref = 0.0f;
        for (size_t j = 0; j < (size_t)K; ++j) ref += w[i * K + j] * x[j];
        const float got = qtype == QMAT_INT8
                              ? dot_q8(qx, sx, q8 + i * K, scales8[i], K)
                              : dot_q4(qx, sx, q4 + i * (K / 2), scales4 + i * (K / QMAT_Q4_GROUP), K);
        if (i < MATVEC_ROWS) {
            const float expected = got + bias[i];
            const float tolerance = 1.0e-6f * (1.0f + fabsf(expected));
            if (fabsf(matvec[i] - expected) > tolerance) {
                if (error != NULL && error_capacity > 0) {
                    snprintf(error, error_capacity,
                             "qmat %s four-row matvec mismatch at row %zu",
                             qtype == QMAT_INT8 ? "int8" : "int4", i);
                }
                return -1;
            }
        }
        ref_energy += ref * ref;
        const float denom = fabsf(ref) > 1.0e-3f ? fabsf(ref) : 1.0e-3f;
        const float rel = fabsf(got - ref) / denom;
        if (rel > max_rel) max_rel = rel;
    }
    /* INT8 ~ a few percent; INT4 is coarser (16 levels/group). */
    const float limit = qtype == QMAT_INT8 ? 0.05f : 0.20f;
    if (!(ref_energy > 0.0f) || max_rel > limit) {
        if (error != NULL && error_capacity > 0) {
            snprintf(error, error_capacity, "qmat %s relative error too large: %.4f",
                     qtype == QMAT_INT8 ? "int8" : "int4", (double)max_rel);
        }
        return -1;
    }
    return 0;
}

/* The row-blocked dispatch must be bit-exact against a single serial call:
 * same kernel, same four-row unroll inside every block, no reduction split
 * across a block boundary.  The blocks are walked directly rather than through
 * the pool so the partition arithmetic (offsets into weights/scales/bias and
 * the short trailing block) is checked at any thread count.  N is deliberately
 * not a multiple of QMAT_ROW_BLOCK so the remainder path is exercised. */
static int self_test_rows_blocked(int qtype, char *error, size_t error_capacity) {
    enum { N = 200, K = 256 };
    int status = -1;
    float *w = malloc((size_t)N * K * sizeof(float));
    float *x = malloc((size_t)K * sizeof(float));
    float *bias = malloc((size_t)N * sizeof(float));
    float *serial = malloc((size_t)N * sizeof(float));
    float *blocked = malloc((size_t)N * sizeof(float));
    int8_t *qx = malloc((size_t)K);
    int8_t *q8 = malloc((size_t)N * K);
    uint8_t *q4 = malloc((size_t)N * K / 2u);
    float *scales8 = malloc((size_t)N * sizeof(float));
    float *scales4 = malloc((size_t)N * (K / QMAT_Q4_GROUP) * sizeof(float));
#if defined(MYNAH_QMAT_F16)
    __fp16 *h16 = malloc((size_t)N * K * sizeof(__fp16));
    const int h16_ok = h16 != NULL;
#else
    void *h16 = NULL;
    const int h16_ok = 1;   /* unused on targets without half converts */
#endif
    if (w == NULL || x == NULL || bias == NULL || serial == NULL || blocked == NULL ||
        qx == NULL || q8 == NULL || q4 == NULL || scales8 == NULL || scales4 == NULL ||
        !h16_ok) {
        if (error != NULL && error_capacity > 0)
            snprintf(error, error_capacity, "qmat blocked self-test out of memory");
        goto done;
    }
    for (size_t i = 0; i < (size_t)N * K; ++i)
        w[i] = sinf(0.013f * (float)i) * (0.5f + 0.5f * cosf(0.0007f * (float)i));
    for (size_t j = 0; j < (size_t)K; ++j) x[j] = cosf(0.011f * (float)j) - 0.3f;
    for (size_t i = 0; i < (size_t)N; ++i) bias[i] = (float)i * 0.03125f - 0.5f;
    const float sx = quantize_act_int8(qx, x, K);
    const void *weights;
    const float *scales;
    if (qtype == QMAT_INT8) {
        quantize_weight_int8(w, N, K, q8, scales8);
        weights = q8;
        scales = scales8;
        matvec_q8(serial, qx, sx, q8, scales8, bias, N, K);
#if defined(MYNAH_QMAT_F16)
    } else if (qtype == QMAT_F16) {
        for (size_t i = 0; i < (size_t)N * K; ++i) h16[i] = (__fp16)w[i];
        weights = h16;
        scales = NULL;
        matvec_f16(serial, x, h16, bias, N, K);
#endif
    } else {
        quantize_weight_int4(w, N, K, q4, scales4);
        weights = q4;
        scales = scales4;
        matvec_q4(serial, qx, sx, q4, scales4, bias, N, K);
    }
    for (size_t i = 0; i < (size_t)N; ++i) blocked[i] = 0.0f;
    const qmat_rows_job job = {blocked, qx, x, sx, weights, scales, bias,
                               (size_t)N, (size_t)K, qtype};
    const int blocks = (int)(((size_t)N + QMAT_ROW_BLOCK - 1u) / QMAT_ROW_BLOCK);
    for (int b = 0; b < blocks; ++b) qmat_rows_block((void *)&job, b);
    /* Compare with a tight tolerance rather than memcmp.  The split performs
     * the same arithmetic per row -- a wrong partition would offset weights or
     * scales and be off by a wide margin, which this still catches -- but the
     * build uses -ffast-math (and -mfma on x86), so the compiler may contract
     * or reassociate the final scale/bias differently for a block of 32 rows
     * than for one call of 200, giving last-bit differences.  Bit-identical
     * output was verified end-to-end on macOS/Accelerate; it is not something
     * these flags guarantee across compilers. */
    for (size_t i = 0; i < (size_t)N; ++i) {
        const float denom = fabsf(serial[i]) > 1.0e-3f ? fabsf(serial[i]) : 1.0e-3f;
        if (fabsf(serial[i] - blocked[i]) / denom > 1.0e-6f) {
            if (error != NULL && error_capacity > 0) {
                snprintf(error, error_capacity,
                         "qmat %s blocked matvec differs from serial at row %zu"
                         " (%.9g vs %.9g)",
                         qtype == QMAT_INT8 ? "int8"
                             : (qtype == QMAT_F16 ? "f16" : "int4"), i,
                         (double)serial[i], (double)blocked[i]);
            }
            goto done;
        }
    }
    status = 0;
done:
    free(w); free(x); free(bias); free(serial); free(blocked);
    free(qx); free(q8); free(q4); free(scales8); free(scales4); free(h16);
    return status;
}

#if defined(MYNAH_QMAT_F16)
/* F16 keeps the activation exact and only rounds the weights, so it must land
 * far closer to the f32 reference than INT8 does.  The bound below is ~50x
 * tighter than the INT8 one; if a change loosens it, the accuracy argument for
 * preferring F16 over INT8 no longer holds. */
static int self_test_f16(char *error, size_t error_capacity) {
    enum { N = 64, K = 512 };
    int status = -1;
    float *w = malloc((size_t)N * K * sizeof(float));
    float *x = malloc((size_t)K * sizeof(float));
    float *bias = malloc((size_t)N * sizeof(float));
    float *got = malloc((size_t)N * sizeof(float));
    __fp16 *h = malloc((size_t)N * K * sizeof(__fp16));
    if (w == NULL || x == NULL || bias == NULL || got == NULL || h == NULL) {
        if (error != NULL && error_capacity > 0)
            snprintf(error, error_capacity, "qmat f16 self-test out of memory");
        goto done;
    }
    for (size_t i = 0; i < (size_t)N * K; ++i) {
        w[i] = sinf(0.017f * (float)i) * (0.5f + 0.5f * cosf(0.003f * (float)i));
        h[i] = (__fp16)w[i];
    }
    for (size_t j = 0; j < (size_t)K; ++j) x[j] = cosf(0.011f * (float)j) - 0.3f;
    for (size_t i = 0; i < (size_t)N; ++i) bias[i] = (float)i * 0.125f - 0.25f;
    matvec_f16(got, x, h, bias, N, K);
    float max_rel = 0.0f;
    for (size_t i = 0; i < (size_t)N; ++i) {
        double ref = 0.0;
        for (size_t j = 0; j < (size_t)K; ++j) ref += (double)w[i * K + j] * x[j];
        ref += bias[i];
        const double denom = fabs(ref) > 1.0e-3 ? fabs(ref) : 1.0e-3;
        const float rel = (float)(fabs(got[i] - ref) / denom);
        if (rel > max_rel) max_rel = rel;
    }
    if (!(max_rel <= 1.0e-3f)) {
        if (error != NULL && error_capacity > 0) {
            snprintf(error, error_capacity,
                     "qmat f16 relative error too large: %.5f", (double)max_rel);
        }
        goto done;
    }
    status = 0;
done:
    free(w); free(x); free(bias); free(got); free(h);
    return status;
}
#endif

/* The batching contract: batching must not change a row's result.  If it did,
 * a request's audio would depend on which other requests happened to be in
 * flight with it, which is a nondeterminism regression no tolerance can excuse.
 * So this asserts bit-equality, not closeness, against the same rows computed
 * one at a time. */
static int self_test_batched(int qtype, char *error, size_t error_capacity) {
    enum { N = 96, K = 256, B = 5 };
    int status = -1;
    mynah_qmat_cache *cache = mynah_qmat_cache_new(qtype);
    if (cache == NULL) {
        if (error != NULL && error_capacity > 0)
            snprintf(error, error_capacity, "qmat batched self-test out of memory");
        return -1;
    }
    if (cache->qtype != qtype) {
        /* Unsupported on this target (no half converts): nothing to check. */
        mynah_qmat_cache_free(cache);
        return 0;
    }
    float *w = (float *)malloc((size_t)N * K * sizeof(float));
    float *x = (float *)malloc((size_t)B * K * sizeof(float));
    float *bias = (float *)malloc((size_t)N * sizeof(float));
    float *ref = (float *)malloc((size_t)B * N * sizeof(float));
    float *got = (float *)malloc((size_t)B * N * sizeof(float));
    int8_t *qx = (int8_t *)malloc((size_t)B * K);
    float *sx = (float *)malloc((size_t)B * sizeof(float));
    if (w == NULL || x == NULL || bias == NULL || ref == NULL || got == NULL ||
        qx == NULL || sx == NULL) {
        if (error != NULL && error_capacity > 0)
            snprintf(error, error_capacity, "qmat batched self-test out of memory");
        goto done;
    }
    for (size_t i = 0; i < (size_t)N * K; ++i)
        w[i] = sinf(0.017f * (float)i) * (0.5f + 0.5f * cosf(0.0011f * (float)i));
    for (size_t i = 0; i < (size_t)B * K; ++i) x[i] = cosf(0.019f * (float)i) - 0.25f;
    for (size_t i = 0; i < (size_t)N; ++i) bias[i] = (float)i * 0.015625f - 0.75f;
    const float *in_rows[B];
    float *out_rows[B];
    for (size_t b = 0; b < (size_t)B; ++b) {
        in_rows[b] = x + b * K;
        out_rows[b] = got + b * N;
    }
    for (size_t b = 0; b < (size_t)B; ++b) {
        if (mynah_qmat_linear_resolved(cache, NULL, "batched.self.test", w,
                                       x + b * K, ref + b * N, 1u, K, N, bias,
                                       error, error_capacity) != 0) {
            goto done;
        }
    }
    if (mynah_qmat_linear_batched(cache, NULL, "batched.self.test", w,
                                  in_rows, out_rows, B, K, N, bias, qx, sx,
                                  error, error_capacity) != 0) {
        goto done;
    }
    for (size_t b = 0; b < (size_t)B; ++b) {
        for (size_t i = 0; i < (size_t)N; ++i) {
            const size_t at = b * N + i;
            if (memcmp(&ref[at], &got[at], sizeof(float)) != 0) {
                if (error != NULL && error_capacity > 0)
                    snprintf(error, error_capacity,
                             "qmat batched qtype=%d differs at row %zu col %zu: "
                             "%.9g vs %.9g", qtype, b, i,
                             (double)ref[at], (double)got[at]);
                goto done;
            }
        }
    }
    status = 0;
done:
    free(w); free(x); free(bias); free(ref); free(got); free(qx); free(sx);
    mynah_qmat_cache_free(cache);
    return status;
}

int mynah_qmat_self_test(char *error, size_t error_capacity) {
    if (self_test_one(QMAT_INT8, error, error_capacity) != 0) return -1;
    if (self_test_one(QMAT_INT4, error, error_capacity) != 0) return -1;
    if (self_test_rows_blocked(QMAT_INT8, error, error_capacity) != 0) return -1;
    if (self_test_rows_blocked(QMAT_INT4, error, error_capacity) != 0) return -1;
    if (self_test_batched(QMAT_INT8, error, error_capacity) != 0) return -1;
    if (self_test_batched(QMAT_INT4, error, error_capacity) != 0) return -1;
#if defined(MYNAH_QMAT_F16)
    if (self_test_f16(error, error_capacity) != 0) return -1;
    if (self_test_rows_blocked(QMAT_F16, error, error_capacity) != 0) return -1;
    if (self_test_batched(QMAT_F16, error, error_capacity) != 0) return -1;
#endif
    return 0;
}
