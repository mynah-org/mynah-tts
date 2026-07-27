#include "qmat.h"
#include "kernels.h"

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

/* count at/below this uses the native int dot; above it falls back to the f32
 * BLAS matmul (the prefill, already fast and kept bit-exact). */
#define QMAT_SMALL_COUNT 16
#define QMAT_K_MAX 8192
#define QMAT_Q4_GROUP 32

enum { QMAT_F32 = 0, QMAT_INT8 = 1, QMAT_INT4 = 2 };

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

/* --------------------------------------------------------------- weight cache */
typedef struct {
    char *name;
    int qtype;
    int8_t *q8;      /* INT8 */
    uint8_t *q4;     /* INT4 packed */
    float *scales;   /* INT8: [n]; INT4: [n * k/32] */
    size_t n;
    size_t k;
} qmat_entry;

struct mynah_qmat_cache {
    int qtype; /* QMAT_F32 (off) / QMAT_INT8 / QMAT_INT4 */
    int use_row4;
    qmat_entry *entries;
    size_t count;
    size_t capacity;
};

mynah_qmat_cache *mynah_qmat_cache_new(int enabled) {
    mynah_qmat_cache *c = (mynah_qmat_cache *)calloc(1, sizeof(*c));
    if (c == NULL) return NULL;
    int qtype = QMAT_F32;
    if (enabled < 0) {
        const char *env = getenv("MYNAH_QUANT");
        if (env != NULL && strcmp(env, "int8") == 0) qtype = QMAT_INT8;
        else if (env != NULL && strcmp(env, "int4") == 0) qtype = QMAT_INT4;
    } else if (enabled == QMAT_INT8 || enabled == QMAT_INT4) {
        qtype = enabled;
    }
    c->qtype = qtype;
    c->use_row4 = getenv("MYNAH_QMAT_SINGLE_ROW") == NULL;
    return c;
}

int mynah_qmat_cache_enabled(const mynah_qmat_cache *cache) {
    return cache != NULL && cache->qtype != QMAT_F32;
}

void mynah_qmat_cache_free(mynah_qmat_cache *cache) {
    if (cache == NULL) return;
    for (size_t i = 0; i < cache->count; ++i) {
        free(cache->entries[i].name);
        free(cache->entries[i].q8);
        free(cache->entries[i].q4);
        free(cache->entries[i].scales);
    }
    free(cache->entries);
    free(cache);
}

static const qmat_entry *cache_lookup(const mynah_qmat_cache *cache, const char *name) {
    for (size_t i = 0; i < cache->count; ++i) {
        if (strcmp(cache->entries[i].name, name) == 0) return &cache->entries[i];
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
        qmat_entry *grown = (qmat_entry *)realloc(cache->entries, next * sizeof(*grown));
        if (grown == NULL) return NULL;
        cache->entries = grown;
        cache->capacity = next;
    }
    qmat_entry *e = &cache->entries[cache->count];
    memset(e, 0, sizeof(*e));
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
    } else {
        const size_t groups = k / QMAT_Q4_GROUP;
        e->q4 = (uint8_t *)malloc(n * (k / 2u));
        e->scales = (float *)malloc(n * groups * sizeof(float));
        if (e->q4 == NULL || e->scales == NULL) goto fail;
        quantize_weight_int4(w, n, k, e->q4, e->scales);
    }
    memcpy(e->name, name, strlen(name) + 1u);
    ++cache->count;
    return e;
fail:
    free(e->name);
    free(e->q8);
    free(e->q4);
    free(e->scales);
    return NULL;
}

/* --------------------------------------------------------------- public API */
int mynah_qmat_greedy_argmax_resolved(mynah_qmat_cache *cache, const char *name,
                             const float *weight_data, const float *in, size_t k, size_t n,
                             const float *bias, size_t allowed_rows,
                             unsigned extra_row, int allow_extra, unsigned *argmax,
                             char *error, size_t error_capacity) {
    if (cache == NULL || cache->qtype != QMAT_F32) return 1;
    if (mynah_matvec_argmax_f32(weight_data, in, bias, n, k,
                                allowed_rows, (size_t)extra_row,
                                allow_extra, argmax) != 0) {
        snprintf(error, error_capacity, "invalid greedy projection shape: %s", name);
        return -1;
    }
    return 0;
}

int mynah_qmat_greedy_argmax(mynah_qmat_cache *cache, const mynah_safetensors *file,
                             const char *name, const float *in, size_t k, size_t n,
                             const float *bias, size_t allowed_rows,
                             unsigned extra_row, int allow_extra, unsigned *argmax,
                             char *error, size_t error_capacity) {
    mynah_tensor weight;
    if (mynah_safetensors_get(file, name, &weight) != 0) {
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
        e = cache_lookup(cache, name);
        if (e == NULL) e = cache_insert(cache, name, weight_data, n, k);
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
            matvec_q8(orow, qx, sx, e->q8, e->scales, bias, n, k);
        } else {
            matvec_q4(orow, qx, sx, e->q4, e->scales, bias, n, k);
        }
    }
    return 0;
}

int mynah_qmat_linear(mynah_qmat_cache *cache, const mynah_safetensors *file,
                      const mynah_backend *backend, const char *name,
                      const float *in, float *out, size_t count, size_t k, size_t n,
                      const float *bias, char *error, size_t error_capacity) {
    mynah_tensor weight;
    if (mynah_safetensors_get(file, name, &weight) != 0) {
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

int mynah_qmat_self_test(char *error, size_t error_capacity) {
    if (self_test_one(QMAT_INT8, error, error_capacity) != 0) return -1;
    if (self_test_one(QMAT_INT4, error, error_capacity) != 0) return -1;
    return 0;
}
