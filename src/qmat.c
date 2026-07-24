#include "qmat.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__ARM_FEATURE_DOTPROD)
#include <arm_neon.h>
#endif

/* count at/below this uses the native int8 dot; above it falls back to the f32
 * BLAS matmul (the prefill, already fast and kept bit-exact). */
#define QMAT_SMALL_COUNT 16
#define QMAT_K_MAX 8192

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

/* ---------------------------------------------------------- int8 dot kernels */
static float dot_q8(const int8_t *qx, float sx, const int8_t *w, float ws, size_t k) {
#if defined(__ARM_FEATURE_DOTPROD)
    int32x4_t acc = vdupq_n_s32(0);
    size_t j = 0;
    for (; j + 16 <= k; j += 16) {
        acc = vdotq_s32(acc, vld1q_s8(w + j), vld1q_s8(qx + j));
    }
    int32_t s = vaddvq_s32(acc);
    for (; j < k; ++j) s += (int32_t)w[j] * (int32_t)qx[j];
    return (float)s * ws * sx;
#else
    int32_t s = 0;
    for (size_t j = 0; j < k; ++j) s += (int32_t)w[j] * (int32_t)qx[j];
    return (float)s * ws * sx;
#endif
}

/* --------------------------------------------------------------- weight cache */
typedef struct {
    char *name;
    int8_t *q8;
    float *scales;
    size_t n;
    size_t k;
} qmat_entry;

struct mynah_qmat_cache {
    int enabled;
    qmat_entry *entries;
    size_t count;
    size_t capacity;
};

mynah_qmat_cache *mynah_qmat_cache_new(int enabled) {
    mynah_qmat_cache *c = (mynah_qmat_cache *)calloc(1, sizeof(*c));
    if (c == NULL) return NULL;
    if (enabled < 0) {
        const char *env = getenv("MYNAH_QUANT");
        enabled = (env != NULL && strcmp(env, "int8") == 0) ? 1 : 0;
    }
    c->enabled = enabled;
    return c;
}

int mynah_qmat_cache_enabled(const mynah_qmat_cache *cache) {
    return cache != NULL && cache->enabled;
}

void mynah_qmat_cache_free(mynah_qmat_cache *cache) {
    if (cache == NULL) return;
    for (size_t i = 0; i < cache->count; ++i) {
        free(cache->entries[i].name);
        free(cache->entries[i].q8);
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

/* Quantize `w` [n, k] and store under `name`.  Returns the entry or NULL on OOM. */
static const qmat_entry *cache_insert(mynah_qmat_cache *cache, const char *name,
                                      const float *w, size_t n, size_t k) {
    if (cache->count == cache->capacity) {
        const size_t next = cache->capacity == 0 ? 16u : cache->capacity * 2u;
        qmat_entry *grown = (qmat_entry *)realloc(cache->entries, next * sizeof(*grown));
        if (grown == NULL) return NULL;
        cache->entries = grown;
        cache->capacity = next;
    }
    qmat_entry *e = &cache->entries[cache->count];
    memset(e, 0, sizeof(*e));
    e->name = (char *)malloc(strlen(name) + 1u);
    e->q8 = (int8_t *)malloc(n * k);
    e->scales = (float *)malloc(n * sizeof(float));
    if (e->name == NULL || e->q8 == NULL || e->scales == NULL) {
        free(e->name);
        free(e->q8);
        free(e->scales);
        return NULL;
    }
    memcpy(e->name, name, strlen(name) + 1u);
    e->n = n;
    e->k = k;
    quantize_weight_int8(w, n, k, e->q8, e->scales);
    ++cache->count;
    return e;
}

/* --------------------------------------------------------------- public API */
int mynah_qmat_linear(mynah_qmat_cache *cache, const mynah_safetensors *file,
                      const mynah_backend *backend, const char *name,
                      const float *in, float *out, size_t count, size_t k, size_t n,
                      const float *bias, char *error, size_t error_capacity) {
    mynah_tensor weight;
    if (mynah_safetensors_get(file, name, &weight) != 0) {
        snprintf(error, error_capacity, "model tensor is missing: %s", name);
        return -1;
    }
    const int use_int8 = cache != NULL && cache->enabled && count <= QMAT_SMALL_COUNT &&
                         k <= QMAT_K_MAX;
    if (!use_int8) {
        return mynah_backend_matmul(backend, in, out, count, k, n, weight.data, bias,
                                    error, error_capacity);
    }
    const qmat_entry *e = cache_lookup(cache, name);
    if (e == NULL) e = cache_insert(cache, name, weight.data, n, k);
    if (e == NULL) {
        /* Quantization failed (OOM): fall back to f32 rather than error out. */
        return mynah_backend_matmul(backend, in, out, count, k, n, weight.data, bias,
                                    error, error_capacity);
    }
    int8_t qx[QMAT_K_MAX];
    for (size_t t = 0; t < count; ++t) {
        const float *xr = in + t * k;
        float *orow = out + t * n;
        const float sx = quantize_act_int8(qx, xr, k);
        for (size_t i = 0; i < n; ++i) {
            float v = dot_q8(qx, sx, e->q8 + i * k, e->scales[i], k);
            if (bias != NULL) v += bias[i];
            orow[i] = v;
        }
    }
    return 0;
}

int mynah_qmat_self_test(char *error, size_t error_capacity) {
    enum { N = 48, K = 768 };
    static float w[N * K];
    static float x[K];
    static int8_t q8[N * K];
    static float scales[N];
    static int8_t qx[K];
    /* Deterministic, structured data (no rand): a smooth weight and input. */
    for (size_t i = 0; i < (size_t)N; ++i) {
        for (size_t j = 0; j < (size_t)K; ++j) {
            w[i * K + j] = sinf(0.017f * (float)(i * 7u + j)) * (0.5f + 0.5f * cosf(0.003f * (float)j));
        }
    }
    for (size_t j = 0; j < (size_t)K; ++j) x[j] = cosf(0.011f * (float)j) - 0.3f;
    quantize_weight_int8(w, N, K, q8, scales);
    const float sx = quantize_act_int8(qx, x, K);
    float max_rel = 0.0f;
    float ref_energy = 0.0f;
    for (size_t i = 0; i < (size_t)N; ++i) {
        float ref = 0.0f;
        for (size_t j = 0; j < (size_t)K; ++j) ref += w[i * K + j] * x[j];
        const float got = dot_q8(qx, sx, q8 + i * K, scales[i], K);
        ref_energy += ref * ref;
        const float denom = fabsf(ref) > 1.0e-3f ? fabsf(ref) : 1.0e-3f;
        const float rel = fabsf(got - ref) / denom;
        if (rel > max_rel) max_rel = rel;
    }
    if (!(ref_energy > 0.0f)) {
        if (error != NULL && error_capacity > 0) snprintf(error, error_capacity, "qmat self-test produced no signal");
        return -1;
    }
    /* Per-row absmax int8 on length-768 dot products: a few percent is expected. */
    if (max_rel > 0.05f) {
        if (error != NULL && error_capacity > 0) {
            snprintf(error, error_capacity, "qmat int8 relative error too large: %.4f", (double)max_rel);
        }
        return -1;
    }
    return 0;
}
