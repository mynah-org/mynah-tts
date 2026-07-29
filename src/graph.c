#include "mynah_tts_internal.h"
#include "graph.h"
#include "kernels.h"
#include "threads.h"

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

#if !defined(MYNAH_DISABLE_SIMD) && (defined(__ARM_NEON) || defined(__aarch64__))
#include <arm_neon.h>
#define MYNAH_GRAPH_NEON 1
#endif

typedef struct codec_bnns_cache codec_bnns_cache;

#if defined(MYNAH_USE_ACCELERATE)
#include <Accelerate/Accelerate.h>
#elif defined(MYNAH_USE_OPENBLAS)
#include <cblas.h>
#endif

/*
 * This file deliberately keeps the first native backend boring: one scalar
 * implementation of the Magpie graph, with weights read directly from the
 * converted safetensors pack.  The operation boundaries are the seams for
 * NEON/AVX2/AVX-512, Metal and CUDA backends added later.
 */

static void graph_error(char *error, size_t capacity, const char *message) {
    if (error != NULL && capacity > 0) snprintf(error, capacity, "%s", message);
}

/* Backend-agnostic sgemm.  Replaces direct cblas_sgemm calls so that
 * CUDA/Metal backends can accelerate all matmuls, not just linear(). */
static int graph_sgemm(const mynah_backend *backend,
                       int trans_a, int trans_b,
                       size_t m, size_t n, size_t k,
                       float alpha,
                       const float *a, size_t lda,
                       const float *b, size_t ldb,
                       float beta,
                       float *c, size_t ldc,
                       char *error, size_t error_capacity) {
    return mynah_backend_sgemm(backend, trans_a, trans_b, m, n, k,
                               alpha, a, lda, b, ldb, beta, c, ldc,
                               error, error_capacity);
}

/* Optional phase timing, enabled with MYNAH_TIMING=1, printed to stderr. */
static double phase_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}

static int tensor(const mynah_safetensors *file, const char *name,
                  mynah_tensor *out, char *error, size_t error_capacity) {
    if (mynah_safetensors_get(file, name, out) != 0) {
        snprintf(error, error_capacity, "model tensor is missing: %s", name);
        return -1;
    }
    return 0;
}

static float *allocate_floats(size_t count, char *error, size_t error_capacity) {
    if (count == 0 || count > SIZE_MAX / sizeof(float)) {
        graph_error(error, error_capacity, "invalid graph allocation size");
        return NULL;
    }
    float *value = (float *)calloc(count, sizeof(*value));
    if (value == NULL) graph_error(error, error_capacity, "out of memory in native graph");
    return value;
}

static void layer_norm(const float *input, float *output, size_t length,
                       size_t width, const float *weight) {
    mynah_layernorm_f32(input, weight, NULL, output, length, width, 1.0e-5f);
}

static float gelu_tanh(float x) {
    const float cubic = x * x * x;
    const float inner = 0.7978845608028654f * (x + 0.044715f * cubic);
    return 0.5f * x * (1.0f + tanhf(inner));
}

static void gelu_tanh_array(float *values, size_t length, float *scratch) {
#if defined(MYNAH_USE_ACCELERATE)
    if (scratch != NULL) {
        for (size_t i = 0; i < length; ++i) {
            const float x = values[i];
            const float cubic = x * x * x;
            scratch[i] = 0.7978845608028654f *
                         (x + 0.044715f * cubic);
        }
        size_t offset = 0;
        while (offset < length) {
            const size_t remaining = length - offset;
            const int batch = remaining > (size_t)INT_MAX
                ? INT_MAX : (int)remaining;
            vvtanhf(scratch + offset, scratch + offset, &batch);
            offset += (size_t)batch;
        }
        for (size_t i = 0; i < length; ++i) {
            values[i] = 0.5f * values[i] * (1.0f + scratch[i]);
        }
        return;
    }
#else
    (void)scratch;
#endif
    for (size_t i = 0; i < length; ++i) values[i] = gelu_tanh(values[i]);
}

/* ---- NEON-accelerated attention primitives --------------------------------
 * The single-row decoder attention attn·V weighted sum (hw = 64 elements per
 * position) is vectorised with 4-wide NEON FMA.  The Q·K dot product stays
 * scalar to preserve the exact greedy argmax path (NEON lane reordering
 * changes softmax scores enough to flip tokens).  The axpy accumulation
 * order change is absorbed by downstream layers without affecting EOS.
 * A scalar fallback is always compiled for portability. */

/* out[0..n) += weight * src[0..n) */
static void axpy_f32(float *out, const float *src, float weight, size_t n) {
#if defined(MYNAH_GRAPH_NEON)
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

int mynah_graph_self_test(char *error, size_t error_capacity) {
#if defined(MYNAH_USE_ACCELERATE)
    float values[] = {
        -INFINITY, -10.0f, -3.0f, -1.0f, -0.25f, -0.0f, 0.0f,
        0.125f, 0.5f, 1.0f, 2.0f, 3.0f, 8.0f, INFINITY, NAN,
        -6.75f, 0.03125f, 4.5f, -2.125f
    };
    float expected[sizeof(values) / sizeof(values[0])];
    float scratch[sizeof(values) / sizeof(values[0])];
    const size_t count = sizeof(values) / sizeof(values[0]);
    for (size_t i = 0; i < count; ++i) expected[i] = gelu_tanh(values[i]);
    gelu_tanh_array(values, count, scratch);
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
                     "vForce GELU mismatch at %zu: got %.9g expected %.9g",
                     i, (double)values[i], (double)expected[i]);
        }
        return -1;
    }
#else
    (void)error;
    (void)error_capacity;
#endif
    return 0;
}

static void softmax_row_inplace(float *values, size_t length) {
    float maximum = -FLT_MAX;
    for (size_t i = 0; i < length; ++i) {
        if (values[i] > maximum) maximum = values[i];
    }
    float total = 0.0f;
    for (size_t i = 0; i < length; ++i) {
        values[i] = expf(values[i] - maximum);
        total += values[i];
    }
    const float inverse = total > 0.0f ? 1.0f / total : 0.0f;
    for (size_t i = 0; i < length; ++i) values[i] *= inverse;
}

static int linear(const mynah_backend *backend, const float *input, float *output,
                  size_t length, size_t input_width, size_t output_width,
                  const float *weight, const float *bias, char *error,
                  size_t error_capacity) {
    return mynah_backend_matmul(backend, input, output, length, input_width,
                                output_width, weight, bias, error, error_capacity);
}

static int causal_conv_ffn(const mynah_safetensors *file, const mynah_backend *backend,
                           const char *prefix,
                           size_t layer, const float *input, float *output,
                           size_t length, size_t width, size_t ffn_width,
                           size_t kernel, char *error, size_t error_capacity) {
    char name[256];
    mynah_tensor proj;
    mynah_tensor out_net;
    snprintf(name, sizeof(name), "%s.layers.%zu.pos_ff.proj.conv.weight", prefix, layer);
    if (tensor(file, name, &proj, error, error_capacity) != 0) return -1;
    snprintf(name, sizeof(name), "%s.layers.%zu.pos_ff.o_net.conv.weight", prefix, layer);
    if (tensor(file, name, &out_net, error, error_capacity) != 0) return -1;
    float *hidden = allocate_floats(length * ffn_width, error, error_capacity);
    if (hidden == NULL) return -1;
    if (kernel == 1u) {
        if (linear(backend, input, hidden, length, width, ffn_width, proj.data, NULL,
                   error, error_capacity) != 0) {
            free(hidden);
            return -1;
        }
        /* This buffer is host-owned in the non-resident graph path.  Device
         * GELU APIs require device pointers; use the scalar/reference kernel
         * here so CPU and Metal follow the same host-side contract. */
        mynah_gelu_f32_scalar(hidden, length * ffn_width);
        const int result = linear(backend, hidden, output, length, ffn_width, width,
                                  out_net.data, NULL, error, error_capacity);
        free(hidden);
        return result;
    }
    if (length <= (size_t)INT_MAX && width <= (size_t)INT_MAX && ffn_width <= (size_t)INT_MAX) {
        /* A causal conv-FFN is a sum of `kernel` shifted matmuls (weight tap k),
         * the same trick as conv1d_causal but with time-major rows.  This is the
         * encoder's dominant cost, so accumulate the taps with sgemm instead of
         * the scalar quadruple loop. */
        float *wk = (float *)malloc(ffn_width * width * kernel * sizeof(float));
        if (wk == NULL) {
            free(hidden);
            graph_error(error, error_capacity, "out of memory in causal conv-ffn");
            return -1;
        }
        /* Extract all taps at once: wk[k][o][i] = proj[(o*width+i)*kernel+k].
         * Inner loop over k has stride=1 (sequential reads). */
        for (size_t o = 0; o < ffn_width; ++o) {
            for (size_t i = 0; i < width; ++i) {
                const float *src = proj.data + (o * width + i) * kernel;
                for (size_t k = 0; k < kernel; ++k)
                    wk[k * ffn_width * width + o * width + i] = src[k];
            }
        }
        memset(hidden, 0, length * ffn_width * sizeof(float));
        for (size_t k = 0; k < kernel; ++k) {
            const size_t shift = kernel - 1u - k;
            if (shift >= length) continue;
            graph_sgemm(backend, 0, 1, (int)(length - shift), (int)ffn_width, (int)width, 1.0f, input, (int)width, wk + k * ffn_width * width, (int)width, 1.0f, hidden + shift * ffn_width, (int)ffn_width, error, error_capacity);
        }
        mynah_gelu_f32_scalar(hidden, length * ffn_width);
        /* Extract out_net taps with same optimized pattern. */
        for (size_t o = 0; o < width; ++o) {
            for (size_t i = 0; i < ffn_width; ++i) {
                const float *src = out_net.data + (o * ffn_width + i) * kernel;
                for (size_t k = 0; k < kernel; ++k)
                    wk[k * width * ffn_width + o * ffn_width + i] = src[k];
            }
        }
        memset(output, 0, length * width * sizeof(float));
        for (size_t k = 0; k < kernel; ++k) {
            const size_t shift = kernel - 1u - k;
            if (shift >= length) continue;
            graph_sgemm(backend, 0, 1, (int)(length - shift), (int)width, (int)ffn_width, 1.0f, hidden, (int)ffn_width, wk + k * width * ffn_width, (int)ffn_width, 1.0f, output + shift * width, (int)width, error, error_capacity);
        }
        free(wk);
        free(hidden);
        return 0;
    }
    for (size_t t = 0; t < length; ++t) {
        for (size_t o = 0; o < ffn_width; ++o) {
            float value = 0.0f;
            for (size_t i = 0; i < width; ++i) {
                for (size_t k = 0; k < kernel; ++k) {
                    const long source_t = (long)t - (long)(kernel - 1u) + (long)k;
                    if (source_t < 0) continue;
                    value += proj.data[(o * width + i) * kernel + k] *
                             input[(size_t)source_t * width + i];
                }
            }
            hidden[t * ffn_width + o] = gelu_tanh(value);
        }
    }
    for (size_t t = 0; t < length; ++t) {
        for (size_t o = 0; o < width; ++o) {
            float value = 0.0f;
            for (size_t i = 0; i < ffn_width; ++i) {
                for (size_t k = 0; k < kernel; ++k) {
                    const long source_t = (long)t - (long)(kernel - 1u) + (long)k;
                    if (source_t < 0) continue;
                    value += out_net.data[(o * ffn_width + i) * kernel + k] *
                             hidden[(size_t)source_t * ffn_width + i];
                }
            }
            output[t * width + o] = value;
        }
    }
    free(hidden);
    return 0;
}

static int self_attention(const mynah_safetensors *file, const mynah_backend *backend,
                          const char *prefix,
                          size_t layer, const float *input, float *output,
                          size_t length, size_t width, size_t heads,
                          char *error, size_t error_capacity) {
    char name[256];
    mynah_tensor qkv;
    mynah_tensor projection;
    snprintf(name, sizeof(name), "%s.layers.%zu.self_attention.qkv_net.weight", prefix, layer);
    if (tensor(file, name, &qkv, error, error_capacity) != 0) return -1;
    snprintf(name, sizeof(name), "%s.layers.%zu.self_attention.o_net.weight", prefix, layer);
    if (tensor(file, name, &projection, error, error_capacity) != 0) return -1;
    const size_t head_width = width / heads;
    float *qkv_values = allocate_floats(length * width * 3u, error, error_capacity);
    float *context = allocate_floats(length * width, error, error_capacity);
    float *scores = allocate_floats(length, error, error_capacity);
    if (qkv_values == NULL || context == NULL || scores == NULL) {
        free(qkv_values);
        free(context);
        free(scores);
        return -1;
    }
    if (linear(backend, input, qkv_values, length, width, width * 3u, qkv.data, NULL,
               error, error_capacity) != 0) {
        free(qkv_values);
        free(context);
        free(scores);
        return -1;
    }
    if (length <= (size_t)INT_MAX && head_width <= (size_t)INT_MAX) {
        float *queries = allocate_floats(length * head_width, error, error_capacity);
        float *keys = allocate_floats(length * head_width, error, error_capacity);
        float *values = allocate_floats(length * head_width, error, error_capacity);
        float *score_matrix = allocate_floats(length * length, error, error_capacity);
        float *head_context = allocate_floats(length * head_width, error, error_capacity);
        if (queries == NULL || keys == NULL || values == NULL || score_matrix == NULL ||
            head_context == NULL) {
            free(queries);
            free(keys);
            free(values);
            free(score_matrix);
            free(head_context);
            free(qkv_values);
            free(context);
            free(scores);
            return -1;
        }
        for (size_t head = 0; head < heads; ++head) {
            for (size_t t = 0; t < length; ++t) {
                const float *row = qkv_values + t * width * 3u;
                memcpy(queries + t * head_width, row + head * head_width,
                       head_width * sizeof(float));
                memcpy(keys + t * head_width, row + width + head * head_width,
                       head_width * sizeof(float));
                memcpy(values + t * head_width, row + width * 2u + head * head_width,
                       head_width * sizeof(float));
            }
            graph_sgemm(backend, 0, 1, (int)length, (int)length, (int)head_width, 1.0f / sqrtf((float)head_width), queries, (int)head_width, keys, (int)head_width, 0.0f, score_matrix, (int)length, error, error_capacity);
            for (size_t t = 0; t < length; ++t) {
                for (size_t s = t + 1u; s < length; ++s) score_matrix[t * length + s] = 0.0f;
                softmax_row_inplace(score_matrix + t * length, t + 1u);
                for (size_t s = t + 1u; s < length; ++s) score_matrix[t * length + s] = 0.0f;
            }
            graph_sgemm(backend, 0, 0, (int)length, (int)head_width, (int)length, 1.0f, score_matrix, (int)length, values, (int)head_width, 0.0f, head_context, (int)head_width, error, error_capacity);
            for (size_t t = 0; t < length; ++t) {
                memcpy(context + t * width + head * head_width,
                       head_context + t * head_width, head_width * sizeof(float));
            }
        }
        free(queries);
        free(keys);
        free(values);
        free(score_matrix);
        free(head_context);
        if (linear(backend, context, output, length, width, width, projection.data, NULL,
                   error, error_capacity) != 0) {
            free(qkv_values);
            free(context);
            free(scores);
            return -1;
        }
        free(qkv_values);
        free(context);
        free(scores);
        return 0;
    }
    for (size_t t = 0; t < length; ++t) {
        for (size_t h = 0; h < heads; ++h) {
            float maximum = -FLT_MAX;
            for (size_t s = 0; s <= t; ++s) {
                float score = 0.0f;
                for (size_t d = 0; d < head_width; ++d) {
                    const size_t q_index = t * width * 3u + h * head_width + d;
                    const size_t k_index = s * width * 3u + width + h * head_width + d;
                    score += qkv_values[q_index] * qkv_values[k_index];
                }
                score *= 1.0f / sqrtf((float)head_width);
                scores[s] = score;
                if (score > maximum) maximum = score;
            }
            float denominator = 0.0f;
            for (size_t s = 0; s <= t; ++s) {
                scores[s] = expf(scores[s] - maximum);
                denominator += scores[s];
            }
            for (size_t d = 0; d < head_width; ++d) {
                float value = 0.0f;
                for (size_t s = 0; s <= t; ++s) {
                    const size_t v_index = s * width * 3u + 2u * width + h * head_width + d;
                    value += (scores[s] / denominator) * qkv_values[v_index];
                }
                context[t * width + h * head_width + d] = value;
            }
        }
    }
    if (linear(backend, context, output, length, width, width, projection.data, NULL,
               error, error_capacity) != 0) {
        free(qkv_values);
        free(context);
        free(scores);
        return -1;
    }
    free(qkv_values);
    free(context);
    free(scores);
    return 0;
}

static int cross_attention(const mynah_safetensors *file, const mynah_backend *backend,
                           const char *prefix,
                           size_t layer, const float *input, float *output,
                           size_t length, const float *memory, size_t memory_length,
                           size_t width, char *error, size_t error_capacity) {
    char name[256];
    mynah_tensor q_weight;
    mynah_tensor kv_weight;
    mynah_tensor projection;
    snprintf(name, sizeof(name), "%s.layers.%zu.cross_attention.q_net.weight", prefix, layer);
    if (tensor(file, name, &q_weight, error, error_capacity) != 0) return -1;
    snprintf(name, sizeof(name), "%s.layers.%zu.cross_attention.kv_net.weight", prefix, layer);
    if (tensor(file, name, &kv_weight, error, error_capacity) != 0) return -1;
    snprintf(name, sizeof(name), "%s.layers.%zu.cross_attention.o_net.weight", prefix, layer);
    if (tensor(file, name, &projection, error, error_capacity) != 0) return -1;
    const size_t attention_width = q_weight.shape[0];
    float *q = allocate_floats(length * attention_width, error, error_capacity);
    float *kv = allocate_floats(memory_length * attention_width * 2u, error, error_capacity);
    float *context = allocate_floats(length * attention_width, error, error_capacity);
    float *scores = allocate_floats(memory_length, error, error_capacity);
    if (q == NULL || kv == NULL || context == NULL || scores == NULL) {
        free(q);
        free(kv);
        free(context);
        free(scores);
        return -1;
    }
    if (linear(backend, input, q, length, width, attention_width, q_weight.data, NULL,
               error, error_capacity) != 0 ||
        linear(backend, memory, kv, memory_length, width, attention_width * 2u,
               kv_weight.data, NULL, error, error_capacity) != 0) {
        free(q);
        free(kv);
        free(context);
        free(scores);
        return -1;
    }
    if (length <= (size_t)INT_MAX && memory_length <= (size_t)INT_MAX &&
        attention_width <= (size_t)INT_MAX) {
        float *keys = allocate_floats(memory_length * attention_width, error, error_capacity);
        float *values = allocate_floats(memory_length * attention_width, error, error_capacity);
        float *score_matrix = allocate_floats(length * memory_length, error, error_capacity);
        if (keys == NULL || values == NULL || score_matrix == NULL) {
            free(keys);
            free(values);
            free(score_matrix);
            free(q);
            free(kv);
            free(context);
            free(scores);
            return -1;
        }
        for (size_t s = 0; s < memory_length; ++s) {
            memcpy(keys + s * attention_width, kv + s * attention_width * 2u,
                   attention_width * sizeof(float));
            memcpy(values + s * attention_width,
                   kv + s * attention_width * 2u + attention_width,
                   attention_width * sizeof(float));
        }
        graph_sgemm(backend, 0, 1, (int)length, (int)memory_length, (int)attention_width, 1.0f / sqrtf((float)attention_width), q, (int)attention_width, keys, (int)attention_width, 0.0f, score_matrix, (int)memory_length, error, error_capacity);
        for (size_t t = 0; t < length; ++t) {
            softmax_row_inplace(score_matrix + t * memory_length, memory_length);
        }
        graph_sgemm(backend, 0, 0, (int)length, (int)attention_width, (int)memory_length, 1.0f, score_matrix, (int)memory_length, values, (int)attention_width, 0.0f, context, (int)attention_width, error, error_capacity);
        free(keys);
        free(values);
        free(score_matrix);
        if (linear(backend, context, output, length, attention_width, width, projection.data,
                   NULL, error, error_capacity) != 0) {
            free(q);
            free(kv);
            free(context);
            free(scores);
            return -1;
        }
        free(q);
        free(kv);
        free(context);
        free(scores);
        return 0;
    }
    for (size_t t = 0; t < length; ++t) {
        float maximum = -FLT_MAX;
        for (size_t s = 0; s < memory_length; ++s) {
            float score = 0.0f;
            for (size_t d = 0; d < attention_width; ++d) {
                score += q[t * attention_width + d] * kv[s * attention_width * 2u + d];
            }
            scores[s] = score / sqrtf((float)attention_width);
            if (scores[s] > maximum) maximum = scores[s];
        }
        float denominator = 0.0f;
        for (size_t s = 0; s < memory_length; ++s) {
            scores[s] = expf(scores[s] - maximum);
            denominator += scores[s];
        }
        for (size_t d = 0; d < attention_width; ++d) {
            float value = 0.0f;
            for (size_t s = 0; s < memory_length; ++s) {
                value += (scores[s] / denominator) * kv[s * attention_width * 2u + attention_width + d];
            }
            context[t * attention_width + d] = value;
        }
    }
    if (linear(backend, context, output, length, attention_width, width, projection.data,
               NULL, error, error_capacity) != 0) {
        free(q);
        free(kv);
        free(context);
        free(scores);
        return -1;
    }
    free(q);
    free(kv);
    free(context);
    free(scores);
    return 0;
}

static int transformer_stack(const mynah_safetensors *file, const mynah_backend *backend,
                             const char *prefix,
                             size_t layers, size_t length, size_t width,
                             size_t ffn_width, size_t heads, size_t kernel,
                             int has_cross_attention, int apply_norm_out,
                             const float *memory,
                             size_t memory_length, float *states, float *output,
                             char *error, size_t error_capacity) {
    char name[256];
    mynah_tensor position;
    snprintf(name, sizeof(name), "%s.position_embeddings.weight", prefix);
    if (tensor(file, name, &position, error, error_capacity) != 0) return -1;
    float *working = allocate_floats(length * width, error, error_capacity);
    if (working == NULL) return -1;
    memcpy(working, states, length * width * sizeof(float));
    for (size_t t = 0; t < length; ++t) {
        for (size_t d = 0; d < width; ++d) working[t * width + d] += position.data[t * width + d];
    }
    float *normalized = allocate_floats(length * width, error, error_capacity);
    float *residual = allocate_floats(length * width, error, error_capacity);
    float *memory_normalized = NULL;
    if (normalized == NULL || residual == NULL) {
        free(normalized);
        free(residual);
        free(working);
        return -1;
    }
    for (size_t layer = 0; layer < layers; ++layer) {
        snprintf(name, sizeof(name), "%s.layers.%zu.norm_self.weight", prefix, layer);
        mynah_tensor norm_self;
        if (tensor(file, name, &norm_self, error, error_capacity) != 0) break;
        layer_norm(working, normalized, length, width, norm_self.data);
        if (self_attention(file, backend, prefix, layer, normalized, residual, length, width, heads,
                           error, error_capacity) != 0) break;
        for (size_t i = 0; i < length * width; ++i) working[i] += residual[i];

        if (has_cross_attention) {
            snprintf(name, sizeof(name), "%s.layers.%zu.norm_xattn_query.weight", prefix, layer);
            mynah_tensor norm_query;
            if (tensor(file, name, &norm_query, error, error_capacity) != 0) break;
            layer_norm(working, normalized, length, width, norm_query.data);
            if (memory_normalized == NULL) {
                memory_normalized = allocate_floats(memory_length * width, error, error_capacity);
                if (memory_normalized == NULL) break;
            }
            snprintf(name, sizeof(name), "%s.layers.%zu.norm_xattn_memory.weight", prefix, layer);
            mynah_tensor norm_memory;
            if (tensor(file, name, &norm_memory, error, error_capacity) != 0) break;
            layer_norm(memory, memory_normalized, memory_length, width, norm_memory.data);
            if (cross_attention(file, backend, prefix, layer, normalized, residual, length,
                                memory_normalized, memory_length, width, error,
                                error_capacity) != 0) break;
            for (size_t i = 0; i < length * width; ++i) working[i] += residual[i];
        }

        snprintf(name, sizeof(name), "%s.layers.%zu.norm_pos_ff.weight", prefix, layer);
        mynah_tensor norm_ff;
        if (tensor(file, name, &norm_ff, error, error_capacity) != 0) break;
        layer_norm(working, normalized, length, width, norm_ff.data);
        if (causal_conv_ffn(file, backend, prefix, layer, normalized, residual, length, width,
                            ffn_width, kernel, error, error_capacity) != 0) break;
        for (size_t i = 0; i < length * width; ++i) working[i] += residual[i];
        if (layer + 1u == layers && apply_norm_out) {
            snprintf(name, sizeof(name), "%s.norm_out.weight", prefix);
            mynah_tensor norm_out;
            if (tensor(file, name, &norm_out, error, error_capacity) != 0) break;
            layer_norm(working, output, length, width, norm_out.data);
        }
    }
    if (!apply_norm_out && (error == NULL || error[0] == '\0')) {
        memcpy(output, working, length * width * sizeof(float));
    }
    free(normalized);
    free(residual);
    free(memory_normalized);
    free(working);
    return error == NULL || error[0] == '\0' ? 0 : -1;
}

static int encode_text(const mynah_tts_model *model, const int *ids, size_t count,
                       float **encoded, char *error, size_t error_capacity) {
    const size_t width = model->info.hidden_dim;
    mynah_tensor embedding;
    if (tensor(model->tts, "text_embedding.weight", &embedding, error, error_capacity) != 0) return -1;
    if (count == 0 || count > model->info.text_max_length || embedding.shape[1] != width) {
        graph_error(error, error_capacity, "text length or embedding shape is invalid");
        return -1;
    }
    float *states = allocate_floats(count * width, error, error_capacity);
    float *result = allocate_floats(count * width, error, error_capacity);
    if (states == NULL || result == NULL) {
        free(states);
        free(result);
        return -1;
    }
    for (size_t t = 0; t < count; ++t) {
        if (ids[t] < 0 || (size_t)ids[t] >= embedding.shape[0]) {
            free(states);
            free(result);
            graph_error(error, error_capacity, "text token id is outside vocabulary");
            return -1;
        }
        memcpy(states + t * width, embedding.data + (size_t)ids[t] * width, width * sizeof(float));
    }
    const int result_code = transformer_stack(model->tts, model->backend, "encoder", model->info.encoder_layers,
                                              count, width, width * 4u, 12u, 3u, 0,
                                              1, NULL, 0, states, result, error, error_capacity);
    free(states);
    if (result_code != 0) {
        free(result);
        return -1;
    }
    *encoded = result;
    return 0;
}

typedef struct local_projection_cache local_projection_cache;

struct local_projection_cache {
    size_t stream_count;
    const float **projection_weights;
    const float **projection_biases;
    const float **audio_embeddings;
    size_t *audio_embedding_rows;
};

/* Embed a single stacked frame (one decoder input row).  The codes buffer is
 * laid out one contiguous max_raw_length row per codebook, so read with that
 * stride, not the growing current length, or every codebook after the first
 * reads from the wrong row and corrupts the decoder's audio history. */
static int embed_audio_frame(const mynah_tts_model *model, const unsigned *codes,
                             size_t code_stride, size_t frame, float *row,
                             char *error, size_t error_capacity) {
    const size_t width = model->info.hidden_dim;
    const size_t codebooks = model->info.codebook_count;
    const size_t stacking = model->info.frame_stacking_factor;
    if (stacking != 2u) {
        graph_error(error, error_capacity, "v1 requires a frame stacking factor of two");
        return -1;
    }
    const local_projection_cache *projection_cache =
        (const local_projection_cache *)model->local_projection_cache;
    memset(row, 0, width * sizeof(float));
    for (size_t fs = 0; fs < stacking; ++fs) {
        for (size_t codebook = 0; codebook < codebooks; ++codebook) {
            const unsigned code = codes[codebook * code_stride + frame * stacking + fs];
            const size_t stream = fs * codebooks + codebook;
            const float *table_data = NULL;
            size_t table_rows = 0;
            if (projection_cache != NULL && stream < projection_cache->stream_count) {
                table_data = projection_cache->audio_embeddings[stream];
                table_rows = projection_cache->audio_embedding_rows[stream];
            } else {
                char name[128];
                mynah_tensor table;
                snprintf(name, sizeof(name), "audio_embeddings.%zu.weight", stream);
                if (tensor(model->tts, name, &table, error, error_capacity) != 0) return -1;
                table_data = table.data;
                table_rows = table.shape[0];
            }
            if (code >= table_rows) {
                graph_error(error, error_capacity, "audio token id is outside vocabulary");
                return -1;
            }
            for (size_t d = 0; d < width; ++d)
                row[d] += table_data[(size_t)code * width + d];
        }
    }
    for (size_t d = 0; d < width; ++d) row[d] /= (float)(stacking * codebooks);
    return 0;
}

/* Small KV cache for the autoregressive local transformer.  It has no cross
 * attention and its sequence is at most stream_count+1 (17) positions, so the
 * attention stays scalar and each of the 16 codebook streams is a single-row
 * step instead of re-running the whole stack from scratch every time. */
typedef struct {
    size_t layers;
    size_t width;
    size_t heads;
    size_t head_width;
    size_t ffn_width;
    size_t capacity;
    size_t length;
    float *k; /* layers * capacity * width */
    float *v; /* layers * capacity * width */
    const float *position; /* local_transformer.position_embeddings.weight */
    /* Pre-resolved weight pointers (eliminate per-step snprintf+lookup). */
    const float *norm_self[4];
    const float *qkv_w[4];
    const float *o_w[4];
    const float *norm_ff[4];
    const float *ffn_up_w[4];
    const float *ffn_down_w[4];
    /* Metal resident local-transformer state.  Host K/V and scratch remain
     * for the scalar path; the GPU path never downloads its hidden row. */
    float *dev_k;
    float *dev_v;
    float *dev_x;
    float *dev_nrm;
    float *dev_qkv;
    float *dev_attn;
    float *dev_proj;
    float *dev_hidden;
    float *dev_logits;
    float *dev_input;
    const mynah_backend *dev_backend;
    int gpu_ready;
} local_cache;

static void local_projection_cache_free_impl(local_projection_cache *cache) {
    if (cache == NULL) return;
    free(cache->projection_weights);
    free(cache->projection_biases);
    free(cache->audio_embeddings);
    free(cache->audio_embedding_rows);
    free(cache);
}

void *mynah_graph_local_projection_cache_new(const mynah_tts_model *model) {
    if (model == NULL) return NULL;
    if (model->info.frame_stacking_factor == 0 ||
        model->info.codebook_count > SIZE_MAX / model->info.frame_stacking_factor) {
        return NULL;
    }
    const size_t streams = model->info.codebook_count *
                           model->info.frame_stacking_factor;
    if (streams == 0 || streams > SIZE_MAX / sizeof(float *)) return NULL;
    local_projection_cache *cache = (local_projection_cache *)calloc(1, sizeof(*cache));
    if (cache == NULL) return NULL;
    cache->stream_count = streams;
    cache->projection_weights = (const float **)calloc(streams, sizeof(float *));
    cache->projection_biases = (const float **)calloc(streams, sizeof(float *));
    cache->audio_embeddings = (const float **)calloc(streams, sizeof(float *));
    cache->audio_embedding_rows = (size_t *)calloc(streams, sizeof(size_t));
    if (cache->projection_weights == NULL || cache->projection_biases == NULL ||
        cache->audio_embeddings == NULL || cache->audio_embedding_rows == NULL) {
        local_projection_cache_free_impl(cache);
        return NULL;
    }
    char error[256];
    for (size_t stream = 0; stream < streams; ++stream) {
        char name[256];
        mynah_tensor tensor_view;
        snprintf(name, sizeof(name),
                 "local_transformer_out_projections.%zu.weight", stream);
        if (tensor(model->tts, name, &tensor_view, error, sizeof(error)) != 0) {
            local_projection_cache_free_impl(cache);
            return NULL;
        }
        cache->projection_weights[stream] = tensor_view.data;
        snprintf(name, sizeof(name),
                 "local_transformer_out_projections.%zu.bias", stream);
        if (tensor(model->tts, name, &tensor_view, error, sizeof(error)) != 0) {
            local_projection_cache_free_impl(cache);
            return NULL;
        }
        cache->projection_biases[stream] = tensor_view.data;
        snprintf(name, sizeof(name), "audio_embeddings.%zu.weight", stream);
        if (tensor(model->tts, name, &tensor_view, error, sizeof(error)) != 0) {
            local_projection_cache_free_impl(cache);
            return NULL;
        }
        if (tensor_view.rank != 2u || tensor_view.shape[1] != model->info.hidden_dim) {
            local_projection_cache_free_impl(cache);
            return NULL;
        }
        cache->audio_embeddings[stream] = tensor_view.data;
        cache->audio_embedding_rows[stream] = tensor_view.shape[0];
    }
    return cache;
}

void mynah_graph_local_projection_cache_free(void *opaque) {
    local_projection_cache_free_impl((local_projection_cache *)opaque);
}

typedef struct {
    float *x;
    float *nrm;
    float *qkv;
    float *attn;
    float *proj;
    float *hidden;
    float *scores;
    float *gelu_scratch;
} local_workspace;

static void local_workspace_free(local_workspace *workspace) {
    if (workspace == NULL) return;
    free(workspace->x);
    free(workspace->nrm);
    free(workspace->qkv);
    free(workspace->attn);
    free(workspace->proj);
    free(workspace->hidden);
    free(workspace->scores);
    free(workspace->gelu_scratch);
    memset(workspace, 0, sizeof(*workspace));
}

static int local_workspace_init(local_workspace *workspace, const local_cache *cache,
                                char *error, size_t error_capacity) {
    memset(workspace, 0, sizeof(*workspace));
    workspace->x = allocate_floats(cache->width, error, error_capacity);
    workspace->nrm = allocate_floats(cache->width, error, error_capacity);
    workspace->qkv = allocate_floats(cache->width * 3u, error, error_capacity);
    workspace->attn = allocate_floats(cache->width, error, error_capacity);
    workspace->proj = allocate_floats(cache->width, error, error_capacity);
    workspace->hidden = allocate_floats(cache->ffn_width, error, error_capacity);
    workspace->scores = allocate_floats(cache->capacity, error, error_capacity);
#if defined(MYNAH_USE_ACCELERATE)
    workspace->gelu_scratch = allocate_floats(cache->ffn_width, error, error_capacity);
#endif
    if (workspace->x == NULL || workspace->nrm == NULL || workspace->qkv == NULL ||
        workspace->attn == NULL || workspace->proj == NULL || workspace->hidden == NULL ||
        workspace->scores == NULL
#if defined(MYNAH_USE_ACCELERATE)
        || workspace->gelu_scratch == NULL
#endif
    ) {
        local_workspace_free(workspace);
        return -1;
    }
    return 0;
}

static void local_cache_free(local_cache *cache) {
    if (cache == NULL) return;
    free(cache->k);
    free(cache->v);
    cache->k = NULL;
    cache->v = NULL;
    if (cache->gpu_ready && cache->dev_backend != NULL) {
        mynah_backend_dev_free(cache->dev_backend, cache->dev_k);
        mynah_backend_dev_free(cache->dev_backend, cache->dev_v);
        mynah_backend_dev_free(cache->dev_backend, cache->dev_x);
        mynah_backend_dev_free(cache->dev_backend, cache->dev_nrm);
        mynah_backend_dev_free(cache->dev_backend, cache->dev_qkv);
        mynah_backend_dev_free(cache->dev_backend, cache->dev_attn);
        mynah_backend_dev_free(cache->dev_backend, cache->dev_proj);
        mynah_backend_dev_free(cache->dev_backend, cache->dev_hidden);
        mynah_backend_dev_free(cache->dev_backend, cache->dev_logits);
        mynah_backend_dev_free(cache->dev_backend, cache->dev_input);
    }
    cache->dev_k = NULL;
    cache->dev_v = NULL;
    cache->dev_x = NULL;
    cache->dev_nrm = NULL;
    cache->dev_qkv = NULL;
    cache->dev_attn = NULL;
    cache->dev_proj = NULL;
    cache->dev_hidden = NULL;
    cache->dev_logits = NULL;
    cache->dev_input = NULL;
    cache->dev_backend = NULL;
    cache->gpu_ready = 0;
}

static int local_cache_init(const mynah_tts_model *model, local_cache *cache,
                            size_t capacity, char *error, size_t error_capacity) {
    memset(cache, 0, sizeof(*cache));
    const size_t width = model->info.hidden_dim;
    cache->layers = model->info.local_transformer_layers;
    cache->width = width;
    cache->heads = 12u;
    cache->head_width = width / cache->heads;
    cache->ffn_width = width * 4u;
    cache->capacity = capacity;
    mynah_tensor position;
    if (tensor(model->tts, "local_transformer.position_embeddings.weight", &position,
               error, error_capacity) != 0) {
        return -1;
    }
    cache->position = position.data;
    cache->k = allocate_floats(cache->layers * capacity * width, error, error_capacity);
    cache->v = allocate_floats(cache->layers * capacity * width, error, error_capacity);
    if (cache->k == NULL || cache->v == NULL) {
        local_cache_free(cache);
        return -1;
    }
    /* Pre-resolve weight pointers (eliminate per-step snprintf+lookup). */
    for (size_t l = 0; l < cache->layers && l < 4u; ++l) {
        char nm[256]; mynah_tensor t;
        snprintf(nm, sizeof(nm), "local_transformer.layers.%zu.norm_self.weight", l);
        if (tensor(model->tts, nm, &t, error, error_capacity)!=0) return -1;
        cache->norm_self[l] = t.data;
        snprintf(nm, sizeof(nm), "local_transformer.layers.%zu.self_attention.qkv_net.weight", l);
        if (tensor(model->tts, nm, &t, error, error_capacity)!=0) return -1;
        cache->qkv_w[l] = t.data;
        snprintf(nm, sizeof(nm), "local_transformer.layers.%zu.self_attention.o_net.weight", l);
        if (tensor(model->tts, nm, &t, error, error_capacity)!=0) return -1;
        cache->o_w[l] = t.data;
        snprintf(nm, sizeof(nm), "local_transformer.layers.%zu.norm_pos_ff.weight", l);
        if (tensor(model->tts, nm, &t, error, error_capacity)!=0) return -1;
        cache->norm_ff[l] = t.data;
        snprintf(nm, sizeof(nm), "local_transformer.layers.%zu.pos_ff.proj.conv.weight", l);
        if (tensor(model->tts, nm, &t, error, error_capacity)!=0) return -1;
        cache->ffn_up_w[l] = t.data;
        snprintf(nm, sizeof(nm), "local_transformer.layers.%zu.pos_ff.o_net.conv.weight", l);
        if (tensor(model->tts, nm, &t, error, error_capacity)!=0) return -1;
        cache->ffn_down_w[l] = t.data;
    }
    if (mynah_backend_has_dev_ops(model->backend) &&
        mynah_backend_has_attention_dev(model->backend) &&
        cache->layers > 0u && cache->layers <= 4u &&
        cache->layers <= SIZE_MAX / capacity &&
        cache->layers * capacity <= SIZE_MAX / width) {
        const size_t kv_count = cache->layers * capacity * width;
        const size_t qkv_count = width * 3u;
        int ok = 1;
        cache->dev_backend = model->backend;
        ok = ok && mynah_backend_dev_alloc(model->backend, kv_count, &cache->dev_k,
                                            error, error_capacity) == 0;
        ok = ok && mynah_backend_dev_alloc(model->backend, kv_count, &cache->dev_v,
                                            error, error_capacity) == 0;
        ok = ok && mynah_backend_dev_alloc(model->backend, width, &cache->dev_x,
                                            error, error_capacity) == 0;
        ok = ok && mynah_backend_dev_alloc(model->backend, width, &cache->dev_nrm,
                                            error, error_capacity) == 0;
        ok = ok && mynah_backend_dev_alloc(model->backend, qkv_count, &cache->dev_qkv,
                                            error, error_capacity) == 0;
        ok = ok && mynah_backend_dev_alloc(model->backend, width, &cache->dev_attn,
                                            error, error_capacity) == 0;
        ok = ok && mynah_backend_dev_alloc(model->backend, width, &cache->dev_proj,
                                            error, error_capacity) == 0;
        ok = ok && mynah_backend_dev_alloc(model->backend, cache->ffn_width,
                                            &cache->dev_hidden, error, error_capacity) == 0;
        ok = ok && mynah_backend_dev_alloc(model->backend, model->info.audio_vocab_size,
                                            &cache->dev_logits, error, error_capacity) == 0;
        ok = ok && mynah_backend_dev_alloc(model->backend, width, &cache->dev_input,
                                            error, error_capacity) == 0;
        if (ok) {
            cache->gpu_ready = 1;
        } else {
            local_cache_free(cache);
            graph_error(error, error_capacity, "Metal local-transformer buffers unavailable");
            return -1;
        }
    }
    return 0;
}

/* One local-transformer position with all activations and KV state resident on
 * the GPU.  The caller owns the synchronization boundary: argmax must observe
 * this row before the next token embedding is uploaded. */
static int local_step_device(const mynah_tts_model *model, local_cache *cache,
                             const float *dev_input, float **dev_output,
                             char *error, size_t error_capacity) {
    if (model == NULL || cache == NULL || !cache->gpu_ready ||
        dev_input == NULL || dev_output == NULL || cache->dev_backend == NULL)
        return -1;
    const mynah_backend *backend = cache->dev_backend;
    const size_t width = cache->width;
    const size_t p = cache->length;
    const size_t hw = cache->head_width;
    if (p >= cache->capacity) {
        graph_error(error, error_capacity, "Metal local transformer cache overflow");
        return -1;
    }
    if (mynah_backend_batch_begin(backend, error, error_capacity) != 0 ||
        mynah_backend_copy_dev(backend, cache->dev_x, dev_input, width,
                               error, error_capacity) != 0 ||
        mynah_backend_residual_inplace(backend, cache->dev_x,
                                       cache->position + p * width, width,
                                       error, error_capacity) != 0) return -1;
    const float scale = 1.0f / sqrtf((float)hw);
    for (size_t layer = 0; layer < cache->layers; ++layer) {
        if (mynah_backend_layer_norm_inplace(backend, cache->dev_x, cache->dev_nrm,
                                             cache->norm_self[layer], 1u, width,
                                             error, error_capacity) != 0 ||
            mynah_backend_matmul_d2d(backend, cache->dev_nrm, cache->dev_qkv,
                                     1u, width, width * 3u, cache->qkv_w[layer],
                                     NULL, error, error_capacity) != 0 ||
            mynah_backend_self_attention_dev(
                backend, cache->dev_qkv,
                cache->dev_k + layer * cache->capacity * width,
                cache->dev_v + layer * cache->capacity * width,
                p, width, p + 1u, cache->heads, hw, scale, cache->dev_attn,
                error, error_capacity) != 0 ||
            mynah_backend_matmul_d2d(backend, cache->dev_attn, cache->dev_proj,
                                     1u, width, width, cache->o_w[layer], NULL,
                                     error, error_capacity) != 0 ||
            mynah_backend_residual_inplace(backend, cache->dev_x, cache->dev_proj,
                                            width, error, error_capacity) != 0 ||
            mynah_backend_layer_norm_inplace(backend, cache->dev_x, cache->dev_nrm,
                                             cache->norm_ff[layer], 1u, width,
                                             error, error_capacity) != 0 ||
            mynah_backend_matmul_d2d(backend, cache->dev_nrm, cache->dev_hidden,
                                     1u, width, cache->ffn_width,
                                     cache->ffn_up_w[layer], NULL,
                                     error, error_capacity) != 0 ||
            mynah_backend_gelu_inplace(backend, cache->dev_hidden, cache->ffn_width,
                                       error, error_capacity) != 0 ||
            mynah_backend_matmul_d2d(backend, cache->dev_hidden, cache->dev_proj,
                                     1u, cache->ffn_width, width,
                                     cache->ffn_down_w[layer], NULL,
                                     error, error_capacity) != 0 ||
            mynah_backend_residual_inplace(backend, cache->dev_x, cache->dev_proj,
                                            width, error, error_capacity) != 0) return -1;
    }
    cache->length += 1u;
    *dev_output = cache->dev_x;
    return 0;
}

/* Append one position to the local transformer and return its output row. */
static int local_step(const mynah_tts_model *model, local_cache *cache,
                      local_workspace *workspace, const float *input_row, float *out_row,
                      char *error, size_t error_capacity) {
    const size_t width = cache->width;
    const size_t heads = cache->heads;
    const size_t hw = cache->head_width;
    const size_t ffn = cache->ffn_width;
    const size_t p = cache->length;
    if (p >= cache->capacity) {
        graph_error(error, error_capacity, "local transformer cache overflow");
        return -1;
    }
    local_workspace owned;
    if (workspace == NULL) {
        if (local_workspace_init(&owned, cache, error, error_capacity) != 0) return -1;
        workspace = &owned;
    }
    float *x = workspace->x;
    float *nrm = workspace->nrm;
    float *qkv = workspace->qkv;
    float *attn = workspace->attn;
    float *proj = workspace->proj;
    float *hidden = workspace->hidden;
    float *scores = workspace->scores;
    for (size_t d = 0; d < width; ++d) x[d] = input_row[d] + cache->position[p * width + d];
    const float scale = 1.0f / sqrtf((float)hw);
    int failed = 0;
    for (size_t layer = 0; layer < cache->layers && !failed; ++layer) {
        layer_norm(x, nrm, 1u, width, cache->norm_self[layer]);
        if (mynah_backend_matmul(model->backend, nrm, qkv, 1u, width, width * 3u,
                                 cache->qkv_w[layer], NULL, error, error_capacity) != 0) { failed = 1; break; }
        float *kb = cache->k + layer * cache->capacity * width;
        float *vb = cache->v + layer * cache->capacity * width;
        memcpy(kb + p * width, qkv + width, width * sizeof(float));
        memcpy(vb + p * width, qkv + width * 2u, width * sizeof(float));
        for (size_t h = 0; h < heads; ++h) {
            const float *qh = qkv + h * hw;
            float maxv = -FLT_MAX;
            for (size_t s = 0; s <= p; ++s) {
                const float *k = kb + s * width + h * hw;
                float sc = 0.0f;
                    for (size_t d = 0; d < hw; ++d) sc += qh[d] * k[d];
                    sc *= scale;
                scores[s] = sc;
                if (sc > maxv) maxv = sc;
            }
            float denom = 0.0f;
            for (size_t s = 0; s <= p; ++s) { scores[s] = expf(scores[s] - maxv); denom += scores[s]; }
            float *outh = attn + h * hw;
            memset(outh, 0, hw * sizeof(float));
            for (size_t s = 0; s <= p; ++s)
                axpy_f32(outh, vb + s * width + h * hw, scores[s] / denom, hw);
        }
        if (mynah_backend_matmul(model->backend, attn, proj, 1u, width, width,
                                 cache->o_w[layer], NULL, error, error_capacity) != 0) { failed = 1; break; }
        for (size_t d = 0; d < width; ++d) x[d] += proj[d];
        layer_norm(x, nrm, 1u, width, cache->norm_ff[layer]);
        if (mynah_backend_matmul(model->backend, nrm, hidden, 1u, width, ffn,
                                 cache->ffn_up_w[layer], NULL, error, error_capacity) != 0) { failed = 1; break; }
        gelu_tanh_array(hidden, ffn, workspace->gelu_scratch);
        if (mynah_backend_matmul(model->backend, hidden, proj, 1u, ffn, width,
                                 cache->ffn_down_w[layer], NULL, error, error_capacity) != 0) { failed = 1; break; }
        for (size_t d = 0; d < width; ++d) x[d] += proj[d];
    }
    if (!failed) memcpy(out_row, x, width * sizeof(float));
    if (workspace == &owned) local_workspace_free(&owned);
    if (!failed) cache->length += 1u;
    return failed ? -1 : 0;
}

/* The local AR helper emits one stacked frame (two raw codec frames). */
static int sample_local_frame(const mynah_tts_model *model, const float *decoder_last,
                              const float *decoder_dev_last,
                              unsigned *codes, size_t raw_offset, size_t code_stride,
                              size_t generated_raw_length, size_t min_raw_length,
                              float temperature, unsigned topk,
                              uint64_t *rng_state, int *saw_eos,
                              size_t *eos_frame, char *error, size_t error_capacity) {
    const size_t width = model->info.hidden_dim;
    const size_t stream_count = model->info.codebook_count * model->info.frame_stacking_factor;
    const local_projection_cache *projection_cache =
        (const local_projection_cache *)model->local_projection_cache;
    const int reuse_workspace = getenv("MYNAH_LOCAL_STEP_ALLOCS") == NULL;
    local_cache lc;
    local_workspace workspace;
    memset(&lc, 0, sizeof(lc));
    memset(&workspace, 0, sizeof(workspace));
    float *row_in = allocate_floats(width, error, error_capacity);
    float *row_out = allocate_floats(width, error, error_capacity);
    if (row_in == NULL || row_out == NULL ||
        local_cache_init(model, &lc, stream_count + 1u, error, error_capacity) != 0 ||
        (reuse_workspace &&
         local_workspace_init(&workspace, &lc, error, error_capacity) != 0)) {
        free(row_in);
        free(row_out);
        local_workspace_free(&workspace);
        local_cache_free(&lc);
        return -1;
    }
    const size_t vocab = model->info.audio_vocab_size;
    const int sampling = temperature > 0.0f && topk > 1u && rng_state != NULL;
    const char *metal_local_env = getenv("MYNAH_METAL_GPU_LOCAL");
    const char *metal_attention_local_env = getenv("MYNAH_METAL_GPU_ATTENTION");
    const int gpu_local = lc.gpu_ready && lc.dev_backend != NULL &&
        (metal_local_env == NULL || strcmp(metal_local_env, "0") != 0) &&
        (strcmp(mynah_backend_name(lc.dev_backend), "metal") != 0 ||
         (metal_attention_local_env != NULL &&
          strcmp(metal_attention_local_env, "1") == 0));
    const size_t top_count = topk < vocab ? topk : vocab;
    const int reuse_sampler = getenv("MYNAH_LOCAL_SAMPLER_ALLOCS") == NULL;
    float *shared_logits = reuse_sampler
        ? allocate_floats(vocab, error, error_capacity) : NULL;
    size_t *shared_top_indices = reuse_sampler && sampling
        ? (size_t *)malloc(top_count * sizeof(*shared_top_indices)) : NULL;
    float *shared_top_logits = reuse_sampler && sampling
        ? (float *)malloc(top_count * sizeof(*shared_top_logits)) : NULL;
    if ((reuse_sampler && shared_logits == NULL) ||
        (reuse_sampler && sampling &&
         (shared_top_indices == NULL || shared_top_logits == NULL))) {
        free(shared_logits);
        free(shared_top_indices);
        free(shared_top_logits);
        free(row_in);
        free(row_out);
        local_workspace_free(&workspace);
        local_cache_free(&lc);
        graph_error(error, error_capacity, "out of memory allocating local sampler scratch");
        return -1;
    }
    if (!gpu_local || decoder_dev_last == NULL)
        memcpy(row_in, decoder_last, width * sizeof(float));
    if (gpu_local && ((decoder_dev_last != NULL
                           ? mynah_backend_copy_dev(lc.dev_backend, lc.dev_input,
                                                    decoder_dev_last, width,
                                                    error, error_capacity)
                           : mynah_backend_h2d(lc.dev_backend, decoder_last,
                                               lc.dev_input, width,
                                               error, error_capacity)) != 0)) {
        free(shared_logits);
        free(shared_top_indices);
        free(shared_top_logits);
        free(row_in);
        free(row_out);
        local_workspace_free(&workspace);
        local_cache_free(&lc);
        return -1;
    }
    *saw_eos = 0;
    *eos_frame = SIZE_MAX;
    char name[160];
    for (size_t stream = 0; stream < stream_count; ++stream) {
        local_workspace *step_workspace = reuse_workspace ? &workspace : NULL;
        float *dev_row_out = NULL;
        const int step_failed = gpu_local
            ? local_step_device(model, &lc, lc.dev_input, &dev_row_out,
                                error, error_capacity)
            : local_step(model, &lc, step_workspace, row_in, row_out,
                         error, error_capacity);
        if (step_failed != 0) {
            free(shared_logits);
            free(shared_top_indices);
            free(shared_top_logits);
            free(row_in);
            free(row_out);
            local_workspace_free(&workspace);
            local_cache_free(&lc);
            return -1;
        }
        const float *bias_data = NULL;
        if (projection_cache != NULL && stream < projection_cache->stream_count) {
            bias_data = projection_cache->projection_biases[stream];
        } else {
            snprintf(name, sizeof(name),
                     "local_transformer_out_projections.%zu.bias", stream);
            mynah_tensor bias;
            if (tensor(model->tts, name, &bias, error, error_capacity) != 0) {
                free(shared_logits);
                free(shared_top_indices);
                free(shared_top_logits);
                free(row_in);
                free(row_out);
                local_workspace_free(&workspace);
                local_cache_free(&lc);
                return -1;
            }
            bias_data = bias.data;
        }
        const int forbid_eos = generated_raw_length < min_raw_length;
        const unsigned eos_id = model->info.audio_eos_id;
        snprintf(name, sizeof(name), "local_transformer_out_projections.%zu.weight", stream);
        unsigned argmax = 0;
        const float *projection_weight = NULL;
        if (projection_cache != NULL && stream < projection_cache->stream_count) {
            projection_weight = projection_cache->projection_weights[stream];
        } else {
            mynah_tensor projection;
            if (tensor(model->tts, name, &projection, error, error_capacity) != 0) {
                free(shared_logits);
                free(shared_top_indices);
                free(shared_top_logits);
                free(row_in);
                free(row_out);
                local_workspace_free(&workspace);
                local_cache_free(&lc);
                return -1;
            }
            projection_weight = projection.data;
        }
        float *logits = NULL;
        int fused = 1;
        if (gpu_local) {
            if (mynah_backend_matmul_d2d(lc.dev_backend, dev_row_out, lc.dev_logits,
                                          1u, width, vocab, projection_weight,
                                          bias_data, error, error_capacity) != 0) {
                free(shared_logits);
                free(shared_top_indices);
                free(shared_top_logits);
                free(row_in);
                free(row_out);
                local_workspace_free(&workspace);
                local_cache_free(&lc);
                return -1;
            }
            if (!sampling) {
                fused = 0;
                if (mynah_backend_argmax_dev(lc.dev_backend, lc.dev_logits, vocab,
                                              model->info.codebook_size, eos_id,
                                              !forbid_eos, &argmax, error,
                                              error_capacity) != 0) {
                    free(shared_logits);
                    free(shared_top_indices);
                    free(shared_top_logits);
                    free(row_in);
                    free(row_out);
                    local_workspace_free(&workspace);
                    local_cache_free(&lc);
                    return -1;
                }
            } else {
                logits = shared_logits != NULL
                    ? shared_logits : allocate_floats(vocab, error, error_capacity);
                if (logits == NULL ||
                    mynah_backend_d2h(lc.dev_backend, lc.dev_logits, logits, vocab,
                                      error, error_capacity) != 0) {
                    if (logits != shared_logits) free(logits);
                    free(shared_logits);
                    free(shared_top_indices);
                    free(shared_top_logits);
                    free(row_in);
                    free(row_out);
                    local_workspace_free(&workspace);
                    local_cache_free(&lc);
                    return -1;
                }
            }
        } else if (!sampling &&
                   (getenv("MYNAH_FUSED_GREEDY") == NULL ||
                    strcmp(getenv("MYNAH_FUSED_GREEDY"), "0") != 0)) {
            fused = mynah_qmat_greedy_argmax_resolved(
                model->qcache, name, projection_weight, row_out, width, vocab,
                bias_data, model->info.codebook_size, eos_id, !forbid_eos,
                &argmax, error, error_capacity);
            if (fused < 0) {
                free(shared_logits);
                free(shared_top_indices);
                free(shared_top_logits);
                free(row_in);
                free(row_out);
                local_workspace_free(&workspace);
                local_cache_free(&lc);
                return -1;
            }
        }
        if (!gpu_local && fused != 0) {
            logits = shared_logits != NULL
                ? shared_logits : allocate_floats(vocab, error, error_capacity);
            if (logits == NULL ||
                mynah_qmat_linear_resolved(model->qcache, model->backend, name,
                                           projection_weight, row_out, logits,
                                           1u, width, vocab, bias_data,
                                           error, error_capacity) != 0) {
                if (logits != shared_logits) free(logits);
                free(shared_logits);
                free(shared_top_indices);
                free(shared_top_logits);
                free(row_in);
                free(row_out);
                local_workspace_free(&workspace);
                local_cache_free(&lc);
                return -1;
            }
        }
        if (fused != 0) {
            float best = -FLT_MAX;
            for (size_t candidate = 0; candidate < vocab; ++candidate) {
                const int is_code = candidate < model->info.codebook_size;
                const int is_eos = candidate == eos_id;
                if (!is_code && !(is_eos && !forbid_eos)) {
                    logits[candidate] = -FLT_MAX;
                    continue;
                }
                if (logits[candidate] > best) {
                    best = logits[candidate];
                    argmax = (unsigned)candidate;
                }
            }
        }
        unsigned value = argmax;
        /* argmax_or_multinomial: EOS fires if either the greedy or the sampled
         * token is AUDIO_EOS in any codebook of this frame. */
        int stream_eos = (argmax == eos_id);
        if (sampling) {
            size_t *top_indices = shared_top_indices != NULL
                ? shared_top_indices : (size_t *)malloc(top_count * sizeof(*top_indices));
            float *top_logits = shared_top_logits != NULL
                ? shared_top_logits : (float *)malloc(top_count * sizeof(*top_logits));
            if (top_indices == NULL || top_logits == NULL) {
                if (top_indices != shared_top_indices) free(top_indices);
                if (top_logits != shared_top_logits) free(top_logits);
                if (logits != shared_logits) free(logits);
                free(shared_logits);
                free(shared_top_indices);
                free(shared_top_logits);
                free(row_in);
                free(row_out);
                local_workspace_free(&workspace);
                local_cache_free(&lc);
                graph_error(error, error_capacity, "out of memory in audio sampler");
                return -1;
            }
            size_t used = 0;
            for (size_t candidate = 0; candidate < vocab; ++candidate) {
                size_t insert = used;
                while (insert > 0 && logits[candidate] > top_logits[insert - 1u]) --insert;
                if (used < top_count) {
                    for (size_t j = used; j > insert; --j) {
                        top_indices[j] = top_indices[j - 1u];
                        top_logits[j] = top_logits[j - 1u];
                    }
                    top_indices[insert] = candidate;
                    top_logits[insert] = logits[candidate];
                    ++used;
                } else if (insert < top_count) {
                    for (size_t j = top_count - 1u; j > insert; --j) {
                        top_indices[j] = top_indices[j - 1u];
                        top_logits[j] = top_logits[j - 1u];
                    }
                    top_indices[insert] = candidate;
                    top_logits[insert] = logits[candidate];
                }
            }
            float maximum = top_logits[0];
            float total = 0.0f;
            for (size_t j = 0; j < used; ++j) {
                top_logits[j] = expf((top_logits[j] - maximum) / temperature);
                total += top_logits[j];
            }
            uint64_t state = *rng_state;
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            *rng_state = state == 0 ? UINT64_C(0x9e3779b97f4a7c15) : state;
            const float draw = ((float)((*rng_state >> 40) & UINT64_C(0xffffff)) /
                                (float)UINT64_C(0x1000000)) * total;
            float cumulative = 0.0f;
            for (size_t j = 0; j < used; ++j) {
                cumulative += top_logits[j];
                if (draw <= cumulative || j + 1u == used) {
                    value = (unsigned)top_indices[j];
                    break;
                }
            }
            if (top_indices != shared_top_indices) free(top_indices);
            if (top_logits != shared_top_logits) free(top_logits);
        }
        if (logits != shared_logits) free(logits);
        if (value == eos_id) stream_eos = 1;
        if (stream_eos) {
            *saw_eos = 1;
            const size_t frame = stream / model->info.codebook_count;
            if (frame < *eos_frame) *eos_frame = frame;
        }
        /* Feed the sampled token (EOS included) back into the local
         * transformer, but collapse any special token to 0 in the codes we
         * hand to the codec so it only ever sees real FSQ indices. */
        const unsigned emit = value < vocab ? value : 0u;
        const unsigned store = value < model->info.codebook_size ? value : 0u;
        const size_t fs = stream / model->info.codebook_count;
        const size_t codebook = stream % model->info.codebook_count;
        codes[codebook * code_stride + raw_offset + fs] = store;
        const float *audio_table_data = NULL;
        if (projection_cache != NULL && stream < projection_cache->stream_count) {
            audio_table_data = projection_cache->audio_embeddings[stream];
        } else {
            snprintf(name, sizeof(name), "audio_embeddings.%zu.weight", stream);
            mynah_tensor audio_table;
            if (tensor(model->tts, name, &audio_table, error, error_capacity) != 0) {
                free(shared_logits);
                free(shared_top_indices);
                free(shared_top_logits);
                free(row_in);
                free(row_out);
                local_workspace_free(&workspace);
                local_cache_free(&lc);
                return -1;
            }
            audio_table_data = audio_table.data;
        }
        /* The embedding of this stream's token becomes the input row for the
         * next local-transformer position. */
        if (gpu_local) {
            if (mynah_backend_h2d(lc.dev_backend,
                                  audio_table_data + (size_t)emit * width,
                                  lc.dev_input, width, error, error_capacity) != 0) {
                free(shared_logits);
                free(shared_top_indices);
                free(shared_top_logits);
                free(row_in);
                free(row_out);
                local_workspace_free(&workspace);
                local_cache_free(&lc);
                return -1;
            }
        } else {
            memcpy(row_in, audio_table_data + (size_t)emit * width,
                   width * sizeof(float));
        }
    }
    free(shared_logits);
    free(shared_top_indices);
    free(shared_top_logits);
    free(row_in);
    free(row_out);
    local_workspace_free(&workspace);
    local_cache_free(&lc);
    return 0;
}

typedef struct {
    double pack_seconds;
    double gemm_seconds;
    double transpose_seconds;
    double snake_seconds;
    double bnns_create_seconds;
    double bnns_apply_seconds;
    double bnns_destroy_seconds;
    size_t calls;
    size_t transpose_calls;
    size_t snake_calls;
} codec_conv_profile;

#if defined(MYNAH_USE_ACCELERATE)
typedef struct {
    const float *weight;
    const float *bias;
    size_t in_channels;
    size_t out_channels;
    size_t length;
    size_t kernel;
    size_t dilation;
    BNNSFilter filter;
} codec_bnns_entry;

struct codec_bnns_cache {
    codec_bnns_entry *entries;
    size_t count;
    size_t capacity;
    pthread_mutex_t mutex;
};

static void codec_bnns_cache_free(codec_bnns_cache *cache) {
    if (cache == NULL) return;
    pthread_mutex_lock(&cache->mutex);
    for (size_t i = 0; i < cache->count; ++i) {
        if (cache->entries[i].filter != NULL)
            BNNSFilterDestroy(cache->entries[i].filter);
    }
    free(cache->entries);
    pthread_mutex_unlock(&cache->mutex);
    pthread_mutex_destroy(&cache->mutex);
    free(cache);
}
#endif

#if defined(MYNAH_USE_OPENBLAS) && !defined(MYNAH_USE_ACCELERATE)
typedef struct {
    const float *weight;
    size_t in_channels;
    size_t out_channels;
    size_t kernel;
    float *packed;
} codec_tap_entry;

struct codec_bnns_cache {
    codec_tap_entry *entries;
    size_t count;
    size_t capacity;
    pthread_mutex_t mutex;
};
#endif

void *mynah_graph_codec_cache_new(void) {
#if defined(MYNAH_USE_ACCELERATE)
    codec_bnns_cache *cache = (codec_bnns_cache *)calloc(1, sizeof(*cache));
    if (cache == NULL) return NULL;
    if (pthread_mutex_init(&cache->mutex, NULL) != 0) {
        free(cache);
        return NULL;
    }
    return cache;
#elif defined(MYNAH_USE_OPENBLAS)
    codec_bnns_cache *cache = (codec_bnns_cache *)calloc(1, sizeof(*cache));
    if (cache == NULL) return NULL;
    if (pthread_mutex_init(&cache->mutex, NULL) != 0) {
        free(cache);
        return NULL;
    }
    return cache;
#else
    return NULL;
#endif
}

void mynah_graph_codec_cache_free(void *opaque) {
#if defined(MYNAH_USE_ACCELERATE)
    codec_bnns_cache_free((codec_bnns_cache *)opaque);
#elif defined(MYNAH_USE_OPENBLAS)
    codec_bnns_cache *cache = (codec_bnns_cache *)opaque;
    if (cache == NULL) return;
    pthread_mutex_lock(&cache->mutex);
    for (size_t i = 0; i < cache->count; ++i) free(cache->entries[i].packed);
    free(cache->entries);
    pthread_mutex_unlock(&cache->mutex);
    pthread_mutex_destroy(&cache->mutex);
    free(cache);
#else
    (void)opaque;
#endif
}

#if defined(MYNAH_USE_OPENBLAS) && !defined(MYNAH_USE_ACCELERATE)
static float *codec_cached_taps(codec_bnns_cache *cache, const float *weight,
                                size_t in_channels, size_t out_channels,
                                size_t kernel) {
    const char *cache_env = getenv("MYNAH_CODEC_TAP_CACHE");
    if (cache_env != NULL && strcmp(cache_env, "0") == 0) return NULL;
    if (cache == NULL || weight == NULL || in_channels == 0u ||
        out_channels == 0u || kernel == 0u) return NULL;
    pthread_mutex_lock(&cache->mutex);
    for (size_t i = 0; i < cache->count; ++i) {
        codec_tap_entry *entry = &cache->entries[i];
        if (entry->weight == weight && entry->in_channels == in_channels &&
            entry->out_channels == out_channels && entry->kernel == kernel) {
            float *packed = entry->packed;
            pthread_mutex_unlock(&cache->mutex);
            return packed;
        }
    }
    if (kernel > SIZE_MAX / out_channels ||
        kernel * out_channels > SIZE_MAX / in_channels ||
        kernel * out_channels * in_channels > SIZE_MAX / sizeof(float)) {
        pthread_mutex_unlock(&cache->mutex);
        return NULL;
    }
    const size_t per_tap = out_channels * in_channels;
    const size_t total = kernel * per_tap;
    float *packed = (float *)malloc(total * sizeof(*packed));
    if (packed == NULL) {
        pthread_mutex_unlock(&cache->mutex);
        return NULL;
    }
    for (size_t k = 0; k < kernel; ++k) {
        float *tap = packed + k * per_tap;
        for (size_t o = 0; o < out_channels; ++o) {
            for (size_t i = 0; i < in_channels; ++i) {
                tap[o * in_channels + i] =
                    weight[(o * in_channels + i) * kernel + k];
            }
        }
    }
    if (cache->count == cache->capacity) {
        const size_t next_capacity = cache->capacity == 0u
            ? 4u : cache->capacity > SIZE_MAX / 2u
                ? 0u : cache->capacity * 2u;
        if (next_capacity == 0u || next_capacity > SIZE_MAX / sizeof(*cache->entries)) {
            free(packed);
            pthread_mutex_unlock(&cache->mutex);
            return NULL;
        }
        codec_tap_entry *entries = (codec_tap_entry *)realloc(
            cache->entries, next_capacity * sizeof(*entries));
        if (entries == NULL) {
            free(packed);
            pthread_mutex_unlock(&cache->mutex);
            return NULL;
        }
        cache->entries = entries;
        cache->capacity = next_capacity;
    }
    cache->entries[cache->count++] = (codec_tap_entry){
        weight, in_channels, out_channels, kernel, packed};
    pthread_mutex_unlock(&cache->mutex);
    return packed;
}
#endif

#if defined(MYNAH_USE_ACCELERATE)
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif
static int conv1d_causal_bnns(const float *weight, const float *bias,
                              const float *input, float *output,
                              size_t in_channels, size_t out_channels,
                              size_t length, size_t kernel, size_t dilation,
                              codec_bnns_cache *cache,
                              codec_conv_profile *profile) {
    if (length == 0u || kernel == 0u || dilation == 0u ||
        (kernel - 1u) > SIZE_MAX / dilation) {
        return -1;
    }
    BNNSFilter filter = NULL;
    int retained = 0;
    if (cache != NULL) {
        for (size_t i = 0; i < cache->count; ++i) {
            codec_bnns_entry *entry = &cache->entries[i];
            if (entry->weight == weight && entry->bias == bias &&
                entry->in_channels == in_channels && entry->out_channels == out_channels &&
                entry->length == length && entry->kernel == kernel &&
                entry->dilation == dilation) {
                filter = entry->filter;
                retained = 1;
                break;
            }
        }
    }
    BNNSLayerParametersConvolution parameters;
    memset(&parameters, 0, sizeof(parameters));
    parameters.i_desc.layout = BNNSDataLayoutImageCHW;
    parameters.i_desc.size[0] = length;
    parameters.i_desc.size[1] = 1u;
    parameters.i_desc.size[2] = in_channels;
    parameters.i_desc.data_type = BNNSDataTypeFloat32;
    parameters.w_desc.layout = BNNSDataLayoutConvolutionWeightsOIHW;
    parameters.w_desc.size[0] = kernel;
    parameters.w_desc.size[1] = 1u;
    parameters.w_desc.size[2] = in_channels;
    parameters.w_desc.size[3] = out_channels;
    parameters.w_desc.data = (void *)weight;
    parameters.w_desc.data_type = BNNSDataTypeFloat32;
    parameters.o_desc.layout = BNNSDataLayoutImageCHW;
    parameters.o_desc.size[0] = length;
    parameters.o_desc.size[1] = 1u;
    parameters.o_desc.size[2] = out_channels;
    parameters.o_desc.data_type = BNNSDataTypeFloat32;
    parameters.bias.layout = BNNSDataLayoutVector;
    parameters.bias.size[0] = out_channels;
    parameters.bias.data = (void *)bias;
    parameters.bias.data_type = BNNSDataTypeFloat32;
    parameters.activation.function = BNNSActivationFunctionIdentity;
    parameters.x_stride = 1u;
    parameters.y_stride = 1u;
    parameters.x_dilation_stride = dilation;
    parameters.y_dilation_stride = 1u;
    parameters.pad[0] = (kernel - 1u) * dilation;
    double operation_start = profile != NULL ? phase_seconds() : 0.0;
    if (cache != NULL) pthread_mutex_lock(&cache->mutex);
    if (filter == NULL) {
        filter = BNNSFilterCreateLayerConvolution(&parameters, NULL);
        if (filter == NULL) {
            if (cache != NULL) pthread_mutex_unlock(&cache->mutex);
            return -1;
        }
        if (profile != NULL) {
            profile->bnns_create_seconds += phase_seconds() - operation_start;
            operation_start = phase_seconds();
        }
        if (cache != NULL) {
            if (cache->count == cache->capacity) {
                const size_t next = cache->capacity == 0 ? 8u : cache->capacity * 2u;
                codec_bnns_entry *grown = (codec_bnns_entry *)realloc(
                    cache->entries, next * sizeof(*grown));
                if (grown != NULL) {
                    cache->entries = grown;
                    cache->capacity = next;
                }
            }
            if (cache->count < cache->capacity) {
                cache->entries[cache->count++] = (codec_bnns_entry){
                    weight, bias, in_channels, out_channels, length, kernel, dilation, filter};
                retained = 1;
            }
        }
    }
    const int result = BNNSFilterApply(filter, input, output);
    if (profile != NULL) {
        profile->bnns_apply_seconds += phase_seconds() - operation_start;
        operation_start = phase_seconds();
    }
    if (!retained) {
        BNNSFilterDestroy(filter);
        if (profile != NULL) {
            profile->bnns_destroy_seconds += phase_seconds() - operation_start;
        }
    }
    if (cache != NULL) pthread_mutex_unlock(&cache->mutex);
    return result;
}
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
#endif

int mynah_graph_bnns_self_test(char *error, size_t error_capacity) {
#if defined(MYNAH_USE_ACCELERATE)
    enum { IN_CHANNELS = 2, OUT_CHANNELS = 3, LENGTH = 9, KERNEL = 3 };
    float input[IN_CHANNELS * LENGTH];
    float weight[OUT_CHANNELS * IN_CHANNELS * KERNEL];
    float bias[OUT_CHANNELS];
    float expected[OUT_CHANNELS * LENGTH];
    float actual[OUT_CHANNELS * LENGTH];
    for (size_t i = 0; i < IN_CHANNELS * LENGTH; ++i) {
        input[i] = sinf(0.17f * (float)i) - 0.2f;
    }
    for (size_t i = 0; i < OUT_CHANNELS * IN_CHANNELS * KERNEL; ++i) {
        weight[i] = cosf(0.11f * (float)(i + 3u)) * 0.25f;
    }
    for (size_t o = 0; o < OUT_CHANNELS; ++o) bias[o] = (float)o * 0.1f - 0.05f;
    const size_t dilation = 2u;
    for (size_t o = 0; o < OUT_CHANNELS; ++o) {
        for (size_t t = 0; t < LENGTH; ++t) {
            float value = bias[o];
            for (size_t i = 0; i < IN_CHANNELS; ++i) {
                for (size_t k = 0; k < KERNEL; ++k) {
                    const long source = (long)t -
                        (long)((KERNEL - 1u) * dilation) +
                        (long)(k * dilation);
                    if (source >= 0) {
                        value += weight[(o * IN_CHANNELS + i) * KERNEL + k] *
                                 input[i * LENGTH + (size_t)source];
                    }
                }
            }
            expected[o * LENGTH + t] = value;
        }
    }
    if (conv1d_causal_bnns(weight, bias, input, actual,
                           IN_CHANNELS, OUT_CHANNELS, LENGTH,
        KERNEL, dilation, NULL, NULL) != 0) {
        graph_error(error, error_capacity, "BNNS causal-conv self-test failed to apply");
        return -1;
    }
    for (size_t i = 0; i < OUT_CHANNELS * LENGTH; ++i) {
        const float tolerance = 2.0e-5f * (1.0f + fabsf(expected[i]));
        if (fabsf(actual[i] - expected[i]) > tolerance) {
            if (error != NULL && error_capacity > 0) {
                snprintf(error, error_capacity,
                         "BNNS causal-conv mismatch at %zu: got %.9g expected %.9g",
                         i, (double)actual[i], (double)expected[i]);
            }
            return -1;
        }
    }
#else
    (void)error;
    (void)error_capacity;
#endif
    return 0;
}

static int conv1d_causal(const mynah_safetensors *file, const mynah_backend *backend,
                         codec_bnns_cache *bnns_cache,
                         const char *weight_name,
                         const char *bias_name, const float *input, float *output,
                         size_t in_channels, size_t out_channels, size_t length,
                         size_t kernel, size_t dilation,
                         float *columns_workspace, size_t columns_capacity,
                         codec_conv_profile *profile,
                         char *error, size_t error_capacity) {
    (void)columns_workspace;
    (void)columns_capacity;
    mynah_tensor weight;
    mynah_tensor bias;
    if (tensor(file, weight_name, &weight, error, error_capacity) != 0 ||
        tensor(file, bias_name, &bias, error, error_capacity) != 0) return -1;
    if (profile != NULL) profile->calls++;
    const char *tap_env = getenv("MYNAH_CONV_TAP_GEMMS");
    int use_tap_gemms = tap_env != NULL && strcmp(tap_env, "0") != 0;
#if defined(MYNAH_USE_OPENBLAS)
    if (tap_env == NULL) use_tap_gemms = 1;
#endif
    /* GPU fast path: im2col + sgemm in one backend call. */
    if (backend != NULL && in_channels <= (size_t)INT_MAX &&
        out_channels <= (size_t)INT_MAX && length <= (size_t)INT_MAX &&
        kernel <= (size_t)INT_MAX && dilation <= (size_t)INT_MAX) {
        const double t0 = profile != NULL ? phase_seconds() : 0.0;
        if (mynah_backend_conv1d(backend, input, output,
                                 (int)in_channels, (int)out_channels, (int)length,
                                 (int)kernel, (int)dilation,
                                 weight.data, bias.data,
                                 error, error_capacity) == 0) {
            if (profile != NULL) {
                profile->gemm_seconds += phase_seconds() - t0;
            }
            return 0;
        }
        /* Fall through to CPU path on failure. */
    }
#if defined(MYNAH_USE_ACCELERATE)
    if (getenv("MYNAH_CODEC_SGEMM") == NULL &&
        conv1d_causal_bnns(weight.data, bias.data, input, output,
                           in_channels, out_channels, length,
            kernel, dilation, bnns_cache, profile) == 0) {
        return 0;
    }
#endif
    /* Seed the output with the per-channel bias; each kernel tap then adds its
     * contribution.  A causal conv1d is a sum of `kernel` shifted matmuls
     * (weight tap k) [out x in] . input[:, 0:length-shift], so on BLAS builds we
     * accumulate them with sgemm instead of the scalar quadruple loop, which is
     * the codec decoder's dominant cost. */
    for (size_t o = 0; o < out_channels; ++o) {
        for (size_t t = 0; t < length; ++t) output[o * length + t] = bias.data[o];
    }
    if (in_channels <= (size_t)INT_MAX && out_channels <= (size_t)INT_MAX &&
        length <= (size_t)INT_MAX) {
        size_t inner = 0;
        size_t column_count = 0;
        if (in_channels > 0 && !use_tap_gemms &&
            kernel <= SIZE_MAX / in_channels &&
            (inner = in_channels * kernel) <= (size_t)INT_MAX &&
            length <= SIZE_MAX / inner &&
            (column_count = inner * length) <= SIZE_MAX / sizeof(float)) {
            const double pack_start = profile != NULL ? phase_seconds() : 0.0;
            const int owns_columns = columns_workspace == NULL ||
                                     columns_capacity < column_count;
            float *columns = owns_columns
                ? (float *)malloc(column_count * sizeof(*columns))
                : columns_workspace;
            if (columns != NULL) {
                for (size_t i = 0; i < in_channels; ++i) {
                    const float *source_row = input + i * length;
                    for (size_t k = 0; k < kernel; ++k) {
                        const size_t shift = (kernel - 1u - k) * dilation;
                        float *column = columns + (i * kernel + k) * length;
                        const size_t zero_count = shift < length ? shift : length;
                        memset(column, 0, zero_count * sizeof(*column));
                        if (shift < length) {
                            memcpy(column + shift, source_row,
                                   (length - shift) * sizeof(*column));
                        }
                    }
                }
                if (profile != NULL) profile->pack_seconds += phase_seconds() - pack_start;
                const double gemm_start = profile != NULL ? phase_seconds() : 0.0;
                graph_sgemm(backend, 0, 0, (int)out_channels, (int)length, (int)inner, 1.0f, weight.data, (int)inner, columns, (int)length, 1.0f, output, (int)length, error, error_capacity);
                if (profile != NULL) profile->gemm_seconds += phase_seconds() - gemm_start;
                if (owns_columns) free(columns);
                return 0;
            }
        }
        if (in_channels == 0u || out_channels > SIZE_MAX / in_channels ||
            out_channels * in_channels > SIZE_MAX / sizeof(float)) {
            graph_error(error, error_capacity,
                        "causal conv1d tap workspace size overflow");
            return -1;
        }
    float *wk = NULL;
    int owns_wk = 0;
#if defined(MYNAH_USE_OPENBLAS) && !defined(MYNAH_USE_ACCELERATE)
    wk = codec_cached_taps(bnns_cache, weight.data,
                           in_channels, out_channels, kernel);
#endif
    if (wk == NULL) {
        wk = (float *)malloc(out_channels * in_channels * sizeof(float));
        owns_wk = 1;
    }
    if (wk == NULL) {
            graph_error(error, error_capacity, "out of memory in causal conv1d");
            return -1;
        }
        for (size_t k = 0; k < kernel; ++k) {
            const size_t shift = (kernel - 1u - k) * dilation;
            if (shift >= length) continue;
        if (owns_wk) {
            const double pack_start = profile != NULL ? phase_seconds() : 0.0;
            for (size_t o = 0; o < out_channels; ++o) {
                for (size_t i = 0; i < in_channels; ++i) {
                    wk[o * in_channels + i] =
                        weight.data[(o * in_channels + i) * kernel + k];
                }
            }
            if (profile != NULL) profile->pack_seconds += phase_seconds() - pack_start;
        }
        const size_t n = length - shift;
        const float *tap_weights = owns_wk
            ? wk : wk + k * out_channels * in_channels;
        const double gemm_start = profile != NULL ? phase_seconds() : 0.0;
        graph_sgemm(backend, 0, 0, (int)out_channels, (int)n,
                    (int)in_channels, 1.0f, tap_weights, (int)in_channels,
                    input, (int)length, 1.0f, output + shift, (int)length,
                    error, error_capacity);
            if (profile != NULL) profile->gemm_seconds += phase_seconds() - gemm_start;
        }
    if (owns_wk) free(wk);
        return 0;
    }
    for (size_t o = 0; o < out_channels; ++o) {
        for (size_t t = 0; t < length; ++t) {
            float value = bias.data[o];
            for (size_t i = 0; i < in_channels; ++i) {
                for (size_t k = 0; k < kernel; ++k) {
                    const long source = (long)t - (long)((kernel - 1u) * dilation) + (long)(k * dilation);
                    if (source >= 0) value += weight.data[(o * in_channels + i) * kernel + k] *
                                             input[i * length + (size_t)source];
                }
            }
            output[o * length + t] = value;
        }
    }
    return 0;
}

/* Causal transposed conv1d.  Each output channel `o` owns a disjoint output row
 * and reads only its group's input channels, so parallelizing over `o` keeps
 * the per-row accumulation order and is bit-identical to the serial form. */
typedef struct {
    const float *input;
    float *output;
    const float *weight;
    const float *bias;
    size_t length;
    size_t output_length;
    size_t kernel;
    size_t stride;
    size_t in_per_group;
    size_t out_per_group;
} convt_ctx;

static void convt_channel(void *ctx, int oi) {
    const convt_ctx *x = (const convt_ctx *)ctx;
    const size_t o = (size_t)oi;
    const size_t group = o / x->out_per_group;
    const size_t o_local = o % x->out_per_group;
    float *row = x->output + o * x->output_length;
    for (size_t t = 0; t < x->output_length; ++t) row[t] = x->bias[o];
    const size_t i0 = group * x->in_per_group;
    for (size_t i = i0; i < i0 + x->in_per_group; ++i) {
        const float *in_row = x->input + i * x->length;
        const float *w = x->weight + (i * x->out_per_group + o_local) * x->kernel;
        for (size_t t = 0; t < x->length; ++t) {
            const float in_v = in_row[t];
            for (size_t k = 0; k < x->kernel; ++k) {
                const size_t position = t * x->stride + k;
                if (position < x->output_length) row[position] += in_v * w[k];
            }
        }
    }
}

static int conv_transpose_causal(const mynah_safetensors *file, const char *weight_name,
                                 const char *bias_name, const float *input, float *output,
                                 size_t in_channels, size_t out_channels, size_t length,
                                 size_t kernel, size_t stride, size_t groups,
                                 codec_conv_profile *profile,
                                 char *error, size_t error_capacity) {
    const double operation_start = profile != NULL ? phase_seconds() : 0.0;
    mynah_tensor weight;
    mynah_tensor bias;
    if (tensor(file, weight_name, &weight, error, error_capacity) != 0 ||
        tensor(file, bias_name, &bias, error, error_capacity) != 0) return -1;
    const size_t full_length = (length - 1u) * stride + kernel;
    const size_t trim = kernel - stride;
    const size_t output_length = full_length - trim;
    convt_ctx ctx = {input, output, weight.data, bias.data, length, output_length,
                     kernel, stride, in_channels / groups, out_channels / groups};
    mynah_parallel_for((int)out_channels, convt_channel, &ctx);
    if (profile != NULL) {
        profile->transpose_seconds += phase_seconds() - operation_start;
        profile->transpose_calls++;
    }
    return 0;
}

/* Snake activation on the first half of the channels, leaky-ReLU on the rest.
 * Each channel is an independent row, so this parallelizes bit-identically. */
typedef struct {
    float *signal;
    const float *alpha;
    size_t snake_channels;
    size_t length;
} snake_ctx;

static void snake_channel(void *ctx, int c) {
    const snake_ctx *s = (const snake_ctx *)ctx;
    float *row = s->signal + (size_t)c * s->length;
    if ((size_t)c < s->snake_channels) {
        const float a = s->alpha[c];
        for (size_t t = 0; t < s->length; ++t) {
            const float value = row[t];
            const float sn = sinf(a * value);
            row[t] = value + sn * sn / (a + 1.0e-9f);
        }
    } else {
        for (size_t t = 0; t < s->length; ++t) {
            if (row[t] < 0.0f) row[t] *= 0.01f;
        }
    }
}

static int half_snake(const mynah_safetensors *file, const mynah_backend *backend,
                      const char *alpha_name,
                      float *signal, size_t channels, size_t length,
                      codec_conv_profile *profile,
                      char *error, size_t error_capacity) {
    const double operation_start = profile != NULL ? phase_seconds() : 0.0;
    mynah_tensor alpha;
    if (tensor(file, alpha_name, &alpha, error, error_capacity) != 0) return -1;
    const size_t snake_channels = channels / 2u;
    /* GPU path: upload signal → snake kernel (alpha uploaded internally) → download. */
    if (backend != NULL && snake_channels > 0u) {
        float *dev_signal = NULL;
        const size_t n = channels * length;
        if (mynah_backend_upload(backend, signal, n, &dev_signal, error, error_capacity) == 0) {
            if (mynah_backend_snake_dev(backend, dev_signal, alpha.data,
                                        channels, length, snake_channels,
                                        error, error_capacity) == 0) {
                mynah_backend_download(backend, dev_signal, signal, n,
                                       error, error_capacity);
                mynah_backend_sync(backend, error, error_capacity);
                if (profile != NULL) {
                    profile->snake_seconds += phase_seconds() - operation_start;
                    profile->snake_calls++;
                }
                return 0;
            }
        }
        /* Fall through to CPU path on failure. */
    }
#if defined(MYNAH_USE_ACCELERATE)
    if (getenv("MYNAH_SNAKE_SCALAR") == NULL && snake_channels > 0u &&
        length <= SIZE_MAX / snake_channels) {
        const size_t count = snake_channels * length;
        if (count <= (size_t)INT_MAX && count <= SIZE_MAX / sizeof(float)) {
            float *sines = (float *)malloc(count * sizeof(*sines));
            if (sines != NULL) {
                for (size_t c = 0; c < snake_channels; ++c) {
                    const float a = alpha.data[c];
                    vDSP_vsmul(signal + c * length, 1, &a, sines + c * length, 1,
                               (vDSP_Length)length);
                }
                const int vector_count = (int)count;
                vvsinf(sines, sines, &vector_count);
                vDSP_vsq(sines, 1, sines, 1, (vDSP_Length)count);
                for (size_t c = 0; c < snake_channels; ++c) {
                    const float inverse_alpha = 1.0f / (alpha.data[c] + 1.0e-9f);
                    float *row = signal + c * length;
                    vDSP_vsma(sines + c * length, 1, &inverse_alpha,
                              row, 1, row, 1, (vDSP_Length)length);
                }
                free(sines);
                for (size_t c = snake_channels; c < channels; ++c) {
                    float *row = signal + c * length;
                    for (size_t t = 0; t < length; ++t) {
                        if (row[t] < 0.0f) row[t] *= 0.01f;
                    }
                }
                if (profile != NULL) {
                    profile->snake_seconds += phase_seconds() - operation_start;
                    profile->snake_calls++;
                }
                return 0;
            }
        }
    }
#endif
    snake_ctx ctx = {signal, alpha.data, channels / 2u, length};
    mynah_parallel_for((int)channels, snake_channel, &ctx);
    if (profile != NULL) {
        profile->snake_seconds += phase_seconds() - operation_start;
        profile->snake_calls++;
    }
    return 0;
}

static int res_layer(const mynah_safetensors *file, const mynah_backend *backend,
                     codec_bnns_cache *bnns_cache,
                     size_t stage, const float *input,
                     float *output, size_t channels, size_t length,
                     codec_conv_profile *profile, char *error, size_t error_capacity) {
    const size_t kernels[3] = {3u, 7u, 11u};
    const size_t dilations[3] = {1u, 3u, 5u};
    if (channels == 0u || length == 0u || channels > SIZE_MAX / length) {
        graph_error(error, error_capacity, "invalid codec residual workspace size");
        return -1;
    }
    const size_t elements = channels * length;
    float *branch = allocate_floats(elements, error, error_capacity);
    float *current = allocate_floats(elements, error, error_capacity);
    float *activated = allocate_floats(elements, error, error_capacity);
    float *residual = allocate_floats(elements, error, error_capacity);
    if (branch == NULL || current == NULL || activated == NULL || residual == NULL) {
        free(branch);
        free(current);
        free(activated);
        free(residual);
        return -1;
    }
    float *columns_workspace = NULL;
    size_t columns_capacity = 0;
    int needs_columns_workspace = 1;
#if defined(MYNAH_USE_ACCELERATE)
    if (getenv("MYNAH_CODEC_SGEMM") == NULL &&
        getenv("MYNAH_BNNS_IM2COL_WORKSPACE") == NULL) {
        needs_columns_workspace = 0;
    }
    if (needs_columns_workspace &&
        getenv("MYNAH_CODEC_CONV_ALLOCS") == NULL &&
        length > 0u && channels <= SIZE_MAX / 11u &&
        channels * 11u <= SIZE_MAX / length) {
        columns_capacity = channels * 11u * length;
        if (columns_capacity <= SIZE_MAX / sizeof(*columns_workspace)) {
            columns_workspace = (float *)malloc(
                columns_capacity * sizeof(*columns_workspace));
        }
        if (columns_workspace == NULL) columns_capacity = 0;
    }
#endif
    memset(output, 0, elements * sizeof(float));
    char name[256];
    for (size_t branch_index = 0; branch_index < 3u; ++branch_index) {
        memcpy(current, input, elements * sizeof(float));
        for (size_t dilation_index = 0; dilation_index < 3u; ++dilation_index) {
            snprintf(name, sizeof(name),
                     "audio_decoder.res_layers.%zu.res_blocks.%zu.res_blocks.%zu.input_activation.activation.snake_act.alpha",
                     stage, branch_index, dilation_index);
            memcpy(activated, current, elements * sizeof(float));
            if (half_snake(file, backend, name, activated, channels, length, profile,
                           error, error_capacity) != 0) break;
            char weight_name[256];
            char bias_name[256];
            snprintf(weight_name, sizeof(weight_name),
                     "audio_decoder.res_layers.%zu.res_blocks.%zu.res_blocks.%zu.input_conv.conv.weight",
                     stage, branch_index, dilation_index);
            snprintf(bias_name, sizeof(bias_name),
                     "audio_decoder.res_layers.%zu.res_blocks.%zu.res_blocks.%zu.input_conv.conv.bias",
                     stage, branch_index, dilation_index);
            if (conv1d_causal(file, backend, bnns_cache, weight_name, bias_name, activated, residual,
                              channels, channels, length, kernels[branch_index],
                              dilations[dilation_index],
                              columns_workspace, columns_capacity,
                              profile, error, error_capacity) != 0) break;
            snprintf(name, sizeof(name),
                     "audio_decoder.res_layers.%zu.res_blocks.%zu.res_blocks.%zu.skip_activation.activation.snake_act.alpha",
                     stage, branch_index, dilation_index);
            if (half_snake(file, backend, name, residual, channels, length, profile,
                           error, error_capacity) != 0) break;
            snprintf(weight_name, sizeof(weight_name),
                     "audio_decoder.res_layers.%zu.res_blocks.%zu.res_blocks.%zu.skip_conv.conv.weight",
                     stage, branch_index, dilation_index);
            snprintf(bias_name, sizeof(bias_name),
                     "audio_decoder.res_layers.%zu.res_blocks.%zu.res_blocks.%zu.skip_conv.conv.bias",
                     stage, branch_index, dilation_index);
            if (conv1d_causal(file, backend, bnns_cache, weight_name, bias_name, residual, branch,
                              channels, channels, length, kernels[branch_index], 1u,
                              columns_workspace, columns_capacity, profile,
                              error, error_capacity) != 0) break;
            for (size_t i = 0; i < elements; ++i) current[i] += branch[i];
        }
        for (size_t i = 0; i < elements; ++i) output[i] += current[i];
        if (error != NULL && error[0] != '\0') break;
    }
    for (size_t i = 0; i < elements; ++i) output[i] /= 3.0f;
    free(branch);
    free(current);
    free(activated);
    free(residual);
    free(columns_workspace);
    return error == NULL || error[0] == '\0' ? 0 : -1;
}

/* Metal-resident residual stack.  The three branches and three dilated blocks
 * reuse four device workspaces; no intermediate activation crosses back to C. */
static int res_layer_device(const mynah_safetensors *file, const mynah_backend *backend,
                            size_t stage, const float *dev_input, float *dev_output,
                            size_t channels, size_t length, char *error,
                            size_t error_capacity) {
    const size_t kernels[3] = {3u, 7u, 11u};
    const size_t dilations[3] = {1u, 3u, 5u};
    if (file == NULL || backend == NULL || dev_input == NULL || dev_output == NULL ||
        channels == 0u || length == 0u || channels > SIZE_MAX / length) return -1;
    const size_t elements = channels * length;
    float *branch = NULL, *current = NULL, *activated = NULL, *residual = NULL;
    int ok = mynah_backend_dev_alloc(backend, elements, &branch, error, error_capacity) == 0;
    ok = ok && mynah_backend_dev_alloc(backend, elements, &current, error, error_capacity) == 0;
    ok = ok && mynah_backend_dev_alloc(backend, elements, &activated, error, error_capacity) == 0;
    ok = ok && mynah_backend_dev_alloc(backend, elements, &residual, error, error_capacity) == 0;
    float *zeros = ok ? (float *)calloc(elements, sizeof(float)) : NULL;
    if (ok && zeros == NULL) ok = 0;
    if (ok && (mynah_backend_h2d(backend, zeros, dev_output, elements,
                                 error, error_capacity) != 0 ||
               mynah_backend_batch_begin(backend, error, error_capacity) != 0)) ok = 0;
    free(zeros);
    char name[256], weight_name[256], bias_name[256];
    for (size_t branch_index = 0; ok && branch_index < 3u; ++branch_index) {
        ok = mynah_backend_copy_dev(backend, current, dev_input, elements,
                                    error, error_capacity) == 0;
        for (size_t dilation_index = 0; ok && dilation_index < 3u; ++dilation_index) {
            mynah_tensor alpha, weight, bias;
            snprintf(name, sizeof(name),
                     "audio_decoder.res_layers.%zu.res_blocks.%zu.res_blocks.%zu.input_activation.activation.snake_act.alpha",
                     stage, branch_index, dilation_index);
            snprintf(weight_name, sizeof(weight_name),
                     "audio_decoder.res_layers.%zu.res_blocks.%zu.res_blocks.%zu.input_conv.conv.weight",
                     stage, branch_index, dilation_index);
            snprintf(bias_name, sizeof(bias_name),
                     "audio_decoder.res_layers.%zu.res_blocks.%zu.res_blocks.%zu.input_conv.conv.bias",
                     stage, branch_index, dilation_index);
            ok = tensor(file, name, &alpha, error, error_capacity) == 0 &&
                 tensor(file, weight_name, &weight, error, error_capacity) == 0 &&
                 tensor(file, bias_name, &bias, error, error_capacity) == 0 &&
                 mynah_backend_copy_dev(backend, activated, current, elements,
                                        error, error_capacity) == 0 &&
                 mynah_backend_snake_dev(backend, activated, alpha.data, channels,
                                         length, channels / 2u, error,
                                         error_capacity) == 0 &&
                 mynah_backend_conv1d(backend, activated, residual,
                                      (int)channels, (int)channels, (int)length,
                                      (int)kernels[branch_index],
                                      (int)dilations[dilation_index], weight.data,
                                      bias.data, error, error_capacity) == 0;
            snprintf(name, sizeof(name),
                     "audio_decoder.res_layers.%zu.res_blocks.%zu.res_blocks.%zu.skip_activation.activation.snake_act.alpha",
                     stage, branch_index, dilation_index);
            snprintf(weight_name, sizeof(weight_name),
                     "audio_decoder.res_layers.%zu.res_blocks.%zu.res_blocks.%zu.skip_conv.conv.weight",
                     stage, branch_index, dilation_index);
            snprintf(bias_name, sizeof(bias_name),
                     "audio_decoder.res_layers.%zu.res_blocks.%zu.res_blocks.%zu.skip_conv.conv.bias",
                     stage, branch_index, dilation_index);
            if (ok) ok = tensor(file, name, &alpha, error, error_capacity) == 0 &&
                         tensor(file, weight_name, &weight, error, error_capacity) == 0 &&
                         tensor(file, bias_name, &bias, error, error_capacity) == 0 &&
                         mynah_backend_snake_dev(backend, residual, alpha.data, channels,
                                                 length, channels / 2u, error,
                                                 error_capacity) == 0 &&
                         mynah_backend_conv1d(backend, residual, branch,
                                              (int)channels, (int)channels, (int)length,
                                              (int)kernels[branch_index], 1,
                                              weight.data, bias.data, error,
                                              error_capacity) == 0 &&
                         mynah_backend_residual_inplace(backend, current, branch,
                                                        elements, error,
                                                        error_capacity) == 0;
        }
        if (ok) ok = mynah_backend_residual_inplace(backend, dev_output, current,
                                                    elements, error, error_capacity) == 0;
    }
    if (ok) ok = mynah_backend_scale_dev(backend, dev_output, elements, 1.0f / 3.0f,
                                         error, error_capacity) == 0;
    if (mynah_backend_sync(backend, error, error_capacity) != 0) ok = 0;
    mynah_backend_dev_free(backend, branch);
    mynah_backend_dev_free(backend, current);
    mynah_backend_dev_free(backend, activated);
    mynah_backend_dev_free(backend, residual);
    return ok ? 0 : -1;
}

/* Complete NanoCodec decode with persistent Metal activations.  This is kept
 * as a separate graph path so the existing CPU/BNNS implementation remains a
 * reference oracle and a deliberate fallback for non-Metal builds. */
static int decode_codec_resident(const mynah_tts_model *model, const unsigned *codes,
                                size_t raw_length, float **samples, size_t *sample_count,
                                char *error, size_t error_capacity) {
    const mynah_backend *backend = model->backend;
    if (backend == NULL || !mynah_backend_has_dev_ops(backend) ||
        !mynah_backend_has_attention_dev(backend) ||
        (getenv("MYNAH_METAL_CPU_CODEC") != NULL &&
         strcmp(getenv("MYNAH_METAL_CPU_CODEC"), "0") != 0)) return 1;
    const size_t levels[4] = {8u, 7u, 6u, 6u};
    const size_t bases[4] = {1u, 8u, 56u, 336u};
    const size_t groups = 8u;
    const size_t latent_channels = 32u;
    if (raw_length == 0u || raw_length > SIZE_MAX / latent_channels) return -1;
    float *latent = allocate_floats(latent_channels * raw_length, error, error_capacity);
    float *dev_latent = NULL, *current = NULL, *upsampled = NULL, *audio_dev = NULL;
    float *audio = NULL;
    if (latent == NULL) return -1;
    for (size_t t = 0; t < raw_length; ++t) {
        for (size_t group = 0; group < groups; ++group) {
            const unsigned index = codes[group * raw_length + t];
            for (size_t d = 0; d < 4u; ++d) {
                const size_t digit = (index / bases[d]) % levels[d];
                latent[(group * 4u + d) * raw_length + t] =
                    ((float)digit - (float)(levels[d] / 2u)) /
                    (float)(levels[d] / 2u);
            }
        }
    }
    if (mynah_backend_dev_alloc(backend, latent_channels * raw_length, &dev_latent,
                                 error, error_capacity) != 0 ||
        mynah_backend_h2d(backend, latent, dev_latent, latent_channels * raw_length,
                          error, error_capacity) != 0 ||
        mynah_backend_dev_alloc(backend, 864u * raw_length, &current,
                                error, error_capacity) != 0) goto fail;
    free(latent);
    latent = NULL;
    mynah_tensor weight, bias;
    if (tensor(model->codec, "audio_decoder.pre_conv.conv.weight", &weight,
               error, error_capacity) != 0 ||
        tensor(model->codec, "audio_decoder.pre_conv.conv.bias", &bias,
               error, error_capacity) != 0 ||
        mynah_backend_batch_begin(backend, error, error_capacity) != 0 ||
        mynah_backend_conv1d(backend, dev_latent, current, 32, 864, (int)raw_length,
                             7, 1, weight.data, bias.data, error, error_capacity) != 0 ||
        mynah_backend_sync(backend, error, error_capacity) != 0) goto fail;
    mynah_backend_dev_free(backend, dev_latent);
    dev_latent = NULL;
    size_t current_channels = 864u, current_length = raw_length;
    const size_t rates[5] = {8u, 8u, 4u, 2u, 2u};
    char name[256], weight_name[256], bias_name[256];
    for (size_t stage = 0; stage < 5u; ++stage) {
        snprintf(name, sizeof(name),
                 "audio_decoder.activations.%zu.activation.snake_act.alpha", stage);
        if (tensor(model->codec, name, &weight, error, error_capacity) != 0 ||
            mynah_backend_batch_begin(backend, error, error_capacity) != 0 ||
            mynah_backend_snake_dev(backend, current, weight.data, current_channels,
                                    current_length, current_channels / 2u,
                                    error, error_capacity) != 0) goto fail;
        const size_t next_channels = current_channels / 2u;
        if (current_length > SIZE_MAX / rates[stage]) goto fail;
        const size_t next_length = current_length * rates[stage];
        if (next_channels > SIZE_MAX / next_length) goto fail;
        if (mynah_backend_dev_alloc(backend, next_channels * next_length, &upsampled,
                                    error, error_capacity) != 0) goto fail;
        snprintf(weight_name, sizeof(weight_name),
                 "audio_decoder.up_sample_conv_layers.%zu.conv.weight", stage);
        snprintf(bias_name, sizeof(bias_name),
                 "audio_decoder.up_sample_conv_layers.%zu.conv.bias", stage);
        if (tensor(model->codec, weight_name, &weight, error, error_capacity) != 0 ||
            tensor(model->codec, bias_name, &bias, error, error_capacity) != 0 ||
            mynah_backend_conv_transpose_dev(
                backend, current, upsampled, (int)current_channels,
                (int)next_channels, (int)current_length, (int)next_length,
                (int)(rates[stage] * 2u), (int)rates[stage], (int)next_channels,
                weight.data, bias.data, error, error_capacity) != 0) goto fail;
        float *next = NULL;
        if (mynah_backend_dev_alloc(backend, next_channels * next_length, &next,
                                    error, error_capacity) != 0 ||
            res_layer_device(model->codec, backend, stage, upsampled, next,
                             next_channels, next_length, error, error_capacity) != 0) {
            mynah_backend_dev_free(backend, next);
            goto fail;
        }
        mynah_backend_dev_free(backend, current);
        mynah_backend_dev_free(backend, upsampled);
        current = next;
        upsampled = NULL;
        current_channels = next_channels;
        current_length = next_length;
    }
    snprintf(name, sizeof(name),
             "audio_decoder.post_activation.activation.snake_act.alpha");
    snprintf(weight_name, sizeof(weight_name), "audio_decoder.post_conv.conv.weight");
    snprintf(bias_name, sizeof(bias_name), "audio_decoder.post_conv.conv.bias");
    if (tensor(model->codec, name, &weight, error, error_capacity) != 0 ||
        mynah_backend_batch_begin(backend, error, error_capacity) != 0 ||
        mynah_backend_snake_dev(backend, current, weight.data, current_channels,
                                current_length, current_channels / 2u,
                                error, error_capacity) != 0 ||
        tensor(model->codec, bias_name, &bias, error, error_capacity) != 0) goto fail;
    /* Reload the post-conv views after using `weight` for Snake's alpha. */
    if (tensor(model->codec, weight_name, &weight, error, error_capacity) != 0 ||
        tensor(model->codec, bias_name, &bias, error, error_capacity) != 0 ||
        mynah_backend_dev_alloc(backend, current_length, &audio_dev,
                                error, error_capacity) != 0 ||
        mynah_backend_conv1d(backend, current, audio_dev, (int)current_channels, 1,
                             (int)current_length, 3, 1, weight.data, bias.data,
                             error, error_capacity) != 0 ||
        mynah_backend_clip_dev(backend, audio_dev, current_length,
                               error, error_capacity) != 0 ||
        mynah_backend_sync(backend, error, error_capacity) != 0) goto fail;
    audio = allocate_floats(current_length, error, error_capacity);
    if (audio == NULL || mynah_backend_d2h(backend, audio_dev, audio, current_length,
                                           error, error_capacity) != 0) goto fail;
    mynah_backend_dev_free(backend, current);
    mynah_backend_dev_free(backend, audio_dev);
    *samples = audio;
    *sample_count = current_length;
    return 0;
fail:
    free(latent);
    free(audio);
    mynah_backend_dev_free(backend, dev_latent);
    mynah_backend_dev_free(backend, current);
    mynah_backend_dev_free(backend, upsampled);
    mynah_backend_dev_free(backend, audio_dev);
    return -1;
}

static int decode_codec(const mynah_tts_model *model, const unsigned *codes,
                        size_t raw_length, float **samples, size_t *sample_count,
                        char *error, size_t error_capacity) {
    const int resident_codec = decode_codec_resident(model, codes, raw_length,
                                                     samples, sample_count,
                                                     error, error_capacity);
    if (resident_codec != 1) return resident_codec;
    const int timing = getenv("MYNAH_TIMING") != NULL;
    const double codec_start = timing ? phase_seconds() : 0.0;
    double stage_seconds[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    codec_conv_profile conv_profile = {0};
    codec_conv_profile *profile = timing ? &conv_profile : NULL;
    codec_bnns_cache *bnns_cache = (codec_bnns_cache *)model->codec_cache;
    const size_t levels[4] = {8u, 7u, 6u, 6u};
    const size_t bases[4] = {1u, 8u, 56u, 336u};
    const size_t groups = 8u;
    const size_t latent_channels = 32u;
    float *latent = allocate_floats(latent_channels * raw_length, error, error_capacity);
    if (latent == NULL) return -1;
    for (size_t t = 0; t < raw_length; ++t) {
        for (size_t group = 0; group < groups; ++group) {
            const unsigned index = codes[group * raw_length + t];
            for (size_t d = 0; d < 4u; ++d) {
                const size_t digit = (index / bases[d]) % levels[d];
                const float denominator = (float)(levels[d] / 2u);
                latent[(group * 4u + d) * raw_length + t] =
                    ((float)digit - (float)(levels[d] / 2u)) / denominator;
            }
        }
    }
    char weight_name[256];
    char bias_name[256];
    snprintf(weight_name, sizeof(weight_name), "audio_decoder.pre_conv.conv.weight");
    snprintf(bias_name, sizeof(bias_name), "audio_decoder.pre_conv.conv.bias");
    float *current = allocate_floats(864u * raw_length, error, error_capacity);
    if (current == NULL || conv1d_causal(model->codec, model->backend, bnns_cache, weight_name, bias_name, latent,
                                         current, 32u, 864u, raw_length, 7u, 1u,
                                         NULL, 0,
                                         profile, error, error_capacity) != 0) {
        free(latent);
        free(current);
        return -1;
    }
    free(latent);
    const double preconv_end = timing ? phase_seconds() : 0.0;
    size_t current_channels = 864u;
    size_t current_length = raw_length;
    const size_t rates[5] = {8u, 8u, 4u, 2u, 2u};
    for (size_t stage = 0; stage < 5u; ++stage) {
        const double stage_start = timing ? phase_seconds() : 0.0;
        snprintf(weight_name, sizeof(weight_name), "audio_decoder.activations.%zu.activation.snake_act.alpha", stage);
        if (half_snake(model->codec, model->backend, weight_name, current, current_channels, current_length,
                       profile, error, error_capacity) != 0) {
            free(current);
            return -1;
        }
        const size_t next_channels = current_channels / 2u;
        const size_t next_length = current_length * rates[stage];
        float *upsampled = allocate_floats(next_channels * next_length, error, error_capacity);
        if (upsampled == NULL) {
            free(current);
            return -1;
        }
        snprintf(weight_name, sizeof(weight_name), "audio_decoder.up_sample_conv_layers.%zu.conv.weight", stage);
        snprintf(bias_name, sizeof(bias_name), "audio_decoder.up_sample_conv_layers.%zu.conv.bias", stage);
        if (conv_transpose_causal(model->codec, weight_name, bias_name, current, upsampled,
                                  current_channels, next_channels, current_length,
                                  rates[stage] * 2u, rates[stage], next_channels,
                                  profile, error, error_capacity) != 0) {
            free(current);
            free(upsampled);
            return -1;
        }
        free(current);
        current = allocate_floats(next_channels * next_length, error, error_capacity);
        if (current == NULL) {
            free(upsampled);
            return -1;
        }
        if (res_layer(model->codec, model->backend, bnns_cache, stage, upsampled, current, next_channels, next_length, profile,
                      error, error_capacity) != 0) {
            free(upsampled);
            free(current);
            return -1;
        }
        free(upsampled);
        current_channels = next_channels;
        current_length = next_length;
        if (timing) stage_seconds[stage] = phase_seconds() - stage_start;
    }
    snprintf(weight_name, sizeof(weight_name), "audio_decoder.post_activation.activation.snake_act.alpha");
    if (half_snake(model->codec, model->backend, weight_name, current, current_channels, current_length,
                   profile, error, error_capacity) != 0) {
        free(current);
        return -1;
    }
    float *audio = allocate_floats(current_length, error, error_capacity);
    if (audio == NULL) {
        free(current);
        return -1;
    }
    snprintf(weight_name, sizeof(weight_name), "audio_decoder.post_conv.conv.weight");
    snprintf(bias_name, sizeof(bias_name), "audio_decoder.post_conv.conv.bias");
    if (conv1d_causal(model->codec, model->backend, bnns_cache, weight_name, bias_name, current, audio,
                      current_channels, 1u, current_length, 3u, 1u,
                      NULL, 0,
                      profile, error, error_capacity) != 0) {
        free(current);
        free(audio);
        return -1;
    }
    for (size_t i = 0; i < current_length; ++i) {
        if (audio[i] > 1.0f) audio[i] = 1.0f;
        if (audio[i] < -1.0f) audio[i] = -1.0f;
    }
    free(current);
    if (timing) {
        const double codec_end = phase_seconds();
        fprintf(stderr,
                "codec detail: pre=%.3fs stages=[%.3f %.3f %.3f %.3f %.3f] "
                "post=%.3fs conv_calls=%zu pack=%.3fs gemm=%.3fs "
                "transpose_calls=%zu transpose=%.3fs snake_calls=%zu snake=%.3fs\n",
                preconv_end - codec_start, stage_seconds[0], stage_seconds[1],
                stage_seconds[2], stage_seconds[3], stage_seconds[4],
                codec_end - preconv_end - stage_seconds[0] - stage_seconds[1] -
                    stage_seconds[2] - stage_seconds[3] - stage_seconds[4],
                conv_profile.calls, conv_profile.pack_seconds, conv_profile.gemm_seconds,
                conv_profile.transpose_calls, conv_profile.transpose_seconds,
                conv_profile.snake_calls, conv_profile.snake_seconds);
        if (conv_profile.bnns_create_seconds > 0.0) {
            fprintf(stderr,
                    "codec BNNS: create=%.3fs apply=%.3fs destroy=%.3fs\n",
                    conv_profile.bnns_create_seconds,
                    conv_profile.bnns_apply_seconds,
                    conv_profile.bnns_destroy_seconds);
        }
    }
    *samples = audio;
    *sample_count = current_length;
    return 0;
}

/* --- Incremental decoder with KV cache -------------------------------------
 *
 * The offline path used to recompute the whole [context + audio] decoder
 * sequence on every autoregressive step, cross-attention included, even though
 * the text cross-attention is constant for the utterance.  With a 217-frame
 * baked speaker context and hundreds of steps that is cubic work for a tiny
 * model.  The cache keeps per-layer self-attention K/V for every past position
 * and the constant cross-attention K/V for the text memory, so each new frame
 * is a single-row forward pass.  The arithmetic is identical to the full
 * recompute, so generated codes match the non-cached path within fp tolerance.
 */
/* Pre-resolved per-layer weight names and norm pointers so the AR hot loop
 * never calls snprintf or probes the tensor hash table. */
typedef struct {
    const float *norm_self;
    const float *norm_xattn_query;
    const float *norm_pos_ff;
    const float *qkv_w;
    const float *o_self_w;
    const float *q_cross_w;
    const float *o_cross_w;
    const float *ffn_up_w;
    const float *ffn_down_w;
    char qkv[160];
    char o_self[160];
    char q_cross[160];
    char o_cross[160];
    char ffn_up[160];
    char ffn_down[160];
} decoder_layer_resolved;

typedef struct {
    size_t layers;
    size_t width;
    size_t heads;
    size_t head_width;
    size_t ffn_width;
    size_t xattn_width;
    size_t memory_length;
    size_t capacity;
    size_t length;
    float *self_k;   /* layers * capacity * width */
    float *self_v;   /* layers * capacity * width */
    float *cross_k;  /* layers * memory_length * xattn_width */
    float *cross_v;  /* layers * memory_length * xattn_width */
    const float *position; /* decoder.position_embeddings.weight */
    /* pre-resolved weights */
    decoder_layer_resolved *resolved;
    const float *norm_out;
    /* reusable scratch (sized for scratch_rows) */
    size_t scratch_rows;
    float *scratch_x;
    float *scratch_nrm;
    float *scratch_qkv;
    float *scratch_attn;
    float *scratch_proj;
    float *scratch_q_x;
    float *scratch_xctx;
    float *scratch_hidden;
    float *scratch_scores;
    float *scratch_gelu;
    float *scratch_score_matrix;
    float *scratch_head_ctx;
    /* GPU resident-step device buffers (allocated once, reused per step). */
    float *dev_nrm;
    float *dev_qkv;
    float *dev_attn;
    float *dev_proj;
    float *dev_qx;
    float *dev_xctx;
    float *dev_hidden;
    float *dev_x;
    float *dev_self_k;
    float *dev_self_v;
    float *dev_cross_k;
    float *dev_cross_v;
    int dev_allocated;
    int dev_attention_allocated;
    const mynah_backend *dev_backend; /* for dev_free */
} decoder_cache;

static void decoder_cache_free(decoder_cache *cache) {
    if (cache == NULL) return;
    free(cache->self_k);
    free(cache->self_v);
    free(cache->cross_k);
    free(cache->cross_v);
    free(cache->resolved);
    free(cache->scratch_x);
    free(cache->scratch_nrm);
    free(cache->scratch_qkv);
    free(cache->scratch_attn);
    free(cache->scratch_proj);
    free(cache->scratch_q_x);
    free(cache->scratch_xctx);
    free(cache->scratch_hidden);
    free(cache->scratch_scores);
    free(cache->scratch_gelu);
    free(cache->scratch_score_matrix);
    free(cache->scratch_head_ctx);
    if (cache->dev_allocated && cache->dev_backend != NULL) {
        mynah_backend_dev_free(cache->dev_backend, cache->dev_nrm);
        mynah_backend_dev_free(cache->dev_backend, cache->dev_qkv);
        mynah_backend_dev_free(cache->dev_backend, cache->dev_attn);
        mynah_backend_dev_free(cache->dev_backend, cache->dev_proj);
        mynah_backend_dev_free(cache->dev_backend, cache->dev_qx);
        mynah_backend_dev_free(cache->dev_backend, cache->dev_xctx);
        mynah_backend_dev_free(cache->dev_backend, cache->dev_hidden);
        mynah_backend_dev_free(cache->dev_backend, cache->dev_x);
        if (cache->dev_attention_allocated) {
            mynah_backend_dev_free(cache->dev_backend, cache->dev_self_k);
            mynah_backend_dev_free(cache->dev_backend, cache->dev_self_v);
            mynah_backend_dev_free(cache->dev_backend, cache->dev_cross_k);
            mynah_backend_dev_free(cache->dev_backend, cache->dev_cross_v);
        }
    }
    memset(cache, 0, sizeof(*cache));
}

static int decoder_gpu_attention_init(decoder_cache *cache,
                                      const mynah_backend *backend,
                                      char *error, size_t error_capacity) {
    if (cache == NULL || backend == NULL || !mynah_backend_has_attention_dev(backend))
        return -1;
    if (cache->dev_attention_allocated) return 0;
    if (cache->layers == 0 || cache->capacity == 0 || cache->width == 0 ||
        cache->memory_length == 0 || cache->xattn_width == 0 ||
        cache->layers > SIZE_MAX / cache->capacity ||
        cache->layers * cache->capacity > SIZE_MAX / cache->width ||
        cache->layers > SIZE_MAX / cache->memory_length ||
        cache->layers * cache->memory_length > SIZE_MAX / cache->xattn_width) {
        graph_error(error, error_capacity, "GPU attention cache dimensions overflow");
        return -1;
    }
    const size_t self_count = cache->layers * cache->capacity * cache->width;
    const size_t cross_count = cache->layers * cache->memory_length * cache->xattn_width;
    float *self_k = NULL, *self_v = NULL, *cross_k = NULL, *cross_v = NULL;
    if (mynah_backend_dev_alloc(backend, self_count, &self_k, error, error_capacity) != 0 ||
        mynah_backend_dev_alloc(backend, self_count, &self_v, error, error_capacity) != 0 ||
        mynah_backend_dev_alloc(backend, cross_count, &cross_k, error, error_capacity) != 0 ||
        mynah_backend_dev_alloc(backend, cross_count, &cross_v, error, error_capacity) != 0) {
        mynah_backend_dev_free(backend, self_k);
        mynah_backend_dev_free(backend, self_v);
        mynah_backend_dev_free(backend, cross_k);
        mynah_backend_dev_free(backend, cross_v);
        return -1;
    }
    for (size_t layer = 0; layer < cache->layers; ++layer) {
        const size_t self_offset = layer * cache->capacity * cache->width;
        const size_t offset = layer * cache->memory_length * cache->xattn_width;
        if ((cache->length > 0 &&
             (mynah_backend_h2d(backend, cache->self_k + self_offset,
                                self_k + self_offset,
                                cache->length * cache->width,
                                error, error_capacity) != 0 ||
              mynah_backend_h2d(backend, cache->self_v + self_offset,
                                self_v + self_offset,
                                cache->length * cache->width,
                                error, error_capacity) != 0)) ||
            mynah_backend_h2d(backend, cache->cross_k + offset, cross_k + offset,
                               cache->memory_length * cache->xattn_width,
                               error, error_capacity) != 0 ||
            mynah_backend_h2d(backend, cache->cross_v + offset, cross_v + offset,
                              cache->memory_length * cache->xattn_width,
                              error, error_capacity) != 0) {
            mynah_backend_dev_free(backend, self_k);
            mynah_backend_dev_free(backend, self_v);
            mynah_backend_dev_free(backend, cross_k);
            mynah_backend_dev_free(backend, cross_v);
            return -1;
        }
    }
    cache->dev_backend = backend;
    cache->dev_self_k = self_k;
    cache->dev_self_v = self_v;
    cache->dev_cross_k = cross_k;
    cache->dev_cross_v = cross_v;
    cache->dev_attention_allocated = 1;
    return 0;
}

/* Precompute the constant text cross-attention K/V for every decoder layer. */
static int decoder_cache_init(const mynah_tts_model *model, decoder_cache *cache,
                              const float *memory, size_t memory_length,
                              size_t capacity, char *error, size_t error_capacity) {
    memset(cache, 0, sizeof(*cache));
    const size_t width = model->info.hidden_dim;
    cache->layers = model->info.decoder_layers;
    cache->width = width;
    cache->heads = 12u;
    cache->head_width = width / cache->heads;
    cache->ffn_width = width * 4u;
    cache->memory_length = memory_length;
    cache->capacity = capacity;
    cache->length = 0;

    mynah_tensor position;
    mynah_tensor q0;
    if (tensor(model->tts, "decoder.position_embeddings.weight", &position, error, error_capacity) != 0 ||
        tensor(model->tts, "decoder.layers.0.cross_attention.q_net.weight", &q0, error, error_capacity) != 0) {
        return -1;
    }
    cache->position = position.data;
    cache->xattn_width = q0.shape[0];
    const size_t xw = cache->xattn_width;

    cache->self_k = allocate_floats(cache->layers * capacity * width, error, error_capacity);
    cache->self_v = allocate_floats(cache->layers * capacity * width, error, error_capacity);
    cache->cross_k = allocate_floats(cache->layers * memory_length * xw, error, error_capacity);
    cache->cross_v = allocate_floats(cache->layers * memory_length * xw, error, error_capacity);
    float *mem_norm = allocate_floats(memory_length * width, error, error_capacity);
    float *kv = allocate_floats(memory_length * xw * 2u, error, error_capacity);
    if (cache->self_k == NULL || cache->self_v == NULL || cache->cross_k == NULL ||
        cache->cross_v == NULL || mem_norm == NULL || kv == NULL) {
        free(mem_norm);
        free(kv);
        decoder_cache_free(cache);
        return -1;
    }
    char name[256];
    for (size_t layer = 0; layer < cache->layers; ++layer) {
        mynah_tensor norm_memory;
        mynah_tensor kv_net;
        snprintf(name, sizeof(name), "decoder.layers.%zu.norm_xattn_memory.weight", layer);
        if (tensor(model->tts, name, &norm_memory, error, error_capacity) != 0) {
            free(mem_norm);
            free(kv);
            decoder_cache_free(cache);
            return -1;
        }
        layer_norm(memory, mem_norm, memory_length, width, norm_memory.data);
        snprintf(name, sizeof(name), "decoder.layers.%zu.cross_attention.kv_net.weight", layer);
        if (tensor(model->tts, name, &kv_net, error, error_capacity) != 0 ||
            linear(model->backend, mem_norm, kv, memory_length, width, xw * 2u,
                   kv_net.data, NULL, error, error_capacity) != 0) {
            free(mem_norm);
            free(kv);
            decoder_cache_free(cache);
            return -1;
        }
        float *ck = cache->cross_k + layer * memory_length * xw;
        float *cv = cache->cross_v + layer * memory_length * xw;
        for (size_t s = 0; s < memory_length; ++s) {
            memcpy(ck + s * xw, kv + s * xw * 2u, xw * sizeof(float));
            memcpy(cv + s * xw, kv + s * xw * 2u + xw, xw * sizeof(float));
        }
    }
    free(mem_norm);
    free(kv);

    /* Pre-resolve per-layer weight names and norm pointers so decoder_run
     * never calls snprintf or probes the tensor hash table. */
    cache->resolved = (decoder_layer_resolved *)calloc(cache->layers,
                                                       sizeof(*cache->resolved));
    if (cache->resolved == NULL) {
        decoder_cache_free(cache);
        graph_error(error, error_capacity, "out of memory resolving decoder weights");
        return -1;
    }
    for (size_t layer = 0; layer < cache->layers; ++layer) {
        decoder_layer_resolved *r = &cache->resolved[layer];
        mynah_tensor t;
        snprintf(name, sizeof(name), "decoder.layers.%zu.norm_self.weight", layer);
        if (tensor(model->tts, name, &t, error, error_capacity) != 0) { decoder_cache_free(cache); return -1; }
        r->norm_self = t.data;
        snprintf(name, sizeof(name), "decoder.layers.%zu.norm_xattn_query.weight", layer);
        if (tensor(model->tts, name, &t, error, error_capacity) != 0) { decoder_cache_free(cache); return -1; }
        r->norm_xattn_query = t.data;
        snprintf(name, sizeof(name), "decoder.layers.%zu.norm_pos_ff.weight", layer);
        if (tensor(model->tts, name, &t, error, error_capacity) != 0) { decoder_cache_free(cache); return -1; }
        r->norm_pos_ff = t.data;
        snprintf(r->qkv, sizeof(r->qkv), "decoder.layers.%zu.self_attention.qkv_net.weight", layer);
        snprintf(r->o_self, sizeof(r->o_self), "decoder.layers.%zu.self_attention.o_net.weight", layer);
        snprintf(r->q_cross, sizeof(r->q_cross), "decoder.layers.%zu.cross_attention.q_net.weight", layer);
        snprintf(r->o_cross, sizeof(r->o_cross), "decoder.layers.%zu.cross_attention.o_net.weight", layer);
        snprintf(r->ffn_up, sizeof(r->ffn_up), "decoder.layers.%zu.pos_ff.proj.conv.weight", layer);
        snprintf(r->ffn_down, sizeof(r->ffn_down), "decoder.layers.%zu.pos_ff.o_net.conv.weight", layer);
        /* Pre-resolve weight data pointers for direct matmul. */
        if (tensor(model->tts, r->qkv, &t, error, error_capacity) != 0) { decoder_cache_free(cache); return -1; }
        r->qkv_w = t.data;
        if (tensor(model->tts, r->o_self, &t, error, error_capacity) != 0) { decoder_cache_free(cache); return -1; }
        r->o_self_w = t.data;
        if (tensor(model->tts, r->q_cross, &t, error, error_capacity) != 0) { decoder_cache_free(cache); return -1; }
        r->q_cross_w = t.data;
        if (tensor(model->tts, r->o_cross, &t, error, error_capacity) != 0) { decoder_cache_free(cache); return -1; }
        r->o_cross_w = t.data;
        if (tensor(model->tts, r->ffn_up, &t, error, error_capacity) != 0) { decoder_cache_free(cache); return -1; }
        r->ffn_up_w = t.data;
        if (tensor(model->tts, r->ffn_down, &t, error, error_capacity) != 0) { decoder_cache_free(cache); return -1; }
        r->ffn_down_w = t.data;
    }
    {
        mynah_tensor t;
        if (tensor(model->tts, "decoder.norm_out.weight", &t, error, error_capacity) != 0) {
            decoder_cache_free(cache);
            return -1;
        }
        cache->norm_out = t.data;
    }

    /* Pre-allocate reusable scratch sized for the largest decoder_run call
     * (the context prefill).  Single-row AR steps reuse the same buffers. */
    const size_t rows = capacity;
    const size_t scores_len = capacity > memory_length ? capacity : memory_length;
    cache->scratch_rows = rows;
    cache->scratch_x = allocate_floats(rows * width, error, error_capacity);
    cache->scratch_nrm = allocate_floats(rows * width, error, error_capacity);
    cache->scratch_qkv = allocate_floats(rows * width * 3u, error, error_capacity);
    cache->scratch_attn = allocate_floats(rows * width, error, error_capacity);
    cache->scratch_proj = allocate_floats(rows * width, error, error_capacity);
    cache->scratch_q_x = allocate_floats(rows * xw, error, error_capacity);
    cache->scratch_xctx = allocate_floats(rows * xw, error, error_capacity);
    cache->scratch_hidden = allocate_floats(rows * cache->ffn_width, error, error_capacity);
    cache->scratch_scores = allocate_floats(scores_len, error, error_capacity);
    cache->scratch_gelu = allocate_floats(rows * cache->ffn_width, error, error_capacity);
    cache->scratch_score_matrix = allocate_floats(rows * capacity, error, error_capacity);
    cache->scratch_head_ctx = allocate_floats(rows * cache->head_width, error, error_capacity);
    if (cache->scratch_x == NULL || cache->scratch_nrm == NULL ||
        cache->scratch_qkv == NULL || cache->scratch_attn == NULL ||
        cache->scratch_proj == NULL || cache->scratch_q_x == NULL ||
        cache->scratch_xctx == NULL || cache->scratch_hidden == NULL ||
        cache->scratch_scores == NULL ||
        cache->scratch_gelu == NULL || cache->scratch_score_matrix == NULL ||
        cache->scratch_head_ctx == NULL
    ) {
        decoder_cache_free(cache);
        return -1;
    }
    return 0;
}

/* Push `count` new decoder rows through the stack, appending their self K/V to
 * the cache and attending over all cached positions.  `out_last` receives the
 * final-norm output of the last new row (the one used for sampling).
 *
 * All scratch buffers and weight names are pre-resolved in decoder_cache so
 * this function performs zero allocations and zero tensor-name lookups. */
static int decoder_run(const mynah_tts_model *model, decoder_cache *cache,
                       const float *input_rows, size_t count, float *out_last,
                       float **dev_out,
                       char *error, size_t error_capacity) {
    if (dev_out != NULL) *dev_out = NULL;
    const size_t width = cache->width;
    const mynah_backend *backend = model->backend;
    const size_t heads = cache->heads;
    const size_t hw = cache->head_width;
    const size_t ffn = cache->ffn_width;
    const size_t xw = cache->xattn_width;
    const size_t start = cache->length;
    const int profile_prefill = getenv("MYNAH_TIMING") != NULL && count > 1u;
    double self_projection_seconds = 0.0;
    double self_attention_seconds = 0.0;
    double cross_projection_seconds = 0.0;
    double cross_attention_seconds = 0.0;
    double ffn_seconds = 0.0;
    if (count == 0 || start + count > cache->capacity) {
        graph_error(error, error_capacity, "decoder cache capacity exceeded");
        return -1;
    }
    if (count > cache->scratch_rows) {
        graph_error(error, error_capacity, "decoder scratch too small for count");
        return -1;
    }
    const size_t hidden_elements = count * ffn;
    float *x = cache->scratch_x;
    float *nrm = cache->scratch_nrm;
    float *qkv = cache->scratch_qkv;
    float *attn = cache->scratch_attn;
    float *proj = cache->scratch_proj;
    float *q_x = cache->scratch_q_x;
    float *xctx = cache->scratch_xctx;
    float *hidden = cache->scratch_hidden;
    float *scores = cache->scratch_scores;
    float *gelu_scratch = NULL;
#if defined(MYNAH_USE_ACCELERATE)
    if (count > 1u && getenv("MYNAH_GELU_SCALAR") == NULL) {
        gelu_scratch = cache->scratch_gelu;
    }
#endif
    /* For a multi-row call (the context prefill) the self-attention is a dense
     * batched matmul; a single-row decode step stays scalar over the KV cache
     * (a matvec where sgemm's per-call overhead would dominate). */
    float *score_matrix = NULL;
    float *head_ctx = NULL;
    int batched = 0;
    const size_t total_kv = start + count;
    if (count > 1u && total_kv <= (size_t)INT_MAX && hw <= (size_t)INT_MAX) {
        batched = 1;
        score_matrix = cache->scratch_score_matrix;
        head_ctx = cache->scratch_head_ctx;
    }
    for (size_t i = 0; i < count; ++i) {
        const float *pe = cache->position + (start + i) * width;
        for (size_t d = 0; d < width; ++d) x[i * width + d] = input_rows[i * width + d] + pe[d];
    }
    const float self_scale = 1.0f / sqrtf((float)hw);
    const float cross_scale = 1.0f / sqrtf((float)xw);
    int failed = 0;

    /* ---- GPU resident-step fast path (count==1, backend has dev ops) ----
     * All ops on GPU via matmul_d2d (FP16 cuBLAS, device-to-device).
     * Sync only at CPU attention boundaries (2/layer vs 6).
     * Metal keeps this path opt-in until its AR token/EOS parity gate passes;
     * CUDA retains the resident default. */
    const char *metal_attention_env = getenv("MYNAH_METAL_GPU_ATTENTION");
    const int metal_attention_enabled =
        strcmp(mynah_backend_name(backend), "metal") != 0 ||
        (metal_attention_env != NULL && strcmp(metal_attention_env, "1") == 0);
    const int gpu_attention_candidate = count == 1u && mynah_backend_has_dev_ops(backend) &&
        mynah_backend_has_attention_dev(backend) &&
        metal_attention_enabled &&
        (getenv("MYNAH_GPU_RESIDENT") == NULL ||
         strcmp(getenv("MYNAH_GPU_RESIDENT"), "0") != 0);
    if (gpu_attention_candidate) {
        const mynah_backend *bk = backend;
        if (!cache->dev_allocated) {
            cache->dev_backend = bk;
            cache->dev_allocated =
                mynah_backend_dev_alloc(bk, width, &cache->dev_nrm, error, error_capacity)==0
             && mynah_backend_dev_alloc(bk, width*3u, &cache->dev_qkv, error, error_capacity)==0
             && mynah_backend_dev_alloc(bk, width, &cache->dev_attn, error, error_capacity)==0
             && mynah_backend_dev_alloc(bk, width, &cache->dev_proj, error, error_capacity)==0
             && mynah_backend_dev_alloc(bk, xw, &cache->dev_qx, error, error_capacity)==0
             && mynah_backend_dev_alloc(bk, xw, &cache->dev_xctx, error, error_capacity)==0
             && mynah_backend_dev_alloc(bk, ffn, &cache->dev_hidden, error, error_capacity)==0
             && mynah_backend_dev_alloc(bk, width, &cache->dev_x, error, error_capacity)==0;
        }
        const int attention_ready = cache->dev_allocated &&
            decoder_gpu_attention_init(cache, bk, error, error_capacity) == 0;
        if (attention_ready) {
        float *dx=cache->dev_x, *dnrm=cache->dev_nrm, *dqkv=cache->dev_qkv;
        float *dattn=cache->dev_attn, *dproj=cache->dev_proj;
        float *dqx=cache->dev_qx, *dxctx=cache->dev_xctx, *dhidden=cache->dev_hidden;
        float *dself_k=cache->dev_self_k, *dself_v=cache->dev_self_v;
        float *dcross_k=cache->dev_cross_k, *dcross_v=cache->dev_cross_v;
        mynah_backend_h2d(bk, x, dx, width, error, error_capacity);
        for (size_t layer = 0; layer < cache->layers && !failed; ++layer) {
            const decoder_layer_resolved *r = &cache->resolved[layer];
            if (mynah_backend_batch_begin(bk, error, error_capacity) != 0) { failed=1; break; }
            /* ln(GPU) → QKV d2d(GPU, no sync) */
            if (mynah_backend_layer_norm_inplace(bk, dx, dnrm, r->norm_self, 1u, width, error, error_capacity)!=0) { failed=1; break; }
            if (mynah_backend_matmul_d2d(bk, dnrm, dqkv, 1u, width, width*3u, r->qkv_w, NULL, error, error_capacity)!=0) { failed=1; break; }
            if (layer == 0 && getenv("MYNAH_DUMP_GPU_QKV") != NULL) {
                if (mynah_backend_sync(bk, error, error_capacity) != 0 ||
                    mynah_backend_d2h(bk, dqkv, qkv, width * 3u, error, error_capacity) != 0) {
                    failed = 1;
                    break;
                }
                FILE *dump = fopen(getenv("MYNAH_DUMP_GPU_QKV"), "w");
                if (dump != NULL) {
                    for (size_t d = 0; d < width * 3u; ++d) fprintf(dump, "%.9g\n", (double)qkv[d]);
                    fclose(dump);
                }
                if (mynah_backend_h2d(bk, qkv, dqkv, width * 3u, error, error_capacity) != 0) {
                    failed = 1;
                    break;
                }
            }
            /* QKV → resident Metal self-attention + KV append. */
            if (mynah_backend_self_attention_dev(
                    bk, dqkv,
                    dself_k + layer * cache->capacity * width,
                    dself_v + layer * cache->capacity * width,
                    start, width, start + 1u, heads, hw, self_scale, dattn,
                    error, error_capacity) != 0) { failed=1; break; }
            if (layer == 0 && getenv("MYNAH_DUMP_GPU_SELF_ATTN") != NULL) {
                if (mynah_backend_sync(bk, error, error_capacity) != 0 ||
                    mynah_backend_d2h(bk, dattn, attn, width, error, error_capacity) != 0) {
                    failed = 1;
                    break;
                }
                FILE *dump = fopen(getenv("MYNAH_DUMP_GPU_SELF_ATTN"), "w");
                if (dump != NULL) {
                    for (size_t d = 0; d < width; ++d) fprintf(dump, "%.9g\n", (double)attn[d]);
                    fclose(dump);
                }
                if (mynah_backend_h2d(bk, attn, dattn, width, error, error_capacity) != 0) {
                    failed = 1;
                    break;
                }
            }
            /* attn → output d2d(GPU) → residual(GPU) → ln(GPU) → cross-Q d2d(GPU) */
            if (mynah_backend_batch_begin(bk, error, error_capacity) != 0) { failed=1; break; }
            if (mynah_backend_matmul_d2d(bk, dattn, dproj, 1u, width, width, r->o_self_w, NULL, error, error_capacity)!=0) { failed=1; break; }
            if (mynah_backend_residual_inplace(bk, dx, dproj, width, error, error_capacity)!=0) { failed=1; break; }
            if (mynah_backend_layer_norm_inplace(bk, dx, dnrm, r->norm_xattn_query, 1u, width, error, error_capacity)!=0) { failed=1; break; }
            if (mynah_backend_matmul_d2d(bk, dnrm, dqx, 1u, width, xw, r->q_cross_w, NULL, error, error_capacity)!=0) { failed=1; break; }
            /* Q → resident Metal cross-attention over the cached text KV. */
            if (mynah_backend_cross_attention_dev(
                    bk, dqx,
                    dcross_k + layer * cache->memory_length * xw,
                    dcross_v + layer * cache->memory_length * xw,
                    cache->memory_length, xw, 1u, xw, cross_scale,
                    dxctx, error, error_capacity) != 0) { failed=1; break; }
            /* xctx → cross-output d2d(GPU) → residual(GPU) → ln(GPU) → FFN d2d(GPU) */
            if (mynah_backend_batch_begin(bk, error, error_capacity) != 0) { failed=1; break; }
            if (mynah_backend_matmul_d2d(bk, dxctx, dproj, 1u, xw, width, r->o_cross_w, NULL, error, error_capacity)!=0) { failed=1; break; }
            if (mynah_backend_residual_inplace(bk, dx, dproj, width, error, error_capacity)!=0) { failed=1; break; }
            if (mynah_backend_layer_norm_inplace(bk, dx, dnrm, r->norm_pos_ff, 1u, width, error, error_capacity)!=0) { failed=1; break; }
            if (mynah_backend_matmul_d2d(bk, dnrm, dhidden, 1u, width, ffn, r->ffn_up_w, NULL, error, error_capacity)!=0) { failed=1; break; }
            if (getenv("MYNAH_METAL_GPU_GELU") != NULL &&
                strcmp(getenv("MYNAH_METAL_GPU_GELU"), "0") == 0) {
                if (mynah_backend_sync(bk, error, error_capacity) != 0 ||
                    mynah_backend_d2h(bk, dhidden, hidden, ffn,
                                       error, error_capacity) != 0) {
                    failed = 1;
                    break;
                }
                gelu_tanh_array(hidden, ffn, gelu_scratch);
                if (mynah_backend_h2d(bk, hidden, dhidden, ffn,
                                       error, error_capacity) != 0) {
                    failed = 1;
                    break;
                }
            } else if (mynah_backend_gelu_inplace(bk, dhidden, ffn,
                                                   error, error_capacity) != 0) {
                failed=1;
                break;
            }
            if (mynah_backend_matmul_d2d(bk, dhidden, dproj, 1u, ffn, width, r->ffn_down_w, NULL, error, error_capacity)!=0) { failed=1; break; }
            if (mynah_backend_residual_inplace(bk, dx, dproj, width, error, error_capacity)!=0) { failed=1; break; }
        }
        if (!failed && mynah_backend_layer_norm_inplace(
                bk, dx, dnrm, cache->norm_out, 1u, width,
                error, error_capacity) != 0) failed = 1;
        if (!failed && mynah_backend_sync(bk, error, error_capacity) != 0) failed=1;
        if (!failed && dev_out != NULL) *dev_out = dnrm;
        if (!failed && dev_out == NULL &&
            mynah_backend_d2h(bk, dnrm, out_last, width,
                              error, error_capacity) != 0) failed=1;
        if (!failed) {
            cache->length += count;
            return 0;
        }
        }
        /* GPU path failed — fall through to CPU path. */
    }


    for (size_t layer = 0; layer < cache->layers && !failed; ++layer) {
        const decoder_layer_resolved *r = &cache->resolved[layer];
        double operation_start = profile_prefill ? phase_seconds() : 0.0;
        /* self-attention */
        layer_norm(x, nrm, count, width, r->norm_self);
        if (mynah_backend_matmul(model->backend, nrm, qkv,
                              count, width, width * 3u,
                              r->qkv_w, NULL, error, error_capacity) != 0) { failed = 1; break; }
        if (profile_prefill) {
            self_projection_seconds += phase_seconds() - operation_start;
            operation_start = phase_seconds();
        }
        if (layer == 0 && getenv("MYNAH_DUMP_CPU_QKV") != NULL) {
            FILE *dump = fopen(getenv("MYNAH_DUMP_CPU_QKV"), "w");
            if (dump != NULL) {
                for (size_t d = 0; d < width * 3u; ++d) fprintf(dump, "%.9g\n", (double)qkv[d]);
                fclose(dump);
            }
        }
        float *kbase = cache->self_k + layer * cache->capacity * width;
        float *vbase = cache->self_v + layer * cache->capacity * width;
        for (size_t i = 0; i < count; ++i) {
            memcpy(kbase + (start + i) * width, qkv + i * width * 3u + width, width * sizeof(float));
            memcpy(vbase + (start + i) * width, qkv + i * width * 3u + width * 2u, width * sizeof(float));
        }
        if (batched) {
            for (size_t h = 0; h < heads; ++h) {
                graph_sgemm(backend, 0, 1, (int)count, (int)total_kv, (int)hw, self_scale, qkv + h * hw, (int)(width * 3u), kbase + h * hw, (int)width, 0.0f, score_matrix, (int)total_kv, error, error_capacity);
                for (size_t i = 0; i < count; ++i) {
                    const size_t valid = start + i + 1u;
                    float *row = score_matrix + i * total_kv;
                    softmax_row_inplace(row, valid);
                    for (size_t s = valid; s < total_kv; ++s) row[s] = 0.0f;
                }
                graph_sgemm(backend, 0, 0, (int)count, (int)hw, (int)total_kv, 1.0f, score_matrix, (int)total_kv, vbase + h * hw, (int)width, 0.0f, head_ctx, (int)hw, error, error_capacity);
                for (size_t i = 0; i < count; ++i) {
                    memcpy(attn + i * width + h * hw, head_ctx + i * hw, hw * sizeof(float));
                }
            }
        } else
        for (size_t i = 0; i < count; ++i) {
            const size_t abs = start + i;
            const float *qrow = qkv + i * width * 3u;
            for (size_t h = 0; h < heads; ++h) {
                const float *qh = qrow + h * hw;
                float maxv = -FLT_MAX;
                for (size_t s = 0; s <= abs; ++s) {
                    const float *k = kbase + s * width + h * hw;
                    float sc = 0.0f;
                    for (size_t d = 0; d < hw; ++d) sc += qh[d] * k[d];
                    sc *= self_scale;
                    scores[s] = sc;
                    if (sc > maxv) maxv = sc;
                }
                float denom = 0.0f;
                for (size_t s = 0; s <= abs; ++s) { scores[s] = expf(scores[s] - maxv); denom += scores[s]; }
                float *outh = attn + i * width + h * hw;
                memset(outh, 0, hw * sizeof(float));
                for (size_t s = 0; s <= abs; ++s)
                    axpy_f32(outh, vbase + s * width + h * hw, scores[s] / denom, hw);
            }
        }
        if (profile_prefill) {
            self_attention_seconds += phase_seconds() - operation_start;
            operation_start = phase_seconds();
        }
        if (layer == 0 && getenv("MYNAH_DUMP_CPU_SELF_ATTN") != NULL) {
            FILE *dump = fopen(getenv("MYNAH_DUMP_CPU_SELF_ATTN"), "w");
            if (dump != NULL) {
                for (size_t d = 0; d < width; ++d) fprintf(dump, "%.9g\n", (double)attn[d]);
                fclose(dump);
            }
        }
        if (mynah_backend_matmul(model->backend, attn, proj,
                              count, width, width,
                              r->o_self_w, NULL, error, error_capacity) != 0) { failed = 1; break; }
        for (size_t k = 0; k < count * width; ++k) x[k] += proj[k];
        /* cross-attention over cached text memory */
        layer_norm(x, nrm, count, width, r->norm_xattn_query);
        if (mynah_backend_matmul(model->backend, nrm, q_x,
                              count, width, xw,
                              r->q_cross_w, NULL, error, error_capacity) != 0) { failed = 1; break; }
        if (profile_prefill) {
            cross_projection_seconds += phase_seconds() - operation_start;
            operation_start = phase_seconds();
        }
        const float *ck = cache->cross_k + layer * cache->memory_length * xw;
        const float *cv = cache->cross_v + layer * cache->memory_length * xw;
        for (size_t i = 0; i < count; ++i) {
            const float *qh = q_x + i * xw;
            float maxv = -FLT_MAX;
            for (size_t s = 0; s < cache->memory_length; ++s) {
                const float *k = ck + s * xw;
                float sc = 0.0f;
                for (size_t d = 0; d < xw; ++d) sc += qh[d] * k[d];
                sc *= cross_scale;
                scores[s] = sc;
                if (sc > maxv) maxv = sc;
            }
            float denom = 0.0f;
            for (size_t s = 0; s < cache->memory_length; ++s) { scores[s] = expf(scores[s] - maxv); denom += scores[s]; }
            float *outh = xctx + i * xw;
            memset(outh, 0, xw * sizeof(float));
            for (size_t s = 0; s < cache->memory_length; ++s)
                axpy_f32(outh, cv + s * xw, scores[s] / denom, xw);
        }
        if (profile_prefill) {
            cross_attention_seconds += phase_seconds() - operation_start;
            operation_start = phase_seconds();
        }
        if (mynah_backend_matmul(model->backend, xctx, proj,
                              count, xw, width,
                              r->o_cross_w, NULL, error, error_capacity) != 0) { failed = 1; break; }
        for (size_t k = 0; k < count * width; ++k) x[k] += proj[k];
        /* position-wise FFN (kernel size 1) */
        layer_norm(x, nrm, count, width, r->norm_pos_ff);
        if (mynah_backend_matmul(model->backend, nrm, hidden,
                              count, width, ffn,
                              r->ffn_up_w, NULL, error, error_capacity) != 0) { failed = 1; break; }
        if (getenv("MYNAH_GELU_SCALAR") == NULL && count > 1u) {
            mynah_gelu_f32(hidden, hidden_elements);
        } else {
            mynah_gelu_f32_scalar(hidden, hidden_elements);
        }
        if (mynah_backend_matmul(model->backend, hidden, proj,
                              count, ffn, width,
                              r->ffn_down_w, NULL, error, error_capacity) != 0) { failed = 1; break; }
        for (size_t k = 0; k < count * width; ++k) x[k] += proj[k];
        if (profile_prefill) ffn_seconds += phase_seconds() - operation_start;
    }
    if (!failed) {
        layer_norm(x + (count - 1u) * width, out_last, 1u, width, cache->norm_out);
    }
    if (profile_prefill) {
        fprintf(stderr,
                "decoder prefill detail: self_proj=%.3fs self_attn=%.3fs "
                "cross_proj=%.3fs cross_attn=%.3fs ffn=%.3fs\n",
                self_projection_seconds, self_attention_seconds,
                cross_projection_seconds, cross_attention_seconds, ffn_seconds);
    }
    if (!failed) cache->length += count;
    return failed ? -1 : 0;
}

static int emit_stream_samples(mynah_tts_audio_callback callback, void *user_data,
                               const float *samples, size_t count,
                               size_t chunk_samples, char *error,
                               size_t error_capacity) {
    if (callback == NULL || count == 0) return 0;
    size_t offset = 0;
    while (offset < count) {
        const size_t remaining = count - offset;
        const size_t chunk = remaining < chunk_samples ? remaining : chunk_samples;
        if (callback(samples + offset, chunk, user_data) != 0) {
            graph_error(error, error_capacity, "audio callback aborted streaming");
            return -1;
        }
        offset += chunk;
    }
    return 0;
}

int mynah_graph_synthesize_stream(const mynah_tts_model *model,
                                  const mynah_tts_request *request,
                                  float **samples, size_t *sample_count,
                                  mynah_tts_audio_callback callback,
                                  void *user_data, size_t chunk_samples,
                                  char *error, size_t error_capacity) {
    if (samples != NULL) *samples = NULL;
    if (sample_count != NULL) *sample_count = 0;
    if (model == NULL || request == NULL ||
        ((samples == NULL || sample_count == NULL) && callback == NULL) ||
        error == NULL || error_capacity == 0 || request->text_ids == NULL ||
        request->text_length == 0 || (callback != NULL && chunk_samples == 0)) {
        graph_error(error, error_capacity, "invalid synthesis request");
        return -1;
    }
    size_t streamed_samples = 0;
    if (request->speaker >= model->info.speaker_count) {
        graph_error(error, error_capacity, "speaker index is outside the model");
        return -1;
    }
    const size_t width = model->info.hidden_dim;
    const int timing = getenv("MYNAH_TIMING") != NULL;
    const double t_start = timing ? phase_seconds() : 0.0;
    double t_prep = t_start, t_ar = t_start;
    double prep_encode_seconds = 0.0;
    double prep_cross_cache_seconds = 0.0;
    double prep_context_seconds = 0.0;
    double ar_embed_seconds = 0.0;
    double ar_decoder_seconds = 0.0;
    double ar_local_seconds = 0.0;
    float *memory = NULL;
    if (encode_text(model, request->text_ids, request->text_length, &memory,
                    error, error_capacity) != 0) return -1;
    if (getenv("MYNAH_DUMP_ENCODER") != NULL) {
        FILE *ef = fopen(getenv("MYNAH_DUMP_ENCODER"), "w");
        if (ef != NULL) {
            for (size_t t = 0; t < request->text_length; ++t)
                for (size_t d = 0; d < width; ++d)
                    fprintf(ef, "%.9g\n", (double)memory[t * width + d]);
            fclose(ef);
        }
    }
    if (timing) prep_encode_seconds = phase_seconds() - t_start;
    mynah_tensor context_tensor;
    if (tensor(model->tts, "baked_context_embedding.weight", &context_tensor,
               error, error_capacity) != 0 || context_tensor.rank != 2 ||
        context_tensor.shape[1] % width != 0 || request->speaker >= context_tensor.shape[0]) {
        free(memory);
        graph_error(error, error_capacity, "baked speaker context is invalid");
        return -1;
    }
    const size_t context_length = context_tensor.shape[1] / width;
    const size_t max_steps = request->max_steps == 0
        ? (model->info.max_decoder_steps + model->info.frame_stacking_factor - 1u) /
          model->info.frame_stacking_factor
        : request->max_steps;
    const size_t max_raw_length = (max_steps + 1u) * model->info.frame_stacking_factor;
    unsigned *codes = (unsigned *)calloc(model->info.codebook_count * max_raw_length, sizeof(*codes));
    float *out_last = allocate_floats(width, error, error_capacity);
    float *audio_row = allocate_floats(width, error, error_capacity);
    decoder_cache cache;
    memset(&cache, 0, sizeof(cache));
    if (codes == NULL || out_last == NULL || audio_row == NULL) {
        free(memory);
        free(codes);
        free(out_last);
        free(audio_row);
        return -1;
    }
    for (size_t c = 0; c < model->info.codebook_count; ++c) {
        for (size_t t = 0; t < model->info.frame_stacking_factor; ++t) {
            codes[c * max_raw_length + t] = model->info.codebook_size;
        }
    }
    const float *context = context_tensor.data + (size_t)request->speaker * context_tensor.shape[1];
    const size_t min_raw_length = model->info.min_generated_frames;
    size_t predicted_stacks = 0u;
    float temperature = request->temperature;
    if (!(temperature >= 0.0f)) temperature = model->info.default_temperature;
    unsigned topk = request->topk == 0 ? model->info.default_topk : request->topk;
    uint64_t rng_state = request->seed == 0 ? UINT64_C(0x9e3779b97f4a7c15) : request->seed;
    size_t eos_frame = SIZE_MAX;
    /* Cache the constant text cross-attention and prefill the baked speaker
     * context once, so every autoregressive step is a single-row decoder pass. */
    double prep_stage_start = timing ? phase_seconds() : 0.0;
    if (decoder_cache_init(model, &cache, memory, request->text_length,
                           context_length + max_steps + 2u, error, error_capacity) != 0) {
        decoder_cache_free(&cache);
        free(memory);
        free(codes);
        free(out_last);
        free(audio_row);
        return -1;
    }
    if (timing) {
        prep_cross_cache_seconds = phase_seconds() - prep_stage_start;
        prep_stage_start = phase_seconds();
    }
    float *decoder_dev_last = NULL;
    if (decoder_run(model, &cache, context, context_length, out_last,
                    request->use_local_transformer ? &decoder_dev_last : NULL,
                    error, error_capacity) != 0) {
        decoder_cache_free(&cache);
        free(memory);
        free(codes);
        free(out_last);
        free(audio_row);
        return -1;
    }
    if (getenv("MYNAH_DUMP_PREFILL") != NULL) {
        FILE *pf = fopen(getenv("MYNAH_DUMP_PREFILL"), "w");
        if (pf != NULL) {
            if (decoder_dev_last != NULL)
                mynah_backend_d2h(model->backend, decoder_dev_last, out_last, width,
                                  error, error_capacity);
            for (size_t d = 0; d < width; ++d)
                fprintf(pf, "%.9g\n", (double)out_last[d]);
            fclose(pf);
        }
    }
    if (timing) prep_context_seconds = phase_seconds() - prep_stage_start;
    if (timing) t_prep = phase_seconds();
    for (size_t stacked_length = 1u; stacked_length <= max_steps; ++stacked_length) {
        const size_t raw_length = stacked_length * model->info.frame_stacking_factor;
        double stage_start = timing ? phase_seconds() : 0.0;
        if (embed_audio_frame(model, codes, max_raw_length, stacked_length - 1u,
                              audio_row, error, error_capacity) != 0) break;
        if (timing) {
            const double now = phase_seconds();
            ar_embed_seconds += now - stage_start;
            stage_start = now;
        }
        decoder_dev_last = NULL;
        if (decoder_run(model, &cache, audio_row, 1u, out_last,
                        request->use_local_transformer ? &decoder_dev_last : NULL,
                        error, error_capacity) != 0) break;
        if (stacked_length == 1u && getenv("MYNAH_DUMP_HIDDEN") != NULL) {
            FILE *hf = fopen(getenv("MYNAH_DUMP_HIDDEN"), "w");
            if (hf != NULL) {
                if (decoder_dev_last != NULL)
                    mynah_backend_d2h(model->backend, decoder_dev_last, out_last, width,
                                      error, error_capacity);
                for (size_t d = 0; d < model->info.hidden_dim; ++d)
                    fprintf(hf, "%.9g\n", (double)out_last[d]);
                fclose(hf);
            }
        }
        if (timing) {
            const double now = phase_seconds();
            ar_decoder_seconds += now - stage_start;
            stage_start = now;
        }
        int saw_eos = 0;
        size_t step_eos_frame = SIZE_MAX;
        if (request->use_local_transformer) {
            if (sample_local_frame(model, out_last, decoder_dev_last,
                                   codes, raw_length, max_raw_length,
                                   predicted_stacks * model->info.frame_stacking_factor,
                                   min_raw_length, temperature, topk, &rng_state,
                                   &saw_eos, &step_eos_frame, error, error_capacity) != 0) break;
        } else {
            mynah_tensor projection;
            mynah_tensor bias;
            if (tensor(model->tts, "final_proj.weight", &projection, error, error_capacity) != 0 ||
                tensor(model->tts, "final_proj.bias", &bias, error, error_capacity) != 0) break;
            const size_t streams = model->info.codebook_count * model->info.frame_stacking_factor;
            const int forbid_eos = predicted_stacks * model->info.frame_stacking_factor < min_raw_length;
            const unsigned eos_id = model->info.audio_eos_id;
            const size_t vocab = model->info.audio_vocab_size;
            const float *last = out_last;
            for (size_t stream = 0; stream < streams; ++stream) {
                unsigned value = 0;
                float best = -FLT_MAX;
                for (size_t candidate = 0; candidate < vocab; ++candidate) {
                    /* Same forbidden-token rule as the local path: real codes
                     * always, AUDIO_EOS unless too early, nothing else. */
                    const int is_code = candidate < model->info.codebook_size;
                    const int is_eos = candidate == eos_id;
                    if (!is_code && !(is_eos && !forbid_eos)) continue;
                    float score = bias.data[stream * model->info.audio_vocab_size + candidate];
                    const float *row = projection.data + (stream * model->info.audio_vocab_size + candidate) * width;
                    for (size_t d = 0; d < width; ++d) score += row[d] * last[d];
                    if (score > best) {
                        best = score;
                        value = (unsigned)candidate;
                    }
                }
                if (value == eos_id) {
                    saw_eos = 1;
                    const size_t frame = stream / model->info.codebook_count;
                    if (frame < step_eos_frame) step_eos_frame = frame;
                }
                const unsigned store = value < model->info.codebook_size ? value : 0u;
                const size_t fs = stream / model->info.codebook_count;
                const size_t codebook = stream % model->info.codebook_count;
                codes[codebook * max_raw_length + raw_length + fs] = store;
            }
        }
        if (timing) ar_local_seconds += phase_seconds() - stage_start;
        ++predicted_stacks;
        if (callback != NULL) {
            size_t stream_raw = predicted_stacks * model->info.frame_stacking_factor;
            if (step_eos_frame != SIZE_MAX) {
                stream_raw = (predicted_stacks - 1u) * model->info.frame_stacking_factor +
                             step_eos_frame;
            }
            if (stream_raw > 0) {
                unsigned *stream_codes = (unsigned *)calloc(
                    model->info.codebook_count * stream_raw, sizeof(*stream_codes));
                if (stream_codes == NULL) {
                    graph_error(error, error_capacity, "out of memory preparing streamed codes");
                    break;
                }
                for (size_t c = 0; c < model->info.codebook_count; ++c) {
                    memcpy(stream_codes + c * stream_raw,
                           codes + c * max_raw_length + model->info.frame_stacking_factor,
                           stream_raw * sizeof(*stream_codes));
                }
                float *stream_audio = NULL;
                size_t stream_count = 0;
                if (decode_codec(model, stream_codes, stream_raw, &stream_audio,
                                 &stream_count, error, error_capacity) != 0) {
                    free(stream_codes);
                    break;
                }
                if (stream_count < streamed_samples ||
                    emit_stream_samples(callback, user_data,
                                        stream_audio + streamed_samples,
                                        stream_count - streamed_samples, chunk_samples,
                                        error, error_capacity) != 0) {
                    free(stream_audio);
                    free(stream_codes);
                    if (stream_count < streamed_samples) {
                        graph_error(error, error_capacity, "streamed codec output regressed");
                    }
                    break;
                }
                streamed_samples = stream_count;
                free(stream_audio);
                free(stream_codes);
            }
        }
 if (step_eos_frame != SIZE_MAX) {
            eos_frame = step_eos_frame;
        }
        if (saw_eos && predicted_stacks >= 4u) {
            break;
        }
    }
    if (timing) t_ar = phase_seconds();
    const size_t generated_stacks = predicted_stacks;
    size_t generated_raw = generated_stacks * model->info.frame_stacking_factor;
    if (eos_frame != SIZE_MAX && generated_stacks > 0) {
        generated_raw = (generated_stacks - 1u) * model->info.frame_stacking_factor + eos_frame;
    }
    if (getenv("MYNAH_DUMP_CODES") != NULL && generated_stacks > 0) {
        FILE *dump = fopen(getenv("MYNAH_DUMP_CODES"), "w");
        if (dump != NULL) {
            const size_t cb = model->info.codebook_count;
            const size_t fs = model->info.frame_stacking_factor;
            fprintf(dump, "[");
            for (size_t step = 0; step < generated_stacks; ++step) {
                if (step > 0) fprintf(dump, ",");
                fprintf(dump, "[[");
                for (size_t c = 0; c < cb; ++c) {
                    if (c > 0) fprintf(dump, "],[");
                    for (size_t f = 0; f < fs; ++f) {
                        if (f > 0) fprintf(dump, ",");
                        fprintf(dump, "%u", codes[c * max_raw_length + (step + 1u) * fs + f]);
                    }
                }
                fprintf(dump, "]]");
            }
            fprintf(dump, "]\n");
            fclose(dump);
        }
    }
    int result = 0;
    if (error[0] != '\0') {
        result = -1;
    } else if (generated_raw == 0) {
        if (error[0] == '\0') graph_error(error, error_capacity, "decoder generated no audio frames");
        result = -1;
    } else if (callback != NULL) {
        if (streamed_samples == 0) {
            graph_error(error, error_capacity, "stream produced no audio frames");
            result = -1;
        }
    } else {
        unsigned *predicted_codes = (unsigned *)calloc(
            model->info.codebook_count * generated_raw, sizeof(*predicted_codes));
        if (predicted_codes == NULL) {
            graph_error(error, error_capacity, "out of memory copying generated codes");
            result = -1;
        } else {
            for (size_t c = 0; c < model->info.codebook_count; ++c) {
                memcpy(predicted_codes + c * generated_raw,
                       codes + c * max_raw_length + model->info.frame_stacking_factor,
                       generated_raw * sizeof(*predicted_codes));
            }
            result = decode_codec(model, predicted_codes, generated_raw, samples, sample_count,
                                  error, error_capacity);
            free(predicted_codes);
        }
    }
    if (timing) {
        const double t_codec = phase_seconds();
        fprintf(stderr,
                "phase: prep=%.3fs [encode=%.3fs cross_cache=%.3fs context=%.3fs setup=%.3fs] "
                "ar=%.3fs [embed=%.3fs decoder=%.3fs local=%.3fs] "
                "codec=%.3fs (stacks=%zu)\n",
                t_prep - t_start, prep_encode_seconds, prep_cross_cache_seconds,
                prep_context_seconds,
                t_prep - t_start - prep_encode_seconds - prep_cross_cache_seconds -
                    prep_context_seconds,
                t_ar - t_prep, ar_embed_seconds, ar_decoder_seconds,
                ar_local_seconds, t_codec - t_ar, generated_stacks);
    }
    decoder_cache_free(&cache);
    free(memory);
    free(codes);
    free(out_last);
    free(audio_row);
    if (result == 0) error[0] = '\0';
    return result;
}

int mynah_tts_synthesize(const mynah_tts_model *model,
                         const mynah_tts_request *request,
                         float **samples, size_t *sample_count,
                         char *error, size_t error_capacity) {
    return mynah_graph_synthesize_stream(model, request, samples, sample_count,
                                          NULL, NULL, 0, error, error_capacity);
}

void mynah_tts_free_samples(float *samples) {
    free(samples);
}
