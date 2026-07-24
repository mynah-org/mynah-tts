#include "mynah_tts_internal.h"

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
    for (size_t t = 0; t < length; ++t) {
        const float *row = input + t * width;
        float *out = output + t * width;
        float mean = 0.0f;
        for (size_t d = 0; d < width; ++d) mean += row[d];
        mean /= (float)width;
        float variance = 0.0f;
        for (size_t d = 0; d < width; ++d) {
            const float delta = row[d] - mean;
            variance += delta * delta;
        }
        variance /= (float)width;
        const float scale = 1.0f / sqrtf(variance + 1.0e-5f);
        for (size_t d = 0; d < width; ++d) out[d] = (row[d] - mean) * scale * weight[d];
    }
}

static float gelu_tanh(float x) {
    const float cubic = x * x * x;
    const float inner = 0.7978845608028654f * (x + 0.044715f * cubic);
    return 0.5f * x * (1.0f + tanhf(inner));
}

#if defined(MYNAH_USE_ACCELERATE) || defined(MYNAH_USE_OPENBLAS)
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
#endif

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
        for (size_t i = 0; i < length * ffn_width; ++i) hidden[i] = gelu_tanh(hidden[i]);
        const int result = linear(backend, hidden, output, length, ffn_width, width,
                                  out_net.data, NULL, error, error_capacity);
        free(hidden);
        return result;
    }
#if defined(MYNAH_USE_ACCELERATE) || defined(MYNAH_USE_OPENBLAS)
    if (length <= (size_t)INT_MAX && width <= (size_t)INT_MAX && ffn_width <= (size_t)INT_MAX) {
        /* A causal conv-FFN is a sum of `kernel` shifted matmuls (weight tap k),
         * the same trick as conv1d_causal but with time-major rows.  This is the
         * encoder's dominant cost, so accumulate the taps with sgemm instead of
         * the scalar quadruple loop. */
        float *wk = (float *)malloc(ffn_width * width * sizeof(float));
        if (wk == NULL) {
            free(hidden);
            graph_error(error, error_capacity, "out of memory in causal conv-ffn");
            return -1;
        }
        memset(hidden, 0, length * ffn_width * sizeof(float));
        for (size_t k = 0; k < kernel; ++k) {
            const size_t shift = kernel - 1u - k;
            if (shift >= length) continue;
            for (size_t o = 0; o < ffn_width; ++o) {
                for (size_t i = 0; i < width; ++i) wk[o * width + i] = proj.data[(o * width + i) * kernel + k];
            }
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        (int)(length - shift), (int)ffn_width, (int)width,
                        1.0f, input, (int)width, wk, (int)width,
                        1.0f, hidden + shift * ffn_width, (int)ffn_width);
        }
        for (size_t i = 0; i < length * ffn_width; ++i) hidden[i] = gelu_tanh(hidden[i]);
        memset(output, 0, length * width * sizeof(float));
        for (size_t k = 0; k < kernel; ++k) {
            const size_t shift = kernel - 1u - k;
            if (shift >= length) continue;
            for (size_t o = 0; o < width; ++o) {
                for (size_t i = 0; i < ffn_width; ++i) wk[o * ffn_width + i] = out_net.data[(o * ffn_width + i) * kernel + k];
            }
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        (int)(length - shift), (int)width, (int)ffn_width,
                        1.0f, hidden, (int)ffn_width, wk, (int)ffn_width,
                        1.0f, output + shift * width, (int)width);
        }
        free(wk);
        free(hidden);
        return 0;
    }
#endif
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
#if defined(MYNAH_USE_ACCELERATE) || defined(MYNAH_USE_OPENBLAS)
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
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        (int)length, (int)length, (int)head_width,
                        1.0f / sqrtf((float)head_width), queries, (int)head_width,
                        keys, (int)head_width, 0.0f, score_matrix, (int)length);
            for (size_t t = 0; t < length; ++t) {
                for (size_t s = t + 1u; s < length; ++s) score_matrix[t * length + s] = 0.0f;
                softmax_row_inplace(score_matrix + t * length, t + 1u);
                for (size_t s = t + 1u; s < length; ++s) score_matrix[t * length + s] = 0.0f;
            }
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        (int)length, (int)head_width, (int)length,
                        1.0f, score_matrix, (int)length,
                        values, (int)head_width, 0.0f, head_context, (int)head_width);
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
#endif
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
#if defined(MYNAH_USE_ACCELERATE) || defined(MYNAH_USE_OPENBLAS)
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
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    (int)length, (int)memory_length, (int)attention_width,
                    1.0f / sqrtf((float)attention_width), q, (int)attention_width,
                    keys, (int)attention_width, 0.0f, score_matrix, (int)memory_length);
        for (size_t t = 0; t < length; ++t) {
            softmax_row_inplace(score_matrix + t * memory_length, memory_length);
        }
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    (int)length, (int)attention_width, (int)memory_length,
                    1.0f, score_matrix, (int)memory_length,
                    values, (int)attention_width, 0.0f, context, (int)attention_width);
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
#endif
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
    char name[128];
    memset(row, 0, width * sizeof(float));
    for (size_t fs = 0; fs < stacking; ++fs) {
        for (size_t codebook = 0; codebook < codebooks; ++codebook) {
            const unsigned code = codes[codebook * code_stride + frame * stacking + fs];
            snprintf(name, sizeof(name), "audio_embeddings.%zu.weight", fs * codebooks + codebook);
            mynah_tensor table;
            if (tensor(model->tts, name, &table, error, error_capacity) != 0) return -1;
            if (code >= table.shape[0]) {
                graph_error(error, error_capacity, "audio token id is outside vocabulary");
                return -1;
            }
            for (size_t d = 0; d < width; ++d) row[d] += table.data[(size_t)code * width + d];
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
} local_cache;

static void local_cache_free(local_cache *cache) {
    if (cache == NULL) return;
    free(cache->k);
    free(cache->v);
    cache->k = NULL;
    cache->v = NULL;
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
    return 0;
}

/* Append one position to the local transformer and return its output row. */
static int local_step(const mynah_tts_model *model, local_cache *cache,
                      const float *input_row, float *out_row,
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
    float *x = allocate_floats(width, error, error_capacity);
    float *nrm = allocate_floats(width, error, error_capacity);
    float *qkv = allocate_floats(width * 3u, error, error_capacity);
    float *attn = allocate_floats(width, error, error_capacity);
    float *proj = allocate_floats(width, error, error_capacity);
    float *hidden = allocate_floats(ffn, error, error_capacity);
    float *scores = allocate_floats(cache->capacity, error, error_capacity);
    if (x == NULL || nrm == NULL || qkv == NULL || attn == NULL || proj == NULL ||
        hidden == NULL || scores == NULL) {
        free(x); free(nrm); free(qkv); free(attn); free(proj); free(hidden); free(scores);
        return -1;
    }
    for (size_t d = 0; d < width; ++d) x[d] = input_row[d] + cache->position[p * width + d];
    const float scale = 1.0f / sqrtf((float)hw);
    char name[256];
    int failed = 0;
    for (size_t layer = 0; layer < cache->layers && !failed; ++layer) {
        mynah_tensor t;
        snprintf(name, sizeof(name), "local_transformer.layers.%zu.norm_self.weight", layer);
        if (tensor(model->tts, name, &t, error, error_capacity) != 0) { failed = 1; break; }
        layer_norm(x, nrm, 1u, width, t.data);
        snprintf(name, sizeof(name), "local_transformer.layers.%zu.self_attention.qkv_net.weight", layer);
        if (mynah_qmat_linear(model->qcache, model->tts, model->backend, name, nrm, qkv,
                              1u, width, width * 3u, NULL, error, error_capacity) != 0) { failed = 1; break; }
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
            for (size_t d = 0; d < hw; ++d) {
                float acc = 0.0f;
                for (size_t s = 0; s <= p; ++s) acc += (scores[s] / denom) * vb[s * width + h * hw + d];
                outh[d] = acc;
            }
        }
        snprintf(name, sizeof(name), "local_transformer.layers.%zu.self_attention.o_net.weight", layer);
        if (mynah_qmat_linear(model->qcache, model->tts, model->backend, name, attn, proj,
                              1u, width, width, NULL, error, error_capacity) != 0) { failed = 1; break; }
        for (size_t d = 0; d < width; ++d) x[d] += proj[d];
        snprintf(name, sizeof(name), "local_transformer.layers.%zu.norm_pos_ff.weight", layer);
        if (tensor(model->tts, name, &t, error, error_capacity) != 0) { failed = 1; break; }
        layer_norm(x, nrm, 1u, width, t.data);
        snprintf(name, sizeof(name), "local_transformer.layers.%zu.pos_ff.proj.conv.weight", layer);
        if (mynah_qmat_linear(model->qcache, model->tts, model->backend, name, nrm, hidden,
                              1u, width, ffn, NULL, error, error_capacity) != 0) { failed = 1; break; }
        for (size_t i = 0; i < ffn; ++i) hidden[i] = gelu_tanh(hidden[i]);
        snprintf(name, sizeof(name), "local_transformer.layers.%zu.pos_ff.o_net.conv.weight", layer);
        if (mynah_qmat_linear(model->qcache, model->tts, model->backend, name, hidden, proj,
                              1u, ffn, width, NULL, error, error_capacity) != 0) { failed = 1; break; }
        for (size_t d = 0; d < width; ++d) x[d] += proj[d];
    }
    if (!failed) memcpy(out_row, x, width * sizeof(float));
    free(x); free(nrm); free(qkv); free(attn); free(proj); free(hidden); free(scores);
    if (!failed) cache->length += 1u;
    return failed ? -1 : 0;
}

/* The local AR helper emits one stacked frame (two raw codec frames). */
static int sample_local_frame(const mynah_tts_model *model, const float *decoder_last,
                              unsigned *codes, size_t raw_offset, size_t code_stride,
                              size_t generated_raw_length, size_t min_raw_length,
                              float temperature, unsigned topk,
                              uint64_t *rng_state, int *saw_eos,
                              size_t *eos_frame, char *error, size_t error_capacity) {
    const size_t width = model->info.hidden_dim;
    const size_t stream_count = model->info.codebook_count * model->info.frame_stacking_factor;
    local_cache lc;
    memset(&lc, 0, sizeof(lc));
    float *row_in = allocate_floats(width, error, error_capacity);
    float *row_out = allocate_floats(width, error, error_capacity);
    if (row_in == NULL || row_out == NULL ||
        local_cache_init(model, &lc, stream_count + 1u, error, error_capacity) != 0) {
        free(row_in);
        free(row_out);
        local_cache_free(&lc);
        return -1;
    }
    memcpy(row_in, decoder_last, width * sizeof(float));
    *saw_eos = 0;
    *eos_frame = SIZE_MAX;
    char name[160];
    for (size_t stream = 0; stream < stream_count; ++stream) {
        if (local_step(model, &lc, row_in, row_out, error, error_capacity) != 0) {
            free(row_in);
            free(row_out);
            local_cache_free(&lc);
            return -1;
        }
        snprintf(name, sizeof(name), "local_transformer_out_projections.%zu.weight", stream);
        mynah_tensor weight;
        if (tensor(model->tts, name, &weight, error, error_capacity) != 0) {
            free(row_in);
            free(row_out);
            local_cache_free(&lc);
            return -1;
        }
        snprintf(name, sizeof(name), "local_transformer_out_projections.%zu.bias", stream);
        mynah_tensor bias;
        if (tensor(model->tts, name, &bias, error, error_capacity) != 0) {
            free(row_in);
            free(row_out);
            local_cache_free(&lc);
            return -1;
        }
        const int forbid_eos = generated_raw_length < min_raw_length;
        const unsigned eos_id = model->info.audio_eos_id;
        const size_t vocab = model->info.audio_vocab_size;
        unsigned argmax = 0;
        float best = -FLT_MAX;
        float *logits = allocate_floats(vocab, error, error_capacity);
        if (logits == NULL) {
            free(row_in);
            free(row_out);
            local_cache_free(&lc);
            return -1;
        }
        /* Forbid every special token except AUDIO_EOS, and forbid EOS too while
         * we are still below min_generated_frames (NeMo clear_forbidden_logits). */
        for (size_t candidate = 0; candidate < vocab; ++candidate) {
            const int is_code = candidate < model->info.codebook_size;
            const int is_eos = candidate == eos_id;
            if (!is_code && !(is_eos && !forbid_eos)) {
                logits[candidate] = -FLT_MAX;
                continue;
            }
            float score = bias.data[candidate];
            const float *row = weight.data + candidate * width;
            for (size_t d = 0; d < width; ++d) score += row[d] * row_out[d];
            logits[candidate] = score;
            if (score > best) {
                best = score;
                argmax = (unsigned)candidate;
            }
        }
        unsigned value = argmax;
        /* argmax_or_multinomial: EOS fires if either the greedy or the sampled
         * token is AUDIO_EOS in any codebook of this frame. */
        int stream_eos = (argmax == eos_id);
        if (temperature > 0.0f && topk > 1u && rng_state != NULL) {
            size_t top_count = topk < vocab ? topk : vocab;
            size_t *top_indices = (size_t *)malloc(top_count * sizeof(*top_indices));
            float *top_logits = (float *)malloc(top_count * sizeof(*top_logits));
            if (top_indices == NULL || top_logits == NULL) {
                free(top_indices);
                free(top_logits);
                free(logits);
                free(row_in);
                free(row_out);
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
            free(top_indices);
            free(top_logits);
        }
        free(logits);
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
        snprintf(name, sizeof(name), "audio_embeddings.%zu.weight", stream);
        mynah_tensor audio_table;
        if (tensor(model->tts, name, &audio_table, error, error_capacity) != 0) {
            free(row_in);
            free(row_out);
            local_cache_free(&lc);
            return -1;
        }
        /* The embedding of this stream's token becomes the input row for the
         * next local-transformer position. */
        memcpy(row_in, audio_table.data + (size_t)emit * width, width * sizeof(float));
    }
    free(row_in);
    free(row_out);
    local_cache_free(&lc);
    return 0;
}

static int conv1d_causal(const mynah_safetensors *file, const char *weight_name,
                         const char *bias_name, const float *input, float *output,
                         size_t in_channels, size_t out_channels, size_t length,
                         size_t kernel, size_t dilation, char *error, size_t error_capacity) {
    mynah_tensor weight;
    mynah_tensor bias;
    if (tensor(file, weight_name, &weight, error, error_capacity) != 0 ||
        tensor(file, bias_name, &bias, error, error_capacity) != 0) return -1;
    /* Seed the output with the per-channel bias; each kernel tap then adds its
     * contribution.  A causal conv1d is a sum of `kernel` shifted matmuls
     * (weight tap k) [out x in] . input[:, 0:length-shift], so on BLAS builds we
     * accumulate them with sgemm instead of the scalar quadruple loop, which is
     * the codec decoder's dominant cost. */
    for (size_t o = 0; o < out_channels; ++o) {
        for (size_t t = 0; t < length; ++t) output[o * length + t] = bias.data[o];
    }
#if defined(MYNAH_USE_ACCELERATE) || defined(MYNAH_USE_OPENBLAS)
    if (in_channels <= (size_t)INT_MAX && out_channels <= (size_t)INT_MAX &&
        length <= (size_t)INT_MAX) {
        float *wk = (float *)malloc(out_channels * in_channels * sizeof(float));
        if (wk == NULL) {
            graph_error(error, error_capacity, "out of memory in causal conv1d");
            return -1;
        }
        for (size_t k = 0; k < kernel; ++k) {
            const size_t shift = (kernel - 1u - k) * dilation;
            if (shift >= length) continue;
            for (size_t o = 0; o < out_channels; ++o) {
                for (size_t i = 0; i < in_channels; ++i) {
                    wk[o * in_channels + i] = weight.data[(o * in_channels + i) * kernel + k];
                }
            }
            const size_t n = length - shift;
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        (int)out_channels, (int)n, (int)in_channels,
                        1.0f, wk, (int)in_channels, input, (int)length,
                        1.0f, output + shift, (int)length);
        }
        free(wk);
        return 0;
    }
#endif
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

static int conv_transpose_causal(const mynah_safetensors *file, const char *weight_name,
                                 const char *bias_name, const float *input, float *output,
                                 size_t in_channels, size_t out_channels, size_t length,
                                 size_t kernel, size_t stride, size_t groups,
                                 char *error, size_t error_capacity) {
    mynah_tensor weight;
    mynah_tensor bias;
    if (tensor(file, weight_name, &weight, error, error_capacity) != 0 ||
        tensor(file, bias_name, &bias, error, error_capacity) != 0) return -1;
    const size_t full_length = (length - 1u) * stride + kernel;
    const size_t trim = kernel - stride;
    const size_t output_length = full_length - trim;
    memset(output, 0, out_channels * output_length * sizeof(float));
    for (size_t o = 0; o < out_channels; ++o) {
        for (size_t t = 0; t < output_length; ++t) output[o * output_length + t] = bias.data[o];
    }
    const size_t in_per_group = in_channels / groups;
    const size_t out_per_group = out_channels / groups;
    for (size_t i = 0; i < in_channels; ++i) {
        const size_t group = i / in_per_group;
        for (size_t o_local = 0; o_local < out_per_group; ++o_local) {
            const size_t o = group * out_per_group + o_local;
            for (size_t t = 0; t < length; ++t) {
                for (size_t k = 0; k < kernel; ++k) {
                    const long position = (long)(t * stride + k);
                    if (position >= 0 && (size_t)position < output_length) {
                        const size_t weight_index = (i * out_per_group + o_local) * kernel + k;
                        output[o * output_length + (size_t)position] +=
                            input[i * length + t] * weight.data[weight_index];
                    }
                }
            }
        }
    }
    return 0;
}

static int half_snake(const mynah_safetensors *file, const char *alpha_name,
                      float *signal, size_t channels, size_t length,
                      char *error, size_t error_capacity) {
    mynah_tensor alpha;
    if (tensor(file, alpha_name, &alpha, error, error_capacity) != 0) return -1;
    const size_t snake_channels = channels / 2u;
    for (size_t c = 0; c < snake_channels; ++c) {
        const float a = alpha.data[c];
        for (size_t t = 0; t < length; ++t) {
            const size_t index = c * length + t;
            const float value = signal[index];
            signal[index] = value + sinf(a * value) * sinf(a * value) / (a + 1.0e-9f);
        }
    }
    for (size_t c = snake_channels; c < channels; ++c) {
        for (size_t t = 0; t < length; ++t) {
            const size_t index = c * length + t;
            if (signal[index] < 0.0f) signal[index] *= 0.01f;
        }
    }
    return 0;
}

static int res_layer(const mynah_safetensors *file, size_t stage, const float *input,
                     float *output, size_t channels, size_t length,
                     char *error, size_t error_capacity) {
    const size_t kernels[3] = {3u, 7u, 11u};
    const size_t dilations[3] = {1u, 3u, 5u};
    float *branch = allocate_floats(channels * length, error, error_capacity);
    float *current = allocate_floats(channels * length, error, error_capacity);
    float *activated = allocate_floats(channels * length, error, error_capacity);
    float *residual = allocate_floats(channels * length, error, error_capacity);
    if (branch == NULL || current == NULL || activated == NULL || residual == NULL) {
        free(branch);
        free(current);
        free(activated);
        free(residual);
        return -1;
    }
    memset(output, 0, channels * length * sizeof(float));
    char name[256];
    for (size_t branch_index = 0; branch_index < 3u; ++branch_index) {
        memcpy(current, input, channels * length * sizeof(float));
        for (size_t dilation_index = 0; dilation_index < 3u; ++dilation_index) {
            snprintf(name, sizeof(name),
                     "audio_decoder.res_layers.%zu.res_blocks.%zu.res_blocks.%zu.input_activation.activation.snake_act.alpha",
                     stage, branch_index, dilation_index);
            memcpy(activated, current, channels * length * sizeof(float));
            if (half_snake(file, name, activated, channels, length, error, error_capacity) != 0) break;
            char weight_name[256];
            char bias_name[256];
            snprintf(weight_name, sizeof(weight_name),
                     "audio_decoder.res_layers.%zu.res_blocks.%zu.res_blocks.%zu.input_conv.conv.weight",
                     stage, branch_index, dilation_index);
            snprintf(bias_name, sizeof(bias_name),
                     "audio_decoder.res_layers.%zu.res_blocks.%zu.res_blocks.%zu.input_conv.conv.bias",
                     stage, branch_index, dilation_index);
            if (conv1d_causal(file, weight_name, bias_name, activated, residual,
                              channels, channels, length, kernels[branch_index],
                              dilations[dilation_index], error, error_capacity) != 0) break;
            snprintf(name, sizeof(name),
                     "audio_decoder.res_layers.%zu.res_blocks.%zu.res_blocks.%zu.skip_activation.activation.snake_act.alpha",
                     stage, branch_index, dilation_index);
            if (half_snake(file, name, residual, channels, length, error, error_capacity) != 0) break;
            snprintf(weight_name, sizeof(weight_name),
                     "audio_decoder.res_layers.%zu.res_blocks.%zu.res_blocks.%zu.skip_conv.conv.weight",
                     stage, branch_index, dilation_index);
            snprintf(bias_name, sizeof(bias_name),
                     "audio_decoder.res_layers.%zu.res_blocks.%zu.res_blocks.%zu.skip_conv.conv.bias",
                     stage, branch_index, dilation_index);
            if (conv1d_causal(file, weight_name, bias_name, residual, branch,
                              channels, channels, length, kernels[branch_index], 1u,
                              error, error_capacity) != 0) break;
            for (size_t i = 0; i < channels * length; ++i) current[i] += branch[i];
        }
        for (size_t i = 0; i < channels * length; ++i) output[i] += current[i];
        if (error != NULL && error[0] != '\0') break;
    }
    for (size_t i = 0; i < channels * length; ++i) output[i] /= 3.0f;
    free(branch);
    free(current);
    free(activated);
    free(residual);
    return error == NULL || error[0] == '\0' ? 0 : -1;
}

static int decode_codec(const mynah_tts_model *model, const unsigned *codes,
                        size_t raw_length, float **samples, size_t *sample_count,
                        char *error, size_t error_capacity) {
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
    if (current == NULL || conv1d_causal(model->codec, weight_name, bias_name, latent,
                                         current, 32u, 864u, raw_length, 7u, 1u,
                                         error, error_capacity) != 0) {
        free(latent);
        free(current);
        return -1;
    }
    free(latent);
    size_t current_channels = 864u;
    size_t current_length = raw_length;
    const size_t rates[5] = {8u, 8u, 4u, 2u, 2u};
    for (size_t stage = 0; stage < 5u; ++stage) {
        snprintf(weight_name, sizeof(weight_name), "audio_decoder.activations.%zu.activation.snake_act.alpha", stage);
        if (half_snake(model->codec, weight_name, current, current_channels, current_length,
                       error, error_capacity) != 0) {
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
                                  error, error_capacity) != 0) {
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
        if (res_layer(model->codec, stage, upsampled, current, next_channels, next_length,
                      error, error_capacity) != 0) {
            free(upsampled);
            free(current);
            return -1;
        }
        free(upsampled);
        current_channels = next_channels;
        current_length = next_length;
    }
    snprintf(weight_name, sizeof(weight_name), "audio_decoder.post_activation.activation.snake_act.alpha");
    if (half_snake(model->codec, weight_name, current, current_channels, current_length,
                   error, error_capacity) != 0) {
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
    if (conv1d_causal(model->codec, weight_name, bias_name, current, audio,
                      current_channels, 1u, current_length, 3u, 1u,
                      error, error_capacity) != 0) {
        free(current);
        free(audio);
        return -1;
    }
    for (size_t i = 0; i < current_length; ++i) {
        if (audio[i] > 1.0f) audio[i] = 1.0f;
        if (audio[i] < -1.0f) audio[i] = -1.0f;
    }
    free(current);
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
} decoder_cache;

static void decoder_cache_free(decoder_cache *cache) {
    if (cache == NULL) return;
    free(cache->self_k);
    free(cache->self_v);
    free(cache->cross_k);
    free(cache->cross_v);
    cache->self_k = NULL;
    cache->self_v = NULL;
    cache->cross_k = NULL;
    cache->cross_v = NULL;
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
    return 0;
}

/* Push `count` new decoder rows through the stack, appending their self K/V to
 * the cache and attending over all cached positions.  `out_last` receives the
 * final-norm output of the last new row (the one used for sampling). */
static int decoder_run(const mynah_tts_model *model, decoder_cache *cache,
                       const float *input_rows, size_t count, float *out_last,
                       char *error, size_t error_capacity) {
    const size_t width = cache->width;
    const size_t heads = cache->heads;
    const size_t hw = cache->head_width;
    const size_t ffn = cache->ffn_width;
    const size_t xw = cache->xattn_width;
    const size_t start = cache->length;
    if (count == 0 || start + count > cache->capacity) {
        graph_error(error, error_capacity, "decoder cache capacity exceeded");
        return -1;
    }
    const size_t scores_len = cache->capacity > cache->memory_length
        ? cache->capacity : cache->memory_length;
    float *x = allocate_floats(count * width, error, error_capacity);
    float *nrm = allocate_floats(count * width, error, error_capacity);
    float *qkv = allocate_floats(count * width * 3u, error, error_capacity);
    float *attn = allocate_floats(count * width, error, error_capacity);
    float *proj = allocate_floats(count * width, error, error_capacity);
    float *q_x = allocate_floats(count * xw, error, error_capacity);
    float *xctx = allocate_floats(count * xw, error, error_capacity);
    float *hidden = allocate_floats(count * ffn, error, error_capacity);
    float *scores = allocate_floats(scores_len, error, error_capacity);
    /* For a multi-row call (the context prefill) the self-attention is a dense
     * batched matmul; a single-row decode step stays scalar over the KV cache
     * (a matvec where sgemm's per-call overhead would dominate). */
    const size_t total_kv = start + count;
    float *score_matrix = NULL;
    float *head_ctx = NULL;
    int batched = 0;
#if defined(MYNAH_USE_ACCELERATE) || defined(MYNAH_USE_OPENBLAS)
    if (count > 1u && total_kv <= (size_t)INT_MAX && hw <= (size_t)INT_MAX) {
        batched = 1;
        score_matrix = allocate_floats(count * total_kv, error, error_capacity);
        head_ctx = allocate_floats(count * hw, error, error_capacity);
    }
#endif
    if (x == NULL || nrm == NULL || qkv == NULL || attn == NULL || proj == NULL ||
        q_x == NULL || xctx == NULL || hidden == NULL || scores == NULL ||
        (batched && (score_matrix == NULL || head_ctx == NULL))) {
        free(x); free(nrm); free(qkv); free(attn); free(proj);
        free(q_x); free(xctx); free(hidden); free(scores);
        free(score_matrix); free(head_ctx);
        return -1;
    }
    for (size_t i = 0; i < count; ++i) {
        const float *pe = cache->position + (start + i) * width;
        for (size_t d = 0; d < width; ++d) x[i * width + d] = input_rows[i * width + d] + pe[d];
    }
    const float self_scale = 1.0f / sqrtf((float)hw);
    const float cross_scale = 1.0f / sqrtf((float)xw);
    char name[256];
    int failed = 0;
    for (size_t layer = 0; layer < cache->layers && !failed; ++layer) {
        mynah_tensor t;
        /* self-attention */
        snprintf(name, sizeof(name), "decoder.layers.%zu.norm_self.weight", layer);
        if (tensor(model->tts, name, &t, error, error_capacity) != 0) { failed = 1; break; }
        layer_norm(x, nrm, count, width, t.data);
        snprintf(name, sizeof(name), "decoder.layers.%zu.self_attention.qkv_net.weight", layer);
        if (mynah_qmat_linear(model->qcache, model->tts, model->backend, name, nrm, qkv,
                              count, width, width * 3u, NULL, error, error_capacity) != 0) { failed = 1; break; }
        float *kbase = cache->self_k + layer * cache->capacity * width;
        float *vbase = cache->self_v + layer * cache->capacity * width;
        for (size_t i = 0; i < count; ++i) {
            memcpy(kbase + (start + i) * width, qkv + i * width * 3u + width, width * sizeof(float));
            memcpy(vbase + (start + i) * width, qkv + i * width * 3u + width * 2u, width * sizeof(float));
        }
#if defined(MYNAH_USE_ACCELERATE) || defined(MYNAH_USE_OPENBLAS)
        if (batched) {
            for (size_t h = 0; h < heads; ++h) {
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                            (int)count, (int)total_kv, (int)hw,
                            self_scale, qkv + h * hw, (int)(width * 3u),
                            kbase + h * hw, (int)width, 0.0f,
                            score_matrix, (int)total_kv);
                for (size_t i = 0; i < count; ++i) {
                    const size_t valid = start + i + 1u;
                    float *row = score_matrix + i * total_kv;
                    softmax_row_inplace(row, valid);
                    for (size_t s = valid; s < total_kv; ++s) row[s] = 0.0f;
                }
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            (int)count, (int)hw, (int)total_kv,
                            1.0f, score_matrix, (int)total_kv,
                            vbase + h * hw, (int)width, 0.0f, head_ctx, (int)hw);
                for (size_t i = 0; i < count; ++i) {
                    memcpy(attn + i * width + h * hw, head_ctx + i * hw, hw * sizeof(float));
                }
            }
        } else
#endif
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
                for (size_t d = 0; d < hw; ++d) {
                    float v = 0.0f;
                    for (size_t s = 0; s <= abs; ++s) v += (scores[s] / denom) * vbase[s * width + h * hw + d];
                    outh[d] = v;
                }
            }
        }
        snprintf(name, sizeof(name), "decoder.layers.%zu.self_attention.o_net.weight", layer);
        if (mynah_qmat_linear(model->qcache, model->tts, model->backend, name, attn, proj,
                              count, width, width, NULL, error, error_capacity) != 0) { failed = 1; break; }
        for (size_t k = 0; k < count * width; ++k) x[k] += proj[k];
        /* cross-attention over cached text memory */
        snprintf(name, sizeof(name), "decoder.layers.%zu.norm_xattn_query.weight", layer);
        if (tensor(model->tts, name, &t, error, error_capacity) != 0) { failed = 1; break; }
        layer_norm(x, nrm, count, width, t.data);
        snprintf(name, sizeof(name), "decoder.layers.%zu.cross_attention.q_net.weight", layer);
        if (mynah_qmat_linear(model->qcache, model->tts, model->backend, name, nrm, q_x,
                              count, width, xw, NULL, error, error_capacity) != 0) { failed = 1; break; }
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
            for (size_t d = 0; d < xw; ++d) {
                float v = 0.0f;
                for (size_t s = 0; s < cache->memory_length; ++s) v += (scores[s] / denom) * cv[s * xw + d];
                outh[d] = v;
            }
        }
        snprintf(name, sizeof(name), "decoder.layers.%zu.cross_attention.o_net.weight", layer);
        if (mynah_qmat_linear(model->qcache, model->tts, model->backend, name, xctx, proj,
                              count, xw, width, NULL, error, error_capacity) != 0) { failed = 1; break; }
        for (size_t k = 0; k < count * width; ++k) x[k] += proj[k];
        /* position-wise FFN (kernel size 1) */
        snprintf(name, sizeof(name), "decoder.layers.%zu.norm_pos_ff.weight", layer);
        if (tensor(model->tts, name, &t, error, error_capacity) != 0) { failed = 1; break; }
        layer_norm(x, nrm, count, width, t.data);
        snprintf(name, sizeof(name), "decoder.layers.%zu.pos_ff.proj.conv.weight", layer);
        if (mynah_qmat_linear(model->qcache, model->tts, model->backend, name, nrm, hidden,
                              count, width, ffn, NULL, error, error_capacity) != 0) { failed = 1; break; }
        for (size_t k = 0; k < count * ffn; ++k) hidden[k] = gelu_tanh(hidden[k]);
        snprintf(name, sizeof(name), "decoder.layers.%zu.pos_ff.o_net.conv.weight", layer);
        if (mynah_qmat_linear(model->qcache, model->tts, model->backend, name, hidden, proj,
                              count, ffn, width, NULL, error, error_capacity) != 0) { failed = 1; break; }
        for (size_t k = 0; k < count * width; ++k) x[k] += proj[k];
    }
    if (!failed) {
        mynah_tensor norm_out;
        if (tensor(model->tts, "decoder.norm_out.weight", &norm_out, error, error_capacity) != 0) {
            failed = 1;
        } else {
            layer_norm(x + (count - 1u) * width, out_last, 1u, width, norm_out.data);
        }
    }
    free(x); free(nrm); free(qkv); free(attn); free(proj);
    free(q_x); free(xctx); free(hidden); free(scores);
    free(score_matrix); free(head_ctx);
    if (!failed) cache->length += count;
    return failed ? -1 : 0;
}

int mynah_tts_synthesize(const mynah_tts_model *model,
                         const mynah_tts_request *request,
                         float **samples, size_t *sample_count,
                         char *error, size_t error_capacity) {
    if (samples != NULL) *samples = NULL;
    if (sample_count != NULL) *sample_count = 0;
    if (model == NULL || request == NULL || samples == NULL || sample_count == NULL ||
        error == NULL || error_capacity == 0 || request->text_ids == NULL || request->text_length == 0) {
        graph_error(error, error_capacity, "invalid synthesis request");
        return -1;
    }
    if (request->speaker >= model->info.speaker_count) {
        graph_error(error, error_capacity, "speaker index is outside the model");
        return -1;
    }
    const size_t width = model->info.hidden_dim;
    const int timing = getenv("MYNAH_TIMING") != NULL;
    const double t_start = timing ? phase_seconds() : 0.0;
    double t_prep = t_start, t_ar = t_start;
    float *memory = NULL;
    if (encode_text(model, request->text_ids, request->text_length, &memory,
                    error, error_capacity) != 0) return -1;
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
    if (decoder_cache_init(model, &cache, memory, request->text_length,
                           context_length + max_steps + 2u, error, error_capacity) != 0 ||
        decoder_run(model, &cache, context, context_length, out_last, error, error_capacity) != 0) {
        decoder_cache_free(&cache);
        free(memory);
        free(codes);
        free(out_last);
        free(audio_row);
        return -1;
    }
    if (timing) t_prep = phase_seconds();
    for (size_t stacked_length = 1u; stacked_length <= max_steps; ++stacked_length) {
        const size_t raw_length = stacked_length * model->info.frame_stacking_factor;
        if (embed_audio_frame(model, codes, max_raw_length, stacked_length - 1u,
                              audio_row, error, error_capacity) != 0) break;
        if (decoder_run(model, &cache, audio_row, 1u, out_last, error, error_capacity) != 0) break;
        int saw_eos = 0;
        size_t step_eos_frame = SIZE_MAX;
        if (request->use_local_transformer) {
            if (sample_local_frame(model, out_last,
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
        ++predicted_stacks;
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
    int result = 0;
    if (error[0] != '\0') {
        result = -1;
    } else if (generated_raw == 0) {
        if (error[0] == '\0') graph_error(error, error_capacity, "decoder generated no audio frames");
        result = -1;
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
                "phase: prep=%.3fs ar=%.3fs codec=%.3fs (stacks=%zu)\n",
                t_prep - t_start, t_ar - t_prep, t_codec - t_ar, generated_stacks);
    }
    decoder_cache_free(&cache);
    free(memory);
    free(codes);
    free(out_last);
    free(audio_row);
    if (result == 0) error[0] = '\0';
    return result;
}

void mynah_tts_free_samples(float *samples) {
    free(samples);
}
