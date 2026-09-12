#include "mynah_tts_internal.h"
#include "graph.h"
#include "kernels.h"
#include "mynah_util.h"
#include "conv1d.h"
#include "codec_nanocodec.h"
#include "threads.h"

#include <float.h>
#include <limits.h>

/* Upper bound on requests stepped together.  Slots are cheap in state but each
 * one holds a full KV cache, so this bounds memory as much as it bounds the
 * pointer arrays kept on the stack in the batched step. */
#define MYNAH_MAX_BATCH 16u

/* Left context, in codec frames, that a streamed chunk must be decoded with.
 *
 * The audio decoder is causal end to end, so an output only ever depends on
 * inputs at or before it, and a chunk decoded with at least its receptive field
 * of history is bit-identical to the same frames decoded from the start.
 *
 * The receptive field, walked backwards through the stack with upsampling rates
 * {8, 8, 4, 2, 2}: the final sample-rate conv contributes 6/1024 frames, each
 * stage's residual layer at most 120 positions at its own rate (kernel 11 over
 * dilations 1, 3, 5, each block being an input conv plus a skip conv), every
 * transposed conv one input position, and the frame-rate pre_conv 6 frames.
 * That sums to under 25 frames; 32 leaves margin without costing anything
 * measurable. */
#define STREAM_CONTEXT_FRAMES 32u

/* Frames to accumulate before emitting. The decode cost per chunk is dominated
 * by the fixed context above, so emitting every single step would pay it 16
 * times over for the same audio. At ~21.5 frames/s this is well under 1 s of
 * added latency to the first chunk, and none to the total. */
#define STREAM_EMIT_FRAMES 16u

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


/* Scratch for the weight-stationary batched projections.  Owned by the caller
 * and sized once for the widest k in the graph, so the step stays
 * allocation-free.  Unused at batch 1, where every projection takes the exact
 * single-row path. */
typedef struct {
    int8_t *qx;   /* MYNAH_MAX_BATCH * k_max */
    float *sx;    /* MYNAH_MAX_BATCH */
    size_t k_max;
    size_t capacity;
} batch_scratch;

static void batch_scratch_free(batch_scratch *scratch) {
    if (scratch == NULL) return;
    free(scratch->qx);
    free(scratch->sx);
    memset(scratch, 0, sizeof(*scratch));
}

static int batch_scratch_init(batch_scratch *scratch, size_t batch, size_t k_max,
                              char *error, size_t error_capacity) {
    memset(scratch, 0, sizeof(*scratch));
    if (batch <= 1u) return 0;   /* single-row path never quantizes in bulk */
    if (k_max == 0u || batch > SIZE_MAX / k_max) {
        mynah_graph_error(error, error_capacity, "batch scratch dimensions overflow");
        return -1;
    }
    scratch->qx = (int8_t *)malloc(batch * k_max);
    scratch->sx = (float *)malloc(batch * sizeof(float));
    if (scratch->qx == NULL || scratch->sx == NULL) {
        batch_scratch_free(scratch);
        mynah_graph_error(error, error_capacity, "out of memory allocating batch scratch");
        return -1;
    }
    scratch->k_max = k_max;
    scratch->capacity = batch;
    return 0;
}

static void layer_norm(const float *input, float *output, size_t length,
                       size_t width, const float *weight) {
    mynah_layernorm_f32(input, weight, NULL, output, length, width, 1.0e-5f);
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


static int causal_conv_ffn(const mynah_weights *file, const mynah_backend *backend,
                           const char *prefix,
                           size_t layer, const float *input, float *output,
                           size_t length, size_t width, size_t ffn_width,
                           size_t kernel, char *error, size_t error_capacity) {
    char name[256];
    mynah_tensor proj;
    mynah_tensor out_net;
    snprintf(name, sizeof(name), "%s.layers.%zu.pos_ff.proj.conv.weight", prefix, layer);
    if (mynah_tensor_get(file, name, &proj, error, error_capacity) != 0) return -1;
    snprintf(name, sizeof(name), "%s.layers.%zu.pos_ff.o_net.conv.weight", prefix, layer);
    if (mynah_tensor_get(file, name, &out_net, error, error_capacity) != 0) return -1;
    float *hidden = mynah_alloc_floats(length * ffn_width, error, error_capacity);
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
    const size_t wide = ffn_width > width ? ffn_width : width;
    if (length <= (size_t)INT_MAX && width <= (size_t)INT_MAX &&
        ffn_width <= (size_t)INT_MAX && kernel > 0u &&
        wide <= SIZE_MAX / kernel && wide * kernel <= (size_t)INT_MAX &&
        length <= SIZE_MAX / (wide * kernel)) {
        /* A causal conv-FFN is a single matmul once the input is unfolded over
         * the kernel.  The stored weight is [out][in][kernel] row-major, so an
         * output row is already contiguous in (in * kernel + k): building the
         * matching im2col of the *input* lets sgemm read the weight straight
         * from the mapping.
         *
         * The previous form ran `kernel` shifted matmuls against transposed tap
         * copies, which rebuilt and re-read every conv weight on every call --
         * 340 MB per request on this model, in a stride-`kernel` gather.  The
         * unfolded activation is length * width * kernel instead, which is
         * three orders of magnitude smaller and stays in cache. */
        float *col = mynah_alloc_floats(length * wide * kernel, error, error_capacity);
        if (col == NULL) {
            free(hidden);
            return -1;
        }
        mynah_unfold_causal(input, col, length, width, kernel);
        int failed = mynah_graph_sgemm(backend, 0, 1, length, ffn_width, width * kernel,
                                 1.0f, col, width * kernel,
                                 proj.data, width * kernel, 0.0f,
                                 hidden, ffn_width, error, error_capacity) != 0;
        if (!failed) {
            mynah_gelu_f32_scalar(hidden, length * ffn_width);
            mynah_unfold_causal(hidden, col, length, ffn_width, kernel);
            failed = mynah_graph_sgemm(backend, 0, 1, length, width, ffn_width * kernel,
                                 1.0f, col, ffn_width * kernel,
                                 out_net.data, ffn_width * kernel, 0.0f,
                                 output, width, error, error_capacity) != 0;
        }
        free(col);
        free(hidden);
        return failed ? -1 : 0;
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
            hidden[t * ffn_width + o] = mynah_gelu_tanh(value);
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

static int self_attention(const mynah_weights *file, const mynah_backend *backend,
                          const char *prefix,
                          size_t layer, const float *input, float *output,
                          size_t length, size_t width, size_t heads,
                          char *error, size_t error_capacity) {
    char name[256];
    mynah_tensor qkv;
    mynah_tensor projection;
    snprintf(name, sizeof(name), "%s.layers.%zu.self_attention.qkv_net.weight", prefix, layer);
    if (mynah_tensor_get(file, name, &qkv, error, error_capacity) != 0) return -1;
    snprintf(name, sizeof(name), "%s.layers.%zu.self_attention.o_net.weight", prefix, layer);
    if (mynah_tensor_get(file, name, &projection, error, error_capacity) != 0) return -1;
    const size_t head_width = width / heads;
    float *qkv_values = mynah_alloc_floats(length * width * 3u, error, error_capacity);
    float *context = mynah_alloc_floats(length * width, error, error_capacity);
    float *scores = mynah_alloc_floats(length, error, error_capacity);
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
        float *queries = mynah_alloc_floats(length * head_width, error, error_capacity);
        float *keys = mynah_alloc_floats(length * head_width, error, error_capacity);
        float *values = mynah_alloc_floats(length * head_width, error, error_capacity);
        float *score_matrix = mynah_alloc_floats(length * length, error, error_capacity);
        float *head_context = mynah_alloc_floats(length * head_width, error, error_capacity);
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
            mynah_graph_sgemm(backend, 0, 1, (int)length, (int)length, (int)head_width, 1.0f / sqrtf((float)head_width), queries, (int)head_width, keys, (int)head_width, 0.0f, score_matrix, (int)length, error, error_capacity);
            for (size_t t = 0; t < length; ++t) {
                for (size_t s = t + 1u; s < length; ++s) score_matrix[t * length + s] = 0.0f;
                softmax_row_inplace(score_matrix + t * length, t + 1u);
                for (size_t s = t + 1u; s < length; ++s) score_matrix[t * length + s] = 0.0f;
            }
            mynah_graph_sgemm(backend, 0, 0, (int)length, (int)head_width, (int)length, 1.0f, score_matrix, (int)length, values, (int)head_width, 0.0f, head_context, (int)head_width, error, error_capacity);
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

static int cross_attention(const mynah_weights *file, const mynah_backend *backend,
                           const char *prefix,
                           size_t layer, const float *input, float *output,
                           size_t length, const float *memory, size_t memory_length,
                           size_t width, char *error, size_t error_capacity) {
    char name[256];
    mynah_tensor q_weight;
    mynah_tensor kv_weight;
    mynah_tensor projection;
    snprintf(name, sizeof(name), "%s.layers.%zu.cross_attention.q_net.weight", prefix, layer);
    if (mynah_tensor_get(file, name, &q_weight, error, error_capacity) != 0) return -1;
    snprintf(name, sizeof(name), "%s.layers.%zu.cross_attention.kv_net.weight", prefix, layer);
    if (mynah_tensor_get(file, name, &kv_weight, error, error_capacity) != 0) return -1;
    snprintf(name, sizeof(name), "%s.layers.%zu.cross_attention.o_net.weight", prefix, layer);
    if (mynah_tensor_get(file, name, &projection, error, error_capacity) != 0) return -1;
    const size_t attention_width = q_weight.shape[0];
    float *q = mynah_alloc_floats(length * attention_width, error, error_capacity);
    float *kv = mynah_alloc_floats(memory_length * attention_width * 2u, error, error_capacity);
    float *context = mynah_alloc_floats(length * attention_width, error, error_capacity);
    float *scores = mynah_alloc_floats(memory_length, error, error_capacity);
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
        float *keys = mynah_alloc_floats(memory_length * attention_width, error, error_capacity);
        float *values = mynah_alloc_floats(memory_length * attention_width, error, error_capacity);
        float *score_matrix = mynah_alloc_floats(length * memory_length, error, error_capacity);
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
        mynah_graph_sgemm(backend, 0, 1, (int)length, (int)memory_length, (int)attention_width, 1.0f / sqrtf((float)attention_width), q, (int)attention_width, keys, (int)attention_width, 0.0f, score_matrix, (int)memory_length, error, error_capacity);
        for (size_t t = 0; t < length; ++t) {
            softmax_row_inplace(score_matrix + t * memory_length, memory_length);
        }
        mynah_graph_sgemm(backend, 0, 0, (int)length, (int)attention_width, (int)memory_length, 1.0f, score_matrix, (int)memory_length, values, (int)attention_width, 0.0f, context, (int)attention_width, error, error_capacity);
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

static int transformer_stack(const mynah_weights *file, const mynah_backend *backend,
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
    if (mynah_tensor_get(file, name, &position, error, error_capacity) != 0) return -1;
    float *working = mynah_alloc_floats(length * width, error, error_capacity);
    if (working == NULL) return -1;
    memcpy(working, states, length * width * sizeof(float));
    for (size_t t = 0; t < length; ++t) {
        for (size_t d = 0; d < width; ++d) working[t * width + d] += position.data[t * width + d];
    }
    float *normalized = mynah_alloc_floats(length * width, error, error_capacity);
    float *residual = mynah_alloc_floats(length * width, error, error_capacity);
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
        if (mynah_tensor_get(file, name, &norm_self, error, error_capacity) != 0) break;
        layer_norm(working, normalized, length, width, norm_self.data);
        if (self_attention(file, backend, prefix, layer, normalized, residual, length, width, heads,
                           error, error_capacity) != 0) break;
        for (size_t i = 0; i < length * width; ++i) working[i] += residual[i];

        if (has_cross_attention) {
            snprintf(name, sizeof(name), "%s.layers.%zu.norm_xattn_query.weight", prefix, layer);
            mynah_tensor norm_query;
            if (mynah_tensor_get(file, name, &norm_query, error, error_capacity) != 0) break;
            layer_norm(working, normalized, length, width, norm_query.data);
            if (memory_normalized == NULL) {
                memory_normalized = mynah_alloc_floats(memory_length * width, error, error_capacity);
                if (memory_normalized == NULL) break;
            }
            snprintf(name, sizeof(name), "%s.layers.%zu.norm_xattn_memory.weight", prefix, layer);
            mynah_tensor norm_memory;
            if (mynah_tensor_get(file, name, &norm_memory, error, error_capacity) != 0) break;
            layer_norm(memory, memory_normalized, memory_length, width, norm_memory.data);
            if (cross_attention(file, backend, prefix, layer, normalized, residual, length,
                                memory_normalized, memory_length, width, error,
                                error_capacity) != 0) break;
            for (size_t i = 0; i < length * width; ++i) working[i] += residual[i];
        }

        snprintf(name, sizeof(name), "%s.layers.%zu.norm_pos_ff.weight", prefix, layer);
        mynah_tensor norm_ff;
        if (mynah_tensor_get(file, name, &norm_ff, error, error_capacity) != 0) break;
        layer_norm(working, normalized, length, width, norm_ff.data);
        if (causal_conv_ffn(file, backend, prefix, layer, normalized, residual, length, width,
                            ffn_width, kernel, error, error_capacity) != 0) break;
        for (size_t i = 0; i < length * width; ++i) working[i] += residual[i];
        if (layer + 1u == layers && apply_norm_out) {
            snprintf(name, sizeof(name), "%s.norm_out.weight", prefix);
            mynah_tensor norm_out;
            if (mynah_tensor_get(file, name, &norm_out, error, error_capacity) != 0) break;
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
    if (mynah_tensor_get(model->tts, "text_embedding.weight", &embedding, error, error_capacity) != 0) return -1;
    if (count == 0 || count > model->info.text_max_length || embedding.shape[1] != width) {
        mynah_graph_error(error, error_capacity, "text length or embedding shape is invalid");
        return -1;
    }
    float *states = mynah_alloc_floats(count * width, error, error_capacity);
    float *result = mynah_alloc_floats(count * width, error, error_capacity);
    if (states == NULL || result == NULL) {
        free(states);
        free(result);
        return -1;
    }
    for (size_t t = 0; t < count; ++t) {
        if (ids[t] < 0 || (size_t)ids[t] >= embedding.shape[0]) {
            free(states);
            free(result);
            mynah_graph_error(error, error_capacity, "text token id is outside vocabulary");
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
        mynah_graph_error(error, error_capacity, "v1 requires a frame stacking factor of two");
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
                if (mynah_tensor_get(model->tts, name, &table, error, error_capacity) != 0) return -1;
                table_data = table.data;
                table_rows = table.shape[0];
            }
            if (code >= table_rows) {
                mynah_graph_error(error, error_capacity, "audio token id is outside vocabulary");
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
    /* The tensor names are kept beside the pointers so the quantized path can
     * key its converted-weight cache without re-formatting a name per step.
     * These four matmuls are the hot loop: 16 sequential streams per stacked
     * frame re-read every local-transformer weight, so they dominate decode. */
    char qkv_name[4][96];
    char o_name[4][96];
    char ffn_up_name[4][96];
    char ffn_down_name[4][96];
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

/* Number of audio streams in one stacked frame.
 *
 * codebook_count and frame_stacking_factor are `unsigned` fields read from the
 * pack's model.json, so `a * b` is evaluated in 32-bit unsigned arithmetic and
 * only widened afterwards: a malformed pack could wrap the product before it
 * ever reaches size_t, and a guard written as `a > SIZE_MAX / b` does not stop
 * that because the guard promotes while the multiplication does not.  Widen
 * both operands first so the arithmetic happens in size_t. */
static size_t stacked_stream_count(const mynah_tts_model *model) {
    return (size_t)model->info.codebook_count *
           (size_t)model->info.frame_stacking_factor;
}

void *mynah_graph_local_projection_cache_new(const mynah_tts_model *model) {
    if (model == NULL) return NULL;
    if (model->info.frame_stacking_factor == 0) return NULL;
    const size_t streams = stacked_stream_count(model);
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
        if (mynah_tensor_get(model->tts, name, &tensor_view, error, sizeof(error)) != 0) {
            local_projection_cache_free_impl(cache);
            return NULL;
        }
        cache->projection_weights[stream] = tensor_view.data;
        snprintf(name, sizeof(name),
                 "local_transformer_out_projections.%zu.bias", stream);
        if (mynah_tensor_get(model->tts, name, &tensor_view, error, sizeof(error)) != 0) {
            local_projection_cache_free_impl(cache);
            return NULL;
        }
        cache->projection_biases[stream] = tensor_view.data;
        snprintf(name, sizeof(name), "audio_embeddings.%zu.weight", stream);
        if (mynah_tensor_get(model->tts, name, &tensor_view, error, sizeof(error)) != 0) {
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
    workspace->x = mynah_alloc_floats(cache->width, error, error_capacity);
    workspace->nrm = mynah_alloc_floats(cache->width, error, error_capacity);
    workspace->qkv = mynah_alloc_floats(cache->width * 3u, error, error_capacity);
    workspace->attn = mynah_alloc_floats(cache->width, error, error_capacity);
    workspace->proj = mynah_alloc_floats(cache->width, error, error_capacity);
    workspace->hidden = mynah_alloc_floats(cache->ffn_width, error, error_capacity);
    workspace->scores = mynah_alloc_floats(cache->capacity, error, error_capacity);
#if defined(MYNAH_USE_ACCELERATE)
    workspace->gelu_scratch = mynah_alloc_floats(cache->ffn_width, error, error_capacity);
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
    if (mynah_tensor_get(model->tts, "local_transformer.position_embeddings.weight", &position,
               error, error_capacity) != 0) {
        return -1;
    }
    cache->position = position.data;
    cache->k = mynah_alloc_floats(cache->layers * capacity * width, error, error_capacity);
    cache->v = mynah_alloc_floats(cache->layers * capacity * width, error, error_capacity);
    if (cache->k == NULL || cache->v == NULL) {
        local_cache_free(cache);
        return -1;
    }
    /* Pre-resolve weight pointers (eliminate per-step snprintf+lookup). */
    for (size_t l = 0; l < cache->layers && l < 4u; ++l) {
        char nm[256]; mynah_tensor t;
        snprintf(nm, sizeof(nm), "local_transformer.layers.%zu.norm_self.weight", l);
        if (mynah_tensor_get(model->tts, nm, &t, error, error_capacity)!=0) return -1;
        cache->norm_self[l] = t.data;
        snprintf(nm, sizeof(nm), "local_transformer.layers.%zu.self_attention.qkv_net.weight", l);
        if (mynah_tensor_get(model->tts, nm, &t, error, error_capacity)!=0) return -1;
        cache->qkv_w[l] = t.data;
        snprintf(cache->qkv_name[l], sizeof(cache->qkv_name[l]), "%s", nm);
        snprintf(nm, sizeof(nm), "local_transformer.layers.%zu.self_attention.o_net.weight", l);
        if (mynah_tensor_get(model->tts, nm, &t, error, error_capacity)!=0) return -1;
        cache->o_w[l] = t.data;
        snprintf(cache->o_name[l], sizeof(cache->o_name[l]), "%s", nm);
        snprintf(nm, sizeof(nm), "local_transformer.layers.%zu.norm_pos_ff.weight", l);
        if (mynah_tensor_get(model->tts, nm, &t, error, error_capacity)!=0) return -1;
        cache->norm_ff[l] = t.data;
        snprintf(nm, sizeof(nm), "local_transformer.layers.%zu.pos_ff.proj.conv.weight", l);
        if (mynah_tensor_get(model->tts, nm, &t, error, error_capacity)!=0) return -1;
        cache->ffn_up_w[l] = t.data;
        snprintf(cache->ffn_up_name[l], sizeof(cache->ffn_up_name[l]), "%s", nm);
        snprintf(nm, sizeof(nm), "local_transformer.layers.%zu.pos_ff.o_net.conv.weight", l);
        if (mynah_tensor_get(model->tts, nm, &t, error, error_capacity)!=0) return -1;
        cache->ffn_down_w[l] = t.data;
        snprintf(cache->ffn_down_name[l], sizeof(cache->ffn_down_name[l]), "%s", nm);
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
            mynah_graph_error(error, error_capacity, "Metal local-transformer buffers unavailable");
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
        mynah_graph_error(error, error_capacity, "Metal local transformer cache overflow");
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


/* Per-request state for the local autoregressive helper.
 *
 * The local transformer restarts at position 0 for every stacked frame, but
 * its KV cache, workspace and sampler scratch are shape-constant for the whole
 * request.  Allocating them once and resetting `length` keeps the decode loop
 * allocation-free -- previously every frame paid two KV allocations plus the
 * workspace, the sampler scratch and, under Metal, ten device allocations. */
typedef struct {
    local_cache cache;
    local_workspace workspace;
    float *row_in;
    float *row_out;
    float *logits;        /* audio_vocab_size */
    size_t *top_indices;  /* top_count, sampling only */
    float *top_logits;    /* top_count, sampling only */
    size_t top_count;
} local_frame_state;

static void local_frame_state_free(local_frame_state *state) {
    if (state == NULL) return;
    free(state->row_in);
    free(state->row_out);
    free(state->logits);
    free(state->top_indices);
    free(state->top_logits);
    local_workspace_free(&state->workspace);
    local_cache_free(&state->cache);
    memset(state, 0, sizeof(*state));
}

/* `top_count` is 0 when the request is greedy, so the sampler scratch is only
 * allocated for requests that actually sample. */
static int local_frame_state_init(const mynah_tts_model *model,
                                  local_frame_state *state, size_t top_count,
                                  char *error, size_t error_capacity) {
    memset(state, 0, sizeof(*state));
    const size_t width = model->info.hidden_dim;
    const size_t stream_count = stacked_stream_count(model);
    if (local_cache_init(model, &state->cache, stream_count + 1u,
                         error, error_capacity) != 0) {
        return -1;
    }
    if (local_workspace_init(&state->workspace, &state->cache,
                             error, error_capacity) != 0) {
        local_cache_free(&state->cache);
        mynah_graph_error(error, error_capacity, "out of memory allocating local workspace");
        return -1;
    }
    state->top_count = top_count;
    state->row_in = mynah_alloc_floats(width, error, error_capacity);
    state->row_out = mynah_alloc_floats(width, error, error_capacity);
    state->logits = mynah_alloc_floats(model->info.audio_vocab_size, error, error_capacity);
    if (top_count > 0u) {
        state->top_indices = (size_t *)malloc(top_count * sizeof(*state->top_indices));
        state->top_logits = (float *)malloc(top_count * sizeof(*state->top_logits));
    }
    if (state->row_in == NULL || state->row_out == NULL || state->logits == NULL ||
        (top_count > 0u && (state->top_indices == NULL || state->top_logits == NULL))) {
        local_frame_state_free(state);
        mynah_graph_error(error, error_capacity, "out of memory allocating local frame state");
        return -1;
    }
    return 0;
}

/* One local-transformer position for `batch` independent requests.
 *
 * The local transformer is read far more than the decoder is: it runs once per
 * stacked stream, so a single decode step walks its weights sixteen times.
 * That makes it the larger of the two batching prizes even though it is the
 * smaller network.  Same shape as decoder_step_batch -- projections batched,
 * attention per slot over each slot's own KV -- and likewise exact at batch 1. */
static int local_step_batch(const mynah_tts_model *model, local_cache *const *caches,
                            local_workspace *const *workspaces,
                            const float *const *inputs, float *const *outs,
                            size_t batch, const batch_scratch *scratch,
                            char *error, size_t error_capacity) {
    if (batch == 0u) return 0;
    if (batch > MYNAH_MAX_BATCH) {
        mynah_graph_error(error, error_capacity, "local batch exceeds MYNAH_MAX_BATCH");
        return -1;
    }
    const local_cache *c0 = caches[0];
    const size_t width = c0->width;
    const size_t heads = c0->heads;
    const size_t hw = c0->head_width;
    const size_t ffn = c0->ffn_width;
    const float scale = 1.0f / sqrtf((float)hw);
    int8_t *qx = scratch == NULL ? NULL : scratch->qx;
    float *sx = scratch == NULL ? NULL : scratch->sx;
    const float *in_ptrs[MYNAH_MAX_BATCH];
    float *out_ptrs[MYNAH_MAX_BATCH];

    for (size_t b = 0; b < batch; ++b) {
        local_cache *c = caches[b];
        if (c->length >= c->capacity) {
            mynah_graph_error(error, error_capacity, "local transformer cache overflow");
            return -1;
        }
        float *x = workspaces[b]->x;
        const float *pe = c->position + c->length * width;
        for (size_t d = 0; d < width; ++d) x[d] = inputs[b][d] + pe[d];
    }
    for (size_t layer = 0; layer < c0->layers; ++layer) {
        for (size_t b = 0; b < batch; ++b) {
            layer_norm(workspaces[b]->x, workspaces[b]->nrm, 1u, width,
                       caches[b]->norm_self[layer]);
            in_ptrs[b] = workspaces[b]->nrm;
            out_ptrs[b] = workspaces[b]->qkv;
        }
        if (mynah_qmat_linear_batched(model->qcache, model->backend,
                                      c0->qkv_name[layer], c0->qkv_w[layer],
                                      in_ptrs, out_ptrs, batch, width, width * 3u,
                                      NULL, qx, sx, error, error_capacity) != 0) return -1;
        for (size_t b = 0; b < batch; ++b) {
            local_cache *c = caches[b];
            local_workspace *w = workspaces[b];
            const size_t p = c->length;
            float *kb = c->k + layer * c->capacity * width;
            float *vb = c->v + layer * c->capacity * width;
            memcpy(kb + p * width, w->qkv + width, width * sizeof(float));
            memcpy(vb + p * width, w->qkv + width * 2u, width * sizeof(float));
            for (size_t h = 0; h < heads; ++h) {
                const float *qh = w->qkv + h * hw;
                float maxv = -FLT_MAX;
                for (size_t s = 0; s <= p; ++s) {
                    const float *k = kb + s * width + h * hw;
                    float sc = 0.0f;
                    for (size_t d = 0; d < hw; ++d) sc += qh[d] * k[d];
                    sc *= scale;
                    w->scores[s] = sc;
                    if (sc > maxv) maxv = sc;
                }
                float denom = 0.0f;
                for (size_t s = 0; s <= p; ++s) {
                    w->scores[s] = expf(w->scores[s] - maxv);
                    denom += w->scores[s];
                }
                float *outh = w->attn + h * hw;
                memset(outh, 0, hw * sizeof(float));
                for (size_t s = 0; s <= p; ++s)
                    mynah_axpy_f32(outh, vb + s * width + h * hw, w->scores[s] / denom, hw);
            }
            in_ptrs[b] = w->attn;
            out_ptrs[b] = w->proj;
        }
        if (mynah_qmat_linear_batched(model->qcache, model->backend,
                                      c0->o_name[layer], c0->o_w[layer],
                                      in_ptrs, out_ptrs, batch, width, width,
                                      NULL, qx, sx, error, error_capacity) != 0) return -1;
        for (size_t b = 0; b < batch; ++b) {
            local_workspace *w = workspaces[b];
            for (size_t d = 0; d < width; ++d) w->x[d] += w->proj[d];
            layer_norm(w->x, w->nrm, 1u, width, caches[b]->norm_ff[layer]);
            in_ptrs[b] = w->nrm;
            out_ptrs[b] = w->hidden;
        }
        if (mynah_qmat_linear_batched(model->qcache, model->backend,
                                      c0->ffn_up_name[layer], c0->ffn_up_w[layer],
                                      in_ptrs, out_ptrs, batch, width, ffn,
                                      NULL, qx, sx, error, error_capacity) != 0) return -1;
        for (size_t b = 0; b < batch; ++b) {
            local_workspace *w = workspaces[b];
            mynah_gelu_tanh_array(w->hidden, ffn, w->gelu_scratch);
            in_ptrs[b] = w->hidden;
            out_ptrs[b] = w->proj;
        }
        if (mynah_qmat_linear_batched(model->qcache, model->backend,
                                      c0->ffn_down_name[layer], c0->ffn_down_w[layer],
                                      in_ptrs, out_ptrs, batch, ffn, width,
                                      NULL, qx, sx, error, error_capacity) != 0) return -1;
        for (size_t b = 0; b < batch; ++b) {
            local_workspace *w = workspaces[b];
            for (size_t d = 0; d < width; ++d) w->x[d] += w->proj[d];
        }
    }
    for (size_t b = 0; b < batch; ++b) {
        memcpy(outs[b], workspaces[b]->x, width * sizeof(float));
        caches[b]->length += 1u;
    }
    return 0;
}

/* One request's slice of a batched local frame. */
typedef struct {
    local_frame_state *state;
    const float *decoder_last;
    const float *decoder_dev_last;
    unsigned *codes;
    size_t raw_offset;
    size_t code_stride;
    size_t generated_raw_length;
    size_t min_raw_length;
    float temperature;
    unsigned topk;
    uint64_t *rng_state;
    /* filled in */
    int saw_eos;
    size_t eos_frame;
    int failed;
} local_batch_item;

/* Emit one stacked frame (two raw codec frames) for every item.
 *
 * The stacked streams are autoregressive within a request -- stream n+1 is fed
 * the token sampled at stream n -- so they cannot be batched against each
 * other.  Across requests they can: at any given stream index every item needs
 * the same weights, so the loop runs stream by stream with all items in step.
 *
 * All scratch lives in each item's state and is reused across frames; the only
 * per-frame reset is the KV length, because positions [0, length) are always
 * written before they are read. */
static int sample_local_frame_batch(const mynah_tts_model *model,
                                    local_batch_item *items, size_t count,
                                    const batch_scratch *scratch,
                                    char *error, size_t error_capacity) {
    if (count == 0u) return 0;
    if (count > MYNAH_MAX_BATCH) {
        mynah_graph_error(error, error_capacity, "local frame batch exceeds MYNAH_MAX_BATCH");
        return -1;
    }
    const size_t width = model->info.hidden_dim;
    const size_t stream_count = stacked_stream_count(model);
    const size_t vocab = model->info.audio_vocab_size;
    const unsigned eos_id = model->info.audio_eos_id;
    const local_projection_cache *projection_cache =
        (const local_projection_cache *)model->local_projection_cache;
    const char *metal_local_env = getenv("MYNAH_METAL_GPU_LOCAL");
    const char *metal_attention_local_env = getenv("MYNAH_METAL_GPU_ATTENTION");

    local_cache *caches[MYNAH_MAX_BATCH];
    local_workspace *workspaces[MYNAH_MAX_BATCH];
    const float *step_in[MYNAH_MAX_BATCH];
    float *step_out[MYNAH_MAX_BATCH];
    const float *proj_in[MYNAH_MAX_BATCH];
    float *proj_out[MYNAH_MAX_BATCH];
    size_t live_index[MYNAH_MAX_BATCH];

    /* The resident-GPU local step is a single-request path; a batch of one may
     * still take it. */
    local_cache *c0 = &items[0].state->cache;
    const int gpu_local = count == 1u && c0->gpu_ready && c0->dev_backend != NULL &&
        (metal_local_env == NULL || strcmp(metal_local_env, "0") != 0) &&
        (strcmp(mynah_backend_name(c0->dev_backend), "metal") != 0 ||
         (metal_attention_local_env != NULL &&
          strcmp(metal_attention_local_env, "1") == 0));

    for (size_t i = 0; i < count; ++i) {
        local_batch_item *it = &items[i];
        it->saw_eos = 0;
        it->eos_frame = SIZE_MAX;
        it->failed = 0;
        it->state->cache.length = 0u;
        const size_t top_count = it->topk < vocab ? it->topk : vocab;
        const int sampling = it->temperature > 0.0f && it->topk > 1u &&
                             it->rng_state != NULL;
        if (sampling && top_count > it->state->top_count) {
            mynah_graph_error(error, error_capacity, "local sampler scratch too small");
            return -1;
        }
        if (!gpu_local || it->decoder_dev_last == NULL) {
            memcpy(it->state->row_in, it->decoder_last, width * sizeof(float));
        }
        if (gpu_local) {
            local_cache *lc = &it->state->cache;
            if (((it->decoder_dev_last != NULL
                      ? mynah_backend_copy_dev(lc->dev_backend, lc->dev_input,
                                               it->decoder_dev_last, width,
                                               error, error_capacity)
                      : mynah_backend_h2d(lc->dev_backend, it->decoder_last,
                                          lc->dev_input, width,
                                          error, error_capacity)) != 0)) {
                return -1;
            }
        }
    }
    char name[160];
    for (size_t stream = 0; stream < stream_count; ++stream) {
        size_t live = 0;
        for (size_t i = 0; i < count; ++i) {
            if (items[i].failed) continue;
            live_index[live] = i;
            caches[live] = &items[i].state->cache;
            workspaces[live] = &items[i].state->workspace;
            step_in[live] = items[i].state->row_in;
            step_out[live] = items[i].state->row_out;
            ++live;
        }
        if (live == 0u) break;
        float *dev_row_out = NULL;
        if (gpu_local) {
            if (local_step_device(model, c0, c0->dev_input, &dev_row_out,
                                  error, error_capacity) != 0) return -1;
        } else if (local_step_batch(model, caches, workspaces, step_in, step_out,
                                    live, scratch, error, error_capacity) != 0) {
            return -1;
        }
        /* Output projection: same weight for every item at this stream. */
        const float *bias_data = NULL;
        const float *projection_weight = NULL;
        if (projection_cache != NULL && stream < projection_cache->stream_count) {
            bias_data = projection_cache->projection_biases[stream];
            projection_weight = projection_cache->projection_weights[stream];
        } else {
            snprintf(name, sizeof(name),
                     "local_transformer_out_projections.%zu.bias", stream);
            mynah_tensor bias;
            if (mynah_tensor_get(model->tts, name, &bias, error, error_capacity) != 0) return -1;
            bias_data = bias.data;
            snprintf(name, sizeof(name),
                     "local_transformer_out_projections.%zu.weight", stream);
            mynah_tensor projection;
            if (mynah_tensor_get(model->tts, name, &projection, error, error_capacity) != 0) return -1;
            projection_weight = projection.data;
        }
        snprintf(name, sizeof(name),
                 "local_transformer_out_projections.%zu.weight", stream);

        /* Greedy items can fuse projection and argmax; sampling items need the
         * logits.  Fusing is only worth it alone -- in a batch the shared pass
         * over the projection weight is the bigger win. */
        int need_logits = 0;
        for (size_t j = 0; j < live; ++j) {
            local_batch_item *it = &items[live_index[j]];
            const int sampling = it->temperature > 0.0f && it->topk > 1u &&
                                 it->rng_state != NULL;
            if (sampling) need_logits = 1;
        }
        const int fuse_greedy = live == 1u && !need_logits && !gpu_local &&
            (getenv("MYNAH_FUSED_GREEDY") == NULL ||
             strcmp(getenv("MYNAH_FUSED_GREEDY"), "0") != 0);
        unsigned fused_argmax = 0;
        int fused = 1;
        if (gpu_local) {
            local_batch_item *it = &items[live_index[0]];
            const int sampling = it->temperature > 0.0f && it->topk > 1u &&
                                 it->rng_state != NULL;
            if (mynah_backend_matmul_d2d(c0->dev_backend, dev_row_out, c0->dev_logits,
                                          1u, width, vocab, projection_weight,
                                          bias_data, error, error_capacity) != 0) {
                return -1;
            }
            const int forbid_eos = it->generated_raw_length < it->min_raw_length;
            if (!sampling) {
                fused = 0;
                if (mynah_backend_argmax_dev(c0->dev_backend, c0->dev_logits, vocab,
                                              model->info.codebook_size, eos_id,
                                              !forbid_eos, &fused_argmax, error,
                                              error_capacity) != 0) {
                    return -1;
                }
            } else if (mynah_backend_d2h(c0->dev_backend, c0->dev_logits,
                                          it->state->logits, vocab,
                                          error, error_capacity) != 0) {
                return -1;
            }
        } else if (fuse_greedy) {
            local_batch_item *it = &items[live_index[0]];
            const int forbid_eos = it->generated_raw_length < it->min_raw_length;
            const int outcome = mynah_qmat_greedy_argmax_resolved(
                model->qcache, name, projection_weight, it->state->row_out, width,
                vocab, bias_data, model->info.codebook_size, eos_id, !forbid_eos,
                &fused_argmax, error, error_capacity);
            if (outcome < 0) return -1;
            fused = outcome;
        }
        if (!gpu_local && fused != 0) {
            for (size_t j = 0; j < live; ++j) {
                local_batch_item *it = &items[live_index[j]];
                proj_in[j] = it->state->row_out;
                proj_out[j] = it->state->logits;
            }
            if (mynah_qmat_linear_batched(model->qcache, model->backend, name,
                                          projection_weight, proj_in, proj_out,
                                          live, width, vocab, bias_data,
                                          scratch == NULL ? NULL : scratch->qx,
                                          scratch == NULL ? NULL : scratch->sx,
                                          error, error_capacity) != 0) {
                return -1;
            }
        }
        for (size_t j = 0; j < live; ++j) {
            local_batch_item *it = &items[live_index[j]];
            const int sampling = it->temperature > 0.0f && it->topk > 1u &&
                                 it->rng_state != NULL;
            const size_t top_count = it->topk < vocab ? it->topk : vocab;
            const int forbid_eos = it->generated_raw_length < it->min_raw_length;
            float *logits = it->state->logits;
            unsigned argmax = fused_argmax;
            if (fused != 0) {
                float best = -FLT_MAX;
                argmax = 0;
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
            /* argmax_or_multinomial: EOS fires if either the greedy or the
             * sampled token is AUDIO_EOS in any codebook of this frame. */
            int stream_eos = (argmax == eos_id);
            if (sampling) {
                size_t *top_indices = it->state->top_indices;
                float *top_logits = it->state->top_logits;
                size_t used = 0;
                for (size_t candidate = 0; candidate < vocab; ++candidate) {
                    size_t insert = used;
                    while (insert > 0 && logits[candidate] > top_logits[insert - 1u]) --insert;
                    if (used < top_count) {
                        for (size_t q = used; q > insert; --q) {
                            top_indices[q] = top_indices[q - 1u];
                            top_logits[q] = top_logits[q - 1u];
                        }
                        top_indices[insert] = candidate;
                        top_logits[insert] = logits[candidate];
                        ++used;
                    } else if (insert < top_count) {
                        for (size_t q = top_count - 1u; q > insert; --q) {
                            top_indices[q] = top_indices[q - 1u];
                            top_logits[q] = top_logits[q - 1u];
                        }
                        top_indices[insert] = candidate;
                        top_logits[insert] = logits[candidate];
                    }
                }
                float maximum = top_logits[0];
                float total = 0.0f;
                for (size_t q = 0; q < used; ++q) {
                    top_logits[q] = expf((top_logits[q] - maximum) / it->temperature);
                    total += top_logits[q];
                }
                uint64_t state_bits = *it->rng_state;
                state_bits ^= state_bits << 13;
                state_bits ^= state_bits >> 7;
                state_bits ^= state_bits << 17;
                *it->rng_state = state_bits == 0 ? UINT64_C(0x9e3779b97f4a7c15) : state_bits;
                const float draw = ((float)((*it->rng_state >> 40) & UINT64_C(0xffffff)) /
                                    (float)UINT64_C(0x1000000)) * total;
                float cumulative = 0.0f;
                for (size_t q = 0; q < used; ++q) {
                    cumulative += top_logits[q];
                    if (draw <= cumulative || q + 1u == used) {
                        value = (unsigned)top_indices[q];
                        break;
                    }
                }
            }
            if (value == eos_id) stream_eos = 1;
            if (stream_eos) {
                it->saw_eos = 1;
                const size_t frame = stream / model->info.codebook_count;
                if (frame < it->eos_frame) it->eos_frame = frame;
            }
            /* Feed the sampled token (EOS included) back into the local
             * transformer, but collapse any special token to 0 in the codes we
             * hand to the codec so it only ever sees real FSQ indices. */
            const unsigned emit = value < vocab ? value : 0u;
            const unsigned store = value < model->info.codebook_size ? value : 0u;
            const size_t fs = stream / model->info.codebook_count;
            const size_t codebook = stream % model->info.codebook_count;
            it->codes[codebook * it->code_stride + it->raw_offset + fs] = store;
            const float *audio_table_data = NULL;
            if (projection_cache != NULL && stream < projection_cache->stream_count) {
                audio_table_data = projection_cache->audio_embeddings[stream];
            } else {
                snprintf(name, sizeof(name), "audio_embeddings.%zu.weight", stream);
                mynah_tensor audio_table;
                if (mynah_tensor_get(model->tts, name, &audio_table, error, error_capacity) != 0)
                    return -1;
                audio_table_data = audio_table.data;
            }
            /* The embedding of this stream's token becomes the input row for
             * the next local-transformer position. */
            if (gpu_local) {
                if (mynah_backend_h2d(c0->dev_backend,
                                      audio_table_data + (size_t)emit * width,
                                      c0->dev_input, width, error, error_capacity) != 0) {
                    return -1;
                }
            } else {
                memcpy(it->state->row_in, audio_table_data + (size_t)emit * width,
                       width * sizeof(float));
            }
        }
    }
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
        mynah_graph_error(error, error_capacity, "GPU attention cache dimensions overflow");
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
    if (mynah_tensor_get(model->tts, "decoder.position_embeddings.weight", &position, error, error_capacity) != 0 ||
        mynah_tensor_get(model->tts, "decoder.layers.0.cross_attention.q_net.weight", &q0, error, error_capacity) != 0) {
        return -1;
    }
    cache->position = position.data;
    cache->xattn_width = q0.shape[0];
    const size_t xw = cache->xattn_width;

    cache->self_k = mynah_alloc_floats(cache->layers * capacity * width, error, error_capacity);
    cache->self_v = mynah_alloc_floats(cache->layers * capacity * width, error, error_capacity);
    cache->cross_k = mynah_alloc_floats(cache->layers * memory_length * xw, error, error_capacity);
    cache->cross_v = mynah_alloc_floats(cache->layers * memory_length * xw, error, error_capacity);
    float *mem_norm = mynah_alloc_floats(memory_length * width, error, error_capacity);
    float *kv = mynah_alloc_floats(memory_length * xw * 2u, error, error_capacity);
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
        if (mynah_tensor_get(model->tts, name, &norm_memory, error, error_capacity) != 0) {
            free(mem_norm);
            free(kv);
            decoder_cache_free(cache);
            return -1;
        }
        layer_norm(memory, mem_norm, memory_length, width, norm_memory.data);
        snprintf(name, sizeof(name), "decoder.layers.%zu.cross_attention.kv_net.weight", layer);
        if (mynah_tensor_get(model->tts, name, &kv_net, error, error_capacity) != 0 ||
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
        mynah_graph_error(error, error_capacity, "out of memory resolving decoder weights");
        return -1;
    }
    for (size_t layer = 0; layer < cache->layers; ++layer) {
        decoder_layer_resolved *r = &cache->resolved[layer];
        mynah_tensor t;
        snprintf(name, sizeof(name), "decoder.layers.%zu.norm_self.weight", layer);
        if (mynah_tensor_get(model->tts, name, &t, error, error_capacity) != 0) { decoder_cache_free(cache); return -1; }
        r->norm_self = t.data;
        snprintf(name, sizeof(name), "decoder.layers.%zu.norm_xattn_query.weight", layer);
        if (mynah_tensor_get(model->tts, name, &t, error, error_capacity) != 0) { decoder_cache_free(cache); return -1; }
        r->norm_xattn_query = t.data;
        snprintf(name, sizeof(name), "decoder.layers.%zu.norm_pos_ff.weight", layer);
        if (mynah_tensor_get(model->tts, name, &t, error, error_capacity) != 0) { decoder_cache_free(cache); return -1; }
        r->norm_pos_ff = t.data;
        snprintf(r->qkv, sizeof(r->qkv), "decoder.layers.%zu.self_attention.qkv_net.weight", layer);
        snprintf(r->o_self, sizeof(r->o_self), "decoder.layers.%zu.self_attention.o_net.weight", layer);
        snprintf(r->q_cross, sizeof(r->q_cross), "decoder.layers.%zu.cross_attention.q_net.weight", layer);
        snprintf(r->o_cross, sizeof(r->o_cross), "decoder.layers.%zu.cross_attention.o_net.weight", layer);
        snprintf(r->ffn_up, sizeof(r->ffn_up), "decoder.layers.%zu.pos_ff.proj.conv.weight", layer);
        snprintf(r->ffn_down, sizeof(r->ffn_down), "decoder.layers.%zu.pos_ff.o_net.conv.weight", layer);
        /* Pre-resolve weight data pointers for direct matmul. */
        if (mynah_tensor_get(model->tts, r->qkv, &t, error, error_capacity) != 0) { decoder_cache_free(cache); return -1; }
        r->qkv_w = t.data;
        if (mynah_tensor_get(model->tts, r->o_self, &t, error, error_capacity) != 0) { decoder_cache_free(cache); return -1; }
        r->o_self_w = t.data;
        if (mynah_tensor_get(model->tts, r->q_cross, &t, error, error_capacity) != 0) { decoder_cache_free(cache); return -1; }
        r->q_cross_w = t.data;
        if (mynah_tensor_get(model->tts, r->o_cross, &t, error, error_capacity) != 0) { decoder_cache_free(cache); return -1; }
        r->o_cross_w = t.data;
        if (mynah_tensor_get(model->tts, r->ffn_up, &t, error, error_capacity) != 0) { decoder_cache_free(cache); return -1; }
        r->ffn_up_w = t.data;
        if (mynah_tensor_get(model->tts, r->ffn_down, &t, error, error_capacity) != 0) { decoder_cache_free(cache); return -1; }
        r->ffn_down_w = t.data;
    }
    {
        mynah_tensor t;
        if (mynah_tensor_get(model->tts, "decoder.norm_out.weight", &t, error, error_capacity) != 0) {
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
    cache->scratch_x = mynah_alloc_floats(rows * width, error, error_capacity);
    cache->scratch_nrm = mynah_alloc_floats(rows * width, error, error_capacity);
    cache->scratch_qkv = mynah_alloc_floats(rows * width * 3u, error, error_capacity);
    cache->scratch_attn = mynah_alloc_floats(rows * width, error, error_capacity);
    cache->scratch_proj = mynah_alloc_floats(rows * width, error, error_capacity);
    cache->scratch_q_x = mynah_alloc_floats(rows * xw, error, error_capacity);
    cache->scratch_xctx = mynah_alloc_floats(rows * xw, error, error_capacity);
    cache->scratch_hidden = mynah_alloc_floats(rows * cache->ffn_width, error, error_capacity);
    cache->scratch_scores = mynah_alloc_floats(scores_len, error, error_capacity);
    cache->scratch_gelu = mynah_alloc_floats(rows * cache->ffn_width, error, error_capacity);
    cache->scratch_score_matrix = mynah_alloc_floats(rows * capacity, error, error_capacity);
    cache->scratch_head_ctx = mynah_alloc_floats(rows * cache->head_width, error, error_capacity);
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

/* One autoregressive decoder step for `batch` independent requests.
 *
 * Every projection is a pass over a weight far larger than any cache, so B
 * requests stepping one after another pay B trips to DRAM for the same bytes.
 * Here the projections are batched -- one pass serving all slots -- while
 * everything that touches a slot's own state stays per slot.
 *
 * Attention deliberately does *not* batch: it reads each slot's private KV
 * cache, which no other slot shares, so there is nothing to amortize.  Keeping
 * it per slot also means slots at different positions need no ragged masking --
 * each simply loops over its own length, exactly as a lone request does.
 *
 * At batch 1 this reduces to the single-request path bit for bit, which is what
 * makes it safe to use for both. */
static int decoder_step_batch(const mynah_tts_model *model,
                              decoder_cache *const *caches,
                              const float *const *inputs, float *const *outs,
                              size_t batch, const batch_scratch *scratch,
                              char *error, size_t error_capacity) {
    if (batch == 0u) return 0;
    if (batch > MYNAH_MAX_BATCH) {
        mynah_graph_error(error, error_capacity, "decoder batch exceeds MYNAH_MAX_BATCH");
        return -1;
    }
    const decoder_cache *c0 = caches[0];
    const size_t width = c0->width;
    const size_t heads = c0->heads;
    const size_t hw = c0->head_width;
    const size_t ffn = c0->ffn_width;
    const size_t xw = c0->xattn_width;
    const float self_scale = 1.0f / sqrtf((float)hw);
    const float cross_scale = 1.0f / sqrtf((float)xw);
    int8_t *qx = scratch == NULL ? NULL : scratch->qx;
    float *sx = scratch == NULL ? NULL : scratch->sx;
    const float *in_ptrs[MYNAH_MAX_BATCH];
    float *out_ptrs[MYNAH_MAX_BATCH];

    for (size_t b = 0; b < batch; ++b) {
        decoder_cache *c = caches[b];
        if (c->length >= c->capacity) {
            mynah_graph_error(error, error_capacity, "decoder cache capacity exceeded");
            return -1;
        }
        const float *pe = c->position + c->length * width;
        for (size_t d = 0; d < width; ++d) c->scratch_x[d] = inputs[b][d] + pe[d];
    }
    /* All slots run the same model, so the resolved weights of any of them
     * name the same tensors at the same addresses. */
    for (size_t layer = 0; layer < c0->layers; ++layer) {
        const decoder_layer_resolved *r = &c0->resolved[layer];
        for (size_t b = 0; b < batch; ++b) {
            layer_norm(caches[b]->scratch_x, caches[b]->scratch_nrm, 1u, width,
                       r->norm_self);
            in_ptrs[b] = caches[b]->scratch_nrm;
            out_ptrs[b] = caches[b]->scratch_qkv;
        }
        if (mynah_qmat_linear_batched(model->qcache, model->backend, r->qkv, r->qkv_w,
                                      in_ptrs, out_ptrs, batch, width, width * 3u,
                                      NULL, qx, sx, error, error_capacity) != 0) return -1;
        for (size_t b = 0; b < batch; ++b) {
            decoder_cache *c = caches[b];
            const size_t start = c->length;
            const float *qkv = c->scratch_qkv;
            float *kbase = c->self_k + layer * c->capacity * width;
            float *vbase = c->self_v + layer * c->capacity * width;
            float *scores = c->scratch_scores;
            float *attn = c->scratch_attn;
            memcpy(kbase + start * width, qkv + width, width * sizeof(float));
            memcpy(vbase + start * width, qkv + width * 2u, width * sizeof(float));
            for (size_t h = 0; h < heads; ++h) {
                const float *qh = qkv + h * hw;
                float maxv = -FLT_MAX;
                for (size_t s = 0; s <= start; ++s) {
                    const float *k = kbase + s * width + h * hw;
                    float sc = 0.0f;
                    for (size_t d = 0; d < hw; ++d) sc += qh[d] * k[d];
                    sc *= self_scale;
                    scores[s] = sc;
                    if (sc > maxv) maxv = sc;
                }
                float denom = 0.0f;
                for (size_t s = 0; s <= start; ++s) {
                    scores[s] = expf(scores[s] - maxv);
                    denom += scores[s];
                }
                float *outh = attn + h * hw;
                memset(outh, 0, hw * sizeof(float));
                for (size_t s = 0; s <= start; ++s)
                    mynah_axpy_f32(outh, vbase + s * width + h * hw, scores[s] / denom, hw);
            }
            in_ptrs[b] = attn;
            out_ptrs[b] = c->scratch_proj;
        }
        if (mynah_qmat_linear_batched(model->qcache, model->backend, r->o_self,
                                      r->o_self_w, in_ptrs, out_ptrs, batch,
                                      width, width, NULL, qx, sx,
                                      error, error_capacity) != 0) return -1;
        for (size_t b = 0; b < batch; ++b) {
            decoder_cache *c = caches[b];
            for (size_t d = 0; d < width; ++d) c->scratch_x[d] += c->scratch_proj[d];
            layer_norm(c->scratch_x, c->scratch_nrm, 1u, width, r->norm_xattn_query);
            in_ptrs[b] = c->scratch_nrm;
            out_ptrs[b] = c->scratch_q_x;
        }
        if (mynah_qmat_linear_batched(model->qcache, model->backend, r->q_cross,
                                      r->q_cross_w, in_ptrs, out_ptrs, batch,
                                      width, xw, NULL, qx, sx,
                                      error, error_capacity) != 0) return -1;
        for (size_t b = 0; b < batch; ++b) {
            decoder_cache *c = caches[b];
            const float *ck = c->cross_k + layer * c->memory_length * xw;
            const float *cv = c->cross_v + layer * c->memory_length * xw;
            const float *qh = c->scratch_q_x;
            float *scores = c->scratch_scores;
            float *outh = c->scratch_xctx;
            float maxv = -FLT_MAX;
            for (size_t s = 0; s < c->memory_length; ++s) {
                const float *k = ck + s * xw;
                float sc = 0.0f;
                for (size_t d = 0; d < xw; ++d) sc += qh[d] * k[d];
                sc *= cross_scale;
                scores[s] = sc;
                if (sc > maxv) maxv = sc;
            }
            float denom = 0.0f;
            for (size_t s = 0; s < c->memory_length; ++s) {
                scores[s] = expf(scores[s] - maxv);
                denom += scores[s];
            }
            memset(outh, 0, xw * sizeof(float));
            for (size_t s = 0; s < c->memory_length; ++s)
                mynah_axpy_f32(outh, cv + s * xw, scores[s] / denom, xw);
            in_ptrs[b] = outh;
            out_ptrs[b] = c->scratch_proj;
        }
        if (mynah_qmat_linear_batched(model->qcache, model->backend, r->o_cross,
                                      r->o_cross_w, in_ptrs, out_ptrs, batch,
                                      xw, width, NULL, qx, sx,
                                      error, error_capacity) != 0) return -1;
        for (size_t b = 0; b < batch; ++b) {
            decoder_cache *c = caches[b];
            for (size_t d = 0; d < width; ++d) c->scratch_x[d] += c->scratch_proj[d];
            layer_norm(c->scratch_x, c->scratch_nrm, 1u, width, r->norm_pos_ff);
            in_ptrs[b] = c->scratch_nrm;
            out_ptrs[b] = c->scratch_hidden;
        }
        if (mynah_qmat_linear_batched(model->qcache, model->backend, r->ffn_up,
                                      r->ffn_up_w, in_ptrs, out_ptrs, batch,
                                      width, ffn, NULL, qx, sx,
                                      error, error_capacity) != 0) return -1;
        for (size_t b = 0; b < batch; ++b) {
            decoder_cache *c = caches[b];
            mynah_gelu_f32_scalar(c->scratch_hidden, ffn);
            in_ptrs[b] = c->scratch_hidden;
            out_ptrs[b] = c->scratch_proj;
        }
        if (mynah_qmat_linear_batched(model->qcache, model->backend, r->ffn_down,
                                      r->ffn_down_w, in_ptrs, out_ptrs, batch,
                                      ffn, width, NULL, qx, sx,
                                      error, error_capacity) != 0) return -1;
        for (size_t b = 0; b < batch; ++b) {
            decoder_cache *c = caches[b];
            for (size_t d = 0; d < width; ++d) c->scratch_x[d] += c->scratch_proj[d];
        }
    }
    for (size_t b = 0; b < batch; ++b) {
        decoder_cache *c = caches[b];
        layer_norm(c->scratch_x, outs[b], 1u, width, c->norm_out);
        c->length += 1u;
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
        mynah_graph_error(error, error_capacity, "decoder cache capacity exceeded");
        return -1;
    }
    if (count > cache->scratch_rows) {
        mynah_graph_error(error, error_capacity, "decoder scratch too small for count");
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
                mynah_gelu_tanh_array(hidden, ffn, gelu_scratch);
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

    /* A single decode row is the batched step with one slot.  Routing it here
     * rather than duplicating the body means the batched path is exercised by
     * every existing test, and any drift between the two would show up as an
     * audio change at batch 1. */
    if (count == 1u) {
        decoder_cache *one_cache[1] = { cache };
        const float *one_in[1] = { input_rows };
        float *one_out[1] = { out_last };
        return decoder_step_batch(model, one_cache, one_in, one_out, 1u, NULL,
                                  error, error_capacity);
    }


    for (size_t layer = 0; layer < cache->layers && !failed; ++layer) {
        const decoder_layer_resolved *r = &cache->resolved[layer];
        double operation_start = profile_prefill ? mynah_phase_seconds() : 0.0;
        /* self-attention */
        layer_norm(x, nrm, count, width, r->norm_self);
        if (mynah_qmat_linear_resolved(model->qcache, model->backend,
                                       r->qkv, r->qkv_w,
                                       nrm, qkv, count, width, width * 3u, NULL,
                                       error, error_capacity) != 0) { failed = 1; break; }
        if (profile_prefill) {
            self_projection_seconds += mynah_phase_seconds() - operation_start;
            operation_start = mynah_phase_seconds();
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
                mynah_graph_sgemm(backend, 0, 1, (int)count, (int)total_kv, (int)hw, self_scale, qkv + h * hw, (int)(width * 3u), kbase + h * hw, (int)width, 0.0f, score_matrix, (int)total_kv, error, error_capacity);
                for (size_t i = 0; i < count; ++i) {
                    const size_t valid = start + i + 1u;
                    float *row = score_matrix + i * total_kv;
                    softmax_row_inplace(row, valid);
                    for (size_t s = valid; s < total_kv; ++s) row[s] = 0.0f;
                }
                mynah_graph_sgemm(backend, 0, 0, (int)count, (int)hw, (int)total_kv, 1.0f, score_matrix, (int)total_kv, vbase + h * hw, (int)width, 0.0f, head_ctx, (int)hw, error, error_capacity);
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
                    mynah_axpy_f32(outh, vbase + s * width + h * hw, scores[s] / denom, hw);
            }
        }
        if (profile_prefill) {
            self_attention_seconds += mynah_phase_seconds() - operation_start;
            operation_start = mynah_phase_seconds();
        }
        if (layer == 0 && getenv("MYNAH_DUMP_CPU_SELF_ATTN") != NULL) {
            FILE *dump = fopen(getenv("MYNAH_DUMP_CPU_SELF_ATTN"), "w");
            if (dump != NULL) {
                for (size_t d = 0; d < width; ++d) fprintf(dump, "%.9g\n", (double)attn[d]);
                fclose(dump);
            }
        }
        if (mynah_qmat_linear_resolved(model->qcache, model->backend,
                                       r->o_self, r->o_self_w,
                                       attn, proj, count, width, width, NULL,
                                       error, error_capacity) != 0) { failed = 1; break; }
        for (size_t k = 0; k < count * width; ++k) x[k] += proj[k];
        /* cross-attention over cached text memory */
        layer_norm(x, nrm, count, width, r->norm_xattn_query);
        if (mynah_qmat_linear_resolved(model->qcache, model->backend,
                                       r->q_cross, r->q_cross_w,
                                       nrm, q_x, count, width, xw, NULL,
                                       error, error_capacity) != 0) { failed = 1; break; }
        if (profile_prefill) {
            cross_projection_seconds += mynah_phase_seconds() - operation_start;
            operation_start = mynah_phase_seconds();
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
                mynah_axpy_f32(outh, cv + s * xw, scores[s] / denom, xw);
        }
        if (profile_prefill) {
            cross_attention_seconds += mynah_phase_seconds() - operation_start;
            operation_start = mynah_phase_seconds();
        }
        if (mynah_qmat_linear_resolved(model->qcache, model->backend,
                                       r->o_cross, r->o_cross_w,
                                       xctx, proj, count, xw, width, NULL,
                                       error, error_capacity) != 0) { failed = 1; break; }
        for (size_t k = 0; k < count * width; ++k) x[k] += proj[k];
        /* position-wise FFN (kernel size 1) */
        layer_norm(x, nrm, count, width, r->norm_pos_ff);
        if (mynah_qmat_linear_resolved(model->qcache, model->backend,
                                       r->ffn_up, r->ffn_up_w,
                                       nrm, hidden, count, width, ffn, NULL,
                                       error, error_capacity) != 0) { failed = 1; break; }
        if (getenv("MYNAH_GELU_SCALAR") == NULL && count > 1u) {
            mynah_gelu_f32(hidden, hidden_elements);
        } else {
            mynah_gelu_f32_scalar(hidden, hidden_elements);
        }
        if (mynah_qmat_linear_resolved(model->qcache, model->backend,
                                       r->ffn_down, r->ffn_down_w,
                                       hidden, proj, count, ffn, width, NULL,
                                       error, error_capacity) != 0) { failed = 1; break; }
        for (size_t k = 0; k < count * width; ++k) x[k] += proj[k];
        if (profile_prefill) ffn_seconds += mynah_phase_seconds() - operation_start;
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
            mynah_graph_error(error, error_capacity, "audio callback aborted streaming");
            return -1;
        }
        offset += chunk;
    }
    return 0;
}

/* One request in flight.
 *
 * Everything a request needs to advance one step lives here, so the driver can
 * hold several and step them together.  Nothing is shared between slots except
 * the read-only model, which is what makes the batched projections safe. */
typedef struct {
    /* request and sinks */
    const mynah_tts_request *request;
    float **samples;
    size_t *sample_count;
    mynah_tts_audio_callback callback;
    void *user_data;
    size_t chunk_samples;
    char *error;
    size_t error_capacity;
    /* owned state */
    float *memory;
    unsigned *codes;
    float *out_last;
    float *audio_row;
    decoder_cache cache;
    local_frame_state local_state;
    float *decoder_dev_last;
    /* derived constants */
    const float *context;
    size_t context_length;
    size_t max_steps;
    size_t max_raw_length;
    size_t min_raw_length;
    float temperature;
    unsigned topk;
    /* progress */
    uint64_t rng_state;
    size_t step;              /* 1-based stacked position of the next step */
    size_t predicted_stacks;
    size_t eos_frame;
    size_t streamed_samples;
    size_t streamed_frames;
    int active;
    int failed;
} synth_slot;

static void slot_release(synth_slot *slot) {
    if (slot == NULL) return;
    local_frame_state_free(&slot->local_state);
    decoder_cache_free(&slot->cache);
    free(slot->memory);
    free(slot->codes);
    free(slot->out_last);
    free(slot->audio_row);
    slot->memory = NULL;
    slot->codes = NULL;
    slot->out_last = NULL;
    slot->audio_row = NULL;
}

static int slot_fail(synth_slot *slot, const char *message) {
    if (message != NULL) mynah_graph_error(slot->error, slot->error_capacity, message);
    slot->failed = 1;
    slot->active = 0;
    return -1;
}

/* Everything before the first autoregressive step: text encoding, the constant
 * cross-attention cache, and the baked speaker context prefill.  These are
 * multi-row matmuls that BLAS already runs efficiently and that share nothing
 * between requests, so they stay per slot. */
static int slot_prepare(const mynah_tts_model *model, synth_slot *slot, int dump) {
    const mynah_tts_request *request = slot->request;
    char *error = slot->error;
    const size_t error_capacity = slot->error_capacity;
    const size_t width = model->info.hidden_dim;

    if (slot->samples != NULL) *slot->samples = NULL;
    if (slot->sample_count != NULL) *slot->sample_count = 0;
    if (request == NULL ||
        ((slot->samples == NULL || slot->sample_count == NULL) && slot->callback == NULL) ||
        error == NULL || error_capacity == 0 || request->text_ids == NULL ||
        request->text_length == 0 || (slot->callback != NULL && slot->chunk_samples == 0)) {
        return slot_fail(slot, "invalid synthesis request");
    }
    if (request->speaker >= model->info.speaker_count) {
        return slot_fail(slot, "speaker index is outside the model");
    }
    if (encode_text(model, request->text_ids, request->text_length, &slot->memory,
                    error, error_capacity) != 0) {
        return slot_fail(slot, NULL);
    }
    if (dump && getenv("MYNAH_DUMP_ENCODER") != NULL) {
        FILE *ef = fopen(getenv("MYNAH_DUMP_ENCODER"), "w");
        if (ef != NULL) {
            for (size_t t = 0; t < request->text_length; ++t)
                for (size_t d = 0; d < width; ++d)
                    fprintf(ef, "%.9g\n", (double)slot->memory[t * width + d]);
            fclose(ef);
        }
    }
    mynah_tensor context_tensor;
    if (mynah_tensor_get(model->tts, "baked_context_embedding.weight", &context_tensor,
               error, error_capacity) != 0 || context_tensor.rank != 2 ||
        context_tensor.shape[1] % width != 0 ||
        request->speaker >= context_tensor.shape[0]) {
        return slot_fail(slot, "baked speaker context is invalid");
    }
    slot->context_length = context_tensor.shape[1] / width;
    slot->context = context_tensor.data +
                    (size_t)request->speaker * context_tensor.shape[1];
    slot->max_steps = request->max_steps == 0
        ? (model->info.max_decoder_steps + model->info.frame_stacking_factor - 1u) /
          model->info.frame_stacking_factor
        : request->max_steps;
    slot->max_raw_length = (slot->max_steps + 1u) * model->info.frame_stacking_factor;
    slot->min_raw_length = model->info.min_generated_frames;
    slot->codes = (unsigned *)calloc(model->info.codebook_count * slot->max_raw_length,
                                     sizeof(*slot->codes));
    slot->out_last = mynah_alloc_floats(width, error, error_capacity);
    slot->audio_row = mynah_alloc_floats(width, error, error_capacity);
    if (slot->codes == NULL || slot->out_last == NULL || slot->audio_row == NULL) {
        return slot_fail(slot, "out of memory preparing the request");
    }
    for (size_t c = 0; c < model->info.codebook_count; ++c) {
        for (size_t t = 0; t < model->info.frame_stacking_factor; ++t) {
            slot->codes[c * slot->max_raw_length + t] = model->info.codebook_size;
        }
    }
    slot->temperature = request->temperature;
    if (!(slot->temperature >= 0.0f)) slot->temperature = model->info.default_temperature;
    slot->topk = request->topk == 0 ? model->info.default_topk : request->topk;
    slot->rng_state = request->seed == 0 ? UINT64_C(0x9e3779b97f4a7c15) : request->seed;
    slot->eos_frame = SIZE_MAX;
    if (decoder_cache_init(model, &slot->cache, slot->memory, request->text_length,
                           slot->context_length + slot->max_steps + 2u,
                           error, error_capacity) != 0) {
        return slot_fail(slot, NULL);
    }
    slot->decoder_dev_last = NULL;
    if (decoder_run(model, &slot->cache, slot->context, slot->context_length,
                    slot->out_last,
                    request->use_local_transformer ? &slot->decoder_dev_last : NULL,
                    error, error_capacity) != 0) {
        return slot_fail(slot, NULL);
    }
    if (dump && getenv("MYNAH_DUMP_PREFILL") != NULL) {
        FILE *pf = fopen(getenv("MYNAH_DUMP_PREFILL"), "w");
        if (pf != NULL) {
            if (slot->decoder_dev_last != NULL)
                mynah_backend_d2h(model->backend, slot->decoder_dev_last, slot->out_last,
                                  width, error, error_capacity);
            for (size_t d = 0; d < width; ++d)
                fprintf(pf, "%.9g\n", (double)slot->out_last[d]);
            fclose(pf);
        }
    }
    if (request->use_local_transformer) {
        const size_t vocab = model->info.audio_vocab_size;
        const size_t top_count = slot->temperature > 0.0f && slot->topk > 1u
            ? (slot->topk < vocab ? slot->topk : vocab) : 0u;
        if (local_frame_state_init(model, &slot->local_state, top_count,
                                   error, error_capacity) != 0) {
            return slot_fail(slot, NULL);
        }
    }
    slot->step = 1u;
    slot->active = 1;
    return 0;
}

/* The decoder row is produced; dump it if asked. */
static void slot_dump_hidden(const mynah_tts_model *model, synth_slot *slot, int dump) {
    const size_t width = model->info.hidden_dim;
    if (!dump || slot->step != 1u || getenv("MYNAH_DUMP_HIDDEN") == NULL) return;
    FILE *hf = fopen(getenv("MYNAH_DUMP_HIDDEN"), "w");
    if (hf == NULL) return;
    if (slot->decoder_dev_last != NULL)
        mynah_backend_d2h(model->backend, slot->decoder_dev_last, slot->out_last,
                          width, slot->error, slot->error_capacity);
    for (size_t d = 0; d < width; ++d)
        fprintf(hf, "%.9g\n", (double)slot->out_last[d]);
    fclose(hf);
}

/* Streaming, the EOS decision and the step counter for one slot, once its
 * frame has been sampled.  Clears `active` when the request is finished. */
static int slot_advance(const mynah_tts_model *model, synth_slot *slot,
                        int saw_eos, size_t step_eos_frame) {
    const mynah_tts_request *request = slot->request;
    char *error = slot->error;
    const size_t error_capacity = slot->error_capacity;
    const size_t width = model->info.hidden_dim;
    const size_t raw_length = slot->step * model->info.frame_stacking_factor;
    if (!request->use_local_transformer) {
        mynah_tensor projection;
        mynah_tensor bias;
        if (mynah_tensor_get(model->tts, "final_proj.weight", &projection, error, error_capacity) != 0 ||
            mynah_tensor_get(model->tts, "final_proj.bias", &bias, error, error_capacity) != 0) {
            return slot_fail(slot, NULL);
        }
        const size_t streams = stacked_stream_count(model);
        const int forbid_eos =
            slot->predicted_stacks * model->info.frame_stacking_factor < slot->min_raw_length;
        const unsigned eos_id = model->info.audio_eos_id;
        const size_t vocab = model->info.audio_vocab_size;
        const float *last = slot->out_last;
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
                const float *row = projection.data +
                    (stream * model->info.audio_vocab_size + candidate) * width;
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
            slot->codes[codebook * slot->max_raw_length + raw_length + fs] = store;
        }
    }
    ++slot->predicted_stacks;
    if (slot->callback != NULL) {
        size_t stream_raw = slot->predicted_stacks * model->info.frame_stacking_factor;
        if (step_eos_frame != SIZE_MAX) {
            stream_raw = (slot->predicted_stacks - 1u) * model->info.frame_stacking_factor +
                         step_eos_frame;
        }
        const int finishing = saw_eos || slot->step >= slot->max_steps;
        const size_t fresh = stream_raw > slot->streamed_frames
            ? stream_raw - slot->streamed_frames : 0u;
        if (fresh > 0u && (finishing || fresh >= STREAM_EMIT_FRAMES)) {
            /* Decode a bounded suffix, not the whole prefix.
             *
             * The codec decoder is causal throughout, so an output depends only
             * on inputs at or before it.  Restarting the sequence
             * STREAM_CONTEXT_FRAMES before the first unsent frame therefore
             * reproduces those frames exactly -- only the outputs inside the
             * receptive field are affected by the missing history, and those
             * have already been sent.
             *
             * Re-decoding the whole prefix every step, as this did, is
             * quadratic: a 15 s utterance decoded 26k frames instead of 320 and
             * streaming ran 30x slower than the same request offline. */
            const size_t start = slot->streamed_frames > STREAM_CONTEXT_FRAMES
                ? slot->streamed_frames - STREAM_CONTEXT_FRAMES : 0u;
            const size_t span = stream_raw - start;
            unsigned *stream_codes = (unsigned *)calloc(
                model->info.codebook_count * span, sizeof(*stream_codes));
            if (stream_codes == NULL) {
                return slot_fail(slot, "out of memory preparing streamed codes");
            }
            for (size_t c = 0; c < model->info.codebook_count; ++c) {
                memcpy(stream_codes + c * span,
                       slot->codes + c * slot->max_raw_length +
                           model->info.frame_stacking_factor + start,
                       span * sizeof(*stream_codes));
            }
            float *stream_audio = NULL;
            size_t stream_count = 0;
            if (mynah_nanocodec_decode(model, stream_codes, span, &stream_audio,
                             &stream_count, error, error_capacity) != 0) {
                free(stream_codes);
                return slot_fail(slot, NULL);
            }
            /* The codec's samples-per-frame is a property of the pack, so take
             * it from what it produced rather than assuming a rate. */
            const size_t skip_frames = slot->streamed_frames - start;
            const size_t per_frame = span > 0u ? stream_count / span : 0u;
            const size_t skip = skip_frames * per_frame;
            if (stream_count < skip ||
                emit_stream_samples(slot->callback, slot->user_data,
                                    stream_audio + skip, stream_count - skip,
                                    slot->chunk_samples, error, error_capacity) != 0) {
                const int regressed = stream_count < skip;
                free(stream_audio);
                free(stream_codes);
                return slot_fail(slot, regressed ? "streamed codec output regressed" : NULL);
            }
            slot->streamed_samples += stream_count - skip;
            slot->streamed_frames = stream_raw;
            free(stream_audio);
            free(stream_codes);
        }
    }
    if (step_eos_frame != SIZE_MAX) slot->eos_frame = step_eos_frame;
    if (saw_eos && slot->predicted_stacks >= 4u) {
        slot->active = 0;
        return 0;
    }
    ++slot->step;
    if (slot->step > slot->max_steps) slot->active = 0;
    return 0;
}

/* Turn the accumulated codes into audio for the offline sink. */
static int slot_finalize(const mynah_tts_model *model, synth_slot *slot, int dump) {
    char *error = slot->error;
    const size_t error_capacity = slot->error_capacity;
    const size_t generated_stacks = slot->predicted_stacks;
    size_t generated_raw = generated_stacks * model->info.frame_stacking_factor;
    if (slot->eos_frame != SIZE_MAX && generated_stacks > 0) {
        generated_raw = (generated_stacks - 1u) * model->info.frame_stacking_factor +
                        slot->eos_frame;
    }
    if (dump && getenv("MYNAH_DUMP_CODES") != NULL && generated_stacks > 0) {
        FILE *dumpf = fopen(getenv("MYNAH_DUMP_CODES"), "w");
        if (dumpf != NULL) {
            const size_t cb = model->info.codebook_count;
            const size_t fs = model->info.frame_stacking_factor;
            fprintf(dumpf, "[");
            for (size_t step = 0; step < generated_stacks; ++step) {
                if (step > 0) fprintf(dumpf, ",");
                fprintf(dumpf, "[[");
                for (size_t c = 0; c < cb; ++c) {
                    if (c > 0) fprintf(dumpf, "],[");
                    for (size_t f = 0; f < fs; ++f) {
                        if (f > 0) fprintf(dumpf, ",");
                        fprintf(dumpf, "%u",
                                slot->codes[c * slot->max_raw_length + (step + 1u) * fs + f]);
                    }
                }
                fprintf(dumpf, "]]");
            }
            fprintf(dumpf, "]\n");
            fclose(dumpf);
        }
    }
    if (slot->failed) return -1;
    if (generated_raw == 0) {
        return slot_fail(slot, "decoder generated no audio frames");
    }
    if (slot->callback != NULL) {
        if (slot->streamed_samples == 0) {
            return slot_fail(slot, "stream produced no audio frames");
        }
        return 0;
    }
    unsigned *predicted_codes = (unsigned *)calloc(
        model->info.codebook_count * generated_raw, sizeof(*predicted_codes));
    if (predicted_codes == NULL) {
        return slot_fail(slot, "out of memory copying generated codes");
    }
    for (size_t c = 0; c < model->info.codebook_count; ++c) {
        memcpy(predicted_codes + c * generated_raw,
               slot->codes + c * slot->max_raw_length + model->info.frame_stacking_factor,
               generated_raw * sizeof(*predicted_codes));
    }
    const int result = mynah_nanocodec_decode(model, predicted_codes, generated_raw,
                                    slot->samples, slot->sample_count,
                                    error, error_capacity);
    free(predicted_codes);
    if (result != 0) return slot_fail(slot, NULL);
    return 0;
}

/* Step every live slot together.
 *
 * This is continuous in the sense that matters here: a slot that reaches EOS
 * drops out of the batch immediately and the rest keep going at the smaller
 * width, rather than the whole group waiting for the longest request. */
static int synthesize_slots(const mynah_tts_model *model, synth_slot *slots,
                            size_t count) {
    if (count == 0u) return 0;
    if (count > MYNAH_MAX_BATCH) return -1;
    const int dump_all = count == 1u;
    const int timing = getenv("MYNAH_TIMING") != NULL;
    const double t_start = timing ? mynah_phase_seconds() : 0.0;
    double t_prep = t_start, t_ar = t_start;

    for (size_t i = 0; i < count; ++i) {
        slot_prepare(model, &slots[i], dump_all);
    }
    if (timing) t_prep = mynah_phase_seconds();

    /* One quantized activation buffer for the widest projection in the graph. */
    batch_scratch scratch;
    size_t k_max = 0;
    for (size_t i = 0; i < count; ++i) {
        if (!slots[i].active) continue;
        const decoder_cache *c = &slots[i].cache;
        if (c->width > k_max) k_max = c->width;
        if (c->ffn_width > k_max) k_max = c->ffn_width;
        if (c->xattn_width > k_max) k_max = c->xattn_width;
    }
    char scratch_error[256];
    scratch_error[0] = '\0';
    if (batch_scratch_init(&scratch, count, k_max, scratch_error,
                           sizeof(scratch_error)) != 0) {
        for (size_t i = 0; i < count; ++i) {
            if (slots[i].active) slot_fail(&slots[i], scratch_error);
        }
    }

    decoder_cache *step_caches[MYNAH_MAX_BATCH];
    const float *step_inputs[MYNAH_MAX_BATCH];
    float *step_outs[MYNAH_MAX_BATCH];
    size_t step_slot[MYNAH_MAX_BATCH];
    local_batch_item local_items[MYNAH_MAX_BATCH];
    size_t local_slot[MYNAH_MAX_BATCH];
    for (;;) {
        size_t live = 0;
        for (size_t i = 0; i < count; ++i) {
            synth_slot *slot = &slots[i];
            if (!slot->active) continue;
            if (embed_audio_frame(model, slot->codes, slot->max_raw_length,
                                  slot->step - 1u, slot->audio_row,
                                  slot->error, slot->error_capacity) != 0) {
                slot_fail(slot, NULL);
                continue;
            }
            slot->decoder_dev_last = NULL;
            step_slot[live] = i;
            step_caches[live] = &slot->cache;
            step_inputs[live] = slot->audio_row;
            step_outs[live] = slot->out_last;
            ++live;
        }
        if (live == 0u) break;
        if (live == 1u) {
            /* Alone: go through decoder_run so a resident GPU step is still
             * available.  It reduces to the same batched step on CPU. */
            synth_slot *slot = &slots[step_slot[0]];
            if (decoder_run(model, &slot->cache, slot->audio_row, 1u, slot->out_last,
                            slot->request->use_local_transformer ? &slot->decoder_dev_last
                                                                 : NULL,
                            slot->error, slot->error_capacity) != 0) {
                slot_fail(slot, NULL);
                continue;
            }
        } else if (decoder_step_batch(model, step_caches, step_inputs, step_outs,
                                      live, &scratch, slots[step_slot[0]].error,
                                      slots[step_slot[0]].error_capacity) != 0) {
            /* A batched step fails for all its slots or none: the failure is in
             * shared code, not in one request's data. */
            for (size_t j = 0; j < live; ++j) {
                synth_slot *slot = &slots[step_slot[j]];
                if (j > 0) {
                    mynah_graph_error(slot->error, slot->error_capacity,
                                "batched decoder step failed");
                }
                slot_fail(slot, NULL);
            }
            continue;
        }
        /* Sample the frame.  The local transformer is read once per stacked
         * stream, so a decode step walks its weights sixteen times: batching it
         * across slots matters more than batching the decoder itself. */
        size_t local_count = 0;
        for (size_t j = 0; j < live; ++j) {
            synth_slot *slot = &slots[step_slot[j]];
            if (!slot->active) continue;
            slot_dump_hidden(model, slot, dump_all);
            if (!slot->request->use_local_transformer) continue;
            local_batch_item *item = &local_items[local_count];
            memset(item, 0, sizeof(*item));
            item->state = &slot->local_state;
            item->decoder_last = slot->out_last;
            item->decoder_dev_last = slot->decoder_dev_last;
            item->codes = slot->codes;
            item->raw_offset = slot->step * model->info.frame_stacking_factor;
            item->code_stride = slot->max_raw_length;
            item->generated_raw_length =
                slot->predicted_stacks * model->info.frame_stacking_factor;
            item->min_raw_length = slot->min_raw_length;
            item->temperature = slot->temperature;
            item->topk = slot->topk;
            item->rng_state = &slot->rng_state;
            local_slot[local_count] = step_slot[j];
            ++local_count;
        }
        if (local_count > 0u) {
            char local_error[256];
            local_error[0] = '\0';
            if (sample_local_frame_batch(model, local_items, local_count, &scratch,
                                         local_error, sizeof(local_error)) != 0) {
                for (size_t j = 0; j < local_count; ++j) {
                    slot_fail(&slots[local_slot[j]], local_error);
                }
                continue;
            }
        }
        for (size_t j = 0; j < live; ++j) {
            synth_slot *slot = &slots[step_slot[j]];
            if (!slot->active) continue;
            int saw_eos = 0;
            size_t step_eos_frame = SIZE_MAX;
            if (slot->request->use_local_transformer) {
                for (size_t q = 0; q < local_count; ++q) {
                    if (local_slot[q] != step_slot[j]) continue;
                    saw_eos = local_items[q].saw_eos;
                    step_eos_frame = local_items[q].eos_frame;
                    break;
                }
            }
            slot_advance(model, slot, saw_eos, step_eos_frame);
        }
    }
    if (timing) t_ar = mynah_phase_seconds();

    int result = 0;
    for (size_t i = 0; i < count; ++i) {
        if (slot_finalize(model, &slots[i], dump_all) != 0) result = -1;
    }
    if (timing) {
        fprintf(stderr, "phase: prep=%.3fs ar=%.3fs codec=%.3fs (requests=%zu)\n",
                t_prep - t_start, t_ar - t_prep, mynah_phase_seconds() - t_ar, count);
    }
    batch_scratch_free(&scratch);
    for (size_t i = 0; i < count; ++i) {
        slot_release(&slots[i]);
        if (!slots[i].failed && slots[i].error != NULL && slots[i].error_capacity > 0)
            slots[i].error[0] = '\0';
    }
    return result;
}

int mynah_graph_synthesize_jobs(const mynah_tts_model *model,
                                mynah_graph_job *jobs, size_t count) {
    if (model == NULL || jobs == NULL) return -1;
    if (count == 0u) return 0;
    if (count > MYNAH_MAX_BATCH) return -1;
    synth_slot slots[MYNAH_MAX_BATCH];
    memset(slots, 0, sizeof(slots));
    for (size_t i = 0; i < count; ++i) {
        slots[i].request = jobs[i].request;
        slots[i].samples = jobs[i].samples;
        slots[i].sample_count = jobs[i].sample_count;
        slots[i].callback = jobs[i].callback;
        slots[i].user_data = jobs[i].user_data;
        slots[i].chunk_samples = jobs[i].chunk_samples;
        slots[i].error = jobs[i].error;
        slots[i].error_capacity = jobs[i].error_capacity;
        slots[i].eos_frame = SIZE_MAX;
    }
    const int result = synthesize_slots(model, slots, count);
    for (size_t i = 0; i < count; ++i) jobs[i].result = slots[i].failed ? -1 : 0;
    return result;
}

int mynah_graph_synthesize_stream(const mynah_tts_model *model,
                                  const mynah_tts_request *request,
                                  float **samples, size_t *sample_count,
                                  mynah_tts_audio_callback callback,
                                  void *user_data, size_t chunk_samples,
                                  char *error, size_t error_capacity) {
    if (samples != NULL) *samples = NULL;
    if (sample_count != NULL) *sample_count = 0;
    if (model == NULL || error == NULL || error_capacity == 0) {
        mynah_graph_error(error, error_capacity, "invalid synthesis request");
        return -1;
    }
    mynah_graph_job job;
    memset(&job, 0, sizeof(job));
    job.request = request;
    job.samples = samples;
    job.sample_count = sample_count;
    job.callback = callback;
    job.user_data = user_data;
    job.chunk_samples = chunk_samples;
    job.error = error;
    job.error_capacity = error_capacity;
    return mynah_graph_synthesize_jobs(model, &job, 1u);
}

size_t mynah_tts_max_batch(void) {
    return MYNAH_GRAPH_MAX_JOBS;
}

int mynah_tts_synthesize_batch(const mynah_tts_model *model,
                               mynah_tts_batch_job *jobs, size_t count) {
    if (model == NULL || jobs == NULL) return -1;
    if (count == 0u) return 0;
    if (count > MYNAH_GRAPH_MAX_JOBS) return -1;
    mynah_graph_job graph_jobs[MYNAH_GRAPH_MAX_JOBS];
    memset(graph_jobs, 0, sizeof(graph_jobs));
    for (size_t i = 0; i < count; ++i) {
        graph_jobs[i].request = jobs[i].request;
        graph_jobs[i].samples = jobs[i].samples;
        graph_jobs[i].sample_count = jobs[i].sample_count;
        graph_jobs[i].error = jobs[i].error;
        graph_jobs[i].error_capacity = jobs[i].error_capacity;
    }
    const int result = mynah_graph_synthesize_jobs(model, graph_jobs, count);
    for (size_t i = 0; i < count; ++i) jobs[i].result = graph_jobs[i].result;
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
