#include "conv1d.h"

#include "kernels.h"
#include "mynah_util.h"
#include "threads.h"

#include <limits.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(MYNAH_USE_ACCELERATE)
#include <Accelerate/Accelerate.h>
#elif defined(MYNAH_USE_OPENBLAS)
#include <cblas.h>
#endif

/* Unfold a causal convolution window into rows: col[t][i * kernel + k] is the
 * channel i of the input at t - (kernel - 1) + k, zero before the start.  The
 * (i, k) ordering matches the [out][in][kernel] weight layout, so the unfolded
 * activation multiplies the stored weight directly. */
void mynah_unfold_causal(const float *input, float *col, size_t length,
                          size_t channels, size_t kernel) {
    const size_t unfolded = channels * kernel;
    for (size_t t = 0; t < length; ++t) {
        float *row = col + t * unfolded;
        for (size_t k = 0; k < kernel; ++k) {
            const long source_t = (long)t - (long)(kernel - 1u) + (long)k;
            if (source_t < 0) {
                for (size_t i = 0; i < channels; ++i) row[i * kernel + k] = 0.0f;
            } else {
                const float *in = input + (size_t)source_t * channels;
                for (size_t i = 0; i < channels; ++i) row[i * kernel + k] = in[i];
            }
        }
    }
}


#if defined(MYNAH_USE_ACCELERATE)
/* One filter per (shape, owning thread).  BNNSFilterApply's behaviour with the
 * same filter on two threads is undocumented, and a convolution filter may hold
 * internal workspace sized for its shape, so filters are not shared across
 * threads.  Giving each thread its own also lets the apply run outside the
 * cache lock: with a shared filter the whole convolution had to sit inside the
 * critical section, which serialized every codec call process-wide. */
typedef struct {
    const float *weight;
    const float *bias;
    size_t in_channels;
    size_t out_channels;
    size_t length;
    size_t kernel;
    size_t dilation;
    pthread_t owner;
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
    const pthread_t self = pthread_self();
    /* The lookup must hold the lock: the insert path below grows this array
     * with realloc, which moves it, so scanning it unlocked races with another
     * thread's growth. */
    if (cache != NULL) {
        pthread_mutex_lock(&cache->mutex);
        for (size_t i = 0; i < cache->count; ++i) {
            codec_bnns_entry *entry = &cache->entries[i];
            if (entry->weight == weight && entry->bias == bias &&
                entry->in_channels == in_channels && entry->out_channels == out_channels &&
                entry->length == length && entry->kernel == kernel &&
                entry->dilation == dilation &&
                pthread_equal(entry->owner, self)) {
                filter = entry->filter;
                retained = 1;
                break;
            }
        }
        pthread_mutex_unlock(&cache->mutex);
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
    double operation_start = profile != NULL ? mynah_phase_seconds() : 0.0;
    if (cache != NULL) pthread_mutex_lock(&cache->mutex);
    if (filter == NULL) {
        filter = BNNSFilterCreateLayerConvolution(&parameters, NULL);
        if (filter == NULL) {
            if (cache != NULL) pthread_mutex_unlock(&cache->mutex);
            return -1;
        }
        if (profile != NULL) {
            profile->bnns_create_seconds += mynah_phase_seconds() - operation_start;
            operation_start = mynah_phase_seconds();
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
                    weight, bias, in_channels, out_channels, length, kernel,
                    dilation, self, filter};
                retained = 1;
            }
        }
    }
    if (cache != NULL) pthread_mutex_unlock(&cache->mutex);

    /* Outside the lock: this filter belongs to this thread, either because it
     * was found under our own owner key or because we just created it and no
     * one else can reach it. */
    const int result = BNNSFilterApply(filter, input, output);
    if (profile != NULL) {
        profile->bnns_apply_seconds += mynah_phase_seconds() - operation_start;
        operation_start = mynah_phase_seconds();
    }
    if (!retained) {
        BNNSFilterDestroy(filter);
        if (profile != NULL) {
            profile->bnns_destroy_seconds += mynah_phase_seconds() - operation_start;
        }
    }
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
        mynah_graph_error(error, error_capacity, "BNNS causal-conv self-test failed to apply");
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

int mynah_conv1d_causal(const mynah_weights *file, const mynah_backend *backend,
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
    if (mynah_tensor_get(file, weight_name, &weight, error, error_capacity) != 0 ||
        mynah_tensor_get(file, bias_name, &bias, error, error_capacity) != 0) return -1;
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
        const double t0 = profile != NULL ? mynah_phase_seconds() : 0.0;
        if (mynah_backend_conv1d(backend, input, output,
                                 (int)in_channels, (int)out_channels, (int)length,
                                 (int)kernel, (int)dilation,
                                 weight.data, bias.data,
                                 error, error_capacity) == 0) {
            if (profile != NULL) {
                profile->gemm_seconds += mynah_phase_seconds() - t0;
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
            const double pack_start = profile != NULL ? mynah_phase_seconds() : 0.0;
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
                if (profile != NULL) profile->pack_seconds += mynah_phase_seconds() - pack_start;
                const double gemm_start = profile != NULL ? mynah_phase_seconds() : 0.0;
                mynah_graph_sgemm(backend, 0, 0, (int)out_channels, (int)length, (int)inner, 1.0f, weight.data, (int)inner, columns, (int)length, 1.0f, output, (int)length, error, error_capacity);
                if (profile != NULL) profile->gemm_seconds += mynah_phase_seconds() - gemm_start;
                if (owns_columns) free(columns);
                return 0;
            }
        }
        if (in_channels == 0u || out_channels > SIZE_MAX / in_channels ||
            out_channels * in_channels > SIZE_MAX / sizeof(float)) {
            mynah_graph_error(error, error_capacity,
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
            mynah_graph_error(error, error_capacity, "out of memory in causal conv1d");
            return -1;
        }
        for (size_t k = 0; k < kernel; ++k) {
            const size_t shift = (kernel - 1u - k) * dilation;
            if (shift >= length) continue;
        if (owns_wk) {
            const double pack_start = profile != NULL ? mynah_phase_seconds() : 0.0;
            for (size_t o = 0; o < out_channels; ++o) {
                for (size_t i = 0; i < in_channels; ++i) {
                    wk[o * in_channels + i] =
                        weight.data[(o * in_channels + i) * kernel + k];
                }
            }
            if (profile != NULL) profile->pack_seconds += mynah_phase_seconds() - pack_start;
        }
        const size_t n = length - shift;
        const float *tap_weights = owns_wk
            ? wk : wk + k * out_channels * in_channels;
        const double gemm_start = profile != NULL ? mynah_phase_seconds() : 0.0;
        mynah_graph_sgemm(backend, 0, 0, (int)out_channels, (int)n,
                    (int)in_channels, 1.0f, tap_weights, (int)in_channels,
                    input, (int)length, 1.0f, output + shift, (int)length,
                    error, error_capacity);
            if (profile != NULL) profile->gemm_seconds += mynah_phase_seconds() - gemm_start;
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

int mynah_conv_transpose_causal(const mynah_weights *file, const char *weight_name,
                                 const char *bias_name, const float *input, float *output,
                                 size_t in_channels, size_t out_channels, size_t length,
                                 size_t kernel, size_t stride, size_t groups,
                                 codec_conv_profile *profile,
                                 char *error, size_t error_capacity) {
    const double operation_start = profile != NULL ? mynah_phase_seconds() : 0.0;
    mynah_tensor weight;
    mynah_tensor bias;
    if (mynah_tensor_get(file, weight_name, &weight, error, error_capacity) != 0 ||
        mynah_tensor_get(file, bias_name, &bias, error, error_capacity) != 0) return -1;
    const size_t full_length = (length - 1u) * stride + kernel;
    const size_t trim = kernel - stride;
    const size_t output_length = full_length - trim;
    convt_ctx ctx = {input, output, weight.data, bias.data, length, output_length,
                     kernel, stride, in_channels / groups, out_channels / groups};
    mynah_parallel_for((int)out_channels, convt_channel, &ctx);
    if (profile != NULL) {
        profile->transpose_seconds += mynah_phase_seconds() - operation_start;
        profile->transpose_calls++;
    }
    return 0;
}