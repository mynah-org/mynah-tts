#include "codec_nanocodec.h"

#include "conv1d.h"
#include "dispatch.h"
#include "kernels.h"
#include "mynah_util.h"
#include "threads.h"

#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* vDSP and vForce, used by the snake activation's vectorised path. */
#if defined(MYNAH_USE_ACCELERATE)
#include <Accelerate/Accelerate.h>
#elif defined(MYNAH_USE_OPENBLAS)
#include <cblas.h>
#endif

/* MYNAH_SNAKE_SCALAR falls back from the vectorised Snake activation
 * (vDSP_vsmul + vvsinf + vDSP_vsq + vDSP_vsma over whole channel rows) to the
 * per-sample scalar form.  The vector path exists only on Accelerate builds,
 * so elsewhere this is 0 by construction rather than by environment. */
int mynah_snake_vector_enabled(void) {
#if defined(MYNAH_USE_ACCELERATE)
    static int cached = -1;
    if (cached < 0) cached = getenv("MYNAH_SNAKE_SCALAR") == NULL;
    return cached;
#else
    return 0;
#endif
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

static int half_snake(const mynah_weights *file, const mynah_backend *backend,
                      const char *alpha_name,
                      float *signal, size_t channels, size_t length,
                      codec_conv_profile *profile,
                      char *error, size_t error_capacity) {
    const double operation_start = profile != NULL ? mynah_phase_seconds() : 0.0;
    mynah_tensor alpha;
    if (mynah_tensor_get(file, alpha_name, &alpha, error, error_capacity) != 0) return -1;
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
                    profile->snake_seconds += mynah_phase_seconds() - operation_start;
                    profile->snake_calls++;
                }
                return 0;
            }
        }
        /* Fall through to CPU path on failure. */
    }
#if defined(MYNAH_USE_ACCELERATE)
    if (mynah_snake_vector_enabled() && snake_channels > 0u &&
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
                    profile->snake_seconds += mynah_phase_seconds() - operation_start;
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
        profile->snake_seconds += mynah_phase_seconds() - operation_start;
        profile->snake_calls++;
    }
    return 0;
}

static int res_layer(const mynah_weights *file, const mynah_backend *backend,
                     codec_bnns_cache *bnns_cache,
                     size_t stage, const float *input,
                     float *output, size_t channels, size_t length,
                     codec_conv_profile *profile, char *error, size_t error_capacity) {
    const size_t kernels[3] = {3u, 7u, 11u};
    const size_t dilations[3] = {1u, 3u, 5u};
    if (channels == 0u || length == 0u || channels > SIZE_MAX / length) {
        mynah_graph_error(error, error_capacity, "invalid codec residual workspace size");
        return -1;
    }
    const size_t elements = channels * length;
    float *branch = mynah_alloc_floats(elements, error, error_capacity);
    float *current = mynah_alloc_floats(elements, error, error_capacity);
    float *activated = mynah_alloc_floats(elements, error, error_capacity);
    float *residual = mynah_alloc_floats(elements, error, error_capacity);
    if (branch == NULL || current == NULL || activated == NULL || residual == NULL) {
        free(branch);
        free(current);
        free(activated);
        free(residual);
        return -1;
    }
    float *columns_workspace = NULL;
    size_t columns_capacity = 0;
    /* Only the Accelerate branch below reads it; without BNNS there is no
     * im2col workspace to pre-size at all. */
    int needs_columns_workspace = 1;
    (void)needs_columns_workspace;
#if defined(MYNAH_USE_ACCELERATE)
    if (!mynah_conv1d_sgemm_enabled() &&
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
            if (mynah_conv1d_causal(file, backend, bnns_cache, weight_name, bias_name, activated, residual,
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
            if (mynah_conv1d_causal(file, backend, bnns_cache, weight_name, bias_name, residual, branch,
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
static int res_layer_device(const mynah_weights *file, const mynah_backend *backend,
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
            ok = mynah_tensor_get(file, name, &alpha, error, error_capacity) == 0 &&
                 mynah_tensor_get(file, weight_name, &weight, error, error_capacity) == 0 &&
                 mynah_tensor_get(file, bias_name, &bias, error, error_capacity) == 0 &&
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
            if (ok) ok = mynah_tensor_get(file, name, &alpha, error, error_capacity) == 0 &&
                         mynah_tensor_get(file, weight_name, &weight, error, error_capacity) == 0 &&
                         mynah_tensor_get(file, bias_name, &bias, error, error_capacity) == 0 &&
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
    float *latent = mynah_alloc_floats(latent_channels * raw_length, error, error_capacity);
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
    if (mynah_tensor_get(model->codec, "audio_decoder.pre_conv.conv.weight", &weight,
               error, error_capacity) != 0 ||
        mynah_tensor_get(model->codec, "audio_decoder.pre_conv.conv.bias", &bias,
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
        if (mynah_tensor_get(model->codec, name, &weight, error, error_capacity) != 0 ||
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
        if (mynah_tensor_get(model->codec, weight_name, &weight, error, error_capacity) != 0 ||
            mynah_tensor_get(model->codec, bias_name, &bias, error, error_capacity) != 0 ||
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
    if (mynah_tensor_get(model->codec, name, &weight, error, error_capacity) != 0 ||
        mynah_backend_batch_begin(backend, error, error_capacity) != 0 ||
        mynah_backend_snake_dev(backend, current, weight.data, current_channels,
                                current_length, current_channels / 2u,
                                error, error_capacity) != 0 ||
        mynah_tensor_get(model->codec, bias_name, &bias, error, error_capacity) != 0) goto fail;
    /* Reload the post-conv views after using `weight` for Snake's alpha. */
    if (mynah_tensor_get(model->codec, weight_name, &weight, error, error_capacity) != 0 ||
        mynah_tensor_get(model->codec, bias_name, &bias, error, error_capacity) != 0 ||
        mynah_backend_dev_alloc(backend, current_length, &audio_dev,
                                error, error_capacity) != 0 ||
        mynah_backend_conv1d(backend, current, audio_dev, (int)current_channels, 1,
                             (int)current_length, 3, 1, weight.data, bias.data,
                             error, error_capacity) != 0 ||
        mynah_backend_clip_dev(backend, audio_dev, current_length,
                               error, error_capacity) != 0 ||
        mynah_backend_sync(backend, error, error_capacity) != 0) goto fail;
    audio = mynah_alloc_floats(current_length, error, error_capacity);
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

int mynah_nanocodec_decode(const mynah_tts_model *model, const unsigned *codes,
                        size_t raw_length, float **samples, size_t *sample_count,
                        char *error, size_t error_capacity) {
    const int resident_codec = decode_codec_resident(model, codes, raw_length,
                                                     samples, sample_count,
                                                     error, error_capacity);
    if (resident_codec != 1) return resident_codec;
    const int timing = getenv("MYNAH_TIMING") != NULL;
    const double codec_start = timing ? mynah_phase_seconds() : 0.0;
    double stage_seconds[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    codec_conv_profile conv_profile = {0};
    codec_conv_profile *profile = timing ? &conv_profile : NULL;
    codec_bnns_cache *bnns_cache = (codec_bnns_cache *)model->codec_cache;
    const size_t levels[4] = {8u, 7u, 6u, 6u};
    const size_t bases[4] = {1u, 8u, 56u, 336u};
    const size_t groups = 8u;
    const size_t latent_channels = 32u;
    float *latent = mynah_alloc_floats(latent_channels * raw_length, error, error_capacity);
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
    float *current = mynah_alloc_floats(864u * raw_length, error, error_capacity);
    if (current == NULL || mynah_conv1d_causal(model->codec, model->backend, bnns_cache, weight_name, bias_name, latent,
                                         current, 32u, 864u, raw_length, 7u, 1u,
                                         NULL, 0,
                                         profile, error, error_capacity) != 0) {
        free(latent);
        free(current);
        return -1;
    }
    free(latent);
    const double preconv_end = timing ? mynah_phase_seconds() : 0.0;
    size_t current_channels = 864u;
    size_t current_length = raw_length;
    const size_t rates[5] = {8u, 8u, 4u, 2u, 2u};
    for (size_t stage = 0; stage < 5u; ++stage) {
        const double stage_start = timing ? mynah_phase_seconds() : 0.0;
        snprintf(weight_name, sizeof(weight_name), "audio_decoder.activations.%zu.activation.snake_act.alpha", stage);
        if (half_snake(model->codec, model->backend, weight_name, current, current_channels, current_length,
                       profile, error, error_capacity) != 0) {
            free(current);
            return -1;
        }
        const size_t next_channels = current_channels / 2u;
        const size_t next_length = current_length * rates[stage];
        float *upsampled = mynah_alloc_floats(next_channels * next_length, error, error_capacity);
        if (upsampled == NULL) {
            free(current);
            return -1;
        }
        snprintf(weight_name, sizeof(weight_name), "audio_decoder.up_sample_conv_layers.%zu.conv.weight", stage);
        snprintf(bias_name, sizeof(bias_name), "audio_decoder.up_sample_conv_layers.%zu.conv.bias", stage);
        if (mynah_conv_transpose_causal(model->codec, weight_name, bias_name, current, upsampled,
                                  current_channels, next_channels, current_length,
                                  rates[stage] * 2u, rates[stage], next_channels,
                                  profile, error, error_capacity) != 0) {
            free(current);
            free(upsampled);
            return -1;
        }
        free(current);
        current = mynah_alloc_floats(next_channels * next_length, error, error_capacity);
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
        if (timing) stage_seconds[stage] = mynah_phase_seconds() - stage_start;
    }
    snprintf(weight_name, sizeof(weight_name), "audio_decoder.post_activation.activation.snake_act.alpha");
    if (half_snake(model->codec, model->backend, weight_name, current, current_channels, current_length,
                   profile, error, error_capacity) != 0) {
        free(current);
        return -1;
    }
    float *audio = mynah_alloc_floats(current_length, error, error_capacity);
    if (audio == NULL) {
        free(current);
        return -1;
    }
    snprintf(weight_name, sizeof(weight_name), "audio_decoder.post_conv.conv.weight");
    snprintf(bias_name, sizeof(bias_name), "audio_decoder.post_conv.conv.bias");
    if (mynah_conv1d_causal(model->codec, model->backend, bnns_cache, weight_name, bias_name, current, audio,
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
        const double codec_end = mynah_phase_seconds();
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
/* ======================================================================
 * Dispatch predicate
 * ====================================================================== */
static int probe_snake_vector(const char **why) {
    const int on = mynah_snake_vector_enabled();
    if (why != NULL) {
#if defined(MYNAH_USE_ACCELERATE)
        *why = on ? "[predicate] mynah_snake_vector_enabled(): the SEANet "
                    "Snake activation runs vDSP_vsmul + vvsinf + vDSP_vsq + "
                    "vDSP_vsma over whole channel rows"
                  : "[predicate] mynah_snake_vector_enabled(): "
                    "MYNAH_SNAKE_SCALAR is set, so Snake runs the per-sample "
                    "scalar form -- the rollback path, not the default";
#else
        *why = "[predicate] mynah_snake_vector_enabled(): the vectorised Snake "
               "needs vDSP/vForce, which this build does not link; the scalar "
               "form is the only one compiled";
#endif
    }
    return on;
}

void mynah_codec_dispatch_probes(void) {
    mynah_dispatch_register_probe("codec.snake_vector", probe_snake_vector);
}
