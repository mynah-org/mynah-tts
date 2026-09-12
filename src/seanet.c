/*
 * Causal SEANet decoder and Mimi up/downsample.  See seanet.h for the
 * streaming contract and for why the position counter is part of the state.
 *
 * Layout: channel major [channels][length], i.e. torch [1, C, T].
 */
#include "seanet.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "kernels.h"

/* ------------------------------------------------------------------ utils */

static void sea_set_error(char *error, size_t capacity, const char *format, ...)
    __attribute__((format(printf, 3, 4)));

static void sea_set_error(char *error, size_t capacity, const char *format,
                          ...) {
    if (error == NULL || capacity == 0) return;
    va_list args;
    va_start(args, format);
    (void)vsnprintf(error, capacity, format, args);
    va_end(args);
}

static int sea_mul(size_t a, size_t b, size_t *out) {
    if (a != 0 && b > (size_t)-1 / a) return -1;
    *out = a * b;
    return 0;
}

static int sea_add(size_t a, size_t b, size_t *out) {
    if (b > (size_t)-1 - a) return -1;
    *out = a + b;
    return 0;
}

static size_t sea_max(size_t a, size_t b) { return (a > b) ? a : b; }

void mynah_seanet_elu_f32(const float *input, float *output, size_t n,
                          float alpha) {
    if (input == NULL || output == NULL) return;
    for (size_t i = 0; i < n; ++i) {
        const float x = input[i];
        output[i] = (x > 0.0f) ? x : alpha * (expf(x) - 1.0f);
    }
}

/* ------------------------------------------------ causal streaming conv1d */

static int conv_effective_kernel(const mynah_conv1d_spec *spec, size_t *out) {
    size_t span = 0;
    if (sea_mul(spec->kernel_size - 1u, spec->dilation, &span) != 0) return -1;
    return sea_add(span, 1u, out);
}

static int conv_validate(const mynah_conv1d_spec *spec, size_t *effective,
                         char *error, size_t error_capacity) {
    if (spec == NULL) {
        sea_set_error(error, error_capacity, "conv1d: null spec");
        return -1;
    }
    if (spec->in_channels == 0 || spec->out_channels == 0 ||
        spec->kernel_size == 0 || spec->stride == 0 || spec->dilation == 0 ||
        spec->groups == 0) {
        sea_set_error(error, error_capacity, "conv1d: zero-valued dimension");
        return -1;
    }
    if ((spec->in_channels % spec->groups) != 0 ||
        (spec->out_channels % spec->groups) != 0) {
        sea_set_error(error, error_capacity,
                      "conv1d: channels %zu/%zu not divisible by groups %zu",
                      spec->in_channels, spec->out_channels, spec->groups);
        return -1;
    }
    if (conv_effective_kernel(spec, effective) != 0) {
        sea_set_error(error, error_capacity, "conv1d: kernel span overflow");
        return -1;
    }
    if (*effective < spec->stride) {
        sea_set_error(error, error_capacity,
                      "conv1d: effective kernel %zu < stride %zu", *effective,
                      spec->stride);
        return -1;
    }
    return 0;
}

size_t mynah_causal_conv1d_scratch(const mynah_conv1d_spec *spec,
                                   size_t max_in_len) {
    size_t effective = 0;
    if (conv_validate(spec, &effective, NULL, 0) != 0) return 0;
    const size_t tail = effective - spec->stride;
    if (tail == 0) return 0;
    size_t previous = 0;
    size_t window = 0;
    size_t span = 0;
    if (sea_mul(spec->in_channels, tail, &previous) != 0) return 0;
    if (sea_add(tail, max_in_len, &span) != 0) return 0;
    if (sea_mul(spec->in_channels, span, &window) != 0) return 0;
    size_t total = 0;
    if (sea_add(previous, window, &total) != 0) return 0;
    return total;
}

int mynah_causal_conv1d_init(mynah_causal_conv1d *conv,
                             const mynah_conv1d_spec *spec, size_t max_in_len,
                             float *scratch, size_t scratch_floats, char *error,
                             size_t error_capacity) {
    if (conv == NULL) {
        sea_set_error(error, error_capacity, "conv1d: null object");
        return -1;
    }
    size_t effective = 0;
    if (conv_validate(spec, &effective, error, error_capacity) != 0) return -1;
    if (max_in_len == 0 || (max_in_len % spec->stride) != 0) {
        sea_set_error(error, error_capacity,
                      "conv1d: max_in_len %zu must be a positive multiple of "
                      "stride %zu",
                      max_in_len, spec->stride);
        return -1;
    }
    memset(conv, 0, sizeof(*conv));
    conv->spec = *spec;
    conv->tail = effective - spec->stride;
    conv->max_in_len = max_in_len;
    const size_t needed = mynah_causal_conv1d_scratch(spec, max_in_len);
    if (needed > scratch_floats) {
        sea_set_error(error, error_capacity,
                      "conv1d: scratch too small (%zu < %zu)", scratch_floats,
                      needed);
        return -1;
    }
    if (conv->tail > 0) {
        if (scratch == NULL) {
            sea_set_error(error, error_capacity, "conv1d: null scratch");
            return -1;
        }
        conv->previous = scratch;
        conv->window = scratch + spec->in_channels * conv->tail;
    }
    mynah_causal_conv1d_reset(conv);
    return 0;
}

void mynah_causal_conv1d_reset(mynah_causal_conv1d *conv) {
    if (conv == NULL) return;
    if (conv->previous != NULL && conv->tail > 0) {
        memset(conv->previous, 0,
               conv->spec.in_channels * conv->tail * sizeof(float));
    }
    conv->primed = 0;
}

int mynah_causal_conv1d_apply(mynah_causal_conv1d *conv,
                              const mynah_conv_weights *weights,
                              const float *input, size_t in_len,
                              float *output) {
    if (conv == NULL || weights == NULL || weights->weight == NULL ||
        input == NULL || output == NULL) {
        return -1;
    }
    const mynah_conv1d_spec *spec = &conv->spec;
    if (in_len == 0 || in_len > conv->max_in_len ||
        (in_len % spec->stride) != 0) {
        return -1;
    }
    const size_t tail = conv->tail;
    if (spec->pad_mode == MYNAH_CONV_PAD_REPLICATE && tail > 0 &&
        in_len < tail) {
        /* Upstream asserts the same thing: replicate padding needs at least a
         * full tail of content to keep the ring buffer meaningful. */
        return -1;
    }

    const size_t in_channels = spec->in_channels;
    const float *win = input;
    size_t window_len = in_len;

    if (tail > 0) {
        if (spec->pad_mode == MYNAH_CONV_PAD_REPLICATE && !conv->primed) {
            for (size_t c = 0; c < in_channels; ++c) {
                const float first = input[c * in_len];
                float *dst = conv->previous + c * tail;
                for (size_t i = 0; i < tail; ++i) dst[i] = first;
            }
        }
        window_len = tail + in_len;
        for (size_t c = 0; c < in_channels; ++c) {
            float *dst = conv->window + c * window_len;
            memcpy(dst, conv->previous + c * tail, tail * sizeof(float));
            memcpy(dst + tail, input + c * in_len, in_len * sizeof(float));
        }
        win = conv->window;
    }

    const size_t kernel = spec->kernel_size;
    const size_t dilation = spec->dilation;
    const size_t stride = spec->stride;
    const size_t groups = spec->groups;
    const size_t in_per_group = in_channels / groups;
    const size_t out_per_group = spec->out_channels / groups;
    const size_t out_len = in_len / stride;

    for (size_t oc = 0; oc < spec->out_channels; ++oc) {
        const size_t group = oc / out_per_group;
        const float *weight_row = weights->weight + oc * in_per_group * kernel;
        float *out_row = output + oc * out_len;
        const float bias = (weights->bias != NULL) ? weights->bias[oc] : 0.0f;
        for (size_t n = 0; n < out_len; ++n) {
            const size_t base = n * stride;
            float acc = 0.0f;
            for (size_t j = 0; j < in_per_group; ++j) {
                const float *w = weight_row + j * kernel;
                const float *x =
                    win + (group * in_per_group + j) * window_len + base;
                if (dilation == 1u) {
                    acc += mynah_dot_f32(w, x, kernel);
                } else {
                    for (size_t k = 0; k < kernel; ++k) {
                        acc += w[k] * x[k * dilation];
                    }
                }
            }
            out_row[n] = acc + bias;
        }
    }

    if (tail > 0) {
        /* previous := last `tail` samples of the concatenated window. */
        for (size_t c = 0; c < in_channels; ++c) {
            memcpy(conv->previous + c * tail,
                   conv->window + c * window_len + (window_len - tail),
                   tail * sizeof(float));
        }
        conv->primed = 1;
    }
    return 0;
}

/* --------------------------------------- causal streaming convtranspose1d */

static int convtr_validate(const mynah_convtr1d_spec *spec, char *error,
                           size_t error_capacity) {
    if (spec == NULL) {
        sea_set_error(error, error_capacity, "convtr1d: null spec");
        return -1;
    }
    if (spec->in_channels == 0 || spec->out_channels == 0 ||
        spec->kernel_size == 0 || spec->stride == 0 || spec->groups == 0) {
        sea_set_error(error, error_capacity, "convtr1d: zero-valued dimension");
        return -1;
    }
    if ((spec->in_channels % spec->groups) != 0 ||
        (spec->out_channels % spec->groups) != 0) {
        sea_set_error(error, error_capacity,
                      "convtr1d: channels %zu/%zu not divisible by groups %zu",
                      spec->in_channels, spec->out_channels, spec->groups);
        return -1;
    }
    if (spec->kernel_size < spec->stride) {
        sea_set_error(error, error_capacity,
                      "convtr1d: kernel %zu < stride %zu", spec->kernel_size,
                      spec->stride);
        return -1;
    }
    return 0;
}

static int convtr_full_len(const mynah_convtr1d_spec *spec, size_t in_len,
                           size_t *out) {
    size_t span = 0;
    if (sea_mul(in_len - 1u, spec->stride, &span) != 0) return -1;
    return sea_add(span, spec->kernel_size, out);
}

size_t mynah_causal_convtr1d_scratch(const mynah_convtr1d_spec *spec,
                                     size_t max_in_len) {
    if (convtr_validate(spec, NULL, 0) != 0 || max_in_len == 0) return 0;
    const size_t tail = spec->kernel_size - spec->stride;
    size_t full_len = 0;
    if (convtr_full_len(spec, max_in_len, &full_len) != 0) return 0;
    size_t partial = 0;
    size_t full = 0;
    if (sea_mul(spec->out_channels, tail, &partial) != 0) return 0;
    if (sea_mul(spec->out_channels, full_len, &full) != 0) return 0;
    size_t total = 0;
    if (sea_add(partial, full, &total) != 0) return 0;
    return total;
}

int mynah_causal_convtr1d_init(mynah_causal_convtr1d *convtr,
                               const mynah_convtr1d_spec *spec,
                               size_t max_in_len, float *scratch,
                               size_t scratch_floats, char *error,
                               size_t error_capacity) {
    if (convtr == NULL) {
        sea_set_error(error, error_capacity, "convtr1d: null object");
        return -1;
    }
    if (convtr_validate(spec, error, error_capacity) != 0) return -1;
    if (max_in_len == 0) {
        sea_set_error(error, error_capacity, "convtr1d: max_in_len is zero");
        return -1;
    }
    const size_t needed = mynah_causal_convtr1d_scratch(spec, max_in_len);
    if (needed == 0 || needed > scratch_floats || scratch == NULL) {
        sea_set_error(error, error_capacity,
                      "convtr1d: scratch too small (%zu < %zu)", scratch_floats,
                      needed);
        return -1;
    }
    memset(convtr, 0, sizeof(*convtr));
    convtr->spec = *spec;
    convtr->tail = spec->kernel_size - spec->stride;
    convtr->max_in_len = max_in_len;
    convtr->partial = scratch;
    convtr->full = scratch + spec->out_channels * convtr->tail;
    mynah_causal_convtr1d_reset(convtr);
    return 0;
}

void mynah_causal_convtr1d_reset(mynah_causal_convtr1d *convtr) {
    if (convtr == NULL || convtr->partial == NULL) return;
    if (convtr->tail > 0) {
        memset(convtr->partial, 0,
               convtr->spec.out_channels * convtr->tail * sizeof(float));
    }
}

int mynah_causal_convtr1d_apply(mynah_causal_convtr1d *convtr,
                                const mynah_conv_weights *weights,
                                const float *input, size_t in_len,
                                float *output) {
    if (convtr == NULL || weights == NULL || weights->weight == NULL ||
        input == NULL || output == NULL) {
        return -1;
    }
    if (in_len == 0 || in_len > convtr->max_in_len) return -1;

    const mynah_convtr1d_spec *spec = &convtr->spec;
    size_t full_len = 0;
    if (convtr_full_len(spec, in_len, &full_len) != 0) return -1;
    const size_t tail = convtr->tail;
    const size_t out_len = in_len * spec->stride;
    float *full = convtr->full;

    /* PyTorch adds the bias to every output position. */
    for (size_t oc = 0; oc < spec->out_channels; ++oc) {
        float *row = full + oc * full_len;
        const float bias = (weights->bias != NULL) ? weights->bias[oc] : 0.0f;
        if (bias == 0.0f) {
            memset(row, 0, full_len * sizeof(float));
        } else {
            for (size_t i = 0; i < full_len; ++i) row[i] = bias;
        }
    }

    const size_t kernel = spec->kernel_size;
    const size_t stride = spec->stride;
    const size_t in_per_group = spec->in_channels / spec->groups;
    const size_t out_per_group = spec->out_channels / spec->groups;

    for (size_t ic = 0; ic < spec->in_channels; ++ic) {
        const size_t group = ic / in_per_group;
        const float *weight_row = weights->weight + ic * out_per_group * kernel;
        const float *in_row = input + ic * in_len;
        for (size_t j = 0; j < out_per_group; ++j) {
            const float *w = weight_row + j * kernel;
            float *out_base = full + (group * out_per_group + j) * full_len;
            for (size_t t = 0; t < in_len; ++t) {
                const float value = in_row[t];
                if (value == 0.0f) continue;
                float *dst = out_base + t * stride;
                for (size_t k = 0; k < kernel; ++k) dst[k] += w[k] * value;
            }
        }
    }

    if (tail > 0) {
        /* Faithful to upstream StreamingConvTranspose1d.forward: the carried
         * tail is folded into the head first, and only then is the new tail
         * taken (with the bias removed, because the next call re-adds it). */
        for (size_t oc = 0; oc < spec->out_channels; ++oc) {
            float *row = full + oc * full_len;
            const float *carry = convtr->partial + oc * tail;
            for (size_t i = 0; i < tail; ++i) row[i] += carry[i];
        }
        for (size_t oc = 0; oc < spec->out_channels; ++oc) {
            const float *row = full + oc * full_len;
            float *carry = convtr->partial + oc * tail;
            const float bias =
                (weights->bias != NULL) ? weights->bias[oc] : 0.0f;
            for (size_t i = 0; i < tail; ++i) {
                carry[i] = row[full_len - tail + i] - bias;
            }
        }
    }

    for (size_t oc = 0; oc < spec->out_channels; ++oc) {
        memcpy(output + oc * out_len, full + oc * full_len,
               out_len * sizeof(float));
    }
    return 0;
}

/* ------------------------------------------------------------ SEANet ops */

enum {
    SEA_OP_CONV = 0,
    SEA_OP_CONVTR = 1,
    SEA_OP_RESBLOCK = 2
};

enum {
    SEA_W_FIRST = 0,
    SEA_W_CONVTR = 1,
    SEA_W_BLOCK = 2,
    SEA_W_LAST = 3
};

typedef struct {
    int kind;
    int pre_elu;
    int weight_kind;
    size_t weight_index;
    size_t in_len;  /* maximum input length for this op */
    size_t out_len; /* maximum output length for this op */
    mynah_conv1d_spec conv_spec;
    mynah_convtr1d_spec convtr_spec;
    mynah_conv1d_spec rb1_spec;
    mynah_conv1d_spec rb2_spec;
    mynah_causal_conv1d conv;
    mynah_causal_convtr1d convtr;
    mynah_causal_conv1d rb1;
    mynah_causal_conv1d rb2;
} sea_op;

struct mynah_seanet_state {
    mynah_seanet_config config;
    size_t *ratios; /* owned copy; config.ratios points at it */
    mynah_resample_config up_config;
    int has_upsample;
    size_t max_latent_frames;
    size_t max_encoder_frames;
    size_t encoder_stride;
    size_t hop_length;
    size_t position;

    sea_op *ops;
    size_t n_ops;

    mynah_causal_convtr1d upsample;

    float *arena;
    size_t arena_floats;
    float *work[3];
    size_t work_floats;
};

struct mynah_seanet_downsample {
    mynah_resample_config config;
    mynah_causal_conv1d conv;
    float *arena;
};

static mynah_conv1d_spec sea_conv_spec(size_t in_channels, size_t out_channels,
                                       size_t kernel, size_t stride,
                                       size_t dilation, size_t groups,
                                       mynah_conv_pad_mode pad_mode) {
    mynah_conv1d_spec spec;
    spec.in_channels = in_channels;
    spec.out_channels = out_channels;
    spec.kernel_size = kernel;
    spec.stride = stride;
    spec.dilation = dilation;
    spec.groups = groups;
    spec.pad_mode = pad_mode;
    return spec;
}

/*
 * Walks the decoder topology exactly once.  With `ops == NULL` it only counts
 * and measures, which is how the single arena gets sized before anything is
 * allocated.
 */
static int sea_build_ops(const mynah_seanet_config *config,
                         size_t max_encoder_frames, sea_op *ops,
                         size_t *n_ops_out, size_t *scratch_out,
                         size_t *max_elems_out, char *error,
                         size_t error_capacity) {
    size_t mult = 1u;
    for (size_t i = 0; i < config->n_ratios; ++i) {
        if (sea_mul(mult, 2u, &mult) != 0) {
            sea_set_error(error, error_capacity, "seanet: ratio count overflow");
            return -1;
        }
    }
    size_t channels_here = 0;
    if (sea_mul(mult, config->n_filters, &channels_here) != 0) {
        sea_set_error(error, error_capacity, "seanet: filter count overflow");
        return -1;
    }

    size_t n_ops = 0;
    size_t scratch = 0;
    size_t max_elems = 0;
    size_t len = max_encoder_frames;

    size_t elems = 0;
    if (sea_mul(config->dimension, len, &elems) != 0) {
        sea_set_error(error, error_capacity, "seanet: activation overflow");
        return -1;
    }
    max_elems = sea_max(max_elems, elems);

#define SEA_EMIT_CONV(SPECVAR, MAXLEN, TARGET)                                \
    do {                                                                     \
        const size_t need = mynah_causal_conv1d_scratch(&(SPECVAR), (MAXLEN)); \
        size_t eff = 0;                                                      \
        if (conv_validate(&(SPECVAR), &eff, error, error_capacity) != 0)      \
            return -1;                                                       \
        if (sea_add(scratch, need, &scratch) != 0) {                          \
            sea_set_error(error, error_capacity, "seanet: scratch overflow");  \
            return -1;                                                       \
        }                                                                    \
        (void)(TARGET);                                                      \
    } while (0)

    /* index 0: the entry convolution */
    {
        mynah_conv1d_spec spec =
            sea_conv_spec(config->dimension, channels_here, config->kernel_size,
                          1u, 1u, 1u, MYNAH_CONV_PAD_ZERO);
        SEA_EMIT_CONV(spec, len, 0);
        if (ops != NULL) {
            sea_op *op = &ops[n_ops];
            memset(op, 0, sizeof(*op));
            op->kind = SEA_OP_CONV;
            op->pre_elu = 0;
            op->weight_kind = SEA_W_FIRST;
            op->weight_index = 0;
            op->in_len = len;
            op->out_len = len;
            op->conv_spec = spec;
        }
        ++n_ops;
        if (sea_mul(channels_here, len, &elems) != 0) {
            sea_set_error(error, error_capacity, "seanet: activation overflow");
            return -1;
        }
        max_elems = sea_max(max_elems, elems);
    }

    for (size_t stage = 0; stage < config->n_ratios; ++stage) {
        const size_t ratio = config->ratios[stage];
        if (ratio == 0) {
            sea_set_error(error, error_capacity, "seanet: ratio %zu is zero",
                          stage);
            return -1;
        }
        const size_t out_channels = channels_here / 2u;
        if (out_channels == 0) {
            sea_set_error(error, error_capacity,
                          "seanet: stage %zu collapses to zero channels",
                          stage);
            return -1;
        }
        mynah_convtr1d_spec tspec;
        tspec.in_channels = channels_here;
        tspec.out_channels = out_channels;
        tspec.kernel_size = ratio * 2u;
        tspec.stride = ratio;
        tspec.groups = 1u;
        if (convtr_validate(&tspec, error, error_capacity) != 0) return -1;
        {
            const size_t need = mynah_causal_convtr1d_scratch(&tspec, len);
            if (need == 0 || sea_add(scratch, need, &scratch) != 0) {
                sea_set_error(error, error_capacity,
                              "seanet: convtr scratch overflow at stage %zu",
                              stage);
                return -1;
            }
        }
        const size_t in_len_here = len;
        if (sea_mul(len, ratio, &len) != 0) {
            sea_set_error(error, error_capacity, "seanet: length overflow");
            return -1;
        }
        if (ops != NULL) {
            sea_op *op = &ops[n_ops];
            memset(op, 0, sizeof(*op));
            op->kind = SEA_OP_CONVTR;
            op->pre_elu = 1;
            op->weight_kind = SEA_W_CONVTR;
            op->weight_index = stage;
            op->in_len = in_len_here;
            op->out_len = len;
            op->convtr_spec = tspec;
        }
        ++n_ops;
        if (sea_mul(out_channels, len, &elems) != 0) {
            sea_set_error(error, error_capacity, "seanet: activation overflow");
            return -1;
        }
        max_elems = sea_max(max_elems, elems);

        for (size_t j = 0; j < config->n_residual_layers; ++j) {
            if (config->compress == 0) {
                sea_set_error(error, error_capacity, "seanet: compress is 0");
                return -1;
            }
            const size_t hidden = out_channels / config->compress;
            if (hidden == 0) {
                sea_set_error(error, error_capacity,
                              "seanet: compress %zu collapses stage %zu",
                              config->compress, stage);
                return -1;
            }
            size_t dilation = 1u;
            for (size_t d = 0; d < j; ++d) {
                if (sea_mul(dilation, config->dilation_base, &dilation) != 0) {
                    sea_set_error(error, error_capacity,
                                  "seanet: dilation overflow");
                    return -1;
                }
            }
            mynah_conv1d_spec s1 = sea_conv_spec(
                out_channels, hidden, config->residual_kernel_size, 1u,
                dilation, 1u, MYNAH_CONV_PAD_ZERO);
            mynah_conv1d_spec s2 = sea_conv_spec(hidden, out_channels, 1u, 1u,
                                                 1u, 1u, MYNAH_CONV_PAD_ZERO);
            SEA_EMIT_CONV(s1, len, 0);
            SEA_EMIT_CONV(s2, len, 0);
            if (ops != NULL) {
                sea_op *op = &ops[n_ops];
                memset(op, 0, sizeof(*op));
                op->kind = SEA_OP_RESBLOCK;
                op->pre_elu = 0;
                op->weight_kind = SEA_W_BLOCK;
                op->weight_index = stage * config->n_residual_layers + j;
                op->in_len = len;
                op->out_len = len;
                op->rb1_spec = s1;
                op->rb2_spec = s2;
            }
            ++n_ops;
            if (sea_mul(hidden, len, &elems) != 0) {
                sea_set_error(error, error_capacity,
                              "seanet: activation overflow");
                return -1;
            }
            max_elems = sea_max(max_elems, elems);
        }
        channels_here = out_channels;
    }

    if (channels_here != config->n_filters) {
        sea_set_error(error, error_capacity,
                      "seanet: last stage has %zu channels, expected "
                      "n_filters %zu",
                      channels_here, config->n_filters);
        return -1;
    }

    {
        mynah_conv1d_spec spec =
            sea_conv_spec(config->n_filters, config->channels,
                          config->last_kernel_size, 1u, 1u, 1u,
                          MYNAH_CONV_PAD_ZERO);
        SEA_EMIT_CONV(spec, len, 0);
        if (ops != NULL) {
            sea_op *op = &ops[n_ops];
            memset(op, 0, sizeof(*op));
            op->kind = SEA_OP_CONV;
            op->pre_elu = 1;
            op->weight_kind = SEA_W_LAST;
            op->weight_index = 0;
            op->in_len = len;
            op->out_len = len;
            op->conv_spec = spec;
        }
        ++n_ops;
        if (sea_mul(config->channels, len, &elems) != 0) {
            sea_set_error(error, error_capacity, "seanet: activation overflow");
            return -1;
        }
        max_elems = sea_max(max_elems, elems);
    }

#undef SEA_EMIT_CONV

    if (n_ops_out != NULL) *n_ops_out = n_ops;
    if (scratch_out != NULL) *scratch_out = scratch;
    if (max_elems_out != NULL) *max_elems_out = max_elems;
    return 0;
}

static int sea_config_valid(const mynah_seanet_config *config, char *error,
                            size_t error_capacity) {
    if (config == NULL) {
        sea_set_error(error, error_capacity, "seanet: null config");
        return -1;
    }
    if (config->channels == 0 || config->dimension == 0 ||
        config->n_filters == 0 || config->n_ratios == 0 ||
        config->ratios == NULL || config->kernel_size == 0 ||
        config->last_kernel_size == 0 || config->compress == 0) {
        sea_set_error(error, error_capacity, "seanet: incomplete config");
        return -1;
    }
    if (config->n_residual_layers > 0 &&
        (config->residual_kernel_size == 0 || config->dilation_base == 0)) {
        sea_set_error(error, error_capacity,
                      "seanet: residual layers need kernel and dilation base");
        return -1;
    }
    return 0;
}

mynah_seanet_state *mynah_seanet_state_create(const mynah_seanet_config *config,
                                              const mynah_resample_config *up,
                                              size_t max_latent_frames,
                                              char *error,
                                              size_t error_capacity) {
    if (error != NULL && error_capacity > 0) error[0] = '\0';
    if (sea_config_valid(config, error, error_capacity) != 0) return NULL;
    if (max_latent_frames == 0) {
        sea_set_error(error, error_capacity,
                      "seanet: max_latent_frames is zero");
        return NULL;
    }
    size_t encoder_stride = 1u;
    if (up != NULL) {
        if (up->stride == 0 || up->in_channels == 0 || up->out_channels == 0 ||
            up->groups == 0) {
            sea_set_error(error, error_capacity, "seanet: bad resample config");
            return NULL;
        }
        encoder_stride = up->stride;
    }
    size_t max_encoder_frames = 0;
    if (sea_mul(max_latent_frames, encoder_stride, &max_encoder_frames) != 0) {
        sea_set_error(error, error_capacity, "seanet: frame count overflow");
        return NULL;
    }

    size_t hop = 1u;
    for (size_t i = 0; i < config->n_ratios; ++i) {
        if (config->ratios[i] == 0 ||
            sea_mul(hop, config->ratios[i], &hop) != 0) {
            sea_set_error(error, error_capacity, "seanet: bad ratios");
            return NULL;
        }
    }

    size_t n_ops = 0;
    size_t conv_scratch = 0;
    size_t max_elems = 0;
    if (sea_build_ops(config, max_encoder_frames, NULL, &n_ops, &conv_scratch,
                      &max_elems, error, error_capacity) != 0) {
        return NULL;
    }

    mynah_seanet_state *state = calloc(1u, sizeof(*state));
    if (state == NULL) {
        sea_set_error(error, error_capacity, "seanet: out of memory");
        return NULL;
    }
    state->config = *config;
    state->ratios = calloc(config->n_ratios, sizeof(size_t));
    if (state->ratios == NULL) {
        sea_set_error(error, error_capacity, "seanet: out of memory");
        free(state);
        return NULL;
    }
    memcpy(state->ratios, config->ratios, config->n_ratios * sizeof(size_t));
    state->config.ratios = state->ratios;
    state->max_latent_frames = max_latent_frames;
    state->max_encoder_frames = max_encoder_frames;
    state->encoder_stride = encoder_stride;
    state->hop_length = hop;
    state->position = 0;
    state->n_ops = n_ops;
    state->work_floats = max_elems;

    size_t up_scratch = 0;
    if (up != NULL) {
        mynah_convtr1d_spec uspec;
        uspec.in_channels = up->in_channels;
        uspec.out_channels = up->out_channels;
        uspec.kernel_size = up->stride * 2u;
        uspec.stride = up->stride;
        uspec.groups = up->groups;
        up_scratch = mynah_causal_convtr1d_scratch(&uspec, max_latent_frames);
        if (up_scratch == 0) {
            sea_set_error(error, error_capacity, "seanet: bad upsample config");
            free(state->ratios);
            free(state);
            return NULL;
        }
        state->up_config = *up;
        state->has_upsample = 1;
    }

    size_t work_total = 0;
    size_t arena_floats = 0;
    if (sea_mul(max_elems, 3u, &work_total) != 0 ||
        sea_add(conv_scratch, up_scratch, &arena_floats) != 0 ||
        sea_add(arena_floats, work_total, &arena_floats) != 0) {
        sea_set_error(error, error_capacity, "seanet: arena overflow");
        free(state->ratios);
        free(state);
        return NULL;
    }

    state->ops = calloc(n_ops, sizeof(sea_op));
    state->arena = calloc(arena_floats ? arena_floats : 1u, sizeof(float));
    if (state->ops == NULL || state->arena == NULL) {
        sea_set_error(error, error_capacity, "seanet: out of memory");
        mynah_seanet_state_destroy(state);
        return NULL;
    }
    state->arena_floats = arena_floats;

    size_t counted_ops = 0;
    if (sea_build_ops(&state->config, max_encoder_frames, state->ops,
                      &counted_ops, NULL, NULL, error, error_capacity) != 0 ||
        counted_ops != n_ops) {
        sea_set_error(error, error_capacity, "seanet: topology mismatch");
        mynah_seanet_state_destroy(state);
        return NULL;
    }

    float *cursor = state->arena;
    size_t remaining = arena_floats;

#define SEA_TAKE(N)                                  \
    do {                                             \
        if ((N) > remaining) {                       \
            sea_set_error(error, error_capacity,     \
                          "seanet: arena exhausted"); \
            mynah_seanet_state_destroy(state);       \
            return NULL;                             \
        }                                            \
        remaining -= (N);                            \
    } while (0)

    for (size_t i = 0; i < n_ops; ++i) {
        sea_op *op = &state->ops[i];
        if (op->kind == SEA_OP_CONV) {
            const size_t need =
                mynah_causal_conv1d_scratch(&op->conv_spec, op->in_len);
            SEA_TAKE(need);
            if (mynah_causal_conv1d_init(&op->conv, &op->conv_spec, op->in_len,
                                         cursor, need, error,
                                         error_capacity) != 0) {
                mynah_seanet_state_destroy(state);
                return NULL;
            }
            cursor += need;
        } else if (op->kind == SEA_OP_CONVTR) {
            const size_t need =
                mynah_causal_convtr1d_scratch(&op->convtr_spec, op->in_len);
            SEA_TAKE(need);
            if (mynah_causal_convtr1d_init(&op->convtr, &op->convtr_spec,
                                           op->in_len, cursor, need, error,
                                           error_capacity) != 0) {
                mynah_seanet_state_destroy(state);
                return NULL;
            }
            cursor += need;
        } else {
            const size_t need1 =
                mynah_causal_conv1d_scratch(&op->rb1_spec, op->in_len);
            SEA_TAKE(need1);
            if (mynah_causal_conv1d_init(&op->rb1, &op->rb1_spec, op->in_len,
                                         cursor, need1, error,
                                         error_capacity) != 0) {
                mynah_seanet_state_destroy(state);
                return NULL;
            }
            cursor += need1;
            const size_t need2 =
                mynah_causal_conv1d_scratch(&op->rb2_spec, op->in_len);
            SEA_TAKE(need2);
            if (mynah_causal_conv1d_init(&op->rb2, &op->rb2_spec, op->in_len,
                                         cursor, need2, error,
                                         error_capacity) != 0) {
                mynah_seanet_state_destroy(state);
                return NULL;
            }
            cursor += need2;
        }
    }

    if (state->has_upsample) {
        mynah_convtr1d_spec uspec;
        uspec.in_channels = state->up_config.in_channels;
        uspec.out_channels = state->up_config.out_channels;
        uspec.kernel_size = state->up_config.stride * 2u;
        uspec.stride = state->up_config.stride;
        uspec.groups = state->up_config.groups;
        SEA_TAKE(up_scratch);
        if (mynah_causal_convtr1d_init(&state->upsample, &uspec,
                                       max_latent_frames, cursor, up_scratch,
                                       error, error_capacity) != 0) {
            mynah_seanet_state_destroy(state);
            return NULL;
        }
        cursor += up_scratch;
    }

    for (size_t i = 0; i < 3u; ++i) {
        SEA_TAKE(max_elems);
        state->work[i] = cursor;
        cursor += max_elems;
    }

#undef SEA_TAKE

    return state;
}

void mynah_seanet_state_destroy(mynah_seanet_state *state) {
    if (state == NULL) return;
    free(state->ops);
    free(state->arena);
    free(state->ratios);
    free(state);
}

void mynah_seanet_state_reset(mynah_seanet_state *state) {
    if (state == NULL) return;
    for (size_t i = 0; i < state->n_ops; ++i) {
        sea_op *op = &state->ops[i];
        if (op->kind == SEA_OP_CONV) {
            mynah_causal_conv1d_reset(&op->conv);
        } else if (op->kind == SEA_OP_CONVTR) {
            mynah_causal_convtr1d_reset(&op->convtr);
        } else {
            mynah_causal_conv1d_reset(&op->rb1);
            mynah_causal_conv1d_reset(&op->rb2);
        }
    }
    if (state->has_upsample) mynah_causal_convtr1d_reset(&state->upsample);
    state->position = 0; /* the half everybody forgets */
}

size_t mynah_seanet_state_position(const mynah_seanet_state *state) {
    return (state == NULL) ? 0 : state->position;
}

void mynah_seanet_state_advance(mynah_seanet_state *state,
                                size_t n_latent_frames) {
    if (state == NULL) return;
    size_t delta = 0;
    if (sea_mul(n_latent_frames, state->encoder_stride, &delta) != 0) return;
    size_t next = 0;
    if (sea_add(state->position, delta, &next) != 0) return;
    state->position = next;
}

size_t mynah_seanet_state_encoder_stride(const mynah_seanet_state *state) {
    return (state == NULL) ? 0 : state->encoder_stride;
}

size_t mynah_seanet_state_hop_length(const mynah_seanet_state *state) {
    return (state == NULL) ? 0 : state->hop_length;
}

size_t mynah_seanet_state_samples_per_latent(const mynah_seanet_state *state) {
    if (state == NULL) return 0;
    size_t out = 0;
    if (sea_mul(state->encoder_stride, state->hop_length, &out) != 0) return 0;
    return out;
}

size_t mynah_seanet_state_max_latent_frames(const mynah_seanet_state *state) {
    return (state == NULL) ? 0 : state->max_latent_frames;
}

int mynah_seanet_check_decoder_weights(const mynah_seanet_state *state,
                                       const mynah_seanet_decoder_weights *w,
                                       char *error, size_t error_capacity) {
    if (state == NULL || w == NULL) {
        sea_set_error(error, error_capacity, "seanet: null argument");
        return -1;
    }
    if (w->first.weight == NULL || w->last.weight == NULL) {
        sea_set_error(error, error_capacity,
                      "seanet: first/last conv weights missing");
        return -1;
    }
    if (w->convtr == NULL) {
        sea_set_error(error, error_capacity, "seanet: convtr array missing");
        return -1;
    }
    for (size_t i = 0; i < state->config.n_ratios; ++i) {
        if (w->convtr[i].weight == NULL) {
            sea_set_error(error, error_capacity,
                          "seanet: convtr[%zu] weight missing", i);
            return -1;
        }
    }
    const size_t n_blocks =
        state->config.n_ratios * state->config.n_residual_layers;
    if (n_blocks > 0) {
        if (w->blocks == NULL) {
            sea_set_error(error, error_capacity,
                          "seanet: resblock array missing");
            return -1;
        }
        for (size_t i = 0; i < n_blocks; ++i) {
            if (w->blocks[i].conv1.weight == NULL ||
                w->blocks[i].conv2.weight == NULL) {
                sea_set_error(error, error_capacity,
                              "seanet: resblock[%zu] weights missing", i);
                return -1;
            }
        }
    }
    return 0;
}

int mynah_seanet_upsample(mynah_seanet_state *state,
                          const mynah_conv_weights *weights,
                          const float *input, size_t n_latent_frames,
                          float *output) {
    if (state == NULL || !state->has_upsample) return -1;
    if (n_latent_frames == 0 || n_latent_frames > state->max_latent_frames) {
        return -1;
    }
    return mynah_causal_convtr1d_apply(&state->upsample, weights, input,
                                       n_latent_frames, output);
}

int mynah_seanet_decode(mynah_seanet_state *state,
                        const mynah_seanet_decoder_weights *weights,
                        const float *input, size_t n_encoder_frames,
                        float *output) {
    if (state == NULL || weights == NULL || input == NULL || output == NULL) {
        return -1;
    }
    if (n_encoder_frames == 0 ||
        n_encoder_frames > state->max_encoder_frames) {
        return -1;
    }
    const float alpha = state->config.elu_alpha;
    float *a = state->work[0];
    float *b = state->work[1];
    float *c = state->work[2];

    const float *cur_in = input;
    size_t len = n_encoder_frames;
    size_t channels = state->config.dimension;
    float *cur = NULL;

    for (size_t i = 0; i < state->n_ops; ++i) {
        sea_op *op = &state->ops[i];
        const int last_op = (i + 1u == state->n_ops);

        if (op->kind == SEA_OP_CONV) {
            const mynah_conv_weights *cw = (op->weight_kind == SEA_W_FIRST)
                                               ? &weights->first
                                               : &weights->last;
            const float *src;
            if (cur == NULL) {
                /* The entry convolution reads the caller's buffer and never
                 * has a pre-activation, so the input is never written to. */
                src = cur_in;
            } else {
                if (op->pre_elu) {
                    mynah_seanet_elu_f32(cur, cur, channels * len, alpha);
                }
                src = cur;
            }
            float *dst = last_op ? output : ((src == a) ? b : a);
            if (mynah_causal_conv1d_apply(&op->conv, cw, src, len, dst) != 0) {
                return -1;
            }
            cur = dst;
            channels = op->conv_spec.out_channels;
        } else if (op->kind == SEA_OP_CONVTR) {
            if (cur == NULL) return -1; /* topology always starts with a conv */
            float *src = cur;
            if (op->pre_elu) {
                mynah_seanet_elu_f32(src, src, channels * len, alpha);
            }
            float *dst = (src == a) ? b : a;
            if (mynah_causal_convtr1d_apply(&op->convtr,
                                            &weights->convtr[op->weight_index],
                                            src, len, dst) != 0) {
                return -1;
            }
            cur = dst;
            channels = op->convtr_spec.out_channels;
            len *= op->convtr_spec.stride;
        } else {
            if (cur == NULL) return -1;
            const mynah_seanet_resblock_weights *bw =
                &weights->blocks[op->weight_index];
            float *other = (cur == a) ? b : a;
            /* v = conv2(elu(conv1(elu(x)))); x = x + v  (true_skip shortcut) */
            mynah_seanet_elu_f32(cur, c, channels * len, alpha);
            if (mynah_causal_conv1d_apply(&op->rb1, &bw->conv1, c, len,
                                          other) != 0) {
                return -1;
            }
            const size_t hidden = op->rb1_spec.out_channels;
            mynah_seanet_elu_f32(other, other, hidden * len, alpha);
            if (mynah_causal_conv1d_apply(&op->rb2, &bw->conv2, other, len,
                                          c) != 0) {
                return -1;
            }
            for (size_t k = 0; k < channels * len; ++k) cur[k] += c[k];
        }
    }
    return 0;
}

/* ---------------------------------------------------------- downsample */

mynah_seanet_downsample *mynah_seanet_downsample_create(
    const mynah_resample_config *config, size_t max_in_len, char *error,
    size_t error_capacity) {
    if (error != NULL && error_capacity > 0) error[0] = '\0';
    if (config == NULL || config->stride == 0 || config->in_channels == 0 ||
        config->out_channels == 0 || config->groups == 0) {
        sea_set_error(error, error_capacity, "downsample: bad config");
        return NULL;
    }
    if (max_in_len == 0 || (max_in_len % config->stride) != 0) {
        sea_set_error(error, error_capacity,
                      "downsample: max_in_len %zu must be a multiple of "
                      "stride %zu",
                      max_in_len, config->stride);
        return NULL;
    }
    mynah_conv1d_spec spec = sea_conv_spec(
        config->in_channels, config->out_channels, config->stride * 2u,
        config->stride, 1u, config->groups, MYNAH_CONV_PAD_REPLICATE);
    const size_t need = mynah_causal_conv1d_scratch(&spec, max_in_len);
    mynah_seanet_downsample *down = calloc(1u, sizeof(*down));
    if (down == NULL) {
        sea_set_error(error, error_capacity, "downsample: out of memory");
        return NULL;
    }
    down->config = *config;
    down->arena = calloc(need ? need : 1u, sizeof(float));
    if (down->arena == NULL) {
        sea_set_error(error, error_capacity, "downsample: out of memory");
        free(down);
        return NULL;
    }
    if (mynah_causal_conv1d_init(&down->conv, &spec, max_in_len, down->arena,
                                 need, error, error_capacity) != 0) {
        free(down->arena);
        free(down);
        return NULL;
    }
    return down;
}

void mynah_seanet_downsample_destroy(mynah_seanet_downsample *down) {
    if (down == NULL) return;
    free(down->arena);
    free(down);
}

void mynah_seanet_downsample_reset(mynah_seanet_downsample *down) {
    if (down == NULL) return;
    mynah_causal_conv1d_reset(&down->conv);
}

int mynah_seanet_downsample_apply(mynah_seanet_downsample *down,
                                  const mynah_conv_weights *weights,
                                  const float *input, size_t in_len,
                                  float *output) {
    if (down == NULL) return -1;
    return mynah_causal_conv1d_apply(&down->conv, weights, input, in_len,
                                     output);
}

/* -------------------------------------------------------------- self test */

static float sea_fake(size_t index, size_t salt) {
    const double v = sin((double)(index * 11u + salt * 7u + 1u) * 0.41) * 0.6 +
                     cos((double)(index * 5u + salt * 3u + 2u) * 0.17) * 0.3;
    return (float)v;
}

/* One-shot causal conv reference, written straight off StreamingConv1d with
 * explicit left padding.  Double precision, no shared code. */
static void sea_ref_conv1d(const mynah_conv1d_spec *spec, const float *weight,
                           const float *bias, const float *input, size_t in_len,
                           double *padded, double *out) {
    const size_t eff = (spec->kernel_size - 1u) * spec->dilation + 1u;
    const size_t tail = eff - spec->stride;
    const size_t total = tail + in_len;
    for (size_t c = 0; c < spec->in_channels; ++c) {
        const double pad = (spec->pad_mode == MYNAH_CONV_PAD_REPLICATE)
                               ? (double)input[c * in_len]
                               : 0.0;
        for (size_t i = 0; i < tail; ++i) padded[c * total + i] = pad;
        for (size_t i = 0; i < in_len; ++i) {
            padded[c * total + tail + i] = (double)input[c * in_len + i];
        }
    }
    const size_t out_len = in_len / spec->stride;
    const size_t in_per_group = spec->in_channels / spec->groups;
    const size_t out_per_group = spec->out_channels / spec->groups;
    for (size_t oc = 0; oc < spec->out_channels; ++oc) {
        const size_t group = oc / out_per_group;
        for (size_t n = 0; n < out_len; ++n) {
            double acc = (bias != NULL) ? (double)bias[oc] : 0.0;
            for (size_t j = 0; j < in_per_group; ++j) {
                for (size_t k = 0; k < spec->kernel_size; ++k) {
                    const double w =
                        (double)weight[(oc * in_per_group + j) *
                                           spec->kernel_size +
                                       k];
                    acc += w * padded[(group * in_per_group + j) * total +
                                      n * spec->stride + k * spec->dilation];
                }
            }
            out[oc * out_len + n] = acc;
        }
    }
}

/* One-shot causal transposed conv reference: full convolution, trailing
 * (kernel - stride) samples removed. */
static void sea_ref_convtr1d(const mynah_convtr1d_spec *spec,
                             const float *weight, const float *bias,
                             const float *input, size_t in_len, double *full,
                             double *out) {
    const size_t full_len = (in_len - 1u) * spec->stride + spec->kernel_size;
    const size_t in_per_group = spec->in_channels / spec->groups;
    const size_t out_per_group = spec->out_channels / spec->groups;
    for (size_t oc = 0; oc < spec->out_channels; ++oc) {
        const double b = (bias != NULL) ? (double)bias[oc] : 0.0;
        for (size_t i = 0; i < full_len; ++i) full[oc * full_len + i] = b;
    }
    for (size_t ic = 0; ic < spec->in_channels; ++ic) {
        const size_t group = ic / in_per_group;
        for (size_t j = 0; j < out_per_group; ++j) {
            const size_t oc = group * out_per_group + j;
            for (size_t t = 0; t < in_len; ++t) {
                const double value = (double)input[ic * in_len + t];
                for (size_t k = 0; k < spec->kernel_size; ++k) {
                    const double w =
                        (double)weight[(ic * out_per_group + j) *
                                           spec->kernel_size +
                                       k];
                    full[oc * full_len + t * spec->stride + k] += w * value;
                }
            }
        }
    }
    const size_t out_len = in_len * spec->stride;
    for (size_t oc = 0; oc < spec->out_channels; ++oc) {
        for (size_t i = 0; i < out_len; ++i) {
            out[oc * out_len + i] = full[oc * full_len + i];
        }
    }
}

static int sea_close(float a, double b, double tolerance) {
    const double d = (double)a - b;
    return (d < 0 ? -d : d) <= tolerance;
}

static int sea_test_one_conv(const mynah_conv1d_spec *spec, size_t in_len,
                             size_t salt, int with_bias, const char *label,
                             char *error, size_t error_capacity) {
    const size_t eff = (spec->kernel_size - 1u) * spec->dilation + 1u;
    const size_t tail = eff - spec->stride;
    const size_t out_len = in_len / spec->stride;
    const size_t weight_count =
        spec->out_channels * (spec->in_channels / spec->groups) *
        spec->kernel_size;

    float *weight = calloc(weight_count, sizeof(float));
    float *bias = calloc(spec->out_channels, sizeof(float));
    float *input = calloc(spec->in_channels * in_len, sizeof(float));
    float *got = calloc(spec->out_channels * out_len, sizeof(float));
    double *padded =
        calloc(spec->in_channels * (tail + in_len) + 1u, sizeof(double));
    double *want = calloc(spec->out_channels * out_len, sizeof(double));
    float *scratch = NULL;
    int rc = -1;

    if (weight == NULL || bias == NULL || input == NULL || got == NULL ||
        padded == NULL || want == NULL) {
        sea_set_error(error, error_capacity, "%s: out of memory", label);
        goto fail;
    }
    for (size_t i = 0; i < weight_count; ++i) weight[i] = sea_fake(i, salt);
    for (size_t i = 0; i < spec->out_channels; ++i) {
        bias[i] = with_bias ? sea_fake(i, salt + 1u) : 0.0f;
    }
    for (size_t i = 0; i < spec->in_channels * in_len; ++i) {
        input[i] = sea_fake(i, salt + 2u);
    }

    const size_t need = mynah_causal_conv1d_scratch(spec, in_len);
    scratch = calloc(need ? need : 1u, sizeof(float));
    if (scratch == NULL) {
        sea_set_error(error, error_capacity, "%s: out of memory", label);
        goto fail;
    }

    mynah_conv_weights w;
    w.weight = weight;
    w.bias = with_bias ? bias : NULL;

    mynah_causal_conv1d conv;
    if (mynah_causal_conv1d_init(&conv, spec, in_len, scratch, need, error,
                                 error_capacity) != 0) {
        goto fail;
    }
    if (mynah_causal_conv1d_apply(&conv, &w, input, in_len, got) != 0) {
        sea_set_error(error, error_capacity, "%s: apply failed", label);
        goto fail;
    }
    sea_ref_conv1d(spec, weight, w.bias, input, in_len, padded, want);
    for (size_t i = 0; i < spec->out_channels * out_len; ++i) {
        if (!sea_close(got[i], want[i], 1e-5)) {
            sea_set_error(error, error_capacity,
                          "%s: element %zu = %.9g want %.9g", label,
                          i, (double)got[i], want[i]);
            goto fail;
        }
    }

    /* Streaming continuity: chunked decode must equal the one-shot decode. */
    {
        const size_t chunk = spec->stride;
        if (in_len % chunk == 0 &&
            (spec->pad_mode != MYNAH_CONV_PAD_REPLICATE || chunk >= tail)) {
            float *chunked = calloc(spec->out_channels * out_len,
                                    sizeof(float));
            float *piece_in = calloc(spec->in_channels * chunk, sizeof(float));
            float *piece_out =
                calloc(spec->out_channels * (chunk / spec->stride),
                       sizeof(float));
            if (chunked == NULL || piece_in == NULL || piece_out == NULL) {
                free(chunked);
                free(piece_in);
                free(piece_out);
                sea_set_error(error, error_capacity, "%s: out of memory",
                              label);
                goto fail;
            }
            mynah_causal_conv1d_reset(&conv);
            const size_t piece_out_len = chunk / spec->stride;
            int failed = 0;
            for (size_t off = 0; off < in_len && !failed; off += chunk) {
                for (size_t c = 0; c < spec->in_channels; ++c) {
                    memcpy(piece_in + c * chunk, input + c * in_len + off,
                           chunk * sizeof(float));
                }
                if (mynah_causal_conv1d_apply(&conv, &w, piece_in, chunk,
                                              piece_out) != 0) {
                    failed = 1;
                    break;
                }
                for (size_t c = 0; c < spec->out_channels; ++c) {
                    memcpy(chunked + c * out_len + off / spec->stride,
                           piece_out + c * piece_out_len,
                           piece_out_len * sizeof(float));
                }
            }
            if (!failed) {
                for (size_t i = 0; i < spec->out_channels * out_len; ++i) {
                    if (chunked[i] != got[i]) {
                        failed = 2;
                        sea_set_error(error, error_capacity,
                                      "%s: streaming mismatch at %zu: "
                                      "%.9g vs %.9g",
                                      label, i, (double)chunked[i],
                                      (double)got[i]);
                        break;
                    }
                }
            } else {
                sea_set_error(error, error_capacity, "%s: chunked apply failed",
                              label);
            }
            free(chunked);
            free(piece_in);
            free(piece_out);
            if (failed) goto fail;
        }
    }

    rc = 0;
fail:
    free(weight);
    free(bias);
    free(input);
    free(got);
    free(padded);
    free(want);
    free(scratch);
    return rc;
}

static int sea_test_one_convtr(const mynah_convtr1d_spec *spec, size_t in_len,
                               size_t salt, int with_bias, const char *label,
                               char *error, size_t error_capacity) {
    const size_t out_len = in_len * spec->stride;
    const size_t full_len = (in_len - 1u) * spec->stride + spec->kernel_size;
    const size_t weight_count = spec->in_channels *
                                (spec->out_channels / spec->groups) *
                                spec->kernel_size;
    float *weight = calloc(weight_count, sizeof(float));
    float *bias = calloc(spec->out_channels, sizeof(float));
    float *input = calloc(spec->in_channels * in_len, sizeof(float));
    float *got = calloc(spec->out_channels * out_len, sizeof(float));
    double *full = calloc(spec->out_channels * full_len, sizeof(double));
    double *want = calloc(spec->out_channels * out_len, sizeof(double));
    float *scratch = NULL;
    int rc = -1;

    if (weight == NULL || bias == NULL || input == NULL || got == NULL ||
        full == NULL || want == NULL) {
        sea_set_error(error, error_capacity, "%s: out of memory", label);
        goto fail;
    }
    for (size_t i = 0; i < weight_count; ++i) weight[i] = sea_fake(i, salt);
    for (size_t i = 0; i < spec->out_channels; ++i) {
        bias[i] = with_bias ? sea_fake(i, salt + 1u) : 0.0f;
    }
    for (size_t i = 0; i < spec->in_channels * in_len; ++i) {
        input[i] = sea_fake(i, salt + 2u);
    }

    const size_t need = mynah_causal_convtr1d_scratch(spec, in_len);
    scratch = calloc(need ? need : 1u, sizeof(float));
    if (scratch == NULL) {
        sea_set_error(error, error_capacity, "%s: out of memory", label);
        goto fail;
    }
    mynah_conv_weights w;
    w.weight = weight;
    w.bias = with_bias ? bias : NULL;

    mynah_causal_convtr1d convtr;
    if (mynah_causal_convtr1d_init(&convtr, spec, in_len, scratch, need, error,
                                   error_capacity) != 0) {
        goto fail;
    }
    if (mynah_causal_convtr1d_apply(&convtr, &w, input, in_len, got) != 0) {
        sea_set_error(error, error_capacity, "%s: apply failed", label);
        goto fail;
    }
    sea_ref_convtr1d(spec, weight, w.bias, input, in_len, full, want);
    for (size_t i = 0; i < spec->out_channels * out_len; ++i) {
        if (!sea_close(got[i], want[i], 1e-5)) {
            sea_set_error(error, error_capacity,
                          "%s: element %zu = %.9g want %.9g", label, i,
                          (double)got[i], want[i]);
            goto fail;
        }
    }

    /* Frame-by-frame streaming must reproduce the one-shot result. */
    {
        float *chunked = calloc(spec->out_channels * out_len, sizeof(float));
        float *piece_in = calloc(spec->in_channels, sizeof(float));
        float *piece_out = calloc(spec->out_channels * spec->stride,
                                  sizeof(float));
        if (chunked == NULL || piece_in == NULL || piece_out == NULL) {
            free(chunked);
            free(piece_in);
            free(piece_out);
            sea_set_error(error, error_capacity, "%s: out of memory", label);
            goto fail;
        }
        mynah_causal_convtr1d_reset(&convtr);
        int failed = 0;
        for (size_t t = 0; t < in_len && !failed; ++t) {
            for (size_t c = 0; c < spec->in_channels; ++c) {
                piece_in[c] = input[c * in_len + t];
            }
            if (mynah_causal_convtr1d_apply(&convtr, &w, piece_in, 1u,
                                            piece_out) != 0) {
                failed = 1;
                break;
            }
            for (size_t c = 0; c < spec->out_channels; ++c) {
                memcpy(chunked + c * out_len + t * spec->stride,
                       piece_out + c * spec->stride,
                       spec->stride * sizeof(float));
            }
        }
        if (!failed) {
            for (size_t i = 0; i < spec->out_channels * out_len; ++i) {
                if (!sea_close(chunked[i], (double)got[i], 1e-5)) {
                    failed = 2;
                    sea_set_error(error, error_capacity,
                                  "%s: streaming mismatch at %zu: %.9g vs %.9g",
                                  label, i, (double)chunked[i],
                                  (double)got[i]);
                    break;
                }
            }
        } else {
            sea_set_error(error, error_capacity, "%s: chunked apply failed",
                          label);
        }
        free(chunked);
        free(piece_in);
        free(piece_out);
        if (failed) goto fail;
    }

    rc = 0;
fail:
    free(weight);
    free(bias);
    free(input);
    free(got);
    free(full);
    free(want);
    free(scratch);
    return rc;
}

int mynah_seanet_self_test(char *error, size_t error_capacity) {
    if (error != NULL && error_capacity > 0) error[0] = '\0';

    /* --- ELU --------------------------------------------------------- */
    {
        const float in[4] = {1.5f, 0.0f, -1.0f, -3.0f};
        const double want[4] = {1.5, 0.0, -0.6321205588285577,
                                -0.950212931632136};
        float got[4];
        mynah_seanet_elu_f32(in, got, 4u, 1.0f);
        for (size_t i = 0; i < 4u; ++i) {
            if (!sea_close(got[i], want[i], 1e-6)) {
                sea_set_error(error, error_capacity,
                              "elu[%zu] = %.9g want %.9g", i, (double)got[i],
                              want[i]);
                return -1;
            }
        }
        float half[2] = {-1.0f, 2.0f};
        mynah_seanet_elu_f32(half, half, 2u, 0.5f);
        if (!sea_close(half[0], -0.5 * 0.6321205588285577, 1e-6) ||
            half[1] != 2.0f) {
            sea_set_error(error, error_capacity, "elu alpha not honoured");
            return -1;
        }
    }

    /* --- causal conv1d, every shape that matters --------------------- */
    {
        struct {
            mynah_conv1d_spec spec;
            size_t in_len;
            int bias;
            const char *label;
        } cases[] = {
            {sea_conv_spec(3u, 2u, 3u, 1u, 1u, 1u, MYNAH_CONV_PAD_ZERO), 8u, 1,
             "conv k3"},
            {sea_conv_spec(2u, 3u, 1u, 1u, 1u, 1u, MYNAH_CONV_PAD_ZERO), 6u, 1,
             "conv k1 (no state)"},
            {sea_conv_spec(2u, 2u, 3u, 1u, 2u, 1u, MYNAH_CONV_PAD_ZERO), 9u, 1,
             "conv dilated"},
            {sea_conv_spec(4u, 4u, 4u, 2u, 1u, 1u, MYNAH_CONV_PAD_ZERO), 12u, 1,
             "conv strided"},
            {sea_conv_spec(4u, 4u, 3u, 1u, 1u, 4u, MYNAH_CONV_PAD_ZERO), 8u, 0,
             "conv depthwise"},
            {sea_conv_spec(3u, 2u, 6u, 3u, 1u, 1u, MYNAH_CONV_PAD_REPLICATE),
             12u, 0, "conv replicate"},
        };
        for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
            if (sea_test_one_conv(&cases[i].spec, cases[i].in_len, i + 1u,
                                  cases[i].bias, cases[i].label, error,
                                  error_capacity) != 0) {
                return -1;
            }
        }
    }

    /* --- causal transposed conv, dense and depthwise ------------------ */
    {
        mynah_convtr1d_spec dense = {3u, 2u, 4u, 2u, 1u};
        mynah_convtr1d_spec depthwise = {4u, 4u, 8u, 4u, 4u};
        mynah_convtr1d_spec mimi_like = {6u, 6u, 8u, 4u, 6u};
        if (sea_test_one_convtr(&dense, 5u, 11u, 1, "convtr dense", error,
                                error_capacity) != 0) {
            return -1;
        }
        if (sea_test_one_convtr(&depthwise, 4u, 13u, 0, "convtr depthwise",
                                error, error_capacity) != 0) {
            return -1;
        }
        if (sea_test_one_convtr(&mimi_like, 3u, 17u, 0,
                                "convtr depthwise (upsample shape)", error,
                                error_capacity) != 0) {
            return -1;
        }
    }

    /* --- a whole small SEANet decoder: one shot vs chunked ------------ */
    {
        static const size_t ratios[2] = {2u, 2u};
        mynah_seanet_config config;
        config.channels = 1u;
        config.dimension = 8u;
        config.n_filters = 2u;
        config.n_residual_layers = 1u;
        config.ratios = ratios;
        config.n_ratios = 2u;
        config.kernel_size = 3u;
        config.residual_kernel_size = 3u;
        config.last_kernel_size = 3u;
        config.dilation_base = 2u;
        config.compress = 2u;
        config.elu_alpha = 1.0f;

        mynah_resample_config up;
        up.stride = 2u;
        up.in_channels = 8u;
        up.out_channels = 8u;
        up.groups = 8u;

        const size_t max_latent = 4u;
        const size_t hop = 4u;          /* prod(ratios) */
        const size_t enc_frames = 8u;   /* 4 latent frames * stride 2 */
        const size_t out_len = enc_frames * hop;

        char local[256];
        mynah_seanet_state *one =
            mynah_seanet_state_create(&config, &up, max_latent, local,
                                      sizeof(local));
        mynah_seanet_state *many =
            mynah_seanet_state_create(&config, &up, max_latent, local,
                                      sizeof(local));
        float *input = calloc(config.dimension * enc_frames, sizeof(float));
        float *out_one = calloc(out_len, sizeof(float));
        float *out_many = calloc(out_len, sizeof(float));
        float *piece = calloc(config.dimension * 2u, sizeof(float));
        float *piece_out = calloc(2u * hop, sizeof(float));
        /* decoder weights */
        const size_t mult = 4u; /* 2^n_ratios */
        const size_t c0 = mult * config.n_filters; /* 8 */
        float *w_first = calloc(c0 * config.dimension * config.kernel_size,
                                sizeof(float));
        float *b_first = calloc(c0, sizeof(float));
        float *w_last =
            calloc(config.channels * config.n_filters * config.last_kernel_size,
                   sizeof(float));
        float *b_last = calloc(config.channels, sizeof(float));
        /* stage 0: 8 -> 4, stage 1: 4 -> 2 */
        float *w_ct0 = calloc(8u * 4u * 4u, sizeof(float));
        float *b_ct0 = calloc(4u, sizeof(float));
        float *w_ct1 = calloc(4u * 2u * 4u, sizeof(float));
        float *b_ct1 = calloc(2u, sizeof(float));
        float *w_b0a = calloc(2u * 4u * 3u, sizeof(float)); /* 4 -> 2, k3 */
        float *b_b0a = calloc(2u, sizeof(float));
        float *w_b0b = calloc(4u * 2u * 1u, sizeof(float)); /* 2 -> 4, k1 */
        float *b_b0b = calloc(4u, sizeof(float));
        float *w_b1a = calloc(1u * 2u * 3u, sizeof(float)); /* 2 -> 1, k3 */
        float *b_b1a = calloc(1u, sizeof(float));
        float *w_b1b = calloc(2u * 1u * 1u, sizeof(float)); /* 1 -> 2, k1 */
        float *b_b1b = calloc(2u, sizeof(float));
        int rc = -1;

        if (one == NULL || many == NULL) {
            sea_set_error(error, error_capacity, "seanet decoder create: %s",
                          local);
            goto decoder_done;
        }
        if (input == NULL || out_one == NULL || out_many == NULL ||
            piece == NULL || piece_out == NULL || w_first == NULL ||
            b_first == NULL || w_last == NULL || b_last == NULL ||
            w_ct0 == NULL || b_ct0 == NULL || w_ct1 == NULL || b_ct1 == NULL ||
            w_b0a == NULL || b_b0a == NULL || w_b0b == NULL || b_b0b == NULL ||
            w_b1a == NULL || b_b1a == NULL || w_b1b == NULL || b_b1b == NULL) {
            sea_set_error(error, error_capacity, "seanet decoder: out of "
                                                 "memory");
            goto decoder_done;
        }

#define SEA_FILL(PTR, COUNT, SALT)                            \
    for (size_t fi = 0; fi < (COUNT); ++fi)                   \
        (PTR)[fi] = sea_fake(fi, (SALT));
        SEA_FILL(input, config.dimension * enc_frames, 21u)
        SEA_FILL(w_first, c0 * config.dimension * config.kernel_size, 22u)
        SEA_FILL(b_first, c0, 23u)
        SEA_FILL(w_last,
                 config.channels * config.n_filters * config.last_kernel_size,
                 24u)
        SEA_FILL(b_last, config.channels, 25u)
        SEA_FILL(w_ct0, 8u * 4u * 4u, 26u)
        SEA_FILL(b_ct0, 4u, 27u)
        SEA_FILL(w_ct1, 4u * 2u * 4u, 28u)
        SEA_FILL(b_ct1, 2u, 29u)
        SEA_FILL(w_b0a, 2u * 4u * 3u, 30u)
        SEA_FILL(b_b0a, 2u, 31u)
        SEA_FILL(w_b0b, 4u * 2u, 32u)
        SEA_FILL(b_b0b, 4u, 33u)
        SEA_FILL(w_b1a, 1u * 2u * 3u, 34u)
        SEA_FILL(b_b1a, 1u, 35u)
        SEA_FILL(w_b1b, 2u * 1u, 36u)
        SEA_FILL(b_b1b, 2u, 37u)
#undef SEA_FILL

        mynah_conv_weights convtr_w[2];
        convtr_w[0].weight = w_ct0;
        convtr_w[0].bias = b_ct0;
        convtr_w[1].weight = w_ct1;
        convtr_w[1].bias = b_ct1;
        mynah_seanet_resblock_weights blocks[2];
        blocks[0].conv1.weight = w_b0a;
        blocks[0].conv1.bias = b_b0a;
        blocks[0].conv2.weight = w_b0b;
        blocks[0].conv2.bias = b_b0b;
        blocks[1].conv1.weight = w_b1a;
        blocks[1].conv1.bias = b_b1a;
        blocks[1].conv2.weight = w_b1b;
        blocks[1].conv2.bias = b_b1b;

        mynah_seanet_decoder_weights dw;
        dw.first.weight = w_first;
        dw.first.bias = b_first;
        dw.convtr = convtr_w;
        dw.blocks = blocks;
        dw.last.weight = w_last;
        dw.last.bias = b_last;

        if (mynah_seanet_check_decoder_weights(one, &dw, error,
                                               error_capacity) != 0) {
            goto decoder_done;
        }
        if (mynah_seanet_state_hop_length(one) != hop ||
            mynah_seanet_state_encoder_stride(one) != 2u ||
            mynah_seanet_state_samples_per_latent(one) != 2u * hop) {
            sea_set_error(error, error_capacity,
                          "seanet geometry wrong: hop=%zu stride=%zu",
                          mynah_seanet_state_hop_length(one),
                          mynah_seanet_state_encoder_stride(one));
            goto decoder_done;
        }
        if (mynah_seanet_decode(one, &dw, input, enc_frames, out_one) != 0) {
            sea_set_error(error, error_capacity, "seanet one-shot decode "
                                                 "failed");
            goto decoder_done;
        }
        /* Chunked: one latent frame (2 encoder frames) at a time. */
        {
            int failed = 0;
            for (size_t off = 0; off < enc_frames; off += 2u) {
                for (size_t c = 0; c < config.dimension; ++c) {
                    memcpy(piece + c * 2u, input + c * enc_frames + off,
                           2u * sizeof(float));
                }
                if (mynah_seanet_decode(many, &dw, piece, 2u, piece_out) != 0) {
                    failed = 1;
                    break;
                }
                memcpy(out_many + off * hop, piece_out, 2u * hop *
                                                            sizeof(float));
                mynah_seanet_state_advance(many, 1u);
            }
            if (failed) {
                sea_set_error(error, error_capacity,
                              "seanet chunked decode failed");
                goto decoder_done;
            }
        }
        for (size_t i = 0; i < out_len; ++i) {
            if (!sea_close(out_many[i], (double)out_one[i], 1e-4)) {
                sea_set_error(error, error_capacity,
                              "seanet streaming mismatch at %zu: %.9g vs %.9g",
                              i, (double)out_many[i], (double)out_one[i]);
                goto decoder_done;
            }
        }
        /* The position counter is the other half of the state. */
        if (mynah_seanet_state_position(many) != 4u * 2u) {
            sea_set_error(error, error_capacity,
                          "position counter = %zu, expected %u",
                          mynah_seanet_state_position(many), 8u);
            goto decoder_done;
        }
        if (mynah_seanet_state_position(one) != 0) {
            sea_set_error(error, error_capacity,
                          "position counter advanced without a call");
            goto decoder_done;
        }
        mynah_seanet_state_reset(many);
        if (mynah_seanet_state_position(many) != 0) {
            sea_set_error(error, error_capacity,
                          "reset left the position counter at %zu",
                          mynah_seanet_state_position(many));
            goto decoder_done;
        }

        /* Upsample: frame-by-frame must match the batched call. */
        {
            float *u_in = calloc(up.in_channels * 4u, sizeof(float));
            float *u_batch = calloc(up.out_channels * 4u * up.stride,
                                    sizeof(float));
            float *u_step = calloc(up.out_channels * 4u * up.stride,
                                   sizeof(float));
            float *u_piece_in = calloc(up.in_channels, sizeof(float));
            float *u_piece_out = calloc(up.out_channels * up.stride,
                                        sizeof(float));
            float *u_w = calloc(up.in_channels * (up.out_channels / up.groups) *
                                    up.stride * 2u,
                                sizeof(float));
            int ufail = 0;
            if (u_in == NULL || u_batch == NULL || u_step == NULL ||
                u_piece_in == NULL || u_piece_out == NULL || u_w == NULL) {
                sea_set_error(error, error_capacity, "upsample: out of memory");
                ufail = 1;
            } else {
                for (size_t i = 0; i < up.in_channels * 4u; ++i) {
                    u_in[i] = sea_fake(i, 41u);
                }
                const size_t uw =
                    up.in_channels * (up.out_channels / up.groups) *
                    up.stride * 2u;
                for (size_t i = 0; i < uw; ++i) u_w[i] = sea_fake(i, 42u);
                mynah_conv_weights uww;
                uww.weight = u_w;
                uww.bias = NULL;
                mynah_seanet_state_reset(one);
                if (mynah_seanet_upsample(one, &uww, u_in, 4u, u_batch) != 0) {
                    sea_set_error(error, error_capacity,
                                  "upsample batched failed");
                    ufail = 1;
                }
                if (!ufail) {
                    mynah_seanet_state_reset(many);
                    for (size_t t = 0; t < 4u && !ufail; ++t) {
                        for (size_t c = 0; c < up.in_channels; ++c) {
                            u_piece_in[c] = u_in[c * 4u + t];
                        }
                        if (mynah_seanet_upsample(many, &uww, u_piece_in, 1u,
                                                  u_piece_out) != 0) {
                            sea_set_error(error, error_capacity,
                                          "upsample step failed");
                            ufail = 1;
                            break;
                        }
                        for (size_t c = 0; c < up.out_channels; ++c) {
                            memcpy(u_step + c * 4u * up.stride +
                                       t * up.stride,
                                   u_piece_out + c * up.stride,
                                   up.stride * sizeof(float));
                        }
                    }
                }
                if (!ufail) {
                    for (size_t i = 0; i < up.out_channels * 4u * up.stride;
                         ++i) {
                        if (!sea_close(u_step[i], (double)u_batch[i], 1e-5)) {
                            sea_set_error(error, error_capacity,
                                          "upsample streaming mismatch at %zu: "
                                          "%.9g vs %.9g",
                                          i, (double)u_step[i],
                                          (double)u_batch[i]);
                            ufail = 1;
                            break;
                        }
                    }
                }
            }
            free(u_in);
            free(u_batch);
            free(u_step);
            free(u_piece_in);
            free(u_piece_out);
            free(u_w);
            if (ufail) goto decoder_done;
        }

        rc = 0;
    decoder_done:
        mynah_seanet_state_destroy(one);
        mynah_seanet_state_destroy(many);
        free(input);
        free(out_one);
        free(out_many);
        free(piece);
        free(piece_out);
        free(w_first);
        free(b_first);
        free(w_last);
        free(b_last);
        free(w_ct0);
        free(b_ct0);
        free(w_ct1);
        free(b_ct1);
        free(w_b0a);
        free(b_b0a);
        free(w_b0b);
        free(b_b0b);
        free(w_b1a);
        free(b_b1a);
        free(w_b1b);
        free(b_b1b);
        if (rc != 0) return -1;
    }

    /* --- downsample: replicate padding, streaming ------------------- */
    {
        mynah_resample_config down_cfg;
        down_cfg.stride = 2u;
        down_cfg.in_channels = 3u;
        down_cfg.out_channels = 2u;
        down_cfg.groups = 1u;
        char local[256];
        mynah_seanet_downsample *down =
            mynah_seanet_downsample_create(&down_cfg, 8u, local, sizeof(local));
        if (down == NULL) {
            sea_set_error(error, error_capacity, "downsample create: %s",
                          local);
            return -1;
        }
        const size_t kernel = down_cfg.stride * 2u;
        const size_t wc = down_cfg.out_channels * down_cfg.in_channels * kernel;
        float *w = calloc(wc, sizeof(float));
        float *input = calloc(down_cfg.in_channels * 8u, sizeof(float));
        float *got = calloc(down_cfg.out_channels * 4u, sizeof(float));
        double *padded = calloc(down_cfg.in_channels * (2u + 8u),
                                sizeof(double));
        double *want = calloc(down_cfg.out_channels * 4u, sizeof(double));
        int ok = (w != NULL && input != NULL && got != NULL &&
                  padded != NULL && want != NULL);
        if (ok) {
            for (size_t i = 0; i < wc; ++i) w[i] = sea_fake(i, 51u);
            for (size_t i = 0; i < down_cfg.in_channels * 8u; ++i) {
                input[i] = sea_fake(i, 52u);
            }
            mynah_conv_weights dw;
            dw.weight = w;
            dw.bias = NULL;
            if (mynah_seanet_downsample_apply(down, &dw, input, 8u, got) != 0) {
                sea_set_error(error, error_capacity, "downsample apply failed");
                ok = 0;
            }
            if (ok) {
                mynah_conv1d_spec spec = sea_conv_spec(
                    down_cfg.in_channels, down_cfg.out_channels, kernel,
                    down_cfg.stride, 1u, 1u, MYNAH_CONV_PAD_REPLICATE);
                sea_ref_conv1d(&spec, w, NULL, input, 8u, padded, want);
                for (size_t i = 0; i < down_cfg.out_channels * 4u; ++i) {
                    if (!sea_close(got[i], want[i], 1e-5)) {
                        sea_set_error(error, error_capacity,
                                      "downsample[%zu] = %.9g want %.9g", i,
                                      (double)got[i], want[i]);
                        ok = 0;
                        break;
                    }
                }
            }
        } else {
            sea_set_error(error, error_capacity, "downsample: out of memory");
        }
        mynah_seanet_downsample_destroy(down);
        free(w);
        free(input);
        free(got);
        free(padded);
        free(want);
        if (!ok) return -1;
    }

    return 0;
}
