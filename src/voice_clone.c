/*
 * PocketTTS voice cloning (E7).  See voice_clone.h for the path this walks and
 * for why consent is a parameter here rather than a note in the CLI.
 *
 * Layout convention, as in src/seanet.c: convolutions are channel major,
 * `[channels][length]`, i.e. torch `[1, C, T]`; the transformer is row major,
 * `[positions][d_model]`.  The two transposes between them are the only place
 * this file moves data for its own convenience.
 */
#include "voice_clone.h"

#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "kernels.h"

/* ------------------------------------------------------------------ utils */

static void vc_err(char *error, size_t capacity, const char *format, ...)
    __attribute__((format(printf, 3, 4)));

static void vc_err(char *error, size_t capacity, const char *format, ...) {
    if (error == NULL || capacity == 0) return;
    va_list args;
    va_start(args, format);
    (void)vsnprintf(error, capacity, format, args);
    va_end(args);
}

static int vc_mul(size_t a, size_t b, size_t *out) {
    if (a != 0 && b > (size_t)-1 / a) return -1;
    *out = a * b;
    return 0;
}

static int vc_add(size_t a, size_t b, size_t *out) {
    if (b > (size_t)-1 - a) return -1;
    *out = a + b;
    return 0;
}

static size_t vc_max(size_t a, size_t b) { return (a > b) ? a : b; }

static float *vc_alloc(size_t count, char *error, size_t capacity,
                       const char *what) {
    size_t bytes = 0;
    if (count == 0 || vc_mul(count, sizeof(float), &bytes) != 0) {
        vc_err(error, capacity, "voice clone: %s size overflow", what);
        return NULL;
    }
    float *data = (float *)calloc(count, sizeof(float));
    if (data == NULL) {
        vc_err(error, capacity, "voice clone: out of memory for %s", what);
    }
    return data;
}

/* ---------------------------------------------------------------- consent */

const char *const mynah_voice_clone_consent_notice =
    "Voice cloning requires the explicit and lawful consent of the person "
    "whose voice is being cloned. Affirm consent explicitly; there is no "
    "default that clones silently.";

int mynah_voice_clone_consent_check(const mynah_voice_clone_consent *consent,
                                    char *error, size_t error_capacity) {
    if (consent == NULL || consent->affirmed == 0) {
        vc_err(error, error_capacity, "%s", mynah_voice_clone_consent_notice);
        return -1;
    }
    return 0;
}

/* ------------------------------------------------------------ WAV reading */

void mynah_audio_clip_free(mynah_audio_clip *clip) {
    if (clip == NULL) return;
    free(clip->samples);
    clip->samples = NULL;
    clip->count = 0;
    clip->sample_rate = 0;
}

void mynah_audio_clip_truncate(mynah_audio_clip *clip, double seconds) {
    if (clip == NULL || clip->samples == NULL || seconds <= 0.0) return;
    if (clip->sample_rate == 0) return;
    const double limit = seconds * (double)clip->sample_rate;
    if (limit >= (double)clip->count) return;
    clip->count = (size_t)limit;
}

static uint16_t vc_le16(const unsigned char *p) {
    return (uint16_t)((uint16_t)p[0] | ((uint16_t)p[1] << 8));
}

static uint32_t vc_le32(const unsigned char *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) |
           ((uint32_t)p[3] << 24);
}

static float vc_decode_sample(const unsigned char *p, unsigned format,
                              unsigned bits) {
    if (format == 3u) {
        if (bits == 32u) {
            const uint32_t bitsr = vc_le32(p);
            float value = 0.0f;
            memcpy(&value, &bitsr, sizeof(value));
            return value;
        }
        uint64_t raw = 0;
        for (size_t i = 0; i < 8u; ++i) raw |= (uint64_t)p[i] << (8u * i);
        double value = 0.0;
        memcpy(&value, &raw, sizeof(value));
        return (float)value;
    }
    switch (bits) {
        case 8u:
            /* 8-bit WAV PCM is unsigned with a 128 bias. */
            return ((float)p[0] - 128.0f) / 128.0f;
        case 16u: {
            const int16_t v = (int16_t)vc_le16(p);
            return (float)v / 32768.0f;
        }
        case 24u: {
            int32_t v = (int32_t)((uint32_t)p[0] | ((uint32_t)p[1] << 8) |
                                  ((uint32_t)p[2] << 16));
            if (v & 0x800000) v -= 0x1000000;
            return (float)v / 8388608.0f;
        }
        default: {
            const int32_t v = (int32_t)vc_le32(p);
            return (float)v / 2147483648.0f;
        }
    }
}

int mynah_wav_decode_mono(const void *bytes, size_t size,
                          mynah_audio_clip *out, char *error,
                          size_t error_capacity) {
    if (error != NULL && error_capacity > 0) error[0] = '\0';
    if (out == NULL) {
        vc_err(error, error_capacity, "wav: null output");
        return -1;
    }
    memset(out, 0, sizeof(*out));
    const unsigned char *data = (const unsigned char *)bytes;
    if (data == NULL || size < 44u) {
        vc_err(error, error_capacity, "wav: too short (%zu bytes)", size);
        return -1;
    }
    if (memcmp(data, "RIFF", 4) != 0 || memcmp(data + 8, "WAVE", 4) != 0) {
        vc_err(error, error_capacity, "wav: not a RIFF/WAVE file");
        return -1;
    }

    unsigned format = 0;
    unsigned channels = 0;
    unsigned rate = 0;
    unsigned bits = 0;
    int have_fmt = 0;
    const unsigned char *payload = NULL;
    size_t payload_size = 0;

    size_t at = 12u;
    while (at + 8u <= size) {
        const unsigned char *id = data + at;
        const size_t chunk = (size_t)vc_le32(data + at + 4u);
        const size_t body = at + 8u;
        if (chunk > size - body) {
            /* A truncated final chunk: take what is there rather than fail,
             * which is what every player does with a stream cut short. */
            if (memcmp(id, "data", 4) == 0 && have_fmt) {
                payload = data + body;
                payload_size = size - body;
                break;
            }
            vc_err(error, error_capacity,
                   "wav: chunk '%c%c%c%c' claims %zu bytes past the end",
                   id[0], id[1], id[2], id[3], chunk);
            return -1;
        }
        if (memcmp(id, "fmt ", 4) == 0) {
            if (chunk < 16u) {
                vc_err(error, error_capacity, "wav: fmt chunk is %zu bytes",
                       chunk);
                return -1;
            }
            format = vc_le16(data + body);
            channels = vc_le16(data + body + 2u);
            rate = vc_le32(data + body + 4u);
            bits = vc_le16(data + body + 14u);
            if (format == 0xFFFEu) {
                if (chunk < 40u) {
                    vc_err(error, error_capacity,
                           "wav: extensible fmt chunk is %zu bytes", chunk);
                    return -1;
                }
                /* The sub-format GUID starts with the real format tag. */
                format = vc_le16(data + body + 24u);
            }
            have_fmt = 1;
        } else if (memcmp(id, "data", 4) == 0) {
            payload = data + body;
            payload_size = chunk;
            if (have_fmt) break;
        }
        at = body + chunk + (chunk & 1u); /* chunks are word aligned */
    }

    if (!have_fmt || payload == NULL) {
        vc_err(error, error_capacity, "wav: missing %s chunk",
               have_fmt ? "data" : "fmt ");
        return -1;
    }
    if (format != 1u && format != 3u) {
        vc_err(error, error_capacity, "wav: unsupported format tag %u", format);
        return -1;
    }
    if (format == 1u && bits != 8u && bits != 16u && bits != 24u &&
        bits != 32u) {
        vc_err(error, error_capacity, "wav: unsupported PCM depth %u", bits);
        return -1;
    }
    if (format == 3u && bits != 32u && bits != 64u) {
        vc_err(error, error_capacity, "wav: unsupported float depth %u", bits);
        return -1;
    }
    if (channels == 0u || rate == 0u) {
        vc_err(error, error_capacity, "wav: %u channels at %u Hz", channels,
               rate);
        return -1;
    }

    const size_t width = bits / 8u;
    size_t stride = 0;
    if (vc_mul(width, channels, &stride) != 0 || stride == 0) {
        vc_err(error, error_capacity, "wav: bad frame size");
        return -1;
    }
    const size_t frames = payload_size / stride;
    if (frames == 0) {
        vc_err(error, error_capacity, "wav: no audio frames");
        return -1;
    }
    float *samples = vc_alloc(frames, error, error_capacity, "wav samples");
    if (samples == NULL) return -1;

    const float scale = 1.0f / (float)channels;
    for (size_t i = 0; i < frames; ++i) {
        const unsigned char *frame = payload + i * stride;
        float sum = 0.0f;
        for (size_t c = 0; c < channels; ++c) {
            sum += vc_decode_sample(frame + c * width, format, bits);
        }
        samples[i] = sum * scale;
    }
    out->samples = samples;
    out->count = frames;
    out->sample_rate = rate;
    return 0;
}

int mynah_wav_read_mono(const char *path, mynah_audio_clip *out, char *error,
                        size_t error_capacity) {
    if (error != NULL && error_capacity > 0) error[0] = '\0';
    if (path == NULL || out == NULL) {
        vc_err(error, error_capacity, "wav: null path or output");
        return -1;
    }
    memset(out, 0, sizeof(*out));
    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        vc_err(error, error_capacity, "wav: cannot open %s", path);
        return -1;
    }
    if (fseek(file, 0, SEEK_END) != 0) {
        fclose(file);
        vc_err(error, error_capacity, "wav: cannot seek %s", path);
        return -1;
    }
    const long end = ftell(file);
    if (end <= 0) {
        fclose(file);
        vc_err(error, error_capacity, "wav: %s is empty", path);
        return -1;
    }
    rewind(file);
    const size_t size = (size_t)end;
    unsigned char *bytes = (unsigned char *)malloc(size);
    if (bytes == NULL) {
        fclose(file);
        vc_err(error, error_capacity, "wav: out of memory for %s", path);
        return -1;
    }
    const size_t got = fread(bytes, 1, size, file);
    fclose(file);
    if (got != size) {
        free(bytes);
        vc_err(error, error_capacity, "wav: short read on %s", path);
        return -1;
    }
    const int status = mynah_wav_decode_mono(bytes, size, out, error,
                                             error_capacity);
    free(bytes);
    return status;
}

/* -------------------------------------------------------------- resampler */

/* Modified Bessel function of the first kind, order 0, by its power series.
 * The Kaiser beta here is 5, so the argument never exceeds 5 and the series
 * converges in a couple of dozen terms in double precision. */
static double vc_bessel_i0(double x) {
    const double half = x * 0.5;
    double term = 1.0;
    double sum = 1.0;
    for (int k = 1; k < 64; ++k) {
        term *= (half / (double)k);
        const double contribution = term * term;
        sum += contribution;
        if (contribution < 1e-18 * sum) break;
    }
    return sum;
}

static double vc_sinc(double x) {
    if (x == 0.0) return 1.0;
    const double pix = 3.14159265358979323846 * x;
    return sin(pix) / pix;
}

/*
 * `scipy.signal.firwin(numtaps, cutoff, window=('kaiser', 5.0))`: a windowed
 * sinc scaled so the DC gain is exactly one.  Computed in double and rounded
 * to float once, which is what scipy does (`firwin(...).astype(x.dtype)`).
 */
static int vc_firwin_kaiser(size_t numtaps, double cutoff, double beta,
                            float *taps) {
    if (numtaps == 0 || (numtaps % 2u) == 0) return -1;
    const double alpha = 0.5 * (double)(numtaps - 1u);
    const double denom = vc_bessel_i0(beta);
    if (!(denom > 0.0)) return -1;
    double *work = (double *)calloc(numtaps, sizeof(double));
    if (work == NULL) return -1;
    double sum = 0.0;
    for (size_t n = 0; n < numtaps; ++n) {
        const double m = (double)n - alpha;
        const double ratio = m / alpha;
        double inside = 1.0 - ratio * ratio;
        if (inside < 0.0) inside = 0.0;
        const double window = vc_bessel_i0(beta * sqrt(inside)) / denom;
        work[n] = cutoff * vc_sinc(cutoff * m) * window;
        sum += work[n];
    }
    if (!(fabs(sum) > 0.0)) {
        free(work);
        return -1;
    }
    for (size_t n = 0; n < numtaps; ++n) taps[n] = (float)(work[n] / sum);
    free(work);
    return 0;
}

static size_t vc_gcd(size_t a, size_t b) {
    while (b != 0) {
        const size_t t = a % b;
        a = b;
        b = t;
    }
    return a;
}

/* scipy's `_output_len`: the length `upfirdn` produces. */
static size_t vc_upfirdn_len(size_t taps, size_t count, size_t up,
                             size_t down) {
    if (count == 0) return 0;
    return ((count - 1u) * up + taps - 1u) / down + 1u;
}

int mynah_audio_resample_poly(const float *input, size_t count, size_t up,
                              size_t down, float **out, size_t *out_count,
                              char *error, size_t error_capacity) {
    if (error != NULL && error_capacity > 0) error[0] = '\0';
    if (input == NULL || out == NULL || out_count == NULL) {
        vc_err(error, error_capacity, "resample: null argument");
        return -1;
    }
    *out = NULL;
    *out_count = 0;
    if (up == 0 || down == 0 || count == 0) {
        vc_err(error, error_capacity, "resample: zero up/down/count");
        return -1;
    }
    const size_t g = vc_gcd(up, down);
    up /= g;
    down /= g;
    if (up == 1u && down == 1u) {
        float *copy = vc_alloc(count, error, error_capacity, "resample copy");
        if (copy == NULL) return -1;
        memcpy(copy, input, count * sizeof(float));
        *out = copy;
        *out_count = count;
        return 0;
    }

    size_t scaled = 0;
    if (vc_mul(count, up, &scaled) != 0) {
        vc_err(error, error_capacity, "resample: length overflow");
        return -1;
    }
    const size_t produced = scaled / down + ((scaled % down) != 0 ? 1u : 0u);

    const size_t max_rate = (up > down) ? up : down;
    const double cutoff = 1.0 / (double)max_rate;
    const size_t half_len = 10u * max_rate;
    const size_t numtaps = 2u * half_len + 1u;
    float *taps = vc_alloc(numtaps, error, error_capacity, "resample taps");
    if (taps == NULL) return -1;
    if (vc_firwin_kaiser(numtaps, cutoff, 5.0, taps) != 0) {
        free(taps);
        vc_err(error, error_capacity, "resample: filter design failed");
        return -1;
    }
    /* scipy multiplies the unit-DC-gain filter by `up` in the signal dtype. */
    for (size_t i = 0; i < numtaps; ++i) taps[i] *= (float)up;

    /* Zero padding that centres the output samples, then the group delay is
     * dropped from the front.  Same arithmetic as scipy's resample_poly. */
    const size_t pre_pad = down - (half_len % down);
    size_t post_pad = 0;
    const size_t pre_remove = (half_len + pre_pad) / down;
    while (vc_upfirdn_len(numtaps + pre_pad + post_pad, count, up, down) <
           produced + pre_remove) {
        ++post_pad;
    }
    /* post_pad only ever extends the filter's zero tail, and a zero tap
     * contributes nothing, so it changes the loop bound below and nothing
     * else: `i < numtaps` already stops before it. */

    float *result = vc_alloc(produced, error, error_capacity, "resample out");
    if (result == NULL) {
        free(taps);
        return -1;
    }

    /*
     * out[j] = sum_i taps[i] * x_up[(j + pre_remove) * down - pre_pad - i]
     * with x_up the input upsampled by `up` (zeros between samples), so only
     * the taps whose index matches the phase contribute.
     */
    for (size_t j = 0; j < produced; ++j) {
        const long long base =
            (long long)(j + pre_remove) * (long long)down - (long long)pre_pad;
        /* smallest i >= 0 with (base - i) divisible by up */
        long long phase = base % (long long)up;
        if (phase < 0) phase += (long long)up;
        long long i = phase;
        long long sample = (base - i) / (long long)up;
        float acc = 0.0f;
        while (i < (long long)numtaps && sample >= 0) {
            if (sample < (long long)count) {
                acc += taps[(size_t)i] * input[(size_t)sample];
            }
            i += (long long)up;
            sample -= 1;
        }
        result[j] = acc;
    }
    free(taps);
    *out = result;
    *out_count = produced;
    return 0;
}

int mynah_audio_clip_resample(mynah_audio_clip *clip, unsigned target_rate,
                              char *error, size_t error_capacity) {
    if (error != NULL && error_capacity > 0) error[0] = '\0';
    if (clip == NULL || clip->samples == NULL || clip->count == 0) {
        vc_err(error, error_capacity, "resample: empty clip");
        return -1;
    }
    if (target_rate == 0 || clip->sample_rate == 0) {
        vc_err(error, error_capacity, "resample: zero sample rate");
        return -1;
    }
    if (clip->sample_rate == target_rate) return 0;
    const size_t g = vc_gcd((size_t)clip->sample_rate, (size_t)target_rate);
    float *resampled = NULL;
    size_t produced = 0;
    if (mynah_audio_resample_poly(clip->samples, clip->count,
                                  (size_t)target_rate / g,
                                  (size_t)clip->sample_rate / g, &resampled,
                                  &produced, error, error_capacity) != 0) {
        return -1;
    }
    free(clip->samples);
    clip->samples = resampled;
    clip->count = produced;
    clip->sample_rate = target_rate;
    return 0;
}

/* ----------------------------------------------------------------- config */

void mynah_voice_clone_config_defaults(mynah_voice_clone_config *config) {
    if (config == NULL) return;
    memset(config, 0, sizeof(*config));
    config->seanet.elu_alpha = 1.0f;
    mynah_transformer_ar_config_defaults(&config->encoder_transformer);
    mynah_transformer_ar_config_defaults(&config->backbone);
    config->max_seconds = 30.0;
    config->chunk_latent_frames = 4u;
}

/* -------------------------------------------------------- SEANet encoder */

enum { VC_W_FIRST = 0, VC_W_STAGE = 1, VC_W_LAST = 2 };

typedef struct {
    int is_resblock;
    int pre_elu;
    int weight_slot;
    size_t weight_index;
    size_t in_len; /* chunk length entering this op */
    mynah_conv1d_spec spec;
    mynah_conv1d_spec rb1_spec;
    mynah_conv1d_spec rb2_spec;
    mynah_causal_conv1d conv;
    mynah_causal_conv1d rb1;
    mynah_causal_conv1d rb2;
} vc_op;

struct mynah_voice_encoder {
    mynah_voice_clone_config config;

    size_t hop;                /* prod(ratios), samples per encoder frame   */
    size_t chunk_samples;      /* waveform samples per encode chunk         */
    size_t chunk_encoder;      /* encoder frames per chunk                  */
    size_t chunk_latent;       /* latent frames per chunk                   */
    size_t max_samples;
    size_t max_encoder_frames;
    size_t max_latent_frames;

    vc_op *ops;
    size_t n_ops;
    size_t n_blocks;

    float *conv_scratch;
    size_t conv_scratch_floats;
    float *work[3];
    size_t work_floats;
    float *chunk_in;   /* [chunk_samples], zero padded on the last chunk    */
    float *rows;       /* [chunk_encoder][dimension]                        */
    float *rows_out;   /* [chunk_encoder][dimension]                        */
    float *channels;   /* [dimension][chunk_encoder]                        */
    float *down_out;   /* [latent_dim][chunk_latent]                        */

    mynah_transformer_ar_state *transformer;
    mynah_seanet_downsample *downsample;

    float *latents;      /* [max_latent_frames][latent_dim]  */
    float *conditioning; /* [max_latent_frames][d_model]     */
    size_t frames;
};

static size_t vc_ipow(size_t base, size_t exponent) {
    size_t value = 1u;
    for (size_t i = 0; i < exponent; ++i) value *= base;
    return value;
}

static int vc_conv_spec(mynah_conv1d_spec *spec, size_t in_channels,
                        size_t out_channels, size_t kernel, size_t stride,
                        size_t dilation) {
    if (in_channels == 0 || out_channels == 0 || kernel == 0 || stride == 0) {
        return -1;
    }
    memset(spec, 0, sizeof(*spec));
    spec->in_channels = in_channels;
    spec->out_channels = out_channels;
    spec->kernel_size = kernel;
    spec->stride = stride;
    spec->dilation = dilation;
    spec->groups = 1u;
    spec->pad_mode = MYNAH_CONV_PAD_ZERO; /* seanet pad_mode: "constant" */
    return 0;
}

/*
 * Builds the encoder op list: the decoder's topology run backwards.  Upstream
 * (`modules/seanet.py`, SEANetEncoder) is
 *
 *     conv(channels -> n_filters, kernel_size)
 *     for ratio in reversed(ratios):
 *         n_residual_layers x resblock(mult * n_filters)
 *         ELU, conv(mult*n_filters -> 2*mult*n_filters, 2*ratio, stride ratio)
 *     ELU, conv(mult * n_filters -> dimension, last_kernel_size)
 *
 * which is why the residual blocks come before the downsampling conv here and
 * after the upsampling one in the decoder.
 */
static int vc_build_ops(mynah_voice_encoder *encoder, char *error,
                        size_t capacity) {
    const mynah_seanet_config *cfg = &encoder->config.seanet;
    const size_t n_ops = 2u + cfg->n_ratios * (1u + cfg->n_residual_layers);
    encoder->ops = (vc_op *)calloc(n_ops, sizeof(vc_op));
    if (encoder->ops == NULL) {
        vc_err(error, capacity, "voice clone: out of memory for the encoder");
        return -1;
    }
    encoder->n_ops = n_ops;

    size_t at = 0;
    size_t len = encoder->chunk_samples;
    size_t channels = cfg->channels;
    size_t width = cfg->n_filters;
    size_t work_max = 0;
    size_t scratch = 0;
    size_t block = 0;

    vc_op *op = &encoder->ops[at++];
    op->weight_slot = VC_W_FIRST;
    op->in_len = len;
    if (vc_conv_spec(&op->spec, channels, width, cfg->kernel_size, 1u, 1u) !=
        0) {
        vc_err(error, capacity, "voice clone: bad first convolution");
        return -1;
    }
    work_max = vc_max(work_max, width * len);
    scratch += mynah_causal_conv1d_scratch(&op->spec, len);

    for (size_t stage = 0; stage < cfg->n_ratios; ++stage) {
        const size_t ratio = cfg->ratios[cfg->n_ratios - 1u - stage];
        if (ratio == 0 || (len % ratio) != 0) {
            vc_err(error, capacity,
                   "voice clone: chunk of %zu samples is not divisible by the "
                   "encoder ratio %zu at stage %zu",
                   encoder->chunk_samples, ratio, stage);
            return -1;
        }
        for (size_t j = 0; j < cfg->n_residual_layers; ++j) {
            const size_t hidden = width / (cfg->compress ? cfg->compress : 1u);
            vc_op *rb = &encoder->ops[at++];
            rb->is_resblock = 1;
            rb->weight_index = block++;
            rb->in_len = len;
            if (hidden == 0 ||
                vc_conv_spec(&rb->rb1_spec, width, hidden,
                             cfg->residual_kernel_size, 1u,
                             vc_ipow(cfg->dilation_base, j)) != 0 ||
                vc_conv_spec(&rb->rb2_spec, hidden, width, 1u, 1u, 1u) != 0) {
                vc_err(error, capacity,
                       "voice clone: bad residual block at stage %zu", stage);
                return -1;
            }
            work_max = vc_max(work_max, vc_max(width, hidden) * len);
            scratch += mynah_causal_conv1d_scratch(&rb->rb1_spec, len);
            scratch += mynah_causal_conv1d_scratch(&rb->rb2_spec, len);
        }
        vc_op *down = &encoder->ops[at++];
        down->pre_elu = 1;
        down->weight_slot = VC_W_STAGE;
        down->weight_index = stage;
        down->in_len = len;
        if (vc_conv_spec(&down->spec, width, width * 2u, ratio * 2u, ratio,
                         1u) != 0) {
            vc_err(error, capacity,
                   "voice clone: bad downsampling convolution at stage %zu",
                   stage);
            return -1;
        }
        scratch += mynah_causal_conv1d_scratch(&down->spec, len);
        len /= ratio;
        width *= 2u;
        work_max = vc_max(work_max, width * len);
    }

    vc_op *last = &encoder->ops[at++];
    last->pre_elu = 1;
    last->weight_slot = VC_W_LAST;
    last->in_len = len;
    if (vc_conv_spec(&last->spec, width, cfg->dimension, cfg->last_kernel_size,
                     1u, 1u) != 0) {
        vc_err(error, capacity, "voice clone: bad last convolution");
        return -1;
    }
    scratch += mynah_causal_conv1d_scratch(&last->spec, len);
    work_max = vc_max(work_max, cfg->dimension * len);

    if (at != n_ops || len != encoder->chunk_encoder) {
        vc_err(error, capacity,
               "voice clone: encoder built %zu ops for %zu and %zu frames for "
               "%zu",
               at, n_ops, len, encoder->chunk_encoder);
        return -1;
    }
    encoder->n_blocks = block;
    encoder->work_floats = work_max;
    encoder->conv_scratch_floats = scratch;

    encoder->conv_scratch =
        vc_alloc(scratch, error, capacity, "encoder conv state");
    if (encoder->conv_scratch == NULL) return -1;
    for (size_t i = 0; i < 3u; ++i) {
        encoder->work[i] = vc_alloc(work_max, error, capacity, "encoder work");
        if (encoder->work[i] == NULL) return -1;
    }

    float *cursor = encoder->conv_scratch;
    size_t left = scratch;
    for (size_t i = 0; i < n_ops; ++i) {
        vc_op *item = &encoder->ops[i];
        if (item->is_resblock) {
            const size_t need1 =
                mynah_causal_conv1d_scratch(&item->rb1_spec, item->in_len);
            if (mynah_causal_conv1d_init(&item->rb1, &item->rb1_spec,
                                         item->in_len, cursor, left, error,
                                         capacity) != 0) {
                return -1;
            }
            cursor += need1;
            left -= need1;
            const size_t need2 =
                mynah_causal_conv1d_scratch(&item->rb2_spec, item->in_len);
            if (mynah_causal_conv1d_init(&item->rb2, &item->rb2_spec,
                                         item->in_len, cursor, left, error,
                                         capacity) != 0) {
                return -1;
            }
            cursor += need2;
            left -= need2;
        } else {
            const size_t need =
                mynah_causal_conv1d_scratch(&item->spec, item->in_len);
            if (mynah_causal_conv1d_init(&item->conv, &item->spec,
                                         item->in_len, cursor, left, error,
                                         capacity) != 0) {
                return -1;
            }
            cursor += need;
            left -= need;
        }
    }
    return 0;
}

/* Runs one chunk through the SEANet encoder.  `input` is [channels][in_len],
 * the result lands in `*out` as [dimension][in_len / hop]. */
static int vc_encoder_forward(mynah_voice_encoder *encoder,
                              const mynah_voice_clone_weights *weights,
                              const float *input, size_t in_len,
                              const float **out) {
    const float alpha = encoder->config.seanet.elu_alpha;
    float *a = encoder->work[0];
    float *b = encoder->work[1];
    float *c = encoder->work[2];
    float *cur = NULL;
    size_t channels = encoder->config.seanet.channels;
    size_t len = in_len;

    for (size_t i = 0; i < encoder->n_ops; ++i) {
        vc_op *op = &encoder->ops[i];
        if (op->is_resblock) {
            if (cur == NULL) return -1;
            const mynah_seanet_resblock_weights *bw =
                &weights->blocks[op->weight_index];
            float *other = (cur == a) ? b : a;
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
            continue;
        }

        const mynah_conv_weights *cw = NULL;
        if (op->weight_slot == VC_W_FIRST) {
            cw = &weights->first;
        } else if (op->weight_slot == VC_W_LAST) {
            cw = &weights->last;
        } else {
            cw = &weights->stage_conv[op->weight_index];
        }
        const float *src = NULL;
        if (cur == NULL) {
            src = input; /* the entry conv never has a pre-activation */
        } else {
            if (op->pre_elu) {
                mynah_seanet_elu_f32(cur, cur, channels * len, alpha);
            }
            src = cur;
        }
        float *dst = (src == a) ? b : a;
        if (mynah_causal_conv1d_apply(&op->conv, cw, src, len, dst) != 0) {
            return -1;
        }
        cur = dst;
        channels = op->spec.out_channels;
        len /= op->spec.stride;
    }
    if (cur == NULL) return -1;
    *out = cur;
    return 0;
}

static void vc_transpose(const float *src, size_t rows, size_t cols,
                         float *dst) {
    /* dst[c][r] = src[r][c] */
    for (size_t r = 0; r < rows; ++r) {
        const float *in = src + r * cols;
        for (size_t c = 0; c < cols; ++c) dst[c * rows + r] = in[c];
    }
}

/* ------------------------------------------------------------- lifecycle */

static int vc_config_valid(const mynah_voice_clone_config *config, char *error,
                           size_t capacity) {
    const mynah_seanet_config *sea = &config->seanet;
    if (sea->channels == 0) {
        vc_err(error, capacity, "voice clone: incomplete SEANet configuration");
        return -1;
    }
    if (sea->channels != 1u) {
        vc_err(error, capacity,
               "voice clone: the reference clip is mono, but the encoder wants "
               "%zu channels",
               sea->channels);
        return -1;
    }
    if (sea->dimension == 0 || sea->n_filters == 0 || sea->n_ratios == 0 ||
        sea->ratios == NULL || sea->kernel_size == 0 ||
        sea->last_kernel_size == 0 || sea->residual_kernel_size == 0 ||
        sea->compress == 0 || sea->dilation_base == 0) {
        vc_err(error, capacity, "voice clone: incomplete SEANet configuration");
        return -1;
    }
    if (config->sample_rate == 0 || config->samples_per_frame == 0) {
        vc_err(error, capacity, "voice clone: no sample rate or frame size");
        return -1;
    }
    if (config->downsample.stride == 0 ||
        config->downsample.in_channels != sea->dimension ||
        config->downsample.out_channels == 0) {
        vc_err(error, capacity,
               "voice clone: downsample %zu->%zu stride %zu does not match a "
               "SEANet dimension of %zu",
               config->downsample.in_channels,
               config->downsample.out_channels, config->downsample.stride,
               sea->dimension);
        return -1;
    }
    if (config->encoder_transformer.d_model != sea->dimension) {
        vc_err(error, capacity,
               "voice clone: encoder transformer is d%zu but SEANet emits %zu",
               config->encoder_transformer.d_model, sea->dimension);
        return -1;
    }
    if (config->backbone.d_model == 0 || config->backbone.num_layers == 0) {
        vc_err(error, capacity, "voice clone: no backbone configuration");
        return -1;
    }
    if (!(config->max_seconds > 0.0)) {
        vc_err(error, capacity,
               "voice clone: max_seconds must be positive; every buffer is "
               "sized from it");
        return -1;
    }
    return 0;
}

mynah_voice_encoder *mynah_voice_encoder_create(
    const mynah_voice_clone_config *config, char *error,
    size_t error_capacity) {
    if (error != NULL && error_capacity > 0) error[0] = '\0';
    if (config == NULL) {
        vc_err(error, error_capacity, "voice clone: null configuration");
        return NULL;
    }
    if (vc_config_valid(config, error, error_capacity) != 0) return NULL;

    mynah_voice_encoder *encoder =
        (mynah_voice_encoder *)calloc(1u, sizeof(*encoder));
    if (encoder == NULL) {
        vc_err(error, error_capacity, "voice clone: out of memory");
        return NULL;
    }
    encoder->config = *config;

    size_t hop = 1u;
    for (size_t i = 0; i < config->seanet.n_ratios; ++i) {
        if (vc_mul(hop, config->seanet.ratios[i], &hop) != 0) {
            vc_err(error, error_capacity, "voice clone: ratio overflow");
            mynah_voice_encoder_destroy(encoder);
            return NULL;
        }
    }
    encoder->hop = hop;
    if (hop == 0 ||
        config->samples_per_frame != hop * config->downsample.stride) {
        vc_err(error, error_capacity,
               "voice clone: %zu samples per frame but the encoder hop is %zu "
               "and the downsample stride %zu",
               config->samples_per_frame, hop, config->downsample.stride);
        mynah_voice_encoder_destroy(encoder);
        return NULL;
    }

    const double frames = config->max_seconds * (double)config->sample_rate /
                          (double)config->samples_per_frame;
    if (!(frames >= 1.0) || frames > 1.0e9) {
        vc_err(error, error_capacity, "voice clone: max_seconds %.3f is out of "
                                      "range for a %zu-sample frame",
               config->max_seconds, config->samples_per_frame);
        mynah_voice_encoder_destroy(encoder);
        return NULL;
    }
    encoder->max_latent_frames = (size_t)frames;
    encoder->max_samples =
        encoder->max_latent_frames * config->samples_per_frame;
    encoder->max_encoder_frames =
        encoder->max_latent_frames * config->downsample.stride;

    size_t chunk = config->chunk_latent_frames;
    if (chunk == 0) chunk = 4u;
    if (chunk > encoder->max_latent_frames) chunk = encoder->max_latent_frames;
    encoder->chunk_latent = chunk;
    encoder->chunk_samples = chunk * config->samples_per_frame;
    encoder->chunk_encoder = chunk * config->downsample.stride;

    if (vc_build_ops(encoder, error, error_capacity) != 0) {
        mynah_voice_encoder_destroy(encoder);
        return NULL;
    }

    mynah_transformer_ar_config xf = config->encoder_transformer;
    xf.max_seq_len = encoder->max_encoder_frames;
    encoder->transformer =
        mynah_transformer_ar_state_new(&xf, error, error_capacity);
    if (encoder->transformer == NULL) {
        mynah_voice_encoder_destroy(encoder);
        return NULL;
    }

    encoder->downsample = mynah_seanet_downsample_create(
        &config->downsample, encoder->chunk_encoder, error, error_capacity);
    if (encoder->downsample == NULL) {
        mynah_voice_encoder_destroy(encoder);
        return NULL;
    }

    const size_t latent = config->downsample.out_channels;
    const size_t hidden = config->backbone.d_model;
    encoder->chunk_in = vc_alloc(encoder->chunk_samples, error, error_capacity,
                                 "encoder input");
    encoder->rows = vc_alloc(encoder->chunk_encoder * config->seanet.dimension,
                             error, error_capacity, "encoder rows");
    encoder->rows_out =
        vc_alloc(encoder->chunk_encoder * config->seanet.dimension, error,
                 error_capacity, "encoder rows");
    encoder->channels =
        vc_alloc(encoder->chunk_encoder * config->seanet.dimension, error,
                 error_capacity, "encoder channels");
    encoder->down_out = vc_alloc(encoder->chunk_latent * latent, error,
                                 error_capacity, "latent chunk");
    encoder->latents = vc_alloc(encoder->max_latent_frames * latent, error,
                                error_capacity, "latents");
    encoder->conditioning = vc_alloc(encoder->max_latent_frames * hidden, error,
                                     error_capacity, "conditioning");
    if (encoder->chunk_in == NULL || encoder->rows == NULL ||
        encoder->rows_out == NULL || encoder->channels == NULL ||
        encoder->down_out == NULL || encoder->latents == NULL ||
        encoder->conditioning == NULL) {
        mynah_voice_encoder_destroy(encoder);
        return NULL;
    }
    return encoder;
}

void mynah_voice_encoder_destroy(mynah_voice_encoder *encoder) {
    if (encoder == NULL) return;
    mynah_transformer_ar_state_free(encoder->transformer);
    mynah_seanet_downsample_destroy(encoder->downsample);
    free(encoder->ops);
    free(encoder->conv_scratch);
    for (size_t i = 0; i < 3u; ++i) free(encoder->work[i]);
    free(encoder->chunk_in);
    free(encoder->rows);
    free(encoder->rows_out);
    free(encoder->channels);
    free(encoder->down_out);
    free(encoder->latents);
    free(encoder->conditioning);
    free(encoder);
}

void mynah_voice_encoder_reset(mynah_voice_encoder *encoder) {
    if (encoder == NULL) return;
    for (size_t i = 0; i < encoder->n_ops; ++i) {
        vc_op *op = &encoder->ops[i];
        if (op->is_resblock) {
            mynah_causal_conv1d_reset(&op->rb1);
            mynah_causal_conv1d_reset(&op->rb2);
        } else {
            mynah_causal_conv1d_reset(&op->conv);
        }
    }
    mynah_transformer_ar_state_reset(encoder->transformer);
    mynah_seanet_downsample_reset(encoder->downsample);
    encoder->frames = 0;
}

size_t mynah_voice_encoder_frames(const mynah_voice_encoder *encoder) {
    return (encoder != NULL) ? encoder->frames : 0;
}

const float *mynah_voice_encoder_latents(const mynah_voice_encoder *encoder) {
    return (encoder != NULL) ? encoder->latents : NULL;
}

const float *mynah_voice_encoder_conditioning(
    const mynah_voice_encoder *encoder) {
    return (encoder != NULL) ? encoder->conditioning : NULL;
}

int mynah_voice_clone_check_weights(const mynah_voice_encoder *encoder,
                                    const mynah_voice_clone_weights *weights,
                                    char *error, size_t error_capacity) {
    if (error != NULL && error_capacity > 0) error[0] = '\0';
    if (encoder == NULL || weights == NULL) {
        vc_err(error, error_capacity, "voice clone: null encoder or weights");
        return -1;
    }
    if (weights->first.weight == NULL || weights->last.weight == NULL) {
        vc_err(error, error_capacity,
               "voice clone: the encoder's first/last convolution is missing");
        return -1;
    }
    if (weights->stage_conv == NULL) {
        vc_err(error, error_capacity, "voice clone: stage convolutions missing");
        return -1;
    }
    for (size_t i = 0; i < encoder->config.seanet.n_ratios; ++i) {
        if (weights->stage_conv[i].weight == NULL) {
            vc_err(error, error_capacity,
                   "voice clone: stage convolution %zu missing", i);
            return -1;
        }
    }
    if (encoder->n_blocks > 0 && weights->blocks == NULL) {
        vc_err(error, error_capacity, "voice clone: residual blocks missing");
        return -1;
    }
    for (size_t i = 0; i < encoder->n_blocks; ++i) {
        if (weights->blocks[i].conv1.weight == NULL ||
            weights->blocks[i].conv2.weight == NULL) {
            vc_err(error, error_capacity,
                   "voice clone: residual block %zu is incomplete", i);
            return -1;
        }
    }
    if (weights->downsample.weight == NULL) {
        vc_err(error, error_capacity, "voice clone: downsample weight missing");
        return -1;
    }
    if (weights->speaker_proj == NULL) {
        vc_err(error, error_capacity,
               "voice clone: flow_lm.speaker_proj_weight missing");
        return -1;
    }
    if (encoder->config.insert_bos_before_voice != 0 &&
        weights->bos_before_voice == NULL) {
        vc_err(error, error_capacity,
               "voice clone: insert_bos_before_voice is set but "
               "flow_lm.bos_before_voice is missing");
        return -1;
    }
    return mynah_transformer_ar_check_weights(encoder->transformer,
                                              &weights->encoder_transformer,
                                              error, error_capacity);
}

/* --------------------------------------------------------------- encoding */

int mynah_voice_encoder_encode(mynah_voice_encoder *encoder,
                               const mynah_voice_clone_weights *weights,
                               const mynah_voice_clone_consent *consent,
                               const float *samples, size_t count,
                               char *error, size_t error_capacity) {
    if (error != NULL && error_capacity > 0) error[0] = '\0';
    if (encoder == NULL || weights == NULL || samples == NULL) {
        vc_err(error, error_capacity, "voice clone: null argument to encode");
        return -1;
    }
    if (mynah_voice_clone_consent_check(consent, error, error_capacity) != 0) {
        return -1;
    }
    if (mynah_voice_clone_check_weights(encoder, weights, error,
                                        error_capacity) != 0) {
        return -1;
    }
    if (count == 0) {
        vc_err(error, error_capacity, "voice clone: the reference clip is empty");
        return -1;
    }
    if (count > encoder->max_samples) {
        vc_err(error, error_capacity,
               "voice clone: %zu samples is more than the %.1f s cap (%zu "
               "samples); truncate the clip explicitly",
               count, encoder->config.max_seconds, encoder->max_samples);
        return -1;
    }
    for (size_t i = 0; i < count; ++i) {
        if (!isfinite(samples[i])) {
            vc_err(error, error_capacity,
                   "voice clone: sample %zu is not finite", i);
            return -1;
        }
    }

    mynah_voice_encoder_reset(encoder);

    const mynah_voice_clone_config *config = &encoder->config;
    const size_t frame = config->samples_per_frame;
    /* `pad_for_conv1d(x, frame_size, frame_size)`: zero pad the tail up to a
     * whole 80 ms frame, which is the only padding upstream applies. */
    const size_t padded = ((count + frame - 1u) / frame) * frame;
    const size_t dimension = config->seanet.dimension;
    const size_t latent = config->downsample.out_channels;
    const size_t hidden = config->backbone.d_model;

    size_t done = 0;
    while (done < padded) {
        size_t take = padded - done;
        if (take > encoder->chunk_samples) take = encoder->chunk_samples;

        memset(encoder->chunk_in, 0, take * sizeof(float));
        const size_t real = (done < count) ? (count - done) : 0;
        if (real > 0) {
            memcpy(encoder->chunk_in, samples + done,
                   ((real < take) ? real : take) * sizeof(float));
        }

        const float *encoded = NULL;
        if (vc_encoder_forward(encoder, weights, encoder->chunk_in, take,
                               &encoded) != 0) {
            vc_err(error, error_capacity,
                   "voice clone: the SEANet encoder rejected a %zu-sample chunk",
                   take);
            return -1;
        }
        const size_t enc_frames = take / encoder->hop;

        vc_transpose(encoded, dimension, enc_frames, encoder->rows);
        if (mynah_transformer_ar_prefill(encoder->transformer,
                                         &weights->encoder_transformer,
                                         encoder->rows, enc_frames,
                                         encoder->rows_out) != 0) {
            vc_err(error, error_capacity,
                   "voice clone: the encoder transformer failed at frame %zu",
                   done / encoder->hop);
            return -1;
        }
        vc_transpose(encoder->rows_out, enc_frames, dimension,
                     encoder->channels);

        if (mynah_seanet_downsample_apply(encoder->downsample,
                                          &weights->downsample,
                                          encoder->channels, enc_frames,
                                          encoder->down_out) != 0) {
            vc_err(error, error_capacity,
                   "voice clone: the downsample rejected %zu encoder frames",
                   enc_frames);
            return -1;
        }
        const size_t lat_frames = enc_frames / config->downsample.stride;
        if (encoder->frames + lat_frames > encoder->max_latent_frames) {
            vc_err(error, error_capacity, "voice clone: latent overflow");
            return -1;
        }
        vc_transpose(encoder->down_out, latent, lat_frames,
                     encoder->latents + encoder->frames * latent);
        for (size_t i = 0; i < lat_frames; ++i) {
            const float *row =
                encoder->latents + (encoder->frames + i) * latent;
            float *cond = encoder->conditioning + (encoder->frames + i) * hidden;
            mynah_matvec_f32(weights->speaker_proj, row, cond, hidden, latent);
        }
        encoder->frames += lat_frames;
        done += take;
    }
    return 0;
}

/* --------------------------------------------------------------- prefill */

int mynah_voice_clone_prefill(mynah_transformer_ar_state *state,
                              const mynah_transformer_ar_weights *weights,
                              const float *bos_before_voice,
                              const float *conditioning, size_t frames,
                              char *error, size_t error_capacity) {
    if (error != NULL && error_capacity > 0) error[0] = '\0';
    if (state == NULL || weights == NULL || conditioning == NULL) {
        vc_err(error, error_capacity, "voice clone: null argument to prefill");
        return -1;
    }
    if (frames == 0) {
        vc_err(error, error_capacity, "voice clone: nothing to prefill");
        return -1;
    }
    const mynah_transformer_ar_config *config =
        mynah_transformer_ar_state_config(state);
    if (config == NULL) {
        vc_err(error, error_capacity, "voice clone: the backbone has no config");
        return -1;
    }
    const size_t extra = (bos_before_voice != NULL) ? 1u : 0u;
    size_t needed = 0;
    if (vc_add(frames, extra, &needed) != 0 ||
        needed + mynah_transformer_ar_state_offset(state) >
            config->max_seq_len) {
        vc_err(error, error_capacity,
               "voice clone: a %zu-frame voice does not fit in a %zu-position "
               "KV cache",
               needed, config->max_seq_len);
        return -1;
    }
    if (bos_before_voice != NULL) {
        if (mynah_transformer_ar_prefill(state, weights, bos_before_voice, 1u,
                                         NULL) != 0) {
            vc_err(error, error_capacity,
                   "voice clone: bos_before_voice was rejected");
            return -1;
        }
    }
    if (mynah_transformer_ar_prefill(state, weights, conditioning, frames,
                                     NULL) != 0) {
        vc_err(error, error_capacity,
               "voice clone: the backbone rejected the %zu-frame voice prefix",
               frames);
        return -1;
    }
    return 0;
}

/* ----------------------------------------------------- weight resolution */

struct mynah_voice_clone_weights_owner {
    mynah_voice_clone_weights view;
    mynah_transformer_ar_weights backbone;
    mynah_conv_weights *stage_conv;
    mynah_seanet_resblock_weights *blocks;
    mynah_transformer_ar_layer *encoder_layers;
    mynah_transformer_ar_layer *backbone_layers;
};

static int vc_get(const mynah_weights *file, const char *name,
                  mynah_tensor *out, char *error, size_t capacity) {
    if (mynah_weights_get(file, name, out) != 0) {
        vc_err(error, capacity, "voice clone: missing tensor %s", name);
        return -1;
    }
    return 0;
}

static int vc_get_shaped(const mynah_weights *file, const char *name,
                         size_t rank, const size_t *shape,
                         const float **data, char *error, size_t capacity) {
    mynah_tensor tensor;
    if (vc_get(file, name, &tensor, error, capacity) != 0) return -1;
    if (tensor.rank != rank) {
        vc_err(error, capacity, "voice clone: %s has rank %zu, expected %zu",
               name, tensor.rank, rank);
        return -1;
    }
    for (size_t i = 0; i < rank; ++i) {
        if (shape[i] != 0 && tensor.shape[i] != shape[i]) {
            vc_err(error, capacity,
                   "voice clone: %s axis %zu is %zu, expected %zu", name, i,
                   tensor.shape[i], shape[i]);
            return -1;
        }
    }
    *data = tensor.data;
    return 0;
}

static int vc_get_conv(const mynah_weights *file, const char *prefix,
                       size_t out_channels, size_t in_channels, size_t kernel,
                       int want_bias, mynah_conv_weights *out, char *error,
                       size_t capacity) {
    char name[256];
    const size_t shape[3] = {out_channels, in_channels, kernel};
    (void)snprintf(name, sizeof(name), "%s.weight", prefix);
    if (vc_get_shaped(file, name, 3u, shape, &out->weight, error, capacity) !=
        0) {
        return -1;
    }
    out->bias = NULL;
    if (want_bias) {
        const size_t bias_shape[1] = {out_channels};
        (void)snprintf(name, sizeof(name), "%s.bias", prefix);
        if (vc_get_shaped(file, name, 1u, bias_shape, &out->bias, error,
                          capacity) != 0) {
            return -1;
        }
    }
    return 0;
}

static int vc_get_layer(const mynah_weights *file, const char *prefix,
                        size_t d_model, size_t attn_dim, size_t ffn_dim,
                        int want_layer_scale, mynah_transformer_ar_layer *out,
                        char *error, size_t capacity) {
    char name[256];
    size_t shape[2];
    memset(out, 0, sizeof(*out));

    shape[0] = 3u * attn_dim;
    shape[1] = d_model;
    (void)snprintf(name, sizeof(name), "%s.self_attn.in_proj.weight", prefix);
    if (vc_get_shaped(file, name, 2u, shape, &out->in_proj_weight, error,
                      capacity) != 0) {
        return -1;
    }
    shape[0] = d_model;
    shape[1] = attn_dim;
    (void)snprintf(name, sizeof(name), "%s.self_attn.out_proj.weight", prefix);
    if (vc_get_shaped(file, name, 2u, shape, &out->out_proj_weight, error,
                      capacity) != 0) {
        return -1;
    }
    shape[0] = ffn_dim;
    shape[1] = d_model;
    (void)snprintf(name, sizeof(name), "%s.linear1.weight", prefix);
    if (vc_get_shaped(file, name, 2u, shape, &out->linear1_weight, error,
                      capacity) != 0) {
        return -1;
    }
    shape[0] = d_model;
    shape[1] = ffn_dim;
    (void)snprintf(name, sizeof(name), "%s.linear2.weight", prefix);
    if (vc_get_shaped(file, name, 2u, shape, &out->linear2_weight, error,
                      capacity) != 0) {
        return -1;
    }
    shape[0] = d_model;
    (void)snprintf(name, sizeof(name), "%s.norm1.weight", prefix);
    if (vc_get_shaped(file, name, 1u, shape, &out->norm1_weight, error,
                      capacity) != 0) {
        return -1;
    }
    (void)snprintf(name, sizeof(name), "%s.norm1.bias", prefix);
    if (vc_get_shaped(file, name, 1u, shape, &out->norm1_bias, error,
                      capacity) != 0) {
        return -1;
    }
    (void)snprintf(name, sizeof(name), "%s.norm2.weight", prefix);
    if (vc_get_shaped(file, name, 1u, shape, &out->norm2_weight, error,
                      capacity) != 0) {
        return -1;
    }
    (void)snprintf(name, sizeof(name), "%s.norm2.bias", prefix);
    if (vc_get_shaped(file, name, 1u, shape, &out->norm2_bias, error,
                      capacity) != 0) {
        return -1;
    }
    if (want_layer_scale) {
        (void)snprintf(name, sizeof(name), "%s.layer_scale_1.scale", prefix);
        if (vc_get_shaped(file, name, 1u, shape, &out->layer_scale_1, error,
                          capacity) != 0) {
            return -1;
        }
        (void)snprintf(name, sizeof(name), "%s.layer_scale_2.scale", prefix);
        if (vc_get_shaped(file, name, 1u, shape, &out->layer_scale_2, error,
                          capacity) != 0) {
            return -1;
        }
    }
    return 0;
}

int mynah_voice_clone_weights_load(const mynah_weights *file,
                                   const mynah_voice_clone_config *config,
                                   mynah_voice_clone_weights_owner **out,
                                   char *error, size_t error_capacity) {
    if (error != NULL && error_capacity > 0) error[0] = '\0';
    if (file == NULL || config == NULL || out == NULL) {
        vc_err(error, error_capacity, "voice clone: null argument to load");
        return -1;
    }
    *out = NULL;
    if (vc_config_valid(config, error, error_capacity) != 0) return -1;

    const mynah_seanet_config *sea = &config->seanet;
    mynah_voice_clone_weights_owner *owner =
        (mynah_voice_clone_weights_owner *)calloc(1u, sizeof(*owner));
    if (owner == NULL) {
        vc_err(error, error_capacity, "voice clone: out of memory");
        return -1;
    }
    const size_t n_blocks = sea->n_ratios * sea->n_residual_layers;
    owner->stage_conv =
        (mynah_conv_weights *)calloc(sea->n_ratios, sizeof(mynah_conv_weights));
    owner->blocks = (mynah_seanet_resblock_weights *)calloc(
        (n_blocks > 0) ? n_blocks : 1u, sizeof(mynah_seanet_resblock_weights));
    owner->encoder_layers = (mynah_transformer_ar_layer *)calloc(
        (config->encoder_transformer.num_layers > 0)
            ? config->encoder_transformer.num_layers
            : 1u,
        sizeof(mynah_transformer_ar_layer));
    owner->backbone_layers = (mynah_transformer_ar_layer *)calloc(
        config->backbone.num_layers, sizeof(mynah_transformer_ar_layer));
    if (owner->stage_conv == NULL || owner->blocks == NULL ||
        owner->encoder_layers == NULL || owner->backbone_layers == NULL) {
        vc_err(error, error_capacity, "voice clone: out of memory");
        mynah_voice_clone_weights_free(owner);
        return -1;
    }

    char prefix[256];
    size_t index = 0;
    size_t width = sea->n_filters;
    size_t block = 0;

    (void)snprintf(prefix, sizeof(prefix), "mimi.encoder.model.%zu.conv",
                   index++);
    if (vc_get_conv(file, prefix, width, sea->channels, sea->kernel_size, 1,
                    &owner->view.first, error, error_capacity) != 0) {
        mynah_voice_clone_weights_free(owner);
        return -1;
    }
    for (size_t stage = 0; stage < sea->n_ratios; ++stage) {
        const size_t ratio = sea->ratios[sea->n_ratios - 1u - stage];
        for (size_t j = 0; j < sea->n_residual_layers; ++j) {
            const size_t hidden = width / sea->compress;
            (void)snprintf(prefix, sizeof(prefix),
                           "mimi.encoder.model.%zu.block.1.conv", index);
            if (vc_get_conv(file, prefix, hidden, width,
                            sea->residual_kernel_size, 1,
                            &owner->blocks[block].conv1, error,
                            error_capacity) != 0) {
                mynah_voice_clone_weights_free(owner);
                return -1;
            }
            (void)snprintf(prefix, sizeof(prefix),
                           "mimi.encoder.model.%zu.block.3.conv", index);
            if (vc_get_conv(file, prefix, width, hidden, 1u, 1,
                            &owner->blocks[block].conv2, error,
                            error_capacity) != 0) {
                mynah_voice_clone_weights_free(owner);
                return -1;
            }
            ++block;
            ++index;
        }
        ++index; /* the ELU occupies a module slot */
        (void)snprintf(prefix, sizeof(prefix), "mimi.encoder.model.%zu.conv",
                       index++);
        if (vc_get_conv(file, prefix, width * 2u, width, ratio * 2u, 1,
                        &owner->stage_conv[stage], error, error_capacity) !=
            0) {
            mynah_voice_clone_weights_free(owner);
            return -1;
        }
        width *= 2u;
    }
    ++index; /* the last ELU */
    (void)snprintf(prefix, sizeof(prefix), "mimi.encoder.model.%zu.conv",
                   index);
    if (vc_get_conv(file, prefix, sea->dimension, width, sea->last_kernel_size,
                    1, &owner->view.last, error, error_capacity) != 0) {
        mynah_voice_clone_weights_free(owner);
        return -1;
    }

    const size_t xf_heads = config->encoder_transformer.num_heads;
    size_t xf_head_dim = config->encoder_transformer.head_dim;
    if (xf_head_dim == 0 && xf_heads > 0) {
        xf_head_dim = config->encoder_transformer.d_model / xf_heads;
    }
    for (size_t i = 0; i < config->encoder_transformer.num_layers; ++i) {
        (void)snprintf(prefix, sizeof(prefix),
                       "mimi.encoder_transformer.transformer.layers.%zu", i);
        if (vc_get_layer(file, prefix, config->encoder_transformer.d_model,
                         xf_heads * xf_head_dim,
                         config->encoder_transformer.ffn_dim, 1,
                         &owner->encoder_layers[i], error, error_capacity) !=
            0) {
            mynah_voice_clone_weights_free(owner);
            return -1;
        }
    }
    owner->view.encoder_transformer.layers = owner->encoder_layers;

    if (vc_get_conv(file, "mimi.downsample.conv.conv",
                    config->downsample.out_channels,
                    config->downsample.in_channels,
                    2u * config->downsample.stride, 0, &owner->view.downsample,
                    error, error_capacity) != 0) {
        mynah_voice_clone_weights_free(owner);
        return -1;
    }

    size_t shape[3];
    shape[0] = config->backbone.d_model;
    shape[1] = config->downsample.out_channels;
    if (vc_get_shaped(file, "flow_lm.speaker_proj_weight", 2u, shape,
                      &owner->view.speaker_proj, error, error_capacity) != 0) {
        mynah_voice_clone_weights_free(owner);
        return -1;
    }
    if (config->insert_bos_before_voice) {
        shape[0] = 1u;
        shape[1] = 1u;
        shape[2] = config->backbone.d_model;
        if (vc_get_shaped(file, "flow_lm.bos_before_voice", 3u, shape,
                          &owner->view.bos_before_voice, error,
                          error_capacity) != 0) {
            mynah_voice_clone_weights_free(owner);
            return -1;
        }
    }

    const size_t heads = config->backbone.num_heads;
    size_t head_dim = config->backbone.head_dim;
    if (head_dim == 0 && heads > 0) head_dim = config->backbone.d_model / heads;
    for (size_t i = 0; i < config->backbone.num_layers; ++i) {
        (void)snprintf(prefix, sizeof(prefix), "flow_lm.transformer.layers.%zu",
                       i);
        if (vc_get_layer(file, prefix, config->backbone.d_model,
                         heads * head_dim, config->backbone.ffn_dim, 0,
                         &owner->backbone_layers[i], error, error_capacity) !=
            0) {
            mynah_voice_clone_weights_free(owner);
            return -1;
        }
    }
    owner->backbone.layers = owner->backbone_layers;
    shape[0] = config->backbone.d_model;
    if (vc_get_shaped(file, "flow_lm.out_norm.weight", 1u, shape,
                      &owner->backbone.out_norm_weight, error,
                      error_capacity) != 0 ||
        vc_get_shaped(file, "flow_lm.out_norm.bias", 1u, shape,
                      &owner->backbone.out_norm_bias, error,
                      error_capacity) != 0) {
        mynah_voice_clone_weights_free(owner);
        return -1;
    }

    owner->view.stage_conv = owner->stage_conv;
    owner->view.blocks = owner->blocks;
    *out = owner;
    return 0;
}

void mynah_voice_clone_weights_free(mynah_voice_clone_weights_owner *owner) {
    if (owner == NULL) return;
    free(owner->stage_conv);
    free(owner->blocks);
    free(owner->encoder_layers);
    free(owner->backbone_layers);
    free(owner);
}

const mynah_voice_clone_weights *mynah_voice_clone_weights_view(
    const mynah_voice_clone_weights_owner *owner) {
    return (owner != NULL) ? &owner->view : NULL;
}

const mynah_transformer_ar_weights *mynah_voice_clone_backbone_view(
    const mynah_voice_clone_weights_owner *owner) {
    return (owner != NULL) ? &owner->backbone : NULL;
}

/* ------------------------------------------------------- voice serialiser */

typedef struct {
    char *data;
    size_t len;
    size_t cap;
} vc_str;

static int vc_str_reserve(vc_str *s, size_t extra) {
    size_t need = 0;
    if (vc_add(s->len, extra, &need) != 0) return -1;
    if (need + 1u <= s->cap) return 0;
    size_t cap = (s->cap > 0) ? s->cap : 256u;
    while (cap < need + 1u) {
        if (cap > ((size_t)-1) / 2u) return -1;
        cap *= 2u;
    }
    char *grown = (char *)realloc(s->data, cap);
    if (grown == NULL) return -1;
    s->data = grown;
    s->cap = cap;
    return 0;
}

static int vc_str_add(vc_str *s, const char *text) {
    const size_t n = strlen(text);
    if (vc_str_reserve(s, n) != 0) return -1;
    memcpy(s->data + s->len, text, n);
    s->len += n;
    s->data[s->len] = '\0';
    return 0;
}

static int vc_str_addf(vc_str *s, const char *format, ...)
    __attribute__((format(printf, 2, 3)));

static int vc_str_addf(vc_str *s, const char *format, ...) {
    char stack[512];
    va_list args;
    va_start(args, format);
    const int written = vsnprintf(stack, sizeof(stack), format, args);
    va_end(args);
    if (written < 0 || (size_t)written >= sizeof(stack)) return -1;
    return vc_str_add(s, stack);
}

/* JSON string body, quotes included. */
static int vc_str_add_json(vc_str *s, const char *text) {
    if (vc_str_add(s, "\"") != 0) return -1;
    for (const unsigned char *p = (const unsigned char *)text; *p != '\0';
         ++p) {
        char escape[8];
        const char *piece = escape;
        if (*p == '"') {
            piece = "\\\"";
        } else if (*p == '\\') {
            piece = "\\\\";
        } else if (*p < 0x20) {
            (void)snprintf(escape, sizeof(escape), "\\u%04x", (unsigned)*p);
        } else {
            escape[0] = (char)*p;
            escape[1] = '\0';
        }
        if (vc_str_add(s, piece) != 0) return -1;
    }
    return vc_str_add(s, "\"");
}

static uint16_t vc_f32_to_f16(float value) {
    uint32_t bits = 0;
    memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t exponent = (int32_t)((bits >> 23) & 0xFFu) - 127 + 15;
    uint32_t mantissa = bits & 0x7FFFFFu;
    if (((bits >> 23) & 0xFFu) == 0xFFu) {
        /* Inf or NaN; keep a non-zero mantissa non-zero. */
        return (uint16_t)(sign | 0x7C00u | (mantissa != 0 ? 0x200u : 0u));
    }
    if (exponent >= 0x1F) return (uint16_t)(sign | 0x7C00u); /* overflow */
    if (exponent <= 0) {
        if (exponent < -10) return (uint16_t)sign; /* underflow to zero */
        mantissa |= 0x800000u;
        const uint32_t shift = (uint32_t)(14 - exponent);
        const uint32_t round = 1u << (shift - 1u);
        uint32_t value16 = (mantissa + round) >> shift;
        /* Ties to even, matching the usual round-to-nearest-even. */
        if ((mantissa & ((1u << shift) - 1u)) == round) value16 &= ~1u;
        return (uint16_t)(sign | value16);
    }
    uint32_t out = ((uint32_t)exponent << 10) | (mantissa >> 13);
    const uint32_t rest = mantissa & 0x1FFFu;
    if (rest > 0x1000u || (rest == 0x1000u && ((mantissa >> 13) & 1u) != 0)) {
        out += 1u;
    }
    return (uint16_t)(sign | out);
}

static void vc_put_u16(unsigned char *p, uint16_t value) {
    p[0] = (unsigned char)(value & 0xFFu);
    p[1] = (unsigned char)((value >> 8) & 0xFFu);
}

static void vc_put_u32(unsigned char *p, uint32_t value) {
    for (size_t i = 0; i < 4u; ++i) {
        p[i] = (unsigned char)((value >> (8u * i)) & 0xFFu);
    }
}

static void vc_put_u64(unsigned char *p, uint64_t value) {
    for (size_t i = 0; i < 8u; ++i) {
        p[i] = (unsigned char)((value >> (8u * i)) & 0xFFu);
    }
}

static void vc_put_f32(unsigned char *p, float value) {
    uint32_t bits = 0;
    memcpy(&bits, &value, sizeof(bits));
    vc_put_u32(p, bits);
}

int mynah_voice_serialise(mynah_transformer_ar_state *state,
                          mynah_voice_dtype dtype,
                          const mynah_voice_clone_consent *consent,
                          const char *model_revision, void **out, size_t *size,
                          char *error, size_t error_capacity) {
    if (error != NULL && error_capacity > 0) error[0] = '\0';
    if (state == NULL || out == NULL || size == NULL) {
        vc_err(error, error_capacity, "voice export: null argument");
        return -1;
    }
    *out = NULL;
    *size = 0;
    const mynah_transformer_ar_config *config =
        mynah_transformer_ar_state_config(state);
    if (config == NULL) {
        vc_err(error, error_capacity, "voice export: the state has no config");
        return -1;
    }
    const size_t positions = mynah_transformer_ar_state_offset(state);
    if (positions == 0) {
        vc_err(error, error_capacity,
               "voice export: the KV cache is empty; prefill a voice first");
        return -1;
    }
    size_t heads = config->num_heads;
    size_t head_dim = config->head_dim;
    if (head_dim == 0 && heads > 0) head_dim = config->d_model / heads;
    if (heads == 0 || head_dim == 0) {
        vc_err(error, error_capacity, "voice export: no attention geometry");
        return -1;
    }
    const size_t item = (dtype == MYNAH_VOICE_DTYPE_F16) ? 2u : 4u;
    size_t per_layer = 0;
    if (vc_mul(positions, heads, &per_layer) != 0 ||
        vc_mul(per_layer, head_dim, &per_layer) != 0 ||
        vc_mul(per_layer, 2u, &per_layer) != 0) {
        vc_err(error, error_capacity, "voice export: cache size overflow");
        return -1;
    }
    const size_t cache_bytes = per_layer * item;

    vc_str header;
    memset(&header, 0, sizeof(header));
    int failed = vc_str_add(&header, "{");
    size_t offset = 0;
    for (size_t layer = 0; layer < config->num_layers && failed == 0; ++layer) {
        if (layer > 0) failed |= vc_str_add(&header, ",");
        failed |= vc_str_addf(
            &header,
            "\"transformer.layers.%zu.self_attn/cache\":{\"dtype\":\"%s\","
            "\"shape\":[2,1,%zu,%zu,%zu],\"data_offsets\":[%zu,%zu]},",
            layer, (dtype == MYNAH_VOICE_DTYPE_F16) ? "F16" : "F32", positions,
            heads, head_dim, offset, offset + cache_bytes);
        offset += cache_bytes;
        failed |= vc_str_addf(
            &header,
            "\"transformer.layers.%zu.self_attn/offset\":{\"dtype\":\"I64\","
            "\"shape\":[1],\"data_offsets\":[%zu,%zu]}",
            layer, offset, offset + 8u);
        offset += 8u;
    }
    if (failed == 0) {
        failed |= vc_str_add(&header, ",\"__metadata__\":{\"engine\":\"pocket\"");
        failed |= vc_str_add(&header, ",\"kind\":\"voice_kv\"");
        if (model_revision != NULL) {
            failed |= vc_str_add(&header, ",\"model_revision\":");
            failed |= vc_str_add_json(&header, model_revision);
        }
        if (consent != NULL && consent->affirmed != 0) {
            failed |= vc_str_add(&header, ",\"consent_affirmed\":\"true\"");
            if (consent->source != NULL) {
                failed |= vc_str_add(&header, ",\"consent_source\":");
                failed |= vc_str_add_json(&header, consent->source);
            }
            if (consent->affirmation != NULL) {
                failed |= vc_str_add(&header, ",\"consent_affirmation\":");
                failed |= vc_str_add_json(&header, consent->affirmation);
            }
        }
        failed |= vc_str_add(&header, "}}");
    }
    /* Space padding so the data section starts 8-byte aligned, exactly like
     * the reference serializer. */
    while (failed == 0 && ((8u + header.len) % 8u) != 0) {
        failed |= vc_str_add(&header, " ");
    }
    if (failed != 0) {
        free(header.data);
        vc_err(error, error_capacity, "voice export: header build failed");
        return -1;
    }

    size_t total = 0;
    if (vc_add(8u, header.len, &total) != 0 || vc_add(total, offset, &total) !=
                                                   0) {
        free(header.data);
        vc_err(error, error_capacity, "voice export: file size overflow");
        return -1;
    }
    unsigned char *bytes = (unsigned char *)calloc(total, 1u);
    if (bytes == NULL) {
        free(header.data);
        vc_err(error, error_capacity, "voice export: out of memory");
        return -1;
    }
    vc_put_u64(bytes, (uint64_t)header.len);
    memcpy(bytes + 8u, header.data, header.len);
    free(header.data);

    unsigned char *cursor = bytes + 8u + header.len;
    const size_t half = mynah_transformer_ar_state_kv_half_floats(state);
    const size_t stride = heads * head_dim;
    for (size_t layer = 0; layer < config->num_layers; ++layer) {
        const float *kv = mynah_transformer_ar_state_kv(state, layer);
        if (kv == NULL) {
            free(bytes);
            vc_err(error, error_capacity,
                   "voice export: layer %zu has no KV cache", layer);
            return -1;
        }
        for (size_t part = 0; part < 2u; ++part) {
            const float *source = kv + part * half;
            for (size_t i = 0; i < positions * stride; ++i) {
                const float value = source[i];
                if (!isfinite(value)) {
                    free(bytes);
                    vc_err(error, error_capacity,
                           "voice export: layer %zu holds a non-finite value",
                           layer);
                    return -1;
                }
                if (dtype == MYNAH_VOICE_DTYPE_F16) {
                    vc_put_u16(cursor, vc_f32_to_f16(value));
                    cursor += 2;
                } else {
                    vc_put_f32(cursor, value);
                    cursor += 4;
                }
            }
        }
        vc_put_u64(cursor, (uint64_t)positions);
        cursor += 8;
    }
    *out = bytes;
    *size = total;
    return 0;
}

int mynah_voice_export(mynah_transformer_ar_state *state, const char *path,
                       mynah_voice_dtype dtype,
                       const mynah_voice_clone_consent *consent,
                       const char *model_revision, char *error,
                       size_t error_capacity) {
    if (error != NULL && error_capacity > 0) error[0] = '\0';
    if (path == NULL) {
        vc_err(error, error_capacity, "voice export: null path");
        return -1;
    }
    void *bytes = NULL;
    size_t size = 0;
    if (mynah_voice_serialise(state, dtype, consent, model_revision, &bytes,
                              &size, error, error_capacity) != 0) {
        return -1;
    }
    FILE *file = fopen(path, "wb");
    if (file == NULL) {
        free(bytes);
        vc_err(error, error_capacity, "voice export: cannot write %s", path);
        return -1;
    }
    const size_t written = fwrite(bytes, 1u, size, file);
    const int closed = fclose(file);
    free(bytes);
    if (written != size || closed != 0) {
        vc_err(error, error_capacity, "voice export: short write on %s", path);
        return -1;
    }
    return 0;
}

/* ------------------------------------------------------------- self test */

#define VC_FAIL(...)                                   \
    do {                                               \
        vc_err(error, error_capacity, __VA_ARGS__);    \
        goto done;                                     \
    } while (0)

/* Deterministic values in [-0.5, 0.5); a fixed LCG so a failure reproduces. */
static void vc_fill_random(float *values, size_t count, uint32_t seed) {
    uint32_t state = seed;
    for (size_t i = 0; i < count; ++i) {
        state = state * 1664525u + 1013904223u;
        values[i] = (float)((double)(state >> 8) / 16777216.0 - 0.5);
    }
}

/*
 * Naive convolution: pad the whole sequence on the left and slide.  This is
 * the definition `mynah_causal_conv1d` streams, written out independently so
 * the encoder composition is checked against something that shares no code
 * with it.  `in` is [in_ch][len], `out` is [out_ch][len / stride].
 */
static int vc_ref_conv(const float *in, size_t in_ch, size_t len,
                       const float *w, const float *b, size_t out_ch, size_t k,
                       size_t stride, size_t dilation, int replicate,
                       float *out) {
    const size_t pad = (k - 1u) * dilation + 1u - stride;
    const size_t plen = len + pad;
    float *padded = (float *)calloc(in_ch * plen, sizeof(float));
    if (padded == NULL) return -1;
    for (size_t c = 0; c < in_ch; ++c) {
        const float edge = replicate ? in[c * len] : 0.0f;
        for (size_t i = 0; i < pad; ++i) padded[c * plen + i] = edge;
        memcpy(padded + c * plen + pad, in + c * len, len * sizeof(float));
    }
    const size_t out_len = len / stride;
    for (size_t oc = 0; oc < out_ch; ++oc) {
        for (size_t n = 0; n < out_len; ++n) {
            double acc = (b != NULL) ? (double)b[oc] : 0.0;
            for (size_t j = 0; j < in_ch; ++j) {
                for (size_t t = 0; t < k; ++t) {
                    acc += (double)w[(oc * in_ch + j) * k + t] *
                           (double)padded[j * plen + n * stride + t * dilation];
                }
            }
            out[oc * out_len + n] = (float)acc;
        }
    }
    free(padded);
    return 0;
}

static void vc_ref_elu(float *values, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        if (values[i] <= 0.0f) values[i] = expf(values[i]) - 1.0f;
    }
}

/* ---- WAV --------------------------------------------------------------- */

static unsigned char *vc_build_wav(unsigned format, unsigned channels,
                                   unsigned rate, unsigned bits,
                                   const unsigned char *payload,
                                   size_t payload_size, size_t *size) {
    const size_t total = 44u + payload_size;
    unsigned char *bytes = (unsigned char *)calloc(total, 1u);
    if (bytes == NULL) return NULL;
    memcpy(bytes, "RIFF", 4);
    vc_put_u32(bytes + 4, (uint32_t)(total - 8u));
    memcpy(bytes + 8, "WAVEfmt ", 8);
    vc_put_u32(bytes + 16, 16u);
    vc_put_u16(bytes + 20, (uint16_t)format);
    vc_put_u16(bytes + 22, (uint16_t)channels);
    vc_put_u32(bytes + 24, rate);
    vc_put_u32(bytes + 28, rate * channels * (bits / 8u));
    vc_put_u16(bytes + 32, (uint16_t)(channels * (bits / 8u)));
    vc_put_u16(bytes + 34, (uint16_t)bits);
    memcpy(bytes + 36, "data", 4);
    vc_put_u32(bytes + 40, (uint32_t)payload_size);
    memcpy(bytes + 44, payload, payload_size);
    *size = total;
    return bytes;
}

static int vc_self_test_wav(char *error, size_t error_capacity) {
    int status = -1;
    unsigned char raw[64];
    unsigned char *file = NULL;
    size_t size = 0;
    mynah_audio_clip clip;
    memset(&clip, 0, sizeof(clip));

    /* PCM16 stereo: the mono result is the average of the two channels. */
    const int16_t pcm16[8] = {0, 0, 32767, -32768, 1000, 3000, -4000, -2000};
    for (size_t i = 0; i < 8u; ++i) {
        vc_put_u16(raw + i * 2u, (uint16_t)pcm16[i]);
    }
    file = vc_build_wav(1u, 2u, 8000u, 16u, raw, 16u, &size);
    if (file == NULL) VC_FAIL("wav test: out of memory");
    if (mynah_wav_decode_mono(file, size, &clip, error, error_capacity) != 0) {
        goto done;
    }
    if (clip.count != 4u || clip.sample_rate != 8000u) {
        VC_FAIL("wav test: PCM16 stereo gave %zu frames at %u Hz", clip.count,
                clip.sample_rate);
    }
    for (size_t i = 0; i < 4u; ++i) {
        const float want = 0.5f * ((float)pcm16[i * 2u] / 32768.0f +
                                   (float)pcm16[i * 2u + 1u] / 32768.0f);
        if (fabsf(clip.samples[i] - want) > 1e-7f) {
            VC_FAIL("wav test: PCM16 frame %zu is %.9g, want %.9g", i,
                    (double)clip.samples[i], (double)want);
        }
    }
    mynah_audio_clip_free(&clip);
    free(file);
    file = NULL;

    /* PCM24 mono. */
    const int32_t pcm24[3] = {0, 8388607, -8388608};
    for (size_t i = 0; i < 3u; ++i) {
        const uint32_t v = (uint32_t)pcm24[i];
        raw[i * 3u] = (unsigned char)(v & 0xFFu);
        raw[i * 3u + 1u] = (unsigned char)((v >> 8) & 0xFFu);
        raw[i * 3u + 2u] = (unsigned char)((v >> 16) & 0xFFu);
    }
    file = vc_build_wav(1u, 1u, 24000u, 24u, raw, 9u, &size);
    if (file == NULL) VC_FAIL("wav test: out of memory");
    if (mynah_wav_decode_mono(file, size, &clip, error, error_capacity) != 0) {
        goto done;
    }
    if (clip.count != 3u) VC_FAIL("wav test: PCM24 gave %zu frames", clip.count);
    if (fabsf(clip.samples[0]) > 1e-7f ||
        fabsf(clip.samples[1] - 8388607.0f / 8388608.0f) > 1e-7f ||
        fabsf(clip.samples[2] + 1.0f) > 1e-7f) {
        VC_FAIL("wav test: PCM24 decoded %.9g %.9g %.9g",
                (double)clip.samples[0], (double)clip.samples[1],
                (double)clip.samples[2]);
    }
    mynah_audio_clip_free(&clip);
    free(file);
    file = NULL;

    /* float32 mono, the format the oracle's reference.wav uses. */
    const float floats[4] = {0.0f, 0.25f, -0.75f, 1.0f};
    for (size_t i = 0; i < 4u; ++i) vc_put_f32(raw + i * 4u, floats[i]);
    file = vc_build_wav(3u, 1u, 24000u, 32u, raw, 16u, &size);
    if (file == NULL) VC_FAIL("wav test: out of memory");
    if (mynah_wav_decode_mono(file, size, &clip, error, error_capacity) != 0) {
        goto done;
    }
    if (clip.count != 4u) VC_FAIL("wav test: float32 gave %zu frames",
                                  clip.count);
    for (size_t i = 0; i < 4u; ++i) {
        if (clip.samples[i] != floats[i]) {
            VC_FAIL("wav test: float32 frame %zu is %.9g, want %.9g", i,
                    (double)clip.samples[i], (double)floats[i]);
        }
    }
    mynah_audio_clip_truncate(&clip, 2.0 / 24000.0);
    if (clip.count != 2u) {
        VC_FAIL("wav test: truncation left %zu frames", clip.count);
    }

    /* A file with no fmt chunk must be refused rather than guessed at. */
    mynah_audio_clip_free(&clip);
    memcpy(file + 12, "junk", 4);
    if (mynah_wav_decode_mono(file, size, &clip, NULL, 0) == 0) {
        mynah_audio_clip_free(&clip);
        VC_FAIL("wav test: a file without a fmt chunk was accepted");
    }
    status = 0;

done:
    mynah_audio_clip_free(&clip);
    free(file);
    return status;
}

/* ---- resampler --------------------------------------------------------- */

static int vc_self_test_resample(char *error, size_t error_capacity) {
    int status = -1;
    float *out = NULL;
    size_t produced = 0;
    float input[512];
    const double pi = 3.14159265358979323846;

    for (size_t i = 0; i < 512u; ++i) {
        input[i] = (float)sin(2.0 * pi * 3.0 * (double)i / 512.0);
    }

    /* Equal rates copy, and the gcd reduction has to see that. */
    if (mynah_audio_resample_poly(input, 512u, 48000u, 48000u, &out, &produced,
                                  error, error_capacity) != 0) {
        goto done;
    }
    if (produced != 512u || memcmp(out, input, sizeof(input)) != 0) {
        VC_FAIL("resample test: identity produced %zu samples", produced);
    }
    free(out);
    out = NULL;

    /* DC gain is one: a constant signal stays constant away from the edges. */
    for (size_t i = 0; i < 512u; ++i) input[i] = 0.75f;
    if (mynah_audio_resample_poly(input, 512u, 1u, 2u, &out, &produced, error,
                                  error_capacity) != 0) {
        goto done;
    }
    if (produced != 256u) {
        VC_FAIL("resample test: 2:1 of 512 gave %zu samples", produced);
    }
    for (size_t i = 32u; i < produced - 32u; ++i) {
        if (fabsf(out[i] - 0.75f) > 1e-3f) {
            VC_FAIL("resample test: DC at %zu is %.9g, want 0.75", i,
                    (double)out[i]);
        }
    }
    free(out);
    out = NULL;

    /*
     * A band-limited sine survives decimation with the right phase: output j
     * must equal the input at time j * down / up.  This is what catches a
     * group delay that is off by a sample, which no amplitude check sees.
     */
    for (size_t i = 0; i < 512u; ++i) {
        input[i] = (float)sin(2.0 * pi * 5.0 * (double)i / 512.0);
    }
    if (mynah_audio_resample_poly(input, 512u, 1u, 2u, &out, &produced, error,
                                  error_capacity) != 0) {
        goto done;
    }
    for (size_t i = 40u; i < produced - 40u; ++i) {
        const double want = sin(2.0 * pi * 5.0 * (double)(2u * i) / 512.0);
        if (fabs((double)out[i] - want) > 2e-3) {
            VC_FAIL("resample test: 2:1 sine at %zu is %.9g, want %.9g", i,
                    (double)out[i], want);
        }
    }
    free(out);
    out = NULL;

    /* Upsampling by a non-trivial ratio keeps the same waveform on the grid;
     * 3:2 exercises a phase that is neither 0 nor a whole input sample. */
    if (mynah_audio_resample_poly(input, 512u, 3u, 2u, &out, &produced, error,
                                  error_capacity) != 0) {
        goto done;
    }
    if (produced != 768u) {
        VC_FAIL("resample test: 3:2 of 512 gave %zu samples", produced);
    }
    for (size_t i = 60u; i < produced - 60u; ++i) {
        const double want =
            sin(2.0 * pi * 5.0 * ((double)i * 2.0 / 3.0) / 512.0);
        if (fabs((double)out[i] - want) > 3e-3) {
            VC_FAIL("resample test: 3:2 sine at %zu is %.9g, want %.9g", i,
                    (double)out[i], want);
        }
    }
    status = 0;

done:
    free(out);
    return status;
}

/* ---- encoder ----------------------------------------------------------- */

/* The tiny topology the encoder test runs, sized so the naive reference can be
 * written out by hand: one stage, ratio 2, 1 residual layer. */
#define VC_T_FILTERS 2u
#define VC_T_DIM 4u
#define VC_T_RATIO 2u
#define VC_T_LATENT 2u
#define VC_T_LEN 24u

typedef struct {
    float first_w[VC_T_FILTERS * 1u * 3u];
    float first_b[VC_T_FILTERS];
    float rb1_w[1u * VC_T_FILTERS * 3u];
    float rb1_b[1u];
    float rb2_w[VC_T_FILTERS * 1u * 1u];
    float rb2_b[VC_T_FILTERS];
    float stage_w[(VC_T_FILTERS * 2u) * VC_T_FILTERS * 4u];
    float stage_b[VC_T_FILTERS * 2u];
    float last_w[VC_T_DIM * (VC_T_FILTERS * 2u) * 2u];
    float last_b[VC_T_DIM];
    float down_w[VC_T_LATENT * VC_T_DIM * 4u];
    float proj_w[VC_T_DIM * VC_T_LATENT];
    float layer[4096];
} vc_test_weights;

static void vc_test_setup(vc_test_weights *store, mynah_voice_clone_config *cfg,
                          mynah_voice_clone_weights *weights,
                          mynah_conv_weights *stage,
                          mynah_seanet_resblock_weights *block,
                          mynah_transformer_ar_layer *layer,
                          const size_t *ratios) {
    vc_fill_random((float *)store, sizeof(*store) / sizeof(float), 12345u);

    mynah_voice_clone_config_defaults(cfg);
    cfg->seanet.channels = 1u;
    cfg->seanet.dimension = VC_T_DIM;
    cfg->seanet.n_filters = VC_T_FILTERS;
    cfg->seanet.n_residual_layers = 1u;
    cfg->seanet.ratios = ratios;
    cfg->seanet.n_ratios = 1u;
    cfg->seanet.kernel_size = 3u;
    cfg->seanet.residual_kernel_size = 3u;
    cfg->seanet.last_kernel_size = 2u;
    cfg->seanet.dilation_base = 2u;
    cfg->seanet.compress = 2u;
    cfg->seanet.elu_alpha = 1.0f;

    cfg->downsample.stride = 2u;
    cfg->downsample.in_channels = VC_T_DIM;
    cfg->downsample.out_channels = VC_T_LATENT;
    cfg->downsample.groups = 1u;

    cfg->encoder_transformer.d_model = VC_T_DIM;
    cfg->encoder_transformer.num_heads = 1u;
    cfg->encoder_transformer.head_dim = VC_T_DIM;
    cfg->encoder_transformer.num_layers = 1u;
    cfg->encoder_transformer.ffn_dim = 4u;
    cfg->encoder_transformer.context = 3u;

    cfg->backbone.d_model = VC_T_DIM;
    cfg->backbone.num_heads = 1u;
    cfg->backbone.head_dim = VC_T_DIM;
    cfg->backbone.num_layers = 1u;
    cfg->backbone.ffn_dim = 4u;

    cfg->sample_rate = 4u;
    cfg->samples_per_frame = VC_T_RATIO * cfg->downsample.stride;
    cfg->insert_bos_before_voice = 0;
    cfg->max_seconds = (double)VC_T_LEN / 4.0;
    cfg->chunk_latent_frames = 0;

    stage->weight = store->stage_w;
    stage->bias = store->stage_b;
    block->conv1.weight = store->rb1_w;
    block->conv1.bias = store->rb1_b;
    block->conv2.weight = store->rb2_w;
    block->conv2.bias = store->rb2_b;

    float *cursor = store->layer;
    memset(layer, 0, sizeof(*layer));
    layer->in_proj_weight = cursor;
    cursor += 3u * VC_T_DIM * VC_T_DIM;
    layer->out_proj_weight = cursor;
    cursor += VC_T_DIM * VC_T_DIM;
    layer->linear1_weight = cursor;
    cursor += 4u * VC_T_DIM;
    layer->linear2_weight = cursor;
    cursor += VC_T_DIM * 4u;
    layer->norm1_weight = cursor;
    cursor += VC_T_DIM;
    layer->norm1_bias = cursor;
    cursor += VC_T_DIM;
    layer->norm2_weight = cursor;
    cursor += VC_T_DIM;
    layer->norm2_bias = cursor;
    cursor += VC_T_DIM;
    layer->layer_scale_1 = cursor;
    cursor += VC_T_DIM;
    layer->layer_scale_2 = cursor;

    memset(weights, 0, sizeof(*weights));
    weights->first.weight = store->first_w;
    weights->first.bias = store->first_b;
    weights->stage_conv = stage;
    weights->blocks = block;
    weights->last.weight = store->last_w;
    weights->last.bias = store->last_b;
    weights->encoder_transformer.layers = layer;
    weights->downsample.weight = store->down_w;
    weights->downsample.bias = NULL;
    weights->speaker_proj = store->proj_w;
    weights->bos_before_voice = NULL;
}

/* conv -> resblock -> ELU -> strided conv -> ELU -> conv, written out. */
static int vc_ref_encoder(const vc_test_weights *store, const float *input,
                          size_t len, float *out) {
    int status = -1;
    const size_t half = len / VC_T_RATIO;
    float *x0 = (float *)calloc(VC_T_FILTERS * len, sizeof(float));
    float *h = (float *)calloc(VC_T_FILTERS * len, sizeof(float));
    float *v = (float *)calloc(VC_T_FILTERS * len, sizeof(float));
    float *s = (float *)calloc(VC_T_FILTERS * 2u * half, sizeof(float));
    if (x0 == NULL || h == NULL || v == NULL || s == NULL) goto done;

    if (vc_ref_conv(input, 1u, len, store->first_w, store->first_b,
                    VC_T_FILTERS, 3u, 1u, 1u, 0, x0) != 0) {
        goto done;
    }
    memcpy(v, x0, VC_T_FILTERS * len * sizeof(float));
    vc_ref_elu(v, VC_T_FILTERS * len);
    if (vc_ref_conv(v, VC_T_FILTERS, len, store->rb1_w, store->rb1_b, 1u, 3u,
                    1u, 1u, 0, h) != 0) {
        goto done;
    }
    vc_ref_elu(h, len);
    if (vc_ref_conv(h, 1u, len, store->rb2_w, store->rb2_b, VC_T_FILTERS, 1u,
                    1u, 1u, 0, v) != 0) {
        goto done;
    }
    for (size_t i = 0; i < VC_T_FILTERS * len; ++i) x0[i] += v[i];
    vc_ref_elu(x0, VC_T_FILTERS * len);
    if (vc_ref_conv(x0, VC_T_FILTERS, len, store->stage_w, store->stage_b,
                    VC_T_FILTERS * 2u, 4u, VC_T_RATIO, 1u, 0, s) != 0) {
        goto done;
    }
    vc_ref_elu(s, VC_T_FILTERS * 2u * half);
    if (vc_ref_conv(s, VC_T_FILTERS * 2u, half, store->last_w, store->last_b,
                    VC_T_DIM, 2u, 1u, 1u, 0, out) != 0) {
        goto done;
    }
    status = 0;

done:
    free(x0);
    free(h);
    free(v);
    free(s);
    return status;
}

static int vc_self_test_encoder(char *error, size_t error_capacity) {
    int status = -1;
    const size_t ratios[1] = {VC_T_RATIO};
    vc_test_weights *store = NULL;
    mynah_voice_encoder *one = NULL;
    mynah_voice_encoder *many = NULL;
    float *want = NULL;
    float *down_ref = NULL;
    mynah_seanet_downsample *down = NULL;
    mynah_voice_clone_config cfg;
    mynah_voice_clone_weights weights;
    mynah_conv_weights stage;
    mynah_seanet_resblock_weights block;
    mynah_transformer_ar_layer layer;
    mynah_voice_clone_consent consent;
    float input[VC_T_LEN];

    store = (vc_test_weights *)calloc(1u, sizeof(*store));
    want = (float *)calloc(VC_T_DIM * VC_T_LEN, sizeof(float));
    down_ref = (float *)calloc(VC_T_LATENT * VC_T_LEN, sizeof(float));
    if (store == NULL || want == NULL || down_ref == NULL) {
        VC_FAIL("encoder test: out of memory");
    }
    vc_test_setup(store, &cfg, &weights, &stage, &block, &layer, ratios);
    vc_fill_random(input, VC_T_LEN, 987u);

    memset(&consent, 0, sizeof(consent));
    consent.affirmed = 1;
    consent.source = "self test";

    /* One chunk covering the whole clip, against the naive reference. */
    cfg.chunk_latent_frames = VC_T_LEN / cfg.samples_per_frame;
    one = mynah_voice_encoder_create(&cfg, error, error_capacity);
    if (one == NULL) goto done;
    const float *got = NULL;
    if (vc_encoder_forward(one, &weights, input, VC_T_LEN, &got) != 0) {
        VC_FAIL("encoder test: the forward pass failed");
    }
    if (vc_ref_encoder(store, input, VC_T_LEN, want) != 0) {
        VC_FAIL("encoder test: the reference failed");
    }
    const size_t frames = VC_T_LEN / VC_T_RATIO;
    for (size_t i = 0; i < VC_T_DIM * frames; ++i) {
        if (fabsf(got[i] - want[i]) > 1e-5f) {
            VC_FAIL("encoder test: output %zu is %.9g, want %.9g", i,
                    (double)got[i], (double)want[i]);
        }
    }

    /* The replicate-padded downsample against the same naive reference: the
     * edge is the first frame repeated, not zero. */
    down = mynah_seanet_downsample_create(&cfg.downsample, frames, error,
                                          error_capacity);
    if (down == NULL) goto done;
    if (mynah_seanet_downsample_apply(down, &weights.downsample, got, frames,
                                      down_ref) != 0) {
        VC_FAIL("encoder test: the downsample failed");
    }
    float *replicate = (float *)calloc(VC_T_LATENT * frames, sizeof(float));
    if (replicate == NULL) VC_FAIL("encoder test: out of memory");
    if (vc_ref_conv(got, VC_T_DIM, frames, store->down_w, NULL, VC_T_LATENT,
                    4u, 2u, 1u, 1, replicate) != 0) {
        free(replicate);
        VC_FAIL("encoder test: the downsample reference failed");
    }
    for (size_t i = 0; i < VC_T_LATENT * (frames / 2u); ++i) {
        if (fabsf(down_ref[i] - replicate[i]) > 1e-5f) {
            const double a = (double)down_ref[i];
            const double b = (double)replicate[i];
            free(replicate);
            VC_FAIL("encoder test: downsample %zu is %.9g, want %.9g", i, a, b);
        }
    }
    free(replicate);

    /* The whole pipeline, chunked against unchunked: convolutions carry their
     * ring buffers and the transformer its KV, so this must be exact. */
    if (mynah_voice_encoder_encode(one, &weights, &consent, input, VC_T_LEN,
                                   error, error_capacity) != 0) {
        goto done;
    }
    cfg.chunk_latent_frames = 1u;
    many = mynah_voice_encoder_create(&cfg, error, error_capacity);
    if (many == NULL) goto done;
    if (mynah_voice_encoder_encode(many, &weights, &consent, input, VC_T_LEN,
                                   error, error_capacity) != 0) {
        goto done;
    }
    if (mynah_voice_encoder_frames(one) != mynah_voice_encoder_frames(many) ||
        mynah_voice_encoder_frames(one) != VC_T_LEN / cfg.samples_per_frame) {
        VC_FAIL("encoder test: %zu frames in one chunk but %zu in six",
                mynah_voice_encoder_frames(one),
                mynah_voice_encoder_frames(many));
    }
    const float *cond_one = mynah_voice_encoder_conditioning(one);
    const float *cond_many = mynah_voice_encoder_conditioning(many);
    for (size_t i = 0; i < mynah_voice_encoder_frames(one) * VC_T_DIM; ++i) {
        if (!isfinite(cond_one[i])) {
            VC_FAIL("encoder test: conditioning %zu is not finite", i);
        }
        if (fabsf(cond_one[i] - cond_many[i]) > 1e-5f) {
            VC_FAIL("encoder test: chunking changed conditioning %zu: %.9g vs "
                    "%.9g",
                    i, (double)cond_one[i], (double)cond_many[i]);
        }
    }

    /* Consent is not optional and not defaulted. */
    if (mynah_voice_encoder_encode(one, &weights, NULL, input, VC_T_LEN, NULL,
                                   0) == 0) {
        VC_FAIL("encoder test: cloning without consent was allowed");
    }
    consent.affirmed = 0;
    if (mynah_voice_encoder_encode(one, &weights, &consent, input, VC_T_LEN,
                                   NULL, 0) == 0) {
        VC_FAIL("encoder test: cloning with unaffirmed consent was allowed");
    }
    status = 0;

done:
    mynah_seanet_downsample_destroy(down);
    mynah_voice_encoder_destroy(one);
    mynah_voice_encoder_destroy(many);
    free(store);
    free(want);
    free(down_ref);
    return status;
}

/* ---- export ------------------------------------------------------------ */

static int vc_self_test_export(char *error, size_t error_capacity) {
    int status = -1;
    mynah_transformer_ar_state *state = NULL;
    unsigned char *bytes = NULL;
    size_t size = 0;
    mynah_transformer_ar_config config;
    mynah_voice_clone_consent consent;

    /* f32 -> f16 on the values that actually appear in a KV cache, plus the
     * boundaries where a hand-rolled conversion goes wrong. */
    const float pairs[6] = {0.0f, 1.0f, -2.0f, 0.5f, 65504.0f, 1e-8f};
    const uint16_t expect[6] = {0x0000u, 0x3C00u, 0xC000u,
                                0x3800u, 0x7BFFu, 0x0000u};
    for (size_t i = 0; i < 6u; ++i) {
        const uint16_t got = vc_f32_to_f16(pairs[i]);
        if (got != expect[i]) {
            VC_FAIL("export test: f16 of %.9g is 0x%04x, want 0x%04x",
                    (double)pairs[i], (unsigned)got, (unsigned)expect[i]);
        }
    }

    mynah_transformer_ar_config_defaults(&config);
    config.d_model = 4u;
    config.num_heads = 1u;
    config.head_dim = 4u;
    config.num_layers = 2u;
    config.ffn_dim = 4u;
    config.max_seq_len = 8u;
    state = mynah_transformer_ar_state_new(&config, error, error_capacity);
    if (state == NULL) goto done;

    if (mynah_voice_serialise(state, MYNAH_VOICE_DTYPE_F16, NULL, NULL,
                              (void **)&bytes, &size, NULL, 0) == 0) {
        VC_FAIL("export test: an empty KV cache was exported");
    }
    if (mynah_transformer_ar_state_set_offset(state, 3u, error,
                                              error_capacity) != 0) {
        goto done;
    }
    memset(&consent, 0, sizeof(consent));
    consent.affirmed = 1;
    consent.source = "a \"quoted\" path";
    consent.affirmation = "self test";
    if (mynah_voice_serialise(state, MYNAH_VOICE_DTYPE_F16, &consent, "rev0",
                              (void **)&bytes, &size, error,
                              error_capacity) != 0) {
        goto done;
    }
    uint64_t header_len = 0;
    for (size_t i = 0; i < 8u; ++i) {
        header_len |= (uint64_t)bytes[i] << (8u * i);
    }
    const size_t payload = 2u * (2u * 3u * 1u * 4u * 2u + 8u);
    if (size != 8u + (size_t)header_len + payload) {
        VC_FAIL("export test: %zu bytes for a %zu-byte header and %zu of data",
                size, (size_t)header_len, payload);
    }
    if (((8u + (size_t)header_len) % 8u) != 0) {
        VC_FAIL("export test: the data section starts at %zu, not 8-aligned",
                8u + (size_t)header_len);
    }
    if (memchr(bytes + 8, '\0', (size_t)header_len) != NULL) {
        VC_FAIL("export test: the header holds a NUL");
    }
    char *header = (char *)calloc((size_t)header_len + 1u, 1u);
    if (header == NULL) VC_FAIL("export test: out of memory");
    memcpy(header, bytes + 8, (size_t)header_len);
    const int complete =
        strstr(header, "\"transformer.layers.0.self_attn/cache\"") != NULL &&
        strstr(header, "\"transformer.layers.1.self_attn/offset\"") != NULL &&
        strstr(header, "\"shape\":[2,1,3,1,4]") != NULL &&
        strstr(header, "\"dtype\":\"F16\"") != NULL &&
        strstr(header, "\"dtype\":\"I64\"") != NULL &&
        strstr(header, "a \\\"quoted\\\" path") != NULL;
    free(header);
    if (!complete) VC_FAIL("export test: the header is missing an entry");

    /* The trailing I64 of each layer is the frame count. */
    for (size_t layer = 0; layer < 2u; ++layer) {
        const size_t at = 8u + (size_t)header_len +
                          (layer + 1u) * (2u * 3u * 4u * 2u) + layer * 8u;
        uint64_t value = 0;
        for (size_t i = 0; i < 8u; ++i) {
            value |= (uint64_t)bytes[at + i] << (8u * i);
        }
        if (value != 3u) {
            VC_FAIL("export test: layer %zu records offset %llu, want 3", layer,
                    (unsigned long long)value);
        }
    }
    status = 0;

done:
    free(bytes);
    mynah_transformer_ar_state_free(state);
    return status;
}

int mynah_voice_clone_self_test(char *error, size_t error_capacity) {
    if (error != NULL && error_capacity > 0) error[0] = '\0';
    if (vc_self_test_wav(error, error_capacity) != 0) return -1;
    if (vc_self_test_resample(error, error_capacity) != 0) return -1;
    if (vc_self_test_encoder(error, error_capacity) != 0) return -1;
    if (vc_self_test_export(error, error_capacity) != 0) return -1;
    return 0;
}

#undef VC_FAIL
