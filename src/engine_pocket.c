/* PocketTTS engine. Contract, measurements and known limits: engine_pocket.h.
 *
 * SPDX-License-Identifier: MIT */
#include "engine_pocket.h"

#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ingot/safetensors.h"

#include "flow_head.h"
#include "kernels.h"
#include "mynah_tts_internal.h"
#include "mynah_util.h"
#include "seanet.h"
#include "tokenizer_sentencepiece.h"
#include "transformer_ar.h"
#include "weights.h"

#define POCKET_ENGINE_NAME "pocket"
#define POCKET_MAX_RATIOS 8u
#define POCKET_NAME_MAX 256u
#define POCKET_PATH_MAX 4096u
#define POCKET_MANIFEST_MAX (4u * 1024u * 1024u)

/* ------------------------------------------------------------------ errors */

#if defined(__GNUC__)
static void pocket_error(char *error, size_t capacity, const char *format, ...)
    __attribute__((format(printf, 3, 4)));
#endif

static void pocket_error(char *error, size_t capacity, const char *format, ...) {
    if (error == NULL || capacity == 0) return;
    va_list args;
    va_start(args, format);
    vsnprintf(error, capacity, format, args);
    va_end(args);
}

static int pocket_mul(size_t a, size_t b, size_t *out) {
    if (a != 0 && b > SIZE_MAX / a) return -1;
    *out = a * b;
    return 0;
}

static int pocket_add(size_t a, size_t b, size_t *out) {
    if (a > SIZE_MAX - b) return -1;
    *out = a + b;
    return 0;
}

static char *pocket_strdup(const char *value, size_t length) {
    char *copy = (char *)malloc(length + 1u);
    if (copy == NULL) return NULL;
    memcpy(copy, value, length);
    copy[length] = '\0';
    return copy;
}

/* ------------------------------------------------------------------- JSON
 *
 * A manifest reader, not a JSON library: it walks the document structurally
 * (so a key nested inside another object is never mistaken for a top-level
 * one) and it never allocates. `substr`-style scanning was the alternative and
 * it silently confuses "weights"."tts" with a top-level "tts".
 */

static const char *pj_ws(const char *p) {
    while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') ++p;
    return p;
}

/* `p` at the opening quote; returns just past the closing quote, or NULL. */
static const char *pj_end_string(const char *p) {
    if (*p != '"') return NULL;
    ++p;
    while (*p != '\0') {
        if (*p == '\\') {
            if (p[1] == '\0') return NULL;
            p += 2;
            continue;
        }
        if (*p == '"') return p + 1;
        ++p;
    }
    return NULL;
}

static const char *pj_end_value(const char *p) {
    p = pj_ws(p);
    if (*p == '"') return pj_end_string(p);
    if (*p == '{' || *p == '[') {
        /* Only one bracket kind is counted. JSON nests properly, so the other
         * kind is always balanced inside and cannot unbalance this count. */
        const char open = *p;
        const char close = (open == '{') ? '}' : ']';
        size_t depth = 0;
        while (*p != '\0') {
            if (*p == '"') {
                const char *s = pj_end_string(p);
                if (s == NULL) return NULL;
                p = s;
                continue;
            }
            if (*p == open) {
                ++depth;
            } else if (*p == close) {
                --depth;
                if (depth == 0) return p + 1;
            }
            ++p;
        }
        return NULL;
    }
    while (*p != '\0' && *p != ',' && *p != '}' && *p != ']' && *p != ' ' &&
           *p != '\t' && *p != '\r' && *p != '\n') {
        ++p;
    }
    return p;
}

/* The value of `key` in the object starting at `object`, or NULL. */
static const char *pj_object_get(const char *object, const char *key) {
    if (object == NULL || key == NULL) return NULL;
    const char *p = pj_ws(object);
    if (*p != '{') return NULL;
    ++p;
    const size_t key_length = strlen(key);
    for (;;) {
        p = pj_ws(p);
        if (*p != '"') return NULL;
        const char *name = p + 1;
        const char *name_end = pj_end_string(p);
        if (name_end == NULL) return NULL;
        const size_t name_length = (size_t)(name_end - name) - 1u;
        p = pj_ws(name_end);
        if (*p != ':') return NULL;
        const char *value = pj_ws(p + 1);
        if (name_length == key_length && memcmp(name, key, key_length) == 0) {
            return value;
        }
        p = pj_end_value(value);
        if (p == NULL) return NULL;
        p = pj_ws(p);
        if (*p != ',') return NULL;
        ++p;
    }
}

static const char *pj_array_first(const char *value) {
    if (value == NULL) return NULL;
    const char *p = pj_ws(value);
    if (*p != '[') return NULL;
    p = pj_ws(p + 1);
    return (*p == ']' || *p == '\0') ? NULL : p;
}

static const char *pj_array_next(const char *element) {
    const char *p = pj_end_value(element);
    if (p == NULL) return NULL;
    p = pj_ws(p);
    if (*p != ',') return NULL;
    p = pj_ws(p + 1);
    return (*p == ']' || *p == '\0') ? NULL : p;
}

static int pj_number(const char *value, double *out) {
    if (value == NULL) return -1;
    char *end = NULL;
    errno = 0;
    const double parsed = strtod(value, &end);
    if (errno != 0 || end == value) return -1;
    *out = parsed;
    return 0;
}

static int pj_bool(const char *value, int *out) {
    if (value == NULL) return -1;
    if (strncmp(value, "true", 4) == 0) {
        *out = 1;
        return 0;
    }
    if (strncmp(value, "false", 5) == 0) {
        *out = 0;
        return 0;
    }
    return -1;
}

static int pj_string_copy(const char *value, char *out, size_t capacity) {
    if (value == NULL || out == NULL || capacity == 0 || *value != '"') return -1;
    ++value;
    size_t used = 0;
    while (*value != '\0' && *value != '"') {
        char c = *value++;
        if (c == '\\') {
            const char escape = *value++;
            switch (escape) {
                case '"': c = '"'; break;
                case '\\': c = '\\'; break;
                case '/': c = '/'; break;
                case 'n': c = '\n'; break;
                case 't': c = '\t'; break;
                case 'r': c = '\r'; break;
                default: return -1; /* \u and friends: refuse, never guess */
            }
        }
        if (used + 1u >= capacity) return -1;
        out[used++] = c;
    }
    if (*value != '"') return -1;
    out[used] = '\0';
    return 0;
}

/* ------------------------------------------------------- manifest getters */

static int cfg_size_value(const char *value, size_t *out) {
    double number = 0.0;
    if (pj_number(value, &number) != 0) return -1;
    if (!(number >= 0.0) || number > (double)(SIZE_MAX / 2u)) return -1;
    const double rounded = floor(number + 0.5);
    if (fabs(number - rounded) > 1e-6) return -1;
    *out = (size_t)rounded;
    return 0;
}

static int cfg_need_size(const char *root, const char *key, size_t *out,
                         char *error, size_t capacity) {
    if (cfg_size_value(pj_object_get(root, key), out) != 0) {
        pocket_error(error, capacity,
                     "model.json: \"%s\" is missing or not a whole number", key);
        return -1;
    }
    return 0;
}

static void cfg_opt_size(const char *root, const char *key, size_t *out,
                         size_t fallback) {
    if (cfg_size_value(pj_object_get(root, key), out) != 0) *out = fallback;
}

static int cfg_need_double(const char *root, const char *key, double *out,
                           char *error, size_t capacity) {
    if (pj_number(pj_object_get(root, key), out) != 0) {
        pocket_error(error, capacity, "model.json: \"%s\" is missing or not a number",
                     key);
        return -1;
    }
    return 0;
}

static void cfg_opt_double(const char *root, const char *key, double *out,
                           double fallback) {
    if (pj_number(pj_object_get(root, key), out) != 0) *out = fallback;
}

static void cfg_opt_bool(const char *root, const char *key, int *out, int fallback) {
    if (pj_bool(pj_object_get(root, key), out) != 0) *out = fallback;
}

static void cfg_opt_string(const char *root, const char *key, char *out,
                           size_t capacity, const char *fallback) {
    if (pj_string_copy(pj_object_get(root, key), out, capacity) != 0) {
        snprintf(out, capacity, "%s", fallback);
    }
}

/* ------------------------------------------------------------ the manifest */

typedef struct {
    /* backbone */
    size_t hidden_dim;
    size_t heads;
    size_t head_dim;
    size_t layers;
    size_t ffn_dim;
    float layernorm_eps;

    /* text conditioning */
    size_t embed_rows;
    size_t vocab_size;
    size_t padding_id;
    size_t max_tokens_per_chunk;

    /* latent space and flow head */
    size_t latent_dim;
    size_t flow_dim;
    size_t flow_depth;
    size_t flow_time_conds;
    size_t flow_freqs;
    size_t flow_decode_steps;
    float flow_layernorm_eps;
    float eos_threshold;
    float temperature;

    /* codec */
    size_t codec_dim;
    size_t ratios[POCKET_MAX_RATIOS];
    size_t n_ratios;
    size_t upsample_stride;
    size_t codec_tf_dim;
    size_t codec_tf_ffn;
    size_t codec_tf_heads;
    size_t codec_tf_layers;
    size_t codec_tf_context;
    size_t audio_channels;
    size_t samples_per_frame;

    /* SEANet shape, derived from the tensors unless the manifest declares it */
    size_t n_filters;
    size_t n_residual_layers;
    size_t compress;
    size_t kernel_size;
    size_t residual_kernel_size;
    size_t last_kernel_size;
    size_t dilation_base;
    float elu_alpha;

    /* driver-facing */
    unsigned sample_rate;
    double frame_rate;
    size_t speaker_count;
    size_t min_audio_frames;
    size_t audio_emit_frames;
    size_t frames_after_eos;
    size_t default_max_steps;
    int uses_cfg;

    /* cloning only, kept so the generation difference stays visible */
    int insert_bos_before_voice;
    size_t speaker_proj_input_dim;

    char weights_tts[128];
    char tokenizer_file[128];
    char speakers_file[128];
    char voices_dir[128];
} pocket_config;

typedef struct {
    char *name;
    char *file; /* relative to the pack directory */
} pocket_voice;

/* ---------------------------------------------------------- model weights */

struct mynah_engine_state {
    pocket_config cfg;

    char *model_dir;
    mynah_weights *weights;
    int owns_weights;

    /* backbone */
    mynah_transformer_ar_layer *backbone_layers;
    mynah_transformer_ar_weights backbone;

    /* flow head */
    mynah_flow_time_embed_weights *time_embed;
    mynah_flow_res_block_weights *res_blocks;
    mynah_flow_head_weights flow;

    /* Mimi decoder transformer */
    mynah_transformer_ar_layer *codec_layers;
    mynah_transformer_ar_weights codec_transformer;

    /* Mimi upsample + SEANet decoder */
    mynah_conv_weights upsample;
    mynah_conv_weights *decoder_convtr;
    mynah_seanet_resblock_weights *decoder_blocks;
    mynah_seanet_decoder_weights decoder;

    /* single tensors */
    const float *embed_table;
    const float *bos_emb;
    const float *emb_mean;
    const float *emb_std;
    const float *input_linear;
    const float *out_eos_weight;
    const float *out_eos_bias;
    const float *quantizer_proj;
    const float *speaker_proj;     /* cloning only; may be NULL */
    const float *bos_before_voice; /* cloning only; may be NULL */

    pocket_voice *voices;
    size_t voice_count;

    mynah_sp *tokenizer;
};

/* ------------------------------------------------------------- per request */

struct mynah_engine_ctx {
    mynah_engine_state *state;

    int *text_ids;
    size_t text_length;
    unsigned speaker;
    float temperature;
    float noise_std;
    size_t max_steps;

    mynah_transformer_ar_state *backbone;
    mynah_flow_head *flow;
    mynah_transformer_ar_state *codec_transformer;
    mynah_seanet_state *codec;

    ingot_st *voice_file;
    size_t voice_positions;
    float *voice_kv; /* [2][voice_positions][heads][head_dim] */

    float *text_embed; /* [text_length][hidden_dim] */
    float *step_input; /* [hidden_dim] */
    float *hidden;     /* [hidden_dim] */
    float *noise;      /* [latent_dim] */
    float *flow_out;   /* [latent_dim] */
    float *latents;    /* [max_steps][latent_dim] */
    float *denorm;     /* [latent_dim] */
    float *codec_in;   /* [codec_dim] */
    float *codec_up;   /* [codec_dim][upsample_stride] */
    float *codec_seq;  /* [upsample_stride][codec_tf_dim] */
    float *codec_out;  /* [upsample_stride][codec_tf_dim] */
    float *codec_back; /* [codec_dim][upsample_stride] */
    float *pcm;        /* [samples_per_frame] */

    size_t frames;
    size_t decoded_frames;
    size_t step;
    size_t frames_after_eos;
    size_t eos_step; /* SIZE_MAX until the logit first crosses the threshold */
    int prepared;
    int eos;
    /* A per-request failure leaves the backbone one position ahead of the
     * frame history, which no later call can reconcile. The request is over;
     * saying so is the difference between a failed request and a context that
     * keeps generating from a state nothing produced. */
    int broken;
    float eos_logit;

    uint64_t seed;
    uint64_t rng;
    int have_spare;
    float spare;

    mynah_pocket_noise_fn noise_fn;
    void *noise_user;
};

struct mynah_engine_scratch {
    size_t batch;
};

/* --------------------------------------------------------------- the RNG
 *
 * Per context, never global (CLAUDE.md rule 3): two requests in one process
 * must not be able to consume each other's draws. splitmix64 plus Box-Muller;
 * the spare normal is kept so a frame costs one transcendental pair per two
 * values rather than per value. */

static uint64_t pocket_rng_next(uint64_t *state) {
    *state += UINT64_C(0x9E3779B97F4A7C15);
    uint64_t z = *state;
    z = (z ^ (z >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94D049BB133111EB);
    return z ^ (z >> 31);
}

/* Uniform in (0, 1]: never 0, because log() of it is taken. */
static double pocket_rng_unit(uint64_t *state) {
    const uint64_t bits = pocket_rng_next(state) >> 11;
    return ((double)bits + 1.0) * (1.0 / 9007199254740993.0);
}

static float pocket_rng_normal(mynah_engine_ctx *ctx) {
    if (ctx->have_spare) {
        ctx->have_spare = 0;
        return ctx->spare;
    }
    const double u1 = pocket_rng_unit(&ctx->rng);
    const double u2 = pocket_rng_unit(&ctx->rng);
    const double radius = sqrt(-2.0 * log(u1));
    const double angle = 6.283185307179586476925286766559 * u2;
    ctx->spare = (float)(radius * sin(angle));
    ctx->have_spare = 1;
    return (float)(radius * cos(angle));
}

/* ------------------------------------------------------ tensor resolution */

/* Resolves `name` and checks its shape. A zero in `shape` means "any extent".
 * Every dimension that the manifest already declares is checked here, so a
 * pack whose numbers disagree with its tensors fails at load with the name in
 * the message instead of producing plausible noise. */
static int pocket_tensor(const mynah_weights *weights, const char *name,
                         size_t rank, const size_t *shape, const float **out,
                         char *error, size_t capacity) {
    mynah_tensor tensor;
    if (mynah_tensor_get(weights, name, &tensor, error, capacity) != 0) return -1;
    if (tensor.rank != rank) {
        pocket_error(error, capacity, "%s: rank %zu, expected %zu", name,
                     tensor.rank, rank);
        return -1;
    }
    for (size_t d = 0; d < rank; ++d) {
        if (shape[d] != 0 && tensor.shape[d] != shape[d]) {
            pocket_error(error, capacity, "%s: dim %zu is %zu, expected %zu", name,
                         d, tensor.shape[d], shape[d]);
            return -1;
        }
    }
    *out = tensor.data;
    return 0;
}

/* Same, but a missing tensor is not an error: *out stays NULL. */
static int pocket_tensor_optional(const mynah_weights *weights, const char *name,
                                  size_t rank, const size_t *shape,
                                  const float **out, char *error,
                                  size_t capacity) {
    mynah_tensor tensor;
    *out = NULL;
    if (mynah_weights_get(weights, name, &tensor) != 0) return 0;
    return pocket_tensor(weights, name, rank, shape, out, error, capacity);
}

static int pocket_shape_of(const mynah_weights *weights, const char *name,
                           mynah_tensor *out, char *error, size_t capacity) {
    return mynah_tensor_get(weights, name, out, error, capacity);
}

/* ------------------------------------------------------------ config load */

static char *pocket_read_text(const char *path, size_t limit, char *error,
                              size_t capacity) {
    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        pocket_error(error, capacity, "cannot open %s", path);
        return NULL;
    }
    char *data = NULL;
    size_t size = 0;
    size_t used = 0;
    for (;;) {
        if (used == size) {
            const size_t grown = (size == 0) ? 8192u : size * 2u;
            if (grown > limit + 1u) {
                free(data);
                fclose(file);
                pocket_error(error, capacity, "%s is larger than %zu bytes", path,
                             limit);
                return NULL;
            }
            char *bigger = (char *)realloc(data, grown);
            if (bigger == NULL) {
                free(data);
                fclose(file);
                pocket_error(error, capacity, "out of memory reading %s", path);
                return NULL;
            }
            data = bigger;
            size = grown;
        }
        const size_t got = fread(data + used, 1, size - used, file);
        used += got;
        if (got == 0) break;
    }
    const int failed = ferror(file);
    fclose(file);
    if (failed || data == NULL) {
        free(data);
        pocket_error(error, capacity, "cannot read %s", path);
        return NULL;
    }
    if (used == size) {
        char *bigger = (char *)realloc(data, size + 1u);
        if (bigger == NULL) {
            free(data);
            pocket_error(error, capacity, "out of memory reading %s", path);
            return NULL;
        }
        data = bigger;
    }
    data[used] = '\0';
    return data;
}

static int pocket_join(char *out, size_t capacity, const char *directory,
                       const char *name, char *error, size_t error_capacity) {
    const int written = snprintf(out, capacity, "%s/%s", directory, name);
    if (written <= 0 || (size_t)written >= capacity) {
        pocket_error(error, error_capacity, "path is too long: %s/%s", directory,
                     name);
        return -1;
    }
    return 0;
}

static int pocket_config_load(const char *manifest, pocket_config *cfg,
                              char *error, size_t capacity) {
    memset(cfg, 0, sizeof(*cfg));

    char engine[32];
    if (pj_string_copy(pj_object_get(manifest, "engine"), engine, sizeof(engine)) != 0 ||
        strcmp(engine, POCKET_ENGINE_NAME) != 0) {
        pocket_error(error, capacity,
                     "model.json does not declare engine \"" POCKET_ENGINE_NAME "\"");
        return -1;
    }

    double number = 0.0;
    if (cfg_need_size(manifest, "hidden_dim", &cfg->hidden_dim, error, capacity) != 0 ||
        cfg_need_size(manifest, "attention_heads", &cfg->heads, error, capacity) != 0 ||
        cfg_need_size(manifest, "head_dim", &cfg->head_dim, error, capacity) != 0 ||
        cfg_need_size(manifest, "transformer_layers", &cfg->layers, error, capacity) != 0 ||
        cfg_need_size(manifest, "ffn_dim", &cfg->ffn_dim, error, capacity) != 0 ||
        cfg_need_size(manifest, "text_embedding_rows", &cfg->embed_rows, error, capacity) != 0 ||
        cfg_need_size(manifest, "text_vocab_size", &cfg->vocab_size, error, capacity) != 0 ||
        cfg_need_size(manifest, "latent_dim", &cfg->latent_dim, error, capacity) != 0 ||
        cfg_need_size(manifest, "flow_dim", &cfg->flow_dim, error, capacity) != 0 ||
        cfg_need_size(manifest, "flow_res_blocks", &cfg->flow_depth, error, capacity) != 0 ||
        cfg_need_size(manifest, "flow_time_conditions", &cfg->flow_time_conds, error, capacity) != 0 ||
        cfg_need_size(manifest, "flow_time_freqs", &cfg->flow_freqs, error, capacity) != 0 ||
        cfg_need_size(manifest, "codec_dim", &cfg->codec_dim, error, capacity) != 0 ||
        cfg_need_size(manifest, "codec_upsample_stride", &cfg->upsample_stride, error, capacity) != 0 ||
        cfg_need_size(manifest, "codec_transformer_dim", &cfg->codec_tf_dim, error, capacity) != 0 ||
        cfg_need_size(manifest, "codec_transformer_ffn", &cfg->codec_tf_ffn, error, capacity) != 0 ||
        cfg_need_size(manifest, "codec_transformer_heads", &cfg->codec_tf_heads, error, capacity) != 0 ||
        cfg_need_size(manifest, "codec_transformer_layers", &cfg->codec_tf_layers, error, capacity) != 0 ||
        cfg_need_size(manifest, "codec_transformer_context", &cfg->codec_tf_context, error, capacity) != 0 ||
        cfg_need_size(manifest, "audio_channels", &cfg->audio_channels, error, capacity) != 0 ||
        cfg_need_size(manifest, "samples_per_frame", &cfg->samples_per_frame, error, capacity) != 0 ||
        cfg_need_size(manifest, "speaker_count", &cfg->speaker_count, error, capacity) != 0) {
        return -1;
    }

    if (cfg_need_double(manifest, "sample_rate", &number, error, capacity) != 0) return -1;
    if (!(number > 0.0) || number > (double)UINT_MAX) {
        pocket_error(error, capacity, "model.json: sample_rate is out of range");
        return -1;
    }
    cfg->sample_rate = (unsigned)number;
    if (cfg_need_double(manifest, "frame_rate", &cfg->frame_rate, error, capacity) != 0) {
        return -1;
    }
    if (!(cfg->frame_rate > 0.0)) {
        pocket_error(error, capacity, "model.json: frame_rate must be positive");
        return -1;
    }
    if (cfg_need_double(manifest, "eos_threshold", &number, error, capacity) != 0) return -1;
    cfg->eos_threshold = (float)number;
    if (cfg_need_double(manifest, "temperature", &number, error, capacity) != 0) return -1;
    if (!(number >= 0.0)) {
        pocket_error(error, capacity, "model.json: temperature must not be negative");
        return -1;
    }
    cfg->temperature = (float)number;

    cfg_opt_double(manifest, "layernorm_eps", &number, 1e-5);
    cfg->layernorm_eps = (float)number;
    cfg_opt_double(manifest, "flow_layernorm_eps", &number, 1e-6);
    cfg->flow_layernorm_eps = (float)number;

    cfg_opt_size(manifest, "text_padding_id", &cfg->padding_id, cfg->vocab_size);
    cfg_opt_size(manifest, "max_tokens_per_chunk", &cfg->max_tokens_per_chunk, 0u);
    cfg_opt_size(manifest, "flow_decode_steps", &cfg->flow_decode_steps, 1u);
    cfg_opt_size(manifest, "speaker_proj_input_dim", &cfg->speaker_proj_input_dim,
                 cfg->latent_dim);
    cfg_opt_bool(manifest, "insert_bos_before_voice", &cfg->insert_bos_before_voice, 0);
    cfg_opt_bool(manifest, "uses_cfg", &cfg->uses_cfg, 0);

    /* Not declared by the pack today. `min_audio_frames` is 0 because upstream
     * has no minimum; `audio_emit_frames` is 1 because E2-3 measured a single
     * carried 80 ms frame as exact, so there is nothing to wait for. */
    cfg_opt_size(manifest, "min_generated_frames", &cfg->min_audio_frames, 0u);
    cfg_opt_size(manifest, "audio_emit_frames", &cfg->audio_emit_frames, 1u);
    cfg_opt_size(manifest, "max_decoder_steps", &cfg->default_max_steps, 0u);
    /*
     * Generation does not stop at the EOS crossing: upstream records the step
     * and keeps going for `frames_after_eos` more frames, which is what closes
     * the last word instead of clipping it
     * (`models/tts_model.py::_autoregressive_generation`). The value is
     * `model_recommended_frames_after_eos` when the language config declares
     * one (only `french_24l` does, with 8); otherwise it is
     * `prepare_text_prompt`'s guess plus 2, and that guess is 3 for inputs of
     * four words or fewer and 1 otherwise.
     *
     * The word count is a property of the *text*, which the engine never sees
     * (the request carries token ids), so the default here is the long-input
     * value and a caller holding the text can override it per context with
     * `mynah_engine_pocket_set_frames_after_eos`. Verified against the oracle:
     * the crossing is at step 49 and the dump is 52 frames, i.e. exactly 3.
     */
    cfg_opt_size(manifest, "model_recommended_frames_after_eos",
                 &cfg->frames_after_eos, SIZE_MAX);
    if (cfg->frames_after_eos == SIZE_MAX) {
        cfg_opt_size(manifest, "frames_after_eos", &cfg->frames_after_eos, 3u);
    }

    /* SEANet parameters the manifest does not declare are derived from the
     * tensor shapes later; a manifest value, when present, wins. */
    cfg_opt_size(manifest, "codec_n_filters", &cfg->n_filters, 0u);
    cfg_opt_size(manifest, "codec_n_residual_layers", &cfg->n_residual_layers, SIZE_MAX);
    cfg_opt_size(manifest, "codec_compress", &cfg->compress, 0u);
    cfg_opt_size(manifest, "codec_kernel_size", &cfg->kernel_size, 0u);
    cfg_opt_size(manifest, "codec_residual_kernel_size", &cfg->residual_kernel_size, 0u);
    cfg_opt_size(manifest, "codec_last_kernel_size", &cfg->last_kernel_size, 0u);
    /* Unobservable while n_residual_layers == 1 (every dilation is 1), so it
     * is a documented default rather than a derivation. */
    cfg_opt_size(manifest, "codec_dilation_base", &cfg->dilation_base, 2u);
    cfg_opt_double(manifest, "codec_elu_alpha", &number, 1.0);
    cfg->elu_alpha = (float)number;

    cfg_opt_string(manifest, "tokenizer_file", cfg->tokenizer_file,
                   sizeof(cfg->tokenizer_file), "tokenizer.model");
    cfg_opt_string(manifest, "speakers_file", cfg->speakers_file,
                   sizeof(cfg->speakers_file), "speakers.json");
    cfg_opt_string(manifest, "voices_dir", cfg->voices_dir, sizeof(cfg->voices_dir),
                   "voices");
    {
        const char *weights = pj_object_get(manifest, "weights");
        if (pj_string_copy(pj_object_get(weights, "tts"), cfg->weights_tts,
                           sizeof(cfg->weights_tts)) != 0) {
            snprintf(cfg->weights_tts, sizeof(cfg->weights_tts), "tts.safetensors");
        }
    }

    const char *ratios = pj_array_first(pj_object_get(manifest, "codec_ratios"));
    if (ratios == NULL) {
        pocket_error(error, capacity, "model.json: \"codec_ratios\" is missing or empty");
        return -1;
    }
    for (; ratios != NULL; ratios = pj_array_next(ratios)) {
        if (cfg->n_ratios >= POCKET_MAX_RATIOS) {
            pocket_error(error, capacity, "model.json: more than %u codec ratios",
                         POCKET_MAX_RATIOS);
            return -1;
        }
        size_t ratio = 0;
        if (cfg_size_value(ratios, &ratio) != 0 || ratio == 0) {
            pocket_error(error, capacity, "model.json: codec_ratios[%zu] is invalid",
                         cfg->n_ratios);
            return -1;
        }
        cfg->ratios[cfg->n_ratios++] = ratio;
    }

    if (cfg->heads == 0 || cfg->head_dim == 0 ||
        cfg->heads * cfg->head_dim != cfg->hidden_dim) {
        pocket_error(error, capacity,
                     "model.json: attention_heads %zu * head_dim %zu != hidden_dim %zu",
                     cfg->heads, cfg->head_dim, cfg->hidden_dim);
        return -1;
    }
    if (cfg->codec_tf_heads == 0 ||
        cfg->codec_tf_dim % cfg->codec_tf_heads != 0) {
        pocket_error(error, capacity,
                     "model.json: codec_transformer_dim %zu is not a multiple of "
                     "codec_transformer_heads %zu",
                     cfg->codec_tf_dim, cfg->codec_tf_heads);
        return -1;
    }
    if (cfg->codec_tf_dim != cfg->codec_dim) {
        pocket_error(error, capacity,
                     "model.json: codec_transformer_dim %zu != codec_dim %zu; this "
                     "engine has no projection between them",
                     cfg->codec_tf_dim, cfg->codec_dim);
        return -1;
    }
    if (cfg->flow_time_conds != 2u) {
        /* The two conditions are pinned at s = 0 and t = 1 below. One time
         * condition means flow matching with an ODE integrator, which is a
         * different head and a different loop, not a smaller array. */
        pocket_error(error, capacity,
                     "model.json: flow_time_conditions is %zu; only the LSD pair "
                     "(s = 0, t = 1) is implemented",
                     cfg->flow_time_conds);
        return -1;
    }
    if (cfg->flow_decode_steps != 1u) {
        pocket_error(error, capacity,
                     "model.json: flow_decode_steps is %zu; only the distilled "
                     "one-step head is implemented",
                     cfg->flow_decode_steps);
        return -1;
    }
    if (cfg->uses_cfg) {
        pocket_error(error, capacity,
                     "model.json: uses_cfg is set; CFG is distilled into these "
                     "weights and is not implemented");
        return -1;
    }
    if (cfg->vocab_size == 0 || cfg->embed_rows <= cfg->vocab_size) {
        pocket_error(error, capacity,
                     "model.json: text_embedding_rows %zu must exceed text_vocab_size %zu",
                     cfg->embed_rows, cfg->vocab_size);
        return -1;
    }
    {
        size_t hop = 1u;
        for (size_t i = 0; i < cfg->n_ratios; ++i) {
            if (pocket_mul(hop, cfg->ratios[i], &hop) != 0) {
                pocket_error(error, capacity, "model.json: codec_ratios overflow");
                return -1;
            }
        }
        size_t per_frame = 0;
        if (pocket_mul(hop, cfg->upsample_stride, &per_frame) != 0 ||
            per_frame != cfg->samples_per_frame) {
            pocket_error(error, capacity,
                         "model.json: prod(codec_ratios) * codec_upsample_stride = %zu "
                         "!= samples_per_frame %zu",
                         per_frame, cfg->samples_per_frame);
            return -1;
        }
    }
    return 0;
}

/* ------------------------------------------------------------- speakers */

static void pocket_voices_free(pocket_voice *voices, size_t count) {
    if (voices == NULL) return;
    for (size_t i = 0; i < count; ++i) {
        free(voices[i].name);
        free(voices[i].file);
    }
    free(voices);
}

static int pocket_voices_load(mynah_engine_state *state, const char *path,
                              char *error, size_t capacity) {
    char *text = pocket_read_text(path, POCKET_MANIFEST_MAX, error, capacity);
    if (text == NULL) return -1;

    const char *first = pj_array_first(pj_object_get(text, "voices"));
    size_t count = 0;
    for (const char *e = first; e != NULL; e = pj_array_next(e)) ++count;
    if (count == 0) {
        free(text);
        pocket_error(error, capacity, "%s lists no voices", path);
        return -1;
    }

    pocket_voice *voices = (pocket_voice *)calloc(count, sizeof(*voices));
    if (voices == NULL) {
        free(text);
        pocket_error(error, capacity, "out of memory reading %s", path);
        return -1;
    }

    size_t index = 0;
    for (const char *e = first; e != NULL && index < count; e = pj_array_next(e)) {
        char name[128];
        char file[256];
        if (pj_string_copy(pj_object_get(e, "name"), name, sizeof(name)) != 0 ||
            pj_string_copy(pj_object_get(e, "file"), file, sizeof(file)) != 0) {
            pocket_voices_free(voices, count);
            free(text);
            pocket_error(error, capacity, "%s: voice %zu has no name or file", path,
                         index);
            return -1;
        }
        voices[index].name = pocket_strdup(name, strlen(name));
        voices[index].file = pocket_strdup(file, strlen(file));
        if (voices[index].name == NULL || voices[index].file == NULL) {
            pocket_voices_free(voices, count);
            free(text);
            pocket_error(error, capacity, "out of memory reading %s", path);
            return -1;
        }
        ++index;
    }
    free(text);

    state->voices = voices;
    state->voice_count = count;
    if (state->cfg.speaker_count != 0 && state->cfg.speaker_count != count) {
        pocket_error(error, capacity,
                     "model.json says %zu speakers, %s lists %zu",
                     state->cfg.speaker_count, path, count);
        return -1;
    }
    return 0;
}

/* ------------------------------------------------------ weight resolution */

static int pocket_resolve_backbone(mynah_engine_state *state, char *error,
                                   size_t capacity) {
    const pocket_config *cfg = &state->cfg;
    const size_t attn_dim = cfg->heads * cfg->head_dim;
    char name[POCKET_NAME_MAX];

    state->backbone_layers = (mynah_transformer_ar_layer *)calloc(
        cfg->layers, sizeof(*state->backbone_layers));
    if (state->backbone_layers == NULL) {
        pocket_error(error, capacity, "out of memory resolving the backbone");
        return -1;
    }

    for (size_t l = 0; l < cfg->layers; ++l) {
        mynah_transformer_ar_layer *layer = &state->backbone_layers[l];
        const size_t qkv[2] = {3u * attn_dim, cfg->hidden_dim};
        const size_t out[2] = {cfg->hidden_dim, attn_dim};
        const size_t one[1] = {cfg->hidden_dim};
        const size_t up[2] = {cfg->ffn_dim, cfg->hidden_dim};
        const size_t down[2] = {cfg->hidden_dim, cfg->ffn_dim};

        snprintf(name, sizeof(name), "flow_lm.transformer.layers.%zu.self_attn.in_proj.weight", l);
        if (pocket_tensor(state->weights, name, 2, qkv, &layer->in_proj_weight, error, capacity) != 0) return -1;
        snprintf(name, sizeof(name), "flow_lm.transformer.layers.%zu.self_attn.in_proj.bias", l);
        if (pocket_tensor_optional(state->weights, name, 1, qkv, &layer->in_proj_bias, error, capacity) != 0) return -1;
        snprintf(name, sizeof(name), "flow_lm.transformer.layers.%zu.self_attn.out_proj.weight", l);
        if (pocket_tensor(state->weights, name, 2, out, &layer->out_proj_weight, error, capacity) != 0) return -1;
        snprintf(name, sizeof(name), "flow_lm.transformer.layers.%zu.self_attn.out_proj.bias", l);
        if (pocket_tensor_optional(state->weights, name, 1, one, &layer->out_proj_bias, error, capacity) != 0) return -1;
        snprintf(name, sizeof(name), "flow_lm.transformer.layers.%zu.norm1.weight", l);
        if (pocket_tensor(state->weights, name, 1, one, &layer->norm1_weight, error, capacity) != 0) return -1;
        snprintf(name, sizeof(name), "flow_lm.transformer.layers.%zu.norm1.bias", l);
        if (pocket_tensor(state->weights, name, 1, one, &layer->norm1_bias, error, capacity) != 0) return -1;
        snprintf(name, sizeof(name), "flow_lm.transformer.layers.%zu.norm2.weight", l);
        if (pocket_tensor(state->weights, name, 1, one, &layer->norm2_weight, error, capacity) != 0) return -1;
        snprintf(name, sizeof(name), "flow_lm.transformer.layers.%zu.norm2.bias", l);
        if (pocket_tensor(state->weights, name, 1, one, &layer->norm2_bias, error, capacity) != 0) return -1;
        snprintf(name, sizeof(name), "flow_lm.transformer.layers.%zu.linear1.weight", l);
        if (pocket_tensor(state->weights, name, 2, up, &layer->linear1_weight, error, capacity) != 0) return -1;
        snprintf(name, sizeof(name), "flow_lm.transformer.layers.%zu.linear2.weight", l);
        if (pocket_tensor(state->weights, name, 2, down, &layer->linear2_weight, error, capacity) != 0) return -1;
        /* PocketTTS' backbone builds nn.Identity() for both layer scales: a
         * NULL pointer is that, bit for bit. */
        layer->layer_scale_1 = NULL;
        layer->layer_scale_2 = NULL;
    }

    const size_t one[1] = {cfg->hidden_dim};
    state->backbone.layers = state->backbone_layers;
    if (pocket_tensor(state->weights, "flow_lm.out_norm.weight", 1, one,
                      &state->backbone.out_norm_weight, error, capacity) != 0 ||
        pocket_tensor(state->weights, "flow_lm.out_norm.bias", 1, one,
                      &state->backbone.out_norm_bias, error, capacity) != 0) {
        return -1;
    }
    return 0;
}

static int pocket_resolve_flow(mynah_engine_state *state, char *error,
                               size_t capacity) {
    const pocket_config *cfg = &state->cfg;
    const size_t freq_embed = 2u * cfg->flow_freqs;
    char name[POCKET_NAME_MAX];

    state->time_embed = (mynah_flow_time_embed_weights *)calloc(
        cfg->flow_time_conds, sizeof(*state->time_embed));
    state->res_blocks = (mynah_flow_res_block_weights *)calloc(
        cfg->flow_depth, sizeof(*state->res_blocks));
    if (state->time_embed == NULL || state->res_blocks == NULL) {
        pocket_error(error, capacity, "out of memory resolving the flow head");
        return -1;
    }

    const size_t cond_w[2] = {cfg->flow_dim, cfg->hidden_dim};
    const size_t cond_b[1] = {cfg->flow_dim};
    const size_t in_w[2] = {cfg->flow_dim, cfg->latent_dim};
    if (pocket_tensor(state->weights, "flow_lm.flow_net.cond_embed.weight", 2, cond_w,
                      &state->flow.cond_embed.weight, error, capacity) != 0 ||
        pocket_tensor(state->weights, "flow_lm.flow_net.cond_embed.bias", 1, cond_b,
                      &state->flow.cond_embed.bias, error, capacity) != 0 ||
        pocket_tensor(state->weights, "flow_lm.flow_net.input_proj.weight", 2, in_w,
                      &state->flow.input_proj.weight, error, capacity) != 0 ||
        pocket_tensor(state->weights, "flow_lm.flow_net.input_proj.bias", 1, cond_b,
                      &state->flow.input_proj.bias, error, capacity) != 0) {
        return -1;
    }

    for (size_t t = 0; t < cfg->flow_time_conds; ++t) {
        mynah_flow_time_embed_weights *embed = &state->time_embed[t];
        const size_t mlp0_w[2] = {cfg->flow_dim, freq_embed};
        const size_t mlp2_w[2] = {cfg->flow_dim, cfg->flow_dim};
        snprintf(name, sizeof(name), "flow_lm.flow_net.time_embed.%zu.mlp.0.weight", t);
        if (pocket_tensor(state->weights, name, 2, mlp0_w, &embed->mlp_in.weight, error, capacity) != 0) return -1;
        snprintf(name, sizeof(name), "flow_lm.flow_net.time_embed.%zu.mlp.0.bias", t);
        if (pocket_tensor(state->weights, name, 1, cond_b, &embed->mlp_in.bias, error, capacity) != 0) return -1;
        snprintf(name, sizeof(name), "flow_lm.flow_net.time_embed.%zu.mlp.2.weight", t);
        if (pocket_tensor(state->weights, name, 2, mlp2_w, &embed->mlp_out.weight, error, capacity) != 0) return -1;
        snprintf(name, sizeof(name), "flow_lm.flow_net.time_embed.%zu.mlp.2.bias", t);
        if (pocket_tensor(state->weights, name, 1, cond_b, &embed->mlp_out.bias, error, capacity) != 0) return -1;
        /* The tail normalisation gain. `freqs` is deliberately NOT read: the
         * head computes it and rounds to BF16, which reproduces the stored
         * tensor bit for bit (see flow_head.h). */
        snprintf(name, sizeof(name), "flow_lm.flow_net.time_embed.%zu.mlp.3.alpha", t);
        if (pocket_tensor(state->weights, name, 1, cond_b, &embed->alpha, error, capacity) != 0) return -1;
    }
    state->flow.time_embed = state->time_embed;

    for (size_t b = 0; b < cfg->flow_depth; ++b) {
        mynah_flow_res_block_weights *block = &state->res_blocks[b];
        const size_t adaln_w[2] = {3u * cfg->flow_dim, cfg->flow_dim};
        const size_t adaln_b[1] = {3u * cfg->flow_dim};
        const size_t mlp_w[2] = {cfg->flow_dim, cfg->flow_dim};
        snprintf(name, sizeof(name), "flow_lm.flow_net.res_blocks.%zu.in_ln.weight", b);
        if (pocket_tensor(state->weights, name, 1, cond_b, &block->in_ln_weight, error, capacity) != 0) return -1;
        snprintf(name, sizeof(name), "flow_lm.flow_net.res_blocks.%zu.in_ln.bias", b);
        if (pocket_tensor(state->weights, name, 1, cond_b, &block->in_ln_bias, error, capacity) != 0) return -1;
        snprintf(name, sizeof(name), "flow_lm.flow_net.res_blocks.%zu.adaLN_modulation.1.weight", b);
        if (pocket_tensor(state->weights, name, 2, adaln_w, &block->adaln.weight, error, capacity) != 0) return -1;
        snprintf(name, sizeof(name), "flow_lm.flow_net.res_blocks.%zu.adaLN_modulation.1.bias", b);
        if (pocket_tensor(state->weights, name, 1, adaln_b, &block->adaln.bias, error, capacity) != 0) return -1;
        snprintf(name, sizeof(name), "flow_lm.flow_net.res_blocks.%zu.mlp.0.weight", b);
        if (pocket_tensor(state->weights, name, 2, mlp_w, &block->mlp_in.weight, error, capacity) != 0) return -1;
        snprintf(name, sizeof(name), "flow_lm.flow_net.res_blocks.%zu.mlp.0.bias", b);
        if (pocket_tensor(state->weights, name, 1, cond_b, &block->mlp_in.bias, error, capacity) != 0) return -1;
        snprintf(name, sizeof(name), "flow_lm.flow_net.res_blocks.%zu.mlp.2.weight", b);
        if (pocket_tensor(state->weights, name, 2, mlp_w, &block->mlp_out.weight, error, capacity) != 0) return -1;
        snprintf(name, sizeof(name), "flow_lm.flow_net.res_blocks.%zu.mlp.2.bias", b);
        if (pocket_tensor(state->weights, name, 1, cond_b, &block->mlp_out.bias, error, capacity) != 0) return -1;
    }
    state->flow.res_blocks = state->res_blocks;

    const size_t final_w[2] = {2u * cfg->flow_dim, cfg->flow_dim};
    const size_t final_b[1] = {2u * cfg->flow_dim};
    const size_t linear_w[2] = {cfg->latent_dim, cfg->flow_dim};
    const size_t linear_b[1] = {cfg->latent_dim};
    if (pocket_tensor(state->weights, "flow_lm.flow_net.final_layer.adaLN_modulation.1.weight",
                      2, final_w, &state->flow.final_adaln.weight, error, capacity) != 0 ||
        pocket_tensor(state->weights, "flow_lm.flow_net.final_layer.adaLN_modulation.1.bias",
                      1, final_b, &state->flow.final_adaln.bias, error, capacity) != 0 ||
        pocket_tensor(state->weights, "flow_lm.flow_net.final_layer.linear.weight", 2,
                      linear_w, &state->flow.final_linear.weight, error, capacity) != 0 ||
        pocket_tensor(state->weights, "flow_lm.flow_net.final_layer.linear.bias", 1,
                      linear_b, &state->flow.final_linear.bias, error, capacity) != 0) {
        return -1;
    }
    return 0;
}

static int pocket_resolve_codec_transformer(mynah_engine_state *state, char *error,
                                            size_t capacity) {
    const pocket_config *cfg = &state->cfg;
    const size_t dim = cfg->codec_tf_dim;
    char name[POCKET_NAME_MAX];

    state->codec_layers = (mynah_transformer_ar_layer *)calloc(
        cfg->codec_tf_layers, sizeof(*state->codec_layers));
    if (state->codec_layers == NULL) {
        pocket_error(error, capacity, "out of memory resolving the codec transformer");
        return -1;
    }

    for (size_t l = 0; l < cfg->codec_tf_layers; ++l) {
        mynah_transformer_ar_layer *layer = &state->codec_layers[l];
        const size_t qkv[2] = {3u * dim, dim};
        const size_t square[2] = {dim, dim};
        const size_t one[1] = {dim};
        const size_t up[2] = {cfg->codec_tf_ffn, dim};
        const size_t down[2] = {dim, cfg->codec_tf_ffn};

        snprintf(name, sizeof(name), "mimi.decoder_transformer.transformer.layers.%zu.self_attn.in_proj.weight", l);
        if (pocket_tensor(state->weights, name, 2, qkv, &layer->in_proj_weight, error, capacity) != 0) return -1;
        snprintf(name, sizeof(name), "mimi.decoder_transformer.transformer.layers.%zu.self_attn.out_proj.weight", l);
        if (pocket_tensor(state->weights, name, 2, square, &layer->out_proj_weight, error, capacity) != 0) return -1;
        snprintf(name, sizeof(name), "mimi.decoder_transformer.transformer.layers.%zu.norm1.weight", l);
        if (pocket_tensor(state->weights, name, 1, one, &layer->norm1_weight, error, capacity) != 0) return -1;
        snprintf(name, sizeof(name), "mimi.decoder_transformer.transformer.layers.%zu.norm1.bias", l);
        if (pocket_tensor(state->weights, name, 1, one, &layer->norm1_bias, error, capacity) != 0) return -1;
        snprintf(name, sizeof(name), "mimi.decoder_transformer.transformer.layers.%zu.norm2.weight", l);
        if (pocket_tensor(state->weights, name, 1, one, &layer->norm2_weight, error, capacity) != 0) return -1;
        snprintf(name, sizeof(name), "mimi.decoder_transformer.transformer.layers.%zu.norm2.bias", l);
        if (pocket_tensor(state->weights, name, 1, one, &layer->norm2_bias, error, capacity) != 0) return -1;
        snprintf(name, sizeof(name), "mimi.decoder_transformer.transformer.layers.%zu.linear1.weight", l);
        if (pocket_tensor(state->weights, name, 2, up, &layer->linear1_weight, error, capacity) != 0) return -1;
        snprintf(name, sizeof(name), "mimi.decoder_transformer.transformer.layers.%zu.linear2.weight", l);
        if (pocket_tensor(state->weights, name, 2, down, &layer->linear2_weight, error, capacity) != 0) return -1;
        /* Mimi builds LayerScale(0.01) here, unlike the backbone. */
        snprintf(name, sizeof(name), "mimi.decoder_transformer.transformer.layers.%zu.layer_scale_1.scale", l);
        if (pocket_tensor(state->weights, name, 1, one, &layer->layer_scale_1, error, capacity) != 0) return -1;
        snprintf(name, sizeof(name), "mimi.decoder_transformer.transformer.layers.%zu.layer_scale_2.scale", l);
        if (pocket_tensor(state->weights, name, 1, one, &layer->layer_scale_2, error, capacity) != 0) return -1;
    }
    state->codec_transformer.layers = state->codec_layers;
    state->codec_transformer.out_norm_weight = NULL;
    state->codec_transformer.out_norm_bias = NULL;
    return 0;
}

/*
 * The SEANet decoder's module indices, and the four shape facts the manifest
 * does not carry (filters, residual depth, compression, kernel sizes), come
 * from the checkpoint itself. Upstream's module list per stage is
 * [ELU, ConvTranspose1d, ResidualBlock x n_residual_layers], with one entry
 * conv before and [ELU, conv] after, which fixes every index below.
 */
static int pocket_resolve_decoder(mynah_engine_state *state, char *error,
                                  size_t capacity) {
    pocket_config *cfg = &state->cfg;
    char name[POCKET_NAME_MAX];
    mynah_tensor tensor;

    if (cfg->n_residual_layers == SIZE_MAX) {
        size_t blocks = 0;
        for (size_t i = 0; i < 256u; ++i) {
            snprintf(name, sizeof(name), "mimi.decoder.model.%zu.block.1.conv.weight", i);
            if (mynah_weights_get(state->weights, name, &tensor) == 0) ++blocks;
        }
        if (blocks % cfg->n_ratios != 0) {
            pocket_error(error, capacity,
                         "mimi.decoder: %zu residual blocks do not divide into %zu stages",
                         blocks, cfg->n_ratios);
            return -1;
        }
        cfg->n_residual_layers = blocks / cfg->n_ratios;
    }
    const size_t per_stage = 2u + cfg->n_residual_layers;

    /* The entry convolution fixes dimension, kernel size and the filter count. */
    if (pocket_shape_of(state->weights, "mimi.decoder.model.0.conv.weight", &tensor,
                        error, capacity) != 0) {
        return -1;
    }
    if (tensor.rank != 3 || tensor.shape[1] != cfg->codec_dim) {
        pocket_error(error, capacity,
                     "mimi.decoder.model.0.conv.weight has shape rank %zu, in %zu; "
                     "expected [*, codec_dim %zu, k]",
                     tensor.rank, tensor.rank > 1 ? tensor.shape[1] : 0u, cfg->codec_dim);
        return -1;
    }
    if (cfg->kernel_size == 0) cfg->kernel_size = tensor.shape[2];
    {
        size_t filters = tensor.shape[0];
        for (size_t i = 0; i < cfg->n_ratios; ++i) {
            if (filters % 2u != 0) {
                pocket_error(error, capacity,
                             "mimi.decoder: %zu entry channels do not halve %zu times",
                             tensor.shape[0], cfg->n_ratios);
                return -1;
            }
            filters /= 2u;
        }
        if (cfg->n_filters == 0) cfg->n_filters = filters;
        if (cfg->n_filters != filters) {
            pocket_error(error, capacity,
                         "model.json codec_n_filters %zu disagrees with the tensors (%zu)",
                         cfg->n_filters, filters);
            return -1;
        }
    }

    /* The last convolution fixes the audio channel count and its kernel. */
    const size_t last_index = 1u + cfg->n_ratios * per_stage + 1u;
    snprintf(name, sizeof(name), "mimi.decoder.model.%zu.conv.weight", last_index);
    if (pocket_shape_of(state->weights, name, &tensor, error, capacity) != 0) return -1;
    if (tensor.rank != 3 || tensor.shape[0] != cfg->audio_channels ||
        tensor.shape[1] != cfg->n_filters) {
        pocket_error(error, capacity,
                     "%s: expected [audio_channels %zu, n_filters %zu, k]", name,
                     cfg->audio_channels, cfg->n_filters);
        return -1;
    }
    if (cfg->last_kernel_size == 0) cfg->last_kernel_size = tensor.shape[2];

    state->decoder_convtr = (mynah_conv_weights *)calloc(cfg->n_ratios,
                                                         sizeof(*state->decoder_convtr));
    size_t block_count = 0;
    if (pocket_mul(cfg->n_ratios, cfg->n_residual_layers, &block_count) != 0) {
        pocket_error(error, capacity, "mimi.decoder: residual block count overflow");
        return -1;
    }
    state->decoder_blocks = (mynah_seanet_resblock_weights *)calloc(
        block_count == 0 ? 1u : block_count, sizeof(*state->decoder_blocks));
    if (state->decoder_convtr == NULL || state->decoder_blocks == NULL) {
        pocket_error(error, capacity, "out of memory resolving the SEANet decoder");
        return -1;
    }

    size_t channels = cfg->n_filters << cfg->n_ratios;
    for (size_t stage = 0; stage < cfg->n_ratios; ++stage) {
        const size_t base = 1u + stage * per_stage;
        const size_t out_channels = channels / 2u;
        const size_t ratio = cfg->ratios[stage];
        const size_t convtr_w[3] = {channels, out_channels, 2u * ratio};
        const size_t convtr_b[1] = {out_channels};

        snprintf(name, sizeof(name), "mimi.decoder.model.%zu.convtr.weight", base + 1u);
        if (pocket_tensor(state->weights, name, 3, convtr_w,
                          &state->decoder_convtr[stage].weight, error, capacity) != 0) {
            return -1;
        }
        snprintf(name, sizeof(name), "mimi.decoder.model.%zu.convtr.bias", base + 1u);
        if (pocket_tensor_optional(state->weights, name, 1, convtr_b,
                                   &state->decoder_convtr[stage].bias, error,
                                   capacity) != 0) {
            return -1;
        }

        for (size_t j = 0; j < cfg->n_residual_layers; ++j) {
            const size_t index = base + 2u + j;
            mynah_seanet_resblock_weights *block =
                &state->decoder_blocks[stage * cfg->n_residual_layers + j];
            snprintf(name, sizeof(name), "mimi.decoder.model.%zu.block.1.conv.weight", index);
            if (pocket_shape_of(state->weights, name, &tensor, error, capacity) != 0) return -1;
            if (tensor.rank != 3 || tensor.shape[1] != out_channels ||
                tensor.shape[0] == 0) {
                pocket_error(error, capacity, "%s: expected [hidden, %zu, k]", name,
                             out_channels);
                return -1;
            }
            if (cfg->residual_kernel_size == 0) cfg->residual_kernel_size = tensor.shape[2];
            if (cfg->compress == 0) {
                if (out_channels % tensor.shape[0] != 0) {
                    pocket_error(error, capacity,
                                 "%s: %zu channels do not divide %zu", name,
                                 tensor.shape[0], out_channels);
                    return -1;
                }
                cfg->compress = out_channels / tensor.shape[0];
            }
            const size_t hidden = out_channels / cfg->compress;
            const size_t conv1_w[3] = {hidden, out_channels, cfg->residual_kernel_size};
            const size_t conv1_b[1] = {hidden};
            const size_t conv2_w[3] = {out_channels, hidden, 1u};
            const size_t conv2_b[1] = {out_channels};
            if (pocket_tensor(state->weights, name, 3, conv1_w, &block->conv1.weight,
                              error, capacity) != 0) {
                return -1;
            }
            snprintf(name, sizeof(name), "mimi.decoder.model.%zu.block.1.conv.bias", index);
            if (pocket_tensor_optional(state->weights, name, 1, conv1_b,
                                       &block->conv1.bias, error, capacity) != 0) {
                return -1;
            }
            snprintf(name, sizeof(name), "mimi.decoder.model.%zu.block.3.conv.weight", index);
            if (pocket_tensor(state->weights, name, 3, conv2_w, &block->conv2.weight,
                              error, capacity) != 0) {
                return -1;
            }
            snprintf(name, sizeof(name), "mimi.decoder.model.%zu.block.3.conv.bias", index);
            if (pocket_tensor_optional(state->weights, name, 1, conv2_b,
                                       &block->conv2.bias, error, capacity) != 0) {
                return -1;
            }
        }
        channels = out_channels;
    }
    if (cfg->compress == 0) cfg->compress = 1u;
    if (cfg->residual_kernel_size == 0) cfg->residual_kernel_size = 1u;

    {
        const size_t first_w[3] = {cfg->n_filters << cfg->n_ratios, cfg->codec_dim,
                                   cfg->kernel_size};
        const size_t first_b[1] = {cfg->n_filters << cfg->n_ratios};
        if (pocket_tensor(state->weights, "mimi.decoder.model.0.conv.weight", 3, first_w,
                          &state->decoder.first.weight, error, capacity) != 0 ||
            pocket_tensor_optional(state->weights, "mimi.decoder.model.0.conv.bias", 1,
                                   first_b, &state->decoder.first.bias, error,
                                   capacity) != 0) {
            return -1;
        }
    }
    {
        const size_t last_w[3] = {cfg->audio_channels, cfg->n_filters,
                                  cfg->last_kernel_size};
        const size_t last_b[1] = {cfg->audio_channels};
        snprintf(name, sizeof(name), "mimi.decoder.model.%zu.conv.weight", last_index);
        if (pocket_tensor(state->weights, name, 3, last_w, &state->decoder.last.weight,
                          error, capacity) != 0) {
            return -1;
        }
        snprintf(name, sizeof(name), "mimi.decoder.model.%zu.conv.bias", last_index);
        if (pocket_tensor_optional(state->weights, name, 1, last_b,
                                   &state->decoder.last.bias, error, capacity) != 0) {
            return -1;
        }
    }
    state->decoder.convtr = state->decoder_convtr;
    state->decoder.blocks = state->decoder_blocks;

    /* Mimi's ConvTrUpsample1d: depthwise, kernel 2 * stride, no bias. */
    {
        const size_t up_w[3] = {cfg->codec_dim, 1u, 2u * cfg->upsample_stride};
        const size_t up_b[1] = {cfg->codec_dim};
        if (pocket_tensor(state->weights, "mimi.upsample.convtr.convtr.weight", 3, up_w,
                          &state->upsample.weight, error, capacity) != 0 ||
            pocket_tensor_optional(state->weights, "mimi.upsample.convtr.convtr.bias", 1,
                                   up_b, &state->upsample.bias, error, capacity) != 0) {
            return -1;
        }
    }
    return 0;
}

static int pocket_resolve_singles(mynah_engine_state *state, char *error,
                                  size_t capacity) {
    const pocket_config *cfg = &state->cfg;
    const size_t embed[2] = {cfg->embed_rows, cfg->hidden_dim};
    const size_t latent[1] = {cfg->latent_dim};
    const size_t input_w[2] = {cfg->hidden_dim, cfg->latent_dim};
    const size_t eos_w[2] = {1u, cfg->hidden_dim};
    const size_t eos_b[1] = {1u};
    const size_t quant_w[3] = {cfg->codec_dim, cfg->latent_dim, 1u};
    const size_t speaker_w[2] = {cfg->hidden_dim, cfg->speaker_proj_input_dim};
    const size_t bos_before[3] = {1u, 1u, cfg->hidden_dim};

    if (pocket_tensor(state->weights, "flow_lm.conditioner.embed.weight", 2, embed,
                      &state->embed_table, error, capacity) != 0 ||
        pocket_tensor(state->weights, "flow_lm.bos_emb", 1, latent, &state->bos_emb,
                      error, capacity) != 0 ||
        pocket_tensor(state->weights, "flow_lm.emb_mean", 1, latent, &state->emb_mean,
                      error, capacity) != 0 ||
        pocket_tensor(state->weights, "flow_lm.emb_std", 1, latent, &state->emb_std,
                      error, capacity) != 0 ||
        pocket_tensor(state->weights, "flow_lm.input_linear.weight", 2, input_w,
                      &state->input_linear, error, capacity) != 0 ||
        pocket_tensor(state->weights, "flow_lm.out_eos.weight", 2, eos_w,
                      &state->out_eos_weight, error, capacity) != 0 ||
        pocket_tensor(state->weights, "flow_lm.out_eos.bias", 1, eos_b,
                      &state->out_eos_bias, error, capacity) != 0 ||
        pocket_tensor(state->weights, "mimi.quantizer.output_proj.weight", 3, quant_w,
                      &state->quantizer_proj, error, capacity) != 0) {
        return -1;
    }
    /* Cloning-only, and the two tensors that differ between generations. They
     * are validated when present so a mismatched pack fails here rather than
     * the first time someone clones a voice. */
    if (pocket_tensor_optional(state->weights, "flow_lm.speaker_proj_weight", 2,
                               speaker_w, &state->speaker_proj, error, capacity) != 0 ||
        pocket_tensor_optional(state->weights, "flow_lm.bos_before_voice", 3, bos_before,
                               &state->bos_before_voice, error, capacity) != 0) {
        return -1;
    }
    if (cfg->insert_bos_before_voice && state->bos_before_voice == NULL) {
        pocket_error(error, capacity,
                     "model.json sets insert_bos_before_voice but "
                     "flow_lm.bos_before_voice is missing");
        return -1;
    }
    return 0;
}

/* -------------------------------------------------------------- model_init */

static void pocket_model_free(mynah_engine_state *state) {
    if (state == NULL) return;
    mynah_sp_close(state->tokenizer);
    pocket_voices_free(state->voices, state->voice_count);
    free(state->backbone_layers);
    free(state->time_embed);
    free(state->res_blocks);
    free(state->codec_layers);
    free(state->decoder_convtr);
    free(state->decoder_blocks);
    if (state->owns_weights) mynah_weights_close(state->weights);
    free(state->model_dir);
    free(state);
}

static int pocket_model_init(const mynah_tts_model *model,
                             mynah_engine_state **out, char *error,
                             size_t capacity) {
    if (out != NULL) *out = NULL;
    if (model == NULL || out == NULL || model->model_dir == NULL) {
        pocket_error(error, capacity, "pocket: no model to initialise");
        return -1;
    }

    mynah_engine_state *state = (mynah_engine_state *)calloc(1, sizeof(*state));
    if (state == NULL) {
        pocket_error(error, capacity, "out of memory creating the pocket engine");
        return -1;
    }
    state->model_dir = pocket_strdup(model->model_dir, strlen(model->model_dir));
    if (state->model_dir == NULL) {
        pocket_model_free(state);
        pocket_error(error, capacity, "out of memory creating the pocket engine");
        return -1;
    }

    char path[POCKET_PATH_MAX];
    if (pocket_join(path, sizeof(path), state->model_dir, "model.json", error,
                    capacity) != 0) {
        pocket_model_free(state);
        return -1;
    }
    char *manifest = pocket_read_text(path, POCKET_MANIFEST_MAX, error, capacity);
    if (manifest == NULL) {
        pocket_model_free(state);
        return -1;
    }
    const int configured = pocket_config_load(manifest, &state->cfg, error, capacity);
    free(manifest);
    if (configured != 0) {
        pocket_model_free(state);
        return -1;
    }
    if (state->cfg.default_max_steps == 0) {
        state->cfg.default_max_steps = model->info.max_decoder_steps;
    }

    /* The pack loader owns the tensors when it has already opened them; a
     * caller that hands over a model without them gets them opened here, and
     * this engine then owns that handle. */
    if (model->tts != NULL) {
        state->weights = model->tts;
        state->owns_weights = 0;
    } else {
        if (pocket_join(path, sizeof(path), state->model_dir, state->cfg.weights_tts,
                        error, capacity) != 0 ||
            mynah_weights_open(path, &state->weights, error, capacity) != 0) {
            pocket_model_free(state);
            return -1;
        }
        state->owns_weights = 1;
    }

    if (pocket_resolve_singles(state, error, capacity) != 0 ||
        pocket_resolve_backbone(state, error, capacity) != 0 ||
        pocket_resolve_flow(state, error, capacity) != 0 ||
        pocket_resolve_codec_transformer(state, error, capacity) != 0 ||
        pocket_resolve_decoder(state, error, capacity) != 0) {
        pocket_model_free(state);
        return -1;
    }

    if (pocket_join(path, sizeof(path), state->model_dir, state->cfg.speakers_file,
                    error, capacity) != 0 ||
        pocket_voices_load(state, path, error, capacity) != 0) {
        pocket_model_free(state);
        return -1;
    }

    if (pocket_join(path, sizeof(path), state->model_dir, state->cfg.tokenizer_file,
                    error, capacity) != 0 ||
        mynah_sp_open(path, &state->tokenizer, error, capacity) != 0) {
        pocket_model_free(state);
        return -1;
    }
    if (mynah_sp_vocab_size(state->tokenizer) != state->cfg.vocab_size) {
        pocket_error(error, capacity,
                     "%s has %zu pieces, model.json says text_vocab_size %zu", path,
                     mynah_sp_vocab_size(state->tokenizer), state->cfg.vocab_size);
        pocket_model_free(state);
        return -1;
    }

    *out = state;
    if (error != NULL && capacity > 0) error[0] = '\0';
    return 0;
}

static int pocket_caps(const mynah_tts_model *model,
                       const mynah_engine_state *state, mynah_engine_caps *out) {
    (void)model;
    if (state == NULL || out == NULL) return -1;
    const pocket_config *cfg = &state->cfg;
    memset(out, 0, sizeof(*out));
    out->sample_rate = cfg->sample_rate;
    out->frame_rate = cfg->frame_rate;
    /* One AR step is one latent, and one latent is one 80 ms frame: there is
     * no stacking and no delay pattern in a continuous-latent model. */
    out->frames_per_step = 1u;
    out->audio_emit_frames = (unsigned)cfg->audio_emit_frames;
    out->min_audio_frames = (unsigned)cfg->min_audio_frames;
    out->default_max_steps = (unsigned)cfg->default_max_steps;
    /* Batching is measured before it is claimed. */
    out->max_batch = 1u;
    out->voice_count = (unsigned)state->voice_count;
    out->needs_cfg = 0u;
    out->is_discrete_codec = 0u;
    out->latent_dim = (unsigned)cfg->latent_dim;
    return 0;
}

/* ----------------------------------------------------------------- context */

static void pocket_ctx_free(mynah_engine_ctx *ctx) {
    if (ctx == NULL) return;
    mynah_transformer_ar_state_free(ctx->backbone);
    mynah_transformer_ar_state_free(ctx->codec_transformer);
    mynah_flow_head_destroy(ctx->flow);
    mynah_seanet_state_destroy(ctx->codec);
    ingot_st_close(ctx->voice_file);
    free(ctx->voice_kv);
    free(ctx->text_ids);
    free(ctx->text_embed);
    free(ctx->step_input);
    free(ctx->hidden);
    free(ctx->noise);
    free(ctx->flow_out);
    free(ctx->latents);
    free(ctx->denorm);
    free(ctx->codec_in);
    free(ctx->codec_up);
    free(ctx->codec_seq);
    free(ctx->codec_out);
    free(ctx->codec_back);
    free(ctx->pcm);
    free(ctx);
}

/* Opens the voice and reads its length, without loading anything yet: the KV
 * itself is copied in `prepare`, so a reset costs no allocation. */
static int pocket_voice_open(mynah_engine_ctx *ctx, char *error, size_t capacity) {
    const mynah_engine_state *state = ctx->state;
    const pocket_config *cfg = &state->cfg;
    char path[POCKET_PATH_MAX];
    if (pocket_join(path, sizeof(path), state->model_dir,
                    state->voices[ctx->speaker].file, error, capacity) != 0) {
        return -1;
    }
    if (ingot_st_open(&ctx->voice_file, path, error, capacity) != 0) return -1;

    char name[POCKET_NAME_MAX];
    size_t positions = 0;
    for (size_t l = 0; l < cfg->layers; ++l) {
        snprintf(name, sizeof(name), "transformer.layers.%zu.self_attn/cache", l);
        const ingot_st_tensor *tensor = ingot_st_find(ctx->voice_file, name);
        if (tensor == NULL) {
            pocket_error(error, capacity, "voice %s has no %s",
                         state->voices[ctx->speaker].name, name);
            return -1;
        }
        /* [K/V, batch, T, heads, head_dim]: the batch axis is the only thing
         * between this file and `transformer_ar`'s own layout. */
        if (tensor->rank != 5 || tensor->shape[0] != 2u || tensor->shape[1] != 1u ||
            tensor->shape[3] != cfg->heads || tensor->shape[4] != cfg->head_dim) {
            pocket_error(error, capacity,
                         "voice %s: %s is not [2, 1, T, %zu, %zu]",
                         state->voices[ctx->speaker].name, name, cfg->heads,
                         cfg->head_dim);
            return -1;
        }
        if (l == 0) {
            positions = (size_t)tensor->shape[2];
        } else if ((size_t)tensor->shape[2] != positions) {
            pocket_error(error, capacity,
                         "voice %s: layer %zu has %llu positions, layer 0 has %zu",
                         state->voices[ctx->speaker].name, l,
                         (unsigned long long)tensor->shape[2], positions);
            return -1;
        }

        snprintf(name, sizeof(name), "transformer.layers.%zu.self_attn/offset", l);
        const ingot_st_tensor *offset = ingot_st_find(ctx->voice_file, name);
        if (offset == NULL || offset->nelem != 1u) {
            pocket_error(error, capacity, "voice %s has no scalar %s",
                         state->voices[ctx->speaker].name, name);
            return -1;
        }
        float declared = 0.0f;
        if (ingot_st_to_f32(ctx->voice_file, offset, &declared) != 0 ||
            (size_t)declared != positions) {
            /* A partially filled cache would put the NaN padding upstream's
             * `_expand_kv_cache` writes inside the prefix. */
            pocket_error(error, capacity,
                         "voice %s: %s says %g of %zu positions are valid",
                         state->voices[ctx->speaker].name, name, (double)declared,
                         positions);
            return -1;
        }
    }
    if (positions == 0) {
        pocket_error(error, capacity, "voice %s is empty",
                     state->voices[ctx->speaker].name);
        return -1;
    }
    ctx->voice_positions = positions;
    return 0;
}

static int pocket_ctx_new(const mynah_tts_model *model, mynah_engine_state *state,
                          const mynah_tts_request *request, size_t max_steps,
                          uint64_t seed, mynah_engine_ctx **out_ctx, char *error,
                          size_t capacity) {
    (void)model;
    if (out_ctx != NULL) *out_ctx = NULL;
    if (state == NULL || request == NULL || out_ctx == NULL) {
        pocket_error(error, capacity, "pocket: null argument creating a context");
        return -1;
    }
    const pocket_config *cfg = &state->cfg;

    if (request->speaker >= state->voice_count) {
        pocket_error(error, capacity, "speaker %u is out of range (%zu voices)",
                     request->speaker, state->voice_count);
        return -1;
    }
    if (request->text_ids == NULL || request->text_length == 0) {
        pocket_error(error, capacity, "pocket: the request has no text");
        return -1;
    }
    if (max_steps == 0) max_steps = cfg->default_max_steps;
    if (max_steps == 0) {
        pocket_error(error, capacity,
                     "pocket: no step budget; pass max_steps or declare "
                     "max_decoder_steps in model.json");
        return -1;
    }

    mynah_engine_ctx *ctx = (mynah_engine_ctx *)calloc(1, sizeof(*ctx));
    if (ctx == NULL) {
        pocket_error(error, capacity, "out of memory creating a pocket context");
        return -1;
    }
    ctx->state = state;
    ctx->speaker = request->speaker;
    ctx->max_steps = max_steps;
    ctx->text_length = request->text_length;
    ctx->temperature = (request->temperature >= 0.0f) ? request->temperature
                                                      : cfg->temperature;
    ctx->noise_std = sqrtf(ctx->temperature);
    ctx->seed = (seed == 0) ? UINT64_C(0x9e3779b97f4a7c15) : seed;
    ctx->frames_after_eos = (cfg->frames_after_eos < max_steps)
                                ? cfg->frames_after_eos
                                : max_steps;

    ctx->text_ids = (int *)calloc(ctx->text_length, sizeof(*ctx->text_ids));
    if (ctx->text_ids == NULL) {
        pocket_ctx_free(ctx);
        pocket_error(error, capacity, "out of memory copying the request text");
        return -1;
    }
    for (size_t i = 0; i < ctx->text_length; ++i) {
        const int id = request->text_ids[i];
        if (id < 0 || (size_t)id >= cfg->vocab_size) {
            pocket_ctx_free(ctx);
            pocket_error(error, capacity,
                         "text id %d at position %zu is outside [0, %zu); row %zu is "
                         "the padding row and is never emitted by the tokenizer",
                         id, i, cfg->vocab_size, cfg->padding_id);
            return -1;
        }
        ctx->text_ids[i] = id;
    }

    if (pocket_voice_open(ctx, error, capacity) != 0) {
        pocket_ctx_free(ctx);
        return -1;
    }

    /* Sizes. Every buffer the AR loop and the codec touch is allocated here so
     * that neither allocates again (CLAUDE.md rule 4). */
    const size_t attn_dim = cfg->heads * cfg->head_dim;
    size_t backbone_capacity = 0;
    size_t voice_floats = 0;
    size_t text_floats = 0;
    size_t latent_floats = 0;
    size_t codec_positions = 0;
    size_t up_floats = 0;
    size_t pcm_floats = 0;
    size_t codec_frames = 0;
    if (pocket_add(max_steps, 1u, &codec_frames) != 0 ||
        pocket_add(ctx->voice_positions, ctx->text_length, &backbone_capacity) != 0 ||
        pocket_add(backbone_capacity, codec_frames, &backbone_capacity) != 0 ||
        pocket_mul(ctx->voice_positions, attn_dim, &voice_floats) != 0 ||
        pocket_mul(voice_floats, 2u, &voice_floats) != 0 ||
        pocket_mul(ctx->text_length, cfg->hidden_dim, &text_floats) != 0 ||
        pocket_mul(max_steps, cfg->latent_dim, &latent_floats) != 0 ||
        pocket_mul(codec_frames, cfg->upsample_stride, &codec_positions) != 0 ||
        pocket_mul(cfg->codec_dim, cfg->upsample_stride, &up_floats) != 0 ||
        pocket_mul(cfg->samples_per_frame, cfg->audio_channels, &pcm_floats) != 0) {
        pocket_ctx_free(ctx);
        pocket_error(error, capacity, "pocket: request size overflow");
        return -1;
    }

    ctx->voice_kv = mynah_alloc_floats(voice_floats, error, capacity);
    ctx->text_embed = mynah_alloc_floats(text_floats, error, capacity);
    ctx->step_input = mynah_alloc_floats(cfg->hidden_dim, error, capacity);
    ctx->hidden = mynah_alloc_floats(cfg->hidden_dim, error, capacity);
    ctx->noise = mynah_alloc_floats(cfg->latent_dim, error, capacity);
    ctx->flow_out = mynah_alloc_floats(cfg->latent_dim, error, capacity);
    ctx->latents = mynah_alloc_floats(latent_floats, error, capacity);
    ctx->denorm = mynah_alloc_floats(cfg->latent_dim, error, capacity);
    ctx->codec_in = mynah_alloc_floats(cfg->codec_dim, error, capacity);
    ctx->codec_up = mynah_alloc_floats(up_floats, error, capacity);
    ctx->codec_seq = mynah_alloc_floats(up_floats, error, capacity);
    ctx->codec_out = mynah_alloc_floats(up_floats, error, capacity);
    ctx->codec_back = mynah_alloc_floats(up_floats, error, capacity);
    ctx->pcm = mynah_alloc_floats(pcm_floats, error, capacity);
    if (ctx->voice_kv == NULL || ctx->text_embed == NULL || ctx->step_input == NULL ||
        ctx->hidden == NULL || ctx->noise == NULL || ctx->flow_out == NULL ||
        ctx->latents == NULL || ctx->denorm == NULL || ctx->codec_in == NULL ||
        ctx->codec_up == NULL || ctx->codec_seq == NULL || ctx->codec_out == NULL ||
        ctx->codec_back == NULL || ctx->pcm == NULL) {
        pocket_ctx_free(ctx);
        return -1;
    }

    mynah_transformer_ar_config backbone;
    mynah_transformer_ar_config_defaults(&backbone);
    backbone.d_model = cfg->hidden_dim;
    backbone.num_heads = cfg->heads;
    backbone.head_dim = cfg->head_dim;
    backbone.num_layers = cfg->layers;
    backbone.ffn_dim = cfg->ffn_dim;
    backbone.max_seq_len = backbone_capacity;
    backbone.context = 0u; /* the LM attends to the whole prefix */
    backbone.layernorm_eps = cfg->layernorm_eps;
    ctx->backbone = mynah_transformer_ar_state_new(&backbone, error, capacity);

    mynah_transformer_ar_config codec;
    mynah_transformer_ar_config_defaults(&codec);
    codec.d_model = cfg->codec_tf_dim;
    codec.num_heads = cfg->codec_tf_heads;
    codec.head_dim = cfg->codec_tf_dim / cfg->codec_tf_heads;
    codec.num_layers = cfg->codec_tf_layers;
    codec.ffn_dim = cfg->codec_tf_ffn;
    codec.max_seq_len = codec_positions;
    codec.context = cfg->codec_tf_context;
    codec.layernorm_eps = cfg->layernorm_eps;
    ctx->codec_transformer = mynah_transformer_ar_state_new(&codec, error, capacity);

    mynah_flow_head_config flow;
    mynah_flow_head_config_defaults(&flow);
    flow.latent_dim = cfg->latent_dim;
    flow.cond_dim = cfg->hidden_dim;
    flow.hidden_dim = cfg->flow_dim;
    flow.depth = cfg->flow_depth;
    flow.num_time_conds = cfg->flow_time_conds;
    flow.freq_embed_dim = 2u * cfg->flow_freqs;
    flow.layernorm_eps = cfg->flow_layernorm_eps;
    ctx->flow = mynah_flow_head_create(&flow, error, capacity);

    mynah_seanet_config seanet;
    memset(&seanet, 0, sizeof(seanet));
    seanet.channels = cfg->audio_channels;
    seanet.dimension = cfg->codec_dim;
    seanet.n_filters = cfg->n_filters;
    seanet.n_residual_layers = cfg->n_residual_layers;
    seanet.ratios = cfg->ratios;
    seanet.n_ratios = cfg->n_ratios;
    seanet.kernel_size = cfg->kernel_size;
    seanet.residual_kernel_size = cfg->residual_kernel_size;
    seanet.last_kernel_size = cfg->last_kernel_size;
    seanet.dilation_base = cfg->dilation_base;
    seanet.compress = cfg->compress;
    seanet.elu_alpha = cfg->elu_alpha;

    mynah_resample_config upsample;
    upsample.stride = cfg->upsample_stride;
    upsample.in_channels = cfg->codec_dim;
    upsample.out_channels = cfg->codec_dim;
    upsample.groups = cfg->codec_dim; /* depthwise: [512, 1, 32], not [512, 512, 32] */
    /* One latent frame per call: the streaming unit and the offline unit are
     * the same call, so there is no second code path to keep in step. */
    ctx->codec = mynah_seanet_state_create(&seanet, &upsample, 1u, error, capacity);

    if (ctx->backbone == NULL || ctx->codec_transformer == NULL || ctx->flow == NULL ||
        ctx->codec == NULL) {
        pocket_ctx_free(ctx);
        return -1;
    }
    if (mynah_transformer_ar_check_weights(ctx->backbone, &state->backbone, error,
                                           capacity) != 0 ||
        mynah_transformer_ar_check_weights(ctx->codec_transformer,
                                           &state->codec_transformer, error,
                                           capacity) != 0 ||
        mynah_flow_head_check_weights(ctx->flow, &state->flow, error, capacity) != 0 ||
        mynah_seanet_check_decoder_weights(ctx->codec, &state->decoder, error,
                                           capacity) != 0) {
        pocket_ctx_free(ctx);
        return -1;
    }
    if (mynah_seanet_state_samples_per_latent(ctx->codec) != cfg->samples_per_frame) {
        pocket_error(error, capacity,
                     "codec produces %zu samples per latent, model.json says %zu",
                     mynah_seanet_state_samples_per_latent(ctx->codec),
                     cfg->samples_per_frame);
        pocket_ctx_free(ctx);
        return -1;
    }

    *out_ctx = ctx;
    if (error != NULL && capacity > 0) error[0] = '\0';
    return 0;
}

/* Voice KV, then the text prefix. Shared by `prepare` and `reset`, because a
 * rewind is exactly a re-entry into this state and nothing else. */
static int pocket_seed_context(mynah_engine_ctx *ctx, char *error, size_t capacity) {
    const mynah_engine_state *state = ctx->state;
    const pocket_config *cfg = &state->cfg;

    mynah_transformer_ar_state_reset(ctx->backbone);
    mynah_transformer_ar_state_reset(ctx->codec_transformer);
    mynah_seanet_state_reset(ctx->codec);
    mynah_flow_head_reset(ctx->flow);
    ctx->frames = 0;
    ctx->decoded_frames = 0;
    ctx->step = 0;
    ctx->eos_step = SIZE_MAX;
    ctx->eos = 0;
    ctx->broken = 0;
    ctx->eos_logit = 0.0f;
    ctx->rng = ctx->seed;
    ctx->have_spare = 0;
    ctx->spare = 0.0f;

    char name[POCKET_NAME_MAX];
    for (size_t l = 0; l < cfg->layers; ++l) {
        snprintf(name, sizeof(name), "transformer.layers.%zu.self_attn/cache", l);
        const ingot_st_tensor *tensor = ingot_st_find(ctx->voice_file, name);
        if (tensor == NULL ||
            ingot_st_to_f32(ctx->voice_file, tensor, ctx->voice_kv) != 0) {
            pocket_error(error, capacity, "cannot read %s from voice %s", name,
                         state->voices[ctx->speaker].name);
            return -1;
        }
        if (mynah_transformer_ar_state_load_kv(ctx->backbone, l, ctx->voice_kv,
                                               ctx->voice_positions, error,
                                               capacity) != 0) {
            return -1;
        }
    }
    if (mynah_transformer_ar_state_set_offset(ctx->backbone, ctx->voice_positions,
                                              error, capacity) != 0) {
        return -1;
    }

    /* The LUT conditioner: a row lookup, no BOS, no EOS, no normalisation. */
    for (size_t i = 0; i < ctx->text_length; ++i) {
        const size_t id = (size_t)ctx->text_ids[i];
        memcpy(ctx->text_embed + i * cfg->hidden_dim,
               state->embed_table + id * cfg->hidden_dim,
               cfg->hidden_dim * sizeof(float));
    }
    if (mynah_transformer_ar_prefill(ctx->backbone, &state->backbone, ctx->text_embed,
                                     ctx->text_length, NULL) != 0) {
        pocket_error(error, capacity, "pocket: the text prefill failed");
        return -1;
    }
    ctx->prepared = 1;
    return 0;
}

static int pocket_prepare(mynah_engine_ctx *ctx, char *error, size_t capacity) {
    if (ctx == NULL) {
        pocket_error(error, capacity, "pocket: null context");
        return -1;
    }
    return pocket_seed_context(ctx, error, capacity);
}

static int pocket_reset(mynah_engine_ctx *ctx, char *error, size_t capacity) {
    if (ctx == NULL || !ctx->prepared) {
        pocket_error(error, capacity, "pocket: reset before prepare");
        return -1;
    }
    return pocket_seed_context(ctx, error, capacity);
}

/* ------------------------------------------------------------------- steps */

static int pocket_step_batch(mynah_engine_ctx *const *ctxs, size_t count,
                             mynah_engine_scratch *scratch, char *error,
                             size_t capacity) {
    (void)scratch;
    if (ctxs == NULL) {
        pocket_error(error, capacity, "pocket: null batch");
        return -1;
    }
    for (size_t i = 0; i < count; ++i) {
        mynah_engine_ctx *ctx = ctxs[i];
        if (ctx == NULL || !ctx->prepared) {
            pocket_error(error, capacity, "pocket: request %zu is not prepared", i);
            return -1;
        }
        const mynah_engine_state *state = ctx->state;
        const pocket_config *cfg = &state->cfg;
        if (ctx->eos || ctx->broken) {
            pocket_error(error, capacity,
                         "pocket: request %zu has already finished; reset it first", i);
            return -1;
        }
        if (ctx->step >= ctx->max_steps) {
            pocket_error(error, capacity,
                         "pocket: request %zu is past its %zu step budget", i,
                         ctx->max_steps);
            return -1;
        }
        /* BOS is a tracked fact, not a NaN: no latent yet means bos_emb. */
        const float *previous =
            (ctx->frames > 0)
                ? ctx->latents + (ctx->frames - 1u) * cfg->latent_dim
                : state->bos_emb;
        mynah_matvec_f32(state->input_linear, previous, ctx->step_input,
                         cfg->hidden_dim, cfg->latent_dim);
        if (mynah_transformer_ar_step(ctx->backbone, &state->backbone, ctx->step_input,
                                      ctx->hidden) != 0) {
            pocket_error(error, capacity, "pocket: backbone step %zu failed for "
                                          "request %zu",
                         ctx->step, i);
            return -1;
        }
    }
    return 0;
}

static int pocket_emit_batch(mynah_engine_ctx *const *ctxs, size_t count,
                             mynah_engine_step_result *results,
                             mynah_engine_scratch *scratch, char *error,
                             size_t capacity) {
    (void)scratch;
    if (ctxs == NULL || results == NULL) {
        pocket_error(error, capacity, "pocket: null batch");
        return -1;
    }
    for (size_t i = 0; i < count; ++i) {
        mynah_engine_ctx *ctx = ctxs[i];
        memset(&results[i], 0, sizeof(results[i]));
        if (ctx == NULL || !ctx->prepared || ctx->broken) {
            pocket_error(error, capacity,
                         "pocket: request %zu is not prepared, or has failed", i);
            return -1;
        }
        const mynah_engine_state *state = ctx->state;
        const pocket_config *cfg = &state->cfg;

        ctx->eos_logit = mynah_dot_f32(state->out_eos_weight, ctx->hidden,
                                       cfg->hidden_dim) +
                         state->out_eos_bias[0];
        if (ctx->eos_step == SIZE_MAX && ctx->eos_logit > cfg->eos_threshold &&
            ctx->frames >= cfg->min_audio_frames) {
            ctx->eos_step = ctx->step;
        }
        /* The crossing itself does not end the request: upstream emits its
         * frame and `frames_after_eos` more, and breaks *before* queueing the
         * latent of the terminal step. So the terminal step is the one whose
         * single frame is invalid, hence eos_frame 0 and nothing appended. */
        if (ctx->eos_step != SIZE_MAX &&
            ctx->step >= ctx->eos_step + ctx->frames_after_eos) {
            ctx->eos = 1;
            ++ctx->step;
            results[i].eos = 1;
            results[i].eos_frame = 0u;
            results[i].frames_appended = 0u;
            continue;
        }

        if (ctx->noise_fn != NULL) {
            if (ctx->noise_fn(ctx->noise_user, ctx->noise, cfg->latent_dim,
                              ctx->step) != 0) {
                ctx->broken = 1;
                results[i].failed = 1;
                continue;
            }
        } else {
            for (size_t d = 0; d < cfg->latent_dim; ++d) {
                ctx->noise[d] = ctx->noise_std * pocket_rng_normal(ctx);
            }
        }

        /* s and t, pinned at the endpoints; the manifest check at load time is
         * what makes this array the right length. */
        static const float times[2] = {0.0f, 1.0f};
        if (mynah_flow_head_forward(ctx->flow, &state->flow, ctx->hidden, times,
                                    ctx->noise, ctx->flow_out) != 0) {
            ctx->broken = 1;
            results[i].failed = 1;
            continue;
        }
        /* One LSD step from s = 0 to t = 1: the integration is the addition. */
        float *latent = ctx->latents + ctx->frames * cfg->latent_dim;
        for (size_t d = 0; d < cfg->latent_dim; ++d) {
            latent[d] = ctx->noise[d] + ctx->flow_out[d];
        }
        ++ctx->frames;
        ++ctx->step;
        results[i].eos = 0;
        results[i].eos_frame = 1u;
        results[i].frames_appended = 1u;
    }
    return 0;
}

static size_t pocket_frame_count(const mynah_engine_ctx *ctx) {
    return (ctx == NULL) ? 0u : ctx->frames;
}

/* Only the frame history shrinks. The codec has already consumed whatever was
 * decoded and cannot be rewound, so a truncation below `decoded_frames` is not
 * undone here; `decode_audio` refuses the next range instead of quietly
 * decoding frames that no longer exist. */
static void pocket_truncate(mynah_engine_ctx *ctx, size_t frame_count) {
    if (ctx == NULL) return;
    if (frame_count < ctx->frames) ctx->frames = frame_count;
}

/* ------------------------------------------------------------------ audio */

static int pocket_decode_audio(mynah_engine_ctx *ctx, size_t first_frame,
                               size_t frame_count, float **out_samples,
                               size_t *out_count, char *error, size_t capacity) {
    if (out_samples != NULL) *out_samples = NULL;
    if (out_count != NULL) *out_count = 0;
    if (ctx == NULL || out_samples == NULL || out_count == NULL) {
        pocket_error(error, capacity, "pocket: null argument decoding audio");
        return -1;
    }
    const mynah_engine_state *state = ctx->state;
    const pocket_config *cfg = &state->cfg;

    if (first_frame != ctx->decoded_frames) {
        /* The codec carries state instead of replaying context (E2-3), so a
         * gap or a rewind cannot be served; saying so beats emitting audio
         * that is quietly built on the wrong history. */
        pocket_error(error, capacity,
                     "pocket: frame ranges must be contiguous; asked for %zu, the "
                     "codec is at %zu",
                     first_frame, ctx->decoded_frames);
        return -1;
    }
    /* `first_frame > frames` is reachable: `truncate` can drop the history
     * below what has already been decoded, and the codec cannot be rewound.
     * Checking it separately keeps the subtraction below from wrapping. */
    if (first_frame > ctx->frames || frame_count > ctx->frames - first_frame) {
        pocket_error(error, capacity, "pocket: asked for frames [%zu, %zu) of %zu",
                     first_frame, first_frame + frame_count, ctx->frames);
        return -1;
    }
    if (frame_count == 0) return 0;

    size_t samples = 0;
    if (pocket_mul(frame_count, cfg->samples_per_frame, &samples) != 0) {
        pocket_error(error, capacity, "pocket: sample count overflow");
        return -1;
    }
    float *pcm = mynah_alloc_floats(samples, error, capacity);
    if (pcm == NULL) return -1;

    const size_t stride = cfg->upsample_stride;
    const size_t dim = cfg->codec_dim;
    for (size_t f = 0; f < frame_count; ++f) {
        const float *latent = ctx->latents + (first_frame + f) * cfg->latent_dim;
        for (size_t d = 0; d < cfg->latent_dim; ++d) {
            ctx->denorm[d] = latent[d] * state->emb_std[d] + state->emb_mean[d];
        }
        /* quantizer.output_proj is Conv1d(32, 512, 1): one matvec per frame. */
        mynah_matvec_f32(state->quantizer_proj, ctx->denorm, ctx->codec_in, dim,
                         cfg->latent_dim);

        if (mynah_seanet_upsample(ctx->codec, &state->upsample, ctx->codec_in, 1u,
                                  ctx->codec_up) != 0) {
            free(pcm);
            pocket_error(error, capacity, "pocket: the codec upsample failed");
            return -1;
        }

        /* The decoder transformer runs at the encoder frame rate and its inner
         * layers see [positions, channels]; the transpose belongs here, at the
         * same place the reference puts it. */
        for (size_t c = 0; c < dim; ++c) {
            for (size_t t = 0; t < stride; ++t) {
                ctx->codec_seq[t * dim + c] = ctx->codec_up[c * stride + t];
            }
        }
        /* Two counters, both mandatory (E2-3): the ring buffers inside the
         * SEANet state, and this position. If they ever disagree the audio
         * degrades smoothly and silently, so they are compared instead. */
        if (mynah_seanet_state_position(ctx->codec) !=
            mynah_transformer_ar_state_offset(ctx->codec_transformer)) {
            free(pcm);
            pocket_error(error, capacity,
                         "pocket: codec position %zu != decoder transformer offset %zu",
                         mynah_seanet_state_position(ctx->codec),
                         mynah_transformer_ar_state_offset(ctx->codec_transformer));
            return -1;
        }
        if (mynah_transformer_ar_prefill(ctx->codec_transformer,
                                         &state->codec_transformer, ctx->codec_seq,
                                         stride, ctx->codec_out) != 0) {
            free(pcm);
            pocket_error(error, capacity, "pocket: the decoder transformer failed");
            return -1;
        }
        for (size_t t = 0; t < stride; ++t) {
            for (size_t c = 0; c < dim; ++c) {
                ctx->codec_back[c * stride + t] = ctx->codec_out[t * dim + c];
            }
        }
        if (mynah_seanet_decode(ctx->codec, &state->decoder, ctx->codec_back, stride,
                                ctx->pcm) != 0) {
            free(pcm);
            pocket_error(error, capacity, "pocket: the SEANet decoder failed");
            return -1;
        }
        mynah_seanet_state_advance(ctx->codec, 1u);
        memcpy(pcm + f * cfg->samples_per_frame, ctx->pcm,
               cfg->samples_per_frame * sizeof(float));
    }

    ctx->decoded_frames += frame_count;
    *out_samples = pcm;
    *out_count = samples;
    return 0;
}

/* ---------------------------------------------------------------- scratch */

static int pocket_scratch_new(const mynah_tts_model *model,
                              mynah_engine_state *state, size_t batch,
                              mynah_engine_scratch **out, char *error,
                              size_t capacity) {
    (void)model;
    (void)state;
    if (out == NULL) return -1;
    *out = NULL;
    /* Every buffer an AR step touches belongs to the context it steps, so the
     * batched scratch has nothing to hold; it exists to bound the batch. */
    mynah_engine_scratch *scratch =
        (mynah_engine_scratch *)calloc(1, sizeof(*scratch));
    if (scratch == NULL) {
        pocket_error(error, capacity, "out of memory creating pocket scratch");
        return -1;
    }
    scratch->batch = batch;
    *out = scratch;
    return 0;
}

static void pocket_scratch_free(mynah_engine_scratch *scratch) { free(scratch); }

/* ----------------------------------------------------------------- vtable */

static const mynah_tts_engine pocket_engine = {
    POCKET_ENGINE_NAME,
    pocket_model_init,
    pocket_model_free,
    pocket_caps,
    pocket_ctx_new,
    pocket_prepare,
    pocket_reset,
    pocket_ctx_free,
    pocket_step_batch,
    pocket_emit_batch,
    pocket_frame_count,
    pocket_truncate,
    pocket_decode_audio,
    pocket_scratch_new,
    pocket_scratch_free,
    NULL,
};

const mynah_tts_engine *mynah_engine_pocket(void) { return &pocket_engine; }

/* ------------------------------------------------------------- accessors */

size_t mynah_engine_pocket_voice_count(const mynah_engine_state *state) {
    return (state == NULL) ? 0u : state->voice_count;
}

const char *mynah_engine_pocket_voice_name(const mynah_engine_state *state,
                                           size_t index) {
    if (state == NULL || index >= state->voice_count) return NULL;
    return state->voices[index].name;
}

int mynah_engine_pocket_voice_index(const mynah_engine_state *state,
                                    const char *name) {
    if (state == NULL || name == NULL) return -1;
    for (size_t i = 0; i < state->voice_count; ++i) {
        if (strcmp(state->voices[i].name, name) == 0) return (int)i;
    }
    return -1;
}

int mynah_engine_pocket_tokenize(const mynah_engine_state *state, const char *text,
                                 size_t text_length, int **out_ids,
                                 size_t *out_count, char *error, size_t capacity) {
    if (state == NULL || state->tokenizer == NULL || out_ids == NULL ||
        out_count == NULL) {
        pocket_error(error, capacity, "pocket: no tokenizer to encode with");
        return -1;
    }
    return mynah_sp_encode(state->tokenizer, text, text_length, out_ids, out_count,
                           error, capacity);
}

int mynah_engine_pocket_set_frames_after_eos(mynah_engine_ctx *ctx, size_t frames) {
    if (ctx == NULL || frames > ctx->max_steps) return -1;
    ctx->frames_after_eos = frames;
    return 0;
}

size_t mynah_engine_pocket_frames_after_eos(const mynah_engine_ctx *ctx) {
    return (ctx == NULL) ? 0u : ctx->frames_after_eos;
}

int mynah_engine_pocket_set_noise(mynah_engine_ctx *ctx, mynah_pocket_noise_fn fn,
                                  void *user_data) {
    if (ctx == NULL) return -1;
    ctx->noise_fn = fn;
    ctx->noise_user = user_data;
    return 0;
}

const float *mynah_engine_pocket_hidden(const mynah_engine_ctx *ctx,
                                        size_t *out_count) {
    if (ctx == NULL) return NULL;
    if (out_count != NULL) *out_count = ctx->state->cfg.hidden_dim;
    return ctx->hidden;
}

float mynah_engine_pocket_eos_logit(const mynah_engine_ctx *ctx) {
    return (ctx == NULL) ? 0.0f : ctx->eos_logit;
}

const float *mynah_engine_pocket_latent(const mynah_engine_ctx *ctx, size_t index,
                                        size_t *out_count) {
    if (ctx == NULL || index >= ctx->frames) return NULL;
    if (out_count != NULL) *out_count = ctx->state->cfg.latent_dim;
    return ctx->latents + index * ctx->state->cfg.latent_dim;
}
