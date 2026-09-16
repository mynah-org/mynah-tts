#include "mynah_tts.h"
#include "mynah_tts_internal.h"
#include "graph.h"
#include "json.h"

#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

struct mynah_tts_stream {
    const mynah_tts_model *model;
    mynah_tts_request request;
    int *text_ids;
    size_t text_length;
    size_t text_capacity;
    size_t chunk_samples;
    mynah_tts_audio_callback callback;
    void *user_data;
    int flushed;
};

static void stream_error(char *error, size_t capacity, const char *message) {
    if (error != NULL && capacity > 0) snprintf(error, capacity, "%s", message);
}

static int stream_reserve(mynah_tts_stream *stream, size_t extra,
                          char *error, size_t error_capacity) {
    if (stream->text_length > SIZE_MAX - extra) {
        stream_error(error, error_capacity, "stream token length overflow");
        return -1;
    }
    const size_t required = stream->text_length + extra;
    if (required <= stream->text_capacity) return 0;
    size_t capacity = stream->text_capacity == 0 ? 64u : stream->text_capacity;
    while (capacity < required) {
        if (capacity > SIZE_MAX / 2u) {
            capacity = required;
            break;
        }
        capacity *= 2u;
    }
    if (capacity > SIZE_MAX / sizeof(*stream->text_ids)) {
        stream_error(error, error_capacity, "stream token allocation overflow");
        return -1;
    }
    int *grown = (int *)realloc(stream->text_ids, capacity * sizeof(*grown));
    if (grown == NULL) {
        stream_error(error, error_capacity, "out of memory growing stream tokens");
        return -1;
    }
    stream->text_ids = grown;
    stream->text_capacity = capacity;
    return 0;
}

static void set_error(char *error, size_t capacity, const char *message) {
    if (capacity == 0) return;
    snprintf(error, capacity, "%s", message);
}

static char *duplicate_string(const char *value) {
    const size_t length = strlen(value);
    char *copy = (char *)malloc(length + 1u);
    if (copy != NULL) {
        memcpy(copy, value, length + 1u);
    }
    return copy;
}

static int regular_file(const char *path) {
    struct stat st;
    return stat(path, &st) == 0 && S_ISREG(st.st_mode);
}

static char *read_file(const char *path, size_t *length, char *error,
                       size_t error_capacity) {
    /* Open first, then stat the descriptor.  Checking the path with stat() and
     * opening it afterwards leaves a window in which the path can be swapped
     * for something else -- a symlink to a device or a much larger file -- so
     * the checks would describe one file while the read consumed another.
     * fstat() inspects exactly the object this handle refers to. */
    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        snprintf(error, error_capacity, "cannot open manifest: %s", path);
        return NULL;
    }
    struct stat st;
    if (fstat(fileno(file), &st) != 0 || !S_ISREG(st.st_mode) || st.st_size < 0 ||
        (uintmax_t)st.st_size > 16u * 1024u * 1024u) {
        fclose(file);
        snprintf(error, error_capacity, "cannot read manifest: %s", path);
        return NULL;
    }
    const size_t size = (size_t)st.st_size;
    char *data = (char *)malloc(size + 1u);
    if (data == NULL || fread(data, 1, size, file) != size) {
        free(data);
        fclose(file);
        snprintf(error, error_capacity, "cannot read manifest: %s", path);
        return NULL;
    }
    fclose(file);
    data[size] = '\0';
    *length = size;
    return data;
}

/* ------------------------------------------------------- manifest getters
 *
 * These used to be strstr. `json_value` searched the whole file for "\"key\""
 * and took whatever followed the next colon, so a key nested inside another
 * object answered a top-level lookup -- and it answered FIRST, because the
 * nested one usually comes earlier in an alphabetically ordered manifest.
 * models/fake-magpie is exactly that shape: "codec": { "sample_rate": 22050 }
 * sits above the top-level "sample_rate", and the loader has been reading the
 * nested one all along. It happens to hold the same number, which is the only
 * reason nobody noticed.
 *
 * Each getter now takes a PATH, so nesting is expressible and a pack no longer
 * has to flatten itself to be readable: "codec.samples_per_frame" and
 * "codec_ratios[0]" are ordinary lookups. The flat packs on disk keep working
 * unchanged -- a one-segment path is just a top-level member -- and
 * `*_or_nested` reads the flat spelling first and a nested one as a fallback,
 * which is what lets a converter start nesting without stranding the packs
 * already built. */

static int manifest_string(const mynah_json_value *root, const char *path,
                           char *out, size_t capacity) {
    mynah_json_value value;
    if (mynah_json_lookup(root, path, &value) != 0) return -1;
    return mynah_json_as_string(&value, out, capacity);
}

static int manifest_unsigned(const mynah_json_value *root, const char *path,
                             unsigned *out) {
    mynah_json_value value;
    if (mynah_json_lookup(root, path, &value) != 0) return -1;
    return mynah_json_as_unsigned(&value, out);
}

static int manifest_double(const mynah_json_value *root, const char *path,
                           double *out) {
    mynah_json_value value;
    if (mynah_json_lookup(root, path, &value) != 0) return -1;
    return mynah_json_as_number(&value, out);
}

/* The flat spelling wins; the nested one is the fallback. Order matters: a pack
 * that carries both must keep meaning what it meant before this change. */
static int manifest_unsigned_or_nested(const mynah_json_value *root,
                                       const char *flat, const char *nested,
                                       unsigned *out) {
    if (manifest_unsigned(root, flat, out) == 0) return 0;
    return manifest_unsigned(root, nested, out);
}

static int required_pack_file(const char *directory, const char *name,
                              char *error, size_t error_capacity) {
    char path[4096];
    const int length = snprintf(path, sizeof(path), "%s/%s", directory, name);
    if (length <= 0 || (size_t)length >= sizeof(path) || !regular_file(path)) {
        snprintf(error, error_capacity, "model pack is missing %s", name);
        return -1;
    }
    return 0;
}

int mynah_tts_model_open_device(const char *model_dir, mynah_tts_device device,
                                mynah_tts_model **out_model,
                                char *error, size_t error_capacity) {
    if (out_model != NULL) *out_model = NULL;
    if (model_dir == NULL || out_model == NULL || error == NULL ||
        error_capacity == 0) {
        return -1;
    }
    if (device != MYNAH_TTS_DEVICE_CPU && device != MYNAH_TTS_DEVICE_METAL &&
        device != MYNAH_TTS_DEVICE_CUDA) {
        set_error(error, error_capacity, "invalid model device");
        return -1;
    }
    char manifest_path[4096];
    const int path_length = snprintf(manifest_path, sizeof(manifest_path),
                                     "%s/model.json", model_dir);
    if (path_length <= 0 || (size_t)path_length >= sizeof(manifest_path)) {
        set_error(error, error_capacity, "model path is too long");
        return -1;
    }
    size_t manifest_length = 0;
    char *manifest = read_file(manifest_path, &manifest_length, error,
                                error_capacity);
    if (manifest == NULL) return -1;
    /* Parsed ONCE, and validated whole. A manifest that is broken anywhere is
     * broken for every key, which is the only reading that does not depend on
     * which field the loader happened to ask for first -- and the refusal names
     * the byte, because "model.json is invalid" sends whoever converted the
     * pack back to read 4 kB of JSON by eye. */
    mynah_json_value root;
    mynah_json_error json_error;
    if (mynah_json_parse(manifest, manifest_length, &root, &json_error) != 0) {
        snprintf(error, error_capacity,
                 "model.json is not valid JSON at byte %zu: %s",
                 json_error.offset, json_error.message);
        free(manifest);
        return -1;
    }
    if (root.type != MYNAH_JSON_OBJECT) {
        free(manifest);
        set_error(error, error_capacity, "model.json is not a JSON object");
        return -1;
    }
    /* The engine decides what a valid pack looks like, so it is read before
     * anything else is required. Magpie keeps its codec in a second file;
     * PocketTTS has one continuous-latent decoder that lives with the rest of
     * the weights, so demanding codec.safetensors would reject a correct pack. */
    char engine_name[32];
    if (manifest_string(&root, "engine", engine_name, sizeof(engine_name)) != 0) {
        free(manifest);
        set_error(error, error_capacity, "model.json has no engine");
        return -1;
    }
    const int is_magpie = strcmp(engine_name, "magpie") == 0;
    if (required_pack_file(model_dir, "tts.safetensors", error, error_capacity) != 0) {
        free(manifest);
        return -1;
    }
    if (is_magpie &&
        required_pack_file(model_dir, "codec.safetensors", error, error_capacity) != 0) {
        free(manifest);
        return -1;
    }

    mynah_tts_model *model = (mynah_tts_model *)calloc(1, sizeof(*model));
    if (model == NULL) {
        free(manifest);
        set_error(error, error_capacity, "out of memory creating model");
        return -1;
    }
    model->model_dir = duplicate_string(model_dir);
    if (model->model_dir == NULL || manifest_string(&root, "engine", model->info.engine,
                                                sizeof(model->info.engine)) != 0 ||
        manifest_string(&root, "revision", model->info.revision,
                    sizeof(model->info.revision)) != 0 ||
        manifest_string(&root, "dtype", model->info.dtype,
                    sizeof(model->info.dtype)) != 0 ||
        manifest_unsigned_or_nested(&root, "sample_rate", "codec.sample_rate",
                                    &model->info.sample_rate) != 0 ||
        manifest_double(&root, "frame_rate", &model->info.frame_rate) != 0 ||
        manifest_unsigned(&root, "hidden_dim", &model->info.hidden_dim) != 0 ||
        manifest_unsigned(&root, "speaker_count", &model->info.speaker_count) != 0 ||
        manifest_unsigned(&root, "text_vocab_size", &model->info.text_vocab_size) != 0 ||
        manifest_double(&root, "temperature", &(double){0.0}) != 0) {
        free(model->model_dir);
        free(model);
        free(manifest);
        set_error(error, error_capacity, "model.json is missing v1 metadata");
        return -1;
    }
    /* Magpie's discrete-codec metadata. A continuous-latent engine has none of
     * it, and these fields disappear from the public header with E1-5; until
     * then they stay zero for anything that is not Magpie. */
    if (is_magpie &&
        (manifest_unsigned(&root, "frame_stacking_factor",
                       &model->info.frame_stacking_factor) != 0 ||
         manifest_unsigned(&root, "codebook_count", &model->info.codebook_count) != 0 ||
         manifest_unsigned(&root, "codebook_size", &model->info.codebook_size) != 0 ||
         manifest_unsigned(&root, "audio_vocab_size", &model->info.audio_vocab_size) != 0 ||
         manifest_unsigned(&root, "encoder_layers", &model->info.encoder_layers) != 0 ||
         manifest_unsigned(&root, "decoder_layers", &model->info.decoder_layers) != 0 ||
         manifest_unsigned(&root, "local_transformer_layers",
                       &model->info.local_transformer_layers) != 0 ||
         manifest_unsigned(&root, "text_max_length", &model->info.text_max_length) != 0 ||
         manifest_unsigned(&root, "topk", &model->info.default_topk) != 0)) {
        free(model->model_dir);
        free(model);
        free(manifest);
        set_error(error, error_capacity, "model.json is missing Magpie metadata");
        return -1;
    }
    if (is_magpie) {
        if (manifest_unsigned(&root, "max_decoder_steps", &model->info.max_decoder_steps) != 0) {
            free(model->model_dir);
            free(model);
            free(manifest);
            set_error(error, error_capacity, "model.json is missing max_decoder_steps");
            return -1;
        }
    } else {
        /* text_max_length bounds the encoder prefill; PocketTTS chunks instead. */
        if (manifest_unsigned(&root, "text_max_length", &model->info.text_max_length) != 0) {
            model->info.text_max_length = model->info.text_vocab_size;
        }
        /* A continuous-latent model has no pack-level step ceiling: upstream
         * estimates one per request from the token count. This is only the
         * runaway guard, so it is generous and derived from the frame rate
         * rather than being a magic constant. */
        if (manifest_unsigned(&root, "max_decoder_steps", &model->info.max_decoder_steps) != 0) {
            const double seconds = 120.0;
            const double frames = model->info.frame_rate > 0.0
                ? model->info.frame_rate * seconds : 1500.0;
            model->info.max_decoder_steps = (unsigned)frames;
        }
    }
    {
        double temperature = 0.0;
        if (manifest_double(&root, "temperature", &temperature) != 0 ||
            temperature < 0.0 || temperature > 100.0) {
            free(model->model_dir);
            free(model);
            free(manifest);
            set_error(error, error_capacity, "model.json has invalid inference temperature");
            return -1;
        }
        model->info.default_temperature = (float)temperature;
        /* OPTIONAL, and its absence means something. A pack that names no
         * language is saying its weights serve any of them: that is true of a
         * Magpie pack, which carries a `languages` LIST and selects only a
         * tokenizer from it. A pack that names one is saying the opposite, and
         * a caller must not batch it with, or route another language's request
         * to, these weights. `"languages"` and `"language_to_tokenizer"` are
         * different keys and are not read here -- which used to be a property
         * of a needle that carried its own closing quote, and is now a property
         * of asking an object for a member by name. Either way, matching one of
         * them would bind a multilingual pack to a single language. */
        if (manifest_string(&root, "language", model->info.language,
                            sizeof(model->info.language)) != 0) {
            model->info.language[0] = '\0';
        }
        /* The first key to be read as a PATH. models/fake-magpie carries it
         * both flat and under "inference", and the strstr reader found the
         * nested one because "inference" sorts first -- so reading the nested
         * spelling as a fallback is not a new feature here, it is the
         * behaviour the pack has always had, now said out loud. */
        if (manifest_unsigned_or_nested(&root, "min_generated_frames",
                                        "inference.min_generated_frames",
                                        &model->info.min_generated_frames) != 0) {
            model->info.min_generated_frames = 4u;
        }
        /* Magpie special audio tokens follow the codec codebook.  Prefer the
         * explicit ids from model.json; otherwise fall back to the NeMo
         * SpecialAudioToken convention (BOS first, EOS second). */
        if (is_magpie) {
            if (manifest_unsigned(&root, "audio_bos_id", &model->info.audio_bos_id) != 0) {
                model->info.audio_bos_id = model->info.codebook_size;
            }
            if (manifest_unsigned(&root, "audio_eos_id", &model->info.audio_eos_id) != 0) {
                model->info.audio_eos_id = model->info.codebook_size + 1u;
            }
        }
    }
    char tensor_error[256];
    tensor_error[0] = '\0';
    char tts_path[4096];
    char codec_path[4096];
    const int tts_path_length = snprintf(tts_path, sizeof(tts_path), "%s/tts.safetensors", model_dir);
    const int codec_path_length = snprintf(codec_path, sizeof(codec_path), "%s/codec.safetensors", model_dir);
    if (tts_path_length <= 0 || (size_t)tts_path_length >= sizeof(tts_path) ||
        codec_path_length <= 0 || (size_t)codec_path_length >= sizeof(codec_path) ||
        mynah_weights_open(tts_path, &model->tts, tensor_error, sizeof(tensor_error)) != 0 ||
        (is_magpie &&
         mynah_weights_open(codec_path, &model->codec, tensor_error, sizeof(tensor_error)) != 0)) {
        mynah_weights_close(model->tts);
        mynah_weights_close(model->codec);
        free(model->model_dir);
        free(model);
        free(manifest);
        snprintf(error, error_capacity, "cannot load model tensors: %s", tensor_error);
        return -1;
    }
    if (mynah_backend_open(device, &model->backend, error, error_capacity) != 0) {
        mynah_weights_close(model->tts);
        mynah_weights_close(model->codec);
        free(model->model_dir);
        free(model);
        free(manifest);
        return -1;
    }
    /* -1 reads MYNAH_QUANT and otherwise stays on f32.
     *
     * PocketTTS defaults to f16 instead, and the reason is a property of the
     * checkpoint rather than a tolerance we decided to accept: the weights are
     * stored bf16, which has 8 mantissa bits, and f16 has 11, so the conversion
     * is lossless for these values. Measured against the oracle, f16 hidden
     * states sit 40-80x inside the 1e-4 tolerance (max abs 2.6e-06) while the
     * engine runs about twice as fast (RTF 0.245 against 0.52 on an M1).
     *
     * int8 is NOT enabled by default for it: parity fails by 600-1400x, it
     * generates an extra frame, and log-mel correlation against f32 drops to
     * 0.921. It stays available through MYNAH_QUANT for anyone who wants the
     * speed and has listened to the result.
     *
     * An explicit MYNAH_QUANT always wins, including MYNAH_QUANT=f32. On a
     * target without half converts the cache downgrades to f32 on its own and
     * --dispatch-map reports it, so this is a preference, not an assumption. */
    /* The qtype code src/qmat.c:97 documents. Named rather than written as a
     * bare 3, following the same precedent as src/dispatch.c; qmat.h should
     * export the enum so neither of us has to mirror it. */
    enum { QMAT_QTYPE_F16 = 3 };
    /* mynah_qmat_qtype_from_env(), not getenv: `MYNAH_QUANT=` empty and a typo
     * both mean "nothing was asked for" there, and this branch has to agree
     * with that or the two of us reintroduce the third meaning between us. */
    int qtype_request = -1;
    if (!is_magpie && mynah_qmat_qtype_from_env() < 0) {
        qtype_request = QMAT_QTYPE_F16;
    }
    model->qcache = mynah_qmat_cache_new(qtype_request);
    if (model->qcache == NULL) {
        mynah_backend_close(model->backend);
        mynah_weights_close(model->tts);
        mynah_weights_close(model->codec);
        free(model->model_dir);
        free(model);
        free(manifest);
        set_error(error, error_capacity, "out of memory creating quant cache");
        return -1;
    }
    if (is_magpie) {
        model->codec_cache = mynah_graph_codec_cache_new();
        model->local_projection_cache = mynah_graph_local_projection_cache_new(model);
    }
    snprintf(model->info.device, sizeof(model->info.device), "%s",
             mynah_backend_name(model->backend));
    free(manifest);
    *out_model = model;
    error[0] = '\0';
    return 0;
}

int mynah_tts_model_open(const char *model_dir, mynah_tts_model **out_model,
                         char *error, size_t error_capacity) {
    return mynah_tts_model_open_device(model_dir, MYNAH_TTS_DEVICE_CPU, out_model,
                                       error, error_capacity);
}

void mynah_tts_model_close(mynah_tts_model *model) {
    if (model == NULL) return;
    mynah_graph_local_projection_cache_free(model->local_projection_cache);
    mynah_graph_codec_cache_free(model->codec_cache);
    mynah_qmat_cache_free(model->qcache);
    mynah_backend_close(model->backend);
    mynah_weights_close(model->tts);
    mynah_weights_close(model->codec);
    free(model->model_dir);
    free(model);
}

int mynah_tts_model_get_info(const mynah_tts_model *model,
                             mynah_tts_model_info *info) {
    if (model == NULL || info == NULL) return -1;
    *info = model->info;
    return 0;
}

int mynah_tts_device_self_test(mynah_tts_device device, char *error,
                               size_t error_capacity) {
    return mynah_backend_self_test(device, error, error_capacity);
}

int mynah_tts_stream_open(const mynah_tts_model *model,
                          const mynah_tts_request *request,
                          size_t chunk_samples,
                          mynah_tts_audio_callback callback,
                          void *user_data,
                          mynah_tts_stream **out_stream,
                          char *error, size_t error_capacity) {
    if (out_stream != NULL) *out_stream = NULL;
    if (model == NULL || request == NULL || out_stream == NULL ||
        callback == NULL || chunk_samples == 0 || error == NULL ||
        error_capacity == 0) {
        stream_error(error, error_capacity, "invalid stream arguments");
        return -1;
    }
    if (request->text_length > 0 && request->text_ids == NULL) {
        stream_error(error, error_capacity, "stream text ids are missing");
        return -1;
    }
    mynah_tts_stream *stream = (mynah_tts_stream *)calloc(1, sizeof(*stream));
    if (stream == NULL) {
        stream_error(error, error_capacity, "out of memory creating stream");
        return -1;
    }
    stream->model = model;
    stream->request = *request;
    stream->request.text_ids = NULL;
    stream->request.text_length = 0;
    stream->chunk_samples = chunk_samples;
    stream->callback = callback;
    stream->user_data = user_data;
    if (request->text_length > 0 &&
        (stream_reserve(stream, request->text_length, error, error_capacity) != 0 ||
         (request->text_length > 0 && stream->text_ids == NULL))) {
        mynah_tts_stream_close(stream);
        return -1;
    }
    if (request->text_length > 0) {
        memcpy(stream->text_ids, request->text_ids,
               request->text_length * sizeof(*stream->text_ids));
        stream->text_length = request->text_length;
    }
    *out_stream = stream;
    error[0] = '\0';
    return 0;
}

int mynah_tts_stream_push(mynah_tts_stream *stream, const int *text_ids,
                          size_t text_length, char *error, size_t error_capacity) {
    if (stream == NULL || error == NULL || error_capacity == 0 ||
        (text_length > 0 && text_ids == NULL)) {
        stream_error(error, error_capacity, "invalid stream push arguments");
        return -1;
    }
    if (stream->flushed) {
        stream_error(error, error_capacity, "stream is already flushed");
        return -1;
    }
    if (stream_reserve(stream, text_length, error, error_capacity) != 0) return -1;
    if (text_length > 0) {
        memcpy(stream->text_ids + stream->text_length, text_ids,
               text_length * sizeof(*stream->text_ids));
        stream->text_length += text_length;
    }
    error[0] = '\0';
    return 0;
}

int mynah_tts_stream_flush(mynah_tts_stream *stream,
                           char *error, size_t error_capacity) {
    if (stream == NULL || error == NULL || error_capacity == 0) {
        stream_error(error, error_capacity, "invalid stream flush arguments");
        return -1;
    }
    if (stream->flushed) {
        error[0] = '\0';
        return 0;
    }
    if (stream->text_length == 0) {
        stream_error(error, error_capacity, "cannot flush an empty stream");
        return -1;
    }
    stream->request.text_ids = stream->text_ids;
    stream->request.text_length = stream->text_length;
    if (mynah_graph_synthesize_stream(stream->model, &stream->request, NULL, NULL,
                                      stream->callback, stream->user_data,
                                      stream->chunk_samples, error, error_capacity) != 0) {
        return -1;
    }
    stream->flushed = 1;
    error[0] = '\0';
    return 0;
}

void mynah_tts_stream_close(mynah_tts_stream *stream) {
    if (stream == NULL) return;
    free(stream->text_ids);
    free(stream);
}
