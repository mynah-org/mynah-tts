#include "mynah_tts.h"
#include "mynah_tts_internal.h"
#include "graph.h"

#include <ctype.h>
#include <errno.h>
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
    struct stat st;
    if (stat(path, &st) != 0 || !S_ISREG(st.st_mode) || st.st_size < 0 ||
        (uintmax_t)st.st_size > 16u * 1024u * 1024u) {
        snprintf(error, error_capacity, "cannot read manifest: %s", path);
        return NULL;
    }
    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        snprintf(error, error_capacity, "cannot open manifest: %s", path);
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

static const char *json_value(const char *json, const char *key) {
    char needle[96];
    const int written = snprintf(needle, sizeof(needle), "\"%s\"", key);
    if (written <= 0 || (size_t)written >= sizeof(needle)) return NULL;
    const char *cursor = strstr(json, needle);
    if (cursor == NULL) return NULL;
    cursor += written;
    while (*cursor != '\0' && (*cursor == ' ' || *cursor == '\t' ||
                               *cursor == '\r' || *cursor == '\n' ||
                               *cursor == ':')) {
        ++cursor;
    }
    return *cursor == '\0' ? NULL : cursor;
}

static int json_string(const char *json, const char *key, char *out,
                       size_t capacity) {
    const char *value = json_value(json, key);
    if (value == NULL || *value != '"' || capacity == 0) return -1;
    ++value;
    size_t used = 0;
    while (*value != '\0' && *value != '"') {
        if (*value == '\\' && value[1] != '\0') ++value;
        if (used + 1u >= capacity) return -1;
        out[used++] = *value++;
    }
    if (*value != '"') return -1;
    out[used] = '\0';
    return 0;
}

static int json_unsigned(const char *json, const char *key, unsigned *out) {
    const char *value = json_value(json, key);
    if (value == NULL) return -1;
    char *end = NULL;
    errno = 0;
    const unsigned long parsed = strtoul(value, &end, 10);
    if (errno != 0 || end == value || parsed > UINT_MAX) return -1;
    *out = (unsigned)parsed;
    return 0;
}

static int json_double(const char *json, const char *key, double *out) {
    const char *value = json_value(json, key);
    if (value == NULL) return -1;
    char *end = NULL;
    errno = 0;
    const double parsed = strtod(value, &end);
    if (errno != 0 || end == value) return -1;
    *out = parsed;
    return 0;
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
    (void)manifest_length;
    if (manifest == NULL) return -1;
    if (required_pack_file(model_dir, "tts.safetensors", error, error_capacity) != 0 ||
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
    if (model->model_dir == NULL || json_string(manifest, "engine", model->info.engine,
                                                sizeof(model->info.engine)) != 0 ||
        json_string(manifest, "revision", model->info.revision,
                    sizeof(model->info.revision)) != 0 ||
        json_string(manifest, "dtype", model->info.dtype,
                    sizeof(model->info.dtype)) != 0 ||
        json_unsigned(manifest, "sample_rate", &model->info.sample_rate) != 0 ||
        json_double(manifest, "frame_rate", &model->info.frame_rate) != 0 ||
        json_unsigned(manifest, "frame_stacking_factor",
                      &model->info.frame_stacking_factor) != 0 ||
        json_unsigned(manifest, "codebook_count", &model->info.codebook_count) != 0 ||
        json_unsigned(manifest, "codebook_size", &model->info.codebook_size) != 0 ||
        json_unsigned(manifest, "audio_vocab_size", &model->info.audio_vocab_size) != 0 ||
        json_unsigned(manifest, "hidden_dim", &model->info.hidden_dim) != 0 ||
        json_unsigned(manifest, "encoder_layers", &model->info.encoder_layers) != 0 ||
        json_unsigned(manifest, "decoder_layers", &model->info.decoder_layers) != 0 ||
        json_unsigned(manifest, "local_transformer_layers",
                      &model->info.local_transformer_layers) != 0 ||
        json_unsigned(manifest, "speaker_count", &model->info.speaker_count) != 0 ||
        json_unsigned(manifest, "text_max_length", &model->info.text_max_length) != 0 ||
        json_unsigned(manifest, "text_vocab_size", &model->info.text_vocab_size) != 0 ||
        json_unsigned(manifest, "max_decoder_steps", &model->info.max_decoder_steps) != 0 ||
        json_unsigned(manifest, "topk", &model->info.default_topk) != 0 ||
        json_double(manifest, "temperature", &(double){0.0}) != 0) {
        free(model->model_dir);
        free(model);
        free(manifest);
        set_error(error, error_capacity, "model.json is missing v1 metadata");
        return -1;
    }
    {
        double temperature = 0.0;
        if (json_double(manifest, "temperature", &temperature) != 0 ||
            temperature < 0.0 || temperature > 100.0) {
            free(model->model_dir);
            free(model);
            free(manifest);
            set_error(error, error_capacity, "model.json has invalid inference temperature");
            return -1;
        }
        model->info.default_temperature = (float)temperature;
        if (json_unsigned(manifest, "min_generated_frames", &model->info.min_generated_frames) != 0) {
            model->info.min_generated_frames = 4u;
        }
        /* Magpie special audio tokens follow the codec codebook.  Prefer the
         * explicit ids from model.json; otherwise fall back to the NeMo
         * SpecialAudioToken convention (BOS first, EOS second). */
        if (json_unsigned(manifest, "audio_bos_id", &model->info.audio_bos_id) != 0) {
            model->info.audio_bos_id = model->info.codebook_size;
        }
        if (json_unsigned(manifest, "audio_eos_id", &model->info.audio_eos_id) != 0) {
            model->info.audio_eos_id = model->info.codebook_size + 1u;
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
        mynah_safetensors_open(tts_path, &model->tts, tensor_error, sizeof(tensor_error)) != 0 ||
        mynah_safetensors_open(codec_path, &model->codec, tensor_error, sizeof(tensor_error)) != 0) {
        mynah_safetensors_close(model->tts);
        mynah_safetensors_close(model->codec);
        free(model->model_dir);
        free(model);
        free(manifest);
        snprintf(error, error_capacity, "cannot load model tensors: %s", tensor_error);
        return -1;
    }
    if (mynah_backend_open(device, &model->backend, error, error_capacity) != 0) {
        mynah_safetensors_close(model->tts);
        mynah_safetensors_close(model->codec);
        free(model->model_dir);
        free(model);
        free(manifest);
        return -1;
    }
    /* -1: read MYNAH_QUANT env (int8 opt-in; default f32). */
    model->qcache = mynah_qmat_cache_new(-1);
    if (model->qcache == NULL) {
        mynah_backend_close(model->backend);
        mynah_safetensors_close(model->tts);
        mynah_safetensors_close(model->codec);
        free(model->model_dir);
        free(model);
        free(manifest);
        set_error(error, error_capacity, "out of memory creating quant cache");
        return -1;
    }
    model->codec_cache = mynah_graph_codec_cache_new();
    model->local_projection_cache = mynah_graph_local_projection_cache_new(model);
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
    mynah_safetensors_close(model->tts);
    mynah_safetensors_close(model->codec);
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
    float *samples = NULL;
    size_t sample_count = 0;
    if (mynah_tts_synthesize(stream->model, &stream->request, &samples,
                             &sample_count, error, error_capacity) != 0) {
        mynah_tts_free_samples(samples);
        return -1;
    }
    for (size_t offset = 0; offset < sample_count; ) {
        const size_t remaining = sample_count - offset;
        const size_t count = remaining < stream->chunk_samples
            ? remaining : stream->chunk_samples;
        if (stream->callback(samples + offset, count, stream->user_data) != 0) {
            mynah_tts_free_samples(samples);
            stream_error(error, error_capacity, "audio callback aborted stream");
            return -1;
        }
        offset += count;
    }
    mynah_tts_free_samples(samples);
    stream->flushed = 1;
    error[0] = '\0';
    return 0;
}

void mynah_tts_stream_close(mynah_tts_stream *stream) {
    if (stream == NULL) return;
    free(stream->text_ids);
    free(stream);
}
