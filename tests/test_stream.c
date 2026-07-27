#include "mynah_tts.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    float *samples;
    size_t count;
    size_t capacity;
    size_t callbacks;
    int abort_after_first;
} capture;

static int capture_audio(const float *samples, size_t count, void *opaque) {
    capture *out = (capture *)opaque;
    if (out->abort_after_first && out->callbacks > 0) return 1;
    if (count > SIZE_MAX - out->count) return 1;
    const size_t required = out->count + count;
    if (required > out->capacity) {
        size_t capacity = out->capacity == 0 ? 1024u : out->capacity;
        while (capacity < required) capacity *= 2u;
        float *grown = (float *)realloc(out->samples, capacity * sizeof(*grown));
        if (grown == NULL) return 1;
        out->samples = grown;
        out->capacity = capacity;
    }
    memcpy(out->samples + out->count, samples, count * sizeof(*samples));
    out->count = required;
    ++out->callbacks;
    return 0;
}

static int fail(const char *where, const char *error) {
    fprintf(stderr, "stream test failed at %s: %s\n", where,
            error == NULL ? "unknown error" : error);
    return 1;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s MODEL_DIR\n", argv[0]);
        return 2;
    }
    const int tokens[] = {55, 79, 90, 59, 62, 87, 93, 27, 39, 36, 34};
    char error[256] = {0};
    mynah_tts_model *model = NULL;
    if (mynah_tts_model_open(argv[1], &model, error, sizeof(error)) != 0)
        return fail("model_open", error);
    mynah_tts_request request = {
        .text_ids = tokens,
        .text_length = sizeof(tokens) / sizeof(tokens[0]),
        .speaker = 4,
        .max_steps = 20,
        .temperature = 0.0f,
        .topk = 1,
        .use_local_transformer = 1,
        .seed = 42,
    };
    float *offline = NULL;
    size_t offline_count = 0;
    if (mynah_tts_synthesize(model, &request, &offline, &offline_count,
                             error, sizeof(error)) != 0) {
        mynah_tts_model_close(model);
        return fail("offline", error);
    }
    capture out = {0};
    mynah_tts_stream *stream = NULL;
    request.text_ids = NULL;
    request.text_length = 0;
    if (mynah_tts_stream_open(model, &request, 1024u, capture_audio, &out,
                              &stream, error, sizeof(error)) != 0 ||
        mynah_tts_stream_push(stream, tokens, 5u, error, sizeof(error)) != 0 ||
        mynah_tts_stream_push(stream, tokens + 5u, 6u, error, sizeof(error)) != 0 ||
        mynah_tts_stream_flush(stream, error, sizeof(error)) != 0) {
        mynah_tts_stream_close(stream);
        mynah_tts_free_samples(offline);
        free(out.samples);
        mynah_tts_model_close(model);
        return fail("stream", error);
    }
    if (out.count != offline_count ||
        memcmp(out.samples, offline, offline_count * sizeof(*offline)) != 0 ||
        out.callbacks < 2u) {
        fprintf(stderr, "stream mismatch: offline=%zu stream=%zu callbacks=%zu\n",
                offline_count, out.count, out.callbacks);
        mynah_tts_stream_close(stream);
        mynah_tts_free_samples(offline);
        free(out.samples);
        mynah_tts_model_close(model);
        return 1;
    }
    const size_t callbacks = out.callbacks;
    if (mynah_tts_stream_flush(stream, error, sizeof(error)) != 0 ||
        out.callbacks != callbacks) {
        mynah_tts_stream_close(stream);
        mynah_tts_free_samples(offline);
        free(out.samples);
        mynah_tts_model_close(model);
        return fail("idempotent flush", error);
    }
    mynah_tts_stream_close(stream);
    free(out.samples);

    capture aborted = {.abort_after_first = 1};
    request.text_ids = tokens;
    request.text_length = sizeof(tokens) / sizeof(tokens[0]);
    stream = NULL;
    if (mynah_tts_stream_open(model, &request, 1024u, capture_audio, &aborted,
                              &stream, error, sizeof(error)) != 0 ||
        mynah_tts_stream_flush(stream, error, sizeof(error)) == 0) {
        mynah_tts_stream_close(stream);
        mynah_tts_free_samples(offline);
        free(aborted.samples);
        mynah_tts_model_close(model);
        return fail("callback abort", error);
    }
    mynah_tts_stream_close(stream);
    mynah_tts_free_samples(offline);
    free(aborted.samples);
    mynah_tts_model_close(model);
    puts("stream equivalence: PASS");
    return 0;
}
