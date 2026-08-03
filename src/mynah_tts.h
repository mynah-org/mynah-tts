#ifndef MYNAH_TTS_H
#define MYNAH_TTS_H

#include <stddef.h>
#include <stdint.h>

#define MYNAH_TTS_VERSION "1.3.0"

typedef struct mynah_tts_model mynah_tts_model;

typedef enum {
    MYNAH_TTS_DEVICE_CPU = 0,
    MYNAH_TTS_DEVICE_METAL = 1,
    MYNAH_TTS_DEVICE_CUDA = 2,
} mynah_tts_device;

typedef struct {
    char engine[32];
    char revision[64];
    char dtype[16];
    unsigned sample_rate;
    double frame_rate;
    unsigned frame_stacking_factor;
    unsigned codebook_count;
    unsigned codebook_size;
    unsigned audio_vocab_size;
    unsigned audio_bos_id;
    unsigned audio_eos_id;
    unsigned hidden_dim;
    unsigned encoder_layers;
    unsigned decoder_layers;
    unsigned local_transformer_layers;
    unsigned speaker_count;
    unsigned text_max_length;
    unsigned text_vocab_size;
    unsigned max_decoder_steps;
    unsigned default_topk;
    unsigned min_generated_frames;
    float default_temperature;
    char device[16];
} mynah_tts_model_info;

typedef struct {
    const int *text_ids;
    size_t text_length;
    unsigned speaker;
    unsigned max_steps;
    float temperature;
    unsigned topk;
    int use_local_transformer;
    uint64_t seed;
} mynah_tts_request;

typedef int (*mynah_tts_audio_callback)(const float *samples, size_t count,
                                        void *user_data);
typedef struct mynah_tts_stream mynah_tts_stream;

int mynah_tts_model_open(const char *model_dir, mynah_tts_model **out_model,
                         char *error, size_t error_capacity);
int mynah_tts_model_open_device(const char *model_dir, mynah_tts_device device,
                                mynah_tts_model **out_model,
                                char *error, size_t error_capacity);
void mynah_tts_model_close(mynah_tts_model *model);
int mynah_tts_model_get_info(const mynah_tts_model *model,
                             mynah_tts_model_info *info);

const char *mynah_tts_device_name(mynah_tts_device device);
int mynah_tts_device_self_test(mynah_tts_device device, char *error,
                               size_t error_capacity);

int mynah_tts_synthesize(const mynah_tts_model *model,
                         const mynah_tts_request *request,
                         float **samples, size_t *sample_count,
                         char *error, size_t error_capacity);
void mynah_tts_free_samples(float *samples);

/* Synthesize several requests together.
 *
 * A decode step reads far more weight bytes than it does arithmetic, so
 * requests run one after another each pay their own trip to memory for the same
 * weights.  Stepping them together reads those weights once and serves every
 * request from cache, which raises throughput without touching single-request
 * latency.  Requests are independent: they may differ in text, speaker, seed
 * and length, and one finishing early frees its slot immediately.
 *
 * The audio a request receives is identical to what it would have received
 * alone -- batching reorders independent work, never a reduction.
 *
 * Returns 0 when every job succeeded and -1 when any failed; each job's
 * `result` and `error` say which.  `count` must not exceed
 * mynah_tts_max_batch(). */
typedef struct {
    const mynah_tts_request *request;
    float **samples;
    size_t *sample_count;
    char *error;
    size_t error_capacity;
    int result;
} mynah_tts_batch_job;

size_t mynah_tts_max_batch(void);
int mynah_tts_synthesize_batch(const mynah_tts_model *model,
                               mynah_tts_batch_job *jobs, size_t count);

/* Incremental causal-prefix sink. Push accepts token chunks; flush runs the
 * shared AR graph and emits stable PCM prefixes through the callback. */
int mynah_tts_stream_open(const mynah_tts_model *model,
                          const mynah_tts_request *request,
                          size_t chunk_samples,
                          mynah_tts_audio_callback callback,
                          void *user_data,
                          mynah_tts_stream **out_stream,
                          char *error, size_t error_capacity);
int mynah_tts_stream_push(mynah_tts_stream *stream, const int *text_ids,
                          size_t text_length, char *error, size_t error_capacity);
int mynah_tts_stream_flush(mynah_tts_stream *stream,
                           char *error, size_t error_capacity);
void mynah_tts_stream_close(mynah_tts_stream *stream);

#endif
