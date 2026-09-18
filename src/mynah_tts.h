#ifndef MYNAH_TTS_H
#define MYNAH_TTS_H

#include <stddef.h>
#include <stdint.h>

#define MYNAH_TTS_VERSION "1.4.0"

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
    /* The language these WEIGHTS are bound to, verbatim from model.json's
     * scalar "language" -- "english", "italian", ... -- or EMPTY when the pack
     * does not declare one.
     *
     * Empty is a statement, not a missing value: it says the weights are not
     * language-specific. One Magpie pack serves twelve languages from one set
     * of weights with `language` choosing only a tokenizer, so it declares
     * none and mixing languages in one batch is legal for it. A PocketTTS pack
     * declares exactly one, because the six language models are independently
     * trained and share nothing -- relative L2 of about root-two on every
     * probed tensor including the codec (.work/pocket-tts-model-facts.md 10).
     * For such a pack a batch that mixed languages would be reading the wrong
     * weights for some of its slots.
     *
     * A caller deciding whether it may batch two requests together, or which
     * process may serve a request, must branch on THIS being empty and never
     * on `engine`: the binding is a property of the checkpoint, and a future
     * engine may fall on either side of it. */
    char language[32];
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

/* Materialise everything the model can build BEFORE anyone forks.
 *
 * A pack is opened once and then forked into N workers. Anything the workers
 * build lazily is built N times, privately, because copy-on-write only shares
 * what already exists at the fork. The largest of those is the dtype
 * conversion cache -- a pack whose tensors are bf16 is materialised to f32 on
 * first use, 399 MB on the pinned PocketTTS pack -- and that cache belongs to
 * the MODEL, so building it here makes it one shared copy for the whole tree
 * instead of one private copy per worker.
 *
 * This runs the engine's model_init and frees the state again. It does NOT
 * synthesise: a fork after a synthesis, or from any thread but main, inherits
 * locked mutexes, which is the rule server/prefork.h exists to state. What it
 * leaves behind is exactly the model-owned caches.
 *
 * Optional, idempotent, and safe to skip: every caller works unchanged without
 * it, one worker at a time and slower. Returns 0, or -1 with `error` set. */
int mynah_tts_model_warm(mynah_tts_model *model, char *error,
                         size_t error_capacity);
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

/* How many requests THIS model's engine can actually step together, which is
 * what a scheduler must honour. The bound above is only the runtime's ceiling:
 * an engine may declare less, and handing it more is an error rather than a
 * slow path. Returns 1 for an unknown model. */
size_t mynah_tts_model_max_batch(const mynah_tts_model *model);
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
