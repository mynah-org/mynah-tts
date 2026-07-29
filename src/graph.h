#ifndef MYNAH_GRAPH_H
#define MYNAH_GRAPH_H

#include <stddef.h>
#include "mynah_tts.h"

int mynah_graph_self_test(char *error, size_t error_capacity);
int mynah_graph_bnns_self_test(char *error, size_t error_capacity);
void *mynah_graph_codec_cache_new(void);
void mynah_graph_codec_cache_free(void *cache);
void *mynah_graph_local_projection_cache_new(const mynah_tts_model *model);
void mynah_graph_local_projection_cache_free(void *cache);

/* Largest group of requests that can be stepped together. */
#define MYNAH_GRAPH_MAX_JOBS 16u

/* One request handed to the batched driver.  Either the offline sink
 * (`samples` + `sample_count`) or `callback` must be set, exactly as for
 * mynah_graph_synthesize_stream.  `result` is filled in per job so one failing
 * request does not hide the others. */
typedef struct {
    const mynah_tts_request *request;
    float **samples;
    size_t *sample_count;
    mynah_tts_audio_callback callback;
    void *user_data;
    size_t chunk_samples;
    char *error;
    size_t error_capacity;
    int result;
} mynah_graph_job;

/* Synthesize up to MYNAH_GRAPH_MAX_JOBS requests together, sharing one pass
 * over the decoder weights per step instead of one per request.  Returns 0 when
 * every job succeeded, -1 when any failed; inspect each job's `result`.
 *
 * Output is bit-identical to running the jobs one at a time: the batching only
 * reorders independent work, never a reduction. */
int mynah_graph_synthesize_jobs(const mynah_tts_model *model,
                                mynah_graph_job *jobs, size_t count);

int mynah_graph_synthesize_stream(const mynah_tts_model *model,
                                  const mynah_tts_request *request,
                                  float **samples, size_t *sample_count,
                                  mynah_tts_audio_callback callback,
                                  void *user_data, size_t chunk_samples,
                                  char *error, size_t error_capacity);

#endif
