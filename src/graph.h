#ifndef MYNAH_GRAPH_H
#define MYNAH_GRAPH_H

#include <stddef.h>
#include "mynah_tts.h"

#include "conv1d.h"   /* codec cache lifecycle and the BNNS self-test moved here */
void *mynah_graph_local_projection_cache_new(const mynah_tts_model *model);
void mynah_graph_local_projection_cache_free(void *cache);

/* Ceiling on the driver's fixed-size microbatch arrays.
 *
 * This is the DRIVER's limit, and it is deliberately not the only one: the
 * authority on how wide a batch a model can actually take is the engine's
 * `caps.max_batch` (mynah_engine_caps), which a continuous-latent engine sets
 * to 1 until its batching has been measured. The driver steps
 * min(requested, caps.max_batch, MYNAH_GRAPH_MAX_JOBS) contexts at once; a
 * separate active-slot ceiling is only for continuous services and does not
 * widen engine arithmetic. */
#define MYNAH_GRAPH_MAX_JOBS 64u

/* Maximum resident request slots for a continuous service. A service can keep
 * more contexts alive than it submits in one engine microbatch, which is the
 * capacity seam needed for a C100 target. Offline/public array batching stays
 * capped by MYNAH_GRAPH_MAX_JOBS (64). Individual engines may advertise a
 * narrower safe width. */
#define MYNAH_GRAPH_MAX_ACTIVE 128u

/* Outcomes reported per request. Cancellation is distinct from failure on
 * purpose: a client that hung up did not hit a synthesis bug, and reporting it
 * as one turns every disconnect into a spurious error in the logs. */
#define MYNAH_GRAPH_OK         0
#define MYNAH_GRAPH_FAILED   (-1)
#define MYNAH_GRAPH_CANCELLED (-2)

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

/* Where a long-running driver gets its work from, and where it reports it.
 *
 * This is the difference between a batch and a service. With an array of jobs
 * the driver knows its whole workload before the first step and the last slot
 * to finish decides when the call returns; with a sink it asks for more work
 * at the top of every step, so a request that arrives mid-flight joins the
 * batch already running instead of waiting for it to drain.
 *
 *   next_job   fills `job` and `tag` and returns 1 when a request was
 *              admitted, 0 when there is none. `block` is non-zero only when
 *              the driver has nothing else to do, and a sink that blocks must
 *              return 0 (rather than block forever) once the service is
 *              stopping -- that return is what ends mynah_graph_serve_continuous.
 *              Everything `job` points at -- the request, the error buffer,
 *              the callback's user data -- must stay alive until on_done.
 *   on_done    exactly one call per admitted job, with MYNAH_GRAPH_OK,
 *              MYNAH_GRAPH_FAILED or MYNAH_GRAPH_CANCELLED. The driver has
 *              released the request by then and never touches `tag` again.
 *   cancelled  optional; polled once per step per live request. Non-zero
 *              retires the slot immediately as MYNAH_GRAPH_CANCELLED.
 *   running    optional; non-zero while the service should keep admitting.
 *              Requests already in flight always run to completion.
 */
typedef struct {
    void *ud;
    int  (*next_job)(void *ud, mynah_graph_job *job, void **tag, int block);
    void (*on_done)(void *ud, void *tag, int result);
    int  (*cancelled)(void *ud, void *tag);
    int  (*running)(void *ud);
} mynah_graph_sink;

/* Serve from `sink` until it stops handing out work, stepping up to
 * `max_batch` requests together and admitting new ones between steps.
 *
 * Runs on the calling thread and owns the engine state for its whole lifetime,
 * which is the point: one thread enters the model's mutable caches, so the
 * caller needs no lock around synthesis, and the codec's per-thread filter
 * cache holds one set rather than one per HTTP worker.
 *
 * Returns 0 when every request served succeeded, -1 when any failed. */
int mynah_graph_serve_continuous(const mynah_tts_model *model, size_t max_batch,
                                 mynah_graph_sink *sink);

/* Continuous service with separate arithmetic width and resident capacity.
 * `max_batch` is the largest engine call; `active_capacity` is the number of
 * live request contexts the scheduler may retain. Passing zero for
 * active_capacity preserves the legacy one-number behavior. */
int mynah_graph_serve_continuous_capacity(const mynah_tts_model *model,
                                          size_t max_batch,
                                          size_t active_capacity,
                                          mynah_graph_sink *sink);

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
