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

int mynah_graph_synthesize_stream(const mynah_tts_model *model,
                                  const mynah_tts_request *request,
                                  float **samples, size_t *sample_count,
                                  mynah_tts_audio_callback callback,
                                  void *user_data, size_t chunk_samples,
                                  char *error, size_t error_capacity);

#endif
