#ifndef MYNAH_GRAPH_H
#define MYNAH_GRAPH_H

#include <stddef.h>

int mynah_graph_self_test(char *error, size_t error_capacity);
int mynah_graph_bnns_self_test(char *error, size_t error_capacity);
void *mynah_graph_codec_cache_new(void);
void mynah_graph_codec_cache_free(void *cache);

#endif
