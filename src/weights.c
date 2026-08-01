/* Weight loading over ingot. See weights.h for the contract.
 *
 * The whole file is the adapter between ingot's tensor description and the flat
 * `mynah_tensor` view the graph uses; the container parsing, the validation and
 * the mmap live in the library.
 *
 * SPDX-License-Identifier: MIT */
#include "weights.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* One converted tensor. Only non-F32 dtypes ever land here, so on an F32
 * checkpoint this list stays empty and every `data` pointer is the mapping. */
typedef struct {
    const ingot_st_tensor *tensor;
    float *values;
} conversion;

/* Behind a pointer so `get` can fill the cache through a const handle: the
 * handle is logically read-only, the memo is not part of that promise. */
typedef struct {
    conversion *items;
    size_t count;
    size_t capacity;
    pthread_mutex_t lock;
} conversion_cache;

struct mynah_weights {
    ingot_st *st;
    conversion_cache *cache;
};

static void set_error(char *error, size_t error_capacity, const char *message) {
    if (error != NULL && error_capacity > 0) snprintf(error, error_capacity, "%s", message);
}

int mynah_weights_open(const char *path, mynah_weights **out,
                       char *error, size_t error_capacity) {
    if (out == NULL) return -1;
    *out = NULL;
    if (path == NULL) {
        set_error(error, error_capacity, "no model path given");
        return -1;
    }

    mynah_weights *weights = calloc(1, sizeof(*weights));
    conversion_cache *cache = calloc(1, sizeof(*cache));
    if (weights == NULL || cache == NULL) {
        free(weights);
        free(cache);
        set_error(error, error_capacity, "out of memory opening model weights");
        return -1;
    }
    if (pthread_mutex_init(&cache->lock, NULL) != 0) {
        free(weights);
        free(cache);
        set_error(error, error_capacity, "cannot initialise the weights lock");
        return -1;
    }
    weights->cache = cache;

    if (ingot_st_open(&weights->st, path, error, error_capacity) != 0) {
        pthread_mutex_destroy(&cache->lock);
        free(cache);
        free(weights);
        return -1;
    }
    *out = weights;
    return 0;
}

void mynah_weights_close(mynah_weights *weights) {
    if (weights == NULL) return;
    if (weights->cache != NULL) {
        for (size_t i = 0; i < weights->cache->count; i++) free(weights->cache->items[i].values);
        free(weights->cache->items);
        pthread_mutex_destroy(&weights->cache->lock);
        free(weights->cache);
    }
    ingot_st_close(weights->st);
    free(weights);
}

/* Returns the cached conversion of `t`, converting it on first use. NULL on
 * allocation failure or on a dtype with no f32 form. */
static const float *converted_values(const mynah_weights *weights,
                                     const ingot_st_tensor *t) {
    conversion_cache *cache = weights->cache;
    pthread_mutex_lock(&cache->lock);

    for (size_t i = 0; i < cache->count; i++) {
        if (cache->items[i].tensor == t) {
            const float *hit = cache->items[i].values;
            pthread_mutex_unlock(&cache->lock);
            return hit;
        }
    }

    if (cache->count == cache->capacity) {
        const size_t grown = cache->capacity == 0 ? 8u : cache->capacity * 2u;
        if (grown > SIZE_MAX / sizeof(conversion)) {
            pthread_mutex_unlock(&cache->lock);
            return NULL;
        }
        conversion *items = realloc(cache->items, grown * sizeof(*items));
        if (items == NULL) {
            pthread_mutex_unlock(&cache->lock);
            return NULL;
        }
        cache->items = items;
        cache->capacity = grown;
    }

    if (t->nelem == 0 || (size_t)t->nelem > SIZE_MAX / sizeof(float)) {
        pthread_mutex_unlock(&cache->lock);
        return NULL;
    }
    float *values = malloc((size_t)t->nelem * sizeof(float));
    if (values == NULL || ingot_st_to_f32(weights->st, t, values) != 0) {
        free(values);
        pthread_mutex_unlock(&cache->lock);
        return NULL;
    }

    cache->items[cache->count].tensor = t;
    cache->items[cache->count].values = values;
    cache->count++;
    pthread_mutex_unlock(&cache->lock);
    return values;
}

int mynah_weights_get(const mynah_weights *weights, const char *name,
                      mynah_tensor *out) {
    if (weights == NULL || name == NULL || out == NULL) return -1;

    const ingot_st_tensor *t = ingot_st_find(weights->st, name);
    if (t == NULL) return -1;

    /* The view carries four dimensions. A deeper tensor is refused rather than
     * folded into something that would look plausible downstream. */
    if (t->rank > 4) return -1;

    memset(out, 0, sizeof(*out));
    out->rank = (size_t)t->rank;
    out->count = (size_t)t->nelem;
    for (uint32_t d = 0; d < t->rank; d++) out->shape[d] = (size_t)t->shape[d];

    if (t->dtype == INGOT_DT_F32) {
        out->data = (const float *)ingot_st_data(weights->st, t);   /* zero-copy */
    } else {
        out->data = converted_values(weights, t);
    }
    return out->data != NULL ? 0 : -1;
}
