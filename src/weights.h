#ifndef MYNAH_WEIGHTS_H
#define MYNAH_WEIGHTS_H

/* Model weights, read through ingot (third_party/ingot).
 *
 * This replaces the hand-rolled safetensors parser that used to live in
 * src/safetensors.c. The API is the same shape as before — open, get, close,
 * and a flat view struct the graph fills local variables with — so the call
 * sites did not have to change.
 *
 * Two things are new. Any dtype ingot can read now opens, not just F32/I32/I64,
 * which is what makes a modern bf16 checkpoint loadable; and a tensor that is
 * not F32 is converted once into a buffer this handle owns, so the callers keep
 * treating `data` as a plain `const float *` with no lifetime of its own. */

#include <stddef.h>

#include "ingot/safetensors.h"

typedef struct mynah_weights mynah_weights;

/* The graph's view of one tensor. Rank is capped at 4 because that is what the
 * graph declares; a deeper tensor is an error with a message, never a silent
 * truncation. */
typedef struct {
    const float *data;
    size_t rank;
    size_t shape[4];
    size_t count;
} mynah_tensor;

/* `path` is a .safetensors file or a directory (index.json, then the single
 * file, then every shard in sorted order). On failure *out is NULL. */
int mynah_weights_open(const char *path, mynah_weights **out,
                       char *error, size_t error_capacity);
void mynah_weights_close(mynah_weights *weights);

/* 0 and a filled view, or -1. F32 is zero-copy straight out of the mapping,
 * exactly as before; anything else is converted once and cached. */
int mynah_weights_get(const mynah_weights *weights, const char *name,
                      mynah_tensor *out);

#endif
