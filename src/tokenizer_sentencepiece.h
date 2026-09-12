#ifndef MYNAH_TOKENIZER_SENTENCEPIECE_H
#define MYNAH_TOKENIZER_SENTENCEPIECE_H

/* SentencePiece Unigram encoder, C11, no dependencies.
 *
 * This reads a stock `tokenizer.model` (a serialized `ModelProto`) with a
 * hand-written protobuf reader and runs the same Viterbi segmentation the
 * upstream library runs. It is a separate component from src/tokenizer.c,
 * which is Magpie's BYT5/G2P/IPA front end and a different contract.
 *
 * Scope is deliberately narrow: only the normalizer configuration actually
 * present in the pinned models is accepted (`identity`, empty charsmap,
 * remove_extra_whitespaces off). Anything else is rejected with a message
 * naming the offending field rather than approximated — silently substituting
 * a different normalizer is what produces plausible but wrong tokens.
 *
 * The handle is read-only after open and holds no per-call state, so one
 * handle may be shared by several contexts. Encoding allocates scratch; it
 * belongs in the prefill, never in the autoregressive loop.
 *
 * SPDX-License-Identifier: MIT */

#include <stddef.h>

typedef struct mynah_sp mynah_sp;

/* Opens `path` (mmapped, read-only) or a serialized ModelProto already in
 * memory (copied, so the caller's buffer may go away). On failure *out is NULL
 * and `error` carries the reason. */
int mynah_sp_open(const char *path, mynah_sp **out,
                  char *error, size_t error_capacity);
int mynah_sp_open_memory(const void *data, size_t length, mynah_sp **out,
                         char *error, size_t error_capacity);
void mynah_sp_close(mynah_sp *sp);

/* Encodes `text_length` bytes of `text`. The length is explicit because NUL is
 * legal input: "\0abc" encodes to `_ <0x00> a b c`.
 *
 * `mynah_sp_encode` allocates the id array; it is always non-NULL on success,
 * even for an empty result, and the caller frees it. `mynah_sp_encode_into`
 * writes into the caller's buffer and fails if `capacity` is too small, still
 * reporting the required count in *out_count. No BOS/EOS is added. */
int mynah_sp_encode(const mynah_sp *sp, const char *text, size_t text_length,
                    int **out_ids, size_t *out_count,
                    char *error, size_t error_capacity);
int mynah_sp_encode_into(const mynah_sp *sp, const char *text, size_t text_length,
                         int *ids, size_t capacity, size_t *out_count,
                         char *error, size_t error_capacity);

size_t mynah_sp_vocab_size(const mynah_sp *sp);

/* 0 and a borrowed (piece, length) into the model image, or -1 out of range.
 * The piece is not NUL-terminated. */
int mynah_sp_piece(const mynah_sp *sp, int id, const char **piece, size_t *length);

/* The id of the piece whose type is UNKNOWN, which is what upstream uses; it is
 * not TrainerSpec's `unk_id` field. -1 if `sp` is NULL. */
int mynah_sp_unk_id(const mynah_sp *sp);

/* Model-free self-test: builds a synthetic ModelProto in memory and checks the
 * normalizer rules, the lattice, byte fallback and the rejection paths.
 * 0 on success, -1 with the failing check named in `error`. */
int mynah_sp_self_test(char *error, size_t error_capacity);

#endif
