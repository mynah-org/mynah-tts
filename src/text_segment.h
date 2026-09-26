/* Text segmentation for engines that generate long input one chunk at a time.
 *
 * PocketTTS is trained on single sentences. Upstream's `generate_audio_stream`
 * splits the text before the model sees it (`split_into_best_sentences`):
 * sentence boundaries first, then `, ; :` for a sentence that is still over
 * the limit, then greedy packing of consecutive pieces up to `max_tokens`
 * (50 in `default_parameters.py`). Each chunk is tokenized on its own and
 * generated as its own utterance from the voice state.
 *
 * The split has to happen here, above the engine, because the engine receives
 * token ids and the sentence structure is gone by then (see the E2-5 note in
 * src/engine_pocket.c). This module produces the ids of every chunk,
 * concatenated, plus one length per chunk; the engine consumes them through
 * `mynah_tts_request.segment_lengths`.
 *
 * `first_max_tokens` bounds the FIRST chunk separately: its prefill sits in
 * front of the first audio, while every later chunk is prefilled behind audio
 * the client already has. 0 means "same as max_tokens". */
#ifndef MYNAH_TEXT_SEGMENT_H
#define MYNAH_TEXT_SEGMENT_H

#include <stddef.h>

#include "tokenizer_sentencepiece.h"

/* On success *out_ids and *out_lengths are malloc'd and owned by the caller,
 * *out_segments >= 1 and the lengths sum to *out_count. A text that fits in one
 * chunk yields exactly one segment whose ids are those of the trimmed text. */
int mynah_text_segment(const mynah_sp *sp, const char *text, size_t max_tokens,
                       size_t first_max_tokens, int **out_ids, size_t *out_count,
                       size_t **out_lengths, size_t *out_segments,
                       char *error, size_t error_capacity);

/* Reads MYNAH_POCKET_SEGMENT_TOKENS (0 or unset: segmentation off) and
 * MYNAH_POCKET_FIRST_SEGMENT_TOKENS (0 or unset: same as the first). */
size_t mynah_text_segment_tokens_from_env(void);
size_t mynah_text_segment_first_tokens_from_env(void);

#endif
