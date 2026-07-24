#ifndef MYNAH_TTS_TOKENIZER_H
#define MYNAH_TTS_TOKENIZER_H

#include <stddef.h>

typedef struct mynah_tokenizer mynah_tokenizer;

/* Open the tokenizer for a model pack directory.  Loads all per-language
 * vocabularies and G2P dictionaries found in <model_dir>/tokenizer/.
 * Returns NULL on error. */
mynah_tokenizer *mynah_tokenizer_open(const char *model_dir,
                                      char *error, size_t error_capacity);
void mynah_tokenizer_close(mynah_tokenizer *tok);

/* Encode UTF-8 text for the given language into token IDs.
 * The caller owns *out_ids and must free() it.
 * Returns 0 on success, -1 on error. */
int mynah_tokenizer_encode(const mynah_tokenizer *tok, const char *language,
                           const char *text, int **out_ids, size_t *out_count,
                           char *error, size_t error_capacity);

/* List supported language codes (NULL-terminated). */
const char *const *mynah_tokenizer_languages(const mynah_tokenizer *tok);

#endif
