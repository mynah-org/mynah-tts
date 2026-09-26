/* Text segmentation; see text_segment.h for the contract and its source. */
#include "text_segment.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    size_t begin;  /* byte offsets into the text, trimmed */
    size_t end;
    size_t tokens;
} seg_piece;

typedef struct {
    seg_piece *items;
    size_t count;
    size_t capacity;
} seg_pieces;

static void seg_error(char *error, size_t capacity, const char *fmt, ...) {
    if (error == NULL || capacity == 0u) return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(error, capacity, fmt, ap);
    va_end(ap);
}

static int seg_space(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
}

static void seg_trim(const char *text, size_t *begin, size_t *end) {
    while (*begin < *end && seg_space(text[*begin])) ++*begin;
    while (*end > *begin && seg_space(text[*end - 1u])) --*end;
}

static int seg_count_tokens(const mynah_sp *sp, const char *text, size_t begin,
                            size_t end, size_t *out, char *error, size_t capacity) {
    int *ids = NULL;
    size_t count = 0;
    if (mynah_sp_encode(sp, text + begin, end - begin, &ids, &count, error,
                        capacity) != 0)
        return -1;
    free(ids);
    *out = count;
    return 0;
}

static int seg_push(seg_pieces *list, size_t begin, size_t end, size_t tokens) {
    if (list->count == list->capacity) {
        const size_t grown = list->capacity == 0u ? 16u : list->capacity * 2u;
        if (grown < list->capacity || grown > ((size_t)-1) / sizeof(seg_piece))
            return -1;
        seg_piece *next = (seg_piece *)realloc(list->items, grown * sizeof(seg_piece));
        if (next == NULL) return -1;
        list->items = next;
        list->capacity = grown;
    }
    list->items[list->count].begin = begin;
    list->items[list->count].end = end;
    list->items[list->count].tokens = tokens;
    ++list->count;
    return 0;
}

/* Cut [begin, end) after every run of `marks` that is followed by whitespace or
 * by the end, pushing each trimmed non-empty piece with its token count.
 * Closing quotes and brackets directly after the run stay with the sentence
 * they close. A mark followed by anything else (3.14, e.g. "a,b") does not cut,
 * which is upstream's decimal-period rule in byte form. */
static int seg_split(const mynah_sp *sp, const char *text, size_t begin, size_t end,
                     const char *marks, seg_pieces *out, char *error,
                     size_t capacity) {
    size_t start = begin;
    size_t i = begin;
    while (i < end) {
        if (strchr(marks, text[i]) == NULL || text[i] == '\0') {
            ++i;
            continue;
        }
        size_t j = i;
        while (j < end && text[j] != '\0' &&
               (strchr(marks, text[j]) != NULL || strchr("\"')]", text[j]) != NULL))
            ++j;
        if (j < end && !seg_space(text[j])) {
            i = j;
            continue;
        }
        size_t b = start, e = j;
        seg_trim(text, &b, &e);
        if (e > b) {
            size_t tokens = 0;
            if (seg_count_tokens(sp, text, b, e, &tokens, error, capacity) != 0)
                return -1;
            if (seg_push(out, b, e, tokens) != 0) {
                seg_error(error, capacity, "out of memory segmenting the text");
                return -1;
            }
        }
        start = j;
        i = j;
    }
    size_t b = start, e = end;
    seg_trim(text, &b, &e);
    if (e > b) {
        size_t tokens = 0;
        if (seg_count_tokens(sp, text, b, e, &tokens, error, capacity) != 0) return -1;
        if (seg_push(out, b, e, tokens) != 0) {
            seg_error(error, capacity, "out of memory segmenting the text");
            return -1;
        }
    }
    return 0;
}

int mynah_text_segment(const mynah_sp *sp, const char *text, size_t max_tokens,
                       size_t first_max_tokens, int **out_ids, size_t *out_count,
                       size_t **out_lengths, size_t *out_segments,
                       char *error, size_t error_capacity) {
    if (out_ids != NULL) *out_ids = NULL;
    if (out_lengths != NULL) *out_lengths = NULL;
    if (out_count != NULL) *out_count = 0u;
    if (out_segments != NULL) *out_segments = 0u;
    if (sp == NULL || text == NULL || out_ids == NULL || out_count == NULL ||
        out_lengths == NULL || out_segments == NULL || max_tokens == 0u) {
        seg_error(error, error_capacity, "text segmentation: invalid argument");
        return -1;
    }
    if (first_max_tokens == 0u) first_max_tokens = max_tokens;

    size_t begin = 0u, end = strlen(text);
    seg_trim(text, &begin, &end);
    if (end == begin) {
        seg_error(error, error_capacity, "text segmentation: the text is empty");
        return -1;
    }

    int rc = -1;
    seg_pieces sentences = {0};
    seg_pieces pieces = {0};
    seg_pieces chunks = {0};
    int *ids = NULL;
    size_t *lengths = NULL;

    /* 1. sentences; 2. an oversized sentence falls back to , ; : -- and is kept
     * whole when that finds nothing, exactly as upstream keeps it. */
    if (seg_split(sp, text, begin, end, ".!?", &sentences, error, error_capacity) != 0)
        goto done;
    for (size_t s = 0; s < sentences.count; ++s) {
        const seg_piece *sentence = &sentences.items[s];
        if (sentence->tokens > max_tokens) {
            seg_pieces sub = {0};
            if (seg_split(sp, text, sentence->begin, sentence->end, ",;:", &sub, error,
                          error_capacity) != 0) {
                free(sub.items);
                goto done;
            }
            if (sub.count > 1u) {
                for (size_t k = 0; k < sub.count; ++k) {
                    if (seg_push(&pieces, sub.items[k].begin, sub.items[k].end,
                                 sub.items[k].tokens) != 0) {
                        free(sub.items);
                        seg_error(error, error_capacity,
                                  "out of memory segmenting the text");
                        goto done;
                    }
                }
                free(sub.items);
                continue;
            }
            free(sub.items);
        }
        if (seg_push(&pieces, sentence->begin, sentence->end, sentence->tokens) != 0) {
            seg_error(error, error_capacity, "out of memory segmenting the text");
            goto done;
        }
    }

    /* 3. greedy packing of consecutive pieces. A chunk is the contiguous span
     * of the original text from its first piece to its last, so the text
     * between two packed pieces is the text the caller wrote. */
    for (size_t p = 0; p < pieces.count; ++p) {
        const seg_piece *piece = &pieces.items[p];
        if (chunks.count > 0u) {
            seg_piece *last = &chunks.items[chunks.count - 1u];
            const size_t cap = (chunks.count == 1u) ? first_max_tokens : max_tokens;
            if (last->tokens + piece->tokens <= cap) {
                last->end = piece->end;
                last->tokens += piece->tokens;
                continue;
            }
        }
        if (seg_push(&chunks, piece->begin, piece->end, piece->tokens) != 0) {
            seg_error(error, error_capacity, "out of memory segmenting the text");
            goto done;
        }
    }

    /* 4. tokenize each chunk on its own, as upstream does. */
    lengths = (size_t *)calloc(chunks.count, sizeof(*lengths));
    if (lengths == NULL) {
        seg_error(error, error_capacity, "out of memory segmenting the text");
        goto done;
    }
    size_t total = 0u;
    for (size_t c = 0; c < chunks.count; ++c) {
        int *chunk_ids = NULL;
        size_t chunk_count = 0u;
        if (mynah_sp_encode(sp, text + chunks.items[c].begin,
                            chunks.items[c].end - chunks.items[c].begin, &chunk_ids,
                            &chunk_count, error, error_capacity) != 0)
            goto done;
        if (chunk_count == 0u) {
            free(chunk_ids);
            continue;
        }
        if (total > ((size_t)-1) / sizeof(int) - chunk_count) {
            free(chunk_ids);
            seg_error(error, error_capacity, "text segmentation: size overflow");
            goto done;
        }
        int *grown = (int *)realloc(ids, (total + chunk_count) * sizeof(int));
        if (grown == NULL) {
            free(chunk_ids);
            seg_error(error, error_capacity, "out of memory segmenting the text");
            goto done;
        }
        ids = grown;
        memcpy(ids + total, chunk_ids, chunk_count * sizeof(int));
        free(chunk_ids);
        total += chunk_count;
        lengths[(*out_segments)++] = chunk_count;
    }
    if (*out_segments == 0u) {
        seg_error(error, error_capacity, "text segmentation: the text has no tokens");
        goto done;
    }
    *out_ids = ids;
    *out_count = total;
    *out_lengths = lengths;
    ids = NULL;
    lengths = NULL;
    rc = 0;

done:
    if (rc != 0) *out_segments = 0u;
    free(ids);
    free(lengths);
    free(sentences.items);
    free(pieces.items);
    free(chunks.items);
    return rc;
}

static size_t seg_env_size(const char *name) {
    const char *env = getenv(name);
    if (env == NULL || *env == '\0') return 0u;
    char *end = NULL;
    const long value = strtol(env, &end, 10);
    if (end == env || value <= 0 || value > 100000L) return 0u;
    return (size_t)value;
}

size_t mynah_text_segment_tokens_from_env(void) {
    return seg_env_size("MYNAH_POCKET_SEGMENT_TOKENS");
}

size_t mynah_text_segment_first_tokens_from_env(void) {
    return seg_env_size("MYNAH_POCKET_FIRST_SEGMENT_TOKENS");
}
