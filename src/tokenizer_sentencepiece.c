/* SentencePiece Unigram encoder. See tokenizer_sentencepiece.h for the
 * contract.
 *
 * Layout of this file: protobuf reader, ModelProto parse and guards, piece
 * index, UTF-8 decoder, normalizer, Viterbi + byte fallback, public API, and
 * the model-free self-test at the end.
 *
 * SPDX-License-Identifier: MIT */
#if !defined(_POSIX_C_SOURCE)
#define _POSIX_C_SOURCE 200809L
#endif

#include "tokenizer_sentencepiece.h"

#include <fcntl.h>
#include <limits.h>
#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

/* Upstream's two magic constants (unigram_model.cc). */
#define SP_UNK_PENALTY 10.0f
#define SP_USER_DEFINED_PENALTY 10.0f

/* ModelProto.SentencePiece.Type */
#define SP_TYPE_NORMAL 1
#define SP_TYPE_UNKNOWN 2
#define SP_TYPE_CONTROL 3
#define SP_TYPE_USER_DEFINED 4
#define SP_TYPE_UNUSED 5
#define SP_TYPE_BYTE 6

static const unsigned char sp_space_symbol[3] = {0xE2u, 0x96u, 0x81u}; /* U+2581 */
static const unsigned char sp_replacement[3] = {0xEFu, 0xBFu, 0xBDu};  /* U+FFFD */

struct mynah_sp {
    /* The model image. Every `piece` pointer aims into it, so it outlives the
     * pieces and is released exactly once in close(). */
    const unsigned char *image;
    size_t image_length;
    int image_is_mapped; /* 1 = munmap, 0 = free */

    size_t piece_count;
    const char **piece;
    size_t *piece_length;
    float *score;
    unsigned char *type;

    int unk_id;
    int byte_id[256];
    int byte_fallback;

    int add_dummy_prefix;
    int escape_whitespaces;

    float unk_score;
    size_t max_piece_bytes;

    /* Open-addressed (ptr, len) -> id over the segmentable pieces only.
     * Capacity is a power of two; -1 marks an empty slot. */
    int32_t *index;
    size_t index_capacity;
};

#if defined(__GNUC__)
__attribute__((format(printf, 3, 4)))
#endif
static void set_error(char *error, size_t error_capacity, const char *format, ...) {
    va_list args;
    if (error == NULL || error_capacity == 0) return;
    va_start(args, format);
    vsnprintf(error, error_capacity, format, args);
    va_end(args);
}

/* ------------------------------------------------------------------ protobuf */

typedef struct {
    const unsigned char *data;
    size_t length;
    size_t pos;
} pb_reader;

typedef struct {
    uint32_t field;
    uint32_t wire_type;
    uint64_t varint;            /* wire type 0 */
    uint32_t fixed32;           /* wire type 5 */
    uint64_t fixed64;           /* wire type 1 */
    const unsigned char *bytes; /* wire type 2 */
    size_t bytes_length;
} pb_field;

/* 0 on success, -1 on a truncated or over-long varint. */
static int pb_varint(pb_reader *r, uint64_t *out) {
    uint64_t value = 0;
    unsigned shift = 0;
    size_t used = 0;
    for (;;) {
        unsigned char byte;
        if (r->pos >= r->length) return -1;
        byte = r->data[r->pos++];
        used++;
        if (used == 10 && (byte & 0x7Fu) > 1u) return -1; /* would exceed 64 bits */
        if (used > 10) return -1;
        value |= (uint64_t)(byte & 0x7Fu) << shift;
        if ((byte & 0x80u) == 0) break;
        shift += 7;
    }
    *out = value;
    return 0;
}

/* Reads one field. Returns 1 on success, 0 at clean end of buffer, -1 on a
 * malformed or unsupported record. Group wire types (3 and 4) are rejected
 * rather than looped on, and so are the two reserved ones. */
static int pb_next(pb_reader *r, pb_field *out, char *error, size_t error_capacity) {
    uint64_t key = 0;
    uint64_t length = 0;

    memset(out, 0, sizeof(*out));
    if (r->pos >= r->length) return 0;
    if (pb_varint(r, &key) != 0) {
        set_error(error, error_capacity, "malformed protobuf: truncated field key at offset %zu",
                  r->pos);
        return -1;
    }
    out->field = (uint32_t)(key >> 3);
    out->wire_type = (uint32_t)(key & 7u);
    if (out->field == 0) {
        set_error(error, error_capacity, "malformed protobuf: field number 0 at offset %zu", r->pos);
        return -1;
    }
    switch (out->wire_type) {
    case 0:
        if (pb_varint(r, &out->varint) != 0) {
            set_error(error, error_capacity,
                      "malformed protobuf: truncated varint for field %u", (unsigned)out->field);
            return -1;
        }
        return 1;
    case 1:
        if (r->length - r->pos < 8u) {
            set_error(error, error_capacity,
                      "malformed protobuf: truncated 64-bit field %u", (unsigned)out->field);
            return -1;
        }
        memcpy(&out->fixed64, r->data + r->pos, 8);
        r->pos += 8;
        return 1;
    case 2:
        if (pb_varint(r, &length) != 0) {
            set_error(error, error_capacity,
                      "malformed protobuf: truncated length for field %u", (unsigned)out->field);
            return -1;
        }
        if (length > (uint64_t)(r->length - r->pos)) {
            set_error(error, error_capacity,
                      "malformed protobuf: field %u claims %llu bytes but only %zu remain",
                      (unsigned)out->field, (unsigned long long)length, r->length - r->pos);
            return -1;
        }
        out->bytes = r->data + r->pos;
        out->bytes_length = (size_t)length;
        r->pos += (size_t)length;
        return 1;
    case 5: {
        unsigned char b[4];
        if (r->length - r->pos < 4u) {
            set_error(error, error_capacity,
                      "malformed protobuf: truncated 32-bit field %u", (unsigned)out->field);
            return -1;
        }
        memcpy(b, r->data + r->pos, 4);
        r->pos += 4;
        /* Assembled from the wire bytes so the file's little-endian layout is
         * honoured whatever the host does. */
        out->fixed32 = (uint32_t)b[0] | ((uint32_t)b[1] << 8) | ((uint32_t)b[2] << 16) |
                       ((uint32_t)b[3] << 24);
        return 1;
    }
    case 3:
    case 4:
        set_error(error, error_capacity,
                  "unsupported protobuf: group wire type %u on field %u; this reader does not "
                  "implement groups",
                  (unsigned)out->wire_type, (unsigned)out->field);
        return -1;
    default:
        set_error(error, error_capacity, "malformed protobuf: reserved wire type %u on field %u",
                  (unsigned)out->wire_type, (unsigned)out->field);
        return -1;
    }
}

static float pb_float(uint32_t bits) {
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

/* ---------------------------------------------------------------- piece index */

static uint64_t sp_hash(const void *key, size_t length) {
    const unsigned char *bytes = (const unsigned char *)key;
    uint64_t hash = 1469598103934665603ULL;
    size_t i;
    for (i = 0; i < length; i++) {
        hash ^= (uint64_t)bytes[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

static int sp_is_segmentable(unsigned char type) {
    /* Upstream's InitializePieces puts NORMAL, USER_DEFINED and UNUSED into the
     * searchable table and everything else into a reserved map, which is why
     * "<s>" typed by a user cannot reach the control id. */
    return type == SP_TYPE_NORMAL || type == SP_TYPE_USER_DEFINED || type == SP_TYPE_UNUSED;
}

static void sp_index_insert(mynah_sp *sp, size_t id) {
    const size_t mask = sp->index_capacity - 1u;
    size_t slot = (size_t)(sp_hash(sp->piece[id], sp->piece_length[id]) & (uint64_t)mask);
    while (sp->index[slot] >= 0) {
        const size_t other = (size_t)sp->index[slot];
        if (sp->piece_length[other] == sp->piece_length[id] &&
            memcmp(sp->piece[other], sp->piece[id], sp->piece_length[id]) == 0) {
            return; /* duplicate spelling: the first one wins, as upstream does */
        }
        slot = (slot + 1u) & mask;
    }
    sp->index[slot] = (int32_t)id;
}

static int sp_index_lookup(const mynah_sp *sp, const unsigned char *key, size_t length) {
    const size_t mask = sp->index_capacity - 1u;
    size_t slot = (size_t)(sp_hash(key, length) & (uint64_t)mask);
    while (sp->index[slot] >= 0) {
        const size_t id = (size_t)sp->index[slot];
        if (sp->piece_length[id] == length && memcmp(sp->piece[id], key, length) == 0) {
            return (int)id;
        }
        slot = (slot + 1u) & mask;
    }
    return -1;
}

/* ------------------------------------------------------------- UTF-8 decoding */

/* Consumes one character of `available` bytes at `p`. Returns the byte length
 * and sets *valid. An invalid sequence consumes exactly one byte, which is what
 * makes "\xED\xA0\x80" three U+FFFD and not one. The accepted ranges reject
 * overlongs, encoded surrogates, anything above U+10FFFF and truncations,
 * matching string_util::IsValidDecodeUTF8. */
static size_t utf8_next(const unsigned char *p, size_t available, int *valid) {
    unsigned char lead;
    unsigned char low;
    unsigned char high;
    size_t need;
    size_t i;

    *valid = 0;
    if (available == 0) return 0;
    lead = p[0];
    if (lead < 0x80u) {
        *valid = 1;
        return 1;
    }
    if (lead >= 0xC2u && lead <= 0xDFu) {
        need = 2;
        low = 0x80u;
        high = 0xBFu;
    } else if (lead == 0xE0u) {
        need = 3;
        low = 0xA0u;
        high = 0xBFu;
    } else if (lead >= 0xE1u && lead <= 0xECu) {
        need = 3;
        low = 0x80u;
        high = 0xBFu;
    } else if (lead == 0xEDu) {
        need = 3;
        low = 0x80u;
        high = 0x9Fu;
    } else if (lead >= 0xEEu && lead <= 0xEFu) {
        need = 3;
        low = 0x80u;
        high = 0xBFu;
    } else if (lead == 0xF0u) {
        need = 4;
        low = 0x90u;
        high = 0xBFu;
    } else if (lead >= 0xF1u && lead <= 0xF3u) {
        need = 4;
        low = 0x80u;
        high = 0xBFu;
    } else if (lead == 0xF4u) {
        need = 4;
        low = 0x80u;
        high = 0x8Fu;
    } else {
        return 1;
    }
    if (available < need) return 1;
    if (p[1] < low || p[1] > high) return 1;
    for (i = 2; i < need; i++) {
        if (p[i] < 0x80u || p[i] > 0xBFu) return 1;
    }
    *valid = 1;
    return need;
}

/* ------------------------------------------------------- ModelProto and guards */

static void sp_release(mynah_sp *sp) {
    if (sp == NULL) return;
    free(sp->piece);
    free(sp->piece_length);
    free(sp->score);
    free(sp->type);
    free(sp->index);
    if (sp->image != NULL) {
        if (sp->image_is_mapped) {
            munmap((void *)(uintptr_t)sp->image, sp->image_length);
        } else {
            free((void *)(uintptr_t)sp->image);
        }
    }
    free(sp);
}

/* NormalizerSpec (ModelProto field 3). Only the configuration the pinned models
 * actually carry is accepted; everything else names the field it rejects. */
static int sp_parse_normalizer(mynah_sp *sp, const unsigned char *data, size_t length,
                               char *error, size_t error_capacity) {
    pb_reader reader;
    pb_field field;
    const unsigned char *name = NULL;
    size_t name_length = 0;
    size_t charsmap_length = 0;
    int remove_extra_whitespaces = 1; /* protobuf default is true */
    int status;

    reader.data = data;
    reader.length = length;
    reader.pos = 0;
    while ((status = pb_next(&reader, &field, error, error_capacity)) == 1) {
        switch (field.field) {
        case 1:
            if (field.wire_type != 2) break;
            name = field.bytes;
            name_length = field.bytes_length;
            break;
        case 2:
            if (field.wire_type != 2) break;
            charsmap_length = field.bytes_length;
            break;
        case 3:
            if (field.wire_type != 0) break;
            sp->add_dummy_prefix = field.varint != 0;
            break;
        case 4:
            if (field.wire_type != 0) break;
            remove_extra_whitespaces = field.varint != 0;
            break;
        case 5:
            if (field.wire_type != 0) break;
            sp->escape_whitespaces = field.varint != 0;
            break;
        default:
            break;
        }
    }
    if (status < 0) return -1;

    if (name != NULL && !(name_length == 8 && memcmp(name, "identity", 8) == 0)) {
        set_error(error, error_capacity,
                  "unsupported normalizer \"%.*s\": only \"identity\" is implemented, and an "
                  "approximation would silently change tokenization",
                  (int)(name_length > 64 ? 64 : name_length), (const char *)name);
        return -1;
    }
    if (charsmap_length != 0) {
        set_error(error, error_capacity,
                  "unsupported normalizer: precompiled_charsmap is %zu bytes; this reader only "
                  "accepts an empty charsmap (normalizer \"%.*s\")",
                  charsmap_length, (int)(name_length > 64 ? 64 : name_length),
                  name != NULL ? (const char *)name : "");
        return -1;
    }
    if (remove_extra_whitespaces) {
        set_error(error, error_capacity,
                  "unsupported normalizer: remove_extra_whitespaces is set; the pinned models "
                  "have it off and the collapsing branch is not implemented");
        return -1;
    }
    return 0;
}

/* TrainerSpec (ModelProto field 2). Only the handful of fields that change how
 * encoding behaves is read; the field numbers are the counter-intuitive part,
 * so each is named. */
static int sp_parse_trainer(const unsigned char *data, size_t length, size_t piece_count,
                            int *byte_fallback, char *error, size_t error_capacity) {
    pb_reader reader;
    pb_field field;
    int model_type = 1;
    int have_model_type = 0;
    uint64_t vocab_size = 0;
    int have_vocab_size = 0;
    int treat_whitespace_as_suffix = 0;
    size_t pretokenization_delimiter = 0;
    int status;

    reader.data = data;
    reader.length = length;
    reader.pos = 0;
    while ((status = pb_next(&reader, &field, error, error_capacity)) == 1) {
        switch (field.field) {
        case 3: /* model_type */
            if (field.wire_type != 0) break;
            model_type = (int)field.varint;
            have_model_type = 1;
            break;
        case 4: /* vocab_size */
            if (field.wire_type != 0) break;
            vocab_size = field.varint;
            have_vocab_size = 1;
            break;
        case 24: /* treat_whitespace_as_suffix */
            if (field.wire_type != 0) break;
            treat_whitespace_as_suffix = field.varint != 0;
            break;
        case 35: /* byte_fallback */
            if (field.wire_type != 0) break;
            *byte_fallback = field.varint != 0;
            break;
        case 53: /* pretokenization_delimiter */
            if (field.wire_type != 2) break;
            pretokenization_delimiter = field.bytes_length;
            break;
        default:
            break;
        }
    }
    if (status < 0) return -1;

    if (have_model_type && model_type != 1) {
        set_error(error, error_capacity,
                  "unsupported model_type %d in TrainerSpec: this reader implements UNIGRAM (1) "
                  "only",
                  model_type);
        return -1;
    }
    if (have_vocab_size && vocab_size != (uint64_t)piece_count) {
        set_error(error, error_capacity,
                  "inconsistent model: TrainerSpec.vocab_size is %llu but the file carries %zu "
                  "pieces",
                  (unsigned long long)vocab_size, piece_count);
        return -1;
    }
    if (treat_whitespace_as_suffix) {
        set_error(error, error_capacity,
                  "unsupported model: treat_whitespace_as_suffix is set and the suffix form of "
                  "the space marker is not implemented");
        return -1;
    }
    if (pretokenization_delimiter != 0) {
        set_error(error, error_capacity,
                  "unsupported model: a pretokenization_delimiter of %zu bytes is set and "
                  "pre-splitting is not implemented",
                  pretokenization_delimiter);
        return -1;
    }
    return 0;
}

/* "<0xNN>" -> 0..255, or -1. */
static int sp_byte_of_piece(const char *piece, size_t length) {
    int value = 0;
    int i;
    if (length != 6) return -1;
    if (piece[0] != '<' || piece[1] != '0' || piece[2] != 'x' || piece[5] != '>') return -1;
    for (i = 3; i < 5; i++) {
        const char c = piece[i];
        value <<= 4;
        if (c >= '0' && c <= '9') {
            value |= c - '0';
        } else if (c >= 'A' && c <= 'F') {
            value |= c - 'A' + 10;
        } else if (c >= 'a' && c <= 'f') {
            value |= c - 'a' + 10;
        } else {
            return -1;
        }
    }
    return value;
}

static int sp_parse_piece(const unsigned char *data, size_t length, const char **out_piece,
                          size_t *out_piece_length, float *out_score, unsigned char *out_type,
                          char *error, size_t error_capacity) {
    pb_reader reader;
    pb_field field;
    int status;

    *out_piece = NULL;
    *out_piece_length = 0;
    *out_score = 0.0f;
    *out_type = SP_TYPE_NORMAL; /* protobuf default */

    reader.data = data;
    reader.length = length;
    reader.pos = 0;
    while ((status = pb_next(&reader, &field, error, error_capacity)) == 1) {
        switch (field.field) {
        case 1:
            if (field.wire_type != 2) break;
            *out_piece = (const char *)field.bytes;
            *out_piece_length = field.bytes_length;
            break;
        case 2:
            /* Wire type 5: the score is float32, not double. Reading it as a
             * double would produce plausible garbage. */
            if (field.wire_type != 5) break;
            *out_score = pb_float(field.fixed32);
            break;
        case 3:
            if (field.wire_type != 0) break;
            if (field.varint < 1u || field.varint > 6u) {
                set_error(error, error_capacity, "malformed model: unknown piece type %llu",
                          (unsigned long long)field.varint);
                return -1;
            }
            *out_type = (unsigned char)field.varint;
            break;
        default:
            break;
        }
    }
    return status < 0 ? -1 : 0;
}

static int sp_parse(mynah_sp *sp, char *error, size_t error_capacity) {
    pb_reader reader;
    pb_field field;
    const unsigned char *trainer = NULL;
    size_t trainer_length = 0;
    const unsigned char *normalizer = NULL;
    size_t normalizer_length = 0;
    size_t count = 0;
    size_t i;
    size_t indexable = 0;
    size_t capacity;
    float min_score = 0.0f;
    int have_min_score = 0;
    int have_unk = 0;
    int status;

    /* Pass one: how many pieces, and where the two spec messages live. */
    reader.data = sp->image;
    reader.length = sp->image_length;
    reader.pos = 0;
    while ((status = pb_next(&reader, &field, error, error_capacity)) == 1) {
        if (field.field == 1 && field.wire_type == 2) {
            count++;
        } else if (field.field == 2 && field.wire_type == 2) {
            trainer = field.bytes;
            trainer_length = field.bytes_length;
        } else if (field.field == 3 && field.wire_type == 2) {
            normalizer = field.bytes;
            normalizer_length = field.bytes_length;
        }
    }
    if (status < 0) return -1;

    if (count == 0) {
        set_error(error, error_capacity, "not a SentencePiece model: no pieces in the ModelProto");
        return -1;
    }
    if (count > (size_t)INT_MAX) {
        set_error(error, error_capacity, "model too large: %zu pieces", count);
        return -1;
    }

    if (count > SIZE_MAX / sizeof(const char *) || count > SIZE_MAX / sizeof(size_t) ||
        count > SIZE_MAX / sizeof(float)) {
        set_error(error, error_capacity, "model too large: %zu pieces", count);
        return -1;
    }
    sp->piece = calloc(count, sizeof(*sp->piece));
    sp->piece_length = calloc(count, sizeof(*sp->piece_length));
    sp->score = calloc(count, sizeof(*sp->score));
    sp->type = calloc(count, sizeof(*sp->type));
    if (sp->piece == NULL || sp->piece_length == NULL || sp->score == NULL || sp->type == NULL) {
        set_error(error, error_capacity, "out of memory reading %zu pieces", count);
        return -1;
    }
    sp->piece_count = count;

    /* Pass two: fill the arrays in wire order, which is id order. */
    reader.pos = 0;
    i = 0;
    while ((status = pb_next(&reader, &field, error, error_capacity)) == 1) {
        if (field.field != 1 || field.wire_type != 2) continue;
        if (sp_parse_piece(field.bytes, field.bytes_length, &sp->piece[i], &sp->piece_length[i],
                           &sp->score[i], &sp->type[i], error, error_capacity) != 0) {
            return -1;
        }
        if (sp->piece_length[i] == 0) {
            set_error(error, error_capacity, "malformed model: piece %zu has an empty spelling", i);
            return -1;
        }
        i++;
    }
    if (status < 0) return -1;

    sp->byte_fallback = -1; /* undecided until TrainerSpec or the byte pieces say */
    if (trainer != NULL &&
        sp_parse_trainer(trainer, trainer_length, count, &sp->byte_fallback, error,
                         error_capacity) != 0) {
        return -1;
    }
    if (normalizer != NULL &&
        sp_parse_normalizer(sp, normalizer, normalizer_length, error, error_capacity) != 0) {
        return -1;
    }

    for (i = 0; i < 256; i++) sp->byte_id[i] = -1;
    sp->unk_id = -1;
    for (i = 0; i < count; i++) {
        const unsigned char type = sp->type[i];
        if (type == SP_TYPE_UNKNOWN) {
            if (have_unk) {
                set_error(error, error_capacity,
                          "malformed model: more than one piece of type UNKNOWN (ids %d and %zu)",
                          sp->unk_id, i);
                return -1;
            }
            have_unk = 1;
            sp->unk_id = (int)i;
        } else if (type == SP_TYPE_BYTE) {
            const int byte = sp_byte_of_piece(sp->piece[i], sp->piece_length[i]);
            if (byte < 0) {
                set_error(error, error_capacity,
                          "malformed model: piece %zu is typed BYTE but is not spelled \"<0xNN>\"",
                          i);
                return -1;
            }
            if (sp->byte_id[byte] < 0) sp->byte_id[byte] = (int)i;
        }
        if (type == SP_TYPE_NORMAL) {
            /* min_score is over NORMAL pieces only. Folding in the BYTE and
             * CONTROL pieces, whose score is 0.0, would give unk_score = -10
             * and byte fallback would essentially never be chosen. */
            if (!have_min_score || sp->score[i] < min_score) {
                min_score = sp->score[i];
                have_min_score = 1;
            }
        }
        if (sp_is_segmentable(type)) indexable++;
    }
    if (!have_unk) {
        set_error(error, error_capacity,
                  "malformed model: no piece of type UNKNOWN, so there is no unk id");
        return -1;
    }
    if (!have_min_score) {
        set_error(error, error_capacity, "malformed model: no piece of type NORMAL");
        return -1;
    }
    sp->unk_score = min_score - SP_UNK_PENALTY;

    if (sp->byte_fallback < 0) {
        /* No TrainerSpec: infer it from a complete set of byte pieces. */
        int complete = 1;
        for (i = 0; i < 256; i++) {
            if (sp->byte_id[i] < 0) complete = 0;
        }
        sp->byte_fallback = complete;
    }
    if (sp->byte_fallback) {
        for (i = 0; i < 256; i++) {
            if (sp->byte_id[i] < 0) {
                set_error(error, error_capacity,
                          "malformed model: byte_fallback is set but the piece <0x%02X> is missing",
                          (unsigned)i);
                return -1;
            }
        }
    }

    if (indexable == 0) {
        set_error(error, error_capacity, "malformed model: no segmentable piece");
        return -1;
    }
    capacity = 16;
    while (capacity < indexable * 2u) {
        if (capacity > SIZE_MAX / 2u) {
            set_error(error, error_capacity, "model too large to index: %zu pieces", indexable);
            return -1;
        }
        capacity *= 2u;
    }
    if (capacity > SIZE_MAX / sizeof(*sp->index)) {
        set_error(error, error_capacity, "model too large to index: %zu pieces", indexable);
        return -1;
    }
    sp->index = malloc(capacity * sizeof(*sp->index));
    if (sp->index == NULL) {
        set_error(error, error_capacity, "out of memory indexing %zu pieces", indexable);
        return -1;
    }
    sp->index_capacity = capacity;
    for (i = 0; i < capacity; i++) sp->index[i] = -1;

    sp->max_piece_bytes = 0;
    for (i = 0; i < count; i++) {
        if (!sp_is_segmentable(sp->type[i])) continue;
        sp_index_insert(sp, i);
        if (sp->piece_length[i] > sp->max_piece_bytes) sp->max_piece_bytes = sp->piece_length[i];
    }
    return 0;
}

/* ------------------------------------------------------------------ lifecycle */

static int sp_finish_open(mynah_sp *sp, mynah_sp **out, char *error, size_t error_capacity) {
    if (sp_parse(sp, error, error_capacity) != 0) {
        sp_release(sp);
        return -1;
    }
    *out = sp;
    return 0;
}

int mynah_sp_open_memory(const void *data, size_t length, mynah_sp **out, char *error,
                         size_t error_capacity) {
    mynah_sp *sp;
    unsigned char *copy;

    if (out == NULL) return -1;
    *out = NULL;
    if (data == NULL || length == 0) {
        set_error(error, error_capacity, "no tokenizer model data given");
        return -1;
    }
    sp = calloc(1, sizeof(*sp));
    if (sp == NULL) {
        set_error(error, error_capacity, "out of memory opening the tokenizer model");
        return -1;
    }
    copy = malloc(length);
    if (copy == NULL) {
        free(sp);
        set_error(error, error_capacity, "out of memory copying %zu bytes of tokenizer model",
                  length);
        return -1;
    }
    memcpy(copy, data, length);
    sp->image = copy;
    sp->image_length = length;
    sp->image_is_mapped = 0;
    sp->add_dummy_prefix = 1;
    sp->escape_whitespaces = 1;
    return sp_finish_open(sp, out, error, error_capacity);
}

int mynah_sp_open(const char *path, mynah_sp **out, char *error, size_t error_capacity) {
    mynah_sp *sp;
    struct stat info;
    void *mapping;
    int fd;

    if (out == NULL) return -1;
    *out = NULL;
    if (path == NULL) {
        set_error(error, error_capacity, "no tokenizer model path given");
        return -1;
    }
    fd = open(path, O_RDONLY);
    if (fd < 0) {
        set_error(error, error_capacity, "cannot open tokenizer model \"%s\"", path);
        return -1;
    }
    if (fstat(fd, &info) != 0) {
        close(fd);
        set_error(error, error_capacity, "cannot stat tokenizer model \"%s\"", path);
        return -1;
    }
    if (!S_ISREG(info.st_mode)) {
        close(fd);
        set_error(error, error_capacity, "tokenizer model \"%s\" is not a regular file", path);
        return -1;
    }
    if (info.st_size <= 0 || (uintmax_t)info.st_size > (uintmax_t)SIZE_MAX) {
        close(fd);
        set_error(error, error_capacity, "tokenizer model \"%s\" is empty or too large", path);
        return -1;
    }
    mapping = mmap(NULL, (size_t)info.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (mapping == MAP_FAILED) {
        set_error(error, error_capacity, "cannot map tokenizer model \"%s\"", path);
        return -1;
    }
    sp = calloc(1, sizeof(*sp));
    if (sp == NULL) {
        munmap(mapping, (size_t)info.st_size);
        set_error(error, error_capacity, "out of memory opening the tokenizer model");
        return -1;
    }
    sp->image = mapping;
    sp->image_length = (size_t)info.st_size;
    sp->image_is_mapped = 1;
    sp->add_dummy_prefix = 1;
    sp->escape_whitespaces = 1;
    if (sp_finish_open(sp, out, error, error_capacity) != 0) {
        /* sp_finish_open released everything, including the mapping. The reason
         * is copied out first: snprintf may not read and write the same buffer. */
        char reason[256];
        snprintf(reason, sizeof(reason), "%s", error != NULL && error_capacity > 0 ? error : "");
        set_error(error, error_capacity, "%s (in \"%s\")", reason, path);
        return -1;
    }
    return 0;
}

void mynah_sp_close(mynah_sp *sp) { sp_release(sp); }

size_t mynah_sp_vocab_size(const mynah_sp *sp) { return sp == NULL ? 0 : sp->piece_count; }

int mynah_sp_unk_id(const mynah_sp *sp) { return sp == NULL ? -1 : sp->unk_id; }

int mynah_sp_piece(const mynah_sp *sp, int id, const char **piece, size_t *length) {
    if (sp == NULL || id < 0 || (size_t)id >= sp->piece_count) return -1;
    if (piece != NULL) *piece = sp->piece[id];
    if (length != NULL) *length = sp->piece_length[id];
    return 0;
}

/* ----------------------------------------------------------------- normalizer */

typedef struct {
    unsigned char *bytes;
    size_t length;
    size_t *starts; /* char c occupies [starts[c], starts[c+1]) */
    size_t nchar;
} sp_normalized;

static void sp_normalized_release(sp_normalized *n) {
    free(n->bytes);
    free(n->starts);
    n->bytes = NULL;
    n->starts = NULL;
    n->length = 0;
    n->nchar = 0;
}

/* The four rules, in upstream's order: an empty input short-circuits before the
 * dummy prefix; the prefix is one escaped space; every input character is
 * validated and an invalid one becomes U+FFFD consuming a single byte; only
 * 0x20 is escaped, so tabs, newlines and NBSP fall through to byte fallback. */
static int sp_normalize(const mynah_sp *sp, const char *text, size_t text_length,
                        sp_normalized *out, char *error, size_t error_capacity) {
    const unsigned char *input = (const unsigned char *)text;
    size_t capacity;
    size_t starts_capacity;
    size_t pos = 0;

    out->bytes = NULL;
    out->starts = NULL;
    out->length = 0;
    out->nchar = 0;
    if (text_length == 0) return 0;

    /* Worst case every input byte expands to a three-byte U+FFFD, plus the
     * three-byte dummy prefix. */
    if (text_length > (SIZE_MAX - 8u) / 3u) {
        set_error(error, error_capacity, "text too long to encode: %zu bytes", text_length);
        return -1;
    }
    capacity = text_length * 3u + 3u;
    starts_capacity = text_length + 2u; /* at most one char per byte, plus the prefix and the end */
    if (starts_capacity > SIZE_MAX / sizeof(size_t)) {
        set_error(error, error_capacity, "text too long to encode: %zu bytes", text_length);
        return -1;
    }
    out->bytes = malloc(capacity);
    out->starts = malloc(starts_capacity * sizeof(size_t));
    if (out->bytes == NULL || out->starts == NULL) {
        sp_normalized_release(out);
        set_error(error, error_capacity, "out of memory normalizing %zu bytes", text_length);
        return -1;
    }

    if (sp->add_dummy_prefix) {
        out->starts[out->nchar++] = out->length;
        if (sp->escape_whitespaces) {
            memcpy(out->bytes + out->length, sp_space_symbol, 3);
            out->length += 3;
        } else {
            out->bytes[out->length++] = ' ';
        }
    }
    while (pos < text_length) {
        int valid = 0;
        const size_t used = utf8_next(input + pos, text_length - pos, &valid);
        out->starts[out->nchar++] = out->length;
        if (!valid) {
            memcpy(out->bytes + out->length, sp_replacement, 3);
            out->length += 3;
        } else if (used == 1 && input[pos] == ' ' && sp->escape_whitespaces) {
            memcpy(out->bytes + out->length, sp_space_symbol, 3);
            out->length += 3;
        } else {
            memcpy(out->bytes + out->length, input + pos, used);
            out->length += used;
        }
        pos += used;
    }
    out->starts[out->nchar] = out->length;
    return 0;
}

/* -------------------------------------------------------------------- Viterbi */

/* Runs the lattice over `n` and writes the ids. Positions are characters, not
 * bytes. Arithmetic stays in float32 so the DP matches upstream bit for bit;
 * there is nothing here for a reassociating compiler to regroup. */
static int sp_viterbi(const mynah_sp *sp, const sp_normalized *n, int *ids, size_t ids_capacity,
                      size_t *out_count, char *error, size_t error_capacity) {
    /* The DP accumulator is double, not float, even though upstream's lattice
     * stores float scores. On a ~100k-character input the accumulated path
     * score reaches about -3e5, where a float32 ULP (~0.03) is larger than the
     * gap between competing segmentations, and ties resolve arbitrarily. In
     * float this implementation diverged from sentencepiece on 6 ids out of
     * 102,899 for English and on one German case; in double, four of the five
     * languages are exact over the whole corpus. Double costs 4 bytes per
     * character in a prefill-only array. See .work/tokenizer-sentencepiece.md. */
    double *best = NULL;
    size_t *prev_char = NULL;
    int32_t *prev_id = NULL;
    size_t *order = NULL;
    size_t count = 0;
    size_t c;
    size_t k;
    size_t nodes = 0;
    int result = -1;

    *out_count = 0;
    if (n->nchar == 0) return 0;
    if (n->nchar > SIZE_MAX / sizeof(size_t) - 2u) {
        set_error(error, error_capacity, "text too long to encode");
        return -1;
    }
    best = malloc((n->nchar + 1u) * sizeof(*best));
    prev_char = malloc((n->nchar + 1u) * sizeof(*prev_char));
    prev_id = malloc((n->nchar + 1u) * sizeof(*prev_id));
    order = malloc((n->nchar + 1u) * sizeof(*order));
    if (best == NULL || prev_char == NULL || prev_id == NULL || order == NULL) {
        set_error(error, error_capacity, "out of memory encoding %zu characters", n->nchar);
        goto done;
    }
    for (c = 0; c <= n->nchar; c++) {
        best[c] = -INFINITY;
        prev_char[c] = 0;
        prev_id[c] = -1;
    }
    best[0] = 0.0;

    for (c = 0; c < n->nchar; c++) {
        const size_t start = n->starts[c];
        int has_single = 0;
        size_t cj;
        if (best[c] == -INFINITY) continue; /* cannot happen: the UNK edge keeps every position
                                             * reachable. Kept so a future change fails loudly. */
        for (cj = c + 1u; cj <= n->nchar; cj++) {
            const size_t span = n->starts[cj] - start;
            int id;
            float score;
            if (span > sp->max_piece_bytes) break;
            id = sp_index_lookup(sp, n->bytes + start, span);
            if (id < 0) continue;
            if (sp->type[id] == SP_TYPE_UNUSED) continue; /* skipped, and not a single node */
            if (cj - c == 1u) has_single = 1;
            score = sp->type[id] == SP_TYPE_USER_DEFINED
                        ? (float)(cj - c) * SP_USER_DEFINED_PENALTY
                        : sp->score[id];
            if (best[c] + score > best[cj]) {
                best[cj] = best[c] + score;
                prev_char[cj] = c;
                prev_id[cj] = (int32_t)id;
            }
        }
        if (!has_single) {
            /* The unknown node spans exactly one character, never one byte. */
            if (best[c] + sp->unk_score > best[c + 1u]) {
                best[c + 1u] = best[c] + sp->unk_score;
                prev_char[c + 1u] = c;
                prev_id[c + 1u] = -1;
            }
        }
    }

    if (best[n->nchar] == -INFINITY) {
        set_error(error, error_capacity, "internal error: the lattice left position %zu unreachable",
                  n->nchar);
        goto done;
    }
    for (c = n->nchar; c > 0; c = prev_char[c]) order[nodes++] = c;

    for (k = nodes; k > 0; k--) {
        const size_t end = order[k - 1u];
        const size_t start = prev_char[end];
        const int32_t id = prev_id[end];
        if (id >= 0) {
            if (ids != NULL && count < ids_capacity) ids[count] = (int)id;
            count++;
            continue;
        }
        if (sp->byte_fallback) {
            /* The bytes expanded are those of the normalized buffer, so an
             * invalid 0xFF yields <0xEF> <0xBF> <0xBD>. */
            size_t b;
            for (b = n->starts[start]; b < n->starts[end]; b++) {
                if (ids != NULL && count < ids_capacity) ids[count] = sp->byte_id[n->bytes[b]];
                count++;
            }
        } else {
            if (ids != NULL && count < ids_capacity) ids[count] = sp->unk_id;
            count++;
        }
    }
    *out_count = count;
    result = 0;

done:
    free(best);
    free(prev_char);
    free(prev_id);
    free(order);
    return result;
}

/* ---------------------------------------------------------------- encode API */

int mynah_sp_encode_into(const mynah_sp *sp, const char *text, size_t text_length, int *ids,
                         size_t capacity, size_t *out_count, char *error, size_t error_capacity) {
    sp_normalized normalized;
    size_t count = 0;

    if (out_count != NULL) *out_count = 0;
    if (sp == NULL || out_count == NULL || (text == NULL && text_length != 0) ||
        (ids == NULL && capacity != 0)) {
        set_error(error, error_capacity, "invalid arguments to the SentencePiece encoder");
        return -1;
    }
    if (sp_normalize(sp, text, text_length, &normalized, error, error_capacity) != 0) return -1;
    if (sp_viterbi(sp, &normalized, ids, capacity, &count, error, error_capacity) != 0) {
        sp_normalized_release(&normalized);
        return -1;
    }
    sp_normalized_release(&normalized);
    *out_count = count;
    if (count > capacity) {
        set_error(error, error_capacity, "token buffer too small: %zu ids needed, %zu available",
                  count, capacity);
        return -1;
    }
    return 0;
}

int mynah_sp_encode(const mynah_sp *sp, const char *text, size_t text_length, int **out_ids,
                    size_t *out_count, char *error, size_t error_capacity) {
    sp_normalized normalized;
    size_t count = 0;
    size_t capacity;
    int *ids;

    if (out_ids != NULL) *out_ids = NULL;
    if (out_count != NULL) *out_count = 0;
    if (sp == NULL || out_ids == NULL || out_count == NULL || (text == NULL && text_length != 0)) {
        set_error(error, error_capacity, "invalid arguments to the SentencePiece encoder");
        return -1;
    }
    if (sp_normalize(sp, text, text_length, &normalized, error, error_capacity) != 0) return -1;

    /* One id per normalized byte is the ceiling: a matched piece costs one id
     * for at least one byte, and a byte-fallback character costs one per byte. */
    capacity = normalized.length == 0 ? 1u : normalized.length;
    if (capacity > SIZE_MAX / sizeof(int)) {
        sp_normalized_release(&normalized);
        set_error(error, error_capacity, "text too long to encode");
        return -1;
    }
    ids = malloc(capacity * sizeof(int));
    if (ids == NULL) {
        sp_normalized_release(&normalized);
        set_error(error, error_capacity, "out of memory allocating %zu token ids", capacity);
        return -1;
    }
    if (sp_viterbi(sp, &normalized, ids, capacity, &count, error, error_capacity) != 0) {
        free(ids);
        sp_normalized_release(&normalized);
        return -1;
    }
    sp_normalized_release(&normalized);
    if (count > capacity) { /* unreachable; a bound bug must not become a write */
        free(ids);
        set_error(error, error_capacity,
                  "internal error: %zu ids produced for a %zu-id bound", count, capacity);
        return -1;
    }
    *out_ids = ids;
    *out_count = count;
    return 0;
}

/* ------------------------------------------------------------------ self-test */

/* A growable buffer, only used to assemble the synthetic ModelProto. `failed`
 * latches an allocation failure so the callers can stay branch-free. */
typedef struct {
    unsigned char *data;
    size_t length;
    size_t capacity;
    int failed;
} sp_buf;

static void sp_buf_release(sp_buf *b) {
    free(b->data);
    b->data = NULL;
    b->length = 0;
    b->capacity = 0;
}

static void sp_buf_raw(sp_buf *b, const void *data, size_t length) {
    if (b->failed) return;
    if (b->length + length > b->capacity) {
        size_t capacity = b->capacity == 0 ? 256u : b->capacity;
        unsigned char *grown;
        while (capacity < b->length + length) capacity *= 2u;
        grown = realloc(b->data, capacity);
        if (grown == NULL) {
            b->failed = 1;
            return;
        }
        b->data = grown;
        b->capacity = capacity;
    }
    memcpy(b->data + b->length, data, length);
    b->length += length;
}

static void sp_buf_varint(sp_buf *b, uint64_t value) {
    unsigned char bytes[10];
    size_t n = 0;
    do {
        unsigned char byte = (unsigned char)(value & 0x7Fu);
        value >>= 7;
        if (value != 0) byte |= 0x80u;
        bytes[n++] = byte;
    } while (value != 0);
    sp_buf_raw(b, bytes, n);
}

static void sp_buf_tag(sp_buf *b, uint32_t field, uint32_t wire_type) {
    sp_buf_varint(b, ((uint64_t)field << 3) | (uint64_t)wire_type);
}

static void sp_buf_uint(sp_buf *b, uint32_t field, uint64_t value) {
    sp_buf_tag(b, field, 0);
    sp_buf_varint(b, value);
}

static void sp_buf_bytes(sp_buf *b, uint32_t field, const void *data, size_t length) {
    sp_buf_tag(b, field, 2);
    sp_buf_varint(b, (uint64_t)length);
    sp_buf_raw(b, data, length);
}

static void sp_buf_float(sp_buf *b, uint32_t field, float value) {
    uint32_t bits;
    unsigned char bytes[4];
    memcpy(&bits, &value, sizeof(bits));
    bytes[0] = (unsigned char)(bits & 0xFFu);
    bytes[1] = (unsigned char)((bits >> 8) & 0xFFu);
    bytes[2] = (unsigned char)((bits >> 16) & 0xFFu);
    bytes[3] = (unsigned char)((bits >> 24) & 0xFFu);
    sp_buf_tag(b, field, 5);
    sp_buf_raw(b, bytes, 4);
}

static void sp_buf_piece(sp_buf *b, const char *piece, float score, int type) {
    sp_buf inner;
    memset(&inner, 0, sizeof(inner));
    sp_buf_bytes(&inner, 1, piece, strlen(piece));
    sp_buf_float(&inner, 2, score);
    sp_buf_uint(&inner, 3, (uint64_t)type);
    if (inner.failed) {
        b->failed = 1;
    } else {
        sp_buf_bytes(b, 1, inner.data, inner.length);
    }
    sp_buf_release(&inner);
}

/* The toy vocabulary. Ids are positional: 0 <unk>, 1 <s>, 2..257 the bytes,
 * 258.. the normal pieces, then one USER_DEFINED and one UNUSED. */
typedef struct {
    const char *piece;
    float score;
    int type;
} sp_test_entry;

static const sp_test_entry sp_test_tail[] = {
    {"\xE2\x96\x81", -3.0f, SP_TYPE_NORMAL},         /* 258  _        */
    {"a", -5.0f, SP_TYPE_NORMAL},                    /* 259           */
    {"b", -5.0f, SP_TYPE_NORMAL},                    /* 260           */
    {"c", -5.0f, SP_TYPE_NORMAL},                    /* 261           */
    {"\xE2\x96\x81" "a", -2.0f, SP_TYPE_NORMAL},     /* 262  _a       */
    {"ab", -1.0f, SP_TYPE_NORMAL},                   /* 263           */
    {"abc", -0.5f, SP_TYPE_NORMAL},                  /* 264           */
    {"\xE2\x96\x81" "ab", -1.5f, SP_TYPE_NORMAL},    /* 265  _ab      */
    {"\xE2\x96\x81" "abc", -0.2f, SP_TYPE_NORMAL},   /* 266  _abc     */
    {"<", -6.0f, SP_TYPE_NORMAL},                    /* 267           */
    {"s", -6.0f, SP_TYPE_NORMAL},                    /* 268           */
    {">", -6.0f, SP_TYPE_NORMAL},                    /* 269           */
    {"zz", -35.0f, SP_TYPE_NORMAL},                  /* 270  min score */
    {"QR", -30.0f, SP_TYPE_NORMAL},                  /* 271           */
    {"\xC3\xA9", -4.0f, SP_TYPE_NORMAL},             /* 272  e-acute  */
    {"x", -5.0f, SP_TYPE_NORMAL},                    /* 273           */
    {"\xE2\x96\x81" "x", -2.5f, SP_TYPE_NORMAL},     /* 274  _x       */
    {"[UD]", -1000.0f, SP_TYPE_USER_DEFINED},        /* 275           */
    {"q", -0.1f, SP_TYPE_UNUSED},                    /* 276           */
};

#define SP_TEST_SPACE 258
#define SP_TEST_A 259
#define SP_TEST_B 260
#define SP_TEST_SPACE_A 262
#define SP_TEST_LT 267
#define SP_TEST_S 268
#define SP_TEST_GT 269
#define SP_TEST_QR 271
#define SP_TEST_EACUTE 272
#define SP_TEST_UD 275
#define SP_TEST_BYTE(b) (2 + (b))

typedef struct {
    int model_type;              /* TrainerSpec field 3 */
    long vocab_size;             /* -1 = the real count */
    const char *normalizer_name; /* NULL = omit */
    size_t charsmap_length;
    int remove_extra_whitespaces;
    int treat_whitespace_as_suffix;
    int byte_fallback;
    const char *pretokenization_delimiter;
    int omit_unknown;
    int omit_byte_pieces;
    int inject_group;
} sp_test_options;

static void sp_test_defaults(sp_test_options *options) {
    memset(options, 0, sizeof(*options));
    options->model_type = 1;
    options->vocab_size = -1;
    options->normalizer_name = "identity";
    options->byte_fallback = 1;
}

static int sp_test_build(sp_buf *out, const sp_test_options *options) {
    sp_buf trainer;
    sp_buf normalizer;
    size_t i;
    size_t pieces = 0;

    memset(out, 0, sizeof(*out));
    memset(&trainer, 0, sizeof(trainer));
    memset(&normalizer, 0, sizeof(normalizer));

    if (!options->omit_unknown) {
        sp_buf_piece(out, "<unk>", 0.0f, SP_TYPE_UNKNOWN);
        pieces++;
    }
    sp_buf_piece(out, "<s>", 0.0f, SP_TYPE_CONTROL);
    pieces++;
    if (!options->omit_byte_pieces) {
        for (i = 0; i < 256; i++) {
            char name[8];
            snprintf(name, sizeof(name), "<0x%02X>", (unsigned)i);
            sp_buf_piece(out, name, 0.0f, SP_TYPE_BYTE);
            pieces++;
        }
    }
    for (i = 0; i < sizeof(sp_test_tail) / sizeof(sp_test_tail[0]); i++) {
        sp_buf_piece(out, sp_test_tail[i].piece, sp_test_tail[i].score, sp_test_tail[i].type);
        pieces++;
    }

    sp_buf_uint(&trainer, 3, (uint64_t)options->model_type);
    sp_buf_uint(&trainer, 4,
                options->vocab_size < 0 ? (uint64_t)pieces : (uint64_t)options->vocab_size);
    sp_buf_uint(&trainer, 24, (uint64_t)(options->treat_whitespace_as_suffix != 0));
    sp_buf_uint(&trainer, 35, (uint64_t)(options->byte_fallback != 0));
    sp_buf_uint(&trainer, 40, 77); /* deliberately wrong: the unk id comes from the piece type */
    if (options->pretokenization_delimiter != NULL) {
        sp_buf_bytes(&trainer, 53, options->pretokenization_delimiter,
                     strlen(options->pretokenization_delimiter));
    }

    if (options->normalizer_name != NULL) {
        sp_buf_bytes(&normalizer, 1, options->normalizer_name, strlen(options->normalizer_name));
    }
    if (options->charsmap_length != 0) {
        unsigned char *charsmap = calloc(options->charsmap_length, 1);
        if (charsmap == NULL) {
            normalizer.failed = 1;
        } else {
            sp_buf_bytes(&normalizer, 2, charsmap, options->charsmap_length);
            free(charsmap);
        }
    }
    sp_buf_uint(&normalizer, 3, 1); /* add_dummy_prefix */
    sp_buf_uint(&normalizer, 4, (uint64_t)(options->remove_extra_whitespaces != 0));
    /* escape_whitespaces (field 5) stays absent, so it defaults to true. */

    if (!trainer.failed) sp_buf_bytes(out, 2, trainer.data, trainer.length);
    if (!normalizer.failed) sp_buf_bytes(out, 3, normalizer.data, normalizer.length);
    if (options->inject_group) {
        sp_buf_tag(out, 9, 3); /* start-group, which this reader must refuse */
        sp_buf_uint(out, 10, 1);
        sp_buf_tag(out, 9, 4);
    }
    if (trainer.failed || normalizer.failed) out->failed = 1;
    sp_buf_release(&trainer);
    sp_buf_release(&normalizer);
    return out->failed ? -1 : 0;
}

static int sp_test_fail(char *error, size_t error_capacity, const char *check, const char *detail) {
    set_error(error, error_capacity, "self-test failed at \"%s\": %s", check, detail);
    return -1;
}

static int sp_test_expect(const mynah_sp *sp, const char *check, const char *text,
                          size_t text_length, const int *expected, size_t expected_count,
                          char *error, size_t error_capacity) {
    char detail[512];
    char local[256];
    int *ids = NULL;
    size_t count = 0;
    size_t i;
    size_t used = 0;

    if (mynah_sp_encode(sp, text, text_length, &ids, &count, local, sizeof(local)) != 0) {
        free(ids);
        return sp_test_fail(error, error_capacity, check, local);
    }
    if (count == expected_count) {
        int same = 1;
        for (i = 0; i < count; i++) {
            if (ids[i] != expected[i]) same = 0;
        }
        if (same) {
            free(ids);
            return 0;
        }
    }
    used = (size_t)snprintf(detail, sizeof(detail), "got [");
    for (i = 0; i < count && used + 16u < sizeof(detail); i++) {
        used += (size_t)snprintf(detail + used, sizeof(detail) - used, "%s%d", i ? " " : "", ids[i]);
    }
    if (used + 16u < sizeof(detail)) {
        used += (size_t)snprintf(detail + used, sizeof(detail) - used, "], expected [");
    }
    for (i = 0; i < expected_count && used + 16u < sizeof(detail); i++) {
        used += (size_t)snprintf(detail + used, sizeof(detail) - used, "%s%d", i ? " " : "",
                                 expected[i]);
    }
    if (used + 2u < sizeof(detail)) snprintf(detail + used, sizeof(detail) - used, "]");
    free(ids);
    return sp_test_fail(error, error_capacity, check, detail);
}

/* Builds a variant and asserts that opening it fails with a non-empty message. */
static int sp_test_reject(const sp_test_options *options, int truncate, const char *check,
                          char *error, size_t error_capacity) {
    sp_buf model;
    mynah_sp *sp = NULL;
    char local[256];
    size_t length;
    int opened;

    if (sp_test_build(&model, options) != 0) {
        sp_buf_release(&model);
        return sp_test_fail(error, error_capacity, check, "out of memory building the test model");
    }
    length = truncate ? model.length - 1u : model.length;
    local[0] = '\0';
    opened = mynah_sp_open_memory(model.data, length, &sp, local, sizeof(local));
    sp_buf_release(&model);
    if (opened == 0) {
        mynah_sp_close(sp);
        return sp_test_fail(error, error_capacity, check, "the model opened but should have been "
                                                          "rejected");
    }
    if (sp != NULL) {
        mynah_sp_close(sp);
        return sp_test_fail(error, error_capacity, check, "the handle was not NULL after a failure");
    }
    if (local[0] == '\0') {
        return sp_test_fail(error, error_capacity, check, "rejected without a message");
    }
    return 0;
}

int mynah_sp_self_test(char *error, size_t error_capacity) {
    sp_test_options options;
    sp_buf model;
    mynah_sp *sp = NULL;
    char local[256];
    const char *piece = NULL;
    size_t length = 0;
    int ids[8];
    size_t count = 0;
    int status = -1;

    if (error != NULL && error_capacity > 0) error[0] = '\0';

    sp_test_defaults(&options);
    memset(&model, 0, sizeof(model));
    if (sp_test_build(&model, &options) != 0) {
        sp_buf_release(&model);
        return sp_test_fail(error, error_capacity, "build", "out of memory building the test model");
    }
    if (mynah_sp_open_memory(model.data, model.length, &sp, local, sizeof(local)) != 0) {
        sp_buf_release(&model);
        return sp_test_fail(error, error_capacity, "open", local);
    }

    /* 1. Vocabulary shape, and the unk id read from the piece typed UNKNOWN
     *    rather than from TrainerSpec's unk_id, which the builder sets to 77. */
    if (mynah_sp_vocab_size(sp) != 2u + 256u + sizeof(sp_test_tail) / sizeof(sp_test_tail[0])) {
        sp_test_fail(error, error_capacity, "vocab size", "wrong piece count");
        goto done;
    }
    if (mynah_sp_unk_id(sp) != 0) {
        sp_test_fail(error, error_capacity, "unk id", "the unk id did not come from the piece type");
        goto done;
    }
    if (mynah_sp_piece(sp, 0, &piece, &length) != 0 || length != 5u ||
        memcmp(piece, "<unk>", 5) != 0) {
        sp_test_fail(error, error_capacity, "piece lookup", "piece 0 is not \"<unk>\"");
        goto done;
    }
    if (mynah_sp_piece(sp, (int)mynah_sp_vocab_size(sp), &piece, &length) == 0 ||
        mynah_sp_piece(sp, -1, &piece, &length) == 0) {
        sp_test_fail(error, error_capacity, "piece lookup", "an out-of-range id was accepted");
        goto done;
    }

    /* 2. The empty input short-circuits before the dummy prefix. */
    if (sp_test_expect(sp, "empty input", "", 0, NULL, 0, error, error_capacity) != 0) goto done;

    /* 3. A single space is the prefix plus one escaped space. */
    {
        const int expected[] = {SP_TEST_SPACE, SP_TEST_SPACE};
        if (sp_test_expect(sp, "single space", " ", 1, expected, 2, error, error_capacity) != 0) {
            goto done;
        }
    }

    /* 4. remove_extra_whitespaces is off, so a double space is not collapsed. */
    {
        const int expected[] = {SP_TEST_SPACE_A, SP_TEST_SPACE, SP_TEST_SPACE, SP_TEST_B};
        if (sp_test_expect(sp, "no whitespace collapsing", "a  b", 4, expected, 4, error,
                           error_capacity) != 0) {
            goto done;
        }
    }

    /* 5. Only 0x20 is escaped: a tab is not whitespace here and falls through
     *    to byte fallback. */
    {
        const int expected[] = {SP_TEST_SPACE, SP_TEST_BYTE(0x09)};
        if (sp_test_expect(sp, "tab is not escaped", "\t", 1, expected, 2, error,
                           error_capacity) != 0) {
            goto done;
        }
    }

    /* 6. CONTROL, UNKNOWN and BYTE pieces are not segmentable, so "<s>" typed
     *    by a user cannot reach the reserved id. */
    {
        const int expected[] = {SP_TEST_SPACE, SP_TEST_LT, SP_TEST_S, SP_TEST_GT};
        if (sp_test_expect(sp, "control pieces are unreachable", "<s>", 3, expected, 4, error,
                           error_capacity) != 0) {
            goto done;
        }
    }

    /* 7. An unknown node spans one character, and byte fallback expands the
     *    bytes of the normalized buffer: 0xFF becomes U+FFFD, three ids, and
     *    ED A0 80 becomes three U+FFFD, nine ids. */
    {
        const int expected[] = {SP_TEST_SPACE, SP_TEST_BYTE(0xEF), SP_TEST_BYTE(0xBF),
                                SP_TEST_BYTE(0xBD)};
        if (sp_test_expect(sp, "invalid byte becomes U+FFFD", "\xFF", 1, expected, 4, error,
                           error_capacity) != 0) {
            goto done;
        }
    }
    {
        const int expected[] = {SP_TEST_SPACE,
                                SP_TEST_BYTE(0xEF), SP_TEST_BYTE(0xBF), SP_TEST_BYTE(0xBD),
                                SP_TEST_BYTE(0xEF), SP_TEST_BYTE(0xBF), SP_TEST_BYTE(0xBD),
                                SP_TEST_BYTE(0xEF), SP_TEST_BYTE(0xBF), SP_TEST_BYTE(0xBD)};
        if (sp_test_expect(sp, "surrogate is three replacements", "\xED\xA0\x80", 3, expected, 10,
                           error, error_capacity) != 0) {
            goto done;
        }
    }

    /* 8. A USER_DEFINED piece scores nchar * 10, not its stored score, which
     *    the builder sets to -1000 so the two choices diverge. */
    {
        const int expected[] = {SP_TEST_SPACE, SP_TEST_UD};
        if (sp_test_expect(sp, "user-defined penalty", "[UD]", 4, expected, 2, error,
                           error_capacity) != 0) {
            goto done;
        }
    }

    /* 9. An UNUSED piece is skipped and does not count as a single node, so "q"
     *    has to fall through to byte fallback. */
    {
        const int expected[] = {SP_TEST_SPACE, SP_TEST_BYTE('q')};
        if (sp_test_expect(sp, "unused piece is skipped", "q", 1, expected, 2, error,
                           error_capacity) != 0) {
            goto done;
        }
    }

    /* 10. unk_score is min(NORMAL) - 10. With the minimum taken over every
     *     piece instead, two unknown characters would beat the "QR" piece. */
    {
        const int expected[] = {SP_TEST_SPACE, SP_TEST_QR};
        if (sp_test_expect(sp, "unk score over NORMAL only", "QR", 2, expected, 2, error,
                           error_capacity) != 0) {
            goto done;
        }
    }

    /* 11. NUL is legal input, which is why the API takes a length. */
    {
        const int expected[] = {SP_TEST_SPACE_A, SP_TEST_BYTE(0x00), SP_TEST_B};
        if (sp_test_expect(sp, "NUL in the text", "a\0b", 3, expected, 3, error,
                           error_capacity) != 0) {
            goto done;
        }
    }

    /* A precomposed character passes through untouched: no NFC, no NFKC. */
    {
        const int expected[] = {SP_TEST_SPACE, SP_TEST_EACUTE};
        if (sp_test_expect(sp, "no unicode normalization", "\xC3\xA9", 2, expected, 2, error,
                           error_capacity) != 0) {
            goto done;
        }
    }

    /* 12. encode_into reports the required count and refuses to overflow. */
    count = 999;
    if (mynah_sp_encode_into(sp, "<s>", 3, ids, 2, &count, local, sizeof(local)) == 0) {
        sp_test_fail(error, error_capacity, "encode_into overflow", "a short buffer was accepted");
        goto done;
    }
    if (count != 4u) {
        sp_test_fail(error, error_capacity, "encode_into overflow",
                     "the required count was not reported");
        goto done;
    }
    memset(ids, 0, sizeof(ids));
    if (mynah_sp_encode_into(sp, "<s>", 3, ids, 4, &count, local, sizeof(local)) != 0 ||
        count != 4u || ids[0] != SP_TEST_SPACE || ids[1] != SP_TEST_LT || ids[2] != SP_TEST_S ||
        ids[3] != SP_TEST_GT) {
        sp_test_fail(error, error_capacity, "encode_into", local[0] != '\0' ? local : "wrong ids");
        goto done;
    }

    /* A long input, to exercise the growth bounds under a sanitizer. */
    {
        const size_t n = 4096;
        char *big = malloc(n);
        int *big_ids = NULL;
        size_t big_count = 0;
        int ok;
        if (big == NULL) {
            sp_test_fail(error, error_capacity, "long input", "out of memory");
            goto done;
        }
        memset(big, 'a', n);
        ok = mynah_sp_encode(sp, big, n, &big_ids, &big_count, local, sizeof(local));
        free(big);
        if (ok != 0 || big_count == 0 || big_ids == NULL) {
            free(big_ids);
            sp_test_fail(error, error_capacity, "long input", local[0] != '\0' ? local : "no ids");
            goto done;
        }
        free(big_ids);
    }

    mynah_sp_close(sp);
    sp = NULL;
    sp_buf_release(&model);

    /* 13. The rejection paths. Each variant must fail with a message. */
    {
        struct {
            const char *check;
            int truncate;
        } cases[] = {
            {"reject truncated file", 1},
            {"reject non-unigram model_type", 0},
            {"reject a non-empty charsmap", 0},
            {"reject a foreign normalizer", 0},
            {"reject remove_extra_whitespaces", 0},
            {"reject a vocab_size mismatch", 0},
            {"reject a model with no UNKNOWN piece", 0},
            {"reject missing byte pieces", 0},
            {"reject a group wire type", 0},
            {"reject treat_whitespace_as_suffix", 0},
            {"reject a pretokenization delimiter", 0},
        };
        size_t i;
        for (i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
            sp_test_defaults(&options);
            switch (i) {
            case 1: options.model_type = 2; break;
            case 2: options.charsmap_length = 16; break;
            case 3: options.normalizer_name = "nmt_nfkc"; break;
            case 4: options.remove_extra_whitespaces = 1; break;
            case 5: options.vocab_size = 999; break;
            case 6: options.omit_unknown = 1; break;
            case 7: options.omit_byte_pieces = 1; break;
            case 8: options.inject_group = 1; break;
            case 9: options.treat_whitespace_as_suffix = 1; break;
            case 10: options.pretokenization_delimiter = " "; break;
            default: break;
            }
            if (sp_test_reject(&options, cases[i].truncate, cases[i].check, error,
                               error_capacity) != 0) {
                return -1;
            }
        }
        if (mynah_sp_open_memory(NULL, 0, &sp, local, sizeof(local)) == 0 || sp != NULL) {
            mynah_sp_close(sp);
            return sp_test_fail(error, error_capacity, "reject an empty buffer",
                                "the empty buffer opened");
        }
    }
    return 0;

done:
    mynah_sp_close(sp);
    sp_buf_release(&model);
    return status;
}
