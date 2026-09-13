/* One JSON reader for the whole runtime.
 *
 * WHY THIS EXISTS. There were two readers and neither was a parser. Both found
 * a key with strstr("\"key\"") and then ate whitespace and a colon, which means
 * both of them answered a lookup with a match found ANYWHERE in the document:
 * inside a string value, inside a nested object, inside a comment-like blob of
 * text a caller pasted into "input". `{"voice":"say \"input\": fake",
 * "input":"real"}` gave the fake one. Neither could express `{"a":{"b":1}}`, so
 * the PocketTTS converter flattened 59 scalars to the top level to feed a
 * reader rather than the reader learning to read.
 *
 * WHAT THIS IS. A single-pass validating scanner. It tokenizes objects,
 * arrays, strings, numbers, `true`, `false` and `null` exactly as RFC 8259
 * describes them, refuses everything else, and names the byte offset and the
 * expectation when it refuses -- "invalid JSON" with no locus is not actionable
 * for whoever is holding the request.
 *
 * WHAT IT DOES NOT DO: allocate. There is no node arena and no document object;
 * a value is a (document, start, end, type) span into the caller's buffer, and
 * every operation is a bounded re-scan of a span. Nothing here calls malloc,
 * so nothing here can fail for want of memory and no caller has to free a
 * result. The cost is that a lookup is a scan rather than a hash probe, which
 * for a model.json or an OpenAI request body is not a cost worth a heap.
 *
 * DECIDED BEHAVIOUR, all of it tested in tests/test_json.c:
 *
 *   Depth is bounded at MYNAH_JSON_MAX_DEPTH and the bound is enforced by a
 *   counter, not by the C stack: the scanner is iterative, so a body nested a
 *   million deep is a clean error at the bound and never a crash. This parser
 *   faces the network.
 *
 *   Duplicate keys: THE FIRST ONE WINS. A lookup returns the first member with
 *   that name in document order and a duplicate is not an error. This is what
 *   the strstr readers did, so no pack and no client changes meaning; and it
 *   means a key appended to a body cannot override a key already in it.
 *
 *   Invalid UTF-8 is REJECTED, with the offset of the first bad byte. JSON text
 *   is UTF-8 by definition, the decoded string goes straight into a tokenizer,
 *   and a malformed sequence passed through would surface as nonsense audio or
 *   a tokenizer error at a place that says nothing about the cause. Overlong
 *   encodings, raw surrogates (CESU-8) and anything above U+10FFFF are refused
 *   on the same grounds. Unescaped control bytes below U+0020 in a string are
 *   refused because RFC 8259 forbids them.
 *
 *   \uXXXX surrogate PAIRS ARE REASSEMBLED: "\uD83D\uDE00" decodes to the four
 *   UTF-8 bytes of U+1F600, one codepoint. The old server reader rejected every
 *   surrogate outright, so no emoji and nothing outside the BMP could reach the
 *   tokenizer. A lone high surrogate, a lone low surrogate and a reversed pair
 *   are each refused with a message that says which.
 *
 *   Trailing content after the root value is an error. A document is one value
 *   and optional whitespace, nothing else.
 */
#ifndef MYNAH_JSON_H
#define MYNAH_JSON_H

#include <stddef.h>

/* Maximum container nesting. 32 is far past anything a model pack or an
 * OpenAI-shaped body has ever needed (the deepest real document in this repo
 * is 3) and far below anything that could trouble the scanner, which holds one
 * byte of state per level in a fixed array. Exceeding it is a normal parse
 * error naming the offset of the container that crossed the line. */
#define MYNAH_JSON_MAX_DEPTH 32

/* Longest member name a lookup can match. Names are compared in place when
 * they contain no escape, so this only bounds the decode buffer for an escaped
 * name; a longer name simply never matches. */
#define MYNAH_JSON_NAME_MAX 256

#define MYNAH_JSON_ERROR_MAX 160

typedef enum {
    MYNAH_JSON_NULL = 0,
    MYNAH_JSON_BOOL,
    MYNAH_JSON_NUMBER,
    MYNAH_JSON_STRING,
    MYNAH_JSON_ARRAY,
    MYNAH_JSON_OBJECT
} mynah_json_type;

typedef struct {
    size_t offset;                        /* byte offset where it broke */
    char message[MYNAH_JSON_ERROR_MAX];   /* what was expected there */
} mynah_json_error;

/* A span into the caller's document. `text` is NOT owned and must outlive
 * every value taken from it. `start` is the value's first byte and `end` is one
 * past its last, so a value is exactly text[start, end). */
typedef struct {
    const char *text;
    size_t doc_length;
    size_t start;
    size_t end;
    mynah_json_type type;
} mynah_json_value;

/* Validate a whole document and return its root value.
 *
 * The document is text[0, length); it need not be NUL-terminated and an
 * embedded NUL is simply a byte the grammar has no place for. `error` may be
 * NULL. On failure `out_root` is left untouched and, when `error` is given, it
 * carries the offset and the expectation. Returns 0 on success. */
int mynah_json_parse(const char *text, size_t length,
                     mynah_json_value *out_root, mynah_json_error *error);

/* Member of an OBJECT by exact name, first occurrence. `name` is a literal
 * key, never a path: a key containing '.' or '[' is reachable here and only
 * here. Returns 0 when found. */
int mynah_json_object_get(const mynah_json_value *object, const char *name,
                          mynah_json_value *out);

/* Element of an ARRAY by index. Returns 0 when the index exists. */
int mynah_json_array_get(const mynah_json_value *array, size_t index,
                         mynah_json_value *out);

/* Members of an OBJECT (duplicates counted separately) or elements of an
 * ARRAY. Returns -1 for a scalar. */
int mynah_json_count(const mynah_json_value *value, size_t *out_count);

/* Iterate an OBJECT's members in document order. Start with `*cursor = 0`;
 * each call fills `out_name` (a STRING value: the raw name span) and
 * `out_value` and advances the cursor. Returns 0 while members remain, -1 at
 * the end. `out_name` may be NULL. */
int mynah_json_object_next(const mynah_json_value *object, size_t *cursor,
                           mynah_json_value *out_name,
                           mynah_json_value *out_value);

/* Walk a dotted path from `root`: "codec.samples_per_frame", "codec_ratios[0]",
 * "a.b[2].c". The empty path is the root itself. Path syntax has no escape, so
 * a key that contains '.' or '[' is not reachable this way -- use
 * mynah_json_object_get for it. Returns 0 when the path resolves. */
int mynah_json_lookup(const mynah_json_value *root, const char *path,
                      mynah_json_value *out);

/* Decode a STRING value into `out` as NUL-terminated UTF-8, resolving every
 * escape and reassembling surrogate pairs. Returns -1 if the value is not a
 * string, if the decoded bytes plus the terminator do not fit, or if the string
 * contains \u0000 -- a NUL cannot be carried in a C string, and truncating
 * there silently would hand a tokenizer a prefix of what the caller sent. */
int mynah_json_as_string(const mynah_json_value *value, char *out,
                         size_t capacity);

/* NUMBER as a double. Refuses a value that is not finite (1e999), and refuses
 * a literal longer than 511 bytes rather than guessing at a truncation. */
int mynah_json_as_number(const mynah_json_value *value, double *out);

/* NUMBER that is a non-negative integer in [0, UINT_MAX]. 4.0 is accepted as 4;
 * 4.5, -1 and 1e400 are not. */
int mynah_json_as_unsigned(const mynah_json_value *value, unsigned *out);

/* `true` or `false` only. A 0/1 number is not a boolean here. */
int mynah_json_as_bool(const mynah_json_value *value, int *out);

/* 1 when the value is `null`. */
int mynah_json_is_null(const mynah_json_value *value);

#endif
