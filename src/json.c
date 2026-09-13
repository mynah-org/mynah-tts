/* The one JSON reader. See src/json.h for the decided behaviour and why each
 * choice is the one it is.
 *
 * Shape of the implementation: ONE scanner, used for everything. Validating a
 * document, finding the end of a value so a lookup can step over it, and
 * decoding a string are the same code walking the same bytes with a different
 * destination. There is no second path that could disagree with the first --
 * which is exactly how the previous readers went wrong: a `strstr` that found
 * keys and a separate unescaper that read values had no shared idea of where a
 * string begins and ends, so one of them could be inside a value the other
 * thought was structure.
 *
 * The scanner is ITERATIVE. `scan_value` holds one byte of state per open
 * container in a fixed array of MYNAH_JSON_MAX_DEPTH bytes and loops; it never
 * calls itself. A recursive-descent parser on a network-facing socket is a
 * stack overflow with a `[[[[[[...` body, and no amount of care inside the
 * recursion fixes that.
 *
 * LOCALE: strtod reads the decimal separator from LC_NUMERIC, so in a de_DE
 * locale "1.5" would convert to 1. This runtime never calls setlocale, so the
 * C locale is in force for the whole process and '.' is the separator -- the
 * same assumption the code this replaces made silently.
 */
#include "json.h"

#include <limits.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    const char *text;
    size_t len;
    size_t pos;
    mynah_json_error *err;     /* NULL when the caller wants no message */
} jscan;

/* ------------------------------------------------------------- diagnostics */

/* Every refusal goes through here, so every refusal has an offset. */
#if defined(__GNUC__)
__attribute__((format(printf, 3, 4)))
#endif
static int fail(jscan *s, size_t offset, const char *format, ...) {
    if (s->err != NULL) {
        s->err->offset = offset;
        va_list args;
        va_start(args, format);
        vsnprintf(s->err->message, sizeof(s->err->message), format, args);
        va_end(args);
    }
    return -1;
}

/* "'x'" for something a reader can see, "byte 0x0a" for something it cannot.
 * A message that says `expected ',' or '}'` without saying what was actually
 * there sends the reader back to count bytes by hand. */
static void describe(const char *text, size_t len, size_t pos, char out[16]) {
    if (pos >= len) {
        snprintf(out, 16, "end of input");
        return;
    }
    const unsigned char c = (unsigned char)text[pos];
    if (c >= 0x20u && c < 0x7Fu) snprintf(out, 16, "'%c'", (char)c);
    else snprintf(out, 16, "byte 0x%02x", (unsigned)c);
}

/* ---------------------------------------------------------------- lexemes */

static void skip_ws(jscan *s) {
    while (s->pos < s->len) {
        const char c = s->text[s->pos];
        if (c != ' ' && c != '\t' && c != '\n' && c != '\r') break;
        ++s->pos;
    }
}

/* Length and validity of one raw UTF-8 sequence.
 *
 * The ranges are the ones that make this a validator rather than a length
 * table: 0xC0/0xC1 and the E0/F0 lower bounds reject overlong encodings (the
 * classic way to smuggle a '/' or a '.' past a filter), ED caps at 0x9F so a
 * raw surrogate cannot arrive as CESU-8, and F4 caps at 0x8F so nothing above
 * U+10FFFF gets in. */
static int utf8_sequence(const unsigned char *p, size_t avail, size_t *out_len) {
    const unsigned char b0 = p[0];
    if (b0 < 0x80u) { *out_len = 1u; return 0; }
    size_t need;
    unsigned lo, hi;
    if (b0 >= 0xC2u && b0 <= 0xDFu)      { need = 2u; lo = 0x80u; hi = 0xBFu; }
    else if (b0 == 0xE0u)                { need = 3u; lo = 0xA0u; hi = 0xBFu; }
    else if (b0 >= 0xE1u && b0 <= 0xECu) { need = 3u; lo = 0x80u; hi = 0xBFu; }
    else if (b0 == 0xEDu)                { need = 3u; lo = 0x80u; hi = 0x9Fu; }
    else if (b0 >= 0xEEu && b0 <= 0xEFu) { need = 3u; lo = 0x80u; hi = 0xBFu; }
    else if (b0 == 0xF0u)                { need = 4u; lo = 0x90u; hi = 0xBFu; }
    else if (b0 >= 0xF1u && b0 <= 0xF3u) { need = 4u; lo = 0x80u; hi = 0xBFu; }
    else if (b0 == 0xF4u)                { need = 4u; lo = 0x80u; hi = 0x8Fu; }
    else return -1;
    if (avail < need) return -1;
    if (p[1] < lo || p[1] > hi) return -1;
    for (size_t i = 2u; i < need; ++i) {
        if ((p[i] & 0xC0u) != 0x80u) return -1;
    }
    *out_len = need;
    return 0;
}

static size_t utf8_encode(unsigned cp, char *out) {
    if (cp < 0x80u) {
        out[0] = (char)cp;
        return 1u;
    }
    if (cp < 0x800u) {
        out[0] = (char)(0xC0u | (cp >> 6));
        out[1] = (char)(0x80u | (cp & 0x3Fu));
        return 2u;
    }
    if (cp < 0x10000u) {
        out[0] = (char)(0xE0u | (cp >> 12));
        out[1] = (char)(0x80u | ((cp >> 6) & 0x3Fu));
        out[2] = (char)(0x80u | (cp & 0x3Fu));
        return 3u;
    }
    out[0] = (char)(0xF0u | (cp >> 18));
    out[1] = (char)(0x80u | ((cp >> 12) & 0x3Fu));
    out[2] = (char)(0x80u | ((cp >> 6) & 0x3Fu));
    out[3] = (char)(0x80u | (cp & 0x3Fu));
    return 4u;
}

static int hex4(const char *p, unsigned *out) {
    unsigned value = 0;
    for (int i = 0; i < 4; ++i) {
        const char c = p[i];
        unsigned digit;
        if (c >= '0' && c <= '9')      digit = (unsigned)(c - '0');
        else if (c >= 'a' && c <= 'f') digit = (unsigned)(c - 'a' + 10);
        else if (c >= 'A' && c <= 'F') digit = (unsigned)(c - 'A' + 10);
        else return -1;
        value = (value << 4) | digit;
    }
    *out = value;
    return 0;
}

/* One escape sequence, starting at the backslash. Appends to `out` when
 * decoding. Advances s->pos past the whole escape, both \uXXXX halves of a
 * surrogate pair included. */
static int scan_escape(jscan *s, char *out, size_t capacity, size_t *used) {
    const size_t esc = s->pos;
    ++s->pos;                                  /* the backslash */
    if (s->pos >= s->len) {
        return fail(s, s->pos, "unterminated escape: expected an escape character");
    }
    const char c = s->text[s->pos];
    char plain = 0;
    switch (c) {
        case '"':  plain = '"';  break;
        case '\\': plain = '\\'; break;
        case '/':  plain = '/';  break;
        case 'b':  plain = '\b'; break;
        case 'f':  plain = '\f'; break;
        case 'n':  plain = '\n'; break;
        case 'r':  plain = '\r'; break;
        case 't':  plain = '\t'; break;
        case 'u':  break;
        default: {
            char seen[16];
            describe(s->text, s->len, s->pos, seen);
            return fail(s, s->pos,
                        "invalid escape \\%s: expected one of \" \\ / b f n r t u",
                        seen);
        }
    }
    if (c != 'u') {
        ++s->pos;
        if (out != NULL) {
            if (*used + 2u > capacity) {
                return fail(s, esc, "decoded string does not fit in %zu bytes",
                            capacity);
            }
            out[(*used)++] = plain;
        }
        return 0;
    }

    ++s->pos;                                  /* the 'u' */
    if (s->len - s->pos < 4u) {
        return fail(s, s->pos, "truncated \\u escape: expected 4 hex digits");
    }
    unsigned cp = 0;
    if (hex4(s->text + s->pos, &cp) != 0) {
        return fail(s, s->pos, "invalid \\u escape: expected 4 hex digits");
    }
    s->pos += 4u;

    /* Surrogate pairs. The old server reader refused every surrogate, which
     * put every emoji and everything outside the BMP out of reach of the
     * tokenizer. A pair is one codepoint and is assembled as one. */
    if (cp >= 0xD800u && cp <= 0xDBFFu) {
        if (s->len - s->pos < 2u || s->text[s->pos] != '\\' ||
            s->text[s->pos + 1u] != 'u') {
            return fail(s, s->pos,
                        "lone high surrogate \\u%04X: expected a low surrogate "
                        "\\uDC00-\\uDFFF to follow", cp);
        }
        unsigned low = 0;
        if (s->len - (s->pos + 2u) < 4u ||
            hex4(s->text + s->pos + 2u, &low) != 0) {
            return fail(s, s->pos + 2u,
                        "invalid low surrogate after \\u%04X: expected 4 hex digits",
                        cp);
        }
        if (low < 0xDC00u || low > 0xDFFFu) {
            return fail(s, s->pos + 2u,
                        "\\u%04X after high surrogate \\u%04X: expected "
                        "\\uDC00-\\uDFFF", low, cp);
        }
        s->pos += 6u;
        cp = 0x10000u + ((cp - 0xD800u) << 10) + (low - 0xDC00u);
    } else if (cp >= 0xDC00u && cp <= 0xDFFFu) {
        return fail(s, esc,
                    "lone low surrogate \\u%04X: expected a high surrogate "
                    "\\uD800-\\uDBFF before it", cp);
    }

    if (out != NULL) {
        /* A NUL inside a string is legal JSON and cannot live in a C string.
         * Truncating there would hand a tokenizer a silent prefix of what the
         * caller sent, so the decode refuses and says why. The scanner still
         * accepts the document: this is a limit of the destination, not of the
         * grammar. */
        if (cp == 0u) {
            return fail(s, esc,
                        "\\u0000 cannot be carried in a NUL-terminated string");
        }
        char buf[4];
        const size_t n = utf8_encode(cp, buf);
        if (*used + n + 1u > capacity) {
            return fail(s, esc, "decoded string does not fit in %zu bytes",
                        capacity);
        }
        memcpy(out + *used, buf, n);
        *used += n;
    }
    return 0;
}

/* A string starting at the opening quote. `out == NULL` validates without
 * writing. On success s->pos is one past the closing quote and, when decoding,
 * `out` is NUL-terminated. */
static int scan_string(jscan *s, char *out, size_t capacity, size_t *out_len) {
    const size_t open = s->pos;
    if (s->pos >= s->len || s->text[s->pos] != '"') {
        char seen[16];
        describe(s->text, s->len, s->pos, seen);
        return fail(s, s->pos, "expected '\"' to start a string, found %s", seen);
    }
    ++s->pos;
    size_t used = 0;
    for (;;) {
        if (s->pos >= s->len) {
            return fail(s, s->len,
                        "unterminated string: expected '\"' before end of input "
                        "(string opened at byte %zu)", open);
        }
        const unsigned char c = (unsigned char)s->text[s->pos];
        if (c == '"') {
            ++s->pos;
            break;
        }
        if (c == '\\') {
            if (scan_escape(s, out, capacity, &used) != 0) return -1;
            continue;
        }
        if (c < 0x20u) {
            return fail(s, s->pos,
                        "unescaped control character U+%04X in string: expected "
                        "\\u%04X", (unsigned)c, (unsigned)c);
        }
        size_t seq = 0;
        if (utf8_sequence((const unsigned char *)s->text + s->pos,
                          s->len - s->pos, &seq) != 0) {
            return fail(s, s->pos,
                        "invalid UTF-8 in string at byte 0x%02x: expected a "
                        "well-formed UTF-8 sequence", (unsigned)c);
        }
        if (out != NULL) {
            if (used + seq + 1u > capacity) {
                return fail(s, s->pos, "decoded string does not fit in %zu bytes",
                            capacity);
            }
            memcpy(out + used, s->text + s->pos, seq);
            used += seq;
        }
        s->pos += seq;
    }
    if (out != NULL) {
        if (used + 1u > capacity) {
            return fail(s, s->pos, "decoded string does not fit in %zu bytes",
                        capacity);
        }
        out[used] = '\0';
    }
    if (out_len != NULL) *out_len = used;
    return 0;
}

/* RFC 8259 number: -?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?
 *
 * Written out rather than delegated to strtod, because strtod also accepts
 * "0x10", "INF", "nan", a leading '+' and a leading space -- none of which are
 * JSON, and every one of which the readers this replaces would have taken. */
static int scan_number(jscan *s) {
    if (s->pos < s->len && s->text[s->pos] == '-') ++s->pos;
    if (s->pos >= s->len) {
        return fail(s, s->pos, "expected a digit after '-'");
    }
    if (s->text[s->pos] == '0') {
        ++s->pos;
        if (s->pos < s->len && s->text[s->pos] >= '0' && s->text[s->pos] <= '9') {
            return fail(s, s->pos,
                        "leading zero in number: expected '.', an exponent, or "
                        "the end of the number");
        }
    } else if (s->text[s->pos] >= '1' && s->text[s->pos] <= '9') {
        while (s->pos < s->len && s->text[s->pos] >= '0' && s->text[s->pos] <= '9') {
            ++s->pos;
        }
    } else {
        char seen[16];
        describe(s->text, s->len, s->pos, seen);
        return fail(s, s->pos, "expected a digit, found %s", seen);
    }
    if (s->pos < s->len && s->text[s->pos] == '.') {
        ++s->pos;
        if (s->pos >= s->len || s->text[s->pos] < '0' || s->text[s->pos] > '9') {
            char seen[16];
            describe(s->text, s->len, s->pos, seen);
            return fail(s, s->pos, "expected a digit after '.', found %s", seen);
        }
        while (s->pos < s->len && s->text[s->pos] >= '0' && s->text[s->pos] <= '9') {
            ++s->pos;
        }
    }
    if (s->pos < s->len && (s->text[s->pos] == 'e' || s->text[s->pos] == 'E')) {
        ++s->pos;
        if (s->pos < s->len && (s->text[s->pos] == '+' || s->text[s->pos] == '-')) {
            ++s->pos;
        }
        if (s->pos >= s->len || s->text[s->pos] < '0' || s->text[s->pos] > '9') {
            char seen[16];
            describe(s->text, s->len, s->pos, seen);
            return fail(s, s->pos, "expected a digit in the exponent, found %s", seen);
        }
        while (s->pos < s->len && s->text[s->pos] >= '0' && s->text[s->pos] <= '9') {
            ++s->pos;
        }
    }
    return 0;
}

static int scan_literal(jscan *s, const char *word, mynah_json_type type,
                        mynah_json_type *out_type) {
    const size_t n = strlen(word);
    if (s->len - s->pos < n || memcmp(s->text + s->pos, word, n) != 0) {
        char seen[16];
        describe(s->text, s->len, s->pos, seen);
        return fail(s, s->pos, "expected '%s', found %s", word, seen);
    }
    s->pos += n;
    *out_type = type;
    return 0;
}

/* A scalar: string, number, true, false or null. */
static int scan_scalar(jscan *s, mynah_json_type *out_type) {
    const char c = s->text[s->pos];
    if (c == '"') {
        if (scan_string(s, NULL, 0, NULL) != 0) return -1;
        *out_type = MYNAH_JSON_STRING;
        return 0;
    }
    if (c == 't') return scan_literal(s, "true",  MYNAH_JSON_BOOL, out_type);
    if (c == 'f') return scan_literal(s, "false", MYNAH_JSON_BOOL, out_type);
    if (c == 'n') return scan_literal(s, "null",  MYNAH_JSON_NULL, out_type);
    if (c == '-' || (c >= '0' && c <= '9')) {
        if (scan_number(s) != 0) return -1;
        *out_type = MYNAH_JSON_NUMBER;
        return 0;
    }
    char seen[16];
    describe(s->text, s->len, s->pos, seen);
    return fail(s, s->pos, "expected a value, found %s", seen);
}

/* An object member's name and the colon after it. */
static int scan_member_name(jscan *s) {
    skip_ws(s);
    if (s->pos >= s->len || s->text[s->pos] != '"') {
        char seen[16];
        describe(s->text, s->len, s->pos, seen);
        return fail(s, s->pos,
                    "expected '\"' to start a member name, found %s", seen);
    }
    if (scan_string(s, NULL, 0, NULL) != 0) return -1;
    skip_ws(s);
    if (s->pos >= s->len || s->text[s->pos] != ':') {
        char seen[16];
        describe(s->text, s->len, s->pos, seen);
        return fail(s, s->pos, "expected ':' after a member name, found %s", seen);
    }
    ++s->pos;
    return 0;
}

/* Scan exactly one complete value, leaving s->pos one past its last byte.
 *
 * Iterative, with one byte per open container. `depth` is checked BEFORE the
 * push, so the refusal names the byte that would have crossed the bound. */
static int scan_value(jscan *s, mynah_json_type *out_type) {
    char stack[MYNAH_JSON_MAX_DEPTH];
    size_t depth = 0;
    mynah_json_type root_type = MYNAH_JSON_NULL;
    int have_root_type = 0;

    for (;;) {
        skip_ws(s);
        if (s->pos >= s->len) {
            return fail(s, s->pos, "expected a value, found end of input");
        }
        const char c = s->text[s->pos];
        mynah_json_type type;
        if (c == '{' || c == '[') {
            type = (c == '{') ? MYNAH_JSON_OBJECT : MYNAH_JSON_ARRAY;
            if (!have_root_type) { root_type = type; have_root_type = 1; }
            if (depth == MYNAH_JSON_MAX_DEPTH) {
                return fail(s, s->pos,
                            "maximum nesting depth %d exceeded: expected a "
                            "shallower document", MYNAH_JSON_MAX_DEPTH);
            }
            stack[depth++] = c;
            ++s->pos;
            skip_ws(s);
            const char close = (c == '{') ? '}' : ']';
            if (s->pos < s->len && s->text[s->pos] == close) {
                ++s->pos;                      /* empty container */
                --depth;
            } else {
                if (c == '{' && scan_member_name(s) != 0) return -1;
                continue;                      /* now read its first value */
            }
        } else {
            if (scan_scalar(s, &type) != 0) return -1;
            if (!have_root_type) { root_type = type; have_root_type = 1; }
        }

        /* A complete value just ended. Close as many containers as end here,
         * or take the comma that starts the next member/element. */
        for (;;) {
            if (depth == 0) {
                if (out_type != NULL) *out_type = root_type;
                return 0;
            }
            const char open = stack[depth - 1u];
            skip_ws(s);
            if (s->pos >= s->len) {
                return fail(s, s->pos, "expected ',' or '%c', found end of input",
                            open == '{' ? '}' : ']');
            }
            const char d = s->text[s->pos];
            if (d == ',') {
                ++s->pos;
                skip_ws(s);
                if (s->pos < s->len &&
                    (s->text[s->pos] == '}' || s->text[s->pos] == ']')) {
                    return fail(s, s->pos,
                                "trailing comma: expected a value after ','");
                }
                if (open == '{' && scan_member_name(s) != 0) return -1;
                break;                         /* read the next value */
            }
            if (d == '}' && open == '{') { ++s->pos; --depth; continue; }
            if (d == ']' && open == '[') { ++s->pos; --depth; continue; }
            char seen[16];
            describe(s->text, s->len, s->pos, seen);
            return fail(s, s->pos, "expected ',' or '%c', found %s",
                        open == '{' ? '}' : ']', seen);
        }
    }
}

/* ------------------------------------------------------------------- API */

int mynah_json_parse(const char *text, size_t length,
                     mynah_json_value *out_root, mynah_json_error *error) {
    if (error != NULL) {
        error->offset = 0;
        error->message[0] = '\0';
    }
    jscan s;
    s.text = text;
    s.len = length;
    s.pos = 0;
    s.err = error;
    if (text == NULL || out_root == NULL) {
        return fail(&s, 0, "no document");
    }
    skip_ws(&s);
    if (s.pos >= s.len) {
        return fail(&s, s.pos, "empty document: expected a value");
    }
    const size_t start = s.pos;
    mynah_json_type type = MYNAH_JSON_NULL;
    if (scan_value(&s, &type) != 0) return -1;
    const size_t end = s.pos;
    skip_ws(&s);
    if (s.pos < s.len) {
        char seen[16];
        describe(s.text, s.len, s.pos, seen);
        return fail(&s, s.pos,
                    "trailing content after the document: expected end of input, "
                    "found %s", seen);
    }
    out_root->text = text;
    out_root->doc_length = length;
    out_root->start = start;
    out_root->end = end;
    out_root->type = type;
    return 0;
}

/* A scanner over one already-validated value. No error sink: everything these
 * walks can hit was proved impossible by mynah_json_parse. */
static void subscan(jscan *s, const mynah_json_value *value, size_t pos) {
    s->text = value->text;
    s->len = value->end;          /* never read past the value that owns it */
    s->pos = pos;
    s->err = NULL;
}

static void fill(mynah_json_value *out, const mynah_json_value *owner,
                 size_t start, size_t end, mynah_json_type type) {
    out->text = owner->text;
    out->doc_length = owner->doc_length;
    out->start = start;
    out->end = end;
    out->type = type;
}

int mynah_json_object_next(const mynah_json_value *object, size_t *cursor,
                           mynah_json_value *out_name,
                           mynah_json_value *out_value) {
    if (object == NULL || cursor == NULL || out_value == NULL) return -1;
    if (object->type != MYNAH_JSON_OBJECT) return -1;
    jscan s;
    subscan(&s, object, *cursor == 0 ? object->start + 1u : *cursor);
    skip_ws(&s);
    if (s.pos >= s.len) return -1;
    if (s.text[s.pos] == '}') return -1;
    if (s.text[s.pos] == ',') {
        ++s.pos;
        skip_ws(&s);
    }
    const size_t name_start = s.pos;
    if (scan_string(&s, NULL, 0, NULL) != 0) return -1;
    const size_t name_end = s.pos;
    skip_ws(&s);
    if (s.pos >= s.len || s.text[s.pos] != ':') return -1;
    ++s.pos;
    skip_ws(&s);
    const size_t value_start = s.pos;
    mynah_json_type type = MYNAH_JSON_NULL;
    if (scan_value(&s, &type) != 0) return -1;
    if (out_name != NULL) {
        fill(out_name, object, name_start, name_end, MYNAH_JSON_STRING);
    }
    fill(out_value, object, value_start, s.pos, type);
    *cursor = s.pos;
    return 0;
}

/* Does the member-name span text[start,end) decode to exactly `name`?
 *
 * The common case has no escape, and then the answer is a memcmp against the
 * raw bytes -- no buffer, no decode. Only an escaped name pays for a decode,
 * and a name longer than the buffer simply cannot equal a name that fits in
 * it, which is why running out of room is a mismatch rather than an error. */
static int name_equals(const mynah_json_value *object, size_t start, size_t end,
                       const char *name) {
    const char *text = object->text;
    const size_t raw = end - start;            /* includes both quotes */
    if (raw < 2u) return 0;
    const char *body = text + start + 1u;
    const size_t body_len = raw - 2u;
    if (memchr(body, '\\', body_len) == NULL) {
        const size_t want = strlen(name);
        return body_len == want && memcmp(body, name, want) == 0;
    }
    char decoded[MYNAH_JSON_NAME_MAX];
    jscan s;
    s.text = text;
    s.len = end;
    s.pos = start;
    s.err = NULL;
    if (scan_string(&s, decoded, sizeof(decoded), NULL) != 0) return 0;
    return strcmp(decoded, name) == 0;
}

int mynah_json_object_get(const mynah_json_value *object, const char *name,
                          mynah_json_value *out) {
    if (object == NULL || name == NULL || out == NULL) return -1;
    if (object->type != MYNAH_JSON_OBJECT) return -1;
    size_t cursor = 0;
    mynah_json_value key, value;
    while (mynah_json_object_next(object, &cursor, &key, &value) == 0) {
        /* FIRST MATCH WINS -- see src/json.h on duplicate keys. */
        if (name_equals(object, key.start, key.end, name)) {
            *out = value;
            return 0;
        }
    }
    return -1;
}

int mynah_json_array_get(const mynah_json_value *array, size_t index,
                         mynah_json_value *out) {
    if (array == NULL || out == NULL || array->type != MYNAH_JSON_ARRAY) return -1;
    jscan s;
    subscan(&s, array, array->start + 1u);
    for (size_t i = 0;; ++i) {
        skip_ws(&s);
        if (s.pos >= s.len || s.text[s.pos] == ']') return -1;
        if (i > 0) {
            if (s.text[s.pos] != ',') return -1;
            ++s.pos;
            skip_ws(&s);
        }
        const size_t start = s.pos;
        mynah_json_type type = MYNAH_JSON_NULL;
        if (scan_value(&s, &type) != 0) return -1;
        if (i == index) {
            fill(out, array, start, s.pos, type);
            return 0;
        }
    }
}

int mynah_json_count(const mynah_json_value *value, size_t *out_count) {
    if (value == NULL || out_count == NULL) return -1;
    if (value->type == MYNAH_JSON_OBJECT) {
        size_t cursor = 0, n = 0;
        mynah_json_value member;
        while (mynah_json_object_next(value, &cursor, NULL, &member) == 0) ++n;
        *out_count = n;
        return 0;
    }
    if (value->type != MYNAH_JSON_ARRAY) return -1;
    jscan s;
    subscan(&s, value, value->start + 1u);
    size_t n = 0;
    for (;;) {
        skip_ws(&s);
        if (s.pos >= s.len || s.text[s.pos] == ']') {
            *out_count = n;
            return 0;
        }
        if (n > 0) {
            if (s.text[s.pos] != ',') return -1;
            ++s.pos;
            skip_ws(&s);
        }
        mynah_json_type type = MYNAH_JSON_NULL;
        if (scan_value(&s, &type) != 0) return -1;
        ++n;
    }
}

int mynah_json_lookup(const mynah_json_value *root, const char *path,
                      mynah_json_value *out) {
    if (root == NULL || path == NULL || out == NULL) return -1;
    mynah_json_value here = *root;
    const char *p = path;
    if (*p == '\0') {
        *out = here;
        return 0;
    }
    /* A leading '.' is an empty first segment, i.e. a request for the member
     * named "". It used to be skipped, so ".codec" quietly resolved to "codec"
     * -- a path nobody meant is a lookup nobody can audit. Caught by
     * tests/test_json.c on the suite's first run. */
    if (*p == '.') return -1;
    for (;;) {
        size_t n = 0;
        while (p[n] != '\0' && p[n] != '.' && p[n] != '[') ++n;
        if (n > 0) {
            if (n >= MYNAH_JSON_NAME_MAX) return -1;
            char segment[MYNAH_JSON_NAME_MAX];
            memcpy(segment, p, n);
            segment[n] = '\0';
            mynah_json_value next;
            if (mynah_json_object_get(&here, segment, &next) != 0) return -1;
            here = next;
            p += n;
        }
        while (*p == '[') {
            ++p;
            if (*p < '0' || *p > '9') return -1;
            size_t index = 0;
            while (*p >= '0' && *p <= '9') {
                if (index > (size_t)-1 / 10u) return -1;
                index = index * 10u + (size_t)(*p - '0');
                ++p;
            }
            if (*p != ']') return -1;
            ++p;
            mynah_json_value next;
            if (mynah_json_array_get(&here, index, &next) != 0) return -1;
            here = next;
        }
        if (*p == '\0') {
            *out = here;
            return 0;
        }
        if (*p != '.') return -1;
        ++p;
        if (*p == '\0' || *p == '.' || *p == '[') return -1;
    }
}

int mynah_json_as_string(const mynah_json_value *value, char *out,
                         size_t capacity) {
    if (value == NULL || out == NULL || capacity == 0) return -1;
    if (value->type != MYNAH_JSON_STRING) return -1;
    jscan s;
    s.text = value->text;
    s.len = value->end;
    s.pos = value->start;
    s.err = NULL;
    return scan_string(&s, out, capacity, NULL);
}

int mynah_json_as_number(const mynah_json_value *value, double *out) {
    if (value == NULL || out == NULL) return -1;
    if (value->type != MYNAH_JSON_NUMBER) return -1;
    const size_t n = value->end - value->start;
    /* The span is a validated JSON number, so the only thing that does not fit
     * in this buffer is a literal with hundreds of digits -- refused rather
     * than truncated, because a truncated number converts to a plausible wrong
     * answer instead of an error. */
    char buf[512];
    if (n >= sizeof(buf)) return -1;
    memcpy(buf, value->text + value->start, n);
    buf[n] = '\0';
    char *end = NULL;
    const double parsed = strtod(buf, &end);
    if (end != buf + n) return -1;
    /* 1e999 converts to infinity with ERANGE. Downstream this is a sample rate
     * or a temperature; an infinity there poisons every float it touches, so it
     * is refused here where the offending literal is still in hand. Underflow
     * to zero or to a subnormal is accepted: that value is representable and
     * means what it says. */
    if (!isfinite(parsed)) return -1;
    *out = parsed;
    return 0;
}

int mynah_json_as_unsigned(const mynah_json_value *value, unsigned *out) {
    double number = 0.0;
    if (out == NULL || mynah_json_as_number(value, &number) != 0) return -1;
    if (!(number >= 0.0) || number > (double)UINT_MAX) return -1;
    if (number != floor(number)) return -1;
    *out = (unsigned)number;
    return 0;
}

int mynah_json_as_bool(const mynah_json_value *value, int *out) {
    if (value == NULL || out == NULL || value->type != MYNAH_JSON_BOOL) return -1;
    *out = value->text[value->start] == 't' ? 1 : 0;
    return 0;
}

int mynah_json_is_null(const mynah_json_value *value) {
    return value != NULL && value->type == MYNAH_JSON_NULL;
}
