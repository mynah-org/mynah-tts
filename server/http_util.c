/* glibc hides pthread_setname_np behind _GNU_SOURCE; without it the call
 * compiles as an implicit declaration and the build breaks under gcc 15's
 * default -Werror=implicit-function-declaration.  macOS declares it in
 * <pthread.h> unconditionally, which is why this only ever failed on Linux.
 * server/prefork.c already carries the same guard. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "http_util.h"

#include "json.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* The Linux cap, which is the tighter of the two and the one that fails the
 * call rather than truncating for us. */
#define MYNAH_THREAD_NAME_MAX 16u

void mynah_thread_set_name(const char *name) {
    if (name == NULL || name[0] == '\0') return;
    char buf[MYNAH_THREAD_NAME_MAX];
    snprintf(buf, sizeof(buf), "%s", name);
#if defined(__APPLE__)
    /* Darwin names the calling thread and takes no handle. */
    (void)pthread_setname_np(buf);
#elif defined(__linux__)
    (void)pthread_setname_np(pthread_self(), buf);
#else
    (void)buf;   /* no portable spelling: leave the thread unnamed */
#endif
}

const char *mynah_memmem(const char *hay, size_t hay_len,
                         const char *needle, size_t needle_len) {
    if (needle_len == 0) return hay;
    if (hay_len < needle_len) return NULL;
    for (size_t i = 0; i + needle_len <= hay_len; ++i) {
        if (hay[i] == needle[0] && memcmp(hay + i, needle, needle_len) == 0) {
            return hay + i;
        }
    }
    return NULL;
}

static int ascii_lower(int c) {
    return (c >= 'A' && c <= 'Z') ? c - 'A' + 'a' : c;
}

int mynah_http_header(const char *headers, size_t headers_len,
                      const char *name, char *out, size_t capacity) {
    if (headers == NULL || name == NULL || out == NULL || capacity == 0) return -1;
    const size_t name_len = strlen(name);
    size_t line = 0;
    while (line < headers_len) {
        size_t end = line;
        while (end < headers_len && headers[end] != '\n') ++end;
        size_t stop = end;
        if (stop > line && headers[stop - 1] == '\r') --stop;

        if (stop - line > name_len && headers[line + name_len] == ':') {
            size_t i = 0;
            while (i < name_len &&
                   ascii_lower((unsigned char)headers[line + i]) ==
                   ascii_lower((unsigned char)name[i])) ++i;
            if (i == name_len) {
                size_t v = line + name_len + 1u;
                while (v < stop && (headers[v] == ' ' || headers[v] == '\t')) ++v;
                size_t n = stop - v;
                if (n >= capacity) n = capacity - 1u;
                memcpy(out, headers + v, n);
                out[n] = '\0';
                return 0;
            }
        }
        line = end + 1u;
    }
    return -1;
}

/* ------------------------------------------------------------------- JSON
 *
 * These three are the shapes server/main.c has always called. What changed is
 * underneath: they used to find a key with strstr and then eat whitespace and a
 * colon, which meant a body could name its own keys inside a value it supplied.
 * `{"voice":"say \"input\": fake","input":"real"}` served the fake one. They
 * now PARSE the body -- src/json.c -- and answer only from a real top-level
 * member of a real JSON object.
 *
 * Three consequences worth stating, because each is a behaviour change:
 *
 *   A MALFORMED BODY ANSWERS NOTHING. Every lookup on it fails, rather than
 *   some keys working and others not depending on where the document broke.
 *   http_precheck() already refuses a body that does not start with '{'; a body
 *   that starts well and ends badly now gets the same answer instead of a
 *   half-reading.
 *
 *   A NESTED KEY IS NOT A TOP-LEVEL KEY. `{"options":{"input":"x"}}` has no
 *   top-level "input" and no longer pretends to.
 *
 *   SURROGATE PAIRS DECODE. The old reader refused every \uXXXX in the
 *   surrogate range, so no emoji and nothing outside the BMP could reach the
 *   tokenizer at all. "\ud83d\ude00" is now one codepoint, as it always was.
 *
 * Each call parses the body again. That is two linear passes over at most
 * MAX_BODY bytes per lookup and no allocation whatsoever; for a handful of
 * fields on a request body it is not worth a cache. A caller that wants the
 * body read once can parse it itself with mynah_json_parse() and use
 * mynah_json_object_get() -- which is also the only way to see WHERE a bad body
 * broke, since these three return nothing but success or failure. */

int mynah_json_string(const char *json, const char *key,
                      char *out, size_t capacity) {
    mynah_json_value root, value;
    if (json == NULL || key == NULL || out == NULL || capacity == 0) return -1;
    if (mynah_json_parse(json, strlen(json), &root, NULL) != 0) return -1;
    if (root.type != MYNAH_JSON_OBJECT) return -1;
    if (mynah_json_object_get(&root, key, &value) != 0) return -1;
    return mynah_json_as_string(&value, out, capacity);
}

int mynah_json_number(const char *json, const char *key, double *out) {
    mynah_json_value root, value;
    if (json == NULL || key == NULL || out == NULL) return -1;
    if (mynah_json_parse(json, strlen(json), &root, NULL) != 0) return -1;
    if (root.type != MYNAH_JSON_OBJECT) return -1;
    if (mynah_json_object_get(&root, key, &value) != 0) return -1;
    return mynah_json_as_number(&value, out);
}

int mynah_json_bool(const char *json, const char *key, int *out) {
    mynah_json_value root, value;
    if (json == NULL || key == NULL || out == NULL) return -1;
    if (mynah_json_parse(json, strlen(json), &root, NULL) != 0) return -1;
    if (root.type != MYNAH_JSON_OBJECT) return -1;
    if (mynah_json_object_get(&root, key, &value) != 0) return -1;
    return mynah_json_as_bool(&value, out);
}

size_t mynah_json_escape(const char *in, char *out, size_t capacity) {
    size_t n = 0;
    for (const unsigned char *p = (const unsigned char *)in; *p != '\0'; ++p) {
        const char *esc = NULL;
        char buf[7];
        switch (*p) {
            case '"':  esc = "\\\""; break;
            case '\\': esc = "\\\\"; break;
            case '\n': esc = "\\n";  break;
            case '\r': esc = "\\r";  break;
            case '\t': esc = "\\t";  break;
            default:
                if (*p < 0x20u) {
                    snprintf(buf, sizeof(buf), "\\u%04x", *p);
                    esc = buf;
                }
                break;
        }
        if (esc != NULL) {
            const size_t len = strlen(esc);
            if (n + len >= capacity) return (size_t)-1;
            memcpy(out + n, esc, len);
            n += len;
        } else {
            if (n + 1u >= capacity) return (size_t)-1;
            out[n++] = (char)*p;
        }
    }
    if (n >= capacity) return (size_t)-1;
    out[n] = '\0';
    return n;
}

/* A request line is exactly three tokens separated by single spaces. Anything
 * else -- a missing version, an embedded space, a token that does not fit -- is
 * rejected rather than guessed at: the caller answers 400 and closes, which is
 * the only safe reading of a line we do not understand. */
int mynah_http_request_line(const char *buf, size_t len,
                            char *method, size_t method_capacity,
                            char *path, size_t path_capacity) {
    if (buf == NULL || method == NULL || path == NULL ||
        method_capacity == 0 || path_capacity == 0) return -1;
    method[0] = '\0';
    path[0] = '\0';

    size_t line_end = 0;
    while (line_end < len && buf[line_end] != '\r' && buf[line_end] != '\n') ++line_end;
    if (line_end == len) return -1;          /* no terminator inside the buffer */

    size_t i = 0;
    while (i < line_end && buf[i] != ' ') ++i;
    if (i == 0 || i == line_end) return -1;
    if (i >= method_capacity) return -1;
    memcpy(method, buf, i);
    method[i] = '\0';

    const size_t target = i + 1u;
    size_t end = target;
    while (end < line_end && buf[end] != ' ') ++end;
    if (end == target) return -1;
    if (end == line_end) return -1;          /* no HTTP version: not a request we serve */

    /* Require the version token, and require it to be HTTP. A proxy-style
     * absolute target ("http://host/x") is legal HTTP but this server does not
     * serve one, so the path must be origin-form. */
    if (line_end - (end + 1u) < 5u || memcmp(buf + end + 1u, "HTTP/", 5) != 0) return -1;

    size_t stop = target;
    while (stop < end && buf[stop] != '?' && buf[stop] != '#') ++stop;
    const size_t path_len = stop - target;
    if (path_len == 0 || path_len >= path_capacity) return -1;
    if (buf[target] != '/') return -1;
    memcpy(path, buf + target, path_len);
    path[path_len] = '\0';
    return 0;
}

int mynah_http_media_type_is(const char *value, const char *media) {
    if (value == NULL || media == NULL) return 0;
    while (*value == ' ' || *value == '\t') ++value;
    size_t i = 0;
    while (media[i] != '\0') {
        if (ascii_lower((unsigned char)value[i]) != ascii_lower((unsigned char)media[i])) return 0;
        ++i;
    }
    const char c = value[i];
    return c == '\0' || c == ';' || c == ' ' || c == '\t' || c == '\r' || c == '\n';
}
