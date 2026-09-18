/* Small HTTP helpers for the mynah-tts server: header lookup, portable memmem,
 * request-line splitting, and the by-key JSON accessors the routes call.
 *
 * The accessors are a thin layer over src/json.h now. They used to be "just
 * enough JSON reading", which in practice meant strstr for "\"key\"" -- a
 * reader that could be told where its own keys were by the body it was reading.
 * The parser is one file over and it is smaller than the bugs were. */
#ifndef MYNAH_HTTP_UTIL_H
#define MYNAH_HTTP_UTIL_H

#include <stddef.h>

/* Portable memmem (small needles). NULL when absent. */
const char *mynah_memmem(const char *hay, size_t hay_len,
                         const char *needle, size_t needle_len);

/* Case-insensitive header lookup over a raw header block. Writes at most
 * capacity-1 bytes plus a terminator. Returns 0 when found. */
int mynah_http_header(const char *headers, size_t headers_len,
                      const char *name, char *out, size_t capacity);

/* Read a string by TOP-LEVEL key from a JSON object, decoding every escape and
 * reassembling surrogate pairs, so an emoji arrives as one codepoint. Returns 0
 * when the document parses, the key is a top-level member, its value is a
 * string, and the decoding fits in `capacity` including the terminator.
 *
 * A key inside a nested object or inside another key's VALUE is not this key.
 * A body that is not valid JSON answers nothing, here or below. */
int mynah_json_string(const char *json, const char *key,
                      char *out, size_t capacity);

/* Read a number by top-level key. Returns 0 when found; refuses a value that is
 * not a JSON number, and one that does not convert to a finite double. */
int mynah_json_number(const char *json, const char *key, double *out);

/* Read a boolean by top-level key: `true` or `false` only, never 0/1. */
int mynah_json_bool(const char *json, const char *key, int *out);

/* Splits a request line ("METHOD SP TARGET SP HTTP/1.1") into its method and
 * its path, with any query string or fragment removed. Returns 0 on success.
 *
 * This exists so routing can compare a whole path rather than a prefix of the
 * request line: a prefix test routes "POST /v1/audio/speechXYZ" to
 * /v1/audio/speech, and a route that matches by accident is a request nobody
 * meant to serve -- and, once the descriptor changes owner, one nobody
 * closes. */
int mynah_http_request_line(const char *buf, size_t len,
                            char *method, size_t method_capacity,
                            char *path, size_t path_capacity);

/* Compares the media type of a header value ("application/json; charset=utf-8")
 * against `media`, case-insensitively and ignoring parameters. Returns 1 on a
 * match. */
int mynah_http_media_type_is(const char *value, const char *media);

/* Escape a UTF-8 string into a JSON string body (no surrounding quotes).
 * Returns the number of bytes written, or (size_t)-1 if it does not fit. */
size_t mynah_json_escape(const char *in, char *out, size_t capacity);

/* Names the CALLING thread for the OS, so `top -H`, `ps -M` and /proc/<pid>/task
 * show the worker's thread-ownership table without a debugger attached.
 *
 * It must be called from the thread being named, because the two platform
 * signatures disagree and only that form is available on both:
 *
 *   macOS   int pthread_setname_np(const char *)                self only
 *   Linux   int pthread_setname_np(pthread_t, const char *)     any thread
 *
 * Linux also caps the name at 16 bytes including the terminator and fails the
 * whole call on a longer one, so `name` is truncated here rather than silently
 * dropped there. Naming is diagnostic: failure is ignored, never reported. */
void mynah_thread_set_name(const char *name);

#endif
