/* Small HTTP/JSON helpers for the mynah-tts server: header lookup, portable
 * memmem, and just enough JSON reading to serve an OpenAI-shaped request body.
 * Deliberately minimal -- the server takes a handful of well-known fields, so a
 * full JSON parser would be more surface than the job needs. */
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

/* Read a JSON string value by key, unescaping \" \\ \/ \n \r \t and \uXXXX
 * (as UTF-8). Returns 0 when found and it fits. */
int mynah_json_string(const char *json, const char *key,
                      char *out, size_t capacity);

/* Read a JSON number by key. Returns 0 when found. */
int mynah_json_number(const char *json, const char *key, double *out);

/* Read a JSON boolean by key. Returns 0 when found. */
int mynah_json_bool(const char *json, const char *key, int *out);

/* Escape a UTF-8 string into a JSON string body (no surrounding quotes).
 * Returns the number of bytes written, or (size_t)-1 if it does not fit. */
size_t mynah_json_escape(const char *in, char *out, size_t capacity);

#endif
