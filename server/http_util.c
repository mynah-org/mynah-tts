#include "http_util.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

/* Locate the value for "key" at the top level of a small JSON object. This
 * does not track nesting: the server's request bodies are flat, and anything
 * richer is rejected upstream rather than guessed at here. */
static const char *json_value(const char *json, const char *key) {
    if (json == NULL || key == NULL) return NULL;
    char pattern[96];
    const int written = snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    if (written <= 0 || (size_t)written >= sizeof(pattern)) return NULL;
    const char *at = strstr(json, pattern);
    if (at == NULL) return NULL;
    at += (size_t)written;
    while (*at == ' ' || *at == '\t' || *at == '\n' || *at == '\r') ++at;
    if (*at != ':') return NULL;
    ++at;
    while (*at == ' ' || *at == '\t' || *at == '\n' || *at == '\r') ++at;
    return at;
}

static size_t utf8_encode(unsigned cp, char *out) {
    if (cp < 0x80u) { out[0] = (char)cp; return 1; }
    if (cp < 0x800u) {
        out[0] = (char)(0xC0u | (cp >> 6));
        out[1] = (char)(0x80u | (cp & 0x3Fu));
        return 2;
    }
    out[0] = (char)(0xE0u | (cp >> 12));
    out[1] = (char)(0x80u | ((cp >> 6) & 0x3Fu));
    out[2] = (char)(0x80u | (cp & 0x3Fu));
    return 3;
}

int mynah_json_string(const char *json, const char *key,
                      char *out, size_t capacity) {
    const char *at = json_value(json, key);
    if (at == NULL || *at != '"' || capacity == 0) return -1;
    ++at;
    size_t n = 0;
    while (*at != '\0' && *at != '"') {
        if (n + 4u >= capacity) return -1;   /* leave room for UTF-8 + NUL */
        if (*at == '\\') {
            ++at;
            switch (*at) {
                case '"':  out[n++] = '"';  ++at; break;
                case '\\': out[n++] = '\\'; ++at; break;
                case '/':  out[n++] = '/';  ++at; break;
                case 'b':  out[n++] = '\b'; ++at; break;
                case 'f':  out[n++] = '\f'; ++at; break;
                case 'n':  out[n++] = '\n'; ++at; break;
                case 'r':  out[n++] = '\r'; ++at; break;
                case 't':  out[n++] = '\t'; ++at; break;
                case 'u': {
                    unsigned cp = 0;
                    ++at;
                    for (int i = 0; i < 4; ++i) {
                        const char c = at[i];
                        unsigned d;
                        if (c >= '0' && c <= '9') d = (unsigned)(c - '0');
                        else if (c >= 'a' && c <= 'f') d = (unsigned)(c - 'a' + 10);
                        else if (c >= 'A' && c <= 'F') d = (unsigned)(c - 'A' + 10);
                        else return -1;
                        cp = (cp << 4) | d;
                    }
                    at += 4;
                    /* Surrogates are not reassembled; reject rather than emit
                     * a malformed sequence. */
                    if (cp >= 0xD800u && cp <= 0xDFFFu) return -1;
                    n += utf8_encode(cp, out + n);
                    break;
                }
                default: return -1;
            }
        } else {
            out[n++] = *at++;
        }
    }
    if (*at != '"') return -1;
    out[n] = '\0';
    return 0;
}

int mynah_json_number(const char *json, const char *key, double *out) {
    const char *at = json_value(json, key);
    if (at == NULL || out == NULL) return -1;
    char *end = NULL;
    const double value = strtod(at, &end);
    if (end == at) return -1;
    *out = value;
    return 0;
}

int mynah_json_bool(const char *json, const char *key, int *out) {
    const char *at = json_value(json, key);
    if (at == NULL || out == NULL) return -1;
    if (strncmp(at, "true", 4) == 0)  { *out = 1; return 0; }
    if (strncmp(at, "false", 5) == 0) { *out = 0; return 0; }
    return -1;
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
