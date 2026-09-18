/* Model-free tests for src/json.c and the two readers it replaced.
 *
 * WHY THIS FILE IS LONG. A parser is the one component where the interesting
 * input is the input nobody meant to send: the truncated body, the lone
 * surrogate, the key that is really a substring of a value, the object nested
 * a thousand deep by a client with a grudge. A suite that only proves
 * `{"a":1}` parses proves nothing, because the code it replaced also parsed
 * `{"a":1}`.
 *
 * WHAT IT COVERS, in order:
 *   1  adversarial lookups -- a key found only where it really is
 *   2  every escape, and raw UTF-8
 *   3  surrogate pairs: valid, lone high, lone low, reversed, boundaries
 *   4  numbers at the edges of the grammar and of the double
 *   5  nesting at, and past, the bound -- including a depth no stack survives
 *   6  every truncation of a valid document, byte by byte
 *   7  unterminated strings, trailing garbage, empty documents
 *   8  duplicate keys
 *   9  invalid UTF-8, every family of it
 *  10  structural refusals, and that each one names an offset
 *  11  the value API: paths, arrays, iteration, capacities
 *  12  the server wrappers in server/http_util.c, on the same adversarial input
 *  13  a byte-substitution sweep that must not crash under a sanitizer
 *
 * Documents that must not be NUL-terminated, or that contain a NUL or raw
 * invalid UTF-8, are passed as (pointer, length) and several are copied into
 * an EXACTLY sized heap block first, so a one-byte over-read is an ASan report
 * rather than a test that quietly passes on whatever followed in .rodata.
 *
 * NEGATIVE CONTROL: tests/json_negative_control.sh breaks the parser four ways
 * and requires this suite to catch each one. A suite that has never failed is
 * not evidence.
 */
#include "json.h"
#include "http_util.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned g_checks = 0;
static unsigned g_failures = 0;
static const char *g_section = "";

#define CHECK(cond, ...)                                                       \
    do {                                                                       \
        ++g_checks;                                                            \
        if (!(cond)) {                                                         \
            ++g_failures;                                                      \
            printf("FAIL [%s] %s:%d: ", g_section, __FILE__, __LINE__);        \
            printf(__VA_ARGS__);                                               \
            printf("\n");                                                      \
        }                                                                      \
    } while (0)

static void section(const char *name) { g_section = name; }

/* ------------------------------------------------------------- utilities */

/* A copy with no terminator and not one spare byte, so an over-read is a
 * sanitizer report instead of a lucky pass. Caller frees. */
static char *exact_copy(const char *text, size_t len) {
    char *p = (char *)malloc(len == 0 ? 1u : len);
    if (p == NULL) abort();
    memcpy(p, text, len);
    return p;
}

static int parse_len(const char *text, size_t len, mynah_json_value *root,
                     mynah_json_error *err) {
    char *copy = exact_copy(text, len);
    mynah_json_error local;
    const int rc = mynah_json_parse(copy, len, root, err != NULL ? err : &local);
    /* The value points into `copy`; callers that keep it must use parse_keep. */
    free(copy);
    return rc;
}

/* Parse into a buffer the caller owns, for tests that then walk the value. */
static char *parse_keep(const char *text, size_t len, mynah_json_value *root,
                        int *ok) {
    char *copy = exact_copy(text, len);
    mynah_json_error err;
    *ok = mynah_json_parse(copy, len, root, &err) == 0;
    if (!*ok) {
        printf("       (parse refused at byte %zu: %s)\n", err.offset, err.message);
    }
    return copy;
}

static void ok_doc(const char *doc) {
    mynah_json_value root;
    mynah_json_error err;
    const int rc = parse_len(doc, strlen(doc), &root, &err);
    CHECK(rc == 0, "expected %s to parse, refused at %zu: %s", doc, err.offset,
          err.message);
}

static void bad_doc(const char *doc) {
    mynah_json_value root;
    mynah_json_error err;
    const size_t len = strlen(doc);
    const int rc = parse_len(doc, len, &root, &err);
    CHECK(rc != 0, "expected %s to be refused", doc);
    if (rc != 0) {
        CHECK(err.message[0] != '\0', "refusal of %s carries no message", doc);
        CHECK(err.offset <= len, "refusal of %s names byte %zu past the end (%zu)",
              doc, err.offset, len);
    }
}

/* Refused, and the message says the expected thing. */
static void bad_doc_saying(const char *doc, const char *needle) {
    mynah_json_value root;
    mynah_json_error err;
    const int rc = parse_len(doc, strlen(doc), &root, &err);
    CHECK(rc != 0, "expected %s to be refused", doc);
    if (rc == 0) return;
    CHECK(strstr(err.message, needle) != NULL,
          "refusal of %s says \"%s\", expected it to mention \"%s\"", doc,
          err.message, needle);
}

/* Refused at exactly this byte. */
static void bad_doc_at(const char *doc, size_t len, size_t offset) {
    mynah_json_value root;
    mynah_json_error err;
    const int rc = parse_len(doc, len, &root, &err);
    CHECK(rc != 0, "expected a refusal");
    if (rc == 0) return;
    CHECK(err.offset == offset, "refused at byte %zu, expected byte %zu (%s)",
          err.offset, offset, err.message);
}

/* Look a path up in a document and compare the decoded string. */
static void expect_string(const char *doc, const char *path, const char *want) {
    mynah_json_value root, value;
    int ok = 0;
    char *copy = parse_keep(doc, strlen(doc), &root, &ok);
    if (ok) {
        char out[512];
        if (mynah_json_lookup(&root, path, &value) != 0) {
            CHECK(0, "%s: path '%s' did not resolve", doc, path);
        } else if (mynah_json_as_string(&value, out, sizeof(out)) != 0) {
            CHECK(0, "%s: path '%s' did not decode as a string", doc, path);
        } else {
            CHECK(strcmp(out, want) == 0, "%s: '%s' gave \"%s\", expected \"%s\"",
                  doc, path, out, want);
        }
    } else {
        CHECK(0, "%s: expected it to parse", doc);
    }
    free(copy);
}

static void expect_no_path(const char *doc, const char *path) {
    mynah_json_value root, value;
    int ok = 0;
    char *copy = parse_keep(doc, strlen(doc), &root, &ok);
    CHECK(ok, "%s: expected it to parse", doc);
    if (ok) {
        CHECK(mynah_json_lookup(&root, path, &value) != 0,
              "%s: path '%s' resolved and should not have", doc, path);
    }
    free(copy);
}

static void expect_number(const char *doc, const char *path, double want) {
    mynah_json_value root, value;
    int ok = 0;
    char *copy = parse_keep(doc, strlen(doc), &root, &ok);
    CHECK(ok, "%s: expected it to parse", doc);
    if (ok) {
        double got = 0.0;
        if (mynah_json_lookup(&root, path, &value) != 0 ||
            mynah_json_as_number(&value, &got) != 0) {
            CHECK(0, "%s: '%s' did not read as a number", doc, path);
        } else {
            CHECK(got == want, "%s: '%s' gave %.17g, expected %.17g", doc, path,
                  got, want);
        }
    }
    free(copy);
}

/* Decode the single string value of {"s":<...>} and compare bytes exactly. */
static void expect_decoded(const char *literal, const char *want_bytes,
                           size_t want_len) {
    char doc[512];
    const int n = snprintf(doc, sizeof(doc), "{\"s\":%s}", literal);
    if (n <= 0 || (size_t)n >= sizeof(doc)) abort();
    mynah_json_value root, value;
    int ok = 0;
    char *copy = parse_keep(doc, (size_t)n, &root, &ok);
    CHECK(ok, "%s: expected it to parse", doc);
    if (ok) {
        char out[256];
        memset(out, 0x7E, sizeof(out));
        const int rc = mynah_json_object_get(&root, "s", &value) == 0 &&
                       mynah_json_as_string(&value, out, sizeof(out)) == 0;
        CHECK(rc, "%s: did not decode", doc);
        if (rc) {
            CHECK(strlen(out) == want_len && memcmp(out, want_bytes, want_len) == 0,
                  "%s: decoded %zu bytes, expected %zu", doc, strlen(out), want_len);
        }
    }
    free(copy);
}

/* ------------------------------------------------- 1. adversarial lookups */

static void test_adversarial_keys(void) {
    section("adversarial keys");

    /* The case from the brief: a value that contains the key spelling, with
     * the quotes escaped. strstr found the fake one. */
    expect_string("{\"voice\":\"say \\\"input\\\": fake\",\"input\":\"real\"}",
                  "input", "real");

    /* A nested object must not answer a top-level lookup... */
    expect_no_path("{\"a\":{\"input\":\"x\"}}", "input");
    /* ...and the nested one is reachable by path, which is the point. */
    expect_string("{\"a\":{\"input\":\"x\"}}", "a.input", "x");

    /* A nested key that shadows a real top-level one, nested FIRST in document
     * order -- exactly the models/fake-magpie shape, where "codec": {...}
     * carries a "sample_rate" ahead of the top-level one. */
    expect_number("{\"codec\":{\"sample_rate\":1},\"sample_rate\":22050}",
                  "sample_rate", 22050.0);
    expect_number("{\"codec\":{\"sample_rate\":1},\"sample_rate\":22050}",
                  "codec.sample_rate", 1.0);

    /* A key spelling inside an array element. */
    expect_string("{\"list\":[\"\\\"input\\\": 1\"],\"input\":\"real\"}",
                  "input", "real");

    /* A value ending in a backslash escape: the scanner must not read the
     * closing quote as escaped. */
    expect_string("{\"path\":\"c:\\\\\",\"input\":\"real\"}", "input", "real");
    expect_string("{\"path\":\"c:\\\\\",\"input\":\"real\"}", "path", "c:\\");

    /* Prefix and suffix keys are different keys. Magpie packs carry all three
     * of these, and binding a multilingual pack to one language because
     * "languages" matched a search for "language" is the bug this prevents. */
    expect_string("{\"languages\":[\"it\"],\"language_to_tokenizer\":{},"
                  "\"language\":\"en\"}", "language", "en");
    expect_no_path("{\"languages\":[\"it\"],\"language_to_tokenizer\":{}}",
                   "language");

    /* A key spelling inside another key's NAME. */
    expect_no_path("{\"x_input_y\":1}", "input");

    /* The key is the whole value of another key. */
    expect_string("{\"echo\":\"input\",\"input\":\"real\"}", "input", "real");

    /* Deeply buried, never at the top. */
    expect_no_path("{\"a\":{\"b\":{\"c\":{\"input\":\"deep\"}}}}", "input");
    expect_string("{\"a\":{\"b\":{\"c\":{\"input\":\"deep\"}}}}", "a.b.c.input",
                  "deep");
}

/* -------------------------------------------------- 2. escapes and UTF-8 */

static void test_escapes(void) {
    section("escapes");

    expect_decoded("\"\\\"\"", "\"", 1);
    expect_decoded("\"\\\\\"", "\\", 1);
    expect_decoded("\"\\/\"", "/", 1);
    expect_decoded("\"\\b\"", "\b", 1);
    expect_decoded("\"\\f\"", "\f", 1);
    expect_decoded("\"\\n\"", "\n", 1);
    expect_decoded("\"\\r\"", "\r", 1);
    expect_decoded("\"\\t\"", "\t", 1);
    expect_decoded("\"\\u0041\"", "A", 1);
    expect_decoded("\"\\u0001\"", "\x01", 1);
    expect_decoded("\"\\u007f\"", "\x7f", 1);
    /* The three UTF-8 widths a BMP escape can produce. */
    expect_decoded("\"\\u00e9\"", "\xc3\xa9", 2);                 /* e-acute  */
    expect_decoded("\"\\u07ff\"", "\xdf\xbf", 2);                 /* 2-byte max */
    expect_decoded("\"\\u0800\"", "\xe0\xa0\x80", 3);             /* 3-byte min */
    expect_decoded("\"\\u20ac\"", "\xe2\x82\xac", 3);             /* euro     */
    expect_decoded("\"\\uffff\"", "\xef\xbf\xbf", 3);             /* 3-byte max */
    /* Case-insensitive hex, and escapes mixed with literal text. */
    expect_decoded("\"\\uAbCd\"", "\xea\xaf\x8d", 3);
    expect_decoded("\"a\\nb\\tc\"", "a\nb\tc", 5);
    expect_decoded("\"\\u0041\\u0042\"", "AB", 2);

    /* Raw UTF-8 passes through untouched, at every width. */
    expect_decoded("\"caf\xc3\xa9\"", "caf\xc3\xa9", 5);
    expect_decoded("\"\xe4\xb8\xad\"", "\xe4\xb8\xad", 3);
    expect_decoded("\"\xf0\x9f\x98\x80\"", "\xf0\x9f\x98\x80", 4);

    /* Refusals. */
    bad_doc_saying("{\"s\":\"\\x\"}", "invalid escape");
    bad_doc_saying("{\"s\":\"\\u00\"}", "hex");
    bad_doc_saying("{\"s\":\"\\u00zz\"}", "hex");
    bad_doc("{\"s\":\"\\\"}");                       /* escaped closing quote */
    bad_doc("{\"s\":\"\\");
    bad_doc("{\"s\":\"\\u");
    bad_doc("{\"s\":\"\\u12");

    /* A raw control byte inside a string is not JSON, and the message says
     * which one and what to write instead. */
    bad_doc_saying("{\"s\":\"a\nb\"}", "U+000A");
    bad_doc_saying("{\"s\":\"a\tb\"}", "U+0009");
    {
        const char doc[] = "{\"s\":\"a\x01\x62\"}";
        bad_doc_at(doc, sizeof(doc) - 1u, 7);
    }

    /* Escapes are legal in member NAMES too, and a name is matched decoded. */
    expect_string("{\"a\\u0062\":\"yes\"}", "ab", "yes");
    expect_string("{\"a\\nb\":\"yes\"}", "a\nb", "yes");
}

/* ---------------------------------------------------- 3. surrogate pairs */

static void test_surrogates(void) {
    section("surrogates");

    /* The whole point: an emoji is ONE codepoint, not an error. */
    expect_decoded("\"\\ud83d\\ude00\"", "\xf0\x9f\x98\x80", 4);   /* U+1F600 */
    expect_decoded("\"\\uD83D\\uDE00\"", "\xf0\x9f\x98\x80", 4);
    /* Both ends of the astral range. */
    expect_decoded("\"\\ud800\\udc00\"", "\xf0\x90\x80\x80", 4);   /* U+10000 */
    expect_decoded("\"\\udbff\\udfff\"", "\xf4\x8f\xbf\xbf", 4);   /* U+10FFFF */
    /* A pair surrounded by ordinary text, and two pairs in a row. */
    expect_decoded("\"a\\ud83d\\ude00b\"", "a\xf0\x9f\x98\x80" "b", 6);
    expect_decoded("\"\\ud83d\\ude00\\ud83d\\ude01\"",
                   "\xf0\x9f\x98\x80\xf0\x9f\x98\x81", 8);

    /* Lone high surrogate, at the end of the string and mid-string. */
    bad_doc_saying("{\"s\":\"\\ud83d\"}", "lone high surrogate");
    bad_doc_saying("{\"s\":\"\\ud83dx\"}", "lone high surrogate");
    /* High surrogate followed by an escape that is not a low surrogate. */
    bad_doc_saying("{\"s\":\"\\ud83d\\u0041\"}", "expected \\uDC00-\\uDFFF");
    /* Lone low surrogate. */
    bad_doc_saying("{\"s\":\"\\ude00\"}", "lone low surrogate");
    /* The pair in the wrong order. */
    bad_doc_saying("{\"s\":\"\\ude00\\ud83d\"}", "lone low surrogate");
    /* High surrogate then a high surrogate. */
    bad_doc_saying("{\"s\":\"\\ud83d\\ud83d\"}", "expected \\uDC00-\\uDFFF");
    /* High surrogate then a truncated escape. */
    bad_doc("{\"s\":\"\\ud83d\\u00\"}");
    bad_doc("{\"s\":\"\\ud83d\\\"}");
    /* Just outside the surrogate block on both sides: ordinary codepoints. */
    expect_decoded("\"\\ud7ff\"", "\xed\x9f\xbf", 3);
    expect_decoded("\"\\ue000\"", "\xee\x80\x80", 3);
}

/* ------------------------------------------------------------ 4. numbers */

static void test_numbers(void) {
    section("numbers");

    expect_number("{\"n\":0}", "n", 0.0);
    expect_number("{\"n\":-0}", "n", 0.0);
    expect_number("{\"n\":1}", "n", 1.0);
    expect_number("{\"n\":-1}", "n", -1.0);
    expect_number("{\"n\":3.14}", "n", 3.14);
    expect_number("{\"n\":1e3}", "n", 1000.0);
    expect_number("{\"n\":1E3}", "n", 1000.0);
    expect_number("{\"n\":1e+3}", "n", 1000.0);
    expect_number("{\"n\":1e-3}", "n", 0.001);
    expect_number("{\"n\":-1.5e-3}", "n", -0.0015);
    expect_number("{\"n\":0.0}", "n", 0.0);
    expect_number("{\"n\":1e308}", "n", 1e308);
    expect_number("{\"n\":21.533203125}", "n", 21.533203125);
    expect_number("{\"n\":1e-06}", "n", 1e-06);
    /* Underflow to zero is a representable answer and is accepted. */
    expect_number("{\"n\":1e-400}", "n", 0.0);
    /* 2^53 and one past it: the largest exactly representable integers. */
    expect_number("{\"n\":9007199254740992}", "n", 9007199254740992.0);

    /* Not JSON numbers, whatever strtod would have made of them. */
    bad_doc("{\"n\":01}");
    bad_doc("{\"n\":-01}");
    bad_doc("{\"n\":+1}");
    bad_doc("{\"n\":.5}");
    bad_doc("{\"n\":1.}");
    bad_doc("{\"n\":1.e3}");
    bad_doc("{\"n\":1e}");
    bad_doc("{\"n\":1e+}");
    bad_doc("{\"n\":-}");
    bad_doc("{\"n\":0x10}");
    bad_doc("{\"n\":Infinity}");
    bad_doc("{\"n\":-Infinity}");
    bad_doc("{\"n\":NaN}");
    bad_doc("{\"n\":nan}");
    bad_doc("{\"n\":--1}");
    bad_doc("{\"n\":1..2}");
    bad_doc("{\"n\":1e1e1}");
    bad_doc("{\"n\":1 2}");
    bad_doc("{\"n\":00}");
    bad_doc("{\"n\":1_000}");
    /* The old readers took every one of these through strtod: a leading space,
     * a '+', a hex literal, an infinity. */
    bad_doc("{\"n\": 0x1p3}");

    /* Overflow is refused at the accessor, where the literal is still in hand,
     * rather than becoming an infinity that poisons a float pipeline. */
    {
        mynah_json_value root, v;
        int ok = 0;
        char *copy = parse_keep("{\"n\":1e999}", 11, &root, &ok);
        CHECK(ok, "1e999 is a grammatical JSON number and must parse");
        if (ok) {
            double d = 0.0;
            CHECK(mynah_json_object_get(&root, "n", &v) == 0, "n missing");
            CHECK(mynah_json_as_number(&v, &d) != 0,
                  "1e999 converted to %.17g instead of being refused", d);
        }
        free(copy);
    }
    /* A literal too long for the conversion buffer: refused, never truncated. */
    {
        char doc[900];
        size_t n = (size_t)snprintf(doc, sizeof(doc), "{\"n\":1");
        for (size_t i = 0; i < 600; ++i) doc[n++] = '0';
        doc[n++] = '}';
        mynah_json_value root, v;
        int ok = 0;
        char *copy = parse_keep(doc, n, &root, &ok);
        CHECK(ok, "a 600-digit integer is grammatical and must parse");
        if (ok) {
            double d = 0.0;
            CHECK(mynah_json_object_get(&root, "n", &v) == 0, "n missing");
            CHECK(mynah_json_as_number(&v, &d) != 0,
                  "a 601-byte literal converted to %.17g instead of being refused", d);
        }
        free(copy);
    }

    /* as_unsigned: the model.json accessor. */
    struct { const char *doc; int ok; unsigned want; } u[] = {
        { "{\"n\":0}",           1, 0u },
        { "{\"n\":4}",           1, 4u },
        { "{\"n\":4.0}",         1, 4u },
        { "{\"n\":4294967295}",  1, 4294967295u },
        { "{\"n\":2016}",        1, 2016u },
        { "{\"n\":4294967296}",  0, 0u },
        { "{\"n\":-1}",          0, 0u },
        { "{\"n\":4.5}",         0, 0u },
        { "{\"n\":1e999}",       0, 0u },
        { "{\"n\":\"4\"}",       0, 0u },
        { "{\"n\":true}",        0, 0u },
        { "{\"n\":null}",        0, 0u },
    };
    for (size_t i = 0; i < sizeof(u) / sizeof(u[0]); ++i) {
        mynah_json_value root, v;
        int ok = 0;
        char *copy = parse_keep(u[i].doc, strlen(u[i].doc), &root, &ok);
        CHECK(ok, "%s: expected it to parse", u[i].doc);
        if (ok) {
            unsigned got = 0xDEADBEEFu;
            const int rc = mynah_json_object_get(&root, "n", &v) == 0 &&
                           mynah_json_as_unsigned(&v, &got) == 0;
            CHECK(rc == u[i].ok, "%s: as_unsigned returned %d, expected %d",
                  u[i].doc, rc, u[i].ok);
            if (u[i].ok && rc) {
                CHECK(got == u[i].want, "%s: gave %u, expected %u", u[i].doc, got,
                      u[i].want);
            }
        }
        free(copy);
    }

    /* A number is not a boolean and a boolean is not a number. */
    {
        mynah_json_value root, v;
        int ok = 0, b = 0;
        double d = 0.0;
        char *copy = parse_keep("{\"t\":true,\"f\":false,\"n\":1}", 26, &root, &ok);
        CHECK(ok, "expected it to parse");
        if (ok) {
            CHECK(mynah_json_object_get(&root, "t", &v) == 0 &&
                  mynah_json_as_bool(&v, &b) == 0 && b == 1, "true");
            CHECK(mynah_json_object_get(&root, "f", &v) == 0 &&
                  mynah_json_as_bool(&v, &b) == 0 && b == 0, "false");
            CHECK(mynah_json_object_get(&root, "n", &v) == 0 &&
                  mynah_json_as_bool(&v, &b) != 0, "1 is not a boolean");
            CHECK(mynah_json_object_get(&root, "t", &v) == 0 &&
                  mynah_json_as_number(&v, &d) != 0, "true is not a number");
        }
        free(copy);
    }
}

/* -------------------------------------------------------------- 5. depth */

/* n nested containers around a scalar, e.g. n=2 -> "[[1]]" or "{\"a\":{\"a\":1}}" */
static char *build_nest(size_t n, int as_object, size_t *out_len) {
    const size_t per_open = as_object ? 5u : 1u;     /* {"a": */
    const size_t cap = n * (per_open + 1u) + 8u;
    char *buf = (char *)malloc(cap);
    if (buf == NULL) abort();
    size_t k = 0;
    for (size_t i = 0; i < n; ++i) {
        if (as_object) { memcpy(buf + k, "{\"a\":", 5); k += 5; }
        else buf[k++] = '[';
    }
    buf[k++] = '1';
    for (size_t i = 0; i < n; ++i) buf[k++] = as_object ? '}' : ']';
    *out_len = k;
    return buf;
}

static void test_depth(void) {
    section("depth");

    for (int as_object = 0; as_object <= 1; ++as_object) {
        size_t len = 0;
        char *at = build_nest(MYNAH_JSON_MAX_DEPTH, as_object, &len);
        mynah_json_value root;
        mynah_json_error err;
        CHECK(parse_len(at, len, &root, &err) == 0,
              "%d containers deep must parse (refused at %zu: %s)",
              MYNAH_JSON_MAX_DEPTH, err.offset, err.message);
        free(at);

        char *past = build_nest(MYNAH_JSON_MAX_DEPTH + 1u, as_object, &len);
        const int rc = parse_len(past, len, &root, &err);
        CHECK(rc != 0, "%d containers deep must be refused",
              MYNAH_JSON_MAX_DEPTH + 1);
        if (rc != 0) {
            CHECK(strstr(err.message, "nesting depth") != NULL,
                  "depth refusal says \"%s\"", err.message);
            /* The offset names the byte that would have crossed the bound. */
            const size_t expect = as_object
                ? (size_t)MYNAH_JSON_MAX_DEPTH * 5u
                : (size_t)MYNAH_JSON_MAX_DEPTH;
            CHECK(err.offset == expect, "depth refusal at byte %zu, expected %zu",
                  err.offset, expect);
        }
        free(past);
    }

    /* The case this bound exists for. A recursive-descent parser dies here;
     * this one returns an error and the process keeps serving. */
    {
        const size_t n = 200000;
        char *deep = (char *)malloc(n * 2u + 2u);
        if (deep == NULL) abort();
        for (size_t i = 0; i < n; ++i) deep[i] = '[';
        for (size_t i = 0; i < n; ++i) deep[n + i] = ']';
        mynah_json_value root;
        mynah_json_error err;
        CHECK(mynah_json_parse(deep, n * 2u, &root, &err) != 0,
              "200000 nested arrays must be refused");
        CHECK(strstr(err.message, "nesting depth") != NULL,
              "200000-deep refusal says \"%s\"", err.message);
        /* Unterminated and deep: the other half of the same attack. */
        CHECK(mynah_json_parse(deep, n, &root, &err) != 0,
              "200000 unterminated nested arrays must be refused");
        free(deep);
    }

    /* Mixed nesting counts the same way. */
    {
        char buf[256];
        size_t k = 0;
        for (int i = 0; i < MYNAH_JSON_MAX_DEPTH / 2; ++i) {
            buf[k++] = '[';
            memcpy(buf + k, "{\"a\":", 5);
            k += 5;
        }
        buf[k++] = '1';
        for (int i = 0; i < MYNAH_JSON_MAX_DEPTH / 2; ++i) {
            buf[k++] = '}';
            buf[k++] = ']';
        }
        mynah_json_value root;
        mynah_json_error err;
        CHECK(parse_len(buf, k, &root, &err) == 0,
              "mixed nesting to the bound must parse: %s", err.message);
    }

    /* Depth is about CONTAINERS, not siblings: a wide flat array is fine. */
    {
        const size_t n = 5000;
        char *wide = (char *)malloc(n * 2u + 4u);
        if (wide == NULL) abort();
        size_t k = 0;
        wide[k++] = '[';
        for (size_t i = 0; i < n; ++i) {
            if (i > 0) wide[k++] = ',';
            wide[k++] = '7';
        }
        wide[k++] = ']';
        mynah_json_value root;
        mynah_json_error err;
        CHECK(mynah_json_parse(wide, k, &root, &err) == 0,
              "a 5000-element array must parse: %s", err.message);
        size_t count = 0;
        CHECK(mynah_json_count(&root, &count) == 0 && count == n,
              "counted %zu elements, expected %zu", count, n);
        free(wide);
    }
}

/* --------------------------------------------------------- 6. truncation */

static void test_truncation(void) {
    section("truncation");

    /* Every proper prefix of a valid document is an invalid document -- the
     * root is a container, so nothing short of the last byte can close it.
     * Each prefix is copied to an exactly sized block: a scanner that reads one
     * byte past the length it was given is an ASan report here, and that is the
     * single most likely defect in a parser handed a socket's worth of bytes. */
    const char *doc =
        "{\"engine\":\"magpie\",\"nested\":{\"a\":[1,-2.5e3,true,false,null,"
        "\"\\u00e9\\ud83d\\ude00\"],\"b\":{}},\"list\":[[],{},\"x\"],"
        "\"flag\":true,\"n\":0.5}";
    const size_t len = strlen(doc);

    mynah_json_value root;
    mynah_json_error err;
    CHECK(parse_len(doc, len, &root, &err) == 0,
          "the reference document must parse: %s", err.message);

    unsigned accepted = 0;
    for (size_t cut = 0; cut < len; ++cut) {
        mynah_json_value r;
        mynah_json_error e;
        if (parse_len(doc, cut, &r, &e) == 0) {
            ++accepted;
            CHECK(0, "prefix of %zu bytes was accepted", cut);
        } else {
            CHECK(e.offset <= cut, "prefix of %zu bytes refused at byte %zu", cut,
                  e.offset);
            CHECK(e.message[0] != '\0', "prefix of %zu bytes refused silently", cut);
        }
    }
    CHECK(accepted == 0, "%u prefixes were accepted", accepted);

    /* And every prefix of a document whose root is a plain number: those DO
     * have valid prefixes, so the test is only that nothing crashes and that
     * an accepted prefix reads back as the number it actually is. */
    {
        const char *num = "12345.678e2";
        for (size_t cut = 1; cut <= strlen(num); ++cut) {
            mynah_json_value r;
            mynah_json_error e;
            if (parse_len(num, cut, &r, &e) == 0) {
                CHECK(r.type == MYNAH_JSON_NUMBER, "prefix %zu is not a number", cut);
            }
        }
    }
}

/* ------------------------ 7. unterminated, trailing garbage, empty input */

static void test_malformed(void) {
    section("malformed");

    /* Empty and whitespace-only documents. */
    bad_doc_saying("", "empty document");
    bad_doc_saying("   ", "empty document");
    bad_doc_saying("\n\t\r ", "empty document");
    {
        mynah_json_value root;
        mynah_json_error err;
        CHECK(mynah_json_parse(NULL, 0, &root, &err) != 0, "NULL must be refused");
        CHECK(mynah_json_parse("{}", 2, NULL, &err) != 0,
              "a NULL destination must be refused");
    }

    /* Unterminated strings, in values and in names. */
    bad_doc_saying("{\"a\":\"b}", "unterminated string");
    bad_doc_saying("\"abc", "unterminated string");
    bad_doc_saying("{\"a", "unterminated string");
    bad_doc("{\"a\":\"b\\\"}");
    bad_doc("[\"");

    /* Trailing content. A document is one value and whitespace. */
    bad_doc_saying("{} }", "trailing content");
    bad_doc_saying("{}x", "trailing content");
    bad_doc_saying("{\"a\":1} {\"b\":2}", "trailing content");
    bad_doc_saying("1 2", "trailing content");
    bad_doc_saying("[1][2]", "trailing content");
    bad_doc_saying("null null", "trailing content");
    ok_doc("  {\"a\":1}  \n");            /* whitespace either side is fine */
    ok_doc("\t[1]\r\n");

    /* Structure. */
    bad_doc("{");
    bad_doc("}");
    bad_doc("[");
    bad_doc("]");
    bad_doc("{,}");
    bad_doc("{\"a\"}");
    bad_doc("{\"a\":}");
    bad_doc("{:1}");
    bad_doc("{\"a\":1,}");
    bad_doc("[1,]");
    bad_doc("[,1]");
    bad_doc("[1 2]");
    bad_doc("{\"a\":1 \"b\":2}");
    bad_doc("[}");
    bad_doc("{]");
    bad_doc("{\"a\":[1,2}");
    bad_doc("[{\"a\":1]");
    bad_doc("{'a':1}");                   /* single quotes are not JSON */
    bad_doc("{a:1}");                     /* nor are bare names */
    bad_doc("{\"a\":1,\"b\"}");
    bad_doc("[1,2,]");
    bad_doc("()");

    /* Literals, spelled wrong. */
    bad_doc("tru");
    bad_doc("nul");
    bad_doc("flase");
    bad_doc("TRUE");
    bad_doc("True");
    bad_doc("undefined");
    ok_doc("true");
    ok_doc("false");
    ok_doc("null");

    /* The messages name a locus. */
    bad_doc_at("{\"a\" 1}", 7, 5);                   /* expected ':'          */
    bad_doc_at("{\"a\":1 \"b\":2}", 13, 7);          /* expected ',' or '}'   */
    bad_doc_at("[1,]", 4, 3);                        /* trailing comma        */
    bad_doc_at("{\"a\":01}", 8, 6);                  /* leading zero          */
    bad_doc_saying("{\"a\" 1}", "expected ':'");
    bad_doc_saying("[1 2]", "expected ',' or ']'");
    bad_doc_saying("[1,]", "trailing comma");
    bad_doc_saying("{1:2}", "member name");

    /* Empty containers, and containers of containers, are all fine. */
    ok_doc("{}");
    ok_doc("[]");
    ok_doc("[[]]");
    ok_doc("[{},{}]");
    ok_doc("{\"a\":{},\"b\":[]}");
    ok_doc("{\"\":\"\"}");                            /* empty name and value */
}

/* ----------------------------------------------------- 8. duplicate keys */

static void test_duplicates(void) {
    section("duplicates");

    /* DECIDED: the first one wins, and a duplicate is not an error. It is what
     * the strstr readers did, so nothing changes meaning, and it means a key
     * appended to a body cannot override the key already in it. */
    expect_number("{\"a\":1,\"a\":2}", "a", 1.0);
    expect_number("{\"a\":1,\"b\":9,\"a\":2,\"a\":3}", "a", 1.0);
    expect_string("{\"input\":\"first\",\"input\":\"second\"}", "input", "first");
    /* Different types under the same name: still the first. */
    expect_number("{\"a\":7,\"a\":\"seven\"}", "a", 7.0);
    /* A duplicate that arrives with an escaped spelling of the same name. */
    expect_number("{\"a\":1,\"\\u0061\":2}", "a", 1.0);

    /* The count sees every member, duplicates included. */
    {
        mynah_json_value root;
        int ok = 0;
        char *copy = parse_keep("{\"a\":1,\"a\":2,\"a\":3}", 19, &root, &ok);
        CHECK(ok, "expected it to parse");
        if (ok) {
            size_t n = 0;
            CHECK(mynah_json_count(&root, &n) == 0 && n == 3,
                  "counted %zu members, expected 3", n);
        }
        free(copy);
    }
}

/* -------------------------------------------------------- 9. bad UTF-8 */

/* Each case is the byte(s) that go inside a string, and the offset in the
 * document {"s":"<bytes>"} at which the refusal must land: 6. */
static void test_bad_utf8(void) {
    section("invalid UTF-8");

    static const struct { const char *bytes; size_t len; const char *what; } cases[] = {
        { "\x80",                 1, "bare continuation byte" },
        { "\xbf",                 1, "bare continuation byte" },
        { "\xc0\x80",             2, "overlong NUL" },
        { "\xc0\xaf",             2, "overlong '/'" },
        { "\xc1\xbf",             2, "overlong 0x7f" },
        { "\xe0\x80\xaf",         3, "3-byte overlong" },
        { "\xe0\x9f\xbf",         3, "3-byte overlong, upper edge" },
        { "\xf0\x80\x80\xaf",     4, "4-byte overlong" },
        { "\xf0\x8f\xbf\xbf",     4, "4-byte overlong, upper edge" },
        { "\xed\xa0\x80",         3, "surrogate U+D800 as CESU-8" },
        { "\xed\xbf\xbf",         3, "surrogate U+DFFF as CESU-8" },
        { "\xf4\x90\x80\x80",     4, "above U+10FFFF" },
        { "\xf5\x80\x80\x80",     4, "F5: above U+10FFFF" },
        { "\xfe",                 1, "never a UTF-8 byte" },
        { "\xff",                 1, "never a UTF-8 byte" },
        { "\xe2\x82",             2, "truncated 3-byte sequence" },
        { "\xe2",                 1, "lone lead byte" },
        { "\xf0\x9f\x98",         3, "truncated 4-byte sequence" },
        { "\xe2\x28\xa1",         3, "bad continuation" },
        { "\xc3",                 1, "lead byte at end of string" },
    };

    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
        char doc[64];
        size_t k = 0;
        memcpy(doc + k, "{\"s\":\"", 6); k += 6;
        memcpy(doc + k, cases[i].bytes, cases[i].len); k += cases[i].len;
        memcpy(doc + k, "\"}", 2); k += 2;
        mynah_json_value root;
        mynah_json_error err;
        const int rc = parse_len(doc, k, &root, &err);
        CHECK(rc != 0, "%s must be refused", cases[i].what);
        if (rc != 0) {
            CHECK(err.offset == 6u, "%s refused at byte %zu, expected 6 (%s)",
                  cases[i].what, err.offset, err.message);
        }
    }

    /* Invalid UTF-8 in a member NAME is refused on the same terms. */
    {
        const char doc[] = "{\"\xff\":1}";
        bad_doc_at(doc, sizeof(doc) - 1u, 2);
    }
    /* And outside any string, where no non-ASCII byte is ever legal. */
    {
        const char doc[] = "{\"a\":\xc3\xa9}";
        bad_doc_at(doc, sizeof(doc) - 1u, 5);
    }

    /* Valid UTF-8 at every width, in a name and in a value, still passes. */
    ok_doc("{\"caf\xc3\xa9\":\"\xe4\xb8\xad\xf0\x9f\x98\x80\"}");
    expect_string("{\"caf\xc3\xa9\":\"ok\"}", "caf\xc3\xa9", "ok");

    /* A NUL byte is a byte the grammar has no place for -- and the length, not
     * a terminator, is what bounds the document. */
    {
        const char doc[] = "{\"a\":1\0,\"b\":2}";
        bad_doc_at(doc, sizeof(doc) - 1u, 6);
    }
    /* A NUL INSIDE a string is a control character and refused as one. */
    {
        const char doc[] = "{\"a\":\"x\0y\"}";
        bad_doc_at(doc, sizeof(doc) - 1u, 7);
    }
}

/* ------------------------------------------------------- 10. the value API */

static void test_value_api(void) {
    section("value API");

    const char *doc =
        "{\"engine\":\"pocket\",\"codec\":{\"samples_per_frame\":1024,"
        "\"levels\":[8,7,6,6]},\"codec_ratios\":[6,5,4],"
        "\"deep\":{\"list\":[{\"c\":\"found\"},{\"c\":\"no\"}]},"
        "\"flag\":true,\"none\":null,\"empty\":{},\"emptylist\":[]}";
    mynah_json_value root, v;
    int ok = 0;
    char *copy = parse_keep(doc, strlen(doc), &root, &ok);
    CHECK(ok, "the reference document must parse");
    if (!ok) { free(copy); return; }

    CHECK(root.type == MYNAH_JSON_OBJECT, "root is not an object");

    /* Paths: nested objects, array indices, both at once. */
    expect_number(doc, "codec.samples_per_frame", 1024.0);
    expect_number(doc, "codec.levels[0]", 8.0);
    expect_number(doc, "codec.levels[3]", 6.0);
    expect_number(doc, "codec_ratios[1]", 5.0);
    expect_string(doc, "deep.list[0].c", "found");
    expect_string(doc, "deep.list[1].c", "no");
    expect_no_path(doc, "codec.levels[4]");
    expect_no_path(doc, "codec.missing");
    expect_no_path(doc, "missing.anything");
    expect_no_path(doc, "engine.nested");        /* a path into a scalar */
    expect_no_path(doc, "codec_ratios.0");       /* dots do not index arrays */
    expect_no_path(doc, "codec.levels[");
    expect_no_path(doc, "codec.levels[]");
    expect_no_path(doc, "codec.levels[x]");
    expect_no_path(doc, "codec..levels");
    expect_no_path(doc, ".codec");

    /* The empty path is the root. */
    CHECK(mynah_json_lookup(&root, "", &v) == 0 && v.type == MYNAH_JSON_OBJECT,
          "the empty path must resolve to the root");

    /* Types. */
    CHECK(mynah_json_object_get(&root, "flag", &v) == 0 &&
          v.type == MYNAH_JSON_BOOL, "flag is not a bool");
    CHECK(mynah_json_object_get(&root, "none", &v) == 0 && mynah_json_is_null(&v),
          "none is not null");
    CHECK(mynah_json_object_get(&root, "empty", &v) == 0 &&
          v.type == MYNAH_JSON_OBJECT, "empty is not an object");
    CHECK(mynah_json_object_get(&root, "emptylist", &v) == 0 &&
          v.type == MYNAH_JSON_ARRAY, "emptylist is not an array");

    /* Counts, including of empty containers; scalars have none. */
    {
        size_t n = 0;
        CHECK(mynah_json_count(&root, &n) == 0 && n == 8,
              "root has %zu members, expected 8", n);
        CHECK(mynah_json_lookup(&root, "codec_ratios", &v) == 0 &&
              mynah_json_count(&v, &n) == 0 && n == 3, "codec_ratios count");
        CHECK(mynah_json_lookup(&root, "empty", &v) == 0 &&
              mynah_json_count(&v, &n) == 0 && n == 0, "empty count");
        CHECK(mynah_json_lookup(&root, "emptylist", &v) == 0 &&
              mynah_json_count(&v, &n) == 0 && n == 0, "emptylist count");
        CHECK(mynah_json_lookup(&root, "engine", &v) == 0 &&
              mynah_json_count(&v, &n) != 0, "a string has no member count");
    }

    /* Wrong container, wrong index. */
    CHECK(mynah_json_lookup(&root, "codec_ratios", &v) == 0, "codec_ratios");
    {
        mynah_json_value e;
        CHECK(mynah_json_object_get(&v, "0", &e) != 0,
              "object_get on an array must fail");
        CHECK(mynah_json_array_get(&v, 3, &e) != 0, "index 3 of a 3-element array");
        CHECK(mynah_json_array_get(&v, (size_t)-1, &e) != 0, "index SIZE_MAX");
        CHECK(mynah_json_array_get(&root, 0, &e) != 0,
              "array_get on an object must fail");
        CHECK(mynah_json_lookup(&root, "emptylist", &e) == 0 &&
              mynah_json_array_get(&e, 0, &e) != 0, "index 0 of an empty array");
    }

    /* Iteration sees every member, in document order. */
    {
        static const char *expect[] = { "engine", "codec", "codec_ratios", "deep",
                                        "flag", "none", "empty", "emptylist" };
        size_t cursor = 0, i = 0;
        mynah_json_value name, value;
        while (mynah_json_object_next(&root, &cursor, &name, &value) == 0) {
            char buf[64];
            CHECK(i < sizeof(expect) / sizeof(expect[0]), "too many members");
            if (i >= sizeof(expect) / sizeof(expect[0])) break;
            CHECK(mynah_json_as_string(&name, buf, sizeof(buf)) == 0 &&
                  strcmp(buf, expect[i]) == 0,
                  "member %zu is not %s", i, expect[i]);
            ++i;
        }
        CHECK(i == 8, "iterated %zu members, expected 8", i);
    }

    /* Spans point back at the document exactly. */
    CHECK(mynah_json_object_get(&root, "engine", &v) == 0, "engine");
    CHECK(v.end - v.start == 8u && memcmp(copy + v.start, "\"pocket\"", 8) == 0,
          "the engine span is not the engine literal");

    free(copy);

    /* Non-object roots parse and report their type. */
    {
        struct { const char *doc; mynah_json_type type; } roots[] = {
            { "[1,2,3]", MYNAH_JSON_ARRAY },  { "\"x\"", MYNAH_JSON_STRING },
            { "42", MYNAH_JSON_NUMBER },      { "true", MYNAH_JSON_BOOL },
            { "null", MYNAH_JSON_NULL },      { "{}", MYNAH_JSON_OBJECT },
        };
        for (size_t i = 0; i < sizeof(roots) / sizeof(roots[0]); ++i) {
            mynah_json_value r;
            mynah_json_error e;
            CHECK(parse_len(roots[i].doc, strlen(roots[i].doc), &r, &e) == 0,
                  "%s must parse: %s", roots[i].doc, e.message);
            CHECK(r.type == roots[i].type, "%s has type %d, expected %d",
                  roots[i].doc, (int)r.type, (int)roots[i].type);
        }
    }

    /* Capacity. as_string must fill exactly, and refuse one byte short --
     * including when the byte that would not fit is in the middle of a
     * multi-byte character. */
    {
        mynah_json_value r, s;
        int good = 0;
        char *buf = parse_keep("{\"s\":\"abcd\"}", 12, &r, &good);
        CHECK(good, "expected it to parse");
        if (good) {
            char out[8];
            CHECK(mynah_json_object_get(&r, "s", &s) == 0, "s");
            CHECK(mynah_json_as_string(&s, out, 5) == 0 && strcmp(out, "abcd") == 0,
                  "an exact fit must succeed");
            CHECK(mynah_json_as_string(&s, out, 4) != 0, "one byte short must fail");
            CHECK(mynah_json_as_string(&s, out, 1) != 0, "no room must fail");
            CHECK(mynah_json_as_string(&s, out, 0) != 0, "zero capacity must fail");
        }
        free(buf);

        buf = parse_keep("{\"s\":\"\\ud83d\\ude00\"}", 20, &r, &good);
        CHECK(good, "expected the emoji document to parse");
        if (good) {
            char out[8];
            CHECK(mynah_json_object_get(&r, "s", &s) == 0, "s");
            CHECK(mynah_json_as_string(&s, out, 5) == 0,
                  "4 bytes plus a terminator must fit in 5");
            CHECK(mynah_json_as_string(&s, out, 4) != 0,
                  "a 4-byte character must not be half-written into 4 bytes");
        }
        free(buf);
    }

    /* \u0000: grammatical JSON, but it cannot be carried in a C string, and
     * truncating there silently would hand a tokenizer a prefix of the input. */
    {
        mynah_json_value r, s;
        int good = 0;
        char *buf = parse_keep("{\"s\":\"a\\u0000b\"}", 16, &r, &good);
        CHECK(good, "\\u0000 is grammatical and must parse");
        if (good) {
            char out[16];
            CHECK(mynah_json_object_get(&r, "s", &s) == 0, "s");
            CHECK(mynah_json_as_string(&s, out, sizeof(out)) != 0,
                  "\\u0000 must be refused by the decoder, not truncated");
        }
        free(buf);
    }

    /* A realistic flat model.json, of the shape the packs actually ship. */
    {
        const char *pack =
            "{\n  \"engine\": \"magpie\",\n  \"codec\": {\n    \"codebooks\": 8,\n"
            "    \"levels\": [8, 7, 6, 6],\n    \"sample_rate\": 22050,\n"
            "    \"samples_per_frame\": 1024\n  },\n  \"frame_rate\": 21.533203125,\n"
            "  \"inference\": { \"min_generated_frames\": 4 },\n"
            "  \"language_to_tokenizer\": { \"it\": \"italian_chartokenizer\" },\n"
            "  \"languages\": [\"it\"],\n  \"requires_normalized_text\": true,\n"
            "  \"sample_rate\": 22050,\n  \"temperature\": 0.7,\n"
            "  \"weights\": { \"tts\": \"tts.safetensors\" }\n}\n";
        expect_string(pack, "engine", "magpie");
        expect_number(pack, "frame_rate", 21.533203125);
        expect_number(pack, "sample_rate", 22050.0);
        expect_number(pack, "codec.samples_per_frame", 1024.0);
        expect_number(pack, "inference.min_generated_frames", 4.0);
        expect_string(pack, "languages[0]", "it");
        expect_string(pack, "language_to_tokenizer.it", "italian_chartokenizer");
        expect_string(pack, "weights.tts", "tts.safetensors");
        expect_no_path(pack, "language");       /* "languages" is a different key */
        expect_no_path(pack, "codebooks");      /* nested, not top level          */
    }

    /* A wide object: the last member of a thousand is still found. */
    {
        const size_t n = 1000;
        char *big = (char *)malloc(n * 32u + 8u);
        if (big == NULL) abort();
        size_t k = 0;
        big[k++] = '{';
        for (size_t i = 0; i < n; ++i) {
            k += (size_t)snprintf(big + k, 32u, "%s\"k%zu\":%zu", i ? "," : "", i, i);
        }
        big[k++] = '}';
        mynah_json_value r, e;
        mynah_json_error err;
        CHECK(mynah_json_parse(big, k, &r, &err) == 0, "wide object: %s", err.message);
        double d = 0.0;
        CHECK(mynah_json_object_get(&r, "k999", &e) == 0 &&
              mynah_json_as_number(&e, &d) == 0 && d == 999.0,
              "the thousandth member must be found");
        CHECK(mynah_json_object_get(&r, "k1000", &e) != 0, "k1000 does not exist");
        free(big);
    }
}

/* ------------------------------------------- 11. the server's wrappers */

static void test_http_wrappers(void) {
    section("server wrappers");

    /* The shapes server/main.c calls, on the adversarial body. */
    {
        const char *body =
            "{\"voice\":\"say \\\"input\\\": fake\",\"input\":\"real\","
            "\"stream\":true,\"temperature\":0.25,\"top_k\":80,\"seed\":42}";
        char text[64], voice[64];
        double d = 0.0;
        int b = 0;
        CHECK(mynah_json_string(body, "input", text, sizeof(text)) == 0 &&
              strcmp(text, "real") == 0, "input read as \"%s\"", text);
        CHECK(mynah_json_string(body, "voice", voice, sizeof(voice)) == 0 &&
              strcmp(voice, "say \"input\": fake") == 0,
              "voice read as \"%s\"", voice);
        CHECK(mynah_json_bool(body, "stream", &b) == 0 && b == 1, "stream");
        CHECK(mynah_json_number(body, "temperature", &d) == 0 && d == 0.25,
              "temperature read as %g", d);
        CHECK(mynah_json_number(body, "top_k", &d) == 0 && d == 80.0, "top_k");
        CHECK(mynah_json_number(body, "seed", &d) == 0 && d == 42.0, "seed");
        CHECK(mynah_json_string(body, "missing", text, sizeof(text)) != 0,
              "a missing key must fail");
        CHECK(mynah_json_bool(body, "temperature", &b) != 0,
              "a number is not a boolean");
        CHECK(mynah_json_number(body, "input", &d) != 0, "a string is not a number");
    }

    /* A nested key must not answer a top-level lookup here either. */
    {
        const char *body = "{\"options\":{\"input\":\"nested\"}}";
        char text[64];
        CHECK(mynah_json_string(body, "input", text, sizeof(text)) != 0,
              "a nested 'input' must not be served as the top-level one");
    }

    /* Emoji: the old reader refused every surrogate, so nothing outside the
     * BMP could ever reach the tokenizer. */
    {
        const char *body = "{\"input\":\"hi \\ud83d\\ude00 there\"}";
        char text[64];
        CHECK(mynah_json_string(body, "input", text, sizeof(text)) == 0 &&
              strcmp(text, "hi \xf0\x9f\x98\x80 there") == 0,
              "an emoji must survive the decode: got \"%s\"", text);
    }
    {
        const char *body = "{\"input\":\"\xf0\x9f\x8e\xb5 raw\"}";
        char text[64];
        CHECK(mynah_json_string(body, "input", text, sizeof(text)) == 0 &&
              strcmp(text, "\xf0\x9f\x8e\xb5 raw") == 0, "raw emoji passthrough");
    }

    /* A malformed body serves nothing at all -- and that is deliberate: a
     * reader that answers some keys of a broken document and not others makes
     * the failure depend on which key you asked for. */
    {
        const char *broken[] = {
            "{\"input\":\"a\",}",
            "{\"input\":\"a\"",
            "{\"input\":\"a\"} trailing",
            "{\"input\":\"a\",\"n\":01}",
            "{\"input\":\"a\" \"b\":1}",
            "",
        };
        for (size_t i = 0; i < sizeof(broken) / sizeof(broken[0]); ++i) {
            char text[64];
            double d = 0.0;
            int b = 0;
            CHECK(mynah_json_string(broken[i], "input", text, sizeof(text)) != 0,
                  "broken body %zu served a string", i);
            CHECK(mynah_json_number(broken[i], "n", &d) != 0,
                  "broken body %zu served a number", i);
            CHECK(mynah_json_bool(broken[i], "b", &b) != 0,
                  "broken body %zu served a bool", i);
        }
    }

    /* NULL and capacity handling, which the server relies on. */
    {
        char text[8];
        double d = 0.0;
        int b = 0;
        CHECK(mynah_json_string(NULL, "a", text, sizeof(text)) != 0, "NULL body");
        CHECK(mynah_json_string("{\"a\":1}", NULL, text, sizeof(text)) != 0,
              "NULL key");
        CHECK(mynah_json_string("{\"a\":\"x\"}", "a", NULL, 8) != 0, "NULL out");
        CHECK(mynah_json_string("{\"a\":\"x\"}", "a", text, 0) != 0, "zero capacity");
        CHECK(mynah_json_string("{\"a\":\"toolongforeight\"}", "a", text,
                                sizeof(text)) != 0, "an oversized value must fail");
        CHECK(mynah_json_number("{\"a\":1}", "a", NULL) != 0, "NULL number out");
        CHECK(mynah_json_bool("{\"a\":true}", "a", NULL) != 0, "NULL bool out");
        CHECK(mynah_json_number(NULL, "a", &d) != 0, "NULL body, number");
        CHECK(mynah_json_bool(NULL, "a", &b) != 0, "NULL body, bool");
    }

    /* A body that is not an object serves nothing. */
    {
        char text[16];
        CHECK(mynah_json_string("[{\"input\":\"x\"}]", "input", text,
                                sizeof(text)) != 0,
              "an array body must not answer a key lookup");
        CHECK(mynah_json_string("\"input\"", "input", text, sizeof(text)) != 0,
              "a string body must not answer a key lookup");
    }

    /* mynah_json_escape and the parser are inverses: whatever the server
     * writes into an error message, this parser reads back unchanged. */
    {
        static const char *samples[] = {
            "plain", "with \"quotes\"", "back\\slash", "line\nbreak",
            "tab\there", "carriage\rreturn", "control\x01\x02", "caf\xc3\xa9",
            "\xe4\xb8\xad\xe6\x96\x87", "\xf0\x9f\x98\x80", "",
            "this server holds 'en' and was asked for 'it'",
        };
        for (size_t i = 0; i < sizeof(samples) / sizeof(samples[0]); ++i) {
            char escaped[256], doc[320], back[256];
            CHECK(mynah_json_escape(samples[i], escaped, sizeof(escaped)) !=
                  (size_t)-1, "escape %zu", i);
            const int n = snprintf(doc, sizeof(doc), "{\"message\":\"%s\"}", escaped);
            CHECK(n > 0 && (size_t)n < sizeof(doc), "doc %zu", i);
            CHECK(mynah_json_string(doc, "message", back, sizeof(back)) == 0 &&
                  strcmp(back, samples[i]) == 0,
                  "round trip %zu gave \"%s\"", i, back);
        }
    }
}

/* --------------------------------------------- 12. byte-substitution sweep */

/* Not a fuzzer and not a correctness claim: a sweep that mutates one byte at a
 * time and walks whatever comes back. Its whole value is under ASan and UBSan,
 * where an out-of-bounds read or a shift on a signed char shows up as a report
 * rather than as a wrong answer nobody notices. Deterministic: no RNG. */
static void test_byte_sweep(void) {
    section("byte sweep");

    const char *base =
        "{\"engine\":\"magpie\",\"codec\":{\"levels\":[8,7,6,6]},"
        "\"s\":\"caf\xc3\xa9 \\ud83d\\ude00\",\"n\":-1.5e3,\"b\":true,\"z\":null}";
    const size_t len = strlen(base);
    static const unsigned char subs[] = {
        0x00, 0x01, 0x09, 0x0a, 0x20, 0x22, 0x2c, 0x2e, 0x30, 0x3a, 0x5b, 0x5c,
        0x5d, 0x65, 0x7b, 0x7d, 0x80, 0xc0, 0xe0, 0xed, 0xf4, 0xff
    };
    unsigned parsed = 0;
    for (size_t i = 0; i < len; ++i) {
        for (size_t s = 0; s < sizeof(subs); ++s) {
            char *copy = exact_copy(base, len);
            copy[i] = (char)subs[s];
            mynah_json_value root;
            mynah_json_error err;
            if (mynah_json_parse(copy, len, &root, &err) == 0) {
                ++parsed;
                /* Walk everything a caller could walk. */
                size_t count = 0;
                (void)mynah_json_count(&root, &count);
                if (root.type == MYNAH_JSON_OBJECT) {
                    size_t cursor = 0;
                    mynah_json_value name, value;
                    while (mynah_json_object_next(&root, &cursor, &name, &value) == 0) {
                        char buf[128];
                        (void)mynah_json_as_string(&name, buf, sizeof(buf));
                        (void)mynah_json_as_string(&value, buf, sizeof(buf));
                        double d = 0.0;
                        (void)mynah_json_as_number(&value, &d);
                        unsigned u = 0;
                        (void)mynah_json_as_unsigned(&value, &u);
                        int b = 0;
                        (void)mynah_json_as_bool(&value, &b);
                    }
                    mynah_json_value v;
                    (void)mynah_json_lookup(&root, "codec.levels[2]", &v);
                    (void)mynah_json_lookup(&root, "s", &v);
                }
            } else {
                CHECK(err.offset <= len, "mutation at %zu/%zu refused past the end",
                      i, s);
            }
            free(copy);
        }
    }
    /* A sweep that never parsed anything would be testing nothing. */
    CHECK(parsed > 0, "no mutation of the reference document parsed");
    printf("       (%u of %zu single-byte mutations still parsed)\n", parsed,
           len * sizeof(subs));
}

/* ---------------------------------------------------------------- main */

int main(void) {
    test_adversarial_keys();
    test_escapes();
    test_surrogates();
    test_numbers();
    test_depth();
    test_truncation();
    test_malformed();
    test_duplicates();
    test_bad_utf8();
    test_value_api();
    test_http_wrappers();
    test_byte_sweep();

    printf("json: %u checks, %u failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
