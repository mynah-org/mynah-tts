/* PocketTTS text segmentation against the upstream algorithm.
 *
 *   tests/test_text_segment MODEL.model CASES.jsonl
 *
 * The cases come from tools/oracle_pocket_segments.py, which replays upstream's
 * split_into_best_sentences with the real SentencePiece model and records the
 * token count of every chunk. src/text_segment.c must produce exactly those
 * lengths, and the ids it returns must be each chunk's own encoding: their sum
 * equals the concatenation it hands the engine. */
#include "text_segment.h"
#include "tokenizer_sentencepiece.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int hex_value(int c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s MODEL.model CASES.jsonl\n", argv[0]);
        return 2;
    }
    char error[512];
    mynah_sp *sp = NULL;
    if (mynah_sp_open(argv[1], &sp, error, sizeof(error)) != 0) {
        fprintf(stderr, "cannot open tokenizer: %s\n", error);
        return 1;
    }
    FILE *f = fopen(argv[2], "r");
    if (f == NULL) {
        fprintf(stderr, "cannot open cases: %s\n", argv[2]);
        mynah_sp_close(sp);
        return 1;
    }
    const size_t capacity = 1u << 16;
    char *line = (char *)malloc(capacity);
    char *text = (char *)malloc(capacity / 2u + 1u);
    if (line == NULL || text == NULL) {
        fprintf(stderr, "out of memory\n");
        return 1;
    }
    size_t cases = 0, multi = 0, failures = 0;
    while (fgets(line, (int)capacity, f) != NULL) {
        const char *hex = strstr(line, "\"hex\": \"");
        const char *lens = strstr(line, "\"lengths\": [");
        if (hex == NULL || lens == NULL) continue;
        hex += 8;
        lens += 12;
        size_t n = 0;
        while (hex[2 * n] != '"' && hex[2 * n] != '\0') {
            const int hi = hex_value((unsigned char)hex[2 * n]);
            const int lo = hex_value((unsigned char)hex[2 * n + 1]);
            if (hi < 0 || lo < 0) break;
            text[n++] = (char)((hi << 4) | lo);
        }
        text[n] = '\0';
        size_t expected[64], expected_count = 0, expected_sum = 0;
        for (const char *p = lens; *p != '\0' && *p != ']' && expected_count < 64u;) {
            char *end = NULL;
            const long v = strtol(p, &end, 10);
            if (end == p) { ++p; continue; }
            expected[expected_count++] = (size_t)v;
            expected_sum += (size_t)v;
            p = end;
        }
        ++cases;
        if (expected_count > 1u) ++multi;

        int *ids = NULL;
        size_t *got = NULL;
        size_t count = 0, segments = 0;
        if (mynah_text_segment(sp, text, 50u, 0u, &ids, &count, &got, &segments, error,
                               sizeof(error)) != 0) {
            fprintf(stderr, "case %zu: segmentation failed: %s\n", cases, error);
            ++failures;
            continue;
        }
        int bad = segments != expected_count || count != expected_sum;
        for (size_t s = 0; !bad && s < segments; ++s) bad = got[s] != expected[s];
        if (bad) {
            fprintf(stderr, "case %zu: got %zu segments (", cases, segments);
            for (size_t s = 0; s < segments; ++s) fprintf(stderr, "%s%zu", s ? " " : "", got[s]);
            fprintf(stderr, "), upstream %zu (", expected_count);
            for (size_t s = 0; s < expected_count; ++s)
                fprintf(stderr, "%s%zu", s ? " " : "", expected[s]);
            fprintf(stderr, ") for: %.80s\n", text);
            ++failures;
        }
        free(ids);
        free(got);
    }
    fclose(f);
    free(line);
    free(text);
    mynah_sp_close(sp);
    printf("text segmentation: %zu cases (%zu multi-segment), %zu failures -- %s\n",
           cases, multi, failures, failures == 0u ? "PASS" : "FAIL");
    return failures == 0u && cases > 0u ? 0 : 1;
}
