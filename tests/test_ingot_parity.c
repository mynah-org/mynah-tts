/* A/B gate for the ingot migration. Deleted when this branch is merged.
 *
 * Both readers are linked into this one binary — the prefixes do not collide —
 * and asked the same questions about the same file: how many tensors, and for
 * each name the rank, the shape, the element count and the payload BYTES.
 *
 * Head and tail of the payload are both compared: the head catches a wrong
 * offset, the tail catches a wrong length. Here the files are small enough to
 * compare whole, which is strictly stronger.
 *
 * The fixtures are built in a temp directory, so this runs with no model on
 * disk and belongs in CI.
 *
 * SPDX-License-Identifier: MIT */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "ingot/safetensors.h"
#include "legacy_safetensors.h"

static int checks = 0;
static int failures = 0;

#define CHECK(cond, ...)                                                       \
    do {                                                                       \
        checks++;                                                              \
        if (cond) {                                                            \
            printf("  ok:   ");                                                \
        } else {                                                               \
            failures++;                                                        \
            printf("  FAIL: ");                                                \
        }                                                                      \
        printf(__VA_ARGS__);                                                    \
        printf("\n");                                                          \
    } while (0)

/* Deterministic bytes: every fixture is reproducible from its index alone. */
static unsigned char byte_at(size_t i) {
    return (unsigned char)((i * 167u + 13u) ^ (i >> 3));
}

typedef struct {
    const char *name;
    const char *dtype;
    size_t      rank;
    size_t      shape[4];
    size_t      elem_size;
} fixture_tensor;

static size_t elements(const fixture_tensor *t) {
    size_t n = 1;
    for (size_t d = 0; d < t->rank; d++) n *= t->shape[d];
    return n;
}

/* Writes a safetensors file: u64 header length, JSON header, payload.
 * The header is padded with spaces so the data start is 8-aligned, which the
 * format's zero-copy readers rely on. */
static int write_fixture(const char *path, const fixture_tensor *tensors, size_t count) {
    char header[4096];
    size_t used = 0;
    size_t offset = 0;
    used += (size_t)snprintf(header + used, sizeof(header) - used, "{");
    for (size_t i = 0; i < count; i++) {
        const size_t nbytes = elements(&tensors[i]) * tensors[i].elem_size;
        used += (size_t)snprintf(header + used, sizeof(header) - used,
                                 "%s\"%s\":{\"dtype\":\"%s\",\"shape\":[",
                                 i > 0 ? "," : "", tensors[i].name, tensors[i].dtype);
        for (size_t d = 0; d < tensors[i].rank; d++)
            used += (size_t)snprintf(header + used, sizeof(header) - used, "%s%zu",
                                     d > 0 ? "," : "", tensors[i].shape[d]);
        used += (size_t)snprintf(header + used, sizeof(header) - used,
                                 "],\"data_offsets\":[%zu,%zu]}", offset, offset + nbytes);
        offset += nbytes;
    }
    used += (size_t)snprintf(header + used, sizeof(header) - used, "}");
    while ((used + 8u) % 8u != 0u) header[used++] = ' ';

    FILE *f = fopen(path, "wb");
    if (f == NULL) return -1;
    const uint64_t header_len = (uint64_t)used;
    unsigned char len_le[8];
    for (size_t i = 0; i < 8; i++) len_le[i] = (unsigned char)((header_len >> (8u * i)) & 0xffu);
    if (fwrite(len_le, 1, 8, f) != 8 || fwrite(header, 1, used, f) != used) {
        fclose(f);
        return -1;
    }
    for (size_t i = 0, written = 0; i < count; i++) {
        const size_t nbytes = elements(&tensors[i]) * tensors[i].elem_size;
        for (size_t b = 0; b < nbytes; b++, written++) {
            const unsigned char value = byte_at(written);
            if (fwrite(&value, 1, 1, f) != 1) {
                fclose(f);
                return -1;
            }
        }
    }
    return fclose(f) == 0 ? 0 : -1;
}

/* Every dtype the OLD reader accepts: F32, I32, I64, ranks 1 to 4. */
static const fixture_tensor SUPPORTED[] = {
    {"encoder.weight",     "F32", 2, {3, 5, 0, 0}, 4},
    {"encoder.bias",       "F32", 1, {7, 0, 0, 0}, 4},
    {"decoder.conv",       "F32", 3, {2, 3, 4, 0}, 4},
    {"decoder.deep",       "F32", 4, {2, 2, 3, 3}, 4},
    {"tokenizer.ids",      "I64", 1, {6, 0, 0, 0}, 8},
    {"tokenizer.offsets",  "I32", 2, {2, 3, 0, 0}, 4},
};
static const size_t SUPPORTED_COUNT = sizeof(SUPPORTED) / sizeof(SUPPORTED[0]);

static void gate_same_answers(const char *path) {
    char old_err[256] = {0};
    char new_err[256] = {0};

    mynah_safetensors *old_file = NULL;
    const int old_rc = mynah_safetensors_open(path, &old_file, old_err, sizeof old_err);
    CHECK(old_rc == 0 && old_file != NULL, "the old reader opens the fixture (%s)", old_err);

    ingot_st *new_file = NULL;
    const int new_rc = ingot_st_open(&new_file, path, new_err, sizeof new_err);
    CHECK(new_rc == 0 && new_file != NULL, "ingot opens the fixture (%s)", new_err);
    if (old_rc != 0 || new_rc != 0) return;

    /* 1. the same NUMBER of tensors. If this differs, one of them truncates. */
    CHECK(ingot_st_count(new_file) == SUPPORTED_COUNT,
          "ingot sees all %zu tensors (%zu)", SUPPORTED_COUNT, ingot_st_count(new_file));

    for (size_t i = 0; i < SUPPORTED_COUNT; i++) {
        const fixture_tensor *want = &SUPPORTED[i];

        mynah_tensor old_t;
        const int old_found = mynah_safetensors_get(old_file, want->name, &old_t);
        const ingot_st_tensor *new_t = ingot_st_find(new_file, want->name);
        CHECK(old_found == 0 && new_t != NULL, "%s: both readers find it", want->name);
        if (old_found != 0 || new_t == NULL) continue;

        /* 2. rank, shape and element count, per name. */
        CHECK(old_t.rank == (size_t)new_t->rank, "%s: same rank (%zu / %u)",
              want->name, old_t.rank, new_t->rank);
        int shape_ok = 1;
        for (size_t d = 0; d < old_t.rank && d < new_t->rank; d++)
            if (old_t.shape[d] != (size_t)new_t->shape[d]) shape_ok = 0;
        CHECK(shape_ok, "%s: same shape", want->name);
        CHECK(old_t.count == (size_t)new_t->nelem, "%s: same element count (%zu / %llu)",
              want->name, old_t.count, (unsigned long long)new_t->nelem);

        /* 3. the payload itself, byte for byte. The old reader hands back a
         *    `const float *` whatever the dtype is, so compare as raw bytes:
         *    that is the pointer the callers actually receive. */
        const size_t nbytes = elements(want) * want->elem_size;
        CHECK((size_t)new_t->nbytes == nbytes, "%s: same byte count (%llu / %zu)",
              want->name, (unsigned long long)new_t->nbytes, nbytes);
        const void *old_data = (const void *)old_t.data;
        const void *new_data = ingot_st_data(new_file, new_t);
        CHECK(old_data != NULL && new_data != NULL, "%s: both hand back a pointer", want->name);
        if (old_data != NULL && new_data != NULL)
            CHECK(memcmp(old_data, new_data, nbytes) == 0,
                  "%s: %zu payload bytes identical", want->name, nbytes);

        /* 4. and the bytes are the ones the fixture wrote, so a shared
         *    misreading of the container cannot pass this test. */
        if (want->elem_size == 4 && strcmp(want->dtype, "F32") == 0) {
            size_t base = 0;
            for (size_t j = 0; j < i; j++) base += elements(&SUPPORTED[j]) * SUPPORTED[j].elem_size;
            int matches_fixture = 1;
            const unsigned char *bytes = new_data;
            for (size_t b = 0; b < nbytes; b++)
                if (bytes[b] != byte_at(base + b)) matches_fixture = 0;
            CHECK(matches_fixture, "%s: matches the bytes the fixture wrote", want->name);
        }
    }

    /* 5. a name that is not there fails the same way in both. */
    mynah_tensor missing;
    CHECK(mynah_safetensors_get(old_file, "not.a.tensor", &missing) != 0 &&
          ingot_st_find(new_file, "not.a.tensor") == NULL,
          "a missing name is refused by both");

    mynah_safetensors_close(old_file);
    ingot_st_close(new_file);
}

/* The reason this migration is worth doing: a modern HF checkpoint is bf16,
 * and the old reader rejects the whole file at open. */
static void gate_bf16_is_the_gain(const char *path) {
    static const fixture_tensor BF16[] = {
        {"model.embed_tokens.weight", "BF16", 2, {4, 8, 0, 0}, 2},
    };
    if (write_fixture(path, BF16, 1) != 0) {
        CHECK(0, "could not write the bf16 fixture");
        return;
    }

    /* One buffer each: sharing them would print the old reader's complaint next
     * to ingot's result and read as if ingot had produced it. */
    char old_err[256] = {0};
    char new_err[256] = {0};

    mynah_safetensors *old_file = NULL;
    CHECK(mynah_safetensors_open(path, &old_file, old_err, sizeof old_err) != 0 && old_file == NULL,
          "the old reader still refuses bf16: %s", old_err);
    if (old_file != NULL) mynah_safetensors_close(old_file);

    ingot_st *new_file = NULL;
    CHECK(ingot_st_open(&new_file, path, new_err, sizeof new_err) == 0 && new_file != NULL,
          "ingot opens it%s%s", new_err[0] != '\0' ? ": " : "", new_err);
    if (new_file == NULL) return;

    const ingot_st_tensor *t = ingot_st_find(new_file, "model.embed_tokens.weight");
    CHECK(t != NULL && t->dtype == INGOT_DT_BF16, "and reports it as BF16");
    if (t != NULL) {
        float *converted = malloc((size_t)t->nelem * sizeof(float));
        CHECK(converted != NULL && ingot_st_to_f32(new_file, t, converted) == 0,
              "and converts it to f32 on request");
        free(converted);
    }
    ingot_st_close(new_file);
}

/* The same gate against a real checkpoint: every tensor in the file, not a
 * fixture. ingot enumerates, the old reader is asked for each name, and the
 * payload is compared head AND tail — the head catches a wrong offset, the tail
 * a wrong length. Whole-tensor memcmp for anything up to 1 MiB. */
static void gate_real_file(const char *path) {
    char old_err[256] = {0};
    char new_err[256] = {0};

    mynah_safetensors *old_file = NULL;
    ingot_st *new_file = NULL;
    const int old_rc = mynah_safetensors_open(path, &old_file, old_err, sizeof old_err);
    const int new_rc = ingot_st_open(&new_file, path, new_err, sizeof new_err);
    CHECK(old_rc == 0 && old_file != NULL, "%s: the old reader opens it (%s)", path, old_err);
    CHECK(new_rc == 0 && new_file != NULL, "%s: ingot opens it (%s)", path, new_err);
    if (old_rc != 0 || new_rc != 0) {
        if (old_file != NULL) mynah_safetensors_close(old_file);
        if (new_file != NULL) ingot_st_close(new_file);
        return;
    }

    const size_t total = ingot_st_count(new_file);
    size_t missing = 0, mismatched_shape = 0, mismatched_bytes = 0, compared = 0;
    unsigned long long compared_bytes = 0;

    for (size_t i = 0; i < total; i++) {
        const ingot_st_tensor *t = ingot_st_at(new_file, i);
        if (t == NULL) continue;

        mynah_tensor old_t;
        if (mynah_safetensors_get(old_file, t->name, &old_t) != 0) {
            if (missing == 0) printf("        first missing name: %s\n", t->name);
            missing++;
            continue;
        }

        int ok = old_t.rank == (size_t)t->rank && old_t.count == (size_t)t->nelem;
        for (size_t d = 0; ok && d < old_t.rank && d < 4u; d++)
            if (old_t.shape[d] != (size_t)t->shape[d]) ok = 0;
        if (!ok) {
            if (mismatched_shape == 0) printf("        first shape mismatch: %s\n", t->name);
            mismatched_shape++;
            continue;
        }

        const unsigned char *old_bytes = (const unsigned char *)old_t.data;
        const unsigned char *new_bytes = ingot_st_data(new_file, t);
        const size_t nbytes = (size_t)t->nbytes;
        if (old_bytes == NULL || new_bytes == NULL) {
            mismatched_bytes++;
            continue;
        }
        int same = 1;
        if (nbytes <= 1024u * 1024u) {
            same = memcmp(old_bytes, new_bytes, nbytes) == 0;
            compared_bytes += nbytes;
        } else {
            const size_t edge = 4096;
            same = memcmp(old_bytes, new_bytes, edge) == 0 &&
                   memcmp(old_bytes + nbytes - edge, new_bytes + nbytes - edge, edge) == 0;
            compared_bytes += 2u * edge;
        }
        if (!same) {
            if (mismatched_bytes == 0) printf("        first payload mismatch: %s\n", t->name);
            mismatched_bytes++;
            continue;
        }
        compared++;
    }

    CHECK(missing == 0, "%zu tensors: the old reader finds every name ingot lists (%zu missing)",
          total, missing);
    CHECK(mismatched_shape == 0, "rank, shape and element count agree on all %zu (%zu differ)",
          total, mismatched_shape);
    CHECK(mismatched_bytes == 0, "payloads agree on all %zu (%zu differ, %llu bytes compared)",
          total, mismatched_bytes, compared_bytes);
    CHECK(compared == total, "every tensor was actually compared (%zu / %zu)", compared, total);

    mynah_safetensors_close(old_file);
    ingot_st_close(new_file);
}

int main(int argc, char **argv) {
    /* Any argument is a real .safetensors (or a model pack directory's file) to
     * run the gate against, on top of the synthetic fixtures. */
    for (int a = 1; a < argc; a++) {
        printf("real checkpoint: %s\n", argv[a]);
        gate_real_file(argv[a]);
    }

    char dir[] = "/tmp/mynah_parity_XXXXXX";
    if (mkdtemp(dir) == NULL) {
        fprintf(stderr, "cannot create a temp directory\n");
        return 1;
    }
    char supported_path[512];
    char bf16_path[512];
    snprintf(supported_path, sizeof supported_path, "%s/supported.safetensors", dir);
    snprintf(bf16_path, sizeof bf16_path, "%s/bf16.safetensors", dir);

    printf("old reader vs ingot, same file, same binary\n");
    if (write_fixture(supported_path, SUPPORTED, SUPPORTED_COUNT) != 0) {
        fprintf(stderr, "cannot write the fixture\n");
        return 1;
    }
    gate_same_answers(supported_path);

    printf("what the migration buys\n");
    gate_bf16_is_the_gain(bf16_path);

    remove(supported_path);
    remove(bf16_path);
    rmdir(dir);

    printf("\n%d checks, %d failures\n", checks, failures);
    if (failures == 0) printf("PARITY GATE GREEN\n");
    return failures == 0 ? 0 : 1;
}
