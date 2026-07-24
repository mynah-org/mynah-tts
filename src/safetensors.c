#include "safetensors.h"

#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

typedef struct {
    char *name;
    uint64_t hash;
    size_t rank;
    size_t shape[4];
    size_t offset;
    size_t length;
    size_t element_size;
} tensor_entry;

struct mynah_safetensors {
    int fd;
    unsigned char *mapping;
    size_t file_size;
    size_t data_start;
    tensor_entry *entries;
    size_t entry_count;
    size_t entry_capacity;
    size_t *hash_slots;
    size_t hash_capacity;
};

static uint64_t hash_name(const char *name) {
    uint64_t hash = UINT64_C(1469598103934665603);
    for (const unsigned char *p = (const unsigned char *)name; *p != 0; ++p) {
        hash ^= (uint64_t)*p;
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static void set_error(char *error, size_t capacity, const char *message) {
    if (error != NULL && capacity > 0) snprintf(error, capacity, "%s", message);
}

static const char *skip_space(const char *p, const char *end) {
    while (p < end && (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n')) ++p;
    return p;
}

static const char *parse_string(const char *p, const char *end,
                                const char **begin, size_t *length) {
    if (p >= end || *p != '"') return NULL;
    ++p;
    const char *start = p;
    while (p < end) {
        if (*p == '\\') {
            ++p;
            if (p >= end) return NULL;
        } else if (*p == '"') {
            *begin = start;
            *length = (size_t)(p - start);
            return p + 1;
        }
        ++p;
    }
    return NULL;
}

static const char *skip_value(const char *p, const char *end) {
    p = skip_space(p, end);
    if (p >= end) return NULL;
    if (*p == '"') {
        const char *ignored = NULL;
        size_t ignored_length = 0;
        return parse_string(p, end, &ignored, &ignored_length);
    }
    if (*p != '{' && *p != '[') {
        while (p < end && *p != ',' && *p != '}' && *p != ']') ++p;
        return p;
    }
    const char open = *p;
    const char close = open == '{' ? '}' : ']';
    int depth = 0;
    int in_string = 0;
    for (; p < end; ++p) {
        if (in_string) {
            if (*p == '\\') {
                ++p;
            } else if (*p == '"') {
                in_string = 0;
            }
        } else if (*p == '"') {
            in_string = 1;
        } else if (*p == open) {
            ++depth;
        } else if (*p == close && --depth == 0) {
            return p + 1;
        }
    }
    return NULL;
}

static int parse_number(const char **cursor, const char *end, size_t *out) {
    const char *p = skip_space(*cursor, end);
    if (p >= end || *p < '0' || *p > '9') return -1;
    size_t value = 0;
    while (p < end && *p >= '0' && *p <= '9') {
        const size_t digit = (size_t)(*p - '0');
        if (value > (SIZE_MAX - digit) / 10u) return -1;
        value = value * 10u + digit;
        ++p;
    }
    *cursor = p;
    *out = value;
    return 0;
}

static int parse_number_array(const char **cursor, const char *end,
                              size_t *values, size_t *count) {
    const char *p = skip_space(*cursor, end);
    if (p >= end || *p != '[') return -1;
    ++p;
    size_t used = 0;
    for (;;) {
        p = skip_space(p, end);
        if (p < end && *p == ']') {
            ++p;
            *cursor = p;
            *count = used;
            return 0;
        }
        if (used >= 4 || parse_number(&p, end, &values[used]) != 0) return -1;
        ++used;
        p = skip_space(p, end);
        if (p >= end) return -1;
        if (*p == ',') {
            ++p;
        } else if (*p != ']') {
            return -1;
        }
    }
}

static int add_entry(mynah_safetensors *file, const char *name, size_t name_length,
                     const tensor_entry *entry) {
    if (file->entry_count == file->entry_capacity) {
        const size_t next_capacity = file->entry_capacity == 0 ? 64 : file->entry_capacity * 2;
        tensor_entry *next = (tensor_entry *)realloc(
            file->entries, next_capacity * sizeof(*next));
        if (next == NULL) return -1;
        file->entries = next;
        file->entry_capacity = next_capacity;
    }
    char *copy = (char *)malloc(name_length + 1u);
    if (copy == NULL) return -1;
    memcpy(copy, name, name_length);
    copy[name_length] = '\0';
    file->entries[file->entry_count] = *entry;
    file->entries[file->entry_count].name = copy;
    file->entries[file->entry_count].hash = hash_name(copy);
    ++file->entry_count;
    return 0;
}

static int build_hash_index(mynah_safetensors *file, char *error, size_t error_capacity) {
    if (file->entry_count > SIZE_MAX / 2u) {
        set_error(error, error_capacity, "safetensors index is too large");
        return -1;
    }
    size_t capacity = 16;
    while (capacity < file->entry_count * 2u) {
        if (capacity > SIZE_MAX / 2u) {
            set_error(error, error_capacity, "safetensors index is too large");
            return -1;
        }
        capacity *= 2u;
    }
    file->hash_slots = (size_t *)calloc(capacity, sizeof(*file->hash_slots));
    if (file->hash_slots == NULL) {
        set_error(error, error_capacity, "out of memory indexing safetensors names");
        return -1;
    }
    file->hash_capacity = capacity;
    const size_t mask = capacity - 1u;
    for (size_t index = 0; index < file->entry_count; ++index) {
        size_t slot = (size_t)file->entries[index].hash & mask;
        while (file->hash_slots[slot] != 0) slot = (slot + 1u) & mask;
        file->hash_slots[slot] = index + 1u;
    }
    return 0;
}

static int parse_header(mynah_safetensors *file, const char *header,
                        size_t header_length, char *error, size_t error_capacity) {
    const char *p = header;
    const char *end = header + header_length;
    p = skip_space(p, end);
    if (p >= end || *p++ != '{') {
        set_error(error, error_capacity, "invalid safetensors header");
        return -1;
    }
    for (;;) {
        p = skip_space(p, end);
        if (p >= end) break;
        if (*p == '}') return 0;
        const char *name = NULL;
        size_t name_length = 0;
        p = parse_string(p, end, &name, &name_length);
        if (p == NULL) break;
        p = skip_space(p, end);
        if (p >= end || *p++ != ':') break;
        p = skip_space(p, end);
        if (name_length == 12 && memcmp(name, "__metadata__", 12) == 0) {
            p = skip_value(p, end);
            if (p == NULL) break;
        } else if (p >= end || *p != '{') {
            p = skip_value(p, end);
            if (p == NULL) break;
        } else {
            ++p;
            tensor_entry entry;
            memset(&entry, 0, sizeof(entry));
            char dtype[8] = {0};
            size_t offset_values[2] = {0, 0};
            size_t offset_count = 0;
            int have_dtype = 0;
            int have_shape = 0;
            int have_offsets = 0;
            for (;;) {
                p = skip_space(p, end);
                if (p >= end) break;
                if (*p == '}') {
                    ++p;
                    break;
                }
                const char *field = NULL;
                size_t field_length = 0;
                p = parse_string(p, end, &field, &field_length);
                if (p == NULL) break;
                p = skip_space(p, end);
                if (p >= end || *p++ != ':') break;
                p = skip_space(p, end);
                if (field_length == 5 && memcmp(field, "dtype", 5) == 0) {
                    const char *dtype_value = NULL;
                    size_t dtype_length = 0;
                    p = parse_string(p, end, &dtype_value, &dtype_length);
                    if (p == NULL || dtype_length >= sizeof(dtype)) break;
                    memcpy(dtype, dtype_value, dtype_length);
                    dtype[dtype_length] = '\0';
                    have_dtype = 1;
                } else if (field_length == 5 && memcmp(field, "shape", 5) == 0) {
                    if (parse_number_array(&p, end, entry.shape, &entry.rank) != 0) break;
                    have_shape = 1;
                } else if (field_length == 12 && memcmp(field, "data_offsets", 12) == 0) {
                    if (parse_number_array(&p, end, offset_values, &offset_count) != 0 || offset_count != 2) break;
                    have_offsets = 1;
                } else {
                    p = skip_value(p, end);
                    if (p == NULL) break;
                }
                p = skip_space(p, end);
                if (p < end && *p == ',') ++p;
            }
            size_t element_size = 0;
            if (strcmp(dtype, "F32") == 0) element_size = sizeof(float);
            else if (strcmp(dtype, "I64") == 0) element_size = sizeof(int64_t);
            else if (strcmp(dtype, "I32") == 0) element_size = sizeof(int32_t);
            if (p > end || !have_dtype || !have_shape || !have_offsets || element_size == 0 ||
                offset_values[1] < offset_values[0]) {
                snprintf(error, error_capacity, "unsupported tensor in safetensors header: %.*s dtype=%s fields=%d%d%d",
                         (int)name_length, name, dtype, have_dtype, have_shape, have_offsets);
                return -1;
            }
            entry.offset = offset_values[0];
            entry.length = offset_values[1] - offset_values[0];
            size_t count = 1;
            for (size_t i = 0; i < entry.rank; ++i) {
                if (entry.shape[i] != 0 && count > SIZE_MAX / entry.shape[i]) {
                    set_error(error, error_capacity, "tensor shape overflows size_t");
                    return -1;
                }
                count *= entry.shape[i];
            }
            if (count > SIZE_MAX / element_size || entry.length != count * element_size ||
                entry.offset > file->file_size - file->data_start ||
                entry.length > file->file_size - file->data_start - entry.offset) {
                set_error(error, error_capacity, "tensor data is outside safetensors file");
                return -1;
            }
            entry.element_size = element_size;
            if (add_entry(file, name, name_length, &entry) != 0) {
                set_error(error, error_capacity, "out of memory indexing safetensors");
                return -1;
            }
        }
        p = skip_space(p, end);
        if (p < end && *p == ',') ++p;
    }
    set_error(error, error_capacity, "truncated safetensors header");
    return -1;
}

int mynah_safetensors_open(const char *path, mynah_safetensors **out,
                           char *error, size_t error_capacity) {
    if (out != NULL) *out = NULL;
    if (path == NULL || out == NULL) {
        set_error(error, error_capacity, "invalid safetensors arguments");
        return -1;
    }
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        snprintf(error, error_capacity, "cannot open safetensors: %s", path);
        return -1;
    }
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size < 8) {
        close(fd);
        set_error(error, error_capacity, "safetensors file is too small");
        return -1;
    }
    const size_t file_size = (size_t)st.st_size;
    unsigned char *mapping = (unsigned char *)mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapping == MAP_FAILED) {
        close(fd);
        set_error(error, error_capacity, "cannot mmap safetensors");
        return -1;
    }
    uint64_t header_length_u64 = 0;
    memcpy(&header_length_u64, mapping, sizeof(header_length_u64));
    if (header_length_u64 > file_size - 8u) {
        munmap(mapping, file_size);
        close(fd);
        set_error(error, error_capacity, "invalid safetensors header length");
        return -1;
    }
    mynah_safetensors *file = (mynah_safetensors *)calloc(1, sizeof(*file));
    if (file == NULL) {
        munmap(mapping, file_size);
        close(fd);
        set_error(error, error_capacity, "out of memory opening safetensors");
        return -1;
    }
    file->fd = fd;
    file->mapping = mapping;
    file->file_size = file_size;
    file->data_start = 8u + (size_t)header_length_u64;
    if (parse_header(file, (const char *)(mapping + 8u), (size_t)header_length_u64,
                     error, error_capacity) != 0) {
        mynah_safetensors_close(file);
        return -1;
    }
    if (build_hash_index(file, error, error_capacity) != 0) {
        mynah_safetensors_close(file);
        return -1;
    }
    *out = file;
    if (error != NULL && error_capacity > 0) error[0] = '\0';
    return 0;
}

void mynah_safetensors_close(mynah_safetensors *file) {
    if (file == NULL) return;
    for (size_t i = 0; i < file->entry_count; ++i) free(file->entries[i].name);
    free(file->entries);
    free(file->hash_slots);
    if (file->mapping != NULL) munmap(file->mapping, file->file_size);
    if (file->fd >= 0) close(file->fd);
    free(file);
}

int mynah_safetensors_get(const mynah_safetensors *file, const char *name,
                          mynah_tensor *out) {
    if (file == NULL || name == NULL || out == NULL) return -1;
    if (file->hash_capacity == 0) return -1;
    const uint64_t hash = hash_name(name);
    const size_t mask = file->hash_capacity - 1u;
    size_t slot = (size_t)hash & mask;
    for (;;) {
        const size_t stored = file->hash_slots[slot];
        if (stored == 0) return -1;
        const tensor_entry *entry = &file->entries[stored - 1u];
        if (entry->hash == hash && strcmp(entry->name, name) == 0) {
            out->data = (const float *)(file->mapping + file->data_start + entry->offset);
            out->rank = entry->rank;
            memcpy(out->shape, entry->shape, sizeof(out->shape));
            out->count = 1;
            for (size_t d = 0; d < entry->rank; ++d) out->count *= entry->shape[d];
            return 0;
        }
        slot = (slot + 1u) & mask;
    }
}
