#include "audio.h"
#include "kernels.h"
#include "mynah_tts.h"
#include "qmat.h"

#include <math.h>
#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}

static void usage(const char *program) {
    printf("Usage:\n");
    printf("  %s --self-test\n", program);
    printf("  %s --inspect MODEL_DIR\n", program);
    printf("  %s --write-test-wav OUTPUT.wav\n", program);
    printf("  %s --synthesize MODEL_DIR --tokens IDS --output OUTPUT.wav [options]\n", program);
    printf("      options: --speaker N --max-steps N --temperature F --topk N --seed N\n");
    printf("               --parallel --device cpu|metal|cuda\n");
    printf("  %s --gpu-self-test metal|cuda\n", program);
    printf("\nNative Magpie inference is CPU-first; Metal/CUDA are explicit build variants.\n");
}

static void print_info(const mynah_tts_model_info *info) {
    printf("engine=%s\n", info->engine);
    printf("revision=%s\n", info->revision);
    printf("dtype=%s\n", info->dtype);
    printf("sample_rate=%u\n", info->sample_rate);
    printf("frame_rate=%.3f\n", info->frame_rate);
    printf("frame_stacking_factor=%u\n", info->frame_stacking_factor);
    printf("codebook_count=%u\n", info->codebook_count);
    printf("codebook_size=%u\n", info->codebook_size);
    printf("audio_vocab_size=%u\n", info->audio_vocab_size);
    printf("audio_bos_id=%u\n", info->audio_bos_id);
    printf("audio_eos_id=%u\n", info->audio_eos_id);
    printf("hidden_dim=%u\n", info->hidden_dim);
    printf("encoder_layers=%u\n", info->encoder_layers);
    printf("decoder_layers=%u\n", info->decoder_layers);
    printf("local_transformer_layers=%u\n", info->local_transformer_layers);
    printf("speaker_count=%u\n", info->speaker_count);
    printf("text_max_length=%u\n", info->text_max_length);
    printf("text_vocab_size=%u\n", info->text_vocab_size);
    printf("max_decoder_steps=%u\n", info->max_decoder_steps);
    printf("default_temperature=%.3f\n", info->default_temperature);
    printf("default_topk=%u\n", info->default_topk);
    printf("min_generated_frames=%u\n", info->min_generated_frames);
    printf("device=%s\n", info->device);
}

static int parse_unsigned(const char *text, unsigned *out) {
    char *end = NULL;
    errno = 0;
    const unsigned long value = strtoul(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value > UINT_MAX) return -1;
    *out = (unsigned)value;
    return 0;
}

static int parse_float(const char *text, float *out) {
    char *end = NULL;
    errno = 0;
    const float value = strtof(text, &end);
    if (errno != 0 || end == text || *end != '\0' || !isfinite(value) || value < 0.0f) return -1;
    *out = value;
    return 0;
}

static int parse_seed(const char *text, uint64_t *out) {
    char *end = NULL;
    errno = 0;
    const unsigned long long value = strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0') return -1;
    *out = (uint64_t)value;
    return 0;
}

static int parse_device(const char *text, mynah_tts_device *out) {
    if (strcmp(text, "cpu") == 0) *out = MYNAH_TTS_DEVICE_CPU;
    else if (strcmp(text, "metal") == 0) *out = MYNAH_TTS_DEVICE_METAL;
    else if (strcmp(text, "cuda") == 0) *out = MYNAH_TTS_DEVICE_CUDA;
    else return -1;
    return 0;
}

static int parse_tokens(const char *text, int **tokens, size_t *count) {
    char *copy = (char *)malloc(strlen(text) + 1u);
    if (copy == NULL) return -1;
    strcpy(copy, text);
    size_t capacity = 16;
    int *values = (int *)malloc(capacity * sizeof(*values));
    if (values == NULL) {
        free(copy);
        return -1;
    }
    size_t used = 0;
    for (char *part = strtok(copy, ", \\t"); part != NULL; part = strtok(NULL, ", \\t")) {
        char *end = NULL;
        errno = 0;
        const long value = strtol(part, &end, 10);
        if (errno != 0 || end == part || *end != '\0' || value < 0 || value > INT_MAX) {
            free(copy);
            free(values);
            return -1;
        }
        if (used == capacity) {
            capacity *= 2u;
            int *next = (int *)realloc(values, capacity * sizeof(*values));
            if (next == NULL) {
                free(copy);
                free(values);
                return -1;
            }
            values = next;
        }
        values[used++] = (int)value;
    }
    free(copy);
    if (used == 0) {
        free(values);
        return -1;
    }
    *tokens = values;
    *count = used;
    return 0;
}

static int parse_normalized(const char *model_dir, const char *text,
                            int **tokens, size_t *count) {
    char path[4096];
    if (snprintf(path, sizeof(path), "%s/tokenizer/english_phoneme.tsv", model_dir) <= 0) return -1;
    FILE *vocab = fopen(path, "r");
    if (vocab == NULL) return -1;
    char *copy = (char *)malloc(strlen(text) + 1u);
    int *values = NULL;
    size_t used = 0;
    size_t capacity = 16;
    if (copy == NULL || (values = (int *)malloc(capacity * sizeof(*values))) == NULL) {
        free(copy);
        free(values);
        fclose(vocab);
        return -1;
    }
    strcpy(copy, text);
    for (char *part = strtok(copy, "|"); part != NULL; part = strtok(NULL, "|")) {
        const char *wanted = strcmp(part, "<space>") == 0 ? " " : part;
        rewind(vocab);
        char line[4096];
        int found = -1;
        while (fgets(line, sizeof(line), vocab) != NULL) {
            char *tab = strchr(line, '\t');
            if (tab == NULL) continue;
            *tab++ = '\0';
            char *end = tab + strlen(tab);
            while (end > tab && (end[-1] == '\n' || end[-1] == '\r')) --end;
            *end = '\0';
            if (strcmp(tab, wanted) == 0) {
                found = atoi(line);
                break;
            }
        }
        if (found < 0) {
            free(copy);
            free(values);
            fclose(vocab);
            return -1;
        }
        if (used == capacity) {
            capacity *= 2u;
            int *next = (int *)realloc(values, capacity * sizeof(*values));
            if (next == NULL) {
                free(copy);
                free(values);
                fclose(vocab);
                return -1;
            }
            values = next;
        }
        values[used++] = found;
    }
    free(copy);
    fclose(vocab);
    if (used == 0) {
        free(values);
        return -1;
    }
    *tokens = values;
    *count = used;
    return 0;
}

static int synthesize(int argc, char **argv) {
    const char *model_dir = NULL;
    const char *token_text = NULL;
    const char *normalized_text = NULL;
    const char *output_path = NULL;
    unsigned speaker = 0;
    unsigned max_steps = 0;
    float temperature = -1.0f;
    unsigned topk = 0;
    uint64_t seed = UINT64_C(42);
    int use_local = 1;
    mynah_tts_device device = MYNAH_TTS_DEVICE_CPU;
    for (int i = 2; i < argc; ++i) {
        if (strcmp(argv[i], "--tokens") == 0 && i + 1 < argc) token_text = argv[++i];
        else if (strcmp(argv[i], "--normalized") == 0 && i + 1 < argc) normalized_text = argv[++i];
        else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) output_path = argv[++i];
        else if (strcmp(argv[i], "--speaker") == 0 && i + 1 < argc && parse_unsigned(argv[++i], &speaker) == 0) {}
        else if (strcmp(argv[i], "--max-steps") == 0 && i + 1 < argc && parse_unsigned(argv[++i], &max_steps) == 0) {}
        else if (strcmp(argv[i], "--temperature") == 0 && i + 1 < argc && parse_float(argv[++i], &temperature) == 0) {}
        else if (strcmp(argv[i], "--topk") == 0 && i + 1 < argc && parse_unsigned(argv[++i], &topk) == 0) {}
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc && parse_seed(argv[++i], &seed) == 0) {}
        else if (strcmp(argv[i], "--parallel") == 0) use_local = 0;
        else if (strcmp(argv[i], "--device") == 0 && i + 1 < argc &&
                 parse_device(argv[++i], &device) == 0) {}
        else if (argv[i][0] != '-' && model_dir == NULL) model_dir = argv[i];
        else {
            fprintf(stderr, "invalid synthesis option: %s\n", argv[i]);
            return 2;
        }
    }
    if (model_dir == NULL || (token_text == NULL && normalized_text == NULL) || output_path == NULL ||
        (token_text != NULL && normalized_text != NULL)) {
        fprintf(stderr, "synthesis requires MODEL_DIR, one of --tokens/--normalized and --output\n");
        return 2;
    }
    int *tokens = NULL;
    size_t token_count = 0;
    if ((token_text != NULL && parse_tokens(token_text, &tokens, &token_count) != 0) ||
        (normalized_text != NULL && parse_normalized(model_dir, normalized_text, &tokens, &token_count) != 0)) {
        fprintf(stderr, "invalid token list\n");
        return 2;
    }
    mynah_tts_model *model = NULL;
    char error[256];
    const double load_start = now_seconds();
    if (mynah_tts_model_open_device(model_dir, device, &model, error, sizeof(error)) != 0) {
        fprintf(stderr, "model check failed: %s\n", error);
        free(tokens);
        return 1;
    }
    const double load_seconds = now_seconds() - load_start;
    mynah_tts_model_info info;
    mynah_tts_model_get_info(model, &info);
    if (normalized_text != NULL) {
        int *next = (int *)realloc(tokens, (token_count + 1u) * sizeof(*tokens));
        if (next == NULL) {
            fprintf(stderr, "out of memory appending text EOS\n");
            mynah_tts_model_close(model);
            free(tokens);
            return 1;
        }
        tokens = next;
        tokens[token_count++] = (int)(info.text_vocab_size - 1u);
    }
    mynah_tts_request request = {
        .text_ids = tokens,
        .text_length = token_count,
        .speaker = speaker,
        .max_steps = max_steps,
        .temperature = temperature,
        .topk = topk,
        .seed = seed,
        .use_local_transformer = use_local,
    };
    float *samples = NULL;
    size_t sample_count = 0;
    const double synth_start = now_seconds();
    const int result = mynah_tts_synthesize(model, &request, &samples, &sample_count,
                                             error, sizeof(error));
    const double synth_seconds = now_seconds() - synth_start;
    if (result == 0 && mynah_wav_write_f32(output_path, samples, sample_count,
                                           info.sample_rate, error, sizeof(error)) == 0) {
        const double audio_seconds = info.sample_rate > 0
            ? (double)sample_count / (double)info.sample_rate : 0.0;
        printf("wrote native WAV: %s (%zu samples at %u Hz)\n", output_path,
               sample_count, info.sample_rate);
        printf("timing: load=%.3fs synth=%.3fs audio=%.3fs RTF=%.3f\n",
               load_seconds, synth_seconds, audio_seconds,
               audio_seconds > 0.0 ? synth_seconds / audio_seconds : 0.0);
    } else {
        fprintf(stderr, "native synthesis failed: %s\n", error);
    }
    mynah_tts_free_samples(samples);
    mynah_tts_model_close(model);
    free(tokens);
    return result == 0 ? 0 : 1;
}

static int write_test_wav(const char *path) {
    const unsigned sample_rate = 22050;
    const size_t count = sample_rate / 10u;
    float *samples = (float *)malloc(count * sizeof(*samples));
    if (samples == NULL) {
        fprintf(stderr, "out of memory creating WAV smoke samples\n");
        return 1;
    }
    for (size_t i = 0; i < count; ++i) {
        samples[i] = 0.1f * sinf(2.0f * 3.14159265358979323846f * 440.0f *
                                  (float)i / (float)sample_rate);
    }
    char error[256];
    const int result = mynah_wav_write_f32(path, samples, count, sample_rate,
                                           error, sizeof(error));
    free(samples);
    if (result != 0) {
        fprintf(stderr, "%s\n", error);
        return 1;
    }
    printf("wrote synthetic WAV smoke file: %s\n", path);
    return 0;
}

int main(int argc, char **argv) {
    if (argc == 1 || strcmp(argv[1], "--help") == 0 ||
        strcmp(argv[1], "-h") == 0) {
        usage(argv[0]);
        return 0;
    }
    if (strcmp(argv[1], "--version") == 0) {
        puts(MYNAH_TTS_VERSION);
        return 0;
    }
    if (strcmp(argv[1], "--self-test") == 0) {
        char error[256];
        if (mynah_kernels_self_test(error, sizeof(error)) != 0) {
            fprintf(stderr, "self-test failed: %s\n", error);
            return 1;
        }
        if (mynah_qmat_self_test(error, sizeof(error)) != 0) {
            fprintf(stderr, "qmat self-test failed: %s\n", error);
            return 1;
        }
        if (mynah_tts_device_self_test(MYNAH_TTS_DEVICE_CPU, error, sizeof(error)) != 0) {
            fprintf(stderr, "CPU backend self-test failed: %s\n", error);
            return 1;
        }
        puts("CPU SIMD/scalar backend self-test: PASS");
        return 0;
    }
    if (strcmp(argv[1], "--gpu-self-test") == 0 && argc == 3) {
        mynah_tts_device device;
        char error[256];
        if (parse_device(argv[2], &device) != 0 || device == MYNAH_TTS_DEVICE_CPU) {
            fprintf(stderr, "GPU self-test requires metal or cuda\n");
            return 2;
        }
        if (mynah_tts_device_self_test(device, error, sizeof(error)) != 0) {
            fprintf(stderr, "%s backend self-test failed: %s\n",
                    mynah_tts_device_name(device), error);
            return 1;
        }
        printf("%s backend self-test: PASS\n", mynah_tts_device_name(device));
        return 0;
    }
    if (strcmp(argv[1], "--synthesize") == 0) return synthesize(argc, argv);
    if (strcmp(argv[1], "--write-test-wav") == 0 && argc == 3) {
        return write_test_wav(argv[2]);
    }
    if (strcmp(argv[1], "--inspect") == 0 && argc == 3) {
        mynah_tts_model *model = NULL;
        char error[256];
        if (mynah_tts_model_open(argv[2], &model, error, sizeof(error)) != 0) {
            fprintf(stderr, "model check failed: %s\n", error);
            return 1;
        }
        mynah_tts_model_info info;
        mynah_tts_model_get_info(model, &info);
        print_info(&info);
        mynah_tts_model_close(model);
        return 0;
    }
    usage(argv[0]);
    return 2;
}
