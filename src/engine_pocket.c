/* PocketTTS engine. Contract, measurements and known limits: engine_pocket.h.
 *
 * SPDX-License-Identifier: MIT */
#include "engine_pocket.h"

#include <errno.h>
#include <limits.h>
#include <math.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ingot/safetensors.h"

#include "costmap.h"
#include "dispatch.h"
#include "flow_head.h"
#include "kernels.h"
#include "mynah_tts_internal.h"
#include "mynah_util.h"
#include "qmat.h"
#include "seanet.h"
#include "tokenizer_sentencepiece.h"
#include "voice_clone.h"
#include "transformer_ar.h"
#include "weights.h"

#define POCKET_ENGINE_NAME "pocket"
#define POCKET_MAX_RATIOS 8u
#define POCKET_NAME_MAX 256u
#define POCKET_PATH_MAX 4096u
#define POCKET_MANIFEST_MAX (4u * 1024u * 1024u)
#define POCKET_QNAME_MAX 48u
/* How many requests one resident backbone pass may serve.  CUDA's pointer
 * metadata arena is sized for 64 rows; keeping the engine and driver ceilings
 * equal lets a GPU worker form one real batch at C64 instead of four C16
 * microbatches.  CPU still remains correct at the wider width, while its
 * operator can choose a smaller --max-batch at serving time. */
#define POCKET_MAX_BATCH 64u

/* ------------------------------------------------------------------ errors */

#if defined(__GNUC__)
static void pocket_error(char *error, size_t capacity, const char *format, ...)
    __attribute__((format(printf, 3, 4)));
#endif

static void pocket_error(char *error, size_t capacity, const char *format, ...) {
    if (error == NULL || capacity == 0) return;
    va_list args;
    va_start(args, format);
    vsnprintf(error, capacity, format, args);
    va_end(args);
}

static int pocket_mul(size_t a, size_t b, size_t *out) {
    if (a != 0 && b > SIZE_MAX / a) return -1;
    *out = a * b;
    return 0;
}

static int pocket_add(size_t a, size_t b, size_t *out) {
    if (a > SIZE_MAX - b) return -1;
    *out = a + b;
    return 0;
}

static char *pocket_strdup(const char *value, size_t length) {
    char *copy = (char *)malloc(length + 1u);
    if (copy == NULL) return NULL;
    memcpy(copy, value, length);
    copy[length] = '\0';
    return copy;
}

/* ------------------------------------------------------------------- JSON
 *
 * A manifest reader, not a JSON library: it walks the document structurally
 * (so a key nested inside another object is never mistaken for a top-level
 * one) and it never allocates. `substr`-style scanning was the alternative and
 * it silently confuses "weights"."tts" with a top-level "tts".
 */

static const char *pj_ws(const char *p) {
    while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') ++p;
    return p;
}

/* `p` at the opening quote; returns just past the closing quote, or NULL. */
static const char *pj_end_string(const char *p) {
    if (*p != '"') return NULL;
    ++p;
    while (*p != '\0') {
        if (*p == '\\') {
            if (p[1] == '\0') return NULL;
            p += 2;
            continue;
        }
        if (*p == '"') return p + 1;
        ++p;
    }
    return NULL;
}

static const char *pj_end_value(const char *p) {
    p = pj_ws(p);
    if (*p == '"') return pj_end_string(p);
    if (*p == '{' || *p == '[') {
        /* Only one bracket kind is counted. JSON nests properly, so the other
         * kind is always balanced inside and cannot unbalance this count. */
        const char open = *p;
        const char close = (open == '{') ? '}' : ']';
        size_t depth = 0;
        while (*p != '\0') {
            if (*p == '"') {
                const char *s = pj_end_string(p);
                if (s == NULL) return NULL;
                p = s;
                continue;
            }
            if (*p == open) {
                ++depth;
            } else if (*p == close) {
                --depth;
                if (depth == 0) return p + 1;
            }
            ++p;
        }
        return NULL;
    }
    while (*p != '\0' && *p != ',' && *p != '}' && *p != ']' && *p != ' ' &&
           *p != '\t' && *p != '\r' && *p != '\n') {
        ++p;
    }
    return p;
}

/* The value of `key` in the object starting at `object`, or NULL. */
static const char *pj_object_get(const char *object, const char *key) {
    if (object == NULL || key == NULL) return NULL;
    const char *p = pj_ws(object);
    if (*p != '{') return NULL;
    ++p;
    const size_t key_length = strlen(key);
    for (;;) {
        p = pj_ws(p);
        if (*p != '"') return NULL;
        const char *name = p + 1;
        const char *name_end = pj_end_string(p);
        if (name_end == NULL) return NULL;
        const size_t name_length = (size_t)(name_end - name) - 1u;
        p = pj_ws(name_end);
        if (*p != ':') return NULL;
        const char *value = pj_ws(p + 1);
        if (name_length == key_length && memcmp(name, key, key_length) == 0) {
            return value;
        }
        p = pj_end_value(value);
        if (p == NULL) return NULL;
        p = pj_ws(p);
        if (*p != ',') return NULL;
        ++p;
    }
}

static const char *pj_array_first(const char *value) {
    if (value == NULL) return NULL;
    const char *p = pj_ws(value);
    if (*p != '[') return NULL;
    p = pj_ws(p + 1);
    return (*p == ']' || *p == '\0') ? NULL : p;
}

static const char *pj_array_next(const char *element) {
    const char *p = pj_end_value(element);
    if (p == NULL) return NULL;
    p = pj_ws(p);
    if (*p != ',') return NULL;
    p = pj_ws(p + 1);
    return (*p == ']' || *p == '\0') ? NULL : p;
}

static int pj_number(const char *value, double *out) {
    if (value == NULL) return -1;
    char *end = NULL;
    errno = 0;
    const double parsed = strtod(value, &end);
    if (errno != 0 || end == value) return -1;
    *out = parsed;
    return 0;
}

static int pj_bool(const char *value, int *out) {
    if (value == NULL) return -1;
    if (strncmp(value, "true", 4) == 0) {
        *out = 1;
        return 0;
    }
    if (strncmp(value, "false", 5) == 0) {
        *out = 0;
        return 0;
    }
    return -1;
}

static int pj_string_copy(const char *value, char *out, size_t capacity) {
    if (value == NULL || out == NULL || capacity == 0 || *value != '"') return -1;
    ++value;
    size_t used = 0;
    while (*value != '\0' && *value != '"') {
        char c = *value++;
        if (c == '\\') {
            const char escape = *value++;
            switch (escape) {
                case '"': c = '"'; break;
                case '\\': c = '\\'; break;
                case '/': c = '/'; break;
                case 'n': c = '\n'; break;
                case 't': c = '\t'; break;
                case 'r': c = '\r'; break;
                default: return -1; /* \u and friends: refuse, never guess */
            }
        }
        if (used + 1u >= capacity) return -1;
        out[used++] = c;
    }
    if (*value != '"') return -1;
    out[used] = '\0';
    return 0;
}

/* ------------------------------------------------------- manifest getters */

static int cfg_size_value(const char *value, size_t *out) {
    double number = 0.0;
    if (pj_number(value, &number) != 0) return -1;
    if (!(number >= 0.0) || number > (double)(SIZE_MAX / 2u)) return -1;
    const double rounded = floor(number + 0.5);
    if (fabs(number - rounded) > 1e-6) return -1;
    *out = (size_t)rounded;
    return 0;
}

static int cfg_need_size(const char *root, const char *key, size_t *out,
                         char *error, size_t capacity) {
    if (cfg_size_value(pj_object_get(root, key), out) != 0) {
        pocket_error(error, capacity,
                     "model.json: \"%s\" is missing or not a whole number", key);
        return -1;
    }
    return 0;
}

static void cfg_opt_size(const char *root, const char *key, size_t *out,
                         size_t fallback) {
    if (cfg_size_value(pj_object_get(root, key), out) != 0) *out = fallback;
}

static int cfg_need_double(const char *root, const char *key, double *out,
                           char *error, size_t capacity) {
    if (pj_number(pj_object_get(root, key), out) != 0) {
        pocket_error(error, capacity, "model.json: \"%s\" is missing or not a number",
                     key);
        return -1;
    }
    return 0;
}

static void cfg_opt_double(const char *root, const char *key, double *out,
                           double fallback) {
    if (pj_number(pj_object_get(root, key), out) != 0) *out = fallback;
}

static void cfg_opt_bool(const char *root, const char *key, int *out, int fallback) {
    if (pj_bool(pj_object_get(root, key), out) != 0) *out = fallback;
}

static void cfg_opt_string(const char *root, const char *key, char *out,
                           size_t capacity, const char *fallback) {
    if (pj_string_copy(pj_object_get(root, key), out, capacity) != 0) {
        snprintf(out, capacity, "%s", fallback);
    }
}

/* ------------------------------------------------------------ the manifest */

typedef struct {
    /* backbone */
    size_t hidden_dim;
    size_t heads;
    size_t head_dim;
    size_t layers;
    size_t ffn_dim;
    float layernorm_eps;

    /* text conditioning */
    size_t embed_rows;
    size_t vocab_size;
    size_t padding_id;
    size_t max_tokens_per_chunk;

    /* latent space and flow head */
    size_t latent_dim;
    size_t flow_dim;
    size_t flow_depth;
    size_t flow_time_conds;
    size_t flow_freqs;
    size_t flow_decode_steps;
    float flow_layernorm_eps;
    float eos_threshold;
    float temperature;

    /* codec */
    size_t codec_dim;
    size_t ratios[POCKET_MAX_RATIOS];
    size_t n_ratios;
    size_t upsample_stride;
    /* The ENCODER's stride, for cloning only. The decoder never needs it, so it
     * is optional and defaults to the upsample stride it mirrors. */
    size_t downsample_stride;
    size_t codec_tf_dim;
    size_t codec_tf_ffn;
    size_t codec_tf_heads;
    size_t codec_tf_layers;
    size_t codec_tf_context;
    size_t audio_channels;
    size_t samples_per_frame;

    /* SEANet shape, derived from the tensors unless the manifest declares it */
    size_t n_filters;
    size_t n_residual_layers;
    size_t compress;
    size_t kernel_size;
    size_t residual_kernel_size;
    size_t last_kernel_size;
    size_t dilation_base;
    float elu_alpha;

    /* driver-facing */
    unsigned sample_rate;
    double frame_rate;
    size_t speaker_count;
    size_t min_audio_frames;
    size_t audio_emit_frames;
    size_t frames_after_eos;
    size_t default_max_steps;
    int uses_cfg;

    /* cloning only, kept so the generation difference stays visible */
    int insert_bos_before_voice;
    size_t speaker_proj_input_dim;

    char language[32];
    /* Provenance (E3-13). A voice KV is a slice of THIS checkpoint's attention
     * state, so it is only meaningful against the weights that produced it; the
     * two strings below are what a pack has to identify itself by. */
    char revision[64];
    char weights_tts[128];
    char tokenizer_file[128];
    char speakers_file[128];
    char voices_dir[128];
} pocket_config;

typedef struct {
    char *name;
    char *file;    /* relative to the pack directory                          */
    size_t frames; /* positions speakers.json declares; 0 = not declared      */
} pocket_voice;

/* ------------------------------------------------------ quantization groups
 *
 * WHY THIS EXISTS.  "int8 is on" was one switch, and it moved every projection
 * in this engine at once.  That is not a policy, it is an average: the
 * backbone's [4096][1024] FFN and the flow head's [512][32] input projection
 * have nothing in common except the word "linear", and a per-row symmetric
 * absmax that is harmless on the first is a different proposition on a row of
 * 32 values.  The reference implementation says as much in
 * pocket_tts/quantization.py -- RECOMMENDED_CONFIG = {"attention", "ffn"},
 * with the flow matching network and the Mimi decoder left in float32 -- but
 * "upstream excluded it" is a claim, not a measurement, and this runtime has
 * an oracle and a comparator that upstream did not.
 *
 * So the set is explicit, addressable, and measured one group at a time.  The
 * bits are finer than the names an operator normally uses, on purpose: the
 * interesting question is rarely "is the flow head safe" but "is it the two
 * projections that touch the 32-dimensional latent", and answering that must
 * not require a rebuild.
 *
 * MYNAH_QUANT_GROUPS takes a comma list of the names in pocket_qgroup_names
 * below.  A leading '-' subtracts, so `all,-flow_io` is a sentence.  An
 * unrecognised name FAILS the model load: a typo that silently quantizes
 * nothing would show up as a quality result, which is the worst place to
 * discover it.  MYNAH_QUANT still decides *how* (int8/int4/f16); this decides
 * *what*. */
enum {
    POCKET_QG_BB_QKV     = 1u << 0,  /* backbone fused QKV   [3*1024][1024]  */
    POCKET_QG_BB_OPROJ   = 1u << 1,  /* backbone out_proj    [1024][1024]    */
    POCKET_QG_BB_FFN1    = 1u << 2,  /* backbone linear1     [4096][1024]    */
    POCKET_QG_BB_FFN2    = 1u << 3,  /* backbone linear2     [1024][4096]    */
    POCKET_QG_CT_QKV     = 1u << 4,  /* codec transformer QKV                */
    POCKET_QG_CT_OPROJ   = 1u << 5,
    POCKET_QG_CT_FFN1    = 1u << 6,
    POCKET_QG_CT_FFN2    = 1u << 7,
    POCKET_QG_FLOW_CORE  = 1u << 8,  /* flow head, hidden-to-hidden          */
    POCKET_QG_FLOW_IO    = 1u << 9,  /* flow input_proj + final_linear (k/n=32) */
    POCKET_QG_COND_IN    = 1u << 10, /* flow_lm.input_linear   [1024][32]    */
    POCKET_QG_COND_EOS   = 1u << 11, /* flow_lm.out_eos        [1][1024]     */
    POCKET_QG_CODEC_CONV = 1u << 12, /* quantizer.output_proj + the conv1d stack */
    /* The SEANet decoder's three TRANSPOSED convolutions, and its own group
     * rather than part of codec_conv because it is a different trade and the
     * spec should be able to say so.  Measured here: `convtr.gemm` 93.4 ->
     * 44.8 ms (2.09x) and `codec.conv_stack` 1.20x -> 1.67x, against SNR 37.0
     * -> 32.0 dB and log-mel 0.9971 -> 0.9950 on the same utterance.  That is
     * a real speed/quality trade rather than a free win, and the speed half is
     * an Apple Silicon number while the quality half transfers -- so it is OFF
     * in the default spec and one string away.  PLAN.md E10-14 closes it by
     * re-taking the speed number on the Linux box. */
    POCKET_QG_CODEC_CONVTR = 1u << 13
};

#define POCKET_QG_ATTENTION (POCKET_QG_BB_QKV | POCKET_QG_BB_OPROJ)
#define POCKET_QG_FFN       (POCKET_QG_BB_FFN1 | POCKET_QG_BB_FFN2)
#define POCKET_QG_CODEC_TF                                                 \
    (POCKET_QG_CT_QKV | POCKET_QG_CT_OPROJ | POCKET_QG_CT_FFN1 |           \
     POCKET_QG_CT_FFN2)
#define POCKET_QG_FLOW_NET  (POCKET_QG_FLOW_CORE | POCKET_QG_FLOW_IO)
#define POCKET_QG_COND      (POCKET_QG_COND_IN | POCKET_QG_COND_EOS)
#define POCKET_QG_BITS 14u
#define POCKET_QG_ALL                                                      \
    (POCKET_QG_ATTENTION | POCKET_QG_FFN | POCKET_QG_CODEC_TF |            \
     POCKET_QG_FLOW_NET | POCKET_QG_COND | POCKET_QG_CODEC_CONV |          \
     POCKET_QG_CODEC_CONVTR)

/* THE DEFAULT.  The only statement in this file that is an experimental result
 * rather than a definition, so it is written as the string an operator could
 * have typed, and every clause in it has a measurement behind it.
 *
 * Measured on models/pocket-en, seed 1234, against an f32 run of the same seed
 * (temperature 0, i.e. no sampler noise to hide behind):
 *
 *   the Mimi decoder transformer and the quantizer projection are FEED-FORWARD.
 *   Their error lands on the waveform once and stops there: int8 gives log-mel
 *   corr 0.9994, the identical frame count and the identical EOS step, for
 *   ~50% of the wall clock.  They take int8.
 *
 *   the backbone, the flow head and the input projection are INSIDE the
 *   autoregressive loop.  Their per-step error is only 1-6% -- no single one of
 *   them is anomalous -- but 50 steps of feedback turn that into a different
 *   sampled trajectory: the frame count moves, the EOS step moves, log-mel corr
 *   against f32 falls to 0.73-0.96.  f16 on the same weights measures 1.2e-06
 *   per step, 2.9e-05 end to end, and does not move a single frame.  So they
 *   take f16, which is exact.
 *
 * This spec is portable as written.  Until 3892ba6 f16 was an aarch64-only
 * encoding and this string would have meant "int8 codec, f32 backbone" on
 * x86-64; src/qmat.c now carries F16C/AVX2 and scalar half kernels selected by
 * CPUID, so the f16 half of it is real on both production architectures and
 * degrades to exact f32 only where no kernel exists at all.  Either way it
 * degrades to something EXACT, never to a substitute encoding.
 *
 * The full-int8 configuration is one environment variable away
 * (MYNAH_QUANT_GROUPS=all) and is faster again; what it costs is written
 * above, in frames. */
/* The codec clauses name int8 EXPLICITLY, and they did not always.
 *
 * They used to be bare -- "codec_transformer,codec_conv" -- which in this
 * grammar means "whatever MYNAH_QUANT says", and that was right when the
 * cache's own default was f32 and MYNAH_QUANT was how anyone turned
 * quantization on at all. It stopped being right when src/mynah_tts.c:402
 * started requesting f16 for every non-Magpie engine when MYNAH_QUANT is
 * unset: the base became f16, the two bare clauses inherited f16, and the
 * shipped default quietly became "f16 everywhere" -- the one configuration
 * this comment does not describe.
 *
 * Measured on this pack, seed 1234, bare clauses against explicit ones:
 * codec.transformer 183.5 -> 131.3 ms (1.40x), request.total 770.0 -> 719.8,
 * and the WAV is the same length to the byte, so the frame count and the EOS
 * step are identical -- which is exactly the property the paragraph above
 * measured int8 against. The explicit spelling also produces byte-identical
 * audio to MYNAH_QUANT=int8 under this same spec, which is the other half of
 * the proof that this is what the string always meant to say.
 *
 * Naming the encoding means the spec now says what it does regardless of what
 * the base is, which is what a default that calls itself an experimental
 * result has to do. */
#define POCKET_QG_DEFAULT_SPEC                                                 \
    "codec_transformer,codec_conv,codec_convtr,"                               \
    "backbone:bf16,flow_net:f16,conditioner:f16"

/* The same sentence with the codec clauses PINNED, used when MYNAH_QUANT is
 * unset -- which is the shipped configuration and the one the paragraph above
 * is about.
 *
 * A bare clause means "whatever MYNAH_QUANT says", and that was right while the
 * cache's own default was f32 and MYNAH_QUANT was the only way to turn
 * quantization on. It stopped being right when src/mynah_tts.c:402 began
 * requesting f16 for every non-Magpie engine with MYNAH_QUANT unset: the base
 * became f16, the two bare clauses inherited it, and the shipped default
 * quietly became "f16 everywhere" -- the one configuration the paragraph above
 * does not describe and never measured.
 *
 * Measured on this pack, seed 1234: codec.transformer 183.5 -> 130.9 ms
 * (1.40x), request.total 770.0 -> 708.2, and the WAV is the same length to the
 * byte, so the frame count and the EOS step do not move -- which is exactly the
 * property int8 was accepted on. The result is byte-identical to what
 * MYNAH_QUANT=int8 produces under this spec, which is the other half of the
 * proof that this is what the string always meant.
 *
 * Why two strings rather than one pinned string: MYNAH_QUANT is the knob for
 * "how", and pinning unconditionally would take the codec out of its reach --
 * `MYNAH_QUANT=int4` would silently no longer reach the codec, which is
 * precisely the configuration int4 is wanted for. So the pin applies only where
 * there is no MYNAH_QUANT to obey. */
/* codec_convtr joined both strings on 2026-09-19, and the reason is a serving
 * A/B rather than a kernel microbenchmark. Two ten-minute soaks on the Axion at
 * C90, identical in every other respect, differing only in this clause:
 *
 *   OFF (what this string used to say)  MARGINAL  TTFA p95 544 ms, RTF p95
 *                                       0.842, 5 stalls of 15380, 106.5 audio-s/s
 *   ON                                  GOOD      TTFA p95 492 ms, RTF p95
 *                                       0.728, 0 stalls of 17904, 124.3 audio-s/s
 *
 * So this is not 17% of throughput on top of a qualified configuration: with the
 * clause absent there is NO qualified concurrency on that machine, because the
 * TTFA gate and the stall gate both fail. The three transposed convolutions are
 * the per-slot term b of T_frame(B) = a + b*B, and 1.86x on b at two threads per
 * worker is capacity, not latency.
 *
 * What it costs, unchanged and known: SNR 28.9-32.9 dB against 36.4-37.9 on the
 * codec's own quality gate, log-mel nearly unchanged. The audio the decision was
 * taken on is kept under campioni/ -- captured from the streaming server under
 * C90 of real load, with this clause ON. */
/* backbone:bf16 joined this string on 2026-09-19. The checkpoint is bf16, the
 * production CPU multiplies bf16 natively, and the tiled BFMMLA kernel measures
 * 173.6 GFLOP/s against the f16 path's 11.8 at the prefill's shape -- 1.35x on
 * prep.prefill_proj and RTF 0.151 -> 0.121 end to end on the reference host.
 *
 * TWO THINGS THAT HAD TO BE TRUE FIRST, and both were measured rather than
 * argued:
 *
 *   REPRODUCIBILITY. Every bf16 width takes one kernel, so a row's answer does
 *   not depend on how many requests share the worker. Verified on the live
 *   server: the same request, same seed, same text, repeated fifteen times
 *   under 0, 7, 23 and 47 concurrent others, returned the SAME sha256 every
 *   time. Without that this is a knob, not a default.
 *
 *   NOBODY GETS A WORSE PATH. Where no bf16 vector kernel is compiled -- every
 *   x86 host today -- qmat_qtype_available() downgrades this to f16 rather
 *   than to scalar bf16, which would be slower AND less precise than f16.
 *
 * WHAT IT COSTS, stated because it is not free: against an f32 backbone, f16
 * holds SNR 33.6 dB and correlation 0.9998 while bf16 sits at 3.0-5.4 dB and
 * 0.76-0.85. bf16 does not render the model differently, it DEPARTS from it.
 * The samples behind that decision were listened to before this line changed,
 * and the length, the frame count and the EOS step are identical in all of
 * them. MYNAH_QUANT_GROUPS=backbone:f16 returns to the old default exactly. */
#define POCKET_QG_DEFAULT_SPEC_PINNED                                          \
    "codec_transformer:int8,codec_conv:int8,codec_convtr:int8,"                \
    "backbone:bf16,flow_net:f16,conditioner:f16"

typedef struct {
    const char *name;
    unsigned mask;
} pocket_qgroup_name;

static const pocket_qgroup_name pocket_qgroup_names[] = {
    {"all", POCKET_QG_ALL},
    {"none", 0u},
    {"attention", POCKET_QG_ATTENTION},
    {"ffn", POCKET_QG_FFN},
    {"backbone", POCKET_QG_ATTENTION | POCKET_QG_FFN},
    {"bb_qkv", POCKET_QG_BB_QKV},
    {"bb_oproj", POCKET_QG_BB_OPROJ},
    {"bb_ffn1", POCKET_QG_BB_FFN1},
    {"bb_ffn2", POCKET_QG_BB_FFN2},
    {"codec_transformer", POCKET_QG_CODEC_TF},
    {"ct_qkv", POCKET_QG_CT_QKV},
    {"ct_oproj", POCKET_QG_CT_OPROJ},
    {"ct_ffn1", POCKET_QG_CT_FFN1},
    {"ct_ffn2", POCKET_QG_CT_FFN2},
    {"flow_net", POCKET_QG_FLOW_NET},
    {"flow_core", POCKET_QG_FLOW_CORE},
    {"flow_io", POCKET_QG_FLOW_IO},
    {"conditioner", POCKET_QG_COND},
    {"cond_in", POCKET_QG_COND_IN},
    {"cond_eos", POCKET_QG_COND_EOS},
    {"codec_conv", POCKET_QG_CODEC_CONV},
    {"codec_convtr", POCKET_QG_CODEC_CONVTR}
};

/* Parses `spec` into a mask, and -- when a token carries a `:qtype` suffix --
 * into a per-bit encoding.  `qtype_of_bit[b]` is -1 for "use MYNAH_QUANT's
 * encoding", which is what a bare name means.
 *
 * `backbone:f16,codec_transformer:int8` is the sentence this exists to say:
 * f16 is exact where the error would feed back through the AR loop, int8 where
 * it cannot.  An encoding a build cannot honour resolves to exact f32 inside
 * qmat rather than to a substitute, so one string is correct on every target
 * and a target without the kernel loses speed, never correctness. */
static int pocket_qgroups_parse(const char *spec, unsigned *out,
                                signed char *qtype_of_bit, char *error,
                                size_t capacity) {
    unsigned mask = 0u;
    const char *p = spec;
    *out = 0u;
    if (qtype_of_bit != NULL) {
        for (size_t i = 0; i < POCKET_QG_BITS; ++i) qtype_of_bit[i] = -1;
    }
    while (*p != '\0') {
        while (*p == ',' || *p == ' ' || *p == '\t') ++p;
        if (*p == '\0') break;
        int subtract = 0;
        if (*p == '-' || *p == '!') {
            subtract = 1;
            ++p;
        } else if (*p == '+') {
            ++p;
        }
        const char *start = p;
        while (*p != '\0' && *p != ',' && *p != ' ' && *p != '\t') ++p;
        size_t len = (size_t)(p - start);
        if (len == 0) continue;
        /* name[:qtype] */
        int token_qtype = -1;
        const char *colon = NULL;
        for (size_t i = 0; i < len; ++i) {
            if (start[i] == ':') {
                colon = start + i;
                break;
            }
        }
        if (colon != NULL) {
            char qname[16];
            const size_t qlen = (size_t)(start + len - colon - 1);
            if (qlen == 0 || qlen >= sizeof(qname)) {
                pocket_error(error, capacity,
                             "MYNAH_QUANT_GROUPS: '%.*s' has no encoding after ':'",
                             (int)len, start);
                return -1;
            }
            memcpy(qname, colon + 1, qlen);
            qname[qlen] = '\0';
            token_qtype = mynah_qmat_qtype_from_name(qname);
            if (token_qtype < 0) {
                pocket_error(error, capacity,
                             "MYNAH_QUANT_GROUPS: '%s' is not an encoding "
                             "(f32, int8, int4, f16)",
                             qname);
                return -1;
            }
            len = (size_t)(colon - start);
        }
        const pocket_qgroup_name *hit = NULL;
        for (size_t i = 0; i < sizeof(pocket_qgroup_names) /
                                   sizeof(pocket_qgroup_names[0]);
             ++i) {
            const pocket_qgroup_name *n = &pocket_qgroup_names[i];
            if (strlen(n->name) == len && strncmp(n->name, start, len) == 0) {
                hit = n;
                break;
            }
        }
        if (hit == NULL) {
            pocket_error(error, capacity,
                         "MYNAH_QUANT_GROUPS: unknown group '%.*s' (known: all, "
                         "none, attention, ffn, backbone, bb_qkv, "
                         "bb_oproj, bb_ffn1, bb_ffn2, codec_transformer, ct_qkv, "
                         "ct_oproj, ct_ffn1, ct_ffn2, flow_net, flow_core, "
                         "flow_io, conditioner, cond_in, cond_eos, codec_conv)",
                         (int)len, start);
            return -1;
        }
        if (subtract) {
            mask &= ~hit->mask;
        } else {
            /* `name:f32` is a subtraction spelled as an encoding, so the two
             * ways of saying "leave this exact" cannot disagree. */
            if (token_qtype == 0) {
                mask &= ~hit->mask;
            } else {
                mask |= hit->mask;
            }
            if (qtype_of_bit != NULL && token_qtype > 0) {
                for (size_t b = 0; b < POCKET_QG_BITS; ++b) {
                    if ((hit->mask >> b) & 1u) {
                        qtype_of_bit[b] = (signed char)token_qtype;
                    }
                }
            }
        }
    }
    *out = mask;
    return 0;
}

/* The encoding for one group bit: the token's own, or -1 for the cache's. */
static int pocket_qtype_for(const signed char *qtype_of_bit, unsigned group) {
    if (qtype_of_bit == NULL || group == 0u) return -1;
    for (size_t b = 0; b < POCKET_QG_BITS; ++b) {
        if ((group >> b) & 1u) return qtype_of_bit[b];
    }
    return -1;
}

/* ------------------------------------------------- the linear projection hook
 *
 * `transformer_ar` owns no cache and knows no tensor name on purpose, so the
 * quantized projection path is installed from here: one hook per transformer,
 * carrying the model's shared int8/int4/f16 cache and a table of cache keys
 * built once at load.  Nothing is formatted or allocated per call (AGENTS.md
 * rule 4): the key is a pointer into a flat table indexed by layer and kind.
 *
 * The cache is the model's, not the context's, so N concurrent requests share
 * one quantized copy of the weights (rule 3: weights are shared read-only). */
typedef struct {
    mynah_qmat_cache *qcache;
    const mynah_backend *backend;
    char *names; /* [layers * 4][POCKET_QNAME_MAX], flat */
    size_t layers;
    /* The resolved group mask, and the group each of the four kinds belongs
     * to.  A kind whose bit is clear takes the exact f32 path, so a group can
     * be measured alone without a second binary. */
    unsigned groups;
    unsigned kind_group[4];
    signed char kind_qtype[4]; /* -1 = the cache's encoding */
    const char *block; /* static census label: "backbone" / "codec_tr" */
} pocket_linear_hook;

/* The flow head's equivalent.  Its nine kinds are indexed by (index, kind)
 * rather than (layer, kind): `index` is the residual block or the time
 * condition, and 0 for the singletons. */
typedef struct {
    mynah_qmat_cache *qcache;
    const mynah_backend *backend;
    char *names; /* [indices * MYNAH_FLOW_LINEAR_KIND_COUNT][POCKET_QNAME_MAX] */
    size_t indices;
    unsigned groups;
    signed char core_qtype;
    signed char io_qtype;
} pocket_flow_hook;

/* ------------------------------------------------- the two axes, and the seam
 *
 * A hook above answers WHICH ENCODING a (group, tensor) takes.  The scratch
 * below answers HOW MANY ROWS one call carries.  They are independent, and the
 * merge of the two lanes is exactly the place they meet:
 *
 *   - rows of ONE request (a prefill tile: 16 codec positions, or a text
 *     prefill) may be read from the weight in a single pass, because the row
 *     count is a property of that request's own text and never of who else is
 *     in flight;
 *   - one row of EACH of N requests may also be read in a single pass, but only
 *     through a kernel that is bit-exact per row, or a request's audio would
 *     depend on its neighbours (`mynah_tts.h`).
 *
 * The scratch is deliberately NOT tied to a hook type: there is one per context
 * (for its own tiles) and one per driver batch (for the cross-request step), and
 * wiring a shared hook straight into `linear_user` would have made N requests
 * share one activation buffer -- AGENTS.md rule 3 with a data race attached. */
typedef struct {
    size_t rows;   /* widest call this scratch can serve */
    size_t k_max;  /* widest reduction this scratch can serve */
    int8_t *qx;    /* [rows][k_max] int8 activations */
    float *sx;     /* [rows] activation scales */
    const float **in_ptr;  /* [rows], the row view of a contiguous tile */
    float **out_ptr;       /* [rows] */
} pocket_call;

/* What `linear_user` points at: the shared read-only hook plus this caller's
 * own scratch.  One per transformer per caller, one per flow head per caller. */
typedef struct {
    const pocket_linear_hook *hook;
    pocket_call call;
    /* Set for the duration of `mynah_transformer_ar_prefill` and clear
     * everywhere else, so the cost map can charge the prefill's projections to
     * the prefill.  It is per caller, like the scratch beside it, and the
     * prefill runs on the thread that opened the region -- so no worker ever
     * reads it and there is nothing here to share. */
    int in_prefill;
} pocket_tar_call;

typedef struct {
    const pocket_flow_hook *hook;
    pocket_call call;
} pocket_flow_call;

/* One projection's routing, resolved by the adapter that knows which hook it is
 * holding, so the row-count core below never learns about groups or kinds. */
typedef struct {
    mynah_qmat_cache *qcache;
    const mynah_backend *backend;
    const char *name;  /* the cache key; unused when `quantized` is 0 */
    int qtype;         /* -1 = whatever the cache resolved to */
    int quantized;     /* 0 = this group is not selected: exact f32 */
    int f32_matvec;    /* 1 = mynah_matvec_bias_f32 for a single row */
    /* The census identity of this projection. Both are STATIC strings, never
     * the per-layer cache key: the census keys on the pointers, and merging
     * layers of one shape is what makes the table readable -- the call count
     * says how many layers ran. Set even when the group is not quantized, so
     * an f32 projection is counted rather than invisible. */
    const char *cs_block;
    const char *cs_kind;
} pocket_proj;


/* ---------------------------------------------------------- model weights */

struct mynah_engine_state {
    /* E2-5: the text-chunk seam has been reported once for this model. Benign
     * if two threads race -- the worst case is the warning printed twice. */
    int chunk_warned;

    pocket_config cfg;

    char *model_dir;
    mynah_weights *weights;
    int owns_weights;

    /* shared, borrowed from the model; NULL disables the hook */
    mynah_qmat_cache *qcache;
    const mynah_backend *backend;
    unsigned qgroups; /* resolved MYNAH_QUANT_GROUPS, 0 when quant is off */
    signed char qgroup_qtype[POCKET_QG_BITS]; /* per-bit override, -1 = cache */
    signed char cond_in_qtype;
    signed char cond_eos_qtype;
    signed char codec_conv_qtype;
    signed char codec_convtr_qtype;
    pocket_linear_hook backbone_hook;
    pocket_linear_hook codec_hook;
    pocket_flow_hook flow_hook;
    /* Cache keys for the three projections this file owns directly. */
    char cond_in_key[POCKET_QNAME_MAX];
    char cond_eos_key[POCKET_QNAME_MAX];
    char codec_conv_key[POCKET_QNAME_MAX];

    /* backbone */
    mynah_transformer_ar_layer *backbone_layers;
    mynah_transformer_ar_weights backbone;

    /* flow head */
    mynah_flow_time_embed_weights *time_embed;
    mynah_flow_res_block_weights *res_blocks;
    mynah_flow_head_weights flow;

    /* Mimi decoder transformer */
    mynah_transformer_ar_layer *codec_layers;
    mynah_transformer_ar_weights codec_transformer;

    /* Mimi upsample + SEANet decoder */
    mynah_conv_weights upsample;
    mynah_conv_weights *decoder_convtr;
    mynah_seanet_resblock_weights *decoder_blocks;
    mynah_seanet_decoder_weights decoder;

    /* single tensors */
    const float *embed_table;
    const float *bos_emb;
    const float *emb_mean;
    const float *emb_std;
    const float *input_linear;
    const float *out_eos_weight;
    const float *out_eos_bias;
    const float *quantizer_proj;
    const float *speaker_proj;     /* cloning only; may be NULL */
    const float *bos_before_voice; /* cloning only; may be NULL */

    pocket_voice *voices;
    size_t voice_count;

    mynah_sp *tokenizer;
};

/* Defined below, next to the writer; the context only holds a pointer. */
typedef struct pocket_dump pocket_dump;

/* ------------------------------------------------------------- per request */

struct mynah_engine_ctx {
    mynah_engine_state *state;

    int *text_ids;
    size_t text_length;
    unsigned speaker;
    float temperature;
    float noise_std;
    size_t max_steps;

    mynah_transformer_ar_state *backbone;
    mynah_flow_head *flow;
    mynah_transformer_ar_state *codec_transformer;
    mynah_seanet_state *codec;

    /* The model's weight struct with this context's own projection scratch
     * bound in.  The layers themselves are still the model's. */
    pocket_tar_call backbone_call;
    pocket_tar_call codec_call;
    pocket_flow_call flow_call;
    mynah_transformer_ar_weights backbone_w;
    mynah_transformer_ar_weights codec_w;
    mynah_flow_head_weights flow_w;

    ingot_st *voice_file;
    size_t voice_positions;
    float *voice_kv; /* [2][voice_positions][heads][head_dim] */

    /* ---- the long-form text window (E5-5) -----------------------------
     *
     * `text_length` is what has been ACCEPTED, `text_prefilled` what is already
     * in the backbone KV, and `text_capacity` what was paid for at admission.
     * They are three different numbers only while `text_open` is set; for an
     * ordinary single-text request all three are the request's own length and
     * every branch below is dead.
     *
     * `text_capacity` is the ceiling, and it is declared at admission on
     * purpose -- see `mynah_engine_pocket_reserve_text`. */
    size_t text_capacity;
    size_t text_prefilled;
    int text_open;

    float *text_embed; /* [text_capacity][hidden_dim] */
    float *step_input; /* [hidden_dim] */
    float *hidden;     /* [hidden_dim] */

    /* Optional resident CUDA backbone.  These are deliberately opaque
     * backend-owned buffers represented as float pointers; the CPU state above
     * remains the source of truth for admission, rollback and fallback.  KV is
     * [layers][2][capacity][heads*head_dim], K then V, matching the host state
     * byte layout. */
    float *cuda_backbone_kv;
    float *cuda_x;
    float *cuda_norm;
    float *cuda_qkv;
    float *cuda_attn;
    float *cuda_proj;
    float *cuda_ffn;
    size_t cuda_backbone_capacity;
    size_t cuda_backbone_kv_floats;
    int cuda_backbone_enabled;
    int cuda_backbone_valid;

    /* Optional resident CUDA SEANet decoder. The codec transformer and
     * quantizer boundary remain host-owned for now; this handle owns the
     * causal convolution rings and transposed-convolution tails on device. */
    mynah_backend_decoder *cuda_decoder;
    float *cuda_decoder_input;
    float *cuda_decoder_output;
    int cuda_decoder_enabled;
    int cuda_decoder_graph_enabled;
    int cuda_decoder_started;

    float *noise;      /* [latent_dim] */
    float *flow_out;   /* [latent_dim] */
    float *latents;    /* [max_steps][latent_dim] */
    float *denorm;     /* [latent_dim] */
    float *codec_in;   /* [codec_dim] */
    float *codec_up;   /* [codec_dim][upsample_stride] */
    float *codec_seq;  /* [upsample_stride][codec_tf_dim] */
    float *codec_out;  /* [upsample_stride][codec_tf_dim] */
    float *codec_back; /* [codec_dim][upsample_stride] */
    float *pcm;        /* [samples_per_frame] */

    size_t frames;
    size_t decoded_frames;
    size_t step;
    size_t frames_after_eos;
    size_t eos_step; /* SIZE_MAX until the logit first crosses the threshold */
    int prepared;
    /* A resumable prefill is in flight: the prologue has run and the text is
     * partly in the KV cache. Never 1 at the same time as `prepared`. */
    int seeding;
    int eos;
    /* The step budget was reached before EOS.  Kept as a flag rather than
     * reported as an error: the driver has no budget check of its own, and a
     * batched step fails all its slots or none, so raising an error here would
     * let one request's length cap kill fifteen strangers. */
    int budget_exhausted;
    /* A per-request failure leaves the backbone one position ahead of the
     * frame history, which no later call can reconcile. The request is over;
     * saying so is the difference between a failed request and a context that
     * keeps generating from a state nothing produced. */
    int broken;
    float eos_logit;

    uint64_t seed;
    uint64_t rng;
    int have_spare;
    float spare;

    mynah_pocket_noise_fn noise_fn;
    void *noise_user;

    /* NULL unless MYNAH_POCKET_DUMP is set.  One pointer test per step. */
    pocket_dump *dump;

    /* The request's wall span. It cannot be bracketed by a region stack: the
     * lifecycle is spread over ctx_new/prepare/step/emit/decode/free and, in
     * the server, interleaves with other requests on the same thread. So it is
     * submitted once, as a duration, exactly like the runtime's own derived
     * regions. */
    unsigned long long t_created_ns;
};

struct mynah_engine_scratch {
    size_t batch;
    const mynah_backend *backend; /* borrowed model backend */
    /* Shared resident-CUDA transformer workspace for cross-request batching.
     * The host staging arrays are allocated once with this scratch; each row
     * still keeps its own persistent KV cache in mynah_engine_ctx. */
    float *cuda_host_input;
    float *cuda_host_output;
    float *cuda_host_kv;
    float *cuda_kv_shadow;
    float **cuda_kcache;
    float **cuda_vcache;
    size_t *cuda_positions;
    size_t *cuda_cache_strides;
    float *cuda_x;
    float *cuda_norm;
    float *cuda_qkv;
    float *cuda_attn;
    float *cuda_proj;
    float *cuda_ffn;
    size_t cuda_batch_capacity;
    int cuda_batch_enabled;
    int cuda_graph_enabled;
    int cuda_graph_ready;
    size_t cuda_kv_shadow_floats;

    /* The cross-request step: one stacked activation set and one projection
     * scratch for the whole batch, owned by the driver rather than by any
     * request in it. */
    mynah_transformer_ar_batch *backbone_batch;
    pocket_tar_call backbone_call;
    mynah_transformer_ar_weights backbone_w;
    mynah_transformer_ar_state **states; /* [batch] */
    const float **inputs;                /* [batch] */
    float **outputs;                     /* [batch] */
    /* The flow head, same arrangement.  The row set is not the same as the
     * backbone's: a request whose step was the terminal one draws no latent, so
     * emit runs over a subset of the slots that stepped. */
    mynah_flow_head_batch *flow_batch;
    pocket_flow_call flow_call;
    mynah_flow_head_weights flow_w;
    mynah_flow_head **flow_heads; /* [batch] */
    const float **flow_cond;      /* [batch] */
    const float **flow_noise;     /* [batch] */
    float **flow_out;             /* [batch] */

    /* Optional resident CUDA flow-head workspace.  It is separate from the
     * CPU flow batch because the CUDA graph captures these exact device and
     * pinned-host addresses. */
    float *cuda_flow_host_cond;
    float *cuda_flow_host_noise;
    float *cuda_flow_host_time;
    float *cuda_flow_host_output;
    float *cuda_flow_cond;
    float *cuda_flow_noise;
    float *cuda_flow_time_embed;
    float *cuda_flow_output;
    float *cuda_flow_y;
    float *cuda_flow_silu;
    float *cuda_flow_x;
    float *cuda_flow_norm;
    float *cuda_flow_hidden;
    float *cuda_flow_scratch;
    float *cuda_flow_mod;
    float *cuda_flow_final_mod;
    float *cuda_flow_time_hidden;
    float *cuda_flow_time_output;
    float *cuda_flow_time_sum;
    mynah_backend_flow_linear *cuda_flow_time_mlp_in;
    mynah_backend_flow_linear *cuda_flow_time_mlp_out;
    const float **cuda_flow_time_alpha;
    mynah_backend_flow_block *cuda_flow_blocks;
    mynah_backend_flow_linear cuda_flow_cond_embed;
    mynah_backend_flow_linear cuda_flow_input_proj;
    mynah_backend_flow_linear cuda_flow_final_adaln;
    mynah_backend_flow_linear cuda_flow_final_linear;
    size_t cuda_flow_batch_capacity;
    int cuda_flow_enabled;
    int cuda_flow_graph_enabled;
    int cuda_flow_graph_ready;
};

/* --------------------------------------------------------------- the dump
 *
 * `tests/parity_pocket.py` compares two directories of .npy files.  Until now
 * only the Python oracle could write one, so the only thing this runtime could
 * be compared on was the finished WAV -- which is exactly the wrong resolution
 * for a quantization question: by the time a group's error reaches the
 * waveform it has been through the flow head, the codec and a changed frame
 * count, and "the audio sounds fine" is not a number.
 *
 * So the engine can write the same shape of dump.  It is OFF unless
 * MYNAH_POCKET_DUMP names a directory, every buffer is allocated once in
 * ctx_new (AGENTS.md rule 4: the decode loop still allocates nothing, parses
 * nothing and opens nothing), and the files are written when the context is
 * freed.  Steps are stacked into one array per tensor rather than one file per
 * call: the number of rows IS the frame count, so a group that changes how
 * many frames get generated says so in the shape instead of in a footnote. */
struct pocket_dump {
    char dir[POCKET_PATH_MAX];
    size_t capacity;   /* max_steps                                        */
    size_t steps;      /* AR steps recorded                                */
    size_t frames;     /* latents emitted                                  */
    size_t decoded;    /* frames through the codec                         */
    size_t hidden_dim;
    size_t latent_dim;
    size_t codec_row;  /* upsample_stride * codec_tf_dim                   */
    size_t frame_samples;
    float *hidden;    /* [capacity][hidden_dim]  post-out_norm backbone    */
    float *eos;       /* [capacity]              out_eos logit             */
    float *flow_out;  /* [capacity][latent_dim]  flow head output          */
    float *latent;    /* [capacity][latent_dim]  noise + flow_out          */
    float *denorm;    /* [capacity][latent_dim]  emb_std/emb_mean applied  */
    float *codec_tf;  /* [capacity][codec_row]   decoder transformer out   */
    float *pcm;       /* [capacity][frame_samples]                         */
};

/* --------------------------------------------------------------- the RNG
 *
 * Per context, never global (AGENTS.md rule 3): two requests in one process
 * must not be able to consume each other's draws. splitmix64 plus Box-Muller;
 * the spare normal is kept so a frame costs one transcendental pair per two
 * values rather than per value. */

static uint64_t pocket_rng_next(uint64_t *state) {
    *state += UINT64_C(0x9E3779B97F4A7C15);
    uint64_t z = *state;
    z = (z ^ (z >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94D049BB133111EB);
    return z ^ (z >> 31);
}

/* Uniform in (0, 1]: never 0, because log() of it is taken. */
static double pocket_rng_unit(uint64_t *state) {
    const uint64_t bits = pocket_rng_next(state) >> 11;
    return ((double)bits + 1.0) * (1.0 / 9007199254740993.0);
}

static float pocket_rng_normal(mynah_engine_ctx *ctx) {
    if (ctx->have_spare) {
        ctx->have_spare = 0;
        return ctx->spare;
    }
    const double u1 = pocket_rng_unit(&ctx->rng);
    const double u2 = pocket_rng_unit(&ctx->rng);
    const double radius = sqrt(-2.0 * log(u1));
    const double angle = 6.283185307179586476925286766559 * u2;
    ctx->spare = (float)(radius * sin(angle));
    ctx->have_spare = 1;
    return (float)(radius * cos(angle));
}

/* ------------------------------------------------------------ dump writer */

/* A .npy the way numpy.load() wants it: magic, version 1.0, a padded ASCII
 * header dict, then the raw little-endian f32 rows.  No dependency, and no
 * second format for the comparator to learn. */
static int pocket_npy_write(const char *dir, const char *name, const float *data,
                            size_t rows, size_t cols) {
    char path[POCKET_PATH_MAX];
    if ((size_t)snprintf(path, sizeof(path), "%s/%s.npy", dir, name) >=
        sizeof(path)) {
        return -1;
    }
    FILE *f = fopen(path, "wb");
    if (f == NULL) return -1;
    char dict[128];
    int n;
    if (cols == 0) {
        n = snprintf(dict, sizeof(dict),
                     "{'descr': '<f4', 'fortran_order': False, 'shape': (%zu,), }",
                     rows);
    } else {
        n = snprintf(dict, sizeof(dict),
                     "{'descr': '<f4', 'fortran_order': False, 'shape': (%zu, %zu), }",
                     rows, cols);
    }
    if (n < 0 || (size_t)n >= sizeof(dict)) {
        fclose(f);
        return -1;
    }
    /* The 10-byte preamble plus the header must be a multiple of 64. */
    size_t header = (size_t)n + 1u; /* + '\n' */
    while ((10u + header) % 64u != 0u) ++header;
    const unsigned char preamble[8] = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0};
    const unsigned short len = (unsigned short)header;
    int ok = fwrite(preamble, 1, 8, f) == 8;
    unsigned char lo = (unsigned char)(len & 0xffu);
    unsigned char hi = (unsigned char)((len >> 8) & 0xffu);
    ok = ok && fwrite(&lo, 1, 1, f) == 1 && fwrite(&hi, 1, 1, f) == 1;
    ok = ok && fwrite(dict, 1, (size_t)n, f) == (size_t)n;
    for (size_t i = (size_t)n; ok && i + 1u < header; ++i) ok = fputc(' ', f) != EOF;
    ok = ok && fputc('\n', f) != EOF;
    const size_t count = rows * (cols == 0 ? 1u : cols);
    if (ok && count > 0) ok = fwrite(data, sizeof(float), count, f) == count;
    if (fclose(f) != 0) ok = 0;
    return ok ? 0 : -1;
}

static void pocket_dump_free(pocket_dump *dump) {
    if (dump == NULL) return;
    free(dump->hidden);
    free(dump->eos);
    free(dump->flow_out);
    free(dump->latent);
    free(dump->denorm);
    free(dump->codec_tf);
    free(dump->pcm);
    free(dump);
}

/* Returns NULL when dumping is off OR when a buffer cannot be allocated: a
 * failed dump must not fail synthesis, it just does not happen, and the
 * missing directory is the report. */
static pocket_dump *pocket_dump_open(const pocket_config *cfg, size_t max_steps) {
    const char *dir = getenv("MYNAH_POCKET_DUMP");
    if (dir == NULL || dir[0] == '\0') return NULL;
    pocket_dump *d = (pocket_dump *)calloc(1, sizeof(*d));
    if (d == NULL) return NULL;
    if ((size_t)snprintf(d->dir, sizeof(d->dir), "%s", dir) >= sizeof(d->dir)) {
        free(d);
        return NULL;
    }
    d->capacity = max_steps;
    d->hidden_dim = cfg->hidden_dim;
    d->latent_dim = cfg->latent_dim;
    d->codec_row = cfg->upsample_stride * cfg->codec_tf_dim;
    d->frame_samples = cfg->samples_per_frame * cfg->audio_channels;
    d->hidden = (float *)calloc(max_steps * d->hidden_dim, sizeof(float));
    d->eos = (float *)calloc(max_steps, sizeof(float));
    d->flow_out = (float *)calloc(max_steps * d->latent_dim, sizeof(float));
    d->latent = (float *)calloc(max_steps * d->latent_dim, sizeof(float));
    d->denorm = (float *)calloc(max_steps * d->latent_dim, sizeof(float));
    d->codec_tf = (float *)calloc(max_steps * d->codec_row, sizeof(float));
    d->pcm = (float *)calloc(max_steps * d->frame_samples, sizeof(float));
    if (d->hidden == NULL || d->eos == NULL || d->flow_out == NULL ||
        d->latent == NULL || d->denorm == NULL || d->codec_tf == NULL ||
        d->pcm == NULL) {
        pocket_dump_free(d);
        return NULL;
    }
    return d;
}

static void pocket_dump_flush(const mynah_engine_ctx *ctx) {
    const pocket_dump *d = ctx->dump;
    if (d == NULL || d->steps == 0) return;
    /* Names chosen to land in the right stage of tests/parity_pocket.py's
     * classifier: out_norm -> backbone step, out_eos -> EOS, flow_net ->
     * flow head, quantizer -> Mimi input, decoder_transformer -> stage 10,
     * waveform -> stage 12. */
    pocket_npy_write(d->dir, "flow_lm.out_norm.out.call1", d->hidden, d->steps,
                     d->hidden_dim);
    pocket_npy_write(d->dir, "flow_lm.out_eos.out.call1", d->eos, d->steps, 0u);
    pocket_npy_write(d->dir, "flow_lm.flow_net.out.call1", d->flow_out, d->frames,
                     d->latent_dim);
    pocket_npy_write(d->dir, "stage07.latent.call1", d->latent, d->frames,
                     d->latent_dim);
    pocket_npy_write(d->dir, "stage08.denorm", d->denorm, d->decoded,
                     d->latent_dim);
    pocket_npy_write(d->dir, "mimi.decoder_transformer.out0.call1", d->codec_tf,
                     d->decoded, d->codec_row);
    pocket_npy_write(d->dir, "stage12.waveform", d->pcm,
                     d->decoded * d->frame_samples, 0u);

    char path[POCKET_PATH_MAX];
    if ((size_t)snprintf(path, sizeof(path), "%s/manifest.json", d->dir) >=
        sizeof(path)) {
        return;
    }
    FILE *f = fopen(path, "wb");
    if (f == NULL) return;
    const pocket_config *cfg = &ctx->state->cfg;
    fprintf(f,
            "{\n  \"generator\": \"src/engine_pocket.c\",\n"
            "  \"quant\": \"%s\",\n  \"quant_groups\": \"%s\",\n"
            "  \"language\": \"%s\",\n  \"voice\": \"%s\",\n"
            "  \"seed\": %llu,\n  \"temperature\": %.9g,\n"
            "  \"sampler_decode_steps\": %zu,\n  \"eos_threshold\": %.9g,\n"
            "  \"sample_rate\": %zu,\n  \"steps\": %zu,\n  \"frames\": %zu,\n"
            "  \"decoded_frames\": %zu,\n  \"eos_step\": ",
            mynah_qmat_qtype_name(mynah_qmat_cache_qtype(ctx->state->qcache)),
            mynah_qmat_groups_spec(), cfg->language,
            ctx->state->voices[ctx->speaker].name,
            (unsigned long long)ctx->seed, (double)ctx->temperature,
            cfg->flow_decode_steps, (double)cfg->eos_threshold,
            (size_t)cfg->sample_rate,
            d->steps, d->frames, d->decoded);
    if (ctx->eos_step == SIZE_MAX) fprintf(f, "null");
    else fprintf(f, "%zu", ctx->eos_step);
    fprintf(f, ",\n  \"token_ids\": [");
    for (size_t i = 0; i < ctx->text_length; ++i) {
        fprintf(f, "%s%d", i ? ", " : "", ctx->text_ids[i]);
    }
    /* The engine never sees the caller's string, only ids, so the text is
     * reconstructed from the pieces.  It exists so two dumps can be refused
     * when they are not the same utterance; `token_ids` above is the check
     * that actually decides, and it is exact. */
    fprintf(f, "],\n  \"text\": \"");
    for (size_t i = 0; i < ctx->text_length; ++i) {
        const char *piece = NULL;
        size_t len = 0;
        if (mynah_sp_piece(ctx->state->tokenizer, ctx->text_ids[i], &piece,
                           &len) != 0) {
            continue;
        }
        for (size_t j = 0; j < len; ++j) {
            /* U+2581 LOWER ONE EIGHTH BLOCK is SentencePiece's word boundary.
             * The tokenizer emits it on the first ordinary piece too, but it
             * is not a leading space in the caller's text. */
            if (j + 2u < len && (unsigned char)piece[j] == 0xE2u &&
                (unsigned char)piece[j + 1u] == 0x96u &&
                (unsigned char)piece[j + 2u] == 0x81u) {
                if (!(i == 0u && j == 0u)) fputc(' ', f);
                j += 2u;
                continue;
            }
            const unsigned char c = (unsigned char)piece[j];
            if (c == '"' || c == '\\') fprintf(f, "\\%c", c);
            else if (c < 0x20u) fprintf(f, "\\u%04x", c);
            else fputc(c, f);
        }
    }
    fprintf(f, "\"\n}\n");
    fclose(f);
}

/* ------------------------------------------------------ the linear hook */

/* --------------------------------------------------- resolving a projection
 *
 * These two turn "layer L, kind K of this transformer" / "index I, kind K of the
 * flow head" into the routing the row-count core below needs, and they are the
 * only place that knows about quantization groups. */

/* `input_proj` and `final_linear` carry their own bit (POCKET_QG_FLOW_IO)
 * because they are the two projections whose quantized dimension is the 32-wide
 * latent, which is a different numerical proposition from a 1024-wide one. */
static unsigned pocket_flow_kind_group(mynah_flow_linear_kind kind) {
    switch (kind) {
        case MYNAH_FLOW_LINEAR_INPUT_PROJ:
        case MYNAH_FLOW_LINEAR_FINAL_LINEAR:
            return POCKET_QG_FLOW_IO;
        default:
            return POCKET_QG_FLOW_CORE;
    }
}

/* Static census labels. They must outlive every record and be compared by
 * pointer, which is why they are file-scope literals and not built per call. */
static const char *const pocket_cs_tar_kind[4] = {"qkv", "oproj", "ffn1", "ffn2"};
static const char *const pocket_cs_flow_kind[MYNAH_FLOW_LINEAR_KIND_COUNT] = {
    "tmlp1", "tmlp2", "cond", "inproj", "adaln", "mlp1", "mlp2", "fadaln",
    "fout"};

static int pocket_tar_proj(const pocket_linear_hook *hook, size_t layer,
                           mynah_transformer_ar_linear_kind kind,
                           pocket_proj *out) {
    if (hook == NULL || layer >= hook->layers ||
        (size_t)kind >= 4u) {
        return -1;
    }
    memset(out, 0, sizeof(*out));
    out->qcache = hook->qcache;
    out->backend = hook->backend;
    out->qtype = -1;
    out->cs_block = hook->block != NULL ? hook->block : "transformer";
    out->cs_kind = pocket_cs_tar_kind[(size_t)kind];
    /* A single row of an unselected group must be exactly what `transformer_ar`
     * computes with no hook installed at all.  That is what makes binding the
     * hook unconditionally numerically free, which in turn is what lets the
     * tile path exist even when nothing is quantized. */
    out->f32_matvec = 1;
    if ((hook->groups & hook->kind_group[(size_t)kind]) == 0u) return 0;
    out->quantized = 1;
    out->qtype = hook->kind_qtype[(size_t)kind];
    out->name =
        hook->names + (layer * 4u + (size_t)kind) * (size_t)POCKET_QNAME_MAX;
    return 0;
}

static int pocket_flow_proj(const pocket_flow_hook *hook, size_t index,
                            mynah_flow_linear_kind kind, pocket_proj *out) {
    if (hook == NULL || index >= hook->indices ||
        (size_t)kind >= (size_t)MYNAH_FLOW_LINEAR_KIND_COUNT) {
        return -1;
    }
    memset(out, 0, sizeof(*out));
    out->qcache = hook->qcache;
    out->backend = hook->backend;
    out->qtype = -1;
    out->f32_matvec = 1;
    out->cs_block = "flow";
    out->cs_kind = pocket_cs_flow_kind[(size_t)kind];
    const unsigned group = pocket_flow_kind_group(kind);
    if ((hook->groups & group) == 0u) return 0;
    out->quantized = 1;
    out->qtype = (group == POCKET_QG_FLOW_IO) ? hook->io_qtype : hook->core_qtype;
    out->name = hook->names +
                (index * (size_t)MYNAH_FLOW_LINEAR_KIND_COUNT + (size_t)kind) *
                    (size_t)POCKET_QNAME_MAX;
    return 0;
}

/* ------------------------------------------------------- the row-count core */

/* One row, on exactly the path this group asked for. */
static int pocket_proj_row(const pocket_proj *p, const float *weight,
                           const float *bias, const float *in, float *out,
                           size_t k, size_t n) {
    if (!p->quantized) {
        if (p->f32_matvec) {
            MYNAH_CENSUS_OP(p->cs_block, p->cs_kind, k, n, 1u,
                            MYNAH_CENSUS_PATH_MATVEC_F32, -1);
            mynah_matvec_bias_f32(weight, in, bias, out, n, k);
            return 0;
        }
        MYNAH_CENSUS_OP(p->cs_block, p->cs_kind, k, n, 1u,
                        MYNAH_CENSUS_PATH_GEMM, -1);
        return mynah_backend_matmul(p->backend, in, out, 1u, k, n, weight, bias,
                                    NULL, 0);
    }
    /* The encoding recorded is the one qmat will RESOLVE, not the one asked
     * for: a build that cannot honour f16 downgrades to f32 inside qmat, and a
     * census that recorded the request would describe a run that did not
     * happen. That is the same silent fallback the dispatch table exists to
     * surface, one layer down. */
    MYNAH_CENSUS_OP(p->cs_block, p->cs_kind, k, n, 1u,
                    MYNAH_CENSUS_PATH_MATVEC_Q,
                    mynah_qmat_qtype_resolved(p->qtype >= 0
                                                  ? p->qtype
                                                  : mynah_qmat_cache_qtype(p->qcache)));
    return mynah_qmat_linear_resolved_qt(p->qcache, p->backend, p->name, weight, in,
                                         out, 1u, k, n, bias, p->qtype, NULL, 0);
}

/* The encoding a quantized projection will actually carry, resolved the same
 * way in every census call site. */
static int pocket_cs_qtype(const pocket_proj *p) {
    return mynah_qmat_qtype_resolved(
        p->qtype >= 0 ? p->qtype : mynah_qmat_cache_qtype(p->qcache));
}


/*
 * May `rows` rows share one pass over the weight?
 *
 * The question used to have a second half.  `mynah_qmat_linear_batched` takes
 * no qtype: it gates on `cache->qtype` and, if it is the first caller to touch
 * a tensor, creates the cache entry in the cache's own encoding.  So a group
 * carrying an explicit encoding that differed from the cache's had to be kept
 * off the batched path entirely -- otherwise a first-touch race would decide
 * the group's precision.  That refusal was correct and it was expensive: under
 * `MYNAH_QUANT=int8` the three `:f16` groups in the shipped spec (backbone,
 * flow_net, conditioner) read the weight once per row.
 *
 * E8-5 removed the cause rather than the refusal.  `mynah_qmat_linear_batched_qt`
 * resolves the encoding from the qtype it is HANDED, before it touches the
 * cache, and uses that same value for both the gate and the insert -- so
 * precision still comes from the group spec and never from whichever caller
 * arrived first, which is the property that made the refusal necessary.  What
 * is left here is only the scratch-shape question.
 *
 * A group whose encoding this build cannot honour resolves to exact f32 inside
 * qmat, which declines the weight-stationary path and puts every row back on
 * `_resolved_qt` -- the same bytes either way, because row b is bit-exact
 * against row b computed alone.
 */
static int pocket_proj_batchable(const pocket_proj *p, const pocket_call *call,
                                 size_t rows, size_t k) {
    return p->quantized && call != NULL && rows <= call->rows && k <= call->k_max;
}

static int pocket_proj_batched(const pocket_proj *p, pocket_call *call,
                               const float *weight, const float *bias,
                               const float *const *in_rows,
                               float *const *out_rows, size_t rows, size_t k,
                               size_t n) {
    /* This call and only this call is micro-batching engaging. Every other
     * multi-row path below records ROWLOOP, so "did it engage" is answered by
     * the calls column of these rows and by nothing else. */
    MYNAH_CENSUS_OP(p->cs_block, p->cs_kind, k, n, rows,
                    MYNAH_CENSUS_PATH_BATCHED_Q, pocket_cs_qtype(p));
    return mynah_qmat_linear_batched_qt(p->qcache, p->backend, p->name, weight,
                                        in_rows, out_rows, rows, k, n, bias,
                                        call->qx, call->sx, p->qtype, NULL, 0);
}


/*
 * `count` contiguous rows that are consecutive positions of ONE request.
 *
 * The row count here is a function of that request's own text, or of the codec's
 * fixed 16-position stride, never of who else is in flight -- so this call may
 * read the weight once for the whole tile even when that reassociates the sum
 * differently from a matvec.  That is the point: the Mimi decoder was running 16
 * positions as 16 full passes over the same weights.
 */
static int pocket_proj_tile(const pocket_proj *p, pocket_call *call,
                            const float *weight, const float *bias,
                            const float *in, float *out, size_t count, size_t k,
                            size_t n) {
    if (count == 1u) return pocket_proj_row(p, weight, bias, in, out, k, n);
    if (pocket_proj_batchable(p, call, count, k)) {
        for (size_t b = 0; b < count; ++b) {
            call->in_ptr[b] = in + b * k;
            call->out_ptr[b] = out + b * n;
        }
        return pocket_proj_batched(p, call, weight, bias, call->in_ptr,
                                   call->out_ptr, count, k, n);
    }
    if (!p->quantized) {
        /* The exact f32 matmul this group asked for, once for the whole tile
         * instead of once per row. */
        MYNAH_CENSUS_OP(p->cs_block, p->cs_kind, k, n, count,
                        MYNAH_CENSUS_PATH_GEMM, -1);
        return mynah_backend_matmul(p->backend, in, out, count, k, n, weight, bias,
                                    NULL, 0);
    }
    /* Quantized but not batchable: keep every row on the encoding it would have
     * taken alone rather than silently moving it to another one.
     *
     * THIS IS THE BRANCH THE BANNER COULD NOT SEE. The feature is compiled,
     * supported and resolved ON, and this tile still reads the weight `count`
     * times. It is recorded under its own path name at the tile's width, so it
     * appears in the census as a row-loop of `count` rather than disappearing
     * into `count` innocent-looking matvecs. */
    MYNAH_CENSUS_OP(p->cs_block, p->cs_kind, k, n, count,
                    MYNAH_CENSUS_PATH_ROWLOOP, pocket_cs_qtype(p));
    for (size_t b = 0; b < count; ++b) {
        if (pocket_proj_row(p, weight, bias, in + b * k, out + b * n, k, n) != 0) {
            return -1;
        }
    }
    return 0;
}

/*
 * One row of each of `batch` DIFFERENT requests.
 *
 * `mynah_qmat_linear_batched` quantizes each activation row on its own, so the
 * batch width never reaches the arithmetic and row b comes out bit-identical to
 * row b computed alone.  That is what makes `mynah_tts.h`'s promise true, and it
 * is why this path never falls through to a GEMM the way the tile path does: a
 * GEMM's blocking is a function of the row count, and the row count here is
 * whoever happened to be in flight.
 */
static int pocket_proj_rows(const pocket_proj *p, pocket_call *call,
                            const float *weight, const float *bias,
                            const float *const *in_rows, float *const *out_rows,
                            size_t batch, size_t k, size_t n) {
    if (batch > 1u && pocket_proj_batchable(p, call, batch, k)) {
        return pocket_proj_batched(p, call, weight, bias, in_rows, out_rows, batch,
                                   k, n);
    }
    /* A batch of more than one that did NOT batch: the same finding as above,
     * on the cross-request axis. A batch of exactly one is not a fallback and
     * is left to pocket_proj_row to record as the matvec it is. */
    if (batch > 1u) {
        MYNAH_CENSUS_OP(p->cs_block, p->cs_kind, k, n, batch,
                        MYNAH_CENSUS_PATH_ROWLOOP,
                        p->quantized ? pocket_cs_qtype(p) : -1);
    }
    for (size_t b = 0; b < batch; ++b) {
        if (pocket_proj_row(p, weight, bias, in_rows[b], out_rows[b], k, n) != 0) {
            return -1;
        }
    }
    return 0;
}

/* ------------------------------------------------------------- the adapters */

static int pocket_linear(void *user, size_t layer,
                         mynah_transformer_ar_linear_kind kind,
                         const float *weight, const float *bias, const float *in,
                         float *out, size_t count, size_t k, size_t n) {
    pocket_tar_call *u = (pocket_tar_call *)user;
    pocket_proj proj;
    if (u == NULL || pocket_tar_proj(u->hook, layer, kind, &proj) != 0) return -1;
    if (!u->in_prefill) {
        return pocket_proj_tile(&proj, &u->call, weight, bias, in, out, count, k, n);
    }
    mynah_region_begin(MYNAH_RGN_PREFILL_PROJ);
    const int rc =
        pocket_proj_tile(&proj, &u->call, weight, bias, in, out, count, k, n);
    mynah_region_end(MYNAH_RGN_PREFILL_PROJ);
    return rc;
}

static int pocket_linear_rows(void *user, size_t layer,
                              mynah_transformer_ar_linear_kind kind,
                              const float *weight, const float *bias,
                              const float *const *in_rows, float *const *out_rows,
                              size_t batch, size_t k, size_t n) {
    pocket_tar_call *u = (pocket_tar_call *)user;
    pocket_proj proj;
    if (u == NULL || pocket_tar_proj(u->hook, layer, kind, &proj) != 0) return -1;
    return pocket_proj_rows(&proj, &u->call, weight, bias, in_rows, out_rows, batch,
                            k, n);
}

/* The flow head's `linear` hook carries no count: it is always one row. */
static int pocket_flow_linear(void *user, size_t index,
                              mynah_flow_linear_kind kind, const float *weight,
                              const float *bias, const float *in, float *out,
                              size_t k, size_t n) {
    pocket_flow_call *u = (pocket_flow_call *)user;
    pocket_proj proj;
    if (u == NULL || pocket_flow_proj(u->hook, index, kind, &proj) != 0) return -1;
    return pocket_proj_row(&proj, weight, bias, in, out, k, n);
}

static int pocket_flow_linear_rows(void *user, size_t index,
                                   mynah_flow_linear_kind kind,
                                   const float *weight, const float *bias,
                                   const float *const *in_rows,
                                   float *const *out_rows, size_t batch, size_t k,
                                   size_t n) {
    pocket_flow_call *u = (pocket_flow_call *)user;
    pocket_proj proj;
    if (u == NULL || pocket_flow_proj(u->hook, index, kind, &proj) != 0) return -1;
    return pocket_proj_rows(&proj, &u->call, weight, bias, in_rows, out_rows, batch,
                            k, n);
}

/* The three projections engine_pocket computes itself, routed the same way:
 * `group` clear means the exact matvec that was here before any of this. */
static int pocket_single_linear(const mynah_engine_state *state, unsigned group,
                                const char *key, int qtype, const float *weight,
                                const float *bias, const float *in, float *out,
                                size_t k, size_t n) {
    if ((state->qgroups & group) == 0u) {
        mynah_matvec_bias_f32(weight, in, bias, out, n, k);
        return 0;
    }
    return mynah_qmat_linear_resolved_qt(state->qcache, state->backend, key, weight,
                                         in, out, 1u, k, n, bias, qtype, NULL, 0);
}

static void pocket_call_release(pocket_call *call) {
    if (call == NULL) return;
    free(call->qx);
    free(call->sx);
    free(call->in_ptr);
    free((void *)call->out_ptr);
    memset(call, 0, sizeof(*call));
}

/* Sizes one caller's scratch.  `k_max` is the widest reduction any projection of
 * that block performs -- max(d_model, attn_dim, ffn_dim) for a transformer,
 * max(hidden, cond) for the flow head; nothing here is allowed to be a
 * constant. */
static int pocket_call_init(pocket_call *call, size_t rows, size_t k_max,
                            char *error, size_t capacity) {
    memset(call, 0, sizeof(*call));
    if (rows == 0) rows = 1u;
    size_t qbytes = 0;
    if (pocket_mul(rows, k_max, &qbytes) != 0) {
        pocket_error(error, capacity, "pocket: projection scratch overflow");
        return -1;
    }
    call->qx = (int8_t *)calloc(qbytes ? qbytes : 1u, sizeof(int8_t));
    call->sx = (float *)calloc(rows, sizeof(float));
    call->in_ptr = (const float **)calloc(rows, sizeof(*call->in_ptr));
    call->out_ptr = (float **)calloc(rows, sizeof(*call->out_ptr));
    if (call->qx == NULL || call->sx == NULL || call->in_ptr == NULL ||
        call->out_ptr == NULL) {
        pocket_call_release(call);
        pocket_error(error, capacity, "out of memory sizing a projection scratch");
        return -1;
    }
    call->rows = rows;
    call->k_max = k_max;
    return 0;
}

static int pocket_tar_call_init(pocket_tar_call *user,
                                const pocket_linear_hook *hook, size_t rows,
                                size_t k_max, char *error, size_t capacity) {
    user->hook = hook;
    user->in_prefill = 0;
    return pocket_call_init(&user->call, rows, k_max, error, capacity);
}

static int pocket_flow_call_init(pocket_flow_call *user,
                                 const pocket_flow_hook *hook, size_t rows,
                                 size_t k_max, char *error, size_t capacity) {
    user->hook = hook;
    return pocket_call_init(&user->call, rows, k_max, error, capacity);
}

/* Installs the hooks on a private copy of the model's weight struct.  The layer
 * array stays shared and read-only; only the three hook fields differ, so a
 * caller gets its own scratch without any weight being duplicated. */
static void pocket_bind_hooks(mynah_transformer_ar_weights *out,
                              const mynah_transformer_ar_weights *shared,
                              pocket_tar_call *user) {
    *out = *shared;
    out->linear = pocket_linear;
    out->linear_rows = pocket_linear_rows;
    out->linear_user = user;
}

static void pocket_bind_flow_hooks(mynah_flow_head_weights *out,
                                   const mynah_flow_head_weights *shared,
                                   pocket_flow_call *user) {
    *out = *shared;
    out->linear = pocket_flow_linear;
    out->linear_rows = pocket_flow_linear_rows;
    out->linear_user = user;
}

/* Builds the cache keys for one weight group.  `tag` keeps the backbone's keys,
 * the codec transformer's and the flow head's apart in the model-wide cache,
 * which matters because all three have a group 0. */
static int pocket_hook_init(pocket_linear_hook *hook, const char *tag,
                            size_t layers, mynah_qmat_cache *qcache,
                            const mynah_backend *backend, unsigned groups,
                            const unsigned kind_group[4],
                            const signed char *qtype_of_bit, char *error,
                            size_t capacity) {
    static const char *const kinds[4] = {"qkv", "oproj", "ffn1", "ffn2"};
    size_t slots = 0;
    size_t bytes = 0;
    if (layers == 0) layers = 1u;
    if (pocket_mul(layers, 4u, &slots) != 0 ||
        pocket_mul(slots, (size_t)POCKET_QNAME_MAX, &bytes) != 0) {
        pocket_error(error, capacity, "pocket: %s hook table overflow", tag);
        return -1;
    }
    hook->names = (char *)calloc(bytes, 1u);
    if (hook->names == NULL) {
        pocket_error(error, capacity, "out of memory building the %s hook", tag);
        return -1;
    }
    for (size_t l = 0; l < layers; ++l) {
        for (size_t kind = 0; kind < 4u; ++kind) {
            char *slot = hook->names + (l * 4u + kind) * (size_t)POCKET_QNAME_MAX;
            const int written = snprintf(slot, POCKET_QNAME_MAX, "pocket.%s.%zu.%s",
                                         tag, l, kinds[kind]);
            if (written < 0 || (size_t)written >= POCKET_QNAME_MAX) {
                pocket_error(error, capacity, "pocket: %s hook key truncated", tag);
                return -1;
            }
        }
    }
    hook->layers = layers;
    hook->qcache = qcache;
    hook->backend = backend;
    hook->groups = groups;
    hook->block = tag;   /* static: the caller passes a literal */
    for (size_t i = 0; i < 4u; ++i) {
        hook->kind_group[i] = kind_group[i];
        hook->kind_qtype[i] =
            (signed char)pocket_qtype_for(qtype_of_bit, kind_group[i]);
    }
    return 0;
}

static int pocket_flow_hook_init(pocket_flow_hook *hook, size_t indices,
                                 mynah_qmat_cache *qcache,
                                 const mynah_backend *backend, unsigned groups,
                                 const signed char *qtype_of_bit, char *error,
                                 size_t capacity) {
    static const char *const kinds[MYNAH_FLOW_LINEAR_KIND_COUNT] = {
        "tmlp1", "tmlp2", "cond", "inproj", "adaln", "mlp1", "mlp2", "fadaln",
        "fout"};
    size_t slots = 0;
    size_t bytes = 0;
    if (indices == 0) indices = 1u;
    if (pocket_mul(indices, (size_t)MYNAH_FLOW_LINEAR_KIND_COUNT, &slots) != 0 ||
        pocket_mul(slots, (size_t)POCKET_QNAME_MAX, &bytes) != 0) {
        pocket_error(error, capacity, "pocket: flow hook table overflow");
        return -1;
    }
    hook->names = (char *)calloc(bytes, 1u);
    if (hook->names == NULL) {
        pocket_error(error, capacity, "out of memory building the flow hook");
        return -1;
    }
    for (size_t i = 0; i < indices; ++i) {
        for (size_t kind = 0; kind < (size_t)MYNAH_FLOW_LINEAR_KIND_COUNT; ++kind) {
            char *slot =
                hook->names +
                (i * (size_t)MYNAH_FLOW_LINEAR_KIND_COUNT + kind) *
                    (size_t)POCKET_QNAME_MAX;
            const int written = snprintf(slot, POCKET_QNAME_MAX,
                                         "pocket.flow.%zu.%s", i, kinds[kind]);
            if (written < 0 || (size_t)written >= POCKET_QNAME_MAX) {
                pocket_error(error, capacity, "pocket: flow hook key truncated");
                return -1;
            }
        }
    }
    hook->indices = indices;
    hook->qcache = qcache;
    hook->backend = backend;
    hook->groups = groups;
    hook->core_qtype =
        (signed char)pocket_qtype_for(qtype_of_bit, POCKET_QG_FLOW_CORE);
    hook->io_qtype = (signed char)pocket_qtype_for(qtype_of_bit, POCKET_QG_FLOW_IO);
    return 0;
}

/* ------------------------------------------------------ tensor resolution */

/* Resolves `name` and checks its shape. A zero in `shape` means "any extent".
 * Every dimension that the manifest already declares is checked here, so a
 * pack whose numbers disagree with its tensors fails at load with the name in
 * the message instead of producing plausible noise. */
static int pocket_tensor(const mynah_weights *weights, const char *name,
                         size_t rank, const size_t *shape, const float **out,
                         char *error, size_t capacity) {
    mynah_tensor tensor;
    if (mynah_tensor_get(weights, name, &tensor, error, capacity) != 0) return -1;
    if (tensor.rank != rank) {
        pocket_error(error, capacity, "%s: rank %zu, expected %zu", name,
                     tensor.rank, rank);
        return -1;
    }
    for (size_t d = 0; d < rank; ++d) {
        if (shape[d] != 0 && tensor.shape[d] != shape[d]) {
            pocket_error(error, capacity, "%s: dim %zu is %zu, expected %zu", name,
                         d, tensor.shape[d], shape[d]);
            return -1;
        }
    }
    *out = tensor.data;
    return 0;
}

/* Same, but a missing tensor is not an error: *out stays NULL. */
static int pocket_tensor_optional(const mynah_weights *weights, const char *name,
                                  size_t rank, const size_t *shape,
                                  const float **out, char *error,
                                  size_t capacity) {
    mynah_tensor tensor;
    *out = NULL;
    if (mynah_weights_get(weights, name, &tensor) != 0) return 0;
    return pocket_tensor(weights, name, rank, shape, out, error, capacity);
}

static int pocket_shape_of(const mynah_weights *weights, const char *name,
                           mynah_tensor *out, char *error, size_t capacity) {
    return mynah_tensor_get(weights, name, out, error, capacity);
}

/* ------------------------------------------------------------ config load */

static char *pocket_read_text(const char *path, size_t limit, char *error,
                              size_t capacity) {
    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        pocket_error(error, capacity, "cannot open %s", path);
        return NULL;
    }
    char *data = NULL;
    size_t size = 0;
    size_t used = 0;
    for (;;) {
        if (used == size) {
            const size_t grown = (size == 0) ? 8192u : size * 2u;
            if (grown > limit + 1u) {
                free(data);
                fclose(file);
                pocket_error(error, capacity, "%s is larger than %zu bytes", path,
                             limit);
                return NULL;
            }
            char *bigger = (char *)realloc(data, grown);
            if (bigger == NULL) {
                free(data);
                fclose(file);
                pocket_error(error, capacity, "out of memory reading %s", path);
                return NULL;
            }
            data = bigger;
            size = grown;
        }
        const size_t got = fread(data + used, 1, size - used, file);
        used += got;
        if (got == 0) break;
    }
    const int failed = ferror(file);
    fclose(file);
    if (failed || data == NULL) {
        free(data);
        pocket_error(error, capacity, "cannot read %s", path);
        return NULL;
    }
    if (used == size) {
        char *bigger = (char *)realloc(data, size + 1u);
        if (bigger == NULL) {
            free(data);
            pocket_error(error, capacity, "out of memory reading %s", path);
            return NULL;
        }
        data = bigger;
    }
    data[used] = '\0';
    return data;
}

static int pocket_join(char *out, size_t capacity, const char *directory,
                       const char *name, char *error, size_t error_capacity) {
    const int written = snprintf(out, capacity, "%s/%s", directory, name);
    if (written <= 0 || (size_t)written >= capacity) {
        pocket_error(error, error_capacity, "path is too long: %s/%s", directory,
                     name);
        return -1;
    }
    return 0;
}

static int pocket_config_load(const char *manifest, pocket_config *cfg,
                              char *error, size_t capacity) {
    memset(cfg, 0, sizeof(*cfg));

    char engine[32];
    if (pj_string_copy(pj_object_get(manifest, "engine"), engine, sizeof(engine)) != 0 ||
        strcmp(engine, POCKET_ENGINE_NAME) != 0) {
        pocket_error(error, capacity,
                     "model.json does not declare engine \"" POCKET_ENGINE_NAME "\"");
        return -1;
    }

    double number = 0.0;
    if (cfg_need_size(manifest, "hidden_dim", &cfg->hidden_dim, error, capacity) != 0 ||
        cfg_need_size(manifest, "attention_heads", &cfg->heads, error, capacity) != 0 ||
        cfg_need_size(manifest, "head_dim", &cfg->head_dim, error, capacity) != 0 ||
        cfg_need_size(manifest, "transformer_layers", &cfg->layers, error, capacity) != 0 ||
        cfg_need_size(manifest, "ffn_dim", &cfg->ffn_dim, error, capacity) != 0 ||
        cfg_need_size(manifest, "text_embedding_rows", &cfg->embed_rows, error, capacity) != 0 ||
        cfg_need_size(manifest, "text_vocab_size", &cfg->vocab_size, error, capacity) != 0 ||
        cfg_need_size(manifest, "latent_dim", &cfg->latent_dim, error, capacity) != 0 ||
        cfg_need_size(manifest, "flow_dim", &cfg->flow_dim, error, capacity) != 0 ||
        cfg_need_size(manifest, "flow_res_blocks", &cfg->flow_depth, error, capacity) != 0 ||
        cfg_need_size(manifest, "flow_time_conditions", &cfg->flow_time_conds, error, capacity) != 0 ||
        cfg_need_size(manifest, "flow_time_freqs", &cfg->flow_freqs, error, capacity) != 0 ||
        cfg_need_size(manifest, "codec_dim", &cfg->codec_dim, error, capacity) != 0 ||
        cfg_need_size(manifest, "codec_upsample_stride", &cfg->upsample_stride, error, capacity) != 0 ||
        cfg_need_size(manifest, "codec_transformer_dim", &cfg->codec_tf_dim, error, capacity) != 0 ||
        cfg_need_size(manifest, "codec_transformer_ffn", &cfg->codec_tf_ffn, error, capacity) != 0 ||
        cfg_need_size(manifest, "codec_transformer_heads", &cfg->codec_tf_heads, error, capacity) != 0 ||
        cfg_need_size(manifest, "codec_transformer_layers", &cfg->codec_tf_layers, error, capacity) != 0 ||
        cfg_need_size(manifest, "codec_transformer_context", &cfg->codec_tf_context, error, capacity) != 0 ||
        cfg_need_size(manifest, "audio_channels", &cfg->audio_channels, error, capacity) != 0 ||
        cfg_need_size(manifest, "samples_per_frame", &cfg->samples_per_frame, error, capacity) != 0 ||
        cfg_need_size(manifest, "speaker_count", &cfg->speaker_count, error, capacity) != 0) {
        return -1;
    }

    if (cfg_need_double(manifest, "sample_rate", &number, error, capacity) != 0) return -1;
    if (!(number > 0.0) || number > (double)UINT_MAX) {
        pocket_error(error, capacity, "model.json: sample_rate is out of range");
        return -1;
    }
    cfg->sample_rate = (unsigned)number;
    if (cfg_need_double(manifest, "frame_rate", &cfg->frame_rate, error, capacity) != 0) {
        return -1;
    }
    if (!(cfg->frame_rate > 0.0)) {
        pocket_error(error, capacity, "model.json: frame_rate must be positive");
        return -1;
    }
    if (cfg_need_double(manifest, "eos_threshold", &number, error, capacity) != 0) return -1;
    cfg->eos_threshold = (float)number;
    if (cfg_need_double(manifest, "temperature", &number, error, capacity) != 0) return -1;
    if (!(number >= 0.0)) {
        pocket_error(error, capacity, "model.json: temperature must not be negative");
        return -1;
    }
    cfg->temperature = (float)number;

    cfg_opt_double(manifest, "layernorm_eps", &number, 1e-5);
    cfg->layernorm_eps = (float)number;
    cfg_opt_double(manifest, "flow_layernorm_eps", &number, 1e-6);
    cfg->flow_layernorm_eps = (float)number;

    cfg_opt_size(manifest, "text_padding_id", &cfg->padding_id, cfg->vocab_size);
    cfg_opt_size(manifest, "max_tokens_per_chunk", &cfg->max_tokens_per_chunk, 0u);
    cfg_opt_size(manifest, "flow_decode_steps", &cfg->flow_decode_steps, 1u);
    cfg_opt_size(manifest, "codec_downsample_stride", &cfg->downsample_stride,
                 cfg->upsample_stride);
    cfg_opt_size(manifest, "speaker_proj_input_dim", &cfg->speaker_proj_input_dim,
                 cfg->latent_dim);
    cfg_opt_bool(manifest, "insert_bos_before_voice", &cfg->insert_bos_before_voice, 0);
    cfg_opt_bool(manifest, "uses_cfg", &cfg->uses_cfg, 0);

    /* Not declared by the pack today. `min_audio_frames` is 0 because upstream
     * has no minimum; `audio_emit_frames` is 1 because E2-3 measured a single
     * carried 80 ms frame as exact, so there is nothing to wait for. */
    cfg_opt_size(manifest, "min_generated_frames", &cfg->min_audio_frames, 0u);
    cfg_opt_size(manifest, "audio_emit_frames", &cfg->audio_emit_frames, 1u);
    cfg_opt_size(manifest, "max_decoder_steps", &cfg->default_max_steps, 0u);
    /*
     * Generation does not stop at the EOS crossing: upstream records the step
     * and keeps going for `frames_after_eos` more frames, which is what closes
     * the last word instead of clipping it
     * (`models/tts_model.py::_autoregressive_generation`). The value is
     * `model_recommended_frames_after_eos` when the language config declares
     * one (only `french_24l` does, with 8); otherwise it is
     * `prepare_text_prompt`'s guess plus 2, and that guess is 3 for inputs of
     * four words or fewer and 1 otherwise.
     *
     * The word count is a property of the *text*, which the engine never sees
     * (the request carries token ids), so the default here is the long-input
     * value and a caller holding the text can override it per context with
     * `mynah_engine_pocket_set_frames_after_eos`. Verified against the oracle:
     * the crossing is at step 49 and the dump is 52 frames, i.e. exactly 3.
     */
    cfg_opt_size(manifest, "model_recommended_frames_after_eos",
                 &cfg->frames_after_eos, SIZE_MAX);
    if (cfg->frames_after_eos == SIZE_MAX) {
        cfg_opt_size(manifest, "frames_after_eos", &cfg->frames_after_eos, 3u);
    }

    /* SEANet parameters the manifest does not declare are derived from the
     * tensor shapes later; a manifest value, when present, wins. */
    cfg_opt_size(manifest, "codec_n_filters", &cfg->n_filters, 0u);
    cfg_opt_size(manifest, "codec_n_residual_layers", &cfg->n_residual_layers, SIZE_MAX);
    cfg_opt_size(manifest, "codec_compress", &cfg->compress, 0u);
    cfg_opt_size(manifest, "codec_kernel_size", &cfg->kernel_size, 0u);
    cfg_opt_size(manifest, "codec_residual_kernel_size", &cfg->residual_kernel_size, 0u);
    cfg_opt_size(manifest, "codec_last_kernel_size", &cfg->last_kernel_size, 0u);
    /* Unobservable while n_residual_layers == 1 (every dilation is 1), so it
     * is a documented default rather than a derivation. */
    cfg_opt_size(manifest, "codec_dilation_base", &cfg->dilation_base, 2u);
    cfg_opt_double(manifest, "codec_elu_alpha", &number, 1.0);
    cfg->elu_alpha = (float)number;

    cfg_opt_string(manifest, "language", cfg->language, sizeof(cfg->language),
                   "unknown");
    cfg_opt_string(manifest, "revision", cfg->revision, sizeof(cfg->revision), "");
    cfg_opt_string(manifest, "tokenizer_file", cfg->tokenizer_file,
                   sizeof(cfg->tokenizer_file), "tokenizer.model");
    cfg_opt_string(manifest, "speakers_file", cfg->speakers_file,
                   sizeof(cfg->speakers_file), "speakers.json");
    cfg_opt_string(manifest, "voices_dir", cfg->voices_dir, sizeof(cfg->voices_dir),
                   "voices");
    {
        const char *weights = pj_object_get(manifest, "weights");
        if (pj_string_copy(pj_object_get(weights, "tts"), cfg->weights_tts,
                           sizeof(cfg->weights_tts)) != 0) {
            snprintf(cfg->weights_tts, sizeof(cfg->weights_tts), "tts.safetensors");
        }
    }

    const char *ratios = pj_array_first(pj_object_get(manifest, "codec_ratios"));
    if (ratios == NULL) {
        pocket_error(error, capacity, "model.json: \"codec_ratios\" is missing or empty");
        return -1;
    }
    for (; ratios != NULL; ratios = pj_array_next(ratios)) {
        if (cfg->n_ratios >= POCKET_MAX_RATIOS) {
            pocket_error(error, capacity, "model.json: more than %u codec ratios",
                         POCKET_MAX_RATIOS);
            return -1;
        }
        size_t ratio = 0;
        if (cfg_size_value(ratios, &ratio) != 0 || ratio == 0) {
            pocket_error(error, capacity, "model.json: codec_ratios[%zu] is invalid",
                         cfg->n_ratios);
            return -1;
        }
        cfg->ratios[cfg->n_ratios++] = ratio;
    }

    if (cfg->heads == 0 || cfg->head_dim == 0 ||
        cfg->heads * cfg->head_dim != cfg->hidden_dim) {
        pocket_error(error, capacity,
                     "model.json: attention_heads %zu * head_dim %zu != hidden_dim %zu",
                     cfg->heads, cfg->head_dim, cfg->hidden_dim);
        return -1;
    }
    if (cfg->codec_tf_heads == 0 ||
        cfg->codec_tf_dim % cfg->codec_tf_heads != 0) {
        pocket_error(error, capacity,
                     "model.json: codec_transformer_dim %zu is not a multiple of "
                     "codec_transformer_heads %zu",
                     cfg->codec_tf_dim, cfg->codec_tf_heads);
        return -1;
    }
    if (cfg->codec_tf_dim != cfg->codec_dim) {
        pocket_error(error, capacity,
                     "model.json: codec_transformer_dim %zu != codec_dim %zu; this "
                     "engine has no projection between them",
                     cfg->codec_tf_dim, cfg->codec_dim);
        return -1;
    }
    if (cfg->flow_time_conds != 2u) {
        /* The two conditions are pinned at s = 0 and t = 1 below. One time
         * condition means flow matching with an ODE integrator, which is a
         * different head and a different loop, not a smaller array. */
        pocket_error(error, capacity,
                     "model.json: flow_time_conditions is %zu; only the LSD pair "
                     "(s = 0, t = 1) is implemented",
                     cfg->flow_time_conds);
        return -1;
    }
    if (cfg->flow_decode_steps != 1u) {
        pocket_error(error, capacity,
                     "model.json: flow_decode_steps is %zu; only the distilled "
                     "one-step head is implemented",
                     cfg->flow_decode_steps);
        return -1;
    }
    if (cfg->uses_cfg) {
        pocket_error(error, capacity,
                     "model.json: uses_cfg is set; CFG is distilled into these "
                     "weights and is not implemented");
        return -1;
    }
    if (cfg->vocab_size == 0 || cfg->embed_rows <= cfg->vocab_size) {
        pocket_error(error, capacity,
                     "model.json: text_embedding_rows %zu must exceed text_vocab_size %zu",
                     cfg->embed_rows, cfg->vocab_size);
        return -1;
    }
    {
        size_t hop = 1u;
        for (size_t i = 0; i < cfg->n_ratios; ++i) {
            if (pocket_mul(hop, cfg->ratios[i], &hop) != 0) {
                pocket_error(error, capacity, "model.json: codec_ratios overflow");
                return -1;
            }
        }
        size_t per_frame = 0;
        if (pocket_mul(hop, cfg->upsample_stride, &per_frame) != 0 ||
            per_frame != cfg->samples_per_frame) {
            pocket_error(error, capacity,
                         "model.json: prod(codec_ratios) * codec_upsample_stride = %zu "
                         "!= samples_per_frame %zu",
                         per_frame, cfg->samples_per_frame);
            return -1;
        }
    }
    return 0;
}

/* ------------------------------------------------------------- speakers */

/* ------------------------------------------------------------ provenance
 *
 * E3-13.  A voice file is a KV CACHE: 126-ish positions of this checkpoint's
 * own attention state, written by this checkpoint's own weights.  Handed to a
 * different model, or to a different revision of the same one, the numbers are
 * not merely a different speaker -- they are a prefix the weights never
 * produced, and upstream's symptom for it is that the model THEN NEVER EMITS
 * EOS.  Accepting such a file therefore turns a mis-copied pack into unbounded
 * generation, which for a server is a denial of service it inflicts on itself.
 * So it is refused, and refused at pack load rather than at first synthesis:
 * the pack is either coherent or it is not, and finding out on request 40 000
 * is finding out too late.
 *
 * What actually identifies a voice's provenance, in the pack we ship:
 *
 *   `speakers.json`  -- written by the converter from the same source the
 *                       weights came from -- carries `revision`, `language`
 *                       and `model_sha256`, and one `frames` count per voice.
 *   `model.json`     carries `revision` and `language`.
 *   `source.json`    carries the sha256 of the upstream weights file.
 *
 * Note what `revision` alone does NOT do: every language pack in this family
 * comes from ONE HuggingFace revision, so `revision` is identical for english
 * and italian and cannot tell them apart.  `model_sha256` can (it is per
 * language file) and so can `language`.  The chain enforced below is therefore
 * three links, not one:
 *
 *   the voice FILE     -> its speakers.json entry      (declared `frames`)
 *   speakers.json      -> the weights file             (`model_sha256`)
 *   speakers.json      -> model.json                   (`revision`, `language`)
 *
 * Each link is checked against something already inside the pack, so nothing
 * has to be hashed at load time and the cost is 26 safetensors HEADERS.
 */

/* The sha256 `source.json` records for the weights file, or "" when the file
 * is absent or says nothing.  Optional by design: a pack assembled by hand is
 * still usable, it just has one fewer link in the chain. */
static void pocket_source_weights_sha(const char *model_dir, char *out,
                                      size_t capacity) {
    if (capacity > 0) out[0] = '\0';
    char path[POCKET_PATH_MAX];
    if (pocket_join(path, sizeof(path), model_dir, "source.json", NULL, 0) != 0) {
        return;
    }
    char *text = pocket_read_text(path, POCKET_MANIFEST_MAX, NULL, 0);
    if (text == NULL) return;
    const char *first = pj_array_first(pj_object_get(text, "files"));
    for (const char *e = first; e != NULL; e = pj_array_next(e)) {
        char role[32];
        if (pj_string_copy(pj_object_get(e, "role"), role, sizeof(role)) != 0) continue;
        if (strcmp(role, "weights") != 0) continue;
        (void)pj_string_copy(pj_object_get(e, "sha256"), out, capacity);
        break;
    }
    free(text);
}

/* Every layer of one voice file, checked against the model's own dimensions,
 * with the number of cached positions handed back.
 *
 * Shared by the pack-load sweep and by `ctx_new`, because a voice that is only
 * checked on the path that opens it for use is not checked at pack load, and a
 * voice that is only checked at pack load could still be swapped underneath a
 * long-running process. */
static int pocket_voice_validate(ingot_st *file, const pocket_config *cfg,
                                 const char *voice_name, size_t *out_positions,
                                 char *error, size_t capacity) {
    char name[POCKET_NAME_MAX];
    size_t positions = 0;
    for (size_t l = 0; l < cfg->layers; ++l) {
        snprintf(name, sizeof(name), "transformer.layers.%zu.self_attn/cache", l);
        const ingot_st_tensor *tensor = ingot_st_find(file, name);
        if (tensor == NULL) {
            pocket_error(error, capacity, "voice %s has no %s", voice_name, name);
            return -1;
        }
        /* [K/V, batch, T, heads, head_dim]: the batch axis is the only thing
         * between this file and `transformer_ar`'s own layout.  Heads and
         * head_dim are the first provenance check there is -- they are the
         * model's shape, and a cache from another architecture fails here. */
        if (tensor->rank != 5 || tensor->shape[0] != 2u || tensor->shape[1] != 1u ||
            tensor->shape[3] != cfg->heads || tensor->shape[4] != cfg->head_dim) {
            pocket_error(error, capacity, "voice %s: %s is not [2, 1, T, %zu, %zu]",
                         voice_name, name, cfg->heads, cfg->head_dim);
            return -1;
        }
        if (l == 0) {
            positions = (size_t)tensor->shape[2];
        } else if ((size_t)tensor->shape[2] != positions) {
            pocket_error(error, capacity,
                         "voice %s: layer %zu has %llu positions, layer 0 has %zu",
                         voice_name, l, (unsigned long long)tensor->shape[2],
                         positions);
            return -1;
        }

        snprintf(name, sizeof(name), "transformer.layers.%zu.self_attn/offset", l);
        const ingot_st_tensor *offset = ingot_st_find(file, name);
        if (offset == NULL || offset->nelem != 1u) {
            pocket_error(error, capacity, "voice %s has no scalar %s", voice_name,
                         name);
            return -1;
        }
        float declared = 0.0f;
        if (ingot_st_to_f32(file, offset, &declared) != 0 ||
            (size_t)declared != positions) {
            /* A partially filled cache would put the NaN padding upstream's
             * `_expand_kv_cache` writes inside the prefix. */
            pocket_error(error, capacity,
                         "voice %s: %s says %g of %zu positions are valid",
                         voice_name, name, (double)declared, positions);
            return -1;
        }
    }
    if (positions == 0) {
        pocket_error(error, capacity, "voice %s is empty", voice_name);
        return -1;
    }
    *out_positions = positions;
    return 0;
}

static void pocket_voices_free(pocket_voice *voices, size_t count) {
    if (voices == NULL) return;
    for (size_t i = 0; i < count; ++i) {
        free(voices[i].name);
        free(voices[i].file);
    }
    free(voices);
}

static int pocket_voices_load(mynah_engine_state *state, const char *path,
                              char *error, size_t capacity) {
    char *text = pocket_read_text(path, POCKET_MANIFEST_MAX, error, capacity);
    if (text == NULL) return -1;

    const char *first = pj_array_first(pj_object_get(text, "voices"));
    size_t count = 0;
    for (const char *e = first; e != NULL; e = pj_array_next(e)) ++count;
    if (count == 0) {
        free(text);
        pocket_error(error, capacity, "%s lists no voices", path);
        return -1;
    }

    pocket_voice *voices = (pocket_voice *)calloc(count, sizeof(*voices));
    if (voices == NULL) {
        free(text);
        pocket_error(error, capacity, "out of memory reading %s", path);
        return -1;
    }

    size_t index = 0;
    for (const char *e = first; e != NULL && index < count; e = pj_array_next(e)) {
        char name[128];
        char file[256];
        if (pj_string_copy(pj_object_get(e, "name"), name, sizeof(name)) != 0 ||
            pj_string_copy(pj_object_get(e, "file"), file, sizeof(file)) != 0) {
            pocket_voices_free(voices, count);
            free(text);
            pocket_error(error, capacity, "%s: voice %zu has no name or file", path,
                         index);
            return -1;
        }
        voices[index].name = pocket_strdup(name, strlen(name));
        voices[index].file = pocket_strdup(file, strlen(file));
        if (voices[index].name == NULL || voices[index].file == NULL) {
            pocket_voices_free(voices, count);
            free(text);
            pocket_error(error, capacity, "out of memory reading %s", path);
            return -1;
        }
        voices[index].frames = 0u;
        {
            double declared = 0.0;
            if (pj_number(pj_object_get(e, "frames"), &declared) == 0 &&
                declared > 0.0) {
                voices[index].frames = (size_t)declared;
            }
        }
        ++index;
    }

    /* ---- provenance, link by link (E3-13) --------------------------------
     * All three come out of the pack itself, so nothing is hashed at load. */
    char declared_revision[64];
    char declared_language[32];
    char declared_sha[80];
    const int has_revision =
        pj_string_copy(pj_object_get(text, "revision"), declared_revision,
                       sizeof(declared_revision)) == 0;
    const int has_language =
        pj_string_copy(pj_object_get(text, "language"), declared_language,
                       sizeof(declared_language)) == 0;
    const int has_sha = pj_string_copy(pj_object_get(text, "model_sha256"),
                                       declared_sha, sizeof(declared_sha)) == 0;
    free(text);

    if (has_revision && state->cfg.revision[0] != '\0' &&
        strcmp(declared_revision, state->cfg.revision) != 0) {
        pocket_voices_free(voices, count);
        pocket_error(error, capacity,
                     "%s: these voices belong to revision %s, model.json is %s; a "
                     "voice KV from another revision is a prefix these weights never "
                     "produced and the model then never emits EOS",
                     path, declared_revision, state->cfg.revision);
        return -1;
    }
    /* Every language in this family shares one upstream revision, so the
     * revision check above cannot separate english from italian. This can. */
    if (has_language && state->cfg.language[0] != '\0' &&
        strcmp(state->cfg.language, "unknown") != 0 &&
        strcmp(declared_language, state->cfg.language) != 0) {
        pocket_voices_free(voices, count);
        pocket_error(error, capacity,
                     "%s: these voices are for %s, model.json says %s; the language "
                     "packs are independently trained, not fine-tuned from a shared "
                     "base, so their KV caches are not interchangeable",
                     path, declared_language, state->cfg.language);
        return -1;
    }
    if (has_sha) {
        char weights_sha[80];
        pocket_source_weights_sha(state->model_dir, weights_sha,
                                  sizeof(weights_sha));
        if (weights_sha[0] != '\0' && strcmp(weights_sha, declared_sha) != 0) {
            pocket_voices_free(voices, count);
            pocket_error(error, capacity,
                         "%s: these voices were captured from weights %.16s..., "
                         "source.json records %.16s...",
                         path, declared_sha, weights_sha);
            return -1;
        }
    }

    state->voices = voices;
    state->voice_count = count;
    if (state->cfg.speaker_count != 0 && state->cfg.speaker_count != count) {
        pocket_error(error, capacity,
                     "model.json says %zu speakers, %s lists %zu",
                     state->cfg.speaker_count, path, count);
        return -1;
    }

    /* The last link: each voice FILE against its own table entry. Opening 26
     * safetensors headers is what turns "the table is coherent" into "the
     * files are the ones the table describes" -- the case where somebody drops
     * another pack's voice into this voices/ directory. */
    for (size_t i = 0; i < count; ++i) {
        char voice_path[POCKET_PATH_MAX];
        if (pocket_join(voice_path, sizeof(voice_path), state->model_dir,
                        voices[i].file, error, capacity) != 0) {
            return -1;
        }
        ingot_st *file = NULL;
        if (ingot_st_open(&file, voice_path, error, capacity) != 0) return -1;
        size_t positions = 0;
        const int bad =
            pocket_voice_validate(file, &state->cfg, voices[i].name, &positions,
                                  error, capacity) != 0;
        ingot_st_close(file);
        if (bad) return -1;
        if (voices[i].frames != 0u && positions != voices[i].frames) {
            pocket_error(error, capacity,
                         "voice %s holds %zu positions, %s declares %zu; this file "
                         "does not belong to this pack",
                         voices[i].name, positions, path, voices[i].frames);
            return -1;
        }
    }
    return 0;
}

/* ------------------------------------------------------ weight resolution */

static int pocket_resolve_backbone(mynah_engine_state *state, char *error,
                                   size_t capacity) {
    const pocket_config *cfg = &state->cfg;
    const size_t attn_dim = cfg->heads * cfg->head_dim;
    char name[POCKET_NAME_MAX];

    state->backbone_layers = (mynah_transformer_ar_layer *)calloc(
        cfg->layers, sizeof(*state->backbone_layers));
    if (state->backbone_layers == NULL) {
        pocket_error(error, capacity, "out of memory resolving the backbone");
        return -1;
    }

    for (size_t l = 0; l < cfg->layers; ++l) {
        mynah_transformer_ar_layer *layer = &state->backbone_layers[l];
        const size_t qkv[2] = {3u * attn_dim, cfg->hidden_dim};
        const size_t out[2] = {cfg->hidden_dim, attn_dim};
        const size_t one[1] = {cfg->hidden_dim};
        const size_t up[2] = {cfg->ffn_dim, cfg->hidden_dim};
        const size_t down[2] = {cfg->hidden_dim, cfg->ffn_dim};

        snprintf(name, sizeof(name), "flow_lm.transformer.layers.%zu.self_attn.in_proj.weight", l);
        if (pocket_tensor(state->weights, name, 2, qkv, &layer->in_proj_weight, error, capacity) != 0) return -1;
        snprintf(name, sizeof(name), "flow_lm.transformer.layers.%zu.self_attn.in_proj.bias", l);
        if (pocket_tensor_optional(state->weights, name, 1, qkv, &layer->in_proj_bias, error, capacity) != 0) return -1;
        snprintf(name, sizeof(name), "flow_lm.transformer.layers.%zu.self_attn.out_proj.weight", l);
        if (pocket_tensor(state->weights, name, 2, out, &layer->out_proj_weight, error, capacity) != 0) return -1;
        snprintf(name, sizeof(name), "flow_lm.transformer.layers.%zu.self_attn.out_proj.bias", l);
        if (pocket_tensor_optional(state->weights, name, 1, one, &layer->out_proj_bias, error, capacity) != 0) return -1;
        snprintf(name, sizeof(name), "flow_lm.transformer.layers.%zu.norm1.weight", l);
        if (pocket_tensor(state->weights, name, 1, one, &layer->norm1_weight, error, capacity) != 0) return -1;
        snprintf(name, sizeof(name), "flow_lm.transformer.layers.%zu.norm1.bias", l);
        if (pocket_tensor(state->weights, name, 1, one, &layer->norm1_bias, error, capacity) != 0) return -1;
        snprintf(name, sizeof(name), "flow_lm.transformer.layers.%zu.norm2.weight", l);
        if (pocket_tensor(state->weights, name, 1, one, &layer->norm2_weight, error, capacity) != 0) return -1;
        snprintf(name, sizeof(name), "flow_lm.transformer.layers.%zu.norm2.bias", l);
        if (pocket_tensor(state->weights, name, 1, one, &layer->norm2_bias, error, capacity) != 0) return -1;
        snprintf(name, sizeof(name), "flow_lm.transformer.layers.%zu.linear1.weight", l);
        if (pocket_tensor(state->weights, name, 2, up, &layer->linear1_weight, error, capacity) != 0) return -1;
        snprintf(name, sizeof(name), "flow_lm.transformer.layers.%zu.linear2.weight", l);
        if (pocket_tensor(state->weights, name, 2, down, &layer->linear2_weight, error, capacity) != 0) return -1;
        /* PocketTTS' backbone builds nn.Identity() for both layer scales: a
         * NULL pointer is that, bit for bit. */
        layer->layer_scale_1 = NULL;
        layer->layer_scale_2 = NULL;
    }

    const size_t one[1] = {cfg->hidden_dim};
    state->backbone.layers = state->backbone_layers;
    if (pocket_tensor(state->weights, "flow_lm.out_norm.weight", 1, one,
                      &state->backbone.out_norm_weight, error, capacity) != 0 ||
        pocket_tensor(state->weights, "flow_lm.out_norm.bias", 1, one,
                      &state->backbone.out_norm_bias, error, capacity) != 0) {
        return -1;
    }
    return 0;
}

static int pocket_resolve_flow(mynah_engine_state *state, char *error,
                               size_t capacity) {
    const pocket_config *cfg = &state->cfg;
    const size_t freq_embed = 2u * cfg->flow_freqs;
    char name[POCKET_NAME_MAX];

    state->time_embed = (mynah_flow_time_embed_weights *)calloc(
        cfg->flow_time_conds, sizeof(*state->time_embed));
    state->res_blocks = (mynah_flow_res_block_weights *)calloc(
        cfg->flow_depth, sizeof(*state->res_blocks));
    if (state->time_embed == NULL || state->res_blocks == NULL) {
        pocket_error(error, capacity, "out of memory resolving the flow head");
        return -1;
    }

    const size_t cond_w[2] = {cfg->flow_dim, cfg->hidden_dim};
    const size_t cond_b[1] = {cfg->flow_dim};
    const size_t in_w[2] = {cfg->flow_dim, cfg->latent_dim};
    if (pocket_tensor(state->weights, "flow_lm.flow_net.cond_embed.weight", 2, cond_w,
                      &state->flow.cond_embed.weight, error, capacity) != 0 ||
        pocket_tensor(state->weights, "flow_lm.flow_net.cond_embed.bias", 1, cond_b,
                      &state->flow.cond_embed.bias, error, capacity) != 0 ||
        pocket_tensor(state->weights, "flow_lm.flow_net.input_proj.weight", 2, in_w,
                      &state->flow.input_proj.weight, error, capacity) != 0 ||
        pocket_tensor(state->weights, "flow_lm.flow_net.input_proj.bias", 1, cond_b,
                      &state->flow.input_proj.bias, error, capacity) != 0) {
        return -1;
    }

    for (size_t t = 0; t < cfg->flow_time_conds; ++t) {
        mynah_flow_time_embed_weights *embed = &state->time_embed[t];
        const size_t mlp0_w[2] = {cfg->flow_dim, freq_embed};
        const size_t mlp2_w[2] = {cfg->flow_dim, cfg->flow_dim};
        snprintf(name, sizeof(name), "flow_lm.flow_net.time_embed.%zu.mlp.0.weight", t);
        if (pocket_tensor(state->weights, name, 2, mlp0_w, &embed->mlp_in.weight, error, capacity) != 0) return -1;
        snprintf(name, sizeof(name), "flow_lm.flow_net.time_embed.%zu.mlp.0.bias", t);
        if (pocket_tensor(state->weights, name, 1, cond_b, &embed->mlp_in.bias, error, capacity) != 0) return -1;
        snprintf(name, sizeof(name), "flow_lm.flow_net.time_embed.%zu.mlp.2.weight", t);
        if (pocket_tensor(state->weights, name, 2, mlp2_w, &embed->mlp_out.weight, error, capacity) != 0) return -1;
        snprintf(name, sizeof(name), "flow_lm.flow_net.time_embed.%zu.mlp.2.bias", t);
        if (pocket_tensor(state->weights, name, 1, cond_b, &embed->mlp_out.bias, error, capacity) != 0) return -1;
        /* The tail normalisation gain. `freqs` is deliberately NOT read: the
         * head computes it and rounds to BF16, which reproduces the stored
         * tensor bit for bit (see flow_head.h). */
        snprintf(name, sizeof(name), "flow_lm.flow_net.time_embed.%zu.mlp.3.alpha", t);
        if (pocket_tensor(state->weights, name, 1, cond_b, &embed->alpha, error, capacity) != 0) return -1;
    }
    state->flow.time_embed = state->time_embed;

    for (size_t b = 0; b < cfg->flow_depth; ++b) {
        mynah_flow_res_block_weights *block = &state->res_blocks[b];
        const size_t adaln_w[2] = {3u * cfg->flow_dim, cfg->flow_dim};
        const size_t adaln_b[1] = {3u * cfg->flow_dim};
        const size_t mlp_w[2] = {cfg->flow_dim, cfg->flow_dim};
        snprintf(name, sizeof(name), "flow_lm.flow_net.res_blocks.%zu.in_ln.weight", b);
        if (pocket_tensor(state->weights, name, 1, cond_b, &block->in_ln_weight, error, capacity) != 0) return -1;
        snprintf(name, sizeof(name), "flow_lm.flow_net.res_blocks.%zu.in_ln.bias", b);
        if (pocket_tensor(state->weights, name, 1, cond_b, &block->in_ln_bias, error, capacity) != 0) return -1;
        snprintf(name, sizeof(name), "flow_lm.flow_net.res_blocks.%zu.adaLN_modulation.1.weight", b);
        if (pocket_tensor(state->weights, name, 2, adaln_w, &block->adaln.weight, error, capacity) != 0) return -1;
        snprintf(name, sizeof(name), "flow_lm.flow_net.res_blocks.%zu.adaLN_modulation.1.bias", b);
        if (pocket_tensor(state->weights, name, 1, adaln_b, &block->adaln.bias, error, capacity) != 0) return -1;
        snprintf(name, sizeof(name), "flow_lm.flow_net.res_blocks.%zu.mlp.0.weight", b);
        if (pocket_tensor(state->weights, name, 2, mlp_w, &block->mlp_in.weight, error, capacity) != 0) return -1;
        snprintf(name, sizeof(name), "flow_lm.flow_net.res_blocks.%zu.mlp.0.bias", b);
        if (pocket_tensor(state->weights, name, 1, cond_b, &block->mlp_in.bias, error, capacity) != 0) return -1;
        snprintf(name, sizeof(name), "flow_lm.flow_net.res_blocks.%zu.mlp.2.weight", b);
        if (pocket_tensor(state->weights, name, 2, mlp_w, &block->mlp_out.weight, error, capacity) != 0) return -1;
        snprintf(name, sizeof(name), "flow_lm.flow_net.res_blocks.%zu.mlp.2.bias", b);
        if (pocket_tensor(state->weights, name, 1, cond_b, &block->mlp_out.bias, error, capacity) != 0) return -1;
    }
    state->flow.res_blocks = state->res_blocks;

    const size_t final_w[2] = {2u * cfg->flow_dim, cfg->flow_dim};
    const size_t final_b[1] = {2u * cfg->flow_dim};
    const size_t linear_w[2] = {cfg->latent_dim, cfg->flow_dim};
    const size_t linear_b[1] = {cfg->latent_dim};
    if (pocket_tensor(state->weights, "flow_lm.flow_net.final_layer.adaLN_modulation.1.weight",
                      2, final_w, &state->flow.final_adaln.weight, error, capacity) != 0 ||
        pocket_tensor(state->weights, "flow_lm.flow_net.final_layer.adaLN_modulation.1.bias",
                      1, final_b, &state->flow.final_adaln.bias, error, capacity) != 0 ||
        pocket_tensor(state->weights, "flow_lm.flow_net.final_layer.linear.weight", 2,
                      linear_w, &state->flow.final_linear.weight, error, capacity) != 0 ||
        pocket_tensor(state->weights, "flow_lm.flow_net.final_layer.linear.bias", 1,
                      linear_b, &state->flow.final_linear.bias, error, capacity) != 0) {
        return -1;
    }
    return 0;
}

static int pocket_resolve_codec_transformer(mynah_engine_state *state, char *error,
                                            size_t capacity) {
    const pocket_config *cfg = &state->cfg;
    const size_t dim = cfg->codec_tf_dim;
    char name[POCKET_NAME_MAX];

    state->codec_layers = (mynah_transformer_ar_layer *)calloc(
        cfg->codec_tf_layers, sizeof(*state->codec_layers));
    if (state->codec_layers == NULL) {
        pocket_error(error, capacity, "out of memory resolving the codec transformer");
        return -1;
    }

    for (size_t l = 0; l < cfg->codec_tf_layers; ++l) {
        mynah_transformer_ar_layer *layer = &state->codec_layers[l];
        const size_t qkv[2] = {3u * dim, dim};
        const size_t square[2] = {dim, dim};
        const size_t one[1] = {dim};
        const size_t up[2] = {cfg->codec_tf_ffn, dim};
        const size_t down[2] = {dim, cfg->codec_tf_ffn};

        snprintf(name, sizeof(name), "mimi.decoder_transformer.transformer.layers.%zu.self_attn.in_proj.weight", l);
        if (pocket_tensor(state->weights, name, 2, qkv, &layer->in_proj_weight, error, capacity) != 0) return -1;
        snprintf(name, sizeof(name), "mimi.decoder_transformer.transformer.layers.%zu.self_attn.out_proj.weight", l);
        if (pocket_tensor(state->weights, name, 2, square, &layer->out_proj_weight, error, capacity) != 0) return -1;
        snprintf(name, sizeof(name), "mimi.decoder_transformer.transformer.layers.%zu.norm1.weight", l);
        if (pocket_tensor(state->weights, name, 1, one, &layer->norm1_weight, error, capacity) != 0) return -1;
        snprintf(name, sizeof(name), "mimi.decoder_transformer.transformer.layers.%zu.norm1.bias", l);
        if (pocket_tensor(state->weights, name, 1, one, &layer->norm1_bias, error, capacity) != 0) return -1;
        snprintf(name, sizeof(name), "mimi.decoder_transformer.transformer.layers.%zu.norm2.weight", l);
        if (pocket_tensor(state->weights, name, 1, one, &layer->norm2_weight, error, capacity) != 0) return -1;
        snprintf(name, sizeof(name), "mimi.decoder_transformer.transformer.layers.%zu.norm2.bias", l);
        if (pocket_tensor(state->weights, name, 1, one, &layer->norm2_bias, error, capacity) != 0) return -1;
        snprintf(name, sizeof(name), "mimi.decoder_transformer.transformer.layers.%zu.linear1.weight", l);
        if (pocket_tensor(state->weights, name, 2, up, &layer->linear1_weight, error, capacity) != 0) return -1;
        snprintf(name, sizeof(name), "mimi.decoder_transformer.transformer.layers.%zu.linear2.weight", l);
        if (pocket_tensor(state->weights, name, 2, down, &layer->linear2_weight, error, capacity) != 0) return -1;
        /* Mimi builds LayerScale(0.01) here, unlike the backbone. */
        snprintf(name, sizeof(name), "mimi.decoder_transformer.transformer.layers.%zu.layer_scale_1.scale", l);
        if (pocket_tensor(state->weights, name, 1, one, &layer->layer_scale_1, error, capacity) != 0) return -1;
        snprintf(name, sizeof(name), "mimi.decoder_transformer.transformer.layers.%zu.layer_scale_2.scale", l);
        if (pocket_tensor(state->weights, name, 1, one, &layer->layer_scale_2, error, capacity) != 0) return -1;
    }
    state->codec_transformer.layers = state->codec_layers;
    state->codec_transformer.out_norm_weight = NULL;
    state->codec_transformer.out_norm_bias = NULL;
    return 0;
}

/*
 * The SEANet decoder's module indices, and the four shape facts the manifest
 * does not carry (filters, residual depth, compression, kernel sizes), come
 * from the checkpoint itself. Upstream's module list per stage is
 * [ELU, ConvTranspose1d, ResidualBlock x n_residual_layers], with one entry
 * conv before and [ELU, conv] after, which fixes every index below.
 */
static int pocket_resolve_decoder(mynah_engine_state *state, char *error,
                                  size_t capacity) {
    pocket_config *cfg = &state->cfg;
    char name[POCKET_NAME_MAX];
    mynah_tensor tensor;

    if (cfg->n_residual_layers == SIZE_MAX) {
        size_t blocks = 0;
        for (size_t i = 0; i < 256u; ++i) {
            snprintf(name, sizeof(name), "mimi.decoder.model.%zu.block.1.conv.weight", i);
            if (mynah_weights_get(state->weights, name, &tensor) == 0) ++blocks;
        }
        if (blocks % cfg->n_ratios != 0) {
            pocket_error(error, capacity,
                         "mimi.decoder: %zu residual blocks do not divide into %zu stages",
                         blocks, cfg->n_ratios);
            return -1;
        }
        cfg->n_residual_layers = blocks / cfg->n_ratios;
    }
    const size_t per_stage = 2u + cfg->n_residual_layers;

    /* The entry convolution fixes dimension, kernel size and the filter count. */
    if (pocket_shape_of(state->weights, "mimi.decoder.model.0.conv.weight", &tensor,
                        error, capacity) != 0) {
        return -1;
    }
    if (tensor.rank != 3 || tensor.shape[1] != cfg->codec_dim) {
        pocket_error(error, capacity,
                     "mimi.decoder.model.0.conv.weight has shape rank %zu, in %zu; "
                     "expected [*, codec_dim %zu, k]",
                     tensor.rank, tensor.rank > 1 ? tensor.shape[1] : 0u, cfg->codec_dim);
        return -1;
    }
    if (cfg->kernel_size == 0) cfg->kernel_size = tensor.shape[2];
    {
        size_t filters = tensor.shape[0];
        for (size_t i = 0; i < cfg->n_ratios; ++i) {
            if (filters % 2u != 0) {
                pocket_error(error, capacity,
                             "mimi.decoder: %zu entry channels do not halve %zu times",
                             tensor.shape[0], cfg->n_ratios);
                return -1;
            }
            filters /= 2u;
        }
        if (cfg->n_filters == 0) cfg->n_filters = filters;
        if (cfg->n_filters != filters) {
            pocket_error(error, capacity,
                         "model.json codec_n_filters %zu disagrees with the tensors (%zu)",
                         cfg->n_filters, filters);
            return -1;
        }
    }

    /* The last convolution fixes the audio channel count and its kernel. */
    const size_t last_index = 1u + cfg->n_ratios * per_stage + 1u;
    snprintf(name, sizeof(name), "mimi.decoder.model.%zu.conv.weight", last_index);
    if (pocket_shape_of(state->weights, name, &tensor, error, capacity) != 0) return -1;
    if (tensor.rank != 3 || tensor.shape[0] != cfg->audio_channels ||
        tensor.shape[1] != cfg->n_filters) {
        pocket_error(error, capacity,
                     "%s: expected [audio_channels %zu, n_filters %zu, k]", name,
                     cfg->audio_channels, cfg->n_filters);
        return -1;
    }
    if (cfg->last_kernel_size == 0) cfg->last_kernel_size = tensor.shape[2];

    state->decoder_convtr = (mynah_conv_weights *)calloc(cfg->n_ratios,
                                                         sizeof(*state->decoder_convtr));
    size_t block_count = 0;
    if (pocket_mul(cfg->n_ratios, cfg->n_residual_layers, &block_count) != 0) {
        pocket_error(error, capacity, "mimi.decoder: residual block count overflow");
        return -1;
    }
    state->decoder_blocks = (mynah_seanet_resblock_weights *)calloc(
        block_count == 0 ? 1u : block_count, sizeof(*state->decoder_blocks));
    if (state->decoder_convtr == NULL || state->decoder_blocks == NULL) {
        pocket_error(error, capacity, "out of memory resolving the SEANet decoder");
        return -1;
    }

    size_t channels = cfg->n_filters << cfg->n_ratios;
    for (size_t stage = 0; stage < cfg->n_ratios; ++stage) {
        const size_t base = 1u + stage * per_stage;
        const size_t out_channels = channels / 2u;
        const size_t ratio = cfg->ratios[stage];
        const size_t convtr_w[3] = {channels, out_channels, 2u * ratio};
        const size_t convtr_b[1] = {out_channels};

        snprintf(name, sizeof(name), "mimi.decoder.model.%zu.convtr.weight", base + 1u);
        if (pocket_tensor(state->weights, name, 3, convtr_w,
                          &state->decoder_convtr[stage].weight, error, capacity) != 0) {
            return -1;
        }
        snprintf(name, sizeof(name), "mimi.decoder.model.%zu.convtr.bias", base + 1u);
        if (pocket_tensor_optional(state->weights, name, 1, convtr_b,
                                   &state->decoder_convtr[stage].bias, error,
                                   capacity) != 0) {
            return -1;
        }

        for (size_t j = 0; j < cfg->n_residual_layers; ++j) {
            const size_t index = base + 2u + j;
            mynah_seanet_resblock_weights *block =
                &state->decoder_blocks[stage * cfg->n_residual_layers + j];
            snprintf(name, sizeof(name), "mimi.decoder.model.%zu.block.1.conv.weight", index);
            if (pocket_shape_of(state->weights, name, &tensor, error, capacity) != 0) return -1;
            if (tensor.rank != 3 || tensor.shape[1] != out_channels ||
                tensor.shape[0] == 0) {
                pocket_error(error, capacity, "%s: expected [hidden, %zu, k]", name,
                             out_channels);
                return -1;
            }
            if (cfg->residual_kernel_size == 0) cfg->residual_kernel_size = tensor.shape[2];
            if (cfg->compress == 0) {
                if (out_channels % tensor.shape[0] != 0) {
                    pocket_error(error, capacity,
                                 "%s: %zu channels do not divide %zu", name,
                                 tensor.shape[0], out_channels);
                    return -1;
                }
                cfg->compress = out_channels / tensor.shape[0];
            }
            const size_t hidden = out_channels / cfg->compress;
            const size_t conv1_w[3] = {hidden, out_channels, cfg->residual_kernel_size};
            const size_t conv1_b[1] = {hidden};
            const size_t conv2_w[3] = {out_channels, hidden, 1u};
            const size_t conv2_b[1] = {out_channels};
            if (pocket_tensor(state->weights, name, 3, conv1_w, &block->conv1.weight,
                              error, capacity) != 0) {
                return -1;
            }
            snprintf(name, sizeof(name), "mimi.decoder.model.%zu.block.1.conv.bias", index);
            if (pocket_tensor_optional(state->weights, name, 1, conv1_b,
                                       &block->conv1.bias, error, capacity) != 0) {
                return -1;
            }
            snprintf(name, sizeof(name), "mimi.decoder.model.%zu.block.3.conv.weight", index);
            if (pocket_tensor(state->weights, name, 3, conv2_w, &block->conv2.weight,
                              error, capacity) != 0) {
                return -1;
            }
            snprintf(name, sizeof(name), "mimi.decoder.model.%zu.block.3.conv.bias", index);
            if (pocket_tensor_optional(state->weights, name, 1, conv2_b,
                                       &block->conv2.bias, error, capacity) != 0) {
                return -1;
            }
        }
        channels = out_channels;
    }
    if (cfg->compress == 0) cfg->compress = 1u;
    if (cfg->residual_kernel_size == 0) cfg->residual_kernel_size = 1u;

    {
        const size_t first_w[3] = {cfg->n_filters << cfg->n_ratios, cfg->codec_dim,
                                   cfg->kernel_size};
        const size_t first_b[1] = {cfg->n_filters << cfg->n_ratios};
        if (pocket_tensor(state->weights, "mimi.decoder.model.0.conv.weight", 3, first_w,
                          &state->decoder.first.weight, error, capacity) != 0 ||
            pocket_tensor_optional(state->weights, "mimi.decoder.model.0.conv.bias", 1,
                                   first_b, &state->decoder.first.bias, error,
                                   capacity) != 0) {
            return -1;
        }
    }
    {
        const size_t last_w[3] = {cfg->audio_channels, cfg->n_filters,
                                  cfg->last_kernel_size};
        const size_t last_b[1] = {cfg->audio_channels};
        snprintf(name, sizeof(name), "mimi.decoder.model.%zu.conv.weight", last_index);
        if (pocket_tensor(state->weights, name, 3, last_w, &state->decoder.last.weight,
                          error, capacity) != 0) {
            return -1;
        }
        snprintf(name, sizeof(name), "mimi.decoder.model.%zu.conv.bias", last_index);
        if (pocket_tensor_optional(state->weights, name, 1, last_b,
                                   &state->decoder.last.bias, error, capacity) != 0) {
            return -1;
        }
    }
    state->decoder.convtr = state->decoder_convtr;
    state->decoder.blocks = state->decoder_blocks;

    /* Mimi's ConvTrUpsample1d: depthwise, kernel 2 * stride, no bias. */
    {
        const size_t up_w[3] = {cfg->codec_dim, 1u, 2u * cfg->upsample_stride};
        const size_t up_b[1] = {cfg->codec_dim};
        if (pocket_tensor(state->weights, "mimi.upsample.convtr.convtr.weight", 3, up_w,
                          &state->upsample.weight, error, capacity) != 0 ||
            pocket_tensor_optional(state->weights, "mimi.upsample.convtr.convtr.bias", 1,
                                   up_b, &state->upsample.bias, error, capacity) != 0) {
            return -1;
        }
    }
    return 0;
}

static int pocket_resolve_singles(mynah_engine_state *state, char *error,
                                  size_t capacity) {
    const pocket_config *cfg = &state->cfg;
    const size_t embed[2] = {cfg->embed_rows, cfg->hidden_dim};
    const size_t latent[1] = {cfg->latent_dim};
    const size_t input_w[2] = {cfg->hidden_dim, cfg->latent_dim};
    const size_t eos_w[2] = {1u, cfg->hidden_dim};
    const size_t eos_b[1] = {1u};
    const size_t quant_w[3] = {cfg->codec_dim, cfg->latent_dim, 1u};
    const size_t speaker_w[2] = {cfg->hidden_dim, cfg->speaker_proj_input_dim};
    const size_t bos_before[3] = {1u, 1u, cfg->hidden_dim};

    if (pocket_tensor(state->weights, "flow_lm.conditioner.embed.weight", 2, embed,
                      &state->embed_table, error, capacity) != 0 ||
        pocket_tensor(state->weights, "flow_lm.bos_emb", 1, latent, &state->bos_emb,
                      error, capacity) != 0 ||
        pocket_tensor(state->weights, "flow_lm.emb_mean", 1, latent, &state->emb_mean,
                      error, capacity) != 0 ||
        pocket_tensor(state->weights, "flow_lm.emb_std", 1, latent, &state->emb_std,
                      error, capacity) != 0 ||
        pocket_tensor(state->weights, "flow_lm.input_linear.weight", 2, input_w,
                      &state->input_linear, error, capacity) != 0 ||
        pocket_tensor(state->weights, "flow_lm.out_eos.weight", 2, eos_w,
                      &state->out_eos_weight, error, capacity) != 0 ||
        pocket_tensor(state->weights, "flow_lm.out_eos.bias", 1, eos_b,
                      &state->out_eos_bias, error, capacity) != 0 ||
        pocket_tensor(state->weights, "mimi.quantizer.output_proj.weight", 3, quant_w,
                      &state->quantizer_proj, error, capacity) != 0) {
        return -1;
    }
    /* Cloning-only, and the two tensors that differ between generations. They
     * are validated when present so a mismatched pack fails here rather than
     * the first time someone clones a voice. */
    if (pocket_tensor_optional(state->weights, "flow_lm.speaker_proj_weight", 2,
                               speaker_w, &state->speaker_proj, error, capacity) != 0 ||
        pocket_tensor_optional(state->weights, "flow_lm.bos_before_voice", 3, bos_before,
                               &state->bos_before_voice, error, capacity) != 0) {
        return -1;
    }
    if (cfg->insert_bos_before_voice && state->bos_before_voice == NULL) {
        pocket_error(error, capacity,
                     "model.json sets insert_bos_before_voice but "
                     "flow_lm.bos_before_voice is missing");
        return -1;
    }
    return 0;
}

/* -------------------------------------------------------------- model_init */

static void pocket_model_free(mynah_engine_state *state) {
    if (state == NULL) return;
    mynah_sp_close(state->tokenizer);
    pocket_voices_free(state->voices, state->voice_count);
    free(state->backbone_layers);
    free(state->time_embed);
    free(state->res_blocks);
    free(state->codec_layers);
    free(state->decoder_convtr);
    free(state->decoder_blocks);
    free(state->backbone_hook.names);
    free(state->codec_hook.names);
    free(state->flow_hook.names);
    if (state->owns_weights) mynah_weights_close(state->weights);
    free(state->model_dir);
    free(state);
}

/* ------------------------------------------------- eager weight prepacking
 *
 * WHAT THIS REPLACES.  `mynah_qmat_linear_resolved_qt` and its batched twin
 * build a tensor's quantized copy on FIRST TOUCH: `cache_lookup` misses,
 * `cache_insert` mallocs the packed buffer and converts the whole f32 weight
 * into it, under the cache mutex, on the calling thread.  For the PocketTTS
 * backbone that is 24 tensors and 302 MB of f32 read into 151 MB of f16 -- and
 * the first thing that ever touches those tensors is the text prefill of the
 * first request.
 *
 * MEASURED, GCP Axion (32x Neoverse-V2), `--runs 1` against `--runs 8`, which
 * separates the first call of `prep.decoder_prefill` from the steady-state
 * ones:
 *
 *     cores        1       4       8      16
 *     first     442.3   282.2   255.1   242.9  ms
 *     steady    230.0    68.2    39.4    27.5  ms
 *     first-steady  212.3   214.0   215.7   215.4  ms
 *
 * The steady-state prefill divides 8.4x across 16 cores.  The difference does
 * not move at all -- 212/214/216/215 -- because packing a weight is one thread
 * converting an array while every other thread waits on a mutex it cannot
 * help with.  That flat ~215 ms is the whole of the "prefill does not scale"
 * finding, and `MYNAH_QUANT_GROUPS=none` removes it entirely (first call 38.2
 * ms, steady 37.4 ms, difference 0.8 ms), which is what says it is the pack
 * and not a page-in of the mapped weights.
 *
 * WHY HERE AND NOT IN THE REQUEST.  The cost is per PROCESS, not per request,
 * so it was never a rate -- it was a cliff sitting on whichever request
 * arrived first.  Moving it to model open does three things a warm-up request
 * cannot:
 *
 *   1. It stops being latency.  A caller that opens a model and synthesizes
 *      once -- the CLI, an embedded user, a server started `--no-warmup` --
 *      pays 215 ms of TTFA today for work that has nothing to do with its
 *      text.
 *   2. It lands BEFORE the prefork fork.  `server/main.c` opens every pack
 *      before forking, deliberately, "so the mapped weights are one physical
 *      copy behind the whole tree"; the packed copies were not, because they
 *      were built by each worker's own warm-up AFTER the fork.  Built here
 *      they are ordinary heap pages of the parent and the children share them
 *      copy-on-write like everything else: at 16 workers that is 16 x 151 MB
 *      of resident f16 collapsing to 151 MB, and 16 x 215 ms of startup
 *      collapsing to 215 ms.
 *   3. It makes the cost map honest.  The work is reported under
 *      `runtime.weight_prepack`, inside `runtime.model_load`, instead of
 *      inflating a decoder region by 28% of a request's wall.
 *
 * WHY IT CANNOT MOVE THE AUDIO.  A cache entry is a pure function of (name,
 * weight bytes, n, k, resolved qtype), and every consumer resolves the qtype
 * the same way -- from the group spec, before the cache is touched (E8-5).
 * This walk asks `pocket_tar_proj` / `pocket_flow_proj` for exactly the
 * `pocket_proj` the real call site will build, and hands
 * `mynah_qmat_linear_resolved_qt` the same name, weight pointer, shape and
 * qtype that call would.  So the entry that exists afterwards is the entry
 * that would have existed anyway; the only difference is when it was built and
 * by whom.  A group the spec does not select is skipped, because touching it
 * would CREATE an entry the lazy path never would.
 *
 * The one row it computes is thrown away.  It is the price of building the
 * entry through the public linear rather than through a cache API this file is
 * not allowed to add; it is one matvec against 24 tensors and it does not
 * appear in the shape census, because a prepack is not a graph op and
 * recording it as one would put 37 calls into a census of a run that made
 * none.
 *
 * `MYNAH_PREPACK=0` turns it off, and it is also skipped under
 * `MYNAH_QMAT_ACT_STATS`, where the throwaway row would enter the activation
 * statistic as a call the model never made.
 */
/* ONCE PER CACHE, and the guard is not decoration.
 *
 * `mynah_engine_state` is not the model: `serve()` in src/inference.c calls
 * `engine->model_init` at the top of every driver call and `model_free` at the
 * bottom, so this function runs once per server lifetime but once per
 * `mynah_tts_synthesize()` for a one-shot caller.  The weight cache underneath
 * it is the MODEL's and outlives all of them, so the second call would find
 * every entry already built -- and would still pay one full matvec per tensor
 * to find that out.  Measured before this guard existed: +2.1 ms per repeated
 * serve at 16 threads and +14.2 ms at one, charged to `runtime.model_load`.
 * A prepack that makes the tenth request slower has stopped being a prepack.
 *
 * The memo is the cache pointer, because that is what owns the entries.  A
 * pointer reused by a later cache allocated at the same address costs a prepack
 * that is skipped; a full table costs the same, deliberately, because the
 * failure mode of "prepack anyway" is a cost on every serve and the failure mode
 * of "skip" is the lazy path this function exists to anticipate.  Neither can
 * reach the audio: the entry `cache_insert` builds is the same whoever asks for
 * it.  Inherited across fork(), which is correct -- a child's cache really is
 * packed, copy-on-write.
 */
#define POCKET_PREPACK_MEMO 8
static pthread_mutex_t g_prepack_mutex = PTHREAD_MUTEX_INITIALIZER;
static const void *g_prepacked[POCKET_PREPACK_MEMO];
static size_t g_prepacked_n;

static int pocket_prepack_claim(const void *cache) {
    if (cache == NULL) return 0;
    pthread_mutex_lock(&g_prepack_mutex);
    for (size_t i = 0; i < g_prepacked_n; ++i) {
        if (g_prepacked[i] == cache) {
            pthread_mutex_unlock(&g_prepack_mutex);
            return 0;
        }
    }
    const int room = g_prepacked_n < POCKET_PREPACK_MEMO;
    if (room) g_prepacked[g_prepacked_n++] = cache;
    pthread_mutex_unlock(&g_prepack_mutex);
    return room;
}

static int pocket_prepack_enabled(void) {
    const char *off = getenv("MYNAH_PREPACK");
    if (off != NULL && off[0] == '0' && off[1] == '\0') return 0;
    const char *stats = getenv("MYNAH_QMAT_ACT_STATS");
    if (stats != NULL && stats[0] != '\0' && strcmp(stats, "0") != 0) return 0;
    return 1;
}

/* One tensor, through the same call the graph would make. */
static void pocket_prepack_one(const pocket_proj *p, const float *weight,
                               const float *bias, const float *in, float *out,
                               size_t k, size_t n) {
    if (!p->quantized || weight == NULL || k == 0u || n == 0u) return;
    /* The return is deliberately ignored: a prepack that cannot build an entry
     * has changed nothing -- the lazy path will try again and take whatever
     * fallback it would have taken -- so failing the model open over it would
     * turn a latency optimisation into an outage. */
    (void)mynah_qmat_linear_resolved_qt(p->qcache, p->backend, p->name, weight,
                                        in, out, 1u, k, n, bias, p->qtype, NULL,
                                        0u);
}

/* The four projections of one transformer_ar layer, shaped exactly as
 * `mynah_transformer_ar_linear_kind` documents them and as
 * `pocket_resolve_backbone` validated them against model.json. */
static void pocket_prepack_tar(const pocket_linear_hook *hook,
                               const mynah_transformer_ar_layer *layers,
                               size_t count, size_t d_model, size_t attn_dim,
                               size_t ffn_dim, const float *in, float *out) {
    if (layers == NULL) return;
    for (size_t l = 0; l < count; ++l) {
        const mynah_transformer_ar_layer *w = &layers[l];
        const float *weight[4] = {w->in_proj_weight, w->out_proj_weight,
                                  w->linear1_weight, w->linear2_weight};
        const float *bias[4] = {w->in_proj_bias, w->out_proj_bias,
                                w->linear1_bias, w->linear2_bias};
        const size_t k[4] = {d_model, attn_dim, d_model, ffn_dim};
        const size_t n[4] = {3u * attn_dim, d_model, ffn_dim, d_model};
        for (size_t kind = 0; kind < 4u; ++kind) {
            pocket_proj proj;
            if (pocket_tar_proj(hook, l, (mynah_transformer_ar_linear_kind)kind,
                                &proj) != 0) {
                continue;
            }
            pocket_prepack_one(&proj, weight[kind], bias[kind], in, out, k[kind],
                               n[kind]);
        }
    }
}

static void pocket_prepack_flow_one(const pocket_flow_hook *hook, size_t index,
                                    mynah_flow_linear_kind kind,
                                    const mynah_flow_linear *w, size_t k,
                                    size_t n, const float *in, float *out) {
    pocket_proj proj;
    if (w == NULL || pocket_flow_proj(hook, index, kind, &proj) != 0) return;
    pocket_prepack_one(&proj, w->weight, w->bias, in, out, k, n);
}

/* Every projection the engine can route through the weight cache, built once,
 * here.  Returns nothing: see `pocket_prepack_one`. */
static void pocket_prepack(mynah_engine_state *state) {
    if (state == NULL || state->qgroups == 0u || !pocket_prepack_enabled()) return;
    if (!pocket_prepack_claim(state->qcache)) return;
    const pocket_config *cfg = &state->cfg;
    const size_t attn_dim = cfg->heads * cfg->head_dim;
    const size_t codec_attn = cfg->codec_tf_dim;

    /* Wide enough for the widest reduction and the widest output of any
     * projection below, computed from the manifest the same way `ctx_new`
     * sizes its own scratch.  `in` is zero, which is a legal activation on
     * every path: f16 needs no activation scale at all, and int8/int4 take the
     * `amax == 0` branch of `quantize_act`. */
    size_t k_max = cfg->hidden_dim;
    if (attn_dim > k_max) k_max = attn_dim;
    if (cfg->ffn_dim > k_max) k_max = cfg->ffn_dim;
    if (codec_attn > k_max) k_max = codec_attn;
    if (cfg->codec_tf_ffn > k_max) k_max = cfg->codec_tf_ffn;
    if (cfg->flow_dim > k_max) k_max = cfg->flow_dim;
    if (2u * cfg->flow_freqs > k_max) k_max = 2u * cfg->flow_freqs;
    size_t n_max = 3u * attn_dim;
    if (3u * codec_attn > n_max) n_max = 3u * codec_attn;
    if (cfg->ffn_dim > n_max) n_max = cfg->ffn_dim;
    if (cfg->codec_tf_ffn > n_max) n_max = cfg->codec_tf_ffn;
    if (3u * cfg->flow_dim > n_max) n_max = 3u * cfg->flow_dim;
    if (cfg->codec_dim > n_max) n_max = cfg->codec_dim;
    if (cfg->hidden_dim > n_max) n_max = cfg->hidden_dim;

    float *in = (float *)calloc(k_max ? k_max : 1u, sizeof(float));
    float *out = (float *)calloc(n_max ? n_max : 1u, sizeof(float));
    if (in == NULL || out == NULL) {
        free(in);
        free(out);
        return;
    }

    mynah_region_begin(MYNAH_RGN_PREPACK);
    pocket_prepack_tar(&state->backbone_hook, state->backbone_layers, cfg->layers,
                       cfg->hidden_dim, attn_dim, cfg->ffn_dim, in, out);
    pocket_prepack_tar(&state->codec_hook, state->codec_layers,
                       cfg->codec_tf_layers, cfg->codec_tf_dim, codec_attn,
                       cfg->codec_tf_ffn, in, out);

    const size_t hidden = cfg->flow_dim;
    for (size_t t = 0; t < cfg->flow_time_conds; ++t) {
        const mynah_flow_time_embed_weights *e = &state->time_embed[t];
        pocket_prepack_flow_one(&state->flow_hook, t,
                                MYNAH_FLOW_LINEAR_TIME_MLP_IN, &e->mlp_in,
                                2u * cfg->flow_freqs, hidden, in, out);
        pocket_prepack_flow_one(&state->flow_hook, t,
                                MYNAH_FLOW_LINEAR_TIME_MLP_OUT, &e->mlp_out,
                                hidden, hidden, in, out);
    }
    for (size_t b = 0; b < cfg->flow_depth; ++b) {
        const mynah_flow_res_block_weights *r = &state->res_blocks[b];
        pocket_prepack_flow_one(&state->flow_hook, b,
                                MYNAH_FLOW_LINEAR_BLOCK_ADALN, &r->adaln, hidden,
                                3u * hidden, in, out);
        pocket_prepack_flow_one(&state->flow_hook, b,
                                MYNAH_FLOW_LINEAR_BLOCK_MLP_IN, &r->mlp_in,
                                hidden, hidden, in, out);
        pocket_prepack_flow_one(&state->flow_hook, b,
                                MYNAH_FLOW_LINEAR_BLOCK_MLP_OUT, &r->mlp_out,
                                hidden, hidden, in, out);
    }
    pocket_prepack_flow_one(&state->flow_hook, 0u, MYNAH_FLOW_LINEAR_COND_EMBED,
                            &state->flow.cond_embed, cfg->hidden_dim, hidden, in,
                            out);
    pocket_prepack_flow_one(&state->flow_hook, 0u, MYNAH_FLOW_LINEAR_INPUT_PROJ,
                            &state->flow.input_proj, cfg->latent_dim, hidden, in,
                            out);
    pocket_prepack_flow_one(&state->flow_hook, 0u, MYNAH_FLOW_LINEAR_FINAL_ADALN,
                            &state->flow.final_adaln, hidden, 2u * hidden, in,
                            out);
    pocket_prepack_flow_one(&state->flow_hook, 0u, MYNAH_FLOW_LINEAR_FINAL_LINEAR,
                            &state->flow.final_linear, hidden, cfg->latent_dim,
                            in, out);

    /* The three the engine projects itself, keyed and routed by
     * `pocket_single_linear`; the shapes are its own call sites'. */
    {
        const unsigned group[3] = {POCKET_QG_COND_IN, POCKET_QG_COND_EOS,
                                   POCKET_QG_CODEC_CONV};
        const char *key[3] = {state->cond_in_key, state->cond_eos_key,
                              state->codec_conv_key};
        const int qtype[3] = {state->cond_in_qtype, state->cond_eos_qtype,
                              state->codec_conv_qtype};
        const float *weight[3] = {state->input_linear, state->out_eos_weight,
                                  state->quantizer_proj};
        const float *bias[3] = {NULL, state->out_eos_bias, NULL};
        const size_t k[3] = {cfg->latent_dim, cfg->hidden_dim, cfg->latent_dim};
        const size_t n[3] = {cfg->hidden_dim, 1u, cfg->codec_dim};
        for (size_t i = 0; i < 3u; ++i) {
            if ((state->qgroups & group[i]) == 0u || weight[i] == NULL) continue;
            (void)mynah_qmat_linear_resolved_qt(state->qcache, state->backend,
                                                key[i], weight[i], in, out, 1u,
                                                k[i], n[i], bias[i], qtype[i],
                                                NULL, 0u);
        }
    }
    mynah_region_end(MYNAH_RGN_PREPACK);
    free(in);
    free(out);
}

static int pocket_model_init(const mynah_tts_model *model,
                             mynah_engine_state **out, char *error,
                             size_t capacity) {
    if (out != NULL) *out = NULL;
    if (model == NULL || out == NULL || model->model_dir == NULL) {
        pocket_error(error, capacity, "pocket: no model to initialise");
        return -1;
    }

    mynah_engine_state *state = (mynah_engine_state *)calloc(1, sizeof(*state));
    if (state == NULL) {
        pocket_error(error, capacity, "out of memory creating the pocket engine");
        return -1;
    }
    state->model_dir = pocket_strdup(model->model_dir, strlen(model->model_dir));
    if (state->model_dir == NULL) {
        pocket_model_free(state);
        pocket_error(error, capacity, "out of memory creating the pocket engine");
        return -1;
    }

    char path[POCKET_PATH_MAX];
    if (pocket_join(path, sizeof(path), state->model_dir, "model.json", error,
                    capacity) != 0) {
        pocket_model_free(state);
        return -1;
    }
    char *manifest = pocket_read_text(path, POCKET_MANIFEST_MAX, error, capacity);
    if (manifest == NULL) {
        pocket_model_free(state);
        return -1;
    }
    const int configured = pocket_config_load(manifest, &state->cfg, error, capacity);
    free(manifest);
    if (configured != 0) {
        pocket_model_free(state);
        return -1;
    }
    if (state->cfg.default_max_steps == 0) {
        state->cfg.default_max_steps = model->info.max_decoder_steps;
    }

    /* The pack loader owns the tensors when it has already opened them; a
     * caller that hands over a model without them gets them opened here, and
     * this engine then owns that handle. */
    if (model->tts != NULL) {
        state->weights = model->tts;
        state->owns_weights = 0;
    } else {
        if (pocket_join(path, sizeof(path), state->model_dir, state->cfg.weights_tts,
                        error, capacity) != 0 ||
            mynah_weights_open(path, &state->weights, error, capacity) != 0) {
            pocket_model_free(state);
            return -1;
        }
        state->owns_weights = 1;
    }

    if (pocket_resolve_singles(state, error, capacity) != 0 ||
        pocket_resolve_backbone(state, error, capacity) != 0 ||
        pocket_resolve_flow(state, error, capacity) != 0 ||
        pocket_resolve_codec_transformer(state, error, capacity) != 0 ||
        pocket_resolve_decoder(state, error, capacity) != 0) {
        pocket_model_free(state);
        return -1;
    }

    /* The projection path.
     *
     * Two independent decisions meet here.  WHAT is quantized is
     * MYNAH_QUANT_GROUPS, resolved once and validated so that a typo fails the
     * load instead of silently changing the measurement it was meant to steer.
     * HOW MANY ROWS a call carries is the hook's business and is decided per
     * call, which is why the key tables are built unconditionally: a tile of 16
     * codec positions is one call rather than sixteen even in a build where
     * nothing is quantized at all, and there the tile takes one exact f32 matmul
     * instead of sixteen matvecs.
     *
     * Binding is per CALLER, not here.  `linear_user` has to carry the caller's
     * own activation scratch, so a context binds its own in `ctx_new` and the
     * driver binds one for the batch in `scratch_new`.  Binding unconditionally
     * costs nothing numerically: an unselected group's single row takes
     * `mynah_matvec_bias_f32`, which is exactly what `transformer_ar` and
     * `flow_head` compute with no hook at all. */
    state->backend = model->backend;
    state->qcache = model->qcache;
    for (size_t b = 0; b < POCKET_QG_BITS; ++b) state->qgroup_qtype[b] = -1;
    state->qgroups = 0u;
    if (mynah_qmat_cache_enabled(state->qcache)) {
        const char *spec = mynah_qmat_groups_spec();
        if (strcmp(spec, "default") == 0) {
            /* With no MYNAH_QUANT to obey, the default names its own encodings
             * rather than inheriting a base that has changed underneath it. */
            spec = (mynah_qmat_qtype_from_env() < 0) ? POCKET_QG_DEFAULT_SPEC_PINNED
                                                      : POCKET_QG_DEFAULT_SPEC;
        }
        if (pocket_qgroups_parse(spec, &state->qgroups, state->qgroup_qtype, error,
                                 capacity) != 0) {
            pocket_model_free(state);
            return -1;
        }
    }
    state->cond_in_qtype =
        (signed char)pocket_qtype_for(state->qgroup_qtype, POCKET_QG_COND_IN);
    state->cond_eos_qtype =
        (signed char)pocket_qtype_for(state->qgroup_qtype, POCKET_QG_COND_EOS);
    state->codec_conv_qtype =
        (signed char)pocket_qtype_for(state->qgroup_qtype, POCKET_QG_CODEC_CONV);
    state->codec_convtr_qtype =
        (signed char)pocket_qtype_for(state->qgroup_qtype, POCKET_QG_CODEC_CONVTR);
    {
        static const unsigned bb_kinds[4] = {POCKET_QG_BB_QKV, POCKET_QG_BB_OPROJ,
                                             POCKET_QG_BB_FFN1, POCKET_QG_BB_FFN2};
        static const unsigned ct_kinds[4] = {POCKET_QG_CT_QKV, POCKET_QG_CT_OPROJ,
                                             POCKET_QG_CT_FFN1, POCKET_QG_CT_FFN2};
        const size_t flow_indices =
            (state->cfg.flow_depth > state->cfg.flow_time_conds)
                ? state->cfg.flow_depth
                : state->cfg.flow_time_conds;
        if (pocket_hook_init(&state->backbone_hook, "bb", state->cfg.layers,
                             state->qcache, state->backend, state->qgroups,
                             bb_kinds, state->qgroup_qtype, error, capacity) != 0 ||
            pocket_hook_init(&state->codec_hook, "codec", state->cfg.codec_tf_layers,
                             state->qcache, state->backend, state->qgroups,
                             ct_kinds, state->qgroup_qtype, error, capacity) != 0 ||
            pocket_flow_hook_init(&state->flow_hook, flow_indices, state->qcache,
                                  state->backend, state->qgroups,
                                  state->qgroup_qtype, error, capacity) != 0) {
            pocket_model_free(state);
            return -1;
        }
        snprintf(state->cond_in_key, sizeof(state->cond_in_key), "pocket.cond.in");
        snprintf(state->cond_eos_key, sizeof(state->cond_eos_key), "pocket.cond.eos");
        snprintf(state->codec_conv_key, sizeof(state->codec_conv_key),
                 "pocket.codec.qproj");
    }

    if (pocket_join(path, sizeof(path), state->model_dir, state->cfg.speakers_file,
                    error, capacity) != 0 ||
        pocket_voices_load(state, path, error, capacity) != 0) {
        pocket_model_free(state);
        return -1;
    }

    if (pocket_join(path, sizeof(path), state->model_dir, state->cfg.tokenizer_file,
                    error, capacity) != 0 ||
        mynah_sp_open(path, &state->tokenizer, error, capacity) != 0) {
        pocket_model_free(state);
        return -1;
    }
    if (mynah_sp_vocab_size(state->tokenizer) != state->cfg.vocab_size) {
        pocket_error(error, capacity,
                     "%s has %zu pieces, model.json says text_vocab_size %zu", path,
                     mynah_sp_vocab_size(state->tokenizer), state->cfg.vocab_size);
        pocket_model_free(state);
        return -1;
    }

    pocket_prepack(state);

    *out = state;
    if (error != NULL && capacity > 0) error[0] = '\0';
    return 0;
}

/* A NULL state is the engine-level question -- "what could this engine do with
 * this model at all" -- which `mynah_tts_model_max_batch` asks before any state
 * exists.  Refusing it made the public API answer 1 for an engine that batches
 * 16, so it is answered from `model->info`, and the fields that only the loaded
 * pack knows (the streaming emit threshold, the latent width) stay zero rather
 * than being guessed.  The driver always passes a real state and checks the
 * fields it needs, so it cannot pick up a half-answer by accident. */
static int pocket_caps(const mynah_tts_model *model,
                       const mynah_engine_state *state, mynah_engine_caps *out) {
    if (out == NULL) return -1;
    if (state == NULL) {
        if (model == NULL) return -1;
        memset(out, 0, sizeof(*out));
        out->sample_rate = model->info.sample_rate;
        out->frame_rate = model->info.frame_rate;
        out->frames_per_step = 1u;
        out->min_audio_frames = model->info.min_generated_frames;
        out->default_max_steps = model->info.max_decoder_steps;
        out->max_batch = POCKET_MAX_BATCH;
        out->voice_count = model->info.speaker_count;
        out->is_discrete_codec = 0u;
        return 0;
    }
    (void)model;
    const pocket_config *cfg = &state->cfg;
    memset(out, 0, sizeof(*out));
    out->sample_rate = cfg->sample_rate;
    out->frame_rate = cfg->frame_rate;
    /* One AR step is one latent, and one latent is one 80 ms frame: there is
     * no stacking and no delay pattern in a continuous-latent model. */
    out->frames_per_step = 1u;
    out->audio_emit_frames = (unsigned)cfg->audio_emit_frames;
    out->min_audio_frames = (unsigned)cfg->min_audio_frames;
    out->default_max_steps = (unsigned)cfg->default_max_steps;
    out->max_batch = POCKET_MAX_BATCH;
    out->voice_count = (unsigned)state->voice_count;
    out->needs_cfg = 0u;
    out->is_discrete_codec = 0u;
    out->latent_dim = (unsigned)cfg->latent_dim;
    return 0;
}

/* ----------------------------------------------------------------- context */

static int pocket_all_finite(const float *v, size_t n);

static int pocket_cuda_resident_requested(const mynah_engine_state *state) {
    if (state == NULL || state->backend == NULL ||
        strcmp(mynah_backend_name(state->backend), "cuda") != 0) {
        return 0;
    }
    const char *setting = getenv("MYNAH_CUDA_RESIDENT");
    return setting == NULL || strcmp(setting, "0") != 0;
}

static int pocket_cuda_flow_requested(const mynah_engine_state *state) {
    if (!pocket_cuda_resident_requested(state)) return 0;
    const char *setting = getenv("MYNAH_CUDA_FLOW");
    return setting == NULL || strcmp(setting, "0") != 0;
}

/* The released Pocket configuration pins the LSD time pair at {0, 1}.  Build
 * the same BF16-rounded sinusoidal features once into persistent staging so a
 * flow graph replay never depends on a stack buffer or a host allocation. */
static int pocket_cuda_flow_time_features(const pocket_config *cfg, float *out) {
    if (cfg == NULL || out == NULL || cfg->flow_time_conds != 2u ||
        cfg->flow_freqs == 0u) return -1;
    const double log_period = log(10000.0);
    const size_t half = cfg->flow_freqs;
    for (size_t t = 0; t < 2u; ++t) {
        const float time = t == 0u ? 0.0f : 1.0f;
        float *row = out + t * 2u * half;
        for (size_t i = 0; i < half; ++i) {
            float freq = (float)exp(-log_period * (double)i / (double)half);
            uint32_t bits = 0u;
            memcpy(&bits, &freq, sizeof(bits));
            bits = (bits + UINT32_C(0x7fff) + ((bits >> 16) & 1u)) &
                   UINT32_C(0xffff0000);
            memcpy(&freq, &bits, sizeof(freq));
            const float arg = time * freq;
            row[i] = cosf(arg);
            row[half + i] = sinf(arg);
        }
    }
    return 0;
}

static void pocket_cuda_backbone_release(mynah_engine_ctx *ctx) {
    if (ctx == NULL) return;
    const mynah_backend *backend =
        (ctx->state == NULL) ? NULL : ctx->state->backend;
    if (backend != NULL) {
        mynah_backend_dev_free(backend, ctx->cuda_backbone_kv);
        mynah_backend_dev_free(backend, ctx->cuda_x);
        mynah_backend_dev_free(backend, ctx->cuda_norm);
        mynah_backend_dev_free(backend, ctx->cuda_qkv);
        mynah_backend_dev_free(backend, ctx->cuda_attn);
        mynah_backend_dev_free(backend, ctx->cuda_proj);
        mynah_backend_dev_free(backend, ctx->cuda_ffn);
    }
    ctx->cuda_backbone_kv = NULL;
    ctx->cuda_x = NULL;
    ctx->cuda_norm = NULL;
    ctx->cuda_qkv = NULL;
    ctx->cuda_attn = NULL;
    ctx->cuda_proj = NULL;
    ctx->cuda_ffn = NULL;
    ctx->cuda_backbone_capacity = 0u;
    ctx->cuda_backbone_kv_floats = 0u;
    ctx->cuda_backbone_valid = 0;
}

#define POCKET_CUDA_DECODER_GRAPH_KEY ((size_t)0x200000u)

static void pocket_cuda_decoder_release(mynah_engine_ctx *ctx) {
    if (ctx == NULL) return;
    const mynah_backend *backend =
        ctx->state == NULL ? NULL : ctx->state->backend;
    if (backend != NULL) {
        if (ctx->cuda_decoder != NULL)
            mynah_backend_graph_forget(backend, ctx->cuda_decoder);
        mynah_backend_dev_free(backend, ctx->cuda_decoder_input);
        mynah_backend_dev_free(backend, ctx->cuda_decoder_output);
        mynah_backend_decoder_close(backend, ctx->cuda_decoder);
    }
    ctx->cuda_decoder = NULL;
    ctx->cuda_decoder_input = NULL;
    ctx->cuda_decoder_output = NULL;
    ctx->cuda_decoder_enabled = 0;
    ctx->cuda_decoder_graph_enabled = 0;
    ctx->cuda_decoder_started = 0;
}

static int pocket_cuda_decoder_alloc(mynah_engine_ctx *ctx) {
    if (ctx == NULL || !ctx->cuda_decoder_enabled || ctx->state == NULL ||
        ctx->state->backend == NULL ||
        strcmp(mynah_backend_name(ctx->state->backend), "cuda") != 0) return 0;
    const mynah_engine_state *state = ctx->state;
    const pocket_config *cfg = &state->cfg;
    mynah_backend_decoder_desc desc;
    memset(&desc, 0, sizeof(desc));
    desc.channels = cfg->audio_channels;
    desc.dimension = cfg->codec_dim;
    desc.n_filters = cfg->n_filters;
    desc.n_residual_layers = cfg->n_residual_layers;
    desc.ratios = cfg->ratios;
    desc.n_ratios = cfg->n_ratios;
    desc.kernel_size = cfg->kernel_size;
    desc.residual_kernel_size = cfg->residual_kernel_size;
    desc.last_kernel_size = cfg->last_kernel_size;
    desc.dilation_base = cfg->dilation_base;
    desc.compress = cfg->compress;
    desc.elu_alpha = cfg->elu_alpha;
    desc.first = state->decoder.first;
    desc.convtr = state->decoder.convtr;
    desc.blocks = state->decoder.blocks;
    desc.last = state->decoder.last;

    char ignored[256];
    ignored[0] = '\0';
    if (mynah_backend_decoder_open(state->backend, &desc, cfg->upsample_stride,
                                   &ctx->cuda_decoder, ignored,
                                   sizeof(ignored)) != 0) {
        ctx->cuda_decoder_enabled = 0;
        return 0; /* optional backend: CPU remains authoritative */
    }
    size_t input_floats = 0;
    size_t output_floats = 0;
    if (pocket_mul(cfg->codec_dim, cfg->upsample_stride, &input_floats) != 0 ||
        pocket_mul(cfg->audio_channels, cfg->samples_per_frame, &output_floats) != 0 ||
        mynah_backend_dev_alloc(state->backend, input_floats,
                                &ctx->cuda_decoder_input, ignored,
                                sizeof(ignored)) != 0 ||
        mynah_backend_dev_alloc(state->backend, output_floats,
                                &ctx->cuda_decoder_output, ignored,
                                sizeof(ignored)) != 0) {
        pocket_cuda_decoder_release(ctx);
        return 0;
    }
    const char *graphs = getenv("MYNAH_CUDA_GRAPHS");
    ctx->cuda_decoder_graph_enabled = graphs == NULL || strcmp(graphs, "0") != 0;
    ctx->cuda_decoder_started = 0;
    return 0;
}

/* Return the resident decoder's per-step buffer sizes.  Keeping this check in
 * one place matters because the submit/collect split below is also used by the
 * frame-major server gang: a failed size calculation must remain a pre-state
 * optional-backend fallback, never a half-submitted decoder step. */
static int pocket_cuda_decoder_sizes(const mynah_engine_ctx *ctx,
                                     size_t *input_floats,
                                     size_t *output_floats) {
    if (input_floats != NULL) *input_floats = 0u;
    if (output_floats != NULL) *output_floats = 0u;
    if (ctx == NULL || ctx->state == NULL || input_floats == NULL ||
        output_floats == NULL) return -1;
    const pocket_config *cfg = &ctx->state->cfg;
    if (pocket_mul(cfg->codec_dim, cfg->upsample_stride, input_floats) != 0 ||
        pocket_mul(cfg->audio_channels, cfg->samples_per_frame,
                   output_floats) != 0) return -1;
    return 0;
}

/* Return 0 when the resident decoder has queued this frame, 1 when the
 * optional path is unavailable before it touched decoder state, and -1 after
 * a launch/graph error. Once the GPU rings have advanced, silently switching
 * to the CPU rings would be wrong, so that last case is request-fatal.
 *
 * This function intentionally stops before D2H and synchronization. The
 * frame-major gang uses it for every active context, queues all output copies,
 * then drains the common CUDA stream once. */
static int pocket_cuda_decoder_submit(mynah_engine_ctx *ctx, char *error,
                                      size_t capacity) {
    if (ctx == NULL || !ctx->cuda_decoder_enabled || ctx->cuda_decoder == NULL)
        return 1;
    const mynah_backend *backend = ctx->state->backend;
    const pocket_config *cfg = &ctx->state->cfg;
    size_t input_floats = 0u;
    size_t output_floats = 0u;
    if (pocket_cuda_decoder_sizes(ctx, &input_floats, &output_floats) != 0) {
        if (ctx->cuda_decoder_started) {
            pocket_error(error, capacity, "CUDA decoder size overflow after state advance");
            return -1;
        }
        pocket_cuda_decoder_release(ctx);
        return 1;
    }
    (void)output_floats;
    char local[256];
    local[0] = '\0';
    if (mynah_backend_h2d(backend, ctx->codec_back, ctx->cuda_decoder_input,
                          input_floats, local, sizeof(local)) != 0) {
        if (ctx->cuda_decoder_started) {
            pocket_error(error, capacity, "%s", local[0] != '\0'
                                               ? local
                                               : "CUDA decoder input copy failed after state advance");
            return -1;
        }
        pocket_cuda_decoder_release(ctx);
        return 1;
    }

    int capturing = 0;
    int replay = 0;
    int graph = 1;
    if (ctx->cuda_decoder_graph_enabled) {
        const int status = mynah_backend_graph_begin(
            backend, POCKET_CUDA_DECODER_GRAPH_KEY, ctx->cuda_decoder, &replay,
            local, sizeof(local));
        if (status < 0) {
            if (ctx->cuda_decoder_started) {
                pocket_error(error, capacity, "%s", local[0] != '\0'
                                                   ? local
                                                   : "CUDA decoder graph setup failed after state advance");
                return -1;
            }
            pocket_cuda_decoder_release(ctx);
            return 1;
        }
        if (status == 0) {
            capturing = replay == 0;
        } else {
            graph = 0;
            ctx->cuda_decoder_graph_enabled = 0;
        }
    } else {
        graph = 0;
    }

    if (capturing || !graph) {
        /* The first graph capture does not execute the kernels until replay;
         * a non-graph call executes them directly. */
        ctx->cuda_decoder_started = 1;
        if (mynah_backend_decoder_step(backend, ctx->cuda_decoder,
                                       ctx->cuda_decoder_input,
                                       cfg->upsample_stride,
                                       ctx->cuda_decoder_output, local,
                                       sizeof(local)) != 0) {
            if (capturing) mynah_backend_graph_abort(
                backend, POCKET_CUDA_DECODER_GRAPH_KEY, ctx->cuda_decoder);
            pocket_error(error, capacity, "%s", local[0] != '\0' ? local
                                                                  : "CUDA decoder failed");
            return -1;
        }
        if (capturing) {
            if (mynah_backend_graph_end(backend, POCKET_CUDA_DECODER_GRAPH_KEY,
                                         ctx->cuda_decoder, local,
                                         sizeof(local)) == 0) {
                capturing = 0;
                if (mynah_backend_graph_launch(backend,
                                               POCKET_CUDA_DECODER_GRAPH_KEY,
                                               ctx->cuda_decoder, local,
                                               sizeof(local)) != 0) {
                    pocket_error(error, capacity, "%s", local[0] != '\0'
                                      ? local : "CUDA decoder graph launch failed");
                    return -1;
                }
                (void)mynah_backend_decoder_note_step(backend,
                                                       ctx->cuda_decoder);
            } else {
                /* Capture is an optimization. Its failed capture did not
                 * execute the recorded commands; run one ordinary step. */
                ctx->cuda_decoder_graph_enabled = 0;
                if (mynah_backend_decoder_step(backend, ctx->cuda_decoder,
                                               ctx->cuda_decoder_input,
                                               cfg->upsample_stride,
                                               ctx->cuda_decoder_output, local,
                                               sizeof(local)) != 0) {
                    pocket_error(error, capacity, "%s", local[0] != '\0'
                                  ? local : "CUDA decoder fallback failed");
                    return -1;
                }
            }
        }
    } else if (replay) {
        ctx->cuda_decoder_started = 1;
        if (mynah_backend_graph_launch(backend, POCKET_CUDA_DECODER_GRAPH_KEY,
                                       ctx->cuda_decoder, local,
                                       sizeof(local)) != 0) {
            pocket_error(error, capacity, "%s", local[0] != '\0' ? local
                                                                  : "CUDA decoder graph replay failed");
            return -1;
        }
        (void)mynah_backend_decoder_note_step(backend, ctx->cuda_decoder);
    }
    return 0;
}

/* Queue the resident decoder output into its host PCM buffer.  No sync is
 * allowed here; callers must drain the backend once after all contexts in the
 * gang have submitted their copies. */
static int pocket_cuda_decoder_collect(mynah_engine_ctx *ctx, char *error,
                                       size_t capacity) {
    if (ctx == NULL || !ctx->cuda_decoder_enabled || ctx->cuda_decoder == NULL)
        return -1;
    size_t input_floats = 0u;
    size_t output_floats = 0u;
    if (pocket_cuda_decoder_sizes(ctx, &input_floats, &output_floats) != 0) {
        pocket_error(error, capacity, "CUDA decoder output size overflow");
        return -1;
    }
    (void)input_floats;
    char local[256];
    local[0] = '\0';
    if (mynah_backend_d2h(ctx->state->backend, ctx->cuda_decoder_output,
                          ctx->pcm, output_floats, local, sizeof(local)) != 0) {
        pocket_error(error, capacity, "%s", local[0] != '\0' ? local
                                                              : "CUDA decoder output copy failed");
        return -1;
    }
    return 0;
}

/* Single-context compatibility wrapper.  Offline and streaming callers keep
 * exactly the old semantics; only the gang scheduler uses submit/collect
 * directly. */
static int pocket_cuda_decoder_step(mynah_engine_ctx *ctx, char *error,
                                    size_t capacity) {
    const int submitted = pocket_cuda_decoder_submit(ctx, error, capacity);
    if (submitted != 0) return submitted;
    if (pocket_cuda_decoder_collect(ctx, error, capacity) != 0) return -1;
    char local[256];
    local[0] = '\0';
    if (mynah_backend_sync(ctx->state->backend, local, sizeof(local)) != 0) {
        pocket_error(error, capacity, "%s", local[0] != '\0' ? local
                                                              : "CUDA decoder synchronization failed");
        return -1;
    }
    size_t input_floats = 0u;
    size_t output_floats = 0u;
    if (pocket_cuda_decoder_sizes(ctx, &input_floats, &output_floats) != 0) {
        pocket_error(error, capacity, "CUDA decoder output size overflow");
        return -1;
    }
    (void)input_floats;
    if (!pocket_all_finite(ctx->pcm, output_floats)) {
        pocket_error(error, capacity, "CUDA decoder produced non-finite PCM");
        return -1;
    }
    return 0;
}

static int pocket_cuda_backbone_alloc(mynah_engine_ctx *ctx) {
    if (ctx == NULL || !ctx->cuda_backbone_enabled || ctx->backbone == NULL) {
        return 0;
    }
    const pocket_config *cfg = &ctx->state->cfg;
    const mynah_transformer_ar_config *bc =
        mynah_transformer_ar_state_config(ctx->backbone);
    if (bc == NULL || cfg->heads == 0u || cfg->head_dim == 0u ||
        cfg->layers == 0u || cfg->hidden_dim == 0u || cfg->ffn_dim == 0u ||
        fabsf(cfg->layernorm_eps - 1.0e-5f) > 1.0e-7f) {
        /* The CUDA layer-norm kernel currently implements the transformer
         * family's fixed 1e-5 epsilon.  Unsupported metadata stays on CPU. */
        ctx->cuda_backbone_enabled = 0;
        return 0;
    }
    for (size_t l = 0; l < cfg->layers; ++l) {
        if (ctx->state->backbone_layers[l].layer_scale_1 != NULL ||
            ctx->state->backbone_layers[l].layer_scale_2 != NULL) {
            ctx->cuda_backbone_enabled = 0;
            return 0;
        }
    }

    size_t attn_dim = 0;
    size_t layer_half = 0;
    size_t layer_span = 0;
    size_t kv_floats = 0;
    size_t qkv = 0;
    if (pocket_mul(cfg->heads, cfg->head_dim, &attn_dim) != 0 ||
        pocket_mul(bc->max_seq_len, attn_dim, &layer_half) != 0 ||
        pocket_mul(layer_half, 2u, &layer_span) != 0 ||
        pocket_mul(cfg->layers, layer_span, &kv_floats) != 0 ||
        pocket_mul(attn_dim, 3u, &qkv) != 0) {
        ctx->cuda_backbone_enabled = 0;
        return 0;
    }

    char ignored[256];
#define POCKET_CUDA_ALLOC(field, count)                                      \
    do {                                                                      \
        ignored[0] = '\0';                                                    \
        if (mynah_backend_dev_alloc(ctx->state->backend, (count), &(field),  \
                                    ignored, sizeof(ignored)) != 0) {          \
            pocket_cuda_backbone_release(ctx);                                \
            ctx->cuda_backbone_enabled = 0;                                   \
            return 0;                                                         \
        }                                                                       \
    } while (0)
    POCKET_CUDA_ALLOC(ctx->cuda_backbone_kv, kv_floats);
    POCKET_CUDA_ALLOC(ctx->cuda_x, cfg->hidden_dim);
    POCKET_CUDA_ALLOC(ctx->cuda_norm, cfg->hidden_dim);
    POCKET_CUDA_ALLOC(ctx->cuda_qkv, qkv);
    POCKET_CUDA_ALLOC(ctx->cuda_attn, attn_dim);
    POCKET_CUDA_ALLOC(ctx->cuda_proj, cfg->hidden_dim);
    POCKET_CUDA_ALLOC(ctx->cuda_ffn, cfg->ffn_dim);
#undef POCKET_CUDA_ALLOC
    ctx->cuda_backbone_capacity = bc->max_seq_len;
    ctx->cuda_backbone_kv_floats = kv_floats;
    ctx->cuda_backbone_valid = 0;
    return 0;
}

static int pocket_cuda_backbone_upload(mynah_engine_ctx *ctx, char *error,
                                       size_t capacity) {
    if (ctx == NULL || !ctx->cuda_backbone_enabled ||
        ctx->cuda_backbone_kv == NULL) {
        return 1;
    }
    const pocket_config *cfg = &ctx->state->cfg;
    const mynah_transformer_ar_config *bc =
        mynah_transformer_ar_state_config(ctx->backbone);
    const size_t half = mynah_transformer_ar_state_kv_half_floats(ctx->backbone);
    size_t attn_dim = 0;
    size_t device_layer_half = 0;
    size_t layer_span = 0;
    size_t kv_floats = 0;
    size_t valid_floats = 0;
    const size_t position = mynah_transformer_ar_state_offset(ctx->backbone);
    if (bc == NULL || position > bc->max_seq_len ||
        pocket_mul(cfg->heads, cfg->head_dim, &attn_dim) != 0 ||
        pocket_mul(bc->max_seq_len, attn_dim, &device_layer_half) != 0 ||
        pocket_mul(half, 2u, &layer_span) != 0 ||
        pocket_mul(cfg->layers, layer_span, &kv_floats) != 0 ||
        kv_floats > ctx->cuda_backbone_kv_floats ||
        pocket_mul(position, attn_dim, &valid_floats) != 0 ||
        valid_floats > half) {
        pocket_error(error, capacity, "pocket: CUDA KV upload size overflow");
        return -1;
    }
    for (size_t l = 0; l < cfg->layers; ++l) {
        float *destination = ctx->cuda_backbone_kv + l * (device_layer_half * 2u);
        const float *source = mynah_transformer_ar_state_kv(ctx->backbone, l);
        if (source == NULL) {
            ctx->cuda_backbone_valid = 0;
            return -1;
        }
        /* The host cache is [K capacity][V capacity], but only the prefix up
         * to the current offset can ever be read by attention.  Uploading the
         * zero/unwritten tail made every new request pay for the full KV
         * capacity (and, on 24L, dominated H2D traffic).  Keep the device
         * layout unchanged and copy just the two live contiguous prefixes.
         * A fully populated cache remains one copy, preserving the fast path
         * for a future caller that legitimately fills the whole capacity. */
        if (position == bc->max_seq_len && half == device_layer_half) {
            if (mynah_backend_h2d(ctx->state->backend, source, destination,
                                  layer_span / cfg->layers, error,
                                  capacity) != 0) {
                ctx->cuda_backbone_valid = 0;
                return -1;
            }
        } else if (valid_floats > 0u &&
                   (mynah_backend_h2d(ctx->state->backend, source, destination,
                                      valid_floats, error, capacity) != 0 ||
                    mynah_backend_h2d(ctx->state->backend,
                                      source + half,
                                      destination + device_layer_half,
                                      valid_floats, error, capacity) != 0)) {
            ctx->cuda_backbone_valid = 0;
            return -1;
        }
    }
    ctx->cuda_backbone_valid = 1;
    return 0;
}

/* One resident Pocket autoregressive step.  The host state is deliberately
 * committed only after the device output has been downloaded and validated;
 * this lets the caller retry the same position on CPU after a recoverable CUDA
 * failure without consuming a different KV position. */
static int pocket_cuda_backbone_step(mynah_engine_ctx *ctx, char *error,
                                     size_t capacity) {
    if (ctx == NULL || !ctx->cuda_backbone_enabled ||
        ctx->cuda_backbone_kv == NULL) {
        return 1; /* unavailable: use the normal CPU path */
    }
    const pocket_config *cfg = &ctx->state->cfg;
    const mynah_backend *backend = ctx->state->backend;
    const mynah_transformer_ar_config *bc =
        mynah_transformer_ar_state_config(ctx->backbone);
    if (bc == NULL || ctx->cuda_backbone_capacity != bc->max_seq_len ||
        ctx->cuda_x == NULL || ctx->cuda_norm == NULL || ctx->cuda_qkv == NULL ||
        ctx->cuda_attn == NULL || ctx->cuda_proj == NULL || ctx->cuda_ffn == NULL) {
        return 1;
    }
    const size_t position = mynah_transformer_ar_state_offset(ctx->backbone);
    if (position >= bc->max_seq_len) {
        pocket_error(error, capacity, "pocket: CUDA backbone KV capacity exhausted");
        return -1;
    }
    const size_t attn_dim = cfg->heads * cfg->head_dim;
    const size_t layer_half = bc->max_seq_len * attn_dim;
    char local[256];
    char drain_error[256];
    local[0] = '\0';
    if (mynah_backend_batch_begin(backend, local, sizeof(local)) != 0) goto fail;
    if (!ctx->cuda_backbone_valid &&
        pocket_cuda_backbone_upload(ctx, local, sizeof(local)) != 0) goto fail;
    if (mynah_backend_h2d(backend, ctx->step_input, ctx->cuda_x,
                          cfg->hidden_dim, local, sizeof(local)) != 0) goto fail;

    for (size_t l = 0; l < cfg->layers; ++l) {
        const mynah_transformer_ar_layer *layer = &ctx->state->backbone_layers[l];
        if (mynah_backend_layer_norm_dev(backend, ctx->cuda_x, ctx->cuda_norm,
                                         layer->norm1_weight, layer->norm1_bias,
                                         1u, cfg->hidden_dim, local,
                                         sizeof(local)) != 0 ||
            mynah_backend_matmul_d2d(backend, ctx->cuda_norm, ctx->cuda_qkv,
                                     1u, cfg->hidden_dim, 3u * attn_dim,
                                     layer->in_proj_weight, layer->in_proj_bias,
                                     local, sizeof(local)) != 0 ||
            mynah_backend_rope_dev(backend, ctx->cuda_qkv, position, cfg->heads,
                                   cfg->head_dim, bc->max_period, local,
                                   sizeof(local)) != 0) goto fail;
        float *layer_kv = ctx->cuda_backbone_kv + l * (2u * layer_half);
        if (mynah_backend_self_attention_dev(
                backend, ctx->cuda_qkv, layer_kv, layer_kv + layer_half,
                position, attn_dim, position + 1u, cfg->heads, cfg->head_dim,
                1.0f / sqrtf((float)cfg->head_dim), ctx->cuda_attn, local,
                sizeof(local)) != 0 ||
            mynah_backend_matmul_d2d(backend, ctx->cuda_attn, ctx->cuda_proj,
                                     1u, attn_dim, cfg->hidden_dim,
                                     layer->out_proj_weight, layer->out_proj_bias,
                                     local, sizeof(local)) != 0 ||
            mynah_backend_residual_add_dev(backend, ctx->cuda_x, ctx->cuda_proj,
                                           cfg->hidden_dim, local,
                                           sizeof(local)) != 0 ||
            mynah_backend_layer_norm_dev(backend, ctx->cuda_x, ctx->cuda_norm,
                                         layer->norm2_weight, layer->norm2_bias,
                                         1u, cfg->hidden_dim, local,
                                         sizeof(local)) != 0 ||
            mynah_backend_matmul_d2d(backend, ctx->cuda_norm, ctx->cuda_ffn,
                                     1u, cfg->hidden_dim, cfg->ffn_dim,
                                     layer->linear1_weight, layer->linear1_bias,
                                     local, sizeof(local)) != 0 ||
            mynah_backend_gelu_dev(backend, ctx->cuda_ffn, cfg->ffn_dim, local,
                                   sizeof(local)) != 0 ||
            mynah_backend_matmul_d2d(backend, ctx->cuda_ffn, ctx->cuda_proj,
                                     1u, cfg->ffn_dim, cfg->hidden_dim,
                                     layer->linear2_weight, layer->linear2_bias,
                                     local, sizeof(local)) != 0 ||
            mynah_backend_residual_add_dev(backend, ctx->cuda_x, ctx->cuda_proj,
                                           cfg->hidden_dim, local,
                                           sizeof(local)) != 0) goto fail;
    }
    if (mynah_backend_layer_norm_dev(
            backend, ctx->cuda_x, ctx->cuda_norm, ctx->state->backbone.out_norm_weight,
            ctx->state->backbone.out_norm_bias, 1u, cfg->hidden_dim, local,
            sizeof(local)) != 0 ||
        mynah_backend_d2h(backend, ctx->cuda_norm, ctx->hidden,
                          cfg->hidden_dim, local, sizeof(local)) != 0) goto fail;
    /* Keep only the newly appended K/V slot shadowed in the host cache.  The
     * whole prefix stays resident, but this bounded copy is what makes a
     * later recoverable CUDA failure safe to retry on CPU without replaying
     * the request from the beginning. */
    const size_t host_half =
        mynah_transformer_ar_state_kv_half_floats(ctx->backbone);
    const size_t host_slot = position * attn_dim;
    for (size_t l = 0; l < cfg->layers; ++l) {
        float *host_kv = mynah_transformer_ar_state_kv(ctx->backbone, l);
        const float *device_k =
            ctx->cuda_backbone_kv + l * (2u * layer_half) + host_slot;
        const float *device_v = device_k + layer_half;
        if (host_kv == NULL ||
            mynah_backend_d2h(backend, device_k, host_kv + host_slot,
                              attn_dim, local, sizeof(local)) != 0 ||
            mynah_backend_d2h(backend, device_v, host_kv + host_half + host_slot,
                              attn_dim, local, sizeof(local)) != 0) goto fail;
    }
    if (mynah_backend_sync(backend, local, sizeof(local)) != 0 ||
        !pocket_all_finite(ctx->hidden, cfg->hidden_dim)) goto fail;
    if (mynah_transformer_ar_state_set_offset(ctx->backbone, position + 1u,
                                              local, sizeof(local)) != 0) goto fail;
    if (error != NULL && capacity > 0u) error[0] = '\0';
    return 0;

fail:
    /* A failed CUDA launch can leave the stream in an error state.  Do not
     * pay the same failure on every following frame: the host cache is still
     * authoritative, so this request can continue on CPU until its context is
     * destroyed. */
    (void)mynah_backend_sync(backend, drain_error, sizeof(drain_error));
    ctx->cuda_backbone_enabled = 0;
    ctx->cuda_backbone_valid = 0;
    pocket_error(error, capacity, "%s",
                 local[0] != '\0' ? local : "CUDA resident backbone failed");
    return -1;
}

/* Cross-request CUDA batch: projections and residual/FFN work are stacked,
 * while every row keeps its own KV pointer and absolute position.  A return of
 * 1 means "not eligible" and leaves all host state untouched; -1 means a CUDA
 * failure after submission and invalidates the resident caches so the caller
 * can retry the same rows through the CPU implementation. */
static int pocket_cuda_backbone_step_batch(mynah_engine_ctx *const *ctxs,
                                           size_t count,
                                           mynah_engine_scratch *scratch,
                                           char *error, size_t capacity) {
    if (ctxs == NULL || scratch == NULL || count < 2u ||
        !scratch->cuda_batch_enabled || scratch->backend == NULL ||
        count > scratch->cuda_batch_capacity || count > POCKET_MAX_BATCH) {
        return 1;
    }
    mynah_engine_ctx *first = ctxs[0];
    if (first == NULL || first->state == NULL ||
        strcmp(mynah_backend_name(scratch->backend), "cuda") != 0) return 1;
    const mynah_engine_state *state = first->state;
    const pocket_config *cfg = &state->cfg;
    const mynah_transformer_ar_config *first_config =
        mynah_transformer_ar_state_config(first->backbone);
    if (first_config == NULL) return 1;

    if (scratch->cuda_kcache == NULL || scratch->cuda_vcache == NULL ||
        scratch->cuda_positions == NULL || scratch->cuda_cache_strides == NULL ||
        scratch->cuda_host_kv == NULL || scratch->cuda_kv_shadow == NULL) {
        return 1;
    }
    const size_t attn_dim = cfg->heads * cfg->head_dim;
    const size_t shadow_row = 2u * attn_dim;
    int all_kv_valid = 1;
    for (size_t i = 0; i < count; ++i) {
        mynah_engine_ctx *ctx = ctxs[i];
        if (ctx == NULL || ctx->state != state || !ctx->cuda_backbone_enabled ||
            ctx->cuda_backbone_kv == NULL || ctx->cuda_x == NULL ||
            ctx->cuda_backbone_capacity == 0u) return 1;
        const mynah_transformer_ar_config *config =
            mynah_transformer_ar_state_config(ctx->backbone);
        if (config == NULL || config->d_model != first_config->d_model ||
            config->num_heads != first_config->num_heads ||
            config->head_dim != first_config->head_dim ||
            config->num_layers != first_config->num_layers ||
            config->ffn_dim != first_config->ffn_dim ||
            config->max_period != first_config->max_period ||
            config->layernorm_eps != first_config->layernorm_eps) return 1;
        if (config->max_seq_len == 0u ||
            ctx->cuda_backbone_capacity != config->max_seq_len) return 1;
        scratch->cuda_positions[i] =
            mynah_transformer_ar_state_offset(ctx->backbone);
        if (scratch->cuda_positions[i] >= config->max_seq_len) return -1;
        /* The resident KV allocation is [K/V][position][head].  max_seq_len
         * is the extent of one cache plane, not the stride between positions.
         * The old batch path multiplied by max_seq_len here, so the first
         * multi-request step wrote past every per-request cache while the
         * single-request path (which already uses attn_dim) stayed correct. */
        scratch->cuda_cache_strides[i] = attn_dim;
        if (!ctx->cuda_backbone_valid) all_kv_valid = 0;
        memcpy(scratch->cuda_host_input + i * cfg->hidden_dim,
               ctx->step_input, cfg->hidden_dim * sizeof(float));
    }
    /* Graph replay reads one persistent host pointer table per transformer
     * layer.  The table cannot be a stack array and it cannot be shared by all
     * layers: each layer owns a different KV allocation, while the captured
     * memcpy nodes are replayed later with new requests. */
    for (size_t l = 0; l < cfg->layers; ++l) {
        for (size_t i = 0; i < count; ++i) {
            const mynah_transformer_ar_config *config =
                mynah_transformer_ar_state_config(ctxs[i]->backbone);
            size_t layer_half = 0u;
            size_t layer_span = 0u;
            size_t layer_offset = 0u;
            if (config == NULL ||
                pocket_mul(config->max_seq_len, attn_dim, &layer_half) != 0 ||
                pocket_mul(layer_half, 2u, &layer_span) != 0 ||
                pocket_mul(l, layer_span, &layer_offset) != 0) return -1;
            const size_t metadata_offset = l * scratch->cuda_batch_capacity + i;
            scratch->cuda_kcache[metadata_offset] =
                ctxs[i]->cuda_backbone_kv + layer_offset;
            scratch->cuda_vcache[metadata_offset] =
                scratch->cuda_kcache[metadata_offset] + layer_half;
        }
    }

    char local[256];
    char drain_error[256];
    int graph_replay = 0;
    int graph_capture = 0;
    local[0] = '\0';
    if (mynah_backend_batch_begin(scratch->backend, local, sizeof(local)) != 0)
        goto fail;
    if (scratch->cuda_graph_enabled && scratch->cuda_graph_ready && all_kv_valid) {
        const int graph_rc = mynah_backend_graph_begin(
            scratch->backend, count, scratch, &graph_replay, local,
            sizeof(local));
        if (graph_rc == 0) graph_capture = !graph_replay;
    }

    if (graph_replay) {
        if (mynah_backend_graph_launch(scratch->backend, count, scratch, local,
                                       sizeof(local)) != 0) goto fail;
    } else {
        for (size_t i = 0; i < count; ++i) {
            mynah_engine_ctx *ctx = ctxs[i];
            if (!ctx->cuda_backbone_valid &&
                pocket_cuda_backbone_upload(ctx, local, sizeof(local)) != 0)
                goto fail;
        }
        if (mynah_backend_h2d(scratch->backend, scratch->cuda_host_input,
                              scratch->cuda_x, count * cfg->hidden_dim, local,
                              sizeof(local)) != 0) goto fail;

        for (size_t l = 0; l < cfg->layers; ++l) {
            const mynah_transformer_ar_layer *layer = &state->backbone_layers[l];
            if (mynah_backend_layer_norm_dev(
                    scratch->backend, scratch->cuda_x, scratch->cuda_norm,
                    layer->norm1_weight, layer->norm1_bias, count,
                    cfg->hidden_dim, local, sizeof(local)) != 0 ||
                mynah_backend_matmul_d2d(
                    scratch->backend, scratch->cuda_norm, scratch->cuda_qkv,
                    count, cfg->hidden_dim, 3u * attn_dim,
                    layer->in_proj_weight, layer->in_proj_bias, local,
                    sizeof(local)) != 0) goto fail;
            if (mynah_backend_rope_batch_dev(
                    scratch->backend, scratch->cuda_qkv,
                    scratch->cuda_positions, count, cfg->heads, cfg->head_dim,
                    first_config->max_period, local, sizeof(local)) != 0)
                goto fail;
            if (mynah_backend_self_attention_batch_dev(
                    scratch->backend, scratch->cuda_qkv,
                    scratch->cuda_kcache + l * scratch->cuda_batch_capacity,
                    scratch->cuda_vcache + l * scratch->cuda_batch_capacity,
                    scratch->cuda_positions,
                    scratch->cuda_cache_strides, count, cfg->heads,
                    cfg->head_dim, 1.0f / sqrtf((float)cfg->head_dim),
                    scratch->cuda_attn, local, sizeof(local)) != 0 ||
                mynah_backend_matmul_d2d(
                    scratch->backend, scratch->cuda_attn, scratch->cuda_proj,
                    count, attn_dim, cfg->hidden_dim, layer->out_proj_weight,
                    layer->out_proj_bias, local, sizeof(local)) != 0 ||
                mynah_backend_residual_add_dev(
                    scratch->backend, scratch->cuda_x, scratch->cuda_proj,
                    count * cfg->hidden_dim, local, sizeof(local)) != 0 ||
                mynah_backend_layer_norm_dev(
                    scratch->backend, scratch->cuda_x, scratch->cuda_norm,
                    layer->norm2_weight, layer->norm2_bias, count,
                    cfg->hidden_dim, local, sizeof(local)) != 0 ||
                mynah_backend_matmul_d2d(
                    scratch->backend, scratch->cuda_norm, scratch->cuda_ffn,
                    count, cfg->hidden_dim, cfg->ffn_dim,
                    layer->linear1_weight, layer->linear1_bias, local,
                    sizeof(local)) != 0 ||
                mynah_backend_gelu_dev(scratch->backend, scratch->cuda_ffn,
                                       count * cfg->ffn_dim, local,
                                       sizeof(local)) != 0 ||
                mynah_backend_matmul_d2d(
                    scratch->backend, scratch->cuda_ffn, scratch->cuda_proj,
                    count, cfg->ffn_dim, cfg->hidden_dim,
                    layer->linear2_weight, layer->linear2_bias, local,
                    sizeof(local)) != 0 ||
                mynah_backend_residual_add_dev(
                    scratch->backend, scratch->cuda_x, scratch->cuda_proj,
                    count * cfg->hidden_dim, local, sizeof(local)) != 0 ||
                mynah_backend_gather_kv_batch(
                    scratch->backend,
                    scratch->cuda_kcache + l * scratch->cuda_batch_capacity,
                    scratch->cuda_vcache + l * scratch->cuda_batch_capacity,
                    scratch->cuda_positions,
                    scratch->cuda_cache_strides, count, cfg->heads,
                    cfg->head_dim, scratch->cuda_kv_shadow, local,
                    sizeof(local)) != 0 ||
                mynah_backend_d2h(
                    scratch->backend, scratch->cuda_kv_shadow,
                    scratch->cuda_host_kv + l * count * shadow_row,
                    count * shadow_row, local, sizeof(local)) != 0) goto fail;
        }
        if (mynah_backend_layer_norm_dev(
                scratch->backend, scratch->cuda_x, scratch->cuda_norm,
                state->backbone.out_norm_weight,
                state->backbone.out_norm_bias, count, cfg->hidden_dim, local,
                sizeof(local)) != 0 ||
            mynah_backend_d2h(scratch->backend, scratch->cuda_norm,
                              scratch->cuda_host_output,
                              count * cfg->hidden_dim, local,
                              sizeof(local)) != 0) goto fail;

        if (graph_capture) {
            if (mynah_backend_graph_end(scratch->backend, count, scratch,
                                         local, sizeof(local)) != 0) {
                scratch->cuda_graph_ready = 0;
                mynah_backend_graph_abort(scratch->backend, count, scratch);
                return 1;
            }
            graph_capture = 0;
            if (mynah_backend_graph_launch(scratch->backend, count, scratch,
                                            local, sizeof(local)) != 0) goto fail;
        }
    }
    if (mynah_backend_sync(scratch->backend, local, sizeof(local)) != 0) goto fail;
    for (size_t i = 0; i < count; ++i) {
        if (!pocket_all_finite(scratch->cuda_host_output + i * cfg->hidden_dim,
                               cfg->hidden_dim)) goto fail;
    }
    for (size_t i = 0; i < count; ++i) {
        memcpy(ctxs[i]->hidden, scratch->cuda_host_output + i * cfg->hidden_dim,
               cfg->hidden_dim * sizeof(float));
    }
    /* The graph and the ordinary path both gather only the newly appended
     * slot.  Commit it to the host shadow after the one stream sync; the large
     * prefix remains device-resident. */
    for (size_t l = 0; l < cfg->layers; ++l) {
        for (size_t i = 0; i < count; ++i) {
            mynah_engine_ctx *ctx = ctxs[i];
            float *host_kv = mynah_transformer_ar_state_kv(ctx->backbone, l);
            const size_t host_half =
                mynah_transformer_ar_state_kv_half_floats(ctx->backbone);
            const size_t host_slot = scratch->cuda_positions[i] * attn_dim;
            const float *shadow = scratch->cuda_host_kv +
                l * count * shadow_row + i * shadow_row;
            if (host_kv == NULL) goto fail;
            memcpy(host_kv + host_slot, shadow, attn_dim * sizeof(float));
            memcpy(host_kv + host_half + host_slot, shadow + attn_dim,
                   attn_dim * sizeof(float));
        }
    }
    for (size_t i = 0; i < count; ++i) {
        if (mynah_transformer_ar_state_set_offset(
                ctxs[i]->backbone, scratch->cuda_positions[i] + 1u, local,
                sizeof(local)) != 0) {
            for (size_t j = 0; j < i; ++j) {
                (void)mynah_transformer_ar_state_set_offset(
                    ctxs[j]->backbone, scratch->cuda_positions[j], NULL, 0u);
            }
            goto fail;
        }
    }
    if (scratch->cuda_graph_enabled) scratch->cuda_graph_ready = 1;
    if (error != NULL && capacity > 0u) error[0] = '\0';
    return 0;

fail:
    if (graph_capture) {
        mynah_backend_graph_abort(scratch->backend, count, scratch);
    }
    if (scratch->cuda_graph_enabled) {
        scratch->cuda_graph_ready = 0;
        mynah_backend_graph_forget(scratch->backend, scratch);
    }
    (void)mynah_backend_sync(scratch->backend, drain_error, sizeof(drain_error));
    for (size_t i = 0; i < count; ++i) {
        ctxs[i]->cuda_backbone_enabled = 0;
        ctxs[i]->cuda_backbone_valid = 0;
    }
    pocket_error(error, capacity, "%s",
                 local[0] != '\0' ? local : "CUDA resident backbone batch failed");
    return -1;
}

#define POCKET_CUDA_FLOW_GRAPH_BASE ((size_t)0x100000u)

/* Resident Pocket flow head for the subset of rows that actually emitted a
 * latent. The first call warms all weight caches on the ordinary stream; the
 * next call captures/replays one graph per gathered width. A failure only
 * disables this optional stage and leaves the CPU flow implementation to
 * produce the row, so no request state is consumed by a failed launch. */
static int pocket_cuda_flow_step_batch(const mynah_engine_state *state,
                                       size_t count,
                                       mynah_engine_scratch *scratch,
                                       char *error, size_t capacity) {
    if (state == NULL || scratch == NULL || count == 0u ||
        !scratch->cuda_flow_enabled || scratch->backend == NULL ||
        strcmp(mynah_backend_name(scratch->backend), "cuda") != 0 ||
        count > scratch->cuda_flow_batch_capacity ||
        scratch->flow_cond == NULL || scratch->flow_noise == NULL ||
        scratch->flow_out == NULL || scratch->cuda_flow_host_cond == NULL ||
        scratch->cuda_flow_host_noise == NULL ||
        scratch->cuda_flow_host_time == NULL ||
        scratch->cuda_flow_host_output == NULL ||
        scratch->cuda_flow_cond == NULL || scratch->cuda_flow_noise == NULL ||
        scratch->cuda_flow_time_embed == NULL ||
        scratch->cuda_flow_output == NULL || scratch->cuda_flow_y == NULL ||
        scratch->cuda_flow_silu == NULL || scratch->cuda_flow_x == NULL ||
        scratch->cuda_flow_norm == NULL || scratch->cuda_flow_hidden == NULL ||
        scratch->cuda_flow_scratch == NULL || scratch->cuda_flow_mod == NULL ||
        scratch->cuda_flow_final_mod == NULL ||
        scratch->cuda_flow_time_hidden == NULL ||
        scratch->cuda_flow_time_output == NULL ||
        scratch->cuda_flow_time_sum == NULL ||
        scratch->cuda_flow_time_mlp_in == NULL ||
        scratch->cuda_flow_time_mlp_out == NULL ||
        scratch->cuda_flow_time_alpha == NULL || scratch->cuda_flow_blocks == NULL) {
        return 1;
    }
    const pocket_config *cfg = &state->cfg;
    size_t flow_freq_width = 0u;
    size_t rows_hidden = 0u;
    size_t rows_latent = 0u;
    size_t time_rows = 0u;
    if (pocket_mul(cfg->flow_freqs, 2u, &flow_freq_width) != 0 ||
        pocket_mul(count, cfg->hidden_dim, &rows_hidden) != 0 ||
        pocket_mul(count, cfg->latent_dim, &rows_latent) != 0 ||
        pocket_mul(cfg->flow_time_conds, flow_freq_width, &time_rows) != 0 ||
        flow_freq_width == 0u || cfg->flow_time_conds == 0u) return 1;

    for (size_t i = 0; i < count; ++i) {
        if (scratch->flow_cond[i] == NULL || scratch->flow_noise[i] == NULL ||
            scratch->flow_out[i] == NULL) return 1;
        memcpy(scratch->cuda_flow_host_cond + i * cfg->hidden_dim,
               scratch->flow_cond[i], cfg->hidden_dim * sizeof(float));
        memcpy(scratch->cuda_flow_host_noise + i * cfg->latent_dim,
               scratch->flow_noise[i], cfg->latent_dim * sizeof(float));
    }

    char local[256];
    char drain_error[256];
    local[0] = '\0';
    if (mynah_backend_batch_begin(scratch->backend, local, sizeof(local)) != 0)
        return 1;

    int replay = 0;
    int capturing = 0;
    const size_t graph_key = POCKET_CUDA_FLOW_GRAPH_BASE + count;
    if (scratch->cuda_flow_graph_enabled && scratch->cuda_flow_graph_ready) {
        const int graph_rc = mynah_backend_graph_begin(
            scratch->backend, graph_key, scratch, &replay, local, sizeof(local));
        if (graph_rc == 0) capturing = !replay;
    }

    if (!replay) {
        if (mynah_backend_h2d(scratch->backend, scratch->cuda_flow_host_cond,
                              scratch->cuda_flow_cond, rows_hidden, local,
                              sizeof(local)) != 0 ||
            mynah_backend_h2d(scratch->backend, scratch->cuda_flow_host_noise,
                              scratch->cuda_flow_noise, rows_latent, local,
                              sizeof(local)) != 0 ||
            mynah_backend_h2d(scratch->backend, scratch->cuda_flow_host_time,
                              scratch->cuda_flow_time_embed, time_rows,
                              local, sizeof(local)) != 0) {
            if (capturing) mynah_backend_graph_abort(
                scratch->backend, graph_key, scratch);
            return 1;
        }
        mynah_backend_flow_batch flow;
        memset(&flow, 0, sizeof(flow));
        flow.batch = count;
        flow.latent_dim = cfg->latent_dim;
        flow.cond_dim = cfg->hidden_dim;
        flow.hidden_dim = cfg->flow_dim;
        flow.depth = cfg->flow_depth;
        flow.num_time_conds = cfg->flow_time_conds;
        flow.freq_embed_dim = flow_freq_width;
        flow.layernorm_eps = cfg->flow_layernorm_eps;
        flow.rmsnorm_eps = 1.0e-5f;
        flow.dev_cond = scratch->cuda_flow_cond;
        flow.dev_noise = scratch->cuda_flow_noise;
        flow.dev_time_embed = scratch->cuda_flow_time_embed;
        flow.dev_y = scratch->cuda_flow_y;
        flow.dev_silu = scratch->cuda_flow_silu;
        flow.dev_x = scratch->cuda_flow_x;
        flow.dev_norm = scratch->cuda_flow_norm;
        flow.dev_hidden = scratch->cuda_flow_hidden;
        flow.dev_scratch = scratch->cuda_flow_scratch;
        flow.dev_mod = scratch->cuda_flow_mod;
        flow.dev_final_mod = scratch->cuda_flow_final_mod;
        flow.dev_time_hidden = scratch->cuda_flow_time_hidden;
        flow.dev_time_output = scratch->cuda_flow_time_output;
        flow.dev_time_sum = scratch->cuda_flow_time_sum;
        flow.dev_out = scratch->cuda_flow_output;
        flow.time_mlp_in = scratch->cuda_flow_time_mlp_in;
        flow.time_mlp_out = scratch->cuda_flow_time_mlp_out;
        flow.time_alpha = scratch->cuda_flow_time_alpha;
        flow.cond_embed = &scratch->cuda_flow_cond_embed;
        flow.input_proj = &scratch->cuda_flow_input_proj;
        flow.blocks = scratch->cuda_flow_blocks;
        flow.final_adaln = &scratch->cuda_flow_final_adaln;
        flow.final_linear = &scratch->cuda_flow_final_linear;
        if (mynah_backend_flow_batch_dev(scratch->backend, &flow, local,
                                          sizeof(local)) != 0) {
            if (capturing) mynah_backend_graph_abort(
                scratch->backend, graph_key, scratch);
            return 1;
        }
        if (capturing) {
            if (mynah_backend_graph_end(scratch->backend, graph_key, scratch,
                                         local, sizeof(local)) != 0) {
                scratch->cuda_flow_graph_ready = 0;
                return 1;
            }
            capturing = 0;
            if (mynah_backend_graph_launch(scratch->backend, graph_key, scratch,
                                            local, sizeof(local)) != 0) {
                mynah_backend_graph_forget(scratch->backend, scratch);
                scratch->cuda_flow_enabled = 0;
                return 1;
            }
        } else if (scratch->cuda_flow_graph_enabled) {
            /* This is the warm-up submission. */
            scratch->cuda_flow_graph_ready = 1;
        }
    } else if (mynah_backend_graph_launch(scratch->backend, graph_key, scratch,
                                           local, sizeof(local)) != 0) {
        mynah_backend_graph_forget(scratch->backend, scratch);
        scratch->cuda_flow_enabled = 0;
        scratch->cuda_flow_graph_ready = 0;
        return 1;
    }

    if (mynah_backend_d2h(scratch->backend, scratch->cuda_flow_output,
                          scratch->cuda_flow_host_output, rows_latent, local,
                          sizeof(local)) != 0 ||
        mynah_backend_sync(scratch->backend, local, sizeof(local)) != 0) {
        if (capturing) mynah_backend_graph_abort(
            scratch->backend, graph_key, scratch);
        snprintf(drain_error, sizeof(drain_error), "%s", local);
        (void)mynah_backend_sync(scratch->backend, drain_error,
                                 sizeof(drain_error));
        mynah_backend_graph_forget(scratch->backend, scratch);
        scratch->cuda_flow_enabled = 0;
        scratch->cuda_flow_graph_ready = 0;
        return 1;
    }
    for (size_t i = 0; i < count; ++i) {
        float *out = scratch->flow_out[i];
        const float *src = scratch->cuda_flow_host_output + i * cfg->latent_dim;
        if (!pocket_all_finite(src, cfg->latent_dim)) {
            mynah_backend_graph_forget(scratch->backend, scratch);
            scratch->cuda_flow_enabled = 0;
            scratch->cuda_flow_graph_ready = 0;
            return 1;
        }
        memcpy(out, src, cfg->latent_dim * sizeof(float));
    }
    if (error != NULL && capacity > 0u) error[0] = '\0';
    return 0;
}

static void pocket_ctx_free(mynah_engine_ctx *ctx) {
    if (ctx == NULL) return;
    if (ctx->t_created_ns != 0u) {
        mynah_region_add_ns(MYNAH_RGN_REQUEST,
                            mynah_costmap_now_ns() - ctx->t_created_ns);
        mynah_costmap_request_done();
    }
    pocket_dump_flush(ctx);
    pocket_dump_free(ctx->dump);
    pocket_cuda_backbone_release(ctx);
    pocket_cuda_decoder_release(ctx);
    pocket_call_release(&ctx->backbone_call.call);
    pocket_call_release(&ctx->codec_call.call);
    pocket_call_release(&ctx->flow_call.call);
    mynah_transformer_ar_state_free(ctx->backbone);
    mynah_transformer_ar_state_free(ctx->codec_transformer);
    mynah_flow_head_destroy(ctx->flow);
    mynah_seanet_state_destroy(ctx->codec);
    ingot_st_close(ctx->voice_file);
    free(ctx->voice_kv);
    free(ctx->text_ids);
    free(ctx->text_embed);
    free(ctx->step_input);
    free(ctx->hidden);
    free(ctx->noise);
    free(ctx->flow_out);
    free(ctx->latents);
    free(ctx->denorm);
    free(ctx->codec_in);
    free(ctx->codec_up);
    free(ctx->codec_seq);
    free(ctx->codec_out);
    free(ctx->codec_back);
    free(ctx->pcm);
    free(ctx);
}

/* Opens the voice and reads its length, without loading anything yet: the KV
 * itself is copied in `prepare`, so a reset costs no allocation. */
static int pocket_voice_open(mynah_engine_ctx *ctx, char *error, size_t capacity) {
    const mynah_engine_state *state = ctx->state;
    const pocket_config *cfg = &state->cfg;
    const pocket_voice *voice = &state->voices[ctx->speaker];
    char path[POCKET_PATH_MAX];
    if (pocket_join(path, sizeof(path), state->model_dir, voice->file, error,
                    capacity) != 0) {
        return -1;
    }
    if (ingot_st_open(&ctx->voice_file, path, error, capacity) != 0) return -1;

    size_t positions = 0;
    if (pocket_voice_validate(ctx->voice_file, cfg, voice->name, &positions, error,
                              capacity) != 0) {
        return -1;
    }
    /* Re-checked here, not only at pack load: the file could have been replaced
     * underneath a long-running process, and this is the last moment before its
     * numbers become the model's attention prefix (E3-13). */
    if (voice->frames != 0u && positions != voice->frames) {
        pocket_error(error, capacity,
                     "voice %s holds %zu positions, the pack declares %zu; this "
                     "file does not belong to this pack",
                     voice->name, positions, voice->frames);
        return -1;
    }
    ctx->voice_positions = positions;
    return 0;
}

/* The backbone's configuration at a given KV capacity.  Factored out of
 * `ctx_new` because `reserve_text` has to rebuild the state at a LARGER
 * capacity and the two must not be able to drift: everything except
 * `max_seq_len` is a property of the pack. */
static void pocket_backbone_config(const mynah_engine_ctx *ctx, size_t capacity,
                                   mynah_transformer_ar_config *out) {
    const pocket_config *cfg = &ctx->state->cfg;
    mynah_transformer_ar_config_defaults(out);
    out->d_model = cfg->hidden_dim;
    out->num_heads = cfg->heads;
    out->head_dim = cfg->head_dim;
    out->num_layers = cfg->layers;
    out->ffn_dim = cfg->ffn_dim;
    out->max_seq_len = capacity;
    out->context = 0u; /* the LM attends to the whole prefix */
    out->layernorm_eps = cfg->layernorm_eps;
}

static int pocket_ctx_new(const mynah_tts_model *model, mynah_engine_state *state,
                          const mynah_tts_request *request, size_t max_steps,
                          uint64_t seed, mynah_engine_ctx **out_ctx, char *error,
                          size_t capacity) {
    (void)model;
    if (out_ctx != NULL) *out_ctx = NULL;
    if (state == NULL || request == NULL || out_ctx == NULL) {
        pocket_error(error, capacity, "pocket: null argument creating a context");
        return -1;
    }
    const pocket_config *cfg = &state->cfg;

    if (request->speaker >= state->voice_count) {
        pocket_error(error, capacity, "speaker %u is out of range (%zu voices)",
                     request->speaker, state->voice_count);
        return -1;
    }
    if (request->text_ids == NULL || request->text_length == 0) {
        pocket_error(error, capacity, "pocket: the request has no text");
        return -1;
    }
    if (max_steps == 0) max_steps = cfg->default_max_steps;
    if (max_steps == 0) {
        pocket_error(error, capacity,
                     "pocket: no step budget; pass max_steps or declare "
                     "max_decoder_steps in model.json");
        return -1;
    }

    mynah_engine_ctx *ctx = (mynah_engine_ctx *)calloc(1, sizeof(*ctx));
    if (ctx == NULL) {
        pocket_error(error, capacity, "out of memory creating a pocket context");
        return -1;
    }
    ctx->state = state;
    ctx->cuda_backbone_enabled = pocket_cuda_resident_requested(state);
    ctx->cuda_decoder_enabled = pocket_cuda_resident_requested(state);
    ctx->speaker = request->speaker;
    ctx->max_steps = max_steps;
    ctx->text_length = request->text_length;
    /* No reserve yet: the ceiling is the text that was admitted, which makes
     * every long-form branch below cost exactly one comparison. */
    ctx->text_capacity = request->text_length;
    ctx->temperature = (request->temperature >= 0.0f) ? request->temperature
                                                      : cfg->temperature;
    ctx->noise_std = sqrtf(ctx->temperature);
    ctx->seed = (seed == 0) ? UINT64_C(0x9e3779b97f4a7c15) : seed;
    ctx->frames_after_eos = (cfg->frames_after_eos < max_steps)
                                ? cfg->frames_after_eos
                                : max_steps;
    ctx->dump = pocket_dump_open(cfg, max_steps);

    ctx->text_ids = (int *)calloc(ctx->text_capacity, sizeof(*ctx->text_ids));
    if (ctx->text_ids == NULL) {
        pocket_ctx_free(ctx);
        pocket_error(error, capacity, "out of memory copying the request text");
        return -1;
    }
    for (size_t i = 0; i < ctx->text_length; ++i) {
        const int id = request->text_ids[i];
        if (id < 0 || (size_t)id >= cfg->vocab_size) {
            pocket_ctx_free(ctx);
            pocket_error(error, capacity,
                         "text id %d at position %zu is outside [0, %zu); row %zu is "
                         "the padding row and is never emitted by the tokenizer",
                         id, i, cfg->vocab_size, cfg->padding_id);
            return -1;
        }
        ctx->text_ids[i] = id;
    }

    /* ---- the text-chunk seam, E2-5 -------------------------------------
     *
     * `model.json` declares `max_tokens_per_chunk: 50` and THIS ENGINE DOES NOT
     * APPLY IT.  That is a decision, not an omission, and the reasons are below
     * together with what it costs -- because what it costs is not small.
     *
     * WHERE UPSTREAM PUTS IT.  `generate_audio_stream` splits the TEXT before
     * the model sees it (`split_into_best_sentences`): sentence boundaries
     * first, then commas/semicolons/colons for an oversized sentence, then
     * greedy packing up to `max_tokens`.  Each chunk is a separate generation
     * from a deep copy of the voice state, and the audio is concatenated.  So
     * it is a policy over text, applied above the model -- and the seam here
     * hands this engine TOKEN IDS, by which point the sentence structure the
     * splitter needs is gone.  Putting it here would mean an engine that
     * re-derives text structure from ids, which is the wrong place twice over.
     *
     * WHAT IT COSTS, MEASURED (models/pocket-en, alba, seed 1234, this machine,
     * duration only -- the machine compiles for correctness, so nothing here is
     * a timing claim):
     *
     *   tokens   15    25    40    49    64    78    91   116   150
     *   frames   38    64    96   126   147   170   178   233   295
     *   f/token 2.53  2.56  2.40  2.57  2.30  2.18  1.96  2.01  1.97
     *
     * Two things in that table.  First, NOTHING HAPPENS AT 50: the degradation
     * is smooth, ~2.55 frames per token below the limit falling to ~1.97 well
     * above it, i.e. about a quarter of the speech quietly missing.  That is
     * upstream's documented skip (`.work/pocket-tts-model-facts.md` §8) and we
     * REPRODUCE IT EXACTLY for a single comma-free sentence -- upstream's own
     * splitter does not split one either: its sub-split finds no boundary, it
     * keeps the oversized segment and only logs a warning.
     *
     * Second, and this is the divergence that matters: for text with SENTENCE
     * boundaries upstream would have split, and we do not.  Measured on the
     * same words cut into eight-word sentences, at 152 tokens and beyond the
     * model NEVER EMITS EOS and runs to the step budget -- 900 steps, 72
     * seconds of audio for the input.  That is the same unbounded-generation
     * shape E3-13 is about, reached by long input instead of a wrong voice, and
     * it is why this is a warning rather than a footnote.
     *
     * Re-checked in a later session on DIFFERENT eight-word sentences, which
     * matters because it says the threshold is a property of the length and not
     * of one paragraph: 116 tokens -> 241 frames (2.08 f/token, EOS reached),
     * 173 tokens -> 900 frames, i.e. the whole `--max-steps 900` budget and
     * 72.000 s of audio, with no EOS ever emitted.
     *
     * So: the limit is read, published (`_max_tokens_per_chunk`) and reported,
     * and the split belongs to whoever still holds the text.  The step budget
     * is what bounds the damage in the meantime. */
    if (state->cfg.max_tokens_per_chunk != 0u &&
        ctx->text_length > state->cfg.max_tokens_per_chunk && !state->chunk_warned) {
        state->chunk_warned = 1;
        fprintf(stderr,
                "mynah-tts: %zu text tokens exceeds max_tokens_per_chunk %zu; this "
                "engine does not split text, and past roughly three times the limit "
                "the model may stop emitting EOS and run to the step budget (E2-5)\n",
                ctx->text_length, state->cfg.max_tokens_per_chunk);
    }

    if (pocket_voice_open(ctx, error, capacity) != 0) {
        pocket_ctx_free(ctx);
        return -1;
    }

    /* Sizes. Every buffer the AR loop and the codec touch is allocated here so
     * that neither allocates again (AGENTS.md rule 4). */
    const size_t attn_dim = cfg->heads * cfg->head_dim;
    size_t backbone_capacity = 0;
    size_t voice_floats = 0;
    size_t text_floats = 0;
    size_t latent_floats = 0;
    size_t codec_positions = 0;
    size_t up_floats = 0;
    size_t pcm_floats = 0;
    size_t codec_frames = 0;
    if (pocket_add(max_steps, 1u, &codec_frames) != 0 ||
        pocket_add(ctx->voice_positions, ctx->text_capacity, &backbone_capacity) != 0 ||
        pocket_add(backbone_capacity, codec_frames, &backbone_capacity) != 0 ||
        pocket_mul(ctx->voice_positions, attn_dim, &voice_floats) != 0 ||
        pocket_mul(voice_floats, 2u, &voice_floats) != 0 ||
        pocket_mul(ctx->text_capacity, cfg->hidden_dim, &text_floats) != 0 ||
        pocket_mul(max_steps, cfg->latent_dim, &latent_floats) != 0 ||
        pocket_mul(codec_frames, cfg->upsample_stride, &codec_positions) != 0 ||
        pocket_mul(cfg->codec_dim, cfg->upsample_stride, &up_floats) != 0 ||
        pocket_mul(cfg->samples_per_frame, cfg->audio_channels, &pcm_floats) != 0) {
        pocket_ctx_free(ctx);
        pocket_error(error, capacity, "pocket: request size overflow");
        return -1;
    }

    ctx->voice_kv = mynah_alloc_floats(voice_floats, error, capacity);
    ctx->text_embed = mynah_alloc_floats(text_floats, error, capacity);
    ctx->step_input = mynah_alloc_floats(cfg->hidden_dim, error, capacity);
    ctx->hidden = mynah_alloc_floats(cfg->hidden_dim, error, capacity);
    ctx->noise = mynah_alloc_floats(cfg->latent_dim, error, capacity);
    ctx->flow_out = mynah_alloc_floats(cfg->latent_dim, error, capacity);
    ctx->latents = mynah_alloc_floats(latent_floats, error, capacity);
    ctx->denorm = mynah_alloc_floats(cfg->latent_dim, error, capacity);
    ctx->codec_in = mynah_alloc_floats(cfg->codec_dim, error, capacity);
    ctx->codec_up = mynah_alloc_floats(up_floats, error, capacity);
    ctx->codec_seq = mynah_alloc_floats(up_floats, error, capacity);
    ctx->codec_out = mynah_alloc_floats(up_floats, error, capacity);
    ctx->codec_back = mynah_alloc_floats(up_floats, error, capacity);
    ctx->pcm = mynah_alloc_floats(pcm_floats, error, capacity);
    if (ctx->voice_kv == NULL || ctx->text_embed == NULL || ctx->step_input == NULL ||
        ctx->hidden == NULL || ctx->noise == NULL || ctx->flow_out == NULL ||
        ctx->latents == NULL || ctx->denorm == NULL || ctx->codec_in == NULL ||
        ctx->codec_up == NULL || ctx->codec_seq == NULL || ctx->codec_out == NULL ||
        ctx->codec_back == NULL || ctx->pcm == NULL) {
        pocket_ctx_free(ctx);
        return -1;
    }

    mynah_transformer_ar_config backbone;
    pocket_backbone_config(ctx, backbone_capacity, &backbone);
    ctx->backbone = mynah_transformer_ar_state_new(&backbone, error, capacity);

    mynah_transformer_ar_config codec;
    mynah_transformer_ar_config_defaults(&codec);
    codec.d_model = cfg->codec_tf_dim;
    codec.num_heads = cfg->codec_tf_heads;
    codec.head_dim = cfg->codec_tf_dim / cfg->codec_tf_heads;
    codec.num_layers = cfg->codec_tf_layers;
    codec.ffn_dim = cfg->codec_tf_ffn;
    codec.max_seq_len = codec_positions;
    codec.context = cfg->codec_tf_context;
    codec.layernorm_eps = cfg->layernorm_eps;
    ctx->codec_transformer = mynah_transformer_ar_state_new(&codec, error, capacity);

    mynah_flow_head_config flow;
    mynah_flow_head_config_defaults(&flow);
    flow.latent_dim = cfg->latent_dim;
    flow.cond_dim = cfg->hidden_dim;
    flow.hidden_dim = cfg->flow_dim;
    flow.depth = cfg->flow_depth;
    flow.num_time_conds = cfg->flow_time_conds;
    flow.freq_embed_dim = 2u * cfg->flow_freqs;
    flow.layernorm_eps = cfg->flow_layernorm_eps;
    ctx->flow = mynah_flow_head_create(&flow, error, capacity);

    mynah_seanet_config seanet;
    memset(&seanet, 0, sizeof(seanet));
    seanet.channels = cfg->audio_channels;
    seanet.dimension = cfg->codec_dim;
    seanet.n_filters = cfg->n_filters;
    seanet.n_residual_layers = cfg->n_residual_layers;
    seanet.ratios = cfg->ratios;
    seanet.n_ratios = cfg->n_ratios;
    seanet.kernel_size = cfg->kernel_size;
    seanet.residual_kernel_size = cfg->residual_kernel_size;
    seanet.last_kernel_size = cfg->last_kernel_size;
    seanet.dilation_base = cfg->dilation_base;
    seanet.compress = cfg->compress;
    seanet.elu_alpha = cfg->elu_alpha;
    /* THE `codec_conv` GROUP NOW MEANS THE WHOLE CONV STACK, not just
     * mimi.quantizer.output_proj [512][32].  That projection is one matvec;
     * the decoder's causal convolutions are 42% of `codec.conv_stack` in the
     * configuration production ships, and before this they had no quantized
     * kernel at all -- so a spec that said `codec_conv:int8` was quantizing
     * 0.03% of what its name covers.  int4 and f16 are not offered here: the
     * conv path has one int8 kernel (src/convq8.h says why the transposed
     * convolutions are excluded), so anything else stays exact f32 rather
     * than silently becoming a substitute. */
    seanet.quantize_conv = (state->codec_conv_qtype == 1);
    seanet.quantize_convtr = (state->codec_convtr_qtype == 1);

    mynah_resample_config upsample;
    upsample.stride = cfg->upsample_stride;
    upsample.in_channels = cfg->codec_dim;
    upsample.out_channels = cfg->codec_dim;
    upsample.groups = cfg->codec_dim; /* depthwise: [512, 1, 32], not [512, 512, 32] */
    /* One latent frame per call: the streaming unit and the offline unit are
     * the same call, so there is no second code path to keep in step. */
    ctx->codec = mynah_seanet_state_create(&seanet, &upsample, 1u, error, capacity);

    if (ctx->backbone == NULL || ctx->codec_transformer == NULL || ctx->flow == NULL ||
        ctx->codec == NULL) {
        pocket_ctx_free(ctx);
        return -1;
    }

    /* One projection scratch per transformer, wide enough for a whole prefill
     * tile.  Every number here comes from the manifest. */
    const size_t tile = mynah_transformer_ar_prefill_tile();
    size_t backbone_k = cfg->hidden_dim;
    if (attn_dim > backbone_k) backbone_k = attn_dim;
    if (cfg->ffn_dim > backbone_k) backbone_k = cfg->ffn_dim;
    size_t codec_k = cfg->codec_tf_dim;
    if (cfg->codec_tf_ffn > codec_k) codec_k = cfg->codec_tf_ffn;
    size_t flow_k = cfg->flow_dim;
    if (cfg->hidden_dim > flow_k) flow_k = cfg->hidden_dim;
    if (pocket_tar_call_init(&ctx->backbone_call, &state->backbone_hook, tile,
                             backbone_k, error, capacity) != 0 ||
        pocket_tar_call_init(&ctx->codec_call, &state->codec_hook, tile, codec_k,
                             error, capacity) != 0 ||
        /* The flow head is evaluated one row at a time per request: a tile of
         * one is all this scratch ever needs. */
        pocket_flow_call_init(&ctx->flow_call, &state->flow_hook, 1u, flow_k, error,
                              capacity) != 0) {
        pocket_ctx_free(ctx);
        return -1;
    }
    pocket_bind_hooks(&ctx->backbone_w, &state->backbone, &ctx->backbone_call);
    pocket_bind_hooks(&ctx->codec_w, &state->codec_transformer, &ctx->codec_call);
    pocket_bind_flow_hooks(&ctx->flow_w, &state->flow, &ctx->flow_call);

    if (mynah_transformer_ar_check_weights(ctx->backbone, &ctx->backbone_w, error,
                                           capacity) != 0 ||
        mynah_transformer_ar_check_weights(ctx->codec_transformer,
                                           &ctx->codec_w, error,
                                           capacity) != 0 ||
        mynah_flow_head_check_weights(ctx->flow, &ctx->flow_w, error, capacity) != 0 ||
        mynah_seanet_check_decoder_weights(ctx->codec, &state->decoder, error,
                                           capacity) != 0) {
        pocket_ctx_free(ctx);
        return -1;
    }
    if (mynah_seanet_state_samples_per_latent(ctx->codec) != cfg->samples_per_frame) {
        pocket_error(error, capacity,
                     "codec produces %zu samples per latent, model.json says %zu",
                     mynah_seanet_state_samples_per_latent(ctx->codec),
                     cfg->samples_per_frame);
        pocket_ctx_free(ctx);
        return -1;
    }

    /* CUDA allocations happen before the request can be prepared.  If the
     * optional resident path cannot reserve them, the context remains a valid
     * CPU context and the backend contract's safe fallback is preserved. */
    (void)pocket_cuda_backbone_alloc(ctx);
    (void)pocket_cuda_decoder_alloc(ctx);

    /* Set last, so the failure paths above (which call `_ctx_free`) cannot
     * submit a span or count a request that never ran. */
    ctx->t_created_ns = mynah_costmap_level() ? mynah_costmap_now_ns() : 0u;
    *out_ctx = ctx;
    if (error != NULL && capacity > 0) error[0] = '\0';
    return 0;
}

/*
 * Push accepted-but-not-yet-prefilled text into the backbone KV.
 *
 * THE TILE ALIGNMENT IS THE WHOLE FUNCTION, and it is what makes "N pushes ==
 * one push" true rather than approximately true.  `mynah_transformer_ar_prefill`
 * runs its positions as tiles of at most `_prefill_tile()` rows, counted from
 * the start of THAT CALL, and it presents each tile's projections to the hook
 * as one call with `count == rows`.  For an unquantized group that call is a
 * GEMM whose blocking is a function of the row count, so a 13-row tile and a
 * 16-row tile need not reassociate a row's sum the same way.  Split the same
 * text at a different place and the tiles land differently -- and the audio
 * moves, by an amount no tolerance should be asked to absorb.
 *
 * So a push never prefills a partial tile.  Everything except the last tile is
 * emitted in whole `tile`-sized groups at `tile`-aligned offsets, exactly where
 * a one-shot prefill of the concatenated text would have put them, and the
 * remainder waits in `text_embed` until the text is sealed -- at which point it
 * becomes the one short final tile the one-shot also ends with.  The pending
 * remainder is under `tile` tokens and lives in a buffer that was already
 * allocated, so nothing here allocates and nothing here is unbounded.
 *
 * `final` is the seal: emit the remainder too.
 */
/* `limit` is the resumable-prefill budget in tokens; 0 means "no limit" and
 * reproduces the one-shot behaviour exactly. A bounded flush always stops on a
 * tile boundary, because a tile that starts anywhere else lands where a
 * one-shot prefill did not put it -- the guard below is the same one E5-5
 * wrote, keyed now on whether THIS flush finishes the text rather than on the
 * caller's `final`, which is what it always meant. */
static int pocket_text_flush_limited(mynah_engine_ctx *ctx, int final, size_t limit,
                                     char *error, size_t capacity) {
    const pocket_config *cfg = &ctx->state->cfg;
    const size_t tile = mynah_transformer_ar_prefill_tile();
    size_t want = (final || tile == 0u)
                      ? ctx->text_length
                      : (ctx->text_length / tile) * tile;
    if (limit != 0u && tile != 0u) {
        size_t slice = (limit / tile) * tile;
        if (slice == 0u) slice = tile;           /* never make no progress */
        const size_t stop = ctx->text_prefilled + slice;
        if (stop < want) want = (stop / tile) * tile;
    }
    if (want <= ctx->text_prefilled) return 0;
    const size_t rows = want - ctx->text_prefilled;
    /* The invariant the paragraph above is about, asserted rather than assumed.
     * It restates the line that computed `want`, which is the point: it is a
     * postcondition, and it is what makes a future edit to that line fail here
     * -- loudly, on every quantization profile -- instead of failing as a one
     * ULP difference that only the f32 tile path can see. */
    if (want != ctx->text_length && tile != 0u &&
        ((ctx->text_prefilled % tile) != 0u || (rows % tile) != 0u)) {
        pocket_error(error, capacity,
                     "pocket: a non-final text flush of %zu rows at offset %zu is "
                     "not aligned to the %zu-row prefill tile; the tiles would "
                     "land where a one-shot prefill did not put them",
                     rows, ctx->text_prefilled, tile);
        return -1;
    }
    mynah_region_begin(MYNAH_RGN_PREFILL);
    ctx->backbone_call.in_prefill = 1;
    const int failed =
        mynah_transformer_ar_prefill(
            ctx->backbone, &ctx->backbone_w,
            ctx->text_embed + ctx->text_prefilled * cfg->hidden_dim, rows,
            NULL) != 0;
    ctx->backbone_call.in_prefill = 0;
    mynah_region_end(MYNAH_RGN_PREFILL);
    if (failed) {
        pocket_error(error, capacity,
                     "pocket: the text prefill failed at token %zu of %zu",
                     ctx->text_prefilled, ctx->text_length);
        return -1;
    }
    ctx->text_prefilled = want;
    return 0;
}

static int pocket_text_flush(mynah_engine_ctx *ctx, int final, char *error,
                             size_t capacity) {
    return pocket_text_flush_limited(ctx, final, 0u, error, capacity);
}

/* Everything in the seed that does NOT scale with the text: the resets, the
 * voice KV, the conditioner lookup. It ends with the text prefill at zero, so
 * the caller decides whether to run that in one call or a tile at a time.
 *
 * The split is where it is because of the measurement: this part is ~15 ms and
 * flat in text length, the part after it is 190 ms for a long text and linear
 * in it. Only the linear part is worth interrupting. */
static int pocket_seed_prologue(mynah_engine_ctx *ctx, char *error, size_t capacity) {
    const mynah_engine_state *state = ctx->state;
    const pocket_config *cfg = &state->cfg;

    mynah_transformer_ar_state_reset(ctx->backbone);
    mynah_transformer_ar_state_reset(ctx->codec_transformer);
    mynah_seanet_state_reset(ctx->codec);
    if (ctx->cuda_decoder_enabled && ctx->cuda_decoder != NULL) {
        char decoder_error[256];
        if (mynah_backend_decoder_reset(state->backend, ctx->cuda_decoder,
                                        decoder_error, sizeof(decoder_error)) != 0) {
            /* No GPU decoder frame has been emitted in this generation yet;
             * dropping the optional device state is therefore safe. */
            pocket_cuda_decoder_release(ctx);
        }
    }
    mynah_flow_head_reset(ctx->flow);
    ctx->frames = 0;
    ctx->decoded_frames = 0;
    ctx->step = 0;
    ctx->eos_step = SIZE_MAX;
    ctx->eos = 0;
    ctx->broken = 0;
    ctx->budget_exhausted = 0;
    ctx->eos_logit = 0.0f;
    /* The host reset below is authoritative.  The resident cache is
     * re-uploaded lazily before the next CUDA step, after voice/text prefill
     * has rebuilt every valid host position. */
    ctx->cuda_backbone_valid = 0;
    ctx->rng = ctx->seed;
    ctx->have_spare = 0;
    ctx->spare = 0.0f;
    if (ctx->dump != NULL) {
        ctx->dump->steps = 0;
        ctx->dump->frames = 0;
        ctx->dump->decoded = 0;
    }

    char name[POCKET_NAME_MAX];
    for (size_t l = 0; l < cfg->layers; ++l) {
        snprintf(name, sizeof(name), "transformer.layers.%zu.self_attn/cache", l);
        const ingot_st_tensor *tensor = ingot_st_find(ctx->voice_file, name);
        if (tensor == NULL ||
            ingot_st_to_f32(ctx->voice_file, tensor, ctx->voice_kv) != 0) {
            pocket_error(error, capacity, "cannot read %s from voice %s", name,
                         state->voices[ctx->speaker].name);
            return -1;
        }
        if (mynah_transformer_ar_state_load_kv(ctx->backbone, l, ctx->voice_kv,
                                               ctx->voice_positions, error,
                                               capacity) != 0) {
            return -1;
        }
    }
    if (mynah_transformer_ar_state_set_offset(ctx->backbone, ctx->voice_positions,
                                              error, capacity) != 0) {
        return -1;
    }

    /* The LUT conditioner: a row lookup, no BOS, no EOS, no normalisation. */
    for (size_t i = 0; i < ctx->text_length; ++i) {
        const size_t id = (size_t)ctx->text_ids[i];
        memcpy(ctx->text_embed + i * cfg->hidden_dim,
               state->embed_table + id * cfg->hidden_dim,
               cfg->hidden_dim * sizeof(float));
    }
    /* A closed context is sealed here and prefills its whole text in one call,
     * which is byte for byte what this function did before E5-5 existed.  An
     * OPEN one stops at the last whole tile, because the tokens that will share
     * its final tile have not arrived yet -- see `pocket_text_flush`. */
    ctx->text_prefilled = 0;
    return 0;
}

/* Shared by `prepare` and `reset`, because a rewind is exactly a re-entry into
 * this state and nothing else. */
static int pocket_seed_context(mynah_engine_ctx *ctx, char *error, size_t capacity) {
    ctx->seeding = 0;
    if (pocket_seed_prologue(ctx, error, capacity) != 0) return -1;
    if (pocket_text_flush(ctx, !ctx->text_open, error, capacity) != 0) return -1;
    ctx->prepared = 1;
    return 0;
}

static int pocket_prepare(mynah_engine_ctx *ctx, char *error, size_t capacity) {
    if (ctx == NULL) {
        pocket_error(error, capacity, "pocket: null context");
        return -1;
    }
    mynah_region_begin(MYNAH_RGN_PREPARE);
    const int depth = mynah_region_depth();
    const int rc = pocket_seed_context(ctx, error, capacity);
    mynah_region_unwind(depth);
    mynah_region_end(MYNAH_RGN_PREPARE);
    return rc;
}

/* The resumable half of `prepare` (tts_engine.h, `prepare_slice`).
 *
 * Call one: the prologue, which cannot be interrupted and does not need to be.
 * Every call after it prefills at most `budget` tokens, stopping on a tile
 * boundary so the tiles land exactly where one `prepare` would have put them --
 * that is the bit-identity the seam promises, and `pocket_text_flush_limited`
 * asserts the alignment rather than trusting this caller.
 *
 * `*done` is set when the context is ready for step 1, which is the only moment
 * `prepared` becomes 1: a context that is half prefilled must not look ready to
 * anything that checks that flag. */
static int pocket_prepare_slice(mynah_engine_ctx *ctx, size_t budget, int *done,
                                char *error, size_t capacity) {
    if (ctx == NULL || done == NULL) {
        pocket_error(error, capacity, "pocket: null context or done flag");
        return -1;
    }
    *done = 0;
    mynah_region_begin(MYNAH_RGN_PREPARE);
    const int depth = mynah_region_depth();
    int rc = 0;
    if (!ctx->seeding) {
        rc = pocket_seed_prologue(ctx, error, capacity);
        if (rc == 0) ctx->seeding = 1;
    } else {
        rc = pocket_text_flush_limited(ctx, !ctx->text_open, budget, error, capacity);
    }
    if (rc == 0 && ctx->seeding) {
        const size_t target = ctx->text_open
            ? (ctx->text_length / (mynah_transformer_ar_prefill_tile() > 0u
                                   ? mynah_transformer_ar_prefill_tile() : 1u))
              * (mynah_transformer_ar_prefill_tile() > 0u
                 ? mynah_transformer_ar_prefill_tile() : 1u)
            : ctx->text_length;
        if (ctx->text_prefilled >= target) {
            ctx->prepared = 1;
            ctx->seeding = 0;
            *done = 1;
        }
    }
    mynah_region_unwind(depth);
    mynah_region_end(MYNAH_RGN_PREPARE);
    return rc;
}

static int pocket_reset(mynah_engine_ctx *ctx, char *error, size_t capacity) {
    if (ctx == NULL || !ctx->prepared) {
        pocket_error(error, capacity, "pocket: reset before prepare");
        return -1;
    }
    return pocket_seed_context(ctx, error, capacity);
}

/* ------------------------------------------------------------------- steps */

/* Is every value in `v` finite?
 *
 * `transformer_ar` asks this of its own input and refuses a non-finite one, but
 * it asks INSIDE the forward -- which, on the per-row path, is after the rows
 * before it have already advanced.  Asking here instead turns the one
 * data-dependent refusal a step can raise into a pre-flight, which is what
 * `step_batch`'s atomicity is built out of.
 *
 * The exponent bit test rather than isfinite(), because the answer must not
 * depend on how this file was compiled.  The default CFLAGS pair `-ffast-math`
 * with `-fno-finite-math-only`, which keeps isfinite() honest -- but CFLAGS is
 * `?=` and the GPU variants set their own, so a build where -ffast-math wins
 * and isfinite() folds to a constant is one overridden variable away.  The bit
 * test cannot be optimized into a lie. */
static int pocket_all_finite(const float *v, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        uint32_t bits;
        memcpy(&bits, &v[i], sizeof(bits));
        if ((bits & UINT32_C(0x7f800000)) == UINT32_C(0x7f800000)) return 0;
    }
    return 1;
}

/*
 * One AR step for `count` independent requests.
 *
 * The backbone is 302 MB of f32 weights (151 MB as f16) and one position's
 * worth of arithmetic, so a step run alone is a trip to memory, not a
 * calculation: N requests stepped one after another pay that trip N times for
 * the same bytes.  Stepping them together pays it once.  What must not change
 * is any single request's numbers -- see `pocket_linear_rows`.
 *
 * The previous latent's embedding stays per request: `input_linear` is
 * [1024][32], 128 KB, small enough that stacking it would cost more in
 * bookkeeping than it saves in traffic.
 *
 * ## ATOMIC OVER THE BATCH (E8-6)
 *
 * `tts_engine.h` requires that a non-zero return mean NO context advanced, and
 * the driver's failure isolation is built on it: `step_isolate()` re-steps the
 * batch one context at a time to find whose data was refused, and that re-step
 * is only legal if the refused call moved nobody.  An engine that advances
 * 0..i-1 and then refuses i gets its survivors double-stepped, and no driver
 * can see that from the outside.  At the `max_batch` of 64 this engine
 * declares, that is one bad request corrupting sixty-three strangers.
 *
 * It is enforced in three layers, in this order:
 *
 *   1. **A pre-flight that ends before the first mutation.**  Everything that
 *      can be decided from state alone -- prepared / finished / same model /
 *      not named twice, the batch width, and the KV capacity each backbone
 *      needs for one more position -- is decided for EVERY context before any
 *      of them is touched.
 *   2. **Mutations ordered so the fallible ones come first.**  The input
 *      projection can fail (it reaches the backend matmul), so it runs before
 *      anything advances, and it writes only `step_input`, which is a pure
 *      function of the request's own previous latent -- untouched by a refused
 *      call, so a re-step recomputes the identical bytes.  Its output is then
 *      checked for finiteness, which is the one data-dependent refusal the
 *      backbone raises, so that refusal also happens with every offset still
 *      where it was.
 *   3. **A rollback for what is left.**  After that, the only observable thing
 *      a backbone call advances is each state's KV offset, and
 *      `_state_set_offset` puts it back.  The K/V a partial pass wrote sits at
 *      or past the restored offset, where it is unreachable by exactly the
 *      invariant `_state_reset` already relies on: nothing from the offset
 *      onward is ever read, and it is overwritten before it becomes readable.
 *      So the snapshot/restore below is a true rollback and not a best effort
 *      -- and it holds whether the batch went through `_step_batch` or, with
 *      no batch scratch, through the per-row loop, which is the path that was
 *      actually non-atomic.
 *
 * `budget_exhausted` is decided in the pre-flight and committed at the end for
 * the same reason: on a refused call the context has to look untouched, and a
 * flag the caller can observe is part of "untouched".
 */
static int pocket_step_batch(mynah_engine_ctx *const *ctxs, size_t count,
                             mynah_engine_scratch *scratch, char *error,
                             size_t capacity) {
    if (ctxs == NULL) {
        pocket_error(error, capacity, "pocket: null batch");
        return -1;
    }
    if (count == 0u) return 0;
    /* The staging arrays below are fixed-size because the decode loop must not
     * allocate (AGENTS.md rule 4), so a batch wider than this engine declares
     * is refused rather than silently narrowed -- and refused before anything
     * has moved, which is the whole contract. */
    if (count > POCKET_MAX_BATCH) {
        pocket_error(error, capacity,
                     "pocket: batch of %zu exceeds max_batch %u", count,
                     POCKET_MAX_BATCH);
        return -1;
    }

    /* ---- 1. pre-flight: decided for every context, mutating none of them --- */
    size_t offset_before[POCKET_MAX_BATCH];
    int will_step[POCKET_MAX_BATCH];
    for (size_t i = 0; i < count; ++i) {
        mynah_engine_ctx *ctx = ctxs[i];
        if (ctx == NULL || !ctx->prepared) {
            pocket_error(error, capacity, "pocket: request %zu is not prepared", i);
            return -1;
        }
        /* A context whose text is still open has a partial tile waiting in
         * `text_embed` and is one text position short in the KV.  Stepping it
         * would generate from a prefix the caller has not finished writing, and
         * the audio would be conditioned on text that is missing its tail --
         * silently, and differently depending on where the pushes fell.  It is
         * a caller error, and it is refused here where nothing has moved yet. */
        if (ctx->text_open) {
            pocket_error(error, capacity,
                         "pocket: request %zu still has text open (%zu of %zu "
                         "tokens prefilled); seal it before stepping", i,
                         ctx->text_prefilled, ctx->text_length);
            return -1;
        }
        if (ctx->eos || ctx->broken) {
            pocket_error(error, capacity,
                         "pocket: request %zu has already finished; reset it first", i);
            return -1;
        }
        if (ctx->state != ctxs[0]->state) {
            pocket_error(error, capacity,
                         "pocket: request %zu belongs to a different model", i);
            return -1;
        }
        for (size_t j = 0; j < i; ++j) {
            /* Two slots naming one context would have the second write of a
             * position overwrite the first, and the rollback below would then
             * restore the wrong offset. */
            if (ctxs[j] == ctx) {
                pocket_error(error, capacity,
                             "pocket: request %zu appears twice in the batch", i);
                return -1;
            }
        }
        /* A request that has used its whole step budget retires in `emit`; it
         * must not be stepped, and it must not take the batch down with it. */
        will_step[i] = (ctx->step < ctx->max_steps);
        offset_before[i] = mynah_transformer_ar_state_offset(ctx->backbone);
        if (will_step[i]) {
            const mynah_transformer_ar_config *bc =
                mynah_transformer_ar_state_config(ctx->backbone);
            if (bc == NULL || offset_before[i] >= bc->max_seq_len) {
                /* `ctx_new` sizes the cache at voice + text + max_steps + 1, so
                 * this is unreachable for a context this engine built -- which
                 * is exactly why it is checked here rather than left to fail
                 * inside the forward, one row at a time, after its neighbours
                 * have already moved. */
                pocket_error(error, capacity,
                             "pocket: request %zu has no KV capacity left (%zu positions)",
                             i, offset_before[i]);
                return -1;
            }
        }
    }

    const mynah_engine_state *state = ctxs[0]->state;
    const pocket_config *cfg = &state->cfg;
    const size_t batch_capacity =
        (scratch == NULL) ? 0u
                          : mynah_transformer_ar_batch_capacity(scratch->backbone_batch);
    const int can_gather = scratch != NULL && scratch->backbone_batch != NULL &&
                           scratch->states != NULL && count <= batch_capacity;

    /* ---- 2. the fallible mutation, before anything advances --------------- */
    mynah_region_begin(MYNAH_RGN_STEP);
    mynah_region_begin2(MYNAH_RGN_STEP_EMBED);
    size_t live = 0;
    for (size_t i = 0; i < count; ++i) {
        mynah_engine_ctx *ctx = ctxs[i];
        if (!will_step[i]) continue;
        /* BOS is a tracked fact, not a NaN: no latent yet means bos_emb. */
        const float *previous =
            (ctx->frames > 0)
                ? ctx->latents + (ctx->frames - 1u) * cfg->latent_dim
                : state->bos_emb;
        /* The previous latent's embedding is [1024][32], 128 KB: small enough
         * that stacking it would cost more bookkeeping than it saves traffic,
         * so it stays per request -- and it is its own quantization group. */
        if (pocket_single_linear(state, POCKET_QG_COND_IN, state->cond_in_key,
                                 state->cond_in_qtype, state->input_linear, NULL,
                                 previous, ctx->step_input, cfg->latent_dim,
                                 cfg->hidden_dim) != 0) {
            mynah_region_end2(MYNAH_RGN_STEP_EMBED);
            mynah_region_end(MYNAH_RGN_STEP);
            pocket_error(error, capacity,
                         "pocket: the input projection failed for request %zu", i);
            return -1;
        }
        /* The backbone refuses a non-finite input -- and on the per-row path it
         * refuses it row by row, after the earlier rows have stepped.  Asked
         * here it is a pre-flight, and it names the guilty request instead of
         * whichever row the loop happened to reach first. */
        if (!pocket_all_finite(ctx->step_input, cfg->hidden_dim)) {
            mynah_region_end2(MYNAH_RGN_STEP_EMBED);
            mynah_region_end(MYNAH_RGN_STEP);
            pocket_error(error, capacity,
                         "pocket: request %zu produced a non-finite step input", i);
            return -1;
        }
        if (can_gather) {
            scratch->states[live] = ctx->backbone;
            scratch->inputs[live] = ctx->step_input;
            scratch->outputs[live] = ctx->hidden;
        }
        ++live;
    }
    mynah_region_end2(MYNAH_RGN_STEP_EMBED);

    /* ---- 3. the only call that advances anything, with a rollback --------- */
    mynah_region_begin2(MYNAH_RGN_STEP_BACKBONE);
    int failed = 0;
    size_t failed_at = 0;
    int cuda_used = 0;
    if (live > 1u && live == count) {
        char cuda_error[256];
        cuda_error[0] = '\0';
        const int cuda_rc = pocket_cuda_backbone_step_batch(
            ctxs, count, scratch, cuda_error, sizeof(cuda_error));
        if (cuda_rc == 0) {
            cuda_used = 1;
            (void)mynah_backend_note_backbone_batch(scratch->backend, count);
        }
    }
    if (live == 1u) {
        /* A single request is the latency-critical server path.  CUDA keeps
         * the whole transformer step resident; if a launch/sync fails, the
         * host KV offset is still unchanged and the ordinary CPU step below
         * can safely retry it. */
        mynah_engine_ctx *single = NULL;
        for (size_t i = 0; i < count; ++i) {
            if (will_step[i]) {
                single = ctxs[i];
                break;
            }
        }
        char cuda_error[256];
        cuda_error[0] = '\0';
        const int cuda_rc = pocket_cuda_backbone_step(
            single, cuda_error, sizeof(cuda_error));
        if (cuda_rc == 0) cuda_used = 1;
    }
    if (!cuda_used && live > 1u && can_gather) {
        failed = mynah_transformer_ar_step_batch(scratch->states, live,
                                                 &scratch->backbone_w,
                                                 scratch->backbone_batch,
                                                 scratch->inputs,
                                                 scratch->outputs) != 0;
    } else if (!cuda_used) {
        /* No batch scratch (or a single live request): the same graph, one row
         * at a time.  Not a second implementation -- `_step_batch` of one row
         * is `_step` -- just the path with nothing to share.  It is also the
         * path that cannot fail atomically on its own, which is what the
         * rollback below is for. */
        for (size_t i = 0; i < count && !failed; ++i) {
            mynah_engine_ctx *ctx = ctxs[i];
            if (!will_step[i]) continue;
            if (mynah_transformer_ar_step(ctx->backbone, &ctx->backbone_w,
                                          ctx->step_input, ctx->hidden) != 0) {
                failed = 1;
                failed_at = i;
            }
        }
    }
    mynah_region_end2(MYNAH_RGN_STEP_BACKBONE);
    mynah_region_end(MYNAH_RGN_STEP);
    if (failed) {
        /* Put every offset back.  Whatever K/V a partial pass wrote sits at or
         * past the restored offset, where nothing reads it and the next step
         * overwrites it -- the invariant `_state_reset` is already built on. */
        for (size_t i = 0; i < count; ++i) {
            if (!will_step[i]) continue;
            (void)mynah_transformer_ar_state_set_offset(ctxs[i]->backbone,
                                                        offset_before[i], NULL, 0);
        }
        pocket_error(error, capacity,
                     "pocket: the backbone step failed for request %zu of %zu",
                     failed_at, count);
        return -1;
    }

    /* ---- 4. commit ------------------------------------------------------- */
    for (size_t i = 0; i < count; ++i) {
        if (!will_step[i]) ctxs[i]->budget_exhausted = 1;
    }
    /* The parity dump is captured after the step, once per stepped request,
     * because the step no longer happens inside the per-request loop. */
    for (size_t i = 0; i < count; ++i) {
        mynah_engine_ctx *ctx = ctxs[i];
        if (!will_step[i] || ctx->dump == NULL) continue;
        if (ctx->dump->steps < ctx->dump->capacity) {
            memcpy(ctx->dump->hidden + ctx->dump->steps * cfg->hidden_dim,
                   ctx->hidden, cfg->hidden_dim * sizeof(float));
        }
    }
    return 0;
}

/*
 * Turn each stepped context's hidden state into an appended latent frame.
 *
 * Three passes rather than one loop, because the flow head is the second
 * weight-bound stage of the step and batching it needs its rows gathered first:
 *
 *   1. EOS, per request, and the noise draw from the request's OWN RNG.  The
 *      draw happens here, before any batching, so that a request's noise is a
 *      function of its seed and its step index and of nothing else -- not of
 *      batch width, not of slot order, not of who its neighbours were.
 *   2. One pass over the flow-head weights for every request that is still
 *      drawing a latent.
 *   3. The LSD addition and the bookkeeping.
 */
static int pocket_emit_batch(mynah_engine_ctx *const *ctxs, size_t count,
                             mynah_engine_step_result *results,
                             mynah_engine_scratch *scratch, char *error,
                             size_t capacity) {
    if (ctxs == NULL || results == NULL) {
        pocket_error(error, capacity, "pocket: null batch");
        return -1;
    }
    if (count == 0u) return 0;

    /* s and t, pinned at the endpoints; the manifest check at load time is what
     * makes this array the right length. */
    static const float times[2] = {0.0f, 1.0f};

    size_t gathered = 0;
    const size_t flow_capacity =
        (scratch == NULL) ? 0u : mynah_flow_head_batch_capacity(scratch->flow_batch);

    for (size_t i = 0; i < count; ++i) {
        mynah_engine_ctx *ctx = ctxs[i];
        memset(&results[i], 0, sizeof(results[i]));
        if (ctx == NULL || !ctx->prepared || ctx->broken) {
            pocket_error(error, capacity,
                         "pocket: request %zu is not prepared, or has failed", i);
            return -1;
        }
        if (ctx->budget_exhausted) {
            /* No step ran for this request, so `hidden` is last step's and must
             * not be read.  The budget is a length cap, not a failure: the
             * frames already generated are the answer. */
            ctx->eos = 1;
            results[i].eos = 1;
            results[i].eos_frame = 0u;
            results[i].frames_appended = 0u;
            continue;
        }
        const mynah_engine_state *state = ctx->state;
        const pocket_config *cfg = &state->cfg;

        mynah_region_begin(MYNAH_RGN_EMIT);
        mynah_region_begin(MYNAH_RGN_STEP_HEAD);
        if (pocket_single_linear(state, POCKET_QG_COND_EOS, state->cond_eos_key,
                                 state->cond_eos_qtype, state->out_eos_weight, state->out_eos_bias,
                                 ctx->hidden, &ctx->eos_logit, cfg->hidden_dim,
                                 1u) != 0) {
            mynah_region_end(MYNAH_RGN_STEP_HEAD);
            mynah_region_end(MYNAH_RGN_EMIT);
            pocket_error(error, capacity,
                         "pocket: the EOS projection failed for request %zu", i);
            return -1;
        }
        mynah_region_end(MYNAH_RGN_STEP_HEAD);
        if (ctx->dump != NULL && ctx->dump->steps < ctx->dump->capacity) {
            ctx->dump->eos[ctx->dump->steps] = ctx->eos_logit;
            ++ctx->dump->steps;
        }
        if (ctx->eos_step == SIZE_MAX && ctx->eos_logit > cfg->eos_threshold &&
            ctx->frames >= cfg->min_audio_frames) {
            ctx->eos_step = ctx->step;
        }
        /* The crossing itself does not end the request: upstream emits its
         * frame and `frames_after_eos` more, and breaks *before* queueing the
         * latent of the terminal step. So the terminal step is the one whose
         * single frame is invalid, hence eos_frame 0 and nothing appended. */
        if (ctx->eos_step != SIZE_MAX &&
            ctx->step >= ctx->eos_step + ctx->frames_after_eos) {
            ctx->eos = 1;
            ++ctx->step;
            results[i].eos = 1;
            results[i].eos_frame = 0u;
            results[i].frames_appended = 0u;
            mynah_region_end(MYNAH_RGN_EMIT);
            continue;
        }

        if (ctx->noise_fn != NULL) {
            if (ctx->noise_fn(ctx->noise_user, ctx->noise, cfg->latent_dim,
                              ctx->step) != 0) {
                ctx->broken = 1;
                results[i].failed = 1;
                mynah_region_end(MYNAH_RGN_EMIT);
                continue;
            }
        } else {
            for (size_t d = 0; d < cfg->latent_dim; ++d) {
                ctx->noise[d] = ctx->noise_std * pocket_rng_normal(ctx);
            }
        }
        mynah_region_end(MYNAH_RGN_EMIT);

        if (gathered < flow_capacity && scratch != NULL &&
            scratch->flow_heads != NULL) {
            scratch->flow_heads[gathered] = ctx->flow;
            scratch->flow_cond[gathered] = ctx->hidden;
            scratch->flow_noise[gathered] = ctx->noise;
            scratch->flow_out[gathered] = ctx->flow_out;
        }
        results[i].frames_appended = 1u; /* provisional: marks "wants a latent" */
        ++gathered;
    }

    mynah_region_begin(MYNAH_RGN_EMIT);
    /* MYNAH_RGN_FLOW, not MYNAH_RGN_LOCAL.  This is the flow head, and a local
     * transformer is a different graph: reporting it under "local.total" made
     * the pocket profile read as if it had Magpie's depth head. */
    mynah_region_begin(MYNAH_RGN_FLOW);
    int flow_failed = 0;
    int cuda_flow_used = 0;
    if (gathered > 0) {
        if (scratch != NULL && scratch->cuda_flow_enabled) {
            char cuda_error[256];
            cuda_error[0] = '\0';
            if (pocket_cuda_flow_step_batch(ctxs[0]->state, gathered, scratch,
                                            cuda_error, sizeof(cuda_error)) == 0)
                cuda_flow_used = 1;
        }
        if (!cuda_flow_used && gathered <= flow_capacity && scratch != NULL &&
            scratch->flow_heads != NULL) {
            flow_failed = mynah_flow_head_forward_batch(
                              scratch->flow_heads, gathered, &scratch->flow_w,
                              scratch->flow_cond, times, scratch->flow_noise,
                              scratch->flow_out, scratch->flow_batch) != 0;
        } else if (!cuda_flow_used) {
            for (size_t i = 0; i < count && !flow_failed; ++i) {
                mynah_engine_ctx *ctx = ctxs[i];
                if (results[i].frames_appended == 0u) continue;
                flow_failed = mynah_flow_head_forward(ctx->flow, &ctx->flow_w,
                                                      ctx->hidden, times,
                                                      ctx->noise,
                                                      ctx->flow_out) != 0;
            }
        }
    }
    mynah_region_end(MYNAH_RGN_FLOW);
    mynah_region_end(MYNAH_RGN_EMIT);

    for (size_t i = 0; i < count; ++i) {
        mynah_engine_ctx *ctx = ctxs[i];
        if (results[i].frames_appended == 0u) continue;
        const pocket_config *cfg = &ctx->state->cfg;
        if (flow_failed) {
            /* The pass is shared, so a failure inside it is not attributable to
             * one request; every request that was in it is over. */
            ctx->broken = 1;
            results[i].failed = 1;
            results[i].frames_appended = 0u;
            continue;
        }
        /* One LSD step from s = 0 to t = 1: the integration is the addition. */
        float *latent = ctx->latents + ctx->frames * cfg->latent_dim;
        for (size_t d = 0; d < cfg->latent_dim; ++d) {
            latent[d] = ctx->noise[d] + ctx->flow_out[d];
        }
        if (ctx->dump != NULL && ctx->dump->frames < ctx->dump->capacity) {
            const size_t off = ctx->dump->frames * cfg->latent_dim;
            memcpy(ctx->dump->flow_out + off, ctx->flow_out,
                   cfg->latent_dim * sizeof(float));
            memcpy(ctx->dump->latent + off, latent,
                   cfg->latent_dim * sizeof(float));
            ++ctx->dump->frames;
        }
        ++ctx->frames;
        ++ctx->step;
        results[i].eos = 0;
        results[i].eos_frame = 1u;
        results[i].frames_appended = 1u;
    }
    return 0;
}

static size_t pocket_frame_count(const mynah_engine_ctx *ctx) {
    return (ctx == NULL) ? 0u : ctx->frames;
}

/* Only the frame history shrinks. The codec has already consumed whatever was
 * decoded and cannot be rewound, so a truncation below `decoded_frames` is not
 * undone here; `decode_audio` refuses the next range instead of quietly
 * decoding frames that no longer exist. */
static void pocket_truncate(mynah_engine_ctx *ctx, size_t frame_count) {
    if (ctx == NULL) return;
    if (frame_count < ctx->frames) ctx->frames = frame_count;
}

/* ------------------------------------------------------------------ audio */

/*
 * ONE latent frame through the codec, into `ctx->pcm`.
 *
 * This is the whole codec, and it exists as its own function because the engine
 * now decodes frames from two places -- one context's range, and a gang of
 * ranges belonging to different contexts.  Those are two schedules over one
 * body, not two implementations (AGENTS.md rule 7), and writing it that way is
 * what makes `decode_audio_batch`'s bit-identity STRUCTURAL rather than a
 * property the tests have to keep rediscovering: there is no second arithmetic
 * path for a frame to take when it happens to share a call.
 *
 * Every operation reads and writes this context's own state only -- the codec
 * ring buffers, the decoder transformer's KV, the position counter -- so the
 * order in which contexts are visited cannot reach any of the numbers.
 */
/* Prepare the host-owned portion of one codec frame.  Decoder submission is
 * deliberately outside this helper so a gang can prepare every context before
 * it queues any D2H or synchronization work. */
static int pocket_decode_frame_prepare(mynah_engine_ctx *ctx, size_t frame,
                                       char *error, size_t capacity) {
    const mynah_engine_state *state = ctx->state;
    const pocket_config *cfg = &state->cfg;
    const size_t stride = cfg->upsample_stride;
    const size_t dim = cfg->codec_dim;
    const int depth = mynah_region_depth();

    const float *latent = ctx->latents + frame * cfg->latent_dim;
    mynah_region_begin2(MYNAH_RGN_CODEC_EMBED);
    for (size_t d = 0; d < cfg->latent_dim; ++d) {
        ctx->denorm[d] = latent[d] * state->emb_std[d] + state->emb_mean[d];
    }
    /* quantizer.output_proj is Conv1d(32, 512, 1): one matvec per frame. */
    if (pocket_single_linear(state, POCKET_QG_CODEC_CONV, state->codec_conv_key,
                             state->codec_conv_qtype, state->quantizer_proj, NULL,
                             ctx->denorm, ctx->codec_in, cfg->latent_dim, dim) != 0) {
        mynah_region_unwind(depth);
        pocket_error(error, capacity, "pocket: the quantizer projection failed");
        return -1;
    }
    if (mynah_seanet_upsample(ctx->codec, &state->upsample, ctx->codec_in, 1u,
                              ctx->codec_up) != 0) {
        mynah_region_unwind(depth);
        pocket_error(error, capacity, "pocket: the codec upsample failed");
        return -1;
    }
    mynah_region_end2(MYNAH_RGN_CODEC_EMBED);

    /* The decoder transformer runs at the encoder frame rate and its inner
     * layers see [positions, channels]; the transpose belongs here, at the
     * same place the reference puts it. */
    mynah_region_begin2(MYNAH_RGN_CODEC_TRANSFORMER);
    for (size_t c = 0; c < dim; ++c) {
        for (size_t t = 0; t < stride; ++t) {
            ctx->codec_seq[t * dim + c] = ctx->codec_up[c * stride + t];
        }
    }
    /* Two counters, both mandatory (E2-3): the ring buffers inside the SEANet
     * state, and this position. If they ever disagree the audio degrades
     * smoothly and silently, so they are compared instead. */
    if (mynah_seanet_state_position(ctx->codec) !=
        mynah_transformer_ar_state_offset(ctx->codec_transformer)) {
        mynah_region_unwind(depth);
        pocket_error(error, capacity,
                     "pocket: codec position %zu != decoder transformer offset %zu",
                     mynah_seanet_state_position(ctx->codec),
                     mynah_transformer_ar_state_offset(ctx->codec_transformer));
        return -1;
    }
    if (mynah_transformer_ar_prefill(ctx->codec_transformer, &ctx->codec_w,
                                     ctx->codec_seq, stride, ctx->codec_out) != 0) {
        mynah_region_unwind(depth);
        pocket_error(error, capacity, "pocket: the decoder transformer failed");
        return -1;
    }
    for (size_t t = 0; t < stride; ++t) {
        for (size_t c = 0; c < dim; ++c) {
            ctx->codec_back[c * stride + t] = ctx->codec_out[t * dim + c];
        }
    }
    mynah_region_end2(MYNAH_RGN_CODEC_TRANSFORMER);
    if (ctx->dump != NULL && ctx->dump->decoded < ctx->dump->capacity) {
        const size_t slot = ctx->dump->decoded;
        memcpy(ctx->dump->denorm + slot * cfg->latent_dim, ctx->denorm,
               cfg->latent_dim * sizeof(float));
        memcpy(ctx->dump->codec_tf + slot * ctx->dump->codec_row, ctx->codec_out,
               ctx->dump->codec_row * sizeof(float));
    }
    return 0;
}

/* Finish the common host-side state transition after either a CPU decoder or
 * a resident CUDA decoder has produced ctx->pcm. */
static int pocket_decode_frame_finish(mynah_engine_ctx *ctx, int cuda_decode,
                                      char *error, size_t capacity) {
    const mynah_engine_state *state = ctx->state;
    const pocket_config *cfg = &state->cfg;
    const size_t stride = cfg->upsample_stride;
    const int depth = mynah_region_depth();
    if (cuda_decode < 0 || cuda_decode > 1) {
        pocket_error(error, capacity, "pocket: invalid decoder completion state");
        return -1;
    }
    mynah_region_begin2(MYNAH_RGN_CODEC_CONV);
    if (cuda_decode > 0 &&
        mynah_seanet_decode(ctx->codec, &state->decoder, ctx->codec_back, stride,
                            ctx->pcm) != 0) {
        mynah_region_unwind(depth);
        pocket_error(error, capacity, "pocket: the SEANet decoder failed");
        return -1;
    }
    mynah_region_end2(MYNAH_RGN_CODEC_CONV);
    mynah_seanet_state_advance(ctx->codec, 1u);
    if (ctx->dump != NULL && ctx->dump->decoded < ctx->dump->capacity) {
        memcpy(ctx->dump->pcm + ctx->dump->decoded * ctx->dump->frame_samples,
               ctx->pcm, ctx->dump->frame_samples * sizeof(float));
        ++ctx->dump->decoded;
    }
    return 0;
}

/* One context's compatibility schedule: prepare, submit/collect/sync, then
 * finish.  The gang schedule below calls the same prepare/finish helpers but
 * drains one shared backend stream for all resident decoder submissions. */
static int pocket_decode_frame(mynah_engine_ctx *ctx, size_t frame, char *error,
                               size_t capacity) {
    const int depth = mynah_region_depth();
    if (pocket_decode_frame_prepare(ctx, frame, error, capacity) != 0) {
        mynah_region_unwind(depth);
        return -1;
    }
    const int cuda_decode = pocket_cuda_decoder_step(ctx, error, capacity);
    if (cuda_decode < 0) {
        mynah_region_unwind(depth);
        return -1;
    }
    return pocket_decode_frame_finish(ctx, cuda_decode, error, capacity);
}

/*
 * Admit one range and hand back the buffer its samples go into.
 *
 * Shared by the single and the gang entry points so that a range is judged by
 * exactly one piece of code: the gang's promise is that a context cannot tell
 * who else was in the call, and that has to cover which ranges are legal, not
 * only what the samples come out as.
 *
 * `*out_pcm` is malloc'd and becomes the caller's on success; on any refusal it
 * is NULL and nothing about the context has changed.
 */
static int pocket_decode_admit(mynah_engine_ctx *ctx, size_t first_frame,
                               size_t frame_count, float **out_pcm,
                               size_t *out_samples, char *error, size_t capacity) {
    *out_pcm = NULL;
    *out_samples = 0;
    if (ctx == NULL) {
        pocket_error(error, capacity, "pocket: null argument decoding audio");
        return -1;
    }
    if (first_frame != ctx->decoded_frames) {
        /* The codec carries state instead of replaying context (E2-3), so a
         * gap or a rewind cannot be served; saying so beats emitting audio
         * that is quietly built on the wrong history. */
        pocket_error(error, capacity,
                     "pocket: frame ranges must be contiguous; asked for %zu, the "
                     "codec is at %zu",
                     first_frame, ctx->decoded_frames);
        return -1;
    }
    /* `first_frame > frames` is reachable: `truncate` can drop the history
     * below what has already been decoded, and the codec cannot be rewound.
     * Checking it separately keeps the subtraction below from wrapping. */
    if (first_frame > ctx->frames || frame_count > ctx->frames - first_frame) {
        pocket_error(error, capacity, "pocket: asked for frames [%zu, %zu) of %zu",
                     first_frame, first_frame + frame_count, ctx->frames);
        return -1;
    }
    if (frame_count == 0) return 0;

    size_t samples = 0;
    if (pocket_mul(frame_count, ctx->state->cfg.samples_per_frame, &samples) != 0) {
        pocket_error(error, capacity, "pocket: sample count overflow");
        return -1;
    }
    float *pcm = mynah_alloc_floats(samples, error, capacity);
    if (pcm == NULL) return -1;
    *out_pcm = pcm;
    *out_samples = samples;
    return 0;
}

static int pocket_decode_audio(mynah_engine_ctx *ctx, size_t first_frame,
                               size_t frame_count, float **out_samples,
                               size_t *out_count, char *error, size_t capacity) {
    if (out_samples != NULL) *out_samples = NULL;
    if (out_count != NULL) *out_count = 0;
    if (ctx == NULL || out_samples == NULL || out_count == NULL) {
        pocket_error(error, capacity, "pocket: null argument decoding audio");
        return -1;
    }
    float *pcm = NULL;
    size_t samples = 0;
    if (pocket_decode_admit(ctx, first_frame, frame_count, &pcm, &samples, error,
                            capacity) != 0) {
        return -1;
    }
    if (pcm == NULL) return 0; /* an empty range is a legal no-op */

    const size_t frame_samples = ctx->state->cfg.samples_per_frame;
    mynah_region_begin(MYNAH_RGN_CODEC);
    for (size_t f = 0; f < frame_count; ++f) {
        if (pocket_decode_frame(ctx, first_frame + f, error, capacity) != 0) {
            free(pcm);
            mynah_region_end(MYNAH_RGN_CODEC);
            /* The codec advanced through `f` frames that nobody will ever hear
             * and cannot be rewound, so `decoded_frames` now disagrees with the
             * conv rings. A later range would pass the contiguity check and be
             * decoded from the wrong history; saying the request is over is the
             * difference between a failed request and silent corruption. */
            ctx->broken = 1;
            return -1;
        }
        memcpy(pcm + f * frame_samples, ctx->pcm, frame_samples * sizeof(float));
    }
    mynah_region_end(MYNAH_RGN_CODEC);

    ctx->decoded_frames += frame_count;
    *out_samples = pcm;
    *out_count = samples;
    return 0;
}

/*
 * ---- decode a gang ----------------------------------------------------
 *
 * The contract is in `tts_engine.h`; what follows is how this engine meets it
 * and, just as importantly, what it does NOT yet do.
 *
 * ## Bit-identity, and why it is not a tolerance here
 *
 * Whoever shares a call, how many there are and in what order they appear are a
 * scheduling decision the driver remakes every step on timing, so anything that
 * crossed between rows would make a request's audio depend on server load. The
 * guarantee here is structural rather than tested-and-hoped: every frame goes
 * through the prepare/finish helpers above, which touch this context's buffers
 * and no others. The CUDA schedule may submit several resident decoders before
 * one stream drain, but it never aliases their causal state or changes their
 * arithmetic.
 *
 * ## Frame-major, and what that is and is not worth
 *
 * The gang is walked frame index by frame index, all contexts at each index,
 * rather than context by context. That is NOT a performance claim: the codec
 * transformer and the SEANet stack are still one pass over their own weights
 * per context per frame, exactly as before, and nothing here was benchmarked
 * (this machine compiles for correctness only). Frame-major is chosen because
 * it is the schedule a shared pass would need, and adopting it now means the
 * step that actually shares work is local to this function instead of a
 * restructuring of it.
 *
 * What blocks true cross-request arithmetic is named rather than implied: the
 * 41.8% item is the decoder transformer, and sharing it across contexts needs
 * a cross-request PREFILL in `transformer_ar` -- `_step_batch` takes one
 * position per state, while a codec frame is `upsample_stride` consecutive
 * positions of one state. The resident SEANet gang below is therefore a
 * launch/overlap slice, not a claim that the decoder is already a Bx kernel.
 *
 * ## Blast radius
 *
 * A context whose range is refused, or whose codec fails partway, sets failed[i]
 * and keeps its neighbours running: it is dropped from the remaining frame
 * passes and every other context finishes its own range. The return value is
 * non-zero only for something that belongs to no single context.
 */
static int pocket_cuda_decoder_batch_enabled(void) {
    const char *value = getenv("MYNAH_CUDA_DECODER_BATCH");
    return value == NULL || strcmp(value, "0") != 0;
}

static void pocket_decode_batch_drop(mynah_engine_ctx *ctx, size_t index,
                                     float **out_samples, size_t *out_count,
                                     int *failed, int *reported,
                                     const char *detail, char *error,
                                     size_t capacity) {
    if (ctx != NULL) ctx->broken = 1;
    if (out_samples != NULL && out_samples[index] != NULL) {
        free(out_samples[index]);
        out_samples[index] = NULL;
    }
    if (out_count != NULL) out_count[index] = 0u;
    if (failed != NULL) failed[index] = 1;
    if (reported != NULL && !*reported) {
        pocket_error(error, capacity, "%s", detail != NULL && detail[0] != '\0'
                                             ? detail
                                             : "pocket: decoding audio failed");
        *reported = 1;
    }
}

static int pocket_decode_audio_batch(mynah_engine_ctx *const *ctxs, size_t count,
                                     const size_t *first_frame,
                                     const size_t *frame_count, float **out_samples,
                                     size_t *out_count, int *failed,
                                     mynah_engine_scratch *scratch, char *error,
                                     size_t capacity) {
    /* The driver owns the arrays and pre-clears them; nothing in `scratch` is
     * needed while the codec is per context, and nothing here keeps a pointer
     * into any of them. */
    (void)scratch;
    if (ctxs == NULL || first_frame == NULL || frame_count == NULL ||
        out_samples == NULL || out_count == NULL || failed == NULL) {
        pocket_error(error, capacity, "pocket: null argument decoding a gang");
        return -1;
    }
    if (count == 0u) return 0;
    if (count > POCKET_MAX_BATCH) {
        /* Not a buffer bound -- this function stages nothing and the loop below
         * would serve any width. It is refused because a gang wider than the
         * `max_batch` this engine publishes means the driver and the engine
         * disagree about the declared width, and the useful moment to say so is
         * the first call, not whenever some other limit happens to bite. It
         * belongs to no single context, so it is a whole-call failure. */
        for (size_t i = 0; i < count; ++i) failed[i] = 1;
        pocket_error(error, capacity, "pocket: gang of %zu exceeds max_batch %u",
                     count, POCKET_MAX_BATCH);
        return -1;
    }

    /* `tts_engine.h` grants that no context appears twice, so this is checking
     * the driver rather than the input -- but a duplicate is the one violation
     * that would corrupt audio silently instead of failing: both entries pass
     * the contiguity check, because `decoded_frames` is only advanced at the
     * end, and the frames then interleave through one codec state. At a width
     * of at most 16 the check is 120 comparisons. */
    for (size_t i = 0; i < count; ++i) {
        for (size_t j = 0; j < i; ++j) {
            if (ctxs[j] != ctxs[i]) continue;
            for (size_t k = 0; k < count; ++k) failed[k] = 1;
            pocket_error(error, capacity,
                         "pocket: request %zu appears twice in the gang", i);
            return -1;
        }
    }

    size_t longest = 0;
    int reported = 0;
    char one_error[256];
    for (size_t i = 0; i < count; ++i) {
        float *pcm = NULL;
        size_t samples = 0;
        one_error[0] = '\0';
        if (pocket_decode_admit(ctxs[i], first_frame[i], frame_count[i], &pcm,
                                &samples, one_error, sizeof(one_error)) != 0) {
            failed[i] = 1;
            if (!reported) {
                pocket_error(error, capacity, "%s",
                             one_error[0] != '\0' ? one_error
                                                  : "pocket: decoding audio failed");
                reported = 1;
            }
            continue;
        }
        out_samples[i] = pcm;   /* NULL for an empty range, which is legal */
        out_count[i] = samples;
        if (pcm != NULL && frame_count[i] > longest) longest = frame_count[i];
    }

    if (longest == 0u) return 0;
    const mynah_backend *batch_backend = NULL;
    int use_async_decoder = pocket_cuda_decoder_batch_enabled();
    if (use_async_decoder) {
        for (size_t i = 0; i < count; ++i) {
            if (failed[i] || out_samples[i] == NULL) continue;
            const mynah_backend *backend = ctxs[i]->state->backend;
            if (backend == NULL ||
                (batch_backend != NULL && backend != batch_backend)) {
                use_async_decoder = 0;
                break;
            }
            if (batch_backend == NULL) batch_backend = backend;
        }
        if (batch_backend == NULL ||
            strcmp(mynah_backend_name(batch_backend), "cuda") != 0)
            use_async_decoder = 0;
    }

    mynah_region_begin(MYNAH_RGN_CODEC);
    if (!use_async_decoder) {
        /* This is the unchanged CPU/compatibility schedule. */
        for (size_t f = 0; f < longest; ++f) {
            for (size_t i = 0; i < count; ++i) {
                if (failed[i] || out_samples[i] == NULL || f >= frame_count[i]) continue;
                mynah_engine_ctx *ctx = ctxs[i];
                const size_t frame_samples = ctx->state->cfg.samples_per_frame;
                one_error[0] = '\0';
                if (pocket_decode_frame(ctx, first_frame[i] + f, one_error,
                                        sizeof(one_error)) != 0) {
                    /* This context's codec advanced through frames nobody will
                     * hear and cannot be rewound; its neighbours are untouched. */
                    pocket_decode_batch_drop(ctx, i, out_samples, out_count,
                                             failed, &reported, one_error,
                                             error, capacity);
                    continue;
                }
                memcpy(out_samples[i] + f * frame_samples, ctx->pcm,
                       frame_samples * sizeof(float));
            }
        }
    } else {
        /* Prepare all host state first, then queue every resident decoder.  A
         * CUDA graph identity is per decoder, so graphs remain independent even
         * though their launches share this backend stream. */
        int prepared[POCKET_MAX_BATCH];
        int submitted[POCKET_MAX_BATCH];
        for (size_t f = 0; f < longest; ++f) {
            memset(prepared, 0, sizeof(prepared));
            memset(submitted, 0, sizeof(submitted));
            size_t submitted_count = 0u;

            for (size_t i = 0; i < count; ++i) {
                if (failed[i] || out_samples[i] == NULL || f >= frame_count[i]) continue;
                one_error[0] = '\0';
                if (pocket_decode_frame_prepare(ctxs[i], first_frame[i] + f,
                                                one_error, sizeof(one_error)) != 0) {
                    pocket_decode_batch_drop(ctxs[i], i, out_samples, out_count,
                                             failed, &reported, one_error,
                                             error, capacity);
                    continue;
                }
                prepared[i] = 1;
            }

            for (size_t i = 0; i < count; ++i) {
                if (!prepared[i] || failed[i]) continue;
                mynah_engine_ctx *ctx = ctxs[i];
                one_error[0] = '\0';
                if (!ctx->cuda_decoder_enabled || ctx->cuda_decoder == NULL) {
                    if (pocket_decode_frame_finish(ctx, 1, one_error,
                                                   sizeof(one_error)) != 0)
                        pocket_decode_batch_drop(ctx, i, out_samples, out_count,
                                                 failed, &reported, one_error,
                                                 error, capacity);
                    continue;
                }
                const int status = pocket_cuda_decoder_submit(
                    ctx, one_error, sizeof(one_error));
                if (status == 0) {
                    submitted[i] = 1;
                    ++submitted_count;
                } else if (status > 0) {
                    /* The device path was unavailable before state advance;
                     * CPU remains an allowed per-request fallback. */
                    if (pocket_decode_frame_finish(ctx, 1, one_error,
                                                   sizeof(one_error)) != 0)
                        pocket_decode_batch_drop(ctx, i, out_samples, out_count,
                                                 failed, &reported, one_error,
                                                 error, capacity);
                } else {
                    pocket_decode_batch_drop(ctx, i, out_samples, out_count,
                                             failed, &reported, one_error,
                                             error, capacity);
                }
            }

            if (submitted_count > 0u) {
                (void)mynah_backend_decoder_note_batch(batch_backend,
                                                       submitted_count, 1u);
                for (size_t i = 0; i < count; ++i) {
                    if (!submitted[i] || failed[i]) continue;
                    one_error[0] = '\0';
                    if (pocket_cuda_decoder_collect(ctxs[i], one_error,
                                                    sizeof(one_error)) != 0)
                        pocket_decode_batch_drop(ctxs[i], i, out_samples,
                                                 out_count, failed, &reported,
                                                 one_error, error, capacity);
                }

                char sync_error[256];
                sync_error[0] = '\0';
                const int sync_failed = mynah_backend_sync(
                    batch_backend, sync_error, sizeof(sync_error)) != 0;
                if (sync_failed) {
                    for (size_t i = 0; i < count; ++i) {
                        if (submitted[i] && !failed[i])
                            pocket_decode_batch_drop(ctxs[i], i, out_samples,
                                                     out_count, failed, &reported,
                                                     sync_error, error, capacity);
                    }
                } else {
                    for (size_t i = 0; i < count; ++i) {
                        if (!submitted[i] || failed[i]) continue;
                        size_t input_floats = 0u;
                        size_t output_floats = 0u;
                        one_error[0] = '\0';
                        if (pocket_cuda_decoder_sizes(ctxs[i], &input_floats,
                                                      &output_floats) != 0 ||
                            !pocket_all_finite(ctxs[i]->pcm, output_floats)) {
                            snprintf(one_error, sizeof(one_error),
                                     "CUDA decoder produced non-finite PCM");
                            pocket_decode_batch_drop(ctxs[i], i, out_samples,
                                                     out_count, failed, &reported,
                                                     one_error, error, capacity);
                            continue;
                        }
                        if (pocket_decode_frame_finish(ctxs[i], 0, one_error,
                                                       sizeof(one_error)) != 0)
                            pocket_decode_batch_drop(ctxs[i], i, out_samples,
                                                     out_count, failed, &reported,
                                                     one_error, error, capacity);
                    }
                }
            }

            for (size_t i = 0; i < count; ++i) {
                if (failed[i] || out_samples[i] == NULL || f >= frame_count[i]) continue;
                const size_t frame_samples = ctxs[i]->state->cfg.samples_per_frame;
                memcpy(out_samples[i] + f * frame_samples, ctxs[i]->pcm,
                       frame_samples * sizeof(float));
            }
        }
    }
    mynah_region_end(MYNAH_RGN_CODEC);

    for (size_t i = 0; i < count; ++i) {
        if (failed[i] || out_samples[i] == NULL) continue;
        ctxs[i]->decoded_frames += frame_count[i];
    }
    return 0;
}

/* ---------------------------------------------------------------- scratch */

/* The CUDA attention metadata is tiny, but it is copied by async graph nodes.
 * Keep it pinned just like the activation staging so a graph replay never
 * falls back to an implicit pageable-host staging allocation.  The backend's
 * host allocator is expressed in floats for the activation API; round byte
 * metadata up to that allocator's unit and retain the original pointer. */
static int pocket_host_alloc_bytes(const mynah_backend *backend, size_t bytes,
                                   void **out, char *error, size_t capacity) {
    if (out == NULL || backend == NULL || bytes == 0u ||
        bytes > SIZE_MAX - (sizeof(float) - 1u)) return -1;
    *out = NULL;
    size_t floats = (bytes + sizeof(float) - 1u) / sizeof(float);
    float *raw = NULL;
    if (mynah_backend_host_alloc(backend, floats, &raw, error, capacity) != 0)
        return -1;
    *out = raw;
    return 0;
}

static void pocket_host_free_bytes(const mynah_backend *backend, void *ptr) {
    if (backend != NULL && ptr != NULL)
        mynah_backend_host_free(backend, (float *)ptr);
}

static void pocket_cuda_flow_release(mynah_engine_scratch *scratch) {
    if (scratch == NULL) return;
    const mynah_backend *backend = scratch->backend;
    if (backend != NULL) {
        mynah_backend_dev_free(backend, scratch->cuda_flow_cond);
        mynah_backend_dev_free(backend, scratch->cuda_flow_noise);
        mynah_backend_dev_free(backend, scratch->cuda_flow_time_embed);
        mynah_backend_dev_free(backend, scratch->cuda_flow_output);
        mynah_backend_dev_free(backend, scratch->cuda_flow_y);
        mynah_backend_dev_free(backend, scratch->cuda_flow_silu);
        mynah_backend_dev_free(backend, scratch->cuda_flow_x);
        mynah_backend_dev_free(backend, scratch->cuda_flow_norm);
        mynah_backend_dev_free(backend, scratch->cuda_flow_hidden);
        mynah_backend_dev_free(backend, scratch->cuda_flow_scratch);
        mynah_backend_dev_free(backend, scratch->cuda_flow_mod);
        mynah_backend_dev_free(backend, scratch->cuda_flow_final_mod);
        mynah_backend_dev_free(backend, scratch->cuda_flow_time_hidden);
        mynah_backend_dev_free(backend, scratch->cuda_flow_time_output);
        mynah_backend_dev_free(backend, scratch->cuda_flow_time_sum);
        mynah_backend_host_free(backend, scratch->cuda_flow_host_cond);
        mynah_backend_host_free(backend, scratch->cuda_flow_host_noise);
        mynah_backend_host_free(backend, scratch->cuda_flow_host_time);
        mynah_backend_host_free(backend, scratch->cuda_flow_host_output);
    } else {
        free(scratch->cuda_flow_host_cond);
        free(scratch->cuda_flow_host_noise);
        free(scratch->cuda_flow_host_time);
        free(scratch->cuda_flow_host_output);
    }
    scratch->cuda_flow_host_cond = NULL;
    scratch->cuda_flow_host_noise = NULL;
    scratch->cuda_flow_host_time = NULL;
    scratch->cuda_flow_host_output = NULL;
    scratch->cuda_flow_cond = NULL;
    scratch->cuda_flow_noise = NULL;
    scratch->cuda_flow_time_embed = NULL;
    scratch->cuda_flow_output = NULL;
    scratch->cuda_flow_y = NULL;
    scratch->cuda_flow_silu = NULL;
    scratch->cuda_flow_x = NULL;
    scratch->cuda_flow_norm = NULL;
    scratch->cuda_flow_hidden = NULL;
    scratch->cuda_flow_scratch = NULL;
    scratch->cuda_flow_mod = NULL;
    scratch->cuda_flow_final_mod = NULL;
    scratch->cuda_flow_time_hidden = NULL;
    scratch->cuda_flow_time_output = NULL;
    scratch->cuda_flow_time_sum = NULL;
    free(scratch->cuda_flow_time_mlp_in);
    free(scratch->cuda_flow_time_mlp_out);
    free((void *)scratch->cuda_flow_time_alpha);
    free(scratch->cuda_flow_blocks);
    scratch->cuda_flow_time_mlp_in = NULL;
    scratch->cuda_flow_time_mlp_out = NULL;
    scratch->cuda_flow_time_alpha = NULL;
    scratch->cuda_flow_blocks = NULL;
    memset(&scratch->cuda_flow_cond_embed, 0,
           sizeof(scratch->cuda_flow_cond_embed));
    memset(&scratch->cuda_flow_input_proj, 0,
           sizeof(scratch->cuda_flow_input_proj));
    memset(&scratch->cuda_flow_final_adaln, 0,
           sizeof(scratch->cuda_flow_final_adaln));
    memset(&scratch->cuda_flow_final_linear, 0,
           sizeof(scratch->cuda_flow_final_linear));
    scratch->cuda_flow_batch_capacity = 0u;
    scratch->cuda_flow_enabled = 0;
    scratch->cuda_flow_graph_ready = 0;
}

static void pocket_scratch_free(mynah_engine_scratch *scratch) {
    if (scratch == NULL) return;
    if (scratch->backend != NULL) {
        mynah_backend_graph_forget(scratch->backend, scratch);
        pocket_cuda_flow_release(scratch);
        mynah_backend_dev_free(scratch->backend, scratch->cuda_x);
        mynah_backend_dev_free(scratch->backend, scratch->cuda_norm);
        mynah_backend_dev_free(scratch->backend, scratch->cuda_qkv);
        mynah_backend_dev_free(scratch->backend, scratch->cuda_attn);
        mynah_backend_dev_free(scratch->backend, scratch->cuda_proj);
        mynah_backend_dev_free(scratch->backend, scratch->cuda_ffn);
        mynah_backend_dev_free(scratch->backend, scratch->cuda_kv_shadow);
        mynah_backend_host_free(scratch->backend, scratch->cuda_host_input);
        mynah_backend_host_free(scratch->backend, scratch->cuda_host_output);
        mynah_backend_host_free(scratch->backend, scratch->cuda_host_kv);
    } else {
        pocket_cuda_flow_release(scratch);
        free(scratch->cuda_host_input);
        free(scratch->cuda_host_output);
        free(scratch->cuda_host_kv);
        free(scratch->cuda_kv_shadow);
    }
    if (scratch->backend != NULL) {
        pocket_host_free_bytes(scratch->backend, scratch->cuda_kcache);
        pocket_host_free_bytes(scratch->backend, scratch->cuda_vcache);
        pocket_host_free_bytes(scratch->backend, scratch->cuda_positions);
        pocket_host_free_bytes(scratch->backend, scratch->cuda_cache_strides);
    } else {
        free(scratch->cuda_kcache);
        free(scratch->cuda_vcache);
        free(scratch->cuda_positions);
        free(scratch->cuda_cache_strides);
    }
    mynah_transformer_ar_batch_free(scratch->backbone_batch);
    mynah_flow_head_batch_free(scratch->flow_batch);
    pocket_call_release(&scratch->backbone_call.call);
    pocket_call_release(&scratch->flow_call.call);
    free(scratch->states);
    free(scratch->inputs);
    free((void *)scratch->outputs);
    free(scratch->flow_heads);
    free(scratch->flow_cond);
    free(scratch->flow_noise);
    free((void *)scratch->flow_out);
    free(scratch);
}

/*
 * The driver's scratch: everything one batched step needs that belongs to the
 * batch rather than to a request in it.
 *
 * It is sized for the widest batch the driver will ever run, not for the batch
 * it happens to start with, because admission is continuous: a request that
 * arrives mid-flight widens the batch, and a scratch that was allocated for the
 * narrower one would either overflow or -- worse -- be NULL and send every row
 * back down the per-row path with nothing saying so.
 */
static int pocket_scratch_new(const mynah_tts_model *model,
                              mynah_engine_state *state, size_t batch,
                              mynah_engine_scratch **out, char *error,
                              size_t capacity) {
    (void)model;
    if (out == NULL || state == NULL) return -1;
    *out = NULL;
    if (batch == 0u) batch = 1u;
    mynah_engine_scratch *scratch =
        (mynah_engine_scratch *)calloc(1, sizeof(*scratch));
    if (scratch == NULL) {
        pocket_error(error, capacity, "out of memory creating pocket scratch");
        return -1;
    }
    scratch->batch = batch;
    scratch->backend = state->backend;
    if (batch == 1u) {
        /* Nothing to share: a one-slot driver keeps the single-step path. */
        *out = scratch;
        return 0;
    }

    const pocket_config *cfg = &state->cfg;
    size_t attn_dim = 0u;
    size_t qkv_width = 0u;
    size_t kv_metadata = 0u;
    size_t kv_shadow_floats = 0u;
    if (pocket_mul(cfg->heads, cfg->head_dim, &attn_dim) != 0 ||
        pocket_mul(attn_dim, 3u, &qkv_width) != 0 ||
        pocket_mul(batch, cfg->layers, &kv_metadata) != 0 ||
        pocket_mul(batch, cfg->layers, &kv_shadow_floats) != 0 ||
        pocket_mul(kv_shadow_floats, 2u, &kv_shadow_floats) != 0 ||
        pocket_mul(kv_shadow_floats, attn_dim, &kv_shadow_floats) != 0) {
        pocket_error(error, capacity, "pocket: CUDA scratch size overflow");
        free(scratch);
        return -1;
    }

    /* Reserve the cross-request resident workspace up front.  If the optional
     * CUDA buffers cannot be created, the already-correct CPU batch remains
     * available; no decode call allocates a smaller emergency workspace. */
    if (pocket_cuda_resident_requested(state)) {
        size_t rows_hidden = 0, rows_qkv = 0, rows_attn = 0, rows_ffn = 0;
        if (pocket_mul(batch, cfg->hidden_dim, &rows_hidden) == 0 &&
            pocket_mul(batch, qkv_width, &rows_qkv) == 0 &&
            pocket_mul(batch, attn_dim, &rows_attn) == 0 &&
            pocket_mul(batch, cfg->ffn_dim, &rows_ffn) == 0) {
            scratch->cuda_host_input = NULL;
            scratch->cuda_host_output = NULL;
            scratch->cuda_host_kv = NULL;
            scratch->cuda_kv_shadow = NULL;
            char ignored[256];
            int device_failed =
                mynah_backend_host_alloc(state->backend, rows_hidden,
                                         &scratch->cuda_host_input, ignored,
                                         sizeof(ignored)) != 0;
            if (!device_failed) {
                device_failed =
                    mynah_backend_host_alloc(state->backend, rows_hidden,
                                             &scratch->cuda_host_output, ignored,
                                             sizeof(ignored)) != 0;
            }
            if (!device_failed) {
                device_failed =
                    mynah_backend_host_alloc(state->backend, kv_shadow_floats,
                                             &scratch->cuda_host_kv, ignored,
                                             sizeof(ignored)) != 0;
            }
#define POCKET_CUDA_SCRATCH_ALLOC(field, count)                                \
            do {                                                                \
                if (!device_failed &&                                          \
                    mynah_backend_dev_alloc(state->backend, (count),           \
                                            &(field), ignored, sizeof(ignored)) != 0) \
                    device_failed = 1;                                          \
            } while (0)
            POCKET_CUDA_SCRATCH_ALLOC(scratch->cuda_x, rows_hidden);
            POCKET_CUDA_SCRATCH_ALLOC(scratch->cuda_norm, rows_hidden);
            POCKET_CUDA_SCRATCH_ALLOC(scratch->cuda_qkv, rows_qkv);
            POCKET_CUDA_SCRATCH_ALLOC(scratch->cuda_attn, rows_attn);
            POCKET_CUDA_SCRATCH_ALLOC(scratch->cuda_proj, rows_hidden);
            POCKET_CUDA_SCRATCH_ALLOC(scratch->cuda_ffn, rows_ffn);
            POCKET_CUDA_SCRATCH_ALLOC(scratch->cuda_kv_shadow, kv_shadow_floats);
#undef POCKET_CUDA_SCRATCH_ALLOC
            if (!device_failed &&
                kv_metadata > SIZE_MAX / sizeof(*scratch->cuda_kcache)) {
                device_failed = 1;
            }
            if (!device_failed) {
                device_failed =
                    pocket_host_alloc_bytes(
                        state->backend,
                        kv_metadata * sizeof(*scratch->cuda_kcache),
                        (void **)&scratch->cuda_kcache, ignored,
                        sizeof(ignored)) != 0 ||
                    pocket_host_alloc_bytes(
                        state->backend,
                        kv_metadata * sizeof(*scratch->cuda_vcache),
                        (void **)&scratch->cuda_vcache, ignored,
                        sizeof(ignored)) != 0 ||
                    pocket_host_alloc_bytes(
                        state->backend,
                        batch * sizeof(*scratch->cuda_positions),
                        (void **)&scratch->cuda_positions, ignored,
                        sizeof(ignored)) != 0 ||
                    pocket_host_alloc_bytes(
                        state->backend,
                        batch * sizeof(*scratch->cuda_cache_strides),
                        (void **)&scratch->cuda_cache_strides, ignored,
                        sizeof(ignored)) != 0;
            }
            int flow_failed = 0;
            size_t flow_freq_width = 0u;
            size_t flow_rows = 0u;
            size_t flow_rows_latent = 0u;
            size_t flow_three_hidden = 0u;
            size_t flow_two_hidden = 0u;
            size_t flow_mod_rows = 0u;
            size_t flow_final_mod_rows = 0u;
            size_t flow_time_rows = 0u;
            size_t flow_time_hidden = 0u;
            if (!device_failed && pocket_cuda_flow_requested(state)) {
                if (pocket_mul(cfg->flow_freqs, 2u, &flow_freq_width) != 0 ||
                    pocket_mul(batch, cfg->flow_dim, &flow_rows) != 0 ||
                    pocket_mul(batch, cfg->latent_dim, &flow_rows_latent) != 0 ||
                    pocket_mul(cfg->flow_dim, 3u, &flow_three_hidden) != 0 ||
                    pocket_mul(cfg->flow_dim, 2u, &flow_two_hidden) != 0 ||
                    pocket_mul(batch, flow_three_hidden, &flow_mod_rows) != 0 ||
                    pocket_mul(batch, flow_two_hidden, &flow_final_mod_rows) != 0 ||
                    pocket_mul(cfg->flow_time_conds, flow_freq_width,
                               &flow_time_rows) != 0 ||
                    pocket_mul(cfg->flow_time_conds, cfg->flow_dim,
                               &flow_time_hidden) != 0 ||
                    cfg->flow_time_conds == 0u || cfg->flow_dim < 2u ||
                    pocket_mul(flow_rows, sizeof(float), &flow_rows) != 0 ||
                    pocket_mul(flow_rows_latent, sizeof(float), &flow_rows_latent) != 0 ||
                    pocket_mul(flow_mod_rows, sizeof(float), &flow_mod_rows) != 0 ||
                    pocket_mul(flow_final_mod_rows, sizeof(float),
                               &flow_final_mod_rows) != 0 ||
                    pocket_mul(flow_time_rows, sizeof(float), &flow_time_rows) != 0 ||
                    pocket_mul(flow_time_hidden, sizeof(float),
                               &flow_time_hidden) != 0) {
                    flow_failed = 1;
                }
                /* `flow_rows` etc. are byte counts after the checked
                 * multiplications above; keep the element counts separate for
                 * the device allocator. */
                size_t flow_rows_floats = 0u;
                size_t flow_rows_latent_floats = 0u;
                size_t flow_mod_rows_floats = 0u;
                size_t flow_final_mod_rows_floats = 0u;
                size_t flow_time_rows_floats = 0u;
                size_t flow_time_hidden_floats = 0u;
                if (!flow_failed) {
                    flow_rows_floats = flow_rows / sizeof(float);
                    flow_rows_latent_floats = flow_rows_latent / sizeof(float);
                    flow_mod_rows_floats = flow_mod_rows / sizeof(float);
                    flow_final_mod_rows_floats = flow_final_mod_rows / sizeof(float);
                    flow_time_rows_floats = flow_time_rows / sizeof(float);
                    flow_time_hidden_floats = flow_time_hidden / sizeof(float);
                    if (mynah_backend_host_alloc(state->backend, rows_hidden,
                                                 &scratch->cuda_flow_host_cond,
                                                 ignored, sizeof(ignored)) != 0 ||
                        mynah_backend_host_alloc(state->backend,
                                                 flow_rows_latent_floats,
                                                 &scratch->cuda_flow_host_noise,
                                                 ignored, sizeof(ignored)) != 0 ||
                        mynah_backend_host_alloc(state->backend, flow_time_rows_floats,
                                                 &scratch->cuda_flow_host_time,
                                                 ignored, sizeof(ignored)) != 0 ||
                        mynah_backend_host_alloc(state->backend,
                                                 flow_rows_latent_floats,
                                                 &scratch->cuda_flow_host_output,
                                                 ignored, sizeof(ignored)) != 0 ||
                        pocket_cuda_flow_time_features(
                            cfg, scratch->cuda_flow_host_time) != 0) {
                        flow_failed = 1;
                    }
                }
#define POCKET_CUDA_FLOW_ALLOC(field, count)                                  \
                do {                                                           \
                    if (!flow_failed &&                                       \
                        mynah_backend_dev_alloc(state->backend, (count),      \
                                                &(field), ignored,              \
                                                sizeof(ignored)) != 0)          \
                        flow_failed = 1;                                       \
                } while (0)
                POCKET_CUDA_FLOW_ALLOC(scratch->cuda_flow_cond, rows_hidden);
                POCKET_CUDA_FLOW_ALLOC(scratch->cuda_flow_noise,
                                       flow_rows_latent_floats);
                POCKET_CUDA_FLOW_ALLOC(scratch->cuda_flow_time_embed,
                                       flow_time_rows_floats);
                POCKET_CUDA_FLOW_ALLOC(scratch->cuda_flow_output,
                                       flow_rows_latent_floats);
                POCKET_CUDA_FLOW_ALLOC(scratch->cuda_flow_y, flow_rows_floats);
                POCKET_CUDA_FLOW_ALLOC(scratch->cuda_flow_silu, flow_rows_floats);
                POCKET_CUDA_FLOW_ALLOC(scratch->cuda_flow_x, flow_rows_floats);
                POCKET_CUDA_FLOW_ALLOC(scratch->cuda_flow_norm, flow_rows_floats);
                POCKET_CUDA_FLOW_ALLOC(scratch->cuda_flow_hidden, flow_rows_floats);
                POCKET_CUDA_FLOW_ALLOC(scratch->cuda_flow_scratch, flow_rows_floats);
                POCKET_CUDA_FLOW_ALLOC(scratch->cuda_flow_mod, flow_mod_rows_floats);
                POCKET_CUDA_FLOW_ALLOC(scratch->cuda_flow_final_mod,
                                       flow_final_mod_rows_floats);
                POCKET_CUDA_FLOW_ALLOC(scratch->cuda_flow_time_hidden,
                                       flow_time_hidden_floats);
                POCKET_CUDA_FLOW_ALLOC(scratch->cuda_flow_time_output,
                                       flow_time_hidden_floats);
                POCKET_CUDA_FLOW_ALLOC(scratch->cuda_flow_time_sum,
                                       cfg->flow_dim);
#undef POCKET_CUDA_FLOW_ALLOC
                if (flow_failed) pocket_cuda_flow_release(scratch);
                else {
                    scratch->cuda_flow_batch_capacity = batch;
                    scratch->cuda_flow_enabled = 1;
                    scratch->cuda_flow_graph_enabled =
                        getenv("MYNAH_CUDA_GRAPHS") == NULL ||
                        strcmp(getenv("MYNAH_CUDA_GRAPHS"), "0") != 0;
                    scratch->cuda_flow_graph_ready = 0;
                    scratch->cuda_flow_time_mlp_in =
                        (mynah_backend_flow_linear *)calloc(
                            cfg->flow_time_conds,
                            sizeof(*scratch->cuda_flow_time_mlp_in));
                    scratch->cuda_flow_time_mlp_out =
                        (mynah_backend_flow_linear *)calloc(
                            cfg->flow_time_conds,
                            sizeof(*scratch->cuda_flow_time_mlp_out));
                    scratch->cuda_flow_time_alpha =
                        (const float **)calloc(cfg->flow_time_conds,
                                               sizeof(*scratch->cuda_flow_time_alpha));
                    scratch->cuda_flow_blocks =
                        (mynah_backend_flow_block *)calloc(
                            cfg->flow_depth, sizeof(*scratch->cuda_flow_blocks));
                    if (scratch->cuda_flow_time_mlp_in == NULL ||
                        scratch->cuda_flow_time_mlp_out == NULL ||
                        scratch->cuda_flow_time_alpha == NULL ||
                        scratch->cuda_flow_blocks == NULL) {
                        pocket_cuda_flow_release(scratch);
                    } else {
                        for (size_t t = 0; t < cfg->flow_time_conds; ++t) {
                            const mynah_flow_time_embed_weights *src =
                                &state->time_embed[t];
                            scratch->cuda_flow_time_mlp_in[t].weight =
                                src->mlp_in.weight;
                            scratch->cuda_flow_time_mlp_in[t].bias =
                                src->mlp_in.bias;
                            scratch->cuda_flow_time_mlp_out[t].weight =
                                src->mlp_out.weight;
                            scratch->cuda_flow_time_mlp_out[t].bias =
                                src->mlp_out.bias;
                            scratch->cuda_flow_time_alpha[t] = src->alpha;
                        }
                        for (size_t b = 0; b < cfg->flow_depth; ++b) {
                            const mynah_flow_res_block_weights *src =
                                &state->res_blocks[b];
                            mynah_backend_flow_block *dst =
                                &scratch->cuda_flow_blocks[b];
                            dst->in_ln_weight = src->in_ln_weight;
                            dst->in_ln_bias = src->in_ln_bias;
                            dst->adaln.weight = src->adaln.weight;
                            dst->adaln.bias = src->adaln.bias;
                            dst->mlp_in.weight = src->mlp_in.weight;
                            dst->mlp_in.bias = src->mlp_in.bias;
                            dst->mlp_out.weight = src->mlp_out.weight;
                            dst->mlp_out.bias = src->mlp_out.bias;
                        }
                        scratch->cuda_flow_cond_embed.weight =
                            state->flow.cond_embed.weight;
                        scratch->cuda_flow_cond_embed.bias =
                            state->flow.cond_embed.bias;
                        scratch->cuda_flow_input_proj.weight =
                            state->flow.input_proj.weight;
                        scratch->cuda_flow_input_proj.bias =
                            state->flow.input_proj.bias;
                        scratch->cuda_flow_final_adaln.weight =
                            state->flow.final_adaln.weight;
                        scratch->cuda_flow_final_adaln.bias =
                            state->flow.final_adaln.bias;
                        scratch->cuda_flow_final_linear.weight =
                            state->flow.final_linear.weight;
                        scratch->cuda_flow_final_linear.bias =
                            state->flow.final_linear.bias;
                    }
                }
            }
            if (device_failed) {
                mynah_backend_dev_free(state->backend, scratch->cuda_x);
                mynah_backend_dev_free(state->backend, scratch->cuda_norm);
                mynah_backend_dev_free(state->backend, scratch->cuda_qkv);
                mynah_backend_dev_free(state->backend, scratch->cuda_attn);
                mynah_backend_dev_free(state->backend, scratch->cuda_proj);
                mynah_backend_dev_free(state->backend, scratch->cuda_ffn);
                mynah_backend_dev_free(state->backend, scratch->cuda_kv_shadow);
                scratch->cuda_x = NULL;
                scratch->cuda_norm = NULL;
                scratch->cuda_qkv = NULL;
                scratch->cuda_attn = NULL;
                scratch->cuda_proj = NULL;
                scratch->cuda_ffn = NULL;
                scratch->cuda_kv_shadow = NULL;
                mynah_backend_host_free(state->backend, scratch->cuda_host_input);
                mynah_backend_host_free(state->backend, scratch->cuda_host_output);
                mynah_backend_host_free(state->backend, scratch->cuda_host_kv);
                scratch->cuda_host_input = NULL;
                scratch->cuda_host_output = NULL;
                scratch->cuda_host_kv = NULL;
                pocket_host_free_bytes(state->backend, scratch->cuda_kcache);
                pocket_host_free_bytes(state->backend, scratch->cuda_vcache);
                pocket_host_free_bytes(state->backend, scratch->cuda_positions);
                pocket_host_free_bytes(state->backend, scratch->cuda_cache_strides);
                scratch->cuda_kcache = NULL;
                scratch->cuda_vcache = NULL;
                scratch->cuda_positions = NULL;
                scratch->cuda_cache_strides = NULL;
            } else {
                scratch->cuda_batch_capacity = batch;
                scratch->cuda_batch_enabled = 1;
                scratch->cuda_graph_enabled =
                    getenv("MYNAH_CUDA_GRAPHS") == NULL ||
                    strcmp(getenv("MYNAH_CUDA_GRAPHS"), "0") != 0;
                scratch->cuda_graph_ready = 0;
                scratch->cuda_kv_shadow_floats = kv_shadow_floats;
            }
        }
    }

    size_t backbone_k = cfg->hidden_dim;
    if (attn_dim > backbone_k) backbone_k = attn_dim;
    if (cfg->ffn_dim > backbone_k) backbone_k = cfg->ffn_dim;

    mynah_transformer_ar_config backbone;
    mynah_transformer_ar_config_defaults(&backbone);
    backbone.d_model = cfg->hidden_dim;
    backbone.num_heads = cfg->heads;
    backbone.head_dim = cfg->head_dim;
    backbone.num_layers = cfg->layers;
    backbone.ffn_dim = cfg->ffn_dim;
    /* A capacity, not a shape: each request brings its own, and the batch
     * scratch holds no KV. */
    backbone.max_seq_len = 0u;
    backbone.context = 0u;
    backbone.layernorm_eps = cfg->layernorm_eps;

    scratch->states =
        (mynah_transformer_ar_state **)calloc(batch, sizeof(*scratch->states));
    scratch->inputs = (const float **)calloc(batch, sizeof(*scratch->inputs));
    scratch->outputs = (float **)calloc(batch, sizeof(*scratch->outputs));
    if (scratch->states == NULL || scratch->inputs == NULL ||
        scratch->outputs == NULL ||
        pocket_tar_call_init(&scratch->backbone_call, &state->backbone_hook, batch,
                             backbone_k, error, capacity) != 0) {
        if (scratch->states != NULL && scratch->inputs != NULL &&
            scratch->outputs != NULL) {
            /* pocket_call_init already reported. */
        } else {
            pocket_error(error, capacity, "out of memory creating pocket scratch");
        }
        pocket_scratch_free(scratch);
        return -1;
    }
    pocket_bind_hooks(&scratch->backbone_w, &state->backbone,
                      &scratch->backbone_call);
    scratch->backbone_batch =
        mynah_transformer_ar_batch_new(&backbone, batch, error, capacity);
    if (scratch->backbone_batch == NULL) {
        pocket_scratch_free(scratch);
        return -1;
    }

    size_t flow_k = cfg->flow_dim;
    if (cfg->hidden_dim > flow_k) flow_k = cfg->hidden_dim;
    mynah_flow_head_config flow;
    mynah_flow_head_config_defaults(&flow);
    flow.latent_dim = cfg->latent_dim;
    flow.cond_dim = cfg->hidden_dim;
    flow.hidden_dim = cfg->flow_dim;
    flow.depth = cfg->flow_depth;
    flow.num_time_conds = cfg->flow_time_conds;
    flow.freq_embed_dim = 2u * cfg->flow_freqs;
    flow.layernorm_eps = cfg->flow_layernorm_eps;

    scratch->flow_heads =
        (mynah_flow_head **)calloc(batch, sizeof(*scratch->flow_heads));
    scratch->flow_cond = (const float **)calloc(batch, sizeof(*scratch->flow_cond));
    scratch->flow_noise =
        (const float **)calloc(batch, sizeof(*scratch->flow_noise));
    scratch->flow_out = (float **)calloc(batch, sizeof(*scratch->flow_out));
    if (scratch->flow_heads == NULL || scratch->flow_cond == NULL ||
        scratch->flow_noise == NULL || scratch->flow_out == NULL) {
        pocket_error(error, capacity, "out of memory creating pocket scratch");
        pocket_scratch_free(scratch);
        return -1;
    }
    if (pocket_flow_call_init(&scratch->flow_call, &state->flow_hook, batch, flow_k,
                              error, capacity) != 0) {
        pocket_scratch_free(scratch);
        return -1;
    }
    pocket_bind_flow_hooks(&scratch->flow_w, &state->flow, &scratch->flow_call);
    scratch->flow_batch = mynah_flow_head_batch_new(&flow, batch, error, capacity);
    if (scratch->flow_batch == NULL) {
        pocket_scratch_free(scratch);
        return -1;
    }
    *out = scratch;
    return 0;
}

/* ----------------------------------------------------------------- vtable */

static const mynah_tts_engine pocket_engine = {
    POCKET_ENGINE_NAME,
    pocket_model_init,
    pocket_model_free,
    pocket_caps,
    pocket_ctx_new,
    pocket_prepare,
    pocket_reset,
    pocket_ctx_free,
    pocket_step_batch,
    pocket_emit_batch,
    pocket_frame_count,
    pocket_truncate,
    pocket_decode_audio,
    pocket_scratch_new,
    pocket_scratch_free,
    NULL,                      /* debug_dump: the dump is written in ctx_free */
    pocket_decode_audio_batch, /* APPENDED, never inserted (tts_engine.h) */
    pocket_prepare_slice,      /* APPENDED */
};

const mynah_tts_engine *mynah_engine_pocket(void) { return &pocket_engine; }

/* ------------------------------------------------------------- accessors */

size_t mynah_engine_pocket_max_tokens_per_chunk(const mynah_engine_state *state) {
    return (state == NULL) ? 0u : state->cfg.max_tokens_per_chunk;
}

size_t mynah_engine_pocket_voice_count(const mynah_engine_state *state) {
    return (state == NULL) ? 0u : state->voice_count;
}

const char *mynah_engine_pocket_voice_name(const mynah_engine_state *state,
                                           size_t index) {
    if (state == NULL || index >= state->voice_count) return NULL;
    return state->voices[index].name;
}

int mynah_engine_pocket_voice_index(const mynah_engine_state *state,
                                    const char *name) {
    if (state == NULL || name == NULL) return -1;
    for (size_t i = 0; i < state->voice_count; ++i) {
        if (strcmp(state->voices[i].name, name) == 0) return (int)i;
    }
    return -1;
}

int mynah_engine_pocket_tokenize(const mynah_engine_state *state, const char *text,
                                 size_t text_length, int **out_ids,
                                 size_t *out_count, char *error, size_t capacity) {
    if (state == NULL || state->tokenizer == NULL || out_ids == NULL ||
        out_count == NULL) {
        pocket_error(error, capacity, "pocket: no tokenizer to encode with");
        return -1;
    }
    return mynah_sp_encode(state->tokenizer, text, text_length, out_ids, out_count,
                           error, capacity);
}

/* ------------------------------------------------- long-form text (E5-5)
 *
 * HOW THE WINDOW GROWS, AND WHY IT IS A CEILING RATHER THAN A REALLOC.
 *
 * Three things in a context are sized from the text: `text_ids`, `text_embed`,
 * and the backbone KV's `max_seq_len` (voice + text + max_steps + 1).  Nothing
 * else is -- `latents`, the codec KV and the step budget are all functions of
 * `max_steps` alone -- so the growth problem is narrower than it looks, but it
 * does include a contiguous KV cache that `transformer_ar` allocates once.
 *
 * Three ways to grow it, weighed on the case that decides it: the growth fails
 * on a stream that has already emitted audio.
 *
 *  1. REALLOCATE AND COPY.  The KV is one contiguous block per layer, so this
 *     means a second allocation of the whole cache while the first is still
 *     live -- peak footprint doubles at the exact moment memory is tight.  If
 *     it fails, the request is mid-flight: the caller has PCM in its socket and
 *     no way to un-send it.  There is no answer to that failure; it is not a
 *     503, it is a truncated stream that looked healthy a millisecond earlier.
 *  2. A LINKED RUN OF BLOCKS.  Survives the failure better -- only the new
 *     block is lost -- but a chunked cache is a change to `transformer_ar`'s
 *     attention inner loop, which every engine and every ISA kernel depends on,
 *     to buy an allocation shape nothing else in this runtime wants.  Rejected
 *     on blast radius, not on the idea.
 *  3. A CEILING DECLARED AT ADMISSION.  `reserve_text` is legal only before
 *     `prepare`: it allocates everything the request may ever need while the
 *     request has done nothing, has emitted nothing, and can still be refused.
 *     A failure there IS a 503.  Afterwards `append_text` allocates NOTHING --
 *     it copies into space that was paid for, and a caller who asks for more
 *     than it reserved gets a refusal that leaves the stream exactly where it
 *     was, with both numbers in the message.
 *
 * (3), because it is the only one of the three whose append path contains no
 * allocation at all, which is what turns "the growth failed" from an outcome
 * into a case that cannot arise.  It costs the operator a number they must
 * choose, and the KV is ~48 KB per position per request for this pack, so
 * choosing it badly is expensive -- which is an argument for making it a
 * per-request field rather than a process-wide one, and it is one of the two
 * things this needs from the seam (see engine_pocket.h).
 *
 * Growth is committed all-or-nothing: every new buffer is allocated before any
 * old one is released, so a failed `reserve_text` leaves the context exactly at
 * the capacity it already had.
 */
int mynah_engine_pocket_reserve_text(mynah_engine_ctx *ctx, size_t total_tokens,
                                     char *error, size_t capacity) {
    if (ctx == NULL) {
        pocket_error(error, capacity, "pocket: null context reserving text");
        return -1;
    }
    if (ctx->prepared) {
        /* The KV would have to be rebuilt under a prefix that is already in it.
         * Admission is the only window where growing is free of consequence,
         * and this is the check that keeps it that way. */
        pocket_error(error, capacity,
                     "pocket: reserve_text after prepare; the text ceiling is "
                     "declared at admission");
        return -1;
    }
    /* Declaring a ceiling is what opens the context, whether or not it grows:
     * a caller that reserves exactly what it admitted still intends to push. */
    ctx->text_open = 1;
    if (total_tokens <= ctx->text_capacity) return 0;

    const pocket_config *cfg = &ctx->state->cfg;
    size_t backbone_capacity = 0, text_floats = 0, codec_frames = 0;
    if (pocket_add(ctx->max_steps, 1u, &codec_frames) != 0 ||
        pocket_add(ctx->voice_positions, total_tokens, &backbone_capacity) != 0 ||
        pocket_add(backbone_capacity, codec_frames, &backbone_capacity) != 0 ||
        pocket_mul(total_tokens, cfg->hidden_dim, &text_floats) != 0) {
        pocket_error(error, capacity,
                     "pocket: a text ceiling of %zu tokens overflows", total_tokens);
        return -1;
    }
    /* Everything is allocated before anything is released, so the failure below
     * leaves the context exactly at the capacity it already had. */
    int *ids = (int *)calloc(total_tokens, sizeof(*ids));
    if (ids == NULL) {
        pocket_error(error, capacity,
                     "pocket: out of memory reserving %zu text tokens",
                     total_tokens);
        return -1;
    }
    /* These two write their own message on failure. */
    float *embed = mynah_alloc_floats(text_floats, error, capacity);
    mynah_transformer_ar_config backbone;
    pocket_backbone_config(ctx, backbone_capacity, &backbone);
    mynah_transformer_ar_state *grown =
        embed == NULL ? NULL
                      : mynah_transformer_ar_state_new(&backbone, error, capacity);
    if (embed == NULL || grown == NULL ||
        mynah_transformer_ar_check_weights(grown, &ctx->backbone_w, error,
                                           capacity) != 0) {
        free(ids);
        free(embed);
        mynah_transformer_ar_state_free(grown);
        return -1; /* the context is untouched, at the capacity it already had */
    }
    memcpy(ids, ctx->text_ids, ctx->text_length * sizeof(*ids));
    free(ctx->text_ids);
    ctx->text_ids = ids;
    free(ctx->text_embed);
    ctx->text_embed = embed;
    pocket_cuda_backbone_release(ctx);
    mynah_transformer_ar_state_free(ctx->backbone);
    ctx->backbone = grown;
    ctx->text_capacity = total_tokens;
    (void)pocket_cuda_backbone_alloc(ctx);
    return 0;
}

/*
 * Append text to a context that is prepared but has not stepped.
 *
 * THE ONE REFUSAL THAT IS NOT A LIMITATION.  `ctx->step != 0` is refused, and
 * it is refused because the alternative cannot exist rather than because it was
 * not built.  A one-shot run puts the whole text in the KV before position
 * zero of the audio, so every frame is conditioned on all of it.  A context
 * that has already generated k frames and then receives more text has the
 * layout [voice][text A][step 0..k-1][text B][step k..]; the first k frames
 * were produced by a model that had never seen text B and are already in the
 * caller's socket.  No ordering of a causal cache makes that equal to the
 * one-shot, and regenerating the frames is not open either -- they have been
 * sent.  So appending mid-generation is not byte-identical to anything, which
 * is the property this whole item is gated on, and weakening the gate to admit
 * it would be the same mistake as weakening it to admit N sequential requests.
 *
 * What this DOES buy is real and is what the driver half wants: text may be
 * prefilled as it arrives instead of after it has all arrived, so the prefill
 * overlaps the text stream rather than following it.
 */
int mynah_engine_pocket_append_text(mynah_engine_ctx *ctx, const int *text_ids,
                                    size_t count, char *error, size_t capacity) {
    if (ctx == NULL || (count > 0u && text_ids == NULL)) {
        pocket_error(error, capacity, "pocket: null argument appending text");
        return -1;
    }
    if (!ctx->prepared) {
        pocket_error(error, capacity, "pocket: append_text before prepare");
        return -1;
    }
    /* Ordered before the `text_open` check on purpose.  Both refuse a context
     * that has generated -- `step_batch` will not step one whose text is still
     * open, so a stepped context is always a sealed one -- but only this branch
     * says WHY, and the reason is the whole argument for the refusal rather
     * than an implementation detail.  A caller told "this context was admitted
     * as one complete text" would go looking for a reserve it already made. */
    if (ctx->step != 0u) {
        pocket_error(error, capacity,
                     "pocket: append_text after %zu steps; text appended to a "
                     "context that has already generated cannot be identical to "
                     "the same text prefilled whole, and the frames it already "
                     "emitted cannot be recalled", ctx->step);
        return -1;
    }
    if (!ctx->text_open) {
        pocket_error(error, capacity,
                     "pocket: this context was admitted as one complete text; "
                     "call reserve_text before prepare to open it");
        return -1;
    }
    if (ctx->broken) {
        pocket_error(error, capacity, "pocket: append_text on a failed request");
        return -1;
    }
    if (count == 0u) return 0;
    if (count > ctx->text_capacity - ctx->text_length) {
        /* The refusal the ceiling exists to produce: nothing has been copied,
         * nothing has been prefilled, and the context is still exactly where it
         * was, so the caller may seal and generate what it has. */
        pocket_error(error, capacity,
                     "pocket: %zu more tokens would pass the text ceiling "
                     "(%zu reserved, %zu used); raise it at admission",
                     count, ctx->text_capacity, ctx->text_length);
        return -1;
    }
    const mynah_engine_state *state = ctx->state;
    const pocket_config *cfg = &state->cfg;
    /* Validated before anything is copied, for the same reason `step_batch`
     * validates before it advances: a bad id must refuse, not half-append. */
    for (size_t i = 0; i < count; ++i) {
        const int id = text_ids[i];
        if (id < 0 || (size_t)id >= cfg->vocab_size) {
            pocket_error(error, capacity,
                         "text id %d at append position %zu is outside [0, %zu)",
                         id, i, cfg->vocab_size);
            return -1;
        }
    }
    for (size_t i = 0; i < count; ++i) {
        const size_t at = ctx->text_length + i;
        ctx->text_ids[at] = text_ids[i];
        memcpy(ctx->text_embed + at * cfg->hidden_dim,
               state->embed_table + (size_t)text_ids[i] * cfg->hidden_dim,
               cfg->hidden_dim * sizeof(float));
    }
    ctx->text_length += count;
    if (pocket_text_flush(ctx, 0, error, capacity) != 0) {
        /* The KV now holds an unknown number of the new positions.  Nothing has
         * been emitted -- `step` is still 0 -- but this context can no longer be
         * reasoned about, so it is retired rather than stepped. */
        ctx->broken = 1;
        return -1;
    }
    return 0;
}

/* Close the text: prefill the remainder and allow stepping.  Idempotent, and a
 * no-op on a context that was never opened. */
int mynah_engine_pocket_seal_text(mynah_engine_ctx *ctx, char *error,
                                  size_t capacity) {
    if (ctx == NULL) {
        pocket_error(error, capacity, "pocket: null context sealing text");
        return -1;
    }
    if (!ctx->prepared) {
        pocket_error(error, capacity, "pocket: seal_text before prepare");
        return -1;
    }
    if (!ctx->text_open) return 0;
    if (ctx->step != 0u) {
        pocket_error(error, capacity, "pocket: seal_text after %zu steps",
                     ctx->step);
        return -1;
    }
    if (pocket_text_flush(ctx, 1, error, capacity) != 0) {
        ctx->broken = 1;
        return -1;
    }
    ctx->text_open = 0;
    /* E2-5 again, and this is where it bites hardest: the accumulated length is
     * not known at admission, so a caller that pushes its way past the limit
     * would never have seen the warning `ctx_new` prints.  Long-form does not
     * make the no-EOS failure more or less likely at a given token count -- the
     * KV it builds is the one the one-shot builds -- but it is the feature that
     * makes reaching that count ordinary rather than exotic. */
    {
        const size_t limit = ctx->state->cfg.max_tokens_per_chunk;
        if (limit != 0u && ctx->text_length > limit && !ctx->state->chunk_warned) {
            ctx->state->chunk_warned = 1;
            fprintf(stderr,
                    "mynah-tts: %zu text tokens pushed past max_tokens_per_chunk "
                    "%zu; this engine does not split text, and past roughly three "
                    "times the limit the model may stop emitting EOS and run to "
                    "the step budget (E2-5)\n", ctx->text_length, limit);
        }
    }
    return 0;
}

size_t mynah_engine_pocket_text_length(const mynah_engine_ctx *ctx) {
    return ctx == NULL ? 0u : ctx->text_length;
}

size_t mynah_engine_pocket_text_capacity(const mynah_engine_ctx *ctx) {
    return ctx == NULL ? 0u : ctx->text_capacity;
}

int mynah_engine_pocket_set_frames_after_eos(mynah_engine_ctx *ctx, size_t frames) {
    if (ctx == NULL || frames > ctx->max_steps) return -1;
    ctx->frames_after_eos = frames;
    return 0;
}

size_t mynah_engine_pocket_frames_after_eos(const mynah_engine_ctx *ctx) {
    return (ctx == NULL) ? 0u : ctx->frames_after_eos;
}

int mynah_engine_pocket_set_noise(mynah_engine_ctx *ctx, mynah_pocket_noise_fn fn,
                                  void *user_data) {
    if (ctx == NULL) return -1;
    ctx->noise_fn = fn;
    ctx->noise_user = user_data;
    return 0;
}

const float *mynah_engine_pocket_hidden(const mynah_engine_ctx *ctx,
                                        size_t *out_count) {
    if (ctx == NULL) return NULL;
    if (out_count != NULL) *out_count = ctx->state->cfg.hidden_dim;
    return ctx->hidden;
}

float mynah_engine_pocket_eos_logit(const mynah_engine_ctx *ctx) {
    return (ctx == NULL) ? 0.0f : ctx->eos_logit;
}

const float *mynah_engine_pocket_latent(const mynah_engine_ctx *ctx, size_t index,
                                        size_t *out_count) {
    if (ctx == NULL || index >= ctx->frames) return NULL;
    if (out_count != NULL) *out_count = ctx->state->cfg.latent_dim;
    return ctx->latents + index * ctx->state->cfg.latent_dim;
}

/* ------------------------------------------------- the batching self-check
 *
 * Two properties of `tts_engine.h` that only real weights can test, and that
 * until now nothing did: `step_batch` is atomic over the batch (E8-6), and
 * `decode_audio_batch` is bit-identical per context whoever shared the call
 * (E8-4).  Both had been checked only against the synthetic engine in
 * `tests/test_driver.c`, which is exactly the wrong place to check them: the
 * synthetic engine's step cannot fail halfway through a real graph and its
 * codec carries no state, so neither property was ever exercised where it can
 * actually break.
 *
 * Needs a pack, so it is not part of `--self-test`; it takes an opened model
 * and does the rest itself.
 */

typedef struct {
    const char *text;
    unsigned speaker;
    uint64_t seed;
} pocket_check_case;

#define POCKET_CHECK_MAX 16u

/* Two sets of contexts built from the SAME cases.  Everything in this engine is
 * a deterministic function of (weights, text, voice, seed), so set A and set B
 * are two runs of one experiment and any difference between them is the thing
 * being tested. */
typedef struct {
    mynah_engine_ctx *ctx[POCKET_CHECK_MAX];
    size_t count;
} pocket_check_set;

static void pocket_check_set_free(pocket_check_set *set) {
    for (size_t i = 0; i < set->count; ++i) pocket_ctx_free(set->ctx[i]);
    set->count = 0;
}

static int pocket_check_set_new(mynah_engine_state *state,
                                const mynah_tts_model *model,
                                const pocket_check_case *cases, size_t count,
                                size_t max_steps, pocket_check_set *set,
                                char *error, size_t capacity) {
    memset(set, 0, sizeof(*set));
    for (size_t i = 0; i < count; ++i) {
        int *ids = NULL;
        size_t n_ids = 0;
        if (mynah_engine_pocket_tokenize(state, cases[i].text,
                                         strlen(cases[i].text), &ids, &n_ids, error,
                                         capacity) != 0) {
            pocket_check_set_free(set);
            return -1;
        }
        mynah_tts_request request;
        memset(&request, 0, sizeof(request));
        request.text_ids = ids;
        request.text_length = n_ids;
        /* The atomicity/gang checks need a fixed row count even when a
         * checkpoint was converted with one voice only (for example the
         * official 24L verification pack).  Reuse available voices rather
         * than silently shrinking the batch to voice_count rows. */
        request.speaker = state->voice_count != 0u
                              ? cases[i].speaker % state->voice_count
                              : 0u;
        request.temperature = -1.0f;
        mynah_engine_ctx *ctx = NULL;
        const int bad = pocket_ctx_new(model, state, &request, max_steps,
                                       cases[i].seed, &ctx, error, capacity) != 0;
        free(ids);
        if (bad) {
            pocket_check_set_free(set);
            return -1;
        }
        set->ctx[set->count++] = ctx;
        if (pocket_prepare(ctx, error, capacity) != 0) {
            pocket_check_set_free(set);
            return -1;
        }
        /* Keep the rows available while the self-check deliberately exercises
         * a refusal in the middle of a batch.  EOS timing is model-dependent:
         * the 6L and 24L checkpoints can retire almost every row during the
         * three warm-up steps, which would make the atomicity test fail before
         * it had a victim and a predecessor to inspect.  This only extends the
         * normal post-EOS tail of these private test contexts; it does not
         * change inference or any production request setting. */
        if (mynah_engine_pocket_set_frames_after_eos(ctx, max_steps) != 0) {
            pocket_error(error, capacity,
                         "self-check: unable to keep request %zu live", i);
            pocket_check_set_free(set);
            return -1;
        }
    }
    return 0;
}

/* The contexts of `set` that are still generating, in order.
 *
 * A set of sixteen real requests does not stay sixteen: they reach EOS at
 * different steps, which is the whole reason the driver has slots.  Everything
 * below therefore steps the LIVE subset, exactly as `step_live()` does. */
static size_t pocket_check_live(const pocket_check_set *set,
                                mynah_engine_ctx **out) {
    size_t live = 0;
    for (size_t i = 0; i < set->count; ++i) {
        mynah_engine_ctx *ctx = set->ctx[i];
        if (ctx->eos || ctx->broken) continue;
        out[live++] = ctx;
    }
    return live;
}

/* One AR step for whoever is still live, through the engine's own entry points.
 * Returns the number of contexts that stepped, or -1. */
static long pocket_check_advance(pocket_check_set *set,
                                 mynah_engine_scratch *scratch, char *error,
                                 size_t capacity) {
    mynah_engine_ctx *live_ctx[POCKET_CHECK_MAX];
    mynah_engine_step_result results[POCKET_CHECK_MAX];
    const size_t live = pocket_check_live(set, live_ctx);
    if (live == 0u) return 0;
    mynah_engine_ctx *const *ctxs = live_ctx;
    if (pocket_step_batch(ctxs, live, scratch, error, capacity) != 0) return -1;
    if (pocket_emit_batch(ctxs, live, results, scratch, error, capacity) != 0) {
        return -1;
    }
    for (size_t i = 0; i < live; ++i) {
        if (results[i].failed) {
            pocket_error(error, capacity, "self-check: request %zu failed to emit", i);
            return -1;
        }
    }
    return (long)live;
}

/* E8-6.  A step that refuses must leave every context exactly as it found it.
 *
 * FORCING THE FAILURE, in the two places a step can refuse.  The distinction is
 * not academic: it is the difference between a gate that tests the property and
 * one that only looks like it does.
 *
 *   - `POCKET_INJECT_LATENT_NAN` writes a NaN into the victim's most recent
 *     latent, so `input_linear` produces a non-finite step input.  That is
 *     refused by the PRE-FLIGHT, before any context is touched.  It exercises
 *     layer 2 of the atomicity argument and nothing below it.
 *   - `POCKET_INJECT_KV_NAN` writes a NaN into a position the victim's backbone
 *     already has cached, which makes that row's attention scores non-finite;
 *     `mynah_softmax_f32` refuses them INSIDE the forward, after the rows ahead
 *     of it in the per-row loop have already advanced.  This is the only
 *     failure mode the ROLLBACK exists for, and the latent injection cannot
 *     reach it, because the pre-flight catches that one first.
 *
 * Measured, because a blind gate is worse than no gate.  With the rollback
 * deleted the latent injection still reports PASS, while the KV injection fails
 * on the first per-row width it reaches -- "step atomicity at width 2 (per-row,
 * cached NaN, in-backbone): step_batch is not atomic -- request 0 advanced from
 * 143 to 144 on a call that was refused".  Only one of the two injections is
 * load-bearing, and it is not the obvious one, so both are run.
 *
 * Neither adds a fault-injection hook to the production path: one writes a
 * float a diverged request could have written itself, the other writes one
 * through `_state_kv`, which the header already exposes.  The single corrupted
 * float is saved and restored around the attempt, so what follows is the run
 * that would have happened had the refusal never occurred -- which is the whole
 * question.
 *
 * Run with `scratch == NULL` as well as with a real one, because the two take
 * different paths and it is the NULL one -- the per-row loop -- that is not
 * atomic on its own.  `mynah_transformer_ar_step_batch` validates every row and
 * advances every offset only after the forward has succeeded for all of them,
 * so the batched path was already atomic; the per-row loop is what E8-6 is
 * about.
 */
typedef enum {
    POCKET_INJECT_LATENT_NAN = 0, /* refused by the pre-flight */
    POCKET_INJECT_KV_NAN = 1      /* refused inside the backbone */
} pocket_inject;
static int pocket_check_atomic(mynah_engine_state *state,
                               const mynah_tts_model *model,
                               const pocket_check_case *cases, size_t count,
                               size_t max_steps, mynah_engine_scratch *scratch,
                               pocket_inject mode, const char *what, char *error,
                               size_t capacity) {
    const size_t latent_dim = state->cfg.latent_dim;
    const size_t hidden_dim = state->cfg.hidden_dim;
    const size_t warmup = 3u;
    const size_t victim = count / 2u;

    pocket_check_set a, b;
    if (pocket_check_set_new(state, model, cases, count, max_steps, &a, error,
                             capacity) != 0) {
        return -1;
    }
    if (pocket_check_set_new(state, model, cases, count, max_steps, &b, error,
                             capacity) != 0) {
        pocket_check_set_free(&a);
        return -1;
    }
    int rc = -1;
    for (size_t step = 0; step < warmup; ++step) {
        if (pocket_check_advance(&a, scratch, error, capacity) < 0 ||
            pocket_check_advance(&b, scratch, error, capacity) < 0) {
            goto done;
        }
    }

    /* ---- the refused call, on set B only ---- */
    {
        mynah_engine_ctx *live_ctx[POCKET_CHECK_MAX];
        const size_t live = pocket_check_live(&b, live_ctx);
        if (live < 2u) {
            pocket_error(error, capacity,
                         "%s: only %zu requests still live; the test needs a "
                         "neighbour in front of the victim",
                         what, live);
            goto done;
        }
        /* The victim sits in the middle on purpose: the defect being tested is
         * that contexts BEFORE it in the array advance before it refuses. */
        const size_t v = (victim < live) ? victim : live / 2u;
        const size_t vi = (v == 0u) ? live / 2u : v;
        if (vi == 0u) {
            pocket_error(error, capacity,
                         "%s: the victim landed at slot 0, so nothing steps "
                         "before it and the test would be blind",
                         what);
            goto done;
        }
        mynah_engine_ctx *bad = live_ctx[vi];

        /* Exactly one float is corrupted, and it is put back below. */
        float *poison = NULL;
        if (mode == POCKET_INJECT_LATENT_NAN) {
            poison = bad->latents + (bad->frames - 1u) * latent_dim;
        } else {
            /* The last cached position, which every query attends to whatever
             * `context` is set to -- position 0 would fall outside a sliding
             * window and quietly stop being a fault at all.  K is the first
             * half of the [2][max_seq_len][heads][head_dim] block, and
             * heads*head_dim == hidden_dim is checked at config load. */
            const size_t offset = mynah_transformer_ar_state_offset(bad->backbone);
            float *kv = mynah_transformer_ar_state_kv(bad->backbone, 0u);
            if (kv == NULL || offset == 0u) {
                pocket_error(error, capacity,
                             "%s: no cached position to corrupt", what);
                goto done;
            }
            poison = kv + (offset - 1u) * hidden_dim;
        }
        const float saved = *poison;
        const uint32_t nan_bits = UINT32_C(0x7fc00000);
        memcpy(poison, &nan_bits, sizeof(nan_bits));

        size_t before[POCKET_CHECK_MAX];
        int budget_before[POCKET_CHECK_MAX];
        for (size_t i = 0; i < live; ++i) {
            before[i] = mynah_transformer_ar_state_offset(live_ctx[i]->backbone);
            budget_before[i] = live_ctx[i]->budget_exhausted;
        }
        char ignored[256];
        mynah_engine_ctx *const *ctxs = live_ctx;
        if (pocket_step_batch(ctxs, live, scratch, ignored, sizeof(ignored)) == 0) {
            pocket_error(error, capacity,
                         "%s: a step with a non-finite input was accepted", what);
            goto done;
        }
        for (size_t i = 0; i < live; ++i) {
            const size_t now = mynah_transformer_ar_state_offset(live_ctx[i]->backbone);
            if (now != before[i]) {
                pocket_error(error, capacity,
                             "%s: step_batch is not atomic -- request %zu advanced "
                             "from %zu to %zu on a call that was refused",
                             what, i, before[i], now);
                goto done;
            }
            if (live_ctx[i]->budget_exhausted != budget_before[i] ||
                live_ctx[i]->broken || live_ctx[i]->eos) {
                pocket_error(error, capacity,
                             "%s: a refused step changed request %zu's flags", what, i);
                goto done;
            }
        }
        *poison = saved;
    }

    /* ---- and now the run that should be indistinguishable from A's ---- */
    for (size_t step = 0; step < warmup; ++step) {
        if (pocket_check_advance(&a, scratch, error, capacity) < 0 ||
            pocket_check_advance(&b, scratch, error, capacity) < 0) {
            goto done;
        }
    }
    for (size_t i = 0; i < count; ++i) {
        if (a.ctx[i]->frames != b.ctx[i]->frames ||
            mynah_transformer_ar_state_offset(a.ctx[i]->backbone) !=
                mynah_transformer_ar_state_offset(b.ctx[i]->backbone)) {
            pocket_error(error, capacity,
                         "%s: request %zu diverged in length after a refused step",
                         what, i);
            goto done;
        }
        if (memcmp(a.ctx[i]->hidden, b.ctx[i]->hidden,
                   hidden_dim * sizeof(float)) != 0 ||
            memcmp(a.ctx[i]->latents, b.ctx[i]->latents,
                   a.ctx[i]->frames * latent_dim * sizeof(float)) != 0) {
            pocket_error(error, capacity,
                         "%s: request %zu is not bit-identical to the same request "
                         "that never shared a refused step",
                         what, i);
            goto done;
        }
    }
    rc = 0;
done:
    pocket_check_set_free(&a);
    pocket_check_set_free(&b);
    return rc;
}

/* E8-4.  `decode_audio_batch` must hand each context exactly what
 * `decode_audio` would have handed it alone.
 *
 * Set A decodes context by context; set B decodes the same frames as a gang.
 * The ranges are deliberately ragged -- context i takes 1, 2 or 3 frames a turn
 * -- because the driver ramps each slot's quantum separately and a gang whose
 * members are all the same length is the one case that cannot catch a length
 * assumption. */
static int pocket_check_gang(mynah_engine_state *state,
                             const mynah_tts_model *model,
                             const pocket_check_case *cases, size_t count,
                             size_t max_steps, mynah_engine_scratch *scratch,
                             char *error, size_t capacity) {
    pocket_check_set a, b;
    if (pocket_check_set_new(state, model, cases, count, max_steps, &a, error,
                             capacity) != 0) {
        return -1;
    }
    if (pocket_check_set_new(state, model, cases, count, max_steps, &b, error,
                             capacity) != 0) {
        pocket_check_set_free(&a);
        return -1;
    }
    int rc = -1;
    size_t done_frames[POCKET_CHECK_MAX];
    for (size_t i = 0; i < count; ++i) done_frames[i] = 0;

    for (size_t round = 0; round < 6u; ++round) {
        /* Generate a few more frames for both sets in lockstep. */
        for (size_t step = 0; step < 3u; ++step) {
            if (pocket_check_advance(&a, scratch, error, capacity) < 0 ||
                pocket_check_advance(&b, scratch, error, capacity) < 0) {
                goto done;
            }
        }
        size_t first[POCKET_CHECK_MAX];
        size_t want[POCKET_CHECK_MAX];
        float *got[POCKET_CHECK_MAX];
        size_t got_n[POCKET_CHECK_MAX];
        int failed[POCKET_CHECK_MAX];
        for (size_t i = 0; i < count; ++i) {
            const size_t available = b.ctx[i]->frames - done_frames[i];
            size_t quantum = (i % 3u) + 1u;
            if (quantum > available) quantum = available;
            first[i] = done_frames[i];
            want[i] = quantum;
            got[i] = NULL;
            got_n[i] = 0;
            failed[i] = 0;
        }
        mynah_engine_ctx *const *bctxs = b.ctx;
        if (pocket_decode_audio_batch(bctxs, count, first, want, got, got_n, failed,
                                      scratch, error, capacity) != 0) {
            goto done;
        }
        for (size_t i = 0; i < count; ++i) {
            float *solo = NULL;
            size_t solo_n = 0;
            const int bad = failed[i] ||
                            pocket_decode_audio(a.ctx[i], first[i], want[i], &solo,
                                                &solo_n, error, capacity) != 0;
            if (bad || solo_n != got_n[i] ||
                (solo_n != 0 &&
                 memcmp(solo, got[i], solo_n * sizeof(float)) != 0)) {
                pocket_error(error, capacity,
                             "decode gang: request %zu of %zu differs from its own "
                             "solo decode of frames [%zu, %zu)",
                             i, count, first[i], first[i] + want[i]);
                free(solo);
                /* Only what this loop has not handed back yet: got[0..i-1] were
                 * already freed at the bottom of their own iteration. */
                for (size_t j = i; j < count; ++j) free(got[j]);
                goto done;
            }
            free(solo);
            free(got[i]);
            done_frames[i] += want[i];
        }
    }
    rc = 0;
done:
    pocket_check_set_free(&a);
    pocket_check_set_free(&b);
    return rc;
}

/* E2-5.  Pins what this engine does at and across `max_tokens_per_chunk`:
 * NOTHING.  The text is prefilled in one pass whether it is under the limit or
 * three times over it, so the backbone offset after `prepare` is exactly the
 * voice prefix plus every text token.  That equality is the assertion, and it
 * is what stops chunking from being added here by accident -- if a split ever
 * appears in this engine, this fires. */
static int pocket_check_chunk_seam(mynah_engine_state *state,
                                   const mynah_tts_model *model, char *error,
                                   size_t capacity) {
    const size_t limit = state->cfg.max_tokens_per_chunk;
    if (limit == 0u) return 0; /* the pack declares none; nothing to pin */

    static const char under[] = "The quick brown fox jumps over the lazy dog.";
    static const char over[] =
        "The quick brown fox jumps over the lazy dog while a very patient cat "
        "watches from the warm windowsill and counts every single passing car on "
        "the quiet road below until the evening light finally fades away behind "
        "the distant hills and someone switches on a lamp inside the kitchen "
        "where a kettle is already whistling for the second time that hour.";
    const char *texts[2] = {under, over};
    size_t seen[2] = {0u, 0u};

    for (size_t t = 0; t < 2u; ++t) {
        int *ids = NULL;
        size_t n_ids = 0;
        if (mynah_engine_pocket_tokenize(state, texts[t], strlen(texts[t]), &ids,
                                         &n_ids, error, capacity) != 0) {
            return -1;
        }
        mynah_tts_request request;
        memset(&request, 0, sizeof(request));
        request.text_ids = ids;
        request.text_length = n_ids;
        request.speaker = 0u;
        request.temperature = -1.0f;
        mynah_engine_ctx *ctx = NULL;
        int bad = pocket_ctx_new(model, state, &request, 16u, 7u, &ctx, error,
                                 capacity) != 0;
        free(ids);
        if (bad) return -1;
        if (pocket_prepare(ctx, error, capacity) != 0) {
            pocket_ctx_free(ctx);
            return -1;
        }
        const size_t offset = mynah_transformer_ar_state_offset(ctx->backbone);
        const size_t expect = ctx->voice_positions + ctx->text_length;
        seen[t] = ctx->text_length;
        bad = (offset != expect);
        pocket_ctx_free(ctx);
        if (bad) {
            pocket_error(error, capacity,
                         "chunk seam: %zu text tokens prefilled to offset %zu, "
                         "expected %zu -- the text was split",
                         n_ids, offset, expect);
            return -1;
        }
    }
    if (seen[1] <= limit) {
        pocket_error(error, capacity,
                     "chunk seam: the long text is only %zu tokens against a limit "
                     "of %zu, so the check proves nothing",
                     seen[1], limit);
        return -1;
    }
    if (seen[0] > limit) {
        pocket_error(error, capacity,
                     "chunk seam: the short text is already over the limit");
        return -1;
    }
    return 0;
}

/*
 * E5-5.  A text synthesized in N pushes must be BYTE-IDENTICAL to the same text
 * synthesized in one.  Not correlated, not within a tolerance: identical.
 *
 * WHAT CAN ACTUALLY BREAK IT, which is why the split patterns are chosen rather
 * than arbitrary.  `mynah_transformer_ar_prefill` tiles its positions in groups
 * of `_prefill_tile()` counted from the start of the call, and hands each tile
 * to the projection hook as one call with `count == rows`.  For an unquantized
 * group that is a GEMM whose blocking depends on the row count, so a text split
 * at 13 and a text split at 16 can reassociate a row's sum differently -- and
 * every frame after it is downstream of that row through the AR loop.  The
 * patterns below therefore include splits that are aligned to the tile, splits
 * that are coprime with it, one token at a time, and a first piece shorter than
 * one tile, because those are the four ways the tiling can come apart.
 *
 * THE CONTROLS, because a byte comparison between two runs of the same code is
 * the easiest gate in the world to write blind:
 *
 *   - `reserve`: a one-shot run under a ceiling four times its own text must
 *     equal the same run with no ceiling at all.  This pins that KV CAPACITY is
 *     numerically inert -- if it were not, every push comparison below would be
 *     comparing two runs that were both wrong in the same way, and the whole
 *     check would be vacuous while green.
 *   - `differs`: a different text must produce different bytes.  Cheap, and it
 *     is what catches a harness that compares two empty buffers.
 *   - a non-zero sample count is required, for the same reason.
 */
typedef struct {
    const char *name;
    size_t first;      /* tokens admitted with the request */
    size_t chunk;      /* tokens per append, 0 = one token at a time */
} pocket_split_pattern;

/* Steps one context to its end, alone.  Alone rather than batched on purpose:
 * the question here is the text window, and batching is a separate seam with
 * its own gate -- mixing them would make a failure ambiguous. */
static int pocket_lf_run(mynah_engine_ctx *ctx, mynah_engine_scratch *scratch,
                         char *error, size_t capacity) {
    mynah_engine_ctx *one[1] = {ctx};
    mynah_engine_step_result result[1];
    mynah_engine_ctx *const *ctxs = one;
    while (!ctx->eos && !ctx->broken) {
        if (pocket_step_batch(ctxs, 1u, scratch, error, capacity) != 0) return -1;
        if (pocket_emit_batch(ctxs, 1u, result, scratch, error, capacity) != 0) {
            return -1;
        }
        if (result[0].failed) {
            pocket_error(error, capacity, "long-form: a step failed");
            return -1;
        }
    }
    return 0;
}

/* One context carrying `text` pushed according to `split` (NULL = one shot),
 * run to the end, with all of its audio decoded in one call. */
static int pocket_lf_synth(mynah_engine_state *state, const mynah_tts_model *model,
                           const int *ids, size_t n_ids,
                           const pocket_split_pattern *split, size_t reserve,
                           size_t max_steps, mynah_engine_scratch *scratch,
                           float **out_pcm, size_t *out_n, size_t *out_frames,
                           char *error, size_t capacity) {
    *out_pcm = NULL;
    *out_n = 0;
    *out_frames = 0;
    const size_t first = (split == NULL) ? n_ids : split->first;
    if (first == 0u || first > n_ids) {
        pocket_error(error, capacity, "long-form: bad split");
        return -1;
    }
    mynah_tts_request request;
    memset(&request, 0, sizeof(request));
    request.text_ids = ids;
    request.text_length = first;
    request.speaker = 0u;
    request.temperature = -1.0f;
    mynah_engine_ctx *ctx = NULL;
    if (pocket_ctx_new(model, state, &request, max_steps, 4242u, &ctx, error,
                       capacity) != 0) {
        return -1;
    }
    int bad = 0;
    if (reserve != 0u) {
        bad = mynah_engine_pocket_reserve_text(ctx, reserve, error, capacity) != 0;
    }
    if (!bad) bad = pocket_prepare(ctx, error, capacity) != 0;
    if (!bad && split != NULL) {
        size_t at = first;
        while (at < n_ids && !bad) {
            size_t take = (split->chunk == 0u) ? 1u : split->chunk;
            if (take > n_ids - at) take = n_ids - at;
            bad = mynah_engine_pocket_append_text(ctx, ids + at, take, error,
                                                  capacity) != 0;
            at += take;
        }
    }
    if (!bad && reserve != 0u) {
        bad = mynah_engine_pocket_seal_text(ctx, error, capacity) != 0;
    }
    if (!bad && mynah_engine_pocket_text_length(ctx) != n_ids) {
        pocket_error(error, capacity,
                     "long-form: %zu tokens accepted, %zu pushed",
                     mynah_engine_pocket_text_length(ctx), n_ids);
        bad = 1;
    }
    if (!bad) bad = pocket_lf_run(ctx, scratch, error, capacity) != 0;
    if (!bad) {
        *out_frames = ctx->frames;
        bad = pocket_decode_audio(ctx, 0u, ctx->frames, out_pcm, out_n, error,
                                  capacity) != 0;
    }
    pocket_ctx_free(ctx);
    return bad ? -1 : 0;
}

static int pocket_lf_same(const char *what, const float *a, size_t an,
                          const float *b, size_t bn, char *error, size_t capacity) {
    if (an != bn) {
        pocket_error(error, capacity,
                     "long-form (%s): %zu samples against %zu -- the split "
                     "changed how much audio the request produced", what, bn, an);
        return -1;
    }
    for (size_t i = 0; i < an; ++i) {
        if (memcmp(&a[i], &b[i], sizeof(float)) != 0) {
            pocket_error(error, capacity,
                         "long-form (%s): sample %zu of %zu differs, %.9g against "
                         "%.9g -- pushing the text in pieces is not identical to "
                         "pushing it whole", what, i, an, (double)a[i], (double)b[i]);
            return -1;
        }
    }
    return 0;
}

/*
 * The refusals, which are the half of the design that the byte comparison
 * cannot see.
 *
 * The whole reason the text ceiling is declared at admission is that it moves
 * every allocation to a point where failing is a 503, and leaves `append_text`
 * with nothing to fail at except a bounds check.  That is only worth anything
 * if the bounds check leaves the context USABLE -- a refusal that quietly
 * half-appended would be strictly worse than the allocation failure it
 * replaced.  So the assertion is not "it returned -1", it is "it returned -1
 * and the audio afterwards is byte-identical to a request that had asked for
 * exactly the tokens that were accepted".
 *
 * Five refusals, each with the state it must not have disturbed:
 *
 *   1. reserve_text after prepare   -- the ceiling is an admission-time number
 *   2. append past the ceiling      -- and the request still synthesizes
 *   3. append after a step          -- and the request still synthesizes
 *   4. step while the text is open  -- caught in the pre-flight, nothing moved
 *   5. append without a reserve     -- an ordinary request is closed
 */
static int pocket_check_lf_refusals(mynah_engine_state *state,
                                    const mynah_tts_model *model, char *error,
                                    size_t capacity) {
    static const char text[] =
        "The quick brown fox jumps over the lazy dog while a patient cat watches "
        "from the warm windowsill and counts the passing cars on the quiet road.";
    const size_t max_steps = 12u;
    int *ids = NULL;
    size_t n_ids = 0;
    if (mynah_engine_pocket_tokenize(state, text, strlen(text), &ids, &n_ids, error,
                                     capacity) != 0) {
        return -1;
    }
    int rc = -1;
    float *ref = NULL, *got = NULL;
    size_t ref_n = 0, got_n = 0, frames = 0;
    mynah_engine_ctx *ctx = NULL;
    char scratch_error[512];
    const size_t accepted = n_ids / 2u;
    if (accepted < 4u) {
        pocket_error(error, capacity, "long-form refusals: the text is too short");
        goto done;
    }
    /* What the request WILL have asked for once the over-ceiling push has been
     * refused: the first `accepted` tokens, admitted the ordinary way. */
    if (pocket_lf_synth(state, model, ids, accepted, NULL, 0u, max_steps, NULL,
                        &ref, &ref_n, &frames, error, capacity) != 0) {
        goto done;
    }
    if (ref_n == 0u) {
        pocket_error(error, capacity, "long-form refusals: no reference audio");
        goto done;
    }

    mynah_tts_request request;
    memset(&request, 0, sizeof(request));
    request.text_ids = ids;
    request.text_length = 2u;
    request.speaker = 0u;
    request.temperature = -1.0f;

    /* 5. an ordinary request is closed: append without a reserve is refused. */
    if (pocket_ctx_new(model, state, &request, max_steps, 4242u, &ctx, error,
                       capacity) != 0) {
        goto done;
    }
    if (pocket_prepare(ctx, error, capacity) != 0) goto done;
    if (mynah_engine_pocket_append_text(ctx, ids + 2u, 1u, scratch_error,
                                        sizeof(scratch_error)) == 0) {
        pocket_error(error, capacity,
                     "long-form refusals: append_text succeeded on a context that "
                     "never reserved; an ordinary request must be closed");
        goto done;
    }
    /* 1. and the ceiling cannot be raised once the prefix is in the cache. */
    if (mynah_engine_pocket_reserve_text(ctx, n_ids, scratch_error,
                                         sizeof(scratch_error)) == 0) {
        pocket_error(error, capacity,
                     "long-form refusals: reserve_text succeeded after prepare");
        goto done;
    }
    pocket_ctx_free(ctx);
    ctx = NULL;

    /* 2, 3 and 4, on one context that must survive all three. */
    if (pocket_ctx_new(model, state, &request, max_steps, 4242u, &ctx, error,
                       capacity) != 0) {
        goto done;
    }
    /* One token of headroom, deliberately: the post-step refusal below must be
     * the STEP rule refusing, not the ceiling refusing first.  Without the
     * headroom, deleting the step check leaves this check green -- measured. */
    if (mynah_engine_pocket_reserve_text(ctx, accepted + 1u, error, capacity) != 0 ||
        pocket_prepare(ctx, error, capacity) != 0) {
        goto done;
    }
    /* 4. stepping while the text is open is refused, and nothing moves. */
    {
        mynah_engine_ctx *one[1] = {ctx};
        mynah_engine_ctx *const *ctxs = one;
        const size_t before = mynah_transformer_ar_state_offset(ctx->backbone);
        if (pocket_step_batch(ctxs, 1u, NULL, scratch_error,
                              sizeof(scratch_error)) == 0) {
            pocket_error(error, capacity,
                         "long-form refusals: a context with open text stepped");
            goto done;
        }
        if (mynah_transformer_ar_state_offset(ctx->backbone) != before ||
            ctx->step != 0u) {
            pocket_error(error, capacity,
                         "long-form refusals: the refused step moved the context");
            goto done;
        }
    }
    if (mynah_engine_pocket_append_text(ctx, ids + 2u, accepted - 2u, error,
                                        capacity) != 0) {
        goto done;
    }
    /* 2. two tokens into one token of headroom: refused, and nothing accepted. */
    if (mynah_engine_pocket_append_text(ctx, ids + accepted, 2u, scratch_error,
                                        sizeof(scratch_error)) == 0) {
        pocket_error(error, capacity,
                     "long-form refusals: append_text passed the ceiling of %zu",
                     mynah_engine_pocket_text_capacity(ctx));
        goto done;
    }
    if (mynah_engine_pocket_text_length(ctx) != accepted) {
        pocket_error(error, capacity,
                     "long-form refusals: the refused append left %zu tokens, not "
                     "%zu -- it was not a clean refusal",
                     mynah_engine_pocket_text_length(ctx), accepted);
        goto done;
    }
    if (mynah_engine_pocket_seal_text(ctx, error, capacity) != 0) goto done;
    /* 3. one step, then an append that must be refused for good, then finish. */
    {
        mynah_engine_ctx *one[1] = {ctx};
        mynah_engine_step_result result[1];
        mynah_engine_ctx *const *ctxs = one;
        if (pocket_step_batch(ctxs, 1u, NULL, error, capacity) != 0 ||
            pocket_emit_batch(ctxs, 1u, result, NULL, error, capacity) != 0) {
            goto done;
        }
        /* This one FITS the ceiling, so the ceiling cannot be what refuses it.
         * The REASON is asserted, not just the refusal: a sealed context is
         * also refused by the `text_open` check, so a test that accepted any
         * non-zero return would stay green with the step rule deleted --
         * measured, and it is why the message is matched. */
        scratch_error[0] = '\0';
        if (mynah_engine_pocket_append_text(ctx, ids, 1u, scratch_error,
                                            sizeof(scratch_error)) == 0) {
            pocket_error(error, capacity,
                         "long-form refusals: append_text succeeded on a context "
                         "that had already generated");
            goto done;
        }
        if (strstr(scratch_error, "already generated") == NULL) {
            pocket_error(error, capacity,
                         "long-form refusals: the post-step append was refused for "
                         "the wrong reason (%s)", scratch_error);
            goto done;
        }
    }
    if (pocket_lf_run(ctx, NULL, error, capacity) != 0) goto done;
    if (pocket_decode_audio(ctx, 0u, ctx->frames, &got, &got_n, error,
                            capacity) != 0) {
        goto done;
    }
    if (pocket_lf_same("three refusals, then the accepted text", ref, ref_n, got,
                       got_n, error, capacity) != 0) {
        goto done;
    }
    rc = 0;
done:
    pocket_ctx_free(ctx);
    free(ref);
    free(got);
    free(ids);
    return rc;
}

static int pocket_lf_compare(mynah_engine_state *state,
                             const mynah_tts_model *model, const char *profile,
                             const int *ids, size_t n_ids, const int *other_ids,
                             size_t n_other, mynah_engine_scratch *scratch,
                             char *error, size_t capacity);

/*
 * THE PROFILE MATTERS, and finding out that it does is the reason this runs
 * twice.  Deleting the tile alignment from `pocket_text_flush` -- the naive
 * implementation, where every push prefills everything it has -- leaves this
 * check GREEN under the shipped default and under `MYNAH_QUANT=int8`, and fails
 * it at one ULP on sample 0 under `MYNAH_QUANT_GROUPS=none`.  That is not luck:
 * a quantized group goes through a kernel that is bit-exact per row whatever
 * the row count, while an UNQUANTIZED group's tile is a GEMM whose blocking is
 * a function of the row count.  So the f32 tile path is the only place the
 * split point can reach the arithmetic, and a version of this check that ran
 * only the configured profile would have been one of the gates that pass while
 * testing nothing.
 *
 * The masks are therefore forced off for a second pass and restored.  It is a
 * reach into the state, and the alternative -- asking the operator to run the
 * command three times under three environments -- leaves a CI that runs it once
 * blind to the only failure mode there is.
 */
static int pocket_check_long_form(mynah_engine_state *state,
                                  const mynah_tts_model *model,
                                  mynah_engine_scratch *scratch, char *error,
                                  size_t capacity) {
    static const char text[] =
        "The quick brown fox jumps over the lazy dog while a patient cat watches "
        "from the warm windowsill and counts the passing cars on the quiet road "
        "below, until the evening light finally fades behind the distant hills "
        "and somebody switches on a small lamp inside the kitchen.";
    static const char other[] =
        "Bit identity is not a tolerance, and a gate that cannot fail is not a gate.";
    int *ids = NULL, *other_ids = NULL;
    size_t n_ids = 0, n_other = 0;
    if (mynah_engine_pocket_tokenize(state, text, strlen(text), &ids, &n_ids, error,
                                     capacity) != 0) {
        return -1;
    }
    if (mynah_engine_pocket_tokenize(state, other, strlen(other), &other_ids,
                                     &n_other, error, capacity) != 0) {
        free(ids);
        return -1;
    }
    int rc;
    {
        /* As configured, then with every quantization group forced off. */
        const unsigned qgroups = state->qgroups;
        const unsigned bb = state->backbone_hook.groups;
        const unsigned ct = state->codec_hook.groups;
        const unsigned fl = state->flow_hook.groups;
        rc = pocket_check_lf_refusals(state, model, error, capacity);
        if (rc == 0) {
            rc = pocket_lf_compare(state, model, "as configured", ids, n_ids,
                                   other_ids, n_other, scratch, error, capacity);
        }
        if (rc == 0 && (qgroups | bb | ct | fl) != 0u) {
            state->qgroups = 0u;
            state->backbone_hook.groups = 0u;
            state->codec_hook.groups = 0u;
            state->flow_hook.groups = 0u;
            rc = pocket_lf_compare(state, model, "quantization forced off", ids,
                                   n_ids, other_ids, n_other, scratch, error,
                                   capacity);
            state->qgroups = qgroups;
            state->backbone_hook.groups = bb;
            state->codec_hook.groups = ct;
            state->flow_hook.groups = fl;
        }
    }
    free(ids);
    free(other_ids);
    return rc;
}

static int pocket_lf_compare(mynah_engine_state *state,
                             const mynah_tts_model *model, const char *profile,
                             const int *ids, size_t n_ids, const int *other_ids,
                             size_t n_other, mynah_engine_scratch *scratch,
                             char *error, size_t capacity) {
    const size_t tile = mynah_transformer_ar_prefill_tile();
    const size_t max_steps = 20u;
    int rc = -1;
    float *base = NULL, *got = NULL, *control = NULL;
    size_t base_n = 0, got_n = 0, control_n = 0, base_frames = 0, frames = 0;

    if (n_ids < 3u * tile) {
        pocket_error(error, capacity,
                     "long-form (%s): the text is only %zu tokens against a "
                     "tile of %zu; the split patterns would prove nothing",
                     profile, n_ids, tile);
        goto done;
    }
    /* The reference: one text, one push, no ceiling -- exactly what a request
     * that never heard of E5-5 does. */
    if (pocket_lf_synth(state, model, ids, n_ids, NULL, 0u, max_steps, scratch,
                        &base, &base_n, &base_frames, error, capacity) != 0) {
        goto done;
    }
    if (base_n == 0u) {
        pocket_error(error, capacity,
                     "long-form (%s): the reference produced no audio", profile);
        goto done;
    }
    /* Control 1: capacity is numerically inert.  Without this the comparisons
     * below could all be green and all be wrong together. */
    if (pocket_lf_synth(state, model, ids, n_ids, NULL, 4u * n_ids, max_steps,
                        scratch, &control, &control_n, &frames, error,
                        capacity) != 0) {
        goto done;
    }
    char control_what[160];
    snprintf(control_what, sizeof(control_what),
             "%s, a ceiling four times the text, still one push", profile);
    if (pocket_lf_same(control_what, base, base_n, control, control_n, error,
                       capacity) != 0) {
        goto done;
    }
    free(control);
    control = NULL;
    /* Control 2: a different text differs, so the comparison is not comparing
     * two identical nothings. */
    if (pocket_lf_synth(state, model, other_ids, n_other, NULL, 0u, max_steps,
                        scratch, &control, &control_n, &frames, error,
                        capacity) != 0) {
        goto done;
    }
    if (control_n == base_n &&
        memcmp(control, base, base_n * sizeof(float)) == 0) {
        pocket_error(error, capacity,
                     "long-form (%s): a different text produced identical "
                     "audio; the comparison proves nothing", profile);
        goto done;
    }
    free(control);
    control = NULL;

    const pocket_split_pattern patterns[] = {
        {"one token at a time", 1u, 0u},
        {"a first piece under one tile, then tile-sized", 3u, tile},
        {"tile-aligned throughout", tile, tile},
        {"coprime with the tile", 7u, 5u},
        {"one long piece, then a short tail", n_ids - 2u, 1u},
        {"everything at admission, sealed with no append", n_ids, n_ids},
    };
    for (size_t p = 0; p < sizeof(patterns) / sizeof(patterns[0]); ++p) {
        if (pocket_lf_synth(state, model, ids, n_ids, &patterns[p], n_ids,
                            max_steps, scratch, &got, &got_n, &frames, error,
                            capacity) != 0) {
            goto done;
        }
        if (frames != base_frames) {
            pocket_error(error, capacity,
                         "long-form (%s, %s): %zu frames against %zu", profile,
                         patterns[p].name, frames, base_frames);
            goto done;
        }
        char what[160];
        snprintf(what, sizeof(what), "%s, %s", profile, patterns[p].name);
        if (pocket_lf_same(what, base, base_n, got, got_n, error, capacity) != 0) {
            goto done;
        }
        free(got);
        got = NULL;
    }
    rc = 0;
done:
    free(base);
    free(got);
    free(control);
    return rc;
}

/* ------------------------------------------------------------ cloning
 *
 * The glue E7 was missing. `src/voice_clone.c` has had the arithmetic since
 * 2026-09-12, checked against the PyTorch oracle to 6.4e-06 and covered by a
 * model-free self-test -- but nothing ever built a `mynah_voice_clone_config`
 * from a real pack, so no caller could reach it and `PLAN.md` carried an
 * `export-voice` command that did not exist. This is that one function.
 *
 * Every dimension below comes from `model.json` and none is new. Three of the
 * four sub-configurations are already built elsewhere in this file for the
 * decoder; the encoder mirrors them because upstream's encoder IS the decoder
 * mirrored (`.work/voice-cloning.md`):
 *
 *   seanet              the decoder's, verbatim: same ratios (decoder order,
 *                       reversed internally), filters, kernels, compression
 *   encoder_transformer the codec transformer's shape -- 2 layers, d512,
 *                       context 250 -- which is what the pack declares
 *   downsample          stride 16, [codec_dim] -> [latent_dim], groups 1; the
 *                       mirror of the decoder's depthwise upsample
 *   backbone            the LM's, because the voice IS its KV cache
 *
 * `max_seq_len` is left at zero in both transformers on purpose: the module
 * sizes them from `max_seconds`, and a number invented here could only disagree.
 *
 * The output is a voice file in the pack's own format, so a cloned voice is not
 * a second kind of voice: add it to `speakers.json` and every path that serves a
 * predefined voice serves it, with no code aware of where it came from. */
int mynah_engine_pocket_clone_voice(const mynah_tts_model *model,
                                    const char *wav_path, const char *out_path,
                                    const char *affirmation,
                                    mynah_pocket_clone_report *report,
                                    char *error, size_t capacity) {
    if (model == NULL || wav_path == NULL || out_path == NULL) {
        pocket_error(error, capacity, "clone-voice: model, reference and output "
                                      "are all required");
        return -1;
    }
    if (report != NULL) memset(report, 0, sizeof(*report));

    mynah_engine_state *state = NULL;
    if (pocket_model_init(model, &state, error, capacity) != 0) return -1;
    const pocket_config *cfg = &state->cfg;

    int rc = -1;
    mynah_voice_clone_weights_owner *owner = NULL;
    mynah_voice_encoder *encoder = NULL;
    mynah_transformer_ar_state *voice = NULL;
    mynah_audio_clip clip;
    memset(&clip, 0, sizeof(clip));

    mynah_voice_clone_config vc;
    mynah_voice_clone_config_defaults(&vc);

    vc.seanet.channels = cfg->audio_channels;
    vc.seanet.dimension = cfg->codec_dim;
    vc.seanet.n_filters = cfg->n_filters;
    vc.seanet.n_residual_layers = cfg->n_residual_layers;
    vc.seanet.ratios = cfg->ratios;
    vc.seanet.n_ratios = cfg->n_ratios;
    vc.seanet.kernel_size = cfg->kernel_size;
    vc.seanet.residual_kernel_size = cfg->residual_kernel_size;
    vc.seanet.last_kernel_size = cfg->last_kernel_size;
    vc.seanet.dilation_base = cfg->dilation_base;
    vc.seanet.compress = cfg->compress;
    vc.seanet.elu_alpha = cfg->elu_alpha;

    vc.encoder_transformer.d_model = cfg->codec_tf_dim;
    vc.encoder_transformer.num_heads = cfg->codec_tf_heads;
    vc.encoder_transformer.head_dim = cfg->codec_tf_dim / cfg->codec_tf_heads;
    vc.encoder_transformer.num_layers = cfg->codec_tf_layers;
    vc.encoder_transformer.ffn_dim = cfg->codec_tf_ffn;
    vc.encoder_transformer.context = cfg->codec_tf_context;
    vc.encoder_transformer.layernorm_eps = cfg->layernorm_eps;

    vc.downsample.stride = cfg->downsample_stride;
    vc.downsample.in_channels = cfg->codec_dim;
    vc.downsample.out_channels = cfg->latent_dim;
    vc.downsample.groups = 1u;

    vc.backbone.d_model = cfg->hidden_dim;
    vc.backbone.num_heads = cfg->heads;
    vc.backbone.head_dim = cfg->head_dim;
    vc.backbone.num_layers = cfg->layers;
    vc.backbone.ffn_dim = cfg->ffn_dim;
    vc.backbone.context = 0u;
    vc.backbone.layernorm_eps = cfg->layernorm_eps;

    vc.sample_rate = cfg->sample_rate;
    vc.samples_per_frame = cfg->samples_per_frame;
    vc.insert_bos_before_voice = cfg->insert_bos_before_voice;

    mynah_voice_clone_consent consent;
    consent.affirmed = (affirmation != NULL && *affirmation != '\0');
    consent.source = wav_path;
    consent.affirmation = affirmation;
    if (mynah_voice_clone_consent_check(&consent, error, capacity) != 0) goto done;

    if (mynah_voice_clone_weights_load(state->weights, &vc, &owner, error,
                                       capacity) != 0) {
        goto done;
    }
    const mynah_voice_clone_weights *weights = mynah_voice_clone_weights_view(owner);
    const mynah_transformer_ar_weights *backbone =
        mynah_voice_clone_backbone_view(owner);

    if (mynah_wav_read_mono(wav_path, &clip, error, capacity) != 0) goto done;
    if (report != NULL) {
        report->input_sample_rate = clip.sample_rate;
        report->input_seconds = clip.sample_rate != 0u
            ? (double)clip.count / (double)clip.sample_rate : 0.0;
    }
    /* Truncate at the clip's OWN rate, which is where upstream truncates:
     * before resampling, not after. */
    mynah_audio_clip_truncate(&clip, vc.max_seconds);
    if (mynah_audio_clip_resample(&clip, vc.sample_rate, error, capacity) != 0) {
        goto done;
    }

    encoder = mynah_voice_encoder_create(&vc, error, capacity);
    if (encoder == NULL) goto done;
    if (mynah_voice_clone_check_weights(encoder, weights, error, capacity) != 0) {
        goto done;
    }
    if (mynah_voice_encoder_encode(encoder, weights, &consent, clip.samples,
                                   clip.count, error, capacity) != 0) {
        goto done;
    }

    const size_t frames = mynah_voice_encoder_frames(encoder);
    if (frames == 0u) {
        pocket_error(error, capacity,
                     "clone-voice: the reference produced no frames; it is "
                     "shorter than one %zu-sample frame", cfg->samples_per_frame);
        goto done;
    }
    /* One position per voice frame, plus the BOS this pack inserts before it. */
    vc.backbone.max_seq_len = frames + 1u;
    voice = mynah_transformer_ar_state_new(&vc.backbone, error, capacity);
    if (voice == NULL) goto done;
    if (mynah_voice_clone_prefill(voice, backbone,
                                  cfg->insert_bos_before_voice
                                      ? weights->bos_before_voice : NULL,
                                  mynah_voice_encoder_conditioning(encoder),
                                  frames, error, capacity) != 0) {
        goto done;
    }
    if (mynah_voice_export(voice, out_path, MYNAH_VOICE_DTYPE_F16, &consent,
                           cfg->revision, error, capacity) != 0) {
        goto done;
    }
    if (report != NULL) {
        report->voice_frames = frames;
        report->voice_positions = frames + (cfg->insert_bos_before_voice ? 1u : 0u);
        report->voice_seconds = cfg->frame_rate > 0.0
            ? (double)frames / cfg->frame_rate : 0.0;
        snprintf(report->revision, sizeof(report->revision), "%s", cfg->revision);
    }
    rc = 0;

done:
    mynah_transformer_ar_state_free(voice);
    mynah_voice_encoder_destroy(encoder);
    mynah_voice_clone_weights_free(owner);
    mynah_audio_clip_free(&clip);
    pocket_model_free(state);
    return rc;
}

int mynah_engine_pocket_self_check(const mynah_tts_model *model, char *error,
                                   size_t capacity) {
    if (model == NULL) {
        pocket_error(error, capacity, "pocket self-check: no model");
        return -1;
    }
    static const pocket_check_case cases[POCKET_CHECK_MAX] = {
        {"The quick brown fox jumps over the lazy dog.", 0u, 11u},
        {"Hello there, this is a short one.", 3u, 22u},
        {"Numbers like 1234 are read out loud.", 7u, 33u},
        {"A somewhat longer sentence, with a comma in it, to move the EOS step.", 1u, 44u},
        {"Short.", 5u, 55u},
        {"Another distinct utterance entirely.", 9u, 66u},
        {"Testing, testing, one two three.", 2u, 77u},
        {"The rain in Spain stays mainly in the plain.", 4u, 88u},
        {"Once upon a time there was a runtime with no Python.", 6u, 99u},
        {"Bit identity is not a tolerance.", 8u, 110u},
        {"Sixteen requests walk into a batch.", 10u, 121u},
        {"The codec carries state and never replays context.", 11u, 132u},
        {"One request's failure retires one request.", 12u, 143u},
        {"Append, never insert.", 13u, 154u},
        {"Measure before changing kernels.", 14u, 165u},
        {"CPU is the product.", 15u, 176u},
    };

    mynah_engine_state *state = NULL;
    if (pocket_model_init(model, &state, error, capacity) != 0) return -1;

    int rc = -1;
    if (pocket_check_chunk_seam(state, model, error, capacity) != 0) goto done;
    if (pocket_check_long_form(state, model, NULL, error, capacity) != 0) goto done;
    static const size_t widths[4] = {2u, 4u, 8u, 16u};
    for (size_t w = 0; w < 4u; ++w) {
        size_t count = widths[w];
        if (count > POCKET_CHECK_MAX) count = POCKET_CHECK_MAX;
        mynah_engine_scratch *scratch = NULL;
        if (pocket_scratch_new(model, state, count, &scratch, error, capacity) != 0) {
            goto done;
        }
        /* Both refusal points, on both paths. The per-row path (NULL scratch)
         * crossed with the KV injection is the one combination that fails
         * without the rollback; the other three would pass on their own. */
        static const pocket_inject modes[2] = {POCKET_INJECT_LATENT_NAN,
                                               POCKET_INJECT_KV_NAN};
        static const char *const mode_name[2] = {"latent NaN, pre-flight",
                                                 "cached NaN, in-backbone"};
        char what[96];
        int bad = 0;
        for (size_t m = 0; m < 2u && !bad; ++m) {
            snprintf(what, sizeof(what), "step atomicity at width %zu (batched, %s)",
                     count, mode_name[m]);
            bad = pocket_check_atomic(state, model, cases, count, 12u, scratch,
                                      modes[m], what, error, capacity) != 0;
            if (bad) break;
            snprintf(what, sizeof(what), "step atomicity at width %zu (per-row, %s)",
                     count, mode_name[m]);
            bad = pocket_check_atomic(state, model, cases, count, 12u, NULL,
                                      modes[m], what, error, capacity) != 0;
        }
        if (!bad) {
            bad = pocket_check_gang(state, model, cases, count, 32u, scratch, error,
                                    capacity) != 0;
        }
        pocket_scratch_free(scratch);
        if (bad) goto done;
    }
    rc = 0;
done:
    pocket_model_free(state);
    return rc;
}
