/*
 * PocketTTS voice cloning (E7): a reference waveform in, a backbone KV cache
 * out.  See `.work/voice-cloning.md` for why this is cheap on a continuous
 * latent model and `.work/licensing-and-voice-policy.md` for the consent rule
 * that is wired into the API below rather than bolted onto the CLI.
 *
 * The path, exactly as upstream implements it (`models/tts_model.py`
 * `_encode_audio` / `get_state_for_audio_prompt`, `models/mimi.py`
 * `encode_to_latent`):
 *
 *     wav  -> decode, to mono, truncate to 30 s, resample to 24 kHz
 *          -> zero-pad right to a whole 1920-sample frame
 *          -> mimi.encoder            SEANet, the decoder mirrored: a plain
 *                                     causal conv where the decoder has a
 *                                     transposed one, and the residual block
 *                                     comes *before* the downsampling conv
 *          -> encoder_transformer     2 layers, d512, layer_scale, context 250
 *          -> ConvDownsample1d        stride 16, kernel 32, pad "replicate"
 *                                     -> latents [T][latent_dim]
 *          -> F.linear(speaker_proj)  -> conditioning [T][d_model]
 *          -> cat(bos_before_voice)   if insert_bos_before_voice
 *          -> prefill the backbone    -> the KV cache *is* the voice
 *          -> (optional) serialise the KV as a .safetensors voice file
 *
 * ## What this module is and is not
 *
 * It is the PocketTTS-specific composition of primitives that already exist:
 * `mynah_causal_conv1d` (src/seanet.h) for every convolution including the
 * replicate-padded downsample, `mynah_transformer_ar` (src/transformer_ar.h)
 * for the encoder transformer and for the backbone prefill.  Nothing here
 * reimplements a kernel.  It is the one place that formats PocketTTS tensor
 * names, and it does that only in `_weights_load`, which an engine that has
 * already resolved its weights can skip: every compute entry point takes
 * resolved `const float *`.
 *
 * It is *not* a second inference path.  The prefill goes through the same
 * `mynah_transformer_ar_prefill` a generation uses, and the resulting state is
 * an ordinary backbone state that generation continues from.
 *
 * ## Streaming, because 30 s of 24 kHz at 64 channels is 184 MB
 *
 * The encoder runs in chunks of `chunk_latent_frames` (default 4 = 7680
 * samples).  Every convolution carries its left context in the ring buffer
 * `mynah_causal_conv1d` already owns, and the encoder transformer carries its
 * KV, so chunking is exact rather than approximate — the same property the
 * decoder relies on (src/seanet.h).  The chunk must be a whole number of
 * latent frames so the downsample always sees a multiple of its stride.
 *
 * ## Consent is a parameter, not a policy document
 *
 * Every entry point that turns *supplied audio* into a voice takes a
 * `mynah_voice_clone_consent` and refuses a NULL or unaffirmed one.  There is
 * no default that clones silently, and the affirmation is recorded into the
 * exported voice file's metadata rather than assumed.  Loading a predefined
 * voice that ships in a pack does not go through this module at all.
 */
#ifndef MYNAH_TTS_VOICE_CLONE_H
#define MYNAH_TTS_VOICE_CLONE_H

#include <stddef.h>

#include "seanet.h"
#include "transformer_ar.h"
#include "weights.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------- consent */

/* Notice text for `--help`, the README and the API error path.  One string so
 * the three cannot drift apart. */
extern const char *const mynah_voice_clone_consent_notice;

typedef struct {
    int affirmed;            /* must be non-zero; nothing here defaults to 1 */
    const char *source;      /* what was cloned, e.g. the file path; may be NULL */
    const char *affirmation; /* recorded verbatim into an exported voice file */
} mynah_voice_clone_consent;

/* 0 when `consent` is present and affirmed, -1 with the notice in `error`. */
int mynah_voice_clone_consent_check(const mynah_voice_clone_consent *consent,
                                    char *error, size_t error_capacity);

/* ---------------------------------------------------------- audio input */

typedef struct {
    float *samples;       /* owned, mono, nominally [-1, 1]          */
    size_t count;
    unsigned sample_rate;
} mynah_audio_clip;

void mynah_audio_clip_free(mynah_audio_clip *clip);

/* RIFF/WAVE reader: PCM 8/16/24/32-bit integer and 32/64-bit IEEE float,
 * any channel count, downmixed to mono by averaging.  WAVE_FORMAT_EXTENSIBLE
 * is resolved through its sub-format GUID.  On success `*out` owns memory. */
int mynah_wav_read_mono(const char *path, mynah_audio_clip *out, char *error,
                        size_t error_capacity);
int mynah_wav_decode_mono(const void *bytes, size_t size,
                          mynah_audio_clip *out, char *error,
                          size_t error_capacity);

/* Keeps the first `seconds` at the clip's own rate, which is where upstream
 * truncates: before resampling, not after. */
void mynah_audio_clip_truncate(mynah_audio_clip *clip, double seconds);

/*
 * Rational polyphase resampler, `scipy.signal.resample_poly(x, up, down)` with
 * its default Kaiser(5.0) window: a `2 * (10 * max(up, down)) + 1`-tap
 * linear-phase FIR, zero padding outside the signal, and the group delay
 * removed so output sample j lands at input time j * down / up.  Matched to
 * the reference within tolerance, not bit-exactly (`.work/voice-cloning.md`).
 *
 * `up` and `down` are reduced by their gcd internally; equal factors copy.
 * `*out` is a fresh allocation of `*out_count` floats owned by the caller.
 */
int mynah_audio_resample_poly(const float *input, size_t count, size_t up,
                              size_t down, float **out, size_t *out_count,
                              char *error, size_t error_capacity);

/* Resamples the clip in place to `target_rate`; a no-op at the right rate. */
int mynah_audio_clip_resample(mynah_audio_clip *clip, unsigned target_rate,
                              char *error, size_t error_capacity);

/* -------------------------------------------------------------- config */

/*
 * Every dimension comes from the pack's `model.json`; nothing is defaulted to
 * a PocketTTS number here.  `mynah_voice_clone_config_defaults` fills in only
 * the architecture constants (ELU alpha, RoPE period, the layernorm epsilon,
 * the 30 s cap, the chunk size).  `max_seconds` must be positive: every buffer
 * in the encoder, the encoder transformer's KV included, is sized from it.
 */
typedef struct {
    /* mimi.encoder.  `ratios` is in *decoder* order, e.g. {6, 5, 4}, exactly
     * as `model.json` stores it; the encoder reverses it internally. */
    mynah_seanet_config seanet;
    /* mimi.encoder_transformer.  max_seq_len is computed from max_seconds. */
    mynah_transformer_ar_config encoder_transformer;
    /* mimi.downsample: stride 16, in `seanet.dimension`, out `latent_dim`,
     * groups 1. */
    mynah_resample_config downsample;
    /* flow_lm.transformer, for the prefill.  Only d_model/num_heads/head_dim/
     * num_layers are read here; the state the caller passes to `_prefill`
     * carries its own. */
    mynah_transformer_ar_config backbone;

    unsigned sample_rate;      /* 24000                                     */
    size_t samples_per_frame;  /* 1920 = sample_rate / frame_rate           */
    int insert_bos_before_voice;
    double max_seconds;        /* reference audio cap, 30.0; must be > 0     */
    size_t chunk_latent_frames;/* encoder chunking, 0 = default 4           */
} mynah_voice_clone_config;

void mynah_voice_clone_config_defaults(mynah_voice_clone_config *config);

/* -------------------------------------------------------------- weights */

/*
 * Resolved cloning weights.  Layout notes that are easy to get backwards:
 *
 *   - `stage_conv[i]` is the *downsampling* convolution of stage i, kernel
 *     2*ratio, stride ratio, taken in encoder order (ratios reversed).
 *   - `blocks` is stage-major, `n_ratios * n_residual_layers` entries, and in
 *     the encoder each stage's residual blocks run *before* its downsampling
 *     conv (the decoder is the other way round).
 *   - `speaker_proj` is `[d_model][latent_dim]`, applied as `F.linear`.
 *   - `bos_before_voice` is `[d_model]` or NULL for a checkpoint that has no
 *     such tensor (`english_2026-01`).
 */
typedef struct {
    mynah_conv_weights first;
    const mynah_conv_weights *stage_conv;
    const mynah_seanet_resblock_weights *blocks;
    mynah_conv_weights last;
    mynah_transformer_ar_weights encoder_transformer;
    mynah_conv_weights downsample;
    const float *speaker_proj;
    const float *bos_before_voice;
} mynah_voice_clone_weights;

/*
 * Owns the arrays the view above points into.  `_load` reads the cloning
 * tensors (`mimi.encoder.*`, `mimi.encoder_transformer.*`, `mimi.downsample.*`,
 * `flow_lm.speaker_proj_weight`, `flow_lm.bos_before_voice`) and, as a
 * convenience for a caller that has not resolved the backbone itself, also
 * `flow_lm.transformer.*` plus `flow_lm.out_norm.*`.  The float data stays
 * owned by `file`, which must outlive the owner.
 */
typedef struct mynah_voice_clone_weights_owner mynah_voice_clone_weights_owner;

int mynah_voice_clone_weights_load(const mynah_weights *file,
                                   const mynah_voice_clone_config *config,
                                   mynah_voice_clone_weights_owner **out,
                                   char *error, size_t error_capacity);
void mynah_voice_clone_weights_free(mynah_voice_clone_weights_owner *owner);

const mynah_voice_clone_weights *mynah_voice_clone_weights_view(
    const mynah_voice_clone_weights_owner *owner);
/* The backbone view, for `_prefill`.  An engine that already owns its backbone
 * weights should pass its own instead of loading them twice. */
const mynah_transformer_ar_weights *mynah_voice_clone_backbone_view(
    const mynah_voice_clone_weights_owner *owner);

/* -------------------------------------------------------------- encoder */

typedef struct mynah_voice_encoder mynah_voice_encoder;

/* Allocates every buffer the encode needs, sized from `max_seconds`.  Nothing
 * is allocated afterwards. */
mynah_voice_encoder *mynah_voice_encoder_create(
    const mynah_voice_clone_config *config, char *error,
    size_t error_capacity);
void mynah_voice_encoder_destroy(mynah_voice_encoder *encoder);
void mynah_voice_encoder_reset(mynah_voice_encoder *encoder);

/* Validates that every pointer the configuration requires is present. */
int mynah_voice_clone_check_weights(const mynah_voice_encoder *encoder,
                                    const mynah_voice_clone_weights *weights,
                                    char *error, size_t error_capacity);

/* Samples must already be mono at `config.sample_rate`; use
 * `mynah_audio_clip_resample` first otherwise.  Refuses audio longer than
 * `max_seconds` rather than silently truncating — truncation is the caller's
 * explicit `mynah_audio_clip_truncate`.  Requires affirmed consent. */
int mynah_voice_encoder_encode(mynah_voice_encoder *encoder,
                               const mynah_voice_clone_weights *weights,
                               const mynah_voice_clone_consent *consent,
                               const float *samples, size_t count,
                               char *error, size_t error_capacity);

size_t mynah_voice_encoder_frames(const mynah_voice_encoder *encoder);
/* [frames][latent_dim], the Mimi latents, kept for stage parity. */
const float *mynah_voice_encoder_latents(const mynah_voice_encoder *encoder);
/* [frames][d_model], what the backbone is prefilled with. */
const float *mynah_voice_encoder_conditioning(
    const mynah_voice_encoder *encoder);

/* -------------------------------------------------------------- prefill */

/*
 * Prefills `state` with (optionally) `bos_before_voice` and then the
 * conditioning rows.  The state is *not* reset first: a caller that wants a
 * fresh voice resets it.  On success the state's offset is the voice length,
 * which is what a voice file's `offset` records.
 */
int mynah_voice_clone_prefill(mynah_transformer_ar_state *state,
                              const mynah_transformer_ar_weights *weights,
                              const float *bos_before_voice,
                              const float *conditioning, size_t frames,
                              char *error, size_t error_capacity);

/* ------------------------------------------------------------ export */

typedef enum {
    MYNAH_VOICE_DTYPE_F32 = 0,
    MYNAH_VOICE_DTYPE_F16 = 1
} mynah_voice_dtype;

/*
 * Serialises the KV cache in the upstream voice-file layout:
 * `transformer.layers.N.self_attn/cache` `[2, 1, T, heads, head_dim]` and
 * `transformer.layers.N.self_attn/offset` I64`[1]` = T, tensors in name order,
 * an 8-byte little-endian header length, a JSON header space-padded to an
 * 8-byte data start.  `consent` may be NULL only for re-exporting a state that
 * did not come from supplied audio.
 */
int mynah_voice_export(mynah_transformer_ar_state *state, const char *path,
                       mynah_voice_dtype dtype,
                       const mynah_voice_clone_consent *consent,
                       const char *model_revision, char *error,
                       size_t error_capacity);

/* The same bytes, into a fresh buffer the caller frees.  Exposed because it is
 * what the self test can check without touching the filesystem. */
int mynah_voice_serialise(mynah_transformer_ar_state *state,
                          mynah_voice_dtype dtype,
                          const mynah_voice_clone_consent *consent,
                          const char *model_revision, void **out, size_t *size,
                          char *error, size_t error_capacity);

/* ------------------------------------------------------------ self test */

/*
 * Model-free: the WAV reader over PCM16/PCM24/float32 and stereo downmix, the
 * resampler against an analytically known signal and against its own identity
 * and DC gain, the SEANet encoder against a naive zero-padded reference and
 * against itself when chunked, the replicate-padded downsample against a hand
 * reference, f32<->f16, and the voice serialiser's header.  Returns 0, or -1
 * with a message in `error`.
 */
int mynah_voice_clone_self_test(char *error, size_t error_capacity);

#ifdef __cplusplus
}
#endif

#endif /* MYNAH_TTS_VOICE_CLONE_H */
