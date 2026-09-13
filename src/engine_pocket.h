/*
 * PocketTTS engine (E3): the composition layer.
 *
 * Everything here is glue.  The four pieces that do the arithmetic already
 * exist and are already checked against the Python oracle, so this file must
 * never grow a second copy of any of them:
 *
 *   src/tokenizer_sentencepiece.h  SentencePiece Unigram (45k parity cases)
 *   src/transformer_ar.h           causal backbone + KV cache      (3.4e-06)
 *   src/flow_head.h                SimpleMLPAdaLN, one LSD step    (1.07e-06)
 *   src/seanet.h                   causal SEANet + streaming state (5.7e-07)
 *
 * What is genuinely new is the graph they are wired into, and the state that
 * has to survive between frames.  Both are described by the oracle dump
 * (.work/pocket-tts-oracle.md, "The captured chain"):
 *
 *     voice   -> KV cache loaded straight into the backbone (T positions)
 *     text    -> conditioner.embed -> backbone prefill      (n_tok positions)
 *     step k:  input_linear(k == 0 ? bos_emb : latent[k-1])   [latent]->[d]
 *              backbone (6 layers) -> out_norm             -> hidden [d]
 *              out_eos(hidden)  > eos_threshold            -> remember the step
 *              flow_head(cond = hidden, s = 0, t = 1, x0 = noise) -> v
 *              latent[k] = noise + v                         (LSD, one step)
 *     decode:  denorm(latent) -> quantizer.output_proj -> upsample x16
 *              -> decoder transformer (16 positions) -> SEANet -> 1920 samples
 *
 * Six facts in that listing are measurements, not guesses, and each one is a
 * place a from-scratch implementation goes quietly wrong:
 *
 *  1. **The prefill's own head output is discarded.**  Verified against the
 *     dump: `input_linear` at AR call 1 receives `bos_emb` *exactly*
 *     (0.0 difference), and the codec's first frame is the latent of call 1,
 *     not of call 0.  So the head evaluation the reference performs on the
 *     last text position produces nothing that is ever heard, and this engine
 *     does not perform it at all.  Frame k comes from AR step k, counting the
 *     BOS step as step 0.
 *  2. **The latent is `noise + flow_net(...)`.**  The flow head returns the
 *     LSD velocity; with s = 0 and t = 1 the integration is a single addition.
 *     Verified: `input_linear.in0.call{c} == flow_net.out.call{c-1} +
 *     flow_net.in3.call{c-1}` to 0.0.
 *  3. **The noise is N(0, temperature)**, i.e. `std = sqrt(temperature)`
 *     (measured 0.539 over the dump's 128 draws at temperature 0.3, against
 *     sqrt(0.3) = 0.5477).  The generator is per context (CLAUDE.md rule 3);
 *     there is no global RNG anywhere in this engine.
 *  4. **BOS is tracked explicitly, never as a NaN sentinel.**  Upstream marks
 *     the first audio position with NaN and substitutes `bos_emb` inside the
 *     forward; `transformer_ar` rejects non-finite input on purpose, so this
 *     engine keeps a boolean and feeds `bos_emb` itself.
 *  5. **The EOS crossing is not the end of the utterance.** Upstream records
 *     the first step whose logit crosses the threshold and then generates
 *     `frames_after_eos` further frames, breaking *before* queueing the last
 *     one (`models/tts_model.py::_autoregressive_generation`). Dropping the
 *     tail clips the final word, and it is measurable rather than a matter of
 *     taste: against the pinned dump the crossing is at step 49 while the
 *     reference audio is 52 frames, i.e. exactly the 3 that
 *     `prepare_text_prompt`'s guess (1 for more than four words) plus 2 gives.
 *     `results.eos` is therefore raised on the terminal step, not the
 *     crossing, and that terminal step is the one that appends no frame.
 *
 *  6. **The codec carries state; it never replays context.**  Two pieces, both
 *     mandatory (.work/pocket-tts-oracle.md E2-3): the convolution ring
 *     buffers *and* a position counter advancing by `codec_upsample_stride`
 *     per latent frame.  Replaying left context instead would need 64 frames
 *     (5.12 s) per chunk to be exact.  `decode_audio` therefore accepts only
 *     contiguous, monotonically increasing ranges, and it decodes one latent
 *     frame at a time so that offline and streaming are the same code path
 *     (CLAUDE.md rule 7).
 *
 * No dimension is compiled in.  Everything comes from the pack: `model.json`
 * for the declared numbers, `speakers.json` for the voice table, and the
 * tensor shapes themselves for the handful of SEANet parameters the manifest
 * does not declare (filters, residual depth, compression, kernel sizes).  The
 * two values that are unobservable at `n_residual_layers == 1` - the dilation
 * base and the ELU alpha - have documented defaults and optional manifest
 * overrides.
 *
 * Known limits, stated rather than hidden:
 *
 *   - `max_tokens_per_chunk` from the manifest is read, published and
 *     reported, and deliberately **not applied**: splitting text is a policy
 *     over TEXT and the seam hands this engine token ids.  Measured at the
 *     boundary (E2-5, the block in `ctx_new`): nothing happens at 50, the skip
 *     degrades smoothly from ~2.55 to ~1.97 frames per token -- which matches
 *     upstream for a single comma-free sentence, because its splitter does not
 *     split one either -- but on text WITH sentence boundaries, where upstream
 *     would have split, past ~150 tokens this engine stops emitting EOS and
 *     runs to the step budget.  The splitter belongs above the seam.
 *   - Upstream's `prepare_text_prompt` runs before tokenization: it upcases
 *     the first letter, appends terminal punctuation, and pads inputs under
 *     five words with eight spaces.  The seam hands this engine token ids, so
 *     that belongs to whoever holds the text, and skipping it changes what the
 *     model is asked to say - the same place `frames_after_eos` comes from.
 *   - Voice cloning from a wav is not implemented.  `speaker_proj_weight`,
 *     `bos_before_voice` and `insert_bos_before_voice` are read and validated
 *     so the version difference is visible, but a predefined voice needs none
 *     of them.
 *   - The codec transformer's KV cache is sized for every position the request
 *     may reach (`max_steps * codec_upsample_stride`), because
 *     `transformer_ar` has no ring cache.  With `context: 250` all but the
 *     last 250 positions are dead weight; the fix belongs in `transformer_ar`,
 *     not here.
 */
#ifndef MYNAH_TTS_ENGINE_POCKET_H
#define MYNAH_TTS_ENGINE_POCKET_H

#include <stddef.h>

#include "tts_engine.h"

#ifdef __cplusplus
extern "C" {
#endif

/* The vtable, resolved by `mynah_engine_lookup("pocket")`. */
const mynah_tts_engine *mynah_engine_pocket(void);

/* ------------------------------------------------------- the chunk seam */

/* `max_tokens_per_chunk` from model.json, or 0 when the pack declares none.
 *
 * Published so that whoever holds the TEXT can split it, because that is where
 * the split belongs (E2-5): upstream applies it before tokenization, on
 * sentence and then clause boundaries, and by the time the request reaches this
 * seam it carries token ids and that structure is gone.  This engine never
 * splits; a request over the limit is prefilled whole and says so on stderr
 * once per model.  See the measurement in the header comment. */
size_t mynah_engine_pocket_max_tokens_per_chunk(const mynah_engine_state *state);

/* ---------------------------------------------------------------- voices */

/* Number of predefined voices in the pack, i.e. `speakers.json`. */
size_t mynah_engine_pocket_voice_count(const mynah_engine_state *state);

/* Borrowed name of voice `index`, or NULL when out of range.  The name is what
 * a user types; `mynah_tts_request.speaker` is the index. */
const char *mynah_engine_pocket_voice_name(const mynah_engine_state *state,
                                           size_t index);

/* Index of the voice called `name`, or -1.  Comparison is exact: a voice file
 * belongs to the weights that produced it, so a near-miss must not silently
 * resolve to a different speaker. */
int mynah_engine_pocket_voice_index(const mynah_engine_state *state,
                                    const char *name);

/* ------------------------------------------------------------- tokenizer */

/* The pack's SentencePiece model, owned by the engine state.  The public
 * request carries token ids, so a caller that starts from text needs this.
 * `*out_ids` is malloc'd and owned by the caller. */
int mynah_engine_pocket_tokenize(const mynah_engine_state *state,
                                 const char *text, size_t text_length,
                                 int **out_ids, size_t *out_count, char *error,
                                 size_t error_capacity);

/* ------------------------------------------------------------------- EOS */

/*
 * How many frames follow the EOS crossing before the request ends (fact 5).
 * The pack-wide default is `model_recommended_frames_after_eos` from
 * `model.json`, or 3 - upstream's value for any input longer than four words.
 * The short-input value is 5, but the rule counts *words*, and the engine
 * only ever sees token ids: a caller that still holds the text applies it
 * here. `frames` must not exceed the context's step budget.
 */
int mynah_engine_pocket_set_frames_after_eos(mynah_engine_ctx *ctx, size_t frames);
size_t mynah_engine_pocket_frames_after_eos(const mynah_engine_ctx *ctx);

/* ------------------------------------------------------ parity and debug */

/*
 * Noise injection, which exists for exactly one reason: stage 7 of the oracle
 * is only comparable when the two implementations draw the same `x_0`
 * (.work/pocket-tts-oracle.md, "Stage 7 needs the noise injected from
 * outside").  `step` counts AR steps from 0.  Returning non-zero fails the
 * emit for that request.  NULL restores the context's own generator.
 */
typedef int (*mynah_pocket_noise_fn)(void *user_data, float *noise, size_t n,
                                     size_t step);
int mynah_engine_pocket_set_noise(mynah_engine_ctx *ctx,
                                  mynah_pocket_noise_fn fn, void *user_data);

/* Post-`step_batch` hidden state, `[hidden_dim]`, borrowed: the input to both
 * `out_eos` and the flow head (oracle `flow_lm.out_norm.out`). */
const float *mynah_engine_pocket_hidden(const mynah_engine_ctx *ctx,
                                        size_t *out_count);

/* Post-`emit_batch` EOS logit for the last step (oracle `flow_lm.out_eos`). */
float mynah_engine_pocket_eos_logit(const mynah_engine_ctx *ctx);

/* Borrowed normalized latent of frame `index`, `[latent_dim]`, or NULL.  This
 * is what feeds back into `input_linear`; the codec sees it denormalized. */
const float *mynah_engine_pocket_latent(const mynah_engine_ctx *ctx,
                                        size_t index, size_t *out_count);

/* ------------------------------------------------------ batching self-check */

/*
 * The three seam properties that only real weights can test.
 *
 *  - `step_batch` is ATOMIC over the batch (E8-6): a refused call leaves every
 *    context exactly where it was, which is what makes the driver's
 *    `step_isolate()` legal.  Forced at BOTH points a step can refuse -- a
 *    non-finite latent, caught by the pre-flight, and a NaN in a cached key,
 *    which the attention softmax refuses inside the forward after the rows
 *    ahead of it have already advanced -- on both the batched and the per-row
 *    path, and checked against the identical set of requests that never saw the
 *    refusal.  Only the second injection is load-bearing: with the rollback
 *    deleted, the first still reports PASS.
 *  - `decode_audio_batch` is BIT-IDENTICAL per context (E8-4): the same frames
 *    decoded as a gang and decoded alone come out byte for byte the same, over
 *    ragged per-context ranges at different positions with different voices and
 *    seeds.
 *  - the text-chunk seam is where E2-5 says it is: a text three times over
 *    `max_tokens_per_chunk` prefills in ONE pass, so the backbone offset after
 *    `prepare` is the voice prefix plus every token.  It pins the ABSENCE of
 *    chunking, so that adding it here stops being something that can happen by
 *    accident.
 *
 * All three were, until now, tested only against the synthetic engine in
 * `tests/test_driver.c` -- whose step cannot fail inside a real graph and whose
 * codec carries no state, so none of them had ever been exercised where it can
 * actually break.  Needs a model pack, so it is not part of `--self-test`:
 * pass an opened model and it does the rest.  Runs at widths 2, 4, 8 and 16,
 * and is wired up as `mynah-tts --pocket-self-check MODEL_DIR`.
 *
 * Every bit-identity assertion below has only ever been taken on macOS with
 * clang.  It is NOT known to hold on the Linux ARM production target, where
 * `qmat`'s own batched-vs-single gate currently fails by about one ULP under
 * gcc with `-ffast-math`; until that is understood, a green run here is a
 * statement about this host and this compiler.
 *
 * Returns 0, or -1 with a message in `error`.
 */
int mynah_engine_pocket_self_check(const mynah_tts_model *model, char *error,
                                   size_t error_capacity);

#ifdef __cplusplus
}
#endif

#endif /* MYNAH_TTS_ENGINE_POCKET_H */
