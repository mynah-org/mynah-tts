# E3 — `engine_pocket.c`: PocketTTS in C

Status: **OPEN** · needs E1 (seam) and E2 (oracle)

Facts: [pocket-tts-model-facts.md](pocket-tts-model-facts.md). Do not restate
shapes here; read them from `model.json` at runtime.

## Converter and pack

`tools/convert_pocket.py` produces one pack **per language** — the schema is
byte-identical across languages, so only the tokenizer and the weight values
differ, and nothing in C branches on language.

```
model.json          engine:"pocket", all dims, ratios, strides, thresholds
tts.safetensors     flow_lm.* + mimi.decoder* (+ mimi.encoder* only if cloning)
tokenizer.model     SentencePiece Unigram, per language
voices/*.safetensors  KV-cache prefixes, F16
speakers.json       name, T frames, source dataset, license, commercial_use
source.json         repo, revision, sha256 per file
LICENSES/           CC-BY-4.0 + per-voice licenses (E6)
```

Decisions taken here:

- **Store voice KV in F16.** 6.4 MB/voice F32 → 3.2 MB, 83 MB for 26 voices.
  Verify against the oracle that F16 KV does not move the output audibly; if it
  does, keep F32 and say so in this note.
- **Ship `mimi.encoder*`, `mimi.downsample*` and `speaker_proj_weight`** (9.78M
  params, 19.6 MB BF16). Cloning is required — see
  [voice-cloning.md](voice-cloning.md) — so the source must be the official
  gated repo, not an ungated mirror.
- **Target the current schema** (214 tensors, 109.5M). `english_2026-01` is a
  second parity target only.
- `*_24l` variants are out of scope; they differ only in `num_layers`.

## C modules

| File | Contents |
|---|---|
| `src/engine_pocket.c` | graph composition, AR step, EOS, latent denorm |
| `src/flow_head.c` | `SimpleMLPAdaLN`: time embedding, adaLN modulation, 6 res-blocks, 1 LSD step |
| `src/seanet.c` | causal conv1d / transposed conv, residual blocks, streaming state |
| `src/tokenizer_sentencepiece.c` | Unigram + byte fallback |

Reusable unchanged from the shared layer after E1: attention, RoPE, KV cache,
matvec/matmat, quantization, threads, weights/mmap, the audio sink.

### New kernels needed

- **LayerNorm with bias.** PocketTTS uses LayerNorm, not RMSNorm. `src/kernels.c`
  has `layernorm`; confirm it handles the bias term and matches the oracle before
  assuming it does.
- **GELU tanh approximation**, matching `F.gelu(approximate="tanh")` exactly
  enough for the tolerance. `kernels.c` has a Padé `gelu` — check which one it
  is against the oracle.
- **Causal conv1d and transposed conv1d** with persistent streaming state.
- **adaLN modulation** (scale/shift/gate from a `[1536,512]` projection).

### Things that make this easier than it looks

- QKV is **already fused** in the checkpoint as `[3072,1024]` — one matvec, which
  is exactly what a fused-QKV kernel wants.
- FFN is **not gated**: plain `linear1` → GELU → `linear2`.
- One flow step per frame. No ODE integrator, no scheduler.
- No CFG at runtime: one forward per frame, batch 1.

## Streaming state

The engine owns, per request: backbone KV cache (seeded from the voice file),
SEANet conv ring buffers, decoder-transformer KV (context 250), current frame
index, EOS state, and the text chunker position.

**All of it must be per-context**, following `CLAUDE.md` rule 3. The streaming
left-context constant comes from the E2 measurement, not from a guess, and it is
an engine capability rather than a `#define`.

## Order of work

1. offline, single request, greedy, F32, scalar — parity stages 1-12
2. quantized weights — re-check parity, record the delta
3. streaming, single request — sample-identical to offline
4. streaming, batched through the shared slot driver
5. voice cloning from a wav — E7, required; last only in ordering

## Acceptance gate

- All E2 stages within tolerance, both revisions.
- Stream↔offline sample-identical for the same seed.
- `make self-test` covers the new kernels (LayerNorm+bias, causal conv,
  transposed conv, adaLN, Unigram tokenizer) model-free.
- Tokenizer round-trips a UTF-8 corpus exactly against SentencePiece.
- ubsan + leaks clean.
- WAV smoke for all 6 languages with explicit language/voice/seed.

## Resolved (were open questions)

All three were closed by reading the reference implementation; details and the
exact formulas are in
[pocket-tts-model-facts.md](pocket-tts-model-facts.md) §11.

- `conditioner.embed.weight` is `[4001, 1024]` because `nn.Embedding(n_bins + 1)`
  reserves **row 4000 as padding**. The tokenizer never emits it, and it adds no
  BOS/EOS of its own.
- `flow_net.time_embed.*.mlp.*.alpha` is an **RMSNorm gain**, with a
  non-standard RMSNorm: variance-based (mean-subtracted) and `unbiased=True`,
  i.e. `N-1`. Not the `rmsnorm` in `src/kernels.c`. **Write a separate kernel.**
- The head is **LSD with two time conditions** (`time_embed.0` and `.1`),
  `y = cond_embed(c) + (t_emb(s) + t_emb(t))/2`.

Two more traps found at the same time:

- **Two LayerNorms with different epsilons**: backbone `eps=1e-5` with bias,
  flow head `eps=1e-6`, `unbiased=False`. `final_layer.norm_final` has **no
  affine parameters** — no such tensor exists.
- `time_embed.*.freqs` are deterministic constants and are the only tensors
  identical across all languages. **Compute at load, do not store.**

## Resolved during the converter work

- **`offset` is the length, not a position vector.** It is `I64[1]` and the
  converter asserts it equals `T`, failing on a partially filled cache. (It is
  named `current_end` in the raw file and `offset` once loaded.) The KV loader
  has nothing to interpret.
- **`attention_heads` and `head_dim` are not derivable from the weights**,
  because QKV is pre-fused as `[3072, 1024]`. The only place the split is
  visible is the voice KV shape `[2, 1, T, 16, 64]`, so the converter reads a
  voice before deriving the schema and checks `heads * head_dim == hidden_dim`.
  `codec_transformer_heads` is derived as `codec_dim / head_dim = 8`, which is
  the only defensible derivation — **confirm it against the oracle before the C
  relies on it.**

## Still open

- **Whether F16 voice KV is audibly equivalent to F32.** Measured on conversion:
  **3.905e-03 absolute, 4.405e-04 relative** (worst voice `paul`). No overflow —
  peak magnitude is ~6.7, well inside F16 range — but that absolute error is
  large next to the 1e-4 tolerances used elsewhere, and it perturbs the *voice
  conditioning*, not an intermediate activation. **Verify against the oracle
  before accepting F16 as the default**; the number is printed on every
  conversion so it cannot be forgotten. Falling back to F32 costs 83 MB per
  pack.

## Multi-language packing

Measured: the six language models are **independently trained**, codec included
(relative L2 ≈ √2 on every probed tensor; only the two `freqs` buffers match).
So there is no shared codec and no shared latent space:

- one pack per language, ≈219 MB BF16 / ≈110 MB int8, nothing deduplicated
- a voice KV is valid **only** for the language model that produced it
- a process serving N languages holds N full weight sets resident

`model.json` therefore describes exactly one language. Multi-language is a
*deployment* concern, not a pack concern.

## E3-3 and E3-4 implemented — 2026-09-12

`src/flow_head.{c,h}` and `src/seanet.{c,h}`, wired into `CORE_SOURCES` and
`--self-test`. Parity against the oracle with the real BF16 weights:

| stage | max abs error | tolerance |
|---|---|---|
| flow head, calls 0-3 | 5.4e-07 … **1.07e-06** | 1e-4 |
| `mimi.upsample` | **0.0** (bit-identical) | 1e-4 |
| SEANet decoder, calls 0-3 | 3.0e-07 … **5.7e-07** | 1e-3 |
| SEANet 4×1 frame vs 1×4 frames | **2.98e-07** | — |

The last row is the one that matters for streaming: carrying state per frame
reproduces a four-frame decode, which is E2-3's decision holding in C.

UBSan, ASan and ASan+UBSan clean on both the self-tests and the real-weight
parity; `leaks` reports zero.

Design points worth keeping:

- **Neither `mynah_layernorm_f32` nor `mynah_rmsnorm_f32` was reusable.** The
  first requires a non-NULL weight and `norm_final` has no affine parameters;
  the second is mean-square, not unbiased variance. The self-test **asserts that
  the two RMSNorms disagree**, so substituting one for the other fails the test
  instead of silently changing the output.
- The time-embedding branch is memoized on the bit pattern of the times, since
  `s=0` and `t=1` are constant for a whole utterance. The self-test checks both
  that a cache hit does not change the output and that a different time does.
- The transposed convolution reproduces `StreamingConvTranspose1d` **including
  the bias handling**: `y[:PT] += partial`, then `partial = y[-PT:] - bias`.
- The position counter lives in the state, advances by `encoder_stride ×
  n_latents`, resets with `_reset`, and is reachable only through
  `mynah_seanet_state_position`/`_advance` — so a caller that ignores it is
  ignoring it visibly rather than silently, which is the E2-3 failure mode.
- No tensor name is formatted inside either module; weights arrive as resolved
  `float *`.

### Open inconsistency to reconcile

`conv1d.h` (from E1 step 2) exposes
`mynah_conv1d_causal(const mynah_weights *file, ..., const char *weight_name)` —
it resolves weights **by name**, which is exactly the pattern
[engine-seam-refactor.md](engine-seam-refactor.md) risk 6 warns against. The new
modules take resolved pointers. Both cannot be the shared form; the
pointer-taking one is right, and `conv1d` should be narrowed to match when the
vtable lands.

## Backbone implemented — 2026-09-12

`src/transformer_ar.{c,h}`, the shared causal AR transformer, taking **resolved
weight pointers only** — it never formats or looks up a tensor name, which is
the shape [engine-seam-refactor.md](engine-seam-refactor.md) risk 6 asked for.

Parity against the oracle with the real BF16 weights, the `alba` voice KV loaded
as the 126-position prefix, 26 text tokens prefilled, then the AR step at
position 152:

| stage | max abs | rel L2 |
|---|---|---|
| layer 0, prefill | 2.61e-07 | 1.10e-06 |
| layer 5, prefill | 4.05e-06 | 1.25e-06 |
| 6 layers + `out_norm`, prefill | 6.44e-06 | 2.45e-06 |
| **6 layers + `out_norm`, step** | **3.40e-06** | 1.33e-06 |

Against a 1e-4 tolerance that is 13-1000× under. Layers 0 and 5 are exercised
**in isolation**, by configuring a one-layer transformer with that layer's
weights and feeding it the oracle's own input for that layer — no debug hooks
in the module. A scalar build with no SIMD and no fast-math stays within
7.7e-06. UBSan, ASan and zero leaks.

Decisions worth not undoing:

- **`prefill` is a loop of `step`.** For causal attention with a KV cache they
  are the same function: token *i* writes its own K/V then attends `[0, offset+i]`,
  which is what a masked batched SDPA computes. One implementation (rule 7); a
  GEMM prefill is a future optimization, not a second graph. The side effect is
  that KV continuity is **bit-exact**, not within tolerance.
- **No NaN sentinel inside the module.** Validity is an integer `offset`,
  attention reads only `[lo, offset+i]`, and the buffer is `calloc`ed. NaN is
  *rejected* at three boundaries: loading a KV prefix (where `_expand_kv_cache`
  padding would arrive), the input to prefill/step, and the softmax scores. The
  BOS substitution stays with the caller, before `input_linear`.
- **The KV layout was verified against a real voice file, not assumed.**
  `alba.safetensors` holds `[2, 1, 126, 16, 64]` F32 and the block layout is
  `[2][max_seq][heads][head_dim]`, so loading a voice is two `memcpy` per layer
  with no conversion.
- **RoPE frequencies are computed in f32, as torch does, not in double.** At
  position ~150 a 1e-7 relative drift in the frequencies is already visible in
  `cos()`.
- `layer_scale == NULL` is **bit-identical** to a vector of ones, and the
  self-test checks that rather than accepting a tolerance.
- The RoPE self-test uses a case that **distinguishes interleaved from
  split-halves**; a split implementation would produce `{0,0,1,1}` instead of
  `{0,1,0,1}` and fail.

### Not yet exercised

The `context` sliding window is covered only by the model-free self-test. In the
oracle, Mimi's decoder transformer runs over 16 positions per frame, so
`context: 250` never bites and no reference data exercises it. Closing that needs
a dump longer than 250 frames — part of E2-4, still open.
