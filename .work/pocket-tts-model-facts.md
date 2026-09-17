# PocketTTS — verified model facts

Status: **REFERENCE** (read from the weights, 2026-09-12)

Everything below was read out of the safetensors headers and the SentencePiece
protobuf, not from documentation. Where a fact comes from upstream docs instead,
it says so. Re-verify against the pack the converter produces before any C code
depends on a number here.

Source of truth on disk:
`~/.cache/huggingface/hub/models--kyutai--pocket-tts/snapshots/4925226501…`
(239 files, 2.4 GB, `--exclude "languages/*_24l/*"`).

## 1. What it is

Kyutai Pocket TTS. **Continuous-latent autoregressive**, not codec-based:
`mimi.quantizer` is a `DummyQuantizer`, a pass-through Conv1d 32→512. There are
no codebooks, no RVQ, no delay pattern, no frame stacking, no local transformer.

Paper: "Continuous Audio Language Models" (CALM), arXiv:2509.06926.
Code: <https://github.com/kyutai-labs/pocket-tts> (MIT).
Weights: <https://huggingface.co/kyutai/pocket-tts> (CC-BY-4.0, gated).

| | |
|---|---|
| Latent | **32 continuous dims @ 12.5 Hz** |
| Frame | 1920 samples = **80 ms** @ 24 kHz mono |
| Backbone | 6 layers, `d_model` 1024, 16 heads (head_dim 64), FFN 4096 |
| Head | `SimpleMLPAdaLN`, 6 res-blocks, dim 512 |
| Decode steps | **1** (`DEFAULT_SAMPLER_DECODE_STEPS = 1`, LSD-distilled) |
| CFG | distilled into the weights — **no CFG at inference**, batch 1, one forward/frame |
| Codec | Mimi, SEANet ratios `[6,5,4]`, +16× up/downsample, decoder-transformer 2L d512 `context 250` |
| EOS | `flow_lm.out_eos` linear → 1 logit, threshold `-4.0` |
| Default temperature | 0.7 (`english.yaml` overrides to 0.3) |

## 2. Tensor schema — only two generations exist

| Group | Checkpoints | Tensors | Params |
|---|---|---|---|
| **current** | `english`, `english_2026-04`, `german`, `italian`, `portuguese`, `spanish` | 214 | **109.5M** |
| old | `english_2026-01` == `tts_b6369a24.safetensors` | 213 | 117.9M |

All BF16. `english` and `english_2026-04` are the same file.

**`english_2026-04` vs `italian`: zero differences.** Same 214 names, same
shapes, same dtypes. This is the load-bearing fact for the pack design: one
engine, one graph, and a per-language pack in which only the tokenizer and the
weight *values* change. No language branching anywhere in C.

Diff old → current, in full:

```
+ flow_lm.bos_before_voice          [1, 1, 1024]        (new)
~ flow_lm.speaker_proj_weight       [1024, 512]    -> [1024, 32]
~ mimi.downsample.conv.conv.weight  [512, 512, 32] -> [32, 512, 32]
```

`mimi` drops 27.9M → 20.1M params and the entire drop is that one conv. Voice
conditioning now projects from the 32-dim latent space instead of 512.
Consistent with `insert_bos_before_voice: true` in the current configs.

**Build against the current schema.** The old one is only useful as a second
parity target, and it is what the ungated mirror ships.

## 3. Tensor names (current generation, collapsed)

```
flow_lm.conditioner.embed.weight              [4001, 1024]
flow_lm.bos_emb                               [32]
flow_lm.bos_before_voice                      [1, 1, 1024]
flow_lm.emb_mean / emb_std                    [32]          latent normalization
flow_lm.input_linear.weight                   [1024, 32]
flow_lm.speaker_proj_weight                   [1024, 32]
flow_lm.transformer.layers.{0..5}.self_attn.in_proj.weight   [3072, 1024]  QKV pre-fused
flow_lm.transformer.layers.{0..5}.self_attn.out_proj.weight  [1024, 1024]
flow_lm.transformer.layers.{0..5}.linear1.weight             [4096, 1024]
flow_lm.transformer.layers.{0..5}.linear2.weight             [1024, 4096]
flow_lm.transformer.layers.{0..5}.norm{1,2}.{weight,bias}    [1024]
flow_lm.out_norm.{weight,bias}                [1024]
flow_lm.out_eos.{weight,bias}                 [1,1024] / [1]
flow_lm.flow_net.input_proj.{weight,bias}     [512,32] / [512]
flow_lm.flow_net.cond_embed.{weight,bias}     [512,1024] / [512]
flow_lm.flow_net.time_embed.{0,1}.freqs       [128]
flow_lm.flow_net.time_embed.{0,1}.mlp.*       [512,256] / [512] / alpha [512]
flow_lm.flow_net.res_blocks.{0..5}.in_ln.{weight,bias}       [512]
flow_lm.flow_net.res_blocks.{0..5}.adaLN_modulation.*        [1536,512] / [1536]
flow_lm.flow_net.res_blocks.{0..5}.mlp.{0,2}.*               [512,512] / [512]
flow_lm.flow_net.final_layer.adaLN_modulation.*              [1024,512] / [1024]
flow_lm.flow_net.final_layer.linear.*                        [32,512] / [32]
mimi.quantizer.output_proj.weight             [512, 32, 1]
mimi.upsample.convtr.convtr.weight            [512, 1, 32]
mimi.decoder_transformer.transformer.layers.{0,1}.*   d512, layer_scale, in_proj [1536,512]
mimi.decoder.model.*.conv.weight              [512,512,7]
mimi.decoder.model.*.convtr.weight            [512,256,12]  (3 stages)
mimi.decoder.model.*.block.*.conv.weight      [128,256,3]
mimi.encoder.*                                cloning-only, can be omitted
mimi.downsample.conv.conv.weight              [32,512,32]   cloning-only
```

FFN is **not** gated (plain `linear1`→GELU-tanh→`linear2`). LayerNorm has a
**bias**, unlike the RMSNorm used by Magpie and Qwen — the existing `rmsnorm`
kernel is not a substitute.

## 4. Voices

Two different formats, which is easy to get wrong:

| Set | Content |
|---|---|
| `embeddings/` (v1, old gen only) | `audio_prompt [1, T, 1024]` F32 — a projected prefix |
| `embeddings_v2/`, `embeddings_v3/`, `languages/*/embeddings/` | **the KV cache itself**: 6 layers × `[2, 1, T, 16, 64]` F32 + `current_end [T]` |

`[2, 1, T, 16, 64]` = `[K/V, batch, T, heads, head_dim]`, which independently
confirms `d_model = 16 × 64 = 1024`. `T` ranges 106-137 frames = **8.5-11 s** of
reference audio at 12.5 Hz.

**26 voices in every language, same names across all six** — the KV is
recomputed per language model. The 8 originals (alba, azelma, cosette, eponine,
fantine, javert, jean, marius), 13 newer LibriVox-style ones (anna, bill_boerst,
caro_davy, charles, eve, george, jane, mary, michael, paul, peter_yearsley,
stuart_bell, vera) and the 5 per-language defaults (estelle/fr, giovanni/it,
juergen/de, lola/es, rafael/pt).

Size: 6 × 2 × ~130 × 16 × 64 × 4 B ≈ **6.4 MB/voice in F32**. That is why the
per-language embeddings alone are 1.16 GB. **Store them F16 in the pack**:
3.2 MB/voice, 83 MB for 26.

Because the voice *is* the KV cache, loading a voice needs no prefill at all.
This is a real TTFA advantage over an engine that must encode a reference clip.

## 5. Tokenizer

**SentencePiece Unigram, vocab 4000, `byte_fallback = 1`** (read from the
`trainer_spec` protobuf: `model_type = 1`). One per language, 59-61 KB; the
English one is byte-identical across all three English revisions, and every
other language differs.

The embedding table is `[4001, 1024]`: `nn.Embedding(n_bins + 1, dim)` where the
extra row is **padding** (`text_conditioner.py`, `LUTConditioner.__init__`). It is
never produced by the tokenizer. Encoding is a plain `sp.encode(text)` — **no BOS
or EOS token is added** by the conditioner.

No G2P, no text normalization, no espeak: text goes in as a prefix through the
LUT conditioner. This removes the single most unpleasant part of the Magpie
contract.

## 6. Languages actually available

`english` (= `english_2026-04`), `english_2026-01`, `german`, `italian`,
`portuguese`, `spanish`.

**French has no base checkpoint** — it exists only as `french_24l`, a 672 MB
non-distilled preview. Same for the other `*_24l` variants
(`english_2026-04_24l` is 1.3 GB). These are `num_layers: 24` instead of 6 and
are explicitly labelled preview upstream. They are excluded from the local
download and are **out of scope**; the schema is otherwise the same, so they
cost nothing to add later.

## 7. Upstream numbers (their measurements, not ours)

| Metric | Value |
|---|---|
| Speed | ~6× real-time, MacBook Air M4, CPU |
| Cores | 2 |
| TTFA | ~200 ms |
| int8 dynamic (torchao) | 1.23-1.27× over fp32; 450 MB → 234 MB RSS |
| WER LibriSpeech test-clean | 1.84% |

Third-party ports worth reading, not vendoring: `PocketTTS.cpp` (C++ + ONNX
Runtime, claims 9.2× RTF / 30 ms TTFA on Ryzen 7 3800X int8),
`pocket-tts.cpp` (ggml, pre-alpha), sherpa-onnx (official support).
All of them depend on ONNX Runtime or ggml; none is dependency-free C.

## 8. Known upstream limitations

- The reference implementation is **not thread-safe** and its server does **not
  support concurrent requests** (stated upstream). Concurrency
  is therefore a genuine differentiator for us, not a rewrite for its own sake.
- On long inputs the model **skips parts of sentences**; upstream mitigation is
  chunking (`MAX_TOKEN_PER_CHUNK = 50`). This is a model property. Do not try to
  fix it in the runtime; reproduce the chunking behaviour.

## 9. Numbers that must live in `model.json`, never in code

`32` (latent dim), `1024`, `16`, `64`, `4096`, `6` (layers), `12.5`, `24000`,
`1920`, `16` (up/downsample stride), `4000`/`4001` (vocab), `[6,5,4]` (SEANet
ratios), `250` (decoder-transformer context), `512` (flow dim), `6` (flow
depth), `1` (decode steps), `-4.0` (EOS threshold), `0.7`/`0.3` (temperature).

## 10. Version and language divergence — measured

Tensor payloads were sha256'd and compared across all current-generation
checkpoints, and relative L2 distances measured on representative tensors.

**The six language models are independently trained, not fine-tuned from a
shared base.** Relative L2 distance from `english_2026-04` is ≈1.4 (that is,
≈√2, the value for two unrelated vectors of similar norm) on *every* tensor
probed, in the backbone, in the flow head, in the conditioner embedding **and in
the Mimi codec**:

| Tensor | de | it | pt | es |
|---|---|---|---|---|
| `mimi.decoder.model.0.conv.weight` | 1.372 | 1.391 | 1.464 | 1.526 |
| `mimi.decoder_transformer...out_proj.weight` | 1.449 | 1.452 | 1.465 | 1.444 |
| `flow_lm.transformer.layers.0...out_proj.weight` | 1.222 | 1.326 | 1.255 | 1.274 |
| `flow_lm.flow_net.res_blocks.0.mlp.0.weight` | 1.426 | 1.414 | 1.409 | 1.407 |
| `flow_lm.conditioner.embed.weight` | 1.561 | 1.564 | 1.553 | 1.555 |

Of 214 tensors, **exactly 2 are bit-identical across all six languages**:
`flow_lm.flow_net.time_embed.{0,1}.freqs`. Those are the deterministic sinusoidal
constants `exp(-log(10000) * arange(128) / 128)`.

**Correction, measured 2026-09-12 while implementing `src/flow_head.c`:** an
earlier version of this note said to compute them at load and not store them.
Computing them *exactly* is wrong. The checkpoint stores them in **BF16**, and
the reference implementation uses those rounded values, so an exact computation
diverges: 1.39e-03 on `freqs` itself, 9.79e-04 on the time embedding, and
**8.63e-05 on the emitted latent** — inside the 1e-4 tolerance, but consuming
the entire budget for one constant.

Rounding the computed value to BF16 (round-half-to-even) reproduces the
checkpoint tensor **bit for bit**, 0 mismatches out of 128, and drops the latent
error to 1.07e-06 — two orders of magnitude. So: compute at load, **then round
to BF16**. Either store them or round them; do not compute them in full
precision and assume that is more correct.

`flow_lm.emb_mean` / `emb_std`, the latent normalization, also differ per
language.

### Consequences — these are design constraints, not trivia

1. **Nothing is shareable between language packs, not even the codec.** A
   6-language deployment is 6 × 219 MB BF16, or ≈6 × 110 MB at int8. Budget for
   it; there is no "one codec, six heads" saving to be had.
2. **The latent space is per language.** The VAE differs and so do `emb_mean` /
   `emb_std`, so a latent, a voice KV, or a decoder state from one language is
   meaningless in another. This is why upstream ships a separate `embeddings/`
   directory per language for the *same* 26 voice names.
3. **Continuous batching cannot mix languages.** Slots batched together must
   share a language, because they share the weights being streamed. The server
   scheduler needs a per-language slot group — see
   [streaming-server-v2.md](streaming-server-v2.md).
4. `english` and `english_2026-04` are **byte-identical** — same blob, not just
   the same shapes. Ship one.

## 11. Reference-implementation details that the C code must match

Read out of `kyutai-labs/pocket-tts` (MIT), not guessed. These are the places a
from-scratch implementation silently diverges.

- **Two different LayerNorms, two different epsilons.**
  The backbone uses `torch.nn.LayerNorm(d_model, eps=1e-5)` (with weight *and*
  bias). The flow head uses a **custom** `LayerNorm` (`modules/mlp.py`) with
  `eps=1e-6` and `var(unbiased=False)`.
- **`FinalLayer.norm_final` has no affine parameters** (`elementwise_affine=False`),
  which is why no `flow_net.final_layer.norm_final.*` tensor exists. Do not
  invent one.
- **`time_embed.*.mlp.*.alpha` is an RMSNorm gain, and the RMSNorm is
  non-standard**: `y = x * alpha * rsqrt(eps + var(x))` where `var` is
  `torch.var(dim=-1)` — mean-subtracted **and** with torch's default
  `unbiased=True`, i.e. dividing by `N-1`. It is *not* mean-square RMSNorm and it
  is *not* the `rmsnorm` already in `src/kernels.c`. Getting this wrong produces
  a small, plausible-looking, entirely wrong output.
- **`num_time_conds = 2`** — the checkpoint has `time_embed.0` and `time_embed.1`,
  so the head is LSD with a start and a target time, integrated by `lsd_decode`.
  `flow_matching` (one time condition, `ot_decode`) is the other released option
  and is not what these weights are.
- **Flow head forward, exactly:**
  ```
  y = cond_embed(c) + (time_embed[0](s) + time_embed[1](t)) / 2
  x = input_proj(latent_noise)
  per res block:  shift, scale, gate = Linear(SiLU(y)).chunk(3)
                  h = mlp(in_ln(x) * (1 + scale) + shift)      # mlp = Lin,SiLU,Lin
                  x = x + gate * h
  final:          shift, scale = Linear(SiLU(y)).chunk(2)
                  out = linear(norm_final(x) * (1 + scale) + shift)
  ```
  Note `SiLU` is applied to `y` **before** the adaLN linear, and `modulate` is
  `x * (1 + scale) + shift`.
- **TimestepEmbedder:** `args = t * freqs`; `emb = cat([cos(args), sin(args)])`
  (cos first); then `Linear(256→512)`, `SiLU`, `Linear(512→512)`, `RMSNorm`.
- **Backbone FFN activation** is `F.gelu(x, approximate="tanh")`. `src/kernels.c`
  has a Padé approximation — check it against the oracle rather than assuming the
  two agree to tolerance.

## 12. Is the dataflow reusable across languages and versions? — **yes, entirely**

Asked directly: do the cross-language and cross-version differences mean new
kernels or a different graph? **No. Neither axis costs a single new kernel.**

### Across languages: same graph, different numbers

The tensor schema is identical across all six languages (§2). What varies is
only: the weight *values*, `emb_mean`/`emb_std`, the `tokenizer.model` file, and
the voice KV files. All of it is data the converter writes into the pack.

**Cost of adding a language to a working runtime: zero C code.** Run the
converter, ship a pack. The cost is memory and deployment (§10), not
implementation.

### Across versions: the difference is confined to the cloning path

The three tensors that differ between `english_2026-01` and the current
generation — `speaker_proj_weight`, `bos_before_voice`,
`mimi.downsample.conv.conv.weight` — are used in exactly one place, verified in
`models/tts_model.py`:

```
get_state_for_audio_prompt(voice)
├── predefined voice or .safetensors  -> _import_model_state()   # load KV, done
└── a .wav                            -> _encode_audio()          # cloning only
                                          mimi.encode_to_latent   (uses downsample)
                                          F.linear(.., speaker_proj_weight)
                                          cat([bos_before_voice, prompt])  if configured
                                          prefill through flow_lm -> KV
```

A predefined voice **loads a KV cache straight from the file**: no Mimi encoder,
no `speaker_proj`, no `bos_before_voice`, no prefill at all.

So for generation with the shipped voices — which is the whole product — the
**graph is identical between the two generations**. Supporting both revisions is
free. The version difference only becomes visible if E3-10 (clone from a wav)
is implemented, and even then it is two config values (`speaker_proj` input dim
512 vs 32, `insert_bos_before_voice` true/false), not new code paths.

### What is genuinely new work

Not the language axis and not the version axis, but the **architecture** axis:
continuous latents instead of codebooks, the AdaLN flow head, causal SEANet, the
variance-RMSNorm, SentencePiece Unigram. Those are E3, and they are written once
and then serve every language and both revisions.

### Two implementation details this uncovered

- **NaN is a sentinel.** `flow_lm.forward` starts with
  `sequence = torch.where(torch.isnan(sequence), self.bos_emb, sequence)` — NaN
  positions mean BOS. `_expand_kv_cache` likewise **fills unused KV positions
  with NaN**. A C implementation must either reproduce the sentinel or track
  validity explicitly; it must not let NaN reach a matmul.
- **A voice KV is bound to the weights that produced it.** Upstream refuses
  predefined voices on custom weights, saying the model then "typically never
  emits EOS". This is the §10 point with upstream's own words behind it: never
  let a voice file cross a model boundary, and make the pack enforce it.
