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

Note the embedding table is `[4001, 1024]` — 4000 pieces plus one extra row.
Resolve what row 4000 is before writing the lookup.

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
  support concurrent requests** (stated in the repo's `AGENTS.md`). Concurrency
  is therefore a genuine differentiator for us, not a rewrite for its own sake.
- On long inputs the model **skips parts of sentences**; upstream mitigation is
  chunking (`MAX_TOKEN_PER_CHUNK = 50`). This is a model property. Do not try to
  fix it in the runtime; reproduce the chunking behaviour.

## 9. Numbers that must live in `model.json`, never in code

`32` (latent dim), `1024`, `16`, `64`, `4096`, `6` (layers), `12.5`, `24000`,
`1920`, `16` (up/downsample stride), `4000`/`4001` (vocab), `[6,5,4]` (SEANet
ratios), `250` (decoder-transformer context), `512` (flow dim), `6` (flow
depth), `1` (decode steps), `-4.0` (EOS threshold), `0.7`/`0.3` (temperature).
