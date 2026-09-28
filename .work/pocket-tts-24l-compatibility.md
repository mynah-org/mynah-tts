# Pocket-TTS 6L/24L checkpoint compatibility audit

Status: **verified from official safetensors headers and configs** · 2026-09-24

The matching pair is the official English release at revision
`492522650173a0653b7575cdc25ae09810e5d741`:

- small: `languages/english/model.safetensors`
- large: `languages/english_2026-04_24l/model.safetensors`

The files were compared by reading only the safetensors header (the first
1 MiB range for the remote files; no full checkpoint was needed for this
schema audit). The full 24L file was downloaded later only for the real
converter/CPU gate.

## Schema diff

| field | 6L | 24L |
|---|---:|---:|
| backbone layers | 6 | 24 |
| `d_model` | 1024 | 1024 |
| attention heads | 16 | 16 |
| head dimension | 64 | 64 |
| FFN dimension | 4096 | 4096 |
| fused QKV | `[3072, 1024]` | `[3072, 1024]` |
| attention output | `[1024, 1024]` | `[1024, 1024]` |
| FFN up/down | `[4096,1024]` / `[1024,4096]` | same |
| flow dimension | 512 | 512 |
| flow residual blocks | 6 | 6 |
| flow time conditions/frequencies | 2 / 128 | 2 / 128 |
| Mimi decoder transformer | 2L, d512, FFN2048, 8 heads | same |
| Mimi SEANet ratios | `[6,5,4]` | `[6,5,4]` |
| Mimi up/down stride | 16 / 16 | 16 / 16 |
| sample rate / frame rate | 24 kHz / 12.5 Hz | same |
| model tensor count | 214 | 358 |
| model parameter elements | 109,502,146 | 336,068,290 |
| checkpoint bytes | 219,029,196 | 1,304,206,614 |

The 24L delta is exactly 18 additional backbone blocks × 8 tensors = 144
tensors. The 214 common tensor names have identical shapes. There are no
model-weight tensors present only in the 6L file.

### Backbone layers 0..5

For every layer `0..5`, the name set and shapes are identical in both files:

```text
self_attn.in_proj.weight   [3072, 1024]
self_attn.out_proj.weight  [1024, 1024]
linear1.weight             [4096, 1024]
linear2.weight             [1024, 4096]
norm1.weight, norm1.bias   [1024]
norm2.weight, norm2.bias   [1024]
```

The values are not expected to be identical: the 24L checkpoint is a distinct
undistilled model. The header comparison proves layout/shape compatibility,
not weight-value identity.

### Dtype difference (the important non-layer storage difference)

The 6L model is all `BF16` (214/214 tensors). The 24L model is mixed:

```text
24L F32: 271 tensors = flow_lm + all 24 backbone blocks
24L BF16: 87 tensors = Mimi encoder/decoder/quantizer/upsample/transformers
```

All 127 common non-Mimi tensors change dtype from `BF16` in 6L to `F32` in
24L. The 87 common Mimi tensors remain `BF16`. This is a storage/precision
policy difference, not a graph difference. The converter now accepts mixed
source dtypes and can either normalize to `bf16`/`f32` or preserve each source
dtype with `--dtype source`.

### Flow, lookup, projections and Mimi

Every flow tensor name and shape is common to both files. In particular,
`flow_lm.conditioner.embed.weight` is `[4001,1024]`,
`flow_lm.input_linear.weight` is `[1024,32]`, `out_eos.weight` is `[1,1024]`,
and the flow input/final projections are `[512,32]` and `[32,512]`.

The official English 6L and 24L tokenizer files are both 59,339 bytes with
SHA-256
`d461765ae179566678c93091c5fa6f2984c31bbe990bf1aa62d92c64d91bc3f6`.
The lookup table therefore has the same 4,000-piece contract in this pair.

Mimi has the same tensor names/shapes and architecture in both model files:
the 2-layer decoder transformer, 2-layer encoder transformer, quantizer,
16× upsample, 16× downsample, and 3-stage SEANet decoder. The official YAML
configs independently state the same flow/Mimi dimensions and 24L backbone
depth; the tensor headers are the runtime authority.

### Voice-cache serialization difference

The matching official `alba.safetensors` voice state has:

```text
6L:  6 × (cache F32 [2,1,126,16,64] + offset I64[1]) = 12 tensors
24L: 24 × (cache F32 [2,1,126,16,64] + offset I64[1] + pad I64[1]) = 72 tensors
```

The `pad` metadata is present only in the 24L voice files. It is not a model
block weight and is ignored by the runtime; the converter preserves it while
validating every cache layer against the manifest. Cache layer count is now
derived from `transformer_layers`, not from a six-layer assumption.

## Runtime/converter consequence

One `pocket_config`/`Schema` is sufficient. The existing C runtime already
allocates and iterates the backbone using `cfg->layers`; the CUDA resident KV,
pointer tables, graph metadata and per-layer launch loop do the same. The
actual compatibility fixes required here are:

1. remove the converter's exact `{214,213}` total-tensor rejection;
2. retain malformed-checkpoint detection using the verified eight-tensors-per-
   backbone-block set plus the verified non-backbone graph count;
3. accept source `BF16`, `F32`, or mixed dtypes and convert per tensor;
4. add a `source` dtype policy so 24L CPU/CUDA can retain official F32 flow and
   backbone weights without permanent BF16 expansion;
5. test the actual 24L pack through C loading and inference before claiming
   24L CUDA support.

No separate `pocket_small()` or `pocket_large()` graph is justified by this
evidence.

Sources: [English 6L config](https://raw.githubusercontent.com/kyutai-labs/pocket-tts/main/pocket_tts/config/english_2026-09.yaml),
[English 24L config](https://raw.githubusercontent.com/kyutai-labs/pocket-tts/main/pocket_tts/config/english_2026-04_24l.yaml),
[official checkpoint repository](https://huggingface.co/kyutai/pocket-tts),
[upstream CLI language list](https://github.com/kyutai-labs/pocket-tts/blob/main/docs/CLI%20Commands/generate.md).
