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
- **Omit `mimi.encoder*`, `mimi.downsample*`, `speaker_proj_weight` by default**
  (~20 MB). They are only needed to clone from a new wav. A pack without them is
  also a pack with no gated-weights question attached.
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
5. voice cloning from a wav (needs `mimi.encoder`) — **last, and optional**

## Acceptance gate

- All E2 stages within tolerance, both revisions.
- Stream↔offline sample-identical for the same seed.
- `make self-test` covers the new kernels (LayerNorm+bias, causal conv,
  transposed conv, adaLN, Unigram tokenizer) model-free.
- Tokenizer round-trips a UTF-8 corpus exactly against SentencePiece.
- ubsan + leaks clean.
- WAV smoke for all 6 languages with explicit language/voice/seed.

## Open questions

- `flow_lm.conditioner.embed.weight` is `[4001, 1024]` for a 4000-piece
  vocabulary. **What is row 4000?** Resolve before writing the lookup.
- `flow_net.time_embed.*.mlp.*.alpha` — an `alpha` parameter inside the time
  embedding MLP. Identify the activation it belongs to; do not guess it is Snake.
- `current_end [T]` in the voice files: confirm it is a position/length vector
  and not something the KV load must interpret.
