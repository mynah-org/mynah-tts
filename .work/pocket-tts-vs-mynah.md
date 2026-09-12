# mynah-tts as-is vs PocketTTS — what carries over and what does not

Status: **REFERENCE** (audit 2026-09-12)

Companion notes: [pocket-tts-model-facts.md](pocket-tts-model-facts.md) for the
verified PocketTTS numbers, [engine-seam-refactor.md](engine-seam-refactor.md)
for what to do about the gaps found here.

## 1. The one-line version

The runtime's *infrastructure* transfers almost completely. The runtime's
*model core* transfers not at all, because mynah-tts today is a discrete-codec
engine and PocketTTS is a continuous-latent one. `CLAUDE.md` already anticipated
exactly this case: *"Do not make Magpie's discrete codec API pretend to be a
universal latent API."*

## 2. Architecture diff

| | mynah-tts today (Magpie) | PocketTTS | Carries over? |
|---|---|---|---|
| Audio representation | 8 RVQ codebooks, size 2016, 21.5 fps | 32 continuous dims, 12.5 Hz | **no** |
| Per-frame generation | local transformer → logits per codebook | 1 transformer step + 1 MLP flow step | **no** |
| Frame stacking | yes (factor from config) | none | **no** |
| CFG | runtime, doubles cost | distilled into weights, absent at inference | **no** (simpler) |
| Text frontend | NeMo tokenizer + G2P + text normalization | SentencePiece Unigram 4k, no G2P | **no** (simpler) |
| Norm | RMSNorm | **LayerNorm with bias** | kernel missing |
| FFN | Magpie shape | plain 4×, GELU-tanh, non-gated | partly |
| QKV | separate projections | **pre-fused `[3072,1024]`** in the checkpoint | favours fused matvec |
| Attention | causal + cross-attention | causal only, RoPE | attention core yes |
| Audio decoder | NanoCodec, 22.05 kHz | Mimi/SEANet causal, 24 kHz | conv primitives only |
| Speaker | 5 baked-in IDs | 26 voices as **KV-cache prefixes** + zero-shot cloning | **no** |
| Params | 357M | **109.5M** | — |

## 3. mynah-tts: what is reusable as-is

Verified by reading the tree, not by reputation:

| Component | Where | Note |
|---|---|---|
| safetensors/GGUF loading, mmap, dtype conversion | `src/weights.c` + `third_party/ingot` | dtype-agnostic, handles bf16. Works unchanged. |
| WAV / PCM sink | `src/audio.c` | unchanged |
| Backend resolver (BLAS/scalar) | `src/backend.c`, `src/backend.h` | CPU half is model-agnostic |
| Thread pool | `src/threads.c` (158 LOC) | persistent, condvar, work-stealing; no spawn in the hot loop |
| HTTP server | `server/` (1009 LOC) | OpenAI-shaped routes, chunked PCM streaming, worker pool, bounded queue |
| Slot driver / continuous batching | `src/graph.c:3674-4283` | bit-exact per request, per-slot RNG, shared offline/streaming sink. Assumes only "AR steps → audio decoder", so it generalizes. |
| Streaming suffix decode | `src/graph.c:3907-3968` | bounded left context (`STREAM_CONTEXT_FRAMES 32`), constant cost per chunk. The idea transfers directly to SEANet. |

## 4. mynah-tts: what blocks a second engine

These are findings, each verified in the tree.

1. **The engine seam does not exist.** `PLAN.md` §4 designs `src/tts_engine.h`
   and `src/engine_magpie.c` with `prepare/step/flush/reset`. Neither file
   exists. `CLAUDE.md` names `src/engine_magpie.c` as a source of truth; it is
   not there. `info.engine` is read in `src/mynah_tts.c:214` and used only to
   print a string — **zero dispatch**.

2. **`graph.c` is the monolith `CLAUDE.md` forbids.** 4283 LOC, 51% of `src/`:
   generic primitives, Magpie encoder/decoder, NanoCodec, KV cache and the slot
   driver all in one translation unit. `PLAN.md:1011-1024` already carries this
   as an open TODO ("Move request state, AR stepping and codec state out of
   `graph.c`"), unstarted since July.

3. **The public header is Magpie-shaped.** `src/mynah_tts.h:17-41` exposes
   `codebook_count`, `codebook_size`, `frame_stacking_factor`,
   `audio_bos_id`/`audio_eos_id`, `local_transformer_layers`; `mynah_tts_request`
   carries `use_local_transformer`. This violates the `CLAUDE.md` rule "no Magpie
   dimensions in public headers" **today**, and a continuous-latent model has
   none of these fields.

4. **~60 Magpie tensor names are hardcoded** in `graph.c` (`text_embedding.weight`
   at :717, `audio_embeddings.%zu.weight` at :789, the NanoCodec snake alphas at
   :2374, …).

5. **The JSON parser cannot express a second engine.** `src/mynah_tts.c:112-163`
   is `strstr`-based, flat, no nesting, no arrays. A per-engine config block is
   not representable.

6. **The tokenizer has no SentencePiece.** `src/tokenizer.c:163-170` is five
   hardcoded modes (ByT5 bytes, IPA+G2P, Arabic, kana, pinyin). No BPE, no
   Unigram, no trie. PocketTTS needs Unigram + byte fallback, which is new code.

7. **The converter hardcodes the model.** `tools/convert_magpie.py:139-144`
   writes `codebook_count: 8`, `codebook_size: 2016`, `audio_bos_id: 2016`,
   `audio_eos_id: 2017` as literals rather than reading them.

## 5. Two defects found during this audit, unrelated to PocketTTS

- **The AVX-512 VNNI claim is not backed by code.** `README.md` and commit
  `724d677` attribute the EPYC Zen 5 int8 numbers to AVX-512 VNNI. `src/qmat.c`
  contains no `_mm512_*` and no `_mm256_dpbusd_epi32`; the int8 x86 path is
  `dot_q8_i32_avx2` (`qmat.c:117-134`) using `_mm256_cvtepi8_epi16` +
  `_mm256_madd_epi16`. `SIMD=avx512` (`Makefile:29-31`) only passes compiler
  flags. **The measured 0.427 is real; the attribution is not.**
- **Streaming is serialized process-wide.** `server/main.c:451-460` wraps the
  whole `stream_open/push/flush` in `pthread_mutex_lock(&g.synth_lock)`. One
  stream at a time per process, by design. For a 109M-param model this is the
  single biggest throughput limit — see
  [streaming-server-v2.md](streaming-server-v2.md).

## 6. Why PocketTTS is *easier* than Magpie, not harder

Worth stating plainly, because the instinct is the opposite for a "new
architecture":

- no RVQ, no codebooks, no delay pattern, no frame stacking, no local transformer
- no G2P, no NeMo text normalization, no `--normalized` escape hatch needed
- no CFG at inference: one forward per frame, batch 1
- the flow head is **one** MLP pass per frame (`sampler_decode_steps = 1`), not
  an ODE integration
- the architecture is fully published in YAML; nothing to reverse-engineer
- multiple reference implementations exist for parity (PyTorch, ONNX, Rust)
- 109.5M params against 357M

New C to write: SEANet causal conv stack + up/downsample, the AdaLN flow MLP,
LayerNorm-with-bias, and a SentencePiece Unigram tokenizer. The transformer,
RoPE, KV cache, attention, quantization, threading and the audio sink already
exist.

## 7. Rough compute budget (estimate — to be replaced by measurement)

Per 80 ms frame: backbone ≈ 6 × (1024×3072 + 1024×1024 + 2×1024×4096) ≈
**75.5M MAC**; flow head ≈ 7M MAC. At 12.5 fps that is ≈ **1 GMAC/s** before the
vocoder. Weight traffic at int8 ≈ 118 MB × 12.5 ≈ 1.5 GB/s per stream, against
~59 GB/s on an M1 — so many concurrent streams fit, and **concurrency, not
single-stream speed, is where the work is**.

Expect the SEANet decoder to dominate, as it does in qwen-tts (decoder 631 ms of
which conv stack 604 ms, ≈40% of wall). **Measure before optimizing the
transformer.**
