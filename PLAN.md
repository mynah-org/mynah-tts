# Mynah TTS — implementation plan

> **Mynah TTS** is a small, native C runtime for codec-based and continuous-AR
> TTS models. It is CPU-first, portable across macOS/Linux, and uses Metal or
> CUDA only when explicitly requested. The first engine is
> `nvidia/magpie_tts_multilingual_357m`; **PocketTTS (Kyutai) is the second**.

Status: Magpie v1 is implemented and shipping — model pack, converter, CPU
SIMD/scalar kernels, NanoCodec synthesis, WAV writer, HTTP server with continuous
batching, and optional Metal/CUDA matmul backends. Oracle parity is closed.
Current work is the **second engine**: PocketTTS, a continuous-latent AR model,
which requires the engine seam that has been outstanding since July.

**Current focus is CPU — ARM and x86 together. GPU work is deferred.**

One measured fact shapes several epics: the six PocketTTS language models are
**independently trained and share nothing**, codec included. One pack per
language, no deduplication, voices valid only for the model that produced them,
and continuous batching cannot mix languages.

The graph, however, is **fully shared**: identical across all six languages, and
identical across both checkpoint generations for generation with the shipped
voices. Adding a language costs zero C code; the three tensors that differ
between revisions live only in the clone-from-wav path (E3-10).

## How this plan is organised

`PLAN.md` is a **board**: one line per work item, with a link to the note under
[`.work/`](.work/) that carries the detail. Detail does not belong here — when a
task is picked up, read its note. Session logs, measurements and post-mortems go
in `.work/`; durable performance results go in `docs/performance.md`.
See [`.work/README.md`](.work/README.md) for the convention.

Sections 1-16 below are the durable contract: mission, model contract, repository
shape, kernels, oracle policy, gates. They change only when a decision changes.

## 0. Work board

Legend: `[ ]` open · `[~]` in progress · `[x]` done · `[-]` dropped

### Reference — read before starting anything

- [x] PocketTTS facts verified from the weights, incl. cross-language divergence
      and the reference-implementation traps → [`.work/pocket-tts-model-facts.md`](.work/pocket-tts-model-facts.md)
- [x] Architecture diff, mynah-tts as-is vs PocketTTS → [`.work/pocket-tts-vs-mynah.md`](.work/pocket-tts-vs-mynah.md)
- [x] Magpie-era checkpoints and rejected ideas → [`.work/archive-2026-07-checkpoints.md`](.work/archive-2026-07-checkpoints.md)

### E1 — Engine seam and the `graph.c` split → [`.work/engine-seam-refactor.md`](.work/engine-seam-refactor.md)

Blocks E3. Do it *with* PocketTTS in hand, not before.

**Progress**: `graph.c` (4283 LOC) is gone. Steps 1-4 done, each verified against
the goldens with byte-identical audio:
`mynah_util` + kernels → `conv1d` → `codec_nanocodec` → `engine_magpie.c` +
`inference.c`. `src/` is nine files, largest 2186 lines.

- [ ] E1-1 `src/tts_engine.h`: vtable + capability block. The twelve functions
      `inference.c` calls on the engine are already the list; abstract them.
- [x] E1-2 slot driver lifted into `src/inference.c` (650 LOC)
- [ ] E1-3 lift streaming state into `src/stream.c`; `decode_audio` takes contiguous
      monotonic ranges and the engine owns continuity (E2-3), so left-context stays internal
- [x] E1-4 done differently and deliberately: there is no shared `transformer.c`,
      because those functions format NeMo tensor names. They stayed in
      `engine_magpie.c`; the genuinely shared parts became `conv1d` and `kernels`.
      A shared transformer needs functions taking **resolved weight pointers** —
      that is E3 work, not a move
- [ ] E1-5 remove Magpie fields from `src/mynah_tts.h` (breaking API change)
- [ ] E1-6 real JSON parser with nesting and arrays, replacing `mynah_tts.c:112-163`
- [ ] E1-7 split the converter into a shared pack writer + per-engine metadata
- [ ] E1-8 gate: Magpie stream↔offline output sample-identical to the pre-refactor binary
- [x] E1-0 **prerequisite done**: `tools/make_fake_pack.py` + `tests/refactor_goldens.sh`
      + `make fake-pack | goldens-capture | goldens`. 8 goldens captured;
      `make stream-test` and the full 15-check `make server-test` both pass against it

### E2 — PocketTTS oracle → [`.work/pocket-tts-oracle.md`](.work/pocket-tts-oracle.md)

Independent of E1. **Start here** — it tells E1 which state the seam must model.

- [ ] E2-1 `tools/oracle_pocket.py`: dump all 12 stages, tokenizer → waveform
- [x] E2-1a `tools/oracle_pocket.py` + `make oracle-pocket` written
- [x] E2-1b `tools/oracle_pocket_tokenizer.py` + `tools/corpus_pocket.py`: ~9000 cases
      per language, hex-encoded so invalid UTF-8 survives, deterministic by seed
- [ ] E2-2 `tests/parity_pocket.py` with per-stage tolerances
- [x] E2-3 streaming decode measured: **carry codec state** (error 1e-7 down to a
      1-frame chunk); replaying context needs 64 frames / 5.12 s for exactness, so
      the Magpie replay strategy does not transfer
- [ ] E2-4 dump decoder-transformer behaviour past `context: 250`
- [ ] E2-5 dump the text-chunk seam at `MAX_TOKEN_PER_CHUNK = 50`, including the known skip bug

### E3 — `engine_pocket.c` → [`.work/pocket-tts-engine.md`](.work/pocket-tts-engine.md)

Needs E1 and E2.

- [ ] E3-1 `tools/convert_pocket.py` + per-language model pack; voice KV stored F16
- [x] E3-2 `src/tokenizer_sentencepiece.{c,h}` **done** → [`.work/tokenizer-sentencepiece.md`](.work/tokenizer-sentencepiece.md)
      1734 LOC, in `CORE_SOURCES` and `--self-test`; `make tokenizer-parity` replays
      45,197 oracle cases / 2.1M ids across 5 languages; UBSan, ASan and leaks clean
- [ ] E3-3 `src/flow_head.c`: time embedding, adaLN, 6 res-blocks, 1 LSD step
- [ ] E3-4 `src/seanet.c`: causal conv1d / transposed conv + streaming state
- [ ] E3-5 `src/engine_pocket.c`: AR step, EOS at `-4.0`, latent denorm
- [ ] E3-6 new kernels self-tested model-free: LayerNorm **with bias** (two different
      epsilons), **variance-based RMSNorm** (`unbiased=True`, *not* `kernels.c:rmsnorm`),
      GELU-tanh, causal conv, adaLN
- [ ] E3-7 offline parity across all 12 oracle stages, both checkpoint generations
- [ ] E3-8 streaming sample-identical to offline, then batched through the shared driver
- [ ] E3-9 WAV smoke for all 6 languages with explicit language/voice/seed
- [ ] E3-10 **voice cloning from a wav — required, not optional** (see E7)
- [ ] E3-11 one pack = one language; compute `time_embed.*.freqs` at load instead of storing
- [ ] E3-12 NaN-as-BOS sentinel: reproduce it or track validity explicitly — never let NaN reach a matmul
- [ ] E3-13 pack refuses a voice KV from a different model/revision (upstream: it then never emits EOS)

### E7 — Voice cloning → [`.work/voice-cloning.md`](.work/voice-cloning.md)

Zero-shot cloning is a product requirement. The weights are already in the pack
(`mimi.encoder*` + `downsample` + `speaker_proj`, 9.78M params / 19.6 MB BF16).

- [ ] E7-1 SEANet **encoder** in C — mirrors the decoder, reuses its conv kernels
- [ ] E7-2 encoder transformer (2L d512) — reuses the shared attention
- [ ] E7-3 `ConvDownsample1d`: stride 16, kernel 32, `pad_mode="replicate"`
- [ ] E7-4 `speaker_proj` + optional `bos_before_voice`, config-driven — the only
      place the two checkpoint generations differ
- [ ] E7-5 prefill the reference latents through the backbone to produce the voice KV
- [ ] E7-6 WAV/audio input decode; require 24 kHz mono first
- [ ] E7-7 polyphase resampler to 24 kHz matching `scipy.signal.resample_poly`
      within tolerance — needed for arbitrary input files
- [ ] E7-8 truncate reference audio to 30 s, as upstream does
- [ ] E7-9 `mynah-tts export-voice`: serialize the KV so reload is instant
- [ ] E7-10 consent gate and notice before cloning — see E6

### E4 — CPU kernels, ARM and x86 in one step → [`.work/cpu-kernels-arm-x86.md`](.work/cpu-kernels-arm-x86.md)

**No kernel is done until both ISAs exist and self-test. GPU deferred.**

- [ ] E4-1 **correct the AVX-512 VNNI claim** in `README.md` — the code is AVX2 only
- [ ] E4-2 port `dispatch` (`--dispatch-map`) and `costmap` from qwen-tts
- [ ] E4-3 baseline per-region profile of the PocketTTS path, on M1 and on EPYC
- [ ] E4-4 thread pool upgrade: lane split, deadline priority, `after_fork` (prereq for E5-6)
- [ ] E4-5 the kernel the profile names — scalar reference, then NEON/SDOT/i8mm **and** AVX2/AVX-512/VNNI in one change
- [ ] E4-6 int8 weight prepack with persistent cache, both ISAs
- [ ] E4-7 AMX-INT8 (Linux/x86 only), last
- [ ] E4-8 `Makefile`: `SIMD=` profiles + `ARCH_STAMP` rebuild-on-flag-change

### E5 — Streaming server v2 → [`.work/streaming-server-v2.md`](.work/streaming-server-v2.md)

Supersedes §24 (P0-P3, all landed). The limit now is concurrent streaming.

- [ ] E5-1 **remove the global stream mutex** (`server/main.c:451-460`) — scheduler owns `ctx`
- [ ] E5-2 streaming requests enter the same slot driver as offline; no second path
- [ ] E5-3 async output writer: dedicated thread, bounded queue, send timeout, counters
- [ ] E5-4 cancel on disconnect (`POLLRDHUP`), slot freed within one frame
- [ ] E5-5 long-form: incremental push into a running decode, persistent conv state across flushes
- [ ] E5-6 prefork pinned on Linux (`SCM_RIGHTS`, CoW after weight load) — needs E4-4
- [ ] E5-7 `playback_sim` + soak; publish p50/p95 TTFA, prebuffer, stall rate
- [ ] E5-9 **per-language slot groups** — batching cannot mix languages; decide before E5-1
- [ ] E5-8 gate: **N concurrent streams byte-identical to the same request run alone**

### E6 — Licensing and voice policy → [`.work/licensing-and-voice-policy.md`](.work/licensing-and-voice-policy.md)

- [ ] E6-1 `speakers.json`: `source_dataset`, `license`, `commercial_use` per voice
- [ ] E6-2 default pack ships CC0/CC-BY voices only — `jean` and `cosette` are **non-commercial**
- [ ] E6-3 verify the `embeddings_v3` voice→dataset mapping (assumed CC0, unconfirmed)
- [ ] E6-4 `LICENSES/` + `NOTICE`: CC-BY attribution, modification statement, prohibited-use as notice
- [ ] E6-5 `source.json` pins the official Kyutai repo, never an ungated mirror
- [ ] E6-6 CLI and `/v1/voices` expose the license; `--commercial-only` filter

### Deferred

- [-] GPU (Metal/CUDA) work — existing backends stay as they are. Metal measured
  *slower* than CPU on Apple Silicon (`docs/performance.md:71-80`).
- [-] `*_24l` PocketTTS variants — non-distilled previews, 672 MB-1.3 GB each,
  same schema with `num_layers: 24`. French exists **only** in this form.
- [-] dots.tts (§9) and Chatterbox (§10) — unchanged as later engines.

## 1. Product mission and soul

Mynah TTS is not a NeMo fork or a general tensor framework. It follows the
working style of `../mynah` ASR and `../qwen-tts`: a small C11 runtime that
loads converted weights and runs without Python at runtime.

Non-negotiable principles:

- C11 is the center: cohesive modules, a small stable C API, and no Python or
  C++ in the default CPU path.
- CPU is the product. `make` must build a useful macOS/Linux binary without a
  GPU SDK; Accelerate/OpenBLAS are optional accelerators.
- Offline, streaming, and long-form synthesis share one state machine. Only the
  audio sink changes.
- Model dimensions, tokenizer behavior, codec layout, and sampling defaults
  come from `model.json`, never from Magpie-specific public constants.
- Every new stage is compared with a NeMo/PyTorch oracle before SIMD,
  quantization, or GPU work.
- The hot loop reuses mmap-backed weights and scratch memory; it does not parse,
  allocate, or load files.
- Metal and CUDA are opt-in, resident-aware, and able to fall back to CPU with a
  clear diagnostic.
- Runtime and model weights are separate. Weights, checksums, tokenizer assets,
  and licenses belong to a model pack and are never committed.

| Surface | Name |
|---|---|
| Repository | `mynah-tts` |
| Library | `libmynah_tts` |
| CLI | `mynah-tts` |
| Future server | `mynah-tts-server` |

## 2. v1 target: MagpieTTS

Primary reference: [NVIDIA MagpieTTS Multilingual 357M](https://huggingface.co/nvidia/magpie_tts_multilingual_357m).
The pinned v2607 release is approximately 364M parameters, covers 12 languages
(`ar`, `de`, `en`, `es`, `fr`, `hi`, `it`, `ja`, `ko`, `pt`, `vi`, `zh`), has five
baked-in voices, and produces mono audio at 22.05 kHz. It must be treated as a
different checkpoint from older releases.

The initial conceptual graph is:

```text
UTF-8 text
  -> text normalization / G2P / language tokenizer
  -> causal text encoder
  -> causal decoder with self- and cross-attention
  -> frame stacking
  -> local multi-codebook transformer
  -> NanoCodec decoder
  -> mono float/PCM WAV at 22050 Hz
```

Reference codec: [NVIDIA NanoCodec 22 kHz](https://huggingface.co/nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps).
The initial model card describes 8 codebooks, size 2016, embedding dimension
32, 21.5 frames/s, and FSQ levels `[8, 7, 6, 6]`. Mynah v1 needs the decoder;
the codec encoder is offline tooling only.

These values are hypotheses until conversion confirms them. The checkpoint and
the generated `model.json` are authoritative for layer counts, dimensions,
token IDs, local-transformer mode, KV layout, CFG, and codec state.

v1 supports Magpie's baked-in speakers. It does not promise zero-shot cloning.
Text normalization and G2P must either be exported as deterministic assets or
be made explicit with a `--normalized` contract; the runtime must not silently
replace NeMo text behavior.

## 3. Patterns to reuse

From `../mynah` ASR:

- modular C11 layout (`src`, `cli`, `tools`, `tests`, optional `server`);
- converted model packs, manifests, mmap weights, and explicit model skips;
- an engine/vtable boundary for multiple families;
- Python only for conversion, oracle dumps, inspection, and evaluation via `uv`;
- `make` as the main interface and all artifacts below `build/`;
- CPU/Metal/CUDA capability reporting with safe fallback;
- stage-level numerical parity, UBSan, and native macOS leak checks.

From `../qwen-tts`:

- safetensors views, aligned scratch, and separate kernels for matvec, matmat,
  norms, attention, softmax, sampling, and convolution;
- GCD on macOS and persistent pthread workers on Linux, with the main thread
  participating and no hot-loop allocation;
- `--caps`, deterministic self-tests, and reproducible benchmark reports;
- quantization only after a correct floating-point baseline;
- GPU offload of resident subgraphs rather than host/device copies per op;
- a separable causal audio decoder that can be pipelined later.

Do not copy monolithic source files, placeholder translation units, hardcoded
Qwen dimensions, divergent offline/streaming implementations, or unmeasured
performance claims.

## 4. Repository shape

```text
mynah-tts/
├── src/
│   ├── mynah_tts.h/.c       # public API, model/context lifecycle
│   ├── tts_engine.h         # engine vtable and capabilities
│   ├── engine_magpie.c      # Magpie graph composition
│   ├── text.c/.h            # UTF-8, chunking, special tokens
│   ├── tokenizer.c/.h       # exported tokenizer assets and IDs
│   ├── transformer.c/.h     # shared transformer blocks
│   ├── attention.c/.h       # causal/cross attention and KV cache
│   ├── sampler.c/.h         # greedy, temperature, top-k/top-p, RNG
│   ├── codec_decoder.c/.h   # NanoCodec FSQ and causal decoder
│   ├── audio.c/.h           # WAV/PCM and audio sinks
│   ├── weights.c/.h         # mmap model pack and tensor views
│   ├── kernels.c/.h         # scalar reference kernels
│   ├── kernels_neon.c       # optional ARM kernels
│   ├── kernels_avx.c        # optional x86 kernels
│   ├── threads.c/.h         # cross-platform worker API
│   ├── backend.c/.h         # CPU resolver and capability reporting
│   ├── metal_backend.m      # only in `make metal`
│   └── cuda_backend.cu      # only in `make cuda`
├── cli/main.c
├── tools/
│   ├── inspect_nemo.py
│   ├── convert_magpie.py
│   ├── convert_dots.py
│   ├── convert_chatterbox.py
│   └── oracle/               # offline NeMo/PyTorch reference dumps
├── reference/                # small configs, reports, and tokenizer metadata
├── models/                   # ignored local checkpoints and model packs
├── tests/
├── docs/
├── Makefile
├── CLAUDE.md
├── AGENTS.md
└── PLAN.md
```

Do not create an all-purpose `tensor.c`. A small typed view (`data`, `dtype`,
`ndim`, `shape`, `stride`) is enough to describe existing buffers; hot
operations call explicit kernels so CPU, Metal, and CUDA share a clear seam.

The engine boundary should stay small:

```c
typedef struct {
    int streaming;
    int longform;
    int baked_speakers;
    int local_transformer;
    int codec_decoder;
} mynah_tts_caps;

typedef struct mynah_tts_engine {
    const char *name;
    mynah_tts_caps caps;
    int  (*prepare)(mynah_tts_context *, const mynah_tts_request *);
    int  (*step)(mynah_tts_context *);
    int  (*flush)(mynah_tts_context *);
    void (*reset)(mynah_tts_context *);
} mynah_tts_engine;
```

The model manifest selects the engine. Shared primitives and services are
reused; each engine keeps its graph, conditioning, sampler, EOS, and decoder
state explicit.

## 5. Model pack and converter

The runtime must not read a raw `.nemo` tarball. `convert_magpie.py` must:

1. validate the archive, source revision, checksum, license, and codec;
2. inspect configuration, tensor names/shapes/dtypes, tokenizer, speakers,
   special tokens, KV cache, and local-transformer settings;
3. write normalized `model.json` and row-major mmap-friendly tensor files;
4. export tokenizer/G2P/TN assets, speaker map, and special IDs;
5. emit a tensor report with offsets and checksums;
6. save `source.json` with repository, revision, date, and command line.

Expected pack:

```text
models/magpie-v2607/
├── model.json
├── tts.safetensors
├── codec.safetensors
├── tokenizer/
├── speakers.json
├── source.json
└── LICENSES/
```

`model.json` must describe engine, revision, dtypes, hidden sizes, layers,
heads/KV heads, positional encoding, vocab and special IDs, decoder mode, frame
stacking, codebook count/size/dimension, EOS policy, sample/frame rates, codec
configuration, inference defaults, and language routing.

If exact NeMo text normalization cannot be exported safely, the converter must
mark the pack as requiring normalized input. No silent fallback is allowed.

## 6. Shared inference state

`mynah_tts_synthesize()` and streaming push/flush use the same state machine.
The audio callback receives float PCM chunks; WAV or stdout is only a sink.

Request state owns text tokens, encoder state, decoder position/KV cache, local
transformer state, codec convolution state, RNG/sampler, speaker/language,
CFG, frame counters, EOS, abort, and flush status. Context reset must not leak
the previous speaker or text. Allocate capacity before the autoregressive loop;
grow only for an explicitly supported long-form limit.

Magpie stages:

1. UTF-8, language routing, optional TN/G2P, and tokenization.
2. Causal text encoder with learned positions.
3. Causal decoder with audio BOS, speaker conditioning, self-attention,
   cross-attention, logits, and EOS.
4. Optional CFG using the same context and explicit `cfg_scale`.
5. Stacked-code sampling with model-configured ordering and EOS rules.
6. Local transformer in exactly the mode declared by `model.json`.
7. NanoCodec embedding/FSQ, causal convolutions, upsampling, clipping, and
   final float/int16 conversion.
8. EOS and flush that close the causal queue and keep offline/streaming output
   equivalent within the declared tolerance.

Long-form chunking must preserve language-aware text state, global position,
EOS, and codec state. First ship one-utterance streaming; add long-form only
after chunk boundaries are proven click-free and speaker-safe.

## 7. CPU, SIMD, and memory

The scalar reference uses float32 accumulation with pack-declared bf16/fp16/fp32
weights. Required primitives are embedding lookup, dtype conversion, matvec,
matmat, norm, residual, FFN, causal/cross attention, softmax, KV operations,
sampling, convolution, upsampling, FSQ, and PCM conversion.

| Platform | Baseline | Later candidates |
|---|---|---|
| Apple arm64 | scalar + NEON + Accelerate/GCD | SDOT and fused matvec |
| Linux aarch64 | scalar + NEON + pthread/OpenBLAS | SDOT when available |
| Linux x86-64 | scalar + AVX2/FMA | AVX-512/VNNI/BF16 when verified |
| macOS x86-64 | scalar + optional AVX2 | only when measured |

`SIMD=scalar` always remains available. `--caps` reports compiled and runtime
ISA. Distributed binaries must not rely on `-march=native`; ISA functions are
isolated and tested against scalar output.

Use mmap views for safetensors, aligned reusable scratch, load-time quantized
weight caches, checked `size_t` arithmetic, and explicit ownership. Start with
bf16/fp16/fp32. Evaluate INT8 and Q4 only after audio quality and EOS/length
behavior are measured; keep the codec decoder in f32 until proven otherwise.

## 8. Optional GPU backends

CPU builds include no Metal/CUDA headers. A backend resolver reports `active`,
`not compiled`, `no device`, `unsupported model`, or `runtime failure`. A
recoverable request falls back to CPU; a correctness self-test failure does not.

Metal is macOS-only: resident `MTLBuffer` weights/KV/scratch, reusable command
buffers, and fused prefill, complete autoregressive steps, or NanoCodec work.
Do not offload one tiny matvec if dispatch and synchronization cost more than
the computation. Validate with the same CPU golden dumps.

CUDA is Linux/NVIDIA-only: begin with cuBLAS for prefill/matmat, then resident
weights/KV and a complete decoder step per launch/stream, and only later CUDA
Graphs or custom kernels. `CUDA_ARCH` is configurable and CPU startup must not
require a GPU. Single-request latency and batched throughput are separate
measurements.

The common backend seam can share buffer pools, dtype conversion, GEMM,
matvec, norm, attention, convolution, reductions, capability queries, and
self-tests. The resident graph remains engine-specific:

```text
Magpie:      AR step -> local transformer -> NanoCodec
dots.tts:    LLM step -> flow/MeanFlow NFE loop -> AudioVAE
Chatterbox:  T3 decode -> S3Gen/vocoder -> optional watermark
```

## 9. v2: dots.tts

Reference: [dots.tts](https://github.com/studio-dots-ai/dots.tts), its
[technical report](https://arxiv.org/abs/2606.07080), and the
[`dots.tts-mf` checkpoint](https://huggingface.co/rednote-hilab/dots.tts-mf).

dots.tts is a roughly 2B-parameter continuous AR system, not a discrete codec
model. It combines a semantic encoder, Qwen2.5-1.5B LLM, AR flow-matching or
MeanFlow head, CAM++ speaker x-vector, and a frozen 48 kHz AudioVAE/causal
BigVGAN-style decoder. The distilled `dots.tts-mf` NFE=4 path is the first
candidate for “fast”; the reference GPU profile is an optimization clue, not a
CPU target.

v2 must import the semantic encoder, LLM, flow head, AudioVAE, speaker encoder,
and BPE assets separately. Keep the flow sampler (`num_steps`, schedule,
guidance) explicit, use a bounded paged/ring history, and stream causal audio
patches when the decoder permits it. CPU scalar is required for correctness and
portability; the release remains GPU-first until a measured CPU baseline proves
otherwise. Prompt audio and voice cloning require an explicit consent and
duration policy.

Do not make dots a Magpie flag: continuous latent patches, ODE integration,
full-history conditioning, and AudioVAE state are a separate engine graph that
only reuses tensor primitives and backend services.

## 10. v3: Chatterbox

References: the [official repository](https://github.com/resemble-ai/chatterbox)
and [model card](https://huggingface.co/ResembleAI/chatterbox). The family has
different checkpoints and must not be autodetected by a loose name. Pin Turbo
and Multilingual V3 as separate model-pack variants.

The conceptual graph is:

```text
text tokenizer
  -> T3/Llama autoregressive speech-token generator
     + speaker x-vector/reference audio + language/CFG/exaggeration
  -> speech-token or mel generation (checkpoint-specific S3Gen)
  -> HiFi-GAN-like vocoder/waveform
  -> optional Perth watermark
```

Start with Turbo for latency and then add Multilingual V3 for language coverage
and cross-language reference voice. The converter must pin T3, S3Gen, voice
encoder, tokenizer, conditioning cache, alignment/EOS, and watermark metadata.
Keep `cfg_weight`, `exaggeration`, temperature, and watermark policy in request
state. Reference audio needs explicit consent and bounded cache lifetime.

## 11. Reuse matrix

| Component | Magpie v1 | dots v2 | Chatterbox v3 | Policy |
|---|---|---|---|---|
| mmap/model pack/WAV | yes | yes | yes | shared services |
| tokenizer API | NeMo/G2P | Qwen BPE | checkpoint tokenizer | shared interface, separate assets |
| linear/norm/attention | yes | yes | yes | shared scalar/SIMD/GPU primitives |
| KV/history | causal + cross | paged full history | causal T3 | engine-owned layout |
| sampling/EOS | codebooks | AR + flow | speech tokens/alignment | pluggable implementation |
| audio latent | discrete codes | continuous patches | speech tokens/mel | engine-specific |
| audio decoder | NanoCodec | AudioVAE | S3Gen/vocoder | shared sink, separate graph |
| speaker conditioning | baked IDs | CAM++ x-vector | reference voice | engine-specific |
| GPU residency | TTS + codec | LLM + flow + VAE | T3 + S3Gen | common seam, separate graph |

SIMD work reusable across all models includes dtype conversion, dot products,
matvec/matmat, norms, activations, reductions, RoPE, attention, KV copy,
convolution, upsampling, and quantization packing. v2/v3 additionally need
patch batching, variable-length paged history, DiT/flow math, reference-audio
resampling, x-vector pooling, AudioVAE residual convolutions, and latent
interleaving. Reuse algorithms and tests, not necessarily packing or shapes.

## 12. Makefile contract

The Makefile must be explicit and must never download weights implicitly:

```text
make                         # default CPU build and CLI
make cpu | blas | info       # CPU variants and capability report
make self-test | test        # model-free kernels; available model tests
make bench | bench-matrix     # JSON performance reports
make inspect MODEL=...       # checkpoint/model-pack report
make convert MODEL=...       # Magpie converter
make oracle MODEL=... CODEC=... BYT5=... OUTPUT=... # official CPU reference WAV
make convert-dots MODEL=...  # dots.tts converter
make convert-chatterbox MODEL=... VARIANT=turbo|multilingual-v3
make test-model FAMILY=... MODEL_DIR=...
make metal | cuda             # isolated opt-in backend builds
make gpu-selftest | leaks | ubsan | asan
make lib | shared | install PREFIX=... | dist
make clean
```

Supported variables: `CC`, `CFLAGS`, `LDFLAGS`, `SIMD`, `BLAS`, `ENGINE`,
`MODEL_DIR`, `THREADS`, `CUDA_HOME`, `CUDA_ARCH`, and `EXTRA_CFLAGS`. CPU,
Metal, and CUDA objects must live in separate build directories or be rebuilt
cleanly; they must never be silently mixed.

## 13. Oracle, tests, and benchmarks

Offline Python tooling must dump: normalized text/G2P/token IDs, encoder layer
outputs, decoder prefill and one step of logits, stacked codebooks, local
transformer output, EOS, codec output, and end-to-end WAV. Every stage checks
shape, finiteness, range, absolute/relative tolerance, and argmax/token parity
where applicable. Audio checks duration, RMS, peak, and mel correlation; it
does not require byte-identical output across floating-point backends.

Minimum tests cover UTF-8 and special IDs, scalar-vs-SIMD kernels, synthetic
codec codes, corrupted/missing packs, repeated load/unload, deterministic
sampling, speaker/language isolation, CFG, offline-vs-streaming, long-form
flush, thread counts, UBSan, ASan, and macOS `leaks`.

Benchmarks record commit, OS, CPU, ISA, threads, model revision, seed, dtype,
backend, load time, TTFA, RTF, frames/s, peak RSS, audio duration, and quality.
Do not set an RTF promise before measuring a baseline on representative Apple
and Linux hardware.

## 14. Roadmap and gates

| Milestone | Deliverable | Gate |
|---|---|---|
| M0 | pin Magpie/NanoCodec; inspect archive/config/tokenizer/licenses | repeatable audit |
| M1 | model pack and NeMo/PyTorch oracle | reloadable pack; token parity |
| M2 | mmap, scalar kernels, tokenizer, sampler, thread API | self-tests and UBSan |
| M3 | Magpie encoder/decoder, KV, CFG, stacked codes, local transformer | deterministic logits/codes parity |
| M4 | NanoCodec, WAV/PCM, CLI, errors | first verified WAV |
| M5 | shared streaming/long-form state | offline/stream equivalence; no leaks |
| M6 | NEON/AVX2, threads, measured INT8 | reproducible benchmark; quality held |
| M7 | library/server boundary | API and concurrency tests |
| M8 | Metal matmul/FFN backend | real-device shader self-test and benchmark |
| M9 | CUDA matmul/FFN backend | NVIDIA self-test and benchmark |
| M10 | dots.tts-mf, AudioVAE, flow engine | CPU correctness; bounded GPU-first profile |
| M11 | Chatterbox Turbo, then Multilingual V3 | EOS/audio parity; latency and language report |
| M12 | cross-model batching and hardening | no regressions across all packs |

Do not start GPU work before M3/M4 establish a correct graph. A faster wrong
pipeline is not progress.

## 15. Risks and open decisions

| Risk | Mitigation |
|---|---|
| `.nemo` layout changes | pin revision; version converter and tensor report |
| TN/G2P is not reproducible in C | export assets or require `--normalized` |
| local transformer differs from expectation | read config and implement only verified mode |
| CFG doubles cost | validate first; cache conditional state; benchmark on/off |
| causal codec clicks at chunk boundaries | codec golden tests and offline/stream comparison |
| AR decode is memory-bound | fused matvec, reuse, quantization, batch matmat |
| GPU copies erase gains | resident subgraphs and no per-op synchronization |
| cloning is inferred for Magpie | expose only baked speakers in v1 |
| dots is too large for CPU | CPU correctness + GPU-first release; history budget |
| Chatterbox variants diverge | separate pinned packs and converters |
| voice cloning/watermark misuse | consent, duration, metadata, and capability policy |

M0 must close exact tensor names/shapes, tokenizer format, decoder/KV/EOS rules,
local-transformer mode, NanoCodec causal state, baseline dtype, and runtime/model
license separation. Do not guess them in public headers.

## 16. v1 definition of done

v1 is complete when a pinned Magpie model pack loads on macOS arm64 and Linux
x86-64; `make` builds the CLI/library without GPU SDKs; supported-language and
speaker cases produce verified WAV; offline and streaming share state and pass
goldens; scalar/NEON/AVX2 paths self-test; load/error/UBSan/leak checks are
clean; and benchmark results are reproducible. Metal/CUDA are separate optional
targets with resident buffers, correctness tests, and CPU fallback.

The second and third model families remain follow-up engines, not hidden claims
inside the Magpie API. When this plan and implementation disagree, verified code
and the converted model pack are the source of truth.


---

## Archive

Sections 17-24 — the July/August 2026 session checkpoints, accepted and rejected
optimizations, and the Magpie server roadmap — moved verbatim to
[`.work/archive-2026-07-checkpoints.md`](.work/archive-2026-07-checkpoints.md).
Read it before re-trying Metal on Apple Silicon, vectorized Snake, cross-request
codec batching, or pool-threaded codec convolutions: all four were measured and
closed.
