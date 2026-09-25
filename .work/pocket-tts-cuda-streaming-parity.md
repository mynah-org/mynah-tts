# PocketTTS CUDA and Linux streaming parity

Status: **IN PROGRESS** · opened 2026-09-23

This note opens the PocketTTS CUDA track without changing the CPU product. The
CPU path remains the correctness oracle and the default backend. CUDA is an
explicit, separately linked artifact and a separate engine execution path; it
must never be selected accidentally by a CPU binary or by a server operator who
did not ask for it.

The first useful milestone is not a CUDA matmul. It is a Linux server that can
be built as `build/cuda/mynah-tts-server`, refuses CPU/GPU topology mistakes,
and has enough observability to distinguish a resident Pocket path from a
host-round-trip prototype. This branch now has resident Pocket backbone and
one-step flow-head batch paths with per-width CUDA-Graph capture/replay, plus
a resident single-context SEANet decoder with causal rings, transpose tails,
PCM handoff and a per-context CUDA graph. The Mimi decoder-transformer now has
an opt-in resident CUDA path with compact per-request KV windows and true
cross-request frame-tile batching; real-device stage parity and full decoder
fusion remain open. The target is a qualified C100 on
a representative L40S later; the RTX PRO 6000 is now the bring-up box, not a
substitute for that production qualification.

## Decision

Keep the existing shared request state machine and HTTP contract:

```
HTTP admission -> shared continuous driver -> Pocket engine state -> stream sink
```

Only the engine's compute implementation changes. CPU and CUDA share request
semantics, EOS/frame accounting, cancellation, chunk framing, and stream/batch
tests. They do not share a hot-loop implementation through a generic
per-operation backend call.

The CUDA server is one process per GPU and does not use CPU prefork. The model
and CUDA context are opened in that process, request state is per slot, and
weights/KV/scratch/graphs are owned by the model or request that created them.
`server/prefork.c` already rejects a GPU backend before `fork()`; keep that
refusal as a correctness gate. A requested CUDA backend fails explicitly when
CUDA is not compiled or no device exists. A compatible Pocket context attempts
its resident backbone and decoder allocation; recoverable failures before a
device state is advanced invalidate the device state and retry the same step
through the CPU graph, with the fallback made observable in backend health
before qualification. A decoder
failure after its CUDA causal rings advance is request-fatal rather than
silently switching to stale CPU rings. The server does not pretend that a
partial CUDA graph is fully resident.

The `make cuda-server` target is a build boundary plus the entry point for the
resident slice. `--device cuda` may run Pocket's transformer backbone,
one-step flow head, Mimi decoder transformer, quantizer/causal upsample and
SEANet decoder resident when their metadata and resolved precision are
compatible. The flow and raw-F32 codec paths fall back to the CPU oracle when a
quantized group is selected; this is intentional until matching CUDA quantized
kernels exist. `MYNAH_CUDA_POCKET_CODEC=0` is the explicit A/B escape hatch for
the CPU decoder-transformer oracle. With raw-F32 groups, the resident
transformer keeps activations and KV on device, stages one bounded output tile
per frame gang, and refreshes the host KV oracle only on a window rebase or
retry. The optional `MYNAH_CUDA_CODEC_HOST_MIRROR=0` skips the normal
codec-transformer D2H mirror only when the resident SEANet handoff is active.

## Validation snapshot — 2026-09-23

Passed locally on the ARM development host after the resident/backend changes:

* `make`, `make self-test`, and the full `make test` target, including 13,422
  model-free kernel checks and the server/playback/tooling suites;
* `make server-test MODEL_DIR=models/fake-magpie`, including stream-vs-batch,
  batching, admission, concurrency, cancellation and warmup cases;
* `make -n cuda-server CUDA_ARCH=sm_89` and `git diff --check`.

The resident batch step now also has bucketed CUDA-Graph capture/replay: one
graph per `(scratch, batch-width)` with pinned input/output/KV shadow staging,
layer-specific persistent KV pointer tables, dynamic position metadata for
RoPE/attention, and an ordinary-stream fallback when capture or instantiation
is unavailable. The flow head uses the same bucket policy with persistent
condition/noise/time staging and one bounded latent download per gathered
batch. These paths are source-implemented but not yet launched on this host.

This host has neither `nvcc` nor an NVIDIA device, so CUDA launch/self-test,
stage parity and L40S measurements remain explicitly open. GitHub Actions run
`35877105405` is green for the CLI and Linux CUDA server compile matrix
(`sm_70`, `sm_89`, `sm_90`), with Code Quality `35877105452` and Memory Safety
`35877105478` also green. These compile-only jobs do not substitute for a GPU
gate.

## Verified as-is: CPU PocketTTS

The current CPU path is the baseline to preserve, not a disposable prototype.
The sources of truth are `src/engine_pocket.c`, `src/tts_engine.h`,
`src/inference.c`, `src/seanet.c`, `src/flow_head.c`, the converted
`model.json`, and the oracle notes in [pocket-tts-engine.md](pocket-tts-engine.md).

| Area | Current behaviour | CUDA implication |
|---|---|---|
| Model graph | Continuous latent AR; `step_batch` is the shared engine hook. Pocket publishes a config-driven `max_batch` and one latent frame per AR step. | Keep the same batch and EOS contract; do not encode Pocket as Magpie codebooks. |
| Prefill/AR | Backbone and flow projections are composed through the Pocket engine and shared driver. The CPU path has model-owned qmat/cache state and reused scratch. | A GPU path needs a model-owned resident weight cache and per-request device KV/scratch, not `mynah_backend_matmul()` in every projection. |
| Audio decode | `pocket_decode_audio_batch()` validates a gang but currently decodes each context through `pocket_decode_frame()`; SEANet/decoder state is per request. | A first CUDA implementation can preserve this semantics, but the C100 target needs a batched/resident SEANet path rather than a serial host decoder. |
| Streaming | Offline and HTTP streaming enter the same inference driver. Pocket carries codec state and a monotonic frame position; it does not replay a large audio context. | GPU chunk boundaries must use the same contiguous range contract and must not reset causal state. |
| Precision | The repo has measured f32/f16/bf16/int8 variants and per-group resolution. The shipped ARM profile uses the qualified CPU precision/configuration; int8 experiments are not a free CUDA default. | Start with the exact CPU-approved weight representation. Add int8 only after a separate tensor-level parity gate. |
| Correctness | Oracle stages, tokenizer, stream/offline behaviour, and server stream/batch identity are already tested. | CPU remains the reference for token/frame/EOS checks; floating-point audio uses explicit tolerances, not byte identity across devices. |
| Serving | Linux HTTP streaming, bounded admission, async output, disconnect cancellation, health data and CPU prefork exist. The latest committed profile documents C120 as qualified and C128 as marginal; the user reports a newer C126 result. | Capture the exact current host/commit as the CPU baseline before any GPU comparison. GPU C100 is a new measurement, not an extrapolation. |

The current CPU performance evidence is deliberately split from the GPU plan:
the early single-request Pocket optimization moved f16 RTF from about 2.03 to
0.245 on the development machine, while the current Axion/Neoverse-V2 serving
profile is governed by scheduler, batching and cadence gates. Neither number is
a CUDA estimate. A CUDA result is invalid if it changes the text, seed, EOS
cap, model revision, precision, chunk policy, or request bank.

## Verified as-is: current CUDA path

The existing backend is a useful foundation but is not a Pocket CUDA engine.

* `gpu/cuda/backend_cuda.cu` has cuBLAS-backed host `matmul`/`sgemm`, a weight
  cache, selected device-side matmul/elementwise helpers, im2col/conv support,
  and a small matmul graph prototype. This branch adds device LayerNorm,
  GELU, residual, softmax, RoPE, causal attention, independent-request batch
  attention, resident causal conv/transpose-conv primitives, and model-free
  self-tests for the new kernels, plus resident flow-head kernels and
  asynchronous batch execution.
* The host-facing matmul path copies activations to a mapped buffer, launches a
  GEMM, synchronizes, and copies the result back. That is a correctness and
  bring-up path, not an autoregressive serving path.
* `src/engine_pocket.c:pocket_proj_row()` takes a direct CPU f32 matvec when
  its projection is unquantized. Its tile path can call the generic backend,
  but that still has host-round-trip semantics. Quantized/f16 qmat paths are
  CPU code today (`src/qmat.c:mynah_qmat_linear_resolved_qt()` and its batched
  twin).
* The resident Pocket flow head currently uses raw FP32 model pointers for its
  cached CUDA projections. This is deliberate for the first parity path; it is
  not evidence that the CPU qmat/f16 representation is already reproduced on
  the GPU.
* `pocket_decode_audio_batch()` walks ranges frame-major and invokes the
  resident causal decoder once per context when CUDA is active. Decoder state
  is device-resident, but cross-request SEANet batching and the
  quantizer/codec-transformer device boundary remain open; this is not yet a
  C100 throughput claim.
* `mynah_backend_has_dev_ops()` remains intentionally too coarse for this job:
  the CUDA backend now has more device operations, but that does not certify a
  complete Pocket graph. Pocket capability is still model-specific; backend
  counters are exposed through `mynah_tts_backend_metrics` and `/health`, while
  CUDA runtime values still need a real-device validation.
* Before this branch, `make cuda` built only the CLI while `make server` linked
  the CPU object tree. This branch adds `make cuda-server`, using the CUDA core
  objects, CUDA backend and server objects in `build/cuda/`. The CPU target and
  object tree remain unchanged.

The conclusion is strict: do not route Pocket to CUDA by changing one function
pointer in `pocket_proj`. That would make the graph slower, introduce hidden
H2D/D2H synchronization, and make parity failures difficult to attribute.

### Concrete hazards to remove before resident execution

The static audit found failure modes that a compile-only gate cannot see:

* The CUDA vtable now assigns explicit device contracts for copy/scale/clip,
  argmax, attention, RoPE, resident conv and transposed-conv. A non-null
  matmul pointer still must not be used as a complete resident-graph bit.
* The host-facing `conv1d` contract is separate from `conv1d_dev`; the Pocket
  decoder uses its own resident stateful path so causal rings and
  transposed-convolution tails cannot be confused with stateless primitives.
* Device GELU, LayerNorm, residual, softmax and attention no longer require a
  host round trip in the resident backbone. Device SGEMM honours caller
  leading dimensions. Runtime validation on `nvcc` and a GPU is still pending.
  The batch driver's host input/output slabs use backend-owned pinned staging;
  only the current K/V slot is shadowed back to the host state so a failed
  resident step can retry safely without copying the full cache.
* CUDA GEMM defaults to FP32 inputs/accumulation for the first parity path.
  `MYNAH_CUDA_FAST_MATH=1` is an explicit FP16/Tensor-Core experiment and is
  not part of the parity claim until intermediate tensors, EOS and audio pass.
* Weight caches, batch metadata, graph entries and backend counters are
  backend-owned. `/health` and SIGUSR1 stats expose H2D/D2H bytes, graph
  captures/replays/fallbacks, decoder steps/failures and resident fallbacks.
  Graphs are destroyed before scratch/staging growth and on backend close.
  Resident batch capture/replay is wired through an opaque backend contract;
  GPU launch and stage parity remain open.

## What transfers from `../qwen-tts`

The useful copy is the CUDA execution discipline, not the model graph.

| Qwen implementation/pattern | Transfer to PocketTTS | Do not transfer literally |
|---|---|---|
| `qwen_tts_cuda_talker.cu` resident model state, per-request KV and reused scratch | Ownership layout, lifecycle, pointer-keyed weight cache, one stream per model/worker, explicit warm-up | 28-layer Talker shapes, code-predictor state, RVQ/codebook ordering, Qwen quant formats |
| `qwen_tts_cuda_kernels.cu` custom matvec/attention/quant kernels | Kernel-test structure and the principle that batch matvec becomes matmat | GQA/head dimensions, causal masks, and packed layouts without a Pocket oracle |
| `qwen_tts_cuda_decoder.cu` resident causal ConvNet, im2col/GEMM, conv-transpose and gather | SEANet implementation strategy, persistent scratch, decoder batch tests | Qwen decoder padding, stride/dilation, channel ordering and weight names |
| CUDA graph capture/replay per stable shape | Pocket AR-step and decoder graph buckets keyed by effective batch/shape | A single global graph cache or a graph whose buffers belong to another request/model |
| GPU server's single-process topology and request batching | No prefork after CUDA init, bounded slots, batched streaming | Qwen's flags, worker topology and throughput numbers |
| Qwen compile-only `.github/workflows/cuda.yml` | Explicit gencode matrix, full link, then model-free “backend is compiled” check | Qwen's model-specific self-tests or claims about Pocket performance |

The user-owned projects are MIT/OSS, but copied code still needs a per-file
license/header check, provenance comment and retained notice. A semantic port
is preferred when layouts or ownership differ; copying a kernel is not a reason
to copy its assumptions.

## Kernel and graph parity map

The first CUDA kernel set is intentionally small and testable:

1. f32/bf16/f16 projection matvec and batched matmul, with bias epilogue;
2. LayerNorm, GELU/activation, residual add and scale/clip helpers used by the
   Pocket backbone and flow head;
3. causal self-attention with Pocket's actual head/cache layout, including
   append-to-KV and valid-length masking;
4. flow-head time embedding, variance-RMSNorm, adaLN modulation and the
   one-step latent output, with resident batch/graph execution;
5. causal SEANet conv, residual blocks, and transposed-conv output with the
   exact model-configured strides, padding and streaming state;
6. device-side final latent/audio chunk staging, with one bounded transfer at
   the HTTP sink boundary.

Every item starts with a scalar/CPU reference and a model-free CUDA self-test.
The CUDA kernel is promoted only after shape sweeps, finite-value checks and
maximum error against the reference. Kernel-level parity is separate from
end-to-end parity: matching a matvec does not prove a causal decoder's state
transition.

## Planned implementation slices

### P0 — Baseline and capability contract

Record the exact CPU binary, commit, pack revision, precision, seed, language,
voice and server configuration. Save stage dumps for tokenizer, prefill,
backbone, flow, latent/EOS and audio. Record the user's C126 result separately
from the committed C120/C128 evidence until its log and configuration are in
the repository.

Add an explicit internal capability result with these states:

```
active-resident | compiled-no-device | unsupported-model |
runtime-failure | cpu
```

The server health output and startup log must name the state, model, backend,
GPU identity, resident graph mode, and whether any host fallback is active.

### P1 — Build and CI boundary (started in this branch)

* Keep `make cuda` for the CLI and add `make cuda-server` for the server.
* Build both targets in the existing compile-only CUDA job for explicit
  `sm_70`, `sm_89` (L40S/Ada) and `sm_90`.
* Run `--dispatch-map` and `--gpu-self-test cuda` on the no-GPU runner. The
  test may report no device; it must not report “CUDA backend is not compiled”.
* Add no model download and no GPU runtime assertion to hosted CI.
* Keep CPU `make`, CPU self-tests, sanitizer jobs and server tests as separate
  gates. A green CUDA compile job never substitutes for CPU parity.

### P2 — Resident backend ownership → **started**

Refactor the CUDA state so all persistent allocations are owned by one backend
instance or one explicit Pocket model state. Remove process-global/static
device buffers from the Pocket path; graph and “ones” buffers must be keyed by
backend/device/shape and freed with that owner. Define stream ordering and
pointer lifetime for upload, device allocation, async operation and download.

Separate host-buffer and device-buffer contracts (`conv1d_host` versus
`conv1d_dev`, and the equivalent for the decoder). Honour leading dimensions in
device SGEMM or make the packed-layout precondition explicit and checked. The
backend must expose enough capability for Pocket to reject a partial graph. Do
not infer “resident” from `matmul_dev != NULL`. Backend-owned graph and batch
metadata lifecycle is implemented; explicit capability/health reporting is
still open.

### P3 — Pocket projection and AR step → **backbone slice implemented; parity pending**

The branch adds an internal opaque Pocket CUDA implementation behind the engine
seam. Keep
`src/mynah_tts.h` backend-neutral. The CUDA path should:

* upload/convert model weights once per model and reuse them;
* keep the Pocket backbone KV and transformer intermediates on the device
  across a step;
* batch independent request rows so a shared weight read becomes matmat, with
  per-request KV pointers and positions;
* keep the one-step latent flow head on device where practical — the current
  slice downloads one latent row per gathered batch while retaining all flow
  intermediates resident; EOS reductions and the SEANet decoder remain at the
  engine seam, and the backbone still shadows only the appended K/V slot;
* capture/replay stable full-backbone batch shapes through per-scratch CUDA
  Graph buckets (enabled by default, `MYNAH_CUDA_GRAPHS=0` escape hatch), with
  an uncaptured FP32 fallback;
* leave the CPU `pocket_proj_*`, qmat cache and scalar/SIMD dispatch untouched.

The first implementation uses f32 accumulation with the CPU-approved weight
representation and explicitly disables TF32/fast-math shortcuts until the
stage tolerances are measured. CUDA quantization, FP16 compute and TF32 are
later A/B items, not shortcuts around parity.

### P4 — Resident SEANet streaming decoder → **single-context slice implemented**

Port the actual Pocket SEANet graph, not the Qwen decoder by name. Carry causal
conv rings, decoder-transformer KV and frame position in the per-request device
state. Decode batched contiguous frame ranges and emit a bounded PCM chunk via
the existing asynchronous stream writer. A disconnect must release the request
state without corrupting another slot's graph or scratch.

The first CUDA slice now carries the SEANet causal conv rings and
conv-transpose partial tails on device, uses persistent cuBLAS/device scratch,
captures one stable graph per context, and hands back one bounded PCM frame.
The model-free CUDA self-test compares two consecutive decoder calls against
the CPU SEANet state. The new codec-transformer slice keeps the 250-position
Mimi KV window resident, handles absolute RoPE after window rebases, and
executes frame-major cross-request tiles with row-wise LayerScale. The
quantizer/upsample boundary remains host-side; real-device codec stage parity,
same-backend stream/offline parity and fully batched SEANet kernels are still
required.

### P5 — Linux server integration → **build boundary, graph batch path and metrics implemented**

Use the existing admission/scheduler/sink. For CUDA:

* one process owns one selected GPU;
* `--prefork` is rejected before startup, not after a request arrives;
* `--max-batch` is constrained by the Pocket CUDA capability and VRAM budget;
* cancellation, timeout, bounded queue, health, `/metrics` and
  `/v1/audio/speech`/`/v1/tts` streaming retain the CPU contract;
* the scheduler exposes graph-hit, batch-width, H2D/D2H, sync and fallback
  counters without allocating in the AR loop. `/health` and `/metrics` carry
  backend metrics for CPU (zero CUDA counters) and CUDA (live monotonic
  counters).

Do not add a second HTTP implementation or a CUDA-specific response format.

### P6 — Parity gates

For identical model revision, text, language, voice, seed, sampler, EOS cap,
precision and thread count, compare:

| Stage | Required gate |
|---|---|
| tokenizer | exact token IDs and byte-fallback behaviour |
| prefill/backbone/flow | max absolute/relative error against per-stage tolerance; finite values |
| AR generation | exact EOS position and generated frame count at deterministic settings |
| latent/audio decoder | duration, finite values, peak/RMS, max error and log-mel correlation; no byte-identity claim across CPU/GPU arithmetic |
| same-backend streaming | stream vs offline sample identity or the engine's documented tolerance; contiguous state and cancellation tests |
| batching | each request alone vs in a batch, with no cross-request state contamination |

A GPU self-test failure is fatal. “CUDA unavailable” is a startup status, not
a successful parity result.

### P7 — L40S qualification, when hardware is available

The L40S target is an Ada Lovelace, 48 GB, 864 GB/s card; compile for `sm_89`
and verify the actual device at runtime. The target is **C100**: 100 concurrent
streaming requests under the repository's cadence gates, not merely 100
requests accepted or an aggregate RTF below one.

The campaign is one process and one GPU, warmed with at least two warm-ups and
five measured runs for microbenchmarks. Sweep one variable at a time over
`max_batch` 1/2/4/8/16 and then run the closed-loop C ladder. Record TTFA,
TTFB, stream RTF, required prebuffer, max gap, stalls at each threshold,
throughput, p50/p95, VRAM, host RSS, graph hit rate, H2D/D2H bytes and sync
counts. A C100 claim requires a 30-minute soak, audio/parity gates, no hidden
CPU fallback, and a stable drift window. It cannot be inferred from an RTX,
A100, or Qwen number.

## 2026-09-24 implementation audit and next code slices

The review against the current tree, `../qwen-tts`, and the official vLLM-Omni
CUDA-graph/async-chunk designs found that the resident slice is a correct
bring-up foundation, not yet a qualified C100 path. The important distinction is
between **active request capacity** and **GPU microbatch width**: Pocket currently
publishes `max_batch = 16`, so a CUDA server cannot express C100 concurrent slots
without a scheduler/capacity seam even if the GPU executes only B8/B16 at a time.

The SEANet decoder remains context-serial arithmetically. The
`decode_audio_batch()` path now also runs the Mimi decoder-transformer as a
true cross-request frame-tile batch, with per-request KV pointers and
window-relative attention positions; its activation output is staged once per
tile. Quantizer/upsample and SEANet state are still per-context, so the next
slice owns B1/B2/B4/B8/B16 state arrays and padded `(batch, frames)` decoder
graph buckets. The scheduler separates `--max-batch` (engine microbatch) from
`--max-inflight` (resident slots, up to 128), so a C100 experiment can keep 100
live requests while executing bounded microbatches.

The following flags are explicit experiments, not hidden policy:

```text
MYNAH_CUDA_GRAPHS=0|1              graph capture/replay escape hatch
MYNAH_CUDA_FAST_MATH=0|1           TF32/fast compute experiment; no parity claim
MYNAH_CUDA_DECODER_BATCH=0|1       async decoder gang submission, default on for CUDA
MYNAH_CUDA_POCKET_CODEC=0|1        resident Pocket Mimi decoder transformer, default on for CUDA
MYNAH_CUDA_CODEC=0                  legacy global codec kill switch (also disables Pocket)
MYNAH_CUDA_CODEC_HOST_MIRROR=0|1  keep codec-transformer D2H mirror; default on
MYNAH_QUANT_GROUPS=none             raw-F32 resident parity/bring-up profile
```

`MYNAH_CUDA_DECODER_GEMM` and `MYNAH_CUDA_METRICS` are intentionally not
accepted yet: they would be misleading no-op knobs until packed
ConvTranspose/GEMM and per-stage/bucket timing instrumentation exist. Their
work is tracked by CUDA-04 and CUDA-06/observability respectively.

Metrics must distinguish monotonic backend counters from request timing sums and
future histograms. The current `/metrics` surface includes queue wait, TTFA,
E2E, RTF and audio seconds; TTFB, stage/batch width, decoder bucket hit/miss,
and per-stage H2D/D2H timing remain follow-up work. The backend already exposes
H2D/D2H bytes and calls, graph capture/replay/fallback reasons, sync count,
device VRAM totals, resident fallback counts, and resident Mimi transformer
tile counts/width. Labels stay low-cardinality (`backend`, `arch`,
`precision`, `stage`, `bucket`) so a request ID never enters Prometheus.

The hosted GitHub job remains compile-only for `sm_70`, `sm_89` and `sm_90`.
Its self-test command must not be used as a runtime gate without a device. A
later optional/self-hosted GPU job will run model-free kernels, CPU↔CUDA stage
parity, stream/offline parity, graph on/off, precision variants and
`compute-sanitizer`; until then status is compile-verified, not runtime-verified.

### Work item order

Current slice status: CUDA-01, CUDA-02 and CUDA-03 are implemented in the
local branch and covered by CPU/server gates. The RTX PRO 6000 model-backed
bring-up below proves the resident 6L/24L path and server contract on a real
Blackwell device; stage parity, multi-request arithmetic batching and L4/L40S
qualification remain open.

```text
CUDA-01  async decoder gang submission + backend batch counters (implemented; GPU parity open)
CUDA-02  Prometheus /metrics + GPU/graph/transfer/batch observability (implemented; TTFB/per-stage timing histograms open)
CUDA-03  C100 capacity seam: inflight slots separate from microbatch width (implemented; runtime soak open)
CUDA-04  true stateful decoder B1/B2/B4/B8/B16 kernels and graph buckets
CUDA-05  codec-transformer residency + H2D/D2H overlap (resident frame-tile path implemented; GPU parity/overlap open)
CUDA-06  BF16/FP16/cuBLASLt/fused kernel ladder with stage parity gates
CUDA-07  optional GPU CI, sanitizer and L40S qualification campaign
CUDA-08  Blackwell/Ada architecture stamp, real-device self-test and no-hidden-fallback audit (sm120/sm89 build + model bring-up done; qualification open)
```

## 2026-09-24 24L compatibility audit and work items

The official English pair at revision
`492522650173a0653b7575cdc25ae09810e5d741` was inspected from safetensors
headers without loading the full weights. The result is recorded in
[`.work/pocket-tts-24l-compatibility.md`](pocket-tts-24l-compatibility.md):
24L is the same graph with 24 backbone blocks, not a second engine. The only
model-file differences are 18 additional blocks (144 tensors) and a source
dtype policy change: 6L is all BF16, while 24L stores flow/backbone in F32 and
Mimi in BF16. The corresponding 24L voice cache has 24 layers and an I64
`pad` metadata tensor per layer; cache shape/head geometry is unchanged.

Completed in this slice:

```text
24L-01  header/config audit, layer-0..5 shape comparison and full non-layer diff  DONE
24L-02  converter guard: dynamic block count + mixed source dtype conversion      DONE
24L-03  actual official 24L download, --dtype source conversion and pack verify   DONE
24L-04  C loader/CPU inference/audio gate on the converted official 24L pack     DONE
24L-05  CPU 6L regression: conversion, self-check, cloning and server audio     DONE
24L-08  serial CPU resource report; GPU resource fields remain pending           PARTIAL
```

Still required before 24L support is called complete:

```text
24L-06  CUDA metadata/KV/graph/pointer audit and real 6L stage/EOS parity     PARTIAL
24L-07  real 24L CUDA load/inference/stage/EOS parity on the same CUDA path   PARTIAL
24L-08  GPU resource report: model bytes, VRAM, TTFA, RTF when GPU exists     DONE (bring-up)
24L-09  README support matrix and reproduce commands (TODO; CPU/GPU support was absent)
```

The C Pocket path already uses `cfg->layers` for backbone allocation, CPU
state/KV, resident CUDA KV, batch pointer tables, graph metadata and all layer
loops. The audit found no six-layer allocation in those paths. The remaining
risk is not a second engine design; it is proving that the actual 24L pack's
larger resident allocations and source precision survive the existing CUDA
weight/cache contracts.

## CPU validation gate

The actual official 24L checkpoint was converted with `--dtype source` and
`--voices alba` into a temporary pack (358/358 tensors, mixed F32/BF16), then
passed the pack verifier, the native Pocket self-check, CPU loading and a
valid WAV inference. The existing 6L pack also passed the same native self-
check and a server WAV smoke. CPU voice cloning passed on both packs from the
same 10.32 s, 24 kHz reference: 129 voice frames and 130 KV positions, taking
6.92 s on 6L and 16.17 s on 24L. No model or generated audio is committed.

The correctness oracle used the official Python implementation and the C
runtime with `MYNAH_QUANT_GROUPS=none`, so the C path retained the official
24L F32 flow/backbone precision. For the five-word prompt
`Hello from Pocket TTS today.` with seed `1234`, temperature `0` and 40-step
cap, selected stage comparisons were:

```text
stage / call                         max abs error       correlation
out_norm hidden, first 3 calls      0.0003411           >= 0.999999977
EOS logits, first 3 calls           0.0001841           --
flow output, first 3 calls          0.0005670           >= 0.999999978
waveform (37 frames, 2.96 s)        0.07311             0.99813
```

The shipped default quantized CPU path was also exercised and produced finite
valid audio, but it is recorded separately from the FP32 oracle because the
quantized stage tensors are not expected to be numerically identical.

Serial one-worker server measurements on the local macOS host (`max_batch=1`,
`max_inflight=1`, no warmup, one request; RSS is process RSS, not model size):

| pack/path | model file | server RSS before/after/peak | TTFA | audio / RTF |
|---|---:|---:|---:|---:|
| 6L, shipped quantized CPU | 219.03 MB | 783.9 / 815.2 / 818.2 MB | 0.233 s | 1.84 s / 0.126 |
| 24L, shipped quantized CPU | 1,304.21 MB | 1,947.7 / 2,006.6 / 2,019.0 MB | 0.890 s | 2.72 s / 0.327 |

The 6L pack contains its local voice set while the 24L verification pack has
only `alba`, so RSS is a deployment measurement, not a normalized model-size
comparison. The 24L `alba` voice file is 12.39 MB in the pack (source F32
voice state was 24.78 MB). The FP32 correctness CLI run took 2.307 s of
synthesis for 2.96 s of audio (RTF 0.779). These are CPU/macOS numbers, not
Axion or L40S projections.

Current validation matrix:

```text
                         CPU                                      CUDA
Pocket small / 6L        PASS: converted pack, self-check, clone, server  PASS: CLI + server WAV + chunked PCM smoke
Pocket large / 24L       PASS: official pack, self-check, clone, oracle   PASS: CLI + server WAV + chunked PCM smoke
```

The CUDA path was audited for metadata-driven layer allocation/loops in the
resident backbone, KV cache, graph pointer tables and decoder descriptors;
the remote model-backed smoke now confirms both official packs load and finish
inference on the same path. This is runtime bring-up evidence, not the final
stage/EOS/batch parity gate: a green WAV alone does not prove every boundary
matches CPU.

## 2026-09-24 RTX PRO 6000 Blackwell bring-up

The requested remote box is an NVIDIA RTX PRO 6000 Blackwell Server Edition
with 97,887 MiB VRAM, driver 580.178.04 and CUDA 13.0 runtime. The installed
CUDA 13.2 toolkit compiles the CUDA CLI and server. A clean explicit
`CUDA_ARCH=sm_120` build passed the full model-free CUDA backend self-test on
the real device. A separate clean `CUDA_ARCH=sm_89` build also compiled and
linked, covering the L4/L40S production profile; CI continues to cover sm_70,
sm_89 and sm_90. `CUDA_ARCH=native` is reserved for local bring-up, never for a
portable production artifact.

The bring-up also found and fixed a self-test defect: the single-row RoPE test
checked K values as though they were additional Q pairs. The kernel was
correct; the corrected test now checks Q and K with their shared pair
frequencies. The Makefile now records the requested CUDA architecture in a
stamp so changing from sm_120 to sm_89 (or back) cannot silently relink a cubin
compiled for the previous GPU.

The resident coverage audit is deliberately not a 99%-GPU claim yet. The
backbone attention/FFN, flow head and causal SEANet decoder execute on CUDA
when their opt-in allocations succeed. CPU work still includes tokenization,
sampling/RNG/EOS bookkeeping, host projection boundaries, the codec
transformer, bounded K/V shadow copies for retry safety, and D2H/H2D handoffs
per autoregressive step/frame. The first real model run must report these
transfer counters and CPU utilization; CUDA-05 owns eliminating the remaining
hot-path host boundaries before an AWS L4/L40S capacity claim.

The remote box was prepared with an isolated Hugging Face CLI environment and
the official English 6L/24L pair at revision
`492522650173a0653b7575cdc25ae09810e5d741`. The source checkpoints were
converted remotely with `--dtype source` into `/root/pocket-6l-hf` and
`/root/pocket-24l-hf`; no weights or generated audio are committed here.

## 2026-09-24 real-model RTX PRO 6000 validation

Build/runtime facts:

* RTX PRO 6000 Blackwell Server Edition, 97,887 MiB VRAM, driver 580.178.04;
  CUDA 13.2 toolkit and runtime 13.0.
* Clean `CUDA_ARCH=sm_120` build plus model-free CUDA self-test passed on the
  device. A clean `CUDA_ARCH=sm_89` build also compiled and linked for the
  L4/L40S production profile. `compute-sanitizer --tool memcheck` passed on
  the model-free CUDA self-test with zero reported errors.
* Official remote conversion passed 214/214 tensors for 6L and 358/358 for
  24L. The resulting packs report 6 and 24 backbone layers respectively,
  `d_model=1024`, 16 heads of 64, FFN 4096, and the same flow/Mimi metadata.

Model-backed smoke results, all with seed 1234, temperature 0, one worker and
`MYNAH_CUDA_RESIDENT=1 MYNAH_CUDA_FLOW=1 MYNAH_CUDA_GRAPHS=1`:

| case | result | output | measured synthesis |
|---|---|---:|---:|
| 6L CPU, `max_steps=8` | valid WAV, 24 kHz, finite | 0.48 s | 0.941 s, RTF 1.961 |
| 24L CPU, `max_steps=8` | valid WAV, 24 kHz, finite | 0.64 s | 1.409 s, RTF 2.202 |
| 6L CUDA CLI, `max_steps=8` | valid WAV, 24 kHz, finite | 0.48 s | 0.281 s, RTF 0.586 |
| 24L CUDA CLI, `max_steps=8` | valid WAV, 24 kHz, finite | 0.64 s | 0.433 s, RTF 0.676 |

The corresponding CPU/CUDA PCM smoke comparisons had the same sample counts;
6L correlation was 0.942539 (max int16 difference 7924) and 24L correlation
was 0.995967 (max int16 difference 854). These are useful execution checks,
not the required stage/EOS parity claim; intermediate dumps and deterministic
EOS comparison remain to be added.

The CUDA HTTP server also completed both models with `--max-batch 4`,
`--max-inflight 4`, one worker and no fallback counters. The serial server
microbench used two warm-ups and five measured requests:

| server model | measured wall time | sampled GPU peak | sampled VRAM | backend result |
|---|---:|---:|---:|---|
| 6L | 0.20 s (all 5) | 62% at 200 ms sampling | ~1,033 MiB | 7/7 jobs, graphs 8/8, fallbacks 0 |
| 24L | 0.56 s (all 5) | 93% at 200 ms sampling | ~1,925 MiB | 7/7 jobs, graphs 10/10, fallbacks 0 |

The server timing sums over the seven-request campaigns were 1.494708 s over
17.2 s of 6L audio (RTF ~0.087) and 3.904544 s over 22.4 s of 24L audio
(RTF ~0.174). These are resident Blackwell bring-up numbers, not L40S
qualification numbers; the 200 ms utilization sample is not a sustained
utilization measurement. Process RSS was about 1.25 GB for 6L and 2.43 GB for
24L after requests, with peak RSS about 1.26 GB and 2.48 GB respectively.

Both 6L and 24L also returned `200 audio/pcm` with `Transfer-Encoding:
chunked`, 24 kHz headers and non-empty PCM for `stream:true`. The streaming
requests reported `streams_total=1`, zero failed jobs, zero graph fallbacks,
zero decoder failures and zero resident fallbacks. The test used one request;
cross-request streaming batch parity is still open.

The remaining residency cost is visible in the counters: for the 24L seven
request campaign, 259,031,040 bytes H2D, 58,383,360 bytes D2H, 840
synchronizations, 7 graph captures, 553 replays, 280 decoder steps and 26,880
device matmul calls were recorded. The GPU path is real and useful, but this
transfer/synchronization profile plus the CPU codec-transformer boundary is
why no 99%-GPU or C100 claim is made yet.

## 2026-09-24 final Blackwell screen before shutdown

The box was shut down after all reports were copied to the local, untracked
directory `.work/blackwell-20260924/`. The latest branch commit was `1b1efce`
(`perf: coalesce CUDA backbone KV readback`); the local worktree was clean
before collecting the reports. No CUDA server, soak process or residual GPU
allocation remained at shutdown.

The final 24L C64 wave used the current `max_batch=64` path, prefix-only KV
upload and coalesced backbone KV readback. It completed 64/64 requests with
zero failures and zero resident fallbacks, but was **NOT STREAMABLE**:
`STREAM_RTF p95=2.398`, `TTFA p95=10.09 s`, `stall@500 ms=91%`, aggregate
throughput 9.77 audio-s/wall-s and aggregate RTF 0.102. Health recorded
`backbone_batch_calls=82`, `items=1485`, `max_width=27`, `decoder_batch_calls=107`,
`decoder_batch_items=1447`, 1.762 GB H2D, 303 MB D2H, 3,031 D2H calls, 323
synchronizations, zero graph fallbacks and peak RSS about 3.96 GB. Sampled
VRAM stayed below 4.4 GB; sampled GPU utilisation was bursty rather than near
100%.

The earlier ten-minute C64 soaks are useful threshold evidence, not promotion
evidence: 6L completed 2,829/2,829 with no rejects but `STREAM_RTF p95=5.591`
and 100% stall rate; 24L completed 1,235/1,235 with no rejects but
`STREAM_RTF p95=13.789` and 100% stall rate. The short wave ladder was much
better at low concurrency: 6L was GOOD through C8 and MARGINAL at C16; 24L
was GOOD through C4, MARGINAL at C8 and NOT STREAMABLE at C16. These results
show that the current server is functionally stable and genuinely uses CUDA,
but the CPU codec transformer and launch/synchronization overhead still limit
continuous high-concurrency streaming. They do not support a C100 or 99%-GPU
claim.

## 2026-09-24 resident Mimi decoder-transformer implementation

The next source slice is now implemented locally, but has deliberately not been
promoted to a CUDA result because the Blackwell box is off and this Mac has no
`nvcc` or NVIDIA device.

* `src/transformer_ar.c` exposes the bounded window base/capacity and a
  non-advancing `prepare_window()` operation. This keeps the existing CPU
  sliding-window state authoritative and makes rebasing explicit to the device
  mirror; no CPU inference loop was split into a second model.
* Pocket allocates one device KV window and reusable activations per context,
  plus one pinned frame-major gang workspace. Allocation is driven by
  `model.json` codec metadata and `codec_tf_layers`, not by backbone depth.
* The CUDA tile executes LayerNorm → fused QKV matmul → absolute RoPE →
  window-relative causal attention → output projection → LayerScale residual →
  FFN/GELU → LayerScale residual for every Mimi decoder-transformer layer.
  The batch path uses row-wise LayerScale; a flat elementwise scale would be
  wrong for every request after row zero.
* A scheduler singleton uses the same reusable pinned host staging as the true
  gang path, one codec row at a time, so pageable `codec_seq`/`codec_out` do not
  force an implicit CUDA copy synchronisation when no neighbour is available.
  The standalone CPU/offline compatibility path is unchanged.
* Host KV is not copied back on every frame. It is refreshed only before a
  window rebase or when a failed resident tile needs the CPU retry oracle.
  `MYNAH_CUDA_POCKET_CODEC=0` disables this optional slice for a clean A/B
  against the prior CPU decoder-transformer path. The legacy
  `MYNAH_CUDA_CODEC=0` switch remains honored too.
* Failure recovery distinguishes a partial KV H2D upload from a device mirror
  that already ran kernels: the former keeps the host cache authoritative and
  retries CPU directly; the latter performs one bounded D2H mirror before the
  CPU retry. This avoids importing an incomplete device window into fallback.
* Health/Prometheus diagnostics now include resident codec-transformer tile
  calls/items/max width, in addition to the existing H2D/D2H/sync/matmul and
  decoder counters. A CUDA model-free self-test also covers the new row-wise
  LayerScale kernel.

The next-box gate is intentionally narrow:

```text
1. make cuda-server CUDA_ARCH=sm_89                 compile-only first
2. --gpu-self-test cuda                             model-free kernels + LayerScale
3. 6L and 24L CLI with MYNAH_CUDA_POCKET_CODEC=0/1  finite audio, same sample count
4. dump/compare decoder-transformer outputs          CPU vs CUDA max error/correlation
5. long-form >250 codec positions                   window rebase correctness
6. C2/C4/C8/C16 wave ladder                         codec tile counters + stalls
```

Until those six checks pass on real hardware, the 2x2 matrix above remains the
previous Blackwell smoke result and this new codec-transformer path is marked
**source-implemented / runtime validation pending**.

## 2026-09-24 deep CUDA audit and lifetime corrections

This audit was performed after the previous Blackwell smoke, so the old WAV
result must not be read as proof for the current resident decoder-transformer
source. The following source-level hazards were found and corrected locally:

* Codec-transformer batch metadata now has independent pinned host slices for
  every `(upsample position, layer, request)` K/V pointer table. The previous
  implementation reused a layer-sized table across positions while
  `cudaMemcpyAsync` was still allowed to consume it.
* The codec input/output pointer tables and position/stride arrays already use
  per-position slices; the singleton path now uses the same capacity-strided
  pinned staging instead of overwriting one host row while an async copy may be
  pending.
* If resident causal Mimi upsample fails after it has touched its device tail,
  the bounded tail is imported into the CPU SEANet state before disabling the
  optional path. If that import cannot be completed, the request fails rather
  than silently losing causal history.
* The causal depthwise ConvTranspose step no longer reads and writes the same
  tail slots from different CUDA threads. One thread owns each channel and
  walks the short kernel, preserving the old tail for both `tail == stride`
  (the production shape) and the `tail > stride` overlap case. Destruction of
  request/scratch device state now drains the backend stream first, including
  cancellation and error teardown.
* The single-request seams (`step_input`, hidden, denorm, codec rows and PCM)
  prefer CUDA-pinned host buffers. Allocation failure remains a safe pageable
  fallback and is visible through transfer counters.
* Raw CUDA stages are now explicitly precision-gated. The resident FP32 CUDA
  kernels do not replace CPU int8/f16/bf16 qmat results. CUDA model startup
  prints the resolved Pocket capability, and `MYNAH_QUANT_GROUPS=none` is the
  reproducible raw-F32 bring-up profile.
* The public backend metrics layout was extended for upsample counters and the
  library version was bumped to 1.5.0 so consumers do not silently keep the
  old struct contract.
* Hosted CI now has a separate official CUDA 12.8.1 `sm_120` compile/link job;
  the legacy CUDA 12.6 matrix remains for `sm_70`, `sm_89` and `sm_90`.

The audit also confirms what is still genuinely missing outside these fixes:

```text
control seam       EOS projection, RNG/sampling, and LSD bookkeeping still run on CPU;
                   backbone hidden and flow latent cross the host boundary each AR step
quantized CUDA    no qmat int8/int4/bf16/f16 resident implementation yet; raw FP32 only
SEANet batching   decoder gang submission is asynchronous but arithmetic is still
                   one request/graph at a time; this is not true cross-request batching
codec graphs      Mimi decoder-transformer frame tiles do not yet have a captured
                   graph bucket; dynamic window/pointer metadata still uses ordinary launches
observability     counters expose bytes/calls/fallbacks and tile widths, but not per-stage
                   GPU time or a separate true-arithmetic-vs-gang decoder counter
validation        current source needs a fresh nvcc build, GPU self-test, stage parity,
                   compute-sanitizer and failure-injection run on the next NVIDIA box
```

These are capability gaps, not hidden fallbacks: the CPU oracle is deliberately
selected when a raw CUDA stage cannot preserve the resolved CPU precision. A
99%-GPU or C100 claim remains invalid until the control seam, quantized profile
and true decoder arithmetic are either implemented or measured as acceptable
on the target L4/L40S workload.

## Acceptance gates before calling this done

1. `make` and the existing CPU self-test/parity/server gates pass unchanged.
2. `make cuda cuda-server CUDA_ARCH=sm_89` compiles and links in the pinned
   CUDA container without a GPU.
3. The CUDA binary reports “compiled, no device” rather than “not compiled” in
   model-free CI.
4. Pocket CUDA stage, EOS, stream/offline and batch parity pass on a real CUDA
   device with the documented tolerances.
5. The CUDA server runs without prefork, streams PCM, survives cancellation and
   queue pressure, and reports its backend/residency counters.
6. Only after 1–5: a L40S C100 soak may promote a performance profile. Until
   then the status is compile-verified or runtime-verified, never qualified.

## Explicit non-goals and rejected shortcuts

* No Qwen Talker/code-predictor/RVQ graph is inserted into PocketTTS.
* No per-matvec H2D/D2H loop is accepted as a serving implementation.
* No global CUDA graph, `d_ones`, scratch or weight cache is shared across
  models, devices or request lifetimes.
* No CUDA context is inherited across `fork()`.
* No hidden CPU fallback is enabled by asking for `--device cuda`.
* No Pocket dimension, codebook count, stride, frame rate or head shape is
  hardcoded in generic code; read it from `model.json`.
* No GPU RTF or C100 claim is copied from Qwen, vLLM-Omni, an extrapolation, or
  a no-GPU CI build.

## Research references

The relevant external designs reinforce the same boundaries:

* [vLLM-Omni Qwen3-TTS performance optimization](https://github.com/vllm-project/vllm-omni/blob/main/docs/design/qwen3_omni_tts_performance_optimization.md) uses CUDA graphs on critical decode paths, batching and asynchronous chunks; its measurements are not Mynah measurements.
* [vLLM-Omni shared CUDA graph runner RFC](https://github.com/vllm-project/vllm-omni/issues/4571) describes bucketed `(batch, frames)` capture/replay with static buffers and an explicit capture fallback.
* [vLLM-Omni async chunk design](https://github.com/vllm-project/vllm-omni/blob/main/docs/design/feature/async_chunk.md) supports overlapping chunk I/O and compute without changing the streaming contract.
* [NVIDIA CUDA Graphs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html) documents stream capture and replay to reduce dependent launch overhead.
* [NVIDIA nvcc](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/) documents explicit `sm_89` compilation and why `-arch=native` is unsuitable for headless CI.
* [NVIDIA L40S](https://www.nvidia.com/en-in/data-center/l40s/) is the later qualification target, not a local benchmark result.
