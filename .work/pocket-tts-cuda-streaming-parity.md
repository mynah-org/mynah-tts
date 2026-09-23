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
one-step flow-head batch paths with per-width CUDA-Graph capture/replay; the
SEANet decoder is still CPU-side. The target is a qualified C100 on a
representative L40S later; no rented GPU is part of this work item.

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
its resident backbone allocation; recoverable allocation or runtime failures
invalidate the device state and retry the same step through the CPU graph, with
the fallback made observable as a backend-health item before qualification. The
server does not pretend that a partial CUDA graph is fully resident.

The `make cuda-server` target is a build boundary plus the entry point for the
resident slice. `--device cuda` may run Pocket's transformer backbone and
one-step flow head resident when their metadata is compatible. The flow path is
disabled with `MYNAH_CUDA_FLOW=0`, uses raw FP32 weights in this bring-up, and
falls back to the CPU flow head on a recoverable failure. SEANet decoding
remains CPU-side until its causal-state and audio parity gates pass.

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
* `pocket_decode_audio_batch()` does not use device scratch and invokes the
  causal codec once per context. Existing CUDA codec primitives are therefore
  not evidence that Pocket's SEANet decoder is resident.
* `mynah_backend_has_dev_ops()` remains intentionally too coarse for this job:
  the CUDA backend now has more device operations, but that does not certify a
  complete Pocket graph. Pocket capability is still model-specific; the health
  state and per-stage fallback counters are pending.
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
* The host-facing `conv1d` contract is now separate from `conv1d_dev`; Pocket's
  current SEANet path is deliberately not enabled by that primitive because
  decoder state and stream parity are not implemented yet.
* Device GELU, LayerNorm, residual, softmax and attention no longer require a
  host round trip in the resident backbone. Device SGEMM honours caller
  leading dimensions. Runtime validation on `nvcc` and a GPU is still pending.
  The batch driver's host input/output slabs use backend-owned pinned staging;
  only the current K/V slot is shadowed back to the host state so a failed
  resident step can retry safely without copying the full cache.
* CUDA GEMM defaults to FP32 inputs/accumulation for the first parity path.
  `MYNAH_CUDA_FAST_MATH=1` is an explicit FP16/Tensor-Core experiment and is
  not part of the parity claim until intermediate tensors, EOS and audio pass.
* Weight caches, batch metadata and graph entries are backend-owned. Graphs are
  destroyed before scratch/staging growth and on backend close. Resident batch
  capture/replay is now wired through an opaque backend contract; the remaining
  observability work is to expose graph hits, transfers and resident fallback in
  server health rather than infer them from function pointers.

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

### P4 — Resident SEANet streaming decoder → **not started**

Port the actual Pocket SEANet graph, not the Qwen decoder by name. Carry causal
conv rings, decoder-transformer KV and frame position in the per-request device
state. Decode batched contiguous frame ranges and emit a bounded PCM chunk via
the existing asynchronous stream writer. A disconnect must release the request
state without corrupting another slot's graph or scratch.

The single-request decoder first proves stream/offline identity on the same
backend. Cross-request batching comes after that gate.

### P5 — Linux server integration → **build boundary and graph batch path started; runtime counters pending**

Use the existing admission/scheduler/sink. For CUDA:

* one process owns one selected GPU;
* `--prefork` is rejected before startup, not after a request arrives;
* `--max-batch` is constrained by the Pocket CUDA capability and VRAM budget;
* cancellation, timeout, bounded queue, health and `/v1/audio/speech`/
  `/v1/tts` streaming retain the CPU contract;
* the scheduler exposes graph-hit, batch-width, H2D/D2H, sync and fallback
  counters without allocating in the AR loop (counter publication is still
  pending; the graph path itself is now present).

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
