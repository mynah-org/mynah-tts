IN PROGRESS

# Pocket CUDA runtime gaps and L4 capacity

## Problem

The CUDA Pocket path is resident and model-backed, but the current L4 service
still pays too much per-frame launch/synchronization overhead. The 24L model
also leaves little headroom for increasing request width with the current FP32
KV/state layout. The goal is a measured streaming server path, not a
plausible-looking GPU result.

This note records only implementation techniques and measurements. It does
not copy names or attribution from private reference code.

## Baseline before this item

Machine: NVIDIA L4, 23,034 MiB VRAM, CUDA 12.8, `sm_89`, one server process,
one engine thread, official Pocket 24L English pack, resident raw-F32 path:

```text
MYNAH_QUANT_GROUPS=none
MYNAH_CUDA_RESIDENT=1
MYNAH_CUDA_FLOW=1
MYNAH_CUDA_POCKET_CODEC=1
MYNAH_CUDA_POCKET_DECODER=1
MYNAH_CUDA_DECODER=1
MYNAH_CUDA_CODEC_BATCH=0
MYNAH_CUDA_CODEC_DEVICE_HANDOFF=1
MYNAH_CUDA_CODEC_HOST_MIRROR=0
MYNAH_CUDA_GRAPHS=1
MYNAH_CUDA_DECODER_GRAPHS=1
MYNAH_CUDA_DECODER_BATCH=1
MYNAH_CUDA_DECODER_ARITHMETIC_BATCH=1
```

The clean C64 ladder completed 64/64 HTTP requests at every level with no
disconnects or synthesis errors. GPU utilization reached 99%; peak VRAM was
about 21.8 GiB at C64. The 24L request produced 3.68 s of PCM audio. Aggregate
audio throughput was approximately 6.60 audio-s/s at C40 and 6.82 audio-s/s at
C64. This is a capacity baseline, not a production qualification: the
per-request decoder graph is present, but the full Flow-to-Mimi-to-PCM frame is
not yet one graph.

The true multi-row Mimi transformer path remains opt-in because the first
real-weight parity gate found a large final-waveform delta after the different
cuBLAS accumulation order. The safe default uses resident CUDA Mimi work per
request and still batches the outer decoder/SEANet work where eligible.

The first C96 startup also exposed a server-side fixed-64 worker array: the
automatic worker count followed the wider batch and corrupted the language
metadata returned by `/health`. The worker clamp and array now use the shared
128 active-slot ceiling; this is a required capacity fix, not a benchmark
result.

## Findings from the local implementation comparison

Useful patterns to evaluate here:

1. A preallocated row arena with stable per-row KV offsets, swap-remove
   compaction, and fixed width buckets avoids rebuilding pointer tables and
   makes graph addresses stable.
2. Encode each voice prompt once and retain immutable per-layer device prefixes;
   admission should copy device-to-device into a row instead of re-encoding or
   mirroring the prefix through the host.
3. Batch right-padded text prefill and use continuous iteration-level admission
   so rows join and leave between frame ticks.
4. Capture a complete frame graph per width bucket. The graph should include
   device-side random sampling/EOS and Mimi/SEANet work, with one bounded host
   handoff per frame.
5. A reduced-precision KV cache is a capacity/throughput option, not a default
   correctness change. It needs a waveform/log-mel gate and a per-model opt-in.
6. The current attention kernel is length-aware, which must be preserved; a
   split-K or flash-style reduction is preferable to reading a fixed maximum
   context for every row.

Patterns deliberately not adopted: global RNG state, full-capacity attention
for short rows, unbounded output queues, and failure handling that can kill the
engine thread.

## CPU qualification ideas that transfer to CUDA

A recent 32-core Arm CPU qualification campaign supplies useful CUDA design
constraints, but not kernels to copy verbatim:

1. Model-aware text segmentation is portable. Keep sentence-boundary chunks
   bounded before prefill, and size the prefill slice from the model's measured
   token cost. This protects TTFA and inter-chunk gaps for the 24L graph just as
   it protects CPU workers; it must remain a request-level policy, not a CUDA
   graph shape baked into one model.
2. CPU-wide matrix tiles become batched CUDA GEMM shapes. The equivalent test is
   to choose a small set of supported row widths and capture the complete
   frame at those widths, rather than adding a device kernel for every request
   count.
3. Pooled CPU attention maps to row-batched attention with length-aware
   reductions. It is useful only if it reduces serving RTF; an isolated kernel
   speedup that increases synchronization or tail latency is rejected.
4. Keep scheduler capacity and device microbatch width independent. A wider
   `max-inflight` pool is useful for continuous admission, while the GPU graph
   width stays bounded by VRAM and the TTFA/RTF gates.
5. Preserve the qualification gates: zero stalls, completion, TTFB, TTFA,
   streaming RTF and drift. GPU utilization alone is not a streaming result.

CPU-specific worker counts, Arm I8MM/NEON instructions, and decoder INT8 are
not CUDA changes. The analogous GPU experiments must be measured against the
same audio and service gates; the decoder remains full precision until its
quality boundary is proven.

## BF16 backbone KV capacity experiment

The opt-in `MYNAH_CUDA_KV_DTYPE=bf16` path stores only the Pocket backbone K/V
cache in BF16. Q/K/V projections, attention accumulation, the host shadow and
the codec cache remain FP32; the default remains FP32. It uses the same model
metadata and the same CPU/CUDA graph, with no six-layer special case.

On one NVIDIA L4 (`sm_89`, 23,034 MiB), the official 24L pack passed both the
CUDA self-check and the batched self-check. A single exploratory ladder had no
HTTP failures, disconnects or synthesis errors at every level:

| level | first audio p50/p95 | peak VRAM | aggregate audio/s | GPU max |
|---:|---:|---:|---:|---:|
| C64 | 12.921 / 12.931 s | 12,094 MiB | 6.629 | 99% |
| C68 | 13.796 / 13.804 s | 12,760 MiB | 6.691 | 100% |
| C72 | 14.062 / 14.070 s | 13,428 MiB | 6.837 | 100% |
| C76 | 14.820 / 14.828 s | 14,094 MiB | 6.837 | 100% |
| C80 | 15.588 / 15.596 s | 14,764 MiB | 6.980 | 100% |

The previous FP32 24L run used about 21,948 MiB at C64 and hit CUDA OOM at
C68. This makes BF16 a useful capacity gate, not yet a production quality or
30-minute qualification: the ladder was one wave per level and still needs
repeated TTFA/RTF/audio-parity runs. The 6L CUDA self-check also passes with
the BF16 path, preserving the small-model CPU oracle and the FP32 default.

## Post-merge regression

After integrating the CPU 6L/24L serving changes into the CUDA branch, the
same L4 was rebuilt from the unified checkout:

```text
make clean && make cuda-server cuda CUDA_ARCH=sm_89       EXIT:0
6L CUDA BF16 pocket self-check + batching self-check       PASS
24L CUDA BF16 pocket self-check + batching self-check      PASS
```

The 24L post-merge HTTP smoke used one CUDA server process, one engine thread,
BF16 backbone KV, resident flow/Mimi/decoder and graph capture. It returned
HTTP 200 chunked PCM with 126,720 bytes (2.64 s), 0 failed requests, TTFA
199 ms, scheduler E2E 893 ms and RTF 0.338. The process reported four graph
captures, 95 graph replays, zero graph fallbacks and zero decoder failures;
cross-request batch counters were zero because this was deliberately a single
request smoke, not a throughput measurement.

The CLI self-check still warns that its intentionally long 96-token probe is
not segmented; the server-side CPU/HTTP segmentation path is a separate
request policy and remains enabled after the merge.

## Plan and acceptance gates

- [x] Raise shared scheduler/CUDA metadata/decoder graph capacity to 128 and
  build it on `sm_89`.
- [x] Remove the server worker-array/fixed-64 clamp that corrupted `/health`
  when a wider CUDA batch selected more than 64 HTTP workers.
- [x] Add opt-in BF16 backbone K/V storage with FP32 attention accumulation and
  keep the FP32 default unchanged; 6L and 24L CUDA self-checks pass.
- [x] Probe the 24L capacity ladder above C64 with BF16: C80 is clean at
  14,764 MiB peak in the exploratory wave; repeat/soak qualification remains.
- [ ] Measure small-model C64/C80/C96 after the capacity change; record VRAM,
  GPU utilization, errors, TTFA and aggregate audio throughput.
- [ ] Repeat the 24L BF16 ladder with accepted multi-run TTFA/RTF/audio gates
  and compare against CPU stage/EOS/audio outputs.
- [ ] Keep CPU and single-request CUDA audio/intermediate parity as a gate for
  every batching change.
- [ ] Prototype device voice-prefix storage and a stable row-arena layout
  behind feature flags; compare host/device transfer and memory before making
  either the default.
- [~] Prototype BF16 KV storage and a complete-frame graph separately; BF16
  storage is implemented and capacity-positive, while finite audio, duration,
  audio tolerance and a real repeated throughput win remain acceptance gates.
- [ ] Add startup graph capture only for buckets that fit the selected model and
  device memory; never make startup capture block the HTTP event loop.

## Reproduction

Build on an NVIDIA Linux host:

```text
make cuda-server cuda CUDA_ARCH=sm_89
```

Run the safe 24L server with the environment above, then use the repository
streaming client/ladder with one load-generator process and levels 1, 4, 16,
40, 64. Collect `/health`, `/metrics`, `nvidia-smi` samples, TTFA, completion
time, PCM byte count, RSS and error/disconnect counts. Repeat the same ladder
for the 6L pack before changing a kernel or enabling experimental Mimi batch.

## Current decision

The next safe performance step is capacity measurement and memory accounting,
not speculative kernel replacement. The measured L4 C64 24L result proves a
healthy resident CUDA path but does not prove C100 or justify raising the
production claim until the VRAM and frame cadence gates pass.
