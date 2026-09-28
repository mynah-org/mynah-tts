IN PROGRESS

# Pocket CUDA on 4-vCPU hosts (g6.xlarge): host CPU, stalls, next speedups

## Problem

The C60 result in `.work/pocket-cuda-c60-l4.md` was measured on a vast.ai L4
with 128 host cores. Production L4s are usually AWS `g6.xlarge`: one L4 and
**4 vCPUs**, shared with the HTTP layer and the client-facing process. On
that host the CUDA path cannot lean on the CPU the way it does now:

- the single scheduler thread runs at ~110-117% CPU at C64 (host bookkeeping,
  pointer tables, per-request memcpy, PCM callbacks) and is already the next
  limiter on the 128-core box;
- eight HTTP worker threads plus one writer thread per stream compete with
  it; on 4 vCPUs they steal the scheduler's core;
- the closed-loop soak still showed one 250 ms hiccup; on a starved host that
  becomes a stall pattern.

Target: the same C60 gates on a 4-vCPU host, with the CUDA path using as
little host CPU as possible, and further throughput on the 24L where the
GPU is still at 70-92%.

## Evidence to gather first (no changes before this)

1. Reproduce the 4-vCPU host on the vast box: `taskset -c 0-3` for the server
   (and the load generator on other cores, or a second machine), then the
   C16/C32/C48/C64 ladder and a 10-minute soak. Record scheduler CPU%, HTTP
   worker CPU%, stalls, RTF, TTFA.
2. Per-frame host cost map at C64 with `MYNAH_COST_MAP=2` and `perf`-less
   sampling (`MYNAH_SERVE_PROFILE=1`, NVTX): what the scheduler thread does
   between GPU launches. Suspects: pointer-table uploads per layer per frame,
   per-request host memcpy of PCM, `pocket_all_finite` scans on host,
   float→PCM16 conversion on host, the KV mirror path when not device-owned,
   per-frame `malloc`, the `cudaGraphInstantiate` churn (48 instantiations in
   a 10 s C48 trace), `/health`/`/metrics` string building.
3. Nsight kernel time at C64 on the merged tree: after ConvTranspose, what is
   left (backbone TF32 GEMMs, `k_self_attention_bf16_batch`, tile attention,
   `k_decoder_causal_columns_batch`, `k_conv_transpose_causal_depthwise_step`).

## Candidate levers (measured order, each behind a switch, ABAB, KEEP/REVERT)

- Host: PCM16 conversion and clipping on the device, one D2H of int16 per
  frame for the whole gang; drop `pocket_all_finite` host scans on the hot
  path in favour of a device NaN flag; persistent pointer tables (row arena)
  so no per-layer H2D of tables; fewer HTTP workers by default on CUDA
  (`-w 2`), writer threads coalescing frames; pin the scheduler thread.
- Device: fuse LayerNorm+GEMM epilogues (bias, residual, GELU) into the cuBLAS
  Lt epilogue; batched attention for the backbone step with one block per
  (row, head) reading BF16 K/V in 128-bit loads; per-width graph buckets
  captured at start-up (no instantiate churn); one complete-frame graph per
  bucket once host decisions are device-side (EOS flag, RNG).
- Weights: `MYNAH_CUDA_QUANT=bf16` is 4% slower with the deterministic tile;
  a cuBLAS BF16 tile was +7% but needs a 4e-3 gang band. INT8 weights on
  tensor cores with the same batch-invariance rule; INT8/FP8 KV only after a
  cost model shows KV bandwidth matters at C64.
- Shapes: avoid the `[dim][positions]` scatter into SEANet input by making the
  decoder read row-major; avoid the per-frame `k_tile_gather`.

## Acceptance gates

**Primary production-readiness gate: the 30-minute continuous closed-loop
soak** (mixed short/medium/long, concurrency saturated, no idle gaps:
`tools/l4/soak.sh <tag> <C> 1800` with the matching `MYNAH_L4_BATCH`). Every
new top concurrency (C96/C112/...) is qualified with it first; Poisson is
only an extra workload characterisation, never a substitute.

Same as `.work/pocket-cuda-c60-l4.md` at C60, measured with the server
confined to 4 cores, plus: scheduler thread < 70% of one core at C60, zero
stalls at 250 ms over a 30-minute soak, and CPU `make test` green after every
shared-code change (the CPU path stays untouched).

## Final qualification protocol

A top concurrency is production-qualified only by this sequence, on one
recorded build:

1. **Two independent 30-min saturated closed-loop soaks** at that C, each on
   a freshly started server (`tools/l4/soak.sh <tag> <C> 1800`), different
   tags and ladder seeds, run one after the other.
2. **v2 corpus**: `tools/corpus/pocket_v2_en.jsonl`, 300 distinct original
   English utterances (short / conversational / medium / long paragraphs;
   lists, parentheses, quotes, dashes, ellipses, numbers as words, names,
   dialogue; American spelling so the ASR does not score spelling), drawn
   with weights .25/.35/.28/.12 and four voices, deterministic by seed.
3. **Captured audio**: `MYNAH_L4_SAVE_AUDIO` makes the ladder keep the PCM
   exactly as the server streamed it under load (every Nth request id), as
   WAV, never re-synthesised. The per-request JSONL names corpus id, text,
   voice, seed, WAV and sample count for each.
4. **WER on the captured audio**: `tools/pocket_quality.py --from-jsonl`
   (faster-whisper small.en int8 CPU, parallel workers) gives WER overall,
   per kind and per voice, every utterance > 30% WER, and auto-flags
   words/s outside [1.2, 5.0], silences > 1.5 s (lead, trail, internal),
   clipping > 0.1%, empty/silent streams and an abrupt end (last 50 ms loud).
5. **Human listening**: `audio-<tag>.zip` holds the same WAVs plus a
   manifest (text, voice, WER, flags) for spot listening, flagged rows first.
6. **Bundle**: `tools/l4/bundle.sh <name> <tagA> <tagB>` collects git HEAD
   plus sha256 of the binaries and sources (the box tree may be rsynced), the
   server profile and its `MYNAH_*` env, the model pack's `source.json` and
   `model.json`, the corpus, each soak's summary / per-request JSONL /
   per-window report / server log / VRAM log / quality CSV and summary, the
   audio ZIPs, and a `REPORT.md` with the headline numbers filled in
   (throughput, TTFA and RTF p50/p95, stalls, failures, VRAM, WER) and the
   listening and verdict sections left to a human.

Pass: both soaks meet the gates above with zero failures and zero 250 ms
stalls, no drift between halves, WER in line with the quiet-server gate and
no unexplained flagged audio. At C128 (~130 audio-s/s) one soak streams
~234,000 audio-s, 10.5 GiB of PCM16; `MYNAH_L4_SAVE_EVERY=10` keeps ~3,000
WAVs (~1 GiB, ~6.5 h of audio) per soak, enough for every corpus line
several times in every voice.

## Baseline on the new box (2026-09-28, tree 72cade4+, 24L, C16-C64 x 20 s)

The box is an EPYC 7702 with 128 threads; `taskset -c 0-3 -w 4` models the
g6.xlarge. Packs re-converted from `english_2026-04_24l` / `english_2026-04`.

| run | C16 | C32 | C48 | C64 aud/s | C64 RTF p95 | C64 TTFA p95 | srv CPU | GPU util |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| unpinned, -w 8 | 53.9 | 79.6 | 91.4 | 100.1 | 0.483 | 94 ms | 117% | 94% |
| ~~4 cores, -w 4~~ (NOT pinned, see below) | 53.1 | 76.7 | 88.7 | 97.7 | 0.492 | 94 ms | 117% | 93% |

**Correction (same day):** `ab.sh` started `serve.sh` inside `tmux new-session`,
which does not inherit the caller's environment, so `MYNAH_L4_CPUS` and
`MYNAH_L4_WORKERS` were ignored: the "4 cores" rows ran on all 128 threads with
`-w 8` (checked with `taskset -cp`). The second row is therefore a repeat of
the first, and the claim that 4 vCPUs cost only 2.5% is withdrawn. The A/B rows
of the board stay valid as relative comparisons (every run had the same
unpinned setup). Fixed in the tools; the real 4-core baseline is re-run.

Findings:

- The ~117% CPU is mostly the driver spinning in `cudaStreamSynchronize`
  (device flags were `cudaDeviceScheduleAuto`).
- The GPU is 93-94% busy, so throughput past C64 has to come from device time.
  B64 step: 34 ms mean, 0.5 ms per slot.
- `decoder_graph_captures` > `decoder_graph_replays` at every level (C64: 575
  captures for 207 replays). The SEANet decoder graph is re-captured more
  often than it is reused. This is a device and host cost to chase.

## Audit (2026-09-28, three read-only passes: host CPU, GPU step, web)

Ranked by expected gain at C64; each item gets a switch and a board row.

1. **Backbone decode attention** (`k_self_attention_bf16_batch`): the kernel
   walks the context one position at a time, with a 64-thread tree reduction
   and ~9 `__syncthreads` per position, for 24 layers x 16 heads x 64 rows at
   ~250 positions. It is issue-bound, and it fits the measured 0.5 ms/slot
   slope of the B64 step. Rewrite: chunked online softmax, one thread per
   position, 128-bit BF16 K loads, V accumulated in registers, fixed order per
   row. Switch `MYNAH_CUDA_BACKBONE_ATTN=legacy`.
2. **SEANet decoder graph cache**: the key is the exact ordered gang, so every
   admission or retirement misses (capture + instantiate + `cudaMalloc`/
   `cudaHostAlloc`), and decoder close destroys every graph holding it. Fix:
   one graph per width, re-recorded only to rewrite the pinned pointer tables
   that its memcpy nodes read at launch. Switch
   `MYNAH_CUDA_DECODER_GRAPH_REUSE=0`.
3. **Admission and retirement allocate on the scheduler thread**: ~7
   `cudaHostAlloc`, backbone KV, codec and decoder `cudaMalloc`, and on retire
   dozens of `cudaFree`, each of which synchronises the device. There is also
   a ~100 MB host KV `calloc` per request that device-owned rows never touch.
   Fix: a slot pool of `max_batch` resource sets created at start-up, plus a
   decoder reset instead of close.
4. **Five syncs per frame** (condition, backbone, EOS, flow, PCM), with the
   GPU idle while the host works after each one. `MYNAH_CUDA_SYNC=blocking`
   proved each wake-up gap costs throughput. Fix: condition + backbone + EOS +
   flow in one graph with one sync (noise pre-generated on the host), and
   drop the 256 KB condition D2H and the hidden re-upload.
5. **SEANet fusion**: ELU folded into im2col; kernel-1 convs straight to the
   GEMM; bias via beta; `CUBLAS_COMPUTE_32F_FAST_16F` for the decoder only.
6. **Shared voice-prefix KV** (read the device voice cache instead of a copy
   per request): ~40% less KV traffic when voices repeat, less VRAM, no copy
   at admission.
7. **Backbone per-layer tables**: 120 memcpy nodes per step (positions and
   strides uploaded 24 times). Upload once per step, and use cuBLASLt
   bias/GELU epilogues (15 -> 8 kernels per layer).
8. **Stream output**: `writev` per chunk, a per-slot PCM buffer instead of
   `malloc` per frame, and cancellation read from an atomic set by the writer
   instead of `poll` + `recv(MSG_PEEK)` per row per frame.
9. Later: BF16 backbone and flow with the Mimi tile kept on TF32 cuBLAS (the
   BF16 slowdown came from the Mimi tile falling back to the SIMT kernel); a
   tensor-core batch-invariant tile GEMM for prefill; B>64 once the step is
   lean; codec every 2 frames for streams with a deep buffer.

## VRAM per stream (audit 2026-09-28, matches the measured ~178 MB)

| component | bytes per stream |
|---|---:|
| backbone KV, BF16: 24 x 2 x (126 voice + T text + 1501 steps) x 1024 x 2 B | ~165 MB at T=50 |
| Mimi codec KV window, FP32: 2 x 2 x 500 x 512 x 4 B | 4.1 MB |
| SEANet per-step temporaries (work_a/b/c, im2col `columns`, conv windows, convtr full) | ~5.2 MB |
| real causal state, activations, upsample, decoder IO | < 0.3 MB |

The KV is sized for the 1500-frame (120 s) default step budget; a typical
request writes ~300 of ~1677 positions. Fixed cost ~1.8 GB (weights 1.3 GB and
shared buffers), so ~115 streams fit on 23 GB, hence the C128 OOM. Levers:
grow the KV in 256-position chunks (`MYNAH_CUDA_KV_GROW`, in progress):
~49 MB per stream; share the SEANet temporaries across the gang (-5 MB);
codec window 266 instead of 500 and BF16 (-3 MB); trim the slot pool above
the in-flight high-water mark (idle VRAM only). BF16-resident weights save
only ~0.65 GB once.

## Board

| candidate | baseline | candidate | delta | decision |
|---|---:|---:|---:|---|
| `MYNAH_CUDA_SYNC=blocking` (host sleeps in sync) | C64 97.7 aud/s, 117% CPU | 88.9, 43% CPU | -9% | REVERT as default (opt-in only): each wake-up leaves the GPU idle |
| `MYNAH_CUDA_SYNC=yield` | 97.7, 117% | 96.6, 116% | -1% | REVERT (no CPU saved) |
| decoder graphs off (`MYNAH_CUDA_DECODER_GRAPHS=0`, diagnostic) | 97.7 | 98.4 | +1% | the re-captured graphs were worth nothing |
| decoder graph reuse per width (`MYNAH_CUDA_DECODER_GRAPH_REUSE`, 97e1e0c) | C64 96.7, 568 captures / 206 replays | 98.8, 10 captures / 826 replays | +2% | KEEP (default on); WER 3.94% vs 3.49% off, 0/96 utts > 30% |
| backbone attention rewrite (`MYNAH_CUDA_BACKBONE_ATTN`, 97e1e0c) | C32 76.8, C48 89.9, C64 98.0 (RTF p95 0.489, TTFA p95 98 ms) | 99.9, 110.1, **119.4** (0.431, 87 ms) | **+22-30%** | KEEP (default on); self-check PASS with both kernels; WER 3.76%, 0/96 > 30% |
| slot pool (`MYNAH_CUDA_SLOT_POOL`, 009fb44) | C64 119.1 unpinned / 116.4 on 4 cores | 123.5 / **121.7** | +3.7% / +4.6% | KEEP (default on); leak test: the same request before and after another one is md5-identical with pool 0, 1 and 1 + ZERO_KV; self-check PASS |
| true 4-core host (`taskset -c 0-3 -w 4`, affinity logged) | C64 123.5 unpinned | 121.7, server CPU 110% of 400% | -1.5% | a g6.xlarge host does not limit the L4 |
| max-batch/inflight 96 | C64 at 64/64: 123.5 | C80 119.5 (RTF p95 0.53, TTFA p95 106 ms), C96 117.4 (0.61, 121 ms), 0 stalls | flat aud/s, +50% streams | C96 realtime on one L4; GPU-bound plateau ~120 aud/s |
| max-batch 128 (4 cores) | | C112 113.7 (RTF p95 0.70); C128 fails (VRAM 22.4 GB, 20 streams failed) | | ceiling ~C112 at the current per-request VRAM (~180 MB/stream) |

| tile workspace grow-only (was 6 cudaFree + 6 cudaMalloc at every prefill <-> Mimi dim flip) | C32 102.0, C48 110.3, C64 122.0 | 106.8, 118.9, **129.6** (TTFA p95 78 ms) | +5-8% | KEEP; Nsight 12 s at C64: cudaMalloc 2781 -> 5, cudaFree 2781 -> 5 |
| prefill tile through cuBLAS TF32 (`MYNAH_CUDA_PREFILL_FIXED=0`, diagnostic) | fixed-order SIMT: C64 127.6, C96 130.4 | C64 134.3, C96 **141.2** (RTF p95 0.52, TTFA p95 93 ms) | +5-8% | the prefill split-K SIMT GEMM was 19.5% of GPU time; WER 3.67%. Loses piecewise-prefill bit identity, so not a default yet |
| WMMA TF32 fixed-order tile GEMM, no split-K (`MYNAH_CUDA_TILE_TC`) | SIMT C64 127.6 | 95.4 | -25% | REVERT (default off): ~48 blocks for a few-row prefill; self-check PASS, WER 3.49%. Retry with split-K |

| WMMA TF32 fixed-order with split-K (`MYNAH_CUDA_TILE_TC=1`) | SIMT C64 127.3, C96 130.4 | 128.4, 133.4 | +1-2% | opt-in only: invariant (self-check PASS, WER 3.14%) but far from cuBLAS; needs a pipelined kernel (cp.async double buffer, 128-wide tiles) to be worth it |
| cuBLAS prefill vs fixed order, re-run | fixed C64 127.3, C96 130.4 | C64 133.0, C96 141.8 | +4-9% | opt-in only: the long-form self-check fails (text pushed in pieces vs whole: sample 12003/23040 -0.2719 vs -0.2706). Determinism vs throughput is a product decision |
| pack converted with `--dtype source` (Kyutai F32, 1.30 GB) vs default BF16 pack (672 MB) | BF16 pack: WER 3.32%, C64 94.4 (TC binary) | F32 pack: WER 3.41%, C64 95.1 | = | keep the BF16 pack (Kyutai's own `switch_to_bf16.py` choice); CUDA expands weights to F32 either way |

| **30-min closed-loop soak C96** (tree bb498a6 defaults, max-batch 96, unpinned) | | 38,140 requests, 0 failed, **176.8 aud/s**, RTF p95 0.541, TTFA p95 97 ms, 0 stalls @250/@500, RTF p95 drift +0.6%, VRAM 18.48 -> 18.72 GB, RSS 3.71 -> 3.73 GB | | screening PASS (single soak, built-in text mix, no audio capture; the final qualification needs two soaks on the v2 corpus with capture). Short ladders under-count long requests: soak aud/s is the reference |

Rows above the slot pool ran unpinned with `-w 8` (see the correction above);
all are C32-C64 x 20 s, 24L, BF16 KV unless stated.
