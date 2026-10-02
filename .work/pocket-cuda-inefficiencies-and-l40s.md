# Pocket CUDA: remaining inefficiencies, fusion, FP8 and L40S readiness (2026-10-02)

Read-only analysis of branch `pocket-cuda-c208` at **ed5072b** (shared voice phase 2, decoder fusion part 2 and bf16
decode linears already merged). Line numbers refer to that commit. Labels: **CONFIRMED** = read in the code or
measured (source given); **HYPOTHESIS** = an estimate to be measured. Work already in flight (shared voice phase 2,
bf16 decode linears + `MYNAH_CUDA_BF16_FUSE`, decoder residual/bias/convtr fusion, `MYNAH_CUDA_PREFILL_BF16TC`) is
not repeated; interactions are noted.

Reference numbers used throughout (C160, L4, `.work/pocket-cuda-c208.md`): step ~64 ms per 80 ms frame; NVTX
`step.backbone` 23.7, `codec.total` 22.7, `step.embed` 9.9 ms; decode attention 15.7 ms (before shared voice);
SEANet ~17 ms of which ~12.6 ms glue (im2col 3.1); prefill SIMT tile 8.7 ms; backbone GEMMs ~7 ms. The NVTX
regions sum to ~56 ms, so ~8 ms per step is flow head + EOS + host/sync gaps.

## Executive summary: top 5 actions

| # | action | expected gain (C208, L4) | effort | risk | basis |
|---|---|---|---|---|---|
| 1 | **One device chain and one sync per frame**: condition -> backbone -> EOS -> flow -> LSD add -> codec -> SEANet -> PCM without the 6 intermediate host round trips (noise pre-generated on the host, EOS/latent/NaN flag read back with the PCM) | 3-6 % step, more on a faster GPU; also lower scheduler CPU | M-L | M | ~7 syncs/step CONFIRMED (`/health`: 5,628 syncs / 811 steps; code below); gain HYPOTHESIS |
| 2 | **8-bit suffix KV written inside the attention kernel** (FP8 E4M3 or INT8, static per-(layer, head) scale): the fast kernel already writes the new K/V itself, so quantize-on-write costs no extra pass | 3-6 % step (attention is the largest bandwidth item after shared voice) | M | M (quality gate) | kernel layout CONFIRMED; gain HYPOTHESIS |
| 3 | **Decode-attention efficiency audit then rewrite** (ncu achieved DRAM BW first): a reference PyTorch engine's Triton split-K two-segment kernel takes ~4.0 ms/step at ~160 rows vs 15.7 ms here at C160 (before shared voice) | 0-5 ms/step depending on what ncu shows | S (measure) / M (rewrite) | L-M | both numbers CONFIRMED (different segment lengths, see 1.8); cause HYPOTHESIS |
| 4 | **Per-step metadata once, not per layer**: today 8 H2D memcpy graph nodes per layer (RoPE positions + K/V/positions/strides + 3 prefix tables) = ~192 per step; build per-layer tables on the device once per step and index them by layer | 0.5-1 ms/step | S | L | CONFIRMED in code; gain HYPOTHESIS |
| 5 | **L40S unlock as runtime config**: the 384 compile-time ceilings (`POCKET_MAX_BATCH`, `MYNAH_GRAPH_MAX_JOBS/ACTIVE`, `CUDA_BATCH_META_CAP`), the default width-bucket list ending at 384, and KV growth with malloc/copy/free in the step pre-flight | enables C>384 on 48 GB; removes growth stalls and the C224 growth-peak OOM | S (caps) / M (VMM growth) | L / M | CONFIRMED |

## 1. Hot-path inefficiencies

### 1.1 Blocking syncs per frame (CONFIRMED in code; ~7 per step measured)
All waits are `cudaStreamSynchronize` through `mynah_backend_sync`, with the default `cudaDeviceScheduleAuto`
(`gpu/cuda/backend_cuda.cu:3088`), i.e. a spinning scheduler thread. Between two syncs the GPU idles while the
host does per-row memcpy, `pocket_all_finite` scans and pointer-table building.

| # | stage | where | what crosses | needed? |
|---|---|---|---|---|
| 1 | input projection (condition) | `pocket_cuda_condition_batch`, sync `src/engine_pocket.c:1963` | latents H2D (built on host, `:1941`), step input D2H 4 KB/row, host finite scan `:1974` | no: the device input is reused (`cuda_condition_ready`), the D2H only feeds a host scan |
| 2 | backbone step | `pocket_cuda_backbone_step_batch`, D2H `:7337`, sync `:7358`, scan `:7361` | hidden D2H 4 KB/row | only because flow/EOS are fed from the host |
| 3 | EOS head | `pocket_cuda_eos_batch`, sync `:2022` | 1 float/row | no: EOS does not change this step's latent |
| 4 | flow head | `pocket_cuda_flow_step_batch`, cond H2D staged from host `:7485`, D2H + sync `:7599-7602`, scan `:7616` | hidden re-uploaded (the same bytes that left in #2), noise H2D, flow out D2H | no: hidden is already on the device |
| 5 | noise + LSD add on host | `pocket_emit_batch` `:9673`, `:9740` | 32 floats/row | the RNG must stay the host's for bit identity; it does not depend on the step, so it can be produced ahead |
| 6 | codec transformer | `pocket_cuda_codec_transform_batch`, sync `:10753` | device handoff on; the sync ends the stage | no |
| 7 | SEANet + PCM | `pocket_decode_audio_batch`, sync `:11150` | PCM D2H (float, 7.7 KB/row) | yes: the one sync per frame that must stay |

- Fix (action 1): keep #1-#6 on the stream; pre-generate each row's noise for the next N steps on the host (same RNG
  sequence, so bit identity holds) into a pinned ring that the graph reads; LSD add, denormalisation and the codec
  input on the device; EOS logit, latent (32 floats) and a device NaN flag D2H next to the PCM; host bookkeeping for
  frame N after the single sync. The rollback contract still works: every step is atomic up to the final sync.
- Measure first (cheap): `[SERVE]`/NVTX gap between the end of one region and the start of the next GPU kernel at
  C208, or simply nsys "GPU idle" per step. Expect ~3-8 ms of idle (HYPOTHESIS: 64 - 56 ms NVTX residue at C160,
  nvidia-smi util 85-95 % per `docs/cuda-serving.md`).
- On a faster GPU this matters more: the gaps are host-time, not GPU-time.

### 1.2 Avoidable copies per step (CONFIRMED)
- Hidden D2H + re-upload for the flow head (#2/#4): 2 x 4 KB/row, ~1.7 MB per step at C208; tiny in bytes, but it
  serialises the stage chain.
- Condition D2H (#1): output only used for a host finite scan.
- Flow time embedding H2D every call (`cuda_flow_host_time`): constant per model; upload once.
- PCM D2H in float, then `stream_out_enqueue` converts to int16 on the scheduler thread (`server/stream_out.c:319`):
  convert on the device and copy int16 (half the bytes, no host loop). Small.

### 1.3 Allocations in the hot loop
- **KV growth** (CONFIRMED, `pocket_cuda_backbone_reserve`, `src/engine_pocket.c:6461`): called from the step
  pre-flight (`:9332`); per growth `cudaMalloc` (`:6505`), 2 x 24 device copies (`:6519`), a stream sync, and
  `cudaFree` of the old cache (`:6538`, an implicit device sync). Growth is 256 positions (`:4516`). The slot pool
  keeps grown caches, so growths fade with uptime, but bursts of new long requests still pay it, and the transient
  old+new allocation is where C224 OOMed (22.5 GB). Fix: CUDA VMM (`cuMemAddressReserve` for the slot's maximum,
  `cuMemCreate`/`cuMemMap` 256-position chunks on demand): growth in place, no copy, no free, no sync, no peak.
- `pcm[g]` malloc/free per gang member per step (`src/inference.c:858`, `:864`): ~200 small mallocs per step;
  negligible on glibc, but a per-slot buffer would honour the no-allocation rule of the loop.
- Decoder gang graphs are keyed by exact width (no inert rows), so bursts still capture ~135 decoder graphs during
  traffic (`.work/pocket-cuda-cold-burst.md`). Each capture + `cudaGraphInstantiate` is host time on the scheduler
  thread. Fix: inert pad rows for the gang (a private causal ring per pad row, as `cuda_pad_kv` does for the
  backbone) so the decoder can use the width buckets.

### 1.4 dtype conversions
- Default path: fp32 resident weights on TF32 cuBLAS; `MYNAH_CUDA_BF16_FUSE` removes the f32<->bf16 round trips of the
  backbone (in flight). Remaining:
  - Flow head and Mimi transformer stay fp32/TF32 (`MYNAH_CUDA_QUANT_STAGES` covers them; the reference engine kept
    them fp32 for quality and still reached C208).
  - Codec KV window is fp32 (2 x 2 x window x 512 x 4 B per row): bf16 would halve the Mimi attention reads (~1 ms
    at C208, HYPOTHESIS).
  - SEANet: bf16 operands, fp32 states and glue; the glue kernels read/write fp32 activations (the decoder fusion
    branch removes passes, not the fp32 width).

### 1.5 Work outside graphs / graph overhead
- Per layer the backbone graph replays 8 small H2D memcpy nodes (CONFIRMED: RoPE positions
  `gpu/cuda/backend_cuda.cu:9763`; K/V tables, positions, strides `:9513`/`:9561`; three prefix tables with shared
  voice). That is ~192 memcpy nodes per step, each with its own launch latency, plus the same pattern in the Mimi
  tile. Fix (action 4): one H2D per step of a `[layer][row]` table block (or row-base + layer-stride arithmetic in the
  kernel, since every row's layer planes are equally spaced inside its allocation).
- Separate graphs per stage (condition, backbone, flow, codec, decoder) with host work between them; a full-frame
  graph per width bucket becomes possible after action 1.

### 1.6 Host threads (CONFIRMED in code; CPU measured ~110-125 % of one core at C64-C160)
- Scheduler thread spins in `cudaStreamSynchronize` (`MYNAH_CUDA_SYNC=blocking` measured -9 %: keep the spin until
  there is one sync per frame; then a blocking event per frame costs one wake-up per 64 ms instead of seven).
- One writer thread per stream (`server/stream_out.c:287`): at C208, 208 threads and ~2,600 wake-ups/s; fine on
  4 vCPU today, ~7,500/s at C600.
- Cancellation: `poll(fd, 0)` per row per step on the scheduler thread (`server/stream_out.c:410` via
  `sink_cancelled`, `server/main.c:826`): ~200 syscalls per step while the GPU waits. Let the writer publish a
  "peer gone" atomic (it already sees EPIPE first) and poll only every N steps.
- HTTP workers default to `--max-batch` when `-w` is not given (`server/main.c:2951`); the profile sets `-w 8`.

### 1.7 GPU power cap (CONFIRMED on the same GPU type with a reference PyTorch engine, HYPOTHESIS here)
Under C160-C208 load an L4 sits at its 72 W limit: 71.5 W mean, SM clock 1.51-1.55 GHz mean (1.2-2.04 GHz) instead
of 2.04 GHz, utilisation 80-86 % (nvidia-smi 1 s samples over 30 min). Consequences: compute-bound kernels run ~25 %
below spec; every byte or instruction saved also buys clock. Log `nvidia-smi --query-gpu=power.draw,clocks.sm` in
the knee scripts so A/B arms are compared at equal power state.

### 1.8 Decode attention: why 15.7 ms when a reference kernel takes ~4 ms (CONFIRMED numbers, HYPOTHESIS causes)
- Here: `k_self_attention_bf16_batch_fast` 15.7 ms per step at C160 (nsys, before shared voice), ~4.7 GB of K/V
  read per step, i.e. ~300 GB/s, the L4's DRAM peak. A reference PyTorch engine's Triton split-K two-segment kernel
  (shared voice prefix + per-row suffix) measured 4.0 ms per tick at ~160 rows in its server profile.
- The bytes differ, not only the kernel: (a) that engine reads the voice prefix from one shared L2-resident copy
  (now `MYNAH_CUDA_SHARED_VOICE`, measured -10/11 % RTF here); (b) its text segments are word-limited (20-30 words,
  first segment 10 words) against 50 tokens here (`MYNAH_TEXT_SEGMENT_DEFAULT_TOKENS`, first 24), and the KV
  restarts at every segment in both engines, so its per-row suffix is roughly half as long. Segment length is a
  prosody/product choice, but it is also the attention-cost knob (`MYNAH_POCKET_SEGMENT_TOKENS`).
- Kernel shape here: one 128-thread block per (row, head), one thread per position scoring a 128-position chunk with
  8 x 16-byte loads from rows 2 KB apart, 3 `__syncthreads` per chunk, V accumulated with 2-byte loads. Bandwidth
  bound by construction if the bytes are right.
- Next step (cheap): `ncu --kernel-name k_self_attention_bf16_batch_fast --metrics dram__bytes_read.sum,
  dram__throughput.avg.pct_of_peak_sustained_elapsed` on one replay at C208 with shared voice on. If DRAM throughput
  is near peak, the lever is bytes (8-bit suffix KV, segment length); if it is well below, rewrite the kernel as a
  cooperative tile (BLOCK_N positions x 64 dims per warp-group, coalesced 128-byte rows, `cp.async` double buffer,
  split-K for long rows).

## 2. Fusion opportunities not covered by the in-flight branches

| # | candidate | what goes | gain (C208) | effort | risk |
|---|---|---|---|---|---|
| F1 | RoPE + attention: rotate q and the new k inside `k_self_attention_bf16_batch_fast` (it already reads q, k, v and writes K/V) | `k_rope_qk_batch` / `k_rope_qk_bias_batch` per layer, one f32 qkv round trip | 0.3-0.8 ms | S | L (same fp32 math if the rotation code is shared) |
| F2 | Quantize-on-write KV (FP8/INT8) in the same kernel (action 2) | half of the suffix KV bytes | 3-6 % | M | M |
| F3 | SEANet without im2col: implicit-GEMM conv (cuDNN, or tap-wise GEMMs accumulating with beta = 1 over a time-major layout) | im2col write + reread (3.1 ms at C160) | 1.5-3 ms | L | M (new accumulation order: md5 changes) |
| F4 | EOS head as an extra output column of the last LN consumer (or a warp-dot in the flow-input kernel) | one GEMM launch + sync #3 | small alone, part of action 1 | S | L |
| F5 | Flow head: LN/modulate/gate chains per res block into one kernel each; time embedding constant | ~10-20 small kernels per step | 0.2-0.5 ms | S-M | L |
| F6 | Mimi transformer tile: LN+bf16 store, bias+GELU, bias+residual (the BF16_FUSE pattern) + bf16 codec KV | fp32 round trips at M = rows x 2 | ~1 ms | M | L-M |
| F7 | PCM16 conversion + clipping on the device after the last SEANet conv | host loop + float D2H | <0.5 ms host | S | L |
| F8 | Sampling/LSD add + denorm on the device (part of action 1) | host loops `:9673`, `:9740` | host time | S | L (noise still from the host RNG) |

Interactions: F1/F2 touch the same kernel as `MYNAH_CUDA_SHARED_VOICE` and the `BF16OUT` store of BF16_FUSE; do them
after those merge. F3 conflicts with the decoder-fusion branch's column kernels; do it after part 2 is measured.

## 3. FP8 and INT8

### 3.1 What exists here (CONFIRMED)
- No FP8 path at all (no `CUDA_R_8F_E4M3`/cublasLt in the tree; the build links `-lcublas` only).
- INT8: `MYNAH_CUDA_QUANT=int8` (or `MYNAH_CUDA_Q8=1`) = W8A8 per-row activation absmax with ties-away rounding
  (`k_q8_quantize_rows`, `backend_cuda.cu:425`), `cublasGemmEx` `CUDA_R_8I`/`CUBLAS_COMPUTE_32I` (`:3630`), then
  `k_q8_epilogue` (`:457`) for scales + bias: two extra passes per linear plus an int32 accumulator round trip.
  Status: diagnostic only. The L4 notes record a solo-vs-gang parity failure (max_abs 1.6e-4) for all stages,
  `backbone:int8` alone passing parity but no serving win ("launch/dequantization overhead costs far more"), and
  the CPU int8 backbone measured as a no-op under load (E12).

### 3.2 Why the reference PyTorch FP8 attempt gave nothing (CONFIRMED there)
Its W8A8 FP8 FlowLM linears (per-output-channel weight scale, dynamic activation scale via `torch._scaled_mm`)
measured the same as bf16: tick 44.5 vs 44.3 ms, knees within 0.003-0.007 RTF. Its profile shows the FP8 GEMMs
saving ~3.6 ms per step (4.3 vs 7.9 ms) and the extra elementwise passes adding ~3.4 ms (amax reduce, abs, divide,
cast, output scale, bias, slice/cast: ~8 extra kernels per linear, +1,365 kernels per step). Exactly the INT8 shape
above.

### 3.3 Could it pay here?
- **Weights (backbone)**: bf16 backbone weights are 0.60 GB per step, FP8 0.30 GB: at 300 GB/s that is ~1 ms per
  step, ~1.5 % of a 64 ms step, on top of BF16_FUSE (HYPOTHESIS). At M ~ 200 the bf16 GEMM is close to the L4 ridge
  (121 TFLOPS / 300 GB/s ~ 400 FLOP/B; intensity ~ 2M = 400), so FP8 tensor throughput adds little. It pays only if
  the quantization is free: `k_layer_norm_bf16` and `k_bias_gelu_bf16` already hold the row, so they can emit E4M3 +
  a per-row scale (one more block reduction), and `k_residual_bias_add` / the GELU consumer can apply
  `s_act[m] * s_w[n]`; static per-layer activation scales remove the reduction entirely. Needs cublasLt (FP8 on
  sm_89 is TN-only, K and N multiples of 16, per-tensor scales in the GEMM). Verdict: **not before FP8 KV**; gain too
  small for the quality risk on L4, same ratio on L40S (362 TFLOPS / 864 GB/s ~ 420 FLOP/B).
- **KV (attention)**: the best 8-bit target. Bytes, not FLOPs, set the attention time, and this engine's fast kernel
  already writes the new K/V (`k_self_attention_bf16_batch_fast`, `backend_cuda.cu:9054`), so E4M3 conversion on
  write and dequant-in-register on read add no kernel. Use a static per-(layer, head) scale (calibrated once on the
  voice prefixes + a corpus) or per-64-position block scales stored next to the plane. Keep the shared voice
  prefix bf16 (it is L2-resident anyway); quantize only the per-row suffix. The reference engine measured 8-bit KV
  quality as fine (WER 0.26-0.46 % vs 0.51 % bf16, E4M3 best, E5M2 slightly metallic) and its attention microbench
  1.8-1.9x faster; its end-to-end loss came from separate quantize-on-write kernels, which this design does not have.
- **SEANet in FP8**: the conv GEMMs are the compute-heavier part (~5 ms), but the output is the waveform; bf16 is at
  47.5-49 dB SNR. Not recommended.

## 4. L40S readiness (same sm_89 binary; 142 SMs, 864 GB/s, 96 MB L2, 48 GB)

### 4.1 What is hard-coded or tuned for the L4 (CONFIRMED)

| item | where | L4 value | for L40S |
|---|---|---|---|
| engine batch ceiling | `POCKET_MAX_BATCH 384u` `src/engine_pocket.c:45` | 384 | compile-time; raise (e.g. 768) or size from `--max-batch` |
| driver ceilings | `MYNAH_GRAPH_MAX_JOBS 384u`, `MYNAH_GRAPH_MAX_ACTIVE 384u` `src/graph.h:20,27`; stack arrays in `src/inference.c` | 384 | same (stack arrays of 768 pointers are fine) |
| attention metadata arena | `CUDA_BATCH_META_CAP = 384u` `gpu/cuda/backend_cuda.cu:1415` | 384 | same |
| width buckets | `pocket_default_width_buckets[]` `src/engine_pocket.c:63` ends 192, 256, 384 | | env `MYNAH_CUDA_WIDTH_BUCKETS` already takes a list: add 448, 512, 576, 640, 704, 768 |
| pending queue | `JOB_QUEUE_CAP 256u` `server/main.c:103` | 256 | `--max-pending` exists |
| split-K of the fixed-order tile | `tile_gemm_splits(sms, ...)` `backend_cuda.cu:7686`, SM count read at `:8038` (fallback 40) | 58 SMs | adapts by itself; note the fixed-order result depends on S, so audio md5s differ between L4 and L40S (invariance holds per GPU type) |
| attention launch | grid heads x rows, 128 threads | | scales with rows; nothing to change |
| VRAM | ~75 MB/stream + 1.8 GB fixed (24L, before strip) | C208-C224 | ~600+ streams fit in 48 GB |
| slot-pool prefill + bucket walk at start-up | all `--max-batch` sets + one capture per bucket | ~33 s at 208 | grows with max-batch (minutes at 600): cap the walk or prefill lazily above C384 |
| host | 1 spinning scheduler thread, 1 writer thread per stream | 4 vCPU fine | 4-8 vCPU (g6e.xlarge/2xlarge) |

### 4.2 Expected new bottleneck (HYPOTHESIS)
- Bandwidth (2.9x) and bf16 compute (3.0x) scale alike, so the GPU-side step should shrink ~2.5-2.9x at equal rows,
  and the knee should move to ~C450-550 if the host keeps up.
- What does not scale: per-step host time (7 syncs, per-row memcpy/finite scans/pointer tables, `poll()` per row,
  PCM conversion: all O(rows)) and the launch latency of ~1,000+ small graph nodes. At C500 on a 4-vCPU host the
  scheduler thread likely becomes the limiter before the GPU: action 1 and action 4 are what make the L40S pay.
- Small grids (per-row LN with ~200 blocks, flow head, decoder glue) under-fill 142 SMs at low C; irrelevant at the
  knee.
- Power: 350 W budget, so clocks should stay near boost; the L4's power-cap penalty disappears.

### 4.3 Prepare before renting (all flags/config; L4 defaults unchanged)
1. A build with the four 384 ceilings raised to 768 (or a `MYNAH_MAX_BATCH` runtime knob), checked on the L4 at
   C208 for no regression and `--gpu-self-test` / `--pocket-self-check` PASS.
2. Knee scripts logging `nvidia-smi dmon -s pucm` and per-thread CPU (`pidstat -t`) next to the gates.
3. The best L4 flag set as one env file (shared voice + strip, BF16_FUSE, decoder fuse, prefill bf16tc, KV bf16).

### 4.4 Test plan for a 1-2 h L40S session (time-boxed, every job with a hard timeout)

| t (min) | step | what to read |
|---|---|---|
| 0-15 | provision (`tools/gpu/provision.sh sm_89`, or copy the L4 binary + packs), `nvidia-smi -q` (power limit, clocks), `nproc` | |
| 15-20 | `--gpu-self-test`, `--pocket-self-check` with the best flag set | PASS |
| 20-45 | knee, `--max-batch 384`, C256/C320/C384, 60 s per level, 4 voices | if C384 passes with margin, the ceiling is the limit |
| 45-70 | 768-ceiling build, buckets `...,384,448,512,576,640`, `--max-batch 640`: knee C448/C512/C576/C640 | the real knee; GPU util, SM clock, scheduler CPU % |
| 70-80 | top passing level pinned to 4 cores (`taskset -c 0-3`) vs 8 | does a g6e.xlarge suffice? |
| 80-95 | nsys 10 s at the top level (`--cuda-graph-trace=node`) | step split, GPU idle per step (the action-1 target) |
| 95-110 | optional: `MYNAH_CUDA_PREFILL_FIXED=0` and `MYNAH_CUDA_SYNC=blocking` arms at the top level | prefill and spin cost at high C |

Decision rule: the L40S is worth it per stream if its passing C divided by the L4's C208 exceeds the price ratio
of the two instances; record audio-s/s per dollar, not only C.
