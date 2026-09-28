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

Same as `.work/pocket-cuda-c60-l4.md` at C60, measured with the server
confined to 4 cores, plus: scheduler thread < 70% of one core at C60, zero
stalls at 250 ms over a 30-minute soak, and CPU `make test` green after every
shared-code change (the CPU path stays untouched).

## Board

| candidate | baseline | candidate | delta | decision |
|---|---:|---:|---:|---|
| (none yet) | | | | |
